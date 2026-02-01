"""
Trade Builder for SPY Options Agent
Constructs trade plans with legs, strikes, DTE, and risk parameters
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
from loguru import logger

from strategy.strategy_selector import StrategyType
from strategy.position_sizer import PositionSize
from analysis.regime_classifier import RegimeType, MarketRegime


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class LegAction(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeLeg:
    action: LegAction
    option_type: OptionType
    strike: float
    expiration: str  # ISO date
    quantity: int
    estimated_price: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiration": self.expiration,
            "quantity": self.quantity,
            "estimated_price": self.estimated_price,
        }


@dataclass
class TradePlan:
    strategy: StrategyType
    symbol: str
    legs: List[TradeLeg]
    direction: str  # "bullish", "bearish", "neutral"
    expiration: str
    dte: int
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakeven: Optional[List[float]] = None
    profit_target_pct: float = 50.0   # % of max profit
    stop_loss_pct: float = 100.0      # % of max loss (2x credit for spreads)
    spread_width: Optional[float] = None
    net_credit: Optional[float] = None
    net_debit: Optional[float] = None
    contracts: int = 1
    regime: Optional[str] = None
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "symbol": self.symbol,
            "legs": [leg.to_dict() for leg in self.legs],
            "direction": self.direction,
            "expiration": self.expiration,
            "dte": self.dte,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "breakeven": self.breakeven,
            "profit_target_pct": self.profit_target_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "spread_width": self.spread_width,
            "net_credit": self.net_credit,
            "net_debit": self.net_debit,
            "contracts": self.contracts,
            "regime": self.regime,
            "reasoning": self.reasoning,
        }


# DTE ranges by strategy type
DTE_RANGES = {
    StrategyType.BULL_PUT_SPREAD: (30, 45),
    StrategyType.BEAR_CALL_SPREAD: (30, 45),
    StrategyType.IRON_CONDOR: (30, 45),
    StrategyType.IRON_BUTTERFLY: (21, 35),
    StrategyType.BULL_CALL_DEBIT: (14, 30),
    StrategyType.BEAR_PUT_DEBIT: (14, 30),
    StrategyType.SHORT_PUT: (30, 45),
    StrategyType.SHORT_CALL: (30, 45),
    StrategyType.LONG_CALL: (14, 30),
    StrategyType.LONG_PUT: (14, 30),
    StrategyType.SHORT_STRANGLE: (30, 45),
    StrategyType.CALENDAR_SPREAD: (21, 45),
}

# Spread widths for SPY (rounded to $1 strikes)
SPREAD_WIDTHS = {
    "tight": 2.0,    # $2 wide
    "standard": 3.0,  # $3 wide
    "wide": 5.0,      # $5 wide
}


def _round_strike(price: float, direction: str = "nearest") -> float:
    """Round price to nearest $1 for SPY strikes."""
    if direction == "down":
        return float(int(price))
    elif direction == "up":
        return float(int(price) + 1)
    return round(price)


def _select_expiration(min_dte: int, max_dte: int) -> tuple:
    """Select target expiration date and DTE."""
    target_dte = (min_dte + max_dte) // 2
    exp_date = date.today() + timedelta(days=target_dte)
    # Snap to Friday
    days_to_friday = (4 - exp_date.weekday()) % 7
    exp_date = exp_date + timedelta(days=days_to_friday)
    actual_dte = (exp_date - date.today()).days
    return exp_date.isoformat(), actual_dte


class TradeBuilder:
    """
    Builds trade plans with legs, strikes, and risk parameters.
    Uses live options chain if available, otherwise theoretical strikes.
    """

    def build(
        self,
        strategy: StrategyType,
        price: float,
        regime: MarketRegime,
        iv_data: Optional[Dict] = None,
        position_size: Optional[PositionSize] = None,
    ) -> Optional[TradePlan]:
        """
        Build a complete trade plan.

        Args:
            strategy: StrategyType to construct
            price: Current underlying price
            regime: MarketRegime for context
            iv_data: IV analytics dict
            position_size: PositionSize from sizer

        Returns:
            TradePlan or None if strategy is NO_TRADE
        """
        if strategy == StrategyType.NO_TRADE:
            return None

        contracts = position_size.contracts if position_size else 1

        # Select expiration
        dte_range = DTE_RANGES.get(strategy, (30, 45))
        expiration, dte = _select_expiration(*dte_range)

        # Determine spread width based on IV
        width = self._select_spread_width(iv_data)

        # Build legs based on strategy type
        builders = {
            StrategyType.BULL_PUT_SPREAD: self._build_bull_put_spread,
            StrategyType.BEAR_CALL_SPREAD: self._build_bear_call_spread,
            StrategyType.IRON_CONDOR: self._build_iron_condor,
            StrategyType.IRON_BUTTERFLY: self._build_iron_butterfly,
            StrategyType.BULL_CALL_DEBIT: self._build_bull_call_debit,
            StrategyType.BEAR_PUT_DEBIT: self._build_bear_put_debit,
            StrategyType.SHORT_PUT: self._build_short_put,
            StrategyType.SHORT_CALL: self._build_short_call,
            StrategyType.LONG_CALL: self._build_long_call,
            StrategyType.LONG_PUT: self._build_long_put,
            StrategyType.SHORT_STRANGLE: self._build_short_strangle,
            StrategyType.CALENDAR_SPREAD: self._build_calendar_spread,
        }

        builder_fn = builders.get(strategy)
        if not builder_fn:
            logger.warning(f"No builder for {strategy.value}")
            return None

        try:
            plan = builder_fn(price, width, expiration, dte, contracts)
            plan.regime = regime.regime.value
            plan.reasoning = f"Regime: {regime.regime.value} (confidence {regime.confidence:.0f}%)"
            return plan
        except Exception as e:
            logger.error(f"Error building {strategy.value}: {e}")
            return None

    def _select_spread_width(self, iv_data: Optional[Dict]) -> float:
        """Select spread width based on IV conditions."""
        if iv_data is None:
            return SPREAD_WIDTHS["standard"]

        iv_rank = iv_data.get("iv_rank", 50)
        if iv_rank > 60:
            return SPREAD_WIDTHS["wide"]  # wider in high IV for more premium
        elif iv_rank < 30:
            return SPREAD_WIDTHS["tight"]  # tighter in low IV
        return SPREAD_WIDTHS["standard"]

    # --- Credit Spreads ---

    def _build_bull_put_spread(self, price, width, exp, dte, qty):
        short_strike = _round_strike(price * 0.97, "down")  # ~3% OTM
        long_strike = short_strike - width
        return TradePlan(
            strategy=StrategyType.BULL_PUT_SPREAD,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.SELL, OptionType.PUT, short_strike, exp, qty),
                TradeLeg(LegAction.BUY, OptionType.PUT, long_strike, exp, qty),
            ],
            direction="bullish",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=50, stop_loss_pct=100,
            contracts=qty,
        )

    def _build_bear_call_spread(self, price, width, exp, dte, qty):
        short_strike = _round_strike(price * 1.03, "up")  # ~3% OTM
        long_strike = short_strike + width
        return TradePlan(
            strategy=StrategyType.BEAR_CALL_SPREAD,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.SELL, OptionType.CALL, short_strike, exp, qty),
                TradeLeg(LegAction.BUY, OptionType.CALL, long_strike, exp, qty),
            ],
            direction="bearish",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=50, stop_loss_pct=100,
            contracts=qty,
        )

    def _build_iron_condor(self, price, width, exp, dte, qty):
        put_short = _round_strike(price * 0.97, "down")
        put_long = put_short - width
        call_short = _round_strike(price * 1.03, "up")
        call_long = call_short + width
        return TradePlan(
            strategy=StrategyType.IRON_CONDOR,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.BUY, OptionType.PUT, put_long, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.PUT, put_short, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.CALL, call_short, exp, qty),
                TradeLeg(LegAction.BUY, OptionType.CALL, call_long, exp, qty),
            ],
            direction="neutral",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=50, stop_loss_pct=100,
            contracts=qty,
        )

    def _build_iron_butterfly(self, price, width, exp, dte, qty):
        atm = _round_strike(price)
        return TradePlan(
            strategy=StrategyType.IRON_BUTTERFLY,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.BUY, OptionType.PUT, atm - width, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.PUT, atm, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.CALL, atm, exp, qty),
                TradeLeg(LegAction.BUY, OptionType.CALL, atm + width, exp, qty),
            ],
            direction="neutral",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=25, stop_loss_pct=100,
            contracts=qty,
        )

    # --- Debit Spreads ---

    def _build_bull_call_debit(self, price, width, exp, dte, qty):
        long_strike = _round_strike(price, "nearest")  # ATM
        short_strike = long_strike + width
        return TradePlan(
            strategy=StrategyType.BULL_CALL_DEBIT,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.BUY, OptionType.CALL, long_strike, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.CALL, short_strike, exp, qty),
            ],
            direction="bullish",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=75, stop_loss_pct=50,
            contracts=qty,
        )

    def _build_bear_put_debit(self, price, width, exp, dte, qty):
        long_strike = _round_strike(price, "nearest")  # ATM
        short_strike = long_strike - width
        return TradePlan(
            strategy=StrategyType.BEAR_PUT_DEBIT,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.BUY, OptionType.PUT, long_strike, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.PUT, short_strike, exp, qty),
            ],
            direction="bearish",
            expiration=exp, dte=dte,
            spread_width=width,
            max_loss=width * 100 * qty,
            profit_target_pct=75, stop_loss_pct=50,
            contracts=qty,
        )

    # --- Naked / Single Leg ---

    def _build_short_put(self, price, width, exp, dte, qty):
        strike = _round_strike(price * 0.95, "down")  # 5% OTM
        return TradePlan(
            strategy=StrategyType.SHORT_PUT,
            symbol="SPY",
            legs=[TradeLeg(LegAction.SELL, OptionType.PUT, strike, exp, qty)],
            direction="bullish",
            expiration=exp, dte=dte,
            max_loss=strike * 100 * qty,
            profit_target_pct=50, stop_loss_pct=200,
            contracts=qty,
        )

    def _build_short_call(self, price, width, exp, dte, qty):
        strike = _round_strike(price * 1.05, "up")  # 5% OTM
        return TradePlan(
            strategy=StrategyType.SHORT_CALL,
            symbol="SPY",
            legs=[TradeLeg(LegAction.SELL, OptionType.CALL, strike, exp, qty)],
            direction="bearish",
            expiration=exp, dte=dte,
            profit_target_pct=50, stop_loss_pct=200,
            contracts=qty,
        )

    def _build_long_call(self, price, width, exp, dte, qty):
        strike = _round_strike(price, "nearest")
        return TradePlan(
            strategy=StrategyType.LONG_CALL,
            symbol="SPY",
            legs=[TradeLeg(LegAction.BUY, OptionType.CALL, strike, exp, qty)],
            direction="bullish",
            expiration=exp, dte=dte,
            profit_target_pct=100, stop_loss_pct=50,
            contracts=qty,
        )

    def _build_long_put(self, price, width, exp, dte, qty):
        strike = _round_strike(price, "nearest")
        return TradePlan(
            strategy=StrategyType.LONG_PUT,
            symbol="SPY",
            legs=[TradeLeg(LegAction.BUY, OptionType.PUT, strike, exp, qty)],
            direction="bearish",
            expiration=exp, dte=dte,
            profit_target_pct=100, stop_loss_pct=50,
            contracts=qty,
        )

    def _build_short_strangle(self, price, width, exp, dte, qty):
        put_strike = _round_strike(price * 0.95, "down")
        call_strike = _round_strike(price * 1.05, "up")
        return TradePlan(
            strategy=StrategyType.SHORT_STRANGLE,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.SELL, OptionType.PUT, put_strike, exp, qty),
                TradeLeg(LegAction.SELL, OptionType.CALL, call_strike, exp, qty),
            ],
            direction="neutral",
            expiration=exp, dte=dte,
            profit_target_pct=50, stop_loss_pct=200,
            contracts=qty,
        )

    def _build_calendar_spread(self, price, width, exp, dte, qty):
        strike = _round_strike(price, "nearest")
        near_exp = exp
        # Far expiration ~30 days after near
        far_date = date.fromisoformat(exp) + timedelta(days=30)
        days_to_friday = (4 - far_date.weekday()) % 7
        far_date = far_date + timedelta(days=days_to_friday)
        far_exp = far_date.isoformat()

        return TradePlan(
            strategy=StrategyType.CALENDAR_SPREAD,
            symbol="SPY",
            legs=[
                TradeLeg(LegAction.SELL, OptionType.CALL, strike, near_exp, qty),
                TradeLeg(LegAction.BUY, OptionType.CALL, strike, far_exp, qty),
            ],
            direction="neutral",
            expiration=far_exp, dte=dte + 30,
            profit_target_pct=50, stop_loss_pct=75,
            contracts=qty,
        )
