"""
Risk Manager for SPY Options Agent
6-level decision hierarchy for trade approval/rejection
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
from loguru import logger

from analysis.regime_classifier import MarketRegime, RegimeType
from analysis.liquidity_checker import LiquidityChecker, LiquidityScore
from risk.event_risk import EventRiskChecker, EventRisk
from risk.greeks_monitor import GreeksMonitor, GreeksReport
from strategy.trade_builder import TradePlan
from strategy.strategy_selector import StrategyType


@dataclass
class RiskDecision:
    approved: bool
    trade_plan: Optional[TradePlan]
    reasons: List[str]
    adjustments: List[str]
    level_results: Dict[str, str]  # level_name -> "PASS"/"FAIL"/"WARN"
    risk_score: float  # 0-100, higher = more risky

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "trade_plan": self.trade_plan.to_dict() if self.trade_plan else None,
            "reasons": self.reasons,
            "adjustments": self.adjustments,
            "level_results": self.level_results,
            "risk_score": self.risk_score,
        }


class RiskManager:
    """
    6-level decision hierarchy for trade evaluation.

    Non-overridable (hard stops):
      1. Portfolio risk limits (daily 3%, weekly 5%, max 3 correlated positions)
      2. Liquidity check
      3. Event risk (HIGH level)
      4. Regime alignment

    Overridable (adjust, don't block):
      5. IV conditions (adjust strategy/size)
      6. Signal quality (reduce size)
    """

    def __init__(self, account_value: float = 100000):
        self.account_value = account_value
        self.liquidity_checker = LiquidityChecker()
        self.event_checker = EventRiskChecker()
        self.greeks_monitor = GreeksMonitor()

        # Portfolio tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.max_daily_loss_pct = 0.03   # 3%
        self.max_weekly_loss_pct = 0.05  # 5%
        self.max_correlated_positions = 3

    def evaluate_trade(
        self,
        trade_plan: TradePlan,
        regime: MarketRegime,
        iv_data: Optional[Dict] = None,
        signals: Optional[List] = None,
        positions: Optional[List[Dict]] = None,
        account_value: Optional[float] = None,
    ) -> RiskDecision:
        """
        Evaluate a trade plan through the 6-level hierarchy.

        Args:
            trade_plan: TradePlan to evaluate
            regime: Current MarketRegime
            iv_data: IV analytics dict
            signals: List of Signal objects
            positions: Current open positions
            account_value: Override account value

        Returns:
            RiskDecision with approval status and reasons
        """
        acct = account_value or self.account_value
        positions = positions or []
        reasons = []
        adjustments = []
        level_results = {}
        risk_score = 0

        # === Level 1: Portfolio Risk Limits (NON-OVERRIDABLE) ===
        l1_pass, l1_reasons = self._check_portfolio_limits(positions, acct)
        level_results["1_portfolio_limits"] = "PASS" if l1_pass else "FAIL"
        if not l1_pass:
            reasons.extend(l1_reasons)
            risk_score += 40
            return RiskDecision(
                approved=False, trade_plan=trade_plan,
                reasons=reasons, adjustments=adjustments,
                level_results=level_results, risk_score=min(risk_score, 100),
            )

        # === Level 2: Liquidity Check (NON-OVERRIDABLE) ===
        liquidity = self.liquidity_checker.check_option()
        level_results["2_liquidity"] = "PASS" if liquidity.passed else "FAIL"
        if not liquidity.passed:
            reasons.append(f"Liquidity check failed: {liquidity.details}")
            risk_score += 30
            return RiskDecision(
                approved=False, trade_plan=trade_plan,
                reasons=reasons, adjustments=adjustments,
                level_results=level_results, risk_score=min(risk_score, 100),
            )
        if liquidity.warnings:
            adjustments.extend(liquidity.warnings)

        # === Level 3: Event Risk (NON-OVERRIDABLE for HIGH) ===
        event_risk = self.event_checker.check()
        if event_risk.blocked:
            level_results["3_event_risk"] = "FAIL"
            reasons.append(f"Event risk blocked: {', '.join(event_risk.events)}")
            reasons.append(f"Recommendation: {event_risk.recommendation}")
            risk_score += 35
            return RiskDecision(
                approved=False, trade_plan=trade_plan,
                reasons=reasons, adjustments=adjustments,
                level_results=level_results, risk_score=min(risk_score, 100),
            )
        elif event_risk.risk_level == "MEDIUM":
            level_results["3_event_risk"] = "WARN"
            adjustments.append(f"Event warning: {', '.join(event_risk.events)}")
            risk_score += 10
        else:
            level_results["3_event_risk"] = "PASS"

        # === Level 4: Regime Alignment (NON-OVERRIDABLE) ===
        regime_ok, regime_reasons = self._check_regime_alignment(trade_plan, regime)
        level_results["4_regime_alignment"] = "PASS" if regime_ok else "FAIL"
        if not regime_ok:
            reasons.extend(regime_reasons)
            risk_score += 25
            return RiskDecision(
                approved=False, trade_plan=trade_plan,
                reasons=reasons, adjustments=adjustments,
                level_results=level_results, risk_score=min(risk_score, 100),
            )

        # === Level 5: IV Conditions (OVERRIDABLE — adjust strategy) ===
        iv_ok, iv_notes = self._check_iv_conditions(trade_plan, iv_data)
        level_results["5_iv_conditions"] = "PASS" if iv_ok else "WARN"
        if not iv_ok:
            adjustments.extend(iv_notes)
            risk_score += 10

        # === Level 6: Signal Quality (OVERRIDABLE — reduce size) ===
        sig_ok, sig_notes = self._check_signal_quality(signals, regime)
        level_results["6_signal_quality"] = "PASS" if sig_ok else "WARN"
        if not sig_ok:
            adjustments.extend(sig_notes)
            risk_score += 10

        # Greeks check (advisory)
        if positions:
            greeks_report = self.greeks_monitor.check_portfolio(positions, acct)
            if not greeks_report.within_limits:
                adjustments.extend(greeks_report.warnings)
                adjustments.extend(greeks_report.required_actions)
                risk_score += 15

        approved = True
        if not reasons:
            reasons.append("All risk checks passed")

        return RiskDecision(
            approved=approved,
            trade_plan=trade_plan,
            reasons=reasons,
            adjustments=adjustments,
            level_results=level_results,
            risk_score=min(risk_score, 100),
        )

    def _check_portfolio_limits(
        self, positions: List[Dict], account_value: float
    ) -> tuple:
        """Level 1: Portfolio risk limits."""
        reasons = []

        # Daily loss check
        max_daily = account_value * self.max_daily_loss_pct
        if self.daily_pnl < -max_daily:
            reasons.append(
                f"Daily loss ${abs(self.daily_pnl):,.0f} exceeds "
                f"{self.max_daily_loss_pct:.0%} limit (${max_daily:,.0f})"
            )

        # Weekly loss check
        max_weekly = account_value * self.max_weekly_loss_pct
        if self.weekly_pnl < -max_weekly:
            reasons.append(
                f"Weekly loss ${abs(self.weekly_pnl):,.0f} exceeds "
                f"{self.max_weekly_loss_pct:.0%} limit (${max_weekly:,.0f})"
            )

        # Correlated positions
        spy_positions = [p for p in positions if p.get("symbol") == "SPY"]
        if len(spy_positions) >= self.max_correlated_positions:
            reasons.append(
                f"Max {self.max_correlated_positions} correlated SPY positions reached "
                f"(currently {len(spy_positions)})"
            )

        return (len(reasons) == 0, reasons)

    def _check_regime_alignment(
        self, trade_plan: TradePlan, regime: MarketRegime
    ) -> tuple:
        """Level 4: Check if strategy aligns with regime."""
        reasons = []

        if regime.regime == RegimeType.TRANSITION:
            reasons.append(
                f"Regime is TRANSITION (confidence {regime.confidence:.0f}%); "
                "no trades until regime clarifies"
            )
            return (False, reasons)

        # Check if strategy is in allowed list
        if trade_plan.strategy.value not in regime.allowed_strategies:
            reasons.append(
                f"Strategy {trade_plan.strategy.value} not allowed in "
                f"{regime.regime.value} regime. Allowed: {regime.allowed_strategies}"
            )
            return (False, reasons)

        # Check direction alignment
        if regime.bias == "bullish" and trade_plan.direction == "bearish":
            reasons.append(
                f"Bearish trade conflicts with bullish regime ({regime.regime.value})"
            )
            return (False, reasons)
        elif regime.bias == "bearish" and trade_plan.direction == "bullish":
            reasons.append(
                f"Bullish trade conflicts with bearish regime ({regime.regime.value})"
            )
            return (False, reasons)

        return (True, [])

    def _check_iv_conditions(
        self, trade_plan: TradePlan, iv_data: Optional[Dict]
    ) -> tuple:
        """Level 5: IV conditions (overridable)."""
        if iv_data is None:
            return (True, ["No IV data available; proceeding with caution"])

        notes = []
        iv_rank = iv_data.get("iv_rank", 50)

        # Selling premium in low IV
        credit_strategies = {
            StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD,
            StrategyType.IRON_CONDOR, StrategyType.IRON_BUTTERFLY,
            StrategyType.SHORT_PUT, StrategyType.SHORT_CALL,
            StrategyType.SHORT_STRANGLE,
        }
        if trade_plan.strategy in credit_strategies and iv_rank < 25:
            notes.append(
                f"Selling premium with IV rank {iv_rank:.0f} (low); "
                "consider reducing size or switching to debit strategy"
            )
            return (False, notes)

        # Buying premium in high IV
        debit_strategies = {
            StrategyType.BULL_CALL_DEBIT, StrategyType.BEAR_PUT_DEBIT,
            StrategyType.LONG_CALL, StrategyType.LONG_PUT,
        }
        if trade_plan.strategy in debit_strategies and iv_rank > 75:
            notes.append(
                f"Buying premium with IV rank {iv_rank:.0f} (high); "
                "consider switching to credit strategy"
            )
            return (False, notes)

        return (True, [])

    def _check_signal_quality(
        self, signals: Optional[List], regime: MarketRegime
    ) -> tuple:
        """Level 6: Signal quality (overridable)."""
        if not signals:
            return (False, ["No signals detected; consider reducing position size"])

        notes = []
        tier1_count = sum(1 for s in signals if s.tier == 1)
        total_strength = sum(s.strength for s in signals)
        avg_strength = total_strength / len(signals) if signals else 0

        if tier1_count == 0:
            notes.append("No Tier 1 signals; reduce position size by 50%")
            return (False, notes)

        if avg_strength < 40:
            notes.append(f"Low average signal strength ({avg_strength:.0f}); reduce size")
            return (False, notes)

        return (True, [])

    def update_pnl(self, daily_pnl: float, weekly_pnl: float):
        """Update running P&L for portfolio limit checks."""
        self.daily_pnl = daily_pnl
        self.weekly_pnl = weekly_pnl

    def reset_daily(self):
        """Reset daily P&L (call at start of each trading day)."""
        self.daily_pnl = 0.0

    def reset_weekly(self):
        """Reset weekly P&L (call at start of each week)."""
        self.weekly_pnl = 0.0
