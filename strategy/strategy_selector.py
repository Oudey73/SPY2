"""
Strategy Selector for SPY Options Agent
Maps (regime, IV level, bias) to optimal options strategy
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from loguru import logger

from analysis.regime_classifier import RegimeType, MarketRegime


class StrategyType(Enum):
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    BULL_CALL_DEBIT = "bull_call_debit"
    BEAR_PUT_DEBIT = "bear_put_debit"
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_STRANGLE = "short_strangle"
    CALENDAR_SPREAD = "calendar_spread"
    NO_TRADE = "no_trade"


class IVLevel(Enum):
    HIGH = "high"       # IV rank > 60
    MEDIUM = "medium"   # IV rank 30-60
    LOW = "low"         # IV rank < 30


@dataclass
class StrategyRecommendation:
    strategy: StrategyType
    reasoning: str
    regime: RegimeType
    iv_level: IVLevel
    bias: str
    confidence: float
    alternatives: List[StrategyType]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "reasoning": self.reasoning,
            "regime": self.regime.value,
            "iv_level": self.iv_level.value,
            "bias": self.bias,
            "confidence": self.confidence,
            "alternatives": [a.value for a in self.alternatives],
        }


# Strategy matrix: (regime, iv_level, bias) -> (strategy, reasoning)
STRATEGY_MATRIX: Dict[Tuple, Tuple[StrategyType, str]] = {
    # TRENDING_UP
    (RegimeType.TRENDING_UP, IVLevel.HIGH, "bullish"): (
        StrategyType.BULL_PUT_SPREAD,
        "Sell premium in uptrend with elevated IV; high probability credit spread",
    ),
    (RegimeType.TRENDING_UP, IVLevel.MEDIUM, "bullish"): (
        StrategyType.BULL_PUT_SPREAD,
        "Credit spread in uptrend with moderate IV",
    ),
    (RegimeType.TRENDING_UP, IVLevel.LOW, "bullish"): (
        StrategyType.BULL_CALL_DEBIT,
        "Buy calls cheap in low IV uptrend for directional exposure",
    ),

    # TRENDING_DOWN
    (RegimeType.TRENDING_DOWN, IVLevel.HIGH, "bearish"): (
        StrategyType.BEAR_CALL_SPREAD,
        "Sell premium in downtrend with elevated IV; high probability credit spread",
    ),
    (RegimeType.TRENDING_DOWN, IVLevel.MEDIUM, "bearish"): (
        StrategyType.BEAR_CALL_SPREAD,
        "Credit spread in downtrend with moderate IV",
    ),
    (RegimeType.TRENDING_DOWN, IVLevel.LOW, "bearish"): (
        StrategyType.BEAR_PUT_DEBIT,
        "Buy puts cheap in low IV downtrend for directional exposure",
    ),

    # RANGE_BOUND
    (RegimeType.RANGE_BOUND, IVLevel.HIGH, "neutral"): (
        StrategyType.IRON_CONDOR,
        "Sell premium on both sides in range with high IV; max theta decay",
    ),
    (RegimeType.RANGE_BOUND, IVLevel.MEDIUM, "neutral"): (
        StrategyType.IRON_CONDOR,
        "Sell premium in range with moderate IV; standard iron condor",
    ),
    (RegimeType.RANGE_BOUND, IVLevel.LOW, "neutral"): (
        StrategyType.CALENDAR_SPREAD,
        "Low IV in range; calendar spread benefits from IV expansion",
    ),
    (RegimeType.RANGE_BOUND, IVLevel.HIGH, "bullish"): (
        StrategyType.BULL_PUT_SPREAD,
        "Bullish bias in range with high IV; sell puts below range support",
    ),
    (RegimeType.RANGE_BOUND, IVLevel.HIGH, "bearish"): (
        StrategyType.BEAR_CALL_SPREAD,
        "Bearish bias in range with high IV; sell calls above range resistance",
    ),

    # HIGH_VOLATILITY
    (RegimeType.HIGH_VOLATILITY, IVLevel.HIGH, "neutral"): (
        StrategyType.IRON_CONDOR,
        "Wide iron condor in high vol; collect elevated premium with wide wings",
    ),
    (RegimeType.HIGH_VOLATILITY, IVLevel.HIGH, "bullish"): (
        StrategyType.BULL_PUT_SPREAD,
        "Bullish in high vol; sell put spread far OTM for high probability",
    ),
    (RegimeType.HIGH_VOLATILITY, IVLevel.HIGH, "bearish"): (
        StrategyType.BEAR_CALL_SPREAD,
        "Bearish in high vol; sell call spread far OTM for high probability",
    ),

    # TRANSITION - generally avoid
    (RegimeType.TRANSITION, IVLevel.HIGH, "neutral"): (
        StrategyType.NO_TRADE,
        "Regime unclear with high vol; wait for clarity",
    ),
    (RegimeType.TRANSITION, IVLevel.MEDIUM, "neutral"): (
        StrategyType.NO_TRADE,
        "Regime unclear; wait for directional confirmation",
    ),
    (RegimeType.TRANSITION, IVLevel.LOW, "neutral"): (
        StrategyType.NO_TRADE,
        "Regime unclear with low IV; no edge available",
    ),
}


class StrategySelector:
    """
    Selects optimal options strategy based on regime, IV, and directional bias.
    Uses STRATEGY_MATRIX for lookup with fuzzy matching for unlisted combos.
    """

    def select(
        self,
        regime: MarketRegime,
        iv_data: Optional[Dict] = None,
        direction_bias: Optional[str] = None,
    ) -> StrategyRecommendation:
        """
        Select strategy based on current conditions.

        Args:
            regime: MarketRegime from RegimeClassifier
            iv_data: Dict with iv_rank, current_iv, etc.
            direction_bias: Override bias ("bullish", "bearish", "neutral")

        Returns:
            StrategyRecommendation with strategy and reasoning
        """
        # Determine IV level
        iv_level = self._classify_iv(iv_data)

        # Determine bias
        bias = direction_bias or regime.bias

        # Look up in matrix
        key = (regime.regime, iv_level, bias)
        if key in STRATEGY_MATRIX:
            strategy, reasoning = STRATEGY_MATRIX[key]
        else:
            # Fuzzy match: try neutral bias
            fallback_key = (regime.regime, iv_level, "neutral")
            if fallback_key in STRATEGY_MATRIX:
                strategy, reasoning = STRATEGY_MATRIX[fallback_key]
                reasoning += f" (bias '{bias}' not in matrix; using neutral)"
            else:
                # Try HIGH IV as fallback
                hv_key = (regime.regime, IVLevel.HIGH, "neutral")
                if hv_key in STRATEGY_MATRIX:
                    strategy, reasoning = STRATEGY_MATRIX[hv_key]
                    reasoning += " (fallback match)"
                else:
                    strategy = StrategyType.NO_TRADE
                    reasoning = f"No strategy for {regime.regime.value}/{iv_level.value}/{bias}"

        # Build alternatives
        alternatives = self._get_alternatives(regime.regime, iv_level, bias, strategy)

        # Adjust confidence
        confidence = regime.confidence
        if strategy == StrategyType.NO_TRADE:
            confidence = 0

        return StrategyRecommendation(
            strategy=strategy,
            reasoning=reasoning,
            regime=regime.regime,
            iv_level=iv_level,
            bias=bias,
            confidence=confidence,
            alternatives=alternatives,
        )

    def _classify_iv(self, iv_data: Optional[Dict]) -> IVLevel:
        """Classify IV into HIGH/MEDIUM/LOW from iv_rank."""
        if iv_data is None:
            return IVLevel.MEDIUM

        iv_rank = iv_data.get("iv_rank")
        if iv_rank is None:
            return IVLevel.MEDIUM

        if iv_rank > 60:
            return IVLevel.HIGH
        elif iv_rank >= 30:
            return IVLevel.MEDIUM
        else:
            return IVLevel.LOW

    def _get_alternatives(
        self,
        regime: RegimeType,
        iv_level: IVLevel,
        bias: str,
        primary: StrategyType,
    ) -> List[StrategyType]:
        """Get alternative strategies for the given conditions."""
        alternatives = []
        for key, (strat, _) in STRATEGY_MATRIX.items():
            if key[0] == regime and strat != primary and strat != StrategyType.NO_TRADE:
                if strat not in alternatives:
                    alternatives.append(strat)
        return alternatives[:3]
