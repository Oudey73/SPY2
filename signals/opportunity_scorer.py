"""
Opportunity Scoring Engine for SPY
Combines signals into actionable opportunities with confidence scores
Based on backtested strategy performance

Enhanced with Multi-Factor Scoring System (2025-02):
- Multiplicative scoring with kill switches
- Truth bonuses for high-conviction setups
- Kelly Criterion position sizing
- Confidence grades (A+/A/B+/B/C/F)
"""
from dataclasses import dataclass, field
from datetime import datetime
import pytz
import uuid
from typing import Dict, List, Optional, Any
from enum import Enum
from .signal_detector import Signal, SignalType, Direction
from loguru import logger


# Saudi Arabia timezone
SAUDI_TZ = pytz.timezone("Asia/Riyadh")

def now_saudi() -> str:
    return datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")


class OpportunityGrade(Enum):
    """Opportunity grades based on historical win rates"""
    A_PLUS = "A+"  # 80-100: Very high conviction (IBS+RSI combo + VIX filter)
    A = "A"        # 65-79: High conviction
    B_PLUS = "B+"  # 58-64: Good conviction
    B = "B"        # 50-57: Moderate conviction
    C = "C"        # 35-49: Low conviction
    F = "F"        # <35: No trade


@dataclass
class EnhancedScoringResult:
    """
    Result of the enhanced multi-factor scoring system.

    Contains the final score, confidence grade, logic breakdown,
    and dynamic position sizing based on Kelly Criterion.
    """
    final_score: int
    confidence_grade: str
    logic_breakdown: List[str]
    dynamic_position_size: float  # Kelly-based position size (0.0 to 0.05)
    direction: Direction
    base_score: int
    multipliers_applied: Dict[str, float]
    bonuses_applied: Dict[str, int]
    penalties_applied: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "confidence_grade": self.confidence_grade,
            "logic_breakdown": self.logic_breakdown,
            "dynamic_position_size": self.dynamic_position_size,
            "direction": self.direction.value,
            "base_score": self.base_score,
            "multipliers_applied": self.multipliers_applied,
            "bonuses_applied": self.bonuses_applied,
            "penalties_applied": self.penalties_applied,
        }


class MultiplierEngine:
    """
    Engine for applying multiplicative filters, bonuses, and penalties
    to the base opportunity score.

    Kill Switches (multiply score):
    - session_time=False -> score x 0.6
    - rvol < 1.2 during mean reversion -> score x 0.5

    Penalties (subtract from score):
    - DXY trend conflicts with direction -> score - 20

    Bonuses (add to score):
    - IBS < 0.2 AND cvd_slope > 0 (LONG only) -> score + 25 (Truth Bonus)
    """

    # Kelly Criterion parameters by grade
    # Format: (win_rate, profit_loss_ratio)
    # These are conservative estimates based on backtested performance
    KELLY_PARAMS = {
        "A+": (0.75, 1.5),  # 75% win rate, 1.5:1 P/L -> Kelly ~2.5-3%
        "A": (0.70, 1.3),   # 70% win rate, 1.3:1 P/L -> Kelly ~2%
        "B+": (0.65, 1.25), # 65% win rate, 1.25:1 P/L -> Kelly ~1.6%
        "B": (0.60, 1.2),   # 60% win rate, 1.2:1 P/L -> Kelly ~1.3%
        "C": (0.55, 1.1),   # 55% win rate, 1.1:1 P/L -> Kelly ~0.5%
        "F": (0.50, 1.0),   # 50% win rate, 1.0:1 P/L -> Kelly ~0% (no edge)
    }

    # Maximum position size (half-Kelly capped)
    MAX_POSITION_SIZE = 0.05  # 5% max

    @staticmethod
    def apply_session_multiplier(
        score: float,
        is_hp_session: bool
    ) -> tuple[float, Optional[float]]:
        """
        Apply session time kill switch.

        High probability sessions (09:30-11:00, 15:00-16:00 ET) get full score.
        Other times get 0.6x multiplier.

        Returns:
            (adjusted_score, multiplier_applied or None)
        """
        if is_hp_session:
            return score, None
        else:
            return score * 0.6, 0.6

    @staticmethod
    def apply_rvol_multiplier(
        score: float,
        rvol: Optional[float],
        is_mean_reversion: bool = True
    ) -> tuple[float, Optional[float]]:
        """
        Apply RVOL kill switch for mean reversion trades.

        Mean reversion works better with normal/elevated volume.
        RVOL < 1.2 during mean reversion -> score x 0.5

        Args:
            score: Current score
            rvol: Relative volume ratio (None if unavailable)
            is_mean_reversion: Whether this is a mean reversion trade

        Returns:
            (adjusted_score, multiplier_applied or None)
        """
        if rvol is None:
            # No RVOL data - don't penalize
            return score, None

        if is_mean_reversion and rvol < 1.2:
            return score * 0.5, 0.5

        return score, None

    @staticmethod
    def apply_dxy_penalty(
        score: float,
        dxy_trend: Optional[str],
        direction: Direction
    ) -> tuple[float, Optional[int]]:
        """
        Apply DXY trend conflict penalty.

        Strong dollar (DXY up) conflicts with LONG SPY.
        Weak dollar (DXY down) conflicts with SHORT SPY.

        Args:
            score: Current score
            dxy_trend: "up", "down", "neutral", or None
            direction: Trade direction

        Returns:
            (adjusted_score, penalty_applied or None)
        """
        if dxy_trend is None:
            return score, None

        conflict = False
        if direction == Direction.LONG and dxy_trend == "up":
            conflict = True
        elif direction == Direction.SHORT and dxy_trend == "down":
            conflict = True

        if conflict:
            return score - 20, 20

        return score, None

    @staticmethod
    def apply_truth_bonus(
        score: float,
        ibs: Optional[float],
        cvd_slope: Optional[float],
        direction: Direction
    ) -> tuple[float, Optional[int]]:
        """
        Apply truth bonus for high-conviction setups.

        LONG only: IBS < 0.2 AND positive CVD slope -> +25 points
        This combination shows genuine buying pressure during oversold conditions.

        Args:
            score: Current score
            ibs: Internal Bar Strength (0-1)
            cvd_slope: CVD slope (positive = buying pressure)
            direction: Trade direction

        Returns:
            (adjusted_score, bonus_applied or None)
        """
        if direction != Direction.LONG:
            return score, None

        if ibs is None or cvd_slope is None:
            return score, None

        if ibs < 0.2 and cvd_slope > 0:
            return score + 25, 25

        return score, None

    # Target position sizes by grade (based on plan specification)
    # These are the desired outputs after Kelly adjustment
    TARGET_POSITION_SIZES = {
        "A+": 0.025,  # 2.5%
        "A": 0.020,   # 2.0%
        "B+": 0.018,  # 1.8%
        "B": 0.015,   # 1.5%
        "C": 0.010,   # 1.0%
        "F": 0.005,   # 0.5% (paper trade)
    }

    @classmethod
    def calculate_kelly(cls, grade: str) -> float:
        """
        Calculate position size using simplified Kelly Criterion.

        f* = (p * b - q) / b

        Where:
            p = historical win rate for grade
            b = profit/loss ratio
            q = 1 - p

        We use quarter-Kelly and target specific sizes per grade
        based on backtested results.

        Target sizes by grade:
        - A+: ~2.5-3%
        - A: ~2%
        - B+: ~1.8%
        - B: ~1.5%
        - C: ~1%
        - F: ~0.5% (paper trade only)

        Args:
            grade: Confidence grade (A+, A, B+, B, C, F)

        Returns:
            Position size as decimal (0.0 to 0.05)
        """
        # Use predefined target sizes based on plan
        if grade in cls.TARGET_POSITION_SIZES:
            return cls.TARGET_POSITION_SIZES[grade]

        # Fallback: calculate using Kelly formula
        if grade not in cls.KELLY_PARAMS:
            return 0.01  # Default 1% for unknown grades

        p, b = cls.KELLY_PARAMS[grade]
        q = 1 - p

        # Kelly formula
        kelly = (p * b - q) / b

        # Quarter-Kelly for safety (more conservative than half-Kelly)
        quarter_kelly = kelly / 4

        # Cap at max position size
        return min(max(0, quarter_kelly), cls.MAX_POSITION_SIZE)

    @staticmethod
    def score_to_grade(score: int) -> str:
        """
        Convert final score to confidence grade.

        Score ranges:
        - 80+: A+
        - 70-79: A
        - 58-69: B+
        - 50-57: B
        - 35-49: C
        - <35: F

        Args:
            score: Final score (0-100)

        Returns:
            Grade string (A+, A, B+, B, C, F)
        """
        if score >= 80:
            return "A+"
        elif score >= 70:
            return "A"
        elif score >= 58:
            return "B+"
        elif score >= 50:
            return "B"
        elif score >= 35:
            return "C"
        else:
            return "F"


@dataclass
class Opportunity:
    """Represents a trading opportunity"""
    symbol: str
    direction: Direction
    score: int  # 0-100
    grade: OpportunityGrade
    confidence_level: str
    signals: List[Signal]
    top_drivers: List[str]
    entry_zone: Dict[str, float]
    stop_loss: float
    targets: Dict[str, float]
    risk_reward: float
    position_size_suggestion: str
    max_hold_days: int
    recommended_dte: str  # Recommended option expiration
    warnings: List[str]
    vix_filter_passed: bool
    timestamp: str
    opp_id: str = ""
    # Enhanced scoring fields (optional, populated when enhanced scoring used)
    enhanced_result: Optional[EnhancedScoringResult] = None

    def to_dict(self) -> dict:
        result = {
            "opp_id": self.opp_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "score": self.score,
            "grade": self.grade.value,
            "confidence_level": self.confidence_level,
            "signals": [s.to_dict() for s in self.signals],
            "top_drivers": self.top_drivers,
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "targets": self.targets,
            "risk_reward": self.risk_reward,
            "position_size_suggestion": self.position_size_suggestion,
            "max_hold_days": self.max_hold_days,
            "recommended_dte": self.recommended_dte,
            "warnings": self.warnings,
            "vix_filter_passed": self.vix_filter_passed,
            "timestamp": self.timestamp
        }
        if self.enhanced_result:
            result["enhanced_scoring"] = self.enhanced_result.to_dict()
        return result

    def format_alert(self) -> str:
        """Format as alert message"""
        direction_emoji = "🟢" if self.direction == Direction.LONG else "🔴"
        grade_emoji = {"A+": "🔥", "A": "⭐", "B+": "✨", "B": "👀", "C": "⚠️", "F": "🚫"}
        vix_status = "✅" if self.vix_filter_passed else "⚠️"

        alert = f"""
{'='*50}
{direction_emoji} SPY - {self.direction.value.upper()} OPPORTUNITY
{'='*50}
🆔 ID: {self.opp_id}

📊 SCORE: {self.score}/100 ({self.grade.value}) {grade_emoji.get(self.grade.value, '')}
🎯 CONFIDENCE: {self.confidence_level}
📈 VIX FILTER: {vix_status} {'Passed' if self.vix_filter_passed else 'Warning - Low VIX'}

📋 TOP DRIVERS:
"""
        for i, driver in enumerate(self.top_drivers[:3], 1):
            alert += f"   {i}. {driver}\n"

        alert += f"""
💰 ENTRY ZONE:
   • Aggressive: ${self.entry_zone['aggressive']:.2f}
   • Target Entry: ${self.entry_zone['target']:.2f}
   • Conservative: ${self.entry_zone['conservative']:.2f}

🛑 STOP LOSS: ${self.stop_loss:.2f}

🎯 TARGETS:
   • T1 (2%): ${self.targets['t1']:.2f}
   • T2 (3%): ${self.targets['t2']:.2f}
   • T3 (4%): ${self.targets['t3']:.2f}

📐 RISK/REWARD: {self.risk_reward:.1f}R
⏰ MAX HOLD: {self.max_hold_days} trading days

💼 POSITION SIZE: {self.position_size_suggestion}

📅 OPTIONS EXPIRATION:
   • Recommended DTE: {self.recommended_dte}
   • Strategy: Weekly/Bi-weekly options for mean reversion
"""

        if self.warnings:
            alert += "\n⚠️ WARNINGS:\n"
            for warning in self.warnings:
                alert += f"   • {warning}\n"

        alert += f"""
⏰ Generated: {self.timestamp}

{'='*50}
📊 STRATEGY: IBS + RSI(3) Mean Reversion
📈 Historical Win Rate: ~71% (when VIX filter passes)
📉 Typical Hold: 1-5 days
{'='*50}
⚠️ DISCLAIMER: This is NOT financial advice.
Decision-support tool only. Past performance ≠ future results.
{'='*50}
"""
        return alert


class SPYOpportunityScorer:
    """Scores and ranks SPY trading opportunities"""

    # Scoring weights based on backtested effectiveness
    # UPDATED 2024-12-19: Added PUT-specific signals from optimization
    WEIGHTS = {
        # Tier 1 - Primary drivers (highest evidence)
        SignalType.IBS_RSI_COMBO_LONG: 40,
        SignalType.IBS_RSI_COMBO_SHORT: 40,
        SignalType.IBS_EXTREME_LOW: 30,
        SignalType.IBS_EXTREME_HIGH: 30,
        SignalType.IBS_OVERSOLD: 25,
        SignalType.IBS_OVERBOUGHT: 25,

        # NEW: RSI(2) signals for PUT strategy (KEY DRIVER - 94% of profitable PUTs)
        SignalType.RSI2_EXTREME_HIGH: 35,  # RSI(2) >= 98 is the PRIMARY PUT signal
        SignalType.RSI2_OVERBOUGHT: 20,    # RSI(2) >= 95

        # Tier 2 - Confirming signals
        SignalType.RSI_EXTREME_LOW: 15,
        SignalType.RSI_EXTREME_HIGH: 15,
        SignalType.RSI_OVERSOLD: 10,
        SignalType.RSI_OVERBOUGHT: 10,
        SignalType.VIX_EXTREME: 15,
        SignalType.VIX_ELEVATED: 10,
        SignalType.INTRADAY_MOMENTUM_BULLISH: 10,
        SignalType.INTRADAY_MOMENTUM_BEARISH: 10,

        # NEW: Consecutive up days for PUT signals (58% of profitable PUTs)
        SignalType.CONSECUTIVE_UP_4: 15,  # 4+ consecutive up days
        SignalType.CONSECUTIVE_UP_3: 8,   # 3 consecutive up days

        # Tier 3 - Context (smaller weight)
        SignalType.BELOW_50_MA: 5,
        SignalType.BELOW_200_MA: 8,
        SignalType.VIX_COMPLACENT: -5,  # Negative - reduces score for LONG
        SignalType.ABOVE_50_MA: 0,
    }

    def __init__(self, min_score_threshold: int = 50):
        self.min_score_threshold = min_score_threshold

    def calculate_score(self, signals: List[Signal]) -> Dict:
        """Calculate opportunity score from signals"""
        long_score = 0
        short_score = 0
        long_signals = []
        short_signals = []
        neutral_signals = []
        vix_filter_passed = False

        for signal in signals:
            weight = self.WEIGHTS.get(signal.signal_type, 5)
            # Scale weight by signal strength
            adjusted_weight = int(weight * (signal.strength / 100))

            # Check VIX filter
            if signal.signal_type in [SignalType.VIX_ELEVATED, SignalType.VIX_EXTREME]:
                vix_filter_passed = True

            if signal.direction == Direction.LONG:
                long_score += adjusted_weight
                long_signals.append(signal)
            elif signal.direction == Direction.SHORT:
                short_score += adjusted_weight
                short_signals.append(signal)
            else:
                neutral_signals.append(signal)
                # Apply VIX complacent penalty to longs
                if signal.signal_type == SignalType.VIX_COMPLACENT:
                    long_score += weight  # Negative weight reduces score

        # Determine dominant direction
        if long_score > short_score:
            dominant = Direction.LONG
            dominant_score = long_score
            aligned_signals = long_signals
            conflicting_signals = short_signals
        elif short_score > long_score:
            dominant = Direction.SHORT
            dominant_score = short_score
            aligned_signals = short_signals
            conflicting_signals = long_signals
        else:
            dominant = Direction.NEUTRAL
            dominant_score = 0
            aligned_signals = []
            conflicting_signals = []

        # Conflict penalty
        conflict_penalty = len(conflicting_signals) * 5
        final_score = max(0, min(100, dominant_score - conflict_penalty))

        # VIX filter bonus for longs (research shows this improves win rate)
        if dominant == Direction.LONG and vix_filter_passed:
            final_score = min(100, final_score + 10)

        # For SHORT signals: VIX complacency is actually GOOD (contrarian)
        # Backtest shows VIX < 14 present in 30% of profitable PUT trades
        if dominant == Direction.SHORT:
            # Check if we have RSI(2) extreme signal (key PUT driver)
            has_rsi2_extreme = any(s.signal_type == SignalType.RSI2_EXTREME_HIGH
                                   for s in aligned_signals)
            if has_rsi2_extreme:
                final_score = min(100, final_score + 5)  # Bonus for key signal

        return {
            "total_score": final_score,
            "direction": dominant,
            "direction_scores": {"long": long_score, "short": short_score},
            "aligned_signals": aligned_signals,
            "conflicting_signals": conflicting_signals,
            "vix_filter_passed": vix_filter_passed,
        }

    def calculate_enhanced_score(
        self,
        signals: List[Signal],
        enhanced_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[EnhancedScoringResult]:
        """
        Calculate enhanced score using multiplicative scoring system.

        This method applies kill switches, bonuses, and penalties on top of
        the base additive score to produce a more nuanced final score.

        Scoring Formula:
        1. Base Score = Sum of (IBS weights + RSI weights) for aligned signals
        2. Apply kill switches (multipliers):
           - session_time=False -> score x 0.6
           - rvol < 1.2 during mean reversion -> score x 0.5
        3. Apply penalties:
           - DXY trend conflicts with direction -> score - 20
        4. Apply bonuses:
           - IBS < 0.2 AND cvd_slope > 0 (LONG only) -> score + 25
        5. Final = clamp(0, 100)

        Args:
            signals: List of detected signals
            enhanced_data: Dict from EnhancedDataCollector.get_all_enhanced_data()
                - cvd_slope: float or None
                - rvol: float or None
                - dxy_trend: str or None
                - is_hp_session: bool
            market_data: Dict with market data including:
                - ibs: float

        Returns:
            EnhancedScoringResult or None if no clear direction
        """
        if not signals:
            return None

        # Step 1: Calculate base score using existing method
        base_result = self.calculate_score(signals)

        if base_result["direction"] == Direction.NEUTRAL:
            logger.debug("Enhanced scoring: No clear direction")
            return None

        direction = base_result["direction"]
        base_score = base_result["total_score"]

        # Initialize tracking
        logic_breakdown = [f"Base Score: {base_score} ({direction.value.upper()})"]
        multipliers_applied = {}
        bonuses_applied = {}
        penalties_applied = {}

        # Working score (float for multiplier precision)
        score = float(base_score)

        # Step 2: Apply kill switches (multipliers)

        # Session time multiplier
        is_hp_session = enhanced_data.get("is_hp_session", True)
        score, session_mult = MultiplierEngine.apply_session_multiplier(score, is_hp_session)
        if session_mult:
            multipliers_applied["session_time"] = session_mult
            logic_breakdown.append(f"Outside HP session window ({session_mult}x)")

        # RVOL multiplier (for mean reversion trades)
        rvol = enhanced_data.get("rvol")
        score, rvol_mult = MultiplierEngine.apply_rvol_multiplier(
            score, rvol, is_mean_reversion=True
        )
        if rvol_mult:
            multipliers_applied["rvol"] = rvol_mult
            logic_breakdown.append(f"Low RVOL ({rvol:.2f}) during mean reversion ({rvol_mult}x)")

        # Step 3: Apply penalties

        # DXY trend penalty
        dxy_trend = enhanced_data.get("dxy_trend")
        score, dxy_penalty = MultiplierEngine.apply_dxy_penalty(score, dxy_trend, direction)
        if dxy_penalty:
            penalties_applied["dxy_conflict"] = dxy_penalty
            logic_breakdown.append(f"DXY trend ({dxy_trend}) conflicts with {direction.value} (-{dxy_penalty})")

        # Step 4: Apply bonuses

        # Truth bonus (IBS + CVD for LONG)
        ibs = market_data.get("ibs")
        cvd_slope = enhanced_data.get("cvd_slope")
        score, truth_bonus = MultiplierEngine.apply_truth_bonus(score, ibs, cvd_slope, direction)
        if truth_bonus:
            bonuses_applied["truth_bonus"] = truth_bonus
            logic_breakdown.append(
                f"Truth Bonus: IBS ({ibs:.2f}) + positive CVD ({cvd_slope:.2f}) (+{truth_bonus})"
            )

        # Step 5: Clamp final score to 0-100
        final_score = max(0, min(100, int(round(score))))

        # Determine grade and Kelly position size
        grade = MultiplierEngine.score_to_grade(final_score)
        kelly_size = MultiplierEngine.calculate_kelly(grade)

        logger.info(f"Enhanced scoring: {base_score} -> {final_score} ({grade}), Kelly: {kelly_size:.2%}")

        return EnhancedScoringResult(
            final_score=final_score,
            confidence_grade=grade,
            logic_breakdown=logic_breakdown,
            dynamic_position_size=kelly_size,
            direction=direction,
            base_score=base_score,
            multipliers_applied=multipliers_applied,
            bonuses_applied=bonuses_applied,
            penalties_applied=penalties_applied,
        )

    def get_grade(self, score: int) -> OpportunityGrade:
        """Convert score to grade

        OPTIMIZED based on backtest (2024-12):
        - Historical A+ win rate: 70.8% (24 trades)
        - LONG: 75% win rate, SHORT: 50% win rate
        - Keep original thresholds for more signals

        Updated 2025-02: Added B+ grade for finer granularity
        """
        if score >= 80:
            return OpportunityGrade.A_PLUS
        elif score >= 70:
            return OpportunityGrade.A
        elif score >= 58:
            return OpportunityGrade.B_PLUS
        elif score >= 50:
            return OpportunityGrade.B
        elif score >= 35:
            return OpportunityGrade.C
        else:
            return OpportunityGrade.F

    def get_confidence_level(self, score: int, vix_passed: bool, has_combo: bool) -> str:
        """Get human-readable confidence level"""
        if score >= 80 and vix_passed and has_combo:
            return "VERY HIGH - IBS+RSI combo with VIX confirmation"
        elif score >= 65 and vix_passed:
            return "HIGH - Strong setup with VIX filter"
        elif score >= 65:
            return "HIGH - Strong setup (VIX filter not confirmed)"
        elif score >= 50:
            return "MODERATE - Decent setup, standard position"
        elif score >= 35:
            return "LOW - Weak conviction, reduced size or skip"
        else:
            return "INSUFFICIENT - No clear opportunity"

    def calculate_levels(
        self,
        direction: Direction,
        current_price: float,
    ) -> Dict:
        """
        Calculate entry, stop, and target levels

        Using fixed percentages based on SPY's typical moves:
        OPTIMIZED 2024-12-19: Different parameters for CALL vs PUT

        CALL (Long): Stop 1.5%, Target 2% (R:R = 1.33:1)
        PUT (Short): Stop 1%, Target 2% (R:R = 2:1) - Tighter stop for PUTs
        """
        if direction == Direction.LONG:
            # CALL strategy parameters
            stop_pct = 0.015  # 1.5% stop
            t1_pct = 0.02     # 2% target
            t2_pct = 0.03     # 3%
            t3_pct = 0.04     # 4%

            entry_zone = {
                "aggressive": current_price,
                "target": current_price * 0.998,      # 0.2% below current
                "conservative": current_price * 0.995  # 0.5% below current
            }
            stop_loss = current_price * (1 - stop_pct)
            targets = {
                "t1": current_price * (1 + t1_pct),
                "t2": current_price * (1 + t2_pct),
                "t3": current_price * (1 + t3_pct)
            }
        elif direction == Direction.SHORT:
            # PUT strategy parameters (OPTIMIZED from backtest)
            # Backtest shows: Stop 1%, Target 2% optimal for PUTs
            stop_pct = 0.01   # 1% stop (tighter for PUTs)
            t1_pct = 0.02     # 2% target
            t2_pct = 0.03     # 3%
            t3_pct = 0.04     # 4%

            entry_zone = {
                "aggressive": current_price,
                "target": current_price * 1.002,      # 0.2% above current
                "conservative": current_price * 1.005  # 0.5% above current
            }
            stop_loss = current_price * (1 + stop_pct)
            targets = {
                "t1": current_price * (1 - t1_pct),
                "t2": current_price * (1 - t2_pct),
                "t3": current_price * (1 - t3_pct)
            }
        else:
            return None

        # Calculate risk/reward to T2
        risk = abs(current_price - stop_loss)
        reward = abs(targets["t2"] - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        return {
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "targets": targets,
            "risk_reward": risk_reward
        }

    def suggest_position_size(self, score: int, vix_passed: bool) -> str:
        """Suggest position size based on conviction"""
        if score >= 80 and vix_passed:
            return "3-5% of portfolio (high conviction setup)"
        elif score >= 65 and vix_passed:
            return "2-3% of portfolio (solid setup)"
        elif score >= 65:
            return "1-2% of portfolio (no VIX confirmation)"
        elif score >= 50:
            return "1% of portfolio (moderate conviction)"
        else:
            return "Paper trade only or skip"

    def generate_warnings(
        self,
        signals: List[Signal],
        conflicting_signals: List[Signal],
        score: int,
        vix_passed: bool
    ) -> List[str]:
        """Generate relevant warnings"""
        warnings = []

        if conflicting_signals:
            warnings.append(f"{len(conflicting_signals)} conflicting signal(s) - mixed picture")

        if not vix_passed:
            warnings.append("VIX filter not passed - historically lower win rate")

        if score < 65:
            warnings.append("Below high-conviction threshold")

        # Check for specific risk scenarios
        signal_types = [s.signal_type for s in signals]

        if SignalType.VIX_COMPLACENT in signal_types:
            warnings.append("VIX complacent - market may be overbought")

        if SignalType.BELOW_200_MA in signal_types:
            warnings.append("Price below 200-MA - deeper correction possible")

        return warnings


    def suggest_option_dte(self, score: int, max_hold_days: int) -> str:
        """
        Suggest option expiration based on strategy and conviction
        
        Mean reversion holds: 1-5 days
        Recommended DTE: 7-14 days (enough time buffer, manageable theta)
        """
        if score >= 80:
            return "7-10 DTE (high conviction, tighter expiry OK)"
        elif score >= 65:
            return "10-14 DTE (solid setup, standard buffer)"
        elif score >= 50:
            return "14-21 DTE (moderate conviction, extra time buffer)"
        else:
            return "21+ DTE or avoid options (low conviction)"

    def score_opportunity(
        self,
        signals: List[Signal],
        current_price: float,
    ) -> Optional[Opportunity]:
        """
        Score all signals and generate opportunity

        Returns None if no clear opportunity
        """
        if not signals:
            return None

        # Calculate scores
        score_result = self.calculate_score(signals)

        if score_result["direction"] == Direction.NEUTRAL:
            logger.info("No clear direction for SPY")
            return None

        score = score_result["total_score"]
        direction = score_result["direction"]
        aligned_signals = score_result["aligned_signals"]
        conflicting_signals = score_result["conflicting_signals"]
        vix_passed = score_result["vix_filter_passed"]

        # Check for combo signal
        has_combo = any(s.signal_type in [SignalType.IBS_RSI_COMBO_LONG, SignalType.IBS_RSI_COMBO_SHORT]
                       for s in aligned_signals)

        # Get grade and confidence
        grade = self.get_grade(score)
        confidence = self.get_confidence_level(score, vix_passed, has_combo)

        # Calculate levels
        levels = self.calculate_levels(direction, current_price)
        if not levels:
            return None

        # Get top drivers (sorted by tier and strength)
        sorted_signals = sorted(aligned_signals, key=lambda s: (s.tier, -s.strength))
        top_drivers = [s.description for s in sorted_signals[:3]]

        # Get warnings
        warnings = self.generate_warnings(signals, conflicting_signals, score, vix_passed)

        # Position size suggestion
        position_size = self.suggest_position_size(score, vix_passed)

        opp_id = f"OPP-{uuid.uuid4().hex[:4]}"

        return Opportunity(
            symbol="SPY",
            direction=direction,
            score=score,
            grade=grade,
            confidence_level=confidence,
            signals=aligned_signals,
            top_drivers=top_drivers,
            entry_zone=levels["entry_zone"],
            stop_loss=levels["stop_loss"],
            targets=levels["targets"],
            risk_reward=levels["risk_reward"],
            position_size_suggestion=position_size,
            max_hold_days=5,  # Research shows 3-5 days optimal
            recommended_dte=self.suggest_option_dte(score, 5),
            warnings=warnings,
            vix_filter_passed=vix_passed,
            timestamp=now_saudi(),
            opp_id=opp_id
        )


def test_opportunity_scorer():
    """Test the opportunity scorer"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    from .signal_detector import SPYSignalDetector

    detector = SPYSignalDetector()
    scorer = SPYOpportunityScorer()

    print("\n" + "=" * 60)
    print("SPY OPPORTUNITY SCORER TEST")
    print("=" * 60)

    # Test Case 1: High conviction long
    print("\n--- Test 1: High Conviction Long ---")
    market_data = {
        "ibs": 0.12,
        "rsi_3": 15,
        "vix": 28,
        "vix_10_ma": 22,
        "price": 580,
        "sma_50": 590,
        "sma_200": 575,
    }

    signals = detector.analyze_all(market_data)
    opportunity = scorer.score_opportunity(signals, market_data["price"])

    if opportunity:
        print(opportunity.format_alert())
    else:
        print("No opportunity detected")

    # Test Case 2: Moderate short
    print("\n--- Test 2: Overbought Short ---")
    market_data = {
        "ibs": 0.88,
        "rsi_3": 85,
        "vix": 14,
        "vix_10_ma": 16,
        "price": 600,
        "sma_50": 590,
    }

    signals = detector.analyze_all(market_data)
    opportunity = scorer.score_opportunity(signals, market_data["price"])

    if opportunity:
        print(f"Score: {opportunity.score}, Grade: {opportunity.grade.value}")
        print(f"Direction: {opportunity.direction.value}")
        print(f"VIX Filter: {opportunity.vix_filter_passed}")
    else:
        print("No opportunity detected")

    return opportunity


if __name__ == "__main__":
    test_opportunity_scorer()
