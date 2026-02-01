"""
Opportunity Scoring Engine for SPY
Combines signals into actionable opportunities with confidence scores
Based on backtested strategy performance
"""
from dataclasses import dataclass, field
from datetime import datetime
import pytz
import uuid
from typing import Dict, List, Optional
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
    B = "B"        # 50-64: Moderate conviction
    C = "C"        # 35-49: Low conviction
    F = "F"        # <35: No trade


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

    def to_dict(self) -> dict:
        return {
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

    def format_alert(self) -> str:
        """Format as alert message"""
        direction_emoji = "🟢" if self.direction == Direction.LONG else "🔴"
        grade_emoji = {"A+": "🔥", "A": "⭐", "B": "👀", "C": "⚠️", "F": "🚫"}
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

    def get_grade(self, score: int) -> OpportunityGrade:
        """Convert score to grade

        OPTIMIZED based on backtest (2024-12):
        - Historical A+ win rate: 70.8% (24 trades)
        - LONG: 75% win rate, SHORT: 50% win rate
        - Keep original thresholds for more signals
        """
        if score >= 80:
            return OpportunityGrade.A_PLUS
        elif score >= 65:
            return OpportunityGrade.A
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
