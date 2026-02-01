"""
Signal Detection Module for SPY
Based on backtested strategies with proven edge:
- IBS (Internal Bar Strength) mean reversion
- RSI(3) oversold/overbought
- VIX regime filter
- Intraday momentum
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from loguru import logger
import pytz

# Saudi Arabia timezone (AST / UTC+3)
SAUDI_TZ = pytz.timezone('Asia/Riyadh')

def now_saudi() -> str:
    return datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")


class SignalType(Enum):
    """Types of signals we detect"""
    # Tier 1 - Primary (highest backtested edge)
    IBS_OVERSOLD = "ibs_oversold"              # IBS < 0.2
    IBS_OVERBOUGHT = "ibs_overbought"          # IBS > 0.8
    IBS_EXTREME_LOW = "ibs_extreme_low"        # IBS < 0.15
    IBS_EXTREME_HIGH = "ibs_extreme_high"      # IBS > 0.85
    IBS_RSI_COMBO_LONG = "ibs_rsi_combo_long"  # IBS < 0.2 AND RSI(3) < 20
    IBS_RSI_COMBO_SHORT = "ibs_rsi_combo_short"  # IBS > 0.8 AND RSI(3) > 80

    # Tier 2 - Confirming
    RSI_OVERSOLD = "rsi_oversold"              # RSI(3) < 20
    RSI_OVERBOUGHT = "rsi_overbought"          # RSI(3) > 80
    RSI_EXTREME_LOW = "rsi_extreme_low"        # RSI(3) < 10
    RSI_EXTREME_HIGH = "rsi_extreme_high"      # RSI(3) > 90

    # NEW: RSI(2) signals for PUT strategy (optimized 2024-12-19)
    # Backtest: RSI(2) >= 98 is the KEY driver for PUT signals (94% of winning trades)
    RSI2_EXTREME_HIGH = "rsi2_extreme_high"    # RSI(2) >= 98 - KEY PUT SIGNAL
    RSI2_OVERBOUGHT = "rsi2_overbought"        # RSI(2) >= 95

    # VIX Filter
    VIX_ELEVATED = "vix_elevated"              # VIX > 10-day MA
    VIX_EXTREME = "vix_extreme"                # VIX > 30
    VIX_COMPLACENT = "vix_complacent"          # VIX < 15

    # Trend Context
    BELOW_50_MA = "below_50_ma"                # Price below 50-day MA
    ABOVE_50_MA = "above_50_ma"                # Price above 50-day MA
    BELOW_200_MA = "below_200_ma"              # Price below 200-day MA

    # NEW: Consecutive up days for PUT signals (optimized 2024-12-19)
    CONSECUTIVE_UP_3 = "consecutive_up_3"      # 3 consecutive up days
    CONSECUTIVE_UP_4 = "consecutive_up_4"      # 4+ consecutive up days - strong PUT signal

    # Intraday
    INTRADAY_MOMENTUM_BULLISH = "intraday_momentum_bullish"
    INTRADAY_MOMENTUM_BEARISH = "intraday_momentum_bearish"


class Direction(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    """Represents a detected signal"""
    signal_type: SignalType
    direction: Direction
    strength: int          # 1-100
    value: float           # The actual value that triggered
    threshold: float       # The threshold it crossed
    description: str
    timestamp: str
    tier: int              # 1, 2, or 3 (importance)

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "strength": self.strength,
            "value": self.value,
            "threshold": self.threshold,
            "description": self.description,
            "timestamp": self.timestamp,
            "tier": self.tier
        }


class SPYSignalDetector:
    """
    Detects trading signals for SPY based on proven strategies

    Primary Strategy: IBS + RSI(3) Mean Reversion
    - Historical win rate: 71%+
    - Best when combined with VIX filter
    """

    # Thresholds based on backtested research (OPTIMIZED 2024-12-19)
    # Backtest results (5 years, 2020-2024):
    #   IBS<0.2, RSI<30, A+ grade: 70% win rate, +10% total P&L, 10 trades/year
    #   This is the OPTIMAL configuration balancing win rate and frequency
    THRESHOLDS = {
        # IBS (Internal Bar Strength)
        # Backtest: IBS < 0.2 with RSI < 30 gives 70% win rate
        "ibs_oversold": 0.2,
        "ibs_overbought": 0.8,
        "ibs_extreme_low": 0.15,
        "ibs_extreme_high": 0.85,

        # RSI(3) - OPTIMIZED from 20 to 30
        # Backtest: RSI < 30 captures more opportunities while maintaining edge
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "rsi_extreme_low": 10,
        "rsi_extreme_high": 90,

        # RSI(2) - for PUT signals (optimized 2024-12-19)
        # Backtest: RSI(2) >= 98 present in 94% of profitable PUT signals
        "rsi2_extreme_high": 98,
        "rsi2_overbought": 95,

        # VIX
        # Backtest: VIX 20-25 (elevated) has highest win rate
        "vix_elevated": 18,
        "vix_optimal_low": 18,
        "vix_optimal_high": 25,
        "vix_extreme": 30,
        "vix_complacent": 15,
    }

    def __init__(self, thresholds: dict = None):
        if thresholds:
            self.THRESHOLDS.update(thresholds)

    def detect_ibs_signals(self, ibs: float, prev_ibs: float = None) -> List[Signal]:
        """
        Detect IBS (Internal Bar Strength) signals

        IBS = (Close - Low) / (High - Low)

        Research shows:
        - IBS < 0.2 → next day avg return +0.35%
        - IBS > 0.8 → next day avg return -0.13%
        """
        signals = []
        timestamp = now_saudi()

        if ibs is None:
            return signals

        # Extreme low IBS - strong buy signal
        if ibs <= self.THRESHOLDS["ibs_extreme_low"]:
            strength = min(100, int((1 - ibs / self.THRESHOLDS["ibs_extreme_low"]) * 80 + 50))
            signals.append(Signal(
                signal_type=SignalType.IBS_EXTREME_LOW,
                direction=Direction.LONG,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_extreme_low"],
                description=f"IBS at {ibs:.3f} - Extreme oversold, high probability bounce",
                timestamp=timestamp,
                tier=1
            ))

        # Standard oversold
        elif ibs <= self.THRESHOLDS["ibs_oversold"]:
            strength = min(100, int((1 - ibs / self.THRESHOLDS["ibs_oversold"]) * 60 + 30))
            signals.append(Signal(
                signal_type=SignalType.IBS_OVERSOLD,
                direction=Direction.LONG,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_oversold"],
                description=f"IBS at {ibs:.3f} - Oversold, mean reversion expected",
                timestamp=timestamp,
                tier=1
            ))

        # Extreme high IBS - short signal
        elif ibs >= self.THRESHOLDS["ibs_extreme_high"]:
            strength = min(100, int((ibs - self.THRESHOLDS["ibs_extreme_high"]) / 0.15 * 50 + 40))
            signals.append(Signal(
                signal_type=SignalType.IBS_EXTREME_HIGH,
                direction=Direction.SHORT,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_extreme_high"],
                description=f"IBS at {ibs:.3f} - Extreme overbought, pullback likely",
                timestamp=timestamp,
                tier=1
            ))

        # Standard overbought
        elif ibs >= self.THRESHOLDS["ibs_overbought"]:
            strength = min(100, int((ibs - self.THRESHOLDS["ibs_overbought"]) / 0.2 * 40 + 25))
            signals.append(Signal(
                signal_type=SignalType.IBS_OVERBOUGHT,
                direction=Direction.SHORT,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_overbought"],
                description=f"IBS at {ibs:.3f} - Overbought, weakness expected",
                timestamp=timestamp,
                tier=1
            ))

        return signals

    def detect_rsi_signals(self, rsi: float) -> List[Signal]:
        """
        Detect RSI(3) signals

        RSI(3) is more responsive than RSI(14) for mean reversion
        """
        signals = []
        timestamp = now_saudi()

        if rsi is None:
            return signals

        # Extreme oversold
        if rsi <= self.THRESHOLDS["rsi_extreme_low"]:
            strength = min(100, int((1 - rsi / self.THRESHOLDS["rsi_extreme_low"]) * 70 + 50))
            signals.append(Signal(
                signal_type=SignalType.RSI_EXTREME_LOW,
                direction=Direction.LONG,
                strength=strength,
                value=rsi,
                threshold=self.THRESHOLDS["rsi_extreme_low"],
                description=f"RSI(3) at {rsi:.1f} - Extreme oversold",
                timestamp=timestamp,
                tier=2
            ))

        # Standard oversold
        elif rsi <= self.THRESHOLDS["rsi_oversold"]:
            strength = min(100, int((1 - rsi / self.THRESHOLDS["rsi_oversold"]) * 50 + 25))
            signals.append(Signal(
                signal_type=SignalType.RSI_OVERSOLD,
                direction=Direction.LONG,
                strength=strength,
                value=rsi,
                threshold=self.THRESHOLDS["rsi_oversold"],
                description=f"RSI(3) at {rsi:.1f} - Oversold",
                timestamp=timestamp,
                tier=2
            ))

        # Extreme overbought
        elif rsi >= self.THRESHOLDS["rsi_extreme_high"]:
            strength = min(100, int((rsi - self.THRESHOLDS["rsi_extreme_high"]) / 10 * 50 + 40))
            signals.append(Signal(
                signal_type=SignalType.RSI_EXTREME_HIGH,
                direction=Direction.SHORT,
                strength=strength,
                value=rsi,
                threshold=self.THRESHOLDS["rsi_extreme_high"],
                description=f"RSI(3) at {rsi:.1f} - Extreme overbought",
                timestamp=timestamp,
                tier=2
            ))

        # Standard overbought
        elif rsi >= self.THRESHOLDS["rsi_overbought"]:
            strength = min(100, int((rsi - self.THRESHOLDS["rsi_overbought"]) / 20 * 40 + 20))
            signals.append(Signal(
                signal_type=SignalType.RSI_OVERBOUGHT,
                direction=Direction.SHORT,
                strength=strength,
                value=rsi,
                threshold=self.THRESHOLDS["rsi_overbought"],
                description=f"RSI(3) at {rsi:.1f} - Overbought",
                timestamp=timestamp,
                tier=2
            ))

        return signals

    def detect_rsi2_signals(self, rsi2: float) -> List[Signal]:
        """
        Detect RSI(2) signals - KEY for PUT strategy

        RSI(2) is even more responsive than RSI(3)
        Backtest shows RSI(2) >= 98 present in 94% of profitable PUT trades
        """
        signals = []
        timestamp = now_saudi()

        if rsi2 is None:
            return signals

        # Extreme overbought RSI(2) - PRIMARY PUT signal
        if rsi2 >= self.THRESHOLDS["rsi2_extreme_high"]:
            strength = min(100, int((rsi2 - self.THRESHOLDS["rsi2_extreme_high"]) / 2 * 50 + 70))
            signals.append(Signal(
                signal_type=SignalType.RSI2_EXTREME_HIGH,
                direction=Direction.SHORT,
                strength=strength,
                value=rsi2,
                threshold=self.THRESHOLDS["rsi2_extreme_high"],
                description=f"RSI(2) at {rsi2:.1f} - EXTREME overbought, high probability PUT",
                timestamp=timestamp,
                tier=1
            ))

        # Standard overbought RSI(2)
        elif rsi2 >= self.THRESHOLDS["rsi2_overbought"]:
            strength = min(100, int((rsi2 - self.THRESHOLDS["rsi2_overbought"]) / 3 * 40 + 40))
            signals.append(Signal(
                signal_type=SignalType.RSI2_OVERBOUGHT,
                direction=Direction.SHORT,
                strength=strength,
                value=rsi2,
                threshold=self.THRESHOLDS["rsi2_overbought"],
                description=f"RSI(2) at {rsi2:.1f} - Overbought, PUT opportunity",
                timestamp=timestamp,
                tier=2
            ))

        return signals

    def detect_consecutive_days(self, consecutive_up: int, consecutive_down: int = 0) -> List[Signal]:
        """
        Detect consecutive up/down day signals

        Backtest shows 4+ consecutive up days present in 58% of profitable PUT trades
        3+ consecutive down days present in significant CALL setups
        """
        signals = []
        timestamp = now_saudi()

        if consecutive_up is None:
            consecutive_up = 0

        # 4+ consecutive up days - Strong PUT signal
        if consecutive_up >= 4:
            strength = min(100, int(consecutive_up * 15 + 30))
            signals.append(Signal(
                signal_type=SignalType.CONSECUTIVE_UP_4,
                direction=Direction.SHORT,
                strength=strength,
                value=float(consecutive_up),
                threshold=4.0,
                description=f"{consecutive_up} consecutive up days - Extended rally, PUT opportunity",
                timestamp=timestamp,
                tier=2
            ))

        # 3 consecutive up days
        elif consecutive_up >= 3:
            strength = min(100, int(consecutive_up * 12 + 20))
            signals.append(Signal(
                signal_type=SignalType.CONSECUTIVE_UP_3,
                direction=Direction.SHORT,
                strength=strength,
                value=float(consecutive_up),
                threshold=3.0,
                description=f"{consecutive_up} consecutive up days - Rally extended",
                timestamp=timestamp,
                tier=3
            ))

        return signals

    def detect_ibs_rsi_combo(self, ibs: float, rsi: float) -> List[Signal]:
        """
        Detect the powerful IBS + RSI combination signal

        Research shows this combo improves returns by 9.6 percentage points
        """
        signals = []
        timestamp = now_saudi()

        if ibs is None or rsi is None:
            return signals

        # Long combo: IBS < 0.2 AND RSI(3) < 20
        if (ibs <= self.THRESHOLDS["ibs_oversold"] and
            rsi <= self.THRESHOLDS["rsi_oversold"]):

            # Higher strength for more extreme values
            ibs_score = max(0, (self.THRESHOLDS["ibs_oversold"] - ibs) / self.THRESHOLDS["ibs_oversold"])
            rsi_score = max(0, (self.THRESHOLDS["rsi_oversold"] - rsi) / self.THRESHOLDS["rsi_oversold"])
            strength = min(100, int((ibs_score + rsi_score) * 50 + 50))

            signals.append(Signal(
                signal_type=SignalType.IBS_RSI_COMBO_LONG,
                direction=Direction.LONG,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_oversold"],
                description=f"IBS ({ibs:.3f}) + RSI ({rsi:.1f}) COMBO - High conviction long",
                timestamp=timestamp,
                tier=1
            ))

        # Short combo: IBS > 0.8 AND RSI(3) > 80
        elif (ibs >= self.THRESHOLDS["ibs_overbought"] and
              rsi >= self.THRESHOLDS["rsi_overbought"]):

            ibs_score = max(0, (ibs - self.THRESHOLDS["ibs_overbought"]) / (1 - self.THRESHOLDS["ibs_overbought"]))
            rsi_score = max(0, (rsi - self.THRESHOLDS["rsi_overbought"]) / (100 - self.THRESHOLDS["rsi_overbought"]))
            strength = min(100, int((ibs_score + rsi_score) * 50 + 40))

            signals.append(Signal(
                signal_type=SignalType.IBS_RSI_COMBO_SHORT,
                direction=Direction.SHORT,
                strength=strength,
                value=ibs,
                threshold=self.THRESHOLDS["ibs_overbought"],
                description=f"IBS ({ibs:.3f}) + RSI ({rsi:.1f}) COMBO - High conviction short",
                timestamp=timestamp,
                tier=1
            ))

        return signals

    def detect_vix_signals(self, vix: float, vix_ma: float = None) -> List[Signal]:
        """
        Detect VIX regime signals

        VIX acts as a FILTER, not a standalone signal:
        - VIX > 10-day MA = Fear elevated, better for long entries
        - VIX > 30 = Extreme fear, high probability bounce setup
        - VIX < 15 = Complacency, be cautious with longs
        """
        signals = []
        timestamp = now_saudi()

        if vix is None:
            return signals

        # Extreme VIX - panic mode
        if vix >= self.THRESHOLDS["vix_extreme"]:
            signals.append(Signal(
                signal_type=SignalType.VIX_EXTREME,
                direction=Direction.LONG,  # Contrarian
                strength=70,
                value=vix,
                threshold=self.THRESHOLDS["vix_extreme"],
                description=f"VIX at {vix:.1f} - EXTREME FEAR, high probability bounce",
                timestamp=timestamp,
                tier=2
            ))

        # Elevated VIX (above MA)
        elif vix_ma and vix > vix_ma and vix >= self.THRESHOLDS["vix_elevated"]:
            signals.append(Signal(
                signal_type=SignalType.VIX_ELEVATED,
                direction=Direction.LONG,  # Filter favors longs
                strength=40,
                value=vix,
                threshold=vix_ma,
                description=f"VIX at {vix:.1f} > 10-MA ({vix_ma:.1f}) - Elevated fear, good for mean reversion longs",
                timestamp=timestamp,
                tier=2
            ))

        # Complacent VIX
        elif vix <= self.THRESHOLDS["vix_complacent"]:
            signals.append(Signal(
                signal_type=SignalType.VIX_COMPLACENT,
                direction=Direction.NEUTRAL,  # Warning, not directional
                strength=30,
                value=vix,
                threshold=self.THRESHOLDS["vix_complacent"],
                description=f"VIX at {vix:.1f} - COMPLACENT, be cautious with new longs",
                timestamp=timestamp,
                tier=3
            ))

        return signals

    def detect_ma_signals(self, price: float, sma_50: float, sma_200: float = None) -> List[Signal]:
        """
        Detect moving average context signals

        200-MA as filter reduces drawdowns but underperforms as standalone signal
        """
        signals = []
        timestamp = now_saudi()

        if price is None or sma_50 is None:
            return signals

        # Below 50-day MA - mean reversion context
        if price < sma_50:
            deviation = ((sma_50 - price) / sma_50) * 100
            signals.append(Signal(
                signal_type=SignalType.BELOW_50_MA,
                direction=Direction.LONG,  # Mean reversion context
                strength=min(50, int(deviation * 10)),
                value=price,
                threshold=sma_50,
                description=f"Price ${price:.2f} below 50-MA (${sma_50:.2f}) - Mean reversion context",
                timestamp=timestamp,
                tier=3
            ))
        else:
            signals.append(Signal(
                signal_type=SignalType.ABOVE_50_MA,
                direction=Direction.NEUTRAL,
                strength=20,
                value=price,
                threshold=sma_50,
                description=f"Price ${price:.2f} above 50-MA (${sma_50:.2f}) - Trend intact",
                timestamp=timestamp,
                tier=3
            ))

        # Below 200-day MA - deeper correction
        if sma_200 and price < sma_200:
            signals.append(Signal(
                signal_type=SignalType.BELOW_200_MA,
                direction=Direction.LONG,  # Deep value context
                strength=40,
                value=price,
                threshold=sma_200,
                description=f"Price ${price:.2f} below 200-MA (${sma_200:.2f}) - Deep correction zone",
                timestamp=timestamp,
                tier=3
            ))

        return signals

    def detect_intraday_momentum(self, earlier_return: float, last_30min_return: float) -> List[Signal]:
        """
        Detect intraday momentum signal

        Research: Return in last 30 min tends to continue day's direction
        Sharpe ratio: 1.33 (2007-2024 data)
        """
        signals = []
        timestamp = now_saudi()

        if earlier_return is None:
            return signals

        # Bullish intraday momentum
        if earlier_return > 0.3:  # Day is up >0.3%
            signals.append(Signal(
                signal_type=SignalType.INTRADAY_MOMENTUM_BULLISH,
                direction=Direction.LONG,
                strength=min(60, int(earlier_return * 20)),
                value=earlier_return,
                threshold=0.3,
                description=f"Intraday momentum bullish: Day up {earlier_return:.2f}%, last 30min likely to continue",
                timestamp=timestamp,
                tier=2
            ))

        # Bearish intraday momentum
        elif earlier_return < -0.3:  # Day is down >0.3%
            signals.append(Signal(
                signal_type=SignalType.INTRADAY_MOMENTUM_BEARISH,
                direction=Direction.SHORT,
                strength=min(60, int(abs(earlier_return) * 20)),
                value=earlier_return,
                threshold=-0.3,
                description=f"Intraday momentum bearish: Day down {earlier_return:.2f}%, weakness likely to continue",
                timestamp=timestamp,
                tier=2
            ))

        return signals

    def analyze_all(self, market_data: Dict) -> List[Signal]:
        """
        Analyze all available market data and return detected signals

        market_data should contain:
        - ibs: float (Internal Bar Strength)
        - rsi_3: float (RSI with period 3)
        - rsi_2: float (RSI with period 2 - for PUT signals)
        - vix: float (VIX value)
        - vix_10_ma: float (VIX 10-day MA)
        - price: float (current price)
        - sma_50: float (50-day SMA)
        - sma_200: float (200-day SMA, optional)
        - consecutive_up: int (consecutive up days - for PUT signals)
        - earlier_return_pct: float (for intraday momentum)
        - last_30min_return_pct: float (for intraday momentum)
        """
        all_signals = []

        # IBS signals (Tier 1)
        if market_data.get("ibs") is not None:
            signals = self.detect_ibs_signals(market_data["ibs"])
            all_signals.extend(signals)

        # RSI(3) signals (Tier 2)
        if market_data.get("rsi_3") is not None:
            signals = self.detect_rsi_signals(market_data["rsi_3"])
            all_signals.extend(signals)

        # RSI(2) signals (Tier 1/2 - KEY for PUT strategy)
        if market_data.get("rsi_2") is not None:
            signals = self.detect_rsi2_signals(market_data["rsi_2"])
            all_signals.extend(signals)

        # Consecutive up/down days (for PUT/CALL signals)
        if market_data.get("consecutive_up") is not None:
            signals = self.detect_consecutive_days(
                market_data["consecutive_up"],
                market_data.get("consecutive_down", 0)
            )
            all_signals.extend(signals)

        # IBS + RSI Combo (Tier 1 - most powerful)
        if market_data.get("ibs") is not None and market_data.get("rsi_3") is not None:
            signals = self.detect_ibs_rsi_combo(market_data["ibs"], market_data["rsi_3"])
            all_signals.extend(signals)

        # VIX signals (Tier 2/3 - filter)
        if market_data.get("vix") is not None:
            signals = self.detect_vix_signals(
                market_data["vix"],
                market_data.get("vix_10_ma")
            )
            all_signals.extend(signals)

        # MA signals (Tier 3 - context)
        if market_data.get("price") is not None and market_data.get("sma_50") is not None:
            signals = self.detect_ma_signals(
                market_data["price"],
                market_data["sma_50"],
                market_data.get("sma_200")
            )
            all_signals.extend(signals)

        # Intraday momentum (Tier 2)
        if market_data.get("earlier_return_pct") is not None:
            signals = self.detect_intraday_momentum(
                market_data["earlier_return_pct"],
                market_data.get("last_30min_return_pct")
            )
            all_signals.extend(signals)

        return all_signals


def test_signal_detector():
    """Test the signal detector with sample data"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    detector = SPYSignalDetector()

    print("\n" + "=" * 60)
    print("SPY SIGNAL DETECTOR TEST")
    print("=" * 60)

    # Test Case 1: Strong buy setup
    print("\n--- Test 1: IBS + RSI Combo Buy Signal ---")
    market_data = {
        "ibs": 0.12,        # Very oversold
        "rsi_3": 15,        # Very oversold
        "vix": 28,          # Elevated fear
        "vix_10_ma": 22,
        "price": 580,
        "sma_50": 590,
        "sma_200": 575,
    }
    signals = detector.analyze_all(market_data)
    print(f"  Detected {len(signals)} signals:")
    for s in signals:
        print(f"    [{s.tier}] {s.signal_type.value}: {s.direction.value.upper()} ({s.strength})")
        print(f"        {s.description}")

    # Test Case 2: Overbought setup
    print("\n--- Test 2: Overbought Signal ---")
    market_data = {
        "ibs": 0.88,
        "rsi_3": 85,
        "vix": 14,          # Complacent
        "vix_10_ma": 16,
        "price": 600,
        "sma_50": 590,
    }
    signals = detector.analyze_all(market_data)
    print(f"  Detected {len(signals)} signals:")
    for s in signals:
        print(f"    [{s.tier}] {s.signal_type.value}: {s.direction.value.upper()} ({s.strength})")

    # Test Case 3: Neutral market
    print("\n--- Test 3: Neutral Market ---")
    market_data = {
        "ibs": 0.5,
        "rsi_3": 50,
        "vix": 18,
        "vix_10_ma": 17,
        "price": 595,
        "sma_50": 590,
    }
    signals = detector.analyze_all(market_data)
    print(f"  Detected {len(signals)} signals:")
    for s in signals:
        print(f"    [{s.tier}] {s.signal_type.value}: {s.direction.value.upper()} ({s.strength})")

    return signals


if __name__ == "__main__":
    test_signal_detector()
