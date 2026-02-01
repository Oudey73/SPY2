"""
IV Signal Integration - Combines ORATS IV data with signal detection
"""
from typing import Dict, List, Optional
from datetime import datetime
import pytz
from enum import Enum
from dataclasses import dataclass
from loguru import logger

try:
    from collectors.orats_collector import ORATSCollector
except ImportError:
    ORATSCollector = None


# Saudi Arabia timezone
SAUDI_TZ = pytz.timezone("Asia/Riyadh")

def now_saudi() -> str:
    return datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")


class IVSignalType(Enum):
    IV_HIGH = "iv_high"
    IV_EXTREME = "iv_extreme"
    IV_LOW = "iv_low"
    IV_BACKWARDATION = "iv_backwardation"
    IV_STEEP_PUT_SKEW = "iv_steep_put_skew"
    IV_CONTANGO = "iv_contango"


@dataclass
class IVSignal:
    signal_type: IVSignalType
    strength: int
    value: float
    description: str
    strategy_bias: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "value": self.value,
            "description": self.description,
            "strategy_bias": self.strategy_bias,
            "timestamp": self.timestamp
        }


class IVSignalDetector:
    def __init__(self):
        self.orats = ORATSCollector() if ORATSCollector else None

    def is_configured(self) -> bool:
        return self.orats is not None and self.orats.is_configured()

    def detect_iv_signals(self, symbol: str = "SPY") -> List[IVSignal]:
        if not self.is_configured():
            logger.warning("ORATS not configured")
            return []

        signals = []
        timestamp = now_saudi()

        iv_data = self.orats.get_iv_rank(symbol)
        if iv_data:
            iv_rank = iv_data.get("iv_rank", 50)
            current_iv = iv_data.get("current_iv", 0)

            if iv_rank >= 80:
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_EXTREME,
                    strength=90,
                    value=iv_rank,
                    description=f"IV Rank {iv_rank:.0f} - EXTREME HIGH, sell premium aggressively",
                    strategy_bias="SELL_PREMIUM",
                    timestamp=timestamp
                ))
            elif iv_rank >= 70:
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_HIGH,
                    strength=70,
                    value=iv_rank,
                    description=f"IV Rank {iv_rank:.0f} - High, favor selling premium",
                    strategy_bias="SELL_PREMIUM",
                    timestamp=timestamp
                ))
            elif iv_rank <= 30:
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_LOW,
                    strength=60,
                    value=iv_rank,
                    description=f"IV Rank {iv_rank:.0f} - Low, favor buying premium",
                    strategy_bias="BUY_PREMIUM",
                    timestamp=timestamp
                ))

        ts_data = self.orats.get_term_structure(symbol)
        if ts_data:
            structure = ts_data.get("structure", "FLAT")
            spread = ts_data.get("spread", 0)

            if structure == "BACKWARDATION":
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_BACKWARDATION,
                    strength=75,
                    value=spread,
                    description=f"Term structure INVERTED (spread {spread:.1f}) - stress signal",
                    strategy_bias="CAUTION",
                    timestamp=timestamp
                ))
            elif structure == "CONTANGO":
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_CONTANGO,
                    strength=50,
                    value=spread,
                    description=f"Normal contango (spread {spread:.1f}) - sell front-month",
                    strategy_bias="SELL_FRONT_MONTH",
                    timestamp=timestamp
                ))

        skew_data = self.orats.get_skew(symbol)
        if skew_data:
            skew_type = skew_data.get("skew_type", "NORMAL")
            skewing = skew_data.get("skewing", 0)

            if skew_type == "STEEP_PUT":
                signals.append(IVSignal(
                    signal_type=IVSignalType.IV_STEEP_PUT_SKEW,
                    strength=65,
                    value=abs(skewing),
                    description=f"Steep put skew ({skewing:.3f}) - high hedging demand",
                    strategy_bias="SELL_PUT_SPREADS",
                    timestamp=timestamp
                ))

        return signals

    def get_strategy_recommendation(self, symbol: str = "SPY") -> Optional[Dict]:
        if not self.is_configured():
            return None

        full_data = self.orats.get_all_iv_data(symbol)
        if not full_data:
            return None

        signals = self.detect_iv_signals(symbol)

        return {
            "symbol": symbol,
            "iv_data": full_data,
            "iv_signals": [s.to_dict() for s in signals],
            "strategy_recommendation": full_data.get("strategy_recommendation"),
            "timestamp": now_saudi()
        }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("IV Signal Detector Test")
    print("=" * 50)

    detector = IVSignalDetector()
    if not detector.is_configured():
        print("ORATS not configured!")
    else:
        signals = detector.detect_iv_signals("SPY")
        print(f"Detected {len(signals)} IV signals:")
        for s in signals:
            print(f"  [{s.strength}] {s.signal_type.value}: {s.description}")
            print(f"       Strategy Bias: {s.strategy_bias}")

        print()
        rec = detector.get_strategy_recommendation("SPY")
        if rec:
            print(f"Strategy: {rec.get('strategy_recommendation')}")

