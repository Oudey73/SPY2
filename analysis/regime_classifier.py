"""
Market Regime Classifier for SPY
Detects: TRENDING_UP, TRENDING_DOWN, RANGE_BOUND, HIGH_VOLATILITY, TRANSITION
Uses ADX(14), ATR(5/20) ratio, and HH/HL pattern detection
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd
import numpy as np

try:
    import ta
except ImportError:
    ta = None


class RegimeType(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITION = "transition"


# Strategies allowed per regime
REGIME_ALLOWED_STRATEGIES = {
    RegimeType.TRENDING_UP: [
        "bull_put_spread", "bull_call_debit", "short_put",
    ],
    RegimeType.TRENDING_DOWN: [
        "bear_call_spread", "bear_put_debit", "short_call",
    ],
    RegimeType.RANGE_BOUND: [
        "iron_condor", "iron_butterfly", "short_strangle",
    ],
    RegimeType.HIGH_VOLATILITY: [
        "iron_condor", "bull_put_spread", "bear_call_spread",
    ],
    RegimeType.TRANSITION: [],
}


@dataclass
class MarketRegime:
    regime: RegimeType
    confidence: float  # 0-100
    allowed_strategies: List[str]
    bias: str  # "bullish", "bearish", "neutral"
    size_adjustment: float  # multiplier 0.0-1.0
    adx: Optional[float] = None
    atr_ratio: Optional[float] = None
    vix: Optional[float] = None
    iv_rank: Optional[float] = None
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "allowed_strategies": self.allowed_strategies,
            "bias": self.bias,
            "size_adjustment": self.size_adjustment,
            "adx": self.adx,
            "atr_ratio": self.atr_ratio,
            "vix": self.vix,
            "iv_rank": self.iv_rank,
            "details": self.details,
        }


class RegimeClassifier:
    """
    Classifies the current market regime for SPY.

    Accepts a YahooCollector to reuse fetched OHLCV data.
    Scoring: HIGH_VOL checked first (VIX>25, ATR ratio, IV rank),
    then trending/range, <60% confidence -> TRANSITION.
    """

    def __init__(self, yahoo_collector=None):
        self.yahoo = yahoo_collector

    def classify(self, market_data: Optional[Dict] = None) -> MarketRegime:
        """
        Classify current market regime.

        Args:
            market_data: Dict from agent's collect_market_data() with vix, iv_rank, etc.

        Returns:
            MarketRegime with classification and confidence.
        """
        try:
            # Fetch OHLCV once
            df = self._get_ohlcv()
            if df is None or len(df) < 20:
                return self._fallback_regime("Insufficient OHLCV data")

            # Compute indicators
            adx = self._compute_adx(df)
            atr_ratio = self._compute_atr_ratio(df)
            hh_hl = self._detect_hh_hl(df)

            # Get VIX and IV rank from market_data
            vix = market_data.get("vix") if market_data else None
            iv_rank = market_data.get("iv_rank") if market_data else None

            # --- Step 1: Check HIGH_VOLATILITY first ---
            hv_score = 0
            hv_reasons = []

            if vix is not None and vix > 25:
                hv_score += 35
                hv_reasons.append(f"VIX={vix:.1f}>25")
            if atr_ratio is not None and atr_ratio > 1.5:
                hv_score += 30
                hv_reasons.append(f"ATR_ratio={atr_ratio:.2f}>1.5")
            if iv_rank is not None and iv_rank > 70:
                hv_score += 25
                hv_reasons.append(f"IV_rank={iv_rank:.0f}>70")
            if adx is not None and adx > 30:
                hv_score += 10
                hv_reasons.append(f"ADX={adx:.1f}>30 (high directional)")

            if hv_score >= 50:
                confidence = min(hv_score, 95)
                bias = "neutral"
                if hh_hl == "uptrend":
                    bias = "bullish"
                elif hh_hl == "downtrend":
                    bias = "bearish"
                return MarketRegime(
                    regime=RegimeType.HIGH_VOLATILITY,
                    confidence=confidence,
                    allowed_strategies=REGIME_ALLOWED_STRATEGIES[RegimeType.HIGH_VOLATILITY],
                    bias=bias,
                    size_adjustment=0.5,
                    adx=adx,
                    atr_ratio=atr_ratio,
                    vix=vix,
                    iv_rank=iv_rank,
                    details=f"HIGH_VOL: {', '.join(hv_reasons)}",
                )

            # --- Step 2: Trending vs Range ---
            trend_score = 0
            range_score = 0
            trend_direction = "neutral"
            trend_reasons = []

            # ADX-based
            if adx is not None:
                if adx > 25:
                    trend_score += 35
                    trend_reasons.append(f"ADX={adx:.1f}>25")
                elif adx < 20:
                    range_score += 35
                    trend_reasons.append(f"ADX={adx:.1f}<20 (range)")
                else:
                    trend_score += 10
                    range_score += 10

            # ATR ratio (expanding = trending)
            if atr_ratio is not None:
                if atr_ratio > 1.2:
                    trend_score += 20
                    trend_reasons.append(f"ATR expanding ({atr_ratio:.2f})")
                elif atr_ratio < 0.8:
                    range_score += 20
                    trend_reasons.append(f"ATR contracting ({atr_ratio:.2f})")

            # HH/HL pattern
            if hh_hl == "uptrend":
                trend_score += 30
                trend_direction = "bullish"
                trend_reasons.append("HH/HL pattern (uptrend)")
            elif hh_hl == "downtrend":
                trend_score += 30
                trend_direction = "bearish"
                trend_reasons.append("LH/LL pattern (downtrend)")
            else:
                range_score += 20
                trend_reasons.append("No clear trend pattern")

            # SMA alignment
            sma_trend = self._check_sma_alignment(df)
            if sma_trend == "bullish":
                trend_score += 15
                trend_direction = "bullish"
            elif sma_trend == "bearish":
                trend_score += 15
                trend_direction = "bearish"
            else:
                range_score += 10

            # Determine regime
            if trend_score > range_score:
                confidence = min(trend_score, 95)
                if confidence < 60:
                    return MarketRegime(
                        regime=RegimeType.TRANSITION,
                        confidence=confidence,
                        allowed_strategies=REGIME_ALLOWED_STRATEGIES[RegimeType.TRANSITION],
                        bias=trend_direction,
                        size_adjustment=0.5,
                        adx=adx, atr_ratio=atr_ratio, vix=vix, iv_rank=iv_rank,
                        details=f"TRANSITION (low confidence): {', '.join(trend_reasons)}",
                    )
                regime = RegimeType.TRENDING_UP if trend_direction == "bullish" else RegimeType.TRENDING_DOWN
                return MarketRegime(
                    regime=regime,
                    confidence=confidence,
                    allowed_strategies=REGIME_ALLOWED_STRATEGIES[regime],
                    bias=trend_direction,
                    size_adjustment=1.0 if confidence >= 75 else 0.75,
                    adx=adx, atr_ratio=atr_ratio, vix=vix, iv_rank=iv_rank,
                    details=f"{regime.value}: {', '.join(trend_reasons)}",
                )
            else:
                confidence = min(range_score, 95)
                if confidence < 60:
                    return MarketRegime(
                        regime=RegimeType.TRANSITION,
                        confidence=confidence,
                        allowed_strategies=REGIME_ALLOWED_STRATEGIES[RegimeType.TRANSITION],
                        bias="neutral",
                        size_adjustment=0.5,
                        adx=adx, atr_ratio=atr_ratio, vix=vix, iv_rank=iv_rank,
                        details=f"TRANSITION (low confidence): {', '.join(trend_reasons)}",
                    )
                return MarketRegime(
                    regime=RegimeType.RANGE_BOUND,
                    confidence=confidence,
                    allowed_strategies=REGIME_ALLOWED_STRATEGIES[RegimeType.RANGE_BOUND],
                    bias="neutral",
                    size_adjustment=0.85,
                    adx=adx, atr_ratio=atr_ratio, vix=vix, iv_rank=iv_rank,
                    details=f"RANGE_BOUND: {', '.join(trend_reasons)}",
                )

        except Exception as e:
            logger.error(f"Regime classification error: {e}")
            return self._fallback_regime(str(e))

    def _get_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetch OHLCV via YahooCollector or directly."""
        if self.yahoo is not None:
            return self.yahoo.get_daily_ohlcv("SPY", days=100)
        try:
            import yfinance as yf
            ticker = yf.Ticker("SPY")
            hist = ticker.history(period="100d")
            if hist.empty:
                return None
            return hist[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return None

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Compute ADX(14) using ta library."""
        if ta is None:
            return self._manual_adx(df, period)
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], window=period
            )
            adx_series = adx_indicator.adx()
            val = adx_series.iloc[-1]
            return float(val) if pd.notna(val) else None
        except Exception as e:
            logger.warning(f"ta ADX failed, using manual: {e}")
            return self._manual_adx(df, period)

    def _manual_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Manual ADX calculation as fallback."""
        try:
            high = df["High"].values
            low = df["Low"].values
            close = df["Close"].values

            plus_dm = np.zeros(len(df))
            minus_dm = np.zeros(len(df))
            tr = np.zeros(len(df))

            for i in range(1, len(df)):
                h_diff = high[i] - high[i - 1]
                l_diff = low[i - 1] - low[i]
                plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0
                minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

            # Smoothed averages
            atr = pd.Series(tr).rolling(period).mean().values
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)

            dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
            adx = pd.Series(dx).rolling(period).mean().iloc[-1]
            return float(adx) if pd.notna(adx) else None
        except Exception:
            return None

    def _compute_atr_ratio(self, df: pd.DataFrame) -> Optional[float]:
        """Compute ATR(5) / ATR(20) ratio. >1 = expanding volatility."""
        try:
            if ta is not None:
                atr5 = ta.volatility.AverageTrueRange(
                    high=df["High"], low=df["Low"], close=df["Close"], window=5
                ).average_true_range().iloc[-1]
                atr20 = ta.volatility.AverageTrueRange(
                    high=df["High"], low=df["Low"], close=df["Close"], window=20
                ).average_true_range().iloc[-1]
            else:
                # Manual ATR
                tr = pd.DataFrame({
                    "hl": df["High"] - df["Low"],
                    "hc": abs(df["High"] - df["Close"].shift(1)),
                    "lc": abs(df["Low"] - df["Close"].shift(1)),
                }).max(axis=1)
                atr5 = tr.rolling(5).mean().iloc[-1]
                atr20 = tr.rolling(20).mean().iloc[-1]

            if pd.notna(atr5) and pd.notna(atr20) and atr20 > 0:
                return float(atr5 / atr20)
            return None
        except Exception:
            return None

    def _detect_hh_hl(self, df: pd.DataFrame, lookback: int = 10) -> str:
        """
        Detect Higher Highs / Higher Lows (uptrend) or Lower Highs / Lower Lows (downtrend)
        over the last `lookback` sessions.

        Returns: "uptrend", "downtrend", or "mixed"
        """
        try:
            recent = df.tail(lookback)
            if len(recent) < 5:
                return "mixed"

            highs = recent["High"].values
            lows = recent["Low"].values

            # Check last 3 swing points
            mid = len(highs) // 2
            first_half_high = highs[:mid].max()
            second_half_high = highs[mid:].max()
            first_half_low = lows[:mid].min()
            second_half_low = lows[mid:].min()

            higher_highs = second_half_high > first_half_high
            higher_lows = second_half_low > first_half_low
            lower_highs = second_half_high < first_half_high
            lower_lows = second_half_low < first_half_low

            if higher_highs and higher_lows:
                return "uptrend"
            elif lower_highs and lower_lows:
                return "downtrend"
            return "mixed"
        except Exception:
            return "mixed"

    def _check_sma_alignment(self, df: pd.DataFrame) -> str:
        """Check if SMAs are aligned (bullish: price > 20 > 50, bearish: price < 20 < 50)."""
        try:
            close = df["Close"].iloc[-1]
            sma20 = df["Close"].rolling(20).mean().iloc[-1]
            sma50 = df["Close"].rolling(50).mean().iloc[-1]

            if pd.isna(sma20) or pd.isna(sma50):
                return "neutral"

            if close > sma20 > sma50:
                return "bullish"
            elif close < sma20 < sma50:
                return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _fallback_regime(self, reason: str) -> MarketRegime:
        """Return TRANSITION regime as fallback."""
        logger.warning(f"Regime fallback to TRANSITION: {reason}")
        return MarketRegime(
            regime=RegimeType.TRANSITION,
            confidence=30.0,
            allowed_strategies=[],
            bias="neutral",
            size_adjustment=0.5,
            details=f"Fallback: {reason}",
        )
