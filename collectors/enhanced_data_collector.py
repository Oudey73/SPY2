"""
Enhanced Data Collector for Multi-Factor Scoring System
Provides additional market data inputs for enhanced scoring including:
- CVD (Cumulative Volume Delta) slope - via Polygon.io tick data
- RVOL (Relative Volume) ratio - via Yahoo Finance
- DXY (Dollar Index) trend - via Yahoo Finance
- High probability session detection
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Eastern timezone for market hours
ET = pytz.timezone('US/Eastern')


class EnhancedDataCollector:
    """
    Collector for enhanced market data inputs used by the multi-factor scoring system.

    Implemented:
    - get_dxy_trend() - Real DXY data via Yahoo Finance
    - get_rvol() - Real relative volume calculation
    - is_high_probability_session() - Session time check

    Placeholder (requires paid data source):
    - get_cvd_slope() - Requires tick-level order flow data
    """

    def __init__(self, polygon_collector=None):
        """
        Initialize the enhanced data collector.

        Args:
            polygon_collector: Optional PolygonCollector instance for CVD.
                              If not provided, will create one if API key exists.
        """
        # Cache for DXY data to avoid repeated API calls
        self._dxy_cache: Optional[Dict] = None
        self._dxy_cache_time: Optional[datetime] = None
        self._cache_duration_minutes = 5

        # Cache for RVOL
        self._rvol_cache: Optional[Dict] = None
        self._rvol_cache_time: Optional[datetime] = None

        # Cache for CVD
        self._cvd_cache: Optional[Dict] = None
        self._cvd_cache_time: Optional[datetime] = None
        self._cvd_cache_duration_minutes = 2  # CVD updates more frequently

        # Polygon collector for CVD
        self._polygon = polygon_collector
        if self._polygon is None:
            # Try to create one if API key exists
            polygon_key = os.getenv("POLYGON_API_KEY", "")
            if polygon_key:
                from .polygon_collector import PolygonCollector
                self._polygon = PolygonCollector(api_key=polygon_key)
                logger.info("EnhancedDataCollector: Polygon CVD enabled")

    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """Check if cached data is still valid"""
        if cache_time is None:
            return False
        elapsed = (datetime.now() - cache_time).total_seconds() / 60
        return elapsed < self._cache_duration_minutes

    def get_dxy_trend(self) -> Optional[str]:
        """
        Get the DXY (US Dollar Index) trend direction.

        Fetches DXY data from Yahoo Finance and compares current price
        to 20-period SMA to determine trend.

        DXY trend context for SPY:
        - Strong dollar (DXY up) often correlates with equity weakness
        - Weak dollar (DXY down) often correlates with equity strength

        Returns:
            "up" - DXY above 20-SMA (strong dollar, bearish for SPY)
            "down" - DXY below 20-SMA (weak dollar, bullish for SPY)
            "neutral" - DXY within 0.2% of 20-SMA
            None - If data unavailable
        """
        try:
            # Check cache
            if self._is_cache_valid(self._dxy_cache_time) and self._dxy_cache:
                return self._dxy_cache.get("trend")

            # Fetch DXY data (Yahoo symbol: DX-Y.NYB)
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="30d")

            if hist.empty or len(hist) < 20:
                logger.warning("Insufficient DXY data")
                return None

            # Calculate 20-day SMA
            hist["SMA_20"] = hist["Close"].rolling(window=20).mean()

            current_price = float(hist["Close"].iloc[-1])
            sma_20 = float(hist["SMA_20"].iloc[-1])

            if pd.isna(sma_20):
                return None

            # Determine trend (0.2% threshold for neutral)
            deviation_pct = ((current_price - sma_20) / sma_20) * 100

            if deviation_pct > 0.2:
                trend = "up"
            elif deviation_pct < -0.2:
                trend = "down"
            else:
                trend = "neutral"

            # Cache the result
            self._dxy_cache = {
                "trend": trend,
                "price": current_price,
                "sma_20": sma_20,
                "deviation_pct": deviation_pct,
            }
            self._dxy_cache_time = datetime.now()

            logger.debug(f"DXY: {current_price:.2f} vs SMA20: {sma_20:.2f} ({deviation_pct:+.2f}%) -> {trend}")
            return trend

        except Exception as e:
            logger.error(f"Error fetching DXY data: {e}")
            return None

    def get_dxy_details(self) -> Optional[Dict]:
        """
        Get detailed DXY data including price, SMA, and deviation.

        Returns:
            Dict with: price, sma_20, deviation_pct, trend
            None if data unavailable
        """
        # Ensure cache is populated
        self.get_dxy_trend()
        return self._dxy_cache

    def get_rvol(self, symbol: str = "SPY") -> Optional[float]:
        """
        Get the RVOL (Relative Volume) ratio for SPY.

        RVOL = Current cumulative volume / Average volume at this time of day

        Calculation:
        1. Get current intraday volume
        2. Get historical intraday volume patterns (past 20 days)
        3. Calculate average volume at current time of day
        4. Return ratio

        Values:
        - RVOL > 1.5: High relative volume (unusual activity)
        - RVOL 1.0-1.5: Normal to elevated volume
        - RVOL < 1.0: Below average volume

        Returns:
            RVOL ratio (float), or None if data unavailable
        """
        try:
            # Check cache
            if self._is_cache_valid(self._rvol_cache_time) and self._rvol_cache:
                return self._rvol_cache.get("rvol")

            ticker = yf.Ticker(symbol)

            # Get intraday data for today (5-min intervals)
            today_data = ticker.history(period="1d", interval="5m")
            if today_data.empty:
                logger.warning("No intraday data for RVOL calculation")
                return None

            # Get current cumulative volume
            current_volume = today_data["Volume"].sum()

            # Get the current time in market hours
            now_et = datetime.now(ET)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

            # Minutes since market open
            if now_et < market_open:
                # Pre-market, can't calculate meaningful RVOL
                return None

            minutes_since_open = (now_et - market_open).total_seconds() / 60
            if minutes_since_open <= 0:
                return None

            # Get historical data (past 20 trading days)
            hist_data = ticker.history(period="30d", interval="1d")
            if hist_data.empty or len(hist_data) < 10:
                logger.warning("Insufficient historical data for RVOL")
                return None

            # Calculate average daily volume
            avg_daily_volume = hist_data["Volume"].tail(20).mean()

            # Estimate expected volume at this time of day
            # Assume relatively uniform distribution (simplification)
            # Market is open 390 minutes (9:30 - 16:00)
            total_market_minutes = 390
            time_fraction = min(minutes_since_open / total_market_minutes, 1.0)

            # Adjust for typical intraday volume pattern (U-shaped)
            # More volume at open and close, less in middle
            # Simplified: first/last 30 min have 1.5x avg, middle has 0.8x
            if minutes_since_open <= 30:
                volume_weight = 1.3  # First 30 min higher volume
            elif minutes_since_open >= 360:  # Last 30 min
                volume_weight = 1.3
            else:
                volume_weight = 0.9  # Mid-day lower volume

            expected_volume = avg_daily_volume * time_fraction * volume_weight

            if expected_volume <= 0:
                return None

            # Calculate RVOL
            rvol = current_volume / expected_volume

            # Cache the result
            self._rvol_cache = {
                "rvol": round(rvol, 2),
                "current_volume": int(current_volume),
                "expected_volume": int(expected_volume),
                "avg_daily_volume": int(avg_daily_volume),
                "minutes_since_open": int(minutes_since_open),
            }
            self._rvol_cache_time = datetime.now()

            logger.debug(f"RVOL: {current_volume:,} / {expected_volume:,.0f} = {rvol:.2f}")
            return round(rvol, 2)

        except Exception as e:
            logger.error(f"Error calculating RVOL: {e}")
            return None

    def get_rvol_details(self) -> Optional[Dict]:
        """
        Get detailed RVOL data including volumes and timing.

        Returns:
            Dict with: rvol, current_volume, expected_volume, avg_daily_volume
            None if data unavailable
        """
        # Ensure cache is populated
        self.get_rvol()
        return self._rvol_cache

    def get_cvd_slope(self, periods: int = 5) -> Optional[float]:
        """
        Get the CVD (Cumulative Volume Delta) slope over specified periods.

        CVD measures the cumulative difference between buying and selling volume.
        A positive slope indicates increasing buying pressure.
        A negative slope indicates increasing selling pressure.

        Args:
            periods: Number of 5-minute periods to calculate slope over (default: 5)

        Returns:
            CVD slope value (normalized), or None if data unavailable

        Implementation:
        - Uses Polygon.io tick data when API key is configured
        - Falls back to None if Polygon is not available
        """
        # Check cache first
        if self._is_cvd_cache_valid() and self._cvd_cache:
            return self._cvd_cache.get("cvd_slope_normalized")

        # Use Polygon if available
        if self._polygon and self._polygon.api_key:
            try:
                cvd_data = self._polygon.calculate_cvd(
                    symbol="SPY",
                    lookback_minutes=periods * 5,
                    bar_size_minutes=5
                )

                if cvd_data:
                    # Cache the full CVD data
                    self._cvd_cache = cvd_data
                    self._cvd_cache_time = datetime.now()

                    slope = cvd_data.get("cvd_slope_normalized")
                    logger.debug(
                        f"CVD: slope={slope:.4f}, "
                        f"buy={cvd_data.get('total_buy_volume', 0):,}, "
                        f"sell={cvd_data.get('total_sell_volume', 0):,}, "
                        f"interp={cvd_data.get('interpretation', 'N/A')}"
                    )
                    return slope

            except Exception as e:
                logger.error(f"Error fetching CVD from Polygon: {e}")
                return None

        # No Polygon API - return None
        logger.debug("CVD unavailable: Polygon API key not configured")
        return None

    def _is_cvd_cache_valid(self) -> bool:
        """Check if CVD cache is still valid"""
        if self._cvd_cache_time is None:
            return False
        elapsed = (datetime.now() - self._cvd_cache_time).total_seconds() / 60
        return elapsed < self._cvd_cache_duration_minutes

    def get_cvd_details(self) -> Optional[Dict]:
        """
        Get detailed CVD data including volume breakdown.

        Returns:
            Dict with: cvd_slope, total_buy_volume, total_sell_volume,
                      buy_sell_ratio, interpretation
            None if data unavailable
        """
        # Ensure cache is populated
        self.get_cvd_slope()
        return self._cvd_cache

    def is_high_probability_session(self) -> bool:
        """
        Check if current time is within a high probability trading session.

        Research shows higher win rates during:
        - Market open: 09:30-11:00 ET (first 90 minutes)
        - Power hour: 15:00-16:00 ET (last hour)

        These windows have:
        - Higher volume and liquidity
        - More directional moves
        - Better signal reliability

        Returns:
            True if within high probability session window, False otherwise
        """
        now = datetime.now(ET)

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        current_time = now.time()

        # Morning session: 09:30 - 11:00 ET
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:00", "%H:%M").time()

        # Power hour: 15:00 - 16:00 ET
        power_start = datetime.strptime("15:00", "%H:%M").time()
        power_end = datetime.strptime("16:00", "%H:%M").time()

        in_morning_session = morning_start <= current_time <= morning_end
        in_power_hour = power_start <= current_time <= power_end

        return in_morning_session or in_power_hour

    def get_session_info(self) -> Dict:
        """
        Get detailed session information.

        Returns:
            Dict with session details
        """
        now = datetime.now(ET)
        current_time = now.time()

        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()

        is_market_hours = market_open <= current_time <= market_close and now.weekday() < 5

        session_name = "CLOSED"
        if is_market_hours:
            if current_time <= datetime.strptime("11:00", "%H:%M").time():
                session_name = "MORNING_OPEN"
            elif current_time >= datetime.strptime("15:00", "%H:%M").time():
                session_name = "POWER_HOUR"
            else:
                session_name = "MIDDAY"

        return {
            "current_time_et": now.strftime("%H:%M:%S"),
            "is_market_hours": is_market_hours,
            "is_hp_session": self.is_high_probability_session(),
            "session_name": session_name,
            "weekday": now.strftime("%A"),
        }

    def get_all_enhanced_data(self) -> Dict:
        """
        Collect all enhanced data inputs in a single call.

        Returns:
            Dictionary with all enhanced data fields:
            - cvd_slope: float or None (placeholder)
            - rvol: float or None
            - dxy_trend: str or None ("up", "down", "neutral")
            - is_hp_session: bool
        """
        return {
            "cvd_slope": self.get_cvd_slope(),
            "rvol": self.get_rvol(),
            "dxy_trend": self.get_dxy_trend(),
            "is_hp_session": self.is_high_probability_session(),
        }


# =============================================================================
# ORDER FLOW DATA PROVIDERS (for CVD implementation)
# =============================================================================

ORDER_FLOW_PROVIDERS = """
To implement real CVD (Cumulative Volume Delta) and order flow analysis,
you need tick-level data with trade direction attribution. Options:

FREE / LOW COST:
----------------
1. Polygon.io ($29/mo starter, $79/mo for WebSocket)
   - Tick-by-tick trades with trade conditions
   - Can infer direction from trade conditions
   - API: https://polygon.io/docs/stocks/get_v3_trades__stockticker
   - Best for: Real-time tick data, trade conditions

2. Alpaca Markets (Free with account)
   - Real-time trades via WebSocket
   - Limited historical tick data
   - API: https://alpaca.markets/docs/api-references/market-data-api/
   - Best for: Basic order flow if you have an Alpaca account

PROFESSIONAL ($100-500/mo):
---------------------------
3. Bookmap ($49-149/mo)
   - Full DOM (Depth of Market) visualization
   - Order flow imbalances, large orders
   - Heatmap of limit orders
   - Best for: Visual order flow analysis

4. Sierra Chart ($26-36/mo + data feed)
   - Professional DOM and order flow tools
   - Volume profile, footprint charts
   - Requires separate data feed subscription
   - Best for: Serious traders, custom indicators

5. Tradovate ($0 with funded account)
   - Real-time DOM data
   - Order flow indicators
   - Best for: Futures traders

6. Jigsaw Trading ($579 one-time)
   - Auction Vista, depth analysis
   - Order flow tools
   - Best for: Dedicated order flow traders

DARK POOL DATA ($50-200/mo):
----------------------------
7. FlowAlgo ($99/mo)
   - Dark pool prints
   - Unusual options activity
   - Block trades
   - API available
   - Best for: Dark pool + options flow

8. Unusual Whales ($57-97/mo)
   - Options flow
   - Dark pool data
   - Congress trading
   - API available
   - Best for: Options flow focus

9. BlackBoxStocks ($99/mo)
   - Dark pool data
   - Options flow
   - Real-time alerts
   - Best for: Integrated platform

10. Quiver Quantitative ($10-50/mo)
    - Dark pool data
    - Congress trading
    - Insider transactions
    - API: https://www.quiverquant.com/
    - Best for: Alternative data

RECOMMENDATION FOR THIS PROJECT:
--------------------------------
1. Start with Polygon.io ($29/mo) for tick data
   - Implement CVD using trade conditions
   - Can detect large trades, sweeps

2. Add FlowAlgo or Unusual Whales for dark pool
   - Complements technical signals
   - Unusual activity detection

Example Polygon.io CVD implementation pseudo-code:
```python
# Get tick data
ticks = polygon.get_trades("SPY", timestamp_gte=start, timestamp_lte=end)

# Classify each trade
for tick in ticks:
    if tick.conditions includes 'B' or price >= ask:
        buy_volume += tick.size
    elif tick.conditions includes 'S' or price <= bid:
        sell_volume += tick.size

# Calculate delta
delta = buy_volume - sell_volume
cvd = cumsum(delta)
cvd_slope = linear_regression_slope(cvd[-5:])
```
"""


def test_enhanced_collector():
    """Test the enhanced data collector"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "=" * 60)
    print("ENHANCED DATA COLLECTOR TEST")
    print("=" * 60)

    collector = EnhancedDataCollector()

    # Check Polygon status
    print("\n--- Data Sources ---")
    polygon_configured = collector._polygon is not None and collector._polygon.api_key
    print(f"  Polygon.io (CVD): {'Configured' if polygon_configured else 'Not configured'}")
    print(f"  Yahoo Finance (DXY, RVOL): Available")

    # Test session info
    print("\n--- Session Info ---")
    session = collector.get_session_info()
    for key, value in session.items():
        print(f"  {key}: {value}")

    # Test DXY
    print("\n--- DXY Trend ---")
    dxy_trend = collector.get_dxy_trend()
    print(f"  Trend: {dxy_trend}")
    dxy_details = collector.get_dxy_details()
    if dxy_details:
        print(f"  Price: {dxy_details.get('price', 'N/A'):.2f}")
        print(f"  SMA20: {dxy_details.get('sma_20', 'N/A'):.2f}")
        print(f"  Deviation: {dxy_details.get('deviation_pct', 0):+.2f}%")

    # Test RVOL
    print("\n--- RVOL ---")
    rvol = collector.get_rvol()
    print(f"  RVOL: {rvol}")
    rvol_details = collector.get_rvol_details()
    if rvol_details:
        print(f"  Current Volume: {rvol_details.get('current_volume', 0):,}")
        print(f"  Expected Volume: {rvol_details.get('expected_volume', 0):,}")
        print(f"  Avg Daily Volume: {rvol_details.get('avg_daily_volume', 0):,}")
    elif not collector.get_session_info().get("is_market_hours"):
        print("  (Market closed - RVOL requires market hours)")

    # Test CVD
    print("\n--- CVD (Cumulative Volume Delta) ---")
    if polygon_configured:
        cvd = collector.get_cvd_slope()
        print(f"  CVD Slope: {cvd}")
        cvd_details = collector.get_cvd_details()
        if cvd_details:
            print(f"  Buy Volume: {cvd_details.get('total_buy_volume', 0):,}")
            print(f"  Sell Volume: {cvd_details.get('total_sell_volume', 0):,}")
            print(f"  Buy/Sell Ratio: {cvd_details.get('buy_sell_ratio', 'N/A')}")
            print(f"  Interpretation: {cvd_details.get('interpretation', 'N/A')}")
            print(f"  Trades Analyzed: {cvd_details.get('trades_analyzed', 0):,}")
        elif not collector.get_session_info().get("is_market_hours"):
            print("  (Market closed - CVD requires market hours)")
    else:
        print("  CVD Slope: None")
        print("  (Set POLYGON_API_KEY in .env to enable CVD)")

    # Test combined
    print("\n--- All Enhanced Data ---")
    data = collector.get_all_enhanced_data()
    for key, value in data.items():
        print(f"  {key}: {value}")

    # Print order flow provider info if CVD not configured
    if not polygon_configured:
        print("\n" + "=" * 60)
        print("TO ENABLE CVD:")
        print("=" * 60)
        print("1. Get a Polygon.io API key at https://polygon.io/")
        print("2. Add to .env: POLYGON_API_KEY=your_key_here")
        print("3. Restart the agent")
        print("\nPolygon.io pricing:")
        print("  - Free tier: 5 API calls/min (delayed data)")
        print("  - Starter ($29/mo): Real-time data, more calls")
        print("  - Developer ($79/mo): WebSocket, full tick data")

    return data


if __name__ == "__main__":
    test_enhanced_collector()
