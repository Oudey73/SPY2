"""
Polygon.io Data Collector
Fetches: Real-time quotes, options data, aggregates, tick-level trades
Requires API key ($29/mo for real-time, free tier is delayed)

CVD (Cumulative Volume Delta) calculation:
- Fetches tick-level trades
- Classifies as buy/sell based on trade conditions and price vs bid/ask
- Calculates cumulative delta and slope
"""
import os
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

POLYGON_BASE = "https://api.polygon.io"

# Polygon trade condition codes that indicate buy/sell
# See: https://polygon.io/docs/stocks/get_v3_trades__stocksticker
# These are SIP condition codes
BUY_CONDITIONS = {
    'B',   # Buy side
    'W',   # Weighted average price trade (often institutional buy)
}
SELL_CONDITIONS = {
    'S',   # Sell side
    'T',   # Form T trade (after hours, often selling)
}
# Conditions that indicate we should use price vs quote to determine side
NEUTRAL_CONDITIONS = {
    '@',   # Regular trade
    'F',   # Intermarket sweep
    'I',   # Odd lot trade
    'X',   # Cross trade
}


class PolygonCollector:
    """Collects market data from Polygon.io"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.session = requests.Session()

        if not self.api_key:
            logger.warning("Polygon API key not set. Get one at https://polygon.io/")

    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make API request"""
        if not self.api_key:
            logger.error("Polygon API key required")
            return None

        try:
            params = params or {}
            params["apiKey"] = self.api_key

            url = f"{POLYGON_BASE}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ERROR":
                logger.error(f"Polygon API error: {data.get('error')}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Polygon request error: {e}")
            return None

    def get_real_time_quote(self, symbol: str = "SPY") -> Optional[Dict]:
        """
        Get real-time quote (requires paid plan for real-time)
        Free tier: 15-min delayed
        """
        endpoint = f"/v2/last/nbbo/{symbol}"
        data = self._request(endpoint)

        if data and data.get("results"):
            result = data["results"]
            return {
                "symbol": symbol,
                "bid": result.get("p"),  # Bid price
                "ask": result.get("P"),  # Ask price
                "bid_size": result.get("s"),
                "ask_size": result.get("S"),
                "spread": result.get("P", 0) - result.get("p", 0) if result.get("P") and result.get("p") else None,
                "timestamp": result.get("t"),
            }
        return None

    def get_last_trade(self, symbol: str = "SPY") -> Optional[Dict]:
        """Get last trade price"""
        endpoint = f"/v2/last/trade/{symbol}"
        data = self._request(endpoint)

        if data and data.get("results"):
            result = data["results"]
            return {
                "symbol": symbol,
                "price": result.get("p"),
                "size": result.get("s"),
                "timestamp": result.get("t"),
                "exchange": result.get("x"),
            }
        return None

    def get_previous_close(self, symbol: str = "SPY") -> Optional[Dict]:
        """Get previous day's OHLCV"""
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        data = self._request(endpoint)

        if data and data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "symbol": symbol,
                "open": result.get("o"),
                "high": result.get("h"),
                "low": result.get("l"),
                "close": result.get("c"),
                "volume": result.get("v"),
                "vwap": result.get("vw"),
                "timestamp": result.get("t"),
            }
        return None

    def get_aggregates(
        self,
        symbol: str = "SPY",
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = None,
        to_date: str = None,
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Get aggregate bars (OHLCV)

        Args:
            symbol: Stock symbol
            multiplier: Size of the timespan multiplier
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results

        Returns list of OHLCV bars
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        data = self._request(endpoint, {"limit": limit, "sort": "asc"})

        if data and data.get("results"):
            return [{
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
                "vwap": r.get("vw"),
                "timestamp": r.get("t"),
                "num_transactions": r.get("n"),
            } for r in data["results"]]

        return None

    def get_intraday_bars(self, symbol: str = "SPY", minutes: int = 5) -> Optional[List[Dict]]:
        """Get intraday minute bars for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_aggregates(
            symbol=symbol,
            multiplier=minutes,
            timespan="minute",
            from_date=today,
            to_date=today,
            limit=500
        )

    def get_snapshot(self, symbol: str = "SPY") -> Optional[Dict]:
        """
        Get comprehensive snapshot with current price, prev day, and today's data
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        data = self._request(endpoint)

        if data and data.get("ticker"):
            ticker = data["ticker"]
            day = ticker.get("day", {})
            prev_day = ticker.get("prevDay", {})
            last_trade = ticker.get("lastTrade", {})
            last_quote = ticker.get("lastQuote", {})
            min_data = ticker.get("min", {})

            return {
                "symbol": symbol,
                "updated": ticker.get("updated"),
                "current": {
                    "price": last_trade.get("p"),
                    "bid": last_quote.get("p"),
                    "ask": last_quote.get("P"),
                    "spread": last_quote.get("P", 0) - last_quote.get("p", 0) if last_quote.get("P") and last_quote.get("p") else None,
                },
                "today": {
                    "open": day.get("o"),
                    "high": day.get("h"),
                    "low": day.get("l"),
                    "close": day.get("c"),
                    "volume": day.get("v"),
                    "vwap": day.get("vw"),
                },
                "previous": {
                    "open": prev_day.get("o"),
                    "high": prev_day.get("h"),
                    "low": prev_day.get("l"),
                    "close": prev_day.get("c"),
                    "volume": prev_day.get("v"),
                    "vwap": prev_day.get("vw"),
                },
                "minute": {
                    "open": min_data.get("o"),
                    "high": min_data.get("h"),
                    "low": min_data.get("l"),
                    "close": min_data.get("c"),
                    "volume": min_data.get("v"),
                    "vwap": min_data.get("vw"),
                },
                "change": {
                    "today_change": ticker.get("todaysChange"),
                    "today_change_percent": ticker.get("todaysChangePerc"),
                }
            }
        return None

    def get_options_chain(self, symbol: str = "SPY", expiration_gte: str = None) -> Optional[List[Dict]]:
        """
        Get options contracts for a symbol

        Note: Requires Options tier subscription
        """
        if not expiration_gte:
            expiration_gte = datetime.now().strftime("%Y-%m-%d")

        endpoint = f"/v3/reference/options/contracts"
        data = self._request(endpoint, {
            "underlying_ticker": symbol,
            "expiration_date.gte": expiration_gte,
            "limit": 100
        })

        if data and data.get("results"):
            return data["results"]

        return None

    def calculate_put_call_ratio(self, symbol: str = "SPY") -> Optional[Dict]:
        """
        Calculate put/call ratio from options volume

        Note: Requires Options tier subscription
        """
        try:
            contracts = self.get_options_chain(symbol)
            if not contracts:
                return None

            calls = [c for c in contracts if c.get("contract_type") == "call"]
            puts = [c for c in contracts if c.get("contract_type") == "put"]

            # This is contract count, not volume
            # For volume, would need to query each contract's trades
            call_count = len(calls)
            put_count = len(puts)

            if call_count == 0:
                return None

            pcr = put_count / call_count

            return {
                "symbol": symbol,
                "put_count": put_count,
                "call_count": call_count,
                "put_call_ratio": pcr,
                "interpretation": self._interpret_pcr(pcr),
            }

        except Exception as e:
            logger.error(f"Error calculating PCR: {e}")
            return None

    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret put/call ratio"""
        if pcr >= 1.2:
            return "HIGH_FEAR - Contrarian bullish"
        elif pcr >= 0.9:
            return "ELEVATED_FEAR"
        elif pcr <= 0.5:
            return "COMPLACENT - Contrarian bearish"
        elif pcr <= 0.7:
            return "LOW_FEAR"
        else:
            return "NEUTRAL"

    def get_market_status(self) -> Optional[Dict]:
        """Check if market is open"""
        endpoint = "/v1/marketstatus/now"
        data = self._request(endpoint)

        if data:
            return {
                "market": data.get("market"),
                "exchanges": data.get("exchanges", {}),
                "server_time": data.get("serverTime"),
                "is_market_open": data.get("market") == "open",
            }
        return None

    def get_trades(
        self,
        symbol: str = "SPY",
        timestamp_gte: int = None,
        timestamp_lte: int = None,
        limit: int = 5000,
        order: str = "asc"
    ) -> Optional[List[Dict]]:
        """
        Get tick-level trades for CVD calculation.

        Args:
            symbol: Stock symbol
            timestamp_gte: Start timestamp in nanoseconds (Unix epoch * 1e9)
            timestamp_lte: End timestamp in nanoseconds
            limit: Max number of trades (max 50000)
            order: "asc" or "desc"

        Returns:
            List of trade dicts with: price, size, timestamp, conditions
        """
        endpoint = f"/v3/trades/{symbol}"
        params = {
            "limit": min(limit, 50000),
            "order": order,
        }

        if timestamp_gte:
            params["timestamp.gte"] = timestamp_gte
        if timestamp_lte:
            params["timestamp.lte"] = timestamp_lte

        data = self._request(endpoint, params)

        if data and data.get("results"):
            return [{
                "price": r.get("price"),
                "size": r.get("size"),
                "timestamp": r.get("sip_timestamp"),
                "conditions": r.get("conditions", []),
                "exchange": r.get("exchange"),
            } for r in data["results"]]

        return None

    def get_nbbo_quote(self, symbol: str = "SPY") -> Optional[Dict]:
        """
        Get current NBBO (National Best Bid/Offer) for trade classification.

        Returns:
            Dict with bid, ask, bid_size, ask_size
        """
        endpoint = f"/v3/quotes/{symbol}"
        data = self._request(endpoint, {"limit": 1, "order": "desc"})

        if data and data.get("results") and len(data["results"]) > 0:
            quote = data["results"][0]
            return {
                "bid": quote.get("bid_price"),
                "ask": quote.get("ask_price"),
                "bid_size": quote.get("bid_size"),
                "ask_size": quote.get("ask_size"),
                "timestamp": quote.get("sip_timestamp"),
            }
        return None

    def classify_trade(
        self,
        trade: Dict,
        bid: float = None,
        ask: float = None
    ) -> str:
        """
        Classify a trade as 'buy', 'sell', or 'neutral'.

        Classification logic:
        1. Check trade conditions for explicit buy/sell indicators
        2. If neutral conditions, use price vs bid/ask midpoint
           - Price >= ask: likely buy (aggressive buyer)
           - Price <= bid: likely sell (aggressive seller)
           - Price near mid: neutral

        Args:
            trade: Trade dict with price, conditions
            bid: Current bid price
            ask: Current ask price

        Returns:
            'buy', 'sell', or 'neutral'
        """
        conditions = trade.get("conditions", [])
        price = trade.get("price", 0)

        # Check explicit conditions first
        for cond in conditions:
            if cond in BUY_CONDITIONS:
                return "buy"
            if cond in SELL_CONDITIONS:
                return "sell"

        # Use price vs bid/ask if available
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread = ask - bid

            # If spread is very tight, need price at/above ask for buy
            if spread < 0.02:  # Tight spread
                if price >= ask:
                    return "buy"
                elif price <= bid:
                    return "sell"
            else:
                # Wider spread - use position relative to mid
                if price >= mid + (spread * 0.25):
                    return "buy"
                elif price <= mid - (spread * 0.25):
                    return "sell"

        return "neutral"

    def calculate_cvd(
        self,
        symbol: str = "SPY",
        lookback_minutes: int = 30,
        bar_size_minutes: int = 5
    ) -> Optional[Dict]:
        """
        Calculate Cumulative Volume Delta (CVD) for a symbol.

        CVD = Cumulative sum of (buy_volume - sell_volume)

        A rising CVD indicates buying pressure.
        A falling CVD indicates selling pressure.

        Args:
            symbol: Stock symbol
            lookback_minutes: How far back to look (default 30 min)
            bar_size_minutes: Bar size for aggregation (default 5 min)

        Returns:
            Dict with:
            - cvd_values: List of CVD values per bar
            - cvd_slope: Linear regression slope of CVD
            - total_buy_volume: Total buy-classified volume
            - total_sell_volume: Total sell-classified volume
            - delta: Net delta (buy - sell)
            - bars: Number of bars
        """
        if not self.api_key:
            logger.warning("Polygon API key required for CVD calculation")
            return None

        try:
            # Calculate timestamp range (nanoseconds)
            now = datetime.utcnow()
            start_time = now - timedelta(minutes=lookback_minutes)

            # Convert to nanoseconds (Unix timestamp * 1e9)
            timestamp_gte = int(start_time.timestamp() * 1e9)
            timestamp_lte = int(now.timestamp() * 1e9)

            # Get trades
            trades = self.get_trades(
                symbol=symbol,
                timestamp_gte=timestamp_gte,
                timestamp_lte=timestamp_lte,
                limit=50000
            )

            if not trades or len(trades) < 10:
                logger.warning(f"Insufficient trades for CVD: {len(trades) if trades else 0}")
                return None

            # Get current quote for trade classification
            quote = self.get_nbbo_quote(symbol)
            bid = quote.get("bid") if quote else None
            ask = quote.get("ask") if quote else None

            # Aggregate into bars
            bar_ms = bar_size_minutes * 60 * 1000  # milliseconds
            bars = {}

            for trade in trades:
                ts = trade.get("timestamp", 0)
                if ts == 0:
                    continue

                # Convert nanoseconds to milliseconds and get bar key
                ts_ms = ts // 1_000_000
                bar_key = (ts_ms // bar_ms) * bar_ms

                if bar_key not in bars:
                    bars[bar_key] = {"buy_vol": 0, "sell_vol": 0, "neutral_vol": 0}

                # Classify trade
                side = self.classify_trade(trade, bid, ask)
                size = trade.get("size", 0)

                if side == "buy":
                    bars[bar_key]["buy_vol"] += size
                elif side == "sell":
                    bars[bar_key]["sell_vol"] += size
                else:
                    bars[bar_key]["neutral_vol"] += size

            if len(bars) < 2:
                logger.warning("Not enough bars for CVD calculation")
                return None

            # Sort bars by timestamp
            sorted_bars = sorted(bars.items())

            # Calculate CVD for each bar
            cvd_values = []
            cumulative = 0
            total_buy = 0
            total_sell = 0

            for bar_ts, bar_data in sorted_bars:
                delta = bar_data["buy_vol"] - bar_data["sell_vol"]
                cumulative += delta
                cvd_values.append(cumulative)
                total_buy += bar_data["buy_vol"]
                total_sell += bar_data["sell_vol"]

            # Calculate slope using numpy linear regression
            if len(cvd_values) >= 2:
                x = np.arange(len(cvd_values))
                slope, intercept = np.polyfit(x, cvd_values, 1)
            else:
                slope = 0

            # Normalize slope by average volume for comparability
            avg_volume = (total_buy + total_sell) / max(len(sorted_bars), 1)
            normalized_slope = slope / avg_volume if avg_volume > 0 else 0

            return {
                "cvd_values": cvd_values,
                "cvd_current": cvd_values[-1] if cvd_values else 0,
                "cvd_slope": round(slope, 2),
                "cvd_slope_normalized": round(normalized_slope, 4),
                "total_buy_volume": int(total_buy),
                "total_sell_volume": int(total_sell),
                "delta": int(total_buy - total_sell),
                "buy_sell_ratio": round(total_buy / total_sell, 2) if total_sell > 0 else None,
                "bars": len(sorted_bars),
                "trades_analyzed": len(trades),
                "lookback_minutes": lookback_minutes,
                "interpretation": self._interpret_cvd(normalized_slope, total_buy, total_sell),
            }

        except Exception as e:
            logger.error(f"Error calculating CVD: {e}")
            return None

    def _interpret_cvd(
        self,
        normalized_slope: float,
        buy_vol: int,
        sell_vol: int
    ) -> str:
        """
        Interpret CVD results.

        Args:
            normalized_slope: CVD slope normalized by volume
            buy_vol: Total buy volume
            sell_vol: Total sell volume

        Returns:
            Human-readable interpretation
        """
        # Slope interpretation
        if normalized_slope > 0.01:
            slope_interp = "STRONG_BUYING"
        elif normalized_slope > 0.005:
            slope_interp = "MODERATE_BUYING"
        elif normalized_slope > 0:
            slope_interp = "SLIGHT_BUYING"
        elif normalized_slope > -0.005:
            slope_interp = "SLIGHT_SELLING"
        elif normalized_slope > -0.01:
            slope_interp = "MODERATE_SELLING"
        else:
            slope_interp = "STRONG_SELLING"

        # Volume imbalance
        if buy_vol > 0 and sell_vol > 0:
            ratio = buy_vol / sell_vol
            if ratio > 1.2:
                vol_interp = "buyers dominant"
            elif ratio < 0.8:
                vol_interp = "sellers dominant"
            else:
                vol_interp = "balanced"
        else:
            vol_interp = "unknown"

        return f"{slope_interp} ({vol_interp})"

    def get_cvd_slope(self, symbol: str = "SPY", periods: int = 5) -> Optional[float]:
        """
        Get just the CVD slope value (convenience method).

        Args:
            symbol: Stock symbol
            periods: Number of 5-min periods to analyze

        Returns:
            Normalized CVD slope, or None if unavailable
        """
        cvd_data = self.calculate_cvd(
            symbol=symbol,
            lookback_minutes=periods * 5,
            bar_size_minutes=5
        )

        if cvd_data:
            return cvd_data.get("cvd_slope_normalized")
        return None

    def get_all_spy_data(self) -> Dict:
        """
        Get comprehensive SPY data from Polygon

        Returns all available real-time data
        """
        logger.info("Fetching Polygon data for SPY...")

        snapshot = self.get_snapshot("SPY")
        prev_close = self.get_previous_close("SPY")
        market_status = self.get_market_status()

        # Calculate IBS from today's data if available
        ibs = None
        if snapshot and snapshot.get("today"):
            today = snapshot["today"]
            if today.get("high") and today.get("low") and today.get("close"):
                high_low = today["high"] - today["low"]
                if high_low > 0:
                    ibs = (today["close"] - today["low"]) / high_low

        return {
            "symbol": "SPY",
            "source": "polygon",
            "timestamp": datetime.now().isoformat(),
            "snapshot": snapshot,
            "previous_close": prev_close,
            "market_status": market_status,
            "calculated": {
                "ibs_today": ibs,
            },
            "has_data": snapshot is not None,
        }


def test_polygon_collector():
    """Test the Polygon collector"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    collector = PolygonCollector()

    print("\n" + "=" * 60)
    print("POLYGON.IO COLLECTOR TEST")
    print("=" * 60)

    if not collector.api_key:
        print("\n⚠️  No API key configured!")
        print("Set POLYGON_API_KEY in .env file")
        print("Get free key at: https://polygon.io/")
        return None

    # Test market status
    print("\n--- Market Status ---")
    status = collector.get_market_status()
    if status:
        print(f"  Market: {status['market']}")
        print(f"  Is Open: {status['is_market_open']}")

    # Test snapshot
    print("\n--- SPY Snapshot ---")
    snapshot = collector.get_snapshot("SPY")
    if snapshot:
        current = snapshot.get("current", {})
        today = snapshot.get("today", {})
        change = snapshot.get("change", {})
        print(f"  Price: ${current.get('price', 'N/A')}")
        print(f"  Bid/Ask: ${current.get('bid', 'N/A')} / ${current.get('ask', 'N/A')}")
        print(f"  Today VWAP: ${today.get('vwap', 'N/A')}")
        print(f"  Change: {change.get('today_change_percent', 'N/A')}%")
    else:
        print("  No snapshot data (market may be closed or API issue)")

    # Test previous close
    print("\n--- Previous Close ---")
    prev = collector.get_previous_close("SPY")
    if prev:
        print(f"  Close: ${prev['close']}")
        print(f"  VWAP: ${prev['vwap']}")
        print(f"  Volume: {prev['volume']:,}")

    # Test comprehensive data
    print("\n--- Comprehensive SPY Data ---")
    all_data = collector.get_all_spy_data()
    print(f"  Has Data: {all_data['has_data']}")
    if all_data['calculated']['ibs_today']:
        print(f"  Today's IBS: {all_data['calculated']['ibs_today']:.3f}")

    return all_data


if __name__ == "__main__":
    test_polygon_collector()
