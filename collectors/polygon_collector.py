"""
Polygon.io Data Collector
Fetches: Real-time quotes, options data, aggregates
Requires API key ($29/mo for real-time, free tier is delayed)
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

POLYGON_BASE = "https://api.polygon.io"


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
