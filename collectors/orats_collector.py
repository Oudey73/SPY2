"""
ORATS Data Collector - Professional-grade IV analytics for SPY options trading
"""
import os
import requests
from datetime import datetime
from typing import Dict, Optional
from loguru import logger
from dotenv import load_dotenv
import pytz

load_dotenv()
ORATS_BASE_URL = "https://api.orats.io/datav2"

# Saudi Arabia timezone (AST / UTC+3)
SAUDI_TZ = pytz.timezone('Asia/Riyadh')

def now_saudi() -> str:
    """Get current time in Saudi Arabia"""
    return datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")

class ORATSCollector:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ORATS_API_KEY", "")
        self.base_url = ORATS_BASE_URL
        if not self.api_key:
            logger.warning("ORATS API key not configured")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        if not self.is_configured():
            return None
        try:
            url = f"{self.base_url}/{endpoint}"
            params = params or {}
            params["token"] = self.api_key
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data)
        except requests.exceptions.RequestException as e:
            logger.error(f"ORATS API error: {e}")
            return None

    def get_summaries(self, symbol: str = "SPY") -> Optional[Dict]:
        data = self._make_request("summaries", {"ticker": symbol})
        if data and len(data) > 0:
            return data[0] if isinstance(data, list) else data
        return None

    def get_iv_rank(self, symbol: str = "SPY") -> Optional[Dict]:
        summary = self.get_summaries(symbol)
        if not summary:
            return None
        iv_percentile = summary.get("rip", 50)
        iv_rank = iv_percentile
        current_iv = summary.get("iv30d", 0) * 100
        stock_price = summary.get("stockPrice", 0)
        trade_date = summary.get("tradeDate", "unknown")
        if iv_rank >= 75:
            interp = "EXTREME_HIGH - Premium expensive, SELL bias"
        elif iv_rank >= 50:
            interp = "ELEVATED - Above average, slight SELL bias"
        elif iv_rank >= 25:
            interp = "NORMAL - Fair premium, no strong bias"
        else:
            interp = "LOW - Cheap premium, BUY bias"
        return {"symbol": symbol, "iv_rank": iv_rank, "iv_percentile": iv_percentile, "current_iv": current_iv, "stock_price": stock_price, "interpretation": interp, "trade_date": trade_date, "timestamp": now_saudi()}

    def get_term_structure(self, symbol: str = "SPY") -> Optional[Dict]:
        summary = self.get_summaries(symbol)
        if not summary:
            return None
        iv_10d = summary.get("iv10d", 0) * 100
        iv_30d = summary.get("iv30d", 0) * 100
        iv_60d = summary.get("iv60d", 0) * 100
        iv_90d = summary.get("iv90d", 0) * 100
        contango = summary.get("contango", 0)
        spread = iv_90d - iv_30d
        if spread > 2 or contango > 0.3:
            struct, interp = "CONTANGO", "Normal market - Sell front-month premium"
        elif spread < -2 or contango < -0.3:
            struct, interp = "BACKWARDATION", "STRESS SIGNAL - Near-term fear elevated"
        else:
            struct, interp = "FLAT", "Neutral term structure"
        return {"symbol": symbol, "structure": struct, "iv_10d": iv_10d, "iv_30d": iv_30d, "iv_60d": iv_60d, "iv_90d": iv_90d, "spread": spread, "contango": contango, "interpretation": interp, "timestamp": now_saudi()}

    def get_skew(self, symbol: str = "SPY") -> Optional[Dict]:
        summary = self.get_summaries(symbol)
        if not summary:
            return None
        skewing = summary.get("skewing", 0)
        put_iv_25d = summary.get("dlt25Iv30d", 0) * 100
        call_iv_75d = summary.get("dlt75Iv30d", 0) * 100
        atm_iv = summary.get("iv30d", 0) * 100
        put_skew = put_iv_25d - atm_iv if put_iv_25d > 0 else abs(skewing) * 100
        if skewing < -0.1 or put_skew > 5:
            stype, interp = "STEEP_PUT", "High hedging demand - Consider selling put spreads"
        elif skewing < -0.05 or put_skew > 2:
            stype, interp = "NORMAL", "Normal SPY skew - Standard conditions"
        elif skewing < 0 or put_skew > 0:
            stype, interp = "FLAT", "Unusually flat skew - Complacency signal"
        else:
            stype, interp = "CALL_SKEW", "Rare call skew - Unusual bullish positioning"
        return {"symbol": symbol, "skew_type": stype, "put_iv_25d": put_iv_25d, "call_iv_75d": call_iv_75d, "atm_iv": atm_iv, "put_skew": put_skew, "skewing": skewing, "interpretation": interp, "timestamp": now_saudi()}

    def get_all_iv_data(self, symbol: str = "SPY") -> Optional[Dict]:
        iv_rank = self.get_iv_rank(symbol)
        term_structure = self.get_term_structure(symbol)
        skew = self.get_skew(symbol)
        if not iv_rank:
            return None
        iv_val = iv_rank.get("iv_rank", 50)
        struct = term_structure.get("structure", "FLAT") if term_structure else "UNKNOWN"
        if iv_val >= 70:
            rec = "HIGH_IV_STRESS - Sell premium cautiously" if struct == "BACKWARDATION" else "HIGH_IV - Favor selling premium"
        elif iv_val >= 40:
            rec = "NEUTRAL_IV - Both strategies viable"
        else:
            rec = "LOW_IV - Favor buying premium"
        return {"symbol": symbol, "iv_rank": iv_rank, "term_structure": term_structure, "skew": skew, "strategy_recommendation": rec, "timestamp": now_saudi()}

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("ORATS COLLECTOR TEST")
    c = ORATSCollector()
    if not c.is_configured():
        print("No API key!")
    else:
        print(f"API: {c.api_key[:8]}...")
        iv = c.get_iv_rank("SPY")
        if iv:
            print(f"Date: {iv['trade_date']}, Price: ${iv['stock_price']}")
            print(f"IV Rank: {iv['iv_rank']:.1f}, 30d IV: {iv['current_iv']:.1f}%")
            print(iv["interpretation"])
        ts = c.get_term_structure("SPY")
        if ts:
            print(f"Term: {ts['structure']} - Spread: {ts['spread']:.2f}")
        sk = c.get_skew("SPY")
        if sk:
            print(f"Skew: {sk['skew_type']} ({sk['skewing']:.4f})")
        full = c.get_all_iv_data("SPY")
        if full:
            print(f"Strategy: {full['strategy_recommendation']}")

