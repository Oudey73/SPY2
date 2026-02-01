"""
Yahoo Finance Data Collector
Fetches: SPY price/OHLCV, VIX, historical data for technicals
Free, no API key required
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


class YahooCollector:
    """Collects market data from Yahoo Finance"""

    def __init__(self):
        self.spy = yf.Ticker("SPY")
        self.vix = yf.Ticker("^VIX")

    def get_current_price(self, symbol: str = "SPY") -> Optional[Dict]:
        """Get current price data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get today's data
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                # Market might be closed, get last available
                hist = ticker.history(period="5d")

            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            latest = hist.iloc[-1]

            return {
                "symbol": symbol,
                "price": float(latest["Close"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "timestamp": str(latest.name),
            }
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_daily_ohlcv(self, symbol: str = "SPY", days: int = 100) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV data for technical analysis

        Returns DataFrame with: Open, High, Low, Close, Volume
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")

            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            return hist[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol: str = "SPY", interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Get intraday data for momentum analysis

        Args:
            symbol: Stock symbol
            interval: 1m, 5m, 15m, 30m, 1h

        Returns DataFrame with OHLCV
        """
        try:
            ticker = yf.Ticker(symbol)
            # Yahoo allows max 7 days of intraday data
            hist = ticker.history(period="5d", interval=interval)

            if hist.empty:
                logger.warning(f"No intraday data for {symbol}")
                return None

            return hist[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def calculate_ibs(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Internal Bar Strength (IBS)

        IBS = (Close - Low) / (High - Low)

        IBS < 0.2 = Oversold (buy signal)
        IBS > 0.8 = Overbought (sell signal)
        """
        high_low_range = df["High"] - df["Low"]
        # Avoid division by zero
        high_low_range = high_low_range.replace(0, np.nan)

        ibs = (df["Close"] - df["Low"]) / high_low_range
        return ibs

    def calculate_rsi(self, df: pd.DataFrame, period: int = 3) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)

        Using RSI(3) for mean reversion as per research
        """
        delta = df["Close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df["Close"].rolling(window=period).mean()

    def calculate_consecutive_days(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Calculate consecutive up and down days

        Returns: (consecutive_up_days, consecutive_down_days)
        """
        if len(df) < 2:
            return 0, 0

        # Calculate daily returns
        df = df.copy()
        df['up'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['down'] = (df['Close'] < df['Close'].shift(1)).astype(int)

        # Count consecutive up days (from most recent)
        consecutive_up = 0
        for i in range(len(df) - 1, 0, -1):
            if df['up'].iloc[i] == 1:
                consecutive_up += 1
            else:
                break

        # Count consecutive down days (from most recent)
        consecutive_down = 0
        for i in range(len(df) - 1, 0, -1):
            if df['down'].iloc[i] == 1:
                consecutive_down += 1
            else:
                break

        return consecutive_up, consecutive_down

    def get_vix_data(self) -> Optional[Dict]:
        """Get current VIX data and its moving average"""
        try:
            # Get VIX history for MA calculation
            hist = self.vix.history(period="30d")

            if hist.empty:
                logger.warning("No VIX data available")
                return None

            current_vix = float(hist["Close"].iloc[-1])
            vix_10_ma = float(hist["Close"].rolling(10).mean().iloc[-1])
            vix_20_ma = float(hist["Close"].rolling(20).mean().iloc[-1])

            return {
                "vix": current_vix,
                "vix_10_ma": vix_10_ma,
                "vix_20_ma": vix_20_ma,
                "vix_above_10ma": current_vix > vix_10_ma,
                "vix_above_20ma": current_vix > vix_20_ma,
                "vix_level": self._interpret_vix(current_vix),
                "timestamp": str(hist.index[-1]),
            }

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return None

    def _interpret_vix(self, vix: float) -> str:
        """Interpret VIX level"""
        if vix >= 30:
            return "EXTREME_FEAR"
        elif vix >= 25:
            return "HIGH_FEAR"
        elif vix >= 20:
            return "ELEVATED"
        elif vix >= 15:
            return "NORMAL"
        else:
            return "COMPLACENT"

    def get_spy_technicals(self) -> Optional[Dict]:
        """
        Get SPY with all technical indicators calculated

        Returns comprehensive technical snapshot
        """
        try:
            # Get 100 days of data for indicator calculation
            df = self.get_daily_ohlcv("SPY", days=100)

            if df is None or len(df) < 50:
                logger.warning("Insufficient data for technical analysis")
                return None

            # Calculate indicators
            df["IBS"] = self.calculate_ibs(df)
            df["RSI_2"] = self.calculate_rsi(df, period=2)  # NEW: RSI(2) for PUT signals
            df["RSI_3"] = self.calculate_rsi(df, period=3)
            df["RSI_14"] = self.calculate_rsi(df, period=14)
            df["SMA_10"] = self.calculate_sma(df, 10)
            df["SMA_20"] = self.calculate_sma(df, 20)
            df["SMA_50"] = self.calculate_sma(df, 50)
            df["SMA_200"] = self.calculate_sma(df, 200) if len(df) >= 200 else None

            # Calculate consecutive up/down days (NEW for PUT signals)
            consecutive_up, consecutive_down = self.calculate_consecutive_days(df)

            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Get VIX
            vix_data = self.get_vix_data()

            return {
                "symbol": "SPY",
                "timestamp": str(df.index[-1]),
                "price": {
                    "close": float(latest["Close"]),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "volume": int(latest["Volume"]),
                    "prev_close": float(prev["Close"]),
                    "change_percent": ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100,
                },
                "ibs": {
                    "current": float(latest["IBS"]) if pd.notna(latest["IBS"]) else None,
                    "previous": float(prev["IBS"]) if pd.notna(prev["IBS"]) else None,
                    "is_oversold": latest["IBS"] < 0.2 if pd.notna(latest["IBS"]) else False,
                    "is_overbought": latest["IBS"] > 0.8 if pd.notna(latest["IBS"]) else False,
                },
                "rsi": {
                    "rsi_2": float(latest["RSI_2"]) if pd.notna(latest["RSI_2"]) else None,  # NEW
                    "rsi_3": float(latest["RSI_3"]) if pd.notna(latest["RSI_3"]) else None,
                    "rsi_14": float(latest["RSI_14"]) if pd.notna(latest["RSI_14"]) else None,
                    "is_oversold": latest["RSI_3"] < 20 if pd.notna(latest["RSI_3"]) else False,
                    "is_overbought": latest["RSI_3"] > 80 if pd.notna(latest["RSI_3"]) else False,
                    "rsi2_overbought": latest["RSI_2"] >= 95 if pd.notna(latest["RSI_2"]) else False,  # NEW
                },
                "consecutive_days": {  # NEW section for PUT signals
                    "up": consecutive_up,
                    "down": consecutive_down,
                    "extended_rally": consecutive_up >= 3,
                    "extended_selloff": consecutive_down >= 3,
                },
                "moving_averages": {
                    "sma_10": float(latest["SMA_10"]) if pd.notna(latest["SMA_10"]) else None,
                    "sma_20": float(latest["SMA_20"]) if pd.notna(latest["SMA_20"]) else None,
                    "sma_50": float(latest["SMA_50"]) if pd.notna(latest["SMA_50"]) else None,
                    "sma_200": float(latest["SMA_200"]) if latest["SMA_200"] is not None and pd.notna(latest["SMA_200"]) else None,
                    "above_50_ma": latest["Close"] > latest["SMA_50"] if pd.notna(latest["SMA_50"]) else None,
                    "above_200_ma": latest["Close"] > latest["SMA_200"] if latest["SMA_200"] is not None and pd.notna(latest["SMA_200"]) else None,
                },
                "vix": vix_data,
                "data_quality": "OK" if pd.notna(latest["IBS"]) and pd.notna(latest["RSI_3"]) else "INCOMPLETE",
            }

        except Exception as e:
            logger.error(f"Error calculating SPY technicals: {e}")
            return None

    def get_intraday_momentum(self) -> Optional[Dict]:
        """
        Calculate intraday momentum for last 30-minute signal

        Based on research: Return in last 30 min correlates with earlier return
        """
        try:
            # Get 5-minute data
            df = self.get_intraday_data("SPY", interval="5m")

            if df is None or len(df) < 10:
                return None

            # Get today's data only
            today = datetime.now().date()
            today_df = df[df.index.date == today]

            if len(today_df) < 6:  # Need at least 30 min of data
                # Market might not be open yet, use yesterday
                yesterday = today - timedelta(days=1)
                today_df = df[df.index.date == yesterday]

            if len(today_df) < 6:
                return None

            # Calculate returns
            open_price = today_df["Open"].iloc[0]
            current_price = today_df["Close"].iloc[-1]

            # Last 30 minutes (6 x 5-min bars)
            if len(today_df) >= 6:
                price_30min_ago = today_df["Close"].iloc[-7] if len(today_df) >= 7 else today_df["Open"].iloc[0]
                last_30min_return = ((current_price - price_30min_ago) / price_30min_ago) * 100
            else:
                last_30min_return = 0

            # Earlier day return (open to 30 min before close)
            if len(today_df) >= 7:
                earlier_close = today_df["Close"].iloc[-7]
                earlier_return = ((earlier_close - open_price) / open_price) * 100
            else:
                earlier_return = 0

            full_day_return = ((current_price - open_price) / open_price) * 100

            return {
                "timestamp": str(today_df.index[-1]),
                "open_price": float(open_price),
                "current_price": float(current_price),
                "full_day_return_pct": float(full_day_return),
                "earlier_return_pct": float(earlier_return),
                "last_30min_return_pct": float(last_30min_return),
                # Intraday momentum signal: if day is up, last 30 min tends to continue
                "momentum_signal": "BULLISH" if earlier_return > 0.3 else "BEARISH" if earlier_return < -0.3 else "NEUTRAL",
            }

        except Exception as e:
            logger.error(f"Error calculating intraday momentum: {e}")
            return None


def test_yahoo_collector():
    """Test the Yahoo Finance collector"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    collector = YahooCollector()

    print("\n" + "=" * 60)
    print("YAHOO FINANCE COLLECTOR TEST")
    print("=" * 60)

    # Test current price
    print("\n--- Current SPY Price ---")
    price = collector.get_current_price("SPY")
    if price:
        print(f"  Price: ${price['price']:.2f}")
        print(f"  High: ${price['high']:.2f}")
        print(f"  Low: ${price['low']:.2f}")
        print(f"  Volume: {price['volume']:,}")

    # Test VIX
    print("\n--- VIX Data ---")
    vix = collector.get_vix_data()
    if vix:
        print(f"  VIX: {vix['vix']:.2f}")
        print(f"  10-day MA: {vix['vix_10_ma']:.2f}")
        print(f"  Above 10-MA: {vix['vix_above_10ma']}")
        print(f"  Level: {vix['vix_level']}")

    # Test full technicals
    print("\n--- SPY Technicals ---")
    technicals = collector.get_spy_technicals()
    if technicals:
        print(f"  Price: ${technicals['price']['close']:.2f}")
        print(f"  Change: {technicals['price']['change_percent']:.2f}%")
        print(f"  IBS: {technicals['ibs']['current']:.3f}" if technicals['ibs']['current'] else "  IBS: N/A")
        print(f"  RSI(3): {technicals['rsi']['rsi_3']:.1f}" if technicals['rsi']['rsi_3'] else "  RSI(3): N/A")
        print(f"  IBS Oversold: {technicals['ibs']['is_oversold']}")
        print(f"  RSI Oversold: {technicals['rsi']['is_oversold']}")
        print(f"  Data Quality: {technicals['data_quality']}")

    # Test intraday momentum
    print("\n--- Intraday Momentum ---")
    momentum = collector.get_intraday_momentum()
    if momentum:
        print(f"  Day Return: {momentum['full_day_return_pct']:.2f}%")
        print(f"  Last 30min: {momentum['last_30min_return_pct']:.2f}%")
        print(f"  Signal: {momentum['momentum_signal']}")

    return technicals


if __name__ == "__main__":
    test_yahoo_collector()
