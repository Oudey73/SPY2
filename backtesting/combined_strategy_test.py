"""
Combined CALL + PUT Strategy Performance Test
Tests the complete trading system with both directions
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class CombinedStrategyTest:
    """Test both CALL and PUT strategies together"""

    def __init__(self, start_date: str = "2022-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.spy_data = None

    def load_data(self) -> bool:
        """Load data"""
        print(f"Loading data from {self.start_date} to {self.end_date}...")
        try:
            spy = yf.Ticker("SPY")
            self.spy_data = spy.history(start=self.start_date, end=self.end_date)

            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=self.start_date, end=self.end_date)

            if self.spy_data.empty:
                return False

            self._calculate_indicators(vix_data)
            print(f"Loaded {len(self.spy_data)} days")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def _calculate_indicators(self, vix_data):
        """Calculate all indicators"""
        df = self.spy_data

        # IBS
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['IBS'] = df['IBS'].replace([np.inf, -np.inf], np.nan).fillna(0.5)

        # RSI calculations
        delta = df['Close'].diff()
        for period in [2, 3]:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'] = df[f'RSI_{period}'].fillna(50)

        # MAs
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Returns
        df['Return_3D'] = df['Close'].pct_change(3) * 100

        # Consecutive days
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['Down_Day'] = (df['Close'] < df['Close'].shift(1)).astype(int)
        df['Consec_Up'] = df['Up_Day'].rolling(window=5).sum()
        df['Consec_Down'] = df['Down_Day'].rolling(window=5).sum()

        # Day of week
        df['DayOfWeek'] = df.index.dayofweek

        # VIX
        df['VIX'] = vix_data['Close'].reindex(df.index, method='ffill')
        df['VIX_10_MA'] = vix_data['Close'].rolling(10).mean().reindex(df.index, method='ffill')

        # Above 200MA
        df['Above_200MA'] = df['Close'] > df['SMA_200']

        self.spy_data = df

    def score_call_signal(self, row) -> int:
        """Score CALL (long) signal - based on your existing strategy"""
        score = 0

        # IBS oversold (key driver for CALL)
        if row['IBS'] <= 0.05:
            score += 30
        elif row['IBS'] <= 0.10:
            score += 25
        elif row['IBS'] <= 0.15:
            score += 20
        elif row['IBS'] <= 0.20:
            score += 15

        # RSI(3) oversold
        if row['RSI_3'] <= 5:
            score += 25
        elif row['RSI_3'] <= 10:
            score += 20
        elif row['RSI_3'] <= 20:
            score += 15
        elif row['RSI_3'] <= 30:
            score += 10

        # VIX elevated (better for mean reversion)
        if row['VIX'] >= 25:
            score += 15
        elif row['VIX'] >= 20:
            score += 10
        elif row['VIX'] >= 18:
            score += 5

        # Above 200MA (bullish regime)
        if row['Above_200MA']:
            score += 5

        # Consecutive down days
        if row['Consec_Down'] >= 4:
            score += 10
        elif row['Consec_Down'] >= 3:
            score += 5

        # Strong selloff
        if row['Return_3D'] < -3:
            score += 10

        return score

    def score_put_signal(self, row) -> int:
        """Score PUT (short) signal"""
        score = 0

        # RSI(2) overbought (key driver for PUT)
        if row['RSI_2'] >= 98:
            score += 30
        elif row['RSI_2'] >= 95:
            score += 25
        elif row['RSI_2'] >= 90:
            score += 15

        # IBS overbought
        if row['IBS'] >= 0.95:
            score += 20
        elif row['IBS'] >= 0.90:
            score += 15
        elif row['IBS'] >= 0.85:
            score += 10

        # RSI(3) overbought
        if row['RSI_3'] >= 95:
            score += 15
        elif row['RSI_3'] >= 90:
            score += 10

        # Above upper Bollinger Band
        if row['BB_PctB'] > 1.0:
            score += 10

        # Consecutive up days
        if row['Consec_Up'] >= 4:
            score += 10
        elif row['Consec_Up'] >= 3:
            score += 5

        # Strong rally
        if row['Return_3D'] > 3:
            score += 10

        # VIX complacency
        if row['VIX'] < 14:
            score += 5

        return score

    def run_combined_backtest(self):
        """Run backtest with both CALL and PUT signals"""
        if not self.load_data():
            return

        df = self.spy_data
        all_trades = []
        last_exit_idx = -999

        # Parameters
        call_min_score = 50  # From existing optimized CALL strategy
        put_min_score = 55   # From PUT optimization

        call_stop = 0.015    # 1.5%
        call_target = 0.02   # 2%
        call_max_hold = 5

        put_stop = 0.01      # 1%
        put_target = 0.02    # 2%
        put_max_hold = 3

        for i in range(200, len(df)):
            if i <= last_exit_idx:
                continue

            row = df.iloc[i]
            date = df.index[i]

            call_score = self.score_call_signal(row)
            put_score = self.score_put_signal(row)

            # Prioritize the stronger signal
            if call_score >= call_min_score and call_score > put_score:
                # CALL trade
                direction = "CALL"
                entry_price = row['Close']
                stop_price = entry_price * (1 - call_stop)
                target_price = entry_price * (1 + call_target)
                max_hold = call_max_hold
                score = call_score

            elif put_score >= put_min_score:
                # PUT trade
                direction = "PUT"
                entry_price = row['Close']
                stop_price = entry_price * (1 + put_stop)
                target_price = entry_price * (1 - put_target)
                max_hold = put_max_hold
                score = put_score

            else:
                continue

            # Simulate trade
            exit_price = entry_price
            exit_reason = "max_hold"
            hold_days = max_hold

            for j in range(1, max_hold + 1):
                if i + j >= len(df):
                    break

                future = df.iloc[i + j]

                if direction == "CALL":
                    if future['Low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        hold_days = j
                        break
                    if future['High'] >= target_price:
                        exit_price = target_price
                        exit_reason = "target"
                        hold_days = j
                        break
                else:  # PUT
                    if future['High'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        hold_days = j
                        break
                    if future['Low'] <= target_price:
                        exit_price = target_price
                        exit_reason = "target"
                        hold_days = j
                        break

                exit_price = future['Close']
                hold_days = j

            last_exit_idx = i + hold_days

            # Calculate P&L
            if direction == "CALL":
                pnl = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl = ((entry_price - exit_price) / entry_price) * 100

            all_trades.append({
                'date': str(date.date()),
                'direction': direction,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'score': score,
                'exit_reason': exit_reason,
                'hold': hold_days,
                'vix': row['VIX']
            })

        return all_trades

    def analyze_results(self, trades):
        """Analyze combined results"""
        print("\n" + "=" * 80)
        print("COMBINED CALL + PUT STRATEGY RESULTS")
        print("=" * 80)

        df = pd.DataFrame(trades)

        # Overall stats
        total = len(df)
        wins = len(df[df['pnl'] > 0])
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Trades: {total}")
        print(f"  Win Rate: {wins/total*100:.1f}%")
        print(f"  Avg P&L: {df['pnl'].mean():.3f}%")
        print(f"  Total P&L: {df['pnl'].sum():.2f}%")

        # By direction
        print(f"\nBY DIRECTION:")
        for direction in ['CALL', 'PUT']:
            ddf = df[df['direction'] == direction]
            if len(ddf) > 0:
                dwins = len(ddf[ddf['pnl'] > 0])
                print(f"  {direction}: {len(ddf)} trades, {dwins/len(ddf)*100:.1f}% win rate, "
                      f"Total P&L: {ddf['pnl'].sum():.2f}%")

        # By year
        df['year'] = pd.to_datetime(df['date']).dt.year
        print(f"\nBY YEAR:")
        for year in sorted(df['year'].unique()):
            ydf = df[df['year'] == year]
            ywins = len(ydf[ydf['pnl'] > 0])
            calls = len(ydf[ydf['direction'] == 'CALL'])
            puts = len(ydf[ydf['direction'] == 'PUT'])
            print(f"  {year}: {len(ydf)} trades ({calls} CALL, {puts} PUT), "
                  f"{ywins/len(ydf)*100:.1f}% win, P&L: {ydf['pnl'].sum():.2f}%")

        # Capital simulation
        print(f"\nCAPITAL SIMULATION ($1,000 start):")
        for leverage in [5, 8, 10, 12, 15]:
            capital = 1000
            for _, t in df.iterrows():
                size = capital * 0.25
                lev_pnl = min(max(t['pnl'] * leverage, -100), 100)
                capital += size * (lev_pnl / 100)
                if capital <= 0:
                    capital = 0
                    break
            print(f"  {leverage}x leverage: ${capital:,.2f} ({(capital-1000)/10:+.1f}%)")

        # Monthly breakdown for 2024-2025
        print(f"\nMONTHLY BREAKDOWN (2024-2025):")
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        recent = df[df['year'] >= 2024]
        for month in sorted(recent['month'].unique()):
            mdf = recent[recent['month'] == month]
            mwins = len(mdf[mdf['pnl'] > 0])
            print(f"  {month}: {len(mdf)} trades, {mwins}/{len(mdf)} wins, P&L: {mdf['pnl'].sum():.2f}%")

        # Recent trades
        print(f"\nRECENT TRADES (last 15):")
        print("-" * 90)
        for _, t in df.tail(15).iterrows():
            result = "WIN " if t['pnl'] > 0 else "LOSS"
            print(f"  {t['date']} {t['direction']:4s}: Score {t['score']}, "
                  f"${t['entry']:.2f} -> ${t['exit']:.2f} ({t['pnl']:+.2f}%) [{result}] {t['exit_reason']}")

        return df


def main():
    tester = CombinedStrategyTest(start_date="2022-01-01")
    trades = tester.run_combined_backtest()

    if trades:
        tester.analyze_results(trades)

        print("\n" + "=" * 80)
        print("FINAL STRATEGY SUMMARY")
        print("=" * 80)
        print("""
CALL STRATEGY (Long/Bullish):
  Trigger: Score >= 50 based on:
    - IBS <= 0.20 (oversold)
    - RSI(3) <= 30 (oversold)
    - VIX >= 18 (elevated fear)
    - 3+ consecutive down days
  Trade: Stop 1.5%, Target 2%, Max Hold 5 days
  Best with: VIX > 20 filter for higher conviction

PUT STRATEGY (Short/Bearish):
  Trigger: Score >= 55 based on:
    - RSI(2) >= 90 (overbought)
    - IBS >= 0.85 (overbought)
    - RSI(3) >= 90 (overbought)
    - 3+ consecutive up days
    - Above Bollinger Band
  Trade: Stop 1%, Target 2%, Max Hold 3 days
  Best with: VIX < 18 (complacency)

OPTIONS GUIDANCE:
  - CALL: ATM calls, 7-10 DTE, 8-12x leverage
  - PUT: ATM puts, 5-8 DTE, 8-12x leverage
  - Position size: 25% of capital per trade
  - Never overlap trades (wait for exit before new entry)
""")


if __name__ == "__main__":
    main()
