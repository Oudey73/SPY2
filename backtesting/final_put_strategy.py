"""
Final PUT Strategy - Optimized SHORT/PUT methodology for SPY
Tests the best configurations and produces final recommendations
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class FinalPutStrategy:
    """Final optimized PUT strategy"""

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
        """Calculate indicators"""
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

        # Consecutive up days
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['Consec_Up'] = df['Up_Day'].rolling(window=5).sum()

        # Day of week
        df['DayOfWeek'] = df.index.dayofweek

        # VIX
        df['VIX'] = vix_data['Close'].reindex(df.index, method='ffill')

        # Above 200MA
        df['Above_200MA'] = df['Close'] > df['SMA_200']

        self.spy_data = df

    def score_put_signal(self, row) -> Tuple[int, List[str]]:
        """Score a potential PUT signal"""
        score = 0
        reasons = []

        # RSI(2) - Most important
        if row['RSI_2'] >= 98:
            score += 30
            reasons.append("RSI2>=98")
        elif row['RSI_2'] >= 95:
            score += 25
            reasons.append("RSI2>=95")
        elif row['RSI_2'] >= 90:
            score += 15
            reasons.append("RSI2>=90")

        # IBS
        if row['IBS'] >= 0.95:
            score += 20
            reasons.append("IBS>=0.95")
        elif row['IBS'] >= 0.90:
            score += 15
            reasons.append("IBS>=0.90")
        elif row['IBS'] >= 0.85:
            score += 10
            reasons.append("IBS>=0.85")

        # RSI(3)
        if row['RSI_3'] >= 95:
            score += 15
            reasons.append("RSI3>=95")
        elif row['RSI_3'] >= 90:
            score += 10
            reasons.append("RSI3>=90")

        # Bollinger Band
        if row['BB_PctB'] > 1.0:
            score += 10
            reasons.append("BB>100%")

        # Consecutive up days
        if row['Consec_Up'] >= 4:
            score += 10
            reasons.append("4+UpDays")
        elif row['Consec_Up'] >= 3:
            score += 5
            reasons.append("3UpDays")

        # Strong rally
        if row['Return_3D'] > 3:
            score += 10
            reasons.append("Rally>3%")

        # VIX complacency
        if row['VIX'] < 14:
            score += 5
            reasons.append("VIX<14")

        # Friday
        if row['DayOfWeek'] == 4:
            score += 5
            reasons.append("Friday")

        # Below 200MA (bearish regime)
        if not row['Above_200MA']:
            score += 5
            reasons.append("Below200MA")

        return score, reasons

    def test_strategy(self, min_score: int, stop_pct: float, target_pct: float,
                      max_hold: int) -> Tuple[List, Dict]:
        """Test a specific strategy configuration"""
        df = self.spy_data
        trades = []
        last_exit_idx = -999

        for i in range(200, len(df)):
            if i <= last_exit_idx:
                continue

            row = df.iloc[i]
            score, reasons = self.score_put_signal(row)

            if score < min_score:
                continue

            # Entry
            entry_price = row['Close']
            date = df.index[i]

            stop_price = entry_price * (1 + stop_pct)
            target_price = entry_price * (1 - target_pct)

            exit_price = entry_price
            exit_reason = "max_hold"
            hold_days = max_hold

            for j in range(1, max_hold + 1):
                if i + j >= len(df):
                    break

                future = df.iloc[i + j]

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
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl_pct,
                'score': score,
                'reasons': reasons,
                'exit_reason': exit_reason,
                'hold': hold_days,
                'vix': row['VIX']
            })

        if not trades:
            return [], {'trades': 0}

        wins = sum(1 for t in trades if t['pnl'] > 0)
        stats = {
            'trades': len(trades),
            'wins': wins,
            'win_rate': wins / len(trades) * 100,
            'avg_pnl': sum(t['pnl'] for t in trades) / len(trades),
            'total_pnl': sum(t['pnl'] for t in trades)
        }

        return trades, stats

    def simulate(self, trades, capital=1000, leverage=8, pos_pct=0.25):
        """Simulate capital growth"""
        for t in trades:
            size = capital * pos_pct
            lev_pnl = min(t['pnl'] * leverage, 100)  # Cap gains
            lev_pnl = max(lev_pnl, -100)  # Cap losses
            capital += size * (lev_pnl / 100)
            if capital <= 0:
                return 0
        return capital

    def run_analysis(self):
        """Run full analysis"""
        if not self.load_data():
            return

        print("\n" + "=" * 80)
        print("FINAL PUT STRATEGY ANALYSIS")
        print("=" * 80)

        # Test configurations focusing on higher scores for better win rates
        configs = []

        for min_score in [50, 55, 60, 65, 70, 75, 80]:
            for stop in [0.008, 0.010, 0.012, 0.015]:
                for target in [0.015, 0.020, 0.025, 0.030, 0.035]:
                    for hold in [2, 3]:
                        trades, stats = self.test_strategy(min_score, stop, target, hold)

                        if stats['trades'] >= 15:
                            for lev in [8, 10, 12]:
                                final = self.simulate(trades, leverage=lev)
                                ret = (final - 1000) / 1000 * 100

                                configs.append({
                                    'score': min_score,
                                    'stop': stop,
                                    'target': target,
                                    'hold': hold,
                                    'lev': lev,
                                    'trades': stats['trades'],
                                    'win_rate': stats['win_rate'],
                                    'final': final,
                                    'return': ret
                                })

        # Sort by return
        configs.sort(key=lambda x: x['return'], reverse=True)

        print(f"\nTOP 20 PUT CONFIGURATIONS:")
        print("-" * 90)
        print(f"{'Score':<7} {'Stop':<6} {'Tgt':<6} {'Hold':<5} {'Lev':<5} {'#':<6} {'Win%':<8} {'$1K->':<12} {'Return'}")
        print("-" * 90)

        for c in configs[:20]:
            print(f"{c['score']:<7} {c['stop']*100:.1f}%  {c['target']*100:.1f}%  {c['hold']}d    "
                  f"{c['lev']}x   {c['trades']:<6} {c['win_rate']:.1f}%    "
                  f"${c['final']:,.0f}        {c['return']:+.1f}%")

        # Filter for high win rate
        high_wr = [c for c in configs if c['win_rate'] >= 50]
        high_wr.sort(key=lambda x: x['return'], reverse=True)

        print(f"\nTOP 10 WITH WIN RATE >= 50%:")
        print("-" * 90)
        for c in high_wr[:10]:
            print(f"Score>={c['score']:<3} Stop {c['stop']*100:.1f}% Tgt {c['target']*100:.1f}% "
                  f"Hold {c['hold']}d Lev {c['lev']}x | "
                  f"{c['trades']} trades {c['win_rate']:.1f}% win | "
                  f"${c['final']:,.0f} ({c['return']:+.1f}%)")

        # Best balanced strategy
        print("\n" + "=" * 80)
        print("RECOMMENDED PUT STRATEGY")
        print("=" * 80)

        # Pick best from high win rate
        if high_wr:
            best = high_wr[0]
        else:
            best = configs[0]

        trades, stats = self.test_strategy(best['score'], best['stop'], best['target'], best['hold'])

        print(f"\nConfiguration:")
        print(f"  Minimum Score: {best['score']} (out of 100)")
        print(f"  Stop Loss: {best['stop']*100}%")
        print(f"  Target: {best['target']*100}%")
        print(f"  Max Hold: {best['hold']} days")
        print(f"  Recommended Leverage: {best['lev']}x (ATM puts)")

        print(f"\nPerformance (2022-2025):")
        print(f"  Total Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg P&L per trade: {stats['avg_pnl']:.3f}%")

        print(f"\nCapital Growth ($1,000 start):")
        for lev in [5, 8, 10, 12, 15]:
            final = self.simulate(trades, leverage=lev)
            print(f"  {lev}x leverage: ${final:,.2f} ({(final-1000)/10:+.1f}%)")

        # By year
        trades_df = pd.DataFrame(trades)
        trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year

        print(f"\nPerformance by Year:")
        for year in sorted(trades_df['year'].unique()):
            ydf = trades_df[trades_df['year'] == year]
            wins = len(ydf[ydf['pnl'] > 0])
            print(f"  {year}: {len(ydf)} trades, {wins/len(ydf)*100:.1f}% win rate, "
                  f"P&L: {ydf['pnl'].sum():.2f}%")

        # Signal reasons
        print(f"\nKey Signal Components (frequency):")
        reason_counts = {}
        for t in trades:
            for r in t['reasons']:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"  {reason}: {count}/{len(trades)} ({count/len(trades)*100:.0f}%)")

        # Sample trades
        print(f"\nRecent Trades:")
        print("-" * 80)
        for t in trades[-8:]:
            result = "WIN " if t['pnl'] > 0 else "LOSS"
            print(f"  {t['date']}: Score {t['score']}, ${t['entry']:.2f} -> ${t['exit']:.2f} "
                  f"({t['pnl']:+.2f}%) [{result}] {t['exit_reason']}")

        # Generate the scoring criteria for implementation
        print("\n" + "=" * 80)
        print("IMPLEMENTATION SCORING CRITERIA")
        print("=" * 80)
        print("""
PUT SIGNAL SCORING (max ~100 points):

1. RSI(2) Overbought (0-30 points):
   - RSI(2) >= 98: +30 points
   - RSI(2) >= 95: +25 points
   - RSI(2) >= 90: +15 points

2. IBS High (0-20 points):
   - IBS >= 0.95: +20 points
   - IBS >= 0.90: +15 points
   - IBS >= 0.85: +10 points

3. RSI(3) Overbought (0-15 points):
   - RSI(3) >= 95: +15 points
   - RSI(3) >= 90: +10 points

4. Bollinger Band (0-10 points):
   - Close above upper band (>100%): +10 points

5. Consecutive Up Days (0-10 points):
   - 4+ up days: +10 points
   - 3 up days: +5 points

6. Strong 3-Day Rally (0-10 points):
   - 3-day return > 3%: +10 points

7. VIX Complacency (0-5 points):
   - VIX < 14: +5 points

8. Friday (0-5 points):
   - Day is Friday: +5 points

9. Bear Regime (0-5 points):
   - Below 200 MA: +5 points

MINIMUM SCORE: """ + str(best['score']) + f"""

TRADE PARAMETERS:
- Stop Loss: {best['stop']*100}% (price moves up)
- Target: {best['target']*100}% (price moves down)
- Max Hold: {best['hold']} days
- Exit at stop, target, or max hold days

OPTIONS RECOMMENDATION:
- Use ATM or slightly ITM puts
- Expiry: {best['hold']+3}-{best['hold']+7} days out (buffer for max hold)
- Position size: 25% of capital per trade
""")

        return trades, stats


def main():
    strategy = FinalPutStrategy(start_date="2022-01-01")
    strategy.run_analysis()


if __name__ == "__main__":
    main()
