"""
Advanced PUT Strategy Optimizer
Tests combined patterns and more aggressive configurations
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class AdvancedPutOptimizer:
    """Advanced PUT strategy optimizer with combined signals"""

    def __init__(self, start_date: str = "2022-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.spy_data = None
        self.vix_data = None

    def load_data(self) -> bool:
        """Load historical data"""
        print(f"Loading data from {self.start_date} to {self.end_date}...")

        try:
            spy = yf.Ticker("SPY")
            self.spy_data = spy.history(start=self.start_date, end=self.end_date)

            vix = yf.Ticker("^VIX")
            self.vix_data = vix.history(start=self.start_date, end=self.end_date)

            if self.spy_data.empty or self.vix_data.empty:
                print("Failed to load data")
                return False

            self._calculate_indicators()
            print(f"Loaded {len(self.spy_data)} days of data")
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        df = self.spy_data

        # IBS (Internal Bar Strength)
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['IBS'] = df['IBS'].replace([np.inf, -np.inf], np.nan).fillna(0.5)

        # RSI(3)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        df['RSI_3'] = 100 - (100 / (1 + rs))
        df['RSI_3'] = df['RSI_3'].fillna(50)

        # RSI(2) - even more sensitive
        gain2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
        rs2 = gain2 / loss2
        df['RSI_2'] = 100 - (100 / (1 + rs2))
        df['RSI_2'] = df['RSI_2'].fillna(50)

        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Daily returns
        df['Return'] = df['Close'].pct_change() * 100
        df['Return_2D'] = df['Close'].pct_change(2) * 100
        df['Return_3D'] = df['Close'].pct_change(3) * 100
        df['Return_5D'] = df['Close'].pct_change(5) * 100

        # Consecutive up days
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['Consec_Up'] = df['Up_Day'].rolling(window=5).sum()

        # High relative to recent range
        df['High_10D'] = df['High'].rolling(window=10).max()
        df['Low_10D'] = df['Low'].rolling(window=10).min()
        df['Range_Position'] = (df['Close'] - df['Low_10D']) / (df['High_10D'] - df['Low_10D'])

        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100

        # Day of week
        df['DayOfWeek'] = df.index.dayofweek

        # VIX data
        vix_df = self.vix_data
        vix_df['VIX_5_MA'] = vix_df['Close'].rolling(window=5).mean()
        vix_df['VIX_10_MA'] = vix_df['Close'].rolling(window=10).mean()
        vix_df['VIX_20_MA'] = vix_df['Close'].rolling(window=20).mean()

        df['VIX'] = vix_df['Close'].reindex(df.index, method='ffill')
        df['VIX_5_MA'] = vix_df['VIX_5_MA'].reindex(df.index, method='ffill')
        df['VIX_10_MA'] = vix_df['VIX_10_MA'].reindex(df.index, method='ffill')
        df['VIX_20_MA'] = vix_df['VIX_20_MA'].reindex(df.index, method='ffill')

        # Above/Below key MAs
        df['Above_200MA'] = df['Close'] > df['SMA_200']
        df['Above_50MA'] = df['Close'] > df['SMA_50']
        df['Above_10MA'] = df['Close'] > df['SMA_10']

        # Gap detection
        df['Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

        self.spy_data = df

    def generate_all_signals(self) -> pd.DataFrame:
        """Generate all possible PUT signals with scoring"""
        df = self.spy_data.copy()
        signals_list = []

        for i in range(200, len(df)):
            row = df.iloc[i]
            date = df.index[i]

            # Calculate a composite overbought score
            score = 0
            reasons = []

            # IBS signals (0-25 points)
            if row['IBS'] >= 0.95:
                score += 25
                reasons.append("IBS_95+")
            elif row['IBS'] >= 0.90:
                score += 20
                reasons.append("IBS_90+")
            elif row['IBS'] >= 0.85:
                score += 15
                reasons.append("IBS_85+")
            elif row['IBS'] >= 0.80:
                score += 10
                reasons.append("IBS_80+")
            elif row['IBS'] >= 0.75:
                score += 5
                reasons.append("IBS_75+")

            # RSI(2) signals (0-25 points)
            if row['RSI_2'] >= 98:
                score += 25
                reasons.append("RSI2_98+")
            elif row['RSI_2'] >= 95:
                score += 20
                reasons.append("RSI2_95+")
            elif row['RSI_2'] >= 90:
                score += 15
                reasons.append("RSI2_90+")
            elif row['RSI_2'] >= 85:
                score += 10
                reasons.append("RSI2_85+")
            elif row['RSI_2'] >= 80:
                score += 5
                reasons.append("RSI2_80+")

            # RSI(3) signals (0-20 points)
            if row['RSI_3'] >= 95:
                score += 20
                reasons.append("RSI3_95+")
            elif row['RSI_3'] >= 90:
                score += 15
                reasons.append("RSI3_90+")
            elif row['RSI_3'] >= 85:
                score += 10
                reasons.append("RSI3_85+")
            elif row['RSI_3'] >= 75:
                score += 5
                reasons.append("RSI3_75+")

            # Bollinger Band signals (0-15 points)
            if row['BB_PctB'] > 1.1:
                score += 15
                reasons.append("BB_110%")
            elif row['BB_PctB'] > 1.0:
                score += 10
                reasons.append("BB_100%")
            elif row['BB_PctB'] > 0.95:
                score += 5
                reasons.append("BB_95%")

            # Consecutive up days (0-15 points)
            if row['Consec_Up'] >= 5:
                score += 15
                reasons.append("5_Up_Days")
            elif row['Consec_Up'] >= 4:
                score += 10
                reasons.append("4_Up_Days")
            elif row['Consec_Up'] >= 3:
                score += 5
                reasons.append("3_Up_Days")

            # Strong recent rally (0-15 points)
            if row['Return_3D'] > 4:
                score += 15
                reasons.append("Rally_3D>4%")
            elif row['Return_3D'] > 3:
                score += 10
                reasons.append("Rally_3D>3%")
            elif row['Return_3D'] > 2:
                score += 5
                reasons.append("Rally_3D>2%")

            # VIX complacency bonus (0-10 points)
            if row['VIX'] < 12:
                score += 10
                reasons.append("VIX<12")
            elif row['VIX'] < 14:
                score += 7
                reasons.append("VIX<14")
            elif row['VIX'] < 16:
                score += 3
                reasons.append("VIX<16")

            # Range position (0-10 points)
            if row['Range_Position'] > 0.98:
                score += 10
                reasons.append("Range_Top")
            elif row['Range_Position'] > 0.95:
                score += 5
                reasons.append("Range_High")

            # Friday bonus (0-10 points)
            if row['DayOfWeek'] == 4:  # Friday
                score += 10
                reasons.append("Friday")
            elif row['DayOfWeek'] == 3:  # Thursday
                score += 5
                reasons.append("Thursday")

            # Bear market regime penalty/bonus
            if not row['Above_200MA']:
                score += 10  # Bonus for shorting below 200MA
                reasons.append("Below_200MA")

            # Gap up (0-10 points)
            if row['Gap_Pct'] > 1:
                score += 10
                reasons.append("Gap_Up>1%")
            elif row['Gap_Pct'] > 0.5:
                score += 5
                reasons.append("Gap_Up>0.5%")

            # Only record signals with minimum score
            if score >= 20:  # Lowered threshold
                signals_list.append({
                    'date': date,
                    'price': row['Close'],
                    'score': score,
                    'reasons': reasons,
                    'ibs': row['IBS'],
                    'rsi_2': row['RSI_2'],
                    'rsi_3': row['RSI_3'],
                    'vix': row['VIX'],
                    'bb_pctb': row['BB_PctB'],
                    'consec_up': row['Consec_Up'],
                    'return_3d': row['Return_3D'],
                    'day_of_week': row['DayOfWeek'],
                    'above_200ma': row['Above_200MA'],
                    'gap_pct': row['Gap_Pct']
                })

        return pd.DataFrame(signals_list)

    def backtest_signals(self, signals_df: pd.DataFrame, min_score: int,
                         stop_pct: float, target_pct: float, max_hold: int) -> Tuple[List, Dict]:
        """Backtest signals with given parameters"""
        df = self.spy_data
        trades = []
        last_exit_idx = -999

        filtered = signals_df[signals_df['score'] >= min_score]

        for _, signal in filtered.iterrows():
            date = signal['date']

            try:
                i = df.index.get_loc(date)
            except:
                continue

            if i <= last_exit_idx:
                continue

            entry_price = signal['price']

            # For PUT: profit when price falls
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
                'entry_date': str(date.date()) if hasattr(date, 'date') else str(date),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_days': hold_days,
                'score': signal['score'],
                'vix': signal['vix'],
                'reasons': signal['reasons']
            })

        if not trades:
            return [], {'trades': 0}

        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        total = len(trades)

        stats = {
            'trades': total,
            'wins': wins,
            'win_rate': (wins / total) * 100,
            'avg_pnl': sum(t['pnl_pct'] for t in trades) / total,
            'total_pnl': sum(t['pnl_pct'] for t in trades)
        }

        return trades, stats

    def simulate_capital(self, trades: List, starting: float = 1000,
                         leverage: float = 8.0, position_pct: float = 0.25) -> Dict:
        """Simulate capital with options leverage"""
        capital = starting
        max_cap = starting
        max_dd = 0

        for t in trades:
            size = capital * position_pct
            lev_pnl = t['pnl_pct'] * leverage
            if lev_pnl < -100:
                lev_pnl = -100

            pnl = size * (lev_pnl / 100)
            capital += pnl

            if capital > max_cap:
                max_cap = capital
            dd = (max_cap - capital) / max_cap * 100
            if dd > max_dd:
                max_dd = dd

            if capital <= 0:
                capital = 0
                break

        return {
            'final': capital,
            'return': ((capital - starting) / starting) * 100,
            'max_dd': max_dd
        }

    def run_optimization(self):
        """Run full optimization"""
        if not self.load_data():
            return

        print("\nGenerating all PUT signals...")
        signals = self.generate_all_signals()
        print(f"Generated {len(signals)} potential signals")

        print("\n" + "=" * 80)
        print("SCORE-BASED PUT STRATEGY OPTIMIZATION")
        print("=" * 80)

        # Test different min_score thresholds
        results = []

        for min_score in [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
            for stop_pct in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020]:
                for target_pct in [0.015, 0.018, 0.020, 0.025, 0.030, 0.035]:
                    for max_hold in [2, 3, 4, 5]:
                        trades, stats = self.backtest_signals(
                            signals, min_score, stop_pct, target_pct, max_hold
                        )

                        if stats['trades'] >= 10:
                            for leverage in [8, 10, 12, 15]:
                                cap = self.simulate_capital(trades, leverage=leverage)

                                results.append({
                                    'min_score': min_score,
                                    'stop': stop_pct,
                                    'target': target_pct,
                                    'hold': max_hold,
                                    'leverage': leverage,
                                    'trades': stats['trades'],
                                    'win_rate': stats['win_rate'],
                                    'avg_pnl': stats['avg_pnl'],
                                    'final': cap['final'],
                                    'return': cap['return'],
                                    'max_dd': cap['max_dd']
                                })

        # Sort by return
        results.sort(key=lambda x: x['return'], reverse=True)

        print(f"\nTOP 30 CONFIGURATIONS (sorted by return):")
        print("-" * 100)
        print(f"{'Score':<7} {'Stop':<6} {'Tgt':<6} {'Hold':<5} {'Lev':<5} {'#':<5} {'Win%':<7} {'$1K->':<12} {'Return':<10} {'DD':<8}")
        print("-" * 100)

        for r in results[:30]:
            print(f"{r['min_score']:<7} {r['stop']*100:.1f}%  {r['target']*100:.1f}%  {r['hold']}d    "
                  f"{r['leverage']}x   {r['trades']:<5} {r['win_rate']:.1f}%   "
                  f"${r['final']:,.0f}       {r['return']:+.1f}%     {r['max_dd']:.1f}%")

        # Best risk-adjusted
        for r in results:
            r['risk_adj'] = r['return'] / r['max_dd'] if r['max_dd'] > 0 else 0

        results.sort(key=lambda x: x['risk_adj'], reverse=True)

        print(f"\nTOP 15 RISK-ADJUSTED (Return/DD ratio):")
        print("-" * 100)
        for r in results[:15]:
            print(f"Score>={r['min_score']:<3} Stop {r['stop']*100:.1f}% Tgt {r['target']*100:.1f}% "
                  f"Hold {r['hold']}d Lev {r['leverage']}x | "
                  f"{r['trades']} trades {r['win_rate']:.1f}% win | "
                  f"${r['final']:,.0f} ({r['return']:+.1f}%) R/DD:{r['risk_adj']:.2f}")

        # Deep dive on best
        best = results[0]
        print("\n" + "=" * 80)
        print("DEEP DIVE ON BEST RISK-ADJUSTED STRATEGY")
        print("=" * 80)

        trades, stats = self.backtest_signals(
            signals, best['min_score'], best['stop'], best['target'], best['hold']
        )

        print(f"\nConfiguration:")
        print(f"  Min Score: {best['min_score']}")
        print(f"  Stop Loss: {best['stop']*100}%")
        print(f"  Target: {best['target']*100}%")
        print(f"  Max Hold: {best['hold']} days")
        print(f"  Leverage: {best['leverage']}x")

        print(f"\nPerformance:")
        print(f"  Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg P&L: {stats['avg_pnl']:.3f}%")
        print(f"  Total P&L: {stats['total_pnl']:.2f}%")

        print(f"\nCapital Simulation ($1,000 start):")
        for lev in [5, 8, 10, 12, 15, 20]:
            cap = self.simulate_capital(trades, leverage=lev)
            print(f"  {lev}x leverage: ${cap['final']:,.2f} ({cap['return']:+.1f}%), Max DD: {cap['max_dd']:.1f}%")

        # Show by year
        trades_df = pd.DataFrame(trades)
        trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year

        print(f"\nBy Year:")
        for year in sorted(trades_df['year'].unique()):
            ydf = trades_df[trades_df['year'] == year]
            wins = len(ydf[ydf['pnl_pct'] > 0])
            print(f"  {year}: {len(ydf)} trades, {wins/len(ydf)*100:.1f}% win rate, "
                  f"Total P&L: {ydf['pnl_pct'].sum():.2f}%")

        # Show reason frequency
        print(f"\nMost Common Signal Reasons:")
        reason_counts = {}
        for t in trades:
            for r in t['reasons']:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")

        print(f"\nSample Recent Trades:")
        for t in trades[-10:]:
            result = "WIN" if t['pnl_pct'] > 0 else "LOSS"
            print(f"  {t['entry_date']}: Score {t['score']}, ${t['entry_price']:.2f} -> ${t['exit_price']:.2f} "
                  f"({t['pnl_pct']:+.2f}%) [{result}] - {t['exit_reason']}")

        return results, trades


def main():
    optimizer = AdvancedPutOptimizer(start_date="2022-01-01")
    results, trades = optimizer.run_optimization()


if __name__ == "__main__":
    main()
