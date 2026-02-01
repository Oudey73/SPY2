"""
PUT Strategy Optimizer
Finds the optimal SHORT/PUT strategy for SPY by testing multiple patterns
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class PutSignal:
    """A PUT/SHORT signal"""
    date: datetime
    price: float
    pattern: str
    ibs: float
    rsi: float
    vix: float
    above_200ma: bool
    day_of_week: int  # 0=Monday, 4=Friday


@dataclass
class PutTrade:
    """Result of a PUT trade"""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    pattern: str
    pnl_pct: float
    hold_days: int
    hit_target: bool
    hit_stop: bool
    exit_reason: str
    vix: float
    ibs: float
    rsi: float


class PutStrategyOptimizer:
    """
    Optimizes PUT/SHORT strategies by testing multiple pattern combinations
    """

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

        # RSI(14) for comparison
        gain14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs14 = gain14 / loss14
        df['RSI_14'] = 100 - (100 / (1 + rs14))
        df['RSI_14'] = df['RSI_14'].fillna(50)

        # Moving Averages
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
        df['Return_5D'] = df['Close'].pct_change(5) * 100

        # Consecutive up days
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['Consec_Up'] = df['Up_Day'].rolling(window=5).sum()

        # Day of week
        df['DayOfWeek'] = df.index.dayofweek

        # VIX data
        vix_df = self.vix_data
        vix_df['VIX_10_MA'] = vix_df['Close'].rolling(window=10).mean()
        vix_df['VIX_20_MA'] = vix_df['Close'].rolling(window=20).mean()

        df['VIX'] = vix_df['Close'].reindex(df.index, method='ffill')
        df['VIX_10_MA'] = vix_df['VIX_10_MA'].reindex(df.index, method='ffill')
        df['VIX_20_MA'] = vix_df['VIX_20_MA'].reindex(df.index, method='ffill')

        # Above/Below key MAs
        df['Above_200MA'] = df['Close'] > df['SMA_200']
        df['Above_50MA'] = df['Close'] > df['SMA_50']

        self.spy_data = df

    def test_pattern(self, pattern_name: str, conditions: callable,
                     stop_pct: float = 0.015, target_pct: float = 0.02,
                     max_hold: int = 5) -> Tuple[List[PutTrade], Dict]:
        """
        Test a specific pattern with given conditions

        Args:
            pattern_name: Name of the pattern
            conditions: Function that takes a row and returns True if signal triggers
            stop_pct: Stop loss percentage (e.g., 0.015 = 1.5%)
            target_pct: Target profit percentage
            max_hold: Maximum hold days
        """
        df = self.spy_data
        trades = []
        in_trade = False
        last_exit_idx = -999

        for i in range(200, len(df)):
            # Skip if still in cooldown
            if i <= last_exit_idx + 1:
                continue

            row = df.iloc[i]

            # Check if conditions are met
            try:
                if not conditions(row):
                    continue
            except:
                continue

            # Signal triggered - simulate PUT trade
            entry_price = row['Close']
            entry_date = df.index[i]

            # Calculate stop and target (for PUT: profit when price falls)
            stop_price = entry_price * (1 + stop_pct)   # Stop above
            target_price = entry_price * (1 - target_pct)  # Target below

            exit_price = entry_price
            exit_date = entry_date
            exit_reason = "max_hold"
            hit_target = False
            hit_stop = False

            for j in range(1, max_hold + 1):
                if i + j >= len(df):
                    break

                future_row = df.iloc[i + j]
                high = future_row['High']
                low = future_row['Low']
                close = future_row['Close']

                # Check stop first (price goes up = loss for PUT)
                if high >= stop_price:
                    exit_price = stop_price
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    hit_stop = True
                    break

                # Check target (price goes down = profit for PUT)
                if low <= target_price:
                    exit_price = target_price
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    hit_target = True
                    break

                exit_price = close
                exit_date = df.index[i + j]
                last_exit_idx = i + j

            # Calculate P&L (for PUT: profit when price falls)
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            hold_days = j if 'j' in dir() else max_hold
            last_exit_idx = i + hold_days

            trade = PutTrade(
                entry_date=str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                entry_price=entry_price,
                exit_date=str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                exit_price=exit_price,
                pattern=pattern_name,
                pnl_pct=pnl_pct,
                hold_days=hold_days,
                hit_target=hit_target,
                hit_stop=hit_stop,
                exit_reason=exit_reason,
                vix=row['VIX'],
                ibs=row['IBS'],
                rsi=row['RSI_3']
            )
            trades.append(trade)

        # Calculate stats
        if not trades:
            return [], {"trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

        wins = sum(1 for t in trades if t.pnl_pct > 0)
        total = len(trades)
        avg_pnl = sum(t.pnl_pct for t in trades) / total
        total_pnl = sum(t.pnl_pct for t in trades)

        stats = {
            "pattern": pattern_name,
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total) * 100,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "target_hits": sum(1 for t in trades if t.hit_target),
            "stop_hits": sum(1 for t in trades if t.hit_stop),
            "avg_hold": sum(t.hold_days for t in trades) / total
        }

        return trades, stats

    def simulate_capital(self, trades: List[PutTrade], starting_capital: float = 1000,
                         leverage: float = 8.0, position_pct: float = 0.25) -> Dict:
        """
        Simulate capital growth with options leverage

        Args:
            trades: List of PutTrade results
            starting_capital: Starting capital in dollars
            leverage: Options leverage multiplier (8x for ATM puts)
            position_pct: Percentage of capital per trade
        """
        capital = starting_capital
        capital_history = [starting_capital]
        max_capital = starting_capital
        max_drawdown = 0

        for trade in trades:
            position_size = capital * position_pct

            # Apply leverage to P&L
            leveraged_pnl_pct = trade.pnl_pct * leverage

            # Cap losses at position size (can't lose more than invested)
            if leveraged_pnl_pct < -100:
                leveraged_pnl_pct = -100

            pnl_dollars = position_size * (leveraged_pnl_pct / 100)
            capital += pnl_dollars

            # Track max drawdown
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            capital_history.append(capital)

            # Stop if blew up
            if capital <= 0:
                capital = 0
                break

        return {
            "final_capital": capital,
            "return_pct": ((capital - starting_capital) / starting_capital) * 100,
            "max_drawdown": max_drawdown,
            "capital_history": capital_history,
            "trades_count": len(trades)
        }

    def run_optimization(self):
        """Run full optimization across all pattern combinations"""
        if not self.load_data():
            return

        print("\n" + "=" * 70)
        print("PUT STRATEGY OPTIMIZATION")
        print("=" * 70)

        # Define all patterns to test
        patterns = {
            # Basic Overbought Patterns
            "IBS_High_Only": lambda r: r['IBS'] >= 0.85,
            "RSI3_High_Only": lambda r: r['RSI_3'] >= 80,
            "RSI3_Extreme": lambda r: r['RSI_3'] >= 90,

            # Combined IBS + RSI
            "IBS80_RSI75": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 75,
            "IBS85_RSI80": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 80,
            "IBS90_RSI85": lambda r: r['IBS'] >= 0.9 and r['RSI_3'] >= 85,

            # Trend Filter Patterns
            "Overbought_Below200MA": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 75 and not r['Above_200MA'],
            "Overbought_Above200MA": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 75 and r['Above_200MA'],

            # VIX Patterns
            "VIX_Low_Overbought": lambda r: r['VIX'] < 15 and r['IBS'] >= 0.8 and r['RSI_3'] >= 80,
            "VIX_Complacent": lambda r: r['VIX'] < 13 and r['RSI_3'] >= 75,
            "VIX_Rising_Overbought": lambda r: r['VIX'] > r['VIX_10_MA'] and r['IBS'] >= 0.8 and r['RSI_3'] >= 75,

            # Day of Week Patterns
            "Friday_Overbought": lambda r: r['DayOfWeek'] == 4 and r['IBS'] >= 0.75 and r['RSI_3'] >= 75,
            "Thursday_Overbought": lambda r: r['DayOfWeek'] == 3 and r['IBS'] >= 0.8 and r['RSI_3'] >= 80,
            "Monday_Overbought": lambda r: r['DayOfWeek'] == 0 and r['IBS'] >= 0.8 and r['RSI_3'] >= 80,

            # Consecutive Up Days
            "3_Up_Days": lambda r: r['Consec_Up'] >= 3 and r['RSI_3'] >= 70,
            "4_Up_Days": lambda r: r['Consec_Up'] >= 4 and r['RSI_3'] >= 75,
            "5_Up_Days": lambda r: r['Consec_Up'] >= 5,

            # Bollinger Band Patterns
            "Above_BB_Upper": lambda r: r['Close'] > r['BB_Upper'] and r['RSI_3'] >= 75,
            "BB_PctB_Extreme": lambda r: r['BB_PctB'] > 1.0 and r['IBS'] >= 0.8,

            # Strong Momentum Reversal
            "Strong_2D_Rally": lambda r: r['Return_2D'] > 3 and r['IBS'] >= 0.75,
            "Strong_5D_Rally": lambda r: r['Return_5D'] > 5 and r['RSI_3'] >= 80,

            # Bear Market Bounce
            "Bear_Bounce": lambda r: not r['Above_200MA'] and r['RSI_3'] >= 80 and r['IBS'] >= 0.8,
            "Bear_Bounce_Extreme": lambda r: not r['Above_200MA'] and r['RSI_3'] >= 85 and r['IBS'] >= 0.85,

            # Multi-Factor Combinations
            "Multi_Factor_A": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 80 and r['VIX'] < 18,
            "Multi_Factor_B": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 75 and r['DayOfWeek'] >= 3,
            "Multi_Factor_C": lambda r: r['RSI_3'] >= 85 and r['Consec_Up'] >= 3 and r['VIX'] < 20,

            # Extreme Combinations
            "Extreme_Overbought": lambda r: r['IBS'] >= 0.9 and r['RSI_3'] >= 90,
            "Triple_Extreme": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 85 and r['BB_PctB'] > 0.95,
        }

        # Test different stop/target combinations
        stop_target_combos = [
            (0.010, 0.015, 3),   # Tight stop, small target, short hold
            (0.012, 0.018, 3),
            (0.015, 0.020, 3),
            (0.015, 0.025, 5),
            (0.015, 0.020, 5),   # Default
            (0.020, 0.025, 5),
            (0.020, 0.030, 5),
            (0.010, 0.025, 4),   # Tight stop, wide target
            (0.012, 0.030, 5),
        ]

        results = []

        print(f"\nTesting {len(patterns)} patterns x {len(stop_target_combos)} stop/target combos...")
        print("-" * 70)

        for pattern_name, conditions in patterns.items():
            for stop_pct, target_pct, max_hold in stop_target_combos:
                trades, stats = self.test_pattern(
                    pattern_name, conditions,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    max_hold=max_hold
                )

                if stats['trades'] >= 15:  # Minimum trades for statistical significance
                    capital_result = self.simulate_capital(trades, leverage=8.0)

                    results.append({
                        'pattern': pattern_name,
                        'stop': stop_pct,
                        'target': target_pct,
                        'hold': max_hold,
                        'trades': stats['trades'],
                        'win_rate': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl'],
                        'total_pnl': stats['total_pnl'],
                        'final_capital': capital_result['final_capital'],
                        'return_pct': capital_result['return_pct'],
                        'max_drawdown': capital_result['max_drawdown'],
                        'profit_per_trade': capital_result['return_pct'] / stats['trades'] if stats['trades'] > 0 else 0
                    })

        # Sort by return
        results.sort(key=lambda x: x['return_pct'], reverse=True)

        # Display top 20 results
        print(f"\nTOP 20 PUT STRATEGIES (sorted by return):")
        print("=" * 100)
        print(f"{'Pattern':<25} {'Stop':<6} {'Tgt':<6} {'Hold':<5} {'Trades':<7} {'Win%':<7} {'$1K->':<10} {'Return':<10} {'DD':<8}")
        print("-" * 100)

        for r in results[:20]:
            print(f"{r['pattern'][:24]:<25} {r['stop']*100:.1f}%  {r['target']*100:.1f}%  {r['hold']}d    "
                  f"{r['trades']:<7} {r['win_rate']:.1f}%   ${r['final_capital']:,.0f}    "
                  f"{r['return_pct']:+.1f}%     {r['max_drawdown']:.1f}%")

        print("\n" + "=" * 100)

        # Best by win rate (minimum 20 trades)
        high_trades = [r for r in results if r['trades'] >= 20]
        high_trades.sort(key=lambda x: x['win_rate'], reverse=True)

        print(f"\nTOP 10 BY WIN RATE (min 20 trades):")
        print("-" * 100)
        for r in high_trades[:10]:
            print(f"{r['pattern'][:24]:<25} {r['stop']*100:.1f}%  {r['target']*100:.1f}%  {r['hold']}d    "
                  f"{r['trades']:<7} {r['win_rate']:.1f}%   ${r['final_capital']:,.0f}    "
                  f"{r['return_pct']:+.1f}%")

        # Best risk-adjusted (return / drawdown)
        for r in results:
            r['risk_adj'] = r['return_pct'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0

        results.sort(key=lambda x: x['risk_adj'], reverse=True)

        print(f"\nTOP 10 RISK-ADJUSTED (Return/Drawdown ratio):")
        print("-" * 100)
        for r in results[:10]:
            print(f"{r['pattern'][:24]:<25} {r['stop']*100:.1f}%  {r['target']*100:.1f}%  {r['hold']}d    "
                  f"{r['trades']:<7} {r['win_rate']:.1f}%   ${r['final_capital']:,.0f}    "
                  f"R/DD: {r['risk_adj']:.2f}")

        # Return top result for further analysis
        return results

    def deep_dive_pattern(self, pattern_name: str, conditions: callable,
                          stop_pct: float, target_pct: float, max_hold: int):
        """Deep dive analysis on a specific pattern"""
        print(f"\n{'='*70}")
        print(f"DEEP DIVE: {pattern_name}")
        print(f"Stop: {stop_pct*100}%, Target: {target_pct*100}%, Max Hold: {max_hold} days")
        print(f"{'='*70}")

        trades, stats = self.test_pattern(
            pattern_name, conditions,
            stop_pct=stop_pct, target_pct=target_pct, max_hold=max_hold
        )

        if not trades:
            print("No trades found!")
            return

        print(f"\nBASIC STATS:")
        print(f"  Total Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg P&L: {stats['avg_pnl']:.3f}%")
        print(f"  Total P&L: {stats['total_pnl']:.2f}%")
        print(f"  Target Hits: {stats['target_hits']}")
        print(f"  Stop Hits: {stats['stop_hits']}")
        print(f"  Avg Hold: {stats['avg_hold']:.1f} days")

        # By year
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year

        print(f"\nBY YEAR:")
        for year in sorted(trades_df['year'].unique()):
            year_df = trades_df[trades_df['year'] == year]
            wins = len(year_df[year_df['pnl_pct'] > 0])
            total = len(year_df)
            print(f"  {year}: {total} trades, {wins/total*100:.1f}% win rate, "
                  f"Total P&L: {year_df['pnl_pct'].sum():.2f}%")

        # Capital simulation
        for leverage in [5, 8, 10, 12, 15]:
            result = self.simulate_capital(trades, leverage=leverage)
            print(f"\n  Leverage {leverage}x: $1,000 -> ${result['final_capital']:,.2f} "
                  f"({result['return_pct']:+.1f}%), Max DD: {result['max_drawdown']:.1f}%")

        # Show recent trades
        print(f"\nRECENT TRADES (last 10):")
        print("-" * 70)
        for t in trades[-10:]:
            result = "WIN" if t.pnl_pct > 0 else "LOSS"
            print(f"  {t.entry_date}: Entry ${t.entry_price:.2f} -> Exit ${t.exit_price:.2f} "
                  f"({t.pnl_pct:+.2f}%) [{result}] - {t.exit_reason}")

        return trades, stats


def main():
    """Run the PUT strategy optimization"""
    optimizer = PutStrategyOptimizer(start_date="2022-01-01")

    # Run full optimization
    results = optimizer.run_optimization()

    if not results:
        return

    # Deep dive on top 3 strategies
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF TOP 3 STRATEGIES")
    print("=" * 70)

    # Get top 3 unique patterns
    seen_patterns = set()
    top_3 = []
    for r in results:
        if r['pattern'] not in seen_patterns and len(top_3) < 3:
            top_3.append(r)
            seen_patterns.add(r['pattern'])

    # Define pattern conditions again for deep dive
    pattern_conditions = {
        "Bear_Bounce": lambda r: not r['Above_200MA'] and r['RSI_3'] >= 80 and r['IBS'] >= 0.8,
        "Bear_Bounce_Extreme": lambda r: not r['Above_200MA'] and r['RSI_3'] >= 85 and r['IBS'] >= 0.85,
        "Overbought_Below200MA": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 75 and not r['Above_200MA'],
        "VIX_Low_Overbought": lambda r: r['VIX'] < 15 and r['IBS'] >= 0.8 and r['RSI_3'] >= 80,
        "Friday_Overbought": lambda r: r['DayOfWeek'] == 4 and r['IBS'] >= 0.75 and r['RSI_3'] >= 75,
        "Strong_2D_Rally": lambda r: r['Return_2D'] > 3 and r['IBS'] >= 0.75,
        "Multi_Factor_A": lambda r: r['IBS'] >= 0.8 and r['RSI_3'] >= 80 and r['VIX'] < 18,
        "Multi_Factor_B": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 75 and r['DayOfWeek'] >= 3,
        "Triple_Extreme": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 85 and r['BB_PctB'] > 0.95,
        "4_Up_Days": lambda r: r['Consec_Up'] >= 4 and r['RSI_3'] >= 75,
        "5_Up_Days": lambda r: r['Consec_Up'] >= 5,
        "Extreme_Overbought": lambda r: r['IBS'] >= 0.9 and r['RSI_3'] >= 90,
        "IBS85_RSI80": lambda r: r['IBS'] >= 0.85 and r['RSI_3'] >= 80,
        "IBS_High_Only": lambda r: r['IBS'] >= 0.85,
        "RSI3_High_Only": lambda r: r['RSI_3'] >= 80,
        "RSI3_Extreme": lambda r: r['RSI_3'] >= 90,
    }

    for r in top_3:
        if r['pattern'] in pattern_conditions:
            optimizer.deep_dive_pattern(
                r['pattern'],
                pattern_conditions[r['pattern']],
                r['stop'],
                r['target'],
                r['hold']
            )

    # Final recommendation
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    best = results[0]
    print(f"\nBest Overall Strategy: {best['pattern']}")
    print(f"  Configuration: Stop {best['stop']*100}%, Target {best['target']*100}%, Hold {best['hold']} days")
    print(f"  Performance: {best['trades']} trades, {best['win_rate']:.1f}% win rate")
    print(f"  $1,000 -> ${best['final_capital']:,.2f} ({best['return_pct']:+.1f}%)")
    print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")


if __name__ == "__main__":
    main()
