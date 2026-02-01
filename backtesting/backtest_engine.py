"""
Backtest Engine - Tests SPY signal strategies against historical data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import sys

sys.path.insert(0, '.')
from signals.signal_detector import SPYSignalDetector, SignalType, Direction
from signals.opportunity_scorer import SPYOpportunityScorer, OpportunityGrade


@dataclass
class TradeResult:
    """Result of a single trade"""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    direction: str
    score: int
    grade: str
    pnl_pct: float
    pnl_dollars: float
    hold_days: int
    hit_target: bool
    hit_stop: bool
    exit_reason: str
    signals: List[str]
    vix_at_entry: float
    ibs_at_entry: float
    rsi_at_entry: float


class BacktestEngine:
    """
    Backtests the SPY opportunity signals against historical data
    """

    def __init__(self,
                 start_date: str = "2020-01-01",
                 end_date: str = None,
                 initial_capital: float = 100000):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital
        self.detector = SPYSignalDetector()
        self.scorer = SPYOpportunityScorer()

        # Trade parameters - OPTIMIZED 2024-12-19
        # Previous: SL 2%, T1 1% = 73% win but only +5% P&L (bad R:R)
        # Optimized: SL 1.5%, T1 2% = 63% win but +15% P&L (better R:R)
        self.stop_loss_pct = 0.015  # 1.5% (tighter stop)
        self.target_1_pct = 0.02    # 2% (wider target)
        self.target_2_pct = 0.03    # 3%
        self.max_hold_days = 5

        self.spy_data = None
        self.vix_data = None
        self.trades: List[TradeResult] = []

    def load_data(self) -> bool:
        """Load historical SPY and VIX data"""
        logger.info(f"Loading data from {self.start_date} to {self.end_date}")

        try:
            # Load SPY data
            spy = yf.Ticker("SPY")
            self.spy_data = spy.history(start=self.start_date, end=self.end_date)

            # Load VIX data
            vix = yf.Ticker("^VIX")
            self.vix_data = vix.history(start=self.start_date, end=self.end_date)

            if self.spy_data.empty or self.vix_data.empty:
                logger.error("Failed to load data")
                return False

            # Calculate technical indicators
            self._calculate_indicators()

            logger.info(f"Loaded {len(self.spy_data)} days of SPY data")
            logger.info(f"Loaded {len(self.vix_data)} days of VIX data")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
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

        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Daily returns
        df['Return'] = df['Close'].pct_change() * 100

        # VIX indicators
        vix_df = self.vix_data
        vix_df['VIX_10_MA'] = vix_df['Close'].rolling(window=10).mean()

        # Merge VIX into SPY data
        df['VIX'] = vix_df['Close'].reindex(df.index, method='ffill')
        df['VIX_10_MA'] = vix_df['VIX_10_MA'].reindex(df.index, method='ffill')

        self.spy_data = df

    def _generate_signals_for_date(self, date) -> Optional[Dict]:
        """Generate signals for a specific date"""
        try:
            row = self.spy_data.loc[date]

            market_data = {
                'ibs': row['IBS'],
                'rsi_3': row['RSI_3'],
                'vix': row['VIX'],
                'vix_10_ma': row['VIX_10_MA'],
                'price': row['Close'],
                'sma_50': row['SMA_50'],
                'sma_200': row['SMA_200'],
            }

            # Skip if any key data is missing
            if pd.isna(market_data['sma_50']) or pd.isna(market_data['vix']):
                return None

            signals = self.detector.analyze_all(market_data)

            if not signals:
                return None

            opportunity = self.scorer.score_opportunity(signals, market_data['price'])

            if opportunity:
                return {
                    'date': date,
                    'price': market_data['price'],
                    'opportunity': opportunity,
                    'market_data': market_data,
                    'signals': [s.signal_type.value for s in signals]
                }

            return None

        except Exception as e:
            return None

    def _simulate_trade(self, entry_date, entry_data: Dict) -> Optional[TradeResult]:
        """Simulate a trade from entry to exit"""
        opportunity = entry_data['opportunity']
        entry_price = entry_data['price']
        market_data = entry_data['market_data']

        direction = opportunity.direction.value
        score = opportunity.score
        grade = opportunity.grade.value

        # Calculate stops and targets
        if direction == 'long':
            stop_price = entry_price * (1 - self.stop_loss_pct)
            target_1 = entry_price * (1 + self.target_1_pct)
            target_2 = entry_price * (1 + self.target_2_pct)
        else:
            stop_price = entry_price * (1 + self.stop_loss_pct)
            target_1 = entry_price * (1 - self.target_1_pct)
            target_2 = entry_price * (1 - self.target_2_pct)

        # Find entry date index
        try:
            entry_idx = self.spy_data.index.get_loc(entry_date)
        except:
            return None

        # Simulate forward
        exit_price = entry_price
        exit_date = entry_date
        exit_reason = "max_hold"
        hit_target = False
        hit_stop = False

        for day_offset in range(1, self.max_hold_days + 1):
            if entry_idx + day_offset >= len(self.spy_data):
                break

            future_date = self.spy_data.index[entry_idx + day_offset]
            future_row = self.spy_data.iloc[entry_idx + day_offset]

            high = future_row['High']
            low = future_row['Low']
            close = future_row['Close']

            if direction == 'long':
                # Check stop first (conservative)
                if low <= stop_price:
                    exit_price = stop_price
                    exit_date = future_date
                    exit_reason = "stop_loss"
                    hit_stop = True
                    break
                # Check target
                elif high >= target_1:
                    exit_price = target_1
                    exit_date = future_date
                    exit_reason = "target_1"
                    hit_target = True
                    break
            else:  # short
                if high >= stop_price:
                    exit_price = stop_price
                    exit_date = future_date
                    exit_reason = "stop_loss"
                    hit_stop = True
                    break
                elif low <= target_1:
                    exit_price = target_1
                    exit_date = future_date
                    exit_reason = "target_1"
                    hit_target = True
                    break

            # Update exit to close if holding
            exit_price = close
            exit_date = future_date

        # Calculate P&L
        if direction == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_dollars = (pnl_pct / 100) * self.initial_capital * 0.02  # 2% position size

        hold_days = (exit_date - entry_date).days if isinstance(exit_date, datetime) else day_offset

        return TradeResult(
            entry_date=str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            entry_price=entry_price,
            exit_date=str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            exit_price=exit_price,
            direction=direction,
            score=score,
            grade=grade,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            hold_days=hold_days,
            hit_target=hit_target,
            hit_stop=hit_stop,
            exit_reason=exit_reason,
            signals=entry_data['signals'],
            vix_at_entry=market_data['vix'],
            ibs_at_entry=market_data['ibs'],
            rsi_at_entry=market_data['rsi_3']
        )

    def run_backtest(self, min_grade: str = "A+") -> List[TradeResult]:
        """
        Run backtest for signals meeting minimum grade

        Args:
            min_grade: Minimum grade to include ("A+", "A", "B", "C")
        """
        if self.spy_data is None:
            if not self.load_data():
                return []

        grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "F": 4}
        min_grade_level = grade_order.get(min_grade, 0)

        logger.info(f"Running backtest for grade >= {min_grade}")

        self.trades = []
        in_trade = False
        last_trade_exit = None

        for date in self.spy_data.index[200:]:  # Skip first 200 days for MA calculation
            # Skip if we're still in a trade
            if in_trade and last_trade_exit:
                # Handle timezone comparison
                date_naive = date.tz_localize(None) if date.tzinfo else date
                exit_naive = last_trade_exit.tz_localize(None) if last_trade_exit.tzinfo else last_trade_exit
                if date_naive <= exit_naive:
                    continue

            in_trade = False

            # Generate signals
            signal_data = self._generate_signals_for_date(date)

            if signal_data is None:
                continue

            opportunity = signal_data['opportunity']
            grade_level = grade_order.get(opportunity.grade.value, 4)

            # Check if meets minimum grade
            if grade_level > min_grade_level:
                continue

            # Simulate the trade
            trade_result = self._simulate_trade(date, signal_data)

            if trade_result:
                self.trades.append(trade_result)
                in_trade = True
                # Parse exit date for cooldown
                try:
                    last_trade_exit = pd.Timestamp(trade_result.exit_date)
                except:
                    last_trade_exit = date + timedelta(days=self.max_hold_days)

        logger.info(f"Completed backtest: {len(self.trades)} trades")
        return self.trades

    def get_results_summary(self) -> Dict:
        """Get summary statistics of backtest results"""
        if not self.trades:
            return {"error": "No trades to analyze"}

        df = pd.DataFrame([vars(t) for t in self.trades])

        total_trades = len(df)
        winning_trades = len(df[df['pnl_pct'] > 0])
        losing_trades = len(df[df['pnl_pct'] <= 0])

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0

        # By grade
        grade_stats = {}
        for grade in df['grade'].unique():
            grade_df = df[df['grade'] == grade]
            grade_wins = len(grade_df[grade_df['pnl_pct'] > 0])
            grade_total = len(grade_df)
            grade_stats[grade] = {
                'total': grade_total,
                'wins': grade_wins,
                'win_rate': (grade_wins / grade_total * 100) if grade_total > 0 else 0,
                'avg_pnl': grade_df['pnl_pct'].mean()
            }

        # By VIX filter
        vix_passed = df[df['vix_at_entry'] > df['vix_at_entry'].rolling(10).mean().shift(1)]
        vix_failed = df[df['vix_at_entry'] <= df['vix_at_entry'].rolling(10).mean().shift(1)]

        # By exit reason
        exit_stats = df.groupby('exit_reason').agg({
            'pnl_pct': ['count', 'mean'],
        }).to_dict()

        # By direction
        direction_stats = {}
        for direction in df['direction'].unique():
            dir_df = df[df['direction'] == direction]
            dir_wins = len(dir_df[dir_df['pnl_pct'] > 0])
            dir_total = len(dir_df)
            direction_stats[direction] = {
                'total': dir_total,
                'wins': dir_wins,
                'win_rate': (dir_wins / dir_total * 100) if dir_total > 0 else 0,
                'avg_pnl': dir_df['pnl_pct'].mean()
            }

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'total_pnl_pct': df['pnl_pct'].sum(),
            'avg_pnl_per_trade': df['pnl_pct'].mean(),
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0,
            'max_win': df['pnl_pct'].max(),
            'max_loss': df['pnl_pct'].min(),
            'avg_hold_days': df['hold_days'].mean(),
            'grade_stats': grade_stats,
            'direction_stats': direction_stats,
            'target_hit_rate': (len(df[df['hit_target']]) / total_trades * 100) if total_trades > 0 else 0,
            'stop_hit_rate': (len(df[df['hit_stop']]) / total_trades * 100) if total_trades > 0 else 0,
        }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("SPY BACKTEST ENGINE TEST")
    print("=" * 60)

    engine = BacktestEngine(start_date="2022-01-01")
    trades = engine.run_backtest(min_grade="A+")

    if trades:
        summary = engine.get_results_summary()
        print(f"\nTotal Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Avg P&L: {summary['avg_pnl_per_trade']:.2f}%")
        print(f"Total P&L: {summary['total_pnl_pct']:.2f}%")

        print("\nBy Grade:")
        for grade, stats in summary['grade_stats'].items():
            print(f"  {grade}: {stats['win_rate']:.1f}% win rate ({stats['total']} trades)")
