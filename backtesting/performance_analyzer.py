"""
Performance Analyzer - Analyzes backtest results and suggests improvements
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ImprovementSuggestion:
    """A suggested improvement to the strategy"""
    area: str
    current_value: float
    suggested_value: float
    expected_impact: str
    confidence: str
    rationale: str


class PerformanceAnalyzer:
    """
    Analyzes backtest results to identify areas for improvement
    """

    def __init__(self, trades: List, summary: Dict):
        self.trades = trades
        self.summary = summary
        self.df = pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame()
        self.improvements: List[ImprovementSuggestion] = []

    def analyze_all(self) -> Dict:
        """Run all analysis and return comprehensive report"""
        if self.df.empty:
            return {"error": "No trades to analyze"}

        report = {
            "overall_performance": self._analyze_overall(),
            "signal_analysis": self._analyze_signals(),
            "vix_filter_analysis": self._analyze_vix_filter(),
            "ibs_threshold_analysis": self._analyze_ibs_thresholds(),
            "rsi_threshold_analysis": self._analyze_rsi_thresholds(),
            "timing_analysis": self._analyze_timing(),
            "exit_analysis": self._analyze_exits(),
            "improvements": self._generate_improvements(),
        }

        return report

    def _analyze_overall(self) -> Dict:
        """Analyze overall performance metrics"""
        return {
            "win_rate": self.summary.get('win_rate', 0),
            "target_win_rate": 71,  # Based on research
            "gap": self.summary.get('win_rate', 0) - 71,
            "assessment": "GOOD" if self.summary.get('win_rate', 0) >= 65 else "NEEDS_IMPROVEMENT"
        }

    def _analyze_signals(self) -> Dict:
        """Analyze which signals perform best"""
        signal_performance = {}

        # Flatten signals and analyze
        for idx, row in self.df.iterrows():
            for signal in row['signals']:
                if signal not in signal_performance:
                    signal_performance[signal] = {'wins': 0, 'losses': 0, 'pnl': []}

                if row['pnl_pct'] > 0:
                    signal_performance[signal]['wins'] += 1
                else:
                    signal_performance[signal]['losses'] += 1
                signal_performance[signal]['pnl'].append(row['pnl_pct'])

        # Calculate stats
        for signal, data in signal_performance.items():
            total = data['wins'] + data['losses']
            data['win_rate'] = (data['wins'] / total * 100) if total > 0 else 0
            data['avg_pnl'] = np.mean(data['pnl']) if data['pnl'] else 0
            data['total_trades'] = total

        # Sort by win rate
        sorted_signals = sorted(signal_performance.items(),
                               key=lambda x: x[1]['win_rate'], reverse=True)

        return {
            "best_signals": sorted_signals[:5],
            "worst_signals": sorted_signals[-3:] if len(sorted_signals) > 3 else [],
            "all_signals": signal_performance
        }

    def _analyze_vix_filter(self) -> Dict:
        """Analyze VIX filter effectiveness"""
        # VIX above 10-day MA
        high_vix = self.df[self.df['vix_at_entry'] > 20]
        low_vix = self.df[self.df['vix_at_entry'] <= 20]

        high_vix_wins = len(high_vix[high_vix['pnl_pct'] > 0])
        high_vix_total = len(high_vix)
        high_vix_wr = (high_vix_wins / high_vix_total * 100) if high_vix_total > 0 else 0

        low_vix_wins = len(low_vix[low_vix['pnl_pct'] > 0])
        low_vix_total = len(low_vix)
        low_vix_wr = (low_vix_wins / low_vix_total * 100) if low_vix_total > 0 else 0

        # VIX level buckets
        vix_buckets = {}
        for bucket_name, (low, high) in [
            ("very_low", (0, 15)),
            ("low", (15, 20)),
            ("normal", (20, 25)),
            ("elevated", (25, 30)),
            ("extreme", (30, 100)),
        ]:
            bucket_df = self.df[(self.df['vix_at_entry'] >= low) & (self.df['vix_at_entry'] < high)]
            if len(bucket_df) > 0:
                bucket_wins = len(bucket_df[bucket_df['pnl_pct'] > 0])
                vix_buckets[bucket_name] = {
                    'trades': len(bucket_df),
                    'win_rate': bucket_wins / len(bucket_df) * 100,
                    'avg_pnl': bucket_df['pnl_pct'].mean()
                }

        return {
            "high_vix_win_rate": high_vix_wr,
            "high_vix_trades": high_vix_total,
            "low_vix_win_rate": low_vix_wr,
            "low_vix_trades": low_vix_total,
            "vix_buckets": vix_buckets,
            "recommendation": "FILTER_BY_VIX" if high_vix_wr > low_vix_wr + 5 else "VIX_NEUTRAL"
        }

    def _analyze_ibs_thresholds(self) -> Dict:
        """Analyze IBS threshold effectiveness"""
        # For long trades
        long_df = self.df[self.df['direction'] == 'long']

        ibs_buckets = {}
        for bucket_name, (low, high) in [
            ("extreme_low", (0, 0.1)),
            ("very_low", (0.1, 0.15)),
            ("low", (0.15, 0.2)),
            ("normal_low", (0.2, 0.3)),
        ]:
            bucket_df = long_df[(long_df['ibs_at_entry'] >= low) & (long_df['ibs_at_entry'] < high)]
            if len(bucket_df) > 0:
                bucket_wins = len(bucket_df[bucket_df['pnl_pct'] > 0])
                ibs_buckets[bucket_name] = {
                    'trades': len(bucket_df),
                    'win_rate': bucket_wins / len(bucket_df) * 100,
                    'avg_pnl': bucket_df['pnl_pct'].mean()
                }

        # Find optimal threshold
        best_bucket = max(ibs_buckets.items(), key=lambda x: x[1]['win_rate']) if ibs_buckets else None

        return {
            "long_ibs_buckets": ibs_buckets,
            "best_ibs_range": best_bucket[0] if best_bucket else None,
            "best_win_rate": best_bucket[1]['win_rate'] if best_bucket else 0,
            "current_threshold": 0.15,
            "recommendation": "TIGHTEN_IBS" if best_bucket and best_bucket[0] == "extreme_low" else "KEEP_IBS"
        }

    def _analyze_rsi_thresholds(self) -> Dict:
        """Analyze RSI threshold effectiveness"""
        long_df = self.df[self.df['direction'] == 'long']

        rsi_buckets = {}
        for bucket_name, (low, high) in [
            ("extreme_low", (0, 5)),
            ("very_low", (5, 10)),
            ("low", (10, 15)),
            ("oversold", (15, 20)),
            ("normal", (20, 30)),
        ]:
            bucket_df = long_df[(long_df['rsi_at_entry'] >= low) & (long_df['rsi_at_entry'] < high)]
            if len(bucket_df) > 0:
                bucket_wins = len(bucket_df[bucket_df['pnl_pct'] > 0])
                rsi_buckets[bucket_name] = {
                    'trades': len(bucket_df),
                    'win_rate': bucket_wins / len(bucket_df) * 100,
                    'avg_pnl': bucket_df['pnl_pct'].mean()
                }

        best_bucket = max(rsi_buckets.items(), key=lambda x: x[1]['win_rate']) if rsi_buckets else None

        return {
            "long_rsi_buckets": rsi_buckets,
            "best_rsi_range": best_bucket[0] if best_bucket else None,
            "best_win_rate": best_bucket[1]['win_rate'] if best_bucket else 0,
            "current_threshold": 10,
            "recommendation": "ADJUST_RSI" if best_bucket else "KEEP_RSI"
        }

    def _analyze_timing(self) -> Dict:
        """Analyze timing patterns"""
        # Add day of week
        self.df['entry_day'] = pd.to_datetime(self.df['entry_date']).dt.dayofweek

        day_stats = {}
        for day in range(5):
            day_df = self.df[self.df['entry_day'] == day]
            if len(day_df) > 0:
                day_wins = len(day_df[day_df['pnl_pct'] > 0])
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day]
                day_stats[day_name] = {
                    'trades': len(day_df),
                    'win_rate': day_wins / len(day_df) * 100,
                    'avg_pnl': day_df['pnl_pct'].mean()
                }

        best_day = max(day_stats.items(), key=lambda x: x[1]['win_rate']) if day_stats else None
        worst_day = min(day_stats.items(), key=lambda x: x[1]['win_rate']) if day_stats else None

        return {
            "day_stats": day_stats,
            "best_day": best_day[0] if best_day else None,
            "worst_day": worst_day[0] if worst_day else None,
        }

    def _analyze_exits(self) -> Dict:
        """Analyze exit strategies"""
        exit_stats = {}
        for reason in self.df['exit_reason'].unique():
            reason_df = self.df[self.df['exit_reason'] == reason]
            wins = len(reason_df[reason_df['pnl_pct'] > 0])
            exit_stats[reason] = {
                'trades': len(reason_df),
                'win_rate': wins / len(reason_df) * 100 if len(reason_df) > 0 else 0,
                'avg_pnl': reason_df['pnl_pct'].mean()
            }

        # Analyze hold periods
        hold_analysis = {}
        for days in range(1, 6):
            day_df = self.df[self.df['hold_days'] == days]
            if len(day_df) > 0:
                wins = len(day_df[day_df['pnl_pct'] > 0])
                hold_analysis[f"{days}_day"] = {
                    'trades': len(day_df),
                    'win_rate': wins / len(day_df) * 100,
                    'avg_pnl': day_df['pnl_pct'].mean()
                }

        return {
            "exit_stats": exit_stats,
            "hold_analysis": hold_analysis,
            "avg_hold_days": self.df['hold_days'].mean(),
            "optimal_hold": max(hold_analysis.items(), key=lambda x: x[1]['win_rate'])[0] if hold_analysis else None
        }

    def _generate_improvements(self) -> List[Dict]:
        """Generate specific improvement suggestions"""
        improvements = []

        # 1. VIX Filter improvement
        vix_analysis = self._analyze_vix_filter()
        if vix_analysis['high_vix_win_rate'] > vix_analysis['low_vix_win_rate'] + 10:
            improvements.append({
                "area": "VIX_FILTER",
                "suggestion": "Require VIX > 20 for A+ signals",
                "expected_improvement": f"+{vix_analysis['high_vix_win_rate'] - vix_analysis['low_vix_win_rate']:.1f}% win rate",
                "trade_off": f"Fewer signals ({vix_analysis['high_vix_trades']} vs {vix_analysis['low_vix_trades']})",
                "priority": "HIGH"
            })

        # 2. IBS threshold
        ibs_analysis = self._analyze_ibs_thresholds()
        if ibs_analysis['best_ibs_range'] == 'extreme_low':
            improvements.append({
                "area": "IBS_THRESHOLD",
                "suggestion": "Tighten IBS extreme threshold to 0.10 (from 0.15)",
                "expected_improvement": f"Better signal quality",
                "trade_off": "Fewer but higher quality signals",
                "priority": "MEDIUM"
            })

        # 3. Exit strategy
        exit_analysis = self._analyze_exits()
        if exit_analysis['exit_stats'].get('stop_loss', {}).get('trades', 0) > len(self.df) * 0.3:
            improvements.append({
                "area": "STOP_LOSS",
                "suggestion": "Widen stop loss from 2% to 2.5%",
                "expected_improvement": "Fewer stopped out trades",
                "trade_off": "Larger individual losses when stopped",
                "priority": "MEDIUM"
            })

        # 4. Combo signal requirement
        signal_analysis = self._analyze_signals()
        combo_signal = next((s for s in signal_analysis['all_signals'].items()
                           if 'combo' in s[0]), None)
        if combo_signal and combo_signal[1]['win_rate'] > self.summary['win_rate'] + 5:
            improvements.append({
                "area": "COMBO_REQUIREMENT",
                "suggestion": "Require IBS+RSI combo for A+ grade",
                "expected_improvement": f"+{combo_signal[1]['win_rate'] - self.summary['win_rate']:.1f}% win rate",
                "trade_off": "Fewer A+ signals",
                "priority": "HIGH"
            })

        # 5. Score threshold
        overall = self._analyze_overall()
        if overall['win_rate'] < 65:
            improvements.append({
                "area": "SCORE_THRESHOLD",
                "suggestion": "Increase minimum score for A+ from 80 to 85",
                "expected_improvement": "Higher quality signals only",
                "trade_off": "Fewer alerts",
                "priority": "HIGH"
            })

        return improvements

    def get_improvement_code(self) -> str:
        """Generate code changes for improvements"""
        code_changes = []

        for imp in self._generate_improvements():
            if imp['area'] == 'VIX_FILTER':
                code_changes.append("""
# VIX Filter Enhancement
# In opportunity_scorer.py, add stricter VIX requirement for A+ grade:
# Change line in get_grade() method:
if score >= 80 and vix_passed:  # Require VIX filter for A+
    return OpportunityGrade.A_PLUS
elif score >= 80:
    return OpportunityGrade.A  # Downgrade to A if no VIX
""")

            elif imp['area'] == 'IBS_THRESHOLD':
                code_changes.append("""
# IBS Threshold Tightening
# In signal_detector.py, change THRESHOLDS:
"ibs_extreme_low": 0.10,  # Was 0.15
""")

            elif imp['area'] == 'SCORE_THRESHOLD':
                code_changes.append("""
# Score Threshold Increase
# In opportunity_scorer.py, get_grade() method:
if score >= 85:  # Was 80
    return OpportunityGrade.A_PLUS
""")

        return "\n".join(code_changes)


def generate_performance_report(trades, summary) -> str:
    """Generate a human-readable performance report"""
    analyzer = PerformanceAnalyzer(trades, summary)
    analysis = analyzer.analyze_all()

    report = []
    report.append("=" * 70)
    report.append("SPY OPPORTUNITY AGENT - PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 70)

    # Overall
    overall = analysis['overall_performance']
    report.append(f"\n## OVERALL PERFORMANCE")
    report.append(f"   Win Rate: {overall['win_rate']:.1f}% (Target: {overall['target_win_rate']}%)")
    report.append(f"   Gap: {overall['gap']:+.1f}%")
    report.append(f"   Assessment: {overall['assessment']}")

    # Signal Analysis
    signal = analysis['signal_analysis']
    report.append(f"\n## BEST PERFORMING SIGNALS")
    for sig_name, sig_data in signal['best_signals'][:3]:
        report.append(f"   {sig_name}: {sig_data['win_rate']:.1f}% ({sig_data['total_trades']} trades)")

    # VIX Analysis
    vix = analysis['vix_filter_analysis']
    report.append(f"\n## VIX FILTER ANALYSIS")
    report.append(f"   High VIX (>20): {vix['high_vix_win_rate']:.1f}% win rate ({vix['high_vix_trades']} trades)")
    report.append(f"   Low VIX (<=20): {vix['low_vix_win_rate']:.1f}% win rate ({vix['low_vix_trades']} trades)")
    report.append(f"   Recommendation: {vix['recommendation']}")

    # Improvements
    improvements = analysis['improvements']
    report.append(f"\n## RECOMMENDED IMPROVEMENTS")
    for i, imp in enumerate(improvements, 1):
        report.append(f"   {i}. [{imp['priority']}] {imp['area']}")
        report.append(f"      Suggestion: {imp['suggestion']}")
        report.append(f"      Expected: {imp['expected_improvement']}")
        report.append(f"      Trade-off: {imp['trade_off']}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)
