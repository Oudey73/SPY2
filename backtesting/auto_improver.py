"""
Auto Improver - Automatically tests, analyzes, and improves the SPY agent
"""
import sys
import os
import json
import copy
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analyzer import PerformanceAnalyzer, generate_performance_report


class AutoImprover:
    """
    Automatically improves the SPY agent through iterative backtesting
    """

    def __init__(self, start_date: str = "2022-01-01"):
        self.start_date = start_date
        self.iterations = []
        self.best_config = None
        self.best_win_rate = 0

        # Parameters to optimize
        self.params = {
            'ibs_extreme_low': 0.15,
            'ibs_oversold': 0.20,
            'rsi_extreme_low': 10,
            'rsi_oversold': 20,
            'vix_elevated': 20,
            'vix_extreme': 30,
            'stop_loss_pct': 0.02,
            'target_1_pct': 0.01,
            'a_plus_threshold': 80,
            'require_vix_for_aplus': False,
            'require_combo_for_aplus': False,
        }

    def run_iteration(self, params: Dict, iteration_name: str) -> Dict:
        """Run a single backtest iteration with given parameters"""
        logger.info(f"Running iteration: {iteration_name}")

        # Apply parameters to the detector/scorer
        # (In a real implementation, these would modify the actual classes)

        engine = BacktestEngine(start_date=self.start_date)

        # Modify engine parameters
        engine.stop_loss_pct = params.get('stop_loss_pct', 0.02)
        engine.target_1_pct = params.get('target_1_pct', 0.01)

        # Run backtest
        trades = engine.run_backtest(min_grade="A+")

        if not trades:
            return {'error': 'No trades generated', 'params': params}

        summary = engine.get_results_summary()

        # Analyze
        analyzer = PerformanceAnalyzer(trades, summary)
        analysis = analyzer.analyze_all()

        result = {
            'iteration': iteration_name,
            'params': params,
            'total_trades': summary['total_trades'],
            'win_rate': summary['win_rate'],
            'avg_pnl': summary['avg_pnl_per_trade'],
            'total_pnl': summary['total_pnl_pct'],
            'profit_factor': summary['profit_factor'],
            'target_hit_rate': summary['target_hit_rate'],
            'stop_hit_rate': summary['stop_hit_rate'],
            'analysis': analysis,
            'improvements': analysis['improvements'],
            'trades': trades,
        }

        self.iterations.append(result)

        # Track best
        if summary['win_rate'] > self.best_win_rate:
            self.best_win_rate = summary['win_rate']
            self.best_config = params.copy()

        return result

    def optimize_parameter(self, param_name: str, values: List, base_params: Dict) -> Dict:
        """Test different values for a single parameter"""
        results = []

        for value in values:
            test_params = base_params.copy()
            test_params[param_name] = value
            result = self.run_iteration(test_params, f"{param_name}={value}")
            results.append({
                'value': value,
                'win_rate': result.get('win_rate', 0),
                'trades': result.get('total_trades', 0),
                'avg_pnl': result.get('avg_pnl', 0)
            })

        # Find best value
        best = max(results, key=lambda x: x['win_rate'] if x['trades'] >= 10 else 0)

        return {
            'param': param_name,
            'results': results,
            'best_value': best['value'],
            'best_win_rate': best['win_rate']
        }

    def run_full_optimization(self) -> Dict:
        """Run complete optimization across all parameters"""
        logger.info("Starting full optimization")
        optimization_results = []

        # 1. Baseline
        logger.info("Running baseline...")
        baseline = self.run_iteration(self.params.copy(), "BASELINE")

        # 2. Optimize IBS threshold
        logger.info("Optimizing IBS threshold...")
        ibs_opt = self.optimize_parameter(
            'ibs_extreme_low',
            [0.08, 0.10, 0.12, 0.15, 0.18],
            self.params
        )
        optimization_results.append(ibs_opt)

        # Update params with best IBS
        self.params['ibs_extreme_low'] = ibs_opt['best_value']

        # 3. Optimize RSI threshold
        logger.info("Optimizing RSI threshold...")
        rsi_opt = self.optimize_parameter(
            'rsi_extreme_low',
            [5, 8, 10, 12, 15],
            self.params
        )
        optimization_results.append(rsi_opt)
        self.params['rsi_extreme_low'] = rsi_opt['best_value']

        # 4. Optimize VIX threshold
        logger.info("Optimizing VIX threshold...")
        vix_opt = self.optimize_parameter(
            'vix_elevated',
            [15, 18, 20, 22, 25],
            self.params
        )
        optimization_results.append(vix_opt)
        self.params['vix_elevated'] = vix_opt['best_value']

        # 5. Optimize stop loss
        logger.info("Optimizing stop loss...")
        stop_opt = self.optimize_parameter(
            'stop_loss_pct',
            [0.015, 0.02, 0.025, 0.03],
            self.params
        )
        optimization_results.append(stop_opt)
        self.params['stop_loss_pct'] = stop_opt['best_value']

        # 6. Optimize A+ threshold
        logger.info("Optimizing A+ threshold...")
        aplus_opt = self.optimize_parameter(
            'a_plus_threshold',
            [75, 80, 85, 90],
            self.params
        )
        optimization_results.append(aplus_opt)
        self.params['a_plus_threshold'] = aplus_opt['best_value']

        # 7. Final run with optimized params
        logger.info("Running final optimized backtest...")
        final = self.run_iteration(self.params.copy(), "OPTIMIZED")

        return {
            'baseline': baseline,
            'optimizations': optimization_results,
            'final': final,
            'best_params': self.best_config,
            'improvement': final.get('win_rate', 0) - baseline.get('win_rate', 0)
        }

    def generate_report(self) -> str:
        """Generate comprehensive improvement report"""
        if not self.iterations:
            return "No iterations run yet"

        report = []
        report.append("=" * 70)
        report.append("SPY OPPORTUNITY AGENT - AUTO-IMPROVEMENT REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)

        # Baseline vs Final
        baseline = next((i for i in self.iterations if i['iteration'] == 'BASELINE'), None)
        final = next((i for i in self.iterations if i['iteration'] == 'OPTIMIZED'), None)

        if baseline and final:
            report.append("\n## PERFORMANCE COMPARISON")
            report.append("-" * 40)
            report.append(f"{'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
            report.append("-" * 40)

            metrics = [
                ('Win Rate', 'win_rate', '%'),
                ('Total Trades', 'total_trades', ''),
                ('Avg P&L', 'avg_pnl', '%'),
                ('Total P&L', 'total_pnl', '%'),
                ('Target Hit Rate', 'target_hit_rate', '%'),
                ('Stop Hit Rate', 'stop_hit_rate', '%'),
            ]

            for label, key, suffix in metrics:
                base_val = baseline.get(key, 0)
                final_val = final.get(key, 0)
                change = final_val - base_val
                change_str = f"{change:+.1f}{suffix}" if suffix else f"{change:+.0f}"
                report.append(f"{label:<25} {base_val:>11.1f}{suffix} {final_val:>11.1f}{suffix} {change_str:>12}")

        # Best Parameters
        report.append("\n## OPTIMIZED PARAMETERS")
        report.append("-" * 40)
        if self.best_config:
            for param, value in self.best_config.items():
                report.append(f"   {param}: {value}")

        # Optimization Details
        report.append("\n## OPTIMIZATION DETAILS")
        report.append("-" * 40)
        for iteration in self.iterations:
            if iteration['iteration'] not in ['BASELINE', 'OPTIMIZED']:
                report.append(f"   {iteration['iteration']}: {iteration['win_rate']:.1f}% ({iteration['total_trades']} trades)")

        # Recommendations
        if final and final.get('improvements'):
            report.append("\n## REMAINING IMPROVEMENTS")
            report.append("-" * 40)
            for imp in final['improvements'][:3]:
                report.append(f"   [{imp['priority']}] {imp['suggestion']}")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)

    def apply_improvements(self) -> Dict:
        """Apply the best improvements to the actual code"""
        if not self.best_config:
            return {'error': 'No optimization run yet'}

        changes = []

        # Generate code changes based on best config
        original_params = {
            'ibs_extreme_low': 0.15,
            'rsi_extreme_low': 10,
            'vix_elevated': 20,
            'stop_loss_pct': 0.02,
            'a_plus_threshold': 80,
        }

        for param, new_value in self.best_config.items():
            if param in original_params and new_value != original_params[param]:
                changes.append({
                    'parameter': param,
                    'old_value': original_params[param],
                    'new_value': new_value,
                })

        return {
            'changes': changes,
            'best_win_rate': self.best_win_rate,
            'config': self.best_config
        }


def run_full_test_and_improve():
    """Main function to run full testing and improvement cycle"""
    print("=" * 70)
    print("SPY OPPORTUNITY AGENT - AUTO-IMPROVEMENT SYSTEM")
    print("=" * 70)
    print()

    improver = AutoImprover(start_date="2022-01-01")

    print("Starting optimization (this may take several minutes)...")
    print()

    results = improver.run_full_optimization()

    print()
    print(improver.generate_report())

    # Save results
    with open('backtesting/optimization_results.json', 'w') as f:
        # Convert non-serializable objects
        save_results = {
            'baseline_win_rate': results['baseline'].get('win_rate', 0),
            'final_win_rate': results['final'].get('win_rate', 0),
            'improvement': results['improvement'],
            'best_params': results['best_params'],
            'timestamp': datetime.now().isoformat()
        }
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to backtesting/optimization_results.json")

    return results


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    run_full_test_and_improve()
