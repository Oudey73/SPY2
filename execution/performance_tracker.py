"""
Performance Tracker for SPY Options Agent
Calculates win rate, expectancy, Sharpe, drawdown, and per-regime breakdowns
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger
import numpy as np


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    best_trade: float
    worst_trade: float

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
        }


class PerformanceTracker:
    """
    Calculates trading performance metrics from trade log data.
    """

    def calculate_metrics(
        self, trades: List[Dict], account_value: float = 100000
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from a list of closed trades.

        Args:
            trades: List of trade dicts with 'pnl' field
            account_value: Starting account value for drawdown calc

        Returns:
            PerformanceMetrics
        """
        closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]

        if not closed:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Expectancy = (Win% × Avg Win) + (Loss% × Avg Loss)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Profit factor = Gross Wins / |Gross Losses|
        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        total_pnl = sum(pnls)

        # Max drawdown
        max_dd, max_dd_pct = self._calculate_drawdown(pnls, account_value)

        # Sharpe ratio (annualized, assuming ~252 trading days)
        sharpe = self._calculate_sharpe(pnls)

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            expectancy=float(expectancy),
            profit_factor=float(profit_factor),
            total_pnl=float(total_pnl),
            max_drawdown=float(max_dd),
            max_drawdown_pct=float(max_dd_pct),
            sharpe_ratio=sharpe,
            best_trade=float(max(pnls)),
            worst_trade=float(min(pnls)),
        )

    def analyze_by_regime(self, trades: List[Dict]) -> Dict[str, Dict]:
        """
        Break down performance by regime and strategy.

        Returns:
            Dict mapping regime_name -> {strategy_name -> metrics_dict}
        """
        closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]
        if not closed:
            return {}

        by_regime = {}
        for trade in closed:
            regime = "unknown"
            if trade.get("regime"):
                regime = trade["regime"].get("regime", "unknown") if isinstance(trade["regime"], dict) else str(trade["regime"])

            strategy = "unknown"
            if trade.get("trade_plan"):
                strategy = trade["trade_plan"].get("strategy", "unknown")

            key = f"{regime}"
            if key not in by_regime:
                by_regime[key] = {}
            if strategy not in by_regime[key]:
                by_regime[key][strategy] = []
            by_regime[key][strategy].append(trade)

        result = {}
        for regime, strats in by_regime.items():
            result[regime] = {}
            for strat, strat_trades in strats.items():
                pnls = [t["pnl"] for t in strat_trades]
                wins = [p for p in pnls if p > 0]
                result[regime][strat] = {
                    "trades": len(pnls),
                    "win_rate": len(wins) / len(pnls) if pnls else 0,
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls) if pnls else 0,
                }

        return result

    def generate_report(self, trades: List[Dict], account_value: float = 100000) -> str:
        """Generate a formatted performance summary."""
        metrics = self.calculate_metrics(trades, account_value)
        regime_breakdown = self.analyze_by_regime(trades)

        lines = [
            "=" * 50,
            "PERFORMANCE REPORT",
            "=" * 50,
            f"Total Trades:    {metrics.total_trades}",
            f"Win Rate:        {metrics.win_rate:.1%}",
            f"Avg Win:         ${metrics.avg_win:,.2f}",
            f"Avg Loss:        ${metrics.avg_loss:,.2f}",
            f"Expectancy:      ${metrics.expectancy:,.2f}",
            f"Profit Factor:   {metrics.profit_factor:.2f}",
            f"Total P&L:       ${metrics.total_pnl:,.2f}",
            f"Max Drawdown:    ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1%})",
            f"Sharpe Ratio:    {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "Sharpe Ratio:    N/A",
            f"Best Trade:      ${metrics.best_trade:,.2f}",
            f"Worst Trade:     ${metrics.worst_trade:,.2f}",
        ]

        if regime_breakdown:
            lines.append("")
            lines.append("-" * 50)
            lines.append("BY REGIME")
            lines.append("-" * 50)
            for regime, strats in regime_breakdown.items():
                lines.append(f"\n  {regime}:")
                for strat, data in strats.items():
                    lines.append(
                        f"    {strat}: {data['trades']} trades, "
                        f"{data['win_rate']:.0%} win rate, "
                        f"${data['total_pnl']:,.2f} P&L"
                    )

        lines.append("=" * 50)
        return "\n".join(lines)

    def _calculate_drawdown(self, pnls: List[float], account_value: float) -> tuple:
        """Calculate max drawdown in dollars and percent."""
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = float(abs(min(drawdowns))) if len(drawdowns) > 0 else 0
        max_dd_pct = max_dd / account_value if account_value > 0 else 0
        return max_dd, max_dd_pct

    def _calculate_sharpe(self, pnls: List[float], risk_free_rate: float = 0.05) -> Optional[float]:
        """Calculate annualized Sharpe ratio."""
        if len(pnls) < 2:
            return None
        returns = np.array(pnls)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return None
        # Annualize assuming ~252 trading days
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
        return float(sharpe)

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, expectancy=0,
            profit_factor=0, total_pnl=0, max_drawdown=0,
            max_drawdown_pct=0, sharpe_ratio=None,
            best_trade=0, worst_trade=0,
        )
