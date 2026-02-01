"""
Trade Logger for SPY Options Agent
JSON-based trade log with full context
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import pytz

SAUDI_TZ = pytz.timezone("Asia/Riyadh")

DEFAULT_LOG_PATH = "data/trade_log.json"


class TradeLogger:
    """
    Logs trades to a JSON file with full context including
    regime, IV, Greeks, legs, P&L, and exit reason.
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self.log_path = log_path
        self._ensure_file()

    def _ensure_file(self):
        """Create the log file and directory if they don't exist."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def _load(self) -> List[Dict]:
        """Load all trades from the log file."""
        try:
            with open(self.log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, trades: List[Dict]):
        """Save all trades to the log file."""
        with open(self.log_path, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    def log_entry(
        self,
        trade_plan: Dict,
        regime: Optional[Dict] = None,
        iv_data: Optional[Dict] = None,
        signals: Optional[List[Dict]] = None,
        risk_decision: Optional[Dict] = None,
        notes: str = "",
    ) -> str:
        """
        Log a new trade entry.

        Returns:
            trade_id (UUID string)
        """
        trade_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")

        entry = {
            "trade_id": trade_id,
            "status": "open",
            "entry_timestamp": timestamp,
            "exit_timestamp": None,
            "trade_plan": trade_plan,
            "regime": regime,
            "iv_data": iv_data,
            "signals": signals,
            "risk_decision": risk_decision,
            "entry_price": None,
            "exit_price": None,
            "pnl": None,
            "pnl_pct": None,
            "exit_reason": None,
            "notes": notes,
        }

        trades = self._load()
        trades.append(entry)
        self._save(trades)

        logger.info(f"Trade logged: {trade_id} ({trade_plan.get('strategy', 'unknown')})")
        return trade_id

    def log_exit(
        self,
        trade_id: str,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        exit_reason: str = "manual",
    ) -> bool:
        """
        Log a trade exit.

        Returns:
            True if trade was found and updated
        """
        trades = self._load()
        timestamp = datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")

        for trade in trades:
            if trade["trade_id"] == trade_id:
                trade["status"] = "closed"
                trade["exit_timestamp"] = timestamp
                trade["exit_price"] = exit_price
                trade["pnl"] = pnl
                trade["pnl_pct"] = pnl_pct
                trade["exit_reason"] = exit_reason
                self._save(trades)
                logger.info(f"Trade closed: {trade_id} P&L={pnl} ({exit_reason})")
                return True

        logger.warning(f"Trade {trade_id} not found")
        return False

    def get_open_positions(self) -> List[Dict]:
        """Get all currently open trades."""
        trades = self._load()
        return [t for t in trades if t["status"] == "open"]

    def get_all_trades(self) -> List[Dict]:
        """Get all trades (open and closed)."""
        return self._load()

    def get_closed_trades(self) -> List[Dict]:
        """Get all closed trades."""
        trades = self._load()
        return [t for t in trades if t["status"] == "closed"]

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get a specific trade by ID."""
        trades = self._load()
        for t in trades:
            if t["trade_id"] == trade_id:
                return t
        return None
