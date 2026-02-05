"""
Opportunity Logger
Persists all detected opportunities and tracks portfolio positions/P&L.
Bridges to TradeLogger for active position tracking.
"""
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .trade_logger import TradeLogger

DEFAULT_LOG_PATH = "data/opportunities.json"


class OpportunityLogger:
    """Logs all opportunities and tracks active/closed positions."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH, trade_logger: TradeLogger = None):
        self.log_path = log_path
        self.trade_logger = trade_logger or TradeLogger()
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def _load(self) -> List[Dict]:
        try:
            with open(self.log_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.critical(f"CORRUPT opportunities file: {self.log_path}")
            backup = self.log_path + ".bak"
            if os.path.exists(backup):
                logger.info(f"Recovering from backup: {backup}")
                try:
                    with open(backup, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            logger.critical("No valid backup — returning empty list (DATA LOSS)")
            return []
        except FileNotFoundError:
            return []

    def _save(self, entries: List[Dict]):
        # Backup current file before overwriting
        if os.path.exists(self.log_path):
            backup = self.log_path + ".bak"
            try:
                shutil.copy2(self.log_path, backup)
            except Exception:
                pass
        # Atomic write: write to temp file, then replace
        tmp = self.log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2, default=str)
        os.replace(tmp, self.log_path)

    def log(
        self,
        opportunity_dict: Dict,
        regime_dict: Optional[Dict] = None,
        trade_plan_dict: Optional[Dict] = None,
        risk_decision_dict: Optional[Dict] = None,
    ) -> str:
        """Log an opportunity snapshot. Returns the opp_id."""
        opp_id = opportunity_dict.get("opp_id", "OPP-????")
        entry = {
            "opp_id": opp_id,
            "status": "logged",
            "timestamp": datetime.utcnow().isoformat(),
            "opportunity": opportunity_dict,
            "regime": regime_dict,
            "trade_plan": trade_plan_dict,
            "risk_decision": risk_decision_dict,
            "entry_price": None,
            "exit_price": None,
            "contracts": None,
            "pnl": None,
            "trade_id": None,
        }
        entries = self._load()
        entries.append(entry)
        self._save(entries)
        logger.info(f"Logged opportunity {opp_id}")
        return opp_id

    def get(self, opp_id: str) -> Optional[Dict]:
        """Retrieve an opportunity by ID."""
        for entry in self._load():
            if entry["opp_id"] == opp_id:
                return entry
        return None

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Return the last N logged opportunities."""
        return self._load()[-n:]

    def activate(self, opp_id: str, entry_price: float, contracts: int) -> bool:
        """Mark an opportunity as active and create a matching TradeLogger entry."""
        entries = self._load()
        for entry in entries:
            if entry["opp_id"] == opp_id:
                if entry["status"] != "logged":
                    logger.error(f"Opportunity {opp_id} is already {entry['status']}")
                    return False

                # Create entry in TradeLogger
                trade_id = self.trade_logger.log_entry(
                    trade_plan=entry.get("trade_plan") or {},
                    regime=entry.get("regime"),
                    signals=[s for s in (entry.get("opportunity", {}).get("signals") or [])],
                    risk_decision=entry.get("risk_decision"),
                    notes=f"Activated from {opp_id} @ ${entry_price:.2f} x{contracts}",
                )

                entry["status"] = "active"
                entry["entry_price"] = entry_price
                entry["contracts"] = contracts
                entry["trade_id"] = trade_id
                entry["activated_at"] = datetime.utcnow().isoformat()
                self._save(entries)
                logger.info(f"Activated {opp_id}: {contracts} contracts @ ${entry_price:.2f} (trade_id={trade_id})")
                return True

        logger.error(f"Opportunity {opp_id} not found")
        return False

    def close(self, opp_id: str, exit_price: float, exit_reason: str = "") -> Optional[Dict]:
        """Close an active position and calculate P&L."""
        entries = self._load()
        for entry in entries:
            if entry["opp_id"] == opp_id:
                if entry["status"] != "active":
                    logger.error(f"Opportunity {opp_id} is not active (status={entry['status']})")
                    return None

                direction = entry["opportunity"].get("direction", "long")
                entry_price = entry["entry_price"]
                contracts = entry["contracts"]

                if direction == "long":
                    pnl_per = exit_price - entry_price
                else:
                    pnl_per = entry_price - exit_price

                total_pnl = pnl_per * contracts * 100  # options: 100 shares per contract

                entry["status"] = "closed"
                entry["exit_price"] = exit_price
                entry["pnl"] = total_pnl
                entry["exit_reason"] = exit_reason
                entry["closed_at"] = datetime.utcnow().isoformat()

                # Also close in TradeLogger
                if entry.get("trade_id"):
                    pnl_pct = (pnl_per / entry_price * 100) if entry_price else None
                    self.trade_logger.log_exit(
                        trade_id=entry["trade_id"],
                        exit_price=exit_price,
                        pnl=total_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason or "manual",
                    )

                self._save(entries)
                logger.info(f"Closed {opp_id}: P&L ${total_pnl:+,.2f} ({exit_reason})")
                return entry

        logger.error(f"Opportunity {opp_id} not found")
        return None

    def get_portfolio(self) -> List[Dict]:
        """Return all active positions."""
        return [e for e in self._load() if e["status"] == "active"]

    def get_summary(self) -> str:
        """Formatted portfolio summary: open positions, total P&L, stats."""
        entries = self._load()
        active = [e for e in entries if e["status"] == "active"]
        closed = [e for e in entries if e["status"] == "closed"]

        lines = ["=== Portfolio Summary ==="]

        if active:
            lines.append(f"\nOpen Positions ({len(active)}):")
            for pos in active:
                opp = pos["opportunity"]
                lines.append(
                    f"  {pos['opp_id']} | {opp.get('direction','?').upper()} "
                    f"| {pos['contracts']}x @ ${pos['entry_price']:.2f} "
                    f"| Score {opp.get('score','?')}/{opp.get('grade','?')}"
                )
        else:
            lines.append("\nNo open positions.")

        if closed:
            total_pnl = sum(e.get("pnl", 0) or 0 for e in closed)
            wins = [e for e in closed if (e.get("pnl") or 0) > 0]
            losses = [e for e in closed if (e.get("pnl") or 0) <= 0]
            win_rate = len(wins) / len(closed) * 100 if closed else 0

            lines.append(f"\nClosed Trades: {len(closed)}")
            lines.append(f"Win Rate: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)")
            lines.append(f"Total P&L: ${total_pnl:+,.2f}")

            if wins:
                avg_win = sum(e["pnl"] for e in wins) / len(wins)
                lines.append(f"Avg Win: ${avg_win:+,.2f}")
            if losses:
                avg_loss = sum(e["pnl"] for e in losses) / len(losses)
                lines.append(f"Avg Loss: ${avg_loss:+,.2f}")
        else:
            lines.append("\nNo closed trades yet.")

        lines.append(f"\nTotal Logged: {len(entries)}")
        return "\n".join(lines)
