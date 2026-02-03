"""
Exit Monitor for Bar-Based Position Management
Tracks positions and generates exit alerts based on:
- 3-bar profit rule: Exit 50% if profitable after 3 bars
- 5-bar no-profit rule: Exit 100% if not profitable after 5 bars
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import pytz

# Saudi Arabia timezone for timestamps
SAUDI_TZ = pytz.timezone("Asia/Riyadh")


def now_saudi() -> str:
    return datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S AST")


class AlertType(Enum):
    """Types of exit alerts"""
    PARTIAL_EXIT = "partial_exit"  # Take partial profits
    FULL_EXIT = "full_exit"        # Exit entire position
    STOP_LOSS = "stop_loss"        # Stop loss triggered
    TARGET_HIT = "target_hit"      # Profit target reached


@dataclass
class ExitAlert:
    """Represents an exit alert for a tracked position"""
    opp_id: str
    alert_type: AlertType
    exit_percentage: int  # 50 or 100
    reason: str
    entry_price: float
    current_price: float
    bars_held: int
    pnl_percent: float
    timestamp: str = field(default_factory=now_saudi)

    def to_dict(self) -> dict:
        return {
            "opp_id": self.opp_id,
            "alert_type": self.alert_type.value,
            "exit_percentage": self.exit_percentage,
            "reason": self.reason,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "bars_held": self.bars_held,
            "pnl_percent": self.pnl_percent,
            "timestamp": self.timestamp,
        }


@dataclass
class TrackedPosition:
    """Represents a tracked position"""
    opp_id: str
    entry_price: float
    direction: str  # "long" or "short"
    entry_time: str
    bars_held: int = 0
    partial_exit_taken: bool = False
    stop_loss: Optional[float] = None
    targets: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "opp_id": self.opp_id,
            "entry_price": self.entry_price,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "bars_held": self.bars_held,
            "partial_exit_taken": self.partial_exit_taken,
            "stop_loss": self.stop_loss,
            "targets": self.targets,
        }


class ExitMonitor:
    """
    Monitors positions and generates exit alerts based on bar-based rules.

    Exit Rules:
    1. 3-bar profitable: If position is profitable after 3 bars, exit 50%
    2. 5-bar no-profit: If position is NOT profitable after 5 bars, exit 100%
       (Edge has expired - mean reversion should have occurred by now)

    The "bar" count increments each time check_exit_conditions() is called,
    typically once per scan cycle during market hours.
    """

    def __init__(
        self,
        bars_for_partial: int = 3,
        bars_for_full_exit: int = 5,
        profit_threshold_pct: float = 0.0  # Any profit counts
    ):
        """
        Initialize the exit monitor.

        Args:
            bars_for_partial: Number of bars after which to take partial profit if profitable
            bars_for_full_exit: Number of bars after which to exit if not profitable
            profit_threshold_pct: Minimum profit % to count as "profitable" (default: 0)
        """
        self.bars_for_partial = bars_for_partial
        self.bars_for_full_exit = bars_for_full_exit
        self.profit_threshold_pct = profit_threshold_pct

        # Active positions being tracked: opp_id -> TrackedPosition
        self.positions: Dict[str, TrackedPosition] = {}

        # History of completed positions for analysis
        self.completed: List[Dict] = []

    def register_entry(
        self,
        opp_id: str,
        entry_price: float,
        direction: str = "long",
        stop_loss: Optional[float] = None,
        targets: Optional[Dict[str, float]] = None
    ) -> TrackedPosition:
        """
        Register a new position to track.

        Args:
            opp_id: Unique opportunity ID
            entry_price: Entry price
            direction: "long" or "short"
            stop_loss: Optional stop loss price
            targets: Optional dict with target prices {"t1": price, "t2": price, ...}

        Returns:
            TrackedPosition object
        """
        position = TrackedPosition(
            opp_id=opp_id,
            entry_price=entry_price,
            direction=direction.lower(),
            entry_time=now_saudi(),
            bars_held=0,
            partial_exit_taken=False,
            stop_loss=stop_loss,
            targets=targets,
        )

        self.positions[opp_id] = position
        return position

    def remove_position(self, opp_id: str) -> Optional[TrackedPosition]:
        """
        Remove a position from tracking (e.g., after full exit).

        Args:
            opp_id: Opportunity ID to remove

        Returns:
            The removed position, or None if not found
        """
        if opp_id in self.positions:
            position = self.positions.pop(opp_id)
            self.completed.append(position.to_dict())
            return position
        return None

    def _calculate_pnl_percent(
        self,
        position: TrackedPosition,
        current_price: float
    ) -> float:
        """
        Calculate P&L percentage for a position.

        Args:
            position: TrackedPosition object
            current_price: Current market price

        Returns:
            P&L as a percentage (positive = profit, negative = loss)
        """
        if position.direction == "long":
            pnl = (current_price - position.entry_price) / position.entry_price * 100
        else:  # short
            pnl = (position.entry_price - current_price) / position.entry_price * 100

        return round(pnl, 3)

    def check_exit_conditions(
        self,
        opp_id: str,
        current_price: float
    ) -> Optional[ExitAlert]:
        """
        Check exit conditions for a single position and increment bar count.

        Args:
            opp_id: Opportunity ID to check
            current_price: Current market price

        Returns:
            ExitAlert if exit condition met, None otherwise
        """
        if opp_id not in self.positions:
            return None

        position = self.positions[opp_id]

        # Increment bar count
        position.bars_held += 1

        # Calculate P&L
        pnl_pct = self._calculate_pnl_percent(position, current_price)
        is_profitable = pnl_pct > self.profit_threshold_pct

        # Check stop loss first (if defined)
        if position.stop_loss is not None:
            stop_hit = False
            if position.direction == "long" and current_price <= position.stop_loss:
                stop_hit = True
            elif position.direction == "short" and current_price >= position.stop_loss:
                stop_hit = True

            if stop_hit:
                return ExitAlert(
                    opp_id=opp_id,
                    alert_type=AlertType.STOP_LOSS,
                    exit_percentage=100,
                    reason=f"Stop loss hit at ${position.stop_loss:.2f}",
                    entry_price=position.entry_price,
                    current_price=current_price,
                    bars_held=position.bars_held,
                    pnl_percent=pnl_pct,
                )

        # Check target hit (T2 = full exit signal)
        if position.targets and position.targets.get("t2"):
            t2 = position.targets["t2"]
            target_hit = False
            if position.direction == "long" and current_price >= t2:
                target_hit = True
            elif position.direction == "short" and current_price <= t2:
                target_hit = True

            if target_hit:
                return ExitAlert(
                    opp_id=opp_id,
                    alert_type=AlertType.TARGET_HIT,
                    exit_percentage=100,
                    reason=f"Target T2 hit at ${t2:.2f}",
                    entry_price=position.entry_price,
                    current_price=current_price,
                    bars_held=position.bars_held,
                    pnl_percent=pnl_pct,
                )

        # Rule 1: 3 bars + profitable → partial exit (50%)
        if (position.bars_held >= self.bars_for_partial and
            is_profitable and
            not position.partial_exit_taken):

            position.partial_exit_taken = True
            return ExitAlert(
                opp_id=opp_id,
                alert_type=AlertType.PARTIAL_EXIT,
                exit_percentage=50,
                reason=f"3-bar profit rule: {pnl_pct:+.2f}% gain after {position.bars_held} bars",
                entry_price=position.entry_price,
                current_price=current_price,
                bars_held=position.bars_held,
                pnl_percent=pnl_pct,
            )

        # Rule 2: 5 bars + not profitable → full exit (100%)
        if position.bars_held >= self.bars_for_full_exit and not is_profitable:
            return ExitAlert(
                opp_id=opp_id,
                alert_type=AlertType.FULL_EXIT,
                exit_percentage=100,
                reason=f"5-bar no-profit rule: Edge expired, {pnl_pct:+.2f}% after {position.bars_held} bars",
                entry_price=position.entry_price,
                current_price=current_price,
                bars_held=position.bars_held,
                pnl_percent=pnl_pct,
            )

        return None

    def check_all_positions(self, current_price: float) -> List[ExitAlert]:
        """
        Check exit conditions for all tracked positions.

        Args:
            current_price: Current market price

        Returns:
            List of ExitAlert objects for positions that triggered exit conditions
        """
        alerts = []

        # Copy keys to avoid modification during iteration
        opp_ids = list(self.positions.keys())

        for opp_id in opp_ids:
            alert = self.check_exit_conditions(opp_id, current_price)
            if alert:
                alerts.append(alert)

                # Remove position if full exit
                if alert.exit_percentage == 100:
                    self.remove_position(opp_id)

        return alerts

    def get_position(self, opp_id: str) -> Optional[TrackedPosition]:
        """Get a tracked position by ID"""
        return self.positions.get(opp_id)

    def get_all_positions(self) -> List[TrackedPosition]:
        """Get all currently tracked positions"""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of tracked positions"""
        return len(self.positions)


def test_exit_monitor():
    """Test the exit monitor"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "=" * 60)
    print("EXIT MONITOR TEST")
    print("=" * 60)

    monitor = ExitMonitor()

    # Test Case 1: Register a position
    print("\n--- Test 1: Register Position ---")
    pos = monitor.register_entry(
        opp_id="OPP-test1",
        entry_price=580.00,
        direction="long",
        stop_loss=570.00,
        targets={"t1": 590.00, "t2": 600.00}
    )
    print(f"Registered: {pos.opp_id} at ${pos.entry_price}")
    print(f"Active positions: {monitor.get_position_count()}")

    # Test Case 2: Simulate bars passing with profit
    print("\n--- Test 2: 3-Bar Profit Rule ---")
    for bar in range(1, 5):
        # Price increases each bar
        price = 580.00 + (bar * 2)
        alert = monitor.check_exit_conditions("OPP-test1", price)
        print(f"Bar {bar}: Price ${price:.2f}, Alert: {alert}")
        if alert:
            print(f"  -> {alert.alert_type.value}: {alert.reason}")
            print(f"  -> Exit {alert.exit_percentage}% at {alert.pnl_percent:+.2f}%")

    # Test Case 3: Register a losing position
    print("\n--- Test 3: 5-Bar No-Profit Rule ---")
    pos2 = monitor.register_entry(
        opp_id="OPP-test2",
        entry_price=590.00,
        direction="long"
    )
    print(f"Registered: {pos2.opp_id} at ${pos2.entry_price}")

    for bar in range(1, 7):
        # Price stays flat or goes down slightly
        price = 590.00 - (bar * 0.5)
        alert = monitor.check_exit_conditions("OPP-test2", price)
        print(f"Bar {bar}: Price ${price:.2f}, Alert: {alert}")
        if alert:
            print(f"  -> {alert.alert_type.value}: {alert.reason}")
            print(f"  -> Exit {alert.exit_percentage}% at {alert.pnl_percent:+.2f}%")

    print(f"\nFinal active positions: {monitor.get_position_count()}")
    print(f"Completed positions: {len(monitor.completed)}")

    return monitor


if __name__ == "__main__":
    test_exit_monitor()
