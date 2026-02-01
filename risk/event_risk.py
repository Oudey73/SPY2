"""
Event Risk Checker for SPY Options Agent
Checks FOMC, CPI, NFP, OPEX, VIX expiry calendars for trade lockouts
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class EventRisk:
    blocked: bool
    events: List[str]
    recommendation: str
    risk_level: str  # "HIGH", "MEDIUM", "LOW", "NONE"
    days_to_next: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "events": self.events,
            "recommendation": self.recommendation,
            "risk_level": self.risk_level,
            "days_to_next": self.days_to_next,
        }


# Hardcoded FOMC meeting dates (announcement days) for 2025-2026
FOMC_DATES = [
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 17),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 16),
]

# Hardcoded CPI release dates (typically 2nd or 3rd Tuesday/Wednesday of month)
CPI_DATES = [
    # 2025
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 15), date(2025, 8, 12), date(2025, 9, 10),
    date(2025, 10, 14), date(2025, 11, 12), date(2025, 12, 10),
    # 2026
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 14), date(2026, 5, 12), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 15),
    date(2026, 10, 13), date(2026, 11, 10), date(2026, 12, 9),
]


def _third_friday(year: int, month: int) -> date:
    """Calculate 3rd Friday of a given month (monthly OPEX)."""
    d = date(year, month, 1)
    # Find first Friday
    day_of_week = d.weekday()  # Mon=0, Fri=4
    first_friday = d + timedelta(days=(4 - day_of_week) % 7)
    return first_friday + timedelta(weeks=2)


def _first_friday(year: int, month: int) -> date:
    """Calculate 1st Friday of a given month (NFP release)."""
    d = date(year, month, 1)
    day_of_week = d.weekday()
    return d + timedelta(days=(4 - day_of_week) % 7)


def _wednesday_before_opex(year: int, month: int) -> date:
    """VIX expiry is typically the Wednesday before monthly OPEX."""
    opex = _third_friday(year, month)
    return opex - timedelta(days=2)


class EventRiskChecker:
    """
    Checks upcoming economic events and options expiration dates.

    FOMC: 2-day lockout (day before + day of announcement)
    CPI: 1-day lockout (release day)
    OPEX: Pin risk warning on expiration Friday
    NFP: 1-day warning
    VIX expiry: Warning on expiry Wednesday
    """

    def __init__(self):
        self.fomc_dates = set(FOMC_DATES)
        self.cpi_dates = set(CPI_DATES)

    def check(self, target_date: Optional[date] = None) -> EventRisk:
        """
        Check event risk for a target date.

        Args:
            target_date: Date to check (defaults to today)

        Returns:
            EventRisk with blocked status and recommendations
        """
        if target_date is None:
            target_date = date.today()

        events = []
        blocked = False
        risk_level = "NONE"
        recommendations = []

        # --- FOMC check (2-day lockout) ---
        fomc_day = target_date in self.fomc_dates
        fomc_eve = (target_date + timedelta(days=1)) in self.fomc_dates

        if fomc_day:
            events.append("FOMC announcement today")
            blocked = True
            risk_level = "HIGH"
            recommendations.append("No new positions on FOMC day")
        elif fomc_eve:
            events.append("FOMC announcement tomorrow")
            blocked = True
            risk_level = "HIGH"
            recommendations.append("No new positions day before FOMC")

        # --- CPI check (1-day lockout) ---
        if target_date in self.cpi_dates:
            events.append("CPI release today")
            blocked = True
            risk_level = "HIGH"
            recommendations.append("No new positions on CPI day")

        # --- OPEX check (pin risk warning) ---
        opex = _third_friday(target_date.year, target_date.month)
        if target_date == opex:
            events.append("Monthly OPEX today")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
            recommendations.append("Pin risk: avoid opening near ATM strikes")

        # --- NFP check (1st Friday warning) ---
        nfp = _first_friday(target_date.year, target_date.month)
        if target_date == nfp and target_date != opex:
            events.append("NFP release today (1st Friday)")
            if risk_level == "NONE":
                risk_level = "MEDIUM"
            recommendations.append("NFP volatility expected; reduce size")

        # --- VIX expiry (Wednesday before OPEX) ---
        vix_exp = _wednesday_before_opex(target_date.year, target_date.month)
        if target_date == vix_exp:
            events.append("VIX expiry today")
            if risk_level == "NONE":
                risk_level = "LOW"
            recommendations.append("VIX expiry may cause vol distortion")

        # Calculate days to next major event
        days_to_next = self._days_to_next_event(target_date)

        if not events:
            risk_level = "NONE"
            recommendation = "No significant events; clear to trade"
        else:
            recommendation = "; ".join(recommendations)

        return EventRisk(
            blocked=blocked,
            events=events,
            recommendation=recommendation,
            risk_level=risk_level,
            days_to_next=days_to_next,
        )

    def _days_to_next_event(self, from_date: date) -> Optional[int]:
        """Find days until the next FOMC or CPI event."""
        all_dates = sorted(self.fomc_dates | self.cpi_dates)
        for d in all_dates:
            if d > from_date:
                return (d - from_date).days
        return None

    def get_upcoming_events(self, days_ahead: int = 14) -> List[dict]:
        """List upcoming events within the next N days."""
        today = date.today()
        end = today + timedelta(days=days_ahead)
        events = []

        for d in sorted(self.fomc_dates):
            if today <= d <= end:
                events.append({"date": d.isoformat(), "event": "FOMC", "lockout_days": 2})

        for d in sorted(self.cpi_dates):
            if today <= d <= end:
                events.append({"date": d.isoformat(), "event": "CPI", "lockout_days": 1})

        # Dynamic OPEX
        for month_offset in range(2):
            m = today.month + month_offset
            y = today.year
            if m > 12:
                m -= 12
                y += 1
            opex = _third_friday(y, m)
            if today <= opex <= end:
                events.append({"date": opex.isoformat(), "event": "OPEX", "lockout_days": 0})

        return sorted(events, key=lambda x: x["date"])
