"""
Expiry Advisor for SPY Options
Recommends the best option expiry date by scoring candidate Fridays against
economic events, Mag 7 earnings, OPEX pinning, theta optimization, and conviction.

Read-only deps:
  - risk.event_risk (FOMC_DATES, CPI_DATES, _third_friday, _first_friday, _wednesday_before_opex)
  - strategy.trade_builder (DTE_RANGES)
  - execution.opportunity_logger (OpportunityLogger)
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from risk.event_risk import (
    FOMC_DATES,
    CPI_DATES,
    _third_friday,
    _first_friday,
    _wednesday_before_opex,
)
from strategy.trade_builder import DTE_RANGES
from strategy.strategy_selector import StrategyType
from execution.opportunity_logger import OpportunityLogger


# ---------------------------------------------------------------------------
# Mag 7 Earnings Windows (approximate week-of dates per quarter, 2025-2026)
# Each entry: ticker, approx_earnings_date, spy_weight (used to scale penalty)
# ---------------------------------------------------------------------------
MAG7_EARNINGS: List[Dict] = [
    # ---- Q4 2024 reports (Jan-Feb 2025) ----
    {"ticker": "MSFT",  "date": date(2025, 1, 29),  "spy_weight": 0.072},
    {"ticker": "META",  "date": date(2025, 1, 29),  "spy_weight": 0.026},
    {"ticker": "AAPL",  "date": date(2025, 1, 30),  "spy_weight": 0.074},
    {"ticker": "AMZN",  "date": date(2025, 2, 6),   "spy_weight": 0.038},
    {"ticker": "GOOGL", "date": date(2025, 2, 4),   "spy_weight": 0.042},
    {"ticker": "NVDA",  "date": date(2025, 2, 26),  "spy_weight": 0.065},
    {"ticker": "TSLA",  "date": date(2025, 1, 29),  "spy_weight": 0.018},
    # ---- Q1 2025 reports (Apr-May 2025) ----
    {"ticker": "MSFT",  "date": date(2025, 4, 23),  "spy_weight": 0.072},
    {"ticker": "META",  "date": date(2025, 4, 23),  "spy_weight": 0.026},
    {"ticker": "AAPL",  "date": date(2025, 5, 1),   "spy_weight": 0.074},
    {"ticker": "AMZN",  "date": date(2025, 5, 1),   "spy_weight": 0.038},
    {"ticker": "GOOGL", "date": date(2025, 4, 24),  "spy_weight": 0.042},
    {"ticker": "NVDA",  "date": date(2025, 5, 28),  "spy_weight": 0.065},
    {"ticker": "TSLA",  "date": date(2025, 4, 22),  "spy_weight": 0.018},
    # ---- Q2 2025 reports (Jul-Aug 2025) ----
    {"ticker": "MSFT",  "date": date(2025, 7, 22),  "spy_weight": 0.072},
    {"ticker": "META",  "date": date(2025, 7, 23),  "spy_weight": 0.026},
    {"ticker": "AAPL",  "date": date(2025, 7, 31),  "spy_weight": 0.074},
    {"ticker": "AMZN",  "date": date(2025, 7, 31),  "spy_weight": 0.038},
    {"ticker": "GOOGL", "date": date(2025, 7, 22),  "spy_weight": 0.042},
    {"ticker": "NVDA",  "date": date(2025, 8, 27),  "spy_weight": 0.065},
    {"ticker": "TSLA",  "date": date(2025, 7, 22),  "spy_weight": 0.018},
    # ---- Q3 2025 reports (Oct-Nov 2025) ----
    {"ticker": "MSFT",  "date": date(2025, 10, 22), "spy_weight": 0.072},
    {"ticker": "META",  "date": date(2025, 10, 22), "spy_weight": 0.026},
    {"ticker": "AAPL",  "date": date(2025, 10, 30), "spy_weight": 0.074},
    {"ticker": "AMZN",  "date": date(2025, 10, 30), "spy_weight": 0.038},
    {"ticker": "GOOGL", "date": date(2025, 10, 23), "spy_weight": 0.042},
    {"ticker": "NVDA",  "date": date(2025, 11, 19), "spy_weight": 0.065},
    {"ticker": "TSLA",  "date": date(2025, 10, 22), "spy_weight": 0.018},
    # ---- Q4 2025 reports (Jan-Feb 2026) ----
    {"ticker": "MSFT",  "date": date(2026, 1, 28),  "spy_weight": 0.072},
    {"ticker": "META",  "date": date(2026, 1, 28),  "spy_weight": 0.026},
    {"ticker": "AAPL",  "date": date(2026, 1, 29),  "spy_weight": 0.074},
    {"ticker": "AMZN",  "date": date(2026, 2, 5),   "spy_weight": 0.038},
    {"ticker": "GOOGL", "date": date(2026, 2, 3),   "spy_weight": 0.042},
    {"ticker": "NVDA",  "date": date(2026, 2, 25),  "spy_weight": 0.065},
    {"ticker": "TSLA",  "date": date(2026, 1, 28),  "spy_weight": 0.018},
]

_MAX_MAG7_WEIGHT = max(e["spy_weight"] for e in MAG7_EARNINGS)

# NFP dates (1st Friday) are computed dynamically via _first_friday from event_risk
# but we pre-generate for 2025-2026 for lookup speed
NFP_DATES = [
    _first_friday(y, m) for y in (2025, 2026) for m in range(1, 13)
]

# Map strategy string values (from opportunity JSON) → StrategyType enum
_STRATEGY_LOOKUP = {st.value: st for st in StrategyType}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_fridays(start: date, min_dte: int, max_dte: int) -> List[date]:
    """Generate all Fridays between start+min_dte and start+max_dte."""
    first = start + timedelta(days=min_dte)
    last = start + timedelta(days=max_dte)
    days_until_friday = (4 - first.weekday()) % 7
    current = first + timedelta(days=days_until_friday)
    fridays = []
    while current <= last:
        fridays.append(current)
        current += timedelta(days=7)
    return fridays


def _is_opex(d: date) -> bool:
    """Return True if d is the 3rd Friday of its month."""
    return d == _third_friday(d.year, d.month)


def _days_between(d1: date, d2: date) -> int:
    return abs((d1 - d2).days)


def _is_vix_expiry_week(d: date) -> bool:
    """Check if d falls in the same Mon-Fri week as VIX expiry Wednesday."""
    vix_expiry = _wednesday_before_opex(d.year, d.month)
    week_start = vix_expiry - timedelta(days=vix_expiry.weekday())  # Monday
    week_end = week_start + timedelta(days=4)  # Friday
    return week_start <= d <= week_end


def _is_credit_strategy(strategy_type: StrategyType) -> bool:
    return strategy_type in (
        StrategyType.BULL_PUT_SPREAD,
        StrategyType.BEAR_CALL_SPREAD,
        StrategyType.IRON_CONDOR,
        StrategyType.IRON_BUTTERFLY,
        StrategyType.SHORT_PUT,
        StrategyType.SHORT_CALL,
        StrategyType.SHORT_STRANGLE,
    )


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

def score_candidate(
    candidate: date,
    ref_date: date,
    strategy_type: StrategyType,
    opp_score: int,
) -> Dict:
    """Score a single candidate Friday. Base 100, deduct penalties, add bonuses."""
    score = 100
    reasons: List[str] = []
    dte = (candidate - ref_date).days
    is_credit = _is_credit_strategy(strategy_type)

    # --- FOMC proximity ---
    for fomc in FOMC_DATES:
        gap = _days_between(candidate, fomc)
        if gap == 0:
            score -= 30
            reasons.append(f"FOMC on expiry ({fomc}) -> -30")
        elif gap == 1:
            score -= 25
            reasons.append(f"FOMC 1-day away ({fomc}) -> -25")
        elif gap <= 3:
            score -= 10
            reasons.append(f"FOMC within 3 days ({fomc}) -> -10")

    # --- CPI proximity ---
    for cpi in CPI_DATES:
        gap = _days_between(candidate, cpi)
        if gap == 0:
            score -= 25
            reasons.append(f"CPI on expiry ({cpi}) -> -25")
        elif gap == 1:
            score -= 20
            reasons.append(f"CPI 1-day away ({cpi}) -> -20")
        elif gap <= 3:
            score -= 8
            reasons.append(f"CPI within 3 days ({cpi}) -> -8")

    # --- NFP proximity ---
    for nfp in NFP_DATES:
        gap = _days_between(candidate, nfp)
        if gap == 0:
            score -= 15
            reasons.append(f"NFP on expiry ({nfp}) -> -15")
        elif gap == 1:
            score -= 8
            reasons.append(f"NFP 1-day away ({nfp}) -> -8")

    # --- Mag 7 Earnings proximity ---
    for earn in MAG7_EARNINGS:
        gap = _days_between(candidate, earn["date"])
        if gap <= 2:
            weight_ratio = earn["spy_weight"] / _MAX_MAG7_WEIGHT
            if gap == 0:
                penalty = int(20 * weight_ratio)
            elif gap == 1:
                penalty = int(14 * weight_ratio)
            else:
                penalty = int(8 * weight_ratio)
            penalty = max(penalty, 8)  # floor of -8
            score -= penalty
            reasons.append(
                f"{earn['ticker']} earnings {gap}d away "
                f"({earn['date']}, wt={earn['spy_weight']:.1%}) -> -{penalty}"
            )

    # --- OPEX (3rd Friday) ---
    if _is_opex(candidate):
        if is_credit:
            score -= 10
            reasons.append("Monthly OPEX (credit spread pin risk) -> -10")
        else:
            score -= 5
            reasons.append("Monthly OPEX (debit spread minor risk) -> -5")

    # --- VIX expiry week ---
    if _is_vix_expiry_week(candidate):
        score -= 8
        reasons.append("VIX expiry week (vol distortion) -> -8")

    # --- Theta sweet spot bonus (credit spreads, 21-35 DTE) ---
    if is_credit and 21 <= dte <= 35:
        score += 5
        reasons.append(f"Theta sweet spot ({dte} DTE, credit) -> +5")

    # --- High conviction + tight DTE bonus ---
    if opp_score >= 80 and dte <= 21:
        score += 5
        reasons.append(f"High conviction ({opp_score}) + tight DTE ({dte}d) -> +5")

    return {
        "date": candidate,
        "dte": dte,
        "score": score,
        "reasons": reasons,
    }


def _label_tier(score: int) -> str:
    if score >= 90:
        return "BEST"
    elif score >= 75:
        return "GOOD"
    elif score >= 60:
        return "ACCEPTABLE"
    else:
        return "AVOID"


# ---------------------------------------------------------------------------
# Main advisor entry-point
# ---------------------------------------------------------------------------

def advise_expiry(opp_id: str) -> str:
    """Load opportunity by ID, score all candidate Fridays, return formatted report."""
    opp_logger = OpportunityLogger()
    opp = opp_logger.get(opp_id)

    if opp is None:
        return f"Opportunity '{opp_id}' not found."

    # Extract fields from logged opportunity structure
    opportunity = opp.get("opportunity", {})
    trade_plan = opp.get("trade_plan", {})

    # Determine strategy — prefer trade_plan.strategy, fall back to opportunity
    strategy_str = (
        trade_plan.get("strategy")
        or opportunity.get("strategy")
        or "bull_put_spread"
    )
    strategy_type = _STRATEGY_LOOKUP.get(strategy_str, StrategyType.BULL_PUT_SPREAD)

    opp_score = opportunity.get("score", 50)
    entry_date_str = opp.get("timestamp", "")

    if entry_date_str:
        try:
            ref_date = date.fromisoformat(entry_date_str[:10])
        except ValueError:
            ref_date = date.today()
    else:
        ref_date = date.today()

    # Get DTE range from trade_builder's DTE_RANGES
    min_dte, max_dte = DTE_RANGES.get(strategy_type, (30, 45))
    candidates = _next_fridays(ref_date, min_dte, max_dte)

    if not candidates:
        return (
            f"No candidate Fridays found for {strategy_type.value} "
            f"(DTE {min_dte}-{max_dte} from {ref_date})"
        )

    scored = [
        score_candidate(c, ref_date, strategy_type, opp_score)
        for c in candidates
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    for s in scored:
        s["label"] = _label_tier(s["score"])

    return _format_report(opp, strategy_type, scored, min_dte, max_dte)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_report(
    opp: Dict,
    strategy_type: StrategyType,
    scored: List[Dict],
    min_dte: int,
    max_dte: int,
) -> str:
    best = scored[0]
    is_credit = _is_credit_strategy(strategy_type)
    opportunity = opp.get("opportunity", {})
    direction = opportunity.get("direction", "N/A")
    opp_score = opportunity.get("score", "?")
    grade = opportunity.get("grade", "?")

    lines = [
        "=" * 60,
        "  EXPIRY ADVISOR REPORT",
        "=" * 60,
        "",
        f"  Opportunity  : {opp.get('opp_id', '?')}",
        f"  Direction    : {str(direction).upper()}",
        f"  Strategy     : {strategy_type.value.replace('_', ' ').title()}",
        f"  Conviction   : {opp_score}/100  ({grade})",
        f"  Ref Date     : {opp.get('timestamp', 'N/A')[:10]}",
        f"  DTE Window   : {min_dte}-{max_dte} days",
        f"  Type         : {'Credit (sell premium)' if is_credit else 'Debit (buy premium)'}",
        "",
        "-" * 60,
        f"  RECOMMENDED EXPIRY  ->  {best['date']}  "
        f"({best['dte']} DTE, score {best['score']}/100)",
        "-" * 60,
        "",
        "  ALL CANDIDATES (ranked):",
        "",
    ]

    for i, s in enumerate(scored, 1):
        marker = ">>>" if i == 1 else "   "
        lines.append(
            f"  {marker} {i}. {s['date']}  "
            f"({s['dte']} DTE)  "
            f"Score: {s['score']:>3}  [{s['label']}]"
        )
        if s["reasons"]:
            for r in s["reasons"]:
                lines.append(f"          * {r}")
        else:
            lines.append("          * No event conflicts -- clean window")
        lines.append("")

    best_dates = [s for s in scored if s["label"] == "BEST"]
    avoid_dates = [s for s in scored if s["label"] == "AVOID"]

    lines.append("-" * 60)
    lines.append("  SUMMARY")
    lines.append("-" * 60)

    if best_dates:
        lines.append(
            f"  BEST options : "
            + ", ".join(f"{s['date']} ({s['dte']}d)" for s in best_dates)
        )
    else:
        lines.append("  BEST options : None -- all have some event risk")

    if avoid_dates:
        lines.append(
            f"  AVOID        : "
            + ", ".join(f"{s['date']} ({s['dte']}d)" for s in avoid_dates)
        )

    lines.append("")
    lines.append("  REASONING:")
    if best["reasons"]:
        for r in best["reasons"]:
            lines.append(f"    * {r}")
    else:
        lines.append("    * Clean window with no major event conflicts")

    if is_credit and 21 <= best["dte"] <= 35:
        lines.append("    * Theta decay accelerates in 21-35 DTE zone (optimal for credit)")
    if opp_score and isinstance(opp_score, int) and opp_score >= 80:
        lines.append("    * High conviction setup -- tighter DTE acceptable")

    lines.extend([
        "",
        "=" * 60,
        "  Disclaimer: Event dates are approximate.",
        "  Always verify against the live economic calendar.",
        "=" * 60,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        opp_id = sys.argv[1]
    else:
        opp_id = "OPP-001"
    print(advise_expiry(opp_id))
