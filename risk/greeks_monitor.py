"""
Greeks Monitor for SPY Options Agent
Monitors portfolio-level Greeks against defined limits
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class GreeksReport:
    within_limits: bool
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    breaches: List[str]
    warnings: List[str]
    required_actions: List[str]

    def to_dict(self) -> dict:
        return {
            "within_limits": self.within_limits,
            "portfolio_delta": self.portfolio_delta,
            "portfolio_gamma": self.portfolio_gamma,
            "portfolio_theta": self.portfolio_theta,
            "portfolio_vega": self.portfolio_vega,
            "breaches": self.breaches,
            "warnings": self.warnings,
            "required_actions": self.required_actions,
        }


# Portfolio Greeks limits
LIMITS = {
    "delta_max": 50,        # absolute portfolio delta
    "delta_min": -50,
    "gamma_min": -10,       # negative gamma limit
    "theta_max_pct": 0.005, # 0.5% of account per day
    "vega_long_max": 100,
    "vega_short_min": -50,
}


class GreeksMonitor:
    """
    Monitors portfolio-level Greeks against risk limits.

    Limits:
    - Delta: +/- 50
    - Gamma: min -10 (negative gamma)
    - Theta: max 0.5% of account/day
    - Vega: long max 100, short min -50
    """

    def check_portfolio(
        self,
        positions: List[Dict],
        account_value: float,
    ) -> GreeksReport:
        """
        Check portfolio Greeks against limits.

        Args:
            positions: List of position dicts with 'delta', 'gamma', 'theta', 'vega' keys
            account_value: Total account value

        Returns:
            GreeksReport with breach/warning details
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for pos in positions:
            qty = pos.get("quantity", 1)
            total_delta += pos.get("delta", 0) * qty
            total_gamma += pos.get("gamma", 0) * qty
            total_theta += pos.get("theta", 0) * qty
            total_vega += pos.get("vega", 0) * qty

        breaches = []
        warnings = []
        required_actions = []

        # Delta check
        if abs(total_delta) > LIMITS["delta_max"]:
            breaches.append(f"Delta {total_delta:+.1f} exceeds +/-{LIMITS['delta_max']} limit")
            if total_delta > 0:
                required_actions.append("Reduce long delta: sell calls or buy puts")
            else:
                required_actions.append("Reduce short delta: buy calls or sell puts")
        elif abs(total_delta) > LIMITS["delta_max"] * 0.8:
            warnings.append(f"Delta {total_delta:+.1f} approaching limit (+/-{LIMITS['delta_max']})")

        # Gamma check
        if total_gamma < LIMITS["gamma_min"]:
            breaches.append(f"Negative gamma {total_gamma:.1f} exceeds {LIMITS['gamma_min']} limit")
            required_actions.append("Reduce short gamma: close short options or buy protection")
        elif total_gamma < LIMITS["gamma_min"] * 0.8:
            warnings.append(f"Negative gamma {total_gamma:.1f} approaching limit ({LIMITS['gamma_min']})")

        # Theta check (as % of account)
        if account_value > 0:
            theta_pct = abs(total_theta) / account_value
            max_theta_daily = account_value * LIMITS["theta_max_pct"]
            if abs(total_theta) > max_theta_daily:
                breaches.append(
                    f"Theta ${total_theta:+.0f}/day exceeds "
                    f"${max_theta_daily:.0f}/day ({LIMITS['theta_max_pct']:.1%} of account)"
                )
                required_actions.append("Reduce theta exposure: close short-dated positions")
            elif abs(total_theta) > max_theta_daily * 0.8:
                warnings.append(f"Theta ${total_theta:+.0f}/day approaching daily limit")

        # Vega check
        if total_vega > LIMITS["vega_long_max"]:
            breaches.append(f"Long vega {total_vega:.1f} exceeds {LIMITS['vega_long_max']} limit")
            required_actions.append("Reduce long vega: sell options or add short vega positions")
        elif total_vega < LIMITS["vega_short_min"]:
            breaches.append(f"Short vega {total_vega:.1f} exceeds {LIMITS['vega_short_min']} limit")
            required_actions.append("Reduce short vega: buy options or close short vega")
        elif total_vega > LIMITS["vega_long_max"] * 0.8:
            warnings.append(f"Long vega {total_vega:.1f} approaching limit")
        elif total_vega < LIMITS["vega_short_min"] * 0.8:
            warnings.append(f"Short vega {total_vega:.1f} approaching limit")

        within_limits = len(breaches) == 0

        return GreeksReport(
            within_limits=within_limits,
            portfolio_delta=total_delta,
            portfolio_gamma=total_gamma,
            portfolio_theta=total_theta,
            portfolio_vega=total_vega,
            breaches=breaches,
            warnings=warnings,
            required_actions=required_actions,
        )

    def check_new_trade_impact(
        self,
        current_positions: List[Dict],
        new_trade_greeks: Dict,
        account_value: float,
    ) -> GreeksReport:
        """
        Check what happens to portfolio Greeks if a new trade is added.

        Args:
            current_positions: Existing positions
            new_trade_greeks: Dict with delta, gamma, theta, vega for new trade
            account_value: Account value

        Returns:
            GreeksReport for the hypothetical combined portfolio
        """
        combined = list(current_positions) + [new_trade_greeks]
        return self.check_portfolio(combined, account_value)
