"""
Position Sizer for SPY Options Agent
Risk-based position sizing with regime and confidence adjustments
"""
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from analysis.regime_classifier import RegimeType


@dataclass
class PositionSize:
    contracts: int
    risk_per_trade: float  # dollar amount
    risk_percent: float    # % of account
    max_loss: float        # max loss for position
    confidence_tier: str
    regime_multiplier: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "contracts": self.contracts,
            "risk_per_trade": self.risk_per_trade,
            "risk_percent": self.risk_percent,
            "max_loss": self.max_loss,
            "confidence_tier": self.confidence_tier,
            "regime_multiplier": self.regime_multiplier,
            "reasoning": self.reasoning,
        }


# Risk % by confidence level
CONFIDENCE_RISK_MAP = {
    "very_high": 0.03,   # 3% — A+ grade, confidence > 80
    "high": 0.02,        # 2% — A grade, confidence 65-80
    "moderate": 0.015,   # 1.5% — B grade, confidence 50-65
    "low": 0.01,         # 1% — C grade or below
}

# Regime multipliers
REGIME_MULTIPLIER = {
    RegimeType.TRENDING_UP: 1.0,
    RegimeType.TRENDING_DOWN: 1.0,
    RegimeType.RANGE_BOUND: 0.85,
    RegimeType.HIGH_VOLATILITY: 0.5,
    RegimeType.TRANSITION: 0.5,
}

MAX_CONTRACTS = 20


class PositionSizer:
    """
    Calculates position size based on account value, max loss per contract,
    confidence level, and regime.

    Formula: contracts = (Account × Risk%) / Max Loss per Contract
    Capped at MAX_CONTRACTS.
    """

    def calculate(
        self,
        account_value: float,
        max_loss_per_contract: float,
        confidence: float,
        regime_type: RegimeType = RegimeType.RANGE_BOUND,
    ) -> PositionSize:
        """
        Calculate position size.

        Args:
            account_value: Total account value in dollars
            max_loss_per_contract: Max loss per contract (spread width × 100 - premium)
            confidence: Confidence score 0-100
            regime_type: Current market regime

        Returns:
            PositionSize with number of contracts and details
        """
        if account_value <= 0 or max_loss_per_contract <= 0:
            return PositionSize(
                contracts=0,
                risk_per_trade=0,
                risk_percent=0,
                max_loss=0,
                confidence_tier="none",
                regime_multiplier=0,
                reasoning="Invalid account value or max loss",
            )

        # Determine confidence tier
        if confidence >= 80:
            tier = "very_high"
        elif confidence >= 65:
            tier = "high"
        elif confidence >= 50:
            tier = "moderate"
        else:
            tier = "low"

        base_risk_pct = CONFIDENCE_RISK_MAP[tier]
        regime_mult = REGIME_MULTIPLIER.get(regime_type, 0.75)

        # Adjusted risk
        effective_risk_pct = base_risk_pct * regime_mult
        risk_dollars = account_value * effective_risk_pct

        # Calculate contracts
        contracts = int(risk_dollars / max_loss_per_contract)
        contracts = max(1, min(contracts, MAX_CONTRACTS))

        actual_max_loss = contracts * max_loss_per_contract
        actual_risk_pct = actual_max_loss / account_value

        reasoning = (
            f"{tier} confidence ({confidence:.0f}%) -> {base_risk_pct:.1%} base risk | "
            f"Regime {regime_type.value} -> {regime_mult}x multiplier | "
            f"Effective risk: {effective_risk_pct:.2%} = ${risk_dollars:,.0f} | "
            f"{contracts} contracts × ${max_loss_per_contract:,.0f} = ${actual_max_loss:,.0f}"
        )

        return PositionSize(
            contracts=contracts,
            risk_per_trade=risk_dollars,
            risk_percent=actual_risk_pct * 100,
            max_loss=actual_max_loss,
            confidence_tier=tier,
            regime_multiplier=regime_mult,
            reasoning=reasoning,
        )
