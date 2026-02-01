"""
Options Liquidity Checker for SPY
Validates bid-ask spread, volume, and open interest for options contracts
"""
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class LiquidityScore:
    passed: bool
    quality: str  # "optimal", "acceptable", "marginal", "fail"
    score: int  # 0-100
    bid_ask_ok: bool
    volume_ok: bool
    oi_ok: bool
    warnings: list
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "quality": self.quality,
            "score": self.score,
            "bid_ask_ok": self.bid_ask_ok,
            "volume_ok": self.volume_ok,
            "oi_ok": self.oi_ok,
            "warnings": self.warnings,
            "details": self.details,
        }


class LiquidityChecker:
    """
    Checks options liquidity against min/preferred/optimal thresholds.
    SPY options are generally very liquid; defaults to pass-with-warning
    if no chain data is available.
    """

    # Thresholds: (min, preferred, optimal)
    SPREAD_THRESHOLDS = (0.10, 0.05, 0.02)  # max bid-ask spread as % of mid
    VOLUME_THRESHOLDS = (100, 500, 2000)      # daily volume
    OI_THRESHOLDS = (500, 2000, 10000)        # open interest

    def check_option(
        self,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[int] = None,
        open_interest: Optional[int] = None,
    ) -> LiquidityScore:
        """
        Check liquidity of an option contract.

        If no data provided (common for SPY), returns pass-with-warning
        since SPY options are among the most liquid in the market.
        """
        warnings = []

        # If no data at all, SPY default pass
        if bid is None and ask is None and volume is None and open_interest is None:
            return LiquidityScore(
                passed=True,
                quality="acceptable",
                score=65,
                bid_ask_ok=True,
                volume_ok=True,
                oi_ok=True,
                warnings=["No chain data available; SPY assumed liquid"],
                details="Default pass for SPY (no chain data)",
            )

        score = 0
        bid_ask_ok = True
        volume_ok = True
        oi_ok = True

        # Bid-ask spread check
        if bid is not None and ask is not None and bid > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0

            if spread_pct <= self.SPREAD_THRESHOLDS[2]:
                score += 40  # optimal
            elif spread_pct <= self.SPREAD_THRESHOLDS[1]:
                score += 30  # preferred
            elif spread_pct <= self.SPREAD_THRESHOLDS[0]:
                score += 15  # min
                warnings.append(f"Wide spread: {spread_pct:.1%}")
            else:
                bid_ask_ok = False
                warnings.append(f"Spread too wide: {spread_pct:.1%}")
        else:
            score += 20  # assume ok for SPY
            warnings.append("No bid/ask data")

        # Volume check
        if volume is not None:
            if volume >= self.VOLUME_THRESHOLDS[2]:
                score += 30
            elif volume >= self.VOLUME_THRESHOLDS[1]:
                score += 22
            elif volume >= self.VOLUME_THRESHOLDS[0]:
                score += 10
                warnings.append(f"Low volume: {volume}")
            else:
                volume_ok = False
                warnings.append(f"Very low volume: {volume}")
        else:
            score += 15
            warnings.append("No volume data")

        # Open interest check
        if open_interest is not None:
            if open_interest >= self.OI_THRESHOLDS[2]:
                score += 30
            elif open_interest >= self.OI_THRESHOLDS[1]:
                score += 22
            elif open_interest >= self.OI_THRESHOLDS[0]:
                score += 10
                warnings.append(f"Low OI: {open_interest}")
            else:
                oi_ok = False
                warnings.append(f"Very low OI: {open_interest}")
        else:
            score += 15
            warnings.append("No OI data")

        passed = bid_ask_ok and volume_ok and oi_ok

        if score >= 80:
            quality = "optimal"
        elif score >= 55:
            quality = "acceptable"
        elif score >= 30:
            quality = "marginal"
        else:
            quality = "fail"
            passed = False

        return LiquidityScore(
            passed=passed,
            quality=quality,
            score=score,
            bid_ask_ok=bid_ask_ok,
            volume_ok=volume_ok,
            oi_ok=oi_ok,
            warnings=warnings,
            details=f"Liquidity score {score}/100 ({quality})",
        )
