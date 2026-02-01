"""
Signal Detection Module for SPY
"""
from .signal_detector import SPYSignalDetector, Signal, SignalType, Direction
from .opportunity_scorer import SPYOpportunityScorer, Opportunity, OpportunityGrade

__all__ = [
    "SPYSignalDetector", "Signal", "SignalType", "Direction",
    "SPYOpportunityScorer", "Opportunity", "OpportunityGrade"
]
