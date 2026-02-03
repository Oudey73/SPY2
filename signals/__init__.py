"""
Signal Detection Module for SPY
"""
from .signal_detector import SPYSignalDetector, Signal, SignalType, Direction
from .opportunity_scorer import (
    SPYOpportunityScorer, Opportunity, OpportunityGrade,
    EnhancedScoringResult, MultiplierEngine
)
from .exit_monitor import ExitMonitor, ExitAlert, AlertType, TrackedPosition

__all__ = [
    "SPYSignalDetector", "Signal", "SignalType", "Direction",
    "SPYOpportunityScorer", "Opportunity", "OpportunityGrade",
    "EnhancedScoringResult", "MultiplierEngine",
    "ExitMonitor", "ExitAlert", "AlertType", "TrackedPosition"
]
