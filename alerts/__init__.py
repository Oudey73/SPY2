"""
Alert Systems for SPY Opportunity Agent
"""
from .email_alert import EmailAlertSystem
from .telegram_alert import TelegramAlertSystem

__all__ = ["EmailAlertSystem", "TelegramAlertSystem"]
