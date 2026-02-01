"""
Configuration for SPY Opportunity Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ASSETS TO MONITOR
# =============================================================================
SYMBOL = "SPY"
VIX_SYMBOL = "^VIX"  # Yahoo Finance VIX symbol

# =============================================================================
# API KEYS
# =============================================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ORATS_API_KEY = os.getenv("ORATS_API_KEY", "")

# =============================================================================
# SIGNAL THRESHOLDS (Based on research)
# =============================================================================

THRESHOLDS = {
    # IBS (Internal Bar Strength) - PRIMARY SIGNAL
    # IBS = (Close - Low) / (High - Low)
    "ibs_oversold": 0.2,          # Buy signal when IBS < 0.2
    "ibs_overbought": 0.8,        # Sell/exit signal when IBS > 0.8

    # RSI Settings - Use RSI(3) for mean reversion
    "rsi_period": 3,              # Short-term RSI works better for mean reversion
    "rsi_oversold": 20,           # Buy when RSI(3) < 20
    "rsi_overbought": 80,         # Sell when RSI(3) > 80

    # VIX Filter - Only enter when fear is elevated
    "vix_ma_period": 10,          # 10-day MA of VIX
    "vix_elevated": 20,           # VIX > 20 = elevated fear
    "vix_extreme": 30,            # VIX > 30 = panic mode (best entries)

    # Mean Reversion Exit
    "max_hold_days": 5,           # Exit after 5 days regardless

    # Intraday Momentum (Last 30 minutes)
    "intraday_momentum_threshold": 0.3,  # 0.3% move triggers signal

    # Price deviation from VWAP
    "vwap_deviation_percent": 1.5,  # Alert when price > 1.5% from VWAP
}

# =============================================================================
# SCORING WEIGHTS (Based on backtested effectiveness)
# =============================================================================

SIGNAL_WEIGHTS = {
    # Tier 1 - Primary signals (highest evidence)
    "ibs_extreme": 35,            # IBS < 0.15 or > 0.85
    "ibs_rsi_combo": 30,          # IBS + RSI both triggering

    # Tier 2 - Confirming signals
    "vix_filter_pass": 15,        # VIX > 10-day MA
    "vix_extreme": 10,            # VIX > 30

    # Tier 3 - Supporting signals
    "rsi_extreme": 10,            # RSI(3) < 15 or > 85
    "price_below_ma": 5,          # Price below 50-day MA (mean reversion context)
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

RISK = {
    "max_position_percent": 5,    # Max 5% of portfolio per trade
    "stop_loss_percent": 2,       # 2% stop loss
    "target_1_percent": 1,        # First target: 1%
    "target_2_percent": 2,        # Second target: 2%
    "target_3_percent": 3,        # Third target: 3%
}

# =============================================================================
# MARKET HOURS (Eastern Time)
# =============================================================================

MARKET = {
    "open_hour": 9,
    "open_minute": 30,
    "close_hour": 16,
    "close_minute": 0,
    "timezone": "US/Eastern",

    # Intraday momentum check (last 30 min)
    "momentum_check_hour": 15,
    "momentum_check_minute": 30,
}

# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================

EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", 587)),
    "sender_email": os.getenv("SENDER_EMAIL", ""),
    "sender_password": os.getenv("SENDER_PASSWORD", ""),
    "recipient_email": os.getenv("RECIPIENT_EMAIL", ""),
}

# =============================================================================
# TIMING
# =============================================================================

TIMING = {
    "scan_interval_minutes": 5,    # Check every 5 minutes during market hours
    "daily_scan_time": "09:35",    # Main scan 5 min after open
    "eod_scan_time": "15:55",      # End of day scan
    "heartbeat_hours": 8,          # Status update every 8 hours
}

# =============================================================================
# ACCOUNT & POSITION SIZING
# =============================================================================

ACCOUNT = {
    "value": float(os.getenv("ACCOUNT_VALUE", 100000)),  # Total account value
}

REGIME_CONFIG = {
    "adx_period": 14,
    "atr_short": 5,
    "atr_long": 20,
    "hh_hl_lookback": 10,
    "vix_high_vol_threshold": 25,
    "atr_ratio_high_vol": 1.5,
    "iv_rank_high_vol": 70,
    "adx_trending": 25,
    "adx_ranging": 20,
    "min_confidence": 60,  # below this -> TRANSITION
}

POSITION_SIZING = {
    "max_contracts": 20,
    "risk_pct_very_high": 0.03,   # 3% for A+ (confidence > 80)
    "risk_pct_high": 0.02,        # 2% for A (confidence 65-80)
    "risk_pct_moderate": 0.015,   # 1.5% for B (confidence 50-65)
    "risk_pct_low": 0.01,         # 1% for C or below
    "regime_mult_trending": 1.0,
    "regime_mult_range": 0.85,
    "regime_mult_high_vol": 0.5,
    "regime_mult_transition": 0.5,
}

RISK_LIMITS = {
    "max_daily_loss_pct": 0.03,       # 3% daily stop
    "max_weekly_loss_pct": 0.05,      # 5% weekly stop
    "max_correlated_positions": 3,
    "delta_limit": 50,
    "gamma_min": -10,
    "theta_max_pct": 0.005,           # 0.5% of account per day
    "vega_long_max": 100,
    "vega_short_min": -50,
}
