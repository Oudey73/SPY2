# SPY Options Trading Agent - Improved Specification v2.0

## Executive Summary

This document provides a **complete, implementation-ready specification** for an AI-powered SPY options trading decision-support system. It addresses all gaps in the original specification with:

- **Quantitative thresholds** for all decisions
- **Specific data sources** with fallback options
- **Exact strategy construction rules**
- **Position sizing formulas**
- **Entry/exit mechanics**
- **Greeks management limits**
- **Event risk matrix**

---

## 1. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPY OPTIONS TRADING AGENT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ DATA LAYER   │───►│ ANALYSIS     │───►│ DECISION     │      │
│  │              │    │ ENGINE       │    │ ENGINE       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ • Yahoo      │    │ • Regime     │    │ • Strategy   │      │
│  │ • Polygon    │    │ • IV Analysis│    │   Selector   │      │
│  │ • Tradier    │    │ • Greeks     │    │ • Risk Check │      │
│  │ • CBOE       │    │ • Liquidity  │    │ • Trade Plan │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                  │               │
│                                                  ▼               │
│                            ┌──────────────────────────┐         │
│                            │     OUTPUT / ALERTS      │         │
│                            │  Telegram | Email | Log  │         │
│                            └──────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. REGIME CLASSIFICATION (QUANTITATIVE)

**CRITICAL: Every recommendation must specify current regime. No trade without regime.**

### 2.1 Regime Definitions

```python
REGIME_DEFINITIONS = {
    "TRENDING_UP": {
        "conditions": [
            "SPY > SMA(20) > SMA(50)",
            "ADX(14) > 25",
            "Higher highs AND higher lows (10 sessions)",
            "VIX < 20"
        ],
        "allowed_strategies": ["debit_call_spread", "call_diagonal", "bull_put_spread"],
        "bias": "bullish",
        "premium_approach": "can_buy_or_sell"
    },

    "TRENDING_DOWN": {
        "conditions": [
            "SPY < SMA(20) < SMA(50)",
            "ADX(14) > 25",
            "Lower highs AND lower lows (10 sessions)",
            "VIX > 18 or rising"
        ],
        "allowed_strategies": ["debit_put_spread", "put_diagonal", "bear_call_spread"],
        "bias": "bearish",
        "premium_approach": "prefer_debit"
    },

    "RANGE_BOUND": {
        "conditions": [
            "ADX(14) < 20",
            "Price within ATR(20) * 2 band for 5+ days",
            "No clear HH/HL or LH/LL pattern",
            "VIX: 12-18 (stable)"
        ],
        "allowed_strategies": ["iron_condor", "iron_butterfly", "credit_spread"],
        "bias": "neutral",
        "premium_approach": "prefer_sell"
    },

    "HIGH_VOLATILITY": {
        "conditions": [
            "VIX > 25 OR VIX spike > 20% in 1 day",
            "ATR(5) > ATR(20) * 1.5",
            "Daily range > 1.5%",
            "IV Rank > 70"
        ],
        "allowed_strategies": ["debit_spread", "long_straddle_if_IV_cheap"],
        "bias": "cautious",
        "premium_approach": "defined_risk_only",
        "size_adjustment": 0.5
    },

    "TRANSITION": {
        "conditions": [
            "Regime confidence < 60%",
            "Mixed signals from multiple timeframes"
        ],
        "allowed_strategies": ["small_debit_spreads_only"],
        "bias": "defensive",
        "size_adjustment": 0.5
    }
}
```

### 2.2 Regime Classification Algorithm

```python
def classify_regime(market_data: dict) -> dict:
    """
    Classify current market regime with confidence score

    Returns:
        {
            "regime": str,
            "confidence": float (0-100),
            "factors": list,
            "allowed_strategies": list
        }
    """
    price = market_data["price"]
    sma_20 = market_data["sma_20"]
    sma_50 = market_data["sma_50"]
    adx = market_data["adx_14"]
    vix = market_data["vix"]
    atr_5 = market_data["atr_5"]
    atr_20 = market_data["atr_20"]
    iv_rank = market_data["iv_rank"]

    scores = {
        "TRENDING_UP": 0,
        "TRENDING_DOWN": 0,
        "RANGE_BOUND": 0,
        "HIGH_VOLATILITY": 0
    }

    # HIGH_VOLATILITY takes precedence
    if vix > 25:
        scores["HIGH_VOLATILITY"] += 40
    if atr_5 > atr_20 * 1.5:
        scores["HIGH_VOLATILITY"] += 30
    if iv_rank > 70:
        scores["HIGH_VOLATILITY"] += 30

    # TRENDING checks
    if price > sma_20 > sma_50:
        scores["TRENDING_UP"] += 35
    if price < sma_20 < sma_50:
        scores["TRENDING_DOWN"] += 35
    if adx > 25:
        if price > sma_50:
            scores["TRENDING_UP"] += 35
        else:
            scores["TRENDING_DOWN"] += 35

    # RANGE_BOUND checks
    if adx < 20:
        scores["RANGE_BOUND"] += 40
    if 12 <= vix <= 18:
        scores["RANGE_BOUND"] += 30

    # Get dominant regime
    max_regime = max(scores, key=scores.get)
    max_score = scores[max_regime]

    # Check for TRANSITION
    if max_score < 60:
        return {
            "regime": "TRANSITION",
            "confidence": max_score,
            "factors": ["Mixed signals", "Low confidence"],
            "allowed_strategies": ["small_debit_spreads_only"]
        }

    return {
        "regime": max_regime,
        "confidence": max_score,
        "factors": [...],  # List relevant factors
        "allowed_strategies": REGIME_DEFINITIONS[max_regime]["allowed_strategies"]
    }
```

---

## 3. VOLATILITY ANALYSIS (QUANTITATIVE)

### 3.1 IV Classification

```python
IV_CLASSIFICATION = {
    # IV Rank (30-day lookback)
    "iv_rank": {
        "LOW":      {"range": (0, 25),   "action": "buy_premium_bias"},
        "NORMAL":   {"range": (25, 50),  "action": "neutral"},
        "ELEVATED": {"range": (50, 75),  "action": "sell_premium_bias"},
        "EXTREME":  {"range": (75, 100), "action": "sell_premium_reduced_size"}
    },

    # IV Percentile (1-year lookback) - more stable
    "iv_percentile": {
        "LOW":      {"range": (0, 20),   "action": "buy_premium_bias"},
        "NORMAL":   {"range": (20, 50),  "action": "neutral"},
        "ELEVATED": {"range": (50, 80),  "action": "sell_premium_bias"},
        "EXTREME":  {"range": (80, 100), "action": "sell_premium_reduced_size"}
    }
}
```

### 3.2 IV-RV Spread Decision

```python
def analyze_iv_rv_spread(iv_30: float, rv_30: float) -> dict:
    """
    Compare IV to realized vol to determine premium pricing

    Args:
        iv_30: 30-day implied volatility (annualized)
        rv_30: 30-day realized volatility (annualized)
    """
    spread = iv_30 - rv_30

    if spread > 5:
        return {
            "assessment": "PREMIUM_OVERPRICED",
            "spread": spread,
            "action": "SELL_BIAS",
            "note": f"IV {spread:.1f} pts above RV - premium sellers edge"
        }
    elif spread < -3:
        return {
            "assessment": "PREMIUM_CHEAP",
            "spread": spread,
            "action": "BUY_BIAS",
            "note": f"IV {abs(spread):.1f} pts below RV - consider buying premium"
        }
    else:
        return {
            "assessment": "FAIR",
            "spread": spread,
            "action": "NEUTRAL",
            "note": "IV roughly in line with RV"
        }
```

### 3.3 Term Structure Analysis

```python
def analyze_term_structure(vix: float, vix3m: float, vix6m: float = None) -> dict:
    """
    Analyze VIX term structure for regime confirmation

    Normal: VIX < VIX3M < VIX6M (contango)
    Inverted: VIX > VIX3M (backwardation) - STRESS SIGNAL
    """

    if vix < vix3m:
        # Contango - normal
        steepness = (vix3m - vix) / vix * 100
        return {
            "structure": "CONTANGO",
            "steepness_pct": steepness,
            "interpretation": "Normal market, sell front-month",
            "stress_level": "LOW"
        }
    else:
        # Backwardation - stress
        inversion = (vix - vix3m) / vix3m * 100
        return {
            "structure": "BACKWARDATION",
            "inversion_pct": inversion,
            "interpretation": "STRESS SIGNAL - near-term fear elevated",
            "stress_level": "HIGH" if inversion > 10 else "MODERATE"
        }
```

### 3.4 Skew Analysis

```python
def analyze_skew(put_iv_25d: float, call_iv_25d: float, atm_iv: float) -> dict:
    """
    Analyze put-call skew for sentiment

    SPY typically has put skew (put IV > call IV) due to hedging demand
    """
    put_skew = put_iv_25d - atm_iv
    call_skew = atm_iv - call_iv_25d

    # Normal SPY put skew: 3-8 points
    if put_skew > 10:
        return {
            "skew_type": "STEEP_PUT_SKEW",
            "put_premium": put_skew,
            "interpretation": "High hedging demand - consider put credit spreads",
            "sentiment": "FEARFUL"
        }
    elif put_skew < 2:
        return {
            "skew_type": "FLAT_SKEW",
            "put_premium": put_skew,
            "interpretation": "Complacent - unusual for SPY",
            "sentiment": "COMPLACENT"
        }
    else:
        return {
            "skew_type": "NORMAL",
            "put_premium": put_skew,
            "interpretation": "Normal hedging activity",
            "sentiment": "NEUTRAL"
        }
```

---

## 4. OPTIONS STRATEGY CONSTRUCTION

### 4.1 Strategy Parameters

```python
STRATEGY_RULES = {
    "VERTICAL_SPREAD": {
        "width": {
            "SPY": [2, 3, 5],  # $2, $3, or $5 wide
            "default": "3"
        },
        "dte": {
            "min": 14,
            "max": 45,
            "optimal": 30
        },
        "credit_spread_short_delta": {
            "aggressive": 0.35,
            "standard": 0.25,
            "conservative": 0.15
        },
        "debit_spread_short_delta": {
            "aggressive": 0.60,
            "standard": 0.50,
            "conservative": 0.40
        },
        "profit_target_pct": {
            "credit": 50,  # Close at 50% profit
            "debit": 75    # Close at 75% profit
        },
        "stop_loss_pct": {
            "credit": 200,  # Close at 2x credit received
            "debit": 50     # Close at 50% loss
        }
    },

    "IRON_CONDOR": {
        "wing_width": {
            "SPY": 5,  # $5 wide wings
        },
        "short_put_delta": {
            "standard": 0.20,
            "range": (0.15, 0.25)
        },
        "short_call_delta": {
            "standard": 0.20,
            "range": (0.15, 0.25)
        },
        "dte": {
            "min": 21,
            "max": 45,
            "optimal": 30
        },
        "profit_target_pct": 50,  # Close at 50% of max profit
        "adjustment_trigger": "short_strike_breached"
    },

    "CALENDAR_SPREAD": {
        "front_dte": {
            "min": 7,
            "max": 21,
            "optimal": 14
        },
        "back_dte": {
            "min": 30,
            "max": 60,
            "optimal": 45
        },
        "strike_selection": "ATM",
        "profit_target_pct": 25,
        "stop_loss": "front_month_ITM"
    }
}
```

### 4.2 Strategy Selection Logic

```python
def select_strategy(
    regime: str,
    iv_analysis: dict,
    direction_bias: str,
    risk_budget: float
) -> dict:
    """
    Select optimal strategy based on conditions

    Returns specific trade plan
    """

    strategy_matrix = {
        # (Regime, IV Level, Bias) -> Strategy
        ("TRENDING_UP", "ELEVATED", "bullish"): "bull_put_spread",
        ("TRENDING_UP", "LOW", "bullish"): "debit_call_spread",
        ("TRENDING_UP", "NORMAL", "bullish"): "bull_put_spread",

        ("TRENDING_DOWN", "ELEVATED", "bearish"): "bear_call_spread",
        ("TRENDING_DOWN", "LOW", "bearish"): "debit_put_spread",
        ("TRENDING_DOWN", "NORMAL", "bearish"): "bear_call_spread",

        ("RANGE_BOUND", "ELEVATED", "neutral"): "iron_condor",
        ("RANGE_BOUND", "EXTREME", "neutral"): "iron_condor_narrow",
        ("RANGE_BOUND", "NORMAL", "neutral"): "iron_butterfly",
        ("RANGE_BOUND", "LOW", "neutral"): "long_straddle",  # Rare

        ("HIGH_VOLATILITY", "EXTREME", "any"): "debit_spread_only",
        ("HIGH_VOLATILITY", "ELEVATED", "any"): "defined_risk_credit",

        ("TRANSITION", "any", "any"): "small_debit_spread"
    }

    key = (regime, iv_analysis["level"], direction_bias)
    strategy = strategy_matrix.get(key, "no_trade")

    if strategy == "no_trade":
        return {
            "action": "NO_TRADE",
            "reason": f"No strategy for {key} combination"
        }

    return build_trade_plan(strategy, iv_analysis, risk_budget)
```

---

## 5. RISK MANAGEMENT (NON-OVERRIDABLE)

### 5.1 Trade-Level Limits

```python
RISK_LIMITS = {
    "trade_level": {
        "max_loss_pct_account": 2.0,  # Max 2% loss per trade
        "max_loss_pct_high_conviction": 3.0,  # Allow 3% for A+ setups
        "defined_risk_required": True,  # No naked positions
        "max_contracts_per_trade": 20,  # Absolute cap
    },

    "portfolio_level": {
        "max_delta_exposure": 50,  # SPY-equivalent deltas
        "max_gamma_exposure": 10,  # Negative gamma limit
        "max_theta_daily_pct": 0.5,  # Max daily theta decay
        "max_vega_long": 100,
        "max_vega_short": 50,
        "max_correlated_positions": 3,  # Same-thesis limit
        "max_daily_loss_pct": 3.0,
        "max_weekly_loss_pct": 5.0
    }
}
```

### 5.2 Position Sizing Formula

```python
def calculate_position_size(
    account_value: float,
    max_loss_per_contract: float,
    confidence: str,
    regime: str
) -> dict:
    """
    Calculate position size based on risk and confidence

    Formula:
    Max Contracts = (Account * Risk%) / Max Loss per Contract
    """

    # Base risk percentage by confidence
    risk_pct = {
        "VERY_HIGH": 0.03,  # 3% for A+ setups
        "HIGH": 0.02,       # 2% standard
        "MODERATE": 0.015,  # 1.5%
        "LOW": 0.01         # 1% or skip
    }.get(confidence, 0.01)

    # Regime adjustment
    regime_multiplier = {
        "TRENDING_UP": 1.0,
        "TRENDING_DOWN": 1.0,
        "RANGE_BOUND": 1.0,
        "HIGH_VOLATILITY": 0.5,  # Half size
        "TRANSITION": 0.5
    }.get(regime, 0.5)

    adjusted_risk = risk_pct * regime_multiplier
    max_risk_dollars = account_value * adjusted_risk
    max_contracts = int(max_risk_dollars / max_loss_per_contract)

    # Apply absolute cap
    max_contracts = min(max_contracts, RISK_LIMITS["trade_level"]["max_contracts_per_trade"])

    return {
        "max_contracts": max_contracts,
        "risk_per_contract": max_loss_per_contract,
        "total_risk": max_contracts * max_loss_per_contract,
        "risk_pct_used": adjusted_risk * 100,
        "regime_adjustment": regime_multiplier
    }
```

### 5.3 Greeks Monitoring

```python
def check_portfolio_greeks(positions: list, account_value: float) -> dict:
    """
    Check if portfolio Greeks are within limits

    Returns warnings and required actions
    """
    total_delta = sum(p["delta"] * p["quantity"] * 100 for p in positions)
    total_gamma = sum(p["gamma"] * p["quantity"] * 100 for p in positions)
    total_theta = sum(p["theta"] * p["quantity"] * 100 for p in positions)
    total_vega = sum(p["vega"] * p["quantity"] * 100 for p in positions)

    warnings = []
    actions = []

    # Delta check
    if abs(total_delta) > RISK_LIMITS["portfolio_level"]["max_delta_exposure"]:
        warnings.append(f"DELTA LIMIT BREACHED: {total_delta:.0f} vs limit {RISK_LIMITS['portfolio_level']['max_delta_exposure']}")
        actions.append("REDUCE_OR_HEDGE_DELTA")

    # Negative gamma check (dangerous near expiry)
    if total_gamma < -RISK_LIMITS["portfolio_level"]["max_gamma_exposure"]:
        warnings.append(f"HIGH NEGATIVE GAMMA: {total_gamma:.1f}")
        actions.append("CLOSE_SHORT_OPTIONS_NEAR_EXPIRY")

    # Theta check
    theta_pct = abs(total_theta) / account_value * 100
    if theta_pct > RISK_LIMITS["portfolio_level"]["max_theta_daily_pct"]:
        warnings.append(f"THETA CONCENTRATION: {theta_pct:.2f}% daily")
        actions.append("DIVERSIFY_EXPIRIES")

    # Vega check
    if total_vega > RISK_LIMITS["portfolio_level"]["max_vega_long"]:
        warnings.append(f"LONG VEGA LIMIT: {total_vega:.0f}")
        actions.append("REDUCE_LONG_VEGA")
    elif total_vega < -RISK_LIMITS["portfolio_level"]["max_vega_short"]:
        warnings.append(f"SHORT VEGA LIMIT: {total_vega:.0f}")
        actions.append("REDUCE_SHORT_VEGA_VOL_SPIKE_RISK")

    return {
        "greeks": {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega
        },
        "warnings": warnings,
        "required_actions": actions,
        "status": "OK" if not warnings else "ACTION_REQUIRED"
    }
```

---

## 6. LIQUIDITY REQUIREMENTS

### 6.1 Minimum Thresholds

```python
LIQUIDITY_REQUIREMENTS = {
    "minimum": {
        "bid_ask_spread_pct": 10.0,  # Max 10% of mid
        "volume_daily": 100,
        "open_interest": 500
    },
    "preferred": {
        "bid_ask_spread_pct": 5.0,
        "volume_daily": 500,
        "open_interest": 2000
    },
    "optimal": {
        "bid_ask_spread_pct": 2.0,
        "volume_daily": 1000,
        "open_interest": 5000
    }
}

def check_option_liquidity(option: dict) -> dict:
    """
    Check if option meets liquidity requirements

    Returns liquidity score and pass/fail
    """
    bid = option["bid"]
    ask = option["ask"]
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100

    volume = option["volume"]
    oi = option["open_interest"]

    # Check minimum thresholds
    if spread_pct > LIQUIDITY_REQUIREMENTS["minimum"]["bid_ask_spread_pct"]:
        return {"pass": False, "reason": f"Spread too wide: {spread_pct:.1f}%"}
    if volume < LIQUIDITY_REQUIREMENTS["minimum"]["volume_daily"]:
        return {"pass": False, "reason": f"Volume too low: {volume}"}
    if oi < LIQUIDITY_REQUIREMENTS["minimum"]["open_interest"]:
        return {"pass": False, "reason": f"OI too low: {oi}"}

    # Calculate score
    score = 0
    if spread_pct <= LIQUIDITY_REQUIREMENTS["optimal"]["bid_ask_spread_pct"]:
        score += 40
    elif spread_pct <= LIQUIDITY_REQUIREMENTS["preferred"]["bid_ask_spread_pct"]:
        score += 25
    else:
        score += 10

    if volume >= LIQUIDITY_REQUIREMENTS["optimal"]["volume_daily"]:
        score += 30
    elif volume >= LIQUIDITY_REQUIREMENTS["preferred"]["volume_daily"]:
        score += 20
    else:
        score += 10

    if oi >= LIQUIDITY_REQUIREMENTS["optimal"]["open_interest"]:
        score += 30
    elif oi >= LIQUIDITY_REQUIREMENTS["preferred"]["open_interest"]:
        score += 20
    else:
        score += 10

    return {
        "pass": True,
        "score": score,
        "spread_pct": spread_pct,
        "volume": volume,
        "open_interest": oi,
        "quality": "OPTIMAL" if score >= 80 else "PREFERRED" if score >= 60 else "MINIMUM"
    }
```

---

## 7. ENTRY & EXIT MECHANICS

### 7.1 Entry Rules

```python
ENTRY_RULES = {
    "order_type": "LIMIT_ONLY",

    "pricing": {
        "start_at": "mid",
        "walk_increment": 0.01,  # $0.01 per adjustment
        "walk_interval_seconds": 30,
        "max_walk_pct": 20  # Max 20% of spread width
    },

    "timing": {
        "avoid_open": {"start": "09:30", "end": "09:45"},
        "avoid_close": {"start": "15:45", "end": "16:00"},
        "optimal_windows": [
            {"start": "10:00", "end": "11:30"},
            {"start": "14:00", "end": "15:30"}
        ]
    },

    "scaling": {
        "default": {"initial": 1.0, "add": 0.0},  # Full position
        "high_volatility": {"initial": 0.5, "add": 0.5},  # Scale in
        "event_risk": {"initial": 0.5, "add": 0.0}  # Half size only
    }
}
```

### 7.2 Exit Rules

```python
EXIT_RULES = {
    "profit_target": {
        "credit_spread": 50,      # Close at 50% of max profit
        "debit_spread": 75,       # Close at 75% of max profit
        "iron_condor": 50,        # Close at 50% of max profit
        "calendar": 25            # Close at 25% of max profit
    },

    "stop_loss": {
        "credit_spread": 200,     # Close at 2x credit received
        "debit_spread": 50,       # Close at 50% of debit paid
        "iron_condor": 200,       # Close at 2x credit
        "calendar": "front_ITM"   # Close if front month goes ITM
    },

    "time_stop": {
        "close_at_dte": 7,        # Close all positions at 7 DTE
        "exception": "profitable" # Unless already profitable
    },

    "adjustment_vs_close": {
        "regime_intact": "consider_adjustment",
        "regime_changed": "close_immediately",
        "near_expiry": "close",   # < 7 DTE always close
        "large_loss": "close"     # > 150% of credit
    }
}
```

### 7.3 Order Execution Flow

```python
def execute_entry(trade_plan: dict, current_prices: dict) -> dict:
    """
    Execute entry order with price improvement attempts
    """
    bid = current_prices["bid"]
    ask = current_prices["ask"]
    mid = (bid + ask) / 2
    spread_width = ask - bid

    # Start at mid
    limit_price = mid
    max_walk = spread_width * ENTRY_RULES["pricing"]["max_walk_pct"] / 100

    attempts = []
    filled = False

    while not filled and (limit_price - mid) <= max_walk:
        # Submit order
        order_result = submit_limit_order(trade_plan, limit_price)
        attempts.append({"price": limit_price, "result": order_result})

        if order_result["filled"]:
            filled = True
            break

        # Wait and walk
        time.sleep(ENTRY_RULES["pricing"]["walk_interval_seconds"])
        limit_price += ENTRY_RULES["pricing"]["walk_increment"]

    return {
        "filled": filled,
        "fill_price": limit_price if filled else None,
        "attempts": len(attempts),
        "slippage": (limit_price - mid) if filled else None
    }
```

---

## 8. EVENT RISK CALENDAR

### 8.1 Event Matrix

```python
EVENT_RISK_MATRIX = {
    "FOMC": {
        "frequency": "8x per year",
        "lockout_days_before": 2,
        "actions": [
            "NO new positions 2 days before",
            "CLOSE short gamma 1 day before",
            "IV typically inflated - sell premium AFTER event"
        ],
        "risk_level": "HIGH"
    },

    "CPI": {
        "frequency": "monthly",
        "lockout_days_before": 1,
        "actions": [
            "NO new positions 1 day before",
            "REDUCE size if holding through"
        ],
        "risk_level": "HIGH"
    },

    "NFP": {
        "frequency": "monthly (first Friday)",
        "lockout_days_before": 0,
        "actions": [
            "Normal caution",
            "No lockout needed for SPY"
        ],
        "risk_level": "MODERATE"
    },

    "OPEX": {
        "frequency": "monthly (3rd Friday)",
        "lockout_days_before": 0,
        "actions": [
            "AVOID 0-3 DTE short options",
            "PIN RISK awareness",
            "ROLL or CLOSE by Wednesday before"
        ],
        "risk_level": "MODERATE"
    },

    "VIX_EXPIRY": {
        "frequency": "monthly (Wednesday before OPEX)",
        "lockout_days_before": 0,
        "actions": [
            "Can cause intraday vol spikes",
            "NO new vega positions that day"
        ],
        "risk_level": "LOW"
    },

    "EARNINGS_SEASON": {
        "frequency": "quarterly",
        "lockout_days_before": 0,
        "actions": [
            "SPY less affected but correlations spike",
            "REDUCE size",
            "WIDEN strikes"
        ],
        "risk_level": "LOW"
    }
}
```

### 8.2 Event Check Function

```python
def check_event_risk(target_date: date) -> dict:
    """
    Check for upcoming events that affect trading

    Returns event risk assessment and required actions
    """
    events = get_economic_calendar(target_date, days_ahead=5)

    risk_events = []
    blocked = False

    for event in events:
        config = EVENT_RISK_MATRIX.get(event["type"])
        if not config:
            continue

        days_until = (event["date"] - target_date).days

        if days_until <= config["lockout_days_before"]:
            risk_events.append({
                "event": event["type"],
                "date": event["date"],
                "days_until": days_until,
                "actions": config["actions"],
                "risk_level": config["risk_level"]
            })

            if config["risk_level"] == "HIGH":
                blocked = True

    return {
        "blocked": blocked,
        "events": risk_events,
        "recommendation": "NO_TRADE" if blocked else "PROCEED_WITH_CAUTION" if risk_events else "CLEAR"
    }
```

---

## 9. DATA SOURCES

### 9.1 Required Data & Sources

```python
DATA_SOURCES = {
    "TIER_1_FREE": {
        "yahoo_finance": {
            "data": ["OHLCV", "basic_options_chain", "IV_estimate"],
            "limitations": ["15min delay", "no Greeks", "unreliable IV"],
            "python_package": "yfinance"
        },
        "cboe": {
            "data": ["VIX", "VIX3M", "VIX term structure"],
            "limitations": ["delayed"],
            "api": "https://www.cboe.com/delayed_quotes/"
        },
        "fred": {
            "data": ["risk_free_rate", "economic_indicators"],
            "python_package": "fredapi"
        }
    },

    "TIER_2_FREEMIUM": {
        "tradier": {
            "data": ["real-time quotes", "options_chain", "Greeks", "IV"],
            "cost": "FREE with brokerage account",
            "quality": "HIGH",
            "documentation": "https://documentation.tradier.com/"
        },
        "polygon": {
            "data": ["real-time quotes", "options", "historical"],
            "cost": "$29-99/mo",
            "quality": "HIGH"
        },
        "alpha_vantage": {
            "data": ["options_chain", "historical"],
            "cost": "Free tier available",
            "limitations": ["rate limits"]
        }
    },

    "TIER_3_PREMIUM": {
        "orats": {
            "data": ["IV surfaces", "historical IV", "Greeks", "analytics"],
            "cost": "$99-299/mo",
            "quality": "INSTITUTIONAL"
        },
        "databento": {
            "data": ["tick data", "full options chain"],
            "cost": "Pay per use",
            "quality": "INSTITUTIONAL"
        }
    }
}

RECOMMENDED_SETUP = {
    "minimum_viable": {
        "sources": ["yahoo_finance", "cboe"],
        "cost": "$0/mo",
        "limitations": ["delayed data", "calculate Greeks manually"]
    },
    "recommended": {
        "sources": ["tradier", "cboe", "fred"],
        "cost": "$0 (with Tradier brokerage)",
        "benefits": ["real-time", "Greeks included", "reliable"]
    },
    "optimal": {
        "sources": ["tradier", "orats", "cboe"],
        "cost": "$99-299/mo",
        "benefits": ["institutional quality", "IV analytics", "historical data"]
    }
}
```

### 9.2 Data Collection Module Structure

```python
# collectors/options_collector.py

class SPYOptionsCollector:
    """
    Unified interface for SPY options data collection
    Handles multiple data sources with fallback
    """

    def __init__(self, primary_source: str = "tradier"):
        self.primary = primary_source
        self.sources = self._init_sources()

    def get_options_chain(self, expiry_date: date = None) -> pd.DataFrame:
        """Get full options chain with Greeks and IV"""
        pass

    def get_iv_metrics(self) -> dict:
        """Get IV rank, percentile, term structure"""
        pass

    def get_greeks(self, option_symbol: str) -> dict:
        """Get Greeks for specific option"""
        pass

    def get_vix_data(self) -> dict:
        """Get VIX and term structure"""
        pass
```

---

## 10. OUTPUT FORMAT

### 10.1 Trade Recommendation Structure

```python
TRADE_RECOMMENDATION = {
    "timestamp": "2024-01-15T10:30:00Z",
    "status": "TRADE_RECOMMENDED",  # or "NO_TRADE"

    "market_analysis": {
        "regime": "RANGE_BOUND",
        "regime_confidence": 75,
        "iv_rank": 62,
        "iv_assessment": "ELEVATED",
        "term_structure": "CONTANGO",
        "skew": "NORMAL"
    },

    "strategy": {
        "type": "IRON_CONDOR",
        "reasoning": "Range-bound regime with elevated IV favors premium selling"
    },

    "trade_details": {
        "legs": [
            {"action": "SELL", "strike": 580, "type": "PUT", "expiry": "2024-02-16"},
            {"action": "BUY", "strike": 575, "type": "PUT", "expiry": "2024-02-16"},
            {"action": "SELL", "strike": 600, "type": "CALL", "expiry": "2024-02-16"},
            {"action": "BUY", "strike": 605, "type": "CALL", "expiry": "2024-02-16"}
        ],
        "dte": 32,
        "credit": 1.85,
        "max_loss": 3.15,
        "max_profit": 1.85
    },

    "risk_management": {
        "position_size": 3,
        "total_risk": 945,
        "risk_pct_account": 1.89,
        "profit_target": 0.93,  # 50% of credit
        "stop_loss": 3.70       # 2x credit
    },

    "execution": {
        "entry_method": "LIMIT at mid",
        "max_walk": 0.10,
        "optimal_window": "10:00-11:30 ET"
    },

    "exit_rules": {
        "profit_target_pct": 50,
        "stop_loss_trigger": "2x credit or short strike breach",
        "time_stop": "7 DTE"
    },

    "invalidation": [
        "VIX spikes above 25",
        "SPY breaks outside 575-605 range",
        "Regime change to TRENDING"
    ],

    "confidence": "HIGH",
    "grade": "B+"
}
```

### 10.2 No-Trade Output

```python
NO_TRADE_OUTPUT = {
    "timestamp": "2024-01-15T10:30:00Z",
    "status": "NO_TRADE",

    "reasons": [
        "Event risk: FOMC meeting in 2 days",
        "Regime: TRANSITION (confidence 55%)",
        "IV Rank: 28 (neither elevated nor low)"
    ],

    "market_summary": {
        "regime": "TRANSITION",
        "iv_rank": 28,
        "vix": 16.5,
        "spy_price": 590.25
    },

    "next_action": "Re-evaluate after FOMC announcement",
    "estimated_wait": "2-3 days"
}
```

---

## 11. DECISION HIERARCHY

### 11.1 Conflict Resolution Order

```python
DECISION_HIERARCHY = [
    # 1. RISK LIMITS (Non-negotiable)
    {
        "check": "portfolio_risk_limits",
        "action_if_fail": "NO_TRADE",
        "overridable": False
    },

    # 2. LIQUIDITY (Non-negotiable)
    {
        "check": "option_liquidity",
        "action_if_fail": "NO_TRADE",
        "overridable": False
    },

    # 3. EVENT RISK
    {
        "check": "event_calendar",
        "action_if_fail": "NO_TRADE or REDUCE_SIZE",
        "overridable": False
    },

    # 4. REGIME ALIGNMENT
    {
        "check": "strategy_matches_regime",
        "action_if_fail": "NO_TRADE",
        "overridable": False
    },

    # 5. IV CONDITIONS
    {
        "check": "iv_favorable_for_strategy",
        "action_if_fail": "ADJUST_STRATEGY or NO_TRADE",
        "overridable": True  # Can adjust strategy
    },

    # 6. SIGNAL STRENGTH
    {
        "check": "signal_quality",
        "action_if_fail": "REDUCE_SIZE or NO_TRADE",
        "overridable": True
    }
]
```

---

## 12. LOGGING & LEARNING

### 12.1 Trade Log Structure

```python
TRADE_LOG_SCHEMA = {
    "trade_id": "uuid",
    "entry_timestamp": "datetime",
    "exit_timestamp": "datetime",

    # Context at entry
    "regime_at_entry": "string",
    "iv_rank_at_entry": "float",
    "vix_at_entry": "float",
    "spy_price_at_entry": "float",

    # Trade details
    "strategy_type": "string",
    "legs": "json",
    "entry_price": "float",
    "exit_price": "float",

    # Greeks at entry
    "delta_at_entry": "float",
    "gamma_at_entry": "float",
    "theta_at_entry": "float",
    "vega_at_entry": "float",

    # Outcome
    "pnl_dollars": "float",
    "pnl_pct": "float",
    "max_drawdown": "float",
    "hold_duration_days": "int",

    # Labels
    "outcome_label": "WIN|LOSS|SCRATCH",
    "exit_reason": "PROFIT_TARGET|STOP_LOSS|TIME_STOP|MANUAL",
    "notes": "string"
}
```

### 12.2 Performance Analytics

```python
def calculate_performance_metrics(trades: list) -> dict:
    """
    Calculate key performance metrics
    """
    wins = [t for t in trades if t["pnl_dollars"] > 0]
    losses = [t for t in trades if t["pnl_dollars"] < 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(t["pnl_dollars"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_dollars"] for t in losses) / len(losses) if losses else 0

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Profit factor
    gross_profit = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity_curve = calculate_equity_curve(trades)
    max_dd = calculate_max_drawdown(equity_curve)

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "sharpe_ratio": calculate_sharpe(trades)
    }
```

### 12.3 Strategy Performance by Regime

```python
def analyze_strategy_by_regime(trades: list) -> dict:
    """
    Break down performance by regime and strategy

    Identify what works and what doesn't
    """
    results = {}

    for trade in trades:
        regime = trade["regime_at_entry"]
        strategy = trade["strategy_type"]
        key = f"{regime}_{strategy}"

        if key not in results:
            results[key] = []
        results[key].append(trade)

    # Calculate metrics for each combo
    analysis = {}
    for key, trade_list in results.items():
        metrics = calculate_performance_metrics(trade_list)
        analysis[key] = {
            **metrics,
            "recommendation": "CONTINUE" if metrics["expectancy"] > 0 else "STOP"
        }

    return analysis
```

---

## 13. FORBIDDEN BEHAVIORS

```python
FORBIDDEN = [
    "Chase 'highest win rate' without considering expectancy",
    "Stack indicators without regime context",
    "Recommend naked options by default",
    "Use news headlines as primary signals",
    "Override risk rules for 'confidence'",
    "Trade during event lockout periods",
    "Ignore liquidity requirements",
    "Add to losing positions without plan",
    "Exceed portfolio Greek limits",
    "Predict specific price targets"
]
```

---

## 14. SUCCESS CRITERIA

```python
SUCCESS_METRICS = {
    "primary": {
        "expectancy_positive": True,
        "max_drawdown_lt": 15,  # < 15%
        "profit_factor_gt": 1.3,
        "win_rate_range": (0.45, 0.75)  # Not too low, not chasing high
    },

    "secondary": {
        "trades_per_month": (4, 20),  # Not overtrading
        "regime_alignment_rate": 0.90,  # 90%+ of trades match regime
        "risk_compliance_rate": 1.0  # 100% within risk limits
    },

    "qualitative": {
        "decisions_explainable": True,
        "adapts_to_regime": True,
        "avoids_low_quality_trades": True
    }
}
```

---

## 15. IMPLEMENTATION PHASES

### Phase 1: Foundation (Week 1-2)
- [ ] Data collectors (Yahoo, CBOE VIX)
- [ ] Regime classification engine
- [ ] Basic IV analysis

### Phase 2: Strategy Engine (Week 3-4)
- [ ] Strategy selection logic
- [ ] Position sizing calculator
- [ ] Liquidity checker

### Phase 3: Risk Management (Week 5-6)
- [ ] Greeks monitoring
- [ ] Event calendar integration
- [ ] Portfolio limits enforcement

### Phase 4: Execution & Alerts (Week 7-8)
- [ ] Trade recommendation generator
- [ ] Telegram alerts
- [ ] Trade logging

### Phase 5: Learning & Optimization (Ongoing)
- [ ] Performance tracking
- [ ] Strategy analysis by regime
- [ ] Parameter tuning

---

## APPENDIX A: Sample Code Structure

```
spy-options-agent/
├── collectors/
│   ├── __init__.py
│   ├── yahoo_collector.py      # Free price data
│   ├── options_collector.py    # Options chain (Tradier/Yahoo)
│   ├── vix_collector.py        # VIX data from CBOE
│   └── calendar_collector.py   # Economic calendar
├── analysis/
│   ├── __init__.py
│   ├── regime_classifier.py    # Market regime detection
│   ├── iv_analyzer.py          # IV analysis
│   ├── greeks_calculator.py    # Greeks computation
│   └── liquidity_checker.py    # Liquidity validation
├── strategy/
│   ├── __init__.py
│   ├── strategy_selector.py    # Strategy selection
│   ├── position_sizer.py       # Position sizing
│   └── trade_builder.py        # Build trade plans
├── risk/
│   ├── __init__.py
│   ├── risk_manager.py         # Risk limits enforcement
│   ├── portfolio_monitor.py    # Greeks monitoring
│   └── event_risk.py           # Event calendar checks
├── execution/
│   ├── __init__.py
│   ├── order_manager.py        # Order execution
│   └── exit_manager.py         # Exit rule enforcement
├── alerts/
│   ├── __init__.py
│   ├── telegram_alert.py       # Telegram notifications
│   └── email_alert.py          # Email alerts
├── logging/
│   ├── __init__.py
│   ├── trade_logger.py         # Trade logging
│   └── performance_tracker.py  # Performance analytics
├── config/
│   ├── __init__.py
│   ├── settings.py             # All configuration
│   └── thresholds.py           # Quantitative thresholds
├── agent.py                    # Main agent
└── requirements.txt
```

---

**Document Version:** 2.0
**Last Updated:** 2024-01-15
**Status:** Implementation Ready
