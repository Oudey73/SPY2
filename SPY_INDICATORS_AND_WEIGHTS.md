# SPY Options Trading Agent - Indicators & Weights Documentation

## Overview

This document describes all technical indicators, signal types, and scoring weights used by the SPY Options Trading Agent. The strategy is based on **IBS + RSI Mean Reversion** with VIX filtering, optimized through backtesting on 5 years of data (2020-2024).

**Historical Performance:**
- Win Rate: ~71% (when VIX filter passes)
- Typical Hold: 1-5 days
- Strategy Type: Mean Reversion (buying oversold, selling overbought)

---

## TIER 1 - PRIMARY SIGNALS (Highest Priority)

These are the core signals with the strongest backtested edge.

### IBS (Internal Bar Strength)

**Formula:** `IBS = (Close - Low) / (High - Low)`

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `ibs_extreme_low` | IBS < 0.15 | LONG | 30 | Extreme oversold, high probability bounce |
| `ibs_oversold` | IBS < 0.20 | LONG | 25 | Oversold, mean reversion expected |
| `ibs_overbought` | IBS > 0.80 | SHORT | 25 | Overbought, weakness expected |
| `ibs_extreme_high` | IBS > 0.85 | SHORT | 30 | Extreme overbought, pullback likely |

**Research:** IBS < 0.2 shows next day average return of +0.35%, IBS > 0.8 shows -0.13%

### IBS + RSI Combo (Most Powerful)

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `ibs_rsi_combo_long` | IBS < 0.2 AND RSI(3) < 30 | LONG | 40 | High conviction long setup |
| `ibs_rsi_combo_short` | IBS > 0.8 AND RSI(3) > 70 | SHORT | 40 | High conviction short setup |

**Research:** This combo improves returns by 9.6 percentage points vs IBS alone

### RSI(2) - Key PUT Signal

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `rsi2_extreme_high` | RSI(2) >= 98 | SHORT | 35 | PRIMARY PUT signal - 94% of winning PUTs have this |
| `rsi2_overbought` | RSI(2) >= 95 | SHORT | 20 | PUT opportunity |

**Research:** RSI(2) >= 98 present in 94% of profitable PUT trades (backtest 2020-2024)

---

## TIER 2 - CONFIRMING SIGNALS

These signals add conviction when combined with Tier 1 signals.

### RSI(3) - Short-term Momentum

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `rsi_extreme_low` | RSI(3) < 10 | LONG | 15 | Deep oversold |
| `rsi_oversold` | RSI(3) < 30 | LONG | 10 | Oversold condition |
| `rsi_overbought` | RSI(3) > 70 | SHORT | 10 | Overbought condition |
| `rsi_extreme_high` | RSI(3) > 90 | SHORT | 15 | Deep overbought |

**Note:** RSI(3) threshold was optimized from 20 to 30 to capture more opportunities while maintaining edge.

### VIX - Fear/Volatility Filter

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `vix_extreme` | VIX > 30 | LONG | 15 | Panic mode - high probability bounce |
| `vix_elevated` | VIX > 18 AND VIX > 10-day MA | LONG | 10 | Elevated fear - good for mean reversion longs |

**Research:** VIX 20-25 range has highest win rate for LONG entries. VIX acts as a FILTER, not standalone signal.

### Consecutive Days - Momentum Exhaustion

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `consecutive_up_4` | 4+ consecutive up days | SHORT | 15 | Extended rally - PUT opportunity |
| `consecutive_up_3` | 3 consecutive up days | SHORT | 8 | Rally extended |

**Research:** 4+ consecutive up days present in 58% of profitable PUT trades

### Intraday Momentum

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `intraday_momentum_bullish` | Day up > 0.3% | LONG | 10 | Momentum continuation expected |
| `intraday_momentum_bearish` | Day down > 0.3% | SHORT | 10 | Weakness continuation expected |

**Research:** Sharpe ratio of 1.33 (2007-2024 data) for last 30-min momentum continuation

---

## TIER 3 - CONTEXT SIGNALS

These provide market context and can add small bonuses or penalties.

### Moving Average Context

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `below_200_ma` | Price < 200-day SMA | LONG | 8 | Deep correction zone - value context |
| `below_50_ma` | Price < 50-day SMA | LONG | 5 | Mean reversion context |
| `above_50_ma` | Price > 50-day SMA | NEUTRAL | 0 | Trend intact - no adjustment |

### VIX Complacency Warning

| Signal | Threshold | Direction | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| `vix_complacent` | VIX < 15 | NEUTRAL | -5 | PENALTY for LONG trades - market may be overbought |

**Note:** Low VIX is actually slightly positive for SHORT trades (contrarian signal). Backtest shows VIX < 14 present in 30% of profitable PUT trades.

---

## ORATS IV ANALYTICS (Professional Options Data)

These signals come from ORATS API and provide options-specific insights.

### IV Rank - Premium Pricing

| Signal | Threshold | Strategy Bias | Description |
|--------|-----------|---------------|-------------|
| `iv_extreme` | IV Rank >= 80 | SELL_PREMIUM | Premium is expensive - favor selling options |
| `iv_high` | IV Rank >= 70 | SELL_PREMIUM | Elevated premium - slight sell bias |
| `iv_low` | IV Rank <= 30 | BUY_PREMIUM | Premium is cheap - favor buying options |

### Term Structure - Stress Detection

| Signal | Condition | Strategy Bias | Description |
|--------|-----------|---------------|-------------|
| `iv_backwardation` | 90d IV < 30d IV (spread < -2) | CAUTION | Inverted term structure - market stress signal |
| `iv_contango` | 90d IV > 30d IV (spread > 2) | SELL_FRONT_MONTH | Normal structure - sell near-term premium |

### Skew Analysis - Hedging Demand

| Signal | Condition | Strategy Bias | Description |
|--------|-----------|---------------|-------------|
| `iv_steep_put_skew` | Skewing < -0.1 OR put_skew > 5 | SELL_PUT_SPREADS | High hedging demand - sell put spreads |

---

## SCORING FORMULA

```
Base Score = Sum of (Signal Weight × Signal Strength / 100) for aligned signals

Adjustments:
- Conflict Penalty: -5 points per conflicting signal
- VIX Filter Bonus: +10 points for LONG if VIX filter passed
- RSI(2) Bonus: +5 points for SHORT if RSI(2) >= 98

Final Score = Base Score - Conflict Penalty + Bonuses
Score Range: 0-100
```

### Signal Strength Calculation

Each signal has a dynamic strength (0-100) based on how extreme the value is:
- More extreme values = higher strength
- Example: IBS of 0.05 has higher strength than IBS of 0.18 (both trigger oversold)

---

## OPPORTUNITY GRADES

| Grade | Score Range | Historical Win Rate | Recommended Position Size |
|-------|-------------|---------------------|---------------------------|
| **A+** | 80-100 | ~75% (LONG), ~50% (SHORT) | 3-5% of portfolio |
| **A** | 65-79 | ~70% | 2-3% of portfolio |
| **B** | 50-64 | ~65% | 1% of portfolio |
| **C** | 35-49 | ~55% | Paper trade only |
| **F** | < 35 | N/A | No trade |

---

## TRADE PARAMETERS

### LONG (CALL) Strategy
- Stop Loss: 1.5%
- Target 1: 2%
- Target 2: 3%
- Target 3: 4%
- Risk/Reward to T2: 2:1

### SHORT (PUT) Strategy
- Stop Loss: 1.0% (tighter for PUTs)
- Target 1: 2%
- Target 2: 3%
- Target 3: 4%
- Risk/Reward to T2: 3:1

### Options DTE Recommendations
| Score | Recommended DTE | Rationale |
|-------|-----------------|-----------|
| 80+ | 7-10 DTE | High conviction - tighter expiry OK |
| 65-79 | 10-14 DTE | Solid setup - standard buffer |
| 50-64 | 14-21 DTE | Moderate conviction - extra time buffer |
| < 50 | 21+ DTE or avoid | Low conviction |

---

## KNOWN LIMITATIONS & AREAS FOR IMPROVEMENT

1. **ORATS signals not yet integrated into scoring** - Currently logged but not weighted in final score
2. **No machine learning** - Pure rule-based system
3. **Single asset (SPY only)** - No correlation with other assets
4. **No earnings/events calendar** - Doesn't account for scheduled volatility events
5. **No volume analysis** - Price-based only
6. **No options flow data** - Could add unusual options activity signals
7. **Fixed position sizing** - Could be dynamic based on Kelly Criterion
8. **No portfolio-level risk management** - Single trade focus

---

## BACKTEST RESULTS (2020-2024)

### Overall Performance
- Total Trades: ~50/year (A+ grade only)
- Win Rate: 70.8% (A+ trades)
- LONG Win Rate: 75%
- SHORT Win Rate: 50%

### Best Performing Conditions
1. IBS < 0.2 + RSI(3) < 30 + VIX > 10-MA: 70% win rate
2. RSI(2) >= 98 + 4+ consecutive up days: High probability PUT
3. VIX 20-25 range: Optimal for mean reversion entries

### Worst Performing Conditions
1. VIX < 14 with LONG signals: Lower win rate
2. SHORT signals in strong uptrend: Only 50% win rate
3. Conflicting signals: Significantly reduced edge

---

## SUGGESTED ENHANCEMENTS

1. **Add ORATS signals to scoring weights**
2. **Implement adaptive thresholds based on market regime**
3. **Add volume confirmation signals**
4. **Integrate earnings calendar to avoid trades before announcements**
5. **Add correlation with QQQ/IWM for confirmation**
6. **Implement dynamic position sizing based on signal confidence**
7. **Add options Greeks analysis (delta, gamma exposure)**
8. **Consider adding sentiment indicators (put/call ratio, etc.)**

---

*Document generated for enhancement review*
*Agent Version: 1.0*
*Last Updated: 2024-12-22*
