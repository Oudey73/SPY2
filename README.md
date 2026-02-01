# SPY Opportunity Agent

A Python-based trading signal agent that monitors SPY (S&P 500 ETF) for mean reversion opportunities using technical indicators and options analytics.

## Strategy Overview

**IBS + RSI(3) Mean Reversion with VIX Filter**

| Metric | Value |
|--------|-------|
| Historical Win Rate | ~71% |
| Typical Hold Period | 1-5 days |
| Best Entry Conditions | IBS < 0.2 + RSI(3) < 20 + VIX > 10-MA |

## Features

- Real-time SPY monitoring during market hours
- Multi-source data collection (Yahoo Finance, Polygon.io, ORATS)
- Technical signal detection (IBS, RSI, VIX)
- Professional IV analytics (IV Rank, Term Structure, Skew)
- Telegram and Email alerts
- Scoring system with grade assignments (A+ to F)
- Configurable scan intervals and thresholds

## Data Sources

| Source | API Key Required | What It Provides |
|--------|------------------|------------------|
| Yahoo Finance | No (Free) | Price, OHLCV, VIX, technicals |
| Polygon.io | Yes (Free tier available) | Real-time quotes, options data |
| ORATS | Yes (Paid) | Professional IV analytics |

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Oudey73/SPY.git
cd SPY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Configure your `.env` file with API keys and alert settings.

## Configuration

### Environment Variables

```env
# Required for alerts
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-telegram-chat-id

# Optional - Enhanced data
POLYGON_API_KEY=your-polygon-api-key
ORATS_API_KEY=your-orats-api-key

# Optional - Email alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
RECIPIENT_EMAIL=recipient@email.com
```

### Setting Up Telegram Alerts

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add the bot token and chat ID to your `.env` file

## Usage

### Run Continuous Monitoring
```bash
python agent.py
```

### Run Single Test Scan
```bash
python agent.py --test
```

### Custom Configuration
```bash
python agent.py --min-score 60 --interval 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--min-score` | 50 | Minimum score to trigger alert |
| `--interval` | 15 | Scan interval in minutes |
| `--test` | - | Run single scan and exit |

## Signal Detection

### Primary Signals (Tier 1)

| Signal | Condition | Direction |
|--------|-----------|-----------|
| IBS Extreme Low | IBS < 0.15 | LONG |
| IBS Extreme High | IBS > 0.85 | SHORT |
| IBS + RSI Combo | IBS < 0.2 + RSI(3) < 20 | LONG |
| IBS + RSI Combo | IBS > 0.8 + RSI(3) > 80 | SHORT |

### Confirming Signals (Tier 2)

| Signal | Condition | Effect |
|--------|-----------|--------|
| VIX Filter | VIX > 10-day MA | Confirms fear |
| RSI(2) Extreme | RSI(2) >= 95 | Strong SHORT |
| Consecutive Days | 4+ up days | SHORT bias |

### IV Signals (ORATS)

| Signal | Condition | Strategy Bias |
|--------|-----------|---------------|
| High IV Rank | IV Rank >= 70 | Sell premium |
| Low IV Rank | IV Rank < 30 | Buy premium |
| Backwardation | Near-term IV > Long-term | Stress signal |
| Steep Put Skew | High hedging demand | Sell put spreads |

## Scoring System

| Grade | Score Range | Action |
|-------|-------------|--------|
| A+ | 90-100 | Strong signal - High confidence |
| A | 80-89 | Good signal - Consider entry |
| B | 70-79 | Moderate signal |
| C | 60-69 | Weak signal |
| D | 50-59 | Minimal signal |
| F | < 50 | No action |

## Project Structure

```
SPY/
├── agent.py                 # Main monitoring agent
├── config.py                # Configuration and thresholds
├── collectors/
│   ├── yahoo_collector.py   # Yahoo Finance data
│   ├── polygon_collector.py # Polygon.io real-time data
│   └── orats_collector.py   # ORATS IV analytics
├── signals/
│   ├── signal_detector.py   # Technical signal detection
│   ├── iv_signal_detector.py# IV-based signals
│   └── opportunity_scorer.py# Scoring and grading
├── alerts/
│   ├── telegram_alert.py    # Telegram notifications
│   └── email_alert.py       # Email notifications
├── logs/                    # Daily log files
├── requirements.txt
├── Procfile                 # Heroku deployment
└── railway.toml             # Railway deployment
```

## Deployment

### Railway

The project is configured for Railway deployment:

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy - Railway will use `railway.toml` configuration

### Heroku

Use the included `Procfile`:
```
worker: python agent.py --interval 15
```

## Example Alert

```
=====================================
SPY OPPORTUNITY ALERT
=====================================
Direction: SHORT
Score: 100 (A+)
Price: $690.38

Signals:
  [1] ibs_extreme_high: SHORT (40)
  [2] rsi_extreme_high: SHORT (90)
  [1] rsi2_extreme_high: SHORT (100)
  [2] consecutive_up_4: SHORT (100)

IV Analytics:
  IV Rank: 37.0
  Term Structure: CONTANGO
  Skew: STEEP_PUT

Suggested Action: Consider PUT options
Stop Loss: 1.5%
Target: 2.0%
=====================================
```

## License

MIT

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.
