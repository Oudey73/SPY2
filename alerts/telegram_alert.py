"""
Telegram Alert System
Sends trading opportunity alerts via Telegram bot
"""
import html
import requests
from datetime import datetime
from typing import Optional, Dict
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlertSystem:
    """Sends alerts via Telegram bot"""

    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        # Track sent alerts to avoid spam
        self.sent_alerts: Dict[str, datetime] = {}
        self.cooldown_minutes = 30

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)

    def _send_request(self, method: str, data: dict) -> Optional[dict]:
        """Send request to Telegram API"""
        try:
            url = TELEGRAM_API.format(token=self.bot_token, method=method)
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            result = response.json()

            if not result.get("ok"):
                logger.error(f"Telegram API error: {result.get('description')}")
                return None

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram request error: {e}")
            return None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if sent successfully
        """
        if not self.is_configured():
            logger.error("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return False

        result = self._send_request("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        })

        if result:
            logger.info("Telegram message sent successfully")
            return True
        return False

    def send_opportunity_alert(self, opportunity) -> bool:
        """
        Send a trading opportunity alert

        Args:
            opportunity: Opportunity object from scorer

        Returns:
            True if sent successfully
        """
        # Create alert key for cooldown tracking
        alert_key = f"{opportunity.symbol}_{opportunity.direction.value}"

        if alert_key in self.sent_alerts:
            elapsed = (datetime.utcnow() - self.sent_alerts[alert_key]).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                logger.info(f"Telegram alert on cooldown ({elapsed:.0f}m < {self.cooldown_minutes}m)")
                return False

        # Format message
        message = self._format_opportunity_message(opportunity)

        # Send
        success = self.send_message(message)

        if success:
            self.sent_alerts[alert_key] = datetime.utcnow()

        return success

    def _format_opportunity_message(self, opportunity) -> str:
        """Format opportunity as Telegram message"""
        direction_emoji = "🟢" if opportunity.direction.value == "long" else "🔴"
        grade_emoji = {"A+": "🔥", "A": "⭐", "B": "👀", "C": "⚠️", "F": "🚫"}.get(opportunity.grade.value, "")

        # Handle different entry_zone key names
        entry_aggressive = opportunity.entry_zone.get('aggressive', opportunity.entry_zone.get('high', 0))
        entry_target = opportunity.entry_zone.get('target', opportunity.entry_zone.get('mid', 0))
        entry_conservative = opportunity.entry_zone.get('conservative', opportunity.entry_zone.get('low', 0))

        # Handle stop_loss vs invalidation
        stop_loss = getattr(opportunity, 'stop_loss', None) or getattr(opportunity, 'invalidation', 0)

        # VIX filter status (SPY specific)
        vix_status = ""
        if hasattr(opportunity, 'vix_filter_passed'):
            vix_status = "✅ VIX Filter Passed" if opportunity.vix_filter_passed else "⚠️ VIX Filter Warning"

        # Top drivers (escape HTML entities like < >)
        drivers_text = "\n".join([f"  • {html.escape(d)}" for d in opportunity.top_drivers[:3]])

        # Warnings
        warnings_text = ""
        if opportunity.warnings:
            warnings_text = "\n\n⚠️ <b>Warnings:</b>\n" + "\n".join([f"  • {html.escape(w)}" for w in opportunity.warnings])

        opp_id = getattr(opportunity, 'opp_id', '')
        id_line = f"\n🆔 <b>ID:</b> {opp_id}" if opp_id else ""

        message = f"""
{direction_emoji} <b>{opportunity.symbol} - {opportunity.direction.value.upper()}</b> {direction_emoji}{id_line}

📊 <b>Score:</b> {opportunity.score}/100 ({opportunity.grade.value}) {grade_emoji}
🎯 <b>Confidence:</b> {opportunity.confidence_level}
{vix_status}

📈 <b>Top Drivers:</b>
{drivers_text}

💰 <b>Entry Zone:</b>
  • Aggressive: ${entry_aggressive:,.2f}
  • Target: ${entry_target:,.2f}
  • Conservative: ${entry_conservative:,.2f}

🛑 <b>Stop Loss:</b> ${stop_loss:,.2f}

🎯 <b>Targets:</b>
  • T1: ${opportunity.targets['t1']:,.2f}
  • T2: ${opportunity.targets['t2']:,.2f}
  • T3: ${opportunity.targets['t3']:,.2f}

📐 <b>R:R:</b> {opportunity.risk_reward:.1f}
💼 <b>Size:</b> {opportunity.position_size_suggestion}

📅 <b>Options Expiry:</b> {opportunity.recommended_dte}
{warnings_text}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

<i>⚠️ Not financial advice. DYOR.</i>
"""
        return message.strip()

    def send_trade_recommendation(self, opportunity, trade_plan, regime) -> bool:
        """
        Send a structured trade recommendation with strategy, regime, and legs.

        Args:
            opportunity: Opportunity object from scorer
            trade_plan: TradePlan from trade builder
            regime: MarketRegime from classifier

        Returns:
            True if sent successfully
        """
        alert_key = f"{opportunity.symbol}_{opportunity.direction.value}_rec"

        if alert_key in self.sent_alerts:
            elapsed = (datetime.utcnow() - self.sent_alerts[alert_key]).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False

        direction_emoji = "🟢" if opportunity.direction.value == "long" else "🔴"
        grade_emoji = {"A+": "🔥", "A": "⭐", "B": "👀", "C": "⚠️", "F": "🚫"}.get(opportunity.grade.value, "")
        opp_id = getattr(opportunity, 'opp_id', '')

        # Format legs
        legs_text = ""
        for leg in trade_plan.legs:
            action = leg.action.value.upper()
            opt_type = leg.option_type.value.upper()
            legs_text += f"  • {action} {leg.quantity}x {opt_type} ${leg.strike:.0f} ({leg.expiration})\n"

        # Regime info
        regime_text = f"{regime.regime.value} (confidence {regime.confidence:.0f}%)"

        # Risk info
        max_loss_text = f"${trade_plan.max_loss:,.0f}" if trade_plan.max_loss else "N/A"

        id_line = f"\n🆔 <b>ID:</b> {opp_id}" if opp_id else ""

        message = f"""
{direction_emoji} <b>TRADE RECOMMENDATION: SPY {opportunity.direction.value.upper()}</b> {direction_emoji}{id_line}

📊 <b>Score:</b> {opportunity.score}/100 ({opportunity.grade.value}) {grade_emoji}

🏷 <b>Strategy:</b> {trade_plan.strategy.value.replace('_', ' ').title()}
📈 <b>Regime:</b> {regime_text}
📅 <b>Expiration:</b> {trade_plan.expiration} ({trade_plan.dte} DTE)

📋 <b>Legs:</b>
{legs_text}
💰 <b>Contracts:</b> {trade_plan.contracts}
🛑 <b>Max Loss:</b> {max_loss_text}
🎯 <b>Profit Target:</b> {trade_plan.profit_target_pct:.0f}% of max profit
❌ <b>Stop Loss:</b> {trade_plan.stop_loss_pct:.0f}% of max loss

📈 <b>Top Drivers:</b>
""" + "\n".join([f"  • {html.escape(d)}" for d in opportunity.top_drivers[:3]])

        if opportunity.warnings:
            message += "\n\n⚠️ <b>Warnings:</b>\n" + "\n".join([f"  • {html.escape(w)}" for w in opportunity.warnings])

        message += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        message += "\n\n<i>⚠️ Not financial advice. DYOR.</i>"

        success = self.send_message(message.strip())
        if success:
            self.sent_alerts[alert_key] = datetime.utcnow()
        return success

    def send_test_message(self) -> bool:
        """Send a test message to verify configuration"""
        message = """
🧪 <b>Test Alert - SPY Opportunity Agent</b>

✅ Telegram alerts are working!

You will receive alerts when high-conviction opportunities are detected.

⏰ {time}
""".format(time=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))

        return self.send_message(message)

    def send_heartbeat(self, status: Dict) -> bool:
        """Send periodic status heartbeat"""
        message = f"""
💓 <b>Agent Heartbeat</b>

📊 Scans: {status.get('scans_completed', 0)}
🔔 Alerts: {status.get('alerts_sent', 0)}
✅ Status: {'Healthy' if status.get('healthy', True) else 'Issues'}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send_message(message)

    def send_exit_alert(self, alert) -> bool:
        """
        Send an exit alert for a tracked position.

        Args:
            alert: ExitAlert object with exit details

        Returns:
            True if sent successfully
        """
        # Determine alert emoji and urgency based on type
        alert_type = alert.alert_type.value if hasattr(alert.alert_type, 'value') else str(alert.alert_type)

        type_config = {
            "partial_exit": ("💰", "PARTIAL EXIT", "Take profits on 50%"),
            "full_exit": ("🚨", "FULL EXIT", "Close entire position"),
            "stop_loss": ("🛑", "STOP LOSS", "Stop loss triggered"),
            "target_hit": ("🎯", "TARGET HIT", "Profit target reached"),
        }

        emoji, title, action = type_config.get(
            alert_type,
            ("⚠️", "EXIT ALERT", "Review position")
        )

        # P&L indicator
        pnl_emoji = "🟢" if alert.pnl_percent >= 0 else "🔴"

        message = f"""
{emoji} <b>{title}: SPY</b> {emoji}

🆔 <b>ID:</b> {alert.opp_id}
📊 <b>Action:</b> {action} ({alert.exit_percentage}%)

💵 <b>Entry:</b> ${alert.entry_price:,.2f}
📈 <b>Current:</b> ${alert.current_price:,.2f}
{pnl_emoji} <b>P&L:</b> {alert.pnl_percent:+.2f}%

📊 <b>Bars Held:</b> {alert.bars_held}
📝 <b>Reason:</b> {html.escape(alert.reason)}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

<i>⚠️ Review and take appropriate action.</i>
"""
        return self.send_message(message.strip())


def test_telegram_alerts():
    """Test the Telegram alert system"""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "=" * 60)
    print("TELEGRAM ALERT SYSTEM TEST")
    print("=" * 60)

    telegram = TelegramAlertSystem()

    print(f"\nConfigured: {telegram.is_configured()}")

    if not telegram.is_configured():
        print("\n⚠️  Telegram not configured!")
        print("Set in .env file:")
        print("  TELEGRAM_BOT_TOKEN=your-bot-token")
        print("  TELEGRAM_CHAT_ID=your-chat-id")
        return False

    print("\nSending test message...")
    success = telegram.send_test_message()

    if success:
        print("✅ Test message sent! Check your Telegram.")
    else:
        print("❌ Failed to send test message.")

    return success


if __name__ == "__main__":
    test_telegram_alerts()
