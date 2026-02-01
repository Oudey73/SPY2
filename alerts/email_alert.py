"""
Email Alert System
Sends trading opportunity alerts via email
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class EmailAlertSystem:
    """Sends email alerts for trading opportunities"""

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        sender_email: str = None,
        sender_password: str = None,
        recipient_email: str = None
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", 587))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL", "")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD", "")
        self.recipient_email = recipient_email or os.getenv("RECIPIENT_EMAIL", "")

        # Track sent alerts to avoid spam
        self.sent_alerts: Dict[str, datetime] = {}
        self.cooldown_minutes = 30

    def is_configured(self) -> bool:
        """Check if email is properly configured"""
        return all([
            self.smtp_server,
            self.smtp_port,
            self.sender_email,
            self.sender_password,
            self.recipient_email
        ])

    def should_send(self, alert_key: str) -> bool:
        """Check if we should send this alert (cooldown check)"""
        if alert_key in self.sent_alerts:
            last_sent = self.sent_alerts[alert_key]
            elapsed = (datetime.utcnow() - last_sent).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                logger.info(f"Alert {alert_key} on cooldown ({elapsed:.1f}m < {self.cooldown_minutes}m)")
                return False
        return True

    def send_email(
        self,
        subject: str,
        body_text: str,
        body_html: str = None
    ) -> bool:
        """
        Send an email

        Args:
            subject: Email subject
            body_text: Plain text body
            body_html: Optional HTML body

        Returns:
            True if sent successfully
        """
        if not self.is_configured():
            logger.error("Email not configured. Set environment variables.")
            return False

        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = self.recipient_email

            # Add text part
            part1 = MIMEText(body_text, "plain")
            message.attach(part1)

            # Add HTML part if provided
            if body_html:
                part2 = MIMEText(body_html, "html")
                message.attach(part2)

            # Create secure connection and send
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_email,
                    message.as_string()
                )

            logger.info(f"Email sent successfully: {subject}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check email/password.")
            logger.error("For Gmail, use an App Password: https://myaccount.google.com/apppasswords")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Email error: {e}")
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

        if not self.should_send(alert_key):
            return False

        # Create subject
        direction_emoji = "🟢" if opportunity.direction.value == "long" else "🔴"
        grade_emoji = {"A+": "🔥", "A": "⭐", "B": "👀", "C": "⚠️", "F": "🚫"}

        subject = f"{direction_emoji} {opportunity.symbol} {opportunity.direction.value.upper()} | Score: {opportunity.score} ({opportunity.grade.value}) {grade_emoji.get(opportunity.grade.value, '')}"

        # Create plain text body
        body_text = opportunity.format_alert()

        # Create HTML body for better formatting
        body_html = self._create_html_alert(opportunity)

        # Send
        success = self.send_email(subject, body_text, body_html)

        if success:
            self.sent_alerts[alert_key] = datetime.utcnow()

        return success

    def _create_html_alert(self, opportunity) -> str:
        """Create HTML formatted alert"""
        direction_color = "#22c55e" if opportunity.direction.value == "long" else "#ef4444"
        direction_bg = "#dcfce7" if opportunity.direction.value == "long" else "#fee2e2"

        drivers_html = ""
        for i, driver in enumerate(opportunity.top_drivers[:3], 1):
            drivers_html += f"<li>{driver}</li>"

        warnings_html = ""
        if opportunity.warnings:
            warnings_html = "<h3 style='color: #f59e0b;'>⚠️ Warnings</h3><ul>"
            for warning in opportunity.warnings:
                warnings_html += f"<li>{warning}</li>"
            warnings_html += "</ul>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: {direction_color}; color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9fafb; padding: 20px; border: 1px solid #e5e7eb; }}
        .score-box {{ background: {direction_bg}; padding: 15px; border-radius: 8px; text-align: center; margin: 15px 0; }}
        .score {{ font-size: 48px; font-weight: bold; color: {direction_color}; }}
        .grade {{ font-size: 24px; color: #666; }}
        .section {{ background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e5e7eb; }}
        .section h3 {{ margin-top: 0; color: #374151; }}
        .level {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #f3f4f6; }}
        .disclaimer {{ background: #fef3c7; border: 1px solid #fcd34d; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        .footer {{ text-align: center; color: #9ca3af; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0;">{opportunity.symbol}</h1>
        <h2 style="margin: 10px 0 0 0;">{opportunity.direction.value.upper()} OPPORTUNITY</h2>
    </div>

    <div class="content">
        <div class="score-box">
            <div class="score">{opportunity.score}</div>
            <div class="grade">Grade: {opportunity.grade.value} | {opportunity.confidence_level.split(' - ')[0]}</div>
        </div>

        <div class="section">
            <h3>📈 Top Drivers</h3>
            <ol>
                {drivers_html}
            </ol>
        </div>

        <div class="section">
            <h3>💰 Entry Zone</h3>
            <div class="level"><span>Aggressive:</span> <strong>${opportunity.entry_zone.get('high', opportunity.entry_zone.get('aggressive', 0)):,.2f}</strong></div>
            <div class="level"><span>Mid:</span> <strong>${opportunity.entry_zone.get('mid', opportunity.entry_zone.get('target', 0)):,.2f}</strong></div>
            <div class="level"><span>Conservative:</span> <strong>${opportunity.entry_zone.get('low', opportunity.entry_zone.get('conservative', 0)):,.2f}</strong></div>
        </div>

        <div class="section">
            <h3>🛑 Stop Loss (Invalidation)</h3>
            <div style="font-size: 24px; text-align: center; color: #ef4444; font-weight: bold;">
                ${getattr(opportunity, 'invalidation', None) or getattr(opportunity, 'stop_loss', 0):,.2f}
            </div>
        </div>

        <div class="section">
            <h3>🎯 Targets</h3>
            <div class="level"><span>T1:</span> <strong>${opportunity.targets['t1']:,.2f}</strong></div>
            <div class="level"><span>T2:</span> <strong>${opportunity.targets['t2']:,.2f}</strong></div>
            <div class="level"><span>T3:</span> <strong>${opportunity.targets['t3']:,.2f}</strong></div>
        </div>

        <div class="section">
            <h3>📐 Risk/Reward & Sizing</h3>
            <div class="level"><span>Risk/Reward:</span> <strong>{opportunity.risk_reward:.1f}R</strong></div>
            <div class="level"><span>Suggested Size:</span> <strong>{opportunity.position_size_suggestion}</strong></div>
        </div>

        {warnings_html}

        <div class="disclaimer">
            <strong>⚠️ DISCLAIMER</strong><br>
            This is NOT financial advice. This is a decision-support tool only.
            Always do your own research and verify before taking any trades.
            Past signals do not guarantee future performance.
        </div>

        <div class="footer">
            <p>Generated: {opportunity.timestamp}</p>
            <p>Crypto Opportunity Agent v1.0</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_heartbeat(self, status: Dict) -> bool:
        """
        Send periodic heartbeat/status email

        Args:
            status: Dict with current market status
        """
        subject = "💓 Crypto Agent Heartbeat - System Active"

        body_text = f"""
Crypto Opportunity Agent - Status Report
========================================

Time: {datetime.utcnow().isoformat()}

MARKET OVERVIEW
---------------
"""
        for symbol, data in status.get("markets", {}).items():
            body_text += f"""
{symbol}:
  Price: ${data.get('price', 'N/A'):,.2f}
  Funding: {data.get('funding', 'N/A'):.4f}%
  24h Change: {data.get('change_24h', 'N/A'):.2f}%
"""

        body_text += f"""

SIGNALS DETECTED: {status.get('total_signals', 0)}
OPPORTUNITIES: {status.get('opportunities', 0)}

System Status: {'✅ Healthy' if status.get('healthy', True) else '❌ Issues Detected'}

---
No action required. This is an automated status update.
"""

        return self.send_email(subject, body_text)

    def send_test_email(self) -> bool:
        """Send a test email to verify configuration"""
        subject = "🧪 Crypto Agent - Test Email"
        body_text = f"""
This is a test email from your Crypto Opportunity Agent.

If you received this, your email configuration is working correctly!

Configuration:
- SMTP Server: {self.smtp_server}
- SMTP Port: {self.smtp_port}
- Sender: {self.sender_email}
- Recipient: {self.recipient_email}

Time: {datetime.utcnow().isoformat()}

You will receive alerts when high-conviction trading opportunities are detected.
"""
        return self.send_email(subject, body_text)


def test_email_system():
    """Test the email alert system"""
    print("\n" + "=" * 60)
    print("EMAIL ALERT SYSTEM TEST")
    print("=" * 60)

    email_system = EmailAlertSystem()

    # Check configuration
    print(f"\nConfiguration status: {'✅ Configured' if email_system.is_configured() else '❌ Not configured'}")

    if not email_system.is_configured():
        print("""
To configure email alerts:
1. Copy .env.example to .env
2. Fill in your email settings:
   - SMTP_SERVER (default: smtp.gmail.com)
   - SMTP_PORT (default: 587)
   - SENDER_EMAIL (your email)
   - SENDER_PASSWORD (app password for Gmail)
   - RECIPIENT_EMAIL (where to send alerts)

For Gmail:
- Enable 2-Step Verification
- Generate App Password at: https://myaccount.google.com/apppasswords
""")
        return False

    # Send test email
    print("\nSending test email...")
    success = email_system.send_test_email()

    if success:
        print("✅ Test email sent successfully!")
    else:
        print("❌ Failed to send test email. Check your configuration.")

    return success


if __name__ == "__main__":
    test_email_system()
