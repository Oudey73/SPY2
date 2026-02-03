"""
SPY Opportunity Agent
Main monitoring agent for SPY mean reversion signals
Based on IBS + RSI(3) strategy with VIX filter
"""
import sys
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import schedule
import pytz

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Import our modules
from collectors.yahoo_collector import YahooCollector
from collectors.polygon_collector import PolygonCollector
from collectors.orats_collector import ORATSCollector
from collectors.enhanced_data_collector import EnhancedDataCollector
from signals.signal_detector import SPYSignalDetector
from signals.opportunity_scorer import SPYOpportunityScorer, Opportunity, EnhancedScoringResult
from signals.iv_signal_detector import IVSignalDetector
from signals.exit_monitor import ExitMonitor, ExitAlert, AlertType
from alerts.email_alert import EmailAlertSystem
from alerts.telegram_alert import TelegramAlertSystem

# New modules
from analysis.regime_classifier import RegimeClassifier, RegimeType
from analysis.liquidity_checker import LiquidityChecker
from strategy.strategy_selector import StrategySelector, StrategyType
from strategy.position_sizer import PositionSizer
from strategy.trade_builder import TradeBuilder
from risk.risk_manager import RiskManager
from risk.event_risk import EventRiskChecker
from risk.greeks_monitor import GreeksMonitor
from execution.trade_logger import TradeLogger
from execution.performance_tracker import PerformanceTracker
from execution.opportunity_logger import OpportunityLogger
import config

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
)
logger.add(
    "logs/spy_agent_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="30 days"
)

# Eastern timezone for market hours
ET = pytz.timezone('US/Eastern')


class SPYOpportunityAgent:
    """
    Main agent that monitors SPY for trading opportunities

    Strategy: IBS + RSI(3) Mean Reversion with VIX Filter
    - Historical win rate: ~71%
    - Typical hold: 1-5 days
    - Best entries: IBS < 0.2 + RSI(3) < 20 + VIX > 10-MA
    """

    def __init__(
        self,
        min_score: int = 50,
        scan_interval_minutes: int = 15,
    ):
        self.min_score = min_score
        self.scan_interval = scan_interval_minutes

        # Account value from env or config
        self.account_value = float(os.getenv("ACCOUNT_VALUE", config.ACCOUNT["value"]))

        # Initialize components
        logger.info("Initializing SPY Opportunity Agent...")
        self.yahoo = YahooCollector()
        self.polygon = PolygonCollector()
        self.orats = ORATSCollector()
        self.detector = SPYSignalDetector()
        self.iv_detector = IVSignalDetector()
        self.scorer = SPYOpportunityScorer(min_score_threshold=min_score)
        self.email = EmailAlertSystem()
        self.telegram = TelegramAlertSystem()

        # New pipeline components
        self.regime_classifier = RegimeClassifier(yahoo_collector=self.yahoo)
        self.liquidity_checker = LiquidityChecker()
        self.strategy_selector = StrategySelector()
        self.position_sizer = PositionSizer()
        self.trade_builder = TradeBuilder()
        self.risk_manager = RiskManager(account_value=self.account_value)
        self.event_checker = EventRiskChecker()
        self.greeks_monitor = GreeksMonitor()
        self.trade_logger = TradeLogger()
        self.performance_tracker = PerformanceTracker()
        self.opp_logger = OpportunityLogger(trade_logger=self.trade_logger)

        # Enhanced scoring components (share polygon instance for CVD)
        self.enhanced_collector = EnhancedDataCollector(polygon_collector=self.polygon)
        self.exit_monitor = ExitMonitor(
            bars_for_partial=config.EXIT_MONITOR.get("bars_for_partial", 3),
            bars_for_full_exit=config.EXIT_MONITOR.get("bars_for_full_exit", 5),
        )

        # State tracking
        self.running = False
        self.last_scan = None
        self.scan_count = 0
        self.alert_count = 0
        self.last_opportunity: Optional[Opportunity] = None

        # Alert cooldowns
        self.last_alert_time: Optional[datetime] = None
        self.cooldown_hours = 4  # Don't re-alert for same direction within 4 hours

        logger.info(f"Agent configured: min_score={min_score}, interval={scan_interval_minutes}min")

    def is_market_open(self) -> bool:
        """Check if US market is open"""
        now = datetime.now(ET)

        # Market closed on weekends
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def should_alert(self, opportunity: Opportunity) -> bool:
        """Check if we should send alert"""
        if opportunity.score < self.min_score:
            return False

        # Check cooldown
        if self.last_alert_time:
            elapsed = (datetime.now(timezone.utc) - self.last_alert_time).total_seconds() / 3600
            if elapsed < self.cooldown_hours:
                # Allow alert if direction changed
                if self.last_opportunity and self.last_opportunity.direction == opportunity.direction:
                    logger.info(f"Alert on cooldown ({elapsed:.1f}h < {self.cooldown_hours}h)")
                    return False

        return True

    def collect_market_data(self) -> Optional[Dict]:
        """Collect all market data for SPY"""
        try:
            # Get Yahoo data (primary - always available)
            technicals = self.yahoo.get_spy_technicals()

            if not technicals:
                logger.error("Failed to get Yahoo data")
                return None

            # Get Polygon data if market is open and API key available
            polygon_data = None
            if self.polygon.api_key:
                polygon_data = self.polygon.get_all_spy_data()

            # Get intraday momentum if market is open
            momentum = None
            if self.is_market_open():
                momentum = self.yahoo.get_intraday_momentum()

            # Get live intraday price during market hours (fixes stale daily close issue)
            live_price = technicals["price"]["close"]  # default to daily close
            if self.is_market_open():
                live_data = self.yahoo.get_current_price("SPY")
                if live_data and live_data.get("price"):
                    live_price = live_data["price"]

            # Build market data structure
            market_data = {
                "symbol": "SPY",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": live_price,
                "ibs": technicals["ibs"]["current"],
                "rsi_2": technicals["rsi"].get("rsi_2"),  # NEW: RSI(2) for PUT signals
                "rsi_3": technicals["rsi"]["rsi_3"],
                "rsi_14": technicals["rsi"]["rsi_14"],
                "vix": technicals["vix"]["vix"] if technicals.get("vix") else None,
                "vix_10_ma": technicals["vix"]["vix_10_ma"] if technicals.get("vix") else None,
                "sma_20": technicals["moving_averages"]["sma_20"],
                "sma_50": technicals["moving_averages"]["sma_50"],
                "sma_200": technicals["moving_averages"]["sma_200"],
                "change_percent": technicals["price"]["change_percent"],
                # NEW: Consecutive days for PUT signals
                "consecutive_up": technicals.get("consecutive_days", {}).get("up", 0),
                "consecutive_down": technicals.get("consecutive_days", {}).get("down", 0),
            }

            # Add intraday momentum if available
            if momentum:
                market_data["earlier_return_pct"] = momentum.get("earlier_return_pct")
                market_data["last_30min_return_pct"] = momentum.get("last_30min_return_pct")

            # Add Polygon real-time data if available
            if polygon_data and polygon_data.get("has_data"):
                snapshot = polygon_data.get("snapshot", {})
                if snapshot.get("current", {}).get("price"):
                    market_data["price_realtime"] = snapshot["current"]["price"]
                if polygon_data.get("calculated", {}).get("ibs_today"):
                    market_data["ibs_intraday"] = polygon_data["calculated"]["ibs_today"]

            # Add ORATS IV data if available
            if self.orats.is_configured():
                orats_data = self.orats.get_all_iv_data("SPY")
                if orats_data:
                    iv_rank_data = orats_data.get("iv_rank", {})
                    term_structure = orats_data.get("term_structure", {})
                    skew_data = orats_data.get("skew", {})

                    market_data["iv_rank"] = iv_rank_data.get("iv_rank")
                    market_data["iv_percentile"] = iv_rank_data.get("iv_percentile")
                    market_data["current_iv"] = iv_rank_data.get("current_iv")
                    market_data["term_structure"] = term_structure.get("structure")
                    market_data["term_spread"] = term_structure.get("spread")
                    market_data["skew_type"] = skew_data.get("skew_type")
                    market_data["put_skew"] = skew_data.get("put_skew")
                    market_data["iv_strategy_recommendation"] = orats_data.get("strategy_recommendation")

            # Add enhanced data for multi-factor scoring
            enhanced_data = self.enhanced_collector.get_all_enhanced_data()
            market_data["enhanced"] = enhanced_data

            return market_data

        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return None

    def run_scan(self):
        """Run a complete scan"""
        self.scan_count += 1
        self.last_scan = datetime.now(timezone.utc)

        # Check market hours
        market_status = "OPEN" if self.is_market_open() else "CLOSED"

        logger.info(f"{'='*50}")
        logger.info(f"SCAN #{self.scan_count} - {self.last_scan.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Market Status: {market_status}")
        logger.info(f"{'='*50}")

        # Collect data
        market_data = self.collect_market_data()

        if not market_data:
            logger.error("Failed to collect market data")
            return

        # Log current values
        logger.info(f"SPY: ${market_data['price']:.2f} ({market_data['change_percent']:+.2f}%)")
        if market_data.get('ibs'):
            logger.info(f"IBS: {market_data['ibs']:.3f}")
        if market_data.get('rsi_2'):
            logger.info(f"RSI(2): {market_data['rsi_2']:.1f}")
        if market_data.get('rsi_3'):
            logger.info(f"RSI(3): {market_data['rsi_3']:.1f}")
        if market_data.get('vix'):
            logger.info(f"VIX: {market_data['vix']:.2f}")
        if market_data.get('consecutive_up', 0) >= 2:
            logger.info(f"Consecutive Up Days: {market_data['consecutive_up']}")
        if market_data.get('consecutive_down', 0) >= 2:
            logger.info(f"Consecutive Down Days: {market_data['consecutive_down']}")

        # Log ORATS IV data if available
        if market_data.get('iv_rank') is not None:
            logger.info(f"--- ORATS IV Analytics ---")
            logger.info(f"IV Rank: {market_data['iv_rank']:.1f} | IV: {market_data.get('current_iv', 0):.1f}%")
            if market_data.get('term_structure'):
                logger.info(f"Term Structure: {market_data['term_structure']} (spread: {market_data.get('term_spread', 0):.2f})")
            if market_data.get('skew_type'):
                logger.info(f"Skew: {market_data['skew_type']} (put skew: {market_data.get('put_skew', 0):.2f})")
            if market_data.get('iv_strategy_recommendation'):
                logger.info(f"IV Strategy: {market_data['iv_strategy_recommendation']}")

        # Detect signals
        signals = self.detector.analyze_all(market_data)

        # Detect IV signals from ORATS
        iv_signals = []
        if self.iv_detector.is_configured():
            iv_signals = self.iv_detector.detect_iv_signals("SPY")
            if iv_signals:
                logger.info(f"Detected {len(iv_signals)} IV signal(s):")
                for iv_s in iv_signals:
                    logger.info(f"  [IV] {iv_s.signal_type.value}: {iv_s.description} ({iv_s.strategy_bias})")

        if not signals:
            logger.info("No signals detected")
            return

        logger.info(f"Detected {len(signals)} signal(s):")
        for s in signals:
            logger.info(f"  [{s.tier}] {s.signal_type.value}: {s.direction.value.upper()} ({s.strength})")

        # Score opportunity (basic scoring)
        opportunity = self.scorer.score_opportunity(signals, market_data["price"])

        if not opportunity:
            logger.info("No clear opportunity")
            # Still check exit conditions for tracked positions
            self._check_exit_conditions(market_data["price"])
            return

        # Apply enhanced scoring if enabled
        enhanced_result = None
        if config.ENHANCED_SCORING.get("enabled", True):
            enhanced_data = market_data.get("enhanced", {})
            enhanced_result = self.scorer.calculate_enhanced_score(
                signals=signals,
                enhanced_data=enhanced_data,
                market_data=market_data
            )
            if enhanced_result:
                # Update opportunity with enhanced scoring
                opportunity.enhanced_result = enhanced_result
                opportunity.score = enhanced_result.final_score
                opportunity.grade = self.scorer.get_grade(enhanced_result.final_score)
                # Update position size suggestion with Kelly-based size
                kelly_pct = enhanced_result.dynamic_position_size * 100
                opportunity.position_size_suggestion = f"{kelly_pct:.1f}% of portfolio (Kelly-based, grade {enhanced_result.confidence_grade})"

                # Log enhanced scoring breakdown
                logger.info(f"Enhanced Scoring: {enhanced_result.base_score} -> {enhanced_result.final_score} ({enhanced_result.confidence_grade})")
                for line in enhanced_result.logic_breakdown:
                    logger.info(f"  {line}")

        logger.info(f"Opportunity: {opportunity.direction.value.upper()} Score={opportunity.score} Grade={opportunity.grade.value} ID={opportunity.opp_id}")

        # Check exit conditions for tracked positions
        self._check_exit_conditions(market_data["price"])

        # === Extended Pipeline: Regime -> Strategy -> Risk -> Log ===
        trade_plan = None
        regime = None
        strategy_rec = None
        risk_decision = None

        try:
            # 1. Classify regime
            regime = self.regime_classifier.classify(market_data)
            logger.info(f"Regime: {regime.regime.value} (confidence {regime.confidence:.0f}%, bias={regime.bias})")

            # 2. Check event risk
            event_risk = self.event_checker.check()
            if event_risk.events:
                logger.info(f"Event risk: {', '.join(event_risk.events)} [{event_risk.risk_level}]")
            if event_risk.blocked:
                logger.warning(f"Trade blocked by event risk: {event_risk.recommendation}")

            # 3. Build IV data dict for selectors
            iv_dict = None
            if market_data.get("iv_rank") is not None:
                iv_dict = {
                    "iv_rank": market_data.get("iv_rank"),
                    "current_iv": market_data.get("current_iv"),
                    "term_structure": market_data.get("term_structure"),
                }

            # 4. Select strategy
            direction_bias = "bullish" if opportunity.direction.value == "long" else "bearish"
            strategy_rec = self.strategy_selector.select(regime, iv_dict, direction_bias)
            logger.info(f"Strategy: {strategy_rec.strategy.value} ({strategy_rec.reasoning[:80]})")

            if strategy_rec.strategy != StrategyType.NO_TRADE:
                # 5. Calculate position size
                spread_width = 3.0  # default $3 wide spread
                max_loss_per_contract = spread_width * 100
                position_size = self.position_sizer.calculate(
                    account_value=self.account_value,
                    max_loss_per_contract=max_loss_per_contract,
                    confidence=regime.confidence,
                    regime_type=regime.regime,
                )
                logger.info(f"Position: {position_size.contracts} contracts (risk ${position_size.risk_per_trade:,.0f})")

                # 6. Build trade plan
                trade_plan = self.trade_builder.build(
                    strategy=strategy_rec.strategy,
                    price=market_data["price"],
                    regime=regime,
                    iv_data=iv_dict,
                    position_size=position_size,
                )

                if trade_plan:
                    # 7. Run risk manager
                    open_positions = self.trade_logger.get_open_positions()
                    risk_decision = self.risk_manager.evaluate_trade(
                        trade_plan=trade_plan,
                        regime=regime,
                        iv_data=iv_dict,
                        signals=signals,
                        positions=open_positions,
                        account_value=self.account_value,
                    )
                    logger.info(f"Risk decision: {'APPROVED' if risk_decision.approved else 'REJECTED'} (risk score {risk_decision.risk_score:.0f})")
                    if risk_decision.adjustments:
                        for adj in risk_decision.adjustments:
                            logger.info(f"  Adjustment: {adj}")

                    # 8. Log trade if approved
                    if risk_decision.approved:
                        trade_id = self.trade_logger.log_entry(
                            trade_plan=trade_plan.to_dict(),
                            regime=regime.to_dict(),
                            iv_data=iv_dict,
                            signals=[s.to_dict() for s in signals],
                            risk_decision=risk_decision.to_dict(),
                        )
                        logger.info(f"Trade logged: {trade_id}")

        except Exception as e:
            logger.error(f"Extended pipeline error (non-fatal): {e}")
            # Graceful fallback: existing pipeline still works

        # Log every opportunity regardless of approval
        self.opp_logger.log(
            opportunity_dict=opportunity.to_dict(),
            regime_dict=regime.to_dict() if regime else None,
            trade_plan_dict=trade_plan.to_dict() if trade_plan else None,
            risk_decision_dict=risk_decision.to_dict() if risk_decision else None,
        )

        # Send alert if warranted
        if self.should_alert(opportunity):
            if trade_plan and regime and risk_decision and risk_decision.approved:
                self._send_trade_recommendation(opportunity, trade_plan, regime, risk_decision)
            else:
                self.send_alert(opportunity)
        else:
            logger.info(f"Score {opportunity.score} below threshold {self.min_score} or on cooldown")

        self.last_opportunity = opportunity

    def _send_trade_recommendation(self, opportunity, trade_plan, regime, risk_decision):
        """Send enriched trade recommendation with regime + strategy + risk info."""
        logger.info(f"SENDING TRADE RECOMMENDATION: {trade_plan.strategy.value}")

        # Print to console
        print(opportunity.format_alert())

        # Send enriched Telegram alert
        if self.telegram.is_configured():
            success = self.telegram.send_trade_recommendation(opportunity, trade_plan, regime)
            if success:
                logger.info("Telegram trade recommendation sent")
                self.alert_count += 1
                self.last_alert_time = datetime.now(timezone.utc)
            else:
                logger.error("Failed to send Telegram trade recommendation")
        else:
            logger.warning("Telegram not configured")

        # Email fallback uses standard alert
        if self.email.is_configured():
            self.email.send_opportunity_alert(opportunity)

    def send_alert(self, opportunity: Opportunity):
        """Send alert for opportunity"""
        logger.info(f"🚨 SENDING ALERT: SPY {opportunity.direction.value.upper()} (Score: {opportunity.score})")

        # Print to console
        print(opportunity.format_alert())

        # Send Telegram alert (primary)
        if self.telegram.is_configured():
            success = self.telegram.send_opportunity_alert(opportunity)
            if success:
                logger.info("✅ Telegram alert sent successfully")
                self.alert_count += 1
                self.last_alert_time = datetime.now(timezone.utc)
            else:
                logger.error("❌ Failed to send Telegram alert")
        else:
            logger.warning("📱 Telegram not configured")

        # Send email alert (secondary)
        if self.email.is_configured():
            success = self.email.send_opportunity_alert(opportunity)
            if success:
                logger.info("✅ Email alert sent successfully")
            else:
                logger.error("❌ Failed to send email alert")

    def register_position(self, opportunity: Opportunity):
        """
        Register an opportunity as a position for exit monitoring.

        Call this after a trade is entered to start tracking for exit alerts.

        Args:
            opportunity: The opportunity that was traded
        """
        self.exit_monitor.register_entry(
            opp_id=opportunity.opp_id,
            entry_price=opportunity.entry_zone.get("target", opportunity.entry_zone.get("aggressive")),
            direction=opportunity.direction.value,
            stop_loss=opportunity.stop_loss,
            targets=opportunity.targets,
        )
        logger.info(f"Registered position for exit monitoring: {opportunity.opp_id}")

    def _check_exit_conditions(self, current_price: float):
        """
        Check exit conditions for all tracked positions.

        Args:
            current_price: Current market price
        """
        if self.exit_monitor.get_position_count() == 0:
            return

        alerts = self.exit_monitor.check_all_positions(current_price)

        for alert in alerts:
            logger.info(f"Exit Alert: {alert.opp_id} - {alert.alert_type.value} ({alert.exit_percentage}%)")
            logger.info(f"  Reason: {alert.reason}")
            logger.info(f"  P&L: {alert.pnl_percent:+.2f}% after {alert.bars_held} bars")
            self._send_exit_alert(alert)

    def _send_exit_alert(self, alert: ExitAlert):
        """
        Send an exit alert via configured channels.

        Args:
            alert: ExitAlert object with exit details
        """
        logger.info(f"🚨 SENDING EXIT ALERT: {alert.opp_id} ({alert.alert_type.value})")

        # Send Telegram exit alert
        if self.telegram.is_configured():
            success = self.telegram.send_exit_alert(alert)
            if success:
                logger.info("✅ Telegram exit alert sent")
            else:
                logger.error("❌ Failed to send Telegram exit alert")

    def start(self):
        """Start the monitoring agent"""
        logger.info("🚀 Starting SPY Opportunity Agent...")
        logger.info(f"Strategy: IBS + RSI(3) Mean Reversion with VIX Filter")
        logger.info(f"Historical Win Rate: ~71%")

        self.running = True

        # Check configurations
        if self.telegram.is_configured():
            logger.info("✅ Telegram alerts configured")
        else:
            logger.warning("⚠️  Telegram not configured")

        if self.email.is_configured():
            logger.info("✅ Email alerts configured")
        else:
            logger.warning("⚠️  Email not configured")

        if self.polygon.api_key:
            logger.info("✅ Polygon.io configured (real-time data)")
        else:
            logger.warning("⚠️  Polygon.io not configured (using Yahoo only)")

        if self.orats.is_configured():
            logger.info("✅ ORATS configured (professional IV analytics)")
        else:
            logger.warning("⚠️  ORATS not configured (set ORATS_API_KEY in .env)")

        # Run initial scan
        self.run_scan()

        # Schedule scans during market hours
        # Main scan at 9:35 AM ET (5 min after open)
        schedule.every().day.at("09:35").do(self.run_scan)

        # Regular scans every N minutes during market hours
        schedule.every(self.scan_interval).minutes.do(self._market_hours_scan)

        # End of day scan at 3:55 PM ET
        schedule.every().day.at("15:55").do(self.run_scan)

        logger.info(f"\n⏰ Scheduled scans every {self.scan_interval} minutes during market hours")
        logger.info("Press Ctrl+C to stop\n")

        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _market_hours_scan(self):
        """Only scan during market hours"""
        if self.is_market_open():
            self.run_scan()
        else:
            logger.debug("Market closed, skipping scan")

    def stop(self):
        """Stop the agent"""
        self.running = False
        schedule.clear()

        logger.info("\n" + "=" * 50)
        logger.info("AGENT STOPPED")
        logger.info("=" * 50)
        logger.info(f"Total scans: {self.scan_count}")
        logger.info(f"Alerts sent: {self.alert_count}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="SPY Opportunity Agent")
    parser.add_argument(
        "--min-score",
        type=int,
        default=50,
        help="Minimum score to trigger alert (default: 50)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Scan interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run single scan and exit"
    )

    args = parser.parse_args()

    agent = SPYOpportunityAgent(
        min_score=args.min_score,
        scan_interval_minutes=args.interval
    )

    if args.test:
        logger.info("Running in TEST mode (single scan)")
        agent.run_scan()
    else:
        agent.start()


if __name__ == "__main__":
    main()
