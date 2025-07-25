"""
Comprehensive Notification and Monitoring System
"""
import asyncio
import aiohttp
import smtplib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import requests
from collections import deque, defaultdict
import threading
import time

class NotificationLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"

@dataclass
class NotificationMessage:
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime
    channel: NotificationChannel
    metadata: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class PerformanceAlert:
    alert_type: str
    threshold: float
    current_value: float
    severity: NotificationLevel
    description: str
    timestamp: datetime

class NotificationManager:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Notification channels configuration
        self.telegram_config = config.get('telegram', {})
        self.discord_config = config.get('discord', {})
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhook', {})
        
        # Message queues
        self.message_queue = asyncio.Queue()
        self.failed_messages = deque(maxlen=100)
        self.sent_messages = deque(maxlen=1000)
        
        # Rate limiting
        self.rate_limits = {
            NotificationChannel.TELEGRAM: {'count': 0, 'reset_time': datetime.utcnow()},
            NotificationChannel.DISCORD: {'count': 0, 'reset_time': datetime.utcnow()},
            NotificationChannel.EMAIL: {'count': 0, 'reset_time': datetime.utcnow()}
        }
        self.max_messages_per_hour = {
            NotificationChannel.TELEGRAM: 30,
            NotificationChannel.DISCORD: 50,
            NotificationChannel.EMAIL: 10
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'daily_loss_pct': -0.05,      # -5% daily loss
            'drawdown_pct': -0.10,        # -10% drawdown
            'consecutive_losses': 5,       # 5 consecutive losses
            'position_loss_pct': -0.08,   # -8% single position loss
            'low_balance_pct': 0.20,      # 20% of initial capital
            'high_volatility': 0.08,      # 8% volatility
            'api_errors_per_hour': 10,    # 10 API errors per hour
            'trade_frequency_low': 0.5,   # Less than 0.5 trades per hour
            'trade_frequency_high': 20    # More than 20 trades per hour
        }
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1440)  # 24 hours of minute data
        self.error_counts = defaultdict(int)
        self.last_heartbeat = datetime.utcnow()
        
        # Background tasks
        self.notification_task = None
        self.monitoring_task = None
        self.running = False
        
    async def start(self):
        """Start the notification manager"""
        try:
            self.running = True
            
            # Start background tasks
            self.notification_task = asyncio.create_task(self._process_notifications())
            self.monitoring_task = asyncio.create_task(self._monitor_performance())
            
            self.logger.info("Notification manager started")
            
            # Send startup notification
            await self.send_notification(
                level=NotificationLevel.INFO,
                title="🚀 Trading Bot Started",
                message="Crypto trading bot has started successfully and is ready for trading.",
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.DISCORD]
            )
            
        except Exception as e:
            self.logger.error(f"Error starting notification manager: {e}")
    
    async def stop(self):
        """Stop the notification manager"""
        try:
            self.running = False
            
            # Send shutdown notification
            await self.send_notification(
                level=NotificationLevel.WARNING,
                title="🛑 Trading Bot Stopped",
                message="Crypto trading bot has been stopped.",
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.DISCORD]
            )
            
            # Cancel background tasks
            if self.notification_task:
                self.notification_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Process remaining messages
            await self._process_remaining_messages()
            
            self.logger.info("Notification manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping notification manager: {e}")
    
    async def send_notification(self, level: NotificationLevel, title: str, 
                              message: str, channels: List[NotificationChannel] = None,
                              metadata: Dict = None):
        """Send notification to specified channels"""
        try:
            if channels is None:
                channels = [NotificationChannel.TELEGRAM, NotificationChannel.CONSOLE]
            
            if metadata is None:
                metadata = {}
            
            # Create notification message
            notification = NotificationMessage(
                level=level,
                title=title,
                message=message,
                timestamp=datetime.utcnow(),
                channel=NotificationChannel.TELEGRAM,  # Will be updated per channel
                metadata=metadata
            )
            
            # Queue messages for each channel
            for channel in channels:
                channel_notification = NotificationMessage(
                    level=level,
                    title=title,
                    message=message,
                    timestamp=datetime.utcnow(),
                    channel=channel,
                    metadata=metadata
                )
                
                await self.message_queue.put(channel_notification)
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
    
    async def _process_notifications(self):
        """Background task to process notification queue"""
        while self.running:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check rate limits
                if not self._check_rate_limit(message.channel):
                    self.logger.warning(f"Rate limit exceeded for {message.channel.value}")
                    continue
                
                # Send message
                success = await self._send_message(message)
                
                if success:
                    self.sent_messages.append(message)
                    self._update_rate_limit(message.channel)
                else:
                    # Retry logic
                    if message.retry_count < message.max_retries:
                        message.retry_count += 1
                        await asyncio.sleep(2 ** message.retry_count)  # Exponential backoff
                        await self.message_queue.put(message)
                    else:
                        self.failed_messages.append(message)
                        self.logger.error(f"Failed to send message after {message.max_retries} retries")
                
            except Exception as e:
                self.logger.error(f"Error processing notifications: {e}")
                await asyncio.sleep(1)
    
    async def _send_message(self, message: NotificationMessage) -> bool:
        """Send message to specific channel"""
        try:
            if message.channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram_message(message)
            elif message.channel == NotificationChannel.DISCORD:
                return await self._send_discord_message(message)
            elif message.channel == NotificationChannel.EMAIL:
                return await self._send_email_message(message)
            elif message.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_message(message)
            elif message.channel == NotificationChannel.CONSOLE:
                return self._send_console_message(message)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending message to {message.channel.value}: {e}")
            return False
    
    async def _send_telegram_message(self, message: NotificationMessage) -> bool:
        """Send message to Telegram"""
        try:
            if not self.telegram_config.get('bot_token') or not self.telegram_config.get('chat_id'):
                return False
            
            # Format message
            emoji = self._get_emoji_for_level(message.level)
            formatted_message = f"{emoji} *{message.title}*\n\n{message.message}"
            
            # Add metadata if present
            if message.metadata:
                formatted_message += "\n\n📊 *Details:*"
                for key, value in message.metadata.items():
                    formatted_message += f"\n• {key}: {value}"
            
            # Add timestamp
            formatted_message += f"\n\n🕐 {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            # Send via Telegram API
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            payload = {
                'chat_id': self.telegram_config['chat_id'],
                'text': formatted_message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.error(f"Telegram API error: {response.status}")
                        return False
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def _send_discord_message(self, message: NotificationMessage) -> bool:
        """Send message to Discord"""
        try:
            if not self.discord_config.get('webhook_url'):
                return False
            
            # Format message for Discord
            color = self._get_color_for_level(message.level)
            embed = {
                "title": message.title,
                "description": message.message,
                "color": color,
                "timestamp": message.timestamp.isoformat(),
                "footer": {
                    "text": "Crypto Trading Bot"
                }
            }
            
            # Add metadata fields
            if message.metadata:
                embed["fields"] = []
                for key, value in message.metadata.items():
                    embed["fields"].append({
                        "name": key,
                        "value": str(value),
                        "inline": True
                    })
            
            payload = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_config['webhook_url'], json=payload) as response:
                    if response.status == 204:
                        return True
                    else:
                        self.logger.error(f"Discord webhook error: {response.status}")
                        return False
            
        except Exception as e:
            self.logger.error(f"Error sending Discord message: {e}")
            return False
    
    async def _send_email_message(self, message: NotificationMessage) -> bool:
        """Send email message"""
        try:
            if not all([self.email_config.get('smtp_server'), 
                       self.email_config.get('smtp_port'),
                       self.email_config.get('username'),
                       self.email_config.get('password'),
                       self.email_config.get('to_email')]):
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"[{message.level.value.upper()}] {message.title}"
            
            # Email body
            body = f"{message.message}\n\n"
            
            if message.metadata:
                body += "Details:\n"
                for key, value in message.metadata.items():
                    body += f"• {key}: {value}\n"
            
            body += f"\nTimestamp: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            def send_email():
                try:
                    server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                    server.starttls()
                    server.login(self.email_config['username'], self.email_config['password'])
                    server.send_message(msg)
                    server.quit()
                    return True
                except Exception as e:
                    self.logger.error(f"SMTP error: {e}")
                    return False
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, send_email)
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_webhook_message(self, message: NotificationMessage) -> bool:
        """Send message to webhook"""
        try:
            if not self.webhook_config.get('url'):
                return False
            
            payload = {
                'level': message.level.value,
                'title': message.title,
                'message': message.message,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            }
            
            headers = self.webhook_config.get('headers', {})
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_config['url'], 
                                      json=payload, headers=headers) as response:
                    return response.status < 400
            
        except Exception as e:
            self.logger.error(f"Error sending webhook message: {e}")
            return False
    
    def _send_console_message(self, message: NotificationMessage) -> bool:
        """Send message to console"""
        try:
            level_colors = {
                NotificationLevel.INFO: '\033[92m',      # Green
                NotificationLevel.WARNING: '\033[93m',   # Yellow
                NotificationLevel.ERROR: '\033[91m',     # Red
                NotificationLevel.CRITICAL: '\033[95m'   # Magenta
            }
            
            reset_color = '\033[0m'
            color = level_colors.get(message.level, '')
            
            print(f"{color}[{message.level.value.upper()}] {message.title}{reset_color}")
            print(f"{message.message}")
            
            if message.metadata:
                print("Details:")
                for key, value in message.metadata.items():
                    print(f"  • {key}: {value}")
            
            print(f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print("-" * 50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending console message: {e}")
            return False
    
    def _get_emoji_for_level(self, level: NotificationLevel) -> str:
        """Get emoji for notification level"""
        emojis = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        return emojis.get(level, "📢")
    
    def _get_color_for_level(self, level: NotificationLevel) -> int:
        """Get Discord color for notification level"""
        colors = {
            NotificationLevel.INFO: 0x00ff00,      # Green
            NotificationLevel.WARNING: 0xffff00,   # Yellow
            NotificationLevel.ERROR: 0xff0000,     # Red
            NotificationLevel.CRITICAL: 0xff00ff   # Magenta
        }
        return colors.get(level, 0x0099ff)  # Blue default
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        try:
            if channel not in self.rate_limits:
                return True
            
            rate_limit = self.rate_limits[channel]
            now = datetime.utcnow()
            
            # Reset counter if hour has passed
            if now - rate_limit['reset_time'] >= timedelta(hours=1):
                rate_limit['count'] = 0
                rate_limit['reset_time'] = now
            
            # Check limit
            max_messages = self.max_messages_per_hour.get(channel, 100)
            return rate_limit['count'] < max_messages
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return True
    
    def _update_rate_limit(self, channel: NotificationChannel):
        """Update rate limit counter"""
        try:
            if channel in self.rate_limits:
                self.rate_limits[channel]['count'] += 1
        except Exception as e:
            self.logger.error(f"Error updating rate limit: {e}")
    
    async def _monitor_performance(self):
        """Background task to monitor performance and send alerts"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                # Check for performance alerts
                alerts = await self._check_performance_alerts()
                
                for alert in alerts:
                    await self.send_notification(
                        level=alert.severity,
                        title=f"⚠️ Performance Alert: {alert.alert_type}",
                        message=alert.description,
                        metadata={
                            'threshold': alert.threshold,
                            'current_value': alert.current_value,
                            'alert_type': alert.alert_type
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    async def _check_performance_alerts(self) -> List[PerformanceAlert]:
        """Check for performance-based alerts"""
        alerts = []
        
        try:
            # This would integrate with the risk manager and trading engine
            # For now, we'll create placeholder alerts
            
            # Example: Check if we have recent performance data
            if len(self.performance_history) > 0:
                latest_performance = self.performance_history[-1]
                
                # Check daily loss threshold
                daily_pnl_pct = latest_performance.get('daily_pnl_pct', 0)
                if daily_pnl_pct <= self.alert_thresholds['daily_loss_pct']:
                    alerts.append(PerformanceAlert(
                        alert_type="Daily Loss Limit",
                        threshold=self.alert_thresholds['daily_loss_pct'],
                        current_value=daily_pnl_pct,
                        severity=NotificationLevel.CRITICAL,
                        description=f"Daily loss has reached {daily_pnl_pct:.2%}, exceeding threshold of {self.alert_thresholds['daily_loss_pct']:.2%}",
                        timestamp=datetime.utcnow()
                    ))
                
                # Check drawdown threshold
                drawdown_pct = latest_performance.get('drawdown_pct', 0)
                if drawdown_pct <= self.alert_thresholds['drawdown_pct']:
                    alerts.append(PerformanceAlert(
                        alert_type="Maximum Drawdown",
                        threshold=self.alert_thresholds['drawdown_pct'],
                        current_value=drawdown_pct,
                        severity=NotificationLevel.ERROR,
                        description=f"Portfolio drawdown has reached {drawdown_pct:.2%}, exceeding threshold of {self.alert_thresholds['drawdown_pct']:.2%}",
                        timestamp=datetime.utcnow()
                    ))
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
        
        return alerts
    
    async def _process_remaining_messages(self):
        """Process remaining messages in queue before shutdown"""
        try:
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            
            while not self.message_queue.empty() and (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self._send_message(message)
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing remaining message: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing remaining messages: {e}")
    
    # Trading event notification methods
    async def notify_trade_opened(self, trade_data: Dict):
        """Notify when a trade is opened"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            position_type = trade_data.get('position_type', 'Unknown')
            entry_price = trade_data.get('entry_price', 0)
            quantity = trade_data.get('quantity', 0)
            strategy = trade_data.get('strategy', 'Unknown')
            
            await self.send_notification(
                level=NotificationLevel.INFO,
                title=f"📈 Trade Opened: {symbol}",
                message=f"Opened {position_type} position in {symbol}",
                metadata={
                    'Symbol': symbol,
                    'Type': position_type,
                    'Entry Price': f"${entry_price:.6f}",
                    'Quantity': f"{quantity:.6f}",
                    'Strategy': strategy,
                    'Value': f"${entry_price * quantity:.2f}"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error notifying trade opened: {e}")
    
    async def notify_trade_closed(self, trade_data: Dict):
        """Notify when a trade is closed"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            position_type = trade_data.get('position_type', 'Unknown')
            exit_price = trade_data.get('exit_price', 0)
            pnl = trade_data.get('realized_pnl', 0)
            pnl_pct = trade_data.get('realized_pnl_pct', 0)
            reason = trade_data.get('reason', 'Unknown')
            
            emoji = "💚" if pnl > 0 else "❤️"
            
            await self.send_notification(
                level=NotificationLevel.INFO if pnl > 0 else NotificationLevel.WARNING,
                title=f"{emoji} Trade Closed: {symbol}",
                message=f"Closed {position_type} position in {symbol} with {pnl_pct:.2%} {'profit' if pnl > 0 else 'loss'}",
                metadata={
                    'Symbol': symbol,
                    'Type': position_type,
                    'Exit Price': f"${exit_price:.6f}",
                    'P&L': f"${pnl:.2f}",
                    'P&L %': f"{pnl_pct:.2%}",
                    'Reason': reason
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error notifying trade closed: {e}")
    
    async def notify_risk_alert(self, alert_data: Dict):
        """Notify risk management alerts"""
        try:
            alert_type = alert_data.get('type', 'Risk Alert')
            message = alert_data.get('message', 'Risk threshold exceeded')
            severity = alert_data.get('severity', 'warning')
            
            level = NotificationLevel.WARNING
            if severity == 'critical':
                level = NotificationLevel.CRITICAL
            elif severity == 'error':
                level = NotificationLevel.ERROR
            
            await self.send_notification(
                level=level,
                title=f"🚨 {alert_type}",
                message=message,
                metadata=alert_data.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Error notifying risk alert: {e}")
    
    async def notify_system_status(self, status_data: Dict):
        """Notify system status updates"""
        try:
            status = status_data.get('status', 'Unknown')
            message = status_data.get('message', 'System status update')
            
            level = NotificationLevel.INFO
            if status in ['error', 'critical']:
                level = NotificationLevel.ERROR
            elif status == 'warning':
                level = NotificationLevel.WARNING
            
            await self.send_notification(
                level=level,
                title=f"🔧 System Status: {status.title()}",
                message=message,
                metadata=status_data.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Error notifying system status: {e}")
    
    def update_performance_data(self, performance_data: Dict):
        """Update performance data for monitoring"""
        try:
            performance_data['timestamp'] = datetime.utcnow()
            self.performance_history.append(performance_data)
            
        except Exception as e:
            self.logger.error(f"Error updating performance data: {e}")
    
    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""
        try:
            return {
                'total_sent': len(self.sent_messages),
                'total_failed': len(self.failed_messages),
                'queue_size': self.message_queue.qsize(),
                'rate_limits': {
                    channel.value: {
                        'count': limit['count'],
                        'reset_time': limit['reset_time'].isoformat()
                    }
                    for channel, limit in self.rate_limits.items()
                },
                'last_heartbeat': self.last_heartbeat.isoformat(),
                'running': self.running
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification stats: {e}")
            return {'error': str(e)}

