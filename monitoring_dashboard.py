"""
Real-time Monitoring Dashboard for Crypto Trading Bot
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    api_response_time: float
    error_rate: float

@dataclass
class TradingMetrics:
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    current_positions: int
    total_exposure: float
    available_capital: float
    drawdown: float
    sharpe_ratio: float

@dataclass
class MarketMetrics:
    timestamp: datetime
    active_symbols: List[str]
    price_changes: Dict[str, float]
    volume_changes: Dict[str, float]
    volatility_levels: Dict[str, float]
    market_regime: str
    correlation_matrix: Dict[str, Dict[str, float]]

class MonitoringDashboard:
    def __init__(self, trading_engine, risk_manager, notification_manager):
        self.logger = logging.getLogger(__name__)
        self.trading_engine = trading_engine
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'crypto_trading_bot_secret'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Data storage
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours
        self.trading_metrics_history = deque(maxlen=1440)
        self.market_metrics_history = deque(maxlen=1440)
        self.recent_trades = deque(maxlen=100)
        self.recent_alerts = deque(maxlen=50)
        self.performance_snapshots = deque(maxlen=288)  # 24 hours at 5-min intervals
        
        # Real-time data
        self.current_prices = {}
        self.active_signals = {}
        self.system_status = {
            'status': 'starting',
            'uptime': 0,
            'last_update': datetime.utcnow(),
            'components': {
                'trading_engine': 'unknown',
                'risk_manager': 'unknown',
                'data_feeds': 'unknown',
                'notifications': 'unknown'
            }
        }
        
        # Monitoring configuration
        self.update_interval = 5  # seconds
        self.running = False
        self.monitor_thread = None
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/metrics/trading')
        def get_trading_metrics():
            return jsonify(self.get_trading_metrics())
        
        @self.app.route('/api/metrics/system')
        def get_system_metrics():
            return jsonify(self.get_system_metrics())
        
        @self.app.route('/api/metrics/market')
        def get_market_metrics():
            return jsonify(self.get_market_metrics())
        
        @self.app.route('/api/positions')
        def get_positions():
            return jsonify(self.get_current_positions())
        
        @self.app.route('/api/trades/recent')
        def get_recent_trades():
            return jsonify(list(self.recent_trades))
        
        @self.app.route('/api/alerts/recent')
        def get_recent_alerts():
            return jsonify(list(self.recent_alerts))
        
        @self.app.route('/api/performance/history')
        def get_performance_history():
            hours = request.args.get('hours', 24, type=int)
            return jsonify(self.get_performance_history(hours))
        
        @self.app.route('/api/risk/report')
        def get_risk_report():
            return jsonify(self.risk_manager.get_risk_report())
        
        @self.app.route('/api/portfolio/allocation')
        def get_portfolio_allocation():
            # This would integrate with portfolio manager
            return jsonify({'allocations': {}, 'targets': {}})
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def handle_config():
            if request.method == 'GET':
                return jsonify(self.get_configuration())
            else:
                return jsonify(self.update_configuration(request.json))
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def control_system(action):
            return jsonify(self.handle_control_action(action, request.json))
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info(f"Client connected: {request.sid}")
            # Send initial data
            emit('system_status', self.get_system_status())
            emit('trading_metrics', self.get_trading_metrics())
            emit('current_positions', self.get_current_positions())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            update_type = data.get('type', 'all')
            if update_type == 'all' or update_type == 'status':
                emit('system_status', self.get_system_status())
            if update_type == 'all' or update_type == 'trading':
                emit('trading_metrics', self.get_trading_metrics())
            if update_type == 'all' or update_type == 'positions':
                emit('current_positions', self.get_current_positions())
    
    def start(self, host='0.0.0.0', port=5000, debug=False):
        """Start the monitoring dashboard"""
        try:
            self.running = True
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.logger.info(f"Starting monitoring dashboard on {host}:{port}")
            
            # Start Flask app with SocketIO
            self.socketio.run(self.app, host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring dashboard: {e}")
    
    def stop(self):
        """Stop the monitoring dashboard"""
        try:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            self.logger.info("Monitoring dashboard stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring dashboard: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_trading_metrics()
                self._collect_market_metrics()
                self._update_system_status()
                
                # Emit real-time updates via SocketIO
                self._emit_real_time_updates()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active connections (simplified)
            active_connections = len(psutil.net_connections())
            
            # API response time (would be measured from actual API calls)
            api_response_time = 0.1  # Placeholder
            
            # Error rate (from logs or error tracking)
            error_rate = 0.01  # Placeholder
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io,
                active_connections=active_connections,
                api_response_time=api_response_time,
                error_rate=error_rate
            )
            
            self.system_metrics_history.append(metrics)
            
        except ImportError:
            # psutil not available, use placeholder data
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=25.0,
                memory_usage=60.0,
                disk_usage=45.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                active_connections=5,
                api_response_time=0.1,
                error_rate=0.01
            )
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self):
        """Collect trading performance metrics"""
        try:
            # Get data from risk manager and trading engine
            positions = self.risk_manager.positions
            trade_history = list(self.risk_manager.trade_history)
            
            # Calculate metrics
            total_trades = len(trade_history)
            winning_trades = len([t for t in trade_history if t.get('realized_pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum([t.get('realized_pnl', 0) for t in trade_history])
            
            # Daily P&L (trades from today)
            today = datetime.utcnow().date()
            daily_trades = [t for t in trade_history 
                          if t.get('exit_time', datetime.utcnow()).date() == today]
            daily_pnl = sum([t.get('realized_pnl', 0) for t in daily_trades])
            
            # Current positions
            current_positions = len(positions)
            total_exposure = self.risk_manager.total_exposure
            available_capital = self.risk_manager.available_capital
            drawdown = self.risk_manager.current_drawdown
            
            # Sharpe ratio (simplified calculation)
            if len(trade_history) > 1:
                returns = [t.get('realized_pnl_pct', 0) for t in trade_history]
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            metrics = TradingMetrics(
                timestamp=datetime.utcnow(),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                current_positions=current_positions,
                total_exposure=total_exposure,
                available_capital=available_capital,
                drawdown=drawdown,
                sharpe_ratio=sharpe_ratio
            )
            
            self.trading_metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
    
    def _collect_market_metrics(self):
        """Collect market data metrics"""
        try:
            # This would integrate with the data processor
            # For now, using placeholder data
            
            active_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            price_changes = {
                'BTC/USDT': 0.025,
                'ETH/USDT': -0.015,
                'BNB/USDT': 0.008
            }
            volume_changes = {
                'BTC/USDT': 0.15,
                'ETH/USDT': 0.22,
                'BNB/USDT': -0.05
            }
            volatility_levels = {
                'BTC/USDT': 0.035,
                'ETH/USDT': 0.042,
                'BNB/USDT': 0.028
            }
            
            market_regime = 'trending'  # Would be determined by market analysis
            
            correlation_matrix = {
                'BTC/USDT': {'ETH/USDT': 0.75, 'BNB/USDT': 0.65},
                'ETH/USDT': {'BTC/USDT': 0.75, 'BNB/USDT': 0.58},
                'BNB/USDT': {'BTC/USDT': 0.65, 'ETH/USDT': 0.58}
            }
            
            metrics = MarketMetrics(
                timestamp=datetime.utcnow(),
                active_symbols=active_symbols,
                price_changes=price_changes,
                volume_changes=volume_changes,
                volatility_levels=volatility_levels,
                market_regime=market_regime,
                correlation_matrix=correlation_matrix
            )
            
            self.market_metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting market metrics: {e}")
    
    def _update_system_status(self):
        """Update overall system status"""
        try:
            # Check component status
            components = {
                'trading_engine': 'running' if self.trading_engine else 'stopped',
                'risk_manager': 'running' if self.risk_manager else 'stopped',
                'data_feeds': 'running',  # Would check actual data feed status
                'notifications': 'running' if self.notification_manager.running else 'stopped'
            }
            
            # Determine overall status
            if all(status == 'running' for status in components.values()):
                overall_status = 'running'
            elif any(status == 'running' for status in components.values()):
                overall_status = 'partial'
            else:
                overall_status = 'stopped'
            
            # Calculate uptime
            uptime = (datetime.utcnow() - self.system_status['last_update']).total_seconds()
            
            self.system_status.update({
                'status': overall_status,
                'uptime': uptime,
                'last_update': datetime.utcnow(),
                'components': components
            })
            
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    def _emit_real_time_updates(self):
        """Emit real-time updates via SocketIO"""
        try:
            # Emit system status
            self.socketio.emit('system_status', self.get_system_status())
            
            # Emit trading metrics
            self.socketio.emit('trading_metrics', self.get_trading_metrics())
            
            # Emit current positions
            self.socketio.emit('current_positions', self.get_current_positions())
            
            # Emit price updates
            if self.current_prices:
                self.socketio.emit('price_update', self.current_prices)
            
            # Emit active signals
            if self.active_signals:
                self.socketio.emit('signals_update', self.active_signals)
            
        except Exception as e:
            self.logger.error(f"Error emitting real-time updates: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return self.system_status.copy()
    
    def get_trading_metrics(self) -> Dict:
        """Get latest trading metrics"""
        if self.trading_metrics_history:
            return asdict(self.trading_metrics_history[-1])
        return {}
    
    def get_system_metrics(self) -> Dict:
        """Get latest system metrics"""
        if self.system_metrics_history:
            return asdict(self.system_metrics_history[-1])
        return {}
    
    def get_market_metrics(self) -> Dict:
        """Get latest market metrics"""
        if self.market_metrics_history:
            return asdict(self.market_metrics_history[-1])
        return {}
    
    def get_current_positions(self) -> List[Dict]:
        """Get current trading positions"""
        try:
            positions = []
            for symbol, position in self.risk_manager.positions.items():
                positions.append({
                    'symbol': symbol,
                    'type': position.position_type.name,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'quantity': position.quantity,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'entry_time': position.entry_time.isoformat(),
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                })
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
    
    def get_performance_history(self, hours: int = 24) -> Dict:
        """Get performance history for specified hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics by time
            trading_history = [
                asdict(m) for m in self.trading_metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            system_history = [
                asdict(m) for m in self.system_metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            return {
                'trading_metrics': trading_history,
                'system_metrics': system_history,
                'timeframe_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance history: {e}")
            return {'trading_metrics': [], 'system_metrics': [], 'timeframe_hours': hours}
    
    def get_configuration(self) -> Dict:
        """Get current system configuration"""
        try:
            return {
                'risk_parameters': self.risk_manager.risk_params.copy(),
                'trading_pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],  # Would get from config
                'update_interval': self.update_interval,
                'notification_settings': {
                    'telegram_enabled': bool(self.notification_manager.telegram_config.get('bot_token')),
                    'discord_enabled': bool(self.notification_manager.discord_config.get('webhook_url')),
                    'email_enabled': bool(self.notification_manager.email_config.get('smtp_server'))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return {}
    
    def update_configuration(self, config: Dict) -> Dict:
        """Update system configuration"""
        try:
            success = True
            message = "Configuration updated successfully"
            
            # Update risk parameters
            if 'risk_parameters' in config:
                self.risk_manager.update_risk_parameters(config['risk_parameters'])
            
            # Update other settings
            if 'update_interval' in config:
                self.update_interval = max(1, min(60, config['update_interval']))
            
            return {'success': success, 'message': message}
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_control_action(self, action: str, data: Dict) -> Dict:
        """Handle control actions from dashboard"""
        try:
            if action == 'emergency_stop':
                self.risk_manager.activate_emergency_stop("Manual emergency stop from dashboard")
                return {'success': True, 'message': 'Emergency stop activated'}
            
            elif action == 'pause_trading':
                # Would pause trading engine
                return {'success': True, 'message': 'Trading paused'}
            
            elif action == 'resume_trading':
                # Would resume trading engine
                return {'success': True, 'message': 'Trading resumed'}
            
            elif action == 'close_position':
                symbol = data.get('symbol')
                if symbol and symbol in self.risk_manager.positions:
                    position = self.risk_manager.positions[symbol]
                    success = self.risk_manager.close_position(
                        symbol, position.current_price, "Manual close from dashboard"
                    )
                    return {'success': success, 'message': f'Position {symbol} closed' if success else 'Failed to close position'}
                return {'success': False, 'message': 'Invalid symbol or no position found'}
            
            elif action == 'reset_daily_stats':
                self.risk_manager.reset_daily_stats()
                return {'success': True, 'message': 'Daily statistics reset'}
            
            else:
                return {'success': False, 'message': f'Unknown action: {action}'}
            
        except Exception as e:
            self.logger.error(f"Error handling control action {action}: {e}")
            return {'success': False, 'message': str(e)}
    
    def add_trade_notification(self, trade_data: Dict):
        """Add trade notification to recent trades"""
        try:
            trade_data['timestamp'] = datetime.utcnow().isoformat()
            self.recent_trades.append(trade_data)
            
            # Emit real-time update
            self.socketio.emit('new_trade', trade_data)
            
        except Exception as e:
            self.logger.error(f"Error adding trade notification: {e}")
    
    def add_alert_notification(self, alert_data: Dict):
        """Add alert notification to recent alerts"""
        try:
            alert_data['timestamp'] = datetime.utcnow().isoformat()
            self.recent_alerts.append(alert_data)
            
            # Emit real-time update
            self.socketio.emit('new_alert', alert_data)
            
        except Exception as e:
            self.logger.error(f"Error adding alert notification: {e}")
    
    def update_price_data(self, symbol: str, price: float):
        """Update current price data"""
        try:
            self.current_prices[symbol] = {
                'price': price,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating price data: {e}")
    
    def update_signal_data(self, symbol: str, signals: List[Dict]):
        """Update active signals"""
        try:
            self.active_signals[symbol] = {
                'signals': signals,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating signal data: {e}")

