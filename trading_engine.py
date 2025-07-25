"""
Core Trading Engine for the crypto day trading bot
"""
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from config import Config
from database import DatabaseManager
from exchange_manager import ExchangeManager
from data_processor import DataProcessor

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    metadata: Dict = None

class TradingEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.exchange_manager = ExchangeManager()
        self.data_processor = DataProcessor(self.db_manager)
        
        # Trading state
        self.running = False
        self.positions = {}  # symbol -> position info
        self.pending_orders = {}  # order_id -> order info
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Risk management
        self.max_position_value = 0
        self.daily_loss_limit = 0
        self.current_exposure = 0.0
        
        # Performance tracking
        self.last_performance_update = datetime.utcnow()
        
    def initialize(self) -> bool:
        """Initialize the trading engine"""
        try:
            # Connect to database
            if not self.db_manager.connect():
                self.logger.error("Failed to connect to database")
                return False
            
            # Initialize exchanges
            if not self.exchange_manager.initialize_exchanges():
                self.logger.error("Failed to initialize exchanges")
                return False
            
            # Calculate risk limits based on current balance
            self._update_risk_limits()
            
            self.logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    def start(self):
        """Start the trading engine"""
        if not self.initialize():
            return False
        
        self.running = True
        self.logger.info("Starting trading engine...")
        
        # Start WebSocket feeds
        self.exchange_manager.start_websocket_feeds(
            Config.TRADING_PAIRS, 
            self.data_processor.process_ticker_data
        )
        
        # Start main trading loop
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        # Start performance monitoring
        performance_thread = threading.Thread(target=self._performance_monitor)
        performance_thread.daemon = True
        performance_thread.start()
        
        return True
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.exchange_manager.stop_websocket_feeds()
        self.logger.info("Trading engine stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check for abnormal volatility and pause if needed
                if self._check_abnormal_market_conditions():
                    self.logger.warning("Abnormal market conditions detected, pausing trading")
                    time.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Check daily loss limit
                if self._check_daily_loss_limit():
                    self.logger.warning("Daily loss limit reached, stopping trading")
                    self.stop()
                    break
                
                # Process each trading pair
                for symbol in Config.TRADING_PAIRS:
                    try:
                        self._process_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                
                # Update pending orders
                self._update_pending_orders()
                
                # Sleep before next iteration
                time.sleep(Config.LOOP_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _process_symbol(self, symbol: str):
        """Process trading signals for a specific symbol"""
        try:
            # Get current market data and indicators
            indicators = self.data_processor.get_current_indicators(symbol)
            if not indicators:
                return
            
            # Generate trading signals
            signals = self._generate_signals(symbol, indicators)
            
            # Execute trades based on signals
            for signal in signals:
                if signal.signal_type != SignalType.HOLD:
                    self._execute_signal(signal)
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _generate_signals(self, symbol: str, indicators: Dict) -> List[TradingSignal]:
        """Generate trading signals based on technical analysis"""
        signals = []
        current_price = None
        
        try:
            # Get current price
            ticker = self.exchange_manager.get_ticker(symbol)
            if not ticker:
                return signals
            
            current_price = ticker['last']
            
            # Strategy 1: RSI Oversold/Overbought
            rsi = indicators.get('rsi')
            if rsi:
                if rsi < Config.RSI_OVERSOLD and not self._has_position(symbol):
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.7,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="RSI_Oversold",
                        metadata={"rsi": rsi}
                    ))
                elif rsi > Config.RSI_OVERBOUGHT and self._has_position(symbol):
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="RSI_Overbought",
                        metadata={"rsi": rsi}
                    ))
            
            # Strategy 2: Volume Spike with Price Movement
            if self.data_processor.detect_volume_spike(symbol, Config.VOLUME_SPIKE_THRESHOLD):
                price_change = self.data_processor.get_price_change(symbol, periods=1)
                if price_change:
                    if price_change > 0.005 and not self._has_position(symbol):  # 0.5% up
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.8,
                            price=current_price,
                            timestamp=datetime.utcnow(),
                            strategy="Volume_Spike_Up",
                            metadata={"price_change": price_change}
                        ))
                    elif price_change < -0.005 and self._has_position(symbol):  # 0.5% down
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.8,
                            price=current_price,
                            timestamp=datetime.utcnow(),
                            strategy="Volume_Spike_Down",
                            metadata={"price_change": price_change}
                        ))
            
            # Strategy 3: Moving Average Crossover
            ma_crossover = self.data_processor.detect_ma_crossover(symbol)
            if ma_crossover:
                if ma_crossover == 'golden_cross' and not self._has_position(symbol):
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.6,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="MA_Golden_Cross",
                        metadata={"crossover": ma_crossover}
                    ))
                elif ma_crossover == 'death_cross' and self._has_position(symbol):
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.6,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="MA_Death_Cross",
                        metadata={"crossover": ma_crossover}
                    ))
            
            # Strategy 4: Small Pullback in Uptrend
            price_change_short = self.data_processor.get_price_change(symbol, periods=5)
            price_change_long = self.data_processor.get_price_change(symbol, periods=20)
            
            if price_change_short and price_change_long:
                # Uptrend with small pullback
                if (price_change_long > 0.02 and  # 2% up over 20 periods
                    price_change_short < -0.01 and  # 1% down over 5 periods
                    not self._has_position(symbol)):
                    
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.75,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="Pullback_Buy",
                        metadata={
                            "short_change": price_change_short,
                            "long_change": price_change_long
                        }
                    ))
            
            # Strategy 5: Quick Profit Taking
            if self._has_position(symbol):
                position = self.positions[symbol]
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if profit_pct >= Config.TAKE_PROFIT_PERCENTAGE:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=1.0,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="Take_Profit",
                        metadata={"profit_pct": profit_pct}
                    ))
                elif profit_pct <= -Config.STOP_LOSS_PERCENTAGE:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=1.0,
                        price=current_price,
                        timestamp=datetime.utcnow(),
                        strategy="Stop_Loss",
                        metadata={"loss_pct": profit_pct}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Pre-trade risk checks
            if not self._pre_trade_risk_check(signal):
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                return
            
            # Check slippage
            estimated_slippage = self.exchange_manager.calculate_slippage(
                signal.symbol, signal.signal_type.value, position_size
            )
            
            if estimated_slippage > Config.MAX_SLIPPAGE:
                self.logger.warning(f"Estimated slippage {estimated_slippage:.4f} exceeds limit for {signal.symbol}")
                return
            
            # Determine order type based on market conditions
            order_type = self._determine_order_type(signal)
            
            # Place the order
            order = self.exchange_manager.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                side=signal.signal_type.value,
                amount=position_size,
                price=signal.price if order_type == 'limit' else None
            )
            
            if order:
                # Track the order
                self.pending_orders[order['id']] = {
                    'order': order,
                    'signal': signal,
                    'timestamp': datetime.utcnow()
                }
                
                self.logger.info(f"Order placed: {signal.symbol} {signal.signal_type.value} "
                               f"{position_size} @ {signal.price} ({signal.strategy})")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def _pre_trade_risk_check(self, signal: TradingSignal) -> bool:
        """Perform pre-trade risk checks"""
        try:
            # Check if we already have a position (for buy signals)
            if signal.signal_type == SignalType.BUY and self._has_position(signal.symbol):
                return False
            
            # Check if we don't have a position (for sell signals)
            if signal.signal_type == SignalType.SELL and not self._has_position(signal.symbol):
                return False
            
            # Check maximum exposure
            if signal.signal_type == SignalType.BUY:
                position_value = self._calculate_position_size(signal) * signal.price
                if self.current_exposure + position_value > self.max_position_value:
                    self.logger.warning(f"Position would exceed maximum exposure limit")
                    return False
            
            # Check liquidity
            liquidity_score = self.data_processor.get_liquidity_score(signal.symbol)
            if liquidity_score < 0.3:  # Minimum liquidity threshold
                self.logger.warning(f"Insufficient liquidity for {signal.symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade risk check: {e}")
            return False
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get current balance
            balance = self.exchange_manager.get_balance()
            if not balance:
                return 0.0
            
            # Get USDT balance (or equivalent)
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            if usdt_balance <= 0:
                return 0.0
            
            # Calculate position size as percentage of balance
            max_position_value = usdt_balance * Config.MAX_POSITION_SIZE
            position_size = max_position_value / signal.price
            
            # Adjust for minimum order size requirements
            # This would need to be customized per exchange
            min_order_size = 0.001  # Example minimum
            if position_size < min_order_size:
                return 0.0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _determine_order_type(self, signal: TradingSignal) -> str:
        """Determine whether to use market or limit order"""
        try:
            # Use market orders for high-confidence signals or stop losses
            if (signal.confidence >= 0.9 or 
                signal.strategy in ['Stop_Loss', 'Take_Profit'] or
                self.data_processor.is_abnormal_volatility(signal.symbol)):
                return 'market'
            else:
                return 'limit'
                
        except Exception as e:
            self.logger.error(f"Error determining order type: {e}")
            return 'market'
    
    def _update_pending_orders(self):
        """Update status of pending orders"""
        try:
            completed_orders = []
            
            for order_id, order_info in self.pending_orders.items():
                # Check order status (simplified - would need exchange-specific implementation)
                order = order_info['order']
                signal = order_info['signal']
                
                # For demo purposes, assume orders complete after 30 seconds
                if datetime.utcnow() - order_info['timestamp'] > timedelta(seconds=30):
                    self._handle_order_completion(order, signal)
                    completed_orders.append(order_id)
            
            # Remove completed orders
            for order_id in completed_orders:
                del self.pending_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error updating pending orders: {e}")
    
    def _handle_order_completion(self, order: Dict, signal: TradingSignal):
        """Handle completed order"""
        try:
            # Update positions
            if signal.signal_type == SignalType.BUY:
                self.positions[signal.symbol] = {
                    'entry_price': signal.price,
                    'quantity': order['amount'],
                    'timestamp': datetime.utcnow(),
                    'strategy': signal.strategy
                }
                self.current_exposure += signal.price * order['amount']
                
            elif signal.signal_type == SignalType.SELL:
                if signal.symbol in self.positions:
                    position = self.positions[signal.symbol]
                    profit_loss = (signal.price - position['entry_price']) * position['quantity']
                    
                    # Update performance metrics
                    self.daily_pnl += profit_loss
                    self.total_trades += 1
                    if profit_loss > 0:
                        self.winning_trades += 1
                    
                    self.current_exposure -= position['entry_price'] * position['quantity']
                    del self.positions[signal.symbol]
            
            # Save trade to database
            trade_data = {
                'timestamp': datetime.utcnow(),
                'symbol': signal.symbol,
                'exchange': Config.DEFAULT_EXCHANGE,
                'order_type': 'market',  # Simplified
                'side': signal.signal_type.value,
                'price': signal.price,
                'quantity': order['amount'],
                'profit_loss': 0,  # Would calculate actual P&L
                'strategy_id': 1,  # Would map strategy to ID
                'order_id': order['id']
            }
            
            self.db_manager.save_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error handling order completion: {e}")
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol"""
        return symbol in self.positions
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        return self.daily_pnl <= -self.daily_loss_limit
    
    def _check_abnormal_market_conditions(self) -> bool:
        """Check for abnormal market conditions that should pause trading"""
        try:
            abnormal_count = 0
            for symbol in Config.TRADING_PAIRS:
                if self.data_processor.is_abnormal_volatility(symbol, Config.ABNORMAL_VOLATILITY_THRESHOLD):
                    abnormal_count += 1
            
            # If more than 50% of pairs show abnormal volatility, pause trading
            return abnormal_count > len(Config.TRADING_PAIRS) * 0.5
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return False
    
    def _update_risk_limits(self):
        """Update risk management limits based on current balance"""
        try:
            balance = self.exchange_manager.get_balance()
            if balance:
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                self.max_position_value = usdt_balance * Config.MAX_POSITION_SIZE
                self.daily_loss_limit = usdt_balance * Config.MAX_DAILY_LOSS
                
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")
    
    def _performance_monitor(self):
        """Monitor and update performance metrics"""
        while self.running:
            try:
                if datetime.utcnow() - self.last_performance_update > timedelta(seconds=Config.PERFORMANCE_UPDATE_INTERVAL):
                    self._update_performance_metrics()
                    self.last_performance_update = datetime.utcnow()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                time.sleep(60)
    
    def _update_performance_metrics(self):
        """Update performance metrics in database"""
        try:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            metrics = {
                'timestamp': datetime.utcnow(),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.total_trades - self.winning_trades,
                'total_profit_loss': self.daily_pnl,
                'win_rate': win_rate,
                'daily_return': self.daily_pnl / self.max_position_value if self.max_position_value > 0 else 0
            }
            
            # Save to database (would need to implement this method in DatabaseManager)
            # self.db_manager.save_performance_metrics(metrics)
            
            self.logger.info(f"Performance update - Trades: {self.total_trades}, "
                           f"Win Rate: {win_rate:.2%}, P&L: {self.daily_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_status(self) -> Dict:
        """Get current trading engine status"""
        return {
            'running': self.running,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'daily_pnl': self.daily_pnl,
            'current_exposure': self.current_exposure,
            'open_positions': len(self.positions),
            'pending_orders': len(self.pending_orders)
        }

