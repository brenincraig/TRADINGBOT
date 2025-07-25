"""
Comprehensive Risk Management System for Crypto Trading Bot
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import asyncio
from collections import deque, defaultdict

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class PositionType(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0

@dataclass
class Position:
    symbol: str
    position_type: PositionType
    entry_price: float
    current_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealized_pnl: float
    unrealized_pnl_pct: float
    max_profit: float
    max_loss: float
    trailing_stop: Optional[float]

@dataclass
class RiskMetrics:
    total_exposure: float
    max_position_size: float
    portfolio_var: float  # Value at Risk
    portfolio_beta: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    drawdown_current: float
    drawdown_max: float
    risk_level: RiskLevel

class RiskManager:
    def __init__(self, initial_capital: float = 10000):
        self.logger = logging.getLogger(__name__)
        
        # Capital and portfolio tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.total_exposure = 0.0
        
        # Positions tracking
        self.positions = {}  # symbol -> Position
        self.position_history = deque(maxlen=1000)
        
        # Risk parameters
        self.risk_params = {
            # Position sizing
            'max_position_size_pct': 0.05,  # 5% of capital per position
            'max_total_exposure_pct': 0.8,  # 80% of capital total exposure
            'max_correlation_exposure': 0.3,  # 30% in correlated positions
            
            # Stop loss and take profit
            'default_stop_loss_pct': 0.02,  # 2% stop loss
            'default_take_profit_pct': 0.04,  # 4% take profit
            'trailing_stop_pct': 0.015,  # 1.5% trailing stop
            'max_loss_per_trade_pct': 0.01,  # 1% max loss per trade
            
            # Portfolio limits
            'max_daily_loss_pct': 0.05,  # 5% daily loss limit
            'max_drawdown_pct': 0.15,  # 15% maximum drawdown
            'max_consecutive_losses': 5,  # Max consecutive losing trades
            'max_positions': 10,  # Maximum number of open positions
            
            # Volatility and market conditions
            'high_volatility_threshold': 0.05,  # 5% volatility threshold
            'low_liquidity_threshold': 1000,  # Minimum daily volume in USDT
            'correlation_threshold': 0.7,  # High correlation threshold
            
            # Emergency controls
            'emergency_stop_loss_pct': 0.10,  # 10% emergency stop
            'circuit_breaker_loss_pct': 0.08,  # 8% circuit breaker
            'max_trades_per_hour': 20,  # Rate limiting
            'cooldown_period_minutes': 30,  # Cooldown after losses
        }
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Trade history for risk analysis
        self.trade_history = deque(maxlen=1000)
        self.hourly_trade_count = deque(maxlen=24)  # Last 24 hours
        
        # Market data for risk calculations
        self.price_history = {}  # symbol -> price history
        self.volatility_cache = {}  # symbol -> volatility
        self.correlation_matrix = {}
        
        # Emergency state
        self.emergency_stop = False
        self.circuit_breaker_active = False
        self.last_trade_time = None
        self.cooldown_until = None
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, confidence: float = 1.0) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Risk per trade (distance to stop loss)
            if stop_loss_price <= 0:
                risk_per_unit = entry_price * self.risk_params['default_stop_loss_pct']
            else:
                risk_per_unit = abs(entry_price - stop_loss_price)
            
            # Maximum risk amount
            max_risk_amount = self.current_capital * self.risk_params['max_loss_per_trade_pct']
            
            # Position size based on risk
            risk_based_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Position size based on capital allocation
            max_position_value = self.current_capital * self.risk_params['max_position_size_pct']
            capital_based_size = max_position_value / entry_price
            
            # Adjust for confidence
            confidence_adjusted_size = min(risk_based_size, capital_based_size) * confidence
            
            # Check available capital
            required_capital = confidence_adjusted_size * entry_price
            if required_capital > self.available_capital:
                confidence_adjusted_size = self.available_capital / entry_price * 0.95  # 5% buffer
            
            # Minimum position size check
            min_position_value = 10  # $10 minimum
            if confidence_adjusted_size * entry_price < min_position_value:
                return 0
            
            return max(0, confidence_adjusted_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def validate_trade(self, symbol: str, position_type: PositionType, 
                      quantity: float, price: float) -> Tuple[bool, str]:
        """Validate if a trade can be executed based on risk rules"""
        try:
            # Check emergency stop
            if self.emergency_stop:
                return False, "Emergency stop is active"
            
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False, "Circuit breaker is active"
            
            # Check cooldown period
            if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
                return False, f"In cooldown period until {self.cooldown_until}"
            
            # Check maximum positions
            if len(self.positions) >= self.risk_params['max_positions']:
                return False, f"Maximum positions limit reached ({self.risk_params['max_positions']})"
            
            # Check if already have position in this symbol
            if symbol in self.positions:
                return False, f"Already have position in {symbol}"
            
            # Check daily loss limit
            if self.daily_pnl <= -self.current_capital * self.risk_params['max_daily_loss_pct']:
                return False, "Daily loss limit reached"
            
            # Check maximum drawdown
            if self.current_drawdown >= self.risk_params['max_drawdown_pct']:
                return False, f"Maximum drawdown limit reached: {self.current_drawdown:.2%}"
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_params['max_consecutive_losses']:
                return False, f"Maximum consecutive losses reached: {self.consecutive_losses}"
            
            # Check trade rate limiting
            current_hour_trades = len([t for t in self.hourly_trade_count 
                                     if t > datetime.utcnow() - timedelta(hours=1)])
            if current_hour_trades >= self.risk_params['max_trades_per_hour']:
                return False, "Hourly trade limit reached"
            
            # Check position size
            position_value = quantity * price
            max_position_value = self.current_capital * self.risk_params['max_position_size_pct']
            if position_value > max_position_value:
                return False, f"Position size too large: {position_value:.2f} > {max_position_value:.2f}"
            
            # Check total exposure
            new_total_exposure = self.total_exposure + position_value
            max_total_exposure = self.current_capital * self.risk_params['max_total_exposure_pct']
            if new_total_exposure > max_total_exposure:
                return False, f"Total exposure limit exceeded: {new_total_exposure:.2f} > {max_total_exposure:.2f}"
            
            # Check available capital
            if position_value > self.available_capital:
                return False, f"Insufficient capital: {position_value:.2f} > {self.available_capital:.2f}"
            
            # Check market volatility
            volatility = self.get_symbol_volatility(symbol)
            if volatility > self.risk_params['high_volatility_threshold']:
                # Reduce position size in high volatility
                adjusted_quantity = quantity * 0.5
                if adjusted_quantity * price < 10:  # Minimum position
                    return False, f"High volatility ({volatility:.2%}), position too small after adjustment"
            
            # Check correlation risk
            correlation_risk = self.calculate_correlation_risk(symbol)
            if correlation_risk > self.risk_params['max_correlation_exposure']:
                return False, f"Correlation risk too high: {correlation_risk:.2%}"
            
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def open_position(self, symbol: str, position_type: PositionType, 
                     entry_price: float, quantity: float, 
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """Open a new position with risk management"""
        try:
            # Validate trade
            is_valid, message = self.validate_trade(symbol, position_type, quantity, entry_price)
            if not is_valid:
                self.logger.warning(f"Trade rejected: {message}")
                return False
            
            # Calculate stop loss and take profit if not provided
            if stop_loss is None:
                if position_type == PositionType.LONG:
                    stop_loss = entry_price * (1 - self.risk_params['default_stop_loss_pct'])
                else:
                    stop_loss = entry_price * (1 + self.risk_params['default_stop_loss_pct'])
            
            if take_profit is None:
                if position_type == PositionType.LONG:
                    take_profit = entry_price * (1 + self.risk_params['default_take_profit_pct'])
                else:
                    take_profit = entry_price * (1 - self.risk_params['default_take_profit_pct'])
            
            # Create position
            position = Position(
                symbol=symbol,
                position_type=position_type,
                entry_price=entry_price,
                current_price=entry_price,
                quantity=quantity,
                entry_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                max_profit=0.0,
                max_loss=0.0,
                trailing_stop=None
            )
            
            # Update portfolio state
            self.positions[symbol] = position
            position_value = quantity * entry_price
            self.total_exposure += position_value
            self.available_capital -= position_value
            
            # Record trade
            self.hourly_trade_count.append(datetime.utcnow())
            self.last_trade_time = datetime.utcnow()
            
            self.logger.info(f"Opened {position_type.name} position: {symbol} @ {entry_price:.6f}, "
                           f"qty: {quantity:.6f}, SL: {stop_loss:.6f}, TP: {take_profit:.6f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current market price"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Calculate unrealized P&L
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            position.unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.quantity)
            
            # Update max profit/loss
            position.max_profit = max(position.max_profit, position.unrealized_pnl)
            position.max_loss = min(position.max_loss, position.unrealized_pnl)
            
            # Update trailing stop
            if position.trailing_stop is None and position.unrealized_pnl > 0:
                # Initialize trailing stop when in profit
                trailing_distance = current_price * self.risk_params['trailing_stop_pct']
                if position.position_type == PositionType.LONG:
                    position.trailing_stop = current_price - trailing_distance
                else:
                    position.trailing_stop = current_price + trailing_distance
            
            elif position.trailing_stop is not None:
                # Update trailing stop
                trailing_distance = current_price * self.risk_params['trailing_stop_pct']
                if position.position_type == PositionType.LONG:
                    new_trailing_stop = current_price - trailing_distance
                    position.trailing_stop = max(position.trailing_stop, new_trailing_stop)
                else:
                    new_trailing_stop = current_price + trailing_distance
                    position.trailing_stop = min(position.trailing_stop, new_trailing_stop)
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
    
    def check_exit_conditions(self, symbol: str) -> Tuple[bool, str]:
        """Check if position should be closed based on risk rules"""
        try:
            if symbol not in self.positions:
                return False, "No position found"
            
            position = self.positions[symbol]
            current_price = position.current_price
            
            # Stop loss check
            if position.stop_loss:
                if position.position_type == PositionType.LONG and current_price <= position.stop_loss:
                    return True, f"Stop loss triggered: {current_price:.6f} <= {position.stop_loss:.6f}"
                elif position.position_type == PositionType.SHORT and current_price >= position.stop_loss:
                    return True, f"Stop loss triggered: {current_price:.6f} >= {position.stop_loss:.6f}"
            
            # Take profit check
            if position.take_profit:
                if position.position_type == PositionType.LONG and current_price >= position.take_profit:
                    return True, f"Take profit triggered: {current_price:.6f} >= {position.take_profit:.6f}"
                elif position.position_type == PositionType.SHORT and current_price <= position.take_profit:
                    return True, f"Take profit triggered: {current_price:.6f} <= {position.take_profit:.6f}"
            
            # Trailing stop check
            if position.trailing_stop:
                if position.position_type == PositionType.LONG and current_price <= position.trailing_stop:
                    return True, f"Trailing stop triggered: {current_price:.6f} <= {position.trailing_stop:.6f}"
                elif position.position_type == PositionType.SHORT and current_price >= position.trailing_stop:
                    return True, f"Trailing stop triggered: {current_price:.6f} >= {position.trailing_stop:.6f}"
            
            # Emergency stop loss check
            loss_pct = abs(position.unrealized_pnl_pct)
            if loss_pct >= self.risk_params['emergency_stop_loss_pct']:
                return True, f"Emergency stop loss: {loss_pct:.2%} loss"
            
            # Time-based exit (optional - for scalping strategies)
            position_age = (datetime.utcnow() - position.entry_time).total_seconds() / 3600  # hours
            if position_age > 24:  # Close positions older than 24 hours
                return True, f"Time-based exit: position age {position_age:.1f} hours"
            
            return False, "No exit condition met"
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return False, f"Error: {e}"
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual close") -> bool:
        """Close a position and update portfolio state"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position to close for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Calculate realized P&L
            if position.position_type == PositionType.LONG:
                realized_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                realized_pnl = (position.entry_price - exit_price) * position.quantity
            
            realized_pnl_pct = realized_pnl / (position.entry_price * position.quantity)
            
            # Update portfolio state
            position_value = position.quantity * position.entry_price
            self.total_exposure -= position_value
            self.available_capital += position_value + realized_pnl
            self.current_capital += realized_pnl
            self.daily_pnl += realized_pnl
            
            # Update drawdown tracking
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Update consecutive losses
            if realized_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Record trade in history
            trade_record = {
                'symbol': symbol,
                'position_type': position.position_type.name,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'entry_time': position.entry_time,
                'exit_time': datetime.utcnow(),
                'duration': (datetime.utcnow() - position.entry_time).total_seconds() / 3600,
                'realized_pnl': realized_pnl,
                'realized_pnl_pct': realized_pnl_pct,
                'max_profit': position.max_profit,
                'max_loss': position.max_loss,
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            self.position_history.append(position)
            
            # Remove position
            del self.positions[symbol]
            
            # Check for circuit breaker
            if realized_pnl_pct <= -self.risk_params['circuit_breaker_loss_pct']:
                self.activate_circuit_breaker(f"Large loss: {realized_pnl_pct:.2%}")
            
            # Check for cooldown period
            if realized_pnl < 0 and self.consecutive_losses >= 3:
                self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.risk_params['cooldown_period_minutes'])
                self.logger.warning(f"Entering cooldown period until {self.cooldown_until}")
            
            self.logger.info(f"Closed position: {symbol} @ {exit_price:.6f}, "
                           f"P&L: {realized_pnl:.2f} ({realized_pnl_pct:.2%}), Reason: {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def close_all_positions(self, reason: str = "Emergency close"):
        """Close all open positions"""
        try:
            symbols_to_close = list(self.positions.keys())
            for symbol in symbols_to_close:
                position = self.positions[symbol]
                # Use current price for emergency close
                self.close_position(symbol, position.current_price, reason)
            
            self.logger.warning(f"Closed all positions: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop"""
        try:
            self.emergency_stop = True
            self.close_all_positions(f"Emergency stop: {reason}")
            self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error activating emergency stop: {e}")
    
    def activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker"""
        try:
            self.circuit_breaker_active = True
            self.close_all_positions(f"Circuit breaker: {reason}")
            
            # Auto-deactivate after 1 hour
            asyncio.create_task(self._deactivate_circuit_breaker_after_delay(3600))
            
            self.logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error activating circuit breaker: {e}")
    
    async def _deactivate_circuit_breaker_after_delay(self, delay_seconds: int):
        """Deactivate circuit breaker after delay"""
        await asyncio.sleep(delay_seconds)
        self.circuit_breaker_active = False
        self.logger.info("Circuit breaker deactivated")
    
    def get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol"""
        try:
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                prices = self.price_history[symbol][-20:]  # Last 20 prices
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(24)  # Daily volatility
                self.volatility_cache[symbol] = volatility
                return volatility
            
            return 0.02  # Default 2% volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.02
    
    def calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for a new position"""
        try:
            if not self.positions:
                return 0.0
            
            # Simplified correlation risk calculation
            # In practice, this would use actual price correlation
            existing_symbols = list(self.positions.keys())
            
            # Check for similar symbols (simplified)
            similar_count = 0
            for existing_symbol in existing_symbols:
                # Simple heuristic: same base currency
                if symbol.split('/')[0] == existing_symbol.split('/')[0]:
                    similar_count += 1
            
            correlation_risk = similar_count / len(existing_symbols) if existing_symbols else 0
            return correlation_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not self.positions:
                return 0.0
            
            # Simplified VaR calculation
            position_values = []
            position_volatilities = []
            
            for symbol, position in self.positions.items():
                position_value = position.quantity * position.current_price
                volatility = self.get_symbol_volatility(symbol)
                
                position_values.append(position_value)
                position_volatilities.append(volatility)
            
            # Portfolio volatility (simplified - assumes no correlation)
            portfolio_value = sum(position_values)
            weighted_volatilities = [(v/portfolio_value) * vol for v, vol in zip(position_values, position_volatilities)]
            portfolio_volatility = np.sqrt(sum([vol**2 for vol in weighted_volatilities]))
            
            # VaR calculation
            from scipy.stats import norm
            z_score = norm.ppf(confidence_level)
            var = portfolio_value * portfolio_volatility * z_score
            
            return var
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        try:
            # Calculate portfolio VaR
            portfolio_var = self.calculate_portfolio_var()
            
            # Calculate concentration risk
            if self.positions:
                position_values = [pos.quantity * pos.current_price for pos in self.positions.values()]
                max_position_value = max(position_values)
                concentration_risk = max_position_value / self.current_capital
            else:
                concentration_risk = 0.0
            
            # Calculate leverage ratio
            leverage_ratio = self.total_exposure / self.current_capital if self.current_capital > 0 else 0
            
            # Determine risk level
            risk_level = RiskLevel.LOW
            if (self.current_drawdown > 0.05 or 
                concentration_risk > 0.1 or 
                leverage_ratio > 0.5):
                risk_level = RiskLevel.MEDIUM
            
            if (self.current_drawdown > 0.1 or 
                concentration_risk > 0.2 or 
                leverage_ratio > 0.7 or
                self.consecutive_losses >= 3):
                risk_level = RiskLevel.HIGH
            
            if (self.current_drawdown > 0.15 or 
                concentration_risk > 0.3 or 
                leverage_ratio > 0.8 or
                self.consecutive_losses >= 5 or
                self.emergency_stop or
                self.circuit_breaker_active):
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetrics(
                total_exposure=self.total_exposure,
                max_position_size=self.current_capital * self.risk_params['max_position_size_pct'],
                portfolio_var=portfolio_var,
                portfolio_beta=1.0,  # Simplified
                correlation_risk=0.0,  # Simplified
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                drawdown_current=self.current_drawdown,
                drawdown_max=self.max_drawdown,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                total_exposure=0, max_position_size=0, portfolio_var=0,
                portfolio_beta=0, correlation_risk=0, concentration_risk=0,
                leverage_ratio=0, drawdown_current=0, drawdown_max=0,
                risk_level=RiskLevel.LOW
            )
    
    def update_risk_parameters(self, new_params: Dict):
        """Update risk parameters"""
        try:
            for key, value in new_params.items():
                if key in self.risk_params:
                    old_value = self.risk_params[key]
                    self.risk_params[key] = value
                    self.logger.info(f"Updated risk parameter {key}: {old_value} -> {value}")
                else:
                    self.logger.warning(f"Unknown risk parameter: {key}")
            
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {e}")
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            risk_metrics = self.get_risk_metrics()
            
            # Position summary
            position_summary = []
            for symbol, position in self.positions.items():
                position_summary.append({
                    'symbol': symbol,
                    'type': position.position_type.name,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'quantity': position.quantity,
                    'value': position.quantity * position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'duration_hours': (datetime.utcnow() - position.entry_time).total_seconds() / 3600
                })
            
            # Recent trades summary
            recent_trades = list(self.trade_history)[-10:] if self.trade_history else []
            
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'available': self.available_capital,
                    'total_exposure': self.total_exposure,
                    'daily_pnl': self.daily_pnl
                },
                'risk_metrics': {
                    'risk_level': risk_metrics.risk_level.name,
                    'current_drawdown': risk_metrics.drawdown_current,
                    'max_drawdown': risk_metrics.drawdown_max,
                    'leverage_ratio': risk_metrics.leverage_ratio,
                    'concentration_risk': risk_metrics.concentration_risk,
                    'portfolio_var': risk_metrics.portfolio_var
                },
                'positions': {
                    'count': len(self.positions),
                    'max_allowed': self.risk_params['max_positions'],
                    'details': position_summary
                },
                'trading_stats': {
                    'consecutive_losses': self.consecutive_losses,
                    'max_consecutive_losses': self.risk_params['max_consecutive_losses'],
                    'daily_trades': self.daily_trades,
                    'total_trades': len(self.trade_history)
                },
                'emergency_controls': {
                    'emergency_stop': self.emergency_stop,
                    'circuit_breaker': self.circuit_breaker_active,
                    'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None
                },
                'recent_trades': recent_trades,
                'risk_parameters': self.risk_params.copy()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of each day)"""
        try:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.logger.info("Reset daily statistics")
            
        except Exception as e:
            self.logger.error(f"Error resetting daily stats: {e}")
    
    def deactivate_emergency_controls(self):
        """Manually deactivate emergency controls (use with caution)"""
        try:
            self.emergency_stop = False
            self.circuit_breaker_active = False
            self.cooldown_until = None
            self.consecutive_losses = 0
            self.logger.warning("Emergency controls deactivated manually")
            
        except Exception as e:
            self.logger.error(f"Error deactivating emergency controls: {e}")

