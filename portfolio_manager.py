"""
Portfolio Management and Allocation System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AllocationTarget:
    symbol: str
    target_weight: float
    current_weight: float
    deviation: float
    rebalance_needed: bool
    min_weight: float
    max_weight: float

@dataclass
class PortfolioMetrics:
    total_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    var_95: float
    cvar_95: float

class PortfolioManager:
    def __init__(self, risk_manager, initial_capital: float = 10000):
        self.logger = logging.getLogger(__name__)
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        
        # Portfolio composition
        self.target_allocations = {}  # symbol -> target weight
        self.current_allocations = {}  # symbol -> current weight
        self.allocation_history = deque(maxlen=1000)
        
        # Performance tracking
        self.portfolio_value_history = deque(maxlen=5000)
        self.benchmark_history = deque(maxlen=5000)
        self.returns_history = deque(maxlen=1000)
        
        # Rebalancing parameters
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        self.rebalance_frequency = timedelta(hours=6)  # Rebalance every 6 hours
        self.last_rebalance = datetime.utcnow()
        self.min_rebalance_amount = 50  # Minimum $50 to rebalance
        
        # Allocation constraints
        self.max_single_allocation = 0.3  # 30% max per symbol
        self.min_single_allocation = 0.01  # 1% min per symbol
        self.max_sector_allocation = 0.5  # 50% max per sector
        
        # Dynamic allocation parameters
        self.momentum_lookback = 20  # 20 periods for momentum
        self.volatility_lookback = 50  # 50 periods for volatility
        self.correlation_lookback = 100  # 100 periods for correlation
        
        # Strategy allocations
        self.strategy_allocations = {
            'scalping': 0.2,      # 20% for scalping strategies
            'momentum': 0.3,      # 30% for momentum strategies
            'mean_reversion': 0.25, # 25% for mean reversion
            'arbitrage': 0.15,    # 15% for arbitrage
            'breakout': 0.1       # 10% for breakout strategies
        }
        
        # Market regime adjustments
        self.regime_adjustments = {
            'trending': {'momentum': 1.2, 'mean_reversion': 0.8, 'breakout': 1.1},
            'ranging': {'momentum': 0.8, 'mean_reversion': 1.3, 'arbitrage': 1.1},
            'volatile': {'scalping': 0.7, 'momentum': 0.9, 'arbitrage': 1.2},
            'calm': {'scalping': 1.2, 'mean_reversion': 1.1, 'momentum': 0.9}
        }
        
    def update_target_allocations(self, market_data: Dict, market_regime: str = 'normal'):
        """Update target allocations based on market conditions"""
        try:
            # Get available symbols
            symbols = list(market_data.keys())
            if not symbols:
                return
            
            # Calculate momentum scores
            momentum_scores = self._calculate_momentum_scores(market_data)
            
            # Calculate volatility scores
            volatility_scores = self._calculate_volatility_scores(market_data)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(market_data)
            
            # Risk-adjusted returns
            risk_adjusted_scores = {}
            for symbol in symbols:
                momentum = momentum_scores.get(symbol, 0)
                volatility = volatility_scores.get(symbol, 0.02)
                risk_adjusted_scores[symbol] = momentum / volatility if volatility > 0 else 0
            
            # Apply market regime adjustments
            regime_adjusted_scores = self._apply_regime_adjustments(risk_adjusted_scores, market_regime)
            
            # Optimize portfolio allocation
            optimal_weights = self._optimize_portfolio_allocation(
                regime_adjusted_scores, correlation_matrix, symbols
            )
            
            # Update target allocations
            for symbol, weight in optimal_weights.items():
                self.target_allocations[symbol] = max(
                    self.min_single_allocation,
                    min(self.max_single_allocation, weight)
                )
            
            # Normalize weights to sum to 1
            total_weight = sum(self.target_allocations.values())
            if total_weight > 0:
                for symbol in self.target_allocations:
                    self.target_allocations[symbol] /= total_weight
            
            self.logger.info(f"Updated target allocations: {self.target_allocations}")
            
        except Exception as e:
            self.logger.error(f"Error updating target allocations: {e}")
    
    def _calculate_momentum_scores(self, market_data: Dict) -> Dict[str, float]:
        """Calculate momentum scores for each symbol"""
        momentum_scores = {}
        
        try:
            for symbol, data in market_data.items():
                if len(data) < self.momentum_lookback:
                    momentum_scores[symbol] = 0
                    continue
                
                prices = [d['price'] for d in data[-self.momentum_lookback:]]
                
                # Simple momentum: price change over lookback period
                momentum = (prices[-1] - prices[0]) / prices[0]
                
                # Adjust for consistency (reduce if too volatile)
                price_changes = np.diff(prices) / prices[:-1]
                consistency = 1 - np.std(price_changes)
                
                momentum_scores[symbol] = momentum * max(0, consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum scores: {e}")
        
        return momentum_scores
    
    def _calculate_volatility_scores(self, market_data: Dict) -> Dict[str, float]:
        """Calculate volatility scores for each symbol"""
        volatility_scores = {}
        
        try:
            for symbol, data in market_data.items():
                if len(data) < self.volatility_lookback:
                    volatility_scores[symbol] = 0.02  # Default volatility
                    continue
                
                prices = [d['price'] for d in data[-self.volatility_lookback:]]
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(24)  # Daily volatility
                
                volatility_scores[symbol] = volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility scores: {e}")
        
        return volatility_scores
    
    def _calculate_correlation_matrix(self, market_data: Dict) -> np.ndarray:
        """Calculate correlation matrix between symbols"""
        try:
            symbols = list(market_data.keys())
            if len(symbols) < 2:
                return np.eye(len(symbols))
            
            # Prepare returns data
            returns_data = []
            min_length = min(len(data) for data in market_data.values())
            min_length = min(min_length, self.correlation_lookback)
            
            for symbol in symbols:
                prices = [d['price'] for d in market_data[symbol][-min_length:]]
                returns = np.diff(np.log(prices))
                returns_data.append(returns)
            
            if min_length < 10:  # Not enough data
                return np.eye(len(symbols))
            
            # Calculate correlation matrix
            returns_matrix = np.array(returns_data)
            correlation_matrix = np.corrcoef(returns_matrix)
            
            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return np.eye(len(market_data))
    
    def _apply_regime_adjustments(self, scores: Dict[str, float], market_regime: str) -> Dict[str, float]:
        """Apply market regime adjustments to scores"""
        try:
            if market_regime not in self.regime_adjustments:
                return scores
            
            adjustments = self.regime_adjustments[market_regime]
            adjusted_scores = scores.copy()
            
            # Apply strategy-based adjustments (simplified)
            for symbol, score in scores.items():
                # Determine primary strategy for symbol (simplified heuristic)
                if 'BTC' in symbol or 'ETH' in symbol:
                    strategy = 'momentum'
                elif 'USDT' in symbol:
                    strategy = 'arbitrage'
                else:
                    strategy = 'mean_reversion'
                
                adjustment_factor = adjustments.get(strategy, 1.0)
                adjusted_scores[symbol] = score * adjustment_factor
            
            return adjusted_scores
            
        except Exception as e:
            self.logger.error(f"Error applying regime adjustments: {e}")
            return scores
    
    def _optimize_portfolio_allocation(self, scores: Dict[str, float], 
                                     correlation_matrix: np.ndarray, 
                                     symbols: List[str]) -> Dict[str, float]:
        """Optimize portfolio allocation using mean-variance optimization"""
        try:
            if not symbols or len(symbols) == 1:
                return {symbols[0]: 1.0} if symbols else {}
            
            # Expected returns (based on scores)
            expected_returns = np.array([scores.get(symbol, 0) for symbol in symbols])
            
            # Covariance matrix (simplified using correlation)
            volatilities = np.array([0.02] * len(symbols))  # Assume 2% volatility
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Add regularization to avoid singular matrix
            cov_matrix += np.eye(len(symbols)) * 1e-6
            
            # Optimization objective (maximize Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return -portfolio_return
                
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(self.min_single_allocation, self.max_single_allocation) for _ in symbols]
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / len(symbols)] * len(symbols))
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                # Fallback to equal weights
                optimal_weights = initial_weights
            
            return dict(zip(symbols, optimal_weights))
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / len(symbols) if symbols else 0
            return {symbol: equal_weight for symbol in symbols}
    
    def calculate_current_allocations(self) -> Dict[str, float]:
        """Calculate current portfolio allocations"""
        try:
            total_value = 0
            position_values = {}
            
            # Get position values
            for symbol, position in self.risk_manager.positions.items():
                position_value = position.quantity * position.current_price
                position_values[symbol] = position_value
                total_value += position_value
            
            # Add available cash
            total_value += self.risk_manager.available_capital
            
            # Calculate allocations
            if total_value > 0:
                self.current_allocations = {
                    symbol: value / total_value 
                    for symbol, value in position_values.items()
                }
                
                # Add cash allocation
                cash_allocation = self.risk_manager.available_capital / total_value
                if cash_allocation > 0.01:  # Only show if > 1%
                    self.current_allocations['CASH'] = cash_allocation
            else:
                self.current_allocations = {}
            
            return self.current_allocations
            
        except Exception as e:
            self.logger.error(f"Error calculating current allocations: {e}")
            return {}
    
    def get_rebalancing_targets(self) -> List[AllocationTarget]:
        """Get rebalancing targets"""
        try:
            current_allocations = self.calculate_current_allocations()
            rebalancing_targets = []
            
            # Check all target allocations
            for symbol, target_weight in self.target_allocations.items():
                current_weight = current_allocations.get(symbol, 0)
                deviation = abs(target_weight - current_weight)
                rebalance_needed = deviation > self.rebalance_threshold
                
                target = AllocationTarget(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    deviation=deviation,
                    rebalance_needed=rebalance_needed,
                    min_weight=self.min_single_allocation,
                    max_weight=self.max_single_allocation
                )
                
                rebalancing_targets.append(target)
            
            # Check for positions not in target allocations
            for symbol in current_allocations:
                if symbol not in self.target_allocations and symbol != 'CASH':
                    target = AllocationTarget(
                        symbol=symbol,
                        target_weight=0,
                        current_weight=current_allocations[symbol],
                        deviation=current_allocations[symbol],
                        rebalance_needed=True,
                        min_weight=0,
                        max_weight=0
                    )
                    rebalancing_targets.append(target)
            
            return rebalancing_targets
            
        except Exception as e:
            self.logger.error(f"Error getting rebalancing targets: {e}")
            return []
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        try:
            # Check time-based rebalancing
            time_since_rebalance = datetime.utcnow() - self.last_rebalance
            if time_since_rebalance >= self.rebalance_frequency:
                return True
            
            # Check deviation-based rebalancing
            targets = self.get_rebalancing_targets()
            for target in targets:
                if target.rebalance_needed:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance condition: {e}")
            return False
    
    def execute_rebalancing(self, market_prices: Dict[str, float]) -> bool:
        """Execute portfolio rebalancing"""
        try:
            if not self.should_rebalance():
                return False
            
            targets = self.get_rebalancing_targets()
            total_portfolio_value = self.risk_manager.current_capital
            
            rebalancing_orders = []
            
            for target in targets:
                if not target.rebalance_needed:
                    continue
                
                symbol = target.symbol
                if symbol == 'CASH':
                    continue
                
                current_price = market_prices.get(symbol)
                if not current_price:
                    continue
                
                # Calculate required position change
                target_value = target.target_weight * total_portfolio_value
                current_value = target.current_weight * total_portfolio_value
                value_change = target_value - current_value
                
                # Skip small changes
                if abs(value_change) < self.min_rebalance_amount:
                    continue
                
                # Calculate quantity change
                quantity_change = value_change / current_price
                
                rebalancing_orders.append({
                    'symbol': symbol,
                    'quantity_change': quantity_change,
                    'value_change': value_change,
                    'action': 'buy' if quantity_change > 0 else 'sell',
                    'price': current_price
                })
            
            # Execute rebalancing orders
            successful_orders = 0
            for order in rebalancing_orders:
                success = self._execute_rebalancing_order(order)
                if success:
                    successful_orders += 1
            
            if successful_orders > 0:
                self.last_rebalance = datetime.utcnow()
                self.logger.info(f"Rebalanced portfolio: {successful_orders}/{len(rebalancing_orders)} orders executed")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing rebalancing: {e}")
            return False
    
    def _execute_rebalancing_order(self, order: Dict) -> bool:
        """Execute a single rebalancing order"""
        try:
            symbol = order['symbol']
            quantity_change = order['quantity_change']
            price = order['price']
            
            if quantity_change > 0:
                # Buy more
                from risk_manager import PositionType
                
                # Check if we already have a position
                if symbol in self.risk_manager.positions:
                    # Add to existing position (simplified)
                    position = self.risk_manager.positions[symbol]
                    new_quantity = position.quantity + quantity_change
                    new_value = new_quantity * price
                    
                    # Update position
                    position.quantity = new_quantity
                    self.risk_manager.available_capital -= quantity_change * price
                    self.risk_manager.total_exposure += quantity_change * price
                else:
                    # Open new position
                    success = self.risk_manager.open_position(
                        symbol=symbol,
                        position_type=PositionType.LONG,
                        entry_price=price,
                        quantity=quantity_change
                    )
                    return success
            
            else:
                # Sell some or all
                if symbol in self.risk_manager.positions:
                    position = self.risk_manager.positions[symbol]
                    
                    if abs(quantity_change) >= position.quantity:
                        # Close entire position
                        success = self.risk_manager.close_position(
                            symbol=symbol,
                            exit_price=price,
                            reason="Rebalancing - full exit"
                        )
                        return success
                    else:
                        # Partial close (simplified)
                        position.quantity += quantity_change  # quantity_change is negative
                        value_change = quantity_change * price
                        self.risk_manager.available_capital -= value_change  # Add cash
                        self.risk_manager.total_exposure += value_change  # Reduce exposure
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing rebalancing order: {e}")
            return False
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            if len(self.portfolio_value_history) < 2:
                return PortfolioMetrics(
                    total_value=self.risk_manager.current_capital,
                    total_return=0, annualized_return=0, volatility=0,
                    sharpe_ratio=0, max_drawdown=0, beta=0, alpha=0,
                    information_ratio=0, tracking_error=0, var_95=0, cvar_95=0
                )
            
            # Portfolio values and returns
            portfolio_values = list(self.portfolio_value_history)
            portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Basic metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            # Annualized return
            days = len(portfolio_values) / 24  # Assuming hourly data
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Volatility
            volatility = np.std(portfolio_returns) * np.sqrt(24 * 365) if len(portfolio_returns) > 1 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Beta and Alpha (vs benchmark if available)
            beta = 1.0  # Simplified
            alpha = annualized_return - risk_free_rate  # Simplified
            
            # Information ratio and tracking error
            if len(self.benchmark_history) == len(self.portfolio_value_history):
                benchmark_values = list(self.benchmark_history)
                benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
                
                excess_returns = portfolio_returns - benchmark_returns
                tracking_error = np.std(excess_returns) * np.sqrt(24 * 365)
                information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            else:
                tracking_error = 0
                information_ratio = 0
            
            # Value at Risk (95%)
            if len(portfolio_returns) > 20:
                var_95 = np.percentile(portfolio_returns, 5) * portfolio_values[-1]
                cvar_95 = np.mean([r for r in portfolio_returns if r <= np.percentile(portfolio_returns, 5)]) * portfolio_values[-1]
            else:
                var_95 = 0
                cvar_95 = 0
            
            return PortfolioMetrics(
                total_value=portfolio_values[-1],
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(
                total_value=0, total_return=0, annualized_return=0, volatility=0,
                sharpe_ratio=0, max_drawdown=0, beta=0, alpha=0,
                information_ratio=0, tracking_error=0, var_95=0, cvar_95=0
            )
    
    def update_portfolio_value(self, current_value: float, benchmark_value: Optional[float] = None):
        """Update portfolio value history"""
        try:
            self.portfolio_value_history.append(current_value)
            
            if benchmark_value is not None:
                self.benchmark_history.append(benchmark_value)
            
            # Calculate returns
            if len(self.portfolio_value_history) > 1:
                return_pct = (current_value - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]
                self.returns_history.append(return_pct)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def get_allocation_report(self) -> Dict:
        """Generate comprehensive allocation report"""
        try:
            current_allocations = self.calculate_current_allocations()
            targets = self.get_rebalancing_targets()
            metrics = self.calculate_portfolio_metrics()
            
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': metrics.total_value,
                'target_allocations': self.target_allocations.copy(),
                'current_allocations': current_allocations.copy(),
                'rebalancing_targets': [
                    {
                        'symbol': t.symbol,
                        'target_weight': t.target_weight,
                        'current_weight': t.current_weight,
                        'deviation': t.deviation,
                        'rebalance_needed': t.rebalance_needed
                    } for t in targets
                ],
                'should_rebalance': self.should_rebalance(),
                'last_rebalance': self.last_rebalance.isoformat(),
                'portfolio_metrics': {
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'var_95': metrics.var_95
                },
                'strategy_allocations': self.strategy_allocations.copy(),
                'constraints': {
                    'max_single_allocation': self.max_single_allocation,
                    'min_single_allocation': self.min_single_allocation,
                    'rebalance_threshold': self.rebalance_threshold
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating allocation report: {e}")
            return {'error': str(e)}
    
    def update_strategy_allocations(self, new_allocations: Dict[str, float]):
        """Update strategy allocation weights"""
        try:
            # Validate allocations sum to 1
            total_allocation = sum(new_allocations.values())
            if abs(total_allocation - 1.0) > 0.01:
                self.logger.warning(f"Strategy allocations don't sum to 1: {total_allocation}")
                # Normalize
                for strategy in new_allocations:
                    new_allocations[strategy] /= total_allocation
            
            self.strategy_allocations.update(new_allocations)
            self.logger.info(f"Updated strategy allocations: {self.strategy_allocations}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy allocations: {e}")
    
    def get_diversification_metrics(self) -> Dict:
        """Calculate portfolio diversification metrics"""
        try:
            current_allocations = self.calculate_current_allocations()
            
            if not current_allocations:
                return {}
            
            # Remove cash from calculations
            allocations = {k: v for k, v in current_allocations.items() if k != 'CASH'}
            
            if not allocations:
                return {}
            
            weights = np.array(list(allocations.values()))
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = np.sum(weights ** 2)
            
            # Effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0
            
            # Diversification ratio (simplified)
            diversification_ratio = len(allocations) / effective_positions if effective_positions > 0 else 0
            
            # Concentration in top positions
            sorted_weights = sorted(weights, reverse=True)
            top_3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
            
            return {
                'herfindahl_index': hhi,
                'effective_positions': effective_positions,
                'diversification_ratio': diversification_ratio,
                'top_3_concentration': top_3_concentration,
                'number_of_positions': len(allocations),
                'max_position_weight': max(weights) if len(weights) > 0 else 0,
                'min_position_weight': min(weights) if len(weights) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification metrics: {e}")
            return {}

