"""
Strategy Optimization and Backtesting Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from scipy import optimize
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import sharpe_ratio
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float
    parameters: Dict
    trade_log: List[Dict]
    equity_curve: List[float]
    drawdown_curve: List[float]

class StrategyOptimizer:
    def __init__(self, data_manager, signal_detector):
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.signal_detector = signal_detector
        
        # Optimization settings
        self.optimization_trials = 200
        self.optimization_timeout = 1800  # 30 minutes
        self.cv_folds = 5
        
        # Backtesting settings
        self.initial_capital = 10000
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0005   # 0.05%
        
        # Parameter spaces for different strategies
        self.parameter_spaces = {
            'rsi_mean_reversion': {
                'rsi_period': (10, 30),
                'rsi_oversold': (20, 35),
                'rsi_overbought': (65, 80),
                'profit_target': (0.005, 0.03),
                'stop_loss': (0.01, 0.05)
            },
            'macd_momentum': {
                'macd_fast': (8, 16),
                'macd_slow': (20, 30),
                'macd_signal': (7, 12),
                'profit_target': (0.01, 0.04),
                'stop_loss': (0.01, 0.03)
            },
            'bollinger_bands': {
                'bb_period': (15, 25),
                'bb_std': (1.5, 2.5),
                'profit_target': (0.005, 0.025),
                'stop_loss': (0.01, 0.04)
            },
            'volume_breakout': {
                'volume_threshold': (1.5, 4.0),
                'price_change_threshold': (0.005, 0.02),
                'profit_target': (0.01, 0.05),
                'stop_loss': (0.01, 0.03)
            },
            'scalping': {
                'micro_trend_period': (3, 8),
                'velocity_threshold': (0.0001, 0.001),
                'profit_target': (0.001, 0.005),
                'stop_loss': (0.0005, 0.002)
            }
        }
        
        # Multi-objective optimization weights
        self.optimization_weights = {
            'return': 0.4,
            'sharpe': 0.3,
            'max_drawdown': 0.2,
            'win_rate': 0.1
        }
    
    def optimize_strategy(self, strategy_name: str, symbol: str, 
                         start_date: datetime, end_date: datetime,
                         optimization_metric: str = 'sharpe') -> Dict:
        """Optimize strategy parameters using Optuna"""
        try:
            self.logger.info(f"Starting optimization for {strategy_name} on {symbol}")
            
            # Get historical data
            historical_data = self.data_manager.get_historical_data(symbol, start_date, end_date)
            if len(historical_data) < 1000:
                self.logger.warning(f"Insufficient data for optimization: {len(historical_data)} points")
                return {}
            
            # Define optimization objective
            def objective(trial):
                # Sample parameters based on strategy type
                params = self._sample_parameters(trial, strategy_name)
                
                # Run backtest with these parameters
                result = self._backtest_strategy(strategy_name, symbol, historical_data, params)
                
                # Return optimization metric
                if optimization_metric == 'sharpe':
                    return result.sharpe_ratio
                elif optimization_metric == 'return':
                    return result.annualized_return
                elif optimization_metric == 'calmar':
                    return result.calmar_ratio
                elif optimization_metric == 'multi_objective':
                    return self._calculate_multi_objective_score(result)
                else:
                    return result.sharpe_ratio
            
            # Create and run optimization study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(
                objective, 
                n_trials=self.optimization_trials,
                timeout=self.optimization_timeout,
                show_progress_bar=True
            )
            
            # Get best parameters and results
            best_params = study.best_params
            best_value = study.best_value
            
            # Run final backtest with best parameters
            final_result = self._backtest_strategy(strategy_name, symbol, historical_data, best_params)
            
            optimization_result = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'optimization_metric': optimization_metric,
                'best_value': best_value,
                'best_parameters': best_params,
                'backtest_result': final_result,
                'optimization_trials': len(study.trials),
                'optimization_time': study.trials[-1].datetime_complete - study.trials[0].datetime_start,
                'parameter_importance': optuna.importance.get_param_importances(study)
            }
            
            self.logger.info(f"Optimization completed: {best_value:.4f} with {len(study.trials)} trials")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy {strategy_name}: {e}")
            return {}
    
    def _sample_parameters(self, trial, strategy_name: str) -> Dict:
        """Sample parameters for optimization trial"""
        try:
            if strategy_name not in self.parameter_spaces:
                return {}
            
            param_space = self.parameter_spaces[strategy_name]
            params = {}
            
            for param_name, (min_val, max_val) in param_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error sampling parameters: {e}")
            return {}
    
    def _backtest_strategy(self, strategy_name: str, symbol: str, 
                          data: pd.DataFrame, params: Dict) -> BacktestResult:
        """Run backtest for a strategy with given parameters"""
        try:
            # Initialize backtest state
            capital = self.initial_capital
            position = 0
            position_price = 0
            trades = []
            equity_curve = [capital]
            
            # Generate signals using the strategy
            signals = self._generate_strategy_signals(strategy_name, data, params)
            
            # Execute trades based on signals
            for i, (timestamp, signal) in enumerate(signals.iterrows()):
                current_price = data.iloc[i]['close']
                
                # Handle existing position
                if position != 0:
                    # Check for exit conditions
                    if self._should_exit_position(position, position_price, current_price, params):
                        # Close position
                        trade_return = self._calculate_trade_return(position, position_price, current_price)
                        capital *= (1 + trade_return)
                        
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': timestamp,
                            'entry_price': position_price,
                            'exit_price': current_price,
                            'position_size': position,
                            'return': trade_return,
                            'duration': (timestamp - position_entry_time).total_seconds() / 3600  # hours
                        })
                        
                        position = 0
                        position_price = 0
                
                # Handle new signals
                if position == 0 and signal['signal'] != 0:
                    # Open new position
                    position = signal['signal']  # 1 for long, -1 for short
                    position_price = current_price * (1 + self.slippage * position)  # Apply slippage
                    position_entry_time = timestamp
                    
                    # Apply commission
                    capital *= (1 - self.commission)
                
                equity_curve.append(capital)
            
            # Close any remaining position
            if position != 0:
                current_price = data.iloc[-1]['close']
                trade_return = self._calculate_trade_return(position, position_price, current_price)
                capital *= (1 + trade_return)
                
                trades.append({
                    'entry_time': position_entry_time,
                    'exit_time': data.index[-1],
                    'entry_price': position_price,
                    'exit_price': current_price,
                    'position_size': position,
                    'return': trade_return,
                    'duration': (data.index[-1] - position_entry_time).total_seconds() / 3600
                })
            
            # Calculate performance metrics
            result = self._calculate_backtest_metrics(
                strategy_name, trades, equity_curve, params, data.index[0], data.index[-1]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, total_trades=0,
                avg_trade_duration=0, best_trade=0, worst_trade=0,
                consecutive_wins=0, consecutive_losses=0, calmar_ratio=0,
                sortino_ratio=0, parameters=params, trade_log=[], 
                equity_curve=[], drawdown_curve=[]
            )
    
    def _generate_strategy_signals(self, strategy_name: str, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Generate trading signals for a specific strategy"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            if strategy_name == 'rsi_mean_reversion':
                # Calculate RSI
                rsi = self._calculate_rsi(data['close'], params['rsi_period'])
                
                # Generate signals
                signals.loc[rsi < params['rsi_oversold'], 'signal'] = 1  # Buy
                signals.loc[rsi > params['rsi_overbought'], 'signal'] = -1  # Sell
            
            elif strategy_name == 'macd_momentum':
                # Calculate MACD
                macd, macd_signal, macd_hist = self._calculate_macd(
                    data['close'], params['macd_fast'], params['macd_slow'], params['macd_signal']
                )
                
                # Generate signals
                signals.loc[(macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), 'signal'] = 1
                signals.loc[(macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), 'signal'] = -1
            
            elif strategy_name == 'bollinger_bands':
                # Calculate Bollinger Bands
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                    data['close'], params['bb_period'], params['bb_std']
                )
                
                # Generate signals
                signals.loc[data['close'] < bb_lower, 'signal'] = 1  # Buy at lower band
                signals.loc[data['close'] > bb_upper, 'signal'] = -1  # Sell at upper band
            
            elif strategy_name == 'volume_breakout':
                # Calculate volume ratio
                volume_ma = data['volume'].rolling(window=20).mean()
                volume_ratio = data['volume'] / volume_ma
                
                # Calculate price change
                price_change = data['close'].pct_change()
                
                # Generate signals
                buy_condition = (volume_ratio > params['volume_threshold']) & (price_change > params['price_change_threshold'])
                sell_condition = (volume_ratio > params['volume_threshold']) & (price_change < -params['price_change_threshold'])
                
                signals.loc[buy_condition, 'signal'] = 1
                signals.loc[sell_condition, 'signal'] = -1
            
            elif strategy_name == 'scalping':
                # Calculate micro trend
                micro_trend = data['close'].rolling(window=params['micro_trend_period']).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                )
                
                # Calculate velocity
                velocity = data['close'].diff()
                
                # Generate signals
                buy_condition = (micro_trend > 0) & (velocity > params['velocity_threshold'])
                sell_condition = (micro_trend < 0) & (velocity < -params['velocity_threshold'])
                
                signals.loc[buy_condition, 'signal'] = 1
                signals.loc[sell_condition, 'signal'] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {strategy_name}: {e}")
            return pd.DataFrame(index=data.index, data={'signal': 0})
    
    def _should_exit_position(self, position: int, entry_price: float, 
                             current_price: float, params: Dict) -> bool:
        """Determine if position should be exited"""
        try:
            if position == 1:  # Long position
                profit_pct = (current_price - entry_price) / entry_price
                return (profit_pct >= params['profit_target'] or 
                       profit_pct <= -params['stop_loss'])
            
            elif position == -1:  # Short position
                profit_pct = (entry_price - current_price) / entry_price
                return (profit_pct >= params['profit_target'] or 
                       profit_pct <= -params['stop_loss'])
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit condition: {e}")
            return False
    
    def _calculate_trade_return(self, position: int, entry_price: float, exit_price: float) -> float:
        """Calculate return for a trade"""
        try:
            if position == 1:  # Long position
                return (exit_price - entry_price) / entry_price - self.commission
            elif position == -1:  # Short position
                return (entry_price - exit_price) / entry_price - self.commission
            return 0
            
        except Exception as e:
            self.logger.error(f"Error calculating trade return: {e}")
            return 0
    
    def _calculate_backtest_metrics(self, strategy_name: str, trades: List[Dict], 
                                   equity_curve: List[float], params: Dict,
                                   start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        try:
            if not trades:
                return BacktestResult(
                    strategy_name=strategy_name, total_return=0, annualized_return=0,
                    volatility=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
                    profit_factor=0, total_trades=0, avg_trade_duration=0,
                    best_trade=0, worst_trade=0, consecutive_wins=0,
                    consecutive_losses=0, calmar_ratio=0, sortino_ratio=0,
                    parameters=params, trade_log=[], equity_curve=equity_curve,
                    drawdown_curve=[]
                )
            
            # Basic metrics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Trade statistics
            trade_returns = [trade['return'] for trade in trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if losing_trades else float('inf')
            
            best_trade = max(trade_returns) if trade_returns else 0
            worst_trade = min(trade_returns) if trade_returns else 0
            
            avg_trade_duration = np.mean([trade['duration'] for trade in trades]) if trades else 0
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for trade_return in trade_returns:
                if trade_return > 0:
                    current_wins += 1
                    current_losses = 0
                    consecutive_wins = max(consecutive_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    consecutive_losses = max(consecutive_losses, current_losses)
            
            # Risk metrics
            equity_returns = np.diff(equity_curve) / equity_curve[:-1]
            volatility = np.std(equity_returns) * np.sqrt(252) if len(equity_returns) > 1 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            negative_returns = [r for r in equity_returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            return BacktestResult(
                strategy_name=strategy_name,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trades),
                avg_trade_duration=avg_trade_duration,
                best_trade=best_trade,
                worst_trade=worst_trade,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                parameters=params,
                trade_log=trades,
                equity_curve=equity_curve,
                drawdown_curve=drawdown.tolist()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest metrics: {e}")
            return BacktestResult(
                strategy_name=strategy_name, total_return=0, annualized_return=0,
                volatility=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
                profit_factor=0, total_trades=0, avg_trade_duration=0,
                best_trade=0, worst_trade=0, consecutive_wins=0,
                consecutive_losses=0, calmar_ratio=0, sortino_ratio=0,
                parameters=params, trade_log=[], equity_curve=[],
                drawdown_curve=[]
            )
    
    def _calculate_multi_objective_score(self, result: BacktestResult) -> float:
        """Calculate multi-objective optimization score"""
        try:
            # Normalize metrics to 0-1 scale
            normalized_return = max(0, min(1, (result.annualized_return + 0.5) / 1.0))  # -50% to 50%
            normalized_sharpe = max(0, min(1, (result.sharpe_ratio + 2) / 4))  # -2 to 2
            normalized_drawdown = max(0, min(1, 1 + result.max_drawdown))  # 0 to -100%
            normalized_win_rate = result.win_rate  # Already 0-1
            
            # Weighted combination
            score = (self.optimization_weights['return'] * normalized_return +
                    self.optimization_weights['sharpe'] * normalized_sharpe +
                    self.optimization_weights['max_drawdown'] * normalized_drawdown +
                    self.optimization_weights['win_rate'] * normalized_win_rate)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-objective score: {e}")
            return 0
    
    def cross_validate_strategy(self, strategy_name: str, symbol: str,
                               start_date: datetime, end_date: datetime,
                               params: Dict) -> Dict:
        """Perform time series cross-validation"""
        try:
            # Get historical data
            data = self.data_manager.get_historical_data(symbol, start_date, end_date)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_results = []
            
            for train_idx, test_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Run backtest on test data
                result = self._backtest_strategy(strategy_name, symbol, test_data, params)
                cv_results.append({
                    'fold': len(cv_results) + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades
                })
            
            # Calculate cross-validation statistics
            cv_stats = {
                'mean_return': np.mean([r['total_return'] for r in cv_results]),
                'std_return': np.std([r['total_return'] for r in cv_results]),
                'mean_sharpe': np.mean([r['sharpe_ratio'] for r in cv_results]),
                'std_sharpe': np.std([r['sharpe_ratio'] for r in cv_results]),
                'mean_drawdown': np.mean([r['max_drawdown'] for r in cv_results]),
                'mean_win_rate': np.mean([r['win_rate'] for r in cv_results]),
                'consistency_score': 1 - np.std([r['total_return'] for r in cv_results]),
                'fold_results': cv_results
            }
            
            return cv_stats
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def optimize_portfolio(self, strategies: List[str], symbols: List[str],
                          start_date: datetime, end_date: datetime) -> Dict:
        """Optimize portfolio allocation across strategies and symbols"""
        try:
            # Get individual strategy results
            strategy_results = {}
            
            for strategy in strategies:
                for symbol in symbols:
                    key = f"{strategy}_{symbol}"
                    result = self.optimize_strategy(strategy, symbol, start_date, end_date)
                    if result:
                        strategy_results[key] = result
            
            if not strategy_results:
                return {}
            
            # Extract returns for portfolio optimization
            returns_matrix = []
            strategy_keys = list(strategy_results.keys())
            
            for key in strategy_keys:
                equity_curve = strategy_results[key]['backtest_result'].equity_curve
                returns = np.diff(equity_curve) / equity_curve[:-1]
                returns_matrix.append(returns)
            
            # Align return series (take minimum length)
            min_length = min(len(returns) for returns in returns_matrix)
            returns_matrix = [returns[:min_length] for returns in returns_matrix]
            returns_df = pd.DataFrame(returns_matrix).T
            returns_df.columns = strategy_keys
            
            # Portfolio optimization using mean-variance optimization
            def portfolio_objective(weights):
                weights = np.array(weights)
                portfolio_return = np.sum(returns_df.mean() * weights) * 252  # Annualized
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                return -sharpe  # Minimize negative Sharpe
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(len(strategy_keys)))  # Long-only
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/len(strategy_keys)] * len(strategy_keys))
            
            # Optimize
            result = optimize.minimize(
                portfolio_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(returns_df.mean() * optimal_weights) * 252
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_df.cov() * 252, optimal_weights)))
            portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            portfolio_result = {
                'optimal_weights': dict(zip(strategy_keys, optimal_weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_sharpe,
                'individual_strategies': strategy_results,
                'correlation_matrix': returns_df.corr().to_dict()
            }
            
            return portfolio_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return {}
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

