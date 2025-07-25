"""
Adaptive Learning and Strategy Optimization System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import pickle
import json
from collections import deque, defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyPerformance:
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    average_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    last_updated: datetime

@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    volume_ratio: float
    market_regime: str  # 'trending', 'ranging', 'volatile', 'calm'
    time_of_day: int
    day_of_week: int
    timestamp: datetime

class AdaptiveLearningSystem:
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        
        # Performance tracking
        self.strategy_performance = {}  # strategy_name -> StrategyPerformance
        self.trade_history = deque(maxlen=10000)
        self.market_conditions = deque(maxlen=5000)
        
        # Machine learning models
        self.profit_predictor = None
        self.market_regime_classifier = None
        self.parameter_optimizer = None
        
        # Strategy parameters
        self.strategy_parameters = {}
        self.parameter_bounds = {}
        self.optimization_history = []
        
        # Reinforcement learning components
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike_threshold': 2.0,
            'profit_target': 0.01,
            'stop_loss': 0.02,
            'confidence_threshold': 0.6
        }
        
        # Pattern recognition
        self.pattern_success_rates = defaultdict(lambda: {'wins': 0, 'total': 0})
        self.market_condition_performance = defaultdict(list)
        
    def record_trade(self, trade_data: Dict):
        """Record a completed trade for learning"""
        try:
            trade_record = {
                'timestamp': trade_data['timestamp'],
                'symbol': trade_data['symbol'],
                'strategy': trade_data.get('strategy', 'unknown'),
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'quantity': trade_data['quantity'],
                'profit_loss': trade_data['profit_loss'],
                'profit_pct': trade_data['profit_loss'] / (trade_data['entry_price'] * trade_data['quantity']),
                'hold_time': trade_data.get('hold_time', 0),
                'market_conditions': trade_data.get('market_conditions', {}),
                'signals': trade_data.get('signals', {}),
                'slippage': trade_data.get('slippage', 0),
                'fees': trade_data.get('fees', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Update strategy performance
            self._update_strategy_performance(trade_record)
            
            # Update pattern success rates
            self._update_pattern_performance(trade_record)
            
            # Update Q-learning
            self._update_q_learning(trade_record)
            
            # Trigger parameter optimization if needed
            if len(self.trade_history) % 100 == 0:  # Every 100 trades
                self._trigger_parameter_optimization()
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def record_market_condition(self, condition: MarketCondition):
        """Record current market conditions"""
        try:
            self.market_conditions.append(condition)
            
            # Update market condition performance tracking
            if len(self.trade_history) > 0:
                recent_trades = [t for t in self.trade_history 
                               if t['timestamp'] > datetime.utcnow() - timedelta(hours=1)]
                
                if recent_trades:
                    avg_return = np.mean([t['profit_pct'] for t in recent_trades])
                    self.market_condition_performance[condition.market_regime].append(avg_return)
            
        except Exception as e:
            self.logger.error(f"Error recording market condition: {e}")
    
    def _update_strategy_performance(self, trade_record: Dict):
        """Update performance metrics for a strategy"""
        try:
            strategy_name = trade_record['strategy']
            
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_return=0.0,
                    average_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    profit_factor=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    consecutive_wins=0,
                    consecutive_losses=0,
                    last_updated=datetime.utcnow()
                )
            
            perf = self.strategy_performance[strategy_name]
            profit_pct = trade_record['profit_pct']
            
            # Update basic metrics
            perf.total_trades += 1
            perf.total_return += profit_pct
            perf.average_return = perf.total_return / perf.total_trades
            
            if profit_pct > 0:
                perf.winning_trades += 1
                perf.consecutive_wins += 1
                perf.consecutive_losses = 0
                perf.largest_win = max(perf.largest_win, profit_pct)
            else:
                perf.losing_trades += 1
                perf.consecutive_losses += 1
                perf.consecutive_wins = 0
                perf.largest_loss = min(perf.largest_loss, profit_pct)
            
            perf.win_rate = perf.winning_trades / perf.total_trades
            
            # Calculate advanced metrics
            if perf.winning_trades > 0:
                winning_trades = [t['profit_pct'] for t in self.trade_history 
                                if t['strategy'] == strategy_name and t['profit_pct'] > 0]
                perf.avg_win = np.mean(winning_trades)
            
            if perf.losing_trades > 0:
                losing_trades = [t['profit_pct'] for t in self.trade_history 
                               if t['strategy'] == strategy_name and t['profit_pct'] <= 0]
                perf.avg_loss = np.mean(losing_trades)
            
            # Profit factor
            if perf.avg_loss != 0:
                perf.profit_factor = abs(perf.avg_win * perf.winning_trades) / abs(perf.avg_loss * perf.losing_trades)
            
            # Sharpe ratio (simplified)
            strategy_returns = [t['profit_pct'] for t in self.trade_history if t['strategy'] == strategy_name]
            if len(strategy_returns) > 1:
                perf.sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
            
            # Max drawdown
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            perf.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            perf.last_updated = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def _update_pattern_performance(self, trade_record: Dict):
        """Update pattern recognition success rates"""
        try:
            signals = trade_record.get('signals', {})
            profit_pct = trade_record['profit_pct']
            
            for signal_type, signal_data in signals.items():
                pattern_key = f"{signal_type}_{signal_data.get('pattern', 'unknown')}"
                
                self.pattern_success_rates[pattern_key]['total'] += 1
                if profit_pct > 0:
                    self.pattern_success_rates[pattern_key]['wins'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating pattern performance: {e}")
    
    def _update_q_learning(self, trade_record: Dict):
        """Update Q-learning values"""
        try:
            # Define state based on market conditions and signals
            market_conditions = trade_record.get('market_conditions', {})
            state = self._encode_state(market_conditions)
            
            # Define action based on trade decision
            action = f"{trade_record['strategy']}_{trade_record.get('signal_type', 'unknown')}"
            
            # Reward is the profit percentage
            reward = trade_record['profit_pct']
            
            # Update Q-value using Q-learning formula
            current_q = self.q_table[state][action]
            
            # Get next state (simplified - using current state)
            next_state = state
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            
            # Q-learning update
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
            # Update state-action counts
            self.state_action_counts[state][action] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating Q-learning: {e}")
    
    def _encode_state(self, market_conditions: Dict) -> str:
        """Encode market conditions into a state string"""
        try:
            volatility = market_conditions.get('volatility', 0)
            trend_strength = market_conditions.get('trend_strength', 0)
            volume_ratio = market_conditions.get('volume_ratio', 1)
            market_regime = market_conditions.get('market_regime', 'normal')
            
            # Discretize continuous values
            vol_bucket = 'low' if volatility < 0.02 else 'medium' if volatility < 0.05 else 'high'
            trend_bucket = 'weak' if abs(trend_strength) < 0.3 else 'medium' if abs(trend_strength) < 0.7 else 'strong'
            volume_bucket = 'low' if volume_ratio < 0.8 else 'normal' if volume_ratio < 1.5 else 'high'
            
            return f"{market_regime}_{vol_bucket}_{trend_bucket}_{volume_bucket}"
            
        except Exception as e:
            self.logger.error(f"Error encoding state: {e}")
            return "unknown_unknown_unknown_unknown"
    
    def get_optimal_action(self, market_conditions: Dict) -> Tuple[str, float]:
        """Get optimal action based on current market conditions"""
        try:
            state = self._encode_state(market_conditions)
            
            if state not in self.q_table or not self.q_table[state]:
                return "hold", 0.0
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Exploration: random action
                actions = list(self.q_table[state].keys())
                action = np.random.choice(actions)
                confidence = 0.5
            else:
                # Exploitation: best action
                action = max(self.q_table[state], key=self.q_table[state].get)
                confidence = min(abs(self.q_table[state][action]), 1.0)
            
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Error getting optimal action: {e}")
            return "hold", 0.0
    
    def optimize_strategy_parameters(self, strategy_name: str) -> Dict:
        """Optimize parameters for a specific strategy"""
        try:
            if strategy_name not in self.strategy_performance:
                return {}
            
            # Get historical trades for this strategy
            strategy_trades = [t for t in self.trade_history if t['strategy'] == strategy_name]
            
            if len(strategy_trades) < 50:  # Need minimum data
                return {}
            
            # Define optimization objective
            def objective(trial):
                # Define parameter ranges to optimize
                params = {
                    'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
                    'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
                    'volume_threshold': trial.suggest_float('volume_threshold', 1.5, 4.0),
                    'profit_target': trial.suggest_float('profit_target', 0.005, 0.03),
                    'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                    'confidence_threshold': trial.suggest_float('confidence_threshold', 0.4, 0.8)
                }
                
                # Simulate trades with these parameters
                simulated_return = self._simulate_strategy_performance(strategy_trades, params)
                return simulated_return
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, timeout=300)  # 5 minutes max
            
            best_params = study.best_params
            best_value = study.best_value
            
            # Update strategy parameters
            if strategy_name not in self.strategy_parameters:
                self.strategy_parameters[strategy_name] = {}
            
            self.strategy_parameters[strategy_name].update(best_params)
            
            self.optimization_history.append({
                'timestamp': datetime.utcnow(),
                'strategy': strategy_name,
                'old_performance': self.strategy_performance[strategy_name].average_return,
                'new_performance': best_value,
                'parameters': best_params
            })
            
            self.logger.info(f"Optimized {strategy_name}: {best_value:.4f} return with params {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {e}")
            return {}
    
    def _simulate_strategy_performance(self, trades: List[Dict], params: Dict) -> float:
        """Simulate strategy performance with given parameters"""
        try:
            total_return = 0
            valid_trades = 0
            
            for trade in trades:
                # Apply parameter filters
                signals = trade.get('signals', {})
                market_conditions = trade.get('market_conditions', {})
                
                # Check if trade would have been taken with new parameters
                would_trade = True
                
                # RSI filter
                rsi = signals.get('rsi', 50)
                if trade['profit_pct'] > 0:  # Was a buy
                    if rsi > params['rsi_oversold']:
                        would_trade = False
                else:  # Was a sell
                    if rsi < params['rsi_overbought']:
                        would_trade = False
                
                # Volume filter
                volume_ratio = market_conditions.get('volume_ratio', 1)
                if volume_ratio < params['volume_threshold']:
                    would_trade = False
                
                # Confidence filter
                confidence = signals.get('confidence', 0.5)
                if confidence < params['confidence_threshold']:
                    would_trade = False
                
                if would_trade:
                    # Apply profit target and stop loss
                    original_return = trade['profit_pct']
                    
                    if original_return > 0:
                        # Limit profit to target
                        adjusted_return = min(original_return, params['profit_target'])
                    else:
                        # Limit loss to stop loss
                        adjusted_return = max(original_return, -params['stop_loss'])
                    
                    total_return += adjusted_return
                    valid_trades += 1
            
            return total_return / valid_trades if valid_trades > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error simulating strategy performance: {e}")
            return 0
    
    def _trigger_parameter_optimization(self):
        """Trigger parameter optimization for all strategies"""
        try:
            for strategy_name in self.strategy_performance:
                if self.strategy_performance[strategy_name].total_trades >= 50:
                    self.optimize_strategy_parameters(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Error triggering parameter optimization: {e}")
    
    def train_profit_predictor(self):
        """Train machine learning model to predict trade profitability"""
        try:
            if len(self.trade_history) < 100:
                return
            
            # Prepare training data
            features = []
            targets = []
            
            for trade in self.trade_history:
                feature_vector = self._extract_features(trade)
                if feature_vector is not None:
                    features.append(feature_vector)
                    targets.append(trade['profit_pct'])
            
            if len(features) < 50:
                return
            
            X = np.array(features)
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.profit_predictor = {
                'model': model,
                'scaler': scaler,
                'mse': mse,
                'r2': r2,
                'trained_at': datetime.utcnow()
            }
            
            self.logger.info(f"Trained profit predictor: MSE={mse:.6f}, R2={r2:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training profit predictor: {e}")
    
    def _extract_features(self, trade: Dict) -> Optional[List[float]]:
        """Extract features from trade data for ML model"""
        try:
            signals = trade.get('signals', {})
            market_conditions = trade.get('market_conditions', {})
            
            features = [
                signals.get('rsi', 50) / 100,  # Normalize RSI
                signals.get('macd', 0),
                signals.get('volume_ratio', 1),
                signals.get('confidence', 0.5),
                market_conditions.get('volatility', 0.02),
                market_conditions.get('trend_strength', 0),
                trade.get('hold_time', 0) / 3600,  # Convert to hours
                trade.get('slippage', 0),
                1 if trade['timestamp'].hour < 12 else 0,  # Morning trade
                trade['timestamp'].weekday() / 6,  # Day of week normalized
            ]
            
            # Check for valid features
            if any(np.isnan(f) or np.isinf(f) for f in features):
                return None
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def predict_trade_profitability(self, trade_signals: Dict, market_conditions: Dict) -> float:
        """Predict trade profitability using trained model"""
        try:
            if not self.profit_predictor:
                return 0.0
            
            # Create feature vector
            features = [
                trade_signals.get('rsi', 50) / 100,
                trade_signals.get('macd', 0),
                trade_signals.get('volume_ratio', 1),
                trade_signals.get('confidence', 0.5),
                market_conditions.get('volatility', 0.02),
                market_conditions.get('trend_strength', 0),
                0,  # Hold time (unknown at prediction time)
                0,  # Slippage (unknown at prediction time)
                1 if datetime.utcnow().hour < 12 else 0,
                datetime.utcnow().weekday() / 6,
            ]
            
            # Scale and predict
            X = np.array(features).reshape(1, -1)
            X_scaled = self.profit_predictor['scaler'].transform(X)
            prediction = self.profit_predictor['model'].predict(X_scaled)[0]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting trade profitability: {e}")
            return 0.0
    
    def adapt_thresholds(self):
        """Adapt trading thresholds based on recent performance"""
        try:
            if len(self.trade_history) < 100:
                return
            
            recent_trades = list(self.trade_history)[-100:]  # Last 100 trades
            
            # Analyze RSI performance
            rsi_buy_trades = [t for t in recent_trades if t.get('signals', {}).get('rsi', 50) < 35]
            rsi_sell_trades = [t for t in recent_trades if t.get('signals', {}).get('rsi', 50) > 65]
            
            if len(rsi_buy_trades) > 10:
                buy_win_rate = len([t for t in rsi_buy_trades if t['profit_pct'] > 0]) / len(rsi_buy_trades)
                if buy_win_rate < 0.4:  # Poor performance
                    self.adaptive_thresholds['rsi_oversold'] = max(25, self.adaptive_thresholds['rsi_oversold'] - 2)
                elif buy_win_rate > 0.6:  # Good performance
                    self.adaptive_thresholds['rsi_oversold'] = min(35, self.adaptive_thresholds['rsi_oversold'] + 2)
            
            if len(rsi_sell_trades) > 10:
                sell_win_rate = len([t for t in rsi_sell_trades if t['profit_pct'] > 0]) / len(rsi_sell_trades)
                if sell_win_rate < 0.4:
                    self.adaptive_thresholds['rsi_overbought'] = min(75, self.adaptive_thresholds['rsi_overbought'] + 2)
                elif sell_win_rate > 0.6:
                    self.adaptive_thresholds['rsi_overbought'] = max(65, self.adaptive_thresholds['rsi_overbought'] - 2)
            
            # Adapt volume threshold
            volume_trades = [t for t in recent_trades if t.get('market_conditions', {}).get('volume_ratio', 1) > 2]
            if len(volume_trades) > 10:
                volume_win_rate = len([t for t in volume_trades if t['profit_pct'] > 0]) / len(volume_trades)
                if volume_win_rate < 0.4:
                    self.adaptive_thresholds['volume_spike_threshold'] = min(4.0, self.adaptive_thresholds['volume_spike_threshold'] + 0.2)
                elif volume_win_rate > 0.6:
                    self.adaptive_thresholds['volume_spike_threshold'] = max(1.5, self.adaptive_thresholds['volume_spike_threshold'] - 0.2)
            
            # Adapt profit targets based on market volatility
            avg_volatility = np.mean([t.get('market_conditions', {}).get('volatility', 0.02) for t in recent_trades])
            if avg_volatility > 0.05:  # High volatility
                self.adaptive_thresholds['profit_target'] = min(0.03, self.adaptive_thresholds['profit_target'] * 1.1)
                self.adaptive_thresholds['stop_loss'] = min(0.05, self.adaptive_thresholds['stop_loss'] * 1.1)
            elif avg_volatility < 0.02:  # Low volatility
                self.adaptive_thresholds['profit_target'] = max(0.005, self.adaptive_thresholds['profit_target'] * 0.9)
                self.adaptive_thresholds['stop_loss'] = max(0.01, self.adaptive_thresholds['stop_loss'] * 0.9)
            
            self.logger.info(f"Adapted thresholds: {self.adaptive_thresholds}")
            
        except Exception as e:
            self.logger.error(f"Error adapting thresholds: {e}")
    
    def get_strategy_recommendations(self, market_conditions: Dict) -> List[Dict]:
        """Get strategy recommendations based on current market conditions"""
        try:
            recommendations = []
            
            # Get optimal action from Q-learning
            optimal_action, confidence = self.get_optimal_action(market_conditions)
            
            if confidence > 0.6:
                recommendations.append({
                    'type': 'q_learning',
                    'action': optimal_action,
                    'confidence': confidence,
                    'reason': 'Reinforcement learning recommendation'
                })
            
            # Market regime based recommendations
            market_regime = market_conditions.get('market_regime', 'normal')
            
            if market_regime in self.market_condition_performance:
                avg_performance = np.mean(self.market_condition_performance[market_regime][-20:])  # Last 20 observations
                
                if avg_performance > 0.01:  # Good performance in this regime
                    recommendations.append({
                        'type': 'regime_based',
                        'action': 'increase_activity',
                        'confidence': 0.7,
                        'reason': f'Good historical performance in {market_regime} market'
                    })
                elif avg_performance < -0.01:  # Poor performance
                    recommendations.append({
                        'type': 'regime_based',
                        'action': 'reduce_activity',
                        'confidence': 0.8,
                        'reason': f'Poor historical performance in {market_regime} market'
                    })
            
            # Pattern-based recommendations
            best_patterns = sorted(self.pattern_success_rates.items(), 
                                 key=lambda x: x[1]['wins'] / max(x[1]['total'], 1), 
                                 reverse=True)[:3]
            
            for pattern, stats in best_patterns:
                if stats['total'] > 10 and stats['wins'] / stats['total'] > 0.6:
                    recommendations.append({
                        'type': 'pattern_based',
                        'action': f'focus_on_{pattern}',
                        'confidence': stats['wins'] / stats['total'],
                        'reason': f'High success rate for {pattern}: {stats["wins"]}/{stats["total"]}'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting strategy recommendations: {e}")
            return []
    
    def save_learning_state(self, filepath: str):
        """Save the learning system state to file"""
        try:
            state = {
                'strategy_performance': {k: asdict(v) for k, v in self.strategy_performance.items()},
                'q_table': dict(self.q_table),
                'state_action_counts': dict(self.state_action_counts),
                'adaptive_thresholds': self.adaptive_thresholds,
                'pattern_success_rates': dict(self.pattern_success_rates),
                'strategy_parameters': self.strategy_parameters,
                'optimization_history': self.optimization_history,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Save ML models separately
            if self.profit_predictor:
                model_filepath = filepath.replace('.json', '_models.pkl')
                with open(model_filepath, 'wb') as f:
                    pickle.dump(self.profit_predictor, f)
            
            self.logger.info(f"Saved learning state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving learning state: {e}")
    
    def load_learning_state(self, filepath: str):
        """Load the learning system state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore strategy performance
            for name, perf_dict in state.get('strategy_performance', {}).items():
                perf_dict['last_updated'] = datetime.fromisoformat(perf_dict['last_updated'])
                self.strategy_performance[name] = StrategyPerformance(**perf_dict)
            
            # Restore Q-learning state
            self.q_table = defaultdict(lambda: defaultdict(float), state.get('q_table', {}))
            self.state_action_counts = defaultdict(lambda: defaultdict(int), state.get('state_action_counts', {}))
            
            # Restore other state
            self.adaptive_thresholds = state.get('adaptive_thresholds', self.adaptive_thresholds)
            self.pattern_success_rates = defaultdict(lambda: {'wins': 0, 'total': 0}, 
                                                   state.get('pattern_success_rates', {}))
            self.strategy_parameters = state.get('strategy_parameters', {})
            self.optimization_history = state.get('optimization_history', [])
            
            # Load ML models
            model_filepath = filepath.replace('.json', '_models.pkl')
            try:
                with open(model_filepath, 'rb') as f:
                    self.profit_predictor = pickle.load(f)
            except FileNotFoundError:
                pass
            
            self.logger.info(f"Loaded learning state from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")
    
    def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning system statistics"""
        try:
            stats = {
                'total_trades_recorded': len(self.trade_history),
                'strategies_tracked': len(self.strategy_performance),
                'q_table_size': len(self.q_table),
                'patterns_tracked': len(self.pattern_success_rates),
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'best_strategies': [],
                'worst_strategies': [],
                'optimization_count': len(self.optimization_history),
                'model_performance': {}
            }
            
            # Best and worst strategies
            if self.strategy_performance:
                sorted_strategies = sorted(self.strategy_performance.items(), 
                                         key=lambda x: x[1].average_return, reverse=True)
                
                stats['best_strategies'] = [(name, perf.average_return, perf.win_rate) 
                                          for name, perf in sorted_strategies[:3]]
                stats['worst_strategies'] = [(name, perf.average_return, perf.win_rate) 
                                           for name, perf in sorted_strategies[-3:]]
            
            # Model performance
            if self.profit_predictor:
                stats['model_performance'] = {
                    'mse': self.profit_predictor['mse'],
                    'r2': self.profit_predictor['r2'],
                    'trained_at': self.profit_predictor['trained_at'].isoformat()
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return {}

