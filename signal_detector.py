"""
Advanced Signal Detection Module for High-Frequency Trading
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from collections import deque
import asyncio

from technical_analysis import AdvancedTechnicalAnalysis, PatternSignal
from config import Config

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

class SignalTimeframe(Enum):
    SCALP = "1m"      # 1-5 minutes
    SHORT = "5m"      # 5-30 minutes  
    MEDIUM = "30m"    # 30 minutes - 4 hours
    LONG = "4h"       # 4+ hours

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    timeframe: SignalTimeframe
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    strategy_name: str
    metadata: Dict
    risk_reward_ratio: float = 0.0

class AdvancedSignalDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = AdvancedTechnicalAnalysis()
        
        # Signal history for filtering
        self.signal_history = {}  # symbol -> deque of recent signals
        self.market_regime = "normal"  # normal, volatile, trending, ranging
        
        # Performance tracking
        self.signal_performance = {}  # strategy -> performance metrics
        
        # Real-time data buffers
        self.price_buffers = {}  # symbol -> price data buffer
        self.volume_buffers = {}  # symbol -> volume data buffer
        self.order_book_buffers = {}  # symbol -> order book data buffer
        
    def update_market_data(self, symbol: str, data: Dict):
        """Update real-time market data"""
        try:
            # Initialize buffers if needed
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=1000)
                self.volume_buffers[symbol] = deque(maxlen=1000)
                self.signal_history[symbol] = deque(maxlen=100)
            
            # Add new data
            self.price_buffers[symbol].append({
                'timestamp': data['timestamp'],
                'price': data['price'],
                'volume': data.get('volume', 0)
            })
            
            # Update technical analyzer
            price_data = list(self.price_buffers[symbol])
            self.technical_analyzer.update_price_data(symbol, price_data)
            
        except Exception as e:
            self.logger.error(f"Error updating market data for {symbol}: {e}")
    
    def update_order_book(self, symbol: str, order_book: Dict):
        """Update order book data"""
        try:
            if symbol not in self.order_book_buffers:
                self.order_book_buffers[symbol] = deque(maxlen=100)
            
            self.order_book_buffers[symbol].append({
                'timestamp': datetime.utcnow(),
                'bids': order_book['bids'][:10],  # Top 10 levels
                'asks': order_book['asks'][:10],
                'spread': order_book['asks'][0][0] - order_book['bids'][0][0] if order_book['bids'] and order_book['asks'] else 0
            })
            
        except Exception as e:
            self.logger.error(f"Error updating order book for {symbol}: {e}")
    
    def detect_signals(self, symbol: str) -> List[TradingSignal]:
        """Main signal detection method"""
        signals = []
        
        try:
            if symbol not in self.price_buffers or len(self.price_buffers[symbol]) < 50:
                return signals
            
            # Update market regime
            self._update_market_regime(symbol)
            
            # Detect different types of signals
            signals.extend(self._detect_scalping_signals(symbol))
            signals.extend(self._detect_momentum_signals(symbol))
            signals.extend(self._detect_mean_reversion_signals(symbol))
            signals.extend(self._detect_breakout_signals(symbol))
            signals.extend(self._detect_arbitrage_signals(symbol))
            signals.extend(self._detect_volume_signals(symbol))
            signals.extend(self._detect_pattern_signals(symbol))
            signals.extend(self._detect_microstructure_signals(symbol))
            
            # Filter and rank signals
            signals = self._filter_signals(symbol, signals)
            signals = self._rank_signals(signals)
            
            # Update signal history
            for signal in signals:
                self.signal_history[symbol].append({
                    'timestamp': signal.timestamp,
                    'signal_type': signal.signal_type,
                    'strategy': signal.strategy_name,
                    'confidence': signal.confidence
                })
            
        except Exception as e:
            self.logger.error(f"Error detecting signals for {symbol}: {e}")
        
        return signals
    
    def _detect_scalping_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect ultra-short-term scalping signals"""
        signals = []
        
        try:
            if len(self.price_buffers[symbol]) < 20:
                return signals
            
            recent_prices = [p['price'] for p in list(self.price_buffers[symbol])[-20:]]
            current_price = recent_prices[-1]
            
            # Micro trend detection (last 5 prices)
            micro_trend = np.polyfit(range(5), recent_prices[-5:], 1)[0]
            
            # Price velocity and acceleration
            velocity = recent_prices[-1] - recent_prices[-2]
            acceleration = velocity - (recent_prices[-2] - recent_prices[-3])
            
            # Tick-by-tick momentum
            tick_momentum = sum([(recent_prices[i] - recent_prices[i-1]) for i in range(1, 5)])
            
            # Order book imbalance (if available)
            order_book_signal = self._analyze_order_book_imbalance(symbol)
            
            # Scalping buy signal
            if (micro_trend > 0 and 
                velocity > 0 and 
                acceleration > 0 and
                tick_momentum > 0 and
                order_book_signal == 'bullish'):
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='buy',
                    strength=SignalStrength.MODERATE,
                    timeframe=SignalTimeframe.SCALP,
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 0.999,  # 0.1% stop loss
                    take_profit=current_price * 1.002,  # 0.2% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='scalp_momentum',
                    metadata={
                        'micro_trend': micro_trend,
                        'velocity': velocity,
                        'acceleration': acceleration,
                        'tick_momentum': tick_momentum
                    },
                    risk_reward_ratio=2.0
                ))
            
            # Scalping sell signal
            elif (micro_trend < 0 and 
                  velocity < 0 and 
                  acceleration < 0 and
                  tick_momentum < 0 and
                  order_book_signal == 'bearish'):
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='sell',
                    strength=SignalStrength.MODERATE,
                    timeframe=SignalTimeframe.SCALP,
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 1.001,  # 0.1% stop loss
                    take_profit=current_price * 0.998,  # 0.2% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='scalp_momentum',
                    metadata={
                        'micro_trend': micro_trend,
                        'velocity': velocity,
                        'acceleration': acceleration,
                        'tick_momentum': tick_momentum
                    },
                    risk_reward_ratio=2.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting scalping signals for {symbol}: {e}")
        
        return signals
    
    def _detect_momentum_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect momentum-based signals"""
        signals = []
        
        try:
            indicators = self.technical_analyzer.calculate_advanced_indicators(symbol)
            if not indicators:
                return signals
            
            current_price = list(self.price_buffers[symbol])[-1]['price']
            
            # Multi-timeframe momentum
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            roc_5 = indicators.get('roc_5', 0)
            roc_10 = indicators.get('roc_10', 0)
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1)
            
            # Momentum alignment check
            momentum_score = 0
            if rsi > 50: momentum_score += 1
            if macd > macd_signal: momentum_score += 1
            if roc_5 > 0: momentum_score += 1
            if roc_10 > 0: momentum_score += 1
            if volume_ratio > 1.5: momentum_score += 1
            
            # Strong bullish momentum
            if momentum_score >= 4 and rsi < 70:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='buy',
                    strength=SignalStrength.STRONG,
                    timeframe=SignalTimeframe.SHORT,
                    confidence=0.8,
                    entry_price=current_price,
                    stop_loss=current_price * 0.98,  # 2% stop loss
                    take_profit=current_price * 1.04,  # 4% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='momentum_bullish',
                    metadata={
                        'momentum_score': momentum_score,
                        'rsi': rsi,
                        'macd_diff': macd - macd_signal,
                        'volume_ratio': volume_ratio
                    },
                    risk_reward_ratio=2.0
                ))
            
            # Strong bearish momentum
            elif momentum_score <= 1 and rsi > 30:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='sell',
                    strength=SignalStrength.STRONG,
                    timeframe=SignalTimeframe.SHORT,
                    confidence=0.8,
                    entry_price=current_price,
                    stop_loss=current_price * 1.02,  # 2% stop loss
                    take_profit=current_price * 0.96,  # 4% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='momentum_bearish',
                    metadata={
                        'momentum_score': momentum_score,
                        'rsi': rsi,
                        'macd_diff': macd - macd_signal,
                        'volume_ratio': volume_ratio
                    },
                    risk_reward_ratio=2.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum signals for {symbol}: {e}")
        
        return signals
    
    def _detect_mean_reversion_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect mean reversion signals"""
        signals = []
        
        try:
            indicators = self.technical_analyzer.calculate_advanced_indicators(symbol)
            if not indicators:
                return signals
            
            current_price = list(self.price_buffers[symbol])[-1]['price']
            
            # Bollinger Bands mean reversion
            bb_position = indicators.get('bb_position', 0.5)
            bb_width = indicators.get('bb_width', 0)
            rsi = indicators.get('rsi', 50)
            williams_r = indicators.get('williams_r', -50)
            
            # Oversold mean reversion (buy signal)
            if (bb_position < 0.1 and  # Near lower Bollinger Band
                rsi < 30 and          # Oversold RSI
                williams_r < -80 and  # Oversold Williams %R
                bb_width > 0.02):     # Sufficient volatility
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='buy',
                    strength=SignalStrength.MODERATE,
                    timeframe=SignalTimeframe.SHORT,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * 0.985,  # 1.5% stop loss
                    take_profit=current_price * 1.015,  # 1.5% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='mean_reversion_buy',
                    metadata={
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'williams_r': williams_r,
                        'bb_width': bb_width
                    },
                    risk_reward_ratio=1.0
                ))
            
            # Overbought mean reversion (sell signal)
            elif (bb_position > 0.9 and  # Near upper Bollinger Band
                  rsi > 70 and          # Overbought RSI
                  williams_r > -20 and  # Overbought Williams %R
                  bb_width > 0.02):     # Sufficient volatility
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='sell',
                    strength=SignalStrength.MODERATE,
                    timeframe=SignalTimeframe.SHORT,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * 1.015,  # 1.5% stop loss
                    take_profit=current_price * 0.985,  # 1.5% take profit
                    timestamp=datetime.utcnow(),
                    strategy_name='mean_reversion_sell',
                    metadata={
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'williams_r': williams_r,
                        'bb_width': bb_width
                    },
                    risk_reward_ratio=1.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting mean reversion signals for {symbol}: {e}")
        
        return signals
    
    def _detect_breakout_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect breakout signals"""
        signals = []
        
        try:
            indicators = self.technical_analyzer.calculate_advanced_indicators(symbol)
            if not indicators:
                return signals
            
            current_price = list(self.price_buffers[symbol])[-1]['price']
            
            # Support/resistance breakout
            nearest_resistance = indicators.get('nearest_resistance')
            nearest_support = indicators.get('nearest_support')
            volume_ratio = indicators.get('volume_ratio', 1)
            atr_percent = indicators.get('atr_percent', 0)
            
            # Resistance breakout (buy signal)
            if (nearest_resistance and 
                current_price > nearest_resistance * 1.002 and  # 0.2% above resistance
                volume_ratio > 2.0 and  # High volume confirmation
                atr_percent > 0.01):    # Sufficient volatility
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='buy',
                    strength=SignalStrength.STRONG,
                    timeframe=SignalTimeframe.MEDIUM,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=nearest_resistance * 0.995,  # Below resistance
                    take_profit=current_price * 1.03,  # 3% target
                    timestamp=datetime.utcnow(),
                    strategy_name='resistance_breakout',
                    metadata={
                        'resistance_level': nearest_resistance,
                        'breakout_strength': (current_price - nearest_resistance) / nearest_resistance,
                        'volume_ratio': volume_ratio
                    },
                    risk_reward_ratio=2.0
                ))
            
            # Support breakdown (sell signal)
            elif (nearest_support and 
                  current_price < nearest_support * 0.998 and  # 0.2% below support
                  volume_ratio > 2.0 and  # High volume confirmation
                  atr_percent > 0.01):    # Sufficient volatility
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='sell',
                    strength=SignalStrength.STRONG,
                    timeframe=SignalTimeframe.MEDIUM,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=nearest_support * 1.005,  # Above support
                    take_profit=current_price * 0.97,  # 3% target
                    timestamp=datetime.utcnow(),
                    strategy_name='support_breakdown',
                    metadata={
                        'support_level': nearest_support,
                        'breakdown_strength': (nearest_support - current_price) / nearest_support,
                        'volume_ratio': volume_ratio
                    },
                    risk_reward_ratio=2.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout signals for {symbol}: {e}")
        
        return signals
    
    def _detect_arbitrage_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect arbitrage opportunities"""
        signals = []
        
        try:
            # This would require real-time data from multiple exchanges
            # For now, we'll implement a placeholder that could be extended
            
            # Cross-exchange arbitrage would be implemented here
            # Triangular arbitrage detection
            # Statistical arbitrage opportunities
            
            pass
            
        except Exception as e:
            self.logger.error(f"Error detecting arbitrage signals for {symbol}: {e}")
        
        return signals
    
    def _detect_volume_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect volume-based signals"""
        signals = []
        
        try:
            if len(self.price_buffers[symbol]) < 20:
                return signals
            
            indicators = self.technical_analyzer.calculate_advanced_indicators(symbol)
            current_price = list(self.price_buffers[symbol])[-1]['price']
            
            volume_ratio = indicators.get('volume_ratio', 1)
            volume_trend = indicators.get('volume_trend', 0)
            obv = indicators.get('obv', 0)
            
            # Volume spike with price confirmation
            if volume_ratio > 3.0:  # 3x average volume
                price_change = indicators.get('roc_5', 0)
                
                if price_change > 0.01:  # 1% price increase
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type='buy',
                        strength=SignalStrength.STRONG,
                        timeframe=SignalTimeframe.SHORT,
                        confidence=0.8,
                        entry_price=current_price,
                        stop_loss=current_price * 0.99,
                        take_profit=current_price * 1.02,
                        timestamp=datetime.utcnow(),
                        strategy_name='volume_spike_buy',
                        metadata={
                            'volume_ratio': volume_ratio,
                            'price_change': price_change,
                            'volume_trend': volume_trend
                        },
                        risk_reward_ratio=2.0
                    ))
                
                elif price_change < -0.01:  # 1% price decrease
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type='sell',
                        strength=SignalStrength.STRONG,
                        timeframe=SignalTimeframe.SHORT,
                        confidence=0.8,
                        entry_price=current_price,
                        stop_loss=current_price * 1.01,
                        take_profit=current_price * 0.98,
                        timestamp=datetime.utcnow(),
                        strategy_name='volume_spike_sell',
                        metadata={
                            'volume_ratio': volume_ratio,
                            'price_change': price_change,
                            'volume_trend': volume_trend
                        },
                        risk_reward_ratio=2.0
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting volume signals for {symbol}: {e}")
        
        return signals
    
    def _detect_pattern_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect chart pattern signals"""
        signals = []
        
        try:
            patterns = self.technical_analyzer.detect_chart_patterns(symbol)
            current_price = list(self.price_buffers[symbol])[-1]['price']
            
            for pattern in patterns:
                if pattern.confidence > 0.6:
                    if pattern.direction == 'bullish':
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type='buy',
                            strength=SignalStrength.MODERATE,
                            timeframe=SignalTimeframe.MEDIUM,
                            confidence=pattern.confidence,
                            entry_price=current_price,
                            stop_loss=current_price * 0.97,
                            take_profit=current_price * 1.06,
                            timestamp=datetime.utcnow(),
                            strategy_name=f'pattern_{pattern.pattern_type}',
                            metadata={
                                'pattern_type': pattern.pattern_type,
                                'pattern_strength': pattern.strength,
                                'pattern_metadata': pattern.metadata
                            },
                            risk_reward_ratio=2.0
                        ))
                    
                    elif pattern.direction == 'bearish':
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type='sell',
                            strength=SignalStrength.MODERATE,
                            timeframe=SignalTimeframe.MEDIUM,
                            confidence=pattern.confidence,
                            entry_price=current_price,
                            stop_loss=current_price * 1.03,
                            take_profit=current_price * 0.94,
                            timestamp=datetime.utcnow(),
                            strategy_name=f'pattern_{pattern.pattern_type}',
                            metadata={
                                'pattern_type': pattern.pattern_type,
                                'pattern_strength': pattern.strength,
                                'pattern_metadata': pattern.metadata
                            },
                            risk_reward_ratio=2.0
                        ))
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern signals for {symbol}: {e}")
        
        return signals
    
    def _detect_microstructure_signals(self, symbol: str) -> List[TradingSignal]:
        """Detect market microstructure signals"""
        signals = []
        
        try:
            if symbol not in self.order_book_buffers or len(self.order_book_buffers[symbol]) < 5:
                return signals
            
            current_price = list(self.price_buffers[symbol])[-1]['price']
            recent_books = list(self.order_book_buffers[symbol])[-5:]
            
            # Order book imbalance
            imbalance = self._calculate_order_book_imbalance(recent_books[-1])
            
            # Spread analysis
            spreads = [book['spread'] for book in recent_books]
            avg_spread = np.mean(spreads)
            current_spread = spreads[-1]
            
            # Large order detection (simplified)
            large_bid = max([bid[1] for bid in recent_books[-1]['bids'][:5]]) if recent_books[-1]['bids'] else 0
            large_ask = max([ask[1] for ask in recent_books[-1]['asks'][:5]]) if recent_books[-1]['asks'] else 0
            
            # Microstructure buy signal
            if (imbalance > 0.6 and  # Strong bid side
                current_spread < avg_spread * 0.8 and  # Tightening spread
                large_bid > large_ask * 2):  # Large bid orders
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='buy',
                    strength=SignalStrength.WEAK,
                    timeframe=SignalTimeframe.SCALP,
                    confidence=0.5,
                    entry_price=current_price,
                    stop_loss=current_price * 0.9995,
                    take_profit=current_price * 1.001,
                    timestamp=datetime.utcnow(),
                    strategy_name='microstructure_buy',
                    metadata={
                        'imbalance': imbalance,
                        'spread_ratio': current_spread / avg_spread,
                        'large_bid': large_bid,
                        'large_ask': large_ask
                    },
                    risk_reward_ratio=2.0
                ))
            
            # Microstructure sell signal
            elif (imbalance < 0.4 and  # Strong ask side
                  current_spread < avg_spread * 0.8 and  # Tightening spread
                  large_ask > large_bid * 2):  # Large ask orders
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type='sell',
                    strength=SignalStrength.WEAK,
                    timeframe=SignalTimeframe.SCALP,
                    confidence=0.5,
                    entry_price=current_price,
                    stop_loss=current_price * 1.0005,
                    take_profit=current_price * 0.999,
                    timestamp=datetime.utcnow(),
                    strategy_name='microstructure_sell',
                    metadata={
                        'imbalance': imbalance,
                        'spread_ratio': current_spread / avg_spread,
                        'large_bid': large_bid,
                        'large_ask': large_ask
                    },
                    risk_reward_ratio=2.0
                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting microstructure signals for {symbol}: {e}")
        
        return signals
    
    def _analyze_order_book_imbalance(self, symbol: str) -> str:
        """Analyze order book imbalance"""
        try:
            if symbol not in self.order_book_buffers or not self.order_book_buffers[symbol]:
                return 'neutral'
            
            latest_book = self.order_book_buffers[symbol][-1]
            imbalance = self._calculate_order_book_imbalance(latest_book)
            
            if imbalance > 0.6:
                return 'bullish'
            elif imbalance < 0.4:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error analyzing order book imbalance: {e}")
            return 'neutral'
    
    def _calculate_order_book_imbalance(self, order_book: Dict) -> float:
        """Calculate order book imbalance ratio"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.5
            
            # Calculate total volume on each side (top 5 levels)
            bid_volume = sum([bid[1] for bid in order_book['bids'][:5]])
            ask_volume = sum([ask[1] for ask in order_book['asks'][:5]])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.5
            
            return bid_volume / total_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating order book imbalance: {e}")
            return 0.5
    
    def _update_market_regime(self, symbol: str):
        """Update market regime classification"""
        try:
            if len(self.price_buffers[symbol]) < 50:
                return
            
            indicators = self.technical_analyzer.calculate_advanced_indicators(symbol)
            
            # Volatility measure
            atr_percent = indicators.get('atr_percent', 0)
            
            # Trend strength
            trend_strength = abs(indicators.get('trend_strength', 0))
            
            # Volume activity
            volume_ratio = indicators.get('volume_ratio', 1)
            
            # Classify regime
            if atr_percent > 0.05:  # High volatility
                self.market_regime = "volatile"
            elif trend_strength > 0.5:  # Strong trend
                self.market_regime = "trending"
            elif volume_ratio < 0.5:  # Low volume
                self.market_regime = "ranging"
            else:
                self.market_regime = "normal"
                
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
    
    def _filter_signals(self, symbol: str, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on various criteria"""
        filtered_signals = []
        
        try:
            for signal in signals:
                # Skip if too many recent signals of same type
                recent_signals = [s for s in self.signal_history.get(symbol, []) 
                                if s['timestamp'] > datetime.utcnow() - timedelta(minutes=5)]
                same_type_count = len([s for s in recent_signals if s['signal_type'] == signal.signal_type])
                
                if same_type_count >= 3:  # Max 3 signals of same type in 5 minutes
                    continue
                
                # Filter based on market regime
                if self.market_regime == "volatile" and signal.timeframe == SignalTimeframe.SCALP:
                    continue  # Skip scalping in volatile markets
                
                # Minimum confidence threshold
                if signal.confidence < 0.4:
                    continue
                
                # Risk-reward ratio check
                if signal.risk_reward_ratio < 1.0:
                    continue
                
                filtered_signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return signals
        
        return filtered_signals
    
    def _rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by priority"""
        try:
            # Sort by confidence * strength * risk_reward_ratio
            def signal_score(signal):
                strength_multiplier = {
                    SignalStrength.WEAK: 1,
                    SignalStrength.MODERATE: 2,
                    SignalStrength.STRONG: 3,
                    SignalStrength.VERY_STRONG: 4
                }
                return (signal.confidence * 
                       strength_multiplier[signal.strength] * 
                       signal.risk_reward_ratio)
            
            return sorted(signals, key=signal_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error ranking signals: {e}")
            return signals
    
    def get_signal_performance(self, strategy_name: str) -> Dict:
        """Get performance metrics for a specific strategy"""
        return self.signal_performance.get(strategy_name, {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0
        })
    
    def update_signal_performance(self, signal: TradingSignal, outcome: Dict):
        """Update signal performance tracking"""
        try:
            strategy = signal.strategy_name
            if strategy not in self.signal_performance:
                self.signal_performance[strategy] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'total_return': 0.0
                }
            
            perf = self.signal_performance[strategy]
            perf['total_signals'] += 1
            
            if outcome['profit'] > 0:
                perf['successful_signals'] += 1
            
            perf['total_return'] += outcome['profit']
            perf['win_rate'] = perf['successful_signals'] / perf['total_signals']
            perf['avg_return'] = perf['total_return'] / perf['total_signals']
            
        except Exception as e:
            self.logger.error(f"Error updating signal performance: {e}")

