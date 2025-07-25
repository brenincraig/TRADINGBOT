"""
Data Processor for handling real-time market data and technical analysis
"""
import pandas as pd
import numpy as np
import ta
from collections import deque
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from database import DatabaseManager

class DataProcessor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # In-memory data storage for real-time processing
        self.price_data = {}  # symbol -> deque of price data
        self.volume_data = {}  # symbol -> deque of volume data
        self.indicators = {}  # symbol -> dict of indicators
        
        # Configuration
        self.max_data_points = 1000  # Keep last 1000 data points in memory
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ma_short_period = 10
        self.ma_long_period = 50
        
    def process_ticker_data(self, ticker_data: Dict):
        """Process incoming ticker data"""
        try:
            symbol = ticker_data['symbol']
            price = ticker_data['price']
            volume = ticker_data.get('volume', 0)
            timestamp = ticker_data['timestamp']
            
            # Initialize data structures if needed
            if symbol not in self.price_data:
                self.price_data[symbol] = deque(maxlen=self.max_data_points)
                self.volume_data[symbol] = deque(maxlen=self.max_data_points)
                self.indicators[symbol] = {}
            
            # Add new data point
            self.price_data[symbol].append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })
            
            self.volume_data[symbol].append(volume)
            
            # Calculate technical indicators
            self._calculate_indicators(symbol)
            
            # Save to database
            market_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'exchange': ticker_data['exchange'],
                'close_price': price,
                'volume': volume,
                'rsi': self.indicators[symbol].get('rsi'),
                'macd': self.indicators[symbol].get('macd'),
                'moving_avg_short': self.indicators[symbol].get('ma_short'),
                'moving_avg_long': self.indicators[symbol].get('ma_long')
            }
            
            self.db_manager.save_market_data(market_data)
            
        except Exception as e:
            self.logger.error(f"Error processing ticker data: {e}")
    
    def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol"""
        try:
            if len(self.price_data[symbol]) < self.ma_long_period:
                return  # Not enough data
            
            # Convert to pandas DataFrame for easier calculation
            data = list(self.price_data[symbol])
            df = pd.DataFrame(data)
            df['close'] = df['price']
            
            # RSI
            if len(df) >= self.rsi_period:
                rsi = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period)
                self.indicators[symbol]['rsi'] = rsi.rsi().iloc[-1]
            
            # MACD
            if len(df) >= self.macd_slow:
                macd = ta.trend.MACD(df['close'], 
                                   window_fast=self.macd_fast,
                                   window_slow=self.macd_slow,
                                   window_sign=self.macd_signal)
                self.indicators[symbol]['macd'] = macd.macd().iloc[-1]
                self.indicators[symbol]['macd_signal'] = macd.macd_signal().iloc[-1]
                self.indicators[symbol]['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # Moving Averages
            if len(df) >= self.ma_short_period:
                self.indicators[symbol]['ma_short'] = df['close'].rolling(
                    window=self.ma_short_period).mean().iloc[-1]
            
            if len(df) >= self.ma_long_period:
                self.indicators[symbol]['ma_long'] = df['close'].rolling(
                    window=self.ma_long_period).mean().iloc[-1]
            
            # Volume indicators
            self._calculate_volume_indicators(symbol)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _calculate_volume_indicators(self, symbol: str):
        """Calculate volume-based indicators"""
        try:
            volumes = list(self.volume_data[symbol])
            if len(volumes) < 20:
                return
            
            # Average volume (20 periods)
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            self.indicators[symbol]['volume_ratio'] = volume_ratio
            self.indicators[symbol]['avg_volume'] = avg_volume
            
            # Volume trend
            if len(volumes) >= 10:
                recent_avg = np.mean(volumes[-10:])
                older_avg = np.mean(volumes[-20:-10])
                volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                self.indicators[symbol]['volume_trend'] = volume_trend
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators for {symbol}: {e}")
    
    def get_current_indicators(self, symbol: str) -> Dict:
        """Get current technical indicators for a symbol"""
        return self.indicators.get(symbol, {})
    
    def get_price_change(self, symbol: str, periods: int = 1) -> Optional[float]:
        """Get price change over specified periods"""
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < periods + 1:
                return None
            
            current_price = self.price_data[symbol][-1]['price']
            past_price = self.price_data[symbol][-(periods + 1)]['price']
            
            return (current_price - past_price) / past_price
            
        except Exception as e:
            self.logger.error(f"Error calculating price change for {symbol}: {e}")
            return None
    
    def detect_volume_spike(self, symbol: str, threshold: float = 2.0) -> bool:
        """Detect if there's a volume spike"""
        try:
            indicators = self.indicators.get(symbol, {})
            volume_ratio = indicators.get('volume_ratio', 1.0)
            return volume_ratio >= threshold
        except Exception as e:
            self.logger.error(f"Error detecting volume spike for {symbol}: {e}")
            return False
    
    def detect_price_breakout(self, symbol: str, lookback_periods: int = 20) -> Optional[str]:
        """Detect price breakouts (above resistance or below support)"""
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < lookback_periods + 1:
                return None
            
            prices = [point['price'] for point in list(self.price_data[symbol])[-lookback_periods-1:]]
            current_price = prices[-1]
            historical_prices = prices[:-1]
            
            resistance = max(historical_prices)
            support = min(historical_prices)
            
            if current_price > resistance:
                return 'breakout_up'
            elif current_price < support:
                return 'breakout_down'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout for {symbol}: {e}")
            return None
    
    def detect_rsi_divergence(self, symbol: str, lookback_periods: int = 10) -> Optional[str]:
        """Detect RSI divergence patterns"""
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < lookback_periods + 1:
                return None
            
            indicators = self.indicators.get(symbol, {})
            if 'rsi' not in indicators:
                return None
            
            # Get recent price and RSI data
            recent_data = list(self.price_data[symbol])[-lookback_periods:]
            prices = [point['price'] for point in recent_data]
            
            # This is a simplified divergence detection
            # In practice, you'd want more sophisticated peak/trough detection
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # For a more complete implementation, you'd need historical RSI values
            # This is a placeholder for the concept
            current_rsi = indicators['rsi']
            
            if price_trend > 0 and current_rsi < 30:
                return 'bullish_divergence'
            elif price_trend < 0 and current_rsi > 70:
                return 'bearish_divergence'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting RSI divergence for {symbol}: {e}")
            return None
    
    def detect_ma_crossover(self, symbol: str) -> Optional[str]:
        """Detect moving average crossovers"""
        try:
            indicators = self.indicators.get(symbol, {})
            ma_short = indicators.get('ma_short')
            ma_long = indicators.get('ma_long')
            
            if ma_short is None or ma_long is None:
                return None
            
            # Get previous values to detect crossover
            if len(self.price_data[symbol]) < 2:
                return None
            
            # This is simplified - in practice you'd track previous MA values
            if ma_short > ma_long:
                return 'golden_cross'  # Bullish signal
            elif ma_short < ma_long:
                return 'death_cross'   # Bearish signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting MA crossover for {symbol}: {e}")
            return None
    
    def get_market_volatility(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Calculate market volatility (standard deviation of returns)"""
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < periods + 1:
                return None
            
            prices = [point['price'] for point in list(self.price_data[symbol])[-periods-1:]]
            returns = []
            
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            return np.std(returns) if returns else None
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def is_abnormal_volatility(self, symbol: str, threshold: float = 0.1) -> bool:
        """Detect abnormal volatility that might indicate news events"""
        try:
            # Check recent price change
            price_change = self.get_price_change(symbol, periods=1)
            if price_change and abs(price_change) > threshold:
                return True
            
            # Check volume spike
            if self.detect_volume_spike(symbol, threshold=3.0):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting abnormal volatility for {symbol}: {e}")
            return False
    
    def get_liquidity_score(self, symbol: str) -> float:
        """Calculate a simple liquidity score based on volume"""
        try:
            indicators = self.indicators.get(symbol, {})
            avg_volume = indicators.get('avg_volume', 0)
            
            # Simple scoring: higher volume = higher liquidity
            # This could be enhanced with bid-ask spread data
            if avg_volume > 1000000:
                return 1.0  # High liquidity
            elif avg_volume > 100000:
                return 0.7  # Medium liquidity
            elif avg_volume > 10000:
                return 0.4  # Low liquidity
            else:
                return 0.1  # Very low liquidity
                
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score for {symbol}: {e}")
            return 0.1

