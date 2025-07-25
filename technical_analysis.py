"""
Advanced Technical Analysis Module for Crypto Trading Bot
"""
import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

@dataclass
class PatternSignal:
    pattern_type: str
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float
    metadata: Dict

class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_history = {}  # symbol -> DataFrame
        self.pattern_cache = {}  # symbol -> recent patterns
        
    def update_price_data(self, symbol: str, price_data: List[Dict]):
        """Update price data for technical analysis"""
        try:
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            # Keep only recent data (last 1000 points)
            if len(df) > 1000:
                df = df.tail(1000)
            
            self.price_history[symbol] = df
            
        except Exception as e:
            self.logger.error(f"Error updating price data for {symbol}: {e}")
    
    def calculate_advanced_indicators(self, symbol: str) -> Dict:
        """Calculate advanced technical indicators"""
        if symbol not in self.price_history:
            return {}
        
        df = self.price_history[symbol].copy()
        if len(df) < 50:
            return {}
        
        indicators = {}
        
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['price'], window=20, window_dev=2)
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (df['price'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['price'], df['price'], df['price'], window=14, smooth_window=3)
            indicators['stoch_k'] = stoch.stoch().iloc[-1]
            indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
            
            # Williams %R
            williams_r = ta.momentum.WilliamsRIndicator(df['price'], df['price'], df['price'], lbp=14)
            indicators['williams_r'] = williams_r.williams_r().iloc[-1]
            
            # Commodity Channel Index (CCI)
            cci = ta.trend.CCIIndicator(df['price'], df['price'], df['price'], window=20)
            indicators['cci'] = cci.cci().iloc[-1]
            
            # Average True Range (ATR)
            atr = ta.volatility.AverageTrueRange(df['price'], df['price'], df['price'], window=14)
            indicators['atr'] = atr.average_true_range().iloc[-1]
            indicators['atr_percent'] = indicators['atr'] / df['price'].iloc[-1]
            
            # Parabolic SAR
            psar = ta.trend.PSARIndicator(df['price'], df['price'])
            indicators['psar'] = psar.psar().iloc[-1]
            indicators['psar_direction'] = 'up' if df['price'].iloc[-1] > indicators['psar'] else 'down'
            
            # Money Flow Index (MFI)
            if 'volume' in df.columns:
                mfi = ta.volume.MFIIndicator(df['price'], df['price'], df['price'], df['volume'], window=14)
                indicators['mfi'] = mfi.money_flow_index().iloc[-1]
            
            # Ichimoku Cloud components
            ichimoku = ta.trend.IchimokuIndicator(df['price'], df['price'])
            indicators['ichimoku_a'] = ichimoku.ichimoku_a().iloc[-1]
            indicators['ichimoku_b'] = ichimoku.ichimoku_b().iloc[-1]
            indicators['ichimoku_base'] = ichimoku.ichimoku_base_line().iloc[-1]
            indicators['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line().iloc[-1]
            
            # Volume Profile (simplified)
            if 'volume' in df.columns:
                indicators.update(self._calculate_volume_profile(df))
            
            # Market Structure
            indicators.update(self._analyze_market_structure(df))
            
            # Momentum indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced indicators for {symbol}: {e}")
        
        return indicators
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate volume profile indicators"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {}
            
            # Volume Weighted Average Price (VWAP)
            vwap = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Volume Rate of Change
            volume_roc = df['volume'].pct_change(periods=10).iloc[-1]
            
            # On Balance Volume (OBV)
            obv = ta.volume.OnBalanceVolumeIndicator(df['price'], df['volume'])
            obv_value = obv.on_balance_volume().iloc[-1]
            
            # Volume oscillator
            volume_short = df['volume'].rolling(window=5).mean().iloc[-1]
            volume_long = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_oscillator = (volume_short - volume_long) / volume_long * 100
            
            return {
                'vwap': vwap.iloc[-1],
                'volume_roc': volume_roc,
                'obv': obv_value,
                'volume_oscillator': volume_oscillator
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure (support/resistance, trends)"""
        try:
            if len(df) < 50:
                return {}
            
            prices = df['price'].values
            
            # Find pivot points (simplified)
            highs, lows = self._find_pivot_points(prices)
            
            # Support and resistance levels
            support_levels = self._find_support_resistance(prices, 'support')
            resistance_levels = self._find_support_resistance(prices, 'resistance')
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(prices)
            
            # Price channels
            upper_channel, lower_channel = self._calculate_price_channels(prices)
            
            return {
                'pivot_highs': len(highs),
                'pivot_lows': len(lows),
                'nearest_support': min(support_levels, key=lambda x: abs(x - prices[-1])) if support_levels else None,
                'nearest_resistance': min(resistance_levels, key=lambda x: abs(x - prices[-1])) if resistance_levels else None,
                'trend_strength': trend_strength,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'channel_position': (prices[-1] - lower_channel) / (upper_channel - lower_channel) if upper_channel != lower_channel else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return {}
    
    def _find_pivot_points(self, prices: np.ndarray, window: int = 5) -> Tuple[List, List]:
        """Find pivot highs and lows"""
        highs = []
        lows = []
        
        for i in range(window, len(prices) - window):
            # Pivot high
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                highs.append((i, prices[i]))
            
            # Pivot low
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                lows.append((i, prices[i]))
        
        return highs, lows
    
    def _find_support_resistance(self, prices: np.ndarray, level_type: str) -> List[float]:
        """Find support and resistance levels using clustering"""
        try:
            if level_type == 'support':
                # Find local minima
                candidates = []
                for i in range(5, len(prices) - 5):
                    if prices[i] == min(prices[i-5:i+6]):
                        candidates.append(prices[i])
            else:
                # Find local maxima
                candidates = []
                for i in range(5, len(prices) - 5):
                    if prices[i] == max(prices[i-5:i+6]):
                        candidates.append(prices[i])
            
            if len(candidates) < 2:
                return []
            
            # Cluster similar levels
            candidates = np.array(candidates).reshape(-1, 1)
            scaler = StandardScaler()
            candidates_scaled = scaler.fit_transform(candidates)
            
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(candidates_scaled)
            
            levels = []
            for label in set(clustering.labels_):
                if label != -1:  # Not noise
                    cluster_points = candidates[clustering.labels_ == label]
                    levels.append(np.mean(cluster_points))
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error finding {level_type} levels: {e}")
            return []
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        try:
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Normalize slope by price level
            normalized_slope = slope / np.mean(prices)
            
            # Combine slope and R-squared for trend strength
            trend_strength = normalized_slope * (r_value ** 2)
            
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_price_channels(self, prices: np.ndarray, window: int = 20) -> Tuple[float, float]:
        """Calculate price channels (Donchian channels)"""
        try:
            if len(prices) < window:
                return prices[-1], prices[-1]
            
            recent_prices = prices[-window:]
            upper_channel = np.max(recent_prices)
            lower_channel = np.min(recent_prices)
            
            return upper_channel, lower_channel
            
        except Exception as e:
            self.logger.error(f"Error calculating price channels: {e}")
            return prices[-1], prices[-1]
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum-based indicators"""
        try:
            if len(df) < 20:
                return {}
            
            prices = df['price']
            
            # Rate of Change (ROC)
            roc_5 = prices.pct_change(periods=5).iloc[-1]
            roc_10 = prices.pct_change(periods=10).iloc[-1]
            roc_20 = prices.pct_change(periods=20).iloc[-1]
            
            # Momentum
            momentum = prices.iloc[-1] - prices.iloc[-10]
            
            # Price acceleration
            if len(prices) >= 3:
                acceleration = (prices.iloc[-1] - prices.iloc[-2]) - (prices.iloc[-2] - prices.iloc[-3])
            else:
                acceleration = 0
            
            # Relative Strength vs Market (simplified - would need market index)
            relative_strength = roc_10  # Placeholder
            
            return {
                'roc_5': roc_5,
                'roc_10': roc_10,
                'roc_20': roc_20,
                'momentum': momentum,
                'acceleration': acceleration,
                'relative_strength': relative_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def detect_chart_patterns(self, symbol: str) -> List[PatternSignal]:
        """Detect chart patterns"""
        if symbol not in self.price_history:
            return []
        
        df = self.price_history[symbol]
        if len(df) < 50:
            return []
        
        patterns = []
        
        try:
            # Double top/bottom
            patterns.extend(self._detect_double_patterns(df))
            
            # Head and shoulders
            patterns.extend(self._detect_head_shoulders(df))
            
            # Triangle patterns
            patterns.extend(self._detect_triangles(df))
            
            # Flag and pennant patterns
            patterns.extend(self._detect_flags_pennants(df))
            
            # Wedge patterns
            patterns.extend(self._detect_wedges(df))
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns for {symbol}: {e}")
        
        return patterns
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        try:
            prices = df['price'].values
            highs, lows = self._find_pivot_points(prices, window=3)
            
            # Double top
            if len(highs) >= 2:
                last_two_highs = highs[-2:]
                price_diff = abs(last_two_highs[0][1] - last_two_highs[1][1])
                avg_price = (last_two_highs[0][1] + last_two_highs[1][1]) / 2
                
                if price_diff / avg_price < 0.02:  # Within 2%
                    patterns.append(PatternSignal(
                        pattern_type="double_top",
                        confidence=0.7,
                        direction="bearish",
                        strength=0.8,
                        metadata={"highs": last_two_highs}
                    ))
            
            # Double bottom
            if len(lows) >= 2:
                last_two_lows = lows[-2:]
                price_diff = abs(last_two_lows[0][1] - last_two_lows[1][1])
                avg_price = (last_two_lows[0][1] + last_two_lows[1][1]) / 2
                
                if price_diff / avg_price < 0.02:  # Within 2%
                    patterns.append(PatternSignal(
                        pattern_type="double_bottom",
                        confidence=0.7,
                        direction="bullish",
                        strength=0.8,
                        metadata={"lows": last_two_lows}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            prices = df['price'].values
            highs, lows = self._find_pivot_points(prices, window=3)
            
            # Head and shoulders (need at least 3 highs)
            if len(highs) >= 3:
                last_three_highs = highs[-3:]
                left_shoulder = last_three_highs[0][1]
                head = last_three_highs[1][1]
                right_shoulder = last_three_highs[2][1]
                
                # Check if middle high is significantly higher
                if (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and
                    abs(left_shoulder - right_shoulder) / ((left_shoulder + right_shoulder) / 2) < 0.05):
                    
                    patterns.append(PatternSignal(
                        pattern_type="head_and_shoulders",
                        confidence=0.8,
                        direction="bearish",
                        strength=0.9,
                        metadata={"shoulders": [left_shoulder, right_shoulder], "head": head}
                    ))
            
            # Inverse head and shoulders
            if len(lows) >= 3:
                last_three_lows = lows[-3:]
                left_shoulder = last_three_lows[0][1]
                head = last_three_lows[1][1]
                right_shoulder = last_three_lows[2][1]
                
                # Check if middle low is significantly lower
                if (head < left_shoulder * 0.98 and head < right_shoulder * 0.98 and
                    abs(left_shoulder - right_shoulder) / ((left_shoulder + right_shoulder) / 2) < 0.05):
                    
                    patterns.append(PatternSignal(
                        pattern_type="inverse_head_and_shoulders",
                        confidence=0.8,
                        direction="bullish",
                        strength=0.9,
                        metadata={"shoulders": [left_shoulder, right_shoulder], "head": head}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            prices = df['price'].values[-30:]  # Last 30 points
            
            # Find trend lines
            highs_trend = self._fit_trend_line(prices, 'highs')
            lows_trend = self._fit_trend_line(prices, 'lows')
            
            if highs_trend and lows_trend:
                high_slope = highs_trend['slope']
                low_slope = lows_trend['slope']
                
                # Ascending triangle
                if abs(high_slope) < 0.001 and low_slope > 0.001:
                    patterns.append(PatternSignal(
                        pattern_type="ascending_triangle",
                        confidence=0.6,
                        direction="bullish",
                        strength=0.7,
                        metadata={"high_slope": high_slope, "low_slope": low_slope}
                    ))
                
                # Descending triangle
                elif abs(low_slope) < 0.001 and high_slope < -0.001:
                    patterns.append(PatternSignal(
                        pattern_type="descending_triangle",
                        confidence=0.6,
                        direction="bearish",
                        strength=0.7,
                        metadata={"high_slope": high_slope, "low_slope": low_slope}
                    ))
                
                # Symmetrical triangle
                elif high_slope < -0.001 and low_slope > 0.001:
                    patterns.append(PatternSignal(
                        pattern_type="symmetrical_triangle",
                        confidence=0.5,
                        direction="neutral",
                        strength=0.6,
                        metadata={"high_slope": high_slope, "low_slope": low_slope}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting triangles: {e}")
        
        return patterns
    
    def _detect_flags_pennants(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            prices = df['price'].values
            
            # Look for strong move followed by consolidation
            recent_change = (prices[-1] - prices[-20]) / prices[-20]
            
            if abs(recent_change) > 0.05:  # 5% move
                # Check for consolidation in last 10 periods
                consolidation_range = np.max(prices[-10:]) - np.min(prices[-10:])
                consolidation_pct = consolidation_range / np.mean(prices[-10:])
                
                if consolidation_pct < 0.03:  # Less than 3% range
                    direction = "bullish" if recent_change > 0 else "bearish"
                    pattern_type = "bull_flag" if recent_change > 0 else "bear_flag"
                    
                    patterns.append(PatternSignal(
                        pattern_type=pattern_type,
                        confidence=0.6,
                        direction=direction,
                        strength=0.7,
                        metadata={"move_pct": recent_change, "consolidation_pct": consolidation_pct}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting flags/pennants: {e}")
        
        return patterns
    
    def _detect_wedges(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect wedge patterns (rising, falling)"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            prices = df['price'].values[-30:]
            
            # Find trend lines for highs and lows
            highs_trend = self._fit_trend_line(prices, 'highs')
            lows_trend = self._fit_trend_line(prices, 'lows')
            
            if highs_trend and lows_trend:
                high_slope = highs_trend['slope']
                low_slope = lows_trend['slope']
                
                # Rising wedge (both slopes positive, but high slope less than low slope)
                if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                    patterns.append(PatternSignal(
                        pattern_type="rising_wedge",
                        confidence=0.6,
                        direction="bearish",
                        strength=0.7,
                        metadata={"high_slope": high_slope, "low_slope": low_slope}
                    ))
                
                # Falling wedge (both slopes negative, but low slope less than high slope)
                elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                    patterns.append(PatternSignal(
                        pattern_type="falling_wedge",
                        confidence=0.6,
                        direction="bullish",
                        strength=0.7,
                        metadata={"high_slope": high_slope, "low_slope": low_slope}
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting wedges: {e}")
        
        return patterns
    
    def _fit_trend_line(self, prices: np.ndarray, line_type: str) -> Optional[Dict]:
        """Fit trend line to highs or lows"""
        try:
            if line_type == 'highs':
                # Find local maxima
                points = []
                for i in range(2, len(prices) - 2):
                    if prices[i] >= prices[i-1] and prices[i] >= prices[i+1]:
                        points.append((i, prices[i]))
            else:
                # Find local minima
                points = []
                for i in range(2, len(prices) - 2):
                    if prices[i] <= prices[i-1] and prices[i] <= prices[i+1]:
                        points.append((i, prices[i]))
            
            if len(points) < 2:
                return None
            
            # Fit line to points
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'points': points
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting trend line: {e}")
            return None
    
    def detect_arbitrage_opportunities(self, symbol: str, exchange_data: Dict[str, Dict]) -> List[Dict]:
        """Detect arbitrage opportunities across exchanges"""
        opportunities = []
        
        try:
            if len(exchange_data) < 2:
                return opportunities
            
            exchanges = list(exchange_data.keys())
            
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange1 = exchanges[i]
                    exchange2 = exchanges[j]
                    
                    data1 = exchange_data[exchange1]
                    data2 = exchange_data[exchange2]
                    
                    if not data1 or not data2:
                        continue
                    
                    # Check bid/ask spread arbitrage
                    if data1.get('bid') and data2.get('ask'):
                        if data1['bid'] > data2['ask']:
                            profit_pct = (data1['bid'] - data2['ask']) / data2['ask']
                            if profit_pct > 0.002:  # 0.2% minimum profit
                                opportunities.append({
                                    'type': 'bid_ask_arbitrage',
                                    'buy_exchange': exchange2,
                                    'sell_exchange': exchange1,
                                    'buy_price': data2['ask'],
                                    'sell_price': data1['bid'],
                                    'profit_pct': profit_pct,
                                    'symbol': symbol
                                })
                    
                    if data2.get('bid') and data1.get('ask'):
                        if data2['bid'] > data1['ask']:
                            profit_pct = (data2['bid'] - data1['ask']) / data1['ask']
                            if profit_pct > 0.002:  # 0.2% minimum profit
                                opportunities.append({
                                    'type': 'bid_ask_arbitrage',
                                    'buy_exchange': exchange1,
                                    'sell_exchange': exchange2,
                                    'buy_price': data1['ask'],
                                    'sell_price': data2['bid'],
                                    'profit_pct': profit_pct,
                                    'symbol': symbol
                                })
            
        except Exception as e:
            self.logger.error(f"Error detecting arbitrage opportunities: {e}")
        
        return opportunities
    
    def calculate_market_sentiment(self, symbol: str) -> Dict:
        """Calculate market sentiment indicators"""
        try:
            if symbol not in self.price_history:
                return {}
            
            df = self.price_history[symbol]
            if len(df) < 20:
                return {}
            
            indicators = self.calculate_advanced_indicators(symbol)
            
            # Sentiment scoring
            sentiment_score = 0
            confidence = 0
            
            # RSI sentiment
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 30:
                    sentiment_score += 1  # Oversold, bullish
                elif rsi > 70:
                    sentiment_score -= 1  # Overbought, bearish
                confidence += 0.2
            
            # Bollinger Bands sentiment
            if 'bb_position' in indicators:
                bb_pos = indicators['bb_position']
                if bb_pos < 0.2:
                    sentiment_score += 0.5  # Near lower band, bullish
                elif bb_pos > 0.8:
                    sentiment_score -= 0.5  # Near upper band, bearish
                confidence += 0.15
            
            # MACD sentiment
            if 'macd' in indicators and 'macd_signal' in indicators:
                if indicators['macd'] > indicators['macd_signal']:
                    sentiment_score += 0.5  # Bullish crossover
                else:
                    sentiment_score -= 0.5  # Bearish crossover
                confidence += 0.15
            
            # Volume sentiment
            if 'volume_oscillator' in indicators:
                vol_osc = indicators['volume_oscillator']
                if vol_osc > 10:
                    sentiment_score += 0.3  # High volume, momentum
                elif vol_osc < -10:
                    sentiment_score -= 0.3  # Low volume, weakness
                confidence += 0.1
            
            # Trend sentiment
            if 'trend_strength' in indicators:
                trend = indicators['trend_strength']
                sentiment_score += np.clip(trend * 10, -1, 1)
                confidence += 0.2
            
            # Pattern sentiment
            patterns = self.detect_chart_patterns(symbol)
            for pattern in patterns:
                if pattern.direction == 'bullish':
                    sentiment_score += pattern.confidence * 0.5
                elif pattern.direction == 'bearish':
                    sentiment_score -= pattern.confidence * 0.5
                confidence += 0.1
            
            # Normalize sentiment score
            if confidence > 0:
                sentiment_score = np.clip(sentiment_score / confidence, -1, 1)
            else:
                sentiment_score = 0
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': min(confidence, 1.0),
                'interpretation': self._interpret_sentiment(sentiment_score),
                'components': {
                    'technical': sentiment_score,
                    'patterns': len([p for p in patterns if p.direction != 'neutral'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market sentiment: {e}")
            return {}
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
        if score > 0.5:
            return "strongly_bullish"
        elif score > 0.2:
            return "bullish"
        elif score > -0.2:
            return "neutral"
        elif score > -0.5:
            return "bearish"
        else:
            return "strongly_bearish"

