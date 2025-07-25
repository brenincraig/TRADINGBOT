"""
Arbitrage Detection Module for Cross-Exchange and Statistical Arbitrage
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from collections import defaultdict, deque
import itertools

@dataclass
class ArbitrageOpportunity:
    opportunity_type: str  # 'cross_exchange', 'triangular', 'statistical'
    symbol: str
    exchanges: List[str]
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_absolute: float
    volume_limit: float
    estimated_fees: float
    net_profit: float
    confidence: float
    timestamp: datetime
    metadata: Dict

class ArbitrageDetector:
    def __init__(self, exchange_manager):
        self.logger = logging.getLogger(__name__)
        self.exchange_manager = exchange_manager
        
        # Price data storage
        self.price_data = {}  # exchange -> symbol -> price data
        self.order_books = {}  # exchange -> symbol -> order book
        self.trade_history = {}  # exchange -> symbol -> recent trades
        
        # Arbitrage tracking
        self.active_opportunities = []
        self.opportunity_history = deque(maxlen=1000)
        
        # Configuration
        self.min_profit_threshold = 0.002  # 0.2% minimum profit
        self.max_position_size = 1000  # Maximum position size in USDT
        self.fee_estimates = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'coinbase': {'maker': 0.005, 'taker': 0.005},
            'kraken': {'maker': 0.0016, 'taker': 0.0026}
        }
        
        # Statistical arbitrage parameters
        self.lookback_period = 100
        self.zscore_threshold = 2.0
        self.correlation_threshold = 0.7
        
    def update_market_data(self, exchange: str, symbol: str, data: Dict):
        """Update market data for arbitrage detection"""
        try:
            if exchange not in self.price_data:
                self.price_data[exchange] = {}
            if symbol not in self.price_data[exchange]:
                self.price_data[exchange][symbol] = deque(maxlen=1000)
            
            self.price_data[exchange][symbol].append({
                'timestamp': data['timestamp'],
                'price': data['price'],
                'volume': data.get('volume', 0),
                'bid': data.get('bid'),
                'ask': data.get('ask')
            })
            
        except Exception as e:
            self.logger.error(f"Error updating market data for {exchange} {symbol}: {e}")
    
    def update_order_book(self, exchange: str, symbol: str, order_book: Dict):
        """Update order book data"""
        try:
            if exchange not in self.order_books:
                self.order_books[exchange] = {}
            
            self.order_books[exchange][symbol] = {
                'timestamp': datetime.utcnow(),
                'bids': order_book['bids'][:20],  # Top 20 levels
                'asks': order_book['asks'][:20],
                'spread': order_book['asks'][0][0] - order_book['bids'][0][0] if order_book['bids'] and order_book['asks'] else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating order book for {exchange} {symbol}: {e}")
    
    def detect_arbitrage_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Main method to detect all types of arbitrage opportunities"""
        opportunities = []
        
        try:
            # Cross-exchange arbitrage
            for symbol in symbols:
                opportunities.extend(self._detect_cross_exchange_arbitrage(symbol))
            
            # Triangular arbitrage
            opportunities.extend(self._detect_triangular_arbitrage())
            
            # Statistical arbitrage
            opportunities.extend(self._detect_statistical_arbitrage(symbols))
            
            # Filter and validate opportunities
            opportunities = self._filter_opportunities(opportunities)
            
            # Update tracking
            self.active_opportunities = opportunities
            for opp in opportunities:
                self.opportunity_history.append(opp)
            
        except Exception as e:
            self.logger.error(f"Error detecting arbitrage opportunities: {e}")
        
        return opportunities
    
    def _detect_cross_exchange_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Detect cross-exchange arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get current prices from all exchanges
            exchange_prices = {}
            for exchange in self.exchange_manager.exchanges.keys():
                if (exchange in self.price_data and 
                    symbol in self.price_data[exchange] and 
                    self.price_data[exchange][symbol]):
                    
                    latest_data = self.price_data[exchange][symbol][-1]
                    exchange_prices[exchange] = {
                        'price': latest_data['price'],
                        'bid': latest_data.get('bid'),
                        'ask': latest_data.get('ask'),
                        'timestamp': latest_data['timestamp']
                    }
            
            if len(exchange_prices) < 2:
                return opportunities
            
            # Find arbitrage opportunities between exchange pairs
            for exchange1, exchange2 in itertools.combinations(exchange_prices.keys(), 2):
                data1 = exchange_prices[exchange1]
                data2 = exchange_prices[exchange2]
                
                # Skip if data is too old (more than 10 seconds)
                if (datetime.utcnow() - data1['timestamp']).seconds > 10 or \
                   (datetime.utcnow() - data2['timestamp']).seconds > 10:
                    continue
                
                # Check bid/ask arbitrage (more precise than using last price)
                if data1.get('bid') and data2.get('ask'):
                    if data1['bid'] > data2['ask']:
                        profit_abs = data1['bid'] - data2['ask']
                        profit_pct = profit_abs / data2['ask']
                        
                        if profit_pct > self.min_profit_threshold:
                            # Calculate volume limit based on order book depth
                            volume_limit = self._calculate_volume_limit(exchange1, exchange2, symbol)
                            
                            # Estimate fees
                            fees = self._estimate_fees(exchange1, exchange2, data2['ask'], volume_limit)
                            net_profit = profit_abs - fees
                            net_profit_pct = net_profit / data2['ask']
                            
                            if net_profit_pct > 0:
                                opportunities.append(ArbitrageOpportunity(
                                    opportunity_type='cross_exchange',
                                    symbol=symbol,
                                    exchanges=[exchange1, exchange2],
                                    buy_exchange=exchange2,
                                    sell_exchange=exchange1,
                                    buy_price=data2['ask'],
                                    sell_price=data1['bid'],
                                    profit_percentage=profit_pct,
                                    profit_absolute=profit_abs,
                                    volume_limit=volume_limit,
                                    estimated_fees=fees,
                                    net_profit=net_profit,
                                    confidence=self._calculate_arbitrage_confidence(data1, data2),
                                    timestamp=datetime.utcnow(),
                                    metadata={
                                        'spread1': data1.get('bid', 0) - data1.get('ask', 0),
                                        'spread2': data2.get('bid', 0) - data2.get('ask', 0)
                                    }
                                ))
                
                # Check the reverse direction
                if data2.get('bid') and data1.get('ask'):
                    if data2['bid'] > data1['ask']:
                        profit_abs = data2['bid'] - data1['ask']
                        profit_pct = profit_abs / data1['ask']
                        
                        if profit_pct > self.min_profit_threshold:
                            volume_limit = self._calculate_volume_limit(exchange2, exchange1, symbol)
                            fees = self._estimate_fees(exchange2, exchange1, data1['ask'], volume_limit)
                            net_profit = profit_abs - fees
                            net_profit_pct = net_profit / data1['ask']
                            
                            if net_profit_pct > 0:
                                opportunities.append(ArbitrageOpportunity(
                                    opportunity_type='cross_exchange',
                                    symbol=symbol,
                                    exchanges=[exchange2, exchange1],
                                    buy_exchange=exchange1,
                                    sell_exchange=exchange2,
                                    buy_price=data1['ask'],
                                    sell_price=data2['bid'],
                                    profit_percentage=profit_pct,
                                    profit_absolute=profit_abs,
                                    volume_limit=volume_limit,
                                    estimated_fees=fees,
                                    net_profit=net_profit,
                                    confidence=self._calculate_arbitrage_confidence(data2, data1),
                                    timestamp=datetime.utcnow(),
                                    metadata={
                                        'spread1': data1.get('bid', 0) - data1.get('ask', 0),
                                        'spread2': data2.get('bid', 0) - data2.get('ask', 0)
                                    }
                                ))
            
        except Exception as e:
            self.logger.error(f"Error detecting cross-exchange arbitrage for {symbol}: {e}")
        
        return opportunities
    
    def _detect_triangular_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage opportunities"""
        opportunities = []
        
        try:
            # Common triangular arbitrage paths
            triangular_paths = [
                ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
                ['BTC/USDT', 'BNB/BTC', 'BNB/USDT'],
                ['ETH/USDT', 'BNB/ETH', 'BNB/USDT'],
                ['BTC/USDT', 'ADA/BTC', 'ADA/USDT'],
                ['ETH/USDT', 'ADA/ETH', 'ADA/USDT']
            ]
            
            for exchange in self.exchange_manager.exchanges.keys():
                for path in triangular_paths:
                    opp = self._check_triangular_path(exchange, path)
                    if opp:
                        opportunities.append(opp)
            
        except Exception as e:
            self.logger.error(f"Error detecting triangular arbitrage: {e}")
        
        return opportunities
    
    def _check_triangular_path(self, exchange: str, path: List[str]) -> Optional[ArbitrageOpportunity]:
        """Check a specific triangular arbitrage path"""
        try:
            if exchange not in self.price_data:
                return None
            
            # Get current prices for all pairs in the path
            prices = {}
            for symbol in path:
                if (symbol in self.price_data[exchange] and 
                    self.price_data[exchange][symbol]):
                    latest_data = self.price_data[exchange][symbol][-1]
                    prices[symbol] = {
                        'bid': latest_data.get('bid'),
                        'ask': latest_data.get('ask'),
                        'timestamp': latest_data['timestamp']
                    }
                else:
                    return None  # Missing price data
            
            # Check if all data is recent
            for symbol, data in prices.items():
                if (datetime.utcnow() - data['timestamp']).seconds > 30:
                    return None
            
            # Calculate triangular arbitrage profit
            # Example: BTC/USDT -> ETH/BTC -> ETH/USDT
            # Start with 1000 USDT
            start_amount = 1000
            
            # Path 1: USDT -> BTC -> ETH -> USDT
            step1 = start_amount / prices[path[0]]['ask']  # Buy BTC with USDT
            step2 = step1 / prices[path[1]]['ask']        # Buy ETH with BTC
            final1 = step2 * prices[path[2]]['bid']       # Sell ETH for USDT
            
            profit1 = final1 - start_amount
            profit_pct1 = profit1 / start_amount
            
            # Path 2: USDT -> ETH -> BTC -> USDT (reverse)
            step1_rev = start_amount / prices[path[2]]['ask']  # Buy ETH with USDT
            step2_rev = step1_rev * prices[path[1]]['bid']     # Sell ETH for BTC
            final2 = step2_rev * prices[path[0]]['bid']       # Sell BTC for USDT
            
            profit2 = final2 - start_amount
            profit_pct2 = profit2 / start_amount
            
            # Choose the more profitable path
            if profit_pct1 > profit_pct2 and profit_pct1 > self.min_profit_threshold:
                # Estimate fees for 3 trades
                total_fees = start_amount * 0.003  # Approximate 0.1% per trade
                net_profit = profit1 - total_fees
                
                if net_profit > 0:
                    return ArbitrageOpportunity(
                        opportunity_type='triangular',
                        symbol=f"{path[0]}-{path[1]}-{path[2]}",
                        exchanges=[exchange],
                        buy_exchange=exchange,
                        sell_exchange=exchange,
                        buy_price=0,  # Complex path
                        sell_price=0,  # Complex path
                        profit_percentage=profit_pct1,
                        profit_absolute=profit1,
                        volume_limit=min(start_amount, self.max_position_size),
                        estimated_fees=total_fees,
                        net_profit=net_profit,
                        confidence=0.7,  # Lower confidence for triangular
                        timestamp=datetime.utcnow(),
                        metadata={
                            'path': path,
                            'direction': 'forward',
                            'steps': [step1, step2, final1]
                        }
                    )
            
            elif profit_pct2 > self.min_profit_threshold:
                total_fees = start_amount * 0.003
                net_profit = profit2 - total_fees
                
                if net_profit > 0:
                    return ArbitrageOpportunity(
                        opportunity_type='triangular',
                        symbol=f"{path[2]}-{path[1]}-{path[0]}",
                        exchanges=[exchange],
                        buy_exchange=exchange,
                        sell_exchange=exchange,
                        buy_price=0,
                        sell_price=0,
                        profit_percentage=profit_pct2,
                        profit_absolute=profit2,
                        volume_limit=min(start_amount, self.max_position_size),
                        estimated_fees=total_fees,
                        net_profit=net_profit,
                        confidence=0.7,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'path': list(reversed(path)),
                            'direction': 'reverse',
                            'steps': [step1_rev, step2_rev, final2]
                        }
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking triangular path {path} on {exchange}: {e}")
        
        return None
    
    def _detect_statistical_arbitrage(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Detect statistical arbitrage opportunities"""
        opportunities = []
        
        try:
            # Find correlated pairs
            correlated_pairs = self._find_correlated_pairs(symbols)
            
            for pair in correlated_pairs:
                symbol1, symbol2, correlation = pair
                
                # Calculate spread and z-score
                spread_data = self._calculate_spread(symbol1, symbol2)
                if spread_data:
                    z_score = spread_data['z_score']
                    current_spread = spread_data['current_spread']
                    mean_spread = spread_data['mean_spread']
                    
                    # Statistical arbitrage signals
                    if abs(z_score) > self.zscore_threshold:
                        direction = 'long_short' if z_score > 0 else 'short_long'
                        
                        # Estimate profit potential
                        expected_reversion = mean_spread - current_spread
                        profit_estimate = abs(expected_reversion) * 100  # Simplified
                        
                        opportunities.append(ArbitrageOpportunity(
                            opportunity_type='statistical',
                            symbol=f"{symbol1}/{symbol2}",
                            exchanges=['statistical'],
                            buy_exchange='statistical',
                            sell_exchange='statistical',
                            buy_price=0,
                            sell_price=0,
                            profit_percentage=profit_estimate / 10000,  # Rough estimate
                            profit_absolute=profit_estimate,
                            volume_limit=self.max_position_size,
                            estimated_fees=20,  # Estimated fees for pair trade
                            net_profit=profit_estimate - 20,
                            confidence=min(abs(correlation), 0.9),
                            timestamp=datetime.utcnow(),
                            metadata={
                                'z_score': z_score,
                                'correlation': correlation,
                                'spread': current_spread,
                                'mean_spread': mean_spread,
                                'direction': direction
                            }
                        ))
            
        except Exception as e:
            self.logger.error(f"Error detecting statistical arbitrage: {e}")
        
        return opportunities
    
    def _find_correlated_pairs(self, symbols: List[str]) -> List[Tuple[str, str, float]]:
        """Find correlated symbol pairs for statistical arbitrage"""
        correlated_pairs = []
        
        try:
            # Get price series for all symbols
            price_series = {}
            for symbol in symbols:
                for exchange in self.price_data:
                    if symbol in self.price_data[exchange] and len(self.price_data[exchange][symbol]) >= self.lookback_period:
                        prices = [p['price'] for p in list(self.price_data[exchange][symbol])[-self.lookback_period:]]
                        price_series[symbol] = prices
                        break
            
            # Calculate correlations between all pairs
            for symbol1, symbol2 in itertools.combinations(price_series.keys(), 2):
                if len(price_series[symbol1]) == len(price_series[symbol2]):
                    correlation = np.corrcoef(price_series[symbol1], price_series[symbol2])[0, 1]
                    
                    if abs(correlation) > self.correlation_threshold:
                        correlated_pairs.append((symbol1, symbol2, correlation))
            
        except Exception as e:
            self.logger.error(f"Error finding correlated pairs: {e}")
        
        return correlated_pairs
    
    def _calculate_spread(self, symbol1: str, symbol2: str) -> Optional[Dict]:
        """Calculate spread between two symbols"""
        try:
            # Get price series for both symbols
            prices1 = None
            prices2 = None
            
            for exchange in self.price_data:
                if symbol1 in self.price_data[exchange] and len(self.price_data[exchange][symbol1]) >= self.lookback_period:
                    prices1 = [p['price'] for p in list(self.price_data[exchange][symbol1])[-self.lookback_period:]]
                if symbol2 in self.price_data[exchange] and len(self.price_data[exchange][symbol2]) >= self.lookback_period:
                    prices2 = [p['price'] for p in list(self.price_data[exchange][symbol2])[-self.lookback_period:]]
            
            if prices1 is None or prices2 is None or len(prices1) != len(prices2):
                return None
            
            # Calculate spread (ratio or difference)
            spreads = [p1 / p2 for p1, p2 in zip(prices1, prices2)]  # Ratio spread
            
            # Calculate statistics
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            current_spread = spreads[-1]
            
            # Calculate z-score
            z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
            
            return {
                'current_spread': current_spread,
                'mean_spread': mean_spread,
                'std_spread': std_spread,
                'z_score': z_score,
                'spreads': spreads
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spread for {symbol1}/{symbol2}: {e}")
            return None
    
    def _calculate_volume_limit(self, buy_exchange: str, sell_exchange: str, symbol: str) -> float:
        """Calculate volume limit based on order book depth"""
        try:
            volume_limit = self.max_position_size
            
            # Check order book depth on both exchanges
            if (buy_exchange in self.order_books and 
                symbol in self.order_books[buy_exchange]):
                
                buy_book = self.order_books[buy_exchange][symbol]
                if buy_book['asks']:
                    # Calculate volume available at reasonable price levels
                    ask_volume = sum([ask[1] for ask in buy_book['asks'][:5]])  # Top 5 levels
                    volume_limit = min(volume_limit, ask_volume * buy_book['asks'][0][0])
            
            if (sell_exchange in self.order_books and 
                symbol in self.order_books[sell_exchange]):
                
                sell_book = self.order_books[sell_exchange][symbol]
                if sell_book['bids']:
                    bid_volume = sum([bid[1] for bid in sell_book['bids'][:5]])
                    volume_limit = min(volume_limit, bid_volume * sell_book['bids'][0][0])
            
            return max(volume_limit, 10)  # Minimum $10
            
        except Exception as e:
            self.logger.error(f"Error calculating volume limit: {e}")
            return 100  # Default fallback
    
    def _estimate_fees(self, buy_exchange: str, sell_exchange: str, price: float, volume: float) -> float:
        """Estimate trading fees for arbitrage trade"""
        try:
            buy_fee_rate = self.fee_estimates.get(buy_exchange, {}).get('taker', 0.001)
            sell_fee_rate = self.fee_estimates.get(sell_exchange, {}).get('taker', 0.001)
            
            trade_value = price * volume
            total_fees = trade_value * (buy_fee_rate + sell_fee_rate)
            
            return total_fees
            
        except Exception as e:
            self.logger.error(f"Error estimating fees: {e}")
            return price * volume * 0.002  # Default 0.2% total fees
    
    def _calculate_arbitrage_confidence(self, data1: Dict, data2: Dict) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        try:
            confidence = 0.5  # Base confidence
            
            # Higher confidence for tighter spreads
            spread1 = abs(data1.get('bid', 0) - data1.get('ask', 0))
            spread2 = abs(data2.get('bid', 0) - data2.get('ask', 0))
            
            if spread1 > 0 and spread2 > 0:
                avg_spread = (spread1 + spread2) / 2
                avg_price = (data1['price'] + data2['price']) / 2
                spread_ratio = avg_spread / avg_price
                
                if spread_ratio < 0.001:  # Very tight spreads
                    confidence += 0.3
                elif spread_ratio < 0.005:  # Reasonable spreads
                    confidence += 0.2
            
            # Higher confidence for recent data
            time_diff1 = (datetime.utcnow() - data1['timestamp']).seconds
            time_diff2 = (datetime.utcnow() - data2['timestamp']).seconds
            
            if max(time_diff1, time_diff2) < 5:  # Very recent data
                confidence += 0.2
            elif max(time_diff1, time_diff2) < 15:  # Recent data
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage confidence: {e}")
            return 0.5
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter and validate arbitrage opportunities"""
        filtered = []
        
        try:
            for opp in opportunities:
                # Minimum profit threshold
                if opp.net_profit <= 0:
                    continue
                
                # Minimum confidence
                if opp.confidence < 0.3:
                    continue
                
                # Check for duplicate opportunities
                is_duplicate = False
                for existing in filtered:
                    if (existing.symbol == opp.symbol and 
                        existing.opportunity_type == opp.opportunity_type and
                        existing.buy_exchange == opp.buy_exchange and
                        existing.sell_exchange == opp.sell_exchange):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(opp)
            
            # Sort by net profit descending
            filtered.sort(key=lambda x: x.net_profit, reverse=True)
            
            # Limit to top opportunities
            return filtered[:10]
            
        except Exception as e:
            self.logger.error(f"Error filtering opportunities: {e}")
            return opportunities
    
    def get_opportunity_statistics(self) -> Dict:
        """Get statistics about arbitrage opportunities"""
        try:
            if not self.opportunity_history:
                return {}
            
            recent_opportunities = [opp for opp in self.opportunity_history 
                                 if opp.timestamp > datetime.utcnow() - timedelta(hours=1)]
            
            if not recent_opportunities:
                return {}
            
            stats = {
                'total_opportunities': len(recent_opportunities),
                'avg_profit_pct': np.mean([opp.profit_percentage for opp in recent_opportunities]),
                'max_profit_pct': max([opp.profit_percentage for opp in recent_opportunities]),
                'avg_confidence': np.mean([opp.confidence for opp in recent_opportunities]),
                'opportunity_types': {},
                'exchange_pairs': {}
            }
            
            # Count by type
            for opp in recent_opportunities:
                opp_type = opp.opportunity_type
                if opp_type not in stats['opportunity_types']:
                    stats['opportunity_types'][opp_type] = 0
                stats['opportunity_types'][opp_type] += 1
            
            # Count by exchange pairs
            for opp in recent_opportunities:
                if opp.opportunity_type == 'cross_exchange':
                    pair = f"{opp.buy_exchange}-{opp.sell_exchange}"
                    if pair not in stats['exchange_pairs']:
                        stats['exchange_pairs'][pair] = 0
                    stats['exchange_pairs'][pair] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting opportunity statistics: {e}")
            return {}

