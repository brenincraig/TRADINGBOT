"""
Exchange Manager for handling multiple crypto exchange APIs
"""
import ccxt
import asyncio
import websocket
import json
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from config import Config

class ExchangeManager:
    def __init__(self):
        self.exchanges = {}
        self.websockets = {}
        self.callbacks = {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        
    def initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Binance
            if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': Config.BINANCE_API_KEY,
                    'secret': Config.BINANCE_SECRET_KEY,
                    'sandbox': False,  # Set to True for testing
                    'enableRateLimit': True,
                })
                self.logger.info("Binance exchange initialized")
            
            # Initialize Coinbase
            if Config.COINBASE_API_KEY and Config.COINBASE_SECRET_KEY:
                self.exchanges['coinbase'] = ccxt.coinbasepro({
                    'apiKey': Config.COINBASE_API_KEY,
                    'secret': Config.COINBASE_SECRET_KEY,
                    'passphrase': Config.COINBASE_PASSPHRASE,
                    'sandbox': False,  # Set to True for testing
                    'enableRateLimit': True,
                })
                self.logger.info("Coinbase exchange initialized")
            
            # Initialize Kraken
            if Config.KRAKEN_API_KEY and Config.KRAKEN_SECRET_KEY:
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': Config.KRAKEN_API_KEY,
                    'secret': Config.KRAKEN_SECRET_KEY,
                    'enableRateLimit': True,
                })
                self.logger.info("Kraken exchange initialized")
            
            return len(self.exchanges) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            return False
    
    def get_exchange(self, exchange_name: str):
        """Get exchange instance"""
        return self.exchanges.get(exchange_name.lower())
    
    def get_ticker(self, symbol: str, exchange_name: str = None) -> Optional[Dict]:
        """Get ticker data for a symbol"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                ticker = exchange.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'exchange': exchange_name,
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'timestamp': datetime.utcnow()
                }
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol} on {exchange_name}: {e}")
        return None
    
    def get_order_book(self, symbol: str, exchange_name: str = None, limit: int = 100) -> Optional[Dict]:
        """Get order book data"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                order_book = exchange.fetch_order_book(symbol, limit)
                return {
                    'symbol': symbol,
                    'exchange': exchange_name,
                    'bids': order_book['bids'],
                    'asks': order_book['asks'],
                    'timestamp': datetime.utcnow()
                }
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol} on {exchange_name}: {e}")
        return None
    
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, 
                   price: float = None, exchange_name: str = None) -> Optional[Dict]:
        """Place an order"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if not exchange:
                return None
            
            if order_type.lower() == 'market':
                order = exchange.create_market_order(symbol, side, amount)
            elif order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            self.logger.info(f"Order placed: {symbol} {side} {amount} @ {price or 'market'}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str, exchange_name: str = None) -> bool:
        """Cancel an order"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                exchange.cancel_order(order_id, symbol)
                self.logger.info(f"Order cancelled: {order_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
        return False
    
    def get_balance(self, exchange_name: str = None) -> Optional[Dict]:
        """Get account balance"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                balance = exchange.fetch_balance()
                return balance
        except Exception as e:
            self.logger.error(f"Failed to get balance from {exchange_name}: {e}")
        return None
    
    def get_open_orders(self, symbol: str = None, exchange_name: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                orders = exchange.fetch_open_orders(symbol)
                return orders
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
        return []
    
    def get_trade_history(self, symbol: str = None, limit: int = 100, 
                         exchange_name: str = None) -> List[Dict]:
        """Get trade history"""
        try:
            exchange_name = exchange_name or Config.DEFAULT_EXCHANGE
            exchange = self.get_exchange(exchange_name)
            if exchange:
                trades = exchange.fetch_my_trades(symbol, limit=limit)
                return trades
        except Exception as e:
            self.logger.error(f"Failed to get trade history: {e}")
        return []
    
    def start_websocket_feeds(self, symbols: List[str], callback: Callable):
        """Start WebSocket feeds for real-time data"""
        self.running = True
        self.callbacks['ticker'] = callback
        
        for exchange_name in self.exchanges.keys():
            if exchange_name == 'binance':
                self._start_binance_websocket(symbols)
            elif exchange_name == 'coinbase':
                self._start_coinbase_websocket(symbols)
            elif exchange_name == 'kraken':
                self._start_kraken_websocket(symbols)
    
    def _start_binance_websocket(self, symbols: List[str]):
        """Start Binance WebSocket connection"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 's' in data:  # Ticker data
                    ticker_data = {
                        'symbol': data['s'],
                        'exchange': 'binance',
                        'price': float(data['c']),
                        'volume': float(data['v']),
                        'timestamp': datetime.utcnow()
                    }
                    if self.callbacks.get('ticker'):
                        self.callbacks['ticker'](ticker_data)
            except Exception as e:
                self.logger.error(f"Binance WebSocket message error: {e}")
        
        def on_error(ws, error):
            self.logger.error(f"Binance WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning("Binance WebSocket connection closed")
            if self.running:
                time.sleep(5)
                self._start_binance_websocket(symbols)
        
        # Convert symbols to Binance format
        binance_symbols = [s.replace('/', '').lower() for s in symbols]
        streams = [f"{symbol}@ticker" for symbol in binance_symbols]
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        ws = websocket.WebSocketApp(url,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        def run_websocket():
            ws.run_forever()
        
        thread = threading.Thread(target=run_websocket)
        thread.daemon = True
        thread.start()
        
        self.websockets['binance'] = ws
    
    def _start_coinbase_websocket(self, symbols: List[str]):
        """Start Coinbase WebSocket connection"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('type') == 'ticker':
                    ticker_data = {
                        'symbol': data['product_id'].replace('-', '/'),
                        'exchange': 'coinbase',
                        'price': float(data['price']),
                        'volume': float(data['volume_24h']),
                        'timestamp': datetime.utcnow()
                    }
                    if self.callbacks.get('ticker'):
                        self.callbacks['ticker'](ticker_data)
            except Exception as e:
                self.logger.error(f"Coinbase WebSocket message error: {e}")
        
        def on_open(ws):
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [s.replace('/', '-') for s in symbols],
                "channels": ["ticker"]
            }
            ws.send(json.dumps(subscribe_message))
        
        def on_error(ws, error):
            self.logger.error(f"Coinbase WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning("Coinbase WebSocket connection closed")
            if self.running:
                time.sleep(5)
                self._start_coinbase_websocket(symbols)
        
        ws = websocket.WebSocketApp("wss://ws-feed.exchange.coinbase.com",
                                  on_message=on_message,
                                  on_open=on_open,
                                  on_error=on_error,
                                  on_close=on_close)
        
        def run_websocket():
            ws.run_forever()
        
        thread = threading.Thread(target=run_websocket)
        thread.daemon = True
        thread.start()
        
        self.websockets['coinbase'] = ws
    
    def _start_kraken_websocket(self, symbols: List[str]):
        """Start Kraken WebSocket connection"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, list) and len(data) > 1:
                    if 'ticker' in str(data):
                        # Kraken ticker format is complex, simplified here
                        pass
            except Exception as e:
                self.logger.error(f"Kraken WebSocket message error: {e}")
        
        def on_open(ws):
            subscribe_message = {
                "event": "subscribe",
                "pair": symbols,
                "subscription": {"name": "ticker"}
            }
            ws.send(json.dumps(subscribe_message))
        
        def on_error(ws, error):
            self.logger.error(f"Kraken WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning("Kraken WebSocket connection closed")
            if self.running:
                time.sleep(5)
                self._start_kraken_websocket(symbols)
        
        ws = websocket.WebSocketApp("wss://ws.kraken.com",
                                  on_message=on_message,
                                  on_open=on_open,
                                  on_error=on_error,
                                  on_close=on_close)
        
        def run_websocket():
            ws.run_forever()
        
        thread = threading.Thread(target=run_websocket)
        thread.daemon = True
        thread.start()
        
        self.websockets['kraken'] = ws
    
    def stop_websocket_feeds(self):
        """Stop all WebSocket feeds"""
        self.running = False
        for ws in self.websockets.values():
            if ws:
                ws.close()
        self.websockets.clear()
        self.logger.info("All WebSocket feeds stopped")
    
    def calculate_slippage(self, symbol: str, side: str, amount: float, 
                          exchange_name: str = None) -> float:
        """Calculate estimated slippage for an order"""
        try:
            order_book = self.get_order_book(symbol, exchange_name)
            if not order_book:
                return 0.0
            
            if side.lower() == 'buy':
                asks = order_book['asks']
                total_cost = 0
                total_amount = 0
                
                for price, qty in asks:
                    if total_amount >= amount:
                        break
                    take_qty = min(qty, amount - total_amount)
                    total_cost += price * take_qty
                    total_amount += take_qty
                
                if total_amount > 0:
                    avg_price = total_cost / total_amount
                    best_price = asks[0][0]
                    slippage = (avg_price - best_price) / best_price
                    return slippage
            
            else:  # sell
                bids = order_book['bids']
                total_value = 0
                total_amount = 0
                
                for price, qty in bids:
                    if total_amount >= amount:
                        break
                    take_qty = min(qty, amount - total_amount)
                    total_value += price * take_qty
                    total_amount += take_qty
                
                if total_amount > 0:
                    avg_price = total_value / total_amount
                    best_price = bids[0][0]
                    slippage = (best_price - avg_price) / best_price
                    return slippage
            
        except Exception as e:
            self.logger.error(f"Failed to calculate slippage: {e}")
        
        return 0.0

