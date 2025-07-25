"""
Configuration file for the crypto trading bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Exchange API Keys (set these in .env file)
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
    COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY', '')
    COINBASE_PASSPHRASE = os.getenv('COINBASE_PASSPHRASE', '')
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_SECRET_KEY = os.getenv('KRAKEN_SECRET_KEY', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/crypto_trading')
    
    # Trading Configuration
    TRADING_PAIRS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT', 'XRP/USDT', 'MATIC/USDT'
    ]
    
    # Risk Management
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio per position
    MAX_DAILY_LOSS = 0.05    # 5% maximum daily loss
    STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.01  # 1% take profit
    MAX_SLIPPAGE = 0.005     # 0.5% maximum slippage
    
    # Trading Strategy
    MIN_PROFIT_THRESHOLD = 0.001  # 0.1% minimum profit
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average volume
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    # Notification Settings
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    
    # System Settings
    LOOP_INTERVAL = 1  # seconds between main loop iterations
    DATA_RETENTION_DAYS = 30
    LOG_LEVEL = 'INFO'
    
    # Exchange Settings
    EXCHANGES = ['binance', 'coinbase', 'kraken']
    DEFAULT_EXCHANGE = 'binance'
    
    # WebSocket Settings
    WEBSOCKET_TIMEOUT = 30
    RECONNECT_ATTEMPTS = 5
    
    # Performance Monitoring
    PERFORMANCE_UPDATE_INTERVAL = 300  # 5 minutes
    ABNORMAL_VOLATILITY_THRESHOLD = 0.1  # 10% price change in 1 minute
    
    # Auto-withdrawal settings
    AUTO_WITHDRAW_THRESHOLD = 10000  # USDT
    COLD_WALLET_ADDRESS = os.getenv('COLD_WALLET_ADDRESS', '')

