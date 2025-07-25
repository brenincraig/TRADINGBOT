"""
Main entry point for the crypto trading bot
"""
import logging
import signal
import sys
import time
from datetime import datetime

from trading_engine import TradingEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class CryptoTradingBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trading_engine = TradingEngine()
        self.running = False
        
    def start(self):
        """Start the trading bot"""
        self.logger.info("Starting Crypto Trading Bot...")
        self.logger.info(f"Trading pairs: {Config.TRADING_PAIRS}")
        self.logger.info(f"Max position size: {Config.MAX_POSITION_SIZE * 100}%")
        self.logger.info(f"Daily loss limit: {Config.MAX_DAILY_LOSS * 100}%")
        
        # Start the trading engine
        if self.trading_engine.start():
            self.running = True
            self.logger.info("Trading bot started successfully")
            
            # Keep the main thread alive
            try:
                while self.running:
                    time.sleep(10)
                    
                    # Print status every 5 minutes
                    if int(time.time()) % 300 == 0:
                        status = self.trading_engine.get_status()
                        self.logger.info(f"Status: {status}")
                        
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.stop()
        else:
            self.logger.error("Failed to start trading engine")
            return False
        
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping Crypto Trading Bot...")
        self.running = False
        self.trading_engine.stop()
        self.logger.info("Trading bot stopped")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

def main():
    """Main function"""
    bot = CryptoTradingBot()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, bot.signal_handler)
    signal.signal(signal.SIGTERM, bot.signal_handler)
    
    try:
        bot.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

