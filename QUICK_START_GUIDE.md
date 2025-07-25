# Quick Start Guide - Advanced Crypto Trading Bot

**Get your automated crypto trading bot running in under 10 minutes!**

## Prerequisites

- Linux server (Ubuntu 20.04+ recommended) with 4GB+ RAM
- Docker and Docker Compose installed
- Exchange API credentials (Binance, Coinbase, or Kraken)
- Basic command line knowledge

## 1. Download and Setup

```bash
# Clone the repository
git clone <repository-url>
cd crypto_trading_bot

# Copy environment template
cp .env.example .env
```

## 2. Configure Your Settings

Edit the `.env` file with your settings:

```bash
# Basic Configuration
INITIAL_CAPITAL=10000
PAPER_TRADING=true  # Set to false for live trading

# Exchange API Keys (get from your exchange)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Risk Management (recommended defaults)
MAX_DAILY_LOSS_PCT=0.05
MAX_POSITION_SIZE_PCT=0.03
STOP_LOSS_PCT=0.02

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 3. Start the System

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-bot
```

## 4. Access the Dashboard

Open your browser and go to:
- **Dashboard:** http://localhost:5000
- **Monitoring:** http://localhost:3000 (Grafana)

## 5. Enable Live Trading

Once you've tested in paper trading mode:

1. Set `PAPER_TRADING=false` in your `.env` file
2. Restart the system: `docker-compose restart`
3. Monitor your first live trades in the dashboard

## 6. Production Deployment (Optional)

For cloud deployment with SSL and monitoring:

```bash
# Run the automated deployment script
chmod +x deploy.sh
./deploy.sh --domain your-domain.com --enable-monitoring
```

## Key Features Enabled

✅ **Multi-Exchange Trading:** Binance, Coinbase, Kraken  
✅ **8 Trading Strategies:** Scalping, momentum, mean reversion, breakout, arbitrage, volume, pattern, multi-timeframe  
✅ **50+ Technical Indicators:** RSI, MACD, Bollinger Bands, and more  
✅ **Advanced Risk Management:** Stop-loss, take-profit, position sizing, drawdown protection  
✅ **Real-time Notifications:** Telegram, Discord, email alerts  
✅ **Adaptive Learning:** AI-powered strategy optimization  
✅ **Professional Dashboard:** Real-time monitoring and control  

## Default Trading Pairs

The system starts with these popular pairs:
- BTC/USDT, ETH/USDT, BNB/USDT
- ADA/USDT, DOT/USDT, LINK/USDT
- PEPE/USDT, DOGE/USDT, SHIB/USDT

## Safety Features

🛡️ **Paper Trading Mode:** Test without risking real money  
🛡️ **Daily Loss Limits:** Automatic trading halt at 5% daily loss  
🛡️ **Position Size Limits:** Maximum 3% of capital per trade  
🛡️ **Emergency Stop:** One-click halt all trading  
🛡️ **Circuit Breakers:** Automatic protection during market crashes  

## Getting Help

- **Logs:** `docker-compose logs trading-bot`
- **Status:** Check the dashboard at http://localhost:5000
- **Documentation:** See `SYSTEM_DOCUMENTATION.md` for complete details
- **Troubleshooting:** Common issues and solutions in the main documentation

## Next Steps

1. **Monitor Performance:** Watch your first trades in paper mode
2. **Adjust Strategies:** Customize allocations based on your preferences
3. **Scale Up:** Increase capital allocation as you gain confidence
4. **Deploy to Cloud:** Use the production deployment for 24/7 operation

**Happy Trading! 🚀**

---

*Remember: Cryptocurrency trading involves significant risk. Start with paper trading and small amounts until you're comfortable with the system's behavior.*

