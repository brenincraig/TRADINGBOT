"""
Database models and connection handler for the crypto trading bot
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
import logging
from config import Config

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=False)
    order_type = Column(String(10), nullable=False)  # Market, Limit
    side = Column(String(4), nullable=False)  # Buy, Sell
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    fee_asset = Column(String(10))
    profit_loss = Column(Float, default=0.0)
    strategy_id = Column(Integer, ForeignKey('strategies.strategy_id'))
    order_id = Column(String(50))
    slippage = Column(Float, default=0.0)
    liquidity_snapshot = Column(JSON)
    order_book_depth_snapshot = Column(JSON)
    
    strategy = relationship("Strategy", back_populates="trades")

class Strategy(Base):
    __tablename__ = 'strategies'
    
    strategy_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(Text)
    creation_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON)
    performance_metrics = Column(JSON)
    is_active = Column(Boolean, default=True)
    
    trades = relationship("Trade", back_populates="strategy")

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_quantity = Column(Float)
    ask_quantity = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    moving_avg_short = Column(Float)
    moving_avg_long = Column(Float)

class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset = Column(String(10), nullable=False)
    balance = Column(Float, nullable=False)
    locked_balance = Column(Float, default=0.0)
    usd_value = Column(Float)
    exchange = Column(String(20), nullable=False)

class PerformanceMetrics(Base):
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_profit_loss = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    average_profit = Column(Float, default=0.0)
    average_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)

class DatabaseManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = None
        self.Session = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            Base.metadata.create_all(self.engine)
            self.logger.info("Database connection established successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def get_session(self):
        """Get a new database session"""
        if self.Session:
            return self.Session()
        else:
            raise Exception("Database not connected. Call connect() first.")
    
    def save_trade(self, trade_data):
        """Save a trade to the database"""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            self.logger.info(f"Trade saved: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
            return trade.trade_id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save trade: {e}")
            return None
        finally:
            session.close()
    
    def save_market_data(self, market_data):
        """Save market data to the database"""
        session = self.get_session()
        try:
            data = MarketData(**market_data)
            session.add(data)
            session.commit()
            return data.id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save market data: {e}")
            return None
        finally:
            session.close()
    
    def update_portfolio(self, portfolio_data):
        """Update portfolio balances"""
        session = self.get_session()
        try:
            # Check if record exists
            existing = session.query(Portfolio).filter_by(
                asset=portfolio_data['asset'],
                exchange=portfolio_data['exchange']
            ).first()
            
            if existing:
                for key, value in portfolio_data.items():
                    setattr(existing, key, value)
                existing.timestamp = datetime.utcnow()
            else:
                portfolio = Portfolio(**portfolio_data)
                session.add(portfolio)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to update portfolio: {e}")
            return False
        finally:
            session.close()
    
    def get_recent_trades(self, limit=100):
        """Get recent trades"""
        session = self.get_session()
        try:
            trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
            return trades
        except Exception as e:
            self.logger.error(f"Failed to get recent trades: {e}")
            return []
        finally:
            session.close()
    
    def get_performance_metrics(self):
        """Get latest performance metrics"""
        session = self.get_session()
        try:
            metrics = session.query(PerformanceMetrics).order_by(PerformanceMetrics.timestamp.desc()).first()
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return None
        finally:
            session.close()
    
    def cleanup_old_data(self, days=30):
        """Clean up old market data"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted = session.query(MarketData).filter(MarketData.timestamp < cutoff_date).delete()
            session.commit()
            self.logger.info(f"Cleaned up {deleted} old market data records")
            return deleted
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to cleanup old data: {e}")
            return 0
        finally:
            session.close()

