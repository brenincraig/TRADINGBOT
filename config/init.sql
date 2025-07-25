-- PostgreSQL initialization script for Crypto Trading Bot

-- Create database if not exists
SELECT 'CREATE DATABASE trading_bot'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading_bot');

-- Connect to the trading_bot database
\c trading_bot;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO trading, public;

-- Create tables for trading data
CREATE TABLE IF NOT EXISTS exchanges (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    api_key_encrypted TEXT,
    secret_key_encrypted TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    exchange_id INTEGER REFERENCES exchanges(id),
    is_active BOOLEAN DEFAULT true,
    min_trade_amount DECIMAL(20, 8),
    max_trade_amount DECIMAL(20, 8),
    price_precision INTEGER DEFAULT 8,
    quantity_precision INTEGER DEFAULT 8,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, exchange_id)
);

CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    trading_pair_id INTEGER REFERENCES trading_pairs(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair_id INTEGER REFERENCES trading_pairs(id),
    position_type VARCHAR(10) NOT NULL CHECK (position_type IN ('LONG', 'SHORT')),
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    trailing_stop DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    strategy VARCHAR(50),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8),
    fees DECIMAL(20, 8) DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id),
    trading_pair_id INTEGER REFERENCES trading_pairs(id),
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED')),
    exchange_order_id VARCHAR(100),
    fees DECIMAL(20, 8) DEFAULT 0,
    fee_currency VARCHAR(10),
    executed_at TIMESTAMP WITH TIME ZONE,
    strategy VARCHAR(50),
    signal_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    available_balance DECIMAL(20, 8) NOT NULL,
    total_exposure DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8) NOT NULL,
    realized_pnl DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8) NOT NULL,
    drawdown DECIMAL(10, 6) NOT NULL,
    positions_count INTEGER NOT NULL,
    active_strategies JSONB,
    risk_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair_id INTEGER REFERENCES trading_pairs(id),
    signal_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
    strength DECIMAL(5, 4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    price DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    indicators JSONB,
    metadata JSONB,
    is_executed BOOLEAN DEFAULT false,
    executed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT NOT NULL,
    position_id UUID REFERENCES positions(id),
    trading_pair_id INTEGER REFERENCES trading_pairs(id),
    threshold_value DECIMAL(20, 8),
    current_value DECIMAL(20, 8),
    action_taken VARCHAR(100),
    metadata JSONB,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_performance (
    id BIGSERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    trades_count INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    win_rate DECIMAL(5, 4) NOT NULL DEFAULT 0,
    avg_win DECIMAL(20, 8) NOT NULL DEFAULT 0,
    avg_loss DECIMAL(20, 8) NOT NULL DEFAULT 0,
    profit_factor DECIMAL(10, 4) NOT NULL DEFAULT 0,
    sharpe_ratio DECIMAL(10, 4) NOT NULL DEFAULT 0,
    max_drawdown DECIMAL(10, 6) NOT NULL DEFAULT 0,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, date)
);

-- Create analytics schema tables
SET search_path TO analytics, public;

CREATE TABLE IF NOT EXISTS technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    trading_pair_id INTEGER REFERENCES trading.trading_pairs(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_analysis (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    market_regime VARCHAR(20) NOT NULL,
    volatility_level VARCHAR(20) NOT NULL,
    trend_direction VARCHAR(20) NOT NULL,
    correlation_matrix JSONB,
    sentiment_score DECIMAL(5, 4),
    analysis_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create monitoring schema tables
SET search_path TO monitoring, public;

CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    cpu_usage DECIMAL(5, 2) NOT NULL,
    memory_usage DECIMAL(5, 2) NOT NULL,
    disk_usage DECIMAL(5, 2) NOT NULL,
    network_io JSONB,
    active_connections INTEGER NOT NULL,
    api_response_time DECIMAL(10, 4) NOT NULL,
    error_rate DECIMAL(5, 4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS application_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    level VARCHAR(10) NOT NULL,
    logger VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function VARCHAR(100),
    line_number INTEGER,
    exception_info TEXT,
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    notification_type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    level VARCHAR(20) NOT NULL,
    metadata JSONB,
    sent_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'SENT', 'FAILED')),
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
SET search_path TO trading, public;

-- Market data indexes
CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp ON market_data(trading_pair_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC);

-- Position indexes
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_pair_status ON positions(trading_pair_id, status);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time DESC);

-- Trade indexes
CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_pair_timestamp ON trades(trading_pair_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at DESC);

-- Signal indexes
CREATE INDEX IF NOT EXISTS idx_signals_pair_timestamp ON signals(trading_pair_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(is_executed, timestamp DESC);

-- Portfolio snapshot indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp DESC);

-- Risk event indexes
CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_position ON risk_events(position_id);

-- Strategy performance indexes
CREATE INDEX IF NOT EXISTS idx_strategy_performance_name_date ON strategy_performance(strategy_name, date DESC);

-- Analytics indexes
SET search_path TO analytics, public;
CREATE INDEX IF NOT EXISTS idx_technical_indicators_pair_timestamp ON technical_indicators(trading_pair_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_name ON technical_indicators(indicator_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_analysis_timestamp ON market_analysis(timestamp DESC);

-- Monitoring indexes
SET search_path TO monitoring, public;
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_application_logs_timestamp ON application_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_application_logs_level ON application_logs(level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status, created_at DESC);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
SET search_path TO trading, public;

CREATE TRIGGER update_exchanges_updated_at BEFORE UPDATE ON exchanges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW active_positions AS
SELECT 
    p.*,
    tp.symbol,
    tp.base_currency,
    tp.quote_currency,
    e.name as exchange_name
FROM positions p
JOIN trading_pairs tp ON p.trading_pair_id = tp.id
JOIN exchanges e ON tp.exchange_id = e.id
WHERE p.status = 'OPEN';

CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(realized_pnl) as total_pnl,
    AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
    AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss
FROM positions 
WHERE status = 'CLOSED' AND realized_pnl IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Insert initial data
INSERT INTO exchanges (name, is_active) VALUES 
    ('Binance', true),
    ('Coinbase', true),
    ('Kraken', true)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO trading_user;

-- Reset search path
SET search_path TO public;

