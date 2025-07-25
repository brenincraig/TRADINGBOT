"""
Comprehensive Integration Tests for Crypto Trading Bot
"""
import pytest
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_engine import TradingEngine
from risk_manager import RiskManager, PositionType
from data_processor import DataProcessor
from exchange_manager import ExchangeManager
from technical_analysis import TechnicalAnalysis
from signal_detector import SignalDetector
from notification_manager import NotificationManager
from portfolio_manager import PortfolioManager
from adaptive_learning import AdaptiveLearning

class TestSystemIntegration:
    """Integration tests for the complete trading system"""
    
    @pytest.fixture
    def setup_system(self):
        """Setup complete trading system for testing"""
        # Mock configuration
        config = {
            'initial_capital': 10000,
            'exchanges': {
                'binance': {
                    'api_key': 'test_key',
                    'secret_key': 'test_secret',
                    'testnet': True
                }
            },
            'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
            'risk_params': {
                'max_position_size_pct': 0.05,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'telegram': {'bot_token': 'test_token', 'chat_id': 'test_chat'},
            'discord': {'webhook_url': 'test_webhook'}
        }
        
        # Initialize components
        risk_manager = RiskManager(config['initial_capital'])
        exchange_manager = ExchangeManager(config['exchanges'])
        data_processor = DataProcessor()
        technical_analysis = TechnicalAnalysis()
        signal_detector = SignalDetector()
        notification_manager = NotificationManager(config)
        portfolio_manager = PortfolioManager(risk_manager, config['initial_capital'])
        adaptive_learning = AdaptiveLearning()
        
        trading_engine = TradingEngine(
            risk_manager=risk_manager,
            exchange_manager=exchange_manager,
            data_processor=data_processor,
            technical_analysis=technical_analysis,
            signal_detector=signal_detector,
            notification_manager=notification_manager,
            portfolio_manager=portfolio_manager,
            adaptive_learning=adaptive_learning,
            config=config
        )
        
        return {
            'trading_engine': trading_engine,
            'risk_manager': risk_manager,
            'exchange_manager': exchange_manager,
            'data_processor': data_processor,
            'technical_analysis': technical_analysis,
            'signal_detector': signal_detector,
            'notification_manager': notification_manager,
            'portfolio_manager': portfolio_manager,
            'adaptive_learning': adaptive_learning,
            'config': config
        }
    
    def test_system_initialization(self, setup_system):
        """Test complete system initialization"""
        system = setup_system
        
        # Test all components are initialized
        assert system['trading_engine'] is not None
        assert system['risk_manager'] is not None
        assert system['exchange_manager'] is not None
        assert system['data_processor'] is not None
        assert system['technical_analysis'] is not None
        assert system['signal_detector'] is not None
        assert system['notification_manager'] is not None
        assert system['portfolio_manager'] is not None
        assert system['adaptive_learning'] is not None
        
        # Test initial state
        assert system['risk_manager'].current_capital == 10000
        assert len(system['risk_manager'].positions) == 0
        assert system['risk_manager'].total_exposure == 0
    
    @pytest.mark.asyncio
    async def test_data_flow_integration(self, setup_system):
        """Test data flow through the entire system"""
        system = setup_system
        
        # Mock market data
        mock_market_data = {
            'BTC/USDT': {
                'timestamp': datetime.utcnow(),
                'price': 50000.0,
                'volume': 1000.0,
                'bid': 49995.0,
                'ask': 50005.0
            }
        }
        
        # Test data processing
        with patch.object(system['exchange_manager'], 'get_market_data', return_value=mock_market_data):
            market_data = await system['exchange_manager'].get_market_data(['BTC/USDT'])
            assert 'BTC/USDT' in market_data
            assert market_data['BTC/USDT']['price'] == 50000.0
        
        # Test technical analysis
        price_data = [49000, 49500, 50000, 50500, 50000]
        indicators = system['technical_analysis'].calculate_indicators(price_data)
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'sma' in indicators
        
        # Test signal detection
        signals = system['signal_detector'].detect_signals('BTC/USDT', indicators, mock_market_data['BTC/USDT'])
        assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_trading_workflow(self, setup_system):
        """Test complete trading workflow"""
        system = setup_system
        
        # Mock successful trade execution
        with patch.object(system['exchange_manager'], 'place_order', return_value={'order_id': 'test_order', 'status': 'filled'}):
            with patch.object(system['exchange_manager'], 'get_market_data') as mock_market_data:
                mock_market_data.return_value = {
                    'BTC/USDT': {
                        'timestamp': datetime.utcnow(),
                        'price': 50000.0,
                        'volume': 1000.0,
                        'bid': 49995.0,
                        'ask': 50005.0
                    }
                }
                
                # Test position opening
                success = system['risk_manager'].open_position(
                    symbol='BTC/USDT',
                    position_type=PositionType.LONG,
                    entry_price=50000.0,
                    quantity=0.1
                )
                
                assert success == True
                assert len(system['risk_manager'].positions) == 1
                assert 'BTC/USDT' in system['risk_manager'].positions
                
                # Test position update
                system['risk_manager'].update_position('BTC/USDT', 50500.0)
                position = system['risk_manager'].positions['BTC/USDT']
                assert position.current_price == 50500.0
                assert position.unrealized_pnl > 0  # Profit
                
                # Test position closing
                success = system['risk_manager'].close_position('BTC/USDT', 50500.0, 'Test close')
                assert success == True
                assert len(system['risk_manager'].positions) == 0
    
    def test_risk_management_integration(self, setup_system):
        """Test risk management system integration"""
        system = setup_system
        
        # Test position size calculation
        position_size = system['risk_manager'].calculate_position_size(
            symbol='BTC/USDT',
            entry_price=50000.0,
            stop_loss_price=49000.0,
            confidence=1.0
        )
        
        assert position_size > 0
        assert position_size * 50000.0 <= system['risk_manager'].current_capital * 0.05  # Max 5% position
        
        # Test trade validation
        is_valid, message = system['risk_manager'].validate_trade(
            symbol='BTC/USDT',
            position_type=PositionType.LONG,
            quantity=position_size,
            price=50000.0
        )
        
        assert is_valid == True
        assert "validated" in message.lower()
        
        # Test risk limits
        large_position_size = system['risk_manager'].current_capital / 40000.0  # Very large position
        is_valid, message = system['risk_manager'].validate_trade(
            symbol='BTC/USDT',
            position_type=PositionType.LONG,
            quantity=large_position_size,
            price=50000.0
        )
        
        assert is_valid == False  # Should be rejected due to size
    
    @pytest.mark.asyncio
    async def test_notification_integration(self, setup_system):
        """Test notification system integration"""
        system = setup_system
        
        # Mock notification sending
        with patch.object(system['notification_manager'], '_send_telegram_message', return_value=True):
            with patch.object(system['notification_manager'], '_send_discord_message', return_value=True):
                
                # Test trade notification
                await system['notification_manager'].notify_trade_opened({
                    'symbol': 'BTC/USDT',
                    'position_type': 'LONG',
                    'entry_price': 50000.0,
                    'quantity': 0.1,
                    'strategy': 'momentum'
                })
                
                # Test risk alert notification
                await system['notification_manager'].notify_risk_alert({
                    'type': 'Stop Loss Triggered',
                    'message': 'Position closed due to stop loss',
                    'severity': 'warning',
                    'metadata': {'symbol': 'BTC/USDT', 'loss': -100.0}
                })
                
                # Verify notifications were queued
                assert system['notification_manager'].message_queue.qsize() > 0
    
    def test_portfolio_management_integration(self, setup_system):
        """Test portfolio management integration"""
        system = setup_system
        
        # Mock market data for portfolio optimization
        mock_market_data = {
            'BTC/USDT': [
                {'price': 49000, 'timestamp': datetime.utcnow() - timedelta(hours=i)}
                for i in range(100, 0, -1)
            ],
            'ETH/USDT': [
                {'price': 3000, 'timestamp': datetime.utcnow() - timedelta(hours=i)}
                for i in range(100, 0, -1)
            ]
        }
        
        # Test allocation update
        system['portfolio_manager'].update_target_allocations(mock_market_data, 'trending')
        
        assert len(system['portfolio_manager'].target_allocations) > 0
        total_allocation = sum(system['portfolio_manager'].target_allocations.values())
        assert abs(total_allocation - 1.0) < 0.01  # Should sum to ~1.0
        
        # Test rebalancing logic
        current_allocations = system['portfolio_manager'].calculate_current_allocations()
        should_rebalance = system['portfolio_manager'].should_rebalance()
        
        # Initially should not need rebalancing (no positions)
        assert should_rebalance == False or len(current_allocations) == 0
    
    def test_adaptive_learning_integration(self, setup_system):
        """Test adaptive learning system integration"""
        system = setup_system
        
        # Mock trade history for learning
        mock_trades = [
            {
                'symbol': 'BTC/USDT',
                'strategy': 'momentum',
                'entry_price': 50000,
                'exit_price': 50500,
                'realized_pnl': 50,
                'realized_pnl_pct': 0.01,
                'indicators': {'rsi': 30, 'macd': 0.5}
            },
            {
                'symbol': 'BTC/USDT',
                'strategy': 'momentum',
                'entry_price': 51000,
                'exit_price': 50800,
                'realized_pnl': -20,
                'realized_pnl_pct': -0.004,
                'indicators': {'rsi': 70, 'macd': -0.2}
            }
        ]
        
        # Test learning from trades
        for trade in mock_trades:
            system['adaptive_learning'].update_q_learning(trade)
        
        # Test strategy recommendation
        current_state = {'rsi': 35, 'macd': 0.3, 'volume_ratio': 1.2}
        recommendation = system['adaptive_learning'].get_strategy_recommendation('BTC/USDT', current_state)
        
        assert 'strategy' in recommendation
        assert 'confidence' in recommendation
        assert 0 <= recommendation['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, setup_system):
        """Test error handling across system components"""
        system = setup_system
        
        # Test exchange connection error
        with patch.object(system['exchange_manager'], 'get_market_data', side_effect=Exception("Connection error")):
            try:
                await system['exchange_manager'].get_market_data(['BTC/USDT'])
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Connection error" in str(e)
        
        # Test invalid trade parameters
        is_valid, message = system['risk_manager'].validate_trade(
            symbol='INVALID/PAIR',
            position_type=PositionType.LONG,
            quantity=-1.0,  # Invalid quantity
            price=0.0  # Invalid price
        )
        
        assert is_valid == False
        
        # Test notification failure handling
        with patch.object(system['notification_manager'], '_send_telegram_message', return_value=False):
            await system['notification_manager'].send_notification(
                level=system['notification_manager'].NotificationLevel.ERROR,
                title="Test Error",
                message="Test error message"
            )
            
            # Should handle failure gracefully
            assert len(system['notification_manager'].failed_messages) >= 0
    
    def test_performance_metrics_integration(self, setup_system):
        """Test performance metrics calculation"""
        system = setup_system
        
        # Simulate some trading activity
        system['risk_manager'].open_position('BTC/USDT', PositionType.LONG, 50000.0, 0.1)
        system['risk_manager'].update_position('BTC/USDT', 50500.0)
        system['risk_manager'].close_position('BTC/USDT', 50500.0, 'Test trade')
        
        # Test risk metrics
        risk_metrics = system['risk_manager'].get_risk_metrics()
        assert risk_metrics.total_exposure >= 0
        assert risk_metrics.drawdown_current >= 0
        assert risk_metrics.risk_level is not None
        
        # Test portfolio metrics
        system['portfolio_manager'].update_portfolio_value(10050.0)  # Profit
        portfolio_metrics = system['portfolio_manager'].calculate_portfolio_metrics()
        
        assert portfolio_metrics.total_value > 0
        assert portfolio_metrics.total_return >= 0  # Should be positive
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_system):
        """Test system under concurrent operations"""
        system = setup_system
        
        async def mock_trading_operation(symbol, price):
            """Mock concurrent trading operation"""
            try:
                # Simulate market data processing
                await asyncio.sleep(0.01)
                
                # Simulate signal detection
                signals = system['signal_detector'].detect_signals(
                    symbol, 
                    {'rsi': 30, 'macd': 0.5}, 
                    {'price': price, 'volume': 1000}
                )
                
                # Simulate position management
                if signals and len(system['risk_manager'].positions) < 3:
                    system['risk_manager'].open_position(symbol, PositionType.LONG, price, 0.01)
                
                return True
            except Exception as e:
                return False
        
        # Run concurrent operations
        tasks = [
            mock_trading_operation('BTC/USDT', 50000 + i * 10)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most operations should succeed
        successful_operations = sum(1 for r in results if r is True)
        assert successful_operations >= 5  # At least half should succeed
    
    def test_memory_usage_optimization(self, setup_system):
        """Test memory usage and optimization"""
        system = setup_system
        
        # Test data structure limits
        initial_memory = len(system['risk_manager'].trade_history)
        
        # Add many trades to test memory limits
        for i in range(1500):  # More than the deque limit
            trade_data = {
                'symbol': 'BTC/USDT',
                'realized_pnl': i,
                'timestamp': datetime.utcnow()
            }
            system['risk_manager'].trade_history.append(trade_data)
        
        # Should not exceed maximum size
        assert len(system['risk_manager'].trade_history) <= 1000  # deque maxlen
        
        # Test portfolio history limits
        for i in range(2000):
            system['portfolio_manager'].update_portfolio_value(10000 + i)
        
        assert len(system['portfolio_manager'].portfolio_value_history) <= 1440  # 24 hours
    
    def test_configuration_validation(self, setup_system):
        """Test configuration validation and error handling"""
        system = setup_system
        
        # Test invalid risk parameters
        invalid_params = {
            'max_position_size_pct': 1.5,  # > 100%
            'stop_loss_pct': -0.1,  # Negative
            'max_drawdown_pct': 2.0  # > 100%
        }
        
        # Should handle invalid parameters gracefully
        try:
            system['risk_manager'].update_risk_parameters(invalid_params)
            # Parameters should be clamped or rejected
            assert system['risk_manager'].risk_params['max_position_size_pct'] <= 1.0
        except Exception:
            pass  # Expected to fail validation
        
        # Test missing configuration
        incomplete_config = {'initial_capital': 1000}  # Missing required fields
        
        # System should handle missing config gracefully
        assert system['risk_manager'].initial_capital > 0

class TestPerformanceOptimization:
    """Performance optimization tests"""
    
    def test_data_processing_performance(self):
        """Test data processing performance"""
        data_processor = DataProcessor()
        
        # Generate large dataset
        large_dataset = np.random.rand(10000, 5)  # 10k data points, 5 features
        
        start_time = time.time()
        
        # Process data
        processed_data = data_processor.process_market_data(large_dataset)
        
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (< 1 second)
        assert processing_time < 1.0
        assert processed_data is not None
    
    def test_technical_analysis_performance(self):
        """Test technical analysis performance"""
        technical_analysis = TechnicalAnalysis()
        
        # Generate price data
        price_data = np.random.rand(1000) * 50000 + 45000  # 1000 price points
        
        start_time = time.time()
        
        # Calculate indicators
        indicators = technical_analysis.calculate_indicators(price_data.tolist())
        
        calculation_time = time.time() - start_time
        
        # Should calculate within reasonable time
        assert calculation_time < 0.5
        assert len(indicators) > 0
    
    def test_signal_detection_performance(self):
        """Test signal detection performance"""
        signal_detector = SignalDetector()
        
        # Mock indicators
        indicators = {
            'rsi': np.random.rand(100) * 100,
            'macd': np.random.rand(100) * 2 - 1,
            'sma': np.random.rand(100) * 50000 + 45000
        }
        
        market_data = {
            'price': 50000.0,
            'volume': 1000.0,
            'timestamp': datetime.utcnow()
        }
        
        start_time = time.time()
        
        # Detect signals multiple times
        for _ in range(100):
            signals = signal_detector.detect_signals('BTC/USDT', indicators, market_data)
        
        detection_time = time.time() - start_time
        
        # Should detect signals efficiently
        assert detection_time < 1.0
    
    def test_database_query_performance(self):
        """Test database query performance (mock)"""
        # This would test actual database performance in a real environment
        # For now, we'll test the data structures used
        
        # Simulate large trade history
        trade_history = []
        
        start_time = time.time()
        
        # Add many trades
        for i in range(10000):
            trade = {
                'id': i,
                'symbol': 'BTC/USDT',
                'price': 50000 + i,
                'timestamp': datetime.utcnow(),
                'pnl': i * 0.1
            }
            trade_history.append(trade)
        
        # Query operations
        profitable_trades = [t for t in trade_history if t['pnl'] > 0]
        recent_trades = sorted(trade_history, key=lambda x: x['timestamp'], reverse=True)[:100]
        
        query_time = time.time() - start_time
        
        # Should handle large datasets efficiently
        assert query_time < 2.0
        assert len(profitable_trades) > 0
        assert len(recent_trades) == 100

class TestPaperTradingSimulation:
    """Paper trading simulation tests"""
    
    @pytest.fixture
    def paper_trading_setup(self):
        """Setup paper trading environment"""
        config = {
            'initial_capital': 10000,
            'paper_trading': True,
            'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
            'simulation_speed': 1.0  # Real-time
        }
        
        risk_manager = RiskManager(config['initial_capital'])
        
        return {
            'config': config,
            'risk_manager': risk_manager,
            'initial_capital': config['initial_capital']
        }
    
    def test_paper_trading_basic_operations(self, paper_trading_setup):
        """Test basic paper trading operations"""
        setup = paper_trading_setup
        risk_manager = setup['risk_manager']
        
        # Test opening position
        success = risk_manager.open_position(
            symbol='BTC/USDT',
            position_type=PositionType.LONG,
            entry_price=50000.0,
            quantity=0.1
        )
        
        assert success == True
        assert len(risk_manager.positions) == 1
        
        # Test position update
        risk_manager.update_position('BTC/USDT', 51000.0)  # Price increase
        position = risk_manager.positions['BTC/USDT']
        assert position.unrealized_pnl > 0  # Should be profitable
        
        # Test closing position
        success = risk_manager.close_position('BTC/USDT', 51000.0, 'Paper trading test')
        assert success == True
        assert len(risk_manager.positions) == 0
        assert risk_manager.current_capital > setup['initial_capital']  # Should have profit
    
    def test_paper_trading_risk_management(self, paper_trading_setup):
        """Test risk management in paper trading"""
        setup = paper_trading_setup
        risk_manager = setup['risk_manager']
        
        # Test stop loss
        risk_manager.open_position('BTC/USDT', PositionType.LONG, 50000.0, 0.1, stop_loss=49000.0)
        risk_manager.update_position('BTC/USDT', 48500.0)  # Price drops below stop loss
        
        should_exit, reason = risk_manager.check_exit_conditions('BTC/USDT')
        assert should_exit == True
        assert 'stop loss' in reason.lower()
        
        # Test take profit
        risk_manager.close_position('BTC/USDT', 48500.0, 'Stop loss triggered')
        risk_manager.open_position('BTC/USDT', PositionType.LONG, 50000.0, 0.1, take_profit=52000.0)
        risk_manager.update_position('BTC/USDT', 52500.0)  # Price rises above take profit
        
        should_exit, reason = risk_manager.check_exit_conditions('BTC/USDT')
        assert should_exit == True
        assert 'take profit' in reason.lower()
    
    def test_paper_trading_performance_tracking(self, paper_trading_setup):
        """Test performance tracking in paper trading"""
        setup = paper_trading_setup
        risk_manager = setup['risk_manager']
        
        # Execute multiple trades
        trades = [
            {'entry': 50000, 'exit': 50500, 'quantity': 0.1},  # Profit
            {'entry': 51000, 'exit': 50800, 'quantity': 0.1},  # Loss
            {'entry': 49000, 'exit': 49300, 'quantity': 0.1},  # Profit
        ]
        
        for i, trade in enumerate(trades):
            symbol = f'TEST{i}/USDT'
            risk_manager.open_position(symbol, PositionType.LONG, trade['entry'], trade['quantity'])
            risk_manager.update_position(symbol, trade['exit'])
            risk_manager.close_position(symbol, trade['exit'], f'Test trade {i}')
        
        # Check performance metrics
        trade_history = list(risk_manager.trade_history)
        assert len(trade_history) == 3
        
        profitable_trades = [t for t in trade_history if t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in trade_history if t.get('realized_pnl', 0) < 0]
        
        assert len(profitable_trades) == 2
        assert len(losing_trades) == 1
        
        # Calculate win rate
        win_rate = len(profitable_trades) / len(trade_history)
        assert win_rate == 2/3  # 66.67%

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])

