"""
Basic Tests - Crypto Arbitrage AI System
Unit tests for core components
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Import components to test
from config.settings import settings, Exchange, Strategy
from core.data_collector.exchange_connector import (
    OrderBookSnapshot,
    ExchangeConnector,
    calculate_book_imbalance_fast,
    calculate_weighted_mid_price
)
from core.arbitrage_engine.spatial_arbitrage import (
    ArbitrageOpportunity,
    SpatialArbitrageEngine,
    calculate_arbitrage_profit
)
from core.feature_engineering.feature_extractor import (
    FeatureVector,
    FeatureExtractor
)


# ============================================
# Settings Tests
# ============================================

class TestSettings:
    """Test configuration settings."""
    
    def test_settings_loaded(self):
        """Test that settings are properly loaded."""
        assert settings is not None
        assert settings.environment is not None
        
    def test_exchange_configuration(self):
        """Test exchange configuration."""
        assert len(settings.enabled_exchanges) == 5
        assert Exchange.BINANCE in settings.enabled_exchanges
        assert Exchange.OKX in settings.enabled_exchanges
        
    def test_performance_targets(self):
        """Test performance target configuration."""
        assert settings.performance.max_latency_ms == 30
        assert settings.performance.ml_inference_ms == 2
        assert settings.performance.feature_extraction_ms == 3
        assert settings.performance.min_throughput_per_sec == 10000
        
    def test_risk_limits(self):
        """Test risk limit configuration."""
        assert settings.risk_limits.max_position_size_pct == 0.02
        assert settings.risk_limits.max_daily_loss_pct == 0.03
        assert settings.risk_limits.max_drawdown_pct == 0.05
        assert settings.risk_limits.min_sharpe_ratio == 2.5
        
    def test_capital_allocation(self):
        """Test capital allocation configuration."""
        assert settings.capital.pre_funding_pct == 0.70
        assert settings.capital.rebalancing_pct == 0.30
        
        # Total strategy allocation should be 100%
        total = (settings.capital.spatial_strategy_pct + 
                settings.capital.statistical_strategy_pct + 
                settings.capital.triangular_strategy_pct)
        assert total == 1.0


# ============================================
# OrderBook Tests
# ============================================

class TestOrderBook:
    """Test orderbook functionality."""
    
    def test_orderbook_creation(self):
        """Test creating an orderbook snapshot."""
        bids = np.array([[100.0, 1.0], [99.0, 2.0], [98.0, 3.0]])
        asks = np.array([[101.0, 1.0], [102.0, 2.0], [103.0, 3.0]])
        
        orderbook = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            latency_ms=5.0
        )
        
        assert orderbook.exchange == "binance"
        assert orderbook.symbol == "BTC/USDT"
        assert orderbook.best_bid == (100.0, 1.0)
        assert orderbook.best_ask == (101.0, 1.0)
        assert orderbook.spread == 1.0
        assert orderbook.mid_price == 100.5
        
    def test_book_imbalance_calculation(self):
        """Test order book imbalance calculation."""
        bids = np.array([[100.0, 10.0], [99.0, 5.0], [98.0, 3.0]])
        asks = np.array([[101.0, 5.0], [102.0, 3.0], [103.0, 2.0]])
        
        imbalance = calculate_book_imbalance_fast(bids, asks, levels=3)
        
        # Total bid volume = 18, ask volume = 10
        # Imbalance = (18 - 10) / (18 + 10) = 8/28 = 0.2857
        assert abs(imbalance - 0.2857) < 0.01
        
    def test_weighted_mid_price(self):
        """Test weighted mid price calculation."""
        bids = np.array([[100.0, 10.0], [99.0, 5.0]])
        asks = np.array([[101.0, 5.0], [102.0, 10.0]])
        
        weighted_mid = calculate_weighted_mid_price(bids, asks, depth=2)
        
        # Should be between best bid and ask
        assert 100.0 < weighted_mid < 101.0


# ============================================
# Arbitrage Opportunity Tests
# ============================================

class TestArbitrageOpportunity:
    """Test arbitrage opportunity detection."""
    
    def test_opportunity_creation(self):
        """Test creating an arbitrage opportunity."""
        opp = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="okx",
            buy_price=100.0,
            sell_price=101.0,
            buy_volume=1.0,
            sell_volume=1.0,
            executable_volume=0.5,
            gross_profit_pct=1.0,
            net_profit_pct=0.7,
            net_profit_usd=35.0,
            confidence=0.8,
            timestamp=time.time(),
            latency_ms=10.0,
            fees={'buy': 0.001, 'sell': 0.001, 'transfer': 0}
        )
        
        assert opp.symbol == "BTC/USDT"
        assert opp.spread_pct == 1.0
        assert opp.is_profitable == True
        assert 0 <= opp.risk_score <= 1
        
    def test_profit_calculation(self):
        """Test arbitrage profit calculation."""
        buy_price = 100.0
        sell_price = 101.0
        volume = 1.0
        buy_fee = 0.001
        sell_fee = 0.001
        
        profit_pct, profit_usd = calculate_arbitrage_profit(
            buy_price, sell_price, volume, buy_fee, sell_fee, 0
        )
        
        # Gross profit = 1%, fees = 0.2%, net = 0.8%
        assert profit_pct > 0
        assert profit_usd > 0


# ============================================
# Feature Extraction Tests
# ============================================

class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_feature_vector_creation(self):
        """Test creating a feature vector."""
        features = FeatureVector(
            timestamp=time.time(),
            symbol="BTC/USDT",
            exchange="binance"
        )
        
        # Should have 50+ features
        assert features.feature_count >= 50
        
        # Test conversion to array
        array = features.to_array()
        assert isinstance(array, np.ndarray)
        assert len(array) == features.feature_count
        
    @pytest.mark.asyncio
    async def test_feature_extraction(self):
        """Test feature extraction from orderbook."""
        extractor = FeatureExtractor()
        
        # Create mock orderbook
        bids = np.array([[100.0, 1.0], [99.0, 2.0]])
        asks = np.array([[101.0, 1.0], [102.0, 2.0]])
        
        orderbook = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            latency_ms=5.0
        )
        
        # Extract features
        features = await extractor.extract_features(
            symbol="BTC/USDT",
            exchange="binance",
            orderbook=orderbook
        )
        
        assert features is not None
        assert features.symbol == "BTC/USDT"
        assert features.exchange == "binance"
        assert features.bid_ask_spread == 1.0
        assert features.mid_price == 100.5
        
    def test_feature_importance(self):
        """Test feature importance calculation."""
        extractor = FeatureExtractor()
        importance = extractor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert 'book_imbalance' in importance
        assert 'rsi' in importance
        
        # Total importance should sum to approximately 1
        total = sum(importance.values())
        assert 0.9 <= total <= 1.1


# ============================================
# Async Exchange Connector Tests
# ============================================

@pytest.mark.asyncio
class TestExchangeConnector:
    """Test exchange connector functionality."""
    
    async def test_connector_initialization(self):
        """Test initializing exchange connector."""
        with patch('ccxt.pro.binance') as mock_exchange:
            mock_instance = AsyncMock()
            mock_exchange.return_value = mock_instance
            
            connector = ExchangeConnector(Exchange.BINANCE, ['BTC/USDT'])
            
            # Mock Redis connection
            with patch('redis.asyncio.Redis') as mock_redis:
                mock_redis_instance = AsyncMock()
                mock_redis.return_value = mock_redis_instance
                mock_redis_instance.ping = AsyncMock()
                
                await connector.initialize()
                
                assert connector.exchange == Exchange.BINANCE
                assert connector.symbols == ['BTC/USDT']
                assert connector.is_connected == True
    
    async def test_orderbook_buffer(self):
        """Test orderbook buffering."""
        connector = ExchangeConnector(Exchange.BINANCE, ['BTC/USDT'])
        
        # Create mock orderbook
        bids = np.array([[100.0, 1.0]])
        asks = np.array([[101.0, 1.0]])
        
        orderbook = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            latency_ms=5.0
        )
        
        # Add to buffer
        connector.orderbook_buffer['BTC/USDT'].append(orderbook)
        
        # Retrieve from buffer
        retrieved = connector.get_orderbook('BTC/USDT')
        
        assert retrieved is not None
        assert retrieved.symbol == "BTC/USDT"
        assert retrieved.mid_price == 100.5


# ============================================
# Spatial Arbitrage Engine Tests
# ============================================

@pytest.mark.asyncio
class TestSpatialArbitrageEngine:
    """Test spatial arbitrage engine."""
    
    async def test_opportunity_detection(self):
        """Test detecting arbitrage opportunities."""
        # Create mock connector
        mock_connector = Mock()
        
        # Create mock orderbooks with arbitrage opportunity
        buy_bids = np.array([[99.0, 10.0]])
        buy_asks = np.array([[100.0, 10.0]])  # Can buy at 100
        
        sell_bids = np.array([[101.0, 10.0]])  # Can sell at 101
        sell_asks = np.array([[102.0, 10.0]])
        
        buy_ob = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=buy_bids,
            asks=buy_asks,
            latency_ms=5.0
        )
        
        sell_ob = OrderBookSnapshot(
            exchange="okx",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=sell_bids,
            asks=sell_asks,
            latency_ms=5.0
        )
        
        # Setup mock connector
        mock_connector.orderbooks = {
            'BTC/USDT': {
                'binance': buy_ob,
                'okx': sell_ob
            }
        }
        
        # Create arbitrage engine
        engine = SpatialArbitrageEngine(mock_connector)
        
        # Scan for opportunities
        opportunities = await engine.scan_opportunities(['BTC/USDT'])
        
        assert len(opportunities) > 0
        
        # Check first opportunity
        opp = opportunities[0]
        assert opp.symbol == 'BTC/USDT'
        assert opp.buy_price == 100.0
        assert opp.sell_price == 101.0
        assert opp.spread_pct == 1.0
    
    async def test_risk_limits(self):
        """Test risk limit enforcement."""
        mock_connector = Mock()
        engine = SpatialArbitrageEngine(mock_connector)
        
        # Set daily loss beyond limit
        engine.daily_loss = -1000  # Exceeds limit
        
        # Check risk limits
        is_breached = engine._check_risk_limits()
        
        assert is_breached == True


# ============================================
# Performance Tests
# ============================================

class TestPerformance:
    """Test performance requirements."""
    
    def test_latency_requirements(self):
        """Test that latency requirements are properly configured."""
        total_latency = (
            settings.performance.ml_inference_ms +
            settings.performance.feature_extraction_ms +
            settings.performance.xai_explanation_ms +
            settings.performance.data_collection_ms
        )
        
        # Total should be less than max latency
        assert total_latency <= settings.performance.max_latency_ms
        
    @pytest.mark.benchmark
    def test_feature_extraction_speed(self, benchmark):
        """Benchmark feature extraction speed."""
        extractor = FeatureExtractor()
        
        bids = np.array([[100.0, 1.0]] * 20)
        asks = np.array([[101.0, 1.0]] * 20)
        
        orderbook = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            latency_ms=5.0
        )
        
        async def extract():
            return await extractor.extract_features(
                "BTC/USDT", "binance", orderbook
            )
        
        # Benchmark should complete in <3ms
        result = benchmark.pedantic(
            lambda: asyncio.run(extract()),
            iterations=10,
            rounds=5
        )
        
        # Check that it meets performance target
        assert benchmark.stats['mean'] < 0.003  # 3ms


# ============================================
# Integration Tests
# ============================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests for system components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test end-to-end arbitrage detection flow."""
        # This would require actual connections
        # For now, just verify imports work
        
        from main import CryptoArbitrageSystem
        
        system = CryptoArbitrageSystem(mode="monitor", symbols=["BTC/USDT"])
        
        assert system is not None
        assert system.mode == "monitor"
        assert system.symbols == ["BTC/USDT"]


# ============================================
# Test Runner
# ============================================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=core",
        "--cov-report=html",
        "--cov-report=term"
    ])