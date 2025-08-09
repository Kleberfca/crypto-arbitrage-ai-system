"""
Feature Extractor - Extração de 50+ features para ML
Implementa extração otimizada com latência <3ms
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from numba import jit, prange
import talib
from scipy import stats
from scipy.signal import find_peaks

from config.settings import settings
from core.data_collector.exchange_connector import (
    OrderBookSnapshot,
    TradeData,
    TickerData
)


logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Vetor de features extraídas."""
    
    timestamp: float
    symbol: str
    exchange: str
    
    # Price features (12)
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_15m: float = 0.0
    returns_1h: float = 0.0
    log_returns: float = 0.0
    squared_returns: float = 0.0
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    volatility_15m: float = 0.0
    momentum: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0
    
    # Volume features (10)
    vwap: float = 0.0
    volume_profile: float = 0.0
    volume_imbalance: float = 0.0
    buy_sell_ratio: float = 0.0
    large_order_ratio: float = 0.0
    volume_momentum: float = 0.0
    cumulative_volume_delta: float = 0.0
    volume_weighted_spread: float = 0.0
    trade_intensity: float = 0.0
    volume_concentration: float = 0.0
    
    # Orderbook features (12)
    bid_ask_spread: float = 0.0
    spread_bps: float = 0.0
    mid_price: float = 0.0
    micro_price: float = 0.0
    book_imbalance: float = 0.0
    book_depth_ratio: float = 0.0
    order_flow_imbalance: float = 0.0
    spoofing_indicator: float = 0.0
    liquidity_score: float = 0.0
    orderbook_heat: float = 0.0
    bid_wall_distance: float = 0.0
    ask_wall_distance: float = 0.0
    
    # Technical indicators (20)
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    atr: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    stochastic_k: float = 0.0
    stochastic_d: float = 0.0
    williams_r: float = 0.0
    cci: float = 0.0
    adx: float = 0.0
    obv: float = 0.0
    mfi: float = 0.0
    roc: float = 0.0
    
    # Network features (6)
    gas_price_gwei: float = 0.0
    network_congestion: float = 0.0
    mempool_size: float = 0.0
    pending_transactions: float = 0.0
    block_time: float = 0.0
    hash_rate: float = 0.0
    
    # Cross-asset features (8)
    btc_dominance: float = 0.0
    correlation_btc: float = 0.0
    correlation_eth: float = 0.0
    market_cap_ratio: float = 0.0
    volume_ratio: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    market_sentiment: float = 0.0
    
    # Statistical features (8)
    zscore: float = 0.0
    entropy: float = 0.0
    autocorrelation: float = 0.0
    hurst_exponent: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    price_efficiency: float = 0.0
    information_ratio: float = 0.0
    
    # Execution metrics (4)
    latency_ms: float = 0.0
    data_age_ms: float = 0.0
    confidence: float = 0.0
    feature_quality: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Converte para array numpy."""
        return np.array([
            # Price features
            self.returns_1m, self.returns_5m, self.returns_15m, self.returns_1h,
            self.log_returns, self.squared_returns,
            self.volatility_1m, self.volatility_5m, self.volatility_15m,
            self.momentum, self.acceleration, self.jerk,
            
            # Volume features
            self.vwap, self.volume_profile, self.volume_imbalance,
            self.buy_sell_ratio, self.large_order_ratio, self.volume_momentum,
            self.cumulative_volume_delta, self.volume_weighted_spread,
            self.trade_intensity, self.volume_concentration,
            
            # Orderbook features
            self.bid_ask_spread, self.spread_bps, self.mid_price, self.micro_price,
            self.book_imbalance, self.book_depth_ratio, self.order_flow_imbalance,
            self.spoofing_indicator, self.liquidity_score, self.orderbook_heat,
            self.bid_wall_distance, self.ask_wall_distance,
            
            # Technical indicators
            self.rsi, self.macd, self.macd_signal, self.macd_histogram,
            self.bollinger_upper, self.bollinger_middle, self.bollinger_lower,
            self.atr, self.ema_9, self.ema_21, self.ema_50, self.ema_200,
            self.stochastic_k, self.stochastic_d, self.williams_r,
            self.cci, self.adx, self.obv, self.mfi, self.roc,
            
            # Network features
            self.gas_price_gwei, self.network_congestion, self.mempool_size,
            self.pending_transactions, self.block_time, self.hash_rate,
            
            # Cross-asset features
            self.btc_dominance, self.correlation_btc, self.correlation_eth,
            self.market_cap_ratio, self.volume_ratio, self.funding_rate,
            self.open_interest, self.market_sentiment,
            
            # Statistical features
            self.zscore, self.entropy, self.autocorrelation, self.hurst_exponent,
            self.kurtosis, self.skewness, self.price_efficiency, self.information_ratio,
            
            # Execution metrics
            self.latency_ms, self.data_age_ms, self.confidence, self.feature_quality
        ])
    
    @property
    def feature_count(self) -> int:
        """Retorna número total de features."""
        return len(self.to_array())


class FeatureExtractor:
    """
    Extrator de features otimizado para latência <3ms.
    Extrai 50+ features de dados de mercado.
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Inicializa o extrator.
        
        Args:
            lookback_periods: Períodos de lookback para cálculos
        """
        self.lookback_periods = lookback_periods or {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600
        }
        
        # Price history buffers
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.trade_history: Dict[str, List[TradeData]] = {}
        
        # Technical indicator periods
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.atr_period = 14
        
        # Performance tracking
        self.extraction_times: List[float] = []
        self.feature_cache: Dict[str, FeatureVector] = {}
        self.cache_ttl = 0.1  # 100ms cache
        
        # Network data (mock for now)
        self.network_data = {
            'gas_price_gwei': 25.0,
            'network_congestion': 0.3,
            'mempool_size': 150000,
            'pending_transactions': 2500,
            'block_time': 3.0,
            'hash_rate': 500e18
        }
        
        # Market data (mock for now)
        self.market_data = {
            'btc_dominance': 0.48,
            'market_sentiment': 0.65
        }
    
    async def extract_features(
        self,
        symbol: str,
        exchange: str,
        orderbook: OrderBookSnapshot,
        trades: List[TradeData] = None,
        ticker: TickerData = None
    ) -> FeatureVector:
        """
        Extrai todas as features de dados de mercado.
        
        Args:
            symbol: Símbolo do ativo
            exchange: Exchange
            orderbook: Snapshot do orderbook
            trades: Trades recentes
            ticker: Dados do ticker
            
        Returns:
            Vetor de features extraídas
        """
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{exchange}:{symbol}"
        cached = self.feature_cache.get(cache_key)
        if cached and (time.time() - cached.timestamp) < self.cache_ttl:
            return cached
        
        # Initialize feature vector
        features = FeatureVector(
            timestamp=time.time(),
            symbol=symbol,
            exchange=exchange
        )
        
        # Extract features in parallel for speed
        tasks = [
            self._extract_price_features(features, orderbook, ticker),
            self._extract_volume_features(features, trades, ticker),
            self._extract_orderbook_features(features, orderbook),
            self._extract_technical_indicators(features, symbol),
            self._extract_network_features(features),
            self._extract_cross_asset_features(features, symbol),
            self._extract_statistical_features(features, orderbook)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate execution metrics
        features.latency_ms = orderbook.latency_ms
        features.data_age_ms = (time.time() - orderbook.timestamp) * 1000
        features.confidence = self._calculate_feature_confidence(features)
        features.feature_quality = self._calculate_feature_quality(features)
        
        # Track extraction time
        extraction_time = (time.perf_counter() - start_time) * 1000
        self.extraction_times.append(extraction_time)
        
        if extraction_time > 3:
            logger.warning(f"Slow feature extraction: {extraction_time:.2f}ms")
        
        # Update cache
        self.feature_cache[cache_key] = features
        
        return features
    
    async def _extract_price_features(
        self,
        features: FeatureVector,
        orderbook: OrderBookSnapshot,
        ticker: Optional[TickerData]
    ) -> None:
        """Extrai features baseadas em preço."""
        
        mid_price = orderbook.mid_price
        
        # Store current price
        key = f"{features.exchange}:{features.symbol}"
        if key not in self.price_history:
            self.price_history[key] = deque(maxlen=3600)  # 1 hour of seconds
        
        self.price_history[key].append((time.time(), mid_price))
        
        # Calculate returns
        prices = list(self.price_history[key])
        if len(prices) > 60:
            features.returns_1m = calculate_return(prices, 60)
        if len(prices) > 300:
            features.returns_5m = calculate_return(prices, 300)
        if len(prices) > 900:
            features.returns_15m = calculate_return(prices, 900)
        if len(prices) > 3600:
            features.returns_1h = calculate_return(prices, 3600)
        
        # Log returns
        if len(prices) > 1:
            features.log_returns = np.log(mid_price / prices[-2][1]) if prices[-2][1] > 0 else 0
            features.squared_returns = features.log_returns ** 2
        
        # Volatility
        if len(prices) > 60:
            features.volatility_1m = calculate_volatility(prices, 60)
        if len(prices) > 300:
            features.volatility_5m = calculate_volatility(prices, 300)
        if len(prices) > 900:
            features.volatility_15m = calculate_volatility(prices, 900)
        
        # Momentum, acceleration, jerk
        if len(prices) > 10:
            features.momentum = calculate_momentum(prices, 10)
            features.acceleration = calculate_acceleration(prices, 10)
            features.jerk = calculate_jerk(prices, 10)
    
    async def _extract_volume_features(
        self,
        features: FeatureVector,
        trades: Optional[List[TradeData]],
        ticker: Optional[TickerData]
    ) -> None:
        """Extrai features baseadas em volume."""
        
        if not trades:
            return
        
        # VWAP calculation
        vwap_sum = sum(t.price * t.volume for t in trades)
        volume_sum = sum(t.volume for t in trades)
        features.vwap = vwap_sum / volume_sum if volume_sum > 0 else 0
        
        # Volume profile
        volumes = [t.volume for t in trades]
        if volumes:
            features.volume_profile = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # Buy/Sell ratio
        buy_volume = sum(t.volume for t in trades if t.side == 'buy')
        sell_volume = sum(t.volume for t in trades if t.side == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            features.buy_sell_ratio = buy_volume / total_volume
            features.volume_imbalance = (buy_volume - sell_volume) / total_volume
        
        # Large order ratio (orders > 2x average)
        if volumes:
            avg_volume = np.mean(volumes)
            large_orders = [v for v in volumes if v > 2 * avg_volume]
            features.large_order_ratio = len(large_orders) / len(volumes)
        
        # Volume momentum
        key = f"{features.exchange}:{features.symbol}"
        if key not in self.volume_history:
            self.volume_history[key] = deque(maxlen=100)
        
        self.volume_history[key].append((time.time(), volume_sum))
        
        if len(self.volume_history[key]) > 10:
            recent_volumes = [v[1] for v in list(self.volume_history[key])[-10:]]
            features.volume_momentum = (recent_volumes[-1] - np.mean(recent_volumes[:-1])) / np.std(recent_volumes) if np.std(recent_volumes) > 0 else 0
        
        # Cumulative volume delta
        features.cumulative_volume_delta = buy_volume - sell_volume
        
        # Trade intensity
        time_span = trades[-1].timestamp - trades[0].timestamp if len(trades) > 1 else 1
        features.trade_intensity = len(trades) / max(time_span, 1)
        
        # Volume concentration (Herfindahl index)
        if volumes:
            volume_shares = [v / total_volume for v in volumes if total_volume > 0]
            features.volume_concentration = sum(s**2 for s in volume_shares)
    
    async def _extract_orderbook_features(
        self,
        features: FeatureVector,
        orderbook: OrderBookSnapshot
    ) -> None:
        """Extrai features do orderbook."""
        
        # Basic spread
        features.bid_ask_spread = orderbook.spread
        features.mid_price = orderbook.mid_price
        
        # Spread in basis points
        if orderbook.mid_price > 0:
            features.spread_bps = (orderbook.spread / orderbook.mid_price) * 10000
        
        # Micro price (weighted by size at best bid/ask)
        bid_price, bid_vol = orderbook.best_bid
        ask_price, ask_vol = orderbook.best_ask
        
        if bid_vol + ask_vol > 0:
            features.micro_price = (bid_price * ask_vol + ask_price * bid_vol) / (bid_vol + ask_vol)
        
        # Book imbalance
        features.book_imbalance = calculate_book_imbalance_fast(
            orderbook.bids,
            orderbook.asks,
            levels=10
        )
        
        # Book depth ratio
        bid_depth = np.sum(orderbook.bids[:10, 1]) if len(orderbook.bids) > 0 else 0
        ask_depth = np.sum(orderbook.asks[:10, 1]) if len(orderbook.asks) > 0 else 0
        
        if bid_depth + ask_depth > 0:
            features.book_depth_ratio = bid_depth / (bid_depth + ask_depth)
        
        # Order flow imbalance (using recent changes)
        features.order_flow_imbalance = features.book_imbalance * 0.7 + features.buy_sell_ratio * 0.3
        
        # Spoofing indicator (large orders far from mid)
        features.spoofing_indicator = detect_spoofing(orderbook.bids, orderbook.asks, orderbook.mid_price)
        
        # Liquidity score
        features.liquidity_score = calculate_liquidity_score(orderbook.bids, orderbook.asks)
        
        # Orderbook heat (concentration of orders)
        features.orderbook_heat = calculate_orderbook_heat(orderbook.bids, orderbook.asks)
        
        # Wall distances
        features.bid_wall_distance = find_wall_distance(orderbook.bids, 'bid')
        features.ask_wall_distance = find_wall_distance(orderbook.asks, 'ask')
    
    async def _extract_technical_indicators(
        self,
        features: FeatureVector,
        symbol: str
    ) -> None:
        """Extrai indicadores técnicos."""
        
        key = f"{features.exchange}:{symbol}"
        prices = self.price_history.get(key, [])
        
        if len(prices) < 200:
            return  # Not enough data for indicators
        
        # Convert to numpy array
        price_array = np.array([p[1] for p in prices])
        
        # RSI
        if len(price_array) >= self.rsi_period:
            features.rsi = talib.RSI(price_array, timeperiod=self.rsi_period)[-1]
        
        # MACD
        if len(price_array) >= self.macd_slow:
            macd, signal, hist = talib.MACD(
                price_array,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            features.macd = macd[-1] if len(macd) > 0 else 0
            features.macd_signal = signal[-1] if len(signal) > 0 else 0
            features.macd_histogram = hist[-1] if len(hist) > 0 else 0
        
        # Bollinger Bands
        if len(price_array) >= self.bollinger_period:
            upper, middle, lower = talib.BBANDS(
                price_array,
                timeperiod=self.bollinger_period,
                nbdevup=2,
                nbdevdn=2
            )
            features.bollinger_upper = upper[-1] if len(upper) > 0 else 0
            features.bollinger_middle = middle[-1] if len(middle) > 0 else 0
            features.bollinger_lower = lower[-1] if len(lower) > 0 else 0
        
        # ATR
        high = price_array * 1.001  # Approximate high
        low = price_array * 0.999   # Approximate low
        if len(price_array) >= self.atr_period:
            features.atr = talib.ATR(high, low, price_array, timeperiod=self.atr_period)[-1]
        
        # EMAs
        if len(price_array) >= 9:
            features.ema_9 = talib.EMA(price_array, timeperiod=9)[-1]
        if len(price_array) >= 21:
            features.ema_21 = talib.EMA(price_array, timeperiod=21)[-1]
        if len(price_array) >= 50:
            features.ema_50 = talib.EMA(price_array, timeperiod=50)[-1]
        if len(price_array) >= 200:
            features.ema_200 = talib.EMA(price_array, timeperiod=200)[-1]
        
        # Stochastic
        if len(price_array) >= 14:
            k, d = talib.STOCH(high, low, price_array, fastk_period=14)
            features.stochastic_k = k[-1] if len(k) > 0 else 0
            features.stochastic_d = d[-1] if len(d) > 0 else 0
        
        # Williams %R
        if len(price_array) >= 14:
            features.williams_r = talib.WILLR(high, low, price_array, timeperiod=14)[-1]
        
        # CCI
        if len(price_array) >= 14:
            features.cci = talib.CCI(high, low, price_array, timeperiod=14)[-1]
        
        # ADX
        if len(price_array) >= 14:
            features.adx = talib.ADX(high, low, price_array, timeperiod=14)[-1]
        
        # OBV (simplified)
        volumes = self.volume_history.get(key, [])
        if len(volumes) >= 20:
            volume_array = np.array([v[1] for v in volumes])
            features.obv = talib.OBV(price_array[-len(volume_array):], volume_array)[-1]
        
        # MFI
        if len(price_array) >= 14 and len(volumes) >= 14:
            features.mfi = talib.MFI(high[-14:], low[-14:], price_array[-14:], volume_array[-14:], timeperiod=14)[-1]
        
        # ROC
        if len(price_array) >= 10:
            features.roc = talib.ROC(price_array, timeperiod=10)[-1]
    
    async def _extract_network_features(self, features: FeatureVector) -> None:
        """Extrai features de rede blockchain."""
        
        # Use cached network data (in production, fetch from blockchain APIs)
        features.gas_price_gwei = self.network_data['gas_price_gwei']
        features.network_congestion = self.network_data['network_congestion']
        features.mempool_size = self.network_data['mempool_size']
        features.pending_transactions = self.network_data['pending_transactions']
        features.block_time = self.network_data['block_time']
        features.hash_rate = self.network_data['hash_rate']
    
    async def _extract_cross_asset_features(
        self,
        features: FeatureVector,
        symbol: str
    ) -> None:
        """Extrai features cross-asset."""
        
        # Market dominance
        features.btc_dominance = self.market_data['btc_dominance']
        features.market_sentiment = self.market_data['market_sentiment']
        
        # Correlations (simplified - in production, calculate from price series)
        if 'BTC' in symbol:
            features.correlation_btc = 1.0
            features.correlation_eth = 0.7
        elif 'ETH' in symbol:
            features.correlation_btc = 0.7
            features.correlation_eth = 1.0
        else:
            features.correlation_btc = 0.5
            features.correlation_eth = 0.4
        
        # Market cap and volume ratios (mock)
        features.market_cap_ratio = 0.05
        features.volume_ratio = 0.03
        
        # Funding rate and open interest (mock for spot trading)
        features.funding_rate = 0.0001
        features.open_interest = 1000000
    
    async def _extract_statistical_features(
        self,
        features: FeatureVector,
        orderbook: OrderBookSnapshot
    ) -> None:
        """Extrai features estatísticas."""
        
        key = f"{features.exchange}:{features.symbol}"
        prices = self.price_history.get(key, [])
        
        if len(prices) < 30:
            return
        
        price_array = np.array([p[1] for p in prices])
        returns = np.diff(np.log(price_array))
        
        # Z-score
        if len(returns) > 0:
            features.zscore = (returns[-1] - np.mean(returns)) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Entropy
        features.entropy = calculate_entropy(returns)
        
        # Autocorrelation
        if len(returns) > 10:
            features.autocorrelation = calculate_autocorrelation(returns, lag=1)
        
        # Hurst exponent
        if len(price_array) > 100:
            features.hurst_exponent = calculate_hurst_exponent(price_array)
        
        # Kurtosis and Skewness
        if len(returns) > 0:
            features.kurtosis = stats.kurtosis(returns)
            features.skewness = stats.skew(returns)
        
        # Price efficiency (deviation from random walk)
        features.price_efficiency = calculate_price_efficiency(returns)
        
        # Information ratio (simplified)
        if len(returns) > 0:
            features.information_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    def _calculate_feature_confidence(self, features: FeatureVector) -> float:
        """Calcula confiança nas features extraídas."""
        
        confidence_factors = []
        
        # Data freshness
        age_confidence = max(0, 1 - (features.data_age_ms / 1000))
        confidence_factors.append(age_confidence)
        
        # Latency confidence
        latency_confidence = max(0, 1 - (features.latency_ms / 30))
        confidence_factors.append(latency_confidence)
        
        # Feature completeness
        feature_array = features.to_array()
        non_zero_ratio = np.count_nonzero(feature_array) / len(feature_array)
        confidence_factors.append(non_zero_ratio)
        
        return np.mean(confidence_factors)
    
    def _calculate_feature_quality(self, features: FeatureVector) -> float:
        """Calcula qualidade das features."""
        
        feature_array = features.to_array()
        
        # Check for NaN or Inf
        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            return 0.0
        
        # Check variance (features should have some variance)
        if np.std(feature_array) == 0:
            return 0.5
        
        # Check range (features should be normalized)
        if np.max(np.abs(feature_array)) > 1000:
            return 0.7
        
        return 1.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância das features (mock for now).
        Em produção, isso viria do modelo treinado.
        """
        return {
            'book_imbalance': 0.15,
            'order_flow_imbalance': 0.12,
            'rsi': 0.10,
            'volume_imbalance': 0.09,
            'volatility_5m': 0.08,
            'micro_price': 0.07,
            'macd': 0.06,
            'momentum': 0.05,
            'vwap': 0.04,
            'spread_bps': 0.04,
            'others': 0.20
        }


# Optimized helper functions using Numba
@jit(nopython=True)
def calculate_book_imbalance_fast(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> float:
    """Calcula imbalance do orderbook usando Numba."""
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    bid_volume = 0.0
    ask_volume = 0.0
    
    for i in range(min(levels, len(bids))):
        bid_volume += bids[i, 1]
    
    for i in range(min(levels, len(asks))):
        ask_volume += asks[i, 1]
    
    total = bid_volume + ask_volume
    return (bid_volume - ask_volume) / total if total > 0 else 0.0


@jit(nopython=True)
def calculate_weighted_spread(bids: np.ndarray, asks: np.ndarray, depth: int = 5) -> float:
    """Calcula spread ponderado por volume."""
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    weighted_bid = 0.0
    weighted_ask = 0.0
    bid_weight = 0.0
    ask_weight = 0.0
    
    for i in range(min(depth, len(bids))):
        weighted_bid += bids[i, 0] * bids[i, 1]
        bid_weight += bids[i, 1]
    
    for i in range(min(depth, len(asks))):
        weighted_ask += asks[i, 0] * asks[i, 1]
        ask_weight += asks[i, 1]
    
    if bid_weight == 0 or ask_weight == 0:
        return 0.0
    
    return (weighted_ask / ask_weight) - (weighted_bid / bid_weight)


def calculate_return(prices: List[Tuple[float, float]], period: int) -> float:
    """Calcula retorno para um período."""
    if len(prices) < 2:
        return 0.0
    
    current_price = prices[-1][1]
    
    # Find price from period seconds ago
    target_time = prices[-1][0] - period
    past_price = None
    
    for timestamp, price in reversed(prices[:-1]):
        if timestamp <= target_time:
            past_price = price
            break
    
    if past_price and past_price > 0:
        return (current_price - past_price) / past_price
    
    return 0.0


def calculate_volatility(prices: List[Tuple[float, float]], period: int) -> float:
    """Calcula volatilidade para um período."""
    if len(prices) < 2:
        return 0.0
    
    # Get prices from last period seconds
    target_time = prices[-1][0] - period
    period_prices = [p[1] for t, p in prices if t >= target_time]
    
    if len(period_prices) < 2:
        return 0.0
    
    returns = np.diff(np.log(period_prices))
    return np.std(returns) * np.sqrt(252 * 24 * 3600 / period)  # Annualized


def calculate_momentum(prices: List[Tuple[float, float]], period: int) -> float:
    """Calcula momentum."""
    if len(prices) < period:
        return 0.0
    
    recent_prices = [p[1] for p in prices[-period:]]
    return (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0


def calculate_acceleration(prices: List[Tuple[float, float]], period: int) -> float:
    """Calcula aceleração (segunda derivada)."""
    if len(prices) < period:
        return 0.0
    
    momentums = []
    for i in range(len(prices) - period + 1):
        window = prices[i:i+period]
        mom = calculate_momentum(window, min(5, period))
        momentums.append(mom)
    
    if len(momentums) < 2:
        return 0.0
    
    return momentums[-1] - momentums[-2]


def calculate_jerk(prices: List[Tuple[float, float]], period: int) -> float:
    """Calcula jerk (terceira derivada)."""
    if len(prices) < period + 5:
        return 0.0
    
    accelerations = []
    for i in range(len(prices) - period):
        window = prices[i:i+period+5]
        acc = calculate_acceleration(window, period)
        accelerations.append(acc)
    
    if len(accelerations) < 2:
        return 0.0
    
    return accelerations[-1] - accelerations[-2]


def detect_spoofing(bids: np.ndarray, asks: np.ndarray, mid_price: float) -> float:
    """Detecta possível spoofing no orderbook."""
    if len(bids) < 10 or len(asks) < 10 or mid_price <= 0:
        return 0.0
    
    # Look for large orders far from mid price
    bid_distances = np.abs(bids[:, 0] - mid_price) / mid_price
    ask_distances = np.abs(asks[:, 0] - mid_price) / mid_price
    
    bid_volumes = bids[:, 1]
    ask_volumes = asks[:, 1]
    
    # Find outlier volumes
    bid_outliers = bid_volumes > np.percentile(bid_volumes, 90)
    ask_outliers = ask_volumes > np.percentile(ask_volumes, 90)
    
    # Check if outliers are far from mid
    bid_spoof = np.any(bid_outliers & (bid_distances > 0.005))  # >0.5% from mid
    ask_spoof = np.any(ask_outliers & (ask_distances > 0.005))
    
    return float(bid_spoof or ask_spoof)


def calculate_liquidity_score(bids: np.ndarray, asks: np.ndarray) -> float:
    """Calcula score de liquidez."""
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    # Factors: depth, tightness, resilience
    depth_score = min(1.0, (np.sum(bids[:5, 1]) + np.sum(asks[:5, 1])) / 1000)
    
    spread = asks[0, 0] - bids[0, 0] if len(asks) > 0 and len(bids) > 0 else float('inf')
    mid = (asks[0, 0] + bids[0, 0]) / 2 if spread < float('inf') else 1
    tightness_score = max(0, 1 - (spread / mid) * 100) if mid > 0 else 0
    
    # Resilience (how quickly orderbook fills)
    volume_gradient_bid = np.gradient(bids[:10, 1]) if len(bids) >= 10 else np.array([0])
    volume_gradient_ask = np.gradient(asks[:10, 1]) if len(asks) >= 10 else np.array([0])
    resilience_score = min(1.0, np.mean(np.abs(np.concatenate([volume_gradient_bid, volume_gradient_ask]))) / 10)
    
    return (depth_score + tightness_score + resilience_score) / 3


def calculate_orderbook_heat(bids: np.ndarray, asks: np.ndarray) -> float:
    """Calcula 'heat' do orderbook (concentração de ordens)."""
    if len(bids) < 5 or len(asks) < 5:
        return 0.0
    
    # Calculate concentration using Gini coefficient
    bid_volumes = bids[:20, 1] if len(bids) >= 20 else bids[:, 1]
    ask_volumes = asks[:20, 1] if len(asks) >= 20 else asks[:, 1]
    
    all_volumes = np.concatenate([bid_volumes, ask_volumes])
    
    if len(all_volumes) == 0 or np.sum(all_volumes) == 0:
        return 0.0
    
    sorted_volumes = np.sort(all_volumes)
    n = len(sorted_volumes)
    index = np.arange(1, n + 1)
    
    return (2 * np.sum(index * sorted_volumes)) / (n * np.sum(sorted_volumes)) - (n + 1) / n


def find_wall_distance(orders: np.ndarray, side: str) -> float:
    """Encontra distância até a próxima 'wall' de ordens."""
    if len(orders) < 10:
        return 0.0
    
    volumes = orders[:20, 1] if len(orders) >= 20 else orders[:, 1]
    
    # Find peaks (walls)
    mean_vol = np.mean(volumes)
    std_vol = np.std(volumes)
    
    threshold = mean_vol + 2 * std_vol
    
    for i, vol in enumerate(volumes):
        if vol > threshold:
            return i + 1  # Distance in levels
    
    return 20.0  # No wall found


def calculate_entropy(returns: np.ndarray) -> float:
    """Calcula entropia de Shannon dos retornos."""
    if len(returns) == 0:
        return 0.0
    
    # Discretize returns into bins
    hist, _ = np.histogram(returns, bins=10)
    probs = hist / np.sum(hist)
    
    # Remove zero probabilities
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    return -np.sum(probs * np.log2(probs))


def calculate_autocorrelation(returns: np.ndarray, lag: int = 1) -> float:
    """Calcula autocorrelação."""
    if len(returns) <= lag:
        return 0.0
    
    return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]


def calculate_hurst_exponent(prices: np.ndarray) -> float:
    """Calcula expoente de Hurst (simplicado)."""
    if len(prices) < 100:
        return 0.5
    
    lags = range(2, min(100, len(prices) // 2))
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    
    # Linear fit to log-log plot
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return reg[0]


def calculate_price_efficiency(returns: np.ndarray) -> float:
    """Calcula eficiência de preço (desvio de random walk)."""
    if len(returns) < 10:
        return 0.5
    
    # Variance ratio test
    var_1 = np.var(returns)
    var_n = np.var(np.sum(returns.reshape(-1, min(5, len(returns) // 2)), axis=1))
    
    if var_1 == 0:
        return 0.5
    
    ratio = var_n / (5 * var_1)
    
    # Efficiency: 1 = random walk, <1 = mean reverting, >1 = trending
    return min(2.0, max(0.0, ratio))