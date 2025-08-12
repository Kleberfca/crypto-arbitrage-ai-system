"""
Network Features Module - Features de rede blockchain e mercado
Gas price, network congestion, mempool, dominance e correlações
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from numba import jit
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_correlation_fast(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula correlação entre dois arrays usando Numba.
    
    Args:
        x: Primeiro array
        y: Segundo array
        
    Returns:
        Correlação [-1, 1]
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    # Remove mean
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Calculate correlation
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


@jit(nopython=True)
def calculate_network_stress(
    gas_price: float,
    mempool_size: float,
    pending_txs: float,
    block_time: float
) -> float:
    """
    Calcula stress da rede baseado em múltiplos fatores.
    
    Args:
        gas_price: Preço do gas em Gwei
        mempool_size: Tamanho do mempool
        pending_txs: Transações pendentes
        block_time: Tempo médio de bloco
        
    Returns:
        Network stress score [0, 1]
    """
    # Normalize each factor
    gas_stress = min(gas_price / 100, 1.0)  # 100 Gwei as max
    mempool_stress = min(mempool_size / 500000, 1.0)  # 500k as max
    pending_stress = min(pending_txs / 10000, 1.0)  # 10k as max
    block_stress = min(max(block_time - 3, 0) / 10, 1.0)  # 3s ideal, 13s max
    
    # Weighted average
    stress = (
        gas_stress * 0.4 +
        mempool_stress * 0.2 +
        pending_stress * 0.2 +
        block_stress * 0.2
    )
    
    return min(stress, 1.0)


class NetworkFeatureExtractor:
    """
    Extrator de features de rede blockchain e mercado.
    """
    
    def __init__(self):
        """Inicializa extrator de features de rede."""
        # Network data cache
        self.network_cache = {
            'gas_price_gwei': 25.0,
            'network_congestion': 0.3,
            'mempool_size': 150000,
            'pending_transactions': 2500,
            'block_time': 3.0,
            'hash_rate': 500e18
        }
        
        # Market data cache
        self.market_cache = {
            'btc_dominance': 0.48,
            'total_market_cap': 2.5e12,
            'btc_price': 50000,
            'eth_price': 3000,
            'market_sentiment': 0.65,
            'fear_greed_index': 65
        }
        
        # Cross-asset data
        self.correlation_cache = {}
        self.price_history = {}
        self.funding_rates = {}
        self.open_interest = {}
        
        # Update intervals
        self.last_network_update = 0
        self.last_market_update = 0
        self.update_interval = 60  # seconds
        
    async def extract_network_features(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, float]:
        """
        Extrai features relacionadas à rede blockchain.
        
        Args:
            symbol: Símbolo do ativo
            exchange: Exchange
            
        Returns:
            Dicionário com features de rede
        """
        # Update network data if stale
        if time.time() - self.last_network_update > self.update_interval:
            await self._update_network_data()
        
        features = {
            'gas_price_gwei': self.network_cache['gas_price_gwei'],
            'network_congestion': self.network_cache['network_congestion'],
            'mempool_size': self.network_cache['mempool_size'],
            'pending_transactions': self.network_cache['pending_transactions'],
            'block_time': self.network_cache['block_time'],
            'hash_rate': self.network_cache['hash_rate']
        }
        
        # Calculate network stress
        features['network_stress'] = calculate_network_stress(
            features['gas_price_gwei'],
            features['mempool_size'],
            features['pending_transactions'],
            features['block_time']
        )
        
        # Transaction cost estimate (for arbitrage)
        features['transfer_cost_usd'] = self._estimate_transfer_cost(
            symbol,
            features['gas_price_gwei']
        )
        
        return features
    
    async def extract_cross_asset_features(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, float]:
        """
        Extrai features de correlação cross-asset.
        
        Args:
            symbol: Símbolo do ativo
            current_price: Preço atual
            
        Returns:
            Dicionário com features cross-asset
        """
        # Update market data if stale
        if time.time() - self.last_market_update > self.update_interval:
            await self._update_market_data()
        
        features = {
            'btc_dominance': self.market_cache['btc_dominance'],
            'market_sentiment': self.market_cache['market_sentiment']
        }
        
        # Calculate correlations
        features['correlation_btc'] = await self._calculate_correlation(symbol, 'BTC')
        features['correlation_eth'] = await self._calculate_correlation(symbol, 'ETH')
        
        # Market cap ratio
        features['market_cap_ratio'] = self._calculate_market_cap_ratio(symbol)
        
        # Volume ratio
        features['volume_ratio'] = self._calculate_volume_ratio(symbol)
        
        # Funding rate (for perpetuals)
        features['funding_rate'] = self.funding_rates.get(symbol, 0.0)
        
        # Open interest
        features['open_interest'] = self.open_interest.get(symbol, 0.0)
        
        # Beta to market
        features['market_beta'] = self._calculate_market_beta(symbol)
        
        return features
    
    async def extract_statistical_features(
        self,
        prices: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extrai features estatísticas avançadas.
        
        Args:
            prices: Array de preços
            returns: Array de retornos (opcional)
            
        Returns:
            Dicionário com features estatísticas
        """
        if len(prices) < 2:
            return self._get_default_statistical_features()
        
        if returns is None:
            returns = np.diff(np.log(prices))
        
        features = {}
        
        # Z-score
        features['zscore'] = self._calculate_zscore(prices)
        
        # Entropy
        features['entropy'] = self._calculate_entropy(returns)
        
        # Autocorrelation
        features['autocorrelation'] = self._calculate_autocorrelation(returns)
        
        # Hurst exponent
        features['hurst_exponent'] = self._calculate_hurst_exponent(prices)
        
        # Kurtosis
        features['kurtosis'] = float(np.nan_to_num(np.kurtosis(returns), 0))
        
        # Skewness
        features['skewness'] = float(np.nan_to_num(np.skew(returns), 0))
        
        # Price efficiency
        features['price_efficiency'] = self._calculate_price_efficiency(prices)
        
        # Information ratio
        features['information_ratio'] = self._calculate_information_ratio(returns)
        
        return features
    
    async def _update_network_data(self) -> None:
        """Atualiza dados da rede blockchain."""
        try:
            # In production, fetch from real APIs
            # For now, simulate with realistic variations
            self.network_cache['gas_price_gwei'] = np.random.uniform(15, 50)
            self.network_cache['network_congestion'] = np.random.uniform(0.1, 0.8)
            self.network_cache['mempool_size'] = np.random.uniform(50000, 300000)
            self.network_cache['pending_transactions'] = np.random.uniform(1000, 5000)
            self.network_cache['block_time'] = np.random.uniform(2.5, 4.0)
            self.network_cache['hash_rate'] = np.random.uniform(400e18, 600e18)
            
            self.last_network_update = time.time()
        except Exception as e:
            logger.error(f"Failed to update network data: {e}")
    
    async def _update_market_data(self) -> None:
        """Atualiza dados de mercado."""
        try:
            # In production, fetch from real APIs
            # For now, simulate with realistic variations
            self.market_cache['btc_dominance'] = np.random.uniform(0.45, 0.55)
            self.market_cache['total_market_cap'] = np.random.uniform(2e12, 3e12)
            self.market_cache['btc_price'] = np.random.uniform(45000, 55000)
            self.market_cache['eth_price'] = np.random.uniform(2500, 3500)
            self.market_cache['market_sentiment'] = np.random.uniform(0.3, 0.8)
            self.market_cache['fear_greed_index'] = np.random.uniform(20, 80)
            
            self.last_market_update = time.time()
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
    
    def _estimate_transfer_cost(self, symbol: str, gas_price: float) -> float:
        """
        Estima custo de transferência em USD.
        
        Args:
            symbol: Símbolo do ativo
            gas_price: Preço do gas em Gwei
            
        Returns:
            Custo estimado em USD
        """
        # BEP20 transfer gas usage
        gas_limit = 65000
        
        # Convert gas price to BNB
        gas_cost_bnb = (gas_price * gas_limit) / 1e9
        
        # Assume BNB price (in production, fetch real price)
        bnb_price = 300
        
        return gas_cost_bnb * bnb_price
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Calcula correlação entre dois ativos.
        
        Args:
            symbol1: Primeiro símbolo
            symbol2: Segundo símbolo
            
        Returns:
            Correlação [-1, 1]
        """
        # Check cache
        cache_key = f"{symbol1}_{symbol2}"
        if cache_key in self.correlation_cache:
            cached = self.correlation_cache[cache_key]
            if time.time() - cached['timestamp'] < 300:  # 5 min cache
                return cached['value']
        
        # In production, calculate from real price history
        # For now, return simulated correlation
        if symbol1 == symbol2:
            correlation = 1.0
        elif 'BTC' in symbol1 or 'BTC' in symbol2:
            correlation = np.random.uniform(0.6, 0.9)
        elif 'ETH' in symbol1 or 'ETH' in symbol2:
            correlation = np.random.uniform(0.5, 0.8)
        else:
            correlation = np.random.uniform(0.2, 0.6)
        
        # Cache result
        self.correlation_cache[cache_key] = {
            'value': correlation,
            'timestamp': time.time()
        }
        
        return correlation
    
    def _calculate_market_cap_ratio(self, symbol: str) -> float:
        """
        Calcula ratio de market cap vs mercado total.
        
        Args:
            symbol: Símbolo do ativo
            
        Returns:
            Market cap ratio
        """
        # In production, fetch real market cap
        # For now, simulate based on symbol
        if 'BTC' in symbol:
            return 0.48
        elif 'ETH' in symbol:
            return 0.18
        elif 'BNB' in symbol:
            return 0.04
        else:
            return np.random.uniform(0.001, 0.01)
    
    def _calculate_volume_ratio(self, symbol: str) -> float:
        """
        Calcula ratio de volume vs volume total do mercado.
        
        Args:
            symbol: Símbolo do ativo
            
        Returns:
            Volume ratio
        """
        # In production, fetch real volume data
        # For now, simulate
        if 'BTC' in symbol:
            return 0.35
        elif 'ETH' in symbol:
            return 0.20
        else:
            return np.random.uniform(0.001, 0.05)
    
    def _calculate_market_beta(self, symbol: str) -> float:
        """
        Calcula beta em relação ao mercado.
        
        Args:
            symbol: Símbolo do ativo
            
        Returns:
            Beta value
        """
        # In production, calculate from historical data
        # For now, simulate based on symbol
        if 'BTC' in symbol:
            return 1.0
        elif 'ETH' in symbol:
            return 1.2
        elif any(stable in symbol for stable in ['USDT', 'USDC', 'BUSD']):
            return 0.0
        else:
            return np.random.uniform(0.8, 1.5)
    
    def _calculate_zscore(self, prices: np.ndarray, lookback: int = 20) -> float:
        """
        Calcula z-score do preço atual.
        
        Args:
            prices: Array de preços
            lookback: Período de lookback
            
        Returns:
            Z-score
        """
        if len(prices) < lookback:
            return 0.0
        
        recent_prices = prices[-lookback:]
        mean = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        if std == 0:
            return 0.0
        
        return (prices[-1] - mean) / std
    
    def _calculate_entropy(self, returns: np.ndarray, bins: int = 10) -> float:
        """
        Calcula entropia de Shannon dos retornos.
        
        Args:
            returns: Array de retornos
            bins: Número de bins para histograma
            
        Returns:
            Entropy value
        """
        if len(returns) < bins:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(returns, bins=bins)
        
        # Calculate probabilities
        probs = hist / len(returns)
        probs = probs[probs > 0]  # Remove zeros
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """
        Calcula autocorrelação dos retornos.
        
        Args:
            returns: Array de retornos
            lag: Lag para autocorrelação
            
        Returns:
            Autocorrelation coefficient
        """
        if len(returns) < lag + 1:
            return 0.0
        
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calcula expoente de Hurst (persistência de tendência).
        
        Args:
            prices: Array de preços
            
        Returns:
            Hurst exponent
        """
        if len(prices) < 100:
            return 0.5  # Random walk
        
        # Simplified R/S analysis
        lags = range(2, min(100, len(prices) // 2))
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        
        # Linear fit to log-log plot
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return reg[0]
    
    def _calculate_price_efficiency(self, prices: np.ndarray) -> float:
        """
        Calcula eficiência de preço (variance ratio test).
        
        Args:
            prices: Array de preços
            
        Returns:
            Price efficiency score
        """
        if len(prices) < 48:
            return 0.5
        
        # Calculate variance ratios
        returns_1 = np.diff(np.log(prices))
        returns_2 = np.diff(np.log(prices[::2]))
        returns_4 = np.diff(np.log(prices[::4]))
        
        var_1 = np.var(returns_1)
        var_2 = np.var(returns_2)
        var_4 = np.var(returns_4)
        
        if var_1 == 0:
            return 1.0
        
        vr_2 = var_2 / (2 * var_1)
        vr_4 = var_4 / (4 * var_1)
        
        # Efficiency score (closer to 1 = more efficient)
        efficiency = 1 - abs(1 - (vr_2 + vr_4) / 2)
        
        return max(0, min(1, efficiency))
    
    def _calculate_information_ratio(self, returns: np.ndarray, benchmark_return: float = 0.0) -> float:
        """
        Calcula information ratio.
        
        Args:
            returns: Array de retornos
            benchmark_return: Retorno do benchmark
            
        Returns:
            Information ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_return
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365 * 24 * 60)  # Annualized
    
    def _get_default_statistical_features(self) -> Dict[str, float]:
        """Retorna features estatísticas padrão."""
        return {
            'zscore': 0.0,
            'entropy': 0.0,
            'autocorrelation': 0.0,
            'hurst_exponent': 0.5,
            'kurtosis': 0.0,
            'skewness': 0.0,
            'price_efficiency': 0.5,
            'information_ratio': 0.0
        }


# Utility functions
import time


def calculate_market_regime(
    volatility: float,
    volume: float,
    trend: float
) -> str:
    """
    Identifica regime de mercado.
    
    Args:
        volatility: Volatilidade atual
        volume: Volume relativo
        trend: Força da tendência
        
    Returns:
        Market regime: 'trending', 'ranging', 'volatile', 'quiet'
    """
    if volatility > 0.03 and abs(trend) > 0.02:
        return 'volatile'
    elif abs(trend) > 0.01:
        return 'trending'
    elif volatility < 0.01:
        return 'quiet'
    else:
        return 'ranging'