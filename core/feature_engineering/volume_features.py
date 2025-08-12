"""
Volume Features Module - Extração de features baseadas em volume
Calcula VWAP, volume profile, imbalance e métricas de liquidez
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from numba import jit, prange
import logging
from dataclasses import dataclass

from core.data_collector.exchange_connector import TradeData

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_vwap_fast(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Calcula VWAP (Volume Weighted Average Price) usando Numba.
    
    Args:
        prices: Array de preços
        volumes: Array de volumes
        
    Returns:
        VWAP
    """
    if len(prices) == 0 or len(volumes) == 0:
        return 0.0
    
    total_value = 0.0
    total_volume = 0.0
    
    for i in prange(len(prices)):
        total_value += prices[i] * volumes[i]
        total_volume += volumes[i]
    
    return total_value / total_volume if total_volume > 0 else 0.0


@jit(nopython=True)
def calculate_volume_imbalance_fast(buy_volumes: np.ndarray, sell_volumes: np.ndarray) -> float:
    """
    Calcula imbalance de volume buy/sell.
    
    Args:
        buy_volumes: Volumes de compra
        sell_volumes: Volumes de venda
        
    Returns:
        Imbalance ratio [-1, 1]
    """
    total_buy = np.sum(buy_volumes)
    total_sell = np.sum(sell_volumes)
    total = total_buy + total_sell
    
    if total == 0:
        return 0.0
    
    return (total_buy - total_sell) / total


@jit(nopython=True)
def calculate_cumulative_volume_delta(buy_volumes: np.ndarray, sell_volumes: np.ndarray) -> float:
    """
    Calcula CVD (Cumulative Volume Delta).
    
    Args:
        buy_volumes: Volumes de compra
        sell_volumes: Volumes de venda
        
    Returns:
        CVD
    """
    cvd = 0.0
    for i in range(len(buy_volumes)):
        cvd += buy_volumes[i] - sell_volumes[i] if i < len(sell_volumes) else buy_volumes[i]
    
    return cvd


@jit(nopython=True)
def calculate_volume_profile(volumes: np.ndarray, prices: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula perfil de volume (distribuição de volume por faixa de preço).
    
    Args:
        volumes: Array de volumes
        prices: Array de preços
        n_bins: Número de bins para o histograma
        
    Returns:
        (price_levels, volume_distribution)
    """
    if len(volumes) == 0 or len(prices) == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    
    price_min = np.min(prices)
    price_max = np.max(prices)
    
    if price_min == price_max:
        return np.array([price_min]), np.array([np.sum(volumes)])
    
    # Create price bins
    price_levels = np.linspace(price_min, price_max, n_bins)
    volume_dist = np.zeros(n_bins)
    bin_width = (price_max - price_min) / (n_bins - 1)
    
    # Accumulate volume in each bin
    for i in range(len(prices)):
        bin_idx = int((prices[i] - price_min) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        volume_dist[bin_idx] += volumes[i]
    
    return price_levels, volume_dist


@jit(nopython=True)
def calculate_trade_intensity(trade_counts: np.ndarray, time_window: float) -> float:
    """
    Calcula intensidade de trades (trades por segundo).
    
    Args:
        trade_counts: Contagem de trades por período
        time_window: Janela de tempo em segundos
        
    Returns:
        Trade intensity
    """
    if time_window <= 0:
        return 0.0
    
    return np.sum(trade_counts) / time_window


@jit(nopython=True)
def calculate_large_order_ratio(volumes: np.ndarray, threshold_percentile: float = 90.0) -> float:
    """
    Calcula proporção de ordens grandes.
    
    Args:
        volumes: Array de volumes
        threshold_percentile: Percentil para definir ordem grande
        
    Returns:
        Ratio de ordens grandes
    """
    if len(volumes) == 0:
        return 0.0
    
    threshold = np.percentile(volumes, threshold_percentile)
    large_orders = volumes[volumes >= threshold]
    
    return len(large_orders) / len(volumes)


@jit(nopython=True)
def calculate_volume_momentum(volumes: np.ndarray, lookback: int = 20) -> float:
    """
    Calcula momentum de volume.
    
    Args:
        volumes: Array de volumes
        lookback: Período de lookback
        
    Returns:
        Volume momentum
    """
    if len(volumes) < lookback * 2:
        return 0.0
    
    recent_volume = np.mean(volumes[-lookback:])
    previous_volume = np.mean(volumes[-lookback*2:-lookback])
    
    if previous_volume == 0:
        return 0.0
    
    return (recent_volume - previous_volume) / previous_volume


class VolumeFeatureExtractor:
    """
    Extrator especializado para features de volume.
    """
    
    def __init__(self):
        """Inicializa extrator de features de volume."""
        self.trade_buffer = {}
        self.volume_history = {}
        self.max_buffer_size = 1000
        
    def extract_volume_features(
        self,
        trades: List[TradeData],
        orderbook_volumes: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Extrai todas as features relacionadas a volume.
        
        Args:
            trades: Lista de trades recentes
            orderbook_volumes: (bid_volumes, ask_volumes) do orderbook
            
        Returns:
            Dicionário com features de volume
        """
        if not trades:
            return self._get_default_features()
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        
        # Extract arrays
        all_prices = np.array([t.price for t in trades])
        all_volumes = np.array([t.volume for t in trades])
        buy_volumes = np.array([t.volume for t in buy_trades]) if buy_trades else np.array([0.0])
        sell_volumes = np.array([t.volume for t in sell_trades]) if sell_trades else np.array([0.0])
        
        # Calculate VWAP
        vwap = calculate_vwap_fast(all_prices, all_volumes)
        
        # Volume imbalance
        volume_imbalance = calculate_volume_imbalance_fast(buy_volumes, sell_volumes)
        buy_sell_ratio = len(buy_trades) / len(trades) if trades else 0.5
        
        # CVD
        cvd = calculate_cumulative_volume_delta(buy_volumes, sell_volumes)
        
        # Volume profile
        price_levels, volume_dist = calculate_volume_profile(all_volumes, all_prices)
        volume_profile_score = self._calculate_volume_concentration(volume_dist)
        
        # Trade intensity
        if trades:
            time_span = trades[-1].timestamp - trades[0].timestamp
            trade_intensity = len(trades) / max(time_span, 1.0)
        else:
            trade_intensity = 0.0
        
        # Large order ratio
        large_order_ratio = calculate_large_order_ratio(all_volumes)
        
        # Volume momentum
        volume_momentum = calculate_volume_momentum(all_volumes)
        
        # Volume weighted spread
        volume_weighted_spread = self._calculate_volume_weighted_spread(
            all_prices, all_volumes, vwap
        )
        
        # Volume concentration
        volume_concentration = self._calculate_volume_herfindahl(all_volumes)
        
        return {
            'vwap': vwap,
            'volume_profile': volume_profile_score,
            'volume_imbalance': volume_imbalance,
            'buy_sell_ratio': buy_sell_ratio,
            'large_order_ratio': large_order_ratio,
            'volume_momentum': volume_momentum,
            'cumulative_volume_delta': cvd,
            'volume_weighted_spread': volume_weighted_spread,
            'trade_intensity': trade_intensity,
            'volume_concentration': volume_concentration
        }
    
    def _calculate_volume_concentration(self, volume_dist: np.ndarray) -> float:
        """
        Calcula concentração de volume (0 = disperso, 1 = concentrado).
        
        Args:
            volume_dist: Distribuição de volume
            
        Returns:
            Score de concentração
        """
        if len(volume_dist) == 0:
            return 0.0
        
        total_volume = np.sum(volume_dist)
        if total_volume == 0:
            return 0.0
        
        # Normalize distribution
        normalized = volume_dist / total_volume
        
        # Calculate entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(len(volume_dist))
        
        # Convert to concentration (inverse of entropy)
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def _calculate_volume_weighted_spread(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        vwap: float
    ) -> float:
        """
        Calcula spread ponderado por volume.
        
        Args:
            prices: Array de preços
            volumes: Array de volumes
            vwap: Volume weighted average price
            
        Returns:
            Volume weighted spread
        """
        if len(prices) == 0 or vwap == 0:
            return 0.0
        
        weighted_deviations = np.abs(prices - vwap) * volumes
        total_volume = np.sum(volumes)
        
        if total_volume == 0:
            return 0.0
        
        return np.sum(weighted_deviations) / total_volume
    
    def _calculate_volume_herfindahl(self, volumes: np.ndarray) -> float:
        """
        Calcula Herfindahl Index para concentração de volume.
        
        Args:
            volumes: Array de volumes
            
        Returns:
            Herfindahl index [0, 1]
        """
        if len(volumes) == 0:
            return 0.0
        
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return 0.0
        
        market_shares = volumes / total_volume
        hhi = np.sum(market_shares ** 2)
        
        return hhi
    
    def _get_default_features(self) -> Dict[str, float]:
        """Retorna features padrão quando não há dados suficientes."""
        return {
            'vwap': 0.0,
            'volume_profile': 0.0,
            'volume_imbalance': 0.0,
            'buy_sell_ratio': 0.5,
            'large_order_ratio': 0.0,
            'volume_momentum': 0.0,
            'cumulative_volume_delta': 0.0,
            'volume_weighted_spread': 0.0,
            'trade_intensity': 0.0,
            'volume_concentration': 0.0
        }


# Additional utility functions
def calculate_obv(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Calcula On-Balance Volume (OBV).
    
    Args:
        prices: Array de preços
        volumes: Array de volumes
        
    Returns:
        OBV value
    """
    if len(prices) < 2:
        return 0.0
    
    obv = 0.0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
    
    return obv


def calculate_volume_rsi(volumes: np.ndarray, period: int = 14) -> float:
    """
    Calcula RSI baseado em volume.
    
    Args:
        volumes: Array de volumes
        period: Período do RSI
        
    Returns:
        Volume RSI [0, 100]
    """
    if len(volumes) < period + 1:
        return 50.0
    
    gains = []
    losses = []
    
    for i in range(1, len(volumes)):
        change = volumes[i] - volumes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi