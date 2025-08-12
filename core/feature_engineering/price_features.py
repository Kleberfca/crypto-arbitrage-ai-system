"""
Price Features Module - Extração de features baseadas em preço
Calcula returns, volatility, momentum e outras métricas de preço
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from numba import jit, prange
import logging
from collections import deque

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_returns_fast(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calcula retornos para múltiplos períodos usando Numba.
    
    Args:
        prices: Array de preços
        periods: Períodos para calcular retornos [1min, 5min, 15min, 1h]
        
    Returns:
        Array de retornos para cada período
    """
    n_periods = len(periods)
    returns = np.zeros(n_periods)
    
    if len(prices) < 2:
        return returns
    
    current_price = prices[-1]
    
    for i in prange(n_periods):
        period_idx = int(periods[i])
        if period_idx < len(prices):
            past_price = prices[-period_idx-1]
            if past_price > 0:
                returns[i] = (current_price - past_price) / past_price
    
    return returns


@jit(nopython=True)
def calculate_log_returns(prices: np.ndarray) -> float:
    """
    Calcula log returns.
    
    Args:
        prices: Array de preços
        
    Returns:
        Log return
    """
    if len(prices) < 2:
        return 0.0
    
    if prices[-2] > 0 and prices[-1] > 0:
        return np.log(prices[-1] / prices[-2])
    
    return 0.0


@jit(nopython=True)
def calculate_volatility_fast(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Calcula volatilidade histórica usando Numba.
    
    Args:
        returns: Array de retornos
        annualize: Se deve anualizar a volatilidade
        
    Returns:
        Volatilidade
    """
    if len(returns) < 2:
        return 0.0
    
    vol = np.std(returns)
    
    if annualize:
        # Assuming crypto trades 24/7
        vol *= np.sqrt(365 * 24 * 60)  # Annualized for minute data
    
    return vol


@jit(nopython=True)
def calculate_momentum_indicators(prices: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula momentum, aceleração e jerk.
    
    Args:
        prices: Array de preços
        
    Returns:
        (momentum, acceleration, jerk)
    """
    if len(prices) < 4:
        return (0.0, 0.0, 0.0)
    
    # First derivative - Momentum (velocity)
    momentum = prices[-1] - prices[-2]
    
    # Second derivative - Acceleration
    momentum_prev = prices[-2] - prices[-3]
    acceleration = momentum - momentum_prev
    
    # Third derivative - Jerk
    momentum_prev2 = prices[-3] - prices[-4]
    acceleration_prev = momentum_prev - momentum_prev2
    jerk = acceleration - acceleration_prev
    
    return (momentum, acceleration, jerk)


@jit(nopython=True)
def calculate_price_efficiency(prices: np.ndarray, window: int = 20) -> float:
    """
    Calcula eficiência de preço (Kaufman Efficiency Ratio).
    
    Args:
        prices: Array de preços
        window: Janela para cálculo
        
    Returns:
        Price efficiency ratio [0, 1]
    """
    if len(prices) < window + 1:
        return 0.5
    
    # Direction (net change)
    direction = abs(prices[-1] - prices[-window-1])
    
    # Volatility (sum of absolute changes)
    volatility = 0.0
    for i in range(1, window + 1):
        volatility += abs(prices[-i] - prices[-i-1])
    
    if volatility == 0:
        return 1.0
    
    return direction / volatility


@jit(nopython=True)
def calculate_price_range_metrics(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula métricas baseadas em range de preço.
    
    Args:
        high: Array de máximas
        low: Array de mínimas
        close: Array de fechamentos
        
    Returns:
        (true_range, average_true_range, range_ratio)
    """
    if len(high) < 2:
        return (0.0, 0.0, 0.0)
    
    # True Range
    high_low = high[-1] - low[-1]
    high_close = abs(high[-1] - close[-2])
    low_close = abs(low[-1] - close[-2])
    true_range = max(high_low, high_close, low_close)
    
    # Average True Range (simplified)
    atr_period = min(14, len(high))
    atr_sum = 0.0
    for i in range(atr_period):
        hl = high[-i-1] - low[-i-1]
        if i < len(close) - 1:
            hc = abs(high[-i-1] - close[-i-2])
            lc = abs(low[-i-1] - close[-i-2])
            atr_sum += max(hl, hc, lc)
        else:
            atr_sum += hl
    
    atr = atr_sum / atr_period
    
    # Range ratio
    range_ratio = true_range / atr if atr > 0 else 0.0
    
    return (true_range, atr, range_ratio)


class PriceFeatureExtractor:
    """
    Extrator especializado para features de preço.
    """
    
    def __init__(self):
        """Inicializa extrator de features de preço."""
        self.price_buffers = {}
        self.max_buffer_size = 1000
        
    def extract_price_features(
        self,
        symbol: str,
        prices: List[float],
        timestamps: List[float]
    ) -> Dict[str, float]:
        """
        Extrai todas as features relacionadas a preço.
        
        Args:
            symbol: Símbolo do ativo
            prices: Lista de preços
            timestamps: Lista de timestamps
            
        Returns:
            Dicionário com features de preço
        """
        if len(prices) < 2:
            return self._get_default_features()
        
        # Convert to numpy arrays
        price_array = np.array(prices)
        
        # Calculate returns for different periods
        periods = np.array([60, 300, 900, 3600])  # 1m, 5m, 15m, 1h in seconds
        returns = calculate_returns_fast(price_array, periods)
        
        # Log returns
        log_return = calculate_log_returns(price_array)
        squared_return = log_return ** 2
        
        # Volatility for different periods
        vol_1m = calculate_volatility_fast(price_array[-60:] if len(price_array) >= 60 else price_array)
        vol_5m = calculate_volatility_fast(price_array[-300:] if len(price_array) >= 300 else price_array)
        vol_15m = calculate_volatility_fast(price_array[-900:] if len(price_array) >= 900 else price_array)
        
        # Momentum indicators
        momentum, acceleration, jerk = calculate_momentum_indicators(price_array)
        
        # Price efficiency
        efficiency = calculate_price_efficiency(price_array)
        
        return {
            'returns_1m': returns[0],
            'returns_5m': returns[1],
            'returns_15m': returns[2],
            'returns_1h': returns[3],
            'log_returns': log_return,
            'squared_returns': squared_return,
            'volatility_1m': vol_1m,
            'volatility_5m': vol_5m,
            'volatility_15m': vol_15m,
            'momentum': momentum,
            'acceleration': acceleration,
            'jerk': jerk,
            'price_efficiency': efficiency
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Retorna features padrão quando não há dados suficientes."""
        return {
            'returns_1m': 0.0,
            'returns_5m': 0.0,
            'returns_15m': 0.0,
            'returns_1h': 0.0,
            'log_returns': 0.0,
            'squared_returns': 0.0,
            'volatility_1m': 0.0,
            'volatility_5m': 0.0,
            'volatility_15m': 0.0,
            'momentum': 0.0,
            'acceleration': 0.0,
            'jerk': 0.0,
            'price_efficiency': 0.5
        }


# Utility functions for external use
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calcula Sharpe Ratio.
    
    Args:
        returns: Array de retornos
        risk_free_rate: Taxa livre de risco
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0


def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """
    Calcula Sortino Ratio (usa apenas downside volatility).
    
    Args:
        returns: Array de retornos
        target_return: Retorno alvo
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std if downside_std > 0 else 0.0