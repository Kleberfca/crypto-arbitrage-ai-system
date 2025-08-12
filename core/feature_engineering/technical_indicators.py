"""
Technical Indicators Module - Cálculo de indicadores técnicos
RSI, MACD, Bollinger Bands, ATR, EMAs, Stochastic e outros
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from numba import jit
import talib
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_rsi_fast(prices: np.ndarray, period: int = 14) -> float:
    """
    Calcula RSI (Relative Strength Index) otimizado.
    
    Args:
        prices: Array de preços
        period: Período do RSI
        
    Returns:
        RSI [0, 100]
    """
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


@jit(nopython=True)
def calculate_ema_fast(prices: np.ndarray, period: int) -> float:
    """
    Calcula EMA (Exponential Moving Average) otimizado.
    
    Args:
        prices: Array de preços
        period: Período da EMA
        
    Returns:
        EMA value
    """
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0.0
    
    alpha = 2.0 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    
    return ema


@jit(nopython=True)
def calculate_bollinger_bands_fast(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calcula Bollinger Bands otimizado.
    
    Args:
        prices: Array de preços
        period: Período da média móvel
        num_std: Número de desvios padrão
        
    Returns:
        (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        price = prices[-1] if len(prices) > 0 else 0.0
        return (price, price, price)
    
    recent_prices = prices[-period:]
    middle = np.mean(recent_prices)
    std = np.std(recent_prices)
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return (upper, middle, lower)


@jit(nopython=True)
def calculate_stochastic_fast(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[float, float]:
    """
    Calcula Stochastic Oscillator otimizado.
    
    Args:
        high: Array de máximas
        low: Array de mínimas
        close: Array de fechamentos
        period: Período do stochastic
        smooth_k: Suavização do %K
        smooth_d: Suavização do %D
        
    Returns:
        (k, d)
    """
    if len(close) < period:
        return (50.0, 50.0)
    
    # Calculate %K
    lowest_low = np.min(low[-period:])
    highest_high = np.max(high[-period:])
    
    if highest_high == lowest_low:
        return (50.0, 50.0)
    
    k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
    
    # For %D, we need historical %K values (simplified here)
    d = k  # Simplified - in practice, would be SMA of %K
    
    return (k, d)


class TechnicalIndicatorExtractor:
    """
    Extrator especializado para indicadores técnicos.
    """
    
    def __init__(self):
        """Inicializa extrator de indicadores técnicos."""
        self.price_history = {}
        self.indicator_cache = {}
        self.cache_ttl = 1.0  # 1 second cache
        
    def extract_technical_indicators(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extrai todos os indicadores técnicos.
        
        Args:
            symbol: Símbolo do ativo
            prices: Array de preços
            volumes: Array de volumes (opcional)
            
        Returns:
            Dicionário com indicadores técnicos
        """
        if len(prices) < 2:
            return self._get_default_indicators()
        
        # Generate high/low approximations if not available
        high = prices * 1.001  # Approximate high
        low = prices * 0.999   # Approximate low
        close = prices
        
        indicators = {}
        
        # RSI
        if len(prices) >= 14:
            indicators['rsi'] = talib.RSI(prices, timeperiod=14)[-1] if len(prices) >= 14 else 50.0
        else:
            indicators['rsi'] = calculate_rsi_fast(prices)
        
        # MACD
        if len(prices) >= 26:
            macd, signal, hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1] if len(macd) > 0 else 0.0
            indicators['macd_signal'] = signal[-1] if len(signal) > 0 else 0.0
            indicators['macd_histogram'] = hist[-1] if len(hist) > 0 else 0.0
        else:
            indicators['macd'] = 0.0
            indicators['macd_signal'] = 0.0
            indicators['macd_histogram'] = 0.0
        
        # Bollinger Bands
        if len(prices) >= 20:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bollinger_upper'] = upper[-1] if len(upper) > 0 else prices[-1]
            indicators['bollinger_middle'] = middle[-1] if len(middle) > 0 else prices[-1]
            indicators['bollinger_lower'] = lower[-1] if len(lower) > 0 else prices[-1]
        else:
            upper, middle, lower = calculate_bollinger_bands_fast(prices)
            indicators['bollinger_upper'] = upper
            indicators['bollinger_middle'] = middle
            indicators['bollinger_lower'] = lower
        
        # ATR
        if len(prices) >= 14:
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(prices) >= 14 else 0.0
        else:
            indicators['atr'] = np.std(prices[-14:]) if len(prices) >= 2 else 0.0
        
        # EMAs
        indicators['ema_9'] = talib.EMA(prices, timeperiod=9)[-1] if len(prices) >= 9 else prices[-1]
        indicators['ema_21'] = talib.EMA(prices, timeperiod=21)[-1] if len(prices) >= 21 else prices[-1]
        indicators['ema_50'] = talib.EMA(prices, timeperiod=50)[-1] if len(prices) >= 50 else prices[-1]
        indicators['ema_200'] = talib.EMA(prices, timeperiod=200)[-1] if len(prices) >= 200 else prices[-1]
        
        # Stochastic
        if len(prices) >= 14:
            k, d = talib.STOCH(high, low, close, fastk_period=14)
            indicators['stochastic_k'] = k[-1] if len(k) > 0 else 50.0
            indicators['stochastic_d'] = d[-1] if len(d) > 0 else 50.0
        else:
            k, d = calculate_stochastic_fast(high, low, close)
            indicators['stochastic_k'] = k
            indicators['stochastic_d'] = d
        
        # Williams %R
        if len(prices) >= 14:
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(prices) >= 14 else -50.0
        else:
            indicators['williams_r'] = -50.0
        
        # CCI
        if len(prices) >= 20:
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=20)[-1] if len(prices) >= 20 else 0.0
        else:
            indicators['cci'] = 0.0
        
        # ADX
        if len(prices) >= 14:
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1] if len(prices) >= 14 else 25.0
        else:
            indicators['adx'] = 25.0
        
        # OBV (if volume available)
        if volumes is not None and len(volumes) >= len(prices):
            indicators['obv'] = talib.OBV(prices, volumes)[-1] if len(prices) >= 2 else 0.0
        else:
            indicators['obv'] = 0.0
        
        # MFI (if volume available)
        if volumes is not None and len(volumes) >= 14:
            indicators['mfi'] = talib.MFI(high, low, close, volumes, timeperiod=14)[-1] if len(prices) >= 14 else 50.0
        else:
            indicators['mfi'] = 50.0
        
        # ROC
        if len(prices) >= 10:
            indicators['roc'] = talib.ROC(prices, timeperiod=10)[-1] if len(prices) >= 10 else 0.0
        else:
            indicators['roc'] = 0.0
        
        return indicators
    
    def calculate_advanced_indicators(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcula indicadores avançados adicionais.
        
        Args:
            prices: Array de preços
            volumes: Array de volumes
            
        Returns:
            Dicionário com indicadores avançados
        """
        advanced = {}
        
        if len(prices) < 20:
            return advanced
        
        # Ichimoku Cloud components
        if len(prices) >= 52:
            # Tenkan-sen (Conversion Line)
            high_9 = np.max(prices[-9:])
            low_9 = np.min(prices[-9:])
            advanced['ichimoku_tenkan'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = np.max(prices[-26:])
            low_26 = np.min(prices[-26:])
            advanced['ichimoku_kijun'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            advanced['ichimoku_span_a'] = (advanced['ichimoku_tenkan'] + advanced['ichimoku_kijun']) / 2
            
            # Senkou Span B (Leading Span B)
            high_52 = np.max(prices[-52:])
            low_52 = np.min(prices[-52:])
            advanced['ichimoku_span_b'] = (high_52 + low_52) / 2
        
        # Parabolic SAR
        if len(prices) >= 2:
            high = prices * 1.001
            low = prices * 0.999
            advanced['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]
        
        # Fibonacci Retracement Levels
        if len(prices) >= 100:
            recent_high = np.max(prices[-100:])
            recent_low = np.min(prices[-100:])
            diff = recent_high - recent_low
            
            advanced['fib_0'] = recent_low
            advanced['fib_236'] = recent_low + 0.236 * diff
            advanced['fib_382'] = recent_low + 0.382 * diff
            advanced['fib_500'] = recent_low + 0.500 * diff
            advanced['fib_618'] = recent_low + 0.618 * diff
            advanced['fib_1000'] = recent_high
        
        # Pivot Points
        if len(prices) >= 1:
            high = np.max(prices[-24:]) if len(prices) >= 24 else prices[-1]
            low = np.min(prices[-24:]) if len(prices) >= 24 else prices[-1]
            close = prices[-1]
            
            pivot = (high + low + close) / 3
            advanced['pivot_point'] = pivot
            advanced['pivot_r1'] = 2 * pivot - low
            advanced['pivot_s1'] = 2 * pivot - high
            advanced['pivot_r2'] = pivot + (high - low)
            advanced['pivot_s2'] = pivot - (high - low)
        
        return advanced
    
    def _get_default_indicators(self) -> Dict[str, float]:
        """Retorna indicadores padrão quando não há dados suficientes."""
        return {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bollinger_upper': 0.0,
            'bollinger_middle': 0.0,
            'bollinger_lower': 0.0,
            'atr': 0.0,
            'ema_9': 0.0,
            'ema_21': 0.0,
            'ema_50': 0.0,
            'ema_200': 0.0,
            'stochastic_k': 50.0,
            'stochastic_d': 50.0,
            'williams_r': -50.0,
            'cci': 0.0,
            'adx': 25.0,
            'obv': 0.0,
            'mfi': 50.0,
            'roc': 0.0
        }


# Additional indicator functions
def calculate_vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
    """
    Calcula Vortex Indicator.
    
    Args:
        high: Array de máximas
        low: Array de mínimas
        close: Array de fechamentos
        period: Período
        
    Returns:
        (vi_plus, vi_minus)
    """
    if len(close) < period + 1:
        return (0.0, 0.0)
    
    vm_plus = np.sum(np.abs(high[1:] - low[:-1]))
    vm_minus = np.sum(np.abs(low[1:] - high[:-1]))
    
    true_range = np.sum(np.maximum(
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    
    if true_range == 0:
        return (0.0, 0.0)
    
    vi_plus = vm_plus / true_range
    vi_minus = vm_minus / true_range
    
    return (vi_plus, vi_minus)


def calculate_chaikin_money_flow(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> float:
    """
    Calcula Chaikin Money Flow.
    
    Args:
        high: Array de máximas
        low: Array de mínimas
        close: Array de fechamentos
        volume: Array de volumes
        period: Período
        
    Returns:
        CMF value
    """
    if len(close) < period:
        return 0.0
    
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    money_flow_volume = money_flow_multiplier * volume
    
    cmf = np.sum(money_flow_volume[-period:]) / np.sum(volume[-period:])
    
    return cmf