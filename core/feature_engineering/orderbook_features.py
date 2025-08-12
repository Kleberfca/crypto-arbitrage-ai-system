"""
Orderbook Features Module - Extração de features do order book
Calcula spread, depth, imbalance, micro-price e detecção de spoofing
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from numba import jit, prange
import logging
from scipy.signal import find_peaks

from core.data_collector.exchange_connector import OrderBookSnapshot

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_bid_ask_spread(best_bid: float, best_ask: float) -> float:
    """
    Calcula bid-ask spread absoluto.
    
    Args:
        best_bid: Melhor preço de compra
        best_ask: Melhor preço de venda
        
    Returns:
        Spread absoluto
    """
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    
    return best_ask - best_bid


@jit(nopython=True)
def calculate_spread_bps(best_bid: float, best_ask: float) -> float:
    """
    Calcula spread em basis points.
    
    Args:
        best_bid: Melhor preço de compra
        best_ask: Melhor preço de venda
        
    Returns:
        Spread em bps
    """
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    
    return (spread / mid_price) * 10000  # basis points


@jit(nopython=True)
def calculate_micro_price(bids: np.ndarray, asks: np.ndarray) -> float:
    """
    Calcula micro-price (preço ponderado por volume no topo do book).
    
    Args:
        bids: Array de [price, volume] dos bids
        asks: Array de [price, volume] dos asks
        
    Returns:
        Micro-price
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    best_bid = bids[0, 0]
    best_ask = asks[0, 0]
    bid_volume = bids[0, 1]
    ask_volume = asks[0, 1]
    
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return (best_bid + best_ask) / 2
    
    return (best_bid * ask_volume + best_ask * bid_volume) / total_volume


@jit(nopython=True)
def calculate_book_imbalance(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> float:
    """
    Calcula imbalance do orderbook.
    
    Args:
        bids: Array de [price, volume] dos bids
        asks: Array de [price, volume] dos asks
        levels: Número de níveis para considerar
        
    Returns:
        Imbalance [-1, 1]
    """
    bid_volume = 0.0
    ask_volume = 0.0
    
    max_bid_levels = min(levels, len(bids))
    max_ask_levels = min(levels, len(asks))
    
    for i in range(max_bid_levels):
        bid_volume += bids[i, 1]
    
    for i in range(max_ask_levels):
        ask_volume += asks[i, 1]
    
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    
    return (bid_volume - ask_volume) / total


@jit(nopython=True)
def calculate_book_depth_ratio(bids: np.ndarray, asks: np.ndarray, price_range_pct: float = 0.01) -> float:
    """
    Calcula ratio de profundidade do book dentro de um range de preço.
    
    Args:
        bids: Array de [price, volume] dos bids
        asks: Array de [price, volume] dos asks
        price_range_pct: Range de preço em percentual (1% default)
        
    Returns:
        Depth ratio
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    mid_price = (bids[0, 0] + asks[0, 0]) / 2
    price_threshold = mid_price * price_range_pct
    
    bid_depth = 0.0
    ask_depth = 0.0
    
    # Calculate bid depth within range
    for i in range(len(bids)):
        if mid_price - bids[i, 0] <= price_threshold:
            bid_depth += bids[i, 1]
        else:
            break
    
    # Calculate ask depth within range
    for i in range(len(asks)):
        if asks[i, 0] - mid_price <= price_threshold:
            ask_depth += asks[i, 1]
        else:
            break
    
    total_depth = bid_depth + ask_depth
    if total_depth == 0:
        return 0.0
    
    return bid_depth / total_depth


@jit(nopython=True)
def calculate_order_flow_imbalance(
    bids: np.ndarray,
    asks: np.ndarray,
    prev_bids: np.ndarray,
    prev_asks: np.ndarray
) -> float:
    """
    Calcula imbalance do fluxo de ordens (mudanças no orderbook).
    
    Args:
        bids: Bids atuais
        asks: Asks atuais
        prev_bids: Bids anteriores
        prev_asks: Asks anteriores
        
    Returns:
        Order flow imbalance
    """
    if len(prev_bids) == 0 or len(prev_asks) == 0:
        return 0.0
    
    # Calculate volume changes
    bid_change = 0.0
    ask_change = 0.0
    
    # Compare top 5 levels
    levels = 5
    
    for i in range(min(levels, len(bids), len(prev_bids))):
        if i < len(bids) and i < len(prev_bids):
            if bids[i, 0] == prev_bids[i, 0]:  # Same price level
                bid_change += bids[i, 1] - prev_bids[i, 1]
    
    for i in range(min(levels, len(asks), len(prev_asks))):
        if i < len(asks) and i < len(prev_asks):
            if asks[i, 0] == prev_asks[i, 0]:  # Same price level
                ask_change += asks[i, 1] - prev_asks[i, 1]
    
    total_change = abs(bid_change) + abs(ask_change)
    if total_change == 0:
        return 0.0
    
    return (bid_change - ask_change) / total_change


@jit(nopython=True)
def detect_spoofing_indicator(bids: np.ndarray, asks: np.ndarray) -> float:
    """
    Detecta indicadores de spoofing (ordens grandes longe do spread).
    
    Args:
        bids: Array de bids
        asks: Array de asks
        
    Returns:
        Spoofing indicator [0, 1]
    """
    if len(bids) < 10 or len(asks) < 10:
        return 0.0
    
    # Calculate average volume in top 3 levels
    top_bid_volume = np.mean(bids[:3, 1])
    top_ask_volume = np.mean(asks[:3, 1])
    
    # Calculate average volume in levels 7-10
    deep_bid_volume = np.mean(bids[6:10, 1])
    deep_ask_volume = np.mean(asks[6:10, 1])
    
    # Spoofing indicator: large orders deep in book relative to top
    bid_spoof = deep_bid_volume / (top_bid_volume + 1e-10)
    ask_spoof = deep_ask_volume / (top_ask_volume + 1e-10)
    
    # Normalize to [0, 1]
    spoofing_score = min((bid_spoof + ask_spoof) / 10, 1.0)
    
    return spoofing_score


@jit(nopython=True)
def calculate_liquidity_score(bids: np.ndarray, asks: np.ndarray, depth: int = 20) -> float:
    """
    Calcula score de liquidez baseado em volume e spread.
    
    Args:
        bids: Array de bids
        asks: Array de asks
        depth: Profundidade para análise
        
    Returns:
        Liquidity score [0, 1]
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    # Calculate total volume in top levels
    bid_liquidity = 0.0
    ask_liquidity = 0.0
    
    for i in range(min(depth, len(bids))):
        bid_liquidity += bids[i, 1]
    
    for i in range(min(depth, len(asks))):
        ask_liquidity += asks[i, 1]
    
    total_liquidity = bid_liquidity + ask_liquidity
    
    # Calculate spread impact
    spread = asks[0, 0] - bids[0, 0]
    mid_price = (bids[0, 0] + asks[0, 0]) / 2
    spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 10000
    
    # Combine volume and spread (lower spread = better liquidity)
    spread_factor = 1.0 / (1.0 + spread_bps / 10)  # Normalize spread impact
    
    # Normalize liquidity (assuming 1000 units is good liquidity)
    volume_factor = min(total_liquidity / 1000, 1.0)
    
    return (volume_factor + spread_factor) / 2


class OrderbookFeatureExtractor:
    """
    Extrator especializado para features do orderbook.
    """
    
    def __init__(self):
        """Inicializa extrator de features do orderbook."""
        self.orderbook_history = {}
        self.max_history = 100
        
    def extract_orderbook_features(
        self,
        orderbook: OrderBookSnapshot,
        prev_orderbook: Optional[OrderBookSnapshot] = None
    ) -> Dict[str, float]:
        """
        Extrai todas as features do orderbook.
        
        Args:
            orderbook: Snapshot atual do orderbook
            prev_orderbook: Snapshot anterior (para order flow)
            
        Returns:
            Dicionário com features do orderbook
        """
        bids = orderbook.bids
        asks = orderbook.asks
        
        if len(bids) == 0 or len(asks) == 0:
            return self._get_default_features()
        
        # Basic spread metrics
        bid_ask_spread = calculate_bid_ask_spread(bids[0, 0], asks[0, 0])
        spread_bps = calculate_spread_bps(bids[0, 0], asks[0, 0])
        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        micro_price = calculate_micro_price(bids, asks)
        
        # Book imbalance metrics
        book_imbalance = calculate_book_imbalance(bids, asks)
        book_depth_ratio = calculate_book_depth_ratio(bids, asks)
        
        # Order flow (if previous orderbook available)
        if prev_orderbook is not None:
            order_flow_imbalance = calculate_order_flow_imbalance(
                bids, asks, prev_orderbook.bids, prev_orderbook.asks
            )
        else:
            order_flow_imbalance = 0.0
        
        # Advanced metrics
        spoofing_indicator = detect_spoofing_indicator(bids, asks)
        liquidity_score = calculate_liquidity_score(bids, asks)
        
        # Orderbook heat (concentration of orders)
        orderbook_heat = self._calculate_orderbook_heat(bids, asks)
        
        # Wall distances
        bid_wall_distance, ask_wall_distance = self._find_wall_distances(bids, asks, mid_price)
        
        return {
            'bid_ask_spread': bid_ask_spread,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'micro_price': micro_price,
            'book_imbalance': book_imbalance,
            'book_depth_ratio': book_depth_ratio,
            'order_flow_imbalance': order_flow_imbalance,
            'spoofing_indicator': spoofing_indicator,
            'liquidity_score': liquidity_score,
            'orderbook_heat': orderbook_heat,
            'bid_wall_distance': bid_wall_distance,
            'ask_wall_distance': ask_wall_distance
        }
    
    def _calculate_orderbook_heat(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """
        Calcula "heat" do orderbook (concentração de ordens grandes).
        
        Args:
            bids: Array de bids
            asks: Array de asks
            
        Returns:
            Orderbook heat [0, 1]
        """
        all_volumes = np.concatenate([bids[:, 1], asks[:, 1]])
        
        if len(all_volumes) == 0:
            return 0.0
        
        # Find peaks (large orders)
        mean_volume = np.mean(all_volumes)
        std_volume = np.std(all_volumes)
        threshold = mean_volume + 2 * std_volume
        
        large_orders = all_volumes[all_volumes > threshold]
        
        if len(large_orders) == 0:
            return 0.0
        
        # Heat score based on concentration of large orders
        heat = np.sum(large_orders) / np.sum(all_volumes)
        
        return min(heat, 1.0)
    
    def _find_wall_distances(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float
    ) -> Tuple[float, float]:
        """
        Encontra distância até as "walls" (ordens muito grandes).
        
        Args:
            bids: Array de bids
            asks: Array de asks
            mid_price: Preço médio
            
        Returns:
            (bid_wall_distance_pct, ask_wall_distance_pct)
        """
        # Find bid wall
        bid_wall_distance = 0.0
        if len(bids) > 1:
            bid_volumes = bids[:, 1]
            bid_mean = np.mean(bid_volumes)
            bid_threshold = bid_mean * 5  # Wall = 5x average volume
            
            for i, (price, volume) in enumerate(bids):
                if volume > bid_threshold:
                    bid_wall_distance = (mid_price - price) / mid_price * 100
                    break
        
        # Find ask wall
        ask_wall_distance = 0.0
        if len(asks) > 1:
            ask_volumes = asks[:, 1]
            ask_mean = np.mean(ask_volumes)
            ask_threshold = ask_mean * 5  # Wall = 5x average volume
            
            for i, (price, volume) in enumerate(asks):
                if volume > ask_threshold:
                    ask_wall_distance = (price - mid_price) / mid_price * 100
                    break
        
        return (bid_wall_distance, ask_wall_distance)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Retorna features padrão quando não há dados suficientes."""
        return {
            'bid_ask_spread': 0.0,
            'spread_bps': 0.0,
            'mid_price': 0.0,
            'micro_price': 0.0,
            'book_imbalance': 0.0,
            'book_depth_ratio': 0.5,
            'order_flow_imbalance': 0.0,
            'spoofing_indicator': 0.0,
            'liquidity_score': 0.0,
            'orderbook_heat': 0.0,
            'bid_wall_distance': 0.0,
            'ask_wall_distance': 0.0
        }


# Utility functions
def calculate_effective_spread(
    trades: List[Any],
    mid_prices: List[float]
) -> float:
    """
    Calcula effective spread baseado em trades executados.
    
    Args:
        trades: Lista de trades
        mid_prices: Lista de mid prices no momento dos trades
        
    Returns:
        Effective spread
    """
    if not trades or not mid_prices:
        return 0.0
    
    effective_spreads = []
    
    for trade, mid_price in zip(trades, mid_prices):
        if mid_price > 0:
            effective_spread = 2 * abs(trade.price - mid_price) / mid_price
            effective_spreads.append(effective_spread)
    
    return np.mean(effective_spreads) if effective_spreads else 0.0