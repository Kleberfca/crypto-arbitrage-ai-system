"""
Orderbook Manager - Gestão eficiente de orderbooks
Agregação, métricas em tempo real e detecção de anomalias
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

import numpy as np
from numba import jit
import pandas as pd

from config.settings import settings
from core.data_collector.data_normalizer import NormalizedOrderbook


logger = logging.getLogger(__name__)


class OrderbookDepth(Enum):
    """Profundidade do orderbook."""
    LEVEL_1 = 1  # Best bid/ask only
    LEVEL_5 = 5
    LEVEL_10 = 10
    LEVEL_20 = 20
    FULL = 100


class MarketMicrostructure(Enum):
    """Indicadores de microestrutura de mercado."""
    NORMAL = "normal"
    THIN = "thin"  # Low liquidity
    THICK = "thick"  # High liquidity
    VOLATILE = "volatile"
    SPOOFING = "spoofing"  # Potential manipulation
    CROSSED = "crossed"  # Bid > Ask


@dataclass
class OrderbookSnapshot:
    """Snapshot completo do orderbook com métricas."""
    
    # Basic data
    orderbook: NormalizedOrderbook
    
    # Aggregated metrics
    total_bid_volume: float = 0
    total_ask_volume: float = 0
    weighted_bid_price: float = 0  # VWAP
    weighted_ask_price: float = 0
    
    # Imbalance metrics
    volume_imbalance: float = 0  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    price_pressure: float = 0  # Indicates direction pressure
    
    # Depth metrics
    bid_depth_10pct: float = 0  # Volume within 10% of best bid
    ask_depth_10pct: float = 0
    
    # Microstructure
    effective_spread: float = 0
    realized_spread: float = 0
    market_microstructure: MarketMicrostructure = MarketMicrostructure.NORMAL
    
    # Spoofing detection
    spoofing_score: float = 0  # 0-1, higher = more likely spoofing
    large_orders: List[Tuple[float, float, str]] = field(default_factory=list)  # price, size, side
    
    # Historical comparison
    spread_percentile: float = 0  # Current spread vs historical
    volume_percentile: float = 0  # Current volume vs historical


@dataclass
class OrderbookDelta:
    """Delta/mudança no orderbook."""
    
    timestamp: float
    symbol: str
    exchange: str
    
    # Changes
    bid_changes: List[Tuple[float, float, float]]  # price, old_size, new_size
    ask_changes: List[Tuple[float, float, float]]
    
    # Metrics
    bid_volume_delta: float = 0
    ask_volume_delta: float = 0
    mid_price_delta: float = 0
    spread_delta: float = 0


class OrderbookManager:
    """
    Gerenciador de orderbooks com análise avançada.
    """
    
    def __init__(self):
        """Inicializa gerenciador de orderbooks."""
        # Storage
        self.orderbooks: Dict[str, Dict[str, NormalizedOrderbook]] = defaultdict(dict)
        self.snapshots: Dict[str, Dict[str, OrderbookSnapshot]] = defaultdict(dict)
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Delta tracking
        self.deltas: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Aggregation settings
        self.aggregation_levels = [0.001, 0.005, 0.01, 0.05, 0.1]  # 0.1%, 0.5%, 1%, 5%, 10%
        
        # Spoofing detection
        self.spoofing_threshold = 0.7
        self.large_order_threshold = 10000  # USD
        self.spoofing_history: Dict[str, List[float]] = defaultdict(list)
        
        # Historical statistics
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.update_times = deque(maxlen=1000)
        self.last_cleanup = time.time()
        
        # Configuration
        self.max_orderbook_age = 60  # seconds
        self.cleanup_interval = 300  # 5 minutes
    
    async def update_orderbook(
        self,
        orderbook: NormalizedOrderbook
    ) -> OrderbookSnapshot:
        """
        Atualiza orderbook e calcula métricas.
        
        Args:
            orderbook: Orderbook normalizado
            
        Returns:
            Snapshot com métricas
        """
        start_time = time.perf_counter()
        
        key = f"{orderbook.exchange}:{orderbook.symbol}"
        
        # Store previous for delta calculation
        previous = self.orderbooks[orderbook.exchange].get(orderbook.symbol)
        
        # Update storage
        self.orderbooks[orderbook.exchange][orderbook.symbol] = orderbook
        
        # Calculate snapshot
        snapshot = await self._create_snapshot(orderbook)
        
        # Calculate delta if we have previous
        if previous:
            delta = self._calculate_delta(previous, orderbook)
            self.deltas[key].append(delta)
        
        # Store snapshot
        self.snapshots[orderbook.exchange][orderbook.symbol] = snapshot
        
        # Update history
        self.history[key].append({
            'timestamp': orderbook.timestamp,
            'mid_price': orderbook.mid_price,
            'spread': orderbook.spread,
            'bid_volume': snapshot.total_bid_volume,
            'ask_volume': snapshot.total_ask_volume
        })
        
        # Update spread and volume history
        self.spread_history[key].append(orderbook.spread_pct)
        self.volume_history[key].append(snapshot.total_bid_volume + snapshot.total_ask_volume)
        
        # Track performance
        update_time = (time.perf_counter() - start_time) * 1000
        self.update_times.append(update_time)
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            asyncio.create_task(self._cleanup_old_data())
        
        return snapshot
    
    async def _create_snapshot(
        self,
        orderbook: NormalizedOrderbook
    ) -> OrderbookSnapshot:
        """
        Cria snapshot com métricas calculadas.
        
        Args:
            orderbook: Orderbook normalizado
            
        Returns:
            Snapshot completo
        """
        snapshot = OrderbookSnapshot(orderbook=orderbook)
        
        if not orderbook.bids or not orderbook.asks:
            return snapshot
        
        # Convert to numpy for fast calculations
        bids = np.array(orderbook.bids)
        asks = np.array(orderbook.asks)
        
        # Calculate basic metrics
        metrics = self._calculate_metrics_fast(bids, asks)
        
        snapshot.total_bid_volume = metrics[0]
        snapshot.total_ask_volume = metrics[1]
        snapshot.weighted_bid_price = metrics[2]
        snapshot.weighted_ask_price = metrics[3]
        snapshot.volume_imbalance = metrics[4]
        
        # Calculate depth metrics
        depth_metrics = self._calculate_depth_metrics(bids, asks, orderbook.mid_price)
        snapshot.bid_depth_10pct = depth_metrics[0]
        snapshot.ask_depth_10pct = depth_metrics[1]
        
        # Calculate price pressure
        snapshot.price_pressure = self._calculate_price_pressure(bids, asks)
        
        # Detect microstructure
        snapshot.market_microstructure = self._detect_microstructure(
            orderbook, snapshot
        )
        
        # Spoofing detection
        snapshot.spoofing_score, snapshot.large_orders = await self._detect_spoofing(
            orderbook, snapshot
        )
        
        # Historical percentiles
        key = f"{orderbook.exchange}:{orderbook.symbol}"
        
        if len(self.spread_history[key]) > 10:
            snapshot.spread_percentile = self._calculate_percentile(
                orderbook.spread_pct, list(self.spread_history[key])
            )
        
        if len(self.volume_history[key]) > 10:
            current_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
            snapshot.volume_percentile = self._calculate_percentile(
                current_volume, list(self.volume_history[key])
            )
        
        return snapshot
    
    @jit(nopython=True)
    def _calculate_metrics_fast(
        self,
        bids: np.ndarray,
        asks: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Calcula métricas básicas otimizado.
        
        Args:
            bids: Array de bids
            asks: Array de asks
            
        Returns:
            (bid_volume, ask_volume, weighted_bid, weighted_ask, imbalance)
        """
        # Total volumes
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        
        # Weighted prices (VWAP)
        if bid_volume > 0:
            weighted_bid = np.sum(bids[:, 0] * bids[:, 1]) / bid_volume
        else:
            weighted_bid = 0
        
        if ask_volume > 0:
            weighted_ask = np.sum(asks[:, 0] * asks[:, 1]) / ask_volume
        else:
            weighted_ask = 0
        
        # Volume imbalance
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        else:
            imbalance = 0
        
        return bid_volume, ask_volume, weighted_bid, weighted_ask, imbalance
    
    def _calculate_depth_metrics(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float
    ) -> Tuple[float, float]:
        """
        Calcula métricas de profundidade.
        
        Args:
            bids: Array de bids
            asks: Array de asks
            mid_price: Preço médio
            
        Returns:
            (bid_depth_10pct, ask_depth_10pct)
        """
        if mid_price <= 0:
            return 0, 0
        
        # Calculate 10% price levels
        bid_threshold = mid_price * 0.9
        ask_threshold = mid_price * 1.1
        
        # Sum volume within 10%
        bid_depth = 0
        for price, volume in bids:
            if price >= bid_threshold:
                bid_depth += volume * price  # USD value
            else:
                break
        
        ask_depth = 0
        for price, volume in asks:
            if price <= ask_threshold:
                ask_depth += volume * price
            else:
                break
        
        return bid_depth, ask_depth
    
    def _calculate_price_pressure(
        self,
        bids: np.ndarray,
        asks: np.ndarray
    ) -> float:
        """
        Calcula pressão de preço.
        
        Args:
            bids: Array de bids
            asks: Array de asks
            
        Returns:
            Pressão (-1 a 1, negative = downward pressure)
        """
        if len(bids) < 5 or len(asks) < 5:
            return 0
        
        # Compare top 5 levels
        top_bid_volume = np.sum(bids[:5, 1])
        top_ask_volume = np.sum(asks[:5, 1])
        
        total = top_bid_volume + top_ask_volume
        if total > 0:
            pressure = (top_bid_volume - top_ask_volume) / total
        else:
            pressure = 0
        
        return pressure
    
    def _detect_microstructure(
        self,
        orderbook: NormalizedOrderbook,
        snapshot: OrderbookSnapshot
    ) -> MarketMicrostructure:
        """
        Detecta estado da microestrutura de mercado.
        
        Args:
            orderbook: Orderbook
            snapshot: Snapshot com métricas
            
        Returns:
            Tipo de microestrutura
        """
        # Check for crossed book
        if orderbook.is_crossed:
            return MarketMicrostructure.CROSSED
        
        # Check liquidity
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
        
        if total_volume < 1000:  # Low liquidity (< $1000)
            return MarketMicrostructure.THIN
        elif total_volume > 100000:  # High liquidity (> $100k)
            return MarketMicrostructure.THICK
        
        # Check for spoofing
        if snapshot.spoofing_score > self.spoofing_threshold:
            return MarketMicrostructure.SPOOFING
        
        # Check volatility (wide spread)
        if orderbook.spread_pct > 1:  # > 1% spread
            return MarketMicrostructure.VOLATILE
        
        return MarketMicrostructure.NORMAL
    
    async def _detect_spoofing(
        self,
        orderbook: NormalizedOrderbook,
        snapshot: OrderbookSnapshot
    ) -> Tuple[float, List[Tuple[float, float, str]]]:
        """
        Detecta potencial spoofing/layering.
        
        Args:
            orderbook: Orderbook
            snapshot: Snapshot
            
        Returns:
            (spoofing_score, large_orders)
        """
        large_orders = []
        spoofing_indicators = 0
        total_indicators = 5
        
        # 1. Check for large orders far from best price
        if len(orderbook.bids) > 5:
            for i in range(5, min(10, len(orderbook.bids))):
                price, volume = orderbook.bids[i]
                value = price * volume
                
                if value > self.large_order_threshold:
                    large_orders.append((price, volume, 'bid'))
                    
                    # Check if far from best bid
                    distance = (orderbook.bids[0][0] - price) / orderbook.bids[0][0]
                    if distance > 0.005:  # > 0.5% away
                        spoofing_indicators += 0.2
        
        # 2. Check for large asks far from best
        if len(orderbook.asks) > 5:
            for i in range(5, min(10, len(orderbook.asks))):
                price, volume = orderbook.asks[i]
                value = price * volume
                
                if value > self.large_order_threshold:
                    large_orders.append((price, volume, 'ask'))
                    
                    distance = (price - orderbook.asks[0][0]) / orderbook.asks[0][0]
                    if distance > 0.005:
                        spoofing_indicators += 0.2
        
        # 3. Check for unusual volume distribution
        if snapshot.total_bid_volume > 0 and snapshot.total_ask_volume > 0:
            # Large imbalance but tight spread (potential layering)
            if abs(snapshot.volume_imbalance) > 0.7 and orderbook.spread_pct < 0.1:
                spoofing_indicators += 1
        
        # 4. Check order size distribution
        if len(orderbook.bids) > 3:
            bid_sizes = [b[1] for b in orderbook.bids[:5]]
            if max(bid_sizes) > np.mean(bid_sizes) * 5:  # One order 5x larger
                spoofing_indicators += 1
        
        if len(orderbook.asks) > 3:
            ask_sizes = [a[1] for a in orderbook.asks[:5]]
            if max(ask_sizes) > np.mean(ask_sizes) * 5:
                spoofing_indicators += 1
        
        # 5. Check historical pattern
        key = f"{orderbook.exchange}:{orderbook.symbol}"
        
        # Track if large orders disappear quickly
        if large_orders:
            self.spoofing_history[key].append(len(large_orders))
            
            # If orders appear and disappear frequently
            if len(self.spoofing_history[key]) > 10:
                recent = self.spoofing_history[key][-10:]
                volatility = np.std(recent)
                if volatility > 3:  # High volatility in large order count
                    spoofing_indicators += 1
        
        # Calculate final score
        spoofing_score = min(spoofing_indicators / total_indicators, 1.0)
        
        return spoofing_score, large_orders
    
    def _calculate_delta(
        self,
        previous: NormalizedOrderbook,
        current: NormalizedOrderbook
    ) -> OrderbookDelta:
        """
        Calcula delta entre orderbooks.
        
        Args:
            previous: Orderbook anterior
            current: Orderbook atual
            
        Returns:
            Delta
        """
        delta = OrderbookDelta(
            timestamp=current.timestamp,
            symbol=current.symbol,
            exchange=current.exchange,
            bid_changes=[],
            ask_changes=[]
        )
        
        # Track bid changes
        prev_bids = {price: size for price, size in previous.bids}
        curr_bids = {price: size for price, size in current.bids}
        
        all_bid_prices = set(prev_bids.keys()) | set(curr_bids.keys())
        
        for price in sorted(all_bid_prices, reverse=True):
            old_size = prev_bids.get(price, 0)
            new_size = curr_bids.get(price, 0)
            
            if old_size != new_size:
                delta.bid_changes.append((price, old_size, new_size))
                delta.bid_volume_delta += new_size - old_size
        
        # Track ask changes
        prev_asks = {price: size for price, size in previous.asks}
        curr_asks = {price: size for price, size in current.asks}
        
        all_ask_prices = set(prev_asks.keys()) | set(curr_asks.keys())
        
        for price in sorted(all_ask_prices):
            old_size = prev_asks.get(price, 0)
            new_size = curr_asks.get(price, 0)
            
            if old_size != new_size:
                delta.ask_changes.append((price, old_size, new_size))
                delta.ask_volume_delta += new_size - old_size
        
        # Price changes
        delta.mid_price_delta = current.mid_price - previous.mid_price
        delta.spread_delta = current.spread - previous.spread
        
        return delta
    
    def _calculate_percentile(
        self,
        value: float,
        historical: List[float]
    ) -> float:
        """
        Calcula percentil de um valor.
        
        Args:
            value: Valor atual
            historical: Valores históricos
            
        Returns:
            Percentil (0-100)
        """
        if not historical:
            return 50
        
        return (np.sum(np.array(historical) <= value) / len(historical)) * 100
    
    async def aggregate_orderbook(
        self,
        orderbook: NormalizedOrderbook,
        level: float
    ) -> NormalizedOrderbook:
        """
        Agrega orderbook por nível de preço.
        
        Args:
            orderbook: Orderbook original
            level: Nível de agregação (e.g., 0.01 = 1%)
            
        Returns:
            Orderbook agregado
        """
        if not orderbook.bids or not orderbook.asks:
            return orderbook
        
        # Aggregate bids
        aggregated_bids = defaultdict(float)
        
        for price, volume in orderbook.bids:
            # Round down to aggregation level
            agg_price = np.floor(price / level) * level
            aggregated_bids[agg_price] += volume
        
        # Sort and convert back to list
        new_bids = sorted(
            [(p, v) for p, v in aggregated_bids.items()],
            reverse=True
        )[:orderbook.levels]
        
        # Aggregate asks
        aggregated_asks = defaultdict(float)
        
        for price, volume in orderbook.asks:
            # Round up to aggregation level
            agg_price = np.ceil(price / level) * level
            aggregated_asks[agg_price] += volume
        
        new_asks = sorted(
            [(p, v) for p, v in aggregated_asks.items()]
        )[:orderbook.levels]
        
        # Create new orderbook
        aggregated = NormalizedOrderbook(
            timestamp=orderbook.timestamp,
            symbol=orderbook.symbol,
            exchange=orderbook.exchange,
            bids=new_bids,
            asks=new_asks,
            levels=min(len(new_bids), len(new_asks))
        )
        
        # Recalculate metrics
        if new_bids and new_asks:
            aggregated.mid_price = (new_bids[0][0] + new_asks[0][0]) / 2
            aggregated.spread = new_asks[0][0] - new_bids[0][0]
            aggregated.spread_pct = aggregated.spread / aggregated.mid_price * 100
        
        return aggregated
    
    def get_orderbook(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[NormalizedOrderbook]:
        """
        Retorna orderbook atual.
        
        Args:
            exchange: Nome da exchange
            symbol: Símbolo
            
        Returns:
            Orderbook ou None
        """
        return self.orderbooks.get(exchange, {}).get(symbol)
    
    def get_snapshot(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[OrderbookSnapshot]:
        """
        Retorna snapshot atual.
        
        Args:
            exchange: Nome da exchange
            symbol: Símbolo
            
        Returns:
            Snapshot ou None
        """
        return self.snapshots.get(exchange, {}).get(symbol)
    
    def get_best_prices(
        self,
        symbol: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Retorna melhores preços de todas as exchanges.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dict com best bid/ask por exchange
        """
        best_prices = {}
        
        for exchange, orderbooks in self.orderbooks.items():
            if symbol in orderbooks:
                ob = orderbooks[symbol]
                if ob.bids and ob.asks:
                    best_prices[exchange] = (ob.bids[0][0], ob.asks[0][0])
        
        return best_prices
    
    async def _cleanup_old_data(self) -> None:
        """Limpa dados antigos."""
        current_time = time.time()
        
        # Clean old orderbooks
        for exchange in list(self.orderbooks.keys()):
            for symbol in list(self.orderbooks[exchange].keys()):
                ob = self.orderbooks[exchange][symbol]
                
                if current_time - ob.timestamp > self.max_orderbook_age:
                    del self.orderbooks[exchange][symbol]
                    
                    if symbol in self.snapshots[exchange]:
                        del self.snapshots[exchange][symbol]
        
        # Clean empty exchanges
        for exchange in list(self.orderbooks.keys()):
            if not self.orderbooks[exchange]:
                del self.orderbooks[exchange]
        
        self.last_cleanup = current_time
        
        logger.debug(f"Cleaned old orderbook data")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do gerenciador."""
        total_orderbooks = sum(
            len(obs) for obs in self.orderbooks.values()
        )
        
        avg_update_time = np.mean(list(self.update_times)) if self.update_times else 0
        max_update_time = max(self.update_times) if self.update_times else 0
        
        return {
            'total_orderbooks': total_orderbooks,
            'exchanges': list(self.orderbooks.keys()),
            'avg_update_time_ms': avg_update_time,
            'max_update_time_ms': max_update_time,
            'history_size': sum(len(h) for h in self.history.values()),
            'delta_tracking': sum(len(d) for d in self.deltas.values())
        }