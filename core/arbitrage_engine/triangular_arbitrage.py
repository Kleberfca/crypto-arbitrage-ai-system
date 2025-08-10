"""
Triangular Arbitrage Engine - Arbitragem em ciclos triangulares
Implementa Bellman-Ford otimizado para detecção de ciclos lucrativos
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

import numpy as np
import networkx as nx
from numba import jit, types
from numba.typed import Dict as NumbaDict
import redis.asyncio as redis

from config.settings import settings, get_exchange_config
from core.arbitrage_engine.base_arbitrage import (
    BaseArbitrageEngine,
    BaseOpportunity,
    OpportunityType
)
from core.data_collector.exchange_connector import MultiExchangeConnector


logger = logging.getLogger(__name__)


@dataclass
class TriangularOpportunity(BaseOpportunity):
    """Oportunidade de arbitragem triangular."""
    
    # Path information
    exchange: str
    path: List[str]  # e.g., ['BTC/USDT', 'ETH/BTC', 'ETH/USDT']
    path_readable: str  # e.g., "USDT -> BTC -> ETH -> USDT"
    
    # Financial details
    start_amount: float
    end_amount: float
    profit_amount: float
    
    # Execution details
    execution_sequence: List[Dict[str, Any]]  # Order sequence
    total_fees: float
    min_liquidity: float
    
    # Graph metrics
    negative_cycle_weight: float
    path_length: int


class TriangularArbitrageEngine(BaseArbitrageEngine):
    """
    Motor de arbitragem triangular usando Bellman-Ford otimizado.
    Detecta ciclos lucrativos dentro de uma única exchange.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa engine de arbitragem triangular.
        
        Args:
            connector: Conector multi-exchange
        """
        super().__init__(connector, OpportunityType.TRIANGULAR)
        
        # Configuration
        self.min_profit_threshold = 0.15  # 0.15% minimum profit after fees
        self.max_path_length = 4  # Maximum cycle length
        self.min_liquidity_usd = 1000  # Minimum liquidity per leg
        self.execution_timeout_ms = 100  # Max execution time
        
        # Graph for each exchange
        self.graphs = {}
        self.graph_update_interval = 60  # seconds
        self.last_graph_update = {}
        
        # Fee structure per exchange
        self.exchange_fees = self._load_exchange_fees()
        
        # Cache
        self.path_cache = {}
        self.cache_ttl = 5  # seconds
        self.redis_client = None
        self._init_redis()
        
        # Optimized Bellman-Ford data structures
        self.distance_matrices = {}
        self.predecessor_matrices = {}
        
    def _init_redis(self) -> None:
        """Inicializa conexão Redis para cache."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False  # Binary for performance
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _load_exchange_fees(self) -> Dict[str, float]:
        """Carrega estrutura de fees das exchanges."""
        fees = {}
        for exchange in self.connector.exchanges:
            config = get_exchange_config(exchange)
            fees[exchange.value] = config.get('taker_fee', 0.001)
        return fees
    
    async def scan_opportunities(
        self,
        symbols: List[str]
    ) -> List[TriangularOpportunity]:
        """
        Escaneia oportunidades de arbitragem triangular.
        
        Args:
            symbols: Lista de símbolos disponíveis
            
        Returns:
            Lista de oportunidades triangulares
        """
        start_time = time.perf_counter()
        all_opportunities = []
        
        # Scan each exchange separately
        for exchange in self.connector.exchanges:
            exchange_id = exchange.value
            
            # Update graph if needed
            if await self._should_update_graph(exchange_id):
                await self._update_exchange_graph(exchange_id, symbols)
            
            # Find profitable cycles
            opportunities = await self._find_profitable_cycles(exchange_id)
            all_opportunities.extend(opportunities)
        
        # Sort by profit
        all_opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        
        # Log performance
        scan_time = (time.perf_counter() - start_time) * 1000
        if scan_time > 10:
            logger.warning(f"Slow triangular scan: {scan_time:.2f}ms")
        
        return all_opportunities[:10]  # Return top 10
    
    async def _should_update_graph(self, exchange: str) -> bool:
        """
        Verifica se o grafo precisa ser atualizado.
        
        Args:
            exchange: ID da exchange
            
        Returns:
            True se precisa atualizar
        """
        last_update = self.last_graph_update.get(exchange, 0)
        return time.time() - last_update > self.graph_update_interval
    
    async def _update_exchange_graph(
        self,
        exchange: str,
        symbols: List[str]
    ) -> None:
        """
        Atualiza grafo de pares para uma exchange.
        
        Args:
            exchange: ID da exchange
            symbols: Lista de símbolos
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Get current orderbooks
        connector = self.connector.connectors.get(exchange)
        if not connector:
            return
        
        # Build graph edges
        for symbol in symbols:
            orderbook = connector.get_orderbook(symbol)
            if not orderbook or orderbook.mid_price <= 0:
                continue
            
            # Parse symbol (e.g., 'BTC/USDT')
            parts = symbol.split('/')
            if len(parts) != 2:
                continue
            
            base, quote = parts
            
            # Add edges with negative log prices (for finding positive cycles)
            # Forward edge: quote -> base
            if orderbook.best_ask[0] > 0:
                rate = 1 / orderbook.best_ask[0]
                weight = -math.log(rate * (1 - self.exchange_fees[exchange]))
                G.add_edge(quote, base, 
                          weight=weight,
                          symbol=symbol,
                          direction='buy',
                          price=orderbook.best_ask[0],
                          volume=orderbook.best_ask[1])
            
            # Reverse edge: base -> quote
            if orderbook.best_bid[0] > 0:
                rate = orderbook.best_bid[0]
                weight = -math.log(rate * (1 - self.exchange_fees[exchange]))
                G.add_edge(base, quote,
                          weight=weight,
                          symbol=symbol,
                          direction='sell',
                          price=orderbook.best_bid[0],
                          volume=orderbook.best_bid[1])
        
        # Store graph
        self.graphs[exchange] = G
        self.last_graph_update[exchange] = time.time()
        
        logger.info(f"Updated graph for {exchange}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    async def _find_profitable_cycles(
        self,
        exchange: str
    ) -> List[TriangularOpportunity]:
        """
        Encontra ciclos lucrativos no grafo.
        
        Args:
            exchange: ID da exchange
            
        Returns:
            Lista de oportunidades
        """
        opportunities = []
        
        G = self.graphs.get(exchange)
        if not G or G.number_of_nodes() < 3:
            return opportunities
        
        # Check cache first
        cache_key = f"triangular:{exchange}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    # Deserialize and return
                    pass  # Simplified for brevity
            except:
                pass
        
        # Use Bellman-Ford to find negative cycles
        for start_node in list(G.nodes())[:20]:  # Limit starting nodes for performance
            cycles = self._bellman_ford_cycles(G, start_node)
            
            for cycle in cycles:
                opportunity = await self._create_opportunity(exchange, G, cycle)
                if opportunity and opportunity.is_profitable:
                    opportunities.append(opportunity)
        
        # Cache results
        if self.redis_client and opportunities:
            try:
                # Serialize and cache (simplified)
                await self.redis_client.setex(cache_key, self.cache_ttl, b"cached")
            except:
                pass
        
        return opportunities
    
    def _bellman_ford_cycles(
        self,
        G: nx.DiGraph,
        source: str
    ) -> List[List[str]]:
        """
        Bellman-Ford modificado para encontrar ciclos negativos.
        
        Args:
            G: Grafo dirigido
            source: Nó inicial
            
        Returns:
            Lista de ciclos encontrados
        """
        cycles = []
        
        # Initialize distances
        dist = {node: float('inf') for node in G.nodes()}
        dist[source] = 0
        pred = {node: None for node in G.nodes()}
        
        # Relax edges |V| - 1 times
        for _ in range(len(G.nodes()) - 1):
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    pred[v] = u
        
        # Check for negative cycles
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                # Found negative cycle, trace it back
                cycle = self._trace_cycle(pred, v, source)
                if cycle and len(cycle) <= self.max_path_length:
                    cycles.append(cycle)
        
        return cycles[:5]  # Return max 5 cycles
    
    def _trace_cycle(
        self,
        pred: Dict[str, str],
        start: str,
        source: str
    ) -> Optional[List[str]]:
        """
        Traça um ciclo a partir dos predecessores.
        
        Args:
            pred: Dicionário de predecessores
            start: Nó inicial do ciclo
            source: Nó fonte original
            
        Returns:
            Lista de nós no ciclo
        """
        visited = set()
        path = []
        current = start
        
        # Trace back to find cycle
        while current and current not in visited:
            visited.add(current)
            path.append(current)
            current = pred.get(current)
            
            if current == source:
                path.append(current)
                return path[::-1]  # Reverse to get correct order
            
            if len(path) > self.max_path_length:
                return None
        
        return None
    
    @jit(nopython=True)
    def _calculate_cycle_profit_fast(
        self,
        prices: np.ndarray,
        fees: float
    ) -> float:
        """
        Calcula lucro de um ciclo com Numba.
        
        Args:
            prices: Array de preços no ciclo
            fees: Taxa por transação
            
        Returns:
            Lucro percentual
        """
        profit_multiplier = 1.0
        
        for price in prices:
            profit_multiplier *= price * (1 - fees)
        
        return (profit_multiplier - 1.0) * 100
    
    async def _create_opportunity(
        self,
        exchange: str,
        G: nx.DiGraph,
        cycle: List[str]
    ) -> Optional[TriangularOpportunity]:
        """
        Cria oportunidade a partir de um ciclo.
        
        Args:
            exchange: ID da exchange
            G: Grafo
            cycle: Lista de nós no ciclo
            
        Returns:
            Oportunidade se lucrativa
        """
        if len(cycle) < 3:
            return None
        
        # Build execution sequence
        execution_sequence = []
        path_symbols = []
        total_profit_multiplier = 1.0
        min_liquidity = float('inf')
        
        for i in range(len(cycle) - 1):
            curr_node = cycle[i]
            next_node = cycle[i + 1]
            
            # Get edge data
            edge_data = G.get_edge_data(curr_node, next_node)
            if not edge_data:
                return None
            
            symbol = edge_data['symbol']
            direction = edge_data['direction']
            price = edge_data['price']
            volume = edge_data['volume']
            
            path_symbols.append(symbol)
            
            # Calculate liquidity in USD
            liquidity_usd = volume * price
            min_liquidity = min(min_liquidity, liquidity_usd)
            
            # Update profit multiplier
            fee = self.exchange_fees[exchange]
            if direction == 'buy':
                total_profit_multiplier *= (1 / price) * (1 - fee)
            else:
                total_profit_multiplier *= price * (1 - fee)
            
            execution_sequence.append({
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'volume': volume,
                'from': curr_node,
                'to': next_node
            })
        
        # Calculate final profit
        profit_pct = (total_profit_multiplier - 1) * 100
        
        # Check minimum profit threshold
        if profit_pct < self.min_profit_threshold:
            return None
        
        # Check minimum liquidity
        if min_liquidity < self.min_liquidity_usd:
            return None
        
        # Create readable path
        path_readable = " -> ".join(cycle)
        
        # Calculate amounts
        start_amount = 1000  # Start with $1000
        end_amount = start_amount * total_profit_multiplier
        profit_amount = end_amount - start_amount
        
        # Create opportunity
        opportunity = TriangularOpportunity(
            opportunity_id=f"tri_{exchange}_{int(time.time())}",
            opportunity_type=OpportunityType.TRIANGULAR,
            expected_profit_pct=profit_pct,
            expected_profit_usd=profit_amount,
            confidence=0.9,  # High confidence for atomic execution
            risk_score=0.2,  # Low risk for same-exchange arbitrage
            required_capital=start_amount,
            estimated_execution_time_ms=len(execution_sequence) * 20,  # 20ms per leg
            
            # Triangular specific
            exchange=exchange,
            path=path_symbols,
            path_readable=path_readable,
            start_amount=start_amount,
            end_amount=end_amount,
            profit_amount=profit_amount,
            execution_sequence=execution_sequence,
            total_fees=len(execution_sequence) * self.exchange_fees[exchange] * 100,
            min_liquidity=min_liquidity,
            negative_cycle_weight=-math.log(total_profit_multiplier),
            path_length=len(cycle)
        )
        
        return opportunity
    
    async def validate_opportunity(
        self,
        opportunity: TriangularOpportunity
    ) -> Tuple[bool, str]:
        """
        Valida se oportunidade ainda é viável.
        
        Args:
            opportunity: Oportunidade a validar
            
        Returns:
            (is_valid, reason)
        """
        # Check age
        if opportunity.age_seconds > 2:  # Very short validity for triangular
            return False, "Opportunity expired"
        
        # Get current orderbooks and recalculate
        connector = self.connector.connectors.get(opportunity.exchange)
        if not connector:
            return False, "Exchange not connected"
        
        current_profit_multiplier = 1.0
        
        for step in opportunity.execution_sequence:
            orderbook = connector.get_orderbook(step['symbol'])
            if not orderbook:
                return False, f"No orderbook for {step['symbol']}"
            
            # Check liquidity
            if step['direction'] == 'buy':
                if orderbook.best_ask[1] < step['volume'] * 0.5:
                    return False, "Insufficient liquidity"
                current_price = orderbook.best_ask[0]
                current_profit_multiplier *= (1 / current_price) * (1 - self.exchange_fees[opportunity.exchange])
            else:
                if orderbook.best_bid[1] < step['volume'] * 0.5:
                    return False, "Insufficient liquidity"
                current_price = orderbook.best_bid[0]
                current_profit_multiplier *= current_price * (1 - self.exchange_fees[opportunity.exchange])
        
        # Check if still profitable
        current_profit_pct = (current_profit_multiplier - 1) * 100
        if current_profit_pct < self.min_profit_threshold:
            return False, f"Profit too low: {current_profit_pct:.3f}%"
        
        return True, "Valid"
    
    async def execute_opportunity(
        self,
        opportunity: TriangularOpportunity
    ) -> bool:
        """
        Executa arbitragem triangular.
        
        Args:
            opportunity: Oportunidade a executar
            
        Returns:
            True se executado com sucesso
        """
        start_time = time.perf_counter()
        
        try:
            logger.info(
                f"Executing triangular arbitrage",
                exchange=opportunity.exchange,
                path=opportunity.path_readable,
                profit_pct=f"{opportunity.expected_profit_pct:.3f}%",
                steps=len(opportunity.execution_sequence)
            )
            
            # Execute each leg sequentially (must be atomic)
            for i, step in enumerate(opportunity.execution_sequence):
                logger.debug(
                    f"Step {i+1}/{len(opportunity.execution_sequence)}",
                    symbol=step['symbol'],
                    direction=step['direction'],
                    price=step['price']
                )
                
                # In production, would place actual order
                await asyncio.sleep(0.001)  # Simulate execution
                
                # Check if taking too long
                elapsed = (time.perf_counter() - start_time) * 1000
                if elapsed > self.execution_timeout_ms:
                    logger.warning("Triangular execution timeout")
                    return False
            
            # Update stats
            self.opportunities_executed += 1
            self.total_profit_usd += opportunity.expected_profit_usd
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            logger.info(
                f"Triangular arbitrage executed",
                execution_ms=f"{execution_time:.2f}",
                profit=f"${opportunity.expected_profit_usd:.2f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute triangular arbitrage: {e}")
            return False
    
    def calculate_profit(
        self,
        opportunity: TriangularOpportunity
    ) -> Tuple[float, float]:
        """
        Calcula lucro final.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            (profit_pct, profit_usd)
        """
        return (opportunity.expected_profit_pct, opportunity.expected_profit_usd)
    
    def find_all_cycles(
        self,
        G: nx.DiGraph,
        max_length: int = 4
    ) -> List[List[str]]:
        """
        Encontra todos os ciclos simples até um comprimento máximo.
        
        Args:
            G: Grafo
            max_length: Comprimento máximo do ciclo
            
        Returns:
            Lista de ciclos
        """
        cycles = []
        
        # Use Johnson's algorithm for finding all simple cycles
        try:
            all_cycles = list(nx.simple_cycles(G))
            
            # Filter by length
            for cycle in all_cycles:
                if 3 <= len(cycle) <= max_length:
                    cycles.append(cycle)
                    
                if len(cycles) >= 100:  # Limit for performance
                    break
                    
        except Exception as e:
            logger.debug(f"Error finding cycles: {e}")
        
        return cycles