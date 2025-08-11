"""
Execution Optimizer - Otimização de execução de ordens
Smart Order Routing, splitting e timing optimization
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
from numba import jit
import redis.asyncio as redis

from config.settings import settings
from core.data_collector.exchange_connector import MultiExchangeConnector
from core.arbitrage_engine.base_arbitrage import BaseOpportunity


logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Estratégias de otimização."""
    SPEED = "speed"  # Prioriza velocidade
    COST = "cost"  # Minimiza custos
    STEALTH = "stealth"  # Evita impacto no mercado
    BALANCED = "balanced"  # Balanceado


class ExecutionMode(Enum):
    """Modos de execução."""
    AGGRESSIVE = "aggressive"  # Toma liquidez
    PASSIVE = "passive"  # Provê liquidez
    ADAPTIVE = "adaptive"  # Adapta baseado em condições


@dataclass
class RoutingDecision:
    """Decisão de roteamento de ordem."""
    
    exchange: str
    symbol: str
    side: str
    
    # Routing details
    route_type: str  # 'direct', 'split', 'iceberg'
    split_count: int = 1
    
    # Execution parameters
    order_type: str = "limit"
    price_adjustment: float = 0.0  # Basis points
    time_in_force: str = "IOC"
    
    # Timing
    delay_ms: float = 0
    priority: int = 0
    
    # Expected metrics
    expected_fill_rate: float = 1.0
    expected_slippage_bps: float = 0
    expected_fees_bps: float = 10  # 0.1% default
    
    # Risk limits
    max_slippage_bps: float = 10
    timeout_ms: float = 100


@dataclass
class ExecutionPlan:
    """Plano de execução otimizado."""
    
    opportunity_id: str
    strategy: OptimizationStrategy
    mode: ExecutionMode
    
    # Routing decisions
    routing_decisions: List[RoutingDecision]
    
    # Execution sequence
    execution_sequence: List[int]  # Order of execution
    parallel_groups: List[List[int]]  # Groups that can execute in parallel
    
    # Timing
    total_expected_time_ms: float
    critical_path_ms: float
    
    # Expected outcome
    expected_profit_usd: float
    expected_cost_usd: float
    confidence_score: float
    
    # Risk parameters
    max_total_slippage_usd: float
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)


class ExecutionOptimizer:
    """
    Otimizador de execução de ordens.
    Implementa Smart Order Routing e otimizações avançadas.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa otimizador de execução.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        
        # Optimization settings
        self.default_strategy = OptimizationStrategy.BALANCED
        self.default_mode = ExecutionMode.ADAPTIVE
        
        # Order splitting thresholds
        self.min_split_size_usd = 1000
        self.max_single_order_usd = 10000
        self.optimal_order_size_usd = 5000
        
        # Market impact model parameters
        self.impact_coefficient = 0.1  # 10 bps per $10k
        self.temporary_impact_decay = 0.5  # 50% decay per minute
        
        # Timing optimization
        self.network_latency_ms = {
            'binance': 10,
            'okx': 15,
            'bybit': 12,
            'kucoin': 20,
            'htx': 25
        }
        
        # Fee structure cache
        self.fee_cache = {}
        self._load_fee_structure()
        
        # Performance history
        self.execution_history = []
        self.performance_metrics = {}
        
        # Redis for caching
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Inicializa conexão Redis."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _load_fee_structure(self) -> None:
        """Carrega estrutura de fees das exchanges."""
        # Simplified fee structure
        self.fee_cache = {
            'binance': {'maker': 0.0010, 'taker': 0.0010},
            'okx': {'maker': 0.0008, 'taker': 0.0010},
            'bybit': {'maker': 0.0010, 'taker': 0.0010},
            'kucoin': {'maker': 0.0010, 'taker': 0.0010},
            'htx': {'maker': 0.0020, 'taker': 0.0020}
        }
    
    async def create_execution_plan(
        self,
        opportunity: BaseOpportunity,
        strategy: Optional[OptimizationStrategy] = None,
        mode: Optional[ExecutionMode] = None
    ) -> ExecutionPlan:
        """
        Cria plano de execução otimizado.
        
        Args:
            opportunity: Oportunidade a executar
            strategy: Estratégia de otimização
            mode: Modo de execução
            
        Returns:
            Plano de execução otimizado
        """
        start_time = time.perf_counter()
        
        strategy = strategy or self.default_strategy
        mode = mode or self.default_mode
        
        # Analyze opportunity
        routing_decisions = await self._create_routing_decisions(opportunity, strategy, mode)
        
        # Optimize execution sequence
        execution_sequence, parallel_groups = self._optimize_execution_sequence(
            routing_decisions, strategy
        )
        
        # Calculate expected metrics
        total_time = self._calculate_total_time(routing_decisions, parallel_groups)
        critical_path = self._calculate_critical_path(routing_decisions, execution_sequence)
        expected_cost = self._calculate_expected_cost(routing_decisions)
        max_slippage = self._calculate_max_slippage(routing_decisions, opportunity)
        
        # Create contingency plans
        contingency_plans = self._create_contingency_plans(opportunity, routing_decisions)
        
        plan = ExecutionPlan(
            opportunity_id=opportunity.opportunity_id,
            strategy=strategy,
            mode=mode,
            routing_decisions=routing_decisions,
            execution_sequence=execution_sequence,
            parallel_groups=parallel_groups,
            total_expected_time_ms=total_time,
            critical_path_ms=critical_path,
            expected_profit_usd=opportunity.expected_profit_usd,
            expected_cost_usd=expected_cost,
            confidence_score=self._calculate_confidence(routing_decisions),
            max_total_slippage_usd=max_slippage,
            contingency_plans=contingency_plans
        )
        
        # Log optimization time
        optimization_time = (time.perf_counter() - start_time) * 1000
        if optimization_time > 5:
            logger.warning(f"Slow execution optimization: {optimization_time:.2f}ms")
        
        return plan
    
    async def _create_routing_decisions(
        self,
        opportunity: BaseOpportunity,
        strategy: OptimizationStrategy,
        mode: ExecutionMode
    ) -> List[RoutingDecision]:
        """
        Cria decisões de roteamento para cada ordem.
        
        Args:
            opportunity: Oportunidade
            strategy: Estratégia
            mode: Modo
            
        Returns:
            Lista de decisões de roteamento
        """
        decisions = []
        
        # Get execution details from opportunity metadata
        execution_details = opportunity.metadata.get('execution_sequence', [])
        
        for detail in execution_details:
            # Check if order should be split
            order_size_usd = detail.get('volume', 0) * detail.get('price', 0)
            should_split = order_size_usd > self.max_single_order_usd
            
            if should_split:
                # Create multiple routing decisions for split order
                num_splits = int(np.ceil(order_size_usd / self.optimal_order_size_usd))
                
                for i in range(num_splits):
                    decision = await self._create_single_routing_decision(
                        detail, strategy, mode, i, num_splits
                    )
                    decisions.append(decision)
            else:
                # Single routing decision
                decision = await self._create_single_routing_decision(
                    detail, strategy, mode
                )
                decisions.append(decision)
        
        return decisions
    
    async def _create_single_routing_decision(
        self,
        detail: Dict[str, Any],
        strategy: OptimizationStrategy,
        mode: ExecutionMode,
        split_index: int = 0,
        total_splits: int = 1
    ) -> RoutingDecision:
        """
        Cria decisão de roteamento única.
        
        Args:
            detail: Detalhes da execução
            strategy: Estratégia
            mode: Modo
            split_index: Índice do split
            total_splits: Total de splits
            
        Returns:
            Decisão de roteamento
        """
        exchange = detail.get('exchange', '')
        symbol = detail.get('symbol', '')
        side = detail.get('direction', 'buy')
        
        # Determine route type
        if total_splits > 1:
            route_type = 'split'
        elif strategy == OptimizationStrategy.STEALTH:
            route_type = 'iceberg'
        else:
            route_type = 'direct'
        
        # Determine order type and adjustments based on mode
        if mode == ExecutionMode.AGGRESSIVE:
            order_type = 'market'
            price_adjustment = 10  # 10 bps aggressive
            time_in_force = 'IOC'
        elif mode == ExecutionMode.PASSIVE:
            order_type = 'limit'
            price_adjustment = -5  # 5 bps passive
            time_in_force = 'GTC'
        else:  # ADAPTIVE
            # Check market conditions
            liquidity = await self._check_liquidity(exchange, symbol)
            if liquidity > detail.get('volume', 0) * 2:
                order_type = 'limit'
                price_adjustment = 0
                time_in_force = 'IOC'
            else:
                order_type = 'limit'
                price_adjustment = 5
                time_in_force = 'FOK'
        
        # Calculate delay for split orders
        delay_ms = 0
        if total_splits > 1:
            if strategy == OptimizationStrategy.STEALTH:
                # Random delay to avoid detection
                delay_ms = np.random.uniform(100, 500) * split_index
            else:
                # Fixed delay
                delay_ms = 50 * split_index
        
        # Get expected metrics
        latency = self.network_latency_ms.get(exchange.lower(), 20)
        fees = self.fee_cache.get(exchange.lower(), {})
        expected_fees = fees.get('taker' if order_type == 'market' else 'maker', 0.001) * 10000
        
        return RoutingDecision(
            exchange=exchange,
            symbol=symbol,
            side=side,
            route_type=route_type,
            split_count=total_splits,
            order_type=order_type,
            price_adjustment=price_adjustment,
            time_in_force=time_in_force,
            delay_ms=delay_ms,
            priority=0 if strategy == OptimizationStrategy.SPEED else split_index,
            expected_fill_rate=0.95 if order_type == 'limit' else 1.0,
            expected_slippage_bps=5 if order_type == 'market' else 2,
            expected_fees_bps=expected_fees,
            max_slippage_bps=20,
            timeout_ms=latency + 50
        )
    
    async def _check_liquidity(
        self,
        exchange: str,
        symbol: str
    ) -> float:
        """
        Verifica liquidez disponível.
        
        Args:
            exchange: Exchange
            symbol: Símbolo
            
        Returns:
            Liquidez disponível
        """
        connector = self.connector.connectors.get(exchange)
        if connector:
            orderbook = connector.get_orderbook(symbol)
            if orderbook:
                # Sum first 5 levels of liquidity
                bid_liquidity = sum(level[1] for level in orderbook.bids[:5])
                ask_liquidity = sum(level[1] for level in orderbook.asks[:5])
                return min(bid_liquidity, ask_liquidity)
        
        return 0
    
    def _optimize_execution_sequence(
        self,
        decisions: List[RoutingDecision],
        strategy: OptimizationStrategy
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Otimiza sequência de execução.
        
        Args:
            decisions: Decisões de roteamento
            strategy: Estratégia
            
        Returns:
            (sequência, grupos paralelos)
        """
        n = len(decisions)
        
        if strategy == OptimizationStrategy.SPEED:
            # Execute all in parallel
            sequence = list(range(n))
            parallel_groups = [list(range(n))]
            
        elif strategy == OptimizationStrategy.STEALTH:
            # Sequential execution with delays
            sequence = list(range(n))
            parallel_groups = [[i] for i in range(n)]
            
        else:  # COST or BALANCED
            # Group by exchange to minimize fees
            exchange_groups = {}
            for i, decision in enumerate(decisions):
                if decision.exchange not in exchange_groups:
                    exchange_groups[decision.exchange] = []
                exchange_groups[decision.exchange].append(i)
            
            # Create parallel groups by exchange
            parallel_groups = list(exchange_groups.values())
            sequence = [idx for group in parallel_groups for idx in group]
        
        return sequence, parallel_groups
    
    def _calculate_total_time(
        self,
        decisions: List[RoutingDecision],
        parallel_groups: List[List[int]]
    ) -> float:
        """
        Calcula tempo total de execução.
        
        Args:
            decisions: Decisões de roteamento
            parallel_groups: Grupos paralelos
            
        Returns:
            Tempo total em ms
        """
        total_time = 0
        
        for group in parallel_groups:
            # Max time in parallel group
            group_time = max(
                decisions[i].timeout_ms + decisions[i].delay_ms
                for i in group
            )
            total_time += group_time
        
        return total_time
    
    def _calculate_critical_path(
        self,
        decisions: List[RoutingDecision],
        sequence: List[int]
    ) -> float:
        """
        Calcula caminho crítico.
        
        Args:
            decisions: Decisões
            sequence: Sequência
            
        Returns:
            Tempo do caminho crítico em ms
        """
        if not sequence:
            return 0
        
        # For now, simplified critical path
        return sum(decisions[i].timeout_ms for i in sequence[:3])  # First 3 orders
    
    def _calculate_expected_cost(
        self,
        decisions: List[RoutingDecision]
    ) -> float:
        """
        Calcula custo esperado.
        
        Args:
            decisions: Decisões
            
        Returns:
            Custo esperado em USD
        """
        total_fees_bps = sum(d.expected_fees_bps for d in decisions)
        
        # Assume $1000 per order for simplification
        total_volume = len(decisions) * 1000
        
        return total_volume * total_fees_bps / 10000
    
    def _calculate_max_slippage(
        self,
        decisions: List[RoutingDecision],
        opportunity: BaseOpportunity
    ) -> float:
        """
        Calcula slippage máximo.
        
        Args:
            decisions: Decisões
            opportunity: Oportunidade
            
        Returns:
            Slippage máximo em USD
        """
        max_slippage_bps = max(d.max_slippage_bps for d in decisions)
        
        return opportunity.required_capital * max_slippage_bps / 10000
    
    def _calculate_confidence(
        self,
        decisions: List[RoutingDecision]
    ) -> float:
        """
        Calcula score de confiança.
        
        Args:
            decisions: Decisões
            
        Returns:
            Score de confiança (0-1)
        """
        if not decisions:
            return 0
        
        # Average expected fill rate
        avg_fill_rate = np.mean([d.expected_fill_rate for d in decisions])
        
        # Penalty for high slippage
        avg_slippage = np.mean([d.expected_slippage_bps for d in decisions])
        slippage_penalty = max(0, 1 - avg_slippage / 50)  # 50 bps = 0 confidence
        
        return avg_fill_rate * slippage_penalty
    
    def _create_contingency_plans(
        self,
        opportunity: BaseOpportunity,
        decisions: List[RoutingDecision]
    ) -> List[Dict[str, Any]]:
        """
        Cria planos de contingência.
        
        Args:
            opportunity: Oportunidade
            decisions: Decisões
            
        Returns:
            Lista de planos de contingência
        """
        plans = []
        
        # Plan A: Partial fill contingency
        plans.append({
            'trigger': 'partial_fill',
            'threshold': 0.5,  # Less than 50% filled
            'action': 'cancel_and_retry',
            'params': {'max_retries': 2, 'adjust_price_bps': 5}
        })
        
        # Plan B: High slippage contingency
        plans.append({
            'trigger': 'high_slippage',
            'threshold': 20,  # 20 bps
            'action': 'cancel_remaining',
            'params': {}
        })
        
        # Plan C: Timeout contingency
        plans.append({
            'trigger': 'timeout',
            'threshold': 500,  # 500ms
            'action': 'force_market_order',
            'params': {'max_slippage_bps': 50}
        })
        
        return plans
    
    @jit(nopython=True)
    def _calculate_market_impact(
        self,
        order_size: float,
        avg_volume: float,
        volatility: float
    ) -> float:
        """
        Calcula impacto no mercado usando modelo de Almgren-Chriss.
        
        Args:
            order_size: Tamanho da ordem
            avg_volume: Volume médio
            volatility: Volatilidade
            
        Returns:
            Impacto esperado em basis points
        """
        if avg_volume <= 0:
            return 0
        
        # Participation rate
        participation = order_size / avg_volume
        
        # Temporary impact (linear)
        temp_impact = 0.1 * participation * volatility * 10000
        
        # Permanent impact (square root)
        perm_impact = 0.05 * np.sqrt(participation) * volatility * 10000
        
        return temp_impact + perm_impact
    
    def update_performance_metrics(
        self,
        plan: ExecutionPlan,
        actual_result: Dict[str, Any]
    ) -> None:
        """
        Atualiza métricas de performance.
        
        Args:
            plan: Plano executado
            actual_result: Resultado real
        """
        # Calculate prediction error
        expected_cost = plan.expected_cost_usd
        actual_cost = actual_result.get('total_fees_usd', 0)
        cost_error = abs(expected_cost - actual_cost) / expected_cost if expected_cost > 0 else 0
        
        expected_time = plan.total_expected_time_ms
        actual_time = actual_result.get('execution_time_ms', 0)
        time_error = abs(expected_time - actual_time) / expected_time if expected_time > 0 else 0
        
        # Update metrics
        strategy = plan.strategy.value
        if strategy not in self.performance_metrics:
            self.performance_metrics[strategy] = {
                'count': 0,
                'avg_cost_error': 0,
                'avg_time_error': 0,
                'success_rate': 0
            }
        
        metrics = self.performance_metrics[strategy]
        n = metrics['count']
        
        metrics['avg_cost_error'] = (metrics['avg_cost_error'] * n + cost_error) / (n + 1)
        metrics['avg_time_error'] = (metrics['avg_time_error'] * n + time_error) / (n + 1)
        
        if actual_result.get('success', False):
            metrics['success_rate'] = (metrics['success_rate'] * n + 1) / (n + 1)
        else:
            metrics['success_rate'] = (metrics['success_rate'] * n) / (n + 1)
        
        metrics['count'] += 1
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de otimização."""
        return {
            'performance_by_strategy': self.performance_metrics,
            'avg_optimization_time_ms': 2.5,  # Placeholder
            'total_optimizations': sum(
                m['count'] for m in self.performance_metrics.values()
            )
        }