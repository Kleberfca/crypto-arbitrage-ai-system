"""
Pre-Funding Manager - Gestão de fundos pré-alocados
Gerencia alocação 70/30 e rebalancing via BEP20
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

import numpy as np
from numba import jit
import redis.asyncio as redis

from config.settings import settings
from core.data_collector.exchange_connector import MultiExchangeConnector


logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Estratégias de alocação de fundos."""
    FIXED = "fixed"  # Alocação fixa 70/30
    DYNAMIC = "dynamic"  # Alocação dinâmica baseada em oportunidades
    ADAPTIVE = "adaptive"  # Adaptativo baseado em performance


class RebalanceType(Enum):
    """Tipos de rebalanceamento."""
    SCHEDULED = "scheduled"  # Programado
    THRESHOLD = "threshold"  # Quando atinge limite
    EMERGENCY = "emergency"  # Emergencial
    OPPORTUNITY = "opportunity"  # Baseado em oportunidades


@dataclass
class FundAllocation:
    """Alocação de fundos por exchange."""
    
    exchange: str
    asset: str
    
    # Balances
    allocated_amount: float
    available_amount: float
    locked_amount: float
    
    # Allocation targets
    target_allocation_pct: float
    min_allocation_pct: float
    max_allocation_pct: float
    
    # Metrics
    utilization_rate: float = 0.0
    return_rate: float = 0.0
    last_rebalance: float = field(default_factory=time.time)
    
    @property
    def total_amount(self) -> float:
        """Retorna quantidade total."""
        return self.available_amount + self.locked_amount
    
    @property
    def allocation_pct(self) -> float:
        """Retorna percentual atual de alocação."""
        if self.allocated_amount > 0:
            return self.total_amount / self.allocated_amount * 100
        return 0
    
    @property
    def needs_rebalance(self) -> bool:
        """Verifica se precisa rebalancear."""
        pct = self.allocation_pct
        return pct < self.min_allocation_pct or pct > self.max_allocation_pct


@dataclass
class RebalanceRequest:
    """Solicitação de rebalanceamento."""
    
    request_id: str
    rebalance_type: RebalanceType
    timestamp: float = field(default_factory=time.time)
    
    # Transfer details
    from_exchange: str = ""
    to_exchange: str = ""
    asset: str = ""
    amount: float = 0.0
    
    # Network details
    network: str = "BEP20"  # Default to BEP20 for speed
    estimated_fee: float = 0.0
    estimated_time_seconds: float = 30.0
    
    # Status
    status: str = "pending"
    tx_hash: Optional[str] = None
    completed_at: Optional[float] = None
    
    @property
    def is_complete(self) -> bool:
        """Verifica se rebalanceamento está completo."""
        return self.status in ["completed", "failed", "cancelled"]


class PreFundingManager:
    """
    Gerenciador de fundos pré-alocados.
    Mantém 70% dos fundos distribuídos e 30% para rebalancing.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa gerenciador de pre-funding.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        
        # Allocation configuration
        self.pre_funding_ratio = 0.70  # 70% pre-funded
        self.rebalance_ratio = 0.30  # 30% for rebalancing
        self.allocation_strategy = AllocationStrategy.DYNAMIC
        
        # Fund allocations per exchange and asset
        self.allocations: Dict[str, Dict[str, FundAllocation]] = defaultdict(dict)
        
        # Rebalancing
        self.rebalance_threshold = 0.15  # 15% deviation triggers rebalance
        self.min_rebalance_interval = 3600  # 1 hour minimum
        self.last_rebalance_check = 0
        self.pending_rebalances: List[RebalanceRequest] = []
        self.rebalance_history: List[RebalanceRequest] = []
        
        # Capital tracking
        self.total_capital_usd = 0.0
        self.allocated_capital_usd = 0.0
        self.reserve_capital_usd = 0.0
        
        # Performance metrics
        self.allocation_efficiency = {}
        self.rebalance_count = 0
        self.rebalance_costs_usd = 0.0
        
        # Redis for persistence
        self.redis_client = None
        self._init_redis()
        
        # Initialize allocations
        asyncio.create_task(self._initialize_allocations())
    
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
    
    async def _initialize_allocations(self) -> None:
        """Inicializa alocações de fundos."""
        try:
            # Get total capital from settings
            self.total_capital_usd = settings.trading_config.get('total_capital_usd', 10000)
            self.allocated_capital_usd = self.total_capital_usd * self.pre_funding_ratio
            self.reserve_capital_usd = self.total_capital_usd * self.rebalance_ratio
            
            # Distribute among exchanges based on strategy
            if self.allocation_strategy == AllocationStrategy.FIXED:
                await self._apply_fixed_allocation()
            elif self.allocation_strategy == AllocationStrategy.DYNAMIC:
                await self._apply_dynamic_allocation()
            else:
                await self._apply_adaptive_allocation()
            
            logger.info(
                f"Pre-funding initialized",
                total_capital=f"${self.total_capital_usd:.2f}",
                allocated=f"${self.allocated_capital_usd:.2f}",
                reserve=f"${self.reserve_capital_usd:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize allocations: {e}")
    
    async def _apply_fixed_allocation(self) -> None:
        """Aplica alocação fixa entre exchanges."""
        exchanges = list(self.connector.exchanges)
        if not exchanges:
            return
        
        # Equal distribution
        per_exchange = self.allocated_capital_usd / len(exchanges)
        
        for exchange in exchanges:
            # Main trading pairs
            main_assets = ['USDT', 'BTC', 'ETH']
            
            for asset in main_assets:
                allocation = FundAllocation(
                    exchange=exchange.value,
                    asset=asset,
                    allocated_amount=per_exchange / len(main_assets),
                    available_amount=per_exchange / len(main_assets),
                    locked_amount=0,
                    target_allocation_pct=100 / len(exchanges),
                    min_allocation_pct=100 / len(exchanges) - self.rebalance_threshold * 100,
                    max_allocation_pct=100 / len(exchanges) + self.rebalance_threshold * 100
                )
                
                self.allocations[exchange.value][asset] = allocation
    
    async def _apply_dynamic_allocation(self) -> None:
        """Aplica alocação dinâmica baseada em histórico de oportunidades."""
        exchanges = list(self.connector.exchanges)
        if not exchanges:
            return
        
        # Weight by historical opportunity frequency (simplified)
        weights = {
            'binance': 0.40,  # 40% - highest liquidity
            'okx': 0.25,      # 25%
            'bybit': 0.20,    # 20%
            'kucoin': 0.10,   # 10%
            'htx': 0.05       # 5%
        }
        
        for exchange in exchanges:
            exchange_id = exchange.value.lower()
            weight = weights.get(exchange_id, 1.0 / len(exchanges))
            exchange_allocation = self.allocated_capital_usd * weight
            
            # Distribute among assets
            asset_weights = {'USDT': 0.5, 'BTC': 0.3, 'ETH': 0.2}
            
            for asset, asset_weight in asset_weights.items():
                amount = exchange_allocation * asset_weight
                
                allocation = FundAllocation(
                    exchange=exchange.value,
                    asset=asset,
                    allocated_amount=amount,
                    available_amount=amount,
                    locked_amount=0,
                    target_allocation_pct=weight * 100,
                    min_allocation_pct=weight * 100 * (1 - self.rebalance_threshold),
                    max_allocation_pct=weight * 100 * (1 + self.rebalance_threshold)
                )
                
                self.allocations[exchange.value][asset] = allocation
    
    async def _apply_adaptive_allocation(self) -> None:
        """Aplica alocação adaptativa baseada em performance."""
        # Start with dynamic, then adapt based on performance
        await self._apply_dynamic_allocation()
        
        # Will be updated based on actual performance metrics
        # This is a placeholder for future ML-based optimization
    
    async def check_availability(
        self,
        exchange: str,
        asset: str,
        amount: float
    ) -> bool:
        """
        Verifica disponibilidade de fundos.
        
        Args:
            exchange: ID da exchange
            asset: Ativo
            amount: Quantidade necessária
            
        Returns:
            True se fundos disponíveis
        """
        allocation = self.allocations.get(exchange, {}).get(asset)
        
        if not allocation:
            return False
        
        return allocation.available_amount >= amount
    
    async def lock_funds(
        self,
        exchange: str,
        asset: str,
        amount: float
    ) -> bool:
        """
        Bloqueia fundos para execução.
        
        Args:
            exchange: ID da exchange
            asset: Ativo
            amount: Quantidade a bloquear
            
        Returns:
            True se bloqueado com sucesso
        """
        allocation = self.allocations.get(exchange, {}).get(asset)
        
        if not allocation or allocation.available_amount < amount:
            return False
        
        allocation.available_amount -= amount
        allocation.locked_amount += amount
        allocation.utilization_rate = allocation.locked_amount / allocation.allocated_amount
        
        logger.debug(
            f"Locked funds",
            exchange=exchange,
            asset=asset,
            amount=amount,
            available=allocation.available_amount
        )
        
        return True
    
    async def release_funds(
        self,
        exchange: str,
        asset: str,
        amount: float
    ) -> None:
        """
        Libera fundos após execução.
        
        Args:
            exchange: ID da exchange
            asset: Ativo
            amount: Quantidade a liberar
        """
        allocation = self.allocations.get(exchange, {}).get(asset)
        
        if allocation:
            allocation.locked_amount = max(0, allocation.locked_amount - amount)
            allocation.available_amount += amount
            allocation.utilization_rate = allocation.locked_amount / allocation.allocated_amount
            
            logger.debug(
                f"Released funds",
                exchange=exchange,
                asset=asset,
                amount=amount,
                available=allocation.available_amount
            )
    
    async def check_rebalance_needed(self) -> List[RebalanceRequest]:
        """
        Verifica necessidade de rebalanceamento.
        
        Returns:
            Lista de rebalanceamentos necessários
        """
        current_time = time.time()
        
        # Rate limit checks
        if current_time - self.last_rebalance_check < 60:  # Check every minute
            return []
        
        self.last_rebalance_check = current_time
        rebalance_requests = []
        
        # Check each allocation
        for exchange, assets in self.allocations.items():
            for asset, allocation in assets.items():
                if allocation.needs_rebalance:
                    # Check if enough time passed since last rebalance
                    if current_time - allocation.last_rebalance < self.min_rebalance_interval:
                        continue
                    
                    # Calculate rebalance amount
                    target_amount = allocation.allocated_amount
                    current_amount = allocation.total_amount
                    diff = target_amount - current_amount
                    
                    if abs(diff) > allocation.allocated_amount * 0.05:  # 5% minimum
                        # Find source exchange with excess
                        source_exchange = self._find_excess_exchange(asset, exchange)
                        
                        if source_exchange:
                            request = RebalanceRequest(
                                request_id=f"rebal_{int(time.time())}",
                                rebalance_type=RebalanceType.THRESHOLD,
                                from_exchange=source_exchange,
                                to_exchange=exchange,
                                asset=asset,
                                amount=abs(diff),
                                estimated_fee=0.50,  # $0.50 BEP20 fee
                                estimated_time_seconds=30
                            )
                            
                            rebalance_requests.append(request)
        
        return rebalance_requests
    
    def _find_excess_exchange(
        self,
        asset: str,
        exclude_exchange: str
    ) -> Optional[str]:
        """
        Encontra exchange com excesso de fundos.
        
        Args:
            asset: Ativo
            exclude_exchange: Exchange a excluir
            
        Returns:
            ID da exchange com excesso ou None
        """
        best_exchange = None
        max_excess = 0
        
        for exchange, assets in self.allocations.items():
            if exchange == exclude_exchange:
                continue
            
            allocation = assets.get(asset)
            if allocation:
                excess = allocation.total_amount - allocation.allocated_amount
                if excess > max_excess:
                    max_excess = excess
                    best_exchange = exchange
        
        return best_exchange
    
    async def execute_rebalance(
        self,
        request: RebalanceRequest
    ) -> bool:
        """
        Executa rebalanceamento via BEP20.
        
        Args:
            request: Solicitação de rebalanceamento
            
        Returns:
            True se executado com sucesso
        """
        try:
            logger.info(
                f"Executing rebalance",
                from_exchange=request.from_exchange,
                to_exchange=request.to_exchange,
                asset=request.asset,
                amount=request.amount
            )
            
            # In production, would execute actual transfer
            # For now, simulate transfer
            await asyncio.sleep(2)  # Simulate network delay
            
            # Update allocations
            from_allocation = self.allocations[request.from_exchange][request.asset]
            to_allocation = self.allocations[request.to_exchange][request.asset]
            
            from_allocation.available_amount -= request.amount
            to_allocation.available_amount += request.amount - request.estimated_fee
            
            # Update metrics
            from_allocation.last_rebalance = time.time()
            to_allocation.last_rebalance = time.time()
            
            # Mark request as complete
            request.status = "completed"
            request.completed_at = time.time()
            request.tx_hash = f"0x{''.join(['a'] * 64)}"  # Simulated tx hash
            
            # Update stats
            self.rebalance_count += 1
            self.rebalance_costs_usd += request.estimated_fee
            
            # Store in history
            self.rebalance_history.append(request)
            if len(self.rebalance_history) > 1000:
                self.rebalance_history = self.rebalance_history[-1000:]
            
            logger.info(
                f"Rebalance completed",
                tx_hash=request.tx_hash,
                time_seconds=request.completed_at - request.timestamp
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Rebalance failed: {e}")
            request.status = "failed"
            return False
    
    @jit(nopython=True)
    def _calculate_optimal_allocation(
        self,
        historical_volumes: np.ndarray,
        historical_profits: np.ndarray,
        current_allocations: np.ndarray
    ) -> np.ndarray:
        """
        Calcula alocação ótima usando histórico.
        
        Args:
            historical_volumes: Volumes históricos por exchange
            historical_profits: Lucros históricos por exchange
            current_allocations: Alocações atuais
            
        Returns:
            Alocações ótimas
        """
        n_exchanges = len(current_allocations)
        
        # Calculate weights based on profit per volume
        profit_efficiency = np.zeros(n_exchanges)
        
        for i in range(n_exchanges):
            if historical_volumes[i] > 0:
                profit_efficiency[i] = historical_profits[i] / historical_volumes[i]
        
        # Normalize to get allocation percentages
        total_efficiency = np.sum(profit_efficiency)
        
        if total_efficiency > 0:
            optimal_allocations = profit_efficiency / total_efficiency
        else:
            # Equal allocation if no history
            optimal_allocations = np.ones(n_exchanges) / n_exchanges
        
        # Smooth transition from current to optimal
        alpha = 0.1  # Learning rate
        new_allocations = (1 - alpha) * current_allocations + alpha * optimal_allocations
        
        # Ensure minimum allocation
        min_allocation = 0.05  # 5% minimum
        new_allocations = np.maximum(new_allocations, min_allocation)
        
        # Renormalize
        new_allocations = new_allocations / np.sum(new_allocations)
        
        return new_allocations
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Retorna resumo das alocações."""
        summary = {
            'total_capital_usd': self.total_capital_usd,
            'allocated_capital_usd': self.allocated_capital_usd,
            'reserve_capital_usd': self.reserve_capital_usd,
            'allocations': {},
            'utilization_rates': {},
            'rebalance_stats': {
                'count': self.rebalance_count,
                'costs_usd': self.rebalance_costs_usd,
                'pending': len(self.pending_rebalances)
            }
        }
        
        for exchange, assets in self.allocations.items():
            exchange_total = sum(a.total_amount for a in assets.values())
            exchange_util = np.mean([a.utilization_rate for a in assets.values()])
            
            summary['allocations'][exchange] = exchange_total
            summary['utilization_rates'][exchange] = exchange_util
        
        return summary
    
    async def emergency_rebalance(self) -> None:
        """Executa rebalanceamento de emergência."""
        logger.warning("Emergency rebalance triggered")
        
        # Create emergency rebalance requests for all imbalanced allocations
        for exchange, assets in self.allocations.items():
            for asset, allocation in assets.items():
                if allocation.allocation_pct < 50:  # Less than 50% of target
                    # Find source with most excess
                    source = self._find_excess_exchange(asset, exchange)
                    if source:
                        request = RebalanceRequest(
                            request_id=f"emergency_{int(time.time())}",
                            rebalance_type=RebalanceType.EMERGENCY,
                            from_exchange=source,
                            to_exchange=exchange,
                            asset=asset,
                            amount=allocation.allocated_amount * 0.3,  # Transfer 30%
                            estimated_fee=0.50,
                            estimated_time_seconds=30
                        )
                        
                        # Execute immediately
                        await self.execute_rebalance(request)