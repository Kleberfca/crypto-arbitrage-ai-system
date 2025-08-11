"""
Order Manager - Gerenciamento completo de ordens
Coordena execução, tracking e retry logic
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
import uuid
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
from core.arbitrage_engine.base_arbitrage import BaseOpportunity, OpportunityType


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Estados possíveis de uma ordem."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    """Tipos de ordem."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "immediate_or_cancel"  # Immediate or Cancel
    FOK = "fill_or_kill"  # Fill or Kill


class OrderSide(Enum):
    """Lado da ordem."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Representação de uma ordem."""
    
    # Identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    opportunity_id: str = ""
    exchange: str = ""
    symbol: str = ""
    
    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    filled_at: Optional[float] = None
    
    # Execution
    exchange_order_id: Optional[str] = None
    fees_paid: float = 0.0
    slippage: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def is_complete(self) -> bool:
        """Verifica se ordem está completa."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]
    
    @property
    def fill_ratio(self) -> float:
        """Retorna taxa de preenchimento."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Retorna tempo de execução em ms."""
        if self.submitted_at and self.filled_at:
            return (self.filled_at - self.submitted_at) * 1000
        return None


@dataclass
class ExecutionResult:
    """Resultado de uma execução."""
    
    success: bool
    opportunity_id: str
    orders: List[Order]
    total_profit_usd: float
    total_fees_usd: float
    average_slippage_pct: float
    execution_time_ms: float
    error_message: Optional[str] = None
    
    @property
    def net_profit_usd(self) -> float:
        """Lucro líquido após fees."""
        return self.total_profit_usd - self.total_fees_usd


class OrderManager:
    """
    Gerenciador central de ordens.
    Coordena execução, tracking e recuperação de falhas.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa gerenciador de ordens.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.max_history_size = 10000
        
        # Execution stats
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_volume_usd = 0.0
        self.execution_times = []
        
        # Circuit breaker
        self.failure_counts = defaultdict(int)
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_cooldown = 60  # seconds
        self.circuit_breaker_triggered = {}
        
        # Retry configuration
        self.retry_delays = [0.1, 0.5, 1.0, 2.0]  # Exponential backoff
        
        # Redis for persistence
        self.redis_client = None
        self._init_redis()
        
        # Atomic execution lock
        self.execution_locks = {}
        
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
    
    async def execute_opportunity(
        self,
        opportunity: BaseOpportunity
    ) -> ExecutionResult:
        """
        Executa uma oportunidade de arbitragem.
        
        Args:
            opportunity: Oportunidade a executar
            
        Returns:
            Resultado da execução
        """
        start_time = time.perf_counter()
        
        try:
            # Log execution start
            logger.info(
                f"Starting execution",
                opportunity_id=opportunity.opportunity_id,
                type=opportunity.opportunity_type.value,
                expected_profit=f"${opportunity.expected_profit_usd:.2f}"
            )
            
            # Choose execution strategy based on type
            if opportunity.opportunity_type == OpportunityType.TRIANGULAR:
                result = await self._execute_atomic(opportunity)
            else:
                result = await self._execute_parallel(opportunity)
            
            # Update stats
            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            if result.success:
                self.successful_orders += len(result.orders)
                logger.info(
                    f"Execution successful",
                    net_profit=f"${result.net_profit_usd:.2f}",
                    execution_ms=f"{execution_time:.2f}"
                )
            else:
                self.failed_orders += len(result.orders)
                logger.warning(
                    f"Execution failed",
                    reason=result.error_message
                )
            
            # Store in history
            self.order_history.extend(result.orders)
            if len(self.order_history) > self.max_history_size:
                self.order_history = self.order_history[-self.max_history_size:]
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                opportunity_id=opportunity.opportunity_id,
                orders=[],
                total_profit_usd=0,
                total_fees_usd=0,
                average_slippage_pct=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _execute_atomic(
        self,
        opportunity: BaseOpportunity
    ) -> ExecutionResult:
        """
        Execução atômica para arbitragem triangular.
        Todas as ordens devem ser executadas ou nenhuma.
        
        Args:
            opportunity: Oportunidade triangular
            
        Returns:
            Resultado da execução
        """
        orders = []
        
        # Create lock for atomic execution
        lock_key = f"atomic_{opportunity.opportunity_id}"
        
        # Acquire lock
        async with self._get_lock(lock_key):
            
            # Create orders from execution sequence
            execution_sequence = opportunity.metadata.get('execution_sequence', [])
            
            for step in execution_sequence:
                order = Order(
                    opportunity_id=opportunity.opportunity_id,
                    exchange=step['exchange'],
                    symbol=step['symbol'],
                    side=OrderSide.BUY if step['direction'] == 'buy' else OrderSide.SELL,
                    order_type=OrderType.FOK,  # Fill or Kill for atomic
                    quantity=step['volume'],
                    price=step['price']
                )
                orders.append(order)
            
            # Execute all orders
            executed_orders = []
            
            for order in orders:
                try:
                    # Submit order
                    success = await self._submit_order(order)
                    
                    if not success:
                        # Rollback previous orders
                        await self._rollback_orders(executed_orders)
                        return ExecutionResult(
                            success=False,
                            opportunity_id=opportunity.opportunity_id,
                            orders=orders,
                            total_profit_usd=0,
                            total_fees_usd=sum(o.fees_paid for o in orders),
                            average_slippage_pct=0,
                            execution_time_ms=0,
                            error_message="Atomic execution failed - rolled back"
                        )
                    
                    executed_orders.append(order)
                    
                except Exception as e:
                    # Rollback on any error
                    await self._rollback_orders(executed_orders)
                    raise e
            
            # Calculate results
            total_fees = sum(o.fees_paid for o in orders)
            avg_slippage = np.mean([o.slippage for o in orders])
            
            return ExecutionResult(
                success=True,
                opportunity_id=opportunity.opportunity_id,
                orders=orders,
                total_profit_usd=opportunity.expected_profit_usd,
                total_fees_usd=total_fees,
                average_slippage_pct=avg_slippage,
                execution_time_ms=0
            )
    
    async def _execute_parallel(
        self,
        opportunity: BaseOpportunity
    ) -> ExecutionResult:
        """
        Execução paralela para arbitragem espacial/estatística.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            Resultado da execução
        """
        orders = []
        
        # Create orders based on opportunity type
        if opportunity.opportunity_type == OpportunityType.SPATIAL:
            # Buy and sell orders on different exchanges
            buy_order = Order(
                opportunity_id=opportunity.opportunity_id,
                exchange=opportunity.metadata.get('buy_exchange', ''),
                symbol=opportunity.metadata.get('symbol', ''),
                side=OrderSide.BUY,
                order_type=OrderType.IOC,
                quantity=opportunity.metadata.get('quantity', 0),
                price=opportunity.metadata.get('buy_price')
            )
            
            sell_order = Order(
                opportunity_id=opportunity.opportunity_id,
                exchange=opportunity.metadata.get('sell_exchange', ''),
                symbol=opportunity.metadata.get('symbol', ''),
                side=OrderSide.SELL,
                order_type=OrderType.IOC,
                quantity=opportunity.metadata.get('quantity', 0),
                price=opportunity.metadata.get('sell_price')
            )
            
            orders = [buy_order, sell_order]
        
        elif opportunity.opportunity_type == OpportunityType.STATISTICAL:
            # Pairs trading orders
            pair1_order = Order(
                opportunity_id=opportunity.opportunity_id,
                exchange=opportunity.metadata.get('exchange', ''),
                symbol=opportunity.metadata.get('pair1_symbol', ''),
                side=OrderSide.BUY if opportunity.metadata.get('position_type') == 'long_spread' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=opportunity.metadata.get('pair1_quantity', 0),
                price=opportunity.metadata.get('pair1_price')
            )
            
            pair2_order = Order(
                opportunity_id=opportunity.opportunity_id,
                exchange=opportunity.metadata.get('exchange', ''),
                symbol=opportunity.metadata.get('pair2_symbol', ''),
                side=OrderSide.SELL if opportunity.metadata.get('position_type') == 'long_spread' else OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=opportunity.metadata.get('pair2_quantity', 0),
                price=opportunity.metadata.get('pair2_price')
            )
            
            orders = [pair1_order, pair2_order]
        
        # Submit orders in parallel
        tasks = [self._submit_order_with_retry(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        success = all(r is True for r in results if not isinstance(r, Exception))
        
        # Calculate metrics
        total_fees = sum(o.fees_paid for o in orders)
        avg_slippage = np.mean([o.slippage for o in orders]) if orders else 0
        
        return ExecutionResult(
            success=success,
            opportunity_id=opportunity.opportunity_id,
            orders=orders,
            total_profit_usd=opportunity.expected_profit_usd if success else 0,
            total_fees_usd=total_fees,
            average_slippage_pct=avg_slippage,
            execution_time_ms=0
        )
    
    async def _submit_order(self, order: Order) -> bool:
        """
        Submete uma ordem para a exchange.
        
        Args:
            order: Ordem a submeter
            
        Returns:
            True se sucesso
        """
        # Check circuit breaker
        if self._is_circuit_broken(order.exchange):
            logger.warning(f"Circuit breaker active for {order.exchange}")
            order.status = OrderStatus.REJECTED
            return False
        
        try:
            # Mark as submitted
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = time.time()
            
            # Get exchange connector
            exchange_connector = self.connector.connectors.get(order.exchange)
            if not exchange_connector:
                raise ValueError(f"No connector for {order.exchange}")
            
            # Submit to exchange (simulated for now)
            # In production, would call actual exchange API
            await asyncio.sleep(0.01)  # Simulate network delay
            
            # Simulate execution
            order.status = OrderStatus.FILLED
            order.filled_at = time.time()
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price or 0
            
            # Calculate fees (simplified)
            fee_rate = 0.001  # 0.1%
            order.fees_paid = order.quantity * order.average_fill_price * fee_rate
            
            # Calculate slippage
            if order.price:
                order.slippage = abs(order.average_fill_price - order.price) / order.price
            
            # Track order
            self.active_orders[order.order_id] = order
            self.total_orders += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.FAILED
            
            # Update circuit breaker
            self.failure_counts[order.exchange] += 1
            if self.failure_counts[order.exchange] >= self.circuit_breaker_threshold:
                self._trigger_circuit_breaker(order.exchange)
            
            return False
    
    async def _submit_order_with_retry(
        self,
        order: Order
    ) -> bool:
        """
        Submete ordem com retry logic.
        
        Args:
            order: Ordem a submeter
            
        Returns:
            True se sucesso
        """
        for attempt in range(order.max_retries):
            try:
                success = await self._submit_order(order)
                if success:
                    return True
                
                # Wait before retry
                if attempt < order.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    await asyncio.sleep(delay)
                    order.retry_count += 1
                    
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
        
        return False
    
    async def _rollback_orders(self, orders: List[Order]) -> None:
        """
        Reverte ordens executadas (para atomic execution).
        
        Args:
            orders: Ordens a reverter
        """
        for order in orders:
            if order.status == OrderStatus.FILLED:
                # Create reverse order
                reverse_order = Order(
                    opportunity_id=order.opportunity_id,
                    exchange=order.exchange,
                    symbol=order.symbol,
                    side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=order.filled_quantity
                )
                
                # Submit reverse order
                await self._submit_order(reverse_order)
                
                logger.info(f"Rolled back order {order.order_id}")
    
    def _is_circuit_broken(self, exchange: str) -> bool:
        """
        Verifica se circuit breaker está ativo.
        
        Args:
            exchange: ID da exchange
            
        Returns:
            True se circuit breaker ativo
        """
        if exchange in self.circuit_breaker_triggered:
            triggered_time = self.circuit_breaker_triggered[exchange]
            if time.time() - triggered_time < self.circuit_breaker_cooldown:
                return True
            else:
                # Reset circuit breaker
                del self.circuit_breaker_triggered[exchange]
                self.failure_counts[exchange] = 0
        
        return False
    
    def _trigger_circuit_breaker(self, exchange: str) -> None:
        """
        Ativa circuit breaker para uma exchange.
        
        Args:
            exchange: ID da exchange
        """
        self.circuit_breaker_triggered[exchange] = time.time()
        logger.warning(f"Circuit breaker triggered for {exchange}")
    
    async def _get_lock(self, key: str):
        """
        Context manager para lock distribuído.
        
        Args:
            key: Chave do lock
        """
        # Simplified lock implementation
        # In production, would use Redis or similar
        lock = asyncio.Lock()
        self.execution_locks[key] = lock
        return lock
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancela uma ordem.
        
        Args:
            order_id: ID da ordem
            
        Returns:
            True se cancelado com sucesso
        """
        order = self.active_orders.get(order_id)
        if not order:
            return False
        
        if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Cancelled order {order_id}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de execução."""
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'success_rate': (self.successful_orders / self.total_orders * 100) if self.total_orders > 0 else 0,
            'total_volume_usd': self.total_volume_usd,
            'avg_execution_ms': np.mean(self.execution_times) if self.execution_times else 0,
            'active_orders': len(self.active_orders),
            'circuit_breakers': list(self.circuit_breaker_triggered.keys())
        }
    
    @jit(nopython=True)
    def _calculate_weighted_average_price(
        self,
        prices: np.ndarray,
        quantities: np.ndarray
    ) -> float:
        """
        Calcula preço médio ponderado (VWAP).
        
        Args:
            prices: Array de preços
            quantities: Array de quantidades
            
        Returns:
            Preço médio ponderado
        """
        if len(prices) == 0 or len(quantities) == 0:
            return 0.0
        
        total_value = np.sum(prices * quantities)
        total_quantity = np.sum(quantities)
        
        if total_quantity > 0:
            return total_value / total_quantity
        
        return 0.0