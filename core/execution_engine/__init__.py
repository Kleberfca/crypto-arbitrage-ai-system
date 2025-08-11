"""
Execution Engine - Motor de execução de ordens
Gerencia execução, pre-funding, otimização e controle de slippage
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from core.execution_engine.order_manager import (
    OrderManager,
    Order,
    OrderStatus,
    OrderType,
    ExecutionResult
)

from core.execution_engine.pre_funding_manager import (
    PreFundingManager,
    FundAllocation,
    RebalanceRequest,
    AllocationStrategy
)

from core.execution_engine.execution_optimizer import (
    ExecutionOptimizer,
    ExecutionPlan,
    RoutingDecision,
    OptimizationStrategy
)

from core.execution_engine.slippage_controller import (
    SlippageController,
    SlippageMetrics,
    SlippagePredictor,
    SlippageLimit
)

__all__ = [
    # Order Manager
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'ExecutionResult',
    
    # Pre-Funding Manager
    'PreFundingManager',
    'FundAllocation',
    'RebalanceRequest',
    'AllocationStrategy',
    
    # Execution Optimizer
    'ExecutionOptimizer',
    'ExecutionPlan',
    'RoutingDecision',
    'OptimizationStrategy',
    
    # Slippage Controller
    'SlippageController',
    'SlippageMetrics',
    'SlippagePredictor',
    'SlippageLimit'
]