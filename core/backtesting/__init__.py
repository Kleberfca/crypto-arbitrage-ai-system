"""
Backtesting System - Sistema completo de backtesting
Walk-forward optimization, Monte Carlo e stress testing
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from core.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    TradeRecord
)

from core.backtesting.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    OptimizationWindow,
    WalkForwardResult
)

from core.backtesting.monte_carlo import (
    MonteCarloSimulator,
    SimulationConfig,
    SimulationResult,
    ConfidenceInterval
)

from core.backtesting.stress_testing import (
    StressTester,
    StressScenario,
    StressTestResult,
    ScenarioType
)

__all__ = [
    # Backtest Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'TradeRecord',
    
    # Walk Forward
    'WalkForwardOptimizer',
    'WalkForwardConfig',
    'OptimizationWindow',
    'WalkForwardResult',
    
    # Monte Carlo
    'MonteCarloSimulator',
    'SimulationConfig',
    'SimulationResult',
    'ConfidenceInterval',
    
    # Stress Testing
    'StressTester',
    'StressScenario',
    'StressTestResult',
    'ScenarioType'
]