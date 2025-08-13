"""
Strategy Routes - Endpoints para gerenciamento de estratégias
Configuração, performance e controle de estratégias de arbitragem
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import logging
import time

from api.models.requests import (
    UpdateStrategyConfigRequest,
    BacktestRequest,
    ScanOpportunitiesRequest
)
from api.models.responses import (
    StrategyConfig,
    StrategyPerformance,
    BacktestResult,
    ArbitrageOpportunity,
    SuccessResponse,
    ErrorResponse
)
from config.settings import settings, Strategy
from core.arbitrage_engine.spatial_arbitrage import SpatialArbitrageEngine
# from core.arbitrage_engine.statistical_arbitrage import StatisticalArbitrageEngine
# from core.arbitrage_engine.triangular_arbitrage import TriangularArbitrageEngine
from core.backtesting.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/strategies", tags=["Strategies"])


# ============================================
# Strategy Configuration
# ============================================

@router.get("/", response_model=List[StrategyConfig])
async def get_strategies():
    """
    Get all strategy configurations.
    
    Returns list of all available strategies with their current configurations.
    """
    strategies = []
    
    # Spatial arbitrage
    if Strategy.SPATIAL in settings.enabled_strategies:
        strategies.append(StrategyConfig(
            name="spatial_arbitrage",
            enabled=True,
            capital_allocation_pct=settings.capital.spatial_strategy_pct,
            capital_allocated_usd=10000 * settings.capital.spatial_strategy_pct,  # Mock
            min_profit_pct=0.3,
            max_position_size=settings.risk_limits.max_position_size_pct,
            confidence_threshold=0.75,
            current_positions=0,
            daily_profit_usd=0,
            total_profit_usd=0,
            win_rate_pct=0,
            parameters={
                "min_spread_pct": 0.3,
                "max_latency_ms": 30,
                "use_pre_funding": True
            },
            last_trade=None,
            status="active"
        ))
    
    # Statistical arbitrage
    if Strategy.STATISTICAL in settings.enabled_strategies:
        strategies.append(StrategyConfig(
            name="statistical_arbitrage",
            enabled=True,
            capital_allocation_pct=settings.capital.statistical_strategy_pct,
            capital_allocated_usd=10000 * settings.capital.statistical_strategy_pct,
            min_profit_pct=0.5,
            max_position_size=settings.risk_limits.max_position_size_pct,
            confidence_threshold=0.70,
            current_positions=0,
            daily_profit_usd=0,
            total_profit_usd=0,
            win_rate_pct=0,
            parameters={
                "z_score_entry": 2.0,
                "z_score_exit": 0.5,
                "lookback_period": 100,
                "use_kalman_filter": True
            },
            last_trade=None,
            status="active"
        ))
    
    # Triangular arbitrage
    if Strategy.TRIANGULAR in settings.enabled_strategies:
        strategies.append(StrategyConfig(
            name="triangular_arbitrage",
            enabled=True,
            capital_allocation_pct=settings.capital.triangular_strategy_pct,
            capital_allocated_usd=10000 * settings.capital.triangular_strategy_pct,
            min_profit_pct=0.15,
            max_position_size=settings.risk_limits.max_position_size_pct,
            confidence_threshold=0.80,
            current_positions=0,
            daily_profit_usd=0,
            total_profit_usd=0,
            win_rate_pct=0,
            parameters={
                "min_profit_pct": 0.15,
                "max_path_length": 4,
                "execution_timeout_ms": 100
            },
            last_trade=None,
            status="active"
        ))
    
    return strategies


@router.get("/{strategy_name}", response_model=StrategyConfig)
async def get_strategy(strategy_name: str):
    """
    Get specific strategy configuration.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy configuration and current status
    """
    strategies = await get_strategies()
    
    for strategy in strategies:
        if strategy.name == strategy_name:
            return strategy
    
    raise HTTPException(
        status_code=404,
        detail=f"Strategy '{strategy_name}' not found"
    )


@router.put("/{strategy_name}", response_model=StrategyConfig)
async def update_strategy(
    strategy_name: str,
    request: UpdateStrategyConfigRequest
):
    """
    Update strategy configuration.
    
    Args:
        strategy_name: Name of the strategy
        request: Configuration updates
        
    Returns:
        Updated strategy configuration
    """
    # Get current strategy
    strategy = await get_strategy(strategy_name)
    
    # Update fields if provided
    if request.enabled is not None:
        strategy.enabled = request.enabled
    
    if request.capital_allocation_pct is not None:
        strategy.capital_allocation_pct = request.capital_allocation_pct
        strategy.capital_allocated_usd = 10000 * request.capital_allocation_pct  # Mock
    
    if request.min_profit_pct is not None:
        strategy.min_profit_pct = request.min_profit_pct
    
    if request.max_position_size is not None:
        strategy.max_position_size = request.max_position_size
    
    if request.confidence_threshold is not None:
        strategy.confidence_threshold = request.confidence_threshold
    
    if request.parameters:
        strategy.parameters.update(request.parameters)
    
    # TODO: Actually update configuration in the strategy engines
    
    logger.info(f"Updated strategy configuration: {strategy_name}")
    
    return strategy


@router.post("/{strategy_name}/enable", response_model=SuccessResponse)
async def enable_strategy(strategy_name: str):
    """
    Enable a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Success response
    """
    # TODO: Actually enable the strategy
    
    logger.info(f"Enabled strategy: {strategy_name}")
    
    return SuccessResponse(
        message=f"Strategy '{strategy_name}' enabled successfully"
    )


@router.post("/{strategy_name}/disable", response_model=SuccessResponse)
async def disable_strategy(strategy_name: str):
    """
    Disable a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Success response
    """
    # TODO: Actually disable the strategy
    
    logger.info(f"Disabled strategy: {strategy_name}")
    
    return SuccessResponse(
        message=f"Strategy '{strategy_name}' disabled successfully"
    )


# ============================================
# Strategy Performance
# ============================================

@router.get("/{strategy_name}/performance", response_model=StrategyPerformance)
async def get_strategy_performance(
    strategy_name: str,
    period: str = Query(default="24h", pattern="^[1-9][0-9]*[hdwm]$")
):
    """
    Get strategy performance metrics.
    
    Args:
        strategy_name: Name of the strategy
        period: Time period (1h, 24h, 7d, 30d)
        
    Returns:
        Performance metrics for the specified period
    """
    # Mock performance data
    performance = StrategyPerformance(
        strategy=strategy_name,
        period=period,
        total_trades=150,
        successful_trades=105,
        failed_trades=45,
        win_rate_pct=70.0,
        total_profit_usd=1250.50,
        avg_profit_per_trade=8.33,
        best_trade_usd=125.00,
        worst_trade_usd=-35.50,
        sharpe_ratio=2.8,
        sortino_ratio=3.2,
        max_drawdown_pct=3.5,
        profit_factor=2.1,
        expectancy=8.33,
        daily_stats={
            "avg_trades": 25,
            "avg_profit": 208.42,
            "best_day": 450.00,
            "worst_day": -125.00
        }
    )
    
    return performance


@router.get("/performance/comparison", response_model=List[StrategyPerformance])
async def compare_strategies_performance(
    period: str = Query(default="24h", pattern="^[1-9][0-9]*[hdwm]$")
):
    """
    Compare performance across all strategies.
    
    Args:
        period: Time period for comparison
        
    Returns:
        Performance metrics for all strategies
    """
    strategies = ["spatial_arbitrage", "statistical_arbitrage", "triangular_arbitrage"]
    performances = []
    
    for strategy in strategies:
        if strategy in ["spatial_arbitrage", "statistical_arbitrage", "triangular_arbitrage"]:
            perf = await get_strategy_performance(strategy, period)
            performances.append(perf)
    
    return performances


# ============================================
# Opportunity Detection
# ============================================

@router.post("/opportunities/scan", response_model=List[ArbitrageOpportunity])
async def scan_opportunities(
    request: ScanOpportunitiesRequest,
    background_tasks: BackgroundTasks
):
    """
    Scan for arbitrage opportunities across strategies.
    
    Args:
        request: Scan parameters
        background_tasks: Background task manager
        
    Returns:
        List of detected opportunities
    """
    opportunities = []
    
    # TODO: Actually scan using strategy engines
    # For now, return mock data
    
    if not request.strategies or Strategy.SPATIAL in request.strategies:
        opportunities.append(ArbitrageOpportunity(
            id=f"spatial_{int(time.time())}",
            timestamp=time.time(),
            strategy="spatial",
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="okx",
            buy_price=50000.00,
            sell_price=50150.00,
            spread_pct=0.30,
            gross_profit_pct=0.30,
            net_profit_pct=0.25,
            net_profit_usd=125.00,
            executable_volume=1.0,
            confidence=0.85,
            risk_score=0.2,
            ml_prediction=0.88,
            expires_in_seconds=5.0,
            execution_estimate_ms=25.0,
            fees={"binance": 0.001, "okx": 0.0008},
            metadata={"pre_funded": True}
        ))
    
    # Filter by criteria
    filtered = []
    for opp in opportunities:
        if opp.net_profit_pct >= request.min_profit_pct:
            if opp.confidence >= request.min_confidence:
                if opp.risk_score <= request.max_risk_score:
                    filtered.append(opp)
    
    # Sort by profit and limit
    filtered.sort(key=lambda x: x.net_profit_usd, reverse=True)
    filtered = filtered[:request.limit]
    
    return filtered


# ============================================
# Backtesting
# ============================================

@router.post("/{strategy_name}/backtest", response_model=BacktestResult)
async def run_backtest(
    strategy_name: str,
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """
    Run backtest for a strategy.
    
    Args:
        strategy_name: Name of the strategy
        request: Backtest parameters
        background_tasks: Background task manager
        
    Returns:
        Backtest results with detailed metrics
    """
    # Validate strategy
    if strategy_name not in ["spatial_arbitrage", "statistical_arbitrage", "triangular_arbitrage"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy_name}"
        )
    
    # TODO: Actually run backtest using BacktestEngine
    # For now, return mock results
    
    result = BacktestResult(
        strategy=strategy_name,
        period_start=request.start_date,
        period_end=request.end_date,
        initial_capital=request.initial_capital,
        final_capital=12500.00,
        total_return_pct=25.0,
        annualized_return_pct=30.0,
        total_trades=500,
        win_rate_pct=68.0,
        profit_factor=2.1,
        sharpe_ratio=2.5,
        sortino_ratio=3.0,
        calmar_ratio=2.8,
        max_drawdown_pct=4.5,
        var_95=250.00,
        cvar_95=350.00,
        walk_forward_results=[
            {"window": 1, "return": 2.5, "sharpe": 2.3},
            {"window": 2, "return": 3.1, "sharpe": 2.7}
        ],
        monte_carlo_results={
            "median_return": 24.5,
            "percentile_5": 15.0,
            "percentile_95": 35.0,
            "probability_profit": 0.92
        },
        trade_log=[]  # Omitted for brevity
    )
    
    logger.info(f"Completed backtest for strategy: {strategy_name}")
    
    return result


@router.post("/backtest/all", response_model=List[BacktestResult])
async def run_all_backtests(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """
    Run backtest for all strategies.
    
    Args:
        request: Backtest parameters
        background_tasks: Background task manager
        
    Returns:
        Backtest results for all strategies
    """
    strategies = ["spatial_arbitrage", "statistical_arbitrage", "triangular_arbitrage"]
    results = []
    
    for strategy in strategies:
        result = await run_backtest(strategy, request, background_tasks)
        results.append(result)
    
    return results


# ============================================
# Strategy Control
# ============================================

@router.post("/{strategy_name}/reset", response_model=SuccessResponse)
async def reset_strategy(strategy_name: str):
    """
    Reset strategy to default configuration.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Success response
    """
    # TODO: Actually reset the strategy
    
    logger.info(f"Reset strategy: {strategy_name}")
    
    return SuccessResponse(
        message=f"Strategy '{strategy_name}' reset to defaults successfully"
    )


@router.post("/{strategy_name}/optimize", response_model=SuccessResponse)
async def optimize_strategy(
    strategy_name: str,
    background_tasks: BackgroundTasks
):
    """
    Run optimization for strategy parameters.
    
    Args:
        strategy_name: Name of the strategy
        background_tasks: Background task manager
        
    Returns:
        Success response (optimization runs in background)
    """
    # TODO: Actually run optimization in background
    
    logger.info(f"Started optimization for strategy: {strategy_name}")
    
    return SuccessResponse(
        message=f"Optimization started for strategy '{strategy_name}'",
        data={"task_id": f"opt_{strategy_name}_{int(time.time())}"}
    )