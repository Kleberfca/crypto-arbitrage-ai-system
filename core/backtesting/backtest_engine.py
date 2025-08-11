"""
Backtest Engine - Motor principal de backtesting
Simula estratégias com dados históricos e calcula métricas
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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numba import jit
import redis.asyncio as redis

from config.settings import settings
from core.arbitrage_engine.base_arbitrage import BaseOpportunity, OpportunityType
from core.risk_management.risk_calculator import RiskCalculator, RiskMetrics


logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Registro de uma operação no backtest."""
    
    # Trade identification
    trade_id: str
    opportunity_id: str
    strategy: OpportunityType
    
    # Timing
    entry_time: float
    exit_time: Optional[float] = None
    duration_seconds: float = 0
    
    # Financial
    entry_price: float = 0
    exit_price: float = 0
    position_size: float = 0
    
    # Results
    pnl: float = 0  # Profit and Loss
    pnl_pct: float = 0
    fees_paid: float = 0
    slippage: float = 0
    
    # Risk metrics at entry
    risk_score: float = 0
    confidence: float = 0
    
    # Status
    is_winner: bool = False
    is_closed: bool = False
    
    @property
    def net_pnl(self) -> float:
        """PnL líquido após fees."""
        return self.pnl - self.fees_paid
    
    @property
    def return_on_capital(self) -> float:
        """Retorno sobre capital."""
        if self.position_size > 0:
            return self.net_pnl / self.position_size
        return 0


@dataclass
class BacktestConfig:
    """Configuração do backtest."""
    
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Capital
    initial_capital: float = 10000
    
    # Strategies to test
    enabled_strategies: List[OpportunityType] = field(
        default_factory=lambda: [
            OpportunityType.SPATIAL,
            OpportunityType.STATISTICAL,
            OpportunityType.TRIANGULAR
        ]
    )
    
    # Risk parameters
    max_position_size: float = 0.02  # 2% of capital
    max_daily_loss: float = 0.03  # 3%
    max_drawdown: float = 0.05  # 5%
    
    # Execution assumptions
    slippage_bps: float = 5  # 0.05%
    fee_rate: float = 0.001  # 0.1%
    
    # Data parameters
    tick_interval: int = 1  # Seconds
    use_real_orderbooks: bool = True
    
    # Optimization
    optimize_parameters: bool = False
    optimization_metric: str = "sharpe_ratio"


@dataclass
class BacktestResult:
    """Resultado do backtest."""
    
    # Configuration
    config: BacktestConfig
    
    # Time
    total_days: float = 0
    total_trades: int = 0
    
    # Returns
    total_return: float = 0
    annualized_return: float = 0
    
    # Risk metrics
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    max_drawdown: float = 0
    var_95: float = 0
    
    # Trade statistics
    win_rate: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    profit_factor: float = 0
    expectancy: float = 0
    
    # Strategy breakdown
    returns_by_strategy: Dict[str, float] = field(default_factory=dict)
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    win_rates_by_strategy: Dict[str, float] = field(default_factory=dict)
    
    # Time series
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # Trade records
    trades: List[TradeRecord] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Retorna resumo dos resultados."""
        return {
            'total_return': f"{self.total_return:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'total_trades': self.total_trades,
            'profit_factor': f"{self.profit_factor:.2f}"
        }


class BacktestEngine:
    """
    Motor principal de backtesting.
    Simula estratégias com dados históricos.
    """
    
    def __init__(self):
        """Inicializa motor de backtest."""
        # Configuration
        self.config = None
        
        # Data management
        self.historical_data = {}
        self.orderbook_snapshots = {}
        self.current_timestamp = 0
        
        # Portfolio tracking
        self.capital = 0
        self.positions = {}
        self.open_trades = {}
        self.closed_trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.high_water_mark = 0
        
        # Risk management
        self.risk_calculator = RiskCalculator()
        self.current_risk_metrics = None
        
        # Strategy engines (simplified for backtest)
        self.strategy_engines = {}
        
        # Cache
        self.opportunity_cache = defaultdict(list)
        
        # Redis for data storage
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
    
    async def run_backtest(
        self,
        config: BacktestConfig,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Executa backtest completo.
        
        Args:
            config: Configuração do backtest
            historical_data: Dados históricos opcionais
            
        Returns:
            Resultado do backtest
        """
        start_time = time.perf_counter()
        
        logger.info(
            f"Starting backtest",
            start_date=config.start_date.isoformat(),
            end_date=config.end_date.isoformat(),
            capital=f"${config.initial_capital:.2f}"
        )
        
        # Initialize
        self.config = config
        self.capital = config.initial_capital
        self.high_water_mark = config.initial_capital
        
        # Load historical data
        if historical_data:
            self.historical_data = historical_data
        else:
            await self._load_historical_data()
        
        # Initialize strategy engines
        self._initialize_strategies()
        
        # Run simulation
        await self._run_simulation()
        
        # Calculate final metrics
        result = self._calculate_results()
        
        # Log summary
        execution_time = time.perf_counter() - start_time
        logger.info(
            f"Backtest completed",
            execution_time=f"{execution_time:.2f}s",
            total_return=f"{result.total_return:.2%}",
            sharpe_ratio=f"{result.sharpe_ratio:.2f}",
            max_drawdown=f"{result.max_drawdown:.2%}"
        )
        
        return result
    
    async def _load_historical_data(self) -> None:
        """Carrega dados históricos."""
        # In production, would load from database or files
        # For now, generate synthetic data
        
        logger.info("Loading historical data...")
        
        # Generate timestamps
        current = self.config.start_date
        timestamps = []
        
        while current <= self.config.end_date:
            timestamps.append(current.timestamp())
            current += timedelta(seconds=self.config.tick_interval)
        
        # Generate synthetic price data
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        for symbol in symbols:
            # Random walk with drift
            prices = self._generate_synthetic_prices(
                len(timestamps),
                initial_price=1000 if 'BTC' in symbol else 100,
                volatility=0.02,
                drift=0.0001
            )
            
            self.historical_data[symbol] = {
                'timestamps': timestamps,
                'prices': prices,
                'volumes': np.random.uniform(100, 1000, len(timestamps))
            }
        
        logger.info(f"Loaded data for {len(symbols)} symbols, {len(timestamps)} timestamps")
    
    @jit(nopython=True)
    def _generate_synthetic_prices(
        self,
        n_points: int,
        initial_price: float,
        volatility: float,
        drift: float
    ) -> np.ndarray:
        """
        Gera preços sintéticos usando random walk.
        
        Args:
            n_points: Número de pontos
            initial_price: Preço inicial
            volatility: Volatilidade
            drift: Drift (tendência)
            
        Returns:
            Array de preços
        """
        prices = np.zeros(n_points)
        prices[0] = initial_price
        
        for i in range(1, n_points):
            # Geometric Brownian Motion
            random_shock = np.random.normal(0, 1)
            prices[i] = prices[i-1] * (1 + drift + volatility * random_shock)
            
            # Ensure positive prices
            prices[i] = max(prices[i], initial_price * 0.1)
        
        return prices
    
    def _initialize_strategies(self) -> None:
        """Inicializa engines de estratégia para backtest."""
        # Simplified strategy engines for backtesting
        # In production, would use actual strategy engines
        
        for strategy in self.config.enabled_strategies:
            self.strategy_engines[strategy] = {
                'min_profit_pct': 0.15,  # 0.15% minimum
                'confidence_threshold': 0.7,
                'cooldown_seconds': 60
            }
        
        logger.info(f"Initialized {len(self.strategy_engines)} strategy engines")
    
    async def _run_simulation(self) -> None:
        """Executa simulação tick por tick."""
        timestamps = list(self.historical_data.values())[0]['timestamps']
        
        for i, timestamp in enumerate(timestamps):
            self.current_timestamp = timestamp
            
            # Update market data
            market_data = self._get_market_snapshot(i)
            
            # Detect opportunities
            opportunities = await self._detect_opportunities(market_data)
            
            # Execute opportunities
            for opportunity in opportunities:
                await self._execute_opportunity(opportunity)
            
            # Update open positions
            self._update_positions(market_data)
            
            # Close positions if needed
            await self._check_exit_conditions(market_data)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Risk management
            if i % 100 == 0:  # Check every 100 ticks
                await self._apply_risk_management()
            
            # Daily processing
            if i % 86400 == 0:  # Daily
                self._process_daily_metrics()
        
        # Close all remaining positions
        await self._close_all_positions()
    
    def _get_market_snapshot(self, index: int) -> Dict[str, Any]:
        """
        Obtém snapshot do mercado.
        
        Args:
            index: Índice do timestamp
            
        Returns:
            Dados de mercado
        """
        snapshot = {}
        
        for symbol, data in self.historical_data.items():
            if index < len(data['prices']):
                snapshot[symbol] = {
                    'price': data['prices'][index],
                    'volume': data['volumes'][index],
                    'timestamp': data['timestamps'][index]
                }
        
        return snapshot
    
    async def _detect_opportunities(
        self,
        market_data: Dict[str, Any]
    ) -> List[BaseOpportunity]:
        """
        Detecta oportunidades de arbitragem.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Lista de oportunidades
        """
        opportunities = []
        
        # Simplified opportunity detection for backtest
        
        # Spatial arbitrage
        if OpportunityType.SPATIAL in self.config.enabled_strategies:
            spatial_opps = self._detect_spatial_opportunities(market_data)
            opportunities.extend(spatial_opps)
        
        # Statistical arbitrage
        if OpportunityType.STATISTICAL in self.config.enabled_strategies:
            stat_opps = self._detect_statistical_opportunities(market_data)
            opportunities.extend(stat_opps)
        
        # Triangular arbitrage
        if OpportunityType.TRIANGULAR in self.config.enabled_strategies:
            tri_opps = self._detect_triangular_opportunities(market_data)
            opportunities.extend(tri_opps)
        
        return opportunities
    
    def _detect_spatial_opportunities(
        self,
        market_data: Dict[str, Any]
    ) -> List[BaseOpportunity]:
        """
        Detecta oportunidades de arbitragem espacial.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Oportunidades espaciais
        """
        opportunities = []
        
        # Simplified: look for price differences
        symbols = list(market_data.keys())
        
        for symbol in symbols:
            price = market_data[symbol]['price']
            
            # Simulate price difference between exchanges
            price_diff_pct = np.random.uniform(-0.005, 0.005)  # ±0.5%
            
            if abs(price_diff_pct) > 0.002:  # 0.2% threshold
                opportunity = BaseOpportunity(
                    opportunity_id=f"spatial_{symbol}_{int(self.current_timestamp)}",
                    opportunity_type=OpportunityType.SPATIAL,
                    expected_profit_pct=abs(price_diff_pct) - self.config.fee_rate * 2,
                    expected_profit_usd=self.capital * 0.01 * (abs(price_diff_pct) - self.config.fee_rate * 2),
                    confidence=0.8,
                    risk_score=0.2,
                    required_capital=self.capital * 0.01,
                    estimated_execution_time_ms=50,
                    metadata={
                        'symbol': symbol,
                        'price': price,
                        'price_diff_pct': price_diff_pct
                    }
                )
                
                if opportunity.expected_profit_pct > 0:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_statistical_opportunities(
        self,
        market_data: Dict[str, Any]
    ) -> List[BaseOpportunity]:
        """
        Detecta oportunidades de arbitragem estatística.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Oportunidades estatísticas
        """
        opportunities = []
        
        # Simplified: random mean reversion signals
        if np.random.random() < 0.1:  # 10% chance
            opportunity = BaseOpportunity(
                opportunity_id=f"stat_{int(self.current_timestamp)}",
                opportunity_type=OpportunityType.STATISTICAL,
                expected_profit_pct=np.random.uniform(0.001, 0.005),
                expected_profit_usd=self.capital * 0.02 * np.random.uniform(0.001, 0.005),
                confidence=0.75,
                risk_score=0.3,
                required_capital=self.capital * 0.02,
                estimated_execution_time_ms=1000,
                metadata={
                    'zscore': np.random.uniform(2, 3),
                    'half_life': np.random.uniform(1, 5)
                }
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_triangular_opportunities(
        self,
        market_data: Dict[str, Any]
    ) -> List[BaseOpportunity]:
        """
        Detecta oportunidades de arbitragem triangular.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Oportunidades triangulares
        """
        opportunities = []
        
        # Simplified: random triangular opportunities
        if np.random.random() < 0.05:  # 5% chance
            opportunity = BaseOpportunity(
                opportunity_id=f"tri_{int(self.current_timestamp)}",
                opportunity_type=OpportunityType.TRIANGULAR,
                expected_profit_pct=np.random.uniform(0.0015, 0.003),
                expected_profit_usd=self.capital * 0.01 * np.random.uniform(0.0015, 0.003),
                confidence=0.9,
                risk_score=0.1,
                required_capital=self.capital * 0.01,
                estimated_execution_time_ms=100,
                metadata={
                    'path': ['BTC/USDT', 'ETH/BTC', 'ETH/USDT']
                }
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _execute_opportunity(
        self,
        opportunity: BaseOpportunity
    ) -> None:
        """
        Executa uma oportunidade.
        
        Args:
            opportunity: Oportunidade a executar
        """
        # Check capital
        if opportunity.required_capital > self.capital * self.config.max_position_size:
            return
        
        # Check if we already have too many positions
        if len(self.open_trades) >= 10:
            return
        
        # Apply slippage
        actual_profit_pct = opportunity.expected_profit_pct - self.config.slippage_bps / 10000
        
        if actual_profit_pct <= 0:
            return
        
        # Create trade record
        trade = TradeRecord(
            trade_id=f"trade_{len(self.closed_trades) + len(self.open_trades)}",
            opportunity_id=opportunity.opportunity_id,
            strategy=opportunity.opportunity_type,
            entry_time=self.current_timestamp,
            position_size=opportunity.required_capital,
            risk_score=opportunity.risk_score,
            confidence=opportunity.confidence
        )
        
        # Deduct capital
        self.capital -= opportunity.required_capital
        
        # Add to open trades
        self.open_trades[trade.trade_id] = trade
        
        logger.debug(
            f"Executed trade",
            trade_id=trade.trade_id,
            strategy=trade.strategy.value,
            size=f"${trade.position_size:.2f}"
        )
    
    def _update_positions(self, market_data: Dict[str, Any]) -> None:
        """
        Atualiza posições abertas.
        
        Args:
            market_data: Dados de mercado
        """
        for trade_id, trade in self.open_trades.items():
            # Simulate P&L calculation
            # In production, would use actual price changes
            
            elapsed_time = self.current_timestamp - trade.entry_time
            
            # Simulate mean reversion or trend
            if trade.strategy == OpportunityType.STATISTICAL:
                # Mean reversion - profit increases then decreases
                optimal_time = 3600  # 1 hour
                if elapsed_time < optimal_time:
                    pnl_pct = (elapsed_time / optimal_time) * 0.003
                else:
                    pnl_pct = 0.003 * (2 - elapsed_time / optimal_time)
            else:
                # Quick arbitrage - immediate profit
                pnl_pct = np.random.uniform(0.001, 0.003)
            
            # Apply randomness
            pnl_pct += np.random.normal(0, 0.0005)
            
            # Update trade
            trade.pnl = trade.position_size * pnl_pct
            trade.pnl_pct = pnl_pct
            trade.fees_paid = trade.position_size * self.config.fee_rate
    
    async def _check_exit_conditions(
        self,
        market_data: Dict[str, Any]
    ) -> None:
        """
        Verifica condições de saída.
        
        Args:
            market_data: Dados de mercado
        """
        trades_to_close = []
        
        for trade_id, trade in self.open_trades.items():
            should_close = False
            
            # Time-based exit
            elapsed_time = self.current_timestamp - trade.entry_time
            
            if trade.strategy == OpportunityType.SPATIAL:
                # Quick exit for spatial
                if elapsed_time > 60:  # 1 minute
                    should_close = True
                    
            elif trade.strategy == OpportunityType.STATISTICAL:
                # Longer holding for statistical
                if elapsed_time > 3600:  # 1 hour
                    should_close = True
                    
            elif trade.strategy == OpportunityType.TRIANGULAR:
                # Immediate exit for triangular
                if elapsed_time > 10:  # 10 seconds
                    should_close = True
            
            # Stop loss
            if trade.pnl_pct < -0.002:  # -0.2%
                should_close = True
            
            # Take profit
            if trade.pnl_pct > 0.005:  # 0.5%
                should_close = True
            
            if should_close:
                trades_to_close.append(trade_id)
        
        # Close trades
        for trade_id in trades_to_close:
            await self._close_trade(trade_id)
    
    async def _close_trade(self, trade_id: str) -> None:
        """
        Fecha uma operação.
        
        Args:
            trade_id: ID da operação
        """
        if trade_id not in self.open_trades:
            return
        
        trade = self.open_trades[trade_id]
        
        # Finalize trade
        trade.exit_time = self.current_timestamp
        trade.duration_seconds = trade.exit_time - trade.entry_time
        trade.is_closed = True
        trade.is_winner = trade.net_pnl > 0
        
        # Return capital
        self.capital += trade.position_size + trade.net_pnl
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]
        
        logger.debug(
            f"Closed trade",
            trade_id=trade_id,
            pnl=f"${trade.net_pnl:.2f}",
            duration=f"{trade.duration_seconds:.0f}s"
        )
    
    async def _close_all_positions(self) -> None:
        """Fecha todas as posições abertas."""
        trade_ids = list(self.open_trades.keys())
        
        for trade_id in trade_ids:
            await self._close_trade(trade_id)
    
    def _update_portfolio_metrics(self) -> None:
        """Atualiza métricas do portfolio."""
        # Calculate total equity
        total_equity = self.capital
        
        for trade in self.open_trades.values():
            total_equity += trade.position_size + trade.pnl - trade.fees_paid
        
        # Update equity curve
        self.equity_curve.append(total_equity)
        
        # Update high water mark
        if total_equity > self.high_water_mark:
            self.high_water_mark = total_equity
        
        # Calculate drawdown
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - total_equity) / self.high_water_mark
        else:
            drawdown = 0
    
    async def _apply_risk_management(self) -> None:
        """Aplica gestão de risco."""
        # Calculate current risk metrics
        if len(self.equity_curve) > 20:
            portfolio_values = self.equity_curve[-20:]
            
            # Simple risk checks
            current_equity = self.equity_curve[-1]
            starting_equity = self.config.initial_capital
            
            # Check daily loss limit
            if len(self.equity_curve) > 1:
                daily_loss = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                
                if daily_loss < -self.config.max_daily_loss:
                    # Stop trading for the day
                    logger.warning("Daily loss limit reached")
                    await self._close_all_positions()
            
            # Check max drawdown
            drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
            
            if drawdown > self.config.max_drawdown:
                # Reduce position sizes
                logger.warning(f"Max drawdown reached: {drawdown:.2%}")
                await self._close_all_positions()
    
    def _process_daily_metrics(self) -> None:
        """Processa métricas diárias."""
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def _calculate_results(self) -> BacktestResult:
        """
        Calcula resultados finais do backtest.
        
        Returns:
            Resultado do backtest
        """
        result = BacktestResult(config=self.config)
        
        # Time metrics
        result.total_days = (self.config.end_date - self.config.start_date).days
        result.total_trades = len(self.closed_trades)
        
        # Return metrics
        if self.config.initial_capital > 0:
            final_equity = self.equity_curve[-1] if self.equity_curve else self.config.initial_capital
            result.total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
            
            if result.total_days > 0:
                result.annualized_return = (1 + result.total_return) ** (365 / result.total_days) - 1
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            
            # Sharpe ratio
            if np.std(returns_array) > 0:
                result.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                result.sortino_ratio = np.mean(returns_array) / np.std(downside_returns) * np.sqrt(252)
            
            # Max drawdown
            result.max_drawdown = self._calculate_max_drawdown()
            
            # Calmar ratio
            if result.max_drawdown > 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown
            
            # VaR
            result.var_95 = np.percentile(returns_array, 5)
        
        # Trade statistics
        if self.closed_trades:
            winners = [t for t in self.closed_trades if t.is_winner]
            losers = [t for t in self.closed_trades if not t.is_winner]
            
            result.win_rate = len(winners) / len(self.closed_trades)
            
            if winners:
                result.avg_win = np.mean([t.net_pnl for t in winners])
            
            if losers:
                result.avg_loss = abs(np.mean([t.net_pnl for t in losers]))
            
            # Profit factor
            total_wins = sum(t.net_pnl for t in winners) if winners else 0
            total_losses = abs(sum(t.net_pnl for t in losers)) if losers else 1
            
            if total_losses > 0:
                result.profit_factor = total_wins / total_losses
            
            # Expectancy
            result.expectancy = result.win_rate * result.avg_win - (1 - result.win_rate) * result.avg_loss
        
        # Strategy breakdown
        for strategy in self.config.enabled_strategies:
            strategy_trades = [t for t in self.closed_trades if t.strategy == strategy]
            
            if strategy_trades:
                strategy_return = sum(t.net_pnl for t in strategy_trades) / self.config.initial_capital
                result.returns_by_strategy[strategy.value] = strategy_return
                result.trades_by_strategy[strategy.value] = len(strategy_trades)
                
                strategy_winners = [t for t in strategy_trades if t.is_winner]
                result.win_rates_by_strategy[strategy.value] = len(strategy_winners) / len(strategy_trades)
        
        # Time series
        result.equity_curve = self.equity_curve
        result.daily_returns = self.daily_returns
        result.trades = self.closed_trades
        
        return result
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calcula drawdown máximo.
        
        Returns:
            Drawdown máximo
        """
        if not self.equity_curve:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd