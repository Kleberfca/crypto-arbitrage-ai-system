"""
Monte Carlo Simulator - Simulação Monte Carlo para análise de risco
Gera milhares de cenários para avaliar distribuição de resultados
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats

from core.backtesting.backtest_engine import BacktestResult, TradeRecord


logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Intervalo de confiança."""
    
    confidence_level: float  # e.g., 0.95 for 95%
    lower_bound: float
    upper_bound: float
    median: float
    mean: float
    
    @property
    def width(self) -> float:
        """Largura do intervalo."""
        return self.upper_bound - self.lower_bound


@dataclass
class SimulationConfig:
    """Configuração da simulação Monte Carlo."""
    
    # Number of simulations
    n_simulations: int = 10000
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
    # Simulation parameters
    resample_trades: bool = True  # Bootstrap from historical trades
    randomize_returns: bool = True  # Add noise to returns
    randomize_order: bool = True  # Randomize trade order
    
    # Distribution assumptions
    return_distribution: str = "empirical"  # empirical, normal, t-student
    correlation_method: str = "historical"  # historical, random, zero
    
    # Risk parameters
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.95, 0.99]
    )
    
    # Time horizon
    simulation_days: int = 252  # 1 year
    
    # Parallel processing
    use_parallel: bool = True
    n_workers: Optional[int] = None  # None = use all CPUs


@dataclass
class SimulationResult:
    """Resultado da simulação Monte Carlo."""
    
    config: SimulationConfig
    
    # Distribution of returns
    simulated_returns: np.ndarray
    simulated_sharpe_ratios: np.ndarray
    simulated_max_drawdowns: np.ndarray
    
    # Statistics
    mean_return: float
    median_return: float
    std_return: float
    skewness: float
    kurtosis: float
    
    # Percentiles
    percentiles: Dict[float, float]
    
    # Confidence intervals
    return_confidence_intervals: Dict[float, ConfidenceInterval]
    sharpe_confidence_intervals: Dict[float, ConfidenceInterval]
    drawdown_confidence_intervals: Dict[float, ConfidenceInterval]
    
    # Risk metrics
    var_95: float
    cvar_95: float
    probability_of_loss: float
    probability_of_ruin: float  # Probability of losing everything
    
    # Best/Worst scenarios
    best_case_return: float
    worst_case_return: float
    most_likely_return: float
    
    def summary(self) -> Dict[str, Any]:
        """Retorna resumo dos resultados."""
        return {
            'mean_return': f"{self.mean_return:.2%}",
            'median_return': f"{self.median_return:.2%}",
            'std_return': f"{self.std_return:.2%}",
            'var_95': f"{self.var_95:.2%}",
            'probability_of_loss': f"{self.probability_of_loss:.2%}",
            'best_case': f"{self.best_case_return:.2%}",
            'worst_case': f"{self.worst_case_return:.2%}",
            '95_ci': f"[{self.return_confidence_intervals[0.95].lower_bound:.2%}, "
                     f"{self.return_confidence_intervals[0.95].upper_bound:.2%}]"
        }


class MonteCarloSimulator:
    """
    Simulador Monte Carlo para análise de risco e incerteza.
    """
    
    def __init__(self):
        """Inicializa simulador Monte Carlo."""
        self.historical_trades = []
        self.historical_returns = []
        self.correlation_matrix = None
        
    async def run_simulation(
        self,
        backtest_result: BacktestResult,
        config: Optional[SimulationConfig] = None
    ) -> SimulationResult:
        """
        Executa simulação Monte Carlo.
        
        Args:
            backtest_result: Resultado de backtest histórico
            config: Configuração da simulação
            
        Returns:
            Resultado da simulação
        """
        start_time = time.perf_counter()
        
        if config is None:
            config = SimulationConfig()
        
        logger.info(
            f"Starting Monte Carlo simulation",
            n_simulations=config.n_simulations,
            days=config.simulation_days
        )
        
        # Set random seed
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        # Extract historical data
        self._extract_historical_data(backtest_result)
        
        # Run simulations
        if config.use_parallel and config.n_simulations > 1000:
            simulation_results = await self._run_parallel_simulations(config)
        else:
            simulation_results = self._run_sequential_simulations(config)
        
        # Calculate statistics
        result = self._calculate_statistics(simulation_results, config)
        
        execution_time = time.perf_counter() - start_time
        logger.info(
            f"Monte Carlo simulation completed",
            execution_time=f"{execution_time:.2f}s",
            mean_return=f"{result.mean_return:.2%}",
            var_95=f"{result.var_95:.2%}"
        )
        
        return result
    
    def _extract_historical_data(self, backtest_result: BacktestResult) -> None:
        """
        Extrai dados históricos do backtest.
        
        Args:
            backtest_result: Resultado do backtest
        """
        self.historical_trades = backtest_result.trades
        self.historical_returns = backtest_result.daily_returns
        
        # Calculate correlation matrix if needed
        if len(self.historical_returns) > 20:
            # Simple correlation estimation
            returns_array = np.array(self.historical_returns)
            self.correlation_matrix = np.corrcoef(returns_array.reshape(-1, 1).T)
    
    def _run_sequential_simulations(
        self,
        config: SimulationConfig
    ) -> Dict[str, np.ndarray]:
        """
        Executa simulações sequencialmente.
        
        Args:
            config: Configuração
            
        Returns:
            Resultados das simulações
        """
        simulated_returns = np.zeros(config.n_simulations)
        simulated_sharpes = np.zeros(config.n_simulations)
        simulated_drawdowns = np.zeros(config.n_simulations)
        
        for i in range(config.n_simulations):
            # Run single simulation
            sim_result = self._run_single_simulation(config)
            
            simulated_returns[i] = sim_result['total_return']
            simulated_sharpes[i] = sim_result['sharpe_ratio']
            simulated_drawdowns[i] = sim_result['max_drawdown']
        
        return {
            'returns': simulated_returns,
            'sharpes': simulated_sharpes,
            'drawdowns': simulated_drawdowns
        }
    
    async def _run_parallel_simulations(
        self,
        config: SimulationConfig
    ) -> Dict[str, np.ndarray]:
        """
        Executa simulações em paralelo.
        
        Args:
            config: Configuração
            
        Returns:
            Resultados das simulações
        """
        n_workers = config.n_workers or mp.cpu_count()
        
        # Split simulations among workers
        simulations_per_worker = config.n_simulations // n_workers
        remainder = config.n_simulations % n_workers
        
        tasks = []
        for i in range(n_workers):
            n_sims = simulations_per_worker + (1 if i < remainder else 0)
            
            if n_sims > 0:
                task = self._worker_simulate(config, n_sims)
                tasks.append(task)
        
        # Wait for all workers
        results = await asyncio.gather(*tasks)
        
        # Combine results
        all_returns = np.concatenate([r['returns'] for r in results])
        all_sharpes = np.concatenate([r['sharpes'] for r in results])
        all_drawdowns = np.concatenate([r['drawdowns'] for r in results])
        
        return {
            'returns': all_returns,
            'sharpes': all_sharpes,
            'drawdowns': all_drawdowns
        }
    
    async def _worker_simulate(
        self,
        config: SimulationConfig,
        n_simulations: int
    ) -> Dict[str, np.ndarray]:
        """
        Worker para simulação paralela.
        
        Args:
            config: Configuração
            n_simulations: Número de simulações
            
        Returns:
            Resultados do worker
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._run_worker_simulations,
                config,
                n_simulations
            )
            
            return await future
    
    def _run_worker_simulations(
        self,
        config: SimulationConfig,
        n_simulations: int
    ) -> Dict[str, np.ndarray]:
        """
        Executa simulações em um worker.
        
        Args:
            config: Configuração
            n_simulations: Número de simulações
            
        Returns:
            Resultados
        """
        simulated_returns = np.zeros(n_simulations)
        simulated_sharpes = np.zeros(n_simulations)
        simulated_drawdowns = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            sim_result = self._run_single_simulation(config)
            
            simulated_returns[i] = sim_result['total_return']
            simulated_sharpes[i] = sim_result['sharpe_ratio']
            simulated_drawdowns[i] = sim_result['max_drawdown']
        
        return {
            'returns': simulated_returns,
            'sharpes': simulated_sharpes,
            'drawdowns': simulated_drawdowns
        }
    
    def _run_single_simulation(
        self,
        config: SimulationConfig
    ) -> Dict[str, float]:
        """
        Executa uma única simulação.
        
        Args:
            config: Configuração
            
        Returns:
            Resultados da simulação
        """
        # Generate returns based on configuration
        if config.resample_trades and self.historical_trades:
            simulated_trades = self._resample_trades(config)
            daily_returns = self._trades_to_returns(simulated_trades, config)
        else:
            daily_returns = self._generate_returns(config)
        
        # Add randomization if configured
        if config.randomize_returns:
            daily_returns = self._add_noise(daily_returns)
        
        # Calculate metrics
        total_return = self._calculate_total_return(daily_returns)
        sharpe_ratio = self._calculate_sharpe(daily_returns)
        max_drawdown = self._calculate_max_drawdown(daily_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _resample_trades(
        self,
        config: SimulationConfig
    ) -> List[TradeRecord]:
        """
        Resample trades usando bootstrap.
        
        Args:
            config: Configuração
            
        Returns:
            Trades resampleados
        """
        n_trades = len(self.historical_trades)
        
        if n_trades == 0:
            return []
        
        # Bootstrap sampling with replacement
        indices = np.random.choice(n_trades, size=n_trades, replace=True)
        
        resampled_trades = [self.historical_trades[i] for i in indices]
        
        # Randomize order if configured
        if config.randomize_order:
            np.random.shuffle(resampled_trades)
        
        return resampled_trades
    
    def _trades_to_returns(
        self,
        trades: List[TradeRecord],
        config: SimulationConfig
    ) -> np.ndarray:
        """
        Converte trades para retornos diários.
        
        Args:
            trades: Lista de trades
            config: Configuração
            
        Returns:
            Array de retornos diários
        """
        daily_returns = np.zeros(config.simulation_days)
        
        if not trades:
            return daily_returns
        
        # Distribute trades across days
        trades_per_day = len(trades) / config.simulation_days
        
        for day in range(config.simulation_days):
            # Get trades for this day
            start_idx = int(day * trades_per_day)
            end_idx = int((day + 1) * trades_per_day)
            
            day_trades = trades[start_idx:end_idx]
            
            if day_trades:
                # Calculate daily return from trades
                day_return = sum(t.pnl_pct for t in day_trades) / len(day_trades)
                daily_returns[day] = day_return
        
        return daily_returns
    
    def _generate_returns(
        self,
        config: SimulationConfig
    ) -> np.ndarray:
        """
        Gera retornos baseado em distribuição.
        
        Args:
            config: Configuração
            
        Returns:
            Array de retornos
        """
        if config.return_distribution == "empirical" and self.historical_returns:
            # Sample from historical distribution
            returns = np.random.choice(
                self.historical_returns,
                size=config.simulation_days,
                replace=True
            )
            
        elif config.return_distribution == "normal":
            # Generate from normal distribution
            if self.historical_returns:
                mean = np.mean(self.historical_returns)
                std = np.std(self.historical_returns)
            else:
                mean = 0.0005  # 0.05% daily
                std = 0.01  # 1% daily volatility
            
            returns = np.random.normal(mean, std, config.simulation_days)
            
        elif config.return_distribution == "t-student":
            # Generate from t-distribution (fat tails)
            df = 3  # Degrees of freedom
            
            if self.historical_returns:
                mean = np.mean(self.historical_returns)
                std = np.std(self.historical_returns)
            else:
                mean = 0.0005
                std = 0.01
            
            returns = stats.t.rvs(df, loc=mean, scale=std, size=config.simulation_days)
            
        else:
            # Default to zeros
            returns = np.zeros(config.simulation_days)
        
        return returns
    
    def _add_noise(self, returns: np.ndarray) -> np.ndarray:
        """
        Adiciona ruído aos retornos.
        
        Args:
            returns: Array de retornos
            
        Returns:
            Retornos com ruído
        """
        noise_level = 0.1  # 10% of original volatility
        
        if len(returns) > 0:
            volatility = np.std(returns)
            noise = np.random.normal(0, volatility * noise_level, len(returns))
            returns = returns + noise
        
        return returns
    
    @jit(nopython=True)
    def _calculate_total_return(self, daily_returns: np.ndarray) -> float:
        """
        Calcula retorno total composto.
        
        Args:
            daily_returns: Retornos diários
            
        Returns:
            Retorno total
        """
        if len(daily_returns) == 0:
            return 0
        
        # Compound returns
        total_return = 1.0
        
        for ret in daily_returns:
            total_return *= (1 + ret)
        
        return total_return - 1
    
    @jit(nopython=True)
    def _calculate_sharpe(self, daily_returns: np.ndarray) -> float:
        """
        Calcula Sharpe ratio.
        
        Args:
            daily_returns: Retornos diários
            
        Returns:
            Sharpe ratio
        """
        if len(daily_returns) < 2:
            return 0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return > 0:
            # Annualized Sharpe
            sharpe = mean_return / std_return * np.sqrt(252)
        else:
            sharpe = 0
        
        return sharpe
    
    @jit(nopython=True)
    def _calculate_max_drawdown(self, daily_returns: np.ndarray) -> float:
        """
        Calcula drawdown máximo.
        
        Args:
            daily_returns: Retornos diários
            
        Returns:
            Drawdown máximo
        """
        if len(daily_returns) == 0:
            return 0
        
        # Calculate cumulative returns
        cum_returns = np.zeros(len(daily_returns) + 1)
        cum_returns[0] = 1.0
        
        for i, ret in enumerate(daily_returns):
            cum_returns[i + 1] = cum_returns[i] * (1 + ret)
        
        # Calculate running maximum
        running_max = cum_returns[0]
        max_dd = 0
        
        for value in cum_returns[1:]:
            if value > running_max:
                running_max = value
            
            drawdown = (running_max - value) / running_max if running_max > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_statistics(
        self,
        simulation_results: Dict[str, np.ndarray],
        config: SimulationConfig
    ) -> SimulationResult:
        """
        Calcula estatísticas das simulações.
        
        Args:
            simulation_results: Resultados das simulações
            config: Configuração
            
        Returns:
            Resultado estatístico
        """
        returns = simulation_results['returns']
        sharpes = simulation_results['sharpes']
        drawdowns = simulation_results['drawdowns']
        
        # Basic statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(returns, p)
        
        # Confidence intervals
        return_cis = {}
        sharpe_cis = {}
        drawdown_cis = {}
        
        for confidence in config.confidence_levels:
            alpha = 1 - confidence
            lower_p = (alpha / 2) * 100
            upper_p = (1 - alpha / 2) * 100
            
            return_cis[confidence] = ConfidenceInterval(
                confidence_level=confidence,
                lower_bound=np.percentile(returns, lower_p),
                upper_bound=np.percentile(returns, upper_p),
                median=np.median(returns),
                mean=np.mean(returns)
            )
            
            sharpe_cis[confidence] = ConfidenceInterval(
                confidence_level=confidence,
                lower_bound=np.percentile(sharpes, lower_p),
                upper_bound=np.percentile(sharpes, upper_p),
                median=np.median(sharpes),
                mean=np.mean(sharpes)
            )
            
            drawdown_cis[confidence] = ConfidenceInterval(
                confidence_level=confidence,
                lower_bound=np.percentile(drawdowns, lower_p),
                upper_bound=np.percentile(drawdowns, upper_p),
                median=np.median(drawdowns),
                mean=np.mean(drawdowns)
            )
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        probability_of_loss = np.mean(returns < 0)
        probability_of_ruin = np.mean(returns < -0.99)  # Lose 99% or more
        
        # Best/Worst scenarios
        best_case = np.percentile(returns, 95)
        worst_case = np.percentile(returns, 5)
        most_likely = median_return
        
        return SimulationResult(
            config=config,
            simulated_returns=returns,
            simulated_sharpe_ratios=sharpes,
            simulated_max_drawdowns=drawdowns,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            skewness=skewness,
            kurtosis=kurtosis,
            percentiles=percentiles,
            return_confidence_intervals=return_cis,
            sharpe_confidence_intervals=sharpe_cis,
            drawdown_confidence_intervals=drawdown_cis,
            var_95=var_95,
            cvar_95=cvar_95,
            probability_of_loss=probability_of_loss,
            probability_of_ruin=probability_of_ruin,
            best_case_return=best_case,
            worst_case_return=worst_case,
            most_likely_return=most_likely
        )