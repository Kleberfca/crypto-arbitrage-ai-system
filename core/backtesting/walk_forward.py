"""
Walk-Forward Optimizer - Otimização walk-forward para validação robusta
Evita overfitting através de otimização e validação em janelas deslizantes
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from core.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult


logger = logging.getLogger(__name__)


@dataclass
class OptimizationWindow:
    """Janela de otimização/validação."""
    
    # Time periods
    optimization_start: datetime
    optimization_end: datetime
    validation_start: datetime
    validation_end: datetime
    
    # Results
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_score: float = 0
    validation_score: float = 0
    
    # Metrics
    optimization_sharpe: float = 0
    validation_sharpe: float = 0
    optimization_return: float = 0
    validation_return: float = 0
    
    @property
    def efficiency_ratio(self) -> float:
        """Ratio entre validation e optimization score."""
        if self.optimization_score > 0:
            return self.validation_score / self.optimization_score
        return 0


@dataclass
class WalkForwardConfig:
    """Configuração do walk-forward."""
    
    # Time periods
    total_start: datetime
    total_end: datetime
    
    # Window configuration
    optimization_period_days: int = 180  # 6 months
    validation_period_days: int = 30  # 1 month
    step_days: int = 30  # Step forward
    
    # Optimization
    optimization_metric: str = "sharpe_ratio"
    min_trades_per_window: int = 30
    
    # Parameters to optimize
    parameters_to_optimize: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'max_position_size': (0.01, 0.05),
            'max_daily_loss': (0.01, 0.05),
            'slippage_bps': (1, 10),
            'confidence_threshold': (0.5, 0.9)
        }
    )
    
    # Strategies
    enabled_strategies: List[str] = field(
        default_factory=lambda: ["spatial", "statistical", "triangular"]
    )


@dataclass
class WalkForwardResult:
    """Resultado do walk-forward optimization."""
    
    config: WalkForwardConfig
    
    # Windows
    windows: List[OptimizationWindow] = field(default_factory=list)
    
    # Overall metrics
    avg_efficiency_ratio: float = 0
    total_return: float = 0
    annualized_return: float = 0
    
    # Stability metrics
    parameter_stability: Dict[str, float] = field(default_factory=dict)
    performance_consistency: float = 0
    
    # Out-of-sample performance
    oos_sharpe_ratio: float = 0
    oos_return: float = 0
    oos_max_drawdown: float = 0
    
    # Robustness score (0-100)
    robustness_score: float = 0
    
    def summary(self) -> Dict[str, Any]:
        """Retorna resumo dos resultados."""
        return {
            'windows': len(self.windows),
            'avg_efficiency': f"{self.avg_efficiency_ratio:.2f}",
            'total_return': f"{self.total_return:.2%}",
            'oos_sharpe': f"{self.oos_sharpe_ratio:.2f}",
            'robustness_score': f"{self.robustness_score:.1f}",
            'performance_consistency': f"{self.performance_consistency:.2f}"
        }


class WalkForwardOptimizer:
    """
    Otimizador walk-forward para validação robusta de estratégias.
    """
    
    def __init__(self):
        """Inicializa otimizador walk-forward."""
        self.backtest_engine = BacktestEngine()
        self.current_window = None
        self.optimization_cache = {}
        self.parameter_history = []
    
    async def run_walk_forward(
        self,
        config: WalkForwardConfig,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> WalkForwardResult:
        """
        Executa otimização walk-forward completa.
        
        Args:
            config: Configuração do walk-forward
            historical_data: Dados históricos
            
        Returns:
            Resultado do walk-forward
        """
        start_time = time.perf_counter()
        
        logger.info(
            f"Starting walk-forward optimization",
            start_date=config.total_start.isoformat(),
            end_date=config.total_end.isoformat(),
            windows=self._calculate_num_windows(config)
        )
        
        # Create windows
        windows = self._create_windows(config)
        
        # Run optimization for each window
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Optimize on in-sample period
            optimal_params = await self._optimize_window(
                window, config, historical_data
            )
            window.optimal_parameters = optimal_params
            
            # Validate on out-of-sample period
            validation_result = await self._validate_window(
                window, optimal_params, config, historical_data
            )
            
            # Update window metrics
            self._update_window_metrics(window, validation_result)
            
            # Store parameter history
            self.parameter_history.append(optimal_params)
        
        # Calculate overall results
        result = self._calculate_overall_results(windows, config)
        
        # Run final out-of-sample test
        oos_result = await self._run_final_oos_test(result, config, historical_data)
        self._update_oos_metrics(result, oos_result)
        
        # Calculate robustness score
        result.robustness_score = self._calculate_robustness_score(result)
        
        execution_time = time.perf_counter() - start_time
        logger.info(
            f"Walk-forward completed",
            execution_time=f"{execution_time:.2f}s",
            robustness=f"{result.robustness_score:.1f}",
            oos_sharpe=f"{result.oos_sharpe_ratio:.2f}"
        )
        
        return result
    
    def _calculate_num_windows(self, config: WalkForwardConfig) -> int:
        """Calcula número de janelas."""
        total_days = (config.total_end - config.total_start).days
        window_size = config.optimization_period_days + config.validation_period_days
        
        return max(1, (total_days - window_size) // config.step_days + 1)
    
    def _create_windows(self, config: WalkForwardConfig) -> List[OptimizationWindow]:
        """
        Cria janelas de otimização/validação.
        
        Args:
            config: Configuração
            
        Returns:
            Lista de janelas
        """
        windows = []
        current_start = config.total_start
        
        while True:
            # Optimization period
            opt_start = current_start
            opt_end = opt_start + timedelta(days=config.optimization_period_days)
            
            # Validation period
            val_start = opt_end
            val_end = val_start + timedelta(days=config.validation_period_days)
            
            # Check if we have enough data
            if val_end > config.total_end:
                break
            
            window = OptimizationWindow(
                optimization_start=opt_start,
                optimization_end=opt_end,
                validation_start=val_start,
                validation_end=val_end
            )
            
            windows.append(window)
            
            # Step forward
            current_start += timedelta(days=config.step_days)
        
        return windows
    
    async def _optimize_window(
        self,
        window: OptimizationWindow,
        config: WalkForwardConfig,
        historical_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Otimiza parâmetros para uma janela.
        
        Args:
            window: Janela de otimização
            config: Configuração
            historical_data: Dados históricos
            
        Returns:
            Parâmetros ótimos
        """
        # Define objective function
        def objective(params):
            # Unpack parameters
            param_dict = self._params_array_to_dict(params, config.parameters_to_optimize)
            
            # Run backtest with these parameters
            backtest_config = BacktestConfig(
                start_date=window.optimization_start,
                end_date=window.optimization_end,
                initial_capital=10000,
                max_position_size=param_dict['max_position_size'],
                max_daily_loss=param_dict['max_daily_loss'],
                slippage_bps=param_dict['slippage_bps']
            )
            
            # Run backtest (synchronously for optimization)
            result = asyncio.run(
                self.backtest_engine.run_backtest(backtest_config, historical_data)
            )
            
            # Return negative score (minimization)
            score = self._calculate_optimization_score(result, config.optimization_metric)
            return -score
        
        # Set bounds
        bounds = list(config.parameters_to_optimize.values())
        
        # Run differential evolution optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )
        
        # Convert back to dictionary
        optimal_params = self._params_array_to_dict(
            result.x, config.parameters_to_optimize
        )
        
        # Store optimization score
        window.optimization_score = -result.fun
        
        return optimal_params
    
    async def _validate_window(
        self,
        window: OptimizationWindow,
        params: Dict[str, Any],
        config: WalkForwardConfig,
        historical_data: Optional[Dict[str, Any]]
    ) -> BacktestResult:
        """
        Valida parâmetros no período out-of-sample.
        
        Args:
            window: Janela
            params: Parâmetros otimizados
            config: Configuração
            historical_data: Dados históricos
            
        Returns:
            Resultado da validação
        """
        backtest_config = BacktestConfig(
            start_date=window.validation_start,
            end_date=window.validation_end,
            initial_capital=10000,
            max_position_size=params.get('max_position_size', 0.02),
            max_daily_loss=params.get('max_daily_loss', 0.03),
            slippage_bps=params.get('slippage_bps', 5)
        )
        
        result = await self.backtest_engine.run_backtest(
            backtest_config, historical_data
        )
        
        # Store validation score
        window.validation_score = self._calculate_optimization_score(
            result, config.optimization_metric
        )
        
        return result
    
    def _update_window_metrics(
        self,
        window: OptimizationWindow,
        validation_result: BacktestResult
    ) -> None:
        """
        Atualiza métricas da janela.
        
        Args:
            window: Janela
            validation_result: Resultado da validação
        """
        window.validation_sharpe = validation_result.sharpe_ratio
        window.validation_return = validation_result.total_return
        
        # Efficiency ratio já é calculado como property
    
    def _calculate_optimization_score(
        self,
        result: BacktestResult,
        metric: str
    ) -> float:
        """
        Calcula score de otimização.
        
        Args:
            result: Resultado do backtest
            metric: Métrica a otimizar
            
        Returns:
            Score
        """
        if metric == "sharpe_ratio":
            return result.sharpe_ratio
        elif metric == "total_return":
            return result.total_return
        elif metric == "calmar_ratio":
            return result.calmar_ratio
        elif metric == "profit_factor":
            return result.profit_factor
        else:
            # Default to Sharpe
            return result.sharpe_ratio
    
    def _params_array_to_dict(
        self,
        params: np.ndarray,
        param_config: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Converte array de parâmetros para dicionário.
        
        Args:
            params: Array de parâmetros
            param_config: Configuração dos parâmetros
            
        Returns:
            Dicionário de parâmetros
        """
        param_dict = {}
        
        for i, (name, bounds) in enumerate(param_config.items()):
            if i < len(params):
                param_dict[name] = params[i]
        
        return param_dict
    
    def _calculate_overall_results(
        self,
        windows: List[OptimizationWindow],
        config: WalkForwardConfig
    ) -> WalkForwardResult:
        """
        Calcula resultados gerais.
        
        Args:
            windows: Lista de janelas
            config: Configuração
            
        Returns:
            Resultado geral
        """
        result = WalkForwardResult(config=config, windows=windows)
        
        if not windows:
            return result
        
        # Calculate average efficiency ratio
        efficiency_ratios = [w.efficiency_ratio for w in windows]
        result.avg_efficiency_ratio = np.mean(efficiency_ratios)
        
        # Calculate total return (compound validation returns)
        total_return = 1.0
        for window in windows:
            total_return *= (1 + window.validation_return)
        result.total_return = total_return - 1
        
        # Annualized return
        total_days = (config.total_end - config.total_start).days
        if total_days > 0:
            result.annualized_return = (1 + result.total_return) ** (365 / total_days) - 1
        
        # Calculate parameter stability
        result.parameter_stability = self._calculate_parameter_stability()
        
        # Calculate performance consistency
        validation_sharpes = [w.validation_sharpe for w in windows]
        if len(validation_sharpes) > 1:
            result.performance_consistency = 1 - (np.std(validation_sharpes) / (np.mean(validation_sharpes) + 1e-6))
        
        return result
    
    def _calculate_parameter_stability(self) -> Dict[str, float]:
        """
        Calcula estabilidade dos parâmetros.
        
        Returns:
            Estabilidade por parâmetro
        """
        stability = {}
        
        if len(self.parameter_history) < 2:
            return stability
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.parameter_history)
        
        for column in df.columns:
            # Calculate coefficient of variation
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            if mean_val > 0:
                cv = std_val / mean_val
                stability[column] = 1 - min(cv, 1)  # Convert to stability score
            else:
                stability[column] = 0
        
        return stability
    
    async def _run_final_oos_test(
        self,
        result: WalkForwardResult,
        config: WalkForwardConfig,
        historical_data: Optional[Dict[str, Any]]
    ) -> BacktestResult:
        """
        Executa teste final out-of-sample.
        
        Args:
            result: Resultado do walk-forward
            config: Configuração
            historical_data: Dados históricos
            
        Returns:
            Resultado do teste OOS
        """
        # Use average of optimal parameters from all windows
        if not result.windows:
            return None
        
        avg_params = {}
        for window in result.windows:
            for param, value in window.optimal_parameters.items():
                if param not in avg_params:
                    avg_params[param] = []
                avg_params[param].append(value)
        
        # Calculate averages
        for param in avg_params:
            avg_params[param] = np.mean(avg_params[param])
        
        # Run backtest on entire validation periods
        all_validation_start = result.windows[0].validation_start
        all_validation_end = result.windows[-1].validation_end
        
        backtest_config = BacktestConfig(
            start_date=all_validation_start,
            end_date=all_validation_end,
            initial_capital=10000,
            max_position_size=avg_params.get('max_position_size', 0.02),
            max_daily_loss=avg_params.get('max_daily_loss', 0.03),
            slippage_bps=avg_params.get('slippage_bps', 5)
        )
        
        return await self.backtest_engine.run_backtest(
            backtest_config, historical_data
        )
    
    def _update_oos_metrics(
        self,
        result: WalkForwardResult,
        oos_result: BacktestResult
    ) -> None:
        """
        Atualiza métricas out-of-sample.
        
        Args:
            result: Resultado do walk-forward
            oos_result: Resultado OOS
        """
        if oos_result:
            result.oos_sharpe_ratio = oos_result.sharpe_ratio
            result.oos_return = oos_result.total_return
            result.oos_max_drawdown = oos_result.max_drawdown
    
    def _calculate_robustness_score(
        self,
        result: WalkForwardResult
    ) -> float:
        """
        Calcula score de robustez (0-100).
        
        Args:
            result: Resultado do walk-forward
            
        Returns:
            Score de robustez
        """
        score = 0
        
        # Efficiency ratio contribution (30 points)
        if result.avg_efficiency_ratio > 0.8:
            score += 30
        elif result.avg_efficiency_ratio > 0.6:
            score += 20
        elif result.avg_efficiency_ratio > 0.4:
            score += 10
        
        # Performance consistency contribution (25 points)
        score += result.performance_consistency * 25
        
        # Parameter stability contribution (20 points)
        if result.parameter_stability:
            avg_stability = np.mean(list(result.parameter_stability.values()))
            score += avg_stability * 20
        
        # OOS Sharpe contribution (25 points)
        if result.oos_sharpe_ratio > 2.5:
            score += 25
        elif result.oos_sharpe_ratio > 1.5:
            score += 15
        elif result.oos_sharpe_ratio > 0.5:
            score += 5
        
        return min(score, 100)