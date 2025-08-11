"""
Stress Testing - Testes de stress para cenários extremos
Black swan events, flash crashes e condições adversas de mercado
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
import pandas as pd
from numba import jit

from core.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult


logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Tipos de cenários de stress."""
    BLACK_SWAN = "black_swan"  # Evento raro e extremo
    FLASH_CRASH = "flash_crash"  # Queda súbita de preços
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Crise de liquidez
    VOLATILITY_SPIKE = "volatility_spike"  # Aumento súbito de volatilidade
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # Quebra de correlações
    FAT_TAIL = "fat_tail"  # Eventos de cauda gorda
    REGULATORY_SHOCK = "regulatory_shock"  # Mudanças regulatórias
    EXCHANGE_FAILURE = "exchange_failure"  # Falha de exchange
    NETWORK_CONGESTION = "network_congestion"  # Congestão de rede
    CUSTOM = "custom"  # Cenário customizado


@dataclass
class StressScenario:
    """Definição de um cenário de stress."""
    
    name: str
    scenario_type: ScenarioType
    description: str
    
    # Market shocks
    price_shock: float = 0  # Percentage price change
    volatility_multiplier: float = 1.0  # Volatility increase
    liquidity_reduction: float = 0  # Percentage liquidity reduction
    
    # Spread and slippage
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    
    # Correlations
    correlation_shift: float = 0  # Change in correlations
    
    # Duration
    duration_seconds: float = 3600  # 1 hour default
    recovery_time_seconds: float = 7200  # 2 hours to recover
    
    # Probability and severity
    probability: float = 0.01  # 1% probability
    severity_score: float = 0.5  # 0-1 severity scale
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Resultado de um teste de stress."""
    
    scenario: StressScenario
    
    # Performance metrics
    total_return: float = 0
    max_drawdown: float = 0
    var_95: float = 0
    cvar_95: float = 0
    
    # Survival metrics
    survived: bool = True
    min_equity: float = 0
    time_to_recovery: float = 0  # Seconds
    
    # Trade statistics during stress
    trades_during_stress: int = 0
    win_rate_during_stress: float = 0
    avg_loss_during_stress: float = 0
    
    # Risk metrics
    risk_score: float = 0  # 0-100
    resilience_score: float = 0  # 0-100
    
    # Detailed timeline
    equity_timeline: List[float] = field(default_factory=list)
    stress_periods: List[Tuple[float, float]] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Retorna resumo do resultado."""
        return {
            'scenario': self.scenario.name,
            'survived': self.survived,
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'min_equity': f"${self.min_equity:.2f}",
            'time_to_recovery': f"{self.time_to_recovery/3600:.1f}h",
            'resilience_score': f"{self.resilience_score:.1f}"
        }


class StressTester:
    """
    Sistema de stress testing para cenários extremos.
    """
    
    def __init__(self):
        """Inicializa stress tester."""
        self.backtest_engine = BacktestEngine()
        self.predefined_scenarios = self._create_predefined_scenarios()
        self.test_results = []
    
    def _create_predefined_scenarios(self) -> List[StressScenario]:
        """
        Cria cenários predefinidos de stress.
        
        Returns:
            Lista de cenários
        """
        scenarios = [
            # Black Swan Event
            StressScenario(
                name="Black Swan - 20% Crash",
                scenario_type=ScenarioType.BLACK_SWAN,
                description="Sudden 20% market crash like COVID-19",
                price_shock=-0.20,
                volatility_multiplier=5.0,
                liquidity_reduction=0.50,
                spread_multiplier=10.0,
                slippage_multiplier=5.0,
                duration_seconds=3600,
                recovery_time_seconds=86400,
                probability=0.001,
                severity_score=0.9
            ),
            
            # Flash Crash
            StressScenario(
                name="Flash Crash - 10% in 5 minutes",
                scenario_type=ScenarioType.FLASH_CRASH,
                description="Rapid 10% drop and recovery",
                price_shock=-0.10,
                volatility_multiplier=10.0,
                liquidity_reduction=0.80,
                spread_multiplier=20.0,
                slippage_multiplier=10.0,
                duration_seconds=300,
                recovery_time_seconds=1800,
                probability=0.01,
                severity_score=0.7
            ),
            
            # Liquidity Crisis
            StressScenario(
                name="Liquidity Crisis",
                scenario_type=ScenarioType.LIQUIDITY_CRISIS,
                description="Severe liquidity shortage",
                price_shock=-0.05,
                volatility_multiplier=3.0,
                liquidity_reduction=0.90,
                spread_multiplier=15.0,
                slippage_multiplier=8.0,
                duration_seconds=7200,
                recovery_time_seconds=28800,
                probability=0.02,
                severity_score=0.8
            ),
            
            # Volatility Spike
            StressScenario(
                name="Volatility Explosion",
                scenario_type=ScenarioType.VOLATILITY_SPIKE,
                description="Sudden 10x volatility increase",
                price_shock=0,
                volatility_multiplier=10.0,
                liquidity_reduction=0.30,
                spread_multiplier=5.0,
                slippage_multiplier=3.0,
                duration_seconds=3600,
                recovery_time_seconds=7200,
                probability=0.05,
                severity_score=0.6
            ),
            
            # Correlation Breakdown
            StressScenario(
                name="Correlation Breakdown",
                scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
                description="All correlations go to 1",
                price_shock=-0.08,
                volatility_multiplier=2.0,
                liquidity_reduction=0.20,
                correlation_shift=0.9,
                spread_multiplier=3.0,
                slippage_multiplier=2.0,
                duration_seconds=7200,
                recovery_time_seconds=14400,
                probability=0.03,
                severity_score=0.6
            ),
            
            # Exchange Failure
            StressScenario(
                name="Major Exchange Hack",
                scenario_type=ScenarioType.EXCHANGE_FAILURE,
                description="Major exchange gets hacked",
                price_shock=-0.15,
                volatility_multiplier=4.0,
                liquidity_reduction=0.70,
                spread_multiplier=8.0,
                slippage_multiplier=6.0,
                duration_seconds=86400,
                recovery_time_seconds=604800,  # 1 week
                probability=0.005,
                severity_score=0.85
            ),
            
            # Regulatory Shock
            StressScenario(
                name="Regulatory Ban",
                scenario_type=ScenarioType.REGULATORY_SHOCK,
                description="Major country bans crypto",
                price_shock=-0.30,
                volatility_multiplier=3.0,
                liquidity_reduction=0.60,
                spread_multiplier=5.0,
                slippage_multiplier=4.0,
                duration_seconds=86400,
                recovery_time_seconds=2592000,  # 30 days
                probability=0.002,
                severity_score=0.95
            )
        ]
        
        return scenarios
    
    async def run_stress_test(
        self,
        scenario: StressScenario,
        backtest_config: BacktestConfig,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> StressTestResult:
        """
        Executa teste de stress para um cenário.
        
        Args:
            scenario: Cenário de stress
            backtest_config: Configuração do backtest
            historical_data: Dados históricos
            
        Returns:
            Resultado do teste
        """
        start_time = time.perf_counter()
        
        logger.info(
            f"Running stress test",
            scenario=scenario.name,
            type=scenario.scenario_type.value,
            shock=f"{scenario.price_shock:.1%}"
        )
        
        # Modify historical data to include stress
        stressed_data = self._apply_stress_to_data(
            historical_data, scenario, backtest_config
        )
        
        # Run backtest with stressed data
        backtest_result = await self.backtest_engine.run_backtest(
            backtest_config, stressed_data
        )
        
        # Analyze results
        result = self._analyze_stress_impact(
            backtest_result, scenario, backtest_config
        )
        
        execution_time = time.perf_counter() - start_time
        logger.info(
            f"Stress test completed",
            execution_time=f"{execution_time:.2f}s",
            survived=result.survived,
            max_drawdown=f"{result.max_drawdown:.2%}",
            resilience=f"{result.resilience_score:.1f}"
        )
        
        return result
    
    async def run_all_scenarios(
        self,
        backtest_config: BacktestConfig,
        historical_data: Optional[Dict[str, Any]] = None,
        custom_scenarios: Optional[List[StressScenario]] = None
    ) -> List[StressTestResult]:
        """
        Executa todos os cenários de stress.
        
        Args:
            backtest_config: Configuração do backtest
            historical_data: Dados históricos
            custom_scenarios: Cenários customizados opcionais
            
        Returns:
            Lista de resultados
        """
        scenarios = self.predefined_scenarios.copy()
        
        if custom_scenarios:
            scenarios.extend(custom_scenarios)
        
        results = []
        
        for scenario in scenarios:
            result = await self.run_stress_test(
                scenario, backtest_config, historical_data
            )
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def _apply_stress_to_data(
        self,
        historical_data: Optional[Dict[str, Any]],
        scenario: StressScenario,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """
        Aplica stress aos dados históricos.
        
        Args:
            historical_data: Dados originais
            scenario: Cenário de stress
            config: Configuração
            
        Returns:
            Dados com stress aplicado
        """
        if not historical_data:
            # Generate synthetic stressed data
            return self._generate_stressed_data(scenario, config)
        
        # Copy data
        stressed_data = {}
        
        for symbol, data in historical_data.items():
            stressed_data[symbol] = {
                'timestamps': data['timestamps'].copy(),
                'prices': data['prices'].copy(),
                'volumes': data['volumes'].copy()
            }
        
        # Find stress period (random point in time)
        total_timestamps = len(stressed_data[list(stressed_data.keys())[0]]['timestamps'])
        stress_start_idx = np.random.randint(
            int(total_timestamps * 0.2),
            int(total_timestamps * 0.8)
        )
        
        # Calculate stress duration in ticks
        tick_interval = 1  # seconds
        stress_ticks = int(scenario.duration_seconds / tick_interval)
        recovery_ticks = int(scenario.recovery_time_seconds / tick_interval)
        
        stress_end_idx = min(stress_start_idx + stress_ticks, total_timestamps)
        recovery_end_idx = min(stress_end_idx + recovery_ticks, total_timestamps)
        
        # Apply stress to each symbol
        for symbol in stressed_data:
            prices = stressed_data[symbol]['prices']
            volumes = stressed_data[symbol]['volumes']
            
            # Apply price shock
            if scenario.price_shock != 0:
                for i in range(stress_start_idx, stress_end_idx):
                    # Gradual shock application
                    progress = (i - stress_start_idx) / max(stress_ticks, 1)
                    shock = scenario.price_shock * progress
                    prices[i] *= (1 + shock)
            
            # Apply volatility increase
            if scenario.volatility_multiplier > 1:
                for i in range(stress_start_idx, stress_end_idx):
                    if i > 0:
                        # Add random volatility
                        vol = abs(prices[i] - prices[i-1]) / prices[i-1]
                        extra_vol = vol * (scenario.volatility_multiplier - 1)
                        prices[i] += prices[i] * np.random.normal(0, extra_vol)
            
            # Apply liquidity reduction
            if scenario.liquidity_reduction > 0:
                for i in range(stress_start_idx, stress_end_idx):
                    volumes[i] *= (1 - scenario.liquidity_reduction)
            
            # Recovery period
            if recovery_ticks > 0:
                pre_stress_price = prices[stress_start_idx - 1] if stress_start_idx > 0 else prices[0]
                
                for i in range(stress_end_idx, recovery_end_idx):
                    # Gradual recovery
                    progress = (i - stress_end_idx) / recovery_ticks
                    current_price = prices[i]
                    target_price = pre_stress_price
                    prices[i] = current_price + (target_price - current_price) * progress * 0.5
        
        return stressed_data
    
    def _generate_stressed_data(
        self,
        scenario: StressScenario,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """
        Gera dados sintéticos com stress.
        
        Args:
            scenario: Cenário
            config: Configuração
            
        Returns:
            Dados estressados
        """
        # Generate base data
        n_points = int((config.end_date - config.start_date).total_seconds())
        timestamps = np.arange(n_points) + config.start_date.timestamp()
        
        stressed_data = {}
        
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            # Base price with stress
            base_price = 1000 if 'BTC' in symbol else 100
            prices = np.ones(n_points) * base_price
            
            # Add random walk
            returns = np.random.normal(0, 0.01, n_points)
            
            # Add stress period
            stress_start = n_points // 2
            stress_end = stress_start + int(scenario.duration_seconds)
            
            # Apply shock
            returns[stress_start:stress_end] += scenario.price_shock / (stress_end - stress_start)
            
            # Apply volatility
            returns[stress_start:stress_end] *= scenario.volatility_multiplier
            
            # Calculate prices
            for i in range(1, n_points):
                prices[i] = prices[i-1] * (1 + returns[i])
            
            # Volumes
            volumes = np.random.uniform(100, 1000, n_points)
            volumes[stress_start:stress_end] *= (1 - scenario.liquidity_reduction)
            
            stressed_data[symbol] = {
                'timestamps': timestamps,
                'prices': prices,
                'volumes': volumes
            }
        
        return stressed_data
    
    def _analyze_stress_impact(
        self,
        backtest_result: BacktestResult,
        scenario: StressScenario,
        config: BacktestConfig
    ) -> StressTestResult:
        """
        Analisa impacto do stress.
        
        Args:
            backtest_result: Resultado do backtest
            scenario: Cenário
            config: Configuração
            
        Returns:
            Resultado da análise
        """
        result = StressTestResult(scenario=scenario)
        
        # Basic metrics
        result.total_return = backtest_result.total_return
        result.max_drawdown = backtest_result.max_drawdown
        result.var_95 = backtest_result.var_95
        
        # Survival check
        if backtest_result.equity_curve:
            min_equity = min(backtest_result.equity_curve)
            result.min_equity = min_equity
            result.survived = min_equity > config.initial_capital * 0.1  # Lost less than 90%
        
        # Calculate CVaR
        if backtest_result.daily_returns:
            returns = np.array(backtest_result.daily_returns)
            var_threshold = np.percentile(returns, 5)
            result.cvar_95 = np.mean(returns[returns <= var_threshold])
        
        # Time to recovery
        if backtest_result.equity_curve:
            result.time_to_recovery = self._calculate_recovery_time(
                backtest_result.equity_curve
            )
        
        # Trades during stress (simplified - assume middle period is stress)
        if backtest_result.trades:
            total_trades = len(backtest_result.trades)
            stress_trades = backtest_result.trades[
                total_trades//3 : 2*total_trades//3
            ]
            
            result.trades_during_stress = len(stress_trades)
            
            if stress_trades:
                winners = [t for t in stress_trades if t.is_winner]
                result.win_rate_during_stress = len(winners) / len(stress_trades)
                
                losses = [t.net_pnl for t in stress_trades if not t.is_winner]
                if losses:
                    result.avg_loss_during_stress = abs(np.mean(losses))
        
        # Calculate scores
        result.risk_score = self._calculate_risk_score(result)
        result.resilience_score = self._calculate_resilience_score(result)
        
        # Store timeline
        result.equity_timeline = backtest_result.equity_curve
        
        return result
    
    def _calculate_recovery_time(self, equity_curve: List[float]) -> float:
        """
        Calcula tempo de recuperação após drawdown.
        
        Args:
            equity_curve: Curva de equity
            
        Returns:
            Tempo em segundos
        """
        if len(equity_curve) < 3:
            return 0
        
        # Find maximum drawdown point
        peak = equity_curve[0]
        max_dd_idx = 0
        max_dd = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_idx = i
        
        # Find recovery point
        recovery_idx = len(equity_curve) - 1
        
        for i in range(max_dd_idx + 1, len(equity_curve)):
            if equity_curve[i] >= peak * 0.95:  # 95% recovery
                recovery_idx = i
                break
        
        # Convert to seconds (assuming 1 second per tick)
        return float(recovery_idx - max_dd_idx)
    
    def _calculate_risk_score(self, result: StressTestResult) -> float:
        """
        Calcula score de risco (0-100).
        
        Args:
            result: Resultado do teste
            
        Returns:
            Score de risco
        """
        score = 0
        
        # Drawdown contribution (40 points)
        score += min(result.max_drawdown * 100, 40)
        
        # VaR contribution (20 points)
        score += min(abs(result.var_95) * 500, 20)
        
        # Loss during stress (20 points)
        if result.avg_loss_during_stress > 0:
            score += min(result.avg_loss_during_stress * 1000, 20)
        
        # Survival (20 points)
        if not result.survived:
            score += 20
        
        return min(score, 100)
    
    def _calculate_resilience_score(self, result: StressTestResult) -> float:
        """
        Calcula score de resiliência (0-100).
        
        Args:
            result: Resultado do teste
            
        Returns:
            Score de resiliência
        """
        score = 0
        
        # Survival (40 points)
        if result.survived:
            score += 40
        
        # Quick recovery (20 points)
        if result.time_to_recovery > 0:
            # Less than 1 hour = full points
            if result.time_to_recovery < 3600:
                score += 20
            elif result.time_to_recovery < 7200:
                score += 10
            elif result.time_to_recovery < 86400:
                score += 5
        
        # Limited drawdown (20 points)
        if result.max_drawdown < 0.1:
            score += 20
        elif result.max_drawdown < 0.2:
            score += 10
        elif result.max_drawdown < 0.3:
            score += 5
        
        # Maintained win rate (20 points)
        if result.win_rate_during_stress > 0.5:
            score += 20
        elif result.win_rate_during_stress > 0.3:
            score += 10
        
        return min(score, 100)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo de stress testing.
        
        Returns:
            Relatório consolidado
        """
        if not self.test_results:
            return {'status': 'No tests run'}
        
        # Aggregate results
        survived_count = sum(1 for r in self.test_results if r.survived)
        avg_drawdown = np.mean([r.max_drawdown for r in self.test_results])
        avg_resilience = np.mean([r.resilience_score for r in self.test_results])
        
        # Worst case scenario
        worst_scenario = min(self.test_results, key=lambda r: r.min_equity)
        
        # Best resilience
        best_resilience = max(self.test_results, key=lambda r: r.resilience_score)
        
        return {
            'summary': {
                'total_scenarios': len(self.test_results),
                'survived': survived_count,
                'survival_rate': f"{survived_count/len(self.test_results):.1%}",
                'avg_max_drawdown': f"{avg_drawdown:.2%}",
                'avg_resilience_score': f"{avg_resilience:.1f}"
            },
            'worst_case': {
                'scenario': worst_scenario.scenario.name,
                'min_equity': f"${worst_scenario.min_equity:.2f}",
                'max_drawdown': f"{worst_scenario.max_drawdown:.2%}",
                'survived': worst_scenario.survived
            },
            'best_resilience': {
                'scenario': best_resilience.scenario.name,
                'resilience_score': f"{best_resilience.resilience_score:.1f}",
                'recovery_time': f"{best_resilience.time_to_recovery/3600:.1f}h"
            },
            'scenario_results': [
                {
                    'name': r.scenario.name,
                    'type': r.scenario.scenario_type.value,
                    'survived': r.survived,
                    'max_drawdown': f"{r.max_drawdown:.2%}",
                    'resilience': f"{r.resilience_score:.1f}"
                }
                for r in self.test_results
            ]
        }