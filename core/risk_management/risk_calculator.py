"""
Risk Calculator - Cálculo de métricas de risco
VaR, CVaR, Sharpe, Sortino e outras métricas
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats
from scipy.stats import norm, t
import redis.asyncio as redis

from config.settings import settings


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Níveis de risco."""
    MINIMAL = "minimal"  # < 1% VaR
    LOW = "low"  # 1-2% VaR
    MODERATE = "moderate"  # 2-3% VaR
    HIGH = "high"  # 3-5% VaR
    EXTREME = "extreme"  # > 5% VaR


class VaRMethod(Enum):
    """Métodos de cálculo de VaR."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"  # Adjusted for skewness/kurtosis


@dataclass
class RiskMetrics:
    """Métricas de risco consolidadas."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Value at Risk
    var_95: float = 0  # 95% confidence
    var_99: float = 0  # 99% confidence
    cvar_95: float = 0  # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0
    
    # Performance metrics
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    information_ratio: float = 0
    
    # Drawdown metrics
    current_drawdown: float = 0
    max_drawdown: float = 0
    avg_drawdown: float = 0
    drawdown_duration: int = 0  # Days
    
    # Volatility metrics
    daily_volatility: float = 0
    annualized_volatility: float = 0
    downside_volatility: float = 0
    upside_volatility: float = 0
    
    # Risk-adjusted returns
    risk_adjusted_return: float = 0
    treynor_ratio: float = 0
    jensen_alpha: float = 0
    
    # Greeks (for options-like analysis)
    delta: float = 0  # Price sensitivity
    gamma: float = 0  # Delta sensitivity
    vega: float = 0  # Volatility sensitivity
    
    # Tail risk
    skewness: float = 0
    kurtosis: float = 0
    tail_ratio: float = 0  # 95th percentile / 5th percentile
    
    # Overall risk score (0-100)
    risk_score: float = 0
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'timestamp': self.timestamp,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.annualized_volatility,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value
        }


class RiskCalculator:
    """
    Calculadora principal de métricas de risco.
    Implementa VaR, CVaR, Sharpe, Sortino e análise completa.
    """
    
    def __init__(self):
        """Inicializa calculadora de risco."""
        # Configuration
        self.lookback_period = 100  # Days for historical data
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = 0.02  # 2% annual
        self.target_return = 0.10  # 10% annual target
        
        # Data storage
        self.returns_history = deque(maxlen=1000)
        self.portfolio_values = deque(maxlen=1000)
        self.position_history = []
        
        # Current metrics
        self.current_metrics = RiskMetrics()
        self.metrics_history = deque(maxlen=100)
        
        # VaR calculation method
        self.var_method = VaRMethod.CORNISH_FISHER
        
        # Monte Carlo parameters
        self.mc_simulations = 10000
        self.mc_horizon = 1  # Day
        
        # Cache
        self.metrics_cache = {}
        self.cache_ttl = 60  # seconds
        self.last_calculation = 0
        
        # Redis for persistence
        self.redis_client = None
        self._init_redis()
        
        # Start monitoring
        asyncio.create_task(self._monitor_loop())
    
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
    
    async def calculate_metrics(
        self,
        portfolio_value: float,
        positions: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """
        Calcula todas as métricas de risco.
        
        Args:
            portfolio_value: Valor atual do portfólio
            positions: Posições atuais
            market_data: Dados de mercado opcionais
            
        Returns:
            Métricas calculadas
        """
        start_time = time.perf_counter()
        
        # Update history
        self.portfolio_values.append(portfolio_value)
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                return_pct = (portfolio_value - prev_value) / prev_value
                self.returns_history.append(return_pct)
        
        # Check cache
        if time.time() - self.last_calculation < self.cache_ttl:
            return self.current_metrics
        
        # Calculate returns array
        returns = np.array(list(self.returns_history))
        
        if len(returns) < 20:  # Minimum data points
            logger.debug("Insufficient data for risk calculation")
            return self.current_metrics
        
        # Calculate VaR and CVaR
        var_95, cvar_95 = await self._calculate_var(returns, 0.95)
        var_99, cvar_99 = await self._calculate_var(returns, 0.99)
        
        # Calculate performance metrics
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        calmar = self._calculate_calmar_ratio(returns)
        
        # Calculate drawdown metrics
        dd_current, dd_max, dd_avg, dd_duration = self._calculate_drawdowns()
        
        # Calculate volatility metrics
        vol_daily = np.std(returns) if len(returns) > 0 else 0
        vol_annual = vol_daily * np.sqrt(252)  # Annualized
        vol_downside = self._calculate_downside_volatility(returns)
        vol_upside = self._calculate_upside_volatility(returns)
        
        # Calculate tail risk metrics
        skew = stats.skew(returns) if len(returns) > 2 else 0
        kurt = stats.kurtosis(returns) if len(returns) > 3 else 0
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # Calculate Greeks (simplified)
        delta, gamma, vega = await self._calculate_greeks(positions, market_data)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            var_95, sharpe, dd_max, vol_annual, skew, kurt
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(var_95, risk_score)
        
        # Update metrics
        self.current_metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            current_drawdown=dd_current,
            max_drawdown=dd_max,
            avg_drawdown=dd_avg,
            drawdown_duration=dd_duration,
            daily_volatility=vol_daily,
            annualized_volatility=vol_annual,
            downside_volatility=vol_downside,
            upside_volatility=vol_upside,
            skewness=skew,
            kurtosis=kurt,
            tail_ratio=tail_ratio,
            delta=delta,
            gamma=gamma,
            vega=vega,
            risk_score=risk_score,
            risk_level=risk_level
        )
        
        # Update cache
        self.last_calculation = time.time()
        self.metrics_history.append(self.current_metrics)
        
        # Log calculation time
        calc_time = (time.perf_counter() - start_time) * 1000
        if calc_time > 10:
            logger.warning(f"Slow risk calculation: {calc_time:.2f}ms")
        
        return self.current_metrics
    
    async def _calculate_var(
        self,
        returns: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Calcula VaR e CVaR.
        
        Args:
            returns: Array de retornos
            confidence: Nível de confiança
            
        Returns:
            (VaR, CVaR)
        """
        if len(returns) == 0:
            return 0, 0
        
        if self.var_method == VaRMethod.HISTORICAL:
            var = self._historical_var(returns, confidence)
            cvar = self._historical_cvar(returns, confidence)
            
        elif self.var_method == VaRMethod.PARAMETRIC:
            var = self._parametric_var(returns, confidence)
            cvar = self._parametric_cvar(returns, confidence)
            
        elif self.var_method == VaRMethod.MONTE_CARLO:
            var, cvar = await self._monte_carlo_var(returns, confidence)
            
        else:  # CORNISH_FISHER
            var = self._cornish_fisher_var(returns, confidence)
            cvar = self._historical_cvar(returns, confidence)
        
        return abs(var), abs(cvar)
    
    def _historical_var(self, returns: np.ndarray, confidence: float) -> float:
        """VaR histórico."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _historical_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """CVaR histórico (Expected Shortfall)."""
        var = self._historical_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def _parametric_var(self, returns: np.ndarray, confidence: float) -> float:
        """VaR paramétrico (assume distribuição normal)."""
        mean = np.mean(returns)
        std = np.std(returns)
        return mean + std * norm.ppf(1 - confidence)
    
    def _parametric_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """CVaR paramétrico."""
        mean = np.mean(returns)
        std = np.std(returns)
        alpha = 1 - confidence
        return mean - std * norm.pdf(norm.ppf(alpha)) / alpha
    
    def _cornish_fisher_var(self, returns: np.ndarray, confidence: float) -> float:
        """VaR Cornish-Fisher (ajustado para skewness e kurtosis)."""
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, excess=True)
        
        z = norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
        
        return mean + std * cf_z
    
    async def _monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """VaR Monte Carlo."""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Run simulations
        simulated_returns = np.random.normal(mean, std, self.mc_simulations)
        
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        cvar = np.mean(simulated_returns[simulated_returns <= var])
        
        return var, cvar
    
    @jit(nopython=True)
    def _calculate_sharpe_ratio_fast(
        self,
        returns: np.ndarray,
        risk_free_rate: float
    ) -> float:
        """
        Calcula Sharpe Ratio otimizado com Numba.
        
        Args:
            returns: Retornos
            risk_free_rate: Taxa livre de risco
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 0:
            return mean_excess / std_excess * np.sqrt(252)  # Annualized
        
        return 0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calcula Sharpe Ratio."""
        return self._calculate_sharpe_ratio_fast(returns, self.risk_free_rate)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calcula Sortino Ratio (usa apenas downside volatility)."""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                return np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        return 0
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calcula Calmar Ratio (return / max drawdown)."""
        if len(returns) == 0:
            return 0
        
        annual_return = np.mean(returns) * 252
        _, max_dd, _, _ = self._calculate_drawdowns()
        
        if max_dd > 0:
            return annual_return / max_dd
        
        return 0
    
    def _calculate_drawdowns(self) -> Tuple[float, float, float, int]:
        """
        Calcula métricas de drawdown.
        
        Returns:
            (current_dd, max_dd, avg_dd, duration)
        """
        if len(self.portfolio_values) < 2:
            return 0, 0, 0, 0
        
        values = np.array(list(self.portfolio_values))
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        current_dd = abs(drawdowns[-1])
        max_dd = abs(np.min(drawdowns))
        avg_dd = np.mean(np.abs(drawdowns[drawdowns < 0]))
        
        # Calculate duration
        in_drawdown = drawdowns < 0
        duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                duration = max(duration, current_duration)
            else:
                current_duration = 0
        
        return current_dd, max_dd, avg_dd if not np.isnan(avg_dd) else 0, duration
    
    def _calculate_downside_volatility(self, returns: np.ndarray) -> float:
        """Calcula volatilidade apenas dos retornos negativos."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            return np.std(negative_returns) * np.sqrt(252)
        return 0
    
    def _calculate_upside_volatility(self, returns: np.ndarray) -> float:
        """Calcula volatilidade apenas dos retornos positivos."""
        positive_returns = returns[returns > 0]
        if len(positive_returns) > 0:
            return np.std(positive_returns) * np.sqrt(252)
        return 0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calcula tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 20:
            return 1
        
        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))
        
        if p5 > 0:
            return p95 / p5
        
        return 1
    
    async def _calculate_greeks(
        self,
        positions: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Calcula Greeks simplificados.
        
        Args:
            positions: Posições
            market_data: Dados de mercado
            
        Returns:
            (delta, gamma, vega)
        """
        # Simplified Greeks calculation
        # In production, would use Black-Scholes or similar
        
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        
        for position in positions:
            # Delta: price sensitivity
            size = position.get('size', 0)
            price = position.get('price', 0)
            total_delta += size * price * 0.01  # 1% price move
            
            # Gamma: delta sensitivity
            total_gamma += abs(size) * 0.001
            
            # Vega: volatility sensitivity
            total_vega += abs(size) * price * 0.1  # 10% vol move
        
        return total_delta, total_gamma, total_vega
    
    def _calculate_risk_score(
        self,
        var_95: float,
        sharpe: float,
        max_dd: float,
        volatility: float,
        skewness: float,
        kurtosis: float
    ) -> float:
        """
        Calcula score de risco geral (0-100).
        
        Args:
            var_95: Value at Risk 95%
            sharpe: Sharpe ratio
            max_dd: Maximum drawdown
            volatility: Annual volatility
            skewness: Skewness
            kurtosis: Excess kurtosis
            
        Returns:
            Risk score
        """
        score = 0
        
        # VaR contribution (0-25)
        score += min(abs(var_95) * 500, 25)  # 5% VaR = 25 points
        
        # Sharpe contribution (0-25)
        if sharpe < 0:
            score += 25
        elif sharpe < 1:
            score += (1 - sharpe) * 25
        
        # Drawdown contribution (0-20)
        score += min(max_dd * 100, 20)  # 20% DD = 20 points
        
        # Volatility contribution (0-15)
        score += min(volatility * 50, 15)  # 30% vol = 15 points
        
        # Tail risk contribution (0-15)
        if skewness < -1:  # Negative skew is bad
            score += 10
        
        if kurtosis > 3:  # Fat tails
            score += 5
        
        return min(score, 100)
    
    def _determine_risk_level(self, var_95: float, risk_score: float) -> RiskLevel:
        """
        Determina nível de risco.
        
        Args:
            var_95: VaR 95%
            risk_score: Score de risco
            
        Returns:
            Nível de risco
        """
        var_pct = abs(var_95) * 100
        
        if var_pct < 1 and risk_score < 20:
            return RiskLevel.MINIMAL
        elif var_pct < 2 and risk_score < 40:
            return RiskLevel.LOW
        elif var_pct < 3 and risk_score < 60:
            return RiskLevel.MODERATE
        elif var_pct < 5 and risk_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    async def stress_test(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executa stress test com cenários.
        
        Args:
            scenarios: Lista de cenários
            
        Returns:
            Resultados do stress test
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get('name', 'unnamed')
            shock = scenario.get('shock', 0)  # Percentage shock
            
            # Apply shock to returns
            shocked_returns = np.array(list(self.returns_history)) * (1 + shock)
            
            # Calculate metrics under stress
            var_95, cvar_95 = await self._calculate_var(shocked_returns, 0.95)
            
            results[name] = {
                'shock': shock,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_loss': np.min(shocked_returns) if len(shocked_returns) > 0 else 0
            }
        
        return results
    
    async def _monitor_loop(self) -> None:
        """Loop de monitoramento contínuo."""
        while True:
            try:
                # Auto-calculate metrics if we have portfolio data
                if len(self.portfolio_values) > 0:
                    await self.calculate_metrics(
                        self.portfolio_values[-1],
                        self.position_history
                    )
                
                # Check risk alerts
                if self.current_metrics.risk_level == RiskLevel.EXTREME:
                    logger.critical(f"EXTREME RISK LEVEL: Score {self.current_metrics.risk_score:.1f}")
                elif self.current_metrics.risk_level == RiskLevel.HIGH:
                    logger.warning(f"High risk level: Score {self.current_metrics.risk_score:.1f}")
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(60)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Retorna relatório completo de risco."""
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'historical_stats': {
                'returns_count': len(self.returns_history),
                'avg_return': np.mean(list(self.returns_history)) * 252 if self.returns_history else 0,
                'win_rate': len([r for r in self.returns_history if r > 0]) / len(self.returns_history) if self.returns_history else 0,
                'best_day': max(self.returns_history) if self.returns_history else 0,
                'worst_day': min(self.returns_history) if self.returns_history else 0
            },
            'limits': {
                'var_95_limit': 0.02,  # 2%
                'max_drawdown_limit': 0.05,  # 5%
                'min_sharpe': 2.5
            }
        }