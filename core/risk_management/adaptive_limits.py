"""
Adaptive Limits - Sistema de limites adaptativos de risco
Ajusta limites dinamicamente baseado em condições de mercado
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
from numba import jit
import redis.asyncio as redis

from config.settings import settings
from core.risk_management.risk_calculator import RiskMetrics, RiskLevel


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Regimes de mercado."""
    BULL = "bull"  # Trending up
    BEAR = "bear"  # Trending down
    SIDEWAYS = "sideways"  # Range-bound
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"  # Extreme conditions


class LimitType(Enum):
    """Tipos de limites."""
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    VAR = "var"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"


@dataclass
class DynamicThreshold:
    """Threshold dinâmico adaptativo."""
    
    limit_type: LimitType
    base_value: float
    current_value: float
    min_value: float
    max_value: float
    
    # Adjustment parameters
    adjustment_speed: float = 0.1  # How fast to adjust
    confidence_required: float = 0.7  # Minimum confidence to relax
    
    # History
    value_history: deque = field(default_factory=lambda: deque(maxlen=100))
    adjustment_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    @property
    def adjustment_ratio(self) -> float:
        """Ratio of current to base value."""
        if self.base_value > 0:
            return self.current_value / self.base_value
        return 1.0
    
    @property
    def is_tightened(self) -> bool:
        """Check if limit is tightened."""
        return self.current_value < self.base_value
    
    @property
    def is_relaxed(self) -> bool:
        """Check if limit is relaxed."""
        return self.current_value > self.base_value


class AdaptiveLimits:
    """
    Sistema de limites adaptativos de risco.
    Ajusta limites baseado em performance e condições de mercado.
    """
    
    def __init__(self):
        """Inicializa sistema de limites adaptativos."""
        # Market regime detection
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = deque(maxlen=100)
        self.regime_confidence = 0.5
        
        # Dynamic thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Performance tracking
        self.performance_window = 20  # Days
        self.performance_history = deque(maxlen=self.performance_window)
        self.win_rate_history = deque(maxlen=self.performance_window)
        
        # Market conditions
        self.volatility_history = deque(maxlen=100)
        self.liquidity_history = deque(maxlen=100)
        self.correlation_history = deque(maxlen=100)
        
        # Adjustment parameters
        self.min_data_points = 10  # Minimum data before adjusting
        self.adjustment_interval = 3600  # 1 hour
        self.last_adjustment = 0
        self.emergency_mode = False
        
        # Risk metrics for decisions
        self.current_risk_metrics = None
        self.risk_tolerance = 0.5  # 0=conservative, 1=aggressive
        
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
    
    def _initialize_thresholds(self) -> Dict[LimitType, DynamicThreshold]:
        """Inicializa thresholds dinâmicos."""
        return {
            LimitType.POSITION_SIZE: DynamicThreshold(
                limit_type=LimitType.POSITION_SIZE,
                base_value=0.02,  # 2% of capital
                current_value=0.02,
                min_value=0.005,  # 0.5%
                max_value=0.05  # 5%
            ),
            LimitType.DAILY_LOSS: DynamicThreshold(
                limit_type=LimitType.DAILY_LOSS,
                base_value=0.03,  # 3% daily loss
                current_value=0.03,
                min_value=0.01,  # 1%
                max_value=0.05  # 5%
            ),
            LimitType.DRAWDOWN: DynamicThreshold(
                limit_type=LimitType.DRAWDOWN,
                base_value=0.05,  # 5% max drawdown
                current_value=0.05,
                min_value=0.02,  # 2%
                max_value=0.10  # 10%
            ),
            LimitType.LEVERAGE: DynamicThreshold(
                limit_type=LimitType.LEVERAGE,
                base_value=1.0,  # No leverage
                current_value=1.0,
                min_value=0.5,  # 0.5x
                max_value=2.0  # 2x
            ),
            LimitType.CORRELATION: DynamicThreshold(
                limit_type=LimitType.CORRELATION,
                base_value=0.60,  # 60% max correlation
                current_value=0.60,
                min_value=0.30,  # 30%
                max_value=0.80  # 80%
            ),
            LimitType.VAR: DynamicThreshold(
                limit_type=LimitType.VAR,
                base_value=0.02,  # 2% VaR
                current_value=0.02,
                min_value=0.01,  # 1%
                max_value=0.05  # 5%
            ),
            LimitType.EXPOSURE: DynamicThreshold(
                limit_type=LimitType.EXPOSURE,
                base_value=0.70,  # 70% max exposure
                current_value=0.70,
                min_value=0.30,  # 30%
                max_value=1.00  # 100%
            ),
            LimitType.CONCENTRATION: DynamicThreshold(
                limit_type=LimitType.CONCENTRATION,
                base_value=0.30,  # 30% max concentration
                current_value=0.30,
                min_value=0.10,  # 10%
                max_value=0.50  # 50%
            )
        }
    
    async def detect_market_regime(
        self,
        market_data: Dict[str, Any]
    ) -> MarketRegime:
        """
        Detecta regime de mercado atual.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Regime detectado
        """
        returns = market_data.get('returns', [])
        volatility = market_data.get('volatility', 0.01)
        volume = market_data.get('volume', 0)
        trend = market_data.get('trend', 0)
        
        # Store volatility
        self.volatility_history.append(volatility)
        
        # Detect regime based on multiple factors
        regime_scores = {
            MarketRegime.BULL: 0,
            MarketRegime.BEAR: 0,
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.HIGH_VOLATILITY: 0,
            MarketRegime.LOW_VOLATILITY: 0,
            MarketRegime.CRISIS: 0
        }
        
        # Trend analysis
        if trend > 0.5:
            regime_scores[MarketRegime.BULL] += 0.4
        elif trend < -0.5:
            regime_scores[MarketRegime.BEAR] += 0.4
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 0.3
        
        # Volatility analysis
        avg_volatility = np.mean(list(self.volatility_history)) if self.volatility_history else 0.01
        
        if volatility > avg_volatility * 2:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.3
            if volatility > avg_volatility * 3:
                regime_scores[MarketRegime.CRISIS] += 0.2
        elif volatility < avg_volatility * 0.5:
            regime_scores[MarketRegime.LOW_VOLATILITY] += 0.3
        
        # Returns analysis
        if len(returns) > 10:
            recent_returns = returns[-10:]
            if np.mean(recent_returns) > 0.02:  # 2% average
                regime_scores[MarketRegime.BULL] += 0.2
            elif np.mean(recent_returns) < -0.02:
                regime_scores[MarketRegime.BEAR] += 0.2
            
            # Check for crisis (large negative returns)
            if min(recent_returns) < -0.05:  # 5% loss
                regime_scores[MarketRegime.CRISIS] += 0.3
        
        # Volume analysis (simplified)
        if volume > 0:
            # High volume in volatile market = crisis
            if volatility > avg_volatility * 2 and volume > market_data.get('avg_volume', volume) * 1.5:
                regime_scores[MarketRegime.CRISIS] += 0.1
        
        # Select regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        self.regime_confidence = regime_scores[best_regime]
        
        # Update history
        self.current_regime = best_regime
        self.regime_history.append(best_regime)
        
        logger.info(
            f"Market regime detected",
            regime=best_regime.value,
            confidence=f"{self.regime_confidence:.2f}"
        )
        
        return best_regime
    
    async def adjust_limits(
        self,
        risk_metrics: RiskMetrics,
        performance_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[LimitType, float]:
        """
        Ajusta limites baseado em condições.
        
        Args:
            risk_metrics: Métricas de risco atuais
            performance_data: Dados de performance
            market_data: Dados de mercado
            
        Returns:
            Novos limites
        """
        # Check if enough time passed
        if time.time() - self.last_adjustment < self.adjustment_interval:
            return self.get_current_limits()
        
        # Update data
        self.current_risk_metrics = risk_metrics
        
        if performance_data:
            self.performance_history.append(performance_data.get('daily_return', 0))
            self.win_rate_history.append(performance_data.get('win_rate', 0.5))
        
        # Detect market regime
        regime = await self.detect_market_regime(market_data)
        
        # Check for emergency conditions
        self._check_emergency_conditions(risk_metrics)
        
        # Adjust each threshold
        for limit_type, threshold in self.thresholds.items():
            new_value = await self._adjust_single_limit(
                threshold, regime, risk_metrics, performance_data
            )
            
            # Apply the adjustment
            if new_value != threshold.current_value:
                self._apply_adjustment(threshold, new_value)
        
        self.last_adjustment = time.time()
        
        return self.get_current_limits()
    
    async def _adjust_single_limit(
        self,
        threshold: DynamicThreshold,
        regime: MarketRegime,
        risk_metrics: RiskMetrics,
        performance_data: Dict[str, Any]
    ) -> float:
        """
        Ajusta um limite individual.
        
        Args:
            threshold: Threshold a ajustar
            regime: Regime de mercado
            risk_metrics: Métricas de risco
            performance_data: Dados de performance
            
        Returns:
            Novo valor
        """
        current = threshold.current_value
        adjustment = 0
        
        # Regime-based adjustments
        if regime == MarketRegime.CRISIS:
            # Tighten all limits in crisis
            adjustment = -0.5  # 50% tighter
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Tighten risk limits
            if threshold.limit_type in [LimitType.POSITION_SIZE, LimitType.LEVERAGE, LimitType.EXPOSURE]:
                adjustment = -0.3  # 30% tighter
                
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Can relax some limits if performance is good
            if self._is_performance_good(performance_data):
                adjustment = 0.2  # 20% relaxed
                
        elif regime == MarketRegime.BULL:
            # Slightly relax in bull market with good performance
            if self._is_performance_good(performance_data):
                adjustment = 0.1  # 10% relaxed
                
        elif regime == MarketRegime.BEAR:
            # Tighten in bear market
            adjustment = -0.2  # 20% tighter
        
        # Risk-based adjustments
        if risk_metrics:
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                adjustment = min(adjustment, -0.5)  # At least 50% tighter
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                adjustment = min(adjustment, -0.2)  # At least 20% tighter
            elif risk_metrics.risk_level == RiskLevel.MINIMAL:
                if self.regime_confidence > threshold.confidence_required:
                    adjustment = max(adjustment, 0.1)  # Can relax 10%
        
        # Performance-based adjustments
        if len(self.performance_history) >= self.min_data_points:
            avg_return = np.mean(list(self.performance_history))
            win_rate = np.mean(list(self.win_rate_history))
            
            if avg_return > 0.01 and win_rate > 0.60:  # Good performance
                adjustment = max(adjustment, 0.15)  # Relax up to 15%
            elif avg_return < -0.01 or win_rate < 0.40:  # Poor performance
                adjustment = min(adjustment, -0.25)  # Tighten at least 25%
        
        # Apply adjustment with speed factor
        adjustment *= threshold.adjustment_speed
        
        # Calculate new value
        new_value = current * (1 + adjustment)
        
        # Apply bounds
        new_value = max(threshold.min_value, min(threshold.max_value, new_value))
        
        return new_value
    
    def _is_performance_good(
        self,
        performance_data: Dict[str, Any]
    ) -> bool:
        """
        Verifica se performance é boa.
        
        Args:
            performance_data: Dados de performance
            
        Returns:
            True se performance é boa
        """
        if not performance_data:
            return False
        
        sharpe = performance_data.get('sharpe_ratio', 0)
        win_rate = performance_data.get('win_rate', 0)
        max_dd = performance_data.get('max_drawdown', 1)
        
        return sharpe > 2 and win_rate > 0.55 and max_dd < 0.05
    
    def _check_emergency_conditions(
        self,
        risk_metrics: RiskMetrics
    ) -> None:
        """
        Verifica condições de emergência.
        
        Args:
            risk_metrics: Métricas de risco
        """
        emergency_triggers = []
        
        # Check VaR breach
        if risk_metrics.var_95 > 0.05:  # 5% VaR
            emergency_triggers.append("VaR > 5%")
        
        # Check drawdown
        if risk_metrics.max_drawdown > 0.10:  # 10% drawdown
            emergency_triggers.append("Drawdown > 10%")
        
        # Check risk level
        if risk_metrics.risk_level == RiskLevel.EXTREME:
            emergency_triggers.append("Extreme risk level")
        
        # Check daily loss
        if len(self.performance_history) > 0:
            if self.performance_history[-1] < -0.05:  # 5% daily loss
                emergency_triggers.append("Daily loss > 5%")
        
        # Activate emergency mode if triggered
        if emergency_triggers:
            self.emergency_mode = True
            logger.critical(
                f"EMERGENCY MODE ACTIVATED",
                triggers=emergency_triggers
            )
            
            # Tighten all limits immediately
            for threshold in self.thresholds.values():
                emergency_value = threshold.base_value * 0.3  # 70% reduction
                emergency_value = max(threshold.min_value, emergency_value)
                self._apply_adjustment(threshold, emergency_value, is_emergency=True)
        
        elif self.emergency_mode:
            # Check if we can exit emergency mode
            if risk_metrics.risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]:
                self.emergency_mode = False
                logger.info("Emergency mode deactivated")
    
    def _apply_adjustment(
        self,
        threshold: DynamicThreshold,
        new_value: float,
        is_emergency: bool = False
    ) -> None:
        """
        Aplica ajuste a um threshold.
        
        Args:
            threshold: Threshold a ajustar
            new_value: Novo valor
            is_emergency: Se é ajuste de emergência
        """
        old_value = threshold.current_value
        threshold.current_value = new_value
        
        # Store in history
        threshold.value_history.append(new_value)
        threshold.adjustment_history.append({
            'timestamp': time.time(),
            'old_value': old_value,
            'new_value': new_value,
            'is_emergency': is_emergency,
            'regime': self.current_regime.value
        })
        
        # Log significant changes
        change_pct = abs(new_value - old_value) / old_value * 100 if old_value > 0 else 0
        
        if change_pct > 10:  # More than 10% change
            log_level = logging.CRITICAL if is_emergency else logging.INFO
            logger.log(
                log_level,
                f"Limit adjusted",
                limit_type=threshold.limit_type.value,
                old=f"{old_value:.4f}",
                new=f"{new_value:.4f}",
                change=f"{change_pct:.1f}%"
            )
    
    @jit(nopython=True)
    def _calculate_dynamic_var_limit(
        self,
        returns: np.ndarray,
        volatility: float,
        confidence: float
    ) -> float:
        """
        Calcula limite VaR dinâmico.
        
        Args:
            returns: Retornos históricos
            volatility: Volatilidade atual
            confidence: Nível de confiança
            
        Returns:
            Limite VaR
        """
        if len(returns) < 10:
            return 0.02  # Default 2%
        
        # Calculate empirical VaR
        empirical_var = np.percentile(returns, (1 - confidence) * 100)
        
        # Adjust for current volatility
        hist_volatility = np.std(returns)
        
        if hist_volatility > 0:
            volatility_adjustment = volatility / hist_volatility
            adjusted_var = abs(empirical_var) * volatility_adjustment
        else:
            adjusted_var = abs(empirical_var)
        
        # Apply bounds
        return min(0.05, max(0.01, adjusted_var))  # Between 1% and 5%
    
    def get_current_limits(self) -> Dict[LimitType, float]:
        """Retorna limites atuais."""
        return {
            limit_type: threshold.current_value
            for limit_type, threshold in self.thresholds.items()
        }
    
    def get_limit(self, limit_type: LimitType) -> float:
        """
        Retorna limite específico.
        
        Args:
            limit_type: Tipo de limite
            
        Returns:
            Valor do limite
        """
        threshold = self.thresholds.get(limit_type)
        return threshold.current_value if threshold else 0
    
    def override_limit(
        self,
        limit_type: LimitType,
        value: float,
        reason: str = ""
    ) -> None:
        """
        Override manual de limite.
        
        Args:
            limit_type: Tipo de limite
            value: Novo valor
            reason: Razão do override
        """
        threshold = self.thresholds.get(limit_type)
        
        if threshold:
            old_value = threshold.current_value
            threshold.current_value = max(threshold.min_value, min(threshold.max_value, value))
            
            logger.warning(
                f"Manual limit override",
                limit_type=limit_type.value,
                old=old_value,
                new=threshold.current_value,
                reason=reason
            )
    
    async def _monitor_loop(self) -> None:
        """Loop de monitoramento contínuo."""
        while True:
            try:
                # Check if we should adjust limits
                if len(self.performance_history) >= self.min_data_points:
                    # Auto-adjust if we have metrics
                    if self.current_risk_metrics:
                        performance_data = {
                            'daily_return': self.performance_history[-1] if self.performance_history else 0,
                            'win_rate': self.win_rate_history[-1] if self.win_rate_history else 0.5,
                            'sharpe_ratio': self.current_risk_metrics.sharpe_ratio,
                            'max_drawdown': self.current_risk_metrics.max_drawdown
                        }
                        
                        market_data = {
                            'volatility': self.volatility_history[-1] if self.volatility_history else 0.01,
                            'returns': list(self.performance_history)
                        }
                        
                        await self.adjust_limits(
                            self.current_risk_metrics,
                            performance_data,
                            market_data
                        )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in adaptive limits monitor: {e}")
                await asyncio.sleep(300)
    
    def get_limits_report(self) -> Dict[str, Any]:
        """Retorna relatório de limites."""
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'emergency_mode': self.emergency_mode,
            'limits': {
                limit_type.value: {
                    'current': threshold.current_value,
                    'base': threshold.base_value,
                    'adjustment_ratio': threshold.adjustment_ratio,
                    'is_tightened': threshold.is_tightened,
                    'is_relaxed': threshold.is_relaxed
                }
                for limit_type, threshold in self.thresholds.items()
            },
            'performance': {
                'avg_return': np.mean(list(self.performance_history)) if self.performance_history else 0,
                'avg_win_rate': np.mean(list(self.win_rate_history)) if self.win_rate_history else 0.5,
                'current_volatility': self.volatility_history[-1] if self.volatility_history else 0
            },
            'last_adjustment': self.last_adjustment
        }