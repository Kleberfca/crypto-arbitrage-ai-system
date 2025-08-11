"""
Position Sizer - Dimensionamento inteligente de posições
Kelly Criterion, Risk Parity e otimização de portfolio
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

import numpy as np
from numba import jit
from scipy.optimize import minimize
import redis.asyncio as redis

from config.settings import settings
from core.arbitrage_engine.base_arbitrage import BaseOpportunity


logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Métodos de dimensionamento de posição."""
    FIXED = "fixed"  # Tamanho fixo
    KELLY = "kelly"  # Kelly Criterion
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MEAN_VARIANCE = "mean_variance"  # Markowitz optimization
    DYNAMIC = "dynamic"  # Adaptativo baseado em condições


@dataclass
class PositionLimit:
    """Limites de posição."""
    
    # Single position limits
    max_position_size_usd: float = 10000
    max_position_pct: float = 0.02  # 2% of capital
    
    # Strategy limits
    max_per_strategy_pct: float = 0.40  # 40% per strategy
    max_per_exchange_pct: float = 0.30  # 30% per exchange
    max_per_symbol_pct: float = 0.10  # 10% per symbol
    
    # Risk limits
    max_leverage: float = 1.0  # No leverage by default
    max_correlation: float = 0.60  # 60% max correlation
    
    # Minimum sizes
    min_position_size_usd: float = 100
    min_profitable_size_usd: float = 500  # After fees


@dataclass
class CapitalAllocation:
    """Alocação de capital."""
    
    total_capital: float
    allocated_capital: float
    available_capital: float
    
    # Breakdown by strategy
    strategy_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Breakdown by exchange
    exchange_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Breakdown by symbol
    symbol_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Current positions
    open_positions: int = 0
    total_exposure: float = 0
    
    @property
    def utilization_rate(self) -> float:
        """Taxa de utilização do capital."""
        if self.total_capital > 0:
            return self.allocated_capital / self.total_capital
        return 0
    
    @property
    def concentration_score(self) -> float:
        """Score de concentração (0=diversificado, 1=concentrado)."""
        if not self.strategy_allocations:
            return 0
        
        # Herfindahl index
        total = sum(self.strategy_allocations.values())
        if total > 0:
            shares = [v/total for v in self.strategy_allocations.values()]
            return sum(s**2 for s in shares)
        return 0


class PositionSizer:
    """
    Dimensionador de posições com múltiplas estratégias.
    Implementa Kelly, Risk Parity e otimização avançada.
    """
    
    def __init__(self):
        """Inicializa dimensionador de posições."""
        # Configuration
        self.sizing_method = SizingMethod.DYNAMIC
        self.limits = PositionLimit()
        
        # Capital management
        self.total_capital = settings.trading_config.get('total_capital_usd', 10000)
        self.current_allocation = CapitalAllocation(
            total_capital=self.total_capital,
            allocated_capital=0,
            available_capital=self.total_capital
        )
        
        # Kelly Criterion parameters
        self.kelly_fraction = 0.25  # Use 25% of full Kelly for safety
        self.min_win_rate = 0.55  # Minimum 55% win rate
        self.max_kelly_size = 0.20  # Max 20% per position
        
        # Risk parity parameters
        self.target_risk_contribution = 0.10  # 10% risk per position
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
        # Historical data for calculations
        self.win_rates = defaultdict(lambda: 0.5)  # Default 50% win rate
        self.avg_wins = defaultdict(lambda: 0.02)  # Default 2% win
        self.avg_losses = defaultdict(lambda: 0.01)  # Default 1% loss
        self.correlations = {}
        self.volatilities = defaultdict(lambda: 0.01)
        
        # Performance tracking
        self.position_history = []
        self.sizing_accuracy = []
        
        # Redis for persistence
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
    
    async def calculate_position_size(
        self,
        opportunity: BaseOpportunity,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calcula tamanho ótimo da posição.
        
        Args:
            opportunity: Oportunidade de arbitragem
            market_conditions: Condições de mercado
            
        Returns:
            Tamanho da posição em USD
        """
        # Check available capital
        if self.current_allocation.available_capital < self.limits.min_position_size_usd:
            logger.warning("Insufficient capital for new position")
            return 0
        
        # Calculate base size based on method
        if self.sizing_method == SizingMethod.FIXED:
            size = self._fixed_sizing(opportunity)
            
        elif self.sizing_method == SizingMethod.KELLY:
            size = self._kelly_sizing(opportunity)
            
        elif self.sizing_method == SizingMethod.RISK_PARITY:
            size = await self._risk_parity_sizing(opportunity)
            
        elif self.sizing_method == SizingMethod.MEAN_VARIANCE:
            size = await self._mean_variance_sizing(opportunity)
            
        else:  # DYNAMIC
            size = await self._dynamic_sizing(opportunity, market_conditions)
        
        # Apply limits
        size = self._apply_limits(size, opportunity)
        
        # Check profitability after fees
        if not self._is_profitable_after_fees(size, opportunity):
            logger.debug(f"Position not profitable after fees: ${size:.2f}")
            return 0
        
        # Update allocation
        if size > 0:
            self._update_allocation(size, opportunity)
        
        return size
    
    def _fixed_sizing(self, opportunity: BaseOpportunity) -> float:
        """
        Dimensionamento fixo.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            Tamanho fixo
        """
        # Use fixed percentage of capital
        return self.total_capital * self.limits.max_position_pct
    
    def _kelly_sizing(self, opportunity: BaseOpportunity) -> float:
        """
        Kelly Criterion para dimensionamento.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            Tamanho segundo Kelly
        """
        strategy = opportunity.opportunity_type.value
        
        # Get historical statistics
        win_rate = self.win_rates[strategy]
        avg_win = self.avg_wins[strategy]
        avg_loss = self.avg_losses[strategy]
        
        # Check minimum win rate
        if win_rate < self.min_win_rate:
            return self.limits.min_position_size_usd
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            
            kelly_fraction = (win_rate * b - q) / b
            
            # Apply safety factor
            kelly_fraction *= self.kelly_fraction
            
            # Cap at maximum
            kelly_fraction = min(kelly_fraction, self.max_kelly_size)
            
            if kelly_fraction > 0:
                return self.total_capital * kelly_fraction
        
        return self.limits.min_position_size_usd
    
    async def _risk_parity_sizing(self, opportunity: BaseOpportunity) -> float:
        """
        Risk Parity para equalizar contribuição de risco.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            Tamanho para risk parity
        """
        strategy = opportunity.opportunity_type.value
        
        # Get volatility
        volatility = self.volatilities[strategy]
        
        if volatility > 0:
            # Size inversely proportional to volatility
            # To achieve equal risk contribution
            target_risk = self.total_capital * self.target_risk_contribution
            size = target_risk / volatility
            
            return size
        
        return self.limits.min_position_size_usd
    
    async def _mean_variance_sizing(self, opportunity: BaseOpportunity) -> float:
        """
        Otimização mean-variance (Markowitz).
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            Tamanho ótimo
        """
        # Simplified mean-variance optimization
        # In production, would use full portfolio optimization
        
        expected_return = opportunity.expected_profit_pct / 100
        risk = opportunity.risk_score / 100
        
        if risk > 0:
            # Sharpe ratio optimization
            risk_adjusted_size = expected_return / (risk ** 2)
            
            # Scale to capital
            size = self.total_capital * min(risk_adjusted_size, self.limits.max_position_pct)
            
            return size
        
        return self.limits.min_position_size_usd
    
    async def _dynamic_sizing(
        self,
        opportunity: BaseOpportunity,
        market_conditions: Optional[Dict[str, Any]]
    ) -> float:
        """
        Dimensionamento dinâmico adaptativo.
        
        Args:
            opportunity: Oportunidade
            market_conditions: Condições de mercado
            
        Returns:
            Tamanho adaptativo
        """
        # Start with Kelly as base
        base_size = self._kelly_sizing(opportunity)
        
        # Adjust for market conditions
        if market_conditions:
            volatility = market_conditions.get('volatility', 0.01)
            liquidity = market_conditions.get('liquidity', 1.0)
            trend = market_conditions.get('trend', 0)
            
            # Reduce size in high volatility
            if volatility > 0.02:  # 2% volatility
                base_size *= 0.5
            elif volatility < 0.005:  # Low volatility
                base_size *= 1.2
            
            # Adjust for liquidity
            if liquidity < 0.5:  # Low liquidity
                base_size *= 0.7
            
            # Adjust for trend (momentum)
            if abs(trend) > 0.5:  # Strong trend
                base_size *= 1.1
        
        # Adjust for opportunity confidence
        base_size *= opportunity.confidence
        
        # Adjust for current allocation
        utilization = self.current_allocation.utilization_rate
        if utilization > 0.8:  # High utilization
            base_size *= 0.5
        elif utilization < 0.3:  # Low utilization
            base_size *= 1.3
        
        return base_size
    
    def _apply_limits(
        self,
        size: float,
        opportunity: BaseOpportunity
    ) -> float:
        """
        Aplica limites de posição.
        
        Args:
            size: Tamanho calculado
            opportunity: Oportunidade
            
        Returns:
            Tamanho após limites
        """
        # Apply maximum position size
        size = min(size, self.limits.max_position_size_usd)
        size = min(size, self.total_capital * self.limits.max_position_pct)
        
        # Apply minimum position size
        size = max(size, self.limits.min_position_size_usd)
        
        # Check strategy allocation limit
        strategy = opportunity.opportunity_type.value
        current_strategy_alloc = self.current_allocation.strategy_allocations.get(strategy, 0)
        max_strategy_alloc = self.total_capital * self.limits.max_per_strategy_pct
        
        if current_strategy_alloc + size > max_strategy_alloc:
            size = max(0, max_strategy_alloc - current_strategy_alloc)
        
        # Check available capital
        size = min(size, self.current_allocation.available_capital)
        
        return size
    
    def _is_profitable_after_fees(
        self,
        size: float,
        opportunity: BaseOpportunity
    ) -> bool:
        """
        Verifica se posição é lucrativa após fees.
        
        Args:
            size: Tamanho da posição
            opportunity: Oportunidade
            
        Returns:
            True se lucrativa
        """
        if size < self.limits.min_profitable_size_usd:
            return False
        
        # Estimate fees (simplified)
        fee_rate = 0.001  # 0.1%
        total_fees = size * fee_rate * 2  # Buy and sell
        
        expected_profit = size * (opportunity.expected_profit_pct / 100)
        
        return expected_profit > total_fees * 1.5  # 50% margin above fees
    
    def _update_allocation(
        self,
        size: float,
        opportunity: BaseOpportunity
    ) -> None:
        """
        Atualiza alocação de capital.
        
        Args:
            size: Tamanho da posição
            opportunity: Oportunidade
        """
        # Update capital
        self.current_allocation.allocated_capital += size
        self.current_allocation.available_capital -= size
        self.current_allocation.open_positions += 1
        
        # Update strategy allocation
        strategy = opportunity.opportunity_type.value
        if strategy not in self.current_allocation.strategy_allocations:
            self.current_allocation.strategy_allocations[strategy] = 0
        self.current_allocation.strategy_allocations[strategy] += size
        
        # Update exchange allocation (if available in metadata)
        exchange = opportunity.metadata.get('exchange', 'unknown')
        if exchange not in self.current_allocation.exchange_allocations:
            self.current_allocation.exchange_allocations[exchange] = 0
        self.current_allocation.exchange_allocations[exchange] += size
        
        # Update symbol allocation (if available in metadata)
        symbol = opportunity.metadata.get('symbol', 'unknown')
        if symbol not in self.current_allocation.symbol_allocations:
            self.current_allocation.symbol_allocations[symbol] = 0
        self.current_allocation.symbol_allocations[symbol] += size
        
        # Log allocation
        logger.info(
            f"Position allocated",
            size=f"${size:.2f}",
            strategy=strategy,
            utilization=f"{self.current_allocation.utilization_rate:.2%}"
        )
    
    @jit(nopython=True)
    def _optimize_portfolio_weights(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float
    ) -> np.ndarray:
        """
        Otimiza pesos do portfolio (Markowitz).
        
        Args:
            expected_returns: Retornos esperados
            covariance_matrix: Matriz de covariância
            risk_aversion: Aversão ao risco
            
        Returns:
            Pesos ótimos
        """
        n = len(expected_returns)
        
        # Simplified optimization (in production, use cvxpy or similar)
        # Max: returns - risk_aversion * variance
        
        # Equal weight as starting point
        weights = np.ones(n) / n
        
        # Iterative optimization (simplified)
        for _ in range(100):
            # Calculate gradient
            gradient = expected_returns - risk_aversion * np.dot(covariance_matrix, weights)
            
            # Update weights
            weights += 0.01 * gradient
            
            # Project to simplex (weights sum to 1, all positive)
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
        
        return weights
    
    def update_statistics(
        self,
        strategy: str,
        won: bool,
        profit_pct: float
    ) -> None:
        """
        Atualiza estatísticas para Kelly Criterion.
        
        Args:
            strategy: Estratégia
            won: Se ganhou
            profit_pct: Percentual de lucro/perda
        """
        # Update win rate (exponential moving average)
        alpha = 0.1  # Learning rate
        current_win_rate = self.win_rates[strategy]
        self.win_rates[strategy] = (1 - alpha) * current_win_rate + alpha * (1 if won else 0)
        
        # Update average win/loss
        if won:
            current_avg_win = self.avg_wins[strategy]
            self.avg_wins[strategy] = (1 - alpha) * current_avg_win + alpha * abs(profit_pct)
        else:
            current_avg_loss = self.avg_losses[strategy]
            self.avg_losses[strategy] = (1 - alpha) * current_avg_loss + alpha * abs(profit_pct)
        
        # Update volatility
        current_vol = self.volatilities[strategy]
        self.volatilities[strategy] = (1 - alpha) * current_vol + alpha * abs(profit_pct)
    
    def release_capital(
        self,
        size: float,
        strategy: str,
        exchange: str = "unknown",
        symbol: str = "unknown"
    ) -> None:
        """
        Libera capital após fechar posição.
        
        Args:
            size: Tamanho da posição
            strategy: Estratégia
            exchange: Exchange
            symbol: Símbolo
        """
        # Update capital
        self.current_allocation.allocated_capital -= size
        self.current_allocation.available_capital += size
        self.current_allocation.open_positions -= 1
        
        # Update allocations
        if strategy in self.current_allocation.strategy_allocations:
            self.current_allocation.strategy_allocations[strategy] -= size
            if self.current_allocation.strategy_allocations[strategy] <= 0:
                del self.current_allocation.strategy_allocations[strategy]
        
        if exchange in self.current_allocation.exchange_allocations:
            self.current_allocation.exchange_allocations[exchange] -= size
            if self.current_allocation.exchange_allocations[exchange] <= 0:
                del self.current_allocation.exchange_allocations[exchange]
        
        if symbol in self.current_allocation.symbol_allocations:
            self.current_allocation.symbol_allocations[symbol] -= size
            if self.current_allocation.symbol_allocations[symbol] <= 0:
                del self.current_allocation.symbol_allocations[symbol]
        
        logger.info(
            f"Capital released",
            size=f"${size:.2f}",
            available=f"${self.current_allocation.available_capital:.2f}"
        )
    
    def get_allocation_report(self) -> Dict[str, Any]:
        """Retorna relatório de alocação."""
        return {
            'capital': {
                'total': self.total_capital,
                'allocated': self.current_allocation.allocated_capital,
                'available': self.current_allocation.available_capital,
                'utilization_rate': self.current_allocation.utilization_rate
            },
            'positions': {
                'open': self.current_allocation.open_positions,
                'concentration_score': self.current_allocation.concentration_score
            },
            'by_strategy': dict(self.current_allocation.strategy_allocations),
            'by_exchange': dict(self.current_allocation.exchange_allocations),
            'by_symbol': dict(self.current_allocation.symbol_allocations),
            'kelly_stats': {
                'win_rates': dict(self.win_rates),
                'avg_wins': dict(self.avg_wins),
                'avg_losses': dict(self.avg_losses)
            },
            'limits': {
                'max_position_pct': self.limits.max_position_pct,
                'max_per_strategy_pct': self.limits.max_per_strategy_pct,
                'max_correlation': self.limits.max_correlation
            }
        }