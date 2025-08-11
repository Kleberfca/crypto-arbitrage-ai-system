"""
Correlation Monitor - Monitoramento de correlações entre estratégias
Evita concentração de risco e mantém diversificação
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging

import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import pearsonr, spearmanr
import redis.asyncio as redis

from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Matriz de correlação entre estratégias/ativos."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Correlation matrices
    strategy_correlations: np.ndarray = field(default_factory=lambda: np.array([]))
    exchange_correlations: np.ndarray = field(default_factory=lambda: np.array([]))
    symbol_correlations: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Labels
    strategy_labels: List[str] = field(default_factory=list)
    exchange_labels: List[str] = field(default_factory=list)
    symbol_labels: List[str] = field(default_factory=list)
    
    # Statistics
    avg_correlation: float = 0
    max_correlation: float = 0
    min_correlation: float = 0
    
    def get_correlation(
        self,
        item1: str,
        item2: str,
        matrix_type: str = "strategy"
    ) -> Optional[float]:
        """
        Retorna correlação entre dois items.
        
        Args:
            item1: Primeiro item
            item2: Segundo item
            matrix_type: Tipo de matriz
            
        Returns:
            Correlação ou None
        """
        if matrix_type == "strategy":
            labels = self.strategy_labels
            matrix = self.strategy_correlations
        elif matrix_type == "exchange":
            labels = self.exchange_labels
            matrix = self.exchange_correlations
        else:
            labels = self.symbol_labels
            matrix = self.symbol_correlations
        
        if item1 in labels and item2 in labels:
            idx1 = labels.index(item1)
            idx2 = labels.index(item2)
            
            if idx1 < len(matrix) and idx2 < len(matrix):
                return matrix[idx1, idx2]
        
        return None


@dataclass
class DiversificationScore:
    """Score de diversificação do portfolio."""
    
    score: float  # 0-1 (1 = perfeitamente diversificado)
    
    # Components
    strategy_diversification: float = 0
    exchange_diversification: float = 0
    symbol_diversification: float = 0
    temporal_diversification: float = 0
    
    # Risk metrics
    concentration_risk: float = 0
    correlation_risk: float = 0
    
    @property
    def risk_level(self) -> str:
        """Retorna nível de risco de concentração."""
        if self.score > 0.8:
            return "low"
        elif self.score > 0.5:
            return "moderate"
        else:
            return "high"


@dataclass
class ConcentrationRisk:
    """Risco de concentração."""
    
    # Herfindahl-Hirschman Index (HHI)
    hhi_strategies: float = 0
    hhi_exchanges: float = 0
    hhi_symbols: float = 0
    
    # Concentration ratios
    top1_concentration: float = 0  # Largest position
    top3_concentration: float = 0  # Top 3 positions
    top5_concentration: float = 0  # Top 5 positions
    
    # Risk flags
    is_concentrated: bool = False
    concentration_warnings: List[str] = field(default_factory=list)


class CorrelationMonitor:
    """
    Monitor de correlações para manter diversificação.
    Calcula e monitora correlações entre estratégias, exchanges e símbolos.
    """
    
    def __init__(self):
        """Inicializa monitor de correlações."""
        # Configuration
        self.max_correlation = 0.60  # 60% maximum correlation
        self.lookback_window = 100  # Data points for correlation
        self.update_interval = 300  # 5 minutes
        
        # Data storage
        self.returns_data = defaultdict(lambda: deque(maxlen=self.lookback_window))
        self.position_data = defaultdict(list)
        
        # Current matrices
        self.current_matrix = CorrelationMatrix()
        self.matrix_history = deque(maxlen=100)
        
        # Diversification tracking
        self.current_diversification = DiversificationScore(score=1.0)
        self.concentration_risk = ConcentrationRisk()
        
        # Correlation methods
        self.use_robust_correlation = True  # Use Spearman vs Pearson
        self.correlation_threshold = 0.7  # Alert threshold
        
        # Performance tracking
        self.correlation_breaches = []
        self.last_update = 0
        
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
    
    async def update_correlation_matrix(
        self,
        returns_data: Optional[Dict[str, List[float]]] = None
    ) -> CorrelationMatrix:
        """
        Atualiza matriz de correlação.
        
        Args:
            returns_data: Dados de retorno opcionais
            
        Returns:
            Matriz atualizada
        """
        if returns_data:
            # Update internal data
            for key, returns in returns_data.items():
                self.returns_data[key].extend(returns)
        
        # Calculate correlation matrices
        strategy_matrix = await self._calculate_correlation_matrix("strategy")
        exchange_matrix = await self._calculate_correlation_matrix("exchange")
        symbol_matrix = await self._calculate_correlation_matrix("symbol")
        
        # Update current matrix
        self.current_matrix = CorrelationMatrix(
            strategy_correlations=strategy_matrix['matrix'],
            strategy_labels=strategy_matrix['labels'],
            exchange_correlations=exchange_matrix['matrix'],
            exchange_labels=exchange_matrix['labels'],
            symbol_correlations=symbol_matrix['matrix'],
            symbol_labels=symbol_matrix['labels']
        )
        
        # Calculate statistics
        self._update_matrix_statistics()
        
        # Store in history
        self.matrix_history.append(self.current_matrix)
        self.last_update = time.time()
        
        # Check for high correlations
        await self._check_correlation_breaches()
        
        return self.current_matrix
    
    async def _calculate_correlation_matrix(
        self,
        matrix_type: str
    ) -> Dict[str, Any]:
        """
        Calcula matriz de correlação específica.
        
        Args:
            matrix_type: Tipo de matriz
            
        Returns:
            Matriz e labels
        """
        # Filter data by type
        if matrix_type == "strategy":
            prefix = "strategy_"
        elif matrix_type == "exchange":
            prefix = "exchange_"
        else:
            prefix = "symbol_"
        
        # Get relevant data
        relevant_data = {
            k: v for k, v in self.returns_data.items()
            if k.startswith(prefix)
        }
        
        if len(relevant_data) < 2:
            return {'matrix': np.array([]), 'labels': []}
        
        # Convert to DataFrame for easier calculation
        labels = [k.replace(prefix, "") for k in relevant_data.keys()]
        data_matrix = []
        
        for key in relevant_data:
            data_matrix.append(list(relevant_data[key]))
        
        # Ensure all series have same length
        min_length = min(len(series) for series in data_matrix)
        if min_length < 10:
            return {'matrix': np.eye(len(labels)), 'labels': labels}
        
        data_matrix = np.array([series[:min_length] for series in data_matrix])
        
        # Calculate correlation
        if self.use_robust_correlation:
            correlation_matrix = self._calculate_spearman_correlation(data_matrix)
        else:
            correlation_matrix = self._calculate_pearson_correlation(data_matrix)
        
        return {'matrix': correlation_matrix, 'labels': labels}
    
    @jit(nopython=True)
    def _calculate_pearson_correlation(
        self,
        data_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calcula correlação de Pearson otimizada.
        
        Args:
            data_matrix: Matriz de dados
            
        Returns:
            Matriz de correlação
        """
        n = len(data_matrix)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate correlation
                x = data_matrix[i]
                y = data_matrix[j]
                
                # Remove NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) < 3:
                    correlation = 0
                else:
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    # Pearson correlation
                    mean_x = np.mean(x_clean)
                    mean_y = np.mean(y_clean)
                    
                    numerator = np.sum((x_clean - mean_x) * (y_clean - mean_y))
                    denominator = np.sqrt(np.sum((x_clean - mean_x)**2) * np.sum((y_clean - mean_y)**2))
                    
                    if denominator > 0:
                        correlation = numerator / denominator
                    else:
                        correlation = 0
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def _calculate_spearman_correlation(
        self,
        data_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calcula correlação de Spearman (rank correlation).
        
        Args:
            data_matrix: Matriz de dados
            
        Returns:
            Matriz de correlação
        """
        n = len(data_matrix)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    correlation, _ = spearmanr(data_matrix[i], data_matrix[j])
                    if not np.isnan(correlation):
                        corr_matrix[i, j] = correlation
                        corr_matrix[j, i] = correlation
                except:
                    pass
        
        return corr_matrix
    
    def _update_matrix_statistics(self) -> None:
        """Atualiza estatísticas da matriz de correlação."""
        all_correlations = []
        
        for matrix in [
            self.current_matrix.strategy_correlations,
            self.current_matrix.exchange_correlations,
            self.current_matrix.symbol_correlations
        ]:
            if matrix.size > 0:
                # Get upper triangle (excluding diagonal)
                upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
                all_correlations.extend(upper_triangle)
        
        if all_correlations:
            self.current_matrix.avg_correlation = np.mean(np.abs(all_correlations))
            self.current_matrix.max_correlation = np.max(np.abs(all_correlations))
            self.current_matrix.min_correlation = np.min(all_correlations)
    
    async def _check_correlation_breaches(self) -> None:
        """Verifica violações de limite de correlação."""
        breaches = []
        
        # Check strategy correlations
        matrix = self.current_matrix.strategy_correlations
        labels = self.current_matrix.strategy_labels
        
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if abs(matrix[i, j]) > self.correlation_threshold:
                    breaches.append({
                        'type': 'strategy',
                        'item1': labels[i],
                        'item2': labels[j],
                        'correlation': matrix[i, j],
                        'timestamp': time.time()
                    })
        
        # Log breaches
        for breach in breaches:
            logger.warning(
                f"High correlation detected",
                type=breach['type'],
                items=f"{breach['item1']}-{breach['item2']}",
                correlation=f"{breach['correlation']:.2f}"
            )
        
        self.correlation_breaches.extend(breaches)
        
        # Keep only recent breaches
        cutoff_time = time.time() - 86400  # 24 hours
        self.correlation_breaches = [
            b for b in self.correlation_breaches
            if b['timestamp'] > cutoff_time
        ]
    
    async def calculate_diversification_score(
        self,
        positions: List[Dict[str, Any]]
    ) -> DiversificationScore:
        """
        Calcula score de diversificação.
        
        Args:
            positions: Posições atuais
            
        Returns:
            Score de diversificação
        """
        if not positions:
            return DiversificationScore(score=1.0)
        
        # Calculate strategy diversification
        strategy_div = self._calculate_entropy_diversification(
            [p.get('strategy', 'unknown') for p in positions]
        )
        
        # Calculate exchange diversification
        exchange_div = self._calculate_entropy_diversification(
            [p.get('exchange', 'unknown') for p in positions]
        )
        
        # Calculate symbol diversification
        symbol_div = self._calculate_entropy_diversification(
            [p.get('symbol', 'unknown') for p in positions]
        )
        
        # Calculate temporal diversification (time between entries)
        temporal_div = self._calculate_temporal_diversification(positions)
        
        # Calculate concentration risk
        concentration = await self._calculate_concentration_risk(positions)
        
        # Calculate correlation risk
        correlation_risk = min(self.current_matrix.avg_correlation / self.max_correlation, 1.0)
        
        # Combined score (weighted average)
        combined_score = (
            strategy_div * 0.3 +
            exchange_div * 0.2 +
            symbol_div * 0.2 +
            temporal_div * 0.1 +
            (1 - concentration) * 0.1 +
            (1 - correlation_risk) * 0.1
        )
        
        self.current_diversification = DiversificationScore(
            score=combined_score,
            strategy_diversification=strategy_div,
            exchange_diversification=exchange_div,
            symbol_diversification=symbol_div,
            temporal_diversification=temporal_div,
            concentration_risk=concentration,
            correlation_risk=correlation_risk
        )
        
        return self.current_diversification
    
    def _calculate_entropy_diversification(
        self,
        items: List[str]
    ) -> float:
        """
        Calcula diversificação usando entropia.
        
        Args:
            items: Lista de items
            
        Returns:
            Score de diversificação (0-1)
        """
        if not items:
            return 1.0
        
        # Count frequencies
        counts = defaultdict(int)
        for item in items:
            counts[item] += 1
        
        # Calculate entropy
        total = len(items)
        entropy = 0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize by maximum entropy
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
        
        if max_entropy > 0:
            return entropy / max_entropy
        
        return 0
    
    def _calculate_temporal_diversification(
        self,
        positions: List[Dict[str, Any]]
    ) -> float:
        """
        Calcula diversificação temporal.
        
        Args:
            positions: Posições
            
        Returns:
            Score de diversificação temporal
        """
        if len(positions) < 2:
            return 1.0
        
        # Get entry times
        entry_times = sorted([p.get('entry_time', 0) for p in positions])
        
        # Calculate time differences
        time_diffs = np.diff(entry_times)
        
        if len(time_diffs) > 0:
            # Good diversification = consistent spacing
            avg_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            
            if avg_diff > 0:
                cv = std_diff / avg_diff  # Coefficient of variation
                return max(0, 1 - cv)  # Lower CV = better diversification
        
        return 0.5
    
    async def _calculate_concentration_risk(
        self,
        positions: List[Dict[str, Any]]
    ) -> float:
        """
        Calcula risco de concentração.
        
        Args:
            positions: Posições
            
        Returns:
            Risco de concentração (0-1)
        """
        if not positions:
            return 0
        
        # Calculate position sizes
        sizes = [p.get('size', 0) for p in positions]
        total_size = sum(sizes)
        
        if total_size == 0:
            return 0
        
        # Calculate HHI
        shares = [s / total_size for s in sizes]
        hhi = sum(s**2 for s in shares)
        
        # Calculate concentration ratios
        sorted_shares = sorted(shares, reverse=True)
        
        self.concentration_risk = ConcentrationRisk(
            hhi_strategies=hhi,
            top1_concentration=sorted_shares[0] if len(sorted_shares) > 0 else 0,
            top3_concentration=sum(sorted_shares[:3]) if len(sorted_shares) >= 3 else sum(sorted_shares),
            top5_concentration=sum(sorted_shares[:5]) if len(sorted_shares) >= 5 else sum(sorted_shares)
        )
        
        # Check concentration warnings
        if self.concentration_risk.top1_concentration > 0.4:
            self.concentration_risk.concentration_warnings.append("Single position > 40%")
            self.concentration_risk.is_concentrated = True
        
        if self.concentration_risk.top3_concentration > 0.7:
            self.concentration_risk.concentration_warnings.append("Top 3 positions > 70%")
            self.concentration_risk.is_concentrated = True
        
        if hhi > 0.25:  # HHI > 2500 in traditional scale
            self.concentration_risk.concentration_warnings.append("HHI indicates high concentration")
            self.concentration_risk.is_concentrated = True
        
        return hhi
    
    def should_allow_position(
        self,
        new_position: Dict[str, Any],
        existing_positions: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Verifica se nova posição deve ser permitida.
        
        Args:
            new_position: Nova posição proposta
            existing_positions: Posições existentes
            
        Returns:
            (permitir, razão)
        """
        # Check correlation with existing positions
        new_strategy = new_position.get('strategy', '')
        
        for position in existing_positions:
            existing_strategy = position.get('strategy', '')
            
            correlation = self.current_matrix.get_correlation(
                new_strategy, existing_strategy, "strategy"
            )
            
            if correlation and abs(correlation) > self.max_correlation:
                return False, f"High correlation ({correlation:.2f}) with {existing_strategy}"
        
        # Check concentration
        total_size = sum(p.get('size', 0) for p in existing_positions)
        new_size = new_position.get('size', 0)
        
        if total_size > 0:
            new_concentration = new_size / (total_size + new_size)
            
            if new_concentration > 0.3:  # Single position > 30%
                return False, f"Position too large ({new_concentration:.1%} of portfolio)"
        
        # Check diversification impact
        test_positions = existing_positions + [new_position]
        test_score = asyncio.run(self.calculate_diversification_score(test_positions))
        
        if test_score.score < 0.3:  # Minimum diversification
            return False, f"Would reduce diversification below threshold ({test_score.score:.2f})"
        
        return True, "OK"
    
    async def _monitor_loop(self) -> None:
        """Loop de monitoramento contínuo."""
        while True:
            try:
                # Update correlation matrix
                if time.time() - self.last_update > self.update_interval:
                    await self.update_correlation_matrix()
                    
                    # Log statistics
                    if self.current_matrix.max_correlation > self.correlation_threshold:
                        logger.warning(
                            f"High correlation in portfolio",
                            max_corr=f"{self.current_matrix.max_correlation:.2f}",
                            avg_corr=f"{self.current_matrix.avg_correlation:.2f}"
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in correlation monitor: {e}")
                await asyncio.sleep(60)
    
    def get_correlation_report(self) -> Dict[str, Any]:
        """Retorna relatório de correlações."""
        return {
            'current_matrix': {
                'avg_correlation': self.current_matrix.avg_correlation,
                'max_correlation': self.current_matrix.max_correlation,
                'min_correlation': self.current_matrix.min_correlation,
                'last_update': self.last_update
            },
            'diversification': {
                'score': self.current_diversification.score,
                'risk_level': self.current_diversification.risk_level,
                'strategy_div': self.current_diversification.strategy_diversification,
                'exchange_div': self.current_diversification.exchange_diversification,
                'symbol_div': self.current_diversification.symbol_diversification
            },
            'concentration': {
                'hhi': self.concentration_risk.hhi_strategies,
                'top1': self.concentration_risk.top1_concentration,
                'top3': self.concentration_risk.top3_concentration,
                'is_concentrated': self.concentration_risk.is_concentrated,
                'warnings': self.concentration_risk.concentration_warnings
            },
            'breaches': len(self.correlation_breaches),
            'thresholds': {
                'max_correlation': self.max_correlation,
                'alert_threshold': self.correlation_threshold
            }
        }