"""
Performance Tracker - Sistema de tracking de performance em tempo real
Monitora latência, throughput, success rate e outras métricas críticas
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging
import numpy as np

from core.monitoring.metrics_collector import metrics_collector, MetricType

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Níveis de performance."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance em um momento."""
    timestamp: float
    latency_ms: float
    throughput: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    profit_rate: float  # USD per hour
    opportunities_per_minute: float
    level: PerformanceLevel


@dataclass
class PerformanceTarget:
    """Alvos de performance."""
    metric: str
    target: float
    warning: float
    critical: float
    unit: str = ""
    higher_is_better: bool = True


class PerformanceTracker:
    """
    Tracker de performance em tempo real.
    Monitora e analisa métricas críticas do sistema.
    """
    
    def __init__(self):
        """Inicializa o tracker de performance."""
        # Performance history
        self.history_size = 1000
        self.performance_history: deque = deque(maxlen=self.history_size)
        
        # Current performance
        self.current_performance: Optional[PerformanceSnapshot] = None
        
        # Performance targets (from settings)
        self.targets = self._init_performance_targets()
        
        # Statistics
        self.stats_window = 300  # 5 minutes
        self.performance_stats = {
            'avg_latency': 0,
            'p95_latency': 0,
            'p99_latency': 0,
            'avg_throughput': 0,
            'min_throughput': 0,
            'max_throughput': 0,
            'avg_success_rate': 0,
            'total_profit': 0,
            'profit_per_hour': 0,
            'uptime': 0,
            'availability': 100.0
        }
        
        # Tracking intervals
        self.tracking_interval = 1.0  # seconds
        self.stats_update_interval = 10.0  # seconds
        
        # Performance callbacks
        self.performance_callbacks: List[Callable] = []
        
        # Tracking state
        self.is_tracking = False
        self._tracking_task = None
        self._stats_task = None
        
        # Start time
        self.start_time = time.time()
        
    def _init_performance_targets(self) -> Dict[str, PerformanceTarget]:
        """Inicializa alvos de performance."""
        return {
            'latency_ms': PerformanceTarget(
                metric='latency_ms',
                target=10,
                warning=20,
                critical=30,
                unit='ms',
                higher_is_better=False
            ),
            'throughput': PerformanceTarget(
                metric='throughput',
                target=10000,
                warning=5000,
                critical=1000,
                unit='ops/sec',
                higher_is_better=True
            ),
            'success_rate': PerformanceTarget(
                metric='success_rate',
                target=95,
                warning=80,
                critical=60,
                unit='%',
                higher_is_better=True
            ),
            'error_rate': PerformanceTarget(
                metric='error_rate',
                target=1,
                warning=5,
                critical=10,
                unit='%',
                higher_is_better=False
            ),
            'cpu_usage': PerformanceTarget(
                metric='cpu_usage',
                target=50,
                warning=70,
                critical=90,
                unit='%',
                higher_is_better=False
            ),
            'memory_usage': PerformanceTarget(
                metric='memory_usage',
                target=2048,
                warning=3072,
                critical=4096,
                unit='MB',
                higher_is_better=False
            ),
            'profit_rate': PerformanceTarget(
                metric='profit_rate',
                target=100,
                warning=50,
                critical=0,
                unit='USD/hour',
                higher_is_better=True
            ),
            'opportunities_per_minute': PerformanceTarget(
                metric='opportunities_per_minute',
                target=10,
                warning=5,
                critical=1,
                unit='opp/min',
                higher_is_better=True
            )
        }
    
    async def track_latency(self, operation: str, latency_ms: float) -> None:
        """
        Registra latência de uma operação.
        
        Args:
            operation: Nome da operação
            latency_ms: Latência em millisegundos
        """
        await metrics_collector.collect_metric(
            f'latency_{operation}',
            latency_ms,
            MetricType.HISTOGRAM,
            labels={'operation': operation}
        )
        
        # Update overall latency
        await metrics_collector.collect_metric('latency_ms', latency_ms, MetricType.HISTOGRAM)
        
        # Check if exceeds threshold
        target = self.targets['latency_ms']
        if latency_ms > target.critical:
            logger.warning(f"Critical latency for {operation}: {latency_ms:.2f}ms")
        elif latency_ms > target.warning:
            logger.debug(f"High latency for {operation}: {latency_ms:.2f}ms")
    
    async def track_throughput(self, operations: int, duration: float) -> None:
        """
        Registra throughput.
        
        Args:
            operations: Número de operações
            duration: Duração em segundos
        """
        throughput = operations / duration if duration > 0 else 0
        
        await metrics_collector.collect_metric('throughput', throughput, MetricType.GAUGE)
        
        # Check performance level
        target = self.targets['throughput']
        if throughput < target.critical:
            logger.warning(f"Critical throughput: {throughput:.0f} ops/sec")
    
    async def track_success(self, success: bool, operation: str = None) -> None:
        """
        Registra sucesso/falha de operação.
        
        Args:
            success: Se a operação foi bem-sucedida
            operation: Nome da operação
        """
        labels = {'operation': operation} if operation else {}
        
        if success:
            await metrics_collector.collect_metric(
                'successful_operations',
                1,
                MetricType.COUNTER,
                labels=labels
            )
        else:
            await metrics_collector.collect_metric(
                'failed_operations',
                1,
                MetricType.COUNTER,
                labels=labels
            )
    
    async def track_profit(self, profit_usd: float, strategy: str = None) -> None:
        """
        Registra lucro.
        
        Args:
            profit_usd: Lucro em USD
            strategy: Estratégia utilizada
        """
        labels = {'strategy': strategy} if strategy else {}
        
        await metrics_collector.collect_metric(
            'profit_usd',
            profit_usd,
            MetricType.GAUGE,
            labels=labels
        )
        
        # Update total profit
        current_total = self.performance_stats.get('total_profit', 0)
        self.performance_stats['total_profit'] = current_total + profit_usd
    
    async def track_opportunity(self, strategy: str, exchange: str, profitable: bool) -> None:
        """
        Registra oportunidade de arbitragem.
        
        Args:
            strategy: Estratégia utilizada
            exchange: Exchange
            profitable: Se era lucrativa
        """
        await metrics_collector.collect_metric(
            'opportunities_detected',
            1,
            MetricType.COUNTER,
            labels={
                'strategy': strategy,
                'exchange': exchange,
                'profitable': str(profitable)
            }
        )
    
    def evaluate_performance_level(self, snapshot: PerformanceSnapshot) -> PerformanceLevel:
        """
        Avalia nível de performance baseado no snapshot.
        
        Args:
            snapshot: Snapshot de performance
            
        Returns:
            Nível de performance
        """
        critical_count = 0
        warning_count = 0
        
        # Check each metric against targets
        for metric_name, target in self.targets.items():
            value = getattr(snapshot, metric_name, None)
            if value is None:
                continue
            
            if target.higher_is_better:
                if value < target.critical:
                    critical_count += 1
                elif value < target.warning:
                    warning_count += 1
            else:
                if value > target.critical:
                    critical_count += 1
                elif value > target.warning:
                    warning_count += 1
        
        # Determine overall level
        if critical_count >= 2:
            return PerformanceLevel.CRITICAL
        elif critical_count >= 1:
            return PerformanceLevel.DEGRADED
        elif warning_count >= 2:
            return PerformanceLevel.ACCEPTABLE
        elif warning_count >= 1:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.EXCELLENT
    
    async def capture_snapshot(self) -> PerformanceSnapshot:
        """
        Captura snapshot atual de performance.
        
        Returns:
            Snapshot de performance
        """
        # Get current metrics
        system_metrics = await metrics_collector.get_system_metrics()
        business_metrics = await metrics_collector.get_business_metrics()
        
        # Extract values
        latency = system_metrics.get('latency_ms', {}).get('current', 0)
        throughput = system_metrics.get('throughput', {}).get('current', 0)
        cpu = system_metrics.get('cpu_usage', {}).get('current', 0)
        memory = system_metrics.get('memory_usage', {}).get('current', 0)
        connections = system_metrics.get('active_connections', 0)
        error_rate = system_metrics.get('error_rate', 0)
        
        # Calculate success rate
        trades_executed = business_metrics.get('trades_executed', {})
        success_rate = business_metrics.get('success_rate', {}).get('current', 0)
        
        # Calculate profit rate (per hour)
        total_profit = business_metrics.get('total_profit_usd', {}).get('total', 0)
        uptime_hours = (time.time() - self.start_time) / 3600
        profit_rate = total_profit / uptime_hours if uptime_hours > 0 else 0
        
        # Calculate opportunities per minute
        opportunities = business_metrics.get('opportunities_detected', {}).get('total', 0)
        uptime_minutes = uptime_hours * 60
        opp_per_minute = opportunities / uptime_minutes if uptime_minutes > 0 else 0
        
        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_ms=latency,
            throughput=throughput,
            success_rate=success_rate,
            error_rate=error_rate,
            cpu_usage=cpu,
            memory_usage=memory,
            active_connections=connections,
            profit_rate=profit_rate,
            opportunities_per_minute=opp_per_minute,
            level=PerformanceLevel.GOOD  # Will be evaluated
        )
        
        # Evaluate performance level
        snapshot.level = self.evaluate_performance_level(snapshot)
        
        return snapshot
    
    async def _tracking_loop(self) -> None:
        """Loop de tracking de performance."""
        logger.info("Starting performance tracking loop")
        
        while self.is_tracking:
            try:
                # Capture snapshot
                snapshot = await self.capture_snapshot()
                
                # Store snapshot
                self.performance_history.append(snapshot)
                self.current_performance = snapshot
                
                # Log if performance degraded
                if snapshot.level in [PerformanceLevel.DEGRADED, PerformanceLevel.CRITICAL]:
                    logger.warning(f"Performance {snapshot.level.value}: latency={snapshot.latency_ms:.1f}ms, "
                                 f"success_rate={snapshot.success_rate:.1f}%")
                
                # Execute callbacks
                for callback in self.performance_callbacks:
                    try:
                        await callback(snapshot)
                    except Exception as e:
                        logger.error(f"Error in performance callback: {e}")
                
                # Wait for next tracking
                await asyncio.sleep(self.tracking_interval)
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(5)
    
    async def _stats_update_loop(self) -> None:
        """Loop de atualização de estatísticas."""
        logger.info("Starting stats update loop")
        
        while self.is_tracking:
            try:
                await self._update_statistics()
                await asyncio.sleep(self.stats_update_interval)
            except Exception as e:
                logger.error(f"Error updating statistics: {e}")
                await asyncio.sleep(10)
    
    async def _update_statistics(self) -> None:
        """Atualiza estatísticas de performance."""
        if not self.performance_history:
            return
        
        # Get recent snapshots
        current_time = time.time()
        recent_snapshots = [
            s for s in self.performance_history
            if current_time - s.timestamp <= self.stats_window
        ]
        
        if not recent_snapshots:
            return
        
        # Calculate statistics
        latencies = [s.latency_ms for s in recent_snapshots]
        throughputs = [s.throughput for s in recent_snapshots]
        success_rates = [s.success_rate for s in recent_snapshots]
        
        self.performance_stats.update({
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'avg_throughput': np.mean(throughputs),
            'min_throughput': np.min(throughputs),
            'max_throughput': np.max(throughputs),
            'avg_success_rate': np.mean(success_rates),
            'uptime': time.time() - self.start_time,
            'profit_per_hour': recent_snapshots[-1].profit_rate if recent_snapshots else 0
        })
        
        # Calculate availability
        healthy_count = sum(
            1 for s in recent_snapshots
            if s.level not in [PerformanceLevel.DEGRADED, PerformanceLevel.CRITICAL]
        )
        self.performance_stats['availability'] = (healthy_count / len(recent_snapshots)) * 100
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Gera relatório de performance.
        
        Returns:
            Relatório detalhado de performance
        """
        if not self.current_performance:
            return {
                'status': 'no_data',
                'message': 'No performance data available'
            }
        
        return {
            'timestamp': self.current_performance.timestamp,
            'level': self.current_performance.level.value,
            'current': {
                'latency_ms': self.current_performance.latency_ms,
                'throughput': self.current_performance.throughput,
                'success_rate': self.current_performance.success_rate,
                'error_rate': self.current_performance.error_rate,
                'cpu_usage': self.current_performance.cpu_usage,
                'memory_usage': self.current_performance.memory_usage,
                'profit_rate': self.current_performance.profit_rate,
                'opportunities_per_minute': self.current_performance.opportunities_per_minute
            },
            'statistics': self.performance_stats,
            'targets': {
                name: {
                    'target': t.target,
                    'warning': t.warning,
                    'critical': t.critical,
                    'unit': t.unit
                }
                for name, t in self.targets.items()
            },
            'history_size': len(self.performance_history)
        }
    
    def get_performance_trends(self, window: int = 3600) -> Dict[str, List[float]]:
        """
        Obtém tendências de performance.
        
        Args:
            window: Janela em segundos
            
        Returns:
            Séries temporais de métricas
        """
        current_time = time.time()
        recent = [
            s for s in self.performance_history
            if current_time - s.timestamp <= window
        ]
        
        if not recent:
            return {}
        
        return {
            'timestamps': [s.timestamp for s in recent],
            'latency_ms': [s.latency_ms for s in recent],
            'throughput': [s.throughput for s in recent],
            'success_rate': [s.success_rate for s in recent],
            'error_rate': [s.error_rate for s in recent],
            'profit_rate': [s.profit_rate for s in recent],
            'level': [s.level.value for s in recent]
        }
    
    def register_callback(self, callback: Callable) -> None:
        """
        Registra callback para mudanças de performance.
        
        Args:
            callback: Função callback async
        """
        self.performance_callbacks.append(callback)
    
    async def start(self) -> None:
        """Inicia tracking de performance."""
        if not self.is_tracking:
            self.is_tracking = True
            self._tracking_task = asyncio.create_task(self._tracking_loop())
            self._stats_task = asyncio.create_task(self._stats_update_loop())
            logger.info("Performance tracker started")
    
    async def stop(self) -> None:
        """Para tracking de performance."""
        self.is_tracking = False
        
        tasks = [self._tracking_task, self._stats_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Performance tracker stopped")


# Global instance
performance_tracker = PerformanceTracker()


# Context manager for performance tracking
class track_performance:
    """Context manager para tracking de performance."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self.start_time) * 1000
        await performance_tracker.track_latency(self.operation, duration)
        
        # Track success/failure
        success = exc_type is None
        await performance_tracker.track_success(success, self.operation)