"""
Metrics Collector - Sistema centralizado de coleta de métricas
Coleta e agrega métricas de todos os componentes do sistema
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import numpy as np

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import redis.asyncio as redis

from config.settings import settings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipos de métricas suportadas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricPoint:
    """Ponto de métrica individual."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AggregatedMetrics:
    """Métricas agregadas."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    stddev: float
    last_value: float
    timestamp: float


class MetricsCollector:
    """
    Coletor centralizado de métricas do sistema.
    Agrega e processa métricas de todos os componentes.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Inicializa o coletor de métricas.
        
        Args:
            buffer_size: Tamanho do buffer de métricas
        """
        # Storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.aggregated_metrics: Dict[str, AggregatedMetrics] = {}
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Redis for persistence
        self.redis_client: Optional[redis.Redis] = None
        self._init_redis()
        
        # Collection settings
        self.collection_interval = 1.0  # seconds
        self.aggregation_window = 60  # seconds
        self.retention_period = 86400  # 24 hours
        
        # System metrics
        self.system_metrics = {
            'latency_ms': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': [],
            'active_connections': 0,
            'error_rate': 0
        }
        
        # Business metrics
        self.business_metrics = {
            'opportunities_detected': 0,
            'trades_executed': 0,
            'total_profit_usd': 0.0,
            'success_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Component metrics
        self.component_metrics = defaultdict(dict)
        
        # Start collection
        self.is_running = False
        self._collection_task = None
        
    def _init_prometheus_metrics(self) -> None:
        """Inicializa métricas Prometheus."""
        # System metrics
        self.prom_latency = Histogram(
            'system_latency_milliseconds',
            'System latency in milliseconds',
            buckets=[1, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
        )
        
        self.prom_throughput = Gauge(
            'system_throughput_per_second',
            'System throughput per second'
        )
        
        self.prom_cpu = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.prom_memory = Gauge(
            'memory_usage_mb',
            'Memory usage in MB'
        )
        
        # Business metrics
        self.prom_opportunities = Counter(
            'arbitrage_opportunities_total',
            'Total arbitrage opportunities detected',
            ['strategy', 'exchange']
        )
        
        self.prom_trades = Counter(
            'trades_executed_total',
            'Total trades executed',
            ['strategy', 'exchange', 'symbol']
        )
        
        self.prom_profit = Gauge(
            'total_profit_usd',
            'Total profit in USD'
        )
        
        self.prom_success_rate = Gauge(
            'trade_success_rate',
            'Trade success rate percentage'
        )
        
        # Component health
        self.prom_component_health = Gauge(
            'component_health_status',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component']
        )
        
        # Error metrics
        self.prom_errors = Counter(
            'system_errors_total',
            'Total system errors',
            ['component', 'error_type']
        )
        
    async def _init_redis(self) -> None:
        """Inicializa conexão Redis para persistência."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=1,  # Use different DB for metrics
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for metrics")
        except Exception as e:
            logger.warning(f"Redis not available for metrics: {e}")
            self.redis_client = None
    
    async def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
        component: str = None
    ) -> None:
        """
        Coleta uma métrica individual.
        
        Args:
            name: Nome da métrica
            value: Valor da métrica
            metric_type: Tipo da métrica
            labels: Labels Prometheus
            component: Componente de origem
        """
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        # Add to buffer
        self.metrics_buffer[name].append(metric_point)
        
        # Update component metrics
        if component:
            self.component_metrics[component][name] = value
        
        # Update Prometheus
        await self._update_prometheus_metric(metric_point)
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_metric(metric_point)
    
    async def _update_prometheus_metric(self, metric: MetricPoint) -> None:
        """
        Atualiza métrica Prometheus.
        
        Args:
            metric: Ponto de métrica
        """
        try:
            # System metrics
            if metric.name == 'latency_ms':
                self.prom_latency.observe(metric.value)
            elif metric.name == 'throughput':
                self.prom_throughput.set(metric.value)
            elif metric.name == 'cpu_usage':
                self.prom_cpu.set(metric.value)
            elif metric.name == 'memory_usage':
                self.prom_memory.set(metric.value)
            
            # Business metrics
            elif metric.name == 'opportunities_detected':
                labels = metric.labels or {}
                self.prom_opportunities.labels(
                    strategy=labels.get('strategy', 'unknown'),
                    exchange=labels.get('exchange', 'unknown')
                ).inc()
            elif metric.name == 'trades_executed':
                labels = metric.labels or {}
                self.prom_trades.labels(
                    strategy=labels.get('strategy', 'unknown'),
                    exchange=labels.get('exchange', 'unknown'),
                    symbol=labels.get('symbol', 'unknown')
                ).inc()
            elif metric.name == 'total_profit_usd':
                self.prom_profit.set(metric.value)
            elif metric.name == 'success_rate':
                self.prom_success_rate.set(metric.value)
            
            # Component health
            elif metric.name == 'component_health':
                labels = metric.labels or {}
                self.prom_component_health.labels(
                    component=labels.get('component', 'unknown')
                ).set(metric.value)
            
            # Errors
            elif metric.name == 'error':
                labels = metric.labels or {}
                self.prom_errors.labels(
                    component=labels.get('component', 'unknown'),
                    error_type=labels.get('error_type', 'unknown')
                ).inc()
                
        except Exception as e:
            logger.error(f"Failed to update Prometheus metric: {e}")
    
    async def _persist_metric(self, metric: MetricPoint) -> None:
        """
        Persiste métrica no Redis.
        
        Args:
            metric: Ponto de métrica
        """
        try:
            # Create key
            key = f"metric:{metric.name}:{int(metric.timestamp)}"
            
            # Store metric
            await self.redis_client.hset(
                key,
                mapping={
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'type': metric.metric_type.value,
                    'labels': str(metric.labels)
                }
            )
            
            # Set expiration
            await self.redis_client.expire(key, self.retention_period)
            
        except Exception as e:
            logger.debug(f"Failed to persist metric to Redis: {e}")
    
    async def aggregate_metrics(self, metric_name: str, window: int = None) -> Optional[AggregatedMetrics]:
        """
        Agrega métricas para uma janela de tempo.
        
        Args:
            metric_name: Nome da métrica
            window: Janela de agregação em segundos
            
        Returns:
            Métricas agregadas
        """
        window = window or self.aggregation_window
        
        if metric_name not in self.metrics_buffer:
            return None
        
        # Get recent metrics
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_buffer[metric_name]
            if current_time - m.timestamp <= window
        ]
        
        if not recent_metrics:
            return None
        
        values = [m.value for m in recent_metrics]
        
        return AggregatedMetrics(
            name=metric_name,
            count=len(values),
            sum=np.sum(values),
            min=np.min(values),
            max=np.max(values),
            avg=np.mean(values),
            p50=np.percentile(values, 50),
            p95=np.percentile(values, 95),
            p99=np.percentile(values, 99),
            stddev=np.std(values),
            last_value=values[-1],
            timestamp=current_time
        )
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas do sistema.
        
        Returns:
            Dicionário com métricas do sistema
        """
        metrics = {}
        
        # Aggregate recent metrics
        for metric_name in ['latency_ms', 'throughput', 'cpu_usage', 'memory_usage']:
            agg = await self.aggregate_metrics(metric_name, window=60)
            if agg:
                metrics[metric_name] = {
                    'current': agg.last_value,
                    'avg': agg.avg,
                    'min': agg.min,
                    'max': agg.max,
                    'p95': agg.p95,
                    'p99': agg.p99
                }
        
        # Add current values
        metrics['active_connections'] = self.system_metrics['active_connections']
        metrics['error_rate'] = await self._calculate_error_rate()
        
        return metrics
    
    async def get_business_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas de negócio.
        
        Returns:
            Dicionário com métricas de negócio
        """
        metrics = {}
        
        # Get aggregated business metrics
        for metric_name in ['opportunities_detected', 'trades_executed', 'total_profit_usd', 
                          'success_rate', 'sharpe_ratio', 'max_drawdown']:
            agg = await self.aggregate_metrics(metric_name, window=3600)  # 1 hour
            if agg:
                metrics[metric_name] = {
                    'current': agg.last_value,
                    'total': agg.sum if metric_name.endswith('_detected') or metric_name.endswith('_executed') else agg.last_value,
                    'avg': agg.avg,
                    'max': agg.max
                }
            else:
                metrics[metric_name] = self.business_metrics.get(metric_name, 0)
        
        return metrics
    
    async def get_component_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtém saúde dos componentes.
        
        Returns:
            Dicionário com status de saúde
        """
        health = {}
        
        for component, metrics in self.component_metrics.items():
            health[component] = {
                'status': 'healthy' if metrics.get('health', 0) > 0.8 else 'degraded',
                'metrics': metrics,
                'last_update': metrics.get('last_update', 0)
            }
        
        return health
    
    async def _calculate_error_rate(self) -> float:
        """
        Calcula taxa de erro.
        
        Returns:
            Taxa de erro percentual
        """
        # Get error metrics from last hour
        errors = await self.aggregate_metrics('error', window=3600)
        total_requests = await self.aggregate_metrics('requests', window=3600)
        
        if not total_requests or total_requests.count == 0:
            return 0.0
        
        error_count = errors.count if errors else 0
        return (error_count / total_requests.count) * 100
    
    async def collect_system_stats(self) -> None:
        """Coleta estatísticas do sistema operacional."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await self.collect_metric('cpu_usage', cpu_percent, MetricType.GAUGE)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            await self.collect_metric('memory_usage', memory_mb, MetricType.GAUGE)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self.collect_metric('disk_read_mb', disk_io.read_bytes / (1024 * 1024), MetricType.COUNTER)
                await self.collect_metric('disk_write_mb', disk_io.write_bytes / (1024 * 1024), MetricType.COUNTER)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            await self.collect_metric('network_sent_mb', net_io.bytes_sent / (1024 * 1024), MetricType.COUNTER)
            await self.collect_metric('network_recv_mb', net_io.bytes_recv / (1024 * 1024), MetricType.COUNTER)
            
        except ImportError:
            logger.debug("psutil not available for system stats")
        except Exception as e:
            logger.error(f"Failed to collect system stats: {e}")
    
    async def _collection_loop(self) -> None:
        """Loop de coleta de métricas."""
        logger.info("Starting metrics collection loop")
        
        while self.is_running:
            try:
                # Collect system stats
                await self.collect_system_stats()
                
                # Clean old metrics
                await self._cleanup_old_metrics()
                
                # Aggregate metrics
                for metric_name in self.metrics_buffer.keys():
                    agg = await self.aggregate_metrics(metric_name)
                    if agg:
                        self.aggregated_metrics[metric_name] = agg
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_metrics(self) -> None:
        """Remove métricas antigas do buffer."""
        current_time = time.time()
        
        for metric_name, buffer in self.metrics_buffer.items():
            # Remove metrics older than retention period
            while buffer and (current_time - buffer[0].timestamp) > self.retention_period:
                buffer.popleft()
    
    async def start(self) -> None:
        """Inicia coleta de métricas."""
        if not self.is_running:
            self.is_running = True
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Metrics collector started")
    
    async def stop(self) -> None:
        """Para coleta de métricas."""
        self.is_running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Metrics collector stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtém resumo de todas as métricas.
        
        Returns:
            Dicionário com resumo das métricas
        """
        return {
            'timestamp': time.time(),
            'metrics_collected': len(self.metrics_buffer),
            'total_points': sum(len(buffer) for buffer in self.metrics_buffer.values()),
            'aggregated_metrics': len(self.aggregated_metrics),
            'components_monitored': len(self.component_metrics),
            'buffer_usage': {
                name: len(buffer) for name, buffer in self.metrics_buffer.items()
            }
        }


# Global instance
metrics_collector = MetricsCollector()


# Convenience functions
async def collect_metric(name: str, value: float, **kwargs) -> None:
    """Função de conveniência para coletar métrica."""
    await metrics_collector.collect_metric(name, value, **kwargs)


async def get_system_metrics() -> Dict[str, Any]:
    """Função de conveniência para obter métricas do sistema."""
    return await metrics_collector.get_system_metrics()


async def get_business_metrics() -> Dict[str, Any]:
    """Função de conveniência para obter métricas de negócio."""
    return await metrics_collector.get_business_metrics()