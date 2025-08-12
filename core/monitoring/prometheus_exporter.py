"""
Prometheus Exporter - Exportador de métricas para Prometheus
Expõe métricas do sistema em formato Prometheus
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, Optional, Any, List
from enum import Enum
import logging

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, push_to_gateway
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily
from aiohttp import web

from config.settings import settings

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Exportador de métricas para Prometheus.
    Expõe todas as métricas do sistema em formato Prometheus.
    """
    
    def __init__(self, port: int = 9090):
        """
        Inicializa o exportador Prometheus.
        
        Args:
            port: Porta para expor métricas
        """
        self.port = port
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_system_metrics()
        self._init_business_metrics()
        self._init_strategy_metrics()
        self._init_exchange_metrics()
        self._init_ml_metrics()
        self._init_risk_metrics()
        
        # HTTP server
        self.app = web.Application()
        self.runner = None
        
        # Pushgateway config (optional)
        self.pushgateway_url = getattr(settings, 'prometheus_pushgateway_url', None)
        self.push_interval = 30  # seconds
        
        # Export state
        self.is_running = False
        self._push_task = None
        
    def _init_system_metrics(self) -> None:
        """Inicializa métricas do sistema."""
        # Latency metrics
        self.latency_histogram = Histogram(
            'arbitrage_system_latency_seconds',
            'System latency in seconds',
            ['operation', 'component'],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
            registry=self.registry
        )
        
        # Throughput metrics
        self.throughput_gauge = Gauge(
            'arbitrage_system_throughput_ops_per_second',
            'System throughput in operations per second',
            ['component'],
            registry=self.registry
        )
        
        # Resource usage
        self.cpu_usage = Gauge(
            'arbitrage_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'arbitrage_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_io_read = Counter(
            'arbitrage_disk_read_bytes_total',
            'Total disk read bytes',
            registry=self.registry
        )
        
        self.disk_io_write = Counter(
            'arbitrage_disk_write_bytes_total',
            'Total disk write bytes',
            registry=self.registry
        )
        
        # Network metrics
        self.network_sent = Counter(
            'arbitrage_network_sent_bytes_total',
            'Total network bytes sent',
            registry=self.registry
        )
        
        self.network_received = Counter(
            'arbitrage_network_received_bytes_total',
            'Total network bytes received',
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'arbitrage_active_connections',
            'Number of active connections',
            ['type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'arbitrage_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'arbitrage_error_rate_percent',
            'Error rate percentage',
            registry=self.registry
        )
    
    def _init_business_metrics(self) -> None:
        """Inicializa métricas de negócio."""
        # Trading metrics
        self.opportunities_detected = Counter(
            'arbitrage_opportunities_detected_total',
            'Total arbitrage opportunities detected',
            ['strategy', 'exchange', 'symbol'],
            registry=self.registry
        )
        
        self.opportunities_executed = Counter(
            'arbitrage_opportunities_executed_total',
            'Total arbitrage opportunities executed',
            ['strategy', 'exchange', 'symbol'],
            registry=self.registry
        )
        
        self.trades_total = Counter(
            'arbitrage_trades_total',
            'Total number of trades',
            ['strategy', 'exchange', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.trade_volume = Summary(
            'arbitrage_trade_volume_usd',
            'Trade volume in USD',
            ['strategy', 'exchange', 'symbol'],
            registry=self.registry
        )
        
        # Profit metrics
        self.profit_total = Gauge(
            'arbitrage_profit_total_usd',
            'Total profit in USD',
            registry=self.registry
        )
        
        self.profit_rate = Gauge(
            'arbitrage_profit_rate_usd_per_hour',
            'Profit rate in USD per hour',
            registry=self.registry
        )
        
        self.roi = Gauge(
            'arbitrage_roi_percent',
            'Return on investment percentage',
            registry=self.registry
        )
        
        # Success metrics
        self.success_rate = Gauge(
            'arbitrage_success_rate_percent',
            'Trade success rate percentage',
            ['strategy'],
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'arbitrage_win_rate_percent',
            'Profitable trades percentage',
            ['strategy'],
            registry=self.registry
        )
    
    def _init_strategy_metrics(self) -> None:
        """Inicializa métricas por estratégia."""
        # Spatial arbitrage
        self.spatial_opportunities = Counter(
            'arbitrage_spatial_opportunities_total',
            'Total spatial arbitrage opportunities',
            ['buy_exchange', 'sell_exchange', 'symbol'],
            registry=self.registry
        )
        
        self.spatial_spread = Histogram(
            'arbitrage_spatial_spread_percent',
            'Spatial arbitrage spread percentage',
            buckets=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
            registry=self.registry
        )
        
        # Statistical arbitrage
        self.statistical_zscore = Gauge(
            'arbitrage_statistical_zscore',
            'Statistical arbitrage z-score',
            ['pair'],
            registry=self.registry
        )
        
        self.statistical_half_life = Gauge(
            'arbitrage_statistical_half_life_hours',
            'Mean reversion half-life in hours',
            ['pair'],
            registry=self.registry
        )
        
        # Triangular arbitrage
        self.triangular_cycles = Counter(
            'arbitrage_triangular_cycles_total',
            'Total triangular arbitrage cycles detected',
            ['exchange', 'path'],
            registry=self.registry
        )
        
        self.triangular_profit = Histogram(
            'arbitrage_triangular_profit_percent',
            'Triangular arbitrage profit percentage',
            buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0],
            registry=self.registry
        )
    
    def _init_exchange_metrics(self) -> None:
        """Inicializa métricas por exchange."""
        self.exchange_status = Gauge(
            'arbitrage_exchange_status',
            'Exchange connection status (1=connected, 0=disconnected)',
            ['exchange'],
            registry=self.registry
        )
        
        self.exchange_latency = Histogram(
            'arbitrage_exchange_latency_seconds',
            'Exchange API latency',
            ['exchange', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.exchange_rate_limit = Gauge(
            'arbitrage_exchange_rate_limit_remaining',
            'Remaining rate limit for exchange',
            ['exchange'],
            registry=self.registry
        )
        
        self.orderbook_depth = Gauge(
            'arbitrage_orderbook_depth_usd',
            'Orderbook depth in USD',
            ['exchange', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.orderbook_spread = Gauge(
            'arbitrage_orderbook_spread_bps',
            'Orderbook spread in basis points',
            ['exchange', 'symbol'],
            registry=self.registry
        )
    
    def _init_ml_metrics(self) -> None:
        """Inicializa métricas de machine learning."""
        self.ml_inference_time = Histogram(
            'arbitrage_ml_inference_seconds',
            'ML model inference time',
            ['model'],
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
            registry=self.registry
        )
        
        self.ml_accuracy = Gauge(
            'arbitrage_ml_accuracy_percent',
            'ML model accuracy percentage',
            ['model'],
            registry=self.registry
        )
        
        self.ml_confidence = Histogram(
            'arbitrage_ml_confidence',
            'ML prediction confidence',
            ['model'],
            buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        self.feature_extraction_time = Histogram(
            'arbitrage_feature_extraction_seconds',
            'Feature extraction time',
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01],
            registry=self.registry
        )
        
        self.feature_count = Gauge(
            'arbitrage_feature_count',
            'Number of features extracted',
            registry=self.registry
        )
        
        self.xai_explanation_time = Histogram(
            'arbitrage_xai_explanation_seconds',
            'XAI explanation generation time',
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
            registry=self.registry
        )
    
    def _init_risk_metrics(self) -> None:
        """Inicializa métricas de risco."""
        self.position_size = Gauge(
            'arbitrage_position_size_usd',
            'Current position size in USD',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.exposure = Gauge(
            'arbitrage_exposure_percent',
            'Portfolio exposure percentage',
            ['exchange'],
            registry=self.registry
        )
        
        self.var_95 = Gauge(
            'arbitrage_var_95_usd',
            'Value at Risk (95% confidence) in USD',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'arbitrage_sharpe_ratio',
            'Sharpe ratio',
            ['strategy'],
            registry=self.registry
        )
        
        self.sortino_ratio = Gauge(
            'arbitrage_sortino_ratio',
            'Sortino ratio',
            ['strategy'],
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'arbitrage_max_drawdown_percent',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.correlation = Gauge(
            'arbitrage_strategy_correlation',
            'Correlation between strategies',
            ['strategy1', 'strategy2'],
            registry=self.registry
        )
    
    def update_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Atualiza uma métrica.
        
        Args:
            metric_name: Nome da métrica
            value: Valor da métrica
            labels: Labels da métrica
        """
        try:
            metric = getattr(self, metric_name, None)
            if not metric:
                logger.debug(f"Metric {metric_name} not found")
                return
            
            if labels:
                if isinstance(metric, Counter):
                    metric.labels(**labels).inc(value)
                elif isinstance(metric, Gauge):
                    metric.labels(**labels).set(value)
                elif isinstance(metric, Histogram):
                    metric.labels(**labels).observe(value)
                elif isinstance(metric, Summary):
                    metric.labels(**labels).observe(value)
            else:
                if isinstance(metric, Counter):
                    metric.inc(value)
                elif isinstance(metric, Gauge):
                    metric.set(value)
                elif isinstance(metric, Histogram):
                    metric.observe(value)
                elif isinstance(metric, Summary):
                    metric.observe(value)
                    
        except Exception as e:
            logger.error(f"Failed to update metric {metric_name}: {e}")
    
    async def collect_and_export(self) -> None:
        """Coleta e exporta métricas."""
        try:
            # Import here to avoid circular dependency
            from core.monitoring.metrics_collector import metrics_collector
            from core.monitoring.performance_tracker import performance_tracker
            from core.monitoring.alert_manager import alert_manager
            
            # Get current metrics
            system_metrics = await metrics_collector.get_system_metrics()
            business_metrics = await metrics_collector.get_business_metrics()
            
            # Update Prometheus metrics
            await self._update_prometheus_metrics(system_metrics, business_metrics)
            
            # Push to gateway if configured
            if self.pushgateway_url:
                await self._push_to_gateway()
                
        except Exception as e:
            logger.error(f"Failed to collect and export metrics: {e}")
    
    async def _update_prometheus_metrics(
        self,
        system_metrics: Dict[str, Any],
        business_metrics: Dict[str, Any]
    ) -> None:
        """
        Atualiza métricas Prometheus com dados coletados.
        
        Args:
            system_metrics: Métricas do sistema
            business_metrics: Métricas de negócio
        """
        # Update system metrics
        if 'latency_ms' in system_metrics:
            latency = system_metrics['latency_ms']
            if isinstance(latency, dict):
                self.latency_histogram.labels(
                    operation='system',
                    component='overall'
                ).observe(latency.get('current', 0) / 1000)  # Convert to seconds
        
        if 'throughput' in system_metrics:
            throughput = system_metrics['throughput']
            if isinstance(throughput, dict):
                self.throughput_gauge.labels(component='overall').set(
                    throughput.get('current', 0)
                )
        
        if 'cpu_usage' in system_metrics:
            cpu = system_metrics['cpu_usage']
            if isinstance(cpu, dict):
                self.cpu_usage.set(cpu.get('current', 0))
        
        if 'memory_usage' in system_metrics:
            memory = system_metrics['memory_usage']
            if isinstance(memory, dict):
                self.memory_usage.set(memory.get('current', 0) * 1024 * 1024)  # Convert to bytes
        
        if 'error_rate' in system_metrics:
            self.error_rate.set(system_metrics['error_rate'])
        
        # Update business metrics
        if 'total_profit_usd' in business_metrics:
            profit = business_metrics['total_profit_usd']
            if isinstance(profit, dict):
                self.profit_total.set(profit.get('total', 0))
        
        if 'success_rate' in business_metrics:
            success = business_metrics['success_rate']
            if isinstance(success, dict):
                self.success_rate.labels(strategy='overall').set(
                    success.get('current', 0)
                )
        
        if 'sharpe_ratio' in business_metrics:
            sharpe = business_metrics['sharpe_ratio']
            if isinstance(sharpe, dict):
                self.sharpe_ratio.labels(strategy='overall').set(
                    sharpe.get('current', 0)
                )
    
    async def _push_to_gateway(self) -> None:
        """Envia métricas para Prometheus Pushgateway."""
        if not self.pushgateway_url:
            return
        
        try:
            push_to_gateway(
                self.pushgateway_url,
                job='crypto_arbitrage',
                registry=self.registry
            )
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")
    
    async def metrics_handler(self, request: web.Request) -> web.Response:
        """
        Handler HTTP para endpoint de métricas.
        
        Args:
            request: Requisição HTTP
            
        Returns:
            Resposta com métricas Prometheus
        """
        # Collect latest metrics
        await self.collect_and_export()
        
        # Generate metrics
        metrics = generate_latest(self.registry)
        
        return web.Response(
            body=metrics,
            content_type=CONTENT_TYPE_LATEST
        )
    
    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Handler para health check.
        
        Args:
            request: Requisição HTTP
            
        Returns:
            Status de saúde
        """
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time()
        })
    
    async def _push_loop(self) -> None:
        """Loop para push de métricas."""
        while self.is_running:
            try:
                await self.collect_and_export()
                await asyncio.sleep(self.push_interval)
            except Exception as e:
                logger.error(f"Error in push loop: {e}")
                await asyncio.sleep(60)
    
    async def start(self) -> None:
        """Inicia o exportador Prometheus."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Setup routes
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/health', self.health_handler)
        
        # Start HTTP server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Prometheus exporter started on port {self.port}")
        
        # Start push loop if configured
        if self.pushgateway_url:
            self._push_task = asyncio.create_task(self._push_loop())
    
    async def stop(self) -> None:
        """Para o exportador Prometheus."""
        self.is_running = False
        
        # Stop push task
        if self._push_task:
            self._push_task.cancel()
            try:
                await self._push_task
            except asyncio.CancelledError:
                pass
        
        # Stop HTTP server
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Prometheus exporter stopped")


# Global instance
prometheus_exporter = PrometheusExporter(port=settings.prometheus_port if hasattr(settings, 'prometheus_port') else 9090)