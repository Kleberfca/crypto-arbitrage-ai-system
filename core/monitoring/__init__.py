"""
Monitoring Module - Sistema completo de monitoramento e métricas
Coleta, tracking, alertas e exportação para Prometheus
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from core.monitoring.metrics_collector import (
    MetricsCollector,
    MetricType,
    MetricPoint,
    AggregatedMetrics,
    metrics_collector,
    collect_metric,
    get_system_metrics,
    get_business_metrics
)

from core.monitoring.performance_tracker import (
    PerformanceTracker,
    PerformanceLevel,
    PerformanceSnapshot,
    PerformanceTarget,
    performance_tracker,
    track_performance
)

from core.monitoring.alert_manager import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertCategory,
    alert_manager
)

from core.monitoring.prometheus_exporter import (
    PrometheusExporter,
    prometheus_exporter
)

__all__ = [
    # Metrics Collector
    'MetricsCollector',
    'MetricType',
    'MetricPoint',
    'AggregatedMetrics',
    'metrics_collector',
    'collect_metric',
    'get_system_metrics',
    'get_business_metrics',
    
    # Performance Tracker
    'PerformanceTracker',
    'PerformanceLevel',
    'PerformanceSnapshot',
    'PerformanceTarget',
    'performance_tracker',
    'track_performance',
    
    # Alert Manager
    'AlertManager',
    'Alert',
    'AlertRule',
    'AlertSeverity',
    'AlertCategory',
    'alert_manager',
    
    # Prometheus Exporter
    'PrometheusExporter',
    'prometheus_exporter'
]


# Convenience function to start all monitoring services
async def start_monitoring():
    """Inicia todos os serviços de monitoramento."""
    await metrics_collector.start()
    await performance_tracker.start()
    await alert_manager.start()
    await prometheus_exporter.start()


# Convenience function to stop all monitoring services
async def stop_monitoring():
    """Para todos os serviços de monitoramento."""
    await metrics_collector.stop()
    await performance_tracker.stop()
    await alert_manager.stop()
    await prometheus_exporter.stop()