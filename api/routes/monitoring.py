"""
Monitoring Routes - Endpoints para monitoramento e métricas
Health checks, performance metrics e alertas do sistema
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import logging
import time
import asyncio
from datetime import datetime, timedelta

from api.models.responses import (
    HealthStatus,
    SystemStatus,
    PerformanceMetrics,
    SystemMetrics,
    Alert,
    RiskMetrics,
    SuccessResponse,
    WebSocketMessage
)
from config.settings import settings, Environment
from core.monitoring import (
    metrics_collector,
    performance_tracker,
    alert_manager,
    prometheus_exporter
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])


# ============================================
# Health & Status
# ============================================

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Comprehensive health check of all system components.
    
    Returns detailed status of each service and overall system health.
    """
    start_time = time.perf_counter()
    
    # Check various components
    checks = {}
    services = {}
    
    # Check metrics collector
    try:
        system_metrics = await metrics_collector.get_system_metrics()
        checks['metrics_collector'] = True
        services['metrics_collector'] = 'healthy'
    except Exception as e:
        logger.error(f"Metrics collector unhealthy: {e}")
        checks['metrics_collector'] = False
        services['metrics_collector'] = 'unhealthy'
    
    # Check performance tracker
    try:
        perf_report = performance_tracker.get_performance_report()
        checks['performance_tracker'] = True
        services['performance_tracker'] = perf_report.get('level', 'unknown')
    except Exception as e:
        logger.error(f"Performance tracker unhealthy: {e}")
        checks['performance_tracker'] = False
        services['performance_tracker'] = 'unhealthy'
    
    # Check alert manager
    try:
        alert_stats = alert_manager.get_alert_statistics()
        checks['alert_manager'] = True
        services['alert_manager'] = f"{alert_stats['active_count']} active alerts"
    except Exception as e:
        logger.error(f"Alert manager unhealthy: {e}")
        checks['alert_manager'] = False
        services['alert_manager'] = 'unhealthy'
    
    # Check Redis
    try:
        if metrics_collector.redis_client:
            await metrics_collector.redis_client.ping()
            checks['redis'] = True
            services['redis'] = 'connected'
        else:
            checks['redis'] = False
            services['redis'] = 'not configured'
    except Exception:
        checks['redis'] = False
        services['redis'] = 'disconnected'
    
    # Determine overall status
    critical_checks = ['metrics_collector', 'performance_tracker']
    critical_healthy = all(checks.get(check, False) for check in critical_checks)
    all_healthy = all(checks.values())
    
    if all_healthy:
        overall_status = SystemStatus.HEALTHY
    elif critical_healthy:
        overall_status = SystemStatus.DEGRADED
    else:
        overall_status = SystemStatus.UNHEALTHY
    
    # Calculate health check latency
    health_check_time = (time.perf_counter() - start_time) * 1000
    services['health_check_latency_ms'] = health_check_time
    
    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        uptime_seconds=time.time() - performance_tracker.start_time,
        version=settings.version,
        environment=settings.environment.value,
        services=services,
        checks=checks
    )


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """
    Get current system status overview.
    
    Returns simplified status information.
    """
    health = await health_check()
    perf_report = performance_tracker.get_performance_report()
    alert_stats = alert_manager.get_alert_statistics()
    
    return {
        "status": health.status,
        "uptime_hours": health.uptime_seconds / 3600,
        "performance_level": perf_report.get('level', 'unknown'),
        "active_alerts": alert_stats['active_count'],
        "environment": settings.environment.value,
        "timestamp": time.time()
    }


# ============================================
# Performance Metrics
# ============================================

@router.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """
    Get current system performance metrics.
    
    Returns real-time performance indicators.
    """
    # Get metrics from collectors
    system_metrics = await metrics_collector.get_system_metrics()
    business_metrics = await metrics_collector.get_business_metrics()
    perf_report = performance_tracker.get_performance_report()
    
    # Build response
    return PerformanceMetrics(
        latency_ms=system_metrics.get('latency_ms', {}).get('current', 0),
        throughput_per_sec=system_metrics.get('throughput', {}).get('current', 0),
        opportunities_detected=business_metrics.get('opportunities_detected', {}).get('total', 0),
        opportunities_executed=business_metrics.get('trades_executed', {}).get('total', 0),
        total_profit_usd=business_metrics.get('total_profit_usd', {}).get('total', 0),
        success_rate=business_metrics.get('success_rate', {}).get('current', 0),
        sharpe_ratio=business_metrics.get('sharpe_ratio', {}).get('current', 0),
        sortino_ratio=business_metrics.get('sortino_ratio', {}).get('current', 0),
        max_drawdown_pct=business_metrics.get('max_drawdown', {}).get('current', 0),
        win_rate=perf_report.get('statistics', {}).get('avg_success_rate', 0),
        average_profit_per_trade=business_metrics.get('total_profit_usd', {}).get('total', 0) / 
                                max(business_metrics.get('trades_executed', {}).get('total', 1), 1),
        total_trades=business_metrics.get('trades_executed', {}).get('total', 0),
        active_positions=0,  # TODO: Get from position manager
        cache_hit_rate=85.0,  # Mock
        feature_extraction_ms=system_metrics.get('feature_extraction_ms', {}).get('current', 0),
        ml_inference_ms=2.0,  # Mock
        xai_generation_ms=5.0  # Mock
    )


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get system resource metrics.
    
    Returns CPU, memory, disk, and network usage.
    """
    system_metrics = await metrics_collector.get_system_metrics()
    
    return SystemMetrics(
        cpu_usage_pct=system_metrics.get('cpu_usage', {}).get('current', 0),
        memory_usage_mb=system_metrics.get('memory_usage', {}).get('current', 0),
        disk_usage_pct=0,  # TODO: Implement disk monitoring
        network_in_mbps=0,  # TODO: Implement network monitoring
        network_out_mbps=0,
        active_connections=system_metrics.get('active_connections', 0),
        error_rate_pct=system_metrics.get('error_rate', 0),
        uptime_hours=(time.time() - performance_tracker.start_time) / 3600
    )


@router.get("/metrics/trends")
async def get_metrics_trends(
    window: int = Query(default=3600, description="Time window in seconds"),
    metrics: List[str] = Query(default=["latency_ms", "throughput", "success_rate"])
):
    """
    Get historical trends for specified metrics.
    
    Args:
        window: Time window in seconds
        metrics: List of metrics to retrieve
        
    Returns:
        Time series data for requested metrics
    """
    trends = performance_tracker.get_performance_trends(window)
    
    # Filter requested metrics
    filtered = {
        'timestamps': trends.get('timestamps', []),
    }
    
    for metric in metrics:
        if metric in trends:
            filtered[metric] = trends[metric]
    
    return filtered


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    # Collect and export latest metrics
    await prometheus_exporter.collect_and_export()
    
    # Generate Prometheus format
    metrics = generate_latest(prometheus_exporter.registry)
    
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)


# ============================================
# Alerts Management
# ============================================

@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    active_only: bool = Query(default=True, description="Show only active alerts"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    category: Optional[str] = Query(default=None, description="Filter by category")
):
    """
    Get system alerts.
    
    Args:
        active_only: Whether to show only active alerts
        severity: Filter by severity level
        category: Filter by category
        
    Returns:
        List of alerts matching criteria
    """
    if active_only:
        alerts = alert_manager.get_active_alerts()
    else:
        alerts = list(alert_manager.alert_history)
    
    # Convert to response models
    response_alerts = []
    for alert in alerts:
        if severity and alert.severity.value != severity:
            continue
        if category and alert.category.value != category:
            continue
        
        response_alerts.append(Alert(
            id=alert.id,
            timestamp=alert.timestamp,
            severity=alert.severity.value,
            category=alert.category.value,
            title=alert.rule_name,
            message=alert.message,
            details=alert.details,
            resolved=alert.resolved,
            resolved_at=alert.resolved_at,
            actions_taken=[]
        ))
    
    return response_alerts


@router.get("/alerts/statistics")
async def get_alert_statistics():
    """
    Get alert statistics.
    
    Returns summary of alert activity.
    """
    return alert_manager.get_alert_statistics()


@router.post("/alerts/{alert_id}/acknowledge", response_model=SuccessResponse)
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Success response
    """
    # Find and acknowledge alert
    for rule_name, alert in alert_manager.active_alerts.items():
        if alert.id == alert_id:
            # Mark as acknowledged (resolved)
            await alert_manager.check_resolution(rule_name)
            
            return SuccessResponse(
                message=f"Alert {alert_id} acknowledged"
            )
    
    raise HTTPException(
        status_code=404,
        detail=f"Alert {alert_id} not found"
    )


# ============================================
# Risk Monitoring
# ============================================

@router.get("/risk", response_model=RiskMetrics)
async def get_risk_metrics():
    """
    Get current risk metrics.
    
    Returns comprehensive risk assessment.
    """
    # Get metrics from collectors
    business_metrics = await metrics_collector.get_business_metrics()
    
    # Mock risk data - TODO: Implement actual risk calculations
    return RiskMetrics(
        timestamp=time.time(),
        total_exposure_usd=5000.0,
        exposure_by_exchange={
            "binance": 2000.0,
            "okx": 1500.0,
            "bybit": 1000.0,
            "kucoin": 500.0
        },
        exposure_by_symbol={
            "BTC/USDT": 2500.0,
            "ETH/USDT": 1500.0,
            "BNB/USDT": 1000.0
        },
        var_95=250.0,
        cvar_95=350.0,
        max_drawdown_pct=business_metrics.get('max_drawdown', {}).get('max', 0),
        current_drawdown_pct=0,
        sharpe_ratio=business_metrics.get('sharpe_ratio', {}).get('current', 0),
        sortino_ratio=business_metrics.get('sortino_ratio', {}).get('current', 0),
        correlation_matrix={},
        concentration_risk=0.3,
        liquidity_risk=0.2,
        risk_score=0.25,
        warnings=[]
    )


# ============================================
# WebSocket Monitoring
# ============================================

@router.websocket("/ws")
async def websocket_monitoring(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring.
    
    Streams performance metrics, alerts, and system status.
    """
    await websocket.accept()
    logger.info("WebSocket monitoring connection established")
    
    try:
        # Send updates every second
        while True:
            # Get current metrics
            perf_metrics = await get_performance_metrics()
            active_alerts = alert_manager.get_active_alerts()
            
            # Prepare message
            message = WebSocketMessage(
                type="monitoring_update",
                channel="monitoring",
                data={
                    "performance": perf_metrics.dict(),
                    "alerts": len(active_alerts),
                    "alert_list": [
                        {
                            "id": alert.id,
                            "severity": alert.severity.value,
                            "message": alert.message
                        }
                        for alert in active_alerts[:5]  # Limit to 5 most recent
                    ]
                },
                timestamp=time.time()
            )
            
            # Send message
            await websocket.send_json(message.dict())
            
            # Wait before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info("WebSocket monitoring connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ============================================
# Monitoring Control
# ============================================

@router.post("/control/reset-metrics", response_model=SuccessResponse)
async def reset_metrics():
    """
    Reset all metrics collectors.
    
    Returns success response.
    """
    # Clear metric buffers
    metrics_collector.metrics_buffer.clear()
    metrics_collector.aggregated_metrics.clear()
    
    # Clear performance history
    performance_tracker.performance_history.clear()
    
    # Clear alert history
    alert_manager.alert_history.clear()
    
    logger.info("Metrics reset successfully")
    
    return SuccessResponse(
        message="All metrics have been reset"
    )


@router.post("/control/export-metrics")
async def export_metrics(
    format: str = Query(default="json", description="Export format (json, csv)")
):
    """
    Export current metrics.
    
    Args:
        format: Export format
        
    Returns:
        Metrics data in requested format
    """
    # Get all metrics
    system_metrics = await metrics_collector.get_system_metrics()
    business_metrics = await metrics_collector.get_business_metrics()
    perf_report = performance_tracker.get_performance_report()
    
    export_data = {
        "timestamp": time.time(),
        "system_metrics": system_metrics,
        "business_metrics": business_metrics,
        "performance_report": perf_report
    }
    
    if format == "json":
        return export_data
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}"
        )