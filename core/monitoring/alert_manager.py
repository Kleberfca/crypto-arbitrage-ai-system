"""
Alert Manager - Sistema de gerenciamento de alertas
Detecta condições críticas e envia notificações
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging
import json

import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config.settings import settings
from core.monitoring.performance_tracker import PerformanceLevel, PerformanceSnapshot

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidade do alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Categoria do alerta."""
    PERFORMANCE = "performance"
    TRADING = "trading"
    SYSTEM = "system"
    SECURITY = "security"
    RISK = "risk"
    CONNECTIVITY = "connectivity"


@dataclass
class AlertRule:
    """Regra de alerta."""
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    category: AlertCategory
    threshold: float
    duration: int  # seconds - how long condition must persist
    cooldown: int  # seconds - minimum time between alerts
    message_template: str
    actions: List[str] = field(default_factory=list)  # ['log', 'email', 'webhook']
    enabled: bool = True


@dataclass
class Alert:
    """Alerta individual."""
    id: str
    rule_name: str
    severity: AlertSeverity
    category: AlertCategory
    message: str
    details: Dict[str, Any]
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    notification_sent: bool = False
    
    @property
    def duration(self) -> float:
        """Duração do alerta em segundos."""
        if self.resolved and self.resolved_at:
            return self.resolved_at - self.timestamp
        return time.time() - self.timestamp


class AlertManager:
    """
    Gerenciador de alertas do sistema.
    Monitora condições, gera alertas e envia notificações.
    """
    
    def __init__(self):
        """Inicializa o gerenciador de alertas."""
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert rules
        self.rules = self._init_default_rules()
        self.custom_rules: Dict[str, AlertRule] = {}
        
        # Cooldown tracking
        self.last_alert_time: Dict[str, float] = {}
        
        # Alert conditions tracking
        self.condition_start_time: Dict[str, float] = {}
        
        # Notification channels
        self.notification_handlers = {
            'log': self._send_log_notification,
            'email': self._send_email_notification,
            'webhook': self._send_webhook_notification,
            'telegram': self._send_telegram_notification
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self.is_monitoring = False
        self._monitoring_task = None
        
        # Statistics
        self.alert_stats = defaultdict(lambda: {
            'total': 0,
            'active': 0,
            'resolved': 0,
            'last_triggered': None
        })
        
    def _init_default_rules(self) -> Dict[str, AlertRule]:
        """Inicializa regras padrão de alerta."""
        return {
            'high_latency': AlertRule(
                name='high_latency',
                condition='latency_ms > threshold',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                threshold=20,
                duration=30,
                cooldown=300,
                message_template='High latency detected: {latency_ms:.1f}ms (threshold: {threshold}ms)',
                actions=['log', 'webhook']
            ),
            'critical_latency': AlertRule(
                name='critical_latency',
                condition='latency_ms > threshold',
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.PERFORMANCE,
                threshold=30,
                duration=10,
                cooldown=600,
                message_template='CRITICAL: Latency exceeds {threshold}ms: {latency_ms:.1f}ms',
                actions=['log', 'email', 'webhook']
            ),
            'low_success_rate': AlertRule(
                name='low_success_rate',
                condition='success_rate < threshold',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TRADING,
                threshold=80,
                duration=60,
                cooldown=600,
                message_template='Low success rate: {success_rate:.1f}% (threshold: {threshold}%)',
                actions=['log', 'webhook']
            ),
            'critical_success_rate': AlertRule(
                name='critical_success_rate',
                condition='success_rate < threshold',
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.TRADING,
                threshold=60,
                duration=30,
                cooldown=900,
                message_template='CRITICAL: Success rate below {threshold}%: {success_rate:.1f}%',
                actions=['log', 'email', 'webhook', 'telegram']
            ),
            'high_error_rate': AlertRule(
                name='high_error_rate',
                condition='error_rate > threshold',
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                threshold=5,
                duration=30,
                cooldown=300,
                message_template='High error rate: {error_rate:.1f}% (threshold: {threshold}%)',
                actions=['log', 'webhook']
            ),
            'exchange_disconnected': AlertRule(
                name='exchange_disconnected',
                condition='exchange_connected == False',
                severity=AlertSeverity.ERROR,
                category=AlertCategory.CONNECTIVITY,
                threshold=0,
                duration=10,
                cooldown=300,
                message_template='Exchange {exchange} disconnected',
                actions=['log', 'webhook']
            ),
            'high_cpu_usage': AlertRule(
                name='high_cpu_usage',
                condition='cpu_usage > threshold',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                threshold=80,
                duration=60,
                cooldown=600,
                message_template='High CPU usage: {cpu_usage:.1f}% (threshold: {threshold}%)',
                actions=['log']
            ),
            'high_memory_usage': AlertRule(
                name='high_memory_usage',
                condition='memory_usage > threshold',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                threshold=3072,
                duration=60,
                cooldown=600,
                message_template='High memory usage: {memory_usage:.0f}MB (threshold: {threshold}MB)',
                actions=['log']
            ),
            'daily_loss_limit': AlertRule(
                name='daily_loss_limit',
                condition='daily_loss > threshold',
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.RISK,
                threshold=settings.risk_limits.max_daily_loss_pct * 100,
                duration=0,
                cooldown=86400,
                message_template='CRITICAL: Daily loss limit reached: {daily_loss:.2f}% (limit: {threshold}%)',
                actions=['log', 'email', 'webhook', 'telegram']
            ),
            'max_drawdown': AlertRule(
                name='max_drawdown',
                condition='drawdown > threshold',
                severity=AlertSeverity.ERROR,
                category=AlertCategory.RISK,
                threshold=settings.risk_limits.max_drawdown_pct * 100,
                duration=0,
                cooldown=3600,
                message_template='Maximum drawdown exceeded: {drawdown:.2f}% (limit: {threshold}%)',
                actions=['log', 'email', 'webhook']
            ),
            'no_opportunities': AlertRule(
                name='no_opportunities',
                condition='opportunities_per_minute < threshold',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TRADING,
                threshold=1,
                duration=300,
                cooldown=1800,
                message_template='Low opportunity detection: {opportunities_per_minute:.2f}/min',
                actions=['log']
            ),
            'performance_degraded': AlertRule(
                name='performance_degraded',
                condition='performance_level == "degraded"',
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                threshold=0,
                duration=60,
                cooldown=600,
                message_template='System performance degraded',
                actions=['log', 'webhook']
            ),
            'performance_critical': AlertRule(
                name='performance_critical',
                condition='performance_level == "critical"',
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.PERFORMANCE,
                threshold=0,
                duration=30,
                cooldown=900,
                message_template='CRITICAL: System performance critical',
                actions=['log', 'email', 'webhook', 'telegram']
            )
        }
    
    def add_custom_rule(self, rule: AlertRule) -> None:
        """
        Adiciona regra customizada.
        
        Args:
            rule: Regra de alerta
        """
        self.custom_rules[rule.name] = rule
        logger.info(f"Added custom alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> None:
        """
        Remove regra de alerta.
        
        Args:
            rule_name: Nome da regra
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
        elif rule_name in self.custom_rules:
            del self.custom_rules[rule_name]
        
        logger.info(f"Removed alert rule: {rule_name}")
    
    def evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """
        Avalia se uma regra foi violada.
        
        Args:
            rule: Regra a avaliar
            metrics: Métricas atuais
            
        Returns:
            True se a condição foi violada
        """
        try:
            # Add threshold to metrics for evaluation
            eval_context = {**metrics, 'threshold': rule.threshold}
            
            # Evaluate condition
            # WARNING: In production, use a safe expression evaluator
            result = eval(rule.condition, {"__builtins__": {}}, eval_context)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return False
    
    async def check_alerts(self, metrics: Dict[str, Any]) -> None:
        """
        Verifica todas as regras de alerta.
        
        Args:
            metrics: Métricas atuais do sistema
        """
        all_rules = {**self.rules, **self.custom_rules}
        
        for rule_name, rule in all_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule is violated
            violated = self.evaluate_rule(rule, metrics)
            
            if violated:
                # Track how long condition has been true
                if rule_name not in self.condition_start_time:
                    self.condition_start_time[rule_name] = time.time()
                
                # Check if duration requirement met
                condition_duration = time.time() - self.condition_start_time[rule_name]
                
                if condition_duration >= rule.duration:
                    # Check cooldown
                    last_alert = self.last_alert_time.get(rule_name, 0)
                    if time.time() - last_alert >= rule.cooldown:
                        # Trigger alert
                        await self.trigger_alert(rule, metrics)
            else:
                # Condition no longer true
                if rule_name in self.condition_start_time:
                    del self.condition_start_time[rule_name]
                
                # Check if we should resolve active alert
                await self.check_resolution(rule_name)
    
    async def trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> None:
        """
        Dispara um alerta.
        
        Args:
            rule: Regra violada
            metrics: Métricas atuais
        """
        # Format message
        message = rule.message_template.format(**metrics, threshold=rule.threshold)
        
        # Create alert
        alert = Alert(
            id=f"{rule.name}_{int(time.time())}",
            rule_name=rule.name,
            severity=rule.severity,
            category=rule.category,
            message=message,
            details=metrics,
            timestamp=time.time()
        )
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.alert_stats[rule.name]['total'] += 1
        self.alert_stats[rule.name]['active'] += 1
        self.alert_stats[rule.name]['last_triggered'] = time.time()
        
        # Update cooldown
        self.last_alert_time[rule.name] = time.time()
        
        # Send notifications
        for action in rule.actions:
            if action in self.notification_handlers:
                await self.notification_handlers[action](alert)
        
        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert triggered: {rule.name} - {message}")
    
    async def check_resolution(self, rule_name: str) -> None:
        """
        Verifica se um alerta pode ser resolvido.
        
        Args:
            rule_name: Nome da regra
        """
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                
                # Update statistics
                self.alert_stats[rule_name]['active'] -= 1
                self.alert_stats[rule_name]['resolved'] += 1
                
                # Remove from active
                del self.active_alerts[rule_name]
                
                logger.info(f"Alert resolved: {rule_name} (duration: {alert.duration:.1f}s)")
    
    async def _send_log_notification(self, alert: Alert) -> None:
        """Envia notificação via log."""
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(level, f"[ALERT] {alert.message}")
    
    async def _send_email_notification(self, alert: Alert) -> None:
        """Envia notificação via email."""
        # In production, implement actual email sending
        logger.info(f"Email notification would be sent: {alert.message}")
    
    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Envia notificação via webhook."""
        webhook_url = getattr(settings, 'alert_webhook_url', None)
        
        if not webhook_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'alert_id': alert.id,
                    'rule': alert.rule_name,
                    'severity': alert.severity.value,
                    'category': alert.category.value,
                    'message': alert.message,
                    'details': alert.details,
                    'timestamp': alert.timestamp
                }
                
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        alert.notification_sent = True
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def _send_telegram_notification(self, alert: Alert) -> None:
        """Envia notificação via Telegram."""
        # In production, implement actual Telegram sending
        logger.info(f"Telegram notification would be sent: {alert.message}")
    
    async def _monitoring_loop(self) -> None:
        """Loop de monitoramento de alertas."""
        logger.info("Starting alert monitoring loop")
        
        while self.is_monitoring:
            try:
                # Import here to avoid circular dependency
                from core.monitoring.metrics_collector import metrics_collector
                from core.monitoring.performance_tracker import performance_tracker
                
                # Get current metrics
                system_metrics = await metrics_collector.get_system_metrics()
                business_metrics = await metrics_collector.get_business_metrics()
                
                # Combine metrics
                all_metrics = {**system_metrics, **business_metrics}
                
                # Add performance level
                if performance_tracker.current_performance:
                    all_metrics['performance_level'] = performance_tracker.current_performance.level.value
                
                # Check alerts
                await self.check_alerts(all_metrics)
                
                # Clean old alerts from history
                self._cleanup_old_alerts()
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def _cleanup_old_alerts(self) -> None:
        """Remove alertas antigos do histórico."""
        # Keep only last 24 hours of resolved alerts
        cutoff_time = time.time() - 86400
        
        while self.alert_history and self.alert_history[0].timestamp < cutoff_time:
            self.alert_history.popleft()
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Obtém alertas ativos.
        
        Returns:
            Lista de alertas ativos
        """
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de alertas.
        
        Returns:
            Estatísticas detalhadas
        """
        return {
            'active_count': len(self.active_alerts),
            'total_triggered': sum(stats['total'] for stats in self.alert_stats.values()),
            'by_severity': self._count_by_severity(),
            'by_category': self._count_by_category(),
            'rules': dict(self.alert_stats),
            'history_size': len(self.alert_history)
        }
    
    def _count_by_severity(self) -> Dict[str, int]:
        """Conta alertas por severidade."""
        counts = defaultdict(int)
        for alert in self.active_alerts.values():
            counts[alert.severity.value] += 1
        return dict(counts)
    
    def _count_by_category(self) -> Dict[str, int]:
        """Conta alertas por categoria."""
        counts = defaultdict(int)
        for alert in self.active_alerts.values():
            counts[alert.category.value] += 1
        return dict(counts)
    
    def register_callback(self, callback: Callable) -> None:
        """
        Registra callback para alertas.
        
        Args:
            callback: Função callback async
        """
        self.alert_callbacks.append(callback)
    
    async def start(self) -> None:
        """Inicia monitoramento de alertas."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Alert manager started")
    
    async def stop(self) -> None:
        """Para monitoramento de alertas."""
        self.is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Alert manager stopped")


# Global instance
alert_manager = AlertManager()