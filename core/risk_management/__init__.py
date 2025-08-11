"""
Risk Management System - Sistema completo de gestão de risco
Calcula métricas, dimensiona posições e monitora correlações
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from core.risk_management.risk_calculator import (
    RiskCalculator,
    RiskMetrics,
    RiskLevel,
    VaRMethod
)

from core.risk_management.position_sizer import (
    PositionSizer,
    SizingMethod,
    PositionLimit,
    CapitalAllocation
)

from core.risk_management.correlation_monitor import (
    CorrelationMonitor,
    CorrelationMatrix,
    DiversificationScore,
    ConcentrationRisk
)

from core.risk_management.adaptive_limits import (
    AdaptiveLimits,
    LimitType,
    MarketRegime,
    DynamicThreshold
)

__all__ = [
    # Risk Calculator
    'RiskCalculator',
    'RiskMetrics',
    'RiskLevel',
    'VaRMethod',
    
    # Position Sizer
    'PositionSizer',
    'SizingMethod',
    'PositionLimit',
    'CapitalAllocation',
    
    # Correlation Monitor
    'CorrelationMonitor',
    'CorrelationMatrix',
    'DiversificationScore',
    'ConcentrationRisk',
    
    # Adaptive Limits
    'AdaptiveLimits',
    'LimitType',
    'MarketRegime',
    'DynamicThreshold'
]