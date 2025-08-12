"""
Arbitrage Engine Module - Strategies for cryptocurrency arbitrage
"""

from core.arbitrage_engine.spatial_arbitrage import (
    SpatialArbitrageEngine,
    ArbitrageOpportunity
)

# Imports futuros quando os módulos forem implementados
# from core.arbitrage_engine.base_arbitrage import BaseArbitrageEngine
# from core.arbitrage_engine.statistical_arbitrage import StatisticalArbitrageEngine
# from core.arbitrage_engine.triangular_arbitrage import TriangularArbitrageEngine
# from core.arbitrage_engine.opportunity_detector import UnifiedOpportunityDetector

__all__ = [
    'SpatialArbitrageEngine',
    'ArbitrageOpportunity',
    # Exports futuros
    # 'BaseArbitrageEngine',
    # 'StatisticalArbitrageEngine',
    # 'TriangularArbitrageEngine',
    # 'UnifiedOpportunityDetector'
]