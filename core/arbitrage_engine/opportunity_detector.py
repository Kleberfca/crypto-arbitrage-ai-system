"""
Unified Opportunity Detector - Detector centralizado de oportunidades
Coordena todas as estratégias de arbitragem
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
import numpy as np

from config.settings import settings, Strategy
from core.data_collector.exchange_connector import MultiExchangeConnector
from core.arbitrage_engine.base_arbitrage import BaseOpportunity, OpportunityType
from core.arbitrage_engine.spatial_arbitrage import SpatialArbitrageEngine
# Imports futuros
# from core.arbitrage_engine.statistical_arbitrage import StatisticalArbitrageEngine
# from core.arbitrage_engine.triangular_arbitrage import TriangularArbitrageEngine


logger = logging.getLogger(__name__)


class UnifiedOpportunityDetector:
    """
    Detector unificado que coordena todas as estratégias de arbitragem.
    Gerencia priorização, deduplicação e roteamento de oportunidades.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa detector unificado.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        
        # Initialize engines
        self.engines = {}
        
        # Spatial arbitrage
        if Strategy.SPATIAL in settings.enabled_strategies:
            self.engines[OpportunityType.SPATIAL] = SpatialArbitrageEngine(connector)
        
        # Statistical arbitrage (quando implementado)
        # if Strategy.STATISTICAL in settings.enabled_strategies:
        #     self.engines[OpportunityType.STATISTICAL] = StatisticalArbitrageEngine(connector)
        
        # Triangular arbitrage (quando implementado)
        # if Strategy.TRIANGULAR in settings.enabled_strategies:
        #     self.engines[OpportunityType.TRIANGULAR] = TriangularArbitrageEngine(connector)
        
        # Opportunity management
        self.all_opportunities = []
        self.opportunity_history = defaultdict(list)
        self.max_history_size = 1000
        
        # Performance tracking
        self.detection_stats = defaultdict(lambda: {
            'total': 0,
            'profitable': 0,
            'executed': 0,
            'avg_profit_pct': 0
        })
        
        # Deduplication
        self.seen_opportunities = set()
        self.dedup_window = 5  # seconds
        
    async def detect_all_opportunities(
        self,
        symbols: List[str]
    ) -> List[BaseOpportunity]:
        """
        Detecta oportunidades de todas as estratégias em paralelo.
        
        Args:
            symbols: Lista de símbolos para escanear
            
        Returns:
            Lista consolidada de oportunidades
        """
        start_time = time.perf_counter()
        
        # Create tasks for all engines
        tasks = []
        for opportunity_type, engine in self.engines.items():
            task = asyncio.create_task(
                self._scan_with_timeout(engine, symbols)
            )
            tasks.append((opportunity_type, task))
        
        # Wait for all tasks
        all_opportunities = []
        
        for opportunity_type, task in tasks:
            try:
                opportunities = await task
                all_opportunities.extend(opportunities)
                
                # Update stats
                self.detection_stats[opportunity_type]['total'] += len(opportunities)
                profitable = sum(1 for o in opportunities if o.is_profitable)
                self.detection_stats[opportunity_type]['profitable'] += profitable
                
            except Exception as e:
                logger.error(f"Error detecting {opportunity_type.value} opportunities: {e}")
        
        # Deduplicate
        unique_opportunities = self._deduplicate_opportunities(all_opportunities)
        
        # Prioritize
        prioritized = self._prioritize_opportunities(unique_opportunities)
        
        # Update global list
        self.all_opportunities = prioritized
        
        # Log detection time
        detection_time = (time.perf_counter() - start_time) * 1000
        
        if detection_time > 10:  # Warning if > 10ms
            logger.warning(f"Slow opportunity detection: {detection_time:.2f}ms")
        
        logger.info(
            f"Detected opportunities",
            total=len(prioritized),
            spatial=self.detection_stats[OpportunityType.SPATIAL]['total'],
            detection_ms=f"{detection_time:.2f}"
        )
        
        return prioritized
    
    async def _scan_with_timeout(
        self,
        engine: Any,
        symbols: List[str],
        timeout: float = 5.0
    ) -> List[BaseOpportunity]:
        """
        Escaneia com timeout para evitar travamento.
        
        Args:
            engine: Engine de arbitragem
            symbols: Símbolos para escanear
            timeout: Timeout em segundos
            
        Returns:
            Lista de oportunidades
        """
        try:
            return await asyncio.wait_for(
                engine.scan_opportunities(symbols),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout scanning {engine.opportunity_type.value}")
            return []
    
    def _deduplicate_opportunities(
        self,
        opportunities: List[BaseOpportunity]
    ) -> List[BaseOpportunity]:
        """
        Remove oportunidades duplicadas.
        
        Args:
            opportunities: Lista de oportunidades
            
        Returns:
            Lista sem duplicatas
        """
        unique = []
        current_time = time.time()
        
        # Clean old seen opportunities
        self.seen_opportunities = {
            opp_id for opp_id in self.seen_opportunities
            if current_time - float(opp_id.split('_')[-1]) < self.dedup_window
        }
        
        for opportunity in opportunities:
            # Create unique key
            key = f"{opportunity.opportunity_type.value}_{opportunity.opportunity_id}_{opportunity.timestamp}"
            
            if key not in self.seen_opportunities:
                unique.append(opportunity)
                self.seen_opportunities.add(key)
        
        return unique
    
    def _prioritize_opportunities(
        self,
        opportunities: List[BaseOpportunity]
    ) -> List[BaseOpportunity]:
        """
        Prioriza oportunidades por lucro e confiança.
        
        Args:
            opportunities: Lista de oportunidades
            
        Returns:
            Lista priorizada
        """
        if not opportunities:
            return []
        
        # Calculate priority score
        for i, opp in enumerate(opportunities):
            # Priority based on:
            # 1. Profitability (40%)
            # 2. Confidence (30%)
            # 3. Risk (20%)
            # 4. Age (10%)
            
            profit_score = min(opp.expected_profit_pct / 2.0, 1.0) * 0.4
            confidence_score = opp.confidence * 0.3
            risk_score = (1 - opp.risk_score) * 0.2
            age_score = max(0, 1 - (opp.age_seconds / 10)) * 0.1
            
            priority = profit_score + confidence_score + risk_score + age_score
            opp.priority = int((1 - priority) * 100)  # Lower is better
        
        # Sort by priority
        return sorted(opportunities, key=lambda x: x.priority)
    
    def get_best_opportunity(
        self,
        opportunity_type: Optional[OpportunityType] = None
    ) -> Optional[BaseOpportunity]:
        """
        Retorna melhor oportunidade disponível.
        
        Args:
            opportunity_type: Filtrar por tipo (opcional)
            
        Returns:
            Melhor oportunidade ou None
        """
        if not self.all_opportunities:
            return None
        
        if opportunity_type:
            filtered = [
                o for o in self.all_opportunities 
                if o.opportunity_type == opportunity_type
            ]
            return filtered[0] if filtered else None
        
        return self.all_opportunities[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de detecção."""
        stats = {
            'current_opportunities': len(self.all_opportunities),
            'by_type': {}
        }
        
        for opp_type, data in self.detection_stats.items():
            stats['by_type'][opp_type.value] = {
                'total_detected': data['total'],
                'profitable': data['profitable'],
                'executed': data['executed'],
                'success_rate': (
                    data['executed'] / data['total'] * 100 
                    if data['total'] > 0 else 0
                )
            }
        
        return stats
    
    async def execute_best_opportunity(self) -> bool:
        """
        Executa a melhor oportunidade disponível.
        
        Returns:
            True se executado com sucesso
        """
        best = self.get_best_opportunity()
        
        if not best or not best.is_profitable:
            return False
        
        # Get appropriate engine
        engine = self.engines.get(best.opportunity_type)
        if not engine:
            logger.error(f"No engine for {best.opportunity_type.value}")
            return False
        
        # Validate before execution
        is_valid, reason = await engine.validate_opportunity(best)
        if not is_valid:
            logger.info(f"Opportunity no longer valid: {reason}")
            return False
        
        # Execute
        success = await engine.execute_opportunity(best)
        
        if success:
            self.detection_stats[best.opportunity_type]['executed'] += 1
            logger.info(
                f"Executed {best.opportunity_type.value} opportunity",
                profit=f"${best.expected_profit_usd:.2f}"
            )
        
        return success
    
    async def run_continuous(
        self,
        symbols: List[str],
        interval: float = 0.1
    ) -> None:
        """
        Executa detecção contínua.
        
        Args:
            symbols: Símbolos para monitorar
            interval: Intervalo entre detecções
        """
        logger.info("Starting unified opportunity detection")
        
        while True:
            try:
                # Detect opportunities
                await self.detect_all_opportunities(symbols)
                
                # Auto-execute if configured
                if settings.environment == "production":
                    await self.execute_best_opportunity()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in unified detector: {e}")
                await asyncio.sleep(1)