"""
Base Arbitrage Engine - Classe abstrata para todas as estratégias
Define interface comum e funcionalidades compartilhadas
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

from config.settings import settings
from core.data_collector.exchange_connector import MultiExchangeConnector


logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Tipos de oportunidades de arbitragem."""
    SPATIAL = "spatial"
    STATISTICAL = "statistical"
    TRIANGULAR = "triangular"
    CROSS_CHAIN = "cross_chain"


@dataclass
class BaseOpportunity:
    """Classe base para todas as oportunidades de arbitragem."""
    
    # Identificação
    opportunity_id: str
    opportunity_type: OpportunityType
    timestamp: float = field(default_factory=time.time)
    
    # Dados financeiros
    expected_profit_pct: float
    expected_profit_usd: float
    confidence: float
    risk_score: float
    
    # Execução
    required_capital: float
    estimated_execution_time_ms: float
    priority: int = 0  # 0 = highest
    
    # Metadados
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_profitable(self) -> bool:
        """Verifica se a oportunidade é lucrativa."""
        return self.expected_profit_pct > 0 and self.confidence >= settings.risk_limits.min_confidence_threshold
    
    @property
    def age_seconds(self) -> float:
        """Retorna idade da oportunidade em segundos."""
        return time.time() - self.timestamp
    
    @property
    def is_expired(self, max_age: float = 5.0) -> bool:
        """Verifica se a oportunidade expirou."""
        return self.age_seconds > max_age
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'opportunity_id': self.opportunity_id,
            'type': self.opportunity_type.value,
            'timestamp': self.timestamp,
            'expected_profit_pct': self.expected_profit_pct,
            'expected_profit_usd': self.expected_profit_usd,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'required_capital': self.required_capital,
            'execution_time_ms': self.estimated_execution_time_ms,
            'priority': self.priority,
            'age': self.age_seconds,
            'metadata': self.metadata
        }


class BaseArbitrageEngine(ABC):
    """
    Classe base abstrata para todos os engines de arbitragem.
    Define interface comum e implementa funcionalidades compartilhadas.
    """
    
    def __init__(
        self,
        connector: MultiExchangeConnector,
        opportunity_type: OpportunityType
    ):
        """
        Inicializa engine base.
        
        Args:
            connector: Conector multi-exchange
            opportunity_type: Tipo de oportunidade que este engine detecta
        """
        self.connector = connector
        self.opportunity_type = opportunity_type
        
        # Performance tracking
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.total_profit_usd = 0.0
        self.detection_times = []
        self.execution_times = []
        
        # Buffers
        self.opportunity_buffer = []
        self.max_buffer_size = 100
        
        # Execution control
        self.is_running = False
        self.last_scan_time = 0
        self.min_scan_interval = 0.1  # 100ms minimum between scans
        
    @abstractmethod
    async def scan_opportunities(
        self,
        symbols: List[str]
    ) -> List[BaseOpportunity]:
        """
        Escaneia oportunidades de arbitragem.
        
        Args:
            symbols: Lista de símbolos para escanear
            
        Returns:
            Lista de oportunidades detectadas
        """
        pass
    
    @abstractmethod
    async def validate_opportunity(
        self,
        opportunity: BaseOpportunity
    ) -> Tuple[bool, str]:
        """
        Valida se uma oportunidade ainda é viável.
        
        Args:
            opportunity: Oportunidade a validar
            
        Returns:
            (is_valid, reason)
        """
        pass
    
    @abstractmethod
    async def execute_opportunity(
        self,
        opportunity: BaseOpportunity
    ) -> bool:
        """
        Executa uma oportunidade de arbitragem.
        
        Args:
            opportunity: Oportunidade a executar
            
        Returns:
            True se executado com sucesso
        """
        pass
    
    @abstractmethod
    def calculate_profit(
        self,
        opportunity: BaseOpportunity
    ) -> Tuple[float, float]:
        """
        Calcula lucro esperado de uma oportunidade.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            (profit_pct, profit_usd)
        """
        pass
    
    async def run_continuous(
        self,
        symbols: List[str],
        interval: float = 0.1
    ) -> None:
        """
        Executa escaneamento contínuo.
        
        Args:
            symbols: Símbolos para monitorar
            interval: Intervalo entre scans
        """
        self.is_running = True
        logger.info(f"Starting {self.opportunity_type.value} arbitrage engine")
        
        while self.is_running:
            try:
                # Rate limiting
                time_since_last = time.time() - self.last_scan_time
                if time_since_last < self.min_scan_interval:
                    await asyncio.sleep(self.min_scan_interval - time_since_last)
                
                # Scan for opportunities
                start_time = time.perf_counter()
                opportunities = await self.scan_opportunities(symbols)
                scan_time = (time.perf_counter() - start_time) * 1000
                
                self.detection_times.append(scan_time)
                self.last_scan_time = time.time()
                
                # Update stats
                self.opportunities_found += len(opportunities)
                
                # Update buffer
                self.opportunity_buffer = opportunities[:self.max_buffer_size]
                
                # Log if opportunities found
                if opportunities:
                    best = opportunities[0]
                    logger.info(
                        f"{self.opportunity_type.value} opportunity found",
                        profit_pct=f"{best.expected_profit_pct:.3f}%",
                        confidence=f"{best.confidence:.2f}",
                        scan_time_ms=f"{scan_time:.2f}"
                    )
                
                # Wait before next scan
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in {self.opportunity_type.value} engine: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine."""
        return {
            'type': self.opportunity_type.value,
            'opportunities_found': self.opportunities_found,
            'opportunities_executed': self.opportunities_executed,
            'total_profit_usd': self.total_profit_usd,
            'avg_detection_ms': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_execution_ms': np.mean(self.execution_times) if self.execution_times else 0,
            'buffer_size': len(self.opportunity_buffer),
            'is_running': self.is_running
        }
    
    def stop(self) -> None:
        """Para o engine."""
        self.is_running = False
        logger.info(f"Stopping {self.opportunity_type.value} arbitrage engine")