"""
Spatial Arbitrage Engine - Arbitragem entre exchanges
Detecta e executa oportunidades de arbitragem espacial com latência <30ms
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import logging

import numpy as np
from numba import jit
import pandas as pd

from config.settings import settings, Exchange, get_exchange_config
from core.data_collector.exchange_connector import (
    MultiExchangeConnector, 
    OrderBookSnapshot,
    calculate_weighted_mid_price
)


logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Tipos de arbitragem espacial."""
    SIMPLE = "simple"  # Compra em A, vende em B
    COMPLEX = "complex"  # Multiple hops
    FLASH = "flash"  # Ultra-fast execution


@dataclass
class ArbitrageOpportunity:
    """Representa uma oportunidade de arbitragem espacial."""
    
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    buy_volume: float
    sell_volume: float
    executable_volume: float
    gross_profit_pct: float
    net_profit_pct: float
    net_profit_usd: float
    confidence: float
    timestamp: float
    latency_ms: float
    fees: Dict[str, float]
    transfer_time_seconds: int = 30  # BEP20 transfer time
    
    @property
    def spread_pct(self) -> float:
        """Calcula spread percentual."""
        if self.buy_price == 0:
            return 0
        return ((self.sell_price - self.buy_price) / self.buy_price) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Verifica se é lucrativo após fees."""
        return self.net_profit_pct > settings.risk_limits.min_sharpe_ratio * 0.1
    
    @property
    def risk_score(self) -> float:
        """Calcula score de risco (0-1, menor é melhor)."""
        factors = []
        
        # Volume risk
        volume_ratio = self.executable_volume / max(self.buy_volume, self.sell_volume)
        factors.append(1 - volume_ratio)
        
        # Latency risk
        latency_risk = min(self.latency_ms / 30, 1)  # 30ms target
        factors.append(latency_risk)
        
        # Transfer risk (only if not pre-funded)
        if not self._is_pre_funded():
            transfer_risk = 0.3  # Fixed risk for transfers
            factors.append(transfer_risk)
        
        # Confidence inverse
        factors.append(1 - self.confidence)
        
        return np.mean(factors)
    
    def _is_pre_funded(self) -> bool:
        """Verifica se temos pre-funding nas exchanges."""
        # Simplified - in production, check actual balances
        return True  # Assuming 70% pre-funding as per settings


@dataclass
class ArbitrageStats:
    """Estatísticas de arbitragem."""
    
    opportunities_detected: int = 0
    opportunities_executed: int = 0
    total_volume_usd: float = 0
    total_profit_usd: float = 0
    success_rate: float = 0
    average_profit_pct: float = 0
    average_latency_ms: float = 0
    best_opportunity: Optional[ArbitrageOpportunity] = None
    last_update: float = field(default_factory=time.time)


class SpatialArbitrageEngine:
    """
    Motor de arbitragem espacial entre exchanges.
    Detecta e executa oportunidades com latência <30ms.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa o motor de arbitragem.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        self.stats = ArbitrageStats()
        
        # Configuration
        self.min_profit_pct = 0.3  # 0.3% minimum profit after fees
        self.max_position_size = settings.risk_limits.max_position_size_pct
        self.confidence_threshold = settings.risk_limits.min_confidence_threshold
        
        # Opportunity buffer
        self.opportunity_buffer: List[ArbitrageOpportunity] = []
        self.max_buffer_size = 100
        
        # Execution tracking
        self.executing: Dict[str, bool] = {}
        self.last_execution: Dict[str, float] = {}
        self.execution_cooldown = 1.0  # seconds
        
        # Performance tracking
        self.detection_times: List[float] = []
        self.execution_times: List[float] = []
        
        # Fee cache
        self.exchange_fees = self._load_exchange_fees()
        
    def _load_exchange_fees(self) -> Dict[str, Dict[str, float]]:
        """Carrega fees das exchanges."""
        fees = {}
        for exchange in Exchange:
            config = get_exchange_config(exchange)
            fees[exchange.value] = {
                'maker': config.get('maker_fee', 0.001),
                'taker': config.get('taker_fee', 0.001),
                'withdrawal': config.get('withdrawal_fee_bep20', 0.8)
            }
        return fees
    
    async def scan_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """
        Escaneia oportunidades de arbitragem para múltiplos símbolos.
        
        Args:
            symbols: Lista de símbolos para escanear
            
        Returns:
            Lista de oportunidades detectadas
        """
        start_time = time.perf_counter()
        opportunities = []
        
        # Parallel scanning for all symbols
        tasks = [self._scan_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                opportunities.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error scanning symbol: {result}")
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x.net_profit_usd, reverse=True)
        
        # Update stats
        scan_time = (time.perf_counter() - start_time) * 1000
        self.detection_times.append(scan_time)
        
        if scan_time > 10:  # Warning if scan takes >10ms
            logger.warning(f"Slow opportunity scan: {scan_time:.2f}ms")
        
        # Update buffer
        self.opportunity_buffer = opportunities[:self.max_buffer_size]
        
        # Update statistics
        self.stats.opportunities_detected += len(opportunities)
        if opportunities:
            self.stats.best_opportunity = opportunities[0]
        
        return opportunities
    
    async def _scan_symbol(self, symbol: str) -> List[ArbitrageOpportunity]:
        """
        Escaneia oportunidades para um símbolo específico.
        
        Args:
            symbol: Símbolo para escanear
            
        Returns:
            Lista de oportunidades para o símbolo
        """
        opportunities = []
        
        # Get orderbooks from all exchanges
        orderbooks = self.connector.orderbooks.get(symbol, {})
        
        if len(orderbooks) < 2:
            return opportunities
        
        # Compare all exchange pairs
        exchanges = list(orderbooks.keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                ob1, ob2 = orderbooks[ex1], orderbooks[ex2]
                
                # Check both directions
                opp1 = self._check_arbitrage(symbol, ex1, ex2, ob1, ob2)
                if opp1 and opp1.is_profitable:
                    opportunities.append(opp1)
                
                opp2 = self._check_arbitrage(symbol, ex2, ex1, ob2, ob1)
                if opp2 and opp2.is_profitable:
                    opportunities.append(opp2)
        
        return opportunities
    
    def _check_arbitrage(
        self, 
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_orderbook: OrderBookSnapshot,
        sell_orderbook: OrderBookSnapshot
    ) -> Optional[ArbitrageOpportunity]:
        """
        Verifica oportunidade de arbitragem entre duas exchanges.
        
        Args:
            symbol: Símbolo
            buy_exchange: Exchange para comprar
            sell_exchange: Exchange para vender
            buy_orderbook: Orderbook da exchange de compra
            sell_orderbook: Orderbook da exchange de venda
            
        Returns:
            Oportunidade se existir e for lucrativa
        """
        start_time = time.perf_counter()
        
        # Get best prices
        buy_price, buy_volume = buy_orderbook.best_ask
        sell_price, sell_volume = sell_orderbook.best_bid
        
        # Quick profitability check
        if buy_price <= 0 or sell_price <= 0 or sell_price <= buy_price:
            return None
        
        # Calculate gross spread
        gross_spread_pct = ((sell_price - buy_price) / buy_price) * 100
        
        # Quick filter - need at least 0.5% gross spread
        if gross_spread_pct < 0.5:
            return None
        
        # Calculate executable volume
        executable_volume = min(buy_volume, sell_volume)
        
        # Apply position size limit
        max_position_usd = 10000 * self.max_position_size  # Assume $10k capital
        max_volume = max_position_usd / buy_price
        executable_volume = min(executable_volume, max_volume)
        
        # Calculate fees
        buy_fee = self.exchange_fees[buy_exchange]['taker']
        sell_fee = self.exchange_fees[sell_exchange]['taker']
        
        # Transfer fee only if not pre-funded
        transfer_fee = 0
        if not self._has_pre_funding(buy_exchange, sell_exchange):
            transfer_fee = self.exchange_fees[buy_exchange]['withdrawal'] / (buy_price * executable_volume)
        
        total_fee_pct = (buy_fee + sell_fee + transfer_fee) * 100
        
        # Calculate net profit
        net_profit_pct = gross_spread_pct - total_fee_pct
        net_profit_usd = (net_profit_pct / 100) * (buy_price * executable_volume)
        
        # Calculate confidence based on orderbook depth
        confidence = self._calculate_confidence(
            buy_orderbook, 
            sell_orderbook,
            executable_volume
        )
        
        # Latency calculation
        latency_ms = (time.perf_counter() - start_time) * 1000
        latency_ms += (buy_orderbook.latency_ms + sell_orderbook.latency_ms) / 2
        
        return ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            executable_volume=executable_volume,
            gross_profit_pct=gross_spread_pct,
            net_profit_pct=net_profit_pct,
            net_profit_usd=net_profit_usd,
            confidence=confidence,
            timestamp=time.time(),
            latency_ms=latency_ms,
            fees={
                'buy': buy_fee,
                'sell': sell_fee,
                'transfer': transfer_fee
            }
        )
    
    def _has_pre_funding(self, exchange1: str, exchange2: str) -> bool:
        """
        Verifica se temos pre-funding em ambas exchanges.
        
        Args:
            exchange1: Primeira exchange
            exchange2: Segunda exchange
            
        Returns:
            True se temos fundos pré-alocados
        """
        # Simplified - in production, check actual balances
        # Assume 70% pre-funding as per settings
        return True
    
    def _calculate_confidence(
        self,
        buy_orderbook: OrderBookSnapshot,
        sell_orderbook: OrderBookSnapshot,
        volume: float
    ) -> float:
        """
        Calcula confiança na execução baseado na profundidade do orderbook.
        
        Args:
            buy_orderbook: Orderbook de compra
            sell_orderbook: Orderbook de venda
            volume: Volume a executar
            
        Returns:
            Score de confiança (0-1)
        """
        confidence_factors = []
        
        # 1. Volume availability
        buy_depth = self._calculate_depth_for_volume(buy_orderbook.asks, volume)
        sell_depth = self._calculate_depth_for_volume(sell_orderbook.bids, volume)
        
        volume_confidence = min(buy_depth, sell_depth) / volume if volume > 0 else 0
        confidence_factors.append(min(volume_confidence, 1.0))
        
        # 2. Spread tightness
        spread_bps = (buy_orderbook.spread / buy_orderbook.mid_price) * 10000 if buy_orderbook.mid_price > 0 else 1000
        spread_confidence = max(0, 1 - (spread_bps / 100))  # 100 bps = 0 confidence
        confidence_factors.append(spread_confidence)
        
        # 3. Orderbook balance
        buy_total = np.sum(buy_orderbook.bids[:, 1]) if len(buy_orderbook.bids) > 0 else 0
        sell_total = np.sum(buy_orderbook.asks[:, 1]) if len(buy_orderbook.asks) > 0 else 0
        
        if buy_total + sell_total > 0:
            balance = min(buy_total, sell_total) / max(buy_total, sell_total)
            confidence_factors.append(balance)
        
        # 4. Latency factor
        avg_latency = (buy_orderbook.latency_ms + sell_orderbook.latency_ms) / 2
        latency_confidence = max(0, 1 - (avg_latency / 50))  # 50ms = 0 confidence
        confidence_factors.append(latency_confidence)
        
        return np.mean(confidence_factors)
    
    def _calculate_depth_for_volume(self, orders: np.ndarray, target_volume: float) -> float:
        """
        Calcula profundidade disponível para um volume alvo.
        
        Args:
            orders: Array de ordens [[price, volume], ...]
            target_volume: Volume alvo
            
        Returns:
            Volume disponível até o alvo
        """
        if len(orders) == 0:
            return 0
        
        cumulative_volume = 0
        for price, volume in orders:
            cumulative_volume += volume
            if cumulative_volume >= target_volume:
                return target_volume
        
        return cumulative_volume
    
    async def execute_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Executa uma oportunidade de arbitragem.
        
        Args:
            opportunity: Oportunidade a executar
            
        Returns:
            True se executado com sucesso
        """
        start_time = time.perf_counter()
        
        # Check if already executing this symbol
        if self.executing.get(opportunity.symbol, False):
            logger.warning(f"Already executing {opportunity.symbol}")
            return False
        
        # Check cooldown
        last_exec = self.last_execution.get(opportunity.symbol, 0)
        if time.time() - last_exec < self.execution_cooldown:
            return False
        
        # Validate opportunity is still valid
        if not opportunity.is_profitable:
            logger.info(f"Opportunity no longer profitable: {opportunity.symbol}")
            return False
        
        # Check confidence threshold
        if opportunity.confidence < self.confidence_threshold:
            logger.info(f"Low confidence: {opportunity.confidence:.2f}")
            return False
        
        try:
            # Mark as executing
            self.executing[opportunity.symbol] = True
            
            # Pre-execution validation
            if not await self._validate_execution(opportunity):
                return False
            
            # Execute trades in parallel (if pre-funded)
            if self._has_pre_funding(opportunity.buy_exchange, opportunity.sell_exchange):
                results = await asyncio.gather(
                    self._place_buy_order(opportunity),
                    self._place_sell_order(opportunity),
                    return_exceptions=True
                )
                
                # Check results
                buy_success = not isinstance(results[0], Exception) and results[0]
                sell_success = not isinstance(results[1], Exception) and results[1]
                
                if not (buy_success and sell_success):
                    # Rollback if one side failed
                    await self._rollback_trades(opportunity, buy_success, sell_success)
                    return False
                    
            else:
                # Sequential execution if transfer needed
                buy_success = await self._place_buy_order(opportunity)
                if not buy_success:
                    return False
                
                # Transfer via BEP20 (15-30 seconds)
                transfer_success = await self._transfer_funds(opportunity)
                if not transfer_success:
                    await self._rollback_trades(opportunity, True, False)
                    return False
                
                sell_success = await self._place_sell_order(opportunity)
                if not sell_success:
                    await self._rollback_trades(opportunity, True, False)
                    return False
            
            # Update statistics
            execution_time = (time.perf_counter() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            self.stats.opportunities_executed += 1
            self.stats.total_volume_usd += opportunity.executable_volume * opportunity.buy_price
            self.stats.total_profit_usd += opportunity.net_profit_usd
            
            # Update success rate
            if self.stats.opportunities_detected > 0:
                self.stats.success_rate = self.stats.opportunities_executed / self.stats.opportunities_detected
            
            logger.info(
                f"Executed arbitrage",
                symbol=opportunity.symbol,
                profit_usd=f"${opportunity.net_profit_usd:.2f}",
                execution_ms=f"{execution_time:.2f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
            
        finally:
            # Clear execution flag
            self.executing[opportunity.symbol] = False
            self.last_execution[opportunity.symbol] = time.time()
    
    async def _validate_execution(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Valida se a execução é viável.
        
        Args:
            opportunity: Oportunidade a validar
            
        Returns:
            True se válida para execução
        """
        # Get fresh orderbooks
        buy_ob = self.connector.connectors[opportunity.buy_exchange].get_orderbook(opportunity.symbol)
        sell_ob = self.connector.connectors[opportunity.sell_exchange].get_orderbook(opportunity.symbol)
        
        if not buy_ob or not sell_ob:
            return False
        
        # Check if prices are still favorable
        buy_price, buy_vol = buy_ob.best_ask
        sell_price, sell_vol = sell_ob.best_bid
        
        if buy_price <= 0 or sell_price <= 0:
            return False
        
        # Recalculate profit with fresh prices
        spread_pct = ((sell_price - buy_price) / buy_price) * 100
        total_fees = sum(opportunity.fees.values()) * 100
        net_profit_pct = spread_pct - total_fees
        
        # Still profitable?
        return net_profit_pct >= self.min_profit_pct
    
    async def _place_buy_order(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Coloca ordem de compra.
        
        Args:
            opportunity: Oportunidade sendo executada
            
        Returns:
            True se sucesso
        """
        # Simulated for now - integrate with actual exchange API
        logger.info(
            f"Placing BUY order",
            exchange=opportunity.buy_exchange,
            symbol=opportunity.symbol,
            price=opportunity.buy_price,
            volume=opportunity.executable_volume
        )
        
        # Simulate execution delay
        await asyncio.sleep(0.001)  # 1ms execution
        
        return True
    
    async def _place_sell_order(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Coloca ordem de venda.
        
        Args:
            opportunity: Oportunidade sendo executada
            
        Returns:
            True se sucesso
        """
        # Simulated for now
        logger.info(
            f"Placing SELL order",
            exchange=opportunity.sell_exchange,
            symbol=opportunity.symbol,
            price=opportunity.sell_price,
            volume=opportunity.executable_volume
        )
        
        # Simulate execution delay
        await asyncio.sleep(0.001)  # 1ms execution
        
        return True
    
    async def _transfer_funds(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Transfere fundos via BEP20.
        
        Args:
            opportunity: Oportunidade sendo executada
            
        Returns:
            True se sucesso
        """
        logger.info(
            f"Transferring funds via BEP20",
            from_exchange=opportunity.buy_exchange,
            to_exchange=opportunity.sell_exchange,
            amount=opportunity.executable_volume * opportunity.buy_price
        )
        
        # BEP20 transfer time: 15-30 seconds
        # In production, monitor blockchain for confirmation
        await asyncio.sleep(0.020)  # Simulated 20ms for now
        
        return True
    
    async def _rollback_trades(
        self, 
        opportunity: ArbitrageOpportunity,
        buy_executed: bool,
        sell_executed: bool
    ) -> None:
        """
        Reverte trades em caso de falha parcial.
        
        Args:
            opportunity: Oportunidade que falhou
            buy_executed: Se a compra foi executada
            sell_executed: Se a venda foi executada
        """
        if buy_executed and not sell_executed:
            # Need to sell what we bought
            logger.warning(f"Rolling back buy order for {opportunity.symbol}")
            # Place market sell order on buy exchange
            
        elif sell_executed and not buy_executed:
            # Need to buy back what we sold
            logger.warning(f"Rolling back sell order for {opportunity.symbol}")
            # Place market buy order on sell exchange
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de arbitragem.
        
        Returns:
            Dicionário com estatísticas
        """
        avg_latency = np.mean(self.detection_times) if self.detection_times else 0
        avg_execution = np.mean(self.execution_times) if self.execution_times else 0
        
        return {
            'opportunities_detected': self.stats.opportunities_detected,
            'opportunities_executed': self.stats.opportunities_executed,
            'total_volume_usd': self.stats.total_volume_usd,
            'total_profit_usd': self.stats.total_profit_usd,
            'success_rate': self.stats.success_rate,
            'average_detection_ms': round(avg_latency, 2),
            'average_execution_ms': round(avg_execution, 2),
            'best_opportunity': {
                'symbol': self.stats.best_opportunity.symbol,
                'profit_pct': self.stats.best_opportunity.net_profit_pct,
                'profit_usd': self.stats.best_opportunity.net_profit_usd,
            } if self.stats.best_opportunity else None,
            'last_update': self.stats.last_update
        }
    
    async def run_continuous(self, symbols: List[str], interval: float = 0.1) -> None:
        """
        Executa escaneamento contínuo de oportunidades.
        
        Args:
            symbols: Símbolos para monitorar
            interval: Intervalo entre scans em segundos
        """
        logger.info(f"Starting continuous arbitrage scanning for {symbols}")
        
        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_opportunities(symbols)
                
                if opportunities:
                    # Execute best opportunity if profitable
                    best = opportunities[0]
                    if best.is_profitable and best.confidence >= self.confidence_threshold:
                        await self.execute_opportunity(best)
                
                # Wait before next scan
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(1)


# Optimized helper functions using Numba
@jit(nopython=True)
def calculate_arbitrage_profit(
    buy_price: float,
    sell_price: float,
    volume: float,
    buy_fee: float,
    sell_fee: float,
    transfer_fee: float = 0
) -> Tuple[float, float]:
    """
    Calcula lucro de arbitragem usando Numba.
    
    Args:
        buy_price: Preço de compra
        sell_price: Preço de venda
        volume: Volume a executar
        buy_fee: Taxa de compra (decimal)
        sell_fee: Taxa de venda (decimal)
        transfer_fee: Taxa de transferência (decimal)
        
    Returns:
        (lucro_percentual, lucro_usd)
    """
    if buy_price <= 0:
        return (0.0, 0.0)
    
    # Calculate costs
    buy_cost = buy_price * volume * (1 + buy_fee)
    transfer_cost = buy_price * volume * transfer_fee
    
    # Calculate revenue
    sell_revenue = sell_price * volume * (1 - sell_fee)
    
    # Calculate profit
    total_cost = buy_cost + transfer_cost
    profit_usd = sell_revenue - total_cost
    profit_pct = (profit_usd / total_cost) * 100 if total_cost > 0 else 0
    
    return (profit_pct, profit_usd)


@jit(nopython=True)
def find_optimal_volume(
    buy_orders: np.ndarray,
    sell_orders: np.ndarray,
    max_volume: float,
    buy_fee: float,
    sell_fee: float
) -> float:
    """
    Encontra volume ótimo para maximizar lucro.
    
    Args:
        buy_orders: Ordens de compra [[price, volume], ...]
        sell_orders: Ordens de venda [[price, volume], ...]
        max_volume: Volume máximo permitido
        buy_fee: Taxa de compra
        sell_fee: Taxa de venda
        
    Returns:
        Volume ótimo
    """
    if len(buy_orders) == 0 or len(sell_orders) == 0:
        return 0.0
    
    optimal_volume = 0.0
    max_profit = 0.0
    cumulative_volume = 0.0
    
    # Try different volume levels
    for i in range(min(len(buy_orders), len(sell_orders))):
        buy_price = buy_orders[i, 0]
        sell_price = sell_orders[i, 0]
        
        level_volume = min(
            buy_orders[i, 1],
            sell_orders[i, 1],
            max_volume - cumulative_volume
        )
        
        if level_volume <= 0:
            break
        
        # Calculate profit at this level
        profit_pct, profit_usd = calculate_arbitrage_profit(
            buy_price, sell_price, level_volume,
            buy_fee, sell_fee, 0
        )
        
        if profit_usd > max_profit:
            max_profit = profit_usd
            optimal_volume = cumulative_volume + level_volume
        
        cumulative_volume += level_volume
        
        if cumulative_volume >= max_volume:
            break
    
    return optimal_volume


if __name__ == "__main__":
    # Test spatial arbitrage engine
    async def test():
        """Test spatial arbitrage detection and execution."""
        
        from core.data_collector.exchange_connector import MultiExchangeConnector
        
        # Initialize multi-exchange connector
        exchanges = [Exchange.BINANCE, Exchange.OKX]
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        connector = MultiExchangeConnector(exchanges, symbols)
        await connector.initialize()
        
        # Initialize arbitrage engine
        arbitrage = SpatialArbitrageEngine(connector)
        
        # Run for 60 seconds
        scan_task = asyncio.create_task(
            arbitrage.run_continuous(symbols, interval=0.1)
        )
        
        # Monitor stats
        for _ in range(60):
            await asyncio.sleep(1)
            stats = arbitrage.get_stats()
            print(f"Stats: {stats}")
        
        # Cancel scanning
        scan_task.cancel()
        
        # Close connections
        await connector.close()
    
    # Run test
    asyncio.run(test())