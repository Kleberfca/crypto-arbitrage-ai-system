"""
Statistical Arbitrage Engine - Pairs Trading e Mean Reversion
Implementa cointegração, z-score e Kalman filter
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pykalman import KalmanFilter
import redis.asyncio as redis

from config.settings import settings
from core.arbitrage_engine.base_arbitrage import (
    BaseArbitrageEngine, 
    BaseOpportunity, 
    OpportunityType
)
from core.data_collector.exchange_connector import MultiExchangeConnector


logger = logging.getLogger(__name__)


@dataclass
class StatisticalOpportunity(BaseOpportunity):
    """Oportunidade de arbitragem estatística."""
    
    # Pair information
    pair1_symbol: str
    pair2_symbol: str
    exchange: str
    
    # Statistical metrics
    zscore: float
    hedge_ratio: float
    cointegration_pvalue: float
    half_life: float
    mean_reversion_time: float
    
    # Trading details
    pair1_price: float
    pair2_price: float
    pair1_volume: float
    pair2_volume: float
    position_type: str  # 'long_spread' or 'short_spread'
    
    # Entry/Exit levels
    entry_zscore: float
    exit_zscore: float
    stop_loss_zscore: float


class StatisticalArbitrageEngine(BaseArbitrageEngine):
    """
    Motor de arbitragem estatística com pairs trading.
    Usa cointegração, z-score e Kalman filter para hedge ratio dinâmico.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa engine de arbitragem estatística.
        
        Args:
            connector: Conector multi-exchange
        """
        super().__init__(connector, OpportunityType.STATISTICAL)
        
        # Statistical parameters
        self.min_zscore_entry = 2.0
        self.max_zscore_exit = 0.5
        self.stop_loss_zscore = 3.0
        self.lookback_period = 100
        self.cointegration_threshold = 0.05
        self.min_half_life = 1  # hours
        self.max_half_life = 24  # hours
        
        # Price history
        self.price_history = {}
        self.max_history_size = 500
        
        # Cointegrated pairs cache
        self.cointegrated_pairs = {}
        self.last_cointegration_test = {}
        self.cointegration_test_interval = 3600  # 1 hour
        
        # Kalman filter for dynamic hedge ratio
        self.kalman_filters = {}
        
        # Redis cache
        self.redis_client = None
        self._init_redis()
        
        # Active positions
        self.active_positions = {}
        
    def _init_redis(self) -> None:
        """Inicializa conexão Redis para cache."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def scan_opportunities(
        self,
        symbols: List[str]
    ) -> List[StatisticalOpportunity]:
        """
        Escaneia oportunidades de pairs trading.
        
        Args:
            symbols: Lista de símbolos
            
        Returns:
            Lista de oportunidades estatísticas
        """
        start_time = time.perf_counter()
        opportunities = []
        
        # Update price history
        await self._update_price_history(symbols)
        
        # Find cointegrated pairs
        pairs = await self._find_cointegrated_pairs(symbols)
        
        # Check each pair for trading signals
        for pair in pairs:
            opportunity = await self._check_pair_opportunity(pair)
            if opportunity and opportunity.is_profitable:
                opportunities.append(opportunity)
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        
        # Log performance
        scan_time = (time.perf_counter() - start_time) * 1000
        if scan_time > 10:
            logger.warning(f"Slow statistical scan: {scan_time:.2f}ms")
        
        return opportunities
    
    async def _update_price_history(self, symbols: List[str]) -> None:
        """
        Atualiza histórico de preços.
        
        Args:
            symbols: Lista de símbolos
        """
        for symbol in symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.max_history_size)
            
            # Get current price from all exchanges
            for exchange in self.connector.exchanges:
                orderbook = self.connector.connectors[exchange.value].get_orderbook(symbol)
                if orderbook:
                    self.price_history[symbol].append({
                        'timestamp': time.time(),
                        'exchange': exchange.value,
                        'price': orderbook.mid_price
                    })
    
    async def _find_cointegrated_pairs(
        self,
        symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Encontra pares cointegrados.
        
        Args:
            symbols: Lista de símbolos
            
        Returns:
            Lista de pares cointegrados
        """
        pairs = []
        
        # Check cache first
        cache_key = f"cointegrated_pairs:{':'.join(sorted(symbols))}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    import json
                    return json.loads(cached)
            except:
                pass
        
        # Test all combinations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = await self._test_cointegration(symbols[i], symbols[j])
                if pair:
                    pairs.append(pair)
        
        # Cache results
        if self.redis_client and pairs:
            try:
                import json
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(pairs)
                )
            except:
                pass
        
        return pairs
    
    async def _test_cointegration(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Testa cointegração entre dois ativos.
        
        Args:
            symbol1: Primeiro símbolo
            symbol2: Segundo símbolo
            
        Returns:
            Dados do par se cointegrado
        """
        # Check if we have enough history
        if (symbol1 not in self.price_history or 
            symbol2 not in self.price_history):
            return None
        
        history1 = list(self.price_history[symbol1])
        history2 = list(self.price_history[symbol2])
        
        if len(history1) < self.lookback_period or len(history2) < self.lookback_period:
            return None
        
        # Extract prices
        prices1 = np.array([h['price'] for h in history1[-self.lookback_period:]])
        prices2 = np.array([h['price'] for h in history2[-self.lookback_period:]])
        
        # Remove any NaN or inf values
        mask = np.isfinite(prices1) & np.isfinite(prices2)
        prices1 = prices1[mask]
        prices2 = prices2[mask]
        
        if len(prices1) < 50:  # Minimum samples for cointegration test
            return None
        
        # Test cointegration using Engle-Granger
        try:
            score, pvalue, _ = coint(prices1, prices2)
            
            if pvalue < self.cointegration_threshold:
                # Calculate hedge ratio using OLS
                hedge_ratio = self._calculate_hedge_ratio_ols(prices1, prices2)
                
                # Calculate half-life
                spread = prices1 - hedge_ratio * prices2
                half_life = self._calculate_half_life(spread)
                
                # Check if half-life is in acceptable range
                if self.min_half_life <= half_life <= self.max_half_life:
                    return {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'half_life': half_life
                    }
        except Exception as e:
            logger.debug(f"Cointegration test failed for {symbol1}/{symbol2}: {e}")
        
        return None
    
    def _calculate_hedge_ratio_ols(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray
    ) -> float:
        """
        Calcula hedge ratio usando OLS.
        
        Args:
            prices1: Preços do ativo 1
            prices2: Preços do ativo 2
            
        Returns:
            Hedge ratio
        """
        # Simple OLS: prices1 = alpha + beta * prices2
        beta = np.cov(prices1, prices2)[0, 1] / np.var(prices2)
        return beta
    
    def _calculate_hedge_ratio_kalman(
        self,
        symbol1: str,
        symbol2: str,
        prices1: np.ndarray,
        prices2: np.ndarray
    ) -> float:
        """
        Calcula hedge ratio dinâmico usando Kalman Filter.
        
        Args:
            symbol1: Símbolo 1
            symbol2: Símbolo 2
            prices1: Preços do ativo 1
            prices2: Preços do ativo 2
            
        Returns:
            Hedge ratio atualizado
        """
        pair_key = f"{symbol1}_{symbol2}"
        
        if pair_key not in self.kalman_filters:
            # Initialize Kalman filter
            delta = 1e-5
            trans_cov = delta / (1 - delta) * np.eye(2)
            
            self.kalman_filters[pair_key] = KalmanFilter(
                n_dim_obs=1,
                n_dim_state=2,
                initial_state_mean=np.zeros(2),
                initial_state_covariance=np.ones((2, 2)),
                transition_matrices=np.eye(2),
                transition_covariance=trans_cov,
                observation_covariance=1.0
            )
        
        kf = self.kalman_filters[pair_key]
        
        # Prepare observations
        obs_mat = np.vstack([prices2, np.ones(len(prices2))]).T
        
        # Update and get state means
        state_means, _ = kf.filter(prices1)
        
        # Return latest hedge ratio
        return state_means[-1, 0] if len(state_means) > 0 else 1.0
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calcula meia-vida do spread (Ornstein-Uhlenbeck).
        
        Args:
            spread: Série do spread
            
        Returns:
            Meia-vida em horas
        """
        # Calculate lag
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread_lag
        
        # OLS regression
        if len(spread_lag) > 0:
            theta = np.polyfit(spread_lag, spread_diff, 1)[0]
            if theta < 0:
                half_life = -np.log(2) / theta
                return half_life / 3600  # Convert to hours
        
        return 12.0  # Default 12 hours
    
    @jit(nopython=True)
    def _calculate_zscore_fast(
        self,
        spread: np.ndarray,
        lookback: int
    ) -> float:
        """
        Calcula z-score otimizado com Numba.
        
        Args:
            spread: Array do spread
            lookback: Período para cálculo
            
        Returns:
            Z-score atual
        """
        if len(spread) < lookback:
            return 0.0
        
        recent_spread = spread[-lookback:]
        mean = np.mean(recent_spread)
        std = np.std(recent_spread)
        
        if std > 0:
            return (spread[-1] - mean) / std
        return 0.0
    
    async def _check_pair_opportunity(
        self,
        pair: Dict[str, Any]
    ) -> Optional[StatisticalOpportunity]:
        """
        Verifica oportunidade em um par cointegrado.
        
        Args:
            pair: Dados do par
            
        Returns:
            Oportunidade se existir
        """
        symbol1 = pair['symbol1']
        symbol2 = pair['symbol2']
        
        # Get current prices
        orderbook1 = self.connector.connectors[list(self.connector.exchanges)[0].value].get_orderbook(symbol1)
        orderbook2 = self.connector.connectors[list(self.connector.exchanges)[0].value].get_orderbook(symbol2)
        
        if not orderbook1 or not orderbook2:
            return None
        
        # Get price history
        history1 = list(self.price_history[symbol1])[-self.lookback_period:]
        history2 = list(self.price_history[symbol2])[-self.lookback_period:]
        
        if len(history1) < 20 or len(history2) < 20:
            return None
        
        prices1 = np.array([h['price'] for h in history1])
        prices2 = np.array([h['price'] for h in history2])
        
        # Calculate dynamic hedge ratio with Kalman filter
        hedge_ratio = self._calculate_hedge_ratio_kalman(
            symbol1, symbol2, prices1, prices2
        )
        
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        
        # Calculate z-score
        zscore = self._calculate_zscore_fast(spread, min(50, len(spread)))
        
        # Check for entry signals
        position_type = None
        if zscore > self.min_zscore_entry:
            position_type = 'short_spread'  # Spread is high, expect reversion down
        elif zscore < -self.min_zscore_entry:
            position_type = 'long_spread'  # Spread is low, expect reversion up
        else:
            return None  # No signal
        
        # Calculate expected profit
        expected_return = abs(zscore) * 0.5  # Assume 50% mean reversion
        expected_profit_pct = expected_return * 2  # Approximate conversion
        
        # Estimate execution time
        execution_time = pair['half_life'] * 3600 * 0.5  # Half of half-life in seconds
        
        # Create opportunity
        opportunity = StatisticalOpportunity(
            opportunity_id=f"{symbol1}_{symbol2}_{int(time.time())}",
            opportunity_type=OpportunityType.STATISTICAL,
            expected_profit_pct=expected_profit_pct,
            expected_profit_usd=expected_profit_pct * 1000,  # Assume $1000 position
            confidence=min(0.95, 1.0 - pair['pvalue']),
            risk_score=min(abs(zscore) / 4, 1.0),  # Higher z-score = higher risk
            required_capital=1000.0,
            estimated_execution_time_ms=execution_time * 1000,
            
            # Statistical specific
            pair1_symbol=symbol1,
            pair2_symbol=symbol2,
            exchange=list(self.connector.exchanges)[0].value,
            zscore=zscore,
            hedge_ratio=hedge_ratio,
            cointegration_pvalue=pair['pvalue'],
            half_life=pair['half_life'],
            mean_reversion_time=execution_time,
            pair1_price=orderbook1.mid_price,
            pair2_price=orderbook2.mid_price,
            pair1_volume=orderbook1.best_bid[1],
            pair2_volume=orderbook2.best_bid[1],
            position_type=position_type,
            entry_zscore=zscore,
            exit_zscore=self.max_zscore_exit,
            stop_loss_zscore=self.stop_loss_zscore
        )
        
        return opportunity
    
    async def validate_opportunity(
        self,
        opportunity: StatisticalOpportunity
    ) -> Tuple[bool, str]:
        """
        Valida se oportunidade ainda é viável.
        
        Args:
            opportunity: Oportunidade a validar
            
        Returns:
            (is_valid, reason)
        """
        # Check age
        if opportunity.age_seconds > 30:
            return False, "Opportunity too old"
        
        # Check if pair is still cointegrated
        pair = await self._test_cointegration(
            opportunity.pair1_symbol,
            opportunity.pair2_symbol
        )
        
        if not pair:
            return False, "Pair no longer cointegrated"
        
        # Check current z-score
        history1 = list(self.price_history[opportunity.pair1_symbol])[-50:]
        history2 = list(self.price_history[opportunity.pair2_symbol])[-50:]
        
        prices1 = np.array([h['price'] for h in history1])
        prices2 = np.array([h['price'] for h in history2])
        
        spread = prices1 - opportunity.hedge_ratio * prices2
        current_zscore = self._calculate_zscore_fast(spread, len(spread))
        
        # Check if signal still valid
        if opportunity.position_type == 'long_spread':
            if current_zscore > -self.min_zscore_entry + 0.5:
                return False, "Z-score no longer favorable for long"
        else:
            if current_zscore < self.min_zscore_entry - 0.5:
                return False, "Z-score no longer favorable for short"
        
        return True, "Valid"
    
    async def execute_opportunity(
        self,
        opportunity: StatisticalOpportunity
    ) -> bool:
        """
        Executa pairs trade.
        
        Args:
            opportunity: Oportunidade a executar
            
        Returns:
            True se executado com sucesso
        """
        start_time = time.perf_counter()
        
        try:
            # Log execution
            logger.info(
                f"Executing statistical arbitrage",
                pair=f"{opportunity.pair1_symbol}/{opportunity.pair2_symbol}",
                zscore=f"{opportunity.zscore:.2f}",
                position=opportunity.position_type,
                hedge_ratio=f"{opportunity.hedge_ratio:.3f}"
            )
            
            # In production, would place actual orders
            # For now, simulate execution
            await asyncio.sleep(0.001)  # Simulate execution delay
            
            # Store position for monitoring
            position_key = f"{opportunity.pair1_symbol}_{opportunity.pair2_symbol}"
            self.active_positions[position_key] = {
                'opportunity': opportunity,
                'entry_time': time.time(),
                'entry_zscore': opportunity.zscore,
                'status': 'active'
            }
            
            # Update stats
            self.opportunities_executed += 1
            self.total_profit_usd += opportunity.expected_profit_usd
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            logger.info(
                f"Statistical arbitrage executed",
                execution_ms=f"{execution_time:.2f}",
                expected_profit=f"${opportunity.expected_profit_usd:.2f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute statistical arbitrage: {e}")
            return False
    
    def calculate_profit(
        self,
        opportunity: StatisticalOpportunity
    ) -> Tuple[float, float]:
        """
        Calcula lucro esperado.
        
        Args:
            opportunity: Oportunidade
            
        Returns:
            (profit_pct, profit_usd)
        """
        # Expected profit based on mean reversion
        # Assumes z-score will revert to 0
        expected_move = abs(opportunity.zscore) * 0.5  # Conservative estimate
        
        # Account for transaction costs
        fees = 0.002  # 0.1% each side
        
        profit_pct = expected_move - fees
        profit_usd = profit_pct * opportunity.required_capital
        
        return (profit_pct * 100, profit_usd)
    
    async def monitor_positions(self) -> None:
        """Monitora posições ativas para exit signals."""
        
        for position_key, position_data in list(self.active_positions.items()):
            if position_data['status'] != 'active':
                continue
            
            opportunity = position_data['opportunity']
            
            # Get current z-score
            history1 = list(self.price_history[opportunity.pair1_symbol])[-50:]
            history2 = list(self.price_history[opportunity.pair2_symbol])[-50:]
            
            if len(history1) < 20 or len(history2) < 20:
                continue
            
            prices1 = np.array([h['price'] for h in history1])
            prices2 = np.array([h['price'] for h in history2])
            
            spread = prices1 - opportunity.hedge_ratio * prices2
            current_zscore = self._calculate_zscore_fast(spread, len(spread))
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Mean reversion exit
            if abs(current_zscore) < opportunity.exit_zscore:
                should_exit = True
                exit_reason = "Mean reversion target reached"
            
            # Stop loss
            elif abs(current_zscore) > opportunity.stop_loss_zscore:
                should_exit = True
                exit_reason = "Stop loss triggered"
            
            # Time-based exit (if position too old)
            elif time.time() - position_data['entry_time'] > opportunity.half_life * 3600 * 2:
                should_exit = True
                exit_reason = "Max holding time exceeded"
            
            if should_exit:
                logger.info(
                    f"Exiting statistical position",
                    pair=position_key,
                    reason=exit_reason,
                    entry_zscore=f"{position_data['entry_zscore']:.2f}",
                    exit_zscore=f"{current_zscore:.2f}"
                )
                
                # Mark as closed
                position_data['status'] = 'closed'
                position_data['exit_time'] = time.time()
                position_data['exit_zscore'] = current_zscore