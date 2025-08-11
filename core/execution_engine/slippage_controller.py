"""
Slippage Controller - Controle e prevenção de slippage
Monitora e limita slippage para proteger capital
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

import numpy as np
from numba import jit
import redis.asyncio as redis

from config.settings import settings
from core.data_collector.exchange_connector import MultiExchangeConnector


logger = logging.getLogger(__name__)


class SlippageLevel(Enum):
    """Níveis de slippage."""
    MINIMAL = "minimal"  # < 5 bps
    LOW = "low"  # 5-10 bps
    MODERATE = "moderate"  # 10-20 bps
    HIGH = "high"  # 20-50 bps
    EXTREME = "extreme"  # > 50 bps


@dataclass
class SlippageLimit:
    """Limites de slippage."""
    
    # Absolute limits
    max_slippage_bps: float = 10  # 0.1%
    emergency_stop_bps: float = 50  # 0.5%
    
    # Per-trade limits
    max_single_trade_slippage_bps: float = 20
    
    # Cumulative limits
    max_daily_slippage_usd: float = 100
    max_hourly_slippage_usd: float = 20
    
    # Dynamic adjustments
    adaptive_enabled: bool = True
    min_confidence_for_relaxation: float = 0.9


@dataclass
class SlippageMetrics:
    """Métricas de slippage."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Current metrics
    current_slippage_bps: float = 0
    avg_slippage_bps: float = 0
    max_slippage_bps: float = 0
    
    # Cumulative metrics
    total_slippage_usd: float = 0
    hourly_slippage_usd: float = 0
    daily_slippage_usd: float = 0
    
    # Statistical metrics
    slippage_volatility: float = 0
    slippage_trend: float = 0  # Positive = increasing
    
    # Breakdown by exchange
    slippage_by_exchange: Dict[str, float] = field(default_factory=dict)
    
    # Risk score (0-1)
    risk_score: float = 0
    
    @property
    def level(self) -> SlippageLevel:
        """Retorna nível atual de slippage."""
        if self.current_slippage_bps < 5:
            return SlippageLevel.MINIMAL
        elif self.current_slippage_bps < 10:
            return SlippageLevel.LOW
        elif self.current_slippage_bps < 20:
            return SlippageLevel.MODERATE
        elif self.current_slippage_bps < 50:
            return SlippageLevel.HIGH
        else:
            return SlippageLevel.EXTREME


class SlippagePredictor:
    """
    Modelo preditivo de slippage.
    Usa histórico e condições de mercado para prever slippage.
    """
    
    def __init__(self):
        """Inicializa preditor de slippage."""
        # Model parameters
        self.lookback_window = 100
        self.feature_count = 10
        
        # Simple linear model weights (will be updated online)
        self.weights = np.random.randn(self.feature_count) * 0.01
        self.bias = 0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_momentum = np.zeros(self.feature_count)
        
        # Performance tracking
        self.predictions = deque(maxlen=1000)
        self.actuals = deque(maxlen=1000)
        self.mse_history = deque(maxlen=100)
    
    def extract_features(
        self,
        order_size: float,
        orderbook_depth: float,
        spread: float,
        volatility: float,
        volume: float,
        time_of_day: float,
        exchange_id: int,
        order_type: int,
        market_trend: float,
        liquidity_ratio: float
    ) -> np.ndarray:
        """
        Extrai features para predição.
        
        Returns:
            Array de features
        """
        features = np.array([
            order_size / orderbook_depth if orderbook_depth > 0 else 1,
            spread,
            volatility,
            np.log1p(volume),
            np.sin(2 * np.pi * time_of_day / 86400),  # Cyclical time
            np.cos(2 * np.pi * time_of_day / 86400),
            exchange_id,
            order_type,
            market_trend,
            liquidity_ratio
        ])
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    @jit(nopython=True)
    def predict_slippage_fast(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> float:
        """
        Predição rápida de slippage com Numba.
        
        Args:
            features: Features normalizadas
            weights: Pesos do modelo
            bias: Bias
            
        Returns:
            Slippage predito em bps
        """
        linear_output = np.dot(features, weights) + bias
        
        # Apply sigmoid to bound output
        sigmoid_output = 1 / (1 + np.exp(-linear_output))
        
        # Scale to basis points (0-100)
        return sigmoid_output * 100
    
    def update_model(
        self,
        features: np.ndarray,
        actual_slippage: float
    ) -> None:
        """
        Atualiza modelo com novo exemplo.
        
        Args:
            features: Features do exemplo
            actual_slippage: Slippage real em bps
        """
        # Get prediction
        predicted = self.predict_slippage_fast(features, self.weights, self.bias)
        
        # Calculate error
        error = (actual_slippage / 100) - (predicted / 100)
        
        # Update weights using momentum SGD
        gradient = error * features
        self.weight_momentum = self.momentum * self.weight_momentum + self.learning_rate * gradient
        self.weights += self.weight_momentum
        
        # Update bias
        self.bias += self.learning_rate * error
        
        # Track performance
        self.predictions.append(predicted)
        self.actuals.append(actual_slippage)
        
        if len(self.predictions) > 10:
            recent_mse = np.mean(
                (np.array(list(self.predictions)[-10:]) - np.array(list(self.actuals)[-10:])) ** 2
            )
            self.mse_history.append(recent_mse)


class SlippageController:
    """
    Controlador principal de slippage.
    Monitora, prediz e controla slippage em tempo real.
    """
    
    def __init__(self, connector: MultiExchangeConnector):
        """
        Inicializa controlador de slippage.
        
        Args:
            connector: Conector multi-exchange
        """
        self.connector = connector
        
        # Slippage limits
        self.limits = SlippageLimit()
        
        # Current metrics
        self.current_metrics = SlippageMetrics()
        
        # Historical data
        self.slippage_history = deque(maxlen=10000)
        self.hourly_history = deque(maxlen=24)
        self.daily_total = 0
        
        # Predictor
        self.predictor = SlippagePredictor()
        
        # Control state
        self.emergency_stop = False
        self.restricted_mode = False
        self.last_reset_time = time.time()
        
        # Exchange-specific tracking
        self.exchange_slippage = defaultdict(lambda: deque(maxlen=100))
        
        # Redis for persistence
        self.redis_client = None
        self._init_redis()
        
        # Start monitoring
        asyncio.create_task(self._monitor_loop())
    
    def _init_redis(self) -> None:
        """Inicializa conexão Redis."""
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
    
    async def check_pre_execution(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_size: float,
        order_type: str
    ) -> Tuple[bool, float, str]:
        """
        Verifica slippage antes da execução.
        
        Args:
            exchange: Exchange
            symbol: Símbolo
            side: Lado (buy/sell)
            order_size: Tamanho da ordem
            order_type: Tipo da ordem
            
        Returns:
            (pode_executar, slippage_esperado, razão)
        """
        # Check emergency stop
        if self.emergency_stop:
            return False, 0, "Emergency stop active"
        
        # Check daily limit
        if self.current_metrics.daily_slippage_usd >= self.limits.max_daily_slippage_usd:
            return False, 0, "Daily slippage limit reached"
        
        # Get orderbook
        connector = self.connector.connectors.get(exchange)
        if not connector:
            return False, 0, "Exchange not connected"
        
        orderbook = connector.get_orderbook(symbol)
        if not orderbook:
            return False, 0, "No orderbook available"
        
        # Calculate expected slippage
        expected_slippage = await self._calculate_expected_slippage(
            orderbook, side, order_size, order_type
        )
        
        # Check against limits
        if expected_slippage > self.limits.max_single_trade_slippage_bps:
            return False, expected_slippage, f"Expected slippage too high: {expected_slippage:.2f} bps"
        
        # Check restricted mode
        if self.restricted_mode and expected_slippage > self.limits.max_slippage_bps:
            return False, expected_slippage, "Restricted mode - tighter limits"
        
        return True, expected_slippage, "OK"
    
    async def _calculate_expected_slippage(
        self,
        orderbook: Any,
        side: str,
        order_size: float,
        order_type: str
    ) -> float:
        """
        Calcula slippage esperado.
        
        Args:
            orderbook: Orderbook
            side: Lado
            order_size: Tamanho
            order_type: Tipo
            
        Returns:
            Slippage esperado em bps
        """
        if order_type == 'market':
            # Calculate market order slippage
            if side == 'buy':
                levels = orderbook.asks
                reference_price = orderbook.best_ask[0]
            else:
                levels = orderbook.bids
                reference_price = orderbook.best_bid[0]
            
            if reference_price <= 0:
                return 100  # Maximum slippage
            
            # Walk through orderbook
            remaining_size = order_size
            total_cost = 0
            
            for price, size in levels[:10]:  # Check first 10 levels
                fill_size = min(remaining_size, size)
                total_cost += fill_size * price
                remaining_size -= fill_size
                
                if remaining_size <= 0:
                    break
            
            if remaining_size > 0:
                # Not enough liquidity
                return 100  # Maximum slippage
            
            # Calculate average price
            avg_price = total_cost / order_size
            
            # Calculate slippage in bps
            slippage_bps = abs(avg_price - reference_price) / reference_price * 10000
            
        else:  # Limit order
            # Use predictor for limit orders
            features = self.predictor.extract_features(
                order_size=order_size,
                orderbook_depth=orderbook.total_bid_volume if side == 'sell' else orderbook.total_ask_volume,
                spread=orderbook.spread,
                volatility=0.01,  # Placeholder
                volume=orderbook.total_bid_volume + orderbook.total_ask_volume,
                time_of_day=time.time() % 86400,
                exchange_id=0,  # Placeholder
                order_type=0 if order_type == 'limit' else 1,
                market_trend=0,  # Placeholder
                liquidity_ratio=order_size / orderbook.total_bid_volume if orderbook.total_bid_volume > 0 else 1
            )
            
            slippage_bps = self.predictor.predict_slippage_fast(
                features, self.predictor.weights, self.predictor.bias
            )
        
        return slippage_bps
    
    async def record_execution(
        self,
        exchange: str,
        symbol: str,
        expected_price: float,
        actual_price: float,
        size: float,
        side: str
    ) -> None:
        """
        Registra execução e atualiza métricas.
        
        Args:
            exchange: Exchange
            symbol: Símbolo
            expected_price: Preço esperado
            actual_price: Preço real
            size: Tamanho
            side: Lado
        """
        # Calculate actual slippage
        if expected_price > 0:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
        else:
            slippage_bps = 0
        
        # Calculate USD impact
        slippage_usd = size * abs(actual_price - expected_price)
        
        # Update history
        self.slippage_history.append({
            'timestamp': time.time(),
            'exchange': exchange,
            'symbol': symbol,
            'slippage_bps': slippage_bps,
            'slippage_usd': slippage_usd,
            'size': size
        })
        
        # Update exchange-specific tracking
        self.exchange_slippage[exchange].append(slippage_bps)
        
        # Update current metrics
        await self._update_metrics()
        
        # Update predictor model
        # (Would need actual features from pre-execution)
        
        # Check for alerts
        if slippage_bps > self.limits.max_slippage_bps:
            logger.warning(
                f"High slippage detected",
                exchange=exchange,
                symbol=symbol,
                slippage_bps=f"{slippage_bps:.2f}"
            )
        
        # Check for emergency stop
        if slippage_bps > self.limits.emergency_stop_bps:
            await self.trigger_emergency_stop("Extreme slippage detected")
    
    async def _update_metrics(self) -> None:
        """Atualiza métricas de slippage."""
        current_time = time.time()
        
        # Calculate time-based metrics
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        # Filter recent history
        recent_history = [
            h for h in self.slippage_history
            if h['timestamp'] > hour_ago
        ]
        
        if recent_history:
            # Current metrics
            self.current_metrics.current_slippage_bps = recent_history[-1]['slippage_bps']
            self.current_metrics.avg_slippage_bps = np.mean([h['slippage_bps'] for h in recent_history])
            self.current_metrics.max_slippage_bps = max(h['slippage_bps'] for h in recent_history)
            
            # Cumulative metrics
            self.current_metrics.hourly_slippage_usd = sum(
                h['slippage_usd'] for h in recent_history
            )
            
            daily_history = [
                h for h in self.slippage_history
                if h['timestamp'] > day_ago
            ]
            self.current_metrics.daily_slippage_usd = sum(
                h['slippage_usd'] for h in daily_history
            )
            
            # Statistical metrics
            slippage_values = [h['slippage_bps'] for h in recent_history]
            self.current_metrics.slippage_volatility = np.std(slippage_values)
            
            # Trend (simple linear regression)
            if len(slippage_values) > 2:
                x = np.arange(len(slippage_values))
                self.current_metrics.slippage_trend = np.polyfit(x, slippage_values, 1)[0]
            
            # Exchange breakdown
            for exchange in set(h['exchange'] for h in recent_history):
                exchange_slippages = [
                    h['slippage_bps'] for h in recent_history
                    if h['exchange'] == exchange
                ]
                self.current_metrics.slippage_by_exchange[exchange] = np.mean(exchange_slippages)
            
            # Risk score
            self.current_metrics.risk_score = self._calculate_risk_score()
        
        # Update timestamp
        self.current_metrics.timestamp = current_time
    
    def _calculate_risk_score(self) -> float:
        """
        Calcula score de risco baseado em métricas.
        
        Returns:
            Risk score (0-1)
        """
        score = 0
        
        # Current slippage contribution
        score += min(self.current_metrics.current_slippage_bps / 50, 0.3)
        
        # Average slippage contribution
        score += min(self.current_metrics.avg_slippage_bps / 30, 0.3)
        
        # Volatility contribution
        score += min(self.current_metrics.slippage_volatility / 20, 0.2)
        
        # Trend contribution
        if self.current_metrics.slippage_trend > 0:
            score += min(self.current_metrics.slippage_trend / 10, 0.2)
        
        return min(score, 1.0)
    
    async def trigger_emergency_stop(self, reason: str) -> None:
        """
        Ativa parada de emergência.
        
        Args:
            reason: Razão da parada
        """
        self.emergency_stop = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Notify all systems
        # In production, would send alerts
        
        # Schedule automatic reset after cooldown
        asyncio.create_task(self._reset_emergency_stop())
    
    async def _reset_emergency_stop(self) -> None:
        """Reseta parada de emergência após cooldown."""
        await asyncio.sleep(300)  # 5 minutes cooldown
        self.emergency_stop = False
        logger.info("Emergency stop reset")
    
    async def adjust_limits(
        self,
        market_conditions: Dict[str, Any]
    ) -> None:
        """
        Ajusta limites baseado em condições de mercado.
        
        Args:
            market_conditions: Condições atuais do mercado
        """
        if not self.limits.adaptive_enabled:
            return
        
        volatility = market_conditions.get('volatility', 0.01)
        liquidity = market_conditions.get('liquidity', 1.0)
        confidence = market_conditions.get('confidence', 0.5)
        
        # High volatility = tighter limits
        if volatility > 0.02:  # 2% volatility
            self.limits.max_slippage_bps = 5
            self.restricted_mode = True
        elif volatility < 0.005 and confidence > self.limits.min_confidence_for_relaxation:
            # Low volatility and high confidence = relax limits
            self.limits.max_slippage_bps = 15
            self.restricted_mode = False
        else:
            # Normal conditions
            self.limits.max_slippage_bps = 10
            self.restricted_mode = False
        
        logger.debug(
            f"Adjusted slippage limits",
            max_bps=self.limits.max_slippage_bps,
            restricted=self.restricted_mode
        )
    
    async def _monitor_loop(self) -> None:
        """Loop de monitoramento contínuo."""
        while True:
            try:
                # Update metrics
                await self._update_metrics()
                
                # Check for issues
                if self.current_metrics.risk_score > 0.8:
                    logger.warning(f"High slippage risk: {self.current_metrics.risk_score:.2f}")
                
                # Reset daily counter at midnight
                current_hour = time.localtime().tm_hour
                if current_hour == 0 and time.time() - self.last_reset_time > 3600:
                    self.daily_total = 0
                    self.last_reset_time = time.time()
                    logger.info("Daily slippage counter reset")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in slippage monitor: {e}")
                await asyncio.sleep(60)
    
    def get_slippage_report(self) -> Dict[str, Any]:
        """Retorna relatório de slippage."""
        return {
            'current_metrics': {
                'level': self.current_metrics.level.value,
                'current_bps': self.current_metrics.current_slippage_bps,
                'avg_bps': self.current_metrics.avg_slippage_bps,
                'max_bps': self.current_metrics.max_slippage_bps,
                'hourly_usd': self.current_metrics.hourly_slippage_usd,
                'daily_usd': self.current_metrics.daily_slippage_usd,
                'risk_score': self.current_metrics.risk_score
            },
            'limits': {
                'max_bps': self.limits.max_slippage_bps,
                'emergency_stop_bps': self.limits.emergency_stop_bps,
                'daily_limit_usd': self.limits.max_daily_slippage_usd
            },
            'status': {
                'emergency_stop': self.emergency_stop,
                'restricted_mode': self.restricted_mode
            },
            'by_exchange': dict(self.current_metrics.slippage_by_exchange),
            'predictor_mse': np.mean(list(self.predictor.mse_history)) if self.predictor.mse_history else 0
        }