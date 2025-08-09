"""
Exchange Connector - Conexão WebSocket com as 5 exchanges permitidas
Implementa coleta de dados em tempo real com latência <20ms
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from collections import defaultdict, deque

import ccxt.pro as ccxtpro
import aiohttp
import websockets
from websockets.exceptions import WebSocketException
import numpy as np
from numba import jit
import redis.asyncio as redis
import orjson

from config.settings import settings, Exchange, get_exchange_config


# Configure structured logging
logger = logging.getLogger(__name__)


class MarketDataType(Enum):
    """Tipos de dados de mercado."""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    OHLCV = "ohlcv"
    FUNDING = "funding"


@dataclass
class OrderBookSnapshot:
    """Snapshot do order book com timestamp de alta precisão."""
    exchange: str
    symbol: str
    timestamp: float  # Unix timestamp com microsegundos
    bids: np.ndarray  # [[price, volume], ...]
    asks: np.ndarray  # [[price, volume], ...]
    sequence: Optional[int] = None
    latency_ms: float = 0.0
    
    @property
    def best_bid(self) -> tuple:
        """Retorna melhor bid (preço, volume)."""
        return (self.bids[0, 0], self.bids[0, 1]) if len(self.bids) > 0 else (0, 0)
    
    @property
    def best_ask(self) -> tuple:
        """Retorna melhor ask (preço, volume)."""
        return (self.asks[0, 0], self.asks[0, 1]) if len(self.asks) > 0 else (0, 0)
    
    @property
    def spread(self) -> float:
        """Calcula o spread."""
        bid, _ = self.best_bid
        ask, _ = self.best_ask
        return ask - bid if bid > 0 and ask > 0 else 0
    
    @property
    def mid_price(self) -> float:
        """Calcula o preço médio."""
        bid, _ = self.best_bid
        ask, _ = self.best_ask
        return (bid + ask) / 2 if bid > 0 and ask > 0 else 0


@dataclass
class TradeData:
    """Dados de trade executado."""
    exchange: str
    symbol: str
    timestamp: float
    price: float
    volume: float
    side: str  # 'buy' ou 'sell'
    trade_id: Optional[str] = None


@dataclass
class TickerData:
    """Dados de ticker."""
    exchange: str
    symbol: str
    timestamp: float
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    open_24h: float
    close: float
    change_24h: float
    change_pct_24h: float


class ExchangeConnector:
    """
    Conector unificado para as 5 exchanges via WebSocket.
    Implementa reconexão automática, cache Redis e latência <20ms.
    """
    
    def __init__(self, exchange: Exchange, symbols: List[str]):
        """
        Inicializa o conector.
        
        Args:
            exchange: Exchange enum (BINANCE, OKX, etc)
            symbols: Lista de símbolos para monitorar
        """
        self.exchange = exchange
        self.symbols = symbols
        self.config = get_exchange_config(exchange)
        self.exchange_id = exchange.value
        
        # CCXT Pro exchange object
        self.ccxt_exchange = None
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        self.ws_tasks: List[asyncio.Task] = []
        
        # Data buffers (deque for performance)
        self.orderbook_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.trade_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ticker_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Callbacks
        self.callbacks: Dict[MarketDataType, List[Callable]] = defaultdict(list)
        
        # Redis cache
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance metrics
        self.latency_history = deque(maxlen=1000)
        self.last_update_time: Dict[str, float] = {}
        self.connection_start_time: Optional[float] = None
        
        # Connection state
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = settings.ws_max_reconnect_attempts
        
    async def initialize(self) -> None:
        """Inicializa o conector e estabelece conexões."""
        start_time = time.perf_counter()
        
        try:
            # Initialize CCXT Pro exchange
            await self._init_ccxt_exchange()
            
            # Connect to Redis
            await self._init_redis()
            
            # Start WebSocket connections
            await self._connect_websockets()
            
            self.is_connected = True
            self.connection_start_time = time.time()
            
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Exchange connector initialized",
                exchange=self.exchange_id,
                symbols=self.symbols,
                latency_ms=f"{latency:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize connector: {e}")
            raise
    
    async def _init_ccxt_exchange(self) -> None:
        """Inicializa exchange CCXT Pro."""
        exchange_classes = {
            'binance': ccxtpro.binance,
            'okx': ccxtpro.okx,
            'bybit': ccxtpro.bybit,
            'kucoin': ccxtpro.kucoin,
            'htx': ccxtpro.huobi,
        }
        
        exchange_class = exchange_classes.get(self.exchange_id)
        if not exchange_class:
            raise ValueError(f"Exchange {self.exchange_id} not supported")
        
        # Get API credentials from settings
        api_config = {
            'apiKey': getattr(settings, f"{self.exchange_id}_api_key", None),
            'secret': getattr(settings, f"{self.exchange_id}_api_secret", None),
            'enableRateLimit': True,
            'rateLimit': self.config['rate_limit'],
            'options': {
                'defaultType': 'spot',
                'watchOrderBook': {'limit': 20},
                'watchTrades': {'limit': 100},
            }
        }
        
        # Add passphrase for exchanges that require it
        if self.exchange_id in ['okx', 'kucoin']:
            api_config['password'] = getattr(settings, f"{self.exchange_id}_passphrase", None)
        
        self.ccxt_exchange = exchange_class(api_config)
        
    async def _init_redis(self) -> None:
        """Inicializa conexão com Redis."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False  # Use binary for performance
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def _connect_websockets(self) -> None:
        """Estabelece conexões WebSocket para todos os símbolos."""
        tasks = []
        
        for symbol in self.symbols:
            # Start orderbook stream
            task = asyncio.create_task(
                self._watch_orderbook(symbol)
            )
            self.ws_tasks.append(task)
            tasks.append(task)
            
            # Start trades stream
            task = asyncio.create_task(
                self._watch_trades(symbol)
            )
            self.ws_tasks.append(task)
            tasks.append(task)
            
            # Start ticker stream
            task = asyncio.create_task(
                self._watch_ticker(symbol)
            )
            self.ws_tasks.append(task)
            tasks.append(task)
            
        # Don't wait for tasks to complete (they run continuously)
        await asyncio.sleep(0.1)  # Small delay to ensure connections start
    
    async def _watch_orderbook(self, symbol: str) -> None:
        """
        Monitora orderbook via WebSocket.
        Implementa reconexão automática e cache.
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                while True:
                    start_time = time.perf_counter()
                    
                    # Fetch orderbook from CCXT Pro (uses WebSocket internally)
                    orderbook = await self.ccxt_exchange.watch_order_book(
                        symbol, 
                        limit=20
                    )
                    
                    # Calculate latency
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self.latency_history.append(latency_ms)
                    
                    # Convert to numpy arrays for performance
                    snapshot = OrderBookSnapshot(
                        exchange=self.exchange_id,
                        symbol=symbol,
                        timestamp=orderbook['timestamp'] / 1000,
                        bids=np.array(orderbook['bids'][:20]),
                        asks=np.array(orderbook['asks'][:20]),
                        sequence=orderbook.get('nonce'),
                        latency_ms=latency_ms
                    )
                    
                    # Add to buffer
                    self.orderbook_buffer[symbol].append(snapshot)
                    
                    # Cache in Redis
                    await self._cache_orderbook(symbol, snapshot)
                    
                    # Trigger callbacks
                    await self._trigger_callbacks(
                        MarketDataType.ORDERBOOK, 
                        snapshot
                    )
                    
                    # Update metrics
                    self.last_update_time[f"orderbook_{symbol}"] = time.time()
                    
                    # Check latency target
                    if latency_ms > 20:
                        logger.warning(
                            f"High orderbook latency",
                            symbol=symbol,
                            latency_ms=f"{latency_ms:.2f}"
                        )
                    
            except Exception as e:
                logger.error(f"Orderbook stream error for {symbol}: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(settings.ws_reconnect_interval)
        
        logger.error(f"Max reconnection attempts reached for {symbol} orderbook")
    
    async def _watch_trades(self, symbol: str) -> None:
        """Monitora trades via WebSocket."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                while True:
                    start_time = time.perf_counter()
                    
                    # Fetch trades from CCXT Pro
                    trades = await self.ccxt_exchange.watch_trades(symbol)
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    for trade in trades[-10:]:  # Process last 10 trades
                        trade_data = TradeData(
                            exchange=self.exchange_id,
                            symbol=symbol,
                            timestamp=trade['timestamp'] / 1000,
                            price=trade['price'],
                            volume=trade['amount'],
                            side=trade['side'],
                            trade_id=trade.get('id')
                        )
                        
                        # Add to buffer
                        self.trade_buffer[symbol].append(trade_data)
                        
                        # Cache in Redis
                        await self._cache_trade(symbol, trade_data)
                        
                        # Trigger callbacks
                        await self._trigger_callbacks(
                            MarketDataType.TRADES,
                            trade_data
                        )
                    
                    self.last_update_time[f"trades_{symbol}"] = time.time()
                    
            except Exception as e:
                logger.error(f"Trades stream error for {symbol}: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(settings.ws_reconnect_interval)
    
    async def _watch_ticker(self, symbol: str) -> None:
        """Monitora ticker via WebSocket."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                while True:
                    start_time = time.perf_counter()
                    
                    # Fetch ticker from CCXT Pro
                    ticker = await self.ccxt_exchange.watch_ticker(symbol)
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    ticker_data = TickerData(
                        exchange=self.exchange_id,
                        symbol=symbol,
                        timestamp=ticker['timestamp'] / 1000,
                        bid=ticker['bid'] or 0,
                        ask=ticker['ask'] or 0,
                        last=ticker['last'] or 0,
                        volume_24h=ticker['baseVolume'] or 0,
                        high_24h=ticker['high'] or 0,
                        low_24h=ticker['low'] or 0,
                        open_24h=ticker['open'] or 0,
                        close=ticker['close'] or 0,
                        change_24h=ticker['change'] or 0,
                        change_pct_24h=ticker['percentage'] or 0
                    )
                    
                    # Add to buffer
                    self.ticker_buffer[symbol].append(ticker_data)
                    
                    # Cache in Redis
                    await self._cache_ticker(symbol, ticker_data)
                    
                    # Trigger callbacks
                    await self._trigger_callbacks(
                        MarketDataType.TICKER,
                        ticker_data
                    )
                    
                    self.last_update_time[f"ticker_{symbol}"] = time.time()
                    
            except Exception as e:
                logger.error(f"Ticker stream error for {symbol}: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(settings.ws_reconnect_interval)
    
    async def _cache_orderbook(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """Cache orderbook snapshot in Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"orderbook:{self.exchange_id}:{symbol}"
            
            # Serialize using orjson for speed
            data = orjson.dumps({
                'timestamp': snapshot.timestamp,
                'bids': snapshot.bids.tolist(),
                'asks': snapshot.asks.tolist(),
                'latency_ms': snapshot.latency_ms
            })
            
            await self.redis_client.setex(
                key, 
                settings.redis_ttl_seconds,
                data
            )
        except Exception as e:
            logger.debug(f"Redis cache error: {e}")
    
    async def _cache_trade(self, symbol: str, trade: TradeData) -> None:
        """Cache trade in Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"trades:{self.exchange_id}:{symbol}"
            
            data = orjson.dumps({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'volume': trade.volume,
                'side': trade.side
            })
            
            # Add to sorted set by timestamp
            await self.redis_client.zadd(
                key,
                {data: trade.timestamp}
            )
            
            # Trim old trades (keep last 1000)
            await self.redis_client.zremrangebyrank(key, 0, -1001)
            
            # Set expiry
            await self.redis_client.expire(key, settings.redis_ttl_seconds)
            
        except Exception as e:
            logger.debug(f"Redis cache error: {e}")
    
    async def _cache_ticker(self, symbol: str, ticker: TickerData) -> None:
        """Cache ticker in Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"ticker:{self.exchange_id}:{symbol}"
            
            data = orjson.dumps({
                'timestamp': ticker.timestamp,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'volume_24h': ticker.volume_24h
            })
            
            await self.redis_client.setex(
                key,
                settings.redis_ttl_seconds,
                data
            )
        except Exception as e:
            logger.debug(f"Redis cache error: {e}")
    
    async def _trigger_callbacks(self, data_type: MarketDataType, data: Any) -> None:
        """Trigger registered callbacks for data updates."""
        callbacks = self.callbacks.get(data_type, [])
        
        # Run callbacks concurrently
        if callbacks:
            await asyncio.gather(
                *[callback(data) for callback in callbacks],
                return_exceptions=True
            )
    
    def register_callback(self, data_type: MarketDataType, callback: Callable) -> None:
        """Register callback for specific data type."""
        self.callbacks[data_type].append(callback)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest orderbook snapshot from buffer."""
        buffer = self.orderbook_buffer.get(symbol)
        return buffer[-1] if buffer else None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """Get recent trades from buffer."""
        buffer = self.trade_buffer.get(symbol)
        if not buffer:
            return []
        return list(buffer)[-limit:]
    
    def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get latest ticker from buffer."""
        buffer = self.ticker_buffer.get(symbol)
        return buffer[-1] if buffer else None
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency in ms."""
        if not self.latency_history:
            return 0
        return np.mean(list(self.latency_history))
    
    @property
    def p99_latency(self) -> float:
        """Calculate P99 latency in ms."""
        if not self.latency_history:
            return 0
        return np.percentile(list(self.latency_history), 99)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connections."""
        current_time = time.time()
        
        health = {
            'exchange': self.exchange_id,
            'connected': self.is_connected,
            'uptime_seconds': current_time - self.connection_start_time if self.connection_start_time else 0,
            'average_latency_ms': round(self.average_latency, 2),
            'p99_latency_ms': round(self.p99_latency, 2),
            'symbols_monitored': self.symbols,
            'last_updates': {}
        }
        
        # Check last update times
        for key, timestamp in self.last_update_time.items():
            age_seconds = current_time - timestamp
            health['last_updates'][key] = {
                'age_seconds': round(age_seconds, 2),
                'healthy': age_seconds < 10  # Consider unhealthy if no update for 10s
            }
        
        # Check Redis connection
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health['redis_connected'] = True
            except:
                health['redis_connected'] = False
        else:
            health['redis_connected'] = False
        
        return health
    
    async def close(self) -> None:
        """Close all connections gracefully."""
        logger.info(f"Closing exchange connector for {self.exchange_id}")
        
        # Cancel all WebSocket tasks
        for task in self.ws_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.ws_tasks, return_exceptions=True)
        
        # Close CCXT exchange
        if self.ccxt_exchange:
            await self.ccxt_exchange.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        self.is_connected = False
        logger.info(f"Exchange connector closed for {self.exchange_id}")


class MultiExchangeConnector:
    """
    Gerencia múltiplos conectores de exchange simultaneamente.
    Agrega dados de todas as exchanges para arbitragem.
    """
    
    def __init__(self, exchanges: List[Exchange], symbols: List[str]):
        """
        Inicializa conectores para múltiplas exchanges.
        
        Args:
            exchanges: Lista de exchanges para conectar
            symbols: Lista de símbolos para monitorar
        """
        self.exchanges = exchanges
        self.symbols = symbols
        self.connectors: Dict[str, ExchangeConnector] = {}
        
        # Aggregated data
        self.orderbooks: Dict[str, Dict[str, OrderBookSnapshot]] = defaultdict(dict)
        self.tickers: Dict[str, Dict[str, TickerData]] = defaultdict(dict)
        
    async def initialize(self) -> None:
        """Initialize all exchange connectors."""
        tasks = []
        
        for exchange in self.exchanges:
            connector = ExchangeConnector(exchange, self.symbols)
            self.connectors[exchange.value] = connector
            
            # Register callbacks to aggregate data
            connector.register_callback(
                MarketDataType.ORDERBOOK,
                lambda data, ex=exchange.value: self._on_orderbook_update(ex, data)
            )
            connector.register_callback(
                MarketDataType.TICKER,
                lambda data, ex=exchange.value: self._on_ticker_update(ex, data)
            )
            
            tasks.append(connector.initialize())
        
        await asyncio.gather(*tasks)
        
        logger.info(
            f"Multi-exchange connector initialized",
            exchanges=[e.value for e in self.exchanges],
            symbols=self.symbols
        )
    
    async def _on_orderbook_update(self, exchange: str, orderbook: OrderBookSnapshot) -> None:
        """Handle orderbook update from exchange."""
        self.orderbooks[orderbook.symbol][exchange] = orderbook
    
    async def _on_ticker_update(self, exchange: str, ticker: TickerData) -> None:
        """Handle ticker update from exchange."""
        self.tickers[ticker.symbol][exchange] = ticker
    
    def get_best_bid_ask(self, symbol: str) -> Dict[str, Any]:
        """
        Get best bid/ask across all exchanges.
        Useful for spatial arbitrage detection.
        """
        orderbooks = self.orderbooks.get(symbol, {})
        
        if not orderbooks:
            return {}
        
        best_bid = {'exchange': None, 'price': 0, 'volume': 0}
        best_ask = {'exchange': None, 'price': float('inf'), 'volume': 0}
        
        for exchange, ob in orderbooks.items():
            bid_price, bid_vol = ob.best_bid
            ask_price, ask_vol = ob.best_ask
            
            if bid_price > best_bid['price']:
                best_bid = {'exchange': exchange, 'price': bid_price, 'volume': bid_vol}
            
            if ask_price < best_ask['price'] and ask_price > 0:
                best_ask = {'exchange': exchange, 'price': ask_price, 'volume': ask_vol}
        
        return {
            'symbol': symbol,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask['price'] - best_bid['price'] if best_bid['price'] > 0 else 0,
            'arbitrage_opportunity': best_bid['price'] > best_ask['price']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all connectors."""
        health = {'connectors': {}}
        
        for exchange, connector in self.connectors.items():
            health['connectors'][exchange] = await connector.health_check()
        
        # Overall health
        all_healthy = all(
            c.get('connected', False) 
            for c in health['connectors'].values()
        )
        health['all_connected'] = all_healthy
        
        # Average latency across all exchanges
        latencies = [
            c.get('average_latency_ms', 0) 
            for c in health['connectors'].values()
        ]
        health['average_latency_ms'] = round(np.mean(latencies), 2) if latencies else 0
        
        return health
    
    async def close(self) -> None:
        """Close all connectors."""
        tasks = [connector.close() for connector in self.connectors.values()]
        await asyncio.gather(*tasks)


# Optimized functions using Numba
@jit(nopython=True)
def calculate_weighted_mid_price(bids: np.ndarray, asks: np.ndarray, depth: int = 5) -> float:
    """
    Calculate volume-weighted mid price using Numba for speed.
    
    Args:
        bids: Bid orders [[price, volume], ...]
        asks: Ask orders [[price, volume], ...]
        depth: Number of levels to consider
        
    Returns:
        Weighted mid price
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    bid_value = 0.0
    bid_volume = 0.0
    ask_value = 0.0
    ask_volume = 0.0
    
    for i in range(min(depth, len(bids))):
        bid_value += bids[i, 0] * bids[i, 1]
        bid_volume += bids[i, 1]
    
    for i in range(min(depth, len(asks))):
        ask_value += asks[i, 0] * asks[i, 1]
        ask_volume += asks[i, 1]
    
    if bid_volume == 0 or ask_volume == 0:
        return 0.0
    
    weighted_bid = bid_value / bid_volume
    weighted_ask = ask_value / ask_volume
    
    return (weighted_bid + weighted_ask) / 2


@jit(nopython=True)
def calculate_book_imbalance(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> float:
    """
    Calculate order book imbalance using Numba.
    
    Args:
        bids: Bid orders
        asks: Ask orders
        levels: Number of levels to analyze
        
    Returns:
        Imbalance ratio (-1 to 1)
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    
    bid_volume = 0.0
    ask_volume = 0.0
    
    for i in range(min(levels, len(bids))):
        bid_volume += bids[i, 1]
    
    for i in range(min(levels, len(asks))):
        ask_volume += asks[i, 1]
    
    total_volume = bid_volume + ask_volume
    
    if total_volume == 0:
        return 0.0
    
    return (bid_volume - ask_volume) / total_volume


if __name__ == "__main__":
    # Test connection to Binance
    async def test():
        """Test exchange connector."""
        
        # Test single exchange
        connector = ExchangeConnector(
            Exchange.BINANCE,
            ['BTC/USDT', 'ETH/USDT']
        )
        
        # Register callback to print updates
        def print_orderbook(data):
            print(f"Orderbook update: {data.symbol} - Bid: {data.best_bid[0]:.2f}, Ask: {data.best_ask[0]:.2f}, Latency: {data.latency_ms:.2f}ms")
        
        connector.register_callback(MarketDataType.ORDERBOOK, print_orderbook)
        
        await connector.initialize()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Check health
        health = await connector.health_check()
        print(f"Health check: {json.dumps(health, indent=2)}")
        
        # Check average latency
        print(f"Average latency: {connector.average_latency:.2f}ms")
        print(f"P99 latency: {connector.p99_latency:.2f}ms")
        
        await connector.close()
    
    # Run test
    asyncio.run(test())