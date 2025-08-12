"""
Exchange Connector - Conexões WebSocket multi-exchange
Coleta de dados em tempo real com latência ultra-baixa
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed
import ccxt.pro as ccxtpro

from config.settings import settings
from core.data_collector.data_normalizer import DataNormalizer, NormalizedOrderbook, NormalizedTrade
from core.data_collector.orderbook_manager import OrderbookManager
from core.data_collector.cache_manager import CacheManager, CacheType

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Exchanges suportadas."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    HTX = "htx"


class ConnectionStatus(Enum):
    """Status da conexão."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ExchangeConfig:
    """Configuração de uma exchange."""
    
    name: Exchange
    api_key: str = ""
    api_secret: str = ""
    
    # WebSocket endpoints
    ws_url: str = ""
    ws_public: str = ""
    ws_private: str = ""
    
    # Rate limits
    rate_limit: int = 100  # requests per second
    max_connections: int = 10
    
    # Features
    supports_orderbook: bool = True
    supports_trades: bool = True
    supports_ticker: bool = True
    
    # Fees
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    
    # Reconnection
    max_reconnect_attempts: int = 10
    reconnect_interval: float = 5.0


@dataclass
class MarketData:
    """Dados de mercado consolidados."""
    
    symbol: str
    exchange: Exchange
    timestamp: float = field(default_factory=time.time)
    
    # Orderbook
    orderbook: Optional[NormalizedOrderbook] = None
    
    # Trades
    last_trade: Optional[NormalizedTrade] = None
    trades_24h: List[NormalizedTrade] = field(default_factory=list)
    
    # Ticker
    last_price: float = 0
    volume_24h: float = 0
    high_24h: float = 0
    low_24h: float = 0
    
    # Statistics
    spread: float = 0
    spread_pct: float = 0
    
    @property
    def mid_price(self) -> float:
        """Retorna preço médio."""
        if self.orderbook:
            return self.orderbook.mid_price
        return self.last_price


class ExchangeConnector:
    """
    Conector individual para uma exchange.
    Gerencia WebSocket e normalização de dados.
    """
    
    def __init__(
        self,
        config: ExchangeConfig,
        data_normalizer: DataNormalizer,
        orderbook_manager: OrderbookManager,
        cache_manager: CacheManager
    ):
        """
        Inicializa conector.
        
        Args:
            config: Configuração da exchange
            data_normalizer: Normalizador de dados
            orderbook_manager: Gerenciador de orderbooks
            cache_manager: Gerenciador de cache
        """
        self.config = config
        self.data_normalizer = data_normalizer
        self.orderbook_manager = orderbook_manager
        self.cache_manager = cache_manager
        
        # Connection state
        self.status = ConnectionStatus.DISCONNECTED
        self.ws_connection = None
        self.ccxt_exchange = None
        
        # Subscriptions
        self.subscribed_symbols: Set[str] = set()
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Data storage
        self.market_data: Dict[str, MarketData] = {}
        
        # Rate limiting
        self.request_times = deque(maxlen=self.config.rate_limit)
        self.last_request_time = 0
        
        # Reconnection
        self.reconnect_attempts = 0
        self.connection_task = None
        
        # Statistics
        self.messages_received = 0
        self.messages_processed = 0
        self.errors_count = 0
        self.last_message_time = 0
        
    async def connect(self) -> bool:
        """
        Conecta à exchange.
        
        Returns:
            True se conectado com sucesso
        """
        if self.status == ConnectionStatus.CONNECTED:
            return True
        
        self.status = ConnectionStatus.CONNECTING
        
        try:
            # Initialize CCXT exchange
            await self._init_ccxt()
            
            # Connect WebSocket
            await self._connect_websocket()
            
            self.status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            
            logger.info(f"Connected to {self.config.name.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name.value}: {e}")
            self.status = ConnectionStatus.ERROR
            self.errors_count += 1
            
            # Schedule reconnection
            if self.reconnect_attempts < self.config.max_reconnect_attempts:
                asyncio.create_task(self._reconnect())
            
            return False
    
    async def _init_ccxt(self) -> None:
        """Inicializa exchange CCXT."""
        exchange_class = getattr(ccxtpro, self.config.name.value)
        
        self.ccxt_exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'enableRateLimit': True,
            'rateLimit': 1000 / self.config.rate_limit,  # ms between requests
            'options': {
                'defaultType': 'spot',
                'watchOrderBook': {'limit': 20},
                'watchTrades': {'limit': 100}
            }
        })
    
    async def _connect_websocket(self) -> None:
        """Conecta WebSocket."""
        # Use CCXT Pro for WebSocket
        # It handles reconnection automatically
        
        # Start watching for the first symbol to establish connection
        if self.subscribed_symbols:
            symbol = list(self.subscribed_symbols)[0]
            await self.ccxt_exchange.watch_order_book(symbol)
    
    async def disconnect(self) -> None:
        """Desconecta da exchange."""
        self.status = ConnectionStatus.DISCONNECTED
        
        if self.ccxt_exchange:
            await self.ccxt_exchange.close()
            self.ccxt_exchange = None
        
        if self.connection_task:
            self.connection_task.cancel()
        
        logger.info(f"Disconnected from {self.config.name.value}")
    
    async def _reconnect(self) -> None:
        """Tenta reconectar."""
        self.status = ConnectionStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        logger.info(
            f"Reconnecting to {self.config.name.value} "
            f"(attempt {self.reconnect_attempts}/{self.config.max_reconnect_attempts})"
        )
        
        # Wait before reconnecting
        await asyncio.sleep(self.config.reconnect_interval * self.reconnect_attempts)
        
        # Try to connect
        success = await self.connect()
        
        if success:
            # Resubscribe to symbols
            for symbol in self.subscribed_symbols:
                await self.subscribe_symbol(symbol)
    
    async def subscribe_symbol(
        self,
        symbol: str,
        data_types: List[str] = None
    ) -> bool:
        """
        Inscreve em um símbolo.
        
        Args:
            symbol: Símbolo (e.g., 'BTC/USDT')
            data_types: Tipos de dados ['orderbook', 'trades', 'ticker']
            
        Returns:
            True se inscrito com sucesso
        """
        if data_types is None:
            data_types = ['orderbook', 'trades']
        
        try:
            self.subscribed_symbols.add(symbol)
            
            # Start tasks for each data type
            if 'orderbook' in data_types and self.config.supports_orderbook:
                asyncio.create_task(self._watch_orderbook(symbol))
            
            if 'trades' in data_types and self.config.supports_trades:
                asyncio.create_task(self._watch_trades(symbol))
            
            if 'ticker' in data_types and self.config.supports_ticker:
                asyncio.create_task(self._watch_ticker(symbol))
            
            logger.debug(f"Subscribed to {symbol} on {self.config.name.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def _watch_orderbook(self, symbol: str) -> None:
        """Monitora orderbook."""
        while symbol in self.subscribed_symbols:
            try:
                # Watch orderbook updates
                orderbook = await self.ccxt_exchange.watch_order_book(symbol)
                
                # Normalize data
                normalized = await self.data_normalizer.normalize_orderbook(
                    orderbook,
                    self.config.name.value
                )
                
                if normalized:
                    # Update orderbook manager
                    snapshot = await self.orderbook_manager.update_orderbook(normalized)
                    
                    # Update market data
                    if symbol not in self.market_data:
                        self.market_data[symbol] = MarketData(
                            symbol=symbol,
                            exchange=self.config.name
                        )
                    
                    self.market_data[symbol].orderbook = normalized
                    self.market_data[symbol].spread = normalized.spread
                    self.market_data[symbol].spread_pct = normalized.spread_pct
                    
                    # Cache
                    await self.cache_manager.set(
                        f"orderbook:{self.config.name.value}:{symbol}",
                        normalized,
                        CacheType.ORDERBOOK
                    )
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('orderbook', symbol, normalized)
                    
                    # Update statistics
                    self.messages_received += 1
                    self.messages_processed += 1
                    self.last_message_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error watching orderbook for {symbol}: {e}")
                await asyncio.sleep(1)
    
    async def _watch_trades(self, symbol: str) -> None:
        """Monitora trades."""
        while symbol in self.subscribed_symbols:
            try:
                # Watch trades
                trades = await self.ccxt_exchange.watch_trades(symbol)
                
                # Normalize each trade
                for trade in trades:
                    normalized = await self.data_normalizer.normalize_trade(
                        trade,
                        self.config.name.value
                    )
                    
                    if normalized:
                        # Update market data
                        if symbol not in self.market_data:
                            self.market_data[symbol] = MarketData(
                                symbol=symbol,
                                exchange=self.config.name
                            )
                        
                        self.market_data[symbol].last_trade = normalized
                        self.market_data[symbol].last_price = normalized.price
                        self.market_data[symbol].trades_24h.append(normalized)
                        
                        # Keep only last 1000 trades
                        if len(self.market_data[symbol].trades_24h) > 1000:
                            self.market_data[symbol].trades_24h = \
                                self.market_data[symbol].trades_24h[-1000:]
                        
                        # Cache
                        await self.cache_manager.set(
                            f"trade:{self.config.name.value}:{symbol}:{normalized.trade_id}",
                            normalized,
                            CacheType.TRADE
                        )
                        
                        # Trigger callbacks
                        await self._trigger_callbacks('trade', symbol, normalized)
                        
                        self.messages_processed += 1
                        
            except Exception as e:
                logger.error(f"Error watching trades for {symbol}: {e}")
                await asyncio.sleep(1)
    
    async def _watch_ticker(self, symbol: str) -> None:
        """Monitora ticker."""
        while symbol in self.subscribed_symbols:
            try:
                # Watch ticker
                ticker = await self.ccxt_exchange.watch_ticker(symbol)
                
                # Update market data
                if symbol not in self.market_data:
                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        exchange=self.config.name
                    )
                
                self.market_data[symbol].last_price = ticker.get('last', 0)
                self.market_data[symbol].volume_24h = ticker.get('quoteVolume', 0)
                self.market_data[symbol].high_24h = ticker.get('high', 0)
                self.market_data[symbol].low_24h = ticker.get('low', 0)
                
                # Trigger callbacks
                await self._trigger_callbacks('ticker', symbol, ticker)
                
                self.messages_processed += 1
                
                # Less frequent updates for ticker
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error watching ticker for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def unsubscribe_symbol(self, symbol: str) -> None:
        """
        Remove inscrição de um símbolo.
        
        Args:
            symbol: Símbolo
        """
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
            logger.debug(f"Unsubscribed from {symbol} on {self.config.name.value}")
    
    def register_callback(
        self,
        data_type: str,
        callback: Callable
    ) -> None:
        """
        Registra callback para tipo de dados.
        
        Args:
            data_type: Tipo de dados ('orderbook', 'trade', 'ticker')
            callback: Função callback
        """
        self.callbacks[data_type].append(callback)
    
    async def _trigger_callbacks(
        self,
        data_type: str,
        symbol: str,
        data: Any
    ) -> None:
        """
        Aciona callbacks registrados.
        
        Args:
            data_type: Tipo de dados
            symbol: Símbolo
            data: Dados
        """
        for callback in self.callbacks.get(data_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data, self.config.name)
                else:
                    callback(symbol, data, self.config.name)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def check_rate_limit(self) -> bool:
        """
        Verifica rate limit.
        
        Returns:
            True se pode fazer request
        """
        current_time = time.time()
        
        # Remove old requests
        while self.request_times and self.request_times[0] < current_time - 1:
            self.request_times.popleft()
        
        # Check if under limit
        if len(self.request_times) < self.config.rate_limit:
            self.request_times.append(current_time)
            self.last_request_time = current_time
            return True
        
        return False
    
    def get_orderbook(self, symbol: str) -> Optional[NormalizedOrderbook]:
        """
        Retorna orderbook atual.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Orderbook ou None
        """
        market_data = self.market_data.get(symbol)
        return market_data.orderbook if market_data else None
    
    def get_last_trade(self, symbol: str) -> Optional[NormalizedTrade]:
        """
        Retorna última trade.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Trade ou None
        """
        market_data = self.market_data.get(symbol)
        return market_data.last_trade if market_data else None
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Retorna dados de mercado.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dados de mercado ou None
        """
        return self.market_data.get(symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do conector."""
        uptime = time.time() - self.last_message_time if self.last_message_time else 0
        
        return {
            'exchange': self.config.name.value,
            'status': self.status.value,
            'subscribed_symbols': list(self.subscribed_symbols),
            'messages_received': self.messages_received,
            'messages_processed': self.messages_processed,
            'errors': self.errors_count,
            'reconnect_attempts': self.reconnect_attempts,
            'uptime_seconds': uptime,
            'last_message_ago': time.time() - self.last_message_time if self.last_message_time else None
        }


class MultiExchangeConnector:
    """
    Conector multi-exchange.
    Gerencia múltiplas exchanges simultaneamente.
    """
    
    def __init__(self):
        """Inicializa conector multi-exchange."""
        # Components
        self.data_normalizer = DataNormalizer()
        self.orderbook_manager = OrderbookManager()
        self.cache_manager = CacheManager()
        
        # Exchange configurations
        self.configs = self._load_exchange_configs()
        
        # Connectors
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.exchanges = set(Exchange)
        
        # Global callbacks
        self.global_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.start_time = time.time()
        
    def _load_exchange_configs(self) -> Dict[Exchange, ExchangeConfig]:
        """Carrega configurações das exchanges."""
        configs = {}
        
        # Binance
        configs[Exchange.BINANCE] = ExchangeConfig(
            name=Exchange.BINANCE,
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            ws_url="wss://stream.binance.com:9443/ws",
            rate_limit=1200,
            maker_fee=0.001,
            taker_fee=0.001
        )
        
        # OKX
        configs[Exchange.OKX] = ExchangeConfig(
            name=Exchange.OKX,
            api_key=settings.okx_api_key,
            api_secret=settings.okx_api_secret,
            ws_url="wss://ws.okx.com:8443/ws/v5/public",
            rate_limit=100,
            maker_fee=0.0008,
            taker_fee=0.001
        )
        
        # Bybit
        configs[Exchange.BYBIT] = ExchangeConfig(
            name=Exchange.BYBIT,
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            ws_url="wss://stream.bybit.com/v5/public/spot",
            rate_limit=100,
            maker_fee=0.001,
            taker_fee=0.001
        )
        
        # KuCoin
        configs[Exchange.KUCOIN] = ExchangeConfig(
            name=Exchange.KUCOIN,
            api_key=settings.kucoin_api_key,
            api_secret=settings.kucoin_api_secret,
            ws_url="wss://ws-api.kucoin.com/socket.io/",
            rate_limit=100,
            maker_fee=0.001,
            taker_fee=0.001
        )
        
        # HTX
        configs[Exchange.HTX] = ExchangeConfig(
            name=Exchange.HTX,
            api_key=settings.htx_api_key,
            api_secret=settings.htx_api_secret,
            ws_url="wss://api.huobi.pro/ws",
            rate_limit=100,
            maker_fee=0.002,
            taker_fee=0.002
        )
        
        return configs
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Conecta a todas as exchanges.
        
        Returns:
            Status de conexão por exchange
        """
        results = {}
        
        for exchange, config in self.configs.items():
            connector = ExchangeConnector(
                config,
                self.data_normalizer,
                self.orderbook_manager,
                self.cache_manager
            )
            
            success = await connector.connect()
            
            if success:
                self.connectors[exchange.value] = connector
                
                # Register global callbacks
                for data_type, callbacks in self.global_callbacks.items():
                    for callback in callbacks:
                        connector.register_callback(data_type, callback)
            
            results[exchange.value] = success
        
        logger.info(f"Connected to {sum(results.values())}/{len(results)} exchanges")
        
        return results
    
    async def disconnect_all(self) -> None:
        """Desconecta de todas as exchanges."""
        for connector in self.connectors.values():
            await connector.disconnect()
        
        self.connectors.clear()
        logger.info("Disconnected from all exchanges")
    
    async def subscribe_symbol_all(
        self,
        symbol: str,
        data_types: List[str] = None
    ) -> Dict[str, bool]:
        """
        Inscreve símbolo em todas as exchanges.
        
        Args:
            symbol: Símbolo
            data_types: Tipos de dados
            
        Returns:
            Status por exchange
        """
        results = {}
        
        for exchange_name, connector in self.connectors.items():
            success = await connector.subscribe_symbol(symbol, data_types)
            results[exchange_name] = success
        
        return results
    
    def register_global_callback(
        self,
        data_type: str,
        callback: Callable
    ) -> None:
        """
        Registra callback global.
        
        Args:
            data_type: Tipo de dados
            callback: Função callback
        """
        self.global_callbacks[data_type].append(callback)
        
        # Register in existing connectors
        for connector in self.connectors.values():
            connector.register_callback(data_type, callback)
    
    def get_best_bid_ask(
        self,
        symbol: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Retorna melhor bid/ask de todas as exchanges.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dict com (bid, ask) por exchange
        """
        results = {}
        
        for exchange_name, connector in self.connectors.items():
            orderbook = connector.get_orderbook(symbol)
            
            if orderbook and orderbook.bids and orderbook.asks:
                results[exchange_name] = (
                    orderbook.bids[0][0],
                    orderbook.asks[0][0]
                )
        
        return results
    
    def find_arbitrage_opportunity(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Procura oportunidade de arbitragem simples.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Oportunidade ou None
        """
        bid_asks = self.get_best_bid_ask(symbol)
        
        if len(bid_asks) < 2:
            return None
        
        # Find best bid and ask across exchanges
        best_bid = max(bid_asks.items(), key=lambda x: x[1][0])
        best_ask = min(bid_asks.items(), key=lambda x: x[1][1])
        
        # Check for arbitrage
        if best_bid[1][0] > best_ask[1][1]:
            spread = best_bid[1][0] - best_ask[1][1]
            spread_pct = (spread / best_ask[1][1]) * 100
            
            # Account for fees
            total_fees = (
                self.configs[Exchange[best_bid[0].upper()]].taker_fee +
                self.configs[Exchange[best_ask[0].upper()]].taker_fee
            ) * 100
            
            net_profit_pct = spread_pct - total_fees
            
            if net_profit_pct > 0:
                return {
                    'symbol': symbol,
                    'buy_exchange': best_ask[0],
                    'buy_price': best_ask[1][1],
                    'sell_exchange': best_bid[0],
                    'sell_price': best_bid[1][0],
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'net_profit_pct': net_profit_pct,
                    'timestamp': time.time()
                }
        
        return None
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas globais."""
        stats = {
            'uptime': time.time() - self.start_time,
            'connected_exchanges': len(self.connectors),
            'total_exchanges': len(self.configs),
            'exchanges': {}
        }
        
        for exchange_name, connector in self.connectors.items():
            stats['exchanges'][exchange_name] = connector.get_statistics()
        
        return stats