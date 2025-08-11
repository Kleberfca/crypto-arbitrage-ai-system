"""
Data Normalizer - Normalização e padronização de dados entre exchanges
Converte formatos, padroniza timestamps e trata dados faltantes
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import logging

import numpy as np
from numba import jit
import pandas as pd

from config.settings import settings


logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Formatos de dados suportados."""
    RAW = "raw"
    NORMALIZED = "normalized"
    AGGREGATED = "aggregated"
    COMPRESSED = "compressed"


class ExchangeFormat(Enum):
    """Formatos específicos de exchanges."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    HTX = "htx"


@dataclass
class NormalizedTrade:
    """Trade normalizado."""
    
    # Universal fields
    timestamp: float
    symbol: str
    price: float
    amount: float
    side: str  # 'buy' or 'sell'
    
    # Exchange specific
    exchange: str
    trade_id: str
    
    # Derived fields
    value_usd: float = 0
    normalized_at: float = field(default_factory=time.time)
    
    @property
    def is_buy(self) -> bool:
        """Verifica se é compra."""
        return self.side.lower() in ['buy', 'bid']
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'amount': self.amount,
            'side': self.side,
            'exchange': self.exchange,
            'trade_id': self.trade_id,
            'value_usd': self.value_usd
        }


@dataclass
class NormalizedOrderbook:
    """Orderbook normalizado."""
    
    # Metadata
    timestamp: float
    symbol: str
    exchange: str
    
    # Price levels (price, amount)
    bids: List[List[float]]
    asks: List[List[float]]
    
    # Derived metrics
    mid_price: float = 0
    spread: float = 0
    spread_pct: float = 0
    
    # Liquidity metrics
    bid_liquidity_usd: float = 0
    ask_liquidity_usd: float = 0
    
    # Quality metrics
    levels: int = 0
    is_crossed: bool = False
    
    @property
    def best_bid(self) -> Tuple[float, float]:
        """Retorna melhor bid."""
        if self.bids:
            return (self.bids[0][0], self.bids[0][1])
        return (0, 0)
    
    @property
    def best_ask(self) -> Tuple[float, float]:
        """Retorna melhor ask."""
        if self.asks:
            return (self.asks[0][0], self.asks[0][1])
        return (0, 0)


class DataNormalizer:
    """
    Normalizador de dados entre exchanges.
    Padroniza formatos e trata inconsistências.
    """
    
    def __init__(self):
        """Inicializa normalizador de dados."""
        # Configuration
        self.decimal_precision = 8
        self.timestamp_precision = 3  # milliseconds
        self.max_orderbook_levels = 20
        
        # Exchange mappings
        self.symbol_mappings = self._load_symbol_mappings()
        self.side_mappings = self._load_side_mappings()
        
        # Data quality tracking
        self.normalization_stats = {
            'total_processed': 0,
            'errors': 0,
            'invalid_data': 0,
            'missing_fields': 0
        }
        
        # Performance tracking
        self.processing_times = []
        self.last_reset = time.time()
        
    def _load_symbol_mappings(self) -> Dict[str, Dict[str, str]]:
        """Carrega mapeamento de símbolos entre exchanges."""
        return {
            'binance': {
                'BTCUSDT': 'BTC/USDT',
                'ETHUSDT': 'ETH/USDT',
                'BNBUSDT': 'BNB/USDT'
            },
            'okx': {
                'BTC-USDT': 'BTC/USDT',
                'ETH-USDT': 'ETH/USDT',
                'OKB-USDT': 'OKB/USDT'
            },
            'bybit': {
                'BTCUSDT': 'BTC/USDT',
                'ETHUSDT': 'ETH/USDT',
                'BITUSDT': 'BIT/USDT'
            },
            'kucoin': {
                'BTC-USDT': 'BTC/USDT',
                'ETH-USDT': 'ETH/USDT',
                'KCS-USDT': 'KCS/USDT'
            },
            'htx': {
                'btcusdt': 'BTC/USDT',
                'ethusdt': 'ETH/USDT',
                'htusdt': 'HT/USDT'
            }
        }
    
    def _load_side_mappings(self) -> Dict[str, Dict[str, str]]:
        """Carrega mapeamento de sides entre exchanges."""
        return {
            'binance': {'BUY': 'buy', 'SELL': 'sell'},
            'okx': {'buy': 'buy', 'sell': 'sell'},
            'bybit': {'Buy': 'buy', 'Sell': 'sell'},
            'kucoin': {'buy': 'buy', 'sell': 'sell'},
            'htx': {'buy': 'buy', 'sell': 'sell'}
        }
    
    async def normalize_trade(
        self,
        raw_trade: Dict[str, Any],
        exchange: str
    ) -> Optional[NormalizedTrade]:
        """
        Normaliza uma trade.
        
        Args:
            raw_trade: Trade raw da exchange
            exchange: Nome da exchange
            
        Returns:
            Trade normalizada ou None
        """
        start_time = time.perf_counter()
        
        try:
            # Extract fields based on exchange format
            if exchange.lower() == 'binance':
                normalized = self._normalize_binance_trade(raw_trade)
            elif exchange.lower() == 'okx':
                normalized = self._normalize_okx_trade(raw_trade)
            elif exchange.lower() == 'bybit':
                normalized = self._normalize_bybit_trade(raw_trade)
            elif exchange.lower() == 'kucoin':
                normalized = self._normalize_kucoin_trade(raw_trade)
            elif exchange.lower() == 'htx':
                normalized = self._normalize_htx_trade(raw_trade)
            else:
                logger.warning(f"Unknown exchange format: {exchange}")
                return None
            
            # Add exchange info
            if normalized:
                normalized.exchange = exchange
                
                # Calculate USD value
                normalized.value_usd = normalized.price * normalized.amount
                
                # Update stats
                self.normalization_stats['total_processed'] += 1
            
            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing trade: {e}")
            self.normalization_stats['errors'] += 1
            return None
    
    def _normalize_binance_trade(self, raw: Dict[str, Any]) -> Optional[NormalizedTrade]:
        """Normaliza trade da Binance."""
        try:
            # Binance format: {'id', 'price', 'qty', 'time', 'isBuyerMaker'}
            symbol = self.symbol_mappings['binance'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            return NormalizedTrade(
                timestamp=raw.get('time', time.time()) / 1000,  # Convert ms to seconds
                symbol=symbol,
                price=float(raw.get('price', 0)),
                amount=float(raw.get('qty', 0)),
                side='sell' if raw.get('isBuyerMaker') else 'buy',
                exchange='binance',
                trade_id=str(raw.get('id', ''))
            )
        except:
            return None
    
    def _normalize_okx_trade(self, raw: Dict[str, Any]) -> Optional[NormalizedTrade]:
        """Normaliza trade da OKX."""
        try:
            # OKX format: {'instId', 'px', 'sz', 'ts', 'side'}
            symbol = self.symbol_mappings['okx'].get(
                raw.get('instId', ''), raw.get('instId', '')
            )
            
            return NormalizedTrade(
                timestamp=int(raw.get('ts', time.time() * 1000)) / 1000,
                symbol=symbol,
                price=float(raw.get('px', 0)),
                amount=float(raw.get('sz', 0)),
                side=raw.get('side', 'buy'),
                exchange='okx',
                trade_id=str(raw.get('tradeId', ''))
            )
        except:
            return None
    
    def _normalize_bybit_trade(self, raw: Dict[str, Any]) -> Optional[NormalizedTrade]:
        """Normaliza trade da Bybit."""
        try:
            # Bybit format: {'symbol', 'price', 'size', 'time', 'side'}
            symbol = self.symbol_mappings['bybit'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            return NormalizedTrade(
                timestamp=raw.get('time', time.time()) / 1000,
                symbol=symbol,
                price=float(raw.get('price', 0)),
                amount=float(raw.get('size', 0)),
                side=self.side_mappings['bybit'].get(raw.get('side', ''), 'buy'),
                exchange='bybit',
                trade_id=str(raw.get('id', ''))
            )
        except:
            return None
    
    def _normalize_kucoin_trade(self, raw: Dict[str, Any]) -> Optional[NormalizedTrade]:
        """Normaliza trade da KuCoin."""
        try:
            # KuCoin format: {'symbol', 'price', 'size', 'time', 'side'}
            symbol = self.symbol_mappings['kucoin'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            return NormalizedTrade(
                timestamp=raw.get('time', time.time() * 1000000000) / 1000000000,  # Nanoseconds
                symbol=symbol,
                price=float(raw.get('price', 0)),
                amount=float(raw.get('size', 0)),
                side=raw.get('side', 'buy'),
                exchange='kucoin',
                trade_id=str(raw.get('sequence', ''))
            )
        except:
            return None
    
    def _normalize_htx_trade(self, raw: Dict[str, Any]) -> Optional[NormalizedTrade]:
        """Normaliza trade da HTX."""
        try:
            # HTX format: {'symbol', 'price', 'amount', 'ts', 'direction'}
            symbol = self.symbol_mappings['htx'].get(
                raw.get('symbol', ''), raw.get('symbol', '').upper()
            )
            
            if '/' not in symbol and len(symbol) > 4:
                # Try to parse symbol (e.g., btcusdt -> BTC/USDT)
                symbol = symbol[:-4].upper() + '/' + symbol[-4:].upper()
            
            return NormalizedTrade(
                timestamp=raw.get('ts', time.time() * 1000) / 1000,
                symbol=symbol,
                price=float(raw.get('price', 0)),
                amount=float(raw.get('amount', 0)),
                side=raw.get('direction', 'buy'),
                exchange='htx',
                trade_id=str(raw.get('id', ''))
            )
        except:
            return None
    
    async def normalize_orderbook(
        self,
        raw_orderbook: Dict[str, Any],
        exchange: str
    ) -> Optional[NormalizedOrderbook]:
        """
        Normaliza um orderbook.
        
        Args:
            raw_orderbook: Orderbook raw da exchange
            exchange: Nome da exchange
            
        Returns:
            Orderbook normalizado ou None
        """
        start_time = time.perf_counter()
        
        try:
            # Extract based on exchange
            if exchange.lower() == 'binance':
                normalized = self._normalize_binance_orderbook(raw_orderbook)
            elif exchange.lower() == 'okx':
                normalized = self._normalize_okx_orderbook(raw_orderbook)
            elif exchange.lower() == 'bybit':
                normalized = self._normalize_bybit_orderbook(raw_orderbook)
            elif exchange.lower() == 'kucoin':
                normalized = self._normalize_kucoin_orderbook(raw_orderbook)
            elif exchange.lower() == 'htx':
                normalized = self._normalize_htx_orderbook(raw_orderbook)
            else:
                return None
            
            if normalized:
                # Calculate derived metrics
                self._calculate_orderbook_metrics(normalized)
                
                # Check quality
                if self._validate_orderbook(normalized):
                    self.normalization_stats['total_processed'] += 1
                else:
                    self.normalization_stats['invalid_data'] += 1
                    return None
            
            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing orderbook: {e}")
            self.normalization_stats['errors'] += 1
            return None
    
    def _normalize_binance_orderbook(
        self,
        raw: Dict[str, Any]
    ) -> Optional[NormalizedOrderbook]:
        """Normaliza orderbook da Binance."""
        try:
            # Binance format: {'bids': [[price, qty]], 'asks': [[price, qty]]}
            symbol = self.symbol_mappings['binance'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            bids = [[float(p), float(q)] for p, q in raw.get('bids', [])[:self.max_orderbook_levels]]
            asks = [[float(p), float(q)] for p, q in raw.get('asks', [])[:self.max_orderbook_levels]]
            
            return NormalizedOrderbook(
                timestamp=raw.get('E', time.time()) / 1000,
                symbol=symbol,
                exchange='binance',
                bids=bids,
                asks=asks,
                levels=min(len(bids), len(asks))
            )
        except:
            return None
    
    def _normalize_okx_orderbook(
        self,
        raw: Dict[str, Any]
    ) -> Optional[NormalizedOrderbook]:
        """Normaliza orderbook da OKX."""
        try:
            # OKX format: {'bids': [[price, qty, orders]], 'asks': [[price, qty, orders]]}
            symbol = self.symbol_mappings['okx'].get(
                raw.get('instId', ''), raw.get('instId', '')
            )
            
            # Extract only price and quantity (ignore order count)
            bids = [[float(b[0]), float(b[1])] for b in raw.get('bids', [])[:self.max_orderbook_levels]]
            asks = [[float(a[0]), float(a[1])] for a in raw.get('asks', [])[:self.max_orderbook_levels]]
            
            return NormalizedOrderbook(
                timestamp=int(raw.get('ts', time.time() * 1000)) / 1000,
                symbol=symbol,
                exchange='okx',
                bids=bids,
                asks=asks,
                levels=min(len(bids), len(asks))
            )
        except:
            return None
    
    def _normalize_bybit_orderbook(
        self,
        raw: Dict[str, Any]
    ) -> Optional[NormalizedOrderbook]:
        """Normaliza orderbook da Bybit."""
        try:
            symbol = self.symbol_mappings['bybit'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            bids = [[float(b[0]), float(b[1])] for b in raw.get('b', [])[:self.max_orderbook_levels]]
            asks = [[float(a[0]), float(a[1])] for a in raw.get('a', [])[:self.max_orderbook_levels]]
            
            return NormalizedOrderbook(
                timestamp=raw.get('t', time.time()) / 1000,
                symbol=symbol,
                exchange='bybit',
                bids=bids,
                asks=asks,
                levels=min(len(bids), len(asks))
            )
        except:
            return None
    
    def _normalize_kucoin_orderbook(
        self,
        raw: Dict[str, Any]
    ) -> Optional[NormalizedOrderbook]:
        """Normaliza orderbook da KuCoin."""
        try:
            symbol = self.symbol_mappings['kucoin'].get(
                raw.get('symbol', ''), raw.get('symbol', '')
            )
            
            bids = [[float(p), float(s)] for p, s in raw.get('bids', [])[:self.max_orderbook_levels]]
            asks = [[float(p), float(s)] for p, s in raw.get('asks', [])[:self.max_orderbook_levels]]
            
            return NormalizedOrderbook(
                timestamp=raw.get('time', time.time() * 1000000000) / 1000000000,
                symbol=symbol,
                exchange='kucoin',
                bids=bids,
                asks=asks,
                levels=min(len(bids), len(asks))
            )
        except:
            return None
    
    def _normalize_htx_orderbook(
        self,
        raw: Dict[str, Any]
    ) -> Optional[NormalizedOrderbook]:
        """Normaliza orderbook da HTX."""
        try:
            symbol = self.symbol_mappings['htx'].get(
                raw.get('symbol', ''), raw.get('symbol', '').upper()
            )
            
            if '/' not in symbol and len(symbol) > 4:
                symbol = symbol[:-4].upper() + '/' + symbol[-4:].upper()
            
            # HTX format has nested structure
            tick = raw.get('tick', {})
            bids = [[float(b[0]), float(b[1])] for b in tick.get('bids', [])[:self.max_orderbook_levels]]
            asks = [[float(a[0]), float(a[1])] for a in tick.get('asks', [])[:self.max_orderbook_levels]]
            
            return NormalizedOrderbook(
                timestamp=raw.get('ts', time.time() * 1000) / 1000,
                symbol=symbol,
                exchange='htx',
                bids=bids,
                asks=asks,
                levels=min(len(bids), len(asks))
            )
        except:
            return None
    
    @jit(nopython=True)
    def _calculate_orderbook_metrics_fast(
        self,
        bids: np.ndarray,
        asks: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Calcula métricas do orderbook otimizado.
        
        Args:
            bids: Array de bids
            asks: Array de asks
            
        Returns:
            (mid_price, spread, spread_pct, bid_liquidity, ask_liquidity)
        """
        if len(bids) == 0 or len(asks) == 0:
            return 0, 0, 0, 0, 0
        
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = spread / mid_price * 100 if mid_price > 0 else 0
        
        # Calculate liquidity
        bid_liquidity = np.sum(bids[:, 0] * bids[:, 1])
        ask_liquidity = np.sum(asks[:, 0] * asks[:, 1])
        
        return mid_price, spread, spread_pct, bid_liquidity, ask_liquidity
    
    def _calculate_orderbook_metrics(self, orderbook: NormalizedOrderbook) -> None:
        """Calcula métricas derivadas do orderbook."""
        if not orderbook.bids or not orderbook.asks:
            return
        
        # Convert to numpy for fast calculation
        bids_array = np.array(orderbook.bids)
        asks_array = np.array(orderbook.asks)
        
        # Calculate metrics
        metrics = self._calculate_orderbook_metrics_fast(bids_array, asks_array)
        
        orderbook.mid_price = metrics[0]
        orderbook.spread = metrics[1]
        orderbook.spread_pct = metrics[2]
        orderbook.bid_liquidity_usd = metrics[3]
        orderbook.ask_liquidity_usd = metrics[4]
        
        # Check if crossed
        if orderbook.bids and orderbook.asks:
            orderbook.is_crossed = orderbook.bids[0][0] >= orderbook.asks[0][0]
    
    def _validate_orderbook(self, orderbook: NormalizedOrderbook) -> bool:
        """
        Valida qualidade do orderbook.
        
        Args:
            orderbook: Orderbook a validar
            
        Returns:
            True se válido
        """
        # Check basic structure
        if not orderbook.bids or not orderbook.asks:
            return False
        
        # Check for crossed book
        if orderbook.is_crossed:
            logger.warning(f"Crossed orderbook detected: {orderbook.symbol} on {orderbook.exchange}")
            return False
        
        # Check for unrealistic spread
        if orderbook.spread_pct > 10:  # > 10% spread
            logger.warning(f"Unrealistic spread: {orderbook.spread_pct:.2f}%")
            return False
        
        # Check price levels are sorted
        for i in range(1, len(orderbook.bids)):
            if orderbook.bids[i][0] >= orderbook.bids[i-1][0]:
                return False
        
        for i in range(1, len(orderbook.asks)):
            if orderbook.asks[i][0] <= orderbook.asks[i-1][0]:
                return False
        
        return True
    
    async def normalize_batch(
        self,
        raw_data: List[Dict[str, Any]],
        data_type: str,
        exchange: str
    ) -> List[Union[NormalizedTrade, NormalizedOrderbook]]:
        """
        Normaliza batch de dados.
        
        Args:
            raw_data: Lista de dados raw
            data_type: 'trade' ou 'orderbook'
            exchange: Nome da exchange
            
        Returns:
            Lista de dados normalizados
        """
        normalized = []
        
        # Process in parallel for large batches
        if len(raw_data) > 100:
            tasks = []
            
            for item in raw_data:
                if data_type == 'trade':
                    task = self.normalize_trade(item, exchange)
                else:
                    task = self.normalize_orderbook(item, exchange)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if result and not isinstance(result, Exception):
                    normalized.append(result)
        else:
            # Process sequentially for small batches
            for item in raw_data:
                if data_type == 'trade':
                    result = await self.normalize_trade(item, exchange)
                else:
                    result = await self.normalize_orderbook(item, exchange)
                
                if result:
                    normalized.append(result)
        
        return normalized
    
    def fill_missing_data(
        self,
        data: pd.DataFrame,
        method: str = 'forward'
    ) -> pd.DataFrame:
        """
        Preenche dados faltantes.
        
        Args:
            data: DataFrame com dados
            method: Método de preenchimento ('forward', 'interpolate', 'mean')
            
        Returns:
            DataFrame preenchido
        """
        if method == 'forward':
            return data.fillna(method='ffill')
        elif method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'mean':
            return data.fillna(data.mean())
        else:
            return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de normalização."""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_processed': self.normalization_stats['total_processed'],
            'errors': self.normalization_stats['errors'],
            'invalid_data': self.normalization_stats['invalid_data'],
            'missing_fields': self.normalization_stats['missing_fields'],
            'error_rate': (
                self.normalization_stats['errors'] / 
                max(self.normalization_stats['total_processed'], 1) * 100
            ),
            'avg_processing_time_ms': avg_time,
            'max_processing_time_ms': max(self.processing_times) if self.processing_times else 0
        }
    
    def reset_statistics(self) -> None:
        """Reseta estatísticas."""
        self.normalization_stats = {
            'total_processed': 0,
            'errors': 0,
            'invalid_data': 0,
            'missing_fields': 0
        }
        self.processing_times = []
        self.last_reset = time.time()