"""
Cache Manager - Gerenciamento inteligente de cache com Redis
TTL automático, compressão e estratégias de warming
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
import json
import pickle
import zlib
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

import numpy as np
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from config.settings import settings


logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Tipos de cache com TTL específico."""
    ORDERBOOK = ("orderbook", 5)  # 5 seconds
    TRADE = ("trade", 10)  # 10 seconds
    FEATURE = ("feature", 30)  # 30 seconds
    OPPORTUNITY = ("opportunity", 5)  # 5 seconds
    COINTEGRATION = ("cointegration", 3600)  # 1 hour
    CORRELATION = ("correlation", 300)  # 5 minutes
    STATISTICS = ("statistics", 60)  # 1 minute
    MODEL_PREDICTION = ("model", 10)  # 10 seconds
    GRAPH = ("graph", 60)  # 1 minute for triangular
    ZSCORE = ("zscore", 1)  # 1 second for statistical


class CompressionType(Enum):
    """Tipos de compressão."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"  # Fast compression


class SerializationType(Enum):
    """Tipos de serialização."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    NUMPY = "numpy"


@dataclass
class CacheEntry:
    """Entrada no cache."""
    
    key: str
    value: Any
    cache_type: CacheType
    timestamp: float = field(default_factory=time.time)
    ttl: int = 0
    hits: int = 0
    size_bytes: int = 0
    compressed: bool = False
    
    @property
    def age(self) -> float:
        """Idade em segundos."""
        return time.time() - self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """Verifica se expirou."""
        return self.age > self.ttl if self.ttl > 0 else False


@dataclass
class CacheStatistics:
    """Estatísticas do cache."""
    
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    total_sets: int = 0
    total_deletes: int = 0
    
    bytes_saved: int = 0
    bytes_loaded: int = 0
    
    compression_ratio: float = 0
    
    @property
    def hit_rate(self) -> float:
        """Taxa de acerto do cache."""
        if self.total_requests > 0:
            return self.cache_hits / self.total_requests
        return 0
    
    @property
    def miss_rate(self) -> float:
        """Taxa de erro do cache."""
        return 1 - self.hit_rate


class CacheManager:
    """
    Gerenciador de cache unificado com Redis.
    Otimizado para alta performance e baixa latência.
    """
    
    def __init__(self):
        """Inicializa gerenciador de cache."""
        # Redis configuration
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self._init_redis()
        
        # Cache configuration
        self.default_ttl = 60  # seconds
        self.max_cache_size = 1000000  # 1MB per key
        self.compression_threshold = 1024  # Compress if > 1KB
        
        # Compression settings
        self.compression_type = CompressionType.ZLIB
        self.compression_level = 6  # 1-9, higher = better compression
        
        # Serialization settings
        self.serialization_type = SerializationType.PICKLE
        
        # Local cache (L1 cache)
        self.local_cache: Dict[str, CacheEntry] = {}
        self.local_cache_size = 100  # Max entries in local cache
        self.local_cache_ttl = 1  # 1 second local cache
        
        # Statistics
        self.stats = CacheStatistics()
        self.stats_by_type: Dict[CacheType, CacheStatistics] = defaultdict(CacheStatistics)
        
        # Performance tracking
        self.operation_times = deque(maxlen=1000)
        
        # Warming strategies
        self.warm_cache_enabled = True
        self.warm_cache_patterns: Set[str] = set()
        
        # Cleanup
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # seconds
        
        # Start background tasks
        asyncio.create_task(self._background_cleanup())
        asyncio.create_task(self._cache_warmer())
    
    def _init_redis(self) -> None:
        """Inicializa conexão com Redis."""
        try:
            # Create connection pool for better performance
            self.connection_pool = ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                max_connections=50,
                decode_responses=False  # Binary mode for compression
            )
            
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            logger.info("Redis cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    async def get(
        self,
        key: str,
        cache_type: Optional[CacheType] = None
    ) -> Optional[Any]:
        """
        Obtém valor do cache.
        
        Args:
            key: Chave
            cache_type: Tipo de cache
            
        Returns:
            Valor ou None
        """
        start_time = time.perf_counter()
        
        # Update statistics
        self.stats.total_requests += 1
        if cache_type:
            self.stats_by_type[cache_type].total_requests += 1
        
        # Check local cache first (L1)
        if key in self.local_cache:
            entry = self.local_cache[key]
            
            if not entry.is_expired:
                entry.hits += 1
                self.stats.cache_hits += 1
                
                if cache_type:
                    self.stats_by_type[cache_type].cache_hits += 1
                
                # Track performance
                operation_time = (time.perf_counter() - start_time) * 1000
                self.operation_times.append(('get_local', operation_time))
                
                return entry.value
            else:
                # Remove expired entry
                del self.local_cache[key]
        
        # Check Redis (L2)
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                
                if data:
                    # Deserialize and decompress
                    value = self._deserialize(data)
                    
                    # Update statistics
                    self.stats.cache_hits += 1
                    self.stats.bytes_loaded += len(data)
                    
                    if cache_type:
                        self.stats_by_type[cache_type].cache_hits += 1
                    
                    # Store in local cache
                    ttl = cache_type.value[1] if cache_type else self.default_ttl
                    
                    self._update_local_cache(key, value, cache_type, ttl)
                    
                    # Track performance
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self.operation_times.append(('get_redis', operation_time))
                    
                    return value
                    
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Cache miss
        self.stats.cache_misses += 1
        if cache_type:
            self.stats_by_type[cache_type].cache_misses += 1
        
        # Track performance
        operation_time = (time.perf_counter() - start_time) * 1000
        self.operation_times.append(('get_miss', operation_time))
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_type: Optional[CacheType] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Define valor no cache.
        
        Args:
            key: Chave
            value: Valor
            cache_type: Tipo de cache
            ttl: Time to live customizado
            
        Returns:
            True se sucesso
        """
        start_time = time.perf_counter()
        
        # Update statistics
        self.stats.total_sets += 1
        if cache_type:
            self.stats_by_type[cache_type].total_sets += 1
        
        # Determine TTL
        if ttl is None:
            ttl = cache_type.value[1] if cache_type else self.default_ttl
        
        # Update local cache
        self._update_local_cache(key, value, cache_type, ttl)
        
        # Save to Redis
        if self.redis_client:
            try:
                # Serialize and compress
                data = self._serialize(value)
                original_size = len(data)
                
                # Save to Redis with TTL
                await self.redis_client.setex(key, ttl, data)
                
                # Update statistics
                self.stats.bytes_saved += len(data)
                
                if original_size > len(data):
                    self.stats.compression_ratio = 1 - (len(data) / original_size)
                
                # Track performance
                operation_time = (time.perf_counter() - start_time) * 1000
                self.operation_times.append(('set', operation_time))
                
                return True
                
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                return False
        
        return False
    
    async def delete(self, key: str) -> bool:
        """
        Remove valor do cache.
        
        Args:
            key: Chave
            
        Returns:
            True se removido
        """
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        
        # Remove from Redis
        if self.redis_client:
            try:
                result = await self.redis_client.delete(key)
                self.stats.total_deletes += 1
                return result > 0
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        return False
    
    async def exists(self, key: str) -> bool:
        """
        Verifica se chave existe.
        
        Args:
            key: Chave
            
        Returns:
            True se existe
        """
        # Check local cache
        if key in self.local_cache and not self.local_cache[key].is_expired:
            return True
        
        # Check Redis
        if self.redis_client:
            try:
                return await self.redis_client.exists(key) > 0
            except:
                pass
        
        return False
    
    async def get_many(
        self,
        keys: List[str],
        cache_type: Optional[CacheType] = None
    ) -> Dict[str, Any]:
        """
        Obtém múltiplos valores.
        
        Args:
            keys: Lista de chaves
            cache_type: Tipo de cache
            
        Returns:
            Dict com valores encontrados
        """
        results = {}
        
        # Check local cache first
        redis_keys = []
        
        for key in keys:
            if key in self.local_cache and not self.local_cache[key].is_expired:
                results[key] = self.local_cache[key].value
                self.stats.cache_hits += 1
            else:
                redis_keys.append(key)
        
        # Get remaining from Redis
        if redis_keys and self.redis_client:
            try:
                values = await self.redis_client.mget(redis_keys)
                
                for key, data in zip(redis_keys, values):
                    if data:
                        value = self._deserialize(data)
                        results[key] = value
                        self.stats.cache_hits += 1
                        
                        # Update local cache
                        ttl = cache_type.value[1] if cache_type else self.default_ttl
                        self._update_local_cache(key, value, cache_type, ttl)
                    else:
                        self.stats.cache_misses += 1
                        
            except Exception as e:
                logger.error(f"Redis mget error: {e}")
        
        return results
    
    async def set_many(
        self,
        items: Dict[str, Any],
        cache_type: Optional[CacheType] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Define múltiplos valores.
        
        Args:
            items: Dict de key-value
            cache_type: Tipo de cache
            ttl: Time to live
            
        Returns:
            True se sucesso
        """
        if not items:
            return True
        
        # Determine TTL
        if ttl is None:
            ttl = cache_type.value[1] if cache_type else self.default_ttl
        
        # Update local cache
        for key, value in items.items():
            self._update_local_cache(key, value, cache_type, ttl)
        
        # Save to Redis
        if self.redis_client:
            try:
                # Use pipeline for atomic operation
                pipe = self.redis_client.pipeline()
                
                for key, value in items.items():
                    data = self._serialize(value)
                    pipe.setex(key, ttl, data)
                    self.stats.bytes_saved += len(data)
                
                await pipe.execute()
                self.stats.total_sets += len(items)
                
                return True
                
            except Exception as e:
                logger.error(f"Redis mset error: {e}")
                return False
        
        return False
    
    def _update_local_cache(
        self,
        key: str,
        value: Any,
        cache_type: Optional[CacheType],
        ttl: int
    ) -> None:
        """
        Atualiza cache local.
        
        Args:
            key: Chave
            value: Valor
            cache_type: Tipo
            ttl: TTL
        """
        # Check size limit
        if len(self.local_cache) >= self.local_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.local_cache.keys(),
                key=lambda k: self.local_cache[k].timestamp
            )
            del self.local_cache[oldest_key]
        
        # Add new entry
        self.local_cache[key] = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type or CacheType.STATISTICS,
            ttl=min(ttl, self.local_cache_ttl),
            size_bytes=self._estimate_size(value)
        )
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serializa e comprime valor.
        
        Args:
            value: Valor a serializar
            
        Returns:
            Bytes serializados
        """
        # Serialize based on type
        if self.serialization_type == SerializationType.JSON:
            if isinstance(value, (dict, list)):
                data = json.dumps(value).encode()
            else:
                data = pickle.dumps(value)
        
        elif self.serialization_type == SerializationType.NUMPY:
            if isinstance(value, np.ndarray):
                import io
                buffer = io.BytesIO()
                np.save(buffer, value)
                data = buffer.getvalue()
            else:
                data = pickle.dumps(value)
        
        else:  # PICKLE
            data = pickle.dumps(value)
        
        # Compress if needed
        if len(data) > self.compression_threshold:
            if self.compression_type == CompressionType.ZLIB:
                data = zlib.compress(data, self.compression_level)
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Descomprime e deserializa valor.
        
        Args:
            data: Bytes a deserializar
            
        Returns:
            Valor deserializado
        """
        # Try to decompress
        try:
            data = zlib.decompress(data)
        except:
            pass  # Not compressed
        
        # Deserialize
        try:
            # Try JSON first
            if self.serialization_type == SerializationType.JSON:
                try:
                    return json.loads(data.decode())
                except:
                    pass
            
            # Try numpy
            if self.serialization_type == SerializationType.NUMPY:
                try:
                    import io
                    buffer = io.BytesIO(data)
                    return np.load(buffer)
                except:
                    pass
            
            # Default to pickle
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estima tamanho em bytes.
        
        Args:
            value: Valor
            
        Returns:
            Tamanho estimado
        """
        try:
            if isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value))
            else:
                return len(pickle.dumps(value))
        except:
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Remove chaves por padrão.
        
        Args:
            pattern: Padrão (e.g., "orderbook:*")
            
        Returns:
            Número de chaves removidas
        """
        count = 0
        
        # Clear from local cache
        keys_to_remove = [k for k in self.local_cache if pattern.replace('*', '') in k]
        for key in keys_to_remove:
            del self.local_cache[key]
            count += 1
        
        # Clear from Redis
        if self.redis_client:
            try:
                cursor = 0
                
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    
                    if keys:
                        await self.redis_client.delete(*keys)
                        count += len(keys)
                    
                    if cursor == 0:
                        break
                        
            except Exception as e:
                logger.error(f"Clear pattern error: {e}")
        
        return count
    
    async def get_ttl(self, key: str) -> int:
        """
        Retorna TTL restante.
        
        Args:
            key: Chave
            
        Returns:
            TTL em segundos ou -1 se não existe
        """
        if self.redis_client:
            try:
                return await self.redis_client.ttl(key)
            except:
                pass
        
        return -1
    
    async def extend_ttl(
        self,
        key: str,
        ttl: int
    ) -> bool:
        """
        Estende TTL de uma chave.
        
        Args:
            key: Chave
            ttl: Novo TTL
            
        Returns:
            True se sucesso
        """
        if self.redis_client:
            try:
                return await self.redis_client.expire(key, ttl)
            except:
                pass
        
        return False
    
    def add_warm_pattern(self, pattern: str) -> None:
        """
        Adiciona padrão para cache warming.
        
        Args:
            pattern: Padrão a manter aquecido
        """
        self.warm_cache_patterns.add(pattern)
    
    async def _cache_warmer(self) -> None:
        """Task para manter cache aquecido."""
        while True:
            try:
                if self.warm_cache_enabled and self.warm_cache_patterns:
                    for pattern in self.warm_cache_patterns:
                        # Refresh keys matching pattern
                        if self.redis_client:
                            cursor = 0
                            
                            while True:
                                cursor, keys = await self.redis_client.scan(
                                    cursor, match=pattern, count=10
                                )
                                
                                for key in keys:
                                    # Extend TTL to keep warm
                                    await self.redis_client.expire(key, 60)
                                
                                if cursor == 0:
                                    break
                
                await asyncio.sleep(30)  # Warm every 30 seconds
                
            except Exception as e:
                logger.error(f"Cache warmer error: {e}")
                await asyncio.sleep(60)
    
    async def _background_cleanup(self) -> None:
        """Task para limpeza periódica."""
        while True:
            try:
                # Clean expired local cache
                current_time = time.time()
                
                expired_keys = [
                    k for k, v in self.local_cache.items()
                    if v.is_expired
                ]
                
                for key in expired_keys:
                    del self.local_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired local cache entries")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        avg_time = np.mean([t[1] for t in self.operation_times]) if self.operation_times else 0
        
        stats = {
            'overall': {
                'hit_rate': f"{self.stats.hit_rate:.2%}",
                'miss_rate': f"{self.stats.miss_rate:.2%}",
                'total_requests': self.stats.total_requests,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'bytes_saved': self.stats.bytes_saved,
                'bytes_loaded': self.stats.bytes_loaded,
                'compression_ratio': f"{self.stats.compression_ratio:.2%}"
            },
            'local_cache': {
                'entries': len(self.local_cache),
                'max_size': self.local_cache_size,
                'utilization': f"{len(self.local_cache)/self.local_cache_size:.1%}"
            },
            'performance': {
                'avg_operation_time_ms': avg_time,
                'operations_tracked': len(self.operation_times)
            },
            'warming': {
                'enabled': self.warm_cache_enabled,
                'patterns': list(self.warm_cache_patterns)
            }
        }
        
        # Add per-type statistics
        stats['by_type'] = {}
        
        for cache_type, type_stats in self.stats_by_type.items():
            if type_stats.total_requests > 0:
                stats['by_type'][cache_type.value[0]] = {
                    'hit_rate': f"{type_stats.hit_rate:.2%}",
                    'requests': type_stats.total_requests
                }
        
        return stats
    
    async def health_check(self) -> bool:
        """
        Verifica saúde do cache.
        
        Returns:
            True se saudável
        """
        if not self.redis_client:
            return False
        
        try:
            # Test Redis connection
            await self.redis_client.ping()
            
            # Test basic operations
            test_key = "health_check_test"
            await self.redis_client.setex(test_key, 1, b"test")
            result = await self.redis_client.get(test_key)
            
            return result == b"test"
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False