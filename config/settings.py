"""
Sistema de Arbitragem de Criptomoedas com IA
Configurações Base do Sistema
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import os
from pathlib import Path
from pydantic import BaseSettings, Field, validator
import json


class Environment(str, Enum):
    """Ambientes de execução do sistema."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Exchange(str, Enum):
    """Exchanges permitidas (EXCLUSIVAMENTE estas 5)."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    HTX = "htx"  # Huobi Global


class Strategy(str, Enum):
    """Estratégias de arbitragem implementadas."""
    SPATIAL = "spatial"  # Entre exchanges
    STATISTICAL = "statistical"  # Pairs trading
    TRIANGULAR = "triangular"  # Ciclos triangulares


class Network(str, Enum):
    """Redes blockchain para transferências."""
    BEP20 = "bep20"  # BSC - ÚNICA PERMITIDA (15-30 segundos)
    # ERC20 = "erc20"  # NÃO USAR - muito lento e caro
    # TRC20 = "trc20"  # NÃO USAR - lento


@dataclass
class PerformanceTargets:
    """Metas de performance do sistema."""
    max_latency_ms: int = 30  # Latência máxima end-to-end
    ml_inference_ms: int = 2  # Tempo máximo para decisão IA
    feature_extraction_ms: int = 3  # Tempo para extrair 50+ features
    xai_explanation_ms: int = 5  # Tempo para explicação XAI
    data_collection_ms: int = 20  # Tempo para coletar dados via WebSocket
    min_throughput_per_sec: int = 10000  # Mínimo de updates por segundo
    target_monthly_roi: tuple = (10, 15)  # ROI mensal alvo (10-15%)
    min_sharpe_ratio: float = 2.5  # Sharpe Ratio mínimo
    min_win_rate: float = 0.70  # Taxa de acerto mínima


@dataclass
class RiskLimits:
    """Limites de risco do sistema."""
    max_position_size_pct: float = 0.02  # 2% do capital por posição
    max_daily_loss_pct: float = 0.03  # 3% perda máxima diária
    max_drawdown_pct: float = 0.05  # 5% drawdown máximo
    max_correlation_between_trades: float = 0.60  # 60% correlação máxima
    min_sharpe_ratio: float = 2.5  # Sharpe mínimo
    var_95_limit_pct: float = 0.02  # VaR 95% limite 2%
    max_leverage: float = 3.0  # Alavancagem máxima
    min_confidence_threshold: float = 0.75  # 75% confiança mínima


@dataclass
class CapitalAllocation:
    """Alocação de capital do sistema."""
    pre_funding_pct: float = 0.70  # 70% pré-alocado nas exchanges
    rebalancing_pct: float = 0.30  # 30% para rebalanceamento via BEP20
    spatial_strategy_pct: float = 0.40  # 40% para arbitragem espacial
    statistical_strategy_pct: float = 0.40  # 40% para arbitragem estatística
    triangular_strategy_pct: float = 0.20  # 20% para arbitragem triangular
    reserve_pct: float = 0.05  # 5% reserva de emergência


@dataclass
class MLConfig:
    """Configurações de Machine Learning."""
    min_features: int = 50  # Mínimo de features obrigatórias
    ensemble_models: List[str] = field(default_factory=lambda: [
        "xgboost",  # 30% peso
        "lstm",  # 25% peso
        "transformer",  # 25% peso
        "statistical"  # 20% peso (GARCH/ARIMA)
    ])
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.30,
        "lstm": 0.25,
        "transformer": 0.25,
        "statistical": 0.20
    })
    confidence_threshold: float = 0.75  # 75% confiança mínima
    online_learning_enabled: bool = True
    retrain_interval_hours: int = 24
    min_accuracy: float = 0.85  # 85% acurácia mínima
    walk_forward_windows: int = 10
    monte_carlo_simulations: int = 10000
    

@dataclass
class FeatureConfig:
    """Configuração de Feature Engineering."""
    price_features: List[str] = field(default_factory=lambda: [
        "returns_1m", "returns_5m", "returns_15m", "returns_1h",
        "volatility_1m", "volatility_5m", "volatility_15m",
        "momentum", "acceleration", "jerk",
        "log_returns", "squared_returns"
    ])
    volume_features: List[str] = field(default_factory=lambda: [
        "vwap", "volume_profile", "volume_imbalance",
        "buy_sell_ratio", "large_order_ratio",
        "volume_momentum", "cumulative_volume_delta"
    ])
    orderbook_features: List[str] = field(default_factory=lambda: [
        "bid_ask_spread", "mid_price", "micro_price",
        "book_imbalance", "book_depth_ratio",
        "order_flow_imbalance", "spoofing_indicator",
        "liquidity_score", "orderbook_heat"
    ])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "atr",
        "ema_9", "ema_21", "ema_50", "ema_200",
        "stochastic", "williams_r", "cci", "adx",
        "obv", "cmf", "mfi", "roc", "ppo", "trix"
    ])
    network_features: List[str] = field(default_factory=lambda: [
        "gas_price_gwei", "network_congestion",
        "mempool_size", "pending_transactions",
        "block_time", "hash_rate"
    ])
    cross_asset_features: List[str] = field(default_factory=lambda: [
        "btc_dominance", "correlation_matrix",
        "market_cap_ratio", "volume_ratio",
        "funding_rates", "open_interest"
    ])


class Settings(BaseSettings):
    """Configurações principais do sistema."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT"
    )
    debug: bool = Field(default=False, env="DEBUG")
    
    # Project Info
    project_name: str = "Crypto Arbitrage AI System"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Performance Targets
    performance: PerformanceTargets = PerformanceTargets()
    
    # Risk Management
    risk_limits: RiskLimits = RiskLimits()
    
    # Capital Allocation
    capital: CapitalAllocation = CapitalAllocation()
    
    # Machine Learning
    ml_config: MLConfig = MLConfig()
    
    # Feature Engineering
    features: FeatureConfig = FeatureConfig()
    
    # Exchanges Configuration
    enabled_exchanges: List[Exchange] = Field(
        default=[
            Exchange.BINANCE,
            Exchange.OKX,
            Exchange.BYBIT,
            Exchange.KUCOIN,
            Exchange.HTX
        ]
    )
    
    # Exchange Priorities (1 = highest)
    exchange_priorities: Dict[str, int] = Field(
        default={
            Exchange.BINANCE.value: 1,  # Maior liquidez
            Exchange.OKX.value: 2,  # Boa liquidez
            Exchange.BYBIT.value: 3,  # Execução rápida
            Exchange.KUCOIN.value: 4,  # Novos listings
            Exchange.HTX.value: 5  # Pares específicos
        }
    )
    
    # Strategies Configuration
    enabled_strategies: List[Strategy] = Field(
        default=[
            Strategy.SPATIAL,
            Strategy.STATISTICAL,
            Strategy.TRIANGULAR
        ]
    )
    
    # Network Configuration
    transfer_network: Network = Network.BEP20  # APENAS BEP20!
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ttl_seconds: int = 300  # 5 minutos TTL padrão
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="arbitrage", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="crypto_arbitrage", env="POSTGRES_DB")
    
    @property
    def database_url(self) -> str:
        """Retorna URL de conexão do PostgreSQL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def redis_url(self) -> str:
        """Retorna URL de conexão do Redis."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = 30  # segundos
    ws_reconnect_interval: int = 5  # segundos
    ws_max_reconnect_attempts: int = 10
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "json"  # json ou text
    log_file: Optional[str] = Field(default="logs/arbitrage.log", env="LOG_FILE")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    api_key_header: str = "X-API-Key"
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models"
    logs_dir: Path = data_dir / "logs"
    
    # Exchange API Keys (loaded from environment)
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    
    okx_api_key: Optional[str] = Field(default=None, env="OKX_API_KEY")
    okx_api_secret: Optional[str] = Field(default=None, env="OKX_API_SECRET")
    okx_passphrase: Optional[str] = Field(default=None, env="OKX_PASSPHRASE")
    
    bybit_api_key: Optional[str] = Field(default=None, env="BYBIT_API_KEY")
    bybit_api_secret: Optional[str] = Field(default=None, env="BYBIT_API_SECRET")
    
    kucoin_api_key: Optional[str] = Field(default=None, env="KUCOIN_API_KEY")
    kucoin_api_secret: Optional[str] = Field(default=None, env="KUCOIN_API_SECRET")
    kucoin_passphrase: Optional[str] = Field(default=None, env="KUCOIN_PASSPHRASE")
    
    htx_api_key: Optional[str] = Field(default=None, env="HTX_API_KEY")
    htx_api_secret: Optional[str] = Field(default=None, env="HTX_API_SECRET")
    
    @validator("enabled_exchanges")
    def validate_exchanges(cls, v):
        """Valida que apenas as 5 exchanges permitidas são usadas."""
        allowed = {Exchange.BINANCE, Exchange.OKX, Exchange.BYBIT, 
                  Exchange.KUCOIN, Exchange.HTX}
        if not set(v).issubset(allowed):
            raise ValueError(f"Apenas estas exchanges são permitidas: {allowed}")
        return v
    
    @validator("transfer_network")
    def validate_network(cls, v):
        """Valida que apenas BEP20 é usado para transferências."""
        if v != Network.BEP20:
            raise ValueError("APENAS BEP20 é permitido para transferências!")
        return v
    
    class Config:
        """Configuração do Pydantic."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        

# Singleton instance
settings = Settings()


# Helper functions
def get_exchange_config(exchange: Exchange) -> Dict[str, Any]:
    """Retorna configuração específica de uma exchange."""
    configs = {
        Exchange.BINANCE: {
            "name": "Binance",
            "ws_url": "wss://stream.binance.com:9443/ws",
            "rest_url": "https://api.binance.com",
            "rate_limit": 1200,  # requests per minute
            "min_order_size": 10,  # USDT
            "maker_fee": 0.001,  # 0.1%
            "taker_fee": 0.001,  # 0.1%
            "withdrawal_fee_bep20": 0.8,  # USDT
        },
        Exchange.OKX: {
            "name": "OKX",
            "ws_url": "wss://ws.okx.com:8443/ws/v5/public",
            "rest_url": "https://www.okx.com",
            "rate_limit": 600,
            "min_order_size": 10,
            "maker_fee": 0.0008,
            "taker_fee": 0.001,
            "withdrawal_fee_bep20": 0.8,
        },
        Exchange.BYBIT: {
            "name": "Bybit",
            "ws_url": "wss://stream.bybit.com/v5/public/spot",
            "rest_url": "https://api.bybit.com",
            "rate_limit": 1000,
            "min_order_size": 10,
            "maker_fee": 0.001,
            "taker_fee": 0.001,
            "withdrawal_fee_bep20": 0.8,
        },
        Exchange.KUCOIN: {
            "name": "KuCoin",
            "ws_url": "wss://ws-api-spot.kucoin.com",
            "rest_url": "https://api.kucoin.com",
            "rate_limit": 1800,
            "min_order_size": 10,
            "maker_fee": 0.001,
            "taker_fee": 0.001,
            "withdrawal_fee_bep20": 1.0,
        },
        Exchange.HTX: {
            "name": "HTX (Huobi)",
            "ws_url": "wss://api.huobi.pro/ws",
            "rest_url": "https://api.huobi.pro",
            "rate_limit": 800,
            "min_order_size": 10,
            "maker_fee": 0.002,
            "taker_fee": 0.002,
            "withdrawal_fee_bep20": 0.8,
        }
    }
    return configs.get(exchange, {})


def validate_performance_targets() -> bool:
    """Valida que as metas de performance são alcançáveis."""
    total_time = (
        settings.performance.ml_inference_ms +
        settings.performance.feature_extraction_ms +
        settings.performance.xai_explanation_ms +
        settings.performance.data_collection_ms
    )
    
    if total_time > settings.performance.max_latency_ms:
        raise ValueError(
            f"Soma dos tempos ({total_time}ms) excede latência máxima "
            f"({settings.performance.max_latency_ms}ms)"
        )
    return True


# Validate on import
validate_performance_targets()


if __name__ == "__main__":
    # Print configuration for debugging
    print(f"Environment: {settings.environment}")
    print(f"Enabled Exchanges: {settings.enabled_exchanges}")
    print(f"Enabled Strategies: {settings.enabled_strategies}")
    print(f"Performance Targets: {settings.performance}")
    print(f"Risk Limits: {settings.risk_limits}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")