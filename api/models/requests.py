"""
Request Models - Pydantic models para requisições da API
Define estrutura e validação de dados de entrada
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from config.settings import Exchange, Strategy


# ============================================
# Enums
# ============================================

class OrderSide(str, Enum):
    """Lado da ordem."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Tipo de ordem."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TimeInForce(str, Enum):
    """Tempo de validade da ordem."""
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date


# ============================================
# Arbitrage Requests
# ============================================

class ScanOpportunitiesRequest(BaseModel):
    """Request para scan de oportunidades."""
    symbols: List[str] = Field(
        default=['BTC/USDT', 'ETH/USDT'],
        description="Símbolos para escanear"
    )
    strategies: List[Strategy] = Field(
        default=None,
        description="Estratégias específicas (None = todas)"
    )
    min_profit_pct: float = Field(
        default=0.3,
        ge=0,
        le=100,
        description="Lucro mínimo percentual"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Confiança mínima"
    )
    max_risk_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Score de risco máximo"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Número máximo de oportunidades"
    )


class ExecuteTradeRequest(BaseModel):
    """Request para execução de trade."""
    symbol: str = Field(..., description="Par de trading")
    buy_exchange: str = Field(..., description="Exchange para compra")
    sell_exchange: str = Field(..., description="Exchange para venda")
    volume: float = Field(..., gt=0, description="Volume para executar")
    max_slippage_pct: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Slippage máximo permitido (%)"
    )
    confidence_override: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Override do threshold de confiança"
    )
    use_ml_prediction: bool = Field(
        default=True,
        description="Usar predição ML"
    )
    dry_run: bool = Field(
        default=True,
        description="Modo simulação"
    )
    
    @validator('buy_exchange', 'sell_exchange')
    def validate_exchange(cls, v):
        valid_exchanges = [e.value for e in Exchange]
        if v not in valid_exchanges:
            raise ValueError(f"Exchange deve ser uma de: {valid_exchanges}")
        return v
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if '/' not in v:
            raise ValueError("Símbolo deve estar no formato BASE/QUOTE (ex: BTC/USDT)")
        return v.upper()


class ManualOrderRequest(BaseModel):
    """Request para ordem manual."""
    exchange: str = Field(..., description="Exchange")
    symbol: str = Field(..., description="Símbolo")
    side: OrderSide = Field(..., description="Lado da ordem")
    order_type: OrderType = Field(..., description="Tipo da ordem")
    volume: float = Field(..., gt=0, description="Volume")
    price: Optional[float] = Field(None, gt=0, description="Preço (para ordens limit)")
    time_in_force: TimeInForce = Field(
        default=TimeInForce.GTC,
        description="Tempo de validade"
    )
    stop_price: Optional[float] = Field(
        None,
        gt=0,
        description="Preço de stop"
    )
    
    @validator('price')
    def validate_price_for_limit(cls, v, values):
        if values.get('order_type') == OrderType.LIMIT and v is None:
            raise ValueError("Preço é obrigatório para ordens limit")
        return v


# ============================================
# Strategy Configuration Requests
# ============================================

class UpdateStrategyConfigRequest(BaseModel):
    """Request para atualizar configuração de estratégia."""
    strategy: Strategy = Field(..., description="Estratégia")
    enabled: Optional[bool] = Field(None, description="Habilitar/desabilitar")
    capital_allocation_pct: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Alocação de capital (%)"
    )
    min_profit_pct: Optional[float] = Field(
        None,
        ge=0,
        description="Lucro mínimo (%)"
    )
    max_position_size: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Tamanho máximo de posição"
    )
    confidence_threshold: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Threshold de confiança"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Parâmetros específicos da estratégia"
    )


class BacktestRequest(BaseModel):
    """Request para backtesting."""
    strategy: Strategy = Field(..., description="Estratégia para testar")
    start_date: datetime = Field(..., description="Data inicial")
    end_date: datetime = Field(..., description="Data final")
    initial_capital: float = Field(
        default=10000,
        gt=0,
        description="Capital inicial"
    )
    symbols: List[str] = Field(
        default=['BTC/USDT', 'ETH/USDT'],
        description="Símbolos para testar"
    )
    walk_forward_windows: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Janelas para walk-forward"
    )
    monte_carlo_simulations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Número de simulações Monte Carlo"
    )


# ============================================
# Risk Management Requests
# ============================================

class UpdateRiskLimitsRequest(BaseModel):
    """Request para atualizar limites de risco."""
    max_position_size_pct: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Tamanho máximo de posição (%)"
    )
    max_daily_loss_pct: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Perda máxima diária (%)"
    )
    max_drawdown_pct: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Drawdown máximo (%)"
    )
    min_sharpe_ratio: Optional[float] = Field(
        None,
        ge=0,
        description="Sharpe ratio mínimo"
    )
    max_correlation: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Correlação máxima entre trades"
    )
    var_95_limit: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Limite VaR 95%"
    )


class PositionSizeRequest(BaseModel):
    """Request para calcular tamanho de posição."""
    symbol: str = Field(..., description="Símbolo")
    strategy: Strategy = Field(..., description="Estratégia")
    confidence: float = Field(..., ge=0, le=1, description="Nível de confiança")
    volatility: float = Field(..., ge=0, description="Volatilidade")
    available_capital: float = Field(..., gt=0, description="Capital disponível")
    current_exposure: float = Field(
        default=0,
        ge=0,
        description="Exposição atual"
    )


# ============================================
# Analysis Requests
# ============================================

class PerformanceAnalysisRequest(BaseModel):
    """Request para análise de performance."""
    period: str = Field(
        default="1d",
        pattern="^[1-9][0-9]*[hdwmq]$",
        description="Período (1h, 24h, 7d, 1w, 1m, 3m)"
    )
    strategy: Optional[Strategy] = Field(
        None,
        description="Estratégia específica (None = todas)"
    )
    metrics: List[str] = Field(
        default=['roi', 'sharpe', 'win_rate', 'profit'],
        description="Métricas para incluir"
    )
    group_by: Optional[str] = Field(
        None,
        description="Agrupar por (exchange, symbol, strategy)"
    )


class MarketAnalysisRequest(BaseModel):
    """Request para análise de mercado."""
    symbols: List[str] = Field(..., description="Símbolos para analisar")
    exchanges: Optional[List[str]] = Field(
        None,
        description="Exchanges específicas"
    )
    include_correlations: bool = Field(
        default=True,
        description="Incluir correlações"
    )
    include_volatility: bool = Field(
        default=True,
        description="Incluir volatilidade"
    )
    include_liquidity: bool = Field(
        default=True,
        description="Incluir métricas de liquidez"
    )
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Horas de lookback"
    )


class FeatureAnalysisRequest(BaseModel):
    """Request para análise de features."""
    symbol: str = Field(..., description="Símbolo")
    exchange: str = Field(..., description="Exchange")
    include_importance: bool = Field(
        default=True,
        description="Incluir importância das features"
    )
    include_shap: bool = Field(
        default=False,
        description="Incluir SHAP values"
    )
    sample_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Tamanho da amostra"
    )


# ============================================
# WebSocket Requests
# ============================================

class WebSocketSubscribeRequest(BaseModel):
    """Request para subscrição WebSocket."""
    channels: List[str] = Field(
        ...,
        description="Canais para subscrever"
    )
    symbols: Optional[List[str]] = Field(
        None,
        description="Símbolos específicos"
    )
    exchanges: Optional[List[str]] = Field(
        None,
        description="Exchanges específicas"
    )
    
    @validator('channels')
    def validate_channels(cls, v):
        valid_channels = [
            'opportunities',
            'trades',
            'orderbook',
            'performance',
            'alerts',
            'system'
        ]
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(f"Canal inválido: {channel}. Válidos: {valid_channels}")
        return v


class WebSocketCommandRequest(BaseModel):
    """Request para comando via WebSocket."""
    command: str = Field(..., description="Comando")
    params: Dict[str, Any] = Field(
        default={},
        description="Parâmetros do comando"
    )
    
    @validator('command')
    def validate_command(cls, v):
        valid_commands = [
            'subscribe',
            'unsubscribe',
            'ping',
            'get_stats',
            'pause_trading',
            'resume_trading'
        ]
        if v not in valid_commands:
            raise ValueError(f"Comando inválido: {v}. Válidos: {valid_commands}")
        return v