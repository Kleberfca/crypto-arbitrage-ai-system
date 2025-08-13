"""
Response Models - Pydantic models para respostas da API
Define estrutura e serialização de dados de saída
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================
# Status Enums
# ============================================

class SystemStatus(str, Enum):
    """Status do sistema."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class TradeStatus(str, Enum):
    """Status do trade."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================
# Health & Monitoring Responses
# ============================================

class HealthStatus(BaseModel):
    """Response de health check."""
    status: SystemStatus = Field(..., description="Status geral do sistema")
    timestamp: float = Field(..., description="Timestamp atual")
    uptime_seconds: float = Field(..., description="Uptime em segundos")
    version: str = Field(..., description="Versão do sistema")
    environment: str = Field(..., description="Ambiente atual")
    services: Dict[str, Any] = Field(
        default_factory=dict,
        description="Status dos serviços"
    )
    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Checks de saúde"
    )


class PerformanceMetrics(BaseModel):
    """Response de métricas de performance."""
    latency_ms: float = Field(..., description="Latência média em ms")
    throughput_per_sec: int = Field(..., description="Throughput por segundo")
    opportunities_detected: int = Field(..., description="Oportunidades detectadas")
    opportunities_executed: int = Field(..., description="Oportunidades executadas")
    total_profit_usd: float = Field(..., description="Lucro total em USD")
    success_rate: float = Field(..., description="Taxa de sucesso (%)")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(default=0, description="Sortino ratio")
    max_drawdown_pct: float = Field(default=0, description="Drawdown máximo (%)")
    win_rate: float = Field(default=0, description="Taxa de win (%)")
    average_profit_per_trade: float = Field(default=0, description="Lucro médio por trade")
    total_trades: int = Field(default=0, description="Total de trades")
    active_positions: int = Field(default=0, description="Posições ativas")
    cache_hit_rate: float = Field(default=0, description="Taxa de cache hit (%)")
    feature_extraction_ms: float = Field(default=0, description="Tempo de extração de features (ms)")
    ml_inference_ms: float = Field(default=0, description="Tempo de inferência ML (ms)")
    xai_generation_ms: float = Field(default=0, description="Tempo de geração XAI (ms)")


class SystemMetrics(BaseModel):
    """Response de métricas do sistema."""
    cpu_usage_pct: float = Field(..., description="Uso de CPU (%)")
    memory_usage_mb: float = Field(..., description="Uso de memória (MB)")
    disk_usage_pct: float = Field(..., description="Uso de disco (%)")
    network_in_mbps: float = Field(..., description="Network input (Mbps)")
    network_out_mbps: float = Field(..., description="Network output (Mbps)")
    active_connections: int = Field(..., description="Conexões ativas")
    error_rate_pct: float = Field(..., description="Taxa de erro (%)")
    uptime_hours: float = Field(..., description="Uptime em horas")


# ============================================
# Arbitrage Responses
# ============================================

class ArbitrageOpportunity(BaseModel):
    """Response de oportunidade de arbitragem."""
    id: str = Field(..., description="ID único da oportunidade")
    timestamp: float = Field(..., description="Timestamp da detecção")
    strategy: str = Field(..., description="Estratégia (spatial, statistical, triangular)")
    symbol: str = Field(..., description="Par de trading")
    buy_exchange: str = Field(..., description="Exchange para compra")
    sell_exchange: str = Field(..., description="Exchange para venda")
    buy_price: float = Field(..., ge=0, description="Preço de compra")
    sell_price: float = Field(..., ge=0, description="Preço de venda")
    spread_pct: float = Field(..., description="Spread percentual")
    gross_profit_pct: float = Field(..., description="Lucro bruto (%)")
    net_profit_pct: float = Field(..., description="Lucro líquido (%)")
    net_profit_usd: float = Field(..., description="Lucro líquido em USD")
    executable_volume: float = Field(..., ge=0, description="Volume executável")
    confidence: float = Field(..., ge=0, le=1, description="Score de confiança")
    risk_score: float = Field(..., ge=0, le=1, description="Score de risco")
    ml_prediction: Optional[float] = Field(None, description="Predição ML")
    expires_in_seconds: float = Field(..., description="Tempo até expirar")
    execution_estimate_ms: float = Field(default=0, description="Estimativa de execução (ms)")
    fees: Dict[str, float] = Field(default_factory=dict, description="Taxas detalhadas")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")


class ExecuteTradeResponse(BaseModel):
    """Response de execução de trade."""
    success: bool = Field(..., description="Se a execução foi bem-sucedida")
    trade_id: Optional[str] = Field(None, description="ID único do trade")
    status: TradeStatus = Field(..., description="Status do trade")
    executed_at: float = Field(..., description="Timestamp da execução")
    symbol: str = Field(..., description="Símbolo negociado")
    buy_exchange: str = Field(..., description="Exchange de compra")
    sell_exchange: str = Field(..., description="Exchange de venda")
    requested_volume: float = Field(..., description="Volume solicitado")
    executed_volume: float = Field(..., description="Volume executado")
    buy_price: float = Field(..., description="Preço de compra executado")
    sell_price: float = Field(..., description="Preço de venda executado")
    slippage_pct: float = Field(..., description="Slippage (%)")
    profit_usd: float = Field(..., description="Lucro em USD")
    profit_pct: float = Field(..., description="Lucro percentual")
    fees_usd: float = Field(..., description="Taxas totais em USD")
    execution_time_ms: float = Field(..., description="Tempo de execução (ms)")
    message: str = Field(..., description="Mensagem de status")
    errors: List[str] = Field(default_factory=list, description="Erros encontrados")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detalhes adicionais")


# ============================================
# Strategy Responses
# ============================================

class StrategyConfig(BaseModel):
    """Response de configuração de estratégia."""
    name: str = Field(..., description="Nome da estratégia")
    enabled: bool = Field(..., description="Se está habilitada")
    capital_allocation_pct: float = Field(..., description="Alocação de capital (%)")
    capital_allocated_usd: float = Field(..., description="Capital alocado (USD)")
    min_profit_pct: float = Field(..., description="Lucro mínimo (%)")
    max_position_size: float = Field(..., description="Tamanho máximo de posição")
    confidence_threshold: float = Field(..., description="Threshold de confiança")
    current_positions: int = Field(..., description="Posições atuais")
    daily_profit_usd: float = Field(..., description="Lucro diário (USD)")
    total_profit_usd: float = Field(..., description="Lucro total (USD)")
    win_rate_pct: float = Field(..., description="Taxa de win (%)")
    parameters: Dict[str, Any] = Field(..., description="Parâmetros específicos")
    last_trade: Optional[datetime] = Field(None, description="Último trade")
    status: str = Field(..., description="Status da estratégia")


class StrategyPerformance(BaseModel):
    """Response de performance de estratégia."""
    strategy: str = Field(..., description="Nome da estratégia")
    period: str = Field(..., description="Período analisado")
    total_trades: int = Field(..., description="Total de trades")
    successful_trades: int = Field(..., description="Trades bem-sucedidos")
    failed_trades: int = Field(..., description="Trades falhados")
    win_rate_pct: float = Field(..., description="Taxa de win (%)")
    total_profit_usd: float = Field(..., description="Lucro total (USD)")
    avg_profit_per_trade: float = Field(..., description="Lucro médio por trade")
    best_trade_usd: float = Field(..., description="Melhor trade (USD)")
    worst_trade_usd: float = Field(..., description="Pior trade (USD)")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown_pct: float = Field(..., description="Drawdown máximo (%)")
    profit_factor: float = Field(..., description="Fator de lucro")
    expectancy: float = Field(..., description="Expectativa")
    daily_stats: Dict[str, Any] = Field(default_factory=dict, description="Estatísticas diárias")


# ============================================
# Analysis Responses
# ============================================

class BacktestResult(BaseModel):
    """Response de resultado de backtest."""
    strategy: str = Field(..., description="Estratégia testada")
    period_start: datetime = Field(..., description="Início do período")
    period_end: datetime = Field(..., description="Fim do período")
    initial_capital: float = Field(..., description="Capital inicial")
    final_capital: float = Field(..., description="Capital final")
    total_return_pct: float = Field(..., description="Retorno total (%)")
    annualized_return_pct: float = Field(..., description="Retorno anualizado (%)")
    total_trades: int = Field(..., description="Total de trades")
    win_rate_pct: float = Field(..., description="Taxa de win (%)")
    profit_factor: float = Field(..., description="Fator de lucro")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    max_drawdown_pct: float = Field(..., description="Drawdown máximo (%)")
    var_95: float = Field(..., description="Value at Risk 95%")
    cvar_95: float = Field(..., description="Conditional VaR 95%")
    walk_forward_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Resultados walk-forward"
    )
    monte_carlo_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resultados Monte Carlo"
    )
    trade_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log de trades"
    )


class MarketAnalysis(BaseModel):
    """Response de análise de mercado."""
    timestamp: float = Field(..., description="Timestamp da análise")
    symbols_analyzed: List[str] = Field(..., description="Símbolos analisados")
    market_conditions: str = Field(..., description="Condições de mercado")
    volatility_index: float = Field(..., description="Índice de volatilidade")
    trend_direction: str = Field(..., description="Direção da tendência")
    trend_strength: float = Field(..., description="Força da tendência")
    correlations: Dict[str, float] = Field(
        default_factory=dict,
        description="Correlações entre pares"
    )
    liquidity_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas de liquidez"
    )
    arbitrage_potential: float = Field(..., description="Potencial de arbitragem")
    risk_level: str = Field(..., description="Nível de risco")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recomendações"
    )


class FeatureImportance(BaseModel):
    """Response de importância de features."""
    feature_name: str = Field(..., description="Nome da feature")
    importance: float = Field(..., ge=0, le=1, description="Score de importância")
    category: str = Field(..., description="Categoria da feature")
    rank: int = Field(..., description="Ranking")
    shap_value: Optional[float] = Field(None, description="SHAP value")
    description: str = Field(..., description="Descrição da feature")


class RiskMetrics(BaseModel):
    """Response de métricas de risco."""
    timestamp: float = Field(..., description="Timestamp")
    total_exposure_usd: float = Field(..., description="Exposição total (USD)")
    exposure_by_exchange: Dict[str, float] = Field(
        default_factory=dict,
        description="Exposição por exchange"
    )
    exposure_by_symbol: Dict[str, float] = Field(
        default_factory=dict,
        description="Exposição por símbolo"
    )
    var_95: float = Field(..., description="Value at Risk 95%")
    cvar_95: float = Field(..., description="Conditional VaR 95%")
    max_drawdown_pct: float = Field(..., description="Drawdown máximo (%)")
    current_drawdown_pct: float = Field(..., description="Drawdown atual (%)")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Matriz de correlação"
    )
    concentration_risk: float = Field(..., description="Risco de concentração")
    liquidity_risk: float = Field(..., description="Risco de liquidez")
    risk_score: float = Field(..., ge=0, le=1, description="Score de risco geral")
    warnings: List[str] = Field(default_factory=list, description="Avisos de risco")


# ============================================
# Position & Order Responses
# ============================================

class Position(BaseModel):
    """Response de posição."""
    id: str = Field(..., description="ID da posição")
    symbol: str = Field(..., description="Símbolo")
    exchange: str = Field(..., description="Exchange")
    side: str = Field(..., description="Lado (long/short)")
    entry_price: float = Field(..., description="Preço de entrada")
    current_price: float = Field(..., description="Preço atual")
    volume: float = Field(..., description="Volume")
    value_usd: float = Field(..., description="Valor em USD")
    unrealized_pnl_usd: float = Field(..., description="P&L não realizado (USD)")
    unrealized_pnl_pct: float = Field(..., description="P&L não realizado (%)")
    realized_pnl_usd: float = Field(..., description="P&L realizado (USD)")
    opened_at: datetime = Field(..., description="Data de abertura")
    strategy: str = Field(..., description="Estratégia utilizada")
    stop_loss: Optional[float] = Field(None, description="Stop loss")
    take_profit: Optional[float] = Field(None, description="Take profit")


class Order(BaseModel):
    """Response de ordem."""
    id: str = Field(..., description="ID da ordem")
    exchange: str = Field(..., description="Exchange")
    symbol: str = Field(..., description="Símbolo")
    side: str = Field(..., description="Lado (buy/sell)")
    order_type: str = Field(..., description="Tipo de ordem")
    status: str = Field(..., description="Status da ordem")
    volume: float = Field(..., description="Volume")
    price: Optional[float] = Field(None, description="Preço")
    executed_volume: float = Field(..., description="Volume executado")
    average_price: float = Field(..., description="Preço médio")
    created_at: datetime = Field(..., description="Data de criação")
    updated_at: datetime = Field(..., description="Última atualização")
    time_in_force: str = Field(..., description="Tempo de validade")
    fills: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Execuções parciais"
    )


# ============================================
# Alert & Notification Responses
# ============================================

class Alert(BaseModel):
    """Response de alerta."""
    id: str = Field(..., description="ID do alerta")
    timestamp: float = Field(..., description="Timestamp")
    severity: str = Field(..., description="Severidade (info, warning, error, critical)")
    category: str = Field(..., description="Categoria")
    title: str = Field(..., description="Título")
    message: str = Field(..., description="Mensagem")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detalhes")
    resolved: bool = Field(..., description="Se foi resolvido")
    resolved_at: Optional[float] = Field(None, description="Timestamp da resolução")
    actions_taken: List[str] = Field(default_factory=list, description="Ações tomadas")


# ============================================
# WebSocket Responses
# ============================================

class WebSocketMessage(BaseModel):
    """Response de mensagem WebSocket."""
    type: str = Field(..., description="Tipo de mensagem")
    channel: str = Field(..., description="Canal")
    data: Union[Dict[str, Any], List[Any]] = Field(..., description="Dados")
    timestamp: float = Field(..., description="Timestamp")
    sequence: Optional[int] = Field(None, description="Número de sequência")


class WebSocketSubscriptionConfirmation(BaseModel):
    """Response de confirmação de subscrição."""
    success: bool = Field(..., description="Se a subscrição foi bem-sucedida")
    subscribed_channels: List[str] = Field(..., description="Canais subscritos")
    message: str = Field(..., description="Mensagem")
    connection_id: str = Field(..., description="ID da conexão")


# ============================================
# Generic Responses
# ============================================

class SuccessResponse(BaseModel):
    """Response genérica de sucesso."""
    success: bool = Field(default=True, description="Sucesso")
    message: str = Field(..., description="Mensagem")
    data: Optional[Dict[str, Any]] = Field(None, description="Dados adicionais")


class ErrorResponse(BaseModel):
    """Response genérica de erro."""
    success: bool = Field(default=False, description="Sucesso")
    error: str = Field(..., description="Tipo de erro")
    message: str = Field(..., description="Mensagem de erro")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalhes do erro")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class PaginatedResponse(BaseModel):
    """Response paginada genérica."""
    total: int = Field(..., description="Total de itens")
    page: int = Field(..., description="Página atual")
    page_size: int = Field(..., description="Tamanho da página")
    total_pages: int = Field(..., description="Total de páginas")
    items: List[Any] = Field(..., description="Itens da página")
    has_next: bool = Field(..., description="Se tem próxima página")
    has_previous: bool = Field(..., description="Se tem página anterior")