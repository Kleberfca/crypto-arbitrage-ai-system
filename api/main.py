"""
FastAPI Main Application - Crypto Arbitrage AI System
Provides REST API endpoints for monitoring and control
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, validator
from starlette.responses import Response
import uvicorn

from config.settings import settings, Exchange, Strategy, Environment
from core.data_collector.exchange_connector import MultiExchangeConnector, OrderBookSnapshot
from core.arbitrage_engine.spatial_arbitrage import SpatialArbitrageEngine, ArbitrageOpportunity
from core.feature_engineering.feature_extractor import FeatureExtractor, FeatureVector


# ============================================
# Logging Configuration
# ============================================

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file or 'logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# Prometheus Metrics
# ============================================

# Custom metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
active_websocket_connections = Gauge('active_websocket_connections', 'Number of active WebSocket connections')
arbitrage_opportunities_found = Counter('arbitrage_opportunities_found', 'Total arbitrage opportunities found')
arbitrage_trades_executed = Counter('arbitrage_trades_executed', 'Total arbitrage trades executed')
total_profit_gauge = Gauge('total_profit_usd', 'Total profit in USD')
system_latency_gauge = Gauge('system_latency_ms', 'System latency in milliseconds')


# ============================================
# Global State Management
# ============================================

class SystemState:
    """Global system state management."""
    
    def __init__(self):
        self.connector: Optional[MultiExchangeConnector] = None
        self.spatial_arbitrage: Optional[SpatialArbitrageEngine] = None
        self.statistical_arbitrage = None  # Placeholder
        self.triangular_arbitrage = None  # Placeholder
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.is_initialized = False
        self.startup_time = None
        self.websocket_connections: Set[WebSocket] = set()
        self.background_tasks = []
        
        # Performance tracking
        self.metrics = {
            'opportunities_detected': 0,
            'trades_executed': 0,
            'total_profit_usd': 0.0,
            'success_rate': 0.0,
            'average_latency_ms': 0.0,
            'last_update': time.time()
        }


# Initialize system state
system_state = SystemState()


# ============================================
# Pydantic Models
# ============================================

class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    environment: str = Field(..., description="Current environment")
    services: Dict[str, Any] = Field(default_factory=dict, description="Service statuses")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Health checks")


class PerformanceMetrics(BaseModel):
    """Performance metrics response model."""
    latency_ms: float = Field(..., description="Average latency in milliseconds")
    throughput_per_sec: int = Field(..., description="Throughput per second")
    opportunities_detected: int = Field(..., description="Total opportunities detected")
    opportunities_executed: int = Field(..., description="Total opportunities executed")
    total_profit_usd: float = Field(..., description="Total profit in USD")
    success_rate: float = Field(..., description="Success rate percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    average_feature_extraction_ms: float = Field(..., description="Average feature extraction time")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    active_connections: int = Field(..., description="Number of active connections")


class ArbitrageOpportunityResponse(BaseModel):
    """Arbitrage opportunity response model."""
    id: str = Field(..., description="Unique opportunity ID")
    symbol: str = Field(..., description="Trading pair symbol")
    buy_exchange: str = Field(..., description="Exchange to buy from")
    sell_exchange: str = Field(..., description="Exchange to sell to")
    buy_price: float = Field(..., ge=0, description="Buy price")
    sell_price: float = Field(..., ge=0, description="Sell price")
    spread_pct: float = Field(..., description="Spread percentage")
    net_profit_pct: float = Field(..., description="Net profit percentage")
    net_profit_usd: float = Field(..., description="Net profit in USD")
    executable_volume: float = Field(..., ge=0, description="Executable volume")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")
    timestamp: float = Field(..., description="Opportunity timestamp")
    expires_in_seconds: float = Field(..., description="Time until opportunity expires")


class ExecuteTradeRequest(BaseModel):
    """Trade execution request model."""
    symbol: str = Field(..., description="Trading pair symbol")
    buy_exchange: str = Field(..., description="Exchange to buy from")
    sell_exchange: str = Field(..., description="Exchange to sell to")
    volume: float = Field(..., gt=0, description="Volume to trade")
    max_slippage_pct: float = Field(default=0.1, ge=0, le=1, description="Maximum slippage percentage")
    confidence_override: Optional[float] = Field(None, ge=0, le=1, description="Override confidence threshold")
    dry_run: bool = Field(default=True, description="Execute in dry-run mode")
    
    @validator('buy_exchange', 'sell_exchange')
    def validate_exchange(cls, v):
        valid_exchanges = [e.value for e in Exchange]
        if v not in valid_exchanges:
            raise ValueError(f"Exchange must be one of {valid_exchanges}")
        return v


class ExecuteTradeResponse(BaseModel):
    """Trade execution response model."""
    success: bool = Field(..., description="Whether execution was successful")
    trade_id: Optional[str] = Field(None, description="Unique trade ID")
    executed_volume: float = Field(..., ge=0, description="Volume executed")
    buy_price: float = Field(..., ge=0, description="Actual buy price")
    sell_price: float = Field(..., ge=0, description="Actual sell price")
    profit_usd: float = Field(..., description="Profit in USD")
    execution_time_ms: float = Field(..., ge=0, description="Execution time in milliseconds")
    message: str = Field(..., description="Execution message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    enabled: bool = Field(..., description="Whether strategy is enabled")
    capital_allocation_pct: float = Field(..., ge=0, le=1, description="Capital allocation percentage")
    min_profit_pct: float = Field(..., ge=0, description="Minimum profit percentage")
    max_position_size: float = Field(..., ge=0, le=1, description="Maximum position size")
    confidence_threshold: float = Field(..., ge=0, le=1, description="Confidence threshold")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class SystemConfig(BaseModel):
    """System configuration response model."""
    exchanges: List[str] = Field(..., description="Enabled exchanges")
    strategies: Dict[str, StrategyConfig] = Field(..., description="Strategy configurations")
    risk_limits: Dict[str, float] = Field(..., description="Risk limit settings")
    performance_targets: Dict[str, float] = Field(..., description="Performance targets")
    capital_allocation: Dict[str, float] = Field(..., description="Capital allocation settings")
    ml_config: Dict[str, Any] = Field(..., description="ML configuration")


class FeatureImportance(BaseModel):
    """Feature importance response model."""
    feature_name: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0, le=1, description="Importance score")
    category: str = Field(..., description="Feature category")
    description: Optional[str] = Field(None, description="Feature description")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: float = Field(..., description="Message timestamp")


# ============================================
# WebSocket Connection Manager
# ============================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = {
            'connected_at': time.time(),
            'subscriptions': set()
        }
        active_websocket_connections.set(len(self.active_connections))
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        self.connection_data.pop(websocket, None)
        active_websocket_connections.set(len(self.active_connections))
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
    
    async def broadcast(self, message: dict, subscription_type: Optional[str] = None):
        """Broadcast message to all or subscribed connections."""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                # If subscription_type specified, only send to subscribed connections
                if subscription_type:
                    subscriptions = self.connection_data.get(connection, {}).get('subscriptions', set())
                    if subscription_type not in subscriptions:
                        continue
                
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to send to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Initialize connection manager
manager = ConnectionManager()


# ============================================
# Lifespan Manager
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Initialize connections on startup, cleanup on shutdown.
    """
    logger.info("🚀 Starting Crypto Arbitrage AI System API...")
    
    try:
        # Initialize system components
        system_state.startup_time = time.time()
        
        # Initialize multi-exchange connector
        logger.info("Initializing exchange connections...")
        exchanges = settings.enabled_exchanges
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        
        system_state.connector = MultiExchangeConnector(exchanges, symbols)
        await system_state.connector.initialize()
        
        # Initialize arbitrage engines
        logger.info("Initializing arbitrage engines...")
        system_state.spatial_arbitrage = SpatialArbitrageEngine(system_state.connector)
        
        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        system_state.feature_extractor = FeatureExtractor()
        
        # Start background tasks
        logger.info("Starting background tasks...")
        task1 = asyncio.create_task(continuous_arbitrage_scan())
        task2 = asyncio.create_task(periodic_health_check())
        task3 = asyncio.create_task(metrics_updater())
        
        system_state.background_tasks = [task1, task2, task3]
        system_state.is_initialized = True
        
        logger.info("✅ System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        system_state.is_initialized = False
    
    yield
    
    # Cleanup
    logger.info("Shutting down Crypto Arbitrage AI System API...")
    
    # Cancel background tasks
    for task in system_state.background_tasks:
        task.cancel()
    
    # Wait for tasks to complete
    await asyncio.gather(*system_state.background_tasks, return_exceptions=True)
    
    # Close connections
    if system_state.connector:
        await system_state.connector.close()
    
    logger.info("✅ System shutdown complete")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Crypto Arbitrage AI System API",
    description="Professional cryptocurrency arbitrage system with AI-powered decision making",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if settings.environment == Environment.PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )

# Add Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)


# ============================================
# Health & Monitoring Endpoints
# ============================================

@app.get("/", tags=["Root"], response_class=HTMLResponse)
async def root():
    """Root endpoint with system information."""
    html_content = f"""
    <html>
        <head>
            <title>{settings.project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }}
                h1 {{ color: #00d4ff; }}
                .info {{ background: #2a2a2a; padding: 20px; border-radius: 8px; }}
                a {{ color: #00d4ff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>🚀 {settings.project_name}</h1>
            <div class="info">
                <p><strong>Version:</strong> {settings.version}</p>
                <p><strong>Environment:</strong> {settings.environment.value}</p>
                <p><strong>Status:</strong> {"🟢 Running" if system_state.is_initialized else "🔴 Not Initialized"}</p>
                <p><strong>API Documentation:</strong> <a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
                <p><strong>Health Check:</strong> <a href="/api/v1/health">/api/v1/health</a></p>
                <p><strong>Metrics:</strong> <a href="/metrics">/metrics</a></p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/v1/health", response_model=HealthStatus, tags=["Monitoring"])
async def health_check():
    """
    Comprehensive health check of all system components.
    
    Returns detailed status of each service and overall system health.
    """
    start_time = time.perf_counter()
    
    services = {}
    checks = {}
    
    # Check initialization
    checks['initialized'] = system_state.is_initialized
    
    # Check exchange connections
    if system_state.connector:
        try:
            connector_health = await system_state.connector.health_check()
            services["exchanges"] = connector_health
            checks['exchanges_connected'] = connector_health.get('all_connected', False)
        except Exception as e:
            services["exchanges"] = {"status": "error", "message": str(e)}
            checks['exchanges_connected'] = False
    else:
        services["exchanges"] = {"status": "not_initialized"}
        checks['exchanges_connected'] = False
    
    # Check Redis (mock for now)
    try:
        services["redis"] = {"status": "healthy", "connected": True}
        checks['redis_connected'] = True
    except:
        services["redis"] = {"status": "unhealthy", "connected": False}
        checks['redis_connected'] = False
    
    # Check PostgreSQL (mock for now)
    try:
        services["postgres"] = {"status": "healthy", "connected": True}
        checks['postgres_connected'] = True
    except:
        services["postgres"] = {"status": "unhealthy", "connected": False}
        checks['postgres_connected'] = False
    
    # Check arbitrage engines
    if system_state.spatial_arbitrage:
        services["spatial_arbitrage"] = {
            "status": "healthy",
            "stats": system_state.spatial_arbitrage.get_stats()
        }
        checks['spatial_arbitrage_ready'] = True
    else:
        services["spatial_arbitrage"] = {"status": "not_initialized"}
        checks['spatial_arbitrage_ready'] = False
    
    # Check feature extractor
    if system_state.feature_extractor:
        services["feature_extractor"] = {"status": "healthy", "features": 74}
        checks['feature_extractor_ready'] = True
    else:
        services["feature_extractor"] = {"status": "not_initialized"}
        checks['feature_extractor_ready'] = False
    
    # Calculate overall status
    all_healthy = all(checks.values())
    critical_checks = ['initialized', 'exchanges_connected']
    critical_healthy = all(checks.get(check, False) for check in critical_checks)
    
    if all_healthy:
        overall_status = "healthy"
    elif critical_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    # Calculate uptime
    uptime = time.time() - system_state.startup_time if system_state.startup_time else 0
    
    # Measure health check latency
    health_check_time = (time.perf_counter() - start_time) * 1000
    services["health_check_latency_ms"] = health_check_time
    
    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        uptime_seconds=uptime,
        version=settings.version,
        environment=settings.environment.value,
        services=services,
        checks=checks
    )


@app.get("/api/v1/performance", response_model=PerformanceMetrics, tags=["Monitoring"])
async def get_performance_metrics():
    """
    Get current system performance metrics.
    
    Returns real-time performance indicators including latency, throughput, and profitability.
    """
    if not system_state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Calculate average latency
    avg_latency = 0
    if system_state.connector:
        health = await system_state.connector.health_check()
        avg_latency = health.get("average_latency_ms", 0)
    
    # Calculate feature extraction time
    avg_feature_time = 0
    if system_state.feature_extractor and system_state.feature_extractor.extraction_times:
        avg_feature_time = sum(system_state.feature_extractor.extraction_times) / len(system_state.feature_extractor.extraction_times)
    
    # Get arbitrage stats
    arbitrage_stats = {}
    if system_state.spatial_arbitrage:
        arbitrage_stats = system_state.spatial_arbitrage.get_stats()
    
    return PerformanceMetrics(
        latency_ms=avg_latency,
        throughput_per_sec=10000,  # Mock for now
        opportunities_detected=arbitrage_stats.get("opportunities_detected", 0),
        opportunities_executed=arbitrage_stats.get("opportunities_executed", 0),
        total_profit_usd=arbitrage_stats.get("total_profit_usd", 0),
        success_rate=arbitrage_stats.get("success_rate", 0) * 100,
        sharpe_ratio=2.5,  # Mock for now
        average_feature_extraction_ms=avg_feature_time,
        cache_hit_rate=85.0,  # Mock for now
        active_connections=len(manager.active_connections)
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================
# Arbitrage Endpoints
# ============================================

@app.get("/api/v1/opportunities", response_model=List[ArbitrageOpportunityResponse], tags=["Arbitrage"])
async def get_arbitrage_opportunities(
    symbols: Optional[List[str]] = Query(default=None, description="Filter by symbols"),
    min_profit_pct: float = Query(default=0.3, ge=0, description="Minimum profit percentage"),
    min_confidence: float = Query(default=0.7, ge=0, le=1, description="Minimum confidence"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum opportunities to return")
):
    """
    Get current arbitrage opportunities.
    
    Returns list of profitable arbitrage opportunities across configured exchanges.
    """
    if not system_state.spatial_arbitrage:
        raise HTTPException(status_code=503, detail="Arbitrage engine not initialized")
    
    # Use default symbols if not specified
    if not symbols:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
    
    # Scan for opportunities
    opportunities = await system_state.spatial_arbitrage.scan_opportunities(symbols)
    
    # Filter by criteria
    filtered = [
        opp for opp in opportunities 
        if opp.net_profit_pct >= min_profit_pct and opp.confidence >= min_confidence
    ]
    
    # Sort by profit and limit
    filtered.sort(key=lambda x: x.net_profit_usd, reverse=True)
    filtered = filtered[:limit]
    
    # Update metrics
    arbitrage_opportunities_found.inc(len(filtered))
    
    # Convert to response model
    return [
        ArbitrageOpportunityResponse(
            id=f"{opp.symbol}_{opp.buy_exchange}_{opp.sell_exchange}_{int(opp.timestamp)}",
            symbol=opp.symbol,
            buy_exchange=opp.buy_exchange,
            sell_exchange=opp.sell_exchange,
            buy_price=opp.buy_price,
            sell_price=opp.sell_price,
            spread_pct=opp.spread_pct,
            net_profit_pct=opp.net_profit_pct,
            net_profit_usd=opp.net_profit_usd,
            executable_volume=opp.executable_volume,
            confidence=opp.confidence,
            risk_score=opp.risk_score,
            timestamp=opp.timestamp,
            expires_in_seconds=max(0, 5 - (time.time() - opp.timestamp))  # Assume 5 second validity
        )
        for opp in filtered
    ]


@app.post("/api/v1/execute", response_model=ExecuteTradeResponse, tags=["Arbitrage"])
async def execute_trade(request: ExecuteTradeRequest, background_tasks: BackgroundTasks):
    """
    Execute an arbitrage trade.
    
    Validates and executes arbitrage opportunity with risk management.
    """
    if not system_state.spatial_arbitrage:
        raise HTTPException(status_code=503, detail="Arbitrage engine not initialized")
    
    # Check if live trading is enabled
    if not request.dry_run and settings.environment != Environment.PRODUCTION:
        raise HTTPException(
            status_code=403,
            detail="Live trading only allowed in production environment"
        )
    
    start_time = time.perf_counter()
    
    try:
        # Get current orderbooks
        buy_ob = system_state.connector.connectors[request.buy_exchange].get_orderbook(request.symbol)
        sell_ob = system_state.connector.connectors[request.sell_exchange].get_orderbook(request.symbol)
        
        if not buy_ob or not sell_ob:
            raise HTTPException(status_code=400, detail="Unable to fetch orderbook data")
        
        # Create opportunity object
        opportunity = ArbitrageOpportunity(
            symbol=request.symbol,
            buy_exchange=request.buy_exchange,
            sell_exchange=request.sell_exchange,
            buy_price=buy_ob.best_ask[0],
            sell_price=sell_ob.best_bid[0],
            buy_volume=buy_ob.best_ask[1],
            sell_volume=sell_ob.best_bid[1],
            executable_volume=min(request.volume, buy_ob.best_ask[1], sell_ob.best_bid[1]),
            gross_profit_pct=((sell_ob.best_bid[0] - buy_ob.best_ask[0]) / buy_ob.best_ask[0]) * 100,
            net_profit_pct=0,  # Will be calculated
            net_profit_usd=0,
            confidence=request.confidence_override or 0.8,
            timestamp=time.time(),
            latency_ms=(buy_ob.latency_ms + sell_ob.latency_ms) / 2,
            fees={'buy': 0.001, 'sell': 0.001, 'transfer': 0}
        )
        
        # Calculate net profit
        total_fees = 0.2  # 0.1% each side
        opportunity.net_profit_pct = opportunity.gross_profit_pct - total_fees
        opportunity.net_profit_usd = (opportunity.net_profit_pct / 100) * (opportunity.buy_price * opportunity.executable_volume)
        
        if request.dry_run:
            # Simulate execution
            success = opportunity.net_profit_pct > 0
            message = "Dry run execution successful" if success else "Trade not profitable"
            trade_id = f"DRY-{int(time.time())}-{request.symbol.replace('/', '')}"
        else:
            # Real execution
            success = await system_state.spatial_arbitrage.execute_opportunity(opportunity)
            message = "Trade executed successfully" if success else "Trade execution failed"
            trade_id = f"TRADE-{int(time.time())}-{request.symbol.replace('/', '')}" if success else None
            
            # Update metrics
            if success:
                arbitrage_trades_executed.inc()
                system_state.metrics['trades_executed'] += 1
                system_state.metrics['total_profit_usd'] += opportunity.net_profit_usd
                total_profit_gauge.set(system_state.metrics['total_profit_usd'])
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return ExecuteTradeResponse(
            success=success,
            trade_id=trade_id,
            executed_volume=opportunity.executable_volume if success else 0,
            buy_price=opportunity.buy_price,
            sell_price=opportunity.sell_price,
            profit_usd=opportunity.net_profit_usd if success else 0,
            execution_time_ms=execution_time,
            message=message,
            details={
                'spread_pct': opportunity.spread_pct,
                'fees': opportunity.fees,
                'risk_score': opportunity.risk_score
            }
        )
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Configuration Endpoints
# ============================================

@app.get("/api/v1/config", response_model=SystemConfig, tags=["Configuration"])
async def get_system_config():
    """
    Get current system configuration.
    
    Returns complete system configuration including exchanges, strategies, and limits.
    """
    strategies = {}
    
    # Spatial Arbitrage Config
    strategies["spatial"] = StrategyConfig(
        enabled=Strategy.SPATIAL in settings.enabled_strategies,
        capital_allocation_pct=settings.capital.spatial_strategy_pct,
        min_profit_pct=0.3,
        max_position_size=settings.risk_limits.max_position_size_pct,
        confidence_threshold=settings.risk_limits.min_confidence_threshold,
        parameters={
            "pre_funding_enabled": True,
            "max_spread_pct": 5.0,
            "execution_timeout_ms": 100
        }
    )
    
    # Statistical Arbitrage Config
    strategies["statistical"] = StrategyConfig(
        enabled=Strategy.STATISTICAL in settings.enabled_strategies,
        capital_allocation_pct=settings.capital.statistical_strategy_pct,
        min_profit_pct=0.5,
        max_position_size=settings.risk_limits.max_position_size_pct,
        confidence_threshold=0.8,
        parameters={
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "lookback_period": 20
        }
    )
    
    # Triangular Arbitrage Config
    strategies["triangular"] = StrategyConfig(
        enabled=Strategy.TRIANGULAR in settings.enabled_strategies,
        capital_allocation_pct=settings.capital.triangular_strategy_pct,
        min_profit_pct=0.2,
        max_position_size=settings.risk_limits.max_position_size_pct * 0.5,
        confidence_threshold=0.9,
        parameters={
            "max_path_length": 4,
            "min_liquidity_usd": 1000,
            "execution_sequence": "parallel"
        }
    )
    
    return SystemConfig(
        exchanges=[ex.value for ex in settings.enabled_exchanges],
        strategies=strategies,
        risk_limits={
            "max_position_size_pct": settings.risk_limits.max_position_size_pct * 100,
            "max_daily_loss_pct": settings.risk_limits.max_daily_loss_pct * 100,
            "max_drawdown_pct": settings.risk_limits.max_drawdown_pct * 100,
            "min_sharpe_ratio": settings.risk_limits.min_sharpe_ratio,
            "var_95_limit_pct": settings.risk_limits.var_95_limit_pct * 100,
            "max_leverage": settings.risk_limits.max_leverage
        },
        performance_targets={
            "max_latency_ms": settings.performance.max_latency_ms,
            "ml_inference_ms": settings.performance.ml_inference_ms,
            "feature_extraction_ms": settings.performance.feature_extraction_ms,
            "xai_explanation_ms": settings.performance.xai_explanation_ms,
            "min_throughput_per_sec": settings.performance.min_throughput_per_sec,
            "target_monthly_roi_min": settings.performance.target_monthly_roi[0],
            "target_monthly_roi_max": settings.performance.target_monthly_roi[1]
        },
        capital_allocation={
            "pre_funding_pct": settings.capital.pre_funding_pct * 100,
            "rebalancing_pct": settings.capital.rebalancing_pct * 100,
            "reserve_pct": settings.capital.reserve_pct * 100
        },
        ml_config={
            "min_features": settings.ml_config.min_features,
            "ensemble_models": settings.ml_config.ensemble_models,
            "model_weights": settings.ml_config.model_weights,
            "confidence_threshold": settings.ml_config.confidence_threshold,
            "online_learning_enabled": settings.ml_config.online_learning_enabled,
            "retrain_interval_hours": settings.ml_config.retrain_interval_hours
        }
    )


@app.get("/api/v1/features/importance", response_model=List[FeatureImportance], tags=["Machine Learning"])
async def get_feature_importance():
    """
    Get feature importance from ML models.
    
    Returns ranked list of features by importance for model interpretability.
    """
    if not system_state.feature_extractor:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")
    
    importance = system_state.feature_extractor.get_feature_importance()
    
    # Create categorized feature list
    feature_categories = {
        "book_imbalance": "orderbook",
        "order_flow_imbalance": "orderbook",
        "rsi": "technical",
        "volume_imbalance": "volume",
        "volatility_5m": "price",
        "micro_price": "orderbook",
        "macd": "technical",
        "momentum": "price",
        "vwap": "volume",
        "spread_bps": "orderbook"
    }
    
    result = []
    for feature, score in importance.items():
        if feature != "others":
            result.append(FeatureImportance(
                feature_name=feature,
                importance=score,
                category=feature_categories.get(feature, "other"),
                description=f"Feature {feature} with importance {score:.3f}"
            ))
    
    # Sort by importance
    result.sort(key=lambda x: x.importance, reverse=True)
    return result


# ============================================
# WebSocket Endpoints
# ============================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time data streaming.
    
    Supports multiple subscription types:
    - opportunities: Real-time arbitrage opportunities
    - orderbook: Orderbook updates
    - trades: Trade executions
    - performance: Performance metrics
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get("action") == "subscribe":
                subscription_type = data.get("type")
                if subscription_type:
                    manager.connection_data[websocket]['subscriptions'].add(subscription_type)
                    await manager.send_personal_message({
                        "type": "subscription",
                        "status": "subscribed",
                        "subscription": subscription_type,
                        "timestamp": time.time()
                    }, websocket)
            
            # Handle unsubscribe requests
            elif data.get("action") == "unsubscribe":
                subscription_type = data.get("type")
                if subscription_type:
                    manager.connection_data[websocket]['subscriptions'].discard(subscription_type)
                    await manager.send_personal_message({
                        "type": "subscription",
                        "status": "unsubscribed",
                        "subscription": subscription_type,
                        "timestamp": time.time()
                    }, websocket)
            
            # Handle ping/pong for keepalive
            elif data.get("action") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": time.time()
                }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/opportunities")
async def websocket_opportunities(websocket: WebSocket):
    """
    Dedicated WebSocket for streaming arbitrage opportunities.
    
    Streams real-time arbitrage opportunities as they are detected.
    """
    await manager.connect(websocket)
    manager.connection_data[websocket]['subscriptions'].add('opportunities')
    
    try:
        while True:
            # Send opportunities every second
            if system_state.spatial_arbitrage:
                opportunities = await system_state.spatial_arbitrage.scan_opportunities(
                    ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                )
                
                if opportunities:
                    best = opportunities[0]
                    await websocket.send_json({
                        "type": "opportunity",
                        "data": {
                            "symbol": best.symbol,
                            "buy_exchange": best.buy_exchange,
                            "sell_exchange": best.sell_exchange,
                            "spread_pct": best.spread_pct,
                            "profit_pct": best.net_profit_pct,
                            "profit_usd": best.net_profit_usd,
                            "confidence": best.confidence,
                            "volume": best.executable_volume
                        },
                        "timestamp": time.time()
                    })
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================
# Background Tasks
# ============================================

async def continuous_arbitrage_scan():
    """Background task for continuous arbitrage scanning."""
    logger.info("Starting continuous arbitrage scanning...")
    
    while True:
        try:
            if system_state.spatial_arbitrage:
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
                opportunities = await system_state.spatial_arbitrage.scan_opportunities(symbols)
                
                # Update metrics
                system_state.metrics['opportunities_detected'] = len(opportunities)
                
                # Broadcast to WebSocket subscribers
                if opportunities:
                    best = opportunities[0]
                    await manager.broadcast({
                        "type": "opportunity",
                        "data": {
                            "symbol": best.symbol,
                            "profit_pct": best.net_profit_pct,
                            "profit_usd": best.net_profit_usd,
                            "confidence": best.confidence
                        },
                        "timestamp": time.time()
                    }, subscription_type="opportunities")
                
                # Auto-execute high-confidence opportunities (if enabled)
                if settings.environment == Environment.PRODUCTION:
                    for opp in opportunities[:3]:  # Check top 3
                        if opp.confidence >= 0.9 and opp.net_profit_pct >= 0.5:
                            success = await system_state.spatial_arbitrage.execute_opportunity(opp)
                            if success:
                                system_state.metrics['trades_executed'] += 1
                                system_state.metrics['total_profit_usd'] += opp.net_profit_usd
            
            await asyncio.sleep(0.1)  # 100ms scan interval
            
        except Exception as e:
            logger.error(f"Error in continuous scan: {e}")
            await asyncio.sleep(1)


async def periodic_health_check():
    """Background task for periodic health checks."""
    logger.info("Starting periodic health check...")
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            if system_state.connector:
                health = await system_state.connector.health_check()
                
                # Update latency metric
                if 'average_latency_ms' in health:
                    system_state.metrics['average_latency_ms'] = health['average_latency_ms']
                    system_latency_gauge.set(health['average_latency_ms'])
                
                # Log any unhealthy services
                for exchange, status in health.get("connectors", {}).items():
                    if not status.get("connected", False):
                        logger.warning(f"Exchange {exchange} is disconnected")
                
                # Broadcast health status
                await manager.broadcast({
                    "type": "health",
                    "data": health,
                    "timestamp": time.time()
                }, subscription_type="health")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")


async def metrics_updater():
    """Background task to update Prometheus metrics."""
    logger.info("Starting metrics updater...")
    
    while True:
        try:
            await asyncio.sleep(10)  # Update every 10 seconds
            
            # Update Prometheus gauges
            if system_state.spatial_arbitrage:
                stats = system_state.spatial_arbitrage.get_stats()
                total_profit_gauge.set(stats.get('total_profit_usd', 0))
            
            if system_state.metrics.get('average_latency_ms'):
                system_latency_gauge.set(system_state.metrics['average_latency_ms'])
            
            active_websocket_connections.set(len(manager.active_connections))
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=exc.status_code
    ).inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=500
    ).inc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.environment == Environment.PRODUCTION else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        loop="uvloop"
    )