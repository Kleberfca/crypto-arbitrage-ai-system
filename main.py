"""
Main Execution Script - Crypto Arbitrage AI System
Orchestrates all components and manages the trading lifecycle
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import signal
import sys
import time
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import argparse
import json

import uvloop
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings, Exchange, Strategy
from core.data_collector.exchange_connector import MultiExchangeConnector
from core.arbitrage_engine.spatial_arbitrage import SpatialArbitrageEngine
from core.feature_engineering.feature_extractor import FeatureExtractor


# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file or 'logs/arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


class CryptoArbitrageSystem:
    """
    Main system orchestrator for crypto arbitrage.
    Manages all components and coordinates execution.
    """
    
    def __init__(self, mode: str = "arbitrage", symbols: List[str] = None):
        """
        Initialize the arbitrage system.
        
        Args:
            mode: Execution mode (arbitrage, backtest, monitor)
            symbols: List of symbols to trade
        """
        self.mode = mode
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        
        # Core components
        self.connector: Optional[MultiExchangeConnector] = None
        self.spatial_arbitrage: Optional[SpatialArbitrageEngine] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        
        # State management
        self.is_running = False
        self.start_time: Optional[float] = None
        self.total_profit = 0.0
        self.total_trades = 0
        
        # Performance tracking
        self.performance_stats = {
            'opportunities_found': 0,
            'trades_executed': 0,
            'total_profit_usd': 0.0,
            'average_latency_ms': 0.0,
            'success_rate': 0.0,
            'best_profit_usd': 0.0,
            'worst_loss_usd': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Risk management
        self.daily_loss = 0.0
        self.max_daily_loss = settings.risk_limits.max_daily_loss_pct
        self.current_positions = {}
        
    async def initialize(self) -> None:
        """Initialize all system components."""
        console.print("\n[bold green]🚀 Initializing Crypto Arbitrage AI System...[/bold green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize Exchange Connector
            task = progress.add_task("[cyan]Connecting to exchanges...", total=None)
            self.connector = MultiExchangeConnector(
                settings.enabled_exchanges,
                self.symbols
            )
            await self.connector.initialize()
            progress.update(task, completed=True)
            
            # Initialize Arbitrage Engines
            task = progress.add_task("[cyan]Initializing arbitrage engines...", total=None)
            self.spatial_arbitrage = SpatialArbitrageEngine(self.connector)
            progress.update(task, completed=True)
            
            # Initialize Feature Extractor
            task = progress.add_task("[cyan]Setting up feature extraction...", total=None)
            self.feature_extractor = FeatureExtractor()
            progress.update(task, completed=True)
            
            # Initialize ML Models (mock for now)
            task = progress.add_task("[cyan]Loading ML models...", total=None)
            await asyncio.sleep(0.5)  # Simulate model loading
            progress.update(task, completed=True)
        
        self.start_time = time.time()
        self.is_running = True
        
        # Print system status
        self._print_system_status()
        
        console.print("\n[bold green]✅ System initialized successfully![/bold green]\n")
    
    def _print_system_status(self) -> None:
        """Print beautiful system status table."""
        table = Table(title="System Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Environment", settings.environment.value)
        table.add_row("Exchanges", ", ".join([e.value for e in settings.enabled_exchanges]))
        table.add_row("Symbols", ", ".join(self.symbols))
        table.add_row("Mode", self.mode)
        table.add_row("Max Latency Target", f"{settings.performance.max_latency_ms}ms")
        table.add_row("Min Profit %", f"{0.3}%")
        table.add_row("Max Position Size", f"{settings.risk_limits.max_position_size_pct*100}%")
        table.add_row("Pre-funding", f"{settings.capital.pre_funding_pct*100}%")
        table.add_row("Risk Limits", f"Daily: {self.max_daily_loss*100}%, DD: {settings.risk_limits.max_drawdown_pct*100}%")
        
        console.print(table)
    
    async def run_arbitrage_mode(self) -> None:
        """Run in arbitrage trading mode."""
        console.print("[bold yellow]📊 Starting Arbitrage Trading Mode[/bold yellow]\n")
        
        # Create tasks for different strategies
        tasks = []
        
        if Strategy.SPATIAL in settings.enabled_strategies:
            tasks.append(asyncio.create_task(self._run_spatial_arbitrage()))
        
        if Strategy.STATISTICAL in settings.enabled_strategies:
            tasks.append(asyncio.create_task(self._run_statistical_arbitrage()))
        
        if Strategy.TRIANGULAR in settings.enabled_strategies:
            tasks.append(asyncio.create_task(self._run_triangular_arbitrage()))
        
        # Add monitoring task
        tasks.append(asyncio.create_task(self._monitor_performance()))
        
        # Run all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_spatial_arbitrage(self) -> None:
        """Run spatial arbitrage strategy."""
        logger.info("Starting spatial arbitrage strategy")
        
        while self.is_running:
            try:
                # Check risk limits
                if self._check_risk_limits():
                    logger.warning("Risk limits reached, pausing spatial arbitrage")
                    await asyncio.sleep(60)
                    continue
                
                # Scan for opportunities
                start_time = time.perf_counter()
                opportunities = await self.spatial_arbitrage.scan_opportunities(self.symbols)
                scan_time = (time.perf_counter() - start_time) * 1000
                
                # Update stats
                self.performance_stats['opportunities_found'] += len(opportunities)
                self.performance_stats['average_latency_ms'] = (
                    self.performance_stats['average_latency_ms'] * 0.95 + scan_time * 0.05
                )
                
                # Execute best opportunity if profitable
                if opportunities:
                    best = opportunities[0]
                    
                    # Extract features for ML prediction
                    if self.feature_extractor and self.connector:
                        symbol = best.symbol
                        buy_ob = self.connector.connectors[best.buy_exchange].get_orderbook(symbol)
                        sell_ob = self.connector.connectors[best.sell_exchange].get_orderbook(symbol)
                        
                        if buy_ob:
                            features = await self.feature_extractor.extract_features(
                                symbol, best.buy_exchange, buy_ob
                            )
                            
                            # Check ML confidence (mock for now)
                            ml_confidence = 0.8 + np.random.random() * 0.2
                            best.confidence = best.confidence * 0.5 + ml_confidence * 0.5
                    
                    # Execute if confidence is high
                    if best.confidence >= settings.risk_limits.min_confidence_threshold:
                        if best.net_profit_pct >= 0.3:  # Min 0.3% profit
                            success = await self.spatial_arbitrage.execute_opportunity(best)
                            
                            if success:
                                self.total_trades += 1
                                self.total_profit += best.net_profit_usd
                                self.performance_stats['trades_executed'] += 1
                                self.performance_stats['total_profit_usd'] += best.net_profit_usd
                                
                                # Update best/worst
                                if best.net_profit_usd > self.performance_stats['best_profit_usd']:
                                    self.performance_stats['best_profit_usd'] = best.net_profit_usd
                                
                                self._log_trade(best)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.05)  # 50ms between scans
                
            except Exception as e:
                logger.error(f"Error in spatial arbitrage: {e}")
                await asyncio.sleep(1)
    
    async def _run_statistical_arbitrage(self) -> None:
        """Run statistical arbitrage strategy (placeholder)."""
        logger.info("Starting statistical arbitrage strategy")
        
        while self.is_running:
            try:
                # Placeholder for statistical arbitrage
                # Will implement pairs trading, mean reversion, etc.
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in statistical arbitrage: {e}")
                await asyncio.sleep(5)
    
    async def _run_triangular_arbitrage(self) -> None:
        """Run triangular arbitrage strategy (placeholder)."""
        logger.info("Starting triangular arbitrage strategy")
        
        while self.is_running:
            try:
                # Placeholder for triangular arbitrage
                # Will implement cycle detection, Bellman-Ford, etc.
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in triangular arbitrage: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_performance(self) -> None:
        """Monitor and display performance metrics."""
        
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Calculate metrics
                uptime = time.time() - self.start_time if self.start_time else 0
                success_rate = (
                    self.performance_stats['trades_executed'] / 
                    max(self.performance_stats['opportunities_found'], 1) * 100
                )
                
                # Create performance table
                table = Table(title=f"📈 Performance Monitor - {datetime.now().strftime('%H:%M:%S')}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Uptime", f"{uptime/3600:.1f} hours")
                table.add_row("Opportunities Found", str(self.performance_stats['opportunities_found']))
                table.add_row("Trades Executed", str(self.performance_stats['trades_executed']))
                table.add_row("Success Rate", f"{success_rate:.1f}%")
                table.add_row("Total Profit", f"${self.performance_stats['total_profit_usd']:.2f}")
                table.add_row("Best Trade", f"${self.performance_stats['best_profit_usd']:.2f}")
                table.add_row("Avg Latency", f"{self.performance_stats['average_latency_ms']:.1f}ms")
                table.add_row("Daily Loss", f"${self.daily_loss:.2f}")
                
                # Check exchange health
                if self.connector:
                    health = await self.connector.health_check()
                    table.add_row("Exchanges", "✅ All Connected" if health['all_connected'] else "⚠️ Some Disconnected")
                
                console.clear()
                console.print(table)
                
                # Show recent opportunities
                if self.spatial_arbitrage and self.spatial_arbitrage.opportunity_buffer:
                    recent_table = Table(title="Recent Opportunities")
                    recent_table.add_column("Symbol")
                    recent_table.add_column("Buy")
                    recent_table.add_column("Sell")
                    recent_table.add_column("Profit %")
                    recent_table.add_column("Profit $")
                    recent_table.add_column("Confidence")
                    
                    for opp in self.spatial_arbitrage.opportunity_buffer[:5]:
                        recent_table.add_row(
                            opp.symbol,
                            opp.buy_exchange,
                            opp.sell_exchange,
                            f"{opp.net_profit_pct:.2f}%",
                            f"${opp.net_profit_usd:.2f}",
                            f"{opp.confidence:.2f}"
                        )
                    
                    console.print("\n")
                    console.print(recent_table)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    def _check_risk_limits(self) -> bool:
        """
        Check if risk limits are breached.
        
        Returns:
            True if limits breached, False otherwise
        """
        # Check daily loss limit
        if abs(self.daily_loss) > self.max_daily_loss * 10000:  # Assuming $10k capital
            logger.warning(f"Daily loss limit reached: ${self.daily_loss:.2f}")
            return True
        
        # Check drawdown
        # TODO: Implement drawdown calculation
        
        return False
    
    def _log_trade(self, opportunity: Any) -> None:
        """Log executed trade."""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': opportunity.symbol,
            'buy_exchange': opportunity.buy_exchange,
            'sell_exchange': opportunity.sell_exchange,
            'buy_price': opportunity.buy_price,
            'sell_price': opportunity.sell_price,
            'volume': opportunity.executable_volume,
            'profit_pct': opportunity.net_profit_pct,
            'profit_usd': opportunity.net_profit_usd,
            'confidence': opportunity.confidence
        }
        
        logger.info(f"Trade executed: {json.dumps(trade_log)}")
        
        # Also save to file
        with open('logs/trades.jsonl', 'a') as f:
            f.write(json.dumps(trade_log) + '\n')
    
    async def run_backtest_mode(self) -> None:
        """Run in backtest mode."""
        console.print("[bold yellow]📊 Starting Backtest Mode[/bold yellow]\n")
        console.print("[red]Backtest mode not yet implemented[/red]")
        # TODO: Implement backtesting
    
    async def run_monitor_mode(self) -> None:
        """Run in monitor-only mode."""
        console.print("[bold yellow]👁️ Starting Monitor Mode (No Trading)[/bold yellow]\n")
        
        while self.is_running:
            try:
                # Just scan and display opportunities
                opportunities = await self.spatial_arbitrage.scan_opportunities(self.symbols)
                
                if opportunities:
                    table = Table(title="Available Opportunities")
                    table.add_column("Symbol")
                    table.add_column("Buy Exchange")
                    table.add_column("Sell Exchange")
                    table.add_column("Spread %")
                    table.add_column("Net Profit %")
                    table.add_column("Profit USD")
                    table.add_column("Confidence")
                    
                    for opp in opportunities[:10]:
                        table.add_row(
                            opp.symbol,
                            opp.buy_exchange,
                            opp.sell_exchange,
                            f"{opp.spread_pct:.3f}%",
                            f"{opp.net_profit_pct:.3f}%",
                            f"${opp.net_profit_usd:.2f}",
                            f"{opp.confidence:.2f}"
                        )
                    
                    console.clear()
                    console.print(table)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor mode: {e}")
                await asyncio.sleep(5)
    
    async def run(self) -> None:
        """Main run method."""
        try:
            await self.initialize()
            
            if self.mode == "arbitrage":
                await self.run_arbitrage_mode()
            elif self.mode == "backtest":
                await self.run_backtest_mode()
            elif self.mode == "monitor":
                await self.run_monitor_mode()
            else:
                console.print(f"[red]Unknown mode: {self.mode}[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Received interrupt signal[/yellow]")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            console.print(f"[red]Fatal error: {e}[/red]")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        console.print("\n[yellow]Shutting down system...[/yellow]")
        
        self.is_running = False
        
        # Close all connections
        if self.connector:
            await self.connector.close()
        
        # Print final stats
        if self.start_time:
            runtime = time.time() - self.start_time
            
            table = Table(title="Final Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Runtime", f"{runtime/3600:.2f} hours")
            table.add_row("Total Trades", str(self.total_trades))
            table.add_row("Total Profit", f"${self.total_profit:.2f}")
            table.add_row("Opportunities Found", str(self.performance_stats['opportunities_found']))
            table.add_row("Success Rate", f"{self.performance_stats.get('success_rate', 0):.1f}%")
            
            console.print("\n")
            console.print(table)
        
        console.print("\n[green]✅ System shutdown complete[/green]")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    console.print("\n[yellow]Signal received, initiating shutdown...[/yellow]")
    sys.exit(0)


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Crypto Arbitrage AI System")
    parser.add_argument(
        "--mode",
        choices=["arbitrage", "backtest", "monitor"],
        default="arbitrage",
        help="Execution mode"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print banner
    console.print("""
    [bold cyan]
    ╔═══════════════════════════════════════════════════════╗
    ║     🚀 CRYPTO ARBITRAGE AI SYSTEM v1.0.0 🚀         ║
    ║     Professional Cryptocurrency Arbitrage Bot        ║
    ║     Target: <30ms latency, 10-15% monthly ROI       ║
    ╚═══════════════════════════════════════════════════════╝
    [/bold cyan]
    """)
    
    # Create and run system
    system = CryptoArbitrageSystem(mode=args.mode, symbols=args.symbols)
    
    # Run async main
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()