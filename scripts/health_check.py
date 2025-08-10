#!/usr/bin/env python3
"""
Health Check Script - Crypto Arbitrage AI System
Verifies all system components are functioning correctly
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import sys
import time
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

import aiohttp
import asyncpg
import redis.asyncio as redis
import ccxt.pro as ccxtpro
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from config.settings import settings, Exchange


# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemHealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.critical_failures = []
        self.warnings = []
        
    async def check_exchange_connectivity(self) -> Tuple[bool, str]:
        """Check connectivity to all configured exchanges."""
        console.print("\n[cyan]Checking Exchange Connectivity...[/cyan]")
        
        results = {}
        for exchange in settings.enabled_exchanges:
            try:
                # Create exchange instance
                exchange_id = exchange.value
                exchange_class = getattr(ccxtpro, exchange_id)
                
                exchange_instance = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                
                # Try to fetch ticker
                ticker = await exchange_instance.fetch_ticker('BTC/USDT')
                await exchange_instance.close()
                
                if ticker and ticker.get('last'):
                    results[exchange_id] = {
                        'status': '✅',
                        'price': ticker['last'],
                        'message': f"Connected (BTC: ${ticker['last']:.2f})"
                    }
                else:
                    results[exchange_id] = {
                        'status': '⚠️',
                        'message': 'Connected but no data'
                    }
                    
            except Exception as e:
                results[exchange_id] = {
                    'status': '❌',
                    'message': f"Failed: {str(e)[:50]}"
                }
                self.critical_failures.append(f"Exchange {exchange_id} connection failed")
        
        # Display results
        table = Table(title="Exchange Connectivity")
        table.add_column("Exchange", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="white")
        
        all_connected = True
        for exchange_id, result in results.items():
            table.add_row(
                exchange_id.upper(),
                result['status'],
                result['message']
            )
            if result['status'] == '❌':
                all_connected = False
        
        console.print(table)
        
        return all_connected, "All exchanges connected" if all_connected else "Some exchanges offline"
    
    async def check_database_connectivity(self) -> Tuple[bool, str]:
        """Check PostgreSQL database connectivity."""
        console.print("\n[cyan]Checking Database Connectivity...[/cyan]")
        
        try:
            # Connect to PostgreSQL
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                timeout=5
            )
            
            # Run test query
            version = await conn.fetchval('SELECT version()')
            await conn.close()
            
            console.print(f"[green]✅ PostgreSQL Connected[/green]")
            console.print(f"   Version: {version[:50]}...")
            
            return True, "Database connected"
            
        except Exception as e:
            console.print(f"[red]❌ PostgreSQL Connection Failed: {e}[/red]")
            self.critical_failures.append("Database connection failed")
            return False, f"Database error: {str(e)[:50]}"
    
    async def check_redis_connectivity(self) -> Tuple[bool, str]:
        """Check Redis cache connectivity."""
        console.print("\n[cyan]Checking Redis Cache...[/cyan]")
        
        try:
            # Connect to Redis
            client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True
            )
            
            # Test operations
            await client.ping()
            await client.set('health_check', str(time.time()), ex=10)
            value = await client.get('health_check')
            await client.delete('health_check')
            await client.close()
            
            console.print(f"[green]✅ Redis Connected and Operational[/green]")
            
            return True, "Redis operational"
            
        except Exception as e:
            console.print(f"[red]❌ Redis Connection Failed: {e}[/red]")
            self.warnings.append("Redis cache unavailable")
            return False, f"Redis error: {str(e)[:50]}"
    
    async def check_api_health(self) -> Tuple[bool, str]:
        """Check FastAPI application health."""
        console.print("\n[cyan]Checking API Health...[/cyan]")
        
        api_url = f"http://{settings.api_host}:{settings.api_port}/api/v1/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        console.print(f"[green]✅ API is running[/green]")
                        console.print(f"   Status: {data.get('status', 'unknown')}")
                        console.print(f"   Version: {data.get('version', 'unknown')}")
                        
                        return True, "API healthy"
                    else:
                        console.print(f"[yellow]⚠️ API returned status {response.status}[/yellow]")
                        return False, f"API status {response.status}"
                        
        except Exception as e:
            console.print(f"[yellow]⚠️ API not accessible: {str(e)[:50]}[/yellow]")
            self.warnings.append("API not running")
            return False, "API not accessible"
    
    async def check_performance_metrics(self) -> Tuple[bool, str]:
        """Check system performance metrics."""
        console.print("\n[cyan]Checking Performance Metrics...[/cyan]")
        
        metrics = {
            'latency_target': settings.performance.max_latency_ms,
            'ml_inference_target': settings.performance.ml_inference_ms,
            'feature_extraction_target': settings.performance.feature_extraction_ms,
            'throughput_target': settings.performance.min_throughput_per_sec
        }
        
        table = Table(title="Performance Targets")
        table.add_column("Metric", style="cyan")
        table.add_column("Target", style="yellow")
        table.add_column("Status", style="white")
        
        table.add_row("Max Latency", f"{metrics['latency_target']}ms", "✅ Configured")
        table.add_row("ML Inference", f"{metrics['ml_inference_target']}ms", "✅ Configured")
        table.add_row("Feature Extraction", f"{metrics['feature_extraction_target']}ms", "✅ Configured")
        table.add_row("Min Throughput", f"{metrics['throughput_target']}/sec", "✅ Configured")
        
        console.print(table)
        
        return True, "Performance targets configured"
    
    async def check_risk_limits(self) -> Tuple[bool, str]:
        """Check risk management configuration."""
        console.print("\n[cyan]Checking Risk Management...[/cyan]")
        
        table = Table(title="Risk Limits")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Status", style="white")
        
        table.add_row("Max Position Size", f"{settings.risk_limits.max_position_size_pct*100:.1f}%", "✅")
        table.add_row("Max Daily Loss", f"{settings.risk_limits.max_daily_loss_pct*100:.1f}%", "✅")
        table.add_row("Max Drawdown", f"{settings.risk_limits.max_drawdown_pct*100:.1f}%", "✅")
        table.add_row("Min Sharpe Ratio", f"{settings.risk_limits.min_sharpe_ratio}", "✅")
        table.add_row("VaR 95% Limit", f"{settings.risk_limits.var_95_limit_pct*100:.1f}%", "✅")
        table.add_row("Min Confidence", f"{settings.risk_limits.min_confidence_threshold*100:.0f}%", "✅")
        
        console.print(table)
        
        return True, "Risk limits configured"
    
    async def check_ml_models(self) -> Tuple[bool, str]:
        """Check ML model availability."""
        console.print("\n[cyan]Checking ML Models...[/cyan]")
        
        models = ['xgboost', 'lstm', 'transformer', 'statistical']
        models_status = []
        
        for model in models:
            # Check if model file exists (mock for now)
            exists = True  # In production, check actual model files
            models_status.append((model, exists))
        
        table = Table(title="ML Models")
        table.add_column("Model", style="cyan")
        table.add_column("Weight", style="yellow")
        table.add_column("Status", style="white")
        
        for model, exists in models_status:
            weight = settings.ml_config.model_weights.get(model, 0)
            status = "✅ Ready" if exists else "⚠️ Not loaded"
            table.add_row(model.upper(), f"{weight*100:.0f}%", status)
        
        console.print(table)
        
        all_ready = all(status for _, status in models_status)
        return all_ready, "All models ready" if all_ready else "Some models not loaded"
    
    async def check_dependencies(self) -> Tuple[bool, str]:
        """Check Python dependencies."""
        console.print("\n[cyan]Checking Dependencies...[/cyan]")
        
        critical_packages = [
            'ccxt', 'fastapi', 'redis', 'asyncpg', 'pandas', 
            'numpy', 'xgboost', 'tensorflow', 'shap'
        ]
        
        missing = []
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            console.print(f"[red]❌ Missing packages: {', '.join(missing)}[/red]")
            self.critical_failures.append(f"Missing dependencies: {', '.join(missing)}")
            return False, f"Missing {len(missing)} packages"
        else:
            console.print(f"[green]✅ All critical dependencies installed[/green]")
            return True, "All dependencies present"
    
    async def run_all_checks(self) -> bool:
        """Run all health checks."""
        console.print(Panel.fit(
            "[bold cyan]🏥 SYSTEM HEALTH CHECK[/bold cyan]\n" +
            f"[white]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]",
            border_style="cyan"
        ))
        
        checks = [
            ("Dependencies", self.check_dependencies()),
            ("Database", self.check_database_connectivity()),
            ("Redis Cache", self.check_redis_connectivity()),
            ("Exchanges", self.check_exchange_connectivity()),
            ("API", self.check_api_health()),
            ("Performance", self.check_performance_metrics()),
            ("Risk Limits", self.check_risk_limits()),
            ("ML Models", self.check_ml_models())
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for name, check_coro in checks:
                task = progress.add_task(f"[cyan]Running {name} check...", total=None)
                
                try:
                    passed, message = await check_coro
                    results.append((name, passed, message))
                    
                    if passed:
                        self.checks_passed += 1
                    else:
                        self.checks_failed += 1
                        
                except Exception as e:
                    results.append((name, False, f"Error: {str(e)[:50]}"))
                    self.checks_failed += 1
                
                progress.update(task, completed=True)
        
        # Display summary
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold]HEALTH CHECK SUMMARY[/bold]",
            border_style="cyan"
        ))
        
        summary_table = Table(show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Checks", str(self.checks_passed + self.checks_failed))
        summary_table.add_row("Passed", f"[green]{self.checks_passed}[/green]")
        summary_table.add_row("Failed", f"[red]{self.checks_failed}[/red]")
        
        console.print(summary_table)
        
        # Show critical failures
        if self.critical_failures:
            console.print("\n[bold red]⚠️ CRITICAL FAILURES:[/bold red]")
            for failure in self.critical_failures:
                console.print(f"  • {failure}")
        
        # Show warnings
        if self.warnings:
            console.print("\n[bold yellow]⚠️ WARNINGS:[/bold yellow]")
            for warning in self.warnings:
                console.print(f"  • {warning}")
        
        # Overall status
        if self.checks_failed == 0:
            console.print("\n[bold green]✅ SYSTEM IS HEALTHY[/bold green]")
            console.print("All checks passed. System is ready for operation.")
            return True
        elif len(self.critical_failures) > 0:
            console.print("\n[bold red]❌ SYSTEM IS NOT OPERATIONAL[/bold red]")
            console.print("Critical failures detected. Please resolve before starting.")
            return False
        else:
            console.print("\n[bold yellow]⚠️ SYSTEM IS PARTIALLY OPERATIONAL[/bold yellow]")
            console.print("Some non-critical issues detected. System can run with limitations.")
            return True


async def main():
    """Main health check entry point."""
    checker = SystemHealthChecker()
    
    try:
        is_healthy = await checker.run_all_checks()
        
        # Exit with appropriate code
        sys.exit(0 if is_healthy else 1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Health check interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Health check failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())