#!/usr/bin/env python3
"""
Performance Monitoring Script - Crypto Arbitrage AI System
Real-time monitoring of system performance metrics
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import asyncio
import time
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import logging

import aiohttp
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.chart import Chart
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import settings


# Setup
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, api_url: str = None):
        """Initialize performance monitor."""
        self.api_url = api_url or f"http://{settings.api_host}:{settings.api_port}"
        
        # Metrics storage (circular buffers)
        self.max_history = 1000
        self.latency_history = deque(maxlen=self.max_history)
        self.profit_history = deque(maxlen=self.max_history)
        self.opportunity_history = deque(maxlen=self.max_history)
        self.success_rate_history = deque(maxlen=self.max_history)
        
        # Current metrics
        self.current_metrics = {
            'latency_ms': 0,
            'throughput': 0,
            'opportunities_detected': 0,
            'trades_executed': 0,
            'total_profit_usd': 0,
            'success_rate': 0,
            'sharpe_ratio': 0,
            'uptime_hours': 0
        }
        
        # Performance thresholds
        self.thresholds = {
            'latency_warning': 20,  # ms
            'latency_critical': 30,  # ms
            'success_rate_warning': 60,  # %
            'success_rate_critical': 40,  # %
            'profit_warning': -100,  # USD
            'profit_critical': -500  # USD
        }
        
        # Statistics
        self.start_time = time.time()
        self.last_update = time.time()
        
    async def fetch_metrics(self) -> Dict[str, Any]:
        """Fetch current metrics from API."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get performance metrics
                async with session.get(f"{self.api_url}/api/v1/performance") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        self.current_metrics.update(metrics)
                
                # Get health status
                async with session.get(f"{self.api_url}/api/v1/health") as response:
                    if response.status == 200:
                        health = await response.json()
                        self.current_metrics['uptime_hours'] = health.get('uptime_seconds', 0) / 3600
                
                # Get opportunities
                async with session.get(f"{self.api_url}/api/v1/opportunities") as response:
                    if response.status == 200:
                        opportunities = await response.json()
                        self.current_metrics['current_opportunities'] = len(opportunities)
                
                return self.current_metrics
                
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
            return self.current_metrics
    
    async def update_history(self) -> None:
        """Update metric history."""
        timestamp = time.time()
        
        self.latency_history.append({
            'timestamp': timestamp,
            'value': self.current_metrics.get('latency_ms', 0)
        })
        
        self.profit_history.append({
            'timestamp': timestamp,
            'value': self.current_metrics.get('total_profit_usd', 0)
        })
        
        self.opportunity_history.append({
            'timestamp': timestamp,
            'value': self.current_metrics.get('opportunities_detected', 0)
        })
        
        self.success_rate_history.append({
            'timestamp': timestamp,
            'value': self.current_metrics.get('success_rate', 0)
        })
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        stats = {}
        
        # Latency statistics
        if self.latency_history:
            latencies = [m['value'] for m in self.latency_history]
            stats['latency_avg'] = np.mean(latencies)
            stats['latency_p50'] = np.percentile(latencies, 50)
            stats['latency_p95'] = np.percentile(latencies, 95)
            stats['latency_p99'] = np.percentile(latencies, 99)
            stats['latency_max'] = np.max(latencies)
        
        # Profit statistics
        if self.profit_history:
            profits = [m['value'] for m in self.profit_history]
            if len(profits) > 1:
                returns = np.diff(profits)
                if len(returns) > 0 and np.std(returns) > 0:
                    stats['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                else:
                    stats['sharpe_ratio'] = 0
            stats['total_profit'] = profits[-1] if profits else 0
            stats['profit_per_hour'] = stats['total_profit'] / max(self.current_metrics['uptime_hours'], 1)
        
        # Success rate trend
        if self.success_rate_history:
            recent_rates = [m['value'] for m in list(self.success_rate_history)[-100:]]
            stats['success_rate_trend'] = np.mean(recent_rates)
        
        # Opportunity detection rate
        if self.opportunity_history and len(self.opportunity_history) > 1:
            opportunities = [m['value'] for m in self.opportunity_history]
            time_span = (self.opportunity_history[-1]['timestamp'] - 
                        self.opportunity_history[0]['timestamp'])
            if time_span > 0:
                stats['opportunities_per_minute'] = (opportunities[-1] - opportunities[0]) / (time_span / 60)
        
        return stats
    
    def get_status_color(self, metric: str, value: float) -> str:
        """Get color based on metric value and thresholds."""
        if metric == 'latency':
            if value >= self.thresholds['latency_critical']:
                return 'red'
            elif value >= self.thresholds['latency_warning']:
                return 'yellow'
            return 'green'
        
        elif metric == 'success_rate':
            if value <= self.thresholds['success_rate_critical']:
                return 'red'
            elif value <= self.thresholds['success_rate_warning']:
                return 'yellow'
            return 'green'
        
        elif metric == 'profit':
            if value <= self.thresholds['profit_critical']:
                return 'red'
            elif value <= self.thresholds['profit_warning']:
                return 'yellow'
            return 'green'
        
        return 'white'
    
    def create_dashboard(self) -> Layout:
        """Create rich dashboard layout."""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(Panel(
            f"[bold cyan]📊 CRYPTO ARBITRAGE PERFORMANCE MONITOR[/bold cyan]\n"
            f"[white]Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]",
            border_style="cyan"
        ))
        
        # Split body into columns
        layout["body"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="statistics", ratio=1),
            Layout(name="alerts", ratio=1)
        )
        
        # Metrics panel
        metrics_table = Table(title="Current Metrics", show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        
        latency_color = self.get_status_color('latency', self.current_metrics['latency_ms'])
        success_color = self.get_status_color('success_rate', self.current_metrics['success_rate'])
        profit_color = self.get_status_color('profit', self.current_metrics['total_profit_usd'])
        
        metrics_table.add_row("Latency", f"[{latency_color}]{self.current_metrics['latency_ms']:.1f} ms[/{latency_color}]")
        metrics_table.add_row("Throughput", f"{self.current_metrics['throughput']}/sec")
        metrics_table.add_row("Opportunities", str(self.current_metrics['opportunities_detected']))
        metrics_table.add_row("Trades Executed", str(self.current_metrics['trades_executed']))
        metrics_table.add_row("Success Rate", f"[{success_color}]{self.current_metrics['success_rate']:.1f}%[/{success_color}]")
        metrics_table.add_row("Total Profit", f"[{profit_color}]${self.current_metrics['total_profit_usd']:.2f}[/{profit_color}]")
        metrics_table.add_row("Sharpe Ratio", f"{self.current_metrics.get('sharpe_ratio', 0):.2f}")
        metrics_table.add_row("Uptime", f"{self.current_metrics['uptime_hours']:.1f} hours")
        
        layout["metrics"].update(Panel(metrics_table, border_style="green"))
        
        # Statistics panel
        stats = self.calculate_statistics()
        stats_table = Table(title="Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        if 'latency_avg' in stats:
            stats_table.add_row("Avg Latency", f"{stats['latency_avg']:.1f} ms")
            stats_table.add_row("P95 Latency", f"{stats['latency_p95']:.1f} ms")
            stats_table.add_row("P99 Latency", f"{stats['latency_p99']:.1f} ms")
        
        if 'sharpe_ratio' in stats:
            stats_table.add_row("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
        
        if 'profit_per_hour' in stats:
            stats_table.add_row("Profit/Hour", f"${stats['profit_per_hour']:.2f}")
        
        if 'opportunities_per_minute' in stats:
            stats_table.add_row("Opp/Minute", f"{stats['opportunities_per_minute']:.1f}")
        
        layout["statistics"].update(Panel(stats_table, border_style="blue"))
        
        # Alerts panel
        alerts = self.check_alerts()
        alerts_text = "[bold]System Alerts[/bold]\n\n"
        
        if not alerts:
            alerts_text += "[green]✅ All systems operating normally[/green]"
        else:
            for alert in alerts[:5]:  # Show max 5 alerts
                level = alert['level']
                message = alert['message']
                
                if level == 'critical':
                    alerts_text += f"[red]🔴 {message}[/red]\n"
                elif level == 'warning':
                    alerts_text += f"[yellow]🟡 {message}[/yellow]\n"
                else:
                    alerts_text += f"[blue]🔵 {message}[/blue]\n"
        
        layout["alerts"].update(Panel(alerts_text, border_style="yellow"))
        
        # Footer
        layout["footer"].update(Panel(
            f"[dim]Press Ctrl+C to exit | Refresh: 1s | Buffer: {len(self.latency_history)}/{self.max_history}[/dim]",
            border_style="dim"
        ))
        
        return layout
    
    def check_alerts(self) -> List[Dict[str, str]]:
        """Check for system alerts based on thresholds."""
        alerts = []
        
        # Latency alerts
        if self.current_metrics['latency_ms'] >= self.thresholds['latency_critical']:
            alerts.append({
                'level': 'critical',
                'message': f"Latency critical: {self.current_metrics['latency_ms']:.1f}ms (target: <{self.thresholds['latency_critical']}ms)"
            })
        elif self.current_metrics['latency_ms'] >= self.thresholds['latency_warning']:
            alerts.append({
                'level': 'warning',
                'message': f"Latency warning: {self.current_metrics['latency_ms']:.1f}ms"
            })
        
        # Success rate alerts
        if self.current_metrics['success_rate'] <= self.thresholds['success_rate_critical']:
            alerts.append({
                'level': 'critical',
                'message': f"Success rate critical: {self.current_metrics['success_rate']:.1f}%"
            })
        elif self.current_metrics['success_rate'] <= self.thresholds['success_rate_warning']:
            alerts.append({
                'level': 'warning',
                'message': f"Success rate low: {self.current_metrics['success_rate']:.1f}%"
            })
        
        # Profit alerts
        if self.current_metrics['total_profit_usd'] <= self.thresholds['profit_critical']:
            alerts.append({
                'level': 'critical',
                'message': f"Significant loss: ${self.current_metrics['total_profit_usd']:.2f}"
            })
        elif self.current_metrics['total_profit_usd'] <= self.thresholds['profit_warning']:
            alerts.append({
                'level': 'warning',
                'message': f"Negative profit: ${self.current_metrics['total_profit_usd']:.2f}"
            })
        
        # No recent trades alert
        if self.current_metrics['trades_executed'] == 0 and self.current_metrics['uptime_hours'] > 0.5:
            alerts.append({
                'level': 'warning',
                'message': "No trades executed in last 30 minutes"
            })
        
        return alerts
    
    def export_metrics(self, filepath: str = "metrics_export.json") -> None:
        """Export metrics history to file."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': self.current_metrics,
            'statistics': self.calculate_statistics(),
            'latency_history': list(self.latency_history),
            'profit_history': list(self.profit_history),
            'success_rate_history': list(self.success_rate_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"[green]Metrics exported to {filepath}[/green]")
    
    def create_performance_chart(self) -> None:
        """Create and save performance charts using Plotly."""
        if len(self.latency_history) < 2:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Over Time', 'Cumulative Profit', 
                          'Success Rate', 'Opportunities Detected'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Latency chart
        if self.latency_history:
            timestamps = [m['timestamp'] for m in self.latency_history]
            latencies = [m['value'] for m in self.latency_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=latencies, name='Latency (ms)', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=self.thresholds['latency_warning'], line_dash="dash", 
                         line_color="yellow", row=1, col=1)
            fig.add_hline(y=self.thresholds['latency_critical'], line_dash="dash", 
                         line_color="red", row=1, col=1)
        
        # Profit chart
        if self.profit_history:
            timestamps = [m['timestamp'] for m in self.profit_history]
            profits = [m['value'] for m in self.profit_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=profits, name='Profit ($)', 
                          line=dict(color='green'), fill='tozeroy'),
                row=1, col=2
            )
        
        # Success rate chart
        if self.success_rate_history:
            timestamps = [m['timestamp'] for m in self.success_rate_history]
            rates = [m['value'] for m in self.success_rate_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=rates, name='Success Rate (%)', 
                          line=dict(color='orange')),
                row=2, col=1
            )
        
        # Opportunities chart
        if self.opportunity_history:
            timestamps = [m['timestamp'] for m in self.opportunity_history]
            opportunities = [m['value'] for m in self.opportunity_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=opportunities, name='Opportunities', 
                          line=dict(color='purple')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Crypto Arbitrage Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save to file
        fig.write_html("performance_dashboard.html")
        console.print("[green]Performance chart saved to performance_dashboard.html[/green]")
    
    async def run(self, refresh_interval: float = 1.0) -> None:
        """Run the performance monitor."""
        console.print(Panel.fit(
            "[bold cyan]Starting Performance Monitor[/bold cyan]\n"
            "[dim]Press Ctrl+C to stop and export metrics[/dim]",
            border_style="cyan"
        ))
        
        with Live(self.create_dashboard(), refresh_per_second=1, console=console) as live:
            while True:
                try:
                    # Fetch latest metrics
                    await self.fetch_metrics()
                    
                    # Update history
                    await self.update_history()
                    
                    # Update dashboard
                    live.update(self.create_dashboard())
                    
                    # Save chart periodically (every 60 seconds)
                    if int(time.time()) % 60 == 0:
                        self.create_performance_chart()
                    
                    # Wait before next update
                    await asyncio.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    await asyncio.sleep(5)
        
        # Export metrics on exit
        console.print("\n[yellow]Exporting metrics...[/yellow]")
        self.export_metrics()
        self.create_performance_chart()
        console.print("[green]Monitor stopped[/green]")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitor")
    parser.add_argument("--api-url", type=str, help="API URL")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (seconds)")
    parser.add_argument("--export", type=str, help="Export metrics to file")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(api_url=args.api_url)
    
    if args.export:
        # Just export current metrics and exit
        await monitor.fetch_metrics()
        monitor.export_metrics(args.export)
    else:
        # Run continuous monitoring
        await monitor.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped by user[/yellow]")
        sys.exit(0)