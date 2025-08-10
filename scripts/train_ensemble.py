"""
Train Ensemble Script - Treina todos os modelos do ensemble
Executa treinamento completo com validação e avaliação
"""

import asyncio
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from core.ai_engine.ensemble_model import EnsembleModel
from core.ai_engine.model_trainer import ModelTrainer
from core.ai_engine.online_learning import OnlineLearner
from core.xai_engine.shap_explainer import SHAPExplainer
from core.xai_engine.confidence_calculator import ConfidenceCalculator
from core.feature_engineering.feature_extractor import FeatureExtractor
from core.data_collector.exchange_connector import OrderBookSnapshot


# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Sistema completo de treinamento do ensemble."""
    
    def __init__(self, data_path: str = "data/training"):
        """
        Inicializa trainer.
        
        Args:
            data_path: Caminho para dados de treino
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.trainer = ModelTrainer()
        self.feature_extractor = FeatureExtractor()
        self.shap_explainer = SHAPExplainer()
        self.confidence_calculator = ConfidenceCalculator()
        self.online_learner = OnlineLearner()
        
        self.training_data = None
        self.labels = None
        self.feature_names = None
        
    async def load_training_data(self, filepath: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega dados de treinamento.
        
        Args:
            filepath: Caminho para arquivo de dados
            
        Returns:
            Features e labels
        """
        console.print("[cyan]Loading training data...[/cyan]")
        
        if filepath and Path(filepath).exists():
            # Load from file
            data = pd.read_csv(filepath)
            features = data.drop(['label', 'timestamp'], axis=1, errors='ignore').values
            labels = data['label'].values if 'label' in data else np.random.randint(0, 2, len(data))
            self.feature_names = list(data.columns[:-1])
            
        else:
            # Generate synthetic data for demonstration
            console.print("[yellow]Generating synthetic training data...[/yellow]")
            
            n_samples = 10000
            n_features = 74  # Match feature extractor
            
            # Generate features
            features = np.random.randn(n_samples, n_features)
            
            # Add some patterns
            features[:, 0] *= 2  # Higher variance for first feature
            features[:, 1] = np.sin(np.linspace(0, 4*np.pi, n_samples))  # Sinusoidal
            features[:, 2] = features[:, 0] * 0.5 + np.random.randn(n_samples) * 0.1  # Correlated
            
            # Generate labels based on features (create pattern)
            decision_boundary = (
                features[:, 0] * 0.3 + 
                features[:, 1] * 0.2 + 
                features[:, 2] * 0.1 + 
                np.random.randn(n_samples) * 0.5
            )
            labels = (decision_boundary > 0).astype(int)
            
            # Generate feature names
            self.feature_names = [
                'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
                'volatility_1m', 'volatility_5m', 'volatility_15m',
                'momentum', 'acceleration', 'jerk',
                'vwap', 'volume_profile', 'volume_imbalance',
                'buy_sell_ratio', 'large_order_ratio',
                'bid_ask_spread', 'spread_bps', 'mid_price', 'micro_price',
                'book_imbalance', 'book_depth_ratio', 'order_flow_imbalance',
                'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
                'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
                'stochastic_k', 'stochastic_d', 'williams_r', 'cci', 'adx',
                'gas_price_gwei', 'network_congestion', 'mempool_size',
                'btc_dominance', 'correlation_btc', 'correlation_eth',
                'zscore', 'entropy', 'autocorrelation', 'hurst_exponent',
                'kurtosis', 'skewness', 'price_efficiency', 'information_ratio'
            ]
            
            # Pad feature names if needed
            while len(self.feature_names) < n_features:
                self.feature_names.append(f'feature_{len(self.feature_names)}')
            
            self.feature_names = self.feature_names[:n_features]
        
        self.training_data = features
        self.labels = labels
        
        console.print(f"[green]✓ Loaded {len(features)} samples with {features.shape[1]} features[/green]")
        console.print(f"[green]✓ Positive class ratio: {np.mean(labels):.2%}[/green]")
        
        return features, labels
    
    async def train_ensemble(self) -> Dict[str, Any]:
        """
        Treina ensemble completo.
        
        Returns:
            Métricas de treinamento
        """
        console.print(Panel.fit(
            "[bold cyan]ENSEMBLE TRAINING[/bold cyan]\n"
            "Training XGBoost, LSTM, and Transformer models",
            border_style="cyan"
        ))
        
        # Prepare data
        X_train, X_test, y_train, y_test = await self.trainer.prepare_training_data(
            self.training_data,
            self.labels,
            self.feature_names
        )
        
        # Train with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Train ensemble
            task = progress.add_task("[cyan]Training ensemble models...", total=100)
            
            start_time = time.time()
            results = await self.trainer.train_ensemble(
                X_train, y_train,
                X_test, y_test,
                self.feature_names
            )
            
            progress.update(task, completed=100)
            
            # Initialize SHAP explainer
            task = progress.add_task("[cyan]Initializing XAI system...", total=100)
            
            self.shap_explainer.initialize(
                self.trainer.ensemble.models,
                X_train[:100],  # Use subset for background
                self.feature_names
            )
            
            progress.update(task, completed=100)
        
        # Display results
        self._display_training_results(results)
        
        # Test inference speed
        await self._test_inference_speed(X_test[:10])
        
        # Save models
        model_dir = self.data_path / "models"
        model_dir.mkdir(exist_ok=True)
        self.trainer.ensemble.save_all_models(str(model_dir))
        
        console.print(f"\n[green]✓ Models saved to {model_dir}[/green]")
        
        return results
    
    def _display_training_results(self, results: Dict[str, Any]) -> None:
        """Exibe resultados do treinamento."""
        
        # Training metrics table
        train_table = Table(title="Training Metrics by Model")
        train_table.add_column("Model", style="cyan")
        train_table.add_column("Accuracy", style="green")
        train_table.add_column("Training Time", style="yellow")
        train_table.add_column("Status", style="white")
        
        for model_name, metrics in results['training_metrics'].items():
            if 'error' in metrics:
                train_table.add_row(
                    model_name.upper(),
                    "N/A",
                    "N/A",
                    f"[red]❌ {metrics['error'][:30]}[/red]"
                )
            else:
                train_table.add_row(
                    model_name.upper(),
                    f"{metrics.get('accuracy', 0):.3f}",
                    f"{metrics.get('training_time', 0):.1f}s",
                    "[green]✓ Trained[/green]"
                )
        
        console.print("\n")
        console.print(train_table)
        
        # Evaluation metrics table
        eval_metrics = results['evaluation_metrics']
        
        eval_table = Table(title="Ensemble Evaluation Metrics")
        eval_table.add_column("Metric", style="cyan")
        eval_table.add_column("Value", style="green")
        
        eval_table.add_row("Accuracy", f"{eval_metrics['accuracy']:.3f}")
        eval_table.add_row("Precision", f"{eval_metrics['precision']:.3f}")
        eval_table.add_row("Recall", f"{eval_metrics['recall']:.3f}")
        eval_table.add_row("F1 Score", f"{eval_metrics['f1']:.3f}")
        eval_table.add_row("AUC", f"{eval_metrics['auc']:.3f}")
        eval_table.add_row("Avg Confidence", f"{eval_metrics['avg_confidence']:.3f}")
        eval_table.add_row("High Confidence Ratio", f"{eval_metrics['high_confidence_ratio']:.2%}")
        
        console.print("\n")
        console.print(eval_table)
        
        # Confusion matrix
        if 'confusion_matrix' in eval_metrics:
            cm = eval_metrics['confusion_matrix']
            console.print("\n[cyan]Confusion Matrix:[/cyan]")
            console.print(f"  TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
            console.print(f"  FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")
    
    async def _test_inference_speed(self, test_samples: np.ndarray) -> None:
        """Testa velocidade de inferência."""
        console.print("\n[cyan]Testing inference speed...[/cyan]")
        
        latencies = []
        
        for i, sample in enumerate(test_samples):
            start = time.perf_counter()
            
            # Get ensemble prediction
            prediction = await self.trainer.ensemble.predict(sample)
            
            # Get SHAP explanation (parallel)
            shap_task = asyncio.create_task(
                self.shap_explainer.explain_prediction(
                    list(self.trainer.ensemble.models.keys())[0],
                    sample,
                    prediction.final_prediction
                )
            )
            
            # Calculate confidence
            confidence = self.confidence_calculator.calculate_confidence(
                prediction.model_predictions,
                sample
            )
            
            # Wait for SHAP
            shap_explanation = await shap_task
            
            total_time = (time.perf_counter() - start) * 1000
            latencies.append(total_time)
        
        # Display latency stats
        latency_table = Table(title="Inference Latency Analysis")
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Value (ms)", style="green")
        latency_table.add_column("Target (ms)", style="yellow")
        latency_table.add_column("Status", style="white")
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        latency_table.add_row(
            "Average", f"{avg_latency:.2f}", "2.0",
            "[green]✓[/green]" if avg_latency < 2 else "[red]✗[/red]"
        )
        latency_table.add_row(
            "P95", f"{p95_latency:.2f}", "3.0",
            "[green]✓[/green]" if p95_latency < 3 else "[red]✗[/red]"
        )
        latency_table.add_row(
            "P99", f"{p99_latency:.2f}", "5.0",
            "[green]✓[/green]" if p99_latency < 5 else "[red]✗[/red]"
        )
        latency_table.add_row(
            "Max", f"{max_latency:.2f}", "10.0",
            "[green]✓[/green]" if max_latency < 10 else "[yellow]![/yellow]"
        )
        
        console.print(latency_table)
    
    async def perform_walk_forward_validation(self) -> Dict[str, Any]:
        """Executa walk-forward validation."""
        console.print("\n[cyan]Performing walk-forward validation...[/cyan]")
        
        if self.training_data is None:
            await self.load_training_data()
        
        results = self.trainer.perform_walk_forward_validation(
            self.training_data,
            self.labels,
            n_splits=settings.ml_config.walk_forward_windows
        )
        
        # Display results
        wf_table = Table(title="Walk-Forward Validation Results")
        wf_table.add_column("Metric", style="cyan")
        wf_table.add_column("Mean", style="green")
        wf_table.add_column("Std Dev", style="yellow")
        
        wf_table.add_row(
            "Accuracy",
            f"{results['mean_accuracy']:.3f}",
            f"±{results['std_accuracy']:.3f}"
        )
        wf_table.add_row(
            "AUC",
            f"{results['mean_auc']:.3f}",
            f"±{results['std_auc']:.3f}"
        )
        wf_table.add_row(
            "F1 Score",
            f"{results['mean_f1']:.3f}",
            f"±{results['std_f1']:.3f}"
        )
        
        console.print(wf_table)
        
        return results
    
    def setup_online_learning(self) -> None:
        """Configura sistema de aprendizado online."""
        console.print("\n[cyan]Setting up online learning system...[/cyan]")
        
        self.online_learner.ensemble = self.trainer.ensemble
        
        # Simulate adding experiences
        for i in range(100):
            # Mock experience
            features = np.random.randn(74)
            label = np.random.randint(0, 2)
            confidence = np.random.random()
            reward = np.random.randn() * 100
            
            self.online_learner.add_experience(features, label, confidence, reward)
        
        # Perform incremental update
        metrics = self.online_learner.incremental_update()
        
        console.print(f"[green]✓ Online learning configured[/green]")
        console.print(f"  Buffer size: {len(self.online_learner.experience_buffer)}")
        console.print(f"  SGD Accuracy: {metrics.get('sgd_accuracy', 0):.3f}")
        console.print(f"  River Accuracy: {metrics.get('river_accuracy', 0):.3f}")


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Ensemble Models")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--walk-forward", action="store_true", help="Perform walk-forward validation")
    parser.add_argument("--online", action="store_true", help="Setup online learning")
    parser.add_argument("--schedule", type=str, choices=['once', 'daily', 'weekly'], default='once',
                       help="Training schedule")
    
    args = parser.parse_args()
    
    # Banner
    console.print("""
    [bold cyan]
    ╔═══════════════════════════════════════════════════════╗
    ║       🤖 ENSEMBLE MODEL TRAINING SYSTEM 🤖          ║
    ║         XGBoost + LSTM + Transformer                 ║
    ╚═══════════════════════════════════════════════════════╝
    [/bold cyan]
    """)
    
    trainer = EnsembleTrainer()
    
    try:
        # Load data
        await trainer.load_training_data(args.data)
        
        # Train ensemble
        results = await trainer.train_ensemble()
        
        # Optional steps
        if args.walk_forward:
            await trainer.perform_walk_forward_validation()
        
        if args.online:
            trainer.setup_online_learning()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = trainer.data_path / f"training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]✓ Training complete! Results saved to {results_file}[/green]")
        
        # Schedule if needed
        if args.schedule == 'daily':
            console.print("\n[yellow]Daily training scheduled (not implemented in this demo)[/yellow]")
        elif args.schedule == 'weekly':
            console.print("\n[yellow]Weekly training scheduled (not implemented in this demo)[/yellow]")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        console.print(f"[red]✗ Training failed: {e}[/red]")
        sys.exit(1)
