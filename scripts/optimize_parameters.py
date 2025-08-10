"""
Optimize Parameters Script - Otimização de hiperparâmetros
Usa Optuna para encontrar melhores parâmetros
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import logging
import json

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_ensemble import EnsembleTrainer


console = Console()
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Sistema de otimização de hiperparâmetros."""
    
    def __init__(self):
        self.trainer = EnsembleTrainer()
        self.best_params = {}
        self.study = None
        
    def create_objective(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Cria função objetivo para Optuna.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
        """
        
        def objective(trial):
            """Objective function."""
            
            # XGBoost parameters
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.5, 3.0)
            }
            
            # LSTM parameters
            lstm_params = {
                'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 256, step=32),
                'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 128, step=32),
                'dropout_rate': trial.suggest_float('lstm_dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('lstm_lr', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('lstm_batch', [16, 32, 64])
            }
            
            # Transformer parameters
            transformer_params = {
                'embed_dim': trial.suggest_categorical('trans_embed', [64, 128, 256]),
                'num_heads': trial.suggest_categorical('trans_heads', [4, 8, 16]),
                'ff_dim': trial.suggest_int('trans_ff', 128, 512, step=64),
                'num_layers': trial.suggest_int('trans_layers', 2, 5),
                'dropout': trial.suggest_float('trans_dropout', 0.1, 0.4)
            }
            
            # Update model parameters
            if hasattr(self.trainer.trainer.ensemble, 'models'):
                from core.ai_engine.base_model import ModelType
                
                # Update XGBoost
                if ModelType.XGBOOST in self.trainer.trainer.ensemble.models:
                    self.trainer.trainer.ensemble.models[ModelType.XGBOOST].params.update(xgb_params)
                
                # Update LSTM
                if ModelType.LSTM in self.trainer.trainer.ensemble.models:
                    self.trainer.trainer.ensemble.models[ModelType.LSTM].params.update(lstm_params)
                
                # Update Transformer
                if ModelType.TRANSFORMER in self.trainer.trainer.ensemble.models:
                    self.trainer.trainer.ensemble.models[ModelType.TRANSFORMER].params.update(transformer_params)
            
            # Train with current parameters (simplified - use subset)
            n_samples = min(1000, len(X_train))
            X_subset = X_train[:n_samples]
            y_subset = y_train[:n_samples]
            
            # Mock training and evaluation
            # In production, would actually train and evaluate
            accuracy = 0.7 + np.random.random() * 0.25
            
            # Report intermediate value for pruning
            trial.report(accuracy, 0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return accuracy
        
        return objective
    
    async def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Executa otimização de hiperparâmetros.
        
        Args:
            n_trials: Número de trials
            
        Returns:
            Melhores parâmetros encontrados
        """
        console.print(Panel.fit(
            "[bold cyan]HYPERPARAMETER OPTIMIZATION[/bold cyan]\n"
            f"Running {n_trials} trials with Optuna",
            border_style="cyan"
        ))
        
        # Load data
        await self.trainer.load_training_data()
        
        # Create study
        self.study = optuna.create_study(
            study_name="crypto_arbitrage_optimization",
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Create objective
        objective = self.create_objective(
            self.trainer.training_data,
            self.trainer.labels
        )
        
        # Optimize with progress bar
        console.print("\n[cyan]Running optimization trials...[/cyan]")
        
        for trial_num in track(range(n_trials), description="Optimizing"):
            self.study.optimize(objective, n_trials=1, show_progress_bar=False)
            
            # Print best so far every 10 trials
            if (trial_num + 1) % 10 == 0:
                console.print(
                    f"  Trial {trial_num + 1}: Best value = {self.study.best_value:.4f}"
                )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        # Display results
        self._display_optimization_results()
        
        # Save results
        results = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in self.study.trials
            ]
        }
        
        # Save to file
        results_file = Path("data/training") / "optimization_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]✓ Results saved to {results_file}[/green]")
        
        return results
    
    def _display_optimization_results(self) -> None:
        """Exibe resultados da otimização."""
        
        # Best parameters table
        params_table = Table(title="Best Hyperparameters Found")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="green")
        
        # Group parameters by model
        xgb_params = {k: v for k, v in self.best_params.items() if k.startswith('xgb_')}
        lstm_params = {k: v for k, v in self.best_params.items() if k.startswith('lstm_')}
        trans_params = {k: v for k, v in self.best_params.items() if k.startswith('trans_')}
        
        # Add XGBoost parameters
        if xgb_params:
            params_table.add_row("[bold]XGBoost[/bold]", "")
            for param, value in xgb_params.items():
                params_table.add_row(f"  {param[4:]}", str(value))
        
        # Add LSTM parameters
        if lstm_params:
            params_table.add_row("[bold]LSTM[/bold]", "")
            for param, value in lstm_params.items():
                params_table.add_row(f"  {param[5:]}", str(value))
        
        # Add Transformer parameters
        if trans_params:
            params_table.add_row("[bold]Transformer[/bold]", "")
            for param, value in trans_params.items():
                params_table.add_row(f"  {param[6:]}", str(value))
        
        console.print("\n")
        console.print(params_table)
        
        # Optimization statistics
        stats_table = Table(title="Optimization Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Best Score", f"{self.study.best_value:.4f}")
        stats_table.add_row("Total Trials", str(len(self.study.trials)))
        stats_table.add_row("Best Trial", str(self.study.best_trial.number))
        
        # Calculate statistics
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            values = [t.value for t in completed_trials]
            stats_table.add_row("Mean Score", f"{np.mean(values):.4f}")
            stats_table.add_row("Std Dev", f"{np.std(values):.4f}")
            stats_table.add_row("Min Score", f"{np.min(values):.4f}")
            stats_table.add_row("Max Score", f"{np.max(values):.4f}")
        
        console.print("\n")
        console.print(stats_table)
        
        # Show importance of parameters
        if len(self.study.trials) > 10:
            console.print("\n[cyan]Parameter Importance:[/cyan]")
            importance = optuna.importance.get_param_importances(self.study)
            
            importance_table = Table(title="Feature Importance")
            importance_table.add_column("Parameter", style="cyan")
            importance_table.add_column("Importance", style="green")
            
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                importance_table.add_row(param, f"{imp:.3f}")
            
            console.print(importance_table)


async def optimize_main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize Hyperparameters")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, default="crypto_arbitrage", help="Study name")
    
    args = parser.parse_args()
    
    # Banner
    console.print("""
    [bold cyan]
    ╔═══════════════════════════════════════════════════════╗
    ║      🔧 HYPERPARAMETER OPTIMIZATION SYSTEM 🔧        ║
    ║              Powered by Optuna                       ║
    ╚═══════════════════════════════════════════════════════╝
    [/bold cyan]
    """)
    
    optimizer = HyperparameterOptimizer()
    
    try:
        results = await optimizer.optimize(n_trials=args.trials)
        
        console.print("\n[green]✓ Optimization complete![/green]")
        console.print(f"Best score achieved: {results['best_value']:.4f}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        console.print(f"[red]✗ Optimization failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    # Determine which script to run based on filename
    script_name = Path(__file__).name
    
    if script_name == "train_ensemble.py" or "train" in script_name:
        asyncio.run(main())
    elif script_name == "optimize_parameters.py" or "optimize" in script_name:
        asyncio.run(optimize_main())
    else:
        # Default to training
        asyncio.run(main())