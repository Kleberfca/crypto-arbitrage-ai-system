"""
Ensemble Model - Sistema de voting com pesos configuráveis
Combina XGBoost, LSTM e Transformer para decisão final
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from .base_model import ModelType, PredictionResult
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Resultado do ensemble com detalhes de cada modelo."""
    
    final_prediction: float
    final_confidence: float
    model_predictions: Dict[ModelType, PredictionResult]
    voting_weights: Dict[ModelType, float]
    agreement_score: float
    total_inference_time_ms: float
    timestamp: float = time.time()
    
    @property
    def should_execute(self) -> bool:
        """Decide se deve executar baseado em threshold."""
        return (
            self.final_confidence >= settings.ml_config.confidence_threshold and
            self.agreement_score >= 0.6  # Pelo menos 60% de concordância
        )


class EnsembleModel:
    """
    Sistema de ensemble que combina múltiplos modelos.
    Implementa weighted voting com análise de concordância.
    """
    
    def __init__(self):
        """Inicializa ensemble com modelos configurados."""
        self.models: Dict[ModelType, Any] = {}
        self.weights = settings.ml_config.model_weights.copy()
        self.is_initialized = False
        
        # Performance tracking
        self.prediction_history: List[EnsemblePrediction] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self) -> None:
        """Inicializa todos os modelos do ensemble."""
        logger.info("Initializing ensemble models...")
        
        try:
            # Initialize models
            self.models[ModelType.XGBOOST] = XGBoostModel()
            self.models[ModelType.LSTM] = LSTMModel()
            self.models[ModelType.TRANSFORMER] = TransformerModel()
            
            # TODO: Add statistical model when implemented
            # self.models[ModelType.STATISTICAL] = StatisticalModel()
            
            # Adjust weights if statistical not available
            if ModelType.STATISTICAL not in self.models:
                statistical_weight = self.weights.pop('statistical', 0)
                # Redistribute weight proportionally
                total_remaining = sum(self.weights.values())
                for model_type in self.weights:
                    self.weights[model_type] += (
                        statistical_weight * self.weights[model_type] / total_remaining
                    )
            
            self.is_initialized = True
            logger.info(f"Ensemble initialized with models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {e}")
            raise
    
    async def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Treina todos os modelos do ensemble.
        
        Args:
            X: Features de treino
            y: Labels
            feature_names: Nomes das features
            
        Returns:
            Métricas de treino de cada modelo
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info("Training ensemble models...")
        training_metrics = {}
        
        # Train models in parallel
        tasks = []
        for model_type, model in self.models.items():
            task = asyncio.create_task(
                self._train_model_async(model, X, y, feature_names)
            )
            tasks.append((model_type, task))
        
        # Wait for all training to complete
        for model_type, task in tasks:
            try:
                metrics = await task
                training_metrics[model_type.value] = metrics
                logger.info(f"{model_type.value} training complete: {metrics}")
            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")
                training_metrics[model_type.value] = {"error": str(e)}
        
        return training_metrics
    
    async def _train_model_async(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Treina modelo de forma assíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            model.train,
            X, y, feature_names, 0.2
        )
    
    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Faz predição usando ensemble com voting ponderado.
        
        Args:
            X: Features para predição
            
        Returns:
            Predição do ensemble com detalhes
        """
        if not self.is_initialized:
            raise ValueError("Ensemble not initialized")
        
        start_time = time.perf_counter()
        model_predictions = {}
        
        # Get predictions from all models in parallel
        tasks = []
        for model_type, model in self.models.items():
            if model.is_trained:
                task = asyncio.create_task(
                    self._predict_model_async(model, X)
                )
                tasks.append((model_type, task))
        
        # Collect predictions
        predictions = []
        confidences = []
        weights_used = []
        
        for model_type, task in tasks:
            try:
                result = await task
                model_predictions[model_type] = result
                
                # Get weight for this model
                weight = self.weights.get(model_type.value, 0)
                
                predictions.append(result.prediction)
                confidences.append(result.confidence)
                weights_used.append(weight)
                
            except Exception as e:
                logger.error(f"Prediction failed for {model_type.value}: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Calculate weighted average
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        weights_used = np.array(weights_used)
        
        # Normalize weights
        weights_used = weights_used / weights_used.sum()
        
        # Weighted prediction
        final_prediction = np.average(predictions, weights=weights_used)
        
        # Weighted confidence
        final_confidence = np.average(confidences, weights=weights_used)
        
        # Calculate agreement score (how much models agree)
        agreement_score = self._calculate_agreement(predictions)
        
        # Adjust confidence based on agreement
        final_confidence *= (0.5 + 0.5 * agreement_score)
        
        # Total inference time
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Check latency requirement
        if total_time > 2.0:  # 2ms target
            logger.warning(f"High ensemble inference latency: {total_time:.2f}ms")
        
        # Create result
        result = EnsemblePrediction(
            final_prediction=float(final_prediction),
            final_confidence=float(final_confidence),
            model_predictions=model_predictions,
            voting_weights={
                model_type: float(w) 
                for model_type, w in zip([mt for mt, _ in tasks], weights_used)
            },
            agreement_score=float(agreement_score),
            total_inference_time_ms=total_time
        )
        
        # Store in history
        self.prediction_history.append(result)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        return result
    
    async def _predict_model_async(self, model: Any, X: np.ndarray) -> PredictionResult:
        """Faz predição de modelo de forma assíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            model.predict,
            X
        )
    
    def _calculate_agreement(self, predictions: np.ndarray) -> float:
        """
        Calcula score de concordância entre modelos.
        
        Args:
            predictions: Array de predições
            
        Returns:
            Agreement score (0-1)
        """
        if len(predictions) < 2:
            return 1.0
        
        # Calculate standard deviation
        std = np.std(predictions)
        
        # Lower std = higher agreement
        # Map std to agreement score
        # std of 0.5 (max disagreement) -> 0
        # std of 0 (perfect agreement) -> 1
        agreement = max(0, 1 - (std * 2))
        
        return agreement
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Agrega feature importance de todos os modelos.
        
        Returns:
            Feature importance agregado
        """
        aggregated_importance = {}
        
        for model_type, model in self.models.items():
            if model.is_trained and hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                
                # Weight by model weight
                weight = self.weights.get(model_type.value, 0)
                
                for feature, imp in importance.items():
                    if feature not in aggregated_importance:
                        aggregated_importance[feature] = 0
                    aggregated_importance[feature] += imp * weight
        
        # Sort by importance
        return dict(sorted(
            aggregated_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save_all_models(self, directory: str) -> None:
        """Salva todos os modelos treinados."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_type, model in self.models.items():
            if model.is_trained:
                filepath = os.path.join(directory, f"{model_type.value}_model.pkl")
                model.save(filepath)
                logger.info(f"Saved {model_type.value} to {filepath}")
    
    def load_all_models(self, directory: str) -> None:
        """Carrega todos os modelos salvos."""
        import os
        
        for model_type, model in self.models.items():
            filepath = os.path.join(directory, f"{model_type.value}_model.pkl")
            if os.path.exists(filepath):
                model.load(filepath)
                logger.info(f"Loaded {model_type.value} from {filepath}")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das predições."""
        if not self.prediction_history:
            return {}
        
        predictions = [p.final_prediction for p in self.prediction_history]
        confidences = [p.final_confidence for p in self.prediction_history]
        agreements = [p.agreement_score for p in self.prediction_history]
        latencies = [p.total_inference_time_ms for p in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'avg_prediction': np.mean(predictions),
            'avg_confidence': np.mean(confidences),
            'avg_agreement': np.mean(agreements),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'high_confidence_ratio': sum(1 for c in confidences if c > 0.75) / len(confidences)
        }
