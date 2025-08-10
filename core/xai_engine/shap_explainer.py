"""
SHAP Explainer - SHAP values para explicabilidade
Garante que todas as decisões sejam explicáveis
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shap
from shap import TreeExplainer, DeepExplainer, KernelExplainer

from core.ai_engine.base_model import ModelType


logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """Resultado da explicação SHAP."""
    
    feature_importance: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    base_value: float
    prediction_value: float
    explanation_time_ms: float
    model_type: ModelType
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'feature_importance': self.feature_importance,
            'top_positive': self.top_positive_features,
            'top_negative': self.top_negative_features,
            'base_value': self.base_value,
            'prediction_value': self.prediction_value,
            'explanation_time_ms': self.explanation_time_ms,
            'model_type': self.model_type.value
        }


class SHAPExplainer:
    """
    Sistema de explicabilidade usando SHAP values.
    Explica decisões de todos os modelos do ensemble.
    """
    
    def __init__(self, background_samples: int = 100):
        """
        Inicializa SHAP explainer.
        
        Args:
            background_samples: Número de amostras para background
        """
        self.background_samples = background_samples
        self.explainers: Dict[ModelType, Any] = {}
        self.background_data: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def initialize(
        self,
        models: Dict[ModelType, Any],
        background_data: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """
        Inicializa explainers para cada modelo.
        
        Args:
            models: Dicionário de modelos treinados
            background_data: Dados de background para SHAP
            feature_names: Nomes das features
        """
        self.feature_names = feature_names
        self.background_data = background_data[:self.background_samples]
        
        for model_type, model_instance in models.items():
            if not model_instance.is_trained:
                continue
            
            try:
                if model_type == ModelType.XGBOOST:
                    # TreeExplainer for XGBoost
                    self.explainers[model_type] = TreeExplainer(model_instance.model)
                    
                elif model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
                    # DeepExplainer for neural networks
                    # Note: Requires TensorFlow/PyTorch model
                    # For now, use KernelExplainer as fallback
                    self.explainers[model_type] = KernelExplainer(
                        model_instance.predict,
                        self.background_data
                    )
                    
                else:
                    # KernelExplainer for other models
                    self.explainers[model_type] = KernelExplainer(
                        model_instance.predict,
                        self.background_data
                    )
                
                logger.info(f"SHAP explainer initialized for {model_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize SHAP for {model_type.value}: {e}")
    
    async def explain_prediction(
        self,
        model_type: ModelType,
        features: np.ndarray,
        prediction: float
    ) -> SHAPExplanation:
        """
        Explica predição usando SHAP values.
        
        Args:
            model_type: Tipo do modelo
            features: Features usadas na predição
            prediction: Valor da predição
            
        Returns:
            Explicação SHAP
        """
        if model_type not in self.explainers:
            raise ValueError(f"No explainer for {model_type.value}")
        
        start_time = time.perf_counter()
        
        # Run SHAP explanation in executor (non-blocking)
        loop = asyncio.get_event_loop()
        shap_values = await loop.run_in_executor(
            self.executor,
            self._calculate_shap_values,
            model_type,
            features
        )
        
        # Process SHAP values
        explanation = self._process_shap_values(
            shap_values,
            features,
            prediction,
            model_type
        )
        
        explanation.explanation_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Check latency requirement
        if explanation.explanation_time_ms > 5.0:  # 5ms target
            logger.warning(
                f"High SHAP explanation latency: {explanation.explanation_time_ms:.2f}ms"
            )
        
        return explanation
    
    def _calculate_shap_values(
        self,
        model_type: ModelType,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Calcula SHAP values para features.
        
        Args:
            model_type: Tipo do modelo
            features: Features
            
        Returns:
            SHAP values
        """
        explainer = self.explainers[model_type]
        
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Calculate SHAP values
        if isinstance(explainer, TreeExplainer):
            shap_values = explainer.shap_values(features)
            # For binary classification, might return list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
        else:
            shap_values = explainer.shap_values(features)
        
        # Ensure 1D array
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        return shap_values
    
    def _process_shap_values(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        prediction: float,
        model_type: ModelType
    ) -> SHAPExplanation:
        """
        Processa SHAP values em explicação estruturada.
        
        Args:
            shap_values: SHAP values calculados
            features: Features originais
            prediction: Predição do modelo
            model_type: Tipo do modelo
            
        Returns:
            Explicação processada
        """
        # Create feature importance dictionary
        feature_importance = {}
        for i, (name, value) in enumerate(zip(self.feature_names, shap_values)):
            feature_importance[name] = float(value)
        
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top positive and negative features
        positive_features = [(k, v) for k, v in sorted_features if v > 0][:5]
        negative_features = [(k, v) for k, v in sorted_features if v < 0][:5]
        
        # Get base value (expected value)
        explainer = self.explainers[model_type]
        if hasattr(explainer, 'expected_value'):
            base_value = float(explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            base_value = 0.5  # Default for binary classification
        
        return SHAPExplanation(
            feature_importance=feature_importance,
            top_positive_features=positive_features,
            top_negative_features=negative_features,
            base_value=base_value,
            prediction_value=prediction,
            explanation_time_ms=0,  # Will be set by caller
            model_type=model_type
        )
    
    async def explain_ensemble(
        self,
        model_predictions: Dict[ModelType, float],
        features: np.ndarray
    ) -> Dict[ModelType, SHAPExplanation]:
        """
        Explica predições de todo o ensemble.
        
        Args:
            model_predictions: Predições de cada modelo
            features: Features usadas
            
        Returns:
            Explicações para cada modelo
        """
        explanations = {}
        
        # Explain each model in parallel
        tasks = []
        for model_type, prediction in model_predictions.items():
            if model_type in self.explainers:
                task = self.explain_prediction(model_type, features, prediction)
                tasks.append((model_type, task))
        
        # Wait for all explanations
        for model_type, task in tasks:
            try:
                explanation = await task
                explanations[model_type] = explanation
            except Exception as e:
                logger.error(f"Failed to explain {model_type.value}: {e}")
        
        return explanations
    
    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Calcula importância global das features.
        
        Returns:
            Importância média global
        """
        if not self.background_data:
            return {}
        
        global_importance = {}
        
        for model_type, explainer in self.explainers.items():
            try:
                # Calculate SHAP values for background data
                shap_values = explainer.shap_values(self.background_data)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Average absolute SHAP values
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                
                # Add to global importance
                for i, feature in enumerate(self.feature_names):
                    if feature not in global_importance:
                        global_importance[feature] = 0
                    global_importance[feature] += mean_abs_shap[i]
                    
            except Exception as e:
                logger.error(f"Failed to calculate global importance for {model_type.value}: {e}")
        
        # Normalize
        total = sum(global_importance.values())
        if total > 0:
            for feature in global_importance:
                global_importance[feature] /= total
        
        # Sort by importance
        return dict(sorted(
            global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
