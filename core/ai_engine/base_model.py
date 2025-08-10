"""
Base Model Interface - Abstract class for all ML models
Implements common functionality and enforces interface
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

from config.settings import settings


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos no ensemble."""
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    STATISTICAL = "statistical"


@dataclass
class PredictionResult:
    """Resultado de predição com metadados."""
    
    prediction: float  # 0-1 probability
    confidence: float  # Model confidence
    features_used: List[str]
    inference_time_ms: float
    model_type: ModelType
    timestamp: float = field(default_factory=time.time)
    
    @property
    def should_execute(self) -> bool:
        """Decide se deve executar trade baseado em confidence threshold."""
        return self.confidence >= settings.ml_config.confidence_threshold


class BaseModel(ABC):
    """
    Interface abstrata para todos os modelos de ML.
    Garante consistência e reusabilidade.
    """
    
    def __init__(self, model_type: ModelType, version: str = "1.0.0"):
        """
        Inicializa modelo base.
        
        Args:
            model_type: Tipo do modelo
            version: Versão do modelo
        """
        self.model_type = model_type
        self.version = version
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Performance tracking
        self.training_metrics: Dict[str, float] = {}
        self.inference_times: List[float] = []
        self.predictions_count = 0
        
        # Model parameters
        self.params = self._get_default_params()
        
    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """Retorna parâmetros padrão do modelo."""
        pass
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Cria instância do modelo específico."""
        pass
    
    @abstractmethod
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepara features para o modelo específico.
        
        Args:
            X: Features raw
            
        Returns:
            Features preparadas
        """
        pass
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Treina o modelo.
        
        Args:
            X: Features de treino
            y: Labels (0 ou 1)
            feature_names: Nomes das features
            validation_split: Proporção para validação
            
        Returns:
            Métricas de treinamento
        """
        start_time = time.perf_counter()
        
        try:
            # Store feature names
            self.feature_names = feature_names
            
            # Prepare features
            X_prepared = self._prepare_features(X)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X_prepared)
            
            # Create model
            self.model = self._create_model()
            
            # Train with cross-validation
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=5, scoring='roc_auc'
            )
            
            # Fit on full dataset
            self.model.fit(X_scaled, y)
            
            # Calculate metrics
            self.training_metrics = {
                'accuracy': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'training_time': (time.perf_counter() - start_time),
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            self.is_trained = True
            
            logger.info(
                f"{self.model_type.value} trained",
                accuracy=f"{self.training_metrics['accuracy']:.3f}",
                time=f"{self.training_metrics['training_time']:.2f}s"
            )
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Training failed for {self.model_type.value}: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Faz predição com tracking de performance.
        
        Args:
            X: Features para predição
            
        Returns:
            Resultado da predição
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_type.value} not trained")
        
        start_time = time.perf_counter()
        
        try:
            # Prepare and scale features
            X_prepared = self._prepare_features(X)
            X_scaled = self.scaler.transform(X_prepared.reshape(1, -1))
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X_scaled)[0, 1]
            else:
                prob = self.model.predict(X_scaled)[0]
            
            # Calculate confidence (model-specific)
            confidence = self._calculate_confidence(X_scaled, prob)
            
            # Track performance
            inference_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(inference_time)
            self.predictions_count += 1
            
            # Check latency requirement
            if inference_time > 2.0:  # 2ms limit per model
                logger.warning(
                    f"High inference latency for {self.model_type.value}: {inference_time:.2f}ms"
                )
            
            return PredictionResult(
                prediction=float(prob),
                confidence=confidence,
                features_used=self.feature_names,
                inference_time_ms=inference_time,
                model_type=self.model_type
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type.value}: {e}")
            raise
    
    @abstractmethod
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """
        Calcula confiança na predição (model-specific).
        
        Args:
            X: Features scaled
            prediction: Predição do modelo
            
        Returns:
            Score de confiança (0-1)
        """
        pass
    
    def save(self, filepath: str) -> None:
        """Salva modelo treinado."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type.value,
            'version': self.version,
            'training_metrics': self.training_metrics,
            'params': self.params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Carrega modelo treinado."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.version = model_data['version']
        self.training_metrics = model_data['training_metrics']
        self.params = model_data['params']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    @property
    def average_inference_time(self) -> float:
        """Retorna tempo médio de inferência em ms."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância das features (se disponível).
        
        Returns:
            Dicionário feature -> importance
        """
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        
        return {}
