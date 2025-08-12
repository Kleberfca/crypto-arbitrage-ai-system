"""
LIME Explainer - Local Interpretable Model-agnostic Explanations
Explicações locais para decisões de modelos de ML
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LimeExplanation:
    """Explicação LIME para uma predição."""
    
    instance_id: str
    prediction: float
    prediction_local: float
    intercept: float
    
    # Feature explanations
    feature_weights: Dict[str, float]
    feature_values: Dict[str, float]
    feature_contributions: Dict[str, float]
    
    # Quality metrics
    local_score: float  # R² do modelo local
    coverage: float  # Cobertura da explicação
    
    # Metadata
    model_type: str
    timestamp: float = field(default_factory=time.time)
    num_samples: int = 100
    
    @property
    def top_features(self) -> List[Tuple[str, float]]:
        """Retorna top features por importância."""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:10]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'instance_id': self.instance_id,
            'prediction': self.prediction,
            'prediction_local': self.prediction_local,
            'intercept': self.intercept,
            'top_features': self.top_features,
            'local_score': self.local_score,
            'coverage': self.coverage,
            'model_type': self.model_type
        }


class LimeExplainer:
    """
    LIME explainer para interpretabilidade local de modelos.
    Gera explicações interpretáveis para predições individuais.
    """
    
    def __init__(self):
        """Inicializa LIME explainer."""
        # Configuration
        self.num_samples = 5000  # Número de amostras sintéticas
        self.num_features = 10  # Número de features para explicação
        self.kernel_width = 0.75  # Largura do kernel para ponderação
        self.regularization = 'auto'  # Regularização para Ridge
        
        # Feature statistics
        self.feature_stats = {}
        self.feature_ranges = {}
        
        # Performance tracking
        self.explanation_times = []
        self.explanations_generated = 0
        
        # Cache
        self.explanation_cache = {}
        self.cache_size = 100
        
    def explain_instance(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        predict_fn: Optional[callable] = None,
        num_samples: Optional[int] = None,
        num_features: Optional[int] = None
    ) -> LimeExplanation:
        """
        Explica uma predição individual.
        
        Args:
            model: Modelo a explicar
            instance: Instância a explicar
            feature_names: Nomes das features
            predict_fn: Função de predição customizada
            num_samples: Número de amostras sintéticas
            num_features: Número de features na explicação
            
        Returns:
            Explicação LIME
        """
        start_time = time.perf_counter()
        
        # Use default parameters if not provided
        num_samples = num_samples or self.num_samples
        num_features = num_features or self.num_features
        
        # Get prediction function
        if predict_fn is None:
            predict_fn = self._get_predict_fn(model)
        
        # Generate synthetic neighborhood
        synthetic_data = self._generate_synthetic_data(
            instance, num_samples, feature_names
        )
        
        # Get predictions for synthetic data
        predictions = predict_fn(synthetic_data)
        
        # Calculate weights based on distance
        weights = self._calculate_weights(
            instance, synthetic_data
        )
        
        # Fit local linear model
        explanation = self._fit_local_model(
            instance,
            synthetic_data,
            predictions,
            weights,
            feature_names,
            num_features
        )
        
        # Add model type
        explanation.model_type = type(model).__name__
        
        # Calculate quality metrics
        explanation.local_score = self._calculate_local_fidelity(
            explanation, instance, predict_fn
        )
        
        explanation.coverage = self._calculate_coverage(
            explanation, synthetic_data, predictions, predict_fn
        )
        
        # Update statistics
        self.explanations_generated += 1
        self.explanation_times.append(
            (time.perf_counter() - start_time) * 1000
        )
        
        # Cache explanation
        self._cache_explanation(instance, explanation)
        
        logger.debug(
            f"LIME explanation generated",
            features=len(feature_names),
            samples=num_samples,
            score=f"{explanation.local_score:.3f}"
        )
        
        return explanation
    
    def _get_predict_fn(self, model: Any) -> callable:
        """
        Obtém função de predição do modelo.
        
        Args:
            model: Modelo
            
        Returns:
            Função de predição
        """
        # Check for common model types
        if hasattr(model, 'predict_proba'):
            # Classificação - usar probabilidade da classe positiva
            def predict_fn(X):
                proba = model.predict_proba(X)
                if len(proba.shape) > 1 and proba.shape[1] > 1:
                    return proba[:, 1]  # Classe positiva
                return proba.flatten()
        elif hasattr(model, 'predict'):
            # Regressão ou classificação simples
            def predict_fn(X):
                return model.predict(X).flatten()
        else:
            raise ValueError("Model must have predict or predict_proba method")
        
        return predict_fn
    
    def _generate_synthetic_data(
        self,
        instance: np.ndarray,
        num_samples: int,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Gera dados sintéticos ao redor da instância.
        
        Args:
            instance: Instância original
            num_samples: Número de amostras
            feature_names: Nomes das features
            
        Returns:
            Dados sintéticos
        """
        n_features = len(instance)
        
        # Initialize synthetic data with the original instance
        synthetic = np.zeros((num_samples, n_features))
        synthetic[0] = instance  # First sample is the original
        
        # Generate variations
        for i in range(1, num_samples):
            # Randomly perturb features
            mask = np.random.random(n_features) < 0.5  # 50% chance to change
            
            synthetic[i] = instance.copy()
            
            for j in range(n_features):
                if mask[j]:
                    # Get feature statistics if available
                    feature_name = feature_names[j] if j < len(feature_names) else f"f{j}"
                    
                    if feature_name in self.feature_stats:
                        # Use known statistics
                        mean = self.feature_stats[feature_name].get('mean', 0)
                        std = self.feature_stats[feature_name].get('std', 1)
                        
                        # Sample from normal distribution
                        synthetic[i, j] = np.random.normal(mean, std)
                    else:
                        # Default: perturb by ±20%
                        if instance[j] != 0:
                            synthetic[i, j] = instance[j] * np.random.uniform(0.8, 1.2)
                        else:
                            synthetic[i, j] = np.random.uniform(-0.1, 0.1)
        
        return synthetic
    
    def _calculate_weights(
        self,
        instance: np.ndarray,
        synthetic_data: np.ndarray
    ) -> np.ndarray:
        """
        Calcula pesos baseados na distância.
        
        Args:
            instance: Instância original
            synthetic_data: Dados sintéticos
            
        Returns:
            Pesos
        """
        # Calculate euclidean distances
        distances = np.sqrt(
            np.sum((synthetic_data - instance) ** 2, axis=1)
        )
        
        # Convert to weights using exponential kernel
        kernel_width = self.kernel_width * np.sqrt(len(instance))
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        # Ensure original instance has maximum weight
        weights[0] = 1.0
        
        return weights
    
    def _fit_local_model(
        self,
        instance: np.ndarray,
        synthetic_data: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
        feature_names: List[str],
        num_features: int
    ) -> LimeExplanation:
        """
        Ajusta modelo linear local.
        
        Args:
            instance: Instância original
            synthetic_data: Dados sintéticos
            predictions: Predições
            weights: Pesos
            feature_names: Nomes das features
            num_features: Número de features para usar
            
        Returns:
            Explicação
        """
        # Fit weighted Ridge regression
        if self.regularization == 'auto':
            alpha = 0.01  # Default regularization
        else:
            alpha = self.regularization
        
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(synthetic_data, predictions, sample_weight=weights)
        
        # Get coefficients
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Calculate prediction for original instance
        prediction_original = predictions[0]
        prediction_local = model.predict(instance.reshape(1, -1))[0]
        
        # Create feature dictionary
        feature_weights = {}
        feature_values = {}
        feature_contributions = {}
        
        for i, name in enumerate(feature_names):
            if i < len(coefficients):
                feature_weights[name] = coefficients[i]
                feature_values[name] = instance[i]
                feature_contributions[name] = coefficients[i] * instance[i]
        
        # Select top features by absolute contribution
        if num_features < len(feature_names):
            top_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:num_features]
            
            # Keep only top features
            feature_weights = {k: feature_weights[k] for k, _ in top_features}
            feature_contributions = dict(top_features)
        
        # Create explanation
        explanation = LimeExplanation(
            instance_id=f"lime_{int(time.time())}",
            prediction=prediction_original,
            prediction_local=prediction_local,
            intercept=intercept,
            feature_weights=feature_weights,
            feature_values=feature_values,
            feature_contributions=feature_contributions,
            local_score=model.score(synthetic_data, predictions, sample_weight=weights),
            coverage=1.0,  # Will be updated
            model_type="",
            num_samples=len(synthetic_data)
        )
        
        return explanation
    
    def _calculate_local_fidelity(
        self,
        explanation: LimeExplanation,
        instance: np.ndarray,
        predict_fn: callable
    ) -> float:
        """
        Calcula fidelidade local da explicação.
        
        Args:
            explanation: Explicação
            instance: Instância
            predict_fn: Função de predição
            
        Returns:
            Score de fidelidade (0-1)
        """
        # Generate small perturbations
        n_test = 100
        perturbations = np.random.normal(0, 0.01, (n_test, len(instance)))
        test_instances = instance + perturbations
        
        # Get true predictions
        true_predictions = predict_fn(test_instances)
        
        # Get local model predictions
        local_predictions = np.zeros(n_test)
        
        for i in range(n_test):
            pred = explanation.intercept
            for j, feature_name in enumerate(explanation.feature_weights.keys()):
                if j < len(test_instances[i]):
                    pred += explanation.feature_weights[feature_name] * test_instances[i][j]
            local_predictions[i] = pred
        
        # Calculate R²
        if np.std(true_predictions) > 0:
            correlation = np.corrcoef(true_predictions, local_predictions)[0, 1]
            return max(0, correlation ** 2)
        
        return 0.5
    
    def _calculate_coverage(
        self,
        explanation: LimeExplanation,
        synthetic_data: np.ndarray,
        predictions: np.ndarray,
        predict_fn: callable
    ) -> float:
        """
        Calcula cobertura da explicação.
        
        Args:
            explanation: Explicação
            synthetic_data: Dados sintéticos
            predictions: Predições
            predict_fn: Função de predição
            
        Returns:
            Score de cobertura (0-1)
        """
        # Calculate how well the local model covers the synthetic neighborhood
        local_predictions = np.zeros(len(synthetic_data))
        
        for i in range(len(synthetic_data)):
            pred = explanation.intercept
            for j, feature_name in enumerate(explanation.feature_weights.keys()):
                if j < len(synthetic_data[i]):
                    pred += explanation.feature_weights[feature_name] * synthetic_data[i][j]
            local_predictions[i] = pred
        
        # Calculate coverage as 1 - relative error
        if np.std(predictions) > 0:
            relative_error = np.mean(
                np.abs(predictions - local_predictions) / (np.abs(predictions) + 1e-10)
            )
            coverage = max(0, 1 - relative_error)
        else:
            coverage = 1.0
        
        return coverage
    
    def explain_batch(
        self,
        model: Any,
        instances: np.ndarray,
        feature_names: List[str],
        predict_fn: Optional[callable] = None
    ) -> List[LimeExplanation]:
        """
        Explica múltiplas instâncias.
        
        Args:
            model: Modelo
            instances: Instâncias
            feature_names: Nomes das features
            predict_fn: Função de predição
            
        Returns:
            Lista de explicações
        """
        explanations = []
        
        for instance in instances:
            # Check cache first
            cached = self._get_cached_explanation(instance)
            
            if cached:
                explanations.append(cached)
            else:
                explanation = self.explain_instance(
                    model, instance, feature_names, predict_fn
                )
                explanations.append(explanation)
        
        return explanations
    
    def set_feature_statistics(
        self,
        feature_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Define estatísticas das features para melhor sampling.
        
        Args:
            feature_stats: Estatísticas por feature
        """
        self.feature_stats = feature_stats
        
        # Calculate ranges
        for feature, stats in feature_stats.items():
            if 'min' in stats and 'max' in stats:
                self.feature_ranges[feature] = (
                    stats['min'],
                    stats['max']
                )
    
    def _cache_explanation(
        self,
        instance: np.ndarray,
        explanation: LimeExplanation
    ) -> None:
        """
        Armazena explicação em cache.
        
        Args:
            instance: Instância
            explanation: Explicação
        """
        # Create hash key
        key = hash(instance.tobytes())
        
        # Check cache size
        if len(self.explanation_cache) >= self.cache_size:
            # Remove oldest
            oldest_key = min(
                self.explanation_cache.keys(),
                key=lambda k: self.explanation_cache[k].timestamp
            )
            del self.explanation_cache[oldest_key]
        
        # Add to cache
        self.explanation_cache[key] = explanation
    
    def _get_cached_explanation(
        self,
        instance: np.ndarray
    ) -> Optional[LimeExplanation]:
        """
        Busca explicação em cache.
        
        Args:
            instance: Instância
            
        Returns:
            Explicação ou None
        """
        key = hash(instance.tobytes())
        return self.explanation_cache.get(key)
    
    def compare_explanations(
        self,
        exp1: LimeExplanation,
        exp2: LimeExplanation
    ) -> Dict[str, Any]:
        """
        Compara duas explicações.
        
        Args:
            exp1: Primeira explicação
            exp2: Segunda explicação
            
        Returns:
            Comparação
        """
        # Get common features
        common_features = set(exp1.feature_weights.keys()) & set(exp2.feature_weights.keys())
        
        # Calculate similarity
        if common_features:
            weights1 = np.array([exp1.feature_weights[f] for f in common_features])
            weights2 = np.array([exp2.feature_weights[f] for f in common_features])
            
            # Cosine similarity
            similarity = cosine_similarity(
                weights1.reshape(1, -1),
                weights2.reshape(1, -1)
            )[0, 0]
        else:
            similarity = 0
        
        # Compare top features
        top1 = set([f for f, _ in exp1.top_features[:5]])
        top2 = set([f for f, _ in exp2.top_features[:5]])
        
        feature_overlap = len(top1 & top2) / max(len(top1), len(top2))
        
        return {
            'similarity': similarity,
            'feature_overlap': feature_overlap,
            'prediction_diff': abs(exp1.prediction - exp2.prediction),
            'common_features': list(common_features),
            'unique_exp1': list(set(exp1.feature_weights.keys()) - common_features),
            'unique_exp2': list(set(exp2.feature_weights.keys()) - common_features)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do explainer."""
        avg_time = np.mean(self.explanation_times) if self.explanation_times else 0
        
        return {
            'total_explanations': self.explanations_generated,
            'avg_explanation_time_ms': avg_time,
            'max_explanation_time_ms': max(self.explanation_times) if self.explanation_times else 0,
            'cache_size': len(self.explanation_cache),
            'cache_hit_rate': 0,  # Would need to track hits
            'num_samples_default': self.num_samples,
            'num_features_default': self.num_features
        }