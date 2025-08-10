"""
Confidence Calculator - Cálculo detalhado de confiança
Decompõe confiança em múltiplos fatores
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
from scipy import stats

from core.ai_engine.base_model import ModelType


logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """Breakdown detalhado da confiança."""
    
    overall_confidence: float
    model_agreement: float
    prediction_certainty: float
    feature_quality: float
    market_conditions: float
    historical_accuracy: float
    risk_adjusted_confidence: float
    factors: Dict[str, float]
    
    @property
    def is_high_confidence(self) -> bool:
        """Verifica se é alta confiança."""
        return self.risk_adjusted_confidence >= 0.75


class ConfidenceCalculator:
    """
    Calcula confiança detalhada nas predições.
    Considera múltiplos fatores para decisão robusta.
    """
    
    def __init__(self):
        """Inicializa calculador de confiança."""
        self.historical_accuracy: Dict[ModelType, List[float]] = {}
        self.recent_predictions: List[Dict] = []
        self.market_volatility_threshold = 0.05  # 5% volatility
        
    def calculate_confidence(
        self,
        model_predictions: Dict[ModelType, float],
        feature_vector: np.ndarray,
        market_data: Optional[Dict] = None
    ) -> ConfidenceBreakdown:
        """
        Calcula confiança detalhada.
        
        Args:
            model_predictions: Predições de cada modelo
            feature_vector: Vetor de features usado
            market_data: Dados de mercado opcionais
            
        Returns:
            Breakdown detalhado da confiança
        """
        factors = {}
        
        # 1. Model Agreement (30% weight)
        model_agreement = self._calculate_model_agreement(model_predictions)
        factors['model_agreement'] = model_agreement
        
        # 2. Prediction Certainty (25% weight)
        prediction_certainty = self._calculate_prediction_certainty(model_predictions)
        factors['prediction_certainty'] = prediction_certainty
        
        # 3. Feature Quality (15% weight)
        feature_quality = self._assess_feature_quality(feature_vector)
        factors['feature_quality'] = feature_quality
        
        # 4. Market Conditions (15% weight)
        market_conditions = self._assess_market_conditions(market_data)
        factors['market_conditions'] = market_conditions
        
        # 5. Historical Accuracy (15% weight)
        historical_accuracy = self._get_historical_accuracy(model_predictions.keys())
        factors['historical_accuracy'] = historical_accuracy
        
        # Calculate weighted overall confidence
        weights = {
            'model_agreement': 0.30,
            'prediction_certainty': 0.25,
            'feature_quality': 0.15,
            'market_conditions': 0.15,
            'historical_accuracy': 0.15
        }
        
        overall_confidence = sum(
            factors[key] * weights[key] 
            for key in weights
        )
        
        # Risk adjustment
        risk_factor = self._calculate_risk_adjustment(factors, market_data)
        risk_adjusted_confidence = overall_confidence * risk_factor
        
        return ConfidenceBreakdown(
            overall_confidence=float(overall_confidence),
            model_agreement=float(model_agreement),
            prediction_certainty=float(prediction_certainty),
            feature_quality=float(feature_quality),
            market_conditions=float(market_conditions),
            historical_accuracy=float(historical_accuracy),
            risk_adjusted_confidence=float(risk_adjusted_confidence),
            factors=factors
        )
    
    def _calculate_model_agreement(self, predictions: Dict[ModelType, float]) -> float:
        """
        Calcula concordância entre modelos.
        
        Args:
            predictions: Predições dos modelos
            
        Returns:
            Score de concordância (0-1)
        """
        if len(predictions) < 2:
            return 0.5
        
        values = list(predictions.values())
        
        # Standard deviation of predictions
        std = np.std(values)
        
        # Convert to agreement score
        # std=0 -> agreement=1, std=0.5 -> agreement=0
        agreement = max(0, 1 - (std * 2))
        
        # Check if all models agree on direction (>0.5 or <0.5)
        direction_agreement = all(v > 0.5 for v in values) or all(v < 0.5 for v in values)
        
        if direction_agreement:
            agreement = max(agreement, 0.6)  # Minimum 60% if direction agrees
        
        return agreement
    
    def _calculate_prediction_certainty(self, predictions: Dict[ModelType, float]) -> float:
        """
        Calcula certeza das predições.
        
        Args:
            predictions: Predições dos modelos
            
        Returns:
            Score de certeza (0-1)
        """
        if not predictions:
            return 0.0
        
        # Average distance from 0.5 (decision boundary)
        certainties = [abs(p - 0.5) * 2 for p in predictions.values()]
        avg_certainty = np.mean(certainties)
        
        # Penalize if models are too certain but disagree
        std = np.std(list(predictions.values()))
        if std > 0.2 and avg_certainty > 0.8:
            avg_certainty *= 0.7  # Reduce certainty
        
        return float(avg_certainty)
    
    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """
        Avalia qualidade das features.
        
        Args:
            features: Vetor de features
            
        Returns:
            Score de qualidade (0-1)
        """
        quality_score = 1.0
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            quality_score *= 0.5
        
        # Check for too many zeros (missing data)
        zero_ratio = np.sum(features == 0) / len(features)
        if zero_ratio > 0.3:  # More than 30% zeros
            quality_score *= (1 - zero_ratio)
        
        # Check variance (features should have some variance)
        if np.std(features) < 0.01:
            quality_score *= 0.7
        
        # Check for outliers
        z_scores = np.abs(stats.zscore(features[features != 0]))
        outlier_ratio = np.sum(z_scores > 3) / len(z_scores) if len(z_scores) > 0 else 0
        if outlier_ratio > 0.1:  # More than 10% outliers
            quality_score *= 0.8
        
        return float(max(0.1, quality_score))
    
    def _assess_market_conditions(self, market_data: Optional[Dict]) -> float:
        """
        Avalia condições de mercado.
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Score de condições (0-1)
        """
        if not market_data:
            return 0.5  # Neutral if no data
        
        score = 1.0
        
        # Check volatility
        volatility = market_data.get('volatility', 0)
        if volatility > self.market_volatility_threshold:
            # High volatility reduces confidence
            score *= max(0.3, 1 - (volatility / 0.1))
        
        # Check liquidity
        liquidity = market_data.get('liquidity_score', 1.0)
        score *= min(1.0, liquidity)
        
        # Check spread
        spread_bps = market_data.get('spread_bps', 0)
        if spread_bps > 50:  # High spread
            score *= 0.7
        
        # Check market hours
        is_active_hours = market_data.get('is_active_hours', True)
        if not is_active_hours:
            score *= 0.8
        
        return float(max(0.1, score))
    
    def _get_historical_accuracy(self, model_types: List[ModelType]) -> float:
        """
        Obtém acurácia histórica dos modelos.
        
        Args:
            model_types: Tipos de modelos
            
        Returns:
            Acurácia média histórica
        """
        if not self.historical_accuracy:
            return 0.7  # Default
        
        accuracies = []
        for model_type in model_types:
            if model_type in self.historical_accuracy:
                recent = self.historical_accuracy[model_type][-100:]  # Last 100
                if recent:
                    accuracies.append(np.mean(recent))
        
        return float(np.mean(accuracies)) if accuracies else 0.7
    
    def _calculate_risk_adjustment(
        self,
        factors: Dict[str, float],
        market_data: Optional[Dict]
    ) -> float:
        """
        Calcula ajuste de risco.
        
        Args:
            factors: Fatores de confiança
            market_data: Dados de mercado
            
        Returns:
            Fator de ajuste de risco (0-1)
        """
        risk_factor = 1.0
        
        # Reduce confidence if models disagree
        if factors.get('model_agreement', 0) < 0.5:
            risk_factor *= 0.7
        
        # Reduce if market conditions are poor
        if factors.get('market_conditions', 0) < 0.5:
            risk_factor *= 0.8
        
        # Reduce if features are low quality
        if factors.get('feature_quality', 0) < 0.5:
            risk_factor *= 0.8
        
        # Additional risk from market data
        if market_data:
            # High volatility
            if market_data.get('volatility', 0) > 0.1:
                risk_factor *= 0.7
            
            # Low volume
            if market_data.get('volume_24h', float('inf')) < 100000:
                risk_factor *= 0.8
        
        return float(max(0.3, risk_factor))
    
    def update_historical_accuracy(
        self,
        model_type: ModelType,
        was_correct: bool
    ) -> None:
        """
        Atualiza acurácia histórica.
        
        Args:
            model_type: Tipo do modelo
            was_correct: Se a predição estava correta
        """
        if model_type not in self.historical_accuracy:
            self.historical_accuracy[model_type] = []
        
        self.historical_accuracy[model_type].append(float(was_correct))
        
        # Keep only last 1000
        if len(self.historical_accuracy[model_type]) > 1000:
            self.historical_accuracy[model_type] = self.historical_accuracy[model_type][-1000:]
