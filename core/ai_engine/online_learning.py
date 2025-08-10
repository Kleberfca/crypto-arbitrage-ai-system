"""
Online Learning - Sistema de aprendizado incremental
Atualiza modelos com novos dados sem retreino completo
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from datetime import datetime
import logging

import numpy as np
from sklearn.linear_model import SGDClassifier
from river import linear_model, preprocessing, metrics

from .ensemble_model import EnsembleModel
from config.settings import settings


logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Sistema de aprendizado online para atualização incremental.
    Mantém modelos atualizados com dados recentes.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Inicializa online learner.
        
        Args:
            buffer_size: Tamanho do buffer de experiências
        """
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Online models (lightweight for incremental updates)
        self.online_models = {
            'sgd': SGDClassifier(loss='log', learning_rate='adaptive', eta0=0.01),
            'river': linear_model.LogisticRegression()
        }
        
        # Metrics tracking
        self.online_metrics = {
            'accuracy': deque(maxlen=100),
            'updates': 0,
            'last_update': None
        }
        
        # Main ensemble
        self.ensemble = None
        self.update_frequency = 100  # Update after N new samples
        
    def add_experience(
        self,
        features: np.ndarray,
        label: int,
        confidence: float,
        reward: Optional[float] = None
    ) -> None:
        """
        Adiciona nova experiência ao buffer.
        
        Args:
            features: Features do trade
            label: Label real (1 = profitable, 0 = not)
            confidence: Confiança da predição
            reward: Reward opcional (profit/loss)
        """
        experience = {
            'features': features,
            'label': label,
            'confidence': confidence,
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        
        # Trigger update if buffer is full
        if len(self.experience_buffer) >= self.update_frequency:
            self.incremental_update()
    
    def incremental_update(self) -> Dict[str, float]:
        """
        Realiza atualização incremental dos modelos.
        
        Returns:
            Métricas da atualização
        """
        if len(self.experience_buffer) < 10:
            return {}
        
        logger.info(f"Performing incremental update with {len(self.experience_buffer)} samples")
        
        # Prepare batch
        X = np.array([exp['features'] for exp in self.experience_buffer])
        y = np.array([exp['label'] for exp in self.experience_buffer])
        
        # Update SGD model
        self.online_models['sgd'].partial_fit(X, y, classes=[0, 1])
        
        # Update River model (one by one)
        river_accuracy = 0
        for features, label in zip(X, y):
            # Predict before update
            pred = self.online_models['river'].predict_one(dict(enumerate(features)))
            river_accuracy += (pred == label)
            
            # Update
            self.online_models['river'].learn_one(
                dict(enumerate(features)),
                label
            )
        
        river_accuracy /= len(X)
        
        # Calculate metrics
        sgd_accuracy = self.online_models['sgd'].score(X, y)
        
        metrics = {
            'sgd_accuracy': sgd_accuracy,
            'river_accuracy': river_accuracy,
            'samples_used': len(X),
            'timestamp': time.time()
        }
        
        # Store metrics
        self.online_metrics['accuracy'].append(sgd_accuracy)
        self.online_metrics['updates'] += 1
        self.online_metrics['last_update'] = time.time()
        
        logger.info(
            f"Incremental update complete",
            sgd_acc=f"{sgd_accuracy:.3f}",
            river_acc=f"{river_accuracy:.3f}"
        )
        
        return metrics
    
    def adapt_ensemble_weights(self) -> Dict[str, float]:
        """
        Adapta pesos do ensemble baseado em performance recente.
        
        Returns:
            Novos pesos do ensemble
        """
        if not self.ensemble or len(self.experience_buffer) < 50:
            return {}
        
        # Calculate recent performance per model
        model_performances = {}
        
        for exp in list(self.experience_buffer)[-50:]:
            # Get individual model predictions (if stored)
            # For now, use mock performance
            pass
        
        # Adjust weights based on performance
        # Simplified: increase weight of better performing models
        
        new_weights = self.ensemble.weights.copy()
        
        # Mock adjustment
        total = sum(new_weights.values())
        for model in new_weights:
            new_weights[model] /= total
        
        logger.info(f"Adapted ensemble weights: {new_weights}")
        
        return new_weights
    
    def get_online_prediction(self, features: np.ndarray) -> float:
        """
        Faz predição usando modelos online.
        
        Args:
            features: Features para predição
            
        Returns:
            Probabilidade (0-1)
        """
        # SGD prediction
        if hasattr(self.online_models['sgd'], 'predict_proba'):
            sgd_pred = self.online_models['sgd'].predict_proba(features.reshape(1, -1))[0, 1]
        else:
            sgd_pred = 0.5
        
        # River prediction
        river_pred = self.online_models['river'].predict_proba_one(
            dict(enumerate(features))
        ).get(1, 0.5)
        
        # Average predictions
        return (sgd_pred + river_pred) / 2
    
    def should_retrain_ensemble(self) -> bool:
        """
        Decide se deve retreinar ensemble completo.
        
        Returns:
            True se retreino é necessário
        """
        # Retrain if:
        # 1. Accuracy dropped significantly
        if self.online_metrics['accuracy']:
            recent_acc = np.mean(list(self.online_metrics['accuracy'])[-10:])
            if recent_acc < 0.7:  # Below 70% accuracy
                return True
        
        # 2. Too many updates without retrain
        if self.online_metrics['updates'] > 1000:
            return True
        
        # 3. Time-based (daily)
        if self.online_metrics['last_update']:
            hours_since = (time.time() - self.online_metrics['last_update']) / 3600
            if hours_since > 24:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do online learning."""
        return {
            'buffer_size': len(self.experience_buffer),
            'total_updates': self.online_metrics['updates'],
            'recent_accuracy': np.mean(list(self.online_metrics['accuracy'])[-10:]) if self.online_metrics['accuracy'] else 0,
            'last_update': self.online_metrics['last_update'],
            'should_retrain': self.should_retrain_ensemble()
        }
