"""
XGBoost Model - Gradient Boosting para classificação de oportunidades
30% peso no ensemble, otimizado para latência <0.5ms
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, Any, List, Optional
import logging

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

from .base_model import BaseModel, ModelType, PredictionResult


logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost implementation para arbitragem.
    Foco em classificação binária de oportunidades lucrativas.
    """
    
    def __init__(self, version: str = "1.0.0"):
        """Inicializa XGBoost model."""
        super().__init__(ModelType.XGBOOST, version)
        self.use_gpu = False  # Set True if GPU available
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Parâmetros otimizados para latência e accuracy."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'n_jobs': -1,
            'random_state': 42,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor'
        }
    
    def _create_model(self) -> xgb.XGBClassifier:
        """Cria instância do XGBoost."""
        return xgb.XGBClassifier(**self.params)
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepara features para XGBoost.
        Remove NaN, Inf e normaliza outliers.
        """
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values (outliers)
        percentile_99 = np.percentile(np.abs(X_clean), 99)
        X_clean = np.clip(X_clean, -percentile_99, percentile_99)
        
        return X_clean
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """
        Calcula confiança usando distância da decisão boundary.
        
        Args:
            X: Features scaled
            prediction: Probabilidade predita
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from prediction probability
        base_confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        
        # Get prediction margin if available
        if hasattr(self.model, 'predict_proba'):
            # Get tree predictions for variance
            tree_predictions = []
            for estimator in self.model.estimators_:
                tree_pred = estimator[0].predict(X)[0]
                tree_predictions.append(tree_pred)
            
            # Lower confidence if trees disagree
            if tree_predictions:
                tree_std = np.std(tree_predictions)
                variance_penalty = min(tree_std * 2, 0.5)
                base_confidence *= (1 - variance_penalty)
        
        # Ensure minimum confidence for strong predictions
        if prediction > 0.9 or prediction < 0.1:
            base_confidence = max(base_confidence, 0.8)
        
        return float(min(base_confidence, 1.0))
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """
        Otimiza hyperparâmetros usando RandomizedSearchCV.
        
        Args:
            X: Features de treino
            y: Labels
            n_iter: Número de iterações
            
        Returns:
            Melhores parâmetros encontrados
        """
        param_distributions = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
        
        # Prepare data
        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        )
        
        # Random search
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_scaled, y)
        
        # Update params
        self.params.update(search.best_params_)
        
        logger.info(
            f"Hyperparameter optimization complete",
            best_score=f"{search.best_score_:.3f}",
            best_params=search.best_params_
        )
        
        return search.best_params_
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna feature importance do XGBoost.
        
        Returns:
            Dicionário ordenado por importância
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def predict_with_trees(self, X: np.ndarray, ntree_limit: int = 0) -> PredictionResult:
        """
        Predição usando subset de árvores para latência ainda menor.
        
        Args:
            X: Features
            ntree_limit: Número de árvores a usar (0 = todas)
            
        Returns:
            Resultado da predição
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        start_time = time.perf_counter()
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared.reshape(1, -1))
        
        # Convert to DMatrix for faster prediction
        dmatrix = xgb.DMatrix(X_scaled)
        
        # Predict with limited trees
        prob = self.model.get_booster().predict(
            dmatrix,
            ntree_limit=ntree_limit if ntree_limit > 0 else None
        )[0]
        
        # Calculate confidence
        confidence = self._calculate_confidence(X_scaled, prob)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return PredictionResult(
            prediction=float(prob),
            confidence=confidence,
            features_used=self.feature_names[:ntree_limit] if ntree_limit > 0 else self.feature_names,
            inference_time_ms=inference_time,
            model_type=self.model_type
        )
