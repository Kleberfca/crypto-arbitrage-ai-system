"""
LSTM Model - Long Short-Term Memory para séries temporais
25% peso no ensemble, análise de padrões temporais
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from .base_model import BaseModel, ModelType, PredictionResult


logger = logging.getLogger(__name__)

# Configurar TensorFlow para melhor performance
tf.config.optimizer.set_jit(True)  # XLA compilation
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class LSTMModel(BaseModel):
    """
    LSTM implementation para análise de séries temporais.
    Detecta padrões em sequências de preços e volumes.
    """
    
    def __init__(self, version: str = "1.0.0"):
        """Inicializa LSTM model."""
        super().__init__(ModelType.LSTM, version)
        self.sequence_length = 100  # Últimos 100 timesteps
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_checkpoint = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Parâmetros otimizados para LSTM."""
        return {
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy', 'AUC']
        }
    
    def _create_model(self) -> keras.Model:
        """
        Cria arquitetura LSTM com attention mechanism.
        
        Returns:
            Modelo Keras compilado
        """
        # Input shape: (batch_size, sequence_length, n_features)
        n_features = len(self.feature_names) if self.feature_names else 74
        
        inputs = layers.Input(shape=(self.sequence_length, n_features))
        
        # First LSTM layer with return sequences
        x = layers.LSTM(
            self.params['lstm_units'][0],
            return_sequences=True,
            dropout=self.params['dropout_rate'],
            recurrent_dropout=self.params['dropout_rate']
        )(inputs)
        x = layers.BatchNormalization()(x)
        
        # Second LSTM layer with return sequences for attention
        x = layers.LSTM(
            self.params['lstm_units'][1],
            return_sequences=True,
            dropout=self.params['dropout_rate']
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.params['lstm_units'][1]
        )(x, x)
        x = layers.Add()([x, attention])  # Residual connection
        x = layers.LayerNormalization()(x)
        
        # Final LSTM layer
        x = layers.LSTM(
            self.params['lstm_units'][2],
            dropout=self.params['dropout_rate']
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=self.params['loss'],
            metrics=self.params['metrics']
        )
        
        return model
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepara features para LSTM (sequences).
        
        Args:
            X: Features raw
            
        Returns:
            Features em formato de sequência
        """
        # Handle single sample or batch
        if len(X.shape) == 1:
            # Single sample - create fake sequence
            X_seq = np.tile(X, (self.sequence_length, 1))
            return X_seq.reshape(1, self.sequence_length, -1)
        
        elif len(X.shape) == 2:
            # Batch of samples
            n_samples = X.shape[0]
            
            if n_samples >= self.sequence_length:
                # Create sequences from batch
                sequences = []
                for i in range(n_samples - self.sequence_length + 1):
                    sequences.append(X[i:i + self.sequence_length])
                return np.array(sequences)
            else:
                # Pad if not enough samples
                padding = np.tile(X[0], (self.sequence_length - n_samples, 1))
                X_padded = np.vstack([padding, X])
                return X_padded.reshape(1, self.sequence_length, -1)
        
        elif len(X.shape) == 3:
            # Already in sequence format
            return X
        
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Treina modelo LSTM com callbacks.
        
        Args:
            X: Features de treino
            y: Labels
            feature_names: Nomes das features
            validation_split: Split para validação
            
        Returns:
            Métricas de treinamento
        """
        start_time = time.perf_counter()
        
        try:
            self.feature_names = feature_names
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Prepare sequences
            X_sequences = self._prepare_features(X_scaled)
            
            # Adjust y to match sequence output
            if len(y) > len(X_sequences):
                y_sequences = y[-len(X_sequences):]
            else:
                y_sequences = y
            
            # Create model
            self.model = self._create_model()
            
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.params['reduce_lr_patience'],
                min_lr=1e-6
            )
            
            # Train model
            history = self.model.fit(
                X_sequences, y_sequences,
                batch_size=self.params['batch_size'],
                epochs=self.params['epochs'],
                validation_split=validation_split,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Get final metrics
            val_loss = min(history.history['val_loss'])
            val_accuracy = max(history.history['val_accuracy'])
            val_auc = max(history.history.get('val_auc', [0]))
            
            self.training_metrics = {
                'accuracy': val_accuracy,
                'auc': val_auc,
                'loss': val_loss,
                'training_time': time.perf_counter() - start_time,
                'epochs_trained': len(history.history['loss']),
                'n_samples': len(X_sequences),
                'n_features': X.shape[1]
            }
            
            self.is_trained = True
            
            logger.info(
                f"LSTM trained",
                accuracy=f"{val_accuracy:.3f}",
                auc=f"{val_auc:.3f}",
                epochs=len(history.history['loss']),
                time=f"{self.training_metrics['training_time']:.2f}s"
            )
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Faz predição com LSTM.
        
        Args:
            X: Features (ou sequência de features)
            
        Returns:
            Resultado da predição
        """
        if not self.is_trained:
            raise ValueError("LSTM model not trained")
        
        start_time = time.perf_counter()
        
        try:
            # Scale features
            if len(X.shape) == 1:
                X_scaled = self.feature_scaler.transform(X.reshape(1, -1))
            else:
                X_scaled = self.feature_scaler.transform(X)
            
            # Prepare sequence
            X_seq = self._prepare_features(X_scaled)
            
            # Predict
            prob = self.model.predict(X_seq, verbose=0)[0, 0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(X_seq, prob)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Check latency
            if inference_time > 1.0:  # 1ms target for LSTM
                logger.warning(f"High LSTM inference latency: {inference_time:.2f}ms")
            
            return PredictionResult(
                prediction=float(prob),
                confidence=confidence,
                features_used=self.feature_names,
                inference_time_ms=inference_time,
                model_type=self.model_type
            )
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            raise
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """
        Calcula confiança usando Monte Carlo dropout.
        
        Args:
            X: Sequência de features
            prediction: Predição base
            
        Returns:
            Confidence score
        """
        if not self.is_trained:
            return 0.5
        
        # Monte Carlo predictions (5 forward passes with dropout)
        mc_predictions = []
        for _ in range(5):
            # Enable dropout during inference
            pred = self.model(X, training=True).numpy()[0, 0]
            mc_predictions.append(pred)
        
        # Calculate uncertainty
        mc_std = np.std(mc_predictions)
        
        # High std = low confidence
        confidence = max(0, 1 - (mc_std * 4))
        
        # Boost confidence for extreme predictions
        if prediction > 0.9 or prediction < 0.1:
            confidence = max(confidence, 0.8)
        
        return float(confidence)
    
    def predict_batch(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Predição em batch para melhor throughput.
        
        Args:
            X: Batch de features
            
        Returns:
            Lista de resultados
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        start_time = time.perf_counter()
        
        # Scale all features
        X_scaled = self.feature_scaler.transform(X)
        
        # Prepare sequences
        X_sequences = self._prepare_features(X_scaled)
        
        # Batch prediction
        probs = self.model.predict(X_sequences, batch_size=32, verbose=0)
        
        # Calculate time per sample
        total_time = (time.perf_counter() - start_time) * 1000
        time_per_sample = total_time / len(probs)
        
        # Create results
        results = []
        for prob in probs:
            results.append(PredictionResult(
                prediction=float(prob[0]),
                confidence=0.75,  # Simplified for batch
                features_used=self.feature_names,
                inference_time_ms=time_per_sample,
                model_type=self.model_type
            ))
        
        return results
