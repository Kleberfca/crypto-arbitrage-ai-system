"""
Transformer Model - Self-attention para padrões complexos
25% peso no ensemble, análise de relações não-lineares
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, Any, List, Optional
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel, ModelType, PredictionResult


logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Bloco Transformer com self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CryptoTransformer(nn.Module):
    """Transformer model para arbitragem."""
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global pooling
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        output = self.classifier(x)
        
        return output


class TransformerModel(BaseModel):
    """
    Transformer implementation para detecção de padrões complexos.
    Usa self-attention para capturar relações não-lineares.
    """
    
    def __init__(self, version: str = "1.0.0"):
        """Inicializa Transformer model."""
        super().__init__(ModelType.TRANSFORMER, version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 100
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Parâmetros otimizados para Transformer."""
        return {
            'embed_dim': 128,
            'num_heads': 8,
            'ff_dim': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 30,
            'weight_decay': 0.01,
            'warmup_steps': 500
        }
    
    def _create_model(self) -> CryptoTransformer:
        """Cria modelo Transformer."""
        n_features = len(self.feature_names) if self.feature_names else 74
        
        model = CryptoTransformer(
            input_dim=n_features,
            embed_dim=self.params['embed_dim'],
            num_heads=self.params['num_heads'],
            ff_dim=self.params['ff_dim'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout']
        )
        
        return model.to(self.device)
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepara features para Transformer."""
        # Similar to LSTM preparation
        if len(X.shape) == 1:
            X_seq = np.tile(X, (self.sequence_length, 1))
            return X_seq.reshape(1, self.sequence_length, -1)
        
        elif len(X.shape) == 2:
            n_samples = X.shape[0]
            if n_samples >= self.sequence_length:
                sequences = []
                for i in range(n_samples - self.sequence_length + 1):
                    sequences.append(X[i:i + self.sequence_length])
                return np.array(sequences)
            else:
                padding = np.tile(X[0], (self.sequence_length - n_samples, 1))
                X_padded = np.vstack([padding, X])
                return X_padded.reshape(1, self.sequence_length, -1)
        
        return X
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Treina modelo Transformer."""
        start_time = time.perf_counter()
        
        try:
            self.feature_names = feature_names
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            X_sequences = self._prepare_features(X_scaled)
            
            # Adjust labels
            if len(y) > len(X_sequences):
                y_sequences = y[-len(X_sequences):]
            else:
                y_sequences = y
            
            # Split data
            n_val = int(len(X_sequences) * validation_split)
            X_train = X_sequences[:-n_val]
            y_train = y_sequences[:-n_val]
            X_val = X_sequences[-n_val:]
            y_val = y_sequences[-n_val:]
            
            # Create DataLoaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params['batch_size']
            )
            
            # Create model
            self.model = self._create_model()
            
            # Optimizer with warmup
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params['learning_rate'],
                weight_decay=self.params['weight_decay']
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.params['epochs']
            )
            
            # Training loop
            best_val_loss = float('inf')
            for epoch in range(self.params['epochs']):
                # Train
                self.model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_x).squeeze()
                    loss = F.binary_cross_entropy(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validate
                self.model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x).squeeze()
                        loss = F.binary_cross_entropy(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predictions = (outputs > 0.5).float()
                        val_correct += (predictions == batch_y).sum().item()
                
                # Update scheduler
                scheduler.step()
                
                # Save best model
                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.best_model_state = self.model.state_dict().copy()
            
            # Load best model
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            # Calculate final metrics
            val_accuracy = val_correct / len(X_val)
            
            self.training_metrics = {
                'accuracy': val_accuracy,
                'loss': best_val_loss,
                'training_time': time.perf_counter() - start_time,
                'n_samples': len(X_sequences),
                'n_features': X.shape[1]
            }
            
            self.is_trained = True
            
            logger.info(
                f"Transformer trained",
                accuracy=f"{val_accuracy:.3f}",
                loss=f"{best_val_loss:.4f}",
                time=f"{self.training_metrics['training_time']:.2f}s"
            )
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Faz predição com Transformer."""
        if not self.is_trained:
            raise ValueError("Transformer not trained")
        
        start_time = time.perf_counter()
        
        try:
            # Prepare features
            X_scaled = self.scaler.transform(X.reshape(1, -1) if len(X.shape) == 1 else X)
            X_seq = self._prepare_features(X_scaled)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prob = self.model(X_tensor).cpu().numpy()[0, 0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(X_seq, prob)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            if inference_time > 1.5:  # 1.5ms target
                logger.warning(f"High Transformer inference latency: {inference_time:.2f}ms")
            
            return PredictionResult(
                prediction=float(prob),
                confidence=confidence,
                features_used=self.feature_names,
                inference_time_ms=inference_time,
                model_type=self.model_type
            )
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            raise
    
    def _calculate_confidence(self, X: np.ndarray, prediction: float) -> float:
        """Calcula confiança usando attention weights."""
        # Simplified confidence based on prediction extremity
        base_confidence = abs(prediction - 0.5) * 2
        
        # Boost for extreme predictions
        if prediction > 0.85 or prediction < 0.15:
            base_confidence = max(base_confidence, 0.75)
        
        return float(min(base_confidence, 1.0))
