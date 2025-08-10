"""
Decision Visualizer - Visualização de decisões de IA
Cria visualizações interpretáveis das decisões
Author: Crypto Arbitrage AI Team
Date: 2025
Python 3.11+
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from .shap_explainer import SHAPExplanation
from .confidence_calculator import ConfidenceBreakdown
from core.ai_engine.base_model import ModelType


logger = logging.getLogger(__name__)


class DecisionVisualizer:
    """
    Cria visualizações interativas das decisões de IA.
    Gera gráficos e relatórios explicativos.
    """
    
    def __init__(self, output_dir: str = "data/visualizations"):
        """
        Inicializa visualizador.
        
        Args:
            output_dir: Diretório para salvar visualizações
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_shap_explanation(
        self,
        explanation: SHAPExplanation,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualiza explicação SHAP.
        
        Args:
            explanation: Explicação SHAP
            save_path: Caminho para salvar
            
        Returns:
            Figura Plotly
        """
        # Create waterfall plot
        fig = go.Figure()
        
        # Get top features
        top_features = (
            explanation.top_positive_features[:5] + 
            explanation.top_negative_features[:5]
        )
        
        # Sort by impact
        top_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create waterfall data
        x = [f[0] for f in top_features]
        y = [f[1] for f in top_features]
        
        # Add waterfall trace
        fig.add_trace(go.Waterfall(
            x=x,
            y=y,
            text=[f"{v:.3f}" for v in y],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        # Update layout
        fig.update_layout(
            title=f"SHAP Feature Impact - {explanation.model_type.value}",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            showlegend=False,
            height=500
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"SHAP visualization saved to {save_path}")
        
        return fig
    
    def visualize_confidence_breakdown(
        self,
        breakdown: ConfidenceBreakdown,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualiza breakdown de confiança.
        
        Args:
            breakdown: Breakdown de confiança
            save_path: Caminho para salvar
            
        Returns:
            Figura Plotly
        """
        # Create radar chart
        fig = go.Figure()
        
        categories = list(breakdown.factors.keys())
        values = list(breakdown.factors.values())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Confidence Factors'
        ))
        
        # Add overall confidence as annotation
        fig.add_annotation(
            text=f"Overall Confidence: {breakdown.overall_confidence:.1%}<br>"
                 f"Risk-Adjusted: {breakdown.risk_adjusted_confidence:.1%}",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Confidence Breakdown Analysis",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Confidence visualization saved to {save_path}")
        
        return fig
    
    def create_decision_report(
        self,
        prediction: float,
        confidence: ConfidenceBreakdown,
        shap_explanations: Dict[ModelType, SHAPExplanation],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cria relatório completo da decisão.
        
        Args:
            prediction: Predição final
            confidence: Breakdown de confiança
            shap_explanations: Explicações SHAP por modelo
            metadata: Metadados adicionais
            
        Returns:
            Relatório estruturado
        """
        report = {
            'timestamp': time.time(),
            'decision': {
                'prediction': prediction,
                'recommendation': 'EXECUTE' if confidence.is_high_confidence else 'SKIP',
                'confidence': confidence.overall_confidence,
                'risk_adjusted_confidence': confidence.risk_adjusted_confidence
            },
            'confidence_factors': confidence.factors,
            'model_explanations': {},
            'top_factors': {
                'positive': [],
                'negative': []
            },
            'metadata': metadata
        }
        
        # Aggregate top factors from all models
        all_positive = []
        all_negative = []
        
        for model_type, explanation in shap_explanations.items():
            report['model_explanations'][model_type.value] = {
                'prediction': explanation.prediction_value,
                'base_value': explanation.base_value,
                'top_positive': explanation.top_positive_features[:3],
                'top_negative': explanation.top_negative_features[:3]
            }
            
            all_positive.extend(explanation.top_positive_features)
            all_negative.extend(explanation.top_negative_features)
        
        # Get unique top factors
        positive_dict = {}
        for feature, value in all_positive:
            if feature not in positive_dict:
                positive_dict[feature] = []
            positive_dict[feature].append(value)
        
        negative_dict = {}
        for feature, value in all_negative:
            if feature not in negative_dict:
                negative_dict[feature] = []
            negative_dict[feature].append(value)
        
        # Average and sort
        report['top_factors']['positive'] = sorted(
            [(k, np.mean(v)) for k, v in positive_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        report['top_factors']['negative'] = sorted(
            [(k, np.mean(v)) for k, v in negative_dict.items()],
            key=lambda x: x[1]
        )[:5]
        
        return report
    
    def save_decision_report(
        self,
        report: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Salva relatório de decisão.
        
        Args:
            report: Relatório da decisão
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"decision_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Decision report saved to {filepath}")
        
        return str(filepath)
    
    def create_ensemble_comparison(
        self,
        model_predictions: Dict[ModelType, float],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Cria comparação visual entre modelos do ensemble.
        
        Args:
            model_predictions: Predições de cada modelo
            save_path: Caminho para salvar
            
        Returns:
            Figura Plotly
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Model Predictions", "Prediction Distribution"),
            specs=[[{"type": "bar"}, {"type": "box"}]]
        )
        
        # Bar chart of predictions
        models = list(model_predictions.keys())
        predictions = list(model_predictions.values())
        
        fig.add_trace(
            go.Bar(
                x=[m.value for m in models],
                y=predictions,
                marker_color=['green' if p > 0.5 else 'red' for p in predictions],
                text=[f"{p:.3f}" for p in predictions],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Box plot of prediction distribution
        fig.add_trace(
            go.Box(
                y=predictions,
                name="Predictions",
                boxpoints='all',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title="Ensemble Model Comparison",
            showlegend=False,
            height=400
        )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Prediction", row=1, col=1)
        fig.update_yaxes(title_text="Prediction", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Ensemble comparison saved to {save_path}")
        
        return fig