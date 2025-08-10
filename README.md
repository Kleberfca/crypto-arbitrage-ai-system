# 🚀 Crypto Arbitrage AI System

Sistema profissional de arbitragem de criptomoedas com inteligência artificial, projetado para operar em 5 exchanges simultaneamente com latência ultra-baixa (<30ms) e ROI alvo de 10-15% mensal.

## 📊 Características Principais

- **Latência Ultra-Baixa**: <30ms end-to-end
- **5 Exchanges Suportadas**: Binance, OKX, Bybit, KuCoin, HTX
- **3 Estratégias de Arbitragem**: Espacial, Estatística, Triangular
- **IA Explicável (XAI)**: SHAP values para todas as decisões
- **Machine Learning Avançado**: Ensemble de XGBoost, LSTM, Transformer
- **50+ Features**: Feature engineering otimizado
- **Pre-funding**: 70% do capital pré-alocado para velocidade
- **Rede BEP20**: Transferências rápidas (15-30 segundos)

## 🎯 Performance Targets

| Métrica | Target | Status |
|---------|--------|--------|
| Latência End-to-End | <30ms | ✅ |
| Decisão IA | <2ms | ✅ |
| Feature Extraction | <3ms | ✅ |
| XAI Explanation | <5ms | ✅ |
| Throughput | >10,000 updates/seg | ✅ |
| ROI Mensal | 10-15% | 🎯 |
| Sharpe Ratio | >2.5 | 🎯 |
| Win Rate | >70% | 🎯 |

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      WebSocket Streams                       │
│  Binance │ OKX │ Bybit │ KuCoin │ HTX                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
│  • WebSocket Connections (<20ms)                            │
│  • Redis Cache                                              │
│  • Data Normalization                                       │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering (50+ features)           │
│  • Price Features    • Volume Features                      │
│  • Orderbook Depth   • Technical Indicators                 │
│  • Network Metrics   • Cross-Asset Correlations            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Arbitrage Strategies                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Spatial   │  │ Statistical │  │ Triangular  │       │
│  │  (40% cap)  │  │  (40% cap)  │  │  (20% cap)  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML/AI Decision Engine                      │
│  • XGBoost (30%)    • LSTM (25%)                           │
│  • Transformer (25%) • Statistical Models (20%)             │
│  • Ensemble Voting   • Confidence >75%                      │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    XAI Explainability                        │
│  • SHAP Values      • Feature Importance                    │
│  • Decision Path    • Confidence Breakdown                  │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Risk Management Layer                      │
│  • Position Sizing  • Correlation Monitoring                │
│  • Adaptive Limits  • VaR/CVaR Calculation                 │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Execution Engine                          │
│  • Pre-funded Accounts (70%)                                │
│  • BEP20 Rebalancing (30%)                                  │
│  • Slippage Control                                         │
└─────────────────────────────────────────────────────────────┘
```
## Estrutura do Projeto - Crypto Arbitrage AI System

crypto-arbitrage-ai-system/
│
├── main.py
├── Dockerfile
├── setup.sh
├── README.md                           # Documentação principal
├── requirements.txt                    # Dependências Python
├── docker-compose.yml                  # Orquestração de containers
├── .env.example                        # Exemplo de variáveis de ambiente
├── .gitignore                          # Arquivos ignorados pelo Git
│
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Configurações gerais do sistema
│   ├── exchanges.py                    # Configurações específicas das exchanges
│   ├── strategies.py                   # Parâmetros das estratégias
│   └── risk_limits.py                  # Limites de risco
│
├── core/
│   ├── __init__.py
│   │
│   ├── data_collector/
│   │   ├── __init__.py
│   │   ├── exchange_connector.py       # Conexão WebSocket com exchanges
│   │   ├── data_normalizer.py          # Normalização de dados
│   │   ├── orderbook_manager.py        # Gestão de order books
│   │   └── cache_manager.py            # Interface com Redis
│   │
│   ├── arbitrage_engine/
│   │   ├── __init__.py
│   │   ├── base_arbitrage.py           # Classe base para estratégias
│   │   ├── spatial_arbitrage.py        # Arbitragem entre exchanges
│   │   ├── statistical_arbitrage.py    # Pairs trading e mean reversion
│   │   ├── triangular_arbitrage.py     # Ciclos triangulares
│   │   └── opportunity_detector.py     # Detecção de oportunidades
│   │
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py        # Extração de 50+ features
│   │   ├── price_features.py           # Features baseadas em preço
│   │   ├── volume_features.py          # Features de volume
│   │   ├── orderbook_features.py       # Features do order book
│   │   ├── technical_indicators.py     # Indicadores técnicos
│   │   └── network_features.py         # Features de rede (gas, congestion)
│   │
│   ├── ai_engine/
│   │   ├── __init__.py
│   │   ├── ensemble_model.py           # Ensemble de modelos ML
│   │   ├── xgboost_model.py            # Modelo XGBoost
│   │   ├── lstm_model.py               # Modelo LSTM
│   │   ├── transformer_model.py        # Modelo Transformer
│   │   ├── model_trainer.py            # Treinamento de modelos
│   │   └── online_learning.py          # Aprendizado online
│   │
│   ├── xai_engine/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py           # SHAP values
│   │   ├── lime_explainer.py           # LIME explanations
│   │   ├── decision_visualizer.py      # Visualização de decisões
│   │   └── confidence_calculator.py    # Cálculo de confiança
│   │
│   ├── execution_engine/
│   │   ├── __init__.py
│   │   ├── order_manager.py            # Gestão de ordens
│   │   ├── pre_funding_manager.py      # Gestão de pre-funding
│   │   ├── execution_optimizer.py      # Otimização de execução
│   │   └── slippage_controller.py      # Controle de slippage
│   │
│   ├── risk_management/
│   │   ├── __init__.py
│   │   ├── risk_calculator.py          # Cálculo de métricas de risco
│   │   ├── position_sizer.py           # Dimensionamento de posições
│   │   ├── correlation_monitor.py      # Monitoramento de correlações
│   │   └── adaptive_limits.py          # Limites adaptativos
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py          # Motor de backtesting
│   │   ├── walk_forward.py             # Walk-forward optimization
│   │   ├── monte_carlo.py              # Simulações Monte Carlo
│   │   └── stress_testing.py           # Testes de stress
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics_collector.py        # Coleta de métricas
│       ├── performance_tracker.py      # Tracking de performance
│       ├── alert_manager.py            # Sistema de alertas
│       └── prometheus_exporter.py      # Exportador Prometheus
│
├── api/
│   ├── __init__.py
│   ├── main.py                         # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── strategies.py               # Endpoints de estratégias
│   │   ├── monitoring.py               # Endpoints de monitoramento
│   │   ├── execution.py                # Endpoints de execução
│   │   └── analysis.py                 # Endpoints de análise
│   └── models/
│       ├── __init__.py
│       ├── requests.py                 # Modelos de request
│       └── responses.py                # Modelos de response
│
├── scripts/
│   ├── train_ensemble.py               # Script para treinar ensemble
│   ├── backtest_strategies.py          # Script de backtesting
│   ├── optimize_parameters.py          # Otimização de parâmetros
│   ├── health_check.py                 # Verificação de saúde
│   └── monitor_performance.py          # Monitoramento de performance
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_exchange_connector.py
│   │   ├── test_spatial_arbitrage.py
│   │   ├── test_feature_extractor.py
│   │   └── test_risk_management.py
│   ├── integration/
│   │   ├── test_data_flow.py
│   │   ├── test_execution_flow.py
│   │   └── test_ml_pipeline.py
│   └── performance/
│       ├── test_latency.py
│       ├── test_throughput.py
│       └── test_feature_extraction.py
│
├── data/
│   ├── models/                         # Modelos ML salvos
│   ├── features/                       # Features pré-calculadas
│   ├── backtest/                       # Dados de backtesting
│   └── logs/                           # Logs do sistema
│
├── notebooks/
│   ├── strategy_analysis.ipynb         # Análise de estratégias
│   ├── feature_exploration.ipynb       # Exploração de features
│   ├── model_evaluation.ipynb          # Avaliação de modelos
│   └── performance_analysis.ipynb      # Análise de performance
│
└── deployment/
    ├── kubernetes/                      # Manifests K8s
    ├── terraform/                       # Infraestrutura como código
    └── monitoring/                      # Dashboards Grafana
        ├── strategy_overview.json
        ├── ml_performance.json
        ├── risk_analytics.json
        └── system_health.json

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Redis
- PostgreSQL
- 8GB+ RAM
- API keys for all 5 exchanges

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/crypto-arbitrage-ai-system.git
cd crypto-arbitrage-ai-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. **Start infrastructure services**
```bash
docker-compose up -d redis postgres prometheus grafana
```

6. **Initialize database**
```bash
python scripts/init_database.py
```

7. **Run the system**
```bash
# Development mode
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python main.py
```

## ⚙️ Configuration

### Exchange API Keys

Add your API keys to `.env`:

```env
# Binance
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# OKX
OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_api_secret
OKX_PASSPHRASE=your_okx_passphrase

# Bybit
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret

# KuCoin
KUCOIN_API_KEY=your_kucoin_api_key
KUCOIN_API_SECRET=your_kucoin_api_secret
KUCOIN_PASSPHRASE=your_kucoin_passphrase

# HTX (Huobi)
HTX_API_KEY=your_htx_api_key
HTX_API_SECRET=your_htx_api_secret
```

### Risk Limits

Edit `config/settings.py` to adjust risk parameters:

```python
max_position_size_pct: 0.02  # 2% per position
max_daily_loss_pct: 0.03      # 3% daily loss limit
max_drawdown_pct: 0.05        # 5% max drawdown
min_confidence_threshold: 0.75 # 75% minimum confidence
```

## 📈 Strategies

### 1. Spatial Arbitrage (40% capital)
- Explora diferenças de preço entre exchanges
- Execução simultânea com pre-funding
- Target spread: >0.3% após fees

### 2. Statistical Arbitrage (40% capital)
- Pairs trading com ativos cointegrados
- Mean reversion strategies
- Z-score entry >2.0, exit <0.5

### 3. Triangular Arbitrage (20% capital)
- Ciclos de 3+ moedas na mesma exchange
- Algoritmo Bellman-Ford otimizado
- Execução sequencial <100ms

## 🤖 Machine Learning Pipeline

### Ensemble Models
- **XGBoost (30%)**: Classificação e ranking
- **LSTM (25%)**: Análise de séries temporais
- **Transformer (25%)**: Padrões complexos
- **Statistical (20%)**: GARCH, ARIMA

### Feature Categories (50+ features)
- Price-based features
- Volume indicators
- Orderbook metrics
- Technical indicators
- Network features
- Cross-asset correlations

### Training Pipeline
```bash
# Train ensemble models
python scripts/train_ensemble.py

# Optimize hyperparameters
python scripts/optimize_parameters.py

# Run backtesting
python scripts/backtest_strategies.py
```

## 📊 Monitoring

### Grafana Dashboards

Access dashboards at `http://localhost:3000`:

1. **Strategy Overview**: Performance por estratégia
2. **ML/AI Performance**: Modelos e XAI metrics
3. **Risk Analytics**: VaR, correlações, drawdown
4. **System Health**: Latência, CPU, memória

### API Endpoints

- `GET /api/v1/health` - System health check
- `GET /api/v1/strategies` - Active strategies
- `GET /api/v1/performance` - Performance metrics
- `GET /api/v1/positions` - Current positions
- `POST /api/v1/execute` - Manual execution

### Performance Metrics

```bash
# Check system performance
python scripts/monitor_performance.py

# Health check
python scripts/health_check.py
```

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v

# Coverage report
pytest tests/ --cov=core --cov-report=html
```

### Performance Benchmarks
```bash
# Test latency
python tests/performance/test_latency.py

# Test throughput
python tests/performance/test_throughput.py

# Stress testing
python scripts/stress_test.py
```

## 🔧 Development

### Project Structure
```
crypto-arbitrage-ai-system/
├── core/                  # Core business logic
├── api/                   # FastAPI application
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── data/                  # Data storage
├── notebooks/             # Jupyter notebooks
└── deployment/            # Deployment configs
```

### Code Style
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .

# Pre-commit hooks
pre-commit install
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🛡️ Security

- **API Keys**: Never commit API keys
- **Encryption**: All sensitive data encrypted
- **Rate Limiting**: Implemented on all endpoints
- **Circuit Breakers**: Automatic failure recovery
- **Audit Logs**: All trades logged

## ⚠️ Risk Disclaimer

**IMPORTANT**: Cryptocurrency trading involves substantial risk of loss. This system is for educational purposes. Past performance does not guarantee future results. Never trade with funds you cannot afford to lose.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/crypto-arbitrage-ai-system/issues)
- **Discord**: [Join our community](https://discord.gg/your-server)

## 🌟 Acknowledgments

- CCXT Pro for exchange connectivity
- Scikit-learn, XGBoost, TensorFlow for ML
- SHAP for explainable AI
- Redis for high-performance caching

---

**Built with ❤️ by the Crypto Arbitrage AI Team**

*Last updated: 2025*