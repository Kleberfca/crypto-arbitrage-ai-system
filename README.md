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