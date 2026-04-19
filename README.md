# POLY-BELTANE — Trading Bot para Mercados de Temperatura

Bot de trading automatizado para Polymarket focado em previsão de temperaturas máximas em Munique, usando ensemble de modelos de Machine Learning.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Treinamento](#treinamento)
- [Backtesting](#backtesting)
- [Live Trading](#live-trading)
- [Otimização](#otimização)
- [Métricas](#métricas)
- [Troubleshooting](#troubleshooting)

## 🎯 Visão Geral

O POLY-BELTANE é um sistema de trading automatizado que:

1. **Coleta dados meteorológicos** em tempo real (Wunderground + Open-Meteo)
2. **Previsão ML ensemble** usando LightGBM, XGBoost e Z-Score
3. **Gestão de risco** com Stop-Loss e Fuzzy Gatekeeper
4. **Execução automática** de ordens na Polymarket (CLOB)

### Características

- ✅ Ensemble ponderado (50% LGBM + 30% XGB + 20% Z-Score)
- ✅ Stop-Loss dinâmico (temperatura + probabilidade)
- ✅ Fuzzy Gatekeeper para validação de contexto
- ✅ Phased Entry (3 parcelas) ou Single Entry
- ✅ Otimização automática com Optuna
- ✅ Métricas avançadas (Sharpe, Sortino, Max DD)

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        POLY-BELTANE                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   DADOS      │    │   MODELO     │    │   RISCO      │
│              │    │              │    │              │
│ Wunderground │───▶│ LightGBM     │───▶│ Stop-Loss   │
│ Open-Meteo   │    │ XGBoost      │    │ Gatekeeper   │
│ Histórico    │    │ Z-Score      │    │ Position Mgr │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────┐
│                    ESTRATÉGIA                           │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Phased   │  │ Single   │  │  Causal Map     │    │
│  │ Entry    │  │ Entry    │  │  (Föhn detector) │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                  EXECUÇÃO (CLOB)                       │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐                     │
│  │ Order       │  │ Paper       │                     │
│  │ Executor    │  │ Trading     │                     │
│  └─────────────┘  └─────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### Componentes

| Componente | Ficheiro | Descrição |
|------------|-----------|-----------|
| **Modelo** | `munich_model.py` | Carregamento de modelos + predict_ensemble |
| **Treino** | `munich_train.py` | Walk-Forward Validation + treinamento |
| **Config** | `munich_strategy_config.py` | Configurações dinâmicas da estratégia |
| **Live Bot** | `munich_live_bot_unified.py` | Bot em tempo real |
| **Backtest** | `munich_backtester_unified.py` | Backtester completo |
| **Gatekeeper** | `munich_fuzzy_gatekeeper.py` | Validação de contexto |
| **Stop-Loss** | `munich_stop_loss.py` | Gestão de perdas |
| **Entry** | `munich_phased_entry.py` | Lógica de entrada |
| **Causal** | `munich_causal_features.py` | Fuzzy Cognitive Map (Föhn) |
| **Weather** | `munich_weather.py` | Dados meteorológicos |
| **Polymarket** | `polymarket_orders.py`, `polymarket_clob.py` | Integração CLOB |
| **Metrics** | `bet_metrics.py` | Métricas avançadas |
| **Optimizer** | `munich_optuna_optimizer.py` | Otimização Optuna |

## 📦 Instalação

### Pré-requisitos

- Python 3.12+
- Chave de API Wunderground
- Chave privada Polymarket (para trading real)

### Setup

```bash
# Clonar repositório
git clone <repo-url>
cd POLY-BELTANE

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com as suas chaves
```

### Dependências

```
lightgbm
xgboost
pandas
numpy
scikit-learn
optuna
requests
py-clob-client
rich
```

## ⚙️ Configuração

### Variáveis de Ambiente

```bash
# .env
WU_API_KEY=sua_chave_wunderground
POLY_PRIVATE_KEY=sua_chave_polymarket
POLY_MAX_DAILY_LOSS=100.0
```

### Configuração da Estratégia

```bash
# Criar config padrão
python munich_strategy_config.py create-default

# Ver config atual
python munich_strategy_config.py show
```

#### Estrutura da Config

```json
{
  "stop_loss": {
    "temp_threshold": 0.5,
    "prob_threshold": 0.60,
    "mode": "both"
  },
  "entry": {
    "mode": "single",
    "single_threshold": 0.80,
    "phased_parcel_size": 5.0
  },
  "position": {
    "bet_size": 15.0,
    "cooldown_minutes": 30.0,
    "max_daily_loss": 50.0
  },
  "gatekeeper": {
    "ev_min_threshold": 0.02,
    "zscore_min": 1.0,
    "market_volume_min": 100.0
  }
}
```

## 🧠 Treinamento

### Treinar Modelos

```bash
# Treinar com validação walk-forward
python munich_train.py

# Treinar para anos específicos
python munich_train.py --start-year 2020 --end-year 2023
```

### Saídas

- `models/munich_lgb.pkl` - Modelo LightGBM
- `models/munich_xgb.pkl` - Modelo XGBoost
- `models/munich_config.json` - Configuração dos modelos

## 📊 Backtesting

### Backtest Básico

```bash
# Backtest padrão (5 anos)
python munich_backtester_unified.py

# Com datas específicas
python munich_backtester_unified.py --start 2020-01-01 --end 2023-12-31

# Com modo phased
python munich_backtester_unified.py --mode phased

# Com config personalizada
python munich_backtester_unified.py --config optimized_config.json
```

### Backtest com Métricas Avançadas

```bash
python run_backtest_with_metrics.py --years 1 --mode single
python run_backtest_with_metrics.py --start 2020-01-01 --end 2023-12-31
```

### Comparar Configurações

```bash
python run_backtest_with_metrics.py --compare \
  strategy_configs/default_config.json \
  strategy_configs/optimized_config.json
```

### Saídas

- `backtest_results/metrics_*.json` - Resultados JSON
- `backtest_results/unified_*.json` - Backtest básico

## 🚀 Live Trading

### Modo Paper (Simulação)

```bash
# Modo paper padrão
python munich_live_bot_unified.py --run paper

# Com entrada phased
python munich_live_bot_unified.py --run paper --mode phased
```

### Modo Real

```bash
# Modo real (requer confirmação)
python munich_live_bot_unified.py --run real

# Auto-confirma
python munich_live_bot_unified.py --run real --yes
```

### Horário de Trading

- **Horário ativo**: 08:00 - 20:00 (Berlin time)
- **Fim do dia**: 18:00 (fuso local)
- **Mínimo para sinais**: 11:00

## 🎛️ Otimização

### Otimizar com Optuna

```bash
# Otimizar padrão (50 trials)
python munich_optuna_optimizer.py --trials 50

# Com objetivo específico
python munich_optuna_optimizer.py --trials 100 --objective sharpe

# Integrado com backtester
python munich_optuna_optimizer.py --trials 50 --backtest munich_backtester_unified
```

### Parâmetros Otimizados

- `temp_threshold` - Stop-loss por temperatura
- `prob_threshold` - Stop-loss por probabilidade
- `single_threshold` - Threshold de entrada single
- `p2_threshold`, `p3_threshold` - Thresholds phased
- `ev_min_threshold` - EV mínimo do gatekeeper
- `zscore_min` - Z-score mínimo do gatekeeper

## 📈 Métricas

### Métricas Calculadas

- **Total P&L** - Lucro/prejuízo total
- **ROI %** - Return on Investment
- **Win Rate %** - Taxa de acerto
- **Profit Factor** - Lucro total / Perda total
- **Payoff Ratio** - Win médio / Loss médio
- **Sharpe Ratio** - Retorno ajustado ao risco
- **Sortino Ratio** - Retorno ajustado ao downside
- **Max Drawdown** - Perda máxima
- **EV per Trade** - Expected Value médio

### Visualização

```python
from bet_metrics import BetMetrics

# Criar trades
trades = [
    {"pnl_usd": 15.0, "pnl_pct": 300.0, "outcome": "won", "ask": 0.25},
    {"pnl_usd": -5.0, "pnl_pct": -100.0, "outcome": "lost", "ask": 0.30},
]

# Calcular métricas
metrics = BetMetrics(trades)

# Imprimir relatório Rich
metrics.print_report()
```

## 🔧 Troubleshooting

### Erro: "Module not found"

```bash
# Reinstalar dependências
pip install -r requirements.txt --upgrade
```

### Erro: "Modelo não encontrado"

```bash
# Treinar modelo primeiro
python munich_train.py
```

### Erro: "Sem mercados encontrados"

- Verifique se a API Polymarket está acessível
- Verifique se há mercados ativos para hoje

### Debug Mode

```bash
# Ativar logging
export DEBUG=1

# Ver logs
tail -f logs/munich_live_bot.log
```

## 📁 Estrutura de Ficheiros

```
POLY-BELTANE/
├── models/                      # Modelos treinados
│   ├── munich_lgb.pkl
│   ├── munich_xgb.pkl
│   └── munich_config.json
├── strategy_configs/            # Configurações
│   ├── default_config.json
│   └── optimized_config.json
├── backtest_results/           # Resultados de backtest
├── logs/                       # Logs do bot
├── historic/                   # Dados históricos
│   └── munich.csv
├── munich_*.py                # Módulos principais
├── polymarket_*.py             # Integração Polymarket
├── bet_metrics.py              # Métricas
├── run_backtest_with_metrics.py
└── README.md
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit as suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

[MIT License](LICENSE)

## ⚠️ Disclaimer

Este software é fornecido para fins educacionais. Trading em mercados financeiros envolve risco significativo de perda. Use por sua conta e risco.
