# Munich Max Temp Bot - V4

Bot de trading automatizado para mercados de temperatura máxima em Munique, Alemanha, na plataforma Polymarket.

## Visão Geral

O bot prevê quando a temperatura máxima diária será atingida e executa operações de compra nos contratos correspondentes no Polymarket. Utiliza um ensemble de modelos de machine learning (LightGBM + XGBoost + Z-Score) treinados em 16 anos de dados históricos (2010-2026).

**Versão**: V4  
**Features**: 25 (18 V1 + 7 V2)  
**AUC Ensemble**: 0.967  
**Accuracy Single**: 90.9%  
**Accuracy Phased**: 85.2%

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Wunderground │  │ Open-Meteo   │  │ Polymarket   │    │
│  │ (observações)│  │ (forecast)   │  │ (CLOB API)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ munich_weather.py - Fetch e parsing de dados WU/OM  │  │
│  │ munich_model.py   - Feature engineering (25 feat)   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Engine                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Ensemble: LGBM (50%) + XGB (30%) + Z-Score (20%)    │  │
│  │ ─────────────────────────────────────────────────── │  │
│  │ - predict_ensemble() - p(pico já ocorreu)          │  │
│  │ - StreamingPeakDetector - Detecção de pico         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Entry Logic                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ munich_phased_entry.py                                │  │
│  │   - PhasedEntry: 3 parcelas ($5 cada)              │  │
│  │   - SingleEntry: 1 compra ($15) ← MODO PADRÃO     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Execution Layer                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ munich_live_bot.py  - Loop principal                │  │
│  │ polymarket_clob.py    - Integração CLOB             │  │
│  │ polymarket_orders.py  - Execução de ordens          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring & Alerts                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ munich_display.py   - Terminal UI                  │  │
│  │ tg.py                - Telegram alerts               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Ficheiros Principais

### Core Trading

#### `munich_live_bot.py`
Bot principal que executa o loop de trading em tempo real.

**Funções principais:**
- `run()` - Loop principal de trading
- `get_real_usdc_balance()` - Consulta saldo USDC em modo REAL
- `fetch_market()` - Busca mercado Polymarket
- `compute_ev()` - Calcula Expected Value

**Fluxo de execução:**
1. Carrega modelos
2. Bootstrap de dados históricos do dia
3. Aplica modelo aos slots históricos
4. Carrega mercado Polymarket
5. Loop:
   - Fetch WU latest observation
   - Atualiza running max
   - Calcula ensemble prediction
   - Avalia lógica de entrada (SINGLE/PHASED)
   - Executa ordens se condições satisfeitas
   - Atualiza display e Telegram
   - Sleep smart

**Argumentos:**
```bash
python munich_live_bot.py                           # Interativo (pergunta tudo)
python munich_live_bot.py --mode single --run paper  # Paper single
python munich_live_bot.py --mode single --run real --yes  # Real headless
```

---

#### `munich_phased_entry.py`
Define a lógica de entrada para compra de contratos.

**Classes:**

**SingleEntry** (MODO PADRÃO)
- Compra única quando `p_ensemble >= threshold` (default 75%)
- Simples, sem dependências de forecasts
- Recomendado para trading consistente

```python
entry = SingleEntry(parcel_size=15.0, threshold=0.75)
actions = entry.evaluate(p_ensemble, hour, market, rmax, forecast_agreement)
```

**PhasedEntry** (Avançado)
- **P1 (Value Early)**: 10h-12h, forecast agree, mercado confirma, p em 30-65%
- **P2 (Dupla Confirmação)**: p >= 70%, mercado confirma
- **P3 (Alta Confiança)**: p >= 85%

**Thresholds V4:**
- P1_min: 30%, P1_max: 65%
- P2: 70% (aumentado de 60%)
- P3: 85% (aumentado de 80%)

---

### Machine Learning

#### `munich_model.py`
Modelos de ML e feature engineering.

**Funções principais:**
- `load_models()` - Carrega modelos treinados (LGBM, XGB, configs)
- `build_features()` - Cria 25 features para um slot
- `predict_ensemble()` - Predição ensemble (LGBM + XGB + Z-Score)
- `StreamingPeakDetector` - Detetor de pico baseado em Z-Score

**Features (25 total):**

**V1 (18 features):**
1. `slot_frac` - Fração do dia (0-1)
2. `doy_sin`, `doy_cos` - Ciclo anual
3. `temp_c` - Temperatura atual
4. `running_max` - Máximo até agora
5. `temp_vs_climatology` - Diferença para média de 7 dias
6. `delta_30m`, `delta_1h` - Mudanças de temperatura
7. `accel` - Aceleração da temperatura
8. `recent_slope` - Tendência recente
9. `temp_lag_3` - Temperatura há 3 slots
10. `roll3_std` - Desvio padrão de 3 slots
11. `plateau_indicator` - Indicador de estagnação
12. `morning_max` - Máximo da manhã
13. `radiation_proxy` - Proxy de radiação solar
14. `humidity_drop_1h` - Queda de humidade
15. `prev_7d_avg_max` - Média de 7 dias
16. `seasonal_peak_prior` - Prior sazonal do pico
17. `hour` - Hora do dia (implicita em slot_frac)
18. `doy` - Dia do ano (implicito em doy_sin/cos)

**V2 (7 features):**
19. `dewpoint_c` - Ponto de orvalho
20. `temp_to_dewpoint_gap` - Gap temp-dewpoint
21. `pressure_trend_3h` - Tendência de pressão (3h)
22. `wind_south_proxy` - Proxy de vento sul
23. `wind_speed_kmh` - Velocidade do vento
24. `uv_index` - Índice UV
25. `foehn_indicator` - Indicador de vento foehn

**Ensemble weights:**
- LightGBM: 50%
- XGBoost: 30%
- Z-Score: 20%

---

#### `munich_train.py`
Script para treinar modelos de ML.

**Pipeline:**
1. Carrega dados históricos (2010+) do CSV
2. Constrói features slot a slot (30min CEILING)
3. Walk-Forward Validation (expanding window por ano)
4. Treina LightGBM + XGBoost em paralelo
5. Calcula threshold adaptativo (curva DOY contínua)
6. Salva modelos + configuração

**Uso:**
```bash
python munich_train.py
```

**Outputs:**
- `munich_peak_model/lgbm_peak.pkl` - Modelo LightGBM
- `munich_peak_model/xgb_peak.pkl` - Modelo XGBoost
- `munich_peak_model/peak_model_config.json` - Configuração + feature importances

---

### Backtesting & Otimização

#### `munich_backtester.py`
Backtest completo para validar estratégias.

**Funções principais:**
- `run()` - Executa backtest em dados históricos
- `SimulatedMarket` - Simula mercado Polymarket
- `compute_metrics()` - Calcula métricas de performance
- `plot()` - Gera gráficos de resultados

**Métricas:**
- Correct: % de dias com deteção correta
- Premature: % de dias com deteção prematura
- Missed: % de dias sem deteção
- Lag médio: Tempo médio após o pico real
- Lag ≤ 1h: % de deteções dentro de 1h
- Lag ≤ 2h: % de deteções dentro de 2h

**Uso:**
```bash
python munich_backtester.py --mode single  # Interativo
echo "2010-01-01" | python munich_backtester.py --mode single  # Auto
```

---

#### `optimizer.py` / `optimizer_fast.py`
Scripts para otimizar thresholds de entrada.

**Testa combinações de:**
- SINGLE: [65%, 70%, 75%, 80%, 85%]
- PHASED: P1_min [20-40%], P1_max [60-70%], P2 [65-80%], P3 [80-90%]

**Uso:**
```bash
python optimizer.py --start 2024-01-01 --end 2024-12-31
python optimizer_fast.py --single  # Apenas SINGLE, mais rápido
```

---

### Integrações

#### `munich_weather.py`
Acesso a dados meteorológicos via Wunderground + Open-Meteo.

**Funções:**
- `make_wu_session()` / `make_om_session()` - Sessões HTTP
- `fetch_wu_day_eddm()` - Dados históricos WU (Munique EDDM)
- `fetch_wu_latest()` - Última observação WU
- `fetch_wu_forecast_max()` - Previsão WU
- `fetch_om_forecast_max()` - Previsão Open-Meteo
- `fetch_om_hourly_today()` - Previsão horária OM
- `bootstrap_today()` - Dados do dia para inicializar
- `forecasts_agree()` - Verifica concordância WU/OM

**Dados WU incluem V2 features:**
- dewpoint_c, pressure_hpa, wind_dir_deg, wind_speed_kmh, wind_gust_kmh, uv_index

---

#### `polymarket_clob.py`
Integração com CLOB (Central Order Book) do Polymarket.

**Classes:**
- `ClobClient` - Cliente CLOB
- `OrderBook` - Livro de ordens
- `PositionManager` - Gestão de posições
- `Position` - Representa uma posição
- `PositionStatus` - Estado (open, won, lost)

---

#### `polymarket_orders.py`
Execução de ordens no Polymarket.

**Classes:**
- `OrderExecutor` - Executa ordens reais
- `paper_buy()` - Simula compra em modo paper

---

### Utilitários

#### `munich_config.py`
Configurações globais e constantes.

**Variáveis principais:**
- `WU_API_KEY`, `POLY_PRIVATE_KEY` - Chaves de API
- `POLY_MAX_DAILY_LOSS` - Limite de perda diária ($50)
- `DAY_START=6`, `DAY_END=21` - Horas de operação
- `MONTH_NAMES`, `SEASONS` - Constantes de tempo
- `berlin_date()`, `berlin_now()` - Funções de tempo Berlim

---

#### `munich_display.py`
UI de terminal para exibir status do bot.

**Funções:**
- `display()` - Dashboard principal
- `display_positions()` - Exibe posições
- `draw_chart()` - Desenha gráfico ASCII da temperatura
- `p_col()` - Cores baseadas em probabilidade

---

#### `tg.py`
Envio de alertas via Telegram.

**Métodos:**
- `alert_started()` - Início do bot
- `alert_no_market()` - Mercado não encontrado
- `alert_peak_detected()` - Pico detectado + ordem
- `alert_order_placed()` - Ordem colocada
- `alert_order_failed()` - Ordem falhou
- `alert_bet_blocked()` - Ordem bloqueada
- `alert_stopped()` - Bot parado
- `dashboard()` - Dashboard periódico

---

## Configuração

### Variáveis de Ambiente

```bash
export WU_API_KEY="tua_chave_wunderground"
export POLY_PRIVATE_KEY="0x...tua_private_key_ethereum"
```

### Estrutura de Diretórios

```
polyweather/
├── historic/
│   └── munich.csv           # Dados históricos (2010+)
├── munich_peak_model/
│   ├── lgbm_peak.pkl        # Modelo LightGBM
│   ├── xgb_peak.pkl         # Modelo XGBoost
│   └── peak_model_config.json # Configuração
├── logs/
│   ├── live_YYYY-MM-DD.csv  # Logs diários
│   └── bets_YYYY-MM-DD.json # Registo de apostas
└── backtest_results/
    └── *.png                 # Gráficos de backtest
```

## Modos de Operação

### Modos de Entrada

**SINGLE** (Recomendado)
- 1 compra de $15 quando `p_ensemble >= 75%`
- Simples, sem dependências de forecasts
- 90.9% correct, 9.1% prematuro (2010-2026)

**PHASED** (Avançado)
- 3 compras de $5 em diferentes thresholds
- P1: 10h-12h, forecast agree, p 30-65%
- P2: p >= 70%, mercado confirma
- P3: p >= 85%
- 85.2% correct, 12.7% prematuro (2010-2026)

### Modos de Trading

**PAPER**
- Simula ordens sem executar
- Usa order book real
- Sem riscos financeiros

**REAL**
- Executa ordens reais no Polymarket
- Requer `POLY_PRIVATE_KEY`
- Limite de perda diária: $50

## Fluxo de Dados

### 1. Coleta de Dados
```
Wunderground API → _wu_parse_obs() → slots_so_far[]
Open-Meteo API → fetch_om_forecast_max() → om_forecast{}
```

### 2. Feature Engineering
```
slots_so_far[] + current[] → build_features() → row{25 features}
```

### 3. Predição
```
row{25 features} → predict_ensemble() → p_ensemble (0.0-1.0)
```

### 4. Decisão de Entrada
```
p_ensemble + hour + market → PhasedEntry/SingleEntry.evaluate() → actions[]
```

### 5. Execução
```
actions[] + target_bracket → OrderExecutor.buy() → ordem no Polymarket
```

## Treino de Modelos

### Requisitos
- Dados históricos em `historic/munich.csv`
- 18 features básicas + 7 features V2
- Walk-Forward Validation (expanding window)

### Comando
```bash
python munich_train.py
```

### Resultado Esperado
- AUC > 0.96
- Modelos salvos em `munich_peak_model/`
- Configuração com feature importances

## Backtesting

### Histórico Disponível
- 2010-2026: ~9,168 dias
- ~189,000 slots de 30min

### Comandos
```bash
# Interativo
python munich_backtester.py

# Automático
echo "2010-01-01" | python munich_backtester.py --mode single
echo "2010-01-01" | python munich_backtester.py --mode phased
```

### Resultados 2010-2026

**SINGLE:**
- Correct: 90.9%
- Premature: 9.1%
- Missed: 0.0%
- Lag médio: +1.68h
- Investimento médio: $15/dia

**PHASED:**
- Correct: 85.2%
- Premature: 12.7%
- Missed: 2.0%
- Lag médio: +1.92h
- Investimento médio: $9.13/dia

## Depuração e Logs

### Logs Diários
- `logs/live_YYYY-MM-DD.csv` - Logs de ticks
- `logs/bets_YYYY-MM-DD.json` - Registo de apostas

### Mensagens de Debug
- WU forecast errors
- Saldo updates
- Blocked bet reasons

### Telegram Alerts
- Pico detectado
- Ordem colocada
- Ordem falhou
- Bot parado
- Dashboard periódico (30 min)

## Troubleshooting

### Erro: "The number of features in data (18) is not the same as it was in training data (25)"
**Causa:** Código desatualizado vs modelos novos
**Solução:** `git pull origin INTEGRATION`

### Erro: "AttributeError: 'SingleEntry' object has no attribute '_market_confirms_model'"
**Causa:** Display incompatível com SINGLE mode
**Solução:** `git pull origin INTEGRATION`

### Erro: "wu_missing" em Previsão Dual
**Causa:** API Wunderground retornou erro
**Solução:** Verificar `WU_API_KEY` ou usar apenas Open-Meteo

## Roadmap

### V4 (Atual)
- [x] 7 features V2 adicionadas
- [x] Thresholds PHASED otimizados
- [x] SINGLE como modo default
- [x] Backtest 2010-2026 completo

### Próximas versões
- [ ] Integração com mais fontes de forecast
- [ ] Otimização dinâmica de position sizing
- [ ] Machine Learning para threshold adaptativo
- [ ] Dashboard web

## Licença

Uso pessoal. Respeitar termos de uso do Polymarket e APIs de meteorologia.

## Contacto

Para questões ou suporte, abrir issue no repositório.
