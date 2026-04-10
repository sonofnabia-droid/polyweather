┌─────────────────────────────────────────────────────────────────┐
│                    MUNICH LIVE BOT V3                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WEATHER SOURCES                                                │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ WUnderground │  │ Open-Meteo   │                            │
│  │ (EDDM obs)   │  │ (forecast)   │                            │
│  └──────┬───────┘  └──────┬───────┘                            │
│         │                 │                                      │
│         └────────┬────────┘                                      │
│                  ▼                                               │
│         forecasts_agree()                                        │
│         (tolerance: 2°C)                                         │
│                  │                                               │
│                  ▼                                               │
│  MODELS (Ensemble)                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                 │
│  │ LightGBM │  │ XGBoost  │  │ Z-Score      │                 │
│  │  (50%)   │  │  (30%)   │  │ Streaming(20%)│                 │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘                 │
│       └──────────────┼───────────────┘                          │
│                      ▼                                          │
│              p_ensemble                                          │
│                      │                                          │
│                      ▼                                          │
│  ═══════════════════════════════════════════════                │
│  DUPLA CONFIRMAÇÃO (Modelo + Mercado)                          │
│  ═══════════════════════════════════════════════                │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐           │
│  │ CONDIÇÃO 1: MODELO  │    │ CONDIÇÃO 2: MERCADO │           │
│  │                     │    │                     │           │
│  │ P_ensemble >= THR   │    │ Spread < 5¢        │           │
│  │ OU                  │ E  │ Ask_depth > $50     │           │
│  │ Forecast agreement  │    │ Ask < 0.95          │           │
│  │ (para Parcel 1)     │    │                     │           │
│  └─────────┬───────────┘    └─────────┬───────────┘           │
│            │                          │                         │
│            └────────────┬─────────────┘                         │
│                         ▼                                       │
│  ═══════════════════════════════════════════════                │
│  ENTRY: 3 PARCELAS (sizing faseado)                            │
│  ═══════════════════════════════════════════════                │
│                                                                 │
│  PARCELA 1 (Manhã cedo):                                       │
│    • Hora < 12h                                                │
│    • forecasts_agree() = True                                  │
│    • Mercado liquido (spread + depth)                          │
│    • Size: 30% do risk_per_trade                               │
│    → "Aposta no value matinal antes do modelo ter confiança"   │
│                                                                 │
│  PARCELA 2 (Pico Aproximando):                                 │
│    • p_ensemble >= threshold (ex: 0.60)                        │
│    • Mercado liquido                                           │
│    • Size: 40% do risk_per_trade                               │
│    → "Modelo ganhou confiança"                                 │
│                                                                 │
│  PARCELA 3 (Pico Confirmado):                                  │
│    • p_ensemble >= threshold_alto (ex: 0.80)                   │
│    • Mercado NÃO está resolvido (ask < 0.90)                   │
│    • Size: 30% do risk_per_trade                               │
│    → "Confirmação forte + ainda há edge"                       │
│                                                                 │
│  ═══════════════════════════════════════════════                │
│  DISPLAYS (Terminal + Telegram)                                 │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DUPLA CONFIRMAÇÃO                                              │
│                                                                 │
│  MODELO diz: "pico em X°C" (running max)                       │
│  MERCADO diz: "probabilidade maior em Y°C" (maior ask)         │
│                                                                 │
│  Se X e Y concordam → CONFIRMADO                               │
│                                                                 │
│  Exemplo:                                                       │
│    12°C → ask 10¢                                               │
│    13°C → ask 20¢  ← MERCADO escolhe este (maior ask)          │
│    14°C → ask 5¢                                                │
│                                                                 │
│    Running max = 13°C → MODELO concorda com MERCADO → ✅        │
│    Running max = 14°C → MODELO discorda do MERCADO → ❌         │
│                                                                 │
│  PARCELA 1 (Manhã cedo):                                       │
│    • Hora < 12h                                                 │
│    • forecasts_agree() = True                                   │
│    • Size: $5                                                   │
│                                                                 │
│  PARCELA 2 (Dupla confirmação):                                │
│    • p_ensemble >= threshold                                    │
│    • Bracket com maior ask = bracket do running max             │
│    • Size: $5                                                   │
│                                                                 │
│  PARCELA 3 (Alta confiança):                                   │
│    • p_ensemble >= 0.80                                         │
│    • Size: $5                                                   │
│                                                                 │
│  TELEGRAM (pico detectado):                                     │
│    🔔 PICO DETECTADO                                            │
│      Ensemble: 82.3%                                            │
│      LGBM: 85.1% | XGB: 78.4% | Z-Score: 79.2%               │
│      Mercado: 13°C (ask 20¢) ← highest                        │
└─────────────────────────────────────────────────────────────────┘
