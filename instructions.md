# ARQUITECTURA POLY-BELTANE (UNIFIED2)

> Como as várias partes do sistema se relacionam e o que fazem

---

## 🧠 MODELO (O Cérebro)

O modelo é um ensemble ponderado de três componentes:

| Componente | Peso |
|------------|------|
| LightGBM (LGBM) | 50% |
| XGBoost (XGB) | 30% |
| Z-Score | 20% |

### `munich_train.py`
- Carrega `historic/munich.csv`
- Walk-Forward Validation (expanding window)
- Treina LGBM + XGB final em todos os dados
- **NÃO** usa NEAT, LSTMs ou redes neuronais
- Guarda modelos (`.pkl`) + `config.json`

> ⚠️ **NÃO MEXER em `munich_train.py` ou `munich_model.py`**

---

## ⚙️ GESTÃO DE EXECUÇÃO

### Stop-Loss (Cinto de Segurança)

Exemplo de funcionamento:
1. 14h00: "Comprei UNDER 10ºC"
2. O modelo diz: `P(peak) = 85%`
3. A temperatura sobe para 10.1ºC
4. O Stop-Loss monitoriza a temperatura
5. Se subir para 11ºC → **vende a posição**

📄 **Ficheiro:** `munich_live_bot.py` → Função: `check_stop_loss`

### Porteiro (Fuzzy Gatekeeper)

Verifica o **CONTEXTO** antes de enviar ordens à Polymarket.

### Simulação

Usa **Optuna** para encontrar os melhores thresholds automaticamente.

---

## 🚦 GESTÃO DE CARACTERÍSTICAS (O Porteiro)

### Fuzzy Gatekeeper

Entrada **apenas se**:
1. EV é positivo
2. Forecast concorda
3. Mercado não está "arriscado"
4. Z-Score confirma

> Se o Fuzzy disser **NÃO** (estado = "arriscado"), o modelo ignora e a trade **NÃO** é executada.

📄 **Ficheiro a criar:** `munich_fuzzy_gatekeeper.py`

### Optuna (Otimização Automática)

Otimiza automaticamente:
- `SL_TEMP_THRESHOLD`
- `THRESHOLD`
- `P1/P2/P3 thresholds`
- `cooldown_minutes`

---

## 🔗 ANÁLISE CAUSAL — Opcional (FCMs)

### Fuzzy Cognitive Map

Mapeamento de relações causais:

```
Humidade a baixar  ──────────► Previsão de pico desce
Pressão a subir    ──────────► Z-Score desce
Föhn Activo        ──────────► [REDUZIR confiança no modelo]
```

> **Nota:** NÃO altera a previsão do modelo.  
> Serve para **adicionar features** se quiseres expandir o modelo (ex: `Föhn_indicator`).

📄 **Ficheiro a criar:** `munich_causal_features.py`

---

## 📊 VISUALIZAÇÃO

### Rich Dashboard — `bet_metrics.py`

- Tabela de KPIs
- Painel de Risco
- Formatado com cores (Rich)

> ⚠️ **NÃO mexer a matemática.**  
> A matemática vem do modelo e do ficheiro `bet_metrics.py` (Sharpe, Sortino, MDD).

---

## 🔄 FLUXO DE EXECUÇÃO

Em caso de erros de mercado (`munich_live_bot.py`):

- Loop de leitura contínua
- Stop-Loss por temperatura
- Daily Stop-Loss
- Manual Override
- Recuperação de posição perdida (se não houver mercado de saída)

---

## 📁 ESTADO DOS FICHEIROS

| Ficheiro | Bloco na Arquitectura | Estado | Criar? |
|----------|-----------------------|--------|--------|
| `munich_train.py` | Modelo | ✅ Finalizado | ❌ Já existe |
| `munich_model.py` | Modelo | ✅ Finalizado | ❌ Já existe |
| `munich_phased_entry.py` | Porteiro (P1/P2/P3) | ✅ Finalizado | ❌ Já existe |
| `bet_metrics.py` | Visualização | ✅ Acabado na UNIFIED2 | ❌ Já existe (atualizado) |
| `munich_live_bot.py` | Execução + Stop-Loss | ✅ Acabado na UNIFIED2 | ❌ Já existe (atualizado) |
| `run_backtest_with_metrics.py` | Simulação + Comparação | ⏳ Precisa ser criado | ✅ **CRIAR** |
| `munich_fuzzy_gatekeeper.py` | Porteiro (Fuzzy) | ⏳ Precisa ser criado | ✅ **CRIAR** |
| `munich_causal_features.py` | Causalidade (FCMs) | ⏳ Opcional | ✅ OPCIONAL |
| `polymarket_orders.py` | CLOB / Execution | ✅ Finalizado | ❌ Já existe |
| `munich_weather.py` | Dados WU | ✅ Finalizado | ❌ Já existe |
| `munich_config.py` | Configuração Global | ✅ Finalizado | ❌ Já existe |

---

## 💬 O que faz cada ficheiro (em linguagem simples)

- **`munich_fuzzy_gatekeeper.py`** → O bot quer comprar. O Fuzzy diz: *"Espera, mercado arriscado. Não comprar."*
- **`munich_causal_features.py`** → Se souberes que *"O Föhn está ativo"*, pode alterar a forma como o modelo vê a realidade.

---

> ⛔ **REGRA DE OURO: NÃO MEXER em `munich_train.py` ou `munich_model.py`.**
