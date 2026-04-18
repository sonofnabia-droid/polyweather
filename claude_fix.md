# claude-fix.md
> Instruções de limpeza e correção do projeto Munich.
> Gerado em 2026-04-18. Executar com Claude Code.

---

## Contexto

O projeto tem 3 gerações de ficheiros paralelos que foram unificados.
Os ficheiros `_unified` são o destino final. Este documento lista:
1. Ficheiros a **apagar**
2. **Bugs** a corrigir nos ficheiros unificados
3. **Limpeza** de código morto

---

## 1. Ficheiros a apagar

Apagar os seguintes ficheiros — a sua lógica está completamente coberta
pelas versões `_unified` ou é incompatível com o stack atual (Betfair).

```
munich_backtester.py
munich_backtester_with_stop_loss.py
munich_live_bot.py
munich_live_bot_with_stop_loss.py
```

> **Nota `munich_live_bot_with_stop_loss.py`:** usa `betfairlightweight` e
> `config.py` (não `munich_config.py`). O projeto usa exclusivamente
> Polymarket para trading. Este ficheiro não partilha nenhum módulo com
> o stack atual e deve ser removido.

---

## 2. Bug — `munich_backtester_unified.py`

### 2a. Typo `wind_speed_kpa` → `wind_speed_kmh`

**Severidade:** Alta — a feature `wind_speed_kmh` é sempre carregada com
o valor default `5` em vez dos dados reais do CSV.

**Localizar** (linha ~353, dentro de `load_data()`):
```python
df["wind_speed_kmh"] = pd.to_numeric(df.get("wind_speed_kpa", 5), errors="coerce")
```

**Corrigir para:**
```python
df["wind_speed_kmh"] = pd.to_numeric(df.get("wind_speed_kmh", 5), errors="coerce")
```

---

### 2b. `uv_index` hardcoded em vez de vir do row

**Severidade:** Baixa/média — o modelo foi treinado com `uv_index` real
do CSV mas o backtester passa sempre `3.0`.

**Localizar** dentro de `_run_day()`, no dict `current` (linha ~494):
```python
"uv_index": 3.0,
```

**Corrigir para** (o row já tem a coluna após `load_data()`):
```python
"uv_index": float(row.get("uv_index", 3.0)),
```

E garantir que `load_data()` carrega a coluna `uv_index` do CSV,
a seguir às outras colunas V2 (adicionar se não existir):
```python
df["uv_index"] = pd.to_numeric(df.get("uv_index", 3.0), errors="coerce").fillna(3.0)
```

---

## 3. Limpeza — `munich_model.py`

### 3a. Remover bloco `monthly_threshold` (código morto)

`munich_train.py` nunca guarda `monthly_threshold` no JSON —
guarda `doy_poly_coeffs`. O bloco abaixo carrega sempre um dict vazio
e só aparece no log de arranque como fallback que nunca é atingido.

**Localizar e remover** (linhas ~91-98):
```python
# Threshold adaptativo por mes — chave "1".."12"
raw_thresh = config.get("monthly_threshold", {})
monthly_threshold: dict[int, float] = {}
for k, v in raw_thresh.items():
    try:
        monthly_threshold[int(k)] = float(v)
    except ValueError:
        pass
```

**Remover também** do dict de retorno (linha ~146):
```python
"monthly_threshold":  monthly_threshold,
```

**Remover também** da função `load_model()` (retrocompat), o 4º elemento
do tuple de retorno:
```python
return (
    result["model_lgb"],
    result["feat_cols"],
    result["prior_map"],
    result["monthly_threshold"],   # ← remover esta linha
)
```

> **Atenção:** verificar se `monthly_threshold` é usado em
> `munich_backtester.py` (versão antiga, a apagar) ou nalgum outro
> ficheiro antes de remover. Nos ficheiros `_unified` não é referenciado.

---

## 4. Verificação final após aplicar as correções

Depois de fazer as alterações acima, confirmar:

- [ ] `python munich_train.py --no-wf` corre sem erros
- [ ] `python munich_backtester_unified.py --silent` corre sem erros
- [ ] `grep -r "monthly_threshold" .` não retorna resultados nos ficheiros `_unified`
- [ ] `grep -r "wind_speed_kpa" .` não retorna resultados
- [ ] `grep -r "betfairlightweight" .` não retorna resultados
- [ ] `grep -r "from config import" .` não retorna resultados
  (deve ser `from munich_config import` em todo o lado)

---

## 5. Nota sobre `munich_train.py` (standalone)

O `munich_train.py` tem código de feature engineering duplicado
em relação ao `munich_model.py` (`build_features`). Isto é intencional
por agora — o train é standalone e não depende de `munich_model.py`.
**Não alterar** neste momento. Se a lógica de features mudar no futuro,
atualizar os dois ficheiros em simultâneo.
