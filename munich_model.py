"""
munich_model.py - BRANCH - main
===============
Carregamento do modelo LightGBM, construcao de features, predicao e
gestao do historico diario de maximas.

Exporta:
  load_model()
  build_features(slots_so_far, current, month, doy)
  predict_p(model, feat_cols, slots_so_far, current, month, doy)
  set_seasonal_prior(prior_map)
  get_seasonal_prior(month, hour, slot30)
  compute_prev7(history, d)
  init_history_max()
  save_history_max(history_max)
  update_history_max(history_max, slots_so_far)
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from munich_config import (
    MODEL_LGB, MODEL_CONFIG, FEATURE_COLS,
    MIN_HOUR, C, R, DIM,
    berlin_date, ceil_slot,
)

# ══════════════════════════════════════════════════════
#  SEASONAL PRIOR (estado global, inicializado pelo run())
# ══════════════════════════════════════════════════════
_SEASONAL_PRIOR: dict[tuple, float] = {}


def set_seasonal_prior(prior_map: dict) -> None:
    global _SEASONAL_PRIOR
    _SEASONAL_PRIOR = prior_map or {}


def get_seasonal_prior(month: int, hour: int, slot30: int) -> float:
    if _SEASONAL_PRIOR:
        return _SEASONAL_PRIOR.get((month, hour, slot30), 0.5)
    return 0.5


# ══════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════
def load_model():
    """
    Carrega o modelo LightGBM e a config associada.
    Devolve (model, feat_cols, prior_map, monthly_threshold).
    """
    if not MODEL_LGB.exists():
        raise FileNotFoundError(
            f"\n  {C['red']}Modelo nao encontrado: {MODEL_LGB}{R}\n"
            "  Corre: python munich_train.py"
        )
    model  = joblib.load(MODEL_LGB)
    config = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat   = config.get("feature_cols", FEATURE_COLS)

    # Prior sazonal — chave "month_hour_slot30"
    raw_prior = config.get("seasonal_peak_prior", {})
    prior_map: dict[tuple, float] = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    # Threshold adaptativo por mes — chave "1".."12"
    raw_thresh = config.get("monthly_threshold", {})
    monthly_threshold: dict[int, float] = {}
    for k, v in raw_thresh.items():
        try:
            monthly_threshold[int(k)] = float(v)
        except ValueError:
            pass

    thresh_str = (f"{len(monthly_threshold)} meses"
                  if monthly_threshold else f"{C['yellow']}nao disponivel{R}")
    print(f"  {C['green']}✓{R} LightGBM  AUC={config.get('global_auc', '?')}  "
          f"features={len(feat)}  "
          f"threshold_adaptativo={thresh_str}  "
          f"prior={'sim' if prior_map else C['yellow'] + 'nao' + R}")
    return model, feat, prior_map, monthly_threshold


# ══════════════════════════════════════════════════════
#  FEATURE BUILDER
# ══════════════════════════════════════════════════════
def build_features(slots_so_far: list[dict], current: dict,
                   month: int, doy: int, minute: int = 0) -> dict:
    """
    Constroi uma linha com as 15 features canonicas para o slot actual.
    slots_so_far: lista cronologica incluindo o slot actual como ultimo elemento.
    lag(1) = slot anterior (~30min), lag(3) = ~1.5h atras.
    """
    vals   = [s["temp_c"] for s in slots_so_far]
    n      = len(vals)
    cur    = vals[-1]
    rmax   = max(vals)
    hour   = current["hour"]
    slot30 = current.get("slot30", ceil_slot(hour, minute)[1])

    def lag(k): return vals[-k] if n >= k else vals[0]

    # Contexto matinal: slots ANTERIORES ao corrente com hora <= 12
    morn_vals = [s["temp_c"] for s in slots_so_far[:-1] if s["hour"] <= 12]
    mmax = max(morn_vals) if morn_vals else cur

    prev7     = current.get("prev_7d_avg_max", rmax)
    slot_frac = (hour + slot30 / 60.0) / 24.0

    return {
        "slot_frac":              slot_frac,
        "temp_c":                 cur,
        "running_max":            rmax,
        "pct_of_running_max":     cur / rmax if rmax else 1.0,
        "delta_30m":              cur - lag(2),
        "delta_1h":               cur - lag(3),
        "accel":                  (cur - lag(2)) - (lag(2) - lag(3)),
        "temp_lag_1":             lag(2),
        "temp_lag_3":             lag(4),
        "roll3_mean":             np.mean(vals[-3:]),
        "roll3_std":              np.std(vals[-3:]) if n >= 3 else 0.0,
        "morning_max":            mmax,
        "temp_above_morning_max": cur - mmax,
        "prev_7d_avg_max":        prev7,
        "seasonal_peak_prior":    get_seasonal_prior(month, hour, slot30),
    }


# ══════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════
def predict_p(model, feat_cols, slots_so_far: list[dict], current: dict,
              month: int, doy: int) -> float:
    """
    Devolve P(pico ja ocorreu) para o slot actual.
    Requer pelo menos 4 slots e hora >= MIN_HOUR.
    """
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0

    row   = build_features(slots_so_far, current, month, doy)
    avail = [f for f in feat_cols if f in row]
    X     = pd.DataFrame([row])[avail].fillna(0)

    # Booster.predict devolve probabilidade da classe positiva
    return float(model.predict(X)[0])


# ══════════════════════════════════════════════════════
#  PREV_7D_AVG_MAX
# ══════════════════════════════════════════════════════
def compute_prev7(history: dict, d: date) -> float:
    """
    history: {date: max_temp}
    d: dia actual
    Devolve media dos ultimos 7 dias (excluindo hoje).
    """
    days = sorted(history.keys())
    if d not in days:
        return None

    idx = days.index(d)
    if idx == 0:
        return history[d]

    window = days[max(0, idx - 7):idx]
    vals = [history[x] for x in window]
    return float(np.mean(vals)) if vals else history[d]


# ══════════════════════════════════════════════════════
#  HISTORICO DIARIO (prev_7d_avg_max)
# ══════════════════════════════════════════════════════
def init_history_max() -> dict:
    path = Path("live_history_max.json")
    if path.exists():
        try:
            return {date.fromisoformat(k): float(v)
                    for k, v in json.loads(path.read_text()).items()}
        except Exception:
            pass
    return {}


def save_history_max(history_max: dict) -> None:
    path = Path("live_history_max.json")
    data = {d.isoformat(): v for d, v in history_max.items()}
    path.write_text(json.dumps(data, indent=2))


def update_history_max(history_max: dict, slots_so_far: list[dict]) -> None:
    if not slots_so_far:
        return
    today = berlin_date()
    max_temp_today = max(s["temp_c"] for s in slots_so_far)
    history_max[today] = max_temp_today
    save_history_max(history_max)
