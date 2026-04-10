"""
munich_model.py
===============
Carregamento de modelos (LightGBM + XGBoost), construção de features
ALINHADAS com V1, predição ensemble, z-score streaming, histórico.

Exporta:
  load_models()                    — carrega todos os modelos disponíveis
  build_features(...)              — 18 features V1 (canónicas)
  predict_p(...)                   — LightGBM only (retrocompatibilidade)
  predict_ensemble(...)            — ensemble LGBM + XGB + z-score
  StreamingPeakDetector            — z-score streaming (sem treino)
  set_seasonal_prior / get_seasonal_prior
  compute_prev7 / init_history_max / save_history_max / update_history_max
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from munich_config import (
    MODEL_LGB, MODEL_XGB, MODEL_CONFIG, FEATURE_COLS,
    ENSEMBLE_WEIGHTS,
    MIN_HOUR, C, R, DIM,
    berlin_date, ceil_slot,
)


# ══════════════════════════════════════════════════════
#  SEASONAL PRIOR
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
#  LOAD MODELS — suporta LightGBM + XGBoost
# ══════════════════════════════════════════════════════
def load_models() -> dict:
    """
    Carrega todos os modelos disponíveis.
    Retorna dict com:
      model_lgb, model_xgb, feat_cols, prior_map, monthly_threshold, doy_poly
    """
    if not MODEL_LGB.exists():
        raise FileNotFoundError(
            f"\n  {C['red']}Modelo nao encontrado: {MODEL_LGB}{R}\n"
            "  Corre: python munich_train.py"
        )

    # ── LightGBM ──────────────────────────────────────
    model_lgb = joblib.load(MODEL_LGB)
    config    = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat_cols = config.get("feature_cols", FEATURE_COLS)

    # Prior sazonal
    raw_prior = config.get("seasonal_peak_prior", {})
    prior_map: dict[tuple, float] = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    # Monthly threshold
    raw_thresh = config.get("monthly_threshold", {})
    monthly_threshold: dict[int, float] = {}
    for k, v in raw_thresh.items():
        try:
            monthly_threshold[int(k)] = float(v)
        except ValueError:
            pass

    # Curva DOY contínua
    doy_poly_raw = config.get("doy_poly_coeffs")
    doy_poly = np.array(doy_poly_raw, dtype=float) if doy_poly_raw else None

    # ── XGBoost (opcional) ────────────────────────────
    model_xgb = None
    if MODEL_XGB.exists():
        try:
            model_xgb = joblib.load(MODEL_XGB)
            print(f"  {C['green']}✓{R} XGBoost  features={len(feat_cols)}")
        except Exception as e:
            print(f"  {C['yellow']}⚠ XGBoost load falhou: {e}{R}")

    # ── Ensemble weights da config ────────────────────
    saved_weights = config.get("ensemble_weights", {})
    weights = {k: saved_weights.get(k, v) for k, v in ENSEMBLE_WEIGHTS.items()}

    # Normalizar pesos se XGBoost não disponível
    if model_xgb is None:
        # Redistribuir peso do XGBoost para LightGBM e z-score
        xgb_w = weights.get("xgb", 0.3)
        weights["xgb"] = 0.0
        weights["lgbm"] += xgb_w * 0.6
        weights["zscore"] += xgb_w * 0.4

    # Info
    auc = config.get("global_auc", "?")
    if doy_poly is not None:
        thr_str = f"curva DOY grau {len(doy_poly)-1}"
    elif monthly_threshold:
        thr_str = f"{len(monthly_threshold)} meses"
    else:
        thr_str = f"{C['yellow']}nao disponivel{R}"

    models_str = f"LightGBM"
    if model_xgb:
        models_str += f" + XGBoost"
    models_str += f" + Z-Score"

    print(f"  {C['green']}✓{R} {models_str}  AUC={auc}  "
          f"features={len(feat_cols)}  "
          f"threshold={thr_str}  "
          f"prior={'sim' if prior_map else C['yellow']+'nao'+R}  "
          f"weights=[LGBM {weights['lgbm']:.0%} XGB {weights['xgb']:.0%} Z {weights['zscore']:.0%}]")

    return {
        "model_lgb":          model_lgb,
        "model_xgb":          model_xgb,
        "feat_cols":          feat_cols,
        "prior_map":          prior_map,
        "monthly_threshold":  monthly_threshold,
        "doy_poly":           doy_poly,
        "ensemble_weights":   weights,
    }


def load_model():
    """Retrocompatibilidade — retorna (model_lgb, feat_cols, prior_map, monthly_threshold)."""
    result = load_models()
    return (
        result["model_lgb"],
        result["feat_cols"],
        result["prior_map"],
        result["monthly_threshold"],
    )


# ══════════════════════════════════════════════════════
#  FEATURE BUILDER — 18 FEATURES V1 (CANÓNICAS)
# ══════════════════════════════════════════════════════
def build_features(slots_so_far: list[dict], current: dict,
                   month: int, doy: int, minute: int = 0) -> dict:
    """
    Constrói as 18 features V1 canónicas — IDÊNTICAS ao treino/backtest.

    IMPORTANTE: Qualquer alteração aqui implica re-treino completo.
    """
    vals  = [s["temp_c"] for s in slots_so_far]
    hums  = [s.get("humidity", 70) for s in slots_so_far]
    n     = len(vals)
    cur   = vals[-1]
    rmax  = max(vals)
    hour  = current["hour"]
    slot30 = current.get("slot30", ceil_slot(hour, minute)[1])
    cloud = float(current.get("cloud_cover", 50))

    def lag(k):
        return vals[-k] if n >= k else vals[0]

    def lagh(k):
        return hums[-k] if n >= k else hums[0]

    # Contexto matinal
    morn_vals = [s["temp_c"] for s in slots_so_far[:-1] if s["hour"] <= 12]
    mmax = max(morn_vals) if morn_vals else cur

    prev7     = current.get("prev_7d_avg_max", rmax)
    slot_frac = (hour + slot30 / 60.0) / 24.0

    # recent_slope: OLS sobre últimos 4 slots
    slope_window = vals[-4:] if n >= 4 else vals
    slope = _ols_slope(slope_window)

    # plateau_indicator: std dos últimos 6 slots < 0.4°C
    plat_window = vals[-6:] if n >= 6 else vals
    plateau = 1.0 if (np.std(plat_window) < 0.4 and n >= 4) else 0.0

    # radiation_proxy: posição solar × (1 − cloud cover)
    radiation = float(np.cos((slot_frac - 0.5) * 2 * np.pi)) * (1 - cloud / 100)

    # humidity_drop_1h: queda de humidade = instabilidade convectiva
    hum_drop_1h = lagh(3) - hums[-1] if n >= 3 else 0.0

    return {
        "slot_frac":           slot_frac,
        "doy_sin":             float(np.sin(2 * np.pi * doy / 365)),
        "doy_cos":             float(np.cos(2 * np.pi * doy / 365)),
        "temp_c":              cur,
        "running_max":         rmax,
        "temp_vs_climatology": cur - prev7,
        "delta_30m":           cur - lag(2),
        "delta_1h":            cur - lag(3),
        "accel":               (cur - lag(2)) - (lag(2) - lag(3)),
        "recent_slope":        slope,
        "temp_lag_3":          lag(4),
        "roll3_std":           float(np.std(vals[-3:])) if n >= 3 else 0.0,
        "plateau_indicator":   plateau,
        "morning_max":         mmax,
        "radiation_proxy":     radiation,
        "humidity_drop_1h":    hum_drop_1h,
        "prev_7d_avg_max":     prev7,
        "seasonal_peak_prior": get_seasonal_prior(month, hour, slot30),
    }


def _ols_slope(vals: list[float]) -> float:
    """Slope OLS sobre os últimos k valores."""
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float) - (n - 1) / 2
    y = np.array(vals, dtype=float)
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════
#  PREDICT — LightGBM only (retrocompatibilidade)
# ══════════════════════════════════════════════════════
def predict_p(model, feat_cols, slots_so_far: list[dict], current: dict,
              month: int, doy: int) -> float:
    """P(pico já ocorreu) — LightGBM only."""
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0
    row   = build_features(slots_so_far, current, month, doy)
    avail = [f for f in feat_cols if f in row]
    X     = pd.DataFrame([row])[avail].fillna(0)
    return float(model.predict(X)[0])


# ══════════════════════════════════════════════════════
#  PREDICT ENSEMBLE — LGBM + XGB + Z-Score
# ══════════════════════════════════════════════════════
def predict_ensemble(
    models: dict,
    slots_so_far: list[dict],
    current: dict,
    month: int,
    doy: int,
    zscore_detector: "StreamingPeakDetector | None" = None,
) -> dict:
    """
    Ensemble prediction: LGBM + XGB + Z-Score.

    Retorna:
      p_ensemble:  float — probabilidade final ponderada
      p_lgbm:      float — LightGBM raw
      p_xgb:       float | None — XGBoost raw
      p_zscore:    float | None — z-score raw
      weights:     dict — pesos usados
      components:  dict — detalhe de cada componente
    """
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return {
            "p_ensemble": 0.0, "p_lgbm": 0.0, "p_xgb": None,
            "p_zscore": None, "weights": models["ensemble_weights"],
            "components": {},
        }

    feat_cols = models["feat_cols"]
    row       = build_features(slots_so_far, current, month, doy)
    avail     = [f for f in feat_cols if f in row]
    X         = pd.DataFrame([row])[avail].fillna(0)

    # ── LightGBM ──────────────────────────────────────
    p_lgbm = float(models["model_lgb"].predict(X)[0])

    # ── XGBoost ───────────────────────────────────────
    p_xgb = None
    if models["model_xgb"] is not None:
        try:
            p_xgb = float(models["model_xgb"].predict(X)[0])
        except Exception:
            p_xgb = None

    # ── Z-Score ───────────────────────────────────────
    p_zscore = None
    if zscore_detector is not None:
        cur_temp = current.get("temp_c", vals[-1] if (vals := [s["temp_c"] for s in slots_so_far]) else 0)
        p_zscore = zscore_detector.update(cur_temp)

    # ── Ensemble ──────────────────────────────────────
    w = models["ensemble_weights"]
    p_ensemble = w["lgbm"] * p_lgbm

    if p_xgb is not None:
        p_ensemble += w["xgb"] * p_xgb
    # else: peso já redistribuído em load_models()

    if p_zscore is not None:
        p_ensemble += w["zscore"] * p_zscore
    # else: peso do zscore vai para o resíduo (distribuído em load_models)

    p_ensemble = float(np.clip(p_ensemble, 0.0, 1.0))

    return {
        "p_ensemble": p_ensemble,
        "p_lgbm":     p_lgbm,
        "p_xgb":      p_xgb,
        "p_zscore":   p_zscore,
        "weights":    w,
        "components": {
            "lgbm_contribution":   round(w["lgbm"] * p_lgbm, 4),
            "xgb_contribution":    round(w["xgb"] * p_xgb, 4) if p_xgb is not None else None,
            "zscore_contribution": round(w["zscore"] * p_zscore, 4) if p_zscore is not None else None,
        },
    }


# ══════════════════════════════════════════════════════
#  STREAMING Z-SCORE PEAK DETECTOR
# ══════════════════════════════════════════════════════
class StreamingPeakDetector:
    """
    Detector de pico baseado em z-score suavizado + análise de slope.
    Não requer treino — funciona em tempo real no buffer de streaming.

    Lógica:
      1. Z-score da temperatura actual vs janela recente
         → temp anormalmente alta = zona de pico
      2. Slope OLS dos últimos slots
         → slope ≈ 0 ou negativo = pico já passou
      3. Proximidade ao running max
         → temp ≈ running max = pico provável

    Combinação: sigmoid em cada sinal → média ponderada.
    """

    def __init__(self, lookback: int = 24, threshold_z: float = 1.5):
        """
        Args:
            lookback:     slots no buffer (24 = 12h de dados 30min)
            threshold_z:  z-score acima do qual consideramos "anomalia alta"
        """
        self.lookback    = lookback
        self.threshold_z = threshold_z
        self.buffer: list[float] = []
        self.running_max: float  = -999.0

    def update(self, temp: float) -> float:
        """
        Adiciona nova temperatura e retorna P(peak already occurred).
        """
        self.buffer.append(temp)
        self.running_max = max(self.running_max, temp)

        if len(self.buffer) > self.lookback * 2:
            self.buffer = self.buffer[-self.lookback * 2:]

        if len(self.buffer) < 8:
            return 0.0

        arr = np.array(self.buffer[-self.lookback:])
        n   = len(arr)

        # ── Sinal 1: Z-score ──────────────────────────
        mean = np.mean(arr)
        std  = np.std(arr)
        z    = (temp - mean) / std if std > 0.1 else 0.0
        # Sigmoid: alto z → alta probabilidade
        z_signal = float(1.0 / (1.0 + np.exp(-1.5 * (z - self.threshold_z))))

        # ── Sinal 2: Slope ────────────────────────────
        if n >= 4:
            x = np.arange(n, dtype=float)
            slope = float(np.polyfit(x, arr, 1)[0])
        else:
            slope = 0.0
        # Sigmoid: slope ≈ 0 ou negativo → alta probabilidade
        # slope > 0 (ainda a subir) → baixa probabilidade
        slope_signal = float(1.0 / (1.0 + np.exp(5.0 * slope)))

        # ── Sinal 3: Proximidade ao running max ──────
        pct_of_max = temp / self.running_max if self.running_max > 0 else 1.0
        max_signal = min(1.0, pct_of_max ** 2)

        # ── Combinação ponderada ──────────────────────
        p = 0.40 * z_signal + 0.35 * slope_signal + 0.25 * max_signal

        return float(np.clip(p, 0.0, 1.0))

    def reset(self):
        """Reset para novo dia."""
        self.buffer = []
        self.running_max = -999.0


# ══════════════════════════════════════════════════════
#  PREV_7D_AVG_MAX
# ══════════════════════════════════════════════════════
def compute_prev7(history: dict, d: date) -> float:
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
#  HISTÓRICO DIÁRIO
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
