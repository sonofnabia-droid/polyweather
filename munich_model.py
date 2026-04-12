"""
munich_model.py
===============
Carregamento de modelos (LightGBM + XGBoost), construção de features
V1 canónicas, predição ensemble com predict_proba(), z-score streaming,
gestão do histórico diário de máximas.

ASSINATURA de predict_ensemble:
  predict_ensemble(models, slots_so_far, current, month, doy, zscore_detector=None)

  NOTA: feat_cols NÃO é argumento — vem de models["feat_cols"] internamente.
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from munich_config import (
    MODEL_LGB, MODEL_XGB, MODEL_CONFIG, FEATURE_COLS,
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
#  LOAD MODELS
# ══════════════════════════════════════════════════════
def load_models() -> dict:
    if not MODEL_LGB.exists():
        raise FileNotFoundError(
            f"\n  {C['red']}Modelo nao encontrado: {MODEL_LGB}{R}\n"
            "  Corre: python munich_train.py"
        )

    model_lgb = joblib.load(MODEL_LGB)
    config    = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat_cols = config.get("feature_cols", FEATURE_COLS)

    raw_prior = config.get("seasonal_peak_prior", {})
    prior_map: dict[tuple, float] = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    raw_thresh = config.get("monthly_threshold", {})
    monthly_threshold: dict[int, float] = {}
    for k, v in raw_thresh.items():
        try:
            monthly_threshold[int(k)] = float(v)
        except ValueError:
            pass

    doy_poly_raw = config.get("doy_poly_coeffs")
    doy_poly = np.array(doy_poly_raw, dtype=float) if doy_poly_raw else None

    model_xgb = None
    if MODEL_XGB.exists():
        try:
            model_xgb = joblib.load(MODEL_XGB)
            print(f"  {C['green']}✓{R} XGBoost  features={len(feat_cols)}")
        except Exception as e:
            print(f"  {C['yellow']}⚠ XGBoost load falhou: {e}{R}")

    saved_weights = config.get("ensemble_weights", {})
    weights = {k: saved_weights.get(k, v) for k, v in
               {"lgbm": 0.50, "xgb": 0.30, "zscore": 0.20}.items()}

    if model_xgb is None:
        xgb_w = weights.get("xgb", 0.3)
        weights["xgb"] = 0.0
        weights["lgbm"]   += xgb_w * 0.6
        weights["zscore"] += xgb_w * 0.4

    auc = config.get("global_auc", "?")

    if doy_poly is not None:
        thr_str = f"curva DOY grau {len(doy_poly)-1}"
    elif monthly_threshold:
        thr_str = f"{len(monthly_threshold)} meses"
    else:
        thr_str = f"{C['yellow']}nao disponivel{R}"

    print(f"  {C['green']}✓{R} LightGBM  AUC={auc}  "
          f"features={len(feat_cols)}  "
          f"threshold={thr_str}  "
          f"prior={'sim' if prior_map else C['yellow']+'nao'+R}  "
          f"weights=[LGBM {weights['lgbm']:.0%} "
          f"XGB {weights['xgb']:.0%} Z {weights['zscore']:.0%}]")

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
#  FEATURE BUILDER — 18 FEATURES V1 CANÓNICAS
# ══════════════════════════════════════════════════════
def build_features(slots_so_far: list[dict], current: dict,
                   month: int, doy: int, minute: int = 0) -> dict:
    vals  = [s["temp_c"] for s in slots_so_far]
    hums  = [s.get("humidity", 70) for s in slots_so_far]
    n     = len(vals)
    cur   = vals[-1]
    rmax  = max(vals)
    hour  = current["hour"]
    slot30 = current.get("slot30", ceil_slot(hour, minute)[1])
    cloud = float(current.get("cloud_cover", 50))

    def lag(k):  return vals[-k] if n >= k else vals[0]
    def lagh(k): return hums[-k] if n >= k else hums[0]

    morn_vals = [s["temp_c"] for s in slots_so_far[:-1] if s["hour"] <= 12]
    mmax = max(morn_vals) if morn_vals else cur

    prev7     = current.get("prev_7d_avg_max", rmax)
    slot_frac = (hour + slot30 / 60.0) / 24.0

    # recent_slope
    slope_w = vals[-4:] if n >= 4 else vals
    slope = _ols_slope(slope_w)

    # plateau_indicator
    plat_w  = vals[-6:] if n >= 6 else vals
    plateau = 1.0 if (np.std(plat_w) < 0.4 and n >= 4) else 0.0

    # radiation_proxy
    radiation = float(np.cos((slot_frac - 0.5) * 2 * np.pi)) * (1 - cloud / 100)

    # humidity_drop_1h
    hum_drop_1h = lagh(3) - hums[-1] if n >= 3 else 0.0

    # ── Features V2 (defaults para dados faltantes) ───────────────
    dewpt = float(current.get("dewpoint_c", cur - 10))
    pres  = float(current.get("pressure_hpa", 1013.0))
    wdir  = float(current.get("wind_dir_deg", 0.0))
    wspd  = float(current.get("wind_speed_kmh", 5.0))
    wgst  = float(current.get("wind_gust_kmh", 8.0))
    uv    = float(current.get("uv_index", 3.0))

    temp_to_dewpoint_gap = max(0.0, cur - dewpt)

    # Pressure trend: últimas 3h (6 slots)
    press_vals = [s.get("pressure_hpa", 1013.0) for s in slots_so_far[-6:]]
    if len(press_vals) >= 2:
        pressure_trend_3h = press_vals[-1] - press_vals[0]
    else:
        pressure_trend_3h = 0.0

    # Wind south proxy: alinhamento com 180° (Sul)
    if 135 <= wdir <= 225:
        wind_south_proxy = 1.0 - abs(wdir - 180) / 45.0
    else:
        wind_south_proxy = 0.0

    # Foehn: vento sul + rajadas + seco
    hu = hums[-1] if hums else 70
    foehn_south = 1.0 if 135 <= wdir <= 225 else 0.0
    foehn_gusty = min(1.0, wgst / 30.0)
    foehn_dry   = max(0.0, (80 - hu) / 20.0) if hu < 80 else 0.0
    foehn_indicator = foehn_south * (0.4 + 0.3 * foehn_gusty + 0.3 * foehn_dry)

    return {
        # ── V1 (18) ──
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
        # ── V2 (7) ──
        "dewpoint_c":          dewpt,
        "temp_to_dewpoint_gap": temp_to_dewpoint_gap,
        "pressure_trend_3h":   pressure_trend_3h,
        "wind_south_proxy":    wind_south_proxy,
        "wind_speed_kmh":      wspd,
        "uv_index":            uv,
        "foehn_indicator":     foehn_indicator,
    }


def _ols_slope(vals: list[float]) -> float:
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
    """P(pico já ocorreu) — LightGBM only. Usa predict_proba()."""
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0
    row   = build_features(slots_so_far, current, month, doy)
    avail = [f for f in feat_cols if f in row]
    X     = pd.DataFrame([row])[avail].fillna(0)
    # predict_proba()[:, 1] retorna probabilidade calibrada 0.0–1.0
    return float(model.predict_proba(X)[0, 1])


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

    ASSINATURA: predict_ensemble(models, slots_so_far, current, month, doy, zscore)
    feat_cols NÃO é argumento — vem de models["feat_cols"].
    Usa predict_proba() NÃO predict().

    Retorna dict com p_ensemble, p_lgbm, p_xgb, p_zscore, weights, components.
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
    p_lgbm = float(models["model_lgb"].predict_proba(X)[0, 1])

    # ── XGBoost ───────────────────────────────────────
    p_xgb = None
    if models["model_xgb"] is not None:
        try:
            p_xgb = float(models["model_xgb"].predict_proba(X)[0, 1])
        except Exception:
            p_xgb = None

    # ── Z-Score ───────────────────────────────────────
    p_zscore = None
    if zscore_detector is not None:
        p_zscore = zscore_detector.update(current["temp_c"])

    # ── Ensemble ──────────────────────────────────────
    w = models["ensemble_weights"]
    p_ensemble = w["lgbm"] * p_lgbm

    if p_xgb is not None:
        p_ensemble += w["xgb"] * p_xgb

    if p_zscore is not None:
        p_ensemble += w["zscore"] * p_zscore

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
    """

    def __init__(self, lookback: int = 24, threshold_z: float = 1.5):
        self.lookback    = lookback
        self.threshold_z = threshold_z
        self.buffer: list[float] = []
        self.running_max: float  = -999.0

    def update(self, temp: float) -> float:
        self.buffer.append(temp)
        self.running_max = max(self.running_max, temp)
        if len(self.buffer) > self.lookback * 2:
            self.buffer = self.buffer[-self.lookback * 2:]
        if len(self.buffer) < 8:
            return 0.0

        arr = np.array(self.buffer[-self.lookback:])
        n   = len(arr)

        mean, std = np.mean(arr), np.std(arr)
        z = (temp - mean) / std if std > 0.1 else 0.0
        z_signal = float(1.0 / (1.0 + np.exp(-1.5 * (z - self.threshold_z))))

        if n >= 4:
            slope = float(np.polyfit(np.arange(n, dtype=float), arr, 1)[0])
        else:
            slope = 0.0
        slope_signal = float(1.0 / (1.0 + np.exp(5.0 * slope)))

        pct_of_max = temp / self.running_max if self.running_max > 0 else 1.0
        max_signal = min(1.0, pct_of_max ** 2)

        p = 0.40 * z_signal + 0.35 * slope_signal + 0.25 * max_signal
        return float(np.clip(p, 0.0, 1.0))

    def reset(self):
        self.buffer      = []
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
