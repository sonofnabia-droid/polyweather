"""
munich_backtester.py
====================
Backtest V3: Ensemble (LGBM + XGB + Z-Score) + Phased(3) ou Single(1).

Modos:
  --mode phased  → 3 parcelas $5 (P1 manhã invertida, P2 dupla, P3 alta)
  --mode single  → 1 compra $15 quando p_ensemble >= 75%

Uso:
    python munich_backtester.py
    python munich_backtester.py --mode phased
    python munich_backtester.py --mode single
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from zoneinfo import ZoneInfo
from rich.progress import (Progress, BarColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.console import Console
from rich.table import Table
from rich import box as rich_box

from munich_phased_entry import PhasedEntry, SingleEntry

_console = Console(force_terminal=True)
warnings.filterwarnings("ignore")

OUTPUT_DIR   = Path("backtest_results")
MODEL_LGB    = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_XGB    = Path("munich_peak_model/xgb_peak.pkl")
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
DATA_CSV     = Path("historic/munich.csv")

BERLIN_TZ = ZoneInfo("Europe/Berlin")
DAY_START = 6
DAY_END   = 21
MIN_HOUR  = 6

SEASONS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5],
    "summer": [6, 7, 8],  "autumn": [9, 10, 11],
}
MONTHS_PT = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
]

FEATURE_COLS = [
    # ── V1 Canónicas (18) ──
    "slot_frac", "doy_sin", "doy_cos", "temp_c", "running_max",
    "temp_vs_climatology", "delta_30m", "delta_1h", "accel",
    "recent_slope", "temp_lag_3", "roll3_std", "plateau_indicator",
    "morning_max", "radiation_proxy", "humidity_drop_1h",
    "prev_7d_avg_max", "seasonal_peak_prior",
    # ── V2 Preditivas (7) ──
    "dewpoint_c", "temp_to_dewpoint_gap", "pressure_trend_3h",
    "wind_south_proxy", "wind_speed_kmh", "uv_index", "foehn_indicator",
]


# ══════════════════════════════════════════════════════
#  HELPERS CEILING
# ══════════════════════════════════════════════════════
def ceil_slot(hour: int, minute: int) -> tuple[int, int]:
    if minute < 30:
        return hour, 30
    return hour + 1, 0


def normalize_datetime_ceiling(dt_utc: pd.Timestamp):
    dt_local = dt_utc.astimezone(BERLIN_TZ)
    h, m = dt_local.hour, dt_local.minute
    h2, s2 = ceil_slot(h, m)
    if h2 == 24:
        dt_local = (dt_local + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        h2 = 0
    return dt_local, h2, s2


# ══════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════
def load_models():
    if not MODEL_LGB.exists():
        raise FileNotFoundError("Corre munich_train.py primeiro")

    model_lgb = joblib.load(MODEL_LGB)
    config    = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat_cols = config.get("feature_cols", FEATURE_COLS)

    raw_prior = config.get("seasonal_peak_prior", {})
    prior_map = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    raw_thresh = config.get("monthly_threshold", {})
    monthly_threshold = {}
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
        except Exception:
            pass

    weights = config.get(
        "ensemble_weights",
        {"lgbm": 0.50, "xgb": 0.30, "zscore": 0.20},
    )
    if model_xgb is None:
        xgb_w = weights.get("xgb", 0.30)
        weights["lgbm"]   += xgb_w * 0.6
        weights["zscore"] += xgb_w * 0.4
        weights["xgb"]     = 0.0

    auc = config.get("global_auc", "?")
    models_str = "LightGBM"
    if model_xgb:
        models_str += " + XGBoost"
    models_str += " + Z-Score"

    _console.print(
        f"    [green]✓[/green] {models_str}  AUC=[cyan]{auc}[/cyan]  "
        f"features=[cyan]{len(feat_cols)}[/cyan]  "
        f"weights=[LGBM {weights['lgbm']:.0%} "
        f"XGB {weights['xgb']:.0%} Z {weights['zscore']:.0%}]")

    return {
        "model_lgb": model_lgb, "model_xgb": model_xgb,
        "feat_cols": feat_cols, "prior_map": prior_map,
        "monthly_threshold": monthly_threshold, "doy_poly": doy_poly,
        "ensemble_weights": weights,
    }


# ══════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════
def load_data(csv_path=DATA_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} não encontrado")

    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","

    raw = pd.read_csv(csv_path, sep=sep, low_memory=False)

    if "timestamp_utc" not in raw.columns:
        raise ValueError("CSV sem coluna 'timestamp_utc'")

    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"], errors="coerce")
    if raw["timestamp_utc"].dt.tz is not None:
        raw["timestamp_utc"] = raw["timestamp_utc"].dt.tz_convert(None)
    raw["timestamp_utc"] = raw["timestamp_utc"].dt.tz_localize("UTC")

    dt_locals, dates, hours, slots30 = [], [], [], []
    for ts in raw["timestamp_utc"]:
        dt_loc, h2, s2 = normalize_datetime_ceiling(ts)
        dt_locals.append(dt_loc)
        dates.append(dt_loc.date())
        hours.append(h2)
        slots30.append(s2)

    raw["datetime_local"] = dt_locals
    raw["date"]   = dates
    raw["hour"]   = hours
    raw["slot30"] = slots30
    raw["month"]  = raw["datetime_local"].dt.month
    raw["doy"]    = raw["datetime_local"].dt.dayofyear
    raw["temp_c"] = pd.to_numeric(raw["temp_c"], errors="coerce")

    if "humidity_pct" in raw.columns:
        raw["humidity"] = pd.to_numeric(
            raw["humidity_pct"], errors="coerce")
    else:
        raw["humidity"] = 70.0

    if "sky_cover" in raw.columns:
        raw["cloud_cover"] = pd.to_numeric(
            raw["sky_cover"], errors="coerce")
    else:
        raw["cloud_cover"] = 50.0

    # ── V2: Colunas preditivas ───────────────────────
    raw["dewpoint_c"]     = pd.to_numeric(raw.get("dewpt_c"), errors="coerce")
    raw["pressure_hpa"]   = pd.to_numeric(raw.get("pressure_hpa"), errors="coerce")
    raw["wind_dir_deg"]   = pd.to_numeric(raw.get("wind_dir_deg"), errors="coerce")
    raw["wind_speed_kmh"] = pd.to_numeric(raw.get("wind_speed_kmh"), errors="coerce")
    raw["wind_gust_kmh"]  = pd.to_numeric(raw.get("wind_gust_kmh"), errors="coerce")
    raw["uv_index"]       = pd.to_numeric(raw.get("uv_index"), errors="coerce")

    raw["dewpoint_c"]     = raw["dewpoint_c"].fillna(raw["temp_c"] - 10)
    raw["pressure_hpa"]   = raw["pressure_hpa"].fillna(1013.0)
    raw["wind_dir_deg"]   = raw["wind_dir_deg"].fillna(0.0)
    raw["wind_speed_kmh"] = raw["wind_speed_kmh"].fillna(5.0)
    raw["wind_gust_kmh"]  = raw["wind_gust_kmh"].fillna(8.0)
    raw["uv_index"]       = raw["uv_index"].fillna(3.0)

    df = raw[
        (raw["hour"] >= DAY_START) & (raw["hour"] <= DAY_END)
    ].dropna(subset=["temp_c"]).sort_values(
        ["date", "hour", "slot30"]
    ).reset_index(drop=True)

    print(f"    {len(df):,} slots  {df['date'].nunique()} dias")
    return df


def compute_prev7(df: pd.DataFrame) -> dict:
    daily_max = df.groupby("date")["temp_c"].max().sort_index()
    dates = list(daily_max.index)
    prev7 = {}
    for i, d in enumerate(dates):
        if i == 0:
            prev7[d] = daily_max[d]
        else:
            window = daily_max[dates[max(0, i - 7):i]]
            prev7[d] = float(window.mean()) if len(window) else daily_max[d]
    return prev7


# ══════════════════════════════════════════════════════
#  Z-SCORE STREAMING
# ══════════════════════════════════════════════════════
class ZScoreStreaming:
    def __init__(self, lookback=24, threshold_z=1.5):
        self.lookback    = lookback
        self.threshold_z = threshold_z
        self.buffer      = []
        self.running_max = -999.0

    def update(self, temp: float) -> float:
        self.buffer.append(temp)
        self.running_max = max(self.running_max, temp)
        if len(self.buffer) > self.lookback * 2:
            self.buffer = self.buffer[-self.lookback * 2:]
        if len(self.buffer) < 8:
            return 0.0

        arr  = np.array(self.buffer[-self.lookback:])
        n    = len(arr)
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

        return float(np.clip(
            0.40 * z_signal + 0.35 * slope_signal + 0.25 * max_signal,
            0, 1))

    def reset(self):
        self.buffer      = []
        self.running_max = -999.0


# ══════════════════════════════════════════════════════
#  SIMULATED MARKET
# ══════════════════════════════════════════════════════
class SimulatedMarket:
    """
    De manhã: mercado tem MAIS ruído → highest ask
    NÃO é sempre o running_max → confirmação falha frequentemente.
    À tarde: mercado converge → confirmação funciona.
    """
    def __init__(self, temp_range=range(5, 40)):
        self.temp_range = temp_range

    def get_simulated_brackets(self, p_ensemble: float,
                                running_max: float,
                                hour: int) -> list[dict]:
        brackets = []
        rmax_int = int(round(running_max))

        if hour < 11:
            market_noise = 0.20
        elif hour < 14:
            market_noise = 0.12
        else:
            market_noise = 0.04

        for temp in self.temp_range:
            dist = abs(temp - rmax_int)
            if dist == 0:
                base_ask = min(0.92, p_ensemble * 0.7 + 0.20)
            elif dist == 1:
                base_ask = min(0.70, p_ensemble * 0.5 + 0.10)
            elif dist == 2:
                base_ask = min(0.40, p_ensemble * 0.3 + 0.05)
            elif dist == 3:
                base_ask = min(0.20, p_ensemble * 0.15)
            else:
                base_ask = max(0.03, 0.10 - dist * 0.015)

            if temp < rmax_int - 1:
                base_ask = min(0.97, base_ask + 0.10)

            ask = float(np.clip(
                base_ask + np.random.uniform(-market_noise, market_noise),
                0.02, 0.97))

            is_last  = (temp == self.temp_range[-1])
            is_first = (temp == self.temp_range[0])

            if is_last:
                label, lo, hi = (f"{temp}°C or higher",
                                 float(temp), 99.0)
            elif is_first:
                label, lo, hi = (f"{temp}°C or lower",
                                 -99.0, float(temp))
            else:
                label, lo, hi = f"{temp}°C", float(temp), float(temp)

            brackets.append({
                "label": label, "ask": round(ask, 4),
                "price": round(ask, 4),
                "temp_lo": lo, "temp_hi": hi,
            })

        return brackets


# ══════════════════════════════════════════════════════
#  BUILD SLOT FEATURES (V1 + V2)
# ══════════════════════════════════════════════════════
def build_slot(slots_so_far, current, month, doy, prior_map):
    vals = [s["temp_c"] for s in slots_so_far]
    hums = [s.get("humidity", 70) for s in slots_so_far]
    n    = len(vals)
    cur  = vals[-1]
    hour = current["hour"]
    slot30 = current["slot30"]
    cloud  = float(current.get("cloud_cover", 50))
    hu     = float(current.get("humidity", 70))

    def lag(k):  return vals[-k] if n >= k else vals[0]
    def lagh(k): return hums[-k] if n >= k else hums[0]

    rmax = max(vals)

    morn_vals = [s["temp_c"] for s in slots_so_far[:-1] if s["hour"] <= 12]
    mmax = max(morn_vals) if morn_vals else cur

    prior = prior_map.get((month, hour, slot30), 0.5)
    slot_frac = (hour + slot30 / 60) / 24

    slope_w = vals[-4:] if n >= 4 else vals
    if len(slope_w) >= 2:
        _x = np.arange(len(slope_w), dtype=float) - (len(slope_w) - 1) / 2
        _denom = float((_x * _x).sum())
        slope = (float((_x * np.array(slope_w)).sum() / _denom)
                 if _denom > 0 else 0.0)
    else:
        slope = 0.0

    plat_w  = vals[-6:] if n >= 6 else vals
    plateau = 1.0 if (np.std(plat_w) < 0.4 and n >= 4) else 0.0

    radiation = float(np.cos((slot_frac - 0.5) * 2 * np.pi)) * (1 - cloud / 100)

    hum_drop = lagh(3) - hums[-1] if n >= 3 else 0.0

    prev7 = current["prev_7d_avg_max"]

    # ── V2: Features preditivas ───────────────────────
    dewpt = current.get("dewpoint_c", cur - 10)
    pres  = current.get("pressure_hpa", 1013)
    wdir  = current.get("wind_dir_deg", 0)
    wspd  = current.get("wind_speed_kmh", 5)
    wgst  = current.get("wind_gust_kmh", 8)
    uv    = current.get("uv_index", 3)

    temp_to_dewpoint_gap = max(0.0, cur - dewpt)

    press_vals = [s.get("pressure_hpa", 1013) for s in slots_so_far[-6:]]
    if len(press_vals) >= 2:
        pressure_trend_3h = press_vals[-1] - press_vals[0]
    else:
        pressure_trend_3h = 0.0

    if 135 <= wdir <= 225:
        wind_south_proxy = 1.0 - abs(wdir - 180) / 45.0
    else:
        wind_south_proxy = 0.0

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
        "humidity_drop_1h":    hum_drop,
        "prev_7d_avg_max":     prev7,
        "seasonal_peak_prior": prior,
        # ── V2 (7) ──
        "dewpoint_c":          dewpt,
        "temp_to_dewpoint_gap": temp_to_dewpoint_gap,
        "pressure_trend_3h":   pressure_trend_3h,
        "wind_south_proxy":    wind_south_proxy,
        "wind_speed_kmh":      wspd,
        "uv_index":            uv,
        "foehn_indicator":     foehn_indicator,
    }


# ══════════════════════════════════════════════════════
#  PREDICT ENSEMBLE
# ══════════════════════════════════════════════════════
def predict_ensemble(models, slots_so_far, current,
                     month, doy, zscore_det):
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0, 0.0, None, None

    feat_cols = models["feat_cols"]
    feat  = build_slot(slots_so_far, current, month, doy, models["prior_map"])
    avail = [f for f in feat_cols if f in feat]
    X     = pd.DataFrame([feat])[avail].fillna(0)

    p_lgbm = float(models["model_lgb"].predict_proba(X)[0, 1])

    p_xgb = None
    if models["model_xgb"] is not None:
        try:
            p_xgb = float(models["model_xgb"].predict_proba(X)[0, 1])
        except Exception:
            pass

    p_zscore = zscore_det.update(current["temp_c"]) if zscore_det else None

    w = models["ensemble_weights"]
    p = w["lgbm"] * p_lgbm
    if p_xgb is not None:
        p += w["xgb"] * p_xgb
    if p_zscore is not None:
        p += w["zscore"] * p_zscore

    return float(np.clip(p, 0, 1)), p_lgbm, p_xgb, p_zscore


# ══════════════════════════════════════════════════════
#  RUN BACKTEST
# ══════════════════════════════════════════════════════
def run(df, models, sim_market, mode="phased", parcel_size=5.0):
    feat_cols         = models["feat_cols"]
    prior_map         = models["prior_map"]
    monthly_threshold = models["monthly_threshold"]
    doy_poly          = models["doy_poly"]

    def get_threshold(month, doy=0):
        if doy_poly is not None and doy > 0:
            val = float(np.polyval(doy_poly, (doy - 183) / 183))
            return float(np.clip(val, 0.25, 0.95))
        return monthly_threshold.get(month, 0.75)

    daily_max = df.groupby("date")["temp_c"].max()
    dates_s   = sorted(daily_max.index)
    prev7     = compute_prev7(df)

    results   = []
    slots_out = []

    with Progress(
        TextColumn("[cyan]Dias..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("", total=len(dates_s))

        for d, day_df in df.groupby("date"):
            progress.update(task, advance=1)
            day_df = day_df.sort_values(
                ["hour", "slot30"]).reset_index(drop=True)
            month   = int(day_df["month"].iloc[0])
            doy     = int(day_df["doy"].iloc[0])
            cloud_m = float(day_df["cloud_cover"].mean())

            peak_idx  = day_df["temp_c"].idxmax()
            peak_slot = (int(day_df.loc[peak_idx, "hour"]),
                         int(day_df.loc[peak_idx, "slot30"]))
            peak_temp = float(day_df["temp_c"].max())

            slots_so_far = []
            zscore = ZScoreStreaming()
            zscore.reset()

            if mode == "single":
                entry = SingleEntry(parcel_size=parcel_size * 3)
            else:
                entry = PhasedEntry(parcel_size=parcel_size)

            forecast_agrees = np.random.random() < 0.80

            for _, row in day_df.iterrows():
                h  = int(row["hour"])
                s  = int(row["slot30"])
                t  = float(row["temp_c"])
                cl = float(row["cloud_cover"])
                hu = float(row["humidity"])

                slot_entry = {
                    "hour": h, "slot30": s, "temp_c": t,
                    "cloud_cover": cl, "humidity": hu,
                    "dewpoint_c":    float(row.get("dewpoint_c", t - 10)),
                    "pressure_hpa":  float(row.get("pressure_hpa", 1013)),
                    "wind_dir_deg":  float(row.get("wind_dir_deg", 0)),
                    "wind_speed_kmh":float(row.get("wind_speed_kmh", 5)),
                    "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                    "uv_index":      float(row.get("uv_index", 3)),
                }
                slots_so_far.append(slot_entry)

                if h < MIN_HOUR or len(slots_so_far) < 4:
                    slots_out.append({
                        "date": d, "hour": h, "slot30": s,
                        "temp": t, "p_ens": 0.0,
                        "peak_true_h": peak_slot[0],
                    })
                    continue

                current_extra = {
                    "hour": h, "slot30": s,
                    "cloud_cover": cl, "humidity": hu,
                    "prev_7d_avg_max": prev7.get(d, peak_temp),
                    "temp_c": t,
                    "dewpoint_c":    float(row.get("dewpoint_c", t - 10)),
                    "pressure_hpa":  float(row.get("pressure_hpa", 1013)),
                    "wind_dir_deg":  float(row.get("wind_dir_deg", 0)),
                    "wind_speed_kmh":float(row.get("wind_speed_kmh", 5)),
                    "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                    "uv_index":      float(row.get("uv_index", 3)),
                }

                p_ens, p_lgbm, p_xgb, p_zs = predict_ensemble(
                    models, slots_so_far, current_extra,
                    month, doy, zscore
                )

                running_max = max(sl["temp_c"] for sl in slots_so_far)

                slots_out.append({
                    "date": d, "hour": h, "slot30": s,
                    "temp": round(t, 2), "p_ens": round(p_ens, 3),
                    "peak_true_h": peak_slot[0],
                })

                # ── Simular mercado ────────────────────
                brackets   = sim_market.get_simulated_brackets(
                    p_ens, running_max, h)
                market_sim = {"brackets": brackets}

                fc_agreement = {"valid": forecast_agrees}

                # ── Avaliar parcelas ──────────────────
                actions = entry.evaluate(
                    p_ens, h, market_sim, running_max, fc_agreement)

                for act in actions:
                    if act["size_usdc"] > 0:
                        pidx = act["parcel_idx"]

                        if pidx == 1:
                            best = max(brackets, key=lambda b: b["ask"])
                        else:
                            rmax_int = int(round(running_max))
                            best = next(
                                (b for b in brackets
                                 if b["temp_lo"] <= rmax_int <= b["temp_hi"]),
                                max(brackets, key=lambda b: b["ask"]))

                        entry.mark_bought(pidx, {
                            "hour": h, "slot30": s,
                            "ask": best["ask"],
                            "size_usdc": act["size_usdc"],
                            "bracket_label": best["label"],
                        })

            # ── Resultados do dia ─────────────────────
            def slot_idx(h, s):
                return h * 2 + s // 30

            peak_idx_val = slot_idx(peak_slot[0], peak_slot[1])

            parcel_lags = []
            for i in range(3):
                if (entry.parcel_bought[i]
                        and entry.parcel_records[i] is not None):
                    lag = (slot_idx(
                               entry.parcel_records[i]["hour"],
                               entry.parcel_records[i]["slot30"])
                           - peak_idx_val)
                    parcel_lags.append(lag)
                else:
                    parcel_lags.append(None)

            detected = any(entry.parcel_bought)

            first_slot = next(
                (entry.parcel_records[i]
                 for i in range(3)
                 if entry.parcel_bought[i]
                 and entry.parcel_records[i] is not None),
                None)

            lag_first = None
            if first_slot is not None:
                lag_first = (slot_idx(first_slot["hour"],
                                      first_slot["slot30"])
                             - peak_idx_val)

            correct_lags   = [l for l in parcel_lags
                              if l is not None and l >= 0]
            premature_lags = [l for l in parcel_lags
                              if l is not None and l < 0]
            season = next(
                (s for s, ms in SEASONS.items() if month in ms),
                "spring")

            results.append({
                "date":            d,
                "month":           month,
                "doy":             doy,
                "season":          season,
                "cloud_mean":      round(cloud_m, 1),
                "peak_temp":       round(peak_temp, 2),
                "peak_h_true":     peak_slot[0],
                "detected":        detected,
                "n_parcels":       entry.n_parcels_bought,
                "parcel1_bought":  entry.parcel_bought[0],
                "parcel2_bought":  entry.parcel_bought[1],
                "parcel3_bought":  entry.parcel_bought[2],
                "parcel1_lag":     parcel_lags[0],
                "parcel2_lag":     parcel_lags[1],
                "parcel3_lag":     parcel_lags[2],
                "lag_first_h":     (round(lag_first * 0.5, 1)
                                    if lag_first is not None else None),
                "correct":         (len(correct_lags) > 0
                                    and len(premature_lags) == 0),
                "premature":       len(premature_lags) > 0,
                "missed":          not detected,
                "total_invested":  entry.total_invested,
            })

    return pd.DataFrame(results), pd.DataFrame(slots_out)


# ══════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════
def compute_metrics(results):
    n    = len(results)
    corr = results["correct"].sum()
    prem = results["premature"].sum()
    miss = results["missed"].sum()

    correct_lags_h = (results[results["correct"]
                      & results["lag_first_h"].notna()]
                      ["lag_first_h"].values)

    m = {
        "n_days":       n,
        "correct_pct":  round(corr / n * 100, 1),
        "premature_pct": round(prem / n * 100, 1),
        "missed_pct":   round(miss / n * 100, 1),
        "lag_mean_h":   (round(float(np.mean(correct_lags_h)), 2)
                         if len(correct_lags_h) else None),
        "lag_median_h": (round(float(np.median(correct_lags_h)), 2)
                         if len(correct_lags_h) else None),
        "lag_le1h_pct": (round((correct_lags_h <= 1.0).mean() * 100, 1)
                         if len(correct_lags_h) else 0),
        "lag_le2h_pct": (round((correct_lags_h <= 2.0).mean() * 100, 1)
                         if len(correct_lags_h) else 0),
        "parcel1_pct":  round(results["parcel1_bought"].mean() * 100, 1),
        "parcel2_pct":  round(results["parcel2_bought"].mean() * 100, 1),
        "parcel3_pct":  round(results["parcel3_bought"].mean() * 100, 1),
        "avg_n_parcels": round(results["n_parcels"].mean(), 2),
        "avg_invested": round(results["total_invested"].mean(), 2),
    }

    for season in SEASONS:
        sub = results[results["season"] == season]
        if not sub.empty:
            m[f"{season}_correct_pct"] = round(
                sub["correct"].mean() * 100, 1)
            sl = (sub[sub["correct"]
                      & sub["lag_first_h"].notna()]
                  ["lag_first_h"].values)
            m[f"{season}_lag_mean_h"] = (
                round(float(np.mean(sl)), 2) if len(sl) else None)

    return m


# ══════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════
def plot(results, slots_df, metrics, start_year, mode="phased"):
    print("  A gerar gráficos...")

    BG, PANEL = "#07090D", "#0D1018"
    C = {
        "correct": "#25BE62", "premature": "#F0A500",
        "missed": "#D93838", "blue": "#4D9EFF",
        "muted": "#424C64", "text": "#D8DCE8",
        "grid": "#111520", "border": "#181E2C",
        "purple": "#A855F7",
    }

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": C["border"], "grid.color": C["grid"],
        "text.color": C["text"], "axes.labelcolor": C["muted"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.titlecolor": C["text"], "legend.facecolor": PANEL,
        "legend.edgecolor": "#252E44", "font.family": "monospace",
    })

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    months_range = sorted(results["month"].unique())
    m_lbls = [MONTHS_PT[m - 1] for m in months_range]

    # 1. Detecção por mês
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Resultado por Mês")
    corr_m = results.groupby("month")["correct"].mean() * 100
    prem_m = results.groupby("month")["premature"].mean() * 100
    miss_m = results.groupby("month")["missed"].mean() * 100
    x = np.arange(len(months_range))

    ax1.bar(x, corr_m.reindex(months_range, fill_value=0), 0.6,
            color=C["correct"], alpha=0.88, label="Correcto")
    ax1.bar(x, prem_m.reindex(months_range, fill_value=0), 0.6,
            color=C["premature"], alpha=0.88, label="Prematuro",
            bottom=corr_m.reindex(months_range, fill_value=0))
    bot2 = (corr_m.reindex(months_range, fill_value=0)
            + prem_m.reindex(months_range, fill_value=0))
    ax1.bar(x, miss_m.reindex(months_range, fill_value=0), 0.6,
            color=C["missed"], alpha=0.88, label="Não detectado",
            bottom=bot2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_lbls, fontsize=8)
    ax1.set_ylabel("% dias")
    ax1.set_ylim(0, 110)
    ax1.axhline(80, color=C["muted"], lw=0.8, ls="--", alpha=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Parcelas por mês
    ax2 = fig.add_subplot(gs[0, 2])
    if mode == "phased":
        ax2.set_title("Parcelas Compradas por Mês")
        p1 = results.groupby("month")["parcel1_bought"].mean() * 100
        p2 = results.groupby("month")["parcel2_bought"].mean() * 100
        p3 = results.groupby("month")["parcel3_bought"].mean() * 100
        ax2.bar(x, p1.reindex(months_range, fill_value=0), 0.6,
                color=C["blue"], alpha=0.8, label="P1 Manhã")
        ax2.bar(x, p2.reindex(months_range, fill_value=0), 0.6,
                color=C["purple"], alpha=0.8, label="P2 Pico~",
                bottom=p1.reindex(months_range, fill_value=0))
        bot3 = (p1.reindex(months_range, fill_value=0)
                + p2.reindex(months_range, fill_value=0))
        ax2.bar(x, p3.reindex(months_range, fill_value=0), 0.6,
                color=C["correct"], alpha=0.8, label="P3 Confirmado",
                bottom=bot3)
    else:
        ax2.set_title("Single Buy por Mês")
        p1 = results.groupby("month")["parcel1_bought"].mean() * 100
        ax2.bar(x, p1.reindex(months_range, fill_value=0), 0.6,
                color=C["blue"], alpha=0.8, label="Single")
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_lbls, fontsize=7)
    ax2.set_ylabel("% dias")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Distribuição do lag
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Distribuição do Lag (1ª compra, correctos)")
    correct_lags = results[results["correct"]]["lag_first_h"].dropna().values
    if len(correct_lags):
        vmin, vmax = correct_lags.min(), correct_lags.max()
        bins = np.arange(vmin - 0.5, vmax + 1.0, 0.5)
        cnts, edges = np.histogram(correct_lags, bins=bins)
        for b, cnt in zip(edges[:-1], cnts):
            ax3.bar(b, cnt, width=0.45,
                    color=C["correct"] if b >= 0 else C["premature"],
                    alpha=0.85)
        ax3.axvline(0, color=C["text"], lw=1.2, ls="--")
        ax3.axvline(float(np.mean(correct_lags)), color=C["blue"],
                    lw=1.5, ls="--",
                    label=f"Média: {np.mean(correct_lags):.1f}h")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

    # 4. Lag por parcela
    ax4 = fig.add_subplot(gs[1, 1])
    if mode == "phased":
        ax4.set_title("Lag Médio por Parcela (correctos)")
        parcel_lag_means = []
        parcel_lag_stds  = []
        parcel_labels    = []
        for pidx, pname in [(0, "P1 Manhã"), (1, "P2 Pico~"),
                            (2, "P3 Confirmado")]:
            col = f"parcel{pidx + 1}_lag"
            vals = results[results["correct"]][col].dropna().values * 0.5
            if len(vals):
                parcel_lag_means.append(np.mean(vals))
                parcel_lag_stds.append(np.std(vals))
                parcel_labels.append(pname)
            else:
                parcel_lag_means.append(0)
                parcel_lag_stds.append(0)
                parcel_labels.append(pname)

        colors_p = [C["blue"], C["purple"], C["correct"]]
        ax4.bar(parcel_labels, parcel_lag_means, yerr=parcel_lag_stds,
                color=colors_p, alpha=0.85,
                error_kw={"color": C["muted"], "capsize": 3})
        ax4.axhline(0, color=C["muted"], lw=0.8)
        ax4.set_ylabel("Lag (h)")
    else:
        ax4.set_title("Lag Single Buy (correctos)")
        col = "parcel1_lag"
        vals = results[results["correct"]][col].dropna().values * 0.5
        if len(vals):
            ax4.bar(["Single"], [np.mean(vals)],
                    yerr=[np.std(vals)],
                    color=C["blue"], alpha=0.85,
                    error_kw={"color": C["muted"], "capsize": 3})
        ax4.axhline(0, color=C["muted"], lw=0.8)
        ax4.set_ylabel("Lag (h)")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Nº parcelas por dia
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Distribuição: Compras/Dia")
    n_par = results["n_parcels"].values
    if mode == "phased":
        for v in [0, 1, 2, 3]:
            cnt = (n_par == v).sum()
            col = (C["missed"] if v == 0 else C["premature"] if v == 1
                   else C["purple"] if v == 2 else C["correct"])
            ax5.bar(str(v), cnt, color=col, alpha=0.85)
    else:
        for v, lbl in [(0, "0"), (1, "1")]:
            cnt = (n_par == v).sum()
            col = C["missed"] if v == 0 else C["correct"]
            ax5.bar(lbl, cnt, color=col, alpha=0.85)
    ax5.set_xlabel("Compras")
    ax5.set_ylabel("Dias")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Heatmap p_ensemble
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_title("P(ensemble) — 60 dias amostra")
    sample = results.sample(
        min(60, len(results)), random_state=42).sort_values("date")
    slot_keys = [(h, s)
                 for h in range(DAY_START, DAY_END + 1)
                 for s in [0, 30]]

    pivot = []
    for _, row in sample.iterrows():
        dslots = slots_df[slots_df["date"] == row["date"]].copy()
        dslots["sk"] = list(zip(dslots["hour"], dslots["slot30"]))
        sk_to_p = dict(zip(dslots["sk"], dslots["p_ens"]))
        pivot.append([float(sk_to_p.get((h, s), 0))
                      for h, s in slot_keys])

    if pivot:
        mat = np.array(pivot)
        im  = ax6.imshow(mat, aspect="auto", cmap="RdYlGn",
                         vmin=0, vmax=1, interpolation="nearest",
                         extent=[0, len(slot_keys), len(sample), 0])
        plt.colorbar(im, ax=ax6, fraction=0.015, pad=0.01,
                     label="P(ensemble)")
        htp = [i for i, (h, s) in enumerate(slot_keys) if s == 0]
        htl = [f"{h}h" for h, s in slot_keys if s == 0]
        ax6.set_xticks(htp)
        ax6.set_xticklabels(htl, fontsize=7)
        ax6.set_yticks(range(len(sample)))
        ax6.set_yticklabels(
            [str(d) for d in sample["date"].values], fontsize=6)

    # 7. Investimento médio por mês
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.set_title("Investimento Médio/Dia por Mês")
    inv_m = results.groupby("month")["total_invested"].mean()
    ax7.bar(m_lbls, inv_m.reindex(months_range, fill_value=0),
            color=C["blue"], alpha=0.85)
    ax7.set_ylabel("USDC")
    ax7.grid(True, alpha=0.3, axis="y")
    ax7.tick_params(axis="x", rotation=45, labelsize=7)

    # 8. Por estação
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.set_title("Correcto por Estação")
    season_data = []
    for season in ["winter", "spring", "summer", "autumn"]:
        key = f"{season}_correct_pct"
        if key in metrics:
            season_data.append((season, metrics[key]))
    if season_data:
        s_names, s_vals = zip(*season_data)
        s_cols = [C["blue"] if v >= 70 else C["premature"]
                  for v in s_vals]
        ax8.bar(s_names, s_vals, color=s_cols, alpha=0.85)
        ax8.axhline(80, color=C["muted"], lw=0.8, ls="--", alpha=0.5)
        ax8.set_ylabel("% correcto")
        ax8.grid(True, alpha=0.3, axis="y")

    # 9. Resumo
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis("off")
    mode_tag = "PHASED 3x5" if mode == "phased" else "SINGLE 1x15"
    summary_lines = [
        f"Modo: {mode_tag}",
        f"Dias: {metrics['n_days']}",
        f"Correcto: {metrics['correct_pct']}%",
        f"Prematuro: {metrics['premature_pct']}%",
        f"Não detectado: {metrics['missed_pct']}%",
        "",
        f"Lag médio (correctos): "
        f"+{metrics.get('lag_mean_h', '?')}h",
        f"Lag ≤ 1h: {metrics.get('lag_le1h_pct', 0)}%",
        f"Lag ≤ 2h: {metrics.get('lag_le2h_pct', 0)}%",
        "",
    ]
    if mode == "phased":
        summary_lines += [
            f"P1 Manhã: {metrics['parcel1_pct']}% dias",
            f"P2 Pico~: {metrics['parcel2_pct']}% dias",
            f"P3 Confirmado: {metrics['parcel3_pct']}% dias",
            f"Média parcelas/dia: {metrics['avg_n_parcels']}",
        ]
    else:
        summary_lines += [
            f"Single Buy: {metrics['parcel1_pct']}% dias",
        ]
    summary_lines += [
        f"Invest. médio: ${metrics['avg_invested']:.2f}/dia",
    ]
    ax9.text(0.1, 0.95, "\n".join(summary_lines),
             transform=ax9.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             color=C["text"],
             bbox=dict(boxstyle="round", facecolor=PANEL,
                       edgecolor=C["border"]))

    lag_str = (f"+{metrics['lag_mean_h']}h"
               if metrics.get('lag_mean_h') else "N/A")
    fig.suptitle(
        f"Munich Max Temp — {mode_tag}  "
        f"correcto={metrics['correct_pct']}%  lag={lag_str}  "
        f"invest=${metrics['avg_invested']:.1f}/dia",
        fontsize=13)

    OUTPUT_DIR.mkdir(exist_ok=True)
    mode_slug = "phased" if mode == "phased" else "single"
    out_path = OUTPUT_DIR / f"munich_backtest_{mode_slug}_{start_year}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard guardado: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Munich Backtest V3")
    parser.add_argument("--mode", choices=["phased", "single"],
                        default="phased",
                        help="phased=3 parcelas $5, single=1 compra $15")
    args = parser.parse_args()

    start_str = input("Data de início (YYYY-MM-DD): ").strip()
    start_date = pd.to_datetime(start_str).date()
    end_date   = date.today() - timedelta(days=1)

    mode_label = "PHASED 3x5" if args.mode == "phased" else "SINGLE 1x15"
    print(f"\n  Backtest V3: {start_date} → {end_date}  [{mode_label}]")

    print("\n[1/5] Modelos...")
    models = load_models()

    print("\n[2/5] Dados...")
    df_all = load_data()
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df = df_all[
        (df_all["date"] >= start_date)
        & (df_all["date"] <= end_date)
    ].copy()
    print(f"  {len(df):,} slots no intervalo")

    print(f"\n[3/5] Backtest ({mode_label})...")
    sim_market = SimulatedMarket()
    results, slots_df = run(df, models, sim_market,
                            mode=args.mode, parcel_size=5.0)

    print("\n[4/5] Métricas...")
    metrics = compute_metrics(results)

    # ── Rich Tables ────────────────────────────────────
    _console.print()
    mode_header = "Resultados V3 — PHASED" if args.mode == "phased" \
                  else "Resultados V3 — SINGLE"
    _console.rule(f"[bold cyan]{mode_header}[/bold cyan]")

    t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Label", style="dim", width=26)
    t.add_column("Value", style="bold white")

    t.add_row("Dias analisados", f"{metrics['n_days']}")
    t.add_row("Correcto",
              f"[green]{metrics['correct_pct']}%[/green]")
    t.add_row("Prematuro",
              f"[yellow]{metrics['premature_pct']}%[/yellow]")
    t.add_row("Não detectado",
              f"[red]{metrics['missed_pct']}%[/red]")
    t.add_row("Lag médio (correctos)",
              f"+{metrics.get('lag_mean_h', '?')}h")
    t.add_row("Lag ≤ 1h",
              f"{metrics.get('lag_le1h_pct', 0)}%")
    t.add_row("Lag ≤ 2h",
              f"{metrics.get('lag_le2h_pct', 0)}%")

    if args.mode == "phased":
        t.add_row("P1 Manhã (10h–12h)",
                  f"{metrics['parcel1_pct']}% dias")
        t.add_row("P2 Modelo+Mercado",
                  f"{metrics['parcel2_pct']}% dias")
        t.add_row("P3 Alta confiança",
                  f"{metrics['parcel3_pct']}% dias")
    else:
        t.add_row("Single Buy (p≥75%)",
                  f"{metrics['parcel1_pct']}% dias")
    t.add_row("Invest. médio/dia",
              f"${metrics['avg_invested']:.2f}")
    _console.print(t)

    # Por estação
    _console.rule("[cyan]Por Estação[/cyan]", style="dim")
    t_s = Table(box=rich_box.SIMPLE, show_header=True, padding=(0, 2))
    t_s.add_column("Estação", style="cyan", width=12)
    t_s.add_column("Correcto", justify="right", width=10)
    t_s.add_column("Lag médio", justify="right", width=10)

    season_icons = {
        "winter": "❄️ ", "spring": "🌱",
        "summer": "☀️ ", "autumn": "🍂",
    }
    for season in ["winter", "spring", "summer", "autumn"]:
        key_c = f"{season}_correct_pct"
        key_l = f"{season}_lag_mean_h"
        if key_c in metrics:
            c_pct = metrics[key_c]
            col = ("green" if c_pct >= 70
                   else "yellow" if c_pct >= 50 else "red")
            lag_v = (f"+{metrics[key_l]:.2f}h"
                     if key_l in metrics
                     and metrics[key_l] is not None else "—")
            t_s.add_row(
                f"{season_icons.get(season, '')}{season}",
                f"[{col}]{c_pct:.1f}%[/{col}]",
                lag_v)
    _console.print(t_s)

    print(f"\n[5/5] Dashboard...")
    plot(results, slots_df, metrics, start_date.year, mode=args.mode)


if __name__ == "__main__":
    main()
