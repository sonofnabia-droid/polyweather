"""
munich_train.py
===============
Treino do modelo de pico max temp Munich — V3 Ensemble.

Pipeline:
  1. Carregar historic/munich.csv (desde 2010)
  2. Construir features 30min CEILING (18 V1 + 7 V2 = 25 features)
  3. Walk-Forward Validation (expanding window por ano)
  4. Treinar LightGBM + XGBoost em paralelo
  5. Calcular threshold adaptativo (curva DOY contínua)
  6. Calcular seasonal_peak_prior
  7. Guardar modelos + config

Uso:
    python munich_train.py
    python munich_train.py --no-xgb
    python munich_train.py --doy-degree 7
"""

import argparse
import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  CORES ANSI
# ══════════════════════════════════════════════════════
R   = "\033[0m"
DIM = "\033[2m"

# ══════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════
OUTPUT_DIR  = Path("munich_peak_model")
DATA_CSV    = Path("historic/munich.csv")
BERLIN_TZ   = ZoneInfo("Europe/Berlin")

DAY_START = 6
DAY_END   = 21
MIN_HOUR  = 6

# 18 FEATURES V1 + 7 V2 PREDITIVAS = 25 total
FEATURE_COLS = [
    # ── V1 Canónicas (18) ──
    "slot_frac",
    "doy_sin", "doy_cos",
    "temp_c", "running_max",
    "temp_vs_climatology",
    "delta_30m", "delta_1h", "accel",
    "recent_slope",
    "temp_lag_3",
    "roll3_std",
    "plateau_indicator",
    "morning_max",
    "radiation_proxy",
    "humidity_drop_1h",
    "prev_7d_avg_max",
    "seasonal_peak_prior",
    # ── V2 PREDITIVAS (7) ──
    "dewpoint_c",
    "temp_to_dewpoint_gap",
    "pressure_trend_3h",
    "wind_south_proxy",
    "wind_speed_kmh",
    "uv_index",
    "foehn_indicator",
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
            hour=0, minute=0, second=0, microsecond=0
        )
        h2 = 0
    return dt_local, h2, s2


# ══════════════════════════════════════════════════════
#  LOAD CSV
# ══════════════════════════════════════════════════════
def load_csv(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} não encontrado.")

    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","

    raw = pd.read_csv(csv_path, sep=sep, low_memory=False)

    if "timestamp_utc" not in raw.columns:
        raise ValueError("CSV não contém coluna 'timestamp_utc'.")

    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], errors="coerce")
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
    raw["date"]           = dates
    raw["hour"]           = hours
    raw["slot30"]         = slots30
    raw["month"]          = raw["datetime_local"].dt.month
    raw["doy"]            = raw["datetime_local"].dt.dayofyear

    raw["temp_c"] = pd.to_numeric(raw["temp_c"], errors="coerce")

    if "humidity_pct" in raw.columns:
        raw["humidity"] = pd.to_numeric(raw["humidity_pct"], errors="coerce")
    else:
        raw["humidity"] = 70.0

    if "sky_cover" in raw.columns:
        raw["cloud_cover"] = pd.to_numeric(raw["sky_cover"], errors="coerce")
    else:
        raw["cloud_cover"] = 50.0

    # ── V2: Colunas preditivas ───────────────────────
    raw["dewpoint_c"]     = pd.to_numeric(raw.get("dewpt_c"), errors="coerce")
    raw["pressure_hpa"]   = pd.to_numeric(raw.get("pressure_hpa"), errors="coerce")
    raw["wind_dir_deg"]   = pd.to_numeric(raw.get("wind_dir_deg"), errors="coerce")
    raw["wind_speed_kmh"] = pd.to_numeric(raw.get("wind_speed_kmh"), errors="coerce")
    raw["wind_gust_kmh"]  = pd.to_numeric(raw.get("wind_gust_kmh"), errors="coerce")
    raw["uv_index"]       = pd.to_numeric(raw.get("uv_index"), errors="coerce")

    # Defaults para NaNs (especialmente 2010)
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

    print(f"  {len(df):,} slots  {df['date'].nunique()} dias  "
          f"resolucao: 30min CEILING")
    return df


# ══════════════════════════════════════════════════════
#  BUILD DATASET
# ══════════════════════════════════════════════════════
def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Constrói o dataset de treino slot a slot com 25 features (18 V1 + 7 V2).
    """
    print("  A construir dataset slot a slot...")

    # Daily max e prev7
    daily_max  = df.groupby("date")["temp_c"].max().sort_index()
    dates_list = list(daily_max.index)
    prev7_map  = {}
    for i, d in enumerate(dates_list):
        if i == 0:
            prev7_map[d] = daily_max[d]
        else:
            window = daily_max[dates_list[max(0, i-7):i]]
            prev7_map[d] = float(window.mean()) if len(window) else daily_max[d]

    # Prior sazonal
    slot_counts = {}
    slot_peak   = {}
    for d, day_df in df.groupby("date"):
        peak_idx = day_df["temp_c"].idxmax()
        peak_h   = int(day_df.loc[peak_idx, "hour"])
        peak_s   = int(day_df.loc[peak_idx, "slot30"])

        for _, row in day_df.iterrows():
            h, s = int(row["hour"]), int(row["slot30"])
            m = int(row["month"])
            key = (m, h, s)
            slot_counts[key] = slot_counts.get(key, 0) + 1
            if h > peak_h or (h == peak_h and s >= peak_s):
                slot_peak[key] = slot_peak.get(key, 0) + 1

    prior_map = {}
    for key, total in slot_counts.items():
        prior_map[key] = round(slot_peak.get(key, 0) / total, 4)

    # Construir linhas
    rows = []
    for d, day_df in df.groupby("date"):
        day_df = day_df.sort_values(["hour", "slot30"]).reset_index(drop=True)
        month  = int(day_df["month"].iloc[0])
        doy    = int(day_df["doy"].iloc[0])

        peak_idx = day_df["temp_c"].idxmax()
        peak_h   = int(day_df.loc[peak_idx, "hour"])
        peak_s   = int(day_df.loc[peak_idx, "slot30"])

        slots_so_far = []
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

            label = 1 if (h > peak_h or (h == peak_h and s >= peak_s)) else 0

            if h < MIN_HOUR or len(slots_so_far) < 4:
                continue

            # ── Features V1 ────────────────────────────
            vals = [sl["temp_c"] for sl in slots_so_far]
            hums = [sl.get("humidity", 70) for sl in slots_so_far]
            n    = len(vals)
            cur  = vals[-1]
            rmax = max(vals)

            def lag(k):  return vals[-k] if n >= k else vals[0]
            def lagh(k): return hums[-k] if n >= k else hums[0]

            morn_vals = [sl["temp_c"] for sl in slots_so_far[:-1] if sl["hour"] <= 12]
            mmax = max(morn_vals) if morn_vals else cur

            slot_frac = (h + s / 60) / 24

            # recent_slope
            slope_w = vals[-4:] if n >= 4 else vals
            if len(slope_w) >= 2:
                _x = np.arange(len(slope_w), dtype=float) - (len(slope_w) - 1) / 2
                _denom = float((_x * _x).sum())
                slope = float((_x * np.array(slope_w)).sum() / _denom) if _denom > 0 else 0.0
            else:
                slope = 0.0

            # plateau_indicator
            plat_w  = vals[-6:] if n >= 6 else vals
            plateau = 1.0 if (np.std(plat_w) < 0.4 and n >= 4) else 0.0

            # radiation_proxy
            radiation = float(np.cos((slot_frac - 0.5) * 2 * np.pi)) * (1 - cl / 100)

            # humidity_drop_1h
            hum_drop = lagh(3) - hums[-1] if n >= 3 else 0.0

            prev7 = prev7_map.get(d, rmax)
            prior = prior_map.get((month, h, s), 0.5)

            # ── Features V2 ────────────────────────────
            dewpt = slot_entry["dewpoint_c"]
            pres  = slot_entry["pressure_hpa"]
            wdir  = slot_entry["wind_dir_deg"]
            wspd  = slot_entry["wind_speed_kmh"]
            wgst  = slot_entry["wind_gust_kmh"]
            uv    = slot_entry["uv_index"]

            temp_to_dewpoint_gap = max(0.0, cur - dewpt)

            # Pressure trend: últimas 3h (6 slots)
            press_vals = [sl["pressure_hpa"] for sl in slots_so_far[-6:]]
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
            foehn_south = 1.0 if 135 <= wdir <= 225 else 0.0
            foehn_gusty = min(1.0, wgst / 30.0)
            foehn_dry   = max(0.0, (80 - hu) / 20.0) if hu < 80 else 0.0
            foehn_indicator = foehn_south * (0.4 + 0.3 * foehn_gusty + 0.3 * foehn_dry)

            feat_row = {
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
                "label":               label,
                "date":                d,
                "month":               month,
                "doy":                 doy,
            }
            rows.append(feat_row)

    result = pd.DataFrame(rows)
    print(f"  {len(result):,} samples  "
          f"positivos={result['label'].mean()*100:.1f}%  "
          f"features={len(FEATURE_COLS)}")
    return result, prior_map


# ══════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════
def walk_forward_train(dataset: pd.DataFrame,
                       train_xgb: bool = True) -> dict:
    dates_sorted = sorted(dataset["date"].unique())
    years = sorted(set(d.year for d in dates_sorted))

    print(f"\n  Walk-Forward: {len(years)} folds")

    all_preds_lgb = []
    all_preds_xgb = []
    all_labels    = []
    fold_results  = []

    for i, test_year in enumerate(years):
        train_mask = dataset["date"].apply(lambda d: d.year < test_year)
        test_mask  = dataset["date"].apply(lambda d: d.year == test_year)

        train_df = dataset[train_mask]
        test_df  = dataset[test_mask]

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["label"]
        X_test  = test_df[FEATURE_COLS].fillna(0)
        y_test  = test_df["label"]

        # LightGBM
        lgb = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective="binary", metric="auc", verbose=-1, random_state=42,
        )
        lgb.fit(X_train, y_train)
        preds_lgb = lgb.predict_proba(X_test)[:, 1]
        auc_lgb   = roc_auc_score(y_test, preds_lgb)

        # XGBoost
        auc_xgb    = None
        preds_xgb  = None
        if train_xgb:
            xgb = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                max_leaves=31, min_child_weight=50, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="auc",
                verbosity=0, random_state=42, use_label_encoder=False,
            )
            xgb.fit(X_train, y_train)
            preds_xgb = xgb.predict_proba(X_test)[:, 1]
            auc_xgb   = roc_auc_score(y_test, preds_xgb)

        # Ensemble
        if preds_xgb is not None:
            preds_ens = 0.6 * preds_lgb + 0.4 * preds_xgb
            auc_ens   = roc_auc_score(y_test, preds_ens)
        else:
            auc_ens = auc_lgb

        all_preds_lgb.extend(preds_lgb)
        if preds_xgb is not None:
            all_preds_xgb.extend(preds_xgb)
        all_labels.extend(y_test.values)

        xgb_str = f"  XGB AUC={auc_xgb:.4f}" if auc_xgb else ""
        print(f"    Fold {test_year}: LGBM AUC={auc_lgb:.4f}{xgb_str}  "
              f"Ensemble AUC={auc_ens:.4f}")

        fold_results.append({
            "year": test_year, "auc_lgb": round(auc_lgb, 4),
            "auc_xgb": round(auc_xgb, 4) if auc_xgb else None,
            "auc_ens": round(auc_ens, 4),
            "n_train": len(train_df), "n_test": len(test_df),
        })

    global_auc_lgb = roc_auc_score(all_labels, all_preds_lgb)
    global_auc_xgb = roc_auc_score(all_labels, all_preds_xgb) if all_preds_xgb else None
    global_auc_ens = roc_auc_score(
        all_labels,
        [0.6*l + 0.4*x for l, x in zip(all_preds_lgb, all_preds_xgb)]
    ) if all_preds_xgb else global_auc_lgb

    print(f"\n  Global Walk-Forward AUC:")
    print(f"    LightGBM : {global_auc_lgb:.4f}")
    if global_auc_xgb:
        print(f"    XGBoost  : {global_auc_xgb:.4f}")
    print(f"    Ensemble : {global_auc_ens:.4f}")

    return {
        "global_auc_lgb": global_auc_lgb,
        "global_auc_xgb": global_auc_xgb,
        "global_auc_ens": global_auc_ens,
        "fold_results":   fold_results,
    }


# ══════════════════════════════════════════════════════
#  FINAL TRAIN (todos os dados)
# ══════════════════════════════════════════════════════
def train_final(dataset: pd.DataFrame, train_xgb: bool = True) -> tuple:
    X = dataset[FEATURE_COLS].fillna(0)
    y = dataset["label"]

    lgb = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary", metric="auc", verbose=-1, random_state=42,
    )
    lgb.fit(X, y)
    print(f"  LightGBM treinado em {len(X):,} samples")

    xgb = None
    if train_xgb:
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            max_leaves=31, min_child_weight=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc",
            verbosity=0, random_state=42, use_label_encoder=False,
        )
        xgb.fit(X, y)
        print(f"  XGBoost treinado em {len(X):,} samples")

    return lgb, xgb


# ══════════════════════════════════════════════════════
#  THRESHOLD ADAPTATIVO — curva DOY contínua
# ══════════════════════════════════════════════════════
def compute_doy_threshold(dataset: pd.DataFrame, degree: int = 5) -> np.ndarray:
    doy_thresholds = {}
    for doy in range(1, 366):
        sub = dataset[dataset["doy"] == doy]
        if len(sub) < 20:
            continue
        preds  = sub["seasonal_peak_prior"].values
        labels = sub["label"].values
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.3, 0.95, 0.05):
            pred_bin = (preds >= thr).astype(int)
            f1 = f1_score(labels, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        doy_thresholds[doy] = best_thr

    doys      = np.array(list(doy_thresholds.keys()))
    thrs      = np.array(list(doy_thresholds.values()))
    doys_norm = (doys - 183) / 183
    return np.polyfit(doys_norm, thrs, degree)


# ══════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════
def save_models(lgb, xgb, prior_map, doy_poly, wf_results: dict,
                train_xgb: bool = True):
    OUTPUT_DIR.mkdir(exist_ok=True)

    joblib.dump(lgb, OUTPUT_DIR / "lgbm_peak.pkl")
    print(f"  OK {OUTPUT_DIR / 'lgbm_peak.pkl'}")

    if xgb is not None:
        joblib.dump(xgb, OUTPUT_DIR / "xgb_peak.pkl")
        print(f"  OK {OUTPUT_DIR / 'xgb_peak.pkl'}")

    # ── Feature Importances ──────────────────────────────
    feat_cols = FEATURE_COLS

    lgb_imp = lgb.feature_importances_
    lgb_imp_pct = lgb_imp / lgb_imp.sum()

    if xgb is not None:
        xgb_imp = xgb.feature_importances_
        xgb_imp_pct = xgb_imp / xgb_imp.sum()
        ens_imp = 0.5 * lgb_imp_pct + 0.3 * xgb_imp_pct
        ens_imp_pct = ens_imp / ens_imp.sum()
    else:
        xgb_imp_pct = None
        ens_imp_pct = lgb_imp_pct

    sorted_idx = np.argsort(ens_imp_pct)[::-1]

    lgb_importance_dict = {}
    xgb_importance_dict = {}
    ensemble_importance_dict = {}

    for i in sorted_idx:
        fname = feat_cols[i]
        lgb_importance_dict[fname] = round(float(lgb_imp_pct[i]) * 100, 2)
        if xgb_imp_pct is not None:
            xgb_importance_dict[fname] = round(float(xgb_imp_pct[i]) * 100, 2)
        ensemble_importance_dict[fname] = round(float(ens_imp_pct[i]) * 100, 2)

    # ── Config JSON ────────────────────────────────────
    config = {
        "feature_cols":     feat_cols,
        "resolution":       "30min_ceiling",
        "global_auc":       round(wf_results["global_auc_lgb"], 4),
        "global_auc_xgb":   round(wf_results["global_auc_xgb"], 4) if wf_results.get("global_auc_xgb") else None,
        "global_auc_ens":   round(wf_results["global_auc_ens"], 4),
        "fold_results":     wf_results.get("fold_results", []),
        "doy_poly_coeffs":  doy_poly.tolist() if doy_poly is not None else None,
        "ensemble_weights": {
            "lgbm": 0.50, "xgb": 0.30 if train_xgb else 0.0, "zscore": 0.20,
        },
        "lgbm_feature_importance_pct":  lgb_importance_dict,
        "xgb_feature_importance_pct":   xgb_importance_dict,
        "ensemble_feature_importance_pct": ensemble_importance_dict,
    }

    prior_str = {}
    for (m, h, s), v in prior_map.items():
        prior_str[f"{m}_{h}_{s}"] = v
    config["seasonal_peak_prior"] = prior_str

    (OUTPUT_DIR / "peak_model_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"  OK {OUTPUT_DIR / 'peak_model_config.json'}")

    # ── Mostrar Top 10 ─────────────────────────────────
    print(f"\n  Top 10 Features (Ensemble Importance):")
    v2_names = {"dewpoint_c", "temp_to_dewpoint_gap", "pressure_trend_3h",
                "wind_south_proxy", "wind_speed_kmh", "uv_index", "foehn_indicator"}
    for rank, i in enumerate(sorted_idx[:10], 1):
        pct = ens_imp_pct[i] * 100
        bar = '#' * int(pct / 2)
        v2_tag = " [V2]" if feat_cols[i] in v2_names else ""
        print(f"    {rank:>2}. {feat_cols[i]:<24} {pct:>5.1f}%  {DIM}{bar}{R}{v2_tag}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Munich Peak Model Trainer V3")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Não treinar XGBoost")
    parser.add_argument("--no-wf", action="store_true",
                        help="Skip walk-forward validation")
    parser.add_argument("--doy-degree", type=int, default=5)
    args = parser.parse_args()

    train_xgb = not args.no_xgb

    print("=" * 60)
    print("  Munich Peak Model — Trainer V3 (Ensemble)")
    print("=" * 60)

    print("\n[1/5] A carregar dados...")
    df = load_csv()

    print("\n[2/5] A construir dataset...")
    dataset, prior_map = build_dataset(df)

    wf_results = None
    if not args.no_wf:
        print("\n[3/5] Walk-Forward Validation...")
        wf_results = walk_forward_train(dataset, train_xgb=train_xgb)

    print("\n[4/5] Treino final (todos os dados)...")
    lgb, xgb = train_final(dataset, train_xgb=train_xgb)

    if wf_results is None:
        X = dataset[FEATURE_COLS].fillna(0)
        y = dataset["label"]
        preds_lgb = lgb.predict_proba(X)[:, 1]
        wf_results = {
            "global_auc_lgb": roc_auc_score(y, preds_lgb),
            "global_auc_xgb": None, "global_auc_ens": roc_auc_score(y, preds_lgb),
            "fold_results": [],
        }

    print("\n[5/5] A calcular threshold adaptativo...")
    doy_poly = compute_doy_threshold(dataset, degree=args.doy_degree)

    print("\nA guardar modelos...")
    save_models(lgb, xgb, prior_map, doy_poly, wf_results, train_xgb=train_xgb)

    print("\nTreino completo!")


if __name__ == "__main__":
    main()
