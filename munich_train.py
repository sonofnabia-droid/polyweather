"""
munich_train.py
===============
Treino do modelo de pico max temp Munich — V3 Ensemble.

Pipeline:
  1. Carregar historic/munich.csv (desde 2010)
  2. Construir features 30min CEILING (18 features V1)
  3. Walk-Forward Validation (expanding window)
  4. Treinar LightGBM + XGBoost em paralelo
  5. Calcular threshold adaptativo (curva DOY contínua)
  6. Calcular seasonal_peak_prior
  7. Guardar modelos + config

Uso:
    python munich_train.py
    python munich_train.py --walk-forward --xgb
"""

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from scipy.signal import find_peaks

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════
OUTPUT_DIR  = Path("munich_peak_model")
DATA_CSV    = Path("historic/munich.csv")
BERLIN_TZ   = ZoneInfo("Europe/Berlin")

DAY_START = 6
DAY_END   = 21
MIN_HOUR  = 6

FEATURE_COLS = [
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
]


# ══════════════════════════════════════════════════════
#  HELPERS CEILING
# ══════════════════════════════════════════════════════
def ceil_slot(hour: int, minute: int) -> tuple[int, int]:
    if minute < 30:
        return hour, 30
    else:
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
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o dataset de treino slot a slot.
    Para cada slot, calcula as 18 features V1 e o label (peak_already_passed).
    """
    print("  A construir dataset slot a slot...")

    # Daily max e prev7
    daily_max = df.groupby("date")["temp_c"].max().sort_index()
    dates_list = list(daily_max.index)
    prev7_map = {}
    for i, d in enumerate(dates_list):
        if i == 0:
            prev7_map[d] = daily_max[d]
        else:
            window = daily_max[dates_list[max(0, i-7):i]]
            prev7_map[d] = float(window.mean()) if len(window) else daily_max[d]

    # Prior sazonal: P(pico já passou | month, hour, slot30)
    # Contagem de vezes que o pico ocorreu antes ou neste slot
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

            # Pico já ocorreu se hora actual > hora do pico,
            # ou mesma hora mas slot actual >= slot do pico
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

        peak_idx  = day_df["temp_c"].idxmax()
        peak_h    = int(day_df.loc[peak_idx, "hour"])
        peak_s    = int(day_df.loc[peak_idx, "slot30"])

        slots_so_far = []
        for _, row in day_df.iterrows():
            h = int(row["hour"])
            s = int(row["slot30"])
            t = float(row["temp_c"])
            cl = float(row["cloud_cover"])
            hu = float(row["humidity"])

            slot_entry = {
                "hour": h, "slot30": s, "temp_c": t,
                "cloud_cover": cl, "humidity": hu,
            }
            slots_so_far.append(slot_entry)

            # Label: pico já ocorreu?
            label = 1 if (h > peak_h or (h == peak_h and s >= peak_s)) else 0

            if h < MIN_HOUR or len(slots_so_far) < 4:
                continue

            # Features
            vals = [sl["temp_c"] for sl in slots_so_far]
            hums = [sl.get("humidity", 70) for sl in slots_so_far]
            n    = len(vals)
            cur  = vals[-1]
            rmax = max(vals)

            def lag(k):
                return vals[-k] if n >= k else vals[0]
            def lagh(k):
                return hums[-k] if n >= k else hums[0]

            morn_vals = [sl["temp_c"] for sl in slots_so_far[:-1] if sl["hour"] <= 12]
            mmax = max(morn_vals) if morn_vals else cur

            slot_frac = (h + s/60) / 24

            # recent_slope
            slope_w = vals[-4:] if n >= 4 else vals
            if len(slope_w) >= 2:
                _x = np.arange(len(slope_w), dtype=float) - (len(slope_w)-1)/2
                _denom = float((_x*_x).sum())
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

            feat_row = {
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
    """
    Walk-Forward Validation com expanding window.
    Treina LightGBM (sempre) + XGBoost (opcional) em cada fold.

    Fold strategy: ano de teste, todos os anteriores para treino.
    """
    dates_sorted = sorted(dataset["date"].unique())
    years = sorted(set(d.year for d in dates_sorted))

    print(f"\n  Walk-Forward: {len(years)} folds")

    all_preds_lgb = []
    all_preds_xgb = []
    all_labels    = []
    fold_results  = []

    models_lgb_last = None
    models_xgb_last = None

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

        # ── LightGBM ──────────────────────────────────
        lgb = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary",
            metric="auc",
            verbose=-1,
            random_state=42,
        )
        lgb.fit(X_train, y_train)
        preds_lgb = lgb.predict_proba(X_test)[:, 1]
        auc_lgb   = roc_auc_score(y_test, preds_lgb)
        models_lgb_last = lgb

        # ── XGBoost ───────────────────────────────────
        auc_xgb = None
        preds_xgb = None
        models_xgb_last = None
        if train_xgb:
            xgb = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                max_leaves=31,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                verbosity=0,
                random_state=42,
                use_label_encoder=False,
            )
            xgb.fit(X_train, y_train)
            preds_xgb = xgb.predict_proba(X_test)[:, 1]
            auc_xgb   = roc_auc_score(y_test, preds_xgb)
            models_xgb_last = xgb

        # Ensemble AUC
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
              f"Ensemble AUC={auc_ens:.4f}  "
              f"train={len(train_df):,}  test={len(test_df):,}")

        fold_results.append({
            "year": test_year,
            "auc_lgb": round(auc_lgb, 4),
            "auc_xgb": round(auc_xgb, 4) if auc_xgb else None,
            "auc_ens": round(auc_ens, 4),
            "n_train": len(train_df),
            "n_test":  len(test_df),
        })

    # Global AUC
    global_auc_lgb = roc_auc_score(all_labels, all_preds_lgb)
    global_auc_xgb = roc_auc_score(all_labels, all_preds_xgb) if all_preds_xgb else None
    if all_preds_xgb:
        global_auc_ens = roc_auc_score(
            all_labels,
            [0.6*l + 0.4*x for l, x in zip(all_preds_lgb, all_preds_xgb)]
        )
    else:
        global_auc_ens = global_auc_lgb

    print(f"\n  Global Walk-Forward AUC:")
    print(f"    LightGBM : {global_auc_lgb:.4f}")
    if global_auc_xgb:
        print(f"    XGBoost  : {global_auc_xgb:.4f}")
    print(f"    Ensemble : {global_auc_ens:.4f}")

    return {
        "models_lgb_last": models_lgb_last,
        "models_xgb_last": models_xgb_last,
        "global_auc_lgb":  global_auc_lgb,
        "global_auc_xgb":  global_auc_xgb,
        "global_auc_ens":  global_auc_ens,
        "fold_results":    fold_results,
    }


# ══════════════════════════════════════════════════════
#  FINAL TRAIN (todos os dados)
# ══════════════════════════════════════════════════════
def train_final(dataset: pd.DataFrame, train_xgb: bool = True) -> tuple:
    """Treina modelos finais em TODOS os dados."""
    X = dataset[FEATURE_COLS].fillna(0)
    y = dataset["label"]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        metric="auc",
        verbose=-1,
        random_state=42,
    )
    lgb.fit(X, y)
    print(f"  LightGBM treinado em {len(X):,} samples")

    # XGBoost
    xgb = None
    if train_xgb:
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            max_leaves=31,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            verbosity=0,
            random_state=42,
            use_label_encoder=False,
        )
        xgb.fit(X, y)
        print(f"  XGBoost treinado em {len(X):,} samples")

    return lgb, xgb


# ══════════════════════════════════════════════════════
#  THRESHOLD ADAPTATIVO — curva DOY contínua
# ══════════════════════════════════════════════════════
def compute_doy_threshold(dataset: pd.DataFrame, degree: int = 5) -> np.ndarray:
    """
    Ajusta um polinómio à probabilidade óptima de threshold por dia do ano.
    Para cada DOY, encontra o threshold que maximiza F1 (correct detection).
    """
    from sklearn.metrics import f1_score

    doy_thresholds = {}
    for doy in range(1, 366):
        sub = dataset[dataset["doy"] == doy]
        if len(sub) < 20:
            continue

        preds = sub["seasonal_peak_prior"].values
        labels = sub["label"].values

        best_thr = 0.5
        best_f1  = 0.0
        for thr in np.arange(0.3, 0.95, 0.05):
            pred_binary = (preds >= thr).astype(int)
            f1 = f1_score(labels, pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

        doy_thresholds[doy] = best_thr

    # Ajustar polinómio
    doys  = np.array(list(doy_thresholds.keys()))
    thrs  = np.array(list(doy_thresholds.values()))
    doys_norm = (doys - 183) / 183
    coeffs = np.polyfit(doys_norm, thrs, degree)

    return coeffs


# ══════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════
def save_models(lgb, xgb, prior_map, doy_poly, wf_results: dict,
                train_xgb: bool = True):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # LightGBM
    joblib.dump(lgb, OUTPUT_DIR / "lgbm_peak.pkl")
    print(f"  ✓ {OUTPUT_DIR / 'lgbm_peak.pkl'}")

    # XGBoost
    if xgb is not None:
        joblib.dump(xgb, OUTPUT_DIR / "xgb_peak.pkl")
        print(f"  ✓ {OUTPUT_DIR / 'xgb_peak.pkl'}")

    # Config
    config = {
        "feature_cols":      FEATURE_COLS,
        "resolution":        "30min_ceiling",
        "global_auc":        round(wf_results["global_auc_lgb"], 4),
        "global_auc_xgb":    round(wf_results["global_auc_xgb"], 4) if wf_results.get("global_auc_xgb") else None,
        "global_auc_ens":    round(wf_results["global_auc_ens"], 4),
        "fold_results":      wf_results.get("fold_results", []),
        "doy_poly_coeffs":   doy_poly.tolist() if doy_poly is not None else None,
        "ensemble_weights":  {
            "lgbm":   0.50,
            "xgb":    0.30 if train_xgb else 0.0,
            "zscore": 0.20,
        },
    }

    # Prior sazonal
    prior_str = {}
    for (m, h, s), v in prior_map.items():
        prior_str[f"{m}_{h}_{s}"] = v
    config["seasonal_peak_prior"] = prior_str

    (OUTPUT_DIR / "peak_model_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"  ✓ {OUTPUT_DIR / 'peak_model_config.json'}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Munich Peak Model Trainer V3")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Não treinar XGBoost (só LightGBM)")
    parser.add_argument("--no-wf", action="store_true",
                        help="Skip walk-forward validation")
    parser.add_argument("--doy-degree", type=int, default=5,
                        help="Grau do polinómio DOY threshold (default: 5)")
    args = parser.parse_args()

    train_xgb = not args.no_xgb

    print("=" * 60)
    print("  Munich Peak Model — Trainer V3 (Ensemble)")
    print("=" * 60)

    print("\n[1/5] A carregar dados...")
    df = load_csv()

    print("\n[2/5] A construir dataset...")
    dataset, prior_map = build_dataset(df)

    # Walk-Forward
    wf_results = None
    if not args.no_wf:
        print("\n[3/5] Walk-Forward Validation...")
        wf_results = walk_forward_train(dataset, train_xgb=train_xgb)

    # Treino final
    print("\n[4/5] Treino final (todos os dados)...")
    lgb, xgb = train_final(dataset, train_xgb=train_xgb)

    # Se não fizemos WF, calcular AUC no treino
    if wf_results is None:
        X = dataset[FEATURE_COLS].fillna(0)
        y = dataset["label"]
        preds_lgb = lgb.predict_proba(X)[:, 1]
        from sklearn.metrics import roc_auc_score
        wf_results = {
            "global_auc_lgb": roc_auc_score(y, preds_lgb),
            "global_auc_xgb": None,
            "global_auc_ens": roc_auc_score(y, preds_lgb),
            "fold_results": [],
        }

    # DOY threshold
    print("\n[5/5] A calcular threshold adaptativo...")
    doy_poly = compute_doy_threshold(dataset, degree=args.doy_degree)

    # Save
    print("\nA guardar modelos...")
    save_models(lgb, xgb, prior_map, doy_poly, wf_results, train_xgb=train_xgb)

    print("\n✅ Treino completo!")
    if train_xgb:
        print(f"   LightGBM  AUC: {wf_results['global_auc_lgb']:.4f}")
        print(f"   XGBoost   AUC: {wf_results.get('global_auc_xgb', 'N/A')}")
        print(f"   Ensemble  AUC: {wf_results['global_auc_ens']:.4f}")
    else:
        print(f"   LightGBM  AUC: {wf_results['global_auc_lgb']:.4f}")


if __name__ == "__main__":
    main()
