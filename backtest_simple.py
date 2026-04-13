"""
Backtest super simplificado para debug
"""
import json
import time
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from zoneinfo import ZoneInfo
BERLIN_TZ = ZoneInfo("Europe/Berlin")
MODEL_LGB = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
DATA_CSV = Path("historic/munich.csv")

DAY_START = 6
DAY_END = 21
MIN_HOUR = 6


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


def load_data(csv_path=DATA_CSV):
    print("A carregar dados...")
    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","

    raw = pd.read_csv(csv_path, sep=sep, low_memory=False)
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
    raw["date"] = dates
    raw["hour"] = hours
    raw["slot30"] = slots30
    raw["month"] = raw["datetime_local"].dt.month
    raw["doy"] = raw["datetime_local"].dt.dayofyear
    raw["temp_c"] = pd.to_numeric(raw["temp_c"], errors="coerce")

    if "humidity_pct" in raw.columns:
        raw["humidity"] = pd.to_numeric(raw["humidity_pct"], errors="coerce")
    else:
        raw["humidity"] = 70.0

    if "sky_cover" in raw.columns:
        raw["cloud_cover"] = pd.to_numeric(raw["sky_cover"], errors="coerce")
    else:
        raw["cloud_cover"] = 50.0

    raw["dewpoint_c"] = pd.to_numeric(raw.get("dewpt_c"), errors="coerce")
    raw["pressure_hpa"] = pd.to_numeric(raw.get("pressure_hpa"), errors="coerce")
    raw["wind_dir_deg"] = pd.to_numeric(raw.get("wind_dir_deg"), errors="coerce")
    raw["wind_speed_kmh"] = pd.to_numeric(raw.get("wind_speed_kmh"), errors="coerce")
    raw["wind_gust_kmh"] = pd.to_numeric(raw.get("wind_gust_kmh"), errors="coerce")
    raw["uv_index"] = pd.to_numeric(raw.get("uv_index"), errors="coerce")

    raw["dewpoint_c"] = raw["dewpoint_c"].fillna(raw["temp_c"] - 10)
    raw["pressure_hpa"] = raw["pressure_hpa"].fillna(1013.0)
    raw["wind_dir_deg"] = raw["wind_dir_deg"].fillna(0.0)
    raw["wind_speed_kmh"] = raw["wind_speed_kmh"].fillna(5.0)
    raw["wind_gust_kmh"] = raw["wind_gust_kmh"].fillna(8.0)
    raw["uv_index"] = raw["uv_index"].fillna(3.0)

    df = raw[
        (raw["hour"] >= DAY_START) & (raw["hour"] <= DAY_END)
    ].dropna(subset=["temp_c"]).sort_values(
        ["date", "hour", "slot30"]
    ).reset_index(drop=True)

    print(f"  {len(df):,} slots  {df['date'].nunique()} dias")
    return df


def load_models():
    print("A carregar modelos...")
    model_lgb = joblib.load(MODEL_LGB)
    config = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat_cols = config.get("feature_cols", [])

    raw_prior = config.get("seasonal_peak_prior", {})
    prior_map = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    doy_poly_raw = config.get("doy_poly_coeffs")
    doy_poly = np.array(doy_poly_raw, dtype=float) if doy_poly_raw else None

    print(f"  AUC={config.get('global_auc', '?')}  features={len(feat_cols)}")
    return {
        "model_lgb": model_lgb,
        "feat_cols": feat_cols,
        "prior_map": prior_map,
        "doy_poly": doy_poly,
    }


def compute_prev7(df):
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


def build_slot(slots_so_far, current, month, doy, prior_map):
    vals = [s["temp_c"] for s in slots_so_far]
    hums = [s.get("humidity", 70) for s in slots_so_far]
    n = len(vals)
    cur = vals[-1]
    hour = current["hour"]
    slot30 = current["slot30"]
    cloud = float(current.get("cloud_cover", 50))
    hu = float(current.get("humidity", 70))

    def lag(k): return vals[-k] if n >= k else vals[0]
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
        slope = (float((_x * np.array(slope_w)).sum() / _denom) if _denom > 0 else 0.0)
    else:
        slope = 0.0

    plat_w = vals[-6:] if n >= 6 else vals
    plateau = 1.0 if (np.std(plat_w) < 0.4 and n >= 4) else 0.0
    radiation = float(np.cos((slot_frac - 0.5) * 2 * np.pi)) * (1 - cloud / 100)
    hum_drop = lagh(3) - hums[-1] if n >= 3 else 0.0
    prev7 = current["prev_7d_avg_max"]

    dewpt = current.get("dewpoint_c", cur - 10)
    pres = current.get("pressure_hpa", 1013)
    wdir = current.get("wind_dir_deg", 0)
    wspd = current.get("wind_speed_kmh", 5)
    wgst = current.get("wind_gust_kmh", 8)
    uv = current.get("uv_index", 3)
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
    foehn_dry = max(0.0, (80 - hu) / 20.0) if hu < 80 else 0.0
    foehn_indicator = foehn_south * (0.4 + 0.3 * foehn_gusty + 0.3 * foehn_dry)

    return {
        "slot_frac": slot_frac,
        "doy_sin": float(np.sin(2 * np.pi * doy / 365)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365)),
        "temp_c": cur,
        "running_max": rmax,
        "temp_vs_climatology": cur - prev7,
        "delta_30m": cur - lag(2),
        "delta_1h": cur - lag(3),
        "accel": (cur - lag(2)) - (lag(2) - lag(3)),
        "recent_slope": slope,
        "temp_lag_3": lag(4),
        "roll3_std": float(np.std(vals[-3:])) if n >= 3 else 0.0,
        "plateau_indicator": plateau,
        "morning_max": mmax,
        "radiation_proxy": radiation,
        "humidity_drop_1h": hum_drop,
        "prev_7d_avg_max": prev7,
        "seasonal_peak_prior": prior,
        "dewpoint_c": dewpt,
        "temp_to_dewpoint_gap": temp_to_dewpoint_gap,
        "pressure_trend_3h": pressure_trend_3h,
        "wind_south_proxy": wind_south_proxy,
        "wind_speed_kmh": wspd,
        "uv_index": uv,
        "foehn_indicator": foehn_indicator,
    }


def predict_simple(models, slots_so_far, current, month, doy):
    hour = current["hour"]
    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0

    feat_cols = models["feat_cols"]
    feat = build_slot(slots_so_far, current, month, doy, models["prior_map"])
    avail = [f for f in feat_cols if f in feat]
    X = pd.DataFrame([feat])[avail].fillna(0)

    p_lgbm = float(models["model_lgb"].predict_proba(X)[0, 1])
    return p_lgbm


def run_backtest(df, models, year=None):
    print(f"\nBacktest para {year if year else 'todos os anos'}...")

    def get_threshold(month, doy=0):
        if models["doy_poly"] is not None and doy > 0:
            val = float(np.polyval(models["doy_poly"], (doy - 183) / 183))
            return float(np.clip(val, 0.25, 0.95))
        return 0.75

    prev7 = compute_prev7(df)
    results = []

    start_time = time.time()

    for d, day_df in df.groupby("date"):
        day_df = day_df.sort_values(["hour", "slot30"]).reset_index(drop=True)
        month = int(day_df["month"].iloc[0])
        doy = int(day_df["doy"].iloc[0])

        peak_idx = day_df["temp_c"].idxmax()
        peak_slot = (int(day_df.loc[peak_idx, "hour"]), int(day_df.loc[peak_idx, "slot30"]))
        peak_temp = float(day_df["temp_c"].max())

        slots_so_far = []
        bought = False
        buy_slot = None

        for _, row in day_df.iterrows():
            h = int(row["hour"])
            s = int(row["slot30"])
            t = float(row["temp_c"])
            cl = float(row["cloud_cover"])
            hu = float(row["humidity"])

            slot_entry = {
                "hour": h, "slot30": s, "temp_c": t,
                "cloud_cover": cl, "humidity": hu,
                "dewpoint_c": float(row.get("dewpoint_c", t - 10)),
                "pressure_hpa": float(row.get("pressure_hpa", 1013)),
                "wind_dir_deg": float(row.get("wind_dir_deg", 0)),
                "wind_speed_kmh": float(row.get("wind_speed_kmh", 5)),
                "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                "uv_index": float(row.get("uv_index", 3)),
            }
            slots_so_far.append(slot_entry)

            if h < MIN_HOUR or len(slots_so_far) < 4:
                continue

            current_extra = {
                "hour": h, "slot30": s,
                "cloud_cover": cl, "humidity": hu,
                "prev_7d_avg_max": prev7.get(d, peak_temp),
                "temp_c": t,
                "dewpoint_c": float(row.get("dewpoint_c", t - 10)),
                "pressure_hpa": float(row.get("pressure_hpa", 1013)),
                "wind_dir_deg": float(row.get("wind_dir_deg", 0)),
                "wind_speed_kmh": float(row.get("wind_speed_kmh", 5)),
                "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                "uv_index": float(row.get("uv_index", 3)),
            }

            p_ens = predict_simple(models, slots_so_far, current_extra, month, doy)

            # SINGLE: comprar quando p_ens >= 0.85 (threshold muito alto)
            if p_ens >= 0.85 and not bought:
                bought = True
                buy_slot = (h, s)

        # Resultado do dia
        def slot_idx(h, s):
            return h * 2 + s // 30

        peak_idx_val = slot_idx(peak_slot[0], peak_slot[1])
        correct = False
        lag = None

        if bought:
            buy_idx = slot_idx(buy_slot[0], buy_slot[1])
            lag = (buy_idx - peak_idx_val) * 0.5  # slots de 30min
            correct = lag >= 0

        results.append({
            "date": d,
            "month": month,
            "peak_temp": peak_temp,
            "detected": bought,
            "correct": correct,
            "lag_h": lag,
        })

        if len(results) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {len(results)} dias em {elapsed:.1f}s")

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest simples")
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    models = load_models()
    df = load_data()

    if args.year:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year == args.year].copy()
        df["date"] = df["date"].dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    results = run_backtest(df, models, args.year)

    # Métricas
    n = len(results)
    corr = results["correct"].sum()
    prem = ((results["detected"]) & (~results["correct"])).sum()
    miss = (~results["detected"]).sum()

    correct_lags = results[results["correct"] & results["lag_h"].notna()]["lag_h"].values

    print(f"\n=== RESULTADOS {'ANO ' + str(args.year) if args.year else 'TODOS OS ANOS'} ===")
    print(f"Dias analisados: {n}")
    print(f"Correcto: {corr/n*100:.1f}% ({corr})")
    print(f"Prematuro: {prem/n*100:.1f}% ({prem})")
    print(f"Não detectado: {miss/n*100:.1f}% ({miss})")

    if len(correct_lags) > 0:
        print(f"Lag médio (correctos): {np.mean(correct_lags):.2f}h")
        print(f"Lag ≤ 1h: {(correct_lags <= 1).mean()*100:.1f}%")
        print(f"Lag ≤ 2h: {(correct_lags <= 2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
