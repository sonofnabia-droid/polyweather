"""
optimizer_fast.py
=================
Versão otimizada do optimizer para testar thresholds mais rápido.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

from munich_backtester import (
    load_models, load_data, ZScoreStreaming, SimulatedMarket,
    build_slot, predict_ensemble, ceil_slot, BERLIN_TZ,
    DAY_START, DAY_END, MIN_HOUR, SEASONS, MONTHS_PT,
)

OUTPUT_DIR = Path("backtest_results")


def run_single_test(df, models, sim_market,
                   single_threshold, parcel_size=15.0):
    """Backtest SINGLE com threshold específico."""
    prior_map = models["prior_map"]
    monthly_threshold = models["monthly_threshold"]
    doy_poly = models["doy_poly"]

    def get_threshold(month, doy=0):
        if doy_poly is not None and doy > 0:
            val = float(np.polyval(doy_poly, (doy - 183) / 183))
            return float(np.clip(val, 0.25, 0.95))
        return monthly_threshold.get(month, 0.75)

    daily_max = df.groupby("date")["temp_c"].max()
    dates_s = sorted(daily_max.index)

    from munich_backtester import compute_prev7
    prev7 = compute_prev7(df)

    results = []

    for d, day_df in df.groupby("date"):
        day_df = day_df.sort_values(["hour", "slot30"]).reset_index(drop=True)
        month = int(day_df["month"].iloc[0])
        doy = int(day_df["doy"].iloc[0])
        peak_idx = day_df["temp_c"].idxmax()
        peak_slot = (int(day_df.loc[peak_idx, "hour"]),
                     int(day_df.loc[peak_idx, "slot30"]))
        peak_temp = float(day_df["temp_c"].max())

        slots_so_far = []
        zscore = ZScoreStreaming()
        zscore.reset()

        bought = False
        buy_slot = None

        for _, row in day_df.iterrows():
            h = int(row["hour"])
            s = int(row["slot30"])
            t = float(row["temp_c"])

            slot_entry = {
                "hour": h, "slot30": s, "temp_c": t,
                "cloud_cover": float(row.get("cloud_cover", 50)),
                "humidity": float(row.get("humidity", 70)),
                "dewpoint_c": float(row.get("dewpt_c", t - 10)),
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
                "cloud_cover": float(row.get("cloud_cover", 50)),
                "humidity": float(row.get("humidity", 70)),
                "prev_7d_avg_max": prev7.get(d, peak_temp),
                "temp_c": t,
                "dewpoint_c": float(row.get("dewpt_c", t - 10)),
                "pressure_hpa": float(row.get("pressure_hpa", 1013)),
                "wind_dir_deg": float(row.get("wind_dir_deg", 0)),
                "wind_speed_kmh": float(row.get("wind_speed_kmh", 5)),
                "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                "uv_index": float(row.get("uv_index", 3)),
            }

            p_ens, _, _, _ = predict_ensemble(
                models, slots_so_far, current_extra, month, doy, zscore
            )

            if not bought and p_ens >= single_threshold:
                buy_slot = (h, s)
                bought = True

        def slot_idx(h, s):
            return h * 2 + s // 30

        detected = bought
        lag_first = None
        correct = False
        premature = False

        if detected and buy_slot:
            peak_idx_val = slot_idx(peak_slot[0], peak_slot[1])
            buy_idx_val = slot_idx(buy_slot[0], buy_slot[1])
            lag_first = (buy_idx_val - peak_idx_val) * 0.5

            if lag_first >= 0:
                correct = True
            else:
                premature = True
        else:
            lag_first = None
            correct = False
            premature = False

        season = next(
            (s for s, ms in SEASONS.items() if month in ms),
            "spring")

        results.append({
            "date": d,
            "month": month,
            "doy": doy,
            "season": season,
            "peak_temp": round(peak_temp, 2),
            "peak_h_true": peak_slot[0],
            "detected": detected,
            "correct": correct,
            "premature": premature,
            "missed": not detected,
            "lag_first_h": round(lag_first, 2) if lag_first is not None else None,
            "total_invested": parcel_size if bought else 0,
        })

    return pd.DataFrame(results)


def run_phased_test(df, models, sim_market,
                    thr_p1_min, thr_p1_max, thr_p2, thr_p3,
                    parcel_size=5.0):
    """Backtest PHASED com thresholds específicos."""
    prior_map = models["prior_map"]
    monthly_threshold = models["monthly_threshold"]
    doy_poly = models["doy_poly"]

    daily_max = df.groupby("date")["temp_c"].max()
    dates_s = sorted(daily_max.index)

    from munich_backtester import compute_prev7
    prev7 = compute_prev7(df)

    results = []

    for d, day_df in df.groupby("date"):
        day_df = day_df.sort_values(["hour", "slot30"]).reset_index(drop=True)
        month = int(day_df["month"].iloc[0])
        doy = int(day_df["doy"].iloc[0])
        peak_idx = day_df["temp_c"].idxmax()
        peak_slot = (int(day_df.loc[peak_idx, "hour"]),
                     int(day_df.loc[peak_idx, "slot30"]))
        peak_temp = float(day_df["temp_c"].max())

        slots_so_far = []
        zscore = ZScoreStreaming()
        zscore.reset()

        parcel_bought = [False, False, False]
        parcel_records = [None, None, None]

        forecast_agrees = np.random.random() < 0.80

        for _, row in day_df.iterrows():
            h = int(row["hour"])
            s = int(row["slot30"])
            t = float(row["temp_c"])

            slot_entry = {
                "hour": h, "slot30": s, "temp_c": t,
                "cloud_cover": float(row.get("cloud_cover", 50)),
                "humidity": float(row.get("humidity", 70)),
                "dewpoint_c": float(row.get("dewpt_c", t - 10)),
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
                "cloud_cover": float(row.get("cloud_cover", 50)),
                "humidity": float(row.get("humidity", 70)),
                "prev_7d_avg_max": prev7.get(d, peak_temp),
                "temp_c": t,
                "dewpoint_c": float(row.get("dewpt_c", t - 10)),
                "pressure_hpa": float(row.get("pressure_hpa", 1013)),
                "wind_dir_deg": float(row.get("wind_dir_deg", 0)),
                "wind_speed_kmh": float(row.get("wind_speed_kmh", 5)),
                "wind_gust_kmh": float(row.get("wind_gust_kmh", 8)),
                "uv_index": float(row.get("uv_index", 3)),
            }

            p_ens, _, _, _ = predict_ensemble(
                models, slots_so_far, current_extra, month, doy, zscore
            )

            running_max = max(sl["temp_c"] for sl in slots_so_far)

            brackets = sim_market.get_simulated_brackets(p_ens, running_max, h)
            market_sim = {"brackets": brackets}

            def find_highest_ask_bracket(market):
                if not market or not market.get("brackets"):
                    return None
                return max(market["brackets"],
                          key=lambda b: b.get("ask") or b.get("price") or 0)

            def market_confirms_model(market, running_max, tolerance=1):
                best = find_highest_ask_bracket(market)
                if best is None:
                    return False, "sem mercado"
                best_ask = best.get("ask") or best.get("price") or 0
                best_lo, best_hi = best["temp_lo"], best["temp_hi"]
                rmax_int = int(round(running_max))
                if best_lo <= -99:
                    return False, f"mercado={best['label']}"
                if best_lo <= rmax_int <= best_hi:
                    return True, f"mercado={best['label']} ({best_ask*100:.0f}¢) = {rmax_int}°C"
                mid = best_lo if best_hi >= 99 else (best_lo + best_hi) / 2
                if abs(mid - rmax_int) <= tolerance:
                    return True, f"mercado={best['label']} ({best_ask*100:.0f}¢) ≈ {rmax_int}°C"
                return False, f"mercado={best['label']} ({best_ask*100:.0f}¢) ≠ {rmax_int}°C"

            # P1
            if not parcel_bought[0]:
                in_morning = 10 <= h < 12
                fc_ok = forecast_agrees
                mkt_ok, _ = market_confirms_model(market_sim, running_max)
                model_in_range = thr_p1_min <= p_ens <= thr_p1_max

                if in_morning and fc_ok and mkt_ok and model_in_range:
                    best = find_highest_ask_bracket(market_sim)
                    if best:
                        parcel_bought[0] = True
                        parcel_records[0] = {"hour": h, "slot30": s}

            # P2
            if not parcel_bought[1]:
                mkt_ok, _ = market_confirms_model(market_sim, running_max)
                if p_ens >= thr_p2 and mkt_ok:
                    rmax_int = int(round(running_max))
                    best = next(
                        (b for b in brackets
                         if b["temp_lo"] <= rmax_int <= b["temp_hi"]),
                        max(brackets, key=lambda b: b["ask"]))
                    if best:
                        parcel_bought[1] = True
                        parcel_records[1] = {"hour": h, "slot30": s}

            # P3
            if not parcel_bought[2] and p_ens >= thr_p3:
                rmax_int = int(round(running_max))
                best = next(
                    (b for b in brackets
                     if b["temp_lo"] <= rmax_int <= b["temp_hi"]),
                    max(brackets, key=lambda b: b["ask"]))
                if best:
                    parcel_bought[2] = True
                    parcel_records[2] = {"hour": h, "slot30": s}

        def slot_idx(h, s):
            return h * 2 + s // 30

        peak_idx_val = slot_idx(peak_slot[0], peak_slot[1])

        parcel_lags = []
        for i in range(3):
            if parcel_bought[i] and parcel_records[i] is not None:
                lag = (slot_idx(parcel_records[i]["hour"], parcel_records[i]["slot30"])
                       - peak_idx_val)
                parcel_lags.append(lag)
            else:
                parcel_lags.append(None)

        detected = any(parcel_bought)

        first_slot = next(
            (parcel_records[i] for i in range(3)
             if parcel_bought[i] and parcel_records[i] is not None),
            None)

        lag_first = None
        if first_slot is not None:
            lag_first = (slot_idx(first_slot["hour"], first_slot["slot30"])
                         - peak_idx_val)

        correct_lags = [l for l in parcel_lags if l is not None and l >= 0]
        premature_lags = [l for l in parcel_lags if l is not None and l < 0]

        season = next(
            (s for s, ms in SEASONS.items() if month in ms),
            "spring")

        results.append({
            "date": d,
            "month": month,
            "doy": doy,
            "season": season,
            "peak_temp": round(peak_temp, 2),
            "peak_h_true": peak_slot[0],
            "detected": detected,
            "n_parcels": sum(parcel_bought),
            "parcel1_bought": parcel_bought[0],
            "parcel2_bought": parcel_bought[1],
            "parcel3_bought": parcel_bought[2],
            "parcel1_lag": parcel_lags[0],
            "parcel2_lag": parcel_lags[1],
            "parcel3_lag": parcel_lags[2],
            "lag_first_h": round(lag_first * 0.5, 1) if lag_first is not None else None,
            "correct": (len(correct_lags) > 0 and len(premature_lags) == 0),
            "premature": len(premature_lags) > 0,
            "missed": not detected,
            "total_invested": sum(parcel_size for b in parcel_bought if b),
        })

    return pd.DataFrame(results)


def compute_metrics(results):
    n = len(results)
    corr = results["correct"].sum()
    prem = results["premature"].sum()
    miss = results["missed"].sum()

    correct_lags_h = (results[results["correct"]
                      & results["lag_first_h"].notna()]
                      ["lag_first_h"].values)

    m = {
        "correct_pct": round(corr / n * 100, 1),
        "premature_pct": round(prem / n * 100, 1),
        "missed_pct": round(miss / n * 100, 1),
        "lag_mean_h": round(float(np.mean(correct_lags_h)), 2)
        if len(correct_lags_h) else None,
        "avg_invested": round(results["total_invested"].mean(), 2),
    }

    for season in SEASONS:
        sub = results[results["season"] == season]
        if not sub.empty:
            m[f"{season}_correct_pct"] = round(
                sub["correct"].mean() * 100, 1)

    return m


def main():
    parser = argparse.ArgumentParser(description="Optimizer de Thresholds Rápido")
    parser.add_argument("--start", default="2024-01-01",
                        help="Data início (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="Data fim (YYYY-MM-DD), default: ontem")
    parser.add_argument("--single", action="store_true",
                        help="Testar apenas modo SINGLE")
    parser.add_argument("--phased", action="store_true",
                        help="Testar apenas modo PHASED")
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start).date()
    end_date = (pd.to_datetime(args.end).date() if args.end
                 else date.today() - timedelta(days=1))

    mode = "both"
    if args.single and args.phased:
        mode = "both"
    elif args.single:
        mode = "single"
    elif args.phased:
        mode = "phased"

    print(f"\n  Optimizer Rápido: {start_date} → {end_date}  modo={mode}")

    print("\n[1/4] Modelos...")
    models = load_models()

    print("\n[2/4] Dados...")
    df_all = load_data()
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df = df_all[(df_all["date"] >= start_date)
                & (df_all["date"] <= end_date)].copy()
    print(f"  {len(df):,} slots no intervalo")

    print("\n[3/4] Simulador...")
    sim_market = SimulatedMarket()

    print("\n[4/4] Testando thresholds...")

    all_results = []

    # SINGLE tests
    if mode in ("both", "single"):
        print("\n  === MODO SINGLE ===")
        single_thresholds = [0.75, 0.80, 0.85]

        for i, thr in enumerate(single_thresholds):
            print(f"    [{i+1}/{len(single_thresholds)}] threshold={thr:.0%}...", end=" ", flush=True)
            results = run_single_test(df, models, sim_market, thr)
            m = compute_metrics(results)
            print(f"correct={m['correct_pct']}% "
                  f"prem={m['premature_pct']}% "
                  f"invest=${m['avg_invested']:.2f}")

            all_results.append({
                "mode": "single",
                "threshold": thr,
                **m,
            })

    # PHASED tests - COMBINAÇÕES REDUZIDAS
    if mode in ("both", "phased"):
        print("\n  === MODO PHASED (versão rápida) ===")
        p1_mins = [0.25, 0.30, 0.35]  # Reduzido de 3 para 3
        p1_maxes = [0.60, 0.65, 0.70]  # Reduzido de 3 para 3
        p2_thrs = [0.65, 0.70, 0.75, 0.80]  # Mantido
        p3_thrs = [0.80, 0.85, 0.90]  # Mantido

        total = len(p1_mins) * len(p1_maxes) * len(p2_thrs) * len(p3_thrs)
        print(f"    Testing {total} combinations...")

        count = 0
        for thr_p1_min, thr_p1_max, thr_p2, thr_p3 in product(
                p1_mins, p1_maxes, p2_thrs, p3_thrs):
            count += 1
            print(f"    [{count}/{total}] P1:{thr_p1_min:.0%}-{thr_p1_max:.0%} P2:{thr_p2:.0%} P3:{thr_p3:.0%}...", end=" ", flush=True)
            results = run_phased_test(
                df, models, sim_market,
                thr_p1_min, thr_p1_max, thr_p2, thr_p3
            )
            m = compute_metrics(results)
            print(f"correct={m['correct_pct']}% prem={m['premature_pct']}%")

            all_results.append({
                "mode": "phased",
                "thr_p1_min": thr_p1_min,
                "thr_p1_max": thr_p1_max,
                "thr_p2": thr_p2,
                "thr_p3": thr_p3,
                **m,
            })

    # Save results
    print("\n[5/5] Guardando resultados...")
    results_df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "optimizer_results_fast.csv"
    results_df.to_csv(out_path, index=False)
    print(f"  {out_path}")

    # Print best configs
    print("\n  === MELHORES CONFIGURAÇÕES ===")

    print("\n  SINGLE:")
    single_df = results_df[results_df["mode"] == "single"]
    if not single_df.empty:
        # Balance score: correct * 1.5 - premature * 2 - (avg_invested / 20)
        single_df["score"] = (
            single_df["correct_pct"] * 1.5
            - single_df["premature_pct"] * 2
            - single_df["avg_invested"] / 20
        )
        best = single_df.loc[single_df["score"].idxmax()]
        print(f"    Threshold: {best['threshold']:.0%}")
        print(f"    Correcto: {best['correct_pct']}% | "
              f"Prematuro: {best['premature_pct']}%")
        print(f"    Investimento: ${best['avg_invested']:.2f}")

    print("\n  PHASED:")
    phased_df = results_df[results_df["mode"] == "phased"]
    if not phased_df.empty:
        # Balance score
        phased_df["score"] = (
            phased_df["correct_pct"] * 1.5
            - phased_df["premature_pct"] * 2
            - phased_df["avg_invested"] / 20
        )
        best = phased_df.loc[phased_df["score"].idxmax()]
        print(f"    P1 range: {best['thr_p1_min']:.0%} - {best['thr_p1_max']:.0%}")
        print(f"    P2: {best['thr_p2']:.0%} | P3: {best['thr_p3']:.0%}")
        print(f"    Correcto: {best['correct_pct']}% | "
              f"Prematuro: {best['premature_pct']}%")
        print(f"    Investimento: ${best['avg_invested']:.2f}")


if __name__ == "__main__":
    main()
