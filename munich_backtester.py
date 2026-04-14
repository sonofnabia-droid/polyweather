"""
munich_backtester.py
====================
Backtest V3: Ensemble (LGBM + XGB + Z-Score) + Phased(3) ou Single(1).

Modos:
  --mode phased  → 3 parcelas $5 (P1 manhã invertida, P2 dupla, P3 alta)
  --mode single  → 1 compra $5 quando p_ensemble >= 75%

Stop-Loss por Bracket:
  Se a temperatura subir 1°C acima do tecto do bracket comprado,
  esse bracket nunca resolve YES → simular venda ao bid (perda parcial).

PnL:
  WIN  → payoff = $1/share × shares  (resolve YES)
  STOP → payoff = bid_exit × shares  (stop-loss acionado)
  LOSS → payoff = $0                 (resolve NO, sem stop)

Métricas:
  PnL total/%, Win rate, Sharpe ratio diário, Sortino ratio diário

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

# Tamanho fixo por parcela
PARCEL_SIZE_USDC = 5.0

# Stop-loss: vender se temp subir N graus acima do tecto do bracket
STOP_LOSS_DEGREES = 1.0

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
        raw["humidity"] = pd.to_numeric(raw["humidity_pct"], errors="coerce")
    else:
        raw["humidity"] = 70.0

    if "sky_cover" in raw.columns:
        raw["cloud_cover"] = pd.to_numeric(raw["sky_cover"], errors="coerce")
    else:
        raw["cloud_cover"] = 50.0

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

    get_bid(): simula o bid para exit no stop-loss.
    Spread típico de 3-8% em Polymarket.
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

            # Bid simulado: spread de ~5% (realista para Polymarket)
            spread = float(np.random.uniform(0.03, 0.08))
            bid    = float(np.clip(ask - spread, 0.01, ask - 0.01))

            is_last  = (temp == self.temp_range[-1])
            is_first = (temp == self.temp_range[0])

            if is_last:
                label, lo, hi = (f"{temp}°C or higher", float(temp), 99.0)
            elif is_first:
                label, lo, hi = (f"{temp}°C or lower", -99.0, float(temp))
            else:
                label, lo, hi = f"{temp}°C", float(temp), float(temp)

            brackets.append({
                "label":   label,
                "ask":     round(ask, 4),
                "bid":     round(bid, 4),
                "price":   round(ask, 4),
                "temp_lo": lo,
                "temp_hi": hi,
            })

        return brackets

    def get_bid_for_bracket(self, bracket: dict, p_ensemble: float,
                             running_max: float, hour: int) -> float:
        """
        Retorna um bid simulado para o bracket específico.
        Usado no stop-loss: quando a temp ultrapassou o tecto do bracket,
        o mercado já sabe que vai resolver NO → bid muito baixo.
        """
        all_brackets = self.get_simulated_brackets(p_ensemble, running_max, hour)
        label = bracket.get("label", "")
        for b in all_brackets:
            if b["label"] == label:
                # Quando o stop-loss é acionado, o bracket já está "out of the money"
                # O bid fica ainda mais baixo (mercado já sabe que perdeu)
                degraded_bid = b["bid"] * float(np.random.uniform(0.3, 0.6))
                return float(np.clip(degraded_bid, 0.01, 0.50))
        # Fallback: bid muito baixo
        return float(np.random.uniform(0.02, 0.08))


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

    dewpt = current.get("dewpoint_c", cur - 10)
    pres  = current.get("pressure_hpa", 1013)
    wdir  = current.get("wind_dir_deg", 0)
    wspd  = current.get("wind_speed_kmh", 5)
    wgst  = current.get("wind_gust_kmh", 8)
    uv    = current.get("uv_index", 3)

    temp_to_dewpoint_gap = max(0.0, cur - dewpt)

    press_vals = [s.get("pressure_hpa", 1013) for s in slots_so_far[-6:]]
    pressure_trend_3h = (press_vals[-1] - press_vals[0]
                         if len(press_vals) >= 2 else 0.0)

    if 135 <= wdir <= 225:
        wind_south_proxy = 1.0 - abs(wdir - 180) / 45.0
    else:
        wind_south_proxy = 0.0

    foehn_south = 1.0 if 135 <= wdir <= 225 else 0.0
    foehn_gusty = min(1.0, wgst / 30.0)
    foehn_dry   = max(0.0, (80 - hu) / 20.0) if hu < 80 else 0.0
    foehn_indicator = foehn_south * (0.4 + 0.3 * foehn_gusty + 0.3 * foehn_dry)

    return {
        "slot_frac":            slot_frac,
        "doy_sin":              float(np.sin(2 * np.pi * doy / 365)),
        "doy_cos":              float(np.cos(2 * np.pi * doy / 365)),
        "temp_c":               cur,
        "running_max":          rmax,
        "temp_vs_climatology":  cur - prev7,
        "delta_30m":            cur - lag(2),
        "delta_1h":             cur - lag(3),
        "accel":                (cur - lag(2)) - (lag(2) - lag(3)),
        "recent_slope":         slope,
        "temp_lag_3":           lag(4),
        "roll3_std":            float(np.std(vals[-3:])) if n >= 3 else 0.0,
        "plateau_indicator":    plateau,
        "morning_max":          mmax,
        "radiation_proxy":      radiation,
        "humidity_drop_1h":     hum_drop,
        "prev_7d_avg_max":      prev7,
        "seasonal_peak_prior":  prior,
        "dewpoint_c":           dewpt,
        "temp_to_dewpoint_gap": temp_to_dewpoint_gap,
        "pressure_trend_3h":    pressure_trend_3h,
        "wind_south_proxy":     wind_south_proxy,
        "wind_speed_kmh":       wspd,
        "uv_index":             uv,
        "foehn_indicator":      foehn_indicator,
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
#  STOP-LOSS CHECKER
# ══════════════════════════════════════════════════════
def check_stop_loss(bracket_record: dict, current_temp: float) -> bool:
    """
    Verifica se o stop-loss deve ser acionado para um bracket comprado.

    Lógica:
      - O bracket tem um tecto (temp_hi).
      - Se temp_hi == 99 (bracket "X°C or higher"), nunca há stop-loss
        (temperatura mais alta = melhor para esse bracket).
      - Se current_temp > bracket.temp_hi + STOP_LOSS_DEGREES → stop-loss.

    Args:
        bracket_record: dict com 'temp_hi', 'ask', 'label' do bracket comprado
        current_temp:   temperatura actual no slot

    Returns:
        True se o stop-loss deve ser acionado
    """
    temp_hi = bracket_record.get("temp_hi", 99.0)
    # Bracket "X°C or higher" nunca sofre stop-loss por subida
    if temp_hi >= 99.0:
        return False
    return current_temp > (temp_hi + STOP_LOSS_DEGREES)


def compute_parcel_pnl(parcel_record: dict, peak_temp: float,
                       sim_market: SimulatedMarket,
                       p_ens_at_exit: float = 0.5) -> dict:
    """
    Calcula o PnL de uma parcela comprada.

    Outcomes:
      - STOP_LOSS : temp ultrapassou tecto+1°C antes do fecho → venda ao bid simulado
      - WIN       : bracket resolveu YES → payoff $1/share
      - LOSS      : bracket resolveu NO (temp final fora do bracket) → payoff $0

    Returns:
        dict com 'outcome', 'pnl_usdc', 'payoff_per_share', 'exit_price'
    """
    if parcel_record is None:
        return {"outcome": "no_trade", "pnl_usdc": 0.0,
                "payoff_per_share": 0.0, "exit_price": 0.0}

    ask        = parcel_record.get("ask", 0.5)
    size_usdc  = parcel_record.get("size_usdc", PARCEL_SIZE_USDC)
    shares     = size_usdc / ask if ask > 0 else 0.0
    temp_lo    = parcel_record.get("temp_lo", -99.0)
    temp_hi    = parcel_record.get("temp_hi", 99.0)
    stop_exit  = parcel_record.get("stop_loss_exit_price")

    # 1. Stop-loss foi acionado durante o dia
    if stop_exit is not None:
        exit_price = stop_exit
        payoff     = exit_price * shares
        pnl        = payoff - size_usdc
        return {
            "outcome":          "stop_loss",
            "pnl_usdc":         round(pnl, 4),
            "payoff_per_share": round(exit_price, 4),
            "exit_price":       round(exit_price, 4),
            "shares":           round(shares, 4),
            "invested":         round(size_usdc, 4),
        }

    # 2. Sem stop-loss: avaliar resolução do mercado
    peak_int = int(round(peak_temp))

    # Bracket resolveu YES?
    if temp_hi >= 99.0:
        resolves_yes = (peak_int >= int(temp_lo))
    elif temp_lo <= -99.0:
        resolves_yes = (peak_int <= int(temp_hi))
    else:
        resolves_yes = (int(temp_lo) <= peak_int <= int(temp_hi))

    if resolves_yes:
        exit_price = 1.0  # resolve YES → $1/share
        payoff     = 1.0 * shares
        pnl        = payoff - size_usdc
        return {
            "outcome":          "win",
            "pnl_usdc":         round(pnl, 4),
            "payoff_per_share": 1.0,
            "exit_price":       1.0,
            "shares":           round(shares, 4),
            "invested":         round(size_usdc, 4),
        }
    else:
        # Resolve NO → perda total
        return {
            "outcome":          "loss",
            "pnl_usdc":         round(-size_usdc, 4),
            "payoff_per_share": 0.0,
            "exit_price":       0.0,
            "shares":           round(shares, 4),
            "invested":         round(size_usdc, 4),
        }


# ══════════════════════════════════════════════════════
#  RUN BACKTEST
# ══════════════════════════════════════════════════════
def run(df, models, sim_market, mode="phased"):
    """
    Loop principal do backtest.

    Por cada slot do dia:
      1. Calcular p_ensemble
      2. Avaliar se comprar (via PhasedEntry / SingleEntry)
      3. Verificar stop-loss para todas as parcelas abertas
      4. No fim do dia: calcular PnL de cada parcela
    """
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
                entry = SingleEntry(parcel_size=PARCEL_SIZE_USDC)
            else:
                entry = PhasedEntry(parcel_size=PARCEL_SIZE_USDC)

            forecast_agrees = np.random.random() < 0.80

            # Rastrear stop-loss por parcela:
            # stop_loss_triggered[i] = True se já foi acionado
            stop_loss_triggered = [False, False, False]

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

                # ── CHECK STOP-LOSS para parcelas já compradas ──────
                for pidx in range(3):
                    if (entry.parcel_bought[pidx]
                            and entry.parcel_records[pidx] is not None
                            and not stop_loss_triggered[pidx]
                            and entry.parcel_records[pidx].get("stop_loss_exit_price") is None):

                        rec = entry.parcel_records[pidx]
                        if check_stop_loss(rec, t):
                            # Simular venda ao bid degradado
                            exit_bid = sim_market.get_bid_for_bracket(
                                rec, p_ens, running_max, h)
                            rec["stop_loss_exit_price"] = exit_bid
                            rec["stop_loss_hour"]       = h
                            rec["stop_loss_temp"]       = t
                            stop_loss_triggered[pidx]   = True

                # ── Simular mercado e avaliar parcelas ──────────────
                brackets   = sim_market.get_simulated_brackets(
                    p_ens, running_max, h)
                market_sim = {"brackets": brackets}
                fc_agreement = {"valid": forecast_agrees}

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
                            "hour":          h,
                            "slot30":        s,
                            "ask":           best["ask"],
                            "bid":           best.get("bid", best["ask"] * 0.95),
                            "size_usdc":     act["size_usdc"],
                            "bracket_label": best["label"],
                            "temp_lo":       best["temp_lo"],
                            "temp_hi":       best["temp_hi"],
                            "temp_at_buy":   t,
                            # Stop-loss será preenchido acima se acionado
                            "stop_loss_exit_price": None,
                        })

            # ── Calcular PnL de cada parcela no fim do dia ──────────
            parcel_pnls = []
            for pidx in range(3):
                if (entry.parcel_bought[pidx]
                        and entry.parcel_records[pidx] is not None):
                    pnl_info = compute_parcel_pnl(
                        entry.parcel_records[pidx],
                        peak_temp,
                        sim_market,
                        p_ens_at_exit=p_ens,
                    )
                    parcel_pnls.append(pnl_info)
                else:
                    parcel_pnls.append(
                        {"outcome": "no_trade", "pnl_usdc": 0.0,
                         "payoff_per_share": 0.0, "exit_price": 0.0,
                         "shares": 0.0, "invested": 0.0})

            total_invested = sum(p["invested"] for p in parcel_pnls)
            total_pnl      = sum(p["pnl_usdc"] for p in parcel_pnls)
            total_payoff   = sum(
                p["exit_price"] * p["shares"] for p in parcel_pnls)

            n_wins      = sum(1 for p in parcel_pnls if p["outcome"] == "win")
            n_losses    = sum(1 for p in parcel_pnls if p["outcome"] == "loss")
            n_stops     = sum(1 for p in parcel_pnls if p["outcome"] == "stop_loss")
            n_traded    = sum(1 for p in parcel_pnls if p["outcome"] != "no_trade")

            # ── Métricas de timing (compatíveis com código original) ──
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
                "total_invested":  round(total_invested, 4),
                # ── PnL ──
                "total_pnl":       round(total_pnl, 4),
                "total_payoff":    round(total_payoff, 4),
                "n_wins":          n_wins,
                "n_losses":        n_losses,
                "n_stops":         n_stops,
                "n_traded":        n_traded,
                # Outcomes por parcela
                "p1_outcome":      parcel_pnls[0]["outcome"],
                "p2_outcome":      parcel_pnls[1]["outcome"],
                "p3_outcome":      parcel_pnls[2]["outcome"],
                "p1_pnl":          parcel_pnls[0]["pnl_usdc"],
                "p2_pnl":          parcel_pnls[1]["pnl_usdc"],
                "p3_pnl":          parcel_pnls[2]["pnl_usdc"],
            })

    return pd.DataFrame(results), pd.DataFrame(slots_out)


# ══════════════════════════════════════════════════════
#  METRICS — inclui PnL, Win/Loss, Sharpe, Sortino
# ══════════════════════════════════════════════════════
def compute_metrics(results):
    n    = len(results)
    corr = results["correct"].sum()
    prem = results["premature"].sum()
    miss = results["missed"].sum()

    correct_lags_h = (results[results["correct"]
                      & results["lag_first_h"].notna()]
                      ["lag_first_h"].values)

    # ── PnL ──────────────────────────────────────────
    total_invested = float(results["total_invested"].sum())
    total_pnl      = float(results["total_pnl"].sum())
    pnl_pct        = (total_pnl / total_invested * 100
                      if total_invested > 0 else 0.0)

    # Win/Loss/Stop por parcela individual
    all_outcomes = (
        list(results["p1_outcome"]) +
        list(results["p2_outcome"]) +
        list(results["p3_outcome"])
    )
    n_win_bets  = sum(1 for o in all_outcomes if o == "win")
    n_loss_bets = sum(1 for o in all_outcomes if o == "loss")
    n_stop_bets = sum(1 for o in all_outcomes if o == "stop_loss")
    n_total_bets = n_win_bets + n_loss_bets + n_stop_bets
    win_rate    = (n_win_bets / n_total_bets * 100
                   if n_total_bets > 0 else 0.0)

    # ── Sharpe e Sortino (diários) ────────────────────
    # Usar apenas dias com pelo menos 1 aposta
    traded_days = results[results["n_traded"] > 0]["total_pnl"].values

    if len(traded_days) > 1:
        mean_pnl = np.mean(traded_days)
        std_pnl  = np.std(traded_days, ddof=1)

        # Sharpe (anualizado, base = dias de trading)
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)
                  if std_pnl > 0 else 0.0)

        # Sortino: só downside deviation (perdas abaixo de 0)
        downside = traded_days[traded_days < 0]
        if len(downside) > 1:
            downside_std = np.std(downside, ddof=1)
            sortino = (mean_pnl / downside_std * np.sqrt(252)
                       if downside_std > 0 else np.inf)
        else:
            sortino = np.inf  # Sem dias negativos suficientes
    else:
        sharpe  = 0.0
        sortino = 0.0

    # Avg win / avg loss
    all_pnls = (
        list(results["p1_pnl"]) +
        list(results["p2_pnl"]) +
        list(results["p3_pnl"])
    )
    win_pnls  = [p for p in all_pnls if p > 0]
    loss_pnls = [p for p in all_pnls if p < 0]
    avg_win   = float(np.mean(win_pnls))  if win_pnls  else 0.0
    avg_loss  = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    profit_factor = (abs(sum(win_pnls)) / abs(sum(loss_pnls))
                     if sum(loss_pnls) != 0 else np.inf)

    # Max drawdown (cumulativo)
    cumulative = np.cumsum(results["total_pnl"].values)
    peak_cum   = np.maximum.accumulate(cumulative)
    drawdown   = cumulative - peak_cum
    max_dd     = float(drawdown.min())

    m = {
        "n_days":         n,
        "correct_pct":    round(corr / n * 100, 1),
        "premature_pct":  round(prem / n * 100, 1),
        "missed_pct":     round(miss / n * 100, 1),
        "lag_mean_h":     (round(float(np.mean(correct_lags_h)), 2)
                           if len(correct_lags_h) else None),
        "lag_median_h":   (round(float(np.median(correct_lags_h)), 2)
                           if len(correct_lags_h) else None),
        "lag_le1h_pct":   (round((correct_lags_h <= 1.0).mean() * 100, 1)
                           if len(correct_lags_h) else 0),
        "lag_le2h_pct":   (round((correct_lags_h <= 2.0).mean() * 100, 1)
                           if len(correct_lags_h) else 0),
        "parcel1_pct":    round(results["parcel1_bought"].mean() * 100, 1),
        "parcel2_pct":    round(results["parcel2_bought"].mean() * 100, 1),
        "parcel3_pct":    round(results["parcel3_bought"].mean() * 100, 1),
        "avg_n_parcels":  round(results["n_parcels"].mean(), 2),
        "avg_invested":   round(results["total_invested"].mean(), 2),
        # ── PnL ──
        "total_invested": round(total_invested, 2),
        "total_pnl":      round(total_pnl, 2),
        "pnl_pct":        round(pnl_pct, 2),
        # ── Win/Loss ──
        "n_win_bets":     n_win_bets,
        "n_loss_bets":    n_loss_bets,
        "n_stop_bets":    n_stop_bets,
        "n_total_bets":   n_total_bets,
        "win_rate":       round(win_rate, 1),
        "avg_win":        round(avg_win, 3),
        "avg_loss":       round(avg_loss, 3),
        "profit_factor":  round(float(profit_factor), 3) if np.isfinite(profit_factor) else 999.0,
        # ── Ratios ──
        "sharpe":         round(float(sharpe), 3),
        "sortino":        round(float(sortino), 3) if np.isfinite(sortino) else 999.0,
        "max_drawdown":   round(max_dd, 2),
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
            m[f"{season}_pnl"] = round(float(sub["total_pnl"].sum()), 2)

    return m


# ══════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════
def plot(results, slots_df, metrics, start_year, mode="phased"):
    print("  A gerar gráficos...")

    BG, PANEL = "#07090D", "#0D1018"
    C = {
        "correct":  "#25BE62",
        "premature":"#F0A500",
        "missed":   "#D93838",
        "blue":     "#4D9EFF",
        "muted":    "#424C64",
        "text":     "#D8DCE8",
        "grid":     "#111520",
        "border":   "#181E2C",
        "purple":   "#A855F7",
        "stop":     "#FF6B35",
        "win":      "#25BE62",
        "loss":     "#D93838",
    }

    plt.rcParams.update({
        "figure.facecolor": BG,  "axes.facecolor":  PANEL,
        "axes.edgecolor":   C["border"], "grid.color": C["grid"],
        "text.color":       C["text"],   "axes.labelcolor": C["muted"],
        "xtick.color":      C["muted"],  "ytick.color": C["muted"],
        "axes.titlecolor":  C["text"],   "legend.facecolor": PANEL,
        "legend.edgecolor": "#252E44",   "font.family": "monospace",
    })

    fig = plt.figure(figsize=(22, 24))
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.60, wspace=0.35)

    months_range = sorted(results["month"].unique())
    m_lbls = [MONTHS_PT[m - 1] for m in months_range]
    x = np.arange(len(months_range))

    # ── 1. Detecção por mês ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Resultado por Mês (Timing)")
    corr_m = results.groupby("month")["correct"].mean() * 100
    prem_m = results.groupby("month")["premature"].mean() * 100
    miss_m = results.groupby("month")["missed"].mean() * 100

    ax1.bar(x, corr_m.reindex(months_range, fill_value=0), 0.6,
            color=C["correct"], alpha=0.88, label="Correcto")
    ax1.bar(x, prem_m.reindex(months_range, fill_value=0), 0.6,
            color=C["premature"], alpha=0.88, label="Prematuro",
            bottom=corr_m.reindex(months_range, fill_value=0))
    bot2 = (corr_m.reindex(months_range, fill_value=0)
            + prem_m.reindex(months_range, fill_value=0))
    ax1.bar(x, miss_m.reindex(months_range, fill_value=0), 0.6,
            color=C["missed"], alpha=0.88, label="Não detectado", bottom=bot2)
    ax1.set_xticks(x); ax1.set_xticklabels(m_lbls, fontsize=8)
    ax1.set_ylabel("% dias"); ax1.set_ylim(0, 110)
    ax1.axhline(80, color=C["muted"], lw=0.8, ls="--", alpha=0.5)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

    # ── 2. Parcelas por mês ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if mode == "phased":
        ax2.set_title("Parcelas Compradas/Mês")
        p1 = results.groupby("month")["parcel1_bought"].mean() * 100
        p2 = results.groupby("month")["parcel2_bought"].mean() * 100
        p3 = results.groupby("month")["parcel3_bought"].mean() * 100
        ax2.bar(x, p1.reindex(months_range, fill_value=0), 0.6,
                color=C["blue"], alpha=0.8, label="P1")
        ax2.bar(x, p2.reindex(months_range, fill_value=0), 0.6,
                color=C["purple"], alpha=0.8, label="P2",
                bottom=p1.reindex(months_range, fill_value=0))
        bot3 = (p1.reindex(months_range, fill_value=0)
                + p2.reindex(months_range, fill_value=0))
        ax2.bar(x, p3.reindex(months_range, fill_value=0), 0.6,
                color=C["correct"], alpha=0.8, label="P3", bottom=bot3)
    else:
        ax2.set_title("Single Buy/Mês")
        p1 = results.groupby("month")["parcel1_bought"].mean() * 100
        ax2.bar(x, p1.reindex(months_range, fill_value=0), 0.6,
                color=C["blue"], alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(m_lbls, fontsize=7)
    ax2.set_ylabel("% dias"); ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. PnL acumulado ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title("PnL Acumulado ($)")
    cum_pnl = results["total_pnl"].cumsum().values
    cum_inv = results["total_invested"].cumsum().values
    days_x  = np.arange(len(cum_pnl))

    ax3.fill_between(days_x, cum_pnl, 0,
                     where=(cum_pnl >= 0), alpha=0.25,
                     color=C["win"], label="_nolegend_")
    ax3.fill_between(days_x, cum_pnl, 0,
                     where=(cum_pnl < 0), alpha=0.25,
                     color=C["loss"], label="_nolegend_")
    ax3.plot(days_x, cum_pnl, color=C["win"], lw=1.5, label="PnL")
    ax3.axhline(0, color=C["muted"], lw=0.8, ls="--")

    pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""
    ax3.set_ylabel("USDC"); ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=8)
    ax3.set_title(
        f"PnL Acumulado  Total: {pnl_sign}${metrics['total_pnl']:.2f}"
        f"  ({pnl_sign}{metrics['pnl_pct']:.1f}%)"
        f"  Max DD: ${metrics['max_drawdown']:.2f}")

    # ── 4. PnL por mês (barras) ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("PnL por Mês ($)")
    pnl_m = results.groupby("month")["total_pnl"].sum()
    vals_pnl = pnl_m.reindex(months_range, fill_value=0)
    bar_cols = [C["win"] if v >= 0 else C["loss"] for v in vals_pnl]
    ax4.bar(m_lbls, vals_pnl, color=bar_cols, alpha=0.85)
    ax4.axhline(0, color=C["muted"], lw=0.8)
    ax4.set_ylabel("USDC"); ax4.grid(True, alpha=0.3, axis="y")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)

    # ── 5. Distribuição outcomes (Win/Loss/Stop) ─────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("Outcomes por Aposta")
    total_bets = metrics["n_total_bets"]
    if total_bets > 0:
        outcomes_data = [
            ("WIN",  metrics["n_win_bets"],  C["win"]),
            ("LOSS", metrics["n_loss_bets"], C["loss"]),
            ("STOP", metrics["n_stop_bets"], C["stop"]),
        ]
        for lbl, cnt, col in outcomes_data:
            pct = cnt / total_bets * 100
            ax5.bar(f"{lbl}\n({cnt})", cnt, color=col, alpha=0.85)
            ax5.text(
                ["WIN", "LOSS", "STOP"].index(lbl),
                cnt + total_bets * 0.01,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=8, color=C["text"])
    ax5.set_ylabel("Nº apostas"); ax5.grid(True, alpha=0.3, axis="y")

    # ── 6. Distribuição PnL por aposta ─────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("Distribuição PnL por Aposta ($)")
    all_pnl_vals = (list(results["p1_pnl"]) +
                    list(results["p2_pnl"]) +
                    list(results["p3_pnl"]))
    all_pnl_vals = [v for v in all_pnl_vals if v != 0]
    if all_pnl_vals:
        bins = np.linspace(min(all_pnl_vals) - 0.5,
                           max(all_pnl_vals) + 0.5, 30)
        cnts, edges = np.histogram(all_pnl_vals, bins=bins)
        for b, cnt in zip(edges[:-1], cnts):
            ax6.bar(b, cnt, width=(edges[1] - edges[0]) * 0.9,
                    color=C["win"] if b >= 0 else C["loss"],
                    alpha=0.80)
        ax6.axvline(0, color=C["text"], lw=1.2, ls="--")
        ax6.axvline(float(np.mean(all_pnl_vals)),
                    color=C["blue"], lw=1.5, ls="--",
                    label=f"Média: ${np.mean(all_pnl_vals):.2f}")
        ax6.legend(fontsize=8)
    ax6.set_xlabel("PnL ($)"); ax6.set_ylabel("Freq.")
    ax6.grid(True, alpha=0.3, axis="y")

    # ── 7. Resumo métricas ────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    mode_tag = "PHASED 3×$5" if mode == "phased" else "SINGLE 1×$5"
    sortino_str = (f"{metrics['sortino']:.2f}"
                   if metrics['sortino'] < 900 else "∞")
    summary_lines = [
        f"Modo: {mode_tag}",
        f"Dias: {metrics['n_days']}",
        "─" * 24,
        f"Win rate  : {metrics['win_rate']:.1f}%",
        f"Apostas W : {metrics['n_win_bets']}",
        f"Apostas L : {metrics['n_loss_bets']}",
        f"Stop-loss : {metrics['n_stop_bets']}",
        "─" * 24,
        f"PnL total : ${metrics['total_pnl']:+.2f}",
        f"PnL %     : {metrics['pnl_pct']:+.1f}%",
        f"Avg WIN   : ${metrics['avg_win']:+.3f}",
        f"Avg LOSS  : ${metrics['avg_loss']:+.3f}",
        f"Prof.Fact : {metrics['profit_factor']:.2f}",
        f"Max DD    : ${metrics['max_drawdown']:.2f}",
        "─" * 24,
        f"Sharpe    : {metrics['sharpe']:.3f}",
        f"Sortino   : {sortino_str}",
    ]
    ax7.text(0.05, 0.97, "\n".join(summary_lines),
             transform=ax7.transAxes, fontsize=8.5,
             verticalalignment="top", fontfamily="monospace",
             color=C["text"],
             bbox=dict(boxstyle="round", facecolor=PANEL,
                       edgecolor=C["border"], pad=0.5))

    # ── 8. Heatmap p_ensemble ─────────────────────────
    ax8 = fig.add_subplot(gs[3, :])
    ax8.set_title("P(ensemble) — 60 dias amostra")
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
        im  = ax8.imshow(mat, aspect="auto", cmap="RdYlGn",
                         vmin=0, vmax=1, interpolation="nearest",
                         extent=[0, len(slot_keys), len(sample), 0])
        plt.colorbar(im, ax=ax8, fraction=0.015, pad=0.01,
                     label="P(ensemble)")
        htp = [i for i, (h, s) in enumerate(slot_keys) if s == 0]
        htl = [f"{h}h" for h, s in slot_keys if s == 0]
        ax8.set_xticks(htp); ax8.set_xticklabels(htl, fontsize=7)
        ax8.set_yticks(range(len(sample)))
        ax8.set_yticklabels(
            [str(d) for d in sample["date"].values], fontsize=6)

    # ── 9. PnL por estação ────────────────────────────
    ax9 = fig.add_subplot(gs[4, 0])
    ax9.set_title("PnL por Estação ($)")
    season_labels, season_vals, season_cols = [], [], []
    for season in ["winter", "spring", "summer", "autumn"]:
        key = f"{season}_pnl"
        if key in metrics:
            season_labels.append(season)
            v = metrics[key]
            season_vals.append(v)
            season_cols.append(C["win"] if v >= 0 else C["loss"])
    if season_labels:
        ax9.bar(season_labels, season_vals, color=season_cols, alpha=0.85)
        ax9.axhline(0, color=C["muted"], lw=0.8)
        ax9.set_ylabel("USDC"); ax9.grid(True, alpha=0.3, axis="y")

    # ── 10. Nº apostas/dia ────────────────────────────
    ax10 = fig.add_subplot(gs[4, 1])
    ax10.set_title("Distribuição: Apostas/Dia")
    n_par = results["n_parcels"].values
    max_p = 3 if mode == "phased" else 1
    for v in range(0, max_p + 1):
        cnt = (n_par == v).sum()
        col = (C["missed"] if v == 0 else C["premature"] if v == 1
               else C["purple"] if v == 2 else C["correct"])
        ax10.bar(str(v), cnt, color=col, alpha=0.85)
    ax10.set_xlabel("Apostas"); ax10.set_ylabel("Dias")
    ax10.grid(True, alpha=0.3, axis="y")

    # ── 11. Stop-loss por mês ─────────────────────────
    ax11 = fig.add_subplot(gs[4, 2])
    ax11.set_title("Stop-Loss Acionados/Mês")
    stops_m = (results.groupby("month")["n_stops"].sum()
               .reindex(months_range, fill_value=0))
    ax11.bar(m_lbls, stops_m, color=C["stop"], alpha=0.85)
    ax11.set_ylabel("Nº stops")
    ax11.tick_params(axis="x", rotation=45, labelsize=7)
    ax11.grid(True, alpha=0.3, axis="y")

    pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""
    fig.suptitle(
        f"Munich Max Temp — {mode_tag}  "
        f"correcto={metrics['correct_pct']}%  "
        f"PnL={pnl_sign}${metrics['total_pnl']:.2f} ({pnl_sign}{metrics['pnl_pct']:.1f}%)  "
        f"Sharpe={metrics['sharpe']:.2f}  Sortino={sortino_str}  "
        f"WinRate={metrics['win_rate']:.1f}%",
        fontsize=12)

    OUTPUT_DIR.mkdir(exist_ok=True)
    mode_slug = "phased" if mode == "phased" else "single"
    out_path  = OUTPUT_DIR / f"munich_backtest_{mode_slug}_{start_year}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard guardado: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Munich Backtest V3 + PnL + Stop-Loss")
    parser.add_argument("--mode", choices=["phased", "single"],
                        default="phased",
                        help="phased=3 parcelas $5, single=1 compra $5")
    args = parser.parse_args()

    start_str  = input("Data de início (YYYY-MM-DD): ").strip()
    start_date = pd.to_datetime(start_str).date()
    end_date   = date.today() - timedelta(days=1)

    mode_label = "PHASED 3×$5" if args.mode == "phased" else "SINGLE 1×$5"
    print(f"\n  Backtest V3: {start_date} → {end_date}  [{mode_label}]")
    print(f"  Stop-loss: temp > tecto_bracket + {STOP_LOSS_DEGREES}°C → venda ao bid")

    print("\n[1/5] Modelos...")
    models = load_models()

    print("\n[2/5] Dados...")
    df_all = load_data()
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df = df_all[
        (df_all["date"] >= start_date) &
        (df_all["date"] <= end_date)
    ].copy()
    print(f"  {len(df):,} slots no intervalo")

    print(f"\n[3/5] Backtest ({mode_label})...")
    sim_market = SimulatedMarket()
    results, slots_df = run(df, models, sim_market, mode=args.mode)

    print("\n[4/5] Métricas...")
    metrics = compute_metrics(results)

    # ── Rich Tables ────────────────────────────────────
    _console.print()
    mode_header = f"Resultados V3 — {mode_label}"
    _console.rule(f"[bold cyan]{mode_header}[/bold cyan]")

    # Tabela: Timing
    t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Label", style="dim", width=28)
    t.add_column("Value", style="bold white")

    t.add_row("Dias analisados",     f"{metrics['n_days']}")
    t.add_row("Correcto (timing)",   f"[green]{metrics['correct_pct']}%[/green]")
    t.add_row("Prematuro",           f"[yellow]{metrics['premature_pct']}%[/yellow]")
    t.add_row("Não detectado",       f"[red]{metrics['missed_pct']}%[/red]")
    t.add_row("Lag médio (correctos)",
              f"+{metrics.get('lag_mean_h', '?')}h")
    t.add_row("Lag ≤ 1h / ≤ 2h",
              f"{metrics.get('lag_le1h_pct', 0)}% / {metrics.get('lag_le2h_pct', 0)}%")
    _console.print(t)

    # Tabela: PnL
    _console.rule("[cyan]PnL & Apostas[/cyan]", style="dim")
    t2 = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t2.add_column("Label", style="dim", width=28)
    t2.add_column("Value", style="bold white")

    pnl_col = "green" if metrics["total_pnl"] >= 0 else "red"
    pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""

    t2.add_row("Total investido",
               f"${metrics['total_invested']:.2f}")
    t2.add_row("PnL total",
               f"[{pnl_col}]{pnl_sign}${metrics['total_pnl']:.2f} "
               f"({pnl_sign}{metrics['pnl_pct']:.1f}%)[/{pnl_col}]")
    t2.add_row("Win bets",
               f"[green]{metrics['n_win_bets']}[/green]")
    t2.add_row("Loss bets",
               f"[red]{metrics['n_loss_bets']}[/red]")
    t2.add_row("Stop-loss bets",
               f"[yellow]{metrics['n_stop_bets']}[/yellow]")
    t2.add_row("Win rate",
               f"[green]{metrics['win_rate']:.1f}%[/green]")
    t2.add_row("Avg win / Avg loss",
               f"[green]+${metrics['avg_win']:.3f}[/green] / "
               f"[red]{metrics['avg_loss']:.3f}[/red]")
    t2.add_row("Profit factor",
               f"{metrics['profit_factor']:.2f}")
    t2.add_row("Max drawdown",
               f"[red]${metrics['max_drawdown']:.2f}[/red]")
    _console.print(t2)

    # Tabela: Ratios
    _console.rule("[cyan]Risk Ratios (diários, anual.)[/cyan]", style="dim")
    t3 = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t3.add_column("Label", style="dim", width=28)
    t3.add_column("Value", style="bold white")

    sharpe_col  = "green" if metrics["sharpe"] > 1.0 else "yellow" if metrics["sharpe"] > 0 else "red"
    sortino_str = f"{metrics['sortino']:.3f}" if metrics["sortino"] < 900 else "∞ (sem dias neg.)"
    sortino_col = "green" if metrics["sortino"] > 1.5 or metrics["sortino"] > 900 else "yellow"

    t3.add_row("Sharpe ratio",
               f"[{sharpe_col}]{metrics['sharpe']:.3f}[/{sharpe_col}]")
    t3.add_row("Sortino ratio",
               f"[{sortino_col}]{sortino_str}[/{sortino_col}]")
    _console.print(t3)

    # Tabela: Por estação
    _console.rule("[cyan]Por Estação[/cyan]", style="dim")
    t_s = Table(box=rich_box.SIMPLE, show_header=True, padding=(0, 2))
    t_s.add_column("Estação",  style="cyan",  width=12)
    t_s.add_column("Correcto", justify="right", width=10)
    t_s.add_column("Lag médio",justify="right", width=10)
    t_s.add_column("PnL ($)",  justify="right", width=10)

    season_icons = {
        "winter": "❄️ ", "spring": "🌱",
        "summer": "☀️ ", "autumn": "🍂",
    }
    for season in ["winter", "spring", "summer", "autumn"]:
        key_c = f"{season}_correct_pct"
        key_l = f"{season}_lag_mean_h"
        key_p = f"{season}_pnl"
        if key_c in metrics:
            c_pct  = metrics[key_c]
            s_pnl  = metrics.get(key_p, 0.0)
            col_c  = ("green" if c_pct >= 70
                       else "yellow" if c_pct >= 50 else "red")
            col_p  = "green" if s_pnl >= 0 else "red"
            lag_v  = (f"+{metrics[key_l]:.2f}h"
                      if key_l in metrics
                      and metrics[key_l] is not None else "—")
            pnl_s  = f"+${s_pnl:.2f}" if s_pnl >= 0 else f"-${abs(s_pnl):.2f}"
            t_s.add_row(
                f"{season_icons.get(season, '')}{season}",
                f"[{col_c}]{c_pct:.1f}%[/{col_c}]",
                lag_v,
                f"[{col_p}]{pnl_s}[/{col_p}]",
            )
    _console.print(t_s)

    print(f"\n[5/5] Dashboard...")
    plot(results, slots_df, metrics, start_date.year, mode=args.mode)

    # Exportar CSV com resultados detalhados
    OUTPUT_DIR.mkdir(exist_ok=True)
    mode_slug  = "phased" if args.mode == "phased" else "single"
    csv_path   = OUTPUT_DIR / f"munich_results_{mode_slug}_{start_date.year}.csv"
    results.to_csv(csv_path, index=False)
    print(f"  Resultados CSV: {csv_path}")


if __name__ == "__main__":
    main()
