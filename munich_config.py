"""
munich_config.py
================
Constantes globais, helpers de timezone, ANSI e janelas de sinal WU.
"""

import json
import os
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

warnings.filterwarnings("ignore")

# ── Timezones ─────────────────────────────────────────
_BERLIN = ZoneInfo("Europe/Berlin")
_LOCAL  = ZoneInfo("Europe/Lisbon")


def berlin_now() -> datetime:
    return datetime.now(tz=_BERLIN)

def berlin_date() -> date:
    return berlin_now().date()

def local_now() -> datetime:
    return datetime.now(tz=_LOCAL)


# ── Horário activo ────────────────────────────────────
BOT_ACTIVE_START = 8
BOT_ACTIVE_END   = 20

_SIGNAL_CHECK_WINDOWS = [(18, 32), (45, 55)]
_FAST_POLL_INTERVAL = 2


def _in_signal_window() -> bool:
    m = berlin_now().minute
    return any(lo <= m <= hi for lo, hi in _SIGNAL_CHECK_WINDOWS)


def smart_sleep(interval: int, wu_key: str, wu_sess, last_temp, on_new_obs=None):
    from munich_weather import fetch_wu_latest
    if not _in_signal_window():
        time.sleep(interval)
        return None
    deadline = time.time() + interval
    while time.time() < deadline and _in_signal_window():
        time.sleep(_FAST_POLL_INTERVAL)
        try:
            obs = fetch_wu_latest(wu_key, wu_sess)
        except Exception:
            obs = None
        if obs and obs.get("temp_c") != last_temp:
            if on_new_obs:
                on_new_obs(obs)
            return obs
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return None


def ceil_slot(hour: int, minute: int) -> tuple[int, int]:
    if minute < 30:
        return (hour, 30)
    else:
        return (hour + 1, 0)


# ── Paths ─────────────────────────────────────────────
MODEL_LGB    = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_XGB    = Path("munich_peak_model/xgb_peak.pkl")       # NOVO
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
LOG_DIR      = Path("live_bot_logs")

# ── URLs ──────────────────────────────────────────────
WU_BASE      = "https://api.weather.com/v1/location"
GAMMA_API    = "https://gamma-api.polymarket.com"
OM_FORECAST  = "https://api.open-meteo.com/v1/forecast"      # NOVO
OM_ARCHIVE   = "https://archive-api.open-meteo.com/v1/archive"  # NOVO

# ── Geo ───────────────────────────────────────────────
MUNICH_LAT       = 48.35
MUNICH_LON       = 11.79
MUNICH_LAT_OM    = 48.14    # Open-Meteo usa city center
MUNICH_LON_OM    = 11.58

DAY_START  = 6
DAY_END    = 21
MIN_HOUR   = 6

# ── Chaves de ambiente ────────────────────────────────
WU_API_KEY          = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY    = os.environ.get("POLY_PRIVATE_KEY", "")
POLY_MAX_DAILY_LOSS = float(os.environ.get("POLY_MAX_DAILY_LOSS", "50"))

MONTH_NAMES = {
    1: "january",  2: "february", 3: "march",    4: "april",
    5: "may",      6: "june",     7: "july",      8: "august",
    9: "september",10: "october", 11: "november", 12: "december",
}
SEASONS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5],
    "summer": [6, 7, 8],  "autumn": [9, 10, 11],
}

# ── Features CANÓNICAS — alinhadas com V1 train/backtest ──
# IMPORTANTE: Qualquer alteração implica re-treino completo.
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

# ── Ensemble weights ──────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "lgbm":     0.50,   # LightGBM (modelo principal)
    "xgb":      0.30,   # XGBoost (modelo paralelo)
    "zscore":   0.20,   # Streaming z-score (estatístico)
}

FORECAST_AGREEMENT_TOLERANCE = 2  # °C de diferença aceitável entre WU e OM

# ── ANSI ──────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
C = {
    "cyan":   "\033[96m", "green":  "\033[92m", "yellow": "\033[93m",
    "orange": "\033[33m", "red":    "\033[91m", "blue":   "\033[94m",
    "purple": "\033[95m", "gray":   "\033[90m", "white":  "\033[97m",
}
