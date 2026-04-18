"""
munich_config.py — UNIFIED
================
Constantes globais, helpers de timezone, ANSI e janelas de sinal WU.
Importado por todos os outros modulos — sem dependencias externas
alem da stdlib e requests.

NAO importar munich_weather/model/display aqui para evitar ciclos.
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


# ══════════════════════════════════════════════════════
#  TIMEZONES
# ══════════════════════════════════════════════════════

_BERLIN = ZoneInfo("Europe/Berlin")
_LOCAL  = ZoneInfo("Europe/Lisbon")   # fuso local do bot — ajustar se necessario


def berlin_now() -> datetime:
    """Datetime actual em hora de Berlim/Munich."""
    return datetime.now(tz=_BERLIN)


def berlin_date() -> date:
    """Data actual segundo o relogio de Munich — determina o slug do mercado."""
    return berlin_now().date()


def local_now() -> datetime:
    """Datetime actual no fuso local do bot (Lisboa)."""
    return datetime.now(tz=_LOCAL)


# ══════════════════════════════════════════════════════
#  HORARIO ACTIVO DO BOT (hora local)
# ══════════════════════════════════════════════════════

BOT_ACTIVE_START = 8   # 08:00 hora local (Lisboa)
BOT_ACTIVE_END   = 20  # 20:00 hora local (Lisboa)


# ══════════════════════════════════════════════════════
#  JANELAS DE SINAL EDDM
# ══════════════════════════════════════════════════════

# A estacao reporta tipicamente ~:20 e ~:50 de cada hora.
_SIGNAL_CHECK_WINDOWS = [(18, 32), (45, 55)]  # (min_inicio, min_fim) hora Berlin

# Intervalo de polling rapido dentro das janelas de sinal (segundos)
_FAST_POLL_INTERVAL = 2


def _in_signal_window() -> bool:
    """True se o minuto actual de Berlin esta dentro de uma janela de sinal EDDM."""
    m = berlin_now().minute
    return any(lo <= m <= hi for lo, hi in _SIGNAL_CHECK_WINDOWS)


def smart_sleep(interval: int, wu_key: str, wu_sess, last_temp, on_new_obs=None):
    """
    Substitui time.sleep(interval) no loop principal.

    Fora das janelas de sinal: dorme interval segundos normalmente.
    Dentro das janelas de sinal: faz polling a cada _FAST_POLL_INTERVAL segundos
    ate detectar uma nova temperatura OU sair da janela.

    Devolve a nova observacao WU se foi detectada durante o fast-poll, ou None.

    on_new_obs: callback opcional(obs) chamado assim que nova temp e detectada.
    """
    # Importacao local para evitar ciclo munich_config -> munich_weather -> munich_config
    from munich_weather import fetch_wu_latest

    if not _in_signal_window():
        time.sleep(interval)
        return None

    # Estamos na janela — polling rapido
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

    # Janela passou ou deadline — dormir o restante
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return None


# ══════════════════════════════════════════════════════
#  SLOT 30MIN
# ══════════════════════════════════════════════════════

def ceil_slot(hour: int, minute: int) -> tuple[int, int]:
    """
    Converte (hour, minute) de uma observacao WU para o slot 30min correcto.

    Semantica: truncar para CIMA.
      minute=0-29  -> slot 30 da mesma hora   (ex: 14:20 -> (14, 30))
      minute=30-59 -> slot  0 da hora seguinte (ex: 14:50 -> (15,  0))
    """
    if minute < 30:
        return (hour, 30)
    else:
        return (hour + 1, 0)


# ══════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════

MODEL_LGB    = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_XGB    = Path("munich_peak_model/xgb_peak.pkl")       # Ensemble V2
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
DATA_CSV     = Path("historic/munich.csv")
LOG_DIR      = Path("live_bot_logs")

# Strategy configs (unified architecture)
STRATEGY_CONFIG_DIR = Path("strategy_configs")
DEFAULT_STRATEGY_CONFIG = STRATEGY_CONFIG_DIR / "default_config.json"
OPTIMIZED_STRATEGY_CONFIG = STRATEGY_CONFIG_DIR / "optimized_config.json"

# Backtest results
BACKTEST_RESULTS_DIR = Path("backtest_results")


# ══════════════════════════════════════════════════════
#  URLs
# ══════════════════════════════════════════════════════

WU_BASE      = "https://api.weather.com/v1/location"
GAMMA_API    = "https://gamma-api.polymarket.com"
OM_FORECAST  = "https://api.open-meteo.com/v1/forecast"       # Open-Meteo
OM_ARCHIVE   = "https://archive-api.open-meteo.com/v1/archive" # Open-Meteo histórico


# ══════════════════════════════════════════════════════
#  CHAVES DE AMBIENTE
# ══════════════════════════════════════════════════════

WU_API_KEY          = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY    = os.environ.get("POLY_PRIVATE_KEY", "")
POLY_MAX_DAILY_LOSS = float(os.environ.get("POLY_MAX_DAILY_LOSS", "50"))


# ══════════════════════════════════════════════════════
#  GEO / HORARIO
# ══════════════════════════════════════════════════════

MUNICH_LAT       = 48.35       # WU (aeroporto EDDM)
MUNICH_LON       = 11.79
MUNICH_LAT_OM    = 48.14       # Open-Meteo (city center)
MUNICH_LON_OM    = 11.58

DAY_START  = 6
DAY_END    = 21
MIN_HOUR   = 6


# ══════════════════════════════════════════════════════
#  NOMES DE MESES / ESTACOES
# ══════════════════════════════════════════════════════

MONTH_NAMES = {
    1: "january",  2: "february", 3: "march",    4: "april",
    5: "may",      6: "june",     7: "july",      8: "august",
    9: "september",10: "october", 11: "november", 12: "december",
}
SEASONS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5],
    "summer": [6, 7, 8],  "autumn": [9, 10, 11],
}


# ══════════════════════════════════════════════════════
#  FEATURES CANONICAS — Ensemble V2
# ══════════════════════════════════════════════════════
# IMPORTANTE: Qualquer alteracao implica re-treino completo.
# Estas features devem ser identicas em treino, backtest e live bot.
#
# V1 (single LightGBM) tinha: pct_of_running_max, temp_lag_1,
#   roll3_mean, temp_above_morning_max
# V2 (ensemble) substitui por: doy_sin, doy_cos, temp_vs_climatology,
#   recent_slope, plateau_indicator, radiation_proxy, humidity_drop_1h

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
#  ENSEMBLE CONFIG
# ══════════════════════════════════════════════════════

ENSEMBLE_WEIGHTS = {
    "lgbm":   0.50,   # LightGBM (modelo principal)
    "xgb":    0.30,   # XGBoost (modelo paralelo)
    "zscore": 0.20,   # Streaming z-score (estatistico)
}

FORECAST_AGREEMENT_TOLERANCE = 2  # °C de diferenca aceitavel entre WU e OM


# ══════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════

R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
C = {
    "cyan":   "\033[96m", "green":  "\033[92m", "yellow": "\033[93m",
    "orange": "\033[33m", "red":    "\033[91m", "blue":   "\033[94m",
    "purple": "\033[95m", "gray":   "\033[90m", "white":  "\033[97m",
    "reset":  "\033[0m",
}
