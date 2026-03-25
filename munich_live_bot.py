"""
munich_live_bot.py
==================
Bot de trading ao vivo — Temperatura Maxima Munich — Polymarket.

Modos:
  PAPER — simula ordens; mostra order book real (bid/ask/spread do CLOB)
  REAL  — envia ordens reais ao Polymarket CLOB via py-clob-client
          requer confirmação manual (y/n) + stop-loss diário

Arranque:
  1. Pergunta interactiva: Paper ou Real?
  2. Bootstrap histórico de hoje via WU API (EDDM, desde 00:00)
  3. Aplica LightGBM a toda a série histórica
  4. Loop (cada 60s): nova leitura WU + modelo + dashboard

Dashboard:
  - Curva temperatura ASCII
  - P(pico já ocorreu) em tempo real
  - Order book CLOB (bid / ask / spread / depth)
  - EV calculado sobre o ask (preço real de compra)
  - Bet simulada (PAPER) ou ordem enviada (REAL)

Instalacao:
    pip install requests pandas numpy scikit-learn lightgbm joblib py-clob-client

Variáveis de ambiente obrigatórias:
    export WU_API_KEY="a_tua_chave_wunderground"
    export POLY_PRIVATE_KEY="0x..."

Variáveis opcionais:
    export POLY_MAX_DAILY_LOSS="50"    # stop-loss diário em USDC (default: 50)

Uso:
    python munich_live_bot.py
    python munich_live_bot.py --threshold 0.80
    python munich_live_bot.py --bankroll 200 --kelly 0.5 --min-edge 5
"""

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from polymarket_clob import ClobClient, TradingMode, OrderBook, PositionManager, Position, PositionStatus
from polymarket_orders import OrderExecutor, paper_buy
from tg import TG


warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  TIMEZONE HELPERS
#  IMPORTANTE: hora local = onde corre o bot (ex: Lisboa)
#              hora da station = Europe/Berlin (Munich)
#  O mercado Polymarket muda de slug à meia-noite de Berlim,
#  não à meia-noite local do bot.
# ══════════════════════════════════════════════════════
_BERLIN = ZoneInfo("Europe/Berlin")
_LOCAL  = ZoneInfo("Europe/Lisbon")   # fuso local do bot — ajustar se necessário

def berlin_now() -> datetime:
    """Datetime actual em hora de Berlim/Munich."""
    return datetime.now(tz=_BERLIN)

def berlin_date() -> date:
    """Data actual segundo o relógio de Munich — é esta que determina o slug do mercado."""
    return berlin_now().date()

def local_now() -> datetime:
    """Datetime actual no fuso local do bot (Lisboa).
    NOTA: usa tz=_LOCAL para ser explícito — não depende do sistema operativo."""
    return datetime.now(tz=_LOCAL)

# Horas locais (bot) em que o loop está activo
BOT_ACTIVE_START = 8   # 08:00 hora local (Lisboa)
BOT_ACTIVE_END   = 20  # 20:00 hora local (Lisboa)

# Janelas EDDM: a estação reporta tipicamente ~:20 e ~:50 de cada hora.
# Minutos-alvo dentro da hora para avisar "a verificar sinal"
# (28 min = perto dos :20→:30,  48 min = perto dos :50→:00)
_SIGNAL_CHECK_WINDOWS = [(18, 32), (45, 55)]  # (min_inicio, min_fim) hora Berlin


def ceil_slot(hour: int, minute: int) -> tuple[int, int]:
    """
    Converte (hour, minute) de uma observação WU para o slot 30min correcto.

    Semântica: truncar para CIMA — a observação das 14:50 pertence ao slot
    (15, 0) porque é a leitura mais recente disponível quando se entra na hora 15.
    A observação das 14:20 pertence ao slot (14, 30).

      minute=0-29  → slot  30 da mesma hora   (ex: 14:20 → (14, 30))
      minute=30-59 → slot   0 da hora seguinte (ex: 14:50 → (15,  0))

    Casos limite:
      minute=30 exacto → slot 0 da hora seguinte (14:30 → (15, 0))
      slot além das DAY_END é ignorado pelo chamador
    """
    if minute < 30:
        return (hour, 30)
    else:
        next_h = hour + 1
        return (next_h, 0)

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
MODEL_LGB    = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
LOG_DIR      = Path("live_bot_logs")

WU_BASE      = "https://api.weather.com/v1/location"
GAMMA_API    = "https://gamma-api.polymarket.com"


# ── WUnderground API Key ──────────────────────────────
# Definir variavel de ambiente antes de correr:
#   export WU_API_KEY="a_tua_chave"      (Linux/macOS)
#   set WU_API_KEY=a_tua_chave           (Windows CMD)
#
# Obtem em: https://www.wunderground.com/member/api-keys
WU_API_KEY = os.environ.get("WU_API_KEY", "")

# ── Polymarket CLOB ───────────────────────────────────
# Private key da wallet Polygon (EVM):
#   export POLY_PRIVATE_KEY="0x..."
# Stop-loss diário em USDC (default $50):
#   export POLY_MAX_DAILY_LOSS="50"
POLY_PRIVATE_KEY   = os.environ.get("POLY_PRIVATE_KEY", "")
POLY_MAX_DAILY_LOSS = float(os.environ.get("POLY_MAX_DAILY_LOSS", "50"))

MUNICH_LAT   = 48.35
MUNICH_LON   = 11.79
DAY_START    = 6
DAY_END      = 21
MIN_HOUR     = 6

MONTH_NAMES  = {
    1:"january", 2:"february", 3:"march",    4:"april",
    5:"may",     6:"june",     7:"july",      8:"august",
    9:"september",10:"october",11:"november", 12:"december"
}
SEASONS = {
    "winter":[12,1,2],"spring":[3,4,5],
    "summer":[6,7,8],"autumn":[9,10,11],
}
# ── Lista canónica de features — DEVE ser idêntica em treino, backtest e live bot.
# Qualquer alteração implica re-treino completo.
#
# Critério de selecção: cada feature mede algo que as outras não medem.
#   REMOVIDAS vs versão anterior (31→15):
#   - hour, month (raw)          → redundantes com slot_frac e seasonal_peak_prior
#   - doy_sin/cos, month_sin/cos → a sazonalidade já está capturada no seasonal_peak_prior
#   - hour_sin/cos               → slot_frac contínuo é suficiente e mais simples
#   - delta_2h                   → coberto por delta_1h + accel
#   - temp_lag_2, temp_lag_4, temp_lag_6 → lag_1 e lag_3 dão a curva sem repetição
#   - roll6_mean                 → roll3 já cobre a tendência recente
#   - morning_min                → morning_max + temp_above_morning_max contam a mesma história
#   - humidity                   → correlação fraca com hora do pico, muito ruidoso
#   - cloud_cover                → derivado de wx_phrase categórico, não validado
FEATURE_COLS = [
    # Posição temporal (1 feature, contínua — sem redundância com hora raw ou sin/cos)
    "slot_frac",
    # Temperatura actual e acumulada
    "temp_c", "running_max", "pct_of_running_max",
    # Dinâmica: velocidade e aceleração
    "delta_30m", "delta_1h", "accel",
    # Contexto recente: dois lags não-redundantes com os deltas
    "temp_lag_1", "temp_lag_3",
    # Tendência e volatilidade (janela ~1.5h)
    "roll3_mean", "roll3_std",
    # Contexto matinal (quanto está acima do máximo da manhã)
    "morning_max", "temp_above_morning_max",
    # Contexto histórico
    "prev_7d_avg_max",
    # Prior sazonal — captura mês, doy e hora do pico num único número calibrado
    "seasonal_peak_prior",
]

# ANSI
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
C={
    "cyan":"\033[96m","green":"\033[92m","yellow":"\033[93m",
    "orange":"\033[33m","red":"\033[91m","blue":"\033[94m",
    "purple":"\033[95m","gray":"\033[90m","white":"\033[97m",
}


# ══════════════════════════════════════════════════════
#  1. LOAD MODEL
# ══════════════════════════════════════════════════════
def load_model():
    if not MODEL_LGB.exists():
        raise FileNotFoundError(
            f"\n  {C['red']}Modelo nao encontrado: {MODEL_LGB}{R}\n"
            "  Corre: python munich_train.py"
        )
    model  = joblib.load(MODEL_LGB)
    config = json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {}
    feat   = config.get("feature_cols", FEATURE_COLS)

    # Prior sazonal — chave "month_hour_slot30"
    raw_prior  = config.get("seasonal_peak_prior", {})
    prior_map: dict[tuple, float] = {}
    for k, v in raw_prior.items():
        parts = k.split("_")
        if len(parts) == 3:
            try:
                prior_map[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
            except ValueError:
                pass

    # Threshold adaptativo por mês — chave "1".."12"
    raw_thresh = config.get("monthly_threshold", {})
    monthly_threshold: dict[int, float] = {}
    for k, v in raw_thresh.items():
        try:
            monthly_threshold[int(k)] = float(v)
        except ValueError:
            pass

    thresh_str = (f"{len(monthly_threshold)} meses"
                  if monthly_threshold else f"{C['yellow']}nao disponivel{R}")
    print(f"  {C['green']}✓{R} LightGBM  AUC={config.get('global_auc','?')}  "
          f"features={len(feat)}  "
          f"threshold_adaptativo={thresh_str}  "
          f"prior={'sim' if prior_map else C['yellow']+'nao'+R}")
    return model, feat, prior_map, monthly_threshold


# ══════════════════════════════════════════════════════
#  2. WEATHER UNDERGROUND — SCRAPING
# ══════════════════════════════════════════════════════
def make_wu_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.wunderground.com/",
        "Origin":          "https://www.wunderground.com",
    })
    return s


# ── WU API v1 EDDM ──────────────────────────────────────
# Estacao EDDM Munich Airport — confirmado: devolve 35 obs para hoje
# valid_time_gmt e unix timestamp em UTC → converter para CET
WU_EDDM_URL = f"{WU_BASE}/EDDM:9:DE/observations/historical.json"


def _wu_parse_obs(obs_list: list) -> list[dict]:
    """
    Parser para observacoes WU v1 EDDM.
    Campo de tempo: valid_time_gmt (unix UTC) → converter para CET/CEST (Europe/Berlin).
    Temperatura: campo "temp" (inteiro, graus Celsius em metric).
    """
    rows   = []
    for obs in obs_list:
        temp = obs.get("temp")
        if temp is None:
            continue
        vt = obs.get("valid_time_gmt")
        if vt is None:
            continue
        try:
            from datetime import timezone as _tz
            dt = datetime.fromtimestamp(int(vt), tz=_tz.utc).astimezone(_BERLIN)
        except Exception:
            continue
        # clds: CLR=0, FEW=12, SCT=37, BKN=75, OVC=100, OBS=100, VV=100
        clds_map = {"CLR":0,"SKC":0,"FEW":12,"SCT":37,"BKN":75,
                    "OVC":100,"OBS":100,"VV":100,"X":100}
        clds_raw   = str(obs.get("clds","") or "").upper().strip()
        cloud_cover= clds_map.get(clds_raw, 50)

        rows.append({
            "hour":        dt.hour,
            "minute":      dt.minute,
            "temp_c":      int(round(float(temp))),
            "humidity":    int(round(float(obs.get("rh") or 70))),
            "cloud_cover": cloud_cover,
            "wx":          str(obs.get("wx_phrase","") or ""),
        })
    return rows


def fetch_wu_day_eddm(day: date, api_key: str,
                      session: requests.Session) -> list[dict]:
    """WU EDDM historical — funciona para hoje e dias passados."""
    try:
        r = session.get(WU_EDDM_URL, params={
            "apiKey":    api_key,
            "units":     "m",
            "startDate": day.strftime("%Y%m%d"),
        }, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return _wu_parse_obs(obs) if obs else []
    except Exception:
        return []


def fetch_wu_forecast_max(api_key: str,
                          session: requests.Session) -> dict | None:
    """
    Previsao de temperatura maxima para hoje via WU v3 daily forecast.
    Endpoint: api.weather.com/v3/wx/forecast/daily/5day
    """
    url = "https://api.weather.com/v3/wx/forecast/daily/5day"
    try:
        r = session.get(url, params={
            "apiKey":   api_key,
            "geocode":  "48.354,11.792",
            "units":    "m",
            "language": "en-US",
            "format":   "json",
        }, timeout=15)
        r.raise_for_status()
        d = r.json()
        t_max_list = d.get("temperatureMax", [None])
        t_min_list = d.get("temperatureMin", [None])
        t_max = int(round(float(t_max_list[0]))) if t_max_list and t_max_list[0] is not None else None
        t_min = int(round(float(t_min_list[0]))) if t_min_list and t_min_list[0] is not None else None
        if t_max is None:
            return None
        return {"temp_max": t_max, "temp_min": t_min}
    except Exception:
        return None


def fetch_wu_latest(api_key: str,
                    session: requests.Session) -> dict | None:
    """Leitura mais recente — ultima observacao de hoje (data de Munich/Berlim)."""
    rows = fetch_wu_day_eddm(berlin_date(), api_key, session)
    if not rows:
        return None
    return max(rows, key=lambda r: r["hour"] * 60 + r["minute"])


def bootstrap_today(api_key: str,
                    session: requests.Session) -> tuple[dict, list[dict]]:
    """
    Ao arranque: histórico completo de hoje via EDDM.
    Devolve:
      series_today: {(hour, slot30): temp_c}  — para o gráfico ASCII
      slots_so_far: lista de dicts ordenada por tempo  — para o modelo
    """
    today = berlin_date()
    print(f"  {DIM}WU EDDM historico {today}...{R}", end=" ", flush=True)
    rows = fetch_wu_day_eddm(today, api_key, session)
    if not rows:
        print(f"{C['red']}sem dados WU EDDM{R}")
        return {}, []
    t_vals = [r["temp_c"] for r in rows]
    print(f"{C['green']}{len(rows)} obs  "
          f"{min(t_vals)}°C – {max(t_vals)}°C{R}")

    bootstrap_today._rows_cache = rows

    # series_today para o gráfico  +  obs_min: timestamp real da observação WU por slot
    series: dict[tuple, float] = {}
    obs_min: dict[tuple, tuple] = {}   # {slot_key: (hour_orig, minute_orig)}
    for r in rows:
        key = ceil_slot(r["hour"], r["minute"])
        if key not in series or r["temp_c"] >= series[key]:
            series[key]  = r["temp_c"]
            obs_min[key] = (r["hour"], r["minute"])   # timestamp real WU

    bootstrap_today._obs_min = obs_min   # cache acessível fora

    # slots_so_far para o modelo (lista cronológica sem duplicados por slot)
    seen: set[tuple] = set()
    slots: list[dict] = []
    for r in sorted(rows, key=lambda x: x["hour"]*60 + x["minute"]):
        k = ceil_slot(r["hour"], r["minute"])
        if k not in seen:
            seen.add(k)
            slots.append({
                "hour":        k[0],
                "slot30":      k[1],
                "temp_c":      r["temp_c"],
                "cloud_cover": r.get("cloud_cover", 50),
                "humidity":    r.get("humidity", 70),
            })

    return series, slots


# ══════════════════════════════════════════════════════
#  3. CLOUD COVER — directo da EDDM WU (campo clds)
# ══════════════════════════════════════════════════════
def cloud_from_series(series_today: dict, rows_cache: list) -> dict[int, int]:
    """
    Extrai cloud_cover por hora directamente das observacoes WU EDDM.
    Nao usa Open-Meteo — tudo vem da EDDM.
    """
    cloud = {}
    for r in rows_cache:
        cloud[r["hour"]] = r.get("cloud_cover", 50)
    return cloud

# ══════════════════════════════════════════════════════
#  PREV_7D_AVG_MAX — igual ao trainer/backtester
# ══════════════════════════════════════════════════════

def compute_prev7(history: dict[date, float], d: date) -> float:
    """
    history: {date: max_temp}
    d: dia atual
    Devolve média dos últimos 7 dias (excluindo hoje).
    """
    days = sorted(history.keys())
    if d not in days:
        return None

    idx = days.index(d)
    if idx == 0:
        return history[d]

    window = days[max(0, idx-7):idx]
    vals = [history[x] for x in window]
    return float(np.mean(vals)) if vals else history[d]

# ══════════════════════════════════════════════════════
#  4. FEATURE BUILDER
# ══════════════════════════════════════════════════════
def build_features(slots_so_far: list[dict], current: dict,
                   month: int, doy: int, minute: int = 0) -> dict:
    """
    Constrói uma linha com as 15 features canónicas para o slot actual.
    slots_so_far: lista cronológica incluindo o slot actual como último elemento.
    lag(1) = slot anterior (~30min), lag(3) = ~1.5h atrás.
    """
    vals  = [s["temp_c"] for s in slots_so_far]
    n     = len(vals)
    cur   = vals[-1]
    rmax  = max(vals)
    hour  = current["hour"]
    slot30= current.get("slot30", ceil_slot(hour, minute)[1])

    def lag(k): return vals[-k] if n >= k else vals[0]

    # Contexto matinal: slots ANTERIORES ao corrente com hora <= 12
    morn_vals = [s["temp_c"] for s in slots_so_far[:-1] if s["hour"] <= 12]
    mmax = max(morn_vals) if morn_vals else cur

    prev7     = current.get("prev_7d_avg_max", rmax)
    slot_frac = (hour + slot30 / 60.0) / 24.0

    return {
        "slot_frac":              slot_frac,
        "temp_c":                 cur,
        "running_max":            rmax,
        "pct_of_running_max":     cur / rmax if rmax else 1.0,
        "delta_30m":              cur - lag(2),
        "delta_1h":               cur - lag(3),
        "accel":                  (cur - lag(2)) - (lag(2) - lag(3)),
        "temp_lag_1":             lag(2),
        "temp_lag_3":             lag(4),
        "roll3_mean":             np.mean(vals[-3:]),
        "roll3_std":              np.std(vals[-3:]) if n >= 3 else 0.0,
        "morning_max":            mmax,
        "temp_above_morning_max": cur - mmax,
        "prev_7d_avg_max":        prev7,
        "seasonal_peak_prior":    get_seasonal_prior(month, hour, slot30),
    }


# ══════════════════════════════════════════════════════
#  PREDICT — versão nova (Booster.predict)
# ══════════════════════════════════════════════════════

def predict_p(model, feat_cols, slots_so_far: list[dict], current: dict,
              month: int, doy: int) -> float:
    """
    Devolve P(pico já ocorreu) para o slot atual.
    Requer pelo menos 4 slots e hora >= MIN_HOUR para ter lags e rolling features fiáveis.
    """
    hour = current["hour"]

    if len(slots_so_far) < 4 or hour < MIN_HOUR:
        return 0.0

    row   = build_features(slots_so_far, current, month, doy)
    avail = [f for f in feat_cols if f in row]
    X     = pd.DataFrame([row])[avail].fillna(0)

    # Booster.predict devolve probabilidade da classe positiva
    return float(model.predict(X)[0])

# ══════════════════════════════════════════════════════
#  SEASONAL PRIOR — necessário para o modelo CEILING
# ══════════════════════════════════════════════════════

_SEASONAL_PRIOR: dict[tuple, float] = {}

def set_seasonal_prior(prior_map: dict):
    global _SEASONAL_PRIOR
    _SEASONAL_PRIOR = prior_map or {}

def get_seasonal_prior(month: int, hour: int, slot30: int) -> float:
    if _SEASONAL_PRIOR:
        return _SEASONAL_PRIOR.get((month, hour, slot30), 0.5)
    return 0.5

# ══════════════════════════════════════════════════════
#  HISTÓRICO DIÁRIO — necessário para prev_7d_avg_max
# ══════════════════════════════════════════════════════

def init_history_max() -> dict:
    path = Path("live_history_max.json")
    if path.exists():
        try:
            return {date.fromisoformat(k): float(v)
                    for k, v in json.loads(path.read_text()).items()}
        except:
            pass
    return {}

def save_history_max(history_max: dict):
    path = Path("live_history_max.json")
    data = {d.isoformat(): v for d, v in history_max.items()}
    path.write_text(json.dumps(data, indent=2))

def update_history_max(history_max: dict, slots_so_far: list[dict]):
    if not slots_so_far:
        return
    today = berlin_date()
    max_temp_today = max(s["temp_c"] for s in slots_so_far)
    history_max[today] = max_temp_today
    save_history_max(history_max)

# ══════════════════════════════════════════════════════
#  5. POLYMARKET
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
#  5. POLYMARKET
# ══════════════════════════════════════════════════════
def date_to_slug(d: date) -> str:
    return (f"highest-temperature-in-munich-on-"
            f"{MONTH_NAMES[d.month]}-{d.day}-{d.year}")

import re as _re

def _extract_temp(text: str) -> float | None:
    for pat in [r'([-]?\d+)\s*°?\s*[cC]\b',
                r'([-]?\d+)\s*or\s+(?:higher|lower|above|below)',
                r'be\s+([-]?\d+)', r'^\s*([-]?\d+)\s*$']:
        m = _re.search(pat, str(text), _re.IGNORECASE)
        if m: return float(m.group(1))
    return None

def _bracket_lo(label):
    s = str(label).lower()
    v = _extract_temp(label)
    if v is None: return 0.0
    if any(x in s for x in ("or lower","or below","≤","<=")): return -99.0
    return v

def _bracket_hi(label):
    s = str(label).lower()
    v = _extract_temp(label)
    if v is None: return 99.0
    if any(x in s for x in ("or higher","or above","≥",">=")): return 99.0
    return v

def _normalize_label(text: str) -> str:
    """Normaliza label longo para formato curto ('7°C', '9°C or higher')."""
    if len(text) <= 25: return text
    v = _extract_temp(text)
    if v is None: return text
    s = text.lower()
    if any(x in s for x in ("higher","above","≥",">=")): return f"{v:.0f}°C or higher"
    if any(x in s for x in ("lower","below","≤","<=")): return f"{v:.0f}°C or lower"
    return f"{v:.0f}°C"

def fetch_market(d: date) -> dict | None:
    slug = date_to_slug(d)
    def try_api(params):
        try:
            r = requests.get(f"{GAMMA_API}/events", params=params, timeout=15)
            r.raise_for_status()
            ev = r.json()
            return ev if isinstance(ev, list) else ([ev] if ev else [])
        except: return []

    month_s = MONTH_NAMES[d.month].capitalize()
    events = (try_api({"slug": slug}) or
              try_api({"q": f"highest temperature Munich {month_s} {d.day} {d.year}",
                       "limit": 10}) or
              try_api({"q": f"Munich temperature {d.year}", "limit": 10}))
    if not events: return None

    def is_munich(e):
        t = str(e.get("title","")).lower()
        return ("munich" in t or "munchen" in t) and (
               "temp" in t or "temperature" in t or "highest" in t)

    munich = [e for e in events if isinstance(e, dict) and is_munich(e)]
    if not munich: munich = [e for e in events if isinstance(e, dict)]
    if not munich: return None

    event = max(munich, key=lambda e: float(e.get("volume",0) or 0))
    brackets = []
    for m in event.get("markets", []):
        raw_label = (m.get("groupItemTitle") or m.get("outcomeTitle") or
                     m.get("title") or m.get("question") or "")
        label = _normalize_label(raw_label)
        v = _extract_temp(label)
        if v is None: continue

        outcomes  = m.get("outcomes","[]")
        prices    = m.get("outcomePrices","[]")
        token_ids = m.get("clobTokenIds","[]")
        for x in [outcomes, prices, token_ids]:
            if isinstance(x, str):
                try: x[:] = json.loads(x)
                except: pass
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: outcomes = []
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if isinstance(token_ids, str):
            try: token_ids = json.loads(token_ids)
            except: token_ids = []

        price_yes = None
        token_yes = None
        for i, out in enumerate(outcomes):
            if str(out).lower() in ("yes","true","1"):
                price_yes = float(prices[i]) if i < len(prices) and prices[i] else None
                token_yes = token_ids[i] if i < len(token_ids) else None
                break
        if price_yes is None and prices:
            try: price_yes = float(prices[0])
            except: price_yes = 0.5

        if price_yes is None: continue
        brackets.append({
            "label":    label,
            "price":    round(price_yes, 4),
            "token_id": token_yes,
            "temp_lo":  _bracket_lo(label),
            "temp_hi":  _bracket_hi(label),
            "volume":   float(m.get("volume",0) or 0),
        })

    if not brackets: return None
    brackets.sort(key=lambda b: b["temp_lo"])
    return {
        "title":      event.get("title","Munich Max Temp"),
        "end_date":   event.get("endDate",""),
        "volume":     float(event.get("volume",0) or 0),
        "brackets":   brackets,
        "n_outcomes": len(brackets),
        "slug":       slug,
    }

def find_bracket(market: dict, temp: float) -> dict | None:
    if not market: return None
    tr = round(temp)
    for b in market["brackets"]:
        lo, hi = b["temp_lo"], b["temp_hi"]
        if lo == hi and tr == round(lo):     return b
        if hi == 99  and tr >= lo:           return b
        if lo == -99 and tr <= hi:           return b
        if lo <= temp <= hi:                 return b
    return min(market["brackets"],
               key=lambda b: abs(tr - (b["temp_lo"] if b["temp_hi"]==99
                                       else b["temp_hi"] if b["temp_lo"]==-99
                                       else (b["temp_lo"]+b["temp_hi"])/2)))


# ══════════════════════════════════════════════════════
#  6. EV / BET
# ══════════════════════════════════════════════════════
def compute_ev(p: float, ask: float) -> dict | None:
    """
    Calcula EV usando o ask do CLOB (preço real de compra de YES).
    p    : P(pico já ocorreu) do modelo — proxy de P(bracket correcto)
    ask  : preço ask do CLOB em USDC por share (0–1)
    """
    if not ask or not (0 < ask < 1): return None
    ev    = p - ask
    b     = (1 - ask) / ask
    kelly = max(0.0, (p * b - (1 - p)) / b)
    return {
        "ev":          round(ev, 4),
        "ev_cents":    round(ev * 100, 2),
        "kelly":       round(kelly, 4),
        "edge_pct":    round((p / ask - 1) * 100, 2),
        "ev_positive": ev > 0,
        "ask":         round(ask, 4),
    }

def build_bet_record(bracket, p, ev, bankroll, kelly_frac, mode: TradingMode) -> dict:
    """
    Constrói o dict de bet/ordem — comum a PAPER e REAL.
    O tamanho é calculado aqui; a execução (simulada ou real) é feita no run().
    """
    bet_size = bankroll * ev["kelly"] * kelly_frac
    ask      = ev["ask"]
    shares   = round(bet_size / ask, 4) if ask > 0 else 0
    return {
        "mode":       mode.value,
        "bracket":    bracket["label"],
        "token_id":   bracket.get("token_id"),
        "ask":        round(ask, 4),
        "bid":        round(bracket.get("bid") or ask, 4),
        "spread":     round(bracket.get("spread") or 0, 4),
        "p_true":     round(p, 3),
        "ev_cents":   ev["ev_cents"],
        "edge_pct":   ev["edge_pct"],
        "kelly_full": round(ev["kelly"] * 100, 2),
        "kelly_frac": kelly_frac,
        "bet_size":   round(bet_size, 2),
        "shares":     shares,
        "max_profit": round(shares * (1 - ask), 2),
        "timestamp":  datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════
#  7. TERMINAL CHART
# ══════════════════════════════════════════════════════
def draw_chart(series_today: dict, signals: dict,
               peak_detected: bool) -> list[str]:
    """
    Grafico ASCII da curva de temperatura.
    serie_today: {(hour, minute): temp_c} — todos os pontos do dia
    signals:     {hour: p}               — probabilidade por hora
    Eixo X: slots de 30 min das 6h–20h
    Marcadores: ██ (grande) com cor por probabilidade
    Temperaturas em inteiro (EDDM reporta inteiros)
    """
    lines = []

    # Todos os slots de 30 em 30 min das 6h–20h
    slots = [(h, m) for h in range(DAY_START, DAY_END + 1) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    if not avail:
        return [f"  {DIM}sem dados para grafico{R}"]

    t_min  = min(avail) - 0.5
    t_max  = max(avail) + 0.5
    t_rng  = max(t_max - t_min, 1.0)
    chart_h= 8

    def to_row(t):
        return int((1 - (t - t_min) / t_rng) * (chart_h - 1))

    # Largura: 2 chars por slot
    col_w = 2
    total_w = len(slots) * col_w + 5

    grid = [[" "] * total_w for _ in range(chart_h)]

    # Escala Y (inteiros)
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        label = f"{int(round(t_val)):>3}°"
        for ci, ch in enumerate(label):
            if ci < 4:
                grid[row][ci] = ch

    # Baseline
    for ci in range(4, total_w):
        grid[chart_h - 1][ci] = "─"

    # Plotar pontos — marcador largo (2 chars) com cor
    for si, ((h, m), temp) in enumerate(zip(slots, temps)):
        if temp is None:
            continue
        row = to_row(temp)
        col = 4 + si * col_w
        p   = signals.get(h, 0)
        if p >= 0.80:   sym = f"{C['green']}██{R}"
        elif p >= 0.60: sym = f"{C['yellow']}██{R}"
        elif p >= 0.30: sym = f"{C['orange']}▓▓{R}"
        else:            sym = f"{DIM}▒▒{R}"
        if 0 <= row < chart_h - 1:
            grid[row][col] = sym
            grid[row][col+1] = ""   # consumido pelo marcador largo

    for row in grid:
        lines.append("  " + "".join(row))

    # Eixo X — horas (uma etiqueta por hora, alinhada)
    x_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        # cada hora ocupa 4 chars (2 slots × 2 cols)
        x_line += f"{h:<4}"
    lines.append(f"{DIM}{x_line}{R}")

    # Barra P(pico) por hora
    p_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        p = signals.get(h, 0)
        if p >= 0.80:   cell = f"{C['green']}▓▓{R}  "
        elif p >= 0.60: cell = f"{C['yellow']}▒▒{R}  "
        elif p >= 0.30: cell = f"{C['orange']}░░{R}  "
        else:            cell = f"{DIM}  {R}  "
        p_line += cell
    lines.append(p_line + f" {DIM}P(pico){R}")

    return lines


# ══════════════════════════════════════════════════════
#  8. DISPLAY
# ══════════════════════════════════════════════════════
def p_bar(p, w=14):
    f = round(p*w)
    return "█"*f + "░"*(w-f)

def p_col(p):
    if p < 0.40: return C["gray"]
    if p < 0.65: return C["orange"]
    if p < 0.80: return C["yellow"]
    return C["green"]

def _book_bar(price: float, w: int = 20) -> str:
    """Barra visual para o preço de uma linha do order book."""
    f = round(price * w)
    return "█" * f + "░" * (w - f)

def display_orderbook(book: "OrderBook | None", bracket_label: str = "") -> None:
    """Mostra o order book CLOB no terminal (top-3 bids e asks)."""
    if book is None:
        print(f"    {DIM}Order book CLOB indisponível — a usar preço Gamma{R}")
        return

    bid = book.best_bid
    ask = book.best_ask
    spr = book.spread

    bid_str = f"{bid*100:>5.1f}¢" if bid else "  — "
    ask_str = f"{ask*100:>5.1f}¢" if ask else "  — "
    spr_str = f"{spr*100:.1f}¢"   if spr else "—"

    bid_col = C["green"]
    ask_col = C["red"]

    print(f"    {DIM}{'Bid':>8}  {'':20}  {'Ask':>8}  {'Spread':>8}{R}")
    print(f"    {bid_col}{B}{bid_str:>8}{R}  "
          f"{DIM}{_book_bar((bid or 0))}{R}  "
          f"{ask_col}{B}{ask_str:>8}{R}  "
          f"{DIM}{spr_str:>8}{R}")

    # Top 3 bids e asks lado a lado
    n_levels = min(3, max(len(book.bids), len(book.asks)))
    if n_levels > 1:
        print(f"    {DIM}{'─'*52}{R}")
        for i in range(n_levels):
            b_lv = book.bids[i] if i < len(book.bids) else None
            a_lv = book.asks[i] if i < len(book.asks) else None
            b_s  = f"{b_lv.price*100:>5.1f}¢ × {b_lv.size:>6.0f}" if b_lv else " " * 16
            a_s  = f"{a_lv.price*100:>5.1f}¢ × {a_lv.size:>6.0f}" if a_lv else " " * 16
            print(f"    {DIM}  {bid_col}{b_s}{DIM}    {ask_col}{a_s}{R}")

    print(f"    {DIM}Depth bid: ${book.bid_depth_usdc:,.0f}  "
          f"ask: ${book.ask_depth_usdc:,.0f}{R}")


def display_positions(positions: "PositionManager", trading_mode: TradingMode,
                      usdc_balance: float | None = None) -> None:
    """
    Secção do dashboard: posições abertas e histórico.
    Mostra: hoje em destaque, depois todas as abertas, depois as fechadas.
    """
    all_pos  = positions.all_positions()
    open_pos = positions.open_positions()
    summary  = positions.pnl_summary()

    mode_tag = "PAPER" if trading_mode == TradingMode.PAPER else "REAL"

    # Saldo inline na linha do título
    if trading_mode == TradingMode.REAL:
        if usdc_balance is not None:
            bal_col = C["green"] if usdc_balance >= 10 else C["red"]
            bal_part = f"  {DIM}Saldo:{R} {bal_col}{B}${usdc_balance:,.2f}{R}"
        else:
            bal_part = f"  {DIM}Saldo: a carregar...{R}"
    else:
        bal_part = ""

    print(f"\n  {B}Posições [{mode_tag}]{R}{bal_part}  "
          f"{DIM}abertas:{summary['n_open']}  "
          f"ganhas:{summary['n_won']}  "
          f"perdidas:{summary['n_lost']}{R}")

    if not all_pos:
        print(f"    {DIM}Sem posições registadas ainda.{R}")
        return

    # ── Cabeçalho ─────────────────────────────────────
    print(f"  {DIM}  {'Data':<12} {'Bracket':<18} {'Entrada':>7} "
          f"{'Actual':>7} {'P&L $':>8} {'P&L %':>7} {'Shares':>7}  Status{R}")
    print(f"  {DIM}  {'─'*78}{R}")

    today_str = berlin_date().isoformat()

    def _fmt_position(pos: "Position", highlight: bool = False) -> None:
        mid    = pos.current_mid
        pnl_u  = pos.pnl_usd
        pnl_p  = pos.pnl_pct

        # Cor do P&L
        if pnl_u is None:
            pnl_col = DIM
        elif pnl_u > 0:
            pnl_col = C["green"]
        elif pnl_u < 0:
            pnl_col = C["red"]
        else:
            pnl_col = DIM

        # Status e cor
        status_map = {
            PositionStatus.OPEN:    (f"{C['cyan']}ABERTA{R}",    ""),
            PositionStatus.WON:     (f"{C['green']}{B}GANHOU{R}", "✓"),
            PositionStatus.LOST:    (f"{C['red']}PERDEU{R}",     "✗"),
            PositionStatus.EXPIRED: (f"{DIM}EXPIROU{R}",         "—"),
            PositionStatus.UNKNOWN: (f"{DIM}?{R}",               ""),
        }
        status_str, icon = status_map.get(pos.status, (f"{DIM}?{R}", ""))

        entry_s  = f"{pos.entry_ask*100:.1f}¢"
        mid_s    = f"{mid*100:.1f}¢"   if mid   is not None else f"{DIM}—{R}"
        pnl_u_s  = f"{pnl_u:+.2f}"    if pnl_u is not None else f"{DIM}—{R}"
        pnl_p_s  = f"{pnl_p:+.1f}%"   if pnl_p is not None else f"{DIM}—{R}"

        pre = f"  {B}{C['cyan']}▶ {R}" if highlight else "    "

        print(f"{pre}{pos.date_opened:<12} "
              f"{pos.bracket_label:<18} "
              f"{DIM}{entry_s:>7}{R} "
              f"{mid_s:>7} "
              f"{pnl_col}{pnl_u_s:>8}{R} "
              f"{pnl_col}{pnl_p_s:>7}{R} "
              f"{DIM}{pos.shares:>7.2f}{R}  "
              f"{status_str} {icon}")

    # ── Posição de hoje (destaque) ─────────────────────
    shown_ids: set[str] = set()
    for pos in reversed(all_pos):
        if pos.date_opened == today_str:
            _fmt_position(pos, highlight=True)
            shown_ids.add(pos.order_id)
            break

    # ── Restantes abertas (dias anteriores) ───────────
    other_open = [p for p in open_pos
                  if p.order_id not in shown_ids]
    if other_open:
        print(f"  {DIM}  {'─'*78}{R}")
        for pos in sorted(other_open, key=lambda p: p.date_opened, reverse=True):
            _fmt_position(pos, highlight=False)
            shown_ids.add(pos.order_id)

    # ── Fechadas (won/lost) — máx 5 mais recentes ─────
    closed = [p for p in all_pos
              if p.status in (PositionStatus.WON, PositionStatus.LOST, PositionStatus.EXPIRED)
              and p.order_id not in shown_ids]
    if closed:
        print(f"  {DIM}  {'─'*78}{R}")
        for pos in sorted(closed, key=lambda p: p.date_opened, reverse=True)[:5]:
            _fmt_position(pos, highlight=False)

    # ── Totais ────────────────────────────────────────
    if len(all_pos) > 0:
        pnl_col = C["green"] if summary["total_pnl_usd"] >= 0 else C["red"]
        print(f"  {DIM}  {'─'*78}{R}")
        print(f"    {DIM}Total investido: ${summary['total_invested']:.2f}   "
              f"P&L total: {R}"
              f"{pnl_col}{B}{summary['total_pnl_usd']:+.2f} "
              f"({summary['total_pnl_pct']:+.1f}%){R}")


def display(now, latest_obs, temps_by_hour, series_today, signals, p,
            market, bracket, ev, bet,
            n_wu_reads, bankroll, threshold, peak_detected,
            trading_mode: TradingMode = TradingMode.PAPER,
            daily_loss: float = 0.0, max_daily_loss: float = 50.0,
            usdc_balance: float | None = None,
            positions: "PositionManager | None" = None,
            executor: "OrderExecutor | None" = None,
            open_orders: list | None = None,
            market_date=None,
            bet_blocked_reason=None, bet_placed=False,
            forecast_max=None, berlin_now_dt=None,
            signal_window_label="",
            obs_min_today: dict = None):

    os.system('clear' if os.name != 'nt' else 'cls')
    pc = p_col(p)

    # ── Header ───────────────────────────────────────
    berlin_str    = berlin_now_dt.strftime('%H:%M:%S') if berlin_now_dt else "?"
    local_str     = now.strftime('%Y-%m-%d %H:%M:%S')
    local_tz_name = getattr(_LOCAL, 'key', 'local') if _LOCAL else 'local'

    mode_tag = (f"{C['yellow']}{B}[ PAPER ]{R}" if trading_mode == TradingMode.PAPER
                else f"{C['red']}{B}[ REAL  ]{R}")

    print(f"\n  {B}{C['cyan']}Munich Max Temp — Live Bot{R}  {mode_tag}  "
          f"{DIM}{local_str} {local_tz_name}  "
          f"│  Munich (CET/CEST) {R}{B}{C['white']}{berlin_str}{R}"
          + signal_window_label)
    print(f"  {DIM}Estacao: EDDM Munich Airport (WUnderground){R}")

    # Saldo USDC e stop-loss (modo REAL)
    if trading_mode == TradingMode.REAL:
        if usdc_balance is not None:
            bal_col = C["green"] if usdc_balance >= 10 else C["red"]
            bal_str = f"{bal_col}{B}${usdc_balance:,.2f} USDC{R}"
        else:
            bal_str = f"{C['yellow']}a carregar...{R}"
        loss_col = C["red"] if daily_loss >= max_daily_loss * 0.8 else C["green"]
        print(f"  {B}Saldo:{R} {bal_str}   "
              f"{DIM}Perda hoje:{R} {loss_col}{B}${daily_loss:.2f}{R}"
              f"{DIM} / stop-loss ${max_daily_loss:.0f}{R}")

    print(f"  {DIM}{'─'*58}{R}")

    # ── Gráfico temperatura ────────────────────────────
    print(f"\n  {B}Curva de temperatura hoje{R}  "
          f"{DIM}(● verde=P>80% amarelo=P>60% laranja=P>30%){R}")
    chart_lines = draw_chart(series_today, signals, peak_detected)
    for line in chart_lines:
        print(line)

    # ── Temperatura actual ────────────────────────────
    if series_today:
        rmax_slot    = max(series_today, key=series_today.get)
        rmax         = series_today[rmax_slot]
        rmax_real_ts = (obs_min_today or {}).get(rmax_slot)
        rmax_time_str = (f"{rmax_real_ts[0]}:{rmax_real_ts[1]:02d}"
                         if rmax_real_ts else f"{rmax_slot[0]}h")
    elif temps_by_hour:
        rmax          = max(temps_by_hour.values())
        rmax_peak_h   = max(temps_by_hour, key=temps_by_hour.get)
        rmax_time_str = f"{rmax_peak_h}h"
    else:
        rmax = 0; rmax_time_str = "?"

    print(f"\n  {B}Temperatura actual{R}  {DIM}({n_wu_reads} leituras WU hoje){R}")
    if latest_obs:
        temp_now  = latest_obs["temp_c"]
        hum_now   = latest_obs.get("humidity", 70)
        wx        = latest_obs.get("wx", "")
        cloud_now = latest_obs.get("cloud_cover", 50)
        cloud_str = {0:"Clear",12:"Few clouds",37:"Partly cloudy",
                     75:"Mostly cloudy",100:"Cloudy"}.get(cloud_now, f"{cloud_now}%")
        print(f"    {B}{C['white']}{int(round(temp_now)):>4}°C{R}  "
              f"{DIM}humidade:{int(round(hum_now))}%  {cloud_str}  {wx}{R}")
    print(f"    {DIM}running max:{R} {B}{C['white']}{int(round(rmax))}°C{R}  "
          f"{DIM}@{R} {C['cyan']}{B}{rmax_time_str}{R}")
    if forecast_max:
        fc_col = C["green"] if forecast_max["temp_max"] <= rmax else C["yellow"]
        print(f"    {DIM}previsao WU :{R} max {fc_col}{B}{forecast_max['temp_max']}°C{R}  "
              f"{DIM}min {forecast_max.get('temp_min','?')}°C{R}")

    # ── Sinal do modelo ───────────────────────────────
    print(f"\n  {B}Modelo LightGBM — P(pico ja ocorreu){R}")
    print(f"    {pc}{B}{p_bar(p)}{R}  {pc}{B}{p*100:>5.1f}%{R}  "
          f"{DIM}threshold: {threshold*100:.0f}%{R}")

    if peak_detected:
        status = f"{C['green']}{B}✓ PICO DETECTADO{R}"
    elif p >= 0.60:
        status = f"{C['yellow']}◷ aguardar — {p*100:.0f}% ({100-p*100:.0f}% para threshold){R}"
    else:
        status = f"{C['gray']}○ monitoring — pico provavelmente nao ocorreu ainda{R}"
    print(f"    {status}")

    # ── Mercado Polymarket + Order Book ───────────────
    _md        = market_date or berlin_date()
    _today_ref = berlin_date()
    if _md > _today_ref:
        market_label = (f"{B}Polymarket — Mercado de {_md}{R}  "
                        f"{C['yellow']}{B}[AMANHÃ / FUTURO]{R}")
    else:
        market_label = f"{B}Polymarket — Mercado de Hoje  {DIM}({_md}){R}"
    print(f"\n  {market_label}")
    if not market:
        print(f"    {C['red']}✗ Mercado nao encontrado{R}  "
              f"{DIM}(sera criado esta manha pelo Polymarket){R}")
    else:
        title_str = market["title"][:56]
        print(f"    {B}{C['cyan']}{title_str}{R}")
        print(f"    {DIM}vol: ${market['volume']:,.0f}  "
              f"encerra: {market['end_date'][:10]}  "
              f"{market['n_outcomes']} brackets{R}")
        print()
        # Cabeçalho tabela de brackets — agora com Bid / Ask / Spread
        print(f"  {DIM}  {'Bracket':<18}  {'Bid':>6}  {'Ask':>6}  {'Spread':>6}  {'Bar':16}  {'Vol':>7}{R}")
        print(f"  {DIM}  {'─'*70}{R}")
        for b in market["brackets"]:
            is_t = bracket and b["label"] == bracket["label"]

            b_ask    = b.get("ask") or b.get("price") or 0
            b_bid    = b.get("bid") or b_ask
            b_spread = b.get("spread")

            bw   = 16
            bf   = round(b_ask * bw)
            bbar = "█" * bf + "░" * (bw - bf)

            if b_ask < 0.20:   pc2 = C["green"]
            elif b_ask < 0.50: pc2 = C["cyan"]
            elif b_ask < 0.75: pc2 = C["yellow"]
            else:               pc2 = C["orange"]

            pre  = f"  {B}{C['green']}→ {R}" if is_t else "    "
            tag  = f"  {B}{C['green']}◆ running max{R}" if is_t else ""
            vol_s = f"{DIM}${b.get('volume',0):>6,.0f}{R}" if b.get("volume") else ""
            spr_s = (f"{b_spread*100:.1f}¢" if b_spread else f"{DIM}—{R}")

            print(f"{pre}{b['label']:<18}  "
                  f"{C['green']}{b_bid*100:>5.1f}¢{R}  "
                  f"{C['red']}{b_ask*100:>5.1f}¢{R}  "
                  f"{DIM}{spr_s:>6}{R}  "
                  f"{pc2}{bbar}{R}  "
                  f"{vol_s}{tag}")

        # Order book detalhado do bracket seleccionado
        if bracket and bracket.get("book"):
            print(f"\n  {B}Order Book CLOB — {bracket['label']}{R}")
            display_orderbook(bracket.get("book"), bracket.get("label",""))

    # ── EV ───────────────────────────────────────────
    if bracket and ev:
        print(f"\n  {B}Edge Analysis{R}  {DIM}(EV calculado sobre ask){R}")
        ec = C["green"] if ev["ev_positive"] else C["red"]
        ask_disp = f"{ev['ask']*100:.1f}¢"
        print(f"    Ask: {C['red']}{B}{ask_disp}{R}  "
              f"EV/share: {ec}{B}{ev['ev_cents']:+.1f}¢{R}  "
              f"edge: {ec}{ev['edge_pct']:+.1f}%{R}  "
              f"Kelly: {B}{ev['kelly']*100:.1f}%{R}  "
              f"bankroll: ${bankroll:.0f}")

    # ── Posição de hoje ───────────────────────────────
    if positions is not None:
        display_positions(positions, trading_mode, usdc_balance=usdc_balance)
    else:
        print(f"\n  {B}Posições{R}  {DIM}CLOB não disponível{R}")

    # ── Stop-loss warning ─────────────────────────────
    if trading_mode == TradingMode.REAL and daily_loss >= max_daily_loss:
        print(f"\n  {C['red']}{B}⛔  STOP-LOSS DIÁRIO ATINGIDO — novas ordens bloqueadas{R}")

    # ── Estado da bet ────────────────────────────────
    if not bet and peak_detected and not bet_placed:
        if bet_blocked_reason:
            print(f"\n  {C['yellow']}⚠  Bet bloqueada: {bet_blocked_reason}{R}")
    elif bet_placed and not bet:
        mode_label = "simulada" if trading_mode == TradingMode.PAPER else "enviada"
        print(f"\n  {DIM}  Ordem já {mode_label} anteriormente{R}")

    # ── Bet / Ordem ───────────────────────────────────
    if bet:
        if trading_mode == TradingMode.PAPER:
            header_label = f"{C['yellow']}{B}  ◆  BET SIMULADA (PAPER)  ◆{R}"
            border_col   = C["yellow"]
        else:
            header_label = f"{C['red']}{B}  ◆  ORDEM ENVIADA (REAL)   ◆{R}"
            border_col   = C["red"]

        print(f"\n  {border_col}{B}{'─'*44}{R}")
        print(f"  {header_label}")
        print(f"    Bracket    : {bet['bracket']}")
        print(f"    Bid / Ask  : {bet.get('bid',0)*100:.1f}¢  /  {bet['ask']*100:.1f}¢  "
              f"(spread {bet.get('spread',0)*100:.1f}¢)" if bet.get('spread') else
              f"    Ask        : {bet['ask']*100:.1f}¢")
        print(f"    Kelly      : {bet['kelly_full']:.1f}% × {bet['kelly_frac']} "
              f"= {bet['kelly_full']*bet['kelly_frac']:.1f}%")
        print(f"    Aposta     : ${bet['bet_size']:.2f}  ({bet['shares']:.2f} shares)")
        print(f"    Max profit : +${bet['max_profit']:.2f}")
        if bet.get("order_id"):
            print(f"    Order ID   : {bet['order_id']}")
        if bet.get("status"):
            print(f"    Status     : {bet['status']}")
        print(f"  {border_col}{B}{'─'*44}{R}")

    # ── Painel Polymarket — Balance + Ordens Abertas ──
    print(f"\n  {DIM}{'─'*58}{R}")
    if trading_mode == TradingMode.REAL:
        # Saldo
        if usdc_balance is not None:
            bal_col = C["green"] if usdc_balance >= 10 else C["red"]
            bal_disp = f"{bal_col}{B}${usdc_balance:,.2f} USDC{R}"
        else:
            bal_disp = f"{DIM}a carregar...{R}"
        print(f"  {B}Polymarket{R}  Saldo: {bal_disp}", end="")

        # Ordens abertas no CLOB
        if open_orders is not None:
            n_open = len(open_orders)
            if n_open == 0:
                print(f"   {DIM}Ordens abertas: 0{R}")
            else:
                print(f"   {C['yellow']}{B}Ordens abertas: {n_open}{R}")
                for o in open_orders[:5]:   # máx 5
                    oid   = str(o.get("id") or o.get("orderID") or "?")[:12]
                    side  = o.get("side", "?")
                    price = o.get("price", "?")
                    size  = o.get("size", "?")
                    print(f"    {DIM}{oid}  {side}  price={price}  size={size}{R}")
        else:
            print()   # nova linha
    else:
        # PAPER — só mostrar posições simuladas
        n_sim = len(positions.all_positions()) if positions else 0
        print(f"  {B}Polymarket{R}  {DIM}[PAPER — sem ordens reais]  "
              f"posições simuladas: {n_sim}{R}")

    print(f"\n  {DIM}WU reads hoje: {n_wu_reads}  Ctrl+C para parar{R}\n")

    # ── Prompt de entrada forçada ─────────────────────
    # Aparece sempre, exceto se stop-loss atingido em modo REAL
    stop_loss_hit = (trading_mode == TradingMode.REAL
                     and daily_loss >= max_daily_loss)


def reset_market_state_for_future():
    """
    Reset apenas do estado relacionado com decisão de mercado,
    NÃO apaga dados meteorológicos do dia actual.
    """
    return {
        "peak_detected": False,
        "bet_placed": False,
        "signals": {},
        "forecast_max": None,
        "bracket": None,
    }
# ══════════════════════════════════════════════════════
#  9. LOGGING
# ══════════════════════════════════════════════════════
def log_tick(now, temp, p, peak_detected, bracket, ev, bet,
             path, trading_mode: TradingMode = TradingMode.PAPER,
             bet_blocked_reason=None):
    import csv as _csv

    # --- SAFE ACCESS HELPERS ---
    def safe_get(d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else default

    # --- BRACKET SAFE EXTRACTION ---
    bracket_label = safe_get(bracket, "label")
    ask = safe_get(bracket, "ask")
    bid = safe_get(bracket, "bid")
    price = safe_get(bracket, "price")
    spread = safe_get(bracket, "spread")

    # fallback inteligente
    if ask is None:
        ask = price

    # --- EV / BET SAFE ---
    ev_cents = safe_get(ev, "ev_cents")
    bet_size = safe_get(bet, "bet_size")
    order_id = safe_get(bet, "order_id")

    # --- ROW ---
    row = {
        "timestamp":          now.isoformat(),
        "mode":               trading_mode.value,
        "temp":               temp,
        "p_peak":             round(p, 4) if p is not None else None,
        "peak_detected":      peak_detected,

        "bracket":            bracket_label,
        "ask":                ask,
        "bid":                bid,
        "spread":             spread,

        "ev_cents":           ev_cents,
        "bet_size":           bet_size,
        "bet_placed":         bet is not None,
        "order_id":           order_id,

        "bet_blocked_reason": bet_blocked_reason or "",
    }

    # --- DEBUG INTELIGENTE (IMPORTANTE) ---
    if bracket is None:
        print("⚠ log_tick: bracket=None (provavelmente ainda sem forecast)")

    # --- WRITE CSV ---
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
# ══════════════════════════════════════════════════════
#  10. ARRANQUE INTERACTIVO — escolha de modo
# ══════════════════════════════════════════════════════

def _show_manual_entry_prompt(
    bracket, ev, p, bankroll, kelly_frac,
    trading_mode, executor, market, bets, bets_path, on_placed,
) -> None:
    """
    Pergunta no rodapé do dashboard se o utilizador quer forçar entrada.
    Aparece sempre que não há aposta colocada — independentemente do modelo.
    Útil para testes do fluxo completo de ordem.

    Usa select() com timeout=0 para não bloquear o loop — se não houver
    input pronto, avança imediatamente. Se sys.stdin não suportar select
    (Windows), usa uma thread daemon em vez disso.
    """
    import sys

    # Tentar verificar se há input disponível sem bloquear
    has_input = False
    try:
        import select as _sel
        has_input = bool(_sel.select([sys.stdin], [], [], 0)[0])
    except Exception:
        # Windows / ambientes sem select — não mostrar prompt interactivo
        # (evita bloquear o loop)
        pass

    # Imprimir sempre o rodapé visual — o utilizador pode premir Enter a qualquer momento
    ask_disp = f"{ev['ask']*100:.1f}¢" if ev else "—"
    ev_disp  = f"{ev['ev_cents']:+.1f}¢" if ev else "—"
    mode_tag = f"{C['yellow']}PAPER{R}" if trading_mode == TradingMode.PAPER else f"{C['red']}REAL{R}"

    print(
        f"\n  {C['yellow']}{'─'*52}{R}\n"
        f"  {C['yellow']}{B}  ⚡  OVERRIDE MANUAL  [{mode_tag}{C['yellow']}]{R}\n"
        f"  {DIM}  bracket: {bracket['label']}   ask: {ask_disp}   "
        f"EV: {ev_disp}   P(modelo): {p*100:.0f}%{R}\n"
        f"  {C['yellow']}  Entrar no mercado na mesma?  "
        f"[{C['green']}y{R}{C['yellow']}] sim   [{C['red']}n{R}{C['yellow']} / Enter] não{R}\n"
        f"  {C['yellow']}{'─'*52}{R}"
    )

    if not has_input:
        # Sem input pendente — próximo tick verá a pergunta e poderá responder
        return

    try:
        ans = sys.stdin.readline().strip().lower()
    except Exception:
        return

    if ans != "y":
        return

    # ── Construir e executar a ordem ─────────────────
    bet_record = build_bet_record(bracket, p, ev, bankroll, kelly_frac, trading_mode)

    if trading_mode == TradingMode.PAPER:
        result = paper_buy(
            token_id  = bracket.get("token_id", ""),
            price     = ev["ask"],
            size_usdc = bet_record["bet_size"],
            label     = bracket["label"],
        )
        bet_record["order_id"] = result["order_id"]
        bet_record["status"]   = result["status"]
        print(f"\n  {C['yellow']}✓ Entrada manual PAPER — {bracket['label']} "
              f"a {ev['ask']*100:.1f}¢  ${bet_record['bet_size']:.2f}{R}")
        on_placed(bet_record)

    else:
        # REAL: confirmação extra mesmo no override
        if confirm_real_order(bet_record):
            if not executor:
                print(f"\n  {C['red']}✗ OrderExecutor não disponível{R}")
                return
            result = executor.buy(
                token_id  = bracket.get("token_id", ""),
                price     = ev["ask"],
                size_usdc = bet_record["bet_size"],
                label     = bracket["label"],
            )
            if result["success"]:
                bet_record["order_id"] = result["order_id"]
                bet_record["status"]   = result["status"]
                print(f"\n  {C['red']}✓ Entrada manual REAL — ordem {result['order_id']}{R}")
                on_placed(bet_record)
            else:
                print(f"\n  {C['red']}✗ Falha: {result['error']}{R}")
                time.sleep(2)


# ══════════════════════════════════════════════════════
#  INPUT NÃO-BLOQUEANTE — verifica se o utilizador escreveu algo
# ══════════════════════════════════════════════════════

def _stdin_has_input(timeout: float = 0.0) -> bool:
    """
    Devolve True se há texto disponível em stdin sem bloquear.
    Funciona em Unix/macOS (select) e Windows (msvcrt).
    """
    import sys
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(r)
    except (ImportError, AttributeError):
        # Windows fallback
        try:
            import msvcrt
            return msvcrt.kbhit()
        except ImportError:
            return False


def _read_stdin_line() -> str:
    """Lê uma linha de stdin (chamada apenas depois de _stdin_has_input() == True)."""
    import sys
    try:
        return sys.stdin.readline().strip().lower()
    except Exception:
        return ""


def execute_forced_entry(
    bracket, ask_price, p, ev,
    bankroll, kelly_frac,
    trading_mode, executor, market,
    bets, bets_path,
) -> tuple[dict | None, str | None]:
    """
    Executa uma entrada forçada no mercado (ignora peak_detected e bet_placed).
    Devolve (bet_record | None, erro | None).
    """
    if not bracket or not ask_price or not ev:
        return None, "sem bracket ou preço disponível"

    if not ev["ev_positive"]:
        print(f"\n  {C['yellow']}⚠  EV negativo ({ev['ev_cents']:+.1f}¢) — entrar mesmo assim? (s/n): {R}",
              end="", flush=True)
        try:
            ans = input("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, "cancelado"
        if ans != "s":
            return None, "cancelado pelo utilizador"

    bet_record = build_bet_record(bracket, p, ev, bankroll, kelly_frac, trading_mode)

    if trading_mode == TradingMode.PAPER:
        result = paper_buy(
            token_id  = bracket.get("token_id", ""),
            price     = ask_price,
            size_usdc = bet_record["bet_size"],
            label     = bracket["label"],
        )
        bet_record["order_id"] = result["order_id"]
        bet_record["status"]   = result["status"]
        return bet_record, None

    else:
        # REAL — confirmação manual obrigatória
        if confirm_real_order(bet_record):
            if not executor:
                return None, "OrderExecutor não disponível"
            result = executor.buy(
                token_id  = bracket.get("token_id", ""),
                price     = ask_price,
                size_usdc = bet_record["bet_size"],
                label     = bracket["label"],
            )
            if result["success"]:
                bet_record["order_id"] = result["order_id"]
                bet_record["status"]   = result["status"]
                return bet_record, None
            else:
                return None, result["error"]
        else:
            return None, "confirmação recusada"


def ask_trading_mode() -> TradingMode:
    """Pergunta interactiva no terminal para escolher Paper ou Real."""
    print(f"\n  {B}{C['cyan']}── Munich Live Bot — Selecção de Modo ──────────{R}")
    print(f"  {C['yellow']}[P]{R} PAPER  — simula ordens, order book real do CLOB")
    print(f"  {C['red']}[R]{R} REAL   — envia ordens reais ao Polymarket CLOB")
    print()

    while True:
        try:
            ans = input(f"  Modo? {C['yellow']}[P]{R}aper / {C['red']}[R]{R}eal : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  A sair.")
            raise SystemExit(0)

        if ans in ("p", "paper", ""):
            print(f"\n  {C['yellow']}{B}Modo PAPER seleccionado.{R}  "
                  f"{DIM}Ordens simuladas — nenhum dinheiro real será gasto.{R}\n")
            return TradingMode.PAPER

        if ans in ("r", "real"):
            if not POLY_PRIVATE_KEY:
                print(f"\n  {C['red']}{B}✗  POLY_PRIVATE_KEY não definida.{R}")
                print(f"  {C['red']}Não é possível usar modo REAL sem a chave.{R}\n")
                print("  Define antes de arrancar:")
                print("    export POLY_PRIVATE_KEY=0x...   (Linux/macOS)")
                print("    set POLY_PRIVATE_KEY=0x...       (Windows CMD)\n")
                try:
                    alt = input(f"  Continuar em modo PAPER? ({C['yellow']}s{R}/{C['red']}n{R}): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit(0)
                if alt == "s":
                    print(f"\n  {C['yellow']}{B}Modo PAPER seleccionado.{R}\n")
                    return TradingMode.PAPER
                else:
                    print("  A sair.")
                    raise SystemExit(0)

            # Confirmação extra para REAL + check de saldo
            print(f"\n  {C['red']}{B}⚠  MODO REAL — ordens reais serão enviadas ao Polymarket.{R}")
            print(f"  Stop-loss diário: ${POLY_MAX_DAILY_LOSS:.0f} USDC")

            # ── Verificar saldo USDC antes de confirmar ───
            print(f"  {DIM}A verificar saldo USDC...{R}", end=" ", flush=True)
            usdc_balance_check = None
            try:
                from polymarket_clob import ClobClient as _CC, TradingMode as _TM
                _tmp = _CC(
                    private_key    = POLY_PRIVATE_KEY,
                    mode           = TradingMode.REAL,
                    max_daily_loss = POLY_MAX_DAILY_LOSS,
                    log_dir        = LOG_DIR,
                )
                usdc_balance_check = _tmp.get_usdc_balance()
            except Exception as e:
                print(f"{C['yellow']}indisponível ({e}){R}")

            if usdc_balance_check is not None:
                bal_col = C["green"] if usdc_balance_check >= 10 else C["red"]
                print(f"{bal_col}{B}${usdc_balance_check:,.2f} USDC{R}")
                if usdc_balance_check < 1.0:
                    print(f"\n  {C['red']}{B}✗  Saldo insuficiente (${usdc_balance_check:.2f}).{R}")
                    print(f"  {C['red']}Deposita USDC na tua wallet Polymarket antes de usar modo REAL.{R}\n")
                    try:
                        alt = input(f"  Continuar em modo PAPER? ({C['yellow']}s{R}/{C['red']}n{R}): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        raise SystemExit(0)
                    if alt == "s":
                        print(f"\n  {C['yellow']}{B}Modo PAPER seleccionado.{R}\n")
                        return TradingMode.PAPER
                    else:
                        raise SystemExit(0)
                elif usdc_balance_check < 10.0:
                    print(f"  {C['yellow']}⚠  Saldo baixo — aposta máxima limitada ao saldo disponível.{R}")

            try:
                confirm = input(f"  Confirmas? (escreve {C['red']}REAL{R} para confirmar): ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)

            if confirm == "REAL":
                print(f"\n  {C['red']}{B}Modo REAL activado.{R}\n")
                return TradingMode.REAL
            else:
                print(f"  {DIM}Confirmação inválida — a usar PAPER.{R}\n")
                return TradingMode.PAPER

        print(f"  {C['yellow']}Opção inválida. Escreve P ou R.{R}")


def confirm_real_order(bet: dict) -> bool:
    """Confirmação manual antes de enviar uma ordem REAL."""
    print(f"\n  {C['red']}{B}{'═'*46}{R}")
    print(f"  {C['red']}{B}  ⚠  CONFIRMAR ORDEM REAL  ⚠{R}")
    print(f"  {C['red']}{B}{'═'*46}{R}")
    print(f"    Bracket : {bet['bracket']}")
    print(f"    Ask     : {bet['ask']*100:.1f}¢  (spread {bet.get('spread',0)*100:.1f}¢)")
    print(f"    Aposta  : ${bet['bet_size']:.2f}  ({bet['shares']:.2f} shares YES)")
    print(f"    Max prof: +${bet['max_profit']:.2f}")
    print(f"    EV      : {bet['ev_cents']:+.1f}¢/share   edge: {bet['edge_pct']:+.1f}%")
    print(f"  {C['red']}{B}{'─'*46}{R}")
    try:
        ans = input(f"  Enviar ordem? ({C['green']}y{R}/{C['red']}n{R}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans == "y"


# ══════════════════════════════════════════════════════
#  11. MAIN LOOP
# ══════════════════════════════════════════════════════
def run(wu_key: str, threshold: float, bankroll: float,
        kelly_frac: float, min_edge: float, interval: int,
        no_risk: bool = False,
        headless: bool = False,
        forced_mode: "TradingMode | None" = None):

    LOG_DIR.mkdir(exist_ok=True)

    if not wu_key:
        raise ValueError(
            f"\n  {C['red']}WU_API_KEY nao definida.{R}\n"
            "  Define a variavel de ambiente antes de correr:\n"
            "    export WU_API_KEY=\"a_tua_chave\"    (Linux/macOS)\n"
            "    set WU_API_KEY=a_tua_chave           (Windows CMD)\n"
            "  Obtem a chave em: https://www.wunderground.com/member/api-keys"
        )

    # ── Escolha de modo ───────────────────────────────
    if forced_mode is not None:
        trading_mode = forced_mode
        mode_str = "REAL" if trading_mode == TradingMode.REAL else "PAPER"
        if headless:
            print(f"  [Headless] Modo: {mode_str} — sem perguntas interactivas.")
        else:
            print(f"  Modo definido por argumento: {mode_str}")
        # Em REAL sem headless, ainda valida a chave privada
        if trading_mode == TradingMode.REAL and not POLY_PRIVATE_KEY:
            raise ValueError("POLY_PRIVATE_KEY não definida. Impossível usar modo REAL.")
    else:
        trading_mode = ask_trading_mode()

    # ── Telegram ──────────────────────────────────────
    tg = TG()

    # ── Inicializar CLOB client (order book + posições) ──
    clob = None
    if trading_mode == TradingMode.REAL and not POLY_PRIVATE_KEY:
        raise ValueError("POLY_PRIVATE_KEY não definida. Impossível usar modo REAL.")

    if POLY_PRIVATE_KEY:
        print(f"  {DIM}A inicializar cliente CLOB Polymarket...{R}", end=" ", flush=True)
        try:
            clob = ClobClient(
                private_key    = POLY_PRIVATE_KEY,
                mode           = trading_mode,
                max_daily_loss = POLY_MAX_DAILY_LOSS,
                log_dir        = LOG_DIR,
            )
            print(f"{C['green']}✓{R}")
        except Exception as e:
            print(f"{C['red']}✗ {e}{R}")
            print(f"  {DIM}A continuar sem CLOB (sem order book).{R}")
            clob = None
    else:
        print(f"  {DIM}POLY_PRIVATE_KEY não definida — modo PAPER sem order book CLOB.{R}")

    # ── Inicializar OrderExecutor (execução de ordens) ──
    executor = None
    if POLY_PRIVATE_KEY:
        print(f"  {DIM}A inicializar OrderExecutor...{R}", end=" ", flush=True)
        try:
            executor = OrderExecutor(POLY_PRIVATE_KEY)
            print(f"{C['green']}✓{R}")
        except Exception as e:
            print(f"{C['red']}✗ {e}{R}")
            executor = None

    today     = berlin_date()
    log_path  = LOG_DIR / f"live_{today}.csv"
    bets_path = LOG_DIR / f"bets_{today}.json"
    wu_sess   = make_wu_session()

    # ── Em modo REAL, bankroll = saldo real da conta ──
    if trading_mode == TradingMode.REAL and executor:
        real_balance = executor.get_balance()
        if real_balance is not None and real_balance > 0:
            bankroll = real_balance
        else:
            print(f"  {C['yellow']}⚠  Saldo indisponível — a usar bankroll do argumento (${bankroll:.2f}){R}")

    print(f"\n{B}{C['cyan']}── Munich Live Bot ──────────────────────────────{R}")
    mode_label = f"{C['yellow']}PAPER{R}" if trading_mode == TradingMode.PAPER else f"{C['red']}REAL{R}"
    print(f"  Modo      : {mode_label}")
    print(f"  Threshold : {threshold*100:.0f}%   Min edge: {min_edge}%")
    if trading_mode == TradingMode.REAL:
        print(f"  Bankroll  : {C['green']}{B}${bankroll:.2f} USDC{R}  {DIM}(saldo real da conta){R}   Kelly: ×{kelly_frac}")
        print(f"  Stop-loss : ${POLY_MAX_DAILY_LOSS:.0f} USDC/dia")
    else:
        print(f"  Bankroll  : ${bankroll:.2f}   Kelly: ×{kelly_frac}  {DIM}(simulado){R}")
    print(f"  Intervalo : {interval}s  (WU scraping)")
    print()

    # Carregar modelo
    print("[1/4] A carregar modelo...")
    model, feat_cols, prior_map, monthly_threshold = load_model()
    set_seasonal_prior(prior_map)

    def get_threshold(month: int) -> float:
        return monthly_threshold.get(month, threshold) if monthly_threshold else threshold

    # Bootstrap
    print(f"\n[2/4] Bootstrap — historico de hoje...")
    series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
    obs_min_today = dict(getattr(bootstrap_today, "_obs_min", {}))
    temps_by_hour = {s["hour"]: s["temp_c"] for s in slots_so_far}

    print(f"\n[3/4] Cloud cover das observacoes EDDM...", end=" ", flush=True)
    rows_cache    = getattr(bootstrap_today, "_rows_cache", [])
    cloud_by_hour = cloud_from_series(series_today, rows_cache)
    print(f"{C['green']}✓{R}")

    # Histórico diário (prev7)
    history_max = init_history_max()
    update_history_max(history_max, slots_so_far)

    print(f"\n[4/4] A aplicar modelo ao historico ({len(slots_so_far)} slots 30min)...")
    month   = today.month
    doy     = today.timetuple().tm_yday
    signals = {}

    for i, slot in enumerate(slots_so_far):
        h  = slot["hour"]
        s  = slot["slot30"]
        if h < MIN_HOUR or i < 3:
            continue
        current_extra = {
            "hour": h, "slot30": s,
            "cloud_cover":     slot.get("cloud_cover", 50),
            "humidity":        slot.get("humidity", 70),
            "prev_7d_avg_max": compute_prev7(history_max, today),
        }
        p = predict_p(model, feat_cols, slots_so_far[:i+1], current_extra, month, doy)
        signals[(h, s)] = p

    peak_detected = any(p >= get_threshold(month) for p in signals.values())

    # market_date: data do mercado Polymarket a consultar.
    # Pode diferir de today quando o utilizador escolhe "passar para amanhã".
    # today: sempre a data real de Berlin (para temperatura e logs).
    market_date  = today
    market       = fetch_market(market_date)
    forecast_max = fetch_wu_forecast_max(wu_key, wu_sess)

    # Enriquecer brackets com CLOB (bid/ask/spread)
    if market and clob:
        market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]

    # Saldo inicial (modo REAL)
    usdc_balance = executor.get_balance() if (trading_mode == TradingMode.REAL and executor) else None
    open_orders  = executor.get_open_orders() if (trading_mode == TradingMode.REAL and executor) else None

    # ── Alertas de arranque ───────────────────────────
    clob_mode_str    = "real" if trading_mode == TradingMode.REAL else "paper"
    threshold_month  = get_threshold(today.month)

    tg.alert_started(
        mode            = clob_mode_str,
        bankroll        = bankroll,
        threshold_arg   = threshold,
        threshold_month = threshold_month,
        month           = today.month,
        market          = market,
        today           = today,
    )
    if not market:
        tg.alert_no_market(today)

    # Estado TG — dashboard periódica a cada 30 min
    # Colocar no passado para forçar envio imediato no primeiro tick
    _tg_last_dashboard    = 0
    _tg_dashboard_interval = 30 * 60

    print(f"\n  {DIM}A iniciar loop — Ctrl+C para parar{R}\n")
    time.sleep(2)

    latest_obs = None
    if slots_so_far:
        last = slots_so_far[-1]
        latest_obs = {
            "temp_c":      last["temp_c"],
            "humidity":    last.get("humidity", 70),
            "cloud_cover": last.get("cloud_cover", 50),
            "wx":          "",
            "hour":        last["hour"],
            "minute":      last["slot30"],
        }

    bet_placed   = False
    bets         = []

    try:
        while True:
            now = local_now()

            # Fora do horário ativo
            if not (BOT_ACTIVE_START <= now.hour < BOT_ACTIVE_END):
                time.sleep(60)
                continue

            # Novo dia
            station_date = berlin_date()
            if station_date != today:
                today         = station_date
                market_date   = today          # reset: mercado volta a ser o de hoje
                slots_so_far  = []
                series_today  = {}
                obs_min_today = {}
                temps_by_hour = {}
                signals       = {}
                peak_detected = False
                bet_placed    = False
                bets          = []
                log_path      = LOG_DIR / f"live_{today}.csv"
                bets_path     = LOG_DIR / f"bets_{today}.json"
                month         = today.month
                doy           = today.timetuple().tm_yday
                market        = fetch_market(market_date)

                series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
                obs_min_today  = dict(getattr(bootstrap_today, "_obs_min", {}))
                rows_cache     = getattr(bootstrap_today, "_rows_cache", [])
                cloud_by_hour  = cloud_from_series(series_today, rows_cache)
                temps_by_hour  = {s["hour"]: s["temp_c"] for s in slots_so_far}

                if market and clob:
                    market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]

            # Última leitura WU
            new_obs = fetch_wu_latest(wu_key, wu_sess)
            if new_obs:
                latest_obs = new_obs
                h_obs, m_obs = new_obs["hour"], new_obs["minute"]
                h_slot, s30  = ceil_slot(h_obs, m_obs)

                if DAY_START <= h_slot <= DAY_END:
                    series_today[(h_slot, s30)] = new_obs["temp_c"]
                    obs_min_today[(h_slot, s30)] = (h_obs, m_obs)

                    slot_entry = {
                        "hour": h_slot, "slot30": s30,
                        "temp_c":      new_obs["temp_c"],
                        "cloud_cover": new_obs.get("cloud_cover", 50),
                        "humidity":    new_obs.get("humidity", 70),
                    }
                    exists = any(sl["hour"] == h_slot and sl["slot30"] == s30
                                 for sl in slots_so_far)
                    if exists:
                        for sl in slots_so_far:
                            if sl["hour"] == h_slot and sl["slot30"] == s30:
                                sl.update(slot_entry)
                                break
                    else:
                        slots_so_far.append(slot_entry)
                        slots_so_far.sort(key=lambda x: x["hour"]*60 + x["slot30"])

                cloud_by_hour[h_slot] = new_obs.get("cloud_cover", 50)

            # Actualizar histórico diário
            update_history_max(history_max, slots_so_far)

            # Calcular P
            h_now  = berlin_now().hour
            m_now  = berlin_now().minute
            h_cur, s30_cur = ceil_slot(h_now, m_now)

            p = 0.0
            if len(slots_so_far) >= 4 and h_cur >= MIN_HOUR:
                current_extra = {
                    "hour":            h_cur,
                    "slot30":          s30_cur,
                    "cloud_cover":     cloud_by_hour.get(h_cur, 50.0),
                    "humidity":        latest_obs.get("humidity", 70) if latest_obs else 70,
                    "prev_7d_avg_max": compute_prev7(history_max, today),
                }
                p = predict_p(model, feat_cols, slots_so_far, current_extra, month, doy)
                signals[(h_cur, s30_cur)] = p

            if p >= get_threshold(month) and not peak_detected:
                peak_detected = True
                # Calcular rmax_time_str para o alerta
                if series_today:
                    _rs = max(series_today, key=series_today.get)
                    _rm = series_today[_rs]
                    _obs = (obs_min_today or {}).get(_rs)
                    _rts = f"{_obs[0]}:{_obs[1]:02d}" if _obs else f"{_rs[0]}h"
                else:
                    _rm  = rmax if 'rmax' in dir() else 0
                    _rts = "?"
                tg.alert_peak_detected(p, _rm, _rts, bracket if 'bracket' in dir() else None)

            # ── Alerta de mudança de zona de P ────────
            if tg.zone_changed(p):
                tg.alert_zone_change(p, tg.p_zone(p))

            # Actualizar mercado e forecast periodicamente
            if now.minute % 10 == 0 or not market:
                market = fetch_market(market_date)
                if market and clob:
                    market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]
            if now.minute % 30 == 0 or forecast_max is None:
                forecast_max = fetch_wu_forecast_max(wu_key, wu_sess)

            # ── Janela de sinal EDDM ──────────────────
            berlin_min = berlin_now().minute
            signal_window_label = ""
            in_signal_window = any(lo <= berlin_min <= hi for lo, hi in _SIGNAL_CHECK_WINDOWS)
            if in_signal_window:
                signal_window_label = (f"  {C['cyan']}◉ a verificar sinal (:20){R}"
                                       if 18 <= berlin_min <= 32 else
                                       f"  {C['cyan']}◉ a verificar sinal (:50){R}")

            # Running max
            if series_today:
                rmax_slot = max(series_today, key=series_today.get)
                rmax      = series_today[rmax_slot]
            elif temps_by_hour:
                rmax = max(temps_by_hour.values())
            else:
                rmax = 0

            eff_thr = get_threshold(month)

            # Temperatura de referência para escolher o bracket:
            # - Se o mercado é de hoje → running max de hoje
            # - Se o mercado é de amanhã+ → previsão WU (NUNCA usar rmax de hoje —
            #   seria sempre o bracket já quase a 100¢ no mercado futuro)
            _future_market    = (market_date != today)
            _forecast_temp    = (forecast_max or {}).get("temp_max")
            _forecast_missing = _future_market and _forecast_temp is None

            if _future_market and _forecast_temp is not None:
                bracket_temp = float(_forecast_temp)
            elif _future_market:
                # Sem previsão WU para mercado futuro: não conseguimos escolher
                # bracket correcto — não arriscar com o rmax de hoje.
                bracket_temp = None
            else:
                bracket_temp = rmax

            bracket = find_bracket(market, bracket_temp) if (market and bracket_temp is not None) else None

            # Enriquecer bracket seleccionado com CLOB se necessário
            if bracket and clob and not bracket.get("book"):
                bracket = clob.enrich_bracket(bracket)

            # EV sobre ask (ou price se ask não disponível)
            ask_price = bracket.get("ask") or bracket.get("price") if bracket else None
            ev  = compute_ev(p, ask_price) if ask_price else None

            # Saldo USDC e ordens abertas — refrescar a cada 5 minutos
            if trading_mode == TradingMode.REAL and executor and now.minute % 5 == 0:
                usdc_balance = executor.get_balance()
                open_orders  = executor.get_open_orders()

            bet             = None
            bet_blocked_reason = None

            if not bet_placed:
                # Verificar stop-loss
                if trading_mode == TradingMode.REAL and clob and clob.stop_loss_triggered():
                    bet_blocked_reason = (f"stop-loss diário atingido "
                                         f"(${clob.daily_loss():.2f} >= ${POLY_MAX_DAILY_LOSS:.0f})")

                elif market_date == today and not peak_detected:
                    # Mercado de HOJE: exige pico detectado.
                    # Mercado FUTURO (amanhã+): ignora peak_detected — a aposta é
                    # sobre temperatura futura, não sobre o pico de hoje.
                    bet_blocked_reason = f"pico nao detectado (P={p*100:.0f}% < {eff_thr*100:.0f}%)"

                elif not market:
                    bet_blocked_reason = "sem mercado Polymarket"

                elif not bracket and _forecast_missing:
                    # Mercado futuro mas WU forecast não disponível —
                    # não podemos escolher bracket sem arriscar o rmax de hoje.
                    # Tentar refrescar o forecast agora mesmo.
                    forecast_max = fetch_wu_forecast_max(wu_key, wu_sess)
                    _forecast_temp2 = (forecast_max or {}).get("temp_max")
                    if _forecast_temp2 is not None:
                        bracket_temp = float(_forecast_temp2)
                        bracket = find_bracket(market, bracket_temp)
                        if bracket and clob and not bracket.get("book"):
                            bracket = clob.enrich_bracket(bracket)
                        ask_price = bracket.get("ask") or bracket.get("price") if bracket else None
                        ev = compute_ev(p, ask_price) if ask_price else None
                        bet_blocked_reason = None   # vai reavaliado abaixo
                    else:
                        bet_blocked_reason = (f"mercado futuro ({market_date}) sem previsão WU — "
                                              f"a aguardar forecast para escolher bracket")

                elif not bracket:
                    bet_blocked_reason = "bracket nao identificado"

                elif p < eff_thr:
                    bet_blocked_reason = f"P={p*100:.0f}% abaixo do threshold {eff_thr*100:.0f}%"

                elif ask_price and ask_price >= 0.95:
                    # Re-capturar market_date aqui para garantir que a mensagem é actual
                    _mdate_str = str(market_date)
                    bet_blocked_reason = f"ask {ask_price*100:.1f}¢ >= 95¢ — mercado: {_mdate_str}"
                    _is_future = (market_date != today)
                    print(f"\n  {C['red']}⚠  Preço ask demasiado alto ({ask_price*100:.1f}¢)"
                          f" — mercado {_mdate_str}"
                          f"{' [FUTURO]' if _is_future else ''}{R}")
                    if _is_future:
                        # Já estamos no mercado futuro e ask ainda >= 95¢:
                        # o bracket escolhido pelo forecast provavelmente é muito certo.
                        # Não avançar mais dias — bloquear e aguardar nova previsão.
                        print(f"  {C['yellow']}Bracket futuro já a {ask_price*100:.1f}¢ — "
                              f"a aguardar nova previsão WU.{R}")
                        # Forçar refresh do forecast na próxima iteração
                        forecast_max = None
                    elif headless:
                        resp = "s"
                        print(f"  {DIM}[Headless] A passar automaticamente para mercado de amanhã.{R}")
                    else:
                        resp = input("  Passar para o mercado de amanhã? (s/n): ").strip().lower()
                    if not _is_future and resp == "s":
                        market_date = market_date + timedelta(days=1)
                        # 🔥 RESET CRÍTICO AQUI
                        _reset = reset_market_state_for_future()
                        peak_detected = _reset["peak_detected"]
                        bet_placed    = _reset["bet_placed"]
                        signals       = _reset["signals"]
                        forecast_max  = _reset["forecast_max"]
                        bracket       = _reset["bracket"]

                        print(f"  {C['yellow']}↺ Reset de estado para mercado futuro{R}")# 🔥 RESET CRÍTICO AQUI
                        _reset = reset_market_state_for_future()
                        peak_detected = _reset["peak_detected"]
                        bet_placed    = _reset["bet_placed"]
                        signals       = _reset["signals"]
                        forecast_max  = _reset["forecast_max"]
                        bracket       = _reset["bracket"]

                        print(f"  {C['yellow']}↺ Reset de estado para mercado futuro{R}")
                        print(f"  {C['cyan']}A carregar mercado {market_date}...{R}", end=" ", flush=True)
                        new_market = fetch_market(market_date)
                        if new_market:
                            market = new_market
                            if clob:
                                market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]
                            bets_path = LOG_DIR / f"bets_{market_date}.json"
                            print(f"{C['green']}✓  {new_market['title'][:50]}{R}")
                        else:
                            market_date = market_date - timedelta(days=1)  # reverter
                            print(f"{C['red']}✗ Mercado não encontrado — a manter {market_date}{R}")

                elif not ev or not ev["ev_positive"]:
                    bet_blocked_reason = f"EV negativo ({ev['ev_cents']:+.1f}¢)" if ev else "EV nao calculavel"

                elif ev["edge_pct"] < min_edge:
                    bet_blocked_reason = f"edge {ev['edge_pct']:.1f}% < min {min_edge:.1f}%"

                else:
                    # ── Construir registo de bet ──────
                    bet_record = build_bet_record(bracket, p, ev, bankroll, kelly_frac, trading_mode)

                    if trading_mode == TradingMode.PAPER:
                        # Simular — usar paper_buy() que regista no log
                        result = paper_buy(
                            token_id  = bracket.get("token_id", ""),
                            price     = ask_price,
                            size_usdc = bet_record["bet_size"],
                            label     = bracket["label"],
                        )
                        bet_record["order_id"] = result["order_id"]
                        bet_record["status"]   = result["status"]
                        bet        = bet_record
                        bet_placed = True

                    else:
                        # ── REAL: confirmação manual + OrderExecutor ──
                        do_place = headless or confirm_real_order(bet_record)
                        if headless:
                            print(f"\n  {DIM}[Headless] Ordem REAL auto-confirmada.{R}")
                        if do_place:
                            if not executor:
                                bet_blocked_reason = "OrderExecutor não disponível"
                            else:
                                result = executor.buy(
                                    token_id  = bracket.get("token_id", ""),
                                    price     = ask_price,
                                    size_usdc = bet_record["bet_size"],
                                    label     = bracket["label"],
                                )
                                if result["success"]:
                                    bet_record["order_id"] = result["order_id"]
                                    bet_record["status"]   = result["status"]
                                    bet        = bet_record
                                    bet_placed = True
                                    # Refrescar saldo após compra
                                    usdc_balance = executor.get_balance()
                                    open_orders  = executor.get_open_orders()
                                    print(f"\n  {C['green']}✓ Ordem enviada — ID: {result['order_id']}{R}")
                                    tg.alert_order_placed(bet_record)
                                else:
                                    bet_blocked_reason = f"Ordem falhou: {result['error']}"
                                    print(f"\n  {C['red']}✗ Falha na ordem: {result['error']}{R}")
                                    tg.alert_order_failed(result["error"], bracket)
                        else:
                            bet_blocked_reason = "confirmação recusada pelo utilizador"

                    if bet:
                        bets.append(bet)
                        bets_path.write_text(json.dumps(bets, indent=2, default=str))
                        # Alerta PAPER também
                        if trading_mode == TradingMode.PAPER:
                            tg.alert_order_placed(bet)

            # Sinais por hora para o dashboard
            signals_by_hour = {}
            for (sh, ss), sp in signals.items():
                if sh not in signals_by_hour or sp > signals_by_hour[sh]:
                    signals_by_hour[sh] = sp

            temp_now = latest_obs["temp_c"] if latest_obs else 0

            # ── Refresh posições (bid actual + resolução REAL) ──
            if clob:
                clob.positions.refresh(clob)

            display(
                now, latest_obs, temps_by_hour, series_today, signals_by_hour, p,
                market, bracket, ev, bet,
                len(series_today), bankroll, eff_thr, peak_detected,
                trading_mode    = trading_mode,
                daily_loss      = clob.daily_loss() if clob else 0.0,
                max_daily_loss  = POLY_MAX_DAILY_LOSS,
                usdc_balance    = usdc_balance,
                positions       = clob.positions if clob else None,
                bet_blocked_reason = bet_blocked_reason,
                bet_placed      = bet_placed,
                forecast_max    = forecast_max,
                berlin_now_dt   = berlin_now(),
                market_date     = market_date,
                executor        = executor,
                open_orders     = open_orders,
                signal_window_label = signal_window_label,
                obs_min_today   = obs_min_today,
            )

            log_tick(
                now, temp_now, p, peak_detected, bracket, ev, bet, log_path,
                trading_mode        = trading_mode,
                bet_blocked_reason  = bet_blocked_reason if not bet_placed else None,
            )

            # ── Entrada forçada (override manual, para testes) ───────────
            # O utilizador pode escrever 'f' + Enter a qualquer momento
            # durante o intervalo de espera para forçar entrada no mercado.
            # Funciona mesmo sem pico detectado e mesmo depois de bet_placed.
            stop_loss_hit = (trading_mode == TradingMode.REAL
                             and clob is not None
                             and clob.stop_loss_triggered())

            if not stop_loss_hit and _stdin_has_input():
                line = _read_stdin_line()
                if line == "f":
                    forced_bet, forced_err = execute_forced_entry(
                        bracket      = bracket,
                        ask_price    = ask_price,
                        p            = p,
                        ev           = ev,
                        bankroll     = bankroll,
                        kelly_frac   = kelly_frac,
                        trading_mode = trading_mode,
                        executor     = executor,
                        market       = market,
                        bets         = bets,
                        bets_path    = bets_path,
                    )
                    if forced_bet:
                        bets.append(forced_bet)
                        bets_path.write_text(json.dumps(bets, indent=2, default=str))
                        bet_placed = True
                        print(f"\n  {C['yellow']}{B}◈  Entrada forçada registada — "
                              f"{forced_bet['bracket']}  ${forced_bet['bet_size']:.2f}{R}\n")
                        time.sleep(2)   # pausa para o utilizador ler
                    elif forced_err:
                        print(f"\n  {C['red']}✗ Entrada forçada cancelada: {forced_err}{R}\n")
                        time.sleep(1)

            time.sleep(interval)

            # ── Dashboard periódica Telegram (30 min) ─
            if time.time() - _tg_last_dashboard >= _tg_dashboard_interval:
                _tg_last_dashboard = time.time()
                _rmax_ts = "?"
                if series_today:
                    _rs = max(series_today, key=series_today.get)
                    _obs = (obs_min_today or {}).get(_rs)
                    _rmax_ts = f"{_obs[0]}:{_obs[1]:02d}" if _obs else f"{_rs[0]}h"
                tg.dashboard(
                    today        = today,
                    p            = p,
                    rmax         = rmax,
                    rmax_time    = _rmax_ts,
                    temp_now     = latest_obs["temp_c"] if latest_obs else None,
                    forecast_max = forecast_max,
                    market       = market,
                    bracket      = bracket,
                    ev           = ev,
                    peak_detected = peak_detected,
                    bet          = bets[-1] if bets else None,
                    clob_mode    = clob_mode_str,
                    reason       = "periodic",
                )

    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Stopped.  Logs em ./{LOG_DIR}/{R}")
        tg.alert_stopped(bets, clob_mode_str)
        if bets:
            mode_label = "simuladas" if trading_mode == TradingMode.PAPER else "reais"
            print(f"  {C['green']}{len(bets)} ordens {mode_label} → {bets_path}{R}")

        # Oferecer fecho manual se houver posição aberta
        if clob:
            today_pos = clob.positions.today_position()
            if today_pos and today_pos.status.value == "open":
                print(f"\n  {C['yellow']}Tens uma posição aberta: {today_pos.bracket_label}  "
                      f"entrada {today_pos.entry_ask*100:.1f}¢{R}")
                if headless:
                    ans = "n"
                    print(f"  {DIM}[Headless] Posição mantida aberta (sem input interactivo).{R}")
                else:
                    try:
                        ans = input(f"  Fechar posição ao bid actual? ({C['green']}y{R}/{C['red']}n{R}): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        ans = "n"

                if ans == "y":
                    book = clob.get_orderbook(today_pos.token_id)
                    bid  = book.best_bid if book else None
                    if bid:
                        result = clob.sell_yes(today_pos, bid)
                        if result.success:
                            pnl = today_pos.pnl_usd or 0
                            sign = "+" if pnl >= 0 else ""
                            print(f"  {C['green']}✓ Posição fechada a {bid*100:.1f}¢  "
                                  f"P&L: {sign}${pnl:.2f}{R}")
                        else:
                            print(f"  {C['red']}✗ Falha ao fechar: {result.error}{R}")
                    else:
                        print(f"  {C['red']}Bid não disponível — posição mantida aberta.{R}")


def reset_daily_state():
    """
    Limpa todas as variáveis relacionadas ao dia corrente.
    Deve ser chamada quando o pico do dia for detectado.
    """
    global last_peak_time
    global bracket
    global forecast_data
    global daily_readings
    global pico_detectado

    last_peak_time = None
    bracket = None
    forecast_data = None
    daily_readings = []
    pico_detectado = False
# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Munich Max Temp — Live Bot (WU + Polymarket + LightGBM)"
    )
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--bankroll",  type=float, default=200.0)
    parser.add_argument("--kelly",     type=float, default=0.5)
    parser.add_argument("--min-edge",  type=float, default=5.0)
    parser.add_argument("--interval",  type=int,   default=60)

    # ── Modo headless (Railway / CI / cron) ──────────
    # Desactiva todos os input() interactivos.
    # Uso: python munich_live_bot.py --headless --mode real
    #      python munich_live_bot.py -headless -mode real   (aliases curtos)
    parser.add_argument(
        "--headless", "-headless",
        action="store_true",
        help="Modo não-interactivo: sem perguntas, sem confirmações manuais."
    )
    parser.add_argument(
        "--mode", "-mode",
        choices=["paper", "real", "PAPER", "REAL"],
        default=None,
        help="Modo de trading: paper (simulação) ou real (ordens reais). "
             "Se omitido e não-headless, pergunta interactivamente."
    )

    args = parser.parse_args()

    # Resolver TradingMode a partir do argumento --mode
    forced_mode = None
    if args.mode:
        forced_mode = (TradingMode.REAL
                       if args.mode.lower() == "real"
                       else TradingMode.PAPER)
    elif args.headless:
        # Headless sem --mode → PAPER por segurança
        forced_mode = TradingMode.PAPER
        print("  [Headless] --mode não especificado — a usar PAPER por defeito.")

    run(
        wu_key      = WU_API_KEY,
        threshold   = args.threshold,
        bankroll    = args.bankroll,
        kelly_frac  = args.kelly,
        min_edge    = args.min_edge,
        interval    = args.interval,
        headless    = args.headless,
        forced_mode = forced_mode,
    )


if __name__ == "__main__":
    main()