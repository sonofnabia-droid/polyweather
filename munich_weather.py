"""
munich_weather.py — UNIFIED
=================
Acesso a dados meteorologicos via WU EDDM + Open-Meteo.

Fontes:
  WU (Weather Underground)  — observacoes reais do aeroporto EDDM
  OM (Open-Meteo)           — previsoes gratuitas, sem API key

Exporta:
  make_wu_session()                                     — session WU
  make_om_session()                                     — session Open-Meteo
  fetch_wu_day_eddm(day, api_key, session)              — historico WU
  fetch_wu_latest(api_key, session)                     — ultima obs WU
  fetch_wu_forecast_max(api_key, session)               — previsao max WU
  fetch_om_forecast_max(session)                        — previsao max OM
  fetch_om_hourly_today(session)                        — previsao horaria OM
  fetch_om_latest(session)                              — ultima previsao OM
  bootstrap_today(api_key, session)                     — bootstrap WU
  bootstrap_om_today(session)                           — bootstrap OM
  cloud_from_series(series_today, rows_cache)           — cloud cover WU
  forecasts_agree(wu_forecast, om_forecast)             — acordo dual
"""

import requests
from datetime import date, datetime, timezone as _tz

from munich_config import (
    WU_BASE, OM_FORECAST, OM_ARCHIVE,
    MUNICH_LAT_OM, MUNICH_LON_OM,
    _BERLIN, DIM, C, R,
    DAY_START, DAY_END,
    FORECAST_AGREEMENT_TOLERANCE,
    berlin_date, ceil_slot,
)


# ══════════════════════════════════════════════════════
#  WUNDERGROUND (EDDM)
# ══════════════════════════════════════════════════════

WU_EDDM_URL = f"{WU_BASE}/EDDM:9:DE/observations/historical.json"


def make_wu_session() -> requests.Session:
    """Session com headers para WU (evita 403)."""
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


def _wu_parse_obs(obs_list: list) -> list[dict]:
    """
    Parser para observacoes WU v1 EDDM.

    Campo de tempo: valid_time_gmt (unix UTC) -> Europe/Berlin.
    Temperatura: campo 'temp' (graus Celsius em metric).
    V2: inclui dewpoint, pressure, wind, uv_index para features do ensemble.
    """
    clds_map = {"CLR": 0, "SKC": 0, "FEW": 12, "SCT": 37, "BKN": 75,
                "OVC": 100, "OBS": 100, "VV": 100, "X": 100}
    rows = []
    for obs in obs_list:
        temp = obs.get("temp")
        if temp is None:
            continue
        vt = obs.get("valid_time_gmt")
        if vt is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(vt), tz=_tz.utc).astimezone(_BERLIN)
        except Exception:
            continue

        clds_raw    = str(obs.get("clds", "") or "").upper().strip()
        cloud_cover = clds_map.get(clds_raw, 50)
        temp_c      = int(round(float(temp)))

        # V2 features com defaults se nao disponiveis
        wspd = obs.get("wspd")
        gust = obs.get("gust")

        rows.append({
            "hour":        dt.hour,
            "minute":      dt.minute,
            "temp_c":      temp_c,
            "humidity":    int(round(float(obs.get("rh") or 70))),
            "cloud_cover": cloud_cover,
            "wx":          str(obs.get("wx_phrase", "") or ""),
            "source":      "WU",
            # V2 features
            "dewpoint_c":     float(obs.get("dewpt") or (temp_c - 10)),
            "pressure_hpa":   float(obs.get("pressure") or 1013),
            "wind_dir_deg":   float(obs.get("wdir") or 0),
            "wind_speed_kmh": float(wspd) * 3.6 if wspd else 5.0,
            "wind_gust_kmh":  float(gust) * 3.6 if gust else 8.0,
            "uv_index":       float(obs.get("uv_index") or 3),
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
    Tenta dois caminhos na resposta (formato antigo e novo).
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

        if not d:
            return None

        t_max = None
        t_min = None

        # Caminho 1: temperatureMax/temperatureMin (formato antigo)
        if "temperatureMax" in d and "temperatureMin" in d:
            t_max_list = d.get("temperatureMax", [None])
            t_min_list = d.get("temperatureMin", [None])
            t_max = (int(round(float(t_max_list[0])))
                     if t_max_list and t_max_list[0] is not None else None)
            t_min = (int(round(float(t_min_list[0])))
                     if t_min_list and t_min_list[0] is not None else None)

        # Caminho 2: daily.temperatureMax/daily.temperatureMin (formato novo)
        elif "daily" in d:
            daily = d["daily"]
            if "temperatureMax" in daily and "temperatureMin" in daily:
                t_max_list = daily.get("temperatureMax", [None])
                t_min_list = daily.get("temperatureMin", [None])
                t_max = (int(round(float(t_max_list[0])))
                         if t_max_list and t_max_list[0] is not None else None)
                t_min = (int(round(float(t_min_list[0])))
                         if t_min_list and t_min_list[0] is not None else None)

        if t_max is None:
            return None

        return {"temp_max": t_max, "temp_min": t_min, "source": "WU"}

    except requests.exceptions.HTTPError as e:
        print(f"  {C['yellow']}WU forecast HTTP {e.response.status_code}{R}")
        return None
    except Exception as e:
        print(f"  {C['yellow']}WU forecast error: {e}{R}")
        return None


def fetch_wu_latest(api_key: str,
                    session: requests.Session) -> dict | None:
    """Leitura mais recente — ultima observacao de hoje (data Berlin/Munich)."""
    rows = fetch_wu_day_eddm(berlin_date(), api_key, session)
    if not rows:
        return None
    return max(rows, key=lambda r: r["hour"] * 60 + r["minute"])


def bootstrap_today(api_key: str,
                    session: requests.Session) -> tuple[dict, list[dict]]:
    """
    Ao arranque: historico completo de hoje via EDDM.
    Devolve:
      series_today : {(hour, slot30): temp_c}  — para o grafico ASCII
      slots_so_far : lista de dicts ordenada por tempo — para o modelo
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

    # Cache para acesso externo
    bootstrap_today._rows_cache = rows

    # series_today para o grafico + obs_min: timestamp real por slot
    series: dict[tuple, float] = {}
    obs_min: dict[tuple, tuple] = {}
    for r in rows:
        key = ceil_slot(r["hour"], r["minute"])
        if key not in series or r["temp_c"] >= series[key]:
            series[key]  = r["temp_c"]
            obs_min[key] = (r["hour"], r["minute"])

    bootstrap_today._obs_min = obs_min

    # slots_so_far para o modelo (cronologico, sem duplicados por slot)
    seen:  set[tuple]  = set()
    slots: list[dict]  = []
    for r in sorted(rows, key=lambda x: x["hour"] * 60 + x["minute"]):
        k = ceil_slot(r["hour"], r["minute"])
        if k not in seen:
            seen.add(k)
            slots.append({
                "hour":        k[0],
                "slot30":      k[1],
                "temp_c":      r["temp_c"],
                "cloud_cover": r.get("cloud_cover", 50),
                "humidity":    r.get("humidity", 70),
                "source":      r.get("source", "WU"),
                # V2 features
                "dewpoint_c":      r.get("dewpoint_c", r["temp_c"] - 10),
                "pressure_hpa":    r.get("pressure_hpa", 1013),
                "wind_dir_deg":    r.get("wind_dir_deg", 0),
                "wind_speed_kmh":  r.get("wind_speed_kmh", 5),
                "wind_gust_kmh":   r.get("wind_gust_kmh", 8),
                "uv_index":        r.get("uv_index", 3),
            })

    return series, slots


def cloud_from_series(series_today: dict, rows_cache: list) -> dict[int, int]:
    """Extrai cloud_cover por hora das observacoes WU EDDM."""
    cloud = {}
    for r in rows_cache:
        cloud[r["hour"]] = r.get("cloud_cover", 50)
    return cloud


# ══════════════════════════════════════════════════════
#  OPEN-METEO
# ══════════════════════════════════════════════════════

def make_om_session() -> requests.Session:
    """Open-Meteo nao requer API key — session simples."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "MunichPeakBot/2.0",
        "Accept":     "application/json",
    })
    return s


def fetch_om_forecast_max(session: requests.Session) -> dict | None:
    """
    Previsao de temperatura maxima para hoje via Open-Meteo.
    Gratuito, sem API key.
    """
    try:
        r = session.get(OM_FORECAST, params={
            "latitude":      MUNICH_LAT_OM,
            "longitude":     MUNICH_LON_OM,
            "daily":         "temperature_2m_max,temperature_2m_min,cloud_cover_mean",
            "timezone":      "Europe/Berlin",
            "forecast_days": 1,
        }, timeout=15)
        r.raise_for_status()
        d = r.json()
        daily = d.get("daily", {})
        t_max_list = daily.get("temperature_2m_max", [None])
        t_min_list = daily.get("temperature_2m_min", [None])
        cloud_list = daily.get("cloud_cover_mean", [None])

        t_max = t_max_list[0] if t_max_list and t_max_list[0] is not None else None
        t_min = t_min_list[0] if t_min_list and t_min_list[0] is not None else None
        cloud = cloud_list[0] if cloud_list and cloud_list[0] is not None else None

        if t_max is None:
            return None

        return {
            "temp_max":    int(round(float(t_max))),
            "temp_min":    int(round(float(t_min))) if t_min is not None else None,
            "cloud_cover": int(round(float(cloud))) if cloud is not None else None,
            "source":      "Open-Meteo",
        }
    except Exception as e:
        print(f"  {C['yellow']}OM forecast falhou: {e}{R}")
        return None


def fetch_om_hourly_today(session: requests.Session) -> list[dict]:
    """
    Previsao horaria de hoje via Open-Meteo.
    Util para comparar com observacoes WU em tempo real.
    """
    try:
        r = session.get(OM_FORECAST, params={
            "latitude":      MUNICH_LAT_OM,
            "longitude":     MUNICH_LON_OM,
            "hourly":        "temperature_2m,cloud_cover,relative_humidity_2m",
            "timezone":      "Europe/Berlin",
            "forecast_days": 1,
        }, timeout=15)
        r.raise_for_status()
        d = r.json()
        hourly = d.get("hourly", {})
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        clouds = hourly.get("cloud_cover", [])
        hums   = hourly.get("relative_humidity_2m", [])

        rows = []
        for i, t_str in enumerate(times):
            dt = datetime.fromisoformat(t_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_BERLIN)
            temp  = temps[i]  if i < len(temps)  else None
            cloud = clouds[i] if i < len(clouds) else None
            hum   = hums[i]   if i < len(hums)   else None
            if temp is None:
                continue
            rows.append({
                "hour":        dt.hour,
                "minute":      0,
                "temp_c":      int(round(float(temp))),
                "humidity":    int(round(float(hum))) if hum is not None else 70,
                "cloud_cover": int(round(float(cloud))) if cloud is not None else 50,
                "wx":          "",
                "source":      "Open-Meteo",
            })
        return rows
    except Exception:
        return []


def fetch_om_latest(session: requests.Session) -> dict | None:
    """Ultima previsao horaria do Open-Meteo para a hora actual de Munich."""
    rows = fetch_om_hourly_today(session)
    if not rows:
        return None
    h_now = datetime.now(tz=_BERLIN).hour
    return min(rows, key=lambda r: abs(r["hour"] - h_now))


def bootstrap_om_today(session: requests.Session) -> tuple[dict, list[dict]]:
    """
    Bootstrap com dados horarios do Open-Meteo.
    Devolve formato igual ao bootstrap_today() do WU.
    """
    today = berlin_date()
    print(f"  {DIM}Open-Meteo hourly forecast {today}...{R}", end=" ", flush=True)
    rows = fetch_om_hourly_today(session)
    if not rows:
        print(f"{C['red']}sem dados OM{R}")
        return {}, []
    t_vals = [r["temp_c"] for r in rows]
    print(f"{C['green']}{len(rows)} obs  "
          f"{min(t_vals)}°C – {max(t_vals)}°C{R}")

    series: dict[tuple, float] = {}
    obs_min: dict[tuple, tuple] = {}
    for r in rows:
        key = ceil_slot(r["hour"], r["minute"])
        if key not in series or r["temp_c"] >= series[key]:
            series[key]  = r["temp_c"]
            obs_min[key] = (r["hour"], r["minute"])

    seen: set[tuple] = set()
    slots: list[dict] = []
    for r in sorted(rows, key=lambda x: x["hour"] * 60 + x["minute"]):
        k = ceil_slot(r["hour"], r["minute"])
        if k not in seen:
            seen.add(k)
            slots.append({
                "hour":        k[0],
                "slot30":      k[1],
                "temp_c":      r["temp_c"],
                "cloud_cover": r.get("cloud_cover", 50),
                "humidity":    r.get("humidity", 70),
                "source":      "Open-Meteo",
            })
    return series, slots


# ══════════════════════════════════════════════════════
#  ACORDO DUAL (WU + Open-Meteo)
# ══════════════════════════════════════════════════════

def forecasts_agree(wu_forecast: dict | None,
                    om_forecast: dict | None,
                    tolerance: int = FORECAST_AGREEMENT_TOLERANCE) -> dict:
    """
    Verifica se ambas as fontes concordam na temperatura maxima prevista.

    Retorna:
      valid:         bool — ambas concordam (dentro da tolerancia)
      wu_max:        int | None
      om_max:        int | None
      diff:          int | None — diferenca absoluta em °C
      consensus_max: int | None — media das duas se concordam
      reason:        str — "agree", "disagree_Nc", "wu_missing", "om_missing", "both_missing"
    """
    if wu_forecast is None and om_forecast is None:
        return {
            "valid": False, "wu_max": None, "om_max": None,
            "diff": None, "consensus_max": None, "reason": "both_missing",
        }
    if wu_forecast is None:
        return {
            "valid": False, "wu_max": None,
            "om_max": om_forecast.get("temp_max"),
            "diff": None, "consensus_max": om_forecast.get("temp_max"),
            "reason": "wu_missing",
        }
    if om_forecast is None:
        return {
            "valid": False, "wu_max": wu_forecast.get("temp_max"),
            "om_max": None, "diff": None,
            "consensus_max": wu_forecast.get("temp_max"),
            "reason": "om_missing",
        }

    wu_max = wu_forecast["temp_max"]
    om_max = om_forecast["temp_max"]
    diff   = abs(wu_max - om_max)

    return {
        "valid":         diff <= tolerance,
        "wu_max":        wu_max,
        "om_max":        om_max,
        "diff":          diff,
        "consensus_max": round((wu_max + om_max) / 2),
        "reason":        "agree" if diff <= tolerance else f"disagree_{diff}c",
    }
