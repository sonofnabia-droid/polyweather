"""
munich_weather.py
=================
Acesso a dados meteorologicos via Weather Underground (WU) EDDM.

Exporta:
  make_wu_session()
  fetch_wu_day_eddm(day, api_key, session)
  fetch_wu_latest(api_key, session)
  fetch_wu_forecast_max(api_key, session)
  bootstrap_today(api_key, session)
  cloud_from_series(series_today, rows_cache)
"""

import requests
from datetime import date, datetime, timezone as _tz

from munich_config import (
    WU_BASE, _BERLIN, DIM, C, R,
    DAY_START, DAY_END,
    berlin_date, ceil_slot,
)

# ── URL EDDM ────────────────────────────────────────────
WU_EDDM_URL = f"{WU_BASE}/EDDM:9:DE/observations/historical.json"


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


def _wu_parse_obs(obs_list: list) -> list[dict]:
    """
    Parser para observacoes WU v1 EDDM.
    Campo de tempo: valid_time_gmt (unix UTC) -> converter para CET/CEST (Europe/Berlin).
    Temperatura: campo 'temp' (inteiro, graus Celsius em metric).
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

        rows.append({
            "hour":        dt.hour,
            "minute":      dt.minute,
            "temp_c":      int(round(float(temp))),
            "humidity":    int(round(float(obs.get("rh") or 70))),
            "cloud_cover": cloud_cover,
            "wx":          str(obs.get("wx_phrase", "") or ""),
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
        t_max = (int(round(float(t_max_list[0])))
                 if t_max_list and t_max_list[0] is not None else None)
        t_min = (int(round(float(t_min_list[0])))
                 if t_min_list and t_min_list[0] is not None else None)
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
    Ao arranque: historico completo de hoje via EDDM.
    Devolve:
      series_today : {(hour, slot30): temp_c}  — para o grafico ASCII
      slots_so_far : lista de dicts ordenada por tempo  — para o modelo
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

    # Guardar cache para acesso externo
    bootstrap_today._rows_cache = rows

    # series_today para o grafico + obs_min: timestamp real da observacao WU por slot
    series: dict[tuple, float] = {}
    obs_min: dict[tuple, tuple] = {}
    for r in rows:
        key = ceil_slot(r["hour"], r["minute"])
        if key not in series or r["temp_c"] >= series[key]:
            series[key]  = r["temp_c"]
            obs_min[key] = (r["hour"], r["minute"])

    bootstrap_today._obs_min = obs_min

    # slots_so_far para o modelo (lista cronologica sem duplicados por slot)
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
            })

    return series, slots


def cloud_from_series(series_today: dict, rows_cache: list) -> dict[int, int]:
    """
    Extrai cloud_cover por hora directamente das observacoes WU EDDM.
    Nao usa Open-Meteo — tudo vem da EDDM.
    """
    cloud = {}
    for r in rows_cache:
        cloud[r["hour"]] = r.get("cloud_cover", 50)
    return cloud
