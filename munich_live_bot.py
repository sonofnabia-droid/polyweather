"""
munich_live_bot.py
==================
Bot de trading ao vivo — Temperatura Maxima Munich — Polymarket V3.

Estratégia:
  - Ensemble: LightGBM (50%) + XGBoost (30%) + Z-Score Streaming (20%)
  - Dupla Confirmação: Modelo + Mercado (bracket com MAIOR ask)
  - Entrada faseada em 3 parcelas de $5:
      P1: 10h<=hora<12h + Forecast agreement + Mercado confirma + p>=30%
      P2: Modelo >= 60% + Mercado confirma (highest ask = running max)
      P3: Modelo >= 80%

Variaveis de ambiente:
    export WU_API_KEY="..."
    export POLY_PRIVATE_KEY="0x..."

Uso:
    python munich_live_bot.py
    python munich_live_bot.py --yes   # headless
"""

import argparse
import json
import re as _re
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import requests

from munich_config import (
    R, B, DIM, C,
    WU_API_KEY, POLY_PRIVATE_KEY, POLY_MAX_DAILY_LOSS,
    LOG_DIR, GAMMA_API, MONTH_NAMES,
    DAY_START, DAY_END, MIN_HOUR,
    _SIGNAL_CHECK_WINDOWS,
    berlin_now, berlin_date, local_now, ceil_slot, smart_sleep,
)
from munich_weather import (
    make_wu_session, make_om_session,
    fetch_wu_latest, fetch_wu_forecast_max,
    fetch_om_forecast_max,
    bootstrap_today, cloud_from_series,
    forecasts_agree,
)
from munich_model import (
    load_models, predict_ensemble, StreamingPeakDetector,
    set_seasonal_prior, compute_prev7,
    init_history_max, update_history_max,
)
from munich_phased_entry import PhasedEntry
from munich_display import display, log_tick
from polymarket_clob import (
    ClobClient, TradingMode, OrderBook,
    PositionManager, Position, PositionStatus,
)
from polymarket_orders import OrderExecutor, paper_buy
from tg import TG


# ══════════════════════════════════════════════════════
#  SALDO USDC
# ══════════════════════════════════════════════════════
def get_real_usdc_balance(private_key: str) -> float | None:
    try:
        from py_clob_client.client import ClobClient as _CC
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        _c = _CC(host="https://clob.polymarket.com", key=private_key, chain_id=137)
        _creds = _c.create_or_derive_api_creds()
        _c.set_api_creds(_creds)
        best = 0.0
        for sig in [0, 1, 2]:
            try:
                info = _c.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=sig,
                    )
                )
                bal = int(info.get("balance", "0")) / 1e6
                if bal > best:
                    best = bal
            except Exception:
                pass
        return best if best > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════
#  POLYMARKET — fetch market, bracket helpers, EV
# ══════════════════════════════════════════════════════
def date_to_slug(d: date) -> str:
    return (f"highest-temperature-in-munich-on-"
            f"{MONTH_NAMES[d.month]}-{d.day}-{d.year}")


def _extract_temp(text: str) -> float | None:
    for pat in [r'([-]?\d+)\s*°?\s*[cC]\b',
                r'([-]?\d+)\s*or\s+(?:higher|lower|above|below)',
                r'be\s+([-]?\d+)', r'^\s*([-]?\d+)\s*$']:
        m = _re.search(pat, str(text), _re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _bracket_lo(label):
    v = _extract_temp(label)
    if v is None:
        return 0.0
    s = str(label).lower()
    if any(x in s for x in ("or lower", "or below", "≤", "<=")):
        return -99.0
    return v


def _bracket_hi(label):
    v = _extract_temp(label)
    if v is None:
        return 99.0
    s = str(label).lower()
    if any(x in s for x in ("or higher", "or above", "≥", ">=")):
        return 99.0
    return v


def _normalize_label(text: str) -> str:
    if len(text) <= 25:
        return text
    v = _extract_temp(text)
    if v is None:
        return text
    s = text.lower()
    if any(x in s for x in ("higher", "above", "≥", ">=")):
        return f"{v:.0f}°C or higher"
    if any(x in s for x in ("lower", "below", "≤", "<=")):
        return f"{v:.0f}°C or lower"
    return f"{v:.0f}°C"


def fetch_market(d: date) -> dict | None:
    slug = date_to_slug(d)

    def try_api(params):
        try:
            r = requests.get(f"{GAMMA_API}/events", params=params, timeout=15)
            r.raise_for_status()
            ev = r.json()
            return ev if isinstance(ev, list) else ([ev] if ev else [])
        except Exception:
            return []

    month_s = MONTH_NAMES[d.month].capitalize()
    events = (try_api({"slug": slug}) or
              try_api({"q": f"highest temperature Munich {month_s} {d.day} {d.year}",
                       "limit": 10}) or
              try_api({"q": f"Munich temperature {d.year}", "limit": 10}))
    if not events:
        return None

    def is_munich(e):
        t = str(e.get("title", "")).lower()
        return ("munich" in t or "munchen" in t) and (
            "temp" in t or "temperature" in t or "highest" in t)

    munich = [e for e in events if isinstance(e, dict) and is_munich(e)]
    if not munich:
        munich = [e for e in events if isinstance(e, dict)]
    if not munich:
        return None

    event = max(munich, key=lambda e: float(e.get("volume", 0) or 0))
    brackets = []

    for m in event.get("markets", []):
        raw_label = (m.get("groupItemTitle") or m.get("outcomeTitle") or
                     m.get("title") or m.get("question") or "")
        label = _normalize_label(raw_label)
        v = _extract_temp(label)
        if v is None:
            continue

        outcomes  = m.get("outcomes", "[]")
        prices    = m.get("outcomePrices", "[]")
        token_ids = m.get("clobTokenIds", "[]")

        def _jload(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return []
            return x

        outcomes  = _jload(outcomes)
        prices    = _jload(prices)
        token_ids = _jload(token_ids)

        price_yes = None
        token_yes = None
        for i, out in enumerate(outcomes):
            if str(out).lower() in ("yes", "true", "1"):
                price_yes = float(prices[i]) if i < len(prices) and prices[i] else None
                token_yes = token_ids[i] if i < len(token_ids) else None
                break
        if price_yes is None and prices:
            try:
                price_yes = float(prices[0])
            except Exception:
                price_yes = 0.5
        if price_yes is None:
            continue

        brackets.append({
            "label":    label,
            "price":    round(price_yes, 4),
            "token_id": token_yes,
            "temp_lo":  _bracket_lo(label),
            "temp_hi":  _bracket_hi(label),
            "volume":   float(m.get("volume", 0) or 0),
        })

    if not brackets:
        return None
    brackets.sort(key=lambda b: b["temp_lo"])
    return {
        "title":      event.get("title", "Munich Max Temp"),
        "end_date":   event.get("endDate", ""),
        "volume":     float(event.get("volume", 0) or 0),
        "brackets":   brackets,
        "n_outcomes": len(brackets),
        "slug":       slug,
    }


def find_bracket(market: dict, temp: float) -> dict | None:
    if not market:
        return None
    tr = round(temp)
    for b in market["brackets"]:
        lo, hi = b["temp_lo"], b["temp_hi"]
        if lo == hi and tr == round(lo):
            return b
        if hi == 99 and tr >= lo:
            return b
        if lo == -99 and tr <= hi:
            return b
        if lo <= temp <= hi:
            return b
    return min(market["brackets"],
               key=lambda b: abs(tr - (b["temp_lo"] if b["temp_hi"] == 99
                                       else b["temp_hi"] if b["temp_lo"] == -99
                                       else (b["temp_lo"] + b["temp_hi"]) / 2)))


def compute_ev(p: float, ask: float) -> dict | None:
    if not ask or not (0 < ask < 1) or ask >= 0.95:
        return None
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


# ══════════════════════════════════════════════════════
#  MODO — selecção interactiva
# ══════════════════════════════════════════════════════
def ask_trading_mode() -> TradingMode:
    print(f"\n  {B}{C['cyan']}── Munich Live Bot V3 — Modo ──────────{R}")
    print(f"  {C['yellow']}[P]{R} PAPER  — simula ordens, order book real")
    print(f"  {C['red']}[R]{R} REAL   — envia ordens reais")
    while True:
        try:
            ans = input(f"  Modo? {C['yellow']}[P]{R}aper / {C['red']}[R]{R}eal : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        if ans in ("p", "paper", ""):
            print(f"\n  {C['yellow']}{B}PAPER seleccionado{R}\n")
            return TradingMode.PAPER
        if ans in ("r", "real"):
            if not POLY_PRIVATE_KEY:
                print(f"  {C['red']}POLY_PRIVATE_KEY não definida{R}")
                continue
            try:
                confirm = input(f"  Escreve {C['red']}REAL{R} para confirmar: ").strip()
            except Exception:
                confirm = ""
            if confirm == "REAL":
                print(f"\n  {C['red']}{B}REAL activado{R}\n")
                return TradingMode.REAL
            print(f"  Confirmação inválida — PAPER\n")
            return TradingMode.PAPER
        print(f"  Opção inválida.")


def confirm_real_order(bet: dict) -> bool:
    print(f"\n  {C['red']}{B}⚠ CONFIRMAR ORDEM REAL ⚠{R}")
    print(f"    Bracket : {bet['bracket']}  Ask: {bet['ask']*100:.1f}¢")
    print(f"    Aposta  : ${bet['bet_size']:.2f}  Parcela: P{bet.get('parcel_idx', 0)+1}")
    try:
        ans = input(f"  Enviar? (y/n): ").strip().lower()
    except Exception:
        return False
    return ans == "y"


# ══════════════════════════════════════════════════════
#  STDIN NÃO-BLOQUEANTE
# ══════════════════════════════════════════════════════
def _stdin_has_input(timeout=0.0):
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(r)
    except Exception:
        return False


# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
def run(wu_key: str, threshold: float, bankroll: float,
        kelly_frac: float, min_edge: float, interval: int,
        headless: bool = False):

    LOG_DIR.mkdir(exist_ok=True)

    if not wu_key:
        raise ValueError(f"\n  {C['red']}WU_API_KEY não definida{R}")

    # ── Modo ──────────────────────────────────────────
    if headless:
        trading_mode = TradingMode.REAL if POLY_PRIVATE_KEY else TradingMode.PAPER
    else:
        trading_mode = ask_trading_mode()

    tg = TG()

    # ── CLOB ──────────────────────────────────────────
    clob     = None
    executor = None
    if POLY_PRIVATE_KEY:
        try:
            clob = ClobClient(
                private_key=POLY_PRIVATE_KEY,
                mode=trading_mode,
                max_daily_loss=POLY_MAX_DAILY_LOSS,
                log_dir=LOG_DIR,
            )
            executor = OrderExecutor(POLY_PRIVATE_KEY)
        except Exception as e:
            print(f"  {C['red']}CLOB init falhou: {e}{R}")

    # ── Sessions ──────────────────────────────────────
    wu_sess = make_wu_session()
    om_sess = make_om_session()

    # ── Load Models ───────────────────────────────────
    print("[1/4] A carregar modelos...")
    models = load_models()
    set_seasonal_prior(models["prior_map"])

    def get_threshold(month, doy=0):
        if models["doy_poly"] is not None and doy > 0:
            val = float(np.polyval(models["doy_poly"], (doy - 183) / 183))
            return float(np.clip(val, 0.25, 0.95))
        return models["monthly_threshold"].get(month, threshold)

    # ── Bootstrap ─────────────────────────────────────
    today = berlin_date()
    print(f"\n[2/4] Bootstrap — {today}...")
    series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
    obs_min_today = dict(getattr(bootstrap_today, "_obs_min", {}))
    rows_cache    = getattr(bootstrap_today, "_rows_cache", [])
    cloud_by_hour = cloud_from_series(series_today, rows_cache)
    temps_by_hour = {s["hour"]: s["temp_c"] for s in slots_so_far}

    history_max = init_history_max()
    update_history_max(history_max, slots_so_far)

    # ── Z-Score + PhasedEntry ─────────────────────────
    zscore = StreamingPeakDetector()
    phased = PhasedEntry(parcel_size=5.0)

    # ── Forecasts ─────────────────────────────────────
    wu_forecast = fetch_wu_forecast_max(wu_key, wu_sess)
    om_forecast = fetch_om_forecast_max(om_sess)
    forecast_agreement = forecasts_agree(wu_forecast, om_forecast)

    # ── Aplicar modelo ao histórico ───────────────────
    print(f"\n[3/4] A aplicar modelo ao histórico ({len(slots_so_far)} slots)...")
    month  = today.month
    doy    = today.timetuple().tm_yday
    signals = {}

    for i, slot in enumerate(slots_so_far):
        h, s = slot["hour"], slot["slot30"]
        if h < MIN_HOUR or i < 3:
            continue
        current_extra = {
            "hour": h, "slot30": s,
            "cloud_cover": slot.get("cloud_cover", 50),
            "humidity":    slot.get("humidity", 70),
            "prev_7d_avg_max": compute_prev7(history_max, today),
        }
        # ASSINATURA: predict_ensemble(models, slots_so_far, current, month, doy, zscore)
        ens = predict_ensemble(models, slots_so_far[:i+1],
                              current_extra, month, doy, zscore)
        signals[(h, s)] = ens["p_ensemble"]

    # ── Market ────────────────────────────────────────
    print(f"\n[4/4] A carregar mercado...")
    market_date = today
    market = fetch_market(market_date)
    if market and clob:
        market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]

    usdc_balance = (get_real_usdc_balance(POLY_PRIVATE_KEY)
                   if (trading_mode == TradingMode.REAL and POLY_PRIVATE_KEY)
                   else None)
    open_orders = (executor.get_open_orders()
                   if (trading_mode == TradingMode.REAL and executor)
                   else None)

    # ── Log paths ─────────────────────────────────────
    log_path  = LOG_DIR / f"live_{today}.csv"
    bets_path = LOG_DIR / f"bets_{today}.json"
    bets: list = []

    # ── Telegram start ────────────────────────────────
    clob_mode_str = "real" if trading_mode == TradingMode.REAL else "paper"
    eff_thr = get_threshold(today.month)
    tg.alert_started(clob_mode_str, bankroll, threshold, eff_thr,
                     today.month, market, today)
    if not market:
        tg.alert_no_market(today)

    _tg_last_dashboard    = 0
    _tg_dashboard_interval = 30 * 60

    # ── Latest obs ────────────────────────────────────
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

    print(f"\n  {DIM}Loop iniciado — Ctrl+C para parar{R}\n")

    try:
        while True:
            now = local_now()

            # ── Novo dia ──────────────────────────────
            station_date = berlin_date()
            if station_date != today:
                today       = station_date
                market_date = today
                slots_so_far  = []
                series_today  = {}
                obs_min_today = {}
                temps_by_hour = {}
                cloud_by_hour = {}
                signals       = {}
                phased.reset()
                zscore.reset()
                bets = []
                log_path  = LOG_DIR / f"live_{today}.csv"
                bets_path = LOG_DIR / f"bets_{today}.json"
                month = today.month
                doy   = today.timetuple().tm_yday
                latest_obs = None

                market = fetch_market(market_date)
                if market and clob:
                    market["brackets"] = [
                        clob.enrich_bracket(b) for b in market["brackets"]
                    ]

                try:
                    series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
                    obs_min_today = dict(getattr(bootstrap_today, "_obs_min", {}))
                    rows_cache = getattr(bootstrap_today, "_rows_cache", [])
                    cloud_by_hour = cloud_from_series(series_today, rows_cache)
                    temps_by_hour = {s["hour"]: s["temp_c"] for s in slots_so_far}
                except Exception as e:
                    print(f"  {C['yellow']}Bootstrap falhou: {e}{R}")

                update_history_max(history_max, slots_so_far)

                wu_forecast = fetch_wu_forecast_max(wu_key, wu_sess)
                om_forecast = fetch_om_forecast_max(om_sess)
                forecast_agreement = forecasts_agree(wu_forecast, om_forecast)

                tg.alert_started(clob_mode_str, bankroll, threshold,
                                 get_threshold(today.month), today.month,
                                 market, today)

            # ── WU latest ─────────────────────────────
            new_obs = fetch_wu_latest(wu_key, wu_sess)
            if new_obs:
                latest_obs = new_obs
                h_obs, m_obs = new_obs["hour"], new_obs["minute"]
                h_slot, s30  = ceil_slot(h_obs, m_obs)

                if DAY_START <= h_slot <= DAY_END:
                    series_today[(h_slot, s30)]  = new_obs["temp_c"]
                    obs_min_today[(h_slot, s30)] = (h_obs, m_obs)

                    slot_entry = {
                        "hour": h_slot, "slot30": s30,
                        "temp_c": new_obs["temp_c"],
                        "cloud_cover": new_obs.get("cloud_cover", 50),
                        "humidity": new_obs.get("humidity", 70),
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

            update_history_max(history_max, slots_so_far)

            # ── Ensemble ──────────────────────────────
            h_now  = berlin_now().hour
            m_now  = berlin_now().minute
            h_cur, s30_cur = ceil_slot(h_now, m_now)

            p = 0.0
            ensemble_result = None
            if len(slots_so_far) >= 4 and h_cur >= MIN_HOUR:
                current_extra = {
                    "hour": h_cur, "slot30": s30_cur,
                    "cloud_cover": cloud_by_hour.get(h_cur, 50.0),
                    "humidity": (latest_obs.get("humidity", 70)
                                if latest_obs else 70),
                    "prev_7d_avg_max": compute_prev7(history_max, today),
                }
                # ASSINATURA CORRECTA: 6 args, sem feat_cols
                ensemble_result = predict_ensemble(
                    models, slots_so_far, current_extra,
                    month, doy, zscore
                )
                p = ensemble_result["p_ensemble"]
                signals[(h_cur, s30_cur)] = p

            # ── Forecasts ─────────────────────────────
            if now.minute % 30 == 0:
                wu_forecast = fetch_wu_forecast_max(wu_key, wu_sess)
                om_forecast = fetch_om_forecast_max(om_sess)
                forecast_agreement = forecasts_agree(wu_forecast, om_forecast)

            # ── Market ────────────────────────────────
            if now.minute % 10 == 0 or not market:
                market = fetch_market(market_date)
                if market and clob:
                    market["brackets"] = [
                        clob.enrich_bracket(b) for b in market["brackets"]
                    ]

            # ── Running max ───────────────────────────
            if series_today:
                rmax_slot = max(series_today, key=series_today.get)
                rmax = series_today[rmax_slot]
                rmax_time_str = (f"{obs_min_today.get(rmax_slot, (rmax_slot[0],0))[0]}:"
                                 f"{obs_min_today.get(rmax_slot, (0,0))[1]:02d}")
            elif temps_by_hour:
                rmax = max(temps_by_hour.values())
                rmax_time_str = f"{max(temps_by_hour, key=temps_by_hour.get)}h"
            else:
                rmax = 0
                rmax_time_str = "?"

            # ── Phased Entry ──────────────────────────
            actions = phased.evaluate(
                p, h_cur, market, rmax, forecast_agreement
            )

            # ── Executar parcelas ─────────────────────
            bet                = None
            bet_blocked_reason = None
            target_bracket     = None
            ev                 = None

            for action in actions:
                if action["size_usdc"] <= 0:
                    bet_blocked_reason = action["reason"]
                    continue

                pidx = action["parcel_idx"]

                # Escolher bracket
                if pidx == 1 and market:
                    target_bracket = max(
                        market["brackets"],
                        key=lambda b: b.get("ask") or b.get("price") or 0
                    )
                else:
                    target_bracket = find_bracket(market, rmax)

                if not target_bracket:
                    bet_blocked_reason = f"P{pidx+1}: sem bracket"
                    continue

                ask_price = (target_bracket.get("ask")
                             or target_bracket.get("price"))
                if not ask_price:
                    bet_blocked_reason = f"P{pidx+1}: sem ask"
                    continue

                ev = compute_ev(p, ask_price)

                bet_record = {
                    "parcel_idx": pidx,
                    "mode":       trading_mode.value,
                    "bracket":    target_bracket["label"],
                    "token_id":   target_bracket.get("token_id"),
                    "ask":        round(ask_price, 4),
                    "bid":        round(target_bracket.get("bid") or ask_price, 4),
                    "spread":     round(target_bracket.get("spread") or 0, 4),
                    "p_true":     round(p, 3),
                    "ev_cents":   ev["ev_cents"] if ev else None,
                    "edge_pct":   ev["edge_pct"] if ev else None,
                    "bet_size":   action["size_usdc"],
                    "shares":     round(action["size_usdc"] / ask_price, 2),
                    "max_profit": round(
                        action["size_usdc"] / ask_price * (1 - ask_price), 2),
                    "reason":     action["reason"],
                    "timestamp":  datetime.now().isoformat(),
                }

                # Executar
                if trading_mode == TradingMode.PAPER:
                    result = paper_buy(
                        token_id  = target_bracket.get("token_id", ""),
                        price     = ask_price,
                        size_usdc = action["size_usdc"],
                        label     = f"P{pidx+1}: {target_bracket['label']}",
                    )
                    bet_record["order_id"] = result["order_id"]
                    bet_record["status"]   = result["status"]
                    bet = bet_record
                    phased.mark_bought(pidx, bet_record)
                    bets.append(bet)
                    bets_path.write_text(
                        json.dumps(bets, indent=2, default=str))

                else:  # REAL
                    if headless or confirm_real_order(bet_record):
                        if executor:
                            result = executor.buy(
                                token_id  = target_bracket.get("token_id", ""),
                                price     = ask_price,
                                size_usdc = action["size_usdc"],
                                label     = f"P{pidx+1}: {target_bracket['label']}",
                            )
                            if result["success"]:
                                bet_record["order_id"] = result["order_id"]
                                bet_record["status"]   = result["status"]
                                bet = bet_record
                                phased.mark_bought(pidx, bet_record)
                                bets.append(bet)
                                bets_path.write_text(
                                    json.dumps(bets, indent=2, default=str))
                                usdc_balance = get_real_usdc_balance(
                                    POLY_PRIVATE_KEY)
                            else:
                                bet_blocked_reason = (
                                    f"P{pidx+1}: ordem falhou")
                        else:
                            bet_blocked_reason = "executor não disponível"
                    else:
                        bet_blocked_reason = "confirmação recusada"

                # Telegram
                if bet:
                    if pidx == 0:
                        tg.alert_peak_detected(
                            p_ensemble=p, rmax=rmax,
                            rmax_time=rmax_time_str,
                            bracket=target_bracket,
                            ensemble_result=ensemble_result,
                            market=market)
                    else:
                        tg.alert_order_placed(bet, clob_mode_str)

                break  # 1 parcela por tick

            # ── Display ───────────────────────────────
            signals_by_hour = {}
            for (sh, ss), sp in signals.items():
                if sh not in signals_by_hour or sp > signals_by_hour[sh]:
                    signals_by_hour[sh] = sp

            temp_now = latest_obs["temp_c"] if latest_obs else 0
            eff_thr  = get_threshold(month)

            display(
                now, latest_obs, temps_by_hour, series_today,
                signals_by_hour, p,
                market, target_bracket, ev, bet,
                len(series_today), bankroll, eff_thr,
                phased.n_parcels_bought > 0,
                trading_mode       = trading_mode,
                daily_loss         = clob.daily_loss() if clob else 0.0,
                max_daily_loss     = POLY_MAX_DAILY_LOSS,
                usdc_balance       = usdc_balance,
                positions          = clob.positions if clob else None,
                bet_blocked_reason = bet_blocked_reason,
                bet_placed         = phased.n_parcels_bought > 0,
                forecast_max       = wu_forecast,
                berlin_now_dt      = berlin_now(),
                market_date        = market_date,
                executor           = executor,
                open_orders        = open_orders,
                obs_min_today      = obs_min_today,
                phased             = phased,
                forecast_agreement = forecast_agreement,
                om_forecast        = om_forecast,
                ensemble_result    = ensemble_result,
            )

            log_tick(
                now, temp_now, p, phased.n_parcels_bought > 0,
                target_bracket, ev, bet, log_path,
                trading_mode       = trading_mode,
                bet_blocked_reason = (bet_blocked_reason
                                     if not bet else None),
            )

            # ── Telegram dashboard periódica ───────────
            if time.time() - _tg_last_dashboard >= _tg_dashboard_interval:
                _tg_last_dashboard = time.time()
                chart = draw_chart_plain(
                    series_today, signals_by_hour,
                    phased.n_parcels_bought > 0)
                tg.dashboard(
                    today=today, p=p, rmax=rmax,
                    rmax_time=rmax_time_str,
                    temp_now=temp_now, forecast_max=wu_forecast,
                    market=market, bracket=target_bracket,
                    ev=ev,
                    peak_detected=phased.n_parcels_bought > 0,
                    bet=bets[-1] if bets else None,
                    clob_mode=clob_mode_str,
                    chart=chart, om_forecast=om_forecast,
                    forecast_agreement=forecast_agreement,
                    ensemble_result=ensemble_result,
                    positions_summary=(clob.positions.pnl_summary()
                                       if clob else None),
                )

            # ── Sleep ─────────────────────────────────
            _berlin_h = berlin_now().hour
            _last_t   = latest_obs["temp_c"] if latest_obs else None
            if _berlin_h < 8 or _berlin_h >= 21:
                time.sleep(300)
            else:
                smart_sleep(interval, wu_key, wu_sess, _last_t)

    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Stopped. Logs em ./{LOG_DIR}/{R}")
        tg.alert_stopped(bets, clob_mode_str)


def draw_chart_plain(series_today, signals, peak_detected):
    lines = []
    slots = [(h, m)
             for h in range(DAY_START, DAY_END + 1)
             for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    if not avail:
        return ["sem dados"]
    for si, ((h, m), temp) in enumerate(zip(slots, temps)):
        if temp is None:
            continue
        p = signals.get(h, 0)
        sym = ("▓▓" if p >= 0.80 else "▒▒" if p >= 0.60
               else "░░" if p >= 0.30 else "··")
        lines.append(f"{h:02d}:{m:02d} {int(temp):>2}°C {sym} {p*100:.0f}%")
    return lines


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Munich Max Temp — Live Bot V3")
    parser.add_argument("--threshold", type=float, default=0.46)
    parser.add_argument("--bankroll",  type=float, default=200.0)
    parser.add_argument("--kelly",     type=float, default=0.5)
    parser.add_argument("--min-edge",  type=float, default=5.0)
    parser.add_argument("--interval",  type=int,   default=60)
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Headless")
    args = parser.parse_args()

    run(
        wu_key     = WU_API_KEY,
        threshold  = args.threshold,
        bankroll   = args.bankroll,
        kelly_frac = args.kelly,
        min_edge   = args.min_edge,
        interval   = args.interval,
        headless   = args.yes,
    )


if __name__ == "__main__":
    main()
