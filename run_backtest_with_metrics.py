cat << 'EOF' > munich_live_bot.py
"""
munich_live_bot.py - UNIFIED2 + STOP-LOSS
"""
import argparse
import json
import re as _re
import sys
import time
from datetime import date, datetime, timedelta
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
from munich_display import display, log_tick
from polymarket_clob import (
    ClobClient, TradingMode,
    PositionManager, Position, PositionStatus,
)
from polymarket_orders import OrderExecutor, paper_buy
from tg import TG
from bet_metrics import BetMetrics

FIXED_BET_SIZE = 5.0
STOP_LOSS_TEMP_THRESHOLD = 1.0
MIN_BID_TO_SELL = 0.05

active_position = {
    "is_active": False,
    "token_id": None,
    "shares": 0.0,
    "entry_temp": 0.0,
    "bracket_label": "",
    "current_tick_temp": 0.0,
}

def get_real_usdc_balance(private_key: str) -> float | None:
    try:
        from py_clob_client.client import ClobClient as _CC
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        _c = _CC(host="https://clob.polymarket.com", key=private_key, chain_id=137)
        _creds = _c.create_or_derive_api_creds()
        _c.set_api_creds(_creds)
        best = 0.0
        for sig in [2, 0, 1]:
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

def date_to_slug(d: date) -> str:
    return (f"highest-temperature-in-munich-on-"
            f"{MONTH_NAMES[d.month]}-{d.day}-{d.year}")

def _extract_temp(text: str) -> float | None:
    for pat in [r'([-]?\d+)\s*\*°?\s*[cC]\b',
                r'([-]?\d+)\s*or\s+(?:higher|lower|above|below)',
                r'be\s+([-]?\d+)', r'^\s*([-]?\d+)\s*$']:
        m = _re.search(pat, str(text), _re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def _bracket_lo(label):
    v = _extract_temp(label)
    if v is None: return 0.0
    s = str(label).lower()
    if any(x in s for x in ("or lower", "or below", "<=")): return -99.0
    return v

def _bracket_hi(label):
    v = _extract_temp(label)
    if v is None: return 99.0
    s = str(label).lower()
    if any(x in s for x in ("or higher", "or above", ">=")): return 99.0
    return v

def _normalize_label(text: str) -> str:
    if len(text) <= 25: return text
    v = _extract_temp(text)
    if v is None: return text
    s = text.lower()
    if any(x in s for x in ("higher", "above", ">=")): return f"{v:.0f}C or higher"
    if any(x in s for x in ("lower", "below", "<=")): return  f"{v:.0f}C or lower"
    return f"{v:.0f}C"

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
              try_api({"q": f"highest temperature Munich {month_s} {d.day} {d.year}", "limit": 10}) or
              try_api({"q": f"Munich temperature {d.year}", "limit": 10}))
    if not events: return None

    def is_munich(e):
        t = str(e.get("title", "")).lower()
        return ("munich" in t or "munchen" in t) and ("temp" in t or "temperature" in t or "highest" in t)

    munich = [e for e in events if isinstance(e, dict) and is_munich(e)]
    if not munich: munich = [e for e in events if isinstance(e, dict)]
    if not munich: return None

    event = max(munich, key=lambda e: float(e.get("volume", 0) or 0))
    brackets = []
    for m in event.get("markets", []):
        raw_label = (m.get("groupItemTitle") or m.get("outcomeTitle") or m.get("title") or m.get("question") or "")
        label = _normalize_label(raw_label)
        v = _extract_temp(label)
        if v is None: continue

        def _jload(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except Exception: return []
            return x
            
        outcomes, prices, token_ids = _jload(m.get("outcomes", "[]")), _jload(m.get("outcomePrices","[]")), _jload(m.get("clobTokenIds","[]"))
        price_yes, token_yes = None, None
        for i, out in enumerate(outcomes):
            if str(out).lower() in ("yes", "true", "1"):
                price_yes = float(prices[i]) if i < len(prices) and prices[i] else None
                token_yes = token_ids[i] if i < len(token_ids) else None
                break
        if price_yes is None and prices:
            try: price_yes = float(prices[0])
            except Exception: price_yes = 0.5
        if price_yes is None: continue

        brackets.append({"label": label, "price": round(price_yes, 4), "token_id": token_yes, "temp_lo": _bracket_lo(label), "temp_hi": _bracket_hi(label), "volume": float(m.get("volume", 0) or 0)})

    if not brackets: return None
    brackets.sort(key=lambda b: b["temp_lo"])
    return {"title": event.get("title", "Munich Max Temp"), "end_date": event.get("endDate", ""), "volume": float(event.get("volume", 0) or 0), "brackets": brackets, "n_outcomes": len(brackets), "slug": slug}

def find_bracket(market: dict, temp: float) -> dict | None:
    if not market: return None
    tr = round(temp)
    for b in market["brackets"]:
        lo, hi = b["temp_lo"], b["temp_hi"]
        if lo == hi and tr == round(lo): return b
        if hi == 99  and tr >= lo:       return b
        if lo == -99 and tr <= hi:       return b
        if lo <= temp <= hi:             return b
    return min(market["brackets"], key=lambda b: abs(tr - (b["temp_lo"] if b["temp_hi"] == 99 else b["temp_hi"] if b["temp_lo"] == -99 else (b["temp_lo"] + b["temp_hi"]) / 2)))

def compute_ev(p: float, ask: float) -> dict | None:
    if not ask or not (0 < ask < 1) or ask >= 0.95: return None
    ev, b = p - ask, (1 - ask) / ask
    kelly = max(0.0, (p * b - (1 - p)) / b)
    return {"ev": round(ev, 4), "ev_cents": round(ev * 100, 2), "kelly": round(kelly, 4), "edge_pct": round((p / ask - 1) * 100, 2), "ev_positive": ev > 0, "ask": round(ask, 4)}

def ask_trading_mode() -> TradingMode:
    print(f"\n  {B}{C['cyan']}-- Munich Live Bot -- Seleccao de Modo ----------{R}")
    print(f"  {C['yellow']}[P]{R} PAPER  - simula ordens")
    print(f"  {C['red']}[R]{R} REAL   - ordens reais\n")
    while True:
        try: ans = input(f"  Modo? {C['yellow']}[P]{R}aper / {C['red']}[R]{R}eal : ").strip().lower()
        except (EOFError, KeyboardInterrupt): raise SystemExit(0)
        if ans in ("p", "paper", ""): 
            print(f"\n  {C['yellow']}{B}Modo PAPER.{R}")
            return TradingMode.PAPER
        if ans in ("r", "real"):
            if not POLY_PRIVATE_KEY:
                try:
                    if input(f"  Sem chave. Paper? (s/n): ").strip().lower() != "s": raise SystemExit(0)
                    return TradingMode.PAPER
                except (EOFError, KeyboardInterrupt): raise SystemExit(0)
            print(f"\n  {C['red']}{B}MODO REAL.{R}")
            usdc_bal = get_real_usdc_balance(POLY_PRIVATE_KEY)
            if usdc_bal is not None and usdc_bal < 1.0:
                try:
                    if input(f"  Saldo insuficiente. Paper? (s/n): ").strip().lower() != "s": raise SystemExit(0)
                    return TradingMode.PAPER
                except (EOFError, KeyboardInterrupt): raise SystemExit(0)
            try:
                if input(f"  Confirma? (escreva {C['red']}REAL{R}): ").strip() == "REAL":
                    print(f"\n  {C['red']}{B}Modo REAL activado.{R}\n")
                    return TradingMode.REAL
            except (EOFError, KeyboardInterrupt): raise SystemExit(0)
            return TradingMode.PAPER

def confirm_real_order(bet: dict) -> bool:
    print(f"\n  {C['red']}{B}{'='*46}{R}\n  {C['red']}{B}  CONFIRMAR ORDEM REAL  {R}\n  {C['red']}{B}{'='*46}{R}")
    print(f"    Bracket : {bet['bracket']}\n    Ask     : {bet['ask']*100:.1f}c\n    EV      : {bet.get('ev_cents', 0):+.1f}c")
    try: return input(f"  Enviar? (y/n): ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt): return False

def _stdin_has_input(timeout: float = 0.0) -> bool:
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(r)
    except (ImportError, AttributeError):
        try: import msvcrt; return msvcrt.kbhit()
        except ImportError: return False

def _read_stdin_line() -> str:
    try: return sys.stdin.readline().strip().lower()
    except Exception: return ""

def execute_entry(bracket, ask_price, p, ev, current_temp, trading_mode, executor, headless=False) -> tuple:
    if not bracket or not ask_price:
        return None, "sem bracket ou preco"
    if ev and not ev["ev_positive"]:
        if not headless:
            try:
                if input(f"\n  {C['yellow']}EV negativo. Entrar? (s/n): {R}").strip() != "s":
                    return None, "cancelado"
            except (EOFError, KeyboardInterrupt): return None, "cancelado"
    shares = round(FIXED_BET_SIZE / ask_price, 4)
    if trading_mode == TradingMode.PAPER:
        result = paper_buy(bracket["token_id"], ask_price, FIXED_BET_SIZE, label=bracket["label"])
    else:
        if not headless and not confirm_real_order({"bracket": bracket["label"], "ask": ask_price, "bet_size": FIXED_BET_SIZE, "shares": shares, "max_profit": FIXED_BET_SIZE*((1/ask_price)-1), "ev_cents": ev.get("ev_cents") if ev else 0, "edge_pct": ev.get("edge_pct") if ev else 0}):
            return None, "nao confirmado"
        result = executor.buy(token_id=bracket["token_id"], price=ask_price, size_usdc=FIXED_BET_SIZE, label=bracket["label"])
    if result and result.get("success"):
        global active_position
        active_position["is_active"] = True
        active_position["token_id"] = bracket["token_id"]
        active_position["shares"] = result.get("shares", shares)
        active_position["entry_temp"] = current_temp
        active_position["bracket_label"] = bracket["label"]
        return result, "ok"
    return result, result.get("error", "falha") if result else ("falha desconhecida", "falha desconhecida")

def check_stop_loss(executor) -> bool:
    global active_position
    if not active_position["is_active"]: return False
    current_temp = active_position.get("current_tick_temp", 0.0)
    if current_temp == 0.0: return False
    if current_temp >= (active_position["entry_temp"] + STOP_LOSS_TEMP_THRESHOLD):
        print(f"\n  {C['red']}STOP-LOSS ACTIVADO!{R}")
        print(f"  {C['red']}Entrada {active_position['entry_temp']}C. Atual: {current_temp}C (+{current_temp - active_position['entry_temp']:.1f}C){R}")
        best_prices = executor.get_best_prices(active_position["token_id"])
        current_bid = best_prices.get("bid")
        if current_bid and current_bid >= MIN_BID_TO_SELL:
            print(f"  {C['yellow']}A vender ao bid: {current_bid*100:.1f}c{R}")
            sell_result = executor.sell(
                token_id=active_position["token_id"],
                price=current_bid,
                shares=active_position["shares"],
                label=f"STOP_LOSS_{active_position['bracket_label']}"
            )
            if sell_result["success"]:
                print(f"  {C['green']}VENDIDO.{R}")
            else:
                print(f"  {C['red']}Falha ao vender.{R}")
        else:
            print(f"  {C['yellow']}Bid baixo. Impossivel vender.{R}")
        active_position["is_active"] = False
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Munich Live Bot")
    parser.add_argument("--run", choices=["paper", "real"])
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    if args.run == "paper":
        trading_mode = TradingMode.PAPER
        print(f"  {C['yellow']}{B}Modo PAPER.{R}\n")
    elif args.run == "real":
        trading_mode = TradingMode.REAL
        print(f"  {C['red']}{B}Modo REAL.{R}\n")
    else:
        trading_mode = ask_trading_mode()

    try:
        executor = OrderExecutor(POLY_PRIVATE_KEY) if trading_mode == TradingMode.REAL else None
    except Exception as e:
        print(f"  {C['red']}Erro executor: {e}. Mudando para Paper.{R}")
        trading_mode = TradingMode.PAPER
        executor = None

    models = load_models()
    set_seasonal_prior(models["prior_map"])
    wu_sess = make_wu_session()
    om_sess = make_om_session()
    history_max = init_history_max()
    zscore = StreamingPeakDetector()

    print(f"  {C['green']}Bot iniciado.{R}")

    while True:
        now_berlin = berlin_now()
        h = now_berlin.hour
        if h < DAY_START or h >= DAY_END:
            active_position["is_active"] = False
            smart_sleep(60)
            continue
        try:
            wu_data = fetch_wu_latest(wu_sess)
            if not wu_data:
                smart_sleep(30)
                continue
            current_temp = wu_data["temp_c"]
            active_position["current_tick_temp"] = current_temp
            slots_today = bootstrap_today(wu_data, om_sess) if om_sess else []
        except Exception:
            smart_sleep(30)
            continue

        if len(slots_today) < 4:
            smart_sleep(30)
            continue

        update_history_max(history_max, slots_today)
        current_extra = {
            "hour": h, "slot30": 30, "temp_c": current_temp,
            "cloud_cover": wu_data.get("cloud_cover", 50), "humidity": wu_data.get("humidity", 70),
            "prev_7d_avg_max": compute_prev7(history_max, berlin_date()),
        }
        
        ens = predict_ensemble(models, slots_today, current_extra, now_berlin.month, now_berlin.timetuple().tm_yday, zscore)
        p = ens["p_ensemble"]
        display(p, current_temp, ens)

        if _stdin_has_input():
            cmd = _read_stdin_line()
            if cmd == 'f' and not active_position["is_active"]:
                print(f"\n  {C['yellow']}FORCAR ENTRADA{R}")
                market = fetch_market(berlin_date())
                if market:
                    rmax = max(s["temp_c"] for s in slots_today)
                    bracket = find_bracket(market, rmax)
                    if bracket:
                        ask = bracket.get("price", 0)
                        if ask and 0 < ask < 0.95:
                            ev = compute_ev(p, ask)
                            res, reason = execute_entry(bracket, ask, p, ev, current_temp, trading_mode, executor, args.yes)
                            print(f"  Resultado: {reason}")

        if trading_mode == TradingMode.REAL and executor:
            if check_stop_loss(executor):
                smart_sleep(300)
                continue

        if not active_position["is_active"] and p >= 0.80:
            market = fetch_market(berlin_date())
            if market:
                rmax = max(s["temp_c"] for s in slots_today)
                bracket = find_bracket(market, rmax)
                if bracket:
                    ask = bracket.get("price", 0)
                    if ask and 0 < ask < 0.95:
                        ev = compute_ev(p, ask)
                        if ev and ev["ev_positive"]:
                            print(f"\n  {C['green']}SINAL AUTO: p={p*100:.0f}%{R}")
                            res, reason = execute_entry(bracket, ask, p, ev, current_temp, trading_mode, executor, args.yes)
                            print(f"  Resultado: {reason}")

        smart_sleep(_SIGNAL_CHECK_WINDOWS.get(h, 120))

if __name__ == "__main__":
    main()
EOF
