"""
munich_live_bot.py
==================
Bot de trading ao vivo — Temperatura Maxima Munich — Polymarket.

Modos:
  PAPER — simula ordens; mostra order book real (bid/ask/spread do CLOB)
  REAL  — envia ordens reais ao Polymarket CLOB via py-clob-client
          requer confirmacao manual (y/n) + stop-loss diario

Estrategia de Entrada (3 Fases com Dupla Condicao):
  1. INICIAL: As 10:00 Berlin, se o Mercado confirmar (ask entre 10¢ e 90¢)
  2. AMARELO: Quando P(pico) >= 60%, se o Mercado confirmar
  3. VERDE:   Quando P(pico) >= 80%, se o Mercado confirmar
  
  Cada fase aposta $5.50. Total maximo por dia: $16.50.
  A "Confirmacao do Mercado" evita comprar a 99¢ (sem edge) ou a 1¢ (mercado descarta).

Instalacao:
    pip install requests pandas numpy scikit-learn lightgbm joblib py-clob-client

Variaveis de ambiente obrigatorias:
    export WU_API_KEY="a_tua_chave_wunderground"
    export POLY_PRIVATE_KEY="0x..."

Variaveis opcionais:
    export POLY_MAX_DAILY_LOSS="50"    # stop-loss diario em USDC (default: 50)

Uso:
    python munich_live_bot.py
    python munich_live_bot.py --threshold 0.80
    python munich_live_bot.py --bankroll 200 --min-edge 5
"""

import argparse
import json
import re as _re
import sys
import time
from datetime import date, datetime, timedelta
from enum import IntFlag

import requests

# ── Modulos internos ──────────────────────────────────
from munich_config import (
    R, B, DIM, C,
    WU_API_KEY, POLY_PRIVATE_KEY, POLY_MAX_DAILY_LOSS,
    LOG_DIR, GAMMA_API, MONTH_NAMES,
    DAY_START, DAY_END, MIN_HOUR,
    _SIGNAL_CHECK_WINDOWS,
    berlin_now, berlin_date, local_now, ceil_slot,
    smart_sleep,
)
from munich_weather import (
    make_wu_session, fetch_wu_latest, fetch_wu_forecast_max,
    bootstrap_today, cloud_from_series,
)
from munich_model import (
    load_model, predict_p, set_seasonal_prior,
    compute_prev7, init_history_max, update_history_max,
)
from munich_display import display, log_tick

# ── Polymarket / execucao ─────────────────────────────
from polymarket_clob import ClobClient, TradingMode, OrderBook, PositionManager, Position, PositionStatus
from polymarket_orders import OrderExecutor, paper_buy
from tg import TG


# ══════════════════════════════════════════════════════
#  SISTEMA DE 3 FASES DE APOSTA
# ══════════════════════════════════════════════════════

class BetPhase(IntFlag):
    NONE     = 0      # nenhuma aposta
    INITIAL  = 1      # 10:00 / início do dia → $5.50
    YELLOW   = 2      # P >= 60% (amarelo) → $5.50
    GREEN    = 4      # P >= 80% (verde) → $5.50
    DONE     = 7      # todas colocadas

BET_SIZE_PER_PHASE = 5.50


# ══════════════════════════════════════════════════════
#  SALDO USDC — lê sig_type 0,1,2 e devolve o maior
# ══════════════════════════════════════════════════════
def get_real_usdc_balance(private_key: str) -> float | None:
    """
    Lê o saldo USDC real do CLOB tentando os 3 sig_types.
    O saldo util para ordens esta tipicamente em sig_type=2.
    Devolve o maior saldo encontrado entre os 3 tipos.
    """
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
#  DUPLA CONDIÇÃO: VALIDAÇÃO DE MERCADO
# ══════════════════════════════════════════════════════
def market_confirms_bracket(bracket: dict | None, min_ask: float = 0.10, max_ask: float = 0.90) -> tuple[bool, str]:
    """
    Verifica se o mercado oferece liquidez e preço razoável (Edge).
    
    Condições de FAIL (mercado NÃO confirma):
      - ask < min_ask: mercado diz que é quase impossível (ex: ask a 1¢)
      - ask > max_ask: mercado já sabe que é este (ex: ask a 99¢, sem edge)
      - sem bid ou sem ask: order book vazio, não há liquidez
    """
    if not bracket:
        return False, "sem bracket"
    
    ask = bracket.get("ask") or bracket.get("price")
    bid = bracket.get("bid")
    
    if ask is None:
        return False, "ask indisponivel"
    
    if bid is None or bid <= 0:
        return False, "sem liquidez (sem bid)"
    
    if ask < min_ask:
        return False, f"ask {ask*100:.1f}¢ muito baixo (mercado descarta)"
    
    if ask > max_ask:
        return False, f"ask {ask*100:.1f}¢ muito alto (sem edge)"
    
    return True, f"ask {ask*100:.1f}¢ OK"


def find_best_value_bracket(market: dict, temp: float, max_ask: float = 0.90) -> dict | None:
    """
    Em vez de apanhar só o bracket exato da temperatura atual,
    procura o bracket com MELHOR EDGE onde o mercado ainda está barato
    e é próximo da temperatura real.
    """
    if not market:
        return None
    
    best_bracket = None
    best_score = -999
    
    for b in market["brackets"]:
        ask = b.get("ask") or b.get("price") or 1.0
        
        # Ignorar se o mercado já resolveu ou não há liquidez
        if ask >= max_ask or ask <= 0.01:
            continue
            
        bid = b.get("bid")
        if not bid or bid <= 0:
            continue
            
        lo, hi = b["temp_lo"], b["temp_hi"]
        
        # Pontuação de proximidade
        if lo <= temp <= hi:
            proximity = 1.0  # Match exato
        elif abs(temp - lo) <= 1.5 or abs(temp - hi) <= 1.5:
            proximity = 0.5  # Adjacente próximo
        else:
            proximity = 0.0  # Longe demais
            
        # Score = Proximidade × (1 - Ask). Ask baixo + perto = melhor score
        score = proximity * (1.0 - ask)
        
        if score > best_score:
            best_score = score
            best_bracket = b
            
    return best_bracket


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


def _bracket_lo(label) -> float:
    s = str(label).lower()
    v = _extract_temp(label)
    if v is None: return 0.0
    if any(x in s for x in ("or lower", "or below", "≤", "<=")): return -99.0
    return v


def _bracket_hi(label) -> float:
    s = str(label).lower()
    v = _extract_temp(label)
    if v is None: return 99.0
    if any(x in s for x in ("or higher", "or above", "≥", ">=")): return 99.0
    return v


def _normalize_label(text: str) -> str:
    if len(text) <= 25: return text
    v = _extract_temp(text)
    if v is None: return text
    s = text.lower()
    if any(x in s for x in ("higher", "above", "≥", ">=")): return f"{v:.0f}°C or higher"
    if any(x in s for x in ("lower", "below", "≤", "<=")): return  f"{v:.0f}°C or lower"
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
    if not munich: munich = [e for e in events if isinstance(e, dict)]
    if not munich: return None

    event    = max(munich, key=lambda e: float(e.get("volume", 0) or 0))
    brackets = []
    for m in event.get("markets", []):
        raw_label = (m.get("groupItemTitle") or m.get("outcomeTitle") or
                     m.get("title") or m.get("question") or "")
        label = _normalize_label(raw_label)
        v = _extract_temp(label)
        if v is None: continue

        outcomes  = m.get("outcomes",    "[]")
        prices    = m.get("outcomePrices","[]")
        token_ids = m.get("clobTokenIds","[]")
        
        def _jload(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except Exception: return []
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
            try: price_yes = float(prices[0])
            except Exception: price_yes = 0.5

        if price_yes is None: continue
        brackets.append({
            "label":    label,
            "price":    round(price_yes, 4),
            "token_id": token_yes,
            "temp_lo":  _bracket_lo(label),
            "temp_hi":  _bracket_hi(label),
            "volume":   float(m.get("volume", 0) or 0),
        })

    if not brackets: return None
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
    if not market: return None
    tr = round(temp)
    for b in market["brackets"]:
        lo, hi = b["temp_lo"], b["temp_hi"]
        if lo == hi and tr == round(lo): return b
        if hi == 99  and tr >= lo:       return b
        if lo == -99 and tr <= hi:       return b
        if lo <= temp <= hi:             return b
    return min(market["brackets"],
               key=lambda b: abs(tr - (b["temp_lo"] if b["temp_hi"] == 99
                                       else b["temp_hi"] if b["temp_lo"] == -99
                                       else (b["temp_lo"] + b["temp_hi"]) / 2)))


def compute_ev(p: float, ask: float) -> dict | None:
    """EV calculado sobre o ask do CLOB."""
    if not ask or not (0 < ask < 1): return None
    if ask >= 0.95: return None
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


def build_bet_record(bracket, p, ev, bankroll, kelly_frac, mode: TradingMode,
                     max_daily_loss: float = 10.0, phase_name: str = "UNKNOWN") -> dict:
    """
    Sizing: Fixo em BET_SIZE_PER_PHASE ($5.50).
    A dupla condicao (modelo + mercado) garante que so entramos quando ha edge.
    """
    ask      = ask_price if (ask_price := (bracket.get("ask") or bracket.get("price"))) else (ev["ask"] if ev else 0)
    
    # Tamanho fixo por fase
    bet_size = BET_SIZE_PER_PHASE
    
    shares   = round(bet_size / ask, 4) if ask > 0 else 0
    ev_cents  = ev["ev_cents"] if ev else None
    edge_pct  = ev["edge_pct"] if ev else None
    return {
        "mode":         mode.value,
        "phase":        phase_name,
        "bracket":      bracket["label"],
        "token_id":     bracket.get("token_id"),
        "ask":          round(ask, 4),
        "bid":          round(bracket.get("bid") or ask, 4),
        "spread":       round(bracket.get("spread") or 0, 4),
        "p_true":       round(p, 3),
        "ev_cents":     ev_cents,
        "edge_pct":     edge_pct,
        "sizing":       "fixed_phase",
        "max_daily_loss": max_daily_loss,
        "bet_size":     bet_size,
        "shares":       shares,
        "max_profit":   round(shares * (1 - ask), 2),
        "timestamp":    datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════
#  INPUT NAO-BLOQUEANTE
# ══════════════════════════════════════════════════════
def _stdin_has_input(timeout: float = 0.0) -> bool:
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(r)
    except (ImportError, AttributeError):
        try:
            import msvcrt
            return msvcrt.kbhit()
        except ImportError:
            return False


def _read_stdin_line() -> str:
    try:
        return sys.stdin.readline().strip().lower()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════
#  ENTRADA FORCADA (override 'f')
# ══════════════════════════════════════════════════════
def execute_forced_entry(bracket, ask_price, p, ev,
                         bankroll, kelly_frac,
                         trading_mode, executor, market,
                         bets, bets_path) -> tuple:
    """Executa entrada forcada — ignora fases, modelo e mercado."""
    if not bracket or not ask_price or not ev:
        return None, "sem bracket ou preco disponivel"

    if not ev["ev_positive"]:
        print(f"\n  {C['yellow']}⚠  EV negativo ({ev['ev_cents']:+.1f}¢) — entrar mesmo assim? (s/n): {R}",
              end="", flush=True)
        try:
            ans = input("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, "cancelado"
        if ans != "s":
            return None, "cancelado pelo utilizador"

    bet_record = build_bet_record(bracket, p, ev, bankroll, kelly_frac, trading_mode, 
                                  max_daily_loss=POLY_MAX_DAILY_LOSS, phase_name="FORCED")

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
        if confirm_real_order(bet_record):
            if not executor:
                return None, "OrderExecutor nao disponivel"
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
            return None, "confirmacao recusada"


# ══════════════════════════════════════════════════════
#  MODO — seleccao interactiva
# ══════════════════════════════════════════════════════
def ask_trading_mode() -> TradingMode:
    print(f"\n  {B}{C['cyan']}── Munich Live Bot — Seleccao de Modo ──────────{R}")
    print(f"  {C['yellow']}[P]{R} PAPER  — simula ordens, order book real do CLOB")
    print(f"  {C['red']}[R]{R} REAL   — envia ordens reais ao Polymarket CLOB")
    print(f"  {DIM}Estrategia: 3 Fases ($5.50 cada) com Dupla Condicao{R}")
    print()

    while True:
        try:
            ans = input(f"  Modo? {C['yellow']}[P]{R}aper / {C['red']}[R]{R}eal : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  A sair.")
            raise SystemExit(0)

        if ans in ("p", "paper", ""):
            print(f"\n  {C['yellow']}{B}Modo PAPER seleccionado.{R}  "
                  f"{DIM}Ordens simuladas — nenhum dinheiro real sera gasto.{R}\n")
            return TradingMode.PAPER

        if ans in ("r", "real"):
            if not POLY_PRIVATE_KEY:
                print(f"\n  {C['red']}{B}✗  POLY_PRIVATE_KEY nao definida.{R}\n")
                try:
                    alt = input(f"  Continuar em modo PAPER? ({C['yellow']}s{R}/{C['red']}n{R}): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit(0)
                if alt == "s":
                    print(f"\n  {C['yellow']}{B}Modo PAPER seleccionado.{R}\n")
                    return TradingMode.PAPER
                else:
                    raise SystemExit(0)

            print(f"\n  {C['red']}{B}⚠  MODO REAL — ordens reais serao enviadas ao Polymarket.{R}")
            print(f"  Stop-loss diario: ${POLY_MAX_DAILY_LOSS:.0f} USDC")

            print(f"  {DIM}A verificar saldo USDC...{R}", end=" ", flush=True)
            usdc_balance_check = None
            try:
                _tmp = ClobClient(
                    private_key    = POLY_PRIVATE_KEY,
                    mode           = TradingMode.REAL,
                    max_daily_loss = POLY_MAX_DAILY_LOSS,
                    log_dir        = LOG_DIR,
                )
                usdc_balance_check = get_real_usdc_balance(POLY_PRIVATE_KEY)
            except Exception as e:
                print(f"{C['yellow']}indisponivel ({e}){R}")

            if usdc_balance_check is not None:
                bal_col = C["green"] if usdc_balance_check >= 10 else C["red"]
                print(f"{bal_col}{B}${usdc_balance_check:,.2f} USDC{R}")
                if usdc_balance_check < 1.0:
                    print(f"\n  {C['red']}{B}✗  Saldo insuficiente.{R}\n")
                    try:
                        alt = input(f"  Continuar em modo PAPER? ({C['yellow']}s{R}/{C['red']}n{R}): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        raise SystemExit(0)
                    if alt == "s":
                        return TradingMode.PAPER
                    else:
                        raise SystemExit(0)
                elif usdc_balance_check < 10.0:
                    print(f"  {C['yellow']}⚠  Saldo baixo.{R}")

            try:
                confirm = input(f"  Confirmas? (escreve {C['red']}REAL{R} para confirmar): ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)

            if confirm == "REAL":
                print(f"\n  {C['red']}{B}Modo REAL activado.{R}\n")
                return TradingMode.REAL
            else:
                print(f"  {DIM}Confirmacao invalida — a usar PAPER.{R}\n")
                return TradingMode.PAPER

        print(f"  {C['yellow']}Opcao invalida. Escreve P ou R.{R}")


def confirm_real_order(bet: dict) -> bool:
    phase_str = f" [{bet.get('phase', '')}]" if bet.get('phase') else ""
    print(f"\n  {C['red']}{B}{'═'*46}{R}")
    print(f"  {C['red']}{B}  ⚠  CONFIRMAR ORDEM REAL{phase_str}  ⚠{R}")
    print(f"  {C['red']}{B}{'═'*46}{R}")
    print(f"    Bracket : {bet['bracket']}")
    print(f"    Ask     : {bet['ask']*100:.1f}¢  (spread {bet.get('spread', 0)*100:.1f}¢)")
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
#  MAIN LOOP
# ══════════════════════════════════════════════════════
def run(wu_key: str, threshold: float, bankroll: float,
        kelly_frac: float, min_edge: float, interval: int,
        no_risk: bool = False, headless: bool = False):

    LOG_DIR.mkdir(exist_ok=True)

    if not wu_key:
        raise ValueError(
            f"\n  {C['red']}WU_API_KEY nao definida.{R}\n"
            "  export WU_API_KEY=\"a_tua_chave\"    (Linux/macOS)\n"
            "  Obtem em: https://www.wunderground.com/member/api-keys"
        )

    # Em modo headless (VPS), nao perguntar — usar REAL se POLY_PRIVATE_KEY definida
    if headless:
        trading_mode = TradingMode.REAL if POLY_PRIVATE_KEY else TradingMode.PAPER
        mode_str = "REAL" if trading_mode == TradingMode.REAL else "PAPER"
        print(f"  {DIM}Modo headless: {mode_str} (sem interaccao){R}")
    else:
        trading_mode = ask_trading_mode()
    tg = TG()

    # ── CLOB client ───────────────────────────────────
    clob = None
    if trading_mode == TradingMode.REAL and not POLY_PRIVATE_KEY:
        raise ValueError("POLY_PRIVATE_KEY nao definida. Impossivel usar modo REAL.")

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
            clob = None
    else:
        print(f"  {DIM}POLY_PRIVATE_KEY nao definida — modo PAPER sem order book CLOB.{R}")

    # ── OrderExecutor ─────────────────────────────────
    executor = None
    if POLY_PRIVATE_KEY:
        print(f"  {DIM}A inicializar OrderExecutor...{R}", end=" ", flush=True)
        try:
            executor = OrderExecutor(POLY_PRIVATE_KEY)
            print(f"{C['green']}✓{R}")
        except Exception as e:
            print(f"{C['red']}✗ {e}{R}")

    today     = berlin_date()
    log_path  = LOG_DIR / f"live_{today}.csv"
    bets_path = LOG_DIR / f"bets_{today}.json"
    wu_sess   = make_wu_session()

    if trading_mode == TradingMode.REAL and executor:
        real_balance = get_real_usdc_balance(POLY_PRIVATE_KEY)
        if real_balance is not None and real_balance > 0:
            bankroll = real_balance
        else:
            print(f"  {C['yellow']}⚠  Saldo indisponivel — a usar bankroll do argumento (${bankroll:.2f}){R}")

    print(f"\n{B}{C['cyan']}── Munich Live Bot ──────────────────────────────{R}")
    mode_label = f"{C['yellow']}PAPER{R}" if trading_mode == TradingMode.PAPER else f"{C['red']}REAL{R}"
    print(f"  Modo      : {mode_label}")
    print(f"  Threshold : {threshold*100:.0f}%   Min edge: {min_edge}%")
    print(f"  Estrategia: {B}3 Fases${R} (${BET_SIZE_PER_PHASE:.2f} cada = ${BET_SIZE_PER_PHASE*3:.2f} max/dia)")
    print(f"  Filtros   : Dupla Condicao (Modelo + Mercado Edge 10¢-90¢)")
    if trading_mode == TradingMode.REAL:
        print(f"  Bankroll  : {C['green']}{B}${bankroll:.2f} USDC{R}  {DIM}(saldo real){R}")
        print(f"  Stop-loss : ${POLY_MAX_DAILY_LOSS:.0f} USDC/dia")
    else:
        print(f"  Bankroll  : ${bankroll:.2f}  {DIM}(simulado){R}")
    print(f"  Intervalo : {interval}s  |  Fast-poll nas janelas :18-:32 e :45-:55 (Berlin)")
    print()

    # ── Carregar modelo ───────────────────────────────
    print("[1/4] A carregar modelo...")
    model, feat_cols, prior_map, monthly_threshold = load_model()
    set_seasonal_prior(prior_map)

    def get_threshold(month: int) -> float:
        return monthly_threshold.get(month, threshold) if monthly_threshold else threshold

    # ── Bootstrap ─────────────────────────────────────
    print(f"\n[2/4] Bootstrap — historico de hoje...")
    series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
    obs_min_today = dict(getattr(bootstrap_today, "_obs_min", {}))
    temps_by_hour = {s["hour"]: s["temp_c"] for s in slots_so_far}

    print(f"\n[3/4] Cloud cover das observacoes EDDM...", end=" ", flush=True)
    rows_cache    = getattr(bootstrap_today, "_rows_cache", [])
    cloud_by_hour = cloud_from_series(series_today, rows_cache)
    print(f"{C['green']}✓{R}")

    history_max = init_history_max()
    update_history_max(history_max, slots_so_far)

    print(f"\n[4/4] A aplicar modelo ao historico ({len(slots_so_far)} slots 30min)...")
    month   = today.month
    doy     = today.timetuple().tm_yday
    signals = {}

    for i, slot in enumerate(slots_so_far):
        h = slot["hour"]
        s = slot["slot30"]
        if h < MIN_HOUR or i < 3:
            continue
        current_extra = {
            "hour": h, "slot30": s,
            "cloud_cover":     slot.get("cloud_cover", 50),
            "humidity":        slot.get("humidity", 70),
            "prev_7d_avg_max": compute_prev7(history_max, today),
        }
        p_i = predict_p(model, feat_cols, slots_so_far[:i+1], current_extra, month, doy)
        signals[(h, s)] = p_i

    peak_detected = any(pv >= get_threshold(month) for pv in signals.values())

    market_date  = today
    market       = fetch_market(market_date)
    forecast_max = fetch_wu_forecast_max(wu_key, wu_sess)

    if market and clob:
        market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]

    usdc_balance = get_real_usdc_balance(POLY_PRIVATE_KEY) if (trading_mode == TradingMode.REAL and POLY_PRIVATE_KEY) else None
    open_orders  = executor.get_open_orders() if (trading_mode == TradingMode.REAL and executor) else None

    clob_mode_str   = "real" if trading_mode == TradingMode.REAL else "paper"
    threshold_month = get_threshold(today.month)

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

    # ── ESTADO DE FASES (Substitui o antigo bet_placed) ──
    phases_done: BetPhase = BetPhase.NONE
    bets: list  = []

    try:
        while True:
            now = local_now()

            # Novo dia (Berlin) — corre a qualquer hora, incluindo madrugada
            station_date = berlin_date()
            if station_date != today:
                today         = station_date
                market_date   = today
                slots_so_far  = []
                series_today  = {}
                obs_min_today = {}
                temps_by_hour = {}
                cloud_by_hour = {}
                signals       = {}
                peak_detected = False
                phases_done   = BetPhase.NONE  # Reset das 3 fases
                bets          = []
                log_path      = LOG_DIR / f"live_{today}.csv"
                bets_path     = LOG_DIR / f"bets_{today}.json"
                month         = today.month
                doy           = today.timetuple().tm_yday
                latest_obs    = None

                # Tentar obter mercado com retry
                market = None
                for _attempt in range(3):
                    market = fetch_market(market_date)
                    if market:
                        break
                    time.sleep(30)

                try:
                    series_today, slots_so_far = bootstrap_today(wu_key, wu_sess)
                    obs_min_today = dict(getattr(bootstrap_today, "_obs_min", {}))
                    rows_cache    = getattr(bootstrap_today, "_rows_cache", [])
                    cloud_by_hour = cloud_from_series(series_today, rows_cache)
                    temps_by_hour = {s["hour"]: s["temp_c"] for s in slots_so_far}
                except Exception as _e:
                    print(f"  {C['yellow']}Bootstrap falhou: {_e} — a retomar no proximo tick{R}")

                if market and clob:
                    market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]

                update_history_max(history_max, slots_so_far)

                tg.alert_started(
                    mode            = clob_mode_str,
                    bankroll        = bankroll,
                    threshold_arg   = threshold,
                    threshold_month = get_threshold(today.month),
                    month           = today.month,
                    market          = market,
                    today           = today,
                )

            # ── Ultima leitura WU ──────────────────────
            new_obs = fetch_wu_latest(wu_key, wu_sess)
            if new_obs:
                latest_obs = new_obs
                h_obs, m_obs = new_obs["hour"], new_obs["minute"]
                h_slot, s30  = ceil_slot(h_obs, m_obs)

                if DAY_START <= h_slot <= DAY_END:
                    series_today[(h_slot, s30)]  = new_obs["temp_c"]
                    obs_min_today[(h_slot, s30)]  = (h_obs, m_obs)

                    slot_entry = {
                        "hour":        h_slot, "slot30": s30,
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
                        slots_so_far.sort(key=lambda x: x["hour"] * 60 + x["slot30"])

                cloud_by_hour[h_slot] = new_obs.get("cloud_cover", 50)

            update_history_max(history_max, slots_so_far)

            # ── Calcular P ────────────────────────────
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

            # ── Detectar pico (transicao irreversivel) ──
            if p >= get_threshold(month) and not peak_detected:
                peak_detected = True
                if series_today:
                    _rs  = max(series_today, key=series_today.get)
                    _rm  = series_today[_rs]
                    _obs = (obs_min_today or {}).get(_rs)
                    _rts = f"{_obs[0]}:{_obs[1]:02d}" if _obs else f"{_rs[0]}h"
                else:
                    _rm  = 0
                    _rts = "?"
                tg.alert_peak_detected(p, _rm, _rts, None)

            if not (phases_done & BetPhase.GREEN) and tg.zone_changed(p):
                tg.alert_zone_change(p, tg.p_zone(p))

            # ── Actualizar mercado + forecast periodicamente ──
            if now.minute % 10 == 0 or not market:
                market = fetch_market(market_date)
                if market and clob:
                    market["brackets"] = [clob.enrich_bracket(b) for b in market["brackets"]]
            if now.minute % 30 == 0 or forecast_max is None:
                forecast_max = fetch_wu_forecast_max(wu_key, wu_sess)

            # ── Label da janela de sinal ──────────────
            berlin_min = berlin_now().minute
            signal_window_label = ""
            in_signal_window = any(lo <= berlin_min <= hi for lo, hi in _SIGNAL_CHECK_WINDOWS)
            if in_signal_window:
                signal_window_label = (f"  {C['cyan']}◉ a verificar sinal (:20){R}"
                                       if 18 <= berlin_min <= 32 else
                                       f"  {C['cyan']}◉ a verificar sinal (:50){R}")

            # ── Running max ───────────────────────────
            if series_today:
                rmax_slot = max(series_today, key=series_today.get)
                rmax      = series_today[rmax_slot]
            elif temps_by_hour:
                rmax = max(temps_by_hour.values())
            else:
                rmax = 0

            # FIX CRÍTICO: Garantir que bracket_temp tem sempre um valor válido
            # (No código antigo, se series_today e forecast_max fossem None, dava erro)
            if forecast_max and forecast_max.get("temp_max") is not None:
                bracket_temp = float(forecast_max["temp_max"])
            elif rmax > 0:
                bracket_temp = float(rmax)
            else:
                bracket_temp = 15.0 # Fallback seguro
                
            bracket = find_bracket(market, bracket_temp) if market else None

            # ── DUPLA CONDIÇÃO: BUSCA DINÂMICA ─────────
            # Se o mercado não confirmar o bracket exato (ex: ask a 99.8¢), 
            # procurar o bracket adjacente com melhor edge.
            if bracket:
                mkt_ok_exact, _ = market_confirms_bracket(bracket)
                if not mkt_ok_exact:
                    bracket = find_best_value_bracket(market, rmax if rmax > 0 else bracket_temp)

            if bracket and clob and not bracket.get("book"):
                bracket = clob.enrich_bracket(bracket)

            ask_price = (bracket.get("ask") or bracket.get("price")) if bracket else None
            ev        = compute_ev(p, ask_price) if ask_price else None

            if trading_mode == TradingMode.REAL and executor and now.minute % 5 == 0:
                usdc_balance = get_real_usdc_balance(POLY_PRIVATE_KEY)
                open_orders  = executor.get_open_orders() if executor else None

            # ══════════════════════════════════════════
            #  LÓGICA DE 3 APOSTAS (DUPLO FILTRO)
            # ══════════════════════════════════════════
            bet                = None
            bet_blocked_reason = None
            new_bet_placed     = False

            # 1. Validar Mercado (aplica a todas as fases)
            mkt_ok, mkt_reason = market_confirms_bracket(bracket)

            # Stop-loss bloqueia tudo
            if trading_mode == TradingMode.REAL and clob and clob.stop_loss_triggered():
                bet_blocked_reason = (f"stop-loss diario atingido "
                                     f"(${clob.daily_loss():.2f} >= ${POLY_MAX_DAILY_LOSS:.0f})")
            elif not market:
                bet_blocked_reason = "sem mercado Polymarket"
            elif not bracket:
                bet_blocked_reason = "bracket nao identificado"
            elif phases_done == BetPhase.DONE:
                pass # Silencioso, dia completo
            else:
                # 2. Verificar Triggers (Modelo) + Condição de Mercado
                berlin_h = berlin_now().hour
                berlin_m = berlin_now().minute
                
                phase_to_execute = None
                phase_label = ""
                phase_name = ""

                # Prioridade: VERDE > AMARELO > INICIAL (para não saltar fases)
                if not (phases_done & BetPhase.GREEN) and p >= 0.80 and mkt_ok:
                    phase_to_execute = BetPhase.GREEN
                    phase_label = f"{C['green']}VERDE (P>80%){R}"
                    phase_name = "GREEN"
                elif not (phases_done & BetPhase.YELLOW) and p >= 0.60 and mkt_ok:
                    phase_to_execute = BetPhase.YELLOW
                    phase_label = f"{C['yellow']}AMARELO (P>60%){R}"
                    phase_name = "YELLOW"
                elif not (phases_done & BetPhase.INITIAL) and berlin_h >= 10 and mkt_ok:
                    phase_to_execute = BetPhase.INITIAL
                    phase_label = f"{C['cyan']}INICIAL (10:00){R}"
                    phase_name = "INITIAL"

                if phase_to_execute is None:
                    # Determinar a razão exata do bloqueio para o dashboard
                    if not mkt_ok:
                        bet_blocked_reason = f"MERCADO RECUSA: {mkt_reason}"
                    elif not (phases_done & BetPhase.INITIAL) and berlin_h < 10:
                        bet_blocked_reason = f"aguardar 10:00 Berlin (agora {berlin_h}:{berlin_m:02d})"
                    elif not (phases_done & BetPhase.GREEN) and p < 0.80:
                        bet_blocked_reason = f"aguardar zona verde (P={p*100:.0f}% < 80%)"
                    elif not (phases_done & BetPhase.YELLOW) and p < 0.60:
                        bet_blocked_reason = f"aguardar zona amarela (P={p*100:.0f}% < 60%)"
                else:
                    # 3. EXECUTAR APOSTA DA FASE
                    bet_record = build_bet_record(
                        bracket, p, ev, bankroll, kelly_frac, trading_mode, 
                        max_daily_loss=POLY_MAX_DAILY_LOSS,
                        phase_name=phase_name
                    )

                    if trading_mode == TradingMode.PAPER:
                        result = paper_buy(
                            token_id  = bracket.get("token_id", ""),
                            price     = ask_price,
                            size_usdc = BET_SIZE_PER_PHASE,
                            label     = f"{bracket['label']} [{phase_name}]",
                        )
                        bet_record["order_id"] = result["order_id"]
                        bet_record["status"]   = result["status"]
                        bet        = bet_record
                        phases_done |= phase_to_execute
                        new_bet_placed = True

                    else:
                        _do_order = headless or confirm_real_order(bet_record)
                        if _do_order:
                            if not executor:
                                bet_blocked_reason = "OrderExecutor nao disponivel"
                            else:
                                result = executor.buy(
                                    token_id  = bracket.get("token_id", ""),
                                    price     = ask_price,
                                    size_usdc = BET_SIZE_PER_PHASE,
                                    label     = f"{bracket['label']} [{phase_name}]",
                                )
                                if result["success"]:
                                    bet_record["order_id"] = result["order_id"]
                                    bet_record["status"]   = result["status"]
                                    bet        = bet_record
                                    phases_done |= phase_to_execute
                                    new_bet_placed = True
                                    usdc_balance = get_real_usdc_balance(POLY_PRIVATE_KEY)
                                    open_orders  = executor.get_open_orders() if executor else None
                                    print(f"\n  {C['green']}✓ Ordem {phase_label} enviada — "
                                          f"ID: {result['order_id']}{R}")
                                    tg.alert_order_placed(bet_record)
                                else:
                                    bet_blocked_reason = f"Ordem falhou: {result['error']}"
                                    print(f"\n  {C['red']}✗ Falha na ordem: {result['error']}{R}")
                                    tg.alert_order_failed(result["error"], bracket)
                        else:
                            bet_blocked_reason = "confirmacao recusada pelo utilizador"

                    if bet:
                        bets.append(bet)
                        bets_path.write_text(json.dumps(bets, indent=2, default=str))
                        if trading_mode == TradingMode.PAPER:
                            tg.alert_order_placed(bet)

            # ── Sinais por hora para o dashboard ──────
            signals_by_hour: dict[int, float] = {}
            for (sh, ss), sp in signals.items():
                if sh not in signals_by_hour or sp > signals_by_hour[sh]:
                    signals_by_hour[sh] = sp

            temp_now = latest_obs["temp_c"] if latest_obs else 0

            if clob:
                clob.positions.refresh(clob)

            display(
                now, latest_obs, temps_by_hour, series_today, signals_by_hour, p,
                market, bracket, ev, bet,
                len(series_today), bankroll, eff_thr, peak_detected,
                trading_mode       = trading_mode,
                daily_loss         = clob.daily_loss() if clob else 0.0,
                max_daily_loss     = POLY_MAX_DAILY_LOSS,
                usdc_balance       = usdc_balance,
                positions          = clob.positions if clob else None,
                bet_blocked_reason = bet_blocked_reason,
                bet_placed         = new_bet_placed,
                forecast_max       = forecast_max,
                berlin_now_dt      = berlin_now(),
                market_date        = market_date,
                executor           = executor,
                open_orders        = open_orders,
                signal_window_label= signal_window_label,
                obs_min_today      = obs_min_today,
            )

            log_tick(
                now, temp_now, p, peak_detected, bracket, ev, bet, log_path,
                trading_mode       = trading_mode,
                bet_blocked_reason = bet_blocked_reason if not new_bet_placed else None,
            )

            # ── Override manual ('f' + Enter) ─────────
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
                        print(f"\n  {C['yellow']}{B}◈  Entrada forcada registada — "
                              f"{forced_bet['bracket']}  ${forced_bet['bet_size']:.2f}{R}\n")
                        time.sleep(2)
                    elif forced_err:
                        print(f"\n  {C['red']}✗ Entrada forcada cancelada: {forced_err}{R}\n")
                        time.sleep(1)

            # ── Smart sleep (fast-poll nas janelas EDDM) ──────────────────
            _berlin_h = berlin_now().hour
            _last_t   = latest_obs["temp_c"] if latest_obs else None
            if _berlin_h < 8 or _berlin_h >= 21:
                time.sleep(300)   # 5 min de madrugada
            else:
                smart_sleep(interval, wu_key, wu_sess, _last_t)

            # ── Dashboard periodica Telegram (30 min) ─────────────────────
                        # ── Dashboard periodica Telegram (30 min) ─────────────────────
            if time.time() - _tg_last_dashboard >= _tg_dashboard_interval:
                _tg_last_dashboard = time.time()
                _rmax_ts = "?"
                if series_today:
                    _rs  = max(series_today, key=series_today.get)
                    _obs = (obs_min_today or {}).get(_rs)
                    _rmax_ts = f"{_obs[0]}:{_obs[1]:02d}" if _obs else f"{_rs[0]}h"
                
                # Recolher dados extras para o TG
                _tg_open_pos = []
                _tg_summary = None
                if clob and clob.positions:
                    _tg_open_pos = clob.positions.open_positions()
                    _tg_summary = clob.positions.pnl_summary()

                tg.dashboard(
                    today         = today,
                    p             = p,
                    rmax          = rmax,
                    rmax_time     = _rmax_ts,
                    temp_now      = latest_obs["temp_c"] if latest_obs else None,
                    forecast_max  = forecast_max,
                    market        = market,
                    bracket       = bracket,
                    ev            = ev,
                    peak_detected = peak_detected,
                    bet           = bets[-1] if bets else None,
                    clob_mode     = clob_mode_str,
                    reason        = "periodic",
                    open_positions = _tg_open_pos,
                    positions_summary = _tg_summary,
                    usdc_balance  = usdc_balance,
                    phases_done   = phases_done,
                    bet_blocked_reason = bet_blocked_reason,
                )

    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Stopped.  Logs em ./{LOG_DIR}/{R}")
        tg.alert_stopped(bets, clob_mode_str)
        if bets:
            mode_label = "simuladas" if trading_mode == TradingMode.PAPER else "reais"
            print(f"  {C['green']}{len(bets)} ordens {mode_label} → {bets_path}{R}")

        if clob:
            today_pos = clob.positions.today_position()
            if today_pos and today_pos.status.value == "open":
                print(f"\n  {C['yellow']}Tens uma posicao aberta: {today_pos.bracket_label}  "
                      f"entrada {today_pos.entry_ask*100:.1f}¢{R}")
                try:
                    ans = input(f"  Fechar posicao ao bid actual? ({C['green']}y{R}/{C['red']}n{R}): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "n"
                if ans == "y":
                    book = clob.get_orderbook(today_pos.token_id)
                    bid  = book.best_bid if book else None
                    if bid:
                        result = clob.sell_yes(today_pos, bid)
                        if result.success:
                            pnl  = today_pos.pnl_usd or 0
                            sign = "+" if pnl >= 0 else ""
                            print(f"  {C['green']}✓ Posicao fechada a {bid*100:.1f}¢  "
                                  f"P&L: {sign}${pnl:.2f}{R}")
                        else:
                            print(f"  {C['red']}✗ Falha ao fechar: {result.error}{R}")
                    else:
                        print(f"  {C['red']}Bid nao disponivel — posicao mantida aberta.{R}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Munich Max Temp — Live Bot (WU + Polymarket + LightGBM)"
    )
    parser.add_argument("--threshold",     type=float, default=0.46)
    parser.add_argument("--bankroll",      type=float, default=200.0)
    parser.add_argument("--kelly",         type=float, default=0.5)
    parser.add_argument("--min-edge",      type=float, default=5.0)
    parser.add_argument("--interval",      type=int,   default=60)
    parser.add_argument("--max-daily-loss",type=float, default=50.0,
                        help="Maxima perda aceite por dia em USDC (default: 50)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Modo headless: nao pede confirmacao manual de ordens REAL")
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
