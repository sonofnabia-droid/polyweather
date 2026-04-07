# munich_display.py — versão reescrita com risk-first, barra adaptativa,
# risk_used, risk_remaining, risk_per_trade, sem Kelly.

import csv
import os
import shutil

from munich_config import (
    DAY_START, DAY_END,
    R, B, DIM, C,
    _LOCAL,
    berlin_date,
)
from polymarket_clob import TradingMode, PositionStatus


# ══════════════════════════════════════════════════════
#  HELPERS VISUAIS
# ══════════════════════════════════════════════════════

def p_bar(p, w=14) -> str:
    f = round(p * w)
    return "█" * f + "░" * (w - f)


def p_col(p) -> str:
    if p < 0.40: return C["gray"]
    if p < 0.65: return C["orange"]
    if p < 0.80: return C["yellow"]
    return C["green"]


def _book_bar(price: float, w: int = 20) -> str:
    f = round(price * w)
    return "█" * f + "░" * (w - f)


def risk_bar(used_pct: float) -> str:
    """
    Barra adaptativa (sem limite máximo), monocromática (█ + ▒).
    """
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80

    bar_width = max(20, cols - 40)
    filled = int(bar_width * used_pct)
    empty = bar_width - filled
    return "█" * filled + "▒" * empty


# ══════════════════════════════════════════════════════
#  GRAFICO ASCII
# ══════════════════════════════════════════════════════

def draw_chart(series_today: dict, signals: dict,
               peak_detected: bool,
               plain: bool = False) -> list[str]:
    """
    plain=True → remove cores ANSI (para Telegram)
    """
    lines = []
    slots = [(h, m) for h in range(DAY_START, DAY_END + 1) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    if not avail:
        return ["  sem dados para grafico"]

    t_min  = min(avail) - 0.5
    t_max  = max(avail) + 0.5
    t_rng  = max(t_max - t_min, 1.0)
    chart_h = 8
    col_w   = 2
    total_w = len(slots) * col_w + 5

    grid = [[" "] * total_w for _ in range(chart_h)]

    # Eixo Y
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        label = f"{int(round(t_val)):>3}°"
        for ci, ch in enumerate(label):
            if ci < 4:
                grid[row][ci] = ch

    # Linha base
    for ci in range(4, total_w):
        grid[chart_h - 1][ci] = "─"

    def to_row(t):
        return int((1 - (t - t_min) / t_rng) * (chart_h - 1))

    # Plot
    for si, ((h, m), temp) in enumerate(zip(slots, temps)):
        if temp is None:
            continue
        row = to_row(temp)
        col = 4 + si * col_w
        p   = signals.get(h, 0)

        if plain:
            # versão Telegram (sem cores)
            if p >= 0.80:   sym = "██"
            elif p >= 0.60: sym = "▓▓"
            elif p >= 0.30: sym = "▒▒"
            else:           sym = "░░"
        else:
            # versão terminal (com cores)
            if p >= 0.80:   sym = f"{C['green']}██{R}"
            elif p >= 0.60: sym = f"{C['yellow']}██{R}"
            elif p >= 0.30: sym = f"{C['orange']}▓▓{R}"
            else:           sym = f"{DIM}▒▒{R}"

        if 0 <= row < chart_h - 1:
            grid[row][col]     = sym
            grid[row][col + 1] = ""

    # Converter grid em linhas
    for row in grid:
        lines.append("  " + "".join(row))

    # Eixo X
    x_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        x_line += f"{h:<4}"
    lines.append(x_line)

    # Linha P(pico)
    p_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        pv = signals.get(h, 0)
        if plain:
            if pv >= 0.80:   cell = "▓▓  "
            elif pv >= 0.60: cell = "▒▒  "
            elif pv >= 0.30: cell = "░░  "
            else:            cell = "    "
        else:
            if pv >= 0.80:   cell = f"{C['green']}▓▓{R}  "
            elif pv >= 0.60: cell = f"{C['yellow']}▒▒{R}  "
            elif pv >= 0.30: cell = f"{C['orange']}░░{R}  "
            else:            cell = f"{DIM}  {R}  "
        p_line += cell

    lines.append(p_line + " P(pico)")

    return lines


# ══════════════════════════════════════════════════════
#  ORDER BOOK
# ══════════════════════════════════════════════════════

def display_orderbook(book, bracket_label: str = "") -> None:
    if book is None:
        print(f"    {DIM}Order book CLOB indisponivel — a usar preco Gamma{R}")
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


# ══════════════════════════════════════════════════════
#  POSICOES
# ══════════════════════════════════════════════════════

def display_positions(positions, trading_mode: TradingMode,
                      usdc_balance: float | None = None) -> None:

    all_pos  = positions.all_positions()
    open_pos = positions.open_positions()
    summary  = positions.pnl_summary()
    mode_tag = "PAPER" if trading_mode == TradingMode.PAPER else "REAL"

    if trading_mode == TradingMode.REAL:
        if usdc_balance is not None:
            bal_col  = C["green"] if usdc_balance >= 10 else C["red"]
            bal_part = f"  {DIM}Saldo:{R} {bal_col}{B}${usdc_balance:,.2f}{R}"
        else:
            bal_part = f"  {DIM}Saldo: a carregar...{R}"
    else:
        bal_part = ""

    print(f"\n  {B}Posicoes [{mode_tag}]{R}{bal_part}  "
          f"{DIM}abertas:{summary['n_open']}  "
          f"ganhas:{summary['n_won']}  "
          f"perdidas:{summary['n_lost']}{R}")

    if not all_pos:
        print(f"    {DIM}Sem posicoes registadas ainda.{R}")
        return

    print(f"  {DIM}  {'Data':<12} {'Bracket':<18} {'Entrada':>7} "
          f"{'Actual':>7} {'P&L $':>8} {'P&L %':>7} {'Shares':>7}  Status{R}")
    print(f"  {DIM}  {'─'*78}{R}")

    today_str = berlin_date().isoformat()

    def _fmt(pos, highlight: bool = False) -> None:
        mid   = pos.current_mid
        pnl_u = pos.pnl_usd
        pnl_p = pos.pnl_pct

        if pnl_u is None:    pnl_col = DIM
        elif pnl_u > 0:      pnl_col = C["green"]
        elif pnl_u < 0:      pnl_col = C["red"]
        else:                 pnl_col = DIM

        status_map = {
            PositionStatus.OPEN:    (f"{C['cyan']}ABERTA{R}",    ""),
            PositionStatus.WON:     (f"{C['green']}{B}GANHOU{R}", "✓"),
            PositionStatus.LOST:    (f"{C['red']}PERDEU{R}",     "✗"),
            PositionStatus.EXPIRED: (f"{DIM}EXPIROU{R}",         "—"),
            PositionStatus.UNKNOWN: (f"{DIM}?{R}",               ""),
        }
        status_str, icon = status_map.get(pos.status, (f"{DIM}?{R}", ""))

        entry_s = f"{pos.entry_ask*100:.1f}¢"
        mid_s   = f"{mid*100:.1f}¢"  if mid   is not None else f"{DIM}—{R}"
        pnl_u_s = f"{pnl_u:+.2f}"   if pnl_u is not None else f"{DIM}—{R}"
        pnl_p_s = f"{pnl_p:+.1f}%"  if pnl_p is not None else f"{DIM}—{R}"
        pre     = f"  {B}{C['cyan']}▶ {R}" if highlight else "    "

        print(f"{pre}{pos.date_opened:<12} "
              f"{pos.bracket_label:<18} "
              f"{DIM}{entry_s:>7}{R} "
              f"{mid_s:>7} "
              f"{pnl_col}{pnl_u_s:>8}{R} "
              f"{pnl_col}{pnl_p_s:>7}{R} "
              f"{DIM}{pos.shares:>7.2f}{R}  "
              f"{status_str} {icon}")

    shown_ids: set[str] = set()
    for pos in reversed(all_pos):
        if pos.date_opened == today_str:
            _fmt(pos, highlight=True)
            shown_ids.add(pos.order_id)
            break

    other_open = [p for p in open_pos if p.order_id not in shown_ids]
    if other_open:
        print(f"  {DIM}  {'─'*78}{R}")
        for pos in sorted(other_open, key=lambda p: p.date_opened, reverse=True):
            _fmt(pos)
            shown_ids.add(pos.order_id)

    closed = [p for p in all_pos
              if p.status in (PositionStatus.WON, PositionStatus.LOST, PositionStatus.EXPIRED)
              and p.order_id not in shown_ids]
    if closed:
        print(f"  {DIM}  {'─'*78}{R}")
        for pos in sorted(closed, key=lambda p: p.date_opened, reverse=True)[:5]:
            _fmt(pos)

    if all_pos:
        pnl_col = C["green"] if summary["total_pnl_usd"] >= 0 else C["red"]
        print(f"  {DIM}  {'─'*78}{R}")
        print(f"    {DIM}Total investido: ${summary['total_invested']:.2f}   "
              f"P&L total: {R}"
              f"{pnl_col}{B}{summary['total_pnl_usd']:+.2f} "
              f"({summary['total_pnl_pct']:+.1f}%){R}")

# ══════════════════════════════════════════════════════
#  DASHBOARD PRINCIPAL
# ══════════════════════════════════════════════════════

def display(now, latest_obs, temps_by_hour, series_today, signals, p,
            market, bracket, ev, bet,
            n_wu_reads, bankroll, threshold, peak_detected,
            trading_mode: TradingMode = TradingMode.PAPER,
            daily_loss: float = 0.0, max_daily_loss: float = 50.0,
            usdc_balance: float | None = None,
            positions=None,
            executor=None,
            open_orders: list | None = None,
            market_date=None,
            bet_blocked_reason=None, bet_placed=False,
            forecast_max=None, berlin_now_dt=None,
            signal_window_label="",
            obs_min_today: dict = None):

    os.system('clear' if os.name != 'nt' else 'cls')
    pc = p_col(p)

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

    # ─────────────────────────────────────────────────────────────
    # RISK SECTION (risk-first)
    # ─────────────────────────────────────────────────────────────
    risk_used = min(1.0, daily_loss / max_daily_loss) if max_daily_loss > 0 else 0
    risk_remaining = max_daily_loss - daily_loss
    risk_per_trade = min(max_daily_loss, bankroll * 0.10)

    if trading_mode == TradingMode.REAL:
        if usdc_balance is not None:
            bal_col = C["green"] if usdc_balance >= 10 else C["red"]
            bal_str = f"{bal_col}{B}${usdc_balance:,.2f} USDC{R}"
        else:
            bal_str = f"{C['yellow']}a carregar...{R}"

        print(f"  {B}Saldo:{R} {bal_str}")

        print(f"  {DIM}Risk per trade:{R} ${risk_per_trade:.2f}   "
              f"{DIM}Max daily loss:{R} ${max_daily_loss:.2f}")

        print(f"  {DIM}Risk used:{R} {risk_used*100:5.1f}%")
        print(f"  {risk_bar(risk_used)}")
        print(f"  {DIM}Used:{R} ${daily_loss:.2f}   "
              f"{DIM}Remaining:{R} ${risk_remaining:.2f}")

    print(f"  {DIM}{'─'*58}{R}")

    # ─────────────────────────────────────────────────────────────
    # TEMPERATURE CHART
    # ─────────────────────────────────────────────────────────────
    print(f"\n  {B}Curva de temperatura hoje{R}  "
          f"{DIM}(● verde=P>80% amarelo=P>60% laranja=P>30%){R}")
    for line in draw_chart(series_today, signals, peak_detected):
        print(line)

    # Running max
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
        cloud_str = {0: "Clear", 12: "Few clouds", 37: "Partly cloudy",
                     75: "Mostly cloudy", 100: "Cloudy"}.get(cloud_now, f"{cloud_now}%")
        print(f"    {B}{C['white']}{int(round(temp_now)):>4}°C{R}  "
              f"{DIM}humidade:{int(round(hum_now))}%  {cloud_str}  {wx}{R}")
    print(f"    {DIM}running max:{R} {B}{C['white']}{int(round(rmax))}°C{R}  "
          f"{DIM}@{R} {C['cyan']}{B}{rmax_time_str}{R}")
    if forecast_max:
        fc_col = C["green"] if forecast_max["temp_max"] <= rmax else C["yellow"]
        print(f"    {DIM}previsao WU :{R} max {fc_col}{B}{forecast_max['temp_max']}°C{R}  "
              f"{DIM}min {forecast_max.get('temp_min', '?')}°C{R}")

    # ─────────────────────────────────────────────────────────────
    # MODEL OUTPUT
    # ─────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────
    # MARKET
    # ─────────────────────────────────────────────────────────────
    _md        = market_date or berlin_date()
    _today_ref = berlin_date()
    if _md > _today_ref:
        market_label = (f"{B}Polymarket — Mercado de {_md}{R}  "
                        f"{C['yellow']}{B}[AMANHA / FUTURO]{R}")
    else:
        market_label = f"{B}Polymarket — Mercado de Hoje  {DIM}({_md}){R}"
    print(f"\n  {market_label}")

    if not market:
        print(f"    {C['red']}✗ Mercado nao encontrado{R}  "
              f"{DIM}(sera criado esta manha pelo Polymarket){R}")
    else:
        print(f"    {B}{C['cyan']}{market['title'][:56]}{R}")
        print(f"    {DIM}vol: ${market['volume']:,.0f}  "
              f"encerra: {market['end_date'][:10]}  "
              f"{market['n_outcomes']} brackets{R}")
        print()
        print(f"  {DIM}  {'Bracket':<18}  {'Bid':>6}  {'Ask':>6}  {'Spread':>6}  {'Bar':16}  {'Vol':>7}{R}")
        print(f"  {DIM}  {'─'*70}{R}")
        for b in market["brackets"]:
            is_t    = bracket and b["label"] == bracket["label"]
            b_ask   = b.get("ask") or b.get("price") or 0
            b_bid   = b.get("bid") or b_ask
            b_spread = b.get("spread")
            bw      = 16
            bbar    = "█" * round(b_ask * bw) + "░" * (bw - round(b_ask * bw))
            if b_ask < 0.20:   pc2 = C["green"]
            elif b_ask < 0.50: pc2 = C["cyan"]
            elif b_ask < 0.75: pc2 = C["yellow"]
            else:               pc2 = C["orange"]
            pre   = f"  {B}{C['green']}→ {R}" if is_t else "    "
            tag   = f"  {B}{C['green']}◆ running max{R}" if is_t else ""
            vol_s = f"{DIM}${b.get('volume', 0):>6,.0f}{R}" if b.get("volume") else ""
            spr_s = f"{b_spread*100:.1f}¢" if b_spread else f"{DIM}—{R}"
            print(f"{pre}{b['label']:<18}  "
                  f"{C['green']}{b_bid*100:>5.1f}¢{R}  "
                  f"{C['red']}{b_ask*100:>5.1f}¢{R}  "
                  f"{DIM}{spr_s:>6}{R}  "
                  f"{pc2}{bbar}{R}  "
                  f"{vol_s}{tag}")

        if bracket and bracket.get("book"):
            print(f"\n  {B}Order Book CLOB — {bracket['label']}{R}")
            display_orderbook(bracket.get("book"), bracket.get("label", ""))

    # ─────────────────────────────────────────────────────────────
    # EDGE ANALYSIS (sem Kelly)
    # ─────────────────────────────────────────────────────────────
    if bracket and ev:
        print(f"\n  {B}Edge Analysis{R}  {DIM}(EV calculado sobre ask){R}")
        ec       = C["green"] if ev["ev_positive"] else C["red"]
        ask_disp = f"{ev['ask']*100:.1f}¢"
        print(f"    Ask: {C['red']}{B}{ask_disp}{R}  "
              f"EV/share: {ec}{B}{ev['ev_cents']:+.1f}¢{R}  "
              f"edge: {ec}{ev['edge_pct']:+.1f}%{R}  "
              f"bankroll: ${bankroll:.0f}")

    # ─────────────────────────────────────────────────────────────
    # POSITIONS
    # ─────────────────────────────────────────────────────────────
    if positions is not None:
        display_positions(positions, trading_mode, usdc_balance=usdc_balance)
    else:
        print(f"\n  {B}Posicoes{R}  {DIM}CLOB nao disponivel{R}")

    # ─────────────────────────────────────────────────────────────
    # STOP-LOSS
    # ─────────────────────────────────────────────────────────────
    if trading_mode == TradingMode.REAL and daily_loss >= max_daily_loss:
        print(f"\n  {C['red']}{B}⛔  STOP-LOSS DIARIO ATINGIDO — novas ordens bloqueadas{R}")

    # ─────────────────────────────────────────────────────────────
    # BET SECTION (risk-first, sem Kelly)
    # ─────────────────────────────────────────────────────────────
    if not bet and peak_detected and not bet_placed:
        if bet_blocked_reason:
            print(f"\n  {C['yellow']}⚠  Bet bloqueada: {bet_blocked_reason}{R}")
    elif bet_placed and not bet:
        mode_label = "simulada" if trading_mode == TradingMode.PAPER else "enviada"
        print(f"\n  {DIM}  Ordem ja {mode_label} anteriormente{R}")

    if bet:
        border_col   = C["yellow"] if trading_mode == TradingMode.PAPER else C["red"]
        header_label = (f"{C['yellow']}{B}  ◆  BET SIMULADA (PAPER)  ◆{R}"
                        if trading_mode == TradingMode.PAPER
                        else f"{C['red']}{B}  ◆  ORDEM ENVIADA (REAL)   ◆{R}")
        print(f"\n  {border_col}{B}{'─'*44}{R}")
        print(f"  {header_label}")
        print(f"    Bracket    : {bet['bracket']}")
        if bet.get('spread'):
            print(f"    Bid / Ask  : {bet.get('bid', 0)*100:.1f}¢  /  {bet['ask']*100:.1f}¢  "
                  f"(spread {bet.get('spread', 0)*100:.1f}¢)")
        else:
            print(f"    Ask        : {bet['ask']*100:.1f}¢")

        print(f"    Sizing     : risk-first  (max loss ${bet['max_daily_loss']:.0f})")
        print(f"    Aposta     : ${bet['bet_size']:.2f}  ({bet['shares']:.2f} shares)")
        print(f"    Max profit : +${bet['max_profit']:.2f}")
        if bet.get('order_id'):
            print(f"    Order ID   : {bet['order_id']}")
        if bet.get('status'):
            print(f"    Status     : {bet['status']}")
        print(f"  {border_col}{B}{'─'*44}{R}")

    # ─────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────
    print(f"\n  {DIM}{'─'*58}{R}")
    if trading_mode == TradingMode.REAL:
        if usdc_balance is not None:
            bal_col  = C["green"] if usdc_balance >= 10 else C["red"]
            bal_disp = f"{bal_col}{B}${usdc_balance:,.2f} USDC{R}"
        else:
            bal_disp = f"{DIM}a carregar...{R}"
        print(f"  {B}Polymarket{R}  Saldo: {bal_disp}", end="")
        if open_orders is not None:
            n_open = len(open_orders)
            if n_open == 0:
                print(f"   {DIM}Ordens abertas: 0{R}")
            else:
                print(f"   {C['yellow']}{B}Ordens abertas: {n_open}{R}")
                for o in open_orders[:5]:
                    oid   = str(o.get("id") or o.get("orderID") or "?")[:12]
                    side  = o.get("side", "?")
                    price = o.get("price", "?")
                    size  = o.get("size", "?")
                    print(f"    {DIM}{oid}  {side}  price={price}  size={size}{R}")
        else:
            print()
    else:
        n_sim = len(positions.all_positions()) if positions else 0
        print(f"  {B}Polymarket{R}  {DIM}[PAPER — sem ordens reais]  "
              f"posicoes simuladas: {n_sim}{R}")

    print(f"\n  {DIM}WU reads hoje: {n_wu_reads}  Ctrl+C para parar{R}\n")


# ══════════════════════════════════════════════════════
#  LOGGING CSV
# ══════════════════════════════════════════════════════

def log_tick(now, temp, p, peak_detected, bracket, ev, bet,
             path, trading_mode: TradingMode = TradingMode.PAPER,
             bet_blocked_reason=None) -> None:
    row = {
        "timestamp":          now.isoformat(),
        "mode":               trading_mode.value,
        "temp":               temp,
        "p_peak":             round(p, 4),
        "peak_detected":      peak_detected,
        "bracket":            bracket["label"] if bracket else None,
        "ask":                (bracket.get("ask") or bracket["price"]) if bracket else None,
        "bid":                bracket.get("bid") if bracket else None,
        "spread":             bracket.get("spread") if bracket else None,
        "ev_cents":           ev["ev_cents"] if ev else None,
        "bet_size":           bet["bet_size"] if bet else None,
        "bet_placed":         bet is not None,
        "order_id":           bet.get("order_id") if bet else None,
        "bet_blocked_reason": bet_blocked_reason or "",
    }
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
