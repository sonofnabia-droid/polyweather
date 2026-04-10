# munich_display.py — V3 com risk-first, ensemble breakdown, 3 parcelas e dual forecast

import csv
import os
import shutil

from munich_config import (
    DAY_START, DAY_END,
    R, B, DIM, C,
    _LOCAL, berlin_date,
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
               peak_detected: bool, plain: bool = False) -> list[str]:
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

    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        label = f"{int(round(t_val)):>3}°"
        for ci, ch in enumerate(label):
            if ci < 4: grid[row][ci] = ch

    for ci in range(4, total_w):
        grid[chart_h - 1][ci] = "─"

    def to_row(t): return int((1 - (t - t_min) / t_rng) * (chart_h - 1))

    for si, ((h, m), temp) in enumerate(zip(slots, temps)):
        if temp is None: continue
        row = to_row(temp)
        col = 4 + si * col_w
        p   = signals.get(h, 0)

        if plain:
            sym = "██" if p >= 0.80 else "▓▓" if p >= 0.60 else "▒▒" if p >= 0.30 else "░░"
        else:
            if p >= 0.80:   sym = f"{C['green']}██{R}"
            elif p >= 0.60: sym = f"{C['yellow']}██{R}"
            elif p >= 0.30: sym = f"{C['orange']}▓▓{R}"
            else:           sym = f"{DIM}▒▒{R}"

        if 0 <= row < chart_h - 1:
            grid[row][col]     = sym
            grid[row][col + 1] = ""

    for row in grid:
        lines.append("  " + "".join(row))

    x_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        x_line += f"{h:<4}"
    lines.append(x_line)

    p_line = "  " + " " * 4
    for h in range(DAY_START, DAY_END + 1):
        pv = signals.get(h, 0)
        if plain:
            cell = "▓▓  " if pv >= 0.80 else "▒▒  " if pv >= 0.60 else "░░  " if pv >= 0.30 else "    "
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
        print(f"    {DIM}Order book CLOB indisponivel{R}")
        return
    bid = book.best_bid; ask = book.best_ask; spr = book.spread
    bid_str = f"{bid*100:>5.1f}¢" if bid else "  — "
    ask_str = f"{ask*100:>5.1f}¢" if ask else "  — "
    spr_str = f"{spr*100:.1f}¢" if spr else "—"
    
    print(f"    {DIM}{'Bid':>8}  {'':20}  {'Ask':>8}  {'Spread':>8}{R}")
    print(f"    {C['green']}{B}{bid_str:>8}{R}  {DIM}{_book_bar((bid or 0))}{R}  "
          f"{C['red']}{B}{ask_str:>8}{R}  {DIM}{spr_str:>8}{R}")

    n_levels = min(3, max(len(book.bids), len(book.asks)))
    if n_levels > 1:
        print(f"    {DIM}{'─'*52}{R}")
        for i in range(n_levels):
            b_lv = book.bids[i] if i < len(book.bids) else None
            a_lv = book.asks[i] if i < len(book.asks) else None
            b_s = f"{b_lv.price*100:>5.1f}¢ × {b_lv.size:>6.0f}" if b_lv else " " * 16
            a_s = f"{a_lv.price*100:>5.1f}¢ × {a_lv.size:>6.0f}" if a_lv else " " * 16
            print(f"    {DIM}  {C['green']}{b_s}{DIM}    {C['red']}{a_s}{R}")
    print(f"    {DIM}Depth bid: ${book.bid_depth_usdc:,.0f}  ask: ${book.ask_depth_usdc:,.0f}{R}")


# ══════════════════════════════════════════════════════
#  POSICOES
# ══════════════════════════════════════════════════════

def display_positions(positions, trading_mode: TradingMode,
                      usdc_balance: float | None = None) -> None:
    all_pos  = positions.all_positions()
    open_pos = positions.open_positions()
    summary  = positions.pnl_summary()
    mode_tag = "PAPER" if trading_mode == TradingMode.PAPER else "REAL"

    if trading_mode == TradingMode.REAL and usdc_balance is not None:
        bal_col  = C["green"] if usdc_balance >= 10 else C["red"]
        bal_part = f"  {DIM}Saldo:{R} {bal_col}{B}${usdc_balance:,.2f}{R}"
    else:
        bal_part = ""

    print(f"\n  {B}Posicoes [{mode_tag}]{R}{bal_part}  "
          f"{DIM}abertas:{summary['n_open']}  ganhas:{summary['n_won']}  perdidas:{summary['n_lost']}{R}")

    if not all_pos:
        print(f"    {DIM}Sem posicoes registadas.{R}")
        return

    print(f"  {DIM}  {'Data':<12} {'Bracket':<18} {'Entrada':>7} {'Actual':>7} {'P&L $':>8} {'Status':>10}{R}")
    print(f"  {DIM}  {'─'*68}{R}")

    today_str = berlin_date().isoformat()

    def _fmt(pos, highlight=False):
        mid = pos.current_mid; pnl_u = pos.pnl_usd; pnl_p = pos.pnl_pct
        pnl_col = C["green"] if (pnl_u or 0) > 0 else C["red"] if (pnl_u or 0) < 0 else DIM
        status_map = {
            "open": f"{C['cyan']}ABERTA{R}", "won": f"{C['green']}{B}GANHOU{R}",
            "lost": f"{C['red']}PERDEU{R}", "expired": f"{DIM}EXPIROU{R}",
        }
        status_str = status_map.get(pos.status.value if hasattr(pos.status, 'value') else str(pos.status), f"{DIM}?{R}")
        entry_s = f"{pos.entry_ask*100:.1f}¢"
        mid_s   = f"{mid*100:.1f}¢" if mid is not None else f"{DIM}—{R}"
        pnl_u_s = f"{pnl_u:+.2f}" if pnl_u is not None else f"{DIM}—{R}"
        pre = f"  {B}{C['cyan']}▶ {R}" if highlight else "    "
        print(f"{pre}{pos.date_opened:<12} {pos.bracket_label:<18} {DIM}{entry_s:>7}{R} "
              f"{mid_s:>7} {pnl_col}{pnl_u_s:>8}{R}  {status_str}")

    shown = set()
    for pos in reversed(all_pos):
        if pos.date_opened == today_str and pos.order_id not in shown:
            _fmt(pos, highlight=True); shown.add(pos.order_id); break

    for pos in sorted(open_pos, key=lambda p: p.date_opened, reverse=True):
        if pos.order_id not in shown: _fmt(pos); shown.add(pos.order_id)

    closed = [p for p in all_pos if p.status.value in ("won","lost","expired") and p.order_id not in shown]
    for pos in sorted(closed, key=lambda p: p.date_opened, reverse=True)[:5]:
        _fmt(pos)

    if all_pos:
        pnl_col = C["green"] if summary["total_pnl_usd"] >= 0 else C["red"]
        print(f"  {DIM}  {'─'*68}{R}")
        print(f"    {DIM}Total investido: ${summary['total_invested']:.2f}   P&L total: {R}"
              f"{pnl_col}{B}{summary['total_pnl_usd']:+.2f} ({summary['total_pnl_pct']:+.1f}%){R}")


# ══════════════════════════════════════════════════════
#  DASHBOARD PRINCIPAL
# ══════════════════════════════════════════════════════

def display(now, latest_obs, temps_by_hour, series_today, signals, p,
            market, bracket, ev, bet,
            n_wu_reads, bankroll, threshold, peak_detected,
            trading_mode: TradingMode = TradingMode.PAPER,
            daily_loss: float = 0.0, max_daily_loss: float = 50.0,
            usdc_balance: float | None = None,
            positions=None, executor=None, open_orders: list | None = None,
            market_date=None, bet_blocked_reason=None, bet_placed=False,
            forecast_max=None, berlin_now_dt=None,
            signal_window_label="", obs_min_today: dict = None,
            phased=None, forecast_agreement=None, om_forecast=None,
            ensemble_result=None):

    os.system('clear' if os.name != 'nt' else 'cls')
    pc = p_col(p)

    berlin_str = berlin_now_dt.strftime('%H:%M:%S') if berlin_now_dt else "?"
    local_str  = now.strftime('%Y-%m-%d %H:%M:%S')
    mode_tag   = (f"{C['yellow']}{B}[ PAPER ]{R}" if trading_mode == TradingMode.PAPER
                  else f"{C['red']}{B}[ REAL  ]{R}")

    print(f"\n  {B}{C['cyan']}Munich Max Temp — V3{R}  {mode_tag}  "
          f"{DIM}{local_str}  │  Munich {R}{B}{C['white']}{berlin_str}{R}")

    # ── Risk Section ──────────────────────────────────
    if trading_mode == TradingMode.REAL:
        risk_used = min(1.0, daily_loss / max_daily_loss) if max_daily_loss > 0 else 0
        risk_remaining = max_daily_loss - daily_loss
        if usdc_balance is not None:
            bal_col = C["green"] if usdc_balance >= 10 else C["red"]
            print(f"  {B}Saldo:{R} {bal_col}{B}${usdc_balance:,.2f}{R}   {DIM}Risk used:{R} {risk_used*100:.0f}%   {DIM}Remaining:{R} ${risk_remaining:.2f}")
    print(f"  {DIM}{'─'*58}{R}")

    # ── Temperatura ───────────────────────────────────
    print(f"\n  {B}Curva de temperatura hoje{R}")
    for line in draw_chart(series_today, signals, peak_detected):
        print(line)

    if series_today:
        rmax_slot = max(series_today, key=series_today.get)
        rmax = series_today[rmax_slot]
        rmax_real_ts = (obs_min_today or {}).get(rmax_slot)
        rmax_time_str = (f"{rmax_real_ts[0]}:{rmax_real_ts[1]:02d}" if rmax_real_ts else f"{rmax_slot[0]}h")
    elif temps_by_hour:
        rmax = max(temps_by_hour.values()); rmax_time_str = f"{max(temps_by_hour, key=temps_by_hour.get)}h"
    else:
        rmax = 0; rmax_time_str = "?"

    print(f"\n  {B}Temperatura{R}  {DIM}({n_wu_reads} leituras){R}")
    if latest_obs:
        cloud_now = latest_obs.get("cloud_cover", 50)
        cloud_str = {0:"Clear",12:"Few clouds",37:"Partly cloudy",75:"Mostly cloudy",100:"Cloudy"}.get(cloud_now, f"{cloud_now}%")
        print(f"    {B}{C['white']}{int(round(latest_obs['temp_c'])):>4}°C{R}  {DIM}hum:{int(round(latest_obs.get('humidity',70)))}%  {cloud_str}{R}")
    print(f"    {DIM}running max:{R} {B}{C['white']}{int(round(rmax))}°C{R}  {DIM}@{R} {C['cyan']}{B}{rmax_time_str}{R}")

    # ── Dual Forecast ─────────────────────────────────
    if forecast_max or om_forecast:
        print(f"\n  {B}Previsao Dual{R}")
        if forecast_max:
            fc_col = C["green"] if forecast_max["temp_max"] <= rmax else C["yellow"]
            print(f"    {C['cyan']}WU{R} : max {fc_col}{B}{forecast_max['temp_max']}°C{R}")
        if om_forecast:
            om_col = C["green"] if om_forecast["temp_max"] <= rmax else C["yellow"]
            print(f"    {C['purple']}OM{R} : max {om_col}{B}{om_forecast['temp_max']}°C{R}")
        if forecast_agreement:
            if forecast_agreement["valid"]:
                print(f"    {C['green']}✓ CONCORDAM{R} (diff {forecast_agreement.get('diff','?')}°C)  consenso={B}{forecast_agreement.get('consensus_max','?')}°C{R}")
            else:
                print(f"    {C['red']}✗ DISCORDAM{R} ({forecast_agreement.get('reason','?')})")

    # ── Ensemble ──────────────────────────────────────
    if ensemble_result:
        p_ens = ensemble_result["p_ensemble"]
        print(f"\n  {B}Ensemble{R}  {pc}{B}{p_bar(p_ens)}{R}  {pc}{B}{p_ens*100:>5.1f}%{R}  {DIM}threshold: {threshold*100:.0f}%{R}")

        p_lgbm = ensemble_result.get("p_lgbm", 0)
        p_xgb  = ensemble_result.get("p_xgb")
        p_zs   = ensemble_result.get("p_zscore")

        print(f"    {DIM}LGBM  : {p_lgbm*100:>5.1f}%  (peso 50%){R}")
        if p_xgb is not None:
            print(f"    {DIM}XGB   : {p_xgb*100:>5.1f}%  (peso 30%){R}")
        else:
            print(f"    {DIM}XGB   : {C['yellow']}N/A{R}")
        if p_zs is not None:
            print(f"    {DIM}Z-Score: {p_zs*100:>5.1f}%  (peso 20%){R}")
        else:
            print(f"    {DIM}Z-Score: {C['yellow']}N/A{R}")

        if p_ens >= 0.80:   status = f"{C['green']}{B}✓ PICO DETECTADO{R}"
        elif p_ens >= 0.60: status = f"{C['yellow']}◷ aguardar{R}"
        else:               status = f"{C['gray']}○ monitoring{R}"
        print(f"    {status}")

    # ── Parcelas ──────────────────────────────────────
    if phased is not None:
        print(f"\n  {B}Parcelas — Entrada Faseada{R}  {DIM}($5 cada){R}")
        parcel_names = ["P1 Manhã", "P2 Modelo+Mercado", "P3 Alta confiança"]
        parcel_icons = ["🌅", "⚡", "🔥"]

        for pidx in range(3):
            bought = phased.parcel_bought[pidx]
            rec    = phased.parcel_records[pidx]

            if bought and rec:
                icon = f"{C['green']}✅{R}"
                detail = f"ask {rec.get('ask',0)*100:.0f}¢  ${rec.get('size_usdc',5):.0f}"
                print(f"    {icon} {parcel_icons[pidx]} {parcel_names[pidx]:<20} {B}{detail}{R}")
            else:
                if pidx == 0:
                    fc_ok = forecast_agreement.get("valid", False) if forecast_agreement else False
                    cond = f"hora<12 + fc_agree={'✅' if fc_ok else '❌'}"
                elif pidx == 1:
                    mkt_ok, mkt_detail = phased._market_confirms_model(market, rmax)
                    cond = f"p>60% {'✅' if p >= 0.60 else '❌'} + mercado {'✅' if mkt_ok else '❌'}"
                else:
                    cond = f"p>80% {'✅' if p >= 0.80 else '❌'}"
                icon = f"{C['gray']}⬜{R}"
                print(f"    {icon} {parcel_icons[pidx]} {parcel_names[pidx]:<20} {DIM}{cond}{R}")

        print(f"    {DIM}Total: {phased.n_parcels_bought}/3  ${phased.total_invested:.0f}{R}")

    # ── Mercado ───────────────────────────────────────
    _md = market_date or berlin_date()
    print(f"\n  {B}Polymarket  ({_md}){R}")
    if not market:
        print(f"    {C['red']}✗ Mercado nao encontrado{R}")
    else:
        # Destacar highest ask
        if market.get("brackets"):
            best = max(market["brackets"], key=lambda b: b.get("ask") or b.get("price") or 0)
            best_ask = best.get("ask") or best.get("price", 0)
            print(f"    🏆 {B}Mercado escolheu:{R} {C['cyan']}{B}{best['label']}{R} (ask {best_ask*100:.0f}¢)")

        print(f"    {DIM}vol: ${market['volume']:,.0f}  brackets: {market['n_outcomes']}{R}")
        for b in market["brackets"][:8]:  # mostrar apenas 8 para não floodar
            b_ask = b.get("ask") or b.get("price") or 0
            is_t  = bracket and b["label"] == bracket["label"]
            is_best = b == best if market.get("brackets") else False
            pre = f"  {B}{C['green']}→ {R}" if is_t else "    "
            tag = " ◆ running max" if is_t else (" 🏆 highest ask" if is_best else "")
            print(f"{pre}{b['label']:<18} {b_ask*100:>5.1f}¢{tag}")

        if bracket and bracket.get("book"):
            print(f"\n  {B}Order Book — {bracket['label']}{R}")
            display_orderbook(bracket.get("book"))

    # ── EV ────────────────────────────────────────────
    if bracket and ev:
        print(f"\n  {B}Edge{R}  ask={C['red']}{B}{ev['ask']*100:.1f}¢{R}  "
              f"EV={C['green'] if ev['ev_positive'] else C['red']}{B}{ev['ev_cents']:+.1f}¢{R}  "
              f"edge={B}{ev['edge_pct']:+.1f}%{R}")

    # ── Posições ──────────────────────────────────────
    if positions is not None:
        display_positions(positions, trading_mode, usdc_balance=usdc_balance)

    # ── Bet ───────────────────────────────────────────
    if bet:
        p_idx = bet.get("parcel_idx")
        parcel_s = f"P{p_idx+1}" if p_idx is not None else ""
        border_col = C["yellow"] if trading_mode == TradingMode.PAPER else C["red"]
        print(f"\n  {border_col}{B}{'─'*44}{R}")
        print(f"  {border_col}{B}  ◆  BET {parcel_s}  ◆{R}")
        print(f"    Bracket : {bet['bracket']}")
        print(f"    Ask     : {bet['ask']*100:.1f}¢")
        print(f"    Aposta  : ${bet['bet_size']:.2f}  ({bet['shares']:.2f} shares)")
        print(f"    Max prof: +${bet['max_profit']:.2f}")
        print(f"  {border_col}{B}{'─'*44}{R}")
    elif bet_blocked_reason:
        print(f"\n  {C['yellow']}⚠  Bet bloqueada: {bet_blocked_reason}{R}")

    print(f"\n  {DIM}{'─'*58}{R}")
    print(f"  {DIM}Ctrl+C para parar{R}\n")


# ══════════════════════════════════════════════════════
#  LOGGING CSV
# ══════════════════════════════════════════════════════

def log_tick(now, temp, p, peak_detected, bracket, ev, bet,
             path, trading_mode: TradingMode = TradingMode.PAPER,
             bet_blocked_reason=None) -> None:
    row = {
        "timestamp": now.isoformat(), "mode": trading_mode.value,
        "temp": temp, "p_peak": round(p, 4), "peak_detected": peak_detected,
        "bracket": bracket["label"] if bracket else None,
        "ask": (bracket.get("ask") or bracket["price"]) if bracket else None,
        "ev_cents": ev["ev_cents"] if ev else None,
        "bet_size": bet["bet_size"] if bet else None,
        "bet_placed": bet is not None,
        "parcel_idx": bet.get("parcel_idx") if bet else None,
    }
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: writer.writeheader()
        writer.writerow(row)
