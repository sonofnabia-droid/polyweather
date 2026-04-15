"""
tg.py — Telegram notifier para o munich_live_bot. BRANCH: main
"""

import os
import requests
from datetime import datetime


class TG:
    def __init__(self):
        self.token   = os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._last_p_zone = -1
        if not self.enabled:
            print("  [TG] TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID nao definidos — notificacoes desactivadas")

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────
    # ALERTAS
    # ─────────────────────────────────────────────────────────────

    def alert_started(self, mode: str, bankroll: float,
                      threshold_arg: float, threshold_month: float,
                      month: int, market: dict | None, today) -> bool:
        mode_icon = "🟢" if mode == "real" else "🟡"
        if market:
            mkt = (f"✅ <b>{market['title'][:40]}</b>\n"
                   f"  vol ${market['volume']:,.0f}  | {market['n_outcomes']} brackets")
        else:
            mkt = "⚠️ mercado não encontrado"
        thr_str = (f"{threshold_month*100:.0f}% <i>(adaptativo mês {month})</i>"
                   if abs(threshold_month - threshold_arg) > 0.01
                   else f"{threshold_arg*100:.0f}%")
        lines = [
            f"{mode_icon} <b>Munich Bot arrancou</b>  {today}",
            f"  Modo: <b>{mode.upper()}</b>   Bankroll: <b>${bankroll:.2f}</b>",
            f"  Threshold: {thr_str}",
            f"  {mkt}",
        ]
        return self.send("\n".join(lines))

    def alert_no_market(self, today) -> bool:
        lines = [
            f"⚠️ <b>Mercado não encontrado</b>  {today}",
            "  O Polymarket ainda não criou o mercado de hoje.",
            "  A tentar novamente a cada 10 minutos.",
        ]
        return self.send("\n".join(lines))

    def alert_peak_detected(self, p: float, rmax: float, rmax_time: str,
                            bracket: dict | None) -> bool:
        bracket_str = ""
        if bracket:
            ask = bracket.get("ask") or bracket.get("price", 0)
            bracket_str = f"\n  Bracket alvo: <b>{bracket['label']}</b>  ask {ask*100:.1f}¢"
        lines = [
            "🔔 <b>PICO DETECTADO</b>",
            f"  P = <b>{p*100:.0f}%</b>   running max <b>{int(round(rmax))}°C</b> @{rmax_time}{bracket_str}",
        ]
        return self.send("\n".join(lines))

    def alert_order_placed(self, bet: dict, clob_mode: str = "paper") -> bool:
        simulated = bet.get("simulated", clob_mode != "real")
        mode = "PAPER 🟡" if simulated else "REAL 💰"
        icon = "🟡" if simulated else "✅"

        ask      = bet.get("ask") or bet.get("price", 0)
        size     = bet.get("bet_size") or bet.get("size_usd", 0)
        shares   = bet.get("shares", 0)
        profit   = bet.get("max_profit", 0)
        order_id = bet.get("order_id", "?")

        lines = [
            f"{icon} <b>Ordem colocada [{mode}]</b>",
            f"  Bracket: <b>{bet['bracket']}</b>   Ask: {ask*100:.1f}¢",
            f"  ${size:.2f}  →  {shares:.2f} shares   max +${profit:.2f}",
            f"  ID: <code>{order_id}</code>",
        ]
        return self.send("\n".join(lines))

    def alert_order_failed(self, error: str, bracket: dict | None) -> bool:
        bracket_str = bracket["label"] if bracket else "?"
        lines = [
            "❌ <b>Ordem REAL falhou</b>",
            f"  Bracket: {bracket_str}",
            f"  Erro: <code>{error[:200]}</code>",
        ]
        return self.send("\n".join(lines))

    def alert_position_resolved(self, pos) -> bool:
        won      = pos.status.value == "won"
        icon     = "🏆" if won else "💸"
        result   = "GANHOU" if won else "PERDEU"
        pnl_s    = f"{pos.pnl_usd:+.2f}" if pos.pnl_usd is not None else "?"
        pnl_p    = f"{pos.pnl_pct:+.1f}%" if pos.pnl_pct is not None else "?"
        entry_s  = f"{pos.entry_ask*100:.1f}¢"
        lines = [
            f"{icon} <b>Posição resolvida — {pos.date_opened}</b>",
            f"  Bracket: <b>{pos.bracket_label}</b>",
            f"  Entrada: {entry_s}   Shares: {pos.shares:.2f}",
            f"  Resultado: <b>{result}</b>",
            f"  P&amp;L: <b>{pnl_s}</b> ({pnl_p})",
        ]
        return self.send("\n".join(lines))

    def alert_day_summary(self, day_str: str, day_positions: list,
                          cumulative_summary: dict) -> bool:
        n_bets  = len(day_positions)
        n_won   = sum(1 for p in day_positions if p.status.value == "won")
        n_lost  = sum(1 for p in day_positions if p.status.value == "lost")
        invested = sum(p.size_usdc for p in day_positions)
        day_pnl  = sum(p.pnl_usd for p in day_positions if p.pnl_usd is not None)
        roi      = (day_pnl / invested * 100) if invested > 0 else 0.0

        if n_bets == 0:
            lines = [
                f"📅 <b>Fim do dia — {day_str}</b>",
                "  Sem bets hoje.",
            ]
        else:
            pnl_icon = "📈" if day_pnl >= 0 else "📉"
            lines = [
                f"📅 <b>Fim do dia — {day_str}</b>",
                f"  Bets: {n_bets}   ✅ {n_won}  ❌ {n_lost}",
                f"  Investido: ${invested:.2f}",
                f"  {pnl_icon} P&amp;L dia: <b>{day_pnl:+.2f}</b> ({roi:+.1f}%)",
                "",
            ]
            for pos in day_positions:
                st_icon = "✅" if pos.status.value == "won" else ("❌" if pos.status.value == "lost" else "⏳")
                pnl_s = f"{pos.pnl_usd:+.2f}" if pos.pnl_usd is not None else "?"
                lines.append(
                    f"  {st_icon} {pos.bracket_label:<16} {pos.entry_ask*100:.0f}¢  {pnl_s}"
                )
            lines.append("")

        s = cumulative_summary
        nc = s["n_won"] + s["n_lost"]
        wr_s = f"{s['n_won']/nc*100:.0f}%" if nc > 0 else "—"
        lines += [
            "─────────────────",
            f"  Acumulado: {s['n_won']}W / {s['n_lost']}L   Win rate: <b>{wr_s}</b>",
            f"  Investido total: ${s['total_invested']:.2f}",
            f"  P&amp;L total: <b>{s['total_pnl_usd']:+.2f}</b> ({s['total_pnl_pct']:+.1f}%)",
        ]
        return self.send("\n".join(lines))

    def alert_zone_change(self, p: float, zone: int) -> bool:
        icons  = {0: "⚪", 1: "🟠", 2: "🟡", 3: "🟢"}
        labels = {0: "< 30%", 1: "30–60%", 2: "60–80%", 3: "≥ 80%"}
        lines = [
            f"{icons[zone]} <b>P(pico) mudou de zona</b>",
            f"  Agora: <b>{p*100:.0f}%</b>  ({labels[zone]})",
        ]
        return self.send("\n".join(lines))

    # ─────────────────────────────────────────────────────────────
    # DETECÇÃO DE ZONA
    # ─────────────────────────────────────────────────────────────

    def p_zone(self, p: float) -> int:
        if p >= 0.80: return 3
        if p >= 0.60: return 2
        if p >= 0.30: return 1
        return 0

    def zone_changed(self, p: float) -> bool:
        z = self.p_zone(p)
        if z != self._last_p_zone:
            self._last_p_zone = z
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # DASHBOARD COMPLETA
    # ─────────────────────────────────────────────────────────────

    def dashboard(self,
                  today,
                  p: float,
                  rmax: float,
                  rmax_time: str,
                  temp_now: float | None,
                  forecast_max: dict | None,
                  market: dict | None,
                  bracket: dict | None,
                  ev: dict | None,
                  peak_detected: bool,
                  bet: dict | None,
                  clob_mode: str = None,
                  trading_mode: str = None,
                  chart: list | None = None,
                  reason: str = "periodic",
                  positions_summary: dict | None = None) -> bool:

        # Converter modo para string robusta
        mode = trading_mode or clob_mode or "paper"
        if hasattr(mode, "value"):
            mode_str = mode.value.upper()
        else:
            mode_str = str(mode).replace("TradingMode.", "").upper()

        mode_icon = "🟢" if mode_str == "REAL" else "🟡"
        now_str = datetime.now().strftime("%H:%M")

        lines = [
            f"{mode_icon} <b>Munich Max Temp — Live Bot</b>  [{mode_str}]  {today}  {now_str}  │  Munich (CET/CEST) {now_str}",
            f"  Estação: <b>{forecast_max['station'] if forecast_max and 'station' in forecast_max else 'EDDM Munich Airport (WUnderground)'}</b>",
            f"  ◉ a verificar sinal (:{now_str[-2:]})",
            "  ──────────────────────────────────────────────────────────",
            "",
        ]

        # Chart ASCII
        if chart:
            lines.append("🌡 <b>Curva de temperatura hoje</b>")
            lines.append("<pre>")
            lines.extend(chart)
            lines.append("</pre>")
            lines.append("")

        # Temperatura
        temp_str = f"{int(round(temp_now))}°C" if temp_now is not None else "—"
        fc_str   = f"   prev {forecast_max['temp_max']}°C" if forecast_max else ""
        lines += [
            "🌡 <b>Temperatura actual</b>",
            f"  Agora: <b>{temp_str}</b>   Max: <b>{int(round(rmax))}°C</b> @{rmax_time}{fc_str}",
            "",
        ]

        # Modelo
        p_bar  = _tg_bar(p, width=10)
        peak_str = "  ✓ <b>PICO DETECTADO</b>" if peak_detected else ""
        lines += [
            "🧠 <b>Modelo LightGBM — P(pico já ocorreu)</b>",
            f"  {p_bar}  <b>{p*100:.1f}%</b>{peak_str}",
            "",
        ]

        # Mercado
        if not market:
            lines += ["📋 <b>Mercado</b>: ainda não abriu", ""]
        else:
            lines += [
                "📋 <b>Polymarket — Mercado de Hoje</b>",
                f"  {market['title'][:60]}",
                f"  vol: ${market['volume']:,.0f}  brackets: {market['n_outcomes']}",
                "",
                "<pre>",
                f"{'Bracket':<16} {'Ask':>5}  {'Bar':8}",
                "─" * 32,
            ]
            for b in market["brackets"]:
                arrow   = "→" if bracket and b["label"] == bracket["label"] else " "
                ask_val = b.get("ask") or b.get("price") or 0
                bar     = _tg_bar(ask_val, width=8)
                lines.append(f"{arrow}{b['label']:<15} {ask_val*100:>4.0f}¢  {bar}")
            lines.append("</pre>")
            lines.append("")

        # EV
        if bracket and ev:
            ev_icon = "✅" if ev["ev_positive"] else "❌"
            ask_val = ev.get("ask", bracket.get("ask", bracket.get("price", 0)))
            lines += [
                f"📊 <b>Edge</b>  [{bracket['label']}]",
                f"  {ev_icon} ask {ask_val*100:.1f}¢   EV {ev['ev_cents']:+.1f}¢   edge {ev['edge_pct']:+.1f}%",
                "",
            ]

        # Bet
        if bet:
            simulated = bet.get("simulated", mode_str != "REAL")
            sim_label = "PAPER" if simulated else "REAL"
            ask       = bet.get("ask") or bet.get("price", 0)
            size      = bet.get("bet_size") or bet.get("size_usd", 0)
            lines += [
                f"💰 <b>Bet {sim_label}</b>",
                f"  {bet['bracket']}  ask {ask*100:.1f}¢",
                f"  ${size:.2f}  →  {bet.get('shares', 0):.2f} shares   max +${bet.get('max_profit', 0):.2f}",
                f"  ID: <code>{bet.get('order_id', '?')}</code>",
            ]
        else:
            lines.append("💤 Sem bet ainda")

        # P&L acumulado
        if positions_summary and (positions_summary["n_won"] + positions_summary["n_lost"]) > 0:
            s  = positions_summary
            nc = s["n_won"] + s["n_lost"]
            wr = s["n_won"] / nc * 100
            wr_icon = "📈" if s["total_pnl_usd"] >= 0 else "📉"
            lines += [
                "",
                f"{wr_icon} <b>Resultados acumulados</b>",
                f"  {s['n_won']}W / {s['n_lost']}L   win rate {wr:.0f}%",
                f"  P&amp;L: <b>{s['total_pnl_usd']:+.2f}</b> ({s['total_pnl_pct']:+.1f}%)",
            ]
            if s["n_open"] > 0:
                lines.append(f"  Posições abertas: {s['n_open']}")

        # Rodapé
        reason_str = {
            "periodic":    "⏱ periódico",
            "zone_change": "⚡ mudança de zona",
            "market_open": "📋 mercado abriu",
        }.get(reason, reason)
        lines += ["", f"<i>{reason_str}</i>"]

        return self.send("\n".join(lines))

# ─────────────────────────────────────────────────────────────
# FUNÇÃO GLOBAL
# ─────────────────────────────────────────────────────────────
    
    
def _tg_bar(p: float, width: int = 10) -> str:
    filled = round(min(max(p, 0), 1) * width)
    return "█" * filled + "░" * (width - filled)
