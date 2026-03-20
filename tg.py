"""
tg.py — Telegram notifier minimalista para o munich_live_bot.
Uso:
    from tg import TG
    tg = TG()   # lê TELEGRAM_TOKEN e TELEGRAM_CHAT_ID do ambiente
    tg.send("mensagem")
    tg.dashboard(...)

Variáveis de ambiente:
    TELEGRAM_TOKEN   = token do bot (obtido no @BotFather)
    TELEGRAM_CHAT_ID = o teu chat ID (obtido com @userinfobot)
"""

import os
import requests
from datetime import datetime


class TG:
    def __init__(self):
        self.token   = os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._last_p_zone = -1   # para detectar mudança de zona
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

    # ── Alertas específicos ────────────────────────────

    def alert_started(self, mode: str, bankroll: float,
                      threshold_arg: float, threshold_month: float,
                      month: int, market: dict | None, today) -> bool:
        mode_icon = "🟢" if mode == "real" else "🟡"
        if market:
            mkt = f"✅ <b>{market['title'][:40]}</b>\n  vol ${market['volume']:,.0f}  | {market['n_outcomes']} brackets"
        else:
            mkt = "⚠️ mercado não encontrado"
        thr_str = (f"{threshold_month*100:.0f}% <i>(adaptativo mês {month})</i>"
                   if abs(threshold_month - threshold_arg) > 0.01
                   else f"{threshold_arg*100:.0f}%")
        return self.send(
            f"{mode_icon} <b>Munich Bot arrancou</b>  {today}\n"
            f"  Modo: <b>{mode.upper()}</b>   Bankroll: <b>${bankroll:.2f}</b>\n"
            f"  Threshold: {thr_str}\n"
            f"  {mkt}"
        )

    def alert_no_market(self, today) -> bool:
        return self.send(
            f"⚠️ <b>Mercado não encontrado</b>  {today}\n"
            f"  O Polymarket ainda não criou o mercado de hoje.\n"
            f"  A tentar novamente a cada 10 minutos."
        )

    def alert_peak_detected(self, p: float, rmax: float, rmax_time: str,
                            bracket: dict | None) -> bool:
        bracket_str = f"  Bracket alvo: <b>{bracket['label']}</b>  ask {bracket.get('ask', bracket.get('price', 0))*100:.1f}¢" if bracket else ""
        return self.send(
            f"🔔 <b>PICO DETECTADO</b>\n"
            f"  P = <b>{p*100:.0f}%</b>   running max <b>{int(round(rmax))}°C</b> @{rmax_time}\n"
            f"{bracket_str}"
        )

    def alert_order_placed(self, bet: dict) -> bool:
        mode = "REAL 💰" if not bet.get("simulated") else "PAPER 🟡"
        # compatibilidade com ambos os nomes de campo
        ask      = bet.get("ask") or bet.get("price", 0)
        size     = bet.get("bet_size") or bet.get("size_usd", 0)
        shares   = bet.get("shares", 0)
        profit   = bet.get("max_profit", 0)
        order_id = bet.get("order_id", "?")
        return self.send(
            f"{'✅' if not bet.get('simulated') else '🟡'} <b>Ordem colocada [{mode}]</b>\n"
            f"  Bracket: <b>{bet['bracket']}</b>   Ask: {ask*100:.1f}¢\n"
            f"  ${size:.2f}  →  {shares:.2f} shares   max +${profit:.2f}\n"
            f"  ID: <code>{order_id}</code>"
        )

    def alert_order_failed(self, error: str, bracket: dict | None) -> bool:
        bracket_str = bracket['label'] if bracket else "?"
        return self.send(
            f"❌ <b>Ordem REAL falhou</b>\n"
            f"  Bracket: {bracket_str}\n"
            f"  Erro: <code>{error[:200]}</code>"
        )

    def alert_stopped(self, bets: list, mode: str) -> bool:
        mode_label = "REAL" if mode == "real" else "PAPER"
        if bets:
            return self.send(
                f"🛑 <b>Bot parado</b>  [{mode_label}]\n"
                f"  {len(bets)} ordem(s) registada(s) hoje."
            )
        return self.send(f"🛑 <b>Bot parado</b>  [{mode_label}]  sem ordens hoje.")

    def alert_zone_change(self, p: float, zone: int) -> bool:
        icons = {0: "⚪", 1: "🟠", 2: "🟡", 3: "🟢"}
        labels = {0: "< 30%", 1: "30–60%", 2: "60–80%", 3: "≥ 80%"}
        return self.send(
            f"{icons[zone]} <b>P(pico) mudou de zona</b>\n"
            f"  Agora: <b>{p*100:.0f}%</b>  ({labels[zone]})"
        )

    # ── Dashboard periódica ────────────────────────────

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
                  clob_mode: str,
                  reason: str = "periodic") -> bool:
        """
        Envia dashboard formatada para Telegram.
        reason: "periodic" | "zone_change" | "market_open"
        """
        now_str  = datetime.now().strftime("%H:%M")
        mode_icon = "🟢" if clob_mode == "real" else "🟡"

        lines = [
            f"{mode_icon} <b>Munich Bot</b>  {today}  {now_str}",
            "",
        ]

        # Temperatura
        temp_str = f"{int(round(temp_now))}°C" if temp_now is not None else "—"
        fc_str   = f"   prev {forecast_max['temp_max']}°C" if forecast_max else ""
        lines += [
            f"🌡 <b>Temperatura</b>",
            f"  Agora: <b>{temp_str}</b>   Max: <b>{int(round(rmax))}°C</b> @{rmax_time}{fc_str}",
            "",
        ]

        # Modelo
        p_bar = _tg_bar(p, width=10)
        p_icon = "🟢" if p >= 0.80 else ("🟡" if p >= 0.60 else ("🟠" if p >= 0.30 else "⚪"))
        peak_str = "  ✅ <b>PICO DETECTADO</b>" if peak_detected else ""
        lines += [
            f"🧠 <b>P(pico já ocorreu)</b>",
            f"  {p_icon} {p_bar} <b>{p*100:.0f}%</b>{peak_str}",
            "",
        ]

        # Mercado
        if not market:
            lines += ["📋 <b>Mercado</b>: ainda não abriu", ""]
        else:
            lines += [
                f"📋 <b>{market['title'][:40]}</b>",
                f"  vol ${market['volume']:,.0f}  |  {market['n_outcomes']} brackets",
                "",
            ]
            lines.append("<pre>")
            lines.append(f"{'Bracket':<16} {'Ask':>5}  {'Bar':8}")
            lines.append("─" * 32)
            for b in market["brackets"]:
                arrow   = "→" if bracket and b["label"] == bracket["label"] else " "
                ask_val = b.get("ask") or b.get("price") or 0
                bar     = _tg_bar(ask_val, width=8)
                lines.append(f"{arrow}{b['label']:<15} {ask_val*100:>4.0f}¢  {bar}")
            lines.append("</pre>")

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
            sim_label = "PAPER" if bet.get("simulated") else "REAL"
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

        reason_str = {
            "periodic":    "⏱ periódico",
            "zone_change": "⚡ mudança de zona",
            "market_open": "📋 mercado abriu",
        }.get(reason, reason)
        lines += ["", f"<i>{reason_str}</i>"]

        return self.send("\n".join(lines))

    # ── Detecção de zona ──────────────────────────────

    def p_zone(self, p: float) -> int:
        """Zona: 0=<30%, 1=30-60%, 2=60-80%, 3=>=80%"""
        if p >= 0.80: return 3
        if p >= 0.60: return 2
        if p >= 0.30: return 1
        return 0

    def zone_changed(self, p: float) -> bool:
        """True se P mudou de zona desde o último check."""
        z = self.p_zone(p)
        if z != self._last_p_zone:
            self._last_p_zone = z
            return True
        return False


def _tg_bar(p: float, width: int = 10) -> str:
    filled = round(min(max(p, 0), 1) * width)
    return "█" * filled + "░" * (width - filled)
