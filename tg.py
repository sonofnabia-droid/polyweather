"""
tg.py — UNIFIED
==================
Telegram notifier para o munich_live_bot.
Suporta: ensemble, dual forecast, phased entry, open positions.

Variáveis de ambiente:
    TELEGRAM_TOKEN=...
    TELEGRAM_CHAT_ID=...
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
            print("  [TG] TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID nao definidos "
                  "- notificacoes desactivadas")

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = requests.post(
                "https://api.telegram.org/bot" + self.token + "/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "HTML"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    # ══════════════════════════════════════════════════════
    #  ALERTAS
    # ══════════════════════════════════════════════════════

    def alert_started(self, mode, bankroll, threshold_arg,
                      threshold_month, month, market, today):
        mode_icon = "🟢" if mode == "real" else "🟡"
        if market:
            mkt = ("✅ <b>" + market['title'][:40] + "</b>\n"
                   "  vol $" + f"{market['volume']:,.0f}" + "  | " +
                   str(market['n_outcomes']) + " brackets")
        else:
            mkt = "⚠️ mercado nao encontrado"
        if abs(threshold_month - threshold_arg) > 0.01:
            thr_str = f"{threshold_month*100:.0f}% <i>(adaptativo mes {month})</i>"
        else:
            thr_str = f"{threshold_arg*100:.0f}%"
        lines = [
            mode_icon + " <b>Munich Bot arrancou</b>  " + str(today),
            "  Modo: <b>" + mode.upper() + "</b>   Bankroll: <b>$" + f"{bankroll:.2f}" + "</b>",
            "  Threshold: " + thr_str,
            "  " + mkt,
        ]
        return self.send("\n".join(lines))

    def alert_no_market(self, today):
        return self.send(
            "⚠️ <b>Mercado nao encontrado</b>  " + str(today) +
            "\n  O Polymarket ainda nao criou o mercado de hoje." +
            "\n  A tentar novamente a cada 10 minutos."
        )

    def alert_peak_detected(self, p, rmax, rmax_time,
                            bracket=None, ensemble_result=None,
                            market=None):
        """
        Alerta de pico detectado.
        Retrocompativel: bracket era posicional, agora keyword.
        """
        # Ensemble breakdown
        if ensemble_result:
            p_lgbm = ensemble_result.get("p_lgbm", 0)
            p_xgb  = ensemble_result.get("p_xgb")
            p_zs   = ensemble_result.get("p_zscore")
            p_used = ensemble_result.get("p_ensemble", p)

            lgbm_str = f"{p_lgbm*100:.1f}%"
            xgb_str  = f"{p_xgb*100:.1f}%" if p_xgb is not None else "N/A"
            zs_str   = f"{p_zs*100:.1f}%" if p_zs is not None else "N/A"

            parts = [
                "  🧠 <b>Ensemble</b>: " + f"{p_used*100:.1f}%",
                "  🟦 LGBM: " + lgbm_str,
                "  🟧 XGB:  " + xgb_str,
                "  🟪 Z-Score: " + zs_str,
            ]
            models_block = "\n".join(parts)
        else:
            models_block = "  P = <b>" + f"{p*100:.0f}" + "%</b>"

        # Bracket alvo
        bracket_str = ""
        if bracket:
            ask = bracket.get("ask") or bracket.get("price", 0)
            bracket_str = ("\n  🎯 Bracket: <b>" + bracket['label'] +
                          "</b>  ask " + f"{ask*100:.1f}" + "¢")

        # Mercado highest ask
        market_str = ""
        if market and market.get("brackets"):
            best = max(market["brackets"],
                       key=lambda b: b.get("ask") or b.get("price") or 0)
            best_ask = best.get("ask") or best.get("price", 0)
            market_str = ("\n  🏆 Mercado escolheu: <b>" + best['label'] +
                          "</b> (ask " + f"{best_ask*100:.0f}" + "¢)")

        lines = [
            "🔔 <b>PICO DETECTADO</b>",
            models_block,
            "  🌡 Running max: <b>" + str(int(round(rmax))) +
                "°C</b> @" + rmax_time,
            market_str,
            bracket_str,
        ]
        # Filtrar linhas vazias
        return self.send("\n".join(line for line in lines if line))

    def alert_order_placed(self, bet, clob_mode="paper"):
        simulated = bet.get("simulated", clob_mode != "real")
        mode = "PAPER 🟡" if simulated else "REAL 💰"
        icon = "🟡" if simulated else "✅"

        ask      = bet.get("ask") or bet.get("price", 0)
        size     = bet.get("bet_size") or bet.get("size_usdc", 0)
        shares   = bet.get("shares", 0)
        profit   = bet.get("max_profit", 0)
        order_id = bet.get("order_id", "?")
        p_idx    = bet.get("parcel_idx")
        parcel_s = "P" + str(p_idx + 1) + " " if p_idx is not None else ""

        lines = [
            icon + " <b>Ordem " + parcel_s + "colocada [" + mode + "]</b>",
            "  Bracket: <b>" + bet['bracket'] + "</b>   Ask: " +
                f"{ask*100:.1f}" + "¢",
            "  $" + f"{size:.2f}" + "  →  " + f"{shares:.2f}" +
                " shares   max +$" + f"{profit:.2f}",
            "  ID: <code>" + str(order_id) + "</code>",
        ]
        return self.send("\n".join(lines))

    def alert_order_failed(self, error, bracket=None):
        bracket_str = bracket["label"] if bracket else "?"
        lines = [
            "❌ <b>Ordem REAL falhou</b>",
            "  Bracket: " + bracket_str,
            "  Erro: <code>" + str(error)[:200] + "</code>",
        ]
        return self.send("\n".join(lines))

    def alert_bet_blocked(self, reason, p_ensemble=0.0):
        """Alerta quando uma bet e bloqueada."""
        lines = [
            "🚫 <b>Bet bloqueada</b>",
            "  Motivo: <code>" + str(reason)[:200] + "</code>",
        ]
        if p_ensemble > 0:
            lines.append("  P(ensemble): " + f"{p_ensemble*100:.1f}" + "%")
        return self.send("\n".join(lines))

    def alert_position_resolved(self, pos):
        won      = pos.status.value == "won"
        icon     = "🏆" if won else "💸"
        result   = "GANHOU" if won else "PERDEU"
        pnl_s    = f"{pos.pnl_usd:+.2f}" if pos.pnl_usd is not None else "?"
        pnl_p    = f"{pos.pnl_pct:+.1f}%" if pos.pnl_pct is not None else "?"
        entry_s  = f"{pos.entry_ask*100:.1f}" + "¢"
        lines = [
            icon + " <b>Posicao resolvida — " + pos.date_opened + "</b>",
            "  Bracket: <b>" + pos.bracket_label + "</b>",
            "  Entrada: " + entry_s + "   Shares: " + f"{pos.shares:.2f}",
            "  Resultado: <b>" + result + "</b>",
            "  P&amp;L: <b>" + pnl_s + "</b> (" + pnl_p + ")",
        ]
        return self.send("\n".join(lines))

    def alert_day_summary(self, day_str, day_positions,
                          cumulative_summary):
        n_bets   = len(day_positions)
        n_won    = sum(1 for p in day_positions if p.status.value == "won")
        n_lost   = sum(1 for p in day_positions if p.status.value == "lost")
        invested = sum(p.size_usdc for p in day_positions)
        day_pnl  = sum(p.pnl_usd for p in day_positions
                       if p.pnl_usd is not None)
        roi = (day_pnl / invested * 100) if invested > 0 else 0.0

        if n_bets == 0:
            lines = [
                "📅 <b>Fim do dia — " + day_str + "</b>",
                "  Sem bets hoje.",
            ]
        else:
            pnl_icon = "📈" if day_pnl >= 0 else "📉"
            lines = [
                "📅 <b>Fim do dia — " + day_str + "</b>",
                "  Bets: " + str(n_bets) + "   ✅ " + str(n_won) +
                    "  ❌ " + str(n_lost),
                "  Investido: $" + f"{invested:.2f}",
                "  " + pnl_icon + " P&amp;L dia: <b>" +
                    f"{day_pnl:+.2f}" + "</b> (" + f"{roi:+.1f}" + "%)",
                "",
            ]
            for pos in day_positions:
                if pos.status.value == "won":
                    st_icon = "✅"
                elif pos.status.value == "lost":
                    st_icon = "❌"
                else:
                    st_icon = "⏳"
                pnl_s = (f"{pos.pnl_usd:+.2f}"
                         if pos.pnl_usd is not None else "?")
                lines.append(
                    "  " + st_icon + " " + pos.bracket_label[:16] + "  " +
                    f"{pos.entry_ask*100:.0f}" + "¢  " + pnl_s
                )
            lines.append("")

        s  = cumulative_summary
        nc = s["n_won"] + s["n_lost"]
        wr_s = f"{s['n_won']/nc*100:.0f}%" if nc > 0 else "—"
        lines += [
            "─────────────────",
            "  Acumulado: " + str(s['n_won']) + "W / " +
                str(s['n_lost']) + "L   Win rate: <b>" + wr_s + "</b>",
            "  Investido total: $" + f"{s['total_invested']:.2f}",
            "  P&amp;L total: <b>" + f"{s['total_pnl_usd']:+.2f}" +
                "</b> (" + f"{s['total_pnl_pct']:+.1f}" + "%)",
        ]
        return self.send("\n".join(lines))

    def alert_zone_change(self, p, zone):
        icons  = {0: "⚪", 1: "🟠", 2: "🟡", 3: "🟢"}
        labels = {0: "< 30%", 1: "30-60%", 2: "60-80%", 3: ">= 80%"}
        lines = [
            icons.get(zone, "⚪") + " <b>P(pico) mudou de zona</b>",
            "  Agora: <b>" + f"{p*100:.0f}" + "%</b>  (" +
                labels.get(zone, "?") + ")",
        ]
        return self.send("\n".join(lines))

    # ══════════════════════════════════════════════════════
    #  DETECÇÃO DE ZONA
    # ══════════════════════════════════════════════════════

    def p_zone(self, p):
        if p >= 0.80: return 3
        if p >= 0.60: return 2
        if p >= 0.30: return 1
        return 0

    def zone_changed(self, p):
        z = self.p_zone(p)
        if z != self._last_p_zone:
            self._last_p_zone = z
            return True
        return False

    # ══════════════════════════════════════════════════════
    #  DASHBOARD COMPLETA (periodica 30 em 30 min)
    # ══════════════════════════════════════════════════════

    def dashboard(self, today, p, rmax, rmax_time,
                  temp_now=None, forecast_max=None,
                  market=None, bracket=None, ev=None,
                  peak_detected=False, bet=None,
                  clob_mode=None, trading_mode=None,
                  chart=None, reason="periodic",
                  positions_summary=None,
                  om_forecast=None,
                  forecast_agreement=None,
                  ensemble_result=None,
                  phased=None,
                  usdc_balance=None,
                  bet_blocked_reason=None):

        # Modo robusto
        mode = trading_mode or clob_mode or "paper"
        if hasattr(mode, "value"):
            mode_str = mode.value.upper()
        else:
            mode_str = str(mode).replace("TradingMode.", "").upper()

        mode_icon = "🟢" if mode_str == "REAL" else "🟡"
        now_str = datetime.now().strftime("%H:%M")

        lines = [
            mode_icon + " <b>Munich Max Temp - Live Bot</b>  " +
            "[" + mode_str + "]  " + str(today) + "  " + now_str,
            "  ─────────────────────────────────────────",
            "",
        ]

        # ── Saldo ──────────────────────────────────────
        if usdc_balance is not None:
            bal_icon = "💵" if usdc_balance >= 10 else "⚠️"
            lines.append(
                bal_icon + " <b>Saldo:</b> $" + f"{usdc_balance:,.2f}" + " USDC"
            )
            lines.append("")

        # ── Chart ASCII ────────────────────────────────
        if chart:
            lines.append("🌡 <b>Curva de temperatura hoje</b>")
            lines.extend(chart)
            lines.append("")

        # ── Temperatura ────────────────────────────────
        temp_str = str(int(round(temp_now))) + "°C" if temp_now is not None else "—"
        fc_str = ""
        if forecast_max:
            fc_str = "   prev WU " + str(forecast_max['temp_max']) + "°C"
        lines += [
            "🌡 <b>Temperatura actual</b>",
            "  Agora: <b>" + temp_str + "</b>   " +
            "Max: <b>" + str(int(round(rmax))) + "°C</b> @" +
            rmax_time + fc_str,
            "",
        ]

        # ── Dual Forecast ──────────────────────────────
        if om_forecast:
            lines.append("🌤 <b>Previsao Dual</b>")
            if forecast_max:
                lines.append(
                    "  🟦 WU: max <b>" + str(forecast_max['temp_max']) + "°C</b>"
                )
            lines.append(
                "  🟣 OM: max <b>" + str(om_forecast['temp_max']) + "°C</b>"
            )
            if forecast_agreement:
                if forecast_agreement.get("valid"):
                    diff = forecast_agreement.get("diff", "?")
                    cons = forecast_agreement.get("consensus_max", "?")
                    lines.append(
                        "  ✅ Concordam (diff " + str(diff) +
                        "°C) consenso=" + str(cons) + "°C"
                    )
                else:
                    reason_fc = forecast_agreement.get("reason", "?")
                    lines.append("  ❌ Discordam (" + str(reason_fc) + ")")
            lines.append("")

        # ── Ensemble ───────────────────────────────────
        if ensemble_result:
            p_ens = ensemble_result["p_ensemble"]
            peak_str = "  ✓ <b>PICO DETECTADO</b>" if peak_detected else ""
            p_bar = _tg_bar(p_ens, width=10)
            lines.append("🧠 <b>Ensemble - P(pico)</b>")
            lines.append(
                "  " + p_bar + "  <b>" + f"{p_ens*100:.1f}" + "%</b>" + peak_str
            )
            # Componentes
            comp = []
            p_lgbm = ensemble_result.get("p_lgbm")
            p_xgb  = ensemble_result.get("p_xgb")
            p_zs   = ensemble_result.get("p_zscore")
            if p_lgbm is not None:
                comp.append("LGBM " + f"{p_lgbm*100:.0f}" + "%")
            if p_xgb is not None:
                comp.append("XGB " + f"{p_xgb*100:.0f}" + "%")
            if p_zs is not None:
                comp.append("Z " + f"{p_zs*100:.0f}" + "%")
            if comp:
                lines.append("  " + " | ".join(comp))
            lines.append("")
        else:
            # Fallback: modelo unico
            p_bar = _tg_bar(p, width=10)
            peak_str = "  ✓ <b>PICO DETECTADO</b>" if peak_detected else ""
            lines.append("🧠 <b>Modelo - P(pico)</b>")
            lines.append(
                "  " + p_bar + "  <b>" + f"{p*100:.1f}" + "%</b>" + peak_str
            )
            lines.append("")

        # ── Phased Entry ───────────────────────────────
        if phased is not None:
            is_single = (
                hasattr(phased, 'bought')
                and not hasattr(phased, 'parcel_bought')
            )
            if is_single:
                lines.append("🎯 <b>Entrada - SINGLE</b>")
                if phased.bought:
                    lines.append(
                        "  $" + f"{phased.parcel_size:.0f}" + "  ✅ comprado"
                    )
                else:
                    lines.append(
                        "  $" + f"{phased.parcel_size:.0f}" + "  ⏳ aguardar"
                    )
                lines.append("")
            else:
                p_icons_done = ["✅🌅", "✅⚡", "✅🔥"]
                p_icons_wait = ["⬜🌅", "⬜⚡", "⬜🔥"]
                parts = []
                for i in range(3):
                    if phased.parcel_bought[i]:
                        parts.append(p_icons_done[i])
                    else:
                        parts.append(p_icons_wait[i])
                total_inv = phased.total_invested
                total_max = phased.parcel_size * 3
                lines.append("🎯 <b>Parcelas</b>")
                lines.append(
                    "  " + " ".join(parts) + "  " +
                    "(" + str(phased.n_parcels_bought) + "/3  " +
                    "$" + f"{total_inv:.0f}" + "/$" + f"{total_max:.0f}" + ")"
                )
                lines.append("")

        # ── Bet Blocked ────────────────────────────────
        if bet_blocked_reason:
            lines.append(
                "🚫 <b>Bet bloqueada:</b> <i>" +
                str(bet_blocked_reason)[:100] + "</i>"
            )
            lines.append("")

        # ── Mercado ────────────────────────────────────
        if not market:
            lines.append("📋 <b>Mercado</b>: ainda nao abriu")
            lines.append("")
        else:
            # Highest ask
            if market.get("brackets"):
                best = max(
                    market["brackets"],
                    key=lambda b: b.get("ask") or b.get("price") or 0,
                )
                best_ask = best.get("ask") or best.get("price", 0)
                lines.append("📋 <b>Polymarket</b>")
                lines.append(
                    "  🏆 Highest ask: <b>" + best['label'] + "</b> " +
                    "(" + f"{best_ask*100:.0f}" + "¢)"
                )
                lines.append("")

            # Bracket tabela
            if bracket and market.get("brackets"):
                bks = market["brackets"][:8]
                lines.append("<pre>")
                lines.append(
                    "Bracket          Ask    Bar"
                )
                lines.append("─" * 32)
                for b in bks:
                    arrow = "→" if b["label"] == bracket["label"] else " "
                    ask_val = b.get("ask") or b.get("price") or 0
                    bar = _tg_bar(ask_val, width=8)
                    label_padded = b['label'][:15]
                    lines.append(
                        arrow + label_padded + " " +
                        f"{ask_val*100:>4.0f}" + "¢  " + bar
                    )
                lines.append("</pre>")
                lines.append("")

        # ── EV ──────────────────────────────────────────
        if bracket and ev:
            ev_icon = "✅" if ev["ev_positive"] else "❌"
            ask_val = ev.get("ask", bracket.get("ask", bracket.get("price", 0)))
            lines.append(
                "📊 <b>Edge</b>  [" + bracket['label'] + "]"
            )
            lines.append(
                "  " + ev_icon + " ask " + f"{ask_val*100:.1f}" +
                "¢   EV " + f"{ev['ev_cents']:+.1f}" +
                "¢   edge " + f"{ev['edge_pct']:+.1f}" + "%"
            )
            lines.append("")

        # ── Bet ────────────────────────────────────────
        if bet:
            simulated = bet.get("simulated", mode_str != "REAL")
            sim_label = "PAPER" if simulated else "REAL"
            p_idx = bet.get("parcel_idx")
            parcel_s = "P" + str(p_idx + 1) + " " if p_idx is not None else ""
            ask  = bet.get("ask") or bet.get("price", 0)
            size = bet.get("bet_size") or bet.get("size_usdc", 0)
            lines.append(
                "💰 <b>Bet " + parcel_s + "[" + sim_label + "]</b>"
            )
            lines.append(
                "  " + bet['bracket'] + "  ask " +
                f"{ask*100:.1f}" + "¢"
            )
            lines.append(
                "  $" + f"{size:.2f}" + "  →  " +
                f"{bet.get('shares', 0):.2f}" + " shares   max +$" +
                f"{bet.get('max_profit', 0):.2f}"
            )
        else:
            if not bet_blocked_reason:
                lines.append("💤 Sem bet ainda")

        # ── P&L acumulado ──────────────────────────────
        if positions_summary:
            s  = positions_summary
            nc = s["n_won"] + s["n_lost"]
            if nc > 0:
                wr = s["n_won"] / nc * 100
                wr_icon = "📈" if s["total_pnl_usd"] >= 0 else "📉"
                lines.append("")
                lines.append(wr_icon + " <b>Resultados acumulados</b>")
                lines.append(
                    "  " + str(s['n_won']) + "W / " + str(s['n_lost']) +
                    "L   win rate " + f"{wr:.0f}" + "%"
                )
                lines.append(
                    "  P&amp;L: <b>" + f"{s['total_pnl_usd']:+.2f}" +
                    "</b> (" + f"{s['total_pnl_pct']:+.1f}" + "%)"
                )
                if s.get("n_open", 0) > 0:
                    lines.append(
                        "  Posicoes abertas: " + str(s['n_open'])
                    )

        # Rodape
        reason_map = {
            "periodic":    "⏱ periodico (30m)",
            "zone_change": "⚡ mudanca de zona",
            "market_open": "📋 mercado abriu",
        }
        reason_str = reason_map.get(reason, reason)
        lines.append("")
        lines.append("<i>" + reason_str + "</i>")

        return self.send("\n".join(lines))


# ══════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════

def _tg_bar(p, width=10):
    p_clamped = min(max(p, 0), 1)
    filled = round(p_clamped * width)
    return "█" * filled + "░" * (width - filled)
