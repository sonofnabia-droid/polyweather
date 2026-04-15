"""
tg.py — Telegram notifier para o munich_live_bot V3.  # BRANCH - INTEGRATION
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
            f"{mode_icon} <b>Munich Bot V3 arrancou</b>  {today}",
            f"  Modo: <b>{mode.upper()}</b>   Bankroll: <b>${bankroll:.2f}</b>",
            f"  Threshold: {thr_str}",
            f"  {mkt}",
        ]
        return self.send("\n".join(lines))

    def alert_no_market(self, today) -> bool:
        lines = [
            f"⚠️ <b>Mercado não encontrado</b>  {today}",
            "  O Polymarket ainda não criou o mercado de hoje.",
        ]
        return self.send("\n".join(lines))

    def alert_peak_detected(self, p_ensemble: float, rmax: float,
                            rmax_time: str, bracket: dict | None,
                            ensemble_result: dict | None = None,
                            market: dict | None = None) -> bool:
        """
        Alerta de pico detectado com detalhe dos 3 modelos e confirmação de mercado.
        """
        if ensemble_result:
            p_lgbm = ensemble_result.get("p_lgbm", 0)
            p_xgb  = ensemble_result.get("p_xgb")
            p_zs   = ensemble_result.get("p_zscore")

            lgbm_str = f"{p_lgbm*100:.1f}%"
            xgb_str  = f"{p_xgb*100:.1f}%" if p_xgb is not None else "N/A"
            zs_str   = f"{p_zs*100:.1f}%" if p_zs is not None else "N/A"

            models_block = (
                f"  🟦 <b>LGBM</b>: {lgbm_str}\n"
                f"  🟧 <b>XGB</b>:  {xgb_str}\n"
                f"  🟪 <b>Z-Score</b>: {zs_str}"
            )
        else:
            models_block = f"  P = {p_ensemble*100:.0f}%"

        bracket_str = ""
        if bracket:
            ask = bracket.get("ask") or bracket.get("price", 0)
            bracket_str = f"\n  🎯 Bracket: <b>{bracket['label']}</b>  ask {ask*100:.0f}¢"

        market_str = ""
        if market and market.get("brackets"):
            best = max(market["brackets"],
                       key=lambda b: b.get("ask") or b.get("price") or 0)
            best_ask = best.get("ask") or best.get("price", 0)
            market_str = (f"\n  🏆 <b>Mercado escolheu</b>: {best['label']} "
                         f"(ask {best_ask*100:.0f}¢)")

        lines = [
            "🔔 <b>PICO DETECTADO</b>",
            f"  🧠 <b>Ensemble</b>: {p_ensemble*100:.1f}%",
            models_block,
            f"  🌡 Running max: <b>{int(round(rmax))}°C</b> @{rmax_time}",
            market_str,
            bracket_str,
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
        p_idx    = bet.get("parcel_idx")
        parcel_s = f"P{p_idx+1}" if p_idx is not None else ""

        lines = [
            f"{icon} <b>Ordem colocada [{mode}] {parcel_s}</b>",
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

    def alert_bet_blocked(self, reason: str, p_ensemble: float = 0.0) -> bool:
        """Alerta quando uma bet é bloqueada."""
        lines = [
            "🚫 <b>Bet bloqueada</b>",
            f"  Motivo: <code>{reason[:200]}</code>",
        ]
        if p_ensemble > 0:
            lines.append(f"  p_ensemble: {p_ensemble*100:.1f}%")
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
            lines = [f"📅 <b>Fim do dia — {day_str}</b>", "  Sem bets hoje."]
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
                lines.append(f"  {st_icon} {pos.bracket_label:<16} {pos.entry_ask*100:.0f}¢  {pnl_s}")
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
                  today, p: float, rmax: float, rmax_time: str,
                  temp_now: float | None, forecast_max: dict | None,
                  market: dict | None, bracket: dict | None, ev: dict | None,
                  peak_detected: bool, bet: dict | None,
                  clob_mode: str = None, trading_mode: str = None,
                  chart: list | None = None, reason: str = "periodic",
                  positions_summary: dict | None = None,
                  om_forecast: dict | None = None,
                  forecast_agreement: dict | None = None,
                  ensemble_result: dict | None = None) -> bool:

        mode = trading_mode or clob_mode or "paper"
        if hasattr(mode, "value"):
            mode_str = mode.value.upper()
        else:
            mode_str = str(mode).replace("TradingMode.", "").upper()

        mode_icon = "🟢" if mode_str == "REAL" else "🟡"
        now_str = datetime.now().strftime("%H:%M")

        lines = [
            f"{mode_icon} <b>Munich Max Temp — V3</b>  [{mode_str}]  {today}  {now_str}",
            "  ─────────────────────────────────────────",
            "",
        ]

        # Chart
        if chart:
            lines.append("🌡 <b>Temperatura hoje</b>")
            lines.extend(chart)
            lines.append("")

        # Temperatura
        temp_str = f"{int(round(temp_now))}°C" if temp_now is not None else "—"
        lines += [
            "🌡 <b>Actual</b>",
            f"  Agora: <b>{temp_str}</b>   Max: <b>{int(round(rmax))}°C</b> @{rmax_time}",
            "",
        ]

        # Dual Forecast
        if forecast_max or om_forecast:
            lines.append("🌤 <b>Previsão Dual</b>")
            if forecast_max:
                lines.append(f"  🟦 WU: max <b>{forecast_max['temp_max']}°C</b>")
            if om_forecast:
                lines.append(f"  🟣 OM: max <b>{om_forecast['temp_max']}°C</b>")
            if forecast_agreement:
                if forecast_agreement["valid"]:
                    lines.append(f"  ✅ Concordam (diff {forecast_agreement.get('diff','?')}°C)")
                else:
                    lines.append(f"  ❌ Discordam ({forecast_agreement.get('reason','?')})")
            lines.append("")

        # Ensemble
        if ensemble_result:
            p_ens = ensemble_result["p_ensemble"]
            peak_str = "  ✓ <b>PICO DETECTADO</b>" if peak_detected else ""
            lines += [
                "🧠 <b>Ensemble</b>",
                f"  <b>{p_ens*100:.1f}%</b> {peak_str}",
            ]
            p_lgbm = ensemble_result.get("p_lgbm")
            p_xgb  = ensemble_result.get("p_xgb")
            p_zs   = ensemble_result.get("p_zscore")
            comp_parts = []
            if p_lgbm is not None: comp_parts.append(f"LGBM {p_lgbm*100:.0f}%")
            if p_xgb  is not None: comp_parts.append(f"XGB {p_xgb*100:.0f}%")
            if p_zs   is not None: comp_parts.append(f"Z {p_zs*100:.0f}%")
            if comp_parts:
                lines.append(f"  {' | '.join(comp_parts)}")
            lines.append("")

        # Mercado
        if market:
            best = max(market["brackets"], key=lambda b: b.get("ask") or b.get("price") or 0)
            best_ask = best.get("ask") or best.get("price", 0)
            lines += [
                "📋 <b>Polymarket</b>",
                f"  🏆 Highest ask: <b>{best['label']}</b> ({best_ask*100:.0f}¢)",
                "",
            ]

        # Bet
        if bet:
            p_idx = bet.get("parcel_idx")
            parcel_s = f"P{p_idx+1}" if p_idx is not None else ""
            ask = bet.get("ask") or bet.get("price", 0)
            size = bet.get("bet_size") or 0
            lines += [
                f"💰 <b>Bet {parcel_s}</b>",
                f"  {bet['bracket']}  ask {ask*100:.1f}¢  ${size:.0f}",
            ]
        else:
            lines.append("💤 Sem bet")

        # P&L
        if positions_summary and (positions_summary["n_won"] + positions_summary["n_lost"]) > 0:
            s  = positions_summary
            nc = s["n_won"] + s["n_lost"]
            wr = s["n_won"] / nc * 100
            wr_icon = "📈" if s["total_pnl_usd"] >= 0 else "📉"
            lines += [
                "",
                f"{wr_icon} <b>Acumulado</b>",
                f"  {s['n_won']}W/{s['n_lost']}L  P&amp;L: <b>{s['total_pnl_usd']:+.2f}</b>",
            ]

        reason_str = {"periodic": "⏱ periódico", "zone_change": "⚡ zona"}.get(reason, reason)
        lines += ["", f"<i>{reason_str}</i>"]

        return self.send("\n".join(lines))


def _tg_bar(p: float, width: int = 10) -> str:
    filled = round(min(max(p, 0), 1) * width)
    return "█" * filled + "░" * (width - filled)
