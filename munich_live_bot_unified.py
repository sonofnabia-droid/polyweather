#!/usr/bin/env python3
"""
munich_live_bot_unified.py — UNIFIED
===================================
Bot live completo com ML Ensemble (LGBM + XGB + Z-Score) + Stop-Loss.

Integração:
- Usa modelos ML do munich_model.py (predict_ensemble)
- Usa configurações do munich_strategy_config.py
- Usa stop-loss do munich_stop_loss.py
- Integra com Polymarket via polymarket_orders.py e polymarket_clob.py
- Usa dados weather do munich_weather.py

Uso:
    python munich_live_bot_unified.py
    python munich_live_bot_unified.py --mode phased
    python munich_live_bot_unified.py --config optimized_config.json
    python munich_live_bot_unified.py --run paper
    python munich_live_bot_unified.py --run real --yes
"""

import argparse
import json
import time
import sys
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

import numpy as np
import requests

from munich_config import (
    WU_API_KEY, POLY_PRIVATE_KEY, POLY_MAX_DAILY_LOSS,
    GAMMA_API, LOG_DIR, MONTH_NAMES,
    DAY_START, DAY_END, MIN_HOUR, BOT_ACTIVE_START, BOT_ACTIVE_END,
    C, R, B, DIM, berlin_now, berlin_date, ceil_slot, _SIGNAL_CHECK_WINDOWS,
)
from munich_model import (
    load_models, predict_ensemble, StreamingPeakDetector,
    set_seasonal_prior, build_features, compute_prev7,
    init_history_max, update_history_max,
)
from munich_strategy_config import (
    load_config, save_config,
)
from munich_stop_loss import Position, StopLossChecker, PositionManager
from munich_phased_entry import PhasedEntry, SingleEntry
from munich_weather import (
    make_wu_session, make_om_session,
    fetch_wu_latest, fetch_wu_forecast_max,
    bootstrap_today, cloud_from_series,
)
from polymarket_clob import (
    ClobClient, TradingMode,
    PositionManager as PolyPositionManager,
)
from polymarket_orders import OrderExecutor, paper_buy
from tg import TG
from munich_fuzzy_gatekeeper import FuzzyGatekeeper, create_gatekeeper

# ═════════════════════════════════════════════════════
#  DATA CLASSES PARA LIVE BOT
# ═════════════════════════════════════════════════════

@dataclass
class DailyStats:
    """Estatísticas diárias."""

    date: date
    trades: List = field(default_factory=list)
    total_invested: float = 0.0
    daily_pnl: float = 0.0
    stop_losses_triggered: int = 0
    max_concurrent_positions: int = 0


@dataclass
class SessionStats:
    """Estatísticas da sessão."""

    start_time: datetime
    total_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    stop_losses: int = 0

    @property
    def duration(self) -> timedelta:
        return datetime.now() - self.start_time

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0


# ═════════════════════════════════════════════════════
#  POLYMARKET MARKET FETCHER
# ═════════════════════════════════════════════════════

class PolymarketFetcher:
    """Fetches market data from Polymarket (Gamma API)."""

    def __init__(self, api_url: str = GAMMA_API):
        self.api_url = api_url

    def date_to_slug(self, d: date) -> str:
        """Converte data para slug do mercado."""
        return (f"highest-temperature-in-munich-on-"
                f"{MONTH_NAMES[d.month]}-{d.day}-{d.year}")

    def fetch_market(self, d: date) -> Optional[Dict]:
        """Busca mercado para uma data."""
        import re

        slug = self.date_to_slug(d)

        def try_api(params):
            try:
                r = requests.get(f"{self.api_url}/events", params=params, timeout=15)
                r.raise_for_status()
                ev = r.json()
                return ev if isinstance(ev, list) else ([ev] if ev else [])
            except Exception:
                return []

        month_s = MONTH_NAMES[d.month].capitalize()
        events = (try_api({"slug": slug}) or
                  try_api({"q": f"highest temperature Munich {month_s} {d.day} {d.year}", "limit": 10}) or
                  try_api({"q": f"Munich temperature {d.year}", "limit": 10}))

        if not events:
            return None

        def is_munich(e):
            t = str(e.get("title", "")).lower()
            return ("munich" in t or "munchen" in t) and ("temp" in t or "temperature" in t or "highest" in t)

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
            label = self._normalize_label(raw_label)

            v = self._extract_temp(label)
            if v is None:
                continue

            def _jload(x):
                if isinstance(x, str):
                    try:
                        return json.loads(x)
                    except Exception:
                        return []
                return x

            outcomes, prices, token_ids = (_jload(m.get("outcomes", "[]")),
                                             _jload(m.get("outcomePrices", "[]")),
                                             _jload(m.get("clobTokenIds", "[]")))

            price_yes, token_yes = None, None
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
                "label": label,
                "price": round(price_yes, 4),
                "ask": round(price_yes, 4),
                "token_id": token_yes,
                "temp_lo": self._bracket_lo(label),
                "temp_hi": self._bracket_hi(label),
                "volume": float(m.get("volume", 0) or 0),
            })

        if not brackets:
            return None

        brackets.sort(key=lambda b: b["temp_lo"])

        return {
            "title": event.get("title", "Munich Max Temp"),
            "end_date": event.get("endDate", ""),
            "volume": float(event.get("volume", 0) or 0),
            "brackets": brackets,
            "n_outcomes": len(brackets),
            "slug": slug,
        }

    def _extract_temp(self, text: str) -> Optional[float]:
        """Extrai temperatura do label."""
        import re
        for pat in [r"([-]?\d+)\s*°?\s*[cC]\b",
                    r"([-]?\d+)\s+or\s+(?:higher|lower|above|below)",
                    r"be\s+([-]?\d+)", r"^\s*([-]?\d+)\s*$"]:
            m = re.search(pat, str(text), re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None

    def _bracket_lo(self, label: str) -> float:
        v = self._extract_temp(label)
        if v is None:
            return 0.0
        s = str(label).lower()
        if any(x in s for x in ("or lower", "or below", "<=")):
            return -99.0
        return v

    def _bracket_hi(self, label: str) -> float:
        v = self._extract_temp(label)
        if v is None:
            return 99.0
        s = str(label).lower()
        if any(x in s for x in ("or higher", "or above", ">=")):
            return 99.0
        return v

    def _normalize_label(self, text: str) -> str:
        """Normaliza label do bracket."""
        if len(text) <= 25:
            return text
        v = self._extract_temp(text)
        if v is None:
            return text
        s = text.lower()
        if any(x in s for x in ("higher", "above", ">=")):
            return f"{v:.0f}C or higher"
        if any(x in s for x in ("lower", "below", "<=")):
            return f"{v:.0f}C or lower"
        return f"{v:.0f}C"


# ═════════════════════════════════════════════════════
#  UNIFIED LIVE BOT
# ═════════════════════════════════════════════════════

class UnifiedLiveBot:
    """
    Bot live unificado com ML + Stop-Loss.

    Integra:
    - Modelos ML (LGBM + XGB + Z-Score)
    - Estratégia de entrada (Phased ou Single)
    - Stop-Loss (temperatura + probabilidade)
    - Polymarket (CLOB + Orders)
    - Weather data (Wunderground + Open-Meteo)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        trading_mode: TradingMode = TradingMode.PAPER
    ):
        # Configuração
        self.config = config or load_config()
        self.trading_mode = trading_mode

        # Modelos
        self.models = load_models()
        set_seasonal_prior(self.models["prior_map"])

        # Estado
        self.running = False
        self.today = None

        # Stop-loss
        self.sl_checker = StopLossChecker(self.config.stop_loss)
        self.position_mgr = PositionManager(self.config.stop_loss)

        # Fuzzy Gatekeeper
        self.gatekeeper = create_gatekeeper(self.config.get("gatekeeper"))

        # Entry strategy
        if self.config.entry.mode == "phased":
            self.entry = PhasedEntry(parcel_size=self.config.entry.phased_parcel_size)
        else:
            self.entry = SingleEntry(
                parcel_size=self.config.position.bet_size,
                threshold=self.config.entry.single_threshold
            )

        # Histórico
        self.history_max = init_history_max()
        self.zscore = StreamingPeakDetector()

        # Polymarket
        self.market_fetcher = PolymarketFetcher()
        self.executor = None
        if trading_mode == TradingMode.REAL:
            self.executor = OrderExecutor(POLY_PRIVATE_KEY)

        # Weather sessions
        self.wu_session = make_wu_session()
        self.om_session = make_om_session()

        # Estatísticas
        self.session_stats = SessionStats(start_time=datetime.now())
        self.daily_stats = None

        # Slots hoje
        self.slots_today = []

        # Estado de entrada
        self.last_buy_slot = None
        self.cooldown_slots = int(self.config.position.cooldown_minutes / 30)

    def start(self):
        """Inicia o bot."""
        print(f"\n  {B}{C['cyan']}=== Munich Live Bot (Unified) ==={R}\n")
        print(f"  Mode: {self.trading_mode.value.upper()}")
        print(f"  Entry: {self.config.entry.mode.upper()}")
        print(f"  Stop-Loss: {self.config.stop_loss.mode}")
        print(f"    - temp_threshold: {self.config.stop_loss.temp_threshold}°C")
        print(f"    - prob_threshold: {self.config.stop_loss.prob_threshold*100:.0f}%")
        print()

        self.running = True

        try:
            while self.running:
                self._main_loop()
        except KeyboardInterrupt:
            print(f"\n  {C['yellow']}Stopped by user.{reset}")
        finally:
            self.stop()

    def _main_loop(self):
        """Loop principal do bot."""
        now_berlin = berlin_now()
        h = now_berlin.hour

        # Verificar se estamos no horário ativo
        if h < BOT_ACTIVE_START or h >= BOT_ACTIVE_END:
            self._sleep(60)
            return

        # Verificar se é um novo dia
        current_date = berlin_date()
        if self.today != current_date:
            self._new_day(current_date)

        # Verificar horário de trading
        if h < DAY_START or h >= DAY_END:
            self._sleep(60)
            return

        try:
            # Buscar dados weather
            wu_data = fetch_wu_latest(self.wu_session)
            if not wu_data:
                self._sleep(30)
                return

            current_temp = wu_data["temp_c"]

            # Bootstrap slots (se necessário)
            if len(self.slots_today) < 4:
                self.slots_today = bootstrap_today(wu_data, self.om_session) if self.om_session else []

            if len(self.slots_today) < 4:
                self._sleep(30)
                return

            # Adicionar slot atual
            self.slots_today.append({
                "temp_c": current_temp,
                "cloud_cover": wu_data.get("cloud_cover", 50),
                "humidity": wu_data.get("humidity", 70),
                "dewpoint_c": wu_data.get("dewpoint_c", current_temp - 10),
                "pressure_hpa": wu_data.get("pressure_hpa", 1013),
                "wind_dir_deg": wu_data.get("wind_dir_deg", 0),
                "wind_speed_kmh": wu_data.get("wind_speed_kmh", 5),
            })

            # Atualizar histórico
            update_history_max(self.history_max, self.slots_today)

            # Predição ensemble
            current_extra = {
                "hour": h,
                "slot30": 0 if now_berlin.minute < 30 else 30,
                "temp_c": current_temp,
                "cloud_cover": wu_data.get("cloud_cover", 50),
                "humidity": wu_data.get("humidity", 70),
                "dewpoint_c": wu_data.get("dewpoint_c", current_temp - 10),
                "pressure_hpa": wu_data.get("pressure_hpa", 1013),
                "wind_dir_deg": wu_data.get("wind_dir_deg", 0),
                "wind_speed_kmh": wu_data.get("wind_speed_kmh", 5),
                "uv_index": 3.0,
                "prev_7d_avg_max": compute_prev7(self.history_max, current_date),
            }

            pred = predict_ensemble(
                self.models, self.slots_today, current_extra,
                now_berlin.month, now_berlin.timetuple().tm_yday, self.zscore
            )
            p_ensemble = pred["p_ensemble"]

            # Display
            self._display_prediction(p_ensemble, current_temp, pred)

            # Check stop-loss para posições ativas
            self._check_stop_loss(current_temp, p_ensemble)

            # Check entry
            if not self.position_mgr.get_active_positions():
                self._check_entry(p_ensemble, h, current_temp)

            # Sleep até próximo check
            interval = _SIGNAL_CHECK_WINDOWS.get(h, 120)
            self._sleep(interval)

        except Exception as e:
            print(f"  {C['red']}Error: {e}{reset}")
            self._sleep(60)

    def _new_day(self, new_date: date):
        """Inicia novo dia."""
        if self.daily_stats:
            # Guardar estatísticas do dia anterior
            self._save_daily_stats()

        print(f"\n  {C['cyan']}=== New Day: {new_date} ==={reset}\n")

        self.today = new_date
        self.daily_stats = DailyStats(date=new_date)
        self.slots_today = []
        self.zscore.reset()
        self.last_buy_slot = None

        # Reset entry strategy
        if self.config.entry.mode == "phased":
            self.entry.reset()
        else:
            self.entry.bought = False

    def _check_stop_loss(self, current_temp: float, p_ensemble: float):
        """Verifica stop-loss para posições ativas."""
        active_positions = self.position_mgr.get_active_positions()

        for pos in active_positions:
            should_exit, reason, loss = self.sl_checker.check(
                pos, current_temp, p_ensemble
            )

            if should_exit:
                print(f"\n  {C['red']}STOP-LOSS ACTIVATED!{reset}")
                print(f"  {reason}")

                # Exit position
                self._exit_position(pos, reason, True)

    def _check_entry(self, p_ensemble: float, hour: int, current_temp: float):
        """Verifica sinal de entrada."""
        # Cooldown
        if self.last_buy_slot is not None:
            current_slot = hour * 2 + (0 if berlin_now().minute < 30 else 30)
            if current_slot - self.last_buy_slot < self.cooldown_slots:
                return

        # Buscar mercado
        market = self.market_fetcher.fetch_market(self.today)
        if not market:
            print(f"  {C['dim']}No market found for today.{reset}")
            return

        # Running max
        running_max = max(s["temp_c"] for s in self.slots_today)

        # Forecast agreement
        fc_agreement = {"valid": np.random.random() < 0.80}  # Placeholder

        # Fuzzy Gatekeeper: verificar contexto antes de entrada
        # Buscar bracket mais próximo para o ask
        bracket_for_gatekeeper = self._find_entry_bracket(
            market["brackets"], running_max, 0
        )
        ask_for_gatekeeper = bracket_for_gatekeeper.get("ask", 0.5) if bracket_for_gatekeeper else 0.5

        # Obter componente z-score da predição anterior
        zscore_component = None  # TODO: obter do predict_ensemble

        # Avaliar com Fuzzy Gatekeeper
        gatekeeper_result = self.gatekeeper.evaluate(
            p_ensemble=p_ensemble,
            ask_price=ask_for_gatekeeper,
            forecast_agreement=fc_agreement,
            market=market,
            zscore_component=zscore_component,
            running_max=running_max,
            current_temp=current_temp
        )

        # Se Gatekeeper bloquear, não continuar
        if not gatekeeper_result.allowed:
            print(f"  {C['red']}GATEKEEPER: {gatekeeper_result.reason}{reset}")
            return

        # Se Gatekeeper permitir mas for RISKY, avisar
        if gatekeeper_result.state.value == "risky":
            print(f"  {C['yellow']}GATEKEEPER: {gatekeeper_result.reason}{reset}")

        # Avaliar entrada
        actions = self.entry.evaluate(
            p_ensemble, hour, market, running_max, fc_agreement
        )

        for act in actions:
            if act["size_usdc"] > 0:
                # Encontrar bracket
                bracket = self._find_entry_bracket(
                    market["brackets"], running_max, act["parcel_idx"]
                )

                if bracket and bracket["ask"] >= self.config.position.min_ask:
                    self._enter_position(bracket, act)

    def _find_entry_bracket(
        self,
        brackets: List[Dict],
        running_max: float,
        parcel_idx: int
    ) -> Optional[Dict]:
        """Encontra bracket para entrada."""
        if parcel_idx == 0:
            return max(brackets, key=lambda b: b["ask"])
        else:
            rmax_int = int(round(running_max))
            return next(
                (b for b in brackets
                 if b["temp_lo"] <= rmax_int <= b["temp_hi"]),
                max(brackets, key=lambda b: b["ask"])
            )

    def _enter_position(self, bracket: Dict, action: Dict):
        """Entra em posição."""
        ask = bracket["ask"]
        size_usdc = action["size_usdc"]
        shares = size_usdc / ask

        print(f"\n  {C['green']}SIGNAL: {action['reason']}{reset}")
        print(f"    Bracket: {bracket['label']}")
        print(f"    Ask: {ask*100:.1f}¢")
        print(f"    Size: ${size_usdc:.2f} ({shares:.2f} shares)")

        # Confirmar (modo real)
        if self.trading_mode == TradingMode.REAL:
            confirm = input(f"  Confirm? (y/n): ").strip().lower()
            if confirm != "y":
                return

        # Executar ordem
        if self.trading_mode == TradingMode.PAPER:
            result = paper_buy(
                bracket["token_id"], ask, size_usdc, label=bracket["label"]
            )
        else:
            result = self.executor.buy(
                token_id=bracket["token_id"],
                price=ask,
                size_usdc=size_usdc,
                label=bracket["label"]
            )

        if result and result.get("success"):
            # Criar posição
            position = Position(
                token_id=bracket["token_id"],
                bracket_label=bracket["label"],
                entry_temp=max(s["temp_c"] for s in self.slots_today),
                entry_ask=ask,
                entry_p_ensemble=0.8,  # Placeholder
                entry_time=berlin_now().strftime("%H:%M"),
                shares=shares,
                cost_usdc=size_usdc,
            )

            self.position_mgr.add_position(position)

            # Marcar entry
            parcel_idx = action["parcel_idx"]
            if self.config.entry.mode == "phased":
                self.entry.mark_bought(parcel_idx, {
                    "hour": berlin_now().hour,
                    "slot30": 0 if berlin_now().minute < 30 else 30,
                    "ask": ask,
                    "size_usdc": size_usdc,
                    "bracket_label": bracket["label"],
                })
            else:
                self.entry.bought = True

            self.last_buy_slot = berlin_now().hour * 2 + (0 if berlin_now().minute < 30 else 30)

            # Estatísticas
            self.session_stats.total_trades += 1
            self.daily_stats.total_invested += size_usdc

            print(f"  {C['green']}✓ Position entered.{reset}")
        else:
            print(f"  {C['red']}✗ Order failed.{reset}")

    def _exit_position(self, position: Position, reason: str, is_stop_loss: bool):
        """Sai de posição."""
        # Buscar preço atual
        market = self.market_fetcher.fetch_market(self.today)
        current_ask = 0.5  # Default

        if market:
            for b in market["brackets"]:
                if b["label"] == position.bracket_label:
                    current_ask = b["ask"]
                    break

        # Calcular PnL (simplificado)
        if is_stop_loss:
            pnl = -position.cost_usdc * 0.3  # Assumir 30% perda
            exit_reason = f"STOP-LOSS: {reason}"
        else:
            peak_temp = max(s["temp_c"] for s in self.slots_today)
            won = peak_temp <= position.entry_temp + 0.5
            pnl = position.cost_usdc * (1 / position.entry_ask - 1) if won else -position.cost_usdc
            exit_reason = "natural"

        print(f"    PnL: ${pnl:+.2f}")

        # Remover posição
        self.position_mgr.remove_position(position, pnl)

        # Estatísticas
        self.session_stats.total_pnl += pnl
        self.daily_stats.daily_pnl += pnl

        if pnl > 0:
            self.session_stats.wins += 1
        else:
            self.session_stats.losses += 1

        if is_stop_loss:
            self.session_stats.stop_losses += 1
            self.daily_stats.stop_losses_triggered += 1

    def _display_prediction(self, p: float, temp: float, pred: Dict):
        """Mostra predição atual."""
        p_pct = p * 100
        color = "green" if p >= 0.8 else "yellow" if p >= 0.6 else "dim"

        print(f"  {C[color]}p_ensemble: {p_pct:.0f}% | temp: {temp:.1f}°C{reset}")

        if pred.get("components"):
            comps = pred["components"]
            lgbm_c = comps.get("lgbm_contribution", 0) * 100
            xgb_c = comps.get("xgb_contribution", 0) * 100 if comps.get("xgb_contribution") else None
            zs_c = comps.get("zscore_contribution", 0) * 100 if comps.get("zscore_contribution") else None

            parts = [f"LGBM:{lgbm_c:.0f}%"]
            if xgb_c is not None:
                parts.append(f"XGB:{xgb_c:.0f}%")
            if zs_c is not None:
                parts.append(f"ZS:{zs_c:.0f}%")

            print(f"    {' + '.join(parts)}")

    def _sleep(self, seconds: int):
        """Sleep com atualizações de stats."""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)

    def _save_daily_stats(self):
        """Guarda estatísticas diárias."""
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"daily_{self.daily_stats.date.isoformat()}.json"

        data = {
            "date": self.daily_stats.date.isoformat(),
            "trades": len(self.daily_stats.trades),
            "total_invested": self.daily_stats.total_invested,
            "daily_pnl": self.daily_stats.daily_pnl,
            "stop_losses_triggered": self.daily_stats.stop_losses_triggered,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def stop(self):
        """Para o bot."""
        self.running = False

        # Imprimir estatísticas finais
        print(f"\n  {B}{C['cyan']}=== Session Stats ==={R}\n")
        print(f"  Duration: {self.session_stats.duration}")
        print(f"  Trades: {self.session_stats.total_trades}")
        print(f"  PnL: ${self.session_stats.total_pnl:+.2f}")
        print(f"  Win Rate: {self.session_stats.win_rate:.1f}%")
        print(f"  Stop-Losses: {self.session_stats.stop_losses}")


# ═════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Munich Live Bot")
    parser.add_argument("--run", choices=["paper", "real"])
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--mode", choices=["single", "phased"], default="single")
    parser.add_argument("--config", type=str, help="Config file path")
    args = parser.parse_args()

    # Trading mode
    if args.run == "paper":
        trading_mode = TradingMode.PAPER
    elif args.run == "real":
        trading_mode = TradingMode.REAL
    else:
        trading_mode = TradingMode.PAPER  # Default

    # Config
    config = load_config(Path(args.config) if args.config else None)
    config.entry.mode = args.mode

    print(f"\n  {B}{C['cyan']}=== Munich Live Bot (Unified) ==={R}\n")

    if trading_mode == TradingMode.REAL:
        if not POLY_PRIVATE_KEY:
            print(f"  {C['red']}No POLY_PRIVATE_KEY set. Using paper mode.{reset}")
            trading_mode = TradingMode.PAPER
        else:
            print(f"  {C['red']}REAL MODE{reset}")
    else:
        print(f"  {C['yellow']}PAPER MODE{reset}")

    # Criar e iniciar bot
    bot = UnifiedLiveBot(config=config, trading_mode=trading_mode)
    bot.start()


if __name__ == "__main__":
    main()
