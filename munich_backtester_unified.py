#!/usr/bin/env python3
"""
munich_backtester_unified.py — UNIFIED
=====================================
Backtester completo com ML Ensemble (LGBM + XGB + Z-Score) + Stop-Loss.

Integração:
- Usa modelos ML do munich_model.py (predict_ensemble)
- Usa configurações do munich_strategy_config.py
- Usa stop-loss do munich_stop_loss.py
- Pode ser integrado com Optuna para otimização de parâmetros

Uso:
    python munich_backtester_unified.py
    python munich_backtester_unified.py --start 2020-01-01 --end 2023-12-31
    python munich_backtester_unified.py --mode phased
    python munich_backtester_unified.py --config optimized_config.json
"""

import argparse
import json
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rich.progress import (Progress, BarColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.console import Console
from rich.table import Table
from rich import box as rich_box
from rich.panel import Panel
from rich.rule import Rule

from munich_config import (
    MODEL_LGB, MODEL_XGB, MODEL_CONFIG, DATA_CSV,
    DAY_START, DAY_END, MIN_HOUR,
    BACKTEST_RESULTS_DIR, C, R, B, DIM,
    berlin_date, ceil_slot,
)

# Alias para consistência
reset = C["reset"]
from munich_model import (
    load_models, predict_ensemble, StreamingPeakDetector,
    set_seasonal_prior, build_features, compute_prev7,
)
from munich_strategy_config import (
    load_config, save_config, config_to_params_dict,
    params_dict_to_config, StrategyConfig,
)
from munich_stop_loss import Position, StopLossChecker, PositionManager
from munich_phased_entry import PhasedEntry, SingleEntry

warnings.filterwarnings("ignore")
optuna_logging = logging.getLogger("optuna")
optuna_logging.setLevel(logging.WARNING)

_console = Console(force_terminal=True)


# ═════════════════════════════════════════════════════
#  DATA CLASSES PARA BACKTEST
# ═════════════════════════════════════════════════════

@dataclass
class Trade:
    """Representa um trade (entrada + saída)."""

    trade_id: int
    entry_date: date
    entry_time: str
    entry_temp: float
    entry_p: float
    entry_ask: float
    bracket_label: str
    shares: float
    cost_usdc: float

    exit_date: Optional[date] = None
    exit_time: Optional[str] = None
    exit_temp: Optional[float] = None
    exit_p: Optional[float] = None
    exit_ask: Optional[float] = None
    exit_reason: str = ""

    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_slots: int = 0

    stop_loss_triggered: bool = False
    stop_loss_reason: str = ""


@dataclass
class DailyResult:
    """Resultado de um dia."""

    date: date
    peak_temp: float
    peak_hour: int
    peak_slot: int

    trades: List[Trade] = field(default_factory=list)
    max_positions: int = 0
    daily_pnl: float = 0.0
    stop_losses_triggered: int = 0


@dataclass
class BacktestStats:
    """Estatísticas do backtest."""

    total_days: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    stop_losses: int = 0

    starting_capital: float = 1000.0
    final_capital: float = 1000.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    sharpe_ratio: float = 0.0

    max_drawdown_eur: float = 0.0
    max_drawdown_pct: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0

    win_rate: float = 0.0
    avg_duration_slots: float = 0.0

    # Por estação
    seasonal_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Métricas adicionais
    avg_daily_investment: float = 0.0
    avg_positions_per_day: float = 0.0


# ═════════════════════════════════════════════════════
#  SIMULATED MARKET (POLYMARKET-LIKE)
# ═════════════════════════════════════════════════════

class SimulatedPolymarket:
    """
    Simula mercado Polymarket para backtesting.

    Gera brackets e preços (ask/bid) baseados em:
    - p_ensemble (confiança do modelo)
    - running_max (temperatura máxima até agora)
    - hour (ruído é maior de manhã)
    """

    def __init__(self, temp_range: range = range(5, 40)):
        self.temp_range = temp_range

    def get_brackets(
        self,
        p_ensemble: float,
        running_max: float,
        hour: int
    ) -> List[Dict[str, Any]]:
        """Gera brackets de mercado com preços (ask/bid)."""
        brackets = []
        rmax_int = int(round(running_max))

        # Ruído de mercado: maior de manhã, menor à tarde
        if hour < 11:
            market_noise = 0.20
        elif hour < 14:
            market_noise = 0.12
        else:
            market_noise = 0.04

        for temp in self.temp_range:
            dist = abs(temp - rmax_int)

            # Base ask: influenciado por p_ensemble
            if dist == 0:
                # Bracket do pico: ask mais alto (mercado também acha que é o pico)
                base_ask = min(0.92, p_ensemble * 0.8 + 0.15)
            elif dist == 1:
                base_ask = min(0.70, p_ensemble * 0.6 + 0.10)
            elif dist == 2:
                base_ask = min(0.40, p_ensemble * 0.4 + 0.05)
            elif dist == 3:
                base_ask = min(0.20, p_ensemble * 0.2)
            else:
                base_ask = max(0.03, 0.10 - dist * 0.015)

            # Bracket "or lower": sempre muito barato (mercado não acredita)
            if temp < rmax_int - 1:
                base_ask = min(0.97, base_ask + 0.10)

            # Adicionar ruído e clipar
            ask = float(np.clip(
                base_ask + np.random.uniform(-market_noise, market_noise),
                0.02, 0.97
            ))

            # Bid = ask * (1 - spread), spread ~ 5-10%
            bid = ask * (1.0 - np.random.uniform(0.05, 0.10))

            is_last = (temp == self.temp_range[-1])
            is_first = (temp == self.temp_range[0])

            # Label
            if is_last:
                label, lo, hi = f"{temp}°C or higher", float(temp), 99.0
            elif is_first:
                label, lo, hi = f"{temp}°C or lower", -99.0, float(temp)
            else:
                label, lo, hi = f"{temp}°C", float(temp), float(temp)

            brackets.append({
                "label": label,
                "ask": round(ask, 4),
                "bid": round(bid, 4),
                "temp_lo": lo,
                "temp_hi": hi,
            })

        return brackets

    def find_bracket(
        self,
        brackets: List[Dict],
        temp: float
    ) -> Optional[Dict]:
        """Encontra o bracket correspondente à temperatura."""
        temp_int = int(round(temp))

        for b in brackets:
            lo, hi = b["temp_lo"], b["temp_hi"]
            if lo <= temp_int <= hi:
                return b

        # Fallback: bracket mais próximo
        return min(
            brackets,
            key=lambda b: abs((b["temp_lo"] + b["temp_hi"]) / 2 - temp_int)
        )


# ═════════════════════════════════════════════════════
#  UNIFIED BACKTESTER
# ═════════════════════════════════════════════════════

class UnifiedBacktester:
    """
    Backtester unificado com ML + Stop-Loss.

    Integra:
    - Modelos ML (LGBM + XGB + Z-Score)
    - Estratégia de entrada (Phased ou Single)
    - Stop-Loss (temperatura + probabilidade)
    - Simulated Polymarket
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        models: Optional[dict] = None,
        silent: bool = False
    ):
        self.config = config or load_config()
        self.models = models or load_models()
        self.silent = silent
        self.con = Console() if not silent else None

        # Inicializar prior sazonal
        set_seasonal_prior(self.models["prior_map"])

        # Simulated market
        self.market = SimulatedPolymarket()

        # Stop-loss checker
        self.sl_checker = StopLossChecker(self.config.stop_loss)

        # Estado do backtest
        self.trades: List[Trade] = []
        self.daily_results: List[DailyResult] = []
        self.capital_history: List[float] = []

    def load_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Carrega dados históricos."""
        from zoneinfo import ZoneInfo

        path = Path(DATA_CSV)
        if not path.exists():
            raise FileNotFoundError(f"{path} não encontrado")

        with open(path, "r") as f:
            first = f.readline()
        sep = "\t" if "\t" in first else ","

        df = pd.read_csv(path, sep=sep, low_memory=False)

        # Parse datetime
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        if df["timestamp_utc"].dt.tz is not None:
            df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert(None)
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")

        # Converter para hora local
        berlin = ZoneInfo("Europe/Berlin")
        df["datetime_local"] = df["timestamp_utc"].dt.tz_convert(berlin)

        # Extrações
        dt_locals, dates, hours, slots30 = [], [], [], []
        for ts in df["timestamp_utc"]:
            dt_loc = ts.astimezone(berlin)
            h, m = dt_loc.hour, dt_loc.minute
            h2, s2 = ceil_slot(h, m)
            if h2 == 24:
                dt_loc = dt_loc + timedelta(days=1)
                dt_loc = dt_loc.replace(hour=0)
                h2 = 0
            dt_locals.append(dt_loc)
            dates.append(dt_loc.date())
            hours.append(h2)
            slots30.append(s2)

        df["datetime_local"] = dt_locals
        df["date"] = dates
        df["hour"] = hours
        df["slot30"] = slots30
        df["month"] = df["datetime_local"].dt.month
        df["doy"] = df["datetime_local"].dt.dayofyear
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")

        # Extras
        df["humidity"] = pd.to_numeric(df.get("humidity_pct", 70), errors="coerce")
        df["cloud_cover"] = pd.to_numeric(df.get("sky_cover", 50), errors="coerce")

        # V2 features
        df["dewpoint_c"] = pd.to_numeric(df.get("dewpt_c", df["temp_c"] - 10), errors="coerce")
        df["pressure_hpa"] = pd.to_numeric(df.get("pressure_hpa", 1013), errors="coerce")
        df["wind_dir_deg"] = pd.to_numeric(df.get("wind_dir_deg", 0), errors="coerce")
        df["wind_speed_kmh"] = pd.to_numeric(df.get("wind_speed_kmh", 5), errors="coerce")
        df["uv_index"] = pd.to_numeric(df.get("uv_index", 3.0), errors="coerce").fillna(3.0)

        # Filtrar datas
        df = df[
            (df["date"] >= start_date) &
            (df["date"] <= end_date)
        ].copy()

        # Filtrar horário
        df = df[
            (df["hour"] >= DAY_START) &
            (df["hour"] <= DAY_END)
        ].copy()

        df = df.dropna(subset=["temp_c"]).sort_values(
            ["date", "hour", "slot30"]
        ).reset_index(drop=True)

        return df

    def run_backtest(
        self,
        start_date: date,
        end_date: date
    ) -> BacktestStats:
        """Executa backtest completo."""
        if not self.silent:
            _console.print(
                f"\n  {C['cyan']}Running backtest: {start_date} → {end_date}{reset}"
            )

        # Carregar dados
        df = self.load_data(start_date, end_date)

        if not self.silent:
            _console.print(f"  {C['green']}✓[reset] {len(df):,} slots, {df['date'].nunique()} days")

        # Pre-computar prev7
        daily_max = df.groupby("date")["temp_c"].max().sort_index()
        daily_max_dict = {d: v for d, v in zip(daily_max.index, daily_max.values)}
        prev7_map = compute_prev7(daily_max_dict)

        # Loop por dia
        with Progress(
            TextColumn("[cyan]Days..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console,
            disable=self.silent,
        ) as progress:
            task = progress.add_task("", total=df["date"].nunique())

            for d, day_df in df.groupby("date"):
                progress.update(task, advance=1)

                daily_result = self._run_day(
                    day_df, d, daily_max, prev7_map
                )
                self.daily_results.append(daily_result)

        # Calcular estatísticas
        return self._compute_stats()

    def _run_day(
        self,
        day_df: pd.DataFrame,
        day_date: date,
        daily_max: pd.Series,
        prev7_map: Dict[date, float]
    ) -> DailyResult:
        """Executa backtest para um dia."""
        day_df = day_df.sort_values(["hour", "slot30"]).reset_index(drop=True)

        # Peak do dia
        peak_idx = day_df["temp_c"].idxmax()
        peak_temp = day_df.loc[peak_idx, "temp_c"]
        peak_hour = int(day_df.loc[peak_idx, "hour"])
        peak_slot = int(day_df.loc[peak_idx, "slot30"])

        result = DailyResult(
            date=day_date,
            peak_temp=peak_temp,
            peak_hour=peak_hour,
            peak_slot=peak_slot,
        )

        # Estado do dia
        slots_so_far = []
        zscore = StreamingPeakDetector()
        zscore.reset()

        position_mgr = PositionManager(self.config.stop_loss)

        # Entry strategy
        if self.config.entry.mode == "phased":
            entry = PhasedEntry(parcel_size=self.config.entry.phased_parcel_size)
        else:
            entry = SingleEntry(
                parcel_size=self.config.position.bet_size,
                threshold=self.config.entry.single_threshold
            )

        last_buy_slot = None
        cooldown_slots = int(self.config.position.cooldown_minutes / 30)

        for _, row in day_df.iterrows():
            h, s = int(row["hour"]), int(row["slot30"])
            t = float(row["temp_c"])
            cl = float(row["cloud_cover"])
            hu = float(row["humidity"])

            slot_entry = {
                "hour": h, "slot30": s, "temp_c": t,
                "cloud_cover": cl, "humidity": hu,
                "dewpoint_c": float(row["dewpoint_c"]),
                "pressure_hpa": float(row["pressure_hpa"]),
                "wind_dir_deg": float(row["wind_dir_deg"]),
                "wind_speed_kmh": float(row["wind_speed_kmh"]),
            }
            slots_so_far.append(slot_entry)

            # Cooldown
            if last_buy_slot is not None:
                slots_since_buy = (h * 2 + s // 30) - last_buy_slot
                if slots_since_buy < cooldown_slots:
                    continue

            # Check stop-loss para posições ativas
            if h >= MIN_HOUR and len(slots_so_far) >= 4:
                running_max = max(sl["temp_c"] for sl in slots_so_far)

                # Predição ensemble
                current = {
                    "hour": h, "slot30": s, "temp_c": t,
                    "cloud_cover": cl, "humidity": hu,
                    "dewpoint_c": float(row["dewpoint_c"]),
                    "pressure_hpa": float(row["pressure_hpa"]),
                    "wind_dir_deg": float(row["wind_dir_deg"]),
                    "wind_speed_kmh": float(row["wind_speed_kmh"]),
                    "uv_index": float(row.get("uv_index", 3.0)),
                    "prev_7d_avg_max": prev7_map.get(day_date, t),
                }

                pred = predict_ensemble(
                    self.models, slots_so_far, current,
                    int(row["month"]), int(row["doy"]), zscore
                )
                p_ensemble = pred["p_ensemble"]

                # Update stop-loss
                to_exit = position_mgr.update_positions(t, p_ensemble)

                for pos in to_exit:
                    # Sair da posição (cashout)
                    trade = self._create_exit_trade(pos, day_df, day_date, h, s, True)
                    result.trades.append(trade)
                    position_mgr.remove_position(pos, trade.pnl)

                # Check entry
                if not position_mgr.get_active_positions():
                    # Gerar mercado simulado
                    brackets = self.market.get_brackets(p_ensemble, running_max, h)
                    market_sim = {"brackets": brackets}

                    # Forecast agreement (simulado)
                    fc_agreement = {"valid": np.random.random() < 0.80}

                    # Avaliar entrada
                    actions = entry.evaluate(
                        p_ensemble, h, market_sim, running_max, fc_agreement
                    )

                    for act in actions:
                        if act["size_usdc"] > 0:
                            # Criar posição
                            bracket = self._find_entry_bracket(
                                brackets, running_max, act["parcel_idx"]
                            )

                            if bracket and bracket["ask"] >= self.config.position.min_ask:
                                # Cooldown
                                last_buy_slot = h * 2 + s // 30

                                # Criar trade
                                trade_id = len(self.trades) + len(result.trades) + 1
                                shares = act["size_usdc"] / bracket["ask"]

                                pos = Position(
                                    token_id=f"sim_{trade_id}",
                                    bracket_label=bracket["label"],
                                    entry_temp=t,
                                    entry_ask=bracket["ask"],
                                    entry_p_ensemble=p_ensemble,
                                    entry_time=f"{h:02d}:{s:02d}",
                                    shares=shares,
                                    cost_usdc=act["size_usdc"],
                                )

                                position_mgr.add_position(pos)

                                # Criar trade (entry only)
                                trade = Trade(
                                    trade_id=trade_id,
                                    entry_date=day_date,
                                    entry_time=f"{h:02d}:{s:02d}",
                                    entry_temp=t,
                                    entry_p=p_ensemble,
                                    entry_ask=bracket["ask"],
                                    bracket_label=bracket["label"],
                                    shares=shares,
                                    cost_usdc=act["size_usdc"],
                                )

                                result.trades.append(trade)

        # Fechar posições no fim do dia
        for pos in position_mgr.get_active_positions():
            trade = self._create_exit_trade(pos, day_df, day_date, peak_hour, peak_slot, False)
            result.trades.append(trade)
            position_mgr.remove_position(pos, trade.pnl)

        result.stop_losses_triggered = position_mgr.stop_losses_triggered
        result.daily_pnl = sum(t.pnl for t in result.trades)
        result.max_positions = position_mgr.stop_losses_triggered + len(result.trades)

        return result

    def _find_entry_bracket(
        self,
        brackets: List[Dict],
        running_max: float,
        parcel_idx: int
    ) -> Optional[Dict]:
        """Encontra bracket para entrada."""
        if parcel_idx == 0:
            # Parcela 1: highest ask (mais otimista)
            return max(brackets, key=lambda b: b["ask"])
        else:
            # Outras: running max
            rmax_int = int(round(running_max))
            return next(
                (b for b in brackets
                 if b["temp_lo"] <= rmax_int <= b["temp_hi"]),
                max(brackets, key=lambda b: b["ask"])
            )

    def _create_exit_trade(
        self,
        position: Position,
        day_df: pd.DataFrame,
        day_date: date,
        hour: int,
        slot: int,
        is_stop_loss: bool
    ) -> Trade:
        """Cria trade com saída (stop-loss ou natural)."""
        # Encontrar bracket no fim
        running_max = day_df["temp_c"].max()
        brackets = self.market.get_brackets(0.8, running_max, hour)
        bracket = self.market.find_bracket(brackets, position.entry_temp)

        exit_ask = bracket["ask"] if bracket else position.entry_ask

        # Calcular PnL
        if is_stop_loss:
            # Stop-loss: vender ao bid (simulado)
            exit_bid = bracket.get("bid", exit_ask * 0.95)
            pnl = position.cost_usdc * (exit_bid / position.entry_ask - 1)
            exit_reason = position.exit_reason
        else:
            # Exit natural: verificar se ganhou
            peak_temp = day_df["temp_c"].max()
            entry_temp_float = position.entry_temp

            # Em Polymarket, ganhamos se a temperatura não exceder o bracket
            won = peak_temp <= entry_temp_float + 0.5  # Margem de erro

            if won:
                pnl = position.cost_usdc * (1 / position.entry_ask - 1)
            else:
                pnl = -position.cost_usdc

            exit_reason = "natural"

        trade = Trade(
            trade_id=position.entry_time,
            entry_date=day_date,
            entry_time=position.entry_time,
            entry_temp=position.entry_temp,
            entry_p=position.entry_p_ensemble,
            entry_ask=position.entry_ask,
            bracket_label=position.bracket_label,
            shares=position.shares,
            cost_usdc=position.cost_usdc,

            exit_date=day_date,
            exit_time=f"{hour:02d}:{slot:02d}",
            exit_temp=day_df["temp_c"].max(),
            exit_p=0.5,  # Simplificado
            exit_ask=exit_ask,
            exit_reason=exit_reason,

            pnl=pnl,
            pnl_pct=(pnl / position.cost_usdc) * 100,
            duration_slots=(hour * 2 + slot // 30) - (int(position.entry_time[:2]) * 2 + int(position.entry_time[3:]) // 30),

            stop_loss_triggered=is_stop_loss,
            stop_loss_reason=position.exit_reason if is_stop_loss else "",
        )

        return trade

    def _compute_stats(self) -> BacktestStats:
        """Computa estatísticas do backtest."""
        stats = BacktestStats(
            starting_capital=self.config.position.bet_size * 10  # Assumir bankroll
        )

        # Agregar trades
        for result in self.daily_results:
            stats.total_trades += len(result.trades)
            stats.total_days += 1

            for trade in result.trades:
                if trade.stop_loss_triggered:
                    stats.stop_losses += 1

                if trade.pnl > 0:
                    stats.wins += 1
                else:
                    stats.losses += 1

        stats.total_pnl = sum(t.pnl for res in self.daily_results for t in res.trades)
        stats.final_capital = stats.starting_capital + stats.total_pnl
        stats.roi_pct = (stats.total_pnl / stats.starting_capital) * 100
        stats.win_rate = (stats.wins / stats.total_trades * 100) if stats.total_trades > 0 else 0

        # Win/loss stats
        wins = [t.pnl for res in self.daily_results for t in res.trades if t.pnl > 0]
        losses = [t.pnl for res in self.daily_results for t in res.trades if t.pnl <= 0]

        stats.avg_win = np.mean(wins) if wins else 0
        stats.avg_loss = np.mean(losses) if losses else 0
        stats.max_win = max(wins) if wins else 0
        stats.max_loss = min(losses) if losses else 0

        # Drawdown
        peak = stats.starting_capital
        max_dd = 0
        for result in self.daily_results:
            for trade in result.trades:
                peak += trade.pnl
                dd = max(0, peak - (stats.starting_capital + trade.pnl))
                max_dd = max(max_dd, dd)

        stats.max_drawdown_eur = max_dd
        stats.max_drawdown_pct = (max_dd / stats.starting_capital) * 100

        # Sharpe (simplificado)
        if stats.total_pnl > 0:
            stats.sharpe_ratio = stats.roi_pct / (stats.max_drawdown_pct + 0.01)
        else:
            stats.sharpe_ratio = 0

        # Duration
        durations = [t.duration_slots for res in self.daily_results for t in res.trades]
        stats.avg_duration_slots = np.mean(durations) if durations else 0

        # Por estação
        from munich_config import SEASONS
        for season, months in SEASONS.items():
            seasonal_trades = [
                t for res in self.daily_results for t in res.trades
                if res.date.month in months
            ]

            if seasonal_trades:
                n = len(seasonal_trades)
                wins_n = sum(1 for t in seasonal_trades if t.pnl > 0)
                pnl_n = sum(t.pnl for t in seasonal_trades)

                stats.seasonal_stats[season] = {
                    "n_trades": n,
                    "win_rate": (wins_n / n * 100) if n > 0 else 0,
                    "pnl": pnl_n,
                }

        # Médias diárias
        stats.avg_daily_investment = np.mean([
            sum(t.cost_usdc for t in res.trades)
            for res in self.daily_results
        ]) if self.daily_results else 0

        stats.avg_positions_per_day = np.mean([
            len(res.trades)
            for res in self.daily_results
        ]) if self.daily_results else 0

        return stats


def compute_prev7(daily_max: dict) -> dict:
    """
    Calcula média dos últimos 7 dias para cada data.
    Retorna dict com data -> média dos 7 dias anteriores.
    """
    dates = sorted(daily_max.keys())
    prev7 = {}
    for i, d in enumerate(dates):
        if i == 0:
            prev7[d] = daily_max[d]
        else:
            window = [daily_max[date] for date in dates[max(0, i - 7):i]]
            prev7[d] = float(np.mean(window)) if window else daily_max[d]
    return prev7


# ═════════════════════════════════════════════════════
#  RENDERING
# ═════════════════════════════════════════════════════

def print_stats(stats: BacktestStats):
    """Imprime estatísticas em formato Rich."""
    _console.print()
    _console.print(Rule(title="[bold white]BACKTEST RESULTS[/bold white]", style="cyan"))
    _console.print()

    t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value", style="bold white", width=15)

    t.add_row("Total Days", str(stats.total_days))
    t.add_row("Total Trades", str(stats.total_trades))
    t.add_row("Wins / Losses", f"{stats.wins} / {stats.losses}")
    t.add_row("Stop-Losses", f"{C['yellow']}{stats.stop_losses}{reset}")

    _console.print(t)
    _console.print()

    t2 = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t2.add_column("Metric", style="dim", width=28)
    t2.add_column("Value", style="bold white", width=15)

    pnl_color = "green" if stats.total_pnl > 0 else "red"
    t2.add_row("Starting Capital", f"${stats.starting_capital:.2f}")
    t2.add_row("Final Capital", f"${stats.final_capital:.2f}")
    t2.add_row("Total PnL", f"[{pnl_color}]${stats.total_pnl:+.2f}[/{pnl_color}]")
    t2.add_row("ROI", f"[{pnl_color}]{stats.roi_pct:+.1f}%[/{pnl_color}]")
    t2.add_row("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
    t2.add_row("Max Drawdown", f"[red]{stats.max_drawdown_eur:.2f}€ ({stats.max_drawdown_pct:.1f}%)[reset]")

    _console.print(t2)
    _console.print()

    t3 = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    t3.add_column("Metric", style="dim", width=28)
    t3.add_column("Value", style="bold white", width=15)

    t3.add_row("Win Rate", f"{stats.win_rate:.1f}%")
    t3.add_row("Avg Win", f"${stats.avg_win:.2f}")
    t3.add_row("Avg Loss", f"${stats.avg_loss:.2f}")
    t3.add_row("Max Win", f"${stats.max_win:.2f}")
    t3.add_row("Max Loss", f"${stats.max_loss:.2f}")
    t3.add_row("Avg Duration (slots)", f"{stats.avg_duration_slots:.1f}")

    _console.print(t3)


# ═════════════════════════════════════════════════════
#  EXPORT FOR OPTUNA
# ═════════════════════════════════════════════════════

def run_backtest_for_optuna(config: StrategyConfig) -> Dict[str, float]:
    """
    Função objective para Optuna.

    Recebe config, corre backtest, retorna métricas.
    """
    backtester = UnifiedBacktester(config=config, silent=True)

    # Usar datas padrão para otimização
    end_date = date.today() - timedelta(days=1)
    start_date = date(end_date.year - 3, 1, 1)

    stats = backtester.run_backtest(start_date, end_date)

    return {
        "roi_pct": stats.roi_pct,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "sharpe_ratio": stats.sharpe_ratio,
        "total_trades": stats.total_trades,
        "win_rate": stats.win_rate,
        "stop_losses": stats.stop_losses,
    }


# ═════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════

def main():
    parser = ArgumentParser(description="Unified Munich Backtester")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["single", "phased"], default="single")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--years", type=int, default=5, help="Years to backtest")
    parser.add_argument("--silent", action="store_true", help="Silent mode (for Optuna)")
    args = parser.parse_args()

    # Datas
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)

    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = date(end_date.year - args.years, 1, 1)

    # Config
    config = load_config(Path(args.config) if args.config else None)
    config.entry.mode = args.mode

    print(f"\n  {B}{C['cyan']}=== Unified Backtester ==={R}\n")
    print(f"  Dates: {start_date} → {end_date}")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {args.config or 'default'}")

    # Carregar modelos
    if not args.silent:
        print(f"\n  {C['cyan']}Loading models...{reset}")
    models = load_models()

    # Correr backtest
    backtester = UnifiedBacktester(config=config, models=models, silent=args.silent)
    stats = backtester.run_backtest(start_date, end_date)

    # Imprimir resultados
    if not args.silent:
        print_stats(stats)

        # Guardar
        BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = BACKTEST_RESULTS_DIR / f"unified_{args.mode}_{start_date.isoformat()}_{end_date.isoformat()}.json"

        results = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "mode": args.mode,
            "stats": {
                "total_days": stats.total_days,
                "total_trades": stats.total_trades,
                "wins": stats.wins,
                "losses": stats.losses,
                "stop_losses": stats.stop_losses,
                "roi_pct": float(stats.roi_pct),
                "max_drawdown_pct": float(stats.max_drawdown_pct),
                "sharpe_ratio": float(stats.sharpe_ratio),
                "win_rate": float(stats.win_rate),
                "avg_win": float(stats.avg_win),
                "avg_loss": float(stats.avg_loss),
            },
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  {C['green']}✓[reset] Results saved: {output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    main()
