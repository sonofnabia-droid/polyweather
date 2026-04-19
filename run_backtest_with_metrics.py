#!/usr/bin/env python3
"""
run_backtest_with_metrics.py — UNIFIED
=====================================
Backtest completo com métricas avançadas (BetMetrics + Rich).

Este script:
1. Executa o backtest usando o UnifiedBacktester
2. Calcula métricas avançadas usando BetMetrics
3. Apresenta relatório formatado com Rich
4. Guarda resultados em JSON

Uso:
    python run_backtest_with_metrics.py
    python run_backtest_with_metrics.py --start 2020-01-01 --end 2023-12-31
    python run_backtest_with_metrics.py --mode single --years 5
    python run_backtest_with_metrics.py --config optimized_config.json --compare
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

from munich_config import (
    BACKTEST_RESULTS_DIR, C, R, B,
    berlin_date,
)
from munich_strategy_config import (
    load_config, save_config,
)
from munich_backtester_unified import (
    UnifiedBacktester,
    BacktestStats,
    print_stats,
    run_backtest_for_optuna,
)
from bet_metrics import BetMetrics
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

_console = Console(force_terminal=True)


# ═════════════════════════════════════════════════════
#  TRADE CONVERTER
# ═════════════════════════════════════════════════════

def backtest_to_trades(backtest_stats: BacktestStats) -> List[Dict[str, Any]]:
    """
    Converte resultados do backtest para formato esperado por BetMetrics.

    Formato BetMetrics:
        {
            "pnl_usd": 15.0,
            "pnl_pct": 300.0,
            "outcome": "won",
            "ask": 0.25,
            "date": "2024-01-15"
        }
    """
    trades = []

    # NOTA: O BacktestStats não tem trades detalhados
    # Vamos criar trades sintéticos baseados nas estatísticas
    # Em produção, o UnifiedBacktester deveria guardar trades detalhados

    if backtest_stats.total_trades == 0:
        return trades

    # Criar trades baseados em wins e losses
    n_wins = backtest_stats.wins
    n_losses = backtest_stats.losses

    for i in range(n_wins):
        trades.append({
            "pnl_usd": backtest_stats.avg_win,
            "pnl_pct": (backtest_stats.avg_win / 15.0) * 100,  # Assumindo bet_size=15
            "outcome": "won",
            "ask": 0.25,  # Média estimada
            "date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
        })

    for i in range(n_losses):
        trades.append({
            "pnl_usd": -backtest_stats.avg_loss,
            "pnl_pct": -100.0,
            "outcome": "lost",
            "ask": 0.25,
            "date": (date(2020, 1, 1) + timedelta(days=n_wins + i)).isoformat(),
        })

    return trades


# ═════════════════════════════════════════════════════
#  COMPARAÇÃO DE CONFIGURAÇÕES
# ═════════════════════════════════════════════════════

def compare_configs(
    configs: List[str],
    start_date: date,
    end_date: date
) -> None:
    """
    Compara diferentes configurações de estratégia.

    Args:
        configs: Lista de paths para ficheiros de config
        start_date: Data inicial do backtest
        end_date: Data final do backtest
    """
    _console.print()
    _console.print(Rule(title="[bold cyan]COMPARAÇÃO DE CONFIGURAÇÕES[/bold cyan]", style="cyan"))
    _console.print()

    results = []

    for config_path in configs:
        _console.print(f"  {C['yellow']}Testando: {config_path}R")

        config = load_config(Path(config_path))
        backtester = UnifiedBacktester(config=config, silent=True)
        stats = backtester.run_backtest(start_date, end_date)

        results.append({
            "config": config_path,
            "roi_pct": stats.roi_pct,
            "sharpe_ratio": stats.sharpe_ratio,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "win_rate": stats.win_rate,
            "total_trades": stats.total_trades,
            "stop_losses": stats.stop_losses,
        })

    # Imprimir tabela de comparação
    from rich.table import Table

    table = Table(box=C.get("box", "simple"), title="Resultados por Configuração")
    table.add_column("Config", style="cyan")
    table.add_column("ROI %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD %", justify="right")
    table.add_column("Win Rate %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("SL", justify="right")

    for r in results:
        roi_color = "green" if r["roi_pct"] > 0 else "red"
        table.add_row(
            r["config"],
            f"[{roi_color}]{r['roi_pct']:+.1f}%[/{roi_color}]",
            f"{r['sharpe_ratio']:.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
            f"{r['win_rate']:.1f}%",
            f"{r['total_trades']}",
            f"{r['stop_losses']}",
        )

    _console.print(table)
    _console.print()


# ═════════════════════════════════════════════════════
#  RUN BACKTEST WITH METRICS
# ═════════════════════════════════════════════════════

def run_backtest_with_metrics(
    config_path: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    mode: str = "single",
    years: int = 5,
    silent: bool = False
) -> Dict[str, Any]:
    """
    Executa backtest e calcula métricas avançadas.

    Args:
        config_path: Path para ficheiro de config
        start_date: Data inicial
        end_date: Data final
        mode: Modo de entrada (single ou phased)
        years: Número de anos (se não especificar datas)
        silent: Modo silencioso (para Optuna)

    Returns:
        Dict com todas as métricas e resultados
    """
    # Datas
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    if start_date is None:
        start_date = date(end_date.year - years, 1, 1)

    # Config
    config = load_config(config_path)
    config.entry.mode = mode

    if not silent:
        print(f"\n  {B}{C['cyan']}=== Run Backtest with Metrics ===R")
        print(f"  Dates: {start_date} → {end_date}")
        print(f"  Mode: {mode}")
        print(f"  Config: {config_path or 'default'}")

    # Carregar modelos
    from munich_model import load_models

    if not silent:
        print(f"\n  {C['cyan']}Loading models...R")
    models = load_models()

    # Correr backtest
    backtester = UnifiedBacktester(config=config, models=models, silent=silent)
    backtest_stats = backtester.run_backtest(start_date, end_date)

    # Converter para formato BetMetrics
    trades = backtest_to_trades(backtest_stats)

    # Calcular métricas com BetMetrics
    bet_metrics = BetMetrics(trades)
    metrics_summary = bet_metrics.summary()

    if not silent:
        # Imprimir relatório BetMetrics
        bet_metrics.print_report(title="RELATÓRIO DE MÉTRICAS (BetMetrics)")

        # Imprimir estatísticas do backtest
        print_stats(backtest_stats)

    # Guardar resultados
    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BACKTEST_RESULTS_DIR / f"metrics_{mode}_{start_date.isoformat()}_{end_date.isoformat()}.json"

    results = {
        "config": str(config_path) if config_path else "default",
        "mode": mode,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),

        # Backtest Stats
        "backtest_stats": {
            "total_days": backtest_stats.total_days,
            "total_trades": backtest_stats.total_trades,
            "wins": backtest_stats.wins,
            "losses": backtest_stats.losses,
            "stop_losses": backtest_stats.stop_losses,
            "roi_pct": float(backtest_stats.roi_pct),
            "sharpe_ratio": float(backtest_stats.sharpe_ratio),
            "max_drawdown_pct": float(backtest_stats.max_drawdown_pct),
            "win_rate": float(backtest_stats.win_rate),
            "avg_win": float(backtest_stats.avg_win),
            "avg_loss": float(backtest_stats.avg_loss),
        },

        # BetMetrics
        "bet_metrics": metrics_summary,

        # Seasonal stats
        "seasonal_stats": backtest_stats.seasonal_stats,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if not silent:
        print(f"\n  {C['green']}✓[reset] Results saved: {output_path}")

    return results


# ═════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with metrics (UNIFIED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=5, help="Years to backtest")
    parser.add_argument("--mode", choices=["single", "phased"], default="single",
                       help="Entry mode")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument("--compare", type=str, nargs='+',
                       help="Compare multiple configs (list of config files)")

    args = parser.parse_args()

    # Comparação de configs
    if args.compare:
        end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
        start_date = date.fromisoformat(args.start) if args.start else date(end_date.year - args.years, 1, 1)

        compare_configs(
            configs=args.compare,
            start_date=start_date,
            end_date=end_date
        )
        return

    # Parse datas
    end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else date(end_date.year - args.years, 1, 1)

    # Config path
    config_path = Path(args.config) if args.config else None

    # Correr backtest
    results = run_backtest_with_metrics(
        config_path=config_path,
        start_date=start_date,
        end_date=end_date,
        mode=args.mode,
        years=args.years,
        silent=args.silent
    )

    # Resumo final
    if not args.silent:
        print(f"\n  {B}{C['cyan']}=== RESUMO ===R")
        print(f"  ROI: {results['backtest_stats']['roi_pct']:+.1f}%")
        print(f"  Sharpe: {results['backtest_stats']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {results['bet_metrics']['win_rate']:.1f}%")
        print(f"  Max DD: {results['backtest_stats']['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    main()
