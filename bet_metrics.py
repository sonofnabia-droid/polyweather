"""
bet_metrics.py — Motor de Estatísticas de Trading (UNIFIED2 + RICH)
=====================
Calcula métricas avançadas de performance para Live e Backtest.
Projetado para Polymarket: wins pagam (1-ask)/ask, perdas custam -100%.

Agora com Dashboard Rich integrado.
"""

import math
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

# Instanciar consola Rich uma vez para reuse
_CON = Console()

class BetMetrics:
    """
    Calcula métricas de performance a partir de uma lista de dicionários de trades.
    Formato esperado por trade:
        {
            "pnl_usd": 15.0,      # Lucro/prejuízo em USD
            "pnl_pct": 300.0,     # Lucro/prejuízo em percentagem
            "outcome": "won",     # "won" ou "lost"
            "ask": 0.25,         # Preço de entrada
            "date": "2024-01-15" # Data do trade
        }
    """
    def __init__(self, trades: list[dict]):
        self.trades = trades

    def summary(self) -> dict:
        if not self.trades:
            return self._empty_summary()

        wins = [t for t in self.trades if t.get("outcome") == "won"]
        losses = [t for t in self.trades if t.get("outcome") == "lost"]
        
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses
        
        win_rate = (n_wins / n_total * 100) if n_total > 0 else 0.0
        
        total_pnl_usd = sum(t.get("pnl_usd", 0.0) for t in self.trades)
        total_invested = sum(t.get("size_usdc", 5.0) for t in self.trades)
        total_pnl_pct = (total_pnl_usd / total_invested * 100) if total_invested > 0 else 0.0
        
        gross_profit = sum(t.get("pnl_usd", 0.0) for t in wins)
        gross_loss = abs(sum(t.get("pnl_usd", 0.0) for t in losses))
        
        avg_win_usd = gross_profit / n_wins if n_wins > 0 else 0.0
        avg_loss_usd = gross_loss / n_losses if n_losses > 0 else 0.0
        
        payoff_ratio = (avg_win_usd / avg_loss_usd) if avg_loss_usd > 0 else float('inf')
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        ev_per_trade = total_pnl_usd / n_total if n_total > 0 else 0.0
        avg_ask = sum(t.get("ask", 0.0) for t in self.trades) / n_total if n_total > 0 else 0.0

        equity_curve = self._build_equity_curve()
        sharpe = self._sharpe_ratio(equity_curve)
        sortino = self._sortino_ratio(equity_curve)
        max_dd, max_dd_pct = self._max_drawdown(equity_curve)

        return {
            "n_total": n_total, "n_wins": n_wins, "n_losses": n_losses,
            "win_rate": round(win_rate, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "Inf",
            "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float('inf') else "Inf",
            "avg_win_usd": round(avg_win_usd, 2),
            "avg_loss_usd": round(avg_loss_usd, 2),
            "ev_per_trade": round(ev_per_trade, 2),
            "avg_ask": round(avg_ask, 4),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_usd": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
        }

    def _empty_summary(self) -> dict:
        return {k: 0.0 if isinstance(v, float) else v for k, v in self.summary().items()}

    # ── MÉTODOS DE MATEMÁTICA (Intocados) ──────────
    def _build_equity_curve(self, initial=1000.0) -> list[float]:
        curve = [initial]
        for t in self.trades:
            curve.append(curve[-1] + t.get("pnl_usd", 0.0))
        return curve

    def _daily_returns(self, equity_curve: list[float]) -> list[float]:
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i-1]
            returns.append((equity_curve[i] - prev) / prev if prev > 0 else 0.0)
        return returns

    def _sharpe_ratio(self, equity_curve: list[float], risk_free=0.0) -> float:
        returns = self._daily_returns(equity_curve)
        if len(returns) < 2: return 0.0
        avg_ret = sum(returns) / len(returns)
        std_ret = (sum((r - avg_ret)**2 for r in returns) / (len(returns) - 1))**0.5
        if std_ret == 0: return 0.0
        return ((avg_ret - risk_free) * 365) / (std_ret * math.sqrt(365))

    def _sortino_ratio(self, equity_curve: list[float], risk_free=0.0) -> float:
        returns = self._daily_returns(equity_curve)
        if len(returns) < 2: return 0.0
        avg_ret = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside: return float('inf') if avg_ret > 0 else 0.0
        downside_std = (sum(r**2 for r in downside) / len(downside))**0.5
        if downside_std == 0: return 0.0
        return ((avg_ret - risk_free) * 365) / (downside_std * math.sqrt(365))

    def _max_drawdown(self, equity_curve: list[float]) -> tuple[float, float]:
        peak = equity_curve[0]
        max_dd, max_dd_pct = 0.0, 0.0
        for val in equity_curve:
            if val > peak: peak = val
            dd = peak - val
            dd_pct = (dd / peak * 100) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd, max_dd_pct = dd, dd_pct
        return max_dd, max_dd_pct

    # ── NOVA VISUALIZAÇÃO RICH ──────────────────────
    def print_report(self, title="RELATÓRIO DE PERFORMANCE"):
        """Imprime relatório formatado no terminal usando Rich."""
        s = self.summary()
        
        _CON.print()
        _CON.rule(f"[bold bright_cyan]{title}[/bold bright_cyan]")
        _CON.print()

        # 1. KPIs Principais (Verde/Vermelho dinâmico)
        kpi_table = Table(box=box.DOUBLE_EDGE, border_style="bright_blue", show_header=False, padding=(0, 2))
        kpi_table.add_column("Métrica", style="bold cyan", width=22)
        kpi_table.add_column("Valor", width=20)
        
        def _fmt_usd(v):
            return f"[bold green]+${v:.2f}[/bold green]" if v > 0 else f"[bold red]${v:.2f}[/bold red]"
        
        def _fmt_pct(v):
            return f"[bold green]{v:.2f}%[/bold green]" if v > 0 else f"[bold red]{v:.2f}%[/bold red]"
            
        kpi_table.add_row("Total P&L", _fmt_usd(s['total_pnl_usd']))
        kpi_table.add_row("Win Rate", f"[bold white]{s['win_rate']:.1f}%[/bold white]")
        kpi_table.add_row("Trades (W/L)", f"[green]{s['n_wins']}W[/green] / [red]{s['n_losses']}L[/red]")
        kpi_table.add_row("EV por Trade", _fmt_usd(s['ev_per_trade']))
        kpi_table.add_row("Avg Ask", f"{s['avg_ask']*100:.1f}¢ (risco médio)")
        _CON.print(Panel(kpi_table, title="[bold]Performance Geral[/bold]"))
        _CON.print()

        # 2. Gestão de Risco
        risk_table = Table(box=box.SIMPLE_HEAVY, border_style="red", title="Gestão de Risco", title_style="bold red")
        risk_table.add_column("Métrica", style="dim", width=22)
        risk_table.add_column("Valor", justify="right", width=15)
        
        sharpe_color = "bold green" if s['sharpe_ratio'] > 1.0 else ("yellow" if s['sharpe_ratio'] > 0 else "bold red")
        sortino_color = "bold green" if s['sortino_ratio'] > 1.5 else ("yellow" if s['sortino_ratio'] > 0 else "bold red")
        
        risk_table.add_row("Sharpe Ratio", f"[{sharpe_color}]{s['sharpe_ratio']:.2f}[/{sharpe_color}]")
        risk_table.add_row("Sortino Ratio", f"[{sortino_color}]{s['sortino_ratio']:.2f}[/{sortino_color}]")
        risk_table.add_row("Max Drawdown", f"[bold red]${s['max_drawdown_usd']:.2f} ({s['max_drawdown_pct']:.1f}%)[/bold red]")
        risk_table.add_row("Profit Factor", f"[bold white]{s['profit_factor']}[/bold white]")
        risk_table.add_row("Payoff Ratio", f"[bold white]{s['payoff_ratio']}[/bold white]")
        _CON.print(risk_table)
        _CON.print()

        # 3. Resultados Detalhados
        det_table = Table(box=box.SIMPLE, border_style="dim", title="Análise de Wins/Losses", title_style="bold", show_lines=False)
        det_table.add_column("Vitórias", justify="right", style="green")
        det_table.add_column("Derrotas", justify="right", style="red")
        det_table.add_row(f"Total ganho: +${s['avg_win_usd'] * s['n_wins']:.2f}", f"Total perdido: -${s['avg_loss_usd'] * s['n_losses']:.2f}")
        det_table.add_row(f"Média ganho: [green]+${s['avg_win_usd']:.2f}[/green]", f"Média perdida: [red]-${s['avg_loss_usd']:.2f}[/red]")
        _CON.print(det_table)
        _CON.print()
