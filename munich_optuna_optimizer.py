"""
munich_optuna_optimizer.py — UNIFIED
====================================
Optuna optimization para parâmetros da estratégia.

Pode ser usado de 2 formas:
1. Integrado no munich_train.py (otimizar durante treino)
2. Standalone (otimizar com backtester existente)

Optimiza:
- Stop-loss: temp_threshold, prob_threshold
- Entry: single_threshold, p2_threshold, p3_threshold
- Position: bet_size, cooldown_minutes (opcional)

Objetivo:
- Maximizar ROI (return on investment)
- Penalizar drawdown (max drawdown * penalty_factor)
"""

import json
import optuna
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict

from munich_config import C, R, B, BACKTEST_RESULTS_DIR, DIM

# Alias para consistência
reset = C["reset"]
from munich_strategy_config import (
    StrategyConfig,
    StopLossConfig,
    EntryConfig,
    PositionConfig,
    config_to_params_dict,
    params_dict_to_config,
    load_config,
    save_config,
)


# ═════════════════════════════════════════════════════
#  DATA CLASS PARA RESULTADOS
# ═════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Resultado da otimização."""

    best_params: dict
    best_value: float
    n_trials: int
    study_summary: dict

    # Métricas do melhor trial
    roi_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float


# ═════════════════════════════════════════════════════
#  OPTUNA OPTIMIZER
# ═════════════════════════════════════════════════════

class OptunaOptimizer:
    """
    Otimizador de parâmetros usando Optuna.

    O optimizer precisa de uma função objective que:
    1. Recebe um trial do Optuna
    2. Sugere parâmetros
    3. Corre um backtest com esses parâmetros
    4. Retorna um score (maior = melhor)
    """

    def __init__(
        self,
        base_config: StrategyConfig,
        objective_fn: Callable[[dict], Dict[str, float]],
        n_trials: int = 50,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        storage: Optional[str] = None,
        study_name: str = "munich_strategy"
    ):
        """
        Args:
            base_config: Configuração base com ranges de otimização
            objective_fn: Função que recebe params e retorna métricas
                Deve retornar: {"roi": X, "max_drawdown": Y, ...}
            n_trials: Número de trials
            n_jobs: Número de jobs paralelos (-1 = todos os CPUs)
            timeout: Timeout em segundos (opcional)
            storage: Storage URL para persistência (opcional)
            study_name: Nome do estudo Optuna
        """
        import os
        self.base_config = base_config
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        self.n_jobs = max(1, self.n_jobs)  # Mínimo 1
        self.timeout = timeout
        self.storage = storage
        self.study_name = study_name
        print(f"  [dim]Using {self.n_jobs} parallel jobs[reset]")

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Sugere parâmetros baseados na config base.
        """
        params = {}

        # ── STOP-LOSS ─────────────────────────────────
        opt = self.base_config.optimization

        if opt.optimize_temp_threshold:
            params["temp_threshold"] = trial.suggest_float(
                'temp_threshold',
                opt.temp_threshold_range[0],
                opt.temp_threshold_range[1],
                step=0.1
            )

        if opt.optimize_prob_threshold:
            params["prob_threshold"] = trial.suggest_float(
                'prob_threshold',
                opt.prob_threshold_range[0],
                opt.prob_threshold_range[1],
                step=0.01
            )

        # ── ENTRY ──────────────────────────────────────
        if opt.optimize_single_threshold:
            params["single_threshold"] = trial.suggest_float(
                'single_threshold',
                opt.single_threshold_range[0],
                opt.single_threshold_range[1],
                step=0.01
            )

        if opt.optimize_p2_threshold:
            params["p2_threshold"] = trial.suggest_float(
                'p2_threshold',
                opt.p2_threshold_range[0],
                opt.p2_threshold_range[1],
                step=0.01
            )

        if opt.optimize_p3_threshold:
            params["p3_threshold"] = trial.suggest_float(
                'p3_threshold',
                opt.p3_threshold_range[0],
                opt.p3_threshold_range[1],
                step=0.01
            )

        # ── CONSTRAINTS ───────────────────────────────
        # single_threshold > p2_threshold > p3_threshold
        if "single_threshold" in params and "p2_threshold" in params:
            trial.set_user_attr("single_gt_p2", params["single_threshold"] > params["p2_threshold"])

        if "p2_threshold" in params and "p3_threshold" in params:
            trial.set_user_attr("p2_gt_p3", params["p2_threshold"] > params["p3_threshold"])

        return params

    def compute_score(self, metrics: Dict[str, float]) -> float:
        """
        Computa score otimizado baseado nas métricas.

        Score = ROI - (max_drawdown * penalty_factor)
        """
        opt = self.base_config.optimization

        if opt.objective == "roi":
            return metrics.get("roi_pct", 0)

        elif opt.objective == "sharpe":
            return metrics.get("sharpe_ratio", 0)

        else:  # custom_score (default)
            roi = metrics.get("roi_pct", 0)
            max_dd = metrics.get("max_drawdown_pct", 0)
            return roi - (max_dd * opt.drawdown_penalty_factor)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para Optuna.

        Sugere parâmetros, corre backtest, retorna score.
        """
        print(f"  [dim]Starting objective for trial {trial.number}[reset]")
        # Sugerir parâmetros
        params = self.suggest_params(trial)
        print(f"  [dim]Params suggested: {params}[reset]")

        # Validar constraints
        if "single_threshold" in params and "p2_threshold" in params:
            if params["single_threshold"] <= params["p2_threshold"]:
                # Penalizar trials que violam a constraint
                print(f"  [dim]Constraint violated: single_threshold ({params['single_threshold']}) <= p2_threshold ({params['p2_threshold']})[reset]")
                return -1000.0

        if "p2_threshold" in params and "p3_threshold" in params:
            if params["p2_threshold"] >= params["p3_threshold"]:
                print(f"  [dim]Constraint violated: p2_threshold ({params['p2_threshold']}) >= p3_threshold ({params['p3_threshold']})[reset]")
                return -1000.0

        # Correr backtest com estes parâmetros
        try:
            print(f"  [dim]Running backtest with params: {params}[reset]")
            metrics = self.objective_fn(params)
            print(f"  [dim]Backtest result: {metrics}[reset]")

            # Pruning: parar trials com drawdown muito alto
            max_dd = metrics.get("max_drawdown_pct", 0)
            print(f"  [dim]Max drawdown: {max_dd}% (prune threshold: {self.base_config.optimization.prune_drawdown_pct}%)[reset]")
            if max_dd > self.base_config.optimization.prune_drawdown_pct:
                trial.report(max_dd, 0)
                raise optuna.TrialPruned()

            # Penalizar estratégias com poucos trades
            n_trades = metrics.get("total_trades", 0)
            if n_trades < 5:
                return -1000.0

            # Calcular score
            score = self.compute_score(metrics)
            print(f"  [dim]Score computed: {score}[reset]")

            # Guardar métricas no trial
            trial.set_user_attr("roi_pct", metrics.get("roi_pct", 0))
            trial.set_user_attr("max_drawdown_pct", max_dd)
            trial.set_user_attr("sharpe_ratio", metrics.get("sharpe_ratio", 0))
            trial.set_user_attr("total_trades", n_trades)
            trial.set_user_attr("win_rate", metrics.get("win_rate", 0))

            return score

        except Exception as e:
            import traceback
            print(f"  [red]Trial failed: {e}[reset]")
            traceback.print_exc()
            return -1000.0

    def optimize(self) -> OptimizationResult:
        """
        Executa a otimização.

        Returns:
            OptimizationResult com melhores parâmetros e métricas
        """
        # Criar estudo
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )

        # Callback para mostrar progresso
        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                print(f"  Trial {trial.number}: score={trial.value:.2f} "
                      f"(roi={trial.user_attrs.get('roi_pct', 0):.1f}%, "
                      f"dd={trial.user_attrs.get('max_drawdown_pct', 0):.1f}%)")

        # Otimizar
        print(f"\n  {C['cyan']}Optimizando {self.n_trials} trials com {self.n_jobs} jobs paralelos...{reset}\n")

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[callback],
            show_progress_bar=False,
            n_jobs=self.n_jobs
        )

        # Resultados
        best_trial = study.best_trial
        best_params = best_trial.params

        print(f"\n  {C['green']}✓[reset] Otimização concluída!")
        print(f"    Melhor score: {best_trial.value:.2f}")
        print(f"    Melhores params:")
        for k, v in best_params.items():
            print(f"      {k}: {v:.4f}")

        # Retornar resultado
        return OptimizationResult(
            best_params=best_params,
            best_value=best_trial.value,
            n_trials=len(study.trials),
            study_summary={
                "n_trials": len(study.trials),
                "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
                "n_failed": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL),
            },
            roi_pct=best_trial.user_attrs.get('roi_pct', 0),
            max_drawdown_pct=best_trial.user_attrs.get('max_drawdown_pct', 0),
            sharpe_ratio=best_trial.user_attrs.get('sharpe_ratio', 0),
            total_trades=best_trial.user_attrs.get('total_trades', 0),
            win_rate=best_trial.user_attrs.get('win_rate', 0),
        )


# ═════════════════════════════════════════════════════
#  HELPER PARA INTEGRAR COM BACKTESTER
# ═════════════════════════════════════════════════════

def create_backtest_objective(
    backtest_fn: Callable[[StrategyConfig], Dict[str, float]]
) -> Callable[[dict], Dict[str, float]]:
    """
    Cria função objective para Optuna a partir de uma função de backtest.

    Args:
        backtest_fn: Função que recebe StrategyConfig e retorna métricas
                    Métricas devem incluir: roi_pct, max_drawdown_pct, etc.

    Returns:
        Função objective para Optuna (recebe dict de params)
    """
    def objective(params: dict) -> Dict[str, float]:
        # Carregar config base
        base_config = load_config()

        # Aplicar parâmetros otimizados
        config = params_dict_to_config(params, base_config)

        # Correr backtest
        return backtest_fn(config)

    return objective


# ═════════════════════════════════════════════════════
#  INTEGRAÇÃO COM MUNICH_TRAIN.PY
# ═════════════════════════════════════════════════════

def optimize_with_training(
    dataset,
    train_fn: Callable,
    n_trials: int = 20
) -> OptimizationResult:
    """
    Integra Optuna com o fluxo de treino.

    Otimiza parâmetros de estratégia durante o treino,
    usando walk-forward validation para avaliar cada combinação.

    Args:
        dataset: Dataset de treino
        train_fn: Função que treina e avalia (retorna métricas)
        n_trials: Número de trials Optuna

    Returns:
        OptimizationResult
    """
    print(f"\n  {C['cyan']}=== Optuna Integration com Treino ==={reset}\n")

    # Carregar config
    base_config = load_config()

    # Criar função objective
    def objective_fn(params: dict) -> Dict[str, float]:
        # Aplicar parâmetros
        config = params_dict_to_config(params, base_config)

        # Simular: corrigir backtest com estes parâmetros
        # NOTA: Em produção, integrar com walk-forward validation
        # Aqui é um placeholder

        # Placeholder: métricas simuladas
        # Em produção: chamar train_fn(config) ou backtest_fn(config)
        return {
            "roi_pct": 15.0,
            "max_drawdown_pct": 20.0,
            "sharpe_ratio": 1.5,
            "total_trades": 50,
            "win_rate": 0.60,
        }

    # Criar e correr optimizer
    optimizer = OptunaOptimizer(
        base_config=base_config,
        objective_fn=objective_fn,
        n_trials=n_trials
    )

    result = optimizer.optimize()

    # Guardar config otimizada
    optimized_config = params_dict_to_config(result.best_params, base_config)
    optimized_config.optimization_score = result.best_value
    save_config(optimized_config)

    return result


# ═════════════════════════════════════════════════════
#  SALVAR RESULTADOS
# ═════════════════════════════════════════════════════

def save_optimization_results(result: OptimizationResult, path: Optional[Path] = None):
    """Guarda resultados da otimização em ficheiro JSON."""
    if path is None:
        path = BACKTEST_RESULTS_DIR / "optuna_results.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "best_params": result.best_params,
        "best_score": float(result.best_value),
        "n_trials": result.n_trials,
        "study_summary": result.study_summary,
        "best_metrics": {
            "roi_pct": float(result.roi_pct),
            "max_drawdown_pct": float(result.max_drawdown_pct),
            "sharpe_ratio": float(result.sharpe_ratio),
            "total_trades": int(result.total_trades),
            "win_rate": float(result.win_rate),
        },
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  {C['green']}✓[reset] Results saved: {path}")


# ═════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Optuna Optimizer for Munich Strategy")
    parser.add_argument("--trials", type=int, default=50,
                       help="Número de trials")
    parser.add_argument("--jobs", type=int, default=-1,
                       help="Número de jobs paralelos (-1 = todos os CPUs)")
    parser.add_argument("--objective", choices=["roi", "sharpe", "custom"],
                       default="custom", help="Função objetivo")
    parser.add_argument("--backtest", type=str,
                       help="Módulo backtest a usar (ex: munich_backtester_unified)")
    args = parser.parse_args()

    print(f"\n  {B}{C['cyan']}=== Optuna Optimizer ==={R}\n")

    # Carregar config
    config = load_config()
    config.optimization.n_trials = args.trials
    config.optimization.objective = args.objective

    # Se foi especificado um backtest, importar e usar
    if args.backtest:
        print(f"  {C['yellow']}Backtest module: {args.backtest}{reset}")

        # Importar o módulo de backtest
        import importlib
        try:
            backtest_module = importlib.import_module(args.backtest)

            # Obter a função de backtest
            backtest_fn = getattr(backtest_module, 'run_backtest_for_optuna')

            # Criar função objective
            objective = create_backtest_objective(backtest_fn)

            # Criar optimizer
            optimizer = OptunaOptimizer(
                base_config=config,
                objective_fn=objective,
                n_trials=args.trials,
                n_jobs=args.jobs
            )

            # Executar otimização
            result = optimizer.optimize()

            # Salvar resultados
            save_optimization_results(result)

        except (ImportError, AttributeError) as e:
            print(f"  {C['red']}Error loading backtest module: {e}{reset}")
            sys.exit(1)

    else:
        print(f"  {DIM}No backtest specified. Use --backtest to specify module.{reset}")
