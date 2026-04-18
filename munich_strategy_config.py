"""
munich_strategy_config.py — UNIFIED
================================
Configurações dinâmicas da estratégia (backtest e live).

Todas as configurações são carregadas de ficheiros JSON, permitindo:
- Parâmetros otimizados pelo Optuna serem usados tanto no backtest como no live
- Ajustes sem modificar código
- Versionamento de configurações

Paths:
- default_config.json  - Configurações base (defaults)
- optimized_config.json - Configurações otimizadas pelo Optuna (sobrescreve defaults)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Importar paths do munich_config para consistência
from munich_config import (
    DEFAULT_STRATEGY_CONFIG, OPTIMIZED_STRATEGY_CONFIG, R, B, C
)

DEFAULT_CONFIG_PATH = DEFAULT_STRATEGY_CONFIG
OPTIMIZED_CONFIG_PATH = OPTIMIZED_STRATEGY_CONFIG


# ═════════════════════════════════════════════════════
#  DATA CLASSES PARA CONFIGURAÇÃO
# ═════════════════════════════════════════════════════

@dataclass
class StopLossConfig:
    """Configuração do stop-loss baseado em temperatura e probabilidade."""

    # Stop-loss por TEMPERATURA (em graus)
    # Se a temperatura se mover X graus contra a nossa posição, saímos
    temp_threshold: float = 0.5

    # Stop-loss por PROBABILIDADE (p_ensemble)
    # Se a probabilidade do modelo descer abaixo deste threshold, saímos
    prob_threshold: float = 0.60

    # Qual stop-loss usar: "temperature", "probability", "both", "most_restrictive"
    mode: str = "both"  # "temperature", "probability", "both", "most_restrictive"

    # Usar o mais restritivo dos dois quando mode = "both" ou "most_restrictive"
    use_most_restrictive: bool = True

    # Tolerância antes de activar stop-loss (evita oscilações)
    min_pnl_to_exit: float = -2.0  # Só sai se perda >= 2€ (evita sair em perdas pequenas)


@dataclass
class EntryConfig:
    """Configuração de entrada (parâmetros de decisão de compra)."""

    # Modo de entrada: "single" ou "phased"
    mode: str = "single"

    # SINGLE mode
    single_threshold: float = 0.80  # p_ensemble >= 80% para comprar

    # PHASED mode
    phased_parcel_size: float = 5.0

    # Parcela 1: Value Early (manhã + forecast + mercado)
    p1_hour_min: int = 10
    p1_hour_max: int = 12
    p1_prob_min: float = 0.30  # Modelo tem alguma confiança
    p1_prob_max: float = 0.65  # Pico ainda não ocorreu

    # Parcela 2: Dupla confirmação (modelo + mercado)
    p2_threshold: float = 0.70

    # Parcela 3: Alta confiança
    p3_threshold: float = 0.85

    # Tolerância de temperatura para confirmação de mercado
    temp_tolerance: float = 1.0


@dataclass
class PositionConfig:
    """Configuração de gestão de posição."""

    # Tamanho da aposta (USDC)
    bet_size: float = 15.0  # SINGLE mode (ou 3x5 em PHASED)

    # Intervalo mínimo entre compras (minutos)
    cooldown_minutes: float = 30.0

    # Limites de odds (ask price em Polymarket = 0-1)
    min_ask: float = 0.05  # Mínimo 5¢ (evitar odds muito baixas)
    max_ask: float = 0.95  # Máximo 95¢ (evitar overpay)

    # Limite de perda diária (USDC)
    max_daily_loss: float = 50.0

    # Número máximo de trades por dia
    max_trades_per_day: int = 3


@dataclass
class OptimizationConfig:
    """Configuração para Optuna optimization."""

    # Número de trials
    n_trials: int = 50

    # Parâmetros para otimizar
    optimize_temp_threshold: bool = True
    optimize_prob_threshold: bool = True
    optimize_single_threshold: bool = True
    optimize_p2_threshold: bool = True
    optimize_p3_threshold: bool = True

    # Ranges para otimização (ajustados para evitar sobreposição)
    temp_threshold_range: tuple = (0.2, 2.0)
    prob_threshold_range: tuple = (0.50, 0.85)
    single_threshold_range: tuple = (0.75, 0.95)  # single deve ser maior que p2
    p2_threshold_range: tuple = (0.55, 0.75)     # p2 deve ser menor que single e maior que p3
    p3_threshold_range: tuple = (0.80, 0.95)    # p3 deve ser maior que p2 (para phased entry)

    # Função objetivo: maximize (1) ROI, (2) Sharpe, (3) custom_score
    objective: str = "custom_score"  # "roi", "sharpe", "custom_score"

    # Penalização por drawdown no custom_score
    drawdown_penalty_factor: float = 1.5

    # Pruning: parar trials com drawdown > X%
    prune_drawdown_pct: float = 5000.0


@dataclass
class StrategyConfig:
    """Configuração completa da estratégia."""

    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Metadados
    version: str = "1.0"
    created_at: str = ""
    optimized_at: str = ""
    optimization_score: float = 0.0


# ═════════════════════════════════════════════════════
#  CARREGAMENTO / SALVAMENTO
# ═════════════════════════════════════════════════════

# Importar paths do munich_config para consistência
from munich_config import DEFAULT_STRATEGY_CONFIG, OPTIMIZED_STRATEGY_CONFIG

DEFAULT_CONFIG_PATH = DEFAULT_STRATEGY_CONFIG
OPTIMIZED_CONFIG_PATH = OPTIMIZED_STRATEGY_CONFIG


def load_config(path: Optional[Path] = None) -> StrategyConfig:
    """
    Carrega configuração de ficheiro JSON.

    Se path for None, tenta carregar optimized_config.json, depois default_config.json.
    Se nenhum existir, retorna config com defaults.
    """
    from datetime import datetime

    paths_to_try = []
    if path is not None:
        paths_to_try.append(path)
    paths_to_try.append(OPTIMIZED_CONFIG_PATH)
    paths_to_try.append(DEFAULT_CONFIG_PATH)

    for config_path in paths_to_try:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)

                # Criar config a partir dos dados
                sl_data = data.get("stop_loss", {})
                entry_data = data.get("entry", {})
                pos_data = data.get("position", {})
                opt_data = data.get("optimization", {})

                config = StrategyConfig(
                    stop_loss=StopLossConfig(**sl_data),
                    entry=EntryConfig(**entry_data),
                    position=PositionConfig(**pos_data),
                    optimization=OptimizationConfig(**opt_data),
                    version=data.get("version", "1.0"),
                    created_at=data.get("created_at", ""),
                    optimized_at=data.get("optimized_at", ""),
                    optimization_score=data.get("optimization_score", 0.0),
                )

                source = "OPTIMIZED" if config_path == OPTIMIZED_CONFIG_PATH else "DEFAULT"
                print(f"  [green]✓[reset] Config loaded: {config_path} ({source})")
                return config
            except Exception as e:
                print(f"  [yellow]⚠[reset] Failed to load {config_path}: {e}")
                continue

    # Retornar config com defaults
    config = StrategyConfig()
    config.created_at = datetime.now().isoformat()
    print(f"  [dim]Using default config (no file found)[reset]")
    return config


def save_config(config: StrategyConfig, path: Path = OPTIMIZED_CONFIG_PATH,
                overwrite_default: bool = False) -> None:
    """
    Guarda configuração em ficheiro JSON.

    Args:
        config: Configuração a guardar
        path: Caminho onde guardar (default: optimized_config.json)
        overwrite_default: Se True, guarda em default_config.json (cuidado!)
    """
    from datetime import datetime

    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    # Atualizar timestamp de otimização
    if path == OPTIMIZED_CONFIG_PATH or overwrite_default:
        data["optimized_at"] = datetime.now().isoformat()

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  [green]✓[reset] Config saved: {path}")


def create_default_config() -> StrategyConfig:
    """Cria configuração default e guarda em ficheiro."""
    config = StrategyConfig()
    save_config(config, DEFAULT_CONFIG_PATH, overwrite_default=True)
    return config


# ═════════════════════════════════════════════════════
#  CONVERTE PARA DICIONÁRIO (para Optuna)
# ═════════════════════════════════════════════════════

def config_to_params_dict(config: StrategyConfig) -> dict:
    """
    Converte config para dicionário de parâmetros (útil para Optuna).
    Retorna apenas os parâmetros relevantes para otimização.
    """
    return {
        "temp_threshold": config.stop_loss.temp_threshold,
        "prob_threshold": config.stop_loss.prob_threshold,
        "single_threshold": config.entry.single_threshold,
        "p2_threshold": config.entry.p2_threshold,
        "p3_threshold": config.entry.p3_threshold,
    }


def params_dict_to_config(params: dict, base_config: StrategyConfig) -> StrategyConfig:
    """
    Atualiza config com parâmetros do Optuna.
    Retorna nova config (não modifica a original).
    """
    from dataclasses import replace

    # Atualizar stop_loss
    if "temp_threshold" in params:
        sl = replace(base_config.stop_loss, temp_threshold=params["temp_threshold"])
    else:
        sl = base_config.stop_loss

    if "prob_threshold" in params:
        sl = replace(sl, prob_threshold=params["prob_threshold"])

    # Atualizar entry
    if "single_threshold" in params:
        entry = replace(base_config.entry, single_threshold=params["single_threshold"])
    else:
        entry = base_config.entry

    if "p2_threshold" in params:
        entry = replace(entry, p2_threshold=params["p2_threshold"])

    if "p3_threshold" in params:
        entry = replace(entry, p3_threshold=params["p3_threshold"])

    return replace(base_config, stop_loss=sl, entry=entry)


# ═════════════════════════════════════════════════════
#  CLI HELPER
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from munich_config import C, R, DIM

    print(f"\n  {B}{C['cyan']}=== Munich Strategy Config ==={R}\n")

    if len(sys.argv) > 1 and sys.argv[1] == "create-default":
        create_default_config()
        print(f"\n  {C['green']}✓[reset] Default config created at: {DEFAULT_CONFIG_PATH}")
    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        config = load_config()
        print(f"\n  {C['yellow']}Current Config:{reset}")
        print(json.dumps(asdict(config), indent=2))
    else:
        config = load_config()
        print(f"\n  {C['cyan']}Stop-Loss:{reset}")
        print(f"    temp_threshold: {config.stop_loss.temp_threshold}°C")
        print(f"    prob_threshold: {config.stop_loss.prob_threshold*100:.0f}%")
        print(f"    mode: {config.stop_loss.mode}")

        print(f"\n  {C['cyan']}Entry ({config.entry.mode.upper()}):{reset}")
        if config.entry.mode == "single":
            print(f"    single_threshold: {config.entry.single_threshold*100:.0f}%")
        else:
            print(f"    parcel_size: ${config.entry.phased_parcel_size:.2f}")
            print(f"    P1: {config.entry.p1_hour_min}h-{config.entry.p1_hour_max}h, "
                  f"p=[{config.entry.p1_prob_min*100:.0f}%, {config.entry.p1_prob_max*100:.0f}%]")
            print(f"    P2: {config.entry.p2_threshold*100:.0f}%")
            print(f"    P3: {config.entry.p3_threshold*100:.0f}%")

        print(f"\n  {C['cyan']}Position:{reset}")
        print(f"    bet_size: ${config.position.bet_size:.2f}")
        print(f"    cooldown: {config.position.cooldown_minutes:.0f} min")
        print(f"    max_daily_loss: ${config.position.max_daily_loss:.2f}")
