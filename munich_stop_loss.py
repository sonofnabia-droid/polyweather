"""
munich_stop_loss.py — UNIFIED
=============================
Módulo de stop-loss para Polymarket (Munich Temperature).

Lógica:
- Baseado em TEMPERATURA: se a temperatura se mover X graus contra nós, saímos
- Baseado em PROBABILIDADE: se o modelo (p_ensemble) descer abaixo de threshold
- Baseado em PnL: se a perda exceder um limite

Usado tanto no backtester como no bot live.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from munich_strategy_config import StopLossConfig


# ═════════════════════════════════════════════════════
#  DATA CLASS PARA POSIÇÃO
# ═════════════════════════════════════════════════════

@dataclass
class Position:
    """Representa uma posição aberta."""

    token_id: str
    bracket_label: str
    entry_temp: float  # Temperatura quando entramos
    entry_ask: float   # Ask price quando entramos (0-1)
    entry_p_ensemble: float  # p_ensemble quando entramos
    entry_time: str   # Timestamp
    shares: float
    cost_usdc: float

    # Para tracking do stop-loss
    worst_temp_seen: float = 0.0  # Pior temperatura vista (contra nós)
    worst_p_seen: float = 0.0  # Menor probabilidade vista
    current_temp: float = 0.0
    current_p_ensemble: float = 0.0

    # Estado
    stop_loss_triggered: bool = False
    exit_reason: str = ""


# ═════════════════════════════════════════════════════
#  STOP-LOSS CHECKER
# ═════════════════════════════════════════════════════

class StopLossChecker:
    """
    Verifica se o stop-loss deve ser activado.

    Em Polymarket, estamos a apostar NO (isto é, que a temperatura
    NÃO vai ultrapassar um certo valor). Portanto:
    - Stop-loss por temperatura: temperatura sobe acima do entry_temp + threshold
    - Stop-loss por probabilidade: p_ensemble desce abaixo de prob_threshold
    """

    def __init__(self, config: StopLossConfig):
        self.config = config

    def check(
        self,
        position: Position,
        current_temp: float,
        current_p: float,
        current_ask: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Verifica se o stop-loss deve ser activado.

        Args:
            position: Posição aberta
            current_temp: Temperatura atual
            current_p: p_ensemble atual
            current_ask: Ask price atual (opcional, para cálculo de PnL)

        Returns:
            (should_trigger, reason, estimated_loss_usdc)
        """
        reasons = []
        triggers = []
        losses = []

        # Actualizar tracking
        position.worst_temp_seen = max(
            position.worst_temp_seen, position.entry_temp
        )
        position.worst_p_seen = min(
            position.worst_p_seen if position.worst_p_seen > 0 else 1.0,
            current_p
        )
        position.current_temp = current_temp
        position.current_p_ensemble = current_p

        # ── STOP-LOSS POR TEMPERATURA ─────────────────────
        if self.config.mode in ("temperature", "both"):
            triggered, reason, loss = self._check_temperature_stop(
                position, current_temp
            )
            triggers.append(triggered)
            reasons.append(reason)
            losses.append(loss)

        # ── STOP-LOSS POR PROBABILIDADE ───────────────────
        if self.config.mode in ("probability", "both"):
            triggered, reason, loss = self._check_probability_stop(
                position, current_p, current_ask
            )
            triggers.append(triggered)
            reasons.append(reason)
            losses.append(loss)

        # ── DECISÃO FINAL ───────────────────────────────
        if any(triggers):
            if self.config.use_most_restrictive and len(triggers) > 1:
                # Usar a razão com maior perda estimada
                idx = losses.index(max(losses))
                return True, reasons[idx], losses[idx]
            else:
                # Usar a primeira razão que triggered
                idx = triggers.index(True)
                return True, reasons[idx], losses[idx]

        return False, "", 0.0

    def _check_temperature_stop(
        self,
        position: Position,
        current_temp: float
    ) -> Tuple[bool, str, float]:
        """
        Verifica stop-loss baseado em temperatura.

        Em Polymarket, entramos NO (isto é, apostamos que a temperatura
        não vai ultrapassar o valor do bracket). Se a temperatura subir
        acima do entry + threshold, estamos a perder.
        """
        threshold = self.config.temp_threshold

        # Movimento adverso da temperatura
        temp_movement = current_temp - position.entry_temp

        if temp_movement >= threshold:
            # Estimativa da perda
            # Simplificação: assumindo linearidade entre movimento e perda
            # Em produção, usar modelo de pricing mais preciso
            loss_pct = min(temp_movement / 5.0, 0.95)  # Max 95% perda
            estimated_loss = position.cost_usdc * loss_pct

            # Só activar se perda >= min_pnl_to_exit
            if estimated_loss >= self.config.min_pnl_to_exit:
                reason = (
                    f"STOP-LOSS TEMP: entry={position.entry_temp}°C, "
                    f"current={current_temp}°C (+{temp_movement:.1f}°C, "
                    f"threshold={threshold}°C). Perda est.: ${estimated_loss:.2f}"
                )
                return True, reason, estimated_loss

        return False, "", 0.0

    def _check_probability_stop(
        self,
        position: Position,
        current_p: float,
        current_ask: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Verifica stop-loss baseado em probabilidade (p_ensemble).

        Se o modelo (p_ensemble) desce abaixo de prob_threshold,
        significa que o modelo agora acha que o pico AINDA NÃO ocorreu,
        então estamos provavelmente a perder.
        """
        threshold = self.config.prob_threshold

        if current_p < threshold:
            # Calcular perda estimada
            if current_ask is not None and current_ask > 0:
                # Se temos o ask atual, podemos calcular a perda real
                loss_pct = (position.entry_ask - current_ask) / position.entry_ask
                estimated_loss = position.cost_usdc * abs(loss_pct)
            else:
                # Estimativa baseada na diferença de probabilidade
                prob_drop = position.entry_p_ensemble - current_p
                loss_pct = min(prob_drop, 0.95)
                estimated_loss = position.cost_usdc * loss_pct

            # Só activar se perda >= min_pnl_to_exit
            if estimated_loss >= self.config.min_pnl_to_exit:
                reason = (
                    f"STOP-LOSS PROB: entry p={position.entry_p_ensemble*100:.0f}%, "
                    f"current p={current_p*100:.0f}% (threshold={threshold*100:.0f}%). "
                    f"Perda est.: ${estimated_loss:.2f}"
                )
                return True, reason, estimated_loss

        return False, "", 0.0

    def get_exit_bracket(
        self,
        position: Position,
        market_brackets: list
    ) -> Optional[dict]:
        """
        Encontra o bracket mais próximo para sair (cashout).

        Tenta encontrar um bracket com bid razoável para vender.
        """
        # Em Polymarket, vender = comprar NO (que é o que temos)
        # Portanto, queremos o mesmo bracket que comprámos

        for bracket in market_brackets:
            if bracket["label"] == position.bracket_label:
                return bracket

        # Se não encontrou, encontrar o mais próximo em temperatura
        return min(
            market_brackets,
            key=lambda b: abs(
                (b.get("temp_lo", 0) + b.get("temp_hi", 99)) / 2
                - position.entry_temp
            )
        )


# ═════════════════════════════════════════════════════
#  POSITION MANAGER
# ═════════════════════════════════════════════════════

class PositionManager:
    """
    Gestor de posições com stop-loss integrado.

    Mantém estado das posições abertas, aplica stop-loss,
    e calcula PnL.
    """

    def __init__(self, config: StopLossConfig):
        self.config = config
        self.sl_checker = StopLossChecker(config)
        self.positions: list[Position] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.stop_losses_triggered = 0

    def add_position(self, position: Position) -> None:
        """Adiciona nova posição."""
        position.worst_temp_seen = position.entry_temp
        position.worst_p_seen = position.entry_p_ensemble
        self.positions.append(position)

    def update_positions(
        self,
        current_temp: float,
        current_p: float,
        current_ask: Optional[float] = None
    ) -> list[Position]:
        """
        Actualiza todas as posições e verifica stop-loss.

        Returns:
            Lista de posições para sair (stop-loss activado)
        """
        to_exit = []

        for position in self.positions:
            if position.stop_loss_triggered:
                continue

            should_exit, reason, loss = self.sl_checker.check(
                position, current_temp, current_p, current_ask
            )

            if should_exit:
                position.stop_loss_triggered = True
                position.exit_reason = reason
                to_exit.append(position)
                self.stop_losses_triggered += 1

        return to_exit

    def remove_position(self, position: Position, final_pnl: float) -> None:
        """
        Remove posição após saída (stop-loss ou natural).
        """
        if position in self.positions:
            self.positions.remove(position)
            self.daily_pnl += final_pnl
            self.daily_trades += 1

    def get_active_positions(self) -> list[Position]:
        """Retorna posições ativas (sem stop-loss)."""
        return [p for p in self.positions if not p.stop_loss_triggered]

    def reset_daily(self) -> None:
        """Reseta contadores diários."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.stop_losses_triggered = 0

    def get_daily_stats(self) -> dict:
        """Retorna estatísticas diárias."""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "stop_losses_triggered": self.stop_losses_triggered,
            "active_positions": len(self.get_active_positions()),
        }


# ═════════════════════════════════════════════════════
#  TESTS / DEMO
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    from munich_strategy_config import StopLossConfig, load_config
    from munich_config import C, R, B

    print(f"\n  {B}{C['cyan']}=== Stop-Loss Module Demo ==={R}\n")

    # Carregar config
    full_config = load_config()
    sl_config = full_config.stop_loss

    print(f"  {C['yellow']}Config:{reset}")
    print(f"    temp_threshold: {sl_config.temp_threshold}°C")
    print(f"    prob_threshold: {sl_config.prob_threshold*100:.0f}%")
    print(f"    mode: {sl_config.mode}")
    print(f"    min_pnl_to_exit: ${sl_config.min_pnl_to_exit:.2f}\n")

    # Criar checker
    checker = StopLossChecker(sl_config)

    # Criar posição de teste
    position = Position(
        token_id="test_token",
        bracket_label="10.0°C",
        entry_temp=10.0,
        entry_ask=0.60,
        entry_p_ensemble=0.85,
        entry_time="2024-06-18T14:00:00",
        shares=10.0,
        cost_usdc=6.0,
    )

    print(f"  {C['cyan']}Position:{reset}")
    print(f"    entry_temp: {position.entry_temp}°C")
    print(f"    entry_p: {position.entry_p_ensemble*100:.0f}%")
    print(f"    cost: ${position.cost_usdc:.2f}\n")

    # Teste 1: Temperatura sobe (adverso)
    print(f"  {C['yellow']}Test 1: Temp sobe para 10.8°C (+0.8°C, threshold=0.5°C){reset}")
    should_exit, reason, loss = checker.check(position, 10.8, 0.70, 0.50)
    print(f"    Should exit: {should_exit}")
    print(f"    Reason: {reason}")
    print(f"    Estimated loss: ${loss:.2f}\n")

    # Teste 2: Probabilidade desce
    print(f"  {C['yellow']}Test 2: Prob desce para 55% (threshold=60%){reset}")
    should_exit, reason, loss = checker.check(position, 10.0, 0.55, 0.45)
    print(f"    Should exit: {should_exit}")
    print(f"    Reason: {reason}")
    print(f"    Estimated loss: ${loss:.2f}\n")

    # Teste 3: Tudo OK
    print(f"  {C['yellow']}Test 3: Temp OK, Prob OK{reset}")
    should_exit, reason, loss = checker.check(position, 10.2, 0.82, 0.62)
    print(f"    Should exit: {should_exit}\n")
