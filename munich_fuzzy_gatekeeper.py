#!/usr/bin/env python3
"""
munich_fuzzy_gatekeeper.py — PORTFIFO (Fuzzy Gatekeeper)
====================================================
Verifica o CONTEXTO antes de enviar ordens à Polymarket.

O Porteiro funciona como um filtro de segurança:
- Se o Porteiro disser SIM → O modelo pode entrar
- Se o Porteiro disser NÃO → O modelo ignora e a trade NÃO é executada

Lógica fuzzy para avaliação de risco (estados possíveis):
- "safe":      Contexto favorável, pode entrar
- "risky":     Contexto marginal, entrar com cautela
- "blocked":   Contexto perigoso, BLOQUEAR entrada
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class GatekeeperState(Enum):
    """Estado do Porteiro."""
    SAFE = "safe"
    RISKY = "risky"
    BLOCKED = "blocked"


@dataclass
class GatekeeperResult:
    """Resultado da avaliação do Porteiro."""
    state: GatekeeperState
    allowed: bool
    reason: str
    scores: Dict[str, float]
    details: Dict[str, str]


class FuzzyGatekeeper:
    """
    Porteiro fuzzy para avaliar contexto de entrada.

    Verifica 4 condições principais:
    1. EV é positivo
    2. Forecast concorda
    3. Mercado não está "arriscado"
    4. Z-Score confirma

    Lógica fuzzy:
    - Cada condição retorna um score [0, 1]
    - Scores são ponderados
    - Score total determina o estado
    """

    def __init__(
        self,
        ev_min_threshold: float = 0.02,  # 2% EV mínimo
        zscore_min: float = 1.0,           # Z-score mínimo para confirmar
        market_volume_min: float = 100.0, # Volume mínimo do mercado
        weights: Optional[Dict[str, float]] = None
    ):
        self.ev_min_threshold = ev_min_threshold
        self.zscore_min = zscore_min
        self.market_volume_min = market_volume_min

        # Pesos padrão para cada condição
        self.weights = weights or {
            "ev": 0.35,          # EV é o mais importante
            "forecast": 0.25,   # Forecast concordância
            "market": 0.20,     # Saúde do mercado
            "zscore": 0.20,     # Z-score confirmação
        }

    def evaluate(
        self,
        p_ensemble: float,
        ask_price: float,
        forecast_agreement: Dict,
        market: Optional[Dict],
        zscore_component: Optional[float],
        running_max: float,
        current_temp: float
    ) -> GatekeeperResult:
        """
        Avalia o contexto e determina se a entrada é permitida.

        Args:
            p_ensemble: Probabilidade do ensemble (0-1)
            ask_price: Preço de entrada (0-1)
            forecast_agreement: Dict com {"valid": bool, "detail": str}
            market: Dict do mercado Polymarket (opcional)
            zscore_component: Componente Z-score (opcional)
            running_max: Temperatura máxima até agora
            current_temp: Temperatura atual

        Returns:
            GatekeeperResult com estado, permissão, razão e scores
        """
        scores = {}
        details = {}

        # 1. Verificar EV (Expected Value)
        ev_score, ev_detail = self._check_ev(p_ensemble, ask_price)
        scores["ev"] = ev_score
        details["ev"] = ev_detail

        # 2. Verificar Forecast
        forecast_score, forecast_detail = self._check_forecast(forecast_agreement)
        scores["forecast"] = forecast_score
        details["forecast"] = forecast_detail

        # 3. Verificar Mercado
        market_score, market_detail = self._check_market(market, running_max)
        scores["market"] = market_score
        details["market"] = market_detail

        # 4. Verificar Z-Score
        zscore_score, zscore_detail = self._check_zscore(zscore_component)
        scores["zscore"] = zscore_score
        details["zscore"] = zscore_detail

        # Calcular score total ponderado
        total_score = sum(scores[k] * self.weights[k] for k in scores)

        # Determinar estado
        if total_score >= 0.70:
            state = GatekeeperState.SAFE
            allowed = True
            reason = f"Contexto SAFE (score: {total_score:.2f})"
        elif total_score >= 0.40:
            state = GatekeeperState.RISKY
            allowed = True  # Permitir mas com cautela
            reason = f"Contexto RISKY (score: {total_score:.2f}) - entrar com cautela"
        else:
            state = GatekeeperState.BLOCKED
            allowed = False
            reason = f"Contexto BLOQUEADO (score: {total_score:.2f})"

        return GatekeeperResult(
            state=state,
            allowed=allowed,
            reason=reason,
            scores=scores,
            details=details
        )

    def _check_ev(self, p_ensemble: float, ask_price: float) -> Tuple[float, str]:
        """
        Verifica se o Expected Value é positivo.

        EV = p - ask
        """
        if not ask_price or not (0 < ask_price < 1):
            return 0.0, "preço inválido"

        ev = p_ensemble - ask_price
        ev_pct = (p_ensemble / ask_price - 1) * 100 if ask_price > 0 else 0

        if ev >= self.ev_min_threshold:
            # Score baseado no EV (mais EV = melhor score)
            score = min(1.0, ev / (self.ev_min_threshold * 3) + 0.5)
            return score, f"EV positivo: +{ev*100:.2f}¢ ({ev_pct:+.1f}%)"
        elif ev > 0:
            score = 0.3
            return score, f"EV baixo: +{ev*100:.2f}¢ ({ev_pct:+.1f}%)"
        else:
            return 0.0, f"EV negativo: {ev*100:.2f}¢ ({ev_pct:+.1f}%)"

    def _check_forecast(self, forecast_agreement: Dict) -> Tuple[float, str]:
        """
        Verifica se o forecast concorda com o modelo.
        """
        if not forecast_agreement:
            return 0.5, "forecast indisponível (score neutro)"

        valid = forecast_agreement.get("valid", False)
        detail = forecast_agreement.get("detail", "")

        if valid:
            return 1.0, f"forecast concorda: {detail}"
        else:
            return 0.2, f"forecast discorda: {detail}"

    def _check_market(self, market: Optional[Dict], running_max: float) -> Tuple[float, str]:
        """
        Verifica se o mercado está saudável.
        """
        if not market:
            return 0.5, "mercado indisponível (score neutro)"

        # Verificar volume
        volume = market.get("volume", 0)
        if volume < self.market_volume_min:
            return 0.2, f"volume baixo: ${volume:.0f} < ${self.market_volume_min:.0f}"

        # Verificar brackets
        brackets = market.get("brackets", [])
        if not brackets:
            return 0.0, "sem brackets disponíveis"

        # Verificar se há bracket perto do running max
        rmax_int = int(round(running_max))
        found_bracket = False
        best_ask = 0

        for b in brackets:
            lo, hi = b.get("temp_lo", 0), b.get("temp_hi", 99)
            if lo <= rmax_int <= hi:
                found_bracket = True
                best_ask = b.get("ask", b.get("price", 0))
                break

        if found_bracket and best_ask and 0 < best_ask < 0.95:
            # Score baseado no volume e ask
            vol_score = min(1.0, volume / 1000)  # Normalizar volume
            ask_score = 1.0 - best_ask  # Ask mais baixo = melhor
            score = (vol_score + ask_score) / 2
            return score, f"mercado OK: volume=${volume:.0f}, ask={best_ask*100:.1f}¢"
        else:
            return 0.3, "sem bracket relevante para temperatura atual"

    def _check_zscore(self, zscore_component: Optional[float]) -> Tuple[float, str]:
        """
        Verifica se o Z-Score confirma o sinal.
        """
        if zscore_component is None:
            return 0.5, "z-score indisponível (score neutro)"

        if zscore_component >= self.zscore_min:
            # Score baseado no z-score (mais alto = melhor)
            score = min(1.0, zscore_component / (self.zscore_min * 2) + 0.5)
            return score, f"z-score confirma: {zscore_component:.2f}"
        else:
            return 0.3, f"z-score baixo: {zscore_component:.2f}"

    def print_result(self, result: GatekeeperResult) -> None:
        """Imprime o resultado da avaliação em formato legível."""
        from munich_config import C, R

        state_colors = {
            GatekeeperState.SAFE: "green",
            GatekeeperState.RISKY: "yellow",
            GatekeeperState.BLOCKED: "red",
        }

        color = state_colors.get(result.state, "white")
        state_text = result.state.value.upper()

        print(f"\n  {C[color]}╔════════════════════════════════════════╗{R}")
        print(f"  {C[color]}║     GATEKEEPER RESULTADO             ║{R}")
        print(f"  {C[color]}╚════════════════════════════════════════╝{R}")
        print(f"\n  Estado: {C[color]}{state_text}{R}")
        print(f"  Permitido: {C['green'] if result.allowed else C['red']}{'SIM' if result.allowed else 'NÃO'}{R}")
        print(f"  Razão: {result.reason}")
        print(f"\n  Scores:")
        for key, score in result.scores.items():
            print(f"    {key:10s}: {score*100:5.1f}%  ({result.details[key]})")
        print()


# ═════════════════════════════════════════════════════
#  FUNÇÕES DE CONVENIÊNCIA
# ═════════════════════════════════════════════════════

def create_gatekeeper(config: Optional[Dict] = None) -> FuzzyGatekeeper:
    """
    Cria um FuzzyGatekeeper com configuração.

    Args:
        config: Dict com configurações personalizadas

    Returns:
        FuzzyGatekeeper configurado
    """
    if config:
        return FuzzyGatekeeper(
            ev_min_threshold=config.get("ev_min_threshold", 0.02),
            zscore_min=config.get("zscore_min", 1.0),
            market_volume_min=config.get("market_volume_min", 100.0),
            weights=config.get("weights")
        )
    return FuzzyGatekeeper()


# ═════════════════════════════════════════════════════
#  MAIN (para testes)
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    # Teste básico do Gatekeeper
    gatekeeper = FuzzyGatekeeper()

    # Simular um cenário
    result = gatekeeper.evaluate(
        p_ensemble=0.85,
        ask_price=0.25,
        forecast_agreement={"valid": True, "detail": "WU e OM concordam"},
        market={
            "volume": 5000,
            "brackets": [
                {"temp_lo": 25, "temp_hi": 25, "ask": 0.25}
            ]
        },
        zscore_component=1.5,
        running_max=25.0,
        current_temp=24.5
    )

    gatekeeper.print_result(result)
