#!/usr/bin/env python3
"""
munich_causal_features.py — FUZZY COGNITIVE MAP (FCM)
=====================================================
Mapeamento de relações causais entre variáveis meteorológicas.

NOTA: NÃO altera a previsão do modelo.
Serve para adicionar features se quiseres expandir o modelo.

Uso:
    from munich_causal_features import CausalMap, FohnDetector, CausalFeatures

    detector = FohnDetector()
    causal_map = CausalMap()
    features = CausalFeatures()

    # Obter features causais para um slot atual
    causal_features = features.extract(
        current_temp=25.0,
        humidity=60,
        pressure_hpa=1013,
        wind_dir_deg=180,
        wind_speed_kmh=15,
        wind_gust_kmh=25,
        slots_so_far=[...]
    )
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ═════════════════════════════════════════════════════
#  FOHN DETECTOR
# ═════════════════════════════════════════════════════

class FohnState(Enum):
    """Estado do vento Föhn."""
    INACTIVE = "inactive"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class FohnResult:
    """Resultado da detecção de Föhn."""
    state: FohnState
    intensity: float  # 0.0 a 1.0
    confidence: float  # 0.0 a 1.0
    factors: Dict[str, float]
    recommendation: str


class FohnDetector:
    """
    Detector do vento Föhn em Munique.

    O Föhn é um vento quente e seco que desce dos Alpes,
    caracterizado por:
    - Direção SUL (135-225 graus)
    - Rajadas fortes (> 30 km/h)
    - Humidade baixa (< 60%)
    - Pressão subindo
    """

    def __init__(
        self,
        wind_south_range: Tuple[float, float] = (135, 225),
        min_wind_speed: float = 10.0,
        min_wind_gust: float = 20.0,
        max_humidity: float = 60.0,
        pressure_rise_threshold: float = 2.0  # hpa/3h
    ):
        self.wind_south_range = wind_south_range
        self.min_wind_speed = min_wind_speed
        self.min_wind_gust = min_wind_gust
        self.max_humidity = max_humidity
        self.pressure_rise_threshold = pressure_rise_threshold

    def detect(
        self,
        wind_dir_deg: float,
        wind_speed_kmh: float,
        wind_gust_kmh: float,
        humidity: float,
        pressure_trend_3h: float = 0.0
    ) -> FohnResult:
        """
        Detecta atividade do Föhn.

        Args:
            wind_dir_deg: Direção do vento em graus
            wind_speed_kmh: Velocidade do vento (km/h)
            wind_gust_kmh: Rajadas (km/h)
            humidity: Humidade (%)
            pressure_trend_3h: Tendência de pressão nas últimas 3h (hpa)

        Returns:
            FohnResult com estado, intensidade e recomendação
        """
        factors = {}

        # 1. Fator de direção (SUL)
        wind_south = 0.0
        if self.wind_south_range[0] <= wind_dir_deg <= self.wind_south_range[1]:
            # Alinhamento perfeito = 1.0 (180 graus)
            perfect = (self.wind_south_range[0] + self.wind_south_range[1]) / 2
            deviation = abs(wind_dir_deg - perfect)
            wind_south = max(0.0, 1.0 - deviation / 45.0)
        factors["wind_south"] = wind_south

        # 2. Fator de velocidade
        speed_factor = min(1.0, wind_speed_kmh / 30.0)
        factors["wind_speed"] = speed_factor

        # 3. Fator de rajadas
        gust_factor = min(1.0, wind_gust_kmh / 40.0)
        factors["wind_gust"] = gust_factor

        # 4. Fator de humidade (inverso: baixa = Föhn)
        humidity_factor = max(0.0, 1.0 - humidity / self.max_humidity)
        factors["humidity_low"] = humidity_factor

        # 5. Fator de pressão (subindo = Föhn)
        pressure_factor = min(1.0, max(0.0, pressure_trend_3h / self.pressure_rise_threshold))
        factors["pressure_rising"] = pressure_factor

        # Calcular intensidade (média ponderada)
        # Direção e rajadas são mais importantes
        intensity = (
            0.30 * wind_south +
            0.15 * speed_factor +
            0.25 * gust_factor +
            0.15 * humidity_factor +
            0.15 * pressure_factor
        )

        # Calcular confiança
        confidence_factors = [
            wind_south,
            speed_factor,
            gust_factor
        ]
        confidence = np.mean(confidence_factors)

        # Determinar estado
        if intensity >= 0.70:
            state = FohnState.STRONG
            recommendation = "Föhn FORTE: reduzir confiança no modelo 40-60%"
        elif intensity >= 0.50:
            state = FohnState.MODERATE
            recommendation = "Föhn MODERADO: reduzir confiança no modelo 20-30%"
        elif intensity >= 0.30:
            state = FohnState.WEAK
            recommendation = "Föhn FRACO: reduzir confiança no modelo 10-20%"
        else:
            state = FohnState.INACTIVE
            recommendation = "Sem Föhn: usar modelo normalmente"

        return FohnResult(
            state=state,
            intensity=intensity,
            confidence=confidence,
            factors=factors,
            recommendation=recommendation
        )


# ═════════════════════════════════════════════════════
#  FUZZY COGNITIVE MAP (FCM)
# ═════════════════════════════════════════════════════

@dataclass
class FCMNode:
    """Nó do Fuzzy Cognitive Map."""
    name: str
    activation: float  # Valor de ativação [-1, 1]

@dataclass
class FCMEdge:
    """Aresta do Fuzzy Cognitive Map."""
    source: str
    target: str
    weight: float  # Peso da relação [-1, 1]


class CausalMap:
    """
    Fuzzy Cognitive Map para relações causais meteorológicas.

    Mapeia como diferentes variáveis se influenciam:

    Variáveis (Nós):
      - humidade: Humidade atual
      - pressao: Pressão atmosférica
      - vento_sul: Vento do sul (Föhn)
      - temperatura: Temperatura atual
      - temp_prevista: Temperatura prevista para pico
      - zscore: Z-score atual
      - modelo_confianca: Confiança no modelo

    Relações (Arestas):
      - humidade ↓ → temp_prevista ↓
      - pressao ↑ → zscore ↓
      - vento_sul ↑ → modelo_confianca ↓ (Föhn)
      - vento_sul ↑ → humidade ↓
    """

    def __init__(self):
        # Inicializar nós
        self.nodes: Dict[str, FCMNode] = {
            "humidade": FCMNode("humidade", 0.0),
            "pressao": FCMNode("pressao", 0.0),
            "vento_sul": FCMNode("vento_sul", 0.0),
            "temperatura": FCMNode("temperatura", 0.0),
            "temp_prevista": FCMNode("temp_prevista", 0.0),
            "zscore": FCMNode("zscore", 0.0),
            "modelo_confianca": FCMNode("modelo_confianca", 0.5),  # Começa neutro
        }

        # Inicializar arestas (relações causais)
        self.edges: List[FCMEdge] = [
            # Humidade baixa → Temperatura prevista baixa
            FCMEdge("humidade", "temp_prevista", 0.7),

            # Pressão subindo → Z-score desce
            FCMEdge("pressao", "zscore", -0.5),

            # Vento sul (Föhn) → Confiança no modelo desce
            FCMEdge("vento_sul", "modelo_confianca", -0.8),

            # Vento sul → Humidade desce
            FCMEdge("vento_sul", "humidade", -0.6),

            # Temperatura alta → Z-score sobe
            FCMEdge("temperatura", "zscore", 0.5),
        ]

        # Criar mapa de adjacência para inferência rápida
        self._build_adjacency_map()

    def _build_adjacency_map(self):
        """Constrói mapa de adjacência para inferência."""
        self.adjacency = {node: [] for node in self.nodes}
        for edge in self.edges:
            if edge.source in self.adjacency:
                self.adjacency[edge.source].append(edge)

    def set_node(self, name: str, activation: float):
        """Define a ativação de um nó [-1, 1]."""
        if name in self.nodes:
            self.nodes[name].activation = float(np.clip(activation, -1.0, 1.0))

    def infer(self, steps: int = 1) -> Dict[str, float]:
        """
        Executa inferência no FCM.

        Propaga a ativação através das arestas por N passos.

        Returns:
            Dict com valores de ativação finais
        """
        activations = {k: v.activation for k, v in self.nodes.items()}

        for _ in range(steps):
            new_activations = {}

            for node_name in self.nodes:
                # Soma das influências
                influence = 0.0
                for edge in self.adjacency.get(node_name, []):
                    influence += activations[edge.target] * edge.weight

                # Sigmoid para manter em [-1, 1]
                new_value = np.tanh(influence)
                new_activations[node_name] = new_value

            activations = new_activations

        return activations

    def get_model_confidence_adjustment(
        self,
        fohn_intensity: float = 0.0,
        humidity: float = 70.0,
        pressure_trend: float = 0.0
    ) -> Tuple[float, str]:
        """
        Obtém o ajuste de confiança no modelo baseado no FCM.

        Args:
            fohn_intensity: Intensidade do Föhn [0, 1]
            humidity: Humidade atual [%]
            pressure_trend: Tendência de pressão (hpa)

        Returns:
            Tuple: (fator_ajuste, razão)
            fator_ajuste: 0.0 a 1.0 (1.0 = sem ajuste)
            razão: Explicação textual
        """
        # Normalizar inputs para [-1, 1]
        fohn_norm = fohn_intensity * 2 - 1  # [0,1] → [-1,1]
        humidity_norm = (humidity - 50) / 50  # [%] → [-1,1]
        pressure_norm = np.clip(pressure_trend / 5, -1, 1)

        # Atualizar nós
        self.set_node("vento_sul", fohn_norm)
        self.set_node("humidade", humidity_norm)
        self.set_node("pressao", pressure_norm)

        # Inferir
        result = self.infer(steps=2)

        # Obter confiança do modelo (converter de [-1,1] para [0,1])
        conf_raw = result.get("modelo_confianca", 0.5)
        confidence = (conf_raw + 1) / 2

        # Criar razão
        reasons = []
        if fohn_intensity > 0.5:
            reasons.append(f"Föhn ativo (intensidade {fohn_intensity:.2f})")
        if humidity < 50:
            reasons.append(f"Humidade baixa ({humidity:.0f}%)")
        if pressure_trend > 1:
            reasons.append(f"Pressão subindo (+{pressure_trend:.1f}hpa)")

        reason = " | ".join(reasons) if reasons else "Condições normais"

        return confidence, reason


# ═════════════════════════════════════════════════════
#  CAUSAL FEATURES EXTRACTOR
# ═════════════════════════════════════════════════════

@dataclass
class CausalFeatureResult:
    """Resultado da extração de features causais."""
    # Detecção de Föhn
    fohn_state: str
    fohn_intensity: float
    fohn_factors: Dict[str, float]

    # Ajuste de confiança
    model_confidence_adjustment: float  # 0.0 a 1.0
    confidence_reason: str

    # Features adicionais (para usar no modelo)
    fohn_indicator: float  # 0.0 a 1.0
    humidity_pressure_score: float  # -1.0 a 1.0
    weather_regime: str  # "stable", "warming", "cooling", "foehn"


class CausalFeatures:
    """
    Extrator de features causais para o modelo.

    Gera features baseadas em relações causais meteorológicas.

    NOTA: Estas features são adicionais e opcionais.
          Podem ser usadas para expandir o modelo se desejado.
    """

    def __init__(self):
        self.fohn_detector = FohnDetector()
        self.causal_map = CausalMap()

    def extract(
        self,
        current_temp: float,
        humidity: float,
        pressure_hpa: float,
        wind_dir_deg: float,
        wind_speed_kmh: float,
        wind_gust_kmh: float,
        slots_so_far: List[Dict],
        pressure_trend_3h: float = 0.0
    ) -> CausalFeatureResult:
        """
        Extrai features causais para o slot atual.

        Args:
            current_temp: Temperatura atual (°C)
            humidity: Humidade (%)
            pressure_hpa: Pressão (hPa)
            wind_dir_deg: Direção do vento (graus)
            wind_speed_kmh: Velocidade do vento (km/h)
            wind_gust_kmh: Rajadas (km/h)
            slots_so_far: Slots até agora
            pressure_trend_3h: Tendência de pressão nas últimas 3h

        Returns:
            CausalFeatureResult com todas as features
        """
        # 1. Detectar Föhn
        fohn_result = self.fohn_detector.detect(
            wind_dir_deg=wind_dir_deg,
            wind_speed_kmh=wind_speed_kmh,
            wind_gust_kmh=wind_gust_kmh,
            humidity=humidity,
            pressure_trend_3h=pressure_trend_3h
        )

        # 2. Calcular ajuste de confiança via FCM
        confidence_adj, reason = self.causal_map.get_model_confidence_adjustment(
            fohn_intensity=fohn_result.intensity,
            humidity=humidity,
            pressure_trend=pressure_trend_3h
        )

        # 3. Calcular score de humidade/pressão
        # Humidade baixa + pressão subindo = potencial de aquecimento
        humidity_norm = (humidity - 50) / 50  # [-1, 1]
        humidity_pressure_score = -humidity_norm + np.clip(pressure_trend_3h / 3, -1, 1)

        # 4. Determinar regime meteorológico
        weather_regime = self._determine_weather_regime(
            fohn_result.state.value,
            slots_so_far,
            current_temp
        )

        return CausalFeatureResult(
            fohn_state=fohn_result.state.value,
            fohn_intensity=fohn_result.intensity,
            fohn_factors=fohn_result.factors,

            model_confidence_adjustment=confidence_adj,
            confidence_reason=reason,

            fohn_indicator=fohn_result.intensity,
            humidity_pressure_score=humidity_pressure_score,
            weather_regime=weather_regime
        )

    def _determine_weather_regime(
        self,
        fohn_state_value: str,
        slots_so_far: List[Dict],
        current_temp: float
    ) -> str:
        """
        Determina o regime meteorológico atual.

        Possíveis regimes:
        - "foehn": Föhn ativo
        - "warming": Temperatura subindo
        - "cooling": Temperatura descendo
        - "stable": Temperatura estável
        """
        if fohn_state_value != "inactive":
            return "foehn"

        if len(slots_so_far) < 4:
            return "stable"

        # Verificar tendência de temperatura
        temps = [s["temp_c"] for s in slots_so_far[-6:]]
        if len(temps) < 3:
            return "stable"

        # Regressão linear simples
        x = np.arange(len(temps))
        slope = np.polyfit(x, temps, 1)[0]

        if slope > 0.2:
            return "warming"
        elif slope < -0.2:
            return "cooling"
        else:
            return "stable"

    def get_adjusted_prediction(
        self,
        p_ensemble: float,
        causal_features: CausalFeatureResult
    ) -> float:
        """
        Ajusta a predição do modelo baseado em features causais.

        NOTA: Isto é apenas para referência.
              NÃO altera a predição oficial do modelo.

        Args:
            p_ensemble: Predição do ensemble [0, 1]
            causal_features: Features causais extraídas

        Returns:
            Predição ajustada (apenas para referência)
        """
        # O ajuste depende da confiança do modelo
        conf_adj = causal_features.model_confidence_adjustment

        # Se o Föhn está ativo, reduzir confiança
        if causal_features.fohn_intensity > 0.5:
            # Predição ajustada = p_ensemble * confiança + (1-confiança) * 0.5
            # Isto empurra para 0.5 (incerteza) quando confiança é baixa
            adjusted = p_ensemble * conf_adj + (1 - conf_adj) * 0.5
            return float(np.clip(adjusted, 0.0, 1.0))

        return p_ensemble


# ═════════════════════════════════════════════════════
#  VISUALIZATION HELPER
# ═════════════════════════════════════════════════════

def print_causal_features(result: CausalFeatureResult):
    """Imprime features causais de forma formatada."""
    from munich_config import C, R

    print(f"\n  {C['cyan']}=== CAUSAL FEATURES ==={R}")

    # Föhn
    fohn_color = {
        "inactive": "green",
        "weak": "yellow",
        "moderate": "yellow",
        "strong": "red"
    }.get(result.fohn_state, "white")

    print(f"\n  Föhn: {C[fohn_color]}{result.fohn_state.upper()}{R} "
          f"(intensidade: {result.fohn_intensity:.2f})")

    # Fatores
    print(f"  Fatores:")
    for k, v in result.fohn_factors.items():
        color = "green" if v > 0.5 else "gray" if v < 0.2 else "yellow"
        print(f"    {k}: {C[color]}{v:.2f}{R}")

    # Ajuste de confiança
    conf_pct = result.model_confidence_adjustment * 100
    conf_color = "green" if conf_pct > 80 else "yellow" if conf_pct > 60 else "red"
    print(f"\n  Ajuste de Confiança: {C[conf_color]}{conf_pct:.0f}%{R}")
    print(f"  Razão: {result.confidence_reason}")

    # Regime meteorológico
    regime_color = {
        "stable": "green",
        "warming": "yellow",
        "cooling": "cyan",
        "foehn": "red"
    }.get(result.weather_regime, "white")
    print(f"\n  Regime: {C[regime_color]}{result.weather_regime.upper()}{R}")

    # Features para modelo
    print(f"\n  Features para Modelo:")
    print(f"    fohn_indicator: {result.fohn_indicator:.3f}")
    print(f"    humidity_pressure_score: {result.humidity_pressure_score:.3f}")


# ═════════════════════════════════════════════════════
#  MAIN (para testes)
# ═════════════════════════════════════════════════════

if __name__ == "__main__":
    # Teste básico do Föhn Detector
    print(f"\n  {'='*46}")
    print(f"  === FÖHN DETECTOR TEST ===")
    print(f"  {'='*46}\n")

    detector = FohnDetector()

    # Cenário 1: Föhn forte
    print(f"  Cenário 1: Föhn FORTE")
    result = detector.detect(
        wind_dir_deg=180,  # Sul
        wind_speed_kmh=25,
        wind_gust_kmh=40,
        humidity=45,
        pressure_trend_3h=3.0
    )
    print(f"    Estado: {result.state.value}")
    print(f"    Intensidade: {result.intensity:.2f}")
    print(f"    Confiança: {result.confidence:.2f}")
    print(f"    Recomendação: {result.recommendation}\n")

    # Cenário 2: Sem Föhn
    print(f"  Cenário 2: Sem Föhn")
    result = detector.detect(
        wind_dir_deg=90,  # Leste
        wind_speed_kmh=5,
        wind_gust_kmh=10,
        humidity=70,
        pressure_trend_3h=0.0
    )
    print(f"    Estado: {result.state.value}")
    print(f"    Intensidade: {result.intensity:.2f}")
    print(f"    Confiança: {result.confidence:.2f}")
    print(f"    Recomendação: {result.recommendation}\n")

    # Teste completo do Causal Features
    print(f"\n  {'='*46}")
    print(f"  === CAUSAL FEATURES TEST ===")
    print(f"  {'='*46}")

    features = CausalFeatures()

    # Slots simulados
    slots = [
        {"temp_c": 20.0},
        {"temp_c": 21.0},
        {"temp_c": 22.0},
        {"temp_c": 23.0},
    ]

    result = features.extract(
        current_temp=24.0,
        humidity=50,
        pressure_hpa=1015,
        wind_dir_deg=180,
        wind_speed_kmh=20,
        wind_gust_kmh=35,
        slots_so_far=slots,
        pressure_trend_3h=2.0
    )

    print_causal_features(result)

    # Teste de ajuste de predição
    p_original = 0.85
    p_adjusted = features.get_adjusted_prediction(p_original, result)
    print(f"\n  Predição original: {p_original:.2f}")
    print(f"  Predição ajustada: {p_adjusted:.2f}")
