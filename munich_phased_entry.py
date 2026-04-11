"""
munich_phased_entry.py
======================
Lógica de entrada: modo PHASED (3 parcelas) ou SINGLE (1 compra).

PHASED:
  P1: Regra TRIPLA — manhã + forecast + mercado confirma
      • 10h <= hora < 12h
      • forecasts_agree() = True
      • Mercado confirma: bracket com MAIOR ask = running max
      • p_ensemble <= 70% (pico ainda NÃO ocorreu → comprar ANTES)

  P2: Dupla confirmação — modelo + mercado
      • p_ensemble >= 60%
      • Mercado confirma

  P3: Alta confiança
      • p_ensemble >= 80%

SINGLE:
  • Compra única quando p_ensemble >= 75%
  • Sem janela horária, sem forecast, sem confirmação de mercado
"""


class PhasedEntry:
    def __init__(self, parcel_size: float = 5.0):
        self.parcel_size = parcel_size
        self.thr_p1_max = 0.70   # P1: limite SUPERIOR (pico ainda não ocorreu)
        self.thr_p2 = 0.60
        self.thr_p3 = 0.80
        self.temp_tolerance = 1

        self.p1_hour_min = 10
        self.p1_hour_max = 12

        self.parcel_bought  = [False, False, False]
        self.parcel_records = [None, None, None]

    def _find_highest_ask_bracket(self, market):
        if not market or not market.get("brackets"):
            return None
        return max(market["brackets"],
                   key=lambda b: b.get("ask") or b.get("price") or 0)

    def _market_confirms_model(self, market, running_max):
        best = self._find_highest_ask_bracket(market)
        if best is None:
            return False, "sem mercado"

        best_ask   = best.get("ask") or best.get("price") or 0
        best_lo    = best["temp_lo"]
        best_hi    = best["temp_hi"]
        best_label = best["label"]
        rmax_int   = int(round(running_max))

        if best_lo <= -99:
            return False, f"mercado={best_label} (or lower)"

        if best_lo <= rmax_int <= best_hi:
            return True, f"mercado={best_label} ({best_ask*100:.0f}¢) = {rmax_int}°C"

        mid = best_lo if best_hi >= 99 else (best_lo + best_hi) / 2
        if abs(mid - rmax_int) <= self.temp_tolerance:
            return True, f"mercado={best_label} ({best_ask*100:.0f}¢) ≈ {rmax_int}°C"

        return False, f"mercado={best_label} ({best_ask*100:.0f}¢) ≠ {rmax_int}°C"

    def evaluate(self, p_ensemble, hour, market, running_max, forecast_agreement):
        actions = []

        # ── PARCELA 1: ANTES do pico (lógica invertida) ─
        if not self.parcel_bought[0]:
            in_morning = self.p1_hour_min <= hour < self.p1_hour_max
            fc_ok = (forecast_agreement is not None
                     and forecast_agreement.get("valid", False))
            mkt_ok, mkt_detail = self._market_confirms_model(market, running_max)
            model_ok = p_ensemble <= self.thr_p1_max

            if in_morning and fc_ok and mkt_ok and model_ok:
                actions.append({
                    "parcel_idx": 0,
                    "size_usdc":  self.parcel_size,
                    "reason":     (f"P1: manhã ({hour}h) + fc agree + "
                                   f"{mkt_detail} + p={p_ensemble*100:.0f}% "
                                   f"(antes do pico)"),
                    "model_ok":   True,
                    "market_ok":  True,
                })
            else:
                reasons = []
                if not in_morning:
                    reasons.append(
                        f"hora={hour}h fora de [{self.p1_hour_min},{self.p1_hour_max})")
                if not fc_ok:
                    reasons.append("forecast disagree")
                if not mkt_ok:
                    reasons.append(f"mercado NÃO ({mkt_detail})")
                if not model_ok:
                    reasons.append(
                        f"p={p_ensemble*100:.0f}% > {self.thr_p1_max*100:.0f}% "
                        f"(pico já passou!)")
                actions.append({
                    "parcel_idx": 0,
                    "size_usdc":  0,
                    "reason":     f"P1 BLOQUEADA: {' | '.join(reasons)}",
                    "model_ok":   model_ok,
                    "market_ok":  mkt_ok,
                })

        # ── PARCELA 2: Dupla confirmação ─────────────
        if not self.parcel_bought[1]:
            model_ok = p_ensemble >= self.thr_p2
            mkt_ok, mkt_detail = self._market_confirms_model(market, running_max)

            if model_ok and mkt_ok:
                actions.append({
                    "parcel_idx": 1,
                    "size_usdc":  self.parcel_size,
                    "reason":     f"P2: p={p_ensemble*100:.0f}% + {mkt_detail}",
                    "model_ok":   True,
                    "market_ok":  True,
                })
            elif model_ok and not mkt_ok:
                actions.append({
                    "parcel_idx": 1,
                    "size_usdc":  0,
                    "reason":     (f"P2 BLOQUEADA: modelo OK "
                                   f"({p_ensemble*100:.0f}%), mercado NÃO "
                                   f"({mkt_detail})"),
                    "model_ok":   True,
                    "market_ok":  False,
                })

        # ── PARCELA 3: Alta confiança ────────────────
        if not self.parcel_bought[2]:
            if p_ensemble >= self.thr_p3:
                actions.append({
                    "parcel_idx": 2,
                    "size_usdc":  self.parcel_size,
                    "reason":     f"P3: p={p_ensemble*100:.0f}% >= 80%",
                    "model_ok":   True,
                    "market_ok":  True,
                })

        return actions

    def mark_bought(self, parcel_idx, record):
        self.parcel_bought[parcel_idx] = True
        self.parcel_records[parcel_idx] = record

    def reset(self):
        self.parcel_bought  = [False, False, False]
        self.parcel_records = [None, None, None]

    @property
    def total_invested(self):
        return sum(self.parcel_size for b in self.parcel_bought if b)

    @property
    def n_parcels_bought(self):
        return sum(self.parcel_bought)


# ══════════════════════════════════════════════════════
#  SINGLE ENTRY — 1 compra quando p >= threshold
# ══════════════════════════════════════════════════════
class SingleEntry:
    """
    Modo SINGLE: compra única quando p_ensemble >= threshold.
    Interface compatível com PhasedEntry (parcel_bought, parcel_records, etc.)
    """

    def __init__(self, parcel_size: float = 15.0, threshold: float = 0.75):
        self.parcel_size = parcel_size
        self.threshold   = threshold
        self.bought      = False
        self.record      = None

    def evaluate(self, p_ensemble, hour, market, running_max, forecast_agreement):
        actions = []

        if not self.bought and p_ensemble >= self.threshold:
            actions.append({
                "parcel_idx": 0,
                "size_usdc":  self.parcel_size,
                "reason":     (f"SINGLE: p={p_ensemble*100:.0f}% >= "
                               f"{self.threshold*100:.0f}%"),
                "model_ok":   True,
                "market_ok":  True,
            })
        elif not self.bought:
            actions.append({
                "parcel_idx": 0,
                "size_usdc":  0,
                "reason":     (f"SINGLE: p={p_ensemble*100:.0f}% < "
                               f"{self.threshold*100:.0f}%"),
                "model_ok":   False,
                "market_ok":  None,
            })

        return actions

    def mark_bought(self, parcel_idx, record):
        self.bought = True
        self.record = record

    def reset(self):
        self.bought = False
        self.record = None

    @property
    def total_invested(self):
        return self.parcel_size if self.bought else 0.0

    @property
    def n_parcels_bought(self):
        return 1 if self.bought else 0

    @property
    def parcel_bought(self):
        return [self.bought, False, False]

    @property
    def parcel_records(self):
        return [self.record, None, None]
