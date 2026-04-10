"""
munich_phased_entry.py
======================
Lógica de entrada faseada em 3 parcelas de $5 com dupla confirmação.

P1: Manhã cedo + Forecast agreement (WU + Open-Meteo)
P2: p_ensemble >= 60% + Mercado confirma (bracket com MAIOR ask = running max)
P3: p_ensemble >= 80%

O "Mercado confirma" significa: o bracket onde o mercado tem a maior 
probabilidade (maior ask) aponta para a mesma temperatura que o nosso 
running max (modelo).
"""

class PhasedEntry:
    def __init__(self, parcel_size: float = 5.0):
        self.parcel_size = parcel_size
        self.thr_p2 = 0.60
        self.thr_p3 = 0.80
        self.temp_tolerance = 1  # °C tolerância para matching
        
        # Estado
        self.parcel_bought  = [False, False, False]
        self.parcel_records = [None, None, None]

    def _find_highest_ask_bracket(self, market):
        """Encontra o bracket com a MAIOR ask no mercado."""
        if not market or not market.get("brackets"):
            return None
        return max(market["brackets"],
                   key=lambda b: b.get("ask") or b.get("price") or 0)

    def _market_confirms_model(self, market, running_max):
        """
        O mercado confirma o modelo quando o bracket com maior ask
        aponta para a mesma temperatura que o nosso running max.
        """
        best = self._find_highest_ask_bracket(market)
        if best is None:
            return False, "sem mercado"

        best_ask   = best.get("ask") or best.get("price") or 0
        best_lo    = best["temp_lo"]
        best_hi    = best["temp_hi"]
        best_label = best["label"]
        rmax_int   = int(round(running_max))

        # "or lower" = não é pico
        if best_lo <= -99:
            return False, f"mercado={best_label} (or lower, não pico)"

        # Running max dentro do bracket?
        if best_lo <= rmax_int <= best_hi:
            return True, f"mercado={best_label} ({best_ask*100:.0f}¢) = {rmax_int}°C"

        # Com tolerância
        mid = best_lo if best_hi >= 99 else (best_lo + best_hi) / 2
        if abs(mid - rmax_int) <= self.temp_tolerance:
            return True, f"mercado={best_label} ({best_ask*100:.0f}¢) ≈ {rmax_int}°C (±{self.temp_tolerance}°C)"

        return False, f"mercado={best_label} ({best_ask*100:.0f}¢) ≠ {rmax_int}°C"

    def evaluate(self, p_ensemble, hour, market, running_max, forecast_agreement):
        """
        Avalia quais parcelas devem ser executadas AGORA.
        Retorna lista de actions.
        """
        actions = []

        # ── PARCELA 1: Manhã cedo ──────────────────
        if not self.parcel_bought[0]:
            fc_ok = (forecast_agreement is not None 
                     and forecast_agreement.get("valid", False))
            if hour < 12 and fc_ok:
                actions.append({
                    "parcel_idx": 0, 
                    "size_usdc": self.parcel_size,
                    "reason": f"P1: manhã ({hour}h) + forecast agreement ✅",
                    "model_ok": True, 
                    "market_ok": True,  # P1 não precisa de confirmação de mercado
                })

        # ── PARCELA 2: Dupla confirmação ───────────
        if not self.parcel_bought[1]:
            model_ok = p_ensemble >= self.thr_p2
            mkt_ok, mkt_detail = self._market_confirms_model(market, running_max)

            if model_ok and mkt_ok:
                actions.append({
                    "parcel_idx": 1, 
                    "size_usdc": self.parcel_size,
                    "reason": f"P2: p={p_ensemble*100:.0f}% + {mkt_detail}",
                    "model_ok": True, 
                    "market_ok": True,
                })
            elif model_ok and not mkt_ok:
                actions.append({
                    "parcel_idx": 1, 
                    "size_usdc": 0,
                    "reason": f"P2 BLOQUEADA: modelo OK, mercado NÃO ({mkt_detail})",
                    "model_ok": True, 
                    "market_ok": False,
                })

        # ── PARCELA 3: Alta confiança ──────────────
        if not self.parcel_bought[2]:
            if p_ensemble >= self.thr_p3:
                actions.append({
                    "parcel_idx": 2, 
                    "size_usdc": self.parcel_size,
                    "reason": f"P3: p={p_ensemble*100:.0f}% >= 80%",
                    "model_ok": True, 
                    "market_ok": True,  # P3 não precisa de confirmação de mercado
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
