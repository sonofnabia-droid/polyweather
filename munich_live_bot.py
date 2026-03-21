import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES E PATHS ---
BERLIN_TZ = ZoneInfo("Europe/Berlin")
MODEL_DIR = "munich_peak_model"
MODEL_PATH = f"{MODEL_DIR}/lgbm_peak.pkl"

# Features exatas definidas no teu munich_train.py
FEATURE_COLS = [
    "slot_frac", "temp_c", "running_max", "pct_of_running_max",
    "delta_30m", "delta_1h", "accel", "temp_lag_1", "temp_lag_3",
    "roll3_mean", "roll3_std", "morning_max", "temp_above_morning_max",
    "prev_7d_avg_max", "seasonal_peak_prior"
]

# --- FUNÇÕES DE APOIO ---
def berlin_now():
    return datetime.now(tz=BERLIN_TZ)

def get_target_date():
    """Proteção anti-salto de dia: após as 20:30, foca no mercado de amanhã."""
    now = berlin_now()
    if now.hour >= 20 and now.minute >= 30:
        return now.date() + timedelta(days=1)
    return now.date()

def build_live_features(slots_so_far, prev7_val, seasonal_prior):
    """Transforma os dados brutos no formato do Modelo."""
    if not slots_so_far: return None
    
    curr = slots_so_far[-1]
    temps = [s['temp_c'] for s in slots_so_far]
    rmax = max(temps)
    
    # Morning Max (até às 12:00 local)
    morn_vals = [s['temp_c'] for s in slots_so_far if s['hour'] <= 12]
    mmax = max(morn_vals) if morn_vals else curr['temp_c']
    
    def lag(n):
        return temps[-n] if len(temps) >= n else temps[0]

    feat = {
        "slot_frac": (curr['hour'] + curr['slot30']/60) / 24,
        "temp_c": curr['temp_c'],
        "running_max": rmax,
        "pct_of_running_max": curr['temp_c'] / rmax if rmax else 1.0,
        "delta_30m": curr['temp_c'] - lag(2),
        "delta_1h": curr['temp_c'] - lag(3),
        "accel": (curr['temp_c'] - lag(2)) - (lag(2) - lag(3)),
        "temp_lag_1": lag(2),
        "temp_lag_3": lag(4),
        "roll3_mean": np.mean(temps[-3:]),
        "roll3_std": np.std(temps[-3:]) if len(temps) >= 3 else 0.0,
        "morning_max": mmax,
        "temp_above_morning_max": curr['temp_c'] - mmax,
        "prev_7d_avg_max": prev7_val,
        "seasonal_peak_prior": seasonal_prior
    }
    return feat

# --- LOOP PRINCIPAL ---
def main():
    print("🚀 MUNICH BOT V1: INICIANDO...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        return
    
    model = joblib.load(MODEL_PATH)
    history_signals = {}
    last_periodic_check = -1

    while True:
        # Inicialização para evitar NameError
        p_pico = 0.0
        
        try:
            target_date = get_target_date()
            now = berlin_now()
            
            # --- PARTE A: RECOLHA ---
            # Aqui deves chamar a tua função de fetch da WU (Weather Underground)
            # snap, series_today, slots_so_far = fetch_wu_data() 
            
            # --- PARTE B: FEATURES E PREDIÇÃO ---
            # (Exemplo de preenchimento manual para o loop não crashar)
            # feat_dict = build_live_features(slots_so_far, prev7_val, prior_val)
            
            if 'feat_dict' in locals() and feat_dict is not None:
                X = pd.DataFrame([feat_dict])[FEATURE_COLS]
                p_pico = float(model.predict(X)[0]) # Probabilidade do Pico ter ocorrido
                history_signals[now.hour] = p_pico

                # --- PARTE C: NOTIFICAÇÃO ---
                is_signal = p_pico >= 0.85
                is_periodic = now.minute in [0, 30] and now.minute != last_periodic_check

                if is_signal or is_periodic:
                    # chart = get_ascii_temp_chart(series_today, history_signals)
                    print(f"[{now.strftime('%H:%M')}] P(PICO): {p_pico:.1%}")
                    last_periodic_check = now.minute
            else:
                print(f"[{now.strftime('%H:%M')}] ⚠️ Aguardando dados da estação...")

            time.sleep(60)

        except Exception as e:
            print(f"❌ Erro no loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
