import argparse
import json
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

# --- SUPRESSÃO DE WARNINGS ---
warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES DE TIMEZONE ---
_BERLIN = ZoneInfo("Europe/Berlin")
_LOCAL  = ZoneInfo("Europe/Lisbon")

# --- CONFIGURAÇÕES DE PATH E AMBIENTE ---
MODEL_LGB    = Path("munich_peak_model/lgbm_peak.pkl")
MODEL_CONFIG = Path("munich_peak_model/peak_model_config.json")
WU_API_KEY   = os.environ.get("WU_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "") # Adicione no Railway
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ANSI COLORS para o Dashboard
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
C={"green":"\033[92m","yellow":"\033[93m","orange":"\033[33m","red":"\033[91m","cyan":"\033[96m","gray":"\033[90m"}

# ══════════════════════════════════════════════════════
#  CLASSE TELEGRAM (Simples)
# ══════════════════════════════════════════════════════
class TG:
    @staticmethod
    def send(msg):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        except: pass

# ══════════════════════════════════════════════════════
#  FUNÇÕES DE DASHBOARD ASCII
# ══════════════════════════════════════════════════════
def draw_chart(series_today, signals):
    """Gera o gráfico de temperatura em ASCII para o log do Railway"""
    lines = []
    slots = [(h, m) for h in range(6, 21) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    if not avail: return ["  (Aguardando dados para gráfico...)"]
    
    t_min, t_max = min(avail) - 0.5, max(avail) + 0.5
    t_rng = max(t_max - t_min, 1.0)
    chart_h = 8
    
    col_w = 2
    grid = [[" "] * (len(slots) * col_w + 5) for _ in range(chart_h)]
    
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        grid[row][0:4] = list(f"{int(round(t_val)):>2}° ")
    
    for si, (slot, temp) in enumerate(zip(slots, temps)):
        if temp is None: continue
        row = int((1 - (temp - t_min) / t_rng) * (chart_h - 1))
        col = 4 + si * col_w
        p = signals.get(slot[0], 0)
        color = C["green"] if p > 0.8 else C["yellow"] if p > 0.5 else C["gray"]
        grid[row][col] = f"{color}█{R}"

    return ["".join(row) for row in grid]

# ══════════════════════════════════════════════════════
#  LOGICA PRINCIPAL
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bankroll", type=float, default=50.0)
    args = parser.parse_args()

    # Bootstrap inicial
    print(f"{B}{C['cyan']}=== MUNICH LIVE BOT OPERACIONAL ==={R}")
    TG.send("🚀 *Bot Munich Iniciado no Railway*")

    # Dicionários de estado (Simulando persistência na sessão)
    series_today = {} 
    signals_history = {}

    while True:
        now_berlin = datetime.now(tz=_BERLIN)
        h, m = now_berlin.hour, now_berlin.minute
        
        # 1. Simulação de coleta de dados (Substitua pela sua função WU real)
        # temp_atual = fetch_wu_latest(...)
        temp_atual = 18 + (h % 5) # Exemplo fictício
        slot_key = (h, 30 if m >= 30 else 0)
        series_today[slot_key] = temp_atual
        
        # 2. Predição (Substitua pela chamada do modelo LightGBM)
        p_pico = 0.42 + (h/50) # Exemplo fictício que sobe com o dia
        signals_history[h] = p_pico

        # 3. RENDERIZAÇÃO DO DASHBOARD NO LOG
        os.system('clear' if os.name == 'posix' else 'cls') # Tenta limpar log
        print("-" * 50)
        print(f" HORA: {now_berlin.strftime('%H:%M:%S')} | MODO: {'REAL' if args.real else 'PAPER'}")
        print(f" TEMP ATUAL: {temp_atual}°C | P(PICO): {p_pico:.2%}")
        print("-" * 50)
        
        for line in draw_chart(series_today, signals_history):
            print(line)
        
        print(f"{DIM}Eixo X: 06h . . . . . . . . 20h{R}")
        print("-" * 50)

        # 4. LÓGICA DE TRADING E TELEGRAM
        if p_pico >= args.threshold:
            msg = f"⚠️ *SINAL ALTO:* P({p_pico:.2%}) atingiu threshold {args.threshold}!"
            print(f"{C['orange']}{msg}{R}")
            TG.send(msg)
            # Executar Ordem Real aqui se args.real...

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
