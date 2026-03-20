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

# --- AMBIENTE (Configurar no Painel do Railway) ---
WU_API_KEY = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ANSI COLORS para o Log do Railway
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
C={"green":"\033[92m","yellow":"\033[93m","orange":"\033[33m","red":"\033[91m","cyan":"\033[96m","gray":"\033[90m"}

# ══════════════════════════════════════════════════════
#  NOTIFICAÇÕES TELEGRAM
# ══════════════════════════════════════════════════════
class TG:
    @staticmethod
    def send(msg):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        except: pass

# ══════════════════════════════════════════════════════
#  FUNÇÕES DE DASHBOARD ASCII (Corrigido)
# ══════════════════════════════════════════════════════
def draw_chart(series_today, signals):
    lines = []
    slots = [(h, m) for h in range(6, 22) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    
    if not avail: return ["  " + DIM + "[Aguardando dados de Munich...]" + R]
    
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
        color = C["green"] if p >= 0.8 else C["yellow"] if p >= 0.5 else C["gray"]
        if 0 <= row < chart_h:
            grid[row][col] = f"{color}█{R}"
            if col+1 < len(grid[row]): grid[row][col+1] = "" 

    return ["".join(row) for row in grid]

# ══════════════════════════════════════════════════════
#  LOOP DE OPERAÇÃO
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    print(f"{B}{C['cyan']}=== MUNICH LIVE BOT OPERACIONAL ==={R}")
    TG.send("🚀 *Bot Munich Online no Railway*")

    series_today = {}
    signals_history = {}

    while True:
        now_berlin = datetime.now(tz=_BERLIN)
        h, m = now_berlin.hour, now_berlin.minute
        
        # --- COLETA E MODELO (Simulado - Integre suas funções aqui) ---
        # Ex: temp_atual = fetch_wu_latest(WU_API_KEY)
        temp_atual = 18 + (h % 6) 
        slot_key = (h, 30 if m >= 30 else 0)
        series_today[slot_key] = temp_atual
        
        # Ex: p_pico = predict_p(...)
        p_pico = 0.45 if h < 13 else 0.82
        signals_history[h] = p_pico

        # --- RENDERIZAÇÃO ---
        print("\n" + "="*65)
        print(f"{B}DASHBOARD MUNICH{R} | {now_berlin.strftime('%H:%M:%S')} | {C['green'] if args.real else C['yellow']}{'REAL' if args.real else 'PAPER'}{R}")
        print(f"TEMP: {B}{temp_atual}°C{R} | P(PICO): {p_pico:.1%} | TARGET: {args.threshold}")
        print("-" * 65)
        
        for line in draw_chart(series_today, signals_history):
            print(line)
        
        print(f"{DIM}Eixo:  06  08  10  12  14  16  18  20  22 (Horas Berlin){R}")
        print("="*65)

        # --- TRADING ---
        if p_pico >= args.threshold:
            msg = f"⚠️ *SINAL:* P({p_pico:.1%}) atingiu o alvo!"
            print(f"{C['orange']}{msg}{R}")
            TG.send(msg)
            # if args.real: executar_ordem(...)

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
