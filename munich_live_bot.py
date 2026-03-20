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

# --- AMBIENTE ---
WU_API_KEY = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ANSI COLORS (O log do Railway suporta cores se PYTHONUNBUFFERED=1)
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
        except Exception as e:
            print(f"Erro Telegram: {e}")

# ══════════════════════════════════════════════════════
#  LÓGICA DE CLIMA E DASHBOARD
# ══════════════════════════════════════════════════════

def get_berlin_now():
    return datetime.now(tz=_BERLIN)

def draw_chart(series_today, signals):
    """Gera o gráfico de temperatura em ASCII"""
    lines = []
    # Janela operacional 06h - 21h
    slots = [(h, m) for h in range(6, 22) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    
    if not avail: return ["  [Aguardando dados de Munich...]"]
    
    t_min, t_max = min(avail) - 0.5, max(avail) + 0.5
    t_rng = max(t_max - t_min, 1.0)
    chart_h = 8
    
    col_w = 2
    total_w = len(slots) * col_w + 5
    grid = [[" "] * total_w for _ in range(chart_h)]
    
    # Escala Y
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        label = f"{int(round(t_val)):>2}° "
        grid[row][0:4] = list(label)
    
    # Plotar Pontos
    for si, (slot, temp) in enumerate(zip(slots, temps)):
        if temp is None: continue
        row = int((1 - (temp - t_min) / t_rng) * (chart_h - 1))
        col = 4 + si * col_w
        p = signals.get(slot[0], 0)
        
        # Cor baseada na probabilidade do pico
        color = C["green"] if p >= 0.8 else C["yellow"] if p >= 0.5 else C["gray"]
        if 0 <= row < chart_h:
            grid[row][col] = f"{color}█{R}"

    return ["".join(row) for row in grid]

# ══════════════════════════════════════════════════════
#  LOOP PRINCIPAL (NON-INTERACTIVE)
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    print(f"{B}{C['cyan']}=== MUNICH LIVE BOT: STARTING CONTAINER ==={R}")
    TG.send("🚀 *Bot Munich Online no Railway*")

    # Estados persistentes na memória do script
    series_today = {}    # {(hora, min): temp}
    signals_history = {} # {hora: p_pico}
    
    while True:
        now_berlin = get_berlin_now()
        h, m = now_berlin.hour, now_berlin.minute
        
        # --- 1. COLETA DE DADOS (Placeholder para sua função WU) ---
        # Substitua aqui pela chamada real: temp_atual = fetch_wu_latest(WU_API_KEY)
        temp_atual = 15 + (h % 10) # Simulação
        slot_key = (h, 30 if m >= 30 else 0)
        series_today[slot_key] = temp_atual
        
        # --- 2. MODELO (Placeholder para o seu LightGBM) ---
        # p_pico = predict_p(model, feat_cols, slots_so_far, ...)
        p_pico = 0.40 if h < 14 else 0.88 # Simulação de pico à tarde
        signals_history[h] = p_pico

        # --- 3. DASHBOARD RENDER (No Railway não usamos 'clear') ---
        print("\n" + "="*60)
        print(f"{B}MUNICH LIVE DASHBOARD{R} | {now_berlin.strftime('%d/%m/%Y %H:%M')}")
        print(f"Status: {'[REAL MODE]' if args.real else '[PAPER MODE]'}")
        print(f"Temp Atual: {B}{temp_atual}°C{R} | P(pico): {p_pico:.2%}")
        print("-" * 60)
        
        # Gráfico
        for line in draw_chart(series_today, signals_history):
            print(line)
        print(f"{DIM}Horas:  06  08  10  12  14  16  18  20{R}")
        print("="*60)

        # --- 4. LÓGICA DE TRADING ---
        if p_pico >= args.threshold:
            ev = p_pico - 0.50 # Exemplo: P - Preço do mercado
            if ev > 0:
                msg = f"💰 *OPORTUNIDADE:* P({p_pico:.1%}) > T({args.threshold})\nEV: {ev*100:.1f}¢ | Temp: {temp_atual}°C"
                print(f"{C['green']}{msg}{R}")
                TG.send(msg)
                
                if args.real:
                    # Chamar sua função de ordem real aqui
                    print(f"{C['orange']}Executando Ordem Real no CLOB...{R}")

        # Forçar o Railway a mostrar o log agora
        sys.stdout.flush()
        
        # Dorme 60 segundos
        time.sleep(60)

if __name__ == "__main__":
    main()import argparse
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

# --- AMBIENTE ---
WU_API_KEY = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ANSI COLORS (O log do Railway suporta cores se PYTHONUNBUFFERED=1)
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
        except Exception as e:
            print(f"Erro Telegram: {e}")

# ══════════════════════════════════════════════════════
#  LÓGICA DE CLIMA E DASHBOARD
# ══════════════════════════════════════════════════════

def get_berlin_now():
    return datetime.now(tz=_BERLIN)

def draw_chart(series_today, signals):
    """Gera o gráfico de temperatura em ASCII"""
    lines = []
    # Janela operacional 06h - 21h
    slots = [(h, m) for h in range(6, 22) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    
    if not avail: return ["  [Aguardando dados de Munich...]"]
    
    t_min, t_max = min(avail) - 0.5, max(avail) + 0.5
    t_rng = max(t_max - t_min, 1.0)
    chart_h = 8
    
    col_w = 2
    total_w = len(slots) * col_w + 5
    grid = [[" "] * total_w for _ in range(chart_h)]
    
    # Escala Y
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        label = f"{int(round(t_val)):>2}° "
        grid[row][0:4] = list(label)
    
    # Plotar Pontos
    for si, (slot, temp) in enumerate(zip(slots, temps)):
        if temp is None: continue
        row = int((1 - (temp - t_min) / t_rng) * (chart_h - 1))
        col = 4 + si * col_w
        p = signals.get(slot[0], 0)
        
        # Cor baseada na probabilidade do pico
        color = C["green"] if p >= 0.8 else C["yellow"] if p >= 0.5 else C["gray"]
        if 0 <= row < chart_h:
            grid[row][col] = f"{color}█{R}"

    return ["".join(row) for row in grid]

# ══════════════════════════════════════════════════════
#  LOOP PRINCIPAL (NON-INTERACTIVE)
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    print(f"{B}{C['cyan']}=== MUNICH LIVE BOT: STARTING CONTAINER ==={R}")
    TG.send("🚀 *Bot Munich Online no Railway*")

    # Estados persistentes na memória do script
    series_today = {}    # {(hora, min): temp}
    signals_history = {} # {hora: p_pico}
    
    while True:
        now_berlin = get_berlin_now()
        h, m = now_berlin.hour, now_berlin.minute
        
        # --- 1. COLETA DE DADOS (Placeholder para sua função WU) ---
        # Substitua aqui pela chamada real: temp_atual = fetch_wu_latest(WU_API_KEY)
        temp_atual = 15 + (h % 10) # Simulação
        slot_key = (h, 30 if m >= 30 else 0)
        series_today[slot_key] = temp_atual
        
        # --- 2. MODELO (Placeholder para o seu LightGBM) ---
        # p_pico = predict_p(model, feat_cols, slots_so_far, ...)
        p_pico = 0.40 if h < 14 else 0.88 # Simulação de pico à tarde
        signals_history[h] = p_pico

        # --- 3. DASHBOARD RENDER (No Railway não usamos 'clear') ---
        print("\n" + "="*60)
        print(f"{B}MUNICH LIVE DASHBOARD{R} | {now_berlin.strftime('%d/%m/%Y %H:%M')}")
        print(f"Status: {'[REAL MODE]' if args.real else '[PAPER MODE]'}")
        print(f"Temp Atual: {B}{temp_atual}°C{R} | P(pico): {p_pico:.2%}")
        print("-" * 60)
        
        # Gráfico
        for line in draw_chart(series_today, signals_history):
            print(line)
        print(f"{DIM}Horas:  06  08  10  12  14  16  18  20{R}")
        print("="*60)

        # --- 4. LÓGICA DE TRADING ---
        if p_pico >= args.threshold:
            ev = p_pico - 0.50 # Exemplo: P - Preço do mercado
            if ev > 0:
                msg = f"💰 *OPORTUNIDADE:* P({p_pico:.1%}) > T({args.threshold})\nEV: {ev*100:.1f}¢ | Temp: {temp_atual}°C"
                print(f"{C['green']}{msg}{R}")
                TG.send(msg)
                
                if args.real:
                    # Chamar sua função de ordem real aqui
                    print(f"{C['orange']}Executando Ordem Real no CLOB...{R}")

        # Forçar o Railway a mostrar o log agora
        sys.stdout.flush()
        
        # Dorme 60 segundos
        time.sleep(60)

if __name__ == "__main__":
    main()
