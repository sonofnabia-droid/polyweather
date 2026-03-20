import argparse
import os
import sys
import time
import warnings
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES ---
_BERLIN = ZoneInfo("Europe/Berlin")
WU_API_KEY = os.environ.get("WU_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Cores ANSI
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"
DIM = "\033[2m"

class TG:
    @staticmethod
    def send(msg):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print(f"⚠️ {Y}Telegram não configurado no Environment.{W}")
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            res = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": msg, 
                "parse_mode": "Markdown"
            }, timeout=10)
            if res.status_code != 200:
                print(f"❌ {R}Erro Telegram ({res.status_code}): {res.text}{W}")
            else:
                print(f"✅ {G}Telegram enviado com sucesso.{W}")
        except Exception as e:
            print(f"❌ {R}Falha de rede no Telegram: {e}{W}")

def draw_chart(series_today, signals):
    """Gera o gráfico ASCII para o modo local"""
    lines = []
    slots = [(h, m) for h in range(6, 22) for m in (0, 30)]
    temps = [series_today.get(s) for s in slots]
    avail = [t for t in temps if t is not None]
    if not avail: return ["  [Aguardando dados...]"]
    
    t_min, t_max = min(avail) - 0.5, max(avail) + 0.5
    t_rng = max(t_max - t_min, 1.0)
    chart_h = 6
    grid = [[" "] * (len(slots) * 2 + 5) for _ in range(chart_h)]
    
    for row in range(chart_h):
        t_val = t_max - (row / (chart_h - 1)) * t_rng
        grid[row][0:4] = list(f"{int(round(t_val)):>2}° ")
    
    for si, (slot, temp) in enumerate(zip(slots, temps)):
        if temp is None: continue
        row = int((1 - (temp - t_min) / t_rng) * (chart_h - 1))
        col = 4 + si * 2
        p = signals.get(slot[0], 0)
        color = G if p >= 0.8 else Y if p >= 0.5 else DIM
        grid[row][col] = f"{color}█{W}"
    return ["".join(row) for row in grid]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true", help="Ativa log minimalista")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    # Log de Inicialização
    start_time = datetime.now().strftime('%T')
    mode_label = "REAL" if args.real else "PAPER"
    print(f"[{start_time}] {C}Iniciando Munich Bot | Modo: {mode_label} | Railway: {args.railway}{W}")
    
    # Notificação de Arranque
    TG.send(f"🚀 *Munich Bot Online*\nModo: `{mode_label}`\nRailway: `{args.railway}`\nTarget: `{args.threshold}`")

    series_today = {}
    signals_history = {}

    while True:
        now = datetime.now(tz=_BERLIN)
        h, m = now.hour, now.minute
        
        # --- DADOS (Simulação - Substitua pelos seus reais) ---
        temp_atual = 21 
        slot_key = (h, 30 if m >= 30 else 0)
        series_today[slot_key] = temp_atual
        p_pico = 0.82
        signals_history[h] = p_pico

        if args.railway:
            # LOG CLEAN (Para Railway)
            status_color = G if p_pico >= args.threshold else W
            print(f"[{now.strftime('%H:%M:%S')}] Temp: {temp_atual}°C | P(pico): {status_color}{p_pico:.1%}{W} | Mode: {mode_label}")
        else:
            # DASHBOARD VISUAL (Para Local)
            print("\n" + "="*50)
            print(f"{C}DASHBOARD MUNICH{W} | {now.strftime('%H:%M')} | {mode_label}")
            for line in draw_chart(series_today, signals_history):
                print(line)
            print(f"P(pico): {G if p_pico >= args.threshold else W}{p_pico:.1%}{W} | Alvo: {args.threshold}")
            print("="*50)

        # Alerta de Sinal
        if p_pico >= args.threshold:
            TG.send(f"⚠️ *SINAL ALTO EM MUNICH*\nTemp: {temp_atual}°C\nP(pico): {p_pico:.1%}")

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
