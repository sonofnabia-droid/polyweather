import argparse
import os
import sys
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES ---
_BERLIN = ZoneInfo("Europe/Berlin")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip() # .strip() remove espaços acidentais

# Cores
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"

class TG:
    @staticmethod
    def send(msg):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            res = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": msg, 
                "parse_mode": "Markdown"
            }, timeout=10)
            if res.status_code != 200:
                print(f"❌ {R}Erro Telegram {res.status_code}: Verifique se deu /start no bot ou se o ID {TELEGRAM_CHAT_ID} está correto.{W}")
        except:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    mode_label = "REAL" if args.real else "PAPER"
    
    # Notificação de Arranque
    print(f"[{datetime.now().strftime('%T')}] {C}Bot Online | Modo: {mode_label} | Railway: {args.railway}{W}")
    TG.send(f"🚀 *Munich Bot Online*\nModo: `{mode_label}`\nRailway: `{args.railway}`")

    while True:
        now = datetime.now(tz=_BERLIN)
        
        # --- Lógica de Dados Real aqui ---
        temp_atual = 20 
        p_pico = 0.82 

        if args.railway:
            # LOG CLEAN (Para Railway)
            status_color = G if p_pico >= args.threshold else W
            print(f"[{now.strftime('%H:%M:%S')}] Temp: {temp_atual}°C | P(pico): {status_color}{p_pico:.1%}{W} | Mode: {mode_label}")
        else:
            # DASHBOARD (Opcional Local)
            print(f"\n--- MUNICH {mode_label} ---")
            print(f"P(pico): {p_pico:.1%} | Alvo: {args.threshold}")

        if p_pico >= args.threshold:
            TG.send(f"⚠️ *Alerta Munich*\nP(pico): {p_pico:.1%} | Temp: {temp_atual}°C")

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
