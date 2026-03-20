import argparse
import json
import os
import sys
import time
import warnings
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES ---
warnings.filterwarnings("ignore")
_BERLIN = ZoneInfo("Europe/Berlin")

# Variáveis de Ambiente
WU_API_KEY = os.environ.get("WU_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Cores ANSI para Log
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"
B, DIM = "\033[1m", "\033[2m"

# ══════════════════════════════════════════════════════
#  SISTEMA DE NOTIFICAÇÃO
# ══════════════════════════════════════════════════════
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
                print(f"❌ {R}Telegram Error {res.status_code}: {res.text}{W}")
        except Exception as e:
            print(f"❌ {R}Telegram Conn Error: {e}{W}")

# ══════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════
def draw_ascii(series):
    """Mini dashboard para modo local."""
    if not series: return ["  [Aguardando dados...]"]
    items = list(series.values())
    t_min, t_max = min(items)-1, max(items)+1
    return [f"Max Temp: {max(items)}°C", f"Min Temp: {min(items)}°C"]

# ══════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    mode_label = "REAL" if args.real else "PAPER"
    print(f"[{datetime.now().strftime('%T')}] {C}Iniciando Bot Munich...{W}")
    
    # Envio de Boas-vindas
    TG.send(f"🚀 *Munich Bot Online*\nModo: `{mode_label}`\nThreshold: `{args.threshold}`")

    series_today = {}

    try:
        while True:
            now = datetime.now(tz=_BERLIN)
            
            # --- SIMULAÇÃO DE DADOS (Substitua pela lógica real) ---
            temp_atual = 19.5 
            p_pico = 0.82
            bid, ask = 0.44, 0.49
            bracket = "19-20°C"
            
            slot = (now.hour, 30 if now.minute >= 30 else 0)
            series_today[slot] = temp_atual

            if args.railway:
                # Log Minimalista e Rico (Railway)
                p_color = G if p_pico > ask else W
                print(f"[{now.strftime('%H:%M:%S')}] {B}{temp_atual}°{W} | "
                      f"P:{p_color}{p_pico:.1%}{W} | {C}{bracket}{W} | "
                      f"CLOB:{G}{int(bid*100)}{W}/{R}{int(ask*100)}{W}¢")
            else:
                # Log Expandido (Local)
                print(f"\n--- {mode_label} DASHBOARD ---")
                print(f"Hora: {now.strftime('%T')} | Temp: {temp_atual}°C")
                print(f"Probabilidade: {p_pico:.1%} | Alvo: {args.threshold}")

            # Lógica de Alerta
            if p_pico >= args.threshold:
                TG.send(f"⚠️ *Alerta de Sinal*\nP(pico): {p_pico:.1%}\nAsk: {ask*100}¢")

            sys.stdout.flush()
            time.sleep(60)

    except KeyboardInterrupt:
        print(f"\n{Y}Bot interrompido pelo usuário.{W}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{R}ERRO FATAL: {e}{W}")
        TG.send(f"❌ *Bot Munich Crashou!*\nErro: `{e}`")
        sys.exit(1)

if __name__ == "__main__":
    main()
