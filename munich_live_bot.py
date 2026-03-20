import argparse
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

# Cores ANSI
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"
B, DIM = "\033[1m", "\033[2m"

class TG:
    @staticmethod
    def send(msg):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        except: pass

def get_snapshot():
    """Consolida dados reais (Placeholder para as tuas funções WU/CLOB)."""
    # Exemplo de valores que viriam das tuas APIs
    return {
        "temp": 8.5,
        "p_pico": 0.12,
        "bid": 0.10,
        "ask": 0.15,
        "bracket": "8-9°C",
        "market_active": True
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()

    # --- EVENTO: ARRANQUE ---
    start_time = datetime.now(tz=_BERLIN).strftime('%H:%M:%S')
    print(f"[{start_time}] {C}Bot Munich Online (Modo: {'REAL' if args.real else 'PAPER'}){W}")
    TG.send(f"🚀 *Bot Munich Iniciado*\nSincronizado às `{start_time}`\nModo: `{'REAL' if args.real else 'PAPER'}`")

    last_tg_hour = -1
    last_tg_minute = -1
    in_standby = False

    while True:
        now = datetime.now(tz=_BERLIN)
        h, m = now.hour, now.minute

        # --- LÓGICA DE JANELA OPERACIONAL ---
        if h < 7 or h >= 21:
            if not in_standby:
                TG.send("🌙 *Janela Fechada:* Munique entrou em standby (21h-07h).")
                in_standby = True
            if m == 0: # Log minimalista no Railway 1x por hora à noite
                print(f"[{now.strftime('%H:%M')}] {DIM}Standby Noturno...{W}")
            time.sleep(60)
            continue
        
        if in_standby:
            TG.send("☀️ *Janela Aberta:* Bot a retomar monitorização.")
            in_standby = False

        # 1. OBTER DADOS (Minuto a Minuto para o Log)
        snap = get_snapshot()
        edge = snap["p_pico"] - snap["ask"]

        # 2. LOG RAILWAY (Sempre minuto a minuto)
        p_color = G if edge > 0 else W
        print(f"[{now.strftime('%H:%M:%S')}] {snap['temp']}° | P:{p_color}{snap['p_pico']:.1%}{W} | Ask:{int(snap['ask']*100)}¢")

        # 3. LÓGICA DE TELEGRAM (Event-Based & 30/30 min)
        # Evento A: Oportunidade Real (Sinal acima do Threshold)
        triggered_signal = (snap["p_pico"] >= args.threshold and edge > 0.05)
        
        # Evento B: Periodicidade (00 ou 30 minutos)
        is_periodic_time = (m in [0, 30] and m != last_tg_minute)

        if triggered_signal or is_periodic_time:
            # Se for sinal, usamos emoji de alerta, se for periódico, emoji de info
            emoji = "⚠️ *SINAL*" if triggered_signal else "📊 *UPDATE*"
            
            ticket = (
                f"{emoji}\n"
                f"🌡️ Temp: `{snap['temp']}°C`\n"
                f"📈 P(pico): `{snap['p_pico']:.1%}`\n"
                f"💰 Ask: `{snap['ask']*100}¢` | Bracket: `{snap['bracket']}`\n"
                f"🕒 `{now.strftime('%H:%M')}`"
            )
            TG.send(ticket)
            last_tg_minute = m # Evita enviar múltiplos no mesmo minuto

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
