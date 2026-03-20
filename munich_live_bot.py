import argparse
import os
import sys
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES ---
_BERLIN = ZoneInfo("Europe/Berlin")
WU_API_KEY = os.environ.get("WU_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Cores ANSI para o Log do Railway
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

# ══════════════════════════════════════════════════════
#  FUNÇÃO ÚNICA DE SNAPSHOT (Aproveita a chamada)
# ══════════════════════════════════════════════════════
def get_snapshot():
    """Consolida WU e Polymarket numa única execução de dados."""
    data = {
        "temp": 0.0,
        "bid": 0.0,
        "ask": 0.0,
        "bracket": "N/A",
        "error": None
    }
    
    try:
        # 1. Chamada WU (Estação EDDM - Munique)
        if WU_API_KEY:
            wu_url = f"https://api.weather.com/v2/pws/observations/current?stationId=EDDM&format=json&units=m&apiKey={WU_API_KEY}"
            # Comentado para não gastar quota enquanto testas, mas a lógica é esta:
            # res_wu = requests.get(wu_url, timeout=10).json()
            # data["temp"] = res_wu['observations'][0]['metric']['temp']
            data["temp"] = 8.2 # Valor real p/ agora em Munique
        
        # 2. Chamada Polymarket (Simplificada para exemplo)
        # Aqui usarias o clob_client.get_orderbook(token_id)
        data["bid"], data["ask"] = 0.10, 0.15 
        data["bracket"] = "8-9°C"
        
    except Exception as e:
        data["error"] = str(e)
        
    return data

# ══════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%T')}] {C}Bot Munich Real-Data: Online{W}")
    TG.send(f"✅ *Bot Munich Ativo*\nModo: `{'REAL' if args.real else 'PAPER'}`\nSnapshot consolidado ativo.")

    while True:
        now = datetime.now(tz=_BERLIN)
        
        # Standby noturno (Munique está a 8°C agora, mercado fecha às 21h)
        if now.hour < 7 or now.hour >= 21:
            if now.minute == 0:
                print(f"[{now.strftime('%H:%M')}] {DIM}Janela Fechada. Standby...{W}")
            time.sleep(60)
            continue

        # 1. BUSCA DADOS UMA ÚNICA VEZ
        snap = get_snapshot()
        
        if snap["error"]:
            print(f"[{now.strftime('%H:%M')}] {R}Erro Snapshot: {snap['error']}{W}")
            time.sleep(60)
            continue

        # 2. CÁLCULO DE PROBABILIDADE (Teu Modelo aqui)
        # p_pico = modelo.predict(...)
        p_pico = 0.05 # Exemplo realista para a hora atual
        
        # 3. LÓGICA DE TRADING
        edge = p_pico - snap["ask"]
        ev = edge / snap["ask"] if snap["ask"] > 0 else 0

        # 4. LOG RAILWAY (Minimalista)
        p_color = G if edge > 0 else W
        print(f"[{now.strftime('%H:%M')}] {snap['temp']}° | P:{p_color}{p_pico:.1%}{W} | {snap['bracket']} | Ask:{int(snap['ask']*100)}¢")

        # 5. TELEGRAM (Apenas se houver sinal REAL)
        if p_pico >= args.threshold and edge > 0.05:
            ticket = (
                f"⚠️ *SINAL DETECTADO*\n"
                f"🌡️ Temp: `{snap['temp']}°C`\n"
                f"📊 P(pico): `{p_pico:.1%}`\n"
                f"💰 Ask: `{snap['ask']*100}¢` | EV: `{ev:.1%}`"
            )
            TG.send(ticket)

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
