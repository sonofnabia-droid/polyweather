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

# Variáveis de Ambiente (Railway)
WU_API_KEY = os.environ.get("WU_API_KEY", "")
POLY_PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Cores ANSI para o Log do Railway
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"
B, DIM = "\033[1m", "\033[2m"

# ══════════════════════════════════════════════════════
#  SISTEMA DE NOTIFICAÇÃO (TELEGRAM RICO)
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
        except:
            pass

# ══════════════════════════════════════════════════════
#  LOGICA DE EXECUÇÃO
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()

    mode_label = "REAL" if args.real else "PAPER"
    
    # Mensagem de Inicialização
    print(f"[{datetime.now().strftime('%T')}] {C}Bot Munich Ativo | Modo: {mode_label}{W}")
    TG.send(f"🚀 *Munich Bot Online*\n━━━━━━━━━━━━━━\nModo: `{mode_label}`\nAlvo: `{args.threshold}`\nBankroll: `${args.bankroll}`")

    try:
        while True:
            now = datetime.now(tz=_BERLIN)
            
            # --- 1. DADOS (SUBSTITUA PELAS SUAS FUNÇÕES REAIS) ---
            temp_atual = 20.2      # vindo de fetch_wu_latest()
            p_pico = 0.88          # vindo de predict_p()
            bid, ask = 0.48, 0.53  # vindo de fetch_market()
            bracket = "20-21°C"    # identificação do mercado
            
            # --- 2. CÁLCULOS DE TRADING ---
            edge = p_pico - ask
            ev = edge / ask if ask > 0 else 0
            
            # --- 3. LOG NO RAILWAY (Minimalista + Inteligente) ---
            # Cor verde no P se o EV for positivo (P > Ask)
            p_color = G if edge > 0 else W
            status_tag = f"{G}[BET]{W}" if args.real and edge > 0.05 else f"{DIM}[OBS]{W}"
            
            log_line = (
                f"[{now.strftime('%H:%M:%S')}] "
                f"{B}{temp_atual:>4.1f}°{W} | "
                f"P:{p_color}{p_pico:>5.1%}{W} | "
                f"MKT:{G}{int(bid*100):>2}{W}/{R}{int(ask*100):<2}{W}¢ | "
                f"EV:{ev:>+5.1%} | {status_tag}"
            )
            print(log_line)

            # --- 4. TELEGRAM (Ticket Detalhado) ---
            # Envia alerta se P atingir threshold ou se houver oportunidade clara
            if p_pico >= args.threshold and edge > 0:
                ticket = (
                    f"⚠️ *OPORTUNIDADE DETECTADA*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🌡️ *Temp Atual:* `{temp_atual}°C`\n"
                    f"🎯 *Mercado:* `{bracket}`\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 *Modelo P:* `{p_pico:.1%}`\n"
                    f"💰 *Preço Buy:* `{ask*100:.1f}¢`\n"
                    f"📈 *Expected Value:* `+{ev:.1%}`\n"
                    f"⚖️ *Edge:* `+{edge*100:.1f}¢`\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🤖 *Ação:* `{'EXECUTANDO ORDEM' if args.real else 'PAPER SIGNAL'}`\n"
                    f"🕒 `{now.strftime('%H:%M:%S')} Munique`"
                )
                TG.send(ticket)
                
                if args.real:
                    # Chame sua função de order_execution aqui
                    pass

            sys.stdout.flush()
            time.sleep(60)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"{R}Erro Fatal: {e}{W}")
        TG.send(f"❌ *Bot Munich Crashou!*\nErro: `{e}`")
        sys.exit(1)

if __name__ == "__main__":
    main()
