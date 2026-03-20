import argparse
import os
import sys
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES ---
_BERLIN = ZoneInfo("Europe/Berlin")
G, Y, R, C, W = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[0m"
DIM = "\033[2m"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--railway", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    mode_label = f"{G}REAL{W}" if args.real else f"{Y}PAPER{W}"
    print(f"[{datetime.now().strftime('%T')}] {C}Bot Smart-Log Ativo | Modo: {mode_label}{W}")

    while True:
        now = datetime.now(tz=_BERLIN)
        
        # --- DADOS SIMULADOS (Substitua pela lógica de mercado real) ---
        temp_atual = 20.4
        p_pico = 0.82
        
        # Informações do Mercado (Exemplo vindo do seu fetch_market/clob)
        bracket_label = "20-21°C"
        bid = 0.48
        ask = 0.52
        spread = ask - bid
        
        # Lógica de Cor para Probabilidade vs Preço
        # Se P(pico) > Ask, temos EV positivo
        ev_color = G if p_pico > ask else W
        
        # LOG SMART-MINIMALISTA (Tudo em uma linha informativa)
        # Formato: [HORA] TEMP | P(PICO) | BRACKET | BID/ASK | SPREAD
        log_line = (
            f"[{now.strftime('%H:%M:%S')}] "
            f"{B}{temp_atual}°C{W} | "
            f"P:{ev_color}{p_pico:.1%}{W} | "
            f"{C}{bracket_label}{W} | "
            f"CLOB:{G}{bid*100:>2.0f}{W}/{R}{ask*100:<2.0f}{W}¢ | "
            f"Spr:{DIM}{spread*100:.1f}¢{W}"
        )
        
        print(log_line)

        # Disparo de Telegram se houver sinal
        if p_pico >= args.threshold:
            # Recomendo enviar um log mais rico no Telegram já que lá não polui o terminal
            pass

        sys.stdout.flush()
        time.sleep(60)

if __name__ == "__main__":
    main()
