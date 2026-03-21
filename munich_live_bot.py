import os
import time
import warnings
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# --- CONFIGURAÇÕES DE TIMEZONE ---
_BERLIN = ZoneInfo("Europe/Berlin")

def berlin_now():
    return datetime.now(tz=_BERLIN)

def get_target_date():
    """Proteção: Após as 20:30 de Berlim, foca no mercado de amanhã."""
    now = berlin_now()
    if now.hour >= 20 and now.minute >= 30:
        return now.date() + timedelta(days=1)
    return now.date()

# --- GERADOR DE GRÁFICOS ASCII ---
def get_ascii_temp_chart(series_today, history_signals, day_start=6, day_end=21):
    """Gera a curva de temperatura com densidade de caracteres (simulando cores)."""
    hours = list(range(day_start, day_end + 1))
    avail = [t for t in series_today.values() if t is not None]
    if not avail: return "   (A aguardar dados da WU...)"
    
    t_min, t_max = min(avail), max(avail)
    rows = 6 
    grid = [["  " for _ in hours] for _ in range(rows)]
    
    for i, h in enumerate(hours):
        # Tenta ler temperatura da hora cheia :00 ou :30
        temp = series_today.get((h, 0)) or series_today.get((h, 30))
        if temp is None: continue
        
        # Mapeamento vertical (Y)
        y = int((rows-1) * (1 - (temp - t_min) / (max(t_max - t_min, 1))))
        
        # Simulação de Cores por Densidade
        p = history_signals.get(h, 0)
        if p >= 0.85:   sym = "██" # Verde (Pico detectado)
        elif p >= 0.60: sym = "▓▓" # Amarelo
        elif p >= 0.30: sym = "▒▒" # Laranja
        else:           sym = "░░" # Cinza/Base
        grid[y][i] = sym

    chart = []
    for r in range(rows):
        y_val = int(t_max - r * (t_max - t_min) / (rows-1))
        chart.append(f"{y_val:>2}° " + "".join(grid[r]))
    
    # Eixo X (Horas - apenas o último dígito para não quebrar layout mobile)
    chart.append("    " + " ".join([f"{h:02}"[1] for h in hours]))
    return "\n".join(chart)

# --- DASHBOARD TELEGRAM ---
def send_telegram_dashboard(snap, series_today, history_signals, p_pico, ev_data, target_date):
    chart = get_ascii_temp_chart(series_today, history_signals)
    now = berlin_now()
    
    # Barra de probabilidade visual
    bar_len = int(p_pico * 12)
    p_bar = "█" * bar_len + "░" * (12 - bar_len)

    msg = (
        f"📊 *MUNICH TEMP DASHBOARD* | `{target_date}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"```\n"
        f"CURVA DE TEMPERATURA\n"
        f"{chart}\n"
        f"----------------------------\n"
        f"TEMP: {snap['temp_c']}°C  | MAX: {snap['running_max']}°C\n"
        f"P(PICO): [{p_bar}] {p_pico:.1%}\n"
        f"STATUS: {'✅ PICO DETECTADO' if p_pico > 0.85 else '🔍 ANALISANDO'}\n"
        f"----------------------------\n"
        f"MERCADO: {snap['bracket_label']}\n"
        f"ASK: {ev_data['ask']*100:>4.1f}¢ | BID: {snap['bid']*100:>4.1f}¢\n"
        f"EDGE: +{ev_data['ev_cents']:.1f}¢ | KELLY: {ev_data['kelly']:.1%}\n"
        f"```\n"
        f"🕒 `Berlin: {now.strftime('%H:%M:%S')}`"
    )
    # TG.send(msg)  # Ativa a tua função de envio aqui
    print(msg) # Debug no terminal

# --- LOOP PRINCIPAL ---
def main():
    history_signals = {} 
    last_periodic_check = -1

    while True:
        try:
            target_date = get_target_date()
            now = berlin_now()
            
            # 1. RECOLHA (WU) E MODELO
            # [Simulação dos teus dados reais]
            # p_pico = modelo.predict(...)
            # snap = {"temp_c": 12, "running_max": 14, "bracket_label": "14°C", "bid": 0.02}
            
            # Atualiza histórico para o gráfico
            history_signals[now.hour] = p_pico 
            
            # 2. FINANCEIRO
            # ev_data = calculate_ev(p_pico, market_ask)

            # 3. LÓGICA DE ENVIO/EXECUÇÃO
            is_signal = (p_pico >= 0.85 and ev_data['ev_cents'] > 5)
            is_periodic = (now.minute in [0, 30] and now.minute != last_periodic_check)

            if is_signal:
                if MODE == "REAL":
                    # execute_trade(...)
                    pass
                send_telegram_dashboard(snap, series_today, history_signals, p_pico, ev_data, target_date)
            
            elif is_periodic:
                send_telegram_dashboard(snap, series_today, history_signals, p_pico, ev_data, target_date)
                last_periodic_check = now.minute

            time.sleep(60)

        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
