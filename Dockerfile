FROM python:3.12-slim

# 🔧 Instalar dependências do sistema (CRÍTICO para lightgbm)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 📁 Working dir
WORKDIR /app

# 📥 Copiar requirements primeiro (cache optimization)
COPY requirements.txt .

# 📦 Instalar deps Python
RUN pip install --no-cache-dir -r requirements.txt

# 📁 Copiar resto do projeto
COPY . .

# 🚀 Run
CMD ["python", "munich_live_bot.py", "--mode", "real", "--headless"]

ENV TELEGRAM_CHAT_ID=1002595623810
ENV TELEGRAM_TOKEN="8667652003:AAG2wPIJTpCJ4Yy6BLzFb4yLZMMXTfKQWyE"
ENV POLY_PRIVATE_KEY="0x89031bb471ac2e8f59d9a4201e4635f463ab15c2438f6d75771c5c1eedfae387"
ENV WU_API_KEY="e1f10a1e78da46f5b10a1e78da96f525"