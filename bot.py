
import os
from dotenv import load_dotenv
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

load_dotenv()

# ==================== CONFIGURAÇÃO ====================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN não encontrado! Define na Railway.")

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")

# Lista para guardar os chat_id dos usuários (para enviares notificações depois)
usuarios = []  # ← aqui vamos guardar quem interagiu com o bot


# ==================== MENU PRINCIPAL ====================
def criar_menu_principal():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("📢 Ver Status", callback_data="status"),
        InlineKeyboardButton("ℹ️ Sobre", callback_data="sobre"),
        InlineKeyboardButton("⚙️ Configurações", callback_data="config"),
        InlineKeyboardButton("❓ Ajuda", callback_data="ajuda")
    )
    return markup


# ==================== COMANDOS ====================
@bot.message_handler(commands=["start"])
def cmd_start(message):
    chat_id = message.chat.id
    
    # Guarda o usuário para poder enviar notificações depois
    if chat_id not in usuarios:
        usuarios.append(chat_id)
        print(f"Novo usuário adicionado: {chat_id}")
    
    bot.send_message(
        chat_id,
        "👋 Olá! Bem-vindo ao bot.\n\nEscolhe uma opção abaixo:",
        reply_markup=criar_menu_principal()
    )


@bot.message_handler(commands=["menu"])
def cmd_menu(message):
    bot.send_message(
        message.chat.id,
        "📋 Menu principal:",
        reply_markup=criar_menu_principal()
    )


# ==================== CALLBACKS (botões do menu) ====================
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if call.data == "status":
        bot.answer_callback_query(call.id, "Status atual: OK ✅")
        bot.send_message(call.message.chat.id, "✅ Tudo a funcionar normalmente!")
    
    elif call.data == "sobre":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Este bot foi criado para te enviar notificações automáticas.")
    
    elif call.data == "config":
        bot.answer_callback_query(call.id, "Em breve...")
        bot.send_message(call.message.chat.id, "⚙️ Configurações em desenvolvimento.")
    
    elif call.data == "ajuda":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Para receber notificações, basta manter o chat aberto.\n\nQualquer dúvida, escreve /ajuda.")


# ==================== FUNÇÃO PARA ENVIAR NOTIFICAÇÕES (usarás mais tarde) ====================
def enviar_notificacao_para_todos(mensagem: str):
    """Chama esta função quando o evento acontecer (mais tarde eu implemento a parte dos eventos)"""
    for chat_id in usuarios[:]:  # cópia para evitar problemas
        try:
            bot.send_message(chat_id, mensagem, parse_mode="HTML")
        except:
            # remove usuário que bloqueou o bot ou apagou o chat
            if chat_id in usuarios:
                usuarios.remove(chat_id)


# ==================== POLLING (roda 24/7) ====================
if __name__ == "__main__":
    print("🤖 Bot iniciado com sucesso!")
    bot.infinity_polling(none_stop=True)
