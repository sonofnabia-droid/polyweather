import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# ────────────────────────────────────────────────
#               CONFIGURAÇÃO
# ────────────────────────────────────────────────

# Token em hardcode (substitui "XYZ" pelo teu token real)
TOKEN = "8711370296:AAFP3_cnhDt6H8gUN-1-YaNL754BUm6kVYs"

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")

# Lista de chat_ids para enviar notificações depois
usuarios = []


# ────────────────────────────────────────────────
#               MENU PRINCIPAL
# ────────────────────────────────────────────────

def criar_menu_principal():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("📢 Ver Status", callback_data="status"),
        InlineKeyboardButton("ℹ️ Sobre", callback_data="sobre"),
        InlineKeyboardButton("⚙️ Configurações", callback_data="config"),
        InlineKeyboardButton("❓ Ajuda", callback_data="ajuda")
    )
    return markup


# ────────────────────────────────────────────────
#               COMANDOS
# ────────────────────────────────────────────────

@bot.message_handler(commands=["start"])
def cmd_start(message):
    chat_id = message.chat.id
    if chat_id not in usuarios:
        usuarios.append(chat_id)
        print(f"Novo usuário: {chat_id}")

    bot.send_message(
        chat_id,
        "👋 Olá! Bem-vindo ao bot.\n\nEscolhe uma opção:",
        reply_markup=criar_menu_principal()
    )


@bot.message_handler(commands=["menu"])
def cmd_menu(message):
    bot.send_message(
        message.chat.id,
        "📋 Menu principal:",
        reply_markup=criar_menu_principal()
    )


# ────────────────────────────────────────────────
#               CALLBACKS (botões)
# ────────────────────────────────────────────────

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if call.data == "status":
        bot.answer_callback_query(call.id, "Status: OK ✅")
        bot.send_message(call.message.chat.id, "✅ Tudo a funcionar!")
    
    elif call.data == "sobre":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Bot de notificações automáticas.")
    
    elif call.data == "config":
        bot.answer_callback_query(call.id, "Em breve...")
        bot.send_message(call.message.chat.id, "⚙️ Configurações (em desenvolvimento)")
    
    elif call.data == "ajuda":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "Mantém o chat aberto para receber notificações.")


# ────────────────────────────────────────────────
#               FUNÇÃO DE NOTIFICAÇÃO
# ────────────────────────────────────────────────

def enviar_notificacao_para_todos(texto: str):
    for chat_id in usuarios[:]:
        try:
            bot.send_message(chat_id, texto, parse_mode="HTML")
        except Exception as e:
            print(f"Erro ao enviar para {chat_id}: {e}")
            if chat_id in usuarios:
                usuarios.remove(chat_id)


# ────────────────────────────────────────────────
#               INÍCIO
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Bot iniciado")
    bot.infinity_polling(none_stop=True, interval=0, timeout=30)
