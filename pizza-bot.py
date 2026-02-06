import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
import whisper # Локальный STT

# Твои импорты LangChain и Pydantic (OrderState, chain, retriever)
# ...
from langchain_openai import ChatOpenAI
# Импортируй свои ранее созданные OrderState, retriever и prompt здесь
from typing import List, Optional
from pydantic import BaseModel, Field, SecretStr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

API_TOKEN = 'ТВОЙ_ТЕЛЕГРАМ_ТОКЕН'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Загружаем модель Whisper локально (она отлично работает на Arch)
stt_model = whisper.load_model("base")

# Словарь для хранения корзин пользователей (вместо st.session_state)
user_carts = {}

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="")
    await message.answer("Привет! Я бот-пиццерия. Можешь писать текстом или прислать голосовое!")

# ОБРАБОТКА ГОЛОСА
@dp.message(F.voice)
async def handle_voice(message: types.Message):
    # 1. Скачиваем файл
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = f"{file_id}.ogg"
    await bot.download_file(file.file_path, file_path)

    # 2. Транскрибация (Whisper)
    result = stt_model.transcribe(file_path)
    user_text = result['text']
    os.remove(file_path) # Чистим за собой

    # 3. Отправляем в твою логику заказа
    await process_order_logic(message, user_text)

# ОБРАБОТКА ТЕКСТА
@dp.message(F.text)
async def handle_text(message: types.Message):
    await process_order_logic(message, message.text)

async def process_order_logic(message: types.Message, user_text: str):
    user_id = message.from_user.id
    
    # Достаем или создаем корзину
    current_cart = user_carts.get(user_id, OrderState(items=[], total_price=0))
    
    # Твоя RAG логика
    context_docs = retriever.invoke(user_text)
    context_text = "\n".join([d.page_content for d in context_docs])
    
    # Вызов Qwen3
    new_state = chain.invoke({
        "input": user_text,
        "context": context_text,
        "current_order": current_cart.model_dump_json(),
        "chat_history": [] # Можно добавить историю из БД
    })

    # Сохраняем состояние
    user_carts[user_id] = new_state

    # Формируем красивый ответ
    cart_msg = "\n".join([f"• {i.name} ({i.size}) x{i.quantity}" for i in new_state.items])
    full_response = (
        f"{new_state.message_to_user}\n\n"
        f"🛒 **Текущая корзина:**\n{cart_msg}\n"
        f"💰 **Итого: {new_state.total_price} ₽**"
    )
    
    await message.answer(full_response, parse_mode="Markdown")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
