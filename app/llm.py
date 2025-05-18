import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
load_dotenv()

MODEL = "gemini-1.5-flash"
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Prompt sistem yang diperkuat
system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.
Your answers must:
- Be written in polite and easily understandable Indonesian.
- Be short and to the point (maximum 2–3 sentences, under 50 words).
- Always respond in Indonesian, even if the input is in another language (e.g., English).
- Avoid repeating the user's question; respond directly with the answer.
Example tone:
User: Cuaca hari ini gimana?
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat.
User: What is the capital of Indonesia?
Assistant: Ibu kota Indonesia adalah Jakarta.
If you're unsure about an answer, say: "Maaf, saya tidak tahu jawabannya."
"""

# Konfigurasi API Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Inisialisasi konfigurasi model
generation_config = {
    "temperature": 0.5,  # Untuk respons lebih deterministik
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 50,  # Batasi panjang respons
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Struktur untuk menyimpan history chat
chat_history = []

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    global chat_history
    if not os.path.exists(CHAT_HISTORY_FILE):
        chat_history = []
        return
    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        chat_history = []
        return
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        chat_history = []

load_chat_history()

def translate_to_indonesian(prompt: str) -> str:
    """
    Terjemahkan prompt ke Bahasa Indonesia jika dalam bahasa lain.
    Args:
        prompt (str): Teks input.
    Returns:
        str: Teks dalam Bahasa Indonesia.
    """
    if not LANGDETECT_AVAILABLE:
        return prompt
    try:
        lang = detect(prompt)
        if lang != "id":
            translate_prompt = f"Terjemahkan ke Bahasa Indonesia: {prompt}"
            model = genai.GenerativeModel(
                model_name=MODEL,
                generation_config={"max_output_tokens": 50},
                safety_settings=safety_settings
            )
            response = model.generate_content(translate_prompt)
            return response.text.strip()
        return prompt
    except Exception as e:
        print(f"[ERROR] Gagal mendeteksi/menterjemahkan bahasa: {e}")
        return prompt

def generate_response(prompt: str) -> str:
    """
    Kirim prompt ke LLM dan kembalikan respons teks.
    Args:
        prompt (str): Teks input.
    Returns:
        str: Respons dalam Bahasa Indonesia.
    """
    try:
        global chat_history
        
        # Terjemahkan prompt jika perlu
        prompt = translate_to_indonesian(prompt)
        
        # Gabungkan system_instruction dengan prompt
        full_prompt = f"{system_instruction}\n\nPertanyaan: {prompt}"
        
        # Buat model chat
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Buat instance chat
        chat = model.start_chat(history=chat_history)
        
        # Kirim pesan dan dapatkan respons
        response = chat.send_message(full_prompt)
        
        # Simpan interaksi terbaru ke history
        if len(chat.history) >= 2:
            user_message = {"role": "user", "parts": [prompt]}
            model_message = {"role": "model", "parts": [response.text]}
            chat_history.append(user_message)
            chat_history.append(model_message)
            save_chat_history()
        
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gagal generate response: {str(e)}")
        return f"Maaf, terjadi kesalahan sistem: {str(e)}"
