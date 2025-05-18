# UAS Praktikum Pemrosesan Bahasa Alami

Proyek ini adalah tugas akhir untuk Praktikum Pemrosesan Bahasa Alami. Aplikasi ini mengintegrasikan **Speech-to-Text (STT)**, **Large Language Model (LLM)**, dan **Text-to-Speech (TTS)** menggunakan teknologi **Whisper.cpp**, **Gemini API**, **Coqui TTS**, **FastAPI**, dan **Gradio**.
 
 ## Struktur Proyek
 - `app/stt.py`: Modul untuk transkripsi suara ke teks menggunakan Whisper.cpp.
 - `app/llm.py`: Modul untuk menghasilkan respons menggunakan Gemini API.
 - `app/tts.py`: Modul untuk sintesis teks ke suara menggunakan Coqui TTS.
 - `app/main.py`: Aplikasi utama dengan FastAPI dan Gradio.
 - `gradio_app/app.py`: Aplikasi utama antarmuka.
 - `requirements.txt`: Daftar dependensi Python.
 
 ## Cara Menjalankan
 1. Clone repository:
    ```bash
    git clone https://github.com/aufazaikra/2208107010070_MuhammadAufaZaikra_UAS_NLP.git
    cd 2208107010070_MuhammadAufaZaikra_UAS_NLP
    ```
 2. Instal dependensi:
    ```bash
    pip install -r requirements.txt
    ```
 3. Siapkan file `.env` dengan `GEMINI_API_KEY`.
 4. Jalankan aplikasi:
    ```bash
    uvicorn main:app --reload
    ```
 5. Akses antarmuka Gradio di browser.
 
 ## Teknologi yang Digunakan
 - **Whisper.cpp**: Untuk STT.
 - **Gemini API**: Untuk LLM.
 - **Coqui TTS**: Untuk TTS.
 - **FastAPI**: Backend API.
 - **Gradio**: Antarmuka pengguna.
 
