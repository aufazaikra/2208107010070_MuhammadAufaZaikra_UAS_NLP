import os
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import komponen STT, LLM, dan TTS
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

app = FastAPI(title="Voice Chatbot API")

# Konfigurasi CORS untuk memungkinkan akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam produksi, ganti dengan domain frontend yang spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Voice Chatbot API is running"}

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Endpoint utama untuk voice chat.
    Menerima file audio, melakukan transkripsi, menghasilkan respons, dan mengonversi ke audio.
    """
    try:
        # Simpan file yang diupload ke direktori sementara
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Step 1: Speech-to-Text (STT)
        text_transcription = transcribe_speech_to_text(content, os.path.splitext(file.filename)[1])
        print(f"Transkripsi: {text_transcription}")
        
        # Step 2: Large Language Model (LLM)
        llm_response = generate_response(text_transcription)
        print(f"Respons LLM: {llm_response}")
        
        # Step 3: Text-to-Speech (TTS)
        tts_output_path = transcribe_text_to_speech(llm_response)
        print(f"Output TTS: {tts_output_path}")
        
        # Hapus file audio input yang sudah tidak diperlukan
        os.remove(temp_file_path)
        
        # Kembalikan file audio hasil sebagai respons
        return FileResponse(
            path=tts_output_path,
            media_type="audio/wav",
            filename="response.wav"
        )
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)