import os
import uuid
import tempfile
import subprocess
import logging
import re
from g2p_id import G2P

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder utilitas TTS
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

# Jalur path ke file model TTS
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# Jalur path ke file konfigurasi
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# Nama speaker yang digunakan (default: wibowo)
COQUI_SPEAKER = "wibowo"

# Pengecekan keberadaan file model dan konfigurasi
for path in [COQUI_MODEL_PATH, COQUI_CONFIG_PATH]:
    if not os.path.exists(path):
        logger.error(f"File tidak ditemukan: {path}")
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

def clean_text(text: str, max_length=100) -> str:
    """
    Bersihkan teks input untuk memastikan hanya teks dalam Bahasa Indonesia yang diproses.
    Batasi panjang teks untuk mencegah error di TTS.
    Args:
        text (str): Teks input.
        max_length (int): Panjang maksimum teks.
    Returns:
        str: Teks yang telah dibersihkan.
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        
        sentences = re.split(r'[.!?]', text)
        indonesian_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if re.search(r'\b(apa|siapa|berapa|adalah|di|ke|dari|dan|atau)\b', sentence, re.IGNORECASE) or \
               not re.search(r'\b(is|the|in|English|answer)\b', sentence, re.IGNORECASE):
                indonesian_phrases.append(sentence)
        
        cleaned_text = ' '.join(indonesian_phrases).strip()
        
        if not cleaned_text and "Jawaban:" in text:
            cleaned_text = text.split("Jawaban:")[-1].strip()
        
        cleaned_text = re.sub(r'[^\w\s.,!?]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length].rsplit(' ', 1)[0] + '.'
        
        if not cleaned_text:
            cleaned_text = "Maaf, teks input tidak dapat diproses."
        
        logger.info(f"Teks setelah pembersihan: {cleaned_text}")
        return cleaned_text
    
    except Exception as e:
        logger.error(f"Gagal membersihkan teks: {str(e)}")
        return "Maaf, teks input tidak dapat diproses."

def text_to_phonemes(text: str) -> str:
    """
    Mengonversi teks ke fonem menggunakan g2p-id.
    Args:
        text (str): Teks input dalam Bahasa Indonesia.
    Returns:
        str: Teks fonemik untuk input TTS.
    """
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            raise ValueError("Teks input kosong setelah pembersihan")

        logger.info(f"Memproses teks ke fonem dengan g2p-id: {cleaned_text}")
        g2p = G2P()
        phonemes = g2p(cleaned_text)
        phoneme_text = " ".join(phonemes)
        if not phoneme_text.strip():
            raise ValueError("Fonem kosong dihasilkan untuk teks input")
        logger.info(f"Fonem dihasilkan: {phoneme_text}")
        return phoneme_text
    
    except Exception as e:
        if "ONNXRuntimeError" in str(e):
            logger.error(f"Error ONNX di g2p-id: {str(e)}")
            logger.warning("Pastikan versi g2p_id dan onnxruntime kompatibel")
        else:
            logger.error(f"Gagal mengonversi teks ke fonem: {str(e)}")
        logger.warning("Menggunakan teks mentah sebagai fallback")
        return cleaned_text

def transcribe_text_to_speech(text: str, speaker: str = COQUI_SPEAKER) -> str:
    """
    Fungsi untuk mengonversi teks menjadi suara menggunakan TTS engine yang ditentukan.
    Args:
        text (str): Teks yang akan diubah menjadi suara.
        speaker (str): ID speaker yang digunakan.
    Returns:
        str: Path ke file audio hasil konversi.
    """
    try:
        phoneme_text = text_to_phonemes(text)
        path = _tts_with_coqui(phoneme_text, speaker)
        return path
    except Exception as e:
        logger.error(f"Gagal menghasilkan audio: {str(e)}")
        return f"[ERROR] Gagal menghasilkan audio: {str(e)}"

def _tts_with_coqui(text: str, speaker: str) -> str:
    """
    Mengonversi teks fonemik ke suara menggunakan model Indonesian-TTS.
    Args:
        text (str): Teks fonemik untuk sintesis suara.
        speaker (str): ID speaker yang digunakan.
    Returns:
        str: Path ke file audio hasil sintesis.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, f"tts_{uuid.uuid4()}.wav")

        cmd = [
            "tts",
            "--text", text,
            "--model_path", COQUI_MODEL_PATH,
            "--config_path", COQUI_CONFIG_PATH,
            "--speaker_idx", speaker,
            "--out_path", output_path
        ]
        
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            logger.info(f"Menjalankan perintah TTS: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=COQUI_DIR,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            if not os.path.exists(output_path):
                raise RuntimeError("Gagal menghasilkan file audio TTS")
            logger.info(f"Audio dihasilkan di: {output_path}")
            persistent_path = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
            os.rename(output_path, persistent_path)
            return persistent_path
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] TTS subprocess gagal: {e.stderr}")
            raise RuntimeError(f"TTS subprocess gagal: {e.stderr}")
        except Exception as e:
            logger.error(f"[ERROR] Gagal memproses TTS: {str(e)}")
            raise RuntimeError(f"Gagal memproses TTS: {str(e)}")

def list_speaker_idxs() -> list:
    """
    Mengambil daftar speaker yang tersedia.
    Returns:
        list: Daftar nama atau ID speaker.
    """
    cmd = [
        "tts",
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--list_speaker_idxs"
    ]
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            cwd=COQUI_DIR,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        speakers = []
        for line in result.stdout.split("\n"):
            if line.startswith("{'") and line.endswith("}"):
                speaker_dict = eval(line.strip())
                speakers.extend(speaker_dict.keys())
        logger.info(f"Daftar speaker: {speakers}")
        return speakers
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Gagal mengambil daftar speaker: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"[ERROR] Gagal memproses daftar speaker: {str(e)}")
        return []

def test_speakers(text: str, speakers: list) -> dict:
    """
    Menguji beberapa speaker untuk membandingkan kualitas audio.
    Args:
        text (str): Teks untuk diuji.
        speakers (list): Daftar speaker yang akan diuji.
    Returns:
        dict: Mapping speaker ke path file audio.
    """
    results = {}
    for speaker in speakers:
        try:
            audio_path = transcribe_text_to_speech(text, speaker)
            results[speaker] = audio_path
            logger.info(f"[TEST] Audio untuk speaker '{speaker}' dihasilkan di: {audio_path}")
        except Exception as e:
            logger.error(f"[TEST] Gagal menguji speaker '{speaker}': {str(e)}")
            results[speaker] = f"[ERROR] {str(e)}"
    return results

if __name__ == "__main__":
    try:
        speakers = list_speaker_idxs()
        logger.info(f"Daftar speaker tersedia: {speakers}")
        if COQUI_SPEAKER not in speakers:
            logger.warning(f"Speaker '{COQUI_SPEAKER}' tidak ditemukan dalam daftar speaker.")
        
        sample_text = "Halo, ini adalah uji coba sintesis suara dalam Bahasa Indonesia."
        test_speakers_list = [COQUI_SPEAKER, "ardi", "gadis"]
        test_speakers_list = [s for s in test_speakers_list if s in speakers]
        logger.info(f"Menguji speaker: {test_speakers_list}")
        test_results = test_speakers(sample_text, test_speakers_list)
        
        for speaker, audio_path in test_results.items():
            logger.info(f"Hasil untuk speaker '{speaker}': {audio_path}")
    except Exception as e:
        logger.error(f"Gagal menjalankan uji coba: {str(e)}")