import queue
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS
import os

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5           # change if you want longer recording
WAV_PATH = "input_en.wav"
OUT_WAV = "output_es.wav"

def record_wav(seconds=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"🎙️ Speak now ({seconds}s)...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=CHANNELS, dtype='float32')
    sd.wait()
    pcm16 = (audio * 32767).astype(np.int16)
    write(WAV_PATH, sr, pcm16)
    print("✅ Recorded.")
    return WAV_PATH

def load_asr(model_size="small"):
    # Use int8 for speed on CPU; set compute_type="float16" for GPU
    return WhisperModel(model_size, compute_type="int8", device="cpu")

def transcribe(whisper, wav_path):
    print("📝 Transcribing (EN)...")
    segments, info = whisper.transcribe(wav_path, language="en", beam_size=5, vad_filter=True)
    text = "".join([seg.text for seg in segments]).strip()
    print(f"EN: {text}")
    return text

def load_mt():
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl

def translate_en2es(tokenizer, model, text):
    print("🌐 Translating → ES...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=256)
    es = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"ES: {es}")
    return es

def load_tts():
    # Spanish VITS voice; first run downloads the model
    return TTS(model_name="tts_models/es/css10/vits")

def speak_es(tts, text, out_path=OUT_WAV):
    print("🔊 Synthesizing Spanish audio...")
    tts.tts_to_file(text=text, file_path=out_path)
    print(f"📁 Saved: {out_path}")
    # Play it back
    import soundfile as sf
    data, sr = sf.read(out_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()

def main():
    wav = record_wav()
    whisper = load_asr("small")   # try "base" for smaller/faster; "medium"/"large-v3" for best accuracy
    en_text = transcribe(whisper, wav)

    tok, mt = load_mt()
    es_text = translate_en2es(tok, mt, en_text)

    tts = load_tts()
    speak_es(tts, es_text)

if __name__ == "__main__":
    main()