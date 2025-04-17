import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import wavio
from transformers import MarianMTModel, MarianTokenizer
import threading
import queue
import time
import os

st.set_page_config(layout="wide") # MUST be the first Streamlit command

# --- Configuration ---
MODEL_SIZE = "base"  # Whisper model size (tiny, base, small, medium, large)
LANGUAGE = "en" # Source language for Whisper
TARGET_LANGUAGE = "zh" # Target language for translation
MODEL_NAME = f'Helsinki-NLP/opus-mt-{LANGUAGE}-{TARGET_LANGUAGE}'
SAMPLERATE = 16000
CHUNK_DURATION_S = 1.0 # Duration of audio chunks to record
SILENCE_THRESHOLD = 500 # Energy threshold to detect silence/non-speech
MIN_CHUNK_ENERGY = 100 # Minimum energy for a chunk to be considered potentially speech
TEMP_AUDIO_FILENAME = "temp_audio_chunk.wav"

# --- Model Loading (Cached) ---
@st.cache_resource
def load_whisper_model(size):
    print(f"Loading Whisper model: {size}")
    return whisper.load_model(size)

@st.cache_resource
def load_translation_model(model_name):
    print(f"Loading translation model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper_model(MODEL_SIZE)
translation_tokenizer, translation_model = load_translation_model(MODEL_NAME)

# --- State Management ---
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = [] # List to store {"en": "...", "zh": "..."} dicts
if 'last_processed_english' not in st.session_state:
    st.session_state.last_processed_english = ""

# --- Core Functions ---

def translate_to_chinese(text):
    """Translates English text to Chinese."""
    try:
        inputs = translation_tokenizer([text], return_tensors="pt", padding=True)
        translated = translation_model.generate(**inputs)
        translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return "[Translation Error]"

def process_audio_chunks():
    """Processes audio chunks from the queue: transcribes and translates."""
    print("Processing thread started.")
    accumulated_audio = np.array([], dtype=np.int16)

    while st.session_state.is_recording or not st.session_state.audio_queue.empty():
        try:
            chunk = st.session_state.audio_queue.get(timeout=1.0) # Wait up to 1 sec
            accumulated_audio = np.concatenate((accumulated_audio, chunk))
            st.session_state.audio_queue.task_done()

            # Basic silence detection heuristic: process if queue is empty or getting large
            # A more sophisticated VAD would be better here.
            process_now = st.session_state.audio_queue.empty() or len(accumulated_audio) > SAMPLERATE * 5 # Process every 5 seconds approx

            if process_now and len(accumulated_audio) > SAMPLERATE * 0.5: # Need at least 0.5s
                 # Save accumulated audio to a temporary file for Whisper
                wavio.write(TEMP_AUDIO_FILENAME, accumulated_audio, SAMPLERATE, sampwidth=2)
                accumulated_audio = np.array([], dtype=np.int16) # Reset accumulator

                try:
                    # Transcribe
                    result = whisper_model.transcribe(TEMP_AUDIO_FILENAME, fp16=False, language=LANGUAGE) # Assuming no GPU for fp16
                    english_text = result["text"].strip()

                    if english_text and english_text.lower() != st.session_state.last_processed_english.lower():
                         # Translate
                        chinese_text = translate_to_chinese(english_text)

                        # Update state (important: use Streamlit's way to update state for UI refresh)
                        st.session_state.last_processed_english = english_text
                        st.session_state.transcript.append({"en": english_text, "zh": chinese_text})
                        # Note: Direct UI updates from thread need st.experimental_rerun or similar,
                        # but here we just update state. The main thread will handle UI redraw.

                        print(f"EN: {english_text} | ZH: {chinese_text}") # Log progress
                    else:
                        print(f"Skipping duplicate/empty transcription: '{english_text}'")


                except Exception as e:
                    st.error(f"Transcription/Processing error: {e}")
                finally:
                    if os.path.exists(TEMP_AUDIO_FILENAME):
                        os.remove(TEMP_AUDIO_FILENAME) # Clean up temp file

        except queue.Empty:
            # If the queue is empty and we are no longer recording, exit
            if not st.session_state.is_recording:
                print("Processing thread finished: Queue empty and not recording.")
                break
            else:
                # Still recording, just means no new chunk within the timeout
                continue
        except Exception as e:
            st.error(f"Error in processing loop: {e}")
            time.sleep(1) # Avoid tight loop on error


    print("Processing thread stopped.")
    # Final cleanup
    if os.path.exists(TEMP_AUDIO_FILENAME):
        try:
            os.remove(TEMP_AUDIO_FILENAME)
        except OSError:
            pass # Ignore if file already removed


def record_audio():
    """Continuously records audio chunks and puts them into the queue."""
    print("Recording thread started.")
    chunk_size = int(SAMPLERATE * CHUNK_DURATION_S)

    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16', blocksize=chunk_size) as stream:
            while st.session_state.is_recording:
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Warning: Audio buffer overflowed!")

                # Simple energy check - avoid queueing complete silence
                energy = np.max(np.abs(audio_chunk))
                if energy > MIN_CHUNK_ENERGY:
                    st.session_state.audio_queue.put(audio_chunk.flatten())
                # No need for sleep, stream.read blocks

    except Exception as e:
        st.error(f"Recording error: {e}")
    finally:
        print("Recording thread stopped.")
        # Signal processing thread that recording has finished by putting None or checking state
        # (Processing thread already checks st.session_state.is_recording)

# --- UI Layout ---
st.title("🎙️ Real-time Voice Transcription & Translation")
st.markdown(f"**Whisper Model:** `{MODEL_SIZE}` | **Translate:** `{LANGUAGE}` to `{TARGET_LANGUAGE}`")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Controls")
    start_button = st.button("Start Recording", key="start_rec", disabled=st.session_state.is_recording)
    stop_button = st.button("Stop Recording", key="stop_rec", disabled=not st.session_state.is_recording)

    if start_button:
        st.session_state.is_recording = True
        st.session_state.transcript = [] # Clear previous transcript
        st.session_state.last_processed_english = ""

        # Clear queue
        while not st.session_state.audio_queue.empty():
            try: st.session_state.audio_queue.get_nowait()
            except queue.Empty: break

        # Start threads
        st.session_state.recording_thread = threading.Thread(target=record_audio, daemon=True)
        st.session_state.processing_thread = threading.Thread(target=process_audio_chunks, daemon=True)
        st.session_state.recording_thread.start()
        st.session_state.processing_thread.start()
        st.rerun() # Rerun to update button states

    if stop_button:
        st.session_state.is_recording = False
        # Wait briefly for threads to notice the state change (optional, depends on thread logic)
        # Threads should ideally check the flag frequently
        # Consider using thread.join() if you need to ensure they fully finish before proceeding
        st.info("Stopping recording... Processing remaining audio might take a moment.")
        # Give a moment for UI to update and threads to potentially finish gracefully
        # A more robust solution might involve checking thread.is_alive() and waiting
        time.sleep(1.5) # Simple wait
        st.rerun() # Rerun to update button states

    st.subheader("Status")
    status_indicator = "🔴 Recording" if st.session_state.is_recording else "⚪ Idle"
    st.markdown(f"**Status:** {status_indicator}")
    qsize = st.session_state.audio_queue.qsize()
    st.markdown(f"**Audio Chunks Queued:** {qsize}")

with col2:
    st.subheader("Latest Output")
    latest_en_placeholder = st.empty()
    latest_zh_placeholder = st.empty()

st.subheader("Full Transcript")
transcript_placeholder = st.empty()

# --- Display Updates (runs on main thread) ---
if st.session_state.transcript:
    latest_entry = st.session_state.transcript[-1]
    latest_en_placeholder.text_area("Latest English", latest_entry["en"], height=100, key="latest_en")
    latest_zh_placeholder.text_area("Latest Chinese", latest_entry["zh"], height=100, key="latest_zh")

    full_transcript_md = ""
    for i, entry in enumerate(st.session_state.transcript):
        full_transcript_md += f"**Segment {i+1}:**\n"
        full_transcript_md += f"EN: {entry['en']}\n"
        full_transcript_md += f"ZH: {entry['zh']}\n\n"
    transcript_placeholder.markdown(full_transcript_md)

else:
    latest_en_placeholder.text_area("Latest English", "[Waiting for speech...]", height=100, key="latest_en_empty")
    latest_zh_placeholder.text_area("Latest Chinese", "[Waiting for speech...]", height=100, key="latest_zh_empty")
    transcript_placeholder.markdown("_Transcript will appear here._")


# Add a small delay and rerun to refresh UI periodically when recording
if st.session_state.is_recording:
    time.sleep(0.5) # Adjust refresh rate as needed
    st.rerun() 