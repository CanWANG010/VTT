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
# SILENCE_THRESHOLD = 500 # (Not currently used in chunking model)
MIN_CHUNK_ENERGY = 100 # Minimum energy for a chunk to be considered potentially speech
TEMP_AUDIO_FILENAME = "temp_audio_chunk.wav"
PROCESS_INTERVAL_S = 3.0 # How often to process accumulated audio

# --- Model Loading (Cached) ---
@st.cache_resource
def load_whisper_model(size):
    print(f"Loading Whisper model: {size}")
    # Add device="cuda" if GPU is available and torch is installed with CUDA support
    return whisper.load_model(size)

@st.cache_resource
def load_translation_model(model_name):
    print(f"Loading translation model: {model_name}")
    # Add device=0 if GPU is available
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
if 'results_queue' not in st.session_state: # Queue for results from processor thread
    st.session_state.results_queue = queue.Queue()
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'stop_event' not in st.session_state: # Event to signal threads to stop
    st.session_state.stop_event = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = [] # List to store {"en": "...", "zh": "..."} dicts

# --- Core Functions (Run in Threads) ---

def translate_to_chinese(text):
    """Translates English text to Chinese. (Runs in processor thread)"""
    # This function itself doesn't interact with Streamlit directly
    try:
        inputs = translation_tokenizer([text], return_tensors="pt", padding=True)
        # If using GPU: inputs = {k: v.to(translation_model.device) for k, v in inputs.items()}
        translated = translation_model.generate(**inputs)
        translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}") # Log error
        # Indicate error back to main thread via queue
        return f"[Translation Error: {e}]"

def process_audio_chunks(stop_event, audio_q, result_q):
    """Processes audio chunks from the queue: transcribes and translates. (Runs in thread)"""
    print("Processing thread started.")
    accumulated_audio = np.array([], dtype=np.int16)
    last_processed_time = time.time()
    last_processed_english_in_thread = "" # Keep track within the thread to avoid duplicates

    while not stop_event.is_set():
        try:
            # Get audio chunk, non-blocking if possible or with short timeout
            chunk = audio_q.get(timeout=0.1)
            accumulated_audio = np.concatenate((accumulated_audio, chunk))
            audio_q.task_done()

            # Process if enough time has passed or if stopping and data exists
            should_process = (time.time() - last_processed_time > PROCESS_INTERVAL_S) or \
                             (stop_event.is_set() and not audio_q.empty())

            if should_process and len(accumulated_audio) > SAMPLERATE * 0.5: # Need some audio
                print(f"Processing accumulated audio of length: {len(accumulated_audio)}")
                audio_to_process = accumulated_audio
                accumulated_audio = np.array([], dtype=np.int16) # Reset accumulator
                last_processed_time = time.time()

                # Save accumulated audio to a temporary file for Whisper
                try:
                    wavio.write(TEMP_AUDIO_FILENAME, audio_to_process, SAMPLERATE, sampwidth=2)

                    # Transcribe
                    # Add fp16=True if using GPU
                    result = whisper_model.transcribe(TEMP_AUDIO_FILENAME, fp16=False, language=LANGUAGE)
                    english_text = result["text"].strip()

                    if english_text and english_text.lower() != last_processed_english_in_thread.lower():
                        last_processed_english_in_thread = english_text # Update last text for duplicate check
                        # Translate
                        chinese_text = translate_to_chinese(english_text)

                        # Put result onto the results queue for the main thread
                        result_q.put({"en": english_text, "zh": chinese_text})
                        print(f"THREAD: EN: {english_text} | ZH: {chinese_text}") # Log progress
                    else:
                        print(f"THREAD: Skipping duplicate/empty transcription: '{english_text}'")

                except Exception as e:
                    print(f"Transcription/Processing error in thread: {e}")
                    result_q.put({"error": f"Processing Error: {e}"}) # Send error to main thread
                finally:
                    if os.path.exists(TEMP_AUDIO_FILENAME):
                        try:
                            os.remove(TEMP_AUDIO_FILENAME) # Clean up temp file
                        except OSError as e:
                             print(f"Error removing temp file: {e}")

        except queue.Empty:
            # No audio chunk received in timeout, check if we need to stop
            if stop_event.is_set() and audio_q.empty():
                print("Processing thread: Stop event set and audio queue empty.")
                break
            # Otherwise, just loop again
            continue
        except Exception as e:
            print(f"Error in processing loop: {e}")
            result_q.put({"error": f"Loop Error: {e}"}) # Send error to main thread
            time.sleep(1) # Avoid tight loop on error

    # Final check for any remaining audio when stopping
    if len(accumulated_audio) > SAMPLERATE * 0.1:
         print(f"Processing remaining audio of length: {len(accumulated_audio)}")
         # (Add final processing logic here - similar to loop above)
         # ... (omitted for brevity, but should ideally process remaining audio)
         pass

    print("Processing thread stopped.")
    # Final cleanup
    if os.path.exists(TEMP_AUDIO_FILENAME):
        try:
            os.remove(TEMP_AUDIO_FILENAME)
        except OSError:
            pass # Ignore if file already removed

def record_audio(stop_event, audio_q):
    """Continuously records audio chunks and puts them into the queue. (Runs in thread)"""
    print("Recording thread started.")
    chunk_size = int(SAMPLERATE * CHUNK_DURATION_S)

    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16', blocksize=chunk_size) as stream:
            while not stop_event.is_set():
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Warning: Audio buffer overflowed!")

                # Simple energy check - avoid queueing complete silence
                energy = np.max(np.abs(audio_chunk))
                if energy > MIN_CHUNK_ENERGY:
                    audio_q.put(audio_chunk.flatten())
                # No need for sleep, stream.read blocks until chunk is available

    except Exception as e:
        print(f"Recording error: {e}")
        # Cannot call st.error here. Maybe put error on result_q?
        # For now, just print to console.
    finally:
        print("Recording thread stopped.")

# --- UI Layout --- (Runs on Main Thread)
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
        st.session_state.stop_event = threading.Event() # Create new stop event

        # Clear queues
        while not st.session_state.audio_queue.empty():
            try: st.session_state.audio_queue.get_nowait()
            except queue.Empty: break
        while not st.session_state.results_queue.empty():
            try: st.session_state.results_queue.get_nowait()
            except queue.Empty: break

        # Start threads, passing stop_event and queues
        st.session_state.recording_thread = threading.Thread(
            target=record_audio,
            args=(st.session_state.stop_event, st.session_state.audio_queue),
            daemon=True
        )
        st.session_state.processing_thread = threading.Thread(
            target=process_audio_chunks,
            args=(st.session_state.stop_event, st.session_state.audio_queue, st.session_state.results_queue),
            daemon=True
        )
        st.session_state.recording_thread.start()
        st.session_state.processing_thread.start()
        st.rerun() # Rerun to update button states and start periodic check

    if stop_button:
        if st.session_state.stop_event:
            st.session_state.stop_event.set() # Signal threads to stop
        st.session_state.is_recording = False
        st.info("Stopping recording... Allowing threads to finish.")
        # Don't rerun immediately, let the periodic check handle UI updates
        # We might want to wait for threads explicitly here if needed:
        # if st.session_state.processing_thread:
        #     st.session_state.processing_thread.join(timeout=5) # Wait max 5 seconds
        # if st.session_state.recording_thread:
        #     st.session_state.recording_thread.join(timeout=2)
        # st.rerun() # Now rerun after attempting to join

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
transcript_placeholder = st.container() # Use container for better updates

# --- Process Results and Update UI (Main Thread) ---
def update_display():
    new_results = False
    # Check the results queue for new transcriptions/translations
    while not st.session_state.results_queue.empty():
        try:
            result = st.session_state.results_queue.get_nowait()
            if "error" in result:
                st.error(result["error"]) # Display errors from threads
            else:
                st.session_state.transcript.append(result)
                new_results = True
            st.session_state.results_queue.task_done()
        except queue.Empty:
            break # Should not happen with check, but good practice
        except Exception as e:
            st.error(f"Error processing result queue: {e}")

    # Update UI elements if there are results or state changes
    if st.session_state.transcript:
        latest_entry = st.session_state.transcript[-1]
        latest_en_placeholder.text_area("Latest English", latest_entry["en"], height=100, key="latest_en")
        latest_zh_placeholder.text_area("Latest Chinese", latest_entry["zh"], height=100, key="latest_zh")

        with transcript_placeholder:
            st.markdown("--- Beginnning of Transcript ---")
            for i, entry in enumerate(reversed(st.session_state.transcript)): # Show newest first
                st.markdown(f"**Segment {len(st.session_state.transcript) - i}:**")
                st.text(f"EN: {entry['en']}")
                st.text(f"ZH: {entry['zh']}")
                st.markdown("---")
    else:
        latest_en_placeholder.text_area("Latest English", "[Waiting for speech...]", height=100, key="latest_en_empty")
        latest_zh_placeholder.text_area("Latest Chinese", "[Waiting for speech...]", height=100, key="latest_zh_empty")
        with transcript_placeholder:
            st.markdown("_Transcript will appear here._")

# Call the update function to process queue and refresh display
update_display()

# Keep app alive and checking queue while recording
if st.session_state.is_recording:
    time.sleep(0.3) # Refresh interval
    st.rerun() 