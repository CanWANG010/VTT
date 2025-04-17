import warnings
import whisper
import sounddevice as sd
import numpy as np
import wavio
from transformers import MarianMTModel, MarianTokenizer
import threading
import queue
import datetime

# Switch to show recording status
show = 0

if not show:
    warnings.filterwarnings("ignore")

# Initialize Whisper model (choose a smaller model like "base" or "small" for faster performance)
model = whisper.load_model("small")

# Initialize MarianMT model and tokenizer for offline translation
model_name = 'Helsinki-NLP/opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Parameters for audio recording
samplerate = 16000
energy_threshold = 1000  # Adjust this threshold based on your environment
silence_duration = 0.5  # Duration (in seconds) of silence to stop recording

# Queue to hold audio files to be processed
audio_queue = queue.Queue()

# Path for the combined script file, name file with current_time saved when start listening
combined_script_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

# Variable to store the last processed English text
last_processed_text = None

# Function to record audio until silence is detected
def record_audio_until_silence(filename, samplerate=16000):
    if show:
        print("Recording...")
    audio_data = []
    silence_counter = 0
    
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            frame = stream.read(int(samplerate * 0.1))[0]  # Read 0.1-second chunks
            audio_data.append(frame)
            
            # Calculate the energy of the current frame
            if np.max(np.abs(frame)) < energy_threshold:
                silence_counter += 0.1
            else:
                silence_counter = 0

            # If silence has been detected for a sufficient duration, stop recording
            if silence_counter >= silence_duration:
                break
    
    audio_data = np.concatenate(audio_data, axis=0)
    wavio.write(filename, audio_data, samplerate, sampwidth=2)
    if show:
        print("Recording complete!")
    audio_queue.put(filename)  # Add the recorded audio to the processing queue

# Function to translate English text to Chinese using the MarianMT model
def translate_to_chinese(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Function to process audio (transcribe and translate)
def process_audio():
    global last_processed_text
    while True:
        filename = audio_queue.get()  # Get the next audio file from the queue
        if filename:
            # Transcribe using Whisper
            result = model.transcribe(filename, fp16=False)
            english_text = result["text"].strip()
            
            # Only process if the new text is different from the last processed text
            if english_text and english_text != last_processed_text:
                print("Recognized English:", english_text)
                last_processed_text = english_text  # Update the last processed text

                # Translate to Chinese
                chinese_text = translate_to_chinese(english_text)
                print("Translated Chinese:", chinese_text)

                # Write both English and Chinese to the combined script file
                with open(combined_script_path, "a") as combined_file:
                    combined_file.write("English: " + english_text + "\n")
                    combined_file.write("Chinese: " + chinese_text + "\n")
                    combined_file.write("\n")  # Add a blank line for separation
        
        audio_queue.task_done()  # Mark the task as done

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Recording started at", current_time)
# Start a thread for continuous audio processing
threading.Thread(target=process_audio, daemon=True).start()

# Continuously record audio until stopped
while True:
    # Record audio and save to a temporary file
    record_audio_until_silence("temp_audio.wav")
