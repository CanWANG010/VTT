import streamlit as st
from RealtimeSTT import AudioToTextRecorder
from transformers import MarianMTModel, MarianTokenizer
import torch

def process_text(text):
    print(text)
    print(translate_to_chinese(text) + '\n')

# Initialize MarianMT model and tokenizer for offline translation
model_name = 'Helsinki-NLP/opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Initialize MarianMT model and tokenizer again for offline chinese to english translation
model_name_chinese = 'Helsinki-NLP/opus-mt-zh-en'
tokenizer_chinese = MarianTokenizer.from_pretrained(model_name_chinese)
translation_model_chinese = MarianMTModel.from_pretrained(model_name_chinese)

# Function to translate English text to Chinese using the MarianMT model
def translate_to_chinese(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Function to translate Chinese text to English using the MarianMT model
def translate_to_english(text):
    inputs = tokenizer_chinese([text], return_tensors="pt", padding=True)
    translated = translation_model_chinese.generate(**inputs)
    translated_text = tokenizer_chinese.decode(translated[0], skip_special_tokens=True)
    return translated

# def start_recording():
#     """Start the recorder and set up callbacks."""
#     st.session_state.recorder = AudioToTextRecorder()  # Initialize the recorder
#     st.session_state.recorder.start()  # Start recording
#     st.session_state.recorder.on_realtime_transcription_update = update_transcription
#     st.session_state.is_recording = True
#     st.write("Recording started. Speak now!")

# def stop_recording():
#     """Stop the recorder."""
#     if st.session_state.recorder:
#         st.session_state.recorder.stop()  # Stop recording
#         st.session_state.recorder = None
#     st.session_state.is_recording = False
#     st.write("Recording stopped.")

# def update_transcription(text):
#     """Update the transcription result in Streamlit."""
#     st.session_state.recognized_text = text
#     st.session_state.translated_text = translate_to_chinese(text)

# Streamlit UI
st.title("Speech to Text Translation")

if "context" not in st.session_state:
    st.session_state["context"] = ""

def update_context(new_text):
    st.session_state["context"] += f"{new_text}\n"

placeholder = st.empty()
with placeholder:
    for i in range(10):  # 模拟动态添加内容
        # time.sleep(1)  # 每隔1秒更新一次
        update_context(f"新上下文 {i+1}")
        # 更新 text_area 的内容
        text_area = st.text_area("动态上下文", value=st.session_state["context"], height=300)

# col1, col2 = st.columns(2)

# # Column 1: Display recognized English text
# with col1:
#     # st.text("Recognized English:")
#     st.text_area("Recognized English:", value=st.session_state.recognized_text, height=100, key="recognized_text_area")

# # Column 2: Display translated Chinese text
# with col2:
#     # st.text("Translated Chinese:")
#     st.text_area("Translated Chinese:", value=st.session_state.get("translated_text", ""), height=100, key="translated_text_area")

# # Start/Stop Recording Button
# if st.session_state.is_recording:
#     if st.button("Stop Recording"):
#         stop_recording()
# else:
#     if st.button("Start Recording"):
#         start_recording()
