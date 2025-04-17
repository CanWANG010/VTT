# import streamlit as st
# import time

# # 初始化 session_state 中的文本内容
# if "context" not in st.session_state:
#     st.session_state["context"] = ""

# # 定义一个函数来更新文本内容
# def update_context(new_text):
#     st.session_state["context"] += f"{new_text}\n"

# st.title("动态更新 Text Area 内容")
# st.write("内容将每隔 1 秒更新一次：")

# # 显示 text_area，内容绑定到 session_state 的 context
# # text_area = st.text_area("动态上下文", value=st.session_state["context"], height=300)

# # 持续更新内容
# placeholder = st.empty()
# with placeholder:
#     for i in range(10):  # 模拟动态添加内容
#         time.sleep(1)  # 每隔1秒更新一次
#         update_context(f"新上下文 {i+1}")
#         # 更新 text_area 的内容
#         text_area = st.text_area("动态上下文", value=st.session_state["context"], height=300)

# import streamlit as st
# from RealtimeSTT import AudioToTextRecorder

# # Define a function to process transcribed text
# def process_text(text):
#     print(text)

# # Initialize the recorder globally
# recorder = None

# # Streamlit app
# def main():
#     global recorder
#     st.title("Audio Recorder with Start/Stop Control")

#     # Session state to track the recording state
#     if "is_recording" not in st.session_state:
#         st.session_state["is_recording"] = False  # Default: not recording

#     # Button to start or stop recording
#     if st.session_state["is_recording"] == False:
#         if st.button("Start Recording"):
#             st.session_state["is_recording"] = True  # Update state
#             st.write("Recording started. Speak now...")
#             recorder = AudioToTextRecorder()
#             recorder.text(process_text)  # Start recording and transcribing
#     else:
#         if st.button("Stop Recording"):
#             st.session_state["is_recording"] = False  # Update state
#             st.write("Recording stopped.")
#             if recorder:
#                 recorder.stop()  # Stop recording
#                 recorder = None

#     st.write("Press the buttons above to control the recording.")

# # Ensure multiprocessing-safe execution
# if __name__ == '__main__':
#     main()


# import streamlit as st
# from RealtimeSTT import AudioToTextRecorder

# # Process transcription text
# def process_text(text):
#     try:
#         print(text)
#     except Exception as e:
#         print(f"Error in process_text: {e}")

# # Initialize the recorder
# recorder = None

# def main():
#     global recorder
#     st.title("Audio Recorder with Start/Stop Control")

#     # Initialize session state
#     if "is_recording" not in st.session_state:
#         st.session_state["is_recording"] = False

#     # Start/Stop recording
#     if not st.session_state["is_recording"]:
#         if st.button("Start Recording"):
#             st.session_state["is_recording"] = True
#             st.write("Recording started.")
#             try:
#                 recorder = AudioToTextRecorder()
#                 recorder.text(process_text)
#             except KeyboardInterrupt:
#                 st.write("Keyboard interrupt detected. Exiting...")
#                 st.session_state["is_recording"] = False
#                 # try:
#                 #     if recorder:
#                 #         recorder.stop()
#                 #         recorder = None
#             # except Exception as e:
#                 # st.write(f"Failed to start recorder: {e}")
#                 # st.session_state["is_recording"] = False
#     else:
#         if st.button("Stop Recording"):
#             st.session_state["is_recording"] = False
#             st.write("Recording stopped.")
#             try:
#                 if recorder:
#                     recorder.stop()
#                     recorder = None
#             except Exception as e:
#                 st.write(f"Failed to stop recorder: {e}")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# from RealtimeSTT import AudioToTextRecorder
# import threading
# import time

# # Function to process transcription text
# def process_text(text):
#     st.session_state["transcription"] += text + "\n"
#     print(text)

# # Background thread to manage the recorder
# def run_recorder():
#     try:
#         recorder = st.session_state["recorder"]
#         recorder.text(process_text)
#     except Exception as e:
#         st.session_state["is_recording"] = False
#         print(f"Error during recording: {e}")

# # Streamlit UI
# def main():
#     st.title("Audio Recorder with Translation")

#     # Initialize session state variables
#     if "is_recording" not in st.session_state:
#         st.session_state["is_recording"] = False
#     if "recorder" not in st.session_state:
#         st.session_state["recorder"] = None
#     if "transcription" not in st.session_state:
#         st.session_state["transcription"] = ""

#     # Start recording
#     if not st.session_state["is_recording"]:
#         if st.button("Start Recording"):
#             st.session_state["is_recording"] = True
#             st.session_state["recorder"] = AudioToTextRecorder()

#             # Start the recorder in a background thread
#             threading.Thread(target=run_recorder, daemon=True).start()
#             st.write("Recording started. Speak now...")

#     # Stop recording
#     else:
#         if st.button("Stop Recording"):
#             st.session_state["is_recording"] = False
#             if st.session_state["recorder"]:
#                 st.session_state["recorder"].stop()
#                 st.session_state["recorder"] = None
#             st.write("Recording stopped.")

#     # Display the transcription
#     st.text_area("Transcription", value=st.session_state["transcription"], height=300)

# if __name__ == "__main__":
#     main()

import streamlit as st
from RealtimeSTT import AudioToTextRecorder
from transformers import MarianMTModel, MarianTokenizer
import threading
import time

# Initialize translation model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Function to translate English text to Chinese
def translate_to_chinese(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Global variables for recorder
recorder = None
is_recording = False

# Initialize session state for transcriptions
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = []

# Function to process transcription text
def process_text(text):
    # Update transcriptions in session state
    st.session_state["transcriptions"].append(f"English: {text}")
    st.session_state["transcriptions"].append(f"Chinese: {translate_to_chinese(text)}")
    # Debug: Print to console
    print(f"English: {text}")
    print(f"Chinese: {translate_to_chinese(text)}")

# Function to run transcription in a thread
def transcription_thread():
    global recorder, is_recording
    try:
        while is_recording:
            recorder.text(process_text)
            time.sleep(0.1)  # Avoid busy looping
    except Exception as e:
        print(f"Error during transcription: {e}")
        is_recording = False

# Streamlit app
def main():
    global recorder, is_recording

    st.title("Real-Time Speech Transcription and Translation")

    # Create a placeholder for the transcription text area
    transcription_placeholder = st.empty()

    # Start/Stop Recording Buttons
    if not is_recording:
        if st.button("Start Recording"):
            # Clear old transcriptions
            st.session_state["transcriptions"] = []
            st.write("Recording started. Speak now...")
            is_recording = True
            recorder = AudioToTextRecorder()  # Initialize the recorder
            threading.Thread(target=transcription_thread, daemon=True).start()
    else:
        if st.button("Stop Recording"):
            st.write("Recording stopped.")
            is_recording = False
            if recorder:
                recorder.stop()
                recorder = None

    # Dynamically update the transcription display
    transcription_placeholder.text_area(
        "Transcriptions",
        value="\n".join(st.session_state["transcriptions"]),
        height=300,
        key="unique_transcription_area"  # Unique key to avoid duplicate IDs
    )

    # Debug: Print transcriptions in the console every second
    if st.button("Print Transcriptions to Console"):
        print("\n".join(st.session_state["transcriptions"]))

if __name__ == "__main__":
    main()

