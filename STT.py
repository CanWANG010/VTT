from RealtimeSTT import AudioToTextRecorder
import datetime
from transformers import MarianMTModel, MarianTokenizer


def process_text(text):
    print('\n'+text+'\n'+translate_to_chinese(text)+'\n')

    with open(combined_script_path, "a") as combined_file:
        combined_file.write("English: " + text + "\n")
        combined_file.write("Chinese: " + translate_to_chinese(text) + "\n\n")
    # print(translate_to_chinese(text) + '\n')

def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Initialize MarianMT model and tokenizer for offline translation
model_name = 'Helsinki-NLP/opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Path for the combined script file, name file with current_time saved when start listening
combined_script_path = current_time() + ".txt"

# Function to translate English text to Chinese using the MarianMT model
def translate_to_chinese(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

if __name__ == '__main__':
    print("Wait until it says 'speak now', Press Control + C to exit")
    recorder = AudioToTextRecorder(language="en")
        
    # detect keyboard interrupt
    try:
        while True:
            # pass
            recorder.text(process_text)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        recorder.stop()