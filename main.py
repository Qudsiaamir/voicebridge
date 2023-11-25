# # # from transformers import MarianMTModel, MarianTokenizer

# # # # Load pre-trained MarianMT model and tokenizer
# # # model_name = "Helsinki-NLP/opus-mt-en-ro"  # English to Romanian translation
# # # tokenizer = MarianTokenizer.from_pretrained(model_name)
# # # model = MarianMTModel.from_pretrained(model_name)

# # # # Tokenize and translate a sentence
# # # sentence = "Hello, how are you?"
# # # input_ids = tokenizer.encode(sentence, return_tensors="pt")

# # # # Generate translation using the model's generate method
# # # translated_ids = model.generate(input_ids)

# # # # Decode the output and print the translation
# # # translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
# # # print("Original sentence:", sentence)
# # # print("Translated sentence:", translated_sentence)

# # from transformers import MarianMTModel, MarianTokenizer

# # # Load pre-trained MarianMT model and tokenizer for English to Tamil translation
# # model_name = "Helsinki-NLP/opus-mt-en-ur"
# # tokenizer = MarianTokenizer.from_pretrained(model_name)
# # model = MarianMTModel.from_pretrained(model_name)

# # # Tokenize and translate a sentence
# # sentence = "Hello, how are you?"
# # input_ids = tokenizer.encode(sentence, return_tensors="pt")

# # # Generate translation using the model's generate method
# # translated_ids = model.generate(input_ids)

# # # Decode the output and print the translation
# # translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
# # print("Original sentence:", sentence)
# # print("Translated sentence (Tamil):", translated_sentence)


from transformers import MarianMTModel, MarianTokenizer
import torch
# Load pre-trained MarianMT model and tokenizer for English to Tamil translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
# model_name = "Helsinki-NLP/opus-mt-en-ur"

tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load the model on the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MarianMTModel.from_pretrained(model_name).to(device)

# Tokenize and translate a sentence
sentence = "Are you interested in a data plan for your simcard?"
input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)

# Generate translation using the model's generate method
translated_ids = model.generate(input_ids)

# Decode the output and print the translation
translated_sentence = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("Original sentence:", sentence)
with open('example', 'w', encoding='utf-8') as file:
    # Write content to the file
    file.write(translated_sentence)
print("Translated sentence (Urdu):", translated_sentence)

# import sounddevice as sd
# from transformers import pipeline

# def record_audio(file_path, duration=5, sample_rate=44100):
#     print(f"Recording {duration} seconds of audio. Speak now...")
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
#     sd.wait()
#     sd.write(file_path, audio_data)
#     print(f"Audio recording saved to {file_path}")

# def transcribe_audio(file_path):
#     asr_pipeline = pipeline("automatic-speech-recognition")
#     transcription = asr_pipeline(file_path)
#     return transcription[0]['sentence']

# if __name__ == "__main__":
#     audio_file_path = "recorded_audio.wav"
    
#     # Record audio
#     record_audio(audio_file_path)

#     # Transcribe audio
#     transcription = transcribe_audio(audio_file_path)
#     print("Transcription:", transcription)

# import sounddevice as sd

# # Display information about available devices
# print(sd.query_devices())
# import pyaudio
# import numpy as np

# def audio_listener(input_device_index, duration, fs):
#     p = pyaudio.PyAudio()

#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=fs,
#                     input=True,
#                     input_device_index=input_device_index,
#                     frames_per_buffer=int(fs * duration))

#     print("Listening...")
#     audio_data = np.frombuffer(stream.read(int(fs * duration)), dtype=np.int16)
#     print("Done listening.")

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     return audio_data

# def audio_speaker(output_device_index, data, fs):
#     p = pyaudio.PyAudio()

#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=fs,
#                     output=True,
#                     output_device_index=output_device_index)

#     print("Playing...")
#     stream.write(data.astype(np.int16).tobytes())
#     print("Done playing.")

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

# def main():
#     # Set the index of your input and output devices
#     input_device_index = 1  # Change this to the index of your microphone
#     output_device_index = 0  # Change this to the index of your speakers

#     # Set the duration and sampling rate
#     duration = 5  # seconds
#     fs = 44100  # Hz

#     # Listen to audio
#     audio_data = audio_listener(input_device_index, duration, fs)

#     # Optionally process the audio (e.g., using numpy operations)
#     # processed_audio = process_audio(audio_data)

#     # Play back the audio
#     audio_speaker(output_device_index, audio_data, fs)

# if __name__ == "__main__":
#     main()


# import sounddevice as sd
# from scipy.io.wavfile import write
# from transformers import pipeline
# def audio_listener(ind, duration, fs):
#     print("Listening...")
#     audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=ind, dtype='int16')
#     sd.wait()
#     # sd.write("recorded_audio.wav", audio_data)
#     print("Done listening.")
#     return audio_data

# def audio_speaker(ind,data, fs):
#     print("Playing...")
#     sd.play(data, samplerate=fs, device=ind)
#     sd.wait()
#     print("Done playing.")


# def transcribe_audio(file_path):
#     asr_pipeline = pipeline("automatic-speech-recognition")
#     transcription = asr_pipeline(file_path)
#     return transcription[0]['sentence']


# def main():
#     # Set the index of your input and output devices
#     input_device_index = 1  # Change this to the index of your microphone
#     output_device_index = 4  # Change this to the index of your speakers

#     # Set the duration and sampling rate
#     duration = 5  # seconds
#     fs = 44100  # Hz

#     # Listen to audio
#     audio_data = audio_listener(input_device_index, duration, fs)
    
#     # Play back the audio
#     audio_speaker(output_device_index,audio_data, fs)
#     write("recorded_audio.wav", audio_data, fs)
# if __name__ == "__main__":
#     main()
#     print('gsdf')
#     transcription = transcribe_audio("recorded_audio.wav")
#     print("Transcription:", transcription)

# import sounddevice as sd
# import numpy as np
# import speech_recognition as sr
# from transformers import pipeline

# def record_audio(sample_rate=44100, duration=2):
#     print("Recording...")
#     audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2)
#     sd.wait()  # Wait until recording is finished
#     return audio_data

# def transcribe_audio(audio_data, recognizer):
#     audio_text = recognizer.recognize_google(audio_data, show_all=False)
#     return audio_text

# def translate_text(text, source_lang='en', target_lang='fr'):
#     translator = pipeline('translation', model=f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')
#     translated_text = translator(text, max_length=50)[0]['translation_text']
#     return translated_text

# if __name__ == "__main__":
#     # Record audio
#     audio_data = record_audio()
#     print('ysdfhyk')
#     # Convert audio to text
#     recognizer = sr.Recognizer()
#     print('sdhfhd')
#     audio_text = transcribe_audio(audio_data.flatten(), recognizer)

#     print("Transcribed Text:", audio_text)

#     # Translate text
#     translated_text = translate_text(audio_text)
#     print("Translated Text:", translated_text)
