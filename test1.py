import speech_recognition as sr
import pyttsx3
from langdetect import detect
from googletrans import Translator

# Create a recognizer object, a translator object, and a Text-to-Speech object
r = sr.Recognizer()
translator = Translator(service_urls=['translate.google.com'])
tts = pyttsx3.init()

while True:
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Speak something...")
        # Adjust the ambient noise level
        r.adjust_for_ambient_noise(source)
        # Listen for audio input from the user
        audio = r.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        input_language = detect(text)

        # Translate speech to English if the detected language is Spanish
        if input_language == "es":
            translation = translator.translate(text, dest='en')
            print(f"Translated to English: {translation.text}")
            # Speak the translated text
            tts.say(translation.text)
            tts.runAndWait()

        # Translate speech to Spanish if the detected language is English
        elif input_language == "en":
            translation = translator.translate(text, dest='es')
            print(f"Translated to Spanish: {translation.text}")
            # Speak the translated text
            tts.say(translation.text)
            tts.runAndWait()

        else:
            print("Unsupported language")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")

    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
