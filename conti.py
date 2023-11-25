import speech_recognition as sr

def continuous_streaming_recognition(language='en-US', show_all=False):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak now...")
        while True:
            try:
            
                # Adjust for ambient noise and listen to the audio stream
                recognizer.adjust_for_ambient_noise(source)
                audio_stream = recognizer.listen(source, timeout=None)
                print(type(audio_stream))
                # Perform continuous speech recognition
                recognized_data = recognizer.recognize_google(audio_stream, language=language, show_all=show_all)

                # Print the recognized speech
                if show_all:
                    print("Possible transcriptions:")
                    for alternative in recognized_data['alternative']:
                        print(f"{alternative['transcript']} (confidence: {alternative['confidence']})")
                else:
                    print("Transcription:", recognized_data)

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    # Set the language code (e.g., 'en-US' for English, 'es-ES' for Spanish)
    language_code = 'en-US'

    # Run continuous streaming recognition
    continuous_streaming_recognition(language=language_code)
