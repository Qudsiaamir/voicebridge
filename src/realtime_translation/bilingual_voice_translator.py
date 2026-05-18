"""Two-way English/Spanish voice translation demo with text-to-speech."""

from __future__ import annotations

import argparse

LANGUAGE_DESTINATIONS = {
    "en": ("es", "Spanish"),
    "es": ("en", "English"),
}


def translate_phrase(text: str, translator) -> tuple[str, str] | None:
    from langdetect import detect

    input_language = detect(text)
    destination = LANGUAGE_DESTINATIONS.get(input_language)

    if destination is None:
        return None

    destination_code, destination_name = destination
    translation = translator.translate(text, dest=destination_code)
    return destination_name, translation.text


def run_voice_translator(speak_output: bool = True) -> None:
    import pyttsx3
    import speech_recognition as sr
    from googletrans import Translator
    from langdetect.lang_detect_exception import LangDetectException

    recognizer = sr.Recognizer()
    translator = Translator(service_urls=["translate.google.com"])
    tts = pyttsx3.init() if speak_output else None

    print("Speak English or Spanish. Press Ctrl+C to stop.")

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio)
            result = translate_phrase(text, translator)

            if result is None:
                print(f"Unsupported language for recognized text: {text}")
                continue

            destination_name, translated_text = result
            print(f"Translated to {destination_name}: {translated_text}")

            if tts is not None:
                tts.say(translated_text)
                tts.runAndWait()

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
        except sr.RequestError as error:
            print(f"Could not request results from Google Speech Recognition service: {error}")
        except LangDetectException:
            print("Could not detect the spoken language.")
        except KeyboardInterrupt:
            print("\nVoice translator stopped by the user.")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate spoken English to Spanish and Spanish to English."
    )
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Print translations without speaking them out loud.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_voice_translator(speak_output=not args.no_speech)


if __name__ == "__main__":
    main()
