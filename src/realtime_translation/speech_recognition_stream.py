"""Continuously transcribe microphone input with Google Speech Recognition."""

from __future__ import annotations

import argparse


def continuous_streaming_recognition(language: str = "en-US", show_all: bool = False) -> None:
    import speech_recognition as sr

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak now. Press Ctrl+C to stop.")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                audio_stream = recognizer.listen(source, timeout=None)
                recognized_data = recognizer.recognize_google(
                    audio_stream,
                    language=language,
                    show_all=show_all,
                )

                if show_all:
                    print("Possible transcriptions:")
                    alternatives = recognized_data.get("alternative", [])
                    for alternative in alternatives:
                        transcript = alternative.get("transcript", "")
                        confidence = alternative.get("confidence", "unknown")
                        print(f"{transcript} (confidence: {confidence})")
                else:
                    print("Transcription:", recognized_data)

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
            except sr.RequestError as error:
                print(f"Could not request results from Google Speech Recognition service: {error}")
            except KeyboardInterrupt:
                print("\nRecognition stopped by the user.")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously transcribe microphone input."
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Google Speech Recognition language code. Default: %(default)s",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all recognition alternatives returned by Google.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    continuous_streaming_recognition(language=args.language, show_all=args.show_all)


if __name__ == "__main__":
    main()
