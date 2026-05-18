"""Experimental Wav2Vec streaming ASR prototype."""

from __future__ import annotations

import argparse
import io


def load_wav2vec2_model(model_path: str, device: str):
    from fairseq.models.wav2vec import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model


def audio_data_to_tensor(audio_data):
    import torchaudio

    wav_buffer = io.BytesIO(audio_data.get_wav_data())
    waveform, _sample_rate = torchaudio.load(wav_buffer, normalize=True)
    return waveform.unsqueeze(0)


def transcribe_audio(model, audio_data, device: str) -> str:
    import torch

    with torch.no_grad():
        inputs = model.feature_extractor(audio_data.squeeze().to(device))
        z = model.feature_aggregator(inputs)
        logits = model.fc(z)
        return model.decode(logits.argmax(dim=-1))


def continuous_streaming_asr(model, language: str = "en-US") -> None:
    import speech_recognition as sr
    import torch

    recognizer = sr.Recognizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with sr.Microphone() as source:
        print("Speak now. Press Ctrl+C to stop.")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                audio_stream = recognizer.listen(source, timeout=None)
                audio_data = audio_data_to_tensor(audio_stream)
                transcription = transcribe_audio(model, audio_data, device)
                print("Transcription:", transcription)
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
            except sr.RequestError as error:
                print(f"Could not request results from Google Speech Recognition service: {error}")
            except KeyboardInterrupt:
                print("\nRecognition stopped by the user.")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experimental Wav2Vec streaming ASR prototype."
    )
    parser.add_argument(
        "--model-path",
        default="facebook/wav2vec2-base-960h",
        help="Fairseq Wav2Vec model path or identifier. Default: %(default)s",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Language hint retained from the original script. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_wav2vec2_model(args.model_path, device=device)
    continuous_streaming_asr(model, language=args.language)


if __name__ == "__main__":
    main()
