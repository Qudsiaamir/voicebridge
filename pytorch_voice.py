import torch
import torchaudio
from fairseq.models.wav2vec import Wav2Vec2Model
import speech_recognition as sr

def load_wav2vec2_model(model_path):
    model = Wav2Vec2Model.from_pretrained(model_path)
    model.eval()
    return model

def transcribe_audio(model, audio_data, device='cuda'):
    with torch.no_grad():
        inputs = model.feature_extractor(audio_data.squeeze().to(device))
        z = model.feature_aggregator(inputs)
        logits = model.fc(z)
        transcription = model.decode(logits.argmax(dim=-1))
        return transcription

def continuous_streaming_asr(model, language='en-US'):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak now...")

        try:
            while True:
                # Capture the audio stream continuously
                audio_stream = recognizer.listen(source, timeout=None)

                # Convert the audio stream to a PyTorch tensor
                waveform, sample_rate = torchaudio.load(audio_stream.frame_data.numpy(), normalize=True)
                audio_data = waveform.unsqueeze(0)

                # Perform continuous ASR
                transcription = transcribe_audio(model, audio_data)

                print("Transcription:", transcription)

        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        except KeyboardInterrupt:
            print("Recognition stopped by the user.")

if __name__ == "__main__":
    # Set the language code (e.g., 'en-US' for English, 'es-ES' for Spanish)
    language_code = 'en-US'

    # Load pre-trained Wav2Vec 2 model (adjust the path accordingly)
    model_path = 'facebook/wav2vec2-base-960h'
    model = load_wav2vec2_model(model_path)

    # Run continuous streaming ASR
    continuous_streaming_asr(model, language=language_code)
