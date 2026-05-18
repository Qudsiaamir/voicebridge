# Realtime Translation

A Python prototype for text and speech translation using Hugging Face translation models, Google Speech Recognition, and text-to-speech.

## What This Project Does

This repository contains a small set of translation demos:

- Translate a sample English sentence with a MarianMT model from Hugging Face.
- Continuously transcribe microphone input with Google Speech Recognition.
- Translate spoken English to Spanish, or spoken Spanish to English, and optionally read the translation aloud.
- Keep GPU, Wav2Vec, and notebook experiments available without mixing them into the main source folder.

## Features

- One-command text translation demo.
- Microphone-based speech recognition stream.
- Bilingual English/Spanish voice translation demo.
- Example output and sample audio files.
- Clean project layout for GitHub visitors.
- Legacy environment freeze preserved for reference.

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- SpeechRecognition
- Google Speech Recognition
- pyttsx3
- langdetect
- googletrans

## Folder Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-experiments.txt
├── docs/
│   └── legacy-environment-freeze.txt
├── examples/
│   ├── sample_input.txt
│   ├── translated_output.txt
│   └── audio/
├── experiments/
│   ├── cupy_gpu_demo.py
│   └── wav2vec_stream.py
├── notebooks/
└── src/
    └── realtime_translation/
        ├── bilingual_voice_translator.py
        ├── speech_recognition_stream.py
        └── translate_text.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first text translation run downloads the selected Hugging Face model, so it requires an internet connection.

## Run Locally

Translate the default sample sentence:

```bash
python src/realtime_translation/translate_text.py
```

Translate custom text:

```bash
python src/realtime_translation/translate_text.py "Hello, how are you?"
```

Use a different MarianMT model:

```bash
python src/realtime_translation/translate_text.py "Hello, how are you?" --model Helsinki-NLP/opus-mt-en-es
```

Run continuous speech recognition:

```bash
python src/realtime_translation/speech_recognition_stream.py --language en-US
```

Run the English/Spanish voice translator:

```bash
python src/realtime_translation/bilingual_voice_translator.py
```

Run it without text-to-speech playback:

```bash
python src/realtime_translation/bilingual_voice_translator.py --no-speech
```

## Environment Variables

No environment variables are required.

## Example Usage

The default text translation command translates:

```text
Are you interested in a data plan for your simcard?
```

By default, the translated text is also saved to:

```text
examples/translated_output.txt
```

Use `--no-output-file` if you only want to print the translation.

## Experiments

Optional experiments live in `experiments/` and may require GPU-specific packages or extra setup:

```bash
pip install -r requirements-experiments.txt
python experiments/wav2vec_stream.py
```

CuPy installation depends on your CUDA version, so it is not pinned in `requirements-experiments.txt`. Install the package that matches your CUDA runtime before running the CuPy demo:

```bash
pip install cupy-cuda12x
python experiments/cupy_gpu_demo.py
```

## Testing and Verification

There are no automated tests yet. Use this syntax check as the current smoke test:

```bash
python -m compileall src experiments
```

There is no build step for this repository.

## Troubleshooting

- `PyAudio` installation fails: install PortAudio first. On Ubuntu/Debian, try `sudo apt-get install portaudio19-dev python3-dev`, then rerun `pip install -r requirements.txt`.
- Microphone is not detected: check your OS microphone permissions and default input device.
- Google Speech Recognition errors: the speech demos require internet access because they call Google's recognition service.
- Hugging Face model download fails: check your internet connection and retry the command.
- CUDA is not available: the main text translation script falls back to CPU automatically.
- `googletrans` errors: `googletrans` is an unofficial client and may occasionally break when Google changes its web translation behavior.

## Roadmap

- Add automated tests for argument parsing and pure helper functions.
- Add a single CLI entry point for all demos.
- Replace experimental Wav2Vec code with a maintained ASR pipeline.
- Add a short recorded sample and expected transcript for reproducible testing.

## License

No license file is currently included. Add a `LICENSE` file before accepting external contributions or reusing this code in another project.
