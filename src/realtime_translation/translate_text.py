"""Translate a single text prompt with a Hugging Face MarianMT model."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-hi"
DEFAULT_TEXT = "Are you interested in a data plan for your simcard?"
DEFAULT_OUTPUT_PATH = Path("examples/translated_output.txt")


def translate_text(text: str, model_name: str = DEFAULT_MODEL) -> str:
    """Translate text with the selected MarianMT model."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    translated_ids = model.generate(input_ids)
    return tokenizer.decode(translated_ids[0], skip_special_tokens=True)


def save_translation(translated_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(translated_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate one sentence using a Hugging Face MarianMT model."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help="Text to translate. Defaults to the original sample sentence.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the translated text. Default: %(default)s",
    )
    parser.add_argument(
        "--no-output-file",
        action="store_true",
        help="Print the translation without updating the sample output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translated_sentence = translate_text(args.text, args.model)

    print("Original sentence:", args.text)
    print("Translated sentence:", translated_sentence)

    if not args.no_output_file:
        save_translation(translated_sentence, args.output)
        print(f"Saved translation to {args.output}")


if __name__ == "__main__":
    main()
