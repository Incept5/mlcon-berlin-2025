"""
Music Score Reading Demo using Vision Models via Ollama

Demonstrates how to use vision models to read the musical key from music scores.
Compares multiple vision-capable models on the same music score image.

Models:
- qwen3-vl:4b-instruct: General-purpose vision-language model
- deepseek-ocr: Specialized OCR model optimized for text extraction

Requirements:
- Ollama installed and running (ollama.com)
- Models pulled: ollama pull qwen3-vl:4b-instruct && ollama pull deepseek-ocr

Usage:
    python read_music.py
"""

import ollama
import time
from typing import List, Dict, Any


# Configuration
MODELS: List[str] = [
    "qwen3-vl:4b-instruct",
    "qwen3-vl:2b-instruct",
    "qwen3-vl:8b-instruct",
    "deepseek-ocr"
]
IMAGE_PATH: str = "data/IMG_9.jpg"


def get_music_prompt(model_name: str) -> str:
    """
    Get the optimal prompt for reading the musical key from a score.

    Args:
        model_name: Name of the vision model

    Returns:
        Optimized prompt string for the model
    """
    return "Read the key of the music score in this image. What is the musical key signature?"


def warmup_model(model_name: str, image_path: str) -> None:
    """Run a dummy inference to load the model into memory."""
    print(f"Warming up model: {model_name}...")

    try:
        # Simple dummy prompt
        dummy_prompt = "What do you see in this image?"

        ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': dummy_prompt,
                'images': [image_path]
            }],
            options={'temperature': 0.0}
        )
        print(f"✓ Model {model_name} loaded and ready")
    except Exception as e:
        print(f"✗ Warmup failed for {model_name}: {e}")


def read_music_key(model_name: str, image_path: str) -> tuple[str, float]:
    """
    Read the musical key from a music score using the specified vision model.

    Args:
        model_name: Name of the Ollama vision model to use
        image_path: Path to the music score image file

    Returns:
        Tuple of (detected musical key, inference time in seconds)

    Raises:
        Exception: If the model fails or is not available
    """
    prompt = get_music_prompt(model_name)

    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }],
        options={'temperature': 0.0}  # Deterministic output
    )
    inference_time = time.time() - start_time

    return response['message']['content'], inference_time


def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_result(model_name: str, text: str, inference_time: float, width: int = 60) -> None:
    """Print music reading result in a formatted way."""
    print(f"\nModel: {model_name}")
    print("-" * width)
    print(text)
    print(f"\n⏱️  Inference time: {inference_time:.2f}s")
    print("=" * width)


def main() -> None:
    """Read the musical key from the specified music score using multiple models."""
    print_header("MUSIC SCORE KEY READING DEMO")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results: Dict[str, Any] = {}

    for model in MODELS:
        try:
            # Warmup the model first
            warmup_model(model, IMAGE_PATH)

            # Read the music key with timing
            text, inference_time = read_music_key(model, IMAGE_PATH)
            print_result(model, text, inference_time)

            results[model] = {
                'success': True,
                'time': inference_time
            }

        except Exception as e:
            print(f"\nModel: {model}")
            print("-" * 60)
            print(f"ERROR: {e}")
            print("=" * 60)
            results[model] = {
                'success': False,
                'error': str(e)
            }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, result in results.items():
        if result['success']:
            print(f"{model}: ✓ Success, {result['time']:.2f}s")
        else:
            print(f"{model}: ✗ ERROR - {result['error']}")
    print("="*60)


if __name__ == "__main__":
    main()
