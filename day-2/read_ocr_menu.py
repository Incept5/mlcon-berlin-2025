"""
Menu OCR Demo using Vision Models via Ollama

Demonstrates how to use vision models for text extraction from images.
Compares multiple OCR-capable models on the same menu image.

Models:
- qwen3-vl:4b-instruct: General-purpose vision-language model
- deepseek-ocr: Specialized OCR model optimized for text extraction

Requirements:
- Ollama installed and running (ollama.com)
- Models pulled: ollama pull qwen3-vl:4b-instruct && ollama pull deepseek-ocr

Usage:
    python translate_menu.py
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
IMAGE_PATH: str = "data/IMG_5.jpg"


def get_ocr_prompt(model_name: str) -> str:
    """
    Get the optimal OCR prompt for the specified model.

    DeepSeek-OCR uses a specialized "Free OCR." prompt for best results.
    Other models use a more descriptive natural language prompt.

    Args:
        model_name: Name of the vision model

    Returns:
        Optimized prompt string for the model
    """
    if "deepseek-ocr" in model_name.lower():
        return "Free OCR."

    return "Read all text from this image. Output only the text exactly as written."


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


def extract_text_from_image(model_name: str, image_path: str) -> tuple[str, float]:
    """
    Extract text from an image using the specified vision model.

    Args:
        model_name: Name of the Ollama vision model to use
        image_path: Path to the image file

    Returns:
        Tuple of (extracted text, inference time in seconds)

    Raises:
        Exception: If the model fails or is not available
    """
    prompt = get_ocr_prompt(model_name)

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
    """Print OCR result in a formatted way."""
    print(f"\nModel: {model_name}")
    print("-" * width)
    print(text)
    print(f"\n⏱️  Inference time: {inference_time:.2f}s")
    print("=" * width)


def main() -> None:
    """Run OCR on the specified image using multiple models."""
    print_header("MENU OCR DEMO")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results: Dict[str, Any] = {}

    for model in MODELS:
        try:
            # Warmup the model first
            warmup_model(model, IMAGE_PATH)

            # Extract text with timing
            text, inference_time = extract_text_from_image(model, IMAGE_PATH)
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
