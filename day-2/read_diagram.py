"""
Diagram Analysis Demo using Vision Models via Ollama

Demonstrates how to use vision models to analyze diagrams and generate Mermaid descriptions.
Compares multiple vision models' ability to understand and describe diagram structures.

Models:
- qwen3-vl:4b-instruct: General-purpose vision-language model
- deepseek-ocr: Specialized OCR model optimized for text extraction

Requirements:
- Ollama installed and running (ollama.com)
- Models pulled: ollama pull qwen3-vl:4b-instruct && ollama pull deepseek-ocr

Usage:
    python read_diagram.py
"""

import ollama
import time
from typing import List, Dict, Any


# Configuration
MODELS: List[str] = [
    "qwen3-vl:30b-a3b-instruct",
]
IMAGE_PATH: str = "data/IMG_10.jpg"


def get_diagram_prompt(model_name: str) -> str:
    """
    Get the optimal diagram analysis prompt for the specified model.

    Args:
        model_name: Name of the vision model

    Returns:
        Optimized prompt string for the model
    """
    return "Analyze this diagram and provide a Mermaid diagram description that represents the structure and relationships shown in the image."


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


def analyze_diagram(model_name: str, image_path: str) -> tuple[str, float]:
    """
    Analyze a diagram image and generate a Mermaid description using the specified vision model.

    Args:
        model_name: Name of the Ollama vision model to use
        image_path: Path to the image file

    Returns:
        Tuple of (mermaid description, inference time in seconds)

    Raises:
        Exception: If the model fails or is not available
    """
    prompt = get_diagram_prompt(model_name)

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
    """Print diagram analysis result in a formatted way."""
    print(f"\nModel: {model_name}")
    print("-" * width)
    print(text)
    print(f"\n⏱️  Inference time: {inference_time:.2f}s")
    print("=" * width)


def main() -> None:
    """Run diagram analysis on the specified image using multiple models."""
    print_header("DIAGRAM ANALYSIS DEMO")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results: Dict[str, Any] = {}

    for model in MODELS:
        try:
            # Warmup the model first
            warmup_model(model, IMAGE_PATH)

            # Analyze diagram with timing
            text, inference_time = analyze_diagram(model, IMAGE_PATH)
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
