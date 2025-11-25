"""
Beer Detection Demo using Vision Models via Ollama

Simple demo comparing multiple vision models for object detection.
- Detects beer containers in images
- Draws bounding boxes with labels
- Saves and displays results for each model
- Converts normalized [0,1000] coordinates to pixel coordinates

Supported Models:
- qwen3-vl:4b-instruct - Returns JSON with bbox_2d arrays
- deepseek-ocr - Uses <|grounding|> token and returns <|ref|>/<|det|> tags

Usage: python find_beer.py
"""

import json
import re
import ollama
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
MODELS = ["qwen3-vl:4b-instruct", "deepseek-ocr"]
IMAGE_PATH = "data/RTS1UI9-1024x659.jpg"

# ============================================================================
# Core Functions
# ============================================================================

def extract_boxes_from_text(text):
    """Extract bounding boxes from model output text."""
    # First, try DeepSeek-OCR format: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]<|/det|>
    deepseek_ref_pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    deepseek_ref_match = re.search(deepseek_ref_pattern, text)

    if deepseek_ref_match:
        label = deepseek_ref_match.group(1).strip()
        det_content = deepseek_ref_match.group(2)

        # Extract all coordinate sets from the det content
        coord_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        coord_matches = re.findall(coord_pattern, det_content)

        if coord_matches:
            return [{"bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                     "label": label, "sub_label": label}
                    for x1, y1, x2, y2 in coord_matches]

    # Try to find JSON in code blocks
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Fallback: find coordinate patterns
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, text)
    if matches:
        return [{"bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                 "label": "Beer", "sub_label": "Beer"}
                for x1, y1, x2, y2 in matches]

    return None


def detect_beer(model_name, image_path):
    """Detect beer in image using specified model."""
    print(f"\n{'='*60}")
    print(f"Processing with model: {model_name}")
    print(f"{'='*60}")

    # DeepSeek-OCR uses special tokens and different format
    if "deepseek-ocr" in model_name.lower():
        prompt = """<|grounding|>Locate all beer glasses, beer mugs, and beer steins in this image."""
    else:
        prompt = """Detect all beer glasses, mugs, or steins in this image.

Return ONLY a JSON array:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "Beer", "sub_label": "Beer"}
]
```

Where x1,y1 is top-left and x2,y2 is bottom-right. Use integers."""

    # Call Ollama
    response = ollama.chat(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }],
        options={'temperature': 0.0}
    )

    output = response['message']['content']
    print(f"\nModel Response:\n{output[:500]}...")

    boxes = extract_boxes_from_text(output)

    if boxes:
        print(f"\n✓ Found {len(boxes)} beer container(s)")
        return boxes
    else:
        print("\n✗ No boxes detected")
        return []


def draw_boxes(image_path, boxes, model_name):
    """Draw bounding boxes on image and save/display."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    print(f"Image size: {img_width}x{img_height}")

    # Load font
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Draw each box
    for i, box in enumerate(boxes, 1):
        bbox = box.get('bbox_2d', [])
        if len(bbox) != 4:
            continue

        # Convert normalized [0,1000] coords to pixels
        x1 = int(bbox[0] / 1000.0 * img_width)
        y1 = int(bbox[1] / 1000.0 * img_height)
        x2 = int(bbox[2] / 1000.0 * img_width)
        y2 = int(bbox[3] / 1000.0 * img_height)

        print(f"  Box {i}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] -> [{x1}, {y1}, {x2}, {y2}]")

        # Draw box and label
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f"Beer {i}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3],
                      fill=(0, 0, 0))
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

    # Save with model-specific name
    model_suffix = model_name.replace(':', '-').replace('/', '-')
    output_path = image_path.replace('.jpg', f'-processed-{model_suffix}.jpg')
    image.save(output_path)
    print(f"\n✓ Saved: {output_path}")

    # Display
    image.show()

    return output_path


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BEER DETECTION DEMO")
    print("="*60)
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results = {}

    for model in MODELS:
        try:
            # Detect beer
            boxes = detect_beer(model, IMAGE_PATH)

            # Draw and display if boxes found
            if boxes:
                output_path = draw_boxes(IMAGE_PATH, boxes, model)
                results[model] = {
                    'boxes': len(boxes),
                    'output': output_path
                }
            else:
                results[model] = {
                    'boxes': 0,
                    'output': None
                }

        except Exception as e:
            print(f"\n✗ Error with {model}: {e}")
            results[model] = {'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, result in results.items():
        if 'error' in result:
            print(f"{model}: ERROR - {result['error']}")
        else:
            print(f"{model}: {result['boxes']} detections -> {result.get('output', 'N/A')}")
    print("="*60)
