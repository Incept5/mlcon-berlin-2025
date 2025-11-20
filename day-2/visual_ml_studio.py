import requests
import json
import base64
import os
from pathlib import Path

def encode_image_to_base64(image_path):
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


def send_request(model, prompt, image_path=None):
    """Send a request to the local LLM and return the response."""
    url = "http://localhost:1234/v1/chat/completions"

    # Prepare the user message
    user_content = [{"type": "text", "text": prompt}]

    # Add image if provided
    if image_path:
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Get image extension to determine MIME type
        image_ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        mime_type = mime_types.get(image_ext, 'image/jpeg')

        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            raise Exception(f"Failed to encode image: {image_path}")

        # Add image to user content
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,  # Changed from -1 to a reasonable limit
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def main():
    images = [
        "data/IMG_1.jpg",  # The ASI camera graphs
        "data/IMG_2.jpg",  # A bar-graph with 5 sections
        "data/IMG_3.jpg",  # A London scene
        "data/IMG_4.jpg",  # A Japanese menu
        "data/IMG_5.jpg",  # A French menu
        "data/IMG_6.jpg",  # Complex handwritten board of beers
        "data/IMG_7.jpg",  # A restaurant receipt
        "data/IMG_8.jpg",  # French menu side-on
        "data/IMG_9.jpg",  # Music (Rêverie, Debussy)
        "data/IMG_10.jpg",  # Diagram
        "data/IMG_11.jpg",  # Computer screen dialog
        "data/IMG_12.jpg",  # Equation (Gamma function)
    ]

    models = [
        "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
        "mlx-community/Qwen2.5-VL-7B-Instruct-bf16",
        # "mlx-community/Qwen2.5-VL-32B-Instruct-bf16",
        "qwen2.5-vl-32b-instruct@8bit",
        "internvl3_5-30b-a3b",
        # "mlx-community/Qwen2.5-VL-72B-Instruct-4bit",
        "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        # "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-bf16",
    ]

    questions = [
        # {"prompt": "How many graphs are there in this image?",
        #  "image": images[0],
        #  },
        # {
        #     "prompt": "Using the graph, what is the Read noise for gain=50?",
        #     "image": images[0],
        # },
        {"prompt": "Please output the equation for Markdown",
         "image": images[11],
         },
        # {"prompt": "Determine the co-ordinates of the 'Create App' button. The format of output should be like { 'Create App': [x1, y1, x2, y2] } ",
        #  "image": images[10],
        #  },
        # {
        #     "prompt": "Describe the scene and try to determine the location the image was taken.",
        #     "image": images[2],
        # },
    ]

    # Test each model with each question
    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 60}")

        for i, question in enumerate(questions):
            prompt = question["prompt"]
            image_path = question["image"]

            print(f"\nQuestion {i + 1}: {prompt}")
            print(f"Image: {os.path.basename(image_path)}")
            print("-" * 40)

            try:
                response = send_request(model_name, prompt, image_path)

                # Extract the actual response text
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    print(f"Response: {content}")
                else:
                    print("Error: Unexpected response format")
                    print(f"Raw response: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()