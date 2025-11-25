"""Visual Ollama Demo Script

This script demonstrates how to use Ollama's vision-capable language models (VLMs)
to analyze and answer questions about images. It supports testing multiple models
with various types of images including graphs, menus, receipts, diagrams, and more.

The script uses Ollama's chat API endpoint and sends images as base64-encoded strings.
It's designed to test the visual understanding capabilities of different Ollama models.
"""

import requests
import json
import base64
import os
from pathlib import Path

def encode_image_to_base64(image_path):
    """Convert image file to base64-encoded string.
    
    Ollama's API requires images to be sent as base64-encoded strings.
    This function reads an image file from disk and converts it to the
    appropriate format for the API.
    
    Args:
        image_path (str): Path to the image file on disk
        
    Returns:
        str: Base64-encoded string representation of the image,
             or None if encoding fails
    """
    try:
        # Open image in binary mode and encode to base64
        with open(image_path, "rb") as image_file:
            # Read the entire file, encode to base64, and convert bytes to string
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


def send_request(model, prompt, image_path=None):
    """Send a chat request to Ollama's API with optional image.
    
    Constructs and sends a request to Ollama's chat endpoint. The request
    includes a system message to set the assistant's behavior, and a user
    message with optional image(s) for visual analysis.
    
    Args:
        model (str): Name of the Ollama model to use (e.g., 'qwen3-vl:4b-instruct')
        prompt (str): The user's question or instruction about the image
        image_path (str, optional): Path to the image file to analyze
        
    Returns:
        dict: JSON response from Ollama containing the model's answer
        
    Raises:
        FileNotFoundError: If the specified image file doesn't exist
        Exception: If the request fails or image encoding fails
    """
    # Ollama's chat API endpoint - must be running locally
    url = "http://localhost:11434/api/chat"

    # The text prompt that will be sent to the model
    user_content = prompt

    # Ollama expects images as a list of base64-encoded strings
    # Multiple images can be sent in a single request
    images = []
    if image_path:
        # Validate that the image file exists before attempting to encode it
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Convert the image to base64 format required by Ollama
        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            raise Exception(f"Failed to encode image: {image_path}")

        # Add the encoded image to the images list
        # Note: This list can contain multiple images if needed
        images.append(base64_image)

    # Thinking models need more tokens for their internal reasoning process
    # Regular instruct models typically need fewer tokens for direct answers
    num_predict = 4096 if "thinking" in model else 1000

    # Construct the API request payload following Ollama's chat format
    payload = {
        "model": model,  # Which model to use for inference
        "messages": [
            {
                "role": "system",
                # System message sets the assistant's behavior and capabilities
                "content": "You are a helpful visual assistant, give detailed and accurate answers."
            },
            {
                "role": "user",
                "content": user_content,  # The user's question/prompt
                "images": images  # List of base64-encoded images (can be empty)
            }
        ],
        "stream": False,  # Get complete response at once, not streamed
        "options": {
            "temperature": 0.7,  # Controls randomness (0.0 = deterministic, 1.0 = creative)
            "num_predict": num_predict  # Maximum tokens to generate in response
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Send POST request to Ollama
        # Timeout of 120 seconds allows for slower models or complex image analysis
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

        # Check if request was successful (HTTP 200 OK)
        if response.status_code == 200:
            return response.json()
        else:
            # Provide detailed error information if request failed
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def main():
    """Main function to test vision models with various images and questions.
    
    This function defines a test suite of images and questions, then runs each
    question against each model to compare their visual understanding capabilities.
    The images cover a wide range of use cases:
    - Technical graphs and charts
    - Text in different languages (Japanese, French)
    - Handwritten content
    - Mathematical equations and musical notation
    - Diagrams and UI screenshots
    """
    
    # Test image collection covering various visual understanding tasks
    images = [
        "data/IMG_1.jpg",   # Technical: ASI camera performance graphs (read noise vs gain)
        "data/IMG_2.jpg",   # Chart: Bar graph with 5 sections
        "data/IMG_3.jpg",   # Scene: Urban London photograph
        "data/IMG_4.jpg",   # Text: Japanese menu (non-Latin script)
        "data/IMG_5.jpg",   # Text: French menu (Latin script with accents)
        "data/IMG_6.jpg",   # Text: Complex handwritten beer list
        "data/IMG_7.jpg",   # Document: Restaurant receipt (structured data)
        "data/IMG_8.jpg",   # Text: French menu photographed at an angle
        "data/IMG_9.jpg",   # Music: Sheet music (Rêverie by Debussy)
        "data/IMG_10.jpg",  # Diagram: Technical diagram
        "data/IMG_11.jpg",  # UI: Computer screen dialog with buttons
        "data/IMG_12.jpg",  # Math: Gamma function equation
    ]

    # List of vision-language models to test
    # Qwen3-VL models come in different sizes and variants:
    # - 2b/4b refers to the number of parameters (2 billion / 4 billion)
    # - 'instruct' models are fine-tuned for following instructions directly
    # - 'thinking' models show their reasoning process (need more tokens)
    models = [
        # "qwen3-vl:2b-instruct",   # Smaller, faster model
        # "qwen3-vl:2b-thinking",   # Shows reasoning process
        "qwen3-vl:4b-instruct",     # Currently active: larger, more capable
        # "qwen3-vl:4b-thinking",   # Larger model with reasoning
    ]

    # Test questions demonstrating different visual reasoning tasks
    questions = [
        # Example: Counting objects in an image
        # {"prompt": "How many graphs are there in this image?",
        #  "image": images[0],
        #  },
        
        # Active test: Reading specific values from a technical graph
        {
            "prompt": "Using the graph, what is the Read noise for gain=50?",
            "image": images[0],  # ASI camera graphs
        },
        
        # Example: Converting mathematical notation to Markdown format
        # {"prompt": "Please output the equation for Markdown",
        #  "image": images[11],  # Gamma function equation
        #  },
        
        # Example: Object detection with coordinate extraction
        # {"prompt": "Determine the co-ordinates of the 'Create App' button. The format of output should be like { 'Create App': [x1, y1, x2, y2] } ",
        #  "image": images[10],  # Computer screen dialog
        #  },
        
        # Example: Scene understanding and location inference
        # {
        #     "prompt": "Describe the scene and try to determine the location the image was taken.",
        #     "image": images[2],  # London street scene
        # },
    ]

    # Run the test suite: each model answers each question
    # This allows comparison of different models' visual understanding capabilities
    for model_name in models:
        # Print a clear separator for each model
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 60}")

        # Test each question with the current model
        for i, question in enumerate(questions):
            prompt = question["prompt"]
            image_path = question["image"]

            # Display the test case information
            print(f"\nQuestion {i + 1}: {prompt}")
            print(f"Image: {os.path.basename(image_path)}")
            print("-" * 40)

            try:
                # Send the request to Ollama and get the response
                response = send_request(model_name, prompt, image_path)

                # Parse and display the response from Ollama
                # Ollama's chat API returns responses in a specific JSON structure
                if 'message' in response:
                    message = response['message']

                    # Extract the final content (the model's answer)
                    if 'content' in message:
                        content = message['content']
                        if content:
                            print(f"Response: {content}")
                        else:
                            # Handle empty responses (can happen with thinking models)
                            if 'thinking' in message and message['thinking']:
                                # Thinking models may use all tokens for reasoning
                                print("Response: (empty - model spent all tokens on thinking, consider increasing token limit)")
                            else:
                                print("Response: (empty)")
                else:
                    # Unexpected response structure - show the raw response for debugging
                    print("Error: Unexpected response format")
                    print(f"Raw response: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")


# Standard Python idiom to run main() when script is executed directly
if __name__ == "__main__":
    main()
