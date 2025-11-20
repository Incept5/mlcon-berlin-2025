import requests
import json
import os

api_key = os.getenv("FIREWORKS_API_KEY", "")

def generate_response(prompt):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/qwen3-30b-a3b", "max_tokens": 5000,
        "top_p": 0.95, "temperature": 0.3, "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Accept": "application/json", "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()
    actual_response = result["choices"][0]["message"]["content"]
    return actual_response

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)