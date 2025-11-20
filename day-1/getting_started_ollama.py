import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen3:4b", "prompt": "Hello",
          "stream": False, "Think": False }
)

data = response.json()
print(data["response"])
