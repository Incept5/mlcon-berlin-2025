import requests

response = requests.post("http://localhost:1234/v1/chat/completions",
            json={"model": "qwen3-4b-mlx",
            "messages": [{"role": "user", "content": "Hello"}]}
)

data = response.json()
print(data["choices"][0]["message"]["content"].strip())