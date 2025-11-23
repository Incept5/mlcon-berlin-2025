import requests

response = requests.post("http://localhost:1234/v1/chat/completions",
      json={"model": "qwen3-vl-4b-instruct",
"messages":[{"role": "user", "content": "Hello"}]}
)

data = response.json()
print(data["choices"][0]["message"]["content"].strip())