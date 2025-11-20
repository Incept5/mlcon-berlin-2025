import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_response(prompt):
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt} ],
        temperature=0.7,
    )
    actual_response = response.choices[0].message.content
    return actual_response

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)