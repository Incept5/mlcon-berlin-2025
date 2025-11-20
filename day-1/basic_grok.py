import os
from openai import OpenAI

api_key = os.getenv("X_API_KEY", "")

def generate_response(prompt):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {"role": "user", "content": "What's the latest news from Trump?"}
        ]
    )
    return(completion.choices[0].message.content)

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)