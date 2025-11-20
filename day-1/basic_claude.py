import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")

client = Anthropic(api_key=api_key)  # Use keyword argument, not positional

def generate_response(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # Updated to correct model name
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    actual_response = response.content[0].text
    return actual_response

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)