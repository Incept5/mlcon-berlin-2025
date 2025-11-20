from mistralai import Mistral
import os

api_key = os.getenv("MISTRAL_API_KEY", "")
client = Mistral(api_key=api_key)

def generate_response(prompt):
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"content": prompt, "role": "user"}],
            stream=False,
            temperature=0.3
        )
        actual_response = response.choices[0].message.content
        return actual_response

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)