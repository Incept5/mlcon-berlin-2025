import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_response(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
        )
        actual_response = response.choices[0].message.content
        return actual_response
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    response = generate_response("Hello")
    print(response)