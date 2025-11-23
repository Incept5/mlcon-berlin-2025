import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello",}],
    model="qwen/qwen3-32b",
)
print(chat_completion.choices[0].message.content)