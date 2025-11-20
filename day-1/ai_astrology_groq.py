import os
from groq import Groq
from datetime import date
from rich.console import Console
from rich.markdown import Markdown

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def main():
    today = date.today().strftime('%A, %d-%m-%Y')
    LLM = "qwen/qwen3-32b"

    name = input("Enter your name: ")
    star_sign = input("Enter your star sign: ")

    system_prompt = f"""You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
        optimistic horoscope for tomorrow. Provide the response in Markdown format.
        Remember, the user is looking for a positive and optimistic outlook on their future.
        Use British English, metric and EU date formats where applicable."""
    instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

    response = client.chat.completions.create(
        reasoning_effort="default", stream=False, model=LLM,
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': instruction, }],
    )
    console = Console()

    # Render the Markdown content
    markdown = Markdown(response.choices[0].message.content)
    console.print(markdown)

if __name__ == "__main__":
    main()