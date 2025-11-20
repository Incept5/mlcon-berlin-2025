import ollama
from datetime import date
from ollama import Options
from rich.console import Console

def main():
    today = date.today().strftime('%A, %d-%m-%Y')
    LLM = "qwen3"

    name = input("Enter your name: ")
    star_sign = input("Enter your star sign: ")

    system_prompt = f"""You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
        optimistic horoscope for tomorrow. Provide the response in Markdown format.
        Remember, the user is looking for a positive and optimistic outlook on their future.
        Use British English, metric and EU date formats where applicable."""

    instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

    response = ollama.chat( model=LLM, think=True, stream=False,
        messages=[ {'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': instruction} ],
        options=Options( temperature=0.8, num_ctx=4096, top_p=0.95, top_k=40, num_predict=-1 ))

    console = Console()

    if hasattr(response.message, 'thinking') and response.message.thinking:
        console.print(f"[bold blue]🤔 Maude's Thinking Process:[/bold blue]\n[dim]{response.message.thinking}[/dim]")
        console.print("\n" + "=" * 50 + "\n")

    console.print("[bold magenta]✨ Your Horoscope:[/bold magenta]")
    console.print(response.message.content)

if __name__ == "__main__":
    main()