import ollama

def generate_response(prompt):
    try:
        response = ollama.generate(
            model="qwen3:4b",
            prompt=prompt,
            think=False,
            options={"num_ctx": 8192,"temperature": 0.7}
        )
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None

def summarise(text):
    prompt = f"You are a summary assistant. Summarise the following text into very short sentence...\n{text}"
    return generate_response(prompt)

def extract(text, what):
    prompt = (f"You are a concise data extraction assistant. Extract {what} from the following text,"
              f"give the answer only, nothing else...\n{text}")
    return generate_response(prompt)

def main():
    text = """## Model Card for Magistral-Small-2506
    Building upon Mistral Small 3.1 (2503), with added reasoning capabilities, undergoing SFT from Magistral Medium traces and RL on top, it's a small, efficient reasoning model with 24B parameters.
    Magistral Small can be deployed locally, fitting within a single RTX 4090 or a 32GB RAM MacBook once quantized.
    Learn more about Magistral in our blog post.
    The model was presented in the paper Magistral.

    Key Features
    * Reasoning: Capable of long chains of reasoning traces before providing an answer.
    * Multilingual: Supports dozens of languages, including English, French, German, Greek, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Nepali, Polish, Portuguese, Romanian, Russian, Serbian, Spanish, Turkish, Ukrainian, Vietnamese, Arabic, Bengali, Chinese, and Farsi.
    * Apache 2.0 License: Open license allowing usage and modification for both commercial and non-commercial purposes.
    * Context Window: A 128k context window, but performance might degrade past 40k. Hence we recommend setting the maximum model length to 40k.

    Benchmark Results
    Model	AIME24 pass@1	AIME25 pass@1	GPQA Diamond	Livecodebench (v5)
    Magistral Medium	73.59%	64.95%	70.83%	59.36%
    Magistral Small	70.68%	62.76%	68.18%	55.84%

    Sampling parameters
    Please make sure to use:
    * top_p: 0.95
    * temperature: 0.7
    * max_tokens: 40960

    Basic Chat Template
    We highly recommend including the default system prompt used during RL for the best results, you can edit and customise it if needed for your specific use case.
    ```
    <s>[SYSTEM_PROMPT]system_prompt
    A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.
    Your thinking process must follow the template below:
    <think>
    Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
    </think>
    Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.
    Problem:
    [/SYSTEM_PROMPT][INST]user_message[/INST]<think>
    reasoning_traces
    </think>
    assistant_response</s>[INST]user_message[/INST]
    ```
    system_prompt, user_message and assistant_response are placeholders.
    We invite you to choose, depending on your use case and requirements, between keeping reasoning traces during multi-turn interactions or keeping only the final assistant response.
    """

    summary = summarise(text)
    print(f"Summary: {summary}")

    data = "Sampling parameters"
    num_paymernts = extract(text, data)
    print(f"{data}: {num_paymernts}")

if __name__ == "__main__":
    main()