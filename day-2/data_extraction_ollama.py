"""
Data Extraction Demo using Ollama

This script demonstrates how to use Ollama with the Qwen3-VL model to perform:
1. Text summarization - condensing long text into concise summaries
2. Data extraction - pulling specific information from unstructured text

The example uses a model card for Magistral-Small-2506 to showcase these capabilities.

Requirements:
- ollama package installed (pip install ollama)
- Ollama running locally with qwen3-vl:4b-instruct model pulled
"""

import ollama


def generate_response(prompt):
    """
    Generate a response from the Ollama model.
    
    This is the core function that interfaces with Ollama's API. It sends a prompt
    to the qwen3-vl:4b-instruct model and returns the generated response.
    
    Args:
        prompt (str): The input prompt/instruction for the model
        
    Returns:
        str: The model's generated response, or None if an error occurs
        
    Model Configuration:
        - model: qwen3-vl:4b-instruct - A 4B parameter vision-language model
        - think: False - Disables extended reasoning mode for faster responses
        - num_ctx: 8192 - Sets context window to 8K tokens
        - temperature: 0.7 - Controls randomness (0=deterministic, 1=creative)
    """
    try:
        response = ollama.generate(
            model="qwen3-vl:4b-instruct",  # Compact vision-language model
            prompt=prompt,
            think=False,  # Disable extended reasoning for concise responses
            options={
                "num_ctx": 8192,      # Context window size in tokens
                "temperature": 0.7    # Balance between deterministic and creative
            }
        )
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None


def summarise(text):
    """
    Summarize long text into a concise sentence.
    
    Uses prompt engineering to instruct the model to act as a summary assistant
    and condense the input text into a very short sentence.
    
    Args:
        text (str): The text to be summarized
        
    Returns:
        str: A very short summary sentence, or None if generation fails
        
    Example:
        >>> long_text = "Model card with multiple paragraphs..."
        >>> summarise(long_text)
        "Magistral-Small-2506 is a 24B reasoning model based on Mistral Small."
    """
    # Craft a focused prompt with clear role and instruction
    prompt = f"You are a summary assistant. Summarise the following text into very short sentence...\n{text}"
    return generate_response(prompt)


def extract(text, what):
    """
    Extract specific information from unstructured text.
    
    Uses the model as a data extraction assistant to find and return only the
    requested information without additional commentary or explanation.
    
    Args:
        text (str): The source text to extract information from
        what (str): Description of what information to extract (e.g., "email addresses",
                   "key features", "benchmark scores")
        
    Returns:
        str: The extracted information only, or None if generation fails
        
    Example:
        >>> text = "Contact us at support@example.com or call 555-0123"
        >>> extract(text, "email address")
        "support@example.com"
        
    Note:
        The prompt explicitly instructs the model to provide "the answer only, 
        nothing else" to get clean, parseable output suitable for further processing.
    """
    # Craft extraction prompt emphasizing concise, answer-only output
    prompt = (f"You are a concise data extraction assistant. Extract {what} from the following text,"
              f"give the answer only, nothing else...\n{text}")
    return generate_response(prompt)


def main():
    """
    Main execution function demonstrating summarization and extraction capabilities.
    
    This demo:
    1. Uses a real model card text for Magistral-Small-2506 as sample input
    2. Generates a concise summary of the entire document
    3. Extracts specific information (sampling parameters) from the text
    
    The example showcases practical use cases:
    - Document summarization for quick understanding
    - Targeted data extraction from technical documentation
    """
    # Sample text: Model card for Magistral-Small-2506
    # This is a real-world example of technical documentation that needs
    # to be processed and understood
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

    # Demo 1: Text Summarization
    # Generate a concise summary of the entire model card
    print("=== Summarization Demo ===")
    summary = summarise(text)
    print(f"Summary: {summary}\n")

    # Demo 2: Data Extraction
    # Extract specific information from the document
    print("=== Data Extraction Demo ===")
    data = "Sampling parameters"  # What we want to extract
    extracted_info = extract(text, data)
    print(f"{data}: {extracted_info}")


# Standard Python idiom to run main() when script is executed directly
if __name__ == "__main__":
    main()
