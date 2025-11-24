"""
Token Visualization Script

This script demonstrates how Large Language Models (LLMs) break down text into tokens
using a tokenizer. Tokens are the fundamental units that LLMs process - they can be
words, subwords, or even individual characters depending on the tokenizer's vocabulary.

Understanding tokenization is crucial because:
1. Token count directly affects API costs and context window limits
2. Different tokenizers split text differently, affecting model behavior
3. Token boundaries can impact model understanding of text

This example uses the Qwen 3 tokenizer (0.6B parameter model) to show the tokenization
and detokenization process.
"""

import transformers

# Load the Qwen 3 tokenizer from HuggingFace
# This tokenizer defines how text is split into discrete tokens that the model can process
# The Qwen family uses a Byte Pair Encoding (BPE) tokenizer with a specific vocabulary
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def tokenize_text(text):
    """
    Convert text string into token IDs.
    
    Tokenization process:
    1. Text is broken into subword units based on the tokenizer's vocabulary
    2. Each subword is mapped to a unique integer ID
    3. These IDs are what the model actually processes
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        list[int]: A list of token IDs representing the input text
        
    Example:
        "Hello world" might become [15339, 1879] depending on the tokenizer's vocabulary
    """
    # return_tensors="pt" returns PyTorch tensors instead of plain Python lists
    # This is the format needed for model inputs, but we convert to list for display
    inputs = tokenizer(text, return_tensors="pt")
    
    # input_ids contains the token IDs - extract them as a Python list
    # [0] gets the first (and only) sequence from the batch dimension
    return inputs['input_ids'].tolist()[0]


def detokenize_tokens(token_ids):
    """
    Convert token IDs back into human-readable text.
    
    Detokenization reverses the tokenization process:
    1. Each token ID is looked up in the tokenizer's vocabulary
    2. The corresponding text pieces are retrieved
    3. They are concatenated back into a readable string
    
    Args:
        token_ids (list[int]): List of token IDs to convert back to text
        
    Returns:
        str: The reconstructed text string
        
    Note:
        Detokenization should reconstruct the original text (with possible 
        whitespace normalization), but the token boundaries may not be obvious
        from the final output.
    """
    return tokenizer.decode(token_ids)


def main():
    """
    Demonstrate tokenization and detokenization process.
    
    This example:
    1. Takes a sample text (the famous "Attention Is All You Need" paper title)
    2. Tokenizes it into individual token IDs
    3. Shows each token and its corresponding ID
    4. Reconstructs the original text from the token IDs
    
    Output format for each token:
        <token_text> = <token_id>
    
    This visualization helps understand:
    - How the model "sees" text
    - Where word/subword boundaries occur
    - That common words may be single tokens while rare words split into multiple tokens
    """
    # Sample text: title of the seminal Transformer paper from 2017
    text = "Attention Is All You Need"
    
    # Convert text to token IDs
    token_ids = tokenize_text(text)

    # Display each token individually with its numeric ID
    # This shows exactly how the text was split by the tokenizer
    print("Individual tokens and their IDs:")
    print("-" * 40)
    for token_id in token_ids:
        # Decode each token individually to see the text piece it represents
        # Note: Some tokens may include leading/trailing spaces as part of the token
        token_text = tokenizer.decode(token_id)
        print(f"'{token_text}' = {token_id}")

    # Verify we can reconstruct the original text from the token IDs
    print("\n" + "=" * 40)
    print("Reconstructed text from tokens:")
    print("-" * 40)
    detokenized_text = detokenize_tokens(token_ids)
    print(f"'{detokenized_text}'")
    
    # Additional information
    print("\n" + "=" * 40)
    print(f"Total tokens: {len(token_ids)}")
    print(f"Original text length: {len(text)} characters")
    print(f"Average characters per token: {len(text) / len(token_ids):.2f}")


# Standard Python idiom: only run main() when script is executed directly
# This allows the functions to be imported without running the demo
if __name__ == "__main__":
    main()
