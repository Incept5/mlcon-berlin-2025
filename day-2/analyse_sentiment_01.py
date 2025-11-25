#!/usr/bin/env python3
"""
Sentiment Analysis Script 01 - Interactive Sentiment Analysis

This is the foundational script that demonstrates:
- How to interact with Ollama's API for LLM inference
- Basic prompt engineering for sentiment classification
- Robust error handling for API interactions
- Interactive command-line interface for user input

Key concepts:
- Uses Ollama's /api/generate endpoint locally (localhost:11434)
- Implements a simple 3-class sentiment classifier (positive/neutral/negative)
- Demonstrates proper exception handling for network requests
- Shows how to extract structured output from LLM responses

Prerequisites:
- Ollama must be installed and running locally
- The specified model (qwen3-vl:4b-instruct) must be pulled
  Run: ollama pull qwen3-vl:4b-instruct
"""

import requests

def analyse_sentiment(text, model="qwen3-vl:4b-instruct"):
    """
    Analyze the sentiment of input text using a local Ollama LLM.
    
    This function demonstrates the complete flow of LLM-based sentiment analysis:
    1. Construct a carefully crafted prompt for the LLM
    2. Send the prompt to Ollama's API endpoint
    3. Parse and clean the LLM's response
    4. Extract a standardized sentiment label
    
    Args:
        text (str): The text to analyze for sentiment
        model (str): The Ollama model to use. Default is 'qwen3-vl:4b-instruct'
                     which is a 4-billion parameter vision-language model that
                     works well for text classification tasks
    
    Returns:
        str: One of 'positive', 'neutral', 'negative', 'unknown', or 'error'
             - 'positive': Text expresses positive sentiment
             - 'neutral': Text is neutral or factual
             - 'negative': Text expresses negative sentiment
             - 'unknown': LLM response couldn't be parsed
             - 'error': API call failed or exception occurred
    
    Example:
        >>> analyse_sentiment("I love this product!")
        'positive'
        >>> analyse_sentiment("The weather is cloudy.")
        'neutral'
    """
    # Construct the prompt with clear, specific instructions
    # This is a critical part of prompt engineering:
    # - Be explicit about the expected output format ("exactly one word")
    # - Provide clear constraints (only 3 possible values)
    # - Use consistent formatting for better LLM understanding
    prompt = f"""
    Analyse the sentiment of the following text and respond with exactly one word:
    'positive', 'neutral', or 'negative'.
    Text: {text}
    Sentiment:
    """

    # Configure the Ollama API endpoint
    # Ollama runs locally by default on port 11434
    url = "http://localhost:11434/api/generate"
    
    # Prepare the request payload
    # - model: specifies which LLM to use
    # - prompt: the text we want the LLM to process
    # - stream: False means we want the complete response at once
    #           (streaming would give us tokens as they're generated)
    payload = { "model": model, "prompt": prompt, "stream": False }

    try:
        # Send the HTTP POST request to Ollama's API
        # The json=payload parameter automatically:
        # - Serializes our dict to JSON
        # - Sets Content-Type header to application/json
        response = requests.post(url, json=payload)
        
        # Check if request was successful (status code 200-299)
        # This will raise an HTTPError for 4xx or 5xx status codes
        response.raise_for_status()
        
        # Parse the JSON response from Ollama
        # Expected format: {"response": "positive", "model": "...", ...}
        result = response.json()
        
        # Extract the LLM's response text and normalize it
        # - get("response", "") safely retrieves the response key
        # - strip() removes leading/trailing whitespace
        # - lower() normalizes to lowercase for consistent matching
        sentiment = result.get("response", "").strip().lower()
        
        # Clean up the response to extract just the sentiment word
        # LLMs sometimes add extra text like "The sentiment is: positive"
        # We need to extract just the sentiment label itself
        sentiment_words = ['positive', 'negative', 'neutral']
        for word in sentiment_words:
            if word in sentiment:
                return word
        
        # If no exact match found, return the cleaned response or 'unknown'
        # This handles edge cases where the LLM doesn't follow instructions
        return sentiment if sentiment else "unknown"
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors:
        # - Connection refused: Ollama not running
        # - Timeout: Request took too long
        # - HTTP errors: Invalid model, API issues
        print(f"Error connecting to Ollama: {e}")
        return "error"
    except Exception as e:
        # Catch any other unexpected errors:
        # - JSON parsing errors
        # - Unexpected response format
        # - Other runtime exceptions
        print(f"Error processing response: {e}")
        return "error"

def main():
    """
    Interactive command-line interface for sentiment analysis.
    
    This function provides a simple REPL (Read-Eval-Print Loop) that:
    1. Prompts the user for text input
    2. Analyzes the sentiment using the LLM
    3. Displays the result
    4. Repeats until the user quits
    
    This interactive mode is ideal for:
    - Testing the sentiment analyzer
    - Exploring how different texts are classified
    - Debugging and experimentation
    """
    # Display welcome message and instructions
    print("Sentiment Analysis Tool")
    print("=======================")
    print("This tool uses a local Ollama model to analyze text sentiment.")
    print("Make sure Ollama is running...\n")
    
    # Main interaction loop - continues until user quits
    while True:
        # Get user input and remove leading/trailing whitespace
        user_text = input("Enter text to analyze (or 'quit' to exit): ").strip()
        
        # Check for quit commands (case-insensitive)
        # Multiple options for user convenience
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        # Validate that user entered some text
        # Empty input would waste an API call
        if not user_text:
            print("Please enter some text to analyze.\n")
            continue
            
        # Provide feedback that analysis is in progress
        # LLM inference can take a few seconds
        print("\nAnalyzing sentiment...")
        
        # Call the sentiment analysis function
        sentiment = analyse_sentiment(user_text)
        
        # Display the result
        if sentiment == "error":
            print("Failed to analyze sentiment. Please check that Ollama is running.\n")
        else:
            # Convert to uppercase for consistent display
            print(f"Sentiment: {sentiment.upper()}\n")

# Standard Python idiom: only run main() when script is executed directly
# This allows the module to be imported without running the interactive loop
if __name__ == "__main__":
    main()
