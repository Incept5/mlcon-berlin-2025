#!/usr/bin/env python3
"""
Sentiment Analysis Script 02 - Batch Processing Evolution

**BUILDS UPON: analyse_sentiment_01.py**

This script demonstrates the evolution from interactive to batch processing:

KEY DIFFERENCES FROM SCRIPT 01:
1. **Batch Processing**: Instead of interactive input, processes a predefined list
2. **Simplified Error Handling**: Removed try-except blocks for cleaner code
   - Assumes Ollama is running (production code would add this back)
   - Focuses on the core sentiment analysis logic
3. **Streamlined Response Parsing**: Improved logic for extracting sentiment
4. **Direct Execution**: Runs automatically without user interaction

WHAT STAYED THE SAME:
- Core analyse_sentiment() function structure
- Same prompt engineering approach
- Same Ollama API endpoint and payload format
- Same 3-class sentiment classification

WHY THIS EVOLUTION MATTERS:
- Shows how to process multiple texts programmatically
- Demonstrates the pattern for integrating sentiment analysis into larger systems
- Serves as a stepping stone toward processing real datasets (Script 03)

USE CASES:
- Automated testing of sentiment analysis
- Processing small batches of text
- Integration into data pipelines
- Comparing sentiment across multiple texts
"""

import requests

def analyse_sentiment(text, model="qwen3-vl:4b-instruct"):
    """
    Analyze sentiment of text using Ollama LLM - Simplified Batch Version.
    
    EVOLUTION FROM SCRIPT 01:
    - Removed try-except error handling (assumes stable environment)
    - Streamlined response parsing with if-elif logic
    - Focus on core functionality for batch processing
    
    This simplified version is appropriate when:
    - Running in a controlled environment
    - Ollama is guaranteed to be running
    - You want cleaner, more readable code for demonstrations
    
    Args:
        text (str): Text to analyze
        model (str): Ollama model name (default: qwen3-vl:4b-instruct)
    
    Returns:
        str: 'positive', 'neutral', or 'negative'
    """
    # Same prompt engineering as Script 01
    # This consistency ensures reliable sentiment classification
    prompt = f"""
    Analyse the sentiment of the following text and respond with exactly one word:
    'positive', 'neutral', or 'negative'.
    Text: {text}
    Sentiment:
    """

    # Identical API configuration as Script 01
    url = "http://localhost:11434/api/generate"
    payload = { "model": model, "prompt": prompt, "stream": False }

    # SIMPLIFIED: No error handling here
    # In production, you'd want to add back the try-except blocks from Script 01
    # This cleaner version is better for demonstrations and controlled environments
    response = requests.post(url, json=payload)
    result = response.json()

    # Extract and normalize the LLM response (same as Script 01)
    sentiment = result.get("response", "").strip().lower()

    # IMPROVED PARSING LOGIC from Script 01
    # More explicit if-elif-else structure for clarity
    # Still handles cases where LLM adds extra text to response
    if sentiment not in ["positive", "neutral", "negative"]:
        # If response isn't an exact match, search for keywords
        if "positive" in sentiment:
            sentiment = "positive"
        elif "negative" in sentiment:
            sentiment = "negative"
        else:
            # Default to neutral if we can't determine sentiment
            # More conservative than Script 01's "unknown"
            sentiment = "neutral"
    return sentiment

if __name__ == "__main__":
    """
    Batch Processing Demo - Evolution from Interactive to Automated
    
    KEY CHANGE FROM SCRIPT 01:
    - No interactive loop or user input
    - Predefined list of test texts
    - Automatic execution and display of results
    
    This pattern demonstrates:
    1. How to process multiple texts programmatically
    2. Consistent formatting of results
    3. Easy integration into larger systems
    4. Foundation for processing real datasets (see Script 03)
    """
    # Test dataset representing the 3 sentiment classes
    # Each example chosen to be clearly positive, neutral, or negative
    # This makes it easy to verify the sentiment analyzer is working correctly
    texts = [
        "I had a wonderful day today!",           # Expected: positive
        "The weather is cloudy.",                 # Expected: neutral  
        "This is the worst service I've ever experienced."  # Expected: negative
    ]

    # Process each text in the batch
    # This loop demonstrates the pattern for integrating sentiment analysis
    # into data processing pipelines
    for text in texts:
        # Analyze sentiment using our LLM-based function
        sentiment = analyse_sentiment(text)
        
        # Display results in a clear, formatted way
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)  # Visual separator for readability