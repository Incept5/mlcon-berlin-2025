import requests

def analyse_sentiment(text, model="llama3.2"):
    prompt = f"""
    Analyse the sentiment of the following text and respond with exactly one word:
    'positive', 'neutral', or 'negative'.
    Text: {text}
    Sentiment:
    """

    url = "http://localhost:11434/api/generate"
    payload = { "model": model, "prompt": prompt, "stream": False }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        sentiment = result.get("response", "").strip().lower()
        
        # Clean up the response to extract just the sentiment word
        sentiment_words = ['positive', 'negative', 'neutral']
        for word in sentiment_words:
            if word in sentiment:
                return word
        
        # If no exact match found, return the cleaned response
        return sentiment if sentiment else "unknown"
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return "error"
    except Exception as e:
        print(f"Error processing response: {e}")
        return "error"

def main():
    print("Sentiment Analysis Tool")
    print("=======================")
    print("This tool uses a local Ollama model to analyze text sentiment.")
    print("Make sure Ollama is running with the llama3.2 model.\n")
    
    while True:
        # Get user input
        user_text = input("Enter text to analyze (or 'quit' to exit): ").strip()
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_text:
            print("Please enter some text to analyze.\n")
            continue
            
        print("\nAnalyzing sentiment...")
        
        # Analyze sentiment
        sentiment = analyse_sentiment(user_text)
        
        if sentiment == "error":
            print("Failed to analyze sentiment. Please check that Ollama is running.\n")
        else:
            print(f"Sentiment: {sentiment.upper()}\n")

if __name__ == "__main__":
    main()
