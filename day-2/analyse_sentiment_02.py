import requests

def analyse_sentiment(text, model="llama3.2"):
    prompt = f"""
    Analyse the sentiment of the following text and respond with exactly one word:
    positive', 'neutral', or 'negative'.
    Text: {text}
    Sentiment:
    """

    url = "http://localhost:11434/api/generate"
    payload = { "model": model, "prompt": prompt,"stream": False }

    response = requests.post(url, json=payload)
    result = response.json()

    sentiment = result.get("response", "").strip().lower()

    if sentiment not in ["positive", "neutral", "negative"]:
        if "positive" in sentiment:
            sentiment = "positive"
        elif "negative" in sentiment:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    return sentiment

if __name__ == "__main__":
    texts = [
        "I had a wonderful day today!",
        "The weather is cloudy.",
        "This is the worst service I've ever experienced."
    ]

    for text in texts:
        sentiment = analyse_sentiment(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)