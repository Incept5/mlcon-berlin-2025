
#!/usr/bin/env python3
"""
Sentiment Analysis Script 04 - Complete Pipeline (First 100 Tweets)

**BUILDS UPON: analyse_sentiment_01.py, analyse_sentiment_02.py, and analyse_sentiment_kaggle.py**

This script represents the COMPLETE sentiment analysis pipeline:

PROGRESSION SUMMARY:
┌─────────────────────────────────────────────────────────────────────┐
│ Script 01: Interactive → Manual text entry, one at a time          │
│ Script 02: Batch       → Predefined list, automated processing     │
│ Script 03: Dataset     → Real-world data loading from Kaggle        │
│ Script 04: Complete    → Full pipeline with analysis & visualization│
└─────────────────────────────────────────────────────────────────────┘

IMPLEMENTS THE 4 STEPS FROM SCRIPT 03:
1. ✓ Import analyse_sentiment() from Script 02
2. ✓ Apply to content column: df['sentiment'] = df['content'].apply(analyse_sentiment)
3. ✓ Analyze distribution: df['sentiment'].value_counts()
4. ✓ Calculate percentages: df['sentiment'].value_counts(normalize=True)

NEW FEATURES:
- Processes first 100 tweets (manageable sample size)
- Progress tracking during analysis
- Results saved to CSV file
- Visualization with matplotlib
- Comprehensive statistics and insights

PREREQUISITES:
- Ollama must be running with qwen3-vl:4b-instruct model
- pip install kagglehub pandas matplotlib
- Kaggle API credentials configured
"""

import os
import pandas as pd
import kagglehub
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def analyse_sentiment(text, model="qwen3-vl:4b-instruct"):
    """
    Analyze sentiment of text using Ollama LLM.
    
    This is the same function from analyse_sentiment_02.py, imported here
    for the complete pipeline.
    
    Args:
        text (str): Text to analyze
        model (str): Ollama model name (default: qwen3-vl:4b-instruct)
    
    Returns:
        str: 'positive', 'neutral', or 'negative'
    """
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
        response.raise_for_status()
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
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return "error"

def load_kaggle_data(data):
    """
    Download and load a Kaggle dataset into a pandas DataFrame.
    
    Same function from analyse_sentiment_kaggle.py
    
    Args:
        data (str): The Kaggle dataset identifier
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    print(f"Downloading {data} dataset...")
    path = kagglehub.dataset_download(f"austinreese/{data}")
    print(f"Dataset downloaded to: {path}")

    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break
        if csv_file:
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"  → {df.shape[0]:,} rows (tweets/records)")
    print(f"  → {df.shape[1]} columns (features)")

    return df

def analyze_tweets_sentiment(df, num_tweets=100):
    """
    STEP 2: Apply sentiment analysis to tweets.
    
    This function implements the core pipeline step of applying
    analyse_sentiment() to each tweet in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        num_tweets (int): Number of tweets to analyze (default: 100)
    
    Returns:
        pd.DataFrame: DataFrame with new 'sentiment' column
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: Analyzing sentiment for first {num_tweets} tweets...")
    print(f"{'='*70}")
    
    # Take only the first N tweets
    df_sample = df.head(num_tweets).copy()
    
    # Initialize sentiment column with 'pending'
    df_sample['sentiment'] = 'pending'
    
    # Process each tweet with progress tracking
    for idx, row in df_sample.iterrows():
        # Calculate progress
        progress = idx + 1
        
        # Display progress
        print(f"Processing tweet {progress}/{num_tweets} ({progress/num_tweets*100:.1f}%)...", end='\r')
        
        # Analyze sentiment
        text = row['content']
        sentiment = analyse_sentiment(text)
        df_sample.at[idx, 'sentiment'] = sentiment
    
    print(f"\n✓ Completed analysis of {num_tweets} tweets!")
    
    return df_sample

def analyze_sentiment_distribution(df):
    """
    STEP 3 & 4: Analyze sentiment distribution.
    
    This function implements:
    - Step 3: Calculate value counts
    - Step 4: Calculate percentages
    
    Args:
        df (pd.DataFrame): DataFrame with 'sentiment' column
    
    Returns:
        tuple: (value_counts, percentages)
    """
    print(f"\n{'='*70}")
    print("STEP 3: Analyzing sentiment distribution...")
    print(f"{'='*70}")
    
    # Step 3: Get value counts
    sentiment_counts = df['sentiment'].value_counts()
    print("\nSentiment Counts:")
    print(sentiment_counts)
    
    print(f"\n{'='*70}")
    print("STEP 4: Calculating sentiment percentages...")
    print(f"{'='*70}")
    
    # Step 4: Get percentages
    sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
    print("\nSentiment Percentages:")
    for sentiment, percentage in sentiment_percentages.items():
        print(f"  {sentiment.capitalize()}: {percentage:.2f}%")
    
    return sentiment_counts, sentiment_percentages

def save_results(df, filename="sentiment_analysis_results.csv"):
    """
    Save analysis results to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        filename (str): Output filename
    """
    print(f"\n{'='*70}")
    print("Saving results to CSV...")
    print(f"{'='*70}")
    
    df.to_csv(filename, index=False)
    print(f"✓ Results saved to: {filename}")

def visualize_results(sentiment_counts, sentiment_percentages):
    """
    Create visualization of sentiment distribution.
    
    Args:
        sentiment_counts (pd.Series): Sentiment value counts
        sentiment_percentages (pd.Series): Sentiment percentages
    """
    print(f"\n{'='*70}")
    print("Creating visualization...")
    print(f"{'='*70}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define colors for sentiments
    colors = {
        'positive': '#2ecc71',  # Green
        'neutral': '#95a5a6',   # Gray
        'negative': '#e74c3c',  # Red
        'error': '#e67e22'      # Orange
    }
    
    # Get colors for each sentiment in the data
    bar_colors = [colors.get(sentiment, '#3498db') for sentiment in sentiment_counts.index]
    
    # Plot 1: Bar chart of counts
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
    ax1.set_xlabel('Sentiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Sentiment Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sentiment_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Pie chart of percentages
    ax2.pie(sentiment_percentages.values, 
            labels=[s.capitalize() for s in sentiment_percentages.index],
            colors=[colors.get(sentiment, '#3498db') for sentiment in sentiment_percentages.index],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Sentiment Distribution (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'sentiment_analysis_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    
    # Show the plot
    plt.show()

def display_sample_tweets(df, num_samples=5):
    """
    Display sample tweets for each sentiment category.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        num_samples (int): Number of samples to show per category
    """
    print(f"\n{'='*70}")
    print(f"Sample Tweets by Sentiment (up to {num_samples} per category)")
    print(f"{'='*70}")
    
    sentiments = ['positive', 'negative', 'neutral']
    
    for sentiment in sentiments:
        print(f"\n{sentiment.upper()} TWEETS:")
        print("-" * 70)
        
        # Get tweets for this sentiment
        sentiment_tweets = df[df['sentiment'] == sentiment].head(num_samples)
        
        if len(sentiment_tweets) == 0:
            print(f"  No {sentiment} tweets found in sample.")
            continue
        
        for idx, row in sentiment_tweets.iterrows():
            tweet_text = row['content']
            # Truncate long tweets
            if len(tweet_text) > 100:
                tweet_text = tweet_text[:97] + "..."
            print(f"  • {tweet_text}")

def main():
    """
    Complete Sentiment Analysis Pipeline
    
    IMPLEMENTS ALL 4 STEPS FROM analyse_sentiment_kaggle.py:
    1. ✓ Import analyse_sentiment() - Done at top of file
    2. ✓ Apply to content column - analyze_tweets_sentiment()
    3. ✓ Analyze distribution - analyze_sentiment_distribution()
    4. ✓ Calculate percentages - analyze_sentiment_distribution()
    
    ADDITIONAL FEATURES:
    - Progress tracking during analysis
    - Results saved to CSV
    - Visualization with matplotlib
    - Sample tweets display
    - Comprehensive statistics
    """
    print("="*70)
    print("COMPLETE SENTIMENT ANALYSIS PIPELINE")
    print("Analyzing First 100 Tweets from Trump Tweets Dataset")
    print("="*70)
    
    # STEP 1: Load the dataset
    print("\n" + "="*70)
    print("STEP 1: Loading Kaggle dataset...")
    print("="*70)
    tweets_df = load_kaggle_data("trump-tweets")
    
    print("\nDataset columns:")
    print(tweets_df.columns.tolist())
    
    # STEP 2: Analyze sentiment for first 100 tweets
    analyzed_df = analyze_tweets_sentiment(tweets_df, num_tweets=100)
    
    # STEP 3 & 4: Analyze distribution and calculate percentages
    sentiment_counts, sentiment_percentages = analyze_sentiment_distribution(analyzed_df)
    
    # Additional: Save results to CSV
    save_results(analyzed_df)
    
    # Additional: Display sample tweets
    display_sample_tweets(analyzed_df)
    
    # Additional: Create visualization
    visualize_results(sentiment_counts, sentiment_percentages)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total tweets analyzed: {len(analyzed_df)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  • sentiment_analysis_results.csv")
    print(f"  • sentiment_analysis_results.png")
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
