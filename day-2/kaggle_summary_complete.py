
"""
Kaggle Tweet Sentiment Analysis Script

This script performs comprehensive sentiment analysis on the Trump tweets dataset from Kaggle.
It downloads the dataset, cleans the text, analyzes sentiment using TextBlob, and generates
detailed visualizations and summaries of the sentiment distribution.

Key Features:
- Downloads dataset from Kaggle using kagglehub
- Cleans tweet text (removes URLs, mentions, hashtags)
- Performs sentiment analysis (polarity and subjectivity)
- Categorizes tweets as Positive/Negative/Neutral
- Generates comprehensive statistical summary
- Creates visualizations (pie chart, histograms, scatter plot)
- Exports results to CSV
"""

import os
import re

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob


def load_kaggle_data(data):
    """
    Download and load a Kaggle dataset.
    
    This function downloads the specified dataset from Kaggle, locates the CSV file
    within the downloaded directory structure, and loads it into a pandas DataFrame.
    
    Args:
        data (str): Name of the Kaggle dataset (without username prefix)
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset
        
    Raises:
        FileNotFoundError: If no CSV file is found in the downloaded dataset
        
    Example:
        df = load_kaggle_data("trump-tweets")
    """
    print(f"Downloading {data} dataset...")
    # Download dataset from Kaggle (format: username/dataset-name)
    path = kagglehub.dataset_download(f"austinreese/{data}")
    print(f"Dataset downloaded to: {path}")

    # Search for CSV file in the downloaded directory structure
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break

    # Validate that a CSV file was found
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset")

    # Load the CSV file into a DataFrame
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"\nDataset shape: {df.shape}")

    return df


def clean_text(text):
    """
    Clean tweet text to improve sentiment analysis accuracy.
    
    Removes noise from tweet text including:
    - URLs (http/https links)
    - User mentions (@username)
    - Hashtags (#hashtag)
    - Extra whitespace
    
    Args:
        text (str): Raw tweet text to clean
        
    Returns:
        str: Cleaned text ready for sentiment analysis
        
    Example:
        clean = clean_text("Check out https://example.com @user #trending")
        # Returns: "Check out"
    """
    # Handle missing/null text values
    if pd.isna(text):
        return ""

    # Remove URLs (matches http, https, and www patterns)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username) and hashtags (#tag)
    # These add noise and don't contribute to sentiment
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Normalize whitespace (remove extra spaces, tabs, newlines)
    text = ' '.join(text.split())

    return text


def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    
    TextBlob provides two sentiment metrics:
    - Polarity: Ranges from -1 (negative) to 1 (positive)
    - Subjectivity: Ranges from 0 (objective) to 1 (subjective)
    
    Args:
        text (str): Text to analyze for sentiment
        
    Returns:
        tuple: (polarity, subjectivity) scores
        
    Example:
        polarity, subjectivity = analyze_sentiment("This is amazing!")
        # Returns: (0.8, 0.9) - highly positive and subjective
    """
    # Handle empty or whitespace-only text
    if not text or text.strip() == "":
        return 0, 0  # neutral sentiment and subjectivity

    # Create TextBlob object and extract sentiment
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def categorize_sentiment(polarity):
    """
    Categorize sentiment into discrete classes based on polarity score.
    
    Classification thresholds:
    - Positive: polarity > 0.1
    - Negative: polarity < -0.1
    - Neutral: -0.1 <= polarity <= 0.1
    
    Args:
        polarity (float): Sentiment polarity score (-1 to 1)
        
    Returns:
        str: Sentiment category ('Positive', 'Negative', or 'Neutral')
        
    Example:
        category = categorize_sentiment(0.5)  # Returns: 'Positive'
        category = categorize_sentiment(-0.3) # Returns: 'Negative'
    """
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def get_sentiment_summary(df):
    """
    Generate comprehensive sentiment analysis summary with statistics.
    
    Produces detailed analysis including:
    - Sentiment distribution (counts and percentages)
    - Average polarity and subjectivity scores
    - Overall sentiment tendency interpretation
    - Most positive and negative tweets
    - Time-based trends (if date column exists)
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
                          Must contain: 'sentiment_category', 'polarity', 
                          'subjectivity', 'content' columns
        
    Returns:
        tuple: (sentiment_counts, avg_polarity, avg_subjectivity)
    """
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)

    # Calculate sentiment distribution across categories
    sentiment_counts = df['sentiment_category'].value_counts()
    total_tweets = len(df)

    # Display overall statistics
    print(f"\nTotal tweets analyzed: {total_tweets:,}")
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"  {sentiment}: {count:,} tweets ({percentage:.1f}%)")

    # Calculate and display average sentiment metrics
    avg_polarity = df['polarity'].mean()
    avg_subjectivity = df['subjectivity'].mean()

    print(f"\nAverage Sentiment Metrics:")
    print(f"  Polarity (sentiment): {avg_polarity:.3f} (range: -1 to 1)")
    print(f"  Subjectivity: {avg_subjectivity:.3f} (range: 0 to 1)")

    # Interpret overall sentiment tendency
    if avg_polarity > 0.1:
        overall_sentiment = "Generally Positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "Generally Negative"
    else:
        overall_sentiment = "Generally Neutral"

    # Interpret subjectivity level
    if avg_subjectivity > 0.5:
        subjectivity_level = "Highly Subjective/Opinionated"
    else:
        subjectivity_level = "More Objective/Factual"

    print(f"\nOverall Analysis:")
    print(f"  Sentiment Tendency: {overall_sentiment}")
    print(f"  Content Style: {subjectivity_level}")

    # Find and display extreme examples
    most_positive = df.loc[df['polarity'].idxmax()]
    most_negative = df.loc[df['polarity'].idxmin()]

    print(f"\nMost Positive Tweet (polarity: {most_positive['polarity']:.3f}):")
    print(f"  {most_positive['content'][:200]}...")

    print(f"\nMost Negative Tweet (polarity: {most_negative['polarity']:.3f}):")
    print(f"  {most_negative['content'][:200]}...")

    # Perform time-based analysis if date information is available
    if 'date' in df.columns:
        try:
            # Convert date column to datetime format
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            # Calculate average sentiment per year
            yearly_sentiment = df.groupby('year')['polarity'].mean()

            print(f"\nSentiment Trends Over Time:")
            for year, sentiment in yearly_sentiment.items():
                print(f"  {year}: {sentiment:.3f}")
        except:
            print("\nNote: Could not perform time-based analysis")

    return sentiment_counts, avg_polarity, avg_subjectivity


if __name__ == "__main__":
    # ========================================================================
    # STEP 1: Load the dataset
    # ========================================================================
    tweets_df = load_kaggle_data("trump-tweets")

    # Display initial data exploration
    print("\nFirst 6 rows of the dataset:")
    print(tweets_df.head(6))

    print("\nColumn names:")
    print(tweets_df.columns.tolist())

    # ========================================================================
    # STEP 2: Identify the text column containing tweets
    # ========================================================================
    # Different datasets use different column names for text content
    text_column = None
    possible_text_columns = ['content', 'text', 'tweet', 'message']

    # Try common text column names first
    for col in possible_text_columns:
        if col in tweets_df.columns:
            text_column = col
            break

    # If no standard column name found, search for a string column with substantial content
    if text_column is None:
        for col in tweets_df.columns:
            # Check if column is text type and has average length > 20 characters
            if tweets_df[col].dtype == 'object' and tweets_df[col].str.len().mean() > 20:
                text_column = col
                break

    # Validate that we found a suitable text column
    if text_column is None:
        print("Could not identify text column for sentiment analysis")
        exit(1)

    print(f"\nUsing '{text_column}' column for sentiment analysis")

    # ========================================================================
    # STEP 3: Clean and prepare the data
    # ========================================================================
    print("\nCleaning tweet text...")
    tweets_df['cleaned_text'] = tweets_df[text_column].apply(clean_text)

    # Remove rows where cleaning resulted in empty text
    # (e.g., tweets that were only URLs or mentions)
    tweets_df = tweets_df[tweets_df['cleaned_text'].str.len() > 0]
    print(f"Tweets after cleaning: {len(tweets_df)}")

    # ========================================================================
    # STEP 4: Perform sentiment analysis
    # ========================================================================
    print("\nPerforming sentiment analysis...")
    # Apply sentiment analysis to each cleaned tweet
    sentiment_results = tweets_df['cleaned_text'].apply(analyze_sentiment)
    
    # Extract polarity and subjectivity scores from results
    tweets_df['polarity'] = sentiment_results.apply(lambda x: x[0])
    tweets_df['subjectivity'] = sentiment_results.apply(lambda x: x[1])
    
    # Categorize sentiment into discrete classes
    tweets_df['sentiment_category'] = tweets_df['polarity'].apply(categorize_sentiment)

    # Rename the original text column for clarity in output
    tweets_df = tweets_df.rename(columns={text_column: 'content'})

    # ========================================================================
    # STEP 5: Generate and display comprehensive summary
    # ========================================================================
    sentiment_counts, avg_polarity, avg_subjectivity = get_sentiment_summary(tweets_df)

    # ========================================================================
    # STEP 6: Create visualizations
    # ========================================================================
    try:
        # Set up matplotlib style and create 2x2 subplot grid
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Sentiment distribution pie chart
        # Shows percentage breakdown of Positive/Negative/Neutral tweets
        sentiment_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sentiment Distribution')
        ax1.set_ylabel('')  # Remove default ylabel for cleaner appearance

        # 2. Polarity distribution histogram
        # Shows frequency distribution of continuous polarity scores
        ax2.hist(tweets_df['polarity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of Sentiment Polarity')
        ax2.set_xlabel('Polarity Score (-1 to 1)')
        ax2.set_ylabel('Number of Tweets')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax2.legend()

        # 3. Subjectivity distribution histogram
        # Shows how objective vs subjective the tweets are
        ax3.hist(tweets_df['subjectivity'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Distribution of Subjectivity')
        ax3.set_xlabel('Subjectivity Score (0 to 1)')
        ax3.set_ylabel('Number of Tweets')

        # 4. Scatter plot of polarity vs subjectivity
        # Shows relationship between sentiment and subjectivity
        # Color-coded by polarity (red = negative, blue = positive)
        scatter = ax4.scatter(tweets_df['subjectivity'], tweets_df['polarity'],
                              alpha=0.5, c=tweets_df['polarity'], cmap='RdYlBu')
        ax4.set_title('Polarity vs Subjectivity')
        ax4.set_xlabel('Subjectivity')
        ax4.set_ylabel('Polarity')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Neutral line
        plt.colorbar(scatter, ax=ax4, label='Polarity')

        # Save and display visualizations
        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'sentiment_analysis_results.png'")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create visualizations: {e}")

    # ========================================================================
    # STEP 7: Save results to CSV
    # ========================================================================
    # Export analysis results for further processing or review
    output_df = tweets_df[['content', 'cleaned_text', 'polarity', 'subjectivity', 'sentiment_category']].copy()
    output_df.to_csv('sentiment_analysis_results.csv', index=False)
    print("\nDetailed results saved to 'sentiment_analysis_results.csv'")

    # Final completion message
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
