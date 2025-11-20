import os
import re

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob


def load_kaggle_data(data):
    print(f"Downloading {data} dataset...")
    path = kagglehub.dataset_download(f"austinreese/{data}")
    print(f"Dataset downloaded to: {path}")

    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset")

    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"\nDataset shape: {df.shape}")

    return df


def clean_text(text):
    """Clean tweet text for better sentiment analysis"""
    if pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags for cleaner analysis
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text or text.strip() == "":
        return 0, 0  # neutral sentiment and subjectivity

    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def categorize_sentiment(polarity):
    """Categorize sentiment based on polarity score"""
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def get_sentiment_summary(df):
    """Generate comprehensive sentiment analysis summary"""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)

    # Overall sentiment distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    total_tweets = len(df)

    print(f"\nTotal tweets analyzed: {total_tweets:,}")
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"  {sentiment}: {count:,} tweets ({percentage:.1f}%)")

    # Average sentiment scores
    avg_polarity = df['polarity'].mean()
    avg_subjectivity = df['subjectivity'].mean()

    print(f"\nAverage Sentiment Metrics:")
    print(f"  Polarity (sentiment): {avg_polarity:.3f} (range: -1 to 1)")
    print(f"  Subjectivity: {avg_subjectivity:.3f} (range: 0 to 1)")

    # Interpretation
    if avg_polarity > 0.1:
        overall_sentiment = "Generally Positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "Generally Negative"
    else:
        overall_sentiment = "Generally Neutral"

    if avg_subjectivity > 0.5:
        subjectivity_level = "Highly Subjective/Opinionated"
    else:
        subjectivity_level = "More Objective/Factual"

    print(f"\nOverall Analysis:")
    print(f"  Sentiment Tendency: {overall_sentiment}")
    print(f"  Content Style: {subjectivity_level}")

    # Most positive and negative tweets
    most_positive = df.loc[df['polarity'].idxmax()]
    most_negative = df.loc[df['polarity'].idxmin()]

    print(f"\nMost Positive Tweet (polarity: {most_positive['polarity']:.3f}):")
    print(f"  {most_positive['content'][:200]}...")

    print(f"\nMost Negative Tweet (polarity: {most_negative['polarity']:.3f}):")
    print(f"  {most_negative['content'][:200]}...")

    # Time-based analysis if date column exists
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            yearly_sentiment = df.groupby('year')['polarity'].mean()

            print(f"\nSentiment Trends Over Time:")
            for year, sentiment in yearly_sentiment.items():
                print(f"  {year}: {sentiment:.3f}")
        except:
            print("\nNote: Could not perform time-based analysis")

    return sentiment_counts, avg_polarity, avg_subjectivity


if __name__ == "__main__":
    # Load the Trump tweets dataset
    tweets_df = load_kaggle_data("trump-tweets")

    print("\nFirst 6 rows of the dataset:")
    print(tweets_df.head(6))

    print("\nColumn names:")
    print(tweets_df.columns.tolist())

    # Identify the text column (usually 'content', 'text', or similar)
    text_column = None
    possible_text_columns = ['content', 'text', 'tweet', 'message']

    for col in possible_text_columns:
        if col in tweets_df.columns:
            text_column = col
            break

    if text_column is None:
        # Use the first string column that might contain tweets
        for col in tweets_df.columns:
            if tweets_df[col].dtype == 'object' and tweets_df[col].str.len().mean() > 20:
                text_column = col
                break

    if text_column is None:
        print("Could not identify text column for sentiment analysis")
        exit(1)

    print(f"\nUsing '{text_column}' column for sentiment analysis")

    # Clean and prepare the data
    print("\nCleaning tweet text...")
    tweets_df['cleaned_text'] = tweets_df[text_column].apply(clean_text)

    # Remove rows with empty text after cleaning
    tweets_df = tweets_df[tweets_df['cleaned_text'].str.len() > 0]
    print(f"Tweets after cleaning: {len(tweets_df)}")

    # Perform sentiment analysis
    print("\nPerforming sentiment analysis...")
    sentiment_results = tweets_df['cleaned_text'].apply(analyze_sentiment)
    tweets_df['polarity'] = sentiment_results.apply(lambda x: x[0])
    tweets_df['subjectivity'] = sentiment_results.apply(lambda x: x[1])
    tweets_df['sentiment_category'] = tweets_df['polarity'].apply(categorize_sentiment)

    # Rename the original text column for clarity
    tweets_df = tweets_df.rename(columns={text_column: 'content'})

    # Generate and display comprehensive summary
    sentiment_counts, avg_polarity, avg_subjectivity = get_sentiment_summary(tweets_df)

    # Create visualizations
    try:
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Sentiment distribution pie chart
        sentiment_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sentiment Distribution')
        ax1.set_ylabel('')

        # 2. Polarity distribution histogram
        ax2.hist(tweets_df['polarity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of Sentiment Polarity')
        ax2.set_xlabel('Polarity Score (-1 to 1)')
        ax2.set_ylabel('Number of Tweets')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax2.legend()

        # 3. Subjectivity distribution histogram
        ax3.hist(tweets_df['subjectivity'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Distribution of Subjectivity')
        ax3.set_xlabel('Subjectivity Score (0 to 1)')
        ax3.set_ylabel('Number of Tweets')

        # 4. Scatter plot of polarity vs subjectivity
        scatter = ax4.scatter(tweets_df['subjectivity'], tweets_df['polarity'],
                              alpha=0.5, c=tweets_df['polarity'], cmap='RdYlBu')
        ax4.set_title('Polarity vs Subjectivity')
        ax4.set_xlabel('Subjectivity')
        ax4.set_ylabel('Polarity')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Polarity')

        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'sentiment_analysis_results.png'")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create visualizations: {e}")

    # Save results to CSV
    output_df = tweets_df[['content', 'cleaned_text', 'polarity', 'subjectivity', 'sentiment_category']].copy()
    output_df.to_csv('sentiment_analysis_results.csv', index=False)
    print("\nDetailed results saved to 'sentiment_analysis_results.csv'")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
