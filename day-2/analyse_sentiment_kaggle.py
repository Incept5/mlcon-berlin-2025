#!/usr/bin/env python3
"""
Sentiment Analysis Script 03 - Real Dataset Integration (Kaggle)

**BUILDS UPON: analyse_sentiment_01.py and analyse_sentiment_02.py**

This script represents the final evolution in the sentiment analysis progression:

PROGRESSION SUMMARY:
┌─────────────────────────────────────────────────────────────────────┐
│ Script 01: Interactive → Manual text entry, one at a time          │
│ Script 02: Batch       → Predefined list, automated processing     │
│ Script 03: Dataset     → Real-world data from Kaggle                │
└─────────────────────────────────────────────────────────────────────┘

KEY DIFFERENCES FROM SCRIPTS 01 & 02:
1. **Real Dataset Integration**: Uses Kaggle API to download actual datasets
2. **Data Infrastructure**: Introduces pandas for data manipulation
3. **Data Discovery**: Demonstrates how to find and load CSV files
4. **Data Exploration**: Shows dataset inspection techniques
5. **Production-Ready Pattern**: Handles file system operations

NEW CONCEPTS INTRODUCED:
- kagglehub: Python library for downloading Kaggle datasets
- pandas: Data manipulation and analysis library
- os.walk(): Recursive file system traversal
- DataFrame operations: head(), shape, to_json()

WHAT THIS PREPARES FOR:
- Combining with analyse_sentiment() from Scripts 01/02
- Processing thousands of tweets or reviews
- Building real sentiment analysis pipelines
- Integration with data science workflows

NOTE: This script focuses on DATA LOADING, not sentiment analysis.
      To complete the pipeline, you would:
      1. Import analyse_sentiment() from Script 01 or 02
      2. Apply it to the 'text' column of the DataFrame
      3. Store results in a new 'sentiment' column
      4. Analyze sentiment distribution across the dataset

PREREQUISITES:
- Kaggle account and API credentials configured
- pip install kagglehub pandas
- Internet connection for dataset download
"""

import os
import pandas as pd
import kagglehub

def load_kaggle_data(data):
    """
    Download and load a Kaggle dataset into a pandas DataFrame.
    
    This function demonstrates the complete workflow for integrating
    real-world datasets into your sentiment analysis pipeline:
    
    WORKFLOW:
    1. Download dataset from Kaggle using kagglehub
    2. Navigate the downloaded file structure
    3. Locate the CSV file
    4. Load into pandas DataFrame
    5. Display basic dataset information
    
    This pattern is reusable for ANY Kaggle dataset, not just tweets.
    
    Args:
        data (str): The Kaggle dataset identifier (format: "username/dataset-name")
                   Example: "austinreese/trump-tweets"
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame
    
    Raises:
        FileNotFoundError: If no CSV file is found in the downloaded dataset
    
    Example:
        >>> df = load_kaggle_data("austinreese/trump-tweets")
        Downloading trump-tweets dataset...
        Dataset downloaded to: /path/to/dataset
        Reading CSV file: /path/to/dataset/tweets.csv
        Dataset shape: (56571, 10)
    """
    # Step 1: Download the dataset from Kaggle
    # kagglehub handles authentication via ~/.kaggle/kaggle.json
    print(f"Downloading {data} dataset...")
    path = kagglehub.dataset_download(f"austinreese/{data}")
    print(f"Dataset downloaded to: {path}")
    
    # The dataset is cached locally, so subsequent runs are fast
    # Path typically looks like: ~/.cache/kagglehub/datasets/...

    # Step 2: Find the CSV file in the downloaded dataset
    # Kaggle datasets can have complex directory structures
    # os.walk() recursively searches all subdirectories
    csv_file = None
    for root, dirs, files in os.walk(path):
        # Iterate through all files in current directory
        for file in files:
            if file.endswith('.csv'):
                # Construct full path to the CSV file
                csv_file = os.path.join(root, file)
                break  # Stop at first CSV file found
        # If CSV found, break outer loop too
        if csv_file:
            break

    # Step 3: Validate that we found a CSV file
    if csv_file is None:
        raise FileNotFoundError(
            "No CSV file found in the downloaded dataset. "
            "The dataset might not contain CSV files, or the download failed."
        )

    # Step 4: Load CSV into pandas DataFrame
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Step 5: Display basic dataset information
    # shape returns (rows, columns) - essential for understanding dataset size
    print(f"\nDataset shape: {df.shape}")
    print(f"  → {df.shape[0]:,} rows (tweets/records)")
    print(f"  → {df.shape[1]} columns (features)")

    # Return the DataFrame for further processing
    # This DataFrame can now be used with analyse_sentiment() from Scripts 01/02
    return df

if __name__ == "__main__":
    """
    Dataset Exploration Demo - Foundation for Production Pipeline
    
    EVOLUTION FROM SCRIPTS 01 & 02:
    ┌────────────────────────────────────────────────────────────────┐
    │ Script 01: analyse_sentiment() + interactive input             │
    │ Script 02: analyse_sentiment() + batch processing              │
    │ Script 03: load_kaggle_data() + dataset exploration            │
    └────────────────────────────────────────────────────────────────┘
    
    TO BUILD COMPLETE PIPELINE:
    1. Import analyse_sentiment from Script 01 or 02
    2. Apply to DataFrame: df['sentiment'] = df['text'].apply(analyse_sentiment)
    3. Analyze results: df['sentiment'].value_counts()
    4. Visualize distribution of sentiments
    
    This script focuses on the DATA LOADING piece of the pipeline.
    It demonstrates how to work with real-world datasets from Kaggle.
    """
    # Load the Trump tweets dataset from Kaggle
    # This dataset contains thousands of tweets - perfect for sentiment analysis
    # Dataset info: https://www.kaggle.com/datasets/austinreese/trump-tweets
    tweets_df = load_kaggle_data("trump-tweets")

    # Explore the dataset structure
    print("\nColumn names in the dataset:")
    print(tweets_df.columns.tolist())
    
    print("\nFirst 6 rows of the dataset:")
    print(tweets_df.head(6))
    # This shows us what data is available:
    # - 'text' column: The actual tweet content (what we'll analyze)
    # - Other columns: Metadata like date, retweets, favorites, etc.

    # Demonstrate different data formats
    # to_json() is useful for API integrations or data exchange
    print("\nFirst 2 rows as JSON:")
    print(tweets_df.head(2).to_json(orient='records', indent=2))
    # orient='records' creates a list of dictionaries
    # indent=2 makes it human-readable

    # Dataset statistics
    print(f"\nTotal number of rows: {len(tweets_df):,}")
    print("\nThis dataset is now ready for sentiment analysis!")
    print("\nNEXT STEPS to complete the pipeline:")
    print("1. Import analyse_sentiment() from Script 01 or 02")
    print("2. Apply to text column: tweets_df['sentiment'] = tweets_df['text'].apply(analyse_sentiment)")
    print("3. Analyze distribution: tweets_df['sentiment'].value_counts()")
    print("4. Calculate percentages: tweets_df['sentiment'].value_counts(normalize=True)")
    print("\nWARNING: Processing thousands of tweets with LLM will take time!")
    print("Consider processing a sample first: tweets_df.head(100)")
