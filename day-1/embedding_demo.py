
"""
Embedding Similarity Demo

This script demonstrates how word embeddings capture semantic relationships between words.
It computes embeddings for a set of words and shows their pairwise similarities, revealing
that semantically related words (like 'tea' and 'coffee') have higher similarity scores
than unrelated words (like 'tea' and 'mud').

Key Concepts:
- Embeddings: Dense vector representations of words that capture semantic meaning
- Cosine Similarity: Measure of similarity between vectors (0-1 scale after normalization)
- Ollama: Local LLM server that provides embedding models
"""

import numpy as np
from tabulate import tabulate
import requests


def get_ollama_embedding(text):
    """
    Retrieve an embedding vector for the given text using Ollama's API.
    
    This function calls the local Ollama server (running on port 11434) to generate
    an embedding using the 'all-minilm' model. The all-minilm model is a lightweight
    sentence transformer that produces 384-dimensional embeddings optimized for
    semantic similarity tasks.
    
    Args:
        text (str): The input text to embed (can be a word, phrase, or sentence)
    
    Returns:
        list: A vector (list of floats) representing the embedding of the input text.
              The dimensionality depends on the model (all-minilm produces 384-d vectors)
    
    Raises:
        requests.exceptions.RequestException: If the Ollama server is not running
                                               or returns an error
    """
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "all-minilm", "prompt": text}
    )
    return response.json()["embedding"]


def main():
    """
    Main function that demonstrates semantic similarity between words using embeddings.
    
    Process:
    1. Define a set of test words with varying semantic relationships
    2. Generate embeddings for each word
    3. Normalize the embeddings to unit length (for cosine similarity)
    4. Calculate pairwise similarities between all words
    5. Display results in a formatted similarity matrix
    
    Expected Results:
    - 'tea' and 'coffee' should have high similarity (both beverages)
    - 'mud' and 'dirt' should have high similarity (both earth materials)
    - Cross-category comparisons should have lower similarity
    - Diagonal values should be 1.000 (perfect self-similarity)
    """
    # Test words chosen to demonstrate semantic groupings:
    # - Beverages: tea, coffee
    # - Earth materials: mud, dirt
    words = ["tea", "coffee", "mud", "dirt"]

    # Step 1: Get embeddings for each word
    # Each embedding is a high-dimensional vector (384 dimensions for all-minilm)
    # that captures the semantic meaning of the word
    embeddings = np.array([get_ollama_embedding(word) for word in words])
    
    # Step 2: Normalize embeddings to unit length
    # Normalization converts each vector to length 1, which allows us to use
    # the dot product as a measure of cosine similarity. After normalization:
    # - Dot product = 1.0 means identical vectors (same direction)
    # - Dot product = 0.0 means orthogonal vectors (completely different)
    # - Dot product close to 1.0 means similar semantic meaning
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Step 3: Calculate pairwise similarities
    # Using matrix multiplication (dot product) on normalized vectors gives us
    # cosine similarity for all pairs. The result is a symmetric matrix where
    # similarities[i][j] represents the similarity between words[i] and words[j]
    similarities = np.dot(embeddings, embeddings.T)

    # Step 4: Format and display the similarity matrix
    # Create a table with word labels on both axes for easy interpretation
    header = [""] + words  # Empty string for top-left cell, then word names
    table = [
        [words[i]] + [f"{similarities[i][j]:.3f}" for j in range(len(words))]
        for i in range(len(words))
    ]
    # Format each row: word label + similarity scores (formatted to 3 decimal places)

    # Display the similarity matrix in a grid format
    # Higher values (closer to 1.000) indicate stronger semantic similarity
    print(tabulate(table, headers=header, tablefmt="grid"))


if __name__ == "__main__":
    # Entry point: Only execute main() when script is run directly
    # (not when imported as a module)
    main()
