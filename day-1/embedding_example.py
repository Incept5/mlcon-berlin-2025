"""
Embedding-based Sentence Similarity Matcher

This script demonstrates how to use embeddings to find semantically similar sentences.
It uses Ollama's local API to generate embeddings and calculates cosine similarity
to measure how closely related sentences are to a user's question.

Key Concepts:
- Embeddings: Vector representations of text that capture semantic meaning
- Cosine Similarity: Metric to measure similarity between vectors (ranges from -1 to 1)
- Standard Deviation Analysis: Statistical measure to understand how similarity scores
  are distributed and identify outliers
"""

import numpy as np
import requests


def get_ollama_embedding(text: str) -> list:
    """
    Get embedding vector from Ollama API for the given text.
    
    Embeddings are dense vector representations that capture the semantic meaning
    of text. Similar meanings result in similar vectors, which can be compared
    mathematically.
    
    Args:
        text: The input text to convert into an embedding vector
        
    Returns:
        list: A numerical vector (typically 384 or 768 dimensions depending on model)
              representing the semantic meaning of the text, or None if an error occurs
              
    Note:
        - Uses qwen3-embedding:0.6b model (lightweight, efficient)
        - Alternative model commented out: all-minilm:33m-l12-v2-fp16
        - Requires Ollama to be running locally on port 11434
    """
    data = {
        # Model selection: smaller models are faster but may be less accurate
        "model": "all-minilm",  # Using the same model as embedding_demo.py
        "prompt": text  # The text to embed
    }
    try:
        # Make POST request to Ollama's embedding endpoint
        response = requests.post("http://localhost:11434/api/embeddings", json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Extract the embedding vector from the JSON response
        embedding = response.json()["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors, effectively measuring
    how similar their directions are regardless of their magnitudes. This makes it
    ideal for comparing embeddings.
    
    Formula: cos(θ) = (a · b) / (||a|| × ||b||)
    
    Args:
        a: First vector (embedding)
        b: Second vector (embedding)
        
    Returns:
        float: Similarity score between -1 and 1, where:
               1.0 = identical direction (very similar)
               0.0 = orthogonal (unrelated)
              -1.0 = opposite direction (very dissimilar)
              
    Note:
        In practice, embeddings typically produce scores between 0.3 and 0.95
        for text similarity comparisons.
    """
    # np.dot(a, b) computes the dot product (sum of element-wise multiplication)
    # np.linalg.norm() computes the vector magnitude (length)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_sentence(question: str, sentences: list) -> list:
    """
    Find and rank sentences by their semantic similarity to the question.
    
    This function performs the following steps:
    1. Converts the question into an embedding vector
    2. Converts each sentence into an embedding vector
    3. Calculates cosine similarity between question and each sentence
    4. Computes statistical measures (mean, standard deviation)
    5. Returns ranked results with similarity scores and statistical context
    
    Args:
        question: The user's input question to match against
        sentences: List of candidate sentences to compare
        
    Returns:
        tuple: (enhanced_similarities, mean_similarity, std_similarity) where:
            - enhanced_similarities: List of tuples (sentence, similarity, std_devs_from_mean)
              sorted by similarity (highest first)
            - mean_similarity: Average similarity score across all sentences
            - std_similarity: Standard deviation of similarity scores
            
    Statistical Context:
        Standard deviation (σ) helps identify how unusual a similarity score is:
        - Scores within 1σ of mean: typical similarity
        - Scores > 2σ above mean: unusually high similarity (strong match)
        - Scores > 3σ above mean: exceptionally high similarity (very strong match)
    """
    # Step 1: Get embedding for the question
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        return [], 0, 0  # Return empty results if embedding fails

    # Step 2: Calculate similarities with all sentences
    similarities = []  # Store (sentence, similarity) tuples
    similarity_scores = []  # Store just the scores for statistical analysis

    for sentence in sentences:
        # Get embedding for this sentence
        sentence_embedding = get_ollama_embedding(sentence)
        
        if sentence_embedding is not None:
            # Calculate how similar this sentence is to the question
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))
            similarity_scores.append(similarity)

    # Step 3: Calculate statistics for context
    # This helps us understand which similarities are unusually high or low
    if similarity_scores:
        mean_similarity = np.mean(similarity_scores)  # Average similarity
        std_similarity = np.std(similarity_scores)    # Spread of similarities

        # Step 4: Add statistical context to each result
        # Standard deviations from mean tells us how unusual each score is
        enhanced_similarities = []
        for sentence, similarity in similarities:
            if std_similarity > 0:
                # Calculate z-score: how many standard deviations from the mean
                std_devs_from_mean = (similarity - mean_similarity) / std_similarity
            else:
                # If all similarities are identical, std_similarity is 0
                std_devs_from_mean = 0
                
            enhanced_similarities.append((sentence, similarity, std_devs_from_mean))

        # Step 5: Sort by similarity score (highest first)
        # This puts the most relevant sentences at the top
        enhanced_similarities.sort(key=lambda x: x[1], reverse=True)

        return enhanced_similarities, mean_similarity, std_similarity
    else:
        # No valid results were obtained
        return [], 0, 0


def main():
    """
    Main interactive loop for the sentence similarity matcher.
    
    Provides a command-line interface where users can:
    1. Enter questions interactively
    2. See which sentences are most similar to their question
    3. View similarity scores and statistical context
    4. Continue asking questions or quit
    """
    # Sample corpus of sentences to search through
    # In a real application, this could be documents, FAQs, knowledge base articles, etc.
    sentences = [
        "The cat sat on the mat.",
        "What is the weather like today?",
        "Python is a popular programming language.",
        "How do I write code?",
        "The quick brown fox jumps over the lazy dog.",
        "What's the best way to learn programming?",
        "Machine learning involves training models on data.",
        "Can you help me debug this code?"
    ]

    # Interactive loop
    while True:
        # Get user input
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        print("\nFinding similar sentences...")
        # Find and rank similar sentences
        results, mean_sim, std_sim = find_most_similar_sentence(question, sentences)

        if results:
            # Display statistical summary
            # This gives context for interpreting individual scores
            print(f"\nStatistics:")
            print(f"Mean similarity: {mean_sim:.4f}")
            print(f"Standard deviation: {std_sim:.4f}")
            
            # Display results
            print("\nResults (sorted by similarity):")
            print("=" * 70)

            # Show each sentence with its similarity metrics
            for sentence, similarity, std_devs in results:
                # Display similarity score (0-1 range, higher is better)
                # Display z-score (σ): how many standard deviations from mean
                #   Positive σ = above average similarity
                #   Negative σ = below average similarity
                #   Values > 2σ are typically considered significant
                print(f"Similarity: {similarity:.4f}, {std_devs:+.2f}σ")
                print(f"Sentence: {sentence}")
                print("-" * 70)
        else:
            print("No results found.")


if __name__ == "__main__":
    """
    Entry point for the script.
    
    Prerequisites:
    - Ollama must be installed and running
    - The embedding model (qwen3-embedding:0.6b) must be downloaded
    - Ollama should be accessible at http://localhost:11434
    
    To install Ollama and the model:
    1. Install Ollama from https://ollama.ai
    2. Run: ollama pull qwen3-embedding:0.6b
    3. Ollama service should start automatically
    """
    print("Enhanced Sentence Similarity Matcher")
    print("Make sure Ollama is running locally on port 11434")
    main()
