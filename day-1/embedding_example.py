import numpy as np
import requests


def get_ollama_embedding(text: str) -> list:
    """Get embedding from Ollama API."""
    data = {
        # "model": "all-minilm:33m-l12-v2-fp16",
        "model": "qwen3-embedding:0.6b",
        "prompt": text
    }
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json=data)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_sentence(question: str, sentences: list) -> list:
    """Find the most similar sentences to the question."""
    # Get embedding for the question
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        return []

    # Calculate similarities with all sentences
    similarities = []
    similarity_scores = []

    for sentence in sentences:
        sentence_embedding = get_ollama_embedding(sentence)
        if sentence_embedding is not None:
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))
            similarity_scores.append(similarity)

    # Calculate statistics for standard deviation analysis
    if similarity_scores:
        mean_similarity = np.mean(similarity_scores)
        std_similarity = np.std(similarity_scores)

        # Add standard deviation information to each result
        enhanced_similarities = []
        for sentence, similarity in similarities:
            if std_similarity > 0:
                std_devs_from_mean = (similarity - mean_similarity) / std_similarity
            else:
                std_devs_from_mean = 0  # All similarities are the same
            enhanced_similarities.append((sentence, similarity, std_devs_from_mean))

        # Sort by similarity score in descending order
        enhanced_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return results with statistics
        return enhanced_similarities, mean_similarity, std_similarity
    else:
        return [], 0, 0


def main():
    # Sample sentences to match against
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

    while True:
        # Get question from user
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        print("\nFinding similar sentences...")
        results, mean_sim, std_sim = find_most_similar_sentence(question, sentences)

        if results:
            print(f"\nStatistics:")
            print(f"Mean similarity: {mean_sim:.4f}")
            print(f"Standard deviation: {std_sim:.4f}")
            print("\nResults (sorted by similarity):")
            print("=" * 70)

            for sentence, similarity, std_devs in results:
                print(f"Similarity: {similarity:.4f}, {std_devs:+.2f}σ")
                print(f"Sentence: {sentence}")
                print("-" * 70)
        else:
            print("No results found.")


if __name__ == "__main__":
    print("Enhanced Sentence Similarity Matcher")
    print("Make sure Ollama is running locally on port 11434")
    main()