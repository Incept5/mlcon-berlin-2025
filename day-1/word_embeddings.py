"""
Word Embeddings Similarity Search and Visualization

This script demonstrates semantic similarity search using word embeddings.
It loads a dictionary of words, generates vector embeddings for them using
a pre-trained transformer model, and finds semantically similar words to
a given set of target words. The results are visualized in 3D space using
PCA dimensionality reduction.

Key Concepts:
- Word Embeddings: Dense vector representations of words that capture semantic meaning
- Semantic Similarity: Words with similar meanings have embeddings that are close in vector space
- Cosine Similarity: Measures the angle between vectors (dot product of normalized vectors)
- PCA: Principal Component Analysis reduces high-dimensional embeddings to 3D for visualization
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

# Set Plotly to render in notebook (useful for Jupyter environments)
pio.renderers.default = 'notebook'

# File path for caching generated embeddings to avoid recomputation
EMBEDDINGS_FILE = "embeddings.pkl"


def load_dictionary(word_file):
    """
    Load a list of words from a dictionary file.
    
    Args:
        word_file (str): Path to the dictionary file (one word per line)
        
    Returns:
        list: List of words stripped of whitespace
        
    Note:
        The default path '/usr/share/dict/words' is a standard Unix dictionary
        containing ~235,000 English words
    """
    words = []
    with open(word_file, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


class EmbeddingWithId:
    """
    Container class to store a word and its corresponding embedding vector.
    
    This wrapper associates each embedding with its original word (id) to
    maintain the mapping after vector operations.
    
    Attributes:
        id (str): The original word
        embedding (np.ndarray): The word's vector embedding (384 dimensions for all-MiniLM-L6-v2)
    """
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding


def generate_and_store_embeddings(all_words):
    """
    Generate embeddings for all words using a pre-trained transformer model and cache them.
    
    This uses the 'all-MiniLM-L6-v2' model from sentence-transformers:
    - Fast and efficient (22M parameters)
    - Produces 384-dimensional embeddings
    - Trained on 1B+ sentence pairs for semantic similarity
    
    Args:
        all_words (list): List of words to embed
        
    Returns:
        list: List of EmbeddingWithId objects containing words and their embeddings
        
    Note:
        Embeddings are cached to disk to avoid expensive recomputation.
        For ~235K words, this takes a few minutes on first run.
    """
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Generate embeddings for all words in batch (efficient GPU/CPU utilization)
    embeddings = model.encode(all_words)

    # Create wrapper objects to maintain word-embedding associations
    embeddings_with_ids = []
    for word, embedding in zip(all_words, embeddings):
        embedding_obj = EmbeddingWithId(word, embedding)
        embeddings_with_ids.append(embedding_obj)

    # Cache embeddings to disk using pickle for fast loading on subsequent runs
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_with_ids, f)

    return embeddings_with_ids


def query_chroma(embeddings_with_ids, target_words, n_neighbors=20):
    """
    Find words most semantically similar to the target words using cosine similarity.
    
    Algorithm:
    1. Generate embeddings for target words
    2. Calculate cosine similarity between each dictionary word and all target words
    3. Average the similarities across all target words
    4. Return the top N words with highest average similarity
    
    This approach finds words that are collectively similar to the entire set of
    target words, rather than similar to any single target word.
    
    Args:
        embeddings_with_ids (list): Pre-computed embeddings for all dictionary words
        target_words (list): Words to find similar neighbors for
        n_neighbors (int): Number of most similar words to return
        
    Returns:
        list: Tuples of (word, average_similarity_score) sorted by similarity (descending)
        
    Mathematical Note:
        Cosine similarity = dot(A, B) / (||A|| * ||B||)
        Since sentence-transformers embeddings are normalized, we can use dot product directly.
    """
    # Load model to generate embeddings for target words
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    target_embeddings = model.encode(target_words)

    all_similarities = []
    # Compare each dictionary word against all target words
    for item in embeddings_with_ids:
        # Calculate dot product (cosine similarity for normalized vectors)
        # Shape: (embedding_dim,) @ (embedding_dim, num_targets) = (num_targets,)
        word_similarities = np.dot(item.embedding, target_embeddings.T)
        
        # Average similarity across all target words
        # This finds words that are generally related to the entire concept space
        average_similarity = np.mean(word_similarities)
        all_similarities.append((item.id, average_similarity))

    # Sort by average similarity (highest first)
    all_similarities.sort(key=lambda item: item[1], reverse=True)

    # Return top N most similar words
    return all_similarities[:n_neighbors]


def load_embeddings():
    """
    Load cached embeddings from disk if available.
    
    Returns:
        list: List of EmbeddingWithId objects, or None if cache doesn't exist
        
    Note:
        Using pickle allows us to preserve the custom EmbeddingWithId objects
        and avoid regenerating embeddings on every run (significant time savings).
    """
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    # Load the system dictionary (Unix/Linux standard location)
    word_file = '/usr/share/dict/words'
    all_words = load_dictionary(word_file)

    model_name = 'all-MiniLM-L6-v2'

    # Load cached embeddings or generate if not available
    embeddings_with_ids = load_embeddings()
    if embeddings_with_ids is None:
        print("Generating embeddings for the first time (this may take a few minutes)...")
        embeddings_with_ids = generate_and_store_embeddings(all_words)
        print("Embeddings generated and cached.")

    # ========== SEMANTIC SIMILARITY SEARCH ==========
    
    # Define target words to find semantic neighbors for
    # Example explores drinks and food: tea/coffee are beverages, mud/dirt are earthy substances
    target_words = ['tea', 'coffee', 'mud', 'dirt', "pretzel", "grape"]
    
    # Find 100 most similar words to our target set
    similar_words = query_chroma(embeddings_with_ids, target_words, 100)

    print(f"Top 50 similar words to {', '.join(target_words)}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

    # ========== 3D VISUALIZATION WITH PCA ==========
    
    # Extract embeddings for target words
    target_embeddings = [embedding for embedding in embeddings_with_ids if embedding.id in target_words]
    
    # Extract embeddings for their nearest neighbors (top similar words)
    neighbor_embeddings = [embedding for embedding in embeddings_with_ids if
                           embedding.id in [word[0] for word in similar_words]]

    # Combine target and neighbor embeddings for joint PCA transformation
    combined_embeddings = np.array([embedding.embedding for embedding in target_embeddings + neighbor_embeddings])

    # Apply PCA to reduce from 384 dimensions to 3 for visualization
    # PCA finds the 3 principal components that capture the most variance
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # ========== PLOTLY 3D SCATTER PLOT ==========
    
    # Create interactive 3D plot
    fig = go.Figure()
    
    # Define indices for target words and neighbors in the reduced embedding space
    neighbor_indices = range(len(target_words), len(reduced_embeddings))
    target_indices = range(len(target_words))

    # Plot target words (red, larger markers)
    fig.add_trace(go.Scatter3d(
        x=reduced_embeddings[target_indices, 0],  # First principal component
        y=reduced_embeddings[target_indices, 1],  # Second principal component
        z=reduced_embeddings[target_indices, 2],  # Third principal component
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=[embedding.id for embedding in target_embeddings],
        name='Target Words',
        textposition='bottom center',  # Position labels below markers
        textfont=dict(
            size=18,
            color='black'
        )
    ))

    # Plot nearest neighbors (blue, smaller markers)
    fig.add_trace(go.Scatter3d(
        x=reduced_embeddings[neighbor_indices, 0],
        y=reduced_embeddings[neighbor_indices, 1],
        z=reduced_embeddings[neighbor_indices, 2],
        mode='markers+text',
        marker=dict(color='blue', size=5),
        text=[embedding.id for embedding in neighbor_embeddings],
        name='Nearest Neighbors'
    ))

    # Configure plot layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Minimize margins for better view
        scene=dict(
            xaxis_title='PCA1',  # First principal component
            yaxis_title='PCA2',  # Second principal component
            zaxis_title='PCA3'   # Third principal component
        )
    )

    # Save plot as standalone HTML file
    html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')
    file_path = '3d_plot.html'
    with open(file_path, 'w') as f:
        f.write(html_string)
    print(f"\nVisualization saved to {file_path}")

    # Open plot in browser for interactive exploration
    fig.show(renderer="browser")
