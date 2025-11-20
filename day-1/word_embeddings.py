from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'notebook'

EMBEDDINGS_FILE = "embeddings.pkl"

def load_dictionary(word_file):
    words = []
    with open(word_file, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words

class EmbeddingWithId:
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding

def generate_and_store_embeddings(all_words):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(all_words)

    embeddings_with_ids = []
    for word, embedding in zip(all_words, embeddings):
        embedding_obj = EmbeddingWithId(word, embedding)
        embeddings_with_ids.append(embedding_obj)

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_with_ids, f)

    return embeddings_with_ids

def query_chroma(embeddings_with_ids, target_words, n_neighbors=20):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    target_embeddings = model.encode(target_words)

    all_similarities = []
    for item in embeddings_with_ids:  # Access the object
        word_similarities = np.dot(item.embedding, target_embeddings.T)
        average_similarity = np.mean(word_similarities)
        all_similarities.append((item.id, average_similarity))

    # Sort by average similarity (descending order)
    all_similarities.sort(key=lambda item: item[1], reverse=True)

    return all_similarities[:n_neighbors]

def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

if __name__ == '__main__':
    word_file = '/usr/share/dict/words'
    all_words = load_dictionary(word_file)

    model_name = 'all-MiniLM-L6-v2'

    embeddings_with_ids = load_embeddings()
    if embeddings_with_ids is None:
        embeddings_with_ids = generate_and_store_embeddings(all_words)

    # Finding Similar Words
    # target_words = ['red', 'green', 'blue', 'man', 'woman', 'dog', 'cat', 'tea', 'coffee', 'mud', 'dirt']
    target_words = ['tea', 'coffee', 'mud', 'dirt', "pretzel", "grape"]
    similar_words = query_chroma(embeddings_with_ids, target_words, 100)

    print(f"Top 50 similar words to {', '.join(target_words)}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

    # Extract embeddings for target words and their nearest neighbors
    target_embeddings = [embedding for embedding in embeddings_with_ids if embedding.id in target_words]
    neighbor_embeddings = [embedding for embedding in embeddings_with_ids if
                           embedding.id in [word[0] for word in similar_words]]

    # Combine these for PCA
    combined_embeddings = np.array([embedding.embedding for embedding in target_embeddings + neighbor_embeddings])

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Create a Plotly figure
    fig = go.Figure()
    neighbor_indices = range(len(target_words), len(reduced_embeddings))
    target_indices = range(len(target_words))

    # Add target words to the plot
    fig.add_trace(go.Scatter3d(x=reduced_embeddings[target_indices, 0],
                               y=reduced_embeddings[target_indices, 1],
                               z=reduced_embeddings[target_indices, 2],
                               mode='markers+text',
                               marker=dict(color='red', size=10),
                               text=[embedding.id for embedding in target_embeddings],
                               name='Target Words',
                                textposition = 'bottom center',  # Position the text below the markers
                                textfont = dict(
                                size=18,  # Set the font size of the text
                                color='black'  # You can also specify the text color here if needed
                                )))

    # Add nearest neighbors to the plot
    fig.add_trace(go.Scatter3d(x=reduced_embeddings[neighbor_indices, 0],
                               y=reduced_embeddings[neighbor_indices, 1],
                               z=reduced_embeddings[neighbor_indices, 2],
                               mode='markers+text',
                               marker=dict(color='blue', size=5),
                               text=[embedding.id for embedding in neighbor_embeddings],
                               name='Nearest Neighbors'))

    # Update layout for better visualization
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis_title='PCA1',
                                 yaxis_title='PCA2',
                                 zaxis_title='PCA3'))

    html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')

    file_path = '3d_plot.html'
    with open(file_path, 'w') as f:
        f.write(html_string)

    fig.show(renderer="browser")
