import numpy as np
from tabulate import tabulate
import requests

def get_ollama_embedding(text):
    response = requests.post("http://localhost:11434/api/embeddings",
                             json={"model": "all-minilm", "prompt": text})
    return response.json()["embedding"]

def main():
    words = ["tea", "coffee", "mud", "dirt"]

    # Get embeddings and normalize them
    embeddings = np.array([get_ollama_embedding(word) for word in words])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Calculate similarities using dot product of normalized vectors
    similarities = np.dot(embeddings, embeddings.T)

    # Create similarity table
    header = [""] + words
    table = [[words[i]] + [f"{similarities[i][j]:.3f}"
                           for j in range(len(words))] for i in range(len(words))]

    print(tabulate(table, headers=header, tablefmt="grid"))

if __name__ == "__main__":
    main()