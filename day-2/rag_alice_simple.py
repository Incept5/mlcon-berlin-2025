import numpy as np
import requests
import re

EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "qwen3-vl:4b-instruct"
OLLAMA_URL = "http://localhost:11434/api"


def get_embedding(text, task_type="document"):
    """Get embedding from Ollama."""
    if task_type == "query":
        prompt = f"task: search result | query: {text}"
    else:
        prompt = f"title: none | text: {text}"

    response = requests.post(
        f"{OLLAMA_URL}/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": prompt}
    )
    return np.array(response.json()["embedding"])


def chunk_text(text, chunk_size=250, overlap=50):
    """Split text into overlapping chunks by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = len(sentence.split())

        if word_count + words > chunk_size and word_count > 0:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_text = " ".join(current_chunk)
            overlap_words = overlap_text.split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            word_count = len(overlap_words)

        current_chunk.append(sentence)
        word_count += words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query_rag(question, chunks, embeddings, top_k=6):
    """Query the RAG system."""
    query_emb = get_embedding(question, "query")

    # Find most similar chunks
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    print(f"Q: {question}\n")
    print("Top matches:")
    for i, idx in enumerate(top_indices, 1):
        preview = chunks[idx][:80].replace('\n', ' ') + "..."
        print(f"  {i}. Chunk {idx:03d} | Score: {scores[idx]:.4f} | \"{preview}\"")
    print()

    # Build context from top chunks
    context = "\n\n".join([f"[{i}] {chunks[idx]}" for i, idx in enumerate(top_indices, 1)])

    # Generate answer
    prompt = f"""Answer the question using the context below. Be specific and quote relevant text.

Question: {question}

Context:
{context}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": 16384, "temperature": 0.05}
        }
    )

    answer = response.json()["message"]["content"]
    print(f"A: {answer}\n")
    print("-" * 70)
    return answer


def main():
    print("Simple RAG Demo - Alice in Wonderland\n")

    # Load text
    with open("data/alice_in_wonderland.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean text - remove table of contents
    text = re.sub(r'\r\n', '\n', text)
    match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
    if match:
        text = text[match.end():]

    # Create chunks
    print("Creating chunks...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks\n")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk, "document"))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(chunks)} embedded")
    print(f"Done! {len(chunks)} chunks ready.\n")
    print("-" * 70)

    # Test questions
    questions = [
        "What was Alice doing at the beginning of the story?",
        "What was written on the bottle that made Alice shrink?",
        "Where was the Cheshire Cat when Alice first met him?",
        "What did the White Rabbit say when Alice first saw him?",
    ]

    for question in questions:
        query_rag(question, chunks, embeddings)


if __name__ == "__main__":
    main()
