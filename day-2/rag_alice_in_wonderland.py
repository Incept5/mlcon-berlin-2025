from typing import List, Tuple
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path

# Configuration
EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "gemma3n:e4b"


@dataclass
class Document:
    text: str
    chunk_id: int
    embedding: np.ndarray = None


class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"

    def get_embedding(self, text: str, task_type: str = "document") -> np.ndarray:
        """Get embedding from Ollama API with proper EmbeddingGemma prompts."""
        if task_type == "query":
            formatted_text = f"task: search result | query: {text}"
        else:
            formatted_text = f"title: none | text: {text}"

        data = {"model": EMBEDDING_MODEL, "prompt": formatted_text}
        response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=30)
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API."""
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 16384,
                "temperature": 0.05,
                "top_p": 0.85
            }
        }
        response = requests.post(f"{self.base_url}/chat", json=data, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


class TextChunker:
    def __init__(self, chunk_size: int = 250, overlap_percentage: float = 0.2, min_chunk_ratio: float = 0.4, min_break_ratio: float = 0.75):
        """
        Initialize TextChunker with configurable parameters.

        Args:
            chunk_size: Target chunk size in words (default 250, ~1000 characters)
            overlap_percentage: Overlap between chunks as ratio (default 0.2 = 20%)
            min_chunk_ratio: Minimum chunk size as ratio of target (default 0.4 = 40%)
            min_break_ratio: Minimum words before seeking break point (default 0.75 = 75%)
        """
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)
        self.min_break_size = int(chunk_size * min_break_ratio)
        self.final_chunk_min_size = int(chunk_size * min_chunk_ratio)

        # Calculate overlap in words
        self.overlap_words = int(chunk_size * overlap_percentage)

    def get_config_info(self) -> str:
        """Return a string describing the current chunking configuration."""
        return (f"Chunking: {self.chunk_size} words, {self.overlap_percentage:.0%} overlap")

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting - overlap handles boundary issues."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)

        # Filter short fragments and clean
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def chunk_text(self, text: str) -> List[Document]:
        """Split text into overlapping chunks with better boundary detection."""
        # Clean text more thoroughly
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Use improved sentence splitting
        sentences = self._split_sentences(text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            # Check if adding this sentence would exceed chunk size
            if (current_words + sentence_words > self.chunk_size and
                current_words >= self.min_chunk_size):

                # Try to find a better breaking point
                chunk_text = ' '.join(current_chunk)

                # Look for paragraph breaks in the last few sentences
                best_break = len(current_chunk)
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1):
                    if ('\n' in current_chunk[j] or
                        current_chunk[j].strip().endswith(('!', '?', '."', '.\'')) and
                        len(' '.join(current_chunk[:j+1]).split()) >= self.min_break_size):
                        best_break = j + 1
                        break

                # Create chunk with better boundary
                final_chunk = current_chunk[:best_break]
                chunk_text = ' '.join(final_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
                chunk_id += 1

                # Create substantial overlap using configured percentage
                target_overlap_words = self.overlap_words
                overlap_sentences = []
                overlap_words = 0

                # Work backwards from the break point to get configured overlap
                for j in range(best_break - 1, -1, -1):
                    sentence_words = len(current_chunk[j].split())
                    if overlap_words + sentence_words <= target_overlap_words:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += sentence_words
                    else:
                        break

                # Ensure we have at least one sentence of overlap if possible
                if not overlap_sentences and best_break > 0:
                    overlap_sentences = [current_chunk[best_break - 1]]

                current_chunk = overlap_sentences
                current_words = overlap_words

            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1

        # Handle final chunk
        if current_chunk:
            if current_words >= self.final_chunk_min_size:  # Create if substantial
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
            elif chunks:  # Merge with last chunk if too small
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:  # Create anyway if it's the only content
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))

        return chunks


class GenericRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None):
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.ollama = OllamaClient()
        self.documents: List[Document] = []
        self._load_document()


    def _clean_text(self, text: str) -> str:
        """Generic text cleaning."""
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove table of contents by finding actual story start
        # Look for the pattern: "CHAPTER I." followed by a title, then start content
        if 'Contents' in text:
            # Find "CHAPTER I." followed by the actual chapter title
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                # Start from the end of this match (after the title and blank line)
                text = text[match.end():]

        return text.strip()

    def _load_document(self):
        """Load and process the document."""
        print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean text
        text = self._clean_text(text)

        # Create chunks
        chunks = self.chunker.chunk_text(text)

        # Generate embeddings
        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.ollama.get_embedding(chunk.text, "document")
                self.documents.append(chunk)
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Embedded {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to embed chunk {i}: {e}")

        print(f"Ready! {len(self.documents)} chunks loaded.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve most relevant chunks using pure semantic similarity."""
        # Get query embedding
        query_embedding = self.ollama.get_embedding(query, "query")

        # Calculate similarities for all documents
        similarities = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, score))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        """Display detailed information about the quality of matches."""
        if not matches:
            return

        # Calculate statistics against ALL chunks for proper significance
        query_embedding = self.ollama.get_embedding(query, "query")
        all_scores = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            all_scores.append(score)

        # Calculate proper statistics
        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            # Calculate significance in standard deviations from global mean
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0

            # Preview of the chunk text
            preview = doc.text[:80].replace('\n', ' ') + "..."

            print(f"  {i}. Chunk: {doc.chunk_id:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        """Answer a question using RAG."""
        if not question.strip():
            return "Please provide a question."

        # Retrieve relevant chunks
        relevant_docs = self._retrieve_chunks(question)

        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."

        # Show match details if requested
        if show_matches:
            self._show_match_details(relevant_docs, question)

        # Build context
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            context_parts.append(f"Context {i}:\n{doc.text}")

        context = "\n\n".join(context_parts)

        # Create simple, universal prompt
        prompt = f"""Answer the question using the provided context passages.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:"""

        # Generate response
        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    """Demo the generic RAG system with configurable chunking."""
    print("Generic RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize RAG system with default settings
        rag = GenericRAG("data/alice_in_wonderland.txt")
        print()

        # Demo questions - specific, detail-oriented questions that demonstrate RAG retrieval
        # Includes multilingual examples to demonstrate universal language support
        questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat when Alice first met him?",
            "What happens when Alice falls down the rabbit hole?",
            "What did the White Rabbit say when Alice first saw him?",
            "Was stand auf der Flasche, die Alice schrumpfen ließ?",  # What was written on the bottle that made Alice shrink?
            "Que faisait Alice au début de l'histoire?",  # What was Alice doing at the beginning of the story?
            "¿Qué dijo el Conejo Blanco cuando Alice lo vio por primera vez?"  # What did the White Rabbit say when Alice first saw him?
        ]

        for question in questions:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)

            answer = rag.query(question, show_matches=True)
            print(f"\nA: {answer}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()