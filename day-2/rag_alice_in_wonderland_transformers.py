from typing import List, Dict, Tuple
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


@dataclass
class Document:
    text: str
    metadata: Dict
    embedding: np.ndarray = None


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerEmbedder:
    # def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using Sentence Transformers."""
        # Tokenize sentences
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling and normalize
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings[0].numpy()


class OllamaLLM:
    def __init__(self, model: str = "gemma3n:e4b"):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API."""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"


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
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))
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
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))
            elif chunks:  # Merge with last chunk if too small
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:  # Create anyway if it's the only content
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))

        return chunks


class SimpleRetriever:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents: List[Document]):
        self.documents = [doc for doc in documents if doc.embedding is not None]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TransformerRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None):
        self.chunker = chunker if chunker else TextChunker()
        self.embedder = TransformerEmbedder()
        self.llm = OllamaLLM()
        self.retriever = SimpleRetriever()

        # Read and process the text file
        self.file_path = Path(file_path)
        self.documents = self._load_and_process_text()
        self.retriever.add_documents(self.documents)

    def _load_and_process_text(self) -> List[Document]:
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

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
                    chunk.embedding = self.embedder.get_embedding(chunk.text)
                    if (i + 1) % 25 == 0 or i == len(chunks) - 1:
                        print(f"  Embedded {i + 1}/{len(chunks)} chunks")
                except Exception as e:
                    print(f"Failed to embed chunk {i}: {e}")
                    chunk.embedding = None

            print(f"Ready! {len([c for c in chunks if c.embedding is not None])} chunks loaded.")
            return chunks

        except Exception as e:
            raise Exception(f"Error processing file {self.file_path}: {str(e)}")

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

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        """Display detailed information about the quality of matches."""
        if not matches:
            return

        # Calculate statistics against ALL chunks for proper significance
        query_embedding = self.embedder.get_embedding(query)
        all_scores = []
        for doc in self.documents:
            if doc.embedding is not None:
                score = self.retriever.cosine_similarity(query_embedding, doc.embedding)
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

            print(f"  {i}. Chunk: {doc.metadata['chunk_id']:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        """Answer a question using RAG."""
        if not question.strip():
            return "Please provide a question."

        try:
            # Generate embedding for the query
            query_embedding = self.embedder.get_embedding(question)

            # Get relevant chunks with similarity scores
            relevant_docs = self.retriever.get_relevant_chunks(query_embedding)

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
            answer = self.llm.generate_response(prompt)
            return answer.strip()

        except Exception as e:
            return f"Error generating response: {e}"


def main():
    """Demo the transformer RAG system with configurable chunking."""
    print("Transformer RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize RAG system with default settings
        rag = TransformerRAG("data/alice_in_wonderland.txt")
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