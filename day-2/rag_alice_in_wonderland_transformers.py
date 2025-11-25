"""RAG implementation using HuggingFace Transformers for embeddings.

BUILDS ON: rag_alice_in_wonderland.py
COMPARES TO: rag_alice_in_wonderland_chromadb.py

KEY DIFFERENCE: Embedding Model Source
- This version uses HuggingFace Transformers instead of Ollama for embeddings
- Uses sentence-transformers models (runs locally with PyTorch)
- Still uses Ollama for the LLM (text generation)
- No persistent storage (like base version, not ChromaDB version)

What's different from base version:
- TransformerEmbedder class replaces OllamaClient.get_embedding()
- Uses sentence-transformers models (all-MiniLM-L6-v2)
- Different embedding process: tokenize → encode → pool → normalize
- OllamaLLM class only handles text generation (not embeddings)
- Slightly different Document dataclass (uses metadata dict)
- SimpleRetriever class for clean separation of concerns

What's the same:
- Same TextChunker for text splitting
- Same retrieval logic (cosine similarity)
- Same query processing and answer generation
- In-memory storage (no persistence)

When to use this version:
- Want more control over embedding model
- Need offline embeddings (no Ollama required for embeddings)
- Experimenting with different transformer models
- Better embedding quality for specific use cases

Trade-offs:
- Requires PyTorch and transformers dependencies
- Different embedding space than Ollama models
- No task-specific prompts (like EmbeddingGemma has)
- Potentially different performance characteristics
"""

from typing import List, Dict, Tuple
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
import requests
from transformers import AutoTokenizer, AutoModel  # NEW: HuggingFace for embeddings
import torch  # NEW: PyTorch for model inference
import torch.nn.functional as F  # NEW: For normalization


@dataclass
class Document:
    """Represents a text chunk with its embedding.
    
    MODIFIED: Uses metadata dict instead of chunk_id directly.
    This provides more flexibility for storing additional metadata.
    
    Attributes:
        text: The actual text content
        metadata: Dict containing chunk_id and potentially other info
        embedding: Vector representation of the text
    """
    text: str
    metadata: Dict  # Changed from chunk_id: int
    embedding: np.ndarray = None


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Average token embeddings with attention mask.
    
    NEW: Helper function for sentence-transformers models.
    
    Transformer models output embeddings for each token. We need to
    combine these into a single sentence embedding. Mean pooling
    averages all token embeddings, weighted by the attention mask
    (so padding tokens don't affect the result).
    
    Args:
        model_output: Raw transformer output (contains token embeddings)
        attention_mask: Mask indicating which tokens are real vs padding
        
    Returns:
        Sentence-level embedding vector
    """
    # Extract token embeddings from model output
    token_embeddings = model_output[0]  # Shape: [batch_size, seq_len, hidden_dim]
    
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings weighted by mask, then divide by sum of mask (mean)
    # Clamp prevents division by zero
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerEmbedder:
    """NEW CLASS: Generate embeddings using HuggingFace Transformers.
    
    REPLACES: OllamaClient.get_embedding() from base version
    
    Uses sentence-transformers models designed for semantic similarity.
    These are pre-trained models specifically optimized for creating
    meaningful sentence embeddings.
    
    Model options:
    - all-MiniLM-L6-v2: Fast, small, good quality (default)
    - all-mpnet-base-v2: Slower, larger, better quality
    """
    
    # def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize transformer model and tokenizer.
        
        Downloads model on first use, then caches locally.
        """
        # Load tokenizer (converts text to token IDs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load transformer model (generates embeddings)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using transformer model.
        
        Process:
        1. Tokenize text (convert to token IDs)
        2. Run through transformer model
        3. Pool token embeddings into sentence embedding
        4. Normalize to unit vector (for cosine similarity)
        
        NOTE: No task_type parameter like Ollama version.
        Sentence-transformers use the same process for both
        documents and queries.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector as numpy array
        """
        # Step 1: Tokenize text into model inputs
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')

        # Step 2: Run through transformer model (no gradient needed for inference)
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Step 3: Pool token embeddings into single sentence embedding
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Step 4: Normalize to unit length (makes cosine similarity = dot product)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert to numpy and return
        return sentence_embeddings[0].numpy()


class OllamaLLM:
    """MODIFIED: Simplified Ollama client for LLM only.
    
    DIFFERENCE: Only handles text generation, not embeddings.
    The embedding functionality moved to TransformerEmbedder.
    
    Uses Ollama's /generate endpoint (simpler than /chat).
    """
    
    def __init__(self, model: str = "qwen3-vl:4b-instruct"):
        """Initialize LLM client.
        
        Args:
            model: Ollama model name for text generation
        """
        self.model = model
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama.
        
        DIFFERENT: Uses /generate endpoint instead of /chat.
        Simpler interface but same functionality.
        """
        data = {
            "model": self.model,
            "prompt": prompt,  # Direct prompt string (not message format)
            "stream": False,
            "options": {
                "num_ctx": 8192,  # Smaller context than base (still sufficient)
                "temperature": 0.3  # Slightly higher than base for variety
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
    """NEW CLASS: Simple in-memory retriever.
    
    ARCHITECTURE: Separates retrieval logic from main RAG class.
    This is cleaner design compared to base version where
    retrieval methods are mixed into GenericRAG class.
    
    Benefits of separation:
    - Could swap out retriever implementations
    - Clearer responsibilities
    - Easier to test
    """
    
    def __init__(self):
        self.documents = []  # In-memory storage (no persistence)

    def add_documents(self, documents: List[Document]):
        """Store documents with valid embeddings.
        
        Args:
            documents: List of Document objects
        """
        # Filter out any documents without embeddings
        self.documents = [doc for doc in documents if doc.embedding is not None]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors.
        
        Same formula as base version.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Find most similar documents to query.
        
        Same algorithm as base version, just in separate class.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        # Calculate similarity with all documents
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))

        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TransformerRAG:
    """RAG system using HuggingFace Transformers for embeddings.
    
    ARCHITECTURE: Separates concerns into specialized components:
    - TransformerEmbedder: Handles embeddings
    - OllamaLLM: Handles text generation
    - SimpleRetriever: Handles similarity search
    - TextChunker: Handles text splitting
    
    This is more modular than base version's monolithic OllamaClient.
    """
    
    def __init__(self, file_path: str, chunker: TextChunker = None):
        """Initialize RAG system with transformer-based embeddings.
        
        DIFFERENT COMPONENTS:
        - embedder: TransformerEmbedder (not OllamaClient)
        - llm: OllamaLLM (separate from embedder)
        - retriever: SimpleRetriever (separate from main class)
        """
        self.chunker = chunker if chunker else TextChunker()
        self.embedder = TransformerEmbedder()  # NEW: Transformer instead of Ollama
        self.llm = OllamaLLM()  # MODIFIED: LLM only, no embeddings
        self.retriever = SimpleRetriever()  # NEW: Separate retriever

        # Load and process the text file
        self.file_path = Path(file_path)
        self.documents = self._load_and_process_text()
        self.retriever.add_documents(self.documents)

    def _load_and_process_text(self) -> List[Document]:
        """Load document, chunk it, and generate embeddings.
        
        DIFFERENT: Uses TransformerEmbedder instead of OllamaClient.
        No task_type parameter needed (transformers use same process for all).
        """
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

            # Read file
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Clean and normalize
            text = self._clean_text(text)

            # Split into chunks (same TextChunker as other versions)
            chunks = self.chunker.chunk_text(text)

            # Generate embeddings using transformer model
            for i, chunk in enumerate(chunks):
                try:
                    # DIFFERENT: No task_type parameter
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
            # Calculate significance (same as other versions)
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0

            # Preview text
            preview = doc.text[:80].replace('\n', ' ') + "..."

            # DIFFERENT: Access chunk_id from metadata dict instead of direct attribute
            print(f"  {i}. Chunk: {doc.metadata['chunk_id']:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        """Answer a question using RAG.
        
        PROCESS: Same as other versions, different embedding source.
        
        1. Generate query embedding using TransformerEmbedder
        2. Retrieve similar chunks using SimpleRetriever
        3. Build context from chunks
        4. Generate answer using OllamaLLM
        """
        if not question.strip():
            return "Please provide a question."

        try:
            # Step 1: Generate query embedding using transformer
            query_embedding = self.embedder.get_embedding(question)

            # Step 2: Retrieve similar chunks
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
    """Demo the transformer-based RAG system.
    
    COMPARISON TO OTHER VERSIONS:
    - Uses HuggingFace transformers for embeddings (not Ollama)
    - Still uses Ollama for LLM text generation
    - No persistent storage (like base, unlike ChromaDB version)
    - More modular architecture with separate components
    
    First run will download the transformer model (~80MB for MiniLM).
    """
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