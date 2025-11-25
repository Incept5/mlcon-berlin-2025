"""Basic RAG (Retrieval-Augmented Generation) implementation using Ollama.

This script demonstrates a simple but complete RAG system that:
1. Chunks text into overlapping segments for better context preservation
2. Generates embeddings using Ollama's EmbeddingGemma model
3. Stores embeddings in memory (no persistence)
4. Retrieves relevant chunks using cosine similarity
5. Generates answers using an LLM with retrieved context

Key Features:
- In-memory storage only (embeddings regenerated each run)
- Configurable text chunking with overlap
- Task-specific prompts for EmbeddingGemma (document vs query)
- Multilingual support through universal embeddings
- Statistical significance analysis for match quality
"""

from typing import List, Tuple
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path

# Configuration - Models for embedding generation and text generation
EMBEDDING_MODEL = "embeddinggemma"  # Used to convert text to vector representations
LLM_MODEL = "gemma3n:e4b"  # Used to generate natural language answers


@dataclass
class Document:
    """Represents a text chunk with its embedding.
    
    Attributes:
        text: The actual text content of the chunk
        chunk_id: Unique identifier for this chunk
        embedding: Vector representation of the text (numpy array)
    """
    text: str
    chunk_id: int
    embedding: np.ndarray = None


class OllamaClient:
    """Client for interacting with local Ollama server.
    
    Handles both embedding generation and LLM text generation.
    Requires Ollama to be running on localhost:11434.
    """
    
    def __init__(self):
        self.base_url = "http://localhost:11434/api"

    def get_embedding(self, text: str, task_type: str = "document") -> np.ndarray:
        """Get embedding from Ollama API with proper EmbeddingGemma prompts.
        
        EmbeddingGemma uses task-specific prompts for better performance:
        - 'query': For search queries (used at query time)
        - 'document': For document chunks (used during indexing)
        
        Args:
            text: Text to embed
            task_type: Either 'query' or 'document'
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Format text with task-specific prompt for EmbeddingGemma
        if task_type == "query":
            formatted_text = f"task: search result | query: {text}"
        else:  # document
            formatted_text = f"title: none | text: {text}"

        data = {"model": EMBEDDING_MODEL, "prompt": formatted_text}
        response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=30)
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API.
        
        Args:
            prompt: The complete prompt including context and question
            
        Returns:
            Generated text response from the LLM
        """
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Get complete response at once
            "options": {
                "num_ctx": 16384,  # Large context window for multiple chunks
                "temperature": 0.05,  # Low temperature for factual, consistent answers
                "top_p": 0.85  # Nucleus sampling for quality
            }
        }
        response = requests.post(f"{self.base_url}/chat", json=data, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


class TextChunker:
    """Intelligent text chunking with overlap and boundary detection.
    
    Creates overlapping chunks to preserve context across boundaries.
    Uses sentence boundaries and paragraph breaks for natural splitting.
    """
    
    def __init__(self, chunk_size: int = 250, overlap_percentage: float = 0.2, min_chunk_ratio: float = 0.4, min_break_ratio: float = 0.75):
        """
        Initialize TextChunker with configurable parameters.

        Args:
            chunk_size: Target chunk size in words (default 250, ~1000 characters)
            overlap_percentage: Overlap between chunks as ratio (default 0.2 = 20%)
                               This ensures context continuity across chunk boundaries
            min_chunk_ratio: Minimum chunk size as ratio of target (default 0.4 = 40%)
                           Prevents creating chunks that are too small
            min_break_ratio: Minimum words before seeking break point (default 0.75 = 75%)
                           Ensures we don't break too early when looking for boundaries
        """
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)  # e.g., 100 words minimum
        self.min_break_size = int(chunk_size * min_break_ratio)  # e.g., 187 words before breaking
        self.final_chunk_min_size = int(chunk_size * min_chunk_ratio)  # Handle last chunk

        # Calculate overlap in words (e.g., 50 words for 20% of 250)
        self.overlap_words = int(chunk_size * overlap_percentage)

    def get_config_info(self) -> str:
        """Return a string describing the current chunking configuration."""
        return (f"Chunking: {self.chunk_size} words, {self.overlap_percentage:.0%} overlap")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns.
        
        Looks for sentence boundaries (., !, ?) followed by whitespace and
        a capital letter or quote. Filters out very short fragments.
        """
        # Split on sentence boundaries: look for punctuation followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)

        # Filter out very short fragments (likely splitting errors) and clean whitespace
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def chunk_text(self, text: str) -> List[Document]:
        """Split text into overlapping chunks with intelligent boundary detection.
        
        Algorithm:
        1. Clean and normalize the text
        2. Split into sentences
        3. Build chunks by adding sentences until target size reached
        4. Find optimal break points (paragraph breaks, sentence ends)
        5. Create overlap by including final sentences of previous chunk
        6. Handle final chunk appropriately (merge if too small)
        
        Returns:
            List of Document objects with text and chunk_id
        """
        # Clean text: normalize line breaks and excessive whitespace
        text = re.sub(r'\r\n', '\n', text)  # Windows to Unix line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple blank lines
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

        # Split into sentences for better chunk boundary detection
        sentences = self._split_sentences(text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            # Check if adding this sentence would exceed target size
            # AND we've reached minimum chunk size
            if (current_words + sentence_words > self.chunk_size and
                current_words >= self.min_chunk_size):

                # Try to find a natural breaking point (paragraph or sentence boundary)
                chunk_text = ' '.join(current_chunk)

                # Look backwards through last 5 sentences for optimal break
                best_break = len(current_chunk)
                # Search backwards for paragraph breaks or strong sentence endings
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1):
                    # Prefer breaking at paragraph boundaries or quoted speech
                    if ('\n' in current_chunk[j] or
                        current_chunk[j].strip().endswith(('!', '?', '."', '.\'')) and
                        len(' '.join(current_chunk[:j+1]).split()) >= self.min_break_size):
                        best_break = j + 1
                        break

                # Create chunk at the optimal boundary
                final_chunk = current_chunk[:best_break]
                chunk_text = ' '.join(final_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
                chunk_id += 1

                # Create overlap: include ending sentences from this chunk in next chunk
                # This preserves context across boundaries
                target_overlap_words = self.overlap_words
                overlap_sentences = []
                overlap_words = 0

                # Work backwards from break point to gather overlap sentences
                for j in range(best_break - 1, -1, -1):
                    sentence_words = len(current_chunk[j].split())
                    if overlap_words + sentence_words <= target_overlap_words:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += sentence_words
                    else:
                        break

                # Ensure we have at least one sentence of overlap if possible
                # This helps maintain context continuity
                if not overlap_sentences and best_break > 0:
                    overlap_sentences = [current_chunk[best_break - 1]]

                # Start next chunk with overlap sentences
                current_chunk = overlap_sentences
                current_words = overlap_words

            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1

        # Handle final chunk: either create, merge, or append as needed
        if current_chunk:
            if current_words >= self.final_chunk_min_size:  # Create if substantial enough
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
            elif chunks:  # Merge with last chunk if too small to stand alone
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:  # Create anyway if it's the only content we have
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))

        return chunks


class GenericRAG:
    """Generic RAG system with in-memory storage.
    
    This is the baseline implementation that:
    - Stores all embeddings in memory (documents list)
    - Regenerates embeddings on each run
    - Uses simple cosine similarity for retrieval
    - Works with any text file
    
    Limitations:
    - No persistence (embeddings lost after program ends)
    - Must recompute embeddings every time
    - All data kept in RAM
    """
    
    def __init__(self, file_path: str, chunker: TextChunker = None):
        """Initialize RAG system and load document.
        
        Args:
            file_path: Path to text file to process
            chunker: Optional TextChunker instance with custom settings
        """
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.ollama = OllamaClient()
        self.documents: List[Document] = []  # In-memory storage of chunks
        self._load_document()  # Load and embed immediately


    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing.
        
        Handles common text issues and removes non-content sections
        like tables of contents.
        """
        # Normalize whitespace and line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Alice in Wonderland specific: Remove table of contents
        # Find the actual story start: "CHAPTER I." followed by title
        if 'Contents' in text:
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                # Start from after the chapter title and blank line
                text = text[match.end():]

        return text.strip()

    def _load_document(self):
        """Load document, chunk it, and generate embeddings.
        
        This runs immediately on initialization. All embeddings are
        generated and stored in memory.
        """
        print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

        # Read the entire file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean and normalize text
        text = self._clean_text(text)

        # Split into overlapping chunks
        chunks = self.chunker.chunk_text(text)

        # Generate embeddings for each chunk (this takes time!)
        # Using "document" task type for proper EmbeddingGemma formatting
        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.ollama.get_embedding(chunk.text, "document")
                self.documents.append(chunk)
                # Progress feedback every 10 chunks
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Embedded {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to embed chunk {i}: {e}")

        print(f"Ready! {len(self.documents)} chunks loaded.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embedding vectors.
        
        Cosine similarity measures the angle between vectors, ranging from
        -1 (opposite) to 1 (identical). Values closer to 1 indicate more
        semantically similar text.
        
        Formula: (a · b) / (||a|| * ||b||)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve most relevant chunks using semantic similarity.
        
        Process:
        1. Convert query to embedding vector (using 'query' task type)
        2. Calculate cosine similarity with all document embeddings
        3. Sort by similarity score
        4. Return top_k results
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (default 10)
            
        Returns:
            List of (Document, similarity_score) tuples, sorted by score
        """
        # Get query embedding using 'query' task type for EmbeddingGemma
        query_embedding = self.ollama.get_embedding(query, "query")

        # Calculate cosine similarity with every document chunk
        # This is the core retrieval step
        similarities = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, score))

        # Sort by similarity score (highest first) and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        """Display detailed information about match quality with statistical analysis.
        
        Shows:
        - Chunk ID and similarity score
        - Statistical significance (standard deviations from mean)
        - Text preview
        
        The significance score shows how unusual/strong the match is compared
        to all chunks. Higher values indicate more relevant matches.
        """
        if not matches:
            return

        # Calculate how each chunk scores against this query
        # This gives us context for significance analysis
        query_embedding = self.ollama.get_embedding(query, "query")
        all_scores = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            all_scores.append(score)

        # Calculate mean and standard deviation of all scores
        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            # How many standard deviations above mean?
            # Higher = more unusually relevant
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0

            # Show first 80 characters of chunk
            preview = doc.text[:80].replace('\n', ' ') + "..."

            print(f"  {i}. Chunk: {doc.chunk_id:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        """Answer a question using RAG (Retrieval-Augmented Generation).
        
        Process:
        1. Retrieve relevant chunks based on semantic similarity
        2. Show match details if requested (for debugging/analysis)
        3. Build context from retrieved chunks
        4. Create prompt with question and context
        5. Generate answer using LLM
        
        Args:
            question: User's question
            show_matches: Whether to display match details (default False)
            
        Returns:
            Generated answer or error message
        """
        if not question.strip():
            return "Please provide a question."

        # Step 1: Retrieve the most relevant chunks
        relevant_docs = self._retrieve_chunks(question)

        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."

        # Step 2: Optionally display match quality information
        if show_matches:
            self._show_match_details(relevant_docs, question)

        # Step 3: Build context string from retrieved chunks
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            context_parts.append(f"Context {i}:\n{doc.text}")

        context = "\n\n".join(context_parts)

        # Step 4: Create prompt with instruction, question, and context
        # This is a simple but effective prompt structure
        prompt = f"""Answer the question using the provided context passages.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:"""

        # Step 5: Generate answer using the LLM
        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    """Demo the generic RAG system with configurable chunking.
    
    Shows:
    - Loading and embedding a document
    - Querying with English questions
    - Multilingual query support (German, French, Spanish)
    - Match quality visualization
    """
    print("Generic RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize RAG system with default chunking settings
        # This will load, chunk, and embed the entire document
        rag = GenericRAG("data/alice_in_wonderland.txt")
        print()

        # Demo questions - mix of English and multilingual queries
        # Shows that embeddings work across languages
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