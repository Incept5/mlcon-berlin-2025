"""RAG (Retrieval-Augmented Generation) for Grimm Fairy Tales using Ollama.

This script demonstrates a complete RAG system that:
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
- Handles both English and German Grimm fairy tales texts
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
LLM_MODEL = "qwen3-vl:4b-instruct"  # Used to generate natural language answers


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

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text response from Ollama API.
        
        Args:
            prompt: The user prompt including context and question
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            Generated text response from the LLM
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,  # Get complete response at once
            "options": {
                "num_ctx": 32768,  # Large context window for multiple chunks
                "temperature": 0.7,  # Moderate creativity for natural answers
                "top_p": 0.95,  # Nucleus sampling for quality
                "top_k": 40  # Consider top 40 tokens at each step
            }
        }
        response = requests.post(f"{self.base_url}/chat", json=data, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


class TextChunker:
    """Intelligent text chunking with overlap and boundary detection.
    
    Creates overlapping chunks to preserve context across boundaries.
    For Grimm fairy tales, uses chapter/story boundaries when possible.
    """
    
    def __init__(self, chunk_size: int = 500, overlap_percentage: float = 0.15, min_chunk_ratio: float = 0.3, min_break_ratio: float = 0.7):
        """
        Initialize TextChunker with configurable parameters.

        Args:
            chunk_size: Target chunk size in words (default 500 for longer fairy tale chunks)
            overlap_percentage: Overlap between chunks as ratio (default 0.15 = 15%)
            min_chunk_ratio: Minimum chunk size as ratio of target (default 0.3 = 30%)
            min_break_ratio: Minimum words before seeking break point (default 0.7 = 70%)
        """
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)
        self.min_break_size = int(chunk_size * min_break_ratio)
        self.final_chunk_min_size = int(chunk_size * min_chunk_ratio)
        self.overlap_words = int(chunk_size * overlap_percentage)

    def get_config_info(self) -> str:
        """Return a string describing the current chunking configuration."""
        return (f"Chunking: {self.chunk_size} words, {self.overlap_percentage:.0%} overlap")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns.
        
        Handles both German and English sentence boundaries.
        """
        # Split on sentence boundaries: punctuation followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)

        # Filter out very short fragments and clean whitespace
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def chunk_text(self, text: str) -> List[Document]:
        """Split text into overlapping chunks with intelligent boundary detection.
        
        For Grimm fairy tales, tries to identify story boundaries (quadruple newlines)
        and create chunks that respect these natural divisions.
        
        Returns:
            List of Document objects with text and chunk_id
        """
        # Clean text: normalize line breaks and excessive whitespace
        text = re.sub(r'\r\n', '\n', text)
        
        # Check if text has clear story boundaries (quadruple newlines)
        stories = re.split(r'\n\n\n\n', text)
        
        # If we have clear story boundaries, treat each as a potential chunk
        if len(stories) > 3:  # Multiple stories detected
            chunks = []
            chunk_id = 0
            
            for story in stories:
                # Clean the story text
                cleaned = ' '.join(story.strip().split())
                if len(cleaned) < 100:  # Skip very short fragments
                    continue
                
                story_words = len(cleaned.split())
                
                # If story is small enough, keep as single chunk
                if story_words <= self.chunk_size * 1.5:
                    chunks.append(Document(text=cleaned, chunk_id=chunk_id))
                    chunk_id += 1
                else:
                    # Story too long, split it using sentence-based chunking
                    story_chunks = self._chunk_long_text(cleaned, chunk_id)
                    chunks.extend(story_chunks)
                    chunk_id += len(story_chunks)
            
            return chunks
        
        # No clear boundaries, use standard sentence-based chunking
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return self._chunk_long_text(text, 0)

    def _chunk_long_text(self, text: str, start_id: int) -> List[Document]:
        """Chunk long text using sentence boundaries and overlap."""
        sentences = self._split_sentences(text)
        
        chunks = []
        chunk_id = start_id
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if (current_words + sentence_words > self.chunk_size and
                current_words >= self.min_chunk_size):

                # Find optimal break point
                best_break = len(current_chunk)
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1):
                    if ('\n' in current_chunk[j] or
                        current_chunk[j].strip().endswith(('!', '?', '."', '.\'')) and
                        len(' '.join(current_chunk[:j+1]).split()) >= self.min_break_size):
                        best_break = j + 1
                        break

                # Create chunk
                final_chunk = current_chunk[:best_break]
                chunk_text = ' '.join(final_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
                chunk_id += 1

                # Create overlap
                target_overlap_words = self.overlap_words
                overlap_sentences = []
                overlap_words = 0

                for j in range(best_break - 1, -1, -1):
                    sentence_words = len(current_chunk[j].split())
                    if overlap_words + sentence_words <= target_overlap_words:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += sentence_words
                    else:
                        break

                if not overlap_sentences and best_break > 0:
                    overlap_sentences = [current_chunk[best_break - 1]]

                current_chunk = overlap_sentences
                current_words = overlap_words

            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1

        # Handle final chunk
        if current_chunk:
            if current_words >= self.final_chunk_min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
            elif chunks:
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))

        return chunks


class GrimmRAG:
    """RAG system for Grimm Fairy Tales with in-memory storage.
    
    This implementation:
    - Stores all embeddings in memory (documents list)
    - Regenerates embeddings on each run
    - Uses cosine similarity for retrieval
    - Handles both German and English fairy tale collections
    
    Limitations:
    - No persistence (embeddings lost after program ends)
    - Must recompute embeddings every time
    - All data kept in RAM
    """
    
    def __init__(self, file_path: str, chunker: TextChunker = None):
        """Initialize RAG system and load document.
        
        Args:
            file_path: Path to text file containing Grimm fairy tales
            chunker: Optional TextChunker instance with custom settings
        """
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.ollama = OllamaClient()
        self.documents: List[Document] = []  # In-memory storage of chunks
        self._load_document()  # Load and embed immediately

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing.
        
        Handles common text issues and removes non-content sections.
        """
        # Normalize whitespace and line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove common headers/footers if present
        # Look for Project Gutenberg boilerplate
        if 'Project Gutenberg' in text:
            # Try to find the actual content start
            matches = list(re.finditer(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG', text))
            if matches:
                text = text[matches[0].end():]
            
            # Try to remove end boilerplate
            matches = list(re.finditer(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG', text))
            if matches:
                text = text[:matches[0].start()]

        return text.strip()

    def _load_document(self):
        """Load document, chunk it, and generate embeddings.
        
        This runs immediately on initialization. All embeddings are
        generated and stored in memory.
        """
        print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

        # Read the entire file
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            with open(self.file_path, 'r', encoding='latin-1') as file:
                text = file.read()

        # Clean and normalize text
        text = self._clean_text(text)

        # Split into overlapping chunks
        chunks = self.chunker.chunk_text(text)
        print(f"  Created {len(chunks)} chunks from document")

        # Generate embeddings for each chunk
        # Using "document" task type for proper EmbeddingGemma formatting
        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.ollama.get_embedding(chunk.text, "document")
                self.documents.append(chunk)
                # Progress feedback every 10 chunks
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Embedded {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"  Failed to embed chunk {i}: {e}")

        print(f"Ready! {len(self.documents)} chunks loaded.\n")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embedding vectors.
        
        Cosine similarity measures the angle between vectors, ranging from
        -1 (opposite) to 1 (identical). Values closer to 1 indicate more
        semantically similar text.
        
        Formula: (a · b) / (||a|| * ||b||)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 6) -> List[Tuple[Document, float]]:
        """Retrieve most relevant chunks using semantic similarity.
        
        Process:
        1. Convert query to embedding vector (using 'query' task type)
        2. Calculate cosine similarity with all document embeddings
        3. Sort by similarity score
        4. Return top_k results
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (default 6)
            
        Returns:
            List of (Document, similarity_score) tuples, sorted by score
        """
        # Get query embedding using 'query' task type for EmbeddingGemma
        query_embedding = self.ollama.get_embedding(query, "query")

        # Calculate cosine similarity with every document chunk
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
        """
        if not matches:
            return

        # Calculate statistical significance
        query_embedding = self.ollama.get_embedding(query, "query")
        all_scores = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            all_scores.append(score)

        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0
            preview = doc.text[:100].replace('\n', ' ') + "..."
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
        # Truncate very long chunks to fit in context window
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            text = doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text
            context_parts.append(f"Context {i}:\n{text}")

        context = "\n\n".join(context_parts)

        # Step 4: Create system prompt and user prompt
        system_prompt = """You are a helpful assistant that answers questions about Grimm fairy tales. 
Use the provided context from the fairy tales to answer the user's question accurately and concisely.
Provide your answer in English, even though the source text may be in German.
Keep your response short and focused on directly answering the question."""

        user_prompt = f"""Based on the following context from Grimm fairy tales, please answer this question: {question}

Context:
{context}

Please provide a short, direct answer in English."""

        # Step 5: Generate answer using the LLM
        try:
            answer = self.ollama.generate_response(user_prompt, system_prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    """Demo the Grimm fairy tales RAG system.
    
    Shows:
    - Loading and embedding a fairy tale collection
    - Querying with English questions
    - Multilingual query support (German)
    - Match quality visualization
    """
    print("Grimm Fairy Tales RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize RAG system with fairy tales
        # Try German version first, fall back to English
        fairy_tale_file = None
        for filename in ["Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt", "Grimms-Fairy-Tales.txt"]:
            if Path(filename).exists():
                fairy_tale_file = filename
                break
        
        if not fairy_tale_file:
            print("Error: Could not find Grimm fairy tales file!")
            print("Looking for: Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt or Grimms-Fairy-Tales.txt")
            return
        
        # Use larger chunks for fairy tales (complete stories when possible)
        chunker = TextChunker(chunk_size=500, overlap_percentage=0.15)
        rag = GrimmRAG(fairy_tale_file, chunker)

        # Demo questions - mix of English and German queries
        questions = [
            "What did the frog king promise the princess in exchange for her golden ball?",
            "What happened to Hansel and Gretel in the forest?",
            "What did Little Red Riding Hood's mother tell her to do?",
            "Who helped Cinderella go to the ball?",
            "What happened to the wolf in the story of the seven little goats?",
            "Was versprach der Froschkönig der Prinzessin für ihre goldene Kugel?",  # German: frog king question
            "Was geschah mit Hänsel und Gretel im Wald?"  # German: Hansel and Gretel question
        ]

        for question in questions:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)

            answer = rag.query(question, show_matches=True)
            print(f"\nA: {answer}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
