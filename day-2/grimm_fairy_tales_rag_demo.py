#!/usr/bin/env python3
"""
Simple Grimm Fairy Tales RAG Search Demo

A minimal RAG (Retrieval-Augmented Generation) system that demonstrates:
1. Document chunking and embedding generation
2. Semantic search using vector similarity
3. Answer generation using retrieved context

The system uses:
- Qwen3-Embedding-0.6B for converting text to semantic vectors
- Cosine similarity for finding relevant document chunks
- Qwen3-VL via Ollama for generating natural language answers

This demonstrates a complete RAG pipeline: index → search → retrieve → generate
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import ollama
from ollama import Options
from rich.console import Console
import tiktoken


# Configuration
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'  # Lightweight embedding model for semantic encoding
EMBEDDING_DIM = 512  # Dimensionality reduction from full model size for efficiency
BATCH_SIZE = 64  # Number of chunks to process simultaneously for embeddings
MAX_TOKEN_LENGTH = 512  # Maximum input length for the embedding model
TOP_K = 6  # Number of most relevant chunks to retrieve for each query


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Extract the last token embeddings efficiently.
    
    This pooling strategy uses the last meaningful token's embedding as the
    representation for the entire text sequence. This is particularly effective
    for decoder-style models and ensures we capture the full context.
    
    Args:
        last_hidden_states: Model output tensor [batch_size, seq_len, hidden_dim]
        attention_mask: Mask indicating valid tokens [batch_size, seq_len]
    
    Returns:
        Pooled embeddings [batch_size, hidden_dim]
    """
    # Check if using left padding (all sequences end at the same position)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        # Simple case: just take the last position for all sequences
        return last_hidden_states[:, -1]
    else:
        # Right padding: need to find the actual last token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 to get last valid index
        batch_size = last_hidden_states.shape[0]
        # Gather the last token embedding for each sequence in the batch
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embeddings(texts: List[str], model, tokenizer, device: str) -> torch.Tensor:
    """
    Generate normalized embedding vectors for a batch of texts.
    
    This function:
    1. Appends end-of-text tokens for proper model processing
    2. Tokenizes and batches the input texts
    3. Runs the embedding model
    4. Pools the output to get single vectors per text
    5. Applies dimension reduction if configured
    6. Normalizes vectors for cosine similarity
    
    Args:
        texts: List of text strings to embed
        model: The embedding model
        tokenizer: Corresponding tokenizer
        device: Device to run computation on (cpu/cuda/mps)
    
    Returns:
        Normalized embedding tensor [batch_size, embedding_dim]
    """
    # Add end-of-text token to signal completion to the model
    texts_with_eot = [text + "<|endoftext|>" for text in texts]
    
    # Tokenize: convert text to model input format with padding/truncation
    batch_dict = tokenizer(
        texts_with_eot, 
        padding=True,  # Pad shorter sequences to match longest in batch
        truncation=True,  # Truncate sequences longer than max_length
        max_length=MAX_TOKEN_LENGTH,
        return_tensors="pt"  # Return PyTorch tensors
    ).to(device)
    
    # Run model inference without computing gradients (faster, less memory)
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    # Pool the model output to get single vector per text
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    # Apply dimension reduction if configured (reduces memory and computation)
    if EMBEDDING_DIM and EMBEDDING_DIM < embeddings.shape[1]:
        embeddings = embeddings[:, :EMBEDDING_DIM]  # Keep only first N dimensions
    
    # Normalize to unit length for cosine similarity (dot product = cosine similarity)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """
    Split document into semantic chunks (chapters) for RAG retrieval.
    
    The Grimm fairy tales text uses quadruple newlines to separate stories.
    Each chunk is cleaned and tracked with metadata including token count.
    
    Token counting is important for:
    - Tracking context window usage
    - Ensuring chunks fit in model limits
    - Cost estimation for API-based models
    
    Args:
        text: Full document text
    
    Returns:
        List of chunk dictionaries with text, id, length, and token count
    """
    # Split on quadruple newlines (chapter boundaries in this document)
    chapters = re.split(r'\n\n\n\n', text)
    
    # Initialize tiktoken encoder for accurate token counting
    # cl100k_base is the encoding used by GPT-4 and many modern models
    encoder = tiktoken.get_encoding("cl100k_base")
    
    chunks = []
    for i, chapter in enumerate(chapters):
        # Normalize whitespace: collapse multiple spaces/newlines into single spaces
        cleaned = ' '.join(chapter.strip().split())
        if len(cleaned) > 100:  # Filter out very short chunks (likely headers or artifacts)
            # Count tokens for this chunk
            token_count = len(encoder.encode(cleaned))
            chunks.append({
                'id': i,
                'text': cleaned,
                'length': len(cleaned),  # Character count (kept for compatibility)
                'tokens': token_count  # Actual token count for LLM context tracking
            })
    
    return chunks


def search_chunks(query: str, chunks: List[Dict[str, Any]], chunk_embeddings: torch.Tensor, 
                 model, tokenizer, device: str) -> List[Dict[str, Any]]:
    """
    Perform semantic search to find most relevant chunks for a query.
    
    This implements vector similarity search:
    1. Embed the query using the same model as chunks
    2. Compute cosine similarity (dot product of normalized vectors)
    3. Return top-K most similar chunks
    
    Args:
        query: Question or search string
        chunks: Original chunk data
        chunk_embeddings: Pre-computed embeddings for all chunks
        model: Embedding model
        tokenizer: Tokenizer
        device: Computation device
    
    Returns:
        List of top-K chunks with similarity scores and metadata
    """
    
    # Embed the query using the same model/process as document chunks
    query_embeddings = get_embeddings([query], model, tokenizer, device)
    query_embedding = query_embeddings[0:1]  # Keep as 2D tensor for matrix ops
    
    # Compute cosine similarities between query and all chunks
    # Since vectors are normalized, dot product = cosine similarity
    with torch.no_grad():
        similarities = chunk_embeddings @ query_embedding.T  # Matrix multiplication
        similarities = similarities.flatten()  # Convert to 1D array of scores
    
    # Find top-K most similar chunks
    # Use min() to handle cases where there are fewer chunks than TOP_K
    top_values, top_indices = torch.topk(similarities, k=min(TOP_K, len(similarities)), largest=True)
    
    # Build results list with chunk content and similarity scores
    results = []
    for i in range(len(top_indices)):
        idx = top_indices[i].item()  # Convert tensor to Python int
        score = top_values[i].item()  # Get similarity score
        chunk = chunks[idx]
        results.append({
            'text': chunk['text'],
            'similarity': score,  # Cosine similarity (0-1 range)
            'length': chunk['length'],
            'tokens': chunk['tokens']
        })
    
    return results


def generate_all_embeddings(chunks: List[Dict[str, Any]], model, tokenizer, device: str) -> torch.Tensor:
    """
    Generate embeddings for all document chunks (indexing phase of RAG).
    
    This processes chunks in batches for efficiency. The resulting embeddings
    are stored in memory for fast similarity search at query time.
    
    Args:
        chunks: List of document chunks
        model: Embedding model
        tokenizer: Tokenizer
        device: Computation device
    
    Returns:
        Tensor of all chunk embeddings [num_chunks, embedding_dim]
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Process chunks in batches to balance memory usage and speed
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        # Get current batch of chunks
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk['text'] for chunk in batch_chunks]
        # Generate embeddings for this batch
        batch_embeddings = get_embeddings(batch_texts, model, tokenizer, device)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings into single tensor
    return torch.cat(all_embeddings, dim=0)


def display_results(query: str, results: List[Dict[str, Any]]) -> None:
    """
    Display search results with similarity scores and token counts.
    
    Shows:
    - Query text
    - Each result's similarity score (higher = more relevant)
    - Token count (important for context window management)
    - Preview of chunk text
    - Total tokens across all results
    """
    print(f"\nQuery: '{query}'")
    print("=" * 50)
    
    # Display each result with metadata
    total_tokens = 0
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        text = result['text']
        tokens = result['tokens']
        total_tokens += tokens
        # Show preview of chunk (first 100 chars)
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"{i}. [Score: {similarity:.3f}] [Tokens: {tokens}] {display_text}")
    
    # Show total token count (important for LLM context limits)
    print(f"\nTotal tokens in results: {total_tokens}")
    print()


def generate_answer(query: str, results: List[Dict[str, Any]]) -> None:
    """
    Generate a complete answer using RAG (Retrieval-Augmented Generation).
    
    This is the "Generation" part of RAG:
    1. Takes the most relevant retrieved chunks as context
    2. Constructs a prompt with both context and question
    3. Uses Qwen3-VL via Ollama to generate a natural language answer
    4. The LLM is grounded by the retrieved context, reducing hallucination
    
    Args:
        query: User's question
        results: Retrieved chunks ranked by relevance
    """
    # Configuration for answer generation
    LLM = "qwen3-vl:4b-instruct"  # Instruction-tuned model for following prompts
    THINKING = False  # Disable chain-of-thought reasoning
    console = Console()  # Rich console for formatted output
    
    # Format context from retrieved chunks
    # Only use top 3 results to stay within context window and focus on most relevant
    context = "\n\n".join([f"Context {i+1}: {result['text'][:1000]}..." if len(result['text']) > 1000 
                           else f"Context {i+1}: {result['text']}" 
                           for i, result in enumerate(results[:3])])  # Truncate very long chunks
    
    # System prompt defines the assistant's role and behavior
    system_prompt = """You are a helpful assistant that answers questions about Grimm fairy tales. 
    Use the provided context from the fairy tales to answer the user's question accurately and concisely.
    Provide your answer in English, even though the source text may be in German.
    Keep your response short and focused on directly answering the question."""
    
    # Construct the user message with both context and question
    # This grounds the model's response in the retrieved information
    instruction = f"""Based on the following context from Grimm fairy tales, please answer this question: {query}

Context:
{context}

Please provide a short, direct answer in English."""
    
    try:
        # Call Ollama with Qwen3 to generate answer
        response = ollama.chat(
            model=LLM,
            think=THINKING,  # Whether to show reasoning process
            stream=False,  # Get complete response at once
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': instruction}
            ],
            options=Options(
                temperature=0.7,  # Moderate creativity (0=deterministic, 1=creative)
                num_ctx=32768,  # 32k context window for long retrieved chunks
                top_p=0.95,  # Nucleus sampling threshold
                top_k=40,  # Consider top 40 tokens at each step
                num_predict=-1  # No limit on response length
            )
        )
        
        # Display the generated answer
        console.print("\n[bold green]RAG Answer:[/bold green]")
        
        # Show reasoning if available (when think=True)
        if hasattr(response.message, 'thinking') and response.message.thinking:
            console.print(f"[dim]Thinking: {response.message.thinking[:200]}...[/dim]")
        
        # Display the final answer
        console.print(f"[blue]{response.message.content}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")


def main() -> None:
    """
    Main demo function demonstrating complete RAG pipeline.
    
    Pipeline stages:
    1. SETUP: Load embedding model and initialize device
    2. INDEX: Load document, chunk it, and generate embeddings
    3. SEARCH: For each query, find relevant chunks via similarity
    4. GENERATE: Use LLM with retrieved context to answer questions
    
    This demonstrates how RAG combines:
    - Information retrieval (semantic search)
    - Language generation (LLM)
    To provide accurate, grounded answers from a knowledge base.
    """
    print("Simple Grimm Tales RAG Search Demo")
    print("=" * 40)
    
    # Setup device - prefer GPU for faster embedding generation
    # mps = Apple Silicon, cuda = NVIDIA GPU, cpu = fallback
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load embedding model and tokenizer
    # This model converts text into semantic vectors for similarity comparison
    print(f"Loading {EMBEDDING_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side='left')
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    
    # Load the document to be searched
    # This is the knowledge base for our RAG system
    try:
        with open("Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt", "r", encoding="utf8") as f:
            text = f.read()
        print(f"Loaded document: {len(text):,} characters")
    except FileNotFoundError:
        print("Error: 'demos/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt' not found!")
        return
    
    # Create chunks - split document into searchable units
    # Each chunk should be semantically coherent (e.g., a complete story)
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings (INDEXING phase)
    # This is done once upfront, then reused for all queries
    # In production, embeddings would be stored in a vector database
    start_time = time.time()
    chunk_embeddings = generate_all_embeddings(chunks, model, tokenizer, device)
    embed_time = time.time() - start_time
    print(f"Embeddings generated in {embed_time:.1f}s")
    
    # Demo queries - questions about the fairy tales
    # These test the system's ability to retrieve relevant passages and generate answers
    queries = [
        "What did the frog king promise the princess in exchange for her golden ball?",
        "What happened to Hansel and Gretel in the forest?",
        "What did Little Red Riding Hood's mother tell her to do?"
    ]
    
    print("\n" + "=" * 60)
    print("SEARCH DEMO")
    print("=" * 60)
    
    # Process each query through the RAG pipeline
    for query in queries:
        # RETRIEVAL: Find relevant chunks via semantic similarity
        start_time = time.time()
        results = search_chunks(query, chunks, chunk_embeddings, model, tokenizer, device)
        search_time = time.time() - start_time
        
        # Display what was retrieved
        display_results(query, results)
        print(f"Search completed in {search_time*1000:.0f}ms")
        
        # GENERATION: Use LLM to synthesize answer from retrieved context
        generate_answer(query, results)
        print("-" * 50)
    
    print("Demo completed!")


if __name__ == "__main__":
    main()
