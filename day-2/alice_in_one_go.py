"""Full Context Question Answering System

This module demonstrates a full-context QA approach where the entire document
is provided to the LLM in each query, as opposed to RAG (Retrieval Augmented
Generation) which only provides relevant chunks.

The system:
1. Loads a complete text document (Alice in Wonderland)
2. Cleans and preprocesses the text
3. Sends the entire document + question to Ollama with large context window
4. Returns detailed answers based on the full document context

This approach is useful when:
- Document size fits within model's context window (64k tokens)
- Questions require understanding of full narrative flow
- Maximum accuracy is needed without risk of missing relevant passages
- Latency is acceptable (processing entire document takes time)
"""

from typing import List
import requests
import re
from pathlib import Path

# Configuration
# The Ollama model to use for question answering
# Should support large context windows (32k-64k+ tokens)
LLM_MODEL = "gemma3"


class OllamaClient:
    """Client for interacting with local Ollama API.
    
    Handles communication with Ollama's chat API endpoint,
    configured for large context windows to support full-document QA.
    """
    
    def __init__(self):
        """Initialize client with Ollama API endpoint."""
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API with large context window.
        
        Args:
            prompt: The complete prompt including document and question
            
        Returns:
            Generated response text, or error message if request fails
            
        Configuration:
            - num_ctx: 65536 tokens (64k context window)
            - temperature: 0.05 (low for factual accuracy)
            - top_p: 0.85 (moderate nucleus sampling)
            - timeout: 300s (5 minutes for large document processing)
        """
        # Prepare API request with large context window configuration
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Get complete response at once (no streaming)
            "options": {
                "num_ctx": 65536,  # 64k token context - allows full document + question
                "temperature": 0.05,  # Very low temperature for factual accuracy
                "top_p": 0.85  # Moderate nucleus sampling for coherent responses
            }
        }

        print(f"Sending request to Ollama... (prompt length: {len(prompt)} chars)")
        try:
            # POST to Ollama chat endpoint with generous timeout for large documents
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=300)  # 5 minute timeout
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Extract the generated text from response
            result = response.json()["message"]["content"]
            print("Response received successfully!")
            return result
        except requests.exceptions.Timeout:
            # Document too large or model too slow
            return "Error: Request timed out - document may be too large for model context"
        except requests.exceptions.RequestException as e:
            # Network issues, API unavailable, or other HTTP errors
            return f"Error: Network/API issue - {e}"


class FullContextQA:
    """Full-context Question Answering system.
    
    Loads an entire document into memory and provides it as full context
    for every question. This contrasts with RAG approaches that retrieve
    only relevant chunks.
    
    Advantages:
    - No risk of missing relevant information
    - Understands full narrative context
    - Simpler architecture (no vector DB needed)
    
    Disadvantages:
    - Slower (processes entire document per query)
    - Limited to documents that fit in context window
    - Higher computational cost per query
    """
    
    def __init__(self, file_path: str):
        """Initialize QA system with a document.
        
        Args:
            file_path: Path to text file to load
            
        Initialization:
            1. Tests Ollama connection
            2. Loads and cleans document text
            3. Stores full text in memory for repeated queries
        """
        self.file_path = Path(file_path)
        self.ollama = OllamaClient()
        self.full_text = ""  # Will store complete cleaned document
        self._test_connection()
        self._load_document()

    def _test_connection(self):
        """Test if Ollama API and model are accessible.
        
        Sends a simple test query to verify:
        - Ollama server is running
        - API endpoint is accessible
        - Specified model is available
        - Model can generate responses
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        print("Testing Ollama connection...")
        try:
            test_response = self.ollama.generate_response("Say 'Hello' to test the connection.")
            if "Error:" not in test_response:
                print(f"✅ Connection test successful: {test_response[:50]}...")
            else:
                print(f"❌ Connection test failed: {test_response}")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        return True

    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text.
        
        Preprocessing steps:
        1. Normalize line breaks (\r\n -> \n)
        2. Reduce excessive blank lines (3+ -> 2)
        3. Normalize whitespace (multiple spaces/tabs -> single space)
        4. Remove table of contents (find actual story start)
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text ready for model consumption
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\r\n', '\n', text)  # Convert Windows line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse multiple spaces/tabs

        # Remove table of contents by finding actual story start
        # Look for the pattern: "CHAPTER I." followed by a title, then start content
        if 'Contents' in text:
            # Find "CHAPTER I." followed by the actual chapter title
            # This skips the TOC and starts at the actual narrative
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                # Start from the end of this match (after the title and blank line)
                text = text[match.end():]

        return text.strip()

    def _load_document(self):
        """Load document from file and prepare for QA.
        
        Steps:
        1. Read file contents (UTF-8 encoding)
        2. Apply text cleaning/normalization
        3. Store in memory for repeated queries
        4. Display statistics (char count, estimated tokens)
        
        Token estimation: ~4 characters per token (rough approximation)
        """
        print(f"Loading document: {self.file_path.name}")

        # Read entire file into memory
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Apply cleaning and normalization
        self.full_text = self._clean_text(text)

        # Display loading statistics
        print(f"Document loaded: {len(self.full_text)} characters")
        print(f"Estimated tokens: ~{len(self.full_text) // 4}")  # Rough estimate: 4 chars/token

    def query(self, question: str) -> str:
        """Answer a question using the complete document as context.
        
        This is the core QA method. For each question:
        1. Validates question is not empty
        2. Creates prompt with full document + question
        3. Sends to LLM with large context window
        4. Returns detailed answer based on complete text
        
        Args:
            question: Natural language question about the document
            
        Returns:
            Detailed answer with quotes from text when appropriate,
            or error message if generation fails
        """
        # Validate input
        if not question.strip():
            return "Please provide a question."

        # Create comprehensive prompt with instructions + full document + question
        # The entire document is included in every query
        prompt = f"""Answer the question using the provided complete text.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the text doesn't contain enough information, say so clearly.
Focus on accuracy - only state what is supported by the text.

Question: {question}

Complete Text:
{self.full_text}

Answer:"""

        # Generate response using full context
        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            # Catch any unexpected errors during generation
            return f"Error generating response: {e}"


def main():
    """Demonstrate full-context QA system with Alice in Wonderland.
    
    This demo:
    1. Initializes the QA system with Alice in Wonderland text
    2. Tests with 3 sample questions in English
    3. Shows how the system handles questions about the complete narrative
    4. Provides additional test questions including multilingual examples
    
    The full question set includes English, German, French, and Spanish
    to demonstrate the model's multilingual capabilities with full context.
    """
    print("Full Context QA Demo System (32k context)")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize full-context system
        # This loads the entire Alice in Wonderland text into memory
        qa = FullContextQA("data/alice_in_wonderland.txt")
        print(f"\nFull-context system ready!\n")

        # Start with just a few questions to test
        # These questions test different aspects:
        # 1. Story opening (narrative context)
        # 2. Specific detail (object description)
        # 3. Character dialogue (direct quotes)
        test_questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "What did the White Rabbit say when Alice first saw him?"
        ]

        print("🧪 Testing with first 3 questions:")
        print("=" * 70)

        # Process each test question
        for i, question in enumerate(test_questions, 1):
            print(f"Question {i}/3: {question}")
            print("-" * 50)

            # Query the system - entire document is sent with each question
            answer = qa.query(question)
            print(f"A: {answer}\n")
            print("=" * 70)

            # If we get an error, stop testing
            # Common errors: Ollama not running, model not available, timeout
            if "Error:" in answer:
                print("Stopping due to error. Check Ollama configuration.")
                break

        print("\n💡 If this works well, you can test more questions or increase context size!")

        # Complete question set for reference (not all run in demo)
        # Includes multilingual questions to test the model's ability to:
        # 1. Process questions in different languages
        # 2. Extract answers from English text
        # 3. Respond in the question's language
        all_questions = [
            # English questions - various types
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat when Alice first met him?",
            "What happens when Alice falls down the rabbit hole?",
            "What did the White Rabbit say when Alice first saw him?",
            # Multilingual variations of same questions
            "Was stand auf der Flasche, die Alice schrumpfen ließ?",  # German: bottle label
            "Que faisait Alice au début de l'histoire?",  # French: Alice's initial activity
            "¿Qué dijo el Conejo Blanco cuando Alice lo vio por primera vez?"  # Spanish: White Rabbit's words
        ]

    except Exception as e:
        # Handle any initialization or runtime errors
        print(f"Error: {e}")
        print("\nMake sure:")
        print("   - Ollama is running on port 11434")
        print("   - Model 'gemma3n:e4b' is available")
        print("   - The data file exists and is readable")


if __name__ == "__main__":
    main()