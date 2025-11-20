from typing import List
import requests
import re
from pathlib import Path

# Configuration
LLM_MODEL = "gemma3"


class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API with 64k context."""
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 65536,  # Start with 64k context
                "temperature": 0.05,
                "top_p": 0.85
            }
        }

        print(f"Sending request to Ollama... (prompt length: {len(prompt)} chars)")
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=300)  # 5 minute timeout
            response.raise_for_status()
            result = response.json()["message"]["content"]
            print("Response received successfully!")
            return result
        except requests.exceptions.Timeout:
            return "Error: Request timed out - document may be too large for model context"
        except requests.exceptions.RequestException as e:
            return f"Error: Network/API issue - {e}"


class FullContextQA:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.ollama = OllamaClient()
        self.full_text = ""
        self._test_connection()
        self._load_document()

    def _test_connection(self):
        """Test if Ollama and the model are working."""
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
        """Load and clean the document."""
        print(f"Loading document: {self.file_path.name}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean text
        self.full_text = self._clean_text(text)

        print(f"Document loaded: {len(self.full_text)} characters")
        print(f"Estimated tokens: ~{len(self.full_text) // 4}")

    def query(self, question: str) -> str:
        """Answer a question using the full document context."""
        if not question.strip():
            return "Please provide a question."

        # Create prompt with full document
        prompt = f"""Answer the question using the provided complete text.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the text doesn't contain enough information, say so clearly.
Focus on accuracy - only state what is supported by the text.

Question: {question}

Complete Text:
{self.full_text}

Answer:"""

        # Generate response
        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    """Demo the full-context QA system vs RAG."""
    print("Full Context QA Demo System (32k context)")
    print("Make sure Ollama is running on port 11434\n")

    try:
        # Initialize full-context system
        qa = FullContextQA("data/alice_in_wonderland.txt")
        print(f"\nFull-context system ready!\n")

        # Start with just a few questions to test
        test_questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "What did the White Rabbit say when Alice first saw him?"
        ]

        print("🧪 Testing with first 3 questions:")
        print("=" * 70)

        for i, question in enumerate(test_questions, 1):
            print(f"Question {i}/3: {question}")
            print("-" * 50)

            answer = qa.query(question)
            print(f"A: {answer}\n")
            print("=" * 70)

            # If we get an error, stop testing
            if "Error:" in answer:
                print("Stopping due to error. Check Ollama configuration.")
                break

        print("\n💡 If this works well, you can test more questions or increase context size!")

        # All questions for reference:
        all_questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat when Alice first met him?",
            "What happens when Alice falls down the rabbit hole?",
            "What did the White Rabbit say when Alice first saw him?",
            "Was stand auf der Flasche, die Alice schrumpfen ließ?",  # What was written on the bottle that made Alice shrink?
            "Que faisait Alice au début de l'histoire?",  # What was Alice doing at the beginning of the story?
            "¿Qué dijo el Conejo Blanco cuando Alice lo vio por primera vez?"  # What did the White Rabbit say when Alice first saw him?
        ]

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("   - Ollama is running on port 11434")
        print("   - Model 'gemma3n:e4b' is available")
        print("   - The data file exists and is readable")


if __name__ == "__main__":
    main()