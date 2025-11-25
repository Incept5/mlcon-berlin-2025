"""
Formatted Response Example with Ollama
======================================

This script demonstrates how to generate structured JSON responses from an LLM using Ollama.
It showcases the ability to constrain model output to valid JSON format, which is crucial for
building reliable AI applications that need to process LLM responses programmatically.

Key Concepts:
- Structured output generation (JSON format constraint)
- Multilingual data extraction
- Response validation and error handling
- Temperature control for consistent outputs

Use Cases:
- Data extraction and structuring
- API response generation
- Structured logging and analysis
- Building AI applications that require predictable output formats
"""

import ollama
import json


def generate_formatted_response(prompt):
    """
    Generate a structured JSON response from the LLM using Ollama.
    
    This function demonstrates the use of format constraints to ensure the model
    returns valid JSON, making it suitable for applications that need to parse
    and process LLM outputs programmatically.
    
    Args:
        prompt (str): The instruction prompt for the LLM. Should specify the desired
                     JSON structure to guide the model's output format.
    
    Returns:
        str: The model's response as a JSON-formatted string, or None if an error occurs.
    
    Implementation Notes:
    ---------------------
    - format="json": Forces the model to output valid JSON only. This is a critical
                     constraint that ensures the response can be reliably parsed.
                     Only accepts "" (no constraint) or "json" (JSON constraint).
    
    - num_ctx=8192: Sets the context window size. Larger values allow the model to
                    consider more tokens but use more memory. 8192 is sufficient for
                    this multilingual list task.
    
    - temperature=0.3: Lower temperature (0.0-1.0 range) makes outputs more deterministic
                       and focused. Good for structured data extraction where consistency
                       is important. Higher values increase creativity/randomness.
    
    Error Handling:
    ---------------
    Uses try-except to catch and report any Ollama API errors (connection issues,
    model not found, invalid parameters, etc.)
    """
    try:
        response = ollama.generate(
            model="qwen3-vl:4b-instruct",  # Qwen3 4B model - efficient multilingual model
            prompt=prompt,
            format="json",  # Critical: Constrains output to valid JSON only
            options={
                "num_ctx": 8192,      # Context window size in tokens
                "temperature": 0.3     # Low temperature for consistent, focused outputs
            }
        )
        # The response is a dictionary; extract the actual text response
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None


def main():
    """
    Main execution function demonstrating structured multilingual data extraction.
    
    Workflow:
    ---------
    1. Define a structured prompt that specifies the exact JSON format desired
    2. Generate the response with JSON format constraint
    3. Parse and validate the JSON response
    4. Pretty-print the result with proper Unicode support
    
    The example extracts numbers 1-10 with translations in multiple languages,
    demonstrating how to create structured multilingual datasets programmatically.
    """
    # Construct a detailed prompt that specifies both the task and the exact JSON structure
    # This guidance is crucial for getting consistently formatted responses
    prompt = """List the numbers from 1 to 10 and their names in
    English, French, Chinese.
    Provide the output in this exact JSON format:
    {
      "numbers": [
        {
          "number": 1,
          "English": "one",
          "French": "un",
          "Chinese": "一"
        },
        ...and so on for numbers 1-10
      ]
    }"""
    
    # Generate the structured response
    response = generate_formatted_response(prompt)

    # Parse and display the JSON response
    try:
        if response:
            # Parse the JSON string into a Python dictionary
            parsed = json.loads(response)
            
            # Pretty-print with proper formatting
            # indent=2: Makes the JSON human-readable with 2-space indentation
            # ensure_ascii=False: Allows Unicode characters (important for Chinese text)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        # Handle cases where the model doesn't return valid JSON (shouldn't happen with format="json")
        print("Received non-JSON response:")
        print(response)


if __name__ == "__main__":
    """
    Script entry point.
    
    Best Practices Demonstrated:
    ---------------------------
    - Using format constraints for structured output
    - Clear prompt engineering with explicit format specifications
    - Proper JSON parsing and error handling
    - Unicode support for multilingual content
    - Temperature tuning for consistency
    
    Potential Extensions:
    --------------------
    - Add response validation against a JSON schema
    - Implement retry logic for failed generations
    - Support for different output formats (YAML, XML)
    - Batch processing multiple prompts
    - Save results to a file for dataset creation
    """
    main()
