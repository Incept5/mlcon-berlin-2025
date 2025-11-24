"""
Local LLM Performance Demonstration Script

This script benchmarks the performance of a locally-running Ollama model (qwen3-vl:8b-instruct)
by executing a series of diverse tasks and measuring their execution time. It demonstrates:
- Integration with local Ollama API
- Performance timing and measurement
- Decorator pattern for timing function execution
- Direct answer generation without explanation

The script uses a variety of task types (factual questions, code generation, mathematics)
to showcase the model's versatility and response times.
"""

import time
import ollama
import re  # Note: Currently imported but not used in the script

def time_execution(func):
    """
    Decorator function that wraps another function to measure and report its execution time.
    
    This decorator:
    1. Records the start time before function execution
    2. Calls the wrapped function with the task parameter
    3. Calculates elapsed time in milliseconds
    4. Prints formatted output showing task number, task description, response, and timing
    
    Args:
        func: The function to wrap (expected to take a task/prompt and return a dict with 'response' key)
    
    Returns:
        wrapper: The wrapped function that includes timing functionality
    """
    def wrapper(i, task):
        # Record start time using monotonic clock for accurate measurement
        start = time.time()
        
        # Execute the wrapped function (LLM query)
        result = func(task)
        
        # Calculate elapsed time in milliseconds for human-readable output
        elapsed = (time.time() - start) * 1000
        
        # Print formatted output with task number, prompt, response, and execution time
        print(f"Task {i+1}: {task}:\n\t\t{result['response']} ({elapsed:.2f} ms)\n")
        
        return result
    return wrapper

# Initialize Ollama client for communicating with local Ollama server
client = ollama.Client()

# Wrap the lambda function with timing decorator
# The lambda creates a query function that:
# - Uses the qwen3-vl:8b-instruct model (8 billion parameter vision-language model)
# - Disables "thinking" output (think=False) for direct answers
# - Uses a system prompt to enforce concise responses without explanations
# - Optional larger model available: :30b-a3b-q8_0 (30B parameters, quantized)
run_query = time_execution(lambda p: client.generate(
    model="qwen3-vl:8b-instruct",  # Base model selection
    prompt=p,                       # User prompt/task
    think=False,                    # Disable chain-of-thought reasoning output
    system="Only provide the answer, no explanation"  # System instruction for concise responses
))

# Warm-up call: Initialize the model with an empty prompt
# This ensures the first timed task doesn't include model loading/initialization overhead
# which would skew the performance measurements
run_query(-1, "")  # Task index -1 indicates this is not a real task

# Define test tasks covering different capabilities:
# - Historical facts (tasks 0-1)
# - Scientific knowledge (task 2)
# - Geographic knowledge (task 3)
# - Code generation (tasks 4-5)
# - Mathematical computation (task 6)
tasks = [
    "Which British king abdicated in the 1936",                                            # Historical fact
    "Where was Salvador Dali born?",                                                       # Geographic/biographical fact
    "What is Einstein's most famous equation?",                                            # Scientific knowledge
    "In which country is Krakatoa?",                                                       # Geographic knowledge
    "Write a single line in Java to print 'Hello World'",                                  # Code generation (Java)
    "Write a single line command for Linux to count the number of 20-letter words in the dictionary",  # Code generation (Bash)
    "What is the derivative of 6x^2-5x+12?"                                               # Mathematical computation
]

# Execute each task and measure performance
# enumerate() provides both the index (i) and the task content
for i, task in enumerate(tasks):
    run_query(i, task)

# Expected output format for each task:
# Task N: [question]
#     [answer] (XX.XX ms)
#
# This allows easy comparison of:
# - Response quality across different task types
# - Performance variations between simple and complex queries
# - Model latency for different categories of prompts
