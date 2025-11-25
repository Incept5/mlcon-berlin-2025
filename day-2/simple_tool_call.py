"""
Simple Tool Calling Demo with Ollama

This script demonstrates how to use Ollama's function/tool calling capabilities
to extend an LLM's abilities. The LLM can decide when to call a function, 
extract the necessary parameters, and incorporate the results into its response.

Key Concepts:
- Tool/Function Definition: Describing available functions to the LLM
- Tool Calling Flow: LLM decides to call function → Execute function → LLM uses result
- Message History: Maintaining conversation context including tool interactions
"""

import json
import requests

# Ollama API endpoint for chat completions
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model that supports tool calling - must be a vision-language model that supports tools
MODEL = "qwen3-vl:4b-instruct"


def convert_currency(amount, from_currency, to_currency):
    """
    Convert an amount from one currency to another using predefined exchange rates.
    
    In a production system, this would call a real-time currency API.
    For this demo, we use hardcoded rates to avoid external dependencies.
    
    Args:
        amount (float): The amount to convert
        from_currency (str): Source currency code (e.g., 'EUR', 'USD')
        to_currency (str): Target currency code (e.g., 'USD', 'GBP')
    
    Returns:
        dict: Contains converted amount, exchange rate, and formatted result text
    """
    # Simplified exchange rates (in reality, fetch from an API like exchangerate-api.com)
    rates = {
        "EUR-USD": 1.10,
        "USD-EUR": 0.91,
        "GBP-USD": 1.27,
        "USD-GBP": 0.79,
        "JPY-USD": 0.0067,
        "USD-JPY": 149.50
    }

    # Look up the exchange rate for this currency pair
    rate_key = f"{from_currency}-{to_currency}"
    rate = rates.get(rate_key, 1.0)  # Default to 1.0 if pair not found
    result = amount * rate

    return {
        "converted_amount": round(result, 2),
        "rate": rate,
        "result_text": f"{amount} {from_currency} = {result:.2f} {to_currency}"
    }


# Tool definitions in OpenAI-compatible format
# This tells the LLM what functions are available and how to call them
tools = [
    {
        "type": "function",  # Indicates this is a function tool
        "function": {
            "name": "convert_currency",  # Function identifier
            "description": "Convert an amount from one currency to another",  # Helps LLM decide when to use this
            "parameters": {
                "type": "object",
                "properties": {
                    # Define each parameter the function accepts
                    "amount": {
                        "type": "number",
                        "description": "The amount to convert"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "The source currency code (e.g., EUR, USD, GBP)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "The target currency code (e.g., EUR, USD, GBP)"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]  # All parameters are mandatory
            }
        }
    }
]


def chat_with_tools(user_message):
    """
    Handle a chat interaction that may involve tool calls.
    
    The flow is:
    1. Send user message with available tools to LLM
    2. LLM responds with either:
       a) A direct text response (no tool needed), OR
       b) A request to call one or more tools
    3. If tools were requested:
       - Execute each tool call
       - Send results back to LLM
       - Get final natural language response
    
    Args:
        user_message (str): The user's question or request
    """
    print(f"USER: {user_message}\n")

    # Initialize conversation with user's message
    messages = [{"role": "user", "content": user_message}]

    # First API call: Send user message + available tools
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,  # Inform LLM about available functions
            "stream": False  # Get complete response at once
        }
    )

    # Extract the assistant's response
    assistant_message = response.json().get("message", {})
    tool_calls = assistant_message.get("tool_calls", [])

    # Check if LLM wants to call any tools
    if tool_calls:
        print(f"Calling {len(tool_calls)} function(s)\n")
        
        # Add assistant's tool call request to message history
        # This is crucial - we need to maintain the full conversation context
        messages.append(assistant_message)

        # Execute each tool call the LLM requested
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            print(f"Function: {function_name}")
            print(f"Arguments: {json.dumps(function_args)}")

            # Route to the appropriate function
            # In a larger system, you'd use a dictionary or registry pattern
            if function_name == "convert_currency":
                result = convert_currency(
                    amount=function_args["amount"],
                    from_currency=function_args["from_currency"],
                    to_currency=function_args["to_currency"]
                )

                print(f"Result: {result['result_text']}\n")

                # Add tool result to conversation history
                # The LLM needs this to formulate its final response
                messages.append({
                    "role": "tool",  # Special role indicating this is a tool result
                    "content": json.dumps(result)  # Function results as JSON string
                })

        # Second API call: Send conversation including tool results
        # The LLM will now incorporate the tool results into a natural language response
        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,  # Full history: user → assistant → tool results
                "stream": False
            }
        )

        # Extract and display the final response
        final_message = final_response.json().get("message", {}).get("content", "")
        print(f"ASSISTANT: {final_message}\n")

    else:
        # No tool calls needed - LLM responded directly
        # This happens when the question doesn't require tool use
        content = assistant_message.get("content", "")
        print(f"ASSISTANT: {content}\n")


if __name__ == "__main__":
    """
    Demo different scenarios:
    1. Currency conversion (triggers tool call)
    2. Another currency conversion (different format, still triggers tool)
    3. General knowledge question (no tool call needed)
    """
    print(f"Model: {MODEL}\n")
    print("-" * 60)

    # Example 1: Simple currency conversion
    chat_with_tools("What is EUR 50 in USD?")
    print("-" * 60)

    # Example 2: Different phrasing, same tool call
    chat_with_tools("Convert 100 USD to GBP")
    print("-" * 60)

    # Example 3: Question that doesn't need tools
    # The LLM will respond directly without calling any functions
    chat_with_tools("What is the capital of France?")
    print("-" * 60)
