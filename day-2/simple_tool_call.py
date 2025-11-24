"""
Simple Tool Calling Demo with Ollama

This script demonstrates how to implement function calling (also known as tool calling)
with a local LLM via Ollama. Function calling allows the LLM to:
1. Recognize when it needs to use a tool/function to answer a question
2. Extract the necessary parameters from the user's question
3. Call the appropriate function with those parameters
4. Use the function's result to formulate a natural language response

The workflow is:
User Question → LLM decides to use tool → Function executes → LLM uses result → Final answer
"""

import json
import requests

# Ollama API endpoint - local server that hosts and runs LLM models
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "granite4"  # IBM's Granite model - good for structured tasks like function calling


def convert_currency(amount, from_currency, to_currency):
    """
    Convert currency using dummy exchange rates.
    
    This is the actual Python function that gets called when the LLM decides
    it needs to convert currency. In a real application, this would call
    an external API like exchangerate-api.com or fixer.io.
    
    Args:
        amount (float): Amount to convert
        from_currency (str): Source currency code (e.g., 'EUR', 'USD')
        to_currency (str): Target currency code (e.g., 'USD', 'GBP')
    
    Returns:
        dict: Contains verification string, converted amount, rate, and result text
    """
    print(f"\n🔧 FUNCTION CALLED: convert_currency()")
    print(f"   Amount: {amount}")
    print(f"   From: {from_currency}")
    print(f"   To: {to_currency}")

    # Dummy exchange rates for demonstration
    # In production, fetch these from a real currency API
    rates = {
        "EUR-USD": 1.10,   # 1 EUR = 1.10 USD
        "USD-EUR": 0.91,   # 1 USD = 0.91 EUR
        "GBP-USD": 1.27,   # 1 GBP = 1.27 USD
        "USD-GBP": 0.79,   # 1 USD = 0.79 GBP
        "JPY-USD": 0.0067, # 1 JPY = 0.0067 USD
        "USD-JPY": 149.50  # 1 USD = 149.50 JPY
    }

    # Look up the exchange rate for the requested conversion
    rate_key = f"{from_currency}-{to_currency}"
    rate = rates.get(rate_key, 1.0)  # Default to 1.0 if rate not found

    # Calculate the converted amount
    result = amount * rate

    # Create verification string to confirm the conversion parameters
    verification = f"{amount:.2f}-{from_currency}-{to_currency}"

    # Return structured data that the LLM can use to formulate its response
    return {
        "verification": verification,
        "converted_amount": round(result, 2),
        "rate": rate,
        "result_text": f"{amount} {from_currency} = {result:.2f} {to_currency}"
    }


# Define the function schema that tells the LLM about available tools
# This is like a contract that describes what functions exist and how to call them
tools = [
    {
        "type": "function",  # Indicates this is a function tool
        "function": {
            "name": "convert_currency",  # Function name the LLM should use
            "description": "Convert an amount from one currency to another",  # Helps LLM decide when to use this
            "parameters": {
                # OpenAPI-style schema defining the function parameters
                "type": "object",
                "properties": {
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
                # Specify which parameters are mandatory
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]


def chat_with_tools(user_message):
    """
    Main function that handles the complete tool calling workflow.
    
    This implements a multi-step conversation with the LLM:
    1. Send user's question along with available tools
    2. Check if LLM wants to call any functions
    3. Execute the requested functions
    4. Send function results back to LLM
    5. Get final natural language response
    
    Args:
        user_message (str): The user's question or prompt
    """
    print(f"💬 USER: {user_message}")
    print("=" * 60)

    # Initialize the conversation history
    # Messages is a list that tracks the entire conversation flow
    messages = [
        {"role": "user", "content": user_message}
    ]

    # STEP 1: Send initial request to Ollama with tool definitions
    print("📤 Sending request to Ollama...")
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,  # Include tool definitions so LLM knows what's available
            "stream": False  # Get complete response at once (not streaming)
        }
    )

    response_data = response.json()
    assistant_message = response_data.get("message", {})

    print("📥 Received response from Ollama")

    # STEP 2: Check if the model wants to call any functions
    # If the LLM recognizes it needs a tool, it returns tool_calls instead of text
    tool_calls = assistant_message.get("tool_calls", [])

    if tool_calls:
        # LLM decided it needs to use one or more tools
        print(f"🎯 Model wants to call {len(tool_calls)} function(s)")

        # Add the assistant's tool call request to the conversation history
        # This is important for maintaining context
        messages.append(assistant_message)

        # STEP 3: Execute each requested tool call
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            print(f"\n📋 Function: {function_name}")
            print(f"📋 Arguments: {json.dumps(function_args, indent=2)}")

            # Map the function name to the actual Python function
            # In a larger application, you'd use a more sophisticated dispatch mechanism
            if function_name == "convert_currency":
                # Execute the actual function with the LLM-provided arguments
                result = convert_currency(
                    amount=function_args["amount"],
                    from_currency=function_args["from_currency"],
                    to_currency=function_args["to_currency"]
                )

                print(f"\n✅ Function result:")
                print(f"   Verification: {result['verification']}")
                print(f"   Converted: {result['result_text']}")
                print(f"   Rate: {result['rate']}")

                # Add the function's result to the conversation history
                # The role "tool" indicates this message contains function output
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)  # Convert result to JSON string
                })

        # STEP 4: Send function results back to the model for final response
        # Now the LLM has the data it needs to answer the user's original question
        print("\n📤 Sending function results back to model...")
        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,  # Includes: user question + tool call + tool result
                "stream": False
            }
        )

        final_data = final_response.json()
        final_message = final_data.get("message", {}).get("content", "")

        # The LLM now uses the function results to formulate a natural language answer
        print(f"🤖 ASSISTANT: {final_message}")

    else:
        # No function call needed - the LLM can answer directly
        # This happens when the question doesn't require any tools
        content = assistant_message.get("content", "")
        print(f"🤖 ASSISTANT: {content}")

    print("\n" + "=" * 60)


# Run demonstration cases
if __name__ == "__main__":
    print("🚀 Starting Function Calling Demo with Ollama")
    print(f"📦 Model: {MODEL}")
    print("\nThis demo shows how LLMs can decide when to use tools/functions")
    print("and how to handle the complete tool calling workflow.\n")

    # Test case 1: Currency conversion (should trigger function call)
    # The LLM should recognize this needs the convert_currency tool
    chat_with_tools("What is EUR 50 in USD?")

    # Test case 2: Another conversion with different format
    # Tests if LLM can extract parameters from different phrasings
    chat_with_tools("Convert 100 USD to GBP")

    # Test case 3: Regular question (should NOT trigger function call)
    # The LLM should recognize this doesn't need any tools and answer directly
    chat_with_tools("What is the capital of France?")
