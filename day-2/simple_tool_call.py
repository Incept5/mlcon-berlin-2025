import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:4b-instruct"


def convert_currency(amount, from_currency, to_currency):
    rates = {
        "EUR-USD": 1.10,
        "USD-EUR": 0.91,
        "GBP-USD": 1.27,
        "USD-GBP": 0.79,
        "JPY-USD": 0.0067,
        "USD-JPY": 149.50
    }

    rate_key = f"{from_currency}-{to_currency}"
    rate = rates.get(rate_key, 1.0)
    result = amount * rate

    return {
        "converted_amount": round(result, 2),
        "rate": rate,
        "result_text": f"{amount} {from_currency} = {result:.2f} {to_currency}"
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another",
            "parameters": {
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
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]


def chat_with_tools(user_message):
    print(f"USER: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False
        }
    )

    assistant_message = response.json().get("message", {})
    tool_calls = assistant_message.get("tool_calls", [])

    if tool_calls:
        print(f"Calling {len(tool_calls)} function(s)\n")
        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            print(f"Function: {function_name}")
            print(f"Arguments: {json.dumps(function_args)}")

            if function_name == "convert_currency":
                result = convert_currency(
                    amount=function_args["amount"],
                    from_currency=function_args["from_currency"],
                    to_currency=function_args["to_currency"]
                )

                print(f"Result: {result['result_text']}\n")

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })

        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False
            }
        )

        final_message = final_response.json().get("message", {}).get("content", "")
        print(f"ASSISTANT: {final_message}\n")

    else:
        content = assistant_message.get("content", "")
        print(f"ASSISTANT: {content}\n")


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")
    print("-" * 60)

    chat_with_tools("What is EUR 50 in USD?")
    print("-" * 60)

    chat_with_tools("Convert 100 USD to GBP")
    print("-" * 60)

    chat_with_tools("What is the capital of France?")
    print("-" * 60)
