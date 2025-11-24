#!/usr/bin/env python3
"""
MCP Client with Ollama - Open Source LLM Demo

This script demonstrates using the MCP server with an open source LLM
running locally via Ollama instead of Claude API. It shows how to:
1. Connect to an MCP server and discover available tools
2. Use Ollama's function calling capability to invoke MCP tools
3. Implement an agentic loop that chains tool calls to answer complex questions

Requirements:
- Ollama must be running locally (ollama serve)
- A model with function calling support (e.g., mistral, llama3.1, qwen2.5)

Example questions:
- "What is the weather in the capital of Germany?"
- "What is the population of France and what's the weather like in Paris?"
- "Tell me about Japan and its weather in Tokyo"

Usage:
    python test_mcp_client_ollama.py
"""

import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:4b-instruct"


def call_ollama(messages, tools=None):
    """
    Call Ollama API with messages and optional tools.
    
    Args:
        messages: List of conversation messages
        tools: Optional list of tool definitions in OpenAI format
    
    Returns:
        dict: Response from Ollama including message and any tool calls
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    
    if tools:
        payload["tools"] = tools
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Ollama: {e}")
        return None


def convert_mcp_tool_to_ollama_format(mcp_tool):
    """
    Convert MCP tool definition to Ollama/OpenAI function calling format.
    
    Args:
        mcp_tool: MCP tool object with name, description, and inputSchema
    
    Returns:
        dict: Tool definition in Ollama-compatible format
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }
    }


async def run_mcp_demo_with_ollama():
    """Run the MCP demo with Ollama LLM."""
    
    # Server parameters for our MCP server
    import os
    import sys
    
    # Use the same Python interpreter as the client (from venv)
    python_path = sys.executable
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    server_params = StdioServerParameters(
        command=python_path,
        args=[server_script],
        env=None
    )
    
    print("🚀 Starting MCP Server...")
    print("=" * 60)
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"✅ Connected to MCP server with {len(tools_list.tools)} tools:")
            for tool in tools_list.tools:
                print(f"   • {tool.name}: {tool.description}")
            print()
            
            # Convert MCP tools to Ollama format
            ollama_tools = [
                convert_mcp_tool_to_ollama_format(tool) 
                for tool in tools_list.tools
            ]
            
            # Demo questions
            questions = [
                "What is the weather in the capital of Germany?",
                "What is the population of France and what's the weather like in Paris?",
                "Tell me about Japan - what's its capital and what's the weather there?"
            ]
            
            for i, question in enumerate(questions, 1):
                print(f"\n{'=' * 60}")
                print(f"Question {i}: {question}")
                print('=' * 60)
                
                # Start conversation with the user's question
                messages = [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
                
                # Agentic loop - allow multiple tool calls
                max_iterations = 10
                iteration = 0
                
                while iteration < max_iterations:
                    iteration += 1
                    
                    # Call Ollama with current messages and available tools
                    response_data = call_ollama(messages, tools=ollama_tools)
                    
                    if not response_data:
                        print("❌ Failed to get response from Ollama")
                        break
                    
                    assistant_message = response_data.get("message", {})
                    
                    # Check if there are tool calls
                    tool_calls = assistant_message.get("tool_calls", [])
                    
                    if tool_calls:
                        # Model wants to use tools
                        # Add assistant message with tool calls to conversation
                        messages.append(assistant_message)
                        
                        # Execute each tool call via MCP
                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            function_args = tool_call["function"]["arguments"]
                            
                            print(f"\n🔧 Calling tool: {function_name}")
                            print(f"   Input: {json.dumps(function_args, indent=2)}")
                            
                            try:
                                # Call the MCP tool
                                result = await session.call_tool(function_name, function_args)
                                
                                # Extract text content from result
                                result_text = ""
                                for content in result.content:
                                    if hasattr(content, "text"):
                                        result_text += content.text
                                
                                print(f"   Result: {result_text[:200]}...")
                                
                                # Add tool result to messages
                                messages.append({
                                    "role": "tool",
                                    "content": result_text
                                })
                                
                            except Exception as e:
                                print(f"   ❌ Error calling tool: {e}")
                                # Add error as tool result
                                messages.append({
                                    "role": "tool",
                                    "content": f"Error: {str(e)}"
                                })
                    else:
                        # No tool calls - we have the final answer
                        content = assistant_message.get("content", "")
                        if content:
                            print(f"\n💬 Answer:\n{content}")
                        break
                
                if iteration >= max_iterations:
                    print(f"\n⚠️ Reached maximum iterations ({max_iterations})")
            
            print(f"\n\n{'=' * 60}")
            print("✅ Demo completed successfully!")
            print('=' * 60)


async def check_ollama_availability():
    """Check if Ollama is running and the model is available."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        # Check if the model is available
        models_data = response.json()
        available_models = [m["name"] for m in models_data.get("models", [])]
        
        if not any(MODEL in m for m in available_models):
            print(f"⚠️  Warning: Model '{MODEL}' not found in Ollama")
            print(f"Available models: {', '.join(available_models)}")
            print(f"\nTo pull the model, run: ollama pull {MODEL}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Cannot connect to Ollama at http://localhost:11434")
        print(f"Make sure Ollama is running with: ollama serve")
        print(f"Error details: {e}")
        return False


if __name__ == "__main__":
    print("🤖 MCP Client with Ollama - Open Source LLM Demo")
    print(f"📦 Model: {MODEL}")
    print()
    
    # Check Ollama availability first
    if not asyncio.run(check_ollama_availability()):
        print("\n❌ Cannot proceed without Ollama running")
        exit(1)
    
    try:
        asyncio.run(run_mcp_demo_with_ollama())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
