# MCP (Model Context Protocol) Demo

This directory contains a demonstration of the Model Context Protocol (MCP) - a standardized way for AI models to interact with external tools and services. The demo shows how to create an MCP server with custom tools and use it with an open-source LLM running locally via Ollama.

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that enables AI models to securely interact with external data sources and tools. It provides a standardized way to:
- Define tools that AI models can use
- Execute those tools with proper input validation
- Return structured results to the AI model

Think of MCP as a universal adapter that lets AI models interact with external services in a consistent, type-safe way.

## Files in this Directory

### 1. `mcp_server.py`
The MCP server implementation that provides two tools:

**Tools:**
- **`get_country_info`**: Retrieves detailed information about a country from restcountries.com API
  - Input: Country name
  - Returns: Capital, population, region, languages, currencies, timezones, etc.

- **`get_weather`**: Gets current weather information from wttr.in API
  - Input: City or location name
  - Returns: Temperature, humidity, wind speed, weather conditions, etc.

**Features:**
- Asynchronous HTTP requests using `httpx`
- Proper error handling for API failures
- Structured JSON responses
- Type-safe tool definitions with input schemas

### 2. `test_mcp_client_ollama.py`
A demonstration client that uses the MCP server with a local Ollama LLM (instead of commercial APIs like Claude).

**Features:**
- **Agentic Loop**: Allows the LLM to make multiple tool calls to answer complex questions
- **Function Calling**: Converts MCP tools to Ollama-compatible format
- **Chain of Thought**: The LLM can gather information step-by-step (e.g., first find capital, then get weather)
- **Error Handling**: Gracefully handles API errors and connection issues

**Example Questions:**
1. "What is the weather in the capital of Germany?"
   - LLM calls `get_country_info("Germany")` to find capital (Berlin)
   - Then calls `get_weather("Berlin")` to get weather
   - Combines results into a natural language answer

2. "What is the population of France and what's the weather like in Paris?"
   - LLM calls `get_country_info("France")` for population
   - Calls `get_weather("Paris")` for weather
   - Synthesizes both pieces of information

3. "Tell me about Japan - what's its capital and what's the weather there?"
   - LLM calls `get_country_info("Japan")` to learn about the country
   - Extracts capital (Tokyo) from the result
   - Calls `get_weather("Tokyo")` for current conditions
   - Provides comprehensive answer

## Prerequisites

### 1. Install Dependencies
```bash
pip install mcp requests httpx
```

### 2. Install and Run Ollama
```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Start Ollama server
ollama serve

# Pull a model with function calling support (in another terminal)
ollama pull qwen2.5:3b
```

**Recommended Models:**
- `qwen3-vl:4b-instruct` - Excellent function calling, fast (default)
- `mistral:7b` - Good function calling, balanced
- `llama3.1:8b` - Strong reasoning, larger model

## Usage

### Running the Demo

1. **Start Ollama** (if not already running):
```bash
ollama serve
```

2. **Run the MCP demo**:
```bash
python test_mcp_client_ollama.py
```

The script will:
1. Start the MCP server as a subprocess
2. Connect to it and list available tools
3. Run through three example questions
4. Show the LLM's reasoning and tool calls
5. Display the final answers

### Expected Output

```
🤖 MCP Client with Ollama - Open Source LLM Demo
📦 Model: qwen2.5:3b

🚀 Starting MCP Server...
============================================================
✅ Connected to MCP server with 2 tools:
   • get_country_info: Get information about a country...
   • get_weather: Get current weather information...

============================================================
Question 1: What is the weather in the capital of Germany?
============================================================

🔧 Calling tool: get_country_info
   Input: {
     "country_name": "Germany"
   }
   Result: {"name": "Germany", "capital": "Berlin", ...}

🔧 Calling tool: get_weather
   Input: {
     "location": "Berlin"
   }
   Result: {"location": "Berlin", "temperature_c": "15", ...}

💬 Answer:
The capital of Germany is Berlin. The current weather in Berlin is...
```

## How It Works

### Architecture Flow

```
┌─────────────┐         ┌─────────────┐         ┌──────────────┐
│   Ollama    │◄───────►│ MCP Client  │◄───────►│  MCP Server  │
│    LLM      │         │   (stdio)   │         │   (stdio)    │
└─────────────┘         └─────────────┘         └──────────────┘
                                                        │
                                                        ▼
                                                  ┌──────────┐
                                                  │ External │
                                                  │   APIs   │
                                                  └──────────┘
```

### Sequence of Operations

1. **Initialization**:
   - Client starts MCP server as subprocess
   - Establishes stdio communication channel
   - Retrieves available tools and their schemas

2. **User Question**:
   - User asks a complex question
   - Client sends question to Ollama with tool definitions

3. **Agentic Loop**:
   - Ollama decides which tools to call
   - Client executes tool calls via MCP server
   - Results are added to conversation history
   - Loop continues until Ollama has enough information

4. **Final Answer**:
   - Ollama synthesizes information from tool calls
   - Returns natural language answer to user

## Technical Details

### MCP Server Communication
- Uses **stdio** (standard input/output) for communication
- Messages are JSON-formatted following MCP protocol
- Async/await for non-blocking I/O operations

### Function Calling
- MCP tools → Ollama function calling format conversion
- Input validation using JSON schemas
- Type-safe parameter passing

### Error Handling
- Connection failures to external APIs
- Invalid tool parameters
- Model timeout or iteration limits
- Graceful degradation on errors

## Customization

### Adding New Tools

Edit `mcp_server.py` to add new tools:

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... existing tools ...
        Tool(
            name="your_new_tool",
            description="What your tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    if name == "your_new_tool":
        # Implement your tool logic here
        result = your_function(arguments["param1"])
        return [TextContent(type="text", text=result)]
```

### Changing the Model

Edit `test_mcp_client_ollama.py`:

```python
MODEL = "mistral:7b"  # or "llama3.1:8b", etc.
```

### Adjusting Iteration Limit

In `test_mcp_client_ollama.py`:

```python
max_iterations = 10  # Increase for more complex questions
```

## Troubleshooting

### Ollama Not Running
```
❌ Error: Cannot connect to Ollama at http://localhost:11434
```
**Solution**: Start Ollama with `ollama serve`

### Model Not Found
```
⚠️ Warning: Model 'qwen2.5:3b' not found in Ollama
```
**Solution**: Pull the model with `ollama pull qwen2.5:3b`

### API Rate Limits
The free APIs used (restcountries.com, wttr.in) may have rate limits. If you encounter issues, wait a few minutes between runs.

### Tool Call Timeout
If the LLM takes too long, adjust the timeout in `call_ollama()`:

```python
response = requests.post(OLLAMA_URL, json=payload, timeout=60)  # Increase timeout
```

## Learning Outcomes

This demo illustrates:
1. **Protocol Standards**: How MCP standardizes tool interaction
2. **Agentic Behavior**: How LLMs can chain tool calls autonomously
3. **Local LLM Usage**: Running powerful AI without external APIs
4. **Function Calling**: How modern LLMs understand and use tools
5. **Error Handling**: Production-ready error management

## Related Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Function Calling Guide](https://ollama.ai/blog/tool-support)

## License

This demo code is provided as-is for educational purposes.
