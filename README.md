
# MLCon Berlin 2025 - Hands-on GenAI Development Bootcamp 🚀

Welcome to the **2-Day Hands-on GenAI Development Bootcamp**! This repository contains all the code, examples, and exercises you'll work through during the workshop.

## 🎯 What You'll Learn

This bootcamp takes you from GenAI fundamentals to building production-ready AI systems:

**Day 1**: Understanding how LLMs work, connecting to models, and building RAG systems  
**Day 2**: Advanced applications including vision AI, sentiment analysis, structured outputs, function calling, and agentic systems

## ✨ What's New in This Edition

**Enhanced Vision AI**:
- Complete vision model testing framework with 12 diverse image types
- OCR across multiple languages (English, Japanese, French, German)
- Technical diagram analysis and data extraction from images
- Mathematical equation and sheet music recognition

**Expanded RAG Examples**:
- New Grimm fairy tales RAG with story-aware chunking
- Multilingual RAG (English and German texts)
- Both local (Ollama) and cloud (Groq) implementations
- Simplified learning versions alongside production code

**Function Calling & Tool Use**:
- Simple starter example (`simple_tool_call.py`) for easy learning
- Comprehensive multi-tool framework with 8+ functions
- Model compatibility testing and comparison

**Agentic Systems with MCP**:
- Complete Model Context Protocol implementation
- Real-world tools (country info, weather lookup)
- Agentic loops with multi-step reasoning
- Works with local Ollama models (no API keys needed!)

**Cloud API Integration**:
- Groq examples for ultra-fast inference
- Direct comparisons between local and cloud approaches
- Best practices for production deployment

## 📋 Prerequisites

Before the workshop, please ensure you have:

- **Python 3.8+** installed on your machine
- **Basic programming knowledge** (Python preferred, but examples available in Java/Kotlin)
- **4GB+ RAM** for running local models
- **10GB+ free disk space** for models and databases
- **Ollama installed** (instructions below)

## 🛠️ Setup Instructions

### 1. Clone This Repository
```bash
git clone <repository-url>
cd mlcon-berlin-2025
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Local AI Models)
Visit [ollama.com](https://ollama.com) and install Ollama for your operating system.

Then download the models we'll use:
```bash
ollama serve  # Start Ollama (run in separate terminal)
ollama pull qwen3:4b          # 2.5GB Fast general-purpose model
ollama pull qwen3             # Alias for qwen3:4b
ollama pull qwen3-vl:4b-instruct  # 2.8GB Vision-language model for image understanding (recommended)
ollama pull qwen3-vl:2b-instruct  # 1.5GB Smaller, faster vision model
ollama pull qwen3-embedding:0.6b  # 432MB For embeddings (alternative to all-minilm)
ollama pull embeddinggemma    # 621MB For embeddings/RAG (recommended for Grimm tales)
ollama pull gemma3n:e4b       # 7.5GB High-quality reasoning model
ollama pull all-minilm        # 45MB Lightweight embedding model
ollama pull llama3.2          # 2.0GB For code completion and sentiment analysis
ollama pull qwen2.5-coder     # 4.7GB For code generation
ollama pull qwen2.5:3b        # 1.9GB Fast model with function calling support
```

**Model Usage by Feature**:
- **Vision/OCR**: `qwen3-vl:4b-instruct` or `qwen3-vl:2b-instruct`
- **RAG/Embeddings**: `embeddinggemma` or `all-minilm`
- **Function Calling**: `qwen3-vl:4b-instruct`, `qwen2.5:3b`, or `mistral:7b`
- **General Chat**: `qwen3:4b` or `llama3.2`
- **Code Generation**: `qwen2.5-coder`

Note: Some scripts reference cloud models (like `llama-3.3-70b-versatile` on Groq) which don't need to be pulled locally.

### 4. (Optional) Set Up API Keys
If you want to try cloud-based models, create a `.env` file or export these variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GROQ_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"
```

### 5. Verify Installation
```bash
# Test that Ollama is working
python day-1/getting_started_ollama.py
```

You should see a response from the AI model!

---

## 📚 Workshop Structure

## Day 1: AI Fundamentals & RAG Systems

### Session 1: Understanding AI & Tokenization 🧠

**Concepts**: How transformers work, tokens, embeddings, attention mechanisms

**Files to explore**:
- `day-1/show_tokens.py` - **START HERE**: See how text becomes tokens
- `day-1/embedding_demo.py` - Understand semantic similarity with embeddings
- `day-1/embedding_example.py` - More embedding examples
- `day-1/3d_plot.html` - Visualize embeddings in 3D space

**What you'll do**:
1. Run `show_tokens.py` to see how different texts are tokenized
2. Experiment with `embedding_demo.py` to see semantic similarity in action
3. Open `3d_plot.html` in your browser to see embedding clusters

**Key takeaways**:
- Text is converted into numbers (tokens) that AI models understand
- Similar meanings have similar embeddings (vectors)
- Embeddings capture semantic meaning, not just word matching

---

### Session 2: Connecting to AI Models 🔌

**Concepts**: Local vs cloud models, API basics, model parameters

**Files to explore**:

**Local Models (Ollama)**:
- `day-1/getting_started_ollama.py` - Your first local AI conversation
- `day-1/getting_started_lm_studio.py` - Alternative local setup
- `day-1/getting_started_groq.py` - Fast cloud inference

**Cloud APIs**:
- `day-1/basic_chatgpt.py` - OpenAI GPT models
- `day-1/basic_claude.py` - Anthropic Claude
- `day-1/basic_groq.py` - Groq (ultra-fast inference)
- `day-1/basic_mistral.py` - Mistral AI
- `day-1/basic_fireworks.py` - Fireworks AI

**Java/Kotlin Examples** (in `day-1/JVM/`):
- `GettingStartedOllama.java` / `.kt` - Local models in Java/Kotlin
- `GettingStartedLmStudio.java` / `.kt` - LM Studio in JVM
- `maven-example/` - Complete Maven project setup

**What you'll do**:
1. Start with `getting_started_ollama.py` - understand the basic pattern
2. Try different models and compare their responses
3. Experiment with model parameters (temperature, top_p)
4. (Optional) Try the Java/Kotlin versions if that's your background

**Key takeaways**:
- Multiple ways to access AI: local (private, free) vs cloud (faster, more powerful)
- All models use similar API patterns
- Temperature controls creativity vs consistency

---

### Session 3: Advanced Model Usage 🎨

**Concepts**: Vision models, prompt engineering, model configuration

**Files to explore**:
- `day-1/intelligent_character_recognition.py` - OCR with vision models
- `day-1/simple_code.py` - Code generation examples
- `day-1/logit_probabilities.py` - Understanding model confidence
- `day-1/ai_astrology.py` - Creative prompt engineering
- `day-1/ai_astrology_groq.py` - Same but with Groq API

**What you'll do**:
1. Run `intelligent_character_recognition.py` with images from `day-2/data/`
2. Try `simple_code.py` to generate code in different languages
3. Experiment with creative prompts in `ai_astrology.py`
4. Check `logit_probabilities.py` to see how confident models are

**Key takeaways**:
- Vision models can analyze images alongside text
- Good prompts make a huge difference in output quality
- Models assign probability scores to their predictions

---

### Session 4: Building RAG Systems 🔍

**Concepts**: Chunking, embeddings, vector search, context retrieval

**Files to explore**:

**Alice in Wonderland RAG Examples**:
- `day-2/rag_alice_in_wonderland.py` - Basic RAG implementation
- `day-2/rag_alice_simple.py` - Simplified version for learning
- `day-2/rag_alice_in_wonderland_chromadb.py` - **PRODUCTION VERSION** with persistent storage
- `day-2/rag_alice_in_wonderland_transformers.py` - Using Transformers library
- `day-2/alice_in_one_go.py` - Text processing utilities

**Grimm Fairy Tales RAG Examples (NEW)**:
- `day-2/rag_grimm_fairy_tales.py` - RAG for Grimm's fairy tales (local Ollama)
- `day-2/rag_grimm_fairy_tales_groq.py` - Same but using Groq cloud API
- `day-2/grimm_fairy_tales_rag_demo.py` - Interactive demo version

**Data**:
- `day-2/data/alice_in_wonderland.txt` - Alice in Wonderland text
- `day-2/Grimms-Fairy-Tales.txt` - **NEW**: English Grimm fairy tales collection
- `day-2/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt` - **NEW**: German Grimm fairy tales (multilingual RAG)

**What you'll do**:
1. Start with `rag_alice_simple.py` or `rag_alice_in_wonderland.py` to understand the basics
2. Progress to `rag_alice_in_wonderland_chromadb.py` for the production approach
3. Try the Grimm fairy tales examples to see RAG with story boundaries
4. Compare in-memory vs persistent storage approaches
5. Test multilingual capabilities with German fairy tales
6. Experiment with different chunk sizes and retrieval parameters
7. Compare local (Ollama) vs cloud (Groq) RAG performance
8. **Bring your own documents** and adapt the code!

**Key takeaways**:
- RAG = Retrieval-Augmented Generation (give models context they don't have)
- Process: Chunk text → Generate embeddings → Store in vector DB → Search → Generate answer
- ChromaDB provides persistent storage so you don't re-embed each time
- Chunking strategy significantly impacts retrieval quality
- Different text types need different chunking strategies (novels vs fairy tales vs technical docs)
- EmbeddingGemma uses task-specific prompts for better retrieval
- Multilingual embeddings enable cross-language RAG

**RAG Architecture**:
```
Your Document → Chunking → Embeddings → Vector DB (ChromaDB)
                                              ↓
User Question → Embedding → Similarity Search → Top Chunks
                                                     ↓
                                         LLM + Context → Answer
```

---

## Day 2: Advanced Applications & Agentic Systems

### Session 5: Text Analysis & Extraction 📊

**Concepts**: Sentiment analysis, data extraction, summarization, web scraping

**Files to explore**:

**Sentiment Analysis**:
- `day-2/analyse_sentiment_01.py` - Basic sentiment analysis
- `day-2/analyse_sentiment_02.py` - Advanced multi-class sentiment
- `day-2/analyse_sentiment_kaggle.py` - Real dataset analysis
- `day-2/sentiment_analysis_results.csv` - Results output
- `day-2/sentiment_analysis_results.png` - Visualizations

**Data Extraction & Summarization**:
- `day-2/data_extraction_ollama.py` - Extract structured data from text (local)
- `day-2/data_extraction_groq.py` - **NEW**: Extract & summarize using Groq API (cloud)
- `day-2/kaggle_summary_complete.py` - Summarize large datasets
- `day-2/scrape.py` - Basic web scraping
- `day-2/scrape_gdpr_article.py` - Extract legal text
- `day-2/gdpr_article_content.txt` - Sample legal text

**What you'll do**:
1. Run sentiment analysis on sample reviews
2. Try analyzing your own text data
3. Extract structured information from unstructured text
4. Compare local (Ollama) vs cloud (Groq) extraction performance
5. Scrape and summarize web content
6. Process the GDPR article to see legal text analysis

**Key takeaways**:
- LLMs excel at understanding and classifying text
- You can extract structured data from unstructured text reliably
- Summarization helps process large documents quickly
- Web scraping + LLM = powerful data extraction pipeline
- Cloud APIs (Groq) offer faster inference for production use
- Local models (Ollama) provide privacy and cost benefits

---

### Session 6: Structured Outputs 📋

**Concepts**: JSON generation, schema validation, reliable formatting

**Files to explore**:
- `day-2/formatted_response_example.py` - Generate structured JSON outputs
- `day-2/payroll.py` - Database interaction with structured data
- `day-2/payroll2.py` - Advanced payroll processing

**What you'll do**:
1. Run `formatted_response_example.py` to see JSON generation
2. Try `payroll.py` to see database + LLM integration
3. Modify schemas to extract different structured data
4. Experiment with validation and error handling

**Key takeaways**:
- Modern LLMs can generate reliable JSON/XML
- Structure your prompts to specify exact output formats
- Validation ensures data quality before database insertion
- Structured outputs enable LLMs to integrate with traditional systems

---

### Session 7: Vision & Multimodal AI 👁️

**Concepts**: Vision-language models, image understanding, OCR, visual reasoning

**Files to explore**:
- `day-2/visual_ollama.py` - **NEW**: Vision model testing framework with multiple image types
- `day-2/visual_ml_studio.py` - Alternative vision implementation
- `day-1/intelligent_character_recognition.py` - OCR with vision models

**Test Images** (in `day-2/data/`):
- `IMG_1.jpg` - Technical graphs (ASI camera specs)
- `IMG_2.jpg` - Bar charts and data visualization
- `IMG_3.jpg` - Urban photography
- `IMG_4.jpg` - Japanese menu (non-Latin script OCR)
- `IMG_5.jpg`, `IMG_8.jpg` - French menus
- `IMG_6.jpg` - Handwritten text (beer list)
- `IMG_7.jpg` - Restaurant receipt
- `IMG_9.jpg` - Sheet music (Debussy)
- `IMG_10.jpg` - Technical diagrams
- `IMG_11.jpg` - UI screenshots (button detection)
- `IMG_12.jpg` - Mathematical equations

**What you'll do**:
1. Run `visual_ollama.py` to test vision models on various image types
2. Try OCR with different scripts (English, Japanese, French)
3. Extract structured data from receipts and menus
4. Analyze technical graphs and extract specific values
5. Test mathematical equation recognition
6. Experiment with UI element detection and coordinates
7. Use your own images!

**Key takeaways**:
- Vision-language models combine text and image understanding
- Qwen3-VL models excel at multilingual OCR
- Can extract structured data from images (receipts, forms)
- Useful for document processing, chart reading, UI automation
- Different model sizes (2b vs 4b) trade speed for accuracy
- "Thinking" models show reasoning but need more tokens

---

### Session 8: Code Generation 💻

**Concepts**: Code completion, fill-in-the-middle, code understanding

**Files to explore**:
- `day-2/fill_in_middle.py` - Code infilling (complete partial code)
- `day-1/simple_code.py` - Generate complete programs

**What you'll do**:
1. Try `fill_in_middle.py` to see code completion in action
2. Generate code in different programming languages
3. Ask for code explanations and refactoring
4. Experiment with different coding models (qwen2.5-coder)

**Key takeaways**:
- Code generation models understand context before and after
- Fill-in-the-middle is how modern IDEs work (like GitHub Copilot)
- Specialized code models outperform general models on coding tasks
- LLMs can explain, debug, and refactor existing code

---

### Bonus: Text-to-Speech 🔊

**Concepts**: Converting AI-generated text to natural speech

**Files to explore**:
- `day-2/cartesia_tts_test.py` - Text-to-speech using Cartesia API

**What you'll do**:
1. Generate natural-sounding speech from text
2. Experiment with different voices and languages
3. Combine with LLM output for voice assistants

**Key takeaways**:
- TTS bridges AI text output with voice interfaces
- Multiple voice options for different use cases
- Useful for accessibility and voice-based applications

---

### Session 9: Function Calling & Tool Use 🛠️

**Concepts**: Tool calling, function definitions, autonomous actions

**Files to explore**:
- `day-2/simple_tool_call.py` - **START HERE**: Simple currency conversion tool example
- `day-2/ollama_function_support.py` - **COMPREHENSIVE**: Full function calling framework with multiple tools
- `day-2/ollama_function_results.csv` - Model compatibility test results

**What you'll do**:
1. **Start with `simple_tool_call.py`** - understand the basic pattern with a single tool
2. Progress to `ollama_function_support.py` for multiple tools and complex scenarios
3. Run tests to see which models support function calling
4. Add your own custom functions/tools
5. See how models decide when to call functions
6. Experiment with function chaining (one function's output → next function's input)
7. Test different models and compare their tool-calling abilities

**Available Functions in the Example**:
- `convert_to_roman_numerals()` - Number conversion
- `convert_fahrenheit_to_centigrade()` - Temperature conversion
- `day_of_the_week()` - Get current day
- `calc_trig_function()` - Trigonometry calculations
- `weather_tool()` - Weather lookup
- `time_tool()` - Current time
- `location_tool()` - GPS location
- `distance_tool()` - Distance calculation

**Key takeaways**:
- Function calling lets LLMs take actions in the real world
- Models analyze user requests and decide which function(s) to call
- Function definitions (schemas) tell the model what tools are available
- This is the foundation of agentic systems!

**Function Calling Flow**:
```
User: "What's 212°F in Celsius as a Roman numeral?"
  ↓
LLM decides: Need convert_fahrenheit_to_centigrade(212)
  ↓
Function returns: 100
  ↓
LLM decides: Need convert_to_roman_numerals(100)
  ↓
Function returns: "C"
  ↓
LLM responds: "212°F is C in Roman numerals"
```

---

### Session 10: Agentic Systems & MCP 🤖

**Concepts**: Autonomous AI agents, reasoning, tool orchestration, Model Context Protocol (MCP)

**Files to explore** (in `day-2/MCP/`):
- `README.md` - **START HERE**: Complete MCP documentation and setup guide
- `mcp_server.py` - MCP server with country info and weather tools
- `test_mcp_client_ollama.py` - Demo client using local Ollama models
- `test_mcp_client.py` - Alternative client implementation

**What you'll learn**:
- What makes a system "agentic"?
- How agents reason and plan
- Multi-step problem solving
- MCP (Model Context Protocol) - standardized tool interaction
- Building autonomous agents that can:
  - Understand complex goals
  - Plan sequences of actions
  - Use multiple tools in combination
  - Self-correct when needed
  - Chain tool calls (e.g., find capital → get weather)

**Agentic System Architecture**:
```
User Goal → Agent (LLM) → Reasoning Loop:
                            1. Understand goal
                            2. Plan approach
                            3. Use tools
                            4. Evaluate results
                            5. Adjust plan
                            6. Repeat until goal met
```

**What you'll do**:
1. Read `day-2/MCP/README.md` for comprehensive MCP introduction
2. Run the MCP demo to see agentic behavior in action
3. Watch the agent chain tool calls ("What's the weather in Germany's capital?")
4. Try adding your own tools to the MCP server
5. Experiment with different question complexities
6. Compare local (Ollama) vs cloud (Claude) agentic behavior
7. See how agents combine RAG + function calling + reasoning

**Key takeaways**:
- Agents = LLMs + Tools + Reasoning loop
- MCP provides standard way for agents to discover and use tools
- Agentic systems can solve complex, multi-step problems
- The future of AI is autonomous agents that can take action

---

## 🎯 Learning Path

### Complete Beginner Path
1. **Day 1 Session 1** → Understand tokens and embeddings
2. **Day 1 Session 2** → Connect to your first AI model
3. **Day 1 Session 4** → Build a simple RAG system
4. **Day 2 Session 5** → Analyze sentiment in text
5. **Day 2 Session 8** → Try function calling

### Experienced Developer Path
1. Review tokenization basics quickly
2. Jump to RAG with ChromaDB (production version)
3. Explore sentiment analysis and structured outputs
4. Deep dive into function calling framework
5. Build your own agentic system

### Project-Oriented Path
1. Identify your use case (e.g., document Q&A, code assistant, data extraction)
2. Find relevant examples in the repo
3. Adapt the code to your data/problem
4. Ask instructors for guidance during hands-on time

---

## 🔧 Key Technologies

### AI Models
- **Ollama**: Run LLMs locally (private, free, offline)
- **OpenAI GPT**: Most capable cloud models
- **Anthropic Claude**: Strong reasoning and safety
- **Groq**: Ultra-fast inference
- **Mistral**: Open-source, multilingual

### RAG & Vector Storage
- **ChromaDB**: Vector database for semantic search
- **Embeddings**: Convert text → numerical vectors

### Data Processing
- **BeautifulSoup**: Web scraping
- **Pandas**: Data manipulation
- **Matplotlib/Plotly**: Visualizations

### Development
- **Gradio**: Build interactive UIs
- **Ollama Python**: Easy local model access

---

## 📦 Repository Structure

```
mlcon-berlin-2025/
│
├── day-1/                          # Day 1: Fundamentals & RAG
│   ├── show_tokens.py             # START: See tokenization
│   ├── embedding_demo.py          # Understand embeddings
│   ├── getting_started_ollama.py  # First AI interaction
│   ├── basic_*.py                 # Various AI providers
│   ├── rag_*.py                   # RAG implementations
│   ├── JVM/                       # Java/Kotlin examples
│   └── ...
│
├── day-2/                          # Day 2: Advanced Applications
│   ├── analyse_sentiment_*.py     # Sentiment analysis (3 versions)
│   ├── data_extraction_ollama.py  # Structured extraction (local)
│   ├── data_extraction_groq.py    # Structured extraction (cloud)
│   ├── formatted_response_*.py    # JSON generation
│   ├── fill_in_middle.py          # Code completion
│   ├── simple_tool_call.py        # Simple function calling demo
│   ├── ollama_function_support.py # Advanced function calling
│   ├── visual_ollama.py           # Vision model testing framework
│   ├── visual_ml_studio.py        # Alternative vision implementation
│   ├── rag_alice_*.py             # Alice RAG (3 versions)
│   ├── rag_grimm_fairy_tales*.py  # Grimm tales RAG (2 versions)
│   ├── grimm_fairy_tales_rag_demo.py  # Interactive RAG demo
│   ├── scrape*.py                 # Web scraping examples
│   ├── payroll*.py                # Database integration
│   ├── Grimms-Fairy-Tales.txt     # English fairy tales
│   ├── Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt  # German fairy tales
│   ├── gdpr_article_content.txt   # Legal text sample
│   ├── data/                      # Sample datasets
│   │   ├── alice_in_wonderland.txt
│   │   └── IMG_*.jpg              # 12 vision test images
│   ├── chroma_db/                 # Vector database storage
│   └── MCP/                       # Model Context Protocol demos
│       ├── README.md              # Complete MCP guide
│       ├── mcp_server.py          # MCP server implementation
│       ├── test_mcp_client_ollama.py  # Ollama client
│       └── test_mcp_client.py     # Alternative client
│
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start Guide

### First 15 Minutes
```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Pull the core models we'll use
ollama pull qwen3:4b
ollama pull qwen3-vl:4b-instruct
ollama pull embeddinggemma

# 3. Test your first AI interaction
python day-1/getting_started_ollama.py

# 4. See tokenization in action
python day-1/show_tokens.py

# 5. Understand embeddings
python day-1/embedding_demo.py
```

### Next Hour
```bash
# Build your first RAG system
python day-2/rag_alice_in_wonderland.py

# Try sentiment analysis
python day-2/analyse_sentiment_01.py

# Test vision models
python day-2/visual_ollama.py

# Explore function calling (start simple)
python day-2/simple_tool_call.py

# Try advanced function calling
python day-2/ollama_function_support.py
```

### Rest of Workshop
- Test vision models with your own images
- Build RAG systems with your documents
- Try Grimm fairy tales in German and English
- Experiment with function calling and tool creation
- Explore MCP for agentic systems
- Compare local (Ollama) vs cloud (Groq) performance
- Build something you can take home!

---

## 💡 Pro Tips

### For Success in This Workshop
1. **Don't just read code - RUN IT**: Every file is designed to be executed
2. **Modify and experiment**: Change prompts, try different models, use your data
3. **Start simple**: Begin with basic examples before diving into complex ones
4. **Ask questions**: Instructors are here to help during hands-on sessions
5. **Take notes**: Document what works for your use case

### Performance Tips
- **Local models**: Smaller models (qwen3:4b, qwen3-vl:2b) are faster but less capable
- **Cloud models**: More expensive but more powerful and faster (Groq is ultra-fast!)
- **RAG systems**: 
  - Chunk size matters - experiment with 200-500 words
  - Use EmbeddingGemma with task-specific prompts for best results
  - Story-aware chunking for narrative texts (see Grimm tales example)
- **Vision models**: 
  - qwen3-vl:4b-instruct is the sweet spot (speed vs accuracy)
  - Use base64 encoding for images in API calls
  - Increase token limits for "thinking" models
- **Function calling**: Not all models support it equally - test before deploying
  - qwen3-vl:4b-instruct and qwen2.5:3b are reliable for function calling
  - See ollama_function_results.csv for model comparisons

### Common Issues
- **Ollama not responding**: Make sure `ollama serve` is running
- **Out of memory**: Use smaller models or reduce batch sizes
  - Vision: Use qwen3-vl:2b instead of 4b
  - Embeddings: Use all-minilm instead of embeddinggemma
- **Slow embeddings**: Use lightweight models like `all-minilm` for testing
- **Model hallucination**: Lower temperature (0.1-0.3) for factual tasks
- **Vision model errors**: 
  - Ensure images are properly base64 encoded
  - Check image file paths are correct
  - Some models need higher token limits for detailed analysis
- **Function calling not working**: 
  - Verify model supports function calling (not all do)
  - Check function schema format matches OpenAI spec
  - Try qwen3-vl:4b-instruct or qwen2.5:3b
- **MCP connection issues**: 
  - Ensure both server and client use stdio communication
  - Check that Ollama is running before starting MCP demo
  - Verify tool schemas are properly formatted

---

## 📖 Additional Resources

### Documentation
- **Ollama**: https://ollama.com/library - Browse available models
- **ChromaDB**: https://docs.trychroma.com - Vector database docs
- **Transformers**: https://huggingface.co/docs/transformers - HuggingFace library
- **OpenAI**: https://platform.openai.com/docs - API documentation
- **Anthropic**: https://docs.anthropic.com - Claude API docs

### Further Learning
- **Prompt Engineering Guide**: https://www.promptingguide.ai
- **RAG Techniques**: Research papers on retrieval-augmented generation
- **LangChain**: Framework for building LLM applications
- **AutoGPT**: Example of autonomous agents

---

## 🤝 Getting Help

### During the Workshop
- **Raise your hand** during hands-on sessions
- **Ask in chat** if remote/hybrid
- **Pair up** with other attendees
- **Check the examples** - most answers are in the code!

### After the Workshop
- **GitHub Issues**: Report bugs or ask questions
- **Community Discord/Slack**: (Link to be provided)
- **Instructor Contact**: (Details provided during workshop)

---

## 🎓 What's Next?

After completing this bootcamp, you can:

1. **Build RAG applications**: Create Q&A systems for your documents
2. **Develop AI agents**: Build tools that use function calling
3. **Deploy to production**: Use ChromaDB and cloud APIs for scale
4. **Integrate with your apps**: Add AI features to existing software
5. **Join the community**: Contribute to open-source AI projects

### Project Ideas
- **Document Q&A bot**: Upload PDFs, ask questions (use RAG + ChromaDB)
- **Multilingual RAG system**: Build cross-language document search
- **Receipt/Invoice processor**: Extract data from images (vision + structured output)
- **Code reviewer**: Analyze code for bugs and improvements
- **Data analyst agent**: Extract insights from CSV/JSON files
- **Customer support bot**: Automate responses with function calling + MCP
- **Content summarizer**: Condense articles, papers, reports
- **Visual documentation assistant**: Analyze diagrams and technical images
- **Agentic research assistant**: Combine web search + RAG + function calling

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

**Workshop Instructor**: John Davies (Ex-Chief Architect at Visa, JP Morgan, BNP Paribas)

**Event**: MLCon Berlin 2025

**Original Creation**: JAX London 2025

---

**Welcome to the bootcamp! Let's build something amazing together! 🚀**

For questions or issues during setup, please arrive early or reach out to the instructors.

Happy Learning! 🎉
