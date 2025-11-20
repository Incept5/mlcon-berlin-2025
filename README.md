
# MLCon Berlin 2025 - Hands-on GenAI Development Bootcamp 🚀

Welcome to the **2-Day Hands-on GenAI Development Bootcamp**! This repository contains all the code, examples, and exercises you'll work through during the workshop.

## 🎯 What You'll Learn

This bootcamp takes you from GenAI fundamentals to building production-ready AI systems:

**Day 1**: Understanding how LLMs work, connecting to models, and building RAG systems  
**Day 2**: Advanced applications including sentiment analysis, structured outputs, function calling, and agentic systems

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
ollama pull gemma3n:e4b       # 7.5GB High-quality reasoning model
ollama pull embeddinggemma    # 621MB For embeddings/RAG
ollama pull all-minilm        # 45MB Alternative embedding model
ollama pull qwen2.5-coder     # 4.7GB For code generation
```

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
- `day-2/rag_alice_in_wonderland.py` - Basic RAG implementation
- `day-2/rag_alice_in_wonderland_chromadb.py` - **PRODUCTION VERSION** with persistent storage
- `day-2/rag_alice_in_wonderland_transformers.py` - Using Transformers library
- `day-2/alice_in_on_go.py` - Text processing utilities

**Data**:
- `day-2/data/alice_in_wonderland.txt` - Sample text for RAG

**What you'll do**:
1. Start with `rag_alice_in_wonderland.py` to understand the basics
2. Progress to `rag_alice_in_wonderland_chromadb.py` for the production approach
3. Ask questions about Alice in Wonderland and see how RAG retrieves context
4. Experiment with different chunk sizes and retrieval parameters
5. **Bring your own documents** and adapt the code!

**Key takeaways**:
- RAG = Retrieval-Augmented Generation (give models context they don't have)
- Process: Chunk text → Generate embeddings → Store in vector DB → Search → Generate answer
- ChromaDB provides persistent storage so you don't re-embed each time
- Chunking strategy significantly impacts retrieval quality

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
- `day-2/data_extraction_ollama.py` - Extract structured data from text
- `day-2/kaggle_summary_complete.py` - Summarize large datasets
- `day-2/scrape.py` - Basic web scraping
- `day-2/scrape_gdpr_article.py` - Extract legal text
- `day-2/gdpr_article_content.txt` - Sample legal text

**What you'll do**:
1. Run sentiment analysis on sample reviews
2. Try analyzing your own text data
3. Extract structured information from unstructured text
4. Scrape and summarize web content
5. Process the GDPR article to see legal text analysis

**Key takeaways**:
- LLMs excel at understanding and classifying text
- You can extract structured data from unstructured text reliably
- Summarization helps process large documents quickly
- Web scraping + LLM = powerful data extraction pipeline

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

### Session 7: Code Generation 💻

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

### Session 8: Function Calling & Tool Use 🛠️

**Concepts**: Tool calling, function definitions, autonomous actions

**Files to explore**:
- `day-2/ollama_function_support.py` - **KEY FILE**: Function calling framework
- `day-2/ollama_function_results.csv` - Test results

**What you'll do**:
1. Study `ollama_function_support.py` to understand function calling
2. Run tests to see which models support function calling
3. Add your own custom functions/tools
4. See how models decide when to call functions
5. Experiment with function chaining (one function's output → next function's input)

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

### Session 9: Agentic Systems 🤖

**Concepts**: Autonomous AI agents, reasoning, tool orchestration, Model Context Protocol (MCP)

**What you'll learn**:
- What makes a system "agentic"?
- How agents reason and plan
- Multi-step problem solving
- MCP architecture for tool integration
- Building autonomous agents that can:
  - Understand goals
  - Plan sequences of actions
  - Use tools to accomplish tasks
  - Self-correct when needed

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
1. Discuss real-world agentic system examples
2. Build a simple agent workflow
3. Explore MCP (Model Context Protocol) for tool integration
4. See how agents combine RAG + function calling + reasoning

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
│   ├── analyse_sentiment_*.py     # Sentiment analysis
│   ├── data_extraction_ollama.py  # Structured extraction
│   ├── formatted_response_*.py    # JSON generation
│   ├── fill_in_middle.py          # Code completion
│   ├── ollama_function_support.py # Function calling (KEY!)
│   ├── rag_*_chromadb.py          # Production RAG
│   ├── data/                      # Sample datasets
│   │   ├── alice_in_wonderland.txt
│   │   └── IMG_*.jpg              # Vision model images
│   └── chroma_db/                 # Vector database storage
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

# 2. Test your first AI interaction
python day-1/getting_started_ollama.py

# 3. See tokenization in action
python day-1/show_tokens.py

# 4. Understand embeddings
python day-1/embedding_demo.py
```

### Next Hour
```bash
# Build your first RAG system
python day-2/rag_alice_in_wonderland.py

# Try sentiment analysis
python day-2/analyse_sentiment_01.py

# Explore function calling
python day-2/ollama_function_support.py
```

### Rest of Workshop
- Experiment with different models
- Try your own data
- Ask questions during hands-on time
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
- **Local models**: Smaller models (qwen3:4b) are faster but less capable
- **Cloud models**: More expensive but more powerful and faster
- **RAG systems**: Chunk size matters - experiment with 200-500 words
- **Function calling**: Not all models support it equally - test before deploying

### Common Issues
- **Ollama not responding**: Make sure `ollama serve` is running
- **Out of memory**: Use smaller models or reduce batch sizes
- **Slow embeddings**: Use lightweight models like `all-minilm` for testing
- **Model hallucination**: Lower temperature (0.1-0.3) for factual tasks

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
- **Document Q&A bot**: Upload PDFs, ask questions (use RAG)
- **Code reviewer**: Analyze code for bugs and improvements
- **Data analyst agent**: Extract insights from CSV/JSON files
- **Customer support bot**: Automate responses with function calling
- **Content summarizer**: Condense articles, papers, reports

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
