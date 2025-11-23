# MLCon Berlin 2025 - Workshop Materials

This repository contains materials and examples for the MLCon Berlin 2025 workshop.

## Quick Start

### Setup Virtual Environment

```bash
# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Scripts

**Option 1: Use the helper script (Recommended)**
```bash
./run_with_venv.sh day-1/your_script.py
./run_with_venv.sh day-2/your_script.py
```

**Option 2: Activate environment manually**
```bash
source .venv/bin/activate
python day-1/your_script.py
```

### Important: Avoiding urllib3 Warnings

The project uses `urllib3 < 2.0` for compatibility. To avoid OpenSSL warnings:

1. **Always use the virtual environment** - use `./run_with_venv.sh` or activate `.venv` before running scripts
2. The helper script automatically uses the correct Python environment
3. Never run scripts with system Python directly (e.g., `python3 script.py` without activating venv)

## Project Structure

### Day 1
- AI and ML examples with various frameworks
- Embedding demonstrations
- Character recognition examples
- JVM examples (Java/Kotlin)

### Day 2
- ChromaDB integration
- Text analysis and sentiment detection
- Alice in Wonderland processing examples

## JVM Examples

See `day-1/JVM/` directory for Java/Kotlin examples:
- Maven-based examples
- Gradle-based examples
- Simple standalone examples

Refer to `day-1/JVM/README.md` for JVM-specific setup instructions.

## Dependencies

Main dependencies include:
- TensorFlow & PyTorch for ML
- Mistral AI, Ollama, Groq for LLM interactions
- ChromaDB for vector storage
- Gradio for UI components
- Various NLP and data processing libraries

See `requirements.txt` for complete list.

## Troubleshooting

### urllib3 Warning
If you see: `NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+`

**Solution:** Use `./run_with_venv.sh` to run your scripts, or ensure you've activated the virtual environment with `source .venv/bin/activate`

### Python Version
Requires Python 3.9 or later.

## License

See LICENSE file for details.
