#!/bin/bash
# Run Token Probability Analyzer with Ollama models
#
# Usage:
#   ./run_logit_analyzer.sh                    # Interactive selection
#   ./run_logit_analyzer.sh qwen3:0.6b        # Specify model
#   ./run_logit_analyzer.sh --list            # List available models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")"

# Handle --list flag
if [ "$1" == "--list" ]; then
    python3 day-1/find_ollama_models.py
    exit 0
fi

# If model specified, use it
if [ -n "$1" ]; then
    MODEL="$1"
    echo -e "${BLUE}🔍 Looking up model: $MODEL${NC}"
    
    # Get model path
    MODEL_PATH=$(python3 day-1/find_ollama_models.py --model "$MODEL" 2>/dev/null | grep "📁 Path:" | cut -d' ' -f3)
    
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}❌ Model '$MODEL' not found!${NC}"
        echo ""
        echo "Available models:"
        python3 day-1/find_ollama_models.py | grep "^[a-z]" | head -20
        exit 1
    fi
    
    echo -e "${GREEN}✅ Found model${NC}"
    echo -e "${BLUE}📁 Path: $MODEL_PATH${NC}"
    echo ""
else
    # Interactive selection
    echo -e "${BLUE}🤖 Select a model for Token Probability Analysis${NC}"
    echo ""
    echo "Recommended models for testing:"
    echo ""
    echo "  1) qwen3:0.6b          (498 MB)  - Smallest, fastest"
    echo "  2) llama3.2:1b         (1.2 GB)  - Good balance"
    echo "  3) gemma3:latest       (3.1 GB)  - High quality"
    echo "  4) mistral:latest      (4.1 GB)  - Very capable"
    echo ""
    echo "  L) List all 26 available models"
    echo "  Q) Quit"
    echo ""
    read -p "Enter choice (1-4, L, Q): " choice
    
    case $choice in
        1)
            MODEL="qwen3:0.6b"
            MODEL_PATH="/Users/iain/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa"
            ;;
        2)
            MODEL="llama3.2:1b"
            MODEL_PATH="/Users/iain/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
            ;;
        3)
            MODEL="gemma3:latest"
            MODEL_PATH="/Users/iain/.ollama/models/blobs/sha256-aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25"
            ;;
        4)
            MODEL="mistral:latest"
            MODEL_PATH="/Users/iain/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
            ;;
        L|l)
            echo ""
            python3 day-1/find_ollama_models.py
            echo ""
            read -p "Enter model name (e.g., qwen3:0.6b): " MODEL
            MODEL_PATH=$(python3 day-1/find_ollama_models.py --model "$MODEL" 2>/dev/null | grep "📁 Path:" | cut -d' ' -f3)
            
            if [ -z "$MODEL_PATH" ]; then
                echo -e "${RED}❌ Model '$MODEL' not found!${NC}"
                exit 1
            fi
            ;;
        Q|q)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model file not found at: $MODEL_PATH${NC}"
    exit 1
fi

# Export model path and run
export GGUF_MODEL_PATH="$MODEL_PATH"
export GGML_METAL_LOG_LEVEL=1  # Reduce Metal logging noise

echo ""
echo -e "${GREEN}🚀 Launching Token Probability Analyzer${NC}"
echo -e "${BLUE}📦 Model: $MODEL${NC}"
echo -e "${BLUE}📁 Path: $MODEL_PATH${NC}"
echo ""
echo -e "${YELLOW}The web interface will open at http://127.0.0.1:7860${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the analyzer
python3 day-1/logit_probabilities.py
