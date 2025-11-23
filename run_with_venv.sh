#!/bin/bash
# Helper script to run Python scripts with the virtual environment activated

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Please create it first with: python3 -m venv .venv"
    exit 1
fi

# Check if a script was provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: ./run_with_venv.sh <python_script.py> [args...]${NC}"
    echo "Example: ./run_with_venv.sh day-1/ai_astrology_groq.py"
    exit 1
fi

# Activate virtual environment and run the script
echo -e "${GREEN}✓ Using virtual environment: $VENV_PATH${NC}"
source "$VENV_PATH/bin/activate"

# Verify urllib3 version
URLLIB3_VERSION=$(pip show urllib3 2>/dev/null | grep Version | cut -d' ' -f2)
echo -e "${GREEN}✓ urllib3 version: $URLLIB3_VERSION${NC}"

# Run the Python script with all arguments
echo -e "${GREEN}✓ Running: python $@${NC}"
echo ""
python "$@"
