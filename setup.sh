#!/usr/bin/env bash
# ============================================================
# ET AI News Platform — One-Shot Setup Script
# Run this ONCE before launching the Streamlit app
# ============================================================
set -e

echo "============================================="
echo "  ET AI-Native News Platform — Setup"
echo "============================================="

# --- 1. Install Ollama (system-level, CPU-optimised) --------
if ! command -v ollama &> /dev/null; then
    echo "[1/5] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1/5] Ollama already installed — skipping."
fi

# --- 2. Start Ollama daemon in background -------------------
echo "[2/5] Starting Ollama service..."
ollama serve &>/tmp/ollama.log &
OLLAMA_PID=$!
sleep 4   # Wait for daemon to be ready

# --- 3. Pull quantized models (4-bit, CPU-friendly) ---------
echo "[3/5] Pulling quantized LLM (llama3.2:3b — ~2 GB)..."
ollama pull llama3.2:3b

echo "      Pulling embedding model via Ollama (nomic-embed-text)..."
ollama pull nomic-embed-text

# --- 4. Install Python dependencies -------------------------
echo "[4/5] Installing Python requirements (CPU-only torch)..."
pip install --upgrade pip
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# --- 5. Verify setup ----------------------------------------
echo "[5/5] Verifying installation..."
python -c "
import lancedb, networkx, langchain, streamlit
from sentence_transformers import SentenceTransformer
print('  ✓ Core libraries OK')
import ollama
print('  ✓ Ollama Python client OK')
print()
print('Setup complete! Run: streamlit run app.py')
"

echo "============================================="
echo "  Setup finished. Launch with:"
echo "  streamlit run app.py"
echo "============================================="
