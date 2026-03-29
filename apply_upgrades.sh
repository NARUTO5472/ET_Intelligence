#!/usr/bin/env bash
# ================================================================
# apply_upgrades.sh — applies all speed + capacity upgrades
# Run from the et_news_platform directory:
#   bash apply_upgrades.sh
#
# What this does:
#   1. Installs the groq Python package (free, 30s)
#   2. Backs up your original files
#   3. Copies all upgraded files into place
#   4. Prints Groq setup instructions
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"

echo "============================================="
echo "  ET AI Platform — Speed & Capacity Upgrade"
echo "============================================="

if [ ! -f "$PROJECT_DIR/app.py" ]; then
    echo "ERROR: app.py not found in $PROJECT_DIR"
    echo "Run from the et_news_platform directory."
    exit 1
fi

# --- 1. Install Groq package ------------------------------------------------
echo "[1/4] Installing groq Python package…"
pip install groq --quiet
echo "  ✓ groq installed"

# --- 2. Back up originals ---------------------------------------------------
BACKUP_DIR="$PROJECT_DIR/.backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/llm" "$BACKUP_DIR/modules" "$BACKUP_DIR/ingestion" "$BACKUP_DIR/utils"

for f in app.py config.py requirements.txt; do
    [ -f "$PROJECT_DIR/$f" ] && cp "$PROJECT_DIR/$f" "$BACKUP_DIR/$f"
done
for f in llm/ollama_client.py modules/news_navigator.py ingestion/rss_fetcher.py utils/ingestion_orchestrator.py; do
    [ -f "$PROJECT_DIR/$f" ] && cp "$PROJECT_DIR/$f" "$BACKUP_DIR/$f"
done
echo "  ✓ Originals backed up to $BACKUP_DIR"

# --- 3. Apply upgrades -------------------------------------------------------
echo "[3/4] Applying upgraded files…"

cp "$SCRIPT_DIR/config.py"                        "$PROJECT_DIR/config.py"
cp "$SCRIPT_DIR/app.py"                           "$PROJECT_DIR/app.py"
cp "$SCRIPT_DIR/requirements.txt"                 "$PROJECT_DIR/requirements.txt"
cp "$SCRIPT_DIR/llm/ollama_client.py"             "$PROJECT_DIR/llm/ollama_client.py"
cp "$SCRIPT_DIR/modules/news_navigator.py"        "$PROJECT_DIR/modules/news_navigator.py"
cp "$SCRIPT_DIR/ingestion/rss_fetcher.py"         "$PROJECT_DIR/ingestion/rss_fetcher.py"
cp "$SCRIPT_DIR/utils/ingestion_orchestrator.py"  "$PROJECT_DIR/utils/ingestion_orchestrator.py"

echo "  ✓ 7 files upgraded"

# --- 4. Groq setup instructions ---------------------------------------------
echo ""
echo "[4/4] GROQ SETUP (takes 30 seconds, free forever)"
echo ""
echo "  1. Go to: https://console.groq.com"
echo "  2. Sign up / log in"
echo "  3. Click 'API Keys' → 'Create API Key'"
echo "  4. Copy the key (starts with gsk_...)"
echo "  5. Run:  export GROQ_API_KEY='gsk_...'"
echo "     Or add to .env:  GROQ_API_KEY=gsk_..."
echo ""
echo "  With Groq: briefings in ~15 seconds"
echo "  Without:   briefings in ~4-8 minutes (Ollama CPU fallback)"
echo ""
echo "============================================="
echo "  Done! Launch with:  streamlit run app.py"
echo "============================================="
