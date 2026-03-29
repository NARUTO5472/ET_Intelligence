#!/usr/bin/env bash
# =============================================================
# apply_fixes.sh — patches the ET AI News Platform in-place
# Run from the project root:  bash apply_fixes.sh
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(pwd)"

echo "============================================="
echo "  ET AI Platform — Applying Fixes"
echo "  Project: $PROJECT_DIR"
echo "============================================="

# Sanity check — make sure we're in the right directory
if [ ! -f "$PROJECT_DIR/app.py" ]; then
    echo "ERROR: app.py not found in $PROJECT_DIR"
    echo "Please run this script from the et_news_platform directory."
    exit 1
fi

# Back up originals (safe to run multiple times)
BACKUP_DIR="$PROJECT_DIR/.backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/llm" "$BACKUP_DIR/modules"
cp "$PROJECT_DIR/config.py"                  "$BACKUP_DIR/config.py"
cp "$PROJECT_DIR/app.py"                     "$BACKUP_DIR/app.py"
cp "$PROJECT_DIR/llm/ollama_client.py"       "$BACKUP_DIR/llm/ollama_client.py"
cp "$PROJECT_DIR/modules/news_navigator.py"  "$BACKUP_DIR/modules/news_navigator.py"
cp "$PROJECT_DIR/modules/story_arc.py"       "$BACKUP_DIR/modules/story_arc.py"
cp "$PROJECT_DIR/modules/my_et.py"           "$BACKUP_DIR/modules/my_et.py"
echo "  ✓ Originals backed up to $BACKUP_DIR"

# Apply fixes
cp "$SCRIPT_DIR/config.py"                  "$PROJECT_DIR/config.py"
cp "$SCRIPT_DIR/app.py"                     "$PROJECT_DIR/app.py"
cp "$SCRIPT_DIR/llm/ollama_client.py"       "$PROJECT_DIR/llm/ollama_client.py"
cp "$SCRIPT_DIR/modules/news_navigator.py"  "$PROJECT_DIR/modules/news_navigator.py"
cp "$SCRIPT_DIR/modules/story_arc.py"       "$PROJECT_DIR/modules/story_arc.py"
cp "$SCRIPT_DIR/modules/my_et.py"           "$PROJECT_DIR/modules/my_et.py"
echo "  ✓ All 6 files patched"

echo ""
echo "============================================="
echo "  Done! Launch with:  streamlit run app.py"
echo "============================================="
