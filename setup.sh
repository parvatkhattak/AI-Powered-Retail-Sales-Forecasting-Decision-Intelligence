#!/bin/bash
# ──────────────────────────────────────────────────────────────
# setup.sh — One-command project environment bootstrap
# Usage: bash setup.sh
# ──────────────────────────────────────────────────────────────

set -e  # Exit immediately if any command fails

echo ""
echo "🛒 Retail AI Capstone — Environment Setup"
echo "──────────────────────────────────────────"

# ── Step 1: Create virtual environment ────────────────────────
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "   ✅ Virtual environment created at ./venv"

# ── Step 2: Upgrade pip ────────────────────────────────────────
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✅ pip upgraded"

# ── Step 3: Install dependencies ──────────────────────────────
echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "   ✅ All packages installed"

# ── Step 4: Create .env from template ─────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ .env file created from .env.example"
    echo ""
    echo "   ⚠️  ACTION REQUIRED: Open .env and add your OPENROUTER_API_KEY"
else
    echo "   ℹ️  .env already exists — skipping"
fi

# ── Step 5: Create required directories if missing ────────────
mkdir -p data/mocks models

# ── Step 6: Check for dataset files ───────────────────────────
echo ""
echo "🗂️  Checking for dataset files..."
if [ ! -f "data/train.csv" ]; then
    echo "   ⚠️  data/train.csv not found!"
    echo "      → Download from: https://www.kaggle.com/c/rossmann-store-sales/data"
    echo "      → Place train.csv and store.csv in the data/ folder"
else
    echo "   ✅ data/train.csv found"
fi

if [ ! -f "data/store.csv" ]; then
    echo "   ⚠️  data/store.csv not found!"
else
    echo "   ✅ data/store.csv found"
fi

# ── Done ───────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your API key to .env"
echo "  2. Place train.csv and store.csv in data/"
echo "  3. Run:  python src/data_pipeline.py"
echo "  4. Run:  python src/model_engine.py --train"
echo "  5. Run:  streamlit run app.py"
echo ""
