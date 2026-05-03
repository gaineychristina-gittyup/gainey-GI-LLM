#!/usr/bin/env bash
# One-shot setup for the GI guidelines RAG prototype on macOS.
#
# Run this from inside the gainey-GI-LLM directory:
#   bash scripts/setup_macos.sh
#
# What it does (in order):
#   1. Installs Homebrew if missing
#   2. Installs Docker Desktop, Python 3.11, git, poppler, tesseract
#   3. Creates a .venv and installs all Python dependencies
#   4. Copies .env.example to .env (you fill in keys after)
#   5. Reminds you to open Docker Desktop, then brings up pgvector
#   6. Verifies the database tables were created
#
# Things this script CAN'T do for you:
#   - Open Docker Desktop the first time (you have to click it once)
#   - Get your Anthropic / Voyage API keys (you have to sign up)
#   - Move a PDF into data/raw_pdfs/AGA/ (you have to download one)
#
# Safe to re-run. Each step checks if it's already done.

set -e

# ---- pretty printing ----------------------------------------------------
green()  { printf "\033[1;32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$1"; }
red()    { printf "\033[1;31m%s\033[0m\n" "$1"; }
step()   { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

# ---- 0. sanity ----------------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
  red "This script is for macOS only."
  exit 1
fi

if [[ ! -f "pyproject.toml" || ! -d "src/ingest" ]]; then
  red "Run this from inside the gainey-GI-LLM directory."
  red "Try: cd ~/gainey-GI-LLM && bash scripts/setup_macos.sh"
  exit 1
fi

# ---- 1. Homebrew --------------------------------------------------------
step "Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  yellow "Homebrew not found — installing (will prompt for your password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Apple Silicon: brew installs into /opt/homebrew and isn't on PATH yet
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
else
  green "Homebrew already installed."
fi

# ---- 2. system tools ----------------------------------------------------
step "Installing system dependencies via Homebrew..."
# Docker Desktop is a cask; the rest are formulae
if ! brew list --cask docker >/dev/null 2>&1; then
  yellow "Installing Docker Desktop (cask)..."
  brew install --cask docker
else
  green "Docker Desktop already installed."
fi

for pkg in python@3.11 git poppler tesseract; do
  if brew list "$pkg" >/dev/null 2>&1; then
    green "$pkg already installed."
  else
    yellow "Installing $pkg..."
    brew install "$pkg"
  fi
done

# ---- 3. python venv -----------------------------------------------------
step "Creating Python virtual environment..."
PYBIN="$(brew --prefix python@3.11)/bin/python3.11"
if [[ ! -x "$PYBIN" ]]; then
  PYBIN="$(command -v python3.11 || true)"
fi
if [[ -z "$PYBIN" ]]; then
  red "Could not locate python3.11 binary — investigate before re-running."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  "$PYBIN" -m venv .venv
  green "Created .venv"
else
  green ".venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet

step "Installing Python dependencies (this takes 5-10 min on the first run)..."
pip install -e ".[dev]" --quiet
green "Python dependencies installed."

# ---- 4. .env scaffolding ------------------------------------------------
step "Setting up .env file..."
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  yellow "Created .env from .env.example."
  yellow "Edit it now: open -a TextEdit .env"
  yellow "  - Paste your Anthropic key after ANTHROPIC_API_KEY="
  yellow "  - Paste your Voyage key after VOYAGE_API_KEY="
else
  green ".env already exists (not overwriting)."
fi

# ---- 5. Docker check ----------------------------------------------------
step "Checking Docker Desktop..."
if ! docker info >/dev/null 2>&1; then
  yellow "Docker Desktop is not running."
  yellow "  1. Open Docker Desktop (from your Applications folder)."
  yellow "  2. Wait for the whale icon in the menu bar to stop animating."
  yellow "  3. Then re-run this script (it will pick up where it left off)."
  exit 0
fi
green "Docker is running."

step "Bringing up pgvector container..."
docker compose up -d

# Give Postgres a few seconds to apply the init script
sleep 5

step "Verifying database schema..."
if docker compose exec -T postgres psql -U gi -d gi_guidelines -c "\dt" 2>/dev/null | grep -qE "documents|chunks"; then
  green "Tables are present in the database."
else
  yellow "Tables not yet visible. Waiting another 5s and retrying..."
  sleep 5
  docker compose exec -T postgres psql -U gi -d gi_guidelines -c "\dt" || true
fi

# ---- 6. final summary ---------------------------------------------------
echo
green "========================================================"
green "Setup complete."
green "========================================================"
echo
echo "Next steps:"
echo "  1. Make sure your .env has both keys filled in."
echo "  2. Drop a guideline PDF into data/raw_pdfs/AGA/."
echo "  3. Run the ingest:"
echo
echo "     source .venv/bin/activate"
echo "     python -m src.ingest.build_index \\"
echo "         --pdf data/raw_pdfs/AGA/<your-file>.pdf \\"
echo "         --society AGA \\"
echo "         --title \"<title from the PDF>\" \\"
echo "         --year 2025 \\"
echo "         --topic gastroparesis"
echo
