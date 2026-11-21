#!/usr/bin/env python3
"""
Local development runner for ScamGraph.
Usage: python run_local.py
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def check_venv():
    if not VENV_PYTHON.exists():
        print("❌ Virtual environment not found. Run setup first.")
        sys.exit(1)


def install_deps():
    print("📦 Installing dependencies...")
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"], check=True)


def train_model():
    print("🧠 Training model (this may take a few minutes)...")
    subprocess.run([str(VENV_PYTHON), "app/model/train.py"], check=True)


def run_server():
    print("🚀 Starting server at http://localhost:8000")
    print("   Press Ctrl+C to stop")
    os.chdir(PROJECT_ROOT)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ScamGraph local development")
    parser.add_argument("command", choices=["install", "train", "run", "all"], nargs="?", default="all")
    args = parser.parse_args()

    check_venv()

    if args.command in ("install", "all"):
        install_deps()

    if args.command in ("train", "all"):
        train_model()

    if args.command in ("run", "all"):
        run_server()


if __name__ == "__main__":
    main()