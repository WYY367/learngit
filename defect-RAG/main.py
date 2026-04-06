"""Entry point for Defect RAG System."""
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.app import main

if __name__ == "__main__":
    main()
