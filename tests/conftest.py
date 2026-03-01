import sys
from pathlib import Path

# Adiciona a raiz do projeto no PYTHONPATH para permitir: import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
