"""
Data loaders for transcripts and knowledge base.
Centralizes paths and JSON loading so API and CLI share the same source.
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
TRANSCRIPTS_PATH = DATA_DIR / "transcripts.json"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"


def load_transcripts(path: Path | None = None) -> list[dict]:
    """Load transcripts from JSON. Defaults to data/transcripts.json."""
    p = path or TRANSCRIPTS_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base(path: Path | None = None) -> list[dict]:
    """Load knowledge base from JSON. Defaults to data/knowledge_base.json."""
    p = path or KNOWLEDGE_BASE_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)
