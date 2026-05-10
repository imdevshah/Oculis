# eval/logger.py
# Logs every query + result to a JSON file so we can analyze and visualize later.

import json
import os
from datetime import datetime
from config import CHROMA_PATH

LOG_PATH = "./eval/logs/oculis_log.json"

def _load():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return json.load(f)

def _save(data):
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def log_query(
    question: str,
    answer: str,
    confidence: float,
    flagged: bool,
    warning: str,
    consistency: float,
    nli: float,
    faithfulness: float,
    chunks: list         # list of chunk dicts from retriever
):
    """
    Call this after every answer() call to log the full result.
    chunks should be the raw list from retrieve() — each dict has
    text, source, page, type, similarity.
    """
    entry = {
        "timestamp":   datetime.now().isoformat(),
        "question":    question,
        "answer":      answer,
        "confidence":  confidence,
        "flagged":     flagged,
        "warning":     warning,
        "scores": {
            "consistency":  consistency,
            "nli":          nli,
            "faithfulness": faithfulness
        },
        "chunks": [
            {
                "text":       c["text"][:80],   # preview only
                "source":     c["source"],
                "page":       c["page"],
                "type":       c["type"],
                "similarity": c["similarity"]
            }
            for c in chunks
        ]
    }

    data = _load()
    data.append(entry)
    _save(data)
    print(f"[Logger] Logged query #{len(data)}: confidence={confidence}")
    return entry