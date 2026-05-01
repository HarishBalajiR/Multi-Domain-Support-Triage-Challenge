"""Local Ollama client wrapper with deterministic defaults."""

from __future__ import annotations

import json
from typing import Any

from ollama import Client

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, RANDOM_SEED


def run_llm_json(prompt: str, *, system_prompt: str) -> dict[str, Any]:
    """Run Ollama and return parsed JSON output."""
    client = Client(host=OLLAMA_BASE_URL, timeout=90.0)
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            options={
                "temperature": 0,
                "seed": RANDOM_SEED,
            },
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response["message"]["content"]
        if not content:
            return {}
        return json.loads(content)
    except Exception:
        return {}
