"""Smoke-test the native OTR Gemma 4 12B GGUF writer lane.

Usage:
    python docs/gemma4/gemma4_test.py
    python docs/gemma4/gemma4_test.py "your prompt"

This loads the same in-process backend used by the writer dropdown row
`unsloth/gemma-4-12b-it-GGUF`. It does not use Ollama, llama-server, or
localhost HTTP.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from nodes import _otr_gguf_backend as gguf


PROMPT = "Write one vivid sentence about an old radio crackling to life."


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or PROMPT
    print("Prompt: " + prompt)
    status = gguf.validate_gemma_gguf_ready()
    print("Readiness: " + repr(status))
    if not status["ok"]:
        return 2

    backend = gguf.GGUFNativeBackend()
    row = types.SimpleNamespace(context_window=gguf.DEFAULT_CONTEXT_WINDOW)
    cache_entry = backend.load(gguf.ROW_ID, row)
    try:
        out = backend.generate(
            cache_entry,
            [{"role": "user", "content": prompt}],
            temperature=0.6,
            max_new_tokens=96,
        )
        print("\nGemma 4 12B: " + out.strip())
        return 0
    finally:
        backend.unload(cache_entry)


if __name__ == "__main__":
    raise SystemExit(main())
