"""Dead-simple Gemma-4 tester (local Ollama OR llama-server). Stdlib only.

Usage:
    python gemma4_test.py                 -> asks the baked-in woodchuck question
    python gemma4_test.py "your prompt"   -> one-shot with your own prompt

Env overrides (so the same script tests a no-Ollama llama-server):
    OLLAMA_BASE_URL    e.g. http://localhost:8080/v1   (default: Ollama's 11434)
    GEMMA4_TEST_MODEL  e.g. gemma4:12b                 (default: unsloth Q4_K_M tag)

The ONE line that makes Gemma-4 work is reasoning_effort="none" below -- without
it Gemma-4 (a thinking model) burns its whole budget on <think> and returns blank.
(llama-server ignores the body field harmlessly; pass --reasoning off there.)
"""
import os
import sys
import json
import urllib.request

BASE_URL = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
URL = BASE_URL + "/chat/completions"
MODEL = os.environ.get("GEMMA4_TEST_MODEL") or "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"

# Baked-in test prompt (override by passing your own on the command line).
PROMPT = "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"

prompt = " ".join(sys.argv[1:]).strip() or PROMPT
print("Prompt: " + prompt)

body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.6,
    "max_tokens": 256,
    "reasoning_effort": "none",   # <-- the magic line (disables Gemma-4's <think>)
}

req = urllib.request.Request(
    URL,
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
)
resp = json.load(urllib.request.urlopen(req, timeout=180))
print("\nGemma-4: " + resp["choices"][0]["message"]["content"].strip())
