"""Dead-simple Gemma-4 tester (local Ollama). No deps beyond Python stdlib.

Usage:
    python gemma4_test.py                 -> asks the baked-in woodchuck question
    python gemma4_test.py "your prompt"   -> one-shot with your own prompt

The ONE line that makes Gemma-4 work is reasoning_effort="none" below -- without
it Gemma-4 (a thinking model) burns its whole budget on <think> and returns blank.
"""
import sys
import json
import urllib.request

URL = "http://localhost:11434/v1/chat/completions"
MODEL = "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"  # change to your pulled tag

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
