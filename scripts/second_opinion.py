"""
second_opinion.py
=================
Optional ChatGPT consultation for the OTR smoke/sanity agent.

When the smoke agent is uncertain about a bug diagnosis or fix, it can
call `ask_chatgpt()` with the flag type, relevant code block, and a
question.  ChatGPT acts as a second-opinion code reviewer — the agent
still makes the final decision.

SETUP:
    Set the environment variable OPENAI_API_KEY before running.

    Windows (cmd):
        set OPENAI_API_KEY=sk-proj-...
    Windows (PowerShell):
        $env:OPENAI_API_KEY = "sk-proj-..."
    Linux:
        export OPENAI_API_KEY="sk-proj-..."

    The key is NEVER stored in any file in this repository.

Usage (from smoke agent or interactive):
    python scripts/second_opinion.py --flag TITLE_STUCK --code "def foo(): ..." --question "Why?"

    Or as a library:
        from second_opinion import ask_chatgpt
        response = ask_chatgpt("TITLE_STUCK", code_block, "What is the root cause?")

AntiGravity: this module is agent-only tooling.  It does not run during
episode generation and has zero impact on the audio pipeline.
"""

import os
import sys
import json
import argparse
import textwrap

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
MAX_TOKENS = 800
TEMPERATURE = 0.3  # low temp for precise diagnostic answers

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior Python developer reviewing code from a ComfyUI custom node
    project called OldTimeRadio (SIGNAL LOST).  The project generates radio drama
    episodes using LLM script generation, Bark TTS, and audio post-processing.

    Platform: Windows, RTX 5080 Laptop, 16 GB VRAM, Python 3.12, torch 2.10.0.

    When asked about a bug:
    1. State the most likely root cause in one sentence.
    2. Suggest the minimal fix (fewest lines changed).
    3. Flag any risk the fix introduces.

    Be terse.  No preamble.  No markdown fences unless showing code.
""")


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
def _get_api_key():
    """Read OPENAI_API_KEY from environment.  Never from a file."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return key


def ask_chatgpt(flag_type, code_block, question, model=MODEL):
    """Send a diagnostic question to ChatGPT and return the response text.

    Args:
        flag_type:  The smoke-check flag (e.g. "TITLE_STUCK", "ALL_SAME_GENDER").
        code_block: The relevant code snippet (~30-60 lines).  Can be empty.
        question:   The specific question for the reviewer.
        model:      OpenAI model ID (default: gpt-4o).

    Returns:
        str: ChatGPT's response text, or an error message starting with
             "ERROR:" if the call failed.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "ERROR: OPENAI_API_KEY not set.  Set the environment variable:\n"
            "  cmd:  set OPENAI_API_KEY=sk-proj-...\n"
            "  PS:   $env:OPENAI_API_KEY = \"sk-proj-...\"\n"
            "Second opinion is optional — the smoke agent can proceed without it."
        )

    user_message = f"FLAG: {flag_type}\n"
    if code_block and code_block.strip():
        # Truncate code to ~3000 chars to stay within reasonable context
        truncated = code_block.strip()[:3000]
        user_message += f"\nCODE:\n```python\n{truncated}\n```\n"
    user_message += f"\nQUESTION: {question}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        choices = body.get("choices", [])
        if not choices:
            return "ERROR: ChatGPT returned no choices."
        return choices[0]["message"]["content"].strip()

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return f"ERROR: HTTP {e.code} from OpenAI API.\n{error_body}"

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ask ChatGPT for a second opinion on a smoke-check flag."
    )
    parser.add_argument("--flag", required=True,
                        help="Flag type, e.g. TITLE_STUCK")
    parser.add_argument("--code", default="",
                        help="Relevant code block (or path to a .py file)")
    parser.add_argument("--question", required=True,
                        help="Specific diagnostic question")
    parser.add_argument("--model", default=MODEL,
                        help=f"OpenAI model (default: {MODEL})")
    args = parser.parse_args()

    # If --code looks like a file path, read it
    code = args.code
    if code and os.path.isfile(code):
        try:
            with open(code, encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            code = f"(Could not read {code}: {e})"

    print(f"\nAsking {args.model} about {args.flag}...\n")
    response = ask_chatgpt(args.flag, code, args.question, model=args.model)
    print(response)
    print()


if __name__ == "__main__":
    main()
