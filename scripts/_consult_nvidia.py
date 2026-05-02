"""
_consult_nvidia.py  --  NVIDIA Build / NIM second-opinion client.
==================================================================

Purpose
-------
Add NVIDIA's hosted models (Mistral-Nemotron, Mistral-Large-3, Step-3.5-Flash,
Llama-3.3-Nemotron, etc.) as a third voice in the OTR round-robin
consultation pattern.  NVIDIA's catalog is OpenAI-compatible at
    https://integrate.api.nvidia.com/v1/chat/completions

Usage
-----
Standalone CLI:
    python scripts/_consult_nvidia.py --question-text "your question"

Importable from _consult_round_robin.py:
    from scripts._consult_nvidia import call_nvidia, DEFAULT_MODELS

Safety
------
- Reads NVIDIA_API_KEY fresh from HKCU\\Environment on every call.
- No telemetry, no third-party libraries, urllib stdlib only.
- Internal QA only, NOT shipped with episodes.

Tier
----
Free tier today is up to 40 RPM with no token cap on most catalog models
(per Jeffrey's NVIDIA Build account, 2026-04-29).  No worries about cost.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSULT_BASE = ROOT / "docs"

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT_SEC = 180

# Default ladder rebuilt 2026-05-01 EVENING per Jeffrey's routing
# guidance for OTR consultations (architecture / code / bug fixes).
#   1. Nemotron 49B  -- primary for agentic and code-pipeline tasks,
#                       optimized for iterative tool-use + terminal
#                       commands. Right tool for OTR's
#                       architecture-decision consults.
#   2. Qwen3 235B    -- heavy MoE reasoner. Bring in only if Nemotron
#                       can't crack a logic error. Slow but deep.
#   3. Mistral-Nemotron -- familiar baseline (matches OTR's local
#                       Mistral-Nemo flavor, useful as a third voice
#                       that thinks like the local writer LLM).
# Older fallbacks dropped: meta/llama-3.3-70b-instruct,
# mistralai/mistral-large-3-675b, stepfun-ai/step-3.5-flash. Those
# weren't recommended specifically and the top three cover OTR's
# needs without diluting the round-robin diversity.
DEFAULT_MODELS = [
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "mistralai/mistral-nemotron",
]


def _read_env_var(name: str) -> str:
    """Read a User-scope env var fresh from HKCU\\Environment."""
    try:
        import winreg  # type: ignore
    except ImportError:
        raise RuntimeError(
            f"winreg not available -- this script requires Windows. "
            f"Set {name} as a User env var via `setx {name} \"...\"`."
        )
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        try:
            value, _ = winreg.QueryValueEx(k, name)
        except FileNotFoundError:
            raise RuntimeError(
                f"{name} not found in HKCU\\Environment. "
                f"Run: setx {name} \"your-key-here\""
            )
    if not value:
        raise RuntimeError(f"{name} is empty.")
    return value


def call_nvidia(
    prompt: str,
    system: str,
    api_key: str,
    models: list[str] | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """
    POST to NVIDIA NIM chat/completions.  OpenAI-compatible schema.

    Returns (model_used, response_text).
    Raises RuntimeError if every candidate model fails.
    """
    candidates = list(models or DEFAULT_MODELS)
    last_err = ""
    for model in candidates:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                last_err = f"{model} -> empty choices: {str(data)[:300]}"
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            if not content.strip():
                last_err = f"{model} -> empty content"
                continue
            return (model, content)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"{model} -> HTTP {e.code}: {err_body[:300]}"
            if e.code in (400, 404) and (
                "model" in err_body.lower() or "not_found" in err_body.lower()
                or "not found" in err_body.lower()
            ):
                print(f"[nvidia] {model} unavailable, trying next...", file=sys.stderr)
                continue
            if e.code == 429:
                print(f"[nvidia] {model} rate-limited (40 RPM), trying next...", file=sys.stderr)
                continue
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
    raise RuntimeError(f"All NVIDIA models failed. Last error: {last_err}")


SYSTEM_PROMPT = (
    "You are a senior systems architect advising a solo developer on a "
    "ComfyUI radio-drama generator (OTR \"SIGNAL LOST\") running on a single "
    "RTX 5080 Laptop / 16 GB VRAM Windows workstation. 100% local, no cloud "
    "for the shipped pipeline. Audio output must remain byte-identical "
    "between runs (rule C7). VRAM ceiling is 14.5 GB. The owner does NOT "
    "want low-level VRAM optimization work. Prefer the smallest change "
    "with the largest payoff. Cite specific files / line numbers when "
    "relevant. Be candid; flag uncertainty rather than bluffing."
)


def _read_question(args: argparse.Namespace) -> str:
    if args.question_text:
        return args.question_text.strip()
    if args.question:
        return Path(args.question).read_text(encoding="utf-8").strip()
    print("Enter your question. End with Ctrl-Z then Enter:", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA NIM consultation.")
    parser.add_argument("--question", type=str)
    parser.add_argument("--question-text", type=str)
    parser.add_argument("--topic", type=str, default="nvidia-consult")
    parser.add_argument("--model", type=str,
                        help="Force a specific NVIDIA model id (skip ladder).")
    args = parser.parse_args()

    question = _read_question(args)
    if not question:
        print("ERROR: empty question.", file=sys.stderr)
        return 1

    api_key = _read_env_var("NVIDIA_API_KEY")
    models = [args.model] if args.model else DEFAULT_MODELS

    print(f"[nvidia] ladder: {models}", file=sys.stderr)
    t0 = time.time()
    model_used, response = call_nvidia(
        prompt=question, system=SYSTEM_PROMPT,
        api_key=api_key, models=models,
    )
    elapsed = time.time() - t0
    print(f"[nvidia] OK: {model_used} in {elapsed:.1f}s", file=sys.stderr)

    today = datetime.date.today().isoformat()
    out_dir = CONSULT_BASE / f"{today}-{args.topic}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nvidia.md"
    out_file.write_text(
        f"# NVIDIA ({model_used}) elapsed={elapsed:.1f}s\n\n{response}\n",
        encoding="utf-8",
    )
    print(str(out_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
