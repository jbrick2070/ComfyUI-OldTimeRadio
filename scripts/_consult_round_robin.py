"""
_consult_round_robin.py  --  Round-robin design / bug-fix consultation.
=======================================================================

Purpose
-------
Route a tough design question or bug investigation through multiple
external models until a grounded answer emerges:

    Round A:  ChatGPT (gpt-5.4 -> gpt-5 -> gpt-4.1 -> gpt-4o ladder)
    Round B:  Gemini   (gemini-3.1-pro-preview -> gemini-2.5-pro)
    Round C:  Synthesis -- this script summarizes agreement / disagreement
              between A and B so I (Claude) can decide the grounded answer
              in the chat context after running this script.

Per CLAUDE.md "Round-Robin Consultation" rule.  Use for non-trivial
design choices, library picks, architecture trade-offs, or bug root
causes that aren't obvious from a stack trace.  Skip for one-line fixes.

Usage
-----
    # interactive (prompts for question on stdin)
    python scripts/_consult_round_robin.py

    # with a question file
    python scripts/_consult_round_robin.py --question docs/consultations/q.md

    # with an inline question
    python scripts/_consult_round_robin.py --question-text "Should I use SDXL or Flux for the SIGNAL LOST anchor images?"

    # specify topic slug for the output directory
    python scripts/_consult_round_robin.py --topic 2026-04-17-anchor-image-pick

Outputs
-------
    docs/superpowers/consultations/<DATE>-<TOPIC>/
        00_question.md
        01_chatgpt.md           (round A response + model used)
        02_gemini.md            (round B response + model used)
        03_synthesis.md         (agreement, disagreement, open questions)
        transcript.json         (raw API responses for audit)

Safety
------
- Reads OPENAI_API_KEY and GEMINI_API_KEY fresh from HKCU:\\Environment.
  Never written to disk.  Never logged.  Never put in argv / env of any
  child process.  Held in memory only for the duration of the HTTP call.
- Stdlib only: urllib.request, json, no third-party deps.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Matches a leading ISO date (YYYY-MM-DD), optionally followed by '-...'.
# Used to detect when --topic already carries a date prefix so we don't
# double-prepend today's date to the output directory.
_LEADING_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(-|$)")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONSULT_BASE = ROOT / "docs" / "superpowers" / "consultations"

# ---------------------------------------------------------------------------
# Model ladders -- newest first, fall back gracefully
# ---------------------------------------------------------------------------
OPENAI_MODELS = ["gpt-5.4", "gpt-5", "gpt-4.1", "gpt-4o"]
GEMINI_MODELS = ["gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-1.5-pro"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)
TIMEOUT_SEC = 180


# ---------------------------------------------------------------------------
# API key access -- registry only, never persisted
# ---------------------------------------------------------------------------

def _read_env_var(name: str, expected_prefix: str | None = None) -> str:
    """Read a User-scope env var fresh from HKCU\\Environment.

    expected_prefix lets us catch obviously-wrong values (e.g. paste
    errors that drop the leading sk- / hf_ / AI prefix).  Pass None
    to skip the prefix check.
    """
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
    if expected_prefix and not value.startswith(expected_prefix):
        raise RuntimeError(
            f"{name} does not start with expected prefix {expected_prefix!r} "
            f"(got first 4 chars: {value[:4]!r}). Probably malformed."
        )
    return value


# ---------------------------------------------------------------------------
# OpenAI call (chat/completions ladder)
# ---------------------------------------------------------------------------

def call_openai(prompt: str, system: str, api_key: str) -> tuple[str, str]:
    """Returns (model_used, response_text).  Falls through model-not-found."""
    last_err = ""
    for model in OPENAI_MODELS:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }).encode("utf-8")
        req = urllib.request.Request(
            OPENAI_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return (model, content)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"{model} -> HTTP {e.code}: {err_body[:300]}"
            if e.code in (404, 400) and (
                "model" in err_body.lower() or "not_found" in err_body.lower()
            ):
                print(f"[openai] {model} unavailable, trying next...", file=sys.stderr)
                continue
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
    raise RuntimeError(f"All OpenAI models failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Gemini call (generativelanguage v1beta ladder)
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, system: str, api_key: str) -> tuple[str, str]:
    """Returns (model_used, response_text).  Falls through model-not-found.

    Gemini's REST API takes the system instruction as a separate field
    (not as a role inside contents).  contents is a flat user-turn list.
    """
    last_err = ""
    for model in GEMINI_MODELS:
        url = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)
        body = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # Pull text out of candidates[0].content.parts[*].text
            cands = data.get("candidates", [])
            if not cands:
                last_err = f"{model} -> empty candidates: {str(data)[:300]}"
                continue
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                last_err = f"{model} -> empty text in response"
                continue
            return (model, text)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"{model} -> HTTP {e.code}: {err_body[:300]}"
            if e.code in (404, 400) and (
                "model" in err_body.lower() or "not found" in err_body.lower()
            ):
                print(f"[gemini] {model} unavailable, trying next...", file=sys.stderr)
                continue
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Synthesis prompt (used as round B input + as standalone summary scaffold)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior systems architect advising a solo developer on a "
    "ComfyUI radio-drama generator (OTR \"SIGNAL LOST\") running on a single "
    "RTX 5080 Laptop / 16 GB VRAM Windows workstation. 100% local, no cloud. "
    "Audio output must remain byte-identical between runs (rule C7). VRAM "
    "ceiling is 14.5 GB. The owner does NOT want low-level VRAM optimization "
    "work (no weight streaming, no Flash Attention chasing). Prefer the "
    "smallest change with the largest payoff. Cite specific files / line "
    "numbers when relevant. Be candid; flag uncertainty rather than bluffing."
)


def gemini_followup_prompt(question: str, openai_response: str, openai_model: str) -> str:
    """Round-B prompt: ask Gemini to react to ChatGPT's answer."""
    return (
        f"You are the second opinion in a round-robin design consultation.\n\n"
        f"## Original question\n\n{question}\n\n"
        f"## ChatGPT ({openai_model}) answered:\n\n{openai_response}\n\n"
        f"## Your task\n\n"
        f"1. State whether you AGREE, PARTIALLY AGREE, or DISAGREE with the "
        f"core recommendation, in one sentence.\n"
        f"2. List any FACTUAL ERRORS in the ChatGPT answer (wrong file paths, "
        f"non-existent libraries, mis-stated VRAM costs, version-pin mistakes, etc).\n"
        f"3. List anything IMPORTANT THAT WAS OMITTED.\n"
        f"4. Give your own short recommendation (3-6 bullets) for what the "
        f"developer should actually do. If you would do something materially "
        f"different from ChatGPT, say so directly.\n"
        f"5. Note any items where you are uncertain and would want to verify "
        f"before committing.\n"
    )


def synthesis_text(
    question: str,
    openai_model: str,
    openai_response: str,
    gemini_model: str,
    gemini_response: str,
) -> str:
    """Static synthesis scaffold -- the actual synthesis happens in Claude's
    chat context after reading the saved files.  This file just gives the
    reader a clean side-by-side."""
    return (
        f"# Synthesis -- {datetime.date.today().isoformat()}\n\n"
        f"**Question:** {question}\n\n"
        f"---\n\n"
        f"## ChatGPT ({openai_model})\n\n{openai_response}\n\n"
        f"---\n\n"
        f"## Gemini ({gemini_model})\n\n{gemini_response}\n\n"
        f"---\n\n"
        f"## To decide (Claude / human)\n\n"
        f"- [ ] Where do the two answers AGREE? (lock those in)\n"
        f"- [ ] Where do they DISAGREE? (resolve via either re-prompting "
        f"with the disagreement spelled out, or by reasoning from the OTR "
        f"constraints in CLAUDE.md)\n"
        f"- [ ] Any facts to verify before acting? (e.g. \"library X exists "
        f"and supports Windows / torch 2.10\")\n"
        f"- [ ] Final grounded recommendation:\n"
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _read_question(args: argparse.Namespace) -> str:
    if args.question_text:
        return args.question_text.strip()
    if args.question:
        return Path(args.question).read_text(encoding="utf-8").strip()
    print("Enter your design / bug question. End with Ctrl-Z then Enter (Windows) or Ctrl-D (POSIX):", file=sys.stderr)
    return sys.stdin.read().strip()


def _slugify(s: str, max_len: int = 40) -> str:
    """Lowercase, alnum + hyphens, max length."""
    out = []
    last_dash = False
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash and out:
            out.append("-")
            last_dash = True
    slug = "".join(out).strip("-")
    return slug[:max_len] or "consultation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Round-robin consultation: ChatGPT + Gemini.")
    parser.add_argument("--question", type=str, help="Path to a question file.")
    parser.add_argument("--question-text", type=str, help="Inline question string.")
    parser.add_argument("--topic", type=str, help="Slug for the output directory; defaults to first 40 chars of question.")
    parser.add_argument("--skip-openai", action="store_true", help="Skip the ChatGPT round (Gemini-only).")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip the Gemini round (ChatGPT-only).")
    args = parser.parse_args()

    question = _read_question(args)
    if not question:
        print("ERROR: empty question.", file=sys.stderr)
        return 1

    today = datetime.date.today().isoformat()
    topic = args.topic or _slugify(question)
    # If the caller already prefixed the topic with an ISO date (e.g.
    # "2026-04-17-anchor-image-pick"), don't double-prepend today's date.
    if _LEADING_ISO_DATE.match(topic):
        dir_name = topic
    else:
        dir_name = f"{today}-{topic}"
    out_dir = CONSULT_BASE / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "00_question.md").write_text(
        f"# Question -- {today}\n\n{question}\n",
        encoding="utf-8",
    )

    transcript: dict = {"question": question, "rounds": []}

    # ---- Round A: OpenAI ----
    openai_model = ""
    openai_response = ""
    if not args.skip_openai:
        try:
            openai_key = _read_env_var("OPENAI_API_KEY", expected_prefix="sk-")
            print(f"[round-robin] Round A -- OpenAI ladder: {OPENAI_MODELS}", file=sys.stderr)
            t0 = time.time()
            openai_model, openai_response = call_openai(
                prompt=question, system=SYSTEM_PROMPT, api_key=openai_key,
            )
            elapsed = time.time() - t0
            print(f"[round-robin] Round A done: {openai_model} in {elapsed:.1f}s", file=sys.stderr)
            (out_dir / "01_chatgpt.md").write_text(
                f"# Round A -- ChatGPT ({openai_model}) elapsed={elapsed:.1f}s\n\n{openai_response}\n",
                encoding="utf-8",
            )
            transcript["rounds"].append({
                "round": "A", "vendor": "openai", "model": openai_model,
                "elapsed_sec": round(elapsed, 2), "response": openai_response,
            })
        except Exception as e:
            print(f"[round-robin] Round A FAILED: {e}", file=sys.stderr)
            (out_dir / "01_chatgpt.md").write_text(
                f"# Round A -- FAILED\n\n{e}\n", encoding="utf-8",
            )

    # ---- Round B: Gemini ----
    gemini_model = ""
    gemini_response = ""
    if not args.skip_gemini:
        try:
            gemini_key = _read_env_var("GEMINI_API_KEY", expected_prefix=None)
            print(f"[round-robin] Round B -- Gemini ladder: {GEMINI_MODELS}", file=sys.stderr)
            # If we have an OpenAI answer, ask Gemini to react to it.
            # Otherwise just ask the original question.
            if openai_response:
                gemini_prompt = gemini_followup_prompt(question, openai_response, openai_model)
            else:
                gemini_prompt = question
            t0 = time.time()
            gemini_model, gemini_response = call_gemini(
                prompt=gemini_prompt, system=SYSTEM_PROMPT, api_key=gemini_key,
            )
            elapsed = time.time() - t0
            print(f"[round-robin] Round B done: {gemini_model} in {elapsed:.1f}s", file=sys.stderr)
            (out_dir / "02_gemini.md").write_text(
                f"# Round B -- Gemini ({gemini_model}) elapsed={elapsed:.1f}s\n\n{gemini_response}\n",
                encoding="utf-8",
            )
            transcript["rounds"].append({
                "round": "B", "vendor": "google", "model": gemini_model,
                "elapsed_sec": round(elapsed, 2), "response": gemini_response,
            })
        except Exception as e:
            print(f"[round-robin] Round B FAILED: {e}", file=sys.stderr)
            (out_dir / "02_gemini.md").write_text(
                f"# Round B -- FAILED\n\n{e}\n", encoding="utf-8",
            )

    # ---- Round C: synthesis scaffold ----
    (out_dir / "03_synthesis.md").write_text(
        synthesis_text(
            question, openai_model, openai_response,
            gemini_model, gemini_response,
        ),
        encoding="utf-8",
    )

    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[round-robin] OK -- outputs under {out_dir}", file=sys.stderr)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
