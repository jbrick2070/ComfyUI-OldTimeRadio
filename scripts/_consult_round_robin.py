"""
_consult_round_robin.py  --  Round-robin design / bug-fix consultation.
=======================================================================

Purpose
-------
Route a tough design question or bug investigation through multiple
external models until a grounded answer emerges:

    Round A:  ChatGPT (gpt-5.4 -> gpt-5.4-pro -> gpt-5.1-codex-max ladder)
    Round B:  Gemini   (gemini-3.1-pro-preview -> gemini-2.5-pro)
    Round C:  NVIDIA   (mistral-nemotron -> llama-3.3-nemotron -> ...)
    Round D:  Synthesis -- this script summarizes agreement / disagreement
              between A, B, and C so I (Claude) can decide the grounded
              answer in the chat context after running this script.

Per CLAUDE.md "Round-Robin Consultation" rule.  Use for non-trivial
design choices, library picks, architecture trade-offs, or bug root
causes that aren't obvious from a stack trace.  Skip for one-line fixes.
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

_LEADING_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(-|$)")

ROOT = Path(__file__).resolve().parent.parent
CONSULT_BASE = ROOT / "docs"

# 2026-05-01 EVENING -- OpenAI ladder rebuilt per current docs
# (platform.openai.com/docs/models, developers.openai.com/api).
#   - gpt-5.5         : flagship reasoning/coding (recommended default)
#   - gpt-5.5-pro     : slower, higher-quality (Responses API only)
#   - gpt-5.4         : prior frontier; serious work
#   - gpt-5.4-pro     : Responses API only
#   - gpt-5.4-mini    : low-latency / cheap testing
#   - gpt-5.4-nano    : even cheaper
#   - gpt-5.3-codex   : current coding-tuned (replaces gpt-5.1-codex-max)
#   - gpt-5           : older flagship, still works
#   - gpt-4.1         : last-gen fallback
#   - gpt-4o-mini     : final fallback (plain gpt-4o is deprecated)
# Dropped per docs review:
#   gpt-5.1-codex-max  (deprecated by 5.3-codex)
#   gpt-5.3-chat-latest (ChatGPT-only model, not for API use)
#   gpt-4o             (deprecated)
OPENAI_MODELS = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.3-codex",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o-mini",
]
# 2026-05-01 EVENING -- Gemini ladder rebuilt per nested Gemini 3.1
# Pro guidance to Jeffrey, refined version of the routing rules:
#   gemini-3.1-pro-preview-customtools : primary for OTR's actual use
#       case (architecture/bug/code consults). The "customtools" variant
#       is explicitly tuned for bash/custom tool workflows, which maps
#       directly to OTR's "ingest a workflow JSON + a code excerpt + a
#       stack trace + reason" pattern.
#   gemini-3.1-pro-preview : same model without the tool-tuned wrapper,
#       for when customtools doesn't fit the request shape.
#   gemini-3-flash-preview : fast multimodal tier. OTR's consults are
#       text-only today, but flash keeps a usable rung in the ladder
#       AND is the right primary if a future caller passes
#       image/video/PDF (Design QA path). 1M ctx, function calling,
#       code execution, computer use, structured outputs.
# Dropped from prior ladder:
#   gemini-3-pro-preview         (shut down March 2026)
#   gemini-pro-latest            (alias resolves unpredictably)
#   gemini-flash-latest          (alias)
#   gemini-2.5-pro / -flash      (legacy, outclassed by 3.x)
#   gemini-3.1-flash-lite-preview (high-volume/cost-sensitive tier
#                                  not used by OTR consults)
#
# ROUTING NOTE: this ladder is for TEXT-ONLY architecture consults
# (the OTR use case as of 2026-05-01). For image/video/frame Design
# QA, the right move is a separate caller targeting
# gemini-3-flash-preview with image content blocks + a structured
# checklist payload (screenshot + goal + constraints + checklist +
# ask for pass/fail JSON). Don't route image-bearing payloads
# through this round-robin.
GEMINI_MODELS = [
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
]

# 2026-05-01 EVENING -- switched OpenAI calls from /v1/chat/completions
# to /v1/responses so the "pro" + reasoning-tuned variants in the ladder
# (gpt-5.5-pro, gpt-5.4-pro, gpt-5.5 with effort="medium") actually work.
# Chat/completions silently 4xxs them. Responses is the canonical
# endpoint for the gpt-5.x line per platform.openai.com/docs/models.
OPENAI_URL = "https://api.openai.com/v1/responses"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)
TIMEOUT_SEC = 180


def _read_env_var(name: str, expected_prefix: str | None = None) -> str:
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
    if expected_prefix and not value.startswith(expected_prefix):
        raise RuntimeError(
            f"{name} does not start with expected prefix {expected_prefix!r} "
            f"(got first 4 chars: {value[:4]!r}). Probably malformed."
        )
    return value


def _extract_responses_text(data: dict) -> str:
    """Pull the assistant text out of a /v1/responses result body.

    The Responses API response shape (raw HTTP, no SDK):

        {
          "id": "...",
          "object": "response",
          "model": "gpt-5.5",
          "output": [
            {
              "type": "message",
              "role": "assistant",
              "content": [
                {"type": "output_text", "text": "..."}
              ]
            }
          ],
          ...
        }

    The SDK exposes a convenience ``response.output_text`` field; the raw
    HTTP payload sometimes mirrors it as a top-level ``output_text`` for
    convenience too. Try both, then walk the output array as a fallback.
    """
    # Convenience field (when present).
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]
    # Walk the output array, collect every output_text fragment in order.
    chunks: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            # Reasoning items (type="reasoning") and tool calls show up here
            # too; we only want the assistant message text.
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("output_text", "text") and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(c for c in chunks if c)


def call_openai(prompt: str, system: str, api_key: str) -> tuple[str, str]:
    """Call the OpenAI Responses API with model fallthrough.

    Returns (model_used, response_text). Falls through on:
      - HTTP 404/400 with "model"/"not_found" in body (model unknown)
      - HTTP 403 (account doesn't have access to this model)
      - HTTP 429 (rate-limited; try a cheaper next-rung model)
      - HTTP 400 with "endpoint" / "responses" / "support" hints
        (some models are scoped to specific endpoints)
    Re-raises on auth errors (401), other 400s with no model hint, and
    transport errors (DNS, TLS, etc.) -- those won't be fixed by the next
    rung.

    Logs the actual error type per Jeffrey 2026-05-01 EVENING note so a
    misconfigured key (permission_error vs model_not_found vs rate_limit)
    is visible without having to instrument the script later.
    """
    last_err = ""
    for model in OPENAI_MODELS:
        # Responses API request shape. Reasoning models (gpt-5.x) accept
        # the ``reasoning.effort`` knob; older models ignore it cleanly.
        is_reasoning = model.startswith("gpt-5") or "codex" in model
        body_dict: dict = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if is_reasoning:
            body_dict["reasoning"] = {"effort": "medium"}
        else:
            # Older models support classic temperature; reasoning models
            # don't (it's controlled by reasoning.effort instead).
            body_dict["temperature"] = 0.4
        body = json.dumps(body_dict).encode("utf-8")
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
            content = _extract_responses_text(data)
            if not content.strip():
                # Empty response -- treat as failure, fall through.
                last_err = f"{model} -> empty response (parse mismatch?)"
                print(
                    f"[openai] {model} returned empty text -- trying next...",
                    file=sys.stderr,
                )
                continue
            return (model, content)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"{model} -> HTTP {e.code}: {err_body[:300]}"
            err_lower = err_body.lower()
            # Per Jeffrey 2026-05-01: log the real error type so we can
            # tell model_not_found from permission/rate/endpoint errors.
            print(
                f"[openai] {model}: HTTP {e.code} -- "
                f"{err_body[:140].replace(chr(10), ' ')}",
                file=sys.stderr,
            )
            should_fall_through = False
            if e.code in (404, 400) and (
                "model" in err_lower
                or "not_found" in err_lower
                or "endpoint" in err_lower
                or "responses" in err_lower
                or "support" in err_lower
            ):
                # Wrong model name OR model not on this endpoint OR
                # request shape mismatch -- next rung.
                should_fall_through = True
            elif e.code == 403:
                # Account lacks access to this model -- next rung.
                should_fall_through = True
            elif e.code == 429:
                # Rate limit -- try cheaper model in the ladder.
                should_fall_through = True
            if should_fall_through:
                print(
                    f"[openai] {model} unavailable, trying next...",
                    file=sys.stderr,
                )
                continue
            # 401 (bad key), other 400s, 5xx -- not fixable by next rung.
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            # Transport layer failure: DNS, TLS, connection refused,
            # timeout. Probably won't be fixed by the next rung either.
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            print(f"[openai] {model}: {type(e).__name__} -- {e}", file=sys.stderr)
            raise RuntimeError(last_err) from e
    raise RuntimeError(
        f"All OpenAI models in ladder failed. Last error: {last_err}"
    )


def call_gemini(prompt: str, system: str, api_key: str) -> tuple[str, str]:
    """Returns (model_used, response_text).  Falls through model-not-found."""
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
            if e.code == 429:
                print(f"[gemini] {model} quota exhausted, trying next...", file=sys.stderr)
                continue
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


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
    return (
        f"You are the second opinion in a round-robin design consultation.\n\n"
        f"## Original question\n\n{question}\n\n"
        f"## ChatGPT ({openai_model}) answered:\n\n{openai_response}\n\n"
        f"## Your task\n\n"
        f"1. State whether you AGREE, PARTIALLY AGREE, or DISAGREE with the "
        f"core recommendation, in one sentence.\n"
        f"2. List any FACTUAL ERRORS in the ChatGPT answer.\n"
        f"3. List anything IMPORTANT THAT WAS OMITTED.\n"
        f"4. Give your own short recommendation (3-6 bullets).\n"
        f"5. Note any items where you are uncertain and would want to verify.\n"
    )


def nvidia_followup_prompt(
    question: str,
    openai_response: str, openai_model: str,
    gemini_response: str, gemini_model: str,
) -> str:
    return (
        f"You are the THIRD opinion in a round-robin design consultation.\n"
        f"Two models have already answered.  Your job is to break ties and\n"
        f"surface anything both of them missed.\n\n"
        f"## Original question\n\n{question}\n\n"
        f"## ChatGPT ({openai_model}) answered:\n\n{openai_response}\n\n"
        f"## Gemini ({gemini_model}) answered:\n\n{gemini_response}\n\n"
        f"## Your task\n\n"
        f"1. Where ChatGPT and Gemini AGREE, state whether you concur or "
        f"see a flaw they missed.\n"
        f"2. Where they DISAGREE, pick a side and explain why -- or propose "
        f"a third path.\n"
        f"3. List any FACTUAL ERRORS or hallucinated APIs / file paths in "
        f"either answer.\n"
        f"4. List anything IMPORTANT THAT BOTH OMITTED.\n"
        f"5. Give your own prioritized recommendation (3-6 bullets).\n"
        f"6. Note any items where you are uncertain and want verification.\n"
    )


def synthesis_text(
    question,
    openai_model, openai_response,
    gemini_model, gemini_response,
    nvidia_model, nvidia_response,
):
    return (
        f"# Synthesis -- {datetime.date.today().isoformat()}\n\n"
        f"**Question:** {question}\n\n"
        f"---\n\n"
        f"## ChatGPT ({openai_model})\n\n{openai_response}\n\n"
        f"---\n\n"
        f"## Gemini ({gemini_model})\n\n{gemini_response}\n\n"
        f"---\n\n"
        f"## NVIDIA ({nvidia_model})\n\n{nvidia_response}\n\n"
        f"---\n\n"
        f"## To decide (Claude / human)\n\n"
        f"- [ ] All three agree:\n"
        f"- [ ] Two-vs-one splits:\n"
        f"- [ ] Facts to verify:\n"
        f"- [ ] Final grounded recommendation:\n"
    )


def _read_question(args: argparse.Namespace) -> str:
    if args.question_text:
        return args.question_text.strip()
    if args.question:
        return Path(args.question).read_text(encoding="utf-8").strip()
    print("Enter your design / bug question. End with Ctrl-Z then Enter:", file=sys.stderr)
    return sys.stdin.read().strip()


def _slugify(s: str, max_len: int = 40) -> str:
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
    parser = argparse.ArgumentParser(description="Round-robin consultation.")
    parser.add_argument("--question", type=str)
    parser.add_argument("--question-text", type=str)
    parser.add_argument("--topic", type=str)
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-nvidia", action="store_true")
    args = parser.parse_args()

    question = _read_question(args)
    if not question:
        print("ERROR: empty question.", file=sys.stderr)
        return 1

    today = datetime.date.today().isoformat()
    topic = args.topic or _slugify(question)
    if _LEADING_ISO_DATE.match(topic):
        prefix = topic
    else:
        prefix = f"{today}-{topic}"
    CONSULT_BASE.mkdir(parents=True, exist_ok=True)

    def out_path(suffix: str) -> Path:
        return CONSULT_BASE / f"{prefix}__{suffix}"

    out_path("00_question.md").write_text(
        f"# Question -- {today}\n\n{question}\n",
        encoding="utf-8",
    )

    transcript: dict = {"question": question, "rounds": []}

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
            out_path("01_chatgpt.md").write_text(
                f"# Round A -- ChatGPT ({openai_model}) elapsed={elapsed:.1f}s\n\n{openai_response}\n",
                encoding="utf-8",
            )
            transcript["rounds"].append({
                "round": "A", "vendor": "openai", "model": openai_model,
                "elapsed_sec": round(elapsed, 2), "response": openai_response,
            })
        except Exception as e:
            print(f"[round-robin] Round A FAILED: {e}", file=sys.stderr)
            out_path("01_chatgpt.md").write_text(
                f"# Round A -- FAILED\n\n{e}\n", encoding="utf-8",
            )

    gemini_model = ""
    gemini_response = ""
    if not args.skip_gemini:
        try:
            gemini_key = _read_env_var("GEMINI_API_KEY", expected_prefix=None)
            print(f"[round-robin] Round B -- Gemini ladder: {GEMINI_MODELS}", file=sys.stderr)
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
            out_path("02_gemini.md").write_text(
                f"# Round B -- Gemini ({gemini_model}) elapsed={elapsed:.1f}s\n\n{gemini_response}\n",
                encoding="utf-8",
            )
            transcript["rounds"].append({
                "round": "B", "vendor": "google", "model": gemini_model,
                "elapsed_sec": round(elapsed, 2), "response": gemini_response,
            })
        except Exception as e:
            print(f"[round-robin] Round B FAILED: {e}", file=sys.stderr)
            out_path("02_gemini.md").write_text(
                f"# Round B -- FAILED\n\n{e}\n", encoding="utf-8",
            )

    nvidia_model = ""
    nvidia_response = ""
    if not args.skip_nvidia:
        try:
            # Local import so the round-robin still works if the NVIDIA
            # module is missing or its key isn't set yet.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _consult_nvidia import (  # type: ignore
                call_nvidia, DEFAULT_MODELS as NVIDIA_MODELS,
            )
            nvidia_key = _read_env_var("NVIDIA_API_KEY", expected_prefix=None)
            print(f"[round-robin] Round C -- NVIDIA ladder: {NVIDIA_MODELS}", file=sys.stderr)
            if openai_response and gemini_response:
                nvidia_prompt = nvidia_followup_prompt(
                    question,
                    openai_response, openai_model,
                    gemini_response, gemini_model,
                )
            elif openai_response or gemini_response:
                # Only one prior round succeeded -- treat NVIDIA as second opinion.
                prior_resp = openai_response or gemini_response
                prior_model = openai_model or gemini_model
                nvidia_prompt = gemini_followup_prompt(question, prior_resp, prior_model)
            else:
                nvidia_prompt = question
            t0 = time.time()
            nvidia_model, nvidia_response = call_nvidia(
                prompt=nvidia_prompt, system=SYSTEM_PROMPT,
                api_key=nvidia_key,
            )
            elapsed = time.time() - t0
            print(f"[round-robin] Round C done: {nvidia_model} in {elapsed:.1f}s", file=sys.stderr)
            out_path("03_nvidia.md").write_text(
                f"# Round C -- NVIDIA ({nvidia_model}) elapsed={elapsed:.1f}s\n\n{nvidia_response}\n",
                encoding="utf-8",
            )
            transcript["rounds"].append({
                "round": "C", "vendor": "nvidia", "model": nvidia_model,
                "elapsed_sec": round(elapsed, 2), "response": nvidia_response,
            })
        except Exception as e:
            print(f"[round-robin] Round C FAILED: {e}", file=sys.stderr)
            out_path("03_nvidia.md").write_text(
                f"# Round C -- FAILED\n\n{e}\n", encoding="utf-8",
            )

    out_path("04_synthesis.md").write_text(
        synthesis_text(
            question,
            openai_model, openai_response,
            gemini_model, gemini_response,
            nvidia_model, nvidia_response,
        ),
        encoding="utf-8",
    )

    out_path("transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[round-robin] OK -- outputs under {CONSULT_BASE} with prefix {prefix}__", file=sys.stderr)
    print(f"{CONSULT_BASE}/{prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
