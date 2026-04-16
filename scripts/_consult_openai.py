"""
_consult_openai.py  --  Three-round design consultation with ChatGPT.
=====================================================================

Purpose
-------
Send the current state of the OTR HyWorld stack to OpenAI's API and ask
for a robustness critique, graphical-fidelity / replacement-module ideas,
and a prioritized v2.0 next-phase plan.  Three sequential rounds; later
rounds get the full transcript so the model can build on its own answers.

Safety
------
- Reads OPENAI_API_KEY from HKCU:\\Environment fresh on every run.
  Never written to disk.  Never logged.  Never put in argv / env of any
  child process.  Held in memory only for the duration of the HTTP call.
- All output written under docs/superpowers/consultations/<DATE>/.
- No telemetry, no third-party libraries, stdlib only.

Usage
-----
    python scripts/_consult_openai.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONSULT_DIR = ROOT / "docs" / "superpowers" / "consultations" / "2026-04-16-chatgpt"
CONSULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model selection -- try newest first, fall back gracefully.
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = ["gpt-5", "gpt-4.1", "gpt-4o"]
API_URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT_SEC = 180


def _read_api_key() -> str:
    """Read OPENAI_API_KEY fresh from HKCU registry. Memory-only."""
    try:
        import winreg  # type: ignore
    except ImportError:
        raise RuntimeError("winreg not available -- this script requires Windows.")
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        try:
            value, _ = winreg.QueryValueEx(k, "OPENAI_API_KEY")
        except FileNotFoundError:
            raise RuntimeError("OPENAI_API_KEY not found in HKCU\\Environment.")
    if not value or not value.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is empty or malformed.")
    return value


def _read_text(rel_path: str) -> str:
    p = ROOT / rel_path
    if not p.exists():
        return f"(missing file: {rel_path})"
    return p.read_text(encoding="utf-8", errors="replace")


def _call_openai(messages: list[dict], api_key: str) -> tuple[str, str]:
    """POST to chat/completions. Returns (model_used, content). Tries model
    candidates in order, falling back on 'model not found' / 4xx errors.
    """
    last_err = ""
    for model in MODEL_CANDIDATES:
        body = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.4,
        }).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
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
            # Only fall through on model-related errors; auth or rate-limit
            # errors should bubble up.
            if e.code in (404, 400) and ("model" in err_body.lower() or "not_found" in err_body.lower()):
                print(f"[consult] {model} unavailable, trying next...", file=sys.stderr)
                continue
            raise RuntimeError(last_err) from e
        except Exception as e:
            last_err = f"{model} -> {type(e).__name__}: {e}"
            raise RuntimeError(last_err) from e
    raise RuntimeError(f"All model candidates failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Context bundle (read from disk every run so it's always current)
# ---------------------------------------------------------------------------

def _build_context_bundle() -> str:
    """Concatenate the relevant source files into one readable bundle."""
    parts = [
        "## CLAUDE.md (project rules)\n```\n" + _read_text("CLAUDE.md") + "\n```\n",
        "## otr_v2/hyworld/bridge.py\n```python\n" + _read_text("otr_v2/hyworld/bridge.py") + "\n```\n",
        "## otr_v2/hyworld/poll.py\n```python\n" + _read_text("otr_v2/hyworld/poll.py") + "\n```\n",
        "## otr_v2/hyworld/renderer.py\n```python\n" + _read_text("otr_v2/hyworld/renderer.py") + "\n```\n",
        "## otr_v2/hyworld/worker.py\n```python\n" + _read_text("otr_v2/hyworld/worker.py") + "\n```\n",
        "## otr_v2/hyworld/shotlist.py\n```python\n" + _read_text("otr_v2/hyworld/shotlist.py") + "\n```\n",
        "## docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md (creative mapping doc -- excerpt)\n```\n"
            + _read_text("docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md")[:18000] + "\n```\n",
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System + round prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior systems architect helping review and harden a custom-node integration for ComfyUI on a single-GPU Windows workstation. The owner is solo, ships from his bedroom, runs everything 100% local with no cloud. Be candid, concrete, and prioritize the smallest changes with the largest payoff. Avoid marketing language. Cite line numbers / file names from the supplied bundle when you make a recommendation. When asked for a plan, prefer numbered phases each sized to one mergeable commit."""

ROUND_1_PROMPT = """ROUND 1 of 3 -- Robustness critique.

CONTEXT BUNDLE:
The following is the entire current state of the HyWorld integration in OTR (a ComfyUI radio-drama generator). Five Python files plus the creative mapping doc plus the project rules file.

{context}

YOUR TASK:
Read the bundle. Critique the HyWorld pipeline (Bridge -> Poll -> Renderer + Worker sidecar) for robustness on a real Windows workstation. Specifically:

1. **Failure modes I'm not handling.** What can break that the current code does not catch? Be concrete: name the function and the case.
2. **Race conditions.** STATUS.json is written by the worker subprocess and read by the poll node. Where can they collide? Are the write/read patterns safe?
3. **Contract violations.** What can a malformed `script_lines` or `production_plan_json` do to the bridge / shotlist / renderer? Where do invariants leak?
4. **Process lifecycle.** The bridge spawns the worker fire-and-forget. What happens when the user cancels the ComfyUI workflow mid-run? When ComfyUI itself crashes? When the worker hangs? Are PIDs tracked correctly? Are zombies possible?
5. **Audio-byte-identical guarantee (C7).** Are there any code paths in renderer.py that could re-encode or modify the audio? Any subprocess invocations missing `-c:a copy`?
6. **Disk hygiene.** `io/hyworld_in/<job_id>/` and `io/hyworld_out/<job_id>/` accumulate forever. Is there a sweep? When should there be?
7. **Logging consistency.** Are log messages structured well enough to debug from a 12-minute episode log without re-running?

Be brutal but actionable. For each finding, propose the fix in one sentence. Output as a numbered list grouped by severity (Critical / Major / Minor). At the end, give a SCORECARD of overall robustness (1-10) with a one-line justification."""

ROUND_2_PROMPT = """ROUND 2 of 3 -- Graphical fidelity & replacement modules.

You've now seen the full HyWorld stack. Reality check:

- WorldMirror 2.0 is the ONLY shipped HY-World 2.0 model. Pano / Stereo / Nav are all "Coming Soon" with no ETA.
- The current worker.py is in stub mode: it writes solid-color PNGs and uses ffmpeg's `zoompan` filter to make Ken Burns motion clips driven by camera adjectives from the shotlist. That's the visual today.
- The design doc Section 11 lists candidate replacement modules: `Diffusion360_ComfyUI` (text -> 360 pano, SDXL-based), `SPAG4D` (pano -> 3DGS, ~6-8 GB VRAM), `ComfyUI-Sharp` (image -> 3DGS, sub-1s, very low VRAM), `SplaTraj` (semantic camera path planning), with `ComfyUI-3D-Pack` as an umbrella.
- Hardware: RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows. Python 3.12, torch 2.10, CUDA 13.0 in main env. Worker runs CPU-only today; GPU work must wait for Bark TTS to finish or coordinate via a VRAM gate.
- HARD CONSTRAINT: audio output must be byte-identical to v1.7 baseline (C7). Audio is king. Worker must never starve the audio pipeline of VRAM during generation. Bark TTS holds the GPU ~12-18 min per episode.

YOUR TASKS:

1. **Lowest-risk highest-payoff visual upgrade.** Of the candidate stand-in modules, which single one should land first? Justify in 3 bullets: visual payoff, risk, integration cost.

2. **Sequencing.** Propose a 3-step ladder from today's Ken Burns stub to "real visual fidelity." For each step, name the module + what visual capability it adds + what blocks it from being step 1.

3. **Missing modules.** What's NOT in the design doc that I should consider? Specifically:
   - Anchor image generation (the design doc waves a hand at "any text-to-image model"). What's the right local SDXL/Flux pickfor SIGNAL LOST's CRT/broadcast-static aesthetic?
   - Splat renderer (we have splat generators; what actually turns a PLY into MP4 frames?)
   - VRAM gate / queue between worker and main ComfyUI (need a coordinator since worker shouldn't fight Bark for VRAM)
   - CRT post-FX pass (renderer mentions `crt_postfx` boolean but nothing implements it)
   - Caching strategy (anchor images per scene? splats per env? hash key shape?)

4. **Replacement module shopping list.** For each missing piece in (3), name a specific local-only open-source repo or model and a 1-sentence reason. Prefer ComfyUI-native nodes when they exist. Be honest about what doesn't have a clean answer yet.

5. **Audio coexistence pattern.** Sketch a coordinator design (dataclass + state machine + 4-5 events) where the worker can request GPU time without crashing Bark or causing OOM. Should it be a file-lock, a ZMQ/Redis token, a simple FIFO queue? What's the simplest thing that's actually safe?

Output as numbered sections matching the 5 tasks. Be specific."""

ROUND_3_PROMPT = """ROUND 3 of 3 -- Prioritized v2.0 next-phase plan.

You've critiqued robustness (Round 1) and proposed graphical upgrades + missing modules (Round 2). Now synthesize.

YOUR TASK: Produce the v2.0 next-phase implementation plan as a sequence of PHASES. Each phase = ONE mergeable PR/commit on the v2.0-alpha branch.

Constraints to honor:
- Each phase must keep the audio pipeline byte-identical (C7) and the Bug Bible regression suite (~25 tests) green.
- Each phase must work end-to-end before merge: the workflow still produces a valid MP4, even if the visuals are still stub-quality.
- Worker must not contend for GPU with Bark TTS during the audio render window.
- VRAM ceiling 14.5 GB peak. No CheckpointLoaderSimple in main graph (C2). Subprocess + spawn for any GPU-heavy work (C3).
- Solo developer, async swimming breaks, prefers honest plans over ambitious ones.
- Branch is v2.0-alpha. Main is frozen.

DELIVERABLE:

```
PHASE A -- <one-line title>
  Goal:        <what this unlocks>
  Files:       <which files change>
  Risk gate:   <what test/probe must pass before merge>
  Rollback:    <how to revert cleanly if it breaks>
  Estimated:   <hours of human + Claude work>
  Why now:     <why this and not Phase B/C>

PHASE B -- ...
PHASE C -- ...
PHASE D -- ...
PHASE E -- ...
```

Aim for 4-6 phases. Order them strictly by lowest-risk-highest-value first. The first phase must be something that could realistically ship today. The last phase can be aspirational ("when HY-Pano 2.0 ships").

After the phase list, add a NOTES section covering:
- The single biggest design risk across the whole plan, and the early signal that would tell us we're hitting it.
- One thing Jeffrey should manually test/verify himself that Claude can't (e.g. visual quality judgment, listening for audio drift).
- One thing to drop from the original design doc Section 11 that you now think is wrong or over-scoped.

End with a one-line FIRST-MOVE recommendation for the very next coding session."""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    api_key = _read_api_key()
    print("[consult] API key loaded (memory only, never written to disk).", file=sys.stderr)

    context = _build_context_bundle()
    print(f"[consult] Context bundle: {len(context):,} chars", file=sys.stderr)

    # Persist the bundle for posterity (no key in here)
    (CONSULT_DIR / "00_context_bundle.md").write_text(context, encoding="utf-8")

    transcript: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    rounds = [
        ("01_round1_robustness.md", ROUND_1_PROMPT.format(context=context)),
        ("02_round2_fidelity.md", ROUND_2_PROMPT),
        ("03_round3_plan.md", ROUND_3_PROMPT),
    ]

    used_model = ""
    for filename, user_prompt in rounds:
        print(f"[consult] Sending {filename} ...", file=sys.stderr)
        transcript.append({"role": "user", "content": user_prompt})
        t0 = time.monotonic()
        try:
            model_used, reply = _call_openai(transcript, api_key)
        except Exception as e:
            err_path = CONSULT_DIR / (filename + ".ERROR.txt")
            err_path.write_text(f"FAILED on {filename}\n{e}\n", encoding="utf-8")
            print(f"[consult] FAIL on {filename}: {e}", file=sys.stderr)
            return 2
        used_model = model_used
        elapsed = time.monotonic() - t0
        print(f"[consult] {filename} OK -- {len(reply):,} chars in {elapsed:.1f}s (model={model_used})", file=sys.stderr)
        transcript.append({"role": "assistant", "content": reply})
        out_path = CONSULT_DIR / filename
        header = (
            f"# {filename} -- model={model_used} elapsed={elapsed:.1f}s\n\n"
            f"## User prompt\n\n{user_prompt}\n\n"
            f"---\n\n## Assistant reply\n\n"
        )
        out_path.write_text(header + reply + "\n", encoding="utf-8")

    # Save the full transcript without the system prompt for easier review
    (CONSULT_DIR / "transcript.json").write_text(
        json.dumps([m for m in transcript if m["role"] != "system"], indent=2),
        encoding="utf-8",
    )
    print(f"[consult] Done. Model used: {used_model}. Output: {CONSULT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
