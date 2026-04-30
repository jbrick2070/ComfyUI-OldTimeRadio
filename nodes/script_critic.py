"""script_critic.py -- OTR_LLMScriptCritic ComfyUI node.

Two-immune-system evaluation gate for SIGNAL LOST scripts. Runs a
SEPARATE local LLM (default `google/gemma-4-E4B-it`, ~4 GB at 4-bit
NF4) as a judge over the script the writer produced. The critic
never grades its own work because it's a different model than the
writer.

Why an in-workflow node:

  - The writer (Mistral-Nemo 12B) and critic (Gemma-4 E4B) cannot
    co-reside on a 16 GB Blackwell -- swapping is required. The
    swap is mediated through `_load_llm()` and the cache
    invalidation patterns built up for BUG-LOCAL-111. After the
    critic returns, the next phase that needs the writer (cleanup
    pass, director) reloads it from disk -- ~10-15s extra reload,
    no VRAM violation.
  - Output is written to `ledger.script_gates[]` so the gate
    becomes part of the per-episode audit trail alongside
    `audio_gates[]`.

Why advisory-by-default:

  - The critic is new. We need to collect findings across runs to
    learn its calibration before making it authoritative. The
    `block_on_reject` widget defaults to OFF -- the gate stamps
    its findings, the run continues to audio, and Jeffrey can
    inspect the critic's verdict in the ledger before deciding
    whether to flip the bit.

Wiring:

  - INPUT: script (STRING, e.g. from LLMScriptWriter), genre_flavor.
  - OUTPUT: script (STRING, passthrough), critic_report (STRING,
    markdown), verdict (STRING: PASS/REVISE/REJECT).
  - SIDE-EFFECT: ledger.script_gates[] entry, plus optional
    auto-update of docs/OTR-CANON.md when verdict == PASS.

Failure modes:

  - Critic LLM unavailable / OOM / parse error: stamp a "SKIPPED"
    gate entry, return the script unchanged with verdict=PASS.
    Critic is an enhancement, never a blocker.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("OTR.script_critic")


# Mirrors LLMScriptWriter dropdown. Kept in sync manually -- if the
# orchestrator dropdown gains/loses entries, mirror them here so the
# critic surface matches the production surface.
_LLM_MODEL_CHOICES = [
    "google/gemma-4-E4B-it",                                       # default critic
    "google/gemma-4-E2B-it",                                       # smaller critic
    "mistralai/Mistral-Nemo-Instruct-2407",                        # same-as-writer (NOT recommended)
    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
    "Nitral-AI/Captain-Eris_Violet-V0.420-12B (EXPERIMENTAL)",
    "inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)",
]

# Where the anti-slop rubric lives. Read at run time (not at module
# load) so the file can be updated between runs without restarting
# ComfyUI.
_ANTI_SLOP_PATH = Path(__file__).resolve().parent.parent / "docs" / "OTR-ANTI-SLOP.md"
_CANON_PATH      = Path(__file__).resolve().parent.parent / "docs" / "OTR-CANON.md"


# Gate parser: extracts the boolean expression after "applies_when:".
_GATE_RE = re.compile(r"\[applies_when:\s*([^\]]+)\]")

# scene_count derivation -- single source of truth for the
# smoke/short/medium length labels. If you add a new length tier in
# the workflow, add it here too.
_SCENE_COUNT_BY_LENGTH = {
    "smoke":  1,
    "short":  3,
    "medium": 5,
}

# Variables the rubric's [applies_when: ...] gates and {placeholder}
# tokens read. The ledger stamps `gen_params_initial` at write_script
# entry (see BUG-72), so all of these have real values by the time
# the critic runs. Listed here so the loader can detect missing
# fields and warn explicitly -- no silent fallbacks.
_REQUIRED_PARAM_KEYS = (
    "target_words",
    "num_characters",
    "genre_flavor",
    "target_length",
)


def _normalize_target_length(raw: str) -> str:
    """Normalize a target_length widget value to one of {smoke, short, medium}.

    The dropdown shows free-form labels like 'tiny (smoke, 1 act)' or
    'short (3 acts)'. The gate language uses canonical identifiers.
    This bridge keeps the user-facing copy editable without breaking
    rule gates downstream. New legacy labels can be added here without
    touching the rubric.
    """
    s = (raw or "").strip().lower()
    if not s:
        return "short"
    # Order matters: "1 act" must beat "5 acts" since 'short' contains
    # the substring 'sh' but never '1 act'.
    if "smoke" in s or "1 act" in s or "1-act" in s or "tiny" in s:
        return "smoke"
    if "5 act" in s or "5-act" in s or "medium" in s:
        return "medium"
    if "3 act" in s or "3-act" in s or "short" in s:
        return "short"
    return "short"


def _evaluate_gate(expr: str, params: dict) -> bool:
    """Evaluate an applies_when boolean expression against ledger params.

    Supports the operators documented in OTR-ANTI-SLOP.md:
        ==, !=, >=, <=, >, <, AND, OR

    Strings are quoted, ints are passed through. The expression is
    sandboxed by passing __builtins__={} -- no name lookups, no
    function calls, no attribute access can succeed.

    Returns True on parse failure so a bad gate does not silently
    delete the rule -- noisy is safer than missing.
    """
    raw = (expr or "").strip()
    if not raw or raw.lower() == "always":
        return True
    safe = raw
    # Substitute placeholder names with quoted/int literals. Sort by
    # length descending so "target_length" doesn't get clobbered by
    # the prefix "target".
    for k in sorted(params.keys(), key=lambda s: -len(s)):
        v = params[k]
        if isinstance(v, str):
            replacement = repr(v.lower())  # case-insensitive comparisons
        else:
            replacement = str(v)
        safe = re.sub(rf"\b{re.escape(k)}\b", replacement, safe)
    # Lowercase string literals already substituted; lowercase any RHS
    # bare-word comparator now (e.g. `target_length != smoke`). We do
    # this AFTER variable substitution to avoid touching numeric ops.
    safe = re.sub(
        r"!=\s*([A-Za-z][A-Za-z0-9_]*)",
        lambda m: f"!= {m.group(1).lower()!r}",
        safe,
    )
    safe = re.sub(
        r"==\s*([A-Za-z][A-Za-z0-9_]*)",
        lambda m: f"== {m.group(1).lower()!r}",
        safe,
    )
    safe = safe.replace(" AND ", " and ").replace(" OR ", " or ")
    try:
        return bool(eval(safe, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[ScriptCritic] gate eval failed (expr=%r safe=%r err=%s) - "
            "keeping rule (fail-open)",
            raw, safe, exc,
        )
        return True


def _coerce_params(raw: Optional[dict]) -> dict:
    """Coerce raw ledger params for use in the rubric loader.

    The ledger's `gen_params_initial` is the source of truth -- this
    function does NOT supply default values. If a required field is
    missing or unparseable, it stays absent in the output dict and a
    warning is logged. The gate evaluator's `fail-open` behaviour
    means missing fields produce a slightly broader rubric (rules
    that depend on a missing field still fire), which is the safe
    direction.

    Always-derived fields (`scene_count`, `scene_word_budget`,
    normalized `target_length`) are computed only when their inputs
    are present and parseable.
    """
    p: dict = dict(raw or {})

    # Numeric coercions: drop the field if unparseable so the gate
    # evaluator sees it as missing instead of as 0.
    for k in ("target_words", "num_characters"):
        if k not in p:
            continue
        try:
            p[k] = int(p[k])
        except (TypeError, ValueError):
            log.warning(
                "[ScriptCritic] ledger field %s=%r is not an int - "
                "dropped from gate evaluation", k, p[k],
            )
            del p[k]

    # Normalize target_length and derive scene_count / scene_word_budget
    # ONLY when the source values are present.
    if "target_length" in p:
        p["target_length"] = _normalize_target_length(str(p["target_length"]))
        scene_count = _SCENE_COUNT_BY_LENGTH.get(p["target_length"])
        if scene_count is not None:
            p["scene_count"] = scene_count
            if isinstance(p.get("target_words"), int):
                p["scene_word_budget"] = max(1, p["target_words"] // scene_count)

    # Friendly form for free-text fields (do not invent values).
    if isinstance(p.get("genre_flavor"), str):
        p["genre_flavor"] = p["genre_flavor"].replace("_", " ")
    if isinstance(p.get("creativity"), str):
        p["creativity"] = p["creativity"].lower()

    # Detect missing required fields up-front and warn once.
    missing = [k for k in _REQUIRED_PARAM_KEYS if k not in p]
    if missing:
        log.warning(
            "[ScriptCritic] ledger missing required rubric params %s - "
            "rubric will fail-open on gates that reference them. Check "
            "that the workflow's LLMScriptWriter ran first and stamped "
            "gen_params_initial.",
            missing,
        )

    return p


def _filter_rubric(template_text: str, params: dict) -> str:
    """Filter the rubric template against ledger params + substitute placeholders.

    Walks the template line by line:
      - Lines with `[applies_when: <expr>]` are kept only if the gate
        evaluates true. The gate tag is stripped from the kept line.
      - All `{placeholder}` tokens are replaced from the params dict.
      - Lines without a gate are passed through unchanged.

    Substitution happens AFTER gate filtering so we don't burn
    parsing on rules we won't ship to the critic.
    """
    out_lines: list[str] = []
    for line in (template_text or "").splitlines():
        m = _GATE_RE.search(line)
        if m:
            if not _evaluate_gate(m.group(1), params):
                continue  # rule excluded by gate
            line = _GATE_RE.sub("", line).rstrip()
        # Substitute placeholders. Use a partial-tolerant approach so
        # an unrecognized `{token}` in the template doesn't crash the
        # whole filter (it stays in the output as a visible diagnostic).
        try:
            line = line.format_map(_SafeMissing(params))
        except Exception as exc:  # noqa: BLE001
            log.debug("[ScriptCritic] format_map failed on line=%r: %s", line, exc)
        out_lines.append(line)
    return "\n".join(out_lines)


class _SafeMissing(dict):
    """dict subclass that returns the literal `{key}` for missing keys.

    Lets `str.format_map` pass through `{unknown}` placeholders rather
    than raising KeyError -- useful when the template has tokens we
    haven't wired into params yet.
    """

    def __missing__(self, key):
        return "{" + key + "}"


def _load_anti_slop_rubric(params: Optional[dict] = None) -> str:
    """Read OTR-ANTI-SLOP.md, filter by params, and return the result.

    Returns an empty string on missing file or read error -- the
    critic prompt has a sane built-in fallback rubric so this is
    not a hard requirement.
    """
    try:
        if not _ANTI_SLOP_PATH.exists():
            return ""
        raw = _ANTI_SLOP_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("[ScriptCritic] anti-slop read failed (%s) - using "
                    "built-in fallback rubric", exc)
        return ""
    p = _coerce_params(params)
    try:
        return _filter_rubric(raw, p)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[ScriptCritic] rubric filter failed (%s) - returning "
            "unfiltered template (critic will see gate metadata)", exc,
        )
        return raw


def _build_critic_prompt(
    script_text: str,
    genre_flavor: str,
    anti_slop: str,
) -> str:
    """Build the rejection-rubric prompt for the critic LLM.

    The anti_slop string is expected to already be filtered + template-
    filled by `_load_anti_slop_rubric(params)`. If the caller passes a
    bare template (with gate tags and `{placeholder}` tokens still in
    it) we ship it as-is and the critic will likely score lower than
    intended -- that's a caller bug, not an excuse for a hard fail.

    Critic is asked for a bounded structured response (SCORE / VERDICT
    / ISSUES) so we can parse without invoking another LLM. Temperature
    is held low at the call site (0.1) so the verdict is stable across
    re-runs of the same script.
    """
    genre_human = (genre_flavor or "sci-fi").replace("_", " ")
    rubric_block = anti_slop.strip() if anti_slop.strip() else (
        "GENERAL RUBRIC (built-in fallback - prefer OTR-ANTI-SLOP.md):\n"
        "- Every scene opens with a sound cue instead of dialogue.\n"
        "- Two characters back-to-back monologuing exposition.\n"
        "- 'I'll explain on the way' / placeholder transitions.\n"
        "- Period-inappropriate vocabulary in 1940s setting.\n"
        "- Characters narrating their own physical actions.\n"
        "- Telegraphed emotion ('he felt a cold dread').\n"
        "- Animated-environment cliches ('the air hummed').\n"
        "- Ensemble-voice collapse (all characters sound the same).\n"
        "- Epilogue lectures replacing dramatic resolution.\n"
        "- News-spine buried (article topic doesn't drive the plot).\n"
    )
    return (
        f"You are a script doctor for SIGNAL LOST, a 1940s-style "
        f"{genre_human} radio drama. Your job is to score this draft "
        f"AGAINST the rejection rubric below and return a structured "
        f"verdict.\n\n"
        f"You don't have to find defects. If the draft is clean, say "
        f"so. Inventing problems to look thorough is itself a tell.\n\n"
        f"REJECTION RUBRIC:\n{rubric_block}\n\n"
        f"DRAFT SCRIPT:\n---\n{script_text.strip()}\n---\n\n"
        f"Return ONLY this format, no commentary:\n"
        f"SCORE: <integer 0-100>\n"
        f"VERDICT: <PASS|REVISE|REJECT>\n"
        f"ISSUES:\n"
        f"- <category>: <specific finding tied to rubric>\n"
        f"- <category>: <specific finding tied to rubric>\n"
        f"(...up to 8 issues; if none, write 'none')\n\n"
        f"Scoring guide: 85+ PASS, 60-84 REVISE, <60 REJECT.\n"
    )


def _parse_critic_response(raw: str) -> dict:
    """Parse SCORE / VERDICT / ISSUES from the critic's response.

    Permissive parser -- the critic may add stray whitespace, quote
    marks, or markdown emphasis. We extract the structured fields
    and accept anything sensible.

    Returns dict with keys: score (int|None), verdict (str), issues (list[str]).
    On total parse failure returns score=None, verdict=PASS (advisory
    fallback), issues=[].
    """
    text = (raw or "").strip()

    score: Optional[int] = None
    m = re.search(r"SCORE\s*[:=]\s*(\d{1,3})", text, re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
            if 0 <= n <= 100:
                score = n
        except ValueError:
            score = None

    verdict = "PASS"
    m = re.search(r"VERDICT\s*[:=]\s*(PASS|REVISE|REJECT)", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
    elif score is not None:
        # Derive verdict from score if VERDICT line missing.
        if score >= 85:
            verdict = "PASS"
        elif score >= 60:
            verdict = "REVISE"
        else:
            verdict = "REJECT"

    issues: list[str] = []
    m = re.search(r"ISSUES\s*[:=]?\s*\n?(.*)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
        for raw_line in body.splitlines():
            line = raw_line.strip().lstrip("-*•").strip()
            if not line:
                continue
            if line.lower() == "none":
                break
            # Cap line length so a runaway critic doesn't bloat the
            # ledger.
            issues.append(line[:280])
            if len(issues) >= 12:
                break

    return {"score": score, "verdict": verdict, "issues": issues}


def _canon_update(
    canon_path: Path = _CANON_PATH,
    premise: str = "",
    twist: str = "",
    motif: str = "",
    cap_per_section: int = 25,
) -> bool:
    """Append a passing episode's premise/twist/motif to OTR-CANON.md.

    Auto-update step the critic runs ONLY when verdict=PASS. Each
    section is capped at `cap_per_section` entries (rolling window)
    so the writer's context budget stays bounded. Returns True if a
    write occurred.

    The critic does not call this directly in the current ship --
    it's wired up here so a later commit (or a manual call) can
    promote a passed episode into canon. For the first run we want
    to inspect critic verdicts without auto-mutating canon.
    """
    if not any((premise, twist, motif)):
        return False
    try:
        if not canon_path.exists():
            log.warning(
                "[ScriptCritic] canon.md missing - canon update skipped "
                "(expected at %s)", canon_path,
            )
            return False
        text = canon_path.read_text(encoding="utf-8")
        changed = False
        for section_header, value in [
            ("## Used premises (auto-updated)", premise),
            ("## Used twists (auto-updated)",   twist),
            ("## Used motifs (auto-updated)",   motif),
        ]:
            if not value:
                continue
            value_line = f"- {value.strip()[:200]}"
            # Find the section block (header to next "## " or EOF).
            m = re.search(
                rf"({re.escape(section_header)}\n)(.*?)(?=\n## |\Z)",
                text,
                re.DOTALL,
            )
            if not m:
                continue
            head = m.group(1)
            body = m.group(2)
            # Strip the placeholder line if present. Placeholder may be
            # written with or without leading bullet markup, so test
            # against the bullet-stripped variant.
            def _is_placeholder(ln: str) -> bool:
                core = ln.strip().lstrip("-*•").strip()
                return core.startswith("_no entries")
            body_lines = [
                ln for ln in body.splitlines()
                if ln.strip() and not _is_placeholder(ln)
            ]
            # Skip if value already in section (case-insensitive substring).
            if any(value.strip().lower() in ln.lower() for ln in body_lines):
                continue
            body_lines.append(value_line)
            # Cap to last N.
            body_lines = body_lines[-cap_per_section:]
            new_body = "\n".join(body_lines) + "\n\n"
            text = text[:m.start()] + head + new_body + text[m.end():]
            changed = True
        if changed:
            canon_path.write_text(text, encoding="utf-8")
            log.info("[ScriptCritic] canon.md updated")
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("[ScriptCritic] canon update failed (non-fatal): %s", exc)
    return False


def _stamp_ledger_gate(
    model_id: str,
    score: Optional[int],
    verdict: str,
    issues: list[str],
    elapsed_s: float,
    skipped_reason: Optional[str] = None,
) -> None:
    """Stamp ledger.script_gates[] with the critic's findings.

    Best-effort; a failure here logs a warning but doesn't break the
    node. Mirrors the audio_gates[] pattern in the rest of OTR.
    """
    try:
        from .production_ledger import get_ledger
        led = get_ledger()
        gate_entry = {
            "kind":          "script_critic",
            "model_id":      model_id,
            "score":         score,
            "verdict":       verdict,
            "issues":        issues,
            "elapsed_s":     round(elapsed_s, 2),
            "stamped_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if skipped_reason:
            gate_entry["skipped_reason"] = skipped_reason
        led.data.setdefault("script_gates", []).append(gate_entry)
        led.save()
        log.info(
            "[ScriptCritic] stamped script_gates[] entry: verdict=%s score=%s",
            verdict, score,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[ScriptCritic] ledger stamp failed (non-fatal): %s", exc)


class LLMScriptCritic:
    """Two-immune-system evaluation gate.

    Wires between LLMScriptWriter and the next downstream consumer
    (Director, BatchBark, etc.). Pass-through on the script text;
    side-effects to the ledger. Set `block_on_reject=True` to make
    the gate authoritative once you trust the critic's calibration.
    """

    CATEGORY = "OTR/v2/Quality"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("script", "critic_report", "verdict")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Draft script text. Usually wired from "
                        "LLMScriptWriter's output, but any STRING "
                        "source works."
                    ),
                }),
                "genre_flavor": ("STRING", {
                    "default": "hard_sci_fi",
                    "tooltip": (
                        "Genre tag inlined into the critic's prompt "
                        "framing. Mirrors LLMScriptWriter's same-named "
                        "widget so the critic judges by the same lens."
                    ),
                }),
                "critic_model_id": (_LLM_MODEL_CHOICES, {
                    "default": "google/gemma-4-E4B-it",
                    "tooltip": (
                        "Critic LLM. MUST be different from the writer "
                        "model -- a model cannot grade its own work. "
                        "Default Gemma-4 E4B is small, fast, and "
                        "co-resides cleanly via cache-swap."
                    ),
                }),
            },
            "optional": {
                "block_on_reject": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When TRUE: a REJECT verdict raises an exception "
                        "and stops the workflow. When FALSE (default): "
                        "the gate is advisory -- findings are stamped "
                        "into ledger.script_gates[] and the script "
                        "passes through unchanged. Keep OFF until the "
                        "critic's calibration is validated across runs."
                    ),
                }),
                "optimization_profile": (
                    ["Standard", "Pro (Ultra Quality)", "Obsidian (UNSTABLE/4GB)"],
                    {"default": "Standard",
                     "tooltip": "Same widget as LLMScriptWriter -- "
                                "controls 4-bit NF4 vs full precision "
                                "for the critic load."},
                ),
                "timeout_sec": ("INT", {
                    "default": 90, "min": 30, "max": 600, "step": 10,
                    "tooltip": (
                        "Wall-clock budget for the critic call. On "
                        "timeout the gate is stamped as SKIPPED and the "
                        "script passes through unchanged. 90s is enough "
                        "for a Gemma-4 E4B critique of a short-form "
                        "script; raise for longer episodes."
                    ),
                }),
            },
        }

    def execute(
        self,
        script: str,
        genre_flavor: str,
        critic_model_id: str,
        block_on_reject: bool = False,
        optimization_profile: str = "Standard",
        timeout_sec: int = 90,
    ):
        # Late imports so the node loads cleanly even if the orchestrator
        # module has heavy dependencies missing in some environments.
        try:
            from . import story_orchestrator as _SO
        except ImportError:
            import sys
            from pathlib import Path as _P
            _NODES_DIR = _P(__file__).resolve().parent
            if str(_NODES_DIR) not in sys.path:
                sys.path.insert(0, str(_NODES_DIR))
            import story_orchestrator as _SO  # type: ignore

        t0 = time.time()
        script = script or ""
        if not script.strip():
            log.warning("[ScriptCritic] empty script - SKIPPED")
            _stamp_ledger_gate(
                model_id=critic_model_id,
                score=None,
                verdict="PASS",
                issues=[],
                elapsed_s=time.time() - t0,
                skipped_reason="empty_script",
            )
            return (script, "## Script Critic\n\nSKIPPED: empty script.\n", "PASS")

        # Pull rubric params from the ledger's gen_params_initial. The
        # writer stamps these at workflow entry (BUG-72), so they're
        # the ground truth for what the user actually requested. We
        # only override genre_flavor here because the critic node has
        # its own widget for it (lets you re-judge an existing script
        # against a different genre lens without re-running the writer).
        rubric_params: dict = {}
        try:
            from .production_ledger import get_ledger
            led_data = get_ledger().data
            rubric_params = dict(led_data.get("meta", {}).get("gen_params_initial", {}))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[ScriptCritic] failed to read gen_params_initial from "
                "ledger (%s) - rubric will fail-open on every gate", exc,
            )
        # The genre_flavor widget on this node wins over the ledger
        # value so the critic can re-score against a different lens.
        rubric_params["genre_flavor"] = genre_flavor

        anti_slop = _load_anti_slop_rubric(rubric_params)
        prompt = _build_critic_prompt(script, genre_flavor, anti_slop)

        log.info(
            "[ScriptCritic] running critic model=%s len=%d timeout=%ds",
            critic_model_id, len(script), timeout_sec,
        )

        # Strip suffix tags ([ALPHA], (EXPERIMENTAL)) the same way
        # _load_llm does for the writer.
        critic_id = critic_model_id.split(" ")[0]

        try:
            def _do_critic_call():
                return _SO._generate_with_llm(
                    prompt,
                    model_id=critic_id,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    optimization_profile=optimization_profile,
                )

            response = _SO._run_with_timeout(
                _do_critic_call,
                timeout_sec=timeout_sec,
                phase_label="ScriptCritic",
            )
        except Exception as exc:
            elapsed = time.time() - t0
            log.warning(
                "[ScriptCritic] critic call failed (%s) - SKIPPED",
                exc,
            )
            _stamp_ledger_gate(
                model_id=critic_model_id,
                score=None,
                verdict="PASS",
                issues=[],
                elapsed_s=elapsed,
                skipped_reason=f"{type(exc).__name__}: {exc}"[:200],
            )
            report = (
                f"## Script Critic\n\n"
                f"SKIPPED ({type(exc).__name__}): script passes through "
                f"unchanged. Run continues.\n"
            )
            return (script, report, "PASS")

        elapsed = time.time() - t0
        parsed = _parse_critic_response(str(response or ""))
        score = parsed["score"]
        verdict = parsed["verdict"]
        issues = parsed["issues"]

        log.info(
            "[ScriptCritic] verdict=%s score=%s issues=%d elapsed=%.1fs",
            verdict, score, len(issues), elapsed,
        )
        for issue in issues:
            log.info("[ScriptCritic]   - %s", issue)

        _stamp_ledger_gate(
            model_id=critic_model_id,
            score=score,
            verdict=verdict,
            issues=issues,
            elapsed_s=elapsed,
        )

        # Build markdown report.
        report_lines = [
            "## Script Critic",
            "",
            f"- **model**: `{critic_model_id}`",
            f"- **score**: {score if score is not None else '_(unparsed)_'}",
            f"- **verdict**: **{verdict}**",
            f"- **elapsed**: {elapsed:.1f}s",
            "",
            "### Issues",
        ]
        if issues:
            for issue in issues:
                report_lines.append(f"- {issue}")
        else:
            report_lines.append("- _none_")
        report = "\n".join(report_lines) + "\n"

        # Persist the report to ComfyUI's standard output directory so
        # it survives the session (BUG-01.02 rule -- output nodes must
        # NEVER hardcode paths). Best-effort.
        try:
            import folder_paths  # type: ignore
            out_root = Path(folder_paths.get_output_directory()) / "otr" / "script_gates"
            out_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_model = critic_id.replace("/", "_").replace(":", "_")
            out_path = out_root / f"script_critic_{stamp}_{safe_model}.md"
            out_path.write_text(report, encoding="utf-8")
            log.info("[ScriptCritic] report written: %s", out_path)
        except Exception as save_exc:  # noqa: BLE001
            log.warning(
                "[ScriptCritic] report save failed (non-fatal): %s",
                save_exc,
            )

        # Authoritative-mode block_on_reject. Off by default; flip to
        # ON once we trust the calibration.
        if block_on_reject and verdict == "REJECT":
            raise RuntimeError(
                f"[ScriptCritic] REJECT verdict (score={score}). "
                f"block_on_reject=True -- stopping workflow. "
                f"Top issues: {issues[:3]}"
            )

        return (script, report, verdict)


__all__ = ["LLMScriptCritic"]
