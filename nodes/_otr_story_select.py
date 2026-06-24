"""nodes/_otr_story_select.py -- Best-of-N structural story-refine selector.

Local-only by DEFAULT, opt-in remote, DETERMINISTIC best-of-N OUTLINE selector
(2026-06-23, 4-round roundtable-converged). NOT a QA-reroll gate: candidates are
FRESH-GENERATED outline structures and the keep-best gate is a PURE deterministic
scorer -- never "ask the same model to try again on the same beats".

This module hosts:
  * StoryScore / score_outline  -- the pure structural scorer (chunk 2).
  * select_best_outline + resolve_best_of_n  -- the cast_seed-keyed selector,
    flag parse, and provider gate (chunk 3); optional remote + cost guard
    (chunk 4).

The scorer runs on the RAW beat intents BEFORE any grounding: build_sq_data
MUTATES intent and substitutes the generic crisis nouns, which would zero out
ungrounded_crisis_density (the roundtable R3 catch). build_sq_data still runs
exactly ONCE downstream on the winning outline -- never here.

Dependency note: this imports only the stdlib-leaf _otr_story_quality_l12 public
helpers at module load (no torch, no _otr_outline cycle). torch is imported
LOCALLY inside select_best_outline (the writer forbids module-level torch).

UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List

# Public L1/L2 helpers. _otr_story_quality_l12 is a stdlib-only leaf
# (hashlib/re/unicodedata) that never imports _otr_outline, so a module-level
# import here forms no cycle and pulls no heavy deps. Package import in
# production; flat import when loaded standalone / under test.
try:
    from ._otr_story_quality_l12 import (
        BEAT_ROLE_IRREVERSIBLE_CHOICE,
        assign_beat_roles,
        count_ungrounded_crisis,
        premise_noun_palette,
        premise_texts,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_story_quality_l12 import (  # type: ignore
        BEAT_ROLE_IRREVERSIBLE_CHOICE,
        assign_beat_roles,
        count_ungrounded_crisis,
        premise_noun_palette,
        premise_texts,
    )

log = logging.getLogger("OTR")

# Token rule MIRRORS _otr_story_quality_l12._TOKEN_RE so the scorer tokenizes
# beat intents identically to count_ungrounded_crisis / premise_noun_palette
# (that symbol is module-private there; re-declared here, not imported, so the
# scorer never depends on a private name).
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

# Voiced == character + announcer, matching _otr_story_quality_l12._is_voiced
# -- the exact scope build_sq_data grounds over. Announcer bookends are voiced
# (Kokoro renders them). Kept local so the scorer never reaches a private name.
_VOICED_ROLES = ("character", "announcer")


def _is_voiced(role: str) -> bool:
    return role in _VOICED_ROLES


# ---------------------------------------------------------------------------
# Refine loop (v1, 2026-06-23) -- critique normalization
# ---------------------------------------------------------------------------
# A grader's biggest_weakness is LLM-generated free text fed back INTO the next
# refine pass's outline prompt (as OutlineRequest.prior_critique). Sanitize it
# to a single bounded line and reject prompt-injection-shaped instructions.
# Empty / malformed / injection => "" (preserves the byte-identical path).
_CRITIQUE_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?(previous|system|developer)\s+"
    r"(instructions|messages|prompt)",
    re.IGNORECASE,
)


def critique_to_hint(biggest_weakness: Any) -> str:
    """Normalize a grader weakness into a single-line, <=200-char, injection-safe
    structural hint for the next refine pass. Returns "" on empty / malformed /
    injection so the next pass's prompt stays byte-identical."""
    s = str(biggest_weakness or "")
    s = s.replace("```", " ").replace("`", " ")     # code fences / backticks
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)           # control chars incl \n \t
    s = re.sub(r"\s+", " ", s).strip()               # collapse -> single line
    if not s or _CRITIQUE_INJECTION_RE.search(s):
        return ""
    if len(s) > 200:
        s = s[:200].rsplit(" ", 1)[0].strip()        # trim at a word boundary
    return s


# ---------------------------------------------------------------------------
# Refine loop (v1) -- holistic story grader (read-only; never breaks the writer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StoryGrade:
    """A holistic 0-100 STRUCTURAL grade for ONE composed story + the single
    biggest structural weakness to fix. error_type is None on success and
    "grader_unparseable" when the grader call/parse failed (=> floor grade)."""

    score_0_100: int
    biggest_weakness: str
    error_type: Any = None


def extract_spoken_text_for_grade(ledger: Any, *, max_chars: int = 4000) -> str:
    """Pull the SPOKEN dialogue from a composed ledger as "SPEAKER: line" rows
    (voiced = character + announcer; music/sfx excluded) so the grader can judge
    character consistency. Caps at max_chars (head + "\\n...\\n" + tail). Accepts
    a Ledger object (reads ``.data["lines"]``) or a raw dict / list of rows."""
    data = getattr(ledger, "data", ledger)
    if isinstance(data, dict):
        rows = data.get("lines", []) or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    parts: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("speaker_role", "") or "") not in _VOICED_ROLES:
            continue
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        speaker = str(row.get("speaker", "") or "").strip() or "?"
        parts.append(f"{speaker}: {text}")
    joined = "\n".join(parts)
    if len(joined) > max_chars:
        half = max_chars // 2
        joined = joined[:half] + "\n...\n" + joined[-half:]
    return joined


def grade_story(composed_text: Any, premise: Any, *, generate_fn) -> "StoryGrade":
    """Grade a composed story 0-100 on STRUCTURE (arc / rising stakes / premise
    grounding / character want) via the existing ``structured_call`` ladder at
    LOW temperature (base 0.1 / structural-retry 0.0 -- structured_call requires
    the retry to be strictly lower than the base; determinism per pass comes from
    the caller seeding the RNG before the grade call). Read-only; NEVER raises
    into the writer -- any failure returns a floor grade
    (error_type="grader_unparseable")."""
    try:
        from ._otr_structured_call import structured_call
        from ._otr_repair_prompts import make_dispatching_repair_factory
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_structured_call import structured_call  # type: ignore
        from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    from pydantic import BaseModel, Field

    class _StoryGradeSchema(BaseModel):
        score: int = Field(..., ge=0, le=100)
        biggest_weakness: str = Field("", max_length=200)

    system = (
        "You are a tough story editor grading a short science-fiction audio "
        "drama on a 0-100 scale (A~=90, B+~=80, B~=75, C+~=68). Judge STRUCTURE "
        "only -- dramatic arc, rising stakes, premise grounding, clear character "
        "wants -- not prose polish. Return ONE JSON object "
        "{\"score\": int 0-100, \"biggest_weakness\": str} where "
        "biggest_weakness names the SINGLE biggest STRUCTURAL flaw to fix (an "
        "arc / stakes / grounding / character problem), NOT a line edit. No "
        "prose, no fences."
    )
    user = (
        f"Premise: {str(premise or '').strip()}\n\n"
        f"Script:\n{str(composed_text or '').strip()}\n\n"
        "Grade it. Return only the JSON object."
    )
    try:
        # LLM slot: creative -- the holistic story grade (refine loop) reuses
        # the writer's creative generate_fn (the model that wrote the story).
        res = structured_call(
            prompt=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=_StoryGradeSchema,
            slot_fn=generate_fn,
            base_temperature=0.1,
            structural_retry_temperature=0.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=128,
            max_attempts=2,
            helper_name="OTR_StoryGrade",
        )
        score = max(0, min(100, int(res.score)))
        weakness = str(getattr(res, "biggest_weakness", "") or "").strip()
        return StoryGrade(score_0_100=score, biggest_weakness=weakness,
                          error_type=None)
    except Exception as exc:  # noqa: BLE001 -- the grader must NEVER break the writer
        log.warning(
            "[refine] grade_story failed (%s); using floor grade",
            type(exc).__name__,
        )
        return StoryGrade(score_0_100=0, biggest_weakness="grader_unparseable",
                          error_type="grader_unparseable")


# ---------------------------------------------------------------------------
# Refine loop (v1) -- flag / widget / provider gate
# ---------------------------------------------------------------------------
_REFINE_MAX_PASSES = 5
# Node dropdown labels -> target grade bar (0-100). "Off" / unknown => 0 (off).
_REFINE_GRADE_MAP = {"off": 0, "c+": 68, "b": 75, "b+": 80, "a": 90}


@dataclass
class RefineConfig:
    requested_passes: int
    effective_passes: int   # TOTAL candidate count incl. the mandatory pass 0
    max_passes: int
    bar: int
    target_grade: str
    provider: str
    clamp_reason: str
    override_source: str


def _parse_refine_bar(widget_target: Any, env) -> tuple:
    """Resolve the target-grade bar (0-100). Env OTR_STORY_REFINE_BAR overrides
    the widget. Returns (bar, target_grade_label, override_source)."""
    raw_env = str(env.get("OTR_STORY_REFINE_BAR", "") or "").strip()
    if raw_env:
        try:
            return max(0, min(100, int(raw_env))), raw_env, "env_bar"
        except ValueError:
            log.warning(
                "[refine] OTR_STORY_REFINE_BAR=%r is not an int; using the "
                "widget", raw_env,
            )
    label = str(widget_target or "Off").strip()
    return _REFINE_GRADE_MAP.get(label.lower(), 0), label, "widget"


def resolve_refine_passes(creative_writing_model: Any, *, widget_target="Off",
                          env=None) -> "RefineConfig":
    """Resolve the refine config. ``effective_passes`` is the TOTAL candidate
    count INCLUDING the mandatory pass 0. Disabled (Off / passes<2 / remote
    writer) => effective_passes=1 (the byte-identical single path; the loop
    short-circuits and the writer runs once)."""
    import os
    if env is None:
        env = os.environ
    bar, target_grade, override_source = _parse_refine_bar(widget_target, env)
    model = str(creative_writing_model or "")

    raw_passes = str(env.get("OTR_STORY_REFINE_PASSES", "") or "").strip()
    if raw_passes:
        try:
            requested = int(raw_passes)
        except ValueError:
            log.warning(
                "[refine] OTR_STORY_REFINE_PASSES=%r is not an int; refine "
                "disabled", raw_passes,
            )
            requested = 1
    else:
        # Widget-driven: a real target grade keeps trying up to the cap.
        requested = _REFINE_MAX_PASSES if bar > 0 else 1

    if bar <= 0:   # Off / no target grade => disabled
        return RefineConfig(1, 1, _REFINE_MAX_PASSES, 0, target_grade, model,
                            "disabled", override_source)
    effective = max(1, min(requested, _REFINE_MAX_PASSES))
    clamp_reason = "max_5" if requested > _REFINE_MAX_PASSES else ""
    if effective < 2:
        return RefineConfig(requested, 1, _REFINE_MAX_PASSES, bar, target_grade,
                            model, "passes_lt_2", override_source)
    if model.startswith(("openrouter:", "comfy:")):
        log.warning(
            "[refine] remote creative writer %r -> refine clamped to 1 pass "
            "(local-only)", model,
        )
        return RefineConfig(requested, 1, _REFINE_MAX_PASSES, bar, target_grade,
                            model, "remote_provider_local_only", override_source)
    return RefineConfig(requested, effective, _REFINE_MAX_PASSES, bar,
                        target_grade, model, clamp_reason, override_source)


# ---------------------------------------------------------------------------
# T2 (2026-06-23) -- critic-axes adapter + keep-best + escalation gate
# ---------------------------------------------------------------------------
# THE grounding catch: StoryCriticReport exposes arc_verdict / reroll_targets /
# flat_lines / continuity_issues / render_priority -- NOT failing_axes /
# regeneration_hint (which _otr_reroll_escalation.decide_escalation_scope
# consumes). This adapter bridges the two so the 5B critic the pipeline already
# runs can actually buy a structural re-plan instead of the refine loop revising
# against only grade_story.biggest_weakness.
#
# arc_verdict -> structural failing axes (a subset of the escalation router's
# STRUCTURAL_AXES = {premise_clarity, continuity, resolution, emotional_arc}).
# A 'strong' arc is no failure; the others map to arc/resolution failures.
_ARC_VERDICT_AXES = {
    "strong": (),
    "uneven": ("emotional_arc",),
    "flat": ("emotional_arc",),
    "mid_collapse": ("resolution", "emotional_arc"),
}


def _target_hint(t: Any) -> str:
    h = getattr(t, "hint", None)
    if h is None and isinstance(t, dict):
        h = t.get("hint")
    return str(h or "").strip()


def critic_report_to_refine_signals(report: Any) -> tuple:
    """ADAPTER (T2): map a StoryCriticReport -> ``(failing_axes, regeneration_hint)``.

    failing_axes  <- arc_verdict (via _ARC_VERDICT_AXES) + 'continuity' when the
                     report names continuity_issues.
    regeneration_hint <- the distinct ``reroll_targets[].hint`` set (bounded),
                     else an arc-verdict summary (else "").
    PURE; never raises -- degrades to ``([], "")`` so it can never break the
    cascade or the refine loop."""
    try:
        arc = str(getattr(report, "arc_verdict", "") or "").strip().lower()
        axes: List[str] = list(_ARC_VERDICT_AXES.get(arc, ()))
        if (getattr(report, "continuity_issues", None) or []) and \
                "continuity" not in axes:
            axes.append("continuity")
        hints: List[str] = []
        for t in (getattr(report, "reroll_targets", None) or []):
            h = _target_hint(t)
            if h and h not in hints:
                hints.append(h)
        if hints:
            regen = "; ".join(hints)
        elif arc and arc != "strong":
            regen = (
                f"the dramatic arc verdict is '{arc}'; strengthen the rising "
                f"stakes and land a decisive on-stage resolution"
            )
        else:
            regen = ""
        if len(regen) > 400:
            regen = regen[:400].rsplit(" ", 1)[0]
        return axes, regen
    except Exception:  # noqa: BLE001 -- the adapter must never break a run
        return [], ""


def critic_escalation_enabled(env=None) -> bool:
    """True when OTR_ENABLE_CRITIC_ESCALATION is truthy. Default OFF => the
    freeze cascade passes the empty Stage-7 signal exactly as today
    (byte-identical). ON => the 5B critic's arc_verdict drives structural
    escalation via the adapter above."""
    import os
    if env is None:
        env = os.environ
    return _env_truthy(env.get("OTR_ENABLE_CRITIC_ESCALATION", ""))


def build_escalation_signal(story_critic_report: Any, meta: Any, *, env=None) -> dict:
    """T2 freeze-cascade helper. Returns the critic_result dict to feed
    ``decide_escalation_scope``. Default OFF (OTR_ENABLE_CRITIC_ESCALATION) =>
    ``{}`` (byte-identical: the cascade falls to LINE/NONE from legacy targets).
    ON => the adapter maps the 5B critic's arc_verdict into a STRUCTURAL signal
    (a non-strong arc -> EPISODE regenerate) and stamps
    meta.story_quality.critic_*. NEVER raises -> degrades to ``{}``."""
    try:
        if not critic_escalation_enabled(env):
            return {}
        axes, hint = critic_report_to_refine_signals(story_critic_report)
        if isinstance(meta, dict):
            sq = meta.setdefault("story_quality", {})
            if isinstance(sq, dict):
                sq["critic_failing_axes"] = list(axes)
                sq["critic_regeneration_hint"] = hint
        return {
            "verdict": "discard" if axes else "ship",
            "failing_axes": list(axes),
            "regeneration_hint": hint,
        }
    except Exception:  # noqa: BLE001 -- the gate must never break the freeze
        return {}


def keep_best_index(scores: List[int]) -> int:
    """Index of the HIGHEST score; ties resolve to the EARLIEST pass. Mirrors the
    refine loop's keep-best comparator (max grade, then -pass_index). Empty =>
    0. The refine loop is non-monotonic by design (a live gemma pass went
    72 -> 65), so keep-best must NOT drift to the last pass."""
    if not scores:
        return 0
    best_i = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best_i]:
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Chunk 2 -- pure structural scorer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StoryScore:
    """Pure, deterministic structural score for ONE candidate outline.

    Lower ``ungrounded_crisis_density`` is better; higher
    ``distinct_conflict_nouns`` and ``premise_grounding`` are better. Computed
    on the RAW beat intents BEFORE grounding. ``character_want_clarity`` and
    ``winner_grade`` were CUT from v0 (no wants data at this stage; the grade
    is unused by the comparator)."""

    ungrounded_crisis_density: float
    distinct_conflict_nouns: int
    premise_grounding: float
    # T4 (2026-06-23) deterministic staging penalty. 0.0 by DEFAULT so an
    # existing StoryScore built the old way is unchanged + still compares equal
    # (the field defaults, positional construction unaffected). Non-zero ONLY
    # when select_best_outline runs with a non-None ``penalty`` weight; folded
    # into the selection comparator (NOT into telemetry) -- adding 0.0 keeps the
    # penalty-None path byte-identical.
    staging_penalty: float = 0.0


# ---------------------------------------------------------------------------
# T4 (2026-06-23) -- deterministic staging penalty (climax on-mic)
# ---------------------------------------------------------------------------
# The cross-episode "console standoff" anti-pattern lands the climax OFF-stage:
# the decisive character beat is followed by an announcer outro that narrates
# the outcome. The dramatic-function spine (_otr_story_quality_l12) already
# guarantees the climax beat = the LAST voiced CHARACTER beat
# (BEAT_ROLE_IRREVERSIBLE_CHOICE). This penalty fires when that climax is NOT
# the final VOICED beat overall (an announcer beat trails it) OR has no intent,
# steering select_best_outline toward outlines whose climax lands on-mic.
_STAGING_PENALTY_WEIGHT = 50.0


def _otr_staging_penalty(outline: Any, *, penalty: float = _STAGING_PENALTY_WEIGHT) -> float:
    """Return ``penalty`` when the climax (the irreversible_choice character
    beat) is NOT the final voiced beat OR has an empty intent; else 0.0. PURE +
    deterministic; never mutates the outline. ``penalty`` is the violation
    weight (default 50.0). On any malformed input degrades to 0.0 (no penalty)
    so it never makes a candidate spuriously lose."""
    try:
        beats = list(getattr(outline, "beats", None) or [])
    except TypeError:  # pragma: no cover -- non-iterable beats
        return 0.0

    voiced: List[Any] = []
    char_ids: List[str] = []
    id_to_beat: dict = {}
    for i, b in enumerate(beats):
        role = str(getattr(b, "speaker_role", "") or "")
        if not _is_voiced(role):
            continue
        voiced.append(b)
        if role == "character":
            bid = str(getattr(b, "beat_id", "") or "") or f"_idx{i}"
            char_ids.append(bid)
            id_to_beat[bid] = b

    if not char_ids or not voiced:
        # No on-stage climax at all -> the worst staging outcome.
        return float(penalty)

    roles = assign_beat_roles(char_ids)
    climax_ids = [bid for bid in char_ids
                  if roles.get(bid) == BEAT_ROLE_IRREVERSIBLE_CHOICE]
    if not climax_ids:  # pragma: no cover -- assign_beat_roles always tags one
        return float(penalty)
    climax_beat = id_to_beat.get(climax_ids[-1])

    # On-mic == the climax character beat is the LAST voiced beat overall (no
    # announcer outro narrating the outcome after it).
    if voiced[-1] is not climax_beat:
        return float(penalty)
    if not str(getattr(climax_beat, "intent", "") or "").strip():
        return float(penalty)
    return 0.0


def resolve_staging_penalty(env=None):
    """Resolve the staging-penalty weight from OTR_ENABLE_STAGING_PENALTY.

    Unset / 0 / false => ``None`` (the byte-identical OFF path). 1/true/on =>
    the default weight; a bare number => that weight. Ships DARK (default OFF)."""
    import os
    if env is None:
        env = os.environ
    raw = str(env.get("OTR_ENABLE_STAGING_PENALTY", "") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return None
    if raw in ("1", "true", "yes", "on"):
        return _STAGING_PENALTY_WEIGHT
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "[staging_penalty] OTR_ENABLE_STAGING_PENALTY=%r is not a number; "
            "staging penalty OFF", raw,
        )
        return None


def score_outline(outline: Any, meta: Any, roster: Any, *,
                  penalty: float | None = None) -> StoryScore:
    """Score a candidate ``outline`` structurally. PURE -- never mutates the
    outline / beats, never calls build_sq_data.

    Metrics (all over VOICED beats = character + announcer):
      * ``ungrounded_crisis_density`` = (sum of count_ungrounded_crisis(
        beat.intent, grounded)) / max(1, total voiced-intent tokens). The
        cross-episode "console standoff" sameness signal -- lower is better.
      * ``distinct_conflict_nouns`` = number of DISTINCT premise-grounded
        content tokens surfaced across the voiced beat intents -- higher better.
      * ``premise_grounding`` = fraction of voiced beats whose intent references
        at least one premise/roster noun -- higher better.

    ``grounded`` palette = premise_noun_palette(roster, premise,
    *premise_texts(meta)) -- identical to build_sq_data's grounding source.
    """
    premise = str(getattr(outline, "premise", "") or "")
    grounded = premise_noun_palette(roster, premise, *premise_texts(meta))

    voiced_intents: List[str] = [
        str(getattr(b, "intent", "") or "")
        for b in (getattr(outline, "beats", None) or [])
        if _is_voiced(str(getattr(b, "speaker_role", "") or ""))
    ]

    total_voiced_beats = len(voiced_intents)
    total_voiced_intent_words = 0
    ungrounded_total = 0
    distinct_grounded: set = set()
    referencing_beats = 0

    for intent in voiced_intents:
        toks = _TOKEN_RE.findall(intent)
        total_voiced_intent_words += len(toks)
        ungrounded_total += count_ungrounded_crisis(intent, grounded)
        beat_refs = False
        for tok in toks:
            low = tok.casefold()
            if low in grounded:
                distinct_grounded.add(low)
                beat_refs = True
        if beat_refs:
            referencing_beats += 1

    density = ungrounded_total / max(1, total_voiced_intent_words)
    grounding = referencing_beats / max(1, total_voiced_beats)
    return StoryScore(
        ungrounded_crisis_density=density,
        distinct_conflict_nouns=len(distinct_grounded),
        premise_grounding=grounding,
        # ``penalty`` is the PRE-COMPUTED staging penalty for THIS outline
        # (_otr_staging_penalty), passed in by select_best_outline. None (the
        # default, and every existing caller) => 0.0 => byte-identical.
        staging_penalty=0.0 if penalty is None else float(penalty),
    )


# ---------------------------------------------------------------------------
# Chunk 3 -- flag parse + provider gate + the selector
# ---------------------------------------------------------------------------
# Local writers run free (no paid call), so a generous cap. The tighter remote
# cap (_REMOTE_BEST_OF_N_MAX = 3) + the fail-closed cost guard live in chunk 4.
_LOCAL_BEST_OF_N_MAX = 6

# Remote best-of-N (chunk 4, OPT-IN). N candidates == N x a PAID call, so the
# remote cap is tighter and the cost guard below is fail-closed.
_REMOTE_BEST_OF_N_MAX = 3
# Operator's global spend / irreversible-action gate (CLAUDE.md). A worst-case
# estimate AT OR ABOVE this REFUSES remote best-of-N (clamps to N=1, LOUD).
_AUTONOMY_CEILING_USD = 20.0
# Conservative per-outline UPPER bounds (over-estimate bias, matching the
# backend's _estimate_request_tokens philosophy). One outline is a TREE of
# small stage calls; this generously bounds the whole tree.
_REMOTE_OUTLINE_TOKENS_EST = 40_000
# A frontier-model upper-bound price (~$60 / 1M tokens) so the USD estimate is
# never an under-count. Real per-candidate cost is recorded post-hoc from the
# backend (cost_usd) when the backend returns it.
_REMOTE_PRICE_PER_1K_TOKENS_USD = 0.06
# Fallback per-run token ceiling when the backend constant can't be imported.
_REMOTE_PER_RUN_TOKEN_CEILING = 300_000

# Deterministic, index-keyed STRUCTURAL-variation instructions (candidate i>=1).
# Each pushes the outline away from the "console standoff" sameness toward a
# different dramatic spine. Candidate 0 always uses "" (byte-identical prompt).
_DIVERSITY_HINTS = (
    "open on the personal stake of one character, not the institutional threat "
    "or the wider crisis",
    "make the central conflict interpersonal -- two people who want incompatible "
    "things -- rather than a race against a system or a countdown",
    "let the turning point be a character's private choice or admission, and keep "
    "the decisive moment on-stage between the characters",
    "ground every beat in the specific premise nouns (the actual contested thing) "
    "and avoid generic control-room hardware as the conflict",
    "structure it as an investigation or negotiation that escalates through what "
    "the characters learn about each other, not through external alarms",
)


def _diversity_hint_for(i: int) -> str:
    """Structural-variation instruction for candidate ``i`` (i >= 1).
    Deterministic by index; candidate 0 always uses "" (handled by the caller)."""
    return _DIVERSITY_HINTS[(i - 1) % len(_DIVERSITY_HINTS)]


def _parse_best_of_n_flag(raw: Any):
    """Parse OTR_STORY_BEST_OF_N -> (requested_n, effective_n, clamp_reason).

    blank / non-int / <=1 => disabled (1, 1, ...); LOUD warn on a non-int value.
    int >= 2 => requested_n = value, effective_n = min(value, _LOCAL_BEST_OF_N_MAX)."""
    s = str(raw if raw is not None else "").strip()
    if s == "":
        return 1, 1, ""
    try:
        val = int(s)
    except (TypeError, ValueError):
        log.warning(
            "[best_of_n] OTR_STORY_BEST_OF_N=%r is not an integer; best-of-N "
            "disabled (single outline path)", raw,
        )
        return 1, 1, "non_int_flag"
    if val <= 1:
        return 1, 1, ""
    requested_n = val
    effective_n = min(val, _LOCAL_BEST_OF_N_MAX)
    clamp_reason = "max_local_6" if val > _LOCAL_BEST_OF_N_MAX else ""
    return requested_n, effective_n, clamp_reason


def resolve_best_of_n(resolved: Any):
    """Resolve the effective candidate count -> (requested_n, effective_n,
    clamp_reason). Reads OTR_STORY_BEST_OF_N, then applies the provider gate.

    Provider gate: a LOCAL creative writer runs best-of-N freely (no paid call).
    A REMOTE writer (``openrouter:`` / ``comfy:``) clamps to N=1 UNLESS the
    operator opts in via OTR_STORY_BEST_OF_N_ALLOW_REMOTE (default OFF); on
    opt-in it applies the tighter remote cap (_REMOTE_BEST_OF_N_MAX) and the
    fail-closed cost guard BEFORE any paid call. When ``effective_n < 2`` the
    writer runs the existing single path -- no selector, no telemetry key (the
    byte-identical path)."""
    import os
    raw = os.environ.get("OTR_STORY_BEST_OF_N", "0")
    requested_n, effective_n, clamp_reason = _parse_best_of_n_flag(raw)
    if effective_n < 2:
        return requested_n, effective_n, clamp_reason

    model = str((resolved or {}).get("creative_writing_model", "") or "")
    if not model.startswith(("openrouter:", "comfy:")):
        return requested_n, effective_n, clamp_reason  # local writer -- free

    # Remote writer -- opt-in only.
    if not _env_truthy(os.environ.get("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", "")):
        log.warning(
            "[best_of_n] remote creative writer %r -> best-of-N clamped to N=1 "
            "(local-only by default; set OTR_STORY_BEST_OF_N_ALLOW_REMOTE=1 to "
            "opt in)", model,
        )
        return requested_n, 1, "remote_provider_local_only"

    # Remote opt-in: tighter cap, THEN the fail-closed cost guard (checked
    # BEFORE the first paid call).
    remote_n = min(effective_n, _REMOTE_BEST_OF_N_MAX)
    guard_n, guard_reason, _est_usd = remote_cost_guard(remote_n)
    if guard_n < 2:
        return requested_n, 1, guard_reason
    reason = "remote_max_3" if effective_n > _REMOTE_BEST_OF_N_MAX else "remote_ok"
    return requested_n, guard_n, reason


@dataclass
class _Candidate:
    index: int
    outline: Any
    score: Any        # StoryScore on success, None on generation failure.
    ok: bool
    error_type: Any   # str on failure, None on success.
    cost_usd: Any = None  # per-candidate remote spend (USD) when measured, else None.


def _merge_best_of_n_telemetry(meta: Any, effective_n: int, winner_index: int,
                               candidates: List["_Candidate"]) -> None:
    """Merge the best_of_n telemetry block into meta.story_quality (never
    replace the dict -- consistent with the L5a setdefault/update rule). plain
    JSON primitives only. ``requested_n`` + ``clamp_reason`` are placeholders
    here; the writer stamps the real gate-derived values after the selector
    returns (it owns the flag parse + provider gate)."""
    if not isinstance(meta, dict):
        return
    scores = []
    for c in candidates:
        if c.ok:
            scores.append({
                "candidate_index": c.index,
                "ok": True,
                "ungrounded_crisis_density": c.score.ungrounded_crisis_density,
                "distinct_conflict_nouns": c.score.distinct_conflict_nouns,
                "premise_grounding": c.score.premise_grounding,
                "cost_usd": c.cost_usd,
            })
        else:
            scores.append({
                "candidate_index": c.index,
                "ok": False,
                "error_type": c.error_type,
                "cost_usd": c.cost_usd,
            })
    sq = meta.setdefault("story_quality", {})
    if isinstance(sq, dict):
        sq["best_of_n"] = {
            "requested_n": effective_n,   # writer overwrites with the true value
            "effective_n": effective_n,
            "winner_index": winner_index,
            "scores": scores,
            "clamp_reason": "",           # writer overwrites with the true value
        }


def select_best_outline(generate_outline_fn, outline_req, *, cast_seed, n, meta,
                        roster, cost_probe=None, penalty: float | None = None):
    """Generate ``n`` candidate outlines under cast_seed-keyed seeds + structural
    diversity_hints, score each with the PURE scorer, and return the best.

    Determinism: candidate ``i`` is seeded with sha256(f"{cast_seed}:outline:{i}").
    Candidate 0 uses diversity_hint="" (byte-identical PROMPT); i>=1 gets a
    structural-variation instruction. Each generation is wrapped in
    try/except OutlineFailedError -> LOUD + continue. Keep-best comparator:
    (ungrounded_crisis_density asc, distinct_conflict_nouns desc,
    premise_grounding desc, candidate index asc). Never-fail: if every candidate
    raised, run ONE deterministic fallback at the i=0 seed + hint=""; if THAT
    raises too, fail LOUD. build_sq_data is NOT called here (runs once
    downstream on the winner). Telemetry is merged into meta.story_quality.
    ``cost_probe`` (optional, remote only) returns cumulative remote spend in
    USD; when given, per-candidate cost_usd deltas are recorded in telemetry."""
    import hashlib
    import random
    import dataclasses
    import torch  # local import; module-level torch is forbidden in the writer.
    try:
        from ._otr_outline import OutlineFailedError
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_outline import OutlineFailedError  # type: ignore

    def _seed_rngs(idx: int) -> None:
        h = hashlib.sha256(
            f"{cast_seed}:outline:{idx}".encode("utf-8")
        ).hexdigest()
        seed_int = int(h, 16)
        # Seed IMMEDIATELY before the call. Best-effort for local backends;
        # remote backends may ignore process seeds -- diversity_hint is the
        # primary diversity lever there (chunk 4).
        torch.manual_seed(seed_int % (2 ** 64))
        random.seed(seed_int % (2 ** 32))

    candidates: List[_Candidate] = []
    for i in range(n):
        hint = "" if i == 0 else _diversity_hint_for(i)
        req_i = dataclasses.replace(outline_req, diversity_hint=hint)
        _seed_rngs(i)
        cost_before = cost_probe() if cost_probe else None
        try:
            outline_i = generate_outline_fn(req_i)
        except OutlineFailedError as exc:
            cost_i = _cost_delta(cost_probe, cost_before)
            log.warning(
                "[best_of_n] candidate %d/%d FAILED to generate (%s); "
                "skipping", i, n, type(exc).__name__,
            )
            candidates.append(_Candidate(i, None, None, False,
                                         type(exc).__name__, cost_i))
            continue
        cost_i = _cost_delta(cost_probe, cost_before)
        # T4 staging penalty: None => byte-identical (no penalty computed, 0.0
        # folded into the comparator below). Non-None => deterministic on-mic
        # climax steering computed INSIDE this best-of-N loop (post-outline,
        # pre-composition) so it steers selection.
        if penalty is None:
            score_i = score_outline(outline_i, meta, roster)
        else:
            _sp = _otr_staging_penalty(outline_i, penalty=penalty)
            score_i = score_outline(outline_i, meta, roster, penalty=_sp)
        candidates.append(_Candidate(i, outline_i, score_i, True, None, cost_i))
        log.info(
            "[best_of_n] candidate %d/%d scored: density=%.4f distinct=%d "
            "grounding=%.3f", i, n, score_i.ungrounded_crisis_density,
            score_i.distinct_conflict_nouns, score_i.premise_grounding,
        )

    ok = [c for c in candidates if c.ok]
    if ok:
        # Primary key folds the T4 staging penalty INTO the density score
        # ("subtract from the final score"; lower-is-better so we ADD it). The
        # penalty (0 or the weight, e.g. 50) dominates the small density float,
        # so a compliant on-mic climax beats an off-mic one. staging_penalty is
        # 0.0 on the penalty-None path => this is byte-identical to the
        # pre-T4 comparator.
        winner = min(ok, key=lambda c: (
            c.score.ungrounded_crisis_density + c.score.staging_penalty,
            -c.score.distinct_conflict_nouns,
            -c.score.premise_grounding,
            c.index,
        ))
        log.info(
            "[best_of_n] winner = candidate %d (%d of %d generated; "
            "density=%.4f distinct=%d grounding=%.3f)",
            winner.index, len(ok), n,
            winner.score.ungrounded_crisis_density,
            winner.score.distinct_conflict_nouns,
            winner.score.premise_grounding,
        )
        _merge_best_of_n_telemetry(meta, n, winner.index, candidates)
        return winner.outline

    # Never-fail: every candidate raised. ONE deterministic fallback at the i=0
    # seed + diversity_hint="" (NOT "normal"). If THAT raises too, fail LOUD.
    log.warning(
        "[best_of_n] all %d candidate(s) failed to generate; running ONE "
        "deterministic fallback (i=0 seed, no hint)", n,
    )
    _seed_rngs(0)
    req0 = dataclasses.replace(outline_req, diversity_hint="")
    outline0 = generate_outline_fn(req0)  # may raise OutlineFailedError -> LOUD
    _merge_best_of_n_telemetry(meta, n, 0, candidates)
    return outline0


# ---------------------------------------------------------------------------
# Chunk 4 -- optional remote best-of-N + fail-closed cost guard
# ---------------------------------------------------------------------------
def _env_truthy(val: Any) -> bool:
    return str(val if val is not None else "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _cost_delta(cost_probe, before):
    """Per-candidate spend = cumulative-after minus cumulative-before. Returns
    None when no probe is wired (local writers). Best-effort; never raises."""
    if cost_probe is None or before is None:
        return None
    try:
        return float(cost_probe()) - float(before)
    except Exception:  # noqa: BLE001 -- cost accounting must never break a run
        return None


def _per_run_token_ceiling() -> int:
    """The OpenRouter backend's per-run token ceiling when importable, else the
    local fallback constant -- so guard (a) reuses the writer's real budget."""
    try:
        from ._otr_openrouter_backend import DEFAULT_MAX_TOKENS_PER_RUN as _C
        return int(_C)
    except Exception:  # noqa: BLE001
        try:
            from _otr_openrouter_backend import (  # type: ignore
                DEFAULT_MAX_TOKENS_PER_RUN as _C,
            )
            return int(_C)
        except Exception:  # noqa: BLE001
            return _REMOTE_PER_RUN_TOKEN_CEILING


def estimate_remote_cost(effective_n, *, per_outline_tokens=None,
                         price_per_1k=None):
    """Conservative worst-case (tokens, usd) for ``effective_n`` remote outline
    generations. Over-estimate bias so the guard errs toward refusing."""
    pot = (_REMOTE_OUTLINE_TOKENS_EST if per_outline_tokens is None
           else per_outline_tokens)
    ppk = (_REMOTE_PRICE_PER_1K_TOKENS_USD if price_per_1k is None
           else price_per_1k)
    worst_tokens = int(effective_n) * int(pot)
    worst_usd = (worst_tokens / 1000.0) * float(ppk)
    return worst_tokens, worst_usd


def remote_cost_guard(effective_n, *, per_outline_tokens=None, price_per_1k=None,
                      per_run_ceiling=None, autonomy_ceiling=None):
    """Fail-closed cost guard, checked BEFORE the first paid call. Returns
    ``(allowed_n, reason, est_usd)``:
      (a) worst-case tokens > the per-run token ceiling  -> clamp to N=1;
      (b) worst-case USD >= the $20 autonomy ceiling      -> clamp to N=1;
      (c) otherwise LOUD-log the estimate + proceed at ``effective_n``."""
    ceiling_tokens = (_per_run_token_ceiling() if per_run_ceiling is None
                      else per_run_ceiling)
    ceiling_usd = (_AUTONOMY_CEILING_USD if autonomy_ceiling is None
                   else autonomy_ceiling)
    worst_tokens, worst_usd = estimate_remote_cost(
        effective_n,
        per_outline_tokens=per_outline_tokens,
        price_per_1k=price_per_1k,
    )
    if worst_tokens > ceiling_tokens:
        log.warning(
            "[best_of_n] remote worst-case ~%d tokens for N=%d exceeds the "
            "per-run token ceiling %d; clamping to N=1",
            worst_tokens, effective_n, ceiling_tokens,
        )
        return 1, "cost_guard_per_run_budget", worst_usd
    if worst_usd >= ceiling_usd:
        log.warning(
            "[best_of_n] remote worst-case ~$%.2f for N=%d >= the $%.0f "
            "autonomy ceiling; REFUSING remote best-of-N, clamping to N=1",
            worst_usd, effective_n, ceiling_usd,
        )
        return 1, "cost_guard_autonomy_ceiling", worst_usd
    log.warning(
        "[best_of_n] remote best-of-N: estimated worst-case spend ~$%.2f for "
        "N=%d outline generation(s); proceeding", worst_usd, effective_n,
    )
    return effective_n, "remote_ok", worst_usd
