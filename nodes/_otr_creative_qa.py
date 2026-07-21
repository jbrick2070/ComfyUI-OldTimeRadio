"""nodes/_otr_creative_qa.py -- Story Spine Targeted Story QA router.

Targeted Story QA is the opt-in judgment round in the story spine
(Narrative Draft -> conditional Radio Editor length pass -> Targeted Story
QA -> [dynamic scoped repair + re-judgment] -> Ledger Scrub -> TTS). It is a
READ-ONLY DEFECT ROUTER, not a grader: it reads the JUST-COMPOSED ledger
at beat granularity and returns a Pydantic-validated `StoryQAVerdict`
whose `verdict` is exactly one of three gates:

  * PASS                -- no severe defect; ship as-is.
  * MICRO_REPAIR_NEEDED -- a LOCALLY-FIXABLE defect (a chopped/stilted
                           line, a flabby or muddy beat) a one-line edit
                           can repair; `flagged_beats` names the beats.
  * REJECT              -- a broad story defect that scopes the guarded
                           repair pass to the whole voiced script before the
                           independent judge runs again. It is repair feedback,
                           never an episode-liveness verdict.

This is the go-forward refactor of the earlier PASS/REPAIR_ONCE/FAIL
grader (docs/otr-go-forward-final.md, Sprint 1). The router answers ONE
question -- "is there a severe defect, and can a tiny edit fix it?" -- and
emits categorical evidence flags (dead_ending / broken_turn /
flat_contrast / unclear_grounding / chopped_dialogue / pacing_failure)
that explain the verdict. It carries NO vanity 1-5 scores and NO
leak / SFW / JSON / speaker-id checks: those are the deterministic Ledger
Scrub's job (Sec 2 invariant; the scrub stays mechanical, fail-closed,
last), so the router never duplicates them.

When enabled, it is fully model-agnostic. There is NO
`critic_model != creative_model` auto-skip: a default single-model setup
must still get QA. Same-model QA is a weaker second opinion (lower recall
on the writer's own blind spots), but the fix is TASK DESIGN, not
disabling the gate: the call (a) sees only the FINAL script on a COLD
CONTEXT (no generation history to rationalize from), (b) uses a
SKEPTICAL / adversarial framing ("assume the writer was careless; find
why it fails"), and (c) holds a HIGH REJECT BAR (only severe, concrete
defects request whole-script repair). A model spots a concrete missing turn or chopped line in
its own output far better than it can judge overall quality, which is
exactly why the router flags DEFECTS, not quality. A different technical
model gives stronger recall and is recommended, not required.

In-process, not a node (D4/D5): this is a helper called inside the
spine's `run_post_script_spine`, mirroring `run_story_brief_reflection`.
It exposes no `INPUT_TYPES`, no node class, and no `model_id` widget
(Prime Directive 6). It receives its model id as a plain `str`
(`critic_model_id`) sourced from the writer's in-process
`resolved["technical_model"]`; the single call site is tagged
`# LLM slot: technical`.

The single LLM call routes through the shared `structured_call` 3-attempt
retry ladder (base -> structural retry -> typed repair). It NEVER raises
into the pipeline: an exhausted ladder or a raising slot fn is caught and
mapped to a fail-OPEN verdict (`verdict="PASS"`), so a QA pass that could
not run ships the episode as-is rather than aborting it (Prime Directive
1 -- audio output must never break over a taste-check failure; go-forward
Sec 8 -- "a QA crash does not break the run"). A fail-open PASS never
triggers a micro-repair and never aborts.

PD3 (workflow JSON): N/A; this module adds no node surface, so the
canonical workflow JSON needs no change (D5).

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation. Placeholders/stubs use descriptive names, never the
forbidden filler word (project rule 5).
"""
# LLM slot: technical
# Reason: the router emits a JSON-schema-constrained categorical verdict
# (a 3-value gate + evidence flags + beat indices), not creative prose,
# so per project rule 6 every structured-JSON pass routes to the
# technical model slot; and per spine D6 the router runs on the "other
# slot" (resolved["technical_model"]) so the writer never judges itself.
from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, Optional, Sequence

from pydantic import BaseModel, Field
from typing import Literal


log = logging.getLogger("OTR")


__all__ = [
    "StoryQAVerdict",
    "run_story_qa",
]


# ---------------------------------------------------------------------------
# Shared imports -- structured_call ladder + typed repair factory.
# Package import in production; flat import when the module is loaded
# standalone / under test. Mirrors the import guard in
# `_otr_story_critic.py` / `_otr_story_brief.py`. These are PURE-Python
# sibling modules (no torch / GPU / network at import), so importing
# them at module load is safe; the heavy LLM closure arrives only at
# call time via the `critic_generate_fn` argument.
# ---------------------------------------------------------------------------

try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
        PostValidationError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
        PostValidationError,
    )

try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


# ---------------------------------------------------------------------------
# Generation params for the single router LLM call.
# ---------------------------------------------------------------------------

# The verdict is a small, flat JSON object: a 3-value verdict, six bool
# flags, a short reason, and one index list. A modest token budget is
# generous. Matches the order of magnitude the structured-pass default
# targets.
_QA_MAX_NEW_TOKENS: int = 600

# Base attempt runs warm enough that the model engages with the judgment
# (is the turn real, does the ending pay off), not so warm it drifts off
# the schema. The structural retry runs STRICTLY COOLER -- the 2B
# principle: lowering entropy during a JSON-schema repair, never raising
# it. `structured_call` asserts this invariant at entry.
_QA_TEMPERATURE: float = 0.4
_QA_RETRY_TEMPERATURE: float = 0.2

# Per-beat text shown to the router is truncated at this many chars so a
# runaway line cannot blow the prompt budget. Mirrors the per-line cap
# `_otr_story_brief._PER_LINE_TEXT_CAP` uses for the reflection input.
_PER_BEAT_TEXT_CAP: int = 240


# ---------------------------------------------------------------------------
# Pydantic schema -- the defect-router verdict (go-forward Sprint 1).
# CATEGORICAL / OBSERVATIONAL ONLY -- a 3-value Literal verdict, six bool
# evidence flags, a short reason, and a beat-index list. No numeric
# quality score anywhere (Sec 2 invariant). No leak / SFW / JSON checks --
# those are the deterministic scrub's job, not the router's.
# ---------------------------------------------------------------------------


class StoryQAVerdict(BaseModel):
    """The read-only Targeted Story QA verdict over a composed episode.

    The router answers one question -- "is there a severe defect, and can a
    one-line edit fix it?" -- and routes to one of three gates. Beat
    indices in `flagged_beats` are positions into the DIALOGUE-beat list
    the router is shown (beat 0 = first character line), not ledger
    `line_id`s: a positional index is what a model returns reliably
    without echoing an opaque id back verbatim.

    Field roles:
      verdict            -- the single gate value: PASS / MICRO_REPAIR_NEEDED
                            / REJECT (whole-script repair scope).
      flagged_beats      -- the dialogue beats a micro-repair should fix
                            (MICRO_REPAIR_NEEDED only; empty otherwise).
      reason             -- one short phrase (about 15 words or fewer)
                            naming the defect behind the verdict.
      dead_ending        -- the episode stops without paying off its setup
                            (a broad defect -> REJECT repair scope).
      broken_turn        -- the central turn / reversal does not logically
                            follow (broad -> REJECT repair scope).
      flat_contrast      -- the voices read as one narrator split in two,
                            with no distinct register (often locally
                            fixable -> MICRO_REPAIR_NEEDED).
      unclear_grounding  -- the premise is never made clear; the listener
                            cannot tell what is going on (structural ->
                            REJECT repair scope).
      chopped_dialogue   -- a line is chopped, stilted, or truncated mid
                            thought (locally fixable -> MICRO_REPAIR_NEEDED).
      pacing_failure     -- a beat is flabby / overlong / stalls the scene
                            (locally fixable -> MICRO_REPAIR_NEEDED).

    All routing criteria live in the prompt, so this pass assumes only that
    the bound model "follows instructions and emits JSON" -- which the
    `structured_call` ladder guarantees.
    """

    verdict: Literal["PASS", "MICRO_REPAIR_NEEDED", "REJECT"]
    flagged_beats: List[int] = Field(default_factory=list)
    reason: str = ""
    dead_ending: bool = False
    broken_turn: bool = False
    flat_contrast: bool = False
    unclear_grounding: bool = False
    chopped_dialogue: bool = False
    pacing_failure: bool = False

    @classmethod
    def fail_open(cls, note: str = "story QA pass could not run") -> "StoryQAVerdict":
        """Return the fail-OPEN verdict used when the router cannot run.

        `verdict="PASS"` with every evidence flag False: a router that
        could not execute ships the episode AS-IS rather than aborting it.
        This is the audio-is-king direction (Prime Directive 1): a QA pass
        that crashes must never REJECT a good episode or trigger a
        spurious micro-repair on a critic failure. Returned by
        `run_story_qa` on ANY failure (an exhausted `structured_call`
        retry ladder or a raising slot fn), keeping the never-raises /
        never-falsely-abort contract absolute (go-forward Sec 8 -- a QA
        crash does not break the run).
        """
        return cls(
            verdict="PASS",
            flagged_beats=[],
            reason=note,
            dead_ending=False,
            broken_turn=False,
            flat_contrast=False,
            unclear_grounding=False,
            chopped_dialogue=False,
            pacing_failure=False,
        )


# ---------------------------------------------------------------------------
# Prompt construction -- ledger reading + the router system prompt.
# ---------------------------------------------------------------------------


def _ledger_data(led: Any) -> dict:
    """Accept either a Ledger object (with `.data`) or a raw dict.

    Duck-typed exactly like `_otr_story_brief._ledger_data` so a test can
    pass a plain dict without constructing a Ledger.
    """
    return getattr(led, "data", led)


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "..."


# The voiced roles the router judges. MUST match _otr_radio_editor's
# VOICED_ROLES (character + announcer), same order, because the spine
# passes a MICRO_REPAIR_NEEDED verdict's `flagged_beats` STRAIGHT to
# micro_repair, which interprets them in the editor's voiced view. If the
# two index spaces diverged, a flagged beat would repair the wrong line.
_VOICED_ROLES = ("character", "announcer")


def _dialogue_beats(ledger: dict) -> List[dict]:
    """Return the VOICED beats (character + announcer), in ledger order.

    These ARE the "beats" the router judges and the `flagged_beats` field
    positions into. The index space MUST match `_otr_radio_editor`'s
    `voiced_beats` (same roles, same order): the spine hands `flagged_beats`
    straight to `micro_repair`, which reads them in the editor's voiced
    view, so a mismatch would repair the wrong line. Music / sfx / scene
    rows are locked structural content and are excluded.

    Read keys, per the live ledger shape (see
    `_otr_story_brief._build_reflection_input`): `ledger["lines"]` is the
    per-line list; each row carries `speaker_role`, `char_id` (or
    `speaker`), and `text`.
    """
    lines: Sequence[dict] = ledger.get("lines") or []
    return [
        ln for ln in lines
        if isinstance(ln, dict)
        and (ln.get("speaker_role") or "").lower() in _VOICED_ROLES
    ]


def _build_story_qa_input(led: Any) -> str:
    """Build the router's user prompt from the FINAL ledger.

    Emits a compact, numbered roster of DIALOGUE beats -- one row per
    character line, prefixed `[beat N | char_id]` -- plus a short premise
    header (title + style slug) so the router can judge premise grounding.
    The beat NUMBER is the index `flagged_beats` must reference, so it is
    shown explicitly and the prompt tells the router to use it.

    COLD CONTEXT by construction: this builds the prompt from the final
    ledger alone -- there is no generation history, no prior draft, no
    chain-of-thought for the model to rationalize from. That is half of
    what keeps same-model QA honest (go-forward Sec 2 rule 1).

    READ-ONLY: reads `led.data` (or a raw dict) and returns a string; it
    never mutates the ledger.
    """
    ledger = _ledger_data(led)
    meta = ledger.get("meta") or {}

    title = (
        (meta.get("episode_title") or "").strip()
        or (meta.get("title") or "").strip()
        or (ledger.get("title") or "").strip()
    )
    style = (meta.get("style") or "").strip()

    parts: List[str] = []
    parts.append(f"TITLE: {title}")
    parts.append(f"STYLE: {style}")
    parts.append("")

    beats = _dialogue_beats(ledger)
    parts.append(f"DIALOGUE BEATS ({len(beats)}), numbered from 0:")
    if not beats:
        parts.append("  (no dialogue beats)")
    for index, beat in enumerate(beats):
        char = (beat.get("char_id") or beat.get("speaker") or "?").strip() or "?"
        text = _truncate((beat.get("text") or "").strip(), _PER_BEAT_TEXT_CAP)
        parts.append(f"  [beat {index} | {char}] {text}")

    parts.append("")
    parts.append(
        "Judge ONLY the numbered dialogue beats above. Every index in "
        "flagged_beats is the integer shown in [beat N]; use only indices "
        f"in range 0..{max(len(beats) - 1, 0)} and invent none. Then emit "
        "the single JSON StoryQAVerdict object."
    )
    return "\n".join(parts)


_QA_SYSTEM_PROMPT = """\
You are a skeptical script doctor doing a final defect check on an audio
drama. Assume the writer was careless and rushed; your job is to find
whether the episode has a SEVERE defect, and if so whether a small edit
can fix it. You are shown only the final spoken beats (the characters and
the announcer), numbered from 0. You do NOT rewrite anything; you only
diagnose and route.

Report only CATEGORICAL observations -- a verdict, yes/no evidence flags,
flagged beat indices, and one short reason. Do NOT score anything on a
numeric scale. Use the integer beat numbers exactly as shown in [beat N].

Decide the verdict by routing the worst defect you find:

REJECT -- a broad defect that needs the whole voiced script in repair scope:
  - dead_ending: the episode stops without paying off what it set up.
  - broken_turn: the central reversal or decision does not logically
    follow from what came before.
  - unclear_grounding: the premise is never made clear; a listener cannot
    tell what is happening or why.
  Set the matching flag(s) true and verdict REJECT. Hold a HIGH BAR:
  use this only for a concrete story-level defect, not a merely imperfect one.

MICRO_REPAIR_NEEDED -- a LOCALLY-FIXABLE defect a one-line edit repairs:
  - chopped_dialogue: a line is chopped, stilted, or cut off mid thought.
  - pacing_failure: a beat is flabby, overlong, or stalls the scene.
  - flat_contrast: two speakers sound like one voice and a line could be
    sharpened to distinguish them.
  Set the matching flag(s) true, verdict MICRO_REPAIR_NEEDED, and list the
  specific beats to fix in flagged_beats. Only flag beats a one-line edit
  can actually fix; never request a micro-repair for a structural defect.

PASS -- no severe defect, or only cosmetic nits. flagged_beats empty.

Give a SHORT reason (about 15 words or fewer) naming the defect behind
your verdict. Do NOT judge profanity, safety, name leaks, JSON, or speaker
ids -- a separate deterministic pass owns those; you judge STORY defects
only.

Return EXACTLY one JSON object matching this schema:
{
  "verdict": "PASS" | "MICRO_REPAIR_NEEDED" | "REJECT",
  "flagged_beats": [int, ...],
  "reason": "...",
  "dead_ending": true | false,
  "broken_turn": true | false,
  "flat_contrast": true | false,
  "unclear_grounding": true | false,
  "chopped_dialogue": true | false,
  "pacing_failure": true | false
}

flagged_beats may be empty. No prose before or after the JSON. No markdown
fences.
"""


# ---------------------------------------------------------------------------
# post_validator -- light flagged-beat range guard.
# ---------------------------------------------------------------------------


def _make_story_qa_post_validator(beat_count: int):
    """Build the optional content-check passed to `structured_call`.

    Deliberately LENIENT (the same posture as the story-critic's
    post_validator): it rejects a verdict ONLY when a returned
    `flagged_beats` index is out of range for the beats the router was
    shown -- a fabricated index would later point the micro-repair at a
    beat that does not exist. It does NOT second-guess the verdict value,
    the evidence flags, or how many beats were flagged: those are exactly
    what the router is for.

    Returns a `post_validator(verdict) -> str | None` closure: an error
    string REJECTS (and advances the ladder), `None` ACCEPTS.
    """

    def post_validator(verdict: "StoryQAVerdict") -> Optional[str]:
        # An empty beat list means there is nothing to range-check
        # against; accept whatever validated rather than reject every
        # verdict on a beat-free ledger.
        if beat_count <= 0:
            return None

        out_of_range = [
            index for index in verdict.flagged_beats
            if not (0 <= index < beat_count)
        ]
        if out_of_range:
            preview = ", ".join(str(i) for i in sorted(set(out_of_range))[:8])
            return (
                f"verdict flags {len(set(out_of_range))} beat index(es) out "
                f"of range 0..{beat_count - 1}: {preview}"
            )
        return None

    return post_validator


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_story_qa(
    led: Any,
    critic_generate_fn: Callable[..., str],
    *,
    critic_model_id: str,
) -> StoryQAVerdict:
    """Run the Targeted Story QA router and RETURN its verdict.

    Mirrors `run_story_brief_reflection`: same post-script call pattern,
    slot plumbing, and fail-loud-then-fail-open handling. READ-ONLY -- it
    reads the ledger and returns a verdict; it NEVER mutates a ledger key.
    The spine consumes the verdict: PASS proceeds, MICRO_REPAIR_NEEDED
    scopes repair to `flagged_beats`, and REJECT scopes repair to the whole
    voiced script. Every repair is independently rejudged; no quality verdict
    owns episode liveness.

    Parameters:
      led
        The Ledger object (`.data`) OR a raw ledger dict -- duck-typed via
        `_ledger_data` so a test can pass a plain dict. Read keys (live
        ledger shape): `ledger["lines"]` (per-line records: `speaker_role`,
        `char_id` / `speaker`, `text`) and `ledger["meta"]`
        (`episode_title` / `title`, `style`).
      critic_generate_fn
        The LLM generate callable -- `critic_generate_fn(messages, *,
        temperature, max_new_tokens[, stop]) -> str`. In production this is
        the writer's in-process technical-slot closure; no `model_id`
        widget exists anywhere (Prime Directive 6).
      critic_model_id  (keyword-only)
        The technical-slot model id, sourced from the writer's in-process
        `resolved["technical_model"]` (D6 -- the "other slot," so the
        writer never judges itself). Carried for logging / attribution;
        the actual generation goes through `critic_generate_fn`.

    NEVER raises (Prime Directive 1 -- audio output must never break). Any
    failure -- an exhausted `structured_call` retry ladder or a raising
    slot fn -- is logged and `StoryQAVerdict.fail_open()` is returned
    (`verdict="PASS"`), so a router that could not run ships the episode
    as-is and never falsely aborts it.
    """
    ledger = _ledger_data(led)
    beat_count = len(_dialogue_beats(ledger))

    user_prompt = _build_story_qa_input(led)
    messages = [
        {"role": "system", "content": _QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    post_validator = _make_story_qa_post_validator(beat_count)

    # The single router call routes through the shared structured_call
    # 3-attempt ladder (base -> structural retry -> typed repair). An
    # exhausted ladder raises StructuredCallFailedError; the slot fn (the
    # LLM closure) may raise arbitrary loader / VRAM exceptions
    # structured_call does NOT catch. BOTH are caught below -- every
    # failure path returns StoryQAVerdict.fail_open(), so a QA crash ships
    # the episode as-is rather than aborting it (Prime Directive 1).
    #
    # The router emits a JSON-schema-constrained categorical verdict, not
    # creative prose, so it routes to the technical model exactly as the
    # reflection and story-critic passes do. Per spine D6 the technical
    # slot is the "other slot" relative to the writer's creative passes,
    # so routing the router here means the writer never judges its own
    # narrative. The model id arrives as the `critic_model_id` str (from
    # the writer's resolved["technical_model"]) -- no new widget, no new
    # model_id parameter (Prime Directive 6).
    #
    # LLM slot: technical -- structured-JSON categorical verdict pass.
    try:
        verdict = structured_call(
            prompt=messages,
            schema=StoryQAVerdict,
            slot_fn=critic_generate_fn,
            base_temperature=_QA_TEMPERATURE,
            structural_retry_temperature=_QA_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=post_validator,
            max_new_tokens=_QA_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_story_qa",
        )
    except StructuredCallFailedError as exc:
        # Ladder exhausted. Attribute the failure to its class via the
        # last error the ladder captured, so a log reader can tell a
        # parse / schema / content failure apart at a glance.
        last = exc.last_error
        if isinstance(last, json.JSONDecodeError):
            reason = "json_parse_failed_after_repair"
        elif isinstance(last, PostValidationError):
            reason = "flagged_beat_validation_failed_after_repair"
        else:
            reason = "structured_call_failed"
        log.warning(
            "[OTR_StoryQA] structured_call exhausted the retry ladder after "
            "%d attempt(s) (model_id=%s, last error: %s); returning fail-open "
            "PASS verdict (reason=%s)",
            exc.attempts, critic_model_id, exc.last_error, reason,
        )
        return StoryQAVerdict.fail_open(reason)
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        # structured_call does not catch slot-fn exceptions: a VRAM /
        # loader / framework failure inside the critic_generate_fn closure
        # lands here. A bare reraise would crash the whole script
        # generation over a taste-check failure -- audio is king, so this
        # pass fails OPEN to PASS instead.
        log.warning(
            "[OTR_StoryQA] critic_generate_fn raised %s: %s (model_id=%s); "
            "returning fail-open PASS verdict",
            type(exc).__name__, exc, critic_model_id,
        )
        return StoryQAVerdict.fail_open("critic_generate_fn_exception")

    log.info(
        "[OTR_StoryQA] verdict=%s (model_id=%s): reason=%r, flagged=%s, "
        "dead_ending=%s, broken_turn=%s, flat_contrast=%s, "
        "unclear_grounding=%s, chopped_dialogue=%s, pacing_failure=%s",
        verdict.verdict, critic_model_id, verdict.reason, verdict.flagged_beats,
        verdict.dead_ending, verdict.broken_turn, verdict.flat_contrast,
        verdict.unclear_grounding, verdict.chopped_dialogue,
        verdict.pacing_failure,
    )
    return verdict


# ---------------------------------------------------------------------------
# Self-test -- stubs critic_generate_fn with canned JSON, no GPU/network.
# Prints `SELF-TEST PASS: N/N` on success and exits 0; any assertion
# failure raises (non-zero exit). Runnable with the venv python directly.
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Run the self-test suite; return the count of checks that passed.

    Stubs `critic_generate_fn` with closures that return canned JSON, so
    the suite runs with ZERO network and ZERO GPU. Exercises: a clean
    PASS, a MICRO_REPAIR_NEEDED with flagged beats, a structural REJECT,
    the empty-ledger path, the fail-open mapping when the slot fn raises,
    and the out-of-range flagged-beat post_validator guard.
    """
    import json as _json

    passed = 0
    total = 0

    # A small, representative ledger: a title + style premise header and
    # five character-dialogue beats, mixed with non-dialogue rows that
    # MUST be excluded from the beat numbering.
    ledger = {
        "meta": {"episode_title": "The Last Signal", "style": "noir_mystery"},
        "cast": [
            {"char_id": "c01", "name": "EDNA", "speaker_role": "character"},
            {"char_id": "c02", "name": "MARLOWE", "speaker_role": "character"},
        ],
        "lines": [
            {"line_id": "s00", "speaker_role": "music", "text": "[theme up]"},
            {"line_id": "l00", "speaker_role": "character", "char_id": "c01",
             "text": "The relay went dark at midnight."},
            {"line_id": "x00", "speaker_role": "music_inter", "text": "[static]"},
            {"line_id": "l01", "speaker_role": "character", "char_id": "c02",
             "text": "Then someone cut the line on purpose."},
            {"line_id": "l02", "speaker_role": "character", "char_id": "c01",
             "text": "I found the tool they left behind."},
            {"line_id": "l03", "speaker_role": "character", "char_id": "c02",
             "text": "That changes everything we assumed."},
            {"line_id": "l04", "speaker_role": "character", "char_id": "c01",
             "text": "Then we end this tonight, together."},
        ],
    }

    # Check 1: a clean PASS verdict round-trips and parses to PASS.
    total += 1
    clean_json = _json.dumps({
        "verdict": "PASS",
        "flagged_beats": [],
        "reason": "turn lands, ending pays off the cut-line setup",
        "dead_ending": False,
        "broken_turn": False,
        "flat_contrast": False,
        "unclear_grounding": False,
        "chopped_dialogue": False,
        "pacing_failure": False,
    })

    def _clean_fn(messages, *, temperature, max_new_tokens, stop=None):
        return clean_json

    verdict_clean = run_story_qa(
        ledger, _clean_fn, critic_model_id="test-technical-model",
    )
    if (
        verdict_clean.verdict == "PASS"
        and verdict_clean.flagged_beats == []
        and verdict_clean.dead_ending is False
    ):
        passed += 1
    else:
        print(f"  FAIL check 1 (clean PASS): {verdict_clean!r}")

    # Check 2: a MICRO_REPAIR_NEEDED verdict with flagged beats round-trips,
    # and the dialogue-beat extraction excluded the music/sfx rows (only 5
    # character beats -> highest valid index is 4).
    total += 1
    repair_json = _json.dumps({
        "verdict": "MICRO_REPAIR_NEEDED",
        "flagged_beats": [2, 4],
        "reason": "two beats read flat and chopped",
        "dead_ending": False,
        "broken_turn": False,
        "flat_contrast": True,
        "unclear_grounding": False,
        "chopped_dialogue": True,
        "pacing_failure": False,
    })

    def _repair_fn(messages, *, temperature, max_new_tokens, stop=None):
        return repair_json

    verdict_repair = run_story_qa(
        ledger, _repair_fn, critic_model_id="test-technical-model",
    )
    if (
        verdict_repair.verdict == "MICRO_REPAIR_NEEDED"
        and verdict_repair.flagged_beats == [2, 4]
        and verdict_repair.chopped_dialogue is True
        and len(_dialogue_beats(ledger)) == 5
    ):
        passed += 1
    else:
        print(f"  FAIL check 2 (MICRO_REPAIR_NEEDED + beat scope): {verdict_repair!r}")

    # Check 3: a structural REJECT verdict round-trips (dead ending), and
    # flagged_beats stays empty (a structural defect names no fix beats).
    total += 1
    reject_json = _json.dumps({
        "verdict": "REJECT",
        "flagged_beats": [],
        "reason": "episode stops with no payoff for the cut line",
        "dead_ending": True,
        "broken_turn": False,
        "flat_contrast": False,
        "unclear_grounding": False,
        "chopped_dialogue": False,
        "pacing_failure": False,
    })

    def _reject_fn(messages, *, temperature, max_new_tokens, stop=None):
        return reject_json

    verdict_reject = run_story_qa(
        ledger, _reject_fn, critic_model_id="test-technical-model",
    )
    if (
        verdict_reject.verdict == "REJECT"
        and verdict_reject.dead_ending is True
        and verdict_reject.flagged_beats == []
    ):
        passed += 1
    else:
        print(f"  FAIL check 3 (structural REJECT): {verdict_reject!r}")

    # Check 4: the prompt builder numbers ONLY the 5 dialogue beats and
    # never raises; it is read-only (the ledger line count is unchanged).
    total += 1
    before_line_count = len(ledger["lines"])
    prompt_text = _build_story_qa_input(ledger)
    if (
        "[beat 0 |" in prompt_text
        and "[beat 4 |" in prompt_text
        and "[beat 5 |" not in prompt_text
        and len(ledger["lines"]) == before_line_count
    ):
        passed += 1
    else:
        print("  FAIL check 4 (prompt beat numbering / read-only)")

    # Check 5: an empty-of-dialogue ledger yields a valid verdict (the
    # post_validator is a no-op with zero beats), proving the beat-free
    # path does not crash.
    total += 1
    empty_ledger = {"meta": {"title": "Silent", "style": "ambient"},
                    "cast": [], "lines": []}
    empty_json = _json.dumps({
        "verdict": "PASS",
        "flagged_beats": [],
        "reason": "no dialogue to judge",
        "dead_ending": False,
        "broken_turn": False,
        "flat_contrast": False,
        "unclear_grounding": False,
        "chopped_dialogue": False,
        "pacing_failure": False,
    })

    def _empty_fn(messages, *, temperature, max_new_tokens, stop=None):
        return empty_json

    verdict_empty = run_story_qa(
        empty_ledger, _empty_fn, critic_model_id="test-technical-model",
    )
    if verdict_empty.verdict == "PASS" and len(_dialogue_beats(empty_ledger)) == 0:
        passed += 1
    else:
        print(f"  FAIL check 5 (empty-dialogue ledger): {verdict_empty!r}")

    # Check 6: a raising slot fn maps to the fail-OPEN verdict
    # (verdict="PASS") and NEVER propagates the exception (Prime Directive
    # 1 -- a QA crash must not abort a good episode).
    total += 1

    def _raising_fn(messages, *, temperature, max_new_tokens, stop=None):
        raise RuntimeError("simulated VRAM / loader failure inside the slot fn")

    try:
        verdict_raise = run_story_qa(
            ledger, _raising_fn, critic_model_id="test-technical-model",
        )
        if (
            verdict_raise.verdict == "PASS"
            and verdict_raise.flagged_beats == []
        ):
            passed += 1
        else:
            print(f"  FAIL check 6 (fail-open verdict): {verdict_raise!r}")
    except Exception as exc:  # noqa: BLE001 -- the whole point is it must NOT raise
        print(f"  FAIL check 6 (slot fn exception propagated): {exc!r}")

    # Check 7: an out-of-range flagged beat forces the post_validator to
    # reject every attempt, the ladder exhausts, and the result is the
    # fail-open verdict -- proving the range guard is wired and the
    # exhaustion path maps to a (safe) PASS.
    total += 1
    out_of_range_json = _json.dumps({
        "verdict": "MICRO_REPAIR_NEEDED",
        "flagged_beats": [99],
        "reason": "fabricated index",
        "dead_ending": False,
        "broken_turn": False,
        "flat_contrast": False,
        "unclear_grounding": False,
        "chopped_dialogue": True,
        "pacing_failure": False,
    })

    def _out_of_range_fn(messages, *, temperature, max_new_tokens, stop=None):
        return out_of_range_json

    verdict_oor = run_story_qa(
        ledger, _out_of_range_fn, critic_model_id="test-technical-model",
    )
    if verdict_oor.verdict == "PASS":
        passed += 1
    else:
        print(f"  FAIL check 7 (out-of-range flagged-beat guard): {verdict_oor!r}")

    print(f"SELF-TEST PASS: {passed}/{total}")
    return passed if passed == total else -1


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    import sys

    logging.basicConfig(level=logging.WARNING)
    result = _self_test()
    sys.exit(0 if result >= 0 else 1)
