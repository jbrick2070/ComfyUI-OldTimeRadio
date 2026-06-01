"""nodes/_otr_creative_qa.py -- Stream C Creative QA critic (dormant).

The Creative QA critic is the one new always-on round in the 4-stage
story spine (Narrative Draft -> Radio Editor -> Creative QA -> [one
editor repair] -> Ledger Scrub -> TTS). It is a READ-ONLY taste gate
that also absorbs the leak / attribution / SFW checks: it reads the
JUST-COMPOSED ledger at beat granularity, walks a categorical rubric,
and returns a Pydantic-validated `CreativeQAVerdict`. The writer's
single repair loop will consume that verdict later (a `REPAIR_ONCE`
scopes the Radio Editor to the flagged beats); for now this module
only RETURNS the verdict -- it never edits, and nothing yet calls it.

Design, per `docs/otr-story-spine.md` Stream C (and PART 0 drift
corrections):

  * Categorical, never numeric (Sec 2 invariant 2). The verdict is
    yes/no flags + beat indices + short reasons + a `Literal` verdict.
    There are NO 1-5 axis scores -- absolute numeric calibration is the
    capability that varies most between models, so the critic emits only
    observations the ladder can guarantee from any model that "follows
    instructions and emits JSON."
  * The ledger is READ-ONLY to this pass (Sec 2 invariant 3). The
    ledger feeds the FLUX -> HuMo -> LTX -> Bark cascade; this module
    reads beats and returns a verdict, and never mutates a ledger key.
  * In-process, not a node (D4). This is a helper called inside the
    writer's `run()`, mirroring `run_story_brief_reflection`
    (`_otr_story_brief.py:861`), NOT a new node and NOT a broadcast
    output. It exposes no `INPUT_TYPES`, no node class, and no
    `model_id` widget (Prime Directive 6).
  * Technical slot (D6). The critic receives its model id as a plain
    `str` argument (`critic_model_id`) sourced from the writer's
    in-process `resolved["technical_model"]` -- already the "other
    slot" whenever the two writer widgets differ, which satisfies the
    not-self-judge rule for free. The single call site is tagged
    `# LLM slot: technical`.

The single LLM call routes through the shared `structured_call`
3-attempt retry ladder (base -> structural retry -> typed repair),
FAIL-CLOSED: an exhausted ladder raises `StructuredCallFailedError`,
and the slot fn (the LLM closure) may raise arbitrary loader / VRAM
exceptions the ladder does not catch. Both are caught here and mapped
to a single fail-closed verdict (`verdict="FAIL"`) so this pass never
raises into the pipeline (Prime Directive 1 -- audio output must never
break). The writer's one repair cycle is reserved for `REPAIR_ONCE`;
a hard `FAIL` does not spend it.

This module is DORMANT until the writer wires Stage 3 into `run()`
(mirror `_otr_constrained_generate.py:16-27`). Wiring it, moving the
post-script LLM unload after the critic (D8), and the editor repair
loop are later, separate work and are explicitly out of scope here.

PD3 (workflow JSON): N/A; this module adds no node surface, so the
canonical workflow JSON (`workflows/otr_scifi_16gb_full.json`, 29
nodes) needs no change (D5).

The critic is a structured rubric evaluator -- it emits typed
categorical verdicts and beat indices, no creative prose -- so its one
LLM call is tagged `# LLM slot: technical`.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
# LLM slot: technical
# Reason: the critic emits a JSON-schema-constrained categorical
# verdict (flags + beat indices), not creative prose; per project
# rule 6 every structured-JSON pass routes to the technical model slot,
# and per spine D6 the critic runs on the "other slot"
# (resolved["technical_model"]) so the writer never judges itself.
from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, Optional, Sequence

from pydantic import BaseModel, Field
from typing import Literal


log = logging.getLogger("OTR")


__all__ = [
    "CreativeQAVerdict",
    "run_creative_qa",
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
# Generation params for the single critic LLM call.
# ---------------------------------------------------------------------------

# The verdict is a small, flat JSON object: a dozen flags, a handful of
# short reasons, and two short index lists. It is nowhere near the size
# of a whole-script rubric report, so a modest token budget is generous.
# Matches the order of magnitude the structured-pass default targets.
_CRITIC_MAX_NEW_TOKENS: int = 700

# Base attempt runs warm enough that the model engages with taste
# (turn / earned ending / distinct voices), not so warm it drifts off
# the schema. The structural retry runs STRICTLY COOLER -- the 2B
# principle: lowering entropy during a JSON-schema repair, never raising
# it. `structured_call` asserts this invariant at entry.
_CRITIC_TEMPERATURE: float = 0.4
_CRITIC_RETRY_TEMPERATURE: float = 0.2

# Per-beat text shown to the critic is truncated at this many chars so a
# runaway line cannot blow the prompt budget. Mirrors the per-line cap
# `_otr_story_brief._PER_LINE_TEXT_CAP` uses for the reflection input.
_PER_BEAT_TEXT_CAP: int = 240


# ---------------------------------------------------------------------------
# Pydantic schema -- the 12-field categorical verdict (Stream C contract).
# Mirrors the BaseModel style of `_otr_story_critic.py`: typed Literal
# verdict, short reason strings, index lists with `Field(default_factory)`.
# CATEGORICAL / OBSERVATIONAL ONLY -- no numeric 1-5 axis scores anywhere
# (Sec 2 invariant 2).
# ---------------------------------------------------------------------------


class CreativeQAVerdict(BaseModel):
    """The read-only Creative QA verdict over a composed episode.

    Exactly the 12 fields the spine's Stream C schema specifies. Every
    field is categorical (a bool, a `Literal`, a short reason string, an
    Optional beat index, or a list of beat indices) -- NEVER a numeric
    quality score. Beat indices are positions into the DIALOGUE-beat
    list the critic is shown (beat 0 = first character line), not ledger
    `line_id`s: the spine names these `*_beat_index` / `*_indices`, and
    a positional index is what a model can return reliably without
    having to echo an opaque id back verbatim.

    Field roles:
      has_turn              -- does the story contain a real turning
                               point (a reversal / decision that changes
                               the situation), not just rising talk?
      turn_beat_index       -- which dialogue beat lands the turn, or
                               None if `has_turn` is False / unclear.
      ending_earned         -- does the ending pay off what was set up,
                               rather than arriving by fiat?
      ending_note           -- one short phrase on the ending (<= ~15
                               words); why it lands or what it lacks.
      grounded_in_premise   -- does the episode stay on its own premise /
                               news facts, inventing nothing it does not
                               imply?
      voices_distinct       -- do the characters sound like different
                               people, not one narrator split in two?
      weakest_beat_index    -- the single dialogue beat most worth a
                               repair, or None if nothing stands out.
      weakest_problem       -- one short phrase naming that beat's
                               problem (<= ~12 words).
      overlong_line_indices -- dialogue beats whose line runs longer
                               than one breath (a radio-pacing flag).
      cast_name_leak_indices-- dialogue beats whose spoken text leaks a
                               cast member's name (the F3 / BUG-295
                               attribution leak the scrub also guards).
      sfw_ok                -- is every line safe-for-work and
                               non-violent?
      verdict               -- the single deterministic gate value:
                               PASS / REPAIR_ONCE / FAIL.

    Verdict rule (pass-biased -- P4/P5 lesson: brittle gates kill good
    runs):
      PASS         -- clean, or only cosmetic issues.
      REPAIR_ONCE  -- recoverable: a missing turn, an unearned ending, a
                      weak beat, overlong lines, a leak, or an SFW slip,
                      all fixable by editing NAMED beats.
      FAIL         -- unrecoverable (e.g. wholly off-premise). Reserved
                      for the genuinely irredeemable.

    All criteria live in the prompt, so this pass assumes only that the
    bound model "follows instructions and emits JSON" -- which the
    `structured_call` ladder guarantees.
    """

    has_turn: bool
    turn_beat_index: Optional[int] = None
    ending_earned: bool
    ending_note: str = ""
    grounded_in_premise: bool
    voices_distinct: bool
    weakest_beat_index: Optional[int] = None
    weakest_problem: str = ""
    overlong_line_indices: List[int] = Field(default_factory=list)
    cast_name_leak_indices: List[int] = Field(default_factory=list)
    sfw_ok: bool
    verdict: Literal["PASS", "REPAIR_ONCE", "FAIL"]

    @classmethod
    def fail_closed(cls, note: str = "creative QA pass could not run") -> "CreativeQAVerdict":
        """Return the fail-closed verdict used when the critic cannot run.

        `verdict="FAIL"` with every flag set to its NOT-clean value, so a
        critic that could not execute is observable and never silently
        reads as a clean PASS. The writer's single repair cycle is NOT
        spent by a `FAIL` (only `REPAIR_ONCE` consumes it), so this
        fail-closed path leaves the pipeline's one repair budget intact
        for a real recoverable verdict elsewhere.

        Returned by `run_creative_qa` on ANY failure (an exhausted
        `structured_call` retry ladder or a raising slot fn), keeping the
        never-raises contract absolute (Prime Directive 1 -- audio must
        never break over a flavor-text pass).
        """
        return cls(
            has_turn=False,
            turn_beat_index=None,
            ending_earned=False,
            ending_note=note,
            grounded_in_premise=False,
            voices_distinct=False,
            weakest_beat_index=None,
            weakest_problem=note,
            overlong_line_indices=[],
            cast_name_leak_indices=[],
            sfw_ok=False,
            verdict="FAIL",
        )


# ---------------------------------------------------------------------------
# Prompt construction -- ledger reading + the rubric system prompt.
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


def _dialogue_beats(ledger: dict) -> List[dict]:
    """Return the character-dialogue lines, in ledger order.

    These ARE the "beats" the critic judges and the `*_beat_index` /
    `*_indices` fields position into: announcer / music / sfx / scene
    rows are locked structural content that carry no dramatic turn, so
    they are excluded -- showing them would only tempt the critic to flag
    a beat nothing downstream can act on. Mirrors the character-line
    scope `_otr_story_critic._critic_character_lines` uses
    (`speaker_role == "character"`).

    Read keys, per the live ledger shape (see
    `_otr_story_brief._build_reflection_input`,
    `_otr_story_brief.py:511`): `ledger["lines"]` is the per-line list;
    each row carries `speaker_role`, `char_id` (or `speaker`), and
    `text`.
    """
    lines: Sequence[dict] = ledger.get("lines") or []
    return [
        ln for ln in lines
        if isinstance(ln, dict)
        and (ln.get("speaker_role") or "").lower() == "character"
    ]


def _build_creative_qa_input(led: Any) -> str:
    """Build the critic's user prompt from the ledger.

    Emits a compact, numbered roster of DIALOGUE beats -- one row per
    character line, prefixed `[beat N | char_id]` -- plus a short premise
    header (title + style slug) so the critic can judge "grounded in
    premise." The beat NUMBER is the index every `*_beat_index` /
    `*_indices` field must reference, so it is shown explicitly and the
    prompt tells the critic to use it.

    READ-ONLY: this reads `led.data` (or a raw dict) and returns a
    string. It never mutates the ledger.
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
        "Judge ONLY the numbered dialogue beats above. Every beat index "
        "you return (turn_beat_index, weakest_beat_index, "
        "overlong_line_indices, cast_name_leak_indices) is the integer "
        "shown in [beat N]; use only indices in range 0.."
        f"{max(len(beats) - 1, 0)} and invent none. Then emit the single "
        "JSON CreativeQAVerdict object."
    )
    return "\n".join(parts)


_CRITIC_SYSTEM_PROMPT = """\
You are a Creative QA critic for an audio drama. You are shown the
episode's character-dialogue beats, numbered from 0. Your job is a
READ-ONLY taste check plus a few cleanliness flags. You do not rewrite
anything; you only observe and return a verdict.

Report only CATEGORICAL observations -- yes/no flags, beat indices, and
short reasons. Do NOT score anything on a numeric scale. Use the integer
beat numbers exactly as shown in [beat N].

Answer each field:

has_turn -- Is there a real turning point: a reversal, decision, or
discovery that changes the situation, not just more rising talk? If yes,
set turn_beat_index to the beat where it lands; if no, set has_turn false
and turn_beat_index null.

ending_earned -- Does the final beat pay off what the episode set up,
rather than arriving by fiat? Give a SHORT ending_note (about 15 words or
fewer) on why it lands or what it lacks.

grounded_in_premise -- Does the episode stay on its own premise and the
facts it was given, inventing no people, places, or objects the premise
does not imply?

voices_distinct -- Do the characters sound like different people, with
different register and attitude, rather than one narrator split in two?

weakest_beat_index -- The SINGLE dialogue beat most worth a repair, or
null if none stands out. Name its problem in weakest_problem (about 12
words or fewer).

overlong_line_indices -- The beats whose spoken line runs longer than one
breath (a radio-pacing concern). Empty if every line is spoken-length.

cast_name_leak_indices -- The beats whose spoken text says a character's
own name aloud where it does not belong. Empty if none.

sfw_ok -- Is every line safe-for-work and non-violent?

verdict -- Choose ONE, pass-biased:
  PASS         -- clean, or only cosmetic issues.
  REPAIR_ONCE  -- recoverable: a missing turn, an unearned ending, a weak
                  beat, overlong lines, a leak, or an SFW slip, all fixable
                  by editing the named beats.
  FAIL         -- unrecoverable, e.g. wholly off-premise. Reserve this for
                  the genuinely irredeemable; prefer REPAIR_ONCE when an
                  edit could save it.

Return EXACTLY one JSON object matching this schema:
{
  "has_turn": true | false,
  "turn_beat_index": int | null,
  "ending_earned": true | false,
  "ending_note": "...",
  "grounded_in_premise": true | false,
  "voices_distinct": true | false,
  "weakest_beat_index": int | null,
  "weakest_problem": "...",
  "overlong_line_indices": [int, ...],
  "cast_name_leak_indices": [int, ...],
  "sfw_ok": true | false,
  "verdict": "PASS" | "REPAIR_ONCE" | "FAIL"
}

Every list may be empty. No prose before or after the JSON. No markdown
fences.
"""


# ---------------------------------------------------------------------------
# post_validator -- light beat-index range guard.
# ---------------------------------------------------------------------------


def _make_creative_qa_post_validator(beat_count: int):
    """Build the optional content-check passed to `structured_call`.

    Deliberately LENIENT (the same posture as
    `_otr_story_critic._make_critic_post_validator`): it rejects a
    verdict ONLY when a returned beat index is out of range for the beats
    the critic was shown -- a fabricated index would later point the
    writer's repair loop at a beat that does not exist. It does NOT
    second-guess the critic's taste judgement, the verdict value, or how
    many beats were flagged: those are exactly what the critic is for.

    Returns a `post_validator(verdict) -> str | None` closure: an error
    string REJECTS (and advances the ladder), `None` ACCEPTS.
    """

    def post_validator(verdict: "CreativeQAVerdict") -> Optional[str]:
        # An empty beat list means there is nothing to range-check
        # against; accept whatever validated rather than reject every
        # verdict on a beat-free ledger.
        if beat_count <= 0:
            return None

        out_of_range: List[int] = []

        def _check(index: Optional[int]) -> None:
            if index is not None and not (0 <= index < beat_count):
                out_of_range.append(index)

        _check(verdict.turn_beat_index)
        _check(verdict.weakest_beat_index)
        for index in verdict.overlong_line_indices:
            _check(index)
        for index in verdict.cast_name_leak_indices:
            _check(index)

        if out_of_range:
            preview = ", ".join(str(i) for i in sorted(set(out_of_range))[:8])
            return (
                f"verdict references {len(set(out_of_range))} beat "
                f"index(es) out of range 0..{beat_count - 1}: {preview}"
            )
        return None

    return post_validator


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_creative_qa(
    led: Any,
    critic_generate_fn: Callable[..., str],
    *,
    critic_model_id: str,
) -> CreativeQAVerdict:
    """Run the Stream C Creative QA critic and RETURN its verdict.

    Mirrors `run_story_brief_reflection` (`_otr_story_brief.py:861`):
    same post-script call pattern, slot plumbing, and fail-loud-then-
    fail-closed handling. READ-ONLY -- it reads the ledger and returns a
    verdict; it NEVER mutates a ledger key. The writer's single repair
    loop will consume the returned verdict later; this module is dormant,
    so for now it only returns it.

    Parameters:
      led
        The Ledger object (`.data`) OR a raw ledger dict -- duck-typed
        via `_ledger_data` so a test can pass a plain dict. Read keys
        (live ledger shape, see `_otr_story_brief.py:511`):
        `ledger["lines"]` (per-line records: `speaker_role`, `char_id` /
        `speaker`, `text`) and `ledger["meta"]` (`episode_title` /
        `title`, `style`).
      critic_generate_fn
        The LLM generate callable -- `critic_generate_fn(messages, *,
        temperature, max_new_tokens[, stop]) -> str`. In production this
        is the writer's in-process technical-slot closure; no `model_id`
        widget exists anywhere (Prime Directive 6).
      critic_model_id  (keyword-only)
        The technical-slot model id, sourced from the writer's in-process
        `resolved["technical_model"]` (D6 -- the "other slot," so the
        writer never judges itself). Carried for logging / attribution;
        the actual generation goes through `critic_generate_fn`.

    NEVER raises (Prime Directive 1 -- audio output must never break).
    Any failure -- an exhausted `structured_call` retry ladder or a
    raising slot fn -- is logged and `CreativeQAVerdict.fail_closed()` is
    returned (`verdict="FAIL"`), which does NOT consume the writer's one
    repair cycle.
    """
    ledger = _ledger_data(led)
    beat_count = len(_dialogue_beats(ledger))

    user_prompt = _build_creative_qa_input(led)
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    post_validator = _make_creative_qa_post_validator(beat_count)

    # The single critic call routes through the shared structured_call
    # 3-attempt ladder (base -> structural retry -> typed repair),
    # FAIL-CLOSED. An exhausted ladder raises StructuredCallFailedError;
    # the slot fn (the LLM closure) may raise arbitrary loader / VRAM
    # exceptions structured_call does NOT catch. BOTH are caught below --
    # every failure path returns CreativeQAVerdict.fail_closed(), keeping
    # the never-raises contract absolute (Prime Directive 1).
    #
    # The Creative QA critic emits a JSON-schema-constrained categorical
    # verdict (flags + beat indices), not creative prose, so it routes to
    # the technical model exactly as the reflection and story-critic passes
    # do. Per spine D6 the technical slot is the "other slot" relative to
    # the writer's creative passes, so routing the critic here means the
    # writer never judges its own narrative. The model id arrives as the
    # `critic_model_id` str (from the writer's resolved["technical_model"])
    # -- no new widget, no new model_id parameter (Prime Directive 6).
    #
    # LLM slot: technical -- structured-JSON categorical verdict pass.
    try:
        verdict = structured_call(
            prompt=messages,
            schema=CreativeQAVerdict,
            slot_fn=critic_generate_fn,
            base_temperature=_CRITIC_TEMPERATURE,
            structural_retry_temperature=_CRITIC_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=post_validator,
            max_new_tokens=_CRITIC_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_creative_qa",
        )
    except StructuredCallFailedError as exc:
        # Ladder exhausted. Attribute the failure to its class via the
        # last error the ladder captured, so a log reader can tell a
        # parse / schema / content failure apart at a glance.
        last = exc.last_error
        if isinstance(last, json.JSONDecodeError):
            reason = "json_parse_failed_after_repair"
        elif isinstance(last, PostValidationError):
            reason = "beat_index_validation_failed_after_repair"
        else:
            reason = "structured_call_failed"
        log.warning(
            "[OTR_CreativeQA] structured_call exhausted the retry ladder "
            "after %d attempt(s) (model_id=%s, last error: %s); returning "
            "fail-closed verdict (reason=%s)",
            exc.attempts, critic_model_id, exc.last_error, reason,
        )
        return CreativeQAVerdict.fail_closed(reason)
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        # structured_call does not catch slot-fn exceptions: a VRAM /
        # loader / framework failure inside the critic_generate_fn
        # closure lands here. A bare reraise would crash the whole script
        # generation over a taste-check failure -- audio is king, so this
        # pass fails closed instead.
        log.warning(
            "[OTR_CreativeQA] critic_generate_fn raised %s: %s "
            "(model_id=%s); returning fail-closed verdict",
            type(exc).__name__, exc, critic_model_id,
        )
        return CreativeQAVerdict.fail_closed("critic_generate_fn_exception")

    log.info(
        "[OTR_CreativeQA] verdict=%s (model_id=%s): has_turn=%s, "
        "ending_earned=%s, grounded=%s, voices_distinct=%s, sfw_ok=%s, "
        "weakest_beat=%s, %d overlong line(s), %d cast-name leak(s)",
        verdict.verdict, critic_model_id, verdict.has_turn,
        verdict.ending_earned, verdict.grounded_in_premise,
        verdict.voices_distinct, verdict.sfw_ok, verdict.weakest_beat_index,
        len(verdict.overlong_line_indices),
        len(verdict.cast_name_leak_indices),
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
    PASS, a REPAIR_ONCE with flagged beats, the empty-ledger path, the
    fail-closed mapping when the slot fn raises, and the out-of-range
    beat-index post_validator guard.
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
            {"line_id": "x00", "speaker_role": "sfx", "text": "[static]"},
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
        "has_turn": True,
        "turn_beat_index": 3,
        "ending_earned": True,
        "ending_note": "resolves the cut-line setup with a shared decision",
        "grounded_in_premise": True,
        "voices_distinct": True,
        "weakest_beat_index": None,
        "weakest_problem": "",
        "overlong_line_indices": [],
        "cast_name_leak_indices": [],
        "sfw_ok": True,
        "verdict": "PASS",
    })

    def _clean_fn(messages, *, temperature, max_new_tokens, stop=None):
        return clean_json

    verdict_clean = run_creative_qa(
        ledger, _clean_fn, critic_model_id="test-technical-model",
    )
    if (
        verdict_clean.verdict == "PASS"
        and verdict_clean.has_turn is True
        and verdict_clean.turn_beat_index == 3
        and verdict_clean.sfw_ok is True
        and verdict_clean.overlong_line_indices == []
    ):
        passed += 1
    else:
        print(f"  FAIL check 1 (clean PASS): {verdict_clean!r}")

    # Check 2: a REPAIR_ONCE verdict with flagged beats round-trips, and
    # the dialogue-beat extraction excluded the music/sfx rows (only 5
    # character beats -> highest valid index is 4).
    total += 1
    repair_json = _json.dumps({
        "has_turn": False,
        "turn_beat_index": None,
        "ending_earned": False,
        "ending_note": "ending arrives by fiat, no real payoff",
        "grounded_in_premise": True,
        "voices_distinct": True,
        "weakest_beat_index": 2,
        "weakest_problem": "states a fact, plays no moment",
        "overlong_line_indices": [4],
        "cast_name_leak_indices": [],
        "sfw_ok": True,
        "verdict": "REPAIR_ONCE",
    })

    def _repair_fn(messages, *, temperature, max_new_tokens, stop=None):
        return repair_json

    verdict_repair = run_creative_qa(
        ledger, _repair_fn, critic_model_id="test-technical-model",
    )
    if (
        verdict_repair.verdict == "REPAIR_ONCE"
        and verdict_repair.has_turn is False
        and verdict_repair.weakest_beat_index == 2
        and verdict_repair.overlong_line_indices == [4]
        and len(_dialogue_beats(ledger)) == 5
    ):
        passed += 1
    else:
        print(f"  FAIL check 2 (REPAIR_ONCE + beat scope): {verdict_repair!r}")

    # Check 3: the prompt builder numbers ONLY the 5 dialogue beats and
    # never raises; it is read-only (the ledger line count is unchanged).
    total += 1
    before_line_count = len(ledger["lines"])
    prompt_text = _build_creative_qa_input(ledger)
    if (
        "[beat 0 |" in prompt_text
        and "[beat 4 |" in prompt_text
        and "[beat 5 |" not in prompt_text
        and len(ledger["lines"]) == before_line_count
    ):
        passed += 1
    else:
        print("  FAIL check 3 (prompt beat numbering / read-only)")

    # Check 4: an empty-of-dialogue ledger yields a valid verdict (the
    # post_validator is a no-op with zero beats), proving the beat-free
    # path does not crash.
    total += 1
    empty_ledger = {"meta": {"title": "Silent", "style": "ambient"},
                    "cast": [], "lines": []}
    empty_json = _json.dumps({
        "has_turn": False,
        "turn_beat_index": None,
        "ending_earned": False,
        "ending_note": "no dialogue to judge",
        "grounded_in_premise": True,
        "voices_distinct": True,
        "weakest_beat_index": None,
        "weakest_problem": "",
        "overlong_line_indices": [],
        "cast_name_leak_indices": [],
        "sfw_ok": True,
        "verdict": "REPAIR_ONCE",
    })

    def _empty_fn(messages, *, temperature, max_new_tokens, stop=None):
        return empty_json

    verdict_empty = run_creative_qa(
        empty_ledger, _empty_fn, critic_model_id="test-technical-model",
    )
    if verdict_empty.verdict == "REPAIR_ONCE" and len(_dialogue_beats(empty_ledger)) == 0:
        passed += 1
    else:
        print(f"  FAIL check 4 (empty-dialogue ledger): {verdict_empty!r}")

    # Check 5: a raising slot fn maps to the fail-closed verdict
    # (verdict="FAIL") and NEVER propagates the exception (Prime
    # Directive 1 -- audio must never break).
    total += 1

    def _raising_fn(messages, *, temperature, max_new_tokens, stop=None):
        raise RuntimeError("simulated VRAM / loader failure inside the slot fn")

    try:
        verdict_raise = run_creative_qa(
            ledger, _raising_fn, critic_model_id="test-technical-model",
        )
        if (
            verdict_raise.verdict == "FAIL"
            and verdict_raise.sfw_ok is False
            and verdict_raise.has_turn is False
        ):
            passed += 1
        else:
            print(f"  FAIL check 5 (fail-closed verdict): {verdict_raise!r}")
    except Exception as exc:  # noqa: BLE001 -- the whole point is it must NOT raise
        print(f"  FAIL check 5 (slot fn exception propagated): {exc!r}")

    # Check 6: an out-of-range beat index forces the post_validator to
    # reject every attempt, the ladder exhausts, and the result is the
    # fail-closed verdict -- proving the range guard is wired and the
    # exhaustion path maps to FAIL.
    total += 1
    out_of_range_json = _json.dumps({
        "has_turn": True,
        "turn_beat_index": 99,
        "ending_earned": True,
        "ending_note": "ok",
        "grounded_in_premise": True,
        "voices_distinct": True,
        "weakest_beat_index": None,
        "weakest_problem": "",
        "overlong_line_indices": [],
        "cast_name_leak_indices": [],
        "sfw_ok": True,
        "verdict": "PASS",
    })

    def _out_of_range_fn(messages, *, temperature, max_new_tokens, stop=None):
        return out_of_range_json

    verdict_oor = run_creative_qa(
        ledger, _out_of_range_fn, critic_model_id="test-technical-model",
    )
    if verdict_oor.verdict == "FAIL":
        passed += 1
    else:
        print(f"  FAIL check 6 (out-of-range index guard): {verdict_oor!r}")

    print(f"SELF-TEST PASS: {passed}/{total}")
    return passed if passed == total else -1


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    import sys

    logging.basicConfig(level=logging.WARNING)
    result = _self_test()
    sys.exit(0 if result >= 0 else 1)
