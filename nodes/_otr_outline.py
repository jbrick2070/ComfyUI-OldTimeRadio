"""nodes/_otr_outline.py

Grammar-validated outline generation for the v2.0 LedgerScriptWriter path.

Scope: science-fiction audio drama outlines grounded in real science
stories. The user supplies the science seed and a free-form style
descriptor; the LLM picks whatever dialogue register fits. NO period
anchoring -- no 1940s coaxing, no era constraints. The local model uses
its own trained distribution for dialogue style.

Pydantic schema for Beat[] + JSON-mode prompting + parse-or-reroll-or-repair
loop. NOT using lm-format-enforcer (compat unverified against transformers
5.x and the lib has been quiet for ~9 months); we get the same reliability
via deterministic post-hoc validation with a 3-attempt retry budget where
the third attempt is a repair call.

Status: Phase 1 of v2.0 sprint. Does NOT touch the in-flight legacy path.
Caller (eventual OTR_LedgerScriptWriter) is responsible for loading the
model via story_orchestrator._load_llm and passing the handle in. This
module does not load models.

Public surface:
    Beat                  -- pydantic model: one outline beat
    Outline               -- pydantic model: full episode outline
    OutlineRequest        -- frozen dataclass: input parameters
    OutlineFailedError    -- raised after 3 failed attempts
    generate_outline(...) -- main entrypoint
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

log = logging.getLogger("OTR")


__all__ = [
    "Beat",
    "Outline",
    "OutlineRequest",
    "OutlineFailedError",
    "generate_outline",
]


# ---------------------------------------------------------------------------
# SpeakerRole literal -- mirror of _otr_speaker_role.VALID_SPEAKER_ROLES
# ---------------------------------------------------------------------------

# Mirror of _otr_speaker_role.VALID_SPEAKER_ROLES. Duplicated to keep
# this module's imports stdlib+pydantic only at module load. The
# _check_speaker_role_alignment() function below verifies equality
# on first use of generate_outline and logs if the constants drift.
SpeakerRole = Literal[
    "character",
    "announcer",
    "music_open",
    "music_close",
    "music_inter",
    "sfx",
]


# ---------------------------------------------------------------------------
# Beat schema
# ---------------------------------------------------------------------------


class Beat(BaseModel):
    """One beat of the outline. Lines are generated 1:1 from beats."""

    beat_id: str = Field(
        ...,
        pattern=r"^b\d{3}$",
        description="Stable ID, format 'b001', 'b002', monotonic per outline",
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Character name in ALL CAPS, or 'NARRATOR' for music/sfx beats",
    )
    speaker_role: SpeakerRole = Field(
        ...,
        description="Routing role; see _otr_speaker_role for HuMo vs LTX-radio dispatch",
    )
    intent: str = Field(
        ...,
        min_length=4,
        max_length=200,
        description="What this beat accomplishes narratively, one sentence",
    )
    target_words: int = Field(
        ...,
        ge=3,
        le=80,
        description="Approximate word count for the dialogue line",
    )
    mood: str = Field(
        ...,
        min_length=2,
        max_length=40,
        description="Tone descriptor, e.g. 'tense', 'wry', 'foreboding'",
    )
    sfx_cue: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Optional [SFX:] hint for the surrounding line",
    )

    @field_validator("speaker")
    @classmethod
    def _speaker_uppercase(cls, v: str) -> str:
        return v.strip().upper()


# ---------------------------------------------------------------------------
# Outline schema
# ---------------------------------------------------------------------------


class Outline(BaseModel):
    """Full episode outline. The Outline IS the macro-plan; line composer
    consumes Beat-by-Beat and writes the ledger row by row.
    """

    title: str = Field(..., min_length=3, max_length=80)
    premise: str = Field(..., min_length=10, max_length=400)
    setting: str = Field(..., min_length=4, max_length=120)
    time_of_day: str = Field(..., min_length=3, max_length=40)
    cast: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Declared cast in canonical ALL CAPS form",
    )
    beats: list[Beat] = Field(..., min_length=4, max_length=24)

    @field_validator("cast")
    @classmethod
    def _cast_uppercase(cls, v: list[str]) -> list[str]:
        return [c.strip().upper() for c in v]

    @model_validator(mode="after")
    def _beats_speakers_consistent_with_cast(self) -> "Outline":
        cast_set = set(self.cast)
        for b in self.beats:
            if b.speaker_role == "character" and b.speaker not in cast_set:
                raise ValueError(
                    f"Beat {b.beat_id} has speaker {b.speaker!r} "
                    f"not in declared cast {sorted(cast_set)}"
                )
        ids = [b.beat_id for b in self.beats]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate beat_ids in outline: {ids}")
        return self


# ---------------------------------------------------------------------------
# OutlineRequest -- frozen input parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlineRequest:
    """Input parameters for generate_outline. Frozen so call sites
    can't accidentally mutate after construction.

    Cast contract (2026-05-10): the cast is no longer produced by the
    outline LLM. The writer locks the cast FIRST via
    nodes/_otr_casting.lock_cast() and passes the character names
    into this request via `character_cast`. The outline LLM is told
    those names are the cast it MUST use; a post-validation guard
    rejects any outline that drifts.
    """

    news_seed: str           # The real science story / factual seed.
                             # Back-compat field: callers who have no
                             # news_interpreter brief (e.g. early-stage
                             # tests, or the writer's fallback path
                             # when build_news_briefs raised) pass the
                             # raw seed here. When script_brief is
                             # non-empty it takes precedence in the
                             # prompt.
    style: str               # User-selected style, e.g. "psychological slow-burn",
                             # "pulp adventure", "hard sci-fi procedural", "noir thriller".
                             # Field renamed from style_hint 2026-05-10 — Jeffrey:
                             # "no 'hint', it's just style". User-visible widget name
                             # is 'style', so the dataclass field matches.
    target_words: int        # Canonical length unit (validated below). Words are
                             # the single source of truth for story planning;
                             # there is no seconds field — see Jeffrey 2026-05-10.
    character_cast: tuple[str, ...]
                             # ALL-CAPS character names from the LOCKED cast.
                             # Excludes ANNOUNCER (the writer hardcodes
                             # speaker="ANNOUNCER" on announcer-role beats so
                             # the LLM never needs to handle ANNOUNCER itself).
                             # 1-6 names. Validated below. NO default --
                             # callers MUST supply this. (Removing the default
                             # was a round-robin 2026-05-10 nit: an empty-tuple
                             # default would crash __post_init__ immediately,
                             # which is a worse failure mode than a clear
                             # TypeError from the dataclass constructor.)
    script_brief: str = ""
                             # OPTIONAL. news_interpreter's purpose-specific
                             # distillation of the article for script planning
                             # (premise arc, central tension, beat hooks).
                             # When non-empty, the prompt routes through the
                             # "Story brief" branch with a "develops this
                             # brief" closing verb -- because the brief is a
                             # distilled story plan, not raw factual material.
                             # When empty, the prompt falls back to news_seed
                             # under the "Science story (the factual seed)"
                             # label with the original "extrapolates from the
                             # science story" verb. Commit 3 (news_interpreter
                             # sprint, ADR docs/news_interpreter_adr.md);
                             # branch added in the post-sprint prompt
                             # tightening pass (2026-05-10).
    key_terms: tuple[str, ...] = ()
                             # OPTIONAL. news_interpreter's verbatim
                             # journalistic terms (people, places, technology)
                             # the dialogue MUST surface. Injected into the
                             # prompt as a "Required terms" line when non-
                             # empty so the outline can plan beats that
                             # naturally land them.

    def __post_init__(self) -> None:
        n = len(self.character_cast)
        if not (1 <= n <= 6):
            raise ValueError(
                f"character_cast must have 1-6 names, got {n}: "
                f"{self.character_cast!r}"
            )
        if self.target_words < 5:
            raise ValueError(
                f"target_words must be >= 5, got {self.target_words}"
            )
        for name in self.character_cast:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"character_cast names must be non-empty strings, "
                    f"got {name!r}"
                )
            if name != name.upper():
                raise ValueError(
                    f"character_cast names must be ALL CAPS, got {name!r}"
                )

    @property
    def cast_size(self) -> int:
        """Back-compat accessor. Reads len(character_cast)."""
        return len(self.character_cast)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a story editor. You produce JSON outlines for short science-fiction
audio dramas grounded in real science stories.

Your job is to plan the episode, not write the dialogue. Each beat names the
speaker, what they accomplish narratively, target word count, and mood. The
dialogue itself is generated by a separate process and will use whatever
register fits the story and style.

OUTPUT FORMAT
- Return exactly one JSON object.
- No prose before or after the JSON.
- No markdown code fences.
- The JSON must match this schema:

{
  "title":       string (3-80 chars),
  "premise":     string (10-400 chars),
  "setting":     string (4-120 chars),
  "time_of_day": string (3-40 chars),
  "cast":        array of 1-6 ALL-CAPS character names,
  "beats":       array of 4-24 beat objects, where each beat is:
                 {
                   "beat_id":      "b001", "b002", ... (monotonic),
                   "speaker":      ALL-CAPS name from cast, or "NARRATOR"
                                   for music/sfx beats,
                   "speaker_role": one of: "character", "announcer",
                                   "music_open", "music_close",
                                   "music_inter", "sfx",
                   "intent":       one sentence describing what the beat
                                   accomplishes narratively (4-200 chars),
                   "target_words": integer 3-80,
                   "mood":         tone descriptor (2-40 chars),
                   "sfx_cue":      optional string (max 80 chars), or null
                 }

CONSTRAINTS
- Every beat with speaker_role "character" MUST have a speaker that
  appears in the cast array.
- The first beat is typically speaker_role "music_open" or "announcer".
- The last beat is typically speaker_role "music_close" or "announcer".
- Beats should follow a clear arc: setup, complication, resolution.
- The premise must be grounded in the science story provided in the
  user prompt -- extrapolate dramatically from it, do not contradict it.
- Do not include the dialogue text in the outline. Only the intent.
"""


def _build_user_prompt(req: OutlineRequest) -> str:
    cast_line = ", ".join(req.character_cast)
    # news_interpreter brief takes precedence over raw news_seed.
    # When the writer has a script_brief from build_news_briefs, the
    # prompt labels the source line as a brief (it already contains
    # the distilled premise arc + central tension + beat hooks) and
    # the closing verb says DEVELOPS the brief, not EXTRAPOLATES from
    # raw material -- the dramatic extrapolation is already done.
    # When the writer is on the graceful-degrade path (brief LLM call
    # failed), the original "Science story (the factual seed)" label
    # + "extrapolates" verb still apply to the raw RSS payload.
    brief = req.script_brief.strip()
    if brief:
        source_line = f"Story brief: {brief}"
        develop_verb = "develops this brief"
    else:
        source_line = f"Science story (the factual seed): {req.news_seed}"
        develop_verb = "extrapolates from the science story"
    parts = [
        "Plan a science-fiction audio drama outline.",
        "",
        source_line,
    ]
    if req.key_terms:
        terms_line = ", ".join(req.key_terms)
        # The outline LLM writes intent + mood, not dialogue lines
        # (the line composer does that). Right plane to address: the
        # beats it plans must be ones that NATURALLY surface these
        # terms when the line composer renders them. Post-assembly
        # key_terms audit (commit 4) is what enforces presence in
        # the finished dialogue.
        parts.append(
            f"Required terms (plan beats that surface these in "
            f"dialogue): {terms_line}"
        )
    parts.extend([
        f"Style: {req.style}",
        f"Cast (already chosen -- use exactly these names in "
        f"character-role beats): {cast_line}",
        f"Target total dialogue length: ~{req.target_words} words "
        f"(sum of per-beat target_words should land near this number).",
        "",
    ])
    head = "\n".join(parts)
    return (
        f"{head}\n"
        f"Build a dramatic outline that {develop_verb} in the chosen "
        f"style. Echo the cast list verbatim in the JSON \"cast\" "
        f"field. Return only the JSON outline."
    )


_REPAIR_PROMPT_TEMPLATE = """\
Your previous response did not validate against the required JSON schema.

YOUR PREVIOUS RESPONSE:
{prev_response}

VALIDATION ERROR:
{validation_error}

Return ONLY corrected JSON that matches the schema. Do not explain. Do not add prose. Do not wrap in markdown fences. Output the corrected JSON object and nothing else.
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_block(raw: str) -> str:
    """Try three strategies in order:
       1. Strip ```json ... ``` or ``` ... ``` markdown fences if present.
       2. Slice from first '{' to last '}' (handles preambles like
          "Here's the JSON: { ... }").
       3. Return raw stripped (let json.loads raise the error).
    Always returns a string; never raises.
    """
    if not raw:
        return ""
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]
    return s


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class OutlineFailedError(RuntimeError):
    """Raised after generate_outline exhausts all retry attempts.

    Attributes:
        attempts: list of (raw_response, error_message) tuples per attempt
        request:  the OutlineRequest that was being processed
    """

    def __init__(
        self,
        attempts: list[tuple[str, str]],
        request: OutlineRequest,
    ) -> None:
        self.attempts = attempts
        self.request = request
        last_err = attempts[-1][1] if attempts else "no attempts"
        super().__init__(
            f"Outline generation failed after {len(attempts)} attempts. "
            f"Last error: {last_err}"
        )


# ---------------------------------------------------------------------------
# SpeakerRole drift check
# ---------------------------------------------------------------------------

_SPEAKER_ROLE_CHECKED = False


def _check_speaker_role_alignment() -> None:
    """Verify SpeakerRole literal matches _otr_speaker_role's canonical
    list. Lazy import -- only runs once per process, only when
    generate_outline is actually called.
    """
    global _SPEAKER_ROLE_CHECKED
    if _SPEAKER_ROLE_CHECKED:
        return
    _SPEAKER_ROLE_CHECKED = True
    try:
        from . import _otr_speaker_role as _srm
    except ImportError:
        return
    expected = set(_srm.VALID_SPEAKER_ROLES)
    actual = set(SpeakerRole.__args__)  # type: ignore[attr-defined]
    if expected != actual:
        log.warning(
            "[OTR_Outline] SpeakerRole drift: _otr_outline=%s, "
            "_otr_speaker_role=%s. Update _otr_outline.SpeakerRole "
            "to match the canonical list.",
            sorted(actual), sorted(expected),
        )


# ---------------------------------------------------------------------------
# generate_outline -- main entrypoint
# ---------------------------------------------------------------------------


def generate_outline(
    generate_fn,             # (messages, *, temperature, max_new_tokens) -> str
    req: OutlineRequest,
    *,
    max_attempts: int = 3,
    base_temperature: float = 0.7,
    max_new_tokens: int = 1500,
) -> Outline:
    """Generate a validated Outline. Reroll-then-repair on validation failure.

    Retry strategy:
      Attempt 1: fresh generation, temperature = base_temperature (0.7).
      Attempt 2: fresh generation, temperature = base_temperature + 0.1 (0.8).
      Attempt 3: REPAIR call, temperature 0.3, prompt includes the LAST raw
                 response and the exact ValidationError message.

    Caller adapter (lives in OTR_LedgerScriptWriter, NOT this module):

        def _make_generate_fn(llm_cache_entry):
            model = llm_cache_entry["model"]
            tokenizer = llm_cache_entry["tokenizer"]
            def generate_fn(messages, *, temperature, max_new_tokens):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.92,
                    max_new_tokens=max_new_tokens,
                )
                return tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            return generate_fn

    Raises:
        OutlineFailedError: if all attempts fail validation.
        ValueError: if max_attempts < 1 or generate_fn is not callable.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if not callable(generate_fn):
        raise ValueError("generate_fn must be callable")

    _check_speaker_role_alignment()

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(req)
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    attempts: list[tuple[str, str]] = []

    for attempt_idx in range(max_attempts):
        is_repair = (attempt_idx == max_attempts - 1) and attempt_idx >= 2

        if is_repair and attempts:
            prev_raw, prev_err = attempts[-1]
            repair_user = _REPAIR_PROMPT_TEMPLATE.format(
                prev_response=prev_raw,
                validation_error=prev_err,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": repair_user},
            ]
            temp = 0.3
            log.info(
                "[OTR_Outline] attempt %d/%d: repair call (temp=%.2f)",
                attempt_idx + 1, max_attempts, temp,
            )
        else:
            messages = base_messages
            temp = base_temperature + (0.1 * attempt_idx)
            log.info(
                "[OTR_Outline] attempt %d/%d: fresh generation (temp=%.2f)",
                attempt_idx + 1, max_attempts, temp,
            )

        try:
            raw = generate_fn(
                messages,
                temperature=temp,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = f"generate_fn raised: {type(exc).__name__}: {exc}"
            log.warning("[OTR_Outline] %s", err_msg)
            attempts.append(("", err_msg))
            continue

        last_raw = raw or ""
        json_str = _extract_json_block(last_raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            err_msg = f"json.JSONDecodeError: {exc}"
            log.warning("[OTR_Outline] attempt %d failed: %s", attempt_idx + 1, err_msg)
            attempts.append((last_raw, err_msg))
            continue

        try:
            outline = Outline.model_validate(data)
        except ValidationError as exc:
            err_msg = f"ValidationError: {exc}"
            log.warning("[OTR_Outline] attempt %d failed: %s", attempt_idx + 1, err_msg)
            attempts.append((last_raw, err_msg))
            continue

        # Post-pydantic cast-contract check: the LLM's outline.cast
        # MUST match the locked character_cast we passed in.
        # Exact-equality comparison: order matters AND duplicates
        # matter (a set-equality check would let
        # outline.cast=["ALICE","ALICE","BOB"] pass when locked was
        # ("ALICE","BOB")).
        #
        # IMPORTANT type detail (round-robin 2026-05-10): outline.cast
        # is parsed as a list by pydantic; req.character_cast is a
        # tuple. `list != tuple` is True even when contents match,
        # so we must cast req.character_cast to list before comparing,
        # otherwise the check would FALSE-positive on every clean run
        # and trigger the reroll loop indefinitely.
        expected_cast = list(req.character_cast)
        if outline.cast != expected_cast:
            extra = set(outline.cast) - set(expected_cast)
            missing = set(expected_cast) - set(outline.cast)
            dups = [
                n for n in outline.cast
                if outline.cast.count(n) > 1
            ]
            err_msg = (
                "CastContractError: outline.cast drifted from locked "
                f"character_cast. extra (invented): {sorted(extra)!r}, "
                f"missing (dropped): {sorted(missing)!r}, "
                f"duplicates: {sorted(set(dups))!r}. "
                f"Expected exactly (in order): {expected_cast!r}, "
                f"got: {outline.cast!r}"
            )
            log.warning(
                "[OTR_Outline] attempt %d failed: %s",
                attempt_idx + 1, err_msg,
            )
            attempts.append((last_raw, err_msg))
            continue

        log.info(
            "[OTR_Outline] success on attempt %d/%d: %d beats, cast=%s",
            attempt_idx + 1, max_attempts,
            len(outline.beats), outline.cast,
        )
        return outline

    raise OutlineFailedError(attempts=attempts, request=req)


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_outline.py` or `python -m nodes._otr_outline`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_outline.py self-test ===")

    # Test 1: Beat schema rejects bad inputs.
    print("\n[Test 1] Beat schema validation")
    try:
        Beat(beat_id="bad", speaker="X", speaker_role="character",
             intent="test test", target_words=10, mood="ok")
        print("  FAIL: bad beat_id was accepted")
    except ValidationError:
        print("  PASS: bad beat_id rejected")
    try:
        Beat(beat_id="b001", speaker="X", speaker_role="character",
             intent="test test", target_words=0, mood="ok")
        print("  FAIL: target_words=0 was accepted")
    except ValidationError:
        print("  PASS: target_words=0 rejected")

    # Test 2: speaker uppercased.
    b = Beat(beat_id="b001", speaker="aegeus", speaker_role="character",
             intent="introduce stakes", target_words=12, mood="tense")
    assert b.speaker == "AEGEUS", f"expected AEGEUS, got {b.speaker}"
    print("\n[Test 2] speaker uppercase canonicalization: PASS")

    # Test 3: Outline rejects beat speaker not in cast.
    print("\n[Test 3] Outline cross-validates beat speakers vs cast")
    bad_data = {
        "title": "Test",
        "premise": "A test premise of sufficient length.",
        "setting": "A test set",
        "time_of_day": "midnight",
        "cast": ["AEGEUS", "MARCUS"],
        "beats": [
            {"beat_id": f"b00{i+1}", "speaker": "STRANGER", "speaker_role": "character",
             "intent": "speak out of turn", "target_words": 12, "mood": "tense"}
            for i in range(4)
        ],
    }
    try:
        Outline.model_validate(bad_data)
        print("  FAIL: orphan speaker was accepted")
    except ValidationError as e:
        print(f"  PASS: orphan speaker rejected ({type(e).__name__})")

    # Test 4: Music/SFX beats can have any speaker.
    print("\n[Test 4] Music/SFX beats bypass cast check")
    ok_data = {
        "title": "Test",
        "premise": "A test premise of sufficient length.",
        "setting": "A test set",
        "time_of_day": "midnight",
        "cast": ["AEGEUS"],
        "beats": [
            {"beat_id": "b001", "speaker": "INTRO", "speaker_role": "music_open",
             "intent": "open the show", "target_words": 5, "mood": "bold"},
            {"beat_id": "b002", "speaker": "AEGEUS", "speaker_role": "character",
             "intent": "set the scene", "target_words": 15, "mood": "wry"},
            {"beat_id": "b003", "speaker": "AEGEUS", "speaker_role": "character",
             "intent": "complication arrives", "target_words": 20, "mood": "tense"},
            {"beat_id": "b004", "speaker": "OUTRO", "speaker_role": "music_close",
             "intent": "close the show", "target_words": 5, "mood": "resolute"},
        ],
    }
    o = Outline.model_validate(ok_data)
    assert len(o.beats) == 4
    print("  PASS: music/sfx beats accepted with non-cast speakers")

    # Test 5: JSON extraction handles fences, preambles, raw.
    print("\n[Test 5] _extract_json_block strategies")
    cases = [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the JSON: {"a": 1} hope this helps', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ('not json at all', 'not json at all'),
        ('', ''),
    ]
    for raw, expected in cases:
        got = _extract_json_block(raw)
        marker = "PASS" if got == expected else "FAIL"
        print(f"  {marker}: {raw!r:50} -> {got!r}")

    # Test 6: Round-trip serialize/deserialize.
    print("\n[Test 6] Outline JSON round-trip")
    j = o.model_dump_json()
    o2 = Outline.model_validate_json(j)
    assert o2 == o
    print("  PASS: round-trip OK")

    # Test 7: Speaker-role alignment runs without raising.
    print("\n[Test 7] _check_speaker_role_alignment runs without raising")
    _check_speaker_role_alignment()
    print("  PASS")

    # Test 8: OutlineRequest validates inputs.
    print("\n[Test 8] OutlineRequest input validation")
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=tuple(f"NAME{i}" for i in range(10)),
            target_words=150,
        )
        print("  FAIL: character_cast=10 accepted")
    except ValueError:
        print("  PASS: character_cast of 10 names rejected (must be 1-6)")
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("alice",),  # not uppercase
            target_words=150,
        )
        print("  FAIL: lowercase character_cast accepted")
    except ValueError:
        print("  PASS: lowercase character_cast rejected")

    # Test 9: OutlineFailedError carries diagnostics.
    print("\n[Test 9] OutlineFailedError shape")
    err = OutlineFailedError(
        attempts=[("raw1", "err1"), ("raw2", "err2")],
        request=OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE", "BOB"),
            target_words=150,
        ),
    )
    assert len(err.attempts) == 2
    assert err.request.cast_size == 2
    assert err.request.character_cast == ("ALICE", "BOB")
    assert "2 attempts" in str(err)
    print("  PASS")

    # Test 10: cast-contract drift check rejects mismatched outlines.
    print("\n[Test 10] generate_outline rejects cast drift")
    drift_outline_json = json.dumps({
        "title": "Test", "premise": "A test premise about science.",
        "setting": "A lab", "time_of_day": "Morning",
        # request will lock ("ALICE", "BOB") -- LLM returned CAROL instead
        "cast": ["ALICE", "CAROL"],
        "beats": [
            {"beat_id": "b001", "speaker": "NARRATOR",
             "speaker_role": "music_open",
             "intent": "open", "target_words": 5, "mood": "bold"},
            {"beat_id": "b002", "speaker": "ALICE",
             "speaker_role": "character",
             "intent": "speak", "target_words": 10, "mood": "wry"},
            {"beat_id": "b003", "speaker": "CAROL",
             "speaker_role": "character",
             "intent": "speak", "target_words": 10, "mood": "wry"},
            {"beat_id": "b004", "speaker": "NARRATOR",
             "speaker_role": "music_close",
             "intent": "close", "target_words": 5, "mood": "resolute"},
        ],
    })

    def _drift_gen_fn(messages, *, temperature, max_new_tokens):
        return drift_outline_json

    try:
        generate_outline(
            _drift_gen_fn,
            OutlineRequest(
                news_seed="x", style="y",
                character_cast=("ALICE", "BOB"),
                target_words=150,
            ),
            max_attempts=2,
        )
        print("  FAIL: cast drift was silently accepted")
    except OutlineFailedError as exc:
        last_err = exc.attempts[-1][1]
        assert "CastContractError" in last_err, \
            f"expected CastContractError in error, got: {last_err!r}"
        print("  PASS: cast drift rejected with CastContractError")

    print("\n=== all self-tests passed ===")
