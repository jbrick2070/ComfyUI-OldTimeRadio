"""My Story -- the lane that turns a person's own idea into tonight's episode.

WHAT MAKES THIS LANE ITS OWN. Every other bank starts from something the
machine found: a feed item, an archive post, a public-domain text, a spark
drawn from a deck. This one starts from what a person typed, and that changes
the shape of the work. The first pass is not a summariser, it is an INTERPRETER
-- it decides which of their words are requirements and which are asides, then
records what it had to assume. Selected acts bind; character count is guidance.

THE PASS GRAPH, and why it is shaped this way:

  P0 interpret  (technical) -- their fields -> what they actually asked for.
  P1 treatment  (creative)  -- the plan: title, cast, per-act shape, ending.
  P2 acts       (creative)  -- ONE CALL PER ACT, carrying the previous act's
                              tail and the cast not yet heard.
  P3 frame      (creative)  -- announcer open/close, coda, music cues.
  P4 voices     (python)    -- one larynx per character, deterministically.
  P5 assemble   (python)    -- the five ledger hierarchies and the receipts.

Act-at-a-time is the deliberate difference from the sibling content-owned lane,
which asks for the whole play in one markup artifact and retries the WHOLE play
when the markup breaks. Here a failed act retries only itself: an act is a
bounded thing to ask for and a bounded thing to lose.

THE PERSON'S WORDS ARE THE SOURCE. The model adapts their material into the
selected acts, with a flexible cast guided by their story. A gender they did not state is
never guessed from a name. Interpretation notes and detected source conflicts
remain in the receipt; requested and actual character counts stay separate.

WHAT THIS LANE DOES NOT DO. It runs no cameo roll: the cast belongs to the
person who described it, and a house cameo they did not ask for would overrule
them. It declares no `line_composer_system` seam, which routes the freeze to
`content_owned_readonly` -- accepted text and authorized cleanup are verified.
It never refuses a story for its length, its language or its taste.

UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pydantic import BaseModel, Field, field_validator

try:
    from . import _otr_canon as _OTRC
    from . import _otr_casting as _OTRCAST
    from . import _otr_story_input as _SI
    from . import _otr_word_delivery as _OTRWD
    from ._otr_structured_call import structured_call, PostValidationError
    from ._otr_generation_budget import ProviderCapacityMessages
    from ._otr_script_prep import clean_spoken_text
    from ._otr_json import parse_first_json_object
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover -- flat / standalone test import
    import _otr_canon as _OTRC  # type: ignore
    import _otr_casting as _OTRCAST  # type: ignore
    import _otr_story_input as _SI  # type: ignore
    import _otr_word_delivery as _OTRWD  # type: ignore
    from _otr_structured_call import structured_call, PostValidationError  # type: ignore
    from _otr_generation_budget import ProviderCapacityMessages  # type: ignore
    from _otr_script_prep import clean_spoken_text  # type: ignore
    from _otr_json import parse_first_json_object  # type: ignore
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore

try:
    from ..config import cast_pools as _POOLS
except ImportError:  # pragma: no cover -- flat / standalone test import
    from config import cast_pools as _POOLS  # type: ignore

log = logging.getLogger("OTR")

#: Ledger receipt version for `meta.my_story`.
MY_STORY_SCHEMA = "my_story_v1"

ANNOUNCER_NAME = "ANNOUNCER"
ANNOUNCER_CHAR_ID = "announcer"

#: Base / structural-retry temperatures. The retry rung is always LOWER: a
#: structural failure is answered with less entropy, not more.
_TEMP = {
    "interpret": (0.3, 0.2),
    "treatment": (0.85, 0.5),
    "act": (0.85, 0.5),
    "frame": (0.6, 0.4),
}

#: How much of the previous act to carry forward. Bounded on purpose -- the
#: whole prior act would grow the prompt with every act and eventually crowd
#: out the artifact itself.
_PRIOR_ACT_LINES = 4


class MyStoryError(RuntimeError):
    """A My Story pass failed terminally. Names the pass, never the sibling lane.

    Its own type rather than a borrowed one: a message reading
    "[scifi_news_pro] pass failed" on a listener's own story would send the
    next reader to the wrong module.
    """

    def __init__(self, pass_id: str, detail: str) -> None:
        super().__init__("[my_story] pass %r failed: %s" % (pass_id, detail))
        self.pass_id = pass_id
        self.detail = detail


class MyStoryCastError(MyStoryError):
    """The cast cannot be built or cannot be heard."""


# ---------------------------------------------------------------------------
# P0 -- interpretation
# ---------------------------------------------------------------------------

class Requirement(BaseModel):
    id: str = ""
    text: str = ""
    kind: str = "other"
    source_field: str = "idea"
    strength: str = "required"


class NamedCast(BaseModel):
    name: str = ""
    notes: str = ""
    stated_gender: str = ""
    speaking: bool = True
    required: bool = True

    @field_validator("stated_gender", mode="before")
    @classmethod
    def _norm_gender(cls, value):
        """Normalize stated vocabulary without inventing or erasing a gender."""
        if value is None:
            return ""
        try:
            from ._otr_roster_gender import canonical_bank_gender
        except ImportError:  # pragma: no cover -- flat load
            from _otr_roster_gender import canonical_bank_gender  # type: ignore
        canon = str(canonical_bank_gender(value) or "").strip().lower()
        return canon


class CastPlan(BaseModel):
    requested: int = 2
    planned: int = 2
    exclusive: bool = False
    reason: str = ""


class Conflict(BaseModel):
    requirement_id: str = ""
    why: str = ""
    resolution: str = ""


class StoryInterpretation(BaseModel):
    requirements: "list[Requirement]" = Field(default_factory=list)
    named_cast: "list[NamedCast]" = Field(default_factory=list)
    cast_plan: CastPlan = Field(default_factory=CastPlan)
    setting_brief: str = ""
    assumptions: "list[str]" = Field(default_factory=list)
    conflicts: "list[Conflict]" = Field(default_factory=list)

    def required_speakers(self) -> "list[str]":
        seen: "list[str]" = []
        for row in self.named_cast:
            if row.required and row.speaking:
                name = row.name.strip()
                if name and name not in seen:
                    seen.append(name)
        return seen

    def gender_by_name(self) -> "dict[str, str]":
        return {r.name.strip(): r.stated_gender
                for r in self.named_cast if r.stated_gender}


# ---------------------------------------------------------------------------
# P1 -- treatment
# ---------------------------------------------------------------------------

class CastMember(BaseModel):
    name: str = Field(min_length=1)
    role: str = ""
    character_description: str = ""
    gender: str = ""
    age_band: str = "n/a"
    # `speech_register`, not `register`: the bare name shadows a pydantic
    # BaseModel attribute and pydantic warns about it at class construction.
    # The seam asks for "register"; the alias keeps the prompt's word while
    # the field keeps a name that is safe to own.
    speech_register: str = Field(default="", alias="register")
    timbre: str = ""

    model_config = {"populate_by_name": True}

    @field_validator("gender", mode="before")
    @classmethod
    def _canonical_gender(cls, value):
        """Fix synonyms; missing/other values use the shared open voice pool."""
        try:
            from ._otr_roster_gender import canonical_bank_gender
        except ImportError:  # pragma: no cover -- flat load
            from _otr_roster_gender import canonical_bank_gender  # type: ignore
        return str(canonical_bank_gender(value) or "").strip().lower()

    @field_validator("age_band", mode="before")
    @classmethod
    def _norm_age(cls, value):
        text = str(value or "").strip().lower().replace("\\", "/")
        return "n/a" if text in {"", "na", "none", "null", "unknown", "-"} else text


class ActPlan(BaseModel):
    n: int = 1
    purpose: str = ""
    scene_setting: str = ""
    turns: "list[str]" = Field(default_factory=list)
    ending_state: str = ""


class StoryTreatment(BaseModel):
    title: str = ""
    logline: str = ""
    dramatic_question: str = ""
    setting: str = ""
    time_of_day: str = "night"
    cast: "list[CastMember]" = Field(min_length=1)
    acts: "list[ActPlan]" = Field(min_length=1)
    ending: str = ""

    def names(self) -> "list[str]":
        return [c.name.strip() for c in self.cast]


# ---------------------------------------------------------------------------
# P2 -- one act
# ---------------------------------------------------------------------------

class SpokenLine(BaseModel):
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ActScript(BaseModel):
    n: int = 1
    scene_setting: str = ""
    lines: "list[SpokenLine]" = Field(min_length=1)


# ---------------------------------------------------------------------------
# P3 -- the announcer's frame
# ---------------------------------------------------------------------------

class StoryFrame(BaseModel):
    announcer_intro: "list[str]" = Field(default_factory=list)
    announcer_outro: "list[str]" = Field(default_factory=list)
    coda: str = ""
    music_open: str = ""
    music_close: str = ""
    music_inter: "list[str]" = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tail handoff -- plain parts; the WRITER builds WriterTailContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MyStoryOutlineView:
    """The tail's outline duck-type: .premise + .title (+ .setting)."""

    premise: str
    title: str
    setting: str = ""


@dataclass
class MyStoryTailParts:
    outline_view: MyStoryOutlineView
    canon: Any
    final_title_override: str | None
    run_story_spine: bool = False
    refine_active: bool = False
    my_story_meta: "dict | None" = None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _seam(pack: Any, name: str) -> str:
    stages = getattr(pack, "prompt_stages", None) or {}
    text = str(stages.get(name) or "").strip()
    if not text:
        raise MyStoryError(name, "the story pack declares no %r seam" % name)
    return text


def _helper_ctx(slot_scheduler: Any, name: str):
    """Attribute slot calls to a named helper; inert under unit tests."""
    if slot_scheduler is None:
        class _Null:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False
        return _Null()
    return slot_scheduler.helper_context(name)


def _ledger_meta(led: Any) -> dict:
    """Read metadata from the current ledger, including transaction rollback."""
    return led.data.setdefault("meta", {})


def _require_ledger_save(led: Any, what: str) -> None:
    """Save, and refuse to continue if it did not land.

    ``Ledger.save()`` returns the path on success and None on failure without
    raising, so an unchecked call is a write that can silently not happen --
    and every downstream node reads this file from disk.
    """
    if led.save() is None:
        raise MyStoryError(
            "ledger_save",
            "the ledger did not persist after %s -- refusing to continue, "
            "because every downstream node reads it from disk" % what,
        )


def _norm_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _interstitial_count(act_count: int, include_act_breaks: bool) -> int:
    """Music belongs to boundaries between acts, never to the checkbox alone."""
    return max(0, act_count - 1) if include_act_breaks else 0


def _resolve_seed() -> int:
    """Episode seed. Honours OTR_EPISODE_SEED when pinned, else OS entropy."""
    try:
        from ._otr_shared import env as otr_env
    except ImportError:  # pragma: no cover -- flat load
        from _otr_shared import env as otr_env  # type: ignore
    raw = str(otr_env.get("OTR_EPISODE_SEED", "") or "").strip()
    if raw:
        try:
            return int(raw, 10)
        except ValueError:
            log.warning("[my_story] OTR_EPISODE_SEED=%r is not an integer; "
                        "drawing fresh entropy instead", raw)
    return random.SystemRandom().getrandbits(32)


def _call(pass_id: str, bundle: Any, *, attempt_receipts=None, **kwargs) -> Any:
    """Use the shared capacity contract and retain actual attempt evidence."""
    del bundle
    kwargs["prompt"] = ProviderCapacityMessages(kwargs["prompt"])
    kwargs["max_new_tokens"] = None

    def completed(number, raw, error):
        if attempt_receipts is not None:
            attempt_receipts.append({
                "pass_id": pass_id, "attempt": number, "raw_output": raw,
                "status": "accepted" if error is None else "failed",
                "error": None if error is None else str(error),
            })

    # LLM slot: per-sub-pass -- caller supplies the creative or technical slot.
    return structured_call(on_attempt_complete=completed, **kwargs)


def _full_artifact_repair(instruction: str):
    """Give existing post-validation repair the entire parsed draft to revise.

    The generic repair's 400-character echo cannot show the end of a treatment
    or an act. Keep this at the authoring seam; all other typed repairs remain
    shared, and the same post-validator still decides acceptance.
    """
    typed = make_dispatching_repair_factory()

    def repair(*, original_prompt, failed_output, error):
        if not isinstance(error, PostValidationError):
            return typed(original_prompt=original_prompt,
                         failed_output=failed_output, error=error)
        return [
            *[dict(message) for message in original_prompt],
            {"role": "assistant", "content": failed_output},
            {"role": "user", "content": (
                "Repair the complete draft above. %s\n"
                "The validation problem is: %s\n"
                "Preserve its story events, relationships and ending while "
                "correcting the structure. Return the complete corrected JSON "
                "object, with no commentary."
                % (instruction, error)
            )},
        ]
    return repair


# ---------------------------------------------------------------------------
# P0 -- interpret (planning notes, not another admission gate)
# ---------------------------------------------------------------------------

def _pass_interpret(technical_fn, pack, bundle, *, requested: int,
                    act_count: int, include_act_breaks: bool,
                    attempt_receipts=None) -> StoryInterpretation:
    norm = bundle.normalized
    fields = "\n\n".join(
        "%s:\n%s" % (_SI.FIELD_LABELS[name].upper(), getattr(norm, name))
        for name in _SI.CREATIVE_FIELDS if getattr(norm, name)
    )
    base, retry = _TEMP["interpret"]
    return _call(
        "interpret", bundle, attempt_receipts=attempt_receipts,
        prompt=[
            {"role": "system", "content": _seam(pack, "my_story_interpret_system")},
            {"role": "user", "content": (
                "WHAT THEY WROTE:\n\n%s\n\n"
                "THEIR SETTINGS:\n"
                "- characters requested: %d\n"
                "- acts: %d\n"
                "- music cues between acts: %d\n\n"
                "Interpret it now."
                % (fields, requested, act_count,
                   _interstitial_count(act_count, include_act_breaks))
            )},
        ],
        schema=StoryInterpretation,
        slot_fn=technical_fn,
        base_temperature=base,
        structural_retry_temperature=retry,
        repair_prompt_factory=make_dispatching_repair_factory(),
        max_attempts=3,
        helper_name="my_story_interpret",
    )


# ---------------------------------------------------------------------------
# P1 -- treatment
# ---------------------------------------------------------------------------

def _make_treatment_validator(act_count: int):
    def check(model: StoryTreatment) -> "str | None":
        names = model.names()
        folded = {_norm_ws(name).casefold() for name in names}
        if len(folded) != len(names) or "" in folded:
            return "cast names must be nonempty and unique"
        if ANNOUNCER_NAME.casefold() in folded:
            return "ANNOUNCER is reserved for the frame; give story characters distinct names"
        problems = []
        if len(model.acts) != act_count:
            problems.append("acts has %d entries; the selected count is %d"
                            % (len(model.acts), act_count))
        return "; ".join(problems) or None
    return check


def _pass_treatment(creative_fn, pack, bundle, interp: StoryInterpretation,
                    *, act_count: int, requested_characters: int,
                    include_act_breaks: bool, attempt_receipts=None) -> StoryTreatment:
    base, retry = _TEMP["treatment"]
    return _call(
        "treatment", bundle, attempt_receipts=attempt_receipts,
        prompt=[
            {"role": "system", "content": _seam(pack, "my_story_treatment_system")},
            {"role": "user", "content": (
                "THEIR IDEA, AS WRITTEN:\n\n%s\n\n"
                "THE INTERPRETATION:\n%s\n\n"
                "SELECTED ACTS: %d (binding). REQUESTED SPEAKING CHARACTERS: %d "
                "(flexible, announcer excluded).\n"
                "Let the supplied story guide the cast; preserve its people. Music cues between acts: %d.\n"
                "Plan the episode now."
                % (_SI.project_payload(bundle, "")["full_text"],
                   json.dumps(interp.model_dump(), ensure_ascii=False, indent=2),
                   act_count, requested_characters,
                   _interstitial_count(act_count, include_act_breaks))
            )},
        ],
        schema=StoryTreatment,
        slot_fn=creative_fn,
        base_temperature=base,
        structural_retry_temperature=retry,
        repair_prompt_factory=_full_artifact_repair(
            "Reorganize the treatment into exactly %d acts. The requested "
            "character count is flexible; preserve the listener's people, "
            "story material, relationships and ending; change the act grouping to fit."
            % act_count),
        post_validator=_make_treatment_validator(act_count),
        max_attempts=3,
        helper_name="my_story_treatment",
    )


# ---------------------------------------------------------------------------
# P2 -- the acts
# ---------------------------------------------------------------------------

def _make_act_validator(treatment: StoryTreatment, n: int,
                        must_speak: "tuple[str, ...]"):
    allowed = {_norm_ws(name).casefold(): name for name in treatment.names()}

    def check(model: ActScript) -> "str | None":
        heard: "set[str]" = set()
        for line in model.lines:
            key = _norm_ws(line.speaker).casefold()
            if key not in allowed:
                return ("%r is not in the cast; the speakers are %s"
                        % (line.speaker, ", ".join(treatment.names())))
            if not _norm_ws(line.text):
                return "%s has an empty line" % line.speaker
            line.speaker = allowed[key]
            if clean_spoken_text(line.text).strip():
                heard.add(line.speaker)
        missing = [name for name in must_speak if name not in heard]
        if missing:
            return (
                "this is the last act and %s has not spoken anywhere in the "
                "story yet; give them lines here"
                % ", ".join(repr(name) for name in missing)
            )
        return None
    return check


def _prior_digest(prev: "ActScript | None", plan: "ActPlan | None") -> str:
    """A bounded tail of the previous act, so this one continues rather than restarts."""
    if prev is None:
        return "This is the first act."
    tail = prev.lines[-_PRIOR_ACT_LINES:]
    spoken = "\n".join("%s: %s" % (l.speaker, _norm_ws(l.text)) for l in tail)
    ending = (plan.ending_state if plan is not None else "") or ""
    return ("HOW THE PREVIOUS ACT ENDED:\n%s\n\n%s"
            % (spoken, ("WHERE THAT LEAVES THE STORY:\n" + ending) if ending else ""))


def _pass_act(creative_fn, pack, bundle, treatment: StoryTreatment,
              plan: ActPlan, prev: "ActScript | None",
              prev_plan: "ActPlan | None", *, must_speak: "tuple[str, ...]",
              is_last: bool, attempt_receipts=None) -> ActScript:
    base, retry = _TEMP["act"]
    cast_block = "\n".join(
        "- %s (%s, %s): %s" % (c.name, c.gender, c.role or "in the story",
                               c.character_description)
        for c in treatment.cast
    )
    unheard = ""
    if must_speak:
        unheard = ("\nNOT YET HEARD IN THIS STORY: %s.%s\n"
                   % (", ".join(must_speak),
                      " This is the LAST act, so they must speak here."
                      if is_last else ""))
    return _call(
        "act_%d" % plan.n, bundle, attempt_receipts=attempt_receipts,
        prompt=[
            {"role": "system", "content": _seam(pack, "my_story_act_system")},
            {"role": "user", "content": (
                "THE ACCEPTED TREATMENT:\n%s\n\nTHE CAST:\n%s\n%s\n"
                "%s\n\nTHIS ACT (act %d of %d):\n"
                "- where: %s\n- what it accomplishes: %s\n- its beats: %s\n"
                "- where it should leave the story: %s\n\n"
                "Write act %d now."
                % (json.dumps(treatment.model_dump(by_alias=True), ensure_ascii=False), cast_block, unheard,
                   _prior_digest(prev, prev_plan), plan.n, len(treatment.acts),
                   plan.scene_setting or treatment.setting, plan.purpose,
                   "; ".join(plan.turns) or "as the story needs",
                   plan.ending_state, plan.n)
            )},
        ],
        schema=ActScript,
        slot_fn=creative_fn,
        base_temperature=base,
        structural_retry_temperature=retry,
        repair_prompt_factory=_full_artifact_repair(
            "Keep this one act and the locked treatment cast. Give the "
            "unheard cast named in the validation problem actual spoken "
            "dialogue; stage directions are not speech."),
        post_validator=_make_act_validator(treatment, plan.n, must_speak if is_last else ()),
        max_attempts=3,
        helper_name="my_story_act_%d" % plan.n,
    )


# ---------------------------------------------------------------------------
# P3 -- the frame
# ---------------------------------------------------------------------------

def _pass_frame(creative_fn, pack, bundle, treatment: StoryTreatment,
                *, attribution: str, inter_wanted: int, attempt_receipts=None) -> StoryFrame:
    base, retry = _TEMP["frame"]
    return _call(
        "frame", bundle, attempt_receipts=attempt_receipts,
        prompt=[
            {"role": "system", "content": _seam(pack, "my_story_frame_system")},
            {"role": "user", "content": (
                "THE EPISODE: %s\n%s\n\nSETTING: %s\n\n"
                "ATTRIBUTION SENTENCE (include verbatim):\n%s\n\n"
                "Interstitial cues wanted: %d.\n\nWrite the frame now."
                % (treatment.title, treatment.logline, treatment.setting,
                   attribution, inter_wanted)
            )},
        ],
        schema=StoryFrame,
        slot_fn=creative_fn,
        base_temperature=base,
        structural_retry_temperature=retry,
        repair_prompt_factory=make_dispatching_repair_factory(),
        max_attempts=3,
        helper_name="my_story_frame",
    )


# ---------------------------------------------------------------------------
# P4 -- voices (pure python, no model call)
# ---------------------------------------------------------------------------

def _nearest_timbre(text: str, fallback: str) -> str:
    """Map the treatment's free-text timbre onto the shared vocabulary.

    The voice picker ranks candidates by matching a timbre WORD against each
    voice's short description, so an unmapped adjective simply ranks nothing.
    Falling back to the rotating vocabulary keeps the ensemble varied instead
    of collapsing every character onto one column.
    """
    words = str(text or "").lower()
    for known in _OTRCAST._TIMBRE_VOCAB:
        if known in words:
            return known
    return fallback


def _assign_voices(treatment: StoryTreatment, rng: random.Random) -> "list[dict]":
    """Cast rows: announcer c01, then characters c02.. in treatment order.

    The LLM invented the people; Python picks the larynx. Deterministic under
    a fixed seed, and two characters never share a voice -- the pool is
    narrowed before each pick and the result is asserted afterwards.
    """
    announcer = dict(_POOLS.pick_announcer(rng))
    announcer["char_id"] = "c01"
    rows: "list[dict]" = [announcer]
    taken: "set[str]" = {str(announcer.get("voice_preset") or "")}
    for i, member in enumerate(treatment.cast):
        pool = _POOLS.open_voice_pool(set(taken))
        if not pool:
            raise MyStoryCastError(
                "voices",
                "the voice stock ran out at character %d (%s); no two "
                "characters may share a voice"
                % (i + 1, member.name),
            )
        slot = _OTRCAST.EnsembleSlot(
            char_id="c%02d" % (i + 2),
            name=member.name,
            gender=member.gender,
            timbre=_nearest_timbre(
                member.timbre,
                _OTRCAST._TIMBRE_VOCAB[i % len(_OTRCAST._TIMBRE_VOCAB)]),
            role=_OTRCAST._ROLE_VOCAB[i % len(_OTRCAST._ROLE_VOCAB)],
        )
        preset = _OTRCAST.python_assign_voice_preset(
            slot, available_voices=pool, rng=rng, age_band=member.age_band)
        taken.add(preset)
        rows.append({
            "char_id": slot.char_id,
            "name": member.name,
            "character_description": member.character_description,
            "gender": member.gender,
            "tts_model": "bark",
            "voice_preset": preset,
            "voice_params": None,
        })
    _OTRCAST._assert_unique_bark_voices(rows)
    return rows


# ---------------------------------------------------------------------------
# P5 -- assembly (pure python, no model call)
# ---------------------------------------------------------------------------

def _music_sentinel(shot_id: str, role: str, seq: int = 0) -> dict:
    """A music cue's placeholder line row.

    Text is empty and char_id == speaker_role == the role: music[] is the cue
    authority, and this row exists so the cue has a position in the timeline.
    """
    lid = "%s_music" % shot_id if seq == 0 else "%s_music_%d" % (shot_id, seq + 1)
    return {
        "line_id": lid, "beat_id": None, "shot_id": shot_id,
        "char_id": role, "speaker_role": role, "boundary": None,
        "text": "",
    }


def _assemble(led: Any, treatment: StoryTreatment, acts: "list[ActScript]",
              frame: StoryFrame, cast_rows: "list[dict]", *,
              owner_bank: str, interpretation: StoryInterpretation,
              include_act_breaks: bool) -> None:
    """Emit all five ledger hierarchies. Timing stays unset -- SceneSequencer owns it."""
    char_id_by_name = {
        _norm_ws(r["name"]).casefold(): r["char_id"]
        for r in cast_rows if r["name"] != ANNOUNCER_NAME
    }
    led.set_cast(cast_rows)
    meta = _ledger_meta(led)
    meta["cast_status"] = "locked"

    scene_rows: "list[dict]" = []
    shot_rows: "list[dict]" = []
    beat_rows: "list[dict]" = []
    line_rows: "list[dict]" = []
    music_rows: "list[dict]" = []

    def spoken(line_id: str, shot_id: str, char_id: str, role: str,
               speaker: str, text: str, boundary: "str | None") -> dict:
        return {
            "line_id": line_id, "beat_id": line_id, "shot_id": shot_id,
            "speaker": speaker, "char_id": char_id, "speaker_role": role,
            "boundary": boundary, "text": _norm_ws(text),
        }

    def beat(row: dict, scene_id: "str | None") -> None:
        beat_rows.append({
            "beat_id": row["beat_id"], "shot_id": row["shot_id"],
            "scene_id": scene_id, "speaker": row["speaker"],
            "char_id": row["char_id"], "line_ids": [row["line_id"]],
        })

    # --- preamble: opening theme + the announcer's open ------------------
    shot_rows.append({"shot_id": "shot_000", "scene_id": None,
                      "description": "preamble"})
    opening = _music_sentinel("shot_000", "music_open")
    line_rows.append(opening)
    music_rows.append({
        "cue_id": "opening", "description": frame.music_open,
        "generation_prompt": frame.music_open, "placement": "opening",
        "anchor_line_id": opening["line_id"],
    })
    for k, text in enumerate(text for text in frame.announcer_intro if text.strip()):
        row = spoken("shot_000_b%d" % (k + 1), "shot_000", ANNOUNCER_CHAR_ID,
                     "announcer", ANNOUNCER_NAME, text,
                     "shot_start" if k == 0 else "beat_start")
        line_rows.append(row)
        beat(row, None)
    led.set_lines(line_rows)
    _require_ledger_save(led, "the assembled preamble")

    # --- one scene and one shot per act ----------------------------------
    inter_seq = 0
    inter_wanted = _interstitial_count(len(acts), include_act_breaks)
    for act in acts:
        scene_id = "s%02d" % act.n
        shot_id = "shot_%03d" % act.n
        plan = next((a for a in treatment.acts if a.n == act.n), None)
        setting = act.scene_setting or (plan.scene_setting if plan else "") \
            or treatment.setting
        scene_rows.append({"scene_id": scene_id, "description": setting})
        shot_rows.append({"shot_id": shot_id, "scene_id": scene_id,
                          "description": setting})
        # Consecutive lines from one speaker are ONE turn: a beat is a
        # continuous turn, and splitting it would hand the voice bus two
        # clips where the audience hears one person still talking.
        runs: "list[tuple[str, list[str]]]" = []
        for line in act.lines:
            if runs and _norm_ws(runs[-1][0]).casefold() == _norm_ws(line.speaker).casefold():
                runs[-1][1].append(line.text)
            else:
                runs.append((line.speaker, [line.text]))
        for k, (speaker, texts) in enumerate(runs):
            row = spoken(
                "%s_b%d" % (shot_id, k + 1), shot_id,
                char_id_by_name[_norm_ws(speaker).casefold()], "character",
                speaker, " ".join(_norm_ws(t) for t in texts),
                "shot_start" if k == 0 else "beat_start")
            line_rows.append(row)
            beat(row, scene_id)
        # An interstitial cue after every act but the last.
        if inter_seq < inter_wanted:
            cue = frame.music_inter[inter_seq] if inter_seq < len(frame.music_inter) else ""
            sentinel = _music_sentinel(shot_id, "music_inter")
            line_rows.append(sentinel)
            inter_seq += 1
            music_rows.append({
                "cue_id": "inter_%02d" % inter_seq, "description": cue,
                "generation_prompt": cue, "placement": "interstitial",
                "anchor_line_id": sentinel["line_id"],
            })
        led.set_lines(line_rows)
        _require_ledger_save(led, "assembled act %d" % act.n)

    # --- postamble: the announcer's close, the coda, the closing theme ---
    post_shot = "shot_%03d" % (len(acts) + 1)
    shot_rows.append({"shot_id": post_shot, "scene_id": None,
                      "description": "postamble"})
    k = 0
    for text in list(frame.announcer_outro) + [frame.coda]:
        if not text.strip():
            continue
        k += 1
        row = spoken("%s_b%d" % (post_shot, k), post_shot, ANNOUNCER_CHAR_ID,
                     "announcer", ANNOUNCER_NAME, text,
                     "shot_start" if k == 1 else "beat_start")
        line_rows.append(row)
        beat(row, None)
    closing = _music_sentinel(post_shot, "music_close")
    line_rows.append(closing)
    music_rows.append({
        "cue_id": "closing", "description": frame.music_close,
        "generation_prompt": frame.music_close, "placement": "closing",
        "anchor_line_id": closing["line_id"],
    })

    led.set_scenes(scene_rows)
    led.set_shots(shot_rows)
    led.set_beats(beat_rows)
    led.set_lines(line_rows)
    led.set_music(music_rows)

    meta = _ledger_meta(led)
    meta.setdefault("source_bank", owner_bank)
    meta.setdefault("my_story", {})["music_cue_disposition"] = [
        {"proposal_index": i, "description": cue,
         "disposition": "unused_surplus" if include_act_breaks else "act_breaks_disabled"}
        for i, cue in enumerate(frame.music_inter)
        if i >= inter_wanted
    ]

    # The authorship receipt: every voiced row's text must be a verbatim
    # constituent of an artifact we accepted. This is what the read-only
    # freeze re-verifies instead of rewriting the text.
    try:
        from ._otr_content_authorship import stamp_receipt
    except ImportError:  # pragma: no cover -- flat load
        from _otr_content_authorship import stamp_receipt  # type: ignore
    stamp_receipt(
        led.data, owner_bank=owner_bank,
        accepted_artifacts={
            "interpretation": interpretation.model_dump(mode="json"),
            "treatment": treatment.model_dump(mode="json"),
            "acts": [a.model_dump(mode="json") for a in acts],
            "frame": frame.model_dump(mode="json"),
        },
    )

    # The delivery text the voice nodes actually speak (Dr. -> Doctor, digits
    # -> words), stamped beside the sealed canonical text, never over it.
    try:
        from ._otr_readiness import stamp_text_for_tts_delivery
    except ImportError:  # pragma: no cover -- flat load
        from _otr_readiness import stamp_text_for_tts_delivery  # type: ignore
    stamp_text_for_tts_delivery(led)

    _require_ledger_save(led, "the assembled ledger and its receipts")


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def run_my_story_episode(
    *,
    payload: "dict[str, Any]",
    pack: Any,
    resolved: dict,
    led: Any,
    meta: dict,
    creative_fn: Callable[..., str],
    technical_fn: Callable[..., str],
    slot_scheduler: Any,
    source_bank_row: Any,
    episode_root: Any,
    episode_id: str,
) -> MyStoryTailParts:
    """Turn one person's typed idea into a complete, performable ledger."""
    del episode_root, episode_id, payload  # the bundle below is the source

    source_meta = dict(resolved.get("source_meta") or {})
    stored = source_meta.get("story_input")
    if not isinstance(stored, Mapping):
        raise MyStoryError(
            "input",
            "this run reached the My Story runner without an admitted input "
            "bundle; the writer's admission step did not execute",
        )
    bundle = _SI.StoryInputBundle(
        fields=_SI.RawStoryFields(**dict(stored.get("fields") or {})),
        normalized=_SI.RawStoryFields(**dict(stored.get("normalized") or {})),
        request=_SI.StoryRequest(**dict(stored.get("request") or {})),
        digest=str(stored.get("digest") or ""),
    )

    act_count = int(resolved.get("act_count") or 1)
    include_act_breaks = bool(resolved.get("include_act_breaks", True))
    requested = max(1, int(resolved.get("num_characters") or 1))
    raw_requested = int(bundle.request.num_characters)
    author = str(source_meta.get("story_author") or "")

    seed = _resolve_seed()
    rng = random.Random(seed)
    meta = _ledger_meta(led)
    meta["episode_seed"] = seed
    _OTRWD.stamp_contract(meta, owner="my_story")

    receipts: "list[dict]" = []

    def receipt(pass_id: str, model_id: str, temp: float,
                tokens: "int | None") -> None:
        receipts.append({"pass_id": pass_id, "model_id": model_id,
                         "temp": temp, "max_new_tokens": tokens,
                         "budget_mode": "python" if model_id == "python" else "provider_capacity",
                         "status": "completed"})

    creative_model = str(resolved.get("creative_writing_model") or "")
    technical_model = str(resolved.get("technical_model") or "")

    story: "dict[str, Any]" = {
        "schema_version": MY_STORY_SCHEMA,
        "seed": seed,
        "draft_digest": bundle.digest,
        "attempts": [],
        "counts": {
            "requested_acts": act_count, "proposed_acts": None,
            "accepted_acts": None, "actual_acts": None,
            "requested_characters": raw_requested, "planned_characters": None,
            "proposed_characters": None, "accepted_characters": None,
            "actual_characters": None,
        },
        "notes": [
            "The person's own typed fields are the sole source.",
            "Selected acts bind treatment acceptance; character count is flexible guidance.",
        ],
    }
    meta["my_story"] = story
    meta["news"] = None

    # The cameo knob belongs to the house, and this cast belongs to the
    # person who described it. Recorded rather than silently ignored.
    if resolved.get("lemmy_force") is not None:
        story["lemmy_knob_ignored"] = True
        log.info("[my_story] the cameo setting does not apply on this bank: "
                 "the cast is the listener's own")

    primary_error: BaseException | None = None
    try:
        # --- P0 interpret ----------------------------------------------------
        # Interpretation records intent; selected acts own the act structure.
        with _helper_ctx(slot_scheduler, "my_story_interpret"):
            interp = _pass_interpret(
                technical_fn, pack, bundle, requested=requested,
                act_count=act_count, include_act_breaks=include_act_breaks,
                attempt_receipts=story["attempts"])
        receipt("interpret", technical_model, _TEMP["interpret"][0],
                None)
        story["interpretation"] = interp.model_dump(mode="json")
        story["cast_plan"] = interp.cast_plan.model_dump(mode="json")
        story["counts"]["planned_characters"] = interp.cast_plan.planned
        if interp.assumptions:
            log.info("[my_story] filled %d unstated detail(s): %s",
                     len(interp.assumptions), "; ".join(interp.assumptions[:3]))
        if interp.conflicts:
            for conflict in interp.conflicts:
                log.warning("[my_story] could not honour %r as written: %s -- "
                            "the story will %s", conflict.requirement_id,
                            conflict.why, conflict.resolution)
        _require_ledger_save(led, "the story interpretation")

        # --- P1 treatment ----------------------------------------------------
        with _helper_ctx(slot_scheduler, "my_story_treatment"):
            treatment = _pass_treatment(
                creative_fn, pack, bundle, interp,
                act_count=act_count, requested_characters=requested,
                include_act_breaks=include_act_breaks, attempt_receipts=story["attempts"])
        receipt("treatment", creative_model, _TEMP["treatment"][0],
                None)
        story["treatment_proposal"] = treatment.model_dump(mode="json")
        story["counts"].update(accepted_acts=len(treatment.acts),
                               accepted_characters=len(treatment.cast))
        story["act_number_normalization"] = {
            "treatment": [{"original": plan.n, "slot": i}
                          for i, plan in enumerate(treatment.acts, 1)],
            "replies": [],
        }
        for i, plan in enumerate(treatment.acts, 1):
            plan.n = i
        names = {_norm_ws(c.name).casefold(): c for c in treatment.cast}
        discrepancies = []
        for person in interp.named_cast:
            member = names.get(_norm_ws(person.name).casefold())
            if person.required and person.speaking and member is None:
                discrepancies.append({"name": person.name, "kind": "named_speaker_not_in_selected_cast"})
            elif member is not None and person.stated_gender and person.stated_gender != member.gender:
                discrepancies.append({"name": person.name, "kind": "stated_gender_differs",
                                      "stated": person.stated_gender, "accepted": member.gender})
        story["fidelity_discrepancies"] = discrepancies
        story["treatment"] = treatment.model_dump(mode="json")
        _require_ledger_save(led, "the treatment")

        # --- P2 acts, one call each ------------------------------------------
        acts: "list[ActScript]" = []
        heard: "set[str]" = set()
        everyone = list(treatment.names())
        for index, plan in enumerate(treatment.acts):
            is_last = index == len(treatment.acts) - 1
            # Only the LAST act carries the requirement; earlier acts are merely
            # told who has not spoken, so the story can hold someone back for
            # effect without the ladder fighting it.
            unheard = tuple(n for n in everyone if n not in heard)
            must_speak = unheard
            prev = acts[-1] if acts else None
            prev_plan = treatment.acts[index - 1] if index else None
            with _helper_ctx(slot_scheduler, "my_story_act_%d" % plan.n):
                act = _pass_act(creative_fn, pack, bundle, treatment, plan, prev,
                                prev_plan, must_speak=must_speak, is_last=is_last,
                                attempt_receipts=story["attempts"])
            story["act_number_normalization"]["replies"].append(
                {"original": act.n, "slot": plan.n})
            act.n = plan.n
            acts.append(act)
            for line in act.lines:
                if not clean_spoken_text(line.text).strip():
                    continue
                key = _norm_ws(line.speaker).casefold()
                for name in everyone:
                    if _norm_ws(name).casefold() == key:
                        heard.add(name)
            receipt("act_%d" % plan.n, creative_model, _TEMP["act"][0],
                    None)
            story["acts_accepted"] = len(acts)
            _require_ledger_save(led, "act %d" % plan.n)

        # Every character the treatment cast must be heard. The freeze fails an
        # episode with a silent cast member, and failing there would waste the
        # whole render; this says so now, and names them.
        silent = [name for name in everyone if name not in heard]
        if silent:
            raise MyStoryCastError(
                "acts",
                "%s never speaks in the finished story. Every character in the "
                "cast must be heard." % ", ".join(repr(n) for n in silent),
            )

        # --- P3 frame --------------------------------------------------------
        attribution = _SI.attribution_sentence(author)
        inter_wanted = _interstitial_count(len(acts), include_act_breaks)
        with _helper_ctx(slot_scheduler, "my_story_frame"):
            frame = _pass_frame(creative_fn, pack, bundle, treatment,
                                attribution=attribution, inter_wanted=inter_wanted,
                                attempt_receipts=story["attempts"])
        receipt("frame", creative_model, _TEMP["frame"][0], None)
        story["frame_proposal"] = frame.model_dump(mode="json")
        spoken_frame = " ".join(frame.announcer_intro + frame.announcer_outro + [frame.coda])
        if _norm_ws(attribution) not in _norm_ws(spoken_frame):
            frame.announcer_outro.append(attribution)
        story["frame"] = frame.model_dump(mode="json")
        story["attribution"] = _SI.attribution_receipt(author)

        # --- P4 voices (no model call) ---------------------------------------
        cast_rows = _assign_voices(treatment, rng)
        receipt("voices", "python", 0.0, None)

        # --- P5 assemble (no model call) -------------------------------------
        _assemble(led, treatment, acts, frame, cast_rows, interpretation=interp,
                  include_act_breaks=include_act_breaks,
                  owner_bank=str(getattr(source_bank_row, "source_bank_id", "")
                                 or "my_story"))
        receipt("assemble", "python", 0.0, None)

        # `_assemble` saves, and Ledger.save() rebinds led.data -- reacquire.
        meta = _ledger_meta(led)
        story = meta.setdefault("my_story", story)
        story["counts"].update(actual_acts=len(led.data.get("scenes") or []),
                               actual_characters=_OTRCAST.count_locked_characters(
                                   led.data.get("cast") or []))
        story["pass_receipts"] = receipts
        story["delivery_telemetry"] = _OTRWD.stamp_actual(
            led.data, stage="my_story_assembled")

        # The cameo contract this lane never rolled, stamped rather than omitted
        # so a reader can tell a declined cameo from one never considered.
        meta["cast_contract"] = _OTRCAST.content_owned_cast_contract(
            source_bank_id=str(getattr(source_bank_row, "source_bank_id", "")
                               or "my_story"),
            num_characters_request=raw_requested,
            num_characters_locked=_OTRCAST.count_locked_characters(
                led.data.get("cast") or []),
            decision=None,
        )
        _require_ledger_save(led, "the My Story receipts")

        canon = _OTRC.episode_canon_from_outline_dict({
            "title": treatment.title,
            "premise": treatment.dramatic_question,
            "setting": treatment.setting,
            "time_of_day": treatment.time_of_day or "night",
            "sound_palette": [],
        })
        log.info("[my_story] complete: seed=%s cast=%d acts=%d by=%r",
                 seed, len(treatment.cast), len(acts), author or "(a listener)")
        return MyStoryTailParts(
            outline_view=MyStoryOutlineView(
                premise=treatment.dramatic_question,
                title=treatment.title,
                setting=treatment.setting,
            ),
            canon=canon,
            final_title_override=treatment.title if treatment.title.strip() else None,
            run_story_spine=False,
            refine_active=False,
            my_story_meta=story,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        for attempt in story["attempts"]:
            if attempt["pass_id"] != "treatment":
                continue
            try:
                proposed = parse_first_json_object(attempt["raw_output"])
            except (ValueError, TypeError):
                continue
            if isinstance(proposed, dict):
                for field, key in (("acts", "proposed_acts"), ("cast", "proposed_characters")):
                    if story["counts"][key] is None and isinstance(proposed.get(field), list):
                        story["counts"][key] = len(proposed[field])
        try:
            _require_ledger_save(led, "the My Story attempt history")
        except Exception as save_error:
            if primary_error is None:
                raise
            # Preserve the original provider/cancellation failure while making
            # the missing durable receipt explicit in both traceback and log.
            if hasattr(primary_error, "add_note"):
                primary_error.add_note("My Story attempt history also failed to persist: %s" % save_error)
            log.error("[my_story] attempt history did not persist during failure: %s",
                      save_error, exc_info=True)



__all__ = [
    "ActScript",
    "CastMember",
    "CastPlan",
    "Conflict",
    "MY_STORY_SCHEMA",
    "MyStoryCastError",
    "MyStoryError",
    "MyStoryOutlineView",
    "MyStoryTailParts",
    "NamedCast",
    "Requirement",
    "SpokenLine",
    "StoryFrame",
    "StoryInterpretation",
    "StoryTreatment",
    "run_my_story_episode",
]
