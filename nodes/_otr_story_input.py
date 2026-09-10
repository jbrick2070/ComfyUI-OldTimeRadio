"""My Story input intake -- the listener's own words, typed and admitted once.

PURE. No file I/O, no model call, no runner import, and nothing from the
writer. It takes the raw widget strings, decides whether this run is a My Story
run at all, refuses the combinations that cannot mean anything, and projects
the accepted bundle into the seven-string source payload every downstream
consumer already reads.

THE ONE ADMISSION CONTRACT, CALLED FROM THREE PLACES. The writer checks twice
(once before the rolls, on the literal widget values, and once after the bank
row is bound) and the workflow validator checks the queued prompt before it
downloads a single visual asset. All three call :func:`check_selection`, so
they cannot disagree about what is admissible -- a disagreement between a
pre-flight check and the thing it is protecting is worse than no check at all.

WHAT IS DELIBERATELY NOT HERE: persistence (``_otr_story_drafts`` owns the one
draft file), prose generation (the model writes; this module only carries what
the listener typed), and any judgement about whether an idea is *good*. Length,
style and taste never refuse a run.

Stdlib only. UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Bundle schema version. Bump only when the DIGESTED shape changes -- the
#: digest is an identity, and a reader that cannot recompute it cannot verify
#: the draft it is holding.
STORY_INPUT_SCHEMA = "my_story_input_v1"

#: The bank-row default that switches a bank onto this route. Absent means the
#: legacy source route, which is every shipped bank today.
INPUT_MODE_LEGACY = "legacy"
INPUT_MODE_USER_FIELDS = "user_fields_v1"
KNOWN_INPUT_MODES = frozenset({INPUT_MODE_LEGACY, INPUT_MODE_USER_FIELDS})

#: The four creative fields, in the order they are shown and projected. The
#: author is deliberately NOT one of them: naming who a story is by is not an
#: idea for a story, and a run carrying only an author has nothing to write.
CREATIVE_FIELDS = ("idea", "characters", "plot", "setting")

#: The three fields that belong to My Story alone. `custom_premise` is shared
#: with every other bank and keeps its existing meaning there, so it is not in
#: this set -- typing a premise on the archive lane is ordinary, typing
#: character notes there is a mistake worth naming.
DEDICATED_FIELDS = ("characters", "plot", "setting", "author")

#: How much of a creative field the headline projection may show.
_HEADLINE_CHARS = 80


class StoryInputError(ValueError):
    """An input combination that cannot mean anything. Always actionable.

    A ValueError subclass on purpose: ComfyUI surfaces it as a node error with
    the message intact, which is the whole point -- the listener needs to read
    what to change, not a stack trace.
    """


@dataclass(frozen=True)
class RawStoryFields:
    """Exactly what the listener typed, before any normalization.

    Kept beside the normalized form rather than replacing it: the raw text is
    the EVIDENCE of what was asked for, and a bug in normalization must never
    be able to erase it.
    """

    idea: str = ""
    characters: str = ""
    plot: str = ""
    setting: str = ""
    author: str = ""

    def as_dict(self) -> dict:
        return {
            "idea": self.idea,
            "characters": self.characters,
            "plot": self.plot,
            "setting": self.setting,
            "author": self.author,
        }


@dataclass(frozen=True)
class StoryRequest:
    """The non-text controls, captured BEFORE the bank and style rolls.

    ``source_bank_requested`` and ``visual_style_requested`` are the LITERAL
    widget values, sentinel included. Capturing them after the rolls would
    record the drawn result as though the listener had asked for it, and would
    change the draft digest between the validator and the writer -- the two
    would then persist and verify different identities for one run.
    """

    num_characters: int = 2
    act_count: str = ""
    include_act_breaks: bool = True
    source_bank_requested: str = ""
    visual_style_requested: str = ""

    def as_dict(self) -> dict:
        return {
            "num_characters": int(self.num_characters),
            "act_count": str(self.act_count),
            "include_act_breaks": bool(self.include_act_breaks),
            "source_bank_requested": str(self.source_bank_requested),
            "visual_style_requested": str(self.visual_style_requested),
        }


@dataclass(frozen=True)
class StoryInputPolicy:
    """The resolved input route for one bank id.

    ``bank_row`` may be None -- at the writer's first check site the value in
    the widget can still be the roll sentinel, which is not a bank id and never
    will be. A None row answers `legacy`, which is what lets the pre-roll check
    run at all.
    """

    mode: str = INPUT_MODE_LEGACY
    bank_id: str = ""

    @property
    def is_user_fields(self) -> bool:
        return self.mode == INPUT_MODE_USER_FIELDS


@dataclass(frozen=True)
class StoryInputBundle:
    """The admitted input: raw evidence, normalized view, controls, identity."""

    fields: RawStoryFields
    normalized: RawStoryFields
    request: StoryRequest
    digest: str
    schema_version: str = STORY_INPUT_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "fields": self.fields.as_dict(),
            "normalized": self.normalized.as_dict(),
            "request": self.request.as_dict(),
            "digest": self.digest,
        }


def _text(value: Any, name: str) -> str:
    """One widget value as a string, or a named refusal.

    A non-string here means the caller handed us a link, a number or a list
    where a STRING widget was declared. Guessing is how a link's `[node, slot]`
    pair ends up inside an episode as prose.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise StoryInputError(
            "my_story: %s must be text, got %s. A linked or converted input "
            "cannot be read here; type the value on the node, or leave it "
            "empty." % (name, type(value).__name__)
        )
    return value


def capture_raw(
    *,
    idea: Any = "",
    characters: Any = "",
    plot: Any = "",
    setting: Any = "",
    author: Any = "",
) -> RawStoryFields:
    """The five text widgets, verbatim. No stripping, no I/O."""
    return RawStoryFields(
        idea=_text(idea, "the story input"),
        characters=_text(characters, "character notes"),
        plot=_text(plot, "plot ideas"),
        setting=_text(setting, "setting"),
        author=_text(author, "story by"),
    )


def normalize(fields: RawStoryFields) -> RawStoryFields:
    """Whitespace-normalized view. The raw fields are never modified.

    Collapses runs of blank lines and trims each end. Interior single newlines
    survive, because a listener who typed a list of characters on separate
    lines meant those lines.
    """
    def one(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", str(text or "").replace("\r\n", "\n")).strip()

    return RawStoryFields(
        idea=one(fields.idea),
        characters=one(fields.characters),
        plot=one(fields.plot),
        setting=one(fields.setting),
        author=one(fields.author),
    )


def creative_input_present(fields: RawStoryFields) -> bool:
    """Is there anything to write a story FROM?

    The author is excluded deliberately -- see :data:`CREATIVE_FIELDS`.
    """
    norm = normalize(fields)
    return any(getattr(norm, name) for name in CREATIVE_FIELDS)


def dedicated_fields_present(fields: RawStoryFields) -> bool:
    """Did the listener fill a field only My Story reads?"""
    norm = normalize(fields)
    return any(getattr(norm, name) for name in DEDICATED_FIELDS)


def filled_dedicated_fields(fields: RawStoryFields) -> "tuple[str, ...]":
    """Which My Story-only fields carry text, for a message that names them."""
    norm = normalize(fields)
    return tuple(name for name in DEDICATED_FIELDS if getattr(norm, name))


#: The label each field shows in the graph, so an error names what the reader
#: sees rather than the internal key.
FIELD_LABELS = {
    "idea": "Story input",
    "characters": "Character names and notes",
    "plot": "Plot ideas",
    "setting": "Setting",
    "author": "Story by",
}


def check_selection(
    fields: RawStoryFields,
    policy: StoryInputPolicy,
    *,
    source_ref: str = "",
    replay_from: str = "",
    snapshot_manifest_configured: bool = False,
    creative_input_deferred: bool = False,
) -> None:
    """Refuse every input combination that cannot produce an episode.

    Called at all three admission points with the same arguments, so the cheap
    early refusal and the real one are the same decision. Returns None on
    success; raises :class:`StoryInputError` with the fix in the message.
    """
    filled = filled_dedicated_fields(fields)
    if not policy.is_user_fields:
        # The listener typed into fields no other bank reads. Silently
        # dropping them is the failure this check exists to prevent: the
        # episode would render, ignore the character notes, and look correct.
        if filled:
            names = ", ".join(FIELD_LABELS[name] for name in filled)
            raise StoryInputError(
                "my_story: %s %s filled, but the selected source is %s. Those "
                "fields are read only by the My Story bank. Select "
                "'my_story' in Source, or clear those fields."
                % (names, "is" if len(filled) == 1 else "are",
                   _describe_selection(policy))
            )
        return

    # --- from here down the listener DID select My Story -------------------
    if str(replay_from or "").strip():
        raise StoryInputError(
            "my_story: replay_from is set, which re-renders a frozen episode "
            "and would ignore everything you typed. Clear replay_from to "
            "generate from your story input."
        )
    if snapshot_manifest_configured:
        raise StoryInputError(
            "my_story: a source-snapshot manifest is configured "
            "(OTR_SOURCE_SNAPSHOT_MANIFEST), which replays a frozen source "
            "and would replace your story input. Unset that variable to "
            "generate from what you typed."
        )
    if str(source_ref or "").strip():
        raise StoryInputError(
            "my_story: source_ref names an external source to fetch, and this "
            "bank has no fetcher -- your own words are the source. Clear "
            "source_ref, or pick a bank that resolves references."
        )
    if not creative_input_present(fields) and not creative_input_deferred:
        raise StoryInputError(
            "my_story: nothing to write from. Type your idea in '%s' -- or "
            "fill any of '%s', '%s', '%s'. The character count and story size "
            "are settings, not an idea, and naming who the story is by is not "
            "one either."
            % (FIELD_LABELS["idea"], FIELD_LABELS["characters"],
               FIELD_LABELS["plot"], FIELD_LABELS["setting"])
        )


def _describe_selection(policy: StoryInputPolicy) -> str:
    """Name the selected source in a way a reader recognises."""
    bank = str(policy.bank_id or "").strip()
    if not bank:
        return "not a My Story selection"
    return repr(bank)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))


def compute_digest(fields: RawStoryFields, request: StoryRequest) -> str:
    """Content identity for one submission.

    Over the RAW fields and the pre-roll request only. Deliberately excludes
    every timestamp and every rolled result, so the same submission has the
    same identity at the validator and at the writer, and re-running an
    unchanged input reuses its draft instead of littering new ones.
    """
    return hashlib.sha256(_canonical_json({
        "schema_version": STORY_INPUT_SCHEMA,
        "fields": fields.as_dict(),
        "request": request.as_dict(),
    }).encode("utf-8")).hexdigest()


def build_bundle(fields: RawStoryFields, request: StoryRequest) -> StoryInputBundle:
    """The admitted bundle. Pure and deterministic."""
    return StoryInputBundle(
        fields=fields,
        normalized=normalize(fields),
        request=request,
        digest=compute_digest(fields, request),
    )


def _labelled_projection(norm: RawStoryFields) -> str:
    """The listener's fields as one labelled block for the model to read.

    Labelled rather than concatenated because the boundaries carry meaning: a
    name under CHARACTERS is a cast requirement, the same name inside PLOT may
    be someone merely mentioned. Blank sections are omitted so the model is
    never handed an empty heading to fill.
    """
    parts = []
    for name, heading in (("idea", "IDEA"), ("characters", "CHARACTERS"),
                          ("plot", "PLOT"), ("setting", "SETTING"),
                          ("author", "BY")):
        value = getattr(norm, name)
        if value:
            parts.append("%s:\n%s" % (heading, value))
    return "\n\n".join(parts)


def first_creative_value(norm: RawStoryFields) -> str:
    """The first non-blank creative field, in display order."""
    for name in CREATIVE_FIELDS:
        value = getattr(norm, name)
        if value:
            return value
    return ""


def project_payload(bundle: StoryInputBundle, today: str) -> dict:
    """The seven-key source payload, synthesized from the listener's fields.

    Shape parity with every other lane is the point: downstream code reads a
    source payload, not a bank. ``full_text`` is the labelled projection and
    ``seed_text`` the first creative field, so the payload's own contract
    (seed_text non-empty) holds for every combination that passed
    :func:`check_selection`.
    """
    norm = bundle.normalized
    seed = first_creative_value(norm)
    if not seed:
        raise StoryInputError(
            "my_story: cannot project an empty submission -- check_selection "
            "must run before the payload is built."
        )
    headline = seed.replace("\n", " ").strip()
    if len(headline) > _HEADLINE_CHARS:
        headline = headline[:_HEADLINE_CHARS].rstrip() + "..."
    return {
        "headline": "My Story: " + headline,
        "summary": "",
        "full_text": _labelled_projection(norm),
        "source": "My Story (listener idea)",
        "date": str(today or ""),
        "link": "",
        "seed_text": seed,
    }


# ---------------------------------------------------------------------------
# Attribution -- who the story is by
# ---------------------------------------------------------------------------

#: What the announcer says when nobody is named. Never the operator's name,
#: never a machine byline: a listener supplied the idea and that is the honest
#: sentence.
ANONYMOUS_ATTRIBUTION = "Tonight's story comes from one of our listeners."

#: Printed credit when nobody is named. The bank row carries the same wording
#: as its default; this constant is what the override falls back to.
ANONYMOUS_CREDIT = (
    "a story from a listener's own idea, produced by machine for this broadcast"
)


def attribution_sentence(author: str) -> str:
    """The spoken attribution line. Python owns this, not the model.

    Authored here so the name reaches the microphone exactly as it was typed:
    a model asked to "mention the author" will paraphrase, and a paraphrased
    name is the wrong name.
    """
    name = str(author or "").strip()
    if not name:
        return ANONYMOUS_ATTRIBUTION
    return "Tonight's story is by %s." % name


def credits_source_line(author: str) -> str:
    """The printed credit for the closing crawl."""
    name = str(author or "").strip()
    if not name:
        return ANONYMOUS_CREDIT
    return "a story by %s, produced by machine for this broadcast" % name


def attribution_receipt(author: str) -> dict:
    """What the ledger records about attribution, including its absence."""
    name = str(author or "").strip()
    return {
        "author": name,
        "sentence": attribution_sentence(name),
        "credits_source_line": credits_source_line(name),
        "source": "story_author widget" if name else "none supplied",
    }


# ---------------------------------------------------------------------------
# Queued-prompt admission (the workflow validator's half)
# ---------------------------------------------------------------------------

#: A ComfyUI link in a queued prompt is ``[node_id, output_slot]``. A widget
#: value is the literal. Telling them apart is what lets the validator refuse a
#: blank literal submission while DEFERRING a linked one to the writer.
def is_link(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and bool(value[0])
            and type(value[1]) is int and value[1] >= 0)


@dataclass
class QueuedWriter:
    """One writer node in the queued prompt that this validator gates."""

    node_id: str
    inputs: Mapping[str, Any] = field(default_factory=dict)

    def literal(self, name: str, default: str = "") -> Any:
        """The literal value, or the link, or the default when absent.

        Absent means an older saved graph that predates the field -- which is
        an empty string, never a link. Reading it as anything else would make
        every pre-My-Story workflow look like it had a deferred input.
        """
        if name not in self.inputs:
            return default
        return self.inputs[name]


def writers_gated_by(prompt: Any, unique_id: Any,
                     *, writer_class: str = "OTR_LedgerScriptWriter",
                     gate_input: str = "gate_in") -> "list[QueuedWriter]":
    """Every writer whose gate input is wired to THIS validator.

    Reachability, not proximity: a graph may hold several writers and several
    validators, and a validator speaks only for the writers that actually
    depend on it. The validator has one output slot, so any link out of it is
    ``[unique_id, 0]``.
    """
    out: "list[QueuedWriter]" = []
    if not isinstance(prompt, Mapping):
        return out
    me = str(unique_id).strip() if unique_id is not None else ""
    if not me:
        return out
    for node_id, node in prompt.items():
        if not isinstance(node, Mapping):
            continue
        if str(node.get("class_type") or "") != writer_class:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        gate = inputs.get(gate_input)
        if is_link(gate) and str(gate[0]) == me and gate[1] == 0:
            out.append(QueuedWriter(node_id=str(node_id), inputs=inputs))
    return out


def read_queued_fields(writer: QueuedWriter) -> "tuple[RawStoryFields, tuple[str, ...]]":
    """(fields, deferred) for one queued writer.

    A linked field contributes NOTHING to the fields object and is named in
    ``deferred`` instead, so a bundle whose only creative text arrives over a
    link is deferred to the writer's own check rather than refused here for
    being blank.
    """
    mapping = {
        "idea": "custom_premise",
        "characters": "story_characters",
        "plot": "story_plot",
        "setting": "story_setting",
        "author": "story_author",
    }
    values: "dict[str, str]" = {}
    deferred: "list[str]" = []
    for name, widget in mapping.items():
        raw = writer.literal(widget, "")
        if is_link(raw):
            deferred.append(name)
            values[name] = ""
            continue
        values[name] = _text(raw, FIELD_LABELS[name])
    return RawStoryFields(**values), tuple(deferred)


@dataclass
class AdmittedWriter:
    """One queued writer whose literal My Story input was admitted here."""

    node_id: str
    bundle: StoryInputBundle


def check_queued_prompt(
    prompt: Any,
    unique_id: Any,
    *,
    resolve_policy,
    snapshot_manifest_configured: bool = False,
) -> "list[AdmittedWriter]":
    """Judge every writer this validator gates, BEFORE any asset is fetched.

    This is the half of admission that has to happen outside the writer,
    because by the time the writer runs the validator has already downloaded
    visual weights -- and a blank submission should not cost a multi-gigabyte
    fetch before anyone notices there is no story.

    Returns the writers whose LITERAL input was admitted, so the caller can
    persist their drafts. Raises :class:`StoryInputError` on the first writer
    that cannot run.

    Deferred inputs are not refusals. A creative field arriving over a link
    has no value yet, so such a writer is skipped here and judged by the
    writer's own evaluated check -- which still runs before any model work.
    ``resolve_policy`` is injected rather than imported so this module keeps
    its no-registry-import posture.
    """
    admitted: "list[AdmittedWriter]" = []
    for writer in writers_gated_by(prompt, unique_id):
        fields, deferred = read_queued_fields(writer)
        bank = writer.literal("source_bank", "")
        if is_link(bank):
            # The bank itself is computed upstream; nothing here can be judged.
            continue
        policy = resolve_policy(str(bank or ""))
        source_ref = writer.literal("source_ref", "")
        replay_from = writer.literal("replay_from", "")
        check_selection(
            fields, policy,
            source_ref="" if is_link(source_ref) else source_ref,
            replay_from="" if is_link(replay_from) else replay_from,
            snapshot_manifest_configured=snapshot_manifest_configured,
            creative_input_deferred=any(name in CREATIVE_FIELDS for name in deferred),
        )
        if not policy.is_user_fields:
            continue
        # Only persist a complete request. A linked field or control has no
        # value yet; substituting blanks/defaults creates a different draft.
        if deferred or any(is_link(writer.literal(name)) for name in (
                "source_ref", "replay_from", "num_characters", "act_count",
                "include_act_breaks", "visual_style")):
            continue
        request = StoryRequest(
            num_characters=_int_or(writer.literal("num_characters", 2), 2),
            act_count=_str_or(writer.literal("act_count", "")),
            include_act_breaks=_bool_or(
                writer.literal("include_act_breaks", True), True),
            source_bank_requested=str(bank or ""),
            visual_style_requested=_str_or(writer.literal("visual_style", "")),
        )
        admitted.append(AdmittedWriter(
            node_id=writer.node_id,
            bundle=build_bundle(fields, request),
        ))
    return admitted


def _int_or(value: Any, default: int) -> int:
    """A queued INT, or the default when it is linked or unusable."""
    if is_link(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str_or(value: Any) -> str:
    """A queued COMBO/STRING, or "" when it is linked.

    Never `_text`: a COMBO carries an int or a string depending on the widget,
    and a request field is descriptive rather than load-bearing.
    """
    return "" if is_link(value) or value is None else str(value)


def _bool_or(value: Any, default: bool) -> bool:
    if is_link(value) or value is None:
        return default
    return bool(value)


__all__ = [
    "ANONYMOUS_ATTRIBUTION",
    "AdmittedWriter",
    "check_queued_prompt",
    "ANONYMOUS_CREDIT",
    "CREATIVE_FIELDS",
    "DEDICATED_FIELDS",
    "FIELD_LABELS",
    "INPUT_MODE_LEGACY",
    "INPUT_MODE_USER_FIELDS",
    "KNOWN_INPUT_MODES",
    "QueuedWriter",
    "RawStoryFields",
    "STORY_INPUT_SCHEMA",
    "StoryInputBundle",
    "StoryInputError",
    "StoryInputPolicy",
    "StoryRequest",
    "attribution_receipt",
    "attribution_sentence",
    "build_bundle",
    "capture_raw",
    "check_selection",
    "compute_digest",
    "creative_input_present",
    "credits_source_line",
    "dedicated_fields_present",
    "filled_dedicated_fields",
    "first_creative_value",
    "is_link",
    "normalize",
    "project_payload",
    "read_queued_fields",
    "writers_gated_by",
]
