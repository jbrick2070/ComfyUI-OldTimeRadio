"""``ghost_signal_author`` -- the Ghost Prompt v2 AUTHORING surface.

Prompt v1 asked a deterministic composer to turn a beat into a picture, and on
an unmapped free-text ``beat_intent`` it did the only thing a regex can do: it
copied the first six words and prefixed ``moves with``. That is how a published
lane came to emit *"moves with erin risks exposure by transmitting a"* -- a cast
name in the picture, a sentence with no end, and no visual idea in it at all.

Prompt v2 replaces that slice with CONTROLLED ABSTRACTION. Exactly one short
drawable leaf is authored per beat by the already-selected writer model, and
Python owns everything else: the style cue, the negative, the recurrence motif,
the representation mode, the framing law, the identities, the hashes, the retry
and the fallback. The model is handed no dialogue, no title, no M4 scene wall,
no raw cast prose and no names -- not by policy but by CONSTRUCTION, because
none of those are parameters of the request this module builds.

WHAT LIVES HERE AND WHY IT IS NOT IN ``ghost_signal_prompt``. That module is
PURE by contract -- no I/O, no lazy loader, no tokenizer -- and it stays that
way. This module owns the two things that are not pure: the banana route (a
transform whose gate reads the environment) and the installed SD1 tokenizer (a
lazy import of ComfyUI's own encoder). It never loads an LLM: ShotLock owns
orchestration and hands the generate function in. It never imports the render
driver, the registry or the model loader, so there is no cycle.

THREE INVARIANTS THE REST OF THE LANE DEPENDS ON:

1. **The leaf is the only model-owned field.** A response that tries to set the
   mode, the motif, the style or an id is rejected whole. There is no partial
   salvage across attempts -- one invalid batch retries once, atomically, and a
   second failure receives a complete deterministic batch.
2. **The request hash is the replay identity.** Thirteen keys, compact sorted
   JSON, and the template digest among them -- so changing the wording of the
   prompt, its temperature or its output budget invalidates every stored leaf
   rather than silently replaying text authored under different instructions.
3. **A measured window, not an asserted one.** ComfyUI's SD1 encoder CHUNKS an
   overlength prompt rather than dropping it, so 77 is a salience choice, not a
   transport cliff. It is enforced by asking the installed tokenizer, never by
   a whitespace estimate -- and production fails closed when that tokenizer
   cannot be reached rather than guessing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Callable, Optional

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_LOG = logging.getLogger("OTR.video.ghost_signal_author")

try:  # pragma: no cover -- exercised by both import shapes in the suite
    from . import ghost_signal_prompt as _gsp
except ImportError:  # pragma: no cover -- flat test imports
    import ghost_signal_prompt as _gsp  # type: ignore


# --------------------------------------------------------------------------- #
# Identity. Every constant here is pinned by a test.
# --------------------------------------------------------------------------- #

#: The stored object's schema. Bumped only when the FIELD SET changes.
GHOST_AUTHOR_SCHEMA_VERSION = 1

#: The authoring contract's version. Bumped when the leaf's meaning changes;
#: it is one of the thirteen hashed request keys, so a bump reauthors.
GHOST_AUTHOR_VERSION = "ghost_drawable_beat_v1"

#: THE LEAF-ADMISSION CONTRACT, versioned separately from the author version.
#: It rides `template_sha256`, so tightening or loosening what a leaf may say
#: reauthors every stored object instead of silently replaying one admitted
#: under the old rules. Separate from `GHOST_AUTHOR_VERSION` on purpose: that
#: name also seeds the mode schedule and the fallback pools, so bumping it
#: would reshuffle pictures that did not need to change.
GHOST_VALIDATOR_CONTRACT = "leaf_admission_v2_concrete"

#: The three coordinated representations. ``figure`` is the only one that shows
#: a body, and it shows a body rather than a likeness -- no face is requested
#: and none is promised.
GHOST_MODES = ("figure", "object", "signal")

#: THE CHARACTER CYCLE, and it is deliberately NOT ``GHOST_MODES``.
#:
#: The first draft cycled the three modes evenly, so two of every three
#: character beats were non-figurative and an episode lost its people. Measured
#: against the v1 arm on the same seeds, that was the single biggest visible
#: regression -- the older arm rendered four clear human figures across its
#: eight beats and the new one rendered none.
#:
#: Period four, half of it ``figure``. That honours the operator's actual
#: directive -- *"do not force the same mediocre person into every clip"* --
#: without emptying the episode of people, and it still cannot produce a run of
#: three because no two adjacent entries are equal.
#:
#: **"HALF" IS THE CYCLE'S PROPERTY, NOT EVERY EPISODE'S, and saying otherwise
#: was a false invariant.** For a character count that is not a multiple of
#: four the realised share is a floor or a ceiling depending on the hashed
#: offset: three character beats can yield one figure, and a single character
#: beat can yield none. The guarantee is exactly `floor(n/2)` at worst, and the
#: test says so rather than asserting a half that does not hold.
GHOST_CHARACTER_CYCLE = ("figure", "object", "figure", "signal")

#: The two representations a bookend may take. A radio console is not a person.
GHOST_NON_FIGURE_MODES = ("object", "signal")

#: How a stored leaf came to exist. There is no ``fallback`` boolean: a
#: deliberate deterministic result is ``deterministic_fallback`` plus a nonempty
#: reason, and reuse of one keeps BOTH -- so a replay can never launder a
#: fallback into proof eligibility.
GHOST_AUTHOR_SOURCES = ("writer_llm", "replay", "deterministic_fallback")

#: Leaf shape. The model is TOLD 8--12 words (the ASK constants below, which
#: are what the generated rule text actually says); the hard bounds are wider
#: so a good answer one word outside the request is not thrown away. The
#: comment claimed 6--10 until 2026-08-28 -- a band that matched neither the
#: constants nor the prompt.
GHOST_LEAF_MAX_CHARS = 96
GHOST_LEAF_MIN_WORDS = 5
GHOST_LEAF_MAX_WORDS = 14
GHOST_LEAF_ASK_WORDS_LOW = 8
GHOST_LEAF_ASK_WORDS_HIGH = 12

#: One CLIP window, measured with the installed tokenizer including BOS/EOS.
GHOST_CLIP_WINDOW_TOKENS = 77

#: What an AUTHOR-TIME candidate targets. Below the window on purpose: it
#: preserves the measured v1 worst case (63--69 tokens across all nine shipped
#: packs and all eight live shots) as real headroom for the banana route.
GHOST_AUTHOR_TOKEN_TARGET = 69

#: THE MODEL ID STAMPED WHEN NO MODEL IS CONFIGURED AT ALL (a unit fixture or
#: a legacy local path). It is a real value rather than an empty string because
#: ``model_id`` is one of the thirteen hashed request keys and one of the
#: required stored fields -- and because a later live run with a real model
#: therefore hashes differently and REAUTHORS, which is exactly right: a
#: checked-in clause and a written one are not the same artifact.
#:
#: A CONFIGURED model that was asked and failed keeps its OWN id, so a receipt
#: still names the model whose answer was rejected.
GHOST_DETERMINISTIC_MODEL_ID = "deterministic"

#: The generation contract. These three values are hashed into
#: :data:`GHOST_TEMPLATE_SHA256`, so changing any of them reauthors.
GHOST_BATCH_TEMPERATURE = 0.1

#: THE RETRY TEMPERATURE, and it exists because the first retry could not
#: possibly work. Attempt 2 re-sent a byte-identical prompt at 0.1 -- near
#: greedy -- so a model that wrote a four-word leaf once wrote it again, and
#: the batch fell to deterministic clauses having spent two generations to
#: learn nothing. Attempt 2 now carries the rejection reasons AND samples
#: warmer, so it is a different question rather than the same one asked twice.
GHOST_BATCH_RETRY_TEMPERATURE = 0.45

GHOST_BATCH_BASE_TOKENS = 64
GHOST_BATCH_PER_SHOT_TOKENS = 48


def batch_output_tokens(spec_count) -> int:
    """The output budget for a batch of ``spec_count`` leaves."""
    return int(GHOST_BATCH_BASE_TOKENS
               + GHOST_BATCH_PER_SHOT_TOKENS * int(spec_count))


class GhostAuthorError(ValueError):
    """Base class for every refusal this module raises."""


class GhostAuthorParseError(GhostAuthorError):
    """The model's batch response is not the one accepted envelope."""


class GhostAuthorValidationError(GhostAuthorError):
    """A stored Ghost prompt object is absent, malformed or inconsistent."""


class GhostTokenizerUnavailable(GhostAuthorError):
    """The installed SD1 tokenizer could not be reached in production."""


class GhostBudgetError(GhostAuthorError):
    """A composed prompt exceeds a ceiling that is never trimmed to fit."""


# --------------------------------------------------------------------------- #
# The batch template. Its exact bytes are hashed, so edit it and every stored
# leaf reauthors -- which is the point: a leaf written under other instructions
# is not the same leaf.
# --------------------------------------------------------------------------- #

#: THE WORKED EXAMPLES EARN THEIR PLACE, and the live leg is why (2026-08-22).
#: With the rule text alone, Mistral-Nemo answered in a perfectly valid envelope
#: carrying four-word abstractions -- "signal oscillates, broadcast begins",
#: "silhouette shreds papers, tension builds" -- which the validator rejected on
#: both attempts, so the whole batch fell to the deterministic pools. A word
#: COUNT is a number a model does not feel; a sentence AT the target length is
#: one it can match. The counter-examples are named too, because "tension
#: builds" is not a thing a picture can show and saying so beats hoping.
#:
#: The example objects sit deliberately OUTSIDE the motif allowlists (a shutter,
#: a gauge and a spool are not in ``MOTIF_PROP_WORDS``), so a model that copies
#: one is visibly copying rather than producing something that could pass for a
#: real recurrence motif.
GHOST_BATCH_EXAMPLES = (
    "GOOD, and match this length:\n"
    "  mode=object   -> the shutter tilts on the desk as a shadow crosses "
    "its slats\n"
    "  mode=signal   -> a lamp swings past the crate and the shadow sweeps "
    "the wall\n"
    "  mode=figure   -> a figure lifts the spool and holds it against the "
    "window light\n"
    "BAD, never write these:\n"
    "  tension builds            (not a thing a picture can show)\n"
    "  finality sets in          (not a thing a picture can show)\n"
    "  broadcast begins          (too short, and nothing is drawn)\n"
    "  banded static tightens    (a texture is not a subject)\n"
    "  a waveform pulses         (a graph is not a subject)\n"
)

GHOST_BATCH_RULES = (
    "You write ONE short drawable visual for each shot in a silent, "
    "low-resolution animated film. Each visual must name something a "
    "picture can actually show plus ONE visible change or movement.\n"
    "RULES:\n"
    "1. Answer with JSON only, exactly this shape and nothing else:\n"
    '   {"shots": [{"id": "g000", "drawable_beat": "..."}]}\n'
    "2. One entry per shot id given below, same ids, no extras, no repeats.\n"
    "3. Each drawable_beat is %d-%d words -- count them -- lower case, and "
    "never longer than %d characters. Fewer than %d words is rejected.\n"
    "4. Name a CONCRETE thing and what it does -- an object you could pick "
    "up, or a person. A mood, a feeling, an abstract state, a texture, a "
    "pattern or a waveform is NOT drawable and will be rejected.\n"
    "5. Write only what is visible. No names, no speech, no captions, no "
    "lettering, no camera or lens words, no quality words.\n"
    "6. Never write what is absent. Say what IS in the frame.\n"
    "7. mode=figure SHOWS A PERSON (no face needed). mode=object shows one "
    "real object on a real surface, no person. mode=signal shows a real "
    "object in a dark room with light moving over it, no person.\n"
    "8. Do not repeat the motif text back; it is already in the picture, and "
    "do not reuse the example objects below -- write from the motif you were "
    "given.\n"
    "%s"
) % (GHOST_LEAF_ASK_WORDS_LOW, GHOST_LEAF_ASK_WORDS_HIGH, GHOST_LEAF_MAX_CHARS,
     GHOST_LEAF_MIN_WORDS, GHOST_BATCH_EXAMPLES)

GHOST_BATCH_HEADER = "SHOTS:"

GHOST_BATCH_ENVELOPE = '{"shots": [{"id": "g000", "drawable_beat": "..."}]}'


def _template_identity() -> str:
    """SHA-256 over the exact generation contract, not just the prose.

    Includes the temperature and the output-budget FORMULA, because a leaf
    authored at a different temperature or under a different budget is a
    different artifact even when the words of the instruction are identical.
    """
    payload = json.dumps({
        "author_version": GHOST_AUTHOR_VERSION,
        "envelope": GHOST_BATCH_ENVELOPE,
        "examples": GHOST_BATCH_EXAMPLES,
        "header": GHOST_BATCH_HEADER,
        "output_tokens_base": GHOST_BATCH_BASE_TOKENS,
        "output_tokens_per_shot": GHOST_BATCH_PER_SHOT_TOKENS,
        "rules": GHOST_BATCH_RULES,
        "schema_version": GHOST_AUTHOR_SCHEMA_VERSION,
        "temperature": GHOST_BATCH_TEMPERATURE,
        "retry_temperature": GHOST_BATCH_RETRY_TEMPERATURE,
        # THE VALIDATOR CONTRACT IS PART OF THE GENERATION CONTRACT. v2.1
        # changed what a leaf is ALLOWED to say without changing any hashed
        # key, so a leaf admitted under the old permissive rules replayed
        # untouched under the new ones -- replay does not re-run
        # validate_drawable_beat. Bumping this invalidates those hashes.
        "validator_contract": GHOST_VALIDATOR_CONTRACT,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


GHOST_TEMPLATE_SHA256 = _template_identity()


# --------------------------------------------------------------------------- #
# Safe projection. Everything the model is allowed to see passes through here.
# --------------------------------------------------------------------------- #

#: Roles this lane authors for. Anything else is not a Ghost beat.
GHOST_ROLES = ("character_video", "announcer_visual", "music_visual")

#: Arc phases worth telling the model about. ``scene`` is the default value on
#: nearly every line and carries no visual information, so it is dropped rather
#: than spent -- v1 emitted it into the picture as a literal word.
GHOST_ARC_CUES = {
    "setup": "opening",
    "rising": "building",
    "turn": "turning",
    "climax": "peak",
    "falling": "easing",
    "resolution": "closing",
    "coda": "closing",
}

#: Field-label debris. A cast row pasted into an intent brings its own labels,
#: and ``Face:`` in a prompt is a request for the word, not the thing.
_FIELD_LABEL_RE = re.compile(
    r"\b(face|appearance|description|wardrobe|costume|prop|voice|gender|age|"
    r"name|role|traits|beat|intent|arc|scene|shot|prompt)\s*:", re.IGNORECASE)

_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)

#: Words that make a leaf a lettering instruction rather than a picture.
#: `letter` and `letters` are DELIBERATELY ABSENT. `letter` is in
#: :data:`MOTIF_PROP_WORDS`, so "a charcoal letter" is a motif this module can
#: legally emit -- and a leaf saying "the letter slides across the desk" was
#: then rejected as a lettering request, killing the whole batch over a prop
#: the code itself had chosen. `lettering`, `caption`, `text` and `typography`
#: still cover the real defect.
_LETTERING_WORDS = frozenset({
    "text", "texts", "word", "words", "lettering",
    "caption", "captions", "subtitle", "subtitles", "title", "titles",
    "typography", "font", "fonts", "writing", "written", "inscription",
    "label", "labels", "signage", "logo", "headline",
})

#: Camera / medium / quality boilerplate. Style is Python's job on this lane.
_BOILERPLATE_WORDS = frozenset({
    "camera", "lens", "zoom", "pan", "dolly", "tilt", "crop", "closeup",
    "close-up", "bokeh", "cinematic", "photorealistic", "photoreal",
    "masterpiece", "hdr", "8k", "4k", "render", "rendered", "rendering",
    "octane", "unreal", "artstation", "trending", "award-winning",
    "highly", "ultra", "detailed", "quality", "resolution", "style",
})

#: Human tokens forbidden in ``object`` and ``signal`` leaves. Absence of a
#: person is expressed by choosing a non-human subject, never by asking for it.
#:
#: BODY PARTS ARE NOT ON THIS LIST, and a live leg is why (2026-08-22). It used
#: to carry hand/hands/arm/arms/shoulder/shoulders, and it rejected
#: *"the silver ledger sits on a desk as a clock hand ticks"* -- a CLOCK hand --
#: which killed the whole batch and dropped the episode to deterministic
#: clauses. It also rejected *"the radio dial turns as a hand adjusts the
#: knob"*, which is the archival-documentary look the operator ranked SECOND
#: out of five arms.
#:
#: What these modes owe is that no FULL FIGURE dominates the shot. They do not
#: owe a frame with no people in it anywhere, and a word list that cannot tell
#: a clock hand from a person is a list that costs a live batch to learn that.
_HUMAN_WORDS = frozenset({
    "person", "people", "human", "humans", "man", "men", "woman", "women",
    "boy", "girl", "child", "children", "crowd", "figure", "figures",
    "silhouette", "silhouettes", "portrait", "someone", "somebody",
    "stranger", "lady", "gentleman", "guy",
    # PRONOUNS BELONG HERE and were dropped by mistake when the body parts
    # went. "he turns the dial slowly in darkness" is a person request and was
    # passing object mode.
    "he", "she", "they", "him", "her", "them",
})

#: BARE `face` IS NOT ON THAT LIST, and that is the clock-hand lesson applied a
#: second time. This show's bookend vocabulary is DIALS -- `a glowing radio
#: dial`, `a bakelite radio set` -- and v1's own framing constant read "dial
#: face centered". A whole-word match on `face` rejects "the dial face
#: brightens" exactly as `hand` rejected "a clock hand ticks", and one bad leaf
#: kills the batch. `portrait` still guards the actual face request.

#: Words whose SUBJECT is a texture rather than a thing. A 512x288 SD1.5 draws
#: exactly what it is handed: asked for static it paints static, asked for a
#: lantern it paints a lantern. The first live v2 arms are the measurement --
#: every beat whose leaf named one of these rendered as unreadable texture, and
#: the only beats that survived were the ones naming a real object.
#:
#: NOT a taste judgement. A leaf may still describe light, shadow or movement;
#: it may not make the ABSENCE OF A SUBJECT its subject.
#: `grain`, `noise`, `geometry` and `geometric` were REMOVED after the panel
#: read them: a sack of grain, wood grain on a desk and a sudden noise are all
#: concrete, and whole-word rejection killed the batch for them. What stays is
#: the vocabulary actually measured painting mush -- static, waveforms,
#: gradients, raw texture and pixels -- plus `emblem` and `field`, which the
#: receipt named as mush-makers and which nothing had been banning.
_ABSTRACT_SUBJECT_WORDS = frozenset({
    "static", "waveform", "waveforms", "gradient", "gradients", "texture",
    "textures", "abstraction", "abstract", "pixels", "pixelation",
    "scanlines", "interference", "emblem", "emblems",
})

#: Positive-channel negation. The negative prompt is the exclusion authority;
#: a positive clause that attends to an absent thing summons it instead.
_NEGATION_RE = re.compile(
    r"\b(no|not|without|never|absent|empty of|devoid)\b", re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_WS_RE = re.compile(r"\s+")


def _norm(text) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def normalize_role(role) -> str:
    """The role token, or ``""`` for anything this lane does not author."""
    role = _norm(role)
    return role if role in GHOST_ROLES else ""


def strip_cast_names(text, names) -> str:
    """Remove every known cast name -- whole and by part -- from ``text``.

    Parts as well as wholes because a two-word name is referred to by either
    half in ordinary intent prose, and a first name in a prompt is still a
    proper noun the model will try to draw. Tokens shorter than three
    characters are skipped: an initial is not a name leak, and stripping ``a``
    would destroy the sentence.
    """
    out = str(text or "")
    tokens = []
    for name in (names or ()):
        name = _norm(name)
        if not name:
            continue
        tokens.append(name)
        tokens.extend(name.split())
    # LONGEST FIRST. Removing "Adrian" before "Adrian Spender" would leave a
    # bare surname behind -- still a proper noun, still a leak.
    for token in sorted({t for t in tokens if len(t) >= 3}, key=len,
                        reverse=True):
        out = re.sub(r"\b%s\b" % re.escape(token), " ", out,
                     flags=re.IGNORECASE)
    return _norm(out)


def sanitize_intent(beat_intent, names=()) -> str:
    """The one free-text field the model sees, made safe and lower case.

    Names removed, field labels removed, second person removed, punctuation
    normalized, and the result bounded. It is a HINT about what happens in the
    beat, never a sentence to copy -- v1's whole defect was copying it.
    """
    text = strip_cast_names(beat_intent, names)
    text = _FIELD_LABEL_RE.sub(" ", text)
    text = _SECOND_PERSON_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'["`]+', " ", text)
    text = _norm(text).lower().strip(" ,;:.-")
    words = text.split()
    if len(words) > 24:
        text = " ".join(words[:24])
    return _norm(text)


def normalize_emotion(traits) -> str:
    """The beat's recorded mood as one short lower-case phrase, or ``""``."""
    text = _norm(traits).lower().strip(" ,;:.-")
    if not text:
        return ""
    text = text.split(",")[0].strip()
    words = text.split()
    return " ".join(words[:3])


def map_arc(arc_phase) -> str:
    """A mapped arc cue, or ``""`` when the value carries no visual meaning."""
    return GHOST_ARC_CUES.get(_norm(arc_phase).lower(), "")


# --------------------------------------------------------------------------- #
# Deterministic mode scheduling.
# --------------------------------------------------------------------------- #

def _hash_int(*parts) -> int:
    payload = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def schedule_ghost_modes(entries, episode_seed) -> dict:
    """``{beat_id: mode}`` for one episode's whole Ghost timeline.

    ``entries`` is the ORDERED list of ``(beat_id, role)`` pairs -- ordered
    because the anti-run rule is a property of the timeline, not of a beat.

    CHARACTER BEATS RUN :data:`GHOST_CHARACTER_CYCLE` from a hashed offset --
    ``figure, object, figure, signal``. No two adjacent entries are equal, so a
    character can never start a run and never needs correcting.

    THE GUARANTEE IS ``floor(n/2)`` FIGURES, NOT "half", and the difference is
    not pedantry. The cycle has period four, so for a character count that is
    not a multiple of four the realised share depends on the hashed offset: one
    character beat can yield no figure at all, and three can yield one. Half is
    what the CYCLE contains; floor-half is what an EPISODE is promised.

    BOOKENDS alternate ``object``/``signal`` from their own hashed offset. Only
    a bookend may be corrected, and only when it would create a THIRD identical
    mode in a row -- which keeps clip planning stable and keeps the correction
    away from the assignments the recurrence design depends on.
    """
    seed = _hash_int(episode_seed, "ghost_mode_schedule", GHOST_AUTHOR_VERSION)
    char_offset = seed % len(GHOST_CHARACTER_CYCLE)
    bookend_offset = (seed >> 8) % len(GHOST_NON_FIGURE_MODES)

    out = {}
    scheduled = []
    char_n = 0
    bookend_n = 0
    for beat_id, role in entries:
        beat_id = str(beat_id)
        if normalize_role(role) == "character_video":
            mode = GHOST_CHARACTER_CYCLE[
                (char_offset + char_n) % len(GHOST_CHARACTER_CYCLE)]
            char_n += 1
            fixed = True
        else:
            mode = GHOST_NON_FIGURE_MODES[
                (bookend_offset + bookend_n) % len(GHOST_NON_FIGURE_MODES)]
            bookend_n += 1
            fixed = False
        if (not fixed and len(scheduled) >= 2
                and scheduled[-1] == mode and scheduled[-2] == mode):
            # The other non-figure representation, deterministically.
            mode = GHOST_NON_FIGURE_MODES[
                (GHOST_NON_FIGURE_MODES.index(mode) + 1)
                % len(GHOST_NON_FIGURE_MODES)]
        scheduled.append(mode)
        out[beat_id] = mode
    return out


# --------------------------------------------------------------------------- #
# The recurrence motif -- a compact, allowlisted, NON-FACE identity.
# --------------------------------------------------------------------------- #

#: Colour tokens, split out of the composer's costume bucket. Colour is the one
#: property that survives 512x288 across a sliding context, so it anchors all
#: three representations of one character.
MOTIF_COLOUR_WORDS = (
    "black", "white", "red", "green", "blue", "brown", "grey", "gray",
    "crimson", "scarlet", "navy", "olive", "amber", "gold", "golden",
    "silver", "rust", "ochre", "violet", "purple", "cream", "tan", "charcoal",
)

#: Garment tokens. ``figure`` carries one so the person on screen is wearing
#: something a sampler can draw; the non-figure modes have no use for a coat.
MOTIF_GARMENT_WORDS = (
    "coat", "overcoat", "trenchcoat", "greatcoat", "jacket", "uniform",
    "cloak", "cape", "gown", "dress", "suit", "waistcoat", "vest", "apron",
    "shawl", "robe", "hat", "fedora", "cap", "helmet", "hood", "scarf",
)

#: Silhouette tokens. ``figure`` alone carries one.
MOTIF_SILHOUETTE_WORDS = (
    "tall", "short", "broad", "slight", "stooped", "lean", "heavyset",
    "wiry", "angular", "round", "towering", "compact", "gaunt", "stocky",
    "willowy", "slender", "squat", "burly", "spare", "rangy", "narrow",
)

#: Prop tokens, carried by all three representations of one character.
MOTIF_PROP_WORDS = (
    "lantern", "lamp", "torch", "candle", "revolver", "pistol", "rifle",
    "knife", "blade", "sword", "dagger", "book", "ledger", "journal",
    "letter", "map", "chart", "cane", "staff", "umbrella", "satchel",
    "case", "briefcase", "pipe", "flask", "bottle", "glass", "cup", "key",
    "rope", "chain", "hammer", "wrench", "microphone", "headset",
    "telegraph", "camera", "watch", "spyglass", "telescope", "basket",
    "crate", "lockbox",
)

#: Deterministic fills, used when a cast row supplied no allowlisted token of
#: that kind. Checked in, never generated, never asked of a model.
MOTIF_FALLBACK_POOLS = {
    "colour": ("amber", "rust", "olive", "charcoal", "cream", "navy"),
    "garment": ("coat", "uniform", "shawl", "jacket", "hood", "scarf"),
    "silhouette": ("lean", "broad", "slight", "tall", "compact", "angular"),
    "prop": ("lantern", "key", "ledger", "satchel", "chart", "telegraph"),
}

#: Bookend motifs, by ``(role, mode)``. Real radio HARDWARE that recurs while
#: the unchanged pack cue keeps supplying the anime / archive / material look.
#: A Ghost bookend with an empty motif is invalid, which is why these are
#: constants rather than an optional derivation.
#:
#: They used to read "radio dial emblem" and "broadcast waveform signal". An
#: emblem is not a thing and a waveform is a graph, and the bookends rendered as
#: texture accordingly. A bakelite radio set is a thing.
GHOST_BOOKEND_MOTIFS = {
    ("announcer_visual", "object"): "a bakelite radio set",
    ("announcer_visual", "signal"): "a glowing radio dial",
    ("music_visual", "object"): "a broadcast console",
    ("music_visual", "signal"): "a spinning turntable",
}


def _first_allowlisted(phrases, vocabulary) -> str:
    """The first allowlisted WORD found across ``phrases``, in source order.

    A WORD, never the phrase it came from: copying the phrase is how a cast
    row's prose -- and its landmarks, its hair, its jaw -- would get back into
    a prompt that is supposed to have left all of that behind.
    """
    allowed = frozenset(vocabulary)
    # SOURCE ORDER, WHICH IS WHAT THE DOCSTRING PROMISED. The loop used to walk
    # the VOCABULARY and ask whether the phrase contained each entry, so a
    # phrase naming two allowed colours returned whichever came first in the
    # checked-in tuple rather than the one the cast row led with.
    for phrase in phrases:
        for word in _WORD_RE.findall(str(phrase or "").lower()):
            if word in allowed:
                return word
    return ""


def _motif_tokens(components, seed_int) -> dict:
    """The four allowlisted motif tokens for one character.

    ``components`` is ``ghost_signal_prompt.distill_sigil_components``'s bucket
    map. The LANDMARK bucket is deliberately not read: an asymmetrical facial
    landmark is exactly the face-adjacent recurrence Prompt v2 replaces.
    """
    components = components if isinstance(components, dict) else {}
    costume = components.get("costume") or ""
    prop = components.get("prop") or ""
    silhouette = components.get("silhouette") or ""

    # COLOUR AND GARMENT ARE SCANNED ACROSS BOTH NON-FACE BUCKETS, in costume-
    # first order. The sigil's buckets are GREEDY -- one phrase is consumed by
    # the first bucket that matches it -- so a cast row reading "a broad steady
    # man in a charcoal overcoat" lands whole in the SILHOUETTE bucket and the
    # costume bucket then falls to its neutral pool. Reading only the costume
    # bucket would have thrown away the character's real colour and worn a
    # pooled one instead, which is the opposite of a recurrence cue.
    #
    # The LANDMARK bucket is never scanned. That is where the jaw, the brow,
    # the scar and the hair live, and face-adjacent recurrence is exactly what
    # Prompt v2 exists to replace.
    out = {}
    out["colour"] = _first_allowlisted((costume, silhouette),
                                       MOTIF_COLOUR_WORDS)
    out["garment"] = _first_allowlisted((costume, silhouette),
                                        MOTIF_GARMENT_WORDS)
    out["silhouette"] = _first_allowlisted((silhouette,),
                                           MOTIF_SILHOUETTE_WORDS)
    out["prop"] = _first_allowlisted((prop,), MOTIF_PROP_WORDS)
    for kind, pool in MOTIF_FALLBACK_POOLS.items():
        if not out.get(kind):
            out[kind] = pool[_hash_int(seed_int, "motif", kind) % len(pool)]
    return out


def motif_for_character(components, mode, *, seed_int=0) -> str:
    """The compact recurrence motif for one character in one representation.

    All three representations share COLOUR + PROP, which is what makes the
    episode read as related; ``figure`` additionally carries the silhouette,
    which is the only body property this model holds across a sliding context.
    """
    mode = str(mode or "")
    tokens = _motif_tokens(components, seed_int)
    # "an olive key", not "a olive key". The motif is prompt text a person reads
    # in a receipt and a sampler reads as language; a broken article is a small
    # thing that makes both of them worse.
    article = "an" if tokens["colour"][:1] in "aeiou" else "a"
    if mode == "figure":
        # A PERSON, SAID PLAINLY. The first draft said "<silhouette> silhouette
        # with a <colour> <prop>", and SD1.5 drew vertical black shapes -- it
        # does not know that a "silhouette" is supposed to be someone. v1 said
        # "a man, a broad steady figure, a charcoal coat, holding a folded
        # chart" and got a man in a coat. This is that, minus the name leak.
        sil_article = "an" if tokens["silhouette"][:1] in "aeiou" else "a"
        return "%s %s figure in %s %s %s, carrying %s %s" % (
            sil_article, tokens["silhouette"], article, tokens["colour"],
            tokens["garment"],
            "an" if tokens["prop"][:1] in "aeiou" else "a", tokens["prop"])
    if mode in ("object", "signal"):
        # THE PROP AS ITSELF. "charcoal lantern emblem" is not a thing; a
        # charcoal lantern is. The two beats that survived the first draft are
        # the two whose leaf named a real object, which is the whole finding.
        return "%s %s %s" % (article, tokens["colour"], tokens["prop"])
    raise GhostAuthorError("unknown Ghost representation mode %r" % (mode,))


def motif_for_bookend(role, mode) -> str:
    """The checked-in bookend motif. Missing pair is a constant defect."""
    key = (normalize_role(role), str(mode or ""))
    motif = GHOST_BOOKEND_MOTIFS.get(key)
    if not motif:
        raise GhostAuthorError(
            "no Ghost bookend motif for role/mode %r -- an empty bookend "
            "motif is invalid, so this is a composer constant defect rather "
            "than a beat to fall back on" % (key,))
    return motif


# --------------------------------------------------------------------------- #
# Specs -- what ShotLock projects and what the model actually receives.
# --------------------------------------------------------------------------- #

def opaque_id(ordinal) -> str:
    """``g000``, ``g001``, ... -- the model never sees a ledger identifier."""
    return "g%03d" % int(ordinal)


def build_ghost_author_specs(rows, *, model_id) -> list:
    """Project ShotLock rows into the ordered, safe author specs.

    Each ``row`` is a dict with ``beat_id``, ``role``, ``mode``, ``motif_cue``,
    ``sanitized_intent``, ``normalized_emotion`` and ``mapped_arc`` -- all of
    them already safe. This function adds the opaque id, the ordinal and the
    request hash, and refuses anything shaped wrong.
    """
    specs = []
    for ordinal, row in enumerate(rows or ()):
        role = normalize_role((row or {}).get("role"))
        if not role:
            raise GhostAuthorError(
                "Ghost author row %d has role %r, which this lane does not "
                "author" % (ordinal, (row or {}).get("role")))
        mode = str((row or {}).get("mode") or "")
        if mode not in GHOST_MODES:
            raise GhostAuthorError(
                "Ghost author row %d has mode %r; expected one of %s"
                % (ordinal, mode, ", ".join(GHOST_MODES)))
        if role != "character_video" and mode == "figure":
            raise GhostAuthorError(
                "Ghost author row %d is a %s bookend scheduled as figure -- a "
                "radio console is not a person" % (ordinal, role))
        motif_cue = _norm(row.get("motif_cue"))
        if not motif_cue:
            raise GhostAuthorError(
                "Ghost author row %d has an empty motif_cue; the recurrence "
                "cue is the identity mechanism and may not be absent"
                % (ordinal,))
        beat_id = _norm(row.get("beat_id"))
        if not beat_id:
            raise GhostAuthorError(
                "Ghost author row %d has no beat_id" % (ordinal,))
        spec = {
            "author_version": GHOST_AUTHOR_VERSION,
            "beat_id": beat_id,
            "mapped_arc": _norm(row.get("mapped_arc")),
            "mode": mode,
            "model_id": _norm(model_id),
            "motif_cue": motif_cue,
            "motif_sha256": hashlib.sha256(
                motif_cue.encode("utf-8")).hexdigest(),
            "normalized_emotion": _norm(row.get("normalized_emotion")),
            "ordinal": int(ordinal),
            "role": role,
            "sanitized_intent": _norm(row.get("sanitized_intent")),
            "schema_version": GHOST_AUTHOR_SCHEMA_VERSION,
            "template_sha256": GHOST_TEMPLATE_SHA256,
        }
        spec["request_sha256"] = request_sha256(spec)
        spec["id"] = opaque_id(ordinal)
        specs.append(spec)
    return specs


#: THE THIRTEEN HASHED KEYS, and there is no second spelling of this set.
GHOST_REQUEST_HASH_KEYS = (
    "author_version", "beat_id", "mapped_arc", "mode", "model_id",
    "motif_cue", "motif_sha256", "normalized_emotion", "ordinal", "role",
    "sanitized_intent", "schema_version", "template_sha256",
)


def request_sha256(spec) -> str:
    """SHA-256 over compact, sorted, canonical JSON of the thirteen keys.

    ``beat_id`` is the CANONICAL ledger identity (``b000_music_open`` included).
    Neither the cast-time temporary ``shot_id=beat_id`` nor the durable
    ``shot_id="shot_"+beat_id`` is hashed -- they differ on purpose, and hashing
    either would make the temporary and durable objects disagree.
    """
    payload = {key: spec.get(key) for key in GHOST_REQUEST_HASH_KEYS}
    missing = [k for k, v in payload.items() if v is None]
    if missing:
        raise GhostAuthorError(
            "Ghost request hash is missing key(s) %s" % ", ".join(missing))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def output_sha256(drawable_beat) -> str:
    """The accepted leaf's own bytes -- never the disposition-bearing wrapper.

    Hashing the wrapper would make ``source`` part of the digest, so the same
    text replayed under ``source=replay`` would hash differently from the text
    that was authored. The leaf is the artifact; the disposition is a receipt.
    """
    return hashlib.sha256(str(drawable_beat or "").encode("utf-8")).hexdigest()


def build_batch_prompt(specs) -> str:
    """The single user message for one whole episode's Ghost batch."""
    lines = [GHOST_BATCH_RULES, GHOST_BATCH_HEADER]
    for spec in specs:
        parts = ["id=%s" % spec["id"], "mode=%s" % spec["mode"],
                 "motif=%s" % spec["motif_cue"]]
        if spec.get("sanitized_intent"):
            parts.append("happening=%s" % spec["sanitized_intent"])
        if spec.get("normalized_emotion"):
            parts.append("mood=%s" % spec["normalized_emotion"])
        if spec.get("mapped_arc"):
            parts.append("story=%s" % spec["mapped_arc"])
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Strict parsing. One envelope, no salvage.
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.DOTALL)


def _strip_one_fence(raw) -> str:
    """Remove EXACTLY one enclosing markdown fence -- transport, not content.

    One, and only one: a response wrapped twice, or with prose either side of
    the fence, is a response that did not follow the instruction, and pretending
    otherwise is how a parser starts accepting anything.
    """
    text = str(raw or "").strip()
    match = _FENCE_RE.match(text)
    return match.group("body").strip() if match else text


def _object_pairs_hook(pairs):
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise GhostAuthorParseError(
                "Ghost batch response repeats the key %r" % (key,))
        seen.add(key)
    return dict(pairs)


def parse_batch_response(raw, expected_ids) -> dict:
    """``{opaque_id: drawable_beat}``, or raise.

    ACCEPTS EXACTLY ONE SHAPE:
    ``{"shots": [{"id": "g000", "drawable_beat": "..."}, ...]}`` -- every
    expected id present exactly once, no unknown id, no extra field on the
    envelope or on a row, no trailing object after the JSON.
    """
    expected = list(expected_ids or ())
    body = _strip_one_fence(raw)
    if not body:
        raise GhostAuthorParseError("Ghost batch response is empty")
    decoder = json.JSONDecoder(object_pairs_hook=_object_pairs_hook)
    try:
        payload, end = decoder.raw_decode(body)
    except GhostAuthorParseError:
        raise
    except ValueError as exc:
        raise GhostAuthorParseError(
            "Ghost batch response is not JSON: %s" % (exc,)) from exc
    if body[end:].strip():
        raise GhostAuthorParseError(
            "Ghost batch response carries trailing content after the JSON "
            "object -- prose and second objects are not transport wrapping")
    if not isinstance(payload, dict):
        raise GhostAuthorParseError(
            "Ghost batch response is a %s, not an object"
            % type(payload).__name__)
    if set(payload) != {"shots"}:
        raise GhostAuthorParseError(
            "Ghost batch envelope must carry exactly one key 'shots'; got %s"
            % (", ".join(sorted(payload)) or "nothing"))
    rows = payload.get("shots")
    if not isinstance(rows, list):
        raise GhostAuthorParseError("Ghost batch 'shots' is not a list")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GhostAuthorParseError(
                "Ghost batch row is a %s, not an object" % type(row).__name__)
        if set(row) != {"id", "drawable_beat"}:
            raise GhostAuthorParseError(
                "Ghost batch row must carry exactly id + drawable_beat; got %s"
                % ", ".join(sorted(row)))
        shot_id = row.get("id")
        leaf = row.get("drawable_beat")
        if not isinstance(shot_id, str) or not isinstance(leaf, str):
            raise GhostAuthorParseError(
                "Ghost batch row id/drawable_beat must both be strings")
        if shot_id not in expected:
            raise GhostAuthorParseError(
                "Ghost batch row carries unknown id %r" % (shot_id,))
        if shot_id in out:
            raise GhostAuthorParseError(
                "Ghost batch repeats id %r" % (shot_id,))
        out[shot_id] = _norm(leaf)
    missing = [i for i in expected if i not in out]
    if missing:
        raise GhostAuthorParseError(
            "Ghost batch response is missing id(s) %s" % ", ".join(missing))
    return out


# --------------------------------------------------------------------------- #
# Leaf validation. SHAPE AND BOUNDARY SAFETY -- never a story-vocabulary judge.
# --------------------------------------------------------------------------- #

def validate_drawable_beat(leaf, *, mode, names=()) -> tuple:
    """``(ok, reason)`` for one authored leaf.

    This is deliberately NOT a quality gate. It checks that the leaf is the
    right shape, that nothing crossed the boundary that was not supposed to
    (a name, a second person, a field label), and that it did not answer a
    question Python already answered (lettering, camera, style, a negation, a
    person in a non-figure mode). Whether the picture is any good is the
    operator's eye, not a word list.
    """
    text = _norm(leaf)
    if not text:
        return False, "empty"
    if "\n" in str(leaf) or "\r" in str(leaf):
        return False, "line break"
    if len(text) > GHOST_LEAF_MAX_CHARS:
        return False, "over %d characters" % GHOST_LEAF_MAX_CHARS
    if ":" in text:
        return False, "carries a field label"
    words = text.split()
    if len(words) < GHOST_LEAF_MIN_WORDS:
        return False, "under %d words" % GHOST_LEAF_MIN_WORDS
    if len(words) > GHOST_LEAF_MAX_WORDS:
        return False, "over %d words" % GHOST_LEAF_MAX_WORDS
    lowered = set(_WORD_RE.findall(text.lower()))
    if lowered & _LETTERING_WORDS:
        return False, "asks for lettering"
    if lowered & _BOILERPLATE_WORDS:
        return False, "carries camera/style boilerplate"
    if lowered & _ABSTRACT_SUBJECT_WORDS:
        return False, "names a texture instead of a thing"
    if _SECOND_PERSON_RE.search(text):
        return False, "addresses a second person"
    if _NEGATION_RE.search(text):
        return False, "negates in the positive channel"
    stripped = strip_cast_names(text, names)
    if stripped.lower() != text.lower():
        return False, "carries a cast name"
    if str(mode) in GHOST_NON_FIGURE_MODES and (lowered & _HUMAN_WORDS):
        return False, "requests a person in %s mode" % (mode,)
    tail = words[-1].strip(",.;:-").lower()
    if tail in _gsp._DANGLING_TAIL_WORDS:
        return False, "ends on a dangling function word"
    return True, ""


# --------------------------------------------------------------------------- #
# The deterministic batch. Complete clauses, never a free-text slice.
# --------------------------------------------------------------------------- #

#: Complete, checked-in, mode-specific clauses. Every one of them is a whole
#: drawable idea -- which is precisely what the v1 six-word slice was not.
GHOST_FALLBACK_CLAUSES = {
    "figure": (
        "a figure turns slowly toward a lit doorway",
        "a figure steps forward across a bare wooden floor",
        "a figure lifts one hand and holds it against the light",
        "a figure leans over a table and goes still",
        "a figure straightens and looks off past the frame",
        "a figure turns away and walks into the dark",
        # WIDENED 2026-08-30. Six clauses exhausted on the first 5-ACT episode
        # of the overnight run -- "the figure fallback pool is exhausted: 6
        # clauses, all already used in this episode ... widen the pool", which
        # is the error telling the reader exactly what it wants. A longer
        # episode simply has more beats than six distinct pictures, and the
        # authored path forbids duplicate leaves on purpose.
        "a figure crosses the room and stops at the window",
        "a figure sits down slowly and rests both hands on the table",
        "a figure reaches up and pulls a cord above the desk",
        "a figure stands in the doorway with the light behind",
        "a figure bends toward the floor and rises again",
        "a figure turns a page and holds the paper to the lamp",
        "a figure paces once and settles against the wall",
        "a figure lowers a hand to the table and leaves it there",
        "a figure steps back until the dark takes the edges",
        "a figure tilts toward a sound somewhere off the frame",
        "a figure draws a coat closed and stands very still",
        "a figure moves past the lamp and the shadow swings wide",
    ),
    "object": (
        "it tilts on the table as a shadow crosses it",
        "it catches a rising glow along one worn edge",
        "it rocks once on the wood and settles again",
        "it stands on a desk while dust drifts across it",
        "it turns a quarter and the highlight slides away",
        "it sinks slowly into the shadow at the table edge",
        # Widened with figure (above). Non-figure modes must carry NO human
        # words -- `_HUMAN_WORDS` rejects a clause that "requests a person in
        # object mode" -- so these stay strictly on the thing and the light.
        "it slides an inch across the wood and stops",
        "it gleams once as the lamp above it steadies",
        "it lies on its side while the light narrows",
        "it sits square on the desk and the shadow lengthens",
        "it shifts as the table takes a knock from below",
        "it darkens as the glow behind it fades out",
        "it rests on scattered paper and holds the light",
        "it turns slightly and the engraved edge catches",
        "it stands upright while dust settles around it",
        "it leans against the lamp base and stays there",
        "it tips forward and rights itself on the wood",
        "it holds a thin white line along its upper edge",
    ),
    "signal": (
        "a lamp swings past it and the shadow sweeps the wall",
        "light crawls across it and steadies against the back wall",
        "a warm beam finds it and holds on its worn edge",
        "it sits in the dark as a slow light passes over it",
        "the light on it narrows to a single bright band",
        "the light leaves it and the room goes dim",
        # Widened with figure (above); same no-human-words rule as object.
        "a slow beam crosses it and climbs the far wall",
        "the glow tightens on it and the corners go black",
        "light pulses over it once and holds steady",
        "a shaft of light drops across it from above",
        "the lamp flickers and the shadow jumps behind it",
        "light rakes across it and settles low",
        "a pale wash finds it and spreads to the wall",
        "the beam swings wide and returns to rest on it",
        "light fades from it and the room closes in",
        "a hard edge of light cuts across it",
        "the glow behind it swells and steadies",
        "light slides along it and stops at the seam",
    ),
}

#: The bookends get their own phase-specific clauses so an episode does not
#: open and close on the same picture, which is what a shared pack register
#: produced on every v1 episode.
GHOST_FALLBACK_BOOKENDS = {
    ("opening", "object"): "it lights from cold to warm on the studio desk",
    ("opening", "signal"): "a warm light finds it in the dark studio",
    ("closing", "object"): "it dims slowly and the highlight leaves its edge",
    ("closing", "signal"): "the light slides off it and the studio goes dark",
}


def _bookend_phase(beat_id, ordinal, total) -> str:
    """``opening`` or ``closing`` for a bookend, by id then by position."""
    bid = str(beat_id or "").lower()
    if "open" in bid:
        return "opening"
    if "clos" in bid or "end" in bid:
        return "closing"
    return "opening" if int(ordinal) * 2 < int(total or 1) else "closing"


def deterministic_leaf(spec, *, episode_seed, used=(), total=0) -> str:
    """One complete checked-in clause for a spec, unique within the batch.

    Deterministic collision PROBING rather than modulo-and-hope: two beats in
    the same mode would otherwise land on the same clause about one time in
    six, and an episode that says the same sentence twice is exactly the
    repetition this sprint exists to remove.
    """
    mode = str(spec.get("mode") or "")
    role = normalize_role(spec.get("role"))
    ordinal = int(spec.get("ordinal") or 0)
    if role != "character_video":
        phase = _bookend_phase(spec.get("beat_id"), ordinal, total)
        candidate = GHOST_FALLBACK_BOOKENDS.get((phase, mode), "")
        if candidate and candidate not in used:
            return candidate
    pool = GHOST_FALLBACK_CLAUSES.get(mode) or ()
    if not pool:
        raise GhostAuthorError(
            "no deterministic Ghost clause pool for mode %r" % (mode,))
    start = _hash_int(episode_seed, spec.get("beat_id"), mode,
                      GHOST_AUTHOR_VERSION) % len(pool)
    lowered = {str(u).casefold() for u in used}
    for step in range(len(pool)):
        candidate = pool[(start + step) % len(pool)]
        if candidate.casefold() not in lowered:
            return candidate
    raise GhostAuthorError(
        "the %s fallback pool is exhausted: %d clauses, all already used in "
        "this episode. The authored path forbids duplicate leaves, so the "
        "deterministic path may not quietly ship one -- widen the pool."
        % (mode, len(pool)))


def deterministic_batch(specs, *, episode_seed, already_used=()) -> dict:
    """``{opaque_id: leaf}`` -- a complete batch, never a partial salvage.

    ``already_used`` carries leaves decided elsewhere in the SAME episode --
    replayed rows, typically. Without it a mixed episode could hand two beats
    the same checked-in clause, because each call would probe for collisions
    only against its own subset, and an episode that says the same sentence
    twice is the repetition this sprint exists to remove.
    """
    out = {}
    used = [str(leaf) for leaf in (already_used or ())]
    total = len(specs or ())
    for spec in specs or ():
        leaf = deterministic_leaf(spec, episode_seed=episode_seed,
                                  used=used, total=total)
        used.append(leaf)
        out[spec["id"]] = leaf
    return out


# --------------------------------------------------------------------------- #
# The stored object.
# --------------------------------------------------------------------------- #

#: THE EXACT FIELD SET. Extra or missing keys are a malformed object, and a
#: malformed object fails closed rather than degrading to v1.
GHOST_PROMPT_FIELDS = (
    "schema_version", "author_version", "mode", "motif_cue", "drawable_beat",
    "source", "model_id", "request_sha256", "output_sha256", "fallback_reason",
)

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def build_ghost_prompt_object(spec, drawable_beat, *, source,
                              fallback_reason="") -> dict:
    """Assemble one durable ``ghost_prompt`` object and validate it."""
    obj = {
        "schema_version": GHOST_AUTHOR_SCHEMA_VERSION,
        "author_version": GHOST_AUTHOR_VERSION,
        "mode": spec["mode"],
        "motif_cue": spec["motif_cue"],
        "drawable_beat": _norm(drawable_beat),
        "source": str(source),
        "model_id": spec["model_id"],
        "request_sha256": spec["request_sha256"],
        "output_sha256": output_sha256(_norm(drawable_beat)),
        "fallback_reason": str(fallback_reason or ""),
    }
    validate_ghost_prompt_object(obj)
    return obj


def validate_ghost_prompt_object(obj, *, expected_request_sha256=None) -> dict:
    """Raise unless ``obj`` is a complete, self-consistent stored object."""
    if not isinstance(obj, dict):
        raise GhostAuthorValidationError(
            "ghost_prompt is a %s, not an object" % type(obj).__name__)
    if set(obj) != set(GHOST_PROMPT_FIELDS):
        extra = sorted(set(obj) - set(GHOST_PROMPT_FIELDS))
        missing = sorted(set(GHOST_PROMPT_FIELDS) - set(obj))
        raise GhostAuthorValidationError(
            "ghost_prompt field set is wrong (extra=%s missing=%s)"
            % (",".join(extra) or "-", ",".join(missing) or "-"))
    if obj["schema_version"] != GHOST_AUTHOR_SCHEMA_VERSION:
        raise GhostAuthorValidationError(
            "ghost_prompt schema_version %r is not %d"
            % (obj["schema_version"], GHOST_AUTHOR_SCHEMA_VERSION))
    if obj["author_version"] != GHOST_AUTHOR_VERSION:
        raise GhostAuthorValidationError(
            "ghost_prompt author_version %r is not %r"
            % (obj["author_version"], GHOST_AUTHOR_VERSION))
    if obj["mode"] not in GHOST_MODES:
        raise GhostAuthorValidationError(
            "ghost_prompt mode %r is not one of %s"
            % (obj["mode"], ", ".join(GHOST_MODES)))
    if obj["source"] not in GHOST_AUTHOR_SOURCES:
        raise GhostAuthorValidationError(
            "ghost_prompt source %r is not one of %s"
            % (obj["source"], ", ".join(GHOST_AUTHOR_SOURCES)))
    for field in ("motif_cue", "drawable_beat", "model_id"):
        if not _norm(obj[field]):
            raise GhostAuthorValidationError(
                "ghost_prompt %s is empty" % field)
    for field in ("request_sha256", "output_sha256"):
        if not _HEX64_RE.match(str(obj[field] or "")):
            raise GhostAuthorValidationError(
                "ghost_prompt %s is not 64 lowercase hex characters" % field)
    if output_sha256(obj["drawable_beat"]) != obj["output_sha256"]:
        raise GhostAuthorValidationError(
            "ghost_prompt output_sha256 does not hash its own drawable_beat")
    if obj["source"] == "deterministic_fallback" and not obj["fallback_reason"]:
        raise GhostAuthorValidationError(
            "ghost_prompt source=deterministic_fallback carries no reason -- "
            "the reason is what stops a reuse laundering it into proof")
    if obj["source"] != "deterministic_fallback" and obj["fallback_reason"]:
        raise GhostAuthorValidationError(
            "ghost_prompt source=%s carries a fallback_reason" % obj["source"])
    if (expected_request_sha256 is not None
            and obj["request_sha256"] != expected_request_sha256):
        raise GhostAuthorValidationError(
            "ghost_prompt request_sha256 does not match the recomputed "
            "request identity")
    return obj


# --------------------------------------------------------------------------- #
# The installed SD1 token measurer.
# --------------------------------------------------------------------------- #

_TOKENIZER_CACHE = {}


def _test_mode() -> bool:
    return otr_env.get("OTR_TEST_MODE") == "1"


def _installed_sd1_tokenizer():
    """The lazily instantiated ComfyUI SD1 tokenizer, contract-checked once.

    Contract, asserted rather than assumed: max length 77, BOS 49406, EOS/pad
    49407. Hugging Face's own ``model_max_length`` on this tokenizer reads
    8192, which is metadata about the text model and says nothing about the
    77-token CLIP window -- trusting it is how a counter silently stops
    counting.
    """
    if "tokenizer" in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE["tokenizer"]
    try:
        from comfy.sd1_clip import SD1Tokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001 -- any import failure is the same
        raise GhostTokenizerUnavailable(
            "the installed ComfyUI SD1 tokenizer could not be imported (%s). "
            "Ghost Prompt v2 measures its window with the real encoder and "
            "will not substitute a whitespace estimate." % (exc,)) from exc
    tok = SD1Tokenizer()
    clip = getattr(tok, "clip_l", None)
    if clip is None:
        raise GhostTokenizerUnavailable(
            "the installed SD1Tokenizer exposes no clip_l section")
    if int(getattr(clip, "max_length", 0)) != GHOST_CLIP_WINDOW_TOKENS:
        raise GhostTokenizerUnavailable(
            "the installed SD1 tokenizer reports max_length %r, not %d"
            % (getattr(clip, "max_length", None), GHOST_CLIP_WINDOW_TOKENS))
    if getattr(clip, "start_token", None) is None or \
            getattr(clip, "end_token", None) is None:
        raise GhostTokenizerUnavailable(
            "the installed SD1 tokenizer has no real start/end tokens")
    _TOKENIZER_CACHE["tokenizer"] = tok
    return tok


def measure_clip_tokens(text) -> tuple:
    """``(tokens, windows)`` for ``text`` on the INSTALLED SD1 encoder.

    ``tokens`` is the payload count plus BOS and EOS for every returned
    section; ``windows`` is how many 77-token sections came back. Padded row
    LENGTH is never counted -- every row is padded to 77, so counting it would
    report 77 for a three-word prompt and make the whole gate meaningless.

    Measured, not derived: 75 payload tokens is 77 total in one window, and 76
    payload tokens spills to two.
    """
    tok = _installed_sd1_tokenizer()
    rows = tok.tokenize_with_weights(str(text or ""), return_word_ids=True)["l"]
    payload = sum(1 for row in rows for entry in row if entry[2] != 0)
    windows = len(rows)
    return payload + 2 * windows, windows


def resolve_token_measure(token_measure_fn=None) -> Optional[Callable]:
    """The measurer this call should use, or ``None`` when it may be skipped.

    An INJECTED measurer always gates -- a test that supplies one is asking for
    the gate, not asking to be excused from it. Only the UNAVAILABLE installed
    tokenizer may be skipped, and only under ``OTR_TEST_MODE``; production
    fails closed.
    """
    if token_measure_fn is not None:
        return token_measure_fn
    try:
        _installed_sd1_tokenizer()
    except GhostTokenizerUnavailable as exc:
        if _test_mode():
            # SAID OUT LOUD (agy review, 2026-08-22). A silently ungated run
            # publishes zero token counts that look like a measurement; a
            # reader of a headless log must be able to see that nothing was
            # measured and why.
            _LOG.warning(
                "[ghost_signal_author] token gate SKIPPED under OTR_TEST_MODE: "
                "%s -- receipts will carry no counts and no clip_counter", exc)
            return None
        raise
    return measure_clip_tokens


#: The measurer's own identity, stamped beside every count so a receipt says
#: which counter produced it.
GHOST_CLIP_COUNTER = "comfy.sd1_clip.SD1Tokenizer/%s" % GHOST_AUTHOR_VERSION


# --------------------------------------------------------------------------- #
# PROMPT v3 -- "draw the crux". Resolvers for the three episode-derived slots.
#
# All of this is RENDER-TIME and reads only the ledger meta the driver already
# holds. Nothing here writes to the ledger, changes an authored object, or calls
# a model -- which is what lets a frozen episode replay under v3 with a
# byte-identical `render_request_hash` and therefore a byte-identical seed.
#
# THE ONE FIELD THESE MUST NEVER TOUCH is `story_brief_terms` as a WRITE target:
# `otr_shot_lock._derive_creative` hashes it into `brief_hash`, `brief_hash`
# feeds the per-shot `request_hash`, and `request_hash` is what the render seed
# is derived from. Reading it is free; adding a key to it would move every seed
# in the episode and destroy the A/B this composer exists to make possible.
# --------------------------------------------------------------------------- #

#: WORLD MOTION, keyed by VANTAGE rather than by the stored mode identity.
#:
#: A SEPARATE POOL FROM `GHOST_FALLBACK_CLAUSES`, and that is the point. Those
#: clauses are whole sentences with their own subject -- "a figure turns a page
#: and holds the paper to the lamp", "it slides an inch across the wood and
#: stops" -- so appending one after a crux kernel would read "a vast cold water
#: reservoir, a figure turns a page ..." and re-introduce, at tabletop scale,
#: exactly the unmentioned figure this composer exists to remove. Every clause
#: here is a bare verb phrase with NO subject of its own, so it composes
#: correctly after any kernel.
#:
#: SIZED AGAINST A REAL EXHAUSTION, not a round number. On 2026-08-30 a five-act
#: overnight episode exhausted a six-clause pool on `figure` alone -- character
#: beats are the most common under `GHOST_CHARACTER_CYCLE` -- and under v3 this
#: pool fires on EVERY beat rather than only on a failed batch. The longest real
#: episode observed is 29 shots, so each bucket carries comfortably more than
#: that and `test_ghost_prompt_v3` pins the no-exhaustion claim against it.
GHOST_WORLD_MOTION_V3 = {
    "figure": (
        "the air moving slowly through it",
        "a slow drift across the space",
        "dust turning in the light",
        "the light shifting along the far wall",
        "a stillness broken once and settling",
        "shadows lengthening across the floor",
        "a draught pulling through the room",
        "the far end sinking into the dark",
        "movement passing through and going quiet",
        "the room settling into stillness",
        "light sliding down the far wall",
        "the space breathing once and holding",
        "a slow shift toward the far side",
        "the dark gathering at the edges",
        "the air thickening across the room",
        "a slow tide of shadow crossing it",
        "the ceiling light swaying faintly",
        "the far door standing open on nothing",
        "a draught lifting the loose edges",
        "the floor disappearing into shadow",
        "a slow settling of everything in it",
        "distance opening toward the back",
        "the walls receding into grey",
        "one slow pass of light across it",
        "quiet closing over the space",
        "the corners going soft and dark",
        "a slow current running through it",
        "the light failing gradually at the back",
        "stillness spreading outward",
        "the room holding its breath",
        "shadow pooling toward the middle",
        "a long slow exhale of dust",
    ),
    "object": (
        "drifting slowly",
        "settling in the still air",
        "shifting as the light crosses it",
        "stirring once and going still",
        "moving with the draught",
        "trembling faintly",
        "turning slowly in place",
        "tilting and settling back",
        "gathering dust as it sits",
        "catching the light along one edge",
        "sliding a little and stopping",
        "rocking once and steadying",
        "darkening as the light moves off",
        "shedding a thin fall of dust",
        "leaning slowly out of true",
        "brightening and dimming again",
        "shivering under the draught",
        "settling deeper where it rests",
        "throwing a shadow that lengthens",
        "creasing slowly under its own weight",
        "loosening and lying still",
        "warming under the lamp",
        "cooling into shadow",
        "shifting a fraction and holding",
        "gleaming once and going dull",
        "sagging slowly at one corner",
        "turning a slow quarter and stopping",
        "gathering a film of grey",
        "flexing once in the moving air",
        "coming slowly out of the dark",
        "losing its edge to the shadow",
        "holding a thin line of light",
    ),
    "signal": (
        "the light sweeping slowly across it",
        "a glow rising and falling on it",
        "the beam crossing and leaving it dark",
        "a slow pulse of light over its surface",
        "light crawling along one edge",
        "the glow steadying and then thinning",
        "a bar of light passing over it",
        "the dark closing in and opening again",
        "light guttering across it",
        "a slow bloom of light and its fade",
        "the beam narrowing onto it",
        "light sliding off it into the dark",
        "a flicker travelling its length",
        "the glow drifting off centre",
        "light pooling and draining away",
        "a slow strobe washing over it",
        "the beam swinging wide and back",
        "light climbing it and falling away",
        "a dim wash rising on it",
        "the light breaking up across its face",
        "a slow scan of light down it",
        "the glow tightening to a point",
        "light rolling over it in waves",
        "the dark taking it a piece at a time",
        "a low shimmer moving on it",
        "light drawing back off it slowly",
        "a slow blink of brightness",
        "the beam holding and then sliding",
        "light spilling across and receding",
        "a slow flare and its long decay",
        "the glow crossing it end to end",
        "light failing gradually off its edge",
    ),
}


#: How many terms of any one list v3 will cycle. A brief that returns twenty
#: settings is not twenty different places; it is one place described twenty
#: ways, and cycling all of them would make consecutive beats look unrelated.
GHOST_V3_TERM_LIMIT = 6

#: The longest kernel v3 will compose, in words, before it drops the setting
#: half. Whole units only -- a kernel is never word-sliced, because slicing
#: "data logs" to "data" changes what is drawn.
GHOST_V3_KERNEL_MAX_WORDS = 9


def _spoken_term(term) -> str:
    """One brief term as a human would say it -- see `_otr_brief_reader`.

    The implementation lives beside `_read_brief_field` rather than here, because
    FOUR consumers join `story_brief_terms.setting` into model-facing text (this
    composer, the still-image fallback, the ShotLock derivation prompt and the
    music prompt) and normalising one of them is how the same episode ends up
    spelled two ways in two prompts. This name is kept as the local alias the
    rest of this module already reads through.
    """
    try:
        from .._otr_brief_reader import spoken_term  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_brief_reader import spoken_term  # type: ignore
    return spoken_term(term)


def _setting_terms(meta) -> list:
    """The episode's setting terms as a LIST, bounded, possibly empty.

    ``otr_shot_lock._read_setting`` returns a JOINED string with a hard-coded
    fallback, which is right for its caller and wrong for this one: v3 cycles
    the terms across beats, so it needs them separately, and it must be able to
    see that there are none (a failed brief stamps ``setting: []``) rather than
    receive a stand-in that looks like real episode vocabulary.
    """
    terms = (meta or {}).get("story_brief_terms") or {}
    raw = terms.get("setting") if isinstance(terms, dict) else None
    out = [_spoken_term(t) for t in raw if _spoken_term(t)] if isinstance(raw, list) else []
    if out:
        return out[:GHOST_V3_TERM_LIMIT]
    try:
        from .._otr_brief_reader import _read_brief_field  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        try:
            from _otr_brief_reader import _read_brief_field  # type: ignore
        except ImportError:
            return []
    try:
        raw = _read_brief_field(meta, "setting", default=[])
    except Exception:  # noqa: BLE001 -- an unreadable brief is an absent brief
        return []
    if isinstance(raw, list):
        return [_spoken_term(t) for t in raw if _spoken_term(t)][:GHOST_V3_TERM_LIMIT]
    if isinstance(raw, str) and _spoken_term(raw):
        return [_spoken_term(raw)]
    return []


def resolve_crux_kernel(meta, *, ordinal=0, role="", mode="") -> tuple:
    """The beat's SUBJECT: the story's own thing, in the story's own place.

    TOTAL BY CONSTRUCTION -- four tiers, and the last one cannot fail. It never
    raises and never returns an empty kernel, because the measured rule on this
    lane is that legibility tracks concrete nouns: a prompt that names no thing
    rendered a recognisable subject on 0 of 4 sampled beats. Returning "" here
    would rebuild that condition on every brief-failed episode.

    The tiers, in order:

    1. ``meta.key_objects[i]`` in ``story_brief_terms.setting[j]`` -- the
       story's own object in the story's own place. This is the whole point.
    2. the setting term alone, when the brief gave a place but no objects.
    3. a bounded slice of ``meta.story_brief``, when only the prose survived.
    4. the beat's BOOKEND RADIO OBJECT -- a bakelite radio set, a glowing radio
       dial, a broadcast console, a spinning turntable. Always present, always
       drawable, always right for a radio programme, and never a costume or a
       cast look. A brief-failed episode draws the radio world rather than a
       field of nothing.

    ODOMETER CYCLING, not modulo on both wheels. Cycling the object and the
    place on the same index makes the PAIR repeat every ``len(objects)`` beats:
    with four objects and four settings the design prototype produced seven
    byte-identical kernels in a 29-beat episode. Rolling the place only when the
    object wraps gives a period of ``len(objects) * len(settings)``, which no
    real episode reaches.

    Returns ``(kernel, source)`` where source is one of ``bookend_radio``,
    ``key_object``, ``setting``, ``brief`` or ``bookend``, and is receipted so a
    published episode says which tier fed it.
    """
    ordinal = max(int(ordinal or 0), 0)
    objects = [_spoken_term(o) for o in ((meta or {}).get("key_objects") or [])
               if _spoken_term(o)][:GHOST_V3_TERM_LIMIT]
    places = _setting_terms(meta)

    def _in_place(subject):
        """``<subject> in the <place>``, or the subject alone if that is long."""
        if not places:
            return subject
        place = places[(ordinal // max(len(objects), 1)) % len(places)]
        # A BRIEF MAY LIST THE SAME TERM AS BOTH THE OBJECT AND THE PLACE, and
        # then the pair says a thing is inside itself: "petri dish in the petri
        # dish" was composed on a real episode. The subject is the half worth
        # keeping -- it is what the beat is about -- so the place drops, exactly
        # as it does for an over-long pair below.
        if place.casefold() == subject.casefold():
            return subject
        pair = "%s in the %s" % (subject, place)
        # WHOLE UNITS ONLY: an over-long pair drops the PLACE, never half of
        # either phrase.
        return pair if len(pair.split()) <= GHOST_V3_KERNEL_MAX_WORDS else subject

    # THE RADIO STAYS ON THE BOOKENDS, and it is an operator rule rather than a
    # fallback: "radio objects stay on the announcer and music beds, but placed
    # in the setting (the radio set with the reservoir behind it), not on a
    # table in a dark room". His own rewrite of one reads "a bakelite radio set,
    # with a background of British Columbia's Williston Reservoir with floating
    # driftwood" -- the programme's own object, standing in the story's place.
    #
    # So a bookend takes its SUBJECT from `GHOST_BOOKEND_MOTIFS` and its PLACE
    # from the episode, exactly as a character beat takes both from the episode.
    # Without this the announcer bed drew "handwritten ledgers in the
    # high-security archive" and the radio disappeared from the programme.
    if str(role or "") and str(role or "") != "character_video":
        motif = (GHOST_BOOKEND_MOTIFS.get((str(role), str(mode or "")))
                 or GHOST_BOOKEND_MOTIFS.get(("announcer_visual", str(mode or ""))))
        if motif:
            return _in_place(motif), "bookend_radio"

    if objects:
        return _in_place(objects[ordinal % len(objects)]), "key_object"

    if places:
        return places[ordinal % len(places)], "setting"

    brief = " ".join(str((meta or {}).get("story_brief") or "").split())
    if brief:
        sliced = " ".join(brief.split()[:GHOST_V3_KERNEL_MAX_WORDS])
        sliced = sliced.rstrip(",.;:").strip()
        if sliced:
            return sliced, "brief"

    # Tier 4. `bookend_motif` refuses an unknown role/mode pairing, and a
    # character beat has no bookend at all -- so this resolves the radio object
    # for the beat's own mode and falls back to the announcer's, which every
    # mode in `GHOST_VANTAGE_V3` except `figure` has. A `figure` beat on a
    # brief-failed episode gets the announcer's radio set: not the story, but a
    # real thing in a real programme, which is the whole claim tier 4 makes.
    for key in ((str(role or ""), str(mode or "")),
                ("announcer_visual", str(mode or "")),
                ("announcer_visual", "object")):
        motif = GHOST_BOOKEND_MOTIFS.get(key)
        if motif:
            return motif, "bookend"
    return GHOST_BOOKEND_MOTIFS[("announcer_visual", "object")], "bookend"


def resolve_world_light(meta, *, ordinal=0, mode="") -> str:
    """The episode's own lighting term, on the SLOWEST wheel of the odometer.

    Returns "" on ``signal`` mode: that vantage already says "lit against the
    dark, the light moving", and composing a second lighting clause beside it
    produced two contradictory statements in the design prototype.

    The wheel is deliberately slower than the kernel's. One episode is one
    place, and a light that changed every beat would read as a different room
    every beat -- the operator's rule 2 is that the episode lives in ONE place
    and every beat is a view of it.
    """
    if str(mode or "") == "signal":
        return ""
    terms = (meta or {}).get("story_brief_terms") or {}
    raw = terms.get("lighting") if isinstance(terms, dict) else None
    # SPOKEN, like the settings and the objects. `lighting` is the same
    # LLM-authored list from the same brief and leaks identifier case at the
    # same rate -- measured 481 of 7,978 terms (6.0%) across the episodes on
    # disk, e.g. `storm_light`, `dim_glow`, `harsh_fluorescent`. The first pass
    # at PBUG-20260903-04 normalised `setting` and `key_objects` and missed this
    # sibling, which a source-bank sweep then found live in a rendered prompt:
    # "handheld bronze communicator in the forest, storm_light, ...".
    lights = [_spoken_term(t) for t in raw if _spoken_term(t)] if isinstance(raw, list) else []
    if not lights:
        return ""
    objects = [o for o in ((meta or {}).get("key_objects") or []) if str(o).strip()]
    places = _setting_terms(meta)
    wheel = max(len(objects[:GHOST_V3_TERM_LIMIT]), 1) * max(len(places), 1)
    return lights[:GHOST_V3_TERM_LIMIT][
        (max(int(ordinal or 0), 0) // wheel) % len(lights[:GHOST_V3_TERM_LIMIT])]


def resolve_world_motion(*, mode, episode_seed, ordinal=0) -> str:
    """One world-motion clause for the beat, distinct within the episode.

    STATELESS AND PROVABLY NON-REPEATING, which is why it walks the pool by
    ORDINAL rather than by probing a set of already-used clauses. The render
    driver builds each clip's request independently and shares no per-episode
    state between them, so a collision-probing walk like ``deterministic_leaf``
    has nothing to probe against here. Offsetting a hashed start by the beat
    ordinal gives every beat in a bucket a different clause outright, for any
    episode no longer than the bucket -- which is how the buckets were sized.

    The episode seed only chooses WHERE in the pool the episode starts, so two
    episodes do not open on the same clause; the ordinal does the rest.
    """
    pool = GHOST_WORLD_MOTION_V3.get(str(mode or "")) or ()
    if not pool:
        return ""
    start = _hash_int(episode_seed, "ghost_world_motion_v3", mode) % len(pool)
    return pool[(start + max(int(ordinal or 0), 0)) % len(pool)]


# --------------------------------------------------------------------------- #
# The shared finalizer -- ONE path for author-time admission and render.
# --------------------------------------------------------------------------- #

def _banana_module():
    try:
        from .. import _otr_banana_route as _br  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_banana_route as _br  # type: ignore
    return _br


def finalize_ghost_prompt_v2(*, role, style, mode, motif_cue, drawable_beat,
                             ledger_meta=None, token_measure_fn=None,
                             banana_enabled=None) -> dict:
    """Compose, transform, validate and MEASURE one Ghost v2 prompt.

    The single path both sides use. Author-time admission calls it to find out
    whether a candidate leaf survives the real style, the real banana gate and
    the real tokenizer; the render driver calls it to build the request. Two
    code paths could not have been kept in agreement -- the whole point of the
    admission check is that it predicts the render exactly.

    IT NEVER TRIMS AND NEVER REPAIRS. Style cue, motif, law and negative are
    immutable, and the only lever an over-budget candidate has is a different
    LEAF -- which is the model's field, not this function's.
    """
    composed = _gsp.compose_ghost_prompt_v2(
        role=role, style=style, mode=mode, motif_cue=motif_cue,
        drawable_beat=drawable_beat)
    positive = str(composed["positive"])
    negative = str(composed["negative"])
    components = dict(composed["components"])

    meta = ledger_meta if isinstance(ledger_meta, dict) else {}
    banana = _banana_module()
    variety_key = str(meta.get("freeze_timestamp") or "")
    gate_on = (banana.banana_gate(meta, lane="video")
               if banana_enabled is None else bool(banana_enabled))

    if gate_on:
        result = banana.apply(positive, variety_key=variety_key,
                              shield_quoted_card_text=False)
        positive = result.text
        receipt = banana.receipt_keys(result)
        # THE COMPONENTS GO THROUGH THE SAME TABLE, so a substitution INSIDE
        # the leaf ("revolver" -> a fruit) is recognised as the transform doing
        # its job rather than as the leaf having been lost. Validating the
        # pre-transform components against a post-transform prompt would fail
        # exactly when the route fired, which is the one case that matters.
        components = {
            name: banana.apply(text, variety_key=variety_key,
                               shield_quoted_card_text=False).text
            for name, text in components.items()
        }
    else:
        receipt = banana.off_receipt(positive, variety_key=variety_key)

    for name, text in components.items():
        if text and text not in positive:
            raise GhostBudgetError(
                "Ghost v2 protected component %r did not survive into the "
                "final positive -- a protected field is never trimmed or "
                "repaired, so this is a composer or transform defect" % (name,))

    if len(positive) > _gsp.GHOST_PROMPT_MAX_CHARS:
        raise GhostBudgetError(
            "Ghost v2 positive is %d characters, over the %d ceiling"
            % (len(positive), _gsp.GHOST_PROMPT_MAX_CHARS))

    measure = resolve_token_measure(token_measure_fn)
    if measure is None:
        pos_tokens = pos_windows = neg_tokens = neg_windows = 0
    else:
        pos_tokens, pos_windows = measure(positive)
        neg_tokens, neg_windows = measure(negative)
        for label, tokens, windows in (("positive", pos_tokens, pos_windows),
                                       ("negative", neg_tokens, neg_windows)):
            if windows > 1 or tokens > GHOST_CLIP_WINDOW_TOKENS:
                raise GhostBudgetError(
                    "Ghost v2 %s measures %d installed SD1 tokens across %d "
                    "window(s), over the one-window %d admission -- the render "
                    "boundary refuses this rather than trimming a protected "
                    "field" % (label, tokens, windows,
                               GHOST_CLIP_WINDOW_TOKENS))

    return {
        "positive": positive,
        "negative": negative,
        "components": components,
        "slots": list(composed.get("slots") or ()),
        "banana_receipt": receipt,
        "banana_gate": bool(gate_on),
        "positive_clip_tokens": int(pos_tokens),
        "positive_clip_windows": int(pos_windows),
        "negative_clip_tokens": int(neg_tokens),
        "negative_clip_windows": int(neg_windows),
        "clip_window_max": GHOST_CLIP_WINDOW_TOKENS,
        "clip_counter": GHOST_CLIP_COUNTER if measure is not None else "",
        "measured": measure is not None,
    }


#: The v3 DROP ORDER: whole optional units, cheapest meaning first.
#:
#: NEVER the kernel's subject noun and never the pack cue -- those two are what
#: make the picture legible and what make it look like the pack. And never a
#: WORD out of any unit: slicing "data logs" down to "data" changes the thing
#: being drawn, which is worse than dropping a clause that only decorated it.
#: The kernel's SETTING half is the last resort and is dropped as one piece.
#:
#: TRAILING STYLE GOES FIRST, AND MOTION MOVED BEHIND VANTAGE (2026-09-03).
#: Two deliberate placements, both from the operator's ruling that the budget
#: buys "visual style - key objects per beat story - + movement":
#:
#: * The trailing style clause is an ENRICHMENT, so it is the first thing
#:   surrendered. Adding it must never cost a slot that was already earning its
#:   place; if the prompt is tight, the prompt simply reverts to what it emitted
#:   before this clause existed.
#: * `motion` used to be dropped SECOND, which was harmless only because the
#:   ladder never fired -- v3 measures ~32 tokens against a target of 69. This
#:   clause lengthens prompts and could make it fire for the first time, and
#:   deleting movement before framing directly contradicts the ruling. Framing
#:   (`vantage`) is the cheaper loss: the shot still moves, it is just less
#:   precisely staged.
GHOST_V3_DROP_ORDER = ("trailing_style", "light", "vantage", "motion",
                       "kernel_setting")

#: LAB LEVER: WHERE THE STYLE CUE SITS IN THE PROMPT (operator A/B/C, 2026-09-03).
#:
#: A Ghost v3 prompt currently names the pack TWICE -- `pack_cue` opens it and
#: `trailing_style` closes it:
#:
#:   recursive fractal light field. a bakelite radio set, distance opening
#:   toward the back, wide, the people small in the space, pristine raster
#:   geometry, nested self-similar depth, emissive shader surfaces
#:   |________ pack_cue ________|                    |____ trailing_style ____|
#:
#: The operator wants to know whether that repetition earns its tokens:
#: *"the main diff is including reference to the style at the start AND at the
#: end, or just the start, or just the end."* This is the only knob that
#: question needs, and it is READ-ONLY OBSERVATION -- it changes nothing unless
#: set:
#:
#:   OTR_GHOST_STYLE_PLACEMENT=both   (default; byte-identical to shipping)
#:   OTR_GHOST_STYLE_PLACEMENT=front  arm A -- pack_cue only, trailing dropped
#:   OTR_GHOST_STYLE_PLACEMENT=end    arm B -- trailing only, pack_cue stripped
#:
#: Applied INSIDE the composer rather than to the finished string, so the token
#: measurement and the drop ladder see the arm that will actually render. An arm
#: judged on a prompt the budget never saw is not the arm that shipped.
GHOST_STYLE_PLACEMENT_ENV = "OTR_GHOST_STYLE_PLACEMENT"
GHOST_STYLE_PLACEMENTS = ("both", "front", "end")


def _style_placement() -> str:
    """`both` unless the operator is running the placement A/B/C."""
    raw = str(otr_env.get(GHOST_STYLE_PLACEMENT_ENV, "") or "").strip().lower()
    return raw if raw in GHOST_STYLE_PLACEMENTS else "both"


def _apply_style_placement(composed: dict) -> dict:
    """Drop one end of the style sandwich, per :func:`_style_placement`.

    `front` drops the trailing clause; `end` strips the opening pack cue, which
    the composer joins as ``"<pack_cue>. <rest>"``. The strip is by exact prefix
    match on that spelling -- if the join ever changes, this refuses to guess
    and leaves the prompt alone rather than mangling it.
    """
    placement = _style_placement()
    if placement == "both":
        return composed
    out = dict(composed)
    comp = dict(out.get("components") or {})
    slots = list(out.get("slots") or [])
    positive = str(out.get("positive") or "")

    if placement == "front":
        trailing = str(comp.get("trailing_style") or "")
        if trailing and positive.endswith(", " + trailing):
            positive = positive[: -len(", " + trailing)]
            comp["trailing_style"] = ""
            slots = [s for s in slots if s != "trailing_style"]
    elif placement == "end":
        cue = str(comp.get("pack_cue") or "")
        if cue and positive.startswith(cue + ". "):
            positive = positive[len(cue) + 2:]
            comp["pack_cue"] = ""
            slots = [s for s in slots if s != "pack_cue"]

    out["positive"] = positive
    out["components"] = comp
    out["slots"] = slots
    return out


def finalize_ghost_prompt_v3(*, role, style, mode, ledger_meta=None,
                             ordinal=0, token_measure_fn=None,
                             banana_enabled=None, pack_motion="") -> dict:
    """Resolve, compose, transform, FIT and measure one Ghost v3 prompt.

    The v3 sibling of :func:`finalize_ghost_prompt_v2`, and deliberately a
    SEPARATE function rather than an edit of it. Two reasons, both load-bearing:

    * v2 is still the author-time admission path. ``candidate_fits`` and
      ``assert_shell_fits`` call it while ShotLock is planning, and
      ``build_request_from_shot`` -- which is also ShotLock's cast-time
      preflight -- catches only ``DeferredImageGapError``. A budget raise on
      that path kills plan build for a new episode after the writer LLM has
      already run.
    * v2 is contracted never to trim. v3 MUST trim, because it composes from
      episode vocabulary whose length nobody controls.

    So the contracts differ on purpose, and the difference is the point:
    **v3 never raises FOR BUDGET.** The valve is the drop order, not a refusal,
    and `resolve_crux_kernel` is a total ladder so there is always something to
    compose.

    IT CAN STILL FAIL CLOSED, and deliberately: a row whose `role` is not a real
    role, or whose `mode` has no vantage, raises out of
    `compose_ghost_prompt_v3` exactly as it would out of the v2 composer, and an
    unavailable SD1 tokenizer raises out of `resolve_token_measure` in
    production exactly as it does for v2. Those are malformed input and missing
    infrastructure, not length -- and the surrounding law on this lane is that a
    malformed object fails closed rather than downgrading to something that
    looks like a healthy render. Author and render no longer compose identical text for
    a v3 beat, and that is intended -- v3 reads the ledger, which the author's
    admission check does not have.

    Order of operations, and it matters: fit toward
    ``GHOST_AUTHOR_TOKEN_TARGET`` BEFORE the banana route, then apply the route
    exactly ONCE, then re-measure. The transform can grow a token, so a prompt
    fitted after it would be measured against the wrong text, and a prompt
    transformed twice would overwrite a real substitution receipt with a
    zero-substitution one.
    """
    meta = ledger_meta if isinstance(ledger_meta, dict) else {}
    kernel, kernel_source = resolve_crux_kernel(
        meta, ordinal=ordinal, role=role, mode=mode)
    light = resolve_world_light(meta, ordinal=ordinal, mode=mode)
    # THE PACK'S OWN KINETIC DIRECTION WINS ON A BOOKEND BEAT (2026-09-03).
    # The operator watched a published episode and reported that the announcer
    # and music beats "had basically no movement". They were composing from
    # `GHOST_WORLD_MOTION_V3`, whose clauses are atmospheric by design -- "the
    # glow steadying and then thinning", "cooling into shadow" -- while every
    # style pack had ALREADY authored a kinetic register for exactly these four
    # roles, ending in a camera move ("slow illustrated dolly forward"), which
    # the live v3 path never read.
    #
    # Passed IN by the driver rather than resolved here: the role-to-register
    # key needs the shot id, and `render_driver` states outright that a second
    # role-to-register table in a pure module is a table that drifts once.
    # Empty on a character beat, and empty whenever the pack has no usable
    # register -- either way the generic pool still supplies a motion clause,
    # so a beat never loses the slot entirely.
    motion = str(pack_motion or "").strip() or resolve_world_motion(
        mode=mode, episode_seed=meta.get("episode_seed"), ordinal=ordinal)
    # The pack's own style vocabulary for the END of the prompt. Resolved ONCE
    # here rather than inside `_compose`, which the fitter may call five times.
    trailing = _gsp._trailing_pack_cue(style)

    measure = resolve_token_measure(token_measure_fn)
    dropped: list = []

    def _compose(with_light, with_motion, with_vantage, with_place,
                 with_trailing):
        text_kernel = kernel
        if not with_place and " in the " in kernel:
            text_kernel = kernel.split(" in the ", 1)[0].strip()
        composed = _gsp.compose_ghost_prompt_v3(
            role=role, style=style, mode=mode, kernel=text_kernel,
            light=light if with_light else "",
            motion=motion if with_motion else "")
        if not with_vantage:
            # The vantage is composed by the pure function and removed here as
            # ONE span whose bytes we hold -- never re-joined by hand, so the
            # separators stay exactly what the composer produced.
            #
            # REMOVED FROM THE END, never by `replace`. The vantage is always
            # the last unit, and a bare `.replace` would delete EVERY occurrence
            # of that exact span: a future pack cue, a long setting phrase or a
            # motion clause that happened to coincide would be silently cut out
            # of the middle of the prompt. Trimming the suffix can only ever
            # touch the one the composer just appended.
            vantage = composed["components"]["vantage"]
            suffix = ", " + vantage
            positive = composed["positive"]
            trimmed = (positive[:-len(suffix)] if positive.endswith(suffix)
                       else positive)
            composed = dict(composed)
            composed["positive"] = trimmed
            composed["components"] = dict(composed["components"])
            composed["components"]["vantage"] = ""
            composed["slots"] = [s for s in composed["slots"] if s != "vantage"]
        # APPENDED LAST, AND DELIBERATELY AFTER THE VANTAGE TRIM ABOVE. That
        # trim removes the vantage as a SUFFIX of the composed text, so a style
        # clause added before it would leave the vantage mid-string and the
        # suffix match would silently fail -- the prompt would keep a vantage it
        # was told to drop. Composing without the clause and adding it here
        # keeps that match exact, and is why the pure composer is untouched.
        if with_trailing and trailing:
            composed = dict(composed)
            composed["positive"] = "%s, %s" % (composed["positive"], trailing)
            composed["components"] = dict(composed["components"])
            composed["components"]["trailing_style"] = trailing
            composed["slots"] = list(composed["slots"]) + ["trailing_style"]
        composed = _apply_style_placement(composed)
        return composed

    flags = {"light": True, "motion": True, "vantage": True,
             "kernel_setting": True, "trailing_style": True}
    composed = _compose(True, True, True, True, True)
    if measure is not None:
        for unit in GHOST_V3_DROP_ORDER:
            tokens, windows = measure(composed["positive"])
            if windows <= 1 and tokens <= GHOST_AUTHOR_TOKEN_TARGET:
                break
            flags[unit] = False
            before = composed["positive"]
            composed = _compose(flags["light"], flags["motion"],
                                flags["vantage"], flags["kernel_setting"],
                                flags["trailing_style"])
            # ONLY RECEIPT A UNIT THAT WAS ACTUALLY THERE. `light` is
            # structurally empty on `signal` mode and the kernel's setting half
            # is absent whenever the brief gave no place, so walking the order
            # blind would publish "prompt_dropped: light" for a beat that never
            # had a light clause -- a receipt that reads as a budget decision
            # when nothing was decided.
            if composed["positive"] != before:
                dropped.append(unit)

    positive = str(composed["positive"])
    negative = str(composed["negative"])
    components = dict(composed["components"])

    banana = _banana_module()
    variety_key = str(meta.get("freeze_timestamp") or "")
    gate_on = (banana.banana_gate(meta, lane="video")
               if banana_enabled is None else bool(banana_enabled))
    if gate_on:
        result = banana.apply(positive, variety_key=variety_key,
                              shield_quoted_card_text=False)
        positive = result.text
        receipt = banana.receipt_keys(result)
        components = {
            name: (banana.apply(text, variety_key=variety_key,
                                shield_quoted_card_text=False).text
                   if text else text)
            for name, text in components.items()
        }
    else:
        receipt = banana.off_receipt(positive, variety_key=variety_key)

    for name, text in components.items():
        if text and text not in positive:
            raise GhostBudgetError(
                "Ghost v3 component %r did not survive into the final positive "
                "-- v3 drops WHOLE units before composing, so a missing "
                "component is a composer or transform defect rather than a "
                "budget outcome" % (name,))

    slot_tokens = {}
    if measure is not None:
        pos_tokens, pos_windows = measure(positive)
        neg_tokens, neg_windows = measure(negative)
        slot_tokens = {name: int(measure(text)[0])
                       for name, text in components.items() if text}
    else:
        pos_tokens = pos_windows = neg_tokens = neg_windows = 0

    return {
        "positive": positive,
        "negative": negative,
        "components": components,
        "slots": list(composed.get("slots") or ()),
        "dropped": list(dropped),
        "slot_tokens": slot_tokens,
        "kernel_source": kernel_source,
        "world_motion": motion,
        "banana_receipt": receipt,
        "banana_gate": bool(gate_on),
        "positive_clip_tokens": int(pos_tokens),
        "positive_clip_windows": int(pos_windows),
        "negative_clip_tokens": int(neg_tokens),
        "negative_clip_windows": int(neg_windows),
        "clip_window_max": GHOST_CLIP_WINDOW_TOKENS,
        "clip_counter": GHOST_CLIP_COUNTER if measure is not None else "",
        "measured": measure is not None,
    }


def candidate_fits(*, role, style, mode, motif_cue, drawable_beat,
                   ledger_meta=None, token_measure_fn=None,
                   target_tokens=GHOST_AUTHOR_TOKEN_TARGET) -> tuple:
    """``(ok, reason)`` for an AUTHOR-TIME candidate leaf.

    Author-time aims below the render ceiling on purpose: the leaf is the only
    thing that may change to make a prompt fit, so the composer's own surfaces
    have to keep the headroom the banana route may later spend.
    """
    try:
        final = finalize_ghost_prompt_v2(
            role=role, style=style, mode=mode, motif_cue=motif_cue,
            drawable_beat=drawable_beat, ledger_meta=ledger_meta,
            token_measure_fn=token_measure_fn)
    except (GhostBudgetError, _gsp.GhostPromptError) as exc:
        return False, str(exc)
    if final["measured"] and final["positive_clip_tokens"] > int(target_tokens):
        return False, ("candidate measures %d installed SD1 tokens, over the "
                       "%d author-time target"
                       % (final["positive_clip_tokens"], int(target_tokens)))
    return True, ""


def assert_shell_fits(styles, *, ledger_meta=None, token_measure_fn=None):
    """Prove every shipped pack/mode/longest-motif shell can hold a leaf.

    Run BEFORE any model call. A failure here is a COMPOSER CONSTANT DEFECT --
    a pack cue, a mode law or a motif that leaves no room for the shortest
    legal fallback -- and no number of model retries can fix it, so it must not
    be discovered as a retry loop at author time.
    """
    longest_motif = max(
        [motif_for_character(
            {"silhouette": "towering", "costume": "charcoal overcoat",
             "prop": "briefcase"}, mode)
         for mode in GHOST_MODES]
        + list(GHOST_BOOKEND_MOTIFS.values()), key=len)
    shortest_leaf = min(
        [c for pool in GHOST_FALLBACK_CLAUSES.values() for c in pool]
        + list(GHOST_FALLBACK_BOOKENDS.values()), key=len)
    failures = []
    for style in styles:
        for mode in GHOST_MODES:
            role = ("character_video" if mode == "figure"
                    else "announcer_visual")
            ok, reason = candidate_fits(
                role=role, style=style, mode=mode,
                motif_cue=longest_motif, drawable_beat=shortest_leaf,
                ledger_meta=ledger_meta, token_measure_fn=token_measure_fn)
            if not ok:
                failures.append("%s/%s: %s" % (
                    getattr(style, "style_id", "?"), mode, reason))
    if failures:
        raise GhostBudgetError(
            "Ghost v2 mechanical shell does not fit for %s -- this is a "
            "composer constant defect, not a model retry"
            % "; ".join(failures))


__all__ = [
    "GHOST_AUTHOR_SCHEMA_VERSION", "GHOST_AUTHOR_VERSION", "GHOST_MODES",
    "GHOST_NON_FIGURE_MODES", "GHOST_AUTHOR_SOURCES", "GHOST_LEAF_MAX_CHARS",
    "GHOST_LEAF_MIN_WORDS", "GHOST_LEAF_MAX_WORDS", "GHOST_CLIP_WINDOW_TOKENS",
    "GHOST_AUTHOR_TOKEN_TARGET", "GHOST_BATCH_TEMPERATURE",
    "GHOST_BATCH_BASE_TOKENS", "GHOST_BATCH_PER_SHOT_TOKENS",
    "GHOST_BATCH_RULES", "GHOST_BATCH_EXAMPLES",
    "GHOST_TEMPLATE_SHA256", "GHOST_REQUEST_HASH_KEYS", "GHOST_PROMPT_FIELDS",
    "GHOST_CHARACTER_CYCLE",
    "GHOST_DETERMINISTIC_MODEL_ID",
    "GHOST_BOOKEND_MOTIFS", "GHOST_FALLBACK_CLAUSES",
    "GHOST_FALLBACK_BOOKENDS", "GHOST_ROLES", "GHOST_ARC_CUES",
    "GHOST_CLIP_COUNTER",
    "GhostAuthorError", "GhostAuthorParseError", "GhostAuthorValidationError",
    "GhostTokenizerUnavailable", "GhostBudgetError",
    "batch_output_tokens", "normalize_role", "strip_cast_names",
    "sanitize_intent", "normalize_emotion", "map_arc", "schedule_ghost_modes",
    "motif_for_character", "motif_for_bookend", "opaque_id",
    "build_ghost_author_specs", "request_sha256", "output_sha256",
    "build_batch_prompt", "parse_batch_response", "validate_drawable_beat",
    "deterministic_leaf", "deterministic_batch", "build_ghost_prompt_object",
    "validate_ghost_prompt_object", "measure_clip_tokens",
    "resolve_token_measure", "finalize_ghost_prompt_v2", "candidate_fits",
    "finalize_ghost_prompt_v3", "resolve_crux_kernel", "resolve_world_light",
    "resolve_world_motion", "GHOST_WORLD_MOTION_V3", "GHOST_V3_DROP_ORDER",
    "GHOST_V3_TERM_LIMIT", "GHOST_V3_KERNEL_MAX_WORDS",
    "assert_shell_fits",
]
