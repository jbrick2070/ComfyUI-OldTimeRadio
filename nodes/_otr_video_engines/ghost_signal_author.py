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
import os
import re
from typing import Callable, Optional

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

#: The three coordinated representations. ``figure`` is the only one that may
#: show a body, and even then as a silhouette rather than a likeness.
GHOST_MODES = ("figure", "object", "signal")

#: The two representations a bookend may take. A radio console is not a person.
GHOST_NON_FIGURE_MODES = ("object", "signal")

#: How a stored leaf came to exist. There is no ``fallback`` boolean: a
#: deliberate deterministic result is ``deterministic_fallback`` plus a nonempty
#: reason, and reuse of one keeps BOTH -- so a replay can never launder a
#: fallback into proof eligibility.
GHOST_AUTHOR_SOURCES = ("writer_llm", "replay", "deterministic_fallback")

#: Leaf shape. The model is TOLD 6--10 words; the hard bounds are wider so a
#: good answer one word outside the request is not thrown away.
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
    "  mode=object   -> the shutter tilts and a slow shadow crosses its slats\n"
    "  mode=signal   -> banded static tightens around one bright point and "
    "opens\n"
    "  mode=figure   -> an outline lifts the spool into a narrowing shaft of "
    "light\n"
    "BAD, never write these:\n"
    "  tension builds        (not a thing a picture can show)\n"
    "  finality sets in      (not a thing a picture can show)\n"
    "  broadcast begins      (too short, and nothing is drawn)\n"
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
    "4. Name a CONCRETE thing and what it does. A mood, a feeling or an "
    "abstract state is not drawable.\n"
    "5. Write only what is visible. No names, no speech, no captions, no "
    "lettering, no camera or lens words, no quality words.\n"
    "6. Never write what is absent. Say what IS in the frame.\n"
    "7. mode=figure may show a body as a silhouette. mode=object shows one "
    "isolated thing. mode=signal shows light, static, shadow or waveform "
    "and no person at all.\n"
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
_LETTERING_WORDS = frozenset({
    "text", "texts", "word", "words", "letter", "letters", "lettering",
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
_HUMAN_WORDS = frozenset({
    "person", "people", "man", "men", "woman", "women", "boy", "girl",
    "child", "children", "crowd", "figure", "figures", "silhouette",
    "silhouettes", "face", "faces", "head", "heads", "portrait", "eye",
    "eyes", "mouth", "hand", "hands", "arm", "arms", "shoulder", "shoulders",
    "body", "bodies", "someone", "somebody", "he", "she", "they",
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

    CHARACTER BEATS CYCLE ``figure -> object -> signal`` from a hashed offset.
    An unmodified cycle of period three satisfies the operator's quota by
    construction (two of every three character clips are non-figure) and can
    never produce a run, so no correction is ever applied to a character.

    BOOKENDS alternate ``object``/``signal`` from their own hashed offset. Only
    a bookend may be corrected, and only when it would create a THIRD identical
    mode in a row -- which keeps clip planning stable and keeps the correction
    away from the assignments the recurrence design depends on.
    """
    seed = _hash_int(episode_seed, "ghost_mode_schedule", GHOST_AUTHOR_VERSION)
    char_offset = seed % len(GHOST_MODES)
    bookend_offset = (seed >> 8) % len(GHOST_NON_FIGURE_MODES)

    out = {}
    scheduled = []
    char_n = 0
    bookend_n = 0
    for beat_id, role in entries:
        beat_id = str(beat_id)
        if normalize_role(role) == "character_video":
            mode = GHOST_MODES[(char_offset + char_n) % len(GHOST_MODES)]
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
    "silhouette": ("lean", "broad", "slight", "tall", "compact", "angular"),
    "prop": ("lantern", "key", "ledger", "satchel", "chart", "telegraph"),
}

#: Bookend motifs, by ``(role, mode)``. Compact radio anchors that recur while
#: the unchanged pack cue keeps supplying the anime / archive / material look.
#: A Ghost bookend with an empty motif is invalid, which is why these are
#: constants rather than an optional derivation.
GHOST_BOOKEND_MOTIFS = {
    ("announcer_visual", "object"): "radio dial emblem",
    ("announcer_visual", "signal"): "radio dial signal",
    ("music_visual", "object"): "broadcast console emblem",
    ("music_visual", "signal"): "broadcast waveform signal",
}


def _first_allowlisted(phrases, vocabulary) -> str:
    """The first allowlisted WORD found across ``phrases``, in source order.

    A WORD, never the phrase it came from: copying the phrase is how a cast
    row's prose -- and its landmarks, its hair, its jaw -- would get back into
    a prompt that is supposed to have left all of that behind.
    """
    allowed = tuple(vocabulary)
    for phrase in phrases:
        words = _WORD_RE.findall(str(phrase or "").lower())
        for candidate in allowed:
            if candidate in words:
                return candidate
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
    if mode == "figure":
        # "an olive key", not "a olive key". The motif is prompt text a person
        # reads in a receipt and a sampler reads as language; a broken article
        # is a small thing that makes both of them worse.
        article = "an" if tokens["colour"][:1] in "aeiou" else "a"
        return "%s silhouette with %s %s %s" % (
            tokens["silhouette"], article, tokens["colour"], tokens["prop"])
    if mode == "object":
        return "%s %s emblem" % (tokens["colour"], tokens["prop"])
    if mode == "signal":
        return "%s %s signal" % (tokens["colour"], tokens["prop"])
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
        "a lone outline turns slowly into a widening shaft of light",
        "an outline steps forward as the floor brightens beneath it",
        "an outline lifts one arm and the shadow swings across the wall",
        "an outline leans in while the light narrows to a single band",
        "an outline straightens as a slow glow rises behind it",
        "an outline turns away and the light closes behind it",
    ),
    "object": (
        "the emblem tilts as a slow shadow crosses its face",
        "the emblem catches a rising glow along one worn edge",
        "the emblem rocks once and settles into deeper shadow",
        "the emblem brightens while fine dust drifts across it",
        "the emblem turns a quarter and the highlight slides off",
        "the emblem sinks slowly out of a narrowing pool of light",
    ),
    "signal": (
        "bands of static crush inward and open again",
        "a slow ripple crosses the field and fades to a flat glow",
        "concentric rings tighten toward a single bright point",
        "a bright seam splits the field and drifts apart",
        "the field pulses once and settles into drifting grain",
        "long shadows sweep across the field and thin away",
    ),
}

#: The bookends get their own phase-specific clauses so an episode does not
#: open and close on the same picture, which is what a shared pack register
#: produced on every v1 episode.
GHOST_FALLBACK_BOOKENDS = {
    ("opening", "object"): "the emblem lights from cold to warm and holds",
    ("opening", "signal"): "a single bright line opens into a spreading field",
    ("closing", "object"): "the emblem dims slowly and the highlight leaves it",
    ("closing", "signal"): "the field narrows to one fading horizontal line",
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
    for step in range(len(pool)):
        candidate = pool[(start + step) % len(pool)]
        if candidate not in used:
            return candidate
    return pool[start]


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
    return os.environ.get("OTR_TEST_MODE") == "1"


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
    except GhostTokenizerUnavailable:
        if _test_mode():
            return None
        raise
    return measure_clip_tokens


#: The measurer's own identity, stamped beside every count so a receipt says
#: which counter produced it.
GHOST_CLIP_COUNTER = "comfy.sd1_clip.SD1Tokenizer/%s" % GHOST_AUTHOR_VERSION


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
    "assert_shell_fits",
]
