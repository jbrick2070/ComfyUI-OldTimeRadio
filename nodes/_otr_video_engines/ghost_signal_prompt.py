"""``ghost_signal_prompt`` -- the PURE prompt authority for the Ghost Signal lane.

Ghost Signal (``animatediff15_video``) owns its subject entirely: it declares
``accepts_still = False``, so no still is ever minted for it and no image
carries the look. Everything the viewer sees is decided by the text in this
module, which makes this file the lane's whole visual vocabulary.

PURE BY CONTRACT. No I/O, no model load, no LLM call, no ledger mutation, no
import of ``render_driver``. Everything here is a deterministic function of its
arguments, so two identical inputs produce byte-identical output and the whole
surface is testable on a CPU box with nothing installed. The render driver
resolves the live authorities (style pack, motion clause, pack register) and
hands the resolved values in; this module never reaches back out for them.
That direction is deliberate -- the driver already owns the role-to-register
mapping, and a second copy here is a table that drifts once.

THE COMPOSER'S ORDER IS IMMUTABLE (plan section 6.2). An empty slot is omitted
without reordering the others:

  1. pack cue          the selected visual-style pack's own look token
  2. subject identity  the durable sigil / announcer face / console emblem
  3. framing lock      the role's shot floor
  4. beat action       the authored motion clause, or a deterministic fallback
  5. emotion           the beat's recorded mood
  6. story accent      the beat's arc phase, one phrase
  7. shot law          the affirmative closing directive

WHY A SIGIL AND NOT A DESCRIPTION. SD1.5 at 512x288 across a sliding 16-frame
context will not hold a face. Asking it to is how a lane ships a character who
is a different person every beat. So Ghost does not ask: it distills each cast
member into four HERALDIC cues -- silhouette, one asymmetrical landmark, one
costume colour or item, one handheld prop -- which are the properties a small
model CAN hold across a timeline. The result reads as the same figure beat to
beat because the figure is a shape and a colour, not a likeness.

Character drift is ACCEPTED on this project (operator, 2026-08-14). The sigil is
not an attempt to reopen that; it is what makes a prompt-only lane legible at
all when there is no still to inherit identity from.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

# --------------------------------------------------------------------------- #
# Identity + budgets. Every one of these is pinned by tests.
# --------------------------------------------------------------------------- #

#: The prompt-profile token the adapter declares and the driver branches on.
#: A capability string, never an engine-name comparison.
GHOST_PROMPT_PROFILE = "ghost_signal_v1"

#: Stamped into observability as ``prompt_version`` so a published receipt says
#: which composer wrote the text. Bump with the composer, never in place.
GHOST_PROMPT_VERSION = "ghost_signal_v1"

#: ``prompt_source`` for :func:`render_driver._stamp_prompt_meta`.
GHOST_PROMPT_SOURCE = "ghost_signal"

#: THE HARD CEILING. Ghost owns it (the LTX 240 is a different lane's number and
#: is never touched). 320 leaves the banana route room to substitute one prop
#: without the common funnel ever having to cut a protected clause.
GHOST_PROMPT_MAX_CHARS = 320

#: The normal operating band. Not a gate -- a target the composer aims at so the
#: banana headroom above is real rather than nominal.
GHOST_PROMPT_TARGET_LOW = 260
GHOST_PROMPT_TARGET_HIGH = 280

#: The negative's own ceiling, same number, same phrase-safe rule.
GHOST_NEGATIVE_MAX_CHARS = 320

#: The sigil's own ceiling. It is the one slot that may never be trimmed by the
#: composer, so it has to be bounded where it is BUILT.
GHOST_SIGIL_MAX_CHARS = 110

#: The floor a shrinkable PACK surface may not go below. Under roughly this
#: length a motion register stops describing a movement and a console subject
#: stops describing a thing, so the composer would rather fail loudly than ship
#: a stub. Reached only if a future pack authors surfaces far longer than the
#: nine shipped ones.
GHOST_SHRINK_FLOOR_CHARS = 55

#: LANE HYGIENE, and it leads the negative unconditionally. SD1.5 volunteers
#: lettering into any scene that smells like a sign, a poster or a radio dial --
#: which is most of this show. These five phrases are retained whole even when
#: the pack negative has to be cut short.
LANE_HYGIENE_NEGATIVE = (
    "text", "watermark", "caption", "lettering", "subtitles",
)

#: The affirmative closing directive. AFFIRMATIVE because the negative channel
#: is live on this lane (cfg 8.0 with a real unconditional branch), so
#: exclusions belong in the negative and the positive states what to draw.
GHOST_SHOT_LAW = "steady legible silhouette, one clear action, unbroken shot"

#: Role framing floors. The character floor is a MID-SHOT: closer than that and
#: the model starts trying to render a face it cannot hold.
GHOST_FRAMING = {
    "character_video": "mid-shot or wider, whole figure legible",
    "announcer_visual": "medium shot of the console, dial face centered",
    "music_visual": "wide abstract frame, no people",
}

# --------------------------------------------------------------------------- #
# Sigil vocabularies. Checked in, pinned by tests, deliberately small.
#
# These are matched as WHOLE WORDS against a candidate phrase. A bucket takes
# the FIRST phrase that matches it, in the source's own order, so a cast row
# that leads with its strongest cue keeps that cue.
# --------------------------------------------------------------------------- #

#: Bucket 1 -- silhouette / body shape. The single most durable property across
#: a sliding context: a tall thin figure stays a tall thin figure.
SIGIL_SILHOUETTE_WORDS = frozenset({
    "tall", "short", "broad", "slight", "stooped", "lean", "heavyset",
    "wiry", "angular", "round", "towering", "compact", "gaunt", "stocky",
    "willowy", "hulking", "slender", "squat", "burly", "spare", "massive",
    "diminutive", "rangy", "thickset", "bent", "straight-backed", "narrow",
})

#: Bucket 2 -- one asymmetrical landmark. Asymmetry is what stops a figure
#: reading as a generic body: an eyepatch has a side, a scar has a side.
SIGIL_LANDMARK_WORDS = frozenset({
    "scar", "eyepatch", "patch", "limp", "cane", "sling", "bandage",
    "crooked", "missing", "streak", "burn", "brand", "tattoo", "birthmark",
    "notch", "cropped", "shaved", "braid", "topknot", "moustache", "mustache",
    "beard", "spectacles", "monocle", "goggles", "scarred", "one-eyed",
    "bald", "greying", "grey-streaked", "silver-streaked", "hooked",
})

#: Bucket 3 -- costume colour or item. Colour survives at 512x288 when almost
#: nothing else does, so a coat's colour is worth more here than its cut.
SIGIL_COSTUME_WORDS = frozenset({
    "coat", "overcoat", "trenchcoat", "greatcoat", "jacket", "uniform",
    "cloak", "cape", "gown", "dress", "suit", "waistcoat", "vest", "apron",
    "shawl", "robe", "hat", "fedora", "cap", "helmet", "hood", "scarf",
    "gloves", "boots", "collar", "tie", "sash", "belt", "armour", "armor",
    "black", "white", "red", "green", "blue", "brown", "grey", "gray",
    "crimson", "scarlet", "navy", "olive", "amber", "gold", "golden",
    "silver", "rust", "ochre", "violet", "purple", "cream", "tan", "charcoal",
})

#: Bucket 4 -- one handheld prop. A prop gives the figure something to DO,
#: which is what turns a portrait into a shot with motion in it.
#:
#: The source's own vocabulary is carried as written (operator directive
#: 2026-08-03): a revolver in the source is a revolver here.
SIGIL_PROP_WORDS = frozenset({
    "lantern", "lamp", "torch", "candle", "revolver", "pistol", "rifle",
    "knife", "blade", "sword", "dagger", "book", "ledger", "journal",
    "letter", "map", "chart", "cane", "staff", "stick", "umbrella",
    "satchel", "bag", "case", "briefcase", "pipe", "cigarette", "flask",
    "bottle", "glass", "cup", "key", "keys", "rope", "chain", "hammer",
    "wrench", "microphone", "headset", "telegraph", "camera", "watch",
    "pocketwatch", "spyglass", "telescope", "basket", "crate", "lockbox",
})

#: Ordered bucket table: (bucket name, vocabulary). The ORDER here is the order
#: the four cues are emitted in, and it is fixed.
SIGIL_BUCKETS = (
    ("silhouette", SIGIL_SILHOUETTE_WORDS),
    ("landmark", SIGIL_LANDMARK_WORDS),
    ("costume", SIGIL_COSTUME_WORDS),
    ("prop", SIGIL_PROP_WORDS),
)

#: NEUTRAL CUE POOLS -- the deterministic fill for a bucket the cast row never
#: supplied. These are DELIBERATE HERALDIC CUES, not a hidden model fallback:
#: the selection is a sha256 over (episode_seed, char_id, style_id, source),
#: so it is stable for a character across every beat of an episode and varies
#: between characters. Nothing is generated and nothing is asked of a model.
SIGIL_NEUTRAL_POOLS = {
    "silhouette": (
        "a lean upright figure", "a broad steady figure",
        "a slight quick figure", "a tall stooped figure",
        "a compact square-shouldered figure", "an angular restless figure",
    ),
    "landmark": (
        "one shoulder held higher", "a pale streak through dark hair",
        "a bound left hand", "a heavy squared jaw",
        "a close-cropped head", "a deep-set shadowed brow",
    ),
    "costume": (
        "a charcoal coat", "a rust-red jacket", "an olive uniform",
        "a deep blue overcoat", "a cream collarless shirt",
        "a black shawl",
    ),
    "prop": (
        "carrying a shuttered lantern", "holding a folded chart",
        "gripping a worn leather satchel", "turning a brass key",
        "shouldering a coil of rope", "steadying a heavy ledger",
    ),
}

#: NEUTRAL KINETIC ACTIONS -- the last-resort action fill, same hash domain.
#: Reached only at cast-time preflight on a beat with no clause and no intent.
GHOST_NEUTRAL_ACTIONS = (
    "turns slowly toward the light",
    "steps forward one pace",
    "lifts a hand and holds it still",
    "leans in and settles",
    "shifts weight and looks away",
    "draws breath and straightens",
)

#: BEAT-INTENT -> ACTION. A small checked-in table, because an intent word is
#: not a camera instruction and handing "reveal" to a diffusion model as-is
#: gets a picture of the word, not the movement.
GHOST_INTENT_ACTIONS = {
    "reveal": "turns to reveal what was hidden",
    "confront": "squares up and holds ground",
    "accuse": "levels a pointed hand",
    "plead": "reaches out open-handed",
    "warn": "raises a hand sharply",
    "comfort": "leans in and softens",
    "explain": "gestures evenly while speaking",
    "question": "tilts the head and waits",
    "threaten": "advances one slow step",
    "retreat": "backs away and turns",
    "decide": "stops still and sets the jaw",
    "grieve": "bows the head and stills",
    "celebrate": "lifts both arms loosely",
    "escape": "breaks into a fast stride",
    "search": "sweeps a look across the space",
    "arrive": "enters and pauses at the threshold",
    "depart": "turns away and walks out of frame",
    "recall": "goes still and looks past the frame",
    "command": "gestures once, sharp and final",
    "submit": "lowers the shoulders and yields",
}

# --------------------------------------------------------------------------- #
# Phrase machinery (pure).
# --------------------------------------------------------------------------- #

#: Phrases carrying these are DISCARDED from a sigil source. A camera
#: instruction, a medium name, a backdrop or a second-person address is not a
#: property of the person, and copying one into the sigil pins the WRONG thing
#: for every beat of the episode.
_SIGIL_REJECT_WORDS = frozenset({
    # camera / framing
    "shot", "close-up", "closeup", "framing", "frame", "camera", "lens",
    "angle", "zoom", "pan", "tilt", "dolly", "crop", "headroom", "centered",
    "composition", "portrait", "three-quarter", "bokeh", "depth",
    # medium / rendering
    "photo", "photograph", "photographic", "painting", "illustration",
    "render", "rendering", "cinematic", "film", "lighting", "lit", "grain",
    "8k", "4k", "hdr", "detailed", "masterpiece", "quality", "style",
    # background / setting
    "background", "backdrop", "behind", "scene", "set", "room", "studio",
    "wall", "sky", "street", "interior", "exterior", "environment",
    # second person
    "you", "your", "yours",
})

#: Function words a trimmed phrase may not END on. Checked in and pinned, so a
#: cut lands on a complete thought rather than an unfinished list.
_DANGLING_TAIL_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on",
    "at", "by", "for", "from", "with", "into", "onto", "over", "under",
    "that", "which", "who", "whose", "as", "its", "their", "his", "her",
    "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "--", "-", "",
})

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_PHRASE_SPLIT_RE = re.compile(r"[,;.!?]+")
_WS_RE = re.compile(r"\s+")


def _normalize_ws(text) -> str:
    """Collapse all whitespace runs to single spaces and strip. Pure."""
    return _WS_RE.sub(" ", str(text or "")).strip()


def _split_phrases(text) -> list:
    """Split a source into complete comma / semicolon / sentence phrases.

    Order is preserved -- the buckets below take the FIRST match, so a cast row
    that leads with its strongest cue keeps it."""
    out = []
    for chunk in _PHRASE_SPLIT_RE.split(_normalize_ws(text)):
        phrase = chunk.strip(" -_")
        if phrase:
            out.append(phrase)
    return out


def _words(phrase) -> set:
    """The lowercase word set of a phrase, for whole-word bucket matching."""
    return set(_WORD_RE.findall(str(phrase or "").lower()))


def _strip_name(phrase, name) -> str:
    """Remove the cast NAME from a phrase.

    The name is never emitted: a proper noun in the prompt is a request for a
    specific person the model has no idea about, and on the adaptation lanes it
    is also a metadata leak into the picture."""
    name = _normalize_ws(name)
    if not name:
        return phrase
    out = phrase
    for token in [name] + name.split():
        token = token.strip()
        if len(token) < 3:
            continue
        out = re.sub(r"\b%s\b" % re.escape(token), "", out, flags=re.IGNORECASE)
    return _normalize_ws(out).strip(" ,;-")


def _sigil_hash(episode_seed, char_id, style_id, source_text) -> int:
    """The ONE hash domain for every deterministic Ghost choice.

    Pinned by tests. Includes the source text so two characters with identical
    seeds and ids but different cast rows do not collide into one figure."""
    payload = "|".join((
        str(episode_seed), str(char_id or ""), str(style_id or ""),
        str(source_text or ""),
    ))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _pick(pool, seed_int, salt) -> str:
    """Deterministically choose one entry from a checked-in pool.

    THE SALT IS HASHED WITH THE SEED, NOT ADDED TO IT, and that is a
    correctness fix rather than a style preference. Adding a per-bucket offset
    to one shared seed makes every bucket's choice the SAME function of
    ``seed_int mod len(pool)`` -- so two characters whose seeds happen to be
    congruent mod 6 would receive an identical sigil in ALL FOUR buckets at
    once, not one bucket in six. That collapses the odds of two sparse
    characters being visually indistinguishable from (1/6)^4 to 1/6, which on a
    lane whose entire identity mechanism is these four cues is a real defect.
    Hashing them together makes the four selections independent.
    """
    pool = tuple(pool)
    if not pool:
        return ""
    mixed = int(hashlib.sha256(
        ("%d|%s" % (int(seed_int), salt)).encode("utf-8")).hexdigest()[:16], 16)
    return pool[mixed % len(pool)]


def _trim_to(text, ceiling) -> str:
    """Phrase-aware trim: drop WHOLE trailing comma phrases, then whole words.

    TWO STAGES, AND THE SECOND ONE IS LOAD-BEARING. Dropping comma phrases is
    the preferred cut because it keeps every surviving clause intact. But the
    visual-style packs author their subject and motion-register surfaces as
    long, largely COMMA-FREE prose -- 178 and 209 characters on `recur_frac` --
    and against those a comma-only trimmer is a NO-OP: it finds one phrase,
    cannot drop it, and returns the text unchanged at its original length.

    That is not hypothetical. It is what took the first live Ghost leg down:
    an announcer beat composed to 344 characters, the trimmer returned it
    untouched, and the composer raised rather than shipping an over-budget
    prompt. Falling back to a WORD boundary keeps the "never mid-word" promise
    while making the function actually able to do its job.
    """
    text = _normalize_ws(text)
    ceiling = int(ceiling)
    if len(text) <= ceiling:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    while len(parts) > 1:
        parts.pop()
        candidate = ", ".join(parts)
        if len(candidate) <= ceiling:
            return candidate
    head = (parts[0] if parts else "").strip()
    if len(head) <= ceiling:
        return head
    words = head.split()
    kept = []
    for word in words:
        candidate = " ".join(kept + [word])
        if len(candidate) > ceiling:
            break
        kept.append(word)
    # NEVER END ON A DANGLING FUNCTION WORD. A word-boundary cut is honest but
    # it can still land on "...two nested dial-eyes and a", which reads to the
    # model as an unfinished list and is the kind of trailing fragment that
    # invites it to invent whatever came next. Dropping the dangling tail is
    # the same class of rule as "never mid-word": both are about handing the
    # sampler a complete thought.
    while kept and kept[-1].strip(",.;:-").lower() in _DANGLING_TAIL_WORDS:
        kept.pop()
    # A single word longer than the whole ceiling is the only case with no
    # word boundary to cut on; return it rather than the empty string, and let
    # the caller's own ceiling assertion be the thing that fails.
    return " ".join(kept).rstrip(" ,;:-") or words[0]


# --------------------------------------------------------------------------- #
# The durable subject sigil.
# --------------------------------------------------------------------------- #

#: Cast-row source priority. ``name`` is present only as a SIGNAL that the row
#: is sparse; the name itself is never emitted (see :func:`_strip_name`).
SIGIL_SOURCE_FIELDS = (
    "portrait_prompt", "appearance", "description", "character_description",
)


def distill_subject_sigil(cast_row, *, episode_seed, char_id, style_id) -> str:
    """One durable heraldic identity for a character, as four ordered cues.

    Deterministic and CREDIT-FREE: no model is asked anything, and the only
    non-source input is a sha256 over ``(episode_seed, char_id, style_id,
    source_text)``. The same character therefore carries the same sigil across
    every beat of an episode, and a different one in the next episode.

    A missing cast row is legal and yields a fully pooled sigil -- it never
    reaches for the wardrobe writer or any other author. (Reading the raw row
    rather than ``_appearance_for_char`` is the point: that helper may invoke
    the optional wardrobe writer, which would turn a deterministic identity
    read into a hidden mutation and a credit spend.)
    """
    row = cast_row if isinstance(cast_row, dict) else {}
    name = str(row.get("name") or "")

    source_text = ""
    for field in SIGIL_SOURCE_FIELDS:
        value = _normalize_ws(row.get(field))
        if value:
            source_text = value
            break

    seed_int = _sigil_hash(episode_seed, char_id, style_id, source_text)

    # Candidate phrases: name removed, camera / medium / backdrop / second
    # person discarded. A phrase that empties out after the name is gone was
    # only ever the name.
    candidates = []
    for phrase in _split_phrases(source_text):
        cleaned = _strip_name(phrase, name)
        if not cleaned:
            continue
        if _words(cleaned) & _SIGIL_REJECT_WORDS:
            continue
        candidates.append(cleaned)

    cues = []
    used = set()
    for bucket, vocabulary in SIGIL_BUCKETS:
        chosen = ""
        for index, phrase in enumerate(candidates):
            if index in used:
                continue
            if _words(phrase) & vocabulary:
                chosen = phrase
                used.add(index)
                break
        if not chosen:
            chosen = _pick(SIGIL_NEUTRAL_POOLS[bucket], seed_int, bucket)
        chosen = _normalize_ws(chosen).strip(" ,;-")
        if chosen:
            cues.append(chosen)

    # Explicit gender ONLY. `normalize_gender` maps absence to "other", so
    # calling it on a blank field would invent a claim the cast row never made
    # -- and a wrong gender on a figure the viewer sees every beat is exactly
    # the correctness class the operator kept open when story quality closed.
    gender_word = ""
    raw_gender = str(row.get("gender") or "").strip()
    if raw_gender:
        try:
            from .._otr_roster_gender import normalize_gender  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_roster_gender import normalize_gender  # type: ignore
        normalized = normalize_gender(raw_gender)
        if normalized in ("male", "female"):
            gender_word = "a man" if normalized == "male" else "a woman"

    body = ", ".join(cues)
    sigil = ("%s, %s" % (gender_word, body)) if gender_word else body
    return _trim_to(sigil, GHOST_SIGIL_MAX_CHARS)


# --------------------------------------------------------------------------- #
# Action resolution (pure, ordered, deterministic).
# --------------------------------------------------------------------------- #

def resolve_action(role, *, motion_clause=None, pack_motion_fallback="",
                   beat_intent="", sigil_seed=0) -> str:
    """The beat's action slot, resolved in the plan's exact order.

    ``resolve_motion_clause_text`` legitimately returns ``None`` when the
    optional motion pass is off, when the stored object is a fallback, or when
    only the cast-time temporary shot exists. That is not permission to emit an
    empty action -- a Ghost beat with no action is a still frame with a
    sliding context wasted on it. So:

    1. the whole validated motion clause when present;
    2. otherwise, for announcer / music, the pack's own role register;
    3. otherwise, for a character, the beat intent through the checked-in table
       (an unmapped non-empty intent becomes a bounded ``moves with ...``);
    4. otherwise one neutral kinetic action on the sigil hash domain.

    Every step is pack- or ledger-derived and credit-free.
    """
    clause = _normalize_ws(motion_clause)
    if clause:
        return clause

    if role in ("announcer_visual", "music_visual"):
        register = _normalize_ws(pack_motion_fallback)
        if register:
            return register

    intent = _normalize_ws(beat_intent).lower()
    if intent:
        mapped = GHOST_INTENT_ACTIONS.get(intent)
        if mapped:
            return mapped
        # Unmapped but present: bound it rather than pass an unknown noun into
        # the picture. Six words is enough to carry a movement and short enough
        # that a runaway intent string cannot eat the budget.
        words = _WORD_RE.findall(intent)[:6]
        if words:
            return "moves with %s" % " ".join(words)

    return _pick(GHOST_NEUTRAL_ACTIONS, int(sigil_seed), "action")


# --------------------------------------------------------------------------- #
# The negative.
# --------------------------------------------------------------------------- #

def compose_ghost_negative(style) -> str:
    """Lane hygiene first, then the pack's own ``effective_negative``.

    The hygiene head is retained COMPLETE and the pack phrases are added in
    authored order while the next WHOLE phrase fits. A phrase is never cut in
    half, and this budget is a character budget -- it makes no claim about a
    tokenizer.
    """
    parts = list(LANE_HYGIENE_NEGATIVE)
    seen = {p.lower() for p in parts}

    pack_negative = ""
    if style is not None:
        try:
            from .._otr_visual_styles import effective_negative  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_visual_styles import effective_negative  # type: ignore
        pack_negative = str(effective_negative(style) or "")

    head = ", ".join(parts)
    for chunk in pack_negative.split(","):
        phrase = chunk.strip()
        if not phrase or phrase.lower() in seen:
            continue
        candidate = "%s, %s" % (head, phrase)
        if len(candidate) > GHOST_NEGATIVE_MAX_CHARS:
            break
        head = candidate
        seen.add(phrase.lower())
    return head


# --------------------------------------------------------------------------- #
# The composer.
# --------------------------------------------------------------------------- #

#: Slots the trimmer may drop, in the order it drops them. The subject sigil,
#: the authored motion clause, the mid-shot floor and the shot law are ABSENT
#: from this tuple on purpose: they are the four things the lane exists to
#: guarantee, so if they alone exceed the ceiling that is a composer defect to
#: fail on, not a prompt to quietly truncate.
GHOST_TRIM_ORDER = ("story_accent", "emotion", "framing")


class GhostPromptError(ValueError):
    """The protected Ghost slots alone exceed the hard ceiling."""


def compose_ghost_prompt(*, role, style, subject_sigil="", motion_clause=None,
                         pack_motion_fallback="", beat_intent="", emotion="",
                         story_accent="", open_subject="", sigil_seed=0) -> dict:
    """Compose one Ghost positive prompt plus its negative.

    Returns ``{"positive", "negative", "slots"}`` where ``slots`` is the ordered
    list of slot names actually emitted -- the receipt that says which of the
    seven were present, and the thing a trace row carries so a published episode
    can be asked what its prompt was made of.

    ``role`` is the engine capability token (``character_video`` /
    ``announcer_visual`` / ``music_visual``).

    Never consumes raw dialogue, the episode title, the M4 scene wall, a second
    person, or proper-noun metadata: none of those are parameters, which is the
    only way to guarantee it.
    """
    role = str(role or "")
    slots = []
    pieces = []

    # 2. SUBJECT IDENTITY -- role-owned, and the one slot that differs most.
    subject = ""
    if role == "character_video":
        subject = _normalize_ws(subject_sigil)
    elif role == "announcer_visual":
        # An anthropomorphic console host: the pack's own face surface. No
        # lip-sync is promised anywhere on this lane.
        subject = _normalize_ws(getattr(style, "announcer_subject_face", ""))
    elif role == "music_visual":
        # No human at all. The pack's console/emblem object, or the resolved
        # open subject the driver passes in.
        subject = (_normalize_ws(getattr(style, "announcer_subject_object", ""))
                   or _normalize_ws(open_subject))
    if subject:
        pieces.append(("subject", subject))
        slots.append("subject")

    # 3. FRAMING LOCK.
    framing = GHOST_FRAMING.get(role, "")
    if framing:
        pieces.append(("framing", framing))
        slots.append("framing")

    # 4. ACTION / MOTION.
    action = resolve_action(
        role, motion_clause=motion_clause,
        pack_motion_fallback=pack_motion_fallback,
        beat_intent=beat_intent, sigil_seed=sigil_seed)
    authored = bool(_normalize_ws(motion_clause))
    if action:
        pieces.append(("action", action))
        slots.append("action")

    # 5. EMOTION -- the beat's recorded mood, one phrase.
    emotion_text = _trim_to(_normalize_ws(emotion), 48)
    if emotion_text:
        pieces.append(("emotion", "%s mood" % emotion_text))
        slots.append("emotion")

    # 6. ONE STORY ACCENT -- the beat's arc phase, one phrase.
    accent = _trim_to(_normalize_ws(story_accent), 48)
    if accent:
        pieces.append(("story_accent", accent))
        slots.append("story_accent")

    # 7. SHOT LAW -- affirmative, always last, never trimmed.
    pieces.append(("shot_law", GHOST_SHOT_LAW))
    slots.append("shot_law")

    body = _join(pieces)

    # 1. PACK CUE -- prepended, so it lands FIRST in the emitted order.
    positive = _prefix_pack_cue(style, body)
    if positive != body:
        slots.insert(0, "pack_cue")

    # Trim only the droppable slots, in the published order, until it fits.
    for droppable in GHOST_TRIM_ORDER:
        if len(positive) <= GHOST_PROMPT_MAX_CHARS:
            break
        if droppable == "framing" and role == "character_video":
            # The mid-shot floor is protected on character beats: dropping it
            # is how this lane starts trying to render faces again.
            continue
        if not any(name == droppable for name, _ in pieces):
            continue
        pieces = [(name, text) for name, text in pieces if name != droppable]
        slots = [s for s in slots if s != droppable]
        body = _join(pieces)
        positive = _prefix_pack_cue(style, body)

    # STEP 4 OF THE PUBLISHED TRIM ORDER: shrink the DISTILLED REGISTER PHRASES.
    #
    # THIS IS WHERE THE BOOKENDS LIVE, and the numbers say why it is not
    # optional. A character beat composes from a sigil capped at 110 plus a
    # short action, and fits 320 with room to spare. A BOOKEND pulls two large
    # PACK surfaces instead: `announcer_subject_face` runs 163-178 characters
    # across the nine shipped packs and the `motion_registers` values run
    # 130-209 (the loader budgets them at 240). On `recur_frac` that is
    # 29 + 178 + 209 + 58 = 474 before separators, against a 320 ceiling.
    #
    # Both are pack-derived prose, which is exactly what step 4 names, and
    # NEITHER is a protected field: the protections are the character SIGIL, an
    # AUTHORED motion clause, the mid-shot floor and the shot law. A pack
    # register is a register -- shortening it is the sanctioned cut.
    #
    # Shrunk proportionally so no single surface is gutted while another keeps
    # its full length, and never below a floor where a direction stops being a
    # direction.
    shrinkable = []
    if not authored:
        shrinkable.append("action")
    if role != "character_video":
        # A bookend's subject is a PACK surface, never a distilled sigil.
        shrinkable.append("subject")
    for _pass in range(4):
        if len(positive) <= GHOST_PROMPT_MAX_CHARS or not shrinkable:
            break
        over = len(positive) - GHOST_PROMPT_MAX_CHARS
        share = -(-over // len(shrinkable))          # ceil, so a 1-char overflow still moves
        shrunk = []
        for name, text in pieces:
            if name in shrinkable:
                target = len(text) - share
                if target >= GHOST_SHRINK_FLOOR_CHARS:
                    text = _trim_to(text, target)
            shrunk.append((name, text))
        if shrunk == pieces:
            break                                     # nothing moved; stop rather than spin
        pieces = shrunk
        body = _join(pieces)
        positive = _prefix_pack_cue(style, body)

    if len(positive) > GHOST_PROMPT_MAX_CHARS:
        raise GhostPromptError(
            "Ghost Signal protected slots compose to %d chars, over the %d "
            "ceiling, with nothing droppable left (slots=%s). The composer or "
            "the sigil ceiling is wrong -- runtime truncation is NOT the "
            "preservation mechanism for the subject, the authored motion "
            "clause, the mid-shot floor or the shot law."
            % (len(positive), GHOST_PROMPT_MAX_CHARS, ",".join(slots)))

    return {
        "positive": positive,
        "negative": compose_ghost_negative(style),
        "slots": slots,
    }


def _join(pieces) -> str:
    """Join the ordered slot pieces into one comma-separated prompt."""
    return ", ".join(text for _, text in pieces if _normalize_ws(text))


def _prefix_pack_cue(style, prompt) -> str:
    """Front-anchor the pack's style token. ADDITIVE ONLY -- the shared
    ``prefix_style_cue`` authority, never a second copy of it here."""
    if style is None:
        return prompt
    try:
        from .._otr_visual_styles import prefix_style_cue  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_visual_styles import prefix_style_cue  # type: ignore
    return prefix_style_cue(style, prompt)


__all__ = [
    "GHOST_PROMPT_PROFILE", "GHOST_PROMPT_VERSION", "GHOST_PROMPT_SOURCE",
    "GHOST_PROMPT_MAX_CHARS", "GHOST_PROMPT_TARGET_LOW",
    "GHOST_PROMPT_TARGET_HIGH", "GHOST_NEGATIVE_MAX_CHARS",
    "GHOST_SIGIL_MAX_CHARS", "LANE_HYGIENE_NEGATIVE", "GHOST_SHOT_LAW",
    "GHOST_FRAMING", "GHOST_TRIM_ORDER", "GHOST_INTENT_ACTIONS",
    "GHOST_NEUTRAL_ACTIONS", "SIGIL_BUCKETS", "SIGIL_NEUTRAL_POOLS",
    "SIGIL_SOURCE_FIELDS", "GhostPromptError",
    "distill_subject_sigil", "resolve_action", "compose_ghost_negative",
    "compose_ghost_prompt",
]
