"""The ONE reader of a cast row's character text (operator, 2026-10-18).

A cast row carries its character as three fields, never one blended string:

* ``appearance`` -- what can be DRAWN: age, role, face geometry, the one
  distinctive detail. Stills, portraits and the Ghost sigil read only this.
* ``presence``   -- how the character carries the episode's pressure.
  Behaviour a VIDEO model can act on; LLM context.
* ``voice_cue``  -- the radio-performance cue, kept for traceability. No
  picture model and no line writer reads it (a text model can only obey a
  sound cue with "(whispering)"); voice CASTING reads
  ``meta.cast_voice_slots`` instead.

WHY THE SPLIT. The old ``character_description`` packed all three into one
labelled string ("... Face: ... Presence: ... Voice: ...") and every picture
consumer had to regex them apart. FLUX 3 Klein lettered the Voice and Presence
sentences into stills (PBUG-20261018-03) and three successive cleaners leaked;
the operator ruled the split "for traceability". The cleaner stays as a belt on
the picture readers here: a model can still put an unlabelled voice word into
``appearance``, and no prompt can forbid that absolutely.

EVERY SOURCE BANK fills the same three fields: the shared casting path
(original, media_archive, public_domain, shakespeare), my_story and
scifi_news_pro (their own casting passes), and the hand-written rows in
``config/cast_pools.py``. A field may be empty -- scifi's repair rung ships an
uncast speaker with no prose so the episode still lands -- and every reader
here treats empty as absent.

Stdlib only; cold-import clean.
"""
from __future__ import annotations

try:
    from .cast_picture_text import picture_description
except ImportError:  # pragma: no cover -- flat test imports
    from cast_picture_text import picture_description  # type: ignore

__all__ = [
    "CAST_TEXT_FIELDS", "field", "is_bodiless", "picture_appearance", "still_text",
    "video_text", "card_text",
    "identity_fields",
]

#: The three character fields every cast row carries, in reading order.
CAST_TEXT_FIELDS = ("appearance", "presence", "voice_cue")


def field(row, name: str) -> str:
    """One field, stripped; ``""`` for a missing row, key or non-string.
    A dict row is read by key, any other row (a pydantic cast member) by
    attribute."""
    if row is None:
        return ""
    value = row.get(name) if isinstance(row, dict) else getattr(row, name, None)
    return value.strip() if isinstance(value, str) else ""


#: An appearance that says there is nothing to draw. Live legs wrote "none"
#: (a disembodied voice) and "n/a" (two scifi leads, 2026-10-18); a picture
#: prompt then read "adult woman, none." These count as empty for pictures --
#: the writing passes still see them, where "none" is information.
_NO_APPEARANCE = frozenset({
    "none", "n/a", "na", "n.a.", "nil", "null", "unknown", "unspecified",
    "not specified", "not stated", "invisible", "unseen", "not visible",
    "disembodied", "no body", "no physical form", "-", "--", "?", "tbd",
})


#: Prose that says the same thing at sentence length: the six-bank leg's
#: my_story Voice wrote "no physical appearance, represented only by an
#: auditory effect" (2026-10-18). Any of these anywhere marks it bodiless.
#: Whole phrases that only a bodiless character's appearance says. Never a
#: bare "no physical" / "never seen" / "auditory": real looks carry "no
#: physical scars", "never seen without his hat", "an auditory implant"
#: (Sonnet QA on fe35b8d5). Missing a bodiless phrasing costs little -- the
#: casting prompt already asks for a glow, a shadow or an object instead.
_BODILESS_MARKS = (
    "no physical appearance", "no physical form", "no physical body",
    "no visible form", "no visible body", "has no body", "without a body",
    "bodiless", "disembodied", "only a voice", "only as a voice",
    "just a voice", "a voice only", "voice alone", "faceless voice",
    "auditory effect", "auditory presence", "represented only by",
    "heard but not seen", "felt but not seen", "felt, not seen",
)


def is_bodiless(row) -> bool:
    """True when the appearance says there is nothing to draw."""
    text = field(row, "appearance").strip(" .;,!").lower()
    return text in _NO_APPEARANCE or any(m in text for m in _BODILESS_MARKS)


def picture_appearance(row) -> str:
    """The appearance as a picture may use it: ``""`` when it says there is
    nothing to draw."""
    return "" if is_bodiless(row) else field(row, "appearance")


def still_text(row) -> str:
    """What a STILL or PORTRAIT may draw: the appearance, cleaned."""
    return picture_description(picture_appearance(row), still=True)


def _sentence(text: str) -> str:
    """``text`` ending in sentence punctuation, so joined fields never run
    together ("...harmonica Presence: ..." read as one clause)."""
    text = text.strip().rstrip(" ,;:")
    return text if not text or text[-1] in ".!?" else text + "."


def video_text(row, *, cap: int = 0) -> str:
    """What a VIDEO prompt may show: appearance then presence, cleaned of any
    voice wording (joint audio-video models VOCALIZE what they read).

    ``cap`` budgets the two fields separately, as ``card_text`` does, so a
    capped consumer (ShotLock's 160-char beat batch) keeps the presence a
    plain slice of a long appearance would cut off (Codex wiring review)."""
    if is_bodiless(row):
        # Nothing to show: a bodiless character's "presence" is how it SOUNDS
        # ("a faint, mournful sigh ... crashing waves", the same leg), and a
        # joint audio-video model would voice it.
        return ""
    parts = [picture_description(picture_appearance(row), still=False),
             picture_description(field(row, "presence"), still=False)]
    parts = [p for p in parts if p]
    if cap and cap > 0:
        out, budget = [], int(cap)
        for i, text in enumerate(parts):
            room = max(0, budget // (len(parts) - i) - 1)
            if len(text) > room:
                head = text[:room]
                text = (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(" ,;:.")
            if text:
                out.append(text)
                budget -= len(text) + 2
        parts = out
    if len(parts) == 2:
        parts[0] = _sentence(parts[0])
    return " ".join(parts)


def card_text(row, *, cap: int = 0) -> str:
    """The character for an LLM context pass:
    ``"<appearance>. Presence: <presence>."``.

    NO VOICE CUE (agy review, 2026-10-18). A text model cannot make a line
    sound husky; handed "Voice: husky whisper" beside the slot it writes, a
    small model obeys with "(whispering)". ``voice_cue`` stays on the row for
    traceability and is read by nothing that writes a line. The speech
    register reaches the line writer as ``speech_signature`` -- word choice.

    ``cap`` (characters) budgets each field SEPARATELY -- an equal share of
    what is left, so a short field passes its spare on -- and a capped
    consumer (the story brief's 200, continuity's 220) still sees presence and
    voice instead of losing them to a long appearance. A field that cannot fit
    a word is dropped whole and spends nothing."""
    labelled = [("", field(row, "appearance")),
                ("Presence: ", field(row, "presence"))]
    labelled = [(label, text) for label, text in labelled if text]
    if not (cap and cap > 0):
        return " ".join(label + _sentence(text) for label, text in labelled)
    out = []
    budget = int(cap)
    for i, (label, text) in enumerate(labelled):
        share = budget // (len(labelled) - i)
        room = max(0, share - len(label) - 1)          # 1 for the full stop
        if len(text) > room:
            head = text[:room]
            text = (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(" ,;:.")
        if not text:
            continue
        piece = label + _sentence(text)
        out.append(piece)
        budget -= len(piece) + 1
    return " ".join(out)


def identity_fields(row) -> dict:
    """Every model-written prose field, SEPARATELY, for the name-authority
    guard and gender inference -- the guard reports which field failed and
    reads the words before a name, so the fields must not be joined."""
    out = {name: field(row, name) for name in CAST_TEXT_FIELDS}
    out["speech_signature"] = field(row, "speech_signature")
    return out
