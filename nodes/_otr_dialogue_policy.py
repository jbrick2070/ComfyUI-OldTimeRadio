"""Pure helper module for dynamic dialogue prompt policies."""

from collections.abc import Sequence


# Scoped to LEMMY by its own grammar, because the prompt is read as written.
# The first sentence used to be the subjectless "Convey the Cockney accent
# through phrasing, idiom, cadence, and rhythm", appended whenever Lemmy was
# anywhere in the episode roster -- so the writer took it as a scene-wide
# accent order and re-registered the whole ensemble. The spelling sentence was
# always the point; the accent sentence was meant as its context.
_COCKNEY_ORTHOGRAPHY_RULE = (
    "\n\nFor LEMMY's spoken lines only, convey his Cockney accent through "
    "phrasing, idiom, cadence, and rhythm. Every other character must retain "
    "that character's own speech register; do not give any other character "
    "LEMMY's Cockney phrasing, idiom, cadence, or rhythm. Use standard English "
    "spelling in every spoken line; do not encode pronunciation with phonetic "
    "misspellings."
)


def _active_speakers_have_lemmy(active_speakers: Sequence[str]) -> bool:
    """Return True if LEMMY is one of the speakers this call will voice."""
    if isinstance(active_speakers, (str, bytes)) or not isinstance(
        active_speakers, Sequence
    ):
        raise TypeError("active_speakers must be a sequence of speaker-name strings")
    speakers = tuple(active_speakers)
    if any(not isinstance(speaker, str) for speaker in speakers):
        raise TypeError("active_speakers must contain only speaker-name strings")
    return any(speaker.strip().upper() == "LEMMY" for speaker in speakers)


def append_dialogue_policy(
    system_prompt: str, *, active_speakers: Sequence[str]
) -> str:
    """Append speaker-scoped policy rules to a router-resolved system prompt.

    `active_speakers` is the speakers of THIS model call and nothing wider --
    `LineRequest.speaker` on the per-line path, the `VoicedSlot.speaker` of
    each slot in the group on the exchange path. A full-cast roster, a cast
    row, a `char_id`, or `allowed_people` is a different category: those say
    who exists in the episode, not who is speaking now, and passing one is
    what put a Cockney accent order over every character in the scene. Wrong
    categories raise rather than silently widening the policy.
    """
    system = system_prompt or ""
    if _active_speakers_have_lemmy(active_speakers):
        system = system + _COCKNEY_ORTHOGRAPHY_RULE
    return system
