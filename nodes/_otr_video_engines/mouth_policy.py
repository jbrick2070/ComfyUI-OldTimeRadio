"""WIRE-W7 -- THE MOUTH-STILL POLICY. Who owns the lips, and how many faces.

THE GAP THIS CLOSES (r3 MUST-FIX 11, 2026-07-28): *"No W1-W6 step enforces the
operator's three rulings. Nothing makes every audio-in beat receive a
mouth-bearing still, nothing makes the radio-face the default, nothing enforces
one overheard human face per episode. The plan has the rulings and no owner for
them."* An unowned ruling is a ruling that silently lapses.

THE RULINGS, verbatim where they were given:

1. *"Every audio-in beat gets a still with a mouth."* (operator, 2026-07-28)
2. *"The lips may be a person OR A RADIO."* (same)
3. *"THE SET SPEAKS BY DEFAULT; A FACE MUST BE OVERHEARD. The cabinet is the
   speaker for the announcer (who IS the station), for the music bookends, and
   for characters while they are performing. A human mouth appears only when a
   character stops broadcasting and says something private -- the confession,
   the last line. One face per episode at most, and only for a line the engine
   can hold in a single take."* (house rule, GO_FORWARD)

DECIDED FROM THE FROZEN ROUTE, NEVER INFERRED FROM PROSE. The answer is a
function of the beat's ROLE, its engine's FAMILY and the engine id -- the same
three facts the render dispatcher freezes -- and never of the beat's text. That
is deliberate and it is the operator's own word ("never inferred from prose"):
a policy that read the line would decide differently for the same route
depending on how the writer phrased it, and two beats routed identically would
get different stills.

THE SCHEMA IS NOT EXTENDED, ON PURPOSE. ``_otr_shared.still_plan_helpers``
carries a CLOSED set of ``kind`` tokens and says in as many words that "adding a
fifth token to any of these is an operator decision, never a coder's judgment
call". There is no ``bears_a_mouth`` field and this module does not add one; it
derives the answer from facts that already exist.

PURE AND DEPENDENCY-FREE. It takes plain strings and booleans, so it can be
asked by ShotLock at plan time (where the render_driver helpers live) without
either side importing the other. stdlib-only; UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations


class MouthPolicyError(ValueError):
    """A beat's mouth has no owner, or an episode shows too many faces.

    Deliberately its own type: it is neither a missing input (``FamilyInputGap``)
    nor an unrenderable plan (``CoveragePlanError``). It is a LOOK contract the
    operator ruled on, and the honest failure is to say which ruling and which
    beat rather than to render something the ruling forbids.
    """


#: The families whose engines are DRIVEN BY AUDIO and therefore owe a mouth.
#: ``audio_driven_face`` is HuMo and the cloud avatars; ``audio_conditioned_video``
#: is the LTX-2.3 audio-in lane. Chosen by FAMILY rather than by engine id so a
#: new adapter in either family inherits the ruling instead of slipping past a
#: list nobody remembered to edit.
#:
#: The audio-REACTIVE visualizers (``viz_green`` and friends) are deliberately
#: NOT here. They read the master mix to paint a scope; they are abstract by
#: declaration and have no face to give a mouth to. ``render_driver.
#: _uses_ambient_master_audio`` names them by engine id for the SLICE question,
#: which is a different question from this one.
AUDIO_IN_FAMILIES = ("audio_driven_face", "audio_conditioned_video")

#: The beat's still carries a HUMAN mouth -- a character portrait.
MOUTH_HUMAN = "human"

#: The beat's still carries the RADIO's mouth -- the console face. THE DEFAULT.
MOUTH_RADIO = "radio"

#: The beat is not audio-in and owes no mouth at all.
MOUTH_NONE = "none"

#: The video roles the house rule assigns to the CABINET: the announcer (who
#: IS the station) and the music bookends. Mirrors the role vocabulary
#: ``render_driver._VIDEO_ROLE_NEVER_HUMO_PROXY`` dispatches on -- the same two
#: roles that are structurally redirected away from HuMo, for the same reason
#: ("the radio IS the host", BUG-LOCAL-129).
SET_SPEAKS_ROLES = ("announcer_visual", "music_visual")


def beat_owes_a_mouth(family) -> bool:
    """True iff this beat's engine is driven by audio and therefore owes a
    mouth-bearing still. Pure."""
    return str(family or "") in AUDIO_IN_FAMILIES


def mouth_owner_for_beat(*, engine_id, family, role, is_character_face):
    """Which mouth this beat's still must carry: ``MOUTH_HUMAN``,
    ``MOUTH_RADIO`` or ``MOUTH_NONE``.

    ``is_character_face`` is ``render_driver._is_character_face_beat``'s answer,
    passed in rather than recomputed -- there is one authority for "is this a
    talking head", and a second derivation is a second chance to disagree with
    the dispatcher about which audio a beat gets.

    RAISES on the case r3 said nobody owned: an audio-in beat that is neither a
    character face nor a cabinet role. Such a beat WILL be handed a still and
    WILL animate it, and nothing today says whether that still has lips. The
    honest answer is to name it at plan time rather than discover it in the
    finished episode.
    """
    family = str(family or "")
    role = str(role or "")
    if not beat_owes_a_mouth(family):
        return MOUTH_NONE
    if is_character_face:
        return MOUTH_HUMAN
    if role in SET_SPEAKS_ROLES:
        return MOUTH_RADIO
    raise MouthPolicyError(
        "an audio-in beat on engine %r (family %s, role %r) is neither a "
        "character face nor a cabinet role, so nothing decides whether its "
        "still has a mouth -- and an audio-in engine WILL animate whatever it "
        "is given. NO FALLBACK: route this beat to a character face, give it "
        "one of the cabinet roles %s, or move it off the audio-in families %s."
        % (engine_id, family, role, list(SET_SPEAKS_ROLES),
           list(AUDIO_IN_FAMILIES)))


#: The most human faces one episode may show. The operator's number, and it is
#: ONE rather than zero because the rule is "a face must be OVERHEARD", not
#: "no faces": the confession, the last line.
MAX_HUMAN_FACES_PER_EPISODE = 1


def audit_episode_faces(beats):
    """Enforce the cardinality half of the house rule over a whole episode.

    ``beats`` is an iterable of dicts describing the FROZEN route, one per
    beat::

        {"beat_id", "engine_id", "family", "role", "char_id",
         "is_character_face", "is_multi_clip"}

    Returns the sorted tuple of character ids that wear a human mouth (empty
    when the set speaks throughout, which is the default look).

    TWO REFUSALS, both the operator's sentence:

    * **"One face per episode at most."** Counted by DISTINCT ``char_id``, not
      by beat: a character who says three private lines is still one face. A
      human-mouth beat carrying no char_id is itself a refusal -- an unnamed
      face cannot be counted, so it cannot be capped.
    * **"...and only for a line the engine can hold in a single take."** A
      human-face beat may not be multi-clip. HuMo declares
      ``CONTINUITY_SOFT_REFERENCE``, so a multi-clip face beat is an honest
      JUMP CUT: the same character regenerated mid-line from a different seed,
      cutting to themselves. The cabinet can take that cut; a face cannot.
    """
    faces = {}
    for beat in beats or ():
        owner = mouth_owner_for_beat(
            engine_id=beat.get("engine_id"),
            family=beat.get("family"),
            role=beat.get("role"),
            is_character_face=bool(beat.get("is_character_face")))
        if owner != MOUTH_HUMAN:
            continue
        char_id = str(beat.get("char_id") or "")
        beat_id = str(beat.get("beat_id") or "?")
        if not char_id:
            raise MouthPolicyError(
                "beat %s shows a HUMAN mouth but names no char_id, so the "
                "one-face-per-episode cap cannot be counted against it. NO "
                "FALLBACK -- an unnamed face is an uncapped face." % beat_id)
        if beat.get("is_multi_clip"):
            raise MouthPolicyError(
                "beat %s shows the human face of %r across MORE THAN ONE clip. "
                "The rule is one face per episode at most, and only for a line "
                "the engine can hold in a SINGLE TAKE -- a multi-clip face beat "
                "is a jump cut from a character to a regenerated copy of "
                "themselves mid-line. NO FALLBACK: shorten the line, or let the "
                "cabinet speak it." % (beat_id, char_id))
        faces.setdefault(char_id, []).append(beat_id)
    if len(faces) > MAX_HUMAN_FACES_PER_EPISODE:
        raise MouthPolicyError(
            "this episode shows %d human faces (%s) and the house rule allows "
            "at most %d: THE SET SPEAKS BY DEFAULT; A FACE MUST BE OVERHEARD. "
            "NO FALLBACK -- give the other character(s) the cabinet."
            % (len(faces),
               ", ".join("%s on %s" % (cid, ", ".join(bids))
                         for cid, bids in sorted(faces.items())),
               MAX_HUMAN_FACES_PER_EPISODE))
    return tuple(sorted(faces))


__all__ = [
    "MouthPolicyError",
    "AUDIO_IN_FAMILIES",
    "MOUTH_HUMAN",
    "MOUTH_RADIO",
    "MOUTH_NONE",
    "SET_SPEAKS_ROLES",
    "MAX_HUMAN_FACES_PER_EPISODE",
    "beat_owes_a_mouth",
    "mouth_owner_for_beat",
    "audit_episode_faces",
]
