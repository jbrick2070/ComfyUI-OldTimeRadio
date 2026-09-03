"""OTR_MetaBriefImagePromptGen -- LLM image prompts from the Meta brief (C1).

The image-side mirror of ``OTR_ShotLock``'s per-beat creative derivation and of
``nodes/_otr_music_prompt.py``: every portrait/scene prompt is composed from the
propagating Meta brief (setting / period / mood) + the character's appearance,
optionally refined by ONE LLM call on the writer's slot (V-11, no new model_id
widget) at ``temperature=0`` with the ``prompt_hash`` taken AFTER the call.

LLM refinement is bounded and optional. Empty or failed output uses a
deterministic brief-composed prompt. Python does not classify or reject
authored visual nouns, objects, style terms, or vocabulary. The canonical
compatibility widget remains wired but has no gating effect.

PURE core (``compose_image_prompt_fallback`` / ``derive_image_prompts``): no I/O,
no GPU, no engine imports -- the LLM is injected (tests) or resolved lazily from
the writer slot. Cold-import clean. UTF-8 no BOM, ASCII-only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

# A malformed routing environment is terminal at every reader (2026-07-25).
# route_freeze is stdlib-only at module scope, so this stays cold-import clean.
try:
    from ._otr_shared.route_freeze import RouteFreezeError as _RouteFreezeError
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared.route_freeze import RouteFreezeError as _RouteFreezeError

log = logging.getLogger("OTR")

def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_setting(meta: dict) -> str:
    """Brief setting string (mirrors the music-prompt brief read; fail-soft)."""
    terms = (meta or {}).get("story_brief_terms")
    if not isinstance(terms, dict):
        terms = {}
    setting_raw = terms.get("setting") or []
    if not isinstance(setting_raw, list):
        setting_raw = []
    setting = [str(t).strip() for t in setting_raw if str(t).strip()]
    return ", ".join(setting[:2])


def _appearance_for_char(cast: list, char_id: str) -> str:
    """Appearance text looked up by char_id ONLY (BUG-098), never display name.

    Key chain includes ``character_description`` (operator look-QA
    2026-06-10): the writer stamps the RICH per-character physical
    description on the cast row as ``character_description`` (the
    ``portrait_prompt`` mirror lives under meta.visual_plan keyed by NAME,
    not on the row), so without this key every character fell to the same
    generic setting+anchor fallback -> ONE shared portrait for the whole
    cast, styled as an actor in a radio booth."""
    cid = str(char_id or "")
    for c in cast or []:
        if isinstance(c, dict) and str(c.get("char_id") or "") == cid:
            return str(c.get("portrait_prompt") or c.get("appearance")
                       or c.get("character_description") or "").strip()
    return ""


_FEMALE_PROMPT_TERMS = frozenset({
    "female", "woman", "women", "lady", "girl", "matriarch", "mother",
})
_MALE_PROMPT_TERMS = frozenset({
    "male", "man", "men", "gentleman", "boy", "patriarch", "father",
})


#: Terms that already establish a NON-gendered human subject. Without these the
#: neutral anchor would stack ("person, a person in a long coat...").
_NEUTRAL_PROMPT_TERMS = frozenset({
    "person", "figure", "individual", "adult", "someone", "they",
})


def _ensure_gender_anchor(prompt: str, char: dict) -> str:
    """Add a cast-gender anchor when the prompt forgot one.

    NORMALIZED (item 8, 2026-08-06). This used to compare the raw stored value
    against ``("female","male")``. It lower-cased first, so title-case was never
    the problem -- but `woman`, `man`, `m` and `f` are all live in the published
    corpus and every one of them fell straight through UNANCHORED. Those rows
    plainly state a gender; the equality test just could not hear it.

    ``other`` (253 rows) previously got no anchor at all. It now gets a NEUTRAL
    one, so an `other` row is anchored as a person rather than left to whatever
    the image model infers from the prose.

    Still deliberately ADDITIVE and non-blocking: this function may not reject,
    rewrite or block a prompt (the contract at the portrait payload builder
    below). Contradictions are REPORTED by the consistency audit, not gated here.
    """
    from ._otr_roster_gender import normalize_gender

    text = str(prompt or "").strip()
    if not text:
        return text
    gender = normalize_gender((char or {}).get("gender"))
    words = set(re.findall(r"[a-z]+", text.lower()))
    if gender == "female":
        if words & _FEMALE_PROMPT_TERMS:
            return text
        return f"adult woman, {text}"
    if gender == "male":
        if words & _MALE_PROMPT_TERMS:
            return text
        return f"adult man, {text}"
    # 'other' -- anchor the subject as a person without asserting a binary the
    # ledger did not record. The voice side still has to pick a binary because
    # the bank carries no 'other' references; the PICTURE does not.
    if words & (_NEUTRAL_PROMPT_TERMS | _FEMALE_PROMPT_TERMS | _MALE_PROMPT_TERMS):
        return text
    return f"person, {text}"


#: Shared portrait style anchor. Reworded 2026-06-10 (operator look-QA): the
#: old "studio portrait, neutral lighting" framing read as an ACTOR in a
#: recording booth; portraits must show the CHARACTER in character, in the
#: story's world -- never a voice actor at a microphone.
# Round 5 operator notes (2026-06-10): the wider framing is a KEEPER ("this
# week's portraits show more body -- better"), so it is now intentional
# (three-quarter, not head-and-shoulders). The old "no microphone, not a
# recording studio" NEGATIONS are gone -- negative phrasing plants those
# tokens in the image embedding. Guidance stays in the LLM request; Python does
# not filter the authored prompt vocabulary.
# GEOMETRY-vs-LOOK split (visual-style TOTAL COVERAGE chunk A1, 2026-07-05):
# the *_GEOMETRY constants below are ENGINE-SAFETY framing contracts
# (framing / headroom / face-visibility / mouth-safety) and NEVER move into
# style packs; the LOOK segment (costume/environment/lighting vocabulary) is
# pack-owned (VisualStyle.portrait_look / portrait_look_talking). The
# *_LOOK_DEFAULT constants survive ONLY as the sci_fi_radio extraction
# fixtures + the legacy no-style lane of _style_anchor_for_aspect --
# production callers always pass style= (AST-pinned).
PORTRAIT_GEOMETRY = ("in-character cinematic three-quarter portrait, full head and face "
                     "clearly visible with natural headroom above the head (never crop "
                     "the top of the head)")
WIDE_PORTRAIT_GEOMETRY = ("in-character cinematic medium shot, head and shoulders, face "
                          "clearly visible, subject centred with natural headroom above "
                          "the head (never crop the top of the head)")
TALKING_PORTRAIT_GEOMETRY = (
    "in-character face-forward frontal close-up bust, looking directly at the "
    "camera, the whole face and mouth clearly visible and unobstructed, the "
    "face brightly lit and filling much of the frame, subject centred with "
    "natural headroom above the head (never crop the top of the head)")

#: Extraction fixtures (pack byte-identity pins; sci_fi_radio.json values).
PORTRAIT_LOOK_DEFAULT = ("period-accurate costume and environment, "
                         "dramatic film lighting")
TALKING_PORTRAIT_LOOK_DEFAULT = "period-accurate costume, warm dramatic lighting"
#: The LLM-facing look language _build_char_prompt_request used to hard-code
#: (chunk A1: production reads the pack's portrait_instruction_look; this
#: survives ONLY as the sci_fi_radio extraction fixture).
PORTRAIT_INSTRUCTION_LOOK_DEFAULT = "photographic and period-consistent"

#: The composed legacy anchors -- BYTE-IDENTICAL by construction (fixtures for
#: pre-A1 pins; production reads go through _style_anchor_for_aspect(style=)).
STYLE_ANCHOR = "%s, %s" % (PORTRAIT_GEOMETRY, PORTRAIT_LOOK_DEFAULT)

#: WIDE (16:9) framing anchor. A three-quarter body shot cannot fit a short
#: landscape still without cropping the head, so wide character stills use a
#: head-and-shoulders MEDIUM shot, subject centred with headroom (operator framing
#: catch 2026-06-17: the wide character beats were decapitating the subject).
#: Portrait stills keep the three-quarter look (operator KEEPER 2026-06-10).
STYLE_ANCHOR_WIDE = "%s, %s" % (WIDE_PORTRAIT_GEOMETRY, PORTRAIT_LOOK_DEFAULT)


#: TALKING-lane portrait anchor (S4b, 2026-07-02). The ia2v lip-sync recipe
#: conditions character beats on the PORTRAIT (S4), and proof8 showed the
#: brief-styled cinematic portrait (dark palette, profile/wide composition,
#: tiny face) cannot drive lips -- the isolation A/B needed a bright,
#: face-forward close-up (scene-init 0.57 vs face-forward 2.86). Mirrors the
#: ltx_radio_mouth split that fixed the radio bookends: frontal, mouth
#: visible, warm light; the era/grade/palette tails are SKIPPED for these
#: portraits (see derive_image_prompts). Talking portraits' ONLY style
#: surface is the pack's portrait_look_talking (S4b lip-sync law) -- packs
#: author it conservatively.
STYLE_ANCHOR_TALKING = "%s, %s" % (TALKING_PORTRAIT_GEOMETRY,
                                   TALKING_PORTRAIT_LOOK_DEFAULT)


def _style_anchor_for_aspect(aspect, talking=False, style=None) -> str:
    """Framing anchor for a still's aspect: head-and-shoulders for WIDE (16:9) so
    the head is not cropped by the short frame, three-quarter for PORTRAIT.
    ``talking`` overrides both with the face-forward lip-sync anchor (S4b).

    Chunk A1: geometry stays Python; the LOOK segment comes from the resolved
    ``style`` pack, appended at the SAME position the legacy anchors carried
    it (sci_fi_radio is byte-identical by construction). ``style=None`` is
    the LEGACY fixture lane (tests) -- production callers pass style=
    (AST-pinned); helpers never re-resolve."""
    if talking:
        look = (style.portrait_look_talking if style is not None
                else TALKING_PORTRAIT_LOOK_DEFAULT)
        return "%s, %s" % (TALKING_PORTRAIT_GEOMETRY, look)
    look = style.portrait_look if style is not None else PORTRAIT_LOOK_DEFAULT
    geometry = (WIDE_PORTRAIT_GEOMETRY if str(aspect).lower() == "wide"
                else PORTRAIT_GEOMETRY)
    return "%s, %s" % (geometry, look)

#: The station ANNOUNCER is a synthetic, non-cast portrait subject (CastLock
#: owns ``ledger['cast']``; the announcer is the station voice, never a cast
#: row). Announcer beats are talking beats, so HuMo needs an ``init_image``
#: for them exactly like character beats -- without one the intro/outro
#: starve to the still floor (the b001/b005 keystone gap). The pseudo-id
#: matches the ``char_id`` the writer stamps on announcer lines.
ANNOUNCER_CHAR_ID = "announcer"


# BRIEF-DRIVEN RADIO FORM (2026-07-01): the deterministic brief -> radio-form
# resolver lives in the PURE brief helper (_otr_story_brief_helpers) so both this
# image node AND get_open_subject share it with no circular import. Re-exported
# here (radio_form_from_meta / _RADIO_FORM_DEFAULT) for the image-prompt callers.
try:
    from ._otr_story_brief_helpers import (  # type: ignore # noqa: F401
        radio_form_from_meta, _RADIO_FORM_DEFAULT)
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_story_brief_helpers import (  # type: ignore # noqa: F401
        radio_form_from_meta, _RADIO_FORM_DEFAULT)


#: Radio-HOST FACE styling. The FACE-BEARING looks are for AUDIO-DRIVEN engines
#: only (something animates a mouth); a static/forced bookend uses the FACELESS
#: ``radio_object`` style below (radio-face logic 2026-07-04, operator: "a
#: bookend needs a face only when something animates it").
#:   style="console_face": an ANTHROPOMORPHIC RADIO CONSOLE whose glowing tuning
#:     dial forms an expressive face -- dial-eyes + a radiating needle-fan mouth.
#:     "The radio IS the host"; NO human present. Used for the HuMo radio-host
#:     FACE object AND the HuMo-driven synthetic-announcer portrait.
#:   style="ltx_radio_mouth": the LTX-ONLY mouth-forward dial-face (below).
#:   style="radio_object": a FACELESS stylized radio on its plate -- no dial-face,
#:     no person -- for a static/i2v or force-mapped announcer (nothing drives a
#:     mouth, so no face). Facelessness is carried by the POSITIVE prompt.
#: Overtness is BRIEF-DRIVEN (operator: "brief-driven mix"): subtle/period for
#: noir/deco, more overt/playful for retro-futurist / sci-fi briefs.
_RADIO_CONSOLE_FACE = ("its glowing tuning dial forming an expressive stylized "
                       "face -- two round dial-eyes and a radiating needle-fan "
                       "mouth, an anthropomorphic radio that hosts the broadcast")
#: FACELESS radio-object subject + anchors (radio-face logic 2026-07-04). The
#: person anchors (_style_anchor_for_aspect -> STYLE_ANCHOR*) carry "full head
#: and face", "period-accurate costume", "in-character" -- anatomy/person tokens
#: that would defeat facelessness. radio_object leans on OBJECT/material language
#: only (zero anatomy tokens; facelessness rides the POSITIVE) and
#: an OBJECT anchor, and it never appends _radio_face_overtness (which returns
#: face-bearing text). Exact anchor strings pinned (r3) so devs don't drift.
_RADIO_OBJECT_SUBJECT = ("presented as a stylized tabletop radio set on its "
                         "plate, its bakelite cabinet, woven speaker grille and "
                         "a glowing tuning dial")
# Chunk A2 geometry-vs-look split: the OBJECT anchors decompose into
# isolation/composition GEOMETRY (Python, never in packs) + the pack-owned
# lighting LOOK (VisualStyle.radio_object_look). The composed _RADIO_OBJECT_
# ANCHOR* constants survive as extraction fixtures + the legacy no-style lane.
RADIO_OBJECT_GEOMETRY = ("isolated tabletop still, the whole radio centered "
                         "and fully visible")
RADIO_OBJECT_GEOMETRY_WIDE = ("isolated tabletop still, the whole radio "
                              "centered and fully visible, wide shot")
RADIO_OBJECT_LOOK_DEFAULT = "dramatic film lighting"
_RADIO_OBJECT_ANCHOR = "%s, %s" % (RADIO_OBJECT_GEOMETRY,
                                   RADIO_OBJECT_LOOK_DEFAULT)
_RADIO_OBJECT_ANCHOR_WIDE = "%s, %s" % (RADIO_OBJECT_GEOMETRY_WIDE,
                                        RADIO_OBJECT_LOOK_DEFAULT)
#: LTX-ONLY mouth-forward radio face (talking-radio kibitz r1, 2026-07-01).
#: style="ltx_radio_mouth" is used ONLY by the ltx talking radio-face still mint
#: (the init stills the EXISTING ltx_audio_in bookend engine receives -- no new
#: video model / path). NEVER used by the HuMo hosts: the console_face look
#: above stays byte-unchanged (a mouth-tuned change
#: could hurt HuMo face-readability -- the Codex MUST-FIX #4 split). LTX-2.3
#: has no face/landmark detector; it drives whatever READS as a mouth, so the
#: subject puts a PROMINENT rubbery grille-mouth right after the form noun
#: (FLUX weights earlier tokens). Sub-plan C probes whether this actually
#: lip-syncs; until then ltx_audio_in stays documented as AMBIENT motion.
#: MATERIAL-ANCHORED (live catch 2026-07-02, probe B): the image dispatcher
#: leans on the POSITIVE prompt for facelessness, so
#: the object-row "no human" negative is inert on the still engines -- the
#: first live mint rendered a literal HUMAN face inside the radio. Anatomy
#: words must therefore be bound to APPLIANCE materials in the positive
#: itself (appliance face; lips molded from grille cloth; glass dial-eyes;
#: cabinet fills the frame). No negation words -- image models ignore "no X"
#: in a positive and the person guard greps for the bare tokens.
_RADIO_CONSOLE_MOUTH = ("%s as a living cartoon appliance face: its wide "
                        "woven speaker grille bends into a huge expressive "
                        "rubbery mouth, big cartoon lips molded from grille "
                        "cloth and wood, open mid-speech, and its two round "
                        "glass tuning dials are its eyes -- a face-forward "
                        "anthropomorphic radio made entirely of bakelite, "
                        "wood, grille fabric and chrome, the radio cabinet "
                        "itself filling the frame")
#: Brief keywords that push the face OVERT (playful cartoon) vs subtle.
_RADIO_FACE_OVERT_KEYS = ("space", "orbital", "docking", "spacecraft", "starship",
                          "sci-fi", "science fiction", "futuristic",
                          "retro-futuristic", "atomic age", "galactic")
#: Per-style negative populated on the object row (schemas.py negative_prompt;
#: NO schema change). All three live styles (console_face / ltx_radio_mouth /
#: radio_object) are pure radios with NO person in frame, so they SHARE
#: NOTE (ONE STYLE AUTHORITY, 2026-08-17): these comments used to add that
#: the still dispatcher had NO negative channel and that these constants
#: were therefore cosmetic. That is no longer true -- the dispatcher now
#: composes the pack negative with this per-object value and the
#: negative-capable still engines honour it, so editing these strings is a
#: BEHAVIOUR change, not bookkeeping. The positive-side guarantee still
#: stands and is what the tests pin.
#: RADIO_CONSOLE_NEG (humans OUT). Facelessness rides the POSITIVE,
#: so this is COSMETIC for the still engines -- facelessness rides the POSITIVE.
RADIO_CONSOLE_NEG = "human, person, man, woman, human face, hands, arms, crowd"
#: Back-compat alias (default = the console look).
RADIO_HOST_FACE_NEG = RADIO_CONSOLE_NEG

#: The live radio-host styles. build_radio_host_prompt + radio_host_negative
#: dispatch EXPLICITLY over this set and raise on anything else, so a typo or a
#: stale ``radio_head_person`` caller fails LOUD instead of silently defaulting
#: to a dial-face (radio-face logic 2026-07-04).
_RADIO_HOST_STYLES = ("console_face", "ltx_radio_mouth", "radio_object")


def radio_host_negative(style: str = "console_face") -> str:
    """The negative_prompt for a radio-host still by style. Fail-loud on an
    unknown style (the retired ``radio_head_person`` now RAISES here)."""
    if style in _RADIO_HOST_STYLES:
        return RADIO_CONSOLE_NEG
    raise ValueError(
        "radio_host_negative: unknown radio-host style %r; known styles: %r. "
        "(radio_head_person was retired 2026-07-04 -- radio-face logic.)"
        % (style, _RADIO_HOST_STYLES))

#: The ONE animatable radio-HOST FACE object minted per episode under
#: OTR_ENABLE_HUMO_HOSTS (its own object_id, NOT a cast char_id). render_driver
#: resolves it as the HuMo init_image for the LINELESS announcer/music bookends
#: (which have no cast portrait). MUST match the render_driver + dispatcher
#: literals (kept in sync by string, cross-referenced in those files).
RADIO_HOST_PORTRAIT_ID = "radio_host_portrait"

#: Toggle: default OFF = byte-identical (no radio-host FACE object minted; the
#: bookends keep today's ltx_audio_in animated-console behavior).
def _humo_hosts_enabled() -> bool:
    """True iff OTR_ENABLE_HUMO_HOSTS is opted ON (default OFF)."""
    import os
    return os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1"


#: LTX audio-in bookends use a WIDE radio-face still for their I2V init. Music
#: remains a radio in every visual mode; when it is explicitly audio-driven it
#: receives the same radio-with-lips treatment as the announcer (operator
#: 2026-07-12). Other music engines retain their faceless radio scene still.
#: Object ids MUST match render_driver._ltx_radio_face_object_id.
_LTX_RADIO_FACE_ROLES = ("announcer_visual", "music_visual")


def _ltx_radio_face_object_id(role: str) -> str:
    """object_id of the WIDE radio-face still for ``role`` (ltx talking radio-face).
    Matches render_driver._ltx_radio_face_object_id."""
    return "still_%s_radio_face_169" % str(role or "")


def _radio_face_overtness(meta) -> str:
    """Brief-driven overtness of the radio face (operator: 'brief-driven mix').
    Retro-futurist / sci-fi briefs get a bold playful cartoon face; everything
    else stays a subtle period-authentic dial-face. Pure."""
    try:
        try:
            from ._otr_story_brief_helpers import _radio_form_haystack  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_story_brief_helpers import _radio_form_haystack  # type: ignore
        hay = _radio_form_haystack(meta)
    except Exception:  # noqa: BLE001
        hay = ""
    return ("bold playful cartoon expression, clearly a face"
            if any(k in hay for k in _RADIO_FACE_OVERT_KEYS)
            else "subtle period-authentic dial-face")


def build_radio_host_prompt(meta, aspect: str = "portrait",
                            radio_host_style: str = "console_face",
                            vstyle=None) -> str:
    """FULL prompt for a radio-host still, dispatched by ``radio_host_style``
    (renamed from ``style`` in chunk A1 -- r3: the visual-style pack now
    travels as ``vstyle=``, so the DISPATCH arg needed a non-colliding name).

    ``radio_host_style="console_face"``: an ANTHROPOMORPHIC RADIO CONSOLE (the
    brief-driven form) whose glowing dial forms an expressive face -- "the radio
    IS the host", no human. The animatable dial-face; used for the HuMo
    radio-host FACE object AND the HuMo-driven synthetic-announcer portrait.
    ``radio_host_style="ltx_radio_mouth"`` (talking-radio kibitz r1): the
    LTX-ONLY mouth-forward radio face for the ltx talking radio-face still mint --
    leads with the PROMINENT rubbery grille-mouth; never used by HuMo.
    ``radio_host_style="radio_object"`` (radio-face logic 2026-07-04): a
    FACELESS stylized radio on its plate -- no dial-face, no person -- for a
    static / force-mapped announcer bookend (nothing animates a mouth, so no
    face). Facelessness is carried by the POSITIVE prompt (the still dispatcher
    leans on the POSITIVE): object/material language only, an OBJECT anchor
    (NOT the person anchor), and NO overtness clause.
    All styles are brief-driven (:func:`radio_form_from_meta`); the face styles'
    overtness is brief-driven (:func:`_radio_face_overtness`), framed for the
    slot's ``aspect``, era texture via :func:`finish_visual_prompt`. Deterministic;
    never empty. FAIL-LOUD on an unknown style (retired ``radio_head_person``
    raises). The matching negative (:func:`radio_host_negative`) is populated on
    the object row by the caller.

    Chunk A1: the three SUBJECT texts come from the resolved visual-style pack
    (announcer_subject_face / announcer_subject_ltx_mouth /
    announcer_subject_object); ``vstyle`` is the resolved pack threaded from the
    image-prompt entry (None => this IS the entry: fail-loud resolve)."""
    # Stage 3 de-swallow (kibitz r2 M1): the ImportError shim is the ONLY
    # permitted catch here -- a visual-style resolution error must FAIL the
    # episode LOUD, never silently ship an unstyled prompt (no-fallback law).
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            _resolve_style, finish_visual_prompt)
    _style = _resolve_style(meta, vstyle)
    form = radio_form_from_meta(meta)
    if radio_host_style == "radio_object":
        # FACELESS: object/material language only. NO _radio_face_overtness (it
        # returns face text) and NOT _style_anchor_for_aspect (it adds head/face/
        # costume/in-character person tokens). An OBJECT anchor instead, then the
        # SAME "still" era + grade finish as console_face below (a tense story
        # reads as a tense-lit radio, never a person). Chunk A2: isolation
        # GEOMETRY stays Python; the lighting LOOK comes from the pack.
        geometry = (RADIO_OBJECT_GEOMETRY_WIDE
                    if str(aspect).lower() == "wide"
                    else RADIO_OBJECT_GEOMETRY)
        anchor = "%s, %s" % (geometry, _style.radio_object_look)
        prompt = ", ".join(["%s, %s" % (form, _style.announcer_subject_object),
                            anchor])
    elif radio_host_style == "console_face":
        subject = "%s, %s, %s" % (form, _style.announcer_subject_face,
                                  _radio_face_overtness(meta))
        prompt = ", ".join([subject,
                            _style_anchor_for_aspect(aspect, style=_style)])
    elif radio_host_style == "ltx_radio_mouth":
        subject = "%s, %s" % (
            _style.announcer_subject_ltx_mouth.format(form=form),
            _radio_face_overtness(meta))
        prompt = ", ".join([subject,
                            _style_anchor_for_aspect(aspect, style=_style)])
        # CANONICAL WARM LOOK (operator look direction 2026-07-02, the
        # side-by-side catch: the brief palette -- e.g. "cold blue panel
        # glow" -- plus the grade tail's "heavy vignette, muted color grade"
        # rendered the talking-radio bookend DARK, BLUE and murky). The ltx
        # lip-sync still skips the era palette AND the grade tail entirely
        # and pins the canonical demo's lighting. The console/object styles
        # below keep the full brief-driven tail (console goldens are byte-pinned).
        return "%s, warm dramatic lighting" % prompt
    else:
        raise ValueError(
            "build_radio_host_prompt: unknown radio-host style %r; known "
            "styles: %r. (radio_head_person was retired 2026-07-04 -- "
            "radio-face logic.)" % (radio_host_style, _RADIO_HOST_STYLES))
    # STORY FLAIR (operator 2026-07-01: "all prompts respect the meta brief"):
    # a radio console / radio object is an OBJECT, not a bare human face, so it
    # takes the FULL "still" era tail (story palette + atmosphere + lighting)
    # -- the palette-strip "portrait" profile (BUG-LOCAL-113, for human faces)
    # is NOT wanted here; a tense story should read as a tense-lit radio.
    prompt = finish_visual_prompt(meta, prompt, era_profile="still",
                                  style=_style)
    if _style.image_grade_tail and _style.image_grade_tail not in prompt:
        prompt = "%s, %s" % (prompt, _style.image_grade_tail)
    return prompt


#: Portrait canvas (the proven FLUX portrait dims; unchanged by the spine).
PORTRAIT_W = 832
PORTRAIT_H = 1216


def _landscape_still_dims():
    """(w, h) for SCENE stills: the landscape composite canvas, each dim
    snapped DOWN to /32 (the latent-grid contract). Env-overridable via
    OTR_VIDEO_LANDSCAPE_CANVAS (the same knob the render driver reads)."""
    import os
    raw = os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")
    try:
        w, h = (int(x) for x in raw.lower().split("x", 1))
    except (ValueError, AttributeError):
        w, h = 1472, 832
    return max(32, (w // 32) * 32), max(32, (h // 32) * 32)


def _video_models_from_policy(policy_json):
    """The ``video_models`` map from an image_policy_json (OTR_ImageDirector
    forwards the resolved per-role video engines here). Missing/malformed -> {}.
    Pure, tolerant."""
    try:
        pol = json.loads(policy_json or "{}")
    except (ValueError, TypeError):
        return {}
    if isinstance(pol, dict) and isinstance(pol.get("video_models"), dict):
        return pol["video_models"]
    return {}


def _still_consumer_capabilities(video_models):
    """Ask the dispatcher-owned authority for every role's tri-state capability.

    ``False`` proves that a role is procedural and its prompt may be omitted;
    ``True`` requires authoring; ``None`` remains uncertain and preserves the
    legacy authoring path.  Only an all-False complete map permits the early
    no-image return.
    """
    if not isinstance(video_models, dict):
        return None
    try:
        try:
            from .otr_image_gen_dispatcher import still_consumer_capabilities  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from otr_image_gen_dispatcher import still_consumer_capabilities  # type: ignore
        return still_consumer_capabilities({"video_models": video_models})
    except _RouteFreezeError:
        # A MALFORMED ROUTING ENVIRONMENT IS TERMINAL (2026-07-25 QA, fourth
        # swallow site). Uncertainty legitimately retains legacy authoring; a
        # mistyped OTR_FORCE_ENGINE_MAP is not uncertainty.
        raise
    except Exception:  # noqa: BLE001 -- uncertainty retains legacy authoring
        return None


def _effective_video_family_for_role(video_models, role: str) -> str:
    """The RESOLVED video-engine family for one role after effective rewrites.

    Tolerant: a resolution/import failure returns ``""`` so callers cannot infer
    a lip-host consumer from an incomplete policy. Pure except the effective
    engine's environment reads.
    """
    eng_id = _effective_prompt_engine_for_role(video_models, role)
    if not eng_id:
        return ""
    try:
        try:
            from ._otr_video_engines.render_driver import engine_family  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_video_engines.render_driver import engine_family  # type: ignore
        return str(engine_family(eng_id, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _effective_announcer_family(video_models) -> str:
    """Compatibility wrapper for the announcer radio-style decision."""
    return _effective_video_family_for_role(video_models, "announcer_visual")


def _effective_prompt_engine_for_role(video_models, role: str) -> str:
    """VIDEO engine MetaBrief should plan stills against for ``role``.

    Selected per-role engine, then force-map, then the radio-is-host HuMo
    redirect -- resolved through the ONE route-freeze authority
    (``_otr_shared.route_freeze``, 2026-07-25 chunk 1a) rather than this
    module's own former mirror of the dispatcher's seam. A resolver-import
    failure keeps the selected id (or empty), which is the fail-safe
    image-prompt behavior; a MALFORMED force map is terminal and propagates,
    because minting prompts for a plan that cannot render is not a fail-safe."""
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
        eng_id = str(_rs.engine_id_for_role(video_models or {}, role) or "")
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return ""
    if not eng_id:
        return ""
    try:
        from ._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore
    return str(_rf.effective_engine_for_role(role, eng_id) or eng_id)


def _engine_wants_talking_prompt(engine_id: str) -> bool:
    """Whether a registered video engine currently wants talking init prompts.

    Mirrors ``OTR_VideoDirector._role_talking`` but runs inside MetaBrief so old
    or override-mutated policies cannot leave the image payload missing a still
    that final render will require. Misconfigured engines return False here; the
    render engine remains the loud hard gate."""
    if not engine_id:
        return False
    try:
        try:
            from ._otr_video_engines import registry as _vreg  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_video_engines import registry as _vreg  # type: ignore
        if not _vreg.is_registered(engine_id):
            return False
        eng = _vreg.get_engine(engine_id)
        fn = getattr(eng, "wants_talking_prompt", None)
        return bool(fn()) if callable(fn) else False
    except Exception:  # noqa: BLE001 -- render path owns the hard failure
        return False


def _effective_talking_roles(talking_roles, video_models):
    """Merge forwarded talking policy with render-effective engine truth.

    ``policy["talking"]`` is correct for a fresh, unforced graph. This helper
    upgrades it when a render-time force-map or radio-is-host redirect means the
    final engine is a talking engine, which is the missing-radio-face failure
    mode for ltx_audio_in announcer bookends."""
    out = {str(k): bool(v) for k, v in (talking_roles or {}).items()}
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
        roles = tuple(_rs.ROLE_TO_VIDEO_SLOT)
    except Exception:  # noqa: BLE001
        roles = ("announcer_visual", "music_visual", "character_video")
    for role in roles:
        if out.get(role):
            continue
        eng_id = _effective_prompt_engine_for_role(video_models, role)
        if _engine_wants_talking_prompt(eng_id):
            out[role] = True
    return out


def _still_aspects_from_policy(policy_json):
    """role -> still aspect ('portrait'|'wide') from a director policy's
    ``aspects`` map. OTR_VideoDirector resolves each per-role video engine to its
    render_aspect, OTR_ImageDirector forwards that map into image_policy_json, and
    MetaBrief reads it here. A missing / malformed policy yields an empty map, so
    the caller defaults portrait and an unwired graph keeps the legacy look.
    Pure."""
    try:
        pol = json.loads(policy_json or "{}")
        asp = pol.get("aspects") if isinstance(pol, dict) else None
        if isinstance(asp, dict):
            return {str(k): str(v) for k, v in asp.items()}
    except (ValueError, TypeError):
        pass
    return {}


def _talking_roles_from_policy(policy_json):
    """role -> bool from a director policy's ``talking`` map (S4b): whether the
    role's SELECTED video engine lip-syncs (wants_talking_prompt / the ia2v
    register). Same forwarding chain as ``aspects``; missing/malformed -> {}
    (legacy portrait styling, exactly as before S4b). Pure."""
    try:
        pol = json.loads(policy_json or "{}")
        talk = pol.get("talking") if isinstance(pol, dict) else None
        if isinstance(talk, dict):
            return {str(k): bool(v) for k, v in talk.items()}
    except (ValueError, TypeError):
        pass
    return {}


#: 3D IMAGE STREAMS (2026-06-21) -- checked-in prompt scaffolds for the mesh
#: fodder fork. The MESH FODDER still is what Hunyuan3D actually meshes, so it
#: must be ONE isolated subject on a neutral plate (a cinematic scene still
#: meshes the whole environment -> the clay blob). The positive scaffold carries
#: the isolation discipline (the pipeline mints stills from the POSITIVE prompt;
#: the POSITIVE carries the isolation). The negative scaffold is
#: checked in for the engines/lanes that DO consume a negative and as the canon
#: of what fodder must NOT contain.
MESH_FODDER_POS_SCAFFOLD = (
    "single centered subject, simple clean unbroken silhouette, smooth solid "
    "form, plain matte solid-colour clothing, short tight neat hair, neutral "
    "symmetrical forward stance, full unoccluded three-quarter view, entire head "
    "and body clearly visible, plain seamless neutral mid-grey studio backdrop, "
    "even soft diffuse frontal lighting, no hard shadows, no props, sharp focus, "
    "full natural color"
)
MESH_FODDER_NEG_SCAFFOLD = (
    "busy background, multiple subjects, occlusion, hands over face, hood, "
    "loose flowing hair, hair wisps, fine surface detail, intricate texture, "
    "frills, thin protrusions, jewellery, dramatic shadow, cast shadow, cropped, "
    "scene, environment, props, text, watermark"
)
#: The BACKGROUND PLATE is the subject-free world the mesh stands in front of.
#: Chunk A2 split: the no-subject/composition GEOMETRY stays Python; the
#: "period-accurate set" LOOK is pack-owned (VisualStyle.plate_look). The
#: composed scaffold survives as the extraction fixture + legacy lane.
BACKGROUND_PLATE_GEOMETRY = (
    "empty establishing environment, no people, no subject, no characters, "
    "wide 16:9 cinematic scene, atmospheric depth"
)
PLATE_LOOK_DEFAULT = "period-accurate set"
BACKGROUND_PLATE_POS_SCAFFOLD = "%s, %s" % (BACKGROUND_PLATE_GEOMETRY,
                                            PLATE_LOOK_DEFAULT)
#: Mesh fodder is rendered near-square/portrait (Hunyuan wants an isolated,
#: fully-in-frame subject), independent of the beat's final video aspect.
MESH_FODDER_W = PORTRAIT_W
MESH_FODDER_H = PORTRAIT_H

#: 2026-06-30 mesh-improve item 1+4: the music-role mesh_fodder subject is
#: the RADIO ITSELF (never a generic story object or a character body), and
#: EVERY music_visual beat (open / inter / close) shares this ONE canonical
#: subject id -- the radio is a single recurring on-air object, so the mesh
#: cache should give it identity continuity exactly like a cast character
#: (char_id), instead of minting a fresh, unrelated mesh per beat
#: (the old "obj_<beat>" fallback). Keyed on the always-present video ROLE
#: (never a line/speaker_role lookup, which is absent for lineless synthetic
#: bookend/inter beats -- the exact case that matters most here).
MESH_RADIO_HOST_SUBJECT_ID = "radio_host"


def _mesh_fodder_roles_from_policy(policy_json):
    """The set of image-prompt roles whose paired video engine requires clean
    mesh fodder (OTR_ImageDirector.mesh_fodder_roles_from_video_policy forwards
    the list). A missing/malformed policy yields an empty set -> NO fork (the
    legacy cinematic-scene-still look). Pure, tolerant."""
    try:
        pol = json.loads(policy_json or "{}")
        roles = pol.get("mesh_fodder_roles") if isinstance(pol, dict) else None
        if isinstance(roles, list):
            return {str(r) for r in roles if str(r)}
    except (ValueError, TypeError):
        pass
    return set()


def _portrait_free_roles_from_policy(policy_json):
    """Roles whose paired VIDEO engine DECLARES it never needs a portrait.

    The fifth member of the lane-derived role-set family
    (``_still_aspects_from_policy`` / ``_mesh_fodder_roles_from_policy`` /
    ``_talking_roles_from_policy`` / ``_still_word_roles_from_policy``): the
    VIDEO lane already tells the image phase the aspect, the kind, the framing
    and which composer to use, and this is the one thing it could not say --
    "do not mint a portrait for this role at all".

    THE TRUTH IS THE LANE'S OWN DECLARATION, never a hardcoded engine name.
    Every adapter ships a ``still_plan`` of ``StillPlanRow``; the cheap family
    (``still_flat`` / ``still_pan`` / ``still_motion`` / ``still_word``) already
    declares ``kind="portrait" ... required="never"``. That row has been correct
    and unread since it was written -- ``still_plan_helpers`` says so in terms
    ("Nothing in this module reads the plan for production"), which is why a
    still_word episode has always minted a portrait per cast member that nothing
    on the lane ever reads. Free on an engine that will draw a face; FATAL on
    one that refuses (Ideogram 4 returns a safety placeholder for a person
    close-up, which is what killed the 2026-08-22 live leg).

    Reading the declaration rather than naming ``still_word`` means every future
    lane -- an ultra-low-VRAM animatediff, anything -- is covered the day it
    declares its plan, with no edit here.

    Resolves each role through the SAME force-map / radio-host redirect the
    render phase uses (``_effective_prompt_engine_for_role``), so a run that
    FORCES or redirects a role is judged on the engine that will actually
    render. A missing/malformed policy, an unregistered engine, or an adapter
    with no plan yields NO exemption -- the portrait is minted, which is the
    pre-existing behaviour and the safe direction. Pure, tolerant.
    """
    try:
        pol = json.loads(policy_json or "{}")
    except (ValueError, TypeError):
        return set()
    if not isinstance(pol, dict):
        return set()
    vm = pol.get("video_models")
    if not isinstance(vm, dict):
        return set()
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
            from ._otr_shared import still_plan_helpers as _sp  # type: ignore
            from ._otr_video_engines import registry as _vreg  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
            from _otr_shared import still_plan_helpers as _sp  # type: ignore
            from _otr_video_engines import registry as _vreg  # type: ignore
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return set()
    roles = set()
    for role in _rs.ROLE_TO_VIDEO_SLOT:
        try:
            eng_id = _effective_prompt_engine_for_role(vm, role)
            if not eng_id or not _vreg.is_registered(eng_id):
                continue
            plan = getattr(_vreg.get_engine(eng_id), "still_plan", None) or ()
            for row in plan:
                if (str(getattr(row, "kind", "")) == _sp.KIND_PORTRAIT
                        and str(getattr(row, "required", "")) == _sp.REQUIRED_NEVER):
                    roles.add(role)
                    break
        except _RouteFreezeError:
            # A MALFORMED ROUTING ENVIRONMENT IS TERMINAL (2026-07-25 QA) --
            # the same rule the still_word role resolver above follows.
            raise
        except Exception:  # noqa: BLE001 -- a bad slot never blocks prompts
            continue
    return roles


def _identity_roles_from_policy(policy_json):
    """Roles whose paired VIDEO lane declares its character stills need a FACE.

    The SIXTH member of the lane-derived role-set family, and the one that fixes
    the split this family created. ``_portrait_free_roles_from_policy`` above
    reads ``kind="portrait" required="never"`` and skips minting the portrait --
    correct, and it fixed a real live failure. But the portrait was doing TWO
    jobs: the video lane's init image, and the seed basis that gives every beat
    of one character the SAME FACE (commit 89e82181). Letting one consumer's
    PIXEL decision also decide IDENTITY silently un-shipped the other, and 71 of
    75 character stills rendered unanchored on 2026-08-26 as a result.

    So identity is now its own declaration on the same row family:
    ``StillPlanRow.identity`` is ``portrait_seed`` (the DEFAULT) or ``none``.
    A lane that mints no portrait pixels still gets a stable face, because the
    basis is a TEXT hash and costs no render.

    Returns ``{role: token}``. A role whose engine is unknown, unregistered, or
    planless is ABSENT from the map, and the caller treats absence as the
    default -- the safe direction, exactly as this family's other members do.
    """
    out = {}
    try:
        pol = json.loads(policy_json or "{}")
    except (ValueError, TypeError):
        return out
    if not isinstance(pol, dict):
        return out
    vm = pol.get("video_models")
    if not isinstance(vm, dict):
        return out
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
            from ._otr_shared import still_plan_helpers as _sp  # type: ignore
            from ._otr_video_engines import registry as _vreg  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
            from _otr_shared import still_plan_helpers as _sp  # type: ignore
            from _otr_video_engines import registry as _vreg  # type: ignore
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return out
    for role in _rs.ROLE_TO_VIDEO_SLOT:
        try:
            eng_id = _effective_prompt_engine_for_role(vm, role)
            if not eng_id or not _vreg.is_registered(eng_id):
                continue
            plan = getattr(_vreg.get_engine(eng_id), "still_plan", None) or ()
            for row in plan:
                if str(getattr(row, "kind", "")) == _sp.KIND_SCENE_CHARACTER:
                    out[role] = str(getattr(row, "identity", "")
                                    or _sp.IDENTITY_PORTRAIT_SEED)
                    break
        except _RouteFreezeError:
            raise          # a malformed routing environment is TERMINAL
        except Exception:  # noqa: BLE001 -- a bad slot never blocks prompts
            continue
    return out


def _identity_token_for_role(identity_roles, role):
    """The identity token for one role, defaulting SAFE.

    Absence means "the lane did not say", and the whole point of the default
    direction is that not saying gets you a stable face for free. Only an
    explicit ``none`` turns it off.
    """
    try:
        from ._otr_shared import still_plan_helpers as _sp  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import still_plan_helpers as _sp  # type: ignore
    token = str((identity_roles or {}).get(str(role or "")) or "")
    return token if token in _sp.VALID_IDENTITY else _sp.IDENTITY_PORTRAIT_SEED


def _still_word_roles_from_policy(policy_json):
    """The set of image-prompt roles whose paired VIDEO engine is ``still_word``.

    still_word is model-agnostic: the per-role video engine is ALREADY forwarded
    in ``image_policy_json["video_models"]`` (OTR_ImageDirector, 2026-07-02), so
    this reads it directly and resolves each role via the shared
    ``role_slots.engine_id_for_role`` join (the SAME rule the director uses) --
    no separate director-side precompute needed. A missing/malformed policy, or
    a role_slots import failure in an odd flat-test context, yields an empty set
    -> NO word/title branch (the legacy cinematic scene still). Pure, tolerant.

    Render-effective seam (r3 + radio-is-host): the render-time override/redirect
    rewrites the engine BEFORE render, and the still dispatcher honors it for
    still-consumption; so a run that FORCES or redirects a role to still_word must
    ALSO mint the word/title prompt (else it renders legacy SCENE prompts on a
    word card). Env-unset/no-redirect -> the resolver returns the id unchanged
    -> byte-identical."""
    try:
        pol = json.loads(policy_json or "{}")
    except (ValueError, TypeError):
        return set()
    if not isinstance(pol, dict):
        return set()
    vm = pol.get("video_models")
    if not isinstance(vm, dict):
        return set()
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return set()
    roles = set()
    for role in _rs.ROLE_TO_VIDEO_SLOT:
        try:
            eng_id = _effective_prompt_engine_for_role(vm, role)
            if eng_id == "still_word":
                roles.add(role)
        except _RouteFreezeError:
            # A MALFORMED ROUTING ENVIRONMENT IS TERMINAL AND MUST PROPAGATE
            # (2026-07-25 QA). This loop's broad catch predates the fail-closed
            # force-map contract and was written when the resolver swallowed
            # everything itself. Left alone it silently returns an EMPTY role
            # set on a typo'd OTR_FORCE_ENGINE_MAP, so a still_word lane
            # quietly mints the wrong prompt template and the episode succeeds
            # with the wrong picture -- the exact "config error becomes a
            # confident wrong render" class the single authority exists to
            # kill. A bad SLOT still never blocks prompts; a bad ENVIRONMENT
            # always does.
            raise
        except Exception:  # noqa: BLE001 -- a bad slot never blocks prompts
            continue
    return roles


#: still_word (Sprint B, 2026-07-03) -- the model-agnostic word/title still.
#: WORD mode (character_video / announcer_visual beats) renders the beat's spoken
#: line as large, readable period typography -- a lyric/title CARD, so the words
#: themselves ARE the image (NO_TEXT_CLAUSE is deliberately NOT appended). TITLE
#: mode (music_visual beats) is an ABSTRACT picture evoking meta["episode_title"]
#: with NO words (NO_TEXT_CLAUSE appended). Pure + deterministic + model-agnostic:
#: it composes only the PROMPT string; whichever image engine the role's image
#: slot selects mints it, and the still_word VIDEO engine holds it flat.
_STILL_WORD_MUSIC_ROLE = "music_visual"

# still_word PROMPT v2 (2026-07-04) -- R1 roundtable + kibitz + Fable-final.
# The word card is: quoted words -> legibility guard -> per-EPISODE locked
# lettering -> per-episode COOL backdrop (+ ONE beat mood on character cards) ->
# authored era tail -> grade -> POSITIVE text guard LAST. The per-episode
# lettering + backdrop are LOCKED per episode (deterministic, no LLM) so every
# word card in an episode reads as ONE consistent typographic set; only the beat
# MOOD adjective varies per character card (the operator's consistency ask).

#: Genre classifier over the SAME haystack radio_form uses (meta.style + brief
#: setting/atmosphere). First match wins; an absent/failed brief OR no keyword
#: match -> "default". The genre KEY (noir|sci-fi|western|pulp|default) is the
#: provenance ``backdrop_family`` value AND the key into both maps below.
_STILL_WORD_GENRE_KEYWORDS = (
    ("noir", ("noir", "detective", "deco", "art deco", "1930s", "1940s",
              "prohibition", "hardboiled", "gumshoe")),
    ("sci-fi", ("space", "orbital", "docking", "spacecraft", "starship",
                "space station", "sci-fi", "science fiction", "futuristic",
                "galactic", "interstellar", "robot", "android", "cybernetic")),
    ("western", ("western", "frontier", "old west", "dustbowl", "prairie",
                 "cowboy", "saloon", "outlaw", "gunslinger")),
    ("pulp", ("pulp", "adventure", "jungle", "ray gun", "monster", "horror",
              "crime", "mystery", "swashbuckling")),
)
#: EXTRACTION FIXTURE (chunk C, 2026-07-05): production reads the visual-style
#: pack (``VisualStyle.still_word_typography``); these three maps survive ONLY
#: as the sci_fi_radio byte-identity fixtures (pinned by test_still_word_maps_
#: match; an AST guard forbids production reads). Edit the PACK, not these.
#:
#: LOCKED per-episode lettering. Every phrase ENDS in "capitals" (Fable: lock the
#: CASE, not just the family -- Ideogram mixes case between renders otherwise;
#: "capitals" is the cheapest per-episode consistency lock). No era/period/
#: letterpress words here (era-bias + ink-squash distress moved OUT of the guard).
_STILL_WORD_TYPOGRAPHY = {
    "noir": "heavy condensed sans-serif capitals, tightly spaced",
    "sci-fi": "wide geometric sans-serif capitals, sleek machined edges",
    "western": "bold wood-type slab-serif capitals, sturdy and squared",
    "pulp": "heavy blocky display serif capitals, hard drop shadow",
    "default": "bold clean sans-serif capitals, evenly spaced",
}
#: COOL backdrop -- COLOR + LIGHT DIRECTION + ATMOSPHERE DENSITY only, NEVER an
#: OBJECT / PARTICLE / PLACE (any countable noun becomes high-frequency texture
#: inside the letter counters). Kept in the DARK value register (the value gap,
#: not saturation, keeps caps legible under the grade).
_STILL_WORD_BACKDROP = {
    "noir": "smoky midnight-blue haze, dim slatted lamplight",
    "sci-fi": "deep void-black gradient, cold teal horizon glow",
    "western": "deep amber dusk haze, low golden horizon glow",
    "pulp": "deep crimson gradient haze, dramatic low uplight glow",
    "default": "soft dark gradient haze, faint warm center glow",
}
#: SPLIT out of the old _STILL_WORD_CARD_STYLE: the genre-NEUTRAL legibility /
#: composition tokens stay fixed; the per-episode lettering supplies the idiom.
#: "filling the frame" is a literal instruction the model acts on; "clear space
#: behind the words" is the counterweight to the grade's vignette.
_STILL_WORD_LEGIBILITY_GUARD = (
    "huge high-contrast lettering, razor-sharp edges, centered, filling the "
    "frame, clear space behind the words")
#: POSITIVE text guard, appended LAST (same seam music mode uses for
#: NO_TEXT_CLAUSE). "no captions" earns its slot -- Ideogram's signature card
#: failure is an invented subtitle line.
_STILL_WORD_TEXT_GUARD = (
    "only the quoted words, no other text, no logos, no captions")
#: WORDLESS abstract-title styling for the music card (UNTOUCHED path).
_STILL_WORD_TITLE_MOOD_STYLE = (
    "abstract evocative mood image, atmospheric period illustration, symbolic "
    "non-literal composition, no lettering")

#: Beat-mood ALLOWLIST (the load-bearing safety property, Fable). ONLY these
#: lemmas emit a mood; everything else (voice descriptors, novel LLM adjectives,
#: typos, "neutral", None, empty) -> NO TOKEN. Buckets scanned in PRIORITY order
#: (danger-forward); first bucket with any hit wins. Output token set is exactly
#: {urgent, tense, somber, hopeful, calm}; "frantic" is a KEYWORD -> urgent.
_STILL_WORD_MOOD_BUCKETS = (
    ("urgent", ("urgent", "urgency", "panic", "panicked", "panicking", "alarm",
                "alarmed", "frantic", "desperate", "hurried", "rushed", "racing")),
    ("tense", ("tense", "tension", "nervous", "anxious", "worried", "afraid",
               "fear", "fearful", "frightened", "terrified", "scared", "dread",
               "uneasy", "suspicious", "angry", "furious", "hostile", "menacing",
               "ominous", "agitated")),
    ("somber", ("somber", "sombre", "sad", "grim", "grief", "grieving",
                "mourning", "melancholy", "solemn", "tragic", "regretful",
                "defeated", "bitter", "haunted")),
    ("hopeful", ("hopeful", "hope", "optimistic", "welcoming", "cheerful",
                 "joyful", "happy", "excited", "eager", "relieved",
                 "triumphant", "upbeat")),
    ("calm", ("calm", "reflective", "thoughtful", "serene", "peaceful",
              "gentle", "measured", "steady", "wistful", "quiet")),
)
#: A title card cannot hold a paragraph, but real dialogue runs 15-30 words --
#: NO hard abort (it would propagate out of derive_image_prompts and kill EVERY
#: episode the first time still_word meets a normal line). On overflow: a
#: DETERMINISTIC REDUCTION (first clause, else first N words / chars), applied
#: BEFORE the hash so the composer stays pure. Blank line still fails LOUD.
STILL_WORD_MAX_WORDS = 12
STILL_WORD_MAX_CHARS = 60


def _still_word_genre(meta) -> str:
    """The locked per-episode genre KEY (noir|sci-fi|western|pulp|default) for the
    still_word lettering + backdrop. Absent/failed brief OR no keyword match ->
    "default" (one neutral look everywhere). Deterministic; no LLM; tolerant."""
    try:
        try:
            from ._otr_story_brief_helpers import (  # type: ignore
                get_story_brief_status, _radio_form_haystack)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_story_brief_helpers import (  # type: ignore
                get_story_brief_status, _radio_form_haystack)
        if get_story_brief_status(meta) != "ok":
            return "default"
        hay = _radio_form_haystack(meta)
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return "default"
    for genre, keys in _STILL_WORD_GENRE_KEYWORDS:
        if any(k in hay for k in keys):
            return genre
    return "default"


def _still_word_mood_from_line(line) -> str:
    """The beat MOOD token from a line's free-form ``traits`` (a NULLABLE string,
    NOT an enum) via the strict ALLOWLIST. None / non-dict / non-str traits /
    "neutral" / voice-only -> "" (NO token; a wrong mood is a visible lie, silence
    is the consistent look). Word-boundary lemma match (never raw substring);
    buckets scanned danger-forward, first bucket with any hit wins. Pure."""
    if not isinstance(line, dict):
        return ""
    traits = line.get("traits")
    if not isinstance(traits, str):
        return ""
    lemmas = {t for t in re.split(r"[^a-z]+", traits.lower()) if t}
    for mood, keys in _STILL_WORD_MOOD_BUCKETS:
        if lemmas.intersection(keys):
            return mood
    return ""


def _fold_inner_dquotes(s) -> str:
    """Fold every inner double-quote (straight + curly + guillemets) to a single
    quote so the ONLY double-quote pair in a still_word card is the template's
    outer ``"%s"`` boundary, and SCRUB backslashes for the same reason. Closes
    the nested-quote ambiguity AND the semantic prompt-injection where a stray
    ``", ...`` closes the span early and injects a sibling clause. Keeps
    apostrophes / em-dash / ellipsis / unicode. Pure.

    The backslash scrub is a quote-shield repair, not cosmetics: a card line
    ending in an ODD backslash run makes the template's CLOSING quote read as
    escaped, so the banana route's span scan finds no shielded span and the
    WHOLE card prompt transforms -- putting a substituted word on the one
    audience-readable surface. A backslash is never legible card lettering.

    Whitespace is re-collapsed only for strings that actually carried a
    backslash, so every backslash-free title stays byte-identical (the word-card
    path re-collapses downstream either way; this keeps MUSIC titles from being
    re-seeded for whitespace they always had).

    A line that is ENTIRELY backslashes would scrub to empty and hit the
    caller's NO-FALLBACK blank raise -- killing a render that composes fine
    today. Such input returns a single apostrophe instead: alive, and already
    this function's own substitution vocabulary. Genuinely empty or
    whitespace-only input still returns empty, so the blank raise keeps working
    for the case it was written for."""
    text = str(s or "")
    if "\\" in text:
        scrubbed = re.sub(r"\s+", " ", text.replace("\\", " ")).strip()
        if not scrubbed and text.strip():
            return "'"
        text = scrubbed
    return re.sub(r'[\"“”„‟«»]', "'", text)


def _still_word_fit_card(words: str) -> str:
    """DETERMINISTIC card-length reduction (fail-SOFT): a title card cannot hold a
    paragraph, so an overflowing line is trimmed to the first sentence, then to
    whole COMMA CLAUSES, then to the first ``STILL_WORD_MAX_WORDS`` words, then a
    ``STILL_WORD_MAX_CHARS`` word-boundary cap, and finally its dangling function
    words are dropped. Under both caps -> returned unchanged, byte for byte.
    Applied BEFORE the hash (pure). NEVER an abort (that would kill every episode
    -- the §4D class).

    THE TWO NEW STAGES ARE WHY A SHIPPED CARD READ *"...of a subterranean"* AND
    STOPPED (2026-08-21 Ideogram verdict, item 2). The reduction was working as
    designed and the design was one stage short: a word-boundary cut is honest
    about never splitting a word, but it says nothing about landing mid-thought,
    and on a card whose ENTIRE JOB is the words a viewer reads that as a sentence
    somebody forgot to finish.

    * **Whole comma clauses, before any word cut.** A clause is a complete
      thought; half a clause is not. This is the stage that actually rescues the
      shipped example, because the trailing word there was an ADJECTIVE and no
      function-word list can catch that without a parts-of-speech tagger this
      repo does not ship.
    * **Never end on a dangling function word.** Shared with the Ghost Signal
      composer through ``_otr_shared.text_tails`` rather than copied, because
      two half-written copies of one rule is a table that drifts once.

    A line that reduces to NOTHING (every word a function word) keeps the
    word-capped text instead of returning empty, so the caller's blank raise
    still fires only for genuinely blank input."""
    try:
        from ._otr_shared import text_tails as _tails  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import text_tails as _tails  # type: ignore

    s = str(words or "")
    if len(s.split()) <= STILL_WORD_MAX_WORDS and len(s) <= STILL_WORD_MAX_CHARS:
        return s
    first = re.split(r"[.!?;:]", s, maxsplit=1)[0].strip()
    if (first and len(first.split()) <= STILL_WORD_MAX_WORDS
            and len(first) <= STILL_WORD_MAX_CHARS):
        s = first
    # WHOLE CLAUSES BEFORE WHOLE WORDS. Only taken when it actually fits both
    # ceilings -- a clause prefix that is still too many words is no better than
    # the text it came from, and falling through to the word cap is correct.
    clause = _tails.longest_clause_prefix(s, STILL_WORD_MAX_CHARS)
    if (clause and len(clause.split()) <= STILL_WORD_MAX_WORDS
            and len(clause) <= STILL_WORD_MAX_CHARS):
        s = clause
    if len(s.split()) > STILL_WORD_MAX_WORDS:
        s = " ".join(s.split()[:STILL_WORD_MAX_WORDS])
    if len(s) > STILL_WORD_MAX_CHARS:
        s = _tails.word_boundary_cut(s, STILL_WORD_MAX_CHARS)
    trimmed = _tails.drop_dangling_tail(s)
    return (trimmed or s).strip()



def _episode_title(meta) -> str:
    """The episode title from meta (``episode_title`` -> ``title``), stripped.
    Empty string when absent. Pure."""
    m = meta if isinstance(meta, dict) else {}
    return str(m.get("episode_title") or m.get("title") or "").strip()


def _still_word_clean_line(beat_line: str) -> str:
    """The spoken words for a word card: bracketed ``[stage direction]`` /
    parenthetical ``(aside)`` scrubbed, a leading ``SPEAKER:`` label removed,
    inner double-quotes FOLDED to single (§7 -- so the template's outer ``"%s"``
    is the only double-quote pair), whitespace collapsed. Apostrophes / ellipsis /
    em-dash inside the words are KEPT (they belong to the line). Pure."""
    s = str(beat_line or "")
    s = re.sub(r"\[[^\]]*\]", " ", s)                      # [stage direction]
    s = re.sub(r"\([^)]*\)", " ", s)                        # (aside)
    s = re.sub(r"^\s*[A-Z0-9 .'\-]{1,40}:\s+", "", s)       # leading SPEAKER:
    s = _fold_inner_dquotes(s)                              # §7 quote-fold
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compose_still_word_prompt(meta, role, beat_line, style=None):
    """The still_word base-still prompt (see the module note above). Pure,
    deterministic, MODEL-AGNOSTIC (no engine reference -- the composed prompt is
    identical regardless of which image engine mints it). Returns a flat ``str``.

    ``beat_line`` accepts a DICT (the frozen ledger line -- its ``text`` is the
    words, its free-form ``traits`` drives the beat mood on CHARACTER cards) OR a
    raw STRING (back-compat; no mood). The third param NAME is pinned by a test.

    WORD mode (character_video / announcer_visual): quoted words -> legibility
    guard -> per-episode LOCKED lettering -> per-episode COOL backdrop (+ one
    ``"<mood> mood"`` on character cards) -> authored era tail -> grade ->
    POSITIVE text guard LAST. FAILS LOUD (``ValueError``, NO FALLBACK) on a BLANK
    line; an overflowing line is REDUCED (never aborted). MUSIC mode is the
    UNTOUCHED wordless abstract-title still. All raises precede any prompt hash."""
    # Stage 3 TOTAL-COVERAGE chunk C (2026-07-05): era + grade + the
    # typography / backdrop / title-mood LOOK VALUES all route through the
    # visual-style pack (ImportError shim only; a style error RAISES). The
    # per-episode genre SELECTOR (_still_word_genre) and the per-episode
    # lettering LOCK stay Python (operator lettering-consistency directive
    # 2026-07-04); only the pack-owned vocabulary moves. ``style`` is the
    # ALREADY-RESOLVED VisualStyle when the composer entry threads it (resolve-
    # once, r4 contract); style=None => fail-loud get_visual_style(meta).
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            get_era_tail, NO_TEXT_CLAUSE, _resolve_style)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            get_era_tail, NO_TEXT_CLAUSE, _resolve_style)
    _role = str(role or "")
    _style = _resolve_style(meta, style)
    # The era tail is the one piece here spliced from LLM-authored brief text
    # (atmosphere_line / visual_palette), so it is the one piece that can carry
    # its own double quotes. Fold it: this function's contract is that the
    # template's outer pair is the ONLY double-quote pair in the card, and a
    # second pair does not merely muddle the prompt -- it SHIELDS whatever it
    # wraps from the banana route, so a quoted `"revolver"` in an atmosphere
    # line survives into the picture. (The pack-owned lettering / backdrop /
    # grade values are authored config and verified quote-free.)
    era = _fold_inner_dquotes(
        (get_era_tail(meta, profile="still", style=_style) or "").strip())
    if _role == _STILL_WORD_MUSIC_ROLE:
        # UNTOUCHED music path (operator 2026-07-04: episode titles are proc-gen
        # elsewhere, so the music card stays a WORDLESS abstract mood still).
        # Fold ONCE, then validate the FOLDED value and compose from that same
        # value -- checking the raw title first would let a backslash-only title
        # pass the blank gate and mint `evoking ""`, while the word path below
        # raises for the equivalent input.
        title = _fold_inner_dquotes(_episode_title(meta))
        if not title:
            raise ValueError(
                "compose_still_word_prompt: music-role still_word beat has a "
                "BLANK episode title (meta['episode_title']/['title']) -- cannot "
                "mint the abstract title still. NO FALLBACK.")
        pieces = ['an abstract picture evoking "%s"' % title,
                  _style.still_word_title_mood_style, era,
                  _style.image_grade_tail]
        out = ", ".join(p.strip().rstrip(",") for p in pieces if p and p.strip())
        return "%s, %s" % (out, NO_TEXT_CLAUSE)         # music title = NO words
    # WORD mode (character_video / announcer_visual): the spoken line IS the card.
    _text = beat_line.get("text") if isinstance(beat_line, dict) else beat_line
    words = _still_word_clean_line(_text)
    if not words:
        raise ValueError(
            "compose_still_word_prompt: %s still_word beat has a BLANK spoken "
            "line after the stage-direction scrub (raw=%r) -- cannot mint the "
            "word card. NO FALLBACK." % (_role or "word", beat_line))
    words = _still_word_fit_card(words)                 # deterministic reduction
    genre = _still_word_genre(meta)
    # Genre is the per-episode LOCKED selector (Python); the VALUE it indexes
    # is pack-owned (chunk C). Exact-key indexing on the pack map -- the loader
    # guarantees the {noir,sci-fi,western,pulp,default} key set, so a genre the
    # selector can return always resolves; a broken pack KeyErrors (fail-loud).
    lettering = _style.still_word_typography[genre]
    backdrop = _style.still_word_backdrop[genre]
    # ONE beat mood adjective on CHARACTER cards only (announcer cards: no mood).
    mood = (_still_word_mood_from_line(beat_line)
            if _role == "character_video" else "")
    if mood:
        backdrop = "%s, %s mood" % (backdrop, mood)
    pieces = ['a title card displaying the words "%s"' % words,
              _STILL_WORD_LEGIBILITY_GUARD, lettering, backdrop, era,
              _style.image_grade_tail, _STILL_WORD_TEXT_GUARD]
    return ", ".join(p.strip().rstrip(",") for p in pieces if p and p.strip())


def _iter_beat_lines(lines):
    """(beat_id, line) pairs mirroring OTR_ShotLock's beat-id scheme exactly
    (line_id or beat_%04d over the NON-SKIPPED lines) so a still minted here
    joins the ShotLock shot rows downstream. Pure."""
    live = [ln for ln in (lines or [])
            if isinstance(ln, dict) and not ln.get("skip")]
    for i, ln in enumerate(live):
        yield str(ln.get("line_id") or f"beat_{i:04d}"), ln


def derive_scene_still_targets(lines, fps: int = 25,
                               *, include_synthetic_closing: bool = False):
    """Still-spine ST-2: the SCENE-STILL targets derived from the LINES via pure
    helpers, never from ``video.shots`` (graph order: image gen runs BEFORE
    ShotLock). Returns ``(targets, warnings)``; each target is
    ``{beat_id, kind, role, source}``.

    EVERY beat carries its OWN scene still (rip-sfx-broll 2026-07-01: the
    pool_n_loop POOLING died with the retired_role_a /
    retired_role_b roles -- there is no shared still pool any more).
    An unmapped speaker_role FAILS LOUD (NO FALLBACKS; the old
    _DEFAULT_VIDEO_ROLE fallthrough is gone).

    The OPEN comes from the same pure helper ShotLock uses
    (``derive_opening_music_beat``). That helper needs the first line's
    ``start_s`` -- which the audio path persists to the DISK ledger, not to
    this node's pre-audio ``script_json``. When timing is UNKNOWN (first
    line carries no ``start_s``) the open target is still emitted
    (``source="scene_pretiming"``, warned LOUD): production always opens on
    the music head gap, and an unused still costs one render while a
    MISSING open still costs the 6/5 look.
    """
    warnings: list = []
    targets: list = []
    seen: set = set()

    def _add(beat_id, kind, role, source, char_id=""):
        if beat_id and beat_id not in seen:
            seen.add(beat_id)
            tgt = {"beat_id": beat_id, "kind": kind,
                   "role": role, "source": source}
            if char_id:
                # BUG 1 (2026-06-20): a scene_character target carries the beat's
                # char_id so derive_image_prompts can lead the WIDE still with that
                # character's appearance (a 16:9 character shot, not a radio booth).
                tgt["char_id"] = char_id
            targets.append(tgt)

    try:  # lazy: one source of truth for the open beat + the role map
        from .otr_shot_lock import (
            OPENING_MUSIC_BEAT_ID, SPEAKER_TO_VIDEO_ROLE,
            CHARACTER_BEARING_ROLES, derive_opening_music_beat)
    except ImportError:  # pragma: no cover -- flat test imports
        from otr_shot_lock import (  # type: ignore
            OPENING_MUSIC_BEAT_ID, SPEAKER_TO_VIDEO_ROLE,
            CHARACTER_BEARING_ROLES, derive_opening_music_beat)

    live = [ln for _bid, ln in _iter_beat_lines(lines)]
    beat, _frames = derive_opening_music_beat({"lines": list(lines or [])},
                                              int(fps or 25))
    if beat is not None:
        _add(str(beat.get("beat_id") or OPENING_MUSIC_BEAT_ID),
             "scene_open", str(beat.get("role") or "music_visual"),
             "scene_timed")
    elif live and live[0].get("start_s") is None:
        warnings.append(
            "scene_open b000: line timing absent (pre-audio ledger); "
            "emitting the open still target OPTIMISTICALLY -- production "
            "opens on the music head gap; an unused still is cheap, a "
            "missing open still loses the 6/5 look")
        _add(OPENING_MUSIC_BEAT_ID, "scene_open", "music_visual",
             "scene_pretiming")

    # Per-beat scene still for every beat (continuity). The earlier
    # announcer/music-only cut left dialogue beats with NO scene still -> the
    # LTX-I2V MISSING-STILL LOUD degrade; now every role gets one per-beat.
    # The image dispatcher (accepts_still) is still the ONE place that decides
    # whether a still is actually minted (visualizer / abstract floor -> 0);
    # audio_driven_face (HuMo) keeps its PORTRAIT and ignores the scene still
    # (render_driver family branch) but KEEPS one as OOM-fallback insurance
    # (humo->still_motion needs a scene still). (open b000 added above.)
    for bid, ln in _iter_beat_lines(lines):
        role_key = str(ln.get("speaker_role") or "").strip().lower()
        role = SPEAKER_TO_VIDEO_ROLE.get(role_key)
        if role is None:
            # NO FALLBACKS (rip-sfx-broll 2026-07-01): never a bare .get()
            # default -- an unmapped role (incl. the retired "sfx") is a
            # producer bug / old ledger and fails LOUD here.
            raise ValueError(
                f"derive_scene_still_targets: line {bid!r} carries unmapped "
                f"speaker_role {role_key!r} (known: "
                f"{tuple(SPEAKER_TO_VIDEO_ROLE)}). The 'sfx' role was "
                f"removed 2026-07-01 (rip-sfx-broll); regenerate the episode."
            )
        if role in CHARACTER_BEARING_ROLES:
            # BUG 1 (2026-06-20 operator directive): a CHARACTER beat gets a
            # per-beat 16:9 CHARACTER still (kind=scene_character) leading with the
            # character's appearance -- NOT a generic radio scene still and NOT the
            # vertical portrait.
            _add(bid, "scene_character", role, "scene_role_map",
                 char_id=str(ln.get("char_id") or ""))
        else:
            _add(bid, "scene_beat", role, "scene_role_map")
    if True:
        # UNCONDITIONAL, and it was NOT until 2026-08-12.
        #
        # This read `not any(speaker_role == "music_open")` -- the SAME defect
        # the closing branch below was just fixed for, one branch away. A
        # pre-audio opening sentinel carrying its AUTHORED id suppressed the
        # `music_opening_001` reservation meant to cover it, exactly as
        # `shot_006_music` suppressed the closing one and killed `fastwan_8gb`.
        #
        # I asserted in the closing branch's comment, in PBUG-20260811-02 and in
        # commit 3446af3f that THIS branch was already unconditional and that
        # the closing one had drifted away from it. That was wrong: both were
        # guarded, and the symmetry I claimed did not exist. A cross-check
        # caught it. The two are symmetric NOW.
        #
        # No guard is needed: `_add` deduplicates by exact beat id through
        # `seen`, so when the real line is present the ordinary per-beat row has
        # already claimed the id and this call is a no-op.
        #
        # THE OPENING OWES THE SAME RESERVATION THE CLOSING GETS BELOW, AND FOR
        # THE SAME REASON. Found live, 2026-07-29, profile otr_w45_word_razzle.
        #
        # The open target minted above carries the SYNTHETIC id
        # OPENING_MUSIC_BEAT_ID ("b000_music_open"). EpisodeAssembler then
        # mirrors the opening cue into ledger.lines with start_s=0.0, so by
        # ShotLock time derive_opening_music_beat returns None (0.0 is inside
        # the 2.0s head gap it requires) and NO synthetic beat is inserted --
        # the mirrored line becomes an ordinary beat named "music_opening_001".
        # The producer was therefore the sole owner of a still for a beat id
        # that no shot in the finished episode ever carries.
        #
        # Render-side consumers USED TO paper the split over with a hardcoded
        # alias (render_driver._canonical_visual_beat_id). It was applied
        # INCONSISTENTLY, which is what made the split reachable two ways:
        #   - merge_jump_still_requests does not canonicalize, so a JUMP
        #     coverage plan on the opening beat asked for a jump-segment still
        #     whose base scene still was filed under the other id, and the
        #     episode died LOUD at the image boundary.
        #   - mesh_stage hit the same split quietly: its plate was minted as
        #     plate_b000_music_open, never joined to music_opening_001, and the
        #     textured hero composited over the floor fallback instead.
        # One state, two id spaces, and only some readers held the map.
        #
        # THE ALIAS IS NOW DELETED (PBUG-20260811-02, third and final swing).
        # Every reader uses the shot's own beat id, so there is one id space and
        # no map to hold. The reservation below is unchanged and still correct:
        # it gives the positioned mirror a producer-owned target under the name
        # it will actually have, which is exactly what the consumer now asks
        # for. What changed is that a MISSED reservation is now a plain missing
        # still rather than a silent join failure under a second name.
        #
        # Reserving the assembler's deterministic id gives that beat a
        # producer-owned target under the name it will actually have. The b000
        # target STAYS: when there is a real head gap and no mirrored opening
        # cue, ShotLock does insert the synthetic beat and that target is the
        # correct one. Which of the two gets used is decided downstream by
        # which beat id the episode ends up with -- and now either answer has a
        # still waiting for it. An unused still costs one render; a missing one
        # reaches video dispatch too late to repair safely.
        _add("music_opening_001", "scene_beat", "music_visual",
             "scene_open_pretiming")
    if include_synthetic_closing:
        # EpisodeAssembler mirrors the closing cue into ledger.lines after
        # this node has already produced image_prompts.  Reserve the same
        # deterministic line id that the assembler will mint so ShotLock's
        # later music_closing_001 shot has a producer-owned scene target.
        # This is intentionally optimistic like the synthetic opening target:
        # an unused still is cheap; a missing still reaches video dispatch too
        # late to repair safely.
        #
        # UNCONDITIONAL, exactly like the opening reservation above (live fix
        # 2026-08-12). This used to read
        # `not any(speaker_role == "music_close")`, which asks a DIFFERENT
        # question than the one that matters: it suppressed the reservation
        # whenever ANY closing cue existed, including the pre-audio sentinel
        # carrying its AUTHORED id (`shot_006_music`). The assembler then
        # mirrored that cue under its own deterministic id, and the still for
        # THAT id had never been minted. `fastwan_8gb` died on it:
        #
        #   RenderError: still-spine handoff missing materialized scene still
        #   for shot shot_music_closing_001 beat music_closing_001
        #
        # NO GUARD IS NEEDED AT ALL. `_add` already deduplicates by exact
        # beat id through `seen`, so when the real `music_closing_001` line is
        # present the ordinary per-beat loop has already claimed the id and
        # this call is a no-op. An explicit scan would be a second copy of that
        # policy, which is how the two branches drifted apart in the first
        # place -- so the closing branch is now literally the opening branch's
        # twin, and for the reason the opening one already gives.
        #
        # SCOPE, and it is the important half: this backstop can only ever
        # cover `_001`. `EpisodeAssembler` mints one row per chunk
        # (`music_{cue}_{NNN}`), so a multi-chunk closing produces `_002`,
        # `_003`... which no reservation reaches. Those are covered because the
        # image producer now reads ShotLock's POST-AUDIO ledger, where every
        # minted mirror already exists and the ordinary loop sees them all.
        # The wiring is the general fix; this is the single-chunk backstop.
        _add("music_closing_001", "scene_beat", "music_visual",
             "scene_close_pretiming")
    return targets, warnings


def objects_by_id(payload) -> dict:
    """``{object_id: object}`` accessor over the versioned ``{"objects":[...]}``
    payload (portrait object_ids are the char_ids). Pure, tolerant."""
    out: dict = {}
    for obj in (payload or {}).get("objects") or []:
        if isinstance(obj, dict) and obj.get("object_id"):
            out.setdefault(str(obj["object_id"]), obj)
    return out


def announcer_line_char_ids(lines) -> list:
    """Distinct ``char_id``s of ledger lines spoken by the ANNOUNCER role, in
    first-appearance order (normally just ``["announcer"]``). The video render
    path resolves ``init_image`` by the LINE's char_id, so prompts are keyed
    the same way. Pure; tolerates malformed rows."""
    out: list = []
    for ln in lines or []:
        if not isinstance(ln, dict):
            continue
        if str(ln.get("speaker_role") or "") != "announcer":
            continue
        cid = str(ln.get("char_id") or "") or ANNOUNCER_CHAR_ID
        if cid not in out:
            out.append(cid)
    return out


def compose_image_prompt_fallback(meta: dict, char: dict, aspect: str = "portrait",
                                  talking: bool = False, style=None) -> str:
    """Deterministic brief-composed portrait prompt -- NEVER empty.

    ``"{appearance}, {setting} setting, {style anchor}"`` with empty parts
    dropped; degrades to the style anchor alone if the brief + cast are bare.
    ``talking`` swaps in the face-forward lip-sync anchor (S4b).
    Chunk A1: ``style`` is the resolved visual-style pack threaded from the
    image-prompt entry (None => this IS the entry: fail-loud resolve).
    """
    try:
        from ._otr_story_brief_helpers import _resolve_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import _resolve_style  # type: ignore
    _vstyle = _resolve_style(meta, style)
    # Same key chain as _appearance_for_char incl. character_description
    # (2026-06-10): this fallback is what actually runs whenever the LLM is
    # unavailable, and it read only the two empty keys -- every character got
    # the identical setting+anchor prompt -> ONE shared portrait.
    appearance = str(
        (char or {}).get("portrait_prompt") or (char or {}).get("appearance")
        or (char or {}).get("character_description") or ""
    ).strip()
    setting = _read_setting(meta)
    parts = []
    if appearance:
        parts.append(appearance)
    if setting:
        parts.append(f"{setting} setting")
    parts.append(_style_anchor_for_aspect(aspect, talking=talking,
                                          style=_vstyle))
    return _ensure_gender_anchor(", ".join(parts), char)


def _build_char_prompt_request(char: dict, meta: dict, setting: str,
                               aspect: str = "portrait",
                               talking: bool = False, style=None) -> str:
    """The instruction handed to the writer LLM (temp=0) for one character.

    ``aspect`` ('wide'|'portrait') drives the framing clause so a 16:9 still asks
    for a head-and-shoulders shot (the head fits the short frame) while a portrait
    still keeps the three-quarter look -- both with explicit headroom so the top of
    the head is never cropped (operator framing catch 2026-06-17). ``talking``
    (S4b) overrides both with the face-forward lip-sync framing: the ia2v
    recipe drives the mouth on THIS still, so a profile / distant / dim face
    is unusable (proof8 catch 2026-07-02). Chunk A1: the LLM-facing LOOK
    language (the old hard-coded "photographic and period-consistent") comes
    from the pack's ``portrait_instruction_look``; ``style`` is the resolved
    pack threaded from the entry (None => fail-loud resolve here)."""
    try:
        from ._otr_story_brief_helpers import _resolve_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import _resolve_style  # type: ignore
    _vstyle = _resolve_style(meta, style)
    appearance = _ensure_gender_anchor(
        _appearance_for_char([char], str(char.get("char_id") or "")),
        char)
    if talking:
        framing = (
            "face-forward frontal close-up bust looking DIRECTLY at the camera, "
            "the whole face and mouth clearly visible and unobstructed (never a "
            "profile, never turned away), the face brightly lit and filling much "
            "of the frame, with headroom so the top of the head is never cropped"
        )
    else:
        framing = (
            "head-and-shoulders medium framing, the subject centred with the FULL head "
            "visible and headroom above the head so the top of the head is never cropped"
            if str(aspect).lower() == "wide" else
            "three-quarter framing showing the full head and upper body, with headroom "
            "above the head so the top of the head is never cropped"
        )
    return (
        "Write ONE vivid still-image portrait prompt (a single comma-separated "
        "line, no preamble) for this character. The image MUST depict the "
        "CHARACTER THEMSELVES -- a person with a clearly visible face, "
        f"{framing} -- IN CHARACTER "
        "inside the story's world. NEVER an empty room, an object, or scenery "
        "alone. Ground it in the appearance and the story setting; keep it "
        f"{_vstyle.portrait_instruction_look}.\n"
        f"character_appearance: {appearance or '(unspecified)'}\n"
        f"story_setting: {setting or '(unspecified)'}\n"
        "style_anchor: "
        f"{_style_anchor_for_aspect(aspect, talking=talking, style=_vstyle)}\n"
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.\n"
        "Do not mention radios, microphones, studios, or any broadcasting "
        "equipment anywhere in the prompt -- the character is a person in "
        "the STORY's world, not a performer at a station.\n"
        "Return only the prompt line."
    )


def _build_char_scene_request(char: dict, meta: dict, setting: str,
                              line: dict, style=None) -> str:
    """BUG 1 follow-up (2026-06-20 operator): the per-beat character still must be
    SHOT/BEAT AWARE -- the character IN the moment of THIS beat -- regardless of
    image model (the video lane conditions on the SAME still). Mirrors
    :func:`_build_char_prompt_request` but WIDE 16:9 and grounded in the beat's
    own ``beat_intent`` / ``traits`` / spoken ``text`` so each character beat
    yields a DISTINCT still. Temp=0 like the portrait path -> deterministic.
    Chunk A1: this builder has NO existing look text, so the pack's
    ``scene_instruction_look`` is appended ONLY when non-empty (sci_fi ships
    "" -- byte-identity by construction, r4 AG M1); ``style`` is the resolved
    pack threaded from the entry (None => fail-loud resolve here)."""
    try:
        from ._otr_story_brief_helpers import _resolve_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import _resolve_style  # type: ignore
    _vstyle = _resolve_style(meta, style)
    _look_line = (f"style_look: {_vstyle.scene_instruction_look}\n"
                  if _vstyle.scene_instruction_look else "")
    appearance = _appearance_for_char([char], str(char.get("char_id") or ""))
    ln = line if isinstance(line, dict) else {}
    intent = str(ln.get("beat_intent") or "").strip()[:240]
    mood = str(ln.get("traits") or "").strip()[:80]
    said = str(ln.get("text") or "").strip()[:240]
    return (
        "Write ONE vivid cinematic STILL-image prompt (a single comma-separated "
        "line, no preamble) for a 16:9 LANDSCAPE shot of this character at THIS "
        "moment of the scene. The image MUST show the CHARACTER THEMSELVES -- a "
        "person with a clearly visible face -- as the subject, a medium/wide shot "
        "with the full head and headroom, framed inside the story's world; convey "
        "the ACTION and EMOTION of this beat. Translate the beat into what is "
        "VISIBLE (pose, expression, what they are doing) -- do NOT write the "
        "character's name, dialogue, narration, or any on-screen text. NEVER an "
        "empty room, an object, or scenery alone.\n"
        f"character_appearance: {appearance or '(unspecified)'}\n"
        f"beat_action: {intent or '(unspecified)'}\n"
        f"emotion: {mood or '(unspecified)'}\n"
        f"they_are_saying: {said or '(unspecified)'}\n"
        f"story_setting: {setting or '(unspecified)'}\n"
        f"style_anchor: {_style_anchor_for_aspect('wide', style=_vstyle)}\n"
        f"{_look_line}"
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.\n"
        "Do not mention radios, microphones, studios, or any broadcasting "
        "equipment -- the character is a person in the STORY's world.\n"
        "Return only the prompt line."
    )


def _compose_char_scene_prompt(meta, char_entry, setting, line, llm_fn,
                               warnings, cid, max_reseed=2, vstyle=None):
    """Compose one beat-aware character still prompt without content gates.

    A writer LLM may refine the visual description. Empty, malformed, or failed
    refinement uses the deterministic same-story composer so visual prompt
    authoring cannot abort an otherwise publishable episode.
    """
    ce = char_entry if isinstance(char_entry, dict) else {}
    prompt = ""
    source = "char_scene_template"
    said = str((line or {}).get("text") or "").strip()
    llm_attempted = llm_fn is not None and bool(said)
    if llm_attempted:
        request = _build_char_scene_request(
            ce, meta, setting, line, style=vstyle,
        )
        for attempt in range(max_reseed + 1):
            try:
                raw = llm_fn(request)
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"char-scene llm_fn raised for {cid} ({exc}); "
                    f"reseed {attempt}"
                )
                raw = ""
            candidate = _clean_llm_prompt(raw)
            if candidate:
                prompt = candidate
                source = "char_scene_llm"
                break

    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            compose_still_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            compose_still_prompt)
    if not prompt:
        if llm_attempted:
            warnings.append(
                f"char-scene LLM produced no usable prompt for {cid}; "
                "using deterministic same-story composition"
            )
            source = "char_scene_template_after_llm_miss"
        return (
            compose_still_prompt(
                meta,
                kind="scene_character",
                role="character_video",
                char_entry=ce,
                style=vstyle,
            ),
            source,
        )

    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    style = _resolve_style(meta, vstyle)
    prompt = finish_visual_prompt(
        meta, prompt, era_profile="portrait", style=style,
    )
    # A shared seed with a RE-WORDED subject is still a different face. The LLM
    # receives the appearance only as a hint and may paraphrase it, while the
    # deterministic composer leads with it verbatim -- so only one of the two
    # paths anchored identity at all. Lead with the same tokens here.
    #
    # Keyed off ce.get('char_id'), mirroring _build_char_scene_request exactly.
    # The separately-passed `cid` disagrees whenever ce carries no char_id key,
    # and then the "same token sequence" premise fails silently.
    _app = _appearance_for_char([ce], str(ce.get("char_id") or ""))
    if not _app:
        warnings.append(
            f"char-scene: no appearance text for {cid}; still has NO identity "
            f"anchor (LOUD)")
    else:
        # The twin laundering removed 2026-08-05 with its root fix: the guard
        # now classifies the whole string, so a scarf described as "black/white"
        # reaches the prompt as written instead of "black or white".
        if _app[:40] not in prompt:
            prompt = f"{_app}, {prompt}"
    if style.image_grade_tail and style.image_grade_tail not in prompt:
        prompt = f"{prompt}, {style.image_grade_tail}"
    if not prompt.endswith(NO_TEXT_CLAUSE):
        prompt = f"{prompt}, {NO_TEXT_CLAUSE}"
    return prompt, source


#: Chunk A2 extraction fixture: the general no-character emblem template is
#: pack-owned (VisualStyle.non_character_emblem_fallback, {base} template);
#: this survives ONLY as the sci_fi_radio fixture + the legacy no-style lane.
NON_CHARACTER_EMBLEM_FALLBACK_DEFAULT = (
    "a single emblematic object representing {base}")


def _mesh_fodder_subject_and_source(meta, char_entry, line, setting, role,
                                    style=None) -> "tuple[str, str]":
    """``(subject, prompt_field_source)`` for a mesh_fodder still -> always a
    single, isolated thing the mesher can carve cleanly. A character beat
    meshes the CHARACTER (appearance); the announcer meshes the announcer
    figure (no studio gear, no occlusion); a music beat meshes the RADIO
    ITSELF (2026-06-30 mesh-improve item 1: "music bookend mesh = a 3D radio,
    not a character body" -- never the old generic "object representing the
    story", which minted an arbitrary, unrelated prop); any OTHER
    no-character beat meshes ONE emblematic story object -- chunk A2: that
    emblem TEMPLATE is pack-owned (non_character_emblem_fallback, formatted
    with {base}); music_visual keeps radio_form_from_meta (brief axis, r1
    codex M2 + AG M3). ``style=None`` is the legacy fixture lane. Pure;
    never empty (a bare fallback keeps the mesher fed). The source tag is
    the chunk-A2 provenance value for the minted object."""
    appearance = ""
    if char_entry:
        appearance = _appearance_for_char(
            [char_entry], str((char_entry or {}).get("char_id") or ""))
    if appearance:
        return appearance, "cast:appearance"
    if str(role) == "announcer_visual":
        # C2 (2026-07-01): the announcer mesh is the FACELESS radio OBJECT (brief
        # -driven form), NOT a person-with-a-face -- "only HuMo gets a face". The
        # old "1940s radio announcer in a suit" put a human figure into the mesh,
        # violating that invariant AND hardcoding the era. Faceless + isolated.
        return radio_form_from_meta(meta), "brief:radio_form"
    if str(role) == "music_visual":
        # The music open/inter/close mesh IS the radio -- the recurring on-air
        # object, isolated and clean for the mesher. Brief-driven form (C2), no
        # hardcoded 1940s set. NOT the emblem fallback (r1 codex M2 + AG M3).
        return radio_form_from_meta(meta), "brief:radio_form"
    # No-character beat outside announcer/music: a single emblematic OBJECT
    # from the story world (chunk 6 refines the object_id policy; this keeps
    # the mesher fed cleanly).
    intent = str((line or {}).get("beat_intent") or "").strip()[:120]
    base = intent or setting or "the story"
    template = (style.non_character_emblem_fallback if style is not None
                else NON_CHARACTER_EMBLEM_FALLBACK_DEFAULT)
    return template.format(base=base), "non_character_emblem_fallback"


def _mesh_fodder_subject(meta, char_entry, line, setting, role,
                         style=None) -> str:
    """Subject-only wrapper of :func:`_mesh_fodder_subject_and_source`."""
    return _mesh_fodder_subject_and_source(
        meta, char_entry, line, setting, role, style=style)[0]


def _mesh_style_spice(style) -> str:
    """Short non-default style cue for clean 3D fodder stills.

    Mesh fodder must stay isolated and simple; use only the first positive-tail
    clause so flat/empty-era packs still reach the 3D image stream.
    """
    if str(getattr(style, "style_id", "") or "") == "sci_fi_radio":
        return ""
    raw = str(getattr(style, "positive_tail", "") or "").strip()
    return raw.split(",", 1)[0].strip().rstrip(".")


def _compose_mesh_fodder_prompt(meta, char_entry, line, setting, role,
                                style=None) -> str:
    """``"{subject}, {scaffold}"`` -- the isolated-subject still
    Hunyuan3D meshes. Deterministic (no LLM): the subject identity comes from
    the cast appearance / announcer figure / story object, and the checked-in
    scaffold enforces the neutral-plate isolation. Never empty. Chunk A2:
    ``style`` is the resolved pack threaded from the image-prompt entry
    (None => this IS the entry: fail-loud resolve)."""
    # Stage 3 de-swallow (kibitz r3 -- the seam ALL FOUR reviewers caught):
    # ImportError shim only; a style error RAISES.
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            _resolve_style, get_era_tail)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            _resolve_style, get_era_tail)
    _vstyle = _resolve_style(meta, style)
    subject = _mesh_fodder_subject(meta, char_entry, line, setting, role,
                                   style=_vstyle)
    out = "%s, %s" % (subject, MESH_FODDER_POS_SCAFFOLD)
    spice = _mesh_style_spice(_vstyle)
    if spice and spice.lower() not in out.lower():
        out = "%s, %s" % (out, spice)
    # C4 (2026-07-01): the mesh radio inherits the brief's era TEXTURE (trimmed
    # still profile: atmosphere line + palette/lighting top-2), so a non-1940s
    # brief carries through to the meshed object. The isolation scaffold above
    # still governs silhouette/plate; the tail only tints the look.
    tail = get_era_tail(meta, profile="still", style=_vstyle)
    if tail and tail not in out:
        out = "%s, %s" % (out, tail)
    return out


def _compose_background_plate_prompt(meta, setting, style=None) -> str:
    """The subject-free 16:9 world plate the mesh stands in front of. Leads with
    the story setting, then the 'no subject' GEOMETRY scaffold + the pack's
    ``plate_look`` (chunk A2 split; sci_fi byte-identical by construction).
    Deterministic; never empty. ``style`` threaded from the entry (None =>
    fail-loud resolve here)."""
    # Era tail / grade so the plate matches the episode look. Stage 3
    # de-swallow (kibitz r2 M1): ImportError shim only; a style error RAISES.
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    _style = _resolve_style(meta, style)
    parts = []
    if setting:
        parts.append("%s setting" % setting)
    parts.append("%s, %s" % (BACKGROUND_PLATE_GEOMETRY, _style.plate_look))
    plate = ", ".join(parts)
    plate = finish_visual_prompt(meta, plate, era_profile="still",
                                 style=_style)
    if _style.image_grade_tail and _style.image_grade_tail not in plate:
        plate = "%s, %s" % (plate, _style.image_grade_tail)
    if not plate.endswith(NO_TEXT_CLAUSE):
        plate = "%s, %s" % (plate, NO_TEXT_CLAUSE)
    return plate


def _clean_llm_prompt(raw: str) -> str:
    """First non-empty line of the LLM output, trimmed; '' if unusable."""
    if not raw:
        return ""
    for line in str(raw).splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""


def derive_image_prompts(cast: list, meta: dict, *, llm_fn=None, max_reseed: int = 2,
                         consistency_gate_warn_only: bool = False, lines=None,
                         fps: int = 25, still_aspects=None,
                         mesh_fodder_roles=None, talking_roles=None,
                         still_word_roles=None, video_models=None,
                         portrait_free_roles=None, identity_roles=None):
    """ONE versioned image-object payload: ``{"version": 1, "objects": [...]}``
    (still-spine ST-2 / pass-02 item 1: portraits MIGRATED to the object
    schema in the same patch; no dual-schema shims).

    Each object carries ``object_id`` / ``kind`` / ``role`` / ``w`` / ``h`` /
    ``prompt`` / ``prompt_hash`` / ``source`` plus ``char_id`` (portraits;
    object_id == char_id) or ``beat_id`` (scene stills).

    Portrait path: a bounded writer-LLM refinement is used when it
    returns a non-empty line. Empty or failed refinement uses the deterministic
    same-story template. No Python vocabulary or overlap classifier can reject,
    rewrite, or block the prompt. The compatibility gate argument is ignored.
    Returns ``(payload, warnings)``.

    ``lines`` (optional, the frozen ledger lines, READ-ONLY): announcer
    portrait minting (as before) PLUS the v1 scene-still targets
    (open/announcer/outro via :func:`derive_scene_still_targets`).
    """
    warnings: list = []
    del consistency_gate_warn_only
    still_capabilities = _still_consumer_capabilities(video_models)
    if (still_capabilities is not None
            and all(value is False for value in still_capabilities.values())):
        # All effective video engines are procedural.  This return is before
        # style resolution and every prompt/LLM path: no image payload means no
        # downstream image render and no upstream visual authoring to fail.
        return {"version": 1, "objects": []}, warnings

    def _still_required(role: str) -> bool:
        return (still_capabilities is None
                or still_capabilities.get(role) is not False)

    setting = _read_setting(meta)
    # Chunk A1 (r3 threading): resolve the visual style ONCE at the image-
    # prompt ENTRY and thread it down (helpers never re-resolve). Fail-loud:
    # an unknown meta["visual_style"] stops the episode HERE, before any
    # prompt is composed. ImportError shim only (3A pattern).
    try:
        from ._otr_story_brief_helpers import _resolve_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import _resolve_style  # type: ignore
    _vstyle = _resolve_style(meta)
    talking_roles = _effective_talking_roles(talking_roles, video_models)
    out: dict = {}
    roster = list(cast or [])
    cast_ids = {str(c.get("char_id") or "") for c in roster if isinstance(c, dict)}
    if _still_required("announcer_visual"):
        for cid in announcer_line_char_ids(lines):
            if cid in cast_ids:
                continue                  # a real cast row already covers it
            roster.append({
                "char_id": cid,
                "_synthetic_announcer": True,
            })
    for char in roster:
        if not isinstance(char, dict):
            continue
        cid = str(char.get("char_id") or "")
        if not cid:
            continue
        _is_announcer_row = bool(
            char.get("_synthetic_announcer")
            or str(char.get("name") or "").strip().upper() == "ANNOUNCER")
        _role = "announcer_visual" if _is_announcer_row else "character_video"
        if not _still_required(_role):
            continue
        appearance = _appearance_for_char([char], cid)
        # Framing aspect: synthetic announcers follow announcer_visual, cast
        # characters follow character_video -- so a WIDE video engine gets a
        # head-and-shoulders still the wide render won't decapitate (2026-06-17).
        _aspect = (still_aspects or {}).get(
            "announcer_visual" if _is_announcer_row
            else "character_video", "portrait")
        # TALKING lane (S4b): a cast portrait whose character_video engine
        # lip-syncs (the ia2v register) is the render INIT for the mouth --
        # mint it face-forward + warm, and skip the era/grade tails below
        # (mirrors the ltx_radio_mouth split that fixed the radio bookends).
        # Synthetic announcers keep their radio-host path untouched.
        _talking = bool((talking_roles or {}).get("character_video")) and (
            not _is_announcer_row)
        # The synthetic announcer/radio-host prompt is brief-driven and
        # stamped directly because it follows a distinct faceless/animated
        # radio-host visual contract. The negative is populated on the object row.
        if _is_announcer_row:
            # RADIO FACE LOGIC (2026-07-04): the announcer portrait gets a FACE
            # only when an AUDIO-DRIVEN engine will animate it -- i.e. HuMo hosts
            # opted ON AND the RESOLVED announcer engine family (after
            # OTR_FORCE_ENGINE_MAP) is audio_driven_face. Otherwise (default env,
            # static/i2v engine, or a force-map to a static engine) it is the
            # FACELESS radio_object. The env flag stays a conjunct because it
            # gates whether HuMo survives to the announcer at render (else
            # _enforce_radio_is_host redirects it to ltx_audio_in). The chosen
            # style is STAMPED (radio_host_style) so the cross-node render guard
            # can fail CLOSED on a stale ledger (BUG-LOCAL-129 face-only failure).
            _humo_face = (_humo_hosts_enabled()
                          and _effective_announcer_family(video_models)
                          == "audio_driven_face")
            _astyle = "console_face" if _humo_face else "radio_object"
            _aprompt = build_radio_host_prompt(meta, _aspect,
                                               radio_host_style=_astyle,
                                               vstyle=_vstyle)
            out[cid] = {
                "prompt": _aprompt,
                "prompt_hash": _content_hash(_aprompt),
                "source": "announcer_template",
                "_role": "announcer_visual",
                "negative_prompt": radio_host_negative(_astyle),
                "radio_host_style": _astyle,
            }
            continue
        prompt = ""
        source = "template"
        if llm_fn is not None:
            req = _build_char_prompt_request(char, meta, setting, _aspect,
                                             talking=_talking, style=_vstyle)
            for attempt in range(max_reseed + 1):
                try:
                    raw = llm_fn(req)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"image prompt llm_fn raised for {cid} ({exc}); reseed {attempt}")
                    raw = ""
                cand = _clean_llm_prompt(raw)
                if cand:
                    prompt = cand
                    source = "llm"
                    break
                if attempt < max_reseed:
                    warnings.append(f"empty image prompt for {cid}; reseed {attempt + 1}/{max_reseed}")
        if not prompt:
            prompt = compose_image_prompt_fallback(
                meta,
                char,
                _aspect,
                talking=_talking,
                style=_vstyle,
            )
            if llm_fn is not None:
                warnings.append(
                    f"image prompt LLM produced no usable prompt for {cid}; "
                    "using deterministic same-story composition"
                )
                source = "template_after_llm_miss"
            else:
                source = "template"
        prompt = _ensure_gender_anchor(prompt, char)
        if char.get("_synthetic_announcer"):
            source = "announcer_" + source   # traceable in reports/ledger
        # Finish after optional refinement and before hashing so the
        # stamped hash matches the rendered prompt.
        # TALKING lane (S4b): SKIP both tails -- the era atmosphere + grade
        # darken/mute the face exactly like the brief palette murked the
        # radio still (proof8: near-black profile portraits, motion 0.23) --
        # and end on the canonical warm light instead (idempotent; the
        # talking anchor already carries it on the template path).
        if _talking:
            if "warm dramatic lighting" not in prompt:
                prompt = f"{prompt}, warm dramatic lighting"
        else:
            # Stage 3 de-swallow (kibitz r2 M1): ImportError shim only; a
            # style-resolution error RAISES (never a silent unstyled portrait).
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt)
            # era_profile="portrait": never bleeds the episode's ambient
            # colour palette into character faces (sci-fi = blue wash,
            # period drama = red wash). Only the atmosphere mood line is
            # safe; full palette is explicitly excluded (BUG-LOCAL-113).
            # Chunk A1: _vstyle is the ONE entry-resolved pack (no re-resolve).
            prompt = finish_visual_prompt(meta, prompt,
                                          era_profile="portrait",
                                          style=_vstyle)
            # BUG-411 (operator 2026-06-14: "keep ALL flux consistent with the
            # 6/5 aesthetic"): append the cinematic GRADE tail to PORTRAITS too,
            # so a still_pan beat standing in for a HuMo portrait shows the same
            # graded look as the scene stills/bookend (still_pan
            # animates the minted PNG, so the PNG must carry the grade). The
            # radio broadcast-distress tail stays scene-still-only (a person is
            # not a radio set). Idempotent -- never duplicates. Stage 3: the
            # grade comes from the pack (empty string = no append).
            if (_vstyle.image_grade_tail
                    and _vstyle.image_grade_tail not in prompt):
                prompt = f"{prompt}, {_vstyle.image_grade_tail}"
        out[cid] = {
            "prompt": prompt,
            "prompt_hash": _content_hash(prompt),   # hash AFTER the call
            "source": source,
            "_role": ("announcer_visual" if _is_announcer_row
                      else "character_video"),
        }

    # ---- assemble the ONE versioned object payload (pass-02 item 1) ----
    # Character-still dims follow the SELECTED video engine's aspect: the operator
    # picks humo_1.7B (portrait) vs humo_1.7B_169 (wide) in the role dropdown,
    # OTR_VideoDirector resolves each role's aspect, and generate() passes that
    # role->aspect map in as still_aspects. ONE pick aligns the still to the
    # engine; no map (None) -> portrait (the legacy look, byte-identical).
    from ._otr_shared.aspect import still_dims_for_aspect
    _role_aspects = still_aspects or {}
    objects: list = []
    _portrait_free = set(portrait_free_roles or ())
    # `_role` is POPPED below, so capture the map first -- the identity
    # transport further down needs each character's role to ask the lane
    # what that role declared, and by then the key is gone.
    _role_for_cid = {c: str(i.get("_role") or "character_video")
                     for c, i in out.items()}
    for cid, pinfo in out.items():
        _role = pinfo.pop("_role", "character_video")
        # The VIDEO lane declared it never needs a portrait (StillPlanRow
        # kind=portrait required=never), so do not mint one. This is the same
        # lane-drives-the-image-phase seam as still_aspects / mesh_fodder_roles
        # / talking_roles / still_word_roles -- portraits were simply the one
        # kind minted unconditionally, which every engine tolerated until one
        # refused to draw a face.
        if _role in _portrait_free:
            continue
        _pw, _ph = still_dims_for_aspect(
            _role_aspects.get(_role, "portrait"), PORTRAIT_W, PORTRAIT_H)
        _pobj = {
            "object_id": cid,                 # portrait object_id == char_id
            "kind": "portrait",
            "role": _role,
            "char_id": cid,
            "w": _pw, "h": _ph,
            "prompt": pinfo["prompt"],
            "prompt_hash": pinfo["prompt_hash"],
            "source": pinfo["source"],
        }
        # 2026-07-01: populate the still's existing negative_prompt (schemas.py;
        # no schema change) when set -- e.g. the radio-host "no baby" negative.
        if pinfo.get("negative_prompt"):
            _pobj["negative_prompt"] = pinfo["negative_prompt"]
        # RADIO FACE LOGIC (2026-07-04): carry the announcer portrait's chosen
        # radio-host style onto the object so it survives image dispatch to the
        # ledger row, where the render-side guard fails CLOSED if a faceless
        # radio_object is about to feed a HuMo (audio_driven_face) announcer init.
        if pinfo.get("radio_host_style"):
            _pobj["radio_host_style"] = pinfo["radio_host_style"]
        objects.append(_pobj)

    # RADIO-HOST FACE object (2026-07-01, chunk 3) -- the ONE animatable radio
    # face for a *proven* HuMo bookend consumer. Music is always a radio; it gets
    # lips only when the effective music engine is audio-driven, just as an
    # audio-driven announcer does. A regular still/visualizer music lane gets its
    # normal faceless radio scene still instead. When both bookends are HuMo, use
    # the music slot as the longstanding shared host owner; when only one is HuMo,
    # stamp that role so the dispatcher resolves its valid image slot.
    _radio_host_roles = tuple(
        role for role in ("music_visual", "announcer_visual")
        if (_still_required(role)
            and _effective_video_family_for_role(video_models, role)
            == "audio_driven_face")
    )
    if _humo_hosts_enabled() and _radio_host_roles:
        _rh_role = _radio_host_roles[0]
        _rh_aspect = _role_aspects.get(_rh_role) or "portrait"
        _rh_prompt = build_radio_host_prompt(meta, _rh_aspect,
                                             radio_host_style="console_face",
                                             vstyle=_vstyle)
        _rhw, _rhh = still_dims_for_aspect(_rh_aspect, PORTRAIT_W, PORTRAIT_H)
        objects.append({
            "object_id": RADIO_HOST_PORTRAIT_ID,
            "kind": "portrait",
            "role": _rh_role,
            "w": _rhw, "h": _rhh,
            "prompt": _rh_prompt,
            "prompt_hash": _content_hash(_rh_prompt),
            "negative_prompt": radio_host_negative("console_face"),
            "source": "radio_host_portrait",
        })

    # ADDENDUM -- the WIDE ltx talking radio-face stills for the
    # ltx_audio_in bookends. WIDE by construction (aspect="wide") so the wide
    # ltx_audio_in engine is never fed a pillarboxed portrait -- the exact trap
    # the kibitz flagged. Per-role object_id matches
    # render_driver._ltx_radio_face_object_id; seed-pinned in the dispatcher
    # (shares the bookend seed). No-baby negative populated.
    # When a bookend role's engine lip-syncs (talking_roles via the director
    # policy), the radio-face still mints. A non-audio-driven music engine stays
    # a faceless radio scene; an explicitly LTX-audio music engine gets lips.
    _talk = talking_roles or {}
    if any(_talk.get(r) and _still_required(r) for r in _LTX_RADIO_FACE_ROLES):
        _fw, _fh = still_dims_for_aspect("wide", PORTRAIT_W, PORTRAIT_H)
        for _abrole in _LTX_RADIO_FACE_ROLES:
            if not (_talk.get(_abrole) and _still_required(_abrole)):
                continue
            # Talking-radio kibitz r1 (2026-07-01): the ltx bookend still uses the
            # LTX-ONLY mouth-forward style. The radio IS the host, and ltx_audio_in
            # (no face detector) needs the biggest, clearest mouth region to drive
            # -- the Sub-plan-C probe lever. The same literal radio-with-lips
            # contract applies to announcer and explicitly audio-driven music;
            # ordinary music remains the faceless radio scene. The HuMo
            # console_face look stays a separate family-specific path.
            _fprompt = build_radio_host_prompt(
                meta, "wide", radio_host_style="ltx_radio_mouth",
                vstyle=_vstyle)
            objects.append({
                "object_id": _ltx_radio_face_object_id(_abrole),
                "kind": "portrait",
                "role": _abrole,
                "w": _fw, "h": _fh,
                "prompt": _fprompt,
                "prompt_hash": _content_hash(_fprompt),
                "negative_prompt": radio_host_negative("ltx_radio_mouth"),
                "source": "ltx_radio_face",
            })

    # SCENE-STILL objects (ST-2): open/announcer/outro from pure helpers on
    # the LINES -- never video.shots (image gen runs BEFORE ShotLock). The
    # prompt comes from the shared 5-layer composer (subject parity with the
    # driver's text prompts is locked in tests); no LLM call is needed.
    scene_targets, scene_warns = ([], [])
    if lines:
        try:
            scene_targets, scene_warns = derive_scene_still_targets(
                lines, fps=fps,
                include_synthetic_closing=bool(
                    still_capabilities is None
                    or still_capabilities.get("music_visual") is not False))
        except ValueError:
            # NO FALLBACKS (rip-sfx-broll 2026-07-01): an unmapped
            # speaker_role (e.g. an old "sfx" ledger) is a hard error,
            # never downgraded to a missing-stills warning.
            raise
        except Exception as exc:  # noqa: BLE001 -- deterministic target failure
            # A missing target is an image/video contract failure, not a prose
            # quality decision. Do not let the prompt node hand the dispatcher
            # an apparently valid payload with no scene-still spine.
            raise ValueError(
                f"scene-still derivation failed ({exc}); refusing an image "
                "payload without deterministic scene targets") from exc
    warnings.extend(scene_warns)
    if scene_targets:
        try:
            from ._otr_story_brief_helpers import (  # type: ignore
                compose_still_prompt)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_story_brief_helpers import (  # type: ignore
                compose_still_prompt)
        sw, sh = _landscape_still_dims()
        # BUG 1 (2026-06-20): cast-row + per-beat line lookup so a scene_character
        # still leads with THAT character's appearance AND is BEAT-AWARE (the
        # character in the moment of THIS beat). Image-model agnostic -- the engine
        # that mints it is resolved by the character_video role downstream, and the
        # video lane conditions on the same still.
        _cast_by_id = {str(c.get("char_id") or ""): c
                       for c in roster if isinstance(c, dict)}
        _line_by_beat = {bid: ln for bid, ln in _iter_beat_lines(lines)}
        _fodder_roles = set(mesh_fodder_roles or ())
        _still_word_roles = set(still_word_roles or ())
        for tgt in scene_targets:
            if not _still_required(str(tgt.get("role") or "")):
                continue
            _cid = str(tgt.get("char_id") or "")
            _src = tgt["source"]
            _bid = tgt["beat_id"]
            if str(tgt.get("role") or "") in _still_word_roles:
                # still_word (Sprint B, 2026-07-03): this beat's VIDEO engine is
                # the model-agnostic still_word engine, so its base still is
                # minted from a WORD/TITLE-driven prompt (char/announcer -> the
                # beat's spoken line as a word card; music -> an abstract picture
                # of the episode title, no words), NOT the cinematic scene
                # composer. ONE scene-still object (identical shape to the normal
                # branch, so _still_index picks it up); the delta is purely the
                # prompt. Fail-LOUD (blank line / blank title raises) is inside
                # compose_still_word_prompt. Image-model AGNOSTIC: only the PROMPT
                # is set here; the role's chosen image engine mints it.
                # Pass the LINE DICT (not just the text) so the composer can
                # derive the beat mood from the line's free-form traits; music
                # beats ignore it (title-driven). Back-compat: a str also works.
                _ln = _line_by_beat.get(_bid, {})
                _wprompt = compose_still_word_prompt(
                    meta, str(tgt.get("role") or ""), _ln, style=_vstyle)
                _wobj = {
                    "object_id": f"still_{_bid}",
                    "kind": tgt["kind"],
                    "role": tgt["role"],
                    "beat_id": _bid,
                    "w": sw, "h": sh,
                    "prompt": _wprompt,
                    "prompt_hash": _content_hash(_wprompt),
                    "source": "still_word",
                    # Chunk A2 provenance (additive; r4 codex M3).
                    "visual_style": _vstyle.style_id,
                }
                # Provenance for operator QA: the LOCKED per-episode lettering +
                # backdrop family (music cards excluded -- they carry no lettering).
                if str(tgt.get("role") or "") != _STILL_WORD_MUSIC_ROLE:
                    _swg = _still_word_genre(meta)
                    # Chunk C: lettering VALUE is pack-owned (the per-episode
                    # genre lock stays Python); stamped from the resolved style.
                    _wobj["lettering_style"] = (
                        _vstyle.still_word_typography[_swg])
                    _wobj["backdrop_family"] = _swg
                    _wobj["prompt_field_source"] = (
                        "still_word_typography:%s" % _swg)
                else:
                    _wobj["prompt_field_source"] = (
                        "still_word_title_mood_style")
                if _cid:
                    _wobj["char_id"] = _cid
                objects.append(_wobj)
                continue
            if str(tgt.get("role") or "") in _fodder_roles:
                # 3D IMAGE STREAMS FORK (2026-06-21): this beat's video engine
                # requires clean mesh fodder, so mint TWO objects instead of one
                # cinematic scene still -- a mesh_fodder SUBJECT (what Hunyuan3D
                # carves) + a subject-free scene_background_plate (the world the
                # mesh stands in front of). NO generic scene_* still for this beat
                # (else _still_index's last-write-wins could return the wrong row).
                _ce = _cast_by_id.get(_cid)
                _ln = _line_by_beat.get(_bid, {})
                # 2026-06-30 mesh-improve item 4: every music_visual beat
                # (open/inter/close) shares ONE canonical radio_host subject id
                # -- identity continuity for the recurring on-air object. Keyed
                # on the video ROLE (always present), never a line/speaker_role
                # lookup (absent for lineless synthetic bookend/inter beats).
                _subj_id = _cid or (
                    MESH_RADIO_HOST_SUBJECT_ID
                    if str(tgt.get("role") or "") == "music_visual"
                    else "obj_%s" % _bid)
                _fprompt = _compose_mesh_fodder_prompt(
                    meta, _ce, _ln, setting, tgt["role"], style=_vstyle)
                # Chunk A2 provenance: which arm fed the mesh SUBJECT (the
                # same pure decision the composer just made).
                _ffield = _mesh_fodder_subject_and_source(
                    meta, _ce, _ln, setting, tgt["role"], style=_vstyle)[1]
                _fobj = {
                    "object_id": "meshfodder_%s" % _bid,
                    "kind": "mesh_fodder",
                    "role": tgt["role"],
                    "beat_id": _bid,
                    "mesh_subject_id": _subj_id,
                    "w": MESH_FODDER_W, "h": MESH_FODDER_H,
                    "prompt": _fprompt,
                    "negative_prompt": MESH_FODDER_NEG_SCAFFOLD,
                    "prompt_hash": _content_hash(_fprompt),
                    "source": "mesh_fodder",
                    "visual_style": _vstyle.style_id,
                    "prompt_field_source": _ffield,
                }
                if _cid:
                    _fobj["char_id"] = _cid
                objects.append(_fobj)
                _pprompt = _compose_background_plate_prompt(meta, setting,
                                                            style=_vstyle)
                objects.append({
                    "object_id": "plate_%s" % _bid,
                    "kind": "scene_background_plate",
                    "role": tgt["role"],
                    "beat_id": _bid,
                    "w": sw, "h": sh,
                    "prompt": _pprompt,
                    "prompt_hash": _content_hash(_pprompt),
                    "source": "mesh_background_plate",
                    "visual_style": _vstyle.style_id,
                    "prompt_field_source": "plate_look",
                })
                continue
            if tgt["kind"] == "scene_character":
                _ce = _cast_by_id.get(_cid)
                _ln = _line_by_beat.get(tgt["beat_id"], {})
                sprompt, _csrc = _compose_char_scene_prompt(
                    meta, _ce, setting, _ln, llm_fn, warnings, _cid,
                    vstyle=_vstyle)
                _src = _csrc
                _sfield = "cast:appearance"
            else:
                sprompt = compose_still_prompt(
                    meta, kind=tgt["kind"], role=tgt["role"],
                    beat_id=tgt["beat_id"], style=_vstyle)
                # Chunk A2 provenance: the open/beat scene still leads with
                # the pack open_subjects template -- stamp the exact key via
                # the SAME pure selector compose_still_prompt used.
                try:
                    from ._otr_story_brief_helpers import (  # type: ignore
                        open_subject_key)
                except ImportError:  # pragma: no cover -- flat test imports
                    from _otr_story_brief_helpers import (  # type: ignore
                        open_subject_key)
                _sfield = "open_subjects:%s" % open_subject_key(
                    tgt["role"], tgt["kind"] == "scene_open")
            _obj = {
                "object_id": f"still_{tgt['beat_id']}",
                "kind": tgt["kind"],
                "role": tgt["role"],
                "beat_id": tgt["beat_id"],
                "w": sw, "h": sh,
                "prompt": sprompt,
                "prompt_hash": _content_hash(sprompt),
                "source": _src,
                "visual_style": _vstyle.style_id,
                "prompt_field_source": _sfield,
            }
            if _cid:
                _obj["char_id"] = _cid     # traceability; engine resolves by role
                # IDENTITY TRANSPORT (2026-08-26). A scene_character still seeds
                # from the CHARACTER'S PORTRAIT prompt so every beat of that
                # character draws one face (commit 89e82181). When the video lane
                # declares it never needs portrait PIXELS the portrait object is
                # skipped above (:2012-2022) -- correctly, that is a88cede5's fix
                # -- but the prompt itself was still COMPOSED into `out[cid]` and
                # then discarded, taking the seed basis with it. 71 of 75
                # character stills rendered unanchored on 2026-08-26 because of
                # that.
                #
                # So carry the TEXT, not a hash. The dispatcher must re-derive
                # the basis through its OWN normalize-then-hash chain
                # (`otr_image_gen_dispatcher.normalize_prompt_for_render`),
                # because the hash the seed consumes is computed AFTER the
                # safety clause, the style front-anchor and the banana
                # transform. Hashing here would produce a different number and
                # move every face exactly once.
                #
                # `identity` is the LANE's declaration, transported per object:
                # an ABSENT field cannot be told apart from an explicit "none",
                # and the dispatcher needs that distinction to know whether a
                # missing basis is a defect or the lane's stated intent.
                _pinfo = out.get(_cid) or {}
                _obj["identity"] = _identity_token_for_role(
                    identity_roles, _role_for_cid.get(_cid, ""))
                if _obj["identity"] != "none" and _pinfo.get("prompt"):
                    _obj["identity_prompt"] = _pinfo["prompt"]
                    _obj["identity_prompt_source"] = _pinfo.get("source", "")
            objects.append(_obj)
    required_scene_targets = []
    _required_fodder_roles = set(mesh_fodder_roles or ())
    for target in scene_targets:
        role = str(target.get("role") or "")
        if not _still_required(role):
            continue
        beat_id = str(target.get("beat_id") or "")
        if not beat_id:
            continue
        if role in _required_fodder_roles:
            required_scene_targets.extend((
                {"object_id": f"meshfodder_{beat_id}",
                 "kind": "mesh_fodder", "role": role, "beat_id": beat_id},
                {"object_id": f"plate_{beat_id}",
                 "kind": "scene_background_plate", "role": role,
                 "beat_id": beat_id},
            ))
        else:
            required_scene_targets.append({
                "object_id": f"still_{beat_id}",
                "kind": str(target.get("kind") or "scene_beat"),
                "role": role,
                "beat_id": beat_id,
            })
    payload = {"version": 1, "objects": objects}
    if required_scene_targets:
        # This is an additive, producer-owned receipt. The object list remains
        # the version-1 wire payload; this manifest states which scene targets
        # must be materialized before a video consumer may run.
        payload["required_scene_targets"] = required_scene_targets
    return payload, warnings


def _resolve_writer_llm(meta, warnings):
    """Lazily resolve the writer's slot LLM as a callable(prompt)->str (temp=0).
    Returns None if unavailable -> the deterministic template carries every
    character. Mirrors OTR_ShotLock._resolve_writer_llm; never raises."""
    try:
        from . import otr_shot_lock as _sl
        return _sl._resolve_writer_llm(meta, warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"writer LLM unavailable ({exc}); using template prompts")
        return None


class OTRMetaBriefImagePromptGen:
    """Registered as ``OTR_MetaBriefImagePromptGen``. Brief -> per-character image prompts."""

    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts_json", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "Frozen ledger JSON (cast + meta brief). Image prompts derive from here.",
                }),
            },
            "optional": {
                "image_policy_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "OTR_ImageDirector policy: granularity/seed + per-role still 'aspects' (so character stills match the selected video engine: portrait 832x1216 vs 16:9 832x480).",
                }),
                # `consistency_gate_warn_only` was REMOVED 2026-08-28: the
                # node displayed a Boolean, forwarded it one hop, and the
                # helper deleted it -- "the compatibility gate argument is
                # ignored", in its own words. The helper keeps its parameter;
                # the NODE stops advertising a dead choice.
                "gate_in": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, script_json, image_policy_json="{}",
                 gate_in=""):
        # CANONICAL REPLAY (campaign item 0): the imported rows already carry
        # their prompts; no writer LLM resolves here.
        try:
            from .production_ledger import replay_descriptor as _replay_descriptor
            _rmeta = (json.loads(script_json or "{}") or {}).get("meta") or {}
        except (ValueError, TypeError, ImportError, AttributeError):
            # AttributeError: a non-dict wire (the legacy parser LIST) is not a replay
            _rmeta = {}
        if _rmeta and _replay_descriptor(_rmeta):
            log.warning("[OTR_MetaBriefImagePromptGen] REPLAY: pass-through, no LLM")
            return (json.dumps({"replay": True, "objects": []}), "replay: pass-through")
        try:
            led = json.loads(script_json or "{}")
            if not isinstance(led, dict):
                led = {}
        except (ValueError, TypeError):
            led = {}
        meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
        cast = led.get("cast") if isinstance(led.get("cast"), list) else []
        lines = led.get("lines") if isinstance(led.get("lines"), list) else []
        video_models = _video_models_from_policy(image_policy_json)
        _still_capabilities = _still_consumer_capabilities(video_models)
        if (_still_capabilities is not None
                and all(value is False for value in _still_capabilities.values())):
            payload = {"version": 1, "objects": []}
            return (
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                "image_prompts v1: 0 objects\n"
                "SKIP: no effective video role consumes init_image",
            )

        warnings: list = []
        # Brief disposition, ONCE per run (gap-audit G4 restore).
        try:
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            log_story_brief_disposition(meta, "flux_portrait", log)
        except Exception:  # noqa: BLE001
            pass
        llm_fn = _resolve_writer_llm(meta, warnings)
        payload, warn2 = derive_image_prompts(
            cast, meta, llm_fn=llm_fn,
            consistency_gate_warn_only=False,   # widget removed 2026-08-28
            lines=lines,
            still_aspects=_still_aspects_from_policy(image_policy_json),
            mesh_fodder_roles=_mesh_fodder_roles_from_policy(image_policy_json),
            talking_roles=_talking_roles_from_policy(image_policy_json),
            still_word_roles=_still_word_roles_from_policy(image_policy_json),
            video_models=video_models,
            portrait_free_roles=_portrait_free_roles_from_policy(
                image_policy_json),
            identity_roles=_identity_roles_from_policy(image_policy_json),
        )  # aspects + mesh-fodder + talking + still_word + identity roles + video_models ride in image_policy_json
        warnings.extend(warn2)

        objs = payload.get("objects") or []
        report = [f"image_prompts v{payload.get('version')}: "
                  f"{len(objs)} objects"]
        for obj in objs:
            ident = obj.get("char_id") or obj.get("beat_id") or ""
            report.append(
                f"  {obj['object_id']}: kind={obj['kind']} role={obj['role']}"
                f" {obj['w']}x{obj['h']} id={ident}"
                f" source={obj['source']} hash={obj['prompt_hash'][:8]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_MetaBriefImagePromptGen] %s", w)

        return (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            "\n".join(report),
        )
