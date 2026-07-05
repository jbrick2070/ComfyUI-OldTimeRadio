"""OTR_MetaBriefImagePromptGen -- LLM image prompts from the Meta brief (C1).

The image-side mirror of ``OTR_ShotLock``'s per-beat creative derivation and of
``nodes/_otr_music_prompt.py``: every portrait/scene prompt is composed from the
propagating Meta brief (setting / period / mood) + the character's appearance,
optionally refined by ONE LLM call on the writer's slot (V-11, no new model_id
widget) at ``temperature=0`` with the ``prompt_hash`` taken AFTER the call.

Collapse guard (PASS-IMG SHOULD-FIX + BUG-099): empty / unparseable LLM output
-> reseed up to ``max_reseed`` -> a DETERMINISTIC brief-composed template that is
NEVER empty (a generic portrait must never ship a generic mesh silently, so we
WARN, but we never abort the episode or emit an empty prompt). Appearance is
looked up by ``char_id`` (BUG-098), never the display name.

Story-consistency gate (v1 = a SCHEMA assertion, not a 2nd LLM call): the final
prompt MUST carry the character's appearance token + the brief setting; a
hallucinated / missing trait -> WARN + fall back to the template
(``consistency_gate_warn_only`` toggles fail-closed vs warn on the hard case).

PURE core (``compose_image_prompt_fallback`` / ``derive_image_prompts``): no I/O,
no GPU, no engine imports -- the LLM is injected (tests) or resolved lazily from
the writer slot. Cold-import clean. UTF-8 no BOM, ASCII-only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

log = logging.getLogger("OTR")

#: Stopwords ignored when checking brief-grounding (so "a"/"the" never count).
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for",
    "from", "into", "setting", "portrait", "style", "studio",
})


def _significant_words(s: str) -> set:
    """Significant (len>=4, non-stopword) lowercase words in ``s``."""
    return {w for w in re.findall(r"[a-z]{4,}", (s or "").lower())
            if w not in _STOPWORDS}


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


#: Shared portrait style anchor. Reworded 2026-06-10 (operator look-QA): the
#: old "studio portrait, neutral lighting" framing read as an ACTOR in a
#: recording booth; portraits must show the CHARACTER in character, in the
#: story's world -- never a voice actor at a microphone.
# Round 5 operator notes (2026-06-10): the wider framing is a KEEPER ("this
# week's portraits show more body -- better"), so it is now intentional
# (three-quarter, not head-and-shoulders). The old "no microphone, not a
# recording studio" NEGATIONS are gone -- negative phrasing PLANTS the tokens
# in the image embedding (the c01 giant-mic catch); gear words are instead
# scrubbed from the OUTPUT (see _GEAR_WORDS) and banned in the instruction.
STYLE_ANCHOR = ("in-character cinematic three-quarter portrait, full head and face "
                "clearly visible with natural headroom above the head (never crop "
                "the top of the head), period-accurate costume and environment, "
                "dramatic film lighting")

#: WIDE (16:9) framing anchor. A three-quarter body shot cannot fit a short
#: landscape still without cropping the head, so wide character stills use a
#: head-and-shoulders MEDIUM shot, subject centred with headroom (operator framing
#: catch 2026-06-17: the wide character beats were decapitating the subject).
#: Portrait stills keep the three-quarter look (operator KEEPER 2026-06-10).
STYLE_ANCHOR_WIDE = ("in-character cinematic medium shot, head and shoulders, face "
                     "clearly visible, subject centred with natural headroom above "
                     "the head (never crop the top of the head), period-accurate "
                     "costume and environment, dramatic film lighting")


#: TALKING-lane portrait anchor (S4b, 2026-07-02). The ia2v lip-sync recipe
#: conditions character beats on the PORTRAIT (S4), and proof8 showed the
#: brief-styled cinematic portrait (dark palette, profile/wide composition,
#: tiny face) cannot drive lips -- the isolation A/B needed a bright,
#: face-forward close-up (scene-init 0.57 vs face-forward 2.86). Mirrors the
#: ltx_radio_mouth split that fixed the radio bookends: frontal, mouth
#: visible, warm light; the era/grade/palette tails are SKIPPED for these
#: portraits (see derive_image_prompts).
STYLE_ANCHOR_TALKING = (
    "in-character face-forward frontal close-up bust, looking directly at the "
    "camera, the whole face and mouth clearly visible and unobstructed, the "
    "face brightly lit and filling much of the frame, subject centred with "
    "natural headroom above the head (never crop the top of the head), "
    "period-accurate costume, warm dramatic lighting")


def _style_anchor_for_aspect(aspect, talking=False) -> str:
    """Framing anchor for a still's aspect: head-and-shoulders for WIDE (16:9) so
    the head is not cropped by the short frame, three-quarter for PORTRAIT.
    ``talking`` overrides both with the face-forward lip-sync anchor (S4b)."""
    if talking:
        return STYLE_ANCHOR_TALKING
    return STYLE_ANCHOR_WIDE if str(aspect).lower() == "wide" else STYLE_ANCHOR

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
#: only (zero anatomy tokens; the still dispatcher has NO negative channel) and
#: an OBJECT anchor, and it never appends _radio_face_overtness (which returns
#: face-bearing text). Exact anchor strings pinned (r3) so devs don't drift.
_RADIO_OBJECT_SUBJECT = ("presented as a stylized tabletop radio set on its "
                         "plate, its bakelite cabinet, woven speaker grille and "
                         "a glowing tuning dial")
_RADIO_OBJECT_ANCHOR = ("isolated tabletop still, the whole radio centered and "
                        "fully visible, dramatic film lighting")
_RADIO_OBJECT_ANCHOR_WIDE = ("isolated tabletop still, the whole radio centered "
                             "and fully visible, wide shot, dramatic film "
                             "lighting")
#: LTX-ONLY mouth-forward radio face (talking-radio kibitz r1, 2026-07-01).
#: style="ltx_radio_mouth" is used ONLY by the OTR_LTX_RADIO_FACE still mint
#: (the init stills the EXISTING ltx_audio_in bookend engine receives -- no new
#: video model / path). NEVER used by the HuMo hosts: the console_face look
#: above stays byte-unchanged (a mouth-tuned change
#: could hurt HuMo face-readability -- the Codex MUST-FIX #4 split). LTX-2.3
#: has no face/landmark detector; it drives whatever READS as a mouth, so the
#: subject puts a PROMINENT rubbery grille-mouth right after the form noun
#: (FLUX weights earlier tokens). Sub-plan C probes whether this actually
#: lip-syncs; until then ltx_audio_in stays documented as AMBIENT motion.
#: MATERIAL-ANCHORED (live catch 2026-07-02, probe B): the image dispatcher
#: has NO negative channel (stills mint from the POSITIVE prompt only), so
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
#: RADIO_CONSOLE_NEG (humans OUT). The still dispatcher has no negative channel,
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


#: ADDENDUM A/B (OTR_LTX_RADIO_FACE, SEPARATE from the HuMo-hosts feature): the
#: bookend role that gets a WIDE radio-FACE still for ltx_audio_in I2V init when
#: the A/B toggle is ON. ANNOUNCER-ONLY (operator 2026-07-03): the MUSIC bookend
#: stays faceless (ltx would lip-sync a mouth to music), so render_driver only
#: ever consumes the announcer face (render_driver.py:1145-1155) -- minting a
#: music radio-face still was a dead FLUX render every talking episode, pruned
#: 2026-07-04 (radio-face logic C2). Object_id pattern MUST match render_driver's
#: _ltx_radio_face_object_id (still_<role>_radio_face_169).
_LTX_RADIO_FACE_ROLES = ("announcer_visual",)


def _ltx_radio_face_object_id(role: str) -> str:
    """object_id of the WIDE radio-face still for ``role`` (ltx_audio_in A/B).
    Matches render_driver._ltx_radio_face_object_id."""
    return "still_%s_radio_face_169" % str(role or "")


def _ltx_radio_face_enabled() -> bool:
    """True iff OTR_LTX_RADIO_FACE is opted ON (default OFF). The A/B applies only
    when the routed bookend engine is ltx_audio_in (enforced in render_driver)."""
    import os
    return os.environ.get("OTR_LTX_RADIO_FACE", "0") == "1"


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
                            style: str = "console_face") -> str:
    """FULL prompt for a radio-host still, dispatched by ``style``.

    ``style="console_face"``: an ANTHROPOMORPHIC RADIO CONSOLE (the brief-driven
    form) whose glowing dial forms an expressive face -- "the radio IS the host",
    no human. The animatable dial-face; used for the HuMo radio-host FACE object
    AND the HuMo-driven synthetic-announcer portrait.
    ``style="ltx_radio_mouth"`` (talking-radio kibitz r1): the LTX-ONLY
    mouth-forward radio face for the OTR_LTX_RADIO_FACE still mint -- leads with
    the PROMINENT rubbery grille-mouth; never used by HuMo.
    ``style="radio_object"`` (radio-face logic 2026-07-04): a FACELESS stylized
    radio on its plate -- no dial-face, no person -- for a static / force-mapped
    announcer bookend (nothing animates a mouth, so no face). Facelessness is
    carried by the POSITIVE prompt (the still dispatcher has NO negative
    channel): object/material language only, an OBJECT anchor (NOT the
    person anchor), and NO overtness clause.
    All styles are brief-driven (:func:`radio_form_from_meta`); the face styles'
    overtness is brief-driven (:func:`_radio_face_overtness`), framed for the
    slot's ``aspect``, era texture via :func:`finish_visual_prompt`. Deterministic;
    never empty. FAIL-LOUD on an unknown style (retired ``radio_head_person``
    raises). The matching negative (:func:`radio_host_negative`) is populated on
    the object row by the caller."""
    form = radio_form_from_meta(meta)
    if style == "radio_object":
        # FACELESS: object/material language only. NO _radio_face_overtness (it
        # returns face text) and NOT _style_anchor_for_aspect (it adds head/face/
        # costume/in-character person tokens). An OBJECT anchor instead, then the
        # SAME "still" era + grade finish as console_face below (a tense story
        # reads as a tense-lit radio, never a person).
        anchor = (_RADIO_OBJECT_ANCHOR_WIDE if str(aspect).lower() == "wide"
                  else _RADIO_OBJECT_ANCHOR)
        prompt = ", ".join(["%s, %s" % (form, _RADIO_OBJECT_SUBJECT), anchor])
    elif style == "console_face":
        subject = "%s, %s, %s" % (form, _RADIO_CONSOLE_FACE,
                                  _radio_face_overtness(meta))
        prompt = ", ".join([subject, _style_anchor_for_aspect(aspect)])
    elif style == "ltx_radio_mouth":
        subject = "%s, %s" % (_RADIO_CONSOLE_MOUTH % form,
                              _radio_face_overtness(meta))
        prompt = ", ".join([subject, _style_anchor_for_aspect(aspect)])
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
            "radio-face logic.)" % (style, _RADIO_HOST_STYLES))
    # Stage 3 de-swallow (kibitz r2 M1): the ImportError shim is the ONLY
    # permitted catch here -- a visual-style resolution error must FAIL the
    # episode LOUD, never silently ship an unstyled prompt (no-fallback law).
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            _resolve_style, finish_visual_prompt)
    # STORY FLAIR (operator 2026-07-01: "all prompts respect the meta brief"):
    # a radio console / radio object is an OBJECT, not a bare human face, so it
    # takes the FULL "still" era tail (story palette + atmosphere + lighting)
    # -- the palette-strip "portrait" profile (BUG-LOCAL-113, for human faces)
    # is NOT wanted here; a tense story should read as a tense-lit radio.
    _style = _resolve_style(meta)
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


def _effective_announcer_family(video_models) -> str:
    """The RESOLVED ``announcer_visual`` engine FAMILY, AFTER OTR_FORCE_ENGINE_MAP.

    The radio-face rule keys the announcer portrait on this, NOT the bare
    OTR_ENABLE_HUMO_HOSTS flag: HUMO on but the announcer FORCED to a static
    engine (OTR_FORCE_ENGINE_MAP=announcer_visual=still_flat) still means a
    faceless radio. Resolves the role's engine via the shared role_slots join,
    applies the SAME force-map the dispatcher/render-time override uses
    (:func:`otr_image_gen_dispatcher._effective_engine_after_force_map`), then
    reads the family via :func:`render_driver.engine_family`. Tolerant: any
    resolution/import failure -> "" (the caller then defaults to the FACELESS
    radio_object -- fail SAFE toward faceless). Pure except the env read inside
    the force-map resolver."""
    try:
        try:
            from ._otr_shared import role_slots as _rs  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import role_slots as _rs  # type: ignore
        eng_id = _rs.engine_id_for_role(video_models or {}, "announcer_visual")
    except Exception:  # noqa: BLE001 -- never block prompts on a resolver import
        return ""
    if not eng_id:
        return ""
    try:
        try:
            from .otr_image_gen_dispatcher import (  # type: ignore
                _effective_engine_after_force_map)
        except ImportError:  # pragma: no cover -- flat test imports
            from otr_image_gen_dispatcher import (  # type: ignore
                _effective_engine_after_force_map)
        eng_id = _effective_engine_after_force_map("announcer_visual", eng_id)
    except Exception:  # noqa: BLE001 -- unforced run: keep the director pick
        pass
    try:
        try:
            from ._otr_video_engines.render_driver import engine_family  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_video_engines.render_driver import engine_family  # type: ignore
        return str(engine_family(eng_id, "") or "")
    except Exception:  # noqa: BLE001
        return ""


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
#: there is no per-object negative channel today). The negative scaffold is
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
BACKGROUND_PLATE_POS_SCAFFOLD = (
    "empty establishing environment, no people, no subject, no characters, "
    "wide 16:9 cinematic scene, atmospheric depth, period-accurate set"
)
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


def _still_word_roles_from_policy(policy_json):
    """The set of image-prompt roles whose paired VIDEO engine is ``still_word``.

    still_word is model-agnostic: the per-role video engine is ALREADY forwarded
    in ``image_policy_json["video_models"]`` (OTR_ImageDirector, 2026-07-02), so
    this reads it directly and resolves each role via the shared
    ``role_slots.engine_id_for_role`` join (the SAME rule the director uses) --
    no separate director-side precompute needed. A missing/malformed policy, or
    a role_slots import failure in an odd flat-test context, yields an empty set
    -> NO word/title branch (the legacy cinematic scene still). Pure, tolerant.

    OTR_FORCE_ENGINE_MAP seam (r3): the render-time override rewrites the engine
    BEFORE render, and the still dispatcher honors it for still-consumption; so a
    run that FORCES a role to still_word must ALSO mint the word/title prompt (else
    it renders legacy SCENE prompts on a word card). Resolve the EFFECTIVE engine
    via ``_effective_engine_after_force_map`` before the compare. Env-unset (no
    force map) -> the resolver returns the id unchanged -> byte-identical."""
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
            eng_id = str(_rs.engine_id_for_role(vm, role))
            # Honor OTR_FORCE_ENGINE_MAP (its OWN try so an import failure falls
            # back to the unforced id -- never silently eaten by the outer except).
            try:
                try:
                    from .otr_image_gen_dispatcher import (  # type: ignore
                        _effective_engine_after_force_map)
                except ImportError:  # pragma: no cover -- flat test imports
                    from otr_image_gen_dispatcher import (  # type: ignore
                        _effective_engine_after_force_map)
                eng_id = str(_effective_engine_after_force_map(role, eng_id))
            except Exception:  # noqa: BLE001 -- unforced run stays byte-identical
                pass
            if eng_id == "still_word":
                roles.add(role)
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
# scrubbed era tail -> grade -> POSITIVE text guard LAST. The per-episode
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
#: era-tail text summoners scrubbed (word mode ONLY, MANDATORY -- not a QA-watch):
#: a "flickering neon signs" atmosphere leaks extra glowing text onto the card.
_STILL_WORD_TEXT_SUMMONERS = (
    "marquee", "poster", "magazine", "newspaper", "neon", "sign", "signs",
    "signage", "label", "labels", "theatrical", "billboard", "banner")


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
    outer ``"%s"`` boundary. Closes the nested-quote ambiguity AND the semantic
    prompt-injection where a stray ``", ...`` closes the span early and injects a
    sibling clause. Keeps apostrophes / em-dash / ellipsis / unicode. Pure."""
    return re.sub(r'[\"“”„‟«»]', "'", str(s or ""))


def _still_word_fit_card(words: str) -> str:
    """DETERMINISTIC card-length reduction (fail-SOFT): a title card cannot hold a
    paragraph, so an overflowing line is trimmed to the first clause, else the
    first ``STILL_WORD_MAX_WORDS`` words, then a ``STILL_WORD_MAX_CHARS`` word-
    boundary cap. Under both caps -> returned unchanged. Applied BEFORE the hash
    (pure). NEVER an abort (that would kill every episode -- the §4D class)."""
    s = str(words or "")
    if len(s.split()) <= STILL_WORD_MAX_WORDS and len(s) <= STILL_WORD_MAX_CHARS:
        return s
    first = re.split(r"[.!?;:]", s, maxsplit=1)[0].strip()
    if (first and len(first.split()) <= STILL_WORD_MAX_WORDS
            and len(first) <= STILL_WORD_MAX_CHARS):
        s = first
    if len(s.split()) > STILL_WORD_MAX_WORDS:
        s = " ".join(s.split()[:STILL_WORD_MAX_WORDS])
    if len(s) > STILL_WORD_MAX_CHARS:
        cut = s[:STILL_WORD_MAX_CHARS]
        idx = cut.rfind(" ")
        s = cut[:idx] if idx > 0 else cut
    return s.strip()


def _scrub_still_word_era_tail(era) -> str:
    """Strip text-SUMMONING comma-segments (sign / marquee / newspaper / neon /
    poster / label ...) from the still-word era tail ONLY (word mode). A whole
    comma-segment mentioning a summoner is dropped -- MANDATORY, not a QA-watch
    (a "flickering neon signs" atmosphere would summon extra glowing card text).
    Pure; word-boundary match."""
    kept = []
    for seg in str(era or "").split(","):
        low = seg.lower()
        if seg.strip() and not any(
                re.search(r"\b%s\b" % re.escape(t), low)
                for t in _STILL_WORD_TEXT_SUMMONERS):
            kept.append(seg.strip())
    return ", ".join(kept)


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


def compose_still_word_prompt(meta, role, beat_line):
    """The still_word base-still prompt (see the module note above). Pure,
    deterministic, MODEL-AGNOSTIC (no engine reference -- the composed prompt is
    identical regardless of which image engine mints it). Returns a flat ``str``.

    ``beat_line`` accepts a DICT (the frozen ledger line -- its ``text`` is the
    words, its free-form ``traits`` drives the beat mood on CHARACTER cards) OR a
    raw STRING (back-compat; no mood). The third param NAME is pinned by a test.

    WORD mode (character_video / announcer_visual): quoted words -> legibility
    guard -> per-episode LOCKED lettering -> per-episode COOL backdrop (+ one
    ``"<mood> mood"`` on character cards) -> SCRUBBED era tail -> grade ->
    POSITIVE text guard LAST. FAILS LOUD (``ValueError``, NO FALLBACK) on a BLANK
    line; an overflowing line is REDUCED (never aborted). MUSIC mode is the
    UNTOUCHED wordless abstract-title still. All raises precede any prompt hash."""
    # Stage 3 (kibitz r2 M2): era + grade route through the visual-style pack
    # (ImportError shim only; a style error RAISES). The typography/backdrop
    # maps stay Python (STAGE3_SUBPLAN section 8; operator lettering-
    # consistency directive 2026-07-04).
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            get_era_tail, NO_TEXT_CLAUSE, _resolve_style)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            get_era_tail, NO_TEXT_CLAUSE, _resolve_style)
    _role = str(role or "")
    _style = _resolve_style(meta)
    era = (get_era_tail(meta, profile="still", style=_style) or "").strip()
    if _role == _STILL_WORD_MUSIC_ROLE:
        # UNTOUCHED music path (operator 2026-07-04: episode titles are proc-gen
        # elsewhere, so the music card stays a WORDLESS abstract mood still).
        title = _episode_title(meta)
        if not title:
            raise ValueError(
                "compose_still_word_prompt: music-role still_word beat has a "
                "BLANK episode title (meta['episode_title']/['title']) -- cannot "
                "mint the abstract title still. NO FALLBACK.")
        pieces = ['an abstract picture evoking "%s"' % _fold_inner_dquotes(title),
                  _STILL_WORD_TITLE_MOOD_STYLE, era, _style.image_grade_tail]
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
    lettering = _STILL_WORD_TYPOGRAPHY[genre]
    backdrop = _STILL_WORD_BACKDROP[genre]
    # ONE beat mood adjective on CHARACTER cards only (announcer cards: no mood).
    mood = (_still_word_mood_from_line(beat_line)
            if _role == "character_video" else "")
    if mood:
        backdrop = "%s, %s mood" % (backdrop, mood)
    era = _scrub_still_word_era_tail(era)               # MANDATORY word-mode scrub
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


def derive_scene_still_targets(lines, fps: int = 25):
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
                                  talking: bool = False) -> str:
    """Deterministic brief-composed portrait prompt -- NEVER empty.

    ``"{appearance}, {setting} setting, {style anchor}"`` with empty parts
    dropped; degrades to the style anchor alone if the brief + cast are bare.
    ``talking`` swaps in the face-forward lip-sync anchor (S4b).
    """
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
    parts.append(_style_anchor_for_aspect(aspect, talking=talking))
    return ", ".join(parts)


def _build_char_prompt_request(char: dict, meta: dict, setting: str,
                               aspect: str = "portrait",
                               talking: bool = False) -> str:
    """The instruction handed to the writer LLM (temp=0) for one character.

    ``aspect`` ('wide'|'portrait') drives the framing clause so a 16:9 still asks
    for a head-and-shoulders shot (the head fits the short frame) while a portrait
    still keeps the three-quarter look -- both with explicit headroom so the top of
    the head is never cropped (operator framing catch 2026-06-17). ``talking``
    (S4b) overrides both with the face-forward lip-sync framing: the ia2v
    recipe drives the mouth on THIS still, so a profile / distant / dim face
    is unusable (proof8 catch 2026-07-02)."""
    appearance = _appearance_for_char([char], str(char.get("char_id") or ""))
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
        "photographic and period-consistent.\n"
        f"character_appearance: {appearance or '(unspecified)'}\n"
        f"story_setting: {setting or '(unspecified)'}\n"
        f"style_anchor: {_style_anchor_for_aspect(aspect, talking=talking)}\n"
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.\n"
        "Do not mention radios, microphones, studios, or any broadcasting "
        "equipment anywhere in the prompt -- the character is a person in "
        "the STORY's world, not a performer at a station.\n"
        "Return only the prompt line."
    )


def _build_char_scene_request(char: dict, meta: dict, setting: str,
                              line: dict) -> str:
    """BUG 1 follow-up (2026-06-20 operator): the per-beat character still must be
    SHOT/BEAT AWARE -- the character IN the moment of THIS beat -- regardless of
    image model (the video lane conditions on the SAME still). Mirrors
    :func:`_build_char_prompt_request` but WIDE 16:9 and grounded in the beat's
    own ``beat_intent`` / ``traits`` / spoken ``text`` so each character beat
    yields a DISTINCT still. Temp=0 like the portrait path -> deterministic."""
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
        f"style_anchor: {_style_anchor_for_aspect('wide')}\n"
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.\n"
        "Do not mention radios, microphones, studios, or any broadcasting "
        "equipment -- the character is a person in the STORY's world.\n"
        "Return only the prompt line."
    )


def _compose_char_scene_prompt(meta, char_entry, setting, line, llm_fn,
                               warnings, cid, max_reseed=2):
    """Beat-aware 16:9 character still prompt -> ``(prompt, source)``. LLM-refined
    (the beat's action/emotion) when a writer LLM is available -- so each beat
    differs -- else the deterministic per-character ``scene_character`` composer.
    Always depicts the CHARACTER, never a radio booth. Guards + finishes exactly
    like the portrait path (person guard, gear scrub, era+grade tail, no-text
    clause)."""
    ce = char_entry if isinstance(char_entry, dict) else {}
    prompt = ""
    source = "char_scene_template"
    said = str((line or {}).get("text") or "").strip()
    # A writer LLM present AND a spoken line = the LLM lane was ATTEMPTED; if it
    # then fails/returns junk we FAIL LOUD rather than swap to the deterministic
    # composer (no-fallback rip 2026-07-03). llm_fn=None (no writer LLM) or a
    # dialogue-less beat is the legit template lane, not a failure.
    _llm_attempted = llm_fn is not None and bool(said)
    if _llm_attempted:
        req = _build_char_scene_request(ce, meta, setting, line)
        for attempt in range(max_reseed + 1):
            try:
                raw = llm_fn(req)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"char-scene llm_fn raised for {cid} ({exc}); "
                                f"reseed {attempt}")
                raw = ""
            cand = _clean_llm_prompt(raw)
            if cand:
                prompt = cand
                source = "char_scene_llm"
                break
    # Person/face analysis REMOVED (operator directive 2026-07-04): no person
    # detector discards or annotates a beat still. The gear scrub below still runs;
    # an EMPTY LLM return still fails loud at the gate further down.
    if prompt:
        scrubbed = _scrub_gear_words(prompt)
        if scrubbed != prompt:
            warnings.append(f"char-scene prompt for {cid}: broadcast-gear scrubbed")
            prompt = scrubbed or ""
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            compose_still_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            compose_still_prompt)
    if not prompt:
        if _llm_attempted:
            raise RuntimeError(
                "[OTR_ImagePrompt] char-scene: the writer LLM produced no usable "
                "PERSON prompt for %s (empty / non-person / all-gear after %d "
                "reseeds) -- NO deterministic template fallback (no-fallback rip "
                "2026-07-03); a failed char-scene LLM stops the episode."
                % (cid, max_reseed)
            )
        # No writer LLM (or a dialogue-less beat): the deterministic per-character
        # scene composer is the PRIMARY local-lane output, not a fallback.
        return (compose_still_prompt(meta, kind="scene_character",
                                     role="character_video", char_entry=ce),
                source)
    # FINISH like a scene still: era tail + cinematic grade + the no-text clause.
    # Stage 3 de-swallow (kibitz r2 M1): ImportError shim only; a style error
    # RAISES (no silent unstyled prompt).
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    _style = _resolve_style(meta)
    prompt = finish_visual_prompt(meta, prompt, era_profile="portrait",
                                  style=_style)
    if _style.image_grade_tail and _style.image_grade_tail not in prompt:
        prompt = f"{prompt}, {_style.image_grade_tail}"
    if not prompt.endswith(NO_TEXT_CLAUSE):
        prompt = f"{prompt}, {NO_TEXT_CLAUSE}"
    return prompt, source


def _mesh_fodder_subject(meta, char_entry, line, setting, role) -> str:
    """The SUBJECT phrase for a mesh_fodder still -> always a single, isolated
    thing the mesher can carve cleanly. A character beat meshes the CHARACTER
    (appearance); the announcer meshes the announcer figure (no studio gear, no
    occlusion); a music beat meshes the RADIO ITSELF (2026-06-30 mesh-improve
    item 1: "music bookend mesh = a 3D radio, not a character body" -- never
    the old generic "object representing the story", which minted an
    arbitrary, unrelated prop); any OTHER no-character beat meshes ONE
    emblematic story object. Pure; never empty (a bare fallback keeps the
    mesher fed)."""
    appearance = ""
    if char_entry:
        appearance = _appearance_for_char(
            [char_entry], str((char_entry or {}).get("char_id") or ""))
    if appearance:
        return appearance
    if str(role) == "announcer_visual":
        # C2 (2026-07-01): the announcer mesh is the FACELESS radio OBJECT (brief
        # -driven form), NOT a person-with-a-face -- "only HuMo gets a face". The
        # old "1940s radio announcer in a suit" put a human figure into the mesh,
        # violating that invariant AND hardcoding the era. Faceless + isolated.
        return radio_form_from_meta(meta)
    if str(role) == "music_visual":
        # The music open/inter/close mesh IS the radio -- the recurring on-air
        # object, isolated and clean for the mesher. Brief-driven form (C2), no
        # hardcoded 1940s set.
        return radio_form_from_meta(meta)
    # No-character beat outside announcer/music: a single emblematic OBJECT
    # from the story world (chunk 6 refines the object_id policy; this keeps
    # the mesher fed cleanly).
    intent = str((line or {}).get("beat_intent") or "").strip()[:120]
    base = intent or setting or "the story"
    return "a single emblematic object representing %s" % base


def _compose_mesh_fodder_prompt(meta, char_entry, line, setting, role) -> str:
    """``"{subject}, {MESH_FODDER_POS_SCAFFOLD}"`` -- the isolated-subject still
    Hunyuan3D meshes. Deterministic (no LLM): the subject identity comes from
    the cast appearance / announcer figure / story object, and the checked-in
    scaffold enforces the neutral-plate isolation. Never empty."""
    subject = _mesh_fodder_subject(meta, char_entry, line, setting, role)
    out = "%s, %s" % (subject, MESH_FODDER_POS_SCAFFOLD)
    # C4 (2026-07-01): the mesh radio inherits the brief's era TEXTURE (trimmed
    # still profile: atmosphere line + palette/lighting top-2), so a non-1940s
    # brief carries through to the meshed object. The isolation scaffold above
    # still governs silhouette/plate; the tail only tints the look.
    # Stage 3 de-swallow (kibitz r3 -- the seam ALL FOUR reviewers caught):
    # ImportError shim only; a style error RAISES.
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            _resolve_style, get_era_tail)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            _resolve_style, get_era_tail)
    tail = get_era_tail(meta, profile="still", style=_resolve_style(meta))
    if tail and tail not in out:
        out = "%s, %s" % (out, tail)
    return out


def _compose_background_plate_prompt(meta, setting) -> str:
    """The subject-free 16:9 world plate the mesh stands in front of. Leads with
    the story setting, then the checked-in 'no subject' scaffold. Deterministic;
    never empty."""
    parts = []
    if setting:
        parts.append("%s setting" % setting)
    parts.append(BACKGROUND_PLATE_POS_SCAFFOLD)
    plate = ", ".join(parts)
    # Era tail / grade so the plate matches the episode look. Stage 3
    # de-swallow (kibitz r2 M1): ImportError shim only; a style error RAISES.
    try:
        from ._otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import (  # type: ignore
            NO_TEXT_CLAUSE, _resolve_style, finish_visual_prompt)
    _style = _resolve_style(meta)
    plate = finish_visual_prompt(meta, plate, era_profile="still",
                                 style=_style)
    if _style.image_grade_tail and _style.image_grade_tail not in plate:
        plate = "%s, %s" % (plate, _style.image_grade_tail)
    if not plate.endswith(NO_TEXT_CLAUSE):
        plate = "%s, %s" % (plate, NO_TEXT_CLAUSE)
    return plate


#: Person-evidence vocabulary for the portrait guard: an accepted prompt that
#: matches NONE of these almost certainly depicts scenery/objects (the
#: "microphone, no person" live catch, look-QA round 4) -> template fallback.
# Person/face DETECTOR removed (operator directive 2026-07-04): the _PERSON_WORDS
# regex + _depicts_person() analyzer that grep-gated portraits and beat stills are
# gone. Face quality is QA'd visually by the operator reviewing the derived
# prompts, not analyzed here.


#: Broadcast-gear vocabulary (round 5 operator directive 2026-06-10): CHARACTER
#: portrait prompts must not mention radio/mic/studio gear -- the tokens drag
#: FLUX toward microphones and consoles (the c01 giant-mic catch). The
#: ANNOUNCER is exempt (his portrait is radio-styled BY DESIGN).
_GEAR_WORDS = re.compile(
    r"\s*\b(?:radios?|microphones?|mics?|broadcasts?|broadcasters?|"
    r"broadcasting|recording\s+studios?|radio\s+(?:station|studio|set|"
    r"booth)s?|studios?|on[- ]air(?:\s+sign)?)\b[,;]?",
    re.IGNORECASE)


def _scrub_gear_words(prompt: str) -> str:
    """Remove broadcast-gear tokens from a CHARACTER portrait prompt, tidying
    the leftover separators. Pure; '' stays ''."""
    out = _GEAR_WORDS.sub("", prompt or "")
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"(,\s*)+,", ", ", out)
    out = re.sub(r"\s+,", ",", out)
    return out.strip(" ,;").strip()


def _clean_llm_prompt(raw: str) -> str:
    """First non-empty line of the LLM output, trimmed; '' if unusable."""
    if not raw:
        return ""
    for line in str(raw).splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""


#: Radio-form vocabulary (2026-07-01): a brief-driven radio surface is inherently
#: brief-grounded even when its exact words do not overlap the appearance/setting
#: text (a "space-station communications console" for a docking brief). Widening
#: the gate to accept these prevents a valid NON-1940s radio form from failing
#: the overlap check and silently reverting to the (1940s) template.
_RADIO_VOCAB = frozenset({
    "radio", "console", "transceiver", "wireless", "receiver", "comms",
    "communications", "broadcast", "microphone",
})


def _passes_consistency(prompt: str, appearance: str, setting: str) -> bool:
    """v1 schema gate: the prompt must be GROUNDED in the brief -- it shares at
    least one significant word with the character appearance or the story
    setting. Cheap word-overlap check, not a 2nd LLM call. When neither
    appearance nor setting is known, nothing to assert -> passes.

    Widened 2026-07-01: a recognized radio-form token (:data:`_RADIO_VOCAB`) is
    itself brief-grounded, so a brief-driven radio surface passes even when its
    exact words do not overlap. Strictly more permissive -- never fails a prompt
    that passed before."""
    want = _significant_words(appearance) | _significant_words(setting)
    if not want:
        return True
    pw = _significant_words(prompt)
    if want & pw:
        return True
    return bool(pw & _RADIO_VOCAB)


def derive_image_prompts(cast: list, meta: dict, *, llm_fn=None, max_reseed: int = 2,
                         consistency_gate_warn_only: bool = False, lines=None,
                         fps: int = 25, still_aspects=None,
                         mesh_fodder_roles=None, talking_roles=None,
                         still_word_roles=None, video_models=None):
    """ONE versioned image-object payload: ``{"version": 1, "objects": [...]}``
    (still-spine ST-2 / pass-02 item 1: portraits MIGRATED to the object
    schema in the same patch; no dual-schema shims).

    Each object carries ``object_id`` / ``kind`` / ``role`` / ``w`` / ``h`` /
    ``prompt`` / ``prompt_hash`` / ``source`` plus ``char_id`` (portraits;
    object_id == char_id) or ``beat_id`` (scene stills). Guards branch by
    KIND before running: the person guard + gear scrub run ONLY on
    kind=portrait; scene stills get the no-text clause (inside
    ``compose_still_prompt``) and skip the person guard entirely.

    Portrait path: LLM (temp=0, injected or lazily resolved) refines each;
    empty/unparseable -> reseed. NO-FALLBACK (rip 2026-07-03): when a writer LLM
    was ATTEMPTED (``llm_fn`` present) and it produces no usable prompt, or the
    prompt fails the story-consistency gate / person guard / gear scrub, this
    FAILS LOUD (RuntimeError) rather than swapping in the deterministic template.
    ``llm_fn=None`` (no writer LLM configured) is the legit template lane, not a
    failure, and never raises. ``prompt_hash`` is taken AFTER the call.
    Returns ``(payload, warnings)``.

    ``lines`` (optional, the frozen ledger lines, READ-ONLY): announcer
    portrait minting (as before) PLUS the v1 scene-still targets
    (open/announcer/outro via :func:`derive_scene_still_targets`).
    """
    warnings: list = []
    setting = _read_setting(meta)
    out: dict = {}
    roster = list(cast or [])
    cast_ids = {str(c.get("char_id") or "") for c in roster if isinstance(c, dict)}
    for cid in announcer_line_char_ids(lines):
        if cid in cast_ids:
            continue                      # a real cast row already covers it
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
        appearance = _appearance_for_char([char], cid)
        # Framing aspect: synthetic announcers follow announcer_visual, cast
        # characters follow character_video -- so a WIDE video engine gets a
        # head-and-shoulders still the wide render won't decapitate (2026-06-17).
        _aspect = (still_aspects or {}).get(
            "announcer_visual" if char.get("_synthetic_announcer")
            else "character_video", "portrait")
        # TALKING lane (S4b): a cast portrait whose character_video engine
        # lip-syncs (the ia2v register) is the render INIT for the mouth --
        # mint it face-forward + warm, and skip the era/grade tails below
        # (mirrors the ltx_radio_mouth split that fixed the radio bookends).
        # Synthetic announcers keep their radio-host path untouched.
        _talking = bool((talking_roles or {}).get("character_video")) and (
            not char.get("_synthetic_announcer"))
        # C1 + E (2026-07-01 brief-driven radio-host): the synthetic announcer /
        # radio-host prompt is BRIEF-DRIVEN and stamped DIRECTLY -- SKIP the LLM
        # refine. The refine instruction (_build_char_prompt_request: "Do not
        # mention radios, microphones, studios") CONTRADICTS a radio-styled host,
        # so a refined prompt strips the radio tokens, fails _passes_consistency,
        # and reverts to the (old 1940s) template. build_radio_host_prompt already
        # finishes (era tail + grade), so the loop's LLM / consistency /
        # gear-scrub / finish steps are all bypassed for this row (also saves an
        # LLM call). The negative is populated on the object row.
        if char.get("_synthetic_announcer"):
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
            _aprompt = build_radio_host_prompt(meta, _aspect, style=_astyle)
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
                                             talking=_talking)
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
            if llm_fn is not None:
                raise RuntimeError(
                    "[OTR_ImagePrompt] portrait: the writer LLM produced no "
                    "usable prompt for %s after %d reseeds -- NO deterministic "
                    "template fallback (no-fallback rip 2026-07-03); a failed "
                    "portrait LLM stops the episode." % (cid, max_reseed)
                )
            # No writer LLM configured: the deterministic template IS the primary
            # local-lane portrait, not a fallback.
            prompt = compose_image_prompt_fallback(meta, char, _aspect,
                                                   talking=_talking)
            source = "template"
        # Story-consistency gate (schema assertion, v1). The synthetic
        # ANNOUNCER grounds on APPEARANCE ONLY (the radio anchor): an LLM
        # line that drops the radio styling for pure story-setting flavor
        # fails the gate and falls back to the radio template (operator
        # directive 2026-06-09: the announcer gets a RADIO-style image).
        gate_setting = "" if char.get("_synthetic_announcer") else setting
        if not _passes_consistency(prompt, appearance, gate_setting):
            msg = f"image prompt for {cid} missing appearance/setting trait"
            if consistency_gate_warn_only or source != "llm":
                # warn-only toggle, OR the deterministic template itself (no LLM
                # to blame) -- keep the prompt, log loud. Not a model swap.
                warnings.append(msg + " (kept; warn-only or template path)")
            else:
                raise RuntimeError(
                    "[OTR_ImagePrompt] portrait: the writer LLM visual prompt for "
                    "%s failed the story-consistency gate (%s) -- NO template "
                    "fallback (no-fallback rip 2026-07-03)." % (cid, msg)
                )
        # Person/face analysis REMOVED (operator directive 2026-07-04): no
        # automated person detector gates or annotates the portrait. The operator
        # crafts the prompts and QAs faces visually by reviewing them; a token-grep
        # face check earns nothing and previously failed builds / burned credits.
        # The prompt flows straight to the gear scrub + finisher below.
        # GEAR SCRUB (round 5 operator directive): character portraits never
        # mention radio/mic/studio gear -- the tokens pull FLUX toward
        # equipment (the c01 giant-mic catch). The ANNOUNCER keeps his radio
        # styling by design (radio-grounding gate): synthetic announcer rows
        # AND a cast row literally named ANNOUNCER are exempt.
        _is_announcer_row = bool(
            char.get("_synthetic_announcer")
            or str(char.get("name") or "").strip().upper() == "ANNOUNCER")
        if not _is_announcer_row:
            _scrubbed = _scrub_gear_words(prompt)
            if _scrubbed != prompt:
                warnings.append(
                    f"image prompt for {cid}: broadcast-gear tokens scrubbed")
                if not _scrubbed and source == "llm":
                    # An LLM prompt that is ENTIRELY broadcast-gear tokens is a
                    # model miss -- fail loud, no template swap (no-fallback rip).
                    raise RuntimeError(
                        "[OTR_ImagePrompt] portrait: the writer LLM visual prompt "
                        "for %s was ENTIRELY broadcast-gear tokens -- NO template "
                        "fallback (no-fallback rip 2026-07-03)." % cid
                    )
                prompt = _scrubbed or compose_image_prompt_fallback(
                    meta, char, talking=_talking)
        if char.get("_synthetic_announcer"):
            source = "announcer_" + source   # traceable in reports/ledger
        # FINISH the prompt (gap-audit F3, 2026-06-10): era tail + film style
        # tail, restored from the deleted legacy composer. Runs AFTER the
        # consistency + person guards (finishing never re-triggers them) and
        # BEFORE the hash so the stamped hash matches the rendered prompt.
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
                    _resolve_style, finish_visual_prompt)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    _resolve_style, finish_visual_prompt)
            # era_profile="portrait": never bleeds the episode's ambient
            # colour palette into character faces (sci-fi = blue wash,
            # period drama = red wash). Only the atmosphere mood line is
            # safe; full palette is explicitly excluded (BUG-LOCAL-113).
            _style = _resolve_style(meta)
            prompt = finish_visual_prompt(meta, prompt,
                                          era_profile="portrait",
                                          style=_style)
            # BUG-411 (operator 2026-06-14: "keep ALL flux consistent with the
            # 6/5 aesthetic"): append the cinematic GRADE tail to PORTRAITS too,
            # so a still_pan beat standing in for a HuMo portrait shows the same
            # graded look as the scene stills/bookend (still_pan
            # animates the minted PNG, so the PNG must carry the grade). The
            # radio broadcast-distress tail stays scene-still-only (a person is
            # not a radio set). Idempotent -- never duplicates. Stage 3: the
            # grade comes from the pack (empty string = no append).
            if (_style.image_grade_tail
                    and _style.image_grade_tail not in prompt):
                prompt = f"{prompt}, {_style.image_grade_tail}"
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
    for cid, pinfo in out.items():
        _role = pinfo.pop("_role", "character_video")
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

    # RADIO-HOST FACE object (2026-07-01, chunk 3) -- the ONE animatable human
    # face this feature grants. GATED on OTR_ENABLE_HUMO_HOSTS so OFF is
    # byte-identical (no extra object minted). Minted ONCE per episode with its
    # own object_id (kind=portrait -> render_driver._portrait_index resolves it
    # as the HuMo init_image for the LINELESS announcer/music bookends). Aspect
    # FOLLOWS the HuMo bookend slot (music_visual, else announcer_visual) so a
    # wide HuMo engine is not fed a pillarboxed portrait; seed is pinned to
    # OTR_RADIO_BOOKEND_SEED in the dispatcher (open/inter/close share ONE face).
    # The no-baby negative is populated on the row.
    if _humo_hosts_enabled():
        _rh_aspect = (_role_aspects.get("music_visual")
                      or _role_aspects.get("announcer_visual") or "portrait")
        # MUSIC bookends -> the ANTHROPOMORPHIC RADIO CONSOLE face (operator 2026-07-01).
        _rh_prompt = build_radio_host_prompt(meta, _rh_aspect, style="console_face")
        _rhw, _rhh = still_dims_for_aspect(_rh_aspect, PORTRAIT_W, PORTRAIT_H)
        objects.append({
            "object_id": RADIO_HOST_PORTRAIT_ID,
            "kind": "portrait",
            "role": "music_visual",
            "w": _rhw, "h": _rhh,
            "prompt": _rh_prompt,
            "prompt_hash": _content_hash(_rh_prompt),
            "negative_prompt": radio_host_negative("console_face"),
            "source": "radio_host_portrait",
        })

    # ADDENDUM A/B (OTR_LTX_RADIO_FACE) -- the WIDE radio-FACE stills for the
    # ltx_audio_in bookends (option b: ltx animates an AMBIENT face, not lip-sync).
    # SEPARATE, opt-in; GATED so default 0 is byte-identical (no extra objects).
    # WIDE by construction (aspect="wide") so the wide ltx_audio_in engine is never
    # fed a pillarboxed portrait -- the exact trap the kibitz flagged. Per-role
    # object_id matches render_driver._ltx_radio_face_object_id; seed-pinned in the
    # dispatcher (shares the bookend seed). No-baby negative populated.
    # S4c (operator eyeball 2026-07-02, "should be talking lips?"): the A/B is
    # RETIRED into DEFAULT-ON for the ia2v talking lane -- when a bookend
    # role's engine lip-syncs (talking_roles via the director policy), the
    # radio-face still ALWAYS mints; the env toggle remains only for the
    # single-pass (non-talking) recipes.
    _talk = talking_roles or {}
    if (_ltx_radio_face_enabled()
            or any(_talk.get(r) for r in _LTX_RADIO_FACE_ROLES)):
        _fw, _fh = still_dims_for_aspect("wide", PORTRAIT_W, PORTRAIT_H)
        for _abrole in _LTX_RADIO_FACE_ROLES:
            # Talking-radio kibitz r1 (2026-07-01): the ltx bookend still uses the
            # LTX-ONLY mouth-forward style. The radio IS the host, and ltx_audio_in
            # (no face detector) needs the biggest, clearest mouth region to drive
            # -- the Sub-plan-C probe lever. ANNOUNCER-ONLY since 2026-07-03 (music
            # stays faceless; the dead music mint was pruned 2026-07-04). The HuMo
            # console_face look is byte-unchanged (the split); no new model / path.
            _fprompt = build_radio_host_prompt(meta, "wide",
                                               style="ltx_radio_mouth")
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
    # driver's text prompts is locked in tests); no LLM call, no person
    # guard, no gear scrub -- guards branch by kind BEFORE running.
    scene_targets, scene_warns = ([], [])
    if lines:
        try:
            scene_targets, scene_warns = derive_scene_still_targets(
                lines, fps=fps)
        except ValueError:
            # NO FALLBACKS (rip-sfx-broll 2026-07-01): an unmapped
            # speaker_role (e.g. an old "sfx" ledger) is a hard error,
            # never downgraded to a missing-stills warning.
            raise
        except Exception as exc:  # noqa: BLE001 -- stills never kill prompts
            warnings.append(f"scene-still derivation failed ({exc}); "
                            "episode renders without scene stills (LOUD)")
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
                    meta, str(tgt.get("role") or ""), _ln)
                _wobj = {
                    "object_id": f"still_{_bid}",
                    "kind": tgt["kind"],
                    "role": tgt["role"],
                    "beat_id": _bid,
                    "w": sw, "h": sh,
                    "prompt": _wprompt,
                    "prompt_hash": _content_hash(_wprompt),
                    "source": "still_word",
                }
                # Provenance for operator QA: the LOCKED per-episode lettering +
                # backdrop family (music cards excluded -- they carry no lettering).
                if str(tgt.get("role") or "") != _STILL_WORD_MUSIC_ROLE:
                    _swg = _still_word_genre(meta)
                    _wobj["lettering_style"] = _STILL_WORD_TYPOGRAPHY[_swg]
                    _wobj["backdrop_family"] = _swg
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
                    meta, _ce, _ln, setting, tgt["role"])
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
                }
                if _cid:
                    _fobj["char_id"] = _cid
                objects.append(_fobj)
                _pprompt = _compose_background_plate_prompt(meta, setting)
                objects.append({
                    "object_id": "plate_%s" % _bid,
                    "kind": "scene_background_plate",
                    "role": tgt["role"],
                    "beat_id": _bid,
                    "w": sw, "h": sh,
                    "prompt": _pprompt,
                    "prompt_hash": _content_hash(_pprompt),
                    "source": "mesh_background_plate",
                })
                continue
            if tgt["kind"] == "scene_character":
                _ce = _cast_by_id.get(_cid)
                _ln = _line_by_beat.get(tgt["beat_id"], {})
                sprompt, _csrc = _compose_char_scene_prompt(
                    meta, _ce, setting, _ln, llm_fn, warnings, _cid)
                _src = _csrc
            else:
                sprompt = compose_still_prompt(
                    meta, kind=tgt["kind"], role=tgt["role"],
                    beat_id=tgt["beat_id"])
            _obj = {
                "object_id": f"still_{tgt['beat_id']}",
                "kind": tgt["kind"],
                "role": tgt["role"],
                "beat_id": tgt["beat_id"],
                "w": sw, "h": sh,
                "prompt": sprompt,
                "prompt_hash": _content_hash(sprompt),
                "source": _src,
            }
            if _cid:
                _obj["char_id"] = _cid     # traceability; engine resolves by role
            objects.append(_obj)
    return {"version": 1, "objects": objects}, warnings


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
                "consistency_gate_warn_only": ("BOOLEAN", {"default": False}),
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
                 consistency_gate_warn_only=False, gate_in=""):
        try:
            led = json.loads(script_json or "{}")
            if not isinstance(led, dict):
                led = {}
        except (ValueError, TypeError):
            led = {}
        meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
        cast = led.get("cast") if isinstance(led.get("cast"), list) else []
        lines = led.get("lines") if isinstance(led.get("lines"), list) else []

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
            consistency_gate_warn_only=bool(consistency_gate_warn_only),
            lines=lines,
            still_aspects=_still_aspects_from_policy(image_policy_json),
            mesh_fodder_roles=_mesh_fodder_roles_from_policy(image_policy_json),
            talking_roles=_talking_roles_from_policy(image_policy_json),
            still_word_roles=_still_word_roles_from_policy(image_policy_json),
            video_models=_video_models_from_policy(image_policy_json),
        )  # aspects + mesh-fodder + talking + still_word roles + video_models ride in image_policy_json
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
