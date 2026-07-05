"""Brief-driven HuMo radio-host (2026-07-01) -- unit + acceptance tests.

Contract: docs/2026-07-01-brief-driven-radio-host/PLAN_HARDENED.md.

Chunk 1 (radio_form_from_meta / build_radio_host_prompt): a DETERMINISTIC brief
-> radio-form noun (no LLM), and the FULL animatable radio-HOST FACE prompt
(adult, never a baby; the ONLY on-screen face this feature grants).

Acceptance (Plan F): a non-1940s brief (the automated_space_docking episode)
(a) carries a radio-form noun, (b) passes _passes_consistency, (c) is NOT
scrubbed on the announcer-exempt path.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import otr_meta_brief_image_prompt as mbp  # noqa: E402
from nodes import otr_image_gen_dispatcher as disp  # noqa: E402
from nodes._otr_video_engines import render_driver as rd  # noqa: E402


_SPACE_META = {
    "story_brief_status": "ok",
    "style": "science fiction",
    "story_brief": "An automated docking sequence goes wrong in orbit.",
    "story_brief_terms": {
        "setting": ["an automated orbital docking station"],
        "lighting": ["cold blue panel glow"],
        "atmosphere": ["tense"],
    },
}

_NOIR_META = {
    "story_brief_status": "ok",
    "style": "1940s noir detective",
    "story_brief_terms": {"setting": ["a rain-slick city office"]},
}


# --------------------------------------------------------------------------- #
# radio_form_from_meta -- deterministic, no LLM
# --------------------------------------------------------------------------- #
def test_form_space_brief_maps_to_console():
    assert mbp.radio_form_from_meta(_SPACE_META) == (
        "a sleek space-station communications console")


def test_form_noir_brief_maps_to_deco():
    assert mbp.radio_form_from_meta(_NOIR_META) == "an art-deco bakelite tube radio"


def test_form_war_brief_maps_to_field_transceiver():
    meta = {"story_brief_terms": {"setting": ["a wartime military bunker"]}}
    assert mbp.radio_form_from_meta(meta) == (
        "a rugged portable field radio transceiver")


def test_form_default_is_not_the_retired_1940s_studio_anchor():
    # Empty / unknown brief -> a neutral radio FORM, never a 1940s studio.
    form = mbp.radio_form_from_meta({})
    assert form == mbp._RADIO_FORM_DEFAULT
    assert "1940s" not in form
    assert "studio" not in form


def test_form_is_deterministic():
    assert mbp.radio_form_from_meta(_SPACE_META) == mbp.radio_form_from_meta(_SPACE_META)


def test_form_tolerates_bad_meta():
    assert mbp.radio_form_from_meta(None) == mbp._RADIO_FORM_DEFAULT
    assert mbp.radio_form_from_meta("nonsense") == mbp._RADIO_FORM_DEFAULT


# --------------------------------------------------------------------------- #
# build_radio_host_prompt -- adult face, brief-driven form, never a baby
# --------------------------------------------------------------------------- #
def test_host_prompt_carries_the_brief_form_noun():
    p = mbp.build_radio_host_prompt(_SPACE_META)
    assert "space-station communications console" in p


def test_console_face_is_anthropomorphic_radio_no_human():
    # MUSIC style (default): the RADIO is the host -- an anthropomorphic console
    # with a dial-face, NOT a human presenter.
    p = mbp.build_radio_host_prompt(_SPACE_META, radio_host_style="console_face")
    assert "dial" in p and "face" in p            # dial-face (has face/eyes tokens)
    assert "human radio host" not in p            # never the old human host
    assert "human" not in p and "person" not in p  # no human in a pure console


import re


def test_radio_head_person_style_is_retired_and_raises():
    # radio-face logic 2026-07-04: the person-with-a-radio-head style is RETIRED.
    # Both the prompt builder and the negative dispatcher now fail LOUD on it
    # (an explicit closed-set dispatch, never a silent dial-face default).
    with pytest.raises(ValueError):
        mbp.build_radio_host_prompt(_SPACE_META,
                                    radio_host_style="radio_head_person")
    with pytest.raises(ValueError):
        mbp.radio_host_negative("radio_head_person")


def test_radio_object_is_faceless_no_anatomy():
    # The FACELESS static-bookend style: a stylized radio object, carrying the
    # brief-driven form, with ZERO anatomy tokens (facelessness is positive-side
    # -- the still dispatcher has no negative channel). Scope the no-anatomy
    # assertion to the constructed subject + anchor (the era/grade tails are
    # controlled, not sanitized).
    subj = "%s, %s" % (mbp.radio_form_from_meta(_SPACE_META),
                       mbp._RADIO_OBJECT_SUBJECT)
    for anchor in (mbp._RADIO_OBJECT_ANCHOR, mbp._RADIO_OBJECT_ANCHOR_WIDE):
        wording = (subj + ", " + anchor).lower()
        assert not re.search(r"\b(person|man|woman|presenter|figure|face)\b",
                             wording), wording
    p = mbp.build_radio_host_prompt(_SPACE_META, radio_host_style="radio_object")
    assert "space-station communications console" in p   # brief-driven form
    assert "tabletop" in p                                # object framing
    # a radio_object still shares the console negative (humans OUT, cosmetic)
    assert mbp.radio_host_negative("radio_object") == mbp.RADIO_CONSOLE_NEG


def test_radio_object_aspect_uses_object_anchor_not_person_anchor():
    wide = mbp.build_radio_host_prompt(_SPACE_META, aspect="wide",
                                       radio_host_style="radio_object")
    portrait = mbp.build_radio_host_prompt(_SPACE_META, aspect="portrait",
                                           radio_host_style="radio_object")
    assert "wide shot" in wide and "wide shot" not in portrait
    # never the person framing anchors
    for p in (wide, portrait):
        assert "head and shoulders" not in p
        assert "three-quarter" not in p
        assert "in-character" not in p


def test_overtness_is_brief_driven():
    # sci-fi brief -> overt cartoon face; a plain brief -> subtle.
    overt = mbp.build_radio_host_prompt(_SPACE_META, radio_host_style="console_face")
    subtle = mbp.build_radio_host_prompt(
        {"story_brief_terms": {"setting": ["a quiet village kitchen"]}},
        radio_host_style="console_face")
    assert "cartoon" in overt
    assert "subtle" in subtle and "cartoon" not in subtle


def test_host_prompt_aspect_follows_slot():
    wide = mbp.build_radio_host_prompt(_SPACE_META, aspect="wide")
    portrait = mbp.build_radio_host_prompt(_SPACE_META, aspect="portrait")
    assert "head and shoulders" in wide          # STYLE_ANCHOR_WIDE
    assert "three-quarter" in portrait           # STYLE_ANCHOR
    assert wide != portrait


def test_host_prompt_never_empty_on_bare_meta():
    assert mbp.build_radio_host_prompt({}).strip()


def test_negatives_by_style():
    # every live style is a pure radio (no person in frame) -> shares the console
    # negative (humans OUT); the retired radio_head_person now RAISES.
    assert "human" in mbp.radio_host_negative("console_face")
    assert "human" in mbp.radio_host_negative("radio_object")
    assert "human" in mbp.radio_host_negative("ltx_radio_mouth")
    with pytest.raises(ValueError):
        mbp.radio_host_negative("radio_head_person")


# --------------------------------------------------------------------------- #
# Acceptance (Plan F) -- non-1940s brief carries a radio-form noun + grounds
# --------------------------------------------------------------------------- #
def test_acceptance_space_docking_host_reads_as_radio_and_grounds():
    p = mbp.build_radio_host_prompt(_SPACE_META, aspect="portrait")
    # (a) carries a radio-form noun (a radio BODY, not a generic human host)
    assert "console" in p and ("radio" in p or "communications" in p)
    # (b) grounds on the brief (appearance/setting overlap)
    setting = mbp._read_setting(_SPACE_META)
    assert mbp._passes_consistency(p, "", setting)
    # (c) the announcer/radio-host path is gear-scrub EXEMPT -- the form noun
    # survives (scrubbing would strip "radio"/"console"); we assert the token
    # the exempt path preserves is present (never a 1940s studio revert).
    assert "1940s" not in p


# --------------------------------------------------------------------------- #
# Talking-radio kibitz r1 (Sub-plan B): the LTX-ONLY mouth-forward style
# (radio_host_style="ltx_radio_mouth") -- used ONLY by the OTR_LTX_RADIO_FACE still mint
# (the init stills the EXISTING ltx_audio_in engine receives; no new video
# model / path). The HuMo console_face / radio_head_person looks must stay
# BYTE-UNCHANGED (goldens below, captured pre-split @ 5cce9c2).
# --------------------------------------------------------------------------- #
def test_ltx_radio_mouth_leads_with_prominent_mouth():
    p = mbp.build_radio_host_prompt(_SPACE_META, "wide", radio_host_style="ltx_radio_mouth")
    # explicit MOUTH token (contract acceptance), grille-mouth phrasing intact
    assert "rubbery mouth" in p and "speaker grille" in p
    assert "anthropomorphic radio" in p
    # mouth-forward: the mouth clause precedes the eyes clause (FLUX weights
    # earlier tokens more heavily)
    assert p.index("mouth") < p.index("eyes")
    # brief-driven form noun survives
    assert "space-station communications console" in p
    # a pure radio face: no human presenter in the positive prompt
    assert "human" not in p and "person" not in p and "presenter" not in p


def test_ltx_radio_mouth_still_is_warm_not_brief_blue():
    # operator look direction 2026-07-02 (side-by-side catch): the lip-sync
    # still pins the canonical WARM look -- no brief palette ("cold blue
    # panel glow"), no grade tail ("heavy vignette, muted color grade").
    p = mbp.build_radio_host_prompt(_SPACE_META, "wide", radio_host_style="ltx_radio_mouth")
    assert "warm dramatic lighting" in p
    assert "cold blue" not in p
    assert "heavy vignette" not in p and "muted color grade" not in p
    # HuMo styles KEEP the brief-driven tail (golden-pinned elsewhere)
    hp = mbp.build_radio_host_prompt(_SPACE_META, "wide", radio_host_style="console_face")
    assert "cold blue panel glow" in hp


def test_ltx_radio_mouth_negative_is_console_no_human():
    # the mouth radio is a pure radio -> shares RADIO_CONSOLE_NEG (humans OUT;
    # no baby line needed -- no person in frame)
    assert mbp.radio_host_negative("ltx_radio_mouth") == mbp.RADIO_CONSOLE_NEG


def test_ltx_radio_mouth_overtness_is_brief_driven():
    # NOTE: the mouth constant itself says "cartoon appliance face" by design
    # (material anchoring, 2026-07-02) -- so the overtness discriminator is
    # the _radio_face_overtness PHRASE, not the bare "cartoon" token.
    overt = mbp.build_radio_host_prompt(_SPACE_META, radio_host_style="ltx_radio_mouth")
    subtle = mbp.build_radio_host_prompt(
        {"story_brief_terms": {"setting": ["a quiet village kitchen"]}},
        radio_host_style="ltx_radio_mouth")
    assert "bold playful" in overt and "subtle period-authentic" not in overt
    assert "subtle period-authentic" in subtle and "bold playful" not in subtle


#: GOLDENS captured from the PRE-split tree (HEAD 5cce9c2, 2026-07-02) via a
#: live venv run of build_radio_host_prompt. These pin the HuMo host looks
#: byte-for-byte: the ltx_radio_mouth split must NEVER drift them. Update ONLY
#: on a deliberate, operator-approved HuMo look change.
_GOLDEN_CONSOLE_PORTRAIT = (
    "a sleek space-station communications console, its glowing tuning dial "
    "forming an expressive stylized face -- two round dial-eyes and a "
    "radiating needle-fan mouth, an anthropomorphic radio that hosts the "
    "broadcast, bold playful cartoon expression, clearly a face, in-character "
    "cinematic three-quarter portrait, full head and face clearly visible "
    "with natural headroom above the head (never crop the top of the head), "
    "period-accurate costume and environment, dramatic film lighting, cold "
    "blue panel glow, cinematic, 35mm film look, subtle film grain, "
    "volumetric lighting, anamorphic lens, heavy vignette, muted color "
    "grade, sharp focus")
_GOLDEN_CONSOLE_WIDE = (
    "a sleek space-station communications console, its glowing tuning dial "
    "forming an expressive stylized face -- two round dial-eyes and a "
    "radiating needle-fan mouth, an anthropomorphic radio that hosts the "
    "broadcast, bold playful cartoon expression, clearly a face, in-character "
    "cinematic medium shot, head and shoulders, face clearly visible, "
    "subject centred with natural headroom above the head (never crop the "
    "top of the head), period-accurate costume and environment, dramatic "
    "film lighting, cold blue panel glow, cinematic, 35mm film look, subtle "
    "film grain, volumetric lighting, anamorphic lens, heavy vignette, muted "
    "color grade, sharp focus")
_GOLDEN_BARE_CONSOLE_PORTRAIT = (
    "a vintage tabletop tube radio receiver, its glowing tuning dial forming "
    "an expressive stylized face -- two round dial-eyes and a radiating "
    "needle-fan mouth, an anthropomorphic radio that hosts the broadcast, "
    "subtle period-authentic dial-face, in-character cinematic three-quarter "
    "portrait, full head and face clearly visible with natural headroom "
    "above the head (never crop the top of the head), period-accurate "
    "costume and environment, dramatic film lighting, timeless cinematic "
    "aesthetic, cinematic, 35mm film look, subtle film grain, volumetric "
    "lighting, anamorphic lens, heavy vignette, muted color grade, sharp "
    "focus")


def test_humo_console_face_prompts_byte_unchanged():
    assert mbp.build_radio_host_prompt(
        _SPACE_META, "portrait", "console_face") == _GOLDEN_CONSOLE_PORTRAIT
    assert mbp.build_radio_host_prompt(
        _SPACE_META, "wide", "console_face") == _GOLDEN_CONSOLE_WIDE
    assert mbp.build_radio_host_prompt(
        {}, "portrait", "console_face") == _GOLDEN_BARE_CONSOLE_PORTRAIT


def test_mint_split_ltx_mouth_humo_untouched(monkeypatch):
    # BOTH toggles on at MINT time (precedence is a render_driver concern): the
    # ltx radio-face still (announcer-only since 2026-07-03; the music mint was
    # pruned 2026-07-04) carries the mouth style; the HuMo radio_host portrait
    # does NOT. With the announcer engine resolving to HuMo (audio_driven_face),
    # the announcer portrait row is the console dial-face, never the mouth style.
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    out, _w = mbp.derive_image_prompts(
        [], _SPACE_META, llm_fn=None, lines=_bookend_lines(),
        video_models={"announcer_video_model": {"engine_id": "humo"}})
    objs = {o["object_id"]: o for o in out["objects"]}
    assert "rubbery mouth" in objs["still_announcer_visual_radio_face_169"]["prompt"]
    # the pruned music radio-face id is never minted
    assert "still_music_visual_radio_face_169" not in objs
    assert "rubbery" not in objs[mbp.RADIO_HOST_PORTRAIT_ID]["prompt"]
    assert objs[mbp.RADIO_HOST_PORTRAIT_ID]["prompt"] == mbp.build_radio_host_prompt(
        _SPACE_META, "portrait", "console_face")     # the golden-pinned look
    # announcer portrait -> console dial-face (HuMo drives it), not the mouth style
    assert "rubbery" not in objs["announcer"]["prompt"]
    assert objs["announcer"]["radio_host_style"] == "console_face"
    assert objs["announcer"]["prompt"] == mbp.build_radio_host_prompt(
        _SPACE_META, "portrait", "console_face")


# --------------------------------------------------------------------------- #
# Chunk 3: radio_host_portrait object mint (TOGGLE-GATED, seed-pinned, aspect)
# --------------------------------------------------------------------------- #
def _bookend_lines():
    return [
        {"line_id": "b000", "speaker_role": "music_open", "char_id": ""},
        {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"},
    ]


def test_radio_host_portrait_not_minted_when_toggle_off(monkeypatch):
    # Default OFF -> byte-identical: NO extra radio-host object in the payload.
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    out, _w = mbp.derive_image_prompts([], _SPACE_META, llm_fn=None,
                                       lines=_bookend_lines())
    ids = {o["object_id"] for o in out["objects"]}
    assert mbp.RADIO_HOST_PORTRAIT_ID not in ids


def test_radio_host_portrait_minted_when_toggle_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    out, _w = mbp.derive_image_prompts(
        [], _SPACE_META, llm_fn=None, lines=_bookend_lines(),
        still_aspects={"music_visual": "wide"})
    objs = {o["object_id"]: o for o in out["objects"]}
    rh = objs[mbp.RADIO_HOST_PORTRAIT_ID]
    assert rh["kind"] == "portrait" and rh["role"] == "music_visual"
    # MUSIC -> anthropomorphic console face (brief-driven form), no human
    assert "space-station communications console" in rh["prompt"]
    assert "dial" in rh["prompt"] and "face" in rh["prompt"]
    assert "human" in rh["negative_prompt"]        # console keeps humans OUT
    # aspect FOLLOWS the HuMo (music) slot -> WIDE dims (w > h), never pillarboxed
    assert rh["w"] > rh["h"]


def test_radio_host_portrait_seed_is_pinned():
    # object_id radio_host_portrait -> the FIXED bookend seed (4242 default),
    # independent of the request hash, so open/inter/close share ONE face.
    s = disp.resolve_object_seed({"request_seed": 999, "mode": "request_hash"},
                                 mbp.RADIO_HOST_PORTRAIT_ID, "anyhash",
                                 kind="portrait")
    assert s == 4242


# --------------------------------------------------------------------------- #
# Chunk 4: OTR_ENABLE_HUMO_HOSTS toggle in render_driver
# --------------------------------------------------------------------------- #
def _humo_bookend_shot(role="music_visual"):
    return {"shot_id": "shot_b000", "beat_id": "b000", "engine_id": "humo",
            "role": role, "family": "audio_driven_face",
            "target_frame_count": 25, "source_line_ids": [], "char_id": "",
            "creative": {}}


def test_enforce_radio_is_host_redirects_when_toggle_off(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    shot = _humo_bookend_shot()
    rd._enforce_radio_is_host(shot)
    assert shot["engine_id"] == "ltx_audio_in"     # today's behavior byte-for-byte


def test_enforce_radio_is_host_noop_when_toggle_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    shot = _humo_bookend_shot()
    rd._enforce_radio_is_host(shot)
    assert shot["engine_id"] == "humo"             # HuMo radio-host allowed


def _bookend_ledger(tmp_path, with_face=True):
    imgs = []
    if with_face:
        face = tmp_path / "radio_host.png"
        face.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        imgs.append({"object_id": "radio_host_portrait", "kind": "portrait",
                     "path": str(face)})
    return {"video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b000", "char_id": "",
                       "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": imgs}}


def test_build_request_uses_radio_host_portrait_when_toggle_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _bookend_ledger(tmp_path, with_face=True)
    shot = _humo_bookend_shot()
    req = rd.build_request_from_shot(shot, led)
    assert shot["engine_id"] == "humo"             # not redirected
    assert req["observability"]["init_source"] == "radio_host_portrait"
    assert req["observability"]["init_image"] == "radio_host.png"


def test_build_request_fails_loud_without_face_when_toggle_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _bookend_ledger(tmp_path, with_face=False)
    with pytest.raises(rd.RenderError):
        rd.build_request_from_shot(_humo_bookend_shot(), led)


def test_humo_host_bookend_uses_ambient_master_audio(monkeypatch):
    # A lineless music/announcer bookend routed to HuMo under the toggle has no
    # per-line voice but HuMo hard-requires audio_ref -> it must use the ambient
    # MASTER slice (host "sings" to the bed). Off / character-face stay excluded.
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    assert rd._uses_ambient_master_audio("humo", "audio_driven_face",
                                         is_char_face=False, role="music_visual")
    assert rd._uses_ambient_master_audio("humo", "audio_driven_face",
                                         is_char_face=False, role="announcer_visual")
    # a real CHARACTER face beat still uses its own voice (never the mix)
    assert not rd._uses_ambient_master_audio("humo", "audio_driven_face",
                                             is_char_face=True, role="character_video")
    # character_video (not a bookend role) is not ambient even lineless
    assert not rd._uses_ambient_master_audio("humo", "audio_driven_face",
                                             is_char_face=False, role="character_video")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    assert not rd._uses_ambient_master_audio("humo", "audio_driven_face",
                                             is_char_face=False, role="music_visual")


def test_build_request_off_redirects_bookend_to_ltx_audio_in(tmp_path, monkeypatch):
    # Toggle OFF: the announcer bookend is redirected off HuMo so the
    # radio_host_portrait path never triggers. S4c (2026-07-02): the
    # redirected ltx bookend now runs the TALKING register (default dev
    # unet), which REQUIRES the wide radio-face still -- a ledger minted
    # without one fails LOUD (never a silent faceless bookend again), but
    # the redirect itself must already be stamped on the shot.
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    led = _bookend_ledger(tmp_path, with_face=False)
    shot = _humo_bookend_shot(role="announcer_visual")
    with pytest.raises(rd.RenderError, match="radio-face"):
        rd.build_request_from_shot(shot, led)
    assert shot["engine_id"] == "ltx_audio_in"     # redirect happened first


def test_build_request_off_redirect_renders_on_single_pass(tmp_path, monkeypatch):
    # The pre-S4c contract survives on the single-pass recipes: no talking
    # register, no face requirement -- the redirected bookend renders from
    # its scene still exactly as before.
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    led = _bookend_ledger(tmp_path, with_face=False)
    shot = _humo_bookend_shot()
    rd.build_request_from_shot(shot, led)          # must NOT raise
    assert shot["engine_id"] == "ltx_audio_in"


# --------------------------------------------------------------------------- #
# RADIO FACE LOGIC (2026-07-04): the synthetic-announcer portrait row is FACELESS
# (radio_object) when static, the animatable console dial-face when a HuMo
# (audio_driven_face) engine will drive it. The signal is the RESOLVED announcer
# engine family (after OTR_FORCE_ENGINE_MAP), not the bare HUMO flag.
# --------------------------------------------------------------------------- #
def _announcer_row(video_models=None, **kw):
    out, _w = mbp.derive_image_prompts(
        [], _SPACE_META, llm_fn=None, lines=_bookend_lines(),
        video_models=video_models, **kw)
    return {o["object_id"]: o for o in out["objects"]}["announcer"]


def test_announcer_faceless_radio_object_when_humo_off(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    ann = _announcer_row(video_models={"announcer_video_model": {"engine_id": "humo"}})
    assert ann["radio_host_style"] == "radio_object"
    assert ann["prompt"] == mbp.build_radio_host_prompt(
        _SPACE_META, "portrait", "radio_object")
    # faceless: no person tokens in the announcer prompt subject/anchor
    assert "presenter" not in ann["prompt"] and "radio-head" not in ann["prompt"]
    assert ann["negative_prompt"] == mbp.RADIO_CONSOLE_NEG


def test_announcer_console_face_when_humo_on_and_audio_driven(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    ann = _announcer_row(video_models={"announcer_video_model": {"engine_id": "humo"}})
    assert ann["radio_host_style"] == "console_face"
    assert ann["prompt"] == mbp.build_radio_host_prompt(
        _SPACE_META, "portrait", "console_face")


def test_announcer_faceless_when_humo_on_but_engine_static(monkeypatch):
    # HUMO on but the announcer engine is a STATIC engine -> nothing animates a
    # mouth -> faceless radio_object (the family signal, not the bare flag).
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    ann = _announcer_row(
        video_models={"announcer_video_model": {"engine_id": "still_flat"}})
    assert ann["radio_host_style"] == "radio_object"


def test_announcer_faceless_when_humo_on_but_no_video_models(monkeypatch):
    # No policy forwarded -> family unresolved -> fail SAFE toward faceless.
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    ann = _announcer_row(video_models=None)
    assert ann["radio_host_style"] == "radio_object"


def test_announcer_force_map_overrides_to_faceless(monkeypatch):
    # HUMO on + policy selects HuMo for the announcer, but OTR_FORCE_ENGINE_MAP
    # forces announcer_visual to a static engine -> the RESOLVED family is
    # static -> faceless radio_object (r4 must-fix: honor the force map).
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "announcer_visual=still_flat")
    ann = _announcer_row(video_models={"announcer_video_model": {"engine_id": "humo"}})
    assert ann["radio_host_style"] == "radio_object"


def test_announcer_no_force_with_humo_is_console_face(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    ann = _announcer_row(video_models={"announcer_video_model": {"engine_id": "humo"}})
    assert ann["radio_host_style"] == "console_face"


# --------------------------------------------------------------------------- #
# Cross-node provenance guard (fail CLOSED): a ledger minted HUMO-OFF (faceless
# radio_object announcer portrait) but RENDERED HUMO-ON must NOT feed HuMo the
# faceless still (the visual-quicktest frozen-ledger reuse path is real).
# --------------------------------------------------------------------------- #
def _announcer_humo_shot():
    return {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "humo",
            "role": "announcer_visual", "family": "audio_driven_face",
            "target_frame_count": 25, "source_line_ids": ["b001"],
            "char_id": "announcer", "creative": {}}


def _announcer_portrait_ledger(tmp_path, style):
    p = tmp_path / "announcer.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
    row = {"object_id": "announcer", "kind": "portrait", "path": str(p)}
    if style is not None:
        row["radio_host_style"] = style
    return {"video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b001", "char_id": "announcer",
                       "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": [row]}}


def test_render_guard_raises_on_stale_faceless_announcer_init(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _announcer_portrait_ledger(tmp_path, style="radio_object")
    with pytest.raises(rd.RenderError, match="RADIO FACE LOGIC"):
        rd.build_request_from_shot(_announcer_humo_shot(), led)


def test_render_guard_raises_on_missing_style_frozen_ledger(tmp_path, monkeypatch):
    # An old/frozen ledger carries NO radio_host_style -> treated as NOT
    # console_face -> fail closed (never only when it equals radio_object).
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _announcer_portrait_ledger(tmp_path, style=None)
    with pytest.raises(rd.RenderError, match="RADIO FACE LOGIC"):
        rd.build_request_from_shot(_announcer_humo_shot(), led)


def test_render_guard_passes_when_console_face(tmp_path, monkeypatch):
    # console_face is the animatable dial-face -> the guard does NOT fire; the
    # announcer portrait is consumed as the HuMo init.
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _announcer_portrait_ledger(tmp_path, style="console_face")
    req = rd.build_request_from_shot(_announcer_humo_shot(), led)
    assert req["observability"]["init_source"] == "portrait"
    assert req["observability"]["init_image"] == "announcer.png"


def test_render_guard_inert_when_humo_off(tmp_path, monkeypatch):
    # HUMO off: _enforce_radio_is_host redirects the announcer off HuMo before
    # the guard's family check, so a faceless radio_object never trips it.
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    led = _announcer_portrait_ledger(tmp_path, style="radio_object")
    rd.build_request_from_shot(_announcer_humo_shot(), led)   # must NOT raise
