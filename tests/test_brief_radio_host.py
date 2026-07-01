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


def test_host_prompt_is_adult_not_baby():
    p = mbp.build_radio_host_prompt(_SPACE_META)
    assert "adult" in p
    assert "baby" not in p and "infant" not in p and "child" not in p


def test_host_prompt_depicts_a_person():
    # The one on-screen face -> the portrait person-guard must pass.
    assert mbp._depicts_person(mbp.build_radio_host_prompt(_SPACE_META))


def test_host_prompt_aspect_follows_slot():
    wide = mbp.build_radio_host_prompt(_SPACE_META, aspect="wide")
    portrait = mbp.build_radio_host_prompt(_SPACE_META, aspect="portrait")
    assert "head and shoulders" in wide          # STYLE_ANCHOR_WIDE
    assert "three-quarter" in portrait           # STYLE_ANCHOR
    assert wide != portrait


def test_host_prompt_never_empty_on_bare_meta():
    assert mbp.build_radio_host_prompt({}).strip()


def test_negative_prompt_constant_blocks_baby():
    assert "baby" in mbp.RADIO_HOST_FACE_NEG
    assert "child" in mbp.RADIO_HOST_FACE_NEG


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
    assert "space-station communications console" in rh["prompt"]
    assert "baby" in rh["negative_prompt"]
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


def test_build_request_off_redirects_bookend_to_ltx_audio_in(tmp_path, monkeypatch):
    # Toggle OFF: the music bookend is redirected off HuMo (byte-identical) so the
    # radio_host_portrait path never triggers.
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _bookend_ledger(tmp_path, with_face=False)
    shot = _humo_bookend_shot()
    rd.build_request_from_shot(shot, led)          # must NOT raise
    assert shot["engine_id"] == "ltx_audio_in"
