"""ADDENDUM A/B: ltx_audio_in radio-still face vs faceless (OTR_LTX_RADIO_FACE).

Contract: docs/2026-07-01-brief-driven-radio-host/ADDENDUM_ltx_audio_in_ab.md
(kibitz r1 hardened). SEPARATE from the main OTR_ENABLE_HUMO_HOSTS feature.

Covers: the toggle-gated WIDE radio-face still mint (default OFF = byte-identical),
its seed pin, and the render_driver ltx_audio_in bookend init branch -- including
the fail-LOUD paths (absent / not-wide) and the HuMo-hosts precedence rejection.
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

_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 80

_SPACE_META = {
    "story_brief_status": "ok",
    "style": "science fiction",
    "story_brief_terms": {"setting": ["an automated orbital docking station"]},
}


def _bookend_lines():
    return [
        {"line_id": "b000", "speaker_role": "music_open", "char_id": ""},
        {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"},
    ]


# --------------------------------------------------------------------------- #
# A1: mint (toggle-gated, wide, seed-pinned)
# --------------------------------------------------------------------------- #
def test_radio_face_stills_not_minted_when_toggle_off(monkeypatch):
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    out, _w = mbp.derive_image_prompts([], _SPACE_META, llm_fn=None,
                                       lines=_bookend_lines())
    ids = {o["object_id"] for o in out["objects"]}
    assert not any(i.endswith("_radio_face_169") for i in ids)


def test_radio_face_stills_minted_wide_when_toggle_on(monkeypatch):
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    out, _w = mbp.derive_image_prompts([], _SPACE_META, llm_fn=None,
                                       lines=_bookend_lines())
    objs = {o["object_id"]: o for o in out["objects"]}
    for role in ("announcer_visual", "music_visual"):
        oid = "still_%s_radio_face_169" % role
        o = objs[oid]
        assert o["kind"] == "portrait" and o["role"] == role
        assert o["w"] > o["h"], "must be WIDE (never a portrait -> pillarbox)"
        assert "space-station communications console" in o["prompt"]
    # LTX-ONLY mouth-forward styling (talking-radio kibitz r1, SUPERSEDES the
    # earlier per-role HuMo parity): BOTH bookend stills lead with the rubbery
    # grille-mouth -- the radio IS the host -- and keep humans OUT
    # (RADIO_CONSOLE_NEG; no baby line needed, no person in frame).
    for role in ("announcer_visual", "music_visual"):
        o = objs["still_%s_radio_face_169" % role]
        assert "rubbery mouth" in o["prompt"]
        assert "speaker grille" in o["prompt"]
        assert "human" in o["negative_prompt"]
        assert "baby" not in o["negative_prompt"]


def test_radio_face_still_seed_is_pinned():
    s = disp.resolve_object_seed({"request_seed": 7, "mode": "request_hash"},
                                 "still_music_visual_radio_face_169", "h",
                                 kind="portrait")
    assert s == 4242


def test_radio_face_off_is_byte_identical_to_no_toggle(monkeypatch):
    # OFF payload == payload with the var simply absent (no hidden objects).
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    a, _ = mbp.derive_image_prompts([], _SPACE_META, llm_fn=None,
                                    lines=_bookend_lines())
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "0")
    b, _ = mbp.derive_image_prompts([], _SPACE_META, llm_fn=None,
                                    lines=_bookend_lines())
    assert a == b


# --------------------------------------------------------------------------- #
# A2: render_driver ltx_audio_in bookend init branch
# --------------------------------------------------------------------------- #
def _ltx_ledger(tmp_path, face_dims=None, scene=True):
    imgs = []
    if scene:
        s = tmp_path / "scene.png"
        s.write_bytes(_PNG)
        imgs.append({"object_id": "still_b000", "kind": "scene_open",
                     "beat_id": "b000", "path": str(s)})
    if face_dims is not None:
        f = tmp_path / "face.png"
        f.write_bytes(_PNG)
        imgs.append({"object_id": "still_music_visual_radio_face_169",
                     "kind": "portrait", "path": str(f),
                     "w": face_dims[0], "h": face_dims[1]})
    return {"video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b000", "char_id": "",
                       "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": imgs}}


def _ltx_shot(role="music_visual"):
    return {"shot_id": "shot_b000", "beat_id": "b000", "engine_id": "ltx_audio_in",
            "role": role, "family": "audio_conditioned_video",
            "target_frame_count": 25, "source_line_ids": [], "char_id": "",
            "creative": {}}


def test_off_ltx_bookend_uses_faceless_scene_still(tmp_path, monkeypatch):
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    led = _ltx_ledger(tmp_path, face_dims=(1472, 832))   # face present but ignored
    req = rd.build_request_from_shot(_ltx_shot(), led)
    assert req["observability"]["init_source"] == "scene_still"
    assert req["observability"]["init_image"] == "scene.png"


def test_on_ltx_bookend_uses_wide_radio_face_still(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _ltx_ledger(tmp_path, face_dims=(1472, 832))
    req = rd.build_request_from_shot(_ltx_shot(), led)
    assert req["observability"]["init_source"] == "ltx_radio_face"
    assert req["observability"]["init_image"] == "face.png"


def test_on_fails_loud_when_face_still_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _ltx_ledger(tmp_path, face_dims=None)          # no face row
    with pytest.raises(rd.RenderError):
        rd.build_request_from_shot(_ltx_shot(), led)


def test_on_fails_loud_when_face_still_is_portrait(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _ltx_ledger(tmp_path, face_dims=(832, 1216))   # PORTRAIT -> pillarbox trap
    with pytest.raises(rd.RenderError):
        rd.build_request_from_shot(_ltx_shot(), led)


def test_precedence_rejects_when_humo_hosts_also_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    led = _ltx_ledger(tmp_path, face_dims=(1472, 832))
    with pytest.raises(rd.RenderError):
        rd.build_request_from_shot(_ltx_shot(), led)


def test_on_does_not_touch_character_beat(tmp_path, monkeypatch):
    # The A/B is announcer/music-only; a character ltx_audio_in beat never
    # takes the radio face. Under the default ia2v recipe the S4 route owns
    # the character init (cast portrait, 2026-07-02) -- so the fixture needs
    # a cast portrait and the assertion pins the S4 source explicitly.
    monkeypatch.setenv("OTR_LTX_RADIO_FACE", "1")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _ltx_ledger(tmp_path, face_dims=(1472, 832))
    p = tmp_path / "c01_portrait.png"
    p.write_bytes(_PNG)
    led["images"]["images"].append(
        {"object_id": "c01", "kind": "portrait", "path": str(p),
         "w": 832, "h": 1216})
    led["lines"].append({"line_id": "b002", "char_id": "c01",
                         "start_s": 2.0, "dur_s": 2.0})
    shot = dict(_ltx_shot(role="character_video"),
                shot_id="shot_b002", beat_id="b002",
                source_line_ids=["b002"], char_id="c01")
    req = rd.build_request_from_shot(shot, led)
    assert req["observability"]["init_source"] != "ltx_radio_face"
    assert req["observability"]["init_source"] == "character_portrait_ia2v"
