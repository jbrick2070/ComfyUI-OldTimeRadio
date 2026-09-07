"""The obs filename says what MADE the episode.

WHY THIS EXISTS (operator, 2026-09-03). The published copy used to inherit the
archival stem verbatim, so every episode in `otr/obs/` read

    signal_lost_<title>_<ts>_silent_procgen_blended_captioned_with_credits_final.mp4

and a file browser truncated every single row at the identical, useless point --
`..._silent_procgen_blended_captioned_wit...`. The operator sent a screenshot of
exactly that: fifteen rows, indistinguishable past the title.

Worse than useless, that tail is MISLEADING. `procgen` is a compositing stage,
not a render engine, and this very session read it as the engine and built a
whole wrong diagnosis on it ("88 of 89 episodes are static") before the ledgers
corrected it. A name that invites a wrong reading is a defect.

The obs copy now carries the five choices that produced the episode, in the
order the operator picked: episode first (so the folder still sorts by episode),
then style and video engine (the axes he compares, and the ones that must
survive truncation).

The ARCHIVAL copy in `otr/episodes/` is deliberately untouched -- its suffixes
carry pipeline provenance, `otr_caption_burn` strips those exact spellings, and
nothing that globs the archival stem may break.
"""
import os

import pytest

from nodes import otr_master_audio_mux as mux

ARCHIVAL = ("signal_lost_arms_at_the_ready_20260903_092133"
            "_silent_procgen_blended_captioned_with_credits_final.mp4")


class _Ledger:
    """Stand-in for the in-flight ledger module."""

    def __init__(self, payload):
        self.payload = payload

    def in_flight_ledger_path(self):
        return "in-memory"

    def load_ledger_safe(self, _path):
        return self.payload


def _install(monkeypatch, payload):
    import sys
    stub = _Ledger(payload)
    monkeypatch.setitem(sys.modules, "_otr_ledger", stub)
    monkeypatch.setattr(mux, "_otr_ledger", stub, raising=False)
    # The helper imports `from . import _otr_ledger`, so patch the package too.
    import nodes
    monkeypatch.setattr(nodes, "_otr_ledger", stub, raising=False)
    return stub


_FULL = {
    "meta": {"visual_style": "cartoon", "source_bank": "public_domain",
             "char_voice_engine": "indextts2",
             # Added 2026-09-07 with the writer + music fields. Both are real
             # ledger keys, confirmed against a production ledger on the 4060.
             "creative_writing_model": "Qwen/Qwen3.5-4B",
             "music_engine": "musicgen",
             "image_engines": {"by_role": {"character_video": {"z_image_turbo": 4}}}},
    "video": {"shots": [{"engine_id": "wan_ti2v"} for _ in range(8)]},
}


def test_the_pipeline_suffix_tail_is_gone(monkeypatch):
    """`_silent_procgen_blended_captioned_with_credits` is compositing noise and
    must not reach the folder the operator watches."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    for noise in ("procgen", "blended", "captioned", "silent", "with_credits"):
        assert noise not in got, (noise, got)


def test_the_name_carries_every_choice_as_a_short_code(monkeypatch):
    """Operator ruling 2026-09-07: four characters (five for the writer).

    Spelled in full these fields put the name at 249 of its 250-unit budget on
    a ComfyUI Desktop install -- one unit -- and two more dimensions were wanted
    in it. The codes come from `_otr_shared/shortcodes.py`, whose own tests keep
    the table complete against the live dropdowns and engine registry.
    """
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    for field in ("cart", "wti2", "zimg", "idx2", "pubd", "q354b", "mgen"):
        assert field in got, (field, got)
    # and the spelled-out forms are GONE -- that is the point of the change
    for spelled in ("cartoon", "wan_ti2v", "z_image_turbo", "indextts2",
                    "public_domain", "musicgen"):
        assert spelled not in got, (spelled, got)


def test_the_writer_and_music_dimensions_are_present(monkeypatch):
    """Added 2026-09-07. The writer LLM and the music engine were invisible in
    the published name, so two episodes differing only by writer were
    indistinguishable in the folder the operator actually watches."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert "q354b" in got, got
    assert "mgen" in got, got


def test_episode_leads_and_style_follows(monkeypatch):
    """Operator's chosen order: the folder still sorts by episode, and the two
    axes he compares sit immediately after so they survive truncation."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert got.startswith("arms_at_the_ready_20260903_092133__")
    assert got.index("cart") < got.index("wti2") < got.index("zimg")
    assert got.index("wti2") < got.index("pubd")


def test_the_final_marker_survives(monkeypatch):
    """`scripts/otr_pod_obs_bridge.py` keys on `_final` to recognise a published
    episode -- dropping it would make published work invisible to the bridge."""
    _install(monkeypatch, _FULL)
    assert mux._obs_basename(ARCHIVAL).endswith("_final.mp4")


def test_a_lane_with_no_stills_says_none_rather_than_lying(monkeypatch):
    """Ghost/AnimateDiff renders no stills, so `image_engines.by_role` is empty.
    The field must read `none`, not borrow some other episode's engine."""
    payload = {"meta": dict(_FULL["meta"], image_engines={"by_role": {}}),
               "video": {"shots": [{"engine_id": "animatediff15_v3_haunted_video"}]}}
    _install(monkeypatch, payload)
    got = mux._obs_basename(ARCHIVAL)
    assert "__none__" in got
    # The engine id is coded WHOLE. The old `_trim_engine` stripped a trailing
    # `_video`/`_image` because the field position implied the role -- but the
    # shortcode table is keyed on the engine id exactly as the registry spells
    # it, so trimming first would hand it a name it has never heard of and
    # spell the lane `unk`.
    assert "adhv" in got, got
    assert "animatediff" not in got, got


def test_it_fails_soft_to_the_archival_name(monkeypatch):
    """A publish must never die over a filename. THIS TEST EARNED ITS KEEP: the
    first cut of the helper referenced `re` without importing it, and the broad
    except swallowed the NameError -- silently disabling the whole feature while
    every publish still 'worked'."""
    import sys

    class _Boom:
        def in_flight_ledger_path(self):
            raise RuntimeError("ledger unavailable")

    monkeypatch.setitem(sys.modules, "_otr_ledger", _Boom())
    import nodes
    monkeypatch.setattr(nodes, "_otr_ledger", _Boom(), raising=False)
    assert mux._obs_basename(ARCHIVAL) == ARCHIVAL


def test_the_helper_has_its_imports(monkeypatch):
    """The guard for the bug the soft-fallback hid: exercise the REAL body and
    assert it produced a composed name, not the fallback."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert got != ARCHIVAL, "fell back -- the helper body raised"
    assert "__" in got


def test_fields_are_filesystem_safe():
    assert mux._obs_field("weird/name:here") == "weird-name-here"
    assert mux._obs_field("Anime") == "anime"
    assert mux._obs_field(None) == "none"
    assert mux._obs_field("", "nostyle") == "nostyle"


def test_a_very_long_title_is_capped(monkeypatch):
    _install(monkeypatch, _FULL)
    long_stem = ("signal_lost_" + ("a_very_long_episode_title_" * 8)
                 + "20260903_092133_silent_procgen_blended_captioned_with_credits_final.mp4")
    got = mux._obs_basename(long_stem)
    assert len(got) <= mux._OBS_NAME_MAX + 8, len(got)
    assert got.endswith("_final.mp4")
    assert "cart" in got, "the fields must survive the trim, not the title"
