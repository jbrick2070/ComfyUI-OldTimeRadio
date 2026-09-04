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


def test_the_name_carries_all_five_choices(monkeypatch):
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    for field in ("cartoon", "wan_ti2v", "z_image_turbo", "indextts2",
                  "public_domain"):
        assert field in got, (field, got)


def test_episode_leads_and_style_follows(monkeypatch):
    """Operator's chosen order: the folder still sorts by episode, and the two
    axes he compares sit immediately after so they survive truncation."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert got.startswith("arms_at_the_ready_20260903_092133__")
    assert got.index("cartoon") < got.index("wan_ti2v") < got.index("z_image_turbo")
    assert got.index("wan_ti2v") < got.index("public_domain")


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
    # and the engine's role suffix is trimmed -- position already implies it
    assert "animatediff15_v3_haunted__" in got
    assert "haunted_video" not in got


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
    assert "cartoon" in got, "the fields must survive the trim, not the title"
