"""The preflight fetcher must fetch the files the lanes actually load.

The fetcher restates filenames the engine classes already declare, and a
restatement drifts. It did: the haunted bundle fetched `mm-p_0.5.pth` -- the
GOLDEN lane's motion module -- for a lane whose engine declares
`v3_sd15_mm.ckpt`, and it fetched no domain adapter at all. A fresh 8 GB user
would have downloaded 1.7 GB of the wrong file and still been unable to start
the lane the profile is named after.

Nothing caught it because both halves were internally consistent: the profile
named the right weights, the fetcher named plausible ones, and no test read
both. These tests read both.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_fetcher():
    """Import the fetcher by path -- `scripts/` is not a package."""
    path = ROOT / "scripts" / "otr_fetch_lane_weights.py"
    spec = importlib.util.spec_from_file_location("_otr_fetch_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _filenames(fetcher, entries):
    """The basename each entry lands on disk as."""
    return {fetcher.destination_name(entry) for entry in entries}


def test_haunted_bundle_fetches_what_the_haunted_engine_declares():
    """The exact bug: the bundle's files must BE the lane's files."""
    from nodes._otr_video_engines.eng_ghost_signal_official import (
        GhostSignalV3HauntedEngine as Haunted)

    fetcher = _load_fetcher()
    fetched = _filenames(fetcher, fetcher.LANES["haunted"])

    assert Haunted.motion_module_name in fetched, (
        "the haunted lane loads %r but the bundle does not fetch it -- this is "
        "the mm-p_0.5 defect returning" % (Haunted.motion_module_name,))
    assert Haunted.lora_name in fetched, (
        "the haunted lane's domain adapter %r is not fetched; without it the "
        "lane renders the CLEAN picture under a haunted receipt"
        % (Haunted.lora_name,))


def test_the_haunted_bundle_does_not_fetch_a_sibling_lanes_module():
    """A sibling's motion module is 1.7 GB of confidently wrong download."""
    from nodes._otr_video_engines.eng_ghost_signal import (
        GHOST_MOTION_MODULE_NAME as GOLDEN_MODULE)
    from nodes._otr_video_engines.eng_ghost_signal_official import (
        GhostSignalV3HauntedEngine as Haunted)

    fetcher = _load_fetcher()
    fetched = _filenames(fetcher, fetcher.LANES["haunted"])
    if GOLDEN_MODULE != Haunted.motion_module_name:
        assert GOLDEN_MODULE not in fetched, (
            "the haunted bundle fetches %r, which belongs to the GOLDEN lane "
            "(GhostSignalEngine), not to %s"
            % (GOLDEN_MODULE, Haunted.name))


@pytest.mark.parametrize("profile_id", ["otr_nvidia_8gb_haunted"])
def test_every_weight_a_profile_demands_is_in_its_bundle(profile_id):
    """A profile's preflight is a promise; the bundle has to be able to keep it.

    HF model ids (`org/model`) are skipped -- transformers fetches those on
    first use, which is exactly why the fetcher deliberately omits them.
    """
    fetcher = _load_fetcher()
    assert profile_id in fetcher.BUNDLES, (
        "%s ships as a named baseline but no bundle installs its weights"
        % profile_id)

    fetched = set()
    for lane in fetcher.BUNDLES[profile_id]:
        fetched |= _filenames(fetcher, fetcher.LANES[lane])

    profile = json.loads(
        (ROOT / "config" / "profiles" / ("%s.json" % profile_id)).read_text("utf-8"))
    demanded = [m for m in profile["preflight"]["required_models"] if "/" not in m]

    missing = [m for m in demanded if m not in fetched]
    assert not missing, (
        "%s preflights %r, which its bundle %r never downloads -- the profile "
        "would fail preflight on a machine that ran the fetcher exactly as "
        "documented" % (profile_id, missing, fetcher.BUNDLES[profile_id]))


def test_unqualified_8gb_h3_profile_has_no_public_bundle():
    """A draft experiment must never become a one-command support promise."""
    fetcher = _load_fetcher()
    assert "otr_nvidia_8gb_h3" not in fetcher.BUNDLES
