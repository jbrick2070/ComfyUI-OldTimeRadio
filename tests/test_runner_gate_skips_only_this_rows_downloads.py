"""The canonical runner's model gate skips a file only when THIS row fetches it.

FOUND BY REVIEW, 2026-09-25. The gate learned to step aside for weights the
queue-time preflight downloads itself (so an LTX 2.5 row is not refused for
files it is about to fetch). The first cut skipped any file whose basename
appeared ANYWHERE in the download allowlist. `otr_mac16_animatediff` declares
one file, the SD 1.5 checkpoint -- allowlisted for the `sd15` image engine --
so the gate checked nothing, while nothing at queue time fetched that file for
the AnimateDiff lane, which loads it inside its own graph. The row passed the
"refuse in seconds" gate on an empty server and would have died at the first
video beat.

Two fixes, both pinned here: the AnimateDiff lanes now ask the preflight for
their checkpoint, and the gate's skip set is computed from the row's own
engines through the same adapters the preflight asks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nodes import _otr_visual_assets as VA
from nodes._otr_shared import capability_profiles as cp

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SD15 = "v1-5-pruned-emaonly-fp16.safetensors"


def _runner():
    import otr_canonical_api_run as runner  # noqa: PLC0415
    return runner


def _gate(monkeypatch, profile):
    """Run the gate against ``profile`` on a server that can see nothing."""
    runner = _runner()
    monkeypatch.setattr(cp, "load_profile", lambda _name: profile)
    monkeypatch.setattr(runner, "_server_visible_model_names", lambda _s: set())
    return runner._assert_profile_models_present("synthetic", {})


def test_the_animatediff_set_is_every_registered_animatediff_lane():
    """A new AnimateDiff lane that loads the checkpoint must join the set, or
    it is back to having nobody fetch its checkpoint."""
    from nodes._otr_video_engines import registry as vreg  # noqa: PLC0415
    from nodes._otr_video_engines.eng_ghost_signal import (  # noqa: PLC0415
        GhostSignalEngine)
    registered = set()
    for eid in vreg.all_engine_names():
        engine = vreg.get_engine(eid)
        cls = engine if isinstance(engine, type) else type(engine)
        if issubclass(cls, GhostSignalEngine):
            registered.add(eid)
    assert registered == set(VA._ANIMATEDIFF_CHECKPOINT_ENGINES)


@pytest.mark.parametrize("engine", sorted(VA._ANIMATEDIFF_CHECKPOINT_ENGINES))
def test_each_animatediff_lane_downloads_its_own_checkpoint(engine):
    from nodes._otr_video_engines import eng_ghost_signal as gs  # noqa: PLC0415
    assert (gs.GHOST_CHECKPOINT_CATEGORY, gs.GHOST_CHECKPOINT_NAME) in (
        VA.planned_downloads({engine}))
    assert (gs.GHOST_CHECKPOINT_CATEGORY, gs.GHOST_CHECKPOINT_NAME) in VA.MANIFEST


def test_the_animatediff_lanes_stay_out_of_full_coverage():
    """Their motion module and adapter are not allowlisted, so the dropdown
    matrix must not read them as fetching everything."""
    assert not (VA._ANIMATEDIFF_CHECKPOINT_ENGINES & VA._COVERED)


def test_an_allowlisted_file_no_selected_engine_fetches_is_still_checked(monkeypatch):
    profile = {
        "role_overrides": {"announcer_visual": "viz_green",
                           "character_visual": "viz_green",
                           "music_visual": "viz_green"},
        "slot_overrides": {"video_render_engine": "viz_green"},
        "preflight": {"required_models": [_SD15]},
    }
    with pytest.raises(SystemExit) as refused:
        _gate(monkeypatch, profile)
    assert _SD15 in str(refused.value)


def test_a_file_this_rows_engine_fetches_is_not_a_refusal(monkeypatch):
    profile = {
        "role_overrides": {"announcer_visual": "animatediff15_lightning_video"},
        "slot_overrides": {"video_render_engine": "animatediff15_lightning_video"},
        "preflight": {"required_models": [_SD15]},
    }
    assert _gate(monkeypatch, profile) == []


def test_every_matrix_row_gates_only_what_its_own_engines_do_not_fetch(monkeypatch):
    """On an empty server, the only files the gate may refuse over are ones no
    engine of that row downloads at queue time."""
    runner = _runner()
    monkeypatch.setattr(runner, "_server_visible_model_names", lambda _s: set())
    load = cp.load_profile
    for rid in cp.known_profile_ids():
        profile = load(rid)
        picks = set((profile.get("role_overrides") or {}).values())
        slots = profile.get("slot_overrides") or {}
        picks.update(slots.get(k) for k in ("video_render_engine", "music_engine"))
        fetched = set()
        for pick in {p for p in picks if p}:
            try:
                fetched.update(name for _c, name in VA.planned_downloads({pick}))
            except VA.VisualAssetError:
                continue
        try:
            runner._assert_profile_models_present(rid, {})
        except SystemExit as refused:
            text = str(refused)
            assert not any(name in text for name in fetched), (
                "%s was refused over a file its own engines download: %s"
                % (rid, text))


def test_one_engine_that_cannot_plan_does_not_empty_the_skip_list(monkeypatch):
    """MEASURED LIVE 2026-09-24 on otr_16gb_animatediff: the row's z_image_turbo
    image slot could not be planned (its nvfp4 file has no allowlisted
    download), the whole-row plan raised, and the gate refused the SD 1.5
    checkpoint the AnimateDiff lane downloads for itself. Each engine is now
    planned on its own, so one refusal skips only that engine's files."""
    real = VA.planned_downloads

    def planned(engines):
        if "z_image_turbo" in engines:
            raise VA.VisualAssetError("no allowlisted download for this box")
        return real(engines)

    monkeypatch.setattr(VA, "planned_downloads", planned)
    profile = {
        "role_overrides": {"character_image": "z_image_turbo",
                           "character_visual": "animatediff15_v3_haunted_video"},
        "slot_overrides": {"video_render_engine": "animatediff15_v3_haunted_video"},
        "preflight": {"required_models": [_SD15]},
    }
    assert _gate(monkeypatch, profile) == []
