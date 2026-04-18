"""
test_planner.py  --  Day 9 orchestration timeline planner regression
===================================================================

Validates ``otr_v2.hyworld.planner.plan_episode`` across:

    * Backend assignment (explicit override, kind-based inference,
      graceful degradation when identity/composite inputs are missing)
    * Duration handling (C4 10s clamp on motion/loop, default fallback
      when duration_s <= 0, total runtime coverage)
    * Non-repetition sliding window (no duplicate backend+prompt_hash
      pair inside the configurable window)
    * Runtime coverage (target_runtime_s satisfied by scene rotation)
    * Handoff selection (ltx_motion / wan21_loop pick the most recent
      in-scene still producer as upstream)
    * Shotlist JSON schema (matches bridge's ``{"shots": [...]}`` contract)
    * 3-min dry run: 180s outline emits clean, non-repeating jobs

All torch-free, no GPU required.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _mini_outline(runtime_s: float = 30.0) -> dict:
    """Minimal 2-scene outline with one beat per kind."""
    return {
        "title": "PLANNER TEST",
        "runtime_s": runtime_s,
        "scenes": [
            {
                "scene_id": "scene_01",
                "location": "Cockpit",
                "characters": ["BABA"],
                "beats": [
                    {
                        "beat_id": "b01",
                        "kind": "establishing",
                        "prompt": "wide shot of the cockpit at dawn",
                        "duration_s": 6.0,
                    },
                    {
                        "beat_id": "b02",
                        "kind": "close_up",
                        "prompt": "Baba staring at the control panel",
                        "duration_s": 4.0,
                        "character": "BABA",
                        "refs": ["refs/baba_01.png", "refs/baba_02.png"],
                    },
                    {
                        "beat_id": "b03",
                        "kind": "motion",
                        "prompt": "camera pushes in slowly",
                        "duration_s": 8.0,
                    },
                ],
            },
            {
                "scene_id": "scene_02",
                "location": "Corridor",
                "characters": ["BOOEY"],
                "beats": [
                    {
                        "beat_id": "b04",
                        "kind": "loop",
                        "prompt": "ambient corridor lights flickering",
                        "duration_s": 9.0,
                    },
                    {
                        "beat_id": "b05",
                        "kind": "insert",
                        "prompt": "HUD overlay",
                        "duration_s": 3.0,
                        "mask_prompt": "viewport frame",
                        "insert_prompt": "red alert graphic",
                    },
                ],
            },
        ],
    }


# ------------------------------------------------------------------
# Module surface / imports
# ------------------------------------------------------------------


def test_module_imports_cleanly():
    from otr_v2.hyworld import planner

    assert hasattr(planner, "plan_episode")
    assert hasattr(planner, "PlannerJob")
    assert hasattr(planner, "PlannerResult")
    assert hasattr(planner, "emit_shotlist_json")
    assert hasattr(planner, "write_shotlist")


def test_public_constants_sane():
    from otr_v2.hyworld.planner import (
        MAX_MOTION_DURATION_S,
        DEFAULT_NONREPEAT_WINDOW,
        HANDOFF_BACKENDS,
        IDENTITY_BACKENDS,
        COMPOSITE_BACKENDS,
    )

    assert MAX_MOTION_DURATION_S == 10.0  # C4
    assert DEFAULT_NONREPEAT_WINDOW >= 1
    assert "ltx_motion" in HANDOFF_BACKENDS
    assert "wan21_loop" in HANDOFF_BACKENDS
    assert "pulid_portrait" in IDENTITY_BACKENDS
    assert "florence2_sdxl_comp" in COMPOSITE_BACKENDS


# ------------------------------------------------------------------
# Backend assignment
# ------------------------------------------------------------------


def test_establishing_kind_picks_flux_anchor():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 6.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "establishing",
                "prompt": "wide cockpit", "duration_s": 6.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "flux_anchor"


def test_close_up_with_character_picks_pulid():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 4.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "close_up",
                "prompt": "Baba looks up",
                "duration_s": 4.0,
                "character": "BABA",
                "refs": ["baba.png"],
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "pulid_portrait"
    assert r.jobs[0].character == "BABA"
    assert r.jobs[0].refs == ("baba.png",)


def test_close_up_without_refs_degrades_to_keyframe():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 4.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "close_up",
                "prompt": "unnamed close-up",
                "duration_s": 4.0,
            }],
        }],
    }
    r = plan_episode(outline)
    # Without refs we can't do identity lock, so keyframe.
    assert r.jobs[0].backend == "flux_keyframe"


def test_motion_kind_picks_ltx():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 6.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "motion",
                "prompt": "pan across the horizon",
                "duration_s": 6.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "ltx_motion"


def test_loop_kind_picks_wan21():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 8.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "loop",
                "prompt": "idle ambient",
                "duration_s": 8.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "wan21_loop"


def test_insert_with_mask_and_insert_picks_florence():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 3.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "insert",
                "prompt": "overlay",
                "duration_s": 3.0,
                "mask_prompt": "viewport frame",
                "insert_prompt": "red HUD",
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "florence2_sdxl_comp"
    assert r.jobs[0].mask_prompt == "viewport frame"
    assert r.jobs[0].insert_prompt == "red HUD"


def test_insert_without_mask_degrades_to_keyframe():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 3.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "insert",
                "prompt": "overlay",
                "duration_s": 3.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "flux_keyframe"
    assert any("degraded" in w for w in r.warnings)


def test_explicit_backend_overrides_kind():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 5.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "establishing",
                "backend": "flux_keyframe",
                "prompt": "override test",
                "duration_s": 5.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "flux_keyframe"


def test_explicit_unknown_backend_raises():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 5.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1",
                "backend": "not_a_real_backend",
                "prompt": "x", "duration_s": 5.0,
            }],
        }],
    }
    with pytest.raises(ValueError):
        plan_episode(outline)


# ------------------------------------------------------------------
# Duration handling (C4 clamp)
# ------------------------------------------------------------------


def test_motion_duration_clamped_to_C4_cap():
    from otr_v2.hyworld.planner import plan_episode, MAX_MOTION_DURATION_S

    outline = {
        "runtime_s": 20.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "motion",
                "prompt": "long motion",
                "duration_s": 20.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].duration_s == MAX_MOTION_DURATION_S
    assert any("C4" in w for w in r.warnings)


def test_loop_duration_clamped_to_C4_cap():
    from otr_v2.hyworld.planner import plan_episode, MAX_MOTION_DURATION_S

    outline = {
        "runtime_s": 15.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "loop",
                "prompt": "long loop",
                "duration_s": 15.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].duration_s == MAX_MOTION_DURATION_S


def test_still_backend_duration_not_clamped_above_10s():
    from otr_v2.hyworld.planner import plan_episode

    # Still frames (flux_anchor) can hold for 20s without C4 tripping.
    outline = {
        "runtime_s": 20.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "establishing",
                "prompt": "wide",
                "duration_s": 20.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].duration_s == 20.0


def test_negative_duration_replaced_with_default():
    from otr_v2.hyworld.planner import plan_episode, DEFAULT_BEAT_DURATION_S

    outline = {
        "runtime_s": 10.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [{
                "beat_id": "b1", "kind": "establishing",
                "prompt": "x",
                "duration_s": -5.0,
            }],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].duration_s == DEFAULT_BEAT_DURATION_S


# ------------------------------------------------------------------
# Non-repetition window
# ------------------------------------------------------------------


def test_no_duplicate_backend_prompt_in_window():
    """Three identical beats in a row must be nudged to unique hashes
    inside the default window."""
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 15.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [
                {"beat_id": f"b{i}", "kind": "establishing",
                 "prompt": "identical wide shot", "duration_s": 5.0}
                for i in range(3)
            ],
        }],
    }
    r = plan_episode(outline)
    hashes = [(j.backend, j.prompt_hash) for j in r.jobs]
    assert len(set(hashes)) == 3  # all unique after nudging


def test_nonrepeat_window_configurable():
    """Window=1 allows repeats two-apart; window=5 does not."""
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 30.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [
                {"beat_id": f"b{i}", "kind": "establishing",
                 "prompt": f"variant {i % 2}", "duration_s": 5.0}
                for i in range(6)
            ],
        }],
    }

    r1 = plan_episode(outline, nonrepeat_window=1)
    r5 = plan_episode(outline, nonrepeat_window=5)

    # Window=1: alternating a/b/a/b/a/b works without nudging.
    # Window=5: same alternation gets nudged.
    hashes_w1 = [j.prompt_hash for j in r1.jobs]
    hashes_w5 = [j.prompt_hash for j in r5.jobs]
    assert len(set(hashes_w5)) >= len(set(hashes_w1))


def test_nudged_prompt_is_deterministic():
    """Same outline planned twice -> same nudges -> same hashes."""
    from otr_v2.hyworld.planner import plan_episode

    outline = _mini_outline(30)
    r1 = plan_episode(outline)
    r2 = plan_episode(outline)

    assert [j.prompt_hash for j in r1.jobs] == [j.prompt_hash for j in r2.jobs]


# ------------------------------------------------------------------
# Handoff selection
# ------------------------------------------------------------------


def test_motion_beat_picks_prior_still_as_handoff():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 12.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [
                {"beat_id": "b1", "kind": "establishing",
                 "prompt": "wide", "duration_s": 6.0},
                {"beat_id": "b2", "kind": "motion",
                 "prompt": "move", "duration_s": 6.0},
            ],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].backend == "flux_anchor"
    assert r.jobs[1].backend == "ltx_motion"
    assert r.jobs[1].handoff_from == r.jobs[0].shot_id


def test_motion_beat_without_upstream_still_warns():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 6.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [
                {"beat_id": "b1", "kind": "motion",
                 "prompt": "move", "duration_s": 6.0},
            ],
        }],
    }
    r = plan_episode(outline)
    assert r.jobs[0].handoff_from is None
    assert any("upstream still" in w for w in r.warnings)


def test_motion_handoff_crosses_scene_boundary_only_within_scene():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 14.0,
        "scenes": [
            {"scene_id": "s1", "beats": [
                {"beat_id": "b1", "kind": "establishing",
                 "prompt": "wide", "duration_s": 6.0},
            ]},
            {"scene_id": "s2", "beats": [
                {"beat_id": "b2", "kind": "motion",
                 "prompt": "move", "duration_s": 8.0},
            ]},
        ],
    }
    r = plan_episode(outline)
    # b2 is in scene_02; b1 is in scene_01; no in-scene upstream exists.
    motion_job = next(j for j in r.jobs if j.backend == "ltx_motion")
    assert motion_job.handoff_from is None


# ------------------------------------------------------------------
# Runtime coverage
# ------------------------------------------------------------------


def test_runtime_target_respected_when_beats_exceed():
    from otr_v2.hyworld.planner import plan_episode

    outline = _mini_outline(runtime_s=12.0)  # well under the 30s of beats
    r = plan_episode(outline)
    # We stop at the first beat boundary >= target.
    assert r.total_duration_s >= 12.0
    # And we don't massively overshoot (one beat's worth of overshoot max).
    assert r.total_duration_s < 12.0 + 15.0


def test_runtime_target_repeats_scenes_when_beats_are_short():
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "runtime_s": 60.0,
        "scenes": [{
            "scene_id": "s1",
            "beats": [
                {"beat_id": "b1", "kind": "establishing",
                 "prompt": "wide", "duration_s": 6.0},
            ],
        }],
    }
    r = plan_episode(outline)
    assert r.total_duration_s >= 60.0
    # Nudging must have fired since the same beat is used ~10 times.
    hashes = [j.prompt_hash for j in r.jobs]
    assert len(set(hashes)) > 1


def test_override_target_runtime():
    from otr_v2.hyworld.planner import plan_episode

    outline = _mini_outline(runtime_s=30.0)
    r = plan_episode(outline, target_runtime_s=10.0)
    assert r.target_runtime_s == 10.0
    assert r.total_duration_s >= 10.0


def test_empty_outline_returns_empty_timeline():
    from otr_v2.hyworld.planner import plan_episode

    r = plan_episode({"runtime_s": 10.0, "scenes": []})
    assert r.jobs == []
    assert any("zero scenes" in w for w in r.warnings)


# ------------------------------------------------------------------
# Shotlist JSON emission
# ------------------------------------------------------------------


def test_emit_shotlist_json_schema():
    from otr_v2.hyworld.planner import plan_episode, emit_shotlist_json

    r = plan_episode(_mini_outline(runtime_s=15.0))
    shotlist = emit_shotlist_json(r)

    assert "shots" in shotlist
    assert isinstance(shotlist["shots"], list)
    assert shotlist["job_count"] == len(r.jobs)
    assert "target_runtime_s" in shotlist
    assert "total_duration_s" in shotlist

    # Each shot entry has the minimum bridge-contract fields.
    for shot in shotlist["shots"]:
        assert "shot_id" in shot
        assert "backend" in shot
        assert "prompt" in shot
        assert "duration_s" in shot
        assert "prompt_hash" in shot


def test_write_shotlist_to_disk(tmp_path):
    from otr_v2.hyworld.planner import plan_episode, write_shotlist

    r = plan_episode(_mini_outline(runtime_s=10.0))
    out = tmp_path / "shotlist.json"
    write_shotlist(r, out)

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["job_count"] == len(r.jobs)


def test_coerce_outline_accepts_string_json():
    from otr_v2.hyworld.planner import plan_episode

    outline_str = json.dumps(_mini_outline(runtime_s=10.0))
    r = plan_episode(outline_str)
    assert len(r.jobs) > 0


def test_coerce_outline_accepts_path(tmp_path):
    from otr_v2.hyworld.planner import plan_episode

    p = tmp_path / "o.json"
    p.write_text(json.dumps(_mini_outline(runtime_s=10.0)), encoding="utf-8")
    r = plan_episode(p)
    assert len(r.jobs) > 0


# ------------------------------------------------------------------
# 3-minute dry run (Day 9 gate)
# ------------------------------------------------------------------


def test_3min_dry_run_clean():
    """The Day 9 gate: given a 180s outline, the planner must emit a
    clean, non-repeating job list that covers the full runtime."""
    from otr_v2.hyworld.planner import plan_episode

    outline = {
        "title": "SIGNAL LOST 3-MIN DRY RUN",
        "runtime_s": 180.0,
        "scenes": [
            {
                "scene_id": f"scene_{s:02d}",
                "location": ["Cockpit", "Corridor", "Engine Bay", "Bridge"][s % 4],
                "beats": [
                    {
                        "beat_id": f"s{s}_b{b}",
                        "kind": ["establishing", "close_up", "keyframe",
                                 "motion", "loop", "insert"][b % 6],
                        "prompt": f"scene {s} beat {b}",
                        "duration_s": [6.0, 4.0, 5.0, 8.0, 9.0, 3.0][b % 6],
                        "character": "BABA" if b % 6 == 1 else None,
                        "refs": ["baba.png"] if b % 6 == 1 else [],
                        "mask_prompt": "viewport" if b % 6 == 5 else None,
                        "insert_prompt": "HUD" if b % 6 == 5 else None,
                    }
                    for b in range(6)
                ],
            }
            for s in range(3)
        ],
    }

    r = plan_episode(outline)

    # Runtime coverage.
    assert r.total_duration_s >= 180.0
    # Scene diversity.
    scene_ids = {j.scene_id for j in r.jobs}
    assert len(scene_ids) >= 3
    # Non-repetition: no duplicate (backend, prompt_hash) inside the
    # default window.
    window = r.repetition_window
    for i in range(len(r.jobs) - window):
        slice_hashes = [
            (j.backend, j.prompt_hash)
            for j in r.jobs[i:i + window + 1]
        ]
        assert len(slice_hashes) == len(set(slice_hashes)), \
            f"non-repetition violated at index {i}: {slice_hashes}"
    # Backend diversity: at least four different backends used.
    backends_used = {j.backend for j in r.jobs}
    assert len(backends_used) >= 4


def test_all_emitted_backends_are_registered():
    """The planner may never emit a name outside the Day 1-7 registry."""
    from otr_v2.hyworld.planner import plan_episode
    from otr_v2.hyworld.backends import list_backends

    registered = set(list_backends())
    outline = _mini_outline(runtime_s=30.0)
    r = plan_episode(outline)
    for job in r.jobs:
        assert job.backend in registered, (
            f"planner emitted unregistered backend {job.backend!r} "
            f"for shot {job.shot_id!r}"
        )


# ------------------------------------------------------------------
# PlannerJob / PlannerResult dataclasses
# ------------------------------------------------------------------


def test_planner_job_to_dict_round_trip():
    from otr_v2.hyworld.planner import PlannerJob

    j = PlannerJob(
        shot_id="s", backend="flux_anchor", scene_id="sc1",
        prompt="p", duration_s=5.0, prompt_hash="abc",
    )
    d = j.to_dict()
    assert d["shot_id"] == "s"
    assert d["backend"] == "flux_anchor"
    assert d["prompt_hash"] == "abc"
    # Optional fields omitted when empty.
    assert "refs" not in d
    assert "handoff_from" not in d


def test_planner_result_to_dict_includes_diagnostics():
    from otr_v2.hyworld.planner import plan_episode

    r = plan_episode(_mini_outline(runtime_s=10.0))
    d = r.to_dict()
    assert d["job_count"] == len(r.jobs)
    assert "warnings" in d
    assert "target_runtime_s" in d
    assert "repetition_window" in d
