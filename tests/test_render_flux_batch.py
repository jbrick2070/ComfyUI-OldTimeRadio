"""
test_render_flux_batch.py
==========================

Unit tests for scripts/render_flux_batch.py.

Validates the target-list builder + filter logic + prompt graph schema
against a synthetic L2 ledger. No network, no ComfyUI, no FLUX weights —
the orchestrator's HTTP layer is exercised separately when the live
render runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _l2_ledger() -> dict:
    """L2 ledger: 2 cast, 1 scene, 2 shots — shot 1 has speaker repeat
    (ALICE→BOB→ALICE), shot 2 has BOB only. Total unique (shot, speaker)
    pairs = 3."""
    return {
        "schema_version": "l2-2026-04-25",
        "episode_id": "test_ep",
        "cast": [
            {"char_id": "c01", "name": "ALICE",
             "description": "young scientist, dark hair"},
            {"char_id": "c02", "name": "BOB",
             "description": "weathered captain, grey beard"},
        ],
        "scenes": [
            {"scene_id": "scene_lab", "description": "research lab, clean white panels"},
        ],
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_lab",
                "dur_s": 12.0,
                "speakers": ["ALICE", "BOB"],
                "beats": [
                    {"beat_id": "shot_001_b1", "speaker": "ALICE",
                     "line_ids": ["l1"], "dur_s": 4.0, "start_s": 0.0},
                    {"beat_id": "shot_001_b2", "speaker": "BOB",
                     "line_ids": ["l2"], "dur_s": 4.0, "start_s": 4.0},
                    {"beat_id": "shot_001_b3", "speaker": "ALICE",
                     "line_ids": ["l3"], "dur_s": 4.0, "start_s": 8.0},
                ],
            },
            {
                "shot_id": "shot_002",
                "scene_id": "scene_lab",
                "dur_s": 5.0,
                "speakers": ["BOB"],
                "beats": [
                    {"beat_id": "shot_002_b1", "speaker": "BOB",
                     "line_ids": ["l4"], "dur_s": 5.0, "start_s": 12.0},
                ],
            },
        ],
        "lines": [],
    }


# ---------------------------------------------------------------------------
# Module imports cleanly
# ---------------------------------------------------------------------------


def test_module_imports():
    import render_flux_batch  # noqa: F401
    assert hasattr(render_flux_batch, "build_target_list")
    assert hasattr(render_flux_batch, "filter_targets")
    assert hasattr(render_flux_batch, "build_flux_prompt")
    assert hasattr(render_flux_batch, "ComfyClient")
    assert hasattr(render_flux_batch, "slugify")


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


def test_slugify_basic():
    from render_flux_batch import slugify
    assert slugify("ALICE") == "alice"
    assert slugify("Kenji Cross") == "kenji_cross"
    assert slugify("ANN — narrator") == "ann_narrator"


def test_slugify_empty_returns_unnamed():
    from render_flux_batch import slugify
    assert slugify("") == "unnamed"
    assert slugify(None) == "unnamed"
    assert slugify("   ") == "unnamed"


# ---------------------------------------------------------------------------
# Target list — counts and shape
# ---------------------------------------------------------------------------


def test_target_list_emits_one_portrait_per_cast():
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="cinematic test")
    portraits = [t for t in targets if t["kind"] == "portrait"]
    assert len(portraits) == 2
    speakers = {t["speaker"] for t in portraits}
    assert speakers == {"ALICE", "BOB"}


def test_target_list_dedupes_shot_speaker_pairs():
    """ALICE in shot_001 has 2 beats (b1 + b3 after BOB) — emits ONE
    composite, not two. Same speaker in same shot reuses one composite
    per the per-(shot, speaker) policy."""
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="x")
    composites = [t for t in targets if t["kind"] == "composite"]
    # Unique pairs: (shot_001, ALICE), (shot_001, BOB), (shot_002, BOB) = 3
    assert len(composites) == 3
    keys = {(t["shot_id"], t["speaker"]) for t in composites}
    assert keys == {("shot_001", "ALICE"),
                    ("shot_001", "BOB"),
                    ("shot_002", "BOB")}


def test_target_list_total_count_matches_design():
    """Total = N_cast + N_unique_(shot, speaker) pairs."""
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="x")
    # 2 cast + 3 unique (shot, speaker) = 5
    assert len(targets) == 5


def test_target_list_each_target_has_required_fields():
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="x")
    required = {"kind", "target_id", "speaker", "positive_prompt",
                "save_prefix", "out_filename_glob"}
    for t in targets:
        missing = required - set(t.keys())
        assert not missing, f"target {t.get('target_id')} missing {missing}"


def test_portrait_prompt_contains_speaker_description():
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="cinematic")
    alice_portrait = next(t for t in targets
                          if t["kind"] == "portrait" and t["speaker"] == "ALICE")
    # cast[].description carried into the prompt
    assert "young scientist, dark hair" in alice_portrait["positive_prompt"]
    assert "cinematic" in alice_portrait["positive_prompt"]


def test_composite_prompt_combines_speaker_and_scene():
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="STYLE_TAG")
    bob_in_shot1 = next(t for t in targets
                        if t["kind"] == "composite"
                        and t["shot_id"] == "shot_001"
                        and t["speaker"] == "BOB")
    p = bob_in_shot1["positive_prompt"]
    assert "weathered captain" in p             # cast description
    assert "research lab" in p                   # scene description
    assert "STYLE_TAG" in p                       # style tail


def test_save_prefix_uses_humo_naming_convention():
    """Output filenames follow the existing on-disk convention:
       portraits → otr_humo_pass1_portrait_<speaker>
       composites → otr_humo_pass3_<shot>_<speaker>
    so the HuMo orchestrator's find_portrait_for_speaker glob still hits."""
    from render_flux_batch import build_target_list
    targets = build_target_list(_l2_ledger(), style_tail="x")
    for t in targets:
        if t["kind"] == "portrait":
            assert t["save_prefix"].startswith("otr_humo_pass1_portrait_")
        elif t["kind"] == "composite":
            assert t["save_prefix"].startswith("otr_humo_pass3_")
            assert t["speaker"].lower().replace(" ", "_") in t["save_prefix"].lower()


# ---------------------------------------------------------------------------
# Scope filters
# ---------------------------------------------------------------------------


def test_filter_all_returns_everything():
    from render_flux_batch import build_target_list, filter_targets
    targets = build_target_list(_l2_ledger(), style_tail="x")
    assert filter_targets(targets, "all") == targets


def test_filter_portraits_only():
    from render_flux_batch import build_target_list, filter_targets
    targets = build_target_list(_l2_ledger(), style_tail="x")
    out = filter_targets(targets, "portraits-only")
    assert all(t["kind"] == "portrait" for t in out)
    assert len(out) == 2


def test_filter_composites_only():
    from render_flux_batch import build_target_list, filter_targets
    targets = build_target_list(_l2_ledger(), style_tail="x")
    out = filter_targets(targets, "composites-only")
    assert all(t["kind"] == "composite" for t in out)
    assert len(out) == 3


def test_filter_custom_by_shot():
    from render_flux_batch import build_target_list, filter_targets
    targets = build_target_list(_l2_ledger(), style_tail="x")
    out = filter_targets(targets, "custom:shot_id=shot_002")
    # Only the composite for (shot_002, BOB)
    assert len(out) == 1
    assert out[0]["shot_id"] == "shot_002"
    assert out[0]["speaker"] == "BOB"


def test_filter_custom_by_speaker():
    from render_flux_batch import build_target_list, filter_targets
    targets = build_target_list(_l2_ledger(), style_tail="x")
    out = filter_targets(targets, "custom:speaker=ALICE")
    # ALICE's portrait + (shot_001, ALICE) composite = 2
    assert len(out) == 2
    assert {t["kind"] for t in out} == {"portrait", "composite"}


def test_filter_unknown_scope_raises():
    from render_flux_batch import filter_targets
    with pytest.raises(ValueError):
        filter_targets([], "unknown-scope")


# ---------------------------------------------------------------------------
# FLUX prompt graph
# ---------------------------------------------------------------------------


def test_flux_prompt_has_required_nodes():
    from render_flux_batch import build_flux_prompt
    p = build_flux_prompt(
        checkpoint="flux1-dev-fp8.safetensors",
        positive="hello",
        negative="bad",
        save_prefix="test",
        seed=42,
        width=1024,
        height=1024,
        steps=20,
        cfg=3.5,
    )
    classes = {node["class_type"] for node in p.values()}
    assert "CheckpointLoaderSimple" in classes
    assert "CLIPTextEncode" in classes  # both pos + neg
    assert "EmptyLatentImage" in classes
    assert "KSampler" in classes
    assert "VAEDecode" in classes
    assert "SaveImage" in classes


def test_flux_prompt_seed_deterministic():
    from render_flux_batch import build_flux_prompt
    p1 = build_flux_prompt(checkpoint="x", positive="a", negative="b",
                            save_prefix="s", seed=12345, width=1024,
                            height=1024, steps=20, cfg=3.5)
    p2 = build_flux_prompt(checkpoint="x", positive="a", negative="b",
                            save_prefix="s", seed=12345, width=1024,
                            height=1024, steps=20, cfg=3.5)
    assert p1["5"]["inputs"]["seed"] == p2["5"]["inputs"]["seed"] == 12345


def test_flux_prompt_save_prefix_lands_in_save_node():
    from render_flux_batch import build_flux_prompt
    p = build_flux_prompt(checkpoint="x", positive="a", negative="b",
                          save_prefix="my_prefix", seed=1, width=1024,
                          height=1024, steps=20, cfg=3.5)
    # Find the SaveImage node and verify
    save_node = next(n for n in p.values() if n["class_type"] == "SaveImage")
    assert save_node["inputs"]["filename_prefix"] == "my_prefix"


# ---------------------------------------------------------------------------
# Astrotech-shape integration (no real ledger needed, simulate counts)
# ---------------------------------------------------------------------------


def test_target_count_matches_astrotech_design():
    """Build a 6-cast / 23-shot / 46-beat / 44-unique-pair ledger shape
    (matches actual astrotech ledger) and confirm the orchestrator emits
    6 portraits + 44 composites = 50 total."""
    from render_flux_batch import build_target_list

    cast_names = ["ANN", "ANNOUNCER", "MEREDITH", "RYAN", "SEAN", "VANCE"]
    cast = [{"char_id": f"c{i + 1:02d}", "name": n, "description": f"desc_{n}"}
            for i, n in enumerate(cast_names)]
    scenes = [{"scene_id": "scene_facility", "description": "AstroTech facility"}]

    # Hand-rolled shot/beat counts approximating astrotech (43 unique pairs).
    shot_specs = [
        ("shot_001", ["ANNOUNCER", "ANN"]),         # 2 unique
        ("shot_002", ["MEREDITH"]),                  # 1
        ("shot_003", ["VANCE", "MEREDITH"]),         # 2
        ("shot_004", ["SEAN", "RYAN", "MEREDITH"]),  # 3
        ("shot_005", ["VANCE", "MEREDITH"]),         # 2
        ("shot_006", ["RYAN"]),                       # 1
        ("shot_007", ["MEREDITH", "RYAN"]),           # 2
        ("shot_008", ["MEREDITH"]),                   # 1
        ("shot_009", ["VANCE", "RYAN"]),              # 2
        ("shot_010", ["MEREDITH", "ANN"]),            # 2
        ("shot_011", ["SEAN", "MEREDITH", "SEAN"]),   # 2 (SEAN dedup)
        ("shot_012", ["VANCE", "SEAN", "ANN"]),       # 3
        ("shot_013", ["MEREDITH", "RYAN"]),           # 2
        ("shot_014", ["MEREDITH", "VANCE"]),          # 2
        ("shot_015", ["SEAN"]),                       # 1
        ("shot_016", ["MEREDITH", "VANCE"]),          # 2
        ("shot_017", ["MEREDITH", "RYAN"]),           # 2
        ("shot_018", ["MEREDITH", "SEAN", "MEREDITH"]),  # 2 (MEREDITH dedup)
        ("shot_019", ["SEAN", "VANCE"]),              # 2
        ("shot_020", ["MEREDITH"]),                   # 1
        ("shot_021", ["RYAN", "MEREDITH"]),           # 2
        ("shot_022", ["SEAN", "ANN", "MEREDITH"]),    # 3
        ("shot_023", ["RYAN", "ANNOUNCER"]),          # 2
    ]
    shots = []
    for shot_id, beat_speakers in shot_specs:
        beats = []
        for i, sp in enumerate(beat_speakers):
            beats.append({"beat_id": f"{shot_id}_b{i + 1}", "speaker": sp,
                          "line_ids": [], "start_s": 0.0, "dur_s": 5.0})
        shots.append({"shot_id": shot_id, "scene_id": "scene_facility",
                      "dur_s": len(beat_speakers) * 5.0,
                      "speakers": list(dict.fromkeys(beat_speakers)),
                      "beats": beats})
    led = {"schema_version": "l2-2026-04-25", "episode_id": "astrotech_shape",
           "cast": cast, "scenes": scenes, "shots": shots, "lines": []}

    targets = build_target_list(led, style_tail="x")
    n_p = sum(1 for t in targets if t["kind"] == "portrait")
    n_c = sum(1 for t in targets if t["kind"] == "composite")
    assert n_p == 6
    assert n_c == 44
    assert len(targets) == 50  # 6 portraits + 44 (shot, speaker) pairs
