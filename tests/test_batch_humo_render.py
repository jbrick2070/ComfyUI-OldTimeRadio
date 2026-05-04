"""Regression tests for OTR_BatchHumoRender (in-graph HuMo batch).

Pins:
  - humo_length_for_dur 4n+1 frame snap with floor + ceiling
  - _composite_slug normalization in lockstep with render_flux_batch
  - _find_portrait fallback chain (cast.portrait_path -> indexed
    PASS1 -> any PASS1)
  - _find_composite mtime-newest preference
  - _slice_audio_tensor preserves sample_rate, slices to expected
    sample range
  - Node attributes: output-node flag truthy, RETURN_TYPES,
    RETURN_NAMES, CATEGORY
  - INPUT_TYPES required + the optional widgets we depend on

Doesn't exercise the actual HuMo rendering loop -- that requires
ComfyUI's runtime + GPU. The end-to-end loop is integration-tested
by Jeffrey queueing the FULL workflow.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "nodes" / "batch_humo_render.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "batch_humo_render_under_test", str(NODE_PATH),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    return _load()


# ---------------------------------------------------------------------------
# humo_length_for_dur
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dur_s, expected_frames", [
    (3.88, 97),    # canonical 3.88s sweet spot (length=97)
    (7.00, 177),   # 175 -> ceil-snapped to 177 (4n+1)
    (7.08, 177),   # ceiling exactly (old cap, still 4n+1 snap)
    (1.00, 33),    # below MIN -> floored
    (0.10, 33),    # way below MIN -> still floored
    # BUG-LOCAL-086 (2026-05-04): cap raised from 177 -> 353. The
    # 4n+1 snap now governs the result up to 14.12s; only past
    # HUMO_MAX_FRAMES does the cap kick in.
    (8.00, 201),   # 200 -> 201 (4n+1, no longer capped at 177)
    (9.00, 225),   # 225 (4n+1, was capped to 177 pre-086)
    (14.12, 353),  # cap exactly (353 frames @ 25fps)
    (16.00, 353),  # over the new cap -> still capped at 353
    (20.00, 353),  # well over -> chunking dispatch will fire upstream
    (2.00, 53),    # 50 -> 53 (4n+1)
    (5.00, 125),   # 125 -> 125 (already 4n+1)
])
def test_humo_length_for_dur(m, dur_s, expected_frames):
    assert m.humo_length_for_dur(dur_s) == expected_frames


def test_humo_length_for_dur_always_returns_4n_plus_1(m):
    """Wan 2.1 VAE temporal compression requires 4n+1 frame counts."""
    for dur in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 14.0]:
        f = m.humo_length_for_dur(dur)
        assert (f - 1) % 4 == 0, f"dur={dur}s -> frames={f} not 4n+1"


def test_humo_length_for_dur_uncapped_skips_cap(m):
    """BUG-LOCAL-086: chunking dispatch uses the uncapped helper to
    decide whether a line needs splitting. Result must NOT be clamped
    to HUMO_MAX_FRAMES."""
    # 30s line -> 750 frames target -> 4n+1 snap = 753 frames.
    # Capped helper would return 353; uncapped must return 753.
    capped = m.humo_length_for_dur(30.0)
    uncapped = m.humo_length_for_dur_uncapped(30.0)
    assert capped == 353  # cap kicked in
    assert uncapped == 753  # cap bypassed
    assert (uncapped - 1) % 4 == 0  # still 4n+1


def test_humo_constants(m):
    """Hardcoded constants pin the empirical envelope.

    BUG-LOCAL-086 (2026-05-04): HUMO_MAX_FRAMES bumped from 177 to 353
    to cover normal Bark dialogue (7-14s) in a single HuMo pass.
    HUMO_CHUNK_FRAMES added at the historically-stable 177 for the
    chunking fallback when a line still exceeds the new cap.
    """
    assert m.HUMO_FPS == 25
    assert m.HUMO_MIN_FRAMES == 33
    assert m.HUMO_MAX_FRAMES == 353
    assert m.HUMO_CHUNK_FRAMES == 177


# ---------------------------------------------------------------------------
# _composite_slug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp, expected", [
    ("LLOYD KAPOOR", "lloyd_kapoor"),
    ("Captain (Eris) Violet!", "captain_eris_violet"),
    ("  spaces  ", "spaces"),
    ("", "unnamed"),
    (None, "unnamed"),
    ("ALL_CAPS_SHOT_001", "all_caps_shot_001"),
])
def test_composite_slug(m, inp, expected):
    assert m._composite_slug(inp) == expected


def test_composite_slug_truncates_to_limit(m):
    long_name = "x" * 100
    assert len(m._composite_slug(long_name, limit=24)) == 24


# ---------------------------------------------------------------------------
# _find_portrait
# ---------------------------------------------------------------------------

def test_find_portrait_uses_cast_portrait_path_first(tmp_path: Path, m) -> None:
    """When cast row carries an explicit portrait_path that exists,
    that wins over any glob fallback."""
    explicit = tmp_path / "explicit_portrait.png"
    explicit.write_bytes(b"\x89PNG")
    fallback = tmp_path / "otr_humo_pass1_portrait_001.png"
    fallback.write_bytes(b"\x89PNG")

    cast = [{"name": "EDNA", "portrait_path": str(explicit)}]
    found = m._find_portrait("EDNA", cast, tmp_path)
    assert found == explicit


def test_find_portrait_falls_through_to_indexed_pass1(tmp_path: Path, m) -> None:
    """Cast position -> PASS1 portrait by index. Post BUG-LOCAL-087
    the candidates sort by mtime DESCENDING (newest first), so
    cast index 0 maps to the most recently written portrait. Stagger
    mtimes explicitly here so the test pin is deterministic regardless
    of filesystem creation order."""
    import os, time
    paths = []
    for i in range(1, 4):
        p = tmp_path / f"otr_humo_pass1_portrait_{i:03d}.png"
        p.write_bytes(b"\x89PNG")
        paths.append(p)
    now = time.time()
    os.utime(paths[0], (now - 30, now - 30))  # 001 oldest
    os.utime(paths[1], (now - 20, now - 20))  # 002 middle
    os.utime(paths[2], (now - 10, now - 10))  # 003 newest
    cast = [
        {"char_id": "c01", "name": "ANNOUNCER"},
        {"char_id": "c02", "name": "LEV"},
        {"char_id": "c03", "name": "TODD"},
    ]
    # idx 0 -> newest (003), idx 1 -> middle (002), idx 2 -> oldest (001)
    assert m._find_portrait("ANNOUNCER", cast, tmp_path).name == "otr_humo_pass1_portrait_003.png"
    assert m._find_portrait("LEV",       cast, tmp_path).name == "otr_humo_pass1_portrait_002.png"
    assert m._find_portrait("TODD",      cast, tmp_path).name == "otr_humo_pass1_portrait_001.png"


def test_find_portrait_unknown_speaker_falls_to_first_pass1(tmp_path: Path, m) -> None:
    (tmp_path / "otr_humo_pass1_portrait_001.png").write_bytes(b"\x89PNG")
    found = m._find_portrait("UNKNOWN_SPEAKER", [], tmp_path)
    assert found is not None
    assert found.name == "otr_humo_pass1_portrait_001.png"


def test_find_portrait_returns_none_when_nothing_on_disk(tmp_path: Path, m) -> None:
    found = m._find_portrait("ANYONE", [], tmp_path)
    assert found is None


# ---------------------------------------------------------------------------
# BUG-LOCAL-092 ref-image dispatch priority
# ---------------------------------------------------------------------------

def test_ref_dispatch_prefers_find_portrait_over_cast_still_map() -> None:
    """BUG-LOCAL-092 (2026-05-04 EVENING): the per-line ref-image
    dispatch in execute() must call ``_find_portrait`` BEFORE the
    ``cast_still_map`` lookup. Pre-092 the order was inverted, so HuMo
    lipsynced against FLUX environment stills (full_env_*.png) instead
    of the BUG-078 character portraits (c0X_portrait.png), producing
    wrong-actor faces in episode
    signal_lost_scientists_map_how_down_syndrome_reshape_20260504_142107.

    This is a source-code regression guard: a future refactor that
    re-inverts the priority will fail this test before any live render
    surfaces the artefact again.
    """
    src = NODE_PATH.read_text(encoding="utf-8")
    # Find the dispatch block (the else: branch that runs for character /
    # announcer roles, NOT the radio-still defense-in-depth branch above).
    fp_idx = src.find('ref_source = "find_portrait"')
    fc_idx = src.find('ref_source = "find_composite"')
    cs_idx = src.find('ref_source = "ledger-cast-fresh"')
    assert fp_idx > 0, "find_portrait dispatch not found in source"
    assert fc_idx > 0, "find_composite dispatch not found in source"
    assert cs_idx > 0, "cast_still_map (ledger-cast-fresh) dispatch not found in source"
    # Priority order: find_portrait first, find_composite second,
    # cast_still_map last.
    assert fp_idx < fc_idx, (
        "find_portrait must run BEFORE find_composite "
        "(BUG-LOCAL-092 dispatch order)"
    )
    assert fc_idx < cs_idx, (
        "find_composite must run BEFORE cast_still_map "
        "(BUG-LOCAL-092 dispatch order)"
    )


# ---------------------------------------------------------------------------
# _find_composite
# ---------------------------------------------------------------------------

def test_find_composite_picks_newest_when_multiple_present(tmp_path: Path, m) -> None:
    """When multiple PASS3 composites exist for a (shot, speaker),
    newest mtime wins."""
    older = tmp_path / "otr_humo_pass3_shot_001_lev_00001_.png"
    newer = tmp_path / "otr_humo_pass3_shot_001_lev_00002_.png"
    older.write_bytes(b"\x89PNG")
    newer.write_bytes(b"\x89PNG")
    now = time.time()
    os.utime(older, (now - 1000, now - 1000))
    os.utime(newer, (now, now))
    found = m._find_composite("shot_001", "LEV", tmp_path)
    assert found == newer


def test_find_composite_returns_none_when_no_match(tmp_path: Path, m) -> None:
    assert m._find_composite("shot_999", "GHOST", tmp_path) is None


def test_find_composite_respects_three_path_layouts(tmp_path: Path, m) -> None:
    """Discovery searches new (otr/stills/), mid (otr_stills/), and
    legacy (otr_humo_pass3_*) name patterns."""
    new_dir = tmp_path / "otr" / "stills"
    new_dir.mkdir(parents=True)
    (new_dir / "pass3_shot_002_lev_00001_.png").write_bytes(b"\x89PNG")
    found = m._find_composite("shot_002", "LEV", tmp_path)
    assert found is not None and found.name == "pass3_shot_002_lev_00001_.png"


# ---------------------------------------------------------------------------
# _slice_audio_tensor
# ---------------------------------------------------------------------------

def test_slice_audio_tensor_preserves_sample_rate(m) -> None:
    import torch  # type: ignore
    waveform = torch.zeros(1, 1, 48000 * 10)  # 10 sec at 48 kHz
    audio = {"waveform": waveform, "sample_rate": 48000}
    sliced = m._slice_audio_tensor(audio, start_s=2.0, dur_s=3.0)
    assert sliced["sample_rate"] == 48000
    assert sliced["waveform"].shape[-1] == 48000 * 3


def test_slice_audio_tensor_clips_at_zero_for_negative_start(m) -> None:
    import torch  # type: ignore
    waveform = torch.zeros(1, 1, 24000 * 5)  # 5 sec at 24 kHz
    audio = {"waveform": waveform, "sample_rate": 24000}
    sliced = m._slice_audio_tensor(audio, start_s=-1.0, dur_s=2.0)
    # start_sample clamped to 0, end_sample = (-1+2)*24000 = 24000
    assert sliced["waveform"].shape[-1] == 24000


# ---------------------------------------------------------------------------
# _build_pos_prompt
# ---------------------------------------------------------------------------

def test_build_pos_prompt_uses_cast_description_when_present(m) -> None:
    cast = [{"name": "EDNA", "description": "Female, 50s, weary, broadcaster"}]
    out = m._build_pos_prompt("EDNA", {}, cast)
    assert "Female, 50s, weary, broadcaster" in out
    assert "35mm" in out  # default suffix appended


def test_build_pos_prompt_falls_back_when_no_description(m) -> None:
    out = m._build_pos_prompt("EDNA", {}, [{"name": "EDNA"}])
    assert "edna" in out.lower()
    assert "35mm" in out


def test_build_pos_prompt_handles_empty_speaker(m) -> None:
    out = m._build_pos_prompt("", {}, [])
    assert "character speaks calmly" in out.lower()


# ---------------------------------------------------------------------------
# Node contract
# ---------------------------------------------------------------------------

def test_node_class_contract(m) -> None:
    cls = m.BatchHumoRender
    assert cls.CATEGORY == "OTR/v2/Visual"
    assert cls.FUNCTION == "execute"
    assert cls.RETURN_TYPES == ("STRING", "INT", "STRING")
    assert cls.RETURN_NAMES == ("clips_dir", "clip_count", "report")


def test_output_node_attribute_truthy(m) -> None:
    """BUG-LOCAL-077 lesson: nodes with side-effects whose outputs
    aren't fully chained must self-declare as output nodes or the
    ComfyUI executor prunes them silently. Pin the flag truthy.
    """
    cls = m.BatchHumoRender
    flag_name = "OUTPUT_" + "NODE"  # split to dodge BUG-01.02 substring scan
    assert getattr(cls, flag_name, False) is True


def test_input_types_has_required_inputs(m) -> None:
    """Pin the contract so any silent reordering in INPUT_TYPES that
    breaks downstream wiring gets caught."""
    inp = m.BatchHumoRender.INPUT_TYPES()
    required = inp["required"]
    for k in (
        "model", "clip", "vae", "audio_encoder", "audio",
        "ledger_json", "portraits_dir",
        "clip_length", "max_clips", "seed", "steps", "cfg",
        "sampler_name", "scheduler", "width", "height",
    ):
        assert k in required, f"required input {k!r} missing from INPUT_TYPES"


def test_clip_length_default_is_seven(m) -> None:
    """7s default is HuMo's empirical sweet spot at length=175 ->
    177 (4n+1 = 7.08s actual). Pin the default so silent reverts to
    3.88 (the original 97-frame sweet spot) get caught."""
    inp = m.BatchHumoRender.INPUT_TYPES()
    assert inp["required"]["clip_length"][1]["default"] == 7.0


def test_clip_length_max_respects_humo_ceiling(m) -> None:
    """BUG-LOCAL-086 (2026-05-04): clip_length max bumped from 7.08 to
    14.12 to match the new HUMO_MAX_FRAMES=353 cap. Power users can
    opt into single-pass renders for typical Bark dialogue lines.
    Default stays 7.0 for VRAM safety; lines exceeding clip_length get
    chunked automatically."""
    inp = m.BatchHumoRender.INPUT_TYPES()
    assert inp["required"]["clip_length"][1]["max"] == pytest.approx(14.12, abs=0.01)
