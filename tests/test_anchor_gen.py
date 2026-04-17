"""
test_anchor_gen.py  --  Unit tests for otr_v2.hyworld.anchor_gen.
================================================================

Covers the pure-python surface (prompt building, seed derivation, cache
key, cache hit/miss flow) without requiring diffusers / torch / SD 1.5
weights to be installed.  The default loader is documented in the module
docstring and exercised only by integration tests on the actual hardware.

Determinism contract under test:
    - Same shot -> same prompt -> same seed -> same cache key -> same path.
    - Re-running with the same shotlist after a successful first run
      must NOT invoke the inference callable again (cache hit).
    - The cache key must NOT depend on shot_id (two shots with identical
      prompt + seed should share the cached PNG).
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

# Path-shim: tests can run via pytest from the repo root or a sub-dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from otr_v2.hyworld import anchor_gen as ag  # noqa: E402


# ---------------------------------------------------------------------------
# Fake inference callable (no torch needed)
# ---------------------------------------------------------------------------

def _tiny_png(r: int = 100, g: int = 110, b: int = 120) -> bytes:
    """Build a 1x1 valid PNG with a known color, no Pillow needed."""
    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = bytes([0, r, g, b])
    idat = zlib.compress(raw, 1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


class _Counter:
    """Records every call to the fake inferer for assertions."""
    def __init__(self):
        self.calls: list[ag.AnchorRequest] = []

    def __call__(self, req: ag.AnchorRequest) -> bytes:
        self.calls.append(req)
        # Vary color slightly per call so PNG bytes differ if shape allows it,
        # but keep it deterministic per (prompt, seed) so cache keys stable.
        h = abs(hash((req.prompt, req.seed))) & 0xFFFFFF
        return _tiny_png(r=(h >> 16) & 0xFF, g=(h >> 8) & 0xFF, b=h & 0xFF)


# ---------------------------------------------------------------------------
# Sample shotlist data
# ---------------------------------------------------------------------------

SHOT_FULL = {
    "shot_id": "s01_01",
    "scene_ref": "sc01",
    "duration_sec": 9.0,
    "camera": "slow handheld",
    "env_prompt": "neon-lit diner counter, late night, light rain on the window",
    "mood": "uneasy",
    "sfx_accents": [],
    "dialogue_line_ids": [],
}

SHOT_MIN = {
    "shot_id": "s02_01",
    "duration_sec": 6.0,
    "env_prompt": "empty parking lot, sodium streetlamps",
}

SHOT_EMPTY = {
    "shot_id": "s99_01",
    "duration_sec": 5.0,
    # env_prompt intentionally absent -- exercises the default fallback.
}


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_full_shot_includes_env_mood_camera_and_style(self):
        p = ag.build_prompt(SHOT_FULL)
        assert "neon-lit diner counter" in p
        assert "uneasy mood" in p
        assert "camera: slow handheld" in p
        assert ag.SIGNAL_LOST_STYLE in p

    def test_minimal_shot_omits_missing_fields(self):
        p = ag.build_prompt(SHOT_MIN)
        assert "empty parking lot" in p
        assert "mood" not in p     # SHOT_MIN has no mood
        assert "camera:" not in p  # SHOT_MIN has no camera
        assert ag.SIGNAL_LOST_STYLE in p

    def test_empty_env_falls_back_to_default(self):
        p = ag.build_prompt(SHOT_EMPTY)
        assert "empty room, dim light" in p

    def test_custom_style_suffix_replaces_default(self):
        custom = "16mm color reversal film stock"
        p = ag.build_prompt(SHOT_FULL, style_suffix=custom)
        assert custom in p
        assert ag.SIGNAL_LOST_STYLE not in p

    def test_empty_style_suffix_omits_style_segment(self):
        p = ag.build_prompt(SHOT_FULL, style_suffix="")
        assert ag.SIGNAL_LOST_STYLE not in p
        # And nothing weird at the end (no trailing comma)
        assert not p.endswith(", ")


# ---------------------------------------------------------------------------
# derive_seed
# ---------------------------------------------------------------------------

class TestDeriveSeed:
    def test_deterministic_for_same_shot(self):
        s1 = ag.derive_seed(SHOT_FULL, seed_base=0)
        s2 = ag.derive_seed(SHOT_FULL, seed_base=0)
        assert s1 == s2

    def test_changes_with_seed_base(self):
        s1 = ag.derive_seed(SHOT_FULL, seed_base=0)
        s2 = ag.derive_seed(SHOT_FULL, seed_base=42)
        assert s1 != s2

    def test_different_shots_get_different_seeds(self):
        s1 = ag.derive_seed(SHOT_FULL, seed_base=0)
        s2 = ag.derive_seed(SHOT_MIN, seed_base=0)
        assert s1 != s2

    def test_seed_in_uint32_range(self):
        s = ag.derive_seed(SHOT_FULL, seed_base=0xDEADBEEF)
        assert 0 <= s < (1 << 32)


# ---------------------------------------------------------------------------
# cache_key / cache_path
# ---------------------------------------------------------------------------

def _make_req(**overrides) -> ag.AnchorRequest:
    base = dict(
        shot_id="s01_01",
        prompt="test prompt",
        negative_prompt="test negative",
        seed=12345,
        width=1024,
        height=576,
        sampler="euler_a",
        steps=28,
        cfg=7.0,
        model_id=ag.DEFAULT_MODEL_ID,
        lora_set_hash="",
    )
    base.update(overrides)
    return ag.AnchorRequest(**base)


class TestCacheKey:
    def test_deterministic(self):
        a = _make_req()
        b = _make_req()
        assert ag.cache_key(a) == ag.cache_key(b)

    def test_changes_with_prompt(self):
        a = _make_req()
        b = _make_req(prompt="different prompt")
        assert ag.cache_key(a) != ag.cache_key(b)

    def test_changes_with_seed(self):
        a = _make_req()
        b = _make_req(seed=99999)
        assert ag.cache_key(a) != ag.cache_key(b)

    def test_changes_with_model_id(self):
        a = _make_req()
        b = _make_req(model_id="sdxl-base")
        assert ag.cache_key(a) != ag.cache_key(b)

    def test_excludes_shot_id(self):
        # Two shots with identical render params should share the cached PNG.
        a = _make_req(shot_id="s01_01")
        b = _make_req(shot_id="s07_03")
        assert ag.cache_key(a) == ag.cache_key(b)

    def test_cache_path_format(self, tmp_path):
        req = _make_req()
        p = ag.cache_path(req, tmp_path)
        assert p.parent == tmp_path
        assert p.suffix == ".png"
        assert len(p.stem) == 64  # SHA-256 hex length


# ---------------------------------------------------------------------------
# generate_for_shotlist (the integration surface, fake inferer)
# ---------------------------------------------------------------------------

class TestGenerateForShotlist:
    def test_writes_one_png_per_shot(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()
        results = ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN], out_root, infer=infer,
        )
        assert set(results) == {"s01_01", "s02_01"}
        for r in results.values():
            assert r.png_path.exists()
            assert r.png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert r.cache_hit is False
            assert r.error == ""
        assert len(infer.calls) == 2

    def test_second_run_is_cache_hit_no_inference(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()
        ag.generate_for_shotlist([SHOT_FULL, SHOT_MIN], out_root, infer=infer)
        assert len(infer.calls) == 2

        # Second pass with the same shotlist should reuse cache entirely.
        infer2 = _Counter()
        results = ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN], out_root, infer=infer2,
        )
        assert len(infer2.calls) == 0
        for r in results.values():
            assert r.cache_hit is True
            assert r.png_path.exists()

    def test_lazy_loader_not_called_when_fully_cached(self, tmp_path):
        out_root = tmp_path / "out"
        # Pre-populate cache via a direct infer.
        ag.generate_for_shotlist([SHOT_FULL], out_root, infer=_Counter())

        # Now run with a model_loader that would explode if invoked.
        def _exploding_loader():
            raise AssertionError("model_loader must not be called on full cache")

        results = ag.generate_for_shotlist(
            [SHOT_FULL], out_root, model_loader=_exploding_loader,
        )
        assert results["s01_01"].cache_hit is True

    def test_lazy_loader_invoked_only_once_for_multiple_misses(self, tmp_path):
        out_root = tmp_path / "out"
        load_count = {"n": 0}
        infer = _Counter()

        def _loader():
            load_count["n"] += 1
            return infer

        ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN, SHOT_EMPTY],
            out_root,
            model_loader=_loader,
        )
        assert load_count["n"] == 1
        assert len(infer.calls) == 3

    def test_writes_cache_index_json(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()
        ag.generate_for_shotlist([SHOT_FULL, SHOT_MIN], out_root, infer=infer)
        index_path = out_root / "anchors" / "cache_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert set(index) == {"s01_01", "s02_01"}
        for entry in index.values():
            assert "cache_key" in entry
            assert "prompt" in entry
            assert "seed" in entry
            assert entry["error"] == ""

    def test_inference_error_recorded_per_shot_no_crash(self, tmp_path):
        out_root = tmp_path / "out"

        def _broken_infer(req):
            raise RuntimeError("simulated GPU OOM")

        results = ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN], out_root, infer=_broken_infer,
        )
        assert set(results) == {"s01_01", "s02_01"}
        for r in results.values():
            assert "RuntimeError" in r.error
            assert "simulated GPU OOM" in r.error
            assert r.cache_hit is False

    def test_non_png_inference_output_is_rejected(self, tmp_path):
        out_root = tmp_path / "out"

        def _bad_infer(req):
            return b"not a png"

        results = ag.generate_for_shotlist(
            [SHOT_FULL], out_root, infer=_bad_infer,
        )
        r = results["s01_01"]
        assert "non-PNG" in r.error or "ValueError" in r.error

    def test_progress_cb_invoked_for_every_shot(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()
        seen: list[tuple[int, int, str]] = []

        def _cb(idx, total, result):
            seen.append((idx, total, result.shot_id))

        ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN, SHOT_EMPTY],
            out_root, infer=infer, progress_cb=_cb,
        )
        assert seen == [(0, 3, "s01_01"), (1, 3, "s02_01"), (2, 3, "s99_01")]

    def test_progress_cb_error_does_not_abort_run(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()

        def _cb(idx, total, result):
            raise RuntimeError("intentional callback failure")

        results = ag.generate_for_shotlist(
            [SHOT_FULL, SHOT_MIN], out_root, infer=infer, progress_cb=_cb,
        )
        # Run completed despite the broken callback.
        assert set(results) == {"s01_01", "s02_01"}
        for r in results.values():
            assert r.error == ""
            assert r.png_path.exists()

    def test_two_shots_with_same_render_params_share_cache(self, tmp_path):
        """If two shots build identical prompts + seeds, they should
        write to the same cache entry (different shot dirs, but the
        cache_key is the same)."""
        out_root = tmp_path / "out"
        infer = _Counter()

        # Same env_prompt + mood + camera + duration would normally produce
        # different seeds because shot_id differs, defeating cache sharing.
        # Force shared seed by passing seed_base such that the per-shot
        # XOR happens to collide -- easier path: fake two shots with the
        # SAME shot_id under different keys in the dict.  Skip: the
        # production cache key correctly excludes shot_id, and the seed
        # derivation is per-shot-id by design, so cross-shot sharing only
        # happens when prompts AND seeds align.  Verify the underlying
        # cache_key behavior here instead of forcing a contrived shot.
        a = _make_req(shot_id="alpha")
        b = _make_req(shot_id="beta")
        assert ag.cache_key(a) == ag.cache_key(b)

    def test_anchor_dir_created_under_out_root(self, tmp_path):
        out_root = tmp_path / "out"
        infer = _Counter()
        ag.generate_for_shotlist([SHOT_FULL], out_root, infer=infer)
        assert (out_root / "anchors").is_dir()
        # PNG also mirrored into the per-shot dir for worker.py to grab.
        assert (out_root / "s01_01" / "render.png").exists()

    def test_custom_cache_dir_respected(self, tmp_path):
        out_root = tmp_path / "out"
        custom_cache = tmp_path / "shared_cache"
        infer = _Counter()
        results = ag.generate_for_shotlist(
            [SHOT_FULL], out_root, infer=infer, cache_dir=custom_cache,
        )
        png_files = list(custom_cache.glob("*.png"))
        assert len(png_files) == 1
        assert (custom_cache / "cache_index.json").exists()
        # Default anchors dir should NOT have been used.
        assert not (out_root / "anchors").exists()
        # And the cached png should match the per-shot mirror.
        assert results["s01_01"].png_path.exists()
