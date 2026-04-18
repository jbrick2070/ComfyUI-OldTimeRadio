"""
tests/test_character_regression.py  --  Day 12 character identity gate
=======================================================================

Regression suite for ``otr_v2.hyworld.character_regression``.  Exercises
the ROADMAP Day 12 bar:

    SSIM > 0.85 on cropped faces between Scene 1 and Scene 3

The suite is torch-free and runs the real pulid_portrait stub backend
under OTR_PULID_STUB=1 to produce deterministic solid-color PNGs so
that:

    same character + same refs -> SSIM = 1.0 (identity lock holds)
    same character + drifted refs -> SSIM drops below gate (failure surfaces)
    different characters        -> SSIM falls well below gate

No real PuLID-FLUX weights touched; no GPU required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from otr_v2.hyworld import character_regression as cr
from otr_v2.hyworld.backends import pulid_portrait as pp


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _write_shotlist(job_dir: Path, shots: list[dict]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "shotlist.json").write_text(
        json.dumps({"shots": shots}), encoding="utf-8",
    )


def _canary_root(base: Path) -> Path:
    """Create an otr_root/ with io/hyworld_{in,out} ready to use."""
    (base / "io" / "hyworld_in").mkdir(parents=True, exist_ok=True)
    (base / "io" / "hyworld_out").mkdir(parents=True, exist_ok=True)
    return base


def _job_dir_for(otr_root: Path, job_id: str) -> Path:
    return otr_root / "io" / "hyworld_in" / job_id


def _out_dir_for(otr_root: Path, job_id: str) -> Path:
    return otr_root / "io" / "hyworld_out" / job_id


def _populate_scene(
    scene_id: str,
    otr_root: Path,
    shots: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Render one scene's pulid_portrait shots into otr_root/io/hyworld_out/<scene_id>/.

    Each meta.json is patched post-render to record the ``scene_id`` so
    :func:`character_regression.find_portraits` can walk it.  Returns
    the per-scene out_dir used by find_portraits.
    """
    monkeypatch.setenv("OTR_PULID_STUB", "1")
    _canary_root(otr_root)
    scene_job_dir = _job_dir_for(otr_root, scene_id)
    scene_out_dir = _out_dir_for(otr_root, scene_id)
    _write_shotlist(scene_job_dir, shots)

    # Backend takes no args; derives out_dir from job_dir's parent tree.
    backend = pp.PulidPortraitBackend()
    backend.run(scene_job_dir)

    # Stamp scene_id into every per-shot meta.json so the regression
    # walker can read it back.
    for shot in shots:
        shot_id = shot["shot_id"]
        meta_path = scene_out_dir / shot_id / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["scene_id"] = scene_id
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    return scene_out_dir


# ---------------------------------------------------------------------
# module contract
# ---------------------------------------------------------------------


def test_module_imports_torch_free(tmp_path: Path) -> None:
    """character_regression must be importable with no torch dependency."""
    script = textwrap.dedent(
        """
        import sys
        # Pretend torch is unimportable by shimming it to raise.
        class _Blocker:
            def find_module(self, name, path=None): return self if name == "torch" else None
            def load_module(self, name): raise ImportError("torch blocked")
        sys.meta_path.insert(0, _Blocker())

        from otr_v2.hyworld import character_regression as cr  # noqa: F401

        # Core API surface must exist.
        assert hasattr(cr, "SSIM_GATE")
        assert hasattr(cr, "compute_ssim")
        assert hasattr(cr, "regress_character")
        assert hasattr(cr, "regress_cast")
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"torch-free import failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == "OK"


def test_ssim_gate_is_roadmap_day_12_bar() -> None:
    assert cr.SSIM_GATE == 0.85


# ---------------------------------------------------------------------
# SSIM math
# ---------------------------------------------------------------------


def test_ssim_solid_identical_triple_is_one() -> None:
    assert cr.ssim_solid((120, 30, 200), (120, 30, 200)) == pytest.approx(1.0)


def test_ssim_solid_max_divergence_is_low() -> None:
    # Black vs white -- maximum per-channel divergence.
    val = cr.ssim_solid((0, 0, 0), (255, 255, 255))
    # Using the simplified solid-ssim formula: (2*0*255 + C1)/(0 + 255^2 + C1)
    expected = (2.0 * 0.0 * 255.0 + 6.5025) / (0.0 + 255.0 ** 2 + 6.5025)
    assert val == pytest.approx(expected, rel=1e-6)
    assert val < 0.01


def test_ssim_solid_reduction_modes_agree_on_identity() -> None:
    triple = (80, 80, 80)
    for reduction in ("min", "mean", "product"):
        assert cr.ssim_solid(triple, triple, reduction=reduction) == pytest.approx(1.0)


def test_ssim_solid_reduction_modes_differ_on_divergence() -> None:
    # Unbalanced per-channel divergence: R close, G far, B mid.
    a = (100, 40, 140)
    b = (110, 240, 200)
    mn = cr.ssim_solid(a, b, reduction="min")
    mean = cr.ssim_solid(a, b, reduction="mean")
    product = cr.ssim_solid(a, b, reduction="product")
    assert mn < mean           # min is the worst channel
    assert product < mn        # product punishes divergence hardest
    assert 0.0 < mn < 1.0


def test_ssim_solid_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError):
        cr.ssim_solid((1, 1, 1), (1, 1, 1), reduction="harmonic")


def test_ssim_channel_is_symmetric() -> None:
    f = cr._ssim_channel_solid
    for a, b in [(10, 200), (40, 255), (0, 128), (77, 77)]:
        assert f(a, b) == pytest.approx(f(b, a), rel=1e-9)


# ---------------------------------------------------------------------
# stub PNG decoder
# ---------------------------------------------------------------------


def test_stub_decoder_roundtrips_known_color(tmp_path: Path) -> None:
    png_path = tmp_path / "solid.png"
    pp._stub_png(png_path, 123, 45, 200, width=64, height=64)
    assert cr._decode_stub_solid_rgb(png_path) == (123, 45, 200)


def test_stub_decoder_roundtrips_minimum_channel(tmp_path: Path) -> None:
    # _color_from_refs floors at 40 per channel -- the decoder should
    # handle the bottom of that range cleanly.
    png_path = tmp_path / "min.png"
    pp._stub_png(png_path, 40, 40, 40, width=32, height=32)
    assert cr._decode_stub_solid_rgb(png_path) == (40, 40, 40)


def test_stub_decoder_rejects_non_png(tmp_path: Path) -> None:
    bad = tmp_path / "not.png"
    bad.write_bytes(b"this is not a PNG")
    with pytest.raises(ValueError):
        cr._decode_stub_solid_rgb(bad)


def test_compute_ssim_auto_uses_stub_decoder(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    pp._stub_png(a, 50, 100, 150, width=32, height=32)
    pp._stub_png(b, 50, 100, 150, width=32, height=32)
    assert cr.compute_ssim(a, b, mode="auto") == pytest.approx(1.0)
    assert cr.compute_ssim(a, b, mode="stub") == pytest.approx(1.0)


def test_compute_ssim_auto_detects_divergence(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    pp._stub_png(a, 50, 50, 50, width=32, height=32)
    pp._stub_png(b, 240, 30, 240, width=32, height=32)
    val = cr.compute_ssim(a, b, mode="auto")
    assert val < cr.SSIM_GATE


def test_compute_ssim_invalid_mode_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    pp._stub_png(a, 10, 10, 10, width=16, height=16)
    pp._stub_png(b, 10, 10, 10, width=16, height=16)
    with pytest.raises(ValueError):
        cr.compute_ssim(a, b, mode="nonsense")


# ---------------------------------------------------------------------
# find_portraits walker
# ---------------------------------------------------------------------


def test_find_portraits_walks_scene_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"

    _populate_scene(
        "scene_01",
        otr_root,
        shots=[{
            "shot_id": "s01_b02",
            "character": "BABA",
            "refs": ["refs/baba_01.png", "refs/baba_02.png"],
            "env_prompt": "cockpit close up",
        }],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03",
        otr_root,
        shots=[{
            "shot_id": "s03_b04",
            "character": "BABA",
            "refs": ["refs/baba_01.png", "refs/baba_02.png"],
            "env_prompt": "airlock tight shot",
        }],
        monkeypatch=monkeypatch,
    )

    samples = cr.find_portraits(out_root, "BABA")
    assert len(samples) == 2
    scene_ids = sorted({s.scene_id for s in samples})
    assert scene_ids == ["scene_01", "scene_03"]
    for s in samples:
        assert s.character == "BABA"
        assert s.render_path.exists()
        assert s.refs_hash  # stamped by pulid stub meta


def test_find_portraits_ignores_other_characters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"

    _populate_scene(
        "scene_01",
        otr_root,
        shots=[
            {"shot_id": "s01_baba", "character": "BABA",
             "refs": ["refs/baba.png"]},
            {"shot_id": "s01_booey", "character": "BOOEY",
             "refs": ["refs/booey.png"]},
        ],
        monkeypatch=monkeypatch,
    )

    baba = cr.find_portraits(out_root, "BABA")
    booey = cr.find_portraits(out_root, "BOOEY")
    assert len(baba) == 1
    assert len(booey) == 1
    assert baba[0].character == "BABA"
    assert booey[0].character == "BOOEY"


def test_find_portraits_returns_empty_when_missing(tmp_path: Path) -> None:
    assert cr.find_portraits(tmp_path / "nonexistent", "BABA") == []


# ---------------------------------------------------------------------
# regress_character: Day 12 bar
# ---------------------------------------------------------------------


def test_same_refs_across_scenes_locks_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BABA with SAME refs in scene 1 and scene 3 must pass the gate."""
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"
    refs = ["refs/baba_01.png", "refs/baba_02.png"]

    _populate_scene(
        "scene_01", otr_root,
        shots=[{
            "shot_id": "s01_b02", "character": "BABA", "refs": refs,
            "env_prompt": "cockpit close up",
        }],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[{
            "shot_id": "s03_b04", "character": "BABA", "refs": refs,
            "env_prompt": "airlock tight shot",
        }],
        monkeypatch=monkeypatch,
    )

    result = cr.regress_character(out_root, "BABA")
    assert result.gate_ok is True
    assert result.min_ssim == pytest.approx(1.0)
    assert result.mean_ssim == pytest.approx(1.0)
    assert len(result.pairs) == 1
    assert result.pairs[0][2] == pytest.approx(1.0)


def test_different_refs_break_identity_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BABA with DIFFERENT refs in scene 1 vs scene 3 fails the gate.

    This is the Day 12 regression signal -- if the pipeline accidentally
    fed the wrong ref pack to scene 3, SSIM drops and the gate trips.
    """
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"

    _populate_scene(
        "scene_01", otr_root,
        shots=[{
            "shot_id": "s01_b02", "character": "BABA",
            "refs": ["refs/baba_A.png", "refs/baba_B.png"],
        }],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[{
            "shot_id": "s03_b04", "character": "BABA",
            # wildly different ref names -> different refs_hash
            # -> different stub color -> SSIM drops.
            "refs": ["refs/baba_drift_Z.png", "refs/baba_drift_Y.png"],
        }],
        monkeypatch=monkeypatch,
    )

    result = cr.regress_character(out_root, "BABA")
    assert result.gate_ok is False
    assert result.min_ssim < cr.SSIM_GATE


def test_character_regression_across_scene_1_and_scene_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full ROADMAP Day 12 bar: BABA + BOOEY likeness across scenes 1 & 3."""
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"
    baba_refs = ["refs/baba_01.png", "refs/baba_02.png"]
    booey_refs = ["refs/booey_01.png", "refs/booey_02.png"]

    _populate_scene(
        "scene_01", otr_root,
        shots=[
            {"shot_id": "s01_baba", "character": "BABA", "refs": baba_refs},
            {"shot_id": "s01_booey", "character": "BOOEY", "refs": booey_refs},
        ],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[
            {"shot_id": "s03_baba", "character": "BABA", "refs": baba_refs},
            {"shot_id": "s03_booey", "character": "BOOEY", "refs": booey_refs},
        ],
        monkeypatch=monkeypatch,
    )

    cast_report = cr.regress_cast(out_root, ("BABA", "BOOEY"))
    assert set(cast_report) == {"BABA", "BOOEY"}
    for c, result in cast_report.items():
        assert result.gate_ok, (
            f"Day 12 gate failed for {c}: min_ssim={result.min_ssim}"
        )
        assert result.min_ssim > cr.SSIM_GATE
        assert len(result.pairs) >= 1


def test_regress_character_within_scene_pairs_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within-scene shot-to-shot SSIMs must NOT factor into the gate."""
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"

    _populate_scene(
        "scene_01", otr_root,
        shots=[
            {"shot_id": "s01_a", "character": "BABA",
             "refs": ["refs/baba_A.png"]},
            {"shot_id": "s01_b", "character": "BABA",
             # different refs within the same scene -> would produce
             # divergent SSIM if within-scene pairs were included
             "refs": ["refs/baba_totally_different.png"]},
        ],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[{
            "shot_id": "s03_a", "character": "BABA",
            "refs": ["refs/baba_A.png"],
        }],
        monkeypatch=monkeypatch,
    )

    result = cr.regress_character(out_root, "BABA")
    # Only cross-scene pairs should appear: (scene_01/s01_a, scene_03/s03_a)
    # and (scene_01/s01_b, scene_03/s03_a).  That's 2 pairs.
    assert len(result.pairs) == 2
    for (a, b, _s) in result.pairs:
        scene_a = a.split("/")[0]
        scene_b = b.split("/")[0]
        assert scene_a != scene_b


def test_regress_character_empty_when_no_samples(tmp_path: Path) -> None:
    result = cr.regress_character(tmp_path, "BABA")
    assert result.gate_ok is True
    assert result.samples == []
    assert result.min_ssim == pytest.approx(1.0)
    assert any("no pulid_portrait samples" in n for n in result.notes)


def test_regress_character_single_scene_is_not_testable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A character with only one scene represented cannot fail the gate
    -- the regression requires ≥ 2 scenes."""
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"
    _populate_scene(
        "scene_01", otr_root,
        shots=[{
            "shot_id": "s01_a", "character": "BABA",
            "refs": ["refs/baba.png"],
        }],
        monkeypatch=monkeypatch,
    )
    result = cr.regress_character(out_root, "BABA")
    assert result.gate_ok is True
    assert len(result.samples) == 1
    assert any("only one scene" in n for n in result.notes)


def test_regress_cast_aggregates_per_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"

    _populate_scene(
        "scene_01", otr_root,
        shots=[
            {"shot_id": "s01_baba", "character": "BABA", "refs": ["a.png"]},
            {"shot_id": "s01_booey", "character": "BOOEY", "refs": ["b.png"]},
        ],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[
            {"shot_id": "s03_baba", "character": "BABA", "refs": ["a.png"]},
            {"shot_id": "s03_booey", "character": "BOOEY", "refs": ["b.png"]},
        ],
        monkeypatch=monkeypatch,
    )

    report = cr.regress_cast(out_root, ["BABA", "BOOEY"])
    assert set(report.keys()) == {"BABA", "BOOEY"}
    assert all(r.gate_ok for r in report.values())


def test_result_to_dict_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    otr_root = tmp_path
    out_root = otr_root / "io" / "hyworld_out"
    refs = ["refs/baba.png"]
    _populate_scene(
        "scene_01", otr_root,
        shots=[{"shot_id": "s01", "character": "BABA", "refs": refs}],
        monkeypatch=monkeypatch,
    )
    _populate_scene(
        "scene_03", otr_root,
        shots=[{"shot_id": "s03", "character": "BABA", "refs": refs}],
        monkeypatch=monkeypatch,
    )
    result = cr.regress_character(out_root, "BABA")
    d = result.to_dict()
    for key in (
        "character", "gate", "min_ssim", "mean_ssim", "gate_ok",
        "n_samples", "n_pairs", "pairs", "samples", "notes",
    ):
        assert key in d
    assert d["character"] == "BABA"
    assert d["gate"] == cr.SSIM_GATE
    assert d["n_samples"] == 2
    assert d["n_pairs"] == 1
    # JSON-serialisable
    json.dumps(d)


# ---------------------------------------------------------------------
# real-mode SSIM path (import guard)
# ---------------------------------------------------------------------


def test_compute_real_ssim_requires_pillow_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When PIL or numpy is missing, real-mode SSIM raises cleanly."""
    # Shim both imports to raise.
    import builtins
    real_import = builtins.__import__

    def _blocker(name, *args, **kwargs):
        if name in ("PIL", "PIL.Image", "numpy"):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocker)

    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    pp._stub_png(png_a, 10, 20, 30, width=16, height=16)
    pp._stub_png(png_b, 40, 50, 60, width=16, height=16)

    with pytest.raises(ImportError, match="Pillow"):
        cr.compute_ssim(png_a, png_b, mode="real")
