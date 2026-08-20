"""THE UPSCALE LANE'S MISSING WEIGHT GATE (2026-08-19).

WHY THIS FILE EXISTS. The video namespace has Gate 1 of
`docs/VIDEO_LANE_PREFLIGHT.md` -- "every declared weight resolves via
folder_paths or a documented env pin" -- and `tests/test_lane_preflight_matrix.py`
enforces it per lane. The UPSCALE namespace has thirteen test files and not one
of them asks whether the engine's checkpoint is reachable. That asymmetry is how
the following went unnoticed:

* the headless model-paths yaml mapped `upscale_models` to
  `C:/ComfyUI-Models/upscale_models/`, which on this box holds a `.cache`
  directory and nothing else;
* so the running headless server's own `UpscaleModelLoader` reported
  `options: []` -- an EMPTY model list, confirmed live against `/object_info`;
* and `spandrel_esrgan` kept working anyway, because
  `eng_spandrel_esrgan._resolve_model` has a repo-relative fallback that finds
  `<comfy_root>/models/upscale_models/`.

**THE ENGINE WAS NEVER DEAD, AND THAT IS THE INTERESTING PART.** A safety net
was silently carrying the lane while the primary route was broken, which is
strictly worse than a clean failure: nothing was red, so nothing got fixed, and
the day the fallback stops applying -- a different repo layout, a packaged
install, a move of the ComfyUI root -- the lane dies with no warning and no
test to catch it. The yaml now lists BOTH roots so the primary path works; these
tests make sure the question is asked from now on.

Pure CPU: no CUDA, no model load, no running server. Every check is a read of a
declaration or a stat of a file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
HEADLESS_YAML = REPO / "scripts" / "_otr_headless_model_paths.yaml"


def _headless_roots(key: str) -> list:
    """The directories the HEADLESS boot maps ``key`` to.

    Read from the yaml the launcher actually passes to
    ``--extra-model-paths-config``, because a mapping nobody loads is not a
    mapping. A scalar value is one root; a block scalar is several.
    """
    doc = yaml.safe_load(HEADLESS_YAML.read_text(encoding="utf-8"))
    section = doc["comfyui_desktop"]
    raw = section[key]
    return [line.strip() for line in str(raw).split() if line.strip()]


def test_the_headless_yaml_is_still_parseable():
    """It is hand-maintained and feeds every headless boot; a syntax error
    here takes the whole server down before OTR gets a say."""
    doc = yaml.safe_load(HEADLESS_YAML.read_text(encoding="utf-8"))
    assert "comfyui_desktop" in doc
    assert doc["comfyui_desktop"]["base_path"]


def test_every_registered_upscale_engine_can_reach_its_checkpoint():
    """THE GATE ITSELF -- the upscale twin of video Gate 1.

    Asks each registered engine to resolve its own checkpoint through its own
    resolver, which is the only question that matters. An engine declaring no
    model requirement (``off``) is skipped rather than excused by name, so a
    future third engine is covered the day it registers.
    """
    os.environ.setdefault("OTR_TEST_MODE", "1")
    from nodes._otr_upscale_engines import registry as ureg

    checked = 0
    for name in ureg.all_engine_names():
        row = ureg.CAPABILITIES.get(name, {})
        if not row.get("model_requirements"):
            continue
        engine = ureg.get_engine(name)
        resolver = getattr(engine, "_resolve_model", None)
        assert callable(resolver), (
            "%s declares model_requirements %r but exposes no _resolve_model "
            "for preflight to check -- the video namespace's lesson L1 in the "
            "upscale namespace" % (name, row["model_requirements"]))
        _candidates, path = resolver()
        assert path, (
            "%s cannot reach its checkpoint. It declares %r; no candidate "
            "directory contains it. Drop the weight under models/"
            "upscale_models/ or add its real location to the "
            "upscale_models key of scripts/_otr_headless_model_paths.yaml"
            % (name, row["model_requirements"]))
        assert os.path.getsize(path) > 1024 * 1024, (
            "%s resolved %r but it is under 1 MB -- a truncated download or "
            "an LFS pointer, not a checkpoint" % (name, path))
        checked += 1
    assert checked, "no upscale engine declares a model requirement any more"


def test_the_headless_boot_maps_upscale_models_somewhere_that_has_weights():
    """THE DEFECT THIS FILE WAS WRITTEN FOR, pinned so it cannot come back.

    The engine's repo-relative fallback is a SAFETY NET, not the route. If the
    only directory the headless server is told about is empty, then ComfyUI's
    own ``UpscaleModelLoader`` offers an empty dropdown -- which is exactly
    what it did (`options: []`, live) -- and the lane is one refactor away from
    dying silently. At least one mapped root must actually contain a weight.
    """
    roots = _headless_roots("upscale_models")
    assert roots, "the headless yaml maps upscale_models nowhere at all"
    populated = {}
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            continue
        weights = [f for f in p.iterdir()
                   if f.is_file() and f.suffix.lower() in
                   (".pth", ".safetensors", ".ckpt", ".bin")]
        if weights:
            populated[root] = [w.name for w in weights]
    assert populated, (
        "NONE of the upscale_models roots the headless boot is given contains "
        "a weight file: %r. The engine's repo-relative fallback may still save "
        "it, but ComfyUI's own UpscaleModelLoader will show an EMPTY dropdown "
        "and the lane is surviving on a safety net rather than its primary "
        "route." % (roots,))


def test_the_pinned_esrgan_filename_matches_what_is_on_disk():
    """The engine pins a filename; the gate above proves *a* weight exists.
    This proves it is THE one the engine names, so a differently-named
    RealESRGAN build cannot satisfy the check above while the loader still
    misses."""
    os.environ.setdefault("OTR_TEST_MODE", "1")
    from nodes._otr_upscale_engines import registry as ureg

    if "spandrel_esrgan" not in set(ureg.all_engine_names()):
        pytest.skip("spandrel_esrgan is no longer registered")
    engine = ureg.get_engine("spandrel_esrgan")
    pinned = getattr(engine, "_model_filename", "")
    assert pinned, "the engine no longer pins a checkpoint filename"
    _candidates, path = engine._resolve_model()
    assert path and os.path.basename(path) == pinned, (
        "resolved %r but the engine pins %r" % (path, pinned))
