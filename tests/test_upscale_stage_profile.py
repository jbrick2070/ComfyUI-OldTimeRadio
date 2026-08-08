"""Queue item 8 (2026-08-08): the upscale_stage profile schema.

Asserts:
* `upscale_stage` is an OPTIONAL top-level section; a profile that omits it
  passes validation (operator-dirty `otr_g4_wan_ti2v.json` is the load-bearing
  case per section-10 no-touch).
* A profile that names an UNKNOWN engine fails cross_validate_profile with
  an accumulated `ProfileError` (not a raw ImportError).
* A profile that specifies `engine` but omits `device` is legal (Sonnet 5
  MF-3: device is in _SECTION_OPTIONAL_KEYS).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nodes._otr_shared.capability_profiles import (
    PROFILE_DIR, ProfileError, load_profile, load_widget_mapping,
    cross_validate_profile, validate_profile_shape,
)


def _decls():
    from nodes._otr_audio_engines.registry import CAPABILITIES as A
    from nodes._otr_video_engines.registry import CAPABILITIES as V
    from nodes._otr_image_engines.registry import CAPABILITIES as I
    from nodes._otr_upscale_engines.registry import CAPABILITIES as U
    return {"audio": A, "video": V, "image": I, "upscale": U}


def test_ship_profile_loads_and_cross_validates():
    p = load_profile("otr_upscale_ship")
    assert p["upscale_stage"]["engine"] == "spandrel_esrgan"
    assert p["upscale_stage"]["device"] == "cuda:0"
    cross_validate_profile(p, load_widget_mapping(), _decls())


def test_profile_without_upscale_stage_still_loads():
    """Every shipping profile that predates queue item 8 has no
    upscale_stage section; they must all keep loading."""
    p = load_profile("otr_g4_wan_ti2v")
    assert "upscale_stage" not in p


def test_partial_upscale_stage_engine_only_is_legal():
    """Sonnet 5 MF-3: `device` is optional; a profile writing just
    {"engine": "spandrel_esrgan"} must pass validation."""
    src = json.loads(Path(PROFILE_DIR, "otr_w45_wan_ti2v.json").read_text(encoding="utf-8"))
    src["id"] = "otr_upscale_partial_test"
    src["display_name"] = "test only"
    src["upscale_stage"] = {"engine": "spandrel_esrgan"}
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "otr_upscale_partial_test.json"
        tp.write_text(json.dumps(src), encoding="utf-8")
        loaded = load_profile("otr_upscale_partial_test", profile_dir=td)
        assert loaded["upscale_stage"] == {"engine": "spandrel_esrgan"}


def test_unknown_engine_raises_profile_error():
    src = json.loads(Path(PROFILE_DIR, "otr_w45_wan_ti2v.json").read_text(encoding="utf-8"))
    src["id"] = "otr_upscale_bad"
    src["upscale_stage"] = {"engine": "not_a_real_engine", "device": "cpu"}
    with pytest.raises(ProfileError) as excinfo:
        cross_validate_profile(src, load_widget_mapping(), _decls())
    assert "upscale_stage.engine" in str(excinfo.value)
    assert "not registered" in str(excinfo.value)


def test_retired_engine_raises_profile_error():
    src = json.loads(Path(PROFILE_DIR, "otr_w45_wan_ti2v.json").read_text(encoding="utf-8"))
    src["id"] = "otr_upscale_retired_test"
    src["upscale_stage"] = {"engine": "some_retired_id", "device": "cpu"}
    # Monkeypatch the retired frozenset for this test.
    from nodes._otr_upscale_engines import registry as reg
    orig = reg.RETIRED_UPSCALE_ENGINE_IDS
    reg.RETIRED_UPSCALE_ENGINE_IDS = frozenset({"some_retired_id"})
    try:
        with pytest.raises(ProfileError) as excinfo:
            cross_validate_profile(src, load_widget_mapping(), _decls())
        assert "retired" in str(excinfo.value)
    finally:
        reg.RETIRED_UPSCALE_ENGINE_IDS = orig


def test_widget_mapping_upscale_registry_is_valid():
    """The `upscale_stage.engine` widget-mapping entry uses `registry:
    "upscale"` -- adding a new registry token requires updating
    `_REGISTRY_NAMES` at capability_profiles.py:314. Validate the mapping
    parses cleanly under the extended validator."""
    m = load_widget_mapping()
    entry = m["managed"].get("upscale_stage.engine")
    assert entry is not None
    assert entry["registry"] == "upscale"


def test_widget_mapping_upscale_device_is_null_registry():
    """`upscale_stage.device` is a free-form string validated at runtime by
    resolve_device -- not a registry-membership check."""
    m = load_widget_mapping()
    entry = m["managed"].get("upscale_stage.device")
    assert entry is not None
    assert entry["registry"] is None
