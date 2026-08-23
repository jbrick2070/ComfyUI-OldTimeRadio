"""tests/test_preflight_required_models_are_gateable.py -- the preflight gate
only enforces what `/object_info` can actually answer.

FOUND BY ITEM F, 2026-08-23. The canonical runner's preflight model gate shipped
2026-08-22 and was validated against the `ghost_signal*` profiles -- the only
ones whose `preflight.required_models` hold weight FILENAMES. Every profile using
the LOGICAL-ID vocabulary could not pass it at any time, for any state of the
disk, because `/object_info` enumerates filenames and has never contained
`real-esrgan-x2plus`, `wan2.2-ti2v-5b` or `google/gemma-4-E2B-it`.

`otr_upscale_ship` sat in the go-forward queue as "unexercised" because of this.
`RealESRGAN_x2plus.pth` was visible in `/object_info` the whole time -- confirmed
live on the running server, three RealESRGAN weights listed.

TWO RULES ARE PINNED HERE:

1. A gate may enforce a claim it can verify and must only REPORT one it cannot.
   `_is_weight_filename` is that split, so it gets real cases on both sides --
   including the ids that contain dots (`wan2.2-ti2v-5b`, `ltx-2.3-22b-dev`),
   which is why the suffix list is closed rather than a "contains a dot" test.
2. A profile that names a weight filename must name the SAME one its engine
   loads. That is the drift this file exists to catch: the profile and
   `eng_spandrel_esrgan._model_filename` are two copies of one fact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_PROFILES = _REPO / "config" / "profiles"
_SCRIPTS = _REPO / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _gate():
    """Import the runner's split without dragging its CLI into collection."""
    import otr_canonical_api_run as runner  # noqa: PLC0415
    return runner


@pytest.mark.parametrize("name", [
    "RealESRGAN_x2plus.pth",
    "v1-5-pruned-emaonly-fp16.safetensors",
    "v3_sd15_mm.ckpt",
    "v3_sd15_adapter.ckpt",
    "mm-p_0.5.pth",
    "gemma-4-12b-it-Q4_K_M.gguf",
    "model.onnx",
])
def test_weight_filenames_are_gateable(name):
    assert _gate()._is_weight_filename(name), (
        f"{name!r} is a weight filename and MUST be enforced against "
        f"/object_info -- this is the check that caught the missing "
        f"v3_sd15_adapter.ckpt in seconds instead of 428.")


@pytest.mark.parametrize("name", [
    "real-esrgan-x2plus",          # the id that blocked otr_upscale_ship
    "wan2.2-ti2v-5b",              # DOTTED, and still an id -- not a file
    "ltx-2.3-22b-dev",             # dotted id
    "ltx-2.3-22b-dev-gguf",        # "gguf" as a SUFFIX WORD, not an extension
    "google/gemma-4-E2B-it",       # HF repo id; never in /object_info at all
    "gemma-3-12b",
    "umt5-xxl-encoder",
    "wan2.2_vae",
])
def test_logical_and_repo_ids_are_not_gateable(name):
    assert not _gate()._is_weight_filename(name), (
        f"{name!r} is a logical/repo id, not a filename. /object_info cannot "
        f"speak to it, so treating its absence as a failure blocks a profile "
        f"whose weights are on disk -- the exact defect item F found.")


def test_ltx_gguf_id_is_not_mistaken_for_a_gguf_file():
    """The one genuinely tricky pair, called out because it nearly collides."""
    g = _gate()
    assert g._is_weight_filename("model-Q4_K_M.gguf")
    assert not g._is_weight_filename("ltx-2.3-22b-dev-gguf")


def _profiles_with_required_models():
    out = {}
    for path in sorted(_PROFILES.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        req = ((data.get("preflight") or {}).get("required_models") or [])
        if req:
            out[path.name] = req
    return out


def test_the_upscale_profiles_name_the_file_their_engine_actually_loads():
    """Profile and engine hold one fact in two places -- pin them together."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import (  # noqa: PLC0415
        SpandrelEsrgan as _Eng)
    engine_file = _Eng._model_filename
    declared = _profiles_with_required_models()
    for profile in ("otr_upscale_ship.json", "otr_upscale_ltx_probe.json"):
        assert profile in declared, f"{profile} lost its required_models"
        assert engine_file in declared[profile], (
            f"{profile} declares {declared[profile]!r} but "
            f"eng_spandrel_esrgan loads {engine_file!r}. These are one fact in "
            f"two files; a rename on either side must move both.")


def test_every_ghost_signal_requirement_stays_enforced():
    """The gate's load-bearing case, read from the PROFILES not a fixture list.

    REPLACED A TAUTOLOGY, 2026-08-23. This test previously ended in
    `assert isinstance(enforced, bool)` -- and `_is_weight_filename` returns
    `str.endswith(...)`, which is always a bool, so the assertion could not fail
    for any implementation. QA proved it by substituting a classifier that
    returned True for everything and then False for everything: both mutants
    passed. It asserted nothing while claiming to assert totality.

    What actually matters is this: `ghost_signal*` is the family whose weights
    the gate really guards -- the missing `v3_sd15_adapter.ckpt` is the catch
    that justified building it. If `_is_weight_filename` ever regressed toward
    False, every one of those would silently become "reported" and the gate
    would be off while still printing reassuring lines. The parametrized tests
    above cannot catch that, because they hardcode their own names instead of
    reading what the profiles declare.
    """
    g = _gate()
    declared = _profiles_with_required_models()
    ghosts = {p: n for p, n in declared.items() if p.startswith("otr_ghost_signal")}
    assert ghosts, "no ghost_signal profile declares required_models"
    for profile, names in sorted(ghosts.items()):
        unenforced = [n for n in names if not g._is_weight_filename(n)]
        assert not unenforced, (
            f"{profile} declares {unenforced!r}, which the gate would only "
            f"REPORT, not enforce. These are the weights whose absence must "
            f"stop a render in seconds rather than 428.")


def test_the_gate_still_enforces_something_repo_wide():
    """A classifier stuck on False disables the gate everywhere, silently."""
    g = _gate()
    enforced = [n for names in _profiles_with_required_models().values()
                for n in names if g._is_weight_filename(n)]
    assert len(enforced) >= 5, (
        f"only {len(enforced)} requirement(s) repo-wide are enforced. The "
        f"preflight gate is effectively OFF -- suspect _is_weight_filename.")
