"""tests/test_preflight_required_models_are_gateable.py -- the preflight gate
only enforces what `/object_info` can actually answer.

FOUND BY ITEM F, 2026-08-23. The canonical runner's preflight model gate shipped
2026-08-22 and was validated against the `ghost_signal*` profiles -- the only
ones whose `preflight.required_models` hold weight FILENAMES. Every profile using
the LOGICAL-ID vocabulary could not pass it at any time, for any state of the
disk, because `/object_info` enumerates filenames and has never contained
`real-esrgan-x2plus`, `wan2.2-ti2v-5b` or `google/gemma-4-E2B-it`.

The upscale profile sat in the go-forward queue as "unexercised" because of this.
`RealESRGAN_x2plus.pth` was visible in `/object_info` the whole time -- confirmed
live on the running server, three RealESRGAN weights listed.

TWO RULES ARE PINNED HERE:

1. A gate may enforce a claim it can verify and must only REPORT one it cannot.
   `_is_weight_filename` is that split, so it gets real cases on both sides --
   including the ids that contain dots (`wan2.2-ti2v-5b`, `ltx-2.3-22b-dev`),
   which is why the suffix list is closed rather than a "contains a dot" test.
2. A profile that names a weight filename must name the SAME one its engine
   loads -- the profile and the engine are two copies of one fact. The upscale
   profiles that first carried this are retired with the other lab rigs; the
   live instance is the LTX 2.5 rows, pinned in
   tests/test_portability_profile_contracts.py.

Every profile read here is a `config/workflow_matrix.json` row: the matrix is
the only source `load_profile` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nodes._otr_shared import capability_profiles as cp

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
#: A real row running the haunted ghost-signal lane, whose three weights are
#: FILENAMES -- the case the gate was built for.
_GHOST_ROW = "otr_8gb_animatediff"

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
    "model.onnx",
])
def test_weight_filenames_are_gateable(name):
    assert _gate()._is_weight_filename(name), (
        f"{name!r} is a weight filename and MUST be enforced against "
        f"/object_info -- this is the check that caught the missing "
        f"v3_sd15_adapter.ckpt in seconds instead of 428.")


@pytest.mark.parametrize("name", [
    "real-esrgan-x2plus",          # the id that blocked the upscale profile
    "wan2.2-ti2v-5b",              # DOTTED, and still an id -- not a file
    "ltx-2.3-22b-dev",             # dotted id
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


def _profiles_with_required_models():
    out = {}
    for rid in cp.known_profile_ids():
        data = cp.load_profile(rid)
        req = ((data.get("preflight") or {}).get("required_models") or [])
        if req:
            out[rid] = req
    return out


def _ghost_lane_weights(profile_id):
    """The weights the ghost-signal (AnimateDiff) lanes of one row load, read
    from each lane's own CAPABILITIES row. Empty for a row running none."""
    from nodes._otr_video_engines import registry as vreg  # noqa: PLC0415
    from nodes._otr_video_engines.eng_ghost_signal import (  # noqa: PLC0415
        GhostSignalEngine)
    prof = cp.load_profile(profile_id)
    weights = set()
    for key, pick in (prof.get("role_overrides") or {}).items():
        if not key.endswith("_visual") or not vreg.is_registered(pick):
            continue
        if isinstance(vreg.get_engine(pick), GhostSignalEngine):
            weights.update(vreg.CAPABILITIES[pick]["model_requirements"])
    return weights


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

    A row lists its writer too (an HF repo id the gate only reports, by
    design), so what is held here is the part that is the ghost lane's own:
    every name the row declares that its ghost lanes load.
    """
    g = _gate()
    declared = _profiles_with_required_models()
    ghosts = {}
    for profile, names in declared.items():
        lane = _ghost_lane_weights(profile)
        listed = [n for n in names if n in lane]
        if listed:
            ghosts[profile] = listed
    assert _GHOST_ROW in ghosts, (
        "no ghost-signal row declares its lane's weights: %r" % sorted(ghosts))
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


# --------------------------------------------------------------------------- #
# RULE 3, added 2026-08-26: OFFLINE SCHEMAS ARE NOT A SERVER.
# Same principle as rule 1 above, applied to the axis it originally missed.
# --------------------------------------------------------------------------- #

def test_offline_schemas_report_a_missing_weight_instead_of_refusing(capsys):
    """`--offline-schemas` has no server and no `--extra-model-paths-config`,
    so its model lists reflect the CALLING process's folder_paths, not the
    roots the real server booted with. Refusing on that evidence blocks a
    profile whose weights are on disk.

    Caught live: a `--dry-run --offline-schemas` validation sweep reported
    the haunted ghost-signal profile FAIL on all three of its weights -- two
    of which sit under roots the headless yaml explicitly names -- and the
    same profile passed preflight against the running server moments later. The
    message even asserted "the running server cannot see", a claim it had not
    made and, offline, could not make.
    """
    g = _gate()
    empty_schemas: dict = {}
    checked = g._assert_profile_models_present(
        _GHOST_ROW, empty_schemas, offline=True)
    out = capsys.readouterr().out
    assert "NOT checked" in out, "offline preflight must say it did not check"
    assert "not evidence of absence" in out
    assert checked == [], (
        "offline verified nothing, so it must not report names as 'visible to "
        "the server' -- that contradicts the notice printed beside it")


def test_the_live_gate_still_refuses_a_missing_weight():
    """The offline carve-out must not disarm the real gate. With `offline`
    False and a schema set that lists nothing, a declared weight filename is
    genuinely unseeable and must still stop the run in seconds."""
    g = _gate()
    with pytest.raises(SystemExit) as exc:
        g._assert_profile_models_present(
            _GHOST_ROW, {}, offline=False)
    assert "PREFLIGHT FAIL" in str(exc.value)
