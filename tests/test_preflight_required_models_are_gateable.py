"""tests/test_preflight_required_models_are_gateable.py -- `required_models`
names weight FILES, and the runner's gate checks every one it cannot download.

The canonical runner checks `preflight.required_models` against the server's
`/object_info`, which lists FILENAMES. A logical id (`ltxv-2b-0.9.8-distilled`)
or an HF repo id (`google/gemma-4-E2B-it`) can never appear there, so an entry
like that could only be skipped with a note -- a declaration nothing checked.
Three rows carried them until 2026-09-25, one naming a writer the row does not
even use. The profile schema now refuses any entry that is not a weight file,
so the gate has one kind of entry and checks all of it.

Every profile read here is a `config/workflow_matrix.json` row: the matrix is
the only source `load_profile` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nodes._otr_shared import capability_profiles as cp

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

#: A weight no engine downloads, so the gate has something to check.
_UNFETCHED = "operator_supplied_weight.safetensors"


def _gate():
    """Import the runner without dragging its CLI into collection."""
    import otr_canonical_api_run as runner  # noqa: PLC0415
    return runner


def _real_row_declaring(name):
    """A complete matrix row whose preflight declares only ``name``."""
    row = cp.load_profile("otr_8gb_low")
    return dict(row, preflight={"required_models": [name], "required_keys": []})


def _row_declaring(monkeypatch, *names):
    """A synthetic row that selects nothing downloadable and declares ``names``."""
    profile = {
        "role_overrides": {"announcer_visual": "viz_green"},
        "preflight": {"required_models": list(names)},
    }
    monkeypatch.setattr(cp, "load_profile", lambda _name: profile)
    return "synthetic"


@pytest.mark.parametrize("name", [
    "v1-5-pruned-emaonly-fp16.safetensors",
    "v3_sd15_mm.ckpt",
    "v3_sd15_adapter.ckpt",
    "mm-p_0.5.pth",
    "model.onnx",
])
def test_weight_filenames_are_accepted(name):
    cp.validate_profile_shape(_real_row_declaring(name))


@pytest.mark.parametrize("name", [
    "ltxv-2b-0.9.8-distilled",     # logical id
    "t5xxl_fp16",                  # logical id
    "ltx-2.3-22b-dev",             # dotted, and still an id -- not a file
    "google/gemma-4-E2B-it",       # HF repo id
    "google/gemma-4-12b-it",
])
def test_a_row_naming_an_id_is_refused_at_load(name):
    with pytest.raises(cp.ProfileError, match="required_models"):
        cp.validate_profile_shape(_real_row_declaring(name))


def test_every_matrix_row_declares_only_weight_files():
    for rid in cp.known_profile_ids():
        names = ((cp.load_profile(rid).get("preflight") or {})
                 .get("required_models") or [])
        assert all(cp.is_weight_filename(n) for n in names), (rid, names)


def test_offline_schemas_report_a_missing_weight_instead_of_refusing(
        monkeypatch, capsys):
    """`--offline-schemas` has no server and no `--extra-model-paths-config`,
    so its model lists reflect the CALLING process's folder_paths, not the
    roots the real server booted with. Refusing on that evidence blocks a
    profile whose weights are on disk.

    Caught live: a `--dry-run --offline-schemas` validation sweep reported
    the haunted ghost-signal profile FAIL on all three of its weights -- two
    of which sit under roots the headless yaml explicitly names -- and the
    same profile passed preflight against the running server moments later.
    """
    row = _row_declaring(monkeypatch, _UNFETCHED)
    checked = _gate()._assert_profile_models_present(row, {}, offline=True)
    out = capsys.readouterr().out
    assert "NOT checked" in out, "offline preflight must say it did not check"
    assert "not evidence of absence" in out
    assert checked == [], (
        "offline verified nothing, so it must not report names as 'visible to "
        "the server' -- that contradicts the notice printed beside it")


def test_the_live_gate_still_refuses_a_missing_weight(monkeypatch):
    """The offline carve-out must not disarm the real gate. A declared file
    that nothing downloads and the server cannot see stops the run."""
    row = _row_declaring(monkeypatch, _UNFETCHED)
    with pytest.raises(SystemExit) as exc:
        _gate()._assert_profile_models_present(row, {}, offline=False)
    assert "PREFLIGHT FAIL" in str(exc.value)
    assert _UNFETCHED in str(exc.value)
