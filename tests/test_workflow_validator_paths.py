"""GATE B S2 code-defect tests -- node-63 path resolution.

The two verified defects (switchable-workflow decision doc, section 2):
  * `_load_workflow` resolved non-empty RELATIVE paths against the process
    CWD;
  * `IS_CHANGED` had the same defect, so a relative path hashed a phantom
    mtime=0 and on-disk edits never re-triggered validation.

Fix contract (`_resolve_workflow_path`, shared by both):
  empty -> canonical default; relative -> REPO ROOT anchored; absolute -> as-is.

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os

import pytest
import pathlib

from nodes import _otr_workflow_validator as wv

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_REL = os.path.join("workflows", "otr_canonical.json")


def test_empty_path_resolves_to_canonical_default():
    assert wv._resolve_workflow_path("") == wv._DEFAULT_WORKFLOW_PATH


def test_relative_path_resolves_against_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a CWD that contains NO workflows/ dir
    p = wv._resolve_workflow_path(CANONICAL_REL)
    assert p == REPO_ROOT / CANONICAL_REL
    assert p.is_file(), "repo-root anchored relative path must hit the real file"


def test_absolute_path_inside_a_workflow_folder_is_taken(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_INPUT_DIR", str(tmp_path))
    abs_path = tmp_path / "anything.json"
    assert wv._resolve_workflow_path(str(abs_path)) == abs_path


# Registry review, 2026-10: "policy-v0.6: arbitrary-file-read, path-traversal".
# The widget reaches the node from an unauthenticated /prompt, so a path outside
# the workflow folders -- or one walking out of the pack with "../" -- is refused.

@pytest.mark.parametrize("value", [
    "../../../secrets.json",
    "workflows/../../outside.json",
])
def test_a_traversal_out_of_the_pack_is_refused(value):
    with pytest.raises(ValueError, match="traversal"):
        wv._resolve_workflow_path(value)


def test_an_absolute_path_outside_every_workflow_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_INPUT_DIR", str(tmp_path / "input"))
    with pytest.raises(ValueError, match="outside every permitted read root"):
        wv._resolve_workflow_path(str(tmp_path / "elsewhere" / "wf.json"))


def test_is_changed_never_touches_or_raises_on_a_refused_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_INPUT_DIR", str(tmp_path / "input"))
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    sig = wv.WorkflowValidator.IS_CHANGED(str(outside), True, True)
    assert sig.split("|")[1] == "0"


def test_load_workflow_relative_path_from_foreign_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = wv._load_workflow(CANONICAL_REL)
    assert wf.get("nodes"), "relative load from a foreign CWD must parse the master"


def test_is_changed_hashes_real_mtime_for_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sig = wv.WorkflowValidator.IS_CHANGED(CANONICAL_REL, True, True)
    mtime_field = sig.split("|")[-3]
    assert mtime_field != "0", (
        "IS_CHANGED must hash the real file mtime for a relative path "
        f"(got signature {sig!r}) -- mtime=0 is the CWD-defect signature"
    )


def test_is_changed_signature_tracks_disk_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_INPUT_DIR", str(tmp_path))
    f = tmp_path / "wf.json"
    f.write_text("{}", encoding="utf-8")
    sig1 = wv.WorkflowValidator.IS_CHANGED(str(f), True, True)
    os.utime(f, ns=(1_000_000_000, 2_000_000_000))
    sig2 = wv.WorkflowValidator.IS_CHANGED(str(f), True, True)
    assert sig1 != sig2
