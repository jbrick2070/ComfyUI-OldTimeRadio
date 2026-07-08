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


def test_absolute_path_taken_as_is(tmp_path):
    abs_path = tmp_path / "anything.json"
    assert wv._resolve_workflow_path(str(abs_path)) == abs_path


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


def test_is_changed_signature_tracks_disk_changes(tmp_path):
    f = tmp_path / "wf.json"
    f.write_text("{}", encoding="utf-8")
    sig1 = wv.WorkflowValidator.IS_CHANGED(str(f), True, True)
    os.utime(f, ns=(1_000_000_000, 2_000_000_000))
    sig2 = wv.WorkflowValidator.IS_CHANGED(str(f), True, True)
    assert sig1 != sig2
