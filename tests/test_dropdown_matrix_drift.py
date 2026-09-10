"""The dropdown matrix may not drift from the registries or the manifests.

WHY THIS TEST EXISTS, and it is not hypothetical. README carried a
hand-typed table saying `flux2_klein` was **proven** on a 16 GB Mac. Both
halves of that were true in isolation -- it HAD rendered on an M4, and the
receipt is real -- but its declaration is ``device_backends: ["cuda"]``, so
`availability()` refuses it on every Mac profile. The table promised a stranger
something the code blocks, which is the single worst direction a compatibility
claim can be wrong in.

The cause was collapsing two questions into one word: *will OTR offer me this
here* and *will it fit*. So the generator answers them separately, derives the
first from the code, and refuses to build when a curated receipt lands on a cell
the code refuses. These tests are what make that binding rather than advisory.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "otr_dropdown_matrix.py")


def _generator():
    spec = importlib.util.spec_from_file_location("_otr_dropdown_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_doc_and_readme_match_the_live_code():
    """`--check` writes nothing and fails if either surface is stale."""
    r = subprocess.run([sys.executable, _SCRIPT, "--check"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, (
        "docs/DROPDOWN_MATRIX.md or README's generated block no longer matches "
        "the registries, the profiles or the fetch manifests. Regenerate in "
        "this commit:\n    python scripts/otr_dropdown_matrix.py\n\n"
        + r.stdout + r.stderr)


def test_every_registered_engine_has_a_row():
    """A new engine cannot ship without appearing in the matrix.

    Compared as a SUBSET, not equality: the table also carries the WRITER
    dimension, which does not live in any engine registry (writers come from
    `_otr_model_catalog`'s curated list). Equality here asserted that the only
    thing worth charting is a registry engine -- which is exactly the
    assumption that let the matrix ship with no writers at all while every
    published episode turned on one.
    """
    M = _generator()
    caps = M.registry_capabilities()
    declared = {e for table in caps.values() for e in table}
    charted = {row["engine"] for row in M.build_rows()}
    assert declared <= charted, (
        "engines missing from the matrix: %s" % sorted(declared - charted))


def test_every_curated_writer_has_a_row():
    """The writer dimension, pinned the same way.

    Added after an audit of the published episodes found the matrix silent on
    writers -- while all seven Mac episodes used the same one, and it is the
    largest single download in the graph.
    """
    M = _generator()
    charted = {row["engine"] for row in M.build_rows()
               if row["namespace"] == "writer"}
    cat = M._load("nodes/_otr_model_catalog.py", "_odm_catalog_test")
    declared = {m.repo_id for m in cat._active_curated_models()
                if getattr(m, "provider", "local") == "local"}
    assert declared == charted, (
        "writers missing from the matrix: %s" % sorted(declared - charted))


def test_no_curated_receipt_sits_on_a_cell_the_code_refuses():
    """The flux2_klein trap, as a gate.

    A curated `proven` on a machine whose profile refuses the engine means one
    of two things, and both are defects: the receipt is wrong, or the
    declaration is stale and never earned its backend back.
    """
    M = _generator()
    assert M.conflicts(M.build_rows()) == []


def test_every_unrouted_engine_declares_why():
    """No silent provisioning gaps.

    An engine with no fetcher lane is fine when the provisioner SAYS why --
    it auto-downloads through the HF cache, it has its own installer, it is a
    hosted API, it is pure code. An engine with neither a lane nor a reason is
    one nothing will ever fetch weights for, and a profile selecting it fails
    to provision. That is worth failing a build over rather than discovering
    on somebody else's machine.
    """
    M = _generator()
    orphans = sorted(row["public"] for row in M.build_rows()
                     if row["friction"] == "unrouted")
    assert orphans == [], (
        "these engines have no provisioning lane and no NO_LANE_REASON entry "
        "in scripts/otr_provision.py: %s" % orphans)


def test_lane_sizes_match_their_manifests():
    """LANE_INFO is the pick list people read before spending the bandwidth.

    Its own docstring says a wrong number there is worse than no number. Every
    lane whose manifest carries `expected_bytes` on all of its artifacts is
    checked against the sum; `download_facts` raises when they disagree.
    """
    M = _generator()
    M.download_facts()          # raises SystemExit on drift


def test_the_two_questions_stay_separate():
    """Selectability is derived; only memory verdicts may be curated.

    If the curated file ever grows a key that overrides an availability
    answer, the generator has stopped deriving the half that cannot go stale.
    """
    M = _generator()
    curated = M.load_curated().get("engines", {})
    allowed = {"size_gb", "memory", "note"}
    for engine, row in curated.items():
        extra = set(row) - allowed
        assert not extra, (
            "%s carries %s in docs/dropdown_matrix.json; only %s may be "
            "curated -- everything else is derived from the code."
            % (engine, sorted(extra), sorted(allowed)))
        verdicts = set(row.get("memory", {}).values())
        # "measured" is a DISTINCT state from "proven" and the split is the
        # whole go-forward test plan: proven = a published episode used it;
        # measured = it ran on that hardware in a lab test and nothing has
        # shipped with it. Collapsing them once put bark and musicgen -- which
        # have Metal timings and zero episodes -- in the same column as engines
        # that had carried a whole episode.
        assert verdicts <= {"proven", "measured", "fits", "oom", "no",
                            "unknown"}, (
            "%s uses an unknown memory verdict: %s" % (engine, sorted(verdicts)))


def test_machine_columns_name_real_shipped_profiles():
    """Every column is reproducible -- a reader can run that exact profile."""
    for machine in _generator().MACHINES:
        path = os.path.join(_REPO, "config/profiles/%s.json" % machine["profile"])
        assert os.path.exists(path), (
            "column %r names profile %r, which does not exist"
            % (machine["label"], machine["profile"]))
