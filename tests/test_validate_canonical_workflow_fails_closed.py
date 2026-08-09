"""The workflow validator script must FAIL when it cannot run its contract.

WHY THIS FILE EXISTS. `scripts/validate_canonical_workflow.py` advertises, in
its own module docstring, that it "Runs OTR_WorkflowValidator's contract check"
and is "Callable from the suite gate + from CI". It did neither reliably: when
it could not resolve NODE_CLASS_MAPPINGS it printed a SKIPPED line to stderr and
returned an EMPTY problem list, which `main()` reads as "nothing wrong" -> prints
OK -> exits 0.

And the resolution could never succeed. The package directory is
`ComfyUI-OldTimeRadio` with a HYPHEN, so the old
`importlib.import_module(REPO_ROOT.name.replace("-", "_"))` was permanently
unsatisfiable -- ComfyUI loads the pack BY PATH. So the headline check had never
run on any box, while the script reported OK. It was reproduced live and
recorded in docs/HANDOFF_LOG.md, and `scripts/otr_macbeth_probe.py` had to route
around it to get a real contract check.

A comment cannot fail. These tests can.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    """Load the script by path -- it lives in scripts/, not an importable pkg."""
    path = REPO_ROOT / "scripts" / "validate_canonical_workflow.py"
    spec = importlib.util.spec_from_file_location(
        "_vcw_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_contract_failure_is_a_problem_not_a_silent_skip(tmp_path, monkeypatch):
    """The regression itself: an unresolvable package must PRODUCE A PROBLEM.

    Pointing REPO_ROOT at a directory with no __init__.py reproduces exactly
    what the hyphenated-name import used to do on every box. The old code
    returned [] here, which is what made `main()` exit 0.
    """
    vcw = _load_script_module()
    fake_root = tmp_path / "Fake-Pack"
    (fake_root / "nodes").mkdir(parents=True)
    monkeypatch.setattr(vcw, "REPO_ROOT", fake_root)

    problems = vcw._run_validator_contract({"nodes": [], "links": []}, "unit")

    assert problems, (
        "an unrunnable contract returned NO problems -- that is the fail-open "
        "that made the script exit 0 having validated nothing"
    )
    joined = " ".join(problems)
    assert "CONTRACT NOT RUN" in joined, joined


def test_package_loads_by_path_despite_the_hyphenated_directory(tmp_path):
    """The root cause, pinned. `import ComfyUI_OldTimeRadio` cannot ever work;
    loading __init__.py by path can. If someone 'simplifies' the loader back to
    import_module, this goes red."""
    vcw = _load_script_module()
    assert "-" in REPO_ROOT.name, (
        f"{REPO_ROOT.name!r} has no hyphen, so this test no longer guards "
        f"anything -- re-derive the root cause before deleting it"
    )
    pkg = vcw._load_otr_package()
    ncm = getattr(pkg, "NODE_CLASS_MAPPINGS", None)
    assert ncm, "package loaded but exposed no NODE_CLASS_MAPPINGS"
    assert len(ncm) > 1


def test_canonical_workflow_passes_the_real_contract():
    """Now that the contract actually runs, the canonical graph must pass it.

    Before the fix this assertion was unreachable: the contract was skipped, so
    a green here proved only that the skip path worked.
    """
    vcw = _load_script_module()
    wf = vcw._load(REPO_ROOT / "workflows" / "otr_canonical.json")
    problems = vcw._run_validator_contract(wf, "canonical")
    assert problems == [], problems


def test_main_exits_nonzero_when_the_contract_cannot_run(tmp_path, monkeypatch):
    """End-to-end on the exit CODE, because that is what a gate reads."""
    vcw = _load_script_module()
    fake_root = tmp_path / "Fake-Pack"
    (fake_root / "nodes").mkdir(parents=True)
    monkeypatch.setattr(vcw, "REPO_ROOT", fake_root)

    rc = vcw.main([str(REPO_ROOT / "workflows" / "otr_canonical.json")])
    assert rc == 1, f"expected a failing exit code, got {rc!r}"


def test_main_exits_zero_on_the_real_canonical_graph():
    """The control. Fail-closed must not mean fail-always."""
    vcw = _load_script_module()
    rc = vcw.main([str(REPO_ROOT / "workflows" / "otr_canonical.json")])
    assert rc == 0, f"canonical workflow should validate cleanly, got rc={rc}"
