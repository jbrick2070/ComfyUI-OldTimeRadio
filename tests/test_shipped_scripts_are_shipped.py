"""tests/test_shipped_scripts_are_shipped.py

EVERY `scripts/` FILE A SHIPPED NODE NEEDS AT RUNTIME MUST SURVIVE
`.comfyignore` (2026-09-05).

This defect has now shipped TWICE. On 2026-09-01 the blanket `scripts/*`
exclusion broke three TTS engines, each of which resolves its worker as
`os.path.join(_REPO_ROOT, "scripts", ...)`; three negations were added. On
2026-09-05 the published `2.0.0-alpha.20` zip was downloaded and grepped, and
`nodes/_otr_video_engines/eng_mesh_stage.py:84` was found doing exactly the same
thing with `otr_mesh_stage_blender.py`, which was NOT negated -- so the mesh/3D
lane could not start on any registry install.

Both times the tree was fine and only the BUNDLE was broken, which is why a test
that reads the tree cannot catch it. This one reads `.comfyignore` the way git
does and answers the only question that matters: for every runtime `scripts/`
path a shipped node builds, does that file survive the exclusion rules?

The matcher is `pathspec`'s `gitwildmatch`, not a hand-rolled one: `.comfyignore`
is gitignore syntax, and re-implementing its negation precedence is exactly how
the original `scripts/` vs `scripts/*` bug got past a reader -- a negation cannot
re-include a file whose PARENT directory is excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODES = REPO / "nodes"
COMFYIGNORE = REPO / ".comfyignore"

#: A shipped node building a path into `scripts/`. Matches the two spellings the
#: pack actually uses: `os.path.join(..., "scripts", "x.py")` and `/ "scripts"`.
_JOIN = re.compile(
    r'os\.path\.join\([^)]*["\']scripts["\']\s*,\s*["\']([^"\']+)["\']'
    r'|["\']scripts["\']\s*/\s*["\']([^"\']+)["\']'
)


def _runtime_script_requirements() -> set[str]:
    found: set[str] = set()
    for py in NODES.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        for m in _JOIN.finditer(py.read_text(encoding="utf-8", errors="replace")):
            name = m.group(1) or m.group(2)
            if name:
                found.add("scripts/" + name)
    return found


def _is_excluded(rel: str) -> bool:
    """True iff `.comfyignore` alone excludes `rel`.

    `pathspec` is the canonical gitignore implementation and takes the rules
    from ONE file, which is what we want -- `git check-ignore` has no
    `--exclude-from`, and its `core.excludesFile` route layers `.gitignore` on
    top, which would report files as excluded for reasons that have nothing to
    do with the bundle. NEITHER path may skip: a guard that skips is how the
    mesh_stage defect reached the registry in the first place.
    """
    lines = COMFYIGNORE.read_text(encoding="utf-8").splitlines()
    try:
        import pathspec
    except ImportError:  # pragma: no cover -- pathspec ships with the venv
        pytest.fail(
            "pathspec is required to evaluate .comfyignore; install it rather "
            "than letting this guard skip")
    spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return spec.match_file(rel)


def test_at_least_one_requirement_is_detected():
    """A regex that silently matches nothing would make every assertion below
    vacuously true -- the exact way this class of test rots."""
    reqs = _runtime_script_requirements()
    assert reqs, "no runtime scripts/ requirement found; the matcher has drifted"


@pytest.mark.parametrize("rel", sorted(_runtime_script_requirements()))
def test_every_runtime_script_survives_comfyignore(rel):
    assert (REPO / rel).is_file(), (
        "%s is required at runtime but is not in the tree" % rel)
    assert not _is_excluded(rel), (
        "%s is required at runtime -- a shipped node joins this path and uses it "
        "-- but .comfyignore EXCLUDES it, so the published zip cannot have it. "
        "Add `!%s`. This is the mesh_stage/TTS-worker defect returning." % (rel, rel))


def test_the_shipped_installer_does_not_execute_remote_code():
    """`scripts/_otr_indextts2_install.ps1` ships in the registry bundle. It used
    to bootstrap uv with `irm <url> | iex` -- a remote script piped into
    execution, inside a pack asking a reviewer to trust its account of what it
    runs. Fetch-and-execute stays out of anything we ship."""
    ps1 = REPO / "scripts" / "_otr_indextts2_install.ps1"
    if _is_excluded("scripts/_otr_indextts2_install.ps1"):
        return  # not shipped -> nothing to protect, and NOT a skip
    body = ps1.read_text(encoding="utf-8", errors="replace")
    code = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("#"))
    for pattern in ("| iex", "|iex", "Invoke-Expression", "DownloadString"):
        assert pattern not in code, (
            "shipped installer executes remote code (%r)" % pattern)
