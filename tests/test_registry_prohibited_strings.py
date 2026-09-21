"""PBUG-20260902-04: the Comfy Registry security scan marks a version critical
and Flagged (admin tags any-code-execute + credential-access) when a shipped
file carries the hidden-input type name of the logged-in account's session
bearer -- the string its pylint scanner reports as "Prohibited string
detected". 2.0.0-alpha.13 through alpha.15 were all flagged on exactly that
finding in nodes/OTR_LedgerScriptWriter.py, and no version can go Active
while it is present.

This walks every Python file the registry bundle ships and refuses the literal
anywhere -- code, string or comment -- so it cannot come back through a merge.
The literal is assembled at runtime so this file never spells it either.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The session-bearer hidden-input type name, assembled so it never appears
# verbatim in the tree.
PROHIBITED = "AUTH_TOKEN" + "_COMFY_ORG"

# Mirrors .comfyignore: these prefixes never ship in the registry bundle.
_UNSHIPPED_PREFIXES = (
    "tests/", "kibitz-runs/", ".github/", ".claude/", ".cursor/", "docs/",
    "kibitz-plugin/", "tmp/",
    # 2026-09-04: developer tooling, excluded from the bundle on a fable
    # pre-publish review -- nothing shipped imports it, and one of its files
    # was contributing an os.environ finding to the registry scan for code
    # that is not part of the product.
    "tools/",
)
# .comfyignore excludes `scripts/*` and then re-includes individual files
# with a leading `!`. Both lists below are DERIVED from that file rather than
# mirrored by hand: the hand-written mirror had drifted in both directions at
# once -- it named a worker that is not re-included (so does not ship) and
# omitted two that are (so the guard was not reading two shipped files, which
# is a false negative in a test whose job is to keep a banned literal out of
# the bundle).
#
# LIMIT, stated rather than left to be discovered: this reads the two shapes
# .comfyignore actually uses -- a `!` re-include, and a literal path with no
# wildcard. A future entry using a new glob shape needs this updated, and
# tests/test_registry_prohibited_strings.py::test_the_shipped_set_matches_comfyignore
# is what will say so.
def _comfyignore_lines():
    path = REPO_ROOT / ".comfyignore"
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


#: Paths .comfyignore pulls back IN with a leading `!`.
_SHIPPED_SCRIPTS = frozenset(
    line[1:] for line in _comfyignore_lines()
    if line.startswith("!") and line[1:].startswith("scripts/")
)

#: Individual files excluded by an exact path (no wildcard), e.g. the
#: IndexTTS2 engine, which is excluded as a pair with its worker.
_UNSHIPPED_FILES = frozenset(
    line for line in _comfyignore_lines()
    if not line.startswith(("!", "/"))
    and not any(ch in line for ch in "*?[")
    and line.endswith(".py")
)


def _shipped_python_files() -> list[Path]:
    """Every shipped file the scanner reads -- .py AND the data files
    beside them. The 2026-09-02 fix cleared the .py declaration and this
    helper only listed *.py, so 14 copies of the literal survived in the
    shipped Partner pin (nodes/_otr_shared/partner_nodes.yaml) with no
    test watching them."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "*.py", "*.yaml", "*.yml", "*.json"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git ls-files unavailable: {exc}")
    files = []
    for rel in out.splitlines():
        rel = rel.strip().replace("\\", "/")
        if not rel:
            continue
        if rel.startswith("scripts/") and rel not in _SHIPPED_SCRIPTS:
            continue
        if rel.startswith(_UNSHIPPED_PREFIXES):
            continue
        if rel in _UNSHIPPED_FILES:
            continue
        files.append(REPO_ROOT / rel)
    assert files, "git ls-files returned no shipped Python files"
    return files


def test_no_shipped_file_names_the_session_bearer_hidden_input():
    offenders = []
    for path in _shipped_python_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if PROHIBITED in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "registry prohibited string (session-bearer hidden input) present in "
        "shipped files -- every version carrying it is Flagged: "
        + ", ".join(offenders)
    )


def test_the_shipped_set_matches_comfyignore():
    """The parsers above read two shapes; fail if `.comfyignore` grows a third.

    `_SHIPPED_SCRIPTS` and `_UNSHIPPED_FILES` are derived by reading
    `.comfyignore` directly rather than mirroring it, because the hand-written
    mirror they replaced had drifted in both directions at once. That
    derivation understands exactly two shapes: a `!` re-include naming a plain
    path, and a literal path with no wildcard.

    This is the check the comment above promises. It does NOT re-implement
    gitignore -- it asserts the assumptions the parsers rest on, so a future
    entry in a shape they cannot read fails here instead of silently shrinking
    the guarded set. A wildcard in a `!` line, or a trailing comment after one,
    would both slip past `line[1:].strip()` and take a shipped file out of the
    scan with nothing to say so.
    """
    lines = _comfyignore_lines()
    assert lines, ".comfyignore is empty or unreadable"

    bare = [l for l in lines if l.startswith("!")]
    assert bare, "no `!` re-includes found; the derivation would be empty"

    for line in bare:
        rest = line[1:]
        assert rest == rest.strip(), (
            "a `!` line has surrounding whitespace, which the parser keeps "
            "verbatim: %r" % line)
        assert "#" not in rest, (
            "a `!` line carries a trailing comment; the parser would treat it "
            "as part of the path: %r" % line)
        assert not any(ch in rest for ch in "*?["), (
            "a `!` line uses a glob, which the parser cannot expand -- the "
            "files it matches would drop out of the guarded set silently: %r"
            % line)

    # Every re-included script must be a real file, or the derivation is
    # naming something that cannot be scanned.
    for rel in _SHIPPED_SCRIPTS:
        assert (REPO_ROOT / rel).is_file(), (
            "%s is re-included by .comfyignore but does not exist" % rel)

    # And every single-file exclusion must be real too, or it is silently
    # excluding nothing while looking like it excludes something.
    for rel in _UNSHIPPED_FILES:
        assert (REPO_ROOT / rel).is_file(), (
            "%s is excluded by .comfyignore but does not exist" % rel)
