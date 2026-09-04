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
    "tests/", "kibitz-runs/", ".github/", ".claude/", "docs/",
    "kibitz-plugin/", "tmp/",
    # 2026-09-04: developer tooling, excluded from the bundle on a fable
    # pre-publish review -- nothing shipped imports it, and one of its files
    # was contributing an os.environ finding to the registry scan for code
    # that is not part of the product.
    "tools/",
)
# scripts/* is excluded except the three subprocess workers that ship.
_SHIPPED_SCRIPTS = {
    "scripts/_otr_chatterbox_worker.py",
    "scripts/_otr_dia_worker.py",
    "scripts/_otr_indextts2_worker.py",
}


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
