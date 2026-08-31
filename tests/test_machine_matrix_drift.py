"""The machine matrix must not drift from the profiles it claims to describe.

WHY THIS TEST EXISTS, and it is not hypothetical. README told users an 8 GB card
"has rendered **nothing**" and marked the haunted lane "?" at 8 GB, for days
after that exact card published nine episodes across five source banks. Both
statements were true when written. Nobody re-read them.

A compatibility claim is the worst kind of documentation to hand-maintain,
because it goes stale in the direction that costs a reader the most: not "this
might work and doesn't", but "your card cannot do this" about the thing it has
already done. So the table is generated, and this test is what makes the
generation binding rather than advisory.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "otr_machine_matrix.py")


def test_matrix_and_readme_are_in_sync_with_the_profiles():
    """`--check` writes nothing and fails if either surface is stale."""
    r = subprocess.run([sys.executable, _SCRIPT, "--check"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, (
        "docs/MACHINE_MATRIX.md or README's generated block no longer matches "
        "config/profiles/. Regenerate with:\n"
        "    python scripts/otr_machine_matrix.py\n\n" + r.stdout + r.stderr)


def test_declared_machine_classes_agree_with_their_profiles():
    """A class naming a profile that contradicts it must refuse to generate.

    The guard raises rather than warns on purpose: a table that quietly
    disagrees with the profiles is worse than no table, since a reader cannot
    tell which of the two to believe.
    """
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402

    profiles = M.load_profiles()
    M.load_classes(profiles)                # raises on contradiction

    by_id = {p["id"]: p for p in profiles}
    for row, prof in M.load_classes(profiles):
        if prof is None:
            continue                        # a declared gap is allowed
        assert prof["id"] in by_id
        want = row.get("gpu_vendor")
        if want:
            assert prof["vendor"] in (want, "?"), (
                "%s is %s but its class declares %s"
                % (prof["id"], prof["vendor"], want))


def test_proven_receipts_carry_their_evidence():
    """PROVEN is the strongest claim the table makes; it must be checkable.

    Receipts live on the MATRIX ROW, not on a profile file -- a profile has a
    declared shape and an extra key there broke `build_variants --check`. A
    receipt without hardware, a count and a pointer to evidence is an opinion
    wearing a verdict's badge, which is what the hand-written table had become.
    """
    import json
    with io.open(os.path.join(_REPO, "config", "machine_classes.json"),
                 encoding="utf-8") as fh:
        matrix = json.load(fh)

    seen = 0
    for row in matrix.get("classes", []):
        for receipt in (row.get("proven") or []):
            seen += 1
            for field in ("hardware", "episodes", "evidence"):
                assert receipt.get(field), (
                    "%s has a proven receipt missing %r: %r"
                    % (row.get("key"), field, receipt))
            assert int(receipt["episodes"]) > 0, (
                "%s claims proof with zero episodes" % row.get("key"))
    assert seen, ("no machine class carries a proven receipt -- the matrix's "
                  "strongest column is empty")
