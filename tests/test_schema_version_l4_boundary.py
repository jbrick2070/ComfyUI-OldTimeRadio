"""B6: the l4 boundary, and the write path that must not erase it.

WHAT THE BUMP IS FOR. `scripts/audit_spoken_citations.py` requires a
`spoken_coda_source` receipt on any provenance-owned ledger newer than its
legacy set -- and until this bump the CURRENT version was itself listed as
legacy, so every live ledger sorted as history and the requirement never fired
once. Bumping the constant is what switches the audit on.

WHY THE WRITE PATH HAD TO CHANGE IN THE SAME COMMIT. `save_ledger_safe` used
to restamp `CURRENT_SCHEMA_VERSION` unconditionally, and six post-hoc tools
call it over EXISTING episodes. A read-only corpus measurement found 43 of 48
provenance-carrying ledgers that predate the receipt: without a preserve
policy, one routine write-back promotes each of them past what it actually
contains, and the audit then reports a receipt they never could have carried
as a REGRESSION ON HISTORY. Turning enforcement on while leaving the promoter
armed would have manufactured 43 false failures on day one.

The tests below cover both halves: the boundary is drawn in the right place,
and history cannot be dragged across it by accident.

Pure Python; no GPU, no model, no network.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nodes import _otr_ledger as _OTRL  # noqa: E402
from nodes import _otr_ledger_freeze as _LFC  # noqa: E402
from nodes import production_ledger as _PL  # noqa: E402

#: The version this commit pins. Spelled out rather than derived, because a
#: test that reads the constant it is meant to pin proves nothing.
L4 = "l4-2026-08-07"

#: The version l4 replaced. Still a real, readable lineage member.
PREVIOUS = "l3-2026-05-14"

#: An older real version. Ten live ledgers actually sit here.
OLDER = "l3-2026-05-08"


def _ledger_path(tmp_path: Path, episode_id: str) -> Path:
    audio = tmp_path / "output" / "otr" / "episodes" / episode_id / "audio"
    audio.mkdir(parents=True)
    (tmp_path / "output" / "otr" / "obs").mkdir(parents=True, exist_ok=True)
    return audio / f"{episode_id}_ledger.json"


# ---------------------------------------------------------------------------
# The constant and everything that derives from it
# ---------------------------------------------------------------------------

def test_the_current_version_is_l4():
    assert _OTRL.CURRENT_SCHEMA_VERSION == L4


def test_the_derivation_chain_stays_in_lockstep():
    """One constant, two derived mirrors. If any link starts hardcoding, a
    ledger written by one path fails the other path's exact-equality gate."""
    assert _PL.Ledger.SCHEMA_VERSION == _OTRL.CURRENT_SCHEMA_VERSION
    assert _LFC.EXPECTED_SCHEMA_VERSION == _PL.Ledger.SCHEMA_VERSION


@pytest.mark.parametrize("module_path", [
    "nodes/production_ledger.py",
    "nodes/_otr_ledger_freeze.py",
])
def test_the_defensive_fallback_literals_do_not_lie(module_path):
    """Both mirrors pull the version live and fall back to a hardcoded string
    if the import fails. That branch is unreachable in practice, which is
    exactly why it rots -- and a fallback that reports a version the build
    stopped writing is worse than no fallback at all. Read from source: the
    live value comes from the import, so only the text can show the literal."""
    source = (REPO / module_path).read_text(encoding="utf-8")
    literals = set(re.findall(r'SCHEMA_VERSION[^=\n]*=\s*"(l\d-\d{4}-\d{2}-\d{2})"',
                              source))
    assert literals == {L4}, (
        f"{module_path} carries stale fallback literal(s) {sorted(literals)}")


# ---------------------------------------------------------------------------
# save_ledger_safe -- the four version states
# ---------------------------------------------------------------------------

def test_a_ledger_with_no_version_is_stamped_current(tmp_path):
    path = _ledger_path(tmp_path, "absent_version")
    ledger = {"episode_id": "absent_version", "lines": []}

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == L4
    assert saved["meta"]["schema_version"] == L4


def test_a_ledger_already_current_is_stamped_current(tmp_path):
    path = _ledger_path(tmp_path, "already_current")
    ledger = {"episode_id": "already_current", "schema_version": L4, "lines": []}

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == L4
    assert saved["meta"]["schema_version"] == L4


@pytest.mark.parametrize("legacy", [PREVIOUS, OLDER, "l1-2026-04-24"])
def test_a_legacy_ledger_is_never_promoted_by_a_write_back(tmp_path, legacy):
    """THE 43-LEDGER CASE. A post-hoc tool touching an old episode must leave
    its version where it found it. Promotion here makes a receipt the ledger
    never could have carried look like one that was dropped."""
    path = _ledger_path(tmp_path, "legacy_writeback")
    ledger = {
        "episode_id": "legacy_writeback",
        "schema_version": legacy,
        "lines": [],
        "meta": {"provenance": {"status": "public_domain_us"}},
    }

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == legacy, "history was promoted"
    assert saved["meta"]["schema_version"] == legacy


def test_a_preserved_version_is_mirrored_so_the_two_surfaces_agree(tmp_path):
    """The top level is authoritative -- it is what the freeze validator pins --
    so a disagreeing `meta` mirror is corrected DOWN to the preserved value,
    never used as a reason to promote."""
    path = _ledger_path(tmp_path, "disagreeing_mirror")
    ledger = {
        "episode_id": "disagreeing_mirror",
        "schema_version": OLDER,
        "lines": [],
        "meta": {"schema_version": "l3-2026-04-28"},
    }

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == OLDER
    assert saved["meta"]["schema_version"] == OLDER


def test_the_preserve_branch_says_so_out_loud(tmp_path, caplog):
    """A ledger that stays behind must be observable at the moment it happens,
    not reconstructed later from a puzzling audit report."""
    path = _ledger_path(tmp_path, "warns_on_preserve")
    ledger = {"episode_id": "warns_on_preserve", "schema_version": OLDER,
              "lines": []}

    with caplog.at_level("WARNING", logger="OTR"):
        assert _OTRL.save_ledger_safe(path, ledger)

    assert any(OLDER in r.getMessage() and "NOT migrated" in r.getMessage()
               for r in caplog.records), "the non-migration was silent"


def test_a_non_string_version_is_repaired_rather_than_preserved(tmp_path):
    """Preserve applies to a real lineage string. A corrupt value is not
    history worth keeping -- carrying it forward would put a ledger in a state
    no reader can classify."""
    path = _ledger_path(tmp_path, "corrupt_version")
    ledger = {"episode_id": "corrupt_version", "schema_version": 3, "lines": []}

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == L4


def test_preserving_the_version_does_not_skip_the_rest_of_the_save(tmp_path):
    """The version branch sits at the top of the write. A legacy ledger must
    still get its meta.paths block -- the preserve policy narrows ONE field,
    not the whole save."""
    path = _ledger_path(tmp_path, "legacy_still_saves")
    ledger = {"episode_id": "legacy_still_saves", "schema_version": OLDER,
              "lines": []}

    assert _OTRL.save_ledger_safe(path, ledger)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "paths" in saved["meta"]
    assert saved["meta"]["paths"]["layout"] == "per-episode-workspace"


# ---------------------------------------------------------------------------
# A live ledger is born current and freezes cleanly
# ---------------------------------------------------------------------------

def test_a_fresh_ledger_is_born_at_l4(tmp_path):
    led = _PL.Ledger("signal_lost_l4_probe_20260807_000000", str(tmp_path))
    assert led.data["schema_version"] == L4


def test_a_fresh_ledger_passes_the_exact_equality_freeze_gate(tmp_path):
    """`_check_meta_invariants` compares `schema_version` for EXACT equality,
    so a producer and the freeze validator disagreeing by one bump is a hard
    error rather than a warning. This is the assertion that would have caught
    a half-applied sweep."""
    led = _PL.Ledger("signal_lost_l4_freeze_20260807_000000", str(tmp_path))
    errors: list = []
    warnings: list = []

    _LFC._check_meta_invariants(led.data, errors, warnings)

    schema_errors = [e for e in errors if "schema_version" in e]
    assert schema_errors == [], f"a freshly born ledger failed the gate: {errors}"


def test_a_legacy_ledger_still_fails_the_freeze_gate_loudly():
    """The mirror image, and the reason the preserve policy is not a way to
    freeze old episodes: a preserved ledger is readable and auditable, but it
    is NOT current, and the freeze gate must keep saying so."""
    errors: list = []
    warnings: list = []

    _LFC._check_meta_invariants(
        {"schema_version": PREVIOUS, "meta": {}}, errors, warnings)

    assert any("schema_version" in e for e in errors)


# ---------------------------------------------------------------------------
# The dormant promoter -- pinned so it stays dormant
# ---------------------------------------------------------------------------

def test_ledger_init_never_resumes_from_disk(tmp_path):
    """`Ledger.save()` stamps the current version unconditionally at
    `production_ledger.py:1461-1462` -- the only other place in the tree that
    assigns `schema_version`. It is harmless ONLY because a Ledger is born
    fresh and never adopts an on-disk ledger's version. Pin that, or the
    promoter quietly starts mattering the first time someone adds a resume."""
    episode_id = "signal_lost_resume_probe_20260807_000000"
    path = _ledger_path(tmp_path, episode_id)
    path.write_text(json.dumps({
        "episode_id": episode_id,
        "schema_version": OLDER,
        "lines": [{"line_id": "l001", "text": "from disk"}],
    }), encoding="utf-8")

    led = _PL.Ledger(episode_id, str(tmp_path))

    assert led.data["schema_version"] == L4, "a fresh Ledger is born current"
    assert led.data["lines"] == [], (
        "Ledger.__init__ adopted on-disk rows -- it now RESUMES, and the "
        "unconditional promotion in save() is no longer dormant")


def test_nothing_outside_production_ledger_constructs_a_ledger():
    """The two constructors (`new_ledger`, `get_ledger`) both build a fresh
    episode. A construction site anywhere else is the shape of a resume, which
    is the flow that would arm the promoter above."""
    offenders = []
    for path in sorted((REPO / "nodes").rglob("*.py")) + \
            sorted((REPO / "scripts").rglob("*.py")):
        if path.name == "production_ledger.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Ledger"):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == [], (
        f"Ledger constructed outside its module at {offenders}; if this is a "
        "resume, the unconditional schema stamp in Ledger.save() will promote "
        "whatever was on disk")
