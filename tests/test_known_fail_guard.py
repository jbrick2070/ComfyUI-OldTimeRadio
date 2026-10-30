"""The known-failure guard is keyed by machine, and each entry is held to the
ledger and to a measured failure signature (tests/conftest.py MACHINE_FACTS,
apple/known-failures.md). A machine the table does not know has an empty set
and no environment change, so the 5080's run is what it was before the table
existed."""
from __future__ import annotations

import re
from pathlib import Path

from tests import conftest as guard

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "apple" / "known-failures.md"


def test_a_machine_the_table_does_not_know_has_no_expected_failures():
    for host in ("IDREAM", "idream", "idream.local", "", None, "a-box-nobody-named", "MRKT-2"):
        assert guard.expected_failures(host) == {}, host


def test_the_machine_key_is_the_first_label_upper_cased():
    assert guard.machine_key("mrkt") == "MRKT"
    assert guard.machine_key("MRKT.lan") == "MRKT"
    assert guard.machine_key(" idream ") == "IDREAM"
    assert guard.machine_key("") == "" and guard.machine_key(None) == ""
    assert guard.expected_failures("mrkt.lan") == guard.expected_failures("MRKT")


def test_every_known_failure_names_a_real_test_and_is_in_the_ledger():
    """A nodeid that no longer exists, or that the ledger does not explain,
    is a set entry nobody can act on."""
    ledger = LEDGER.read_text(encoding="utf-8")
    for machine, facts in guard.MACHINE_FACTS.items():
        assert machine == machine.upper() and "." not in machine, machine
        root = facts.get("models_root", "")
        assert not root or Path(root).is_absolute(), (machine, root)
        entries = facts["expected_failed"]
        assert entries, machine
        for nodeid, signature in entries.items():
            path, _, rest = nodeid.partition("::")
            name = rest.split("::")[-1].split("[")[0]
            source = (REPO / path).read_text(encoding="utf-8")
            assert re.search(r"^\s*def %s\(" % re.escape(name), source, re.M), nodeid
            assert isinstance(signature, str) and signature.strip(), nodeid
            assert nodeid in ledger, "not in apple/known-failures.md: %s" % nodeid


def test_the_verdict_counts_only_what_ran_and_holds_each_failure_to_its_signature():
    expected = {"t.py::known": "the known reason", "t.py::other_known": "its reason",
                "t.py::never_ran": "whatever"}
    executed = {"t.py::known", "t.py::other_known", "t.py::new", "t.py::fine"}
    failed = {
        "t.py::known": "E   AssertionError: the known reason\nassert None",
        "t.py::other_known": "E   TypeError: something else entirely",
        "t.py::new": "E   AssertionError: brand new",
    }
    new, promotable = guard.guard_verdict(executed, failed, expected)
    assert set(new) == {"t.py::other_known", "t.py::new"}
    assert "its reason" in new["t.py::other_known"]
    assert new["t.py::new"] == "not a known failure"
    # A known failure that ran and passed is promotable; one that never
    # started is neither passing nor failing.
    assert promotable == set()
    new2, promotable2 = guard.guard_verdict({"t.py::fixed", "t.py::fine"}, {}, {"t.py::fixed": "x", **expected})
    assert new2 == {} and promotable2 == {"t.py::fixed"}
    assert guard.guard_verdict(set(), {}, expected) == ({}, set())


def test_a_report_counts_as_failed_when_it_failed_or_xfailed_and_not_when_it_xpassed():
    from types import SimpleNamespace as Rep

    passed = Rep(failed=False, skipped=False, longreprtext="")
    broke = Rep(failed=True, skipped=False, longreprtext="E   AssertionError: nan")
    xfailed = Rep(failed=False, skipped=True, wasxfail="known", longreprtext="E   assert (None)")
    xpassed = Rep(failed=True, skipped=False, longreprtext="[XPASS(strict)] known")
    assert guard.report_outcome([None, None, None]) == (False, None)
    assert guard.report_outcome([passed, passed, passed]) == (True, None)
    assert guard.report_outcome([passed, broke, passed]) == (True, "E   AssertionError: nan")
    # A known failure that failed as its guard mark expects carries its text,
    # so the signature can be checked ...
    assert guard.report_outcome([passed, xfailed, passed], known=True) == (True, "E   assert (None)")
    # ... but an xfail a test author placed is an accepted outcome, as ever
    # (the 2026-10-30 full suite: a cloud-partner conformance row the guard
    # had read as a regression).
    assert guard.report_outcome([passed, xfailed, passed]) == (True, None)
    # A strict xfail that passed is pytest's own failure and the guard's
    # promotable, never "failed a different way".
    assert guard.report_outcome([passed, xpassed, passed], known=True) == (True, None)
    # A setup error on a known failure is still a failure with its text.
    assert guard.report_outcome([broke, None, None], known=True) == (True, "E   AssertionError: nan")


def test_this_box_runs_with_its_own_facts_or_none():
    """The live values are the table's for this hostname: a renamed box reads
    as no facts, never as another machine's."""
    facts = guard.MACHINE_FACTS.get(guard.MACHINE, {})
    assert guard.EXPECTED_FAILURES == facts.get("expected_failed", {})
    assert guard.EXPECTED_FAILED_NODEIDS == frozenset(guard.EXPECTED_FAILURES)
