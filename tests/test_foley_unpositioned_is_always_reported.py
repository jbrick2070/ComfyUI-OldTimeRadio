"""``foley_unpositioned`` must appear in the mux receipt even when it is zero.

2026-08-30. The counter used to be gated on a non-zero value, which makes the
number unreadable by anyone who goes looking for it. The 4060 ran the sentinel
fix on 8 GB hardware, went to this receipt to confirm how many beats owned no
master-mix window, and could only report:

    "foley_unpositioned= -- NOT EMITTED. No foley receipt line of any kind
     appears in this leg's log. I'm reporting that as absence, not as zero:
     I cannot tell from this run whether the counter is zero or simply not
     logged on this path."

That is the correct way to read a conditional counter and it is exactly why one
is useless: *absent* and *zero* must not share a representation. A count of
unpositioned beats is a standing invariant somebody will come to verify --
``otr_master_audio_mux`` itself says "a bed quietly missing from half an episode
must not be invisible" -- so it is reported at every value.

The neighbouring counters (``muted_samples``, ``skipped``, ``conform_notes``)
stay conditional deliberately: those are exceptions worth noticing, not
invariants worth confirming. This test does not touch them.

**SCOPE, and this test would overstate itself without it.** Removing the gate
did NOT answer the question the 4060 was actually asking, because the counter
was only the smaller half of what it hit. The entire foley block -- the whole
``report`` list that every ``foley_*`` line is appended to -- sits behind
``_foley_route()`` / ``foley_stems.is_foley_route``, which is true only when a
role renders on the foley engine. The profile under test was
``animatediff15_v3_haunted_video``, so that block never ran and NO foley line
printed at all; "no foley receipt line of any kind" was the complete finding,
not a symptom of this gate.

And the counter could not have seen a bridge there regardless: it increments per
foley STEM lacking ``start_s`` (``foley_stems.py``), and its own comment scopes
that to "on a foley lane it still produces a stem". AnimateDiff has no audio
path, so a haunted-lane bridge produces no stem to count.

So what this test protects is real but narrow: **on a genuine foley route**, the
count is legible at zero. Making a bridge's unpositioned status visible on a
NON-foley lane would be a different receipt at a different site, and no test
here claims it exists.
"""
from __future__ import annotations

import ast
import pathlib

from nodes import otr_master_audio_mux as mux


def _append_calls_for(needle: str):
    """Every ``report.append(...)`` whose literal text mentions ``needle``."""
    src = pathlib.Path(mux.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "append"):
            continue
        if needle in ast.unparse(node):
            found.append(node)
    return tree, found


def _guards_containing(tree, target: ast.AST):
    """The ``if`` statements that lexically enclose ``target``."""
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                if child is target:
                    guards.append(node)
    return guards


def test_the_counter_is_reported_at_every_value():
    tree, calls = _append_calls_for("foley_unpositioned")
    assert calls, (
        "the foley_unpositioned receipt line vanished entirely -- it is how a "
        "reader confirms how many beats owned no master-mix window")

    for call in calls:
        for guard in _guards_containing(tree, call):
            test_src = ast.unparse(guard.test)
            assert "unpositioned" not in test_src, (
                "foley_unpositioned is emitted only when `%s` -- so a reader "
                "who does not see the line cannot tell zero from not-logged. "
                "That ambiguity cost the 4060 a signal it was asked to confirm."
                % test_src)


def test_the_receipt_still_names_why_a_bridge_is_normal():
    """Removing the gate must not remove what the line told the reader."""
    src = pathlib.Path(mux.__file__).read_text(encoding="utf-8")
    i = src.find("foley_unpositioned")
    assert i > 0
    window = src[max(0, i - 400): i + 400]
    for phrase in ("music_inter", "no master-mix slot"):
        assert phrase in window, (
            "the receipt dropped %r -- a bare count reads as a fault when for "
            "a music bridge it is the normal case" % phrase)


def test_the_exception_counters_stay_conditional():
    """Scope guard: this change is about ONE invariant, not a logging sweep."""
    src = pathlib.Path(mux.__file__).read_text(encoding="utf-8")
    for counter in ("muted_samples", "skipped", "conform_notes"):
        assert 'if stats["%s"]:' % counter in src, (
            "stats[%r] stopped being conditional. Those are exceptions worth "
            "noticing, not invariants worth confirming at zero -- reporting "
            "them always would bury the one line that must always appear."
            % counter)
