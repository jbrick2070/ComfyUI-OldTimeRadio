"""The LEMMY cameo knob has to REACH the lane runners, not merely be mapped.

WHY THIS FILE EXISTS. The widget shipped, the mapping dict was correct, and a
test pinned the mapping -- and the knob was still inert on every dispatched
lane, because `run()` never forwarded `lemmy_cameo` to `_resolve_inputs`. Only
the legacy in-line path saw it, by reading the `run()` local directly. A test
that checks the mapping dict passes happily while the operator's choice is
dropped on the floor, which is exactly what happened.

So these tests pin ARRIVAL, in two independent ways: the resolver puts the
tri-state in `resolved["lemmy_force"]`, and `run()` actually hands it over.
"""
import ast
import os

import pytest

from nodes.OTR_LedgerScriptWriter import (
    _LEMMY_CAMEO_CHOICES,
    _LEMMY_CAMEO_FORCE,
    _resolve_inputs,
)

_WRITER_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nodes", "OTR_LedgerScriptWriter.py",
)


@pytest.mark.parametrize("choice, expected", [
    ("roll (~11% chance)", None),
    ("always include", True),
    ("never include", False),
])
def test_the_resolver_lands_the_tri_state_under_lemmy_force(choice, expected):
    """Every consumer reads `resolved["lemmy_force"]`, so that is what the
    resolver owes them -- a tri-state, not the widget's display string."""
    resolved = _resolve_inputs(lemmy_cameo=choice)

    assert resolved["lemmy_force"] is expected


def test_the_default_is_the_natural_roll():
    """Omitting the knob must behave exactly as 'roll', or an old workflow with
    a short widgets_values vector would silently change the cameo rate."""
    assert _resolve_inputs()["lemmy_force"] is None


def test_run_forwards_the_cameo_choice_to_the_resolver():
    """THE REGRESSION GUARD. Parsed off disk rather than mocked, because the
    defect was a missing argument at one call site -- something a mock of
    `_resolve_inputs` would have papered over by accepting anything.
    """
    with open(_WRITER_SOURCE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=_WRITER_SOURCE)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_inputs"
    ]

    assert len(calls) == 1, (
        "expected exactly one _resolve_inputs call site in the writer; found "
        "%d -- a second one is a second place for the forwarding to be "
        "forgotten" % len(calls)
    )
    forwarded = {kw.arg for kw in calls[0].keywords}
    assert "lemmy_cameo" in forwarded, (
        "run() does not forward lemmy_cameo to _resolve_inputs -- the widget is "
        "inert on every dispatched lane"
    )


def test_the_choices_and_the_force_map_cannot_drift_apart():
    """Two spellings of the same three strings is how a dropdown starts
    offering an option the resolver then rejects."""
    assert list(_LEMMY_CAMEO_FORCE) == _LEMMY_CAMEO_CHOICES
