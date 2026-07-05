"""Writer wires each helper with exactly the slot fn(s) it takes.

The S32 B1 "paired contract" had the writer pass BOTH `creative_fn=`
and `technical_fn=` to all four helpers. Sprint 2A/2D removed the
unused params, so each call site now carries only the slot kwargs its
helper accepts:
  - lock_cast         -- creative_fn only
  - compose_line      -- creative_fn only
  - build_news_briefs -- technical_fn only

pick_style was RETIRED by the 2026-07-05 style-engine consolidation:
style is now a pure sha256(cast_seed) deterministic draw
(build_story_contract()), with zero creative_fn/technical_fn calls --
there is no LLM-driven picker left to pin here.

Test:
* `test_writer_passes_expected_slot_kwargs` -- AST scan over
  `OTR_LedgerScriptWriter.py` asserts every helper call site carries
  exactly the slot kwargs that helper's signature accepts -- a
  tripwire against partial-port states.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
WRITER_PATH = REPO / "nodes" / "OTR_LedgerScriptWriter.py"
# Lane-enablement chunk 3 (2026-07-05): the build_news_briefs call site moved
# out of the writer into the source-payload contract wrapper -- the
# technical_fn pairing invariant survives at the NEW call site.
SOURCE_PAYLOAD_PATH = REPO / "nodes" / "_otr_source_payload.py"


# helper -> (the file that owns its call site, the slot kwargs its signature
# accepts post-Sprint 2A/2D). pick_style retired 2026-07-05 (style-engine
# consolidation) -- see module docstring.
_EXPECTED_KWARGS = {
    "lock_cast":         (WRITER_PATH, {"creative_fn"}),
    "compose_line":      (WRITER_PATH, {"creative_fn"}),
    "build_news_briefs": (SOURCE_PAYLOAD_PATH, {"technical_fn"}),
}


def test_writer_passes_expected_slot_kwargs():
    """For every call site (in the file that owns the helper's call) whose
    callable's terminal attribute is one of the four helpers, assert
    the slot kwargs present match exactly what the helper accepts --
    the expected kwarg is present, and the removed one is absent.
    """
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in {p for p, _ in _EXPECTED_KWARGS.values()}
    }

    failures: list[str] = []
    sites_seen: dict[str, int] = {n: 0 for n in _EXPECTED_KWARGS}

    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `_OTRSP.pick_style(...)`, `_OTRCAST.lock_cast(...)`, etc.
            if isinstance(func, ast.Attribute) and func.attr in _EXPECTED_KWARGS:
                helper = func.attr
            # Also catch bare `pick_style(...)` if anyone imports directly.
            elif isinstance(func, ast.Name) and func.id in _EXPECTED_KWARGS:
                helper = func.id
            else:
                continue
            owner, expected = _EXPECTED_KWARGS[helper]
            if path != owner:
                # Chunk 3: a helper call outside its owning file is a
                # partial-port state -- flag it.
                failures.append(
                    f"{path.name} line {node.lineno}: `{helper}(...)` call "
                    f"site found outside its owning file {owner.name}"
                )
                continue

            sites_seen[helper] += 1
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            slot_kwargs = kwarg_names & {"creative_fn", "technical_fn"}
            if slot_kwargs != expected:
                failures.append(
                    f"{path.name} line {node.lineno}: `{helper}(...)` slot "
                    f"kwargs {sorted(slot_kwargs)} != expected "
                    f"{sorted(expected)}"
                )

    # Every helper must have at least one call site in its owning file.
    not_called = [n for n, count in sites_seen.items() if count == 0]
    assert not not_called, (
        f"Missing call sites for {not_called}; the slot-kwarg "
        f"wiring can't be verified without them."
    )

    assert not failures, (
        "Each writer helper call must pass exactly the slot kwargs the "
        "helper's signature accepts (Sprint 2A/2D removed the unused "
        "S32 B1 paired params).\nOffenders:\n  "
        + "\n  ".join(failures)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
