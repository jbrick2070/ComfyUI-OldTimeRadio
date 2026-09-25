"""apple/DEPENDENCIES.md is exactly what scripts/otr_dependency_doc.py writes.

The doc says "GENERATED ... Do not hand-edit" at the top, and nothing enforced
it. On 2026-09-25 a correct fix was made to the doc by hand (Sonnet QA on
6634fd37 caught it): the generator still emitted the false claim the fix
removed, so the next regeneration would have put it back without a sound.
Fix the generator, regenerate, commit both -- this test fails otherwise.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_committed_doc_is_the_generators_output():
    spec = importlib.util.spec_from_file_location(
        "otr_dependency_doc_parity", REPO / "scripts" / "otr_dependency_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render()
    committed = (REPO / "apple" / "DEPENDENCIES.md").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == rendered, (
        "apple/DEPENDENCIES.md differs from what scripts/otr_dependency_doc.py "
        "renders. Edit the GENERATOR, then run "
        "`python scripts/otr_dependency_doc.py` and commit both.")
