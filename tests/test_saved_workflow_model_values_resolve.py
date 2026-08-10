"""A saved workflow's model values must exist in the LIVE dropdown.

WHY THIS FILE EXISTS -- it was predicted, then it happened.

The 2026-08-07 slug audit (`kibitz-runs/2026-08-07-slugfest/`) flagged that
`workflows/otr_canonical.json` hardcodes `"google/gemma-4-12b-it (11.9 GB)"` --
a SIZE-SUFFIXED label -- and warned that if the suffix ever changed without the
JSON being updated, saved workflows would stop matching the dropdown.

On 2026-08-09 exactly that happened to the OTHER hand-maintained graph:
`workflows/otr_story_only.json` carried the BARE id
`mistralai/Mistral-Nemo-Instruct-2407` while the live COMBO offered
`mistralai/Mistral-Nemo-Instruct-2407 (12.0 GB)`. The submit died with
`value_not_in_list` and the graph could not be run at all. The `--dry-run`
PASSED, so dry-run is not a substitute for this check.

WHY IT CAN DRIFT WITHOUT ANYONE EDITING A LABEL. The badge is not a literal; it
is `_estimate_resident_gb(repo_id)` rendered as `" (%.1f GB)"`
(`nodes/_otr_model_catalog.py:1189-1195`), and its own docstring says it
deliberately reports the SAME number the admission gate computes. So a change to
quantisation policy or to the estimator silently rewrites every label -- and
every saved graph pinned to the old spelling breaks at submit time, minutes into
a run, rather than here in two seconds.

SCOPE: the HAND-MAINTAINED graphs only. The 45 files under workflows/variants/
are generated and are already covered by `scripts/build_variants.py --check`,
which regenerates and diffs them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Hand-maintained graphs. Generated variants are deliberately excluded.
HAND_MAINTAINED = ("otr_canonical.json", "otr_story_only.json")

#: Writer widgets whose value must be a member of a live COMBO choice list.
MODEL_WIDGETS = ("creative_writing_model", "technical_model")


def _writer_input_spec():
    """The LIVE INPUT_TYPES for the writer node -- the same source ComfyUI
    validates a submitted prompt against."""
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
    spec = W.INPUT_TYPES()
    merged = {}
    for section in ("required", "optional"):
        merged.update(spec.get(section) or {})
    return merged


def _saved_widget_values(wf_name: str) -> dict[str, object]:
    """Map writer widget NAME -> saved value, using the positional
    widgets_values order that litegraph stores (BUG-LOCAL-097)."""
    path = REPO_ROOT / "workflows" / wf_name
    wf = json.loads(path.read_text(encoding="utf-8"))
    node = next((n for n in wf.get("nodes", [])
                 if n.get("type") == "OTR_LedgerScriptWriter"), None)
    assert node is not None, f"{wf_name}: no OTR_LedgerScriptWriter node"
    values = node.get("widgets_values") or []

    spec = _writer_input_spec()
    # Positional order = the order INPUT_TYPES declares widget-bearing inputs.
    names = [k for k, v in spec.items()
             if isinstance(v, tuple) and v and not (
                 isinstance(v[0], str) and v[0] in ("IMAGE", "LATENT", "MODEL"))]
    return {name: values[i] for i, name in enumerate(names) if i < len(values)}


@pytest.mark.parametrize("wf_name", HAND_MAINTAINED)
def test_saved_model_values_are_offered_by_the_live_dropdown(wf_name):
    """THE GUARD. Fails here in seconds instead of at submit, minutes in."""
    spec = _writer_input_spec()
    saved = _saved_widget_values(wf_name)

    problems = []
    for widget in MODEL_WIDGETS:
        if widget not in saved or widget not in spec:
            continue
        entry = spec[widget]
        choices = entry[0] if isinstance(entry, tuple) else None
        if not isinstance(choices, (list, tuple)):
            continue                      # not a COMBO on this build
        value = saved[widget]
        if value not in choices:
            problems.append(
                f"{widget}={value!r} is not in the live choice list. "
                f"Nearest offers: "
                f"{[c for c in choices if isinstance(c, str) and str(value).split(' (')[0] in c][:3]}"
            )
    assert not problems, (
        f"{wf_name} cannot be submitted as committed:\n  " +
        "\n  ".join(problems) +
        "\nThe dropdown badge is a COMPUTED estimate, not a literal "
        "(_otr_model_catalog.py:1189-1195), so it can drift without anyone "
        "editing the label. Re-save the graph with the spelling the live "
        "dropdown offers."
    )


@pytest.mark.parametrize("wf_name", HAND_MAINTAINED)
def test_the_writer_widget_count_matches_the_saved_positional_list(wf_name):
    """Positional-drift guard (BUG-LOCAL-097). If widgets_values is shorter or
    longer than the widget-bearing INPUT_TYPES entries, every value after the
    mismatch is silently reading the wrong widget -- which would also make the
    test above assert against the wrong field."""
    path = REPO_ROOT / "workflows" / wf_name
    wf = json.loads(path.read_text(encoding="utf-8"))
    node = next(n for n in wf.get("nodes", [])
                if n.get("type") == "OTR_LedgerScriptWriter")
    saved_len = len(node.get("widgets_values") or [])
    assert saved_len > 0, f"{wf_name}: writer has no widgets_values"
    # Both hand-maintained graphs are known to carry the same writer shape;
    # if one drifts from the other, that is the signal.
    other = [w for w in HAND_MAINTAINED if w != wf_name]
    for o in other:
        o_wf = json.loads((REPO_ROOT / "workflows" / o).read_text(encoding="utf-8"))
        o_node = next(n for n in o_wf.get("nodes", [])
                      if n.get("type") == "OTR_LedgerScriptWriter")
        o_len = len(o_node.get("widgets_values") or [])
        assert saved_len == o_len, (
            f"{wf_name} writer has {saved_len} widgets_values but {o} has "
            f"{o_len} -- one of them missed a widget append"
        )
