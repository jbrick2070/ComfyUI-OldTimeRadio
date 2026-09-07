"""The two SHIPPED templates must select the writer the code calls the default.

PBUG-20260906-09, found by the alpha.25 clean-install drill. `DEFAULT_LLM` was
changed to `Qwen/Qwen3.5-4B` in `_otr_model_catalog.py`, but
`workflows/otr_canonical.json` and `workflows/otr_story_only.json` still had
`google/gemma-4-12b-it (11.9 GB)` saved in both writer widgets. Nothing was
wrong with the constant and nothing was wrong with the graph; they simply
disagreed, and the graph is what runs. A first-run user pressing Run on the
shipped template therefore started a 23.9 GB download of a model the campaign
had already measured as the slower, larger option, instead of the ~4 GB one it
had settled on.

This is the exact failure mode CLAUDE.md section 0 exists for: "Code that is not
wired into this JSON is DEAD". A constant is not a decision until the graph
agrees with it.

WHY THIS RESOLVES SLOTS BY NAME. `widgets_values` is POSITIONAL, and asserting
on a hardcoded index 2/3 would itself rot the first time a widget is inserted
above it -- reporting success against the wrong slot, which is worse than no
test. `serialized_slot_names` is the same helper `build_variants.py` uses to map
saved values back to widget names, so this test drifts only when the real
mapping drifts.

Offline: builds schemas without a running server, reads no model, downloads
nothing.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nodes"))

WRITER_NODE = "OTR_LedgerScriptWriter"
MODEL_SLOTS = ("creative_writing_model", "technical_model")
SHIPPED = ("otr_canonical.json", "otr_story_only.json")


def _load(name):
    return json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))


def _writer_nodes(workflow):
    return [n for n in workflow["nodes"] if n.get("type") == WRITER_NODE]


class ShippedTemplateWriterTests(unittest.TestCase):
    """These two files are the entire first-run experience: they are what
    Browse Templates offers and what a stranger presses Run on."""

    @classmethod
    def setUpClass(cls):
        try:
            from _otr_model_catalog import DEFAULT_LLM, _strip_label_suffix
            from nodes._otr_workflow_apply import serialized_slot_names
            from scripts.build_variants import build_offline_schemas
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest("offline schema build unavailable: %s" % exc)
        cls.DEFAULT_LLM = DEFAULT_LLM
        cls._strip = staticmethod(_strip_label_suffix)
        cls.slots = serialized_slot_names(WRITER_NODE, build_offline_schemas())

    def _slot_index(self, slot_name):
        self.assertIn(slot_name, self.slots,
                      "%r is no longer a saved widget on %s; this test must be "
                      "updated deliberately, not deleted" % (slot_name, WRITER_NODE))
        return self.slots.index(slot_name)

    def test_both_shipped_templates_select_the_default_writer(self):
        for name in SHIPPED:
            workflow = _load(name)
            nodes = _writer_nodes(workflow)
            self.assertTrue(nodes, "%s has no %s node" % (name, WRITER_NODE))
            for node in nodes:
                values = node.get("widgets_values") or []
                for slot in MODEL_SLOTS:
                    index = self._slot_index(slot)
                    self.assertLess(index, len(values),
                                    "%s node %s has too few widget values"
                                    % (name, node["id"]))
                    saved = str(values[index])
                    self.assertEqual(
                        self._strip(saved), self.DEFAULT_LLM,
                        "%s node %s widget %r selects %r but DEFAULT_LLM is %r. "
                        "A constant the shipped graph disagrees with is dead: "
                        "the graph is what runs on a first-run install."
                        % (name, node["id"], slot, saved, self.DEFAULT_LLM))

    def test_the_saved_label_is_a_real_dropdown_option(self):
        """A value that is not an exact member of the dropdown fails validation
        at load, so the saved value must be the label verbatim, size suffix and
        all. The availability markers are stripped because they depend on what
        happens to be cached on the machine running the test, not on the graph."""
        from _otr_model_catalog import (LOCAL_GGUF_SUFFIX, LOCAL_HF_SUFFIX,
                                        NOT_DOWNLOADED_SUFFIX, dropdown_choices)

        def _bare(label):
            for suffix in (NOT_DOWNLOADED_SUFFIX, LOCAL_HF_SUFFIX,
                           LOCAL_GGUF_SUFFIX):
                if suffix and label.endswith(suffix):
                    return label[: -len(suffix)]
            return label

        options = {_bare(c) for c in dropdown_choices()}
        for name in SHIPPED:
            for node in _writer_nodes(_load(name)):
                values = node.get("widgets_values") or []
                for slot in MODEL_SLOTS:
                    saved = str(values[self._slot_index(slot)])
                    self.assertIn(
                        saved, options,
                        "%s selects %r which is not offered by the %r dropdown"
                        % (name, saved, slot))


if __name__ == "__main__":
    unittest.main()
