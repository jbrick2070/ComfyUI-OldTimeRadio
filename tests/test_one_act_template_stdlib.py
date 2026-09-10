"""One-act shipped-template contract; no runtime, GPU or network imports."""
import importlib.util
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


def load_graph(path):
    return json.loads(path.read_text(encoding="utf-8"))


def writer_widgets(graph):
    writers = [node for node in graph["nodes"]
               if node["type"] == "OTR_LedgerScriptWriter"]
    if len(writers) != 1:
        raise AssertionError("Expected exactly one script writer")
    node = writers[0]
    names = [item["widget"]["name"] for item in node["inputs"]
             if item.get("widget")]
    values = node["widgets_values"]
    if len(names) != len(values) or len(set(names)) != len(names):
        raise AssertionError("Writer descriptor/value alignment changed")
    return dict(zip(names, values))


class OneActTemplateTests(unittest.TestCase):
    def test_canonical_starts_with_one_act_and_same_writers(self):
        values = writer_widgets(load_graph(REPO / "workflows/otr_canonical.json"))
        self.assertEqual(values["act_count"], "1")
        # WHICH model is authoritative in test_shipped_template_writer_default,
        # which ties it to DEFAULT_LLM. This module is deliberately stdlib-only
        # (see the docstring) so it must not import the catalog to learn the
        # label. It previously hard-coded 'google/gemma-4-12b-it (11.9 GB)';
        # when DEFAULT_LLM moved to Qwen the literal did not, so the assertion
        # pinned the drift in place rather than catching it (PBUG-20260906-09).
        # What belongs HERE is the structural half: both writer slots agree,
        # and the value is a real COMBO label carrying its size suffix.
        creative = values["creative_writing_model"]
        self.assertEqual(values["technical_model"], creative,
                         "both writer slots must select the same model")
        # The badge is `(<download> GB[, <tags>])` since 2026-09-09 -- the size
        # is the DOWNLOAD, and machine-fit tags may follow it (`mac16`,
        # `mac16-tight`, `nv8`, `gated`, ...). This pattern used to anchor
        # immediately after `GB)`, which made it a single-number parser that
        # rejects every current label; the structural claim it is really making
        # is "a real COMBO value carrying its size", not "exactly one number".
        self.assertRegex(creative, r"^\S+/\S+ \(\d+(\.\d+)? GB(, [\w\- ]+)?\)$",
                         "the size suffix is part of the COMBO value; a bare "
                         "repo id matches no choice and can resolve to index 0")
        self.assertEqual(len(values), 37)
