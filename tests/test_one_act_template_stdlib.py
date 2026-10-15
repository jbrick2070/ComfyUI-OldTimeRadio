"""One-act shipped-template contract; no runtime, GPU or network imports."""
import json
from pathlib import Path
import unittest

from tests._support.writer_slots import value, widget_names


REPO = Path(__file__).resolve().parents[1]

#: The writer's saved control count, measured against the canonical graph on
#: 2026-09-24 after two writer controls were removed with their backend
#: (see e5ff488b), and 36 since 2026-09-25 when `asset_cleanup` was appended
#: as the trailing control. This is a COUNT, not a
#: position -- it is the one number in this module that is allowed to be a
#: literal, and it changes only when a control is genuinely added or dropped.
#: Every VALUE below is found by widget NAME, so nothing here needs
#: renumbering when the writer's controls are reordered.
WRITER_WIDGET_COUNT = 38   # 2026-09-27: + the story bank roll pool


def load_graph(path):
    return json.loads(path.read_text(encoding="utf-8"))


def the_writer(graph):
    """The graph's single script writer, checked fit to be read by name.

    Names must be unique or reading a value by name silently takes the first of
    them. The descriptor/value alignment is the shared helper's own precondition
    -- ``value()`` refuses a node whose counts disagree -- so it is not repeated
    here.
    """
    writers = [node for node in graph["nodes"]
               if node["type"] == "OTR_LedgerScriptWriter"]
    if len(writers) != 1:
        raise AssertionError("Expected exactly one script writer")
    node = writers[0]
    names = widget_names(node)
    if len(set(names)) != len(names):
        raise AssertionError("Writer widget names are not unique")
    return node


class OneActTemplateTests(unittest.TestCase):
    def test_canonical_starts_with_one_act_and_same_writers(self):
        writer = the_writer(load_graph(REPO / "workflows/otr_canonical.json"))
        self.assertEqual(value(writer, "act_count"), "1")
        # WHICH model is authoritative in test_shipped_template_writer_default,
        # which ties it to DEFAULT_LLM. This module is deliberately stdlib-only
        # (see the docstring) so it must not import the catalog to learn the
        # label. It previously hard-coded 'google/gemma-4-12b-it (11.9 GB)';
        # when DEFAULT_LLM moved to Qwen the literal did not, so the assertion
        # pinned the drift in place rather than catching it (PBUG-20260906-09).
        # What belongs HERE is the structural half: both writer slots agree,
        # and the value is a real COMBO label -- since 2026-10-14 the bare id,
        # with no size badge (the size moved to the widget tooltip, so a
        # corrected size can never again change a saved value).
        creative = value(writer, "creative_writing_model")
        self.assertEqual(value(writer, "technical_model"), creative,
                         "both writer slots must select the same model")
        self.assertRegex(
            creative,
            # An HF `org/name` id or a ComfyUI-native `comfy_native:` id, bare.
            r"^[^\s()]+[/:][^\s()]+$",
            "the writer value is the bare id the dropdown offers; a retired "
            "size badge would match no choice")
        self.assertEqual(len(widget_names(writer)), WRITER_WIDGET_COUNT)
        self.assertEqual(len(writer["widgets_values"]), WRITER_WIDGET_COUNT)
