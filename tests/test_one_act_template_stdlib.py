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
        for slot in ("creative_writing_model", "technical_model"):
            self.assertEqual(values[slot], "google/gemma-4-12b-it (11.9 GB)")
        self.assertEqual(len(values), 33)

    def test_story_only_is_exactly_derived_and_one_act(self):
        spec = importlib.util.spec_from_file_location(
            "story_only_builder", REPO / "scripts/build_story_only.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = load_graph(REPO / "workflows/otr_story_only.json")
        self.assertEqual(graph, builder.build())
        self.assertEqual(writer_widgets(graph)["act_count"], "1")

    def test_generated_graphs_inherit_one_act(self):
        paths = sorted((REPO / "workflows/variants").glob("otr_*.json"))
        checked = 0
        for path in paths:
            if path.name.endswith(".env.json"):
                continue
            with self.subTest(graph=path.name):
                graph = load_graph(path)
                self.assertEqual(writer_widgets(graph)["act_count"], "1")
                checked += 1
        self.assertGreater(checked, 0, "No generated graphs were checked")


if __name__ == "__main__":
    unittest.main()
