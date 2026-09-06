"""Live visual-asset gate contracts without importing OTR or model libraries.

Execute only the production validator methods extracted from its AST. Every
relative import is a stdlib stub: no model imports, GPU use, or network calls.
"""
import ast
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes" / "_otr_workflow_validator.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef)
             and node.name == "WorkflowValidator")


class VisualAssetValidatorTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.prompt = {"87": {"class_type": "OTR_VideoDirector"}}
        self.contract_error = None
        self.asset_error = None
        self.drift = []
        self.workflow = {"nodes": [{"id": 1}], "links": []}
        self.path_calls = []

        def resolve_path(value):
            self.path_calls.append(value)
            return SimpleNamespace(is_file=lambda: True,
                                   stat=lambda: SimpleNamespace(st_mtime_ns=123))

        def load_workflow(value):
            self.events.append("load")
            return self.workflow

        def validate_contract(workflow, mapping, **kwargs):
            self.events.append("contract")
            if self.contract_error:
                raise self.contract_error

        def widget_drift(workflow, mapping):
            self.events.append("drift")
            return self.drift

        def ensure_assets(prompt, unique_id):
            self.events.append(("assets", prompt, unique_id))
            if self.asset_error:
                raise self.asset_error

        package = ModuleType("_asset_validator_seam")
        package.__path__ = []
        package.NODE_CLASS_MAPPINGS = {}
        nodes = ModuleType("_asset_validator_seam.nodes")
        nodes.__path__ = []
        validation = ModuleType("_asset_validator_seam.nodes._workflow_validation")
        validation.validate_workflow_contract = validate_contract
        assets = ModuleType("_asset_validator_seam.nodes._otr_visual_assets")
        assets.ensure_prompt_visual_assets = ensure_assets
        self.import_stubs = patch.dict(sys.modules, {
            package.__name__: package, nodes.__name__: nodes,
            validation.__name__: validation, assets.__name__: assets,
        })
        self.import_stubs.start()
        self.addCleanup(self.import_stubs.stop)

        methods = [node for node in CLASS.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"INPUT_TYPES", "IS_CHANGED", "validate"}]
        isolated_class = ast.ClassDef(name="OTR_WorkflowValidator", bases=[],
                                     keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[isolated_class], type_ignores=[]))
        namespace = {
            "__package__": "_asset_validator_seam.nodes",
            "_DEFAULT_WORKFLOW_PATH": ROOT / "workflows" / "otr_canonical.json",
            "_resolve_workflow_path": resolve_path,
            "_load_workflow": load_workflow,
            "widget_vector_drift": widget_drift,
            "log": SimpleNamespace(info=lambda *args: None, error=lambda *args: None),
        }
        exec(compile(module, str(SOURCE), "exec"), namespace)
        self.cls = namespace["OTR_WorkflowValidator"]
        self.node = self.cls()
        self.node._assert_stamp = lambda *args: self.events.append("stamp") or "stamp OK"

    def validate(self, enabled=True, **kwargs):
        return self.node.validate("canonical.json", enabled, True,
                                  prompt=self.prompt, unique_id="63", **kwargs)

    def test_hidden_context_adds_no_widgets(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertEqual(inputs["hidden"], {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"})
        self.assertEqual(list(inputs["required"]),
                         ["workflow_json_path", "validate_anyway", "strict_unknown_types"])
        self.assertEqual(list(inputs["optional"]), ["profile_id", "master_hash", "generated_by"])

    def test_live_cache_rechecks_even_when_hidden_prompt_is_empty(self):
        for unique_id in ("63", 63, "0"):
            with self.subTest(unique_id=unique_id):
                value = self.cls.IS_CHANGED("canonical.json", True, True,
                                            prompt={}, unique_id=unique_id)
                self.assertTrue(math.isnan(value))
        self.assertEqual(self.path_calls, [])

    def test_legacy_cache_fingerprint_is_unchanged(self):
        for unique_id in (None, "", " "):
            with self.subTest(unique_id=unique_id):
                self.assertEqual(self.cls.IS_CHANGED("canonical.json", True, False,
                                 "profile", "hash", "generator", prompt=self.prompt,
                                 unique_id=unique_id),
                                 "canonical.json|123|True|False|profile|hash|generator")

    def test_structural_checks_precede_assets_and_success(self):
        result = self.validate(profile_id="profile")
        self.assertEqual(self.events[:4], ["stamp", "load", "contract", "drift"])
        self.assertEqual(self.events[4], ("assets", self.prompt, "63"))
        self.assertIn("OTR_WorkflowValidator: OK", result[0])

    def test_structural_failure_never_calls_asset_helper(self):
        self.contract_error = ValueError("contract rejected")
        with self.assertRaisesRegex(ValueError, "contract rejected"):
            self.validate()
        self.assertEqual(self.events, ["load", "contract"])

    def test_widget_drift_never_calls_asset_helper(self):
        self.drift = ["saved slots differ"]
        with self.assertRaisesRegex(ValueError, "widget-vector contract drift"):
            self.validate()
        self.assertEqual(self.events, ["load", "contract", "drift"])

    def test_structural_bypass_still_gates_assets(self):
        result = self.validate(False, profile_id="profile")
        self.assertEqual(self.events, ["stamp", ("assets", self.prompt, "63")])
        self.assertIn("contract check skipped", result[0])

    def test_asset_error_and_cancellation_propagate_on_both_paths(self):
        class Cancelled(BaseException):
            pass

        for enabled in (True, False):
            for error in (RuntimeError("assets unavailable"), Cancelled("cancelled")):
                with self.subTest(enabled=enabled, error=type(error).__name__):
                    self.asset_error = error
                    with self.assertRaises(type(error)) as raised:
                        self.validate(enabled)
                    self.assertIs(raised.exception, error)

    def test_legacy_direct_call_forwards_absent_context_to_helper(self):
        self.node.validate("canonical.json", True, True)
        self.assertEqual(self.events[-1], ("assets", None, None))

    def test_canonical_retains_validator_writer_and_director_gate_edges(self):
        import json
        workflow = json.loads((ROOT / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[63]["type"], "OTR_WorkflowValidator")
        self.assertEqual(len(nodes[63]["widgets_values"]), 6)
        self.assertEqual(nodes[1]["type"], "OTR_LedgerScriptWriter")
        self.assertEqual(nodes[87]["type"], "OTR_VideoDirector")
        for link_id, target_id in ((279, 1), (269, 87)):
            link = next(link for link in workflow["links"] if link[0] == link_id)
            self.assertEqual(link[1:4], [63, 0, target_id])
            self.assertEqual(nodes[target_id]["inputs"][link[4]]["name"], "gate_in")
            self.assertEqual(nodes[target_id]["inputs"][link[4]]["link"], link_id)


if __name__ == "__main__":
    unittest.main()
