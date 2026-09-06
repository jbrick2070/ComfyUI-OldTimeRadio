"""PBUG-20260905-01: CPU exclusions must exist before NF4 conversion.

These contract tests need only stdlib (also discoverable by pytest). They
execute the production planner and load call boundary extracted from the AST;
no network, checkpoint data, GPU allocation, or global OTR reimport is needed.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "nodes" / "_otr_model_loader.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def production_function(name):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"ModelLoaderError": RuntimeError}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class ExplicitOffloadTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.config = SimpleNamespace(label="original")
        self.memory = {0: "6.8GiB", "cpu": "32GiB"}
        self.placement = {"model.layers.0": 0, "model.layers.1": "cpu"}
        self.inside_empty = False
        self.skeleton = SimpleNamespace(_no_split_modules={"ResidualBlock"})
        self.skeleton.tie_weights = lambda: self.events.append("tie")

        @contextmanager
        def empty(**kwargs):
            self.assertEqual(kwargs, {"include_buffers": True})
            self.inside_empty = True
            yield
            self.inside_empty = False

        def from_config(config, **kwargs):
            self.assertTrue(self.inside_empty)
            self.assertIsNot(config, self.config)
            config.label = "mutated skeleton config"
            self.events.append(("config", kwargs))
            return self.skeleton

        def infer(skeleton, **kwargs):
            self.assertIs(skeleton, self.skeleton)
            self.assertIn("tie", self.events)
            self.assertFalse(self.inside_empty)
            self.assertEqual(kwargs["max_memory"], self.memory)
            self.assertIsNot(kwargs["max_memory"], self.memory)
            self.assertEqual(kwargs["dtype"], "bf16")
            self.assertEqual(kwargs["no_split_module_classes"], ["ResidualBlock"])
            self.events.append("infer")
            return self.placement

        self.stubs = {
            "accelerate": SimpleNamespace(init_empty_weights=empty, infer_auto_device_map=infer),
            "transformers": SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_config=from_config)),
        }

    def plan(self):
        with patch.dict(sys.modules, self.stubs):
            return production_function("_plan_nf4_cpu_offload")(
                self.config, load_dtype="bf16", max_memory=self.memory, attn_impl="sdpa")

    def test_concrete_map_precedes_quantization_without_mutating_config_or_budget(self):
        self.assertEqual(self.plan(), self.placement)
        self.assertEqual(self.config.label, "original")
        self.assertEqual(self.events[0][1], {
            "trust_remote_code": False, "dtype": "bf16", "attn_implementation": "sdpa"})

    def test_invalid_or_absent_config_fails_loud(self):
        self.config = None
        with self.assertRaisesRegex(RuntimeError, "resolved model config"):
            self.plan()

    def test_invalid_planner_result_cannot_reenter_auto_path(self):
        for value in ("auto", {}, None):
            with self.subTest(value=value):
                self.placement = value
                with self.assertRaisesRegex(RuntimeError, "explicit device map"):
                    self.plan()

    def test_disk_and_all_cpu_plans_fail_before_weight_loading(self):
        for placement, error in (({"layer": "disk"}, "disk offload is not configured"),
                                 ({"": "cpu"}, "no modules on CUDA")):
            with self.subTest(placement=placement):
                self.placement = placement
                with self.assertRaisesRegex(RuntimeError, error):
                    self.plan()

    def test_first_load_and_flagship_are_unchanged_retry_gets_concrete_map(self):
        """Execute the actual production try/retry boundary, not a replica."""
        load = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_llm")
        boundary = next(n for n in ast.walk(load) if isinstance(n, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id == "ValueError" and h.name == "_dispatch_err"
            for h in n.handlers))
        calls = []
        plans = []
        initial_quant = object()
        common = {"device_map": "auto", "max_memory": self.memory, "quantization_config": initial_quant}
        token_model = object()

        def load_model(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1 and common["device_map"] == "auto":
                raise ValueError("Some modules are dispatched on the CPU or the disk")
            return token_model

        def planner(*args, **kwargs):
            plans.append(kwargs)
            return self.placement

        ns = {"AutoModelForCausalLM": SimpleNamespace(from_pretrained=load_model),
              "load_target": "local-snapshot", "model_config": self.config,
              "common_kwargs": common, "needs_4bit": True, "load_dtype": "bf16",
              "attn_impl": "sdpa", "_stripped_model_id": "test-model",
              "BitsAndBytesConfig": lambda **kw: kw, "torch": SimpleNamespace(bfloat16="bf16"),
              "_plan_nf4_cpu_offload": planner, "_runtime_log": lambda *a: None,
              "log": SimpleNamespace(warning=lambda *a: None),
              # PBUG-20260906-07: "test-model" is not a curated native-text
              # row, so it takes the composite path -- the parent config and
              # a plain copy of common_kwargs, i.e. exactly what this test
              # already asserts is unchanged.
              "_native_text_row": False,
              "_init_config": self.config,
              "_init_kwargs": dict(common),
              "_validate_native_text_loading_info": lambda info, **kw: (_ for _ in ()).throw(
                  AssertionError("composite row must not validate a native text load")),
              }
        code = compile(ast.Module(body=[boundary], type_ignores=[]), str(SOURCE), "exec")

        def run():
            # Production recomputes these from common_kwargs immediately before
            # the try, so refresh them here rather than reusing a stale copy --
            # this test deliberately mutates common_kwargs between execs.
            ns["_init_config"] = self.config
            ns["_init_kwargs"] = dict(common)
            exec(code, ns)

        run()
        self.assertIs(ns["model"], token_model)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["device_map"], "auto")
        self.assertIs(calls[0]["quantization_config"], initial_quant)
        self.assertEqual(calls[1]["device_map"], self.placement)
        self.assertTrue(calls[1]["quantization_config"]["llm_int8_enable_fp32_cpu_offload"])
        self.assertTrue(calls[1]["quantization_config"]["bnb_4bit_use_double_quant"])
        self.assertEqual(common["device_map"], "auto")
        self.assertEqual(len(plans), 1)
        calls.clear()
        plans.clear()
        common["device_map"] = {"": 0}  # actual >=14.5GiB production branch
        run()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["device_map"], {"": 0})
        self.assertEqual(plans, [])
        calls.clear()
        common["device_map"] = "auto"
        ns["AutoModelForCausalLM"] = SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append(kw) or token_model)
        run()
        self.assertEqual(len(calls), 1)
        self.assertEqual(plans, [])

        def refuse(*a, **kw):
            raise ValueError("unrelated configuration failure")
        ns["AutoModelForCausalLM"] = SimpleNamespace(from_pretrained=refuse)
        with self.assertRaisesRegex(ValueError, "unrelated configuration failure"):
            run()
        self.assertEqual(plans, [])
        ns["AutoModelForCausalLM"] = SimpleNamespace(from_pretrained=load_model)
        ns["needs_4bit"] = False
        calls.clear()
        with self.assertRaisesRegex(ValueError, "dispatched on the CPU"):
            exec(code, ns)
        self.assertEqual(plans, [])

    def test_8gb_and_5080_budget_and_device_selection_stay_hardware_scoped(self):
        """Run the real selection branches for NF4, without importing a GPU stack."""
        planner = production_function("_plan_max_memory")
        load = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_llm")
        selections = []
        for expression in ("max_memory is not None", "total_vram >= 14.5"):
            expected = ast.dump(ast.parse(expression, mode="eval").body)
            matches = [n for n in ast.walk(load)
                       if isinstance(n, ast.If) and ast.dump(n.test) == expected]
            self.assertEqual(len(matches), 1)
            selections.append(matches[0])
        # The flagship branch is nested inside quant_config is not None;
        # this test's explicit NF4 case satisfies that enclosing condition.
        code = compile(ast.Module(body=selections, type_ignores=[]), str(SOURCE), "exec")
        for total, expected_budget, expected_map in (
            (8.00, {0: "6.8GiB", "cpu": "32GiB"}, "auto"),
            (15.99, {0: "13.5GiB", "cpu": "32GiB"}, {"": 0}),
        ):
            with self.subTest(total_vram=total):
                budget = planner("google/gemma-4-12b-it", total,
                                 cuda_available=True, quant_policy="bnb_nf4")
                self.assertEqual(budget, expected_budget)
                ns = {"max_memory": budget, "common_kwargs": {}, "total_vram": total,
                      "_stripped_model_id": "google/gemma-4-12b-it",
                      "_runtime_log": lambda *a: None}
                exec(code, ns)
                self.assertEqual(ns["common_kwargs"]["max_memory"], expected_budget)
                self.assertEqual(ns["common_kwargs"]["device_map"], expected_map)

    def test_ordinary_cpu_modules_are_allowed_but_cpu_or_meta_nf4_still_rejected(self):
        scan = production_function("_bug098_scan_linear4bit_devices")
        linear = type("Linear4bit", (), {"__module__": "bitsandbytes.nn.modules"})
        quantized = linear()
        ordinary = SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type="cpu")))
        for device, rejected in (("cuda", False), ("cpu", True), ("meta", True)):
            with self.subTest(device=device):
                quantized.weight = SimpleNamespace(device=SimpleNamespace(type=device))
                model = SimpleNamespace(named_modules=lambda: [("gpu", quantized), ("cpu", ordinary)])
                count, off_cuda = scan(model)
                self.assertEqual(count, 1)
                self.assertEqual(bool(off_cuda), rejected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
