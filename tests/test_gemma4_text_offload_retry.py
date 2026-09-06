"""Exact-E4B text-only NF4 retry contracts, without importing a model stack.

Only selected real production functions/class and the real ValueError retry
boundary are compiled from the AST. Model, planner and quantizer objects are
inert stdlib fakes; no OTR/torch/transformers import, weights or network access.
"""
import ast
import builtins
from copy import deepcopy
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "nodes" / "_otr_model_loader.py"
E4B = "google/gemma-4-E4B-it"
REPORT_FIELDS = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
MM_PREFIXES = ("model.audio_tower.", "model.vision_tower.",
               "model.embed_audio.", "model.embed_vision.")


def stdlib_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.partition(".")[0] not in sys.stdlib_module_names:
        raise AssertionError("Non-stdlib import attempted by extracted code: " + name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def production_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = ("ModelLoaderError", "_e4b_text_offload_config",
             "_validate_e4b_text_loading_info")
    nodes = []
    for name in names:
        matches = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        if len(matches) != 1:
            raise AssertionError("Expected one real production definition: " + name)
        nodes.append(matches[0])
    namespace = {"__builtins__": dict(vars(builtins), __import__=stdlib_import)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return tree, namespace


def real_retry_boundary(tree):
    load = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "load_llm")
    matches = [node for node in ast.walk(load) if isinstance(node, ast.Try) and any(
        isinstance(handler.type, ast.Name) and handler.type.id == "ValueError"
        and handler.name == "_dispatch_err" for handler in node.handlers)]
    if len(matches) != 1:
        raise AssertionError("Expected one real NF4 dispatch-refusal retry boundary")
    return compile(ast.Module(body=matches, type_ignores=[]), str(SOURCE), "exec")


def valid_config():
    return SimpleNamespace(
        model_type="gemma4", tie_word_embeddings=True,
        text_config=SimpleNamespace(
            model_type="gemma4_text", tie_word_embeddings=True,
            max_position_embeddings=8192, layer_types=["sliding", "full"],
            rope_parameters={"full": {"factor": 1.0}}),
        vision_config=SimpleNamespace(hidden_size=99))


def valid_info():
    return {field: [] for field in REPORT_FIELDS}


class E4BConfigTests(unittest.TestCase):
    def setUp(self):
        _, self.ns = production_namespace()
        self.select = self.ns["_e4b_text_offload_config"]
        self.error = self.ns["ModelLoaderError"]

    def test_returns_native_text_deepcopy_without_mutating_parent(self):
        config = valid_config()
        before = deepcopy(config)
        selected = self.select(config)
        self.assertIsNot(selected, config.text_config)
        self.assertEqual(selected, config.text_config)
        self.assertEqual(selected.model_type, "gemma4_text")
        self.assertTrue(selected.tie_word_embeddings)
        selected.layer_types.append("mutated")
        selected.rope_parameters["full"]["factor"] = 42
        selected.max_position_embeddings = 1
        self.assertEqual(config, before)

    def test_missing_or_wrong_parent_architecture_fails_closed(self):
        for config in (None, {}, "gemma4", SimpleNamespace(),
                       SimpleNamespace(model_type="gemma3"),
                       SimpleNamespace(model_type="gemma4_text")):
            with self.subTest(config=config), self.assertRaises(self.error):
                self.select(config)

    def test_missing_malformed_or_wrong_text_config_fails_closed(self):
        for text in (None, {}, "gemma4_text", SimpleNamespace(),
                     SimpleNamespace(model_type="gemma4"),
                     SimpleNamespace(model_type="gemma3_text")):
            config = valid_config()
            config.text_config = text
            with self.subTest(text=text), self.assertRaises(self.error):
                self.select(config)
        config = valid_config()
        del config.text_config
        with self.assertRaises(self.error):
            self.select(config)

    def test_both_parent_and_text_embedding_ties_must_be_true(self):
        for target in ("parent", "text"):
            for value in (False, None, 0, 1, "true"):
                config = valid_config()
                owner = config if target == "parent" else config.text_config
                owner.tie_word_embeddings = value
                with self.subTest(target=target, value=value), self.assertRaises(self.error):
                    self.select(config)
            config = valid_config()
            owner = config if target == "parent" else config.text_config
            del owner.tie_word_embeddings
            with self.subTest(target=target, missing=True), self.assertRaises(self.error):
                self.select(config)


class E4BLoadingInfoTests(unittest.TestCase):
    def setUp(self):
        _, ns = production_namespace()
        self.validate = ns["_validate_e4b_text_loading_info"]

    def test_empty_native_report_containers_are_accepted(self):
        for container in (list, tuple, set, frozenset):
            with self.subTest(container=container.__name__):
                self.validate({field: container() for field in REPORT_FIELDS})

    def test_only_exact_multimodal_prefixes_are_allowed(self):
        info = valid_info()
        info["unexpected_keys"] = {prefix + "weight" for prefix in MM_PREFIXES}
        before = deepcopy(info)
        self.validate(info)
        self.assertEqual(info, before)

    def test_unexpected_text_shared_kv_and_prefix_lookalikes_fail(self):
        for key in ("model.layers.0.weight", "model.language_model.layers.0.weight",
                    "model.layers.0.self_attn.k_proj.weight", "lm_head.weight",
                    "audio_tower.weight", "module.model.audio_tower.weight",
                    "model.audio_tower", "model.audio_tower_bad.weight",
                    "model.audio_tower.", "model.vision_tower.",
                    "model.embed_audio.", "model.embed_vision.",
                    "model.vision_tower_extra.weight", "model.embed_audio_extra.weight",
                    "model.embed_vision_extra.weight", "model.audio_towerXweight",
                    "MODEL.audio_tower.weight", ""):
            info = valid_info()
            info["unexpected_keys"] = [key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate(info)

    def test_missing_mismatch_and_error_details_always_fail(self):
        for field, details in (
            ("missing_keys", ["model.layers.0.weight"]),
            ("missing_keys", ["model.vision_tower.weight"]),
            ("mismatched_keys", [("model.layers.0.weight", (2, 3), (4, 5))]),
            ("error_msgs", ["native checkpoint conversion failed"]),
        ):
            info = valid_info()
            info[field] = details
            with self.subTest(field=field, details=details), self.assertRaises(RuntimeError):
                self.validate(info)

    def test_absent_or_malformed_report_is_not_a_clean_load(self):
        for info in (None, [], (), "", "clean", 0, SimpleNamespace()):
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                self.validate(info)
        for omitted in REPORT_FIELDS:
            info = valid_info()
            del info[omitted]
            with self.subTest(omitted=omitted), self.assertRaises(RuntimeError):
                self.validate(info)

    def test_required_containers_reject_none_strings_and_mappings(self):
        for field in REPORT_FIELDS:
            for bad in (None, "", "a.weight", {}, {"weight": "bad"}, 0):
                info = valid_info()
                info[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(RuntimeError):
                    self.validate(info)

    def test_nonstring_key_and_error_entries_raise_runtime_error(self):
        for field in ("missing_keys", "unexpected_keys", "error_msgs"):
            for bad in (None, 12, ("model.audio_tower.weight",), {"weight": 1}):
                info = valid_info()
                info[field] = [bad]
                with self.subTest(field=field, bad=bad), self.assertRaises(RuntimeError):
                    self.validate(info)

    def test_future_conversion_errors_never_silently_pass(self):
        for error in (["failure"], {"weight": "failed"}, "failed", ("failed",)):
            info = valid_info()
            info["conversion_errors"] = error
            with self.subTest(error=error), self.assertRaises(RuntimeError):
                self.validate(info)
        for empty in (None, [], {}, ()):
            info = valid_info()
            info["conversion_errors"] = empty
            self.validate(info)


class E4BRetryBoundaryTests(unittest.TestCase):
    def setUp(self):
        tree, self.ns = production_namespace()
        self.code = real_retry_boundary(tree)
        self.config = valid_config()
        self.original_config = deepcopy(self.config)
        self.initial_quant = object()
        self.common = {"device_map": "auto", "max_memory": {0: "6.8GiB", "cpu": "32GiB"},
                       "quantization_config": self.initial_quant, "trust_remote_code": False}
        self.placement = {"model.layers.0": 0, "model.layers.1": "cpu"}
        self.model = object()
        self.report = valid_info()
        self.failure = ValueError("Some modules are dispatched on the CPU or the disk")
        self.calls, self.plans, self.validation_calls = [], [], []
        self.events = []
        self.retry_failure = None
        self.planner_failure = None
        real_validate = self.ns["_validate_e4b_text_loading_info"]

        def load(*args, **kwargs):
            self.events.append("load")
            self.calls.append((args, kwargs))
            if len(self.calls) == 1 and self.failure is not None:
                raise self.failure
            if len(self.calls) > 1 and self.retry_failure is not None:
                raise self.retry_failure
            if kwargs.get("output_loading_info"):
                return self.model, self.report
            return self.model

        def plan(config, **kwargs):
            self.events.append("plan")
            self.plans.append((config, kwargs))
            if self.planner_failure is not None:
                raise self.planner_failure
            return self.placement

        def validate(info):
            self.events.append("validate")
            self.validation_calls.append(info)
            # The actual variable owning cleanup must exist BEFORE validation.
            self.assertIs(self.ns.get("model"), self.model)
            return real_validate(info)

        self.ns.update({
            "AutoModelForCausalLM": SimpleNamespace(from_pretrained=load),
            "load_target": "fake-local-snapshot", "model_config": self.config,
            "common_kwargs": self.common, "needs_4bit": True, "load_dtype": "bf16",
            "attn_impl": "sdpa", "_stripped_model_id": E4B,
            "BitsAndBytesConfig": lambda **kwargs: kwargs,
            "torch": SimpleNamespace(bfloat16="bf16"),
            "_plan_nf4_cpu_offload": plan,
            "_validate_e4b_text_loading_info": validate,
            "_runtime_log": lambda *args: None,
            "log": SimpleNamespace(warning=lambda *args: None),
        })

    def execute(self):
        exec(self.code, self.ns)

    def test_exact_e4b_retry_uses_same_copied_native_text_config_for_plan_and_load(self):
        self.execute()
        self.assertIs(self.ns["model"], self.model)
        self.assertEqual(self.events, ["load", "plan", "load", "validate"])
        self.assertEqual(len(self.calls), 2)
        first, retry = self.calls[0][1], self.calls[1][1]
        self.assertIs(first["config"], self.config)
        self.assertIs(first["quantization_config"], self.initial_quant)
        self.assertEqual(first["device_map"], "auto")
        self.assertNotIn("key_mapping", first)
        self.assertNotIn("output_loading_info", first)
        copied_config = self.plans[0][0]
        self.assertIs(retry["config"], copied_config)
        self.assertIsNot(copied_config, self.config.text_config)
        self.assertEqual(copied_config, self.config.text_config)
        self.assertEqual(self.config, self.original_config)
        self.assertEqual(retry["device_map"], self.placement)
        self.assertTrue(retry["output_loading_info"])
        self.assertEqual(retry["key_mapping"], {r"^model\.language_model\.": "model."})
        self.assertEqual(self.plans[0][1], {
            "load_dtype": "bf16", "max_memory": self.common["max_memory"], "attn_impl": "sdpa"})
        self.assertEqual(self.calls[0][0], ("fake-local-snapshot",))
        self.assertEqual(self.calls[1][0], ("fake-local-snapshot",))
        self.assertTrue(first["local_files_only"])
        self.assertTrue(retry["local_files_only"])
        self.assertFalse(retry["trust_remote_code"])
        self.assertTrue(retry["quantization_config"]["load_in_4bit"])
        self.assertTrue(retry["quantization_config"]["llm_int8_enable_fp32_cpu_offload"])
        self.assertTrue(retry["quantization_config"]["bnb_4bit_use_double_quant"])
        self.assertEqual(retry["quantization_config"]["bnb_4bit_quant_type"], "nf4")
        self.assertIs(self.common["quantization_config"], self.initial_quant)
        self.assertEqual(self.common["device_map"], "auto")
        self.assertNotIn("key_mapping", self.common)
        self.assertNotIn("output_loading_info", self.common)

    def test_actual_retry_key_mapping_is_anchored_and_does_not_remap_multimodal(self):
        self.execute()
        mapping = self.calls[1][1]["key_mapping"]
        self.assertEqual(len(mapping), 1)
        pattern, replacement = next(iter(mapping.items()))
        self.assertEqual(re.sub(pattern, replacement, "model.language_model.layers.0.weight"),
                         "model.layers.0.weight")
        for key in ("prefix.model.language_model.layers.0.weight", "modelXlanguage_modelXweight",
                    "model.audio_tower.weight", "model.embed_vision.weight", "lm_head.weight"):
            with self.subTest(key=key):
                self.assertEqual(re.sub(pattern, replacement, key), key)

    def test_initial_success_keeps_full_config_and_never_enters_retry(self):
        self.failure = None
        self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0][1]["config"], self.config)
        self.assertIs(self.ns["model"], self.model)
        self.assertEqual(self.plans, [])
        self.assertEqual(self.validation_calls, [])
        self.assertNotIn("key_mapping", self.calls[0][1])
        self.assertNotIn("output_loading_info", self.calls[0][1])

    def test_all_gpu_initial_success_is_unchanged(self):
        self.failure = None
        self.common["device_map"] = {"": 0}
        self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]["device_map"], {"": 0})
        self.assertIs(self.calls[0][1]["config"], self.config)
        self.assertEqual(self.plans, [])
        self.assertEqual(self.validation_calls, [])

    def test_non_e4b_retries_keep_original_full_config_and_no_key_mapping(self):
        for model_id in ("google/gemma-4-12b-it", "google/gemma-4-E2B-it",
                         "google/gemma-4-e4b-it", "google/gemma-4-E4B-it-extra",
                         "other/gemma-4-E4B-it", "google/gemma-4-E4B-it "):
            with self.subTest(model_id=model_id):
                self.calls.clear()
                self.plans.clear()
                self.validation_calls.clear()
                self.ns["_stripped_model_id"] = model_id
                self.execute()
                self.assertEqual(len(self.calls), 2)
                self.assertIs(self.plans[0][0], self.config)
                self.assertIs(self.calls[1][1]["config"], self.config)
                self.assertNotIn("key_mapping", self.calls[1][1])
                self.assertNotIn("output_loading_info", self.calls[1][1])
                self.assertEqual(self.validation_calls, [])

    def test_non_nf4_refusal_is_not_retried(self):
        self.ns["needs_4bit"] = False
        with self.assertRaises(ValueError) as caught:
            self.execute()
        self.assertIs(caught.exception, self.failure)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.plans, [])
        self.assertEqual(self.validation_calls, [])

    def test_other_first_load_errors_propagate_without_retry(self):
        for error in (ValueError("unrelated config failure"), RuntimeError("CUDA out of memory"),
                      OSError("checkpoint unreadable")):
            self.calls.clear()
            self.failure = error
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                self.execute()
            self.assertIs(caught.exception, error)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.plans, [])
            self.assertEqual(self.validation_calls, [])

    def test_invalid_text_config_stops_before_planning_or_second_load(self):
        self.config.text_config.tie_word_embeddings = False
        with self.assertRaises(self.ns["ModelLoaderError"]):
            self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.plans, [])
        self.assertEqual(self.validation_calls, [])

    def test_planner_failure_stops_before_second_load(self):
        self.planner_failure = RuntimeError("explicit map invalid")
        with self.assertRaises(RuntimeError) as caught:
            self.execute()
        self.assertIs(caught.exception, self.planner_failure)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.validation_calls, [])

    def test_retry_loading_failure_is_not_retried_again(self):
        self.retry_failure = RuntimeError("checkpoint load failed")
        with self.assertRaises(RuntimeError) as caught:
            self.execute()
        self.assertIs(caught.exception, self.retry_failure)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.plans), 1)
        self.assertEqual(self.validation_calls, [])

    def test_loading_info_failure_occurs_after_model_assignment_for_cleanup(self):
        self.report["unexpected_keys"] = ["model.layers.0.weight"]
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertIs(self.ns["model"], self.model)
        self.assertEqual(self.validation_calls, [self.report])
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
