"""Native text-decoder loading for multimodal rows driven text-only.

PBUG-20260906-07. A one-act trial on a physical 8 GB RTX 4060 failed after
193.79s with every off-CUDA module named `model.vision_tower.*`: OTR was
loading a multimodal checkpoint whole and dispatching the towers it never
executes to the CPU, which then tripped the BUG-098 materialization guard.
Measured from gemma-4-E2B-it's own checkpoint header, 1410 of its 2011 tensors
are audio and vision towers; only 600 are the text decoder.

The load path is exercised by executing the REAL production boundary extracted
from the AST, not a replica. Pure stdlib: no torch, no OTR package import, no
weights, no network, no GPU.
"""
import ast
import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes" / "_otr_model_loader.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))

HELPERS = (
    "_native_text_load_config",
    "_registry_supplies_text_prefix_mapping",
    "_validate_native_text_loading_info",
)


def stdlib_import(name, *args, **kwargs):
    """Allow only the stdlib copy this module actually needs."""
    if name.split(".")[0] not in {"copy"}:
        raise AssertionError(f"unexpected import in production helper: {name}")
    return builtins.__import__(name, *args, **kwargs)


def production_helpers():
    nodes = []
    for name in HELPERS:
        matches = [n for n in TREE.body
                   if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one production def: {name}")
        nodes.append(matches[0])
    ns = {
        "__builtins__": dict(vars(builtins), __import__=stdlib_import),
        "ModelLoaderError": RuntimeError,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def load_catalog():
    path = ROOT / "nodes" / "_otr_model_catalog.py"
    spec = importlib.util.spec_from_file_location("_nt_catalog", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_nt_catalog"] = module
    spec.loader.exec_module(module)
    return module


NS = production_helpers()
CATALOG = load_catalog()


def gemma_config():
    return SimpleNamespace(
        model_type="gemma4",
        tie_word_embeddings=True,
        text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=2048),
        vision_config=SimpleNamespace(hidden_size=99),
        audio_config=SimpleNamespace(hidden_size=77),
    )


def qwen_config():
    return SimpleNamespace(
        model_type="qwen3_5",
        tie_word_embeddings=True,
        text_config=SimpleNamespace(model_type="qwen3_5_text", hidden_size=2560),
        vision_config=SimpleNamespace(hidden_size=1024),
    )


class NativeTextConfigTests(unittest.TestCase):
    def test_returns_the_text_config_for_both_verified_families(self):
        for factory, expected in ((gemma_config, "gemma4_text"),
                                  (qwen_config, "qwen3_5_text")):
            with self.subTest(family=expected):
                parent = factory()
                text = NS["_native_text_load_config"](parent)
                self.assertEqual(text.model_type, expected)

    def test_result_is_a_copy_so_the_parent_config_is_never_mutated(self):
        parent = gemma_config()
        text = NS["_native_text_load_config"](parent)
        self.assertIsNot(text, parent.text_config)
        text.hidden_size = 1
        self.assertEqual(parent.text_config.hidden_size, 2048)

    def test_missing_text_config_fails_loud_instead_of_falling_back(self):
        """A silent fallback here would reload the towers -- the whole bug."""
        parent = SimpleNamespace(model_type="gemma4", text_config=None)
        with self.assertRaises(RuntimeError) as caught:
            NS["_native_text_load_config"](parent)
        self.assertIn("cannot be split", str(caught.exception))

    def test_text_config_without_model_type_fails_loud(self):
        parent = SimpleNamespace(
            model_type="gemma4", text_config=SimpleNamespace(hidden_size=1))
        with self.assertRaises(RuntimeError):
            NS["_native_text_load_config"](parent)

    def test_no_family_allowlist_is_hardcoded(self):
        """Opting in is the catalog's job; a second ladder here would rot."""
        parent = SimpleNamespace(
            model_type="some_future_vlm",
            text_config=SimpleNamespace(model_type="some_future_vlm_text"))
        self.assertEqual(
            NS["_native_text_load_config"](parent).model_type,
            "some_future_vlm_text")


class LoadingInfoValidationTests(unittest.TestCase):
    def report(self, **overrides):
        base = {"missing_keys": [], "unexpected_keys": [],
                "mismatched_keys": [], "error_msgs": []}
        base.update(overrides)
        return base

    def test_missing_keys_are_rejected(self):
        """Transformers randomly initializes absent weights; a model that
        loaded nothing generates fluent nonsense instead of failing."""
        with self.assertRaises(RuntimeError) as caught:
            NS["_validate_native_text_loading_info"](
                self.report(missing_keys=["model.layers.0.self_attn.q_proj.weight"]),
                model_id="google/gemma-4-E2B-it")
        self.assertIn("randomly initialized", str(caught.exception))

    def test_mismatched_and_error_fields_are_rejected(self):
        for field in ("mismatched_keys", "error_msgs"):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                NS["_validate_native_text_loading_info"](
                    self.report(**{field: ["bad"]}), model_id="x")

    def test_conversion_errors_are_rejected(self):
        with self.assertRaises(RuntimeError):
            NS["_validate_native_text_loading_info"](
                self.report(conversion_errors=["boom"]), model_id="x")

    def test_dropped_towers_are_expected_and_summarized(self):
        dropped = NS["_validate_native_text_loading_info"](
            self.report(unexpected_keys=[
                "model.vision_tower.encoder.layers.0.weight",
                "model.vision_tower.patch_embedder.weight",
                "model.audio_tower.encoder.layers.3.weight",
                "model.embed_vision.weight",
            ]),
            model_id="google/gemma-4-E2B-it")
        self.assertEqual(
            dropped,
            ["model.audio_tower", "model.embed_vision", "model.vision_tower"])

    def test_incomplete_or_malformed_report_is_rejected(self):
        for bad in (None, {}, {"missing_keys": []},
                    {"missing_keys": "x", "unexpected_keys": [],
                     "mismatched_keys": [], "error_msgs": []}):
            with self.subTest(report=bad), self.assertRaises(RuntimeError):
                NS["_validate_native_text_loading_info"](bad, model_id="x")


class RegistryMappingTests(unittest.TestCase):
    def test_answers_a_bool_for_the_two_live_families(self):
        """Against the INSTALLED transformers: qwen3_5_text has a PrefixChange
        that already strips model.language_model, gemma4_text has no entry."""
        for model_type in ("qwen3_5_text", "gemma4_text"):
            with self.subTest(model_type=model_type):
                self.assertIsInstance(
                    NS["_registry_supplies_text_prefix_mapping"](model_type),
                    bool)

    def test_unknown_model_type_means_otr_supplies_the_mapping(self):
        self.assertFalse(
            NS["_registry_supplies_text_prefix_mapping"]("not_a_real_family"))


class CatalogOptInTests(unittest.TestCase):
    def test_only_the_two_reviewed_rows_are_opted_in(self):
        opted = sorted(
            row.repo_id for row in CATALOG.CURATED_LLM_MODELS
            if row.text_only_load == "native_text_decoder")
        self.assertEqual(opted, ["Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it"])

    def test_the_16gb_canonical_writer_is_not_opted_in(self):
        """gemma-4-12b-it carries the SAME loader_backend as the opted-in rows,
        which is exactly why the dispatch key is a separate explicit field. If
        this ever flips, the 16 GB box's qualified writer changed shape."""
        self.assertEqual(
            CATALOG.text_only_load_mode("google/gemma-4-12b-it"), "composite")
        self.assertEqual(
            CATALOG.text_only_load_mode("google/gemma-4-E4B-it"), "composite")

    def test_loader_backend_alone_would_have_been_the_wrong_key(self):
        """Pin the trap itself so nobody 'simplifies' the field away."""
        backends = {
            row.repo_id: row.loader_backend
            for row in CATALOG.CURATED_LLM_MODELS
        }
        self.assertEqual(
            backends["google/gemma-4-12b-it"],
            "transformers_multimodal_text_only")
        self.assertEqual(
            backends["google/gemma-4-E2B-it"],
            "transformers_multimodal_text_only")

    def test_badged_dropdown_labels_resolve(self):
        """A saved graph stores the picker's label, badge and all."""
        self.assertEqual(
            CATALOG.text_only_load_mode("google/gemma-4-E2B-it (3.0 GB)"),
            "native_text_decoder")
        self.assertEqual(
            CATALOG.text_only_load_mode("Qwen/Qwen3.5-4B (4.3 GB)"),
            "native_text_decoder")

    def test_uncurated_id_defaults_to_composite(self):
        self.assertEqual(
            CATALOG.text_only_load_mode("someone/uncurated-local-model"),
            "composite")
        self.assertEqual(CATALOG.text_only_load_mode(""), "composite")

    def test_non_multimodal_rows_are_composite(self):
        self.assertEqual(
            CATALOG.text_only_load_mode("mistralai/Mistral-Nemo-Instruct-2407"),
            "composite")


class NativeTextBoundaryTests(unittest.TestCase):
    """Execute the REAL try/retry boundary with a native-text row."""

    def setUp(self):
        load = next(n for n in TREE.body
                    if isinstance(n, ast.FunctionDef) and n.name == "load_llm")
        boundary = [n for n in ast.walk(load) if isinstance(n, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id == "ValueError"
            and h.name == "_dispatch_err" for h in n.handlers)]
        self.assertEqual(
            len(boundary), 1, "expected exactly one dispatch-refusal boundary")
        self.code = compile(
            ast.Module(body=boundary, type_ignores=[]), str(SOURCE), "exec")

        self.calls = []
        self.plans = []
        self.validated = []
        self.model = object()
        self.text_config = SimpleNamespace(model_type="gemma4_text")
        self.parent_config = gemma_config()
        self.common = {"quantization_config": object(), "trust_remote_code": False}
        self.init_kwargs = dict(self.common)
        self.init_kwargs["output_loading_info"] = True
        self.init_kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
        self.report = {"missing_keys": [], "unexpected_keys": [],
                       "mismatched_keys": [], "error_msgs": []}
        self.first_failure = None

        def load_model(*args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1 and self.first_failure is not None:
                raise self.first_failure
            if kwargs.get("output_loading_info"):
                return self.model, self.report
            return self.model

        def validate(info, *, model_id=None):
            self.validated.append((info, model_id))
            return ["model.vision_tower"]

        self.ns = {
            "AutoModelForCausalLM": SimpleNamespace(from_pretrained=load_model),
            "load_target": "fake-snapshot", "model_config": self.parent_config,
            "common_kwargs": self.common, "needs_4bit": True,
            "load_dtype": "bf16", "attn_impl": "sdpa",
            "_stripped_model_id": "google/gemma-4-E2B-it",
            "BitsAndBytesConfig": lambda **kw: kw,
            "torch": SimpleNamespace(bfloat16="bf16"),
            "_plan_nf4_cpu_offload": lambda cfg, **kw: (
                self.plans.append((cfg, kw)) or {"model.layers.0": 0}),
            "_validate_e4b_text_loading_info": lambda info: None,
            "_validate_native_text_loading_info": validate,
            "_runtime_log": lambda *a: None,
            "log": SimpleNamespace(warning=lambda *a: None),
            "_native_text_row": True,
            "_init_config": self.text_config,
            "_init_kwargs": self.init_kwargs,
        }

    def test_initial_load_uses_the_text_config_and_validates_coverage(self):
        exec(self.code, self.ns)
        self.assertIs(self.ns["model"], self.model)
        self.assertEqual(len(self.calls), 1, "must not need a retry")
        self.assertIs(self.calls[0]["config"], self.text_config)
        self.assertIsNot(self.calls[0]["config"], self.parent_config)
        self.assertTrue(self.calls[0]["output_loading_info"])
        self.assertEqual(self.plans, [], "no CPU-offload planning on success")
        self.assertEqual(len(self.validated), 1)
        self.assertIs(self.validated[0][0], self.report)
        self.assertEqual(self.validated[0][1], "google/gemma-4-E2B-it")

    def test_the_pair_return_is_unpacked_not_bound_to_model(self):
        """output_loading_info=True returns (model, info); binding the tuple
        to `model` fails much later as an opaque AttributeError."""
        exec(self.code, self.ns)
        self.assertNotIsInstance(self.ns["model"], tuple)

    def test_coverage_failure_propagates_after_model_is_bound_for_cleanup(self):
        def angry(info, *, model_id=None):
            raise RuntimeError("native text load rejected missing_keys")
        self.ns["_validate_native_text_loading_info"] = angry
        with self.assertRaisesRegex(RuntimeError, "missing_keys"):
            exec(self.code, self.ns)
        self.assertIs(
            self.ns.get("model"), self.model,
            "model must be bound before validation so cleanup can drop it")

    def test_a_dispatch_refusal_retry_keeps_the_text_config(self):
        """The retry must not silently restore the composite config."""
        self.first_failure = ValueError(
            "Some modules are dispatched on the CPU or the disk")
        exec(self.code, self.ns)
        self.assertEqual(len(self.calls), 2)
        retry = self.calls[1]
        self.assertIs(retry["config"], self.text_config)
        self.assertIsNot(retry["config"], self.parent_config)
        self.assertTrue(retry["output_loading_info"])
        self.assertEqual(
            retry["key_mapping"], {r"^model\.language_model\.": "model."})
        self.assertIs(self.plans[0][0], self.text_config)
        self.assertEqual(len(self.validated), 1)

    def test_common_kwargs_is_never_mutated_by_the_native_path(self):
        exec(self.code, self.ns)
        self.assertNotIn("output_loading_info", self.common)
        self.assertNotIn("key_mapping", self.common)


if __name__ == "__main__":
    unittest.main()
