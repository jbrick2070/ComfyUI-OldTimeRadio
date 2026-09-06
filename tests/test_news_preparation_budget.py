"""News preparation must not consume the short ranking generation deadline.

Compile the real rankers and timeout wrapper from their AST, without importing
OTR, ComfyUI, torch or model libraries. The executor is immediate and the clock
is synthetic: these tests do not create threads, wait, download or load models.
Existing concurrency tests remain the authority for real worker races.
"""
import ast
import builtins
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "nodes" / "story_orchestrator.py"
LOADER = ROOT / "nodes" / "_otr_model_loader.py"


class Cancelled(BaseException):
    """Match Comfy cancellation's BaseException boundary without importing it."""


class DeadlineExceeded(Exception):
    pass


def real_definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for name in names:
        matches = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name == name]
        if len(matches) != 1:
            raise AssertionError("Expected one real production definition: " + name)
        selected.append(matches[0])
    return selected


class NewsHarness:
    def __init__(self, function_name, response):
        self.now = 1000.0
        self.preparation_seconds = 0.0
        self.generation_seconds = 0.0
        self.deadline = None
        self.cancel_on_check = None
        self.checks = 0
        self.events = []
        self.requests = []
        self.generations = []
        self.submissions = []
        self.wait_budgets = []
        self.shutdowns = []
        self.request_error = None
        self.adapter_error = None
        self.response = response
        self.cache_entry = object()

        def request_slot(*args, **kwargs):
            self.events.append("prepare")
            self.requests.append((args, kwargs, self.deadline))
            self.now += self.preparation_seconds
            if self.request_error is not None:
                raise self.request_error
            return self.cache_entry

        def generate_fn(*, messages, temperature, max_new_tokens):
            self.events.append("generate")
            self.generations.append({
                "messages": messages, "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "budget_remaining": self.deadline - self.now,
            })
            self.now += self.generation_seconds
            return self.response

        def make_generate_fn(entry):
            self.events.append("adapter")
            if entry is not self.cache_entry:
                raise AssertionError("Prepared entry was not forwarded unchanged")
            if self.adapter_error is not None:
                raise self.adapter_error
            return generate_fn

        def check_cancelled():
            self.events.append("cancel_check")
            self.checks += 1
            if self.checks == self.cancel_on_check:
                raise Cancelled("synthetic operator cancellation")

        def set_deadline(value):
            self.deadline = value

        loader = SimpleNamespace(
            request_slot=request_slot, make_generate_fn=make_generate_fn,
            raise_if_processing_interrupted=check_cancelled,
            set_generation_deadline=set_deadline,
            GenerationDeadlineExceededError=DeadlineExceeded,
            invalidate_cache_no_gpu_teardown=lambda: self.events.append("invalidate"),
        )
        harness = self

        class ImmediateFuture:
            def __init__(self, worker):
                self.error = None
                self.value = None
                try:
                    self.value = worker()
                except BaseException as exc:
                    self.error = exc

            def result(self, *, timeout):
                harness.wait_budgets.append(timeout)
                if self.error is not None:
                    raise self.error
                return self.value

        class ImmediateExecutor:
            def __init__(self, **kwargs):
                harness.events.append("executor")
                if kwargs.get("max_workers") != 1:
                    raise AssertionError("Expected a single ranking worker")

            def submit(self, worker):
                harness.events.append("submit")
                harness.submissions.append(harness.now)
                return ImmediateFuture(worker)

            def shutdown(self, *, wait):
                harness.shutdowns.append(wait)

        def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "" and tuple(fromlist) == ("_otr_model_loader",):
                return SimpleNamespace(_otr_model_loader=loader)
            if level == 0 and name == "concurrent.futures":
                return SimpleNamespace(ThreadPoolExecutor=ImmediateExecutor,
                                       TimeoutError=TimeoutError)
            raise AssertionError("Unstubbed import attempted: " + name)

        names = ("_LLMTimeout", "_LLMTimeoutWorkflowPause", "_run_with_timeout",
                 "_llm_rank_news_candidates", "_llm_rerank_with_bodies")
        namespace = {
            "__builtins__": dict(vars(builtins), __import__=isolated_import),
            "time": SimpleNamespace(monotonic=lambda: self.now),
            "re": re,
            "log": SimpleNamespace(**{name: lambda *args, **kwargs: None
                                       for name in ("info", "warning", "error")}),
            "_runtime_log": lambda *args: None,
            "vram_reset_peak": lambda *args: None,
            "vram_snapshot": lambda *args: None,
            "_body_rerank_preview": lambda body: body[:800],
        }
        module = ast.Module(body=real_definitions(ORCHESTRATOR, names), type_ignores=[])
        exec(compile(module, str(ORCHESTRATOR), "exec"), namespace)
        self.run = namespace[function_name]
        self.pause = namespace["_LLMTimeoutWorkflowPause"]


class PreparationAssertions:
    def setUp(self):
        self.harness = NewsHarness(self.function_name, self.response)
        self.policy = object()
        self.load_config = object()
        self.pool = [
            {"headline": "Headline one", "full_text": "Body one"},
            {"headline": "Headline two", "full_text": "Body two"},
            {"headline": "Headline three", "full_text": "Body three"},
        ]

    def invoke(self, *, pool=None, load_config=True):
        kwargs = dict(model_id="google/gemma-4-E2B-it", policy=self.policy,
                      load_config=self.load_config if load_config else None)
        if self.function_name == "_llm_rank_news_candidates":
            kwargs["top_k"] = 2
        return self.harness.run(self.pool if pool is None else pool, **kwargs)

    def test_slow_preparation_gets_a_fresh_full_generation_budget(self):
        self.harness.preparation_seconds = 3600.0
        self.harness.generation_seconds = self.budget - 1.0
        result = self.invoke()
        self.assertEqual(result, [self.pool[index] for index in self.order])
        self.assertEqual(self.harness.submissions, [4600.0])
        self.assertEqual(self.harness.requests[0][2], None)
        self.assertEqual(self.harness.generations[0]["budget_remaining"], self.budget)
        self.assertEqual(self.harness.wait_budgets, [1.0])
        self.assertNotIn("invalidate", self.harness.events)
        self.assertIsNone(self.harness.deadline)

    def test_prepares_once_before_submission_and_preserves_call_contract(self):
        self.invoke()
        self.assertEqual(len(self.harness.requests), 1)
        args, kwargs, deadline = self.harness.requests[0]
        self.assertEqual(args, ("technical", "google/gemma-4-E2B-it"))
        self.assertIs(kwargs["policy"], self.policy)
        self.assertIs(kwargs["load_config"], self.load_config)
        self.assertIsNone(deadline)
        self.assertEqual(len(self.harness.generations), 1)
        generation = self.harness.generations[0]
        self.assertEqual(generation["temperature"], 0.05)
        self.assertEqual(generation["max_new_tokens"], self.token_cap)
        self.assertEqual(generation["budget_remaining"], self.budget)
        self.assertEqual(len(generation["messages"]), 1)
        self.assertEqual(generation["messages"][0]["role"], "user")
        self.assertIn(self.prompt_marker, generation["messages"][0]["content"])
        events = self.harness.events
        self.assertLess(events.index("cancel_check"), events.index("prepare"))
        self.assertLess(events.index("prepare"), events.index("adapter"))
        self.assertLess(events.index("adapter"), events.index("submit"))
        self.assertGreaterEqual(events[:events.index("submit")].count("cancel_check"), 2)

    def test_cancel_before_preparation_does_not_acquire_or_start_worker(self):
        self.harness.cancel_on_check = 1
        with self.assertRaises(Cancelled):
            self.invoke(load_config=False)
        self.assertEqual(self.harness.requests, [])
        self.assertEqual(self.harness.submissions, [])
        self.assertEqual(self.harness.generations, [])

    def test_cancel_after_preparation_does_not_start_generation(self):
        self.harness.cancel_on_check = 2
        with self.assertRaises(Cancelled):
            self.invoke(load_config=False)
        self.assertEqual(len(self.harness.requests), 1)
        self.assertEqual(self.harness.submissions, [])
        self.assertEqual(self.harness.generations, [])
        self.assertNotIn("invalidate", self.harness.events)
        self.assertIsNone(self.harness.deadline)

    def test_preparation_error_does_not_create_timed_worker(self):
        error = RuntimeError("synthetic acquisition failure")
        self.harness.request_error = error
        with self.assertRaises(RuntimeError) as caught:
            self.invoke()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.harness.submissions, [])
        self.assertEqual(self.harness.generations, [])
        self.assertNotIn("invalidate", self.harness.events)

    def test_adapter_error_does_not_create_timed_worker(self):
        error = RuntimeError("synthetic adapter failure")
        self.harness.adapter_error = error
        with self.assertRaises(RuntimeError) as caught:
            self.invoke()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.harness.submissions, [])
        self.assertEqual(self.harness.generations, [])

    def test_shortlist_bypass_does_not_prepare_or_check_cancel(self):
        shortlist = self.pool[:self.bypass_count]
        result = self.invoke(pool=shortlist)
        self.assertEqual(result, shortlist)
        self.assertIsNot(result, shortlist)
        self.assertEqual(self.harness.events, [])

    def test_real_generation_overrun_still_pauses_and_invalidates(self):
        self.harness.generation_seconds = self.budget + 1.0
        with self.assertRaises(self.harness.pause):
            self.invoke(load_config=False)
        self.assertEqual(len(self.harness.generations), 1)
        self.assertEqual(self.harness.events.count("invalidate"), 1)
        self.assertEqual(self.harness.shutdowns, [False])
        self.assertIsNone(self.harness.deadline)


class HeadlinePreparationTests(PreparationAssertions, unittest.TestCase):
    function_name = "_llm_rank_news_candidates"
    response = "2,1"
    order = (1, 0)
    budget = 65
    token_cap = 64
    bypass_count = 2
    prompt_marker = "Top 2 indices:"


class BodyPreparationTests(PreparationAssertions, unittest.TestCase):
    function_name = "_llm_rerank_with_bodies"
    response = "2"
    order = (1, 0, 2)
    budget = 40
    token_cap = 8
    bypass_count = 1
    prompt_marker = "Best index:"


class LoaderCheckpointTests(unittest.TestCase):
    def test_native_download_completion_checks_cancel_before_unload_and_load(self):
        request = real_definitions(LOADER, ("request_slot",))[0]
        downloads = [node for node in ast.walk(request)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "auto_download_if_missing"]
        self.assertEqual(len(downloads), 1)
        download = downloads[0]
        checkpoints = [statement.value for statement in request.body
                       if isinstance(statement, ast.Expr)
                       and isinstance(statement.value, ast.Call)
                       and isinstance(statement.value.func, ast.Name)
                       and statement.value.func.id == "raise_if_processing_interrupted"
                       and statement.lineno > download.end_lineno]
        self.assertTrue(checkpoints, "Missing unconditional post-download cancellation check")
        checkpoint = min(checkpoints, key=lambda node: node.lineno)
        later_gpu_calls = [node for node in ast.walk(request)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name)
                           and node.func.id in ("_self_unload", "load_llm")
                           and node.lineno > download.end_lineno]
        self.assertTrue(later_gpu_calls)
        self.assertLess(checkpoint.lineno, min(node.lineno for node in later_gpu_calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
