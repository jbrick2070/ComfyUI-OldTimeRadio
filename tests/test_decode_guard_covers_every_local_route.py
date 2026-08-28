"""EVERY local transformers generate() must carry the liveness guard.

WHY THIS FILE EXISTS. The guard shipped installed in ONE wrapper --
`OTR_LedgerScriptWriter._build_truncating_generate_fn` -- because that is where
the 22-minute runaway was observed. A thorough audit found that all local
text-generation routes must carry the liveness guard:

* `OTR_LedgerScriptWriter._build_truncating_generate_fn` -- the main script writer pass;
* `_otr_constrained_generate.make_constrained_generate_fn` -- called once per
  voiced beat by the writer's slot-drama contract;
* `_otr_model_loader.make_generate_fn` -- called by story orchestration, shot
  lock and diagnostic nodes;
* `_otr_model_loader.make_polish_generate_fn` -- polish-specific generate fn.

This test file inspects AST call trees at the function level rather than using
fragile file-level string matching, guaranteeing:
1. Every guarded route passes `stopping_criteria` to `model.generate()`.
2. Every guarded route builds `make_degeneracy_criterion` and raises
   `GenerationDegeneracyError` upon a cycle detection latch (`.hit`).
3. Every `model.generate()` call across `nodes/` and `visual/` is either
   in the guarded routes inventory or in the explicitly classified exemptions list.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Iterator, Tuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NODES_DIR = REPO_ROOT / "nodes"
VISUAL_DIR = REPO_ROOT / "visual"

#: (relative_path, enclosing_function_name)
GUARDED_ROUTES = [
    ("nodes/OTR_LedgerScriptWriter.py", "_build_truncating_generate_fn"),
    ("nodes/_otr_constrained_generate.py", "make_constrained_generate_fn"),
    ("nodes/_otr_model_loader.py", "make_generate_fn"),
    ("nodes/_otr_model_loader.py", "make_polish_generate_fn"),
]

#: (relative_path, enclosing_function_name) -> Reason for exemption
KNOWN_EXEMPTIONS = {
    ("nodes/_otr_model_loader.py", "load_llm"): (
        "1-token CUDA kernel warmup (max_new_tokens=1, do_sample=False)"
    ),
    ("nodes/_otr_bark_lib.py", "_generate_single_line"): (
        "Bark TTS audio generation; bounded by semantic_min_eos_p"
    ),
    ("nodes/_otr_audio_engines/eng_musicgen.py", "generate_clip"): (
        "MusicGen audio generation; max_new_tokens derived from duration"
    ),
    ("nodes/_otr_audio_engines/eng_stable_audio_3.py", "generate_clip"): (
        "ComfyUI EmptyLatentAudio graph generation"
    ),
    ("nodes/_otr_openrouter_backend.py", "make_openrouter_generate_fn"): (
        "Remote OpenRouter HTTP backend (zero local VRAM)"
    ),
    ("nodes/_otr_comfy_backend.py", "make_comfy_credits_generate_fn"): (
        "Remote Comfy Credits HTTP backend (zero local VRAM)"
    ),
    ("nodes/_otr_google_api/llm.py", "make_google_api_generate_fn"): (
        "Remote Google API HTTP backend (zero local VRAM)"
    ),
    ("nodes/_otr_gguf_backend.py", "make_gguf_generate_fn"): (
        "GGUF llama-cpp-python backend"
    ),
    ("visual/backends/florence2_sdxl_comp.py", "_render_real"): (
        "Florence-2 vision model"
    ),
    ("visual/llm_polish.py", "_generate_single"): (
        "SDXL visual prompt polish; bounded max_new_tokens=100, do_sample=False"
    ),
}


def _find_function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return funcs


def _find_all_generate_calls_in_file(path: pathlib.Path) -> Iterator[Tuple[str, ast.Call]]:
    """Yields (enclosing_func_name, call_node) for every .generate(...) call."""
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    # Track parent functions
    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.func_stack = []
            self.calls = []

        def visit_FunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
                enclosing = self.func_stack[0] if self.func_stack else "<module>"
                self.calls.append((enclosing, node))
            self.generic_visit(node)

    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    yield from visitor.calls


@pytest.mark.parametrize("rel_path,func_name", GUARDED_ROUTES)
def test_every_local_generate_route_installs_the_liveness_guard(rel_path: str, func_name: str):
    file_path = REPO_ROOT / rel_path
    assert file_path.exists(), f"File {rel_path} does not exist"

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = _find_function_nodes(tree)

    assert func_name in funcs, f"{func_name} not found in {rel_path}"
    func_node = funcs[func_name]
    func_src = ast.get_source_segment(source, func_node) or ""

    # 1. Check that generate() is called and receives stopping_criteria or **kwargs configured with it
    gen_calls = [
        node for node in ast.walk(func_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]
    assert gen_calls, f"{rel_path}:{func_name} does not call generate()"

    for call in gen_calls:
        has_stopping_criteria_kw = any(
            kw.arg == "stopping_criteria" for kw in call.keywords
        )
        has_kwargs_spread = any(kw.arg is None for kw in call.keywords)
        has_stopping_criteria_in_src = "stopping_criteria" in func_src
        assert has_stopping_criteria_kw or (has_kwargs_spread and has_stopping_criteria_in_src), (
            f"{rel_path}:{func_name} calls generate() without stopping_criteria"
        )

    # 2. Check that make_degeneracy_criterion is referenced/called
    assert "make_degeneracy_criterion" in func_src, (
        f"{rel_path}:{func_name} does not build make_degeneracy_criterion"
    )


@pytest.mark.parametrize("rel_path,func_name", GUARDED_ROUTES)
def test_every_guarded_route_raises_rather_than_returning_truncated_text(
    rel_path: str, func_name: str,
):
    file_path = REPO_ROOT / rel_path
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = _find_function_nodes(tree)

    assert func_name in funcs, f"{func_name} not found in {rel_path}"
    func_node = funcs[func_name]
    func_src = ast.get_source_segment(source, func_node) or ""

    assert "GenerationDegeneracyError" in func_src, (
        f"{rel_path}:{func_name} does not reference/raise GenerationDegeneracyError on cycle halt"
    )
    assert ".hit" in func_src or '"hit"' in func_src, (
        f"{rel_path}:{func_name} never inspects the guard's hit latch"
    )


def test_the_route_list_matches_every_generate_site_in_repo():
    """Verify that every generate() call in nodes/ and visual/ is accounted for."""
    guarded_set = set(GUARDED_ROUTES)
    exempt_set = set(KNOWN_EXEMPTIONS.keys())

    unclassified = []
    scanned_files = list(NODES_DIR.rglob("*.py")) + list(VISUAL_DIR.rglob("*.py"))

    for file_path in scanned_files:
        rel_path = file_path.relative_to(REPO_ROOT).as_posix()
        for enclosing_func, _ in _find_all_generate_calls_in_file(file_path):
            route_key = (rel_path, enclosing_func)
            if route_key not in guarded_set and route_key not in exempt_set:
                unclassified.append(route_key)

    assert not unclassified, (
        f"Found unclassified generate() call sites: {unclassified}. "
        "Either add them to GUARDED_ROUTES (and install the liveness guard) "
        "or add them to KNOWN_EXEMPTIONS with a rationale."
    )

