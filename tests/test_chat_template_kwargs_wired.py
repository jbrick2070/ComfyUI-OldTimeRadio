"""Every generation-path apply_chat_template must pass chat_template_kwargs.

WHY THIS TEST EXISTS. Qwen/Qwen3.5-4B ships a THINKING chat template. Read from
the published `chat_template.jinja` on 2026-09-06, its generation-prompt tail is:

    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- else %}
            {{- '<think>\\n' }}
        {%- endif %}
    {%- endif %}

So the kwarg is load-bearing in BOTH directions. Passed as false, the prompt
carries a CLOSED think envelope and the model answers directly. Omitted, the
prompt ends with an OPEN `<think>` and the model is forced to reason -- which
under the writer's LMFE-constrained JSON pass produces the degenerate empty
object the GGUF lane already measured on this family.

A missed call site therefore fails SILENTLY and expensively: no exception, no
warning, just worse output hours into a render. Jinja does not raise on an
undefined template variable, so nothing else catches it. This test enumerates
the call sites from the AST so a NEW one cannot be added without the kwarg.

Pure AST parsing. Nothing here imports torch, transformers or OTR, loads a
model, or touches the network.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "nodes"

#: The system-role PROBE renders a throwaway message pair to discover whether a
#: template accepts a system role. It never generates, so think-suppression is
#: irrelevant there. Keyed by file and the call's first positional argument.
PROBE_EXEMPTIONS = {("_otr_loader_backends.py", "probe")}

#: Backends that own their think handling instead of the shared helper. The
#: GGUF lane strips a leading think envelope from the OUTPUT and injects its own
#: no-think directive; the OpenRouter lane never touches a local tokenizer.
SELF_MANAGED_BACKENDS = {"_otr_gguf_backend.py", "_otr_openrouter_backend.py"}


def _first_arg_name(call: ast.Call) -> str:
    if not call.args:
        return ""
    arg = call.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    return ""


def _apply_chat_template_calls():
    """Yield (filename, lineno, call node) for every apply_chat_template call."""
    for path in sorted(NODES.rglob("*.py")):
        if path.name in SELF_MANAGED_BACKENDS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "apply_chat_template":
                yield path.name, node.lineno, node


def _passes_chat_template_kwargs(call: ast.Call) -> bool:
    """True iff the call splats chat_template_kwargs(...) as **kwargs."""
    for kw in call.keywords:
        if kw.arg is not None:  # a named kwarg, not a ** splat
            continue
        value = kw.value
        if not isinstance(value, ast.Call):
            continue
        fn = value.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name == "chat_template_kwargs":
            return True
    return False


class ChatTemplateKwargsWiringTests(unittest.TestCase):
    def test_every_generation_call_site_passes_the_kwargs(self):
        missing = []
        checked = 0
        for filename, lineno, call in _apply_chat_template_calls():
            if (filename, _first_arg_name(call)) in PROBE_EXEMPTIONS:
                continue
            checked += 1
            if not _passes_chat_template_kwargs(call):
                missing.append(f"{filename}:{lineno}")
        self.assertGreater(
            checked, 0, "found no apply_chat_template call sites to check")
        self.assertEqual(
            missing, [],
            "these apply_chat_template calls omit chat_template_kwargs, so a "
            "thinking model is forced to reason there:\n  "
            + "\n  ".join(missing))

    def test_the_probe_exemption_still_matches_a_real_call(self):
        """A stale exemption must not silently excuse a real generation site."""
        found = {
            (filename, _first_arg_name(call))
            for filename, _lineno, call in _apply_chat_template_calls()
        }
        for exemption in PROBE_EXEMPTIONS:
            self.assertIn(
                exemption, found,
                f"exemption {exemption!r} matches no call site any more -- "
                "remove it rather than leaving a hole in the check")

    def test_helper_suppresses_thinking_for_the_named_rows_only(self):
        """chat_template_kwargs must stay an exact-id switch, not a prefix.

        WAS AN AST SHAPE CHECK for a single ``==`` compare, and it broke the
        day a second Qwen row arrived and the comparison became membership in
        a frozenset -- correctly, but the test could only say "not an Eq node"
        when what it means is "not a prefix match". So it now pins the
        PROPERTY, behaviourally: the ids that must fire, a near-miss that must
        not, and a source check that no prefix test crept in.

        A prefix would be a live defect rather than untidiness:
        ``apply_chat_template`` raises on a kwarg its template does not
        declare, so sweeping up a future Qwen row that lacks the switch turns
        every generate call into a crash.
        """
        from nodes._otr_loader_backends import chat_template_kwargs

        for model_id in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B"):
            with self.subTest(fires=model_id):
                self.assertEqual(chat_template_kwargs(model_id),
                                 {"enable_thinking": False})

        # Near-misses: same vendor, same prefix, NOT on the roster.
        for model_id in ("Qwen/Qwen3.5-4B-Base", "Qwen/Qwen2.5-7B-Instruct",
                         "Qwen/Qwen3.8-27B-FP8", "google/gemma-4-12b-it"):
            with self.subTest(silent=model_id):
                self.assertEqual(
                    chat_template_kwargs(model_id), {},
                    f"{model_id!r} is not on the roster but the switch fired; "
                    "a prefix match here crashes apply_chat_template on any "
                    "row whose template does not declare enable_thinking")

        source = (NODES / "_otr_loader_backends.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)
             and n.name == "chat_template_kwargs"),
            None,
        )
        self.assertIsNotNone(
            fn, "chat_template_kwargs is gone; the wiring test is meaningless")
        self.assertNotIn(
            "startswith", ast.dump(fn),
            "the model id test must be exact membership, never a prefix")
        self.assertIn("Qwen/Qwen3.5-4B", source)
        self.assertIn("Qwen/Qwen3.8-27B", source)

    def test_exact_match_is_fed_a_stripped_id_not_a_badged_label(self):
        """The dropdown label carries a ' (4.3 GB)' badge; the comparison must
        see the bare repo id or the suppression silently stops firing."""
        source = (NODES / "_otr_loader_backends.py").read_text(encoding="utf-8")
        self.assertIn(
            'split(" ", 1)[0]', source,
            "chat_template_kwargs must strip a badged dropdown label before "
            "comparing, or a saved workflow value disables think-suppression")


if __name__ == "__main__":
    unittest.main()
