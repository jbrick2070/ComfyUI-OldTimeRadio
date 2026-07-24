"""Sprint 3E -- title scratchpad + late binding.

Two behaviour changes in OTR_LedgerScriptWriter, both covered here:

1. Title scratchpad. `_generate_title_from_script` is no longer a
   single-shot "give me a title" call. It forces the model through a
   scratchpad: extract 3 concrete physical details -> draft 3 candidate
   titles -> emit a final `TITLE:` line. Python parses the title from
   the LAST `TITLE:` line. The title is generated from a whole-arc
   excerpt set (`_build_title_excerpt_set`: opening / middle / ending
   lines) plus the outline premise -- not a thin head-of-script slice.

2. Late binding. The per-line composer runs with the literal
   `EPISODE_TITLE: TBD` in the canon header, so no provisional / outline
   title is ever placed where a beat could speak it. Because the real
   title is bound late (after the script exists), the fragile post-hoc
   verbatim string substitution -- the former section J.6 and its
   title-swap helper -- is removed entirely.

These tests are import-level + source-scan; no torch / GPU needed
(the `_otr_model_catalog` import the writer does at module load is
pure Python, per the writer module docstring).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nodes.OTR_LedgerScriptWriter import (
    _build_title_excerpt_set,
    _generate_title_from_script,
)


PACK_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


SAMPLE_SCRIPT = (
    "[VOICE: ANNOUNCER, neutral] Tonight, on Tales From Beyond.\n\n"
    "[VOICE: AEGEUS, tense] The signal -- it's repeating itself.\n\n"
    "[VOICE: PHOEBE, alarmed] That's impossible. The dish was "
    "decommissioned six years ago.\n\n"
    "[SFX: low rumble of static]"
)


def _scratchpad(details, candidates, final_title):
    """Build a canned scratchpad LLM response: DETAILS + CANDIDATES +
    a final `TITLE:` line, matching the format the prompt asks for."""
    body = "DETAILS:\n" + "\n".join(details)
    body += "\nCANDIDATES:\n" + "\n".join(candidates)
    body += f"\nTITLE: {final_title}"
    return body


# ---------------------------------------------------------------------------
# 1. Scratchpad: parses the title from the last TITLE: line.
# ---------------------------------------------------------------------------


def test_scratchpad_parses_title_from_final_line():
    """A well-formed scratchpad response yields the final TITLE: line's
    title, not a CANDIDATES entry."""
    def gen(*_a, **_kw):
        return _scratchpad(
            ["a decommissioned dish", "a repeating signal", "static"],
            ["First Candidate", "Second Candidate", "The Echo Below"],
            "The Echo Below",
        )

    assert _generate_title_from_script(gen, SAMPLE_SCRIPT) == "The Echo Below"


def test_scratchpad_last_title_line_wins():
    """If the model emits more than one TITLE: line, the LAST one is the
    committed pick."""
    def gen(*_a, **_kw):
        return (
            "TITLE: A False Start\n"
            "DETAILS:\n- a dish\n- a signal\n- static\n"
            "CANDIDATES:\nThe Dish\nThe Signal\nFinal Frequency\n"
            "TITLE: Final Frequency"
        )

    assert _generate_title_from_script(gen, SAMPLE_SCRIPT) == "Final Frequency"


def test_scratchpad_strips_markdown_and_quote_wrappers():
    """`**TITLE:** "Pulse"` -> `Pulse` (prefix + quotes stripped)."""
    def gen(*_a, **_kw):
        return (
            "DETAILS:\n- a dish\n- a signal\n- static\n"
            "CANDIDATES:\nThe Dish\nThe Signal\nPulse\n"
            '**TITLE:** "Pulse"'
        )

    assert _generate_title_from_script(gen, SAMPLE_SCRIPT) == "Pulse"


def test_scratchpad_no_title_line_returns_empty():
    """No parseable TITLE: line -> "" so the caller falls back to
    outline.title."""
    def gen(*_a, **_kw):
        return (
            "DETAILS:\n- a dish\n- a signal\n- static\n"
            "CANDIDATES:\nThe Dish\nThe Signal\nPulse\n"
            "I cannot decide on a single title."
        )

    assert _generate_title_from_script(gen, SAMPLE_SCRIPT) == ""


def test_scratchpad_accepts_any_nonempty_authored_title():
    """Phrase lists and title word counts never discard authored output."""
    for authored in (
        "Untitled",
        "Signal Lost",
        "Episode",
        "Here is a title that the model authored as a complete English "
        "sentence well over ten words long indeed",
    ):
        def gen(*_a, _title=authored, **_kw):
            return _scratchpad(["a", "b", "c"], ["x", "y", "z"], _title)

        assert _generate_title_from_script(
            gen, SAMPLE_SCRIPT,
        ) == authored


def test_scratchpad_empty_script_skips_llm():
    """Empty / blank script -> "" with no LLM call."""
    def trap(*_a, **_kw):
        raise AssertionError("LLM must not be called on an empty script")

    assert _generate_title_from_script(trap, "") == ""
    assert _generate_title_from_script(trap, "   \n\n  ") == ""


def test_scratchpad_llm_raise_returns_empty():
    """An LLM exception is swallowed -> "" (caller falls back)."""
    def gen(*_a, **_kw):
        raise RuntimeError("LLM offline")

    assert _generate_title_from_script(gen, SAMPLE_SCRIPT) == ""


# ---------------------------------------------------------------------------
# 2. Prompt shape: scratchpad forced, news-seed-free, premise allowed,
#    enough token budget to reach the final TITLE: line.
# ---------------------------------------------------------------------------


def test_title_prompt_forces_scratchpad_and_is_news_free():
    """The title prompt must drive the DETAILS / CANDIDATES / TITLE
    scratchpad and must NOT leak news-seed material. The outline
    premise is allowed through (story spine, not the news article)."""
    captured: dict = {}

    def gen(messages, **kw):
        captured["messages"] = messages
        captured["kw"] = kw
        return _scratchpad(["a", "b", "c"], ["x", "y", "z"], "Clean Title")

    out = _generate_title_from_script(
        gen, SAMPLE_SCRIPT, premise="A lonely dish hears its own echo.",
    )
    assert out == "Clean Title"

    prompt = " ".join(m.get("content", "") for m in captured["messages"])
    for marker in ("DETAILS", "CANDIDATES", "TITLE:"):
        assert marker in prompt, f"scratchpad marker {marker!r} missing"
    for forbidden in ("news", "headline", "article", "RSS"):
        assert forbidden.lower() not in prompt.lower(), (
            f"title prompt leaked news-seed token {forbidden!r}"
        )
    assert "key object" not in prompt.lower(), (
        "title prompt should not prime repeated 'key' mystery titles"
    )
    assert "important object" in prompt.lower()
    # The premise IS passed through deliberately.
    assert "lonely dish hears its own echo" in prompt


def test_title_prompt_token_budget_reaches_final_line():
    """The scratchpad (3 details + 3 candidates + final TITLE: line)
    needs a real token budget; the pre-Sprint-3E single-shot pass used
    24 tokens, which would truncate before the model ever reached
    TITLE:."""
    captured: dict = {}

    def gen(messages, **kw):
        captured["kw"] = kw
        return _scratchpad(["a", "b", "c"], ["x", "y", "z"], "Some Title")

    _generate_title_from_script(gen, SAMPLE_SCRIPT)
    assert captured["kw"].get("max_new_tokens", 0) >= 100, (
        "scratchpad pass must request enough tokens to reach the final "
        f"TITLE: line; got {captured['kw'].get('max_new_tokens')}"
    )


# ---------------------------------------------------------------------------
# 3. Whole-arc excerpt set.
# ---------------------------------------------------------------------------


def test_excerpt_set_empty_script():
    """Empty / blank script -> all three windows empty."""
    for blank in ("", "   \n\n  "):
        ex = _build_title_excerpt_set(blank)
        assert ex == {
            "opening_lines": "", "middle_lines": "", "ending_lines": "",
        }


def test_excerpt_set_spans_whole_arc():
    """On a long script the opening / middle / ending windows are all
    populated and the ending window carries content a head-only slice
    would miss -- this is the whole point of the Sprint 3E change."""
    blocks = [f"[VOICE: C, neutral] line {i}" for i in range(40)]
    ex = _build_title_excerpt_set("\n\n".join(blocks))
    assert ex["opening_lines"] and ex["middle_lines"] and ex["ending_lines"]
    assert "line 0" in ex["opening_lines"]
    assert "line 39" in ex["ending_lines"]
    # Ending content must not bleed into the opening window.
    assert "line 39" not in ex["opening_lines"]
    # The middle window must be drawn from the centre, not the head.
    assert "line 0" not in ex["middle_lines"]


def test_excerpt_set_short_script_no_crash():
    """A short script (fewer blocks than the tail window) still resolves
    cleanly -- the opening window is populated, no exception."""
    blocks = [f"[VOICE: C, neutral] line {i}" for i in range(3)]
    ex = _build_title_excerpt_set("\n\n".join(blocks))
    assert ex["opening_lines"]


# ---------------------------------------------------------------------------
# 4. Late binding: EPISODE_TITLE: TBD in the composition header, and
#    the post-hoc verbatim substitution is fully removed.
# ---------------------------------------------------------------------------


def test_composition_header_uses_tbd_literal():
    """The writer builds the per-line composer's canon header with the
    literal `EPISODE_TITLE: TBD`, so no provisional title is spoken."""
    src = WRITER_PATH.read_text(encoding="utf-8")
    assert "EPISODE_TITLE: TBD" in src, (
        "composition header must carry the EPISODE_TITLE: TBD literal"
    )


def test_post_hoc_title_substitution_is_removed():
    """The fragile post-hoc verbatim title-substitution helper, its
    min-length guard, and the J.6 section are all gone. Needles are
    assembled from fragments so this test file's own strings do not
    self-match if it is ever scanned alongside the writer."""
    src = WRITER_PATH.read_text(encoding="utf-8")
    sub_helper_def = "def _substitute" + "_title_in_text"
    assert sub_helper_def not in src, (
        "post-hoc verbatim title-substitution helper must be removed"
    )
    sub_min_guard = "_TITLE_SUB" + "_MIN_LEN"
    assert sub_min_guard not in src, (
        "the title-substitution min-length guard must be removed"
    )
    j6_header = "--- J." + "6."
    assert j6_header not in src, (
        "the post-hoc title-substitution section header must be removed"
    )
    assert "title_substitution" not in (
        # The meta key must not be stamped any more either; allow the
        # word only inside a comment that documents its retirement.
        "\n".join(
            ln for ln in src.splitlines()
            if "title_substitution" in ln and not ln.lstrip().startswith("#")
        )
    ), "meta.title_substitution stamp must be removed"


def test_title_call_passes_premise_for_grounding():
    """AST scan: the J.5 `_generate_title_from_script(...)` call passes
    the outline `premise` keyword so the scratchpad is grounded in the
    story spine, not just the dialogue excerpts. (The module also has
    self-test calls in its `__main__` block that pass no premise --
    those are scoped out by walking only the owning method body.)

    scifi_fable2 S1a (2026-07-10): the J.5 block moved from `run` into
    the extracted `_run_writer_tail` (tail extraction, architecture doc
    sections 11/13) -- the pin follows it there.
    """
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))

    tail_fn = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "OTR_LedgerScriptWriter"
        ):
            for sub in node.body:
                if (
                    isinstance(sub, ast.FunctionDef)
                    and sub.name == "_run_writer_tail"
                ):
                    tail_fn = sub
                    break
    assert tail_fn is not None, (
        "writer must have OTR_LedgerScriptWriter._run_writer_tail "
        "(the S1a tail extraction owns J.5)"
    )

    title_calls = [
        node for node in ast.walk(tail_fn)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_generate_title_from_script"
        )
    ]
    assert title_calls, (
        "_run_writer_tail() must call _generate_title_from_script in J.5"
    )
    assert any(
        "premise" in {kw.arg for kw in call.keywords}
        for call in title_calls
    ), "the J.5 title call must pass `premise=` for scratchpad grounding"
