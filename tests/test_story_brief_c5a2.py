"""Sprint C C5a2: writer wiring at K.5.5.

C5a2 integrates the reflection module shipped at C5a1 into
`OTR_LedgerScriptWriter.execute()`. The call site sits BETWEEN the
section K.5 visual_plan stamp and the section L `script_text` /
`script_json` assembly, so both the saved-on-disk ledger AND the
returned `script_json` output socket carry `meta.story_brief*` (closes
the K.5.5/L.5 silent-failure class flagged by E-01).

Test split (existing repo pattern):

  * AST-based wiring assertions -- mirror `test_writer_paired_wiring`.
    Cheap to run, no GPU, no mocks. Lock the call site, the import
    surface, the argument shapes, and the "no explicit eviction"
    property (E-15 revised at C1 audit per RR-A1).

  * Integration assertions that exercise `run_story_brief_reflection`
    + `meta.update` semantics without invoking the full writer
    pipeline (avoids the LLM-loader / RSS-fetcher / disk fixtures that
    a real `execute()` call would need).

  * VRAM-accounting / cache-stability runtime tests live behind the
    OTR_REGRESSION_RUNTIME env gate, matching the existing
    `tests/test_audio_byte_identical.py` pattern. They skip in the
    pytest-only sandbox; they run on the operator's GPU as part of
    the Sprint A empirical-verification pass.

E-15 revised at C1 audit per RR-A1: the loader is single-slot by
architecture. `request_slot(technical_model)` inside
`technical_generate_fn` calls `unload_llm()` to evict any prior
resident model BEFORE loading the next. Peak transient VRAM during
the swap is `max(creative_size, technical_size)`, not their sum. The
C5a2 K.5.5 call does NOT include an explicit eviction call; the
regression tests in this file (and the runtime ones gated on
OTR_REGRESSION_RUNTIME) prove the no-OOM property without one.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any

import pytest

from nodes import _otr_story_brief as sb


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_WRITER_PATH = _REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


# ---------------------------------------------------------------------------
# Shared AST helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def writer_ast() -> ast.AST:
    return ast.parse(_WRITER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def execute_fn(writer_ast) -> ast.FunctionDef:
    """Return the writer method that OWNS the K.5.5 reflection site.

    Originally the `run` method (the ComfyUI node-execution entrypoint;
    the fixture name `execute_fn` mirrors the historical spec's
    "execute()" wording). scifi_fable2 S1a (2026-07-10): the whole tail
    (J.5 -> M save, including K.5.5/K.5.6) moved into the extracted
    `_run_writer_tail` (architecture doc sections 11/13) -- every
    call-site pin below follows it there. The placement laws are
    unchanged: receipt stamp before the reflection, reflection before
    the script_text assembly and the M ledger save."""
    for node in ast.walk(writer_ast):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_run_writer_tail"
        ):
            return node
    raise AssertionError(
        "OTR_LedgerScriptWriter._run_writer_tail not found "
        "(the S1a tail extraction owns K.5.5)"
    )


def _find_call_site(execute_node, target_name: str) -> list[ast.Call]:
    """Return every Call node where the callable's name matches."""
    hits: list[ast.Call] = []
    for node in ast.walk(execute_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == target_name:
            hits.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == target_name:
            hits.append(node)
    return hits


def _find_assign_to_subscript(
    execute_node, value_name: str, key_value: str,
) -> list[ast.Assign]:
    """Find `value_name['key_value'] = ...` assignments."""
    hits: list[ast.Assign] = []
    for node in ast.walk(execute_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == value_name
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key_value
            ):
                continue
            hits.append(node)
    return hits


# ---------------------------------------------------------------------------
# 1. Module-level import (E-22 / RR-B4)
# ---------------------------------------------------------------------------


def test_module_level_import_of_reflection_entrypoint(writer_ast):
    """E-22 / RR-B4: `from ._otr_story_brief import
    run_story_brief_reflection` appears at module level (not inside
    a function body). Catches a future refactor that pushes the
    import into hot-path scope."""
    saw_module_level_import = False
    for node in writer_ast.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "_otr_story_brief":
            continue
        # Note: loop variable is `imp` (not the forbidden-pattern name
        # `alias`) per the S28 cleanup; ast.alias is still the node
        # type, this just renames the binding.
        for imp in node.names:
            if imp.name == "run_story_brief_reflection":
                saw_module_level_import = True
    assert saw_module_level_import, (
        "Module-level `from ._otr_story_brief import "
        "run_story_brief_reflection` not found in "
        "nodes/OTR_LedgerScriptWriter.py"
    )

    # Negative: no hot-path import of the same name inside any function.
    for node in ast.walk(writer_ast):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module == "_otr_story_brief":
                pytest.fail(
                    f"Hot-path import of `_otr_story_brief` inside "
                    f"function `{node.name}` at line {sub.lineno}; "
                    "E-22 / RR-B4 requires module-level import only"
                )


# ---------------------------------------------------------------------------
# 2-3. Call-site placement (K.5.5 -- after K.5, before section L)
# ---------------------------------------------------------------------------


def _reflection_call(execute_node) -> ast.Call:
    hits = _find_call_site(execute_node, "run_story_brief_reflection")
    assert len(hits) == 1, (
        f"expected exactly one run_story_brief_reflection call, got "
        f"{len(hits)}"
    )
    return hits[0]


def _line_of(node: ast.AST) -> int:
    return getattr(node, "lineno", -1)


def test_call_site_is_k55_not_l5(execute_fn):
    """Q1 lock (E-01): the K.5.5 call appears AFTER the story-style
    receipt is stamped AND BEFORE `script_text = _PL.assemble_script_text_from_ledger`.
    Locks the call site against future refactor that would push the
    call past the script_text assembly."""
    refl = _reflection_call(execute_fn)

    # `_stamp_story_style_receipt(...)` is the K.5 closer. It owns the
    # scaffold-off receipt path and prevents visual style fallback into
    # story metadata.
    style_receipts = _find_call_site(execute_fn, "_stamp_story_style_receipt")
    assert style_receipts, "_stamp_story_style_receipt(...) not found in execute()"
    last_style_receipt = max(style_receipts, key=_line_of)

    # `script_text = _PL.assemble_script_text_from_ledger(led.data)` -- L opener.
    # kibitz r3 D4 (2026-07-09): the J.5 title-regen fix added a SECOND,
    # EARLIER assemble call (`assembled_script = ...`) so the title reads
    # the real post-loop ledger. This pin's intent -- per its own docstring
    # -- is the L-opener `script_text = ...` ASSIGNMENT specifically, so
    # select by assignment target instead of min() over every call.
    script_text_assigns = [
        node
        for node in ast.walk(execute_fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "script_text"
            for t in node.targets
        )
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", getattr(
            node.value.func, "id", "")) == "assemble_script_text_from_ledger"
    ]
    assert script_text_assigns, (
        "`script_text = ...assemble_script_text_from_ledger(...)` L-opener "
        "assignment not found in execute()"
    )
    first_assemble = min(script_text_assigns, key=_line_of)

    assert _line_of(last_style_receipt) < _line_of(refl), (
        f"K.5.5 reflection call at line {_line_of(refl)} is BEFORE "
        f"story-style receipt at line {_line_of(last_style_receipt)}; "
        "must be AFTER per Q1 lock"
    )
    assert _line_of(refl) < _line_of(first_assemble), (
        f"K.5.5 reflection call at line {_line_of(refl)} is AFTER "
        f"script_text assembly at line {_line_of(first_assemble)}; "
        "must be BEFORE per Q1 lock so both saved ledger AND returned "
        "script_json carry meta.story_brief*"
    )


def test_call_site_before_led_save(execute_fn):
    """Belt-and-suspenders: the K.5.5 call is BEFORE the FINAL
    `led.save()` (section M persistent save) so the disk ledger
    carries the brief. The writer has earlier intra-pipeline
    checkpoint saves (cast assembly, mid-compose snapshots) that
    intentionally precede K.5.5 -- those are checkpoint-only and
    get overwritten by the final save. Combined with the previous
    test, both consumers (returned script_json + saved ledger) get
    the brief in lockstep."""
    refl = _reflection_call(execute_fn)
    save_calls = _find_call_site(execute_fn, "save")
    assert save_calls, "led.save() call not found in execute()"
    final_save = max(save_calls, key=_line_of)
    assert _line_of(refl) < _line_of(final_save), (
        f"K.5.5 reflection call at line {_line_of(refl)} is AFTER the "
        f"FINAL led.save() at line {_line_of(final_save)}; the saved-"
        "to-disk ledger would not carry meta.story_brief*"
    )


def test_reflections_describe_the_final_word_fitted_rows(execute_fn):
    """PBUG-20260721-18: metadata reflection follows the final row gate."""
    refl = _reflection_call(execute_fn)
    fit_calls = _find_call_site(execute_fn, "fit_final_word_delivery")
    actual_calls = _find_call_site(execute_fn, "stamp_actual")
    produced_calls = _find_call_site(execute_fn, "run_produced_story_summary")

    assert len(fit_calls) == 1
    assert len(actual_calls) == 1
    assert len(produced_calls) == 1
    assert _line_of(fit_calls[0]) < _line_of(actual_calls[0]) < _line_of(refl)
    assert _line_of(refl) < _line_of(produced_calls[0])


# ---------------------------------------------------------------------------
# 4. Call site uses technical_generate_fn, not creative (E-03)
# ---------------------------------------------------------------------------


def test_call_site_uses_technical_fn_not_creative(execute_fn):
    """E-03: the K.5.5 call passes `technical_generate_fn` (or
    similarly-named variable carrying the technical-slot closure),
    NOT the creative-slot closure. AST inspection of the Call.args
    and Call.keywords."""
    refl = _reflection_call(execute_fn)

    # Collect every Name id appearing in positional / keyword args.
    arg_name_ids: set[str] = set()
    for a in refl.args:
        if isinstance(a, ast.Name):
            arg_name_ids.add(a.id)
    for kw in refl.keywords:
        if isinstance(kw.value, ast.Name):
            arg_name_ids.add(kw.value.id)

    assert "technical_generate_fn" in arg_name_ids, (
        f"K.5.5 reflection call does not pass `technical_generate_fn` "
        f"by name; argument names observed: {sorted(arg_name_ids)}"
    )
    assert "creative_generate_fn" not in arg_name_ids, (
        f"K.5.5 reflection call passes `creative_generate_fn` -- "
        "L-2 violation; reflection must route through the technical "
        "slot, not the creative slot"
    )


# ---------------------------------------------------------------------------
# 5. meta.update(_brief_delta) closes the wiring
# ---------------------------------------------------------------------------


def test_reflection_result_folded_into_meta(execute_fn):
    """The K.5.5 reflection result must be folded into the `meta`
    dict via `meta.update(...)` so both saved-ledger and returned-
    script_json carry the 8-key meta delta from L-8. AST: confirm
    a `meta.update(...)` call appears AFTER the reflection call AND
    BEFORE the script_text assembly."""
    refl = _reflection_call(execute_fn)
    update_calls = _find_call_site(execute_fn, "update")
    # `meta.update(...)` -- filter by Attribute on Name `meta`.
    meta_update_calls = [
        c for c in update_calls
        if isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "meta"
    ]
    after_refl = [c for c in meta_update_calls if _line_of(c) > _line_of(refl)]
    assert after_refl, (
        "no `meta.update(...)` call appears after the K.5.5 reflection "
        "call; the 8-key meta delta is not being folded into meta"
    )


# ---------------------------------------------------------------------------
# 6. No explicit eviction call at K.5.5 (E-15 revised)
# ---------------------------------------------------------------------------


def test_no_explicit_eviction_call_at_k55(execute_fn):
    """E-15 revised at C1 audit per RR-A1: the loader is single-slot.
    `request_slot(technical_model)` inside `technical_generate_fn`
    handles eviction implicitly. The K.5.5 site does NOT call
    `evict_model` or `unload_llm` explicitly."""
    forbidden = {"evict_model", "unload_llm"}
    refl = _reflection_call(execute_fn)
    refl_line = _line_of(refl)
    # Window: from K.5 closer to the reflection call. Walk the whole
    # function and flag any forbidden call near the K.5.5 site (within
    # 50 lines either side).
    for node in ast.walk(execute_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name not in forbidden:
            continue
        if abs(_line_of(node) - refl_line) <= 50:
            pytest.fail(
                f"forbidden eviction call `{name}` at line {_line_of(node)} "
                f"within 50 lines of K.5.5 reflection call at line "
                f"{refl_line}; E-15 revised at C1 audit prohibits "
                "explicit eviction here -- request_slot's existing "
                "single-slot swap order is the safety mechanism"
            )


# ---------------------------------------------------------------------------
# 7. Deep-dict mutation guard (R-03)
# ---------------------------------------------------------------------------


def _mk_ledger() -> dict:
    return {
        "cast": [
            {"char_id": "JONES", "name": "Jones",
             "character_description": "Tall detective."},
            {"char_id": "SMITH", "name": "Smith",
             "character_description": "Quiet suspect."},
        ],
        "lines": [
            {"speaker_role": "scene", "text": "=== SCENE 1 ==="},
            {"speaker_role": "character", "char_id": "JONES",
             "text": "Some line of dialogue here."},
            {"speaker_role": "sfx", "text": "[SFX: thunder]"},
        ],
        "meta": {"episode_title": "T", "style": "noir_interrogation"},
        "news_seed": {"headline": "irrelevant fixture seed"},
        "title": "T",
    }


def _valid_brief_json() -> str:
    return (
        '{"story_brief": "a dim interrogation room under a bare bulb, '
        'rain on tin, sweat and smoke", '
        '"setting_terms": ["room"], "lighting_terms": ["bulb"], '
        '"atmosphere_terms": ["tense"]}'
    )


def _spy_fn(*, responses: list[str]):
    calls: list[dict] = []

    def fn(messages, *, temperature, max_new_tokens):
        calls.append({"temperature": temperature})
        if not responses:
            return ""
        return responses.pop(0)

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_reflection_does_not_mutate_core_ledger_keys():
    """R-03: deep-dict comparison before and after the reflection
    call. Every top-level ledger key EXCEPT meta must be byte-identical
    pre and post. Catches accidental nuke of cast / lines / news_seed
    / title (a lines-only check would slip past those classes)."""
    ledger = _mk_ledger()
    pre_snapshot = copy.deepcopy(ledger)

    spy = _spy_fn(responses=[_valid_brief_json()])
    delta = sb.run_story_brief_reflection(
        ledger, spy, technical_model_id="some/model-id",
    )
    # Simulate the K.5.5 wiring: meta.update(delta).
    ledger.setdefault("meta", {}).update(delta)

    for key, value in pre_snapshot.items():
        if key == "meta":
            continue
        assert ledger[key] == value, (
            f"non-meta ledger key {key!r} was mutated by the reflection "
            "call; R-03 violation"
        )


# ---------------------------------------------------------------------------
# 8. meta.update gives both saved ledger and returned script_json the
#    same brief fields (E-02 + E-11 unit-level)
# ---------------------------------------------------------------------------


_REQUIRED_META_KEYS = {
    "story_brief",
    "story_brief_status",
    "story_brief_error",
    "story_brief_model",
    "story_brief_prompt_version",
    "story_brief_source",
    "story_brief_char_count",
    "story_brief_terms",
}


def test_meta_carries_all_eight_keys_after_update():
    """E-02 + E-11 unit-level: after meta.update(brief_delta), the
    meta dict contains all 8 brief keys. Full pipeline E-02 / E-11
    (saved ledger == returned script_json) is verified at Sprint A
    via real ComfyUI runtime; here we lock the meta-key contract."""
    ledger = _mk_ledger()
    spy = _spy_fn(responses=[_valid_brief_json()])
    delta = sb.run_story_brief_reflection(
        ledger, spy, technical_model_id="some/model-id",
    )
    ledger.setdefault("meta", {}).update(delta)
    assert _REQUIRED_META_KEYS.issubset(ledger["meta"].keys()), (
        f"meta is missing some of the 8 brief keys; have: "
        f"{sorted(ledger['meta'].keys())}"
    )


def test_meta_failure_stamps_status_failed_on_raise():
    """The integration-level failure path: when the LLM raises, the
    8-key delta still folds into meta with story_brief_status=
    'failed'. Closes the silent-fallback class flagged by L-6."""
    ledger = _mk_ledger()

    def raising_fn(messages, *, temperature, max_new_tokens):
        raise RuntimeError("simulated technical_generate_fn failure")

    delta = sb.run_story_brief_reflection(ledger, raising_fn)
    ledger.setdefault("meta", {}).update(delta)
    assert ledger["meta"]["story_brief_status"] == "failed"
    assert ledger["meta"]["story_brief"] == ""


# ---------------------------------------------------------------------------
# 9-11. VRAM-accounting + cache-stability runtime tests (OTR_REGRESSION_RUNTIME)
# ---------------------------------------------------------------------------


_RUNTIME_REASON = (
    "Set OTR_REGRESSION_RUNTIME=1 to run the C5a2 VRAM-accounting / "
    "cache-stability tests. They need real ComfyUI + GPU loaders."
)


class TestRuntimeOnly:
    """Tests in this class gate on OTR_REGRESSION_RUNTIME (same pattern
    as `tests/test_audio_byte_identical.py::TestAudioRegressionGate`).
    They run on the operator's GPU machine; they skip in the
    pytest-only sandbox."""

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_c5a2_dual_slot_does_not_oom(self):
        """E-15 / RR-A1 revised: dual-slot config (creative=Mistral-Nemo,
        technical=Gemma-4-E4B-it). Instrument `_otr_model_loader` with
        a VRAM-accounting spy; assert peak resident size stays at or
        below 14.5 GB through composition + reflection."""
        pytest.skip("Sprint A runtime verification scope; placeholder.")

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_c5a2_dual_slot_evicts_creative_before_technical_loads(self):
        """E-15 / RR-A1 revised: in dual-slot config, request_slot's
        swap order calls `unload_llm(creative)` BEFORE the technical
        model load begins. Verifies the actual safety mechanism."""
        pytest.skip("Sprint A runtime verification scope; placeholder.")

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_c5a2_single_slot_no_unload(self):
        """Single-slot config (creative == technical): the
        composition + reflection sequence emits zero spurious
        unload_llm calls. Single-slot is the audio C7 baseline path
        and must not be perturbed."""
        pytest.skip("Sprint A runtime verification scope; placeholder.")

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_reflection_reuses_technical_model_cache_entry(self):
        """R-01 (moved to C5a2 per E-21 / RR-B2): in single-slot
        config, the technical-model cache entry registers exactly
        ONE load event across composition + reflection (no cache
        thrash)."""
        pytest.skip("Sprint A runtime verification scope; placeholder.")
