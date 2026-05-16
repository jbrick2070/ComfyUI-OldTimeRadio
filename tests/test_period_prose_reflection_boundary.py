"""Sprint D D3 -- reflection boundary tests.

When the writer's creative slot is set to a period model (talkie,
prompt_profile=otr_1940s_v1) the script `lines` produced carry
period-flavored dialogue. The technical slot's reflection pass
(run_story_brief_reflection) reads those lines and produces
meta.story_brief in MODERN ENGLISH. The boundary contract:

  1. _otr_story_brief.py does NOT import OTR_PERIOD_SYSTEM_PROMPT
     or render_few_shot_block or PeriodExemplar -- the reflection
     module is structurally isolated from the period prompt module.

  2. The reflection module's own system-prompt strings do NOT
     contain period slang. A regex sweeps period-diction keywords
     and asserts they're absent.

  3. The reflection module stamps `story_brief_model` from the
     technical_model_id parameter the writer passes, NEVER from
     the creative slot identity.

These are structural AST + regex checks; no LLM execution. The
runtime "does the reflection brief stay modern even when input
lines are period" pass belongs to Sprint A's empirical
verification step (per Sprint D pytest-only acceptance gate).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REFLECTION_SRC = REPO_ROOT / "nodes" / "_otr_story_brief.py"


def test_reflection_module_does_not_import_period_prompts() -> None:
    """`_otr_story_brief.py` must NOT import any symbol from
    `_otr_period_prompts`. If it does, period-specific strings could
    leak into the reflection prompt assembly.
    """
    tree = ast.parse(REFLECTION_SRC.read_text(encoding="utf-8"))
    bad_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "period_prompts" in mod or mod.endswith("_otr_period_prompts"):
                bad_imports.append(
                    f"line {node.lineno}: from {mod!r} import "
                    f"{[a.name for a in node.names]}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "period_prompts" in alias.name:
                    bad_imports.append(
                        f"line {node.lineno}: import {alias.name!r}"
                    )
    assert not bad_imports, (
        "_otr_story_brief.py imports from period_prompts module, "
        "breaking the reflection boundary contract. Found:\n  "
        + "\n  ".join(bad_imports)
    )


# Period slang regex -- words and phrases the OTR_PERIOD_SYSTEM_PROMPT
# explicitly enforces ("swell", "all right" instead of "okay",
# "fellows" instead of "guys", "say" as a greeting, the [SOUND:] /
# [SFX:] / [MUSIC:] stage-direction syntax). If any of these surface
# in the reflection module's hard-coded prompt strings, period flavor
# has leaked into the modern reflection contract.
_PERIOD_SLANG_RE = re.compile(
    r"\b(swell|fellows)\b"
    r"|\b1940s\b"
    r"|\bSuspense, Lights Out\b"
    r"|\bInner Sanctum\b"
    r"|tube radio"
    r"|\[SOUND:\s"
    r"|\[SFX:\s"
    r"|\[MUSIC:\s",
    flags=re.IGNORECASE,
)


def test_reflection_module_prompts_have_no_period_slang_bleed() -> None:
    """Walk the AST of `_otr_story_brief.py` and check every
    string-literal constant assigned at module scope. None of them
    should match the period-slang regex.
    """
    tree = ast.parse(REFLECTION_SRC.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        text = node.value.value
        if _PERIOD_SLANG_RE.search(text):
            target_names = [
                t.id for t in node.targets if isinstance(t, ast.Name)
            ]
            hits.append(
                f"line {node.lineno}: {target_names!r} = "
                f"...period-slang match..."
            )
    assert not hits, (
        "_otr_story_brief.py module-level prompt constants contain "
        "period slang; reflection boundary contract broken. Hits:\n  "
        + "\n  ".join(hits)
    )


def test_reflection_brief_records_technical_slot_model_in_story_brief_model_key() -> None:
    """The `story_brief_model` field in the reflection output must
    be sourced from `technical_model_id` (the parameter the writer
    passes), NEVER from the creative slot identity.

    AST inspection: find every `"story_brief_model":` key in a Dict
    literal and confirm the value references technical_model_id.
    """
    tree = ast.parse(REFLECTION_SRC.read_text(encoding="utf-8"))
    failures: list[str] = []
    found_at_least_one = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if not (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and k.value == "story_brief_model"
            ):
                continue
            found_at_least_one = True
            value_src = ast.unparse(v)
            if "technical_model_id" not in value_src:
                failures.append(
                    f"line {getattr(v, 'lineno', '?')}: "
                    f"story_brief_model = {value_src}; expected "
                    f"to reference technical_model_id"
                )
    assert found_at_least_one, (
        "no 'story_brief_model': ... key found in any Dict literal "
        "of _otr_story_brief.py; reflection output schema may have "
        "changed or the field was renamed"
    )
    assert not failures, "\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
