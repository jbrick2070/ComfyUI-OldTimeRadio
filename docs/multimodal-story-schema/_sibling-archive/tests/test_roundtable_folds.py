"""Tests for the roundtable pass01 folds (gap-hunt panel: Grok 4.3 +
Kimi k2.6 + DeepSeek v4-pro; claims grounded before folding)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from upstream_story_lab.compat import file_sha256
from upstream_story_lab.registry import Registry, RegistryError
from upstream_story_lab.runner import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_runner_injects_context_and_chains_outputs(registry) -> None:
    """DeepSeek/Kimi CONFIRMED: without context + chaining the 4-pass
    experiment is fiction. Pass 1 must see source + tone guardrails; every
    later pass must see the previous pass's output."""

    pack = registry.pack(
        "custom_source_bank", "simple_4_prompt_experimental",
        "simple_4_prompt_experimental",
    )
    pipeline = registry.pipeline("simple_4_prompt_experimental")
    prompts: dict[str, str] = {}

    def llm(slot, pass_id, prompt):
        prompts[pass_id] = prompt
        return f"<{pass_id} output>"

    run_pipeline(
        pack, pipeline, llm,
        source_material_text="A brittle transcription disc surfaces.",
        ledger_schema_text='{"cast": [], "lines": [], "beats": []}',
    )
    p1 = prompts["pass_1_creative_story"]
    assert "SOURCE MATERIAL:" in p1 and "TONE GUARDRAILS:" in p1
    assert "PREVIOUS PASS OUTPUT" not in p1
    p2 = prompts["pass_2_creative_ledger_fill"]
    assert "<pass_1_creative_story output>" in p2
    assert "LEDGER SCHEMA:" in p2
    p4 = prompts["pass_4_technical_ledger_audit"]
    assert "<pass_3_technical_schema_cleanup output>" in p4


def test_forbidden_patterns_stay_out_of_prompts(registry) -> None:
    """The negation-copy rule survives the fold: forbidden PATTERNS are
    never injected into any runner prompt (tone guardrails are)."""

    pack = registry.pack(
        "custom_source_bank", "simple_4_prompt_experimental",
        "simple_4_prompt_experimental",
    )
    pipeline = registry.pipeline("simple_4_prompt_experimental")
    seen: list[str] = []

    def llm(slot, pass_id, prompt):
        seen.append(prompt)
        return "ok"

    run_pipeline(pack, pipeline, llm)
    joined = "\n".join(seen).lower()
    for term in pack.forbidden_plot_patterns:
        assert term.lower() not in joined


def test_executable_pipeline_seam_coverage_is_cross_validated(tmp_path) -> None:
    """DeepSeek CONFIRMED: pack/pipeline combinations were never validated
    together. Removing a pass seam from the experimental pack must fail at
    LOAD, not inside the runner."""

    root = tmp_path / "lab"
    shutil.copytree(ROOT / "fixtures", root / "fixtures")
    path = (root / "fixtures" / "story_packs" / "experimental"
            / "simple_4_prompt_experimental.json")
    pack = json.loads(path.read_text(encoding="utf-8"))
    del pack["prompt_stages"]["pass_3_technical_schema_cleanup"]
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(RegistryError, match="does not cover executable pipeline"):
        Registry(root)


def test_mirror_files_match_manifest_hashes() -> None:
    """Kimi (half-confirmed: the manifest EXISTS, verification did not):
    every hash in PRODUCTION_MIRROR_MANIFEST.md must match the on-disk
    mirror file, so a tampered or partially-refreshed mirror cannot fake a
    green drift suite."""

    manifest = (ROOT / "PRODUCTION_MIRROR_MANIFEST.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^([0-9A-F]{16})\s{2}([^\s].*?)\s{2}\d+\s*$", manifest, flags=re.M,
    )
    assert len(rows) >= 20, "manifest hash table not found or too small"
    checked = 0
    for hash16, rel in rows:
        if "..." in rel:
            continue  # abbreviated doc rows; the full-path rows cover code
        target = ROOT / "production_mirror" / rel.replace("\\", "/")
        assert target.exists(), f"mirror file missing: {rel}"
        actual = file_sha256(target).upper()[:16]
        assert actual == hash16, (
            f"mirror file {rel} does not match its manifest hash - "
            "re-mirror and regenerate the manifest before trusting drift tests"
        )
        checked += 1
    assert checked >= 20
