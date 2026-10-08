"""The two production importers that free local LLM VRAM use the modern handoff.

`nodes/_otr_bark_lib.py` and `nodes/scene_sequencer.py` release the local LLM
through `_otr_model_loader.unload_llm` / `unload_llm_if_local_resident`. The
orchestrator's own copies of the LLM load/unload/cache/generate helpers are
gone (S31 B4), so this file keeps only the positive wiring pin: each importer
must actually import a modern unload helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent

_IMPORTER_PATHS = (
    # nodes/batch_bark_generator.py retired in the audio clean-break (1a); the
    # bark inference lib (_otr_bark_lib.py) carries the unload_llm handoff now.
    "nodes/_otr_bark_lib.py",
    "nodes/scene_sequencer.py",
)


@pytest.mark.parametrize("rel_path", _IMPORTER_PATHS)
def test_importers_use_new_unload_path(rel_path):
    """Production importers must use a modern `_otr_model_loader` unload
    handoff. Cloud-safe handoff sites may import `unload_llm_if_local_resident`;
    local-only handoff sites may import `unload_llm`.
    """
    path = PACK_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    new_imports: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for imp in node.names:
            if (
                module.endswith("_otr_model_loader")
                and imp.name in ("unload_llm", "unload_llm_if_local_resident")
            ):
                new_imports.append(f"line {node.lineno}")
    assert new_imports, (
        f"{rel_path} must import a modern unload helper from "
        f"`_otr_model_loader` to free local LLM VRAM; no such import found"
    )
