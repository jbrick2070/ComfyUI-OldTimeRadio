# -*- coding: utf-8 -*-
"""The template gallery names this pack "Old-Time Radio", however it was installed.

WHY (0d, 2026-09-25). The gallery's Extensions list labels each pack with
its custom_nodes FOLDER name, so a stranger saw "ComfyUI-OldTimeRadio" (git
clone) or "comfyui-old-time-radio" (registry install). Both halves of the
mechanism were read from source, not assumed:

* core -- ComfyUI app/custom_node_manager.py: `/workflow_templates` keys each
  template list by `os.path.basename` of the pack folder, and
  `build_translations()` merges every pack's `locales/<lang>/main.json` into
  the `/i18n` response;
* frontend 1.52.7 -- workflowTemplatesStore.ts labels a custom pack with
  `st(\`templateWorkflows.category.${normalizeI18nKey(moduleName)}\`,
  moduleName)`, where normalizeI18nKey replaces "." with "_".

So one key per install shape. Per-workflow titles are the filename stems and
are NOT localizable for custom packs in this frontend; they stay as they are.

Core caches build_translations(), so a running server shows the new label
only after a restart.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MAIN = REPO / "locales" / "en" / "main.json"
LABEL = "Old-Time Radio"

for _p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_variants as bv  # noqa: E402
from test_shipped_scripts_are_shipped import _is_excluded  # noqa: E402


def _categories():
    raw = MAIN.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8, no BOM"
    return json.loads(raw.decode("utf-8"))["templateWorkflows"]["category"]


def _registry_name():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text,
                        re.MULTILINE | re.DOTALL).group(1)
    return re.search(r'^name\s*=\s*"([^"]+)"', project, re.MULTILINE).group(1)


@pytest.mark.parametrize("folder", ["ComfyUI-OldTimeRadio", "comfyui-old-time-radio"])
def test_both_install_folder_names_are_labelled(folder):
    assert _categories()[folder] == LABEL


def test_the_registry_key_is_the_registry_name():
    """A registry install lands in a folder named for the registry id, which
    is pyproject's [project] name -- and the same id the nodes stamp."""
    name = _registry_name()
    assert name == bv.PACK_CNR_ID
    assert name in _categories()


def test_no_key_is_rewritten_by_the_frontend():
    for key in _categories():
        assert "." not in key, f"normalizeI18nKey would turn {key!r} into another key"


def test_the_locale_file_ships_in_the_registry_bundle():
    assert not _is_excluded("locales/en/main.json")
