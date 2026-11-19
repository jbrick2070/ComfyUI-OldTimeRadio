"""The pack silences huggingface_hub through a path that exists (audit B18).

`import huggingface_hub.utils._logging` raises ModuleNotFoundError on
huggingface_hub 1.x and the pack swallowed it, so the Hub logger was never set
to ERROR. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest

INIT = Path(__file__).resolve().parents[1] / "__init__.py"


def test_the_pack_uses_the_public_logging_module():
    src = INIT.read_text(encoding="utf-8")
    assert "huggingface_hub.utils._logging" not in src
    assert "from huggingface_hub.utils import logging as hfh_logging" in src
    assert 'otr_env.setdefault("HF_HUB_VERBOSITY", "error")' in src


def test_the_public_module_really_sets_the_hub_logger():
    hub_logging = pytest.importorskip("huggingface_hub.utils").logging
    hub_logging.set_verbosity_error()
    assert logging.getLogger("huggingface_hub").getEffectiveLevel() == logging.ERROR
    importlib.import_module("huggingface_hub.utils.logging")
