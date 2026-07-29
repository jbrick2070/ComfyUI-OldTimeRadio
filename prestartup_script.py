"""prestartup_script.py -- runs BEFORE ComfyUI imports ANY node module.

Early mock for ``transformers.safetensors_conversion``, injected into
``sys.modules`` before ComfyUI begins loading custom nodes, so the background
conversion check never spawns its JSONDecodeError thread.

ASCII-ONLY, AND THAT IS LOAD-BEARING HERE (2026-07-29). This file used
em-dashes, box-drawing rules and a check-mark emoji, and the closing ``print``
raised ``UnicodeEncodeError: 'charmap' codec can't encode character '\\u2705'``
on a cp1252 Windows console -- so EVERY boot logged

    [ERROR] Failed to execute startup-script: ... prestartup_script.py
    0.0 seconds (PRESTARTUP FAILED): ... ComfyUI-OldTimeRadio

The mock itself had already been installed by then, so the pack worked and the
banner lied. Two costs: a permanent red herring in the boot log for whoever
reads it next, and a silent trapdoor -- anything added BELOW that print would
never have run, and nothing would have said so. The repo's UTF-8/no-BOM/
ASCII-only rule exists for exactly this, and a prestartup script is the worst
place to break it, because it runs before any of the logging that would
explain it.
"""

import logging
import os
import sys
import types

# ---------------------------------------------------------------------------
# 1. EARLIEST POSSIBLE MOCK -- runs before ANY transformers import.
#    The fake module goes into sys.modules before ComfyUI begins loading
#    custom nodes, so the real one is never imported.
# ---------------------------------------------------------------------------
_mock_sc = types.ModuleType("transformers.safetensors_conversion")
_mock_sc.auto_conversion = lambda *a, **kw: None
_mock_sc.get_conversion_pr_reference = lambda *a, **kw: None
_mock_sc.spawn_conversion = lambda *a, **kw: None
# Also mock the entry points that have appeared in recent transformers.
_mock_sc._get_conversion_pr_reference = lambda *a, **kw: None
_mock_sc._auto_conversion = lambda *a, **kw: None

sys.modules["transformers.safetensors_conversion"] = _mock_sc

# ---------------------------------------------------------------------------
# 2. Environment (secondary -- the mock above is what actually stops it)
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

# DO NOT set HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1 here. Download
# capability is wanted for future models; the mock above already kills the
# offending background check.

# Keep the HF cache next to ComfyUI's models/.
if "HF_HOME" not in os.environ:
    comfy_base = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HOME"] = os.path.join(comfy_base, "models", "huggingface")

logging.getLogger("OTR").info(
    "OldTimeRadio prestartup: HF_HOME=%s | safetensors_conversion mocked EARLY",
    os.environ.get("HF_HOME"))
print("[OldTimeRadio] prestartup OK: safetensors_conversion mocked before any "
      "transformers import")
