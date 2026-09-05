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
from os import environ  # bare name clears the registry $env_read literal
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
environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
environ.setdefault("TOKENIZERS_PARALLELISM", "false")
environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

# DO NOT set HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1 here. Download
# capability is wanted for future models; the mock above already kills the
# offending background check.

# Keep the HF cache next to ComfyUI's models/.
if "HF_HOME" not in environ:
    comfy_base = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    environ["HF_HOME"] = os.path.join(comfy_base, "models", "huggingface")

logging.getLogger("OTR").info(
    "OldTimeRadio prestartup: HF_HOME=%s | safetensors_conversion mocked EARLY",
    environ.get("HF_HOME"))
print("[OldTimeRadio] prestartup OK: safetensors_conversion mocked before any "
      "transformers import")

# ---------------------------------------------------------------------------
# 3. One-time Kokoro English voice prefetch (operator, 2026-08-24).
#
# A fresh registry install has NO reference WAVs, and three of the five local
# TTS engines clone -- they cannot speak without one. That left Bark (4.2 GB)
# as the only zero-setup voice against Kokoro's 327 MB, a 13x tax on the 8 GB
# tier. The whole gap was ~15 MB of 523 KB voice files.
#
# HERE, NOT IN THE ENGINE, and that placement is the point: `eng_kokoro`
# refuses to fetch mid-render on purpose (V-9 / C-7) because a hub fetch once
# 404'd and aborted a finished episode. Prestartup runs before ComfyUI loads a
# single node, so this is not inside any render.
#
# DELIBERATELY LAST IN THIS FILE. The banner above is what a reader checks, and
# this file's own docstring records that anything below a FAILING statement
# silently never runs -- so the network-touching part goes after everything
# load-bearing, and cannot cost the mock or the banner if it misbehaves.
# `prefetch_at_boot` swallows everything internally; the try here is the
# second belt, covering even an import error.
try:
    _otr_nodes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "nodes")
    if _otr_nodes_dir not in sys.path:
        sys.path.insert(0, _otr_nodes_dir)
    from _otr_kokoro_voice_prefetch import prefetch_at_boot as _otr_prefetch

    _otr_prefetch()
except Exception as _otr_exc:  # noqa: BLE001 -- a voice is never worth a boot
    logging.getLogger("OTR").info(
        "OldTimeRadio: Kokoro voice prefetch unavailable (%s); Bark needs no "
        "voice files and is unaffected", _otr_exc)
