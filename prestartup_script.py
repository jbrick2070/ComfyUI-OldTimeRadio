"""
prestartup_script.py — Runs BEFORE ComfyUI imports ANY node modules.
Nuclear early mock for transformers/safetensors_conversion to kill
the JSONDecodeError background thread once and for all.
"""

import io
import os
import sys
import types
import logging

# ─────────────────────────────────────────────────────────────────────────────
# 0. UTF-8 ENFORCEMENT — prevent Windows cp1252 from truncating/crashing logs.
#    ComfyUI Desktop's app/logger.py defaults to console encoding on Windows.
#    Non-ASCII in model names, prompts, or emoji will throw UnicodeEncodeError,
#    silently truncate, or crash. Force UTF-8 with replace before anything runs.
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["PYTHONIOENCODING"] = "utf-8:replace"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

if hasattr(sys, "__stdout__"):
    sys.__stdout__ = sys.stdout
if hasattr(sys, "__stderr__"):
    sys.__stderr__ = sys.stderr

# ─────────────────────────────────────────────────────────────────────────────
# 1. EARLIEST POSSIBLE MOCK — runs before ANY transformers import
#    This is the "nuclear" part. We inject the fake module into sys.modules
#    before ComfyUI even begins loading custom nodes.
# ─────────────────────────────────────────────────────────────────────────────
_mock_sc = types.ModuleType("transformers.safetensors_conversion")
_mock_sc.auto_conversion = lambda *a, **kw: None
_mock_sc.get_conversion_pr_reference = lambda *a, **kw: None
_mock_sc.spawn_conversion = lambda *a, **kw: None
# Also mock any other entry points that have appeared in recent transformers
_mock_sc._get_conversion_pr_reference = lambda *a, **kw: None
_mock_sc._auto_conversion = lambda *a, **kw: None

sys.modules["transformers.safetensors_conversion"] = _mock_sc



# ─────────────────────────────────────────────────────────────────────────────
# 2. Environment variables (still useful, but now secondary)
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

# DO NOT set HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1 here.
# We want download capability for future models. The mock above already
# kills the offending background check.

# Ensure HF cache lives next to ComfyUI models/
if "HF_HOME" not in os.environ:
    comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hf_cache = os.path.join(comfy_base, "models", "huggingface")
    os.environ["HF_HOME"] = hf_cache

logging.getLogger("OTR").info("OldTimeRadio prestartup: HF_HOME=%s | safetensors_conversion mocked EARLY", os.environ.get("HF_HOME"))
print("✅ [OldTimeRadio] prestartup: safetensors_conversion mocked before any transformers import")
