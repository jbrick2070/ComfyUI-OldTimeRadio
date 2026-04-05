"""
prestartup_script.py — Runs before ComfyUI imports any node modules.

Per survival guide Section 44: Set environment variables that must be
in place before transformers/torch read their defaults.

This script is automatically detected and executed by ComfyUI at startup.
"""

import os
import logging

# Suppress transformers warnings that clutter ComfyUI console
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure HuggingFace cache goes to a sensible location (Section 31)
# Only set if not already configured by the user
if "HF_HOME" not in os.environ:
    # Default: alongside ComfyUI models directory
    comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hf_cache = os.path.join(comfy_base, "models", "huggingface")
    os.environ["HF_HOME"] = hf_cache

logging.getLogger("OTR").info("OldTimeRadio prestartup: HF_HOME=%s", os.environ.get("HF_HOME"))
