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

# BUG-006 fix: force HuggingFace Hub offline mode by default. All models the
# OldTimeRadio pipeline uses (Gemma 4, Bark, Parler) are pre-cached. Allowing
# online lookups spawns a background `Thread-auto_conversion` that prints HF
# rate-limit warnings to stderr — those warnings interleave with Gemma's
# stdout streamer and contaminate the visible script log. Forcing offline mode
# eliminates the background thread entirely. Users who genuinely need to
# download a new model can set HF_HUB_OFFLINE=0 in their shell before launch.
# HF_HUB_OFFLINE: intentionally NOT set here. The user has HF_TOKEN configured
# as a Windows env var (setx HF_TOKEN ...) which authenticates requests and
# raises rate limits to 5000/hr. Forcing offline mode would override the token
# and break model downloads. The auto_conversion background thread is suppressed
# by the token itself (authenticated requests don't hit the unauthenticated
# rate limit that was causing the JSONDecodeError in the background thread).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

# Ensure HuggingFace cache goes to a sensible location (Section 31)
# Only set if not already configured by the user
if "HF_HOME" not in os.environ:
    # Default: alongside ComfyUI models directory
    comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hf_cache = os.path.join(comfy_base, "models", "huggingface")
    os.environ["HF_HOME"] = hf_cache

logging.getLogger("OTR").info("OldTimeRadio prestartup: HF_HOME=%s", os.environ.get("HF_HOME"))
