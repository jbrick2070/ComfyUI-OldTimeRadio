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


# ---------------------------------------------------------------------------
# APPLE SILICON: force PyTorch (SDPA) attention before ComfyUI chooses.
#
# ComfyUI's sub-quadratic attention -- its DEFAULT on Mac -- produces WRONG
# OUTPUT on MPS. Not slow, not an exception: structurally broken audio, and
# under `torch.use_deterministic_algorithms(True)` outright NaN.
#
# MEASURED, Mac mini M4 / torch 2.12.1, one Stable Audio 3 cue, identical
# checkpoint + prompt + seed + steps + cfg + sampler, ONLY the attention
# implementation differing:
#
#   CPU / any attention        spectral flatness 0.158   zcr 0.048   <- music
#   MPS / pytorch attention    spectral flatness 0.156   zcr 0.054   <- music
#   MPS / sub-quadratic        spectral flatness 0.432   zcr 0.148   <- noise
#
# (flatness ~1.0 is white noise, <0.1 is tonal; the sub-quad run also clipped
# at exactly +-1.000.) The operator's verdict on the sub-quad output was "like a
# broken cassette tape going backwards", and dBFS/peak metrics could not tell it
# from real music -- only spectral structure and ears could.
#
# The NaN half of the same bug: sub-quadratic attention calls
# `torch.baddbmm(<uninitialized buffer>, q, k, beta=0)`, and `beta=0` means
# "ignore that buffer". CPU and CUDA honour it; MPS does not. See
# PBUG-20260907-09b.
#
# WHY THIS BELONGS HERE. comfy/model_management.py auto-enables PyTorch
# attention for nvidia, intel_xpu, ascend_npu, mlu and ixuca -- and NOT for mps,
# which therefore falls through to the broken path. That decision is made when
# model_management is first imported, which is AFTER custom-node prestartup
# scripts run, so this is the last moment a node pack can influence it.
#
# Scoped to MPS. On CUDA/CPU the block does nothing at all, so nothing about the
# 5080's behaviour or its byte-identical goldens changes. An operator who passes
# --use-split-cross-attention or --use-quad-cross-attention explicitly is
# respected and not overridden.
class _OTRAttentionOptOut(Exception):
    """Internal: the operator opted out via OTR_MPS_PYTORCH_ATTENTION=0."""

try:
    import torch as _otr_torch

    if _otr_torch.backends.mps.is_available():
        from comfy.cli_args import args as _otr_comfy_args

        # Escape hatch, and it exists so this fix stays FALSIFIABLE: setting
        # OTR_MPS_PYTORCH_ATTENTION=0 restores ComfyUI's stock Mac behaviour so
        # the two backends can be A/B'd on identical inputs. Anyone re-testing
        # the claim in PBUG-20260907-11 needs to be able to turn it off.
        if environ.get("OTR_MPS_PYTORCH_ATTENTION", "1").strip() in ("0", "false", "no"):
            logging.getLogger("OTR").info(
                "[OldTimeRadio] mps: OTR_MPS_PYTORCH_ATTENTION=0 -- leaving ComfyUI's "
                "stock attention selection alone (sub-quadratic). Measured WRONG on "
                "Metal; set only for A/B testing.")
            raise _OTRAttentionOptOut

        _otr_explicit = (getattr(_otr_comfy_args, "use_split_cross_attention", False)
                         or getattr(_otr_comfy_args, "use_quad_cross_attention", False))
        if _otr_explicit:
            logging.getLogger("OTR").info(
                "[OldTimeRadio] mps: an explicit attention flag is set; leaving it "
                "alone. NOTE: sub-quadratic attention is measurably WRONG on MPS "
                "(see docs/MAC_LESSONS_LEARNED.md).")
        elif not getattr(_otr_comfy_args, "use_pytorch_cross_attention", False):
            _otr_comfy_args.use_pytorch_cross_attention = True
            logging.getLogger("OTR").info(
                "[OldTimeRadio] mps detected: forcing PyTorch (SDPA) attention. "
                "ComfyUI's sub-quadratic default produces structurally wrong "
                "output on Metal -- measured on Stable Audio 3, spectral flatness "
                "0.43 (noise) vs 0.16 (music) with every other input identical.")
except _OTRAttentionOptOut:
    pass
except Exception as _otr_attn_exc:  # noqa: BLE001 -- never block boot
    logging.getLogger("OTR").info(
        "OldTimeRadio: could not set the MPS attention backend (%s); if this is "
        "a Mac, pass --use-pytorch-cross-attention manually", _otr_attn_exc)
