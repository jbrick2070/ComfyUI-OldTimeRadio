"""LLM model catalog: curated set + local HF cache scan + dropdown builder + validator.

S30 B1a (offline). Pure dataclass + filesystem scan + structural
validator. No HuggingFace API calls fire from this module. B1a2 adds
the network surface (auto_download_if_missing, estimate_model_size_gb,
resolve_hf_token). B1b adds resolve_context_cap + HARD_VRAM_CONTEXT_LIMIT.
B1c adds check_vram_fit.

Catalog discipline:
  * CURATED_LLM_MODELS is the canonical curated set. Annotations
    (requires_auth, vram_fit_tier, loader_backend, approx_safetensors_gb)
    drive dropdown labels, error messages, and backend dispatch.
  * Only entries with vram_fit_tier == "PASS" are advertised in
    dropdown labels and gated-error recovery messages as "16 GB-ready."
  * Module-level constants (DEFAULT_LLM, TEST_TECHNICAL_LLM,
    TEST_OVERSIZED_LLM) are the single source of truth for tests + wiring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat / standalone load
    # STANDALONE LOAD, and the insert belongs HERE, not above the try.
    # `scripts/otr_provision.py` and `otr_make_portable_voice_bank.py` load this
    # file by PATH under a non-package name, deliberately -- "without importing
    # the ComfyUI node package". In that mode the relative rung has no parent
    # package and the flat one needs `nodes/` on sys.path, which those loaders
    # do not add.
    #
    # MUTATING sys.path ONLY IN THIS ARM IS THE POINT (cursor cross-check,
    # 2026-09-04). Doing it before the try would put `nodes/` at sys.path[0] on
    # the ORDINARY packaged import too, which is precisely the enablement
    # `otr_post_upscale_procgen_blend.py` warns about: it makes the flat
    # spelling resolvable everywhere and invites a SECOND module instance of
    # the owner. Here it runs only when the relative rung has already failed,
    # i.e. only when there is no package instance to duplicate.
    import os as _os_boot
    import sys as _sys_boot
    _NODES_DIR = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
    if _NODES_DIR not in _sys_boot.path:
        _sys_boot.path.insert(0, _NODES_DIR)
    from _otr_shared import env as otr_env  # type: ignore

# ---------------------------------------------------------------------------
# Canonical constants -- single source of truth for tests + wiring code.
# Any future rename / casing fix happens here, not in scattered string literals.
# ---------------------------------------------------------------------------

DEFAULT_LLM = "Qwen/Qwen3.5-4B"
"""Fallback for empty/unsaved writer inputs.

WAS ``mistralai/Mistral-Nemo-Instruct-2407`` until 2026-09-06. That row is the
single highest-friction writer in the catalog and it was the value a
freshly-dropped node fell back to: a **24 GB** download that then does NOT fit
an 8 GB card at all (12.0 GB resident badge). A default should be the row most
likely to work on the machine of someone who has changed nothing.

Qwen3.5-4B is that row, measured on a physical 8 GB RTX 4060 on 2026-09-06:
8.68 GB to download, **2.99 GiB resident** under NF4, **14.47 tok/s** -- the
fastest and smallest of every row tested -- ungated, Apache-2.0, and it carried
a complete one-act episode end to end (obs_publish OK, 32m34s) on that card.
Mistral-Nemo remains in the catalog and remains selectable; it is simply no
longer what you get by accident.

ALSO FIXES A LATENT MISMATCH. Every curated dropdown label is
``repo_id + vram_badge_for(repo_id)``, so the option list holds
``'... (12.0 GB)'`` while this constant is the BARE id -- the declared default
was never a member of its own option list, and ComfyUI fell through to whichever
row happened to sit at index 0. That fallback is undefined behaviour dressed as
a default. Consumers normalize through ``_strip_label_suffix``, so a bare id
here is correct; what was wrong was pointing it at a row an 8 GB user cannot run.
"""

TEST_TECHNICAL_LLM = "google/gemma-4-E2B-it"
"""Used by B6 routing tests + the manual VRAM-profile script to drive
Slot 1 != Slot 2. Compact multimodal-text-only technical option."""

TEST_OVERSIZED_LLM = "meta-llama/Llama-3.1-70B-Instruct"
"""Used by B1c VRAM-fit tests as a known-fails-on-16GB target. Not
added to the dropdown."""

# LEGACY dropdown state suffixes. As of 2026-07-16 the dropdown no longer
# appends any download-state badge (a per-user HF cache layout makes the
# "downloaded" state impossible to show correctly for everyone). These are
# RETAINED only so validate_model_id / _strip_label_suffix still normalize a
# value saved by an OLDER workflow (e.g. "mistralai/... [NOT DOWNLOADED]")
# back to the bare repo id. Do NOT append them to new labels.
NOT_DOWNLOADED_SUFFIX = " [NOT DOWNLOADED]"
LOCAL_HF_SUFFIX = " [LOCAL HF]"
LOCAL_GGUF_SUFFIX = " [LOCAL GGUF]"

# ---------------------------------------------------------------------------
# CuratedModel dataclass + curated set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratedModel:
    """A curated LLM the OTR catalog ships with explicit honesty fields.

    Sprint D D1a (2026-05-16): extended with 6 fields to support the
    period-LLM CATEGORY. `prompt_profile` routes the writer's creative
    slot to the period system prompt when set to `otr_1940s_v1`.
    `license` + `license_audit_status` mirror the per-repo audit at
    `docs/model-license-<sanitized>.md`. `chat_template_kind` +
    `stop_tokens` + `context_window` carry the per-backend dispatch
    hints that loader adapters consume. All curated rows currently
    carry `prompt_profile="modern"`; the `otr_1940s_v1` profile is
    reserved for a period model and is not bound to any curated row
    at present.
    """

    repo_id: str
    requires_auth: bool  # gated repo -> True
    loader_backend: Literal[
        "transformers_safetensors",
        "transformers_multimodal_text_only",
        "transformers_gptq_int4",
        "openrouter_http",
        "comfy_credits_http",
        "google_api_http",
        "gguf_native",
    ]
    vram_fit_tier: Literal["PASS", "WARN", "UNKNOWN", "FAIL"]
    approx_safetensors_gb: float  # download size on disk, not VRAM resident
    notes: str = ""
    # Sprint D D1a fields with safe defaults so any future row written
    # against the pre-D1a schema (or any test fixture that omits these)
    # still constructs cleanly. Production rows below set them explicitly.
    prompt_profile: Literal["modern", "otr_1940s_v1"] = "modern"
    chat_template_kind: Literal[
        "transformers_default", "manual", "raw_completion",
    ] = "transformers_default"
    stop_tokens: tuple[str, ...] = ()
    context_window: int = 8192
    license: Literal[
        "mit", "apache_2_0", "non_commercial", "community", "gated_terms",
    ] = "mit"
    license_audit_status: Literal[
        "mit_equivalent", "research_lane", "pending",
    ] = "pending"
    # Remote-LLM provider tag. "local" = the transformers/HF weight path
    # every existing row uses; "openrouter" = a virtual row behind the
    # own-key OpenRouter API (S2); "comfy_credits" = a virtual row behind
    # ComfyUI's credit-billed partner-node proxy (2026-06-01);
    # "google_api" = a virtual row behind the user's Gemini API key;
    # "gguf_native" = a virtual row backed by an in-process llama-cpp-python
    # GGUF loader. It is local VRAM, not a remote/HTTP zero-VRAM row.
    # Default "local" so every pre-existing row and any older fixture that
    # omits the field still constructs unchanged.
    provider: Literal[
        "local", "openrouter", "comfy_credits", "google_api", "gguf_native",
    ] = "local"
    # How a MULTIMODAL checkpoint driven text-only is actually loaded.
    #
    # "composite"           -- load the whole checkpoint, towers included.
    # "native_text_decoder" -- load ONLY the text decoder; the vision/audio
    #                          towers are never materialized.
    #
    # THIS IS A SEPARATE FIELD FROM loader_backend ON PURPOSE, and the reason
    # is a live near-miss (PBUG-20260906-07). `loader_backend ==
    # "transformers_multimodal_text_only"` looks like the natural dispatch key
    # and is NOT sufficient: `google/gemma-4-12b-it` carries that exact value,
    # so dispatching on it would silently change the 16 GB box's qualified
    # canonical writer while fixing an 8 GB one. Opting a row in is a
    # deliberate, per-row, reviewable decision -- never inferred.
    #
    # Default "composite" keeps every pre-existing row, and any fixture that
    # omits the field, loading exactly as before.
    text_only_load: Literal["composite", "native_text_decoder"] = "composite"


CURATED_LLM_MODELS: tuple[CuratedModel, ...] = (
    CuratedModel(
        repo_id="Qwen/Qwen3.5-4B",
        requires_auth=False,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="WARN",
        # Official shards: 9,319,828,096 bytes / 2**30. Disk, not VRAM.
        approx_safetensors_gb=8.68,
        notes="Official Apache-2.0, ungated Qwen3.5 text-only native "
        "Transformers lane; ordinary NF4 policy. THINKING template, suppressed "
        "-- read from the published chat_template.jinja 2026-09-06: with "
        "add_generation_prompt it emits a closed '<think>\\n\\n</think>' "
        "envelope when enable_thinking is false and an OPEN '<think>' "
        "otherwise, so chat_template_kwargs must reach every generate call or "
        "the model is forced to reason (see test_chat_template_kwargs_wired). "
        "Two shards, 9,319,828,096 bytes. 8GB speed, memory and episode "
        "qualification pending; not soak-tested.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
        # Vision tower is dead weight for the writer. Transformers' own
        # conversion registry already strips this family's prefix
        # (conversion_mapping.py: "qwen3_5_text" -> language_model/model), so
        # OTR supplies no key_mapping of its own here.
        text_only_load="native_text_decoder",
    ),
    CuratedModel(
        repo_id="unsloth/Llama-3.2-3B-Instruct",
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="WARN",
        # 6,425,499,648 bytes of safetensors = 5.98 GiB. Disk, not VRAM.
        approx_safetensors_gb=6.43,
        notes="THE NO-QUANTIZATION ROW. Exists for hosts where bitsandbytes "
        "is unavailable -- AMD/ROCm above all -- because every other curated "
        "row needs 4-bit to fit 8 GB and NF4 is a compiled-CUDA path. At "
        "3,212,749,824 params it is 5.98 GiB in bf16, so it fits a 7.99 GiB "
        "card UNQUANTIZED with roughly 2 GiB spare, and needs no bitsandbytes "
        "at all. Plain llama architecture, AutoModelForCausalLM, safetensors, "
        "no trust_remote_code. "
        "MIND THE BADGE: vram_badge_for halves approx_safetensors_gb, which "
        "assumes a 4-bit load, so the picker will show ~3.2 GB while an "
        "unquantized load really costs 5.98 GiB. The badge understates this "
        "row on exactly the hosts it is for. "
        "LICENCE IS NOT PERMISSIVE -- Llama 3.2 Community Licence, see "
        "docs/model-license-unsloth--llama-3.2-3b-instruct.md. Ungated on this "
        "mirror (meta-llama's own repo is gated), verified 2026-09-06. "
        "8 GB speed, memory and episode qualification all pending; no AMD "
        "hardware has run it.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="community",
        license_audit_status="research_lane",
    ),
    CuratedModel(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        # 2026-08-25: was True. The Hugging Face API reports `"gated": false`
        # for this repo -- the flag was demanding an HF_TOKEN for a model that
        # downloads freely, so a fresh install with no token hit
        # GatedModelError on the DEFAULT row and never reached the writer.
        # Only `google/gemma-2-2b-it` is genuinely gated (`"gated": "manual"`).
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="PASS",
        approx_safetensors_gb=24.0,
        notes="Audio C7 regression baseline -- soak-tested. Default for both "
        "slots. Effective context window raised to 16384 (2026-07-19) so the "
        "the local sci-fi 420/720w script pass fits; short legs are unaffected "
        "(they never reached the 8192 output-budget clamp) so C7 audio "
        "byte-identity holds.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=16384,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
    ),
    CuratedModel(
        repo_id="google/gemma-4-E2B-it",
        # 2026-08-25: was True. Gemma 4 is Apache-2.0 and UNGATED, unlike
        # Gemma 2/3 -- the HF API reports `"gated": false`. This flag was the
        # likeliest silent failure on a fresh 8 GB install, because the small
        # writer rows an 8 GB card must use were all refusing without a token.
        requires_auth=False,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="PASS",
        approx_safetensors_gb=6.0,
        notes="Multimodal architecture (matformer / Gemma-3n family) used "
        "in text-only mode. Compact technical-slot option. Of its 2011 "
        "checkpoint tensors only 600 are the text decoder; 1410 are audio and "
        "vision towers the writer never executes, which is why this row loads "
        "the native text decoder (PBUG-20260906-07).",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
        text_only_load="native_text_decoder",
    ),
    CuratedModel(
        repo_id="google/gemma-4-E4B-it",
        # 2026-08-25: was True. Gemma 4 is Apache-2.0 and UNGATED
        # (HF API: `"gated": false`). See the E2B row above.
        requires_auth=False,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="PASS",
        approx_safetensors_gb=9.0,
        notes="Slightly larger technical option, same backend.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
    ),
    CuratedModel(
        repo_id="google/gemma-4-12b-it",
        requires_auth=False,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="PASS",
        approx_safetensors_gb=23.9,
        notes="Official Gemma4Unified in-process Transformers text lane "
        "(transformers>=5.10.4). Fully offline from the canonical HF cache; "
        "NF4 measured at 7.15 GiB allocated / 7.29 GiB peak on the 16 GB "
        "RTX 5080, including coherent prose and LMFE-constrained JSON. No "
        "LoRA, Ollama, llama.cpp, sidecar, or port.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
    ),
    CuratedModel(
        repo_id="google/gemma-2-2b-it",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="PASS",
        approx_safetensors_gb=5.2,
        notes="Smallest curated technical-slot pick (2B, NF4 -- tiny "
        "VRAM). Gemma-2 has no system role by design; the writer "
        "generate path folds system content into the first user "
        "turn via normalize_messages_for_tokenizer (BUG-LOCAL-262). "
        "Gemma-2 ships under the restricted Gemma Terms of Use "
        "(NOT Apache 2.0 -- only Gemma 4 is) so the row is "
        "research-lane: technical-slot use only, not bound in the "
        "default creative-binding workflow JSON.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="gated_terms",
        license_audit_status="research_lane",
    ),
    # 2026-08-25: catalog pruned -- Qwen/Qwen2.5-14B-Instruct removed
    # (operator: "if it doesn't fit nicely or requires Ollama rip it from
    # the dropdown and blast radius"; "I only want easy to load LLMs").
    # It was the LAST WARN-tier row and the only curated row with no
    # weights on disk, so it was a dropdown entry that could not have
    # loaded if anyone picked it. Its own note conceded the case: 28 GB of
    # safetensors "needs quantization or offload to fit 16 GB -- not
    # soak-tested as PASS yet. Available for users with bigger rigs."
    # A dropdown row is a promise the model will load; that one could not
    # keep it on this hardware. Nothing required Ollama -- the GGUF lane is
    # in-process llama-cpp-python -- so that half of the sweep had no
    # targets. See docs/LLM_PREFLIGHT_GUIDE.md for the seven gates a new
    # row must clear, and test_every_curated_local_row_is_pass_tier for the
    # invariant that keeps a WARN row from returning silently.
    # 2026-05-23: catalog pruned -- the two community WARN-tier 12B
    # rows (Captain-Eris_Violet-V0.420-12B, MN-12B-Mag-Mell-R1) were
    # removed. The curated set now also includes the official Gemma 4 12B HF
    # row restored in 2026-07; the optional GGUF peer remains a separate lane.
    # 2026-05-24: gemma-2-2b-it added as the smallest technical-slot
    # pick (BUG-LOCAL-262). Gemma-2's chat template rejects the system
    # role; the generate path normalizes system messages before
    # apply_chat_template so the row is a clean technical pick.
    # No otr_1940s_v1 period row is curated at present. The broken
    # talkie-lm/talkie-1930-13b-it row was removed 2026-05-22 (raw
    # research checkpoint -- no config.json / tokenizer -- crashed the
    # writer at the style picker). The period-routing surface
    # (otr_1940s_v1 profile, GPTQ-int4 backend, _otr_period_prompts)
    # stays parked for a future period model; see the ROADMAP
    # period-model strategy section.
)


def _openrouter_virtual_rows() -> tuple[CuratedModel, ...]:
    """The two virtual OpenRouter rows (S2) -- present ONLY when remote
    is enabled (OPENROUTER_API_KEY set; C6: the OTR_ENABLE_OPENROUTER opt-in
    flag gate was removed). When
    disabled the tuple is empty, so _by_repo_id / dropdowns / Path-1
    validation never see them and the offline baseline is untouched (C3,
    C8). Per FC4 these carry loader_backend='openrouter_http',
    vram_fit_tier='PASS', approx_safetensors_gb=0.0, context_window=8192,
    provider='openrouter'. The real model slug lives in env
    (OPENROUTER_MODEL_A/B); only the named handle appears here, never the
    slug. The rows join the curated set so validate_model_id Path 1
    admits 'openrouter:slot-a|b' with NO validator surgery."""
    try:
        from . import _otr_openrouter_backend as _orb
    except Exception:  # noqa: BLE001 -- a backend import hiccup must never break the catalog
        return ()
    if not _orb.openrouter_enabled():
        return ()
    common = dict(
        requires_auth=False,
        loader_backend="openrouter_http",
        vram_fit_tier="PASS",
        approx_safetensors_gb=0.0,
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=_orb.DEFAULT_CONTEXT_WINDOW,
        license="gated_terms",
        license_audit_status="research_lane",
        provider="openrouter",
    )
    return (
        CuratedModel(
            repo_id=_orb.SLOT_A_ID,
            notes="OpenRouter remote model A (opt-in, default-off). Binds "
            "to OPENROUTER_MODEL_A; zero local VRAM. See "
            "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
            "v2.0-alpha/docs/openrouter-setup.md.",
            **common,
        ),
        CuratedModel(
            repo_id=_orb.SLOT_B_ID,
            notes="OpenRouter remote model B (opt-in, default-off). Binds "
            "to OPENROUTER_MODEL_B; zero local VRAM. See "
            "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
            "v2.0-alpha/docs/openrouter-setup.md.",
            **common,
        ),
    )


def _comfy_virtual_rows() -> tuple[CuratedModel, ...]:
    """The two virtual Comfy Credits rows -- present ONLY when the lane is
    enabled (OTR_ENABLE_COMFY_CREDITS=1). When disabled the tuple is empty,
    so _by_repo_id / dropdowns / Path-1 validation never see them and the
    offline baseline is untouched (mirrors the OpenRouter gate). These carry
    loader_backend='comfy_credits_http', provider='comfy_credits',
    approx_safetensors_gb=0.0; the real catalog slug resolves behind the
    scenes (the comfy slot pickers / recommended default). The rows join the
    curated set so validate_model_id Path 1 admits 'comfy:slot-a|b' with NO
    validator surgery."""
    try:
        from . import _otr_comfy_backend as _occ
    except Exception:  # noqa: BLE001 -- a backend import hiccup must never break the catalog
        return ()
    if not _occ.comfy_credits_enabled():
        return ()
    common = dict(
        requires_auth=False,
        loader_backend="comfy_credits_http",
        vram_fit_tier="PASS",
        approx_safetensors_gb=0.0,
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=_occ.DEFAULT_CONTEXT_WINDOW,
        license="gated_terms",
        license_audit_status="research_lane",
        provider="comfy_credits",
    )
    return (
        CuratedModel(
            repo_id=_occ.SLOT_A_ID,
            notes="Comfy Credits remote model A (opt-in, default-off). "
            "Credit-billed via ComfyUI's partner-node proxy; zero local "
            "VRAM. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/"
            "blob/v2.0-alpha/docs/comfy-credits-setup.md.",
            **common,
        ),
        CuratedModel(
            repo_id=_occ.SLOT_B_ID,
            notes="Comfy Credits remote model B (opt-in, default-off). "
            "Credit-billed via ComfyUI's partner-node proxy; zero local "
            "VRAM. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/"
            "blob/v2.0-alpha/docs/comfy-credits-setup.md.",
            **common,
        ),
    )


def _google_api_virtual_rows() -> tuple[CuratedModel, ...]:
    """The two virtual Google API rows -- present only when a Gemini API key
    is configured. They target the Gemini API / Interactions surface. API keys
    are read from environment at call time and never at import time."""
    try:
        from ._otr_google_api import models as _gai
    except Exception:  # noqa: BLE001 -- catalog import must stay robust
        return ()
    if not _gai.google_api_enabled():
        return ()
    common = dict(
        requires_auth=False,
        loader_backend=_gai.GOOGLE_API_BACKEND_KEY,
        vram_fit_tier="PASS",
        approx_safetensors_gb=0.0,
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=_gai.DEFAULT_CONTEXT_WINDOW,
        license="gated_terms",
        license_audit_status="research_lane",
        provider=_gai.GOOGLE_API_PROVIDER,
    )
    return (
        CuratedModel(
            repo_id=_gai.GOOGLE_API_SLOT_A_ID,
            notes="Google Gemini API remote model A (own-key, zero local VRAM). "
            "Concrete model comes from google_api_slot_a_model.",
            **common,
        ),
        CuratedModel(
            repo_id=_gai.GOOGLE_API_SLOT_B_ID,
            notes="Google Gemini API remote model B (own-key, zero local VRAM). "
            "Concrete model comes from google_api_slot_b_model.",
            **common,
        ),
    )


def _gguf_native_virtual_rows() -> tuple[CuratedModel, ...]:
    """Project every ``_otr_gguf_backend.GGUF_ROWS`` registry row into a
    visible catalog peer.

    The visible handle is the actual GGUF repository id so the dropdown reads
    like a peer to the other Gemma rows. The loader resolves the local file
    from C:\\ComfyUI-Models by default.

    Guards ONLY the optional backend IMPORT (llama-cpp is an optional dep). A
    registry-VALIDATION error (a malformed GGUF_ROWS row) PROPAGATES so a bad
    row fails startup/tests loudly instead of silently deleting the lane.
    ``approx_safetensors_gb`` is DERIVED from the row's pinned bytes; an
    unpinned row projects 0.0 (= UNKNOWN, never a guessed estimate).
    """
    try:
        from . import _otr_gguf_backend as _gguf
    except ImportError:  # optional-dep safe -- backend module unavailable
        return ()
    rows: list[CuratedModel] = []
    for row in _gguf.GGUF_ROWS:
        if row.repo_id == _gguf.ROW_ID:
            notes = (
                "Gemma 4 12B Q8_0 GGUF via in-process llama-cpp-python. "
                "Default file: C:\\ComfyUI-Models\\LLM\\converted\\"
                "gemma-4-12b-it\\gemma-4-12b-it-Q8_0.gguf. No Ollama, no "
                "sidecar, no port."
            )
        else:
            notes = (
                f"{row.repo_id} GGUF via in-process llama-cpp-python "
                f"(local subdir {row.subdir}). No Ollama, no sidecar, no port."
            )
        rows.append(CuratedModel(
            repo_id=row.repo_id,
            requires_auth=row.requires_auth,
            loader_backend=_gguf.GGUF_BACKEND_KEY,
            vram_fit_tier=row.vram_fit_tier,
            approx_safetensors_gb=row.approx_artifact_gb(),
            notes=notes,
            prompt_profile="modern",
            chat_template_kind="transformers_default",
            stop_tokens=row.stop_tokens,
            context_window=row.context_window,
            license=row.license,
            license_audit_status=row.license_audit_status,
            provider="gguf_native",
        ))
    return tuple(rows)


def _curated_with_gguf_native_peer() -> tuple[CuratedModel, ...]:
    """Static curated rows plus the always-visible Gemma 4 12B GGUF peer.

    Keep the 12B GGUF row beside the native Gemma 4 rows in dropdown order
    instead of appending it after unrelated remote slots.
    """
    gguf_rows = _gguf_native_virtual_rows()
    if not gguf_rows:
        return CURATED_LLM_MODELS
    out: list[CuratedModel] = []
    inserted = False
    for row in CURATED_LLM_MODELS:
        out.append(row)
        if row.repo_id == "google/gemma-4-12b-it":
            out.extend(gguf_rows)
            inserted = True
    if not inserted:
        out.extend(gguf_rows)
    return tuple(out)


def _active_curated_models() -> tuple[CuratedModel, ...]:
    """CURATED_LLM_MODELS plus enabled-only HTTP virtual rows.

    Consumers that should surface HTTP lanes when enabled (the dropdown
    builder + validate_model_id Path 1 via _by_repo_id) read THIS.
    Static license/audit tests iterate CURATED_LLM_MODELS directly, so
    the virtual rows never reach them, and GATED_CURATED_MODELS stays
    keyed off the real gated set."""
    return (
        _curated_with_gguf_native_peer()
        + _openrouter_virtual_rows()
        + _comfy_virtual_rows()
        + _google_api_virtual_rows()
    )


def _by_repo_id() -> dict[str, CuratedModel]:
    return {m.repo_id: m for m in _active_curated_models()}


def text_only_load_mode(model_id: str) -> str:
    """``"native_text_decoder"`` or ``"composite"`` for a selected model id.

    Accepts a badged dropdown label as well as a bare repo id -- the picker
    shows ``'google/gemma-4-E2B-it (3.0 GB)'`` and a saved graph stores that
    string, so an exact-match lookup on the raw widget value would quietly
    miss and fall back to the composite path.

    An id with NO curated row returns ``"composite"``: an uncurated local
    cache hit keeps loading exactly as it does today. This is the loader's
    single source of truth for the question, so the decision lives with the
    row's other honesty fields rather than in an id ladder inside the loader.
    """
    row = _by_repo_id().get(_strip_label_suffix(model_id))
    return getattr(row, "text_only_load", "composite") or "composite"


GATED_CURATED_MODELS: frozenset[str] = frozenset(
    m.repo_id for m in CURATED_LLM_MODELS if m.requires_auth
)


# ---------------------------------------------------------------------------
# Local HF cache scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResult:
    """One entry from a local HF cache walk."""

    repo_id: str
    on_disk: bool
    snapshot_path: str | None
    advertised_context: int | None  # max_position_embeddings (or fallback)


def _hf_hub_root() -> Path | None:
    """Resolve the HuggingFace hub cache root the SAME way huggingface_hub
    (and therefore the transformers model loader) does, so the dropdown's
    "downloaded" state can never disagree with where weights actually
    resolve at load time.

    Precedence mirrors huggingface_hub.constants (HF_HUB_CACHE wins, then
    the legacy alias, then HF_HOME/hub, then the library default):
        1. HF_HUB_CACHE env            -- modern canonical override (full path)
        2. HUGGINGFACE_HUB_CACHE env   -- legacy alias (full path)
        3. HF_HOME env + "/hub"        -- OTR's shared-root convention
        4. ~/.cache/huggingface/hub    -- library default

    Read live from os.environ (no import-time cached constant) so the
    resolver reflects the process env at call time and stays monkeypatch-
    testable. HF_HUB_CACHE was the missing branch: the box sets it (the
    var huggingface_hub honors) but the old resolver read only HF_HOME +
    the legacy HUGGINGFACE_HUB_CACHE, so it silently fell through to the
    stale ~/.cache default and mislabeled on-disk models NOT DOWNLOADED.
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        val = otr_env.get(var)
        if val:
            root = Path(val)
            if root.is_dir():
                return root
    hf_home = otr_env.get("HF_HOME")
    if hf_home:
        root = Path(hf_home) / "hub"
        if root.is_dir():
            return root
    default = Path.home() / ".cache" / "huggingface" / "hub"
    if default.is_dir():
        return default
    return None


def _parse_repo_dir_name(name: str) -> str | None:
    """Convert "models--mistralai--Mistral-Nemo-Instruct-2407" to
    "mistralai/Mistral-Nemo-Instruct-2407". Returns None on non-match."""
    if not name.startswith("models--"):
        return None
    rest = name[len("models--") :]
    if "--" not in rest:
        return None
    org, _, repo = rest.partition("--")
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def read_native_context(config: Any) -> int | None:
    """Read the decoder's actual capacity without changing its configuration."""
    def field(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    nested = field(config, "text_config")
    for owner in (nested, config):
        if owner is None:
            continue
        for key in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = field(owner, key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _read_advertised_context(snapshot_path: Path) -> int | None:
    """Read native decoder capacity from this exact snapshot, if available."""
    try:
        import json
        with (snapshot_path / "config.json").open("r", encoding="utf-8") as handle:
            return read_native_context(json.load(handle))
    except (OSError, ValueError, TypeError):
        return None


def _snapshot_is_causal_lm(snapshot_path: str | None) -> bool:
    """True only if the snapshot's config.json declares a decoder-only
    causal-LM architecture (an `architectures` entry ending in
    `ForCausalLM`).

    The HF hub cache is shared by every model type OTR pulls -- writer
    LLMs, FLUX, LTX-Video, Depth-Anything all resolve into the same
    HF_HOME/hub. A bare directory walk cannot tell a story-writer LLM
    from a diffusion or vision checkpoint, so the non-curated dropdown
    discovery path uses this gate to admit only text-generation models.

    Diffusion pipelines (FLUX, LTX-Video) ship a model_index.json and
    carry no root config.json, so they fail the `is_file` check.
    Vision / depth transformers models carry a root config.json whose
    `architectures` is not `*ForCausalLM`. Both are excluded. Returns
    False on a missing path or any read failure -- fail closed, since
    the curated set is added to the dropdown unconditionally and never
    depends on this gate.
    """
    if not snapshot_path:
        return False
    cfg = Path(snapshot_path) / "config.json"
    if not cfg.is_file():
        return False
    try:
        import json

        with cfg.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    archs = data.get("architectures")
    if not isinstance(archs, list):
        return False
    return any(isinstance(a, str) and a.endswith("ForCausalLM") for a in archs)


def _gguf_native_row_on_disk(repo_id: str) -> bool:
    """Side-effect-free GGUF presence probe for dropdown metadata: True iff
    ANY registered artifact for ``repo_id`` resolves to a regular non-zero
    file. The dropdown has no quant context, so any materialized quant counts."""
    try:
        from . import _otr_gguf_backend as _gguf
    except ImportError:  # keep INPUT_TYPES import-safe (optional dep)
        return False
    try:
        return _gguf.gguf_native_row_on_disk(repo_id)
    except Exception:  # noqa: BLE001 -- keep INPUT_TYPES import-safe
        return False


# Weight-file suffixes that mark a materialized (loadable) transformers
# snapshot. A config-only snapshot -- config.json present but no weight
# blob -- is on the HF layout but NOT usable: the loader would re-download
# or fail. Detection must distinguish "config present" from "fully
# materialized + usable" (HF snapshots are symlink farms into ../../blobs;
# a metadata-only pull lands config.json with no weight symlink at all).
_WEIGHT_SUFFIXES = (".safetensors", ".bin")

# A SHARDED repo publishes one of these alongside its shards; its ``weight_map``
# names every shard the load needs. See _snapshot_has_weights.
_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _safe_snapshot_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _shards_named_by_index(snapshot_path: Path) -> set[str] | None:
    """Shard filenames a sharded repo's index declares, or None if unsharded.

    Returns an EMPTY set for an index that exists but cannot be read or names
    nothing -- the caller treats that as incomplete (fail closed), because a
    corrupt index is itself evidence of a half-finished pull.
    """
    for index_name in _WEIGHT_INDEX_NAMES:
        index_path = snapshot_path / index_name
        try:
            if not index_path.is_file():
                continue
            import json as _json
            weight_map = _json.loads(
                index_path.read_text(encoding="utf-8")
            ).get("weight_map")
        except (OSError, ValueError, AttributeError):
            return set()  # present but unreadable -> incomplete, fail closed
        if not isinstance(weight_map, dict) or not weight_map:
            return set()
        return {
            str(name) for name in weight_map.values() if isinstance(name, str)
        }
    return None


def _snapshot_has_weights(snapshot_path: Path) -> bool:
    """True iff `snapshot_path` holds a COMPLETE materialized weight set.

    Follows HF symlinks: a snapshot weight entry is a symlink into
    ``../../blobs/<sha>``; ``stat().st_size`` resolves through the link to
    the real blob size, so a present-but-unmaterialized (broken/absent)
    link never counts. Returns False on any read error -- fail closed.

    SHARD COMPLETENESS IS PART OF "HAS WEIGHTS" (2026-09-06). The older rule
    returned True on the FIRST nonzero weight file it found, which is wrong for
    every multi-shard repo -- and the rows an 8 GB card must consider are
    multi-shard (Qwen/Qwen3.5-4B is 9,319,828,096 bytes over two shards, well
    past HF's 5 GB default shard size). A first download interrupted after
    shard 1 lands -- an operator cancel, a dropped connection, a ComfyUI
    restart, all of which happened during this campaign -- left one shard
    materialized. That made ``on_disk`` True, which makes
    ``auto_download_if_missing`` short-circuit the download, and ``load_llm``
    has no network fallback: the repo would then fail to load on every
    subsequent attempt with no way back except manually clearing the cache.
    A partially downloaded model is NOT on disk, and saying so cost nothing
    but a re-download.
    """
    declared = _shards_named_by_index(snapshot_path)
    if declared is not None:
        if not declared:
            return False  # index present but unusable -> incomplete
        for shard_name in declared:
            shard = snapshot_path / shard_name
            try:
                if shard.stat().st_size <= 0:  # follows symlink to the blob
                    return False
            except OSError:
                return False  # missing / broken symlink -> shard not present
        return True
    try:
        for child in snapshot_path.iterdir():
            if child.suffix.lower() not in _WEIGHT_SUFFIXES:
                continue
            try:
                if child.stat().st_size > 0:  # follows symlink to the blob
                    return True
            except OSError:
                continue  # broken symlink / unresolved blob -> not materialized
    except OSError:
        return False
    return False


def scan_local_llm_cache(hub_root: Path | None = None) -> list[ScanResult]:
    """Walk HF_HOME/hub/models--*/snapshots/* and return one ScanResult
    per resolved snapshot. Offline-only -- no HF API calls.

    `hub_root` override lets tests point at a fixture directory.
    """
    root = hub_root if hub_root is not None else _hf_hub_root()
    if root is None or not root.is_dir():
        return []
    out: list[ScanResult] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        repo_id = _parse_repo_dir_name(child.name)
        if repo_id is None:
            continue
        snapshots_dir = child / "snapshots"
        if not snapshots_dir.is_dir():
            out.append(ScanResult(repo_id, False, None, None))
            continue
        snapshot_paths = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if not snapshot_paths:
            out.append(ScanResult(repo_id, False, None, None))
            continue
        # Prefer the newest snapshot that actually carries weight blobs, so
        # a config-only (metadata) pull is reported NOT usable even though
        # its snapshot dir exists. Fall back to the newest snapshot overall
        # (on_disk stays False) so callers still get a snapshot_path for
        # diagnostics. HF stores multiple snapshots per repo by commit hash.
        weighted = [p for p in snapshot_paths if _snapshot_has_weights(p)]
        if weighted:
            snapshot = max(weighted, key=_safe_snapshot_mtime)
            on_disk = True
        else:
            snapshot = max(snapshot_paths, key=_safe_snapshot_mtime)
            on_disk = False
        ctx = _read_advertised_context(snapshot)
        out.append(ScanResult(repo_id, on_disk, str(snapshot), ctx))
    return out


# ---------------------------------------------------------------------------
# Dropdown builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DropdownEntry:
    label: str  # what the user sees in the ComfyUI dropdown
    repo_id: str  # canonical id (no suffix)
    on_disk: bool
    curated: bool


def build_dropdown_choices(
    hub_root: Path | None = None,
) -> list[DropdownEntry]:
    """List the curated writer models + enabled remote handles + any
    locally-discovered causal-LM repos.

    Labels are the bare repo id / remote handle -- NO download-state badge.
    The "downloaded" state of an HF-cache model depends on each machine's
    HF cache layout (HF_HOME / HF_HUB_CACHE vary per user and even per
    launch, and can be split across roots), so a "[NOT DOWNLOADED]" /
    "[LOCAL HF]" label can never be reliably correct for every user -- a
    wrong badge is worse than none. A model that is not already cached is
    simply fetched by auto_download_if_missing on first Queue; selection is
    never gated on the badge. `on_disk` is still tracked (it drives the
    recovery hint + the auto-download short-circuit), it just no longer
    decorates the visible label.
    """
    scan = {r.repo_id: r for r in scan_local_llm_cache(hub_root=hub_root)}
    entries: list[DropdownEntry] = []
    active = _active_curated_models()
    for m in active:
        provider = getattr(m, "provider", "local")
        if provider == "gguf_native":
            on_disk = _gguf_native_row_on_disk(m.repo_id)
        elif provider != "local":
            # Remote lane (OpenRouter / Comfy Credits / Google API): present
            # in the active set only when its lane is enabled, so selectable.
            on_disk = True
        else:
            on_disk = m.repo_id in scan and scan[m.repo_id].on_disk
        entries.append(DropdownEntry(
            m.repo_id + vram_badge_for(m.repo_id), m.repo_id,
            on_disk, curated=True))
    curated_ids = {m.repo_id for m in active}
    for repo_id, result in scan.items():
        if repo_id in curated_ids:
            continue
        if not result.on_disk:
            continue
        # The HF cache mixes every model type OTR downloads, so a non-curated
        # cache hit is not necessarily a text-generation LLM. Admit it only if
        # its config.json declares a `*ForCausalLM` architecture -- keeps
        # diffusion (FLUX, LTX-Video) and vision (Depth-Anything) checkpoints
        # out of the writer picker. Curated rows above are exempt (the
        # explicit writer set). See BUG-LOCAL-257.
        if not _snapshot_is_causal_lm(result.snapshot_path):
            continue
        entries.append(DropdownEntry(repo_id, repo_id, True, curated=False))
    return entries


def dropdown_choices(hub_root: Path | None = None) -> list[str]:
    """The bare label list ComfyUI INPUT_TYPES wants for a COMBO widget."""
    return [e.label for e in build_dropdown_choices(hub_root=hub_root)]


def default_llm_option() -> str:
    """The exact COMBO label for :data:`DEFAULT_LLM`, size suffix included.

    THE SUFFIX IS PART OF THE VALUE. A saved graph must carry the label the
    dropdown offers -- a bare repo id matches no choice, and an unmatched COMBO
    can resolve to index 0, so a graph that SAYS one model silently runs
    another. That failure is already recorded (2026-08-04, both writer widgets
    rendering red) and is why callers must never hand-build "repo_id (N GB)".

    WHY THIS EXISTS (PBUG-20260906-09). Four test modules hard-coded
    ``"google/gemma-4-12b-it (11.9 GB)"``. When DEFAULT_LLM moved to Qwen those
    literals did not, so the tests asserted the OLD default was shipped -- they
    pinned the drift in place instead of catching it, and a test that fails
    when the default legitimately changes is a test that will be edited rather
    than believed. Deriving the label here means one edit to DEFAULT_LLM moves
    the constant, the shipped graphs' checker, and every test at once.

    Cache-independent by construction: it composes the same way
    :func:`build_dropdown_choices` composes a curated row
    (``repo_id + vram_badge_for(repo_id)``) and never scans the HF cache, so it
    answers identically on a cold box and a warm one.
    """
    return DEFAULT_LLM + vram_badge_for(DEFAULT_LLM)


# ---------------------------------------------------------------------------
# OpenRouter slot-slug picker dropdowns (S1)
# ---------------------------------------------------------------------------
#
# The 2026-06-01 four-dropdown router: creative_writing_model and
# technical_model stay LOCAL + slot-a/b selectors (build_dropdown_choices
# above -- the OpenRouter catalog NEVER appears there). The two NEW pickers
# openrouter_slot_a_model / openrouter_slot_b_model choose the real OpenRouter
# slug from the S0 disk cache (nodes/_otr_openrouter_backend.cached_models()).
# INPUT_TYPES-safe: every tier reads the on-disk cache only, never the network.

OPENROUTER_ENABLE_SENTINEL = "(enable OpenRouter)"
"""Sole choice in a slot picker when remote is disabled. UI-only -- S3
rejects it before backend resolution; it can never resolve as a slug."""

OPENROUTER_EMPTY_CACHE_SENTINEL = "(no OpenRouter models cached -- run refresh_catalog_cache)"
"""Shown when remote is enabled but the catalog cache is missing/empty. The
recommended default is offered alongside so the slot still has a valid pick."""

# 2026-08-07: the "recent" tier is GONE. It contributed 8 of the 21 slot-a
# choices from whatever the disk cache happened to hold, which is the opposite
# of curation -- an uncurated, silently-changing block. Discovery now lives in
# OTR_OPENROUTER_FAVORITES, the allowlist/provider filters, and the explicit
# OTR_OPENROUTER_FULL_CATALOG=1 opt-in, all of which already existed.


def _lead_with_sentinel(sentinel: str, choices: list[str]) -> list[str]:
    """Return ``choices`` with ``sentinel`` guaranteed as the FIRST entry,
    de-duplicated. The enable-sentinel is each slot's 'off / use-local' value
    AND its INPUT_TYPES default; it MUST remain a valid choice in EVERY lane
    state. Otherwise a saved workflow that stores the sentinel fails ComfyUI's
    COMBO validation the instant the lane is enabled and the catalog replaces it
    (BUG-LOCAL-400). Leading with it also keeps 'off' the default for a fresh
    node, so enabling a lane never silently defaults a slot to a billable model.
    """
    out = [sentinel]
    for c in choices:
        if c != sentinel:
            out.append(c)
    return out


def _csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into a stripped, non-empty list.
    Unset / empty -> []."""
    raw = otr_env.get(name)
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _slot_requires_json(slot: str) -> bool:
    """Per-slot structured-output filter. NEVER global: defaults off, so a
    creative model is never hidden from slot A just because slot B needs
    JSON. Reads OTR_OPENROUTER_SLOT_<A|B>_REQUIRE_JSON."""
    var = f"OTR_OPENROUTER_SLOT_{slot.strip().upper()}_REQUIRE_JSON"
    return otr_env.get(var, "0") == "1"


def _is_text_writer_model(m: dict) -> bool:
    """True if a catalog row can serve a WRITER slot: it outputs text and is not
    an image/audio/video generator. The A/B slots drive the LLM writer, so an
    image model like ``google/gemini-3-pro-image`` (which emits a picture or a
    planning monologue, never a usable line) must never appear there.

    TOLERANT of an old cache: a row with no ``output_modalities`` (cached before
    the modality field was captured) is KEPT, so an un-refreshed catalog still
    shows every model rather than going empty. A row is hidden ONLY when we
    positively know its output is non-text (text absent, or image present)."""
    out = m.get("output_modalities")
    if not isinstance(out, list) or not out:
        return True  # unknown -> don't hide (re-run refresh_catalog_cache to populate)
    mods = [str(x).lower() for x in out]
    return ("text" in mods) and ("image" not in mods)


def _filter_catalog_models(models: list[dict], *, slot: str) -> list[dict]:
    """Apply the slot-A/B catalog filters (filters, never a cage). Each is
    independent and unset == no-op:
      * text-output only               -- writer slots hide image/audio/video
                                          generators (OTR_OPENROUTER_ALLOW_NONTEXT=1
                                          to disable); tolerant of an old cache
      * OTR_OPENROUTER_PROVIDER_FILTER -- provider-prefix allowlist (id before '/')
      * OTR_OPENROUTER_MODEL_ALLOWLIST -- exact-id allowlist
      * OTR_OPENROUTER_MODEL_DENYLIST  -- exact-id removal
      * per-slot REQUIRE_JSON          -- keep only supports_json models
    """
    providers = set(_csv_env("OTR_OPENROUTER_PROVIDER_FILTER"))
    allow = set(_csv_env("OTR_OPENROUTER_MODEL_ALLOWLIST"))
    deny = set(_csv_env("OTR_OPENROUTER_MODEL_DENYLIST"))
    require_json = _slot_requires_json(slot)
    text_only = otr_env.get("OTR_OPENROUTER_ALLOW_NONTEXT", "0") != "1"
    out: list[dict] = []
    for m in models:
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        if text_only and not _is_text_writer_model(m):
            continue
        provider = m.get("provider") or (mid.split("/", 1)[0] if "/" in mid else "")
        if providers and provider not in providers:
            continue
        if allow and mid not in allow:
            continue
        if mid in deny:
            continue
        if require_json and not m.get("supports_json"):
            continue
        out.append(m)
    return out


# The CURATED alias set (2026-08-07 curation; supersedes the 2026-06-20 block).
#
# POLICY, in priority order:
#   (a) PREFER `~author/family-latest` routing aliases. They resolve upstream at
#       request time, so a new model in that family is picked up with no edit
#       here and an operator never sees a slug that has gone stale.
#   (b) Carry a CONCRETE id only where a specific version genuinely matters, and
#       give it a date in OPENROUTER_VERIFIED_ON_BY_ID below.
#   (c) NEVER carry a `:free` or promo-priced slug. A `:free` id is a PRICE
#       PROMISE baked into an IDENTIFIER, and price promises expire while
#       identifiers do not -- `tencent/hy3:free` was carried here until its promo
#       ended and the slug stopped resolving. test_openrouter_slug_curation.py
#       enforces this; a comment cannot.
#   (d) Auto-routers ARE now offered, as SELECTABLE ENTRIES ONLY -- never as a
#       default (operator, 2026-08-10). This reverses the previous blanket
#       exclusion, which read "any of them picks a model by criteria we do not
#       control, so one config resolves differently week to week". That is still
#       TRUE, and it is still why no router may be a default; it is not a reason
#       to withhold the choice. See OPENROUTER_CURATED_ROUTERS below for which
#       two, and why only two.
#
#       The budget objection was RETIRED as inconsistent, by the operator:
#       "you can't budget against a -latest either." He is half right, and the
#       measured half matters. A `~latest` alias carries a REAL published price
#       (`~anthropic/claude-opus-latest` = $5/$25 per M on 2026-08-10), so it is
#       discoverable at any moment and merely moves when the vendor ships. A
#       router is priced `-1` -- not published, not discoverable, ever. Both
#       change; only one can be looked up. That is a narrower difference than
#       "unbudgetable" claimed, and it is not zero.
#
# Verified against live /api/v1/models on 2026-08-09: all eleven are listed.
# They are offered even when the disk cache is cold, because a cold cache must
# not hide the curated set -- see openrouter_catalog_dropdown_choices for the
# two states where the block is skipped.
OPENROUTER_CURATED_ALIASES = (
    "~anthropic/claude-opus-latest",
    "~openai/gpt-latest",
    "~google/gemini-pro-latest",
    "~anthropic/claude-sonnet-latest",
    "~anthropic/claude-haiku-latest",
    "~anthropic/claude-fable-latest",
    "~openai/gpt-mini-latest",
    "~google/gemini-flash-latest",
    "~moonshotai/kimi-latest",
    # 2026-08-07: x-ai now publishes a `~latest` resolver, which retired ~30
    # lines of bespoke "pick the author's newest concrete slug" synthesis.
    "~x-ai/grok-latest",
    # 2026-08-09 (chunk B): THE CHEAP SLOT, and it is an alias on purpose.
    # ~$0.08/$0.25 per M -- the cheapest option that is a POINTER rather than a
    # pin. The two cheaper candidates were rejected for the same reason:
    # `qwen/qwen3.7-flash` ($0.03/$0.13) and `inclusionai/ling-2.6-flash`
    # ($0.01/$0.03) are both CONCRETE ids with no `~latest` resolver published
    # by their authors, so shipping either re-creates the hy3 defect at a
    # rounding-error saving. `inclusionai` is additionally a five-model author
    # in the whole catalog -- its cheapest SKU is the likeliest on the board to
    # be retired. Paying 2.6x of almost nothing buys out an entire failure class.
    "~deepseek/deepseek-v4-flash-latest",
)

#: Every CONCRETE (non-alias) OpenRouter id this pack ships, mapped to the date
#: it was last verified against live /api/v1/models. An alias needs no date --
#: it resolves upstream -- but a concrete id is a claim about a specific version
#: that can quietly stop being true. The guard test asserts these keys are
#: EXACTLY the concrete ids shipped, so a new pin cannot be added undated.
#:
#: Down to ONE entry as of chunk B (2026-08-09): the creative default became
#: `~anthropic/claude-opus-latest`, and an alias needs no date because there is
#: no version claim left to go stale. The guard test computes this set from the
#: '~' prefix, so the removal below is not optional bookkeeping -- leaving the
#: old dated pin here would fail `test_every_concrete_id_is_dated`.
#: The auto-routers offered in the A/B slug dropdowns (operator, 2026-08-10:
#: "pick the 1-2 best autos for my workflow and add them").
#:
#: TWO, BECAUSE ONLY TWO ARE ELIGIBLE -- this was measured against live
#: /api/v1/models on 2026-08-10, not chosen by taste. The writer's passes are
#: SCHEMA-CONSTRAINED, so a model that cannot be told to return JSON cannot serve
#: a writer slot:
#:
#:     openrouter/auto          response_format  YES   ctx 2,000,000
#:     openrouter/auto-beta     response_format  YES   ctx 2,000,000
#:     openrouter/bodybuilder   supported_parameters EMPTY   ctx 128,000
#:     openrouter/fusion        supported_parameters EMPTY   ctx 1,000,000
#:     openrouter/pareto-code   supported_parameters EMPTY   ctx 2,000,000
#:
#: The last three declare NO parameters at all, so REQUIRE_JSON would drop them
#: from a writer slot anyway. Listing them would put three dead entries in a
#: dropdown -- the exact shape this pack's slug guards exist to prevent.
#:
#: WHY THE CHECK HAD TO BE MANUAL. The curated block deliberately BYPASSES
#: `_filter_catalog_models` in the default warm view (see
#: `openrouter_catalog_dropdown_choices`), so REQUIRE_JSON does not narrow it.
#: Curated entries are policy, not discovery -- which means the JSON-capability
#: judgement is made HERE, once, by a person, instead of being enforced later.
#:
#: THEY ARE SELECTABLE, NEVER DEFAULT. Neither is a recommended default and
#: neither may become one: a router picks by criteria this pack does not control,
#: so a router default is a config that resolves differently week to week. It is
#: also the one case where `meta["resolved_models"]` earns its keep -- a router
#: run is fully auditable AFTER the fact even though it is not predictable
#: before it, so a trial tells you exactly which model wrote the episode.
#:
#: `response_format` on the ROUTER row is not a promise about the model it lands
#: on. It says the router accepts the parameter, not that today's route honours
#: it. Treat an unparseable writer pass on a router as expected variance, not a
#: new bug.
OPENROUTER_CURATED_ROUTERS: tuple[str, ...] = (
    "openrouter/auto",        # the stable one -- start here
    "openrouter/auto-beta",   # the experimental twin, same capabilities today
)

#: Every CONCRETE (non-alias) id, dated. The routers are concrete -- they carry
#: no `~` -- so they are dated like any other pin even though a router is in
#: practice the most evergreen thing on the board: it cannot go stale, it can
#: only change what it points at. The date records that both were confirmed
#: listed, text-capable and `response_format`-declaring on that day.
OPENROUTER_VERIFIED_ON_BY_ID: dict[str, str] = {
    # `deepseek/deepseek-v4-pro` was here as the recommended TECHNICAL default
    # until 2026-08-10, when both slots moved to the auto-router. The dating rule
    # is an EXACT match against shipped concrete ids, so a date for something no
    # longer shipped is itself a defect -- it would read as a live claim about a
    # slug this pack does not offer.
    "openrouter/auto": "2026-08-10",
    "openrouter/auto-beta": "2026-08-10",
}


def openrouter_catalog_dropdown_choices(slot: str) -> list[str]:
    """Slug-picker choices for openrouter_slot_<slot>_model (slot 'a' / 'b').

    Sentinel-led in EVERY state (BUG-LOCAL-400): OPENROUTER_ENABLE_SENTINEL is
    always choices[0] -- the 'off / use-local' default -- so a saved workflow
    that stores it validates whether or not the lane is enabled.
    Remote disabled -> [OPENROUTER_ENABLE_SENTINEL].
    Remote enabled  -> the sentinel, then an ordered, de-duplicated list:
        1. recommended default for the slot -- the per-slot
           OTR_OPENROUTER_SLOT_x_DEFAULT override when set AND present in the
           filtered cache, else the OPENROUTER_RECOMMENDED_*_DEFAULT constant.
           Always offered first so the slot's default value is selectable even
           if a cold cache or a filter would otherwise hide it.
        2. favorites -- OTR_OPENROUTER_FAVORITES, in operator order, cache-gated
        3. OPENROUTER_CURATED_ALIASES -- the `~family-latest` routing aliases
        4. the full filtered catalog, alphabetically, ONLY under explicit
           narrowing or OTR_OPENROUTER_FULL_CATALOG=1
    Enabled but empty/cold cache ->
    [OPENROUTER_ENABLE_SENTINEL, recommended_default, EMPTY_CACHE_SENTINEL].

    WHERE THE ALIASES ARE *NOT* OFFERED -- two states, both deliberate, and both
    easy to misdescribe (the pre-2026-08-07 comment here claimed they were
    offered "UNCONDITIONALLY", which the code has never done):
        * explicit narrowing (allowlist / provider filter) -- the operator asked
          for an exact set, so it is honoured verbatim;
        * the cold-cache branch -- it returns lead + empty-cache sentinel so the
          operator is pointed at a refresh rather than handed a long list.
    Consequence worth knowing: in the default warm view the aliases BYPASS
    _filter_catalog_models, so REQUIRE_JSON and the denylist do not narrow them.
    They are curated policy, not catalog discovery.

    INPUT_TYPES-safe: reads the disk cache only, never the network.
    """
    s = slot.strip().lower()
    if s not in ("a", "b"):
        raise ValueError(f"slot must be 'a' or 'b', got {slot!r}")
    try:
        from . import _otr_openrouter_backend as _orb
    except Exception:  # noqa: BLE001 -- a backend import hiccup must never break INPUT_TYPES
        return [OPENROUTER_ENABLE_SENTINEL]
    if not _orb.openrouter_enabled():
        return [OPENROUTER_ENABLE_SENTINEL]

    models = _filter_catalog_models(_orb.cached_models(), slot=s)
    by_id = {m["id"]: m for m in models}

    # Tier 1 lead: the per-slot env override iff present in the filtered cache,
    # else the recommended constant ("if set + present, else recommended").
    configured = (otr_env.get(f"OTR_OPENROUTER_SLOT_{s.upper()}_DEFAULT") or "").strip()
    constant = (
        _orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT if s == "a"
        else _orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT
    )
    lead = configured if (configured and configured in by_id) else constant

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(mid: str) -> None:
        if mid and mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    _add(lead)  # always offered -> default value stays selectable

    for fav in _csv_env("OTR_OPENROUTER_FAVORITES"):
        if fav in by_id:
            _add(fav)

    # The CURATED alias block (default view only). SKIPPED when the operator set
    # an EXPLICIT allowlist / provider-filter -- there they asked for an exact
    # narrowed set, so we honour it verbatim.
    allowlist = (otr_env.get("OTR_OPENROUTER_MODEL_ALLOWLIST") or "").strip()
    provider_filter = (otr_env.get("OTR_OPENROUTER_PROVIDER_FILTER") or "").strip()
    explicit_narrowing = bool(allowlist or provider_filter)

    if not explicit_narrowing:
        for mid in OPENROUTER_CURATED_ALIASES:
            _add(mid)
        # The auto-routers, LAST in the curated block so a router is never what
        # the eye lands on first. Same default-view-only rule as the aliases:
        # under explicit narrowing the operator asked for an exact set and gets
        # it verbatim. Offered but never led with -- `lead` above is always a
        # recommended default, and no router is one.
        for mid in OPENROUTER_CURATED_ROUTERS:
            _add(mid)

    # The FULL alphabetical catalog (300+ slugs) is a long scroll that dates
    # fast; in the default view it is OPT-IN (OTR_OPENROUTER_FULL_CATALOG=1) so
    # the dropdown stays short. When the operator is explicitly narrowing
    # (allowlist / provider-filter), show the full filtered set as before.
    if explicit_narrowing or otr_env.get(
            "OTR_OPENROUTER_FULL_CATALOG", "0").strip() == "1":
        for m in sorted(models, key=lambda m: m["id"]):
            _add(m["id"])

    if not models:
        # Cold / fully-filtered cache: keep the recommended default selectable
        # and flag that discovery is empty so the operator runs a refresh.
        return _lead_with_sentinel(
            OPENROUTER_ENABLE_SENTINEL,
            [lead, OPENROUTER_EMPTY_CACHE_SENTINEL],
        )
    return _lead_with_sentinel(OPENROUTER_ENABLE_SENTINEL, ordered)


# ---------------------------------------------------------------------------
# Comfy Credits slot-slug picker dropdowns (2026-06-01, four-dropdown router)
# ---------------------------------------------------------------------------
#
# comfy_slot_a_model / comfy_slot_b_model choose the real credit-billed slug
# from the PINNED partner-node catalog (nodes/_otr_comfy_backend.COMFY_LLM_MODELS)
# -- no disk cache / refresh script (the catalog is a constant, not fetched).
# INPUT_TYPES-safe: reads the constant only, never the network.

COMFY_ENABLE_SENTINEL = "(enable Comfy Credits)"
"""Sole choice in a Comfy slot picker when the lane is disabled. UI-only --
rejected before backend resolution; it can never resolve as a slug."""


def comfy_catalog_dropdown_choices(slot: str) -> list[str]:
    """Slug-picker choices for comfy_slot_<slot>_model (slot 'a' / 'b').

    Sentinel-led in EVERY state (BUG-LOCAL-400): COMFY_ENABLE_SENTINEL is always
    choices[0] -- the 'off / use-local' default -- so a saved workflow that
    stores it validates whether or not the lane is enabled.
    Lane disabled -> [COMFY_ENABLE_SENTINEL].
    Lane enabled  -> the sentinel, then the recommended default for the slot,
    then OTR_COMFY_FAVORITES (operator order), then the full pinned catalog
    alphabetically. Deduped.
    INPUT_TYPES-safe: reads the pinned constant only, never the network.
    """
    s = slot.strip().lower()
    if s not in ("a", "b"):
        raise ValueError(f"slot must be 'a' or 'b', got {slot!r}")
    try:
        from . import _otr_comfy_backend as _occ
    except Exception:  # noqa: BLE001 -- a backend import hiccup must never break INPUT_TYPES
        return [COMFY_ENABLE_SENTINEL]
    if not _occ.comfy_credits_enabled():
        return [COMFY_ENABLE_SENTINEL]

    catalog = list(_occ.COMFY_LLM_MODELS)
    lead = _occ.recommended_slug_for_slot(s)
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(mid: str) -> None:
        if mid and mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    _add(lead)  # always offered -> default value stays selectable
    for fav in _csv_env("OTR_COMFY_FAVORITES"):
        if fav in catalog:
            _add(fav)
    for mid in sorted(catalog):
        _add(mid)
    return _lead_with_sentinel(COMFY_ENABLE_SENTINEL, ordered)


# ---------------------------------------------------------------------------
# Google API slot model picker dropdowns (2026-07-08, direct BYO lane)
# ---------------------------------------------------------------------------


def google_api_catalog_dropdown_choices(slot: str) -> list[str]:
    """Concrete Gemini model choices for google_api_slot_<slot>_model.

    Network-free at INPUT_TYPES time: the list is the sentinel, then static
    official text seeds when a key is present, plus any valid on-disk cache
    entries. The actual key is read only by the backend at call time.
    """
    try:
        from ._otr_google_api import models as _gai
    except Exception:  # noqa: BLE001 -- INPUT_TYPES must stay robust
        return ["(select Google API model)"]
    return _gai.google_api_model_choices(slot)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*$")
"""Conservative HF repo-id regex: <org>/<name>, alnum + . _ - only,
no leading punctuation. Does not allow slashes inside name."""


def _structural_reject(model_id: str) -> str | None:
    """Return a reject-reason string if the id is structurally unsafe,
    else None. Caught BEFORE any admit-path matching."""
    if not model_id or not isinstance(model_id, str):
        return "empty or non-string model_id"
    if "\\" in model_id:
        return "model_id contains backslash (path-like)"
    if model_id.startswith("/"):
        return "model_id starts with '/' (absolute path)"
    if ".." in model_id:
        return "model_id contains '..' (path traversal)"
    if re.match(r"^[A-Za-z]:", model_id):
        return "model_id starts with a drive letter (Windows absolute path)"
    if model_id.endswith(".gguf") or model_id.endswith(".bin"):
        return (
            "model_id ends in unsafe weight format (.gguf/.bin); "
            "OTR ships transformers loader only in S30"
        )
    return None


def _strip_label_suffix(model_id: str) -> str:
    """Strip UI-only dropdown suffixes from a selected model id.

    Also removes a trailing PARENTHETICAL badge -- the VRAM figure the picker
    shows so an operator can tell at a glance whether a model fits the card in
    front of them (``'google/gemma-4-12b-it (11.9 GB)'`` -> the bare repo id).
    Safe because a Hugging Face repo id cannot contain ``' ('``; this mirrors
    ``public_engines.resolve_engine_id``, which does the same for the video
    picker's ``'wan_8gb (16:9)'``.

    IDEMPOTENT AND BACKWARD-COMPATIBLE, which is the whole point: a bare id
    from a saved graph, a profile, a CLI flag or an older workflow passes
    through untouched, so adding the badge cannot break a stored selection."""
    s = model_id.strip()
    for suffix in (
        NOT_DOWNLOADED_SUFFIX,
        LOCAL_HF_SUFFIX,
        LOCAL_GGUF_SUFFIX,
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].rstrip()
    for suffix in (LOCAL_HF_SUFFIX, LOCAL_GGUF_SUFFIX):
        if s.endswith(suffix):
            s = s[: -len(suffix)].rstrip()
    if s.endswith(")") and " (" in s:
        s = s.rsplit(" (", 1)[0].rstrip()
    return s


#: Resident memory as a multiple of the bf16 download size on APPLE SILICON,
#: where nothing is quantized. MEASURED: PBUG-20260907-06 recorded
#: `Qwen/Qwen3.5-4B` (8.68 GB of safetensors) reaching a 14 GB phys_footprint
#: while generating -- 14 / 8.68 = 1.61. One data point, not a curve, and it is
#: deliberately NOT the /2.0 that `_estimate_resident_gb` applies: that divisor
#: assumes 8-bit/NF4, and `bitsandbytes>=0.42.0; sys_platform != 'darwin'` in
#: requirements.txt means bitsandbytes is never installed on macOS, so
#: `llm_quant_policy` can only be `none` there and nothing is halved.
_METAL_BF16_RESIDENT_FACTOR = 14.0 / 8.68

#: The whole machine. Between the comfortable budget and this figure a row is
#: MARGINAL -- proven to run with nothing else resident, and proven to take the
#: host down when something else is. Above it, it cannot fit at all.
_MAC16_PHYSICAL_GB = 16.0

#: What each machine class can actually give a writer, in GB. The NVIDIA rows
#: are the card minus what the rest of the pipeline needs; 14.5 for the 16 GB
#: class is this repo's own DEFAULT_VRAM_CEILING_GB. The Mac row is a
#: JUDGEMENT, not a measurement: 16 GB shared with macOS, and the measured
#: 14.0 GB writer took the machine down twice, so 12.0 is the last figure with
#: any margin left for the video stack that loads after it.
def _fit_budgets():
    """Resolved at CALL time -- ``DEFAULT_VRAM_CEILING_GB`` is defined further
    down this module, and importing it eagerly here is a NameError at import,
    which empties the writer dropdown rather than failing one badge."""
    return (
        ("mac16", 12.0),
        ("nv8", 7.0),
        ("nv16", DEFAULT_VRAM_CEILING_GB),
        ("nv24", 22.0),
    )


#: MEASURED bf16 resident on Apple Silicon, GB, standalone. A real number here
#: OVERRIDES the linear projection below, and it has to, because the projection
#: is wrong for at least one shipped row.
#:
#: `gemma-4-E2B-it` measures ~10 GB at bf16 -- LARGER than Qwen3.5-4B's ~9 GB --
#: while its download is 6.0 GB against Qwen's 8.68 (docs/MAC_LESSONS_LEARNED.md
#: "Levers that do NOT work"). Scaling by download size therefore gets the two
#: rows in the WRONG ORDER, and briefly had this table recommending E2B to Mac
#: users as the safer pick. The catalog's small figure for E2B is its NF4
#: number, and there is no Metal NF4 kernel, so nothing on this platform ever
#: gets it. The lesson's own conclusion: "Qwen3.5-4B at quant `none` is the
#: smallest viable Mac config, which is why the canonical ships exactly that."
#:
#: Add a row here whenever someone measures one with `footprint -p <pid>`;
#: never infer one.
MEASURED_METAL_BF16_GB = {
    # IN COMFYUI, not standalone, and the difference is the whole number:
    # docs/MAC_LESSONS_LEARNED.md records Qwen3.5-4B at "~9 GB standalone /
    # ~14 GB in ComfyUI". This table feeds a dropdown inside ComfyUI, so the
    # standalone figure would understate every row by ~5 GB.
    "Qwen/Qwen3.5-4B": 14.0,          # PBUG-20260907-06, footprint -p
    # E2B has no in-ComfyUI measurement. What IS measured is the comparison,
    # like for like: ~10 GB standalone against Qwen's ~9. So it is at least as
    # large as Qwen in ComfyUI, and 14.0 is the FLOOR rather than a reading --
    # recorded as such rather than left to a projection that puts it 4 GB
    # BELOW Qwen and calls it the safer pick.
    "google/gemma-4-E2B-it": 14.0,
}


def metal_resident_gb(repo_id: str, download_gb: float) -> tuple:
    """``(gb, "measured"|"projected")`` -- what this row costs at bf16 on Metal.

    Measurement first, always. The projection is a single-point linear model
    and is known to invert at least one pair of shipped rows.
    """
    known = MEASURED_METAL_BF16_GB.get(repo_id)
    if known:
        return (float(known), "measured")
    return (download_gb * _METAL_BF16_RESIDENT_FACTOR, "projected")


def fit_tags_for(repo_id: str) -> tuple:
    """Machine classes this writer FITS, smallest first. Derived, never typed.

    WHY THE PICKER NEEDS THIS AND A SIZE IS NOT ENOUGH. The operator's rule
    (2026-08-01) is that a dropdown states "how much VRAM it needs so users
    select only the one they can use". One number cannot do that across these
    machines, because the same row costs DIFFERENT amounts depending on whether
    bitsandbytes exists: `Qwen/Qwen3.5-4B` is ~4.3 GB resident quantized on an
    NVIDIA card and a measured 14 GB unquantized on Apple Silicon. A single
    figure has to be wrong on one of them, and it was wrong on the platform
    where being wrong REBOOTS THE MACHINE.

    So the badge states the download size -- a platform-independent fact -- and
    these tags carry the fit. A missing `mac16` means "do not pick this on a
    16 GB Mac"; it gates nothing, and the row stays selectable everywhere.
    """
    curated = _by_repo_id().get(repo_id)
    if curated is None or getattr(curated, "provider", "local") != "local":
        return ()
    download_gb = float(getattr(curated, "approx_safetensors_gb", 0.0) or 0.0)
    if download_gb <= 0.0:
        return ()
    tags = []
    for name, budget_gb in _fit_budgets():
        # Apple Silicon pays the unquantized price; NVIDIA gets bitsandbytes.
        resident = (metal_resident_gb(repo_id, download_gb)[0]
                    if name == "mac16" else download_gb / 2.0)
        if resident <= budget_gb:
            tags.append(name)
        elif name == "mac16" and resident <= _MAC16_PHYSICAL_GB:
            # MARGINAL, AND THE BINARY VERSION OF THIS WAS WRONG. A first cut
            # tagged only <= 12.0 and dropped `mac16` from `Qwen/Qwen3.5-4B`
            # (14.0 GB projected) -- for a writer that had ALREADY PUBLISHED a
            # complete 23-beat episode on the 16 GB M4 that same day
            # (`lightning_mac_proof_2_..._q354b_...mp4`; `q354b` is this row).
            # Telling an operator a model cannot run when a receipt says it did
            # is the same defect the flux2_klein cell was, pointing the other
            # way.
            #
            # It is also not simply fine: the same combination hard-rebooted the
            # machine hours later, at the writer -> video handover, with a test
            # suite competing for RAM. Both facts are true, so the tag says so
            # rather than picking one -- it fits when nothing else is resident,
            # and there is no margin for anything that is.
            tags.append(name + "-tight")
    return tuple(tags)


def vram_badge_for(repo_id: str) -> str:
    """``' (11.9 GB)'`` for a model whose resident cost is known, else ``''``.

    WHY THE PICKER CARRIES A NUMBER (operator, 2026-08-01): "all dropdowns
    should state the name of the model and how much VRAM it needs so users
    select only the one they can use." A dropdown that offers a model the box
    cannot load is a broken menu, and the failure arrives minutes later as a
    VRAMFitFailedError rather than at the moment of choosing.

    It reports the same figure :func:`_estimate_resident_gb` computes, so the
    badge and the gate cannot drift apart. Remote handles and un-estimable rows
    get no badge rather than a guess -- the download-state badge was removed
    for exactly that reason, and a wrong number is worse than none.

    GGUF ROWS ARE PRICED AT THE DEFAULT CONTEXT, AND SAY SO (PBUG-20260829-17).
    A GGUF row's cost is weights + KV, and KV scales with ``n_ctx`` -- so a
    single number is only meaningful once you know which context it assumes.
    This used to assume the row's MAXIMUM, which priced the rarest case as the
    norm: `unsloth/Qwen3-4B-Instruct-2507-GGUF` showed **7.9 GB** (2.3 weights
    + 5.6 KV at n_ctx 8192) when the same model needs **5.2 GB** at the default
    4096 -- measured on an 8 GB card. Only 6 of 94 shipped profiles request
    8192; 69 request 4096.

    THE COST OF THAT WAS A USER WALKING AWAY. An 8 GB owner reads "(7.9 GB)"
    against an 8 GB card and skips the smallest, cheapest-to-download,
    Apache-2.0 writer in the list -- 2.3 GB on disk, loads with headroom. The
    operator's directive is that a dropdown states "how much VRAM it needs so
    users select only the one they can use"; 7.9 was not what it needs.

    Same defect shape as PBUG-20260829-08 -- pricing the ROW's maximum instead
    of the REQUEST -- which was fixed in the gate and survived here in the
    label. The context is now named in the badge so the number can never again
    be read as unconditional."""
    ctx = None
    suffix = ""
    try:
        from ._otr_shared.llm_policy import LLMRuntimePolicy as _Policy
        row = None
        try:
            from . import _otr_gguf_backend as _gguf
            row = _gguf.gguf_row_for_repo(repo_id)
        except Exception:  # noqa: BLE001 -- not a GGUF row; no context term
            row = None
        if row is not None and row.kv_gb_per_1k:
            ctx = int(getattr(_Policy, "gguf_n_ctx", 4096)
                      if isinstance(getattr(_Policy, "gguf_n_ctx", None), int)
                      else _Policy.__dataclass_fields__["gguf_n_ctx"].default)
            suffix = " @%dk ctx" % (ctx // 1024)
    except Exception:  # noqa: BLE001 -- a badge must never break the picker
        ctx, suffix = None, ""
    try:
        est = _estimate_resident_gb(repo_id, context_cap=ctx)
    except Exception:  # noqa: BLE001 -- a badge must never break the picker
        return ""
    if not est or est <= 0:
        return ""

    # STATE THE DOWNLOAD, THEN WHERE IT FITS. The resident estimate stays the
    # gate's number and is no longer what the label leads with, because it is
    # platform-blind: it halves every row on the stated assumption of 8-bit/NF4
    # loading, which does not exist on Apple Silicon. The shipped Mac default
    # read "(4.3 GB)" for a model that measured 14 GB there, and a reader who
    # trusted that badge lost the machine. The download size is true
    # everywhere; the tags carry what changes.
    curated = _by_repo_id().get(repo_id)
    download_gb = float(getattr(curated, "approx_safetensors_gb", 0.0) or 0.0)
    tags = list(fit_tags_for(repo_id))
    # FRICTION IS PART OF "CAN I USE THIS", not a separate question. A row that
    # fits the machine but needs a licence click and an HF_TOKEN is not the
    # same offer as one that just downloads, and the whole point of the shipped
    # graphs is that a dropdown choice costs nothing but bandwidth. Derived
    # from the row's own flag, never typed.
    if repo_id in GATED_CURATED_MODELS or getattr(curated, "requires_auth", False):
        tags.insert(0, "gated")
    if download_gb > 0 and tags:
        return " (%.1f GB, %s)" % (download_gb, " ".join(tags))
    if download_gb > 0:
        # Fits nothing in the table -- say the size and say nothing false.
        return " (%.1f GB)" % download_gb
    return " (%.1f GB%s)" % (float(est), suffix)


def _top_installed_alternatives(hub_root: Path | None = None) -> list[str]:
    """Up to 5 locally-scanned curated entries. Used in recovery hints."""
    on_disk = [e.repo_id for e in build_dropdown_choices(hub_root=hub_root) if e.on_disk]
    return on_disk[:5]


def _unknown_recovery_hint(model_id: str, reason: str, hub_root: Path | None = None) -> str:
    alts = _top_installed_alternatives(hub_root=hub_root)
    alts_str = ", ".join(alts) if alts else "<none installed>"
    return (
        f"model_id {model_id!r} could not be resolved or downloaded. "
        f"Reason: {reason}. Install via 'huggingface-cli download {model_id}' "
        f"once, or pick from your installed set: {alts_str}"
    )


def validate_model_id(
    model_id: str,
    *,
    auto_download_enabled: bool | None = None,
    allow_remote: bool | None = None,
    hub_root: Path | None = None,
) -> str:
    """Strip the [NOT DOWNLOADED] label suffix, structurally reject
    unsafe ids, then admit on one of:
        1. curated     -- in CURATED_LLM_MODELS
        2. locally-scanned -- matches a folder in HF_HOME/hub
        3. arbitrary org/name -- valid HF shape AND auto-download
           enabled (default ON) OR OTR_MODEL_CATALOG_ALLOW_REMOTE=1

    Returns the normalized (stripped) repo id. Raises UnknownModelError
    with an actionable recovery hint otherwise. Never silently
    substitutes a different model.

    Both auto_download_enabled / allow_remote default to reading the
    env vars OTR_MODEL_CATALOG_AUTO_DOWNLOAD (default 1) and
    OTR_MODEL_CATALOG_ALLOW_REMOTE (default 0) when None.
    """
    from ._otr_model_inputs import UnknownModelError

    if auto_download_enabled is None:
        auto_download_enabled = otr_env.get(
            "OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1"
        ) != "0"
    if allow_remote is None:
        allow_remote = otr_env.get("OTR_MODEL_CATALOG_ALLOW_REMOTE", "0") == "1"

    if not isinstance(model_id, str):
        raise UnknownModelError(
            _unknown_recovery_hint(repr(model_id), "model_id is not a string", hub_root=hub_root)
        )
    normalized = _strip_label_suffix(model_id)
    reason = _structural_reject(normalized)
    if reason is not None:
        raise UnknownModelError(_unknown_recovery_hint(normalized, reason, hub_root=hub_root))

    # Path 1: curated
    if normalized in _by_repo_id():
        return normalized

    # Path 2: locally-scanned (any repo on disk, even if not curated)
    scan = {r.repo_id: r for r in scan_local_llm_cache(hub_root=hub_root)}
    if normalized in scan and scan[normalized].on_disk:
        return normalized

    # Path 3: arbitrary org/name only when network paths are enabled
    if _HF_REPO_ID_RE.match(normalized) and (auto_download_enabled or allow_remote):
        return normalized

    # [OpenRouter S6] In-app hint: an OpenRouter handle that reached here
    # means remote is not enabled (the virtual rows are absent from the
    # curated set, so Path 1 missed). Give a clear, actionable error that
    # names the env vars + the setup guide, instead of the generic hint.
    if normalized.startswith("openrouter:"):
        raise UnknownModelError(
            f"{normalized!r} is an OpenRouter remote model, but remote is "
            f"not enabled. Set OPENROUTER_API_KEY "
            f"(plus OPENROUTER_MODEL_A / OPENROUTER_MODEL_B), then restart "
            f"ComfyUI in a fresh terminal. See "
            f"https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
            f"v2.0-alpha/docs/openrouter-setup.md."
        )

    # In-app hint: a Comfy Credits handle that reached here means the lane
    # is not enabled (the virtual rows are absent from the curated set, so
    # Path 1 missed). Name the env var + setup guide.
    if normalized.startswith("comfy:"):
        raise UnknownModelError(
            f"{normalized!r} is a Comfy Credits remote model, but the lane "
            f"is not enabled. Set OTR_ENABLE_COMFY_CREDITS=1 and log in to a "
            f"Comfy account with credits, then restart ComfyUI in a fresh "
            f"terminal. See "
            f"https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
            f"v2.0-alpha/docs/comfy-credits-setup.md."
        )

    if normalized.startswith("google_api:"):
        raise UnknownModelError(
            f"{normalized!r} is a Google API remote model, but no Gemini API "
            f"key is configured. Set OTR_GOOGLE_API_KEY, GEMINI_API_KEY, or "
            f"GOOGLE_API_KEY, then restart ComfyUI. No local fallback is used."
        )

    raise UnknownModelError(
        _unknown_recovery_hint(
            normalized,
            "did not match curated / locally-scanned / arbitrary-org-name "
            "admit-paths (auto-download disabled or invalid HF shape)",
            hub_root=hub_root,
        )
    )


# ---------------------------------------------------------------------------
# B1b: dynamic context-cap resolution (replaces _otr_model_loader's
# MODEL_CONTEXT_CAPS static dict + DEFAULT_CONTEXT_CAP)
# ---------------------------------------------------------------------------


# Legacy estimates remain available when native metadata is unknown. They do
# not constrain a known model window or reserve KV memory on native HF routes.
DEFAULT_CONTEXT_ESTIMATE = 8192
_CONTEXT_PIN_UNSET = object()


def normalized_context_pin(raw: Any) -> int | None:
    """Normalize an explicit positive integer; absence never invents a pin."""
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _hard_vram_context_limit() -> int | None:
    return normalized_context_pin(otr_env.get("OTR_HARD_VRAM_CONTEXT_LIMIT"))


# Compatibility export for old informational consumers. Runtime resolution
# reads the explicit setting once per request instead of this import-time value.
HARD_VRAM_CONTEXT_LIMIT = _hard_vram_context_limit() or DEFAULT_CONTEXT_ESTIMATE


# Historical curated working-window estimates, used only without native config.
CURATED_CONTEXT_OVERRIDES: dict[str, int] = {
    "Qwen/Qwen3.5-4B": 8192,
    # These historical working windows are estimates until config is available.
    "unsloth/Llama-3.2-3B-Instruct": 8192,
    "mistralai/Mistral-Nemo-Instruct-2407": 16384,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-4-E2B-it": 8192,
    "google/gemma-4-E4B-it": 8192,
    "google/gemma-4-12b-it": 8192,
    # 2026-08-25: the Qwen2.5-14B, Captain-Eris and MN-12B-Mag-Mell entries
    # were removed with (or after) their catalog rows. An override for a
    # repo_id no curated row can name is unreachable -- resolve_context_cap
    # only ever reaches this dict via .get(model_id) for a real selection --
    # so a stale key is dead weight that reads like a supported model.
}


@dataclass(frozen=True)
class ContextCapVerdict:
    """Tiered verdict from resolve_context_cap. Mirrors VRAMFitVerdict
    shape (B1c). Never raises -- the request_slot escalation logic in
    B1c makes a single combined decision."""

    tier: Literal["PASS", "WARN", "UNKNOWN"]
    value: int
    source: str
    native_capacity: int | None = None
    explicit_pin: int | None = None


def _read_config_context(model_id: str, hub_root: Path | None = None) -> int | None:
    """Best-effort read of advertised context window from a locally-
    scanned snapshot's config.json. Returns None if no snapshot is on
    disk or config.json is unreadable."""
    for r in scan_local_llm_cache(hub_root=hub_root):
        if r.repo_id == model_id and r.on_disk and r.advertised_context:
            return r.advertised_context
    return None


def resolve_context_cap(
    model_id: str, *, hub_root: Path | None = None,
    context_pin: Any = _CONTEXT_PIN_UNSET, config: Any = None,
) -> ContextCapVerdict:
    """Prefer actual native capacity; label historical fallbacks as estimates.

    Explicit pins constrain a known window, never expand it. Passing None
    explicitly preserves an unpinned request across download/load/reuse.
    """
    pin = (_hard_vram_context_limit() if context_pin is _CONTEXT_PIN_UNSET
           else normalized_context_pin(context_pin))
    native = (read_native_context(config) if config is not None
              else _read_config_context(model_id, hub_root=hub_root))
    if native is not None:
        value = min(native, pin) if pin is not None else native
        source = ("loaded decoder config" if config is not None else "snapshot config.json")
        source += f" (native {native})"
        if pin is not None:
            source += f", explicit context pin {pin}"
        return ContextCapVerdict("PASS", value, source, native, pin)
    estimate = CURATED_CONTEXT_OVERRIDES.get(model_id, DEFAULT_CONTEXT_ESTIMATE)
    value = min(estimate, pin) if pin is not None else estimate
    source = ("curated context estimate" if model_id in CURATED_CONTEXT_OVERRIDES
              else "unknown native capacity; default context estimate")
    source += f" {estimate}"
    if pin is not None:
        source += f", explicit context pin {pin}"
    return ContextCapVerdict("UNKNOWN", value, source, None, pin)


# ---------------------------------------------------------------------------
# B1c: check_vram_fit -- tiered VRAMFitVerdict (PASS / WARN / UNKNOWN / FAIL)
# ---------------------------------------------------------------------------


# 16 GB rig usable ceiling. 14.5 GB target -- DWM + background apps eat
# the rest. live LibreHardwareMonitor polling is the real-time signal;
# this constant is the conservative ceiling the fit-checker uses for
# the obvious-oversize case (70B-on-16GB).
DEFAULT_VRAM_CEILING_GB = 14.5

# An estimate_gb / ceiling_gb ratio above this triggers FAIL. Below it
# (even with WARN-tier ambiguity) the load proceeds with a logged caution.
_FAIL_RATIO = 1.5


@dataclass(frozen=True)
class VRAMFitVerdict:
    """Tiered verdict from check_vram_fit. Mirrors ContextCapVerdict.
    `soak_tested` only True for curated PASS entries."""

    tier: Literal["PASS", "WARN", "UNKNOWN", "FAIL"]
    estimated_gb: float
    ceiling_gb: float
    reason: str
    soak_tested: bool


# B1d: special-case resident estimates for uncurated ids we KNOW are
# oversize regardless of dtype / quantization. Used so the 70B-on-16GB
# guardrail fires without requiring the model to be added to the
# curated set (we never want to advertise 70B in the dropdown).
# Values are RESIDENT GB estimates -- _estimate_resident_gb returns
# them as-is (curated entries go through the BF16-download/2 path).
SPECIAL_VRAM_ESTIMATES_GB: dict[str, float] = {
    TEST_OVERSIZED_LLM: 42.0,
    # 70B-class roughly 35 GB at NF4, 70 GB at 8-bit, 140 GB BF16; 42
    # sits between NF4 and 8-bit and still trips the 1.5x FAIL ratio on
    # the 14.5 GB ceiling.
}


def _estimate_resident_gb(
    model_id: str,
    *,
    safetensors_gb_hint: float | None = None,
    gguf_quant: str | None = None,
    context_cap: int | None = None,
) -> float | None:
    """Rough heuristic for VRAM resident size on the OTR pipeline.

    OTR's loader (story_orchestrator._load_llm) uses 8-bit / NF4-style
    quantization by default for the Standard / Obsidian profiles, so
    peak resident is roughly half the BF16 safetensors download size.
    Documented numbers per plan section 6:
        Mistral-Nemo:  ~24 GB disk -> ~12 GB resident.
        Gemma-4-E2B:    ~6 GB disk -> ~3 GB resident.

    A factor-of-2 divisor matches both anchor points for curated entries.
    SPECIAL_VRAM_ESTIMATES_GB (B1d) lets us pin an explicit resident
    estimate for an uncurated id we KNOW is oversize -- entries there
    are returned as-is (no halving). `safetensors_gb_hint` lets a caller
    forward an HfApi size estimate for an uncurated remote id; the hint
    is halved like a curated download size.

    The estimate is intentionally coarse -- VRAMFitVerdict tiers
    (PASS/WARN/UNKNOWN/FAIL) are the policy surface; precise math isn't.
    """
    # SPECIAL table wins -- uncurated-but-known-oversize stays in policy
    # surface without polluting the curated dropdown.
    special = SPECIAL_VRAM_ESTIMATES_GB.get(model_id)
    if special is not None:
        return float(special)
    curated = _by_repo_id().get(model_id)
    if curated is not None:
        # gguf_native rows carry the REAL on-disk artifact size (derived from
        # pinned bytes), not a BF16 download -- so NO /2 halve. Peak resident
        # ~= weights on disk + the per-row KV cache at its context window. An
        # unpinned row (approx 0.0 = UNKNOWN) yields None (can't estimate).
        if getattr(curated, "provider", "local") == "gguf_native":
            # PRICE WHAT THE REQUEST ACTUALLY ASKED FOR, not the row's worst
            # case. `curated.approx_safetensors_gb` is `row.approx_artifact_gb()`
            # -- the FIRST pinned artifact, which is Q8_0 on the gemma row -- and
            # the KV term used to read `row.context_window`, the row's MAXIMUM.
            # Together they priced every caller at 11.8 + 5.6 = 17.4 GB no matter
            # what it requested, so a profile asking for Q4_K_M at n_ctx 2048 was
            # judged on a Q8_0 load at 8192 and REFUSED at 2.56x its ceiling.
            # The honest figure for that request is 6.63 + 1.40 = 8.03 GB, a 1.18x
            # WARN. See PBUG-20260829-08.
            weights_gb = 0.0
            kv_gb = 0.0
            try:
                from . import _otr_gguf_backend as _gguf
                _row = _gguf.gguf_row_for_repo(model_id)
                if gguf_quant:
                    try:
                        _fn, _size, _sha = _row.artifact_for_quant(gguf_quant)
                        if _size:
                            weights_gb = float(_size) / (1024.0 ** 3)
                    except Exception:  # noqa: BLE001 -- unknown quant: fall back
                        weights_gb = 0.0
                if _row.kv_gb_per_1k is not None:
                    _ctx = context_cap or _row.context_window
                    kv_gb = (float(_ctx) / 1024.0) * float(_row.kv_gb_per_1k)
            except Exception:  # noqa: BLE001 -- KV additive; absent -> weights only
                kv_gb = 0.0
            if weights_gb <= 0.0:
                # No per-quant pin available: the projected row's own figure.
                weights_gb = float(curated.approx_safetensors_gb)
            if weights_gb <= 0.0:
                return None
            return weights_gb + kv_gb
        return float(curated.approx_safetensors_gb) / 2.0
    if safetensors_gb_hint is not None and safetensors_gb_hint > 0:
        return float(safetensors_gb_hint) / 2.0
    return None


def check_vram_fit(
    model_id: str,
    context_cap: int,
    *,
    ceiling_gb: float | None = None,
    safetensors_gb_hint: float | None = None,
    gguf_quant: str | None = None,
) -> VRAMFitVerdict:
    """Coarse guardrail against the obvious oversize case (70B-on-16GB).
    Returns a tiered verdict (never raises):

      PASS    -- curated entry with soak-tested vram_fit_tier == "PASS"
                 and estimated resident <= ceiling.
      WARN    -- curated WARN entry (ungated 12B/14B at the edge), OR
                 uncurated with parseable size and estimate <= ceiling.
                 Load proceeds with a logged caution.
      UNKNOWN -- uncurated and we can't reliably parse param count /
                 dtype. Load proceeds; rely on the runtime OOM safety net.
      FAIL    -- estimated >= 1.5x ceiling. Only the clearly-oversized
                 case (e.g. Llama-3-70B at ~42 GB resident). Caller raises.

    Honest note: HF config.json has no standardized num_parameters
    field. UNKNOWN is the expected verdict for most uncurated arbitrary
    org/name models. This is a coarse guardrail, not a precise oracle.
    """
    ceiling = ceiling_gb if ceiling_gb is not None else DEFAULT_VRAM_CEILING_GB
    curated = _by_repo_id().get(model_id)
    estimate = _estimate_resident_gb(
        model_id, safetensors_gb_hint=safetensors_gb_hint,
        gguf_quant=gguf_quant, context_cap=context_cap
    )

    # FAIL case first: clearly oversized regardless of curation.
    if estimate is not None and estimate >= ceiling * _FAIL_RATIO:
        return VRAMFitVerdict(
            tier="FAIL",
            estimated_gb=estimate,
            ceiling_gb=ceiling,
            reason=(
                f"estimated {estimate:.1f} GB peak resident vs "
                f"{ceiling:.1f} GB ceiling -- pick a smaller model"
            ),
            soak_tested=False,
        )

    if curated is not None and curated.vram_fit_tier == "PASS" and estimate is not None and estimate <= ceiling:
        return VRAMFitVerdict(
            tier="PASS",
            estimated_gb=estimate,
            ceiling_gb=ceiling,
            reason=f"curated soak-tested PASS @ {estimate:.1f} GB",
            soak_tested=True,
        )

    if curated is not None and estimate is not None:
        return VRAMFitVerdict(
            tier="WARN",
            estimated_gb=estimate,
            ceiling_gb=ceiling,
            reason=(
                f"curated WARN-tier ({curated.vram_fit_tier}) @ "
                f"{estimate:.1f} GB on {ceiling:.1f} GB ceiling -- "
                "may need quantization / offload"
            ),
            soak_tested=False,
        )

    # Hard-to-estimate case: uncurated, no curated rough size.
    return VRAMFitVerdict(
        tier="UNKNOWN",
        estimated_gb=0.0,
        ceiling_gb=ceiling,
        reason=(
            "uncurated model -- HF config.json has no standardized "
            "num_parameters field; rely on runtime OOM safety net + "
            "LibreHardwareMonitor"
        ),
        soak_tested=False,
    )


# ---------------------------------------------------------------------------
# B1a2: HF network surface -- auto-download + size estimate + pre-flight checks
# ---------------------------------------------------------------------------


# Conservative weight-file allow-list. Transformers loader path only.
# Excludes .gguf intentionally (deferred to a future llama.cpp backend).
ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "chat_template*.jinja",
    "*.md",
)

# Pre-flight disk-space margin: leave at least 5 GB free after the
# download lands. Prevents partial-download cleanup brittleness on a
# near-full disk.
_DISK_SPACE_MARGIN_BYTES = 5 * 1024**3


def _format_gated_message(repo_id: str) -> str:
    return (
        f"GatedModelError: {repo_id!r} requires HuggingFace authentication.\n"
        f"To run OTR end-to-end as designed, free one-time setup (~5 min):\n"
        f"  1. Create HF account at https://huggingface.co/join.\n"
        f"  2. Accept the license at https://huggingface.co/{repo_id}.\n"
        f"  3. Set HF_TOKEN in your environment (or HKCU on Windows).\n"
        f"Once configured, this download fires automatically on first Queue."
    )


def estimate_model_size_gb(repo_id: str, *, _hf_api: object | None = None) -> float:
    """Estimate total safetensors download size in GB. Used by the
    pre-fetch disk-space check + queue-UI announcement.

    For curated entries, returns the catalog's approx_safetensors_gb
    immediately (no network call). For uncurated remote ids, calls
    HfApi().model_info(...) -- this is the only path that fires a
    network request; callers MUST be on a user-action code path.

    `_hf_api` is a test seam: pass a mock object with a model_info()
    method to avoid the network call.
    """
    curated = _by_repo_id().get(repo_id)
    if curated is not None:
        return float(curated.approx_safetensors_gb)
    if _hf_api is None:
        from huggingface_hub import HfApi  # local import: defer network deps

        _hf_api = HfApi()
    # B1d: wrap the network call. RepositoryNotFoundError, HfHubHTTPError,
    # ConnectionError, etc. all collapse to UnknownModelError carrying the
    # actionable recovery hint -- callers (e.g. auto_download_if_missing,
    # check_vram_fit) get a stable exception type to surface in the UI.
    try:
        info = _hf_api.model_info(repo_id, files_metadata=True)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 -- broad to also catch network errors
        from ._otr_model_inputs import UnknownModelError

        raise UnknownModelError(
            _unknown_recovery_hint(
                repo_id,
                f"HfApi.model_info failed ({type(exc).__name__}: {exc})",
            )
        ) from exc
    total = 0
    for sibling in getattr(info, "siblings", []) or []:
        size = getattr(sibling, "size", None) or 0
        path = (getattr(sibling, "rfilename", "") or "").lower()
        if path.endswith(".safetensors") or path.endswith(".bin"):
            total += size
    if total <= 0:
        return 0.0
    return float(total) / float(1024**3)


def _free_disk_bytes_for(path: Path) -> int:
    """shutil.disk_usage on the deepest existing parent of `path`."""
    import shutil

    p = path
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return shutil.disk_usage(str(p)).free


def auto_download_if_missing(
    repo_id: str,
    *,
    hub_root: Path | None = None,
    progress_pbar: object | None = None,
    _snapshot_download: object | None = None,
    _hf_api: object | None = None,
) -> str:
    """Resolve `repo_id` to a local snapshot path, downloading on first
    use. Three pre-flight checks fire BEFORE snapshot_download:

        1. OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0 -> UnknownModelError.
        2. Gated curated repo + no HF_TOKEN     -> GatedModelError.
        3. Free disk - estimated size - margin <= 0 -> InsufficientDiskSpaceError.

    Test seams: pass `_snapshot_download` (callable replacing
    huggingface_hub.snapshot_download) and `_hf_api` (object with
    model_info() method) to drive tests without network calls.
    """
    # EXECUTION path -- use the Hub-aware resolver so a cached
    # `hf auth login` is honoured, not just env/HKCU (PBUG-20260829-10).
    from ._otr_hf_auth import resolve_hf_token_runtime as resolve_hf_token
    from ._otr_model_inputs import (
        GatedModelError,
        InsufficientDiskSpaceError,
        UnknownModelError,
    )

    # B1d: local-cache short-circuit FIRST. If the snapshot is already on
    # disk, return the path immediately. This also makes a cached gated
    # repo (e.g. Mistral-Nemo) usable when HF_TOKEN is unset -- the user
    # downloaded it once; we don't punish them for losing their token.
    scan = {r.repo_id: r for r in scan_local_llm_cache(hub_root=hub_root)}
    cached = scan.get(repo_id)
    if cached is not None and cached.on_disk and cached.snapshot_path:
        return cached.snapshot_path

    if otr_env.get("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1") == "0":
        raise UnknownModelError(
            _unknown_recovery_hint(
                repo_id,
                "auto-download disabled via OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0",
                hub_root=hub_root,
            )
        )

    # Pre-flight gated check: must run BEFORE any HF API call (otherwise
    # the user gets a generic 401 from snapshot_download).
    if repo_id in GATED_CURATED_MODELS and resolve_hf_token() is None:
        raise GatedModelError(_format_gated_message(repo_id))

    # Pre-flight size estimate + disk-space check.
    size_gb = estimate_model_size_gb(repo_id, _hf_api=_hf_api)
    size_bytes = int(size_gb * 1024**3)
    hub_root_path = hub_root if hub_root is not None else _hf_hub_root()
    if hub_root_path is None:
        # Fall back to default location for the disk-usage check; the
        # actual download will create the dir.
        hub_root_path = Path.home() / ".cache" / "huggingface" / "hub"
    free_bytes = _free_disk_bytes_for(hub_root_path)
    margin = size_bytes + _DISK_SPACE_MARGIN_BYTES
    if size_bytes > 0 and (free_bytes - margin) < 0:
        free_gb = free_bytes / 1024**3
        raise InsufficientDiskSpaceError(
            f"InsufficientDiskSpaceError: downloading {repo_id} requires "
            f"{size_gb:.1f} GB + {_DISK_SPACE_MARGIN_BYTES / 1024**3:.0f} GB "
            f"margin = {(size_gb + 5):.1f} GB, but only {free_gb:.1f} GB free "
            f"at {hub_root_path}. Free up disk space and retry."
        )

    # Announce download intent to the console (cheap; queue UI gets the
    # ProgressBar separately).
    print(
        f"[OTR] Downloading {repo_id} -- {size_gb:.1f} GB -> "
        f"{hub_root_path} (first run only)"
    )

    if _snapshot_download is None:
        from huggingface_hub import snapshot_download as _snapshot_download  # type: ignore

    # Forward the token + allow_patterns; let the caller wire a
    # ProgressBar via tqdm_class if they're on the worker-thread.
    #
    # cache_dir is NOT optional (fixed 2026-08-25). Every other step in this
    # function already resolves `hub_root_path` and uses it -- the local-cache
    # scan at :1822, the disk-space check at :1849, and the "Downloading ... ->
    # {hub_root_path}" line printed at :1863. Only the download itself did not
    # receive it, so the console announced one destination and the bytes landed
    # in another, and every later reader (scan_local_llm_cache, load_llm) looked
    # where the message said rather than where the file went. The model then
    # reads as MISSING forever and re-downloads on every run.
    #
    # Passing it explicitly is the whole fix, and it has to be explicit:
    # huggingface_hub freezes `constants.HF_HUB_CACHE` at IMPORT time, so
    # `ensure_hf_home()`'s os.environ write and prestartup_script.py's HF_HOME
    # default only reach snapshot_download if they happened before
    # huggingface_hub was first imported. The loader's own comment
    # (_otr_model_loader.py:1150-1152) describes exactly the case where they
    # cannot -- ComfyUI Desktop inheriting a stale HF_HUB_CACHE that the helper
    # has to REPAIR after import. An env var cannot win that race; an argument
    # always does.
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "allow_patterns": list(ALLOW_PATTERNS),
        "token": resolve_hf_token(),
        "cache_dir": str(hub_root_path),
    }
    if progress_pbar is not None:
        kwargs["tqdm_class"] = _make_pbar_tqdm_adapter(progress_pbar)
    result = str(_snapshot_download(**kwargs))  # type: ignore[operator]
    # Drive the node's bar to complete explicitly. The mirrored bars only ever
    # see bytes that actually TRANSFER: a file already in the blob cache
    # returns from hf_hub_download before any progress object exists, so on a
    # resumed download (the likely next run after any failure) the aggregate
    # total counts only the remaining shards, and if EVERY file is cached the
    # total stays 0 and the adapter never publishes at all. The bar would then
    # sit wherever it was left while the download had in fact finished.
    # ComfyUI's ProgressBar also throttles updates below its 0.5%/100ms floor
    # unless value >= total, so the last mirrored write can be dropped even in
    # the ordinary case. One unconditional write on the way out fixes both, and
    # mirrors what the visual asset planner already does on its own way out.
    if progress_pbar is not None:
        try:
            progress_pbar.update_absolute(1000, 1000)  # type: ignore[attr-defined]
        except BaseException:  # noqa: BLE001 -- a bar must not fail a download
            pass
    return result


def _make_pbar_tqdm_adapter(pbar: object) -> type:
    """Build a REAL tqdm subclass that mirrors progress into a ComfyUI ProgressBar.

    WHY THIS SUBCLASSES tqdm INSTEAD OF IMITATING IT (PBUG-20260906-08, found by
    the alpha.25 clean-install drill on the 4060). The previous version was a
    hand-rolled stand-in that kept its counters PRIVATE (`self._total`) and
    implemented only update / close / __enter__ / __exit__ / __iter__. That is not
    the contract huggingface_hub actually uses. `snapshot_download` builds two
    parent bars from this class and then, for every file, runs
    `_snapshot_download._AggregatedTqdm.__init__`:

        reconstruct_progress.total = (reconstruct_progress.total or 0) + total
        transfer_progress.total = (transfer_progress.total or 0) + total
        reconstruct_progress.refresh()

    which READS `.total`, WRITES `.total` back, and calls `.refresh()`. The
    stand-in had none of the three, so every LLM download died with
    `AttributeError: '_PBarTqdm' object has no attribute 'total'` the moment a
    cold cache needed one. The five visual assets survived the same run only
    because the asset planner uses `hf_hub_download` directly and never passes a
    `tqdm_class`.

    Adding `total` and `refresh` by hand would have fixed that traceback and left
    the next attribute to be discovered by the next user, because the surface is
    whatever huggingface_hub decides to touch. Subclassing the real tqdm makes the
    whole surface exist and stay WRITABLE, and it costs no new dependency: tqdm is
    a hard requirement of huggingface_hub itself.

    Two details that are load-bearing, both read off the installed
    `huggingface_hub/utils/tqdm.py::_create_progress_bar`:
      * `name=` is injected ONLY for huggingface_hub's own tqdm subclass. We are a
        vanilla-tqdm subclass, so we normally never see it, but it is popped
        defensively because a future version passing it would raise TqdmKeyError.
      * we must NOT pass `disable=True` to silence the console. tqdm short-circuits
        `update()` when disabled and `display()` is then never called, which
        silently forwards nothing. Writing to a throwaway buffer keeps the console
        clean while leaving the update path live.
    """
    import io

    from tqdm.std import tqdm as _tqdm_base

    class _PBarTqdm(_tqdm_base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            # Not `disable=True`: see the docstring. A private buffer keeps the
            # bar off the console without turning the update path off.
            kwargs.setdefault("file", io.StringIO())
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            """Mirror the bar into ComfyUI instead of drawing it.

            tqdm calls this from refresh(), from update() once the interval has
            elapsed, and from close(). Returning True without calling super()
            means nothing is ever rendered to the buffer either.

            This MUST NOT raise. A progress indicator that kills a 24 GB download
            is worse than no progress indicator, so every failure is swallowed:
            a ProgressBar whose update_absolute throws, a None or zero total mid
            aggregation, and a bar touched after close all have to be survivable.

            WHY BaseException AND NOT Exception. tqdm's own ``refresh()`` is::

                self._lock.acquire()
                self.display()
                self._lock.release()

            with NO try/finally (verified in the installed tqdm 4.70.0,
            std.py). Anything this method raises therefore strands
            ``_lock``, and because TqdmDefaultWriteLock keeps its th_lock as a
            CLASS attribute the next ``refresh()`` on ANY tqdm anywhere in the
            process blocks forever. A KeyboardInterrupt arriving inside
            update_absolute would deadlock the whole ComfyUI server, not just
            this download. Losing one Ctrl-C during a progress paint is the far
            cheaper failure, so the guard is total.
            """
            try:
                total = self.total
                if total:
                    pbar.update_absolute(int(self.n), int(total))  # type: ignore[attr-defined]
            except BaseException:  # noqa: BLE001 -- see docstring: tqdm's
                pass              # refresh() leaks its lock on ANY raise
            return True

    return _PBarTqdm


__all__ = [
    "CuratedModel",
    "CURATED_LLM_MODELS",
    "text_only_load_mode",
    "GATED_CURATED_MODELS",
    "DEFAULT_LLM",
    "TEST_TECHNICAL_LLM",
    "TEST_OVERSIZED_LLM",
    "NOT_DOWNLOADED_SUFFIX",
    "LOCAL_HF_SUFFIX",
    "LOCAL_GGUF_SUFFIX",
    "ALLOW_PATTERNS",
    "HARD_VRAM_CONTEXT_LIMIT",
    "CURATED_CONTEXT_OVERRIDES",
    "DEFAULT_VRAM_CEILING_GB",
    "SPECIAL_VRAM_ESTIMATES_GB",
    "ScanResult",
    "DropdownEntry",
    "ContextCapVerdict",
    "VRAMFitVerdict",
    "scan_local_llm_cache",
    "build_dropdown_choices",
    "dropdown_choices",
    "default_llm_option",
    "openrouter_catalog_dropdown_choices",
    "OPENROUTER_ENABLE_SENTINEL",
    "OPENROUTER_EMPTY_CACHE_SENTINEL",
    "comfy_catalog_dropdown_choices",
    "COMFY_ENABLE_SENTINEL",
    "google_api_catalog_dropdown_choices",
    "validate_model_id",
    "estimate_model_size_gb",
    "auto_download_if_missing",
    "resolve_context_cap",
    "check_vram_fit",
]
