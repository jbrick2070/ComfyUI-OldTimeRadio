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

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Canonical constants -- single source of truth for tests + wiring code.
# Any future rename / casing fix happens here, not in scattered string literals.
# ---------------------------------------------------------------------------

DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"
"""Fallback for empty/unsaved writer inputs; Audio C7 baseline.

The saved canonical workflow intentionally overrides both slots with the
runtime-qualified ``google/gemma-4-12b-it`` row.
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


CURATED_LLM_MODELS: tuple[CuratedModel, ...] = (
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
        "in text-only mode. Compact technical-slot option.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
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
            "docs/openrouter-setup.md.",
            **common,
        ),
        CuratedModel(
            repo_id=_orb.SLOT_B_ID,
            notes="OpenRouter remote model B (opt-in, default-off). Binds "
            "to OPENROUTER_MODEL_B; zero local VRAM. See "
            "docs/openrouter-setup.md.",
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
            "VRAM. See docs/comfy-credits-setup.md.",
            **common,
        ),
        CuratedModel(
            repo_id=_occ.SLOT_B_ID,
            notes="Comfy Credits remote model B (opt-in, default-off). "
            "Credit-billed via ComfyUI's partner-node proxy; zero local "
            "VRAM. See docs/comfy-credits-setup.md.",
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


_HF_REPO_DIR_RE = re.compile(r"^models--(?P<org>[^-]+)--(?P<name>.+)$")


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
        val = os.environ.get(var)
        if val:
            root = Path(val)
            if root.is_dir():
                return root
    hf_home = os.environ.get("HF_HOME")
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


def _read_advertised_context(snapshot_path: Path) -> int | None:
    """Best-effort read of max_position_embeddings (or fallback fields)
    from a snapshot's config.json. Returns None on any failure -- this
    is informational, not load-bearing."""
    cfg = snapshot_path / "config.json"
    if not cfg.is_file():
        return None
    try:
        import json

        with cfg.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("max_position_embeddings", "n_positions", "n_ctx"):
            val = data.get(key)
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        return None
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


def _safe_snapshot_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _snapshot_has_weights(snapshot_path: Path) -> bool:
    """True iff `snapshot_path` holds at least one materialized weight blob.

    Follows HF symlinks: a snapshot weight entry is a symlink into
    ``../../blobs/<sha>``; ``stat().st_size`` resolves through the link to
    the real blob size, so a present-but-unmaterialized (broken/absent)
    link never counts. Returns False on any read error -- fail closed.
    """
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
    raw = os.environ.get(name)
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _slot_requires_json(slot: str) -> bool:
    """Per-slot structured-output filter. NEVER global: defaults off, so a
    creative model is never hidden from slot A just because slot B needs
    JSON. Reads OTR_OPENROUTER_SLOT_<A|B>_REQUIRE_JSON."""
    var = f"OTR_OPENROUTER_SLOT_{slot.strip().upper()}_REQUIRE_JSON"
    return os.environ.get(var, "0") == "1"


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
    text_only = os.environ.get("OTR_OPENROUTER_ALLOW_NONTEXT", "0") != "1"
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
    configured = (os.environ.get(f"OTR_OPENROUTER_SLOT_{s.upper()}_DEFAULT") or "").strip()
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
    allowlist = (os.environ.get("OTR_OPENROUTER_MODEL_ALLOWLIST") or "").strip()
    provider_filter = (os.environ.get("OTR_OPENROUTER_PROVIDER_FILTER") or "").strip()
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
    if explicit_narrowing or os.environ.get(
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


def vram_badge_for(repo_id: str) -> str:
    """``' (11.9 GB)'`` for a model whose resident cost is known, else ``''``.

    WHY THE PICKER CARRIES A NUMBER (operator, 2026-08-01): "all dropdowns
    should state the name of the model and how much VRAM it needs so users
    select only the one they can use." A dropdown that offers a model the box
    cannot load is a broken menu, and the failure arrives minutes later as a
    VRAMFitFailedError rather than at the moment of choosing.

    This deliberately reports the SAME number the admission gate will compute
    (:func:`_estimate_resident_gb`), so the badge and the refusal can never
    disagree. Remote handles and un-estimable rows get no badge rather than a
    guess -- the download-state badge was removed for exactly that reason, and
    a wrong number is worse than none."""
    try:
        est = _estimate_resident_gb(repo_id)
    except Exception:  # noqa: BLE001 -- a badge must never break the picker
        return ""
    if not est or est <= 0:
        return ""
    return " (%.1f GB)" % float(est)


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
        auto_download_enabled = os.environ.get(
            "OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1"
        ) != "0"
    if allow_remote is None:
        allow_remote = os.environ.get("OTR_MODEL_CATALOG_ALLOW_REMOTE", "0") == "1"

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
            f"ComfyUI in a fresh terminal. See docs/openrouter-setup.md."
        )

    # In-app hint: a Comfy Credits handle that reached here means the lane
    # is not enabled (the virtual rows are absent from the curated set, so
    # Path 1 missed). Name the env var + setup guide.
    if normalized.startswith("comfy:"):
        raise UnknownModelError(
            f"{normalized!r} is a Comfy Credits remote model, but the lane "
            f"is not enabled. Set OTR_ENABLE_COMFY_CREDITS=1 and log in to a "
            f"Comfy account with credits, then restart ComfyUI in a fresh "
            f"terminal. See docs/comfy-credits-setup.md."
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


# Hardware-aware ceiling on the effective context window. A small model
# like Gemma-4-E4B advertises a 128k context in its config.json, but
# feeding 128k tokens on a 16 GB card OOMs instantly. The clamp keeps
# the upper bound sane regardless of the model's claim.
#
# Default: 8192 on the 5080 16 GB target. Configurable via
# OTR_HARD_VRAM_CONTEXT_LIMIT so users on bigger hardware can raise it.
def _hard_vram_context_limit() -> int:
    raw = os.environ.get("OTR_HARD_VRAM_CONTEXT_LIMIT")
    if raw:
        try:
            return max(512, int(raw))
        except (TypeError, ValueError):
            pass
    return 8192


HARD_VRAM_CONTEXT_LIMIT = _hard_vram_context_limit()


# Explicit per-model effective-context overrides for the curated set
# only (where config.json advertises a larger window than what the
# inference pipeline can sanely feed). A soak-tested override for a
# vram_fit_tier=="PASS" row is AUTHORITATIVE (see resolve_context_cap):
# it is the value that row was proven at, not a guess to be re-clamped by
# the generic HARD_VRAM_CONTEXT_LIMIT. Mistral-Nemo is raised to 16384
# (2026-07-19) so the local sci-fi 420/720w script pass fits the
# context window (config.json advertises 131072); NF4 4-bit weights + a
# DynamicCache keep the ~1.8 GB KV at 720w inside the 16 GB card
# (live-proven). Short legs never reached the old 8192 output-budget
# clamp, so C7 audio byte-identity holds. Every other row stays at its
# soak-tested 8192 (a WARN-tier override stays clamped by the hard limit).
CURATED_CONTEXT_OVERRIDES: dict[str, int] = {
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


def _read_config_context(model_id: str, hub_root: Path | None = None) -> int | None:
    """Best-effort read of advertised context window from a locally-
    scanned snapshot's config.json. Returns None if no snapshot is on
    disk or config.json is unreadable."""
    for r in scan_local_llm_cache(hub_root=hub_root):
        if r.repo_id == model_id and r.on_disk and r.advertised_context:
            return r.advertised_context
    return None


def resolve_context_cap(
    model_id: str, *, hub_root: Path | None = None
) -> ContextCapVerdict:
    """Resolve the effective context-window cap for `model_id`.

    Returns a tiered verdict (never raises):
        PASS    -- model_id has an explicit override (soak-tested cap). For a
                   vram_fit_tier=="PASS" catalog row the override is
                   AUTHORITATIVE: value = override (un-clamped), because a
                   soak-tested cap is not a guess to be re-clamped by the
                   generic hardware limit -- re-clamping is exactly what pinned
                   Mistral-Nemo to a false 8192 and truncated the 420/720w
                   script pass. If the operator has EXPLICITLY set
                   OTR_HARD_VRAM_CONTEXT_LIMIT (a smaller-card escape hatch) the
                   value is re-clamped to min(override, limit). A WARN/UNKNOWN
                   catalog row's override is not soak-tested, so it also stays
                   min(override, limit).
        WARN    -- config.json parses cleanly but model isn't in the
                   override table; value = min(parsed, HARD_VRAM_CONTEXT_LIMIT).
        UNKNOWN -- neither source resolves; value = HARD_VRAM_CONTEXT_LIMIT
                   (the only safe default; B1c's request_slot makes the
                   combined fit/cap decision).

    The clamp handles two real failure modes:
      * "model says 4k but we feed 8k": parsed used, clamped down to
        limit if needed.
      * "model says 128k, we'd OOM on 16 GB": clamped to limit -- except a
        PASS-tier override, whose window was already proven on the card.
    """
    limit = HARD_VRAM_CONTEXT_LIMIT
    override = CURATED_CONTEXT_OVERRIDES.get(model_id)
    if override is not None:
        row = _by_repo_id().get(model_id)
        row_is_pass = (
            row is not None
            and getattr(row, "vram_fit_tier", None) == "PASS"
        )
        # An operator who pins OTR_HARD_VRAM_CONTEXT_LIMIT is on a card whose
        # budget we must respect even over a soak-tested override.
        env_pinned = bool(os.environ.get("OTR_HARD_VRAM_CONTEXT_LIMIT"))
        if row_is_pass and not env_pinned:
            return ContextCapVerdict(
                tier="PASS",
                value=int(override),
                source=f"curated-override authoritative (PASS, raw {override})",
            )
        return ContextCapVerdict(
            tier="PASS",
            value=min(override, limit),
            source=f"curated-override (raw {override}, clamped to {limit})",
        )
    parsed = _read_config_context(model_id, hub_root=hub_root)
    if parsed is not None:
        return ContextCapVerdict(
            tier="WARN",
            value=min(parsed, limit),
            source=f"config.json (raw {parsed})",
        )
    return ContextCapVerdict(
        tier="UNKNOWN",
        value=limit,
        source=f"unresolved -- defaulted to HARD_VRAM_CONTEXT_LIMIT={limit}",
    )


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

    if os.environ.get("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1") == "0":
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
    return str(_snapshot_download(**kwargs))  # type: ignore[operator]


def _make_pbar_tqdm_adapter(pbar: object) -> type:
    """Build a minimal tqdm-shaped class that forwards update() into a
    ComfyUI ProgressBar. huggingface_hub's tqdm_class hook expects a
    context-manager class with update(), set_description(), __enter__,
    __exit__.
    """

    class _PBarTqdm:
        def __init__(self, *args, **kwargs):
            self._total = kwargs.get("total")
            self._n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1):
            self._n += int(n)
            if self._total and hasattr(pbar, "update_absolute"):
                try:
                    pbar.update_absolute(self._n, int(self._total))  # type: ignore[attr-defined]
                except Exception:
                    pass

        def set_description(self, *_a, **_kw):
            return None

        def close(self):
            return None

        def __iter__(self):
            return iter(())

    return _PBarTqdm


__all__ = [
    "CuratedModel",
    "CURATED_LLM_MODELS",
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
