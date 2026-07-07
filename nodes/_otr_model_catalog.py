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
"""Default for both writer slots. Audio C7 byte-identical baseline."""

TEST_TECHNICAL_LLM = "google/gemma-4-E2B-it"
"""Used by B6 routing tests + the manual VRAM-profile script to drive
Slot 1 != Slot 2. Compact multimodal-text-only technical option."""

TEST_OVERSIZED_LLM = "meta-llama/Llama-3.1-70B-Instruct"
"""Used by B1c VRAM-fit tests as a known-fails-on-16GB target. Not
added to the dropdown."""

_REMOVED_GEMMA4_12B_MODEL_IDS: frozenset[str] = frozenset({"google/gemma-4-12b-it"})

# Suffix appended to a curated dropdown entry whose weights are NOT
# present in the local HF cache. Validator strips this before any
# allow-list check; outputs / meta keys MUST broadcast the stripped id.
NOT_DOWNLOADED_SUFFIX = " [NOT DOWNLOADED]"

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
    # "gguf_native" = a virtual row backed by an in-process llama-cpp-python
    # GGUF loader. It is local VRAM, not a remote/HTTP zero-VRAM row.
    # Default "local" so every pre-existing row and any older fixture that
    # omits the field still constructs unchanged.
    provider: Literal[
        "local", "openrouter", "comfy_credits", "gguf_native",
    ] = "local"


CURATED_LLM_MODELS: tuple[CuratedModel, ...] = (
    CuratedModel(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="PASS",
        approx_safetensors_gb=24.0,
        notes="Audio C7 regression baseline -- soak-tested. Default for both slots.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
    ),
    CuratedModel(
        repo_id="google/gemma-4-E2B-it",
        requires_auth=True,
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
        requires_auth=True,
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
    CuratedModel(
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="WARN",
        approx_safetensors_gb=28.0,
        notes="Ungated; 14B safetensors needs quantization or offload to fit "
        "16 GB -- not soak-tested as PASS yet. Available for users with bigger "
        "rigs; NOT advertised in gated-error recovery hint.",
        prompt_profile="modern",
        chat_template_kind="transformers_default",
        stop_tokens=(),
        context_window=8192,
        license="apache_2_0",
        license_audit_status="mit_equivalent",
    ),
    # 2026-05-23: catalog pruned -- the two community WARN-tier 12B
    # rows (Captain-Eris_Violet-V0.420-12B, MN-12B-Mag-Mell-R1) were
    # removed. The curated set is Mistral-Nemo + gemma-4-E2B +
    # gemma-4-E4B + Qwen2.5-14B-Instruct.
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


def _gguf_native_virtual_rows() -> tuple[CuratedModel, ...]:
    """The native GGUF Gemma 4 12B row.

    The visible handle is the actual GGUF repository id so the dropdown reads
    like a peer to the other Gemma rows. The loader resolves the local Q8_0
    file from C:\\ComfyUI-Models by default.
    """
    try:
        from . import _otr_gguf_backend as _gguf
    except Exception:  # noqa: BLE001 -- catalog import must stay robust
        return ()
    return (
        CuratedModel(
            repo_id=_gguf.ROW_ID,
            requires_auth=False,
            loader_backend=_gguf.GGUF_BACKEND_KEY,
            vram_fit_tier="PASS",
            approx_safetensors_gb=13.4,
            notes="Gemma 4 12B Q8_0 GGUF via in-process llama-cpp-python. "
            "Default file: C:\\ComfyUI-Models\\LLM\\converted\\"
            "gemma-4-12b-it\\gemma-4-12b-it-Q8_0.gguf. No Ollama, no "
            "sidecar, no port.",
            prompt_profile="modern",
            chat_template_kind="transformers_default",
            stop_tokens=(),
            context_window=_gguf.DEFAULT_CONTEXT_WINDOW,
            license="apache_2_0",
            license_audit_status="mit_equivalent",
            provider="gguf_native",
        ),
    )


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
        if row.repo_id == "google/gemma-4-E4B-it":
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
    """Resolve the HuggingFace hub cache root. Order:
        1. HF_HOME env (OTR's canonical location -- see memory)
        2. HUGGINGFACE_HUB_CACHE env
        3. ~/.cache/huggingface/hub  (default)
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        root = Path(hf_home) / "hub"
        if root.is_dir():
            return root
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        root = Path(hub_cache)
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
        # Most recently modified snapshot wins (HF stores multiple
        # snapshots per repo by commit hash).
        snapshot = max(snapshot_paths, key=lambda p: p.stat().st_mtime)
        ctx = _read_advertised_context(snapshot)
        out.append(ScanResult(repo_id, True, str(snapshot), ctx))
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
    """Merge curated set with locally-scanned cache; apply
    [NOT DOWNLOADED] suffix to curated entries not present on disk.
    """
    scan = {r.repo_id: r for r in scan_local_llm_cache(hub_root=hub_root)}
    entries: list[DropdownEntry] = []
    active = _active_curated_models()
    for m in active:
        if getattr(m, "provider", "local") != "local":
            # Remote (OpenRouter / Comfy Credits): no local weights, so the
            # [NOT DOWNLOADED] suffix would be misleading. Show the clean
            # named handle and treat it as available (selectable) whenever
            # it is present -- a remote row is present only when its lane is
            # enabled (C3).
            entries.append(DropdownEntry(m.repo_id, m.repo_id, True, curated=True))
            continue
        on_disk = m.repo_id in scan and scan[m.repo_id].on_disk
        label = m.repo_id if on_disk else m.repo_id + NOT_DOWNLOADED_SUFFIX
        entries.append(DropdownEntry(label, m.repo_id, on_disk, curated=True))
    curated_ids = {m.repo_id for m in active}
    for repo_id, result in scan.items():
        if repo_id in curated_ids:
            continue
        if not result.on_disk:
            continue
        # The HF hub cache mixes every model type OTR downloads, so a
        # non-curated cache hit is not necessarily a text-generation
        # LLM. Admit it to the writer dropdown only if its config.json
        # declares a `*ForCausalLM` architecture; this keeps diffusion
        # (FLUX, LTX-Video) and vision (Depth-Anything) checkpoints out
        # of the model picker. The curated rows above are exempt -- they
        # are the explicit writer set. See BUG-LOCAL-257.
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

# Top-N newest-by-`created` models that lead the "recent" tier.
_OPENROUTER_RECENT_COUNT = 8


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


# 2026-06-20: pruned + future-proof OpenRouter dropdown. The full catalog is
# 300+ concrete slugs (a huge scroll that also dates fast). The dropdown now
# leads with the `~author/family-latest` ROUTING ALIASES (dynamic newest per
# family -- they never need updating when a new model ships) so the common
# frontier picks need no scrolling. These are routing features and may not all
# appear in /api/v1/models, so they are offered UNCONDITIONALLY (valid slugs
# regardless of the cache). The full alphabetical catalog is opt-in via
# OTR_OPENROUTER_FULL_CATALOG=1 for power users who need a specific concrete
# slug. See reference_openrouter_latest_slugs (operator-supplied, doc-confirmed).
OPENROUTER_FRONTIER_LATEST = (
    "~anthropic/claude-opus-latest",
    "~openai/gpt-latest",
    "~google/gemini-pro-latest",
    "~anthropic/claude-sonnet-latest",
    "~anthropic/claude-haiku-latest",
    "~anthropic/claude-fable-latest",
    "~openai/gpt-mini-latest",
    "~google/gemini-flash-latest",
    "~moonshotai/kimi-latest",
)

# Providers that have NO `~latest` resolver today (so the dropdown adds their
# single NEWEST concrete slug from the live catalog -- auto-updates as new
# versions land, no hardcoded version to maintain).
OPENROUTER_NO_LATEST_AUTHORS = ("x-ai",)


# Suffixes that mark a NON-frontier variant (dev builds, small/fast tiers, free
# previews). Excluded when auto-picking an author's frontier model so the
# dropdown keeps e.g. x-ai/grok-4.3, not x-ai/grok-build-0.1 / grok-mini.
_NON_FRONTIER_MARKERS = (
    "build", "mini", "nano", "lite", "fast", "code", "preview", "beta",
    "draft", ":free", "-free", "instant",
)


def _newest_concrete_for_author(models: list[dict], author: str) -> str | None:
    """The newest FRONTIER concrete slug for an author lacking a `~latest` alias
    (e.g. x-ai/Grok). Skips dev-build / mini / preview variants, then takes the
    highest `created` (then highest id as a tiebreak). Falls back to the newest
    of all rows if every row looks non-frontier. None if no rows. Pure; never
    raises."""
    try:
        prefix = author.lower() + "/"
        rows = [m for m in models if str(m.get("id", "")).lower().startswith(prefix)]
        if not rows:
            return None
        frontier = [
            m for m in rows
            if not any(mark in str(m.get("id", "")).lower()
                       for mark in _NON_FRONTIER_MARKERS)
        ]
        pool = frontier or rows

        def _key(m: dict):
            c = m.get("created")
            return (float(c) if isinstance(c, (int, float)) else 0.0, str(m.get("id", "")))

        return sorted(pool, key=_key, reverse=True)[0]["id"]
    except Exception:  # noqa: BLE001
        return None


def openrouter_catalog_dropdown_choices(slot: str) -> list[str]:
    """Slug-picker choices for openrouter_slot_<slot>_model (slot 'a' / 'b').

    Sentinel-led in EVERY state (BUG-LOCAL-400): OPENROUTER_ENABLE_SENTINEL is
    always choices[0] -- the 'off / use-local' default -- so a saved workflow
    that stores it validates whether or not the lane is enabled.
    Remote disabled -> [OPENROUTER_ENABLE_SENTINEL].
    Remote enabled  -> the sentinel, then an ordered, de-duplicated slug list
    drawn from the S0 disk cache, filtered by _filter_catalog_models:
        1. recommended default for the slot -- the per-slot
           OTR_OPENROUTER_SLOT_x_DEFAULT override when set AND present in the
           filtered cache, else the OPENROUTER_RECOMMENDED_*_DEFAULT constant.
           Always offered first so the slot's default value is selectable even
           if a cold cache or a filter would otherwise hide it.
        2. favorites -- OTR_OPENROUTER_FAVORITES, in operator order
        3. recent    -- top-N newest by `created`
        4. the rest  -- alphabetical by id
    Enabled but empty/cold cache ->
    [OPENROUTER_ENABLE_SENTINEL, recommended_default, EMPTY_CACHE_SENTINEL].
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

    # PRUNED + FUTURE-PROOF block (default view only): the `~latest` resolver
    # aliases (dynamic newest, no scrolling, never need version bumps) + the
    # newest concrete slug for providers lacking a `~latest` (e.g. x-ai/Grok).
    # SKIPPED when the operator set an EXPLICIT allowlist / provider-filter --
    # there they asked for an exact narrowed set, so we honour it verbatim.
    allowlist = (os.environ.get("OTR_OPENROUTER_MODEL_ALLOWLIST") or "").strip()
    provider_filter = (os.environ.get("OTR_OPENROUTER_PROVIDER_FILTER") or "").strip()
    explicit_narrowing = bool(allowlist or provider_filter)
    if not explicit_narrowing:
        for mid in OPENROUTER_FRONTIER_LATEST:
            _add(mid)
        for author in OPENROUTER_NO_LATEST_AUTHORS:
            newest = _newest_concrete_for_author(models, author)
            if newest:
                _add(newest)

    def _created(m: dict) -> float:
        c = m.get("created")
        return float(c) if isinstance(c, (int, float)) else 0.0

    for m in sorted(models, key=_created, reverse=True)[:_OPENROUTER_RECENT_COUNT]:
        _add(m["id"])
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
            OPENROUTER_ENABLE_SENTINEL, [lead, OPENROUTER_EMPTY_CACHE_SENTINEL]
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
    """Strip the [NOT DOWNLOADED] suffix if present."""
    s = model_id.strip()
    if s.endswith(NOT_DOWNLOADED_SUFFIX):
        return s[: -len(NOT_DOWNLOADED_SUFFIX)].rstrip()
    return s


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

    if normalized in _REMOVED_GEMMA4_12B_MODEL_IDS:
        raise UnknownModelError(
            f"{normalized!r} was removed from OTR's writer catalog because "
            "the old 11434 sidecar transport was removed and the installed "
            "transformers stack cannot load its gemma4_unified architecture. "
            "Use the GGUF row 'unsloth/gemma-4-12b-it-GGUF' with "
            "gemma-4-12b-it-Q8_0.gguf under C:\\ComfyUI-Models, or choose "
            "Mistral-Nemo / Gemma 4 E2B / "
            "Gemma 4 E4B."
        )

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
# inference pipeline can sanely feed). The Mistral-Nemo entry pins the
# C7 audio-baseline value -- _otr_model_loader's prior static dict said
# 8192 for Mistral-Nemo even though config.json advertises 131072.
# Holding that here keeps audio byte-identity across B1b.
CURATED_CONTEXT_OVERRIDES: dict[str, int] = {
    "mistralai/Mistral-Nemo-Instruct-2407": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-4-E2B-it": 8192,
    "google/gemma-4-E4B-it": 8192,
    "Qwen/Qwen2.5-14B-Instruct": 8192,
    "Nitral-AI/Captain-Eris_Violet-V0.420-12B": 8192,
    "inflatebot/MN-12B-Mag-Mell-R1": 8192,
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
        PASS    -- model_id has an explicit override (soak-tested cap);
                   value = min(override, HARD_VRAM_CONTEXT_LIMIT).
        WARN    -- config.json parses cleanly but model isn't in the
                   override table; value = min(parsed, HARD_VRAM_CONTEXT_LIMIT).
        UNKNOWN -- neither source resolves; value = HARD_VRAM_CONTEXT_LIMIT
                   (the only safe default; B1c's request_slot makes the
                   combined fit/cap decision).

    The clamp handles two real failure modes:
      * "model says 4k but we feed 8k": parsed used, clamped down to
        limit if needed.
      * "model says 128k, we'd OOM on 16 GB": clamped to limit.
    """
    limit = HARD_VRAM_CONTEXT_LIMIT
    override = CURATED_CONTEXT_OVERRIDES.get(model_id)
    if override is not None:
        return ContextCapVerdict(
            tier="PASS",
            value=min(override, limit),
            source=f"curated-override (raw {override})",
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
        model_id, safetensors_gb_hint=safetensors_gb_hint
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
    from ._otr_hf_auth import resolve_hf_token
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
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "allow_patterns": list(ALLOW_PATTERNS),
        "token": resolve_hf_token(),
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
    "validate_model_id",
    "estimate_model_size_gb",
    "auto_download_if_missing",
    "resolve_context_cap",
    "check_vram_fit",
]
