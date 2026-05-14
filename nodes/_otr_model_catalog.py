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

# Suffix appended to a curated dropdown entry whose weights are NOT
# present in the local HF cache. Validator strips this before any
# allow-list check; outputs / meta keys MUST broadcast the stripped id.
NOT_DOWNLOADED_SUFFIX = " [NOT DOWNLOADED]"

# ---------------------------------------------------------------------------
# CuratedModel dataclass + curated set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratedModel:
    """A curated LLM the OTR catalog ships with explicit honesty fields."""

    repo_id: str
    requires_auth: bool  # gated repo -> True
    loader_backend: Literal[
        "transformers_safetensors",
        "transformers_multimodal_text_only",
    ]
    vram_fit_tier: Literal["PASS", "WARN", "UNKNOWN", "FAIL"]
    approx_safetensors_gb: float  # download size on disk, not VRAM resident
    notes: str = ""


CURATED_LLM_MODELS: tuple[CuratedModel, ...] = (
    CuratedModel(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="PASS",
        approx_safetensors_gb=24.0,
        notes="Audio C7 regression baseline -- soak-tested. Default for both slots.",
    ),
    CuratedModel(
        repo_id="google/gemma-4-E2B-it",
        requires_auth=True,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="PASS",
        approx_safetensors_gb=6.0,
        notes="Multimodal architecture (matformer / Gemma-3n family) used "
        "in text-only mode. Compact technical-slot option.",
    ),
    CuratedModel(
        repo_id="google/gemma-4-E4B-it",
        requires_auth=True,
        loader_backend="transformers_multimodal_text_only",
        vram_fit_tier="PASS",
        approx_safetensors_gb=9.0,
        notes="Slightly larger technical option, same backend.",
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
    ),
    CuratedModel(
        repo_id="Nitral-AI/Captain-Eris_Violet-V0.420-12B",
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="WARN",
        approx_safetensors_gb=24.0,
        notes="Ungated community; 12B at the edge, not soak-tested.",
    ),
    CuratedModel(
        repo_id="inflatebot/MN-12B-Mag-Mell-R1",
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="WARN",
        approx_safetensors_gb=24.0,
        notes="Ungated community; same caveat.",
    ),
)


def _by_repo_id() -> dict[str, CuratedModel]:
    return {m.repo_id: m for m in CURATED_LLM_MODELS}


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
    for m in CURATED_LLM_MODELS:
        on_disk = m.repo_id in scan and scan[m.repo_id].on_disk
        label = m.repo_id if on_disk else m.repo_id + NOT_DOWNLOADED_SUFFIX
        entries.append(DropdownEntry(label, m.repo_id, on_disk, curated=True))
    curated_ids = {m.repo_id for m in CURATED_LLM_MODELS}
    for repo_id, result in scan.items():
        if repo_id in curated_ids:
            continue
        if not result.on_disk:
            continue
        entries.append(DropdownEntry(repo_id, repo_id, True, curated=False))
    return entries


def dropdown_choices(hub_root: Path | None = None) -> list[str]:
    """The bare label list ComfyUI INPUT_TYPES wants for a COMBO widget."""
    return [e.label for e in build_dropdown_choices(hub_root=hub_root)]


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


def _estimate_resident_gb(model_id: str) -> float | None:
    """Rough heuristic for VRAM resident size on the OTR pipeline.

    OTR's loader (story_orchestrator._load_llm) uses 8-bit / NF4-style
    quantization by default for the Standard / Obsidian profiles, so
    peak resident is roughly half the BF16 safetensors download size.
    Documented numbers per plan section 6:
        Mistral-Nemo:  ~24 GB disk -> ~12 GB resident.
        Gemma-4-E2B:    ~6 GB disk -> ~3 GB resident.

    A factor-of-2 divisor matches both anchor points. The estimate is
    intentionally coarse -- VRAMFitVerdict tiers (PASS/WARN/UNKNOWN/
    FAIL) are the policy surface; precise math isn't.
    """
    curated = _by_repo_id().get(model_id)
    if curated is None:
        return None
    return float(curated.approx_safetensors_gb) / 2.0


def check_vram_fit(
    model_id: str,
    context_cap: int,
    *,
    ceiling_gb: float | None = None,
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
    estimate = _estimate_resident_gb(model_id)

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
    info = _hf_api.model_info(repo_id, files_metadata=True)  # type: ignore[union-attr]
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
    "ScanResult",
    "DropdownEntry",
    "ContextCapVerdict",
    "VRAMFitVerdict",
    "scan_local_llm_cache",
    "build_dropdown_choices",
    "dropdown_choices",
    "validate_model_id",
    "estimate_model_size_gb",
    "auto_download_if_missing",
    "resolve_context_cap",
    "check_vram_fit",
]
