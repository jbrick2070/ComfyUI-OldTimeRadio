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


__all__ = [
    "CuratedModel",
    "CURATED_LLM_MODELS",
    "GATED_CURATED_MODELS",
    "DEFAULT_LLM",
    "TEST_TECHNICAL_LLM",
    "TEST_OVERSIZED_LLM",
    "NOT_DOWNLOADED_SUFFIX",
    "ScanResult",
    "DropdownEntry",
    "scan_local_llm_cache",
    "build_dropdown_choices",
    "dropdown_choices",
    "validate_model_id",
]
