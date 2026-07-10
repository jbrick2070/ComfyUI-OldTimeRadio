"""LLMRuntimePolicy -- the S1 explicit LLM runtime policy (platform matrix).

docs/2026-07-09-platform-portability-final.md section 3: a FROZEN policy
object built from the writer's widgets in ``_resolve_inputs`` and threaded
``_SlotScheduler -> request_slot(slot, id, policy) -> backend.load(...,
policy)``. NO legacy/auto sentinel anywhere: every field default EQUALS
today's resolved 16 GB (nv50) baseline, so an explicit BASELINE policy
reproduces current behavior exactly while replacing the deleted FA2
auto-probe and tag-based auto-quant.

Stdlib-only by contract (this package is imported by the workflow
validator, which must stay light).

Field classes (one per field -- spec section 2):
  W (widget-mapped) : device, attn_impl, quant_policy, vram_ceiling_gb,
                      gguf_n_ctx, gguf_quant
  E (emit/validate) : lane_allowlist (profile-derived; runtime backstop in
                      request_slot + a per-backend assert in every remote
                      lane)
"""
from __future__ import annotations

import dataclasses
from typing import Any

# Lane vocabulary. Local lanes are named for what runs in-process; remote
# lanes match their catalog ``loader_backend`` families 1:1 via
# LANE_BY_LOADER_BACKEND. A profile's lane_allowlist carries THESE tokens.
LANE_TRANSFORMERS = "transformers"
LANE_GGUF = "gguf"
LANE_OPENROUTER = "openrouter"
LANE_COMFY_CREDITS = "comfy_credits"
LANE_GOOGLE_API = "google_api"

ALL_LANES: tuple[str, ...] = (
    LANE_TRANSFORMERS,
    LANE_GGUF,
    LANE_OPENROUTER,
    LANE_COMFY_CREDITS,
    LANE_GOOGLE_API,
)

# Catalog row ``loader_backend`` -> lane token. Rows with no loader_backend
# (plain HF curated rows) are the transformers lane.
LANE_BY_LOADER_BACKEND: dict[str, str] = {
    "transformers_safetensors": LANE_TRANSFORMERS,
    "transformers_multimodal_text_only": LANE_TRANSFORMERS,
    "transformers_gptq_int4": LANE_TRANSFORMERS,
    "gguf_native": LANE_GGUF,
    "openrouter_http": LANE_OPENROUTER,
    "comfy_credits_http": LANE_COMFY_CREDITS,
    "google_api_http": LANE_GOOGLE_API,
}

_DEVICES = ("cuda", "cpu", "mps")
_ATTN_IMPLS = ("sdpa", "flash_attention_2", "eager")
_QUANT_POLICIES = ("bnb_nf4", "bnb_8bit", "none")
_GGUF_QUANTS = ("Q8_0", "Q6_K", "Q4_K_M")


class LLMPolicyError(ValueError):
    """A policy field is outside its enum / range. Fail loud at build."""


@dataclasses.dataclass(frozen=True)
class LLMRuntimePolicy:
    """Frozen, validated LLM runtime policy.

    Defaults are the nv50 16 GB baseline (spec section 4): the identity
    policy. ``vram_ceiling_gb == 0`` means the pre-download VRAM-fit gate
    is DISABLED (cpu tier only -- there is no VRAM to fit).
    """

    device: str = "cuda"
    attn_impl: str = "sdpa"
    quant_policy: str = "bnb_nf4"
    vram_ceiling_gb: float = 14.5
    gguf_n_ctx: int = 4096
    gguf_quant: str = "Q8_0"
    lane_allowlist: tuple[str, ...] = ALL_LANES

    def __post_init__(self) -> None:
        if self.device not in _DEVICES:
            raise LLMPolicyError(
                f"llm.device {self.device!r} not in {_DEVICES}")
        if self.attn_impl not in _ATTN_IMPLS:
            raise LLMPolicyError(
                f"llm.attn_impl {self.attn_impl!r} not in {_ATTN_IMPLS}")
        if self.quant_policy not in _QUANT_POLICIES:
            raise LLMPolicyError(
                f"llm.quant_policy {self.quant_policy!r} not in "
                f"{_QUANT_POLICIES}")
        if not isinstance(self.vram_ceiling_gb, (int, float)) or (
                self.vram_ceiling_gb < 0):
            raise LLMPolicyError(
                f"llm.vram_ceiling_gb must be >= 0 (0 = disabled, cpu tier "
                f"only); got {self.vram_ceiling_gb!r}")
        if not isinstance(self.gguf_n_ctx, int) or self.gguf_n_ctx < 512:
            raise LLMPolicyError(
                f"llm.gguf_n_ctx must be an int >= 512; got "
                f"{self.gguf_n_ctx!r}")
        if self.gguf_quant not in _GGUF_QUANTS:
            raise LLMPolicyError(
                f"llm.gguf_quant {self.gguf_quant!r} not in {_GGUF_QUANTS}")
        lanes = tuple(self.lane_allowlist)
        unknown = [l for l in lanes if l not in ALL_LANES]
        if unknown or not lanes:
            raise LLMPolicyError(
                f"llm.lane_allowlist must be a non-empty subset of "
                f"{ALL_LANES}; got {self.lane_allowlist!r}")
        object.__setattr__(self, "lane_allowlist", lanes)

    # ------------------------------------------------------------------
    def admits_lane(self, lane: str) -> bool:
        return lane in self.lane_allowlist

    def cache_key(self) -> tuple:
        """The fields that shape a LOADED artifact. A resident model whose
        cache_key differs from the requested policy's must be torn down and
        reloaded -- silent stale-policy reuse is exactly the bug class this
        campaign kills. vram_ceiling_gb (pre-load gate) and lane_allowlist
        (admission) do not shape the artifact and are excluded."""
        return (self.device, self.attn_impl, self.quant_policy,
                self.gguf_n_ctx, self.gguf_quant)


#: The nv50 16 GB identity policy -- today's resolved baseline values.
BASELINE_POLICY = LLMRuntimePolicy()


def lane_for_row(row: Any) -> str:
    """Lane token for a catalog row (None row = transformers lane)."""
    lb = getattr(row, "loader_backend", None) if row is not None else None
    if lb is None:
        return LANE_TRANSFORMERS
    try:
        return LANE_BY_LOADER_BACKEND[str(lb)]
    except KeyError:
        raise LLMPolicyError(
            f"catalog row loader_backend {lb!r} has no lane mapping -- add "
            f"it to LANE_BY_LOADER_BACKEND (no silent default)") from None


__all__ = [
    "ALL_LANES",
    "BASELINE_POLICY",
    "LANE_BY_LOADER_BACKEND",
    "LANE_COMFY_CREDITS",
    "LANE_GGUF",
    "LANE_GOOGLE_API",
    "LANE_OPENROUTER",
    "LANE_TRANSFORMERS",
    "LLMPolicyError",
    "LLMRuntimePolicy",
    "lane_for_row",
]
