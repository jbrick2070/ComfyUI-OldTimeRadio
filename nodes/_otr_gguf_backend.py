"""Native GGUF writer backend for Gemma 4 12B.

This is the in-process GGUF lane for the writer catalog row
``unsloth/gemma-4-12b-it-GGUF``. It uses ``llama-cpp-python`` directly
inside the ComfyUI Python process; it does not start a sidecar, does not
open a port, and does not call Ollama.

The default weight path follows the operator's shared model root:

    C:\\ComfyUI-Models\\LLM\\converted\\gemma-4-12b-it\\gemma-4-12b-it-Q8_0.gguf

Override with ``GEMMA4_12B_GGUF_PATH`` when needed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from ._otr_generation_budget import (
    GenerationContextOverflowError,
    estimate_prompt_tokens,
    fit_output_tokens,
)

log = logging.getLogger("OTR")

GGUF_BACKEND_KEY = "gguf_native"
PROVIDER = "gguf_native"
ROW_ID = "unsloth/gemma-4-12b-it-GGUF"

DEFAULT_GGUF_FILENAME = "gemma-4-12b-it-Q8_0.gguf"
EXPECTED_Q8_0_SIZE_BYTES = 12_669_646_240
DEFAULT_CONTEXT_WINDOW = 4096

# KV-cache cost per 1024 context cells, in GB. Conservative: the preflight would rather
# refuse a load than OOM mid-generation. Override with GEMMA4_12B_KV_GB_PER_1K once the
# real figure has been MEASURED on this box (log the resident VRAM after a load and
# divide) -- a guessed constant that blocks a good config is as bad as one that lets a
# bad config through.
KV_GB_PER_1K_CTX = 0.7
DEFAULT_OUTPUT_TOKENS_CAP = 512
DEFAULT_N_BATCH = 512
DEFAULT_N_GPU_LAYERS = -1

# S1 platform-portability (2026-07-10): quant -> (filename, expected size,
# expected sha256). THE integrity authority for the gemma row -- one table,
# read by the registry row and by the env-fallback path alike.
#
# A quant is PINNED when it carries BOTH a size and a sha. An unpinned quant
# is now REFUSED at load rather than accepted unchecked (A6, 2026-07-27):
# both integrity checks were conditional on their pin being present, so an
# unpinned quant skipped integrity entirely and a truncated download was
# reported ready -- on the very quant the 8 GB profile selects.
# GEMMA4_12B_GGUF_PATH remains the explicit whole-path escape hatch, and a
# file it names is outside this table by construction.
#
# Q8_0 and Q4_K_M were pinned 2026-07-27 by MEASUREMENT on this box, not by
# transcription from a model card. Q4_K_M was hashed in all three copies
# present -- the loader's own converted/gemma-4-12b-it path plus two unsloth
# mirrors -- and all three agree byte for byte; the Q8_0 measurement
# reproduced its already-pinned size exactly, which is what corroborates the
# provenance of the set. Q6_K has never been on this box; pin it the same
# way the first time it is, and never from a guess.
GGUF_ARTIFACTS: dict[str, tuple[str, int | None, str | None]] = {
    "Q8_0": (
        "gemma-4-12b-it-Q8_0.gguf",
        EXPECTED_Q8_0_SIZE_BYTES,
        "74d2d4f0b5b08ca8589d1a5f50e689c0984469f3cedbdc7d67458c6e9e35496a",
    ),
    "Q6_K": ("gemma-4-12b-it-Q6_K.gguf", None, None),
    "Q4_K_M": (
        "gemma-4-12b-it-Q4_K_M.gguf",
        7121860000,
        "43fec98c5102b1c446b4ddd0a9439f1db3a2e1f2e0b8cd143ce1ea619a9403d6",
    ),
}

_DLL_DIR_HANDLES: list[Any] = []
_PRELOADED_DLLS: list[Any] = []
_DLL_RUNTIME_PREPARED = False


class GGUFNativeError(RuntimeError):
    """Base error for the native GGUF lane."""


class GGUFNativeConfigError(GGUFNativeError):
    """The lane was selected but its local config is unusable."""


class GGUFNativeCallFailedError(GGUFNativeError):
    """The native GGUF call failed."""


class GGUFRegistryError(GGUFNativeError):
    """A GGUF_ROWS entry is malformed. Raised at construction -- an
    explicit raise (never an ``assert``, which vanishes under ``-O``) so a
    bad row fails tests/startup loudly instead of silently deleting a lane."""


# ---------------------------------------------------------------------------
# GGUF row registry -- the canonical multi-row source of truth
# ---------------------------------------------------------------------------
# One GGUFRow per shippable GGUF writer model. The catalog projects each row
# into a visible dropdown peer; the preflight resolves a per-slot load_config
# from the selected row + policy. Adding a model is a data edit here, not new
# code.

#: Pinned top_k for every native GGUF generate call. 40 is also llama-cpp's
#: own default, so pinning it is byte-identical for the top_k dimension.
GGUF_TOP_K = 40

#: Closed think-strip policy enum. "none" = leave output untouched;
#: "qwen3_no_think" = strip a single LEADING <think>...</think> envelope when
#: no response_format is in force (see _strip_leading_think_envelope).
_THINK_POLICIES = frozenset({"none", "qwen3_no_think"})

#: characters that make a subdir/filename unsafe (path traversal / absolute).
_UNSAFE_PATH_RE = re.compile(r"(^\s*$)|(\.\.)|([\\/])|(^[A-Za-z]:)|(^~)")


def _require_safe_component(kind: str, repo_id: str, value: str) -> None:
    if not isinstance(value, str) or _UNSAFE_PATH_RE.search(value):
        raise GGUFRegistryError(
            f"GGUF_ROWS[{repo_id!r}]: {kind} {value!r} is not a plain, safe "
            "path component (no separators, parent refs, drive, or ~)."
        )


@dataclasses.dataclass(frozen=True)
class GGUFRow:
    """A single native-GGUF writer model row.

    ``artifacts`` maps a quant name -> (filename, size_bytes|None, sha256|None).
    A pinned size/sha is FAIL-LOUD verified at load; None means the artifact
    has never been materialized on this box (absence/existence checked only,
    pin the real values when first derived). ``kv_gb_per_1k`` is None until the
    per-row KV cost has been MEASURED (never a guessed constant).
    """

    repo_id: str
    subdir: str
    artifacts: dict[str, tuple[str, int | None, str | None]]
    context_window: int
    kv_gb_per_1k: float | None
    vram_fit_tier: str
    license: str
    license_audit_status: str
    requires_auth: bool
    stop_tokens: tuple[str, ...]
    think_policy: str

    def __post_init__(self) -> None:
        rid = self.repo_id
        if not isinstance(rid, str) or not rid.strip():
            raise GGUFRegistryError(f"GGUF_ROWS: empty/invalid repo_id {rid!r}")
        _require_safe_component("subdir", rid, self.subdir)
        if not isinstance(self.artifacts, dict) or not self.artifacts:
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: artifacts must be a non-empty dict."
            )
        for quant, spec in self.artifacts.items():
            if not isinstance(quant, str) or not quant.strip():
                raise GGUFRegistryError(
                    f"GGUF_ROWS[{rid!r}]: invalid quant key {quant!r}."
                )
            if not (isinstance(spec, tuple) and len(spec) == 3):
                raise GGUFRegistryError(
                    f"GGUF_ROWS[{rid!r}] quant {quant!r}: artifact must be a "
                    "(filename, size|None, sha|None) 3-tuple."
                )
            filename, size, sha = spec
            _require_safe_component(f"{quant} filename", rid, filename)
            if size is not None and (not isinstance(size, int) or size <= 0):
                raise GGUFRegistryError(
                    f"GGUF_ROWS[{rid!r}] quant {quant!r}: size must be a "
                    f"positive int or None; got {size!r}."
                )
            if sha is not None and (not isinstance(sha, str) or not sha.strip()):
                raise GGUFRegistryError(
                    f"GGUF_ROWS[{rid!r}] quant {quant!r}: sha must be a "
                    "non-empty str or None."
                )
        if not isinstance(self.context_window, int) or self.context_window <= 0:
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: context_window must be a positive int; "
                f"got {self.context_window!r}."
            )
        if self.kv_gb_per_1k is not None and (
            not isinstance(self.kv_gb_per_1k, (int, float))
            or self.kv_gb_per_1k <= 0
        ):
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: kv_gb_per_1k must be a positive number "
                f"or None (measure it); got {self.kv_gb_per_1k!r}."
            )
        if self.think_policy not in _THINK_POLICIES:
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: think_policy {self.think_policy!r} not "
                f"in {sorted(_THINK_POLICIES)}."
            )
        for field_name in ("vram_fit_tier", "license", "license_audit_status"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise GGUFRegistryError(
                    f"GGUF_ROWS[{rid!r}]: {field_name} must be an explicit "
                    f"non-empty str; got {val!r}."
                )
        if not isinstance(self.requires_auth, bool):
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: requires_auth must be an explicit bool."
            )
        if not isinstance(self.stop_tokens, tuple):
            raise GGUFRegistryError(
                f"GGUF_ROWS[{rid!r}]: stop_tokens must be a tuple."
            )

    def artifact_for_quant(
        self, quant: str,
    ) -> tuple[str, int | None, str | None]:
        try:
            return self.artifacts[quant]
        except KeyError:
            raise GGUFNativeConfigError(
                f"GGUF row {self.repo_id!r} has no artifact for quant "
                f"{quant!r} (available: {sorted(self.artifacts)})."
            ) from None

    def approx_artifact_gb(self) -> float:
        """Coarse on-disk size in GiB DERIVED from the first PINNED artifact.
        A row whose artifacts are all unpinned (size=None) projects 0.0 =
        UNKNOWN -- never a measured/guessed estimate."""
        for _filename, size, _sha in self.artifacts.values():
            if size is not None:
                return round(size / (1024 ** 3), 1)
        return 0.0


# The two shipped rows. gemma-4-12b (baseline, PASS) + Qwen3-8B (deferred
# UNKNOWN until the live bakeoff pins size/sha/kv). The gemma row takes
# GGUF_ARTIFACTS verbatim -- one table, one authority. It used to graft a
# sha=None slot onto every gemma quant here, which silently discarded any
# sha the table might carry (A6).
GGUF_ROWS: tuple[GGUFRow, ...] = (
    GGUFRow(
        repo_id=ROW_ID,  # unsloth/gemma-4-12b-it-GGUF
        subdir="gemma-4-12b-it",
        artifacts=dict(GGUF_ARTIFACTS),
        context_window=DEFAULT_CONTEXT_WINDOW,  # 4096
        kv_gb_per_1k=KV_GB_PER_1K_CTX,          # 0.7
        vram_fit_tier="PASS",
        license="apache_2_0",
        license_audit_status="mit_equivalent",
        requires_auth=False,
        stop_tokens=(),
        think_policy="none",
    ),
    GGUFRow(
        repo_id="unsloth/Qwen3-8B-GGUF",
        subdir="Qwen3-8B",
        # PINNED at the 2026-07-16 live bake-off: 3x RESULT SUCCESS + obs asset,
        # both writer slots Qwen, ctx=8192 on CUDA, peak ~11.8 GB (< 14.5),
        # no silent fallback. size/sha256 from the download gate; kv measured
        # 5.60 GB @ n_ctx=8192 -> 0.70 GB / 1k. vram_fit_tier UNKNOWN -> PASS.
        artifacts={
            "Q4_K_M": (
                "Qwen3-8B-Q4_K_M.gguf",
                5027784512,
                "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4",
            ),
        },
        context_window=8192,
        kv_gb_per_1k=0.70,
        vram_fit_tier="PASS",
        license="apache_2_0",
        license_audit_status="mit_equivalent",
        requires_auth=False,
        stop_tokens=(),
        think_policy="qwen3_no_think",
    ),
)


def _build_rows_by_repo() -> dict[str, GGUFRow]:
    out: dict[str, GGUFRow] = {}
    for row in GGUF_ROWS:
        if row.repo_id in out:
            raise GGUFRegistryError(
                f"GGUF_ROWS: duplicate repo_id {row.repo_id!r} -- repo ids "
                "must be unique."
            )
        out[row.repo_id] = row
    return out


# Constructed at import -> a malformed/duplicate row fails module import LOUD.
GGUF_ROWS_BY_REPO: dict[str, GGUFRow] = _build_rows_by_repo()


def gguf_row_for_repo(repo_id: str) -> GGUFRow:
    try:
        return GGUF_ROWS_BY_REPO[repo_id]
    except KeyError:
        raise GGUFNativeConfigError(
            f"no GGUF_ROWS entry for repo_id {repo_id!r} "
            f"(known: {sorted(GGUF_ROWS_BY_REPO)})."
        ) from None


def row_artifact_path(row: GGUFRow, quant: str) -> Path:
    """Resolve the on-disk path for (row, quant). GEMMA4_12B_GGUF_PATH is the
    whole-path escape hatch for the GEMMA row ONLY -- it never resolves a Qwen
    id to a gemma file."""
    filename, _size, _sha = row.artifact_for_quant(quant)
    if row.repo_id == ROW_ID:
        raw = os.environ.get("GEMMA4_12B_GGUF_PATH")
        if raw:
            return Path(raw).expanduser()
    return _models_root() / "LLM" / "converted" / row.subdir / filename


def gguf_native_row_on_disk(repo_id: str) -> bool:
    """True iff ANY registered artifact for this row resolves to a regular,
    non-zero file. The dropdown has no quant context, so a row is "present"
    when at least one of its quants is materialized."""
    try:
        row = gguf_row_for_repo(repo_id)
    except GGUFNativeConfigError:
        return False
    for quant in row.artifacts:
        try:
            p = row_artifact_path(row, quant)
            if p.is_file() and p.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Effective load-config -- one immutable, row-keyed object per gguf slot
# ---------------------------------------------------------------------------


def _strict_int_env(name: str) -> int | None:
    """Parse an int env override. Absent/blank -> None (use the next source in
    precedence). A PRESENT-but-malformed value RAISES -- no silent fallback."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        raise GGUFNativeConfigError(
            f"{name} must be an integer; got {raw!r} (no silent fallback)."
        ) from None


def _resolve_int_override(
    otr_name: str, gemma_name: str | None, default: int,
) -> int:
    """Whitelisted precedence: OTR_GGUF_* > GEMMA4_12B_* (gemma row only) >
    default. Malformed overrides raise via _strict_int_env."""
    v = _strict_int_env(otr_name)
    if v is not None:
        return v
    if gemma_name is not None:
        v = _strict_int_env(gemma_name)
        if v is not None:
            return v
    return default


def _resolve_effective_n_ctx(
    *, policy_n_ctx: int, is_gemma: bool, context_window: int,
) -> int:
    """OTR_GGUF_N_CTX > GEMMA4_12B_N_CTX (gemma only) > policy. VALIDATE (never
    clamp): 512 <= n_ctx <= row.context_window, else RAISE."""
    n_ctx = int(policy_n_ctx)
    source = "policy.gguf_n_ctx"
    v = _strict_int_env("OTR_GGUF_N_CTX")
    if v is not None:
        n_ctx, source = v, "OTR_GGUF_N_CTX"
    elif is_gemma:
        v = _strict_int_env("GEMMA4_12B_N_CTX")
        if v is not None:
            n_ctx, source = v, "GEMMA4_12B_N_CTX"
    if not (512 <= n_ctx <= context_window):
        raise GGUFNativeConfigError(
            f"effective n_ctx {n_ctx} (from {source}) is outside "
            f"[512, {context_window}] for this row -- NO clamp; pick a valid "
            "n_ctx or a row with a larger context_window."
        )
    return n_ctx


def _resolve_gguf_seed() -> int:
    """The GGUF writer seed. ``OTR_GGUF_SEED`` PINS it (a uint32) for a byte-repro
    run; UNSET/blank draws a fresh OS-entropy uint32 -- matching OTR's creative-RNG
    model (fresh cast/style each run unless explicitly pinned, feedback_true_
    randomization). It is an env (NOT a widget) on purpose: a widget named "seed"
    trips the workflow validator's control_after_generate companion-slot rule.

    (2026-07-20, operator: the old hard "OTR_GGUF_SEED is REQUIRED" guard was
    removed as unneeded -- an unset seed defaults to fresh entropy rather than
    failing the writer. An explicitly-set-but-malformed value still fails LOUD.)"""
    raw = os.environ.get("OTR_GGUF_SEED")
    if raw is None or not raw.strip():
        return int.from_bytes(os.urandom(4), "big")   # fresh OS-entropy uint32
    try:
        val = int(raw.strip())
    except ValueError:
        raise GGUFNativeConfigError(
            f"OTR_GGUF_SEED must be a uint32 integer; got {raw!r}."
        ) from None
    if not (0 <= val <= 0xFFFFFFFF):
        raise GGUFNativeConfigError(
            f"OTR_GGUF_SEED must be in [0, 4294967295]; got {val}."
        )
    return val


#: seed derivation algorithm version stamped into the receipt.
GGUF_SEED_ALGO = "v1_base_plus_ordinal_u32"


@dataclasses.dataclass(frozen=True)
class GGUFLoadConfig:
    """The resolved, immutable per-slot GGUF load contract. Threaded from the
    preflight through request_slot / backend.load / the generate factory and
    serialized into the ledger receipt. NO consumer rebuilds it from live env.
    ``reuse_key`` is the resident-model cache identity (load-shaping fields
    only); sampling + seed are generate-time and excluded."""

    repo_id: str
    model_path: str
    quant: str
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    kv_gb_per_1k: float | None
    seed: int
    stop_tokens: tuple[str, ...]
    think_policy: str
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    seed_algo: str = GGUF_SEED_ALGO

    def reuse_key(self) -> tuple:
        return (
            self.repo_id, self.model_path, self.quant,
            self.n_ctx, self.n_batch, self.n_gpu_layers,
        )

    def sampling_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.top_p is not None:
            out["top_p"] = float(self.top_p)
        if self.min_p is not None:
            out["min_p"] = float(self.min_p)
        if self.repeat_penalty is not None:
            out["repeat_penalty"] = float(self.repeat_penalty)
        return out

    def with_sampling(
        self, *, top_p=None, min_p=None, repeat_penalty=None,
    ) -> "GGUFLoadConfig":
        return dataclasses.replace(
            self, top_p=top_p, min_p=min_p, repeat_penalty=repeat_penalty,
        )

    def as_receipt(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "model_path": self.model_path,
            "quant": self.quant,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "kv_gb_per_1k": self.kv_gb_per_1k,
            "seed": self.seed,
            "seed_algo": self.seed_algo,
            "top_k": GGUF_TOP_K,
            "sampling": self.sampling_dict(),
            "stop_tokens": list(self.stop_tokens),
            "think_policy": self.think_policy,
        }

    @classmethod
    def from_receipt(cls, receipt: dict[str, Any]) -> "GGUFLoadConfig":
        """Inverse of :meth:`as_receipt`: reconstruct the immutable load_config a
        downstream phase (freeze cascade / shot lock) needs from the ledger stamp
        ``meta['llm_gguf_load_config'][slot]`` -- so it loads under the EXACT
        contract the writer used, never a live-env rebuild."""
        sampling = receipt.get("sampling") or {}
        return cls(
            repo_id=str(receipt["repo_id"]),
            model_path=str(receipt["model_path"]),
            quant=str(receipt["quant"]),
            n_ctx=int(receipt["n_ctx"]),
            n_batch=int(receipt["n_batch"]),
            n_gpu_layers=int(receipt["n_gpu_layers"]),
            kv_gb_per_1k=(
                float(receipt["kv_gb_per_1k"])
                if receipt.get("kv_gb_per_1k") is not None else None
            ),
            seed=int(receipt["seed"]),
            stop_tokens=tuple(receipt.get("stop_tokens") or ()),
            think_policy=str(receipt["think_policy"]),
            top_p=sampling.get("top_p"),
            min_p=sampling.get("min_p"),
            repeat_penalty=sampling.get("repeat_penalty"),
            seed_algo=str(receipt.get("seed_algo") or GGUF_SEED_ALGO),
        )


def load_config_from_meta(meta: Any, slot: str) -> "GGUFLoadConfig | None":
    """Reconstruct the per-slot GGUFLoadConfig from a ledger ``meta`` stamp, or
    None when the run was not a GGUF run / the slot was not stamped. Lets a
    downstream phase thread the writer's exact load contract into request_slot
    without rebuilding from the live environment."""
    if not isinstance(meta, dict):
        return None
    table = meta.get("llm_gguf_load_config")
    if not isinstance(table, dict):
        return None
    receipt = table.get(slot)
    if not isinstance(receipt, dict) or not receipt:
        return None
    try:
        return GGUFLoadConfig.from_receipt(receipt)
    except (KeyError, TypeError, ValueError):
        return None


def build_gguf_load_config(
    *, repo_id: str, policy: Any, slot: str = "?", seed: int | None = None,
    top_p: float | None = None, min_p: float | None = None,
    repeat_penalty: float | None = None,
) -> GGUFLoadConfig:
    """Resolve the immutable load_config for a selected gguf ``repo_id`` under
    ``policy``. FAIL LOUD on an unknown quant (slot/row/available) and on a
    malformed override. ``seed`` defaults to the required OTR_GGUF_SEED env."""
    row = gguf_row_for_repo(repo_id)
    is_gemma = repo_id == ROW_ID
    quant = policy.gguf_quant
    try:
        _filename, _size, _sha = row.artifact_for_quant(quant)
    except GGUFNativeConfigError as exc:
        raise GGUFNativeConfigError(
            f"slot {slot!r}: {exc}"
        ) from None
    model_path = row_artifact_path(row, quant)
    n_ctx = _resolve_effective_n_ctx(
        policy_n_ctx=policy.gguf_n_ctx, is_gemma=is_gemma,
        context_window=row.context_window,
    )
    n_batch = _resolve_int_override(
        "OTR_GGUF_N_BATCH", "GEMMA4_12B_N_BATCH" if is_gemma else None,
        DEFAULT_N_BATCH,
    )
    default_layers = DEFAULT_N_GPU_LAYERS if policy.device == "cuda" else 0
    n_gpu_layers = _resolve_int_override(
        "OTR_GGUF_N_GPU_LAYERS", "GEMMA4_12B_N_GPU_LAYERS" if is_gemma else None,
        default_layers,
    )
    base_seed = _resolve_gguf_seed() if seed is None else int(seed)
    return GGUFLoadConfig(
        repo_id=repo_id, model_path=str(model_path), quant=quant,
        n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=n_gpu_layers,
        kv_gb_per_1k=row.kv_gb_per_1k, seed=base_seed,
        stop_tokens=row.stop_tokens, think_policy=row.think_policy,
        top_p=top_p, min_p=min_p, repeat_penalty=repeat_penalty,
    )


# ---------------------------------------------------------------------------
# Readiness validation (row path + pinned size/sha, memoized hashing)
# ---------------------------------------------------------------------------

_SHA_MEMO: dict[tuple[str, int, int], str] = {}


def _memoized_sha256(path: Path, size: int, mtime_ns: int) -> str:
    key = (str(path), int(size), int(mtime_ns))
    cached = _SHA_MEMO.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_MEMO[key] = digest
    return digest


def validate_gguf_ready(repo_id: str, quant: str, load_config: Any) -> None:
    """Fail-loud readiness check for a resolved (repo_id, quant, load_config).
    Wraps exists/stat/permission as GGUFNativeConfigError; verifies a pinned
    size + sha when present (existence-only when None); memoizes hashing by
    (resolved path, size, mtime_ns) so a post-replace file re-hashes."""
    row = gguf_row_for_repo(repo_id)
    _filename, expected_size, expected_sha = row.artifact_for_quant(quant)
    path = Path(load_config.model_path)
    try:
        exists = path.exists()
    except OSError as exc:
        raise GGUFNativeConfigError(
            f"cannot stat GGUF path {path} for {repo_id!r}/{quant}: {exc}"
        ) from exc
    if not exists:
        raise GGUFNativeConfigError(
            f"missing {repo_id!r} {quant} GGUF file: {path}."
        )
    try:
        st = path.stat()
    except OSError as exc:
        raise GGUFNativeConfigError(
            f"cannot stat GGUF file {path}: {exc}"
        ) from exc
    if not path.is_file() or st.st_size == 0:
        raise GGUFNativeConfigError(
            f"{repo_id!r} {quant} GGUF path is not a regular non-empty file: "
            f"{path}."
        )
    if expected_size is not None and st.st_size != expected_size:
        raise GGUFNativeConfigError(
            f"incomplete {repo_id!r} {quant} GGUF file: {path} has "
            f"{st.st_size} bytes, expected {expected_size}."
        )
    if expected_sha is not None:
        actual = _memoized_sha256(path, st.st_size, st.st_mtime_ns)
        if actual != expected_sha:
            raise GGUFNativeConfigError(
                f"SHA256 mismatch for {repo_id!r} {quant} at {path}: got "
                f"{actual}, expected {expected_sha}."
            )


# ---------------------------------------------------------------------------
# think-strip + stop merge helpers
# ---------------------------------------------------------------------------

_LEADING_THINK_RE = re.compile(
    r"^\s*<think\b[^>]*>.*?</think\s*>\s*", re.DOTALL | re.IGNORECASE)
#: A leading <think> whose close was clipped -- a caller stop string can cut
#: Qwen3's empty `<think>\n\n</think>` wrapper before `</think>`, leaving an
#: unstrippable, unclosed `<think>`. From that open to end-of-string is all
#: (clipped) wrapper.
_LEADING_DANGLING_THINK_RE = re.compile(r"^\s*<think\b.*\Z", re.DOTALL | re.IGNORECASE)


def _strip_leading_think_envelope(text: str) -> str:
    """Strip ONE leading <think> reasoning wrapper, preserving any LATER literal
    <think> in the body (anchored at the start, non-greedy).

    Handles a COMPLETE leading ``<think>...</think>`` envelope AND a DANGLING
    leading ``<think>...``(clipped, no close). The dangling case appears when a
    caller stop string cuts Qwen3's empty ``<think>\\n\\n</think>`` wrapper before
    the closing tag."""
    if not isinstance(text, str):
        return text
    new = _LEADING_THINK_RE.sub("", text, count=1)
    if new != text:
        return new
    # No complete leading envelope: if the text STARTS with an unclosed <think>
    # open, the whole reply is a clipped wrapper -> drop it (a later literal
    # <think> that is NOT at the start is left untouched by the anchor).
    return _LEADING_DANGLING_THINK_RE.sub("", text, count=1)


#: Qwen3 soft-switch appended to non-structured prompts under the
#: "qwen3_no_think" policy. Without it Qwen3 opens a long <think> reasoning
#: block, so a short-output call (rank/rerank, max_tokens 8-64) exhausts its
#: whole budget mid-thought and never emits the answer. `/no_think` makes the
#: model skip reasoning and answer directly (it still emits an empty
#: <think></think> that _strip_leading_think_envelope removes).
_QWEN3_NO_THINK_DIRECTIVE = "/no_think"


def _apply_qwen3_no_think(messages: Any) -> list:
    """Return a NEW messages list with the Qwen3 `/no_think` soft-switch applied.

    Appends the directive to the LAST user message with string content. If none
    exists, prepends a system directive (Qwen3 honors it there too). Idempotent:
    never appends the directive twice."""
    try:
        out = [dict(m) if isinstance(m, dict) else m for m in messages]
    except TypeError:
        return messages
    for m in reversed(out):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                if _QWEN3_NO_THINK_DIRECTIVE in content:
                    return out
                stripped = content.rstrip()
                m["content"] = (
                    f"{stripped} {_QWEN3_NO_THINK_DIRECTIVE}"
                    if stripped else _QWEN3_NO_THINK_DIRECTIVE
                )
                return out
            break
    out.insert(0, {"role": "system", "content": _QWEN3_NO_THINK_DIRECTIVE})
    return out


def _merge_stop_tokens(row_stops: Any, caller_stop: Any) -> list[str]:
    """Ordered, de-duplicated union of the row's stop_tokens and the caller's
    per-call stop list (row first)."""
    merged: list[str] = []
    for source in (row_stops or (), caller_stop or ()):
        for s in source:
            if s and s not in merged:
                merged.append(s)
    return merged


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _models_root() -> Path:
    raw = (
        os.environ.get("OTR_COMFYUI_MODELS_ROOT")
        or os.environ.get("COMFYUI_MODELS_ROOT")
        or r"C:\ComfyUI-Models"
    )
    return Path(raw).expanduser()


def default_gguf_path(quant: str = "Q8_0") -> Path:
    filename, _size, _sha = gguf_artifact_for_quant(quant)
    return (
        _models_root()
        / "LLM"
        / "converted"
        / "gemma-4-12b-it"
        / filename
    )


def gguf_artifact_for_quant(quant: str) -> tuple[str, int | None, str | None]:
    """(filename, expected_size, expected_sha256) for a policy gguf_quant.
    Unknown quants FAIL LOUD -- the policy enum and this table must agree.
    The sha rides along so the env-fallback path gets the SAME integrity
    contract as the registry path; it used to hard-code None there (A6)."""
    try:
        return GGUF_ARTIFACTS[quant]
    except KeyError:
        raise GGUFNativeConfigError(
            f"gguf_quant {quant!r} has no artifact-table entry "
            f"(known: {sorted(GGUF_ARTIFACTS)}). Add the filename + size "
            "to GGUF_ARTIFACTS -- no guessed filenames."
        ) from None


def resolve_gguf_path(quant: str = "Q8_0") -> Path:
    raw = os.environ.get("GEMMA4_12B_GGUF_PATH")
    return Path(raw).expanduser() if raw else default_gguf_path(quant)


def _site_packages_candidates() -> list[Path]:
    candidates: list[Path] = []
    venv_root = Path(sys.executable).resolve().parents[1]
    candidates.append(venv_root / "Lib" / "site-packages")
    candidates.extend(
        Path(p)
        for p in sys.path
        if p and "site-packages" in p.replace("\\", "/")
    )
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _prepare_windows_llama_dll_runtime() -> None:
    """Preload llama-cpp CUDA DLLs whose dependencies live in pip packages."""
    global _DLL_RUNTIME_PREPARED
    if _DLL_RUNTIME_PREPARED or os.name != "nt":
        return
    _DLL_RUNTIME_PREPARED = True
    import ctypes  # imported lazily so non-Windows test imports stay light

    log.info("[GGUFNative] Preparing Windows DLL search paths and preloading CUDA runtime dependencies...")
    for site_packages in _site_packages_candidates():
        dll_dirs = [
            site_packages / "nvidia" / "cuda_runtime" / "bin",
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "llama_cpp" / "lib",
            site_packages / "torch" / "lib",
        ]
        for dll_dir in dll_dirs:
            if dll_dir.exists() and hasattr(os, "add_dll_directory"):
                log.info("[GGUFNative] Adding DLL directory to search path: %s", dll_dir)
                try:
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
                except Exception as exc:
                    log.warning("[GGUFNative] Failed to add DLL directory %s: %s", dll_dir, exc)
        preload_names = [
            site_packages / "nvidia" / "cuda_runtime" / "bin" / "cudart64_12.dll",
            site_packages / "nvidia" / "cublas" / "bin" / "cublas64_12.dll",
            site_packages / "nvidia" / "cublas" / "bin" / "cublasLt64_12.dll",
            site_packages / "llama_cpp" / "lib" / "ggml-base.dll",
            site_packages / "llama_cpp" / "lib" / "ggml-cpu.dll",
            site_packages / "llama_cpp" / "lib" / "ggml-cuda.dll",
            site_packages / "llama_cpp" / "lib" / "ggml.dll",
            site_packages / "llama_cpp" / "lib" / "llama.dll",
        ]
        for dll_path in preload_names:
            if dll_path.exists():
                log.info("[GGUFNative] Preloading dependency DLL: %s", dll_path.name)
                try:
                    _PRELOADED_DLLS.append(ctypes.WinDLL(str(dll_path)))
                except Exception as exc:
                    log.error("[GGUFNative] Failed to preload DLL %s: %s", dll_path, exc)
                    raise exc


def _import_llama_cpp():
    try:
        _prepare_windows_llama_dll_runtime()
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise GGUFNativeConfigError(
            "llama-cpp-python is not importable in the ComfyUI venv. "
            "Install a CUDA-enabled llama-cpp-python build for this Python "
            "environment before selecting unsloth/gemma-4-12b-it-GGUF. "
            "On Windows CUDA wheels also require importable CUDA 12 runtime "
            "DLLs such as nvidia-cuda-runtime-cu12 and nvidia-cublas-cu12."
        ) from exc
    return Llama


def _load_llama(
    *,
    model_path: Path,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
    verbose: bool,
) -> Any:
    Llama = _import_llama_cpp()
    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=verbose,
    )


def validate_gemma_gguf_ready(*, require_binding: bool = True) -> dict[str, Any]:
    """Return a side-effect-free readiness report for the native GGUF lane."""
    path = resolve_gguf_path()
    out: dict[str, Any] = {
        "ok": False,
        "row_id": ROW_ID,
        "provider": PROVIDER,
        "model_path": str(path),
        "model_exists": path.exists(),
        "model_size": path.stat().st_size if path.exists() else 0,
        "expected_size": EXPECTED_Q8_0_SIZE_BYTES,
        "binding_available": None,
        "error": "",
    }
    if not path.exists():
        out["error"] = f"missing GGUF file: {path}"
        return out
    if path.name == DEFAULT_GGUF_FILENAME and path.stat().st_size != EXPECTED_Q8_0_SIZE_BYTES:
        out["error"] = (
            f"incomplete GGUF file: {path} has {path.stat().st_size} bytes, "
            f"expected {EXPECTED_Q8_0_SIZE_BYTES}"
        )
        return out
    if require_binding:
        try:
            _import_llama_cpp()
            out["binding_available"] = True
        except GGUFNativeConfigError as exc:
            out["binding_available"] = False
            out["error"] = str(exc)
            return out
    out["ok"] = True
    return out


def _llamacpp_response_format(response_format: dict | None) -> dict | None:
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return response_format
    if kind == "json_schema":
        schema_box = response_format.get("json_schema") or {}
        schema = schema_box.get("schema") if isinstance(schema_box, dict) else None
        if not isinstance(schema, dict):
            raise GGUFNativeConfigError(
                "json_schema response_format is missing a schema object."
            )
        return {"type": "json_object", "schema": schema}
    return response_format


class GGUFNativeBackend:
    """LoaderBackend adapter for in-process llama-cpp-python GGUF inference."""

    def load(
        self, repo_id: str, row: Any, policy: Any = None,
        load_config: Any = None,
    ) -> dict[str, Any]:
        # S1 platform-portability (2026-07-10): resolve the explicit policy
        # (None = nv50 baseline: cuda / Q8_0 / n_ctx 4096 -- identical to
        # the previous hardcoded behavior).
        from ._otr_shared.llm_policy import BASELINE_POLICY
        _policy = policy if policy is not None else BASELINE_POLICY

        # GGUF row registry (2026-07-16): when the preflight threads an
        # immutable load_config, every effective value comes FROM it -- no
        # live-env rebuild. When absent (direct/API callers, legacy tests) the
        # env-fallback path below reproduces the pre-registry behavior exactly.
        _registry_row = None
        try:
            _registry_row = gguf_row_for_repo(repo_id)
        except GGUFNativeConfigError:
            _registry_row = None
        if load_config is not None:
            quant = load_config.quant
            model_path = Path(load_config.model_path)
            _lc_row = _registry_row or gguf_row_for_repo(repo_id)
            expected_name, expected_size, expected_sha = _lc_row.artifact_for_quant(quant)
            eff_n_ctx = int(load_config.n_ctx)
            eff_n_batch = int(load_config.n_batch)
            eff_n_gpu_layers = int(load_config.n_gpu_layers)
            eff_kv_rate = (
                float(load_config.kv_gb_per_1k)
                if load_config.kv_gb_per_1k is not None else KV_GB_PER_1K_CTX
            )
            think_policy = load_config.think_policy
            stop_tokens = tuple(load_config.stop_tokens)
            gguf_seed = int(load_config.seed)
        else:
            # Operator directive (2026-07-16, "no local-LM fallbacks"): the
            # env-fallback below is GEMMA-ONLY -- gguf_artifact_for_quant /
            # resolve_gguf_path resolve the gemma artifact regardless of
            # repo_id. A non-gemma row that reaches here without a threaded
            # load_config would SILENTLY load the gemma file (a local-LM / path
            # fallback). Fail LOUD instead: a non-gemma row MUST arrive with an
            # immutable load_config (reconstruct it from the ledger receipt
            # meta['llm_gguf_load_config'] in the calling phase).
            if repo_id != ROW_ID:
                raise GGUFNativeConfigError(
                    f"GGUF backend reached the gemma-only env-fallback for "
                    f"repo_id {repo_id!r} with no threaded load_config; that "
                    f"path resolves the gemma artifact only and would silently "
                    f"load gemma for a non-gemma row. Thread the immutable "
                    f"load_config (reconstruct from "
                    f"meta['llm_gguf_load_config']) into this caller "
                    f"(operator directive: no silent local-LM/path fallback)."
                )
            quant = _policy.gguf_quant
            expected_name, expected_size, expected_sha = (
                gguf_artifact_for_quant(quant)
            )
            model_path = resolve_gguf_path(quant)
            eff_n_ctx = _int_env("GEMMA4_12B_N_CTX", _policy.gguf_n_ctx)
            eff_n_batch = _int_env("GEMMA4_12B_N_BATCH", DEFAULT_N_BATCH)
            _default_layers = DEFAULT_N_GPU_LAYERS if _policy.device == "cuda" else 0
            eff_n_gpu_layers = _int_env("GEMMA4_12B_N_GPU_LAYERS", _default_layers)
            eff_kv_rate = _float_env("GEMMA4_12B_KV_GB_PER_1K", KV_GB_PER_1K_CTX)
            think_policy = _registry_row.think_policy if _registry_row else "none"
            stop_tokens = _registry_row.stop_tokens if _registry_row else ()
            gguf_seed = None
        # Keep the gemma row's human label so its error UX (and the tests
        # pinning it) stay identical; other rows report their repo id.
        _label = "Gemma 4 12B" if repo_id == ROW_ID else repo_id
        if not model_path.exists():
            raise GGUFNativeConfigError(
                f"Missing {_label} {quant} GGUF file: {model_path}. "
                f"Download/convert {expected_name} and place it under "
                "C:\\ComfyUI-Models\\LLM\\converted\\, or set "
                "GEMMA4_12B_GGUF_PATH (gemma row only)."
                + (f" Expected size: {expected_size} bytes."
                   if expected_size else "")
            )
        # A6 (2026-07-27): an artifact with NO pinned integrity is refused,
        # not accepted unchecked. Both checks below are conditional on their
        # pin existing, so an unpinned quant skipped integrity entirely and a
        # partial file was reported ready. Scoped to the artifact this table
        # names: a GEMMA4_12B_GGUF_PATH override deliberately points at a file
        # the table does not describe, and that override is an explicit
        # operator act with its own name on it.
        if model_path.name == expected_name:
            _unpinned = [
                _field for _field, _value in (
                    ("size", expected_size), ("sha256", expected_sha),
                ) if _value is None
            ]
            if _unpinned:
                raise GGUFNativeConfigError(
                    f"Unpinned {_label} {quant} artifact: {expected_name} has "
                    f"no {' and no '.join(_unpinned)} in GGUF_ARTIFACTS, so a "
                    f"truncated or substituted file cannot be detected and "
                    f"this load would be unverified. Derive both from a "
                    f"known-good copy (its Length and Get-FileHash -Algorithm "
                    f"SHA256) and pin them -- never from a guess or a model "
                    f"card."
                )

        if (
            model_path.name == expected_name
            and expected_size is not None
            and model_path.stat().st_size != expected_size
        ):
            raise GGUFNativeConfigError(
                f"Incomplete {_label} {quant} GGUF file: {model_path} "
                f"has {model_path.stat().st_size} bytes, expected "
                f"{expected_size}. Let the download finish or "
                "delete the partial file and retry."
            )
        if expected_sha is not None and model_path.name == expected_name:
            _st = model_path.stat()
            _actual_sha = _memoized_sha256(model_path, _st.st_size, _st.st_mtime_ns)
            if _actual_sha != expected_sha:
                raise GGUFNativeConfigError(
                    f"SHA256 mismatch for {repo_id} {quant} at {model_path}: "
                    f"got {_actual_sha}, expected {expected_sha}."
                )

        # 1. Nuclear Power Wash: Evict ComfyUI models & empty PyTorch CUDA cache
        log.info("[GGUFNative] Running pre-load VRAM eviction to clean resident PyTorch models...")
        try:
            from ._otr_vram_levers import free_otr_pipeline_residue
            free_otr_pipeline_residue(reason="GGUFNative load preflight")
        except Exception as exc:
            log.warning("[GGUFNative] VRAM eviction failed (non-fatal): %s", exc)

        # Effective n_ctx resolved above (load_config when threaded, else the
        # env>policy fallback). The policy default (4096) equals the old
        # row-derived default.
        n_ctx = eff_n_ctx
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )

        # 2. VRAM Gate Preflight Check. S1: FAIL LOUD, never adapt --
        # the old silent 4096->2048 downgrade truncated the
        # original_concept JSON mid-generation (root cause behind
        # d526c8b7); the old "preflight failed (proceeding anyway)"
        # tolerance loaded blind. Both are raises now. A cpu-device
        # policy skips the gate outright (no VRAM to fit).
        import torch
        if (_policy.device == "cuda" and torch.cuda.is_available()
                and not _bool_env("OTR_TEST_MODE", False)):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception as exc:
                raise GGUFNativeConfigError(
                    f"VRAM preflight probe failed ({exc!r}) -- refusing to "
                    "load a ~13 GB GGUF blind. Fix the CUDA runtime or set "
                    "llm device policy to cpu."
                ) from exc
            free_gb = free_bytes / (1024 ** 3)
            # Estimation: weights + SWA KV cache (~0.7 GB per 1024 context cells)
            # + safety overhead.
            #
            # The weight term MUST come from the file on disk. It was hardcoded to
            # 12.07 GB -- the size of the Q8_0 -- so the gate refused every OTHER
            # quant by pricing it as a Q8_0. A 7.12 GB Q4_K_M that fits comfortably
            # would be rejected for needing VRAM it does not use. The quant is the
            # one lever we have when a model will not fit; a gate that cannot see
            # the quant cannot be reasoned with.
            weights_gb = model_path.stat().st_size / (1024 ** 3)
            # Per-row KV rate (load_config-threaded or the env>default fallback
            # resolved above). NO global _KV_GB_PER_1K -- KV is per row.
            kv_rate = eff_kv_rate
            kv_gb = (n_ctx / 1024.0) * kv_rate
            estimated_needed_gb = weights_gb + kv_gb + 0.1
            log.info(
                "[GGUFNative] VRAM Preflight: Free=%.2f GB | Needed=%.2f GB "
                "(weights=%.2f from %s, kv=%.2f @ n_ctx=%d)",
                free_gb, estimated_needed_gb, weights_gb, model_path.name,
                kv_gb, n_ctx,
            )
            if free_gb < estimated_needed_gb:
                raise GGUFNativeConfigError(
                    f"Insufficient VRAM for GGUF n_ctx={n_ctx}: free "
                    f"{free_gb:.2f} GB < needed {estimated_needed_gb:.2f} "
                    f"GB. Lower gguf_n_ctx (policy), free VRAM, or pick a "
                    "smaller quant. NO silent context downgrade (the old "
                    "4096->2048 downgrade truncated generations)."
                )

        # Effective n_batch / n_gpu_layers resolved above (load_config when
        # threaded, else the whitelisted env>default fallback). Device policy:
        # cuda offloads all layers (-1); cpu keeps every layer on host (0).
        n_batch = eff_n_batch
        n_gpu_layers = eff_n_gpu_layers
        verbose = _bool_env(
            "OTR_GGUF_VERBOSE", _bool_env("GEMMA4_12B_VERBOSE", False)
        )

        log.info(
            "[GGUFNative] Initializing llama.cpp Llama instance: n_ctx=%d, n_gpu_layers=%d, n_batch=%d",
            n_ctx, n_gpu_layers, n_batch
        )
        model = _load_llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=verbose,
        )
        cache_entry = {
            "provider": PROVIDER,
            "model_id": repo_id,
            "model": model,
            "model_path": str(model_path),
            "quant": quant,
            "context_cap": n_ctx,
            "context_window": context_window,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            # Stamped at load so generate() honors per-row behavior without
            # re-reading env: stop-token merge, think-strip, per-call seed base.
            "stop_tokens": tuple(stop_tokens),
            "think_policy": think_policy,
            "gguf_seed": gguf_seed,
        }
        log.info(
            "[GGUFNative] load %s -> %s ctx=%d gpu_layers=%d batch=%d",
            cache_entry["model_id"],
            model_path,
            n_ctx,
            n_gpu_layers,
            n_batch,
        )
        return cache_entry

    def generate(
        self,
        model: Any,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        stop: Any = None,
        response_format: dict | None = None,
        grammar: str | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        seed: int | None = None,
        **_ignored: Any,
    ) -> str:
        require_full_output = bool(getattr(
            messages, "_otr_require_full_output_budget", False,
        ))
        reserve_remaining = bool(getattr(
            messages, "_otr_reserve_remaining_output_capacity", False,
        ))
        bounded_capacity = reserve_remaining and max_new_tokens is not None
        fail_on_output_limit = bool(getattr(
            messages, "_otr_fail_on_output_limit", False,
        ))
        if grammar:
            raise GGUFNativeConfigError(
                "The native GGUF lane does not accept raw GBNF `grammar`; "
                "use JSON-schema `response_format`."
            )
        llm = model.get("model") if isinstance(model, dict) else model
        if llm is None:
            raise GGUFNativeConfigError("cache_entry is missing the GGUF model.")
        cap = _int_env("GEMMA4_12B_MAX_NEW_TOKENS", DEFAULT_OUTPUT_TOKENS_CAP)
        requested_tokens = int(max_new_tokens or cap)
        if (require_full_output or bounded_capacity) and requested_tokens > cap:
            capacity = GenerationContextOverflowError(
                f"GGUF {ROW_ID} cannot fit the complete requested output: "
                f"requested_output={requested_tokens}, provider_output_cap={cap}"
            )
            raise GGUFNativeConfigError(str(capacity)) from capacity
        out_tokens = cap if reserve_remaining else requested_tokens
        if out_tokens > cap:
            log.warning(
                "[GGUFNative] output token request capped: requested=%d "
                "effective=%d via GEMMA4_12B_MAX_NEW_TOKENS",
                out_tokens, cap,
            )
            out_tokens = cap
        try:
            fitted_tokens = fit_output_tokens(
                out_tokens,
                context_cap=int(
                    (model.get("context_cap") if isinstance(model, dict) else 0)
                    or DEFAULT_CONTEXT_WINDOW
                ),
                prompt_tokens=estimate_prompt_tokens(messages),
                min_output_tokens=min(64, cap),
                label=f"GGUF {ROW_ID}",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            raise GGUFNativeConfigError(str(exc)) from exc
        if fitted_tokens != out_tokens:
            log.warning(
                "[GGUFNative] output budget clamped: requested=%d effective=%d",
                out_tokens, fitted_tokens,
            )
        out_tokens = fitted_tokens
        # Qwen3 reasoning suppression (root fix 2026-07-16): a "qwen3_no_think"
        # row must be told to SKIP its <think> block on EVERY call. Non-
        # structured short-output calls (rank/rerank, max_tokens 8-64) otherwise
        # spend their whole budget mid-thought and never emit the answer;
        # structured (json_object) calls otherwise emit a degenerate empty "{}"
        # because the JSON grammar blocks the <think> the model still opens
        # (casting DescriptionResponse came back as {} -> pydantic "Field
        # required"). `/no_think` makes Qwen answer/fill directly in both modes.
        rf = _llamacpp_response_format(response_format)
        _think_policy = (
            model.get("think_policy") if isinstance(model, dict) else None
        )
        if _think_policy == "qwen3_no_think":
            messages = _apply_qwen3_no_think(messages)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": out_tokens,
            "temperature": float(temperature if temperature is not None else 0.7),
            # top_k pinned to the registry constant (also llama-cpp's own
            # default, so the pin is byte-identical for the top_k dimension).
            "top_k": GGUF_TOP_K,
        }
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if min_p is not None:
            kwargs["min_p"] = float(min_p)
        if repeat_penalty is not None:
            kwargs["repeat_penalty"] = float(repeat_penalty)
        if seed is not None:
            kwargs["seed"] = int(seed)
        row_stops = model.get("stop_tokens") if isinstance(model, dict) else ()
        merged_stop = _merge_stop_tokens(row_stops, stop)
        if _think_policy == "qwen3_no_think":
            # Qwen3 emits an empty `<think>\n\n</think>` wrapper even under
            # /no_think. A pure-whitespace stop (e.g. "\n\n") would fire INSIDE
            # that wrapper and clip the reply to a dangling "<think>" before the
            # answer is generated. Drop whitespace-only stops for this row;
            # content stops (e.g. "\n[", "\n(") still delimit the answer.
            merged_stop = [s for s in merged_stop if s.strip()]
        if merged_stop:
            kwargs["stop"] = merged_stop
        if rf is not None:
            kwargs["response_format"] = rf
        _row_id = model.get("model_id") if isinstance(model, dict) else None
        try:
            result = llm.create_chat_completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise GGUFNativeCallFailedError(
                f"Native GGUF call failed for {_row_id or ROW_ID}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        text = self._extract_text(
            result, fail_on_output_limit=fail_on_output_limit,
        )
        # think-strip: a qwen3 row emits a leading <think>...</think> envelope on
        # free-form calls (structured passes carry rf and are JSON-constrained).
        # The strip also drops a DANGLING/clipped leading <think>. If nothing but
        # a wrapper was produced, FAIL LOUD -- never return an empty reply
        # (operator directive: no local-LM fallbacks).
        if rf is None and _think_policy == "qwen3_no_think":
            stripped = _strip_leading_think_envelope(text)
            if not stripped.strip():
                raise GGUFNativeCallFailedError(
                    f"Native GGUF (qwen3_no_think) returned only a reasoning "
                    f"wrapper, no answer (raw head: {text[:80]!r}). No fallback "
                    f"(operator directive: no local-LM fallbacks)."
                )
            text = stripped
        return text

    def unload(self, model: Any) -> None:
        llm = model.get("model") if isinstance(model, dict) else model
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                log.debug("[GGUFNative] close() failed: %s", exc)

    @staticmethod
    def _extract_text(
        result: dict, *, fail_on_output_limit: bool = False,
    ) -> str:
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            raise GGUFNativeCallFailedError(
                f"Native GGUF response had no choices: {str(result)[:300]}"
            )
        first = choices[0]
        if first.get("finish_reason") == "length" and fail_on_output_limit:
            raise GGUFNativeCallFailedError(
                "Native GGUF generation exhausted the provider output capacity; "
                "the partial artifact is not eligible for reroll"
            )
        content = (first.get("message") or {}).get("content")
        if content is None and isinstance(first.get("text"), str):
            content = first.get("text")
        if not isinstance(content, str) or not content:
            raise GGUFNativeCallFailedError(
                "Native GGUF model returned empty message content "
                f"(finish_reason={first.get('finish_reason')!r})."
            )
        return content


def make_gguf_generate_fn(
    cache_entry: dict, *, response_format: dict | None = None,
    sampling: dict | None = None, seed: int | None = None,
):
    backend = GGUFNativeBackend()
    bound_rf = response_format
    _s = sampling or {}
    bound_top_p = _s.get("top_p")
    bound_min_p = _s.get("min_p")
    bound_repeat = _s.get("repeat_penalty")
    # Base seed: an explicit arg wins, else the value stamped into cache_entry
    # at load (from the preflight load_config). None => llama-cpp picks its own.
    base_seed = seed
    if base_seed is None and isinstance(cache_entry, dict):
        base_seed = cache_entry.get("gguf_seed")
    _ordinal = {"n": 0}

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None, grammar=None):
        rf = response_format if response_format is not None else bound_rf
        per_call_seed = None
        if base_seed is not None:
            # Deterministic per-call derivation (GGUF_SEED_ALGO): base seed +
            # monotonic call ordinal, wrapped into uint32.
            per_call_seed = (int(base_seed) + _ordinal["n"]) & 0xFFFFFFFF
        _ordinal["n"] += 1
        return backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
            grammar=grammar,
            top_p=bound_top_p,
            min_p=bound_min_p,
            repeat_penalty=bound_repeat,
            seed=per_call_seed,
        )

    generate_fn._otr_gguf_native = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    generate_fn._otr_supports_json_object = True  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "GGUF_BACKEND_KEY",
    "PROVIDER",
    "ROW_ID",
    "DEFAULT_GGUF_FILENAME",
    "EXPECTED_Q8_0_SIZE_BYTES",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_OUTPUT_TOKENS_CAP",
    "GGUF_TOP_K",
    "GGUF_SEED_ALGO",
    "GGUFNativeError",
    "GGUFNativeConfigError",
    "GGUFNativeCallFailedError",
    "GGUFRegistryError",
    "GGUFRow",
    "GGUF_ROWS",
    "GGUF_ROWS_BY_REPO",
    "GGUFLoadConfig",
    "GGUFNativeBackend",
    "gguf_row_for_repo",
    "row_artifact_path",
    "gguf_native_row_on_disk",
    "build_gguf_load_config",
    "validate_gguf_ready",
    "default_gguf_path",
    "resolve_gguf_path",
    "validate_gemma_gguf_ready",
    "make_gguf_generate_fn",
]
