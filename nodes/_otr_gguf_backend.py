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

import logging
import os
import sys
from pathlib import Path
from typing import Any

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

# S1 platform-portability (2026-07-10): quant -> (filename, expected size)
# artifact table. Known quants FAIL LOUD on a byte-size mismatch (truncated
# or wrong file); entries whose size is None are absence-checked only (the
# box has never carried them -- pin the size when one is first derived).
# GEMMA4_12B_GGUF_PATH remains the explicit whole-path escape hatch.
GGUF_ARTIFACTS: dict[str, tuple[str, int | None]] = {
    "Q8_0": ("gemma-4-12b-it-Q8_0.gguf", EXPECTED_Q8_0_SIZE_BYTES),
    "Q6_K": ("gemma-4-12b-it-Q6_K.gguf", None),
    "Q4_K_M": ("gemma-4-12b-it-Q4_K_M.gguf", None),
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
    filename, _size = gguf_artifact_for_quant(quant)
    return (
        _models_root()
        / "LLM"
        / "converted"
        / "gemma-4-12b-it"
        / filename
    )


def gguf_artifact_for_quant(quant: str) -> tuple[str, int | None]:
    """(filename, expected_size) for a policy gguf_quant. Unknown quants
    FAIL LOUD -- the policy enum and this table must agree."""
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

    def load(self, repo_id: str, row: Any, policy: Any = None) -> dict[str, Any]:
        # S1 platform-portability (2026-07-10): resolve the explicit policy
        # (None = nv50 baseline: cuda / Q8_0 / n_ctx 4096 -- identical to
        # the previous hardcoded behavior).
        from ._otr_shared.llm_policy import BASELINE_POLICY
        _policy = policy if policy is not None else BASELINE_POLICY

        quant = _policy.gguf_quant
        expected_name, expected_size = gguf_artifact_for_quant(quant)
        model_path = resolve_gguf_path(quant)
        if not model_path.exists():
            raise GGUFNativeConfigError(
                f"Missing Gemma 4 12B {quant} GGUF file: {model_path}. "
                f"Download/convert {expected_name} from {ROW_ID} and place "
                "it under C:\\ComfyUI-Models\\LLM\\converted\\"
                "gemma-4-12b-it\\, or set GEMMA4_12B_GGUF_PATH."
                + (f" Expected size: {expected_size} bytes."
                   if expected_size else "")
            )
        if (
            model_path.name == expected_name
            and expected_size is not None
            and model_path.stat().st_size != expected_size
        ):
            raise GGUFNativeConfigError(
                f"Incomplete Gemma 4 12B {quant} GGUF file: {model_path} "
                f"has {model_path.stat().st_size} bytes, expected "
                f"{expected_size}. Let the download finish or "
                "delete the partial file and retry."
            )

        # 1. Nuclear Power Wash: Evict ComfyUI models & empty PyTorch CUDA cache
        log.info("[GGUFNative] Running pre-load VRAM eviction to clean resident PyTorch models...")
        try:
            from ._otr_vram_levers import free_otr_pipeline_residue
            free_otr_pipeline_residue(reason="GGUFNative load preflight")
        except Exception as exc:
            log.warning("[GGUFNative] VRAM eviction failed (non-fatal): %s", exc)

        # n_ctx precedence: explicit env escape hatch > policy. The policy
        # default (4096) equals the old row-derived default.
        n_ctx = _int_env("GEMMA4_12B_N_CTX", _policy.gguf_n_ctx)
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
            kv_rate = _float_env("GEMMA4_12B_KV_GB_PER_1K", KV_GB_PER_1K_CTX)
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

        n_batch = _int_env("GEMMA4_12B_N_BATCH", DEFAULT_N_BATCH)
        # Device policy: cuda offloads all layers (-1); cpu keeps every
        # layer on host (0). The env stays the explicit escape hatch.
        _default_layers = DEFAULT_N_GPU_LAYERS if _policy.device == "cuda" else 0
        n_gpu_layers = _int_env("GEMMA4_12B_N_GPU_LAYERS", _default_layers)
        verbose = _bool_env("GEMMA4_12B_VERBOSE", False)

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
            "context_cap": n_ctx,
            "context_window": context_window,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
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
        **_ignored: Any,
    ) -> str:
        if grammar:
            raise GGUFNativeConfigError(
                "The native GGUF lane does not accept raw GBNF `grammar`; "
                "use JSON-schema `response_format`."
            )
        llm = model.get("model") if isinstance(model, dict) else model
        if llm is None:
            raise GGUFNativeConfigError("cache_entry is missing the GGUF model.")
        cap = _int_env("GEMMA4_12B_MAX_NEW_TOKENS", DEFAULT_OUTPUT_TOKENS_CAP)
        out_tokens = int(max_new_tokens or cap)
        if out_tokens > cap:
            log.warning(
                "[GGUFNative] output token request capped: requested=%d "
                "effective=%d via GEMMA4_12B_MAX_NEW_TOKENS",
                out_tokens, cap,
            )
            out_tokens = cap
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": out_tokens,
            "temperature": float(temperature if temperature is not None else 0.7),
        }
        if stop:
            kwargs["stop"] = [s for s in stop if s]
        rf = _llamacpp_response_format(response_format)
        if rf is not None:
            kwargs["response_format"] = rf
        try:
            result = llm.create_chat_completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise GGUFNativeCallFailedError(
                f"Native GGUF call failed for {ROW_ID}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._extract_text(result)

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
    def _extract_text(result: dict) -> str:
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            raise GGUFNativeCallFailedError(
                f"Native GGUF response had no choices: {str(result)[:300]}"
            )
        first = choices[0]
        content = (first.get("message") or {}).get("content")
        if content is None and isinstance(first.get("text"), str):
            content = first.get("text")
        if not isinstance(content, str) or not content:
            raise GGUFNativeCallFailedError(
                "Native GGUF model returned empty message content "
                f"(finish_reason={first.get('finish_reason')!r})."
            )
        return content


def make_gguf_generate_fn(cache_entry: dict, *, response_format: dict | None = None):
    backend = GGUFNativeBackend()
    bound_rf = response_format

    def generate_fn(messages, *, temperature=None, max_new_tokens=None,
                    stop=None, response_format=None, grammar=None):
        rf = response_format if response_format is not None else bound_rf
        return backend.generate(
            cache_entry,
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop=stop,
            response_format=rf,
            grammar=grammar,
        )

    generate_fn._otr_gguf_native = True  # type: ignore[attr-defined]
    generate_fn._otr_response_format = bound_rf  # type: ignore[attr-defined]
    return generate_fn


__all__ = [
    "GGUF_BACKEND_KEY",
    "PROVIDER",
    "ROW_ID",
    "DEFAULT_GGUF_FILENAME",
    "EXPECTED_Q8_0_SIZE_BYTES",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_OUTPUT_TOKENS_CAP",
    "GGUFNativeError",
    "GGUFNativeConfigError",
    "GGUFNativeCallFailedError",
    "GGUFNativeBackend",
    "default_gguf_path",
    "resolve_gguf_path",
    "validate_gemma_gguf_ready",
    "make_gguf_generate_fn",
]
