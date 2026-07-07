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
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_OUTPUT_TOKENS_CAP = 512
DEFAULT_N_BATCH = 512
DEFAULT_N_GPU_LAYERS = -1

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


def default_gguf_path() -> Path:
    return (
        _models_root()
        / "LLM"
        / "converted"
        / "gemma-4-12b-it"
        / DEFAULT_GGUF_FILENAME
    )


def resolve_gguf_path() -> Path:
    raw = os.environ.get("GEMMA4_12B_GGUF_PATH")
    return Path(raw).expanduser() if raw else default_gguf_path()


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

    for site_packages in _site_packages_candidates():
        dll_dirs = [
            site_packages / "nvidia" / "cuda_runtime" / "bin",
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "llama_cpp" / "lib",
            site_packages / "torch" / "lib",
        ]
        for dll_dir in dll_dirs:
            if dll_dir.exists() and hasattr(os, "add_dll_directory"):
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
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
                _PRELOADED_DLLS.append(ctypes.WinDLL(str(dll_path)))


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

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        model_path = resolve_gguf_path()
        if not model_path.exists():
            raise GGUFNativeConfigError(
                f"Missing Gemma 4 12B Q8_0 GGUF file: {model_path}. "
                f"Download {DEFAULT_GGUF_FILENAME} from {ROW_ID} and place it "
                "under C:\\ComfyUI-Models\\LLM\\converted\\gemma-4-12b-it\\, "
                "or set GEMMA4_12B_GGUF_PATH."
            )
        if (
            model_path.name == DEFAULT_GGUF_FILENAME
            and model_path.stat().st_size != EXPECTED_Q8_0_SIZE_BYTES
        ):
            raise GGUFNativeConfigError(
                f"Incomplete Gemma 4 12B Q8_0 GGUF file: {model_path} has "
                f"{model_path.stat().st_size} bytes, expected "
                f"{EXPECTED_Q8_0_SIZE_BYTES}. Let the download finish or "
                "delete the partial file and retry."
            )
        context_window = int(
            getattr(row, "context_window", DEFAULT_CONTEXT_WINDOW)
            or DEFAULT_CONTEXT_WINDOW
        )
        n_ctx = _int_env("GEMMA4_12B_N_CTX", context_window)
        n_batch = _int_env("GEMMA4_12B_N_BATCH", DEFAULT_N_BATCH)
        n_gpu_layers = _int_env("GEMMA4_12B_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)
        verbose = _bool_env("GEMMA4_12B_VERBOSE", False)
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
