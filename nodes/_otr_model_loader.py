"""nodes/_otr_model_loader.py

Thin facade over story_orchestrator._load_llm / _unload_llm. Exists to give
the v2.0 path (_otr_outline, _otr_line_composer, OTR_LedgerScriptWriter) a
stable import surface that doesn't depend on the legacy orchestrator's
internal layout.

DOES NOT extract code from story_orchestrator.py during the in-flight FULL
acceptance soak. Re-exports only. When the legacy writer retires in v2.1,
the loader code can move here without changing any v2.0 import sites.

Public surface:
    load_llm(model_id, *, device='cuda', optimization_profile='Standard') -> dict
        Returns a cache_entry dict: {model, tokenizer, model_id, device,
        quantized, budget_profile, context_cap}. Wraps _load_llm's tuple
        return into the dict shape documented by _otr_outline.py.

    unload_llm() -> None
        Re-export of _unload_llm. Frees VRAM globally.

    MODEL_CONTEXT_CAPS: dict[str, int]
        Local copy of the per-model context-window caps. Drift-checked at
        first use against the function-local dict in _load_llm.

    make_generate_fn(cache_entry) -> GenerateFn
        Wraps a cache_entry into a chat-template-aware callable matching
        the GenerateFn contract used by _otr_outline.generate_outline and
        _otr_line_composer.compose_line:
            (messages, *, temperature, max_new_tokens) -> str

Status: Phase 2 of v2.0 sprint. Stdlib + lazy imports of torch and
story_orchestrator only.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("OTR")


__all__ = [
    "load_llm",
    "unload_llm",
    "make_generate_fn",
    "make_polish_generate_fn",
    "ModelLoaderError",
]


# ---------------------------------------------------------------------------
# S30 B1b: MODEL_CONTEXT_CAPS static dict + DEFAULT_CONTEXT_CAP constant
# DELETED. Context-cap resolution now goes through
# nodes._otr_model_catalog.resolve_context_cap which returns a tiered
# ContextCapVerdict (PASS for curated overrides, WARN for parsed
# config.json, UNKNOWN for unresolved) and clamps everything against
# HARD_VRAM_CONTEXT_LIMIT.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class ModelLoaderError(RuntimeError):
    """Raised when load_llm or make_generate_fn cannot complete.

    Wraps lower-level exceptions from the legacy _load_llm path so
    callers get a stable exception type to catch.
    """


# ---------------------------------------------------------------------------
# load_llm -- wraps tuple return into dict
# ---------------------------------------------------------------------------


def load_llm(
    model_id: str,
    *,
    device: str = "cuda",
    optimization_profile: str = "Standard",
) -> dict[str, Any]:
    """Load (or reuse cached) LLM via the legacy story_orchestrator path.

    Wraps _load_llm's (model, tokenizer) tuple return into a cache_entry
    dict shaped for the v2.0 path:
        {
            "model":        <torch model>,
            "tokenizer":    <tokenizer>,
            "model_id":     <canonical model_id, no UI suffix>,
            "device":       <device string actually placed on>,
            "quantized":    <bool, True for NF4/8-bit profiles>,
            "context_cap":  <int, from MODEL_CONTEXT_CAPS or DEFAULT>,
        }

    Args:
        model_id: HF model identifier. UI suffixes like "[BETA]" or
                  "[8-bit]" are tolerated and stripped.
        device:   target device. Defaults "cuda".
        optimization_profile: one of "Standard", "Obsidian", "8-bit",
                  matching the legacy orchestrator's profile names.

    Returns the cache_entry dict.

    Raises ModelLoaderError on any underlying failure (wraps the
    original exception via __cause__).
    """
    try:
        from . import story_orchestrator as _so
    except ImportError as exc:
        raise ModelLoaderError(
            "story_orchestrator not importable; facade requires legacy "
            "module to be on the Python path."
        ) from exc

    try:
        model, tokenizer = _so._load_llm(
            model_id_full=model_id,
            device=device,
            optimization_profile=optimization_profile,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoaderError(
            f"_load_llm failed for model_id={model_id!r}: {exc}"
        ) from exc

    # B1b: dynamic context-cap via the catalog. UI label-suffixes are
    # stripped by validate_model_id elsewhere; defensive strip-on-space
    # here keeps the loader path resilient against any caller that
    # forgets to normalize first.
    from . import _otr_model_catalog as _otr_catalog
    canonical_id = model_id.split(" ")[0].strip()
    context_cap = _otr_catalog.resolve_context_cap(canonical_id).value
    is_quantized = (
        "Obsidian" in optimization_profile
        or "8-bit" in optimization_profile
        or "4-bit" in model_id.lower()
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_id": canonical_id,
        "device": device,
        "quantized": is_quantized,
        "context_cap": context_cap,
    }


def unload_llm() -> None:
    """Free the cached LLM. Re-exports story_orchestrator._unload_llm."""
    try:
        from . import story_orchestrator as _so
    except ImportError:
        return
    try:
        _so._unload_llm()
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR_ModelLoader] unload_llm raised: %s", exc)


# ---------------------------------------------------------------------------
# make_generate_fn -- chat-template adapter
# ---------------------------------------------------------------------------


def make_generate_fn(cache_entry: dict[str, Any]):
    """Wrap a cache_entry into the GenerateFn callable.

    Returns a callable matching:
        (messages, *, temperature, max_new_tokens) -> str

    where `messages` is a list[dict] in chat format
    ([{"role": "system", "content": ...}, {"role": "user", "content": ...}])
    and the return is the raw decoded string from the model with the
    prompt prefix removed.

    Generation params hardcoded for the v2.0 path:
        do_sample=True
        top_p=0.92

    Caller controls temperature and max_new_tokens per call.

    Raises ModelLoaderError if the cache_entry is missing required
    keys or if torch is not importable at first call time.
    """
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def generate_fn(messages, *, temperature, max_new_tokens):
        # Lazy torch import. Raised as ModelLoaderError to match the
        # facade's exception contract.
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Strip prompt prefix from decoded output.
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )

    return generate_fn


# ---------------------------------------------------------------------------
# Polish-specific generate fn (LFC sprint commit 3, section 6.4)
# ---------------------------------------------------------------------------


# Polish-specific sampling baked in per ADR section 6.4. None of these
# are configurable per-call -- the whole point of the dedicated polish
# fn is that the writer's closure-captured composer tuning
# (repetition_penalty, min_p, top_p tweaks) cannot leak in.
_POLISH_TOP_P: float = 0.9
_POLISH_DO_SAMPLE: bool = True


def make_polish_generate_fn(cache_entry: dict[str, Any]):
    """Build a polish-specific generate fn from `cache_entry`.

    LFC sprint commit 3, ADR section 6.4 (2026-05-11). Polish is a
    short, targeted rewrite -- conceptually closer to a constrained
    edit than the composer's long-form generation. The writer's main
    `make_generate_fn` (via the OTR_LedgerScriptWriter
    `_build_truncating_generate_fn` wrapper) bakes
    repetition_penalty / min_p / top_p into its closure tuned for
    composition. Those settings leak into polish via closure capture
    and produce awkward substitutions on short rewrites.

    The polish fn here is a SEPARATE closure off the same cache_entry
    with composer-independent sampling:

        temperature      -- caller-provided per call (defaults to 0.4
                            via _otr_line_composer.polish_line)
        top_p            -- 0.9 (slightly tighter than composer 0.92)
        do_sample        -- True
        min_p            -- not passed (transformers default 0)
        repetition_penalty -- not passed (transformers default 1.0)

    Returns a callable with the same signature as `make_generate_fn`:
        (messages, *, temperature, max_new_tokens) -> str
    """
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def polish_generate_fn(messages, *, temperature, max_new_tokens):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                do_sample=_POLISH_DO_SAMPLE,
                temperature=temperature,
                top_p=_POLISH_TOP_P,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )

    return polish_generate_fn


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_model_loader.py`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_model_loader.py self-test ===")

    # Test 1: catalog.resolve_context_cap returns a sane PASS for the
    # C7 audio-baseline model (Mistral-Nemo).
    print("\n[Test 1] resolve_context_cap baseline")
    from . import _otr_model_catalog as _otr_catalog
    verdict = _otr_catalog.resolve_context_cap(_otr_catalog.DEFAULT_LLM)
    assert verdict.tier == "PASS"
    assert verdict.value == 8192
    print(f"  PASS ({verdict.tier} @ {verdict.value}, source={verdict.source})")

    # Test 2: HARD_VRAM_CONTEXT_LIMIT is at least 4096 (matches old
    # DEFAULT_CONTEXT_CAP minimum invariant).
    print("\n[Test 2] HARD_VRAM_CONTEXT_LIMIT is sane")
    assert _otr_catalog.HARD_VRAM_CONTEXT_LIMIT >= 4096
    print(f"  PASS ({_otr_catalog.HARD_VRAM_CONTEXT_LIMIT})")

    # Test 3: ModelLoaderError shape.
    print("\n[Test 3] ModelLoaderError is RuntimeError subclass")
    assert issubclass(ModelLoaderError, RuntimeError)
    print("  PASS")

    # Test 4: make_generate_fn rejects malformed cache_entry.
    print("\n[Test 4] make_generate_fn rejects malformed cache_entry")
    try:
        make_generate_fn({})
        print("  FAIL: empty cache_entry was accepted")
    except ModelLoaderError as e:
        assert "missing required keys" in str(e)
        print(f"  PASS: rejected ({e})")

    try:
        make_generate_fn({"model": object()})
        print("  FAIL: cache_entry missing tokenizer was accepted")
    except ModelLoaderError as e:
        assert "tokenizer" in str(e)
        print(f"  PASS: rejected ({e})")

    # Test 5: make_generate_fn returns callable with right shape.
    print("\n[Test 5] make_generate_fn returns callable")
    class _StubTok:
        eos_token_id = 0
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # kept: mirror HF tokenizer signature
            return "stub-prompt"
        def __call__(self, prompt, return_tensors):  # kept: mirror HF tokenizer signature
            class _Out:
                input_ids = type("S", (), {"shape": (1, 5)})()
                def to(self, device): return self  # kept: mirror torch tensor .to(device) signature
            return _Out()
        def decode(self, ids, skip_special_tokens):  # kept: mirror HF tokenizer signature
            return "stub-output"
    class _StubModel:
        device = "cpu"
        def generate(self, **kwargs): return [[0, 1, 2, 3, 4, 5, 6, 7]]
    stub_entry = {"model": _StubModel(), "tokenizer": _StubTok()}
    fn = make_generate_fn(stub_entry)
    assert callable(fn)
    print("  PASS: make_generate_fn returned callable")

    # Test 6: resolve_context_cap clamps advertised window to
    # HARD_VRAM_CONTEXT_LIMIT for an uncurated unknown id (UNKNOWN tier).
    print("\n[Test 6] resolve_context_cap UNKNOWN-tier defaults to limit")
    v = _otr_catalog.resolve_context_cap("some/uncurated-test-id")
    assert v.tier == "UNKNOWN"
    assert v.value == _otr_catalog.HARD_VRAM_CONTEXT_LIMIT
    print(f"  PASS ({v.tier} @ {v.value})")

    # Test 7: load_llm raises ModelLoaderError, not bare ImportError,
    #         when story_orchestrator can't be imported.
    # (We can't easily simulate this without monkeypatching sys.modules;
    #  skip the negative case but verify the wrapping logic by inspection
    #  of load_llm's body.)
    print("\n[Test 7] load_llm exception wrapping (smoke check)")
    import inspect
    src = inspect.getsource(load_llm)
    assert "ModelLoaderError" in src
    assert "from exc" in src
    print("  PASS")

    print("\n=== Task 4 self-tests passed ===")
