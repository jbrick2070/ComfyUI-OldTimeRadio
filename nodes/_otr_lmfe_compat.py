"""Compat shim for lm-format-enforcer + transformers v5.

lm-format-enforcer 0.11.3 hard-imports `PreTrainedTokenizerBase` from
`transformers.tokenization_utils` -- its location in transformers v4.
transformers v5 moved it to `transformers.tokenization_utils_base`
(with the top-level `transformers.PreTrainedTokenizerBase` re-export
also pointing there). The v5 install ships transformers.tokenization_utils
without the symbol, so the third-party import fails with:

    ImportError: cannot import name 'PreTrainedTokenizerBase' from
    'transformers.tokenization_utils'

This module aliases the v4 import path back to the v5 location so the
lmformatenforcer integration layer becomes API-compatible.

IMPORTANT: this shim MUST run BEFORE any import from `lmformatenforcer`
or `lmformatenforcer.integrations.transformers`. Always do
`from . import _otr_lmfe_compat  # noqa: F401` at the top of any module
that imports lmformatenforcer, even transitively.

The shim is a no-op when:
  * transformers is v4 and already exports PreTrainedTokenizerBase from
    transformers.tokenization_utils (the alias check below is True).
  * transformers itself stops exporting PreTrainedTokenizerBase at top
    level -- in which case the shim cannot help and lmformatenforcer
    will fail loudly at import time, which is the right behavior.

Reference: Sprint 10A step 3 (story-generator-final-plan.md). The
constrained-decoding backend choice was lm-format-enforcer over
outlines / xgrammar because it is pure Python, no binary wheels needed
(matches the project's torch 2.10 + cu130 + sm_120 + Windows stack
where prebuilt wheels are scarce -- see project memory
hyworld_2_pivot for context).
"""

# Guard: transformers is an optional dep (lives in the ComfyUI venv; absent in
# the bare pytest sandbox). The module-level import must not raise on import so
# that _otr_constrained_generate (which imports this shim at module scope) stays
# importable in test environments where transformers is not installed.
try:
    import transformers as _transformers  # noqa: F401 -- probed below
    import transformers.tokenization_utils as _tu  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def ensure_lmfe_transformers_compat() -> None:
    """Idempotently re-apply the v4 PreTrainedTokenizerBase alias.

    The first import of this module wires the alias at module-load
    time. Some test fixtures elsewhere in the suite reload
    `transformers.tokenization_utils` (or restore it from a cached
    state) which strips the alias and breaks lm-format-enforcer's
    import path on subsequent tests in the same pytest process.

    Call this function at every use site that imports
    lmformatenforcer (typically once at the top of a factory
    function). The check is cheap (single hasattr) and the alias
    assignment is idempotent. No-op when transformers is absent.
    """
    try:
        # Re-resolve transformers.tokenization_utils every call because
        # an importlib.reload would have replaced the module object in
        # sys.modules; our cached `_tu` would then point at the stale
        # (but still in-memory) module.
        import transformers as _t
        import transformers.tokenization_utils as _live_tu
    except ImportError:
        return  # transformers absent; nothing to shim
    if not hasattr(_live_tu, "PreTrainedTokenizerBase"):
        _live_tu.PreTrainedTokenizerBase = _t.PreTrainedTokenizerBase


# Apply at module-import time as well so simple consumers that don't
# call ensure_lmfe_transformers_compat() still see the alias.
# No-op (returns immediately) when transformers is absent.
ensure_lmfe_transformers_compat()
