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

import transformers
import transformers.tokenization_utils as _tu

if not hasattr(_tu, "PreTrainedTokenizerBase"):
    _tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
