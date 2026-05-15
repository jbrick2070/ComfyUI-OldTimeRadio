"""S31 B6 Fix 4 -- OTR_VisualPromptCoercion missing-model-id loud-fail
contract.

S30 B5 deleted `OTR_VisualLLMSelector`. Visual prompt coercion now
depends on the writer's `creative_writing_model` broadcast output --
the workflow must wire output 4 of `OTR_LedgerScriptWriter` to this
node's `model_id` input.

Pre-fix: if the input was unwired in a workflow, current behavior
silently fell back to rule-based-only cleanup (the `"none"` default
path).

Post-S31 B6 Fix 4: defensive check raises `MissingModelInputError`
loud with a non-empty `error_hint` explaining the wiring fix. The
explicit `"none"` sentinel remains a supported opt-out (for users
who intentionally want rule-only cleanup; they must pass the literal
string `"none"`, not leave the socket unwired).

Pattern matches `OTR_LedgerFreezeCascade.technical_model` check
post-S30 B3 via `_otr_model_inputs.require_model`.
"""

from __future__ import annotations

import pytest


def test_visual_prompt_coercion_missing_model_id_raises_loud():
    """`coerce(...)` called with `model_id=None` / `""` / whitespace
    raises `MissingModelInputError` with a non-empty message
    explaining the wiring fix.
    """
    from visual.prompt_coercion import VisualPromptCoercion
    from nodes._otr_model_inputs import MissingModelInputError

    node = VisualPromptCoercion()
    # Minimal valid script JSON for the parse step (raises happens
    # BEFORE the parse step, but we provide a valid input so the test
    # doesn't pass for the wrong reason).
    script_json = "[]"

    for unwired_value in (None, "", "   ", "\t\n"):
        with pytest.raises(MissingModelInputError) as exc:
            node.coerce(script_json=script_json, model_id=unwired_value)
        msg = str(exc.value)
        assert msg, "MissingModelInputError must have a non-empty message"
        # Plan-specified error-hint content: name the missing
        # broadcast, the writer output index, and the recovery action.
        assert "creative_writing_model" in msg
        assert "OTR_VisualPromptCoercion" in msg
        assert "Wire" in msg, (
            "Error message must include actionable recovery "
            "instruction containing the word 'Wire'."
        )


def test_visual_prompt_coercion_with_wired_model_id_proceeds():
    """`coerce(...)` called with a non-empty model_id string does NOT
    raise `MissingModelInputError`. The `"none"` sentinel and real
    model ids both pass the check; downstream branching decides
    whether to invoke the LLM polish path.
    """
    from visual.prompt_coercion import VisualPromptCoercion
    from nodes._otr_model_inputs import MissingModelInputError

    node = VisualPromptCoercion()
    script_json = "[]"

    # `"none"` sentinel: skips LLM polish, runs rule-only. Returns
    # cleanly without raising.
    try:
        out = node.coerce(script_json=script_json, model_id="none")
    except MissingModelInputError:
        pytest.fail(
            "`model_id=\"none\"` is the intentional rule-only opt-out "
            "sentinel and must NOT trigger MissingModelInputError."
        )
    assert isinstance(out, tuple) and len(out) == 2

    # A real model id: the LLM polish path is attempted; on failure
    # (no actual model loaded in the test env) it degrades gracefully
    # but does NOT raise MissingModelInputError.
    try:
        out = node.coerce(
            script_json=script_json,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
        )
    except MissingModelInputError:
        pytest.fail(
            "A real model id must NOT trigger MissingModelInputError. "
            "If the LLM polish path itself fails the node degrades to "
            "rule-only -- but the missing-input contract is satisfied."
        )
    assert isinstance(out, tuple) and len(out) == 2
