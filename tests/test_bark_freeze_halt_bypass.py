"""Sprint 10A interim regression -- BatchBarkGenerator bypass_freeze_halt widget.

The BUG-LOCAL-276 halt (hotfix 9a77144, 2026-05-26) refuses to render
TTS when meta.freeze_verdict='needs_full_rerun'. Production-safe by
default. But smoke-budget runs (target_words=30, 100) routinely trip
needs_full_rerun because the legacy story critic can't find an arc in
3 character lines -- the halt then blocks the fast-iteration soak
loop.

This module pins the bypass widget contract:
  * INPUT_TYPES exposes bypass_freeze_halt BOOLEAN with default=False
    (production-safe halt preserved).
  * Source-level check: the halt block branches on the kwarg and
    logs a WARNING when bypassed, raises ValueError when not.
  * Workflow JSON node 11 widgets_values length advanced to include
    the new widget at default False.

Tests do NOT load Bark; they exercise the INPUT_TYPES surface +
source-level pins + workflow JSON contract.
"""

from __future__ import annotations

import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BBG_SRC = REPO_ROOT / "nodes" / "batch_bark_generator.py"
WORKFLOW_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


# ---------------------------------------------------------------------------
# INPUT_TYPES contract
# ---------------------------------------------------------------------------


class TestWidgetSurface:
    def test_bypass_widget_present_in_optional(self):
        from nodes.batch_bark_generator import BatchBarkGenerator

        inputs = BatchBarkGenerator.INPUT_TYPES()
        assert "optional" in inputs
        assert "bypass_freeze_halt" in inputs["optional"], (
            "bypass_freeze_halt widget must be in INPUT_TYPES.optional"
        )
        spec = inputs["optional"]["bypass_freeze_halt"]
        assert spec[0] == "BOOLEAN"
        assert spec[1].get("default") is False, (
            "bypass_freeze_halt must default to False so the BUG-LOCAL-276 "
            "production-safe halt remains the out-of-box behaviour"
        )

    def test_generate_batch_signature_accepts_kwarg(self):
        """The generate_batch method must accept bypass_freeze_halt as a
        kwarg with default False. ComfyUI binds widgets to method kwargs
        by name."""
        from nodes.batch_bark_generator import BatchBarkGenerator
        import inspect

        sig = inspect.signature(BatchBarkGenerator.generate_batch)
        params = sig.parameters
        assert "bypass_freeze_halt" in params
        assert params["bypass_freeze_halt"].default is False


# ---------------------------------------------------------------------------
# Source-level pins on the halt block
# ---------------------------------------------------------------------------


class TestHaltBlockSource:
    def _src(self) -> str:
        return BBG_SRC.read_text(encoding="utf-8")

    def test_halt_branches_on_bypass(self):
        """The halt code must branch on the bypass flag -- a code path
        that ALWAYS raises would defeat the widget."""
        src = self._src()
        # The verdict check is followed by 'if bypass_freeze_halt:' before
        # the unconditional raise.
        assert 'if _verdict == "needs_full_rerun":' in src
        # And the conditional branch on bypass_freeze_halt must come
        # AFTER the verdict check.
        verdict_idx = src.index('if _verdict == "needs_full_rerun":')
        bypass_idx = src.index("if bypass_freeze_halt:", verdict_idx)
        assert bypass_idx > verdict_idx, (
            "bypass_freeze_halt branch must live inside the "
            "needs_full_rerun check, not outside it"
        )

    def test_bypass_branch_logs_warning(self):
        """The bypass branch must log a WARNING (loud enough to show in
        soak grep) and must NOT raise. Otherwise the operator would
        have no signal that an unsafe render is in flight."""
        src = self._src()
        bypass_idx = src.index("if bypass_freeze_halt:")
        # Look forward to the next major statement -- there must be a
        # log.warning(...) call before any 'raise' keyword.
        rest = src[bypass_idx:bypass_idx + 2000]
        warning_idx = rest.find("log.warning")
        raise_idx = rest.find("raise")
        assert warning_idx != -1, "bypass branch must call log.warning(...)"
        assert raise_idx == -1 or warning_idx < raise_idx, (
            "bypass branch must NOT raise; log.warning + proceed is "
            "the contract"
        )
        # And the warning must reference the BUG number so soak greps
        # can find it.
        assert "BUG-LOCAL-276" in rest

    def test_halt_branch_still_references_bug(self):
        """The non-bypass branch (the actual halt) must keep referencing
        BUG-LOCAL-276 so the error message points the operator at the
        right BUG_LOG entry."""
        src = self._src()
        verdict_block_start = src.index('if _verdict == "needs_full_rerun":')
        # Window widened (2000 -> 3000) for the BUG-LOCAL-300 quality branch
        # that now sits between the bypass warning and the structural
        # ValueError. The assertion (both reference BUG-LOCAL-276) is unchanged.
        block = src[verdict_block_start:verdict_block_start + 3000]
        # Both the raise ValueError(...) string AND the bypass log
        # message reference BUG-LOCAL-276.
        assert block.count("BUG-LOCAL-276") >= 2, (
            "expected the halt block to reference BUG-LOCAL-276 in both "
            "the warning (bypass path) and the ValueError (halt path); "
            f"found {block.count('BUG-LOCAL-276')} mention(s)"
        )

    def test_halt_error_message_mentions_widget_path(self):
        """The non-bypass error message should mention the bypass widget
        so an operator hitting the halt learns about the escape hatch
        without having to read source."""
        src = self._src()
        # Look for the raise ValueError(...) text inside the halt block.
        verdict_idx = src.index('if _verdict == "needs_full_rerun":')
        block = src[verdict_idx:verdict_idx + 2500]
        assert "bypass_freeze_halt" in block, (
            "ValueError message should name the bypass widget so the "
            "operator can find the escape hatch from the error alone"
        )

    def test_quality_block_branch_renders_not_raises(self):
        """BUG-LOCAL-300 safety/quality split: a freeze_block_class=='quality'
        block must RENDER (log.warning + proceed), never raise -- so a
        renderable story is not blocked on a subjective arc verdict. Only the
        structural else-branch raises."""
        src = self._src()
        verdict_idx = src.index('if _verdict == "needs_full_rerun":')
        q_idx = src.index('_block_class == "quality"', verdict_idx)
        assert q_idx > verdict_idx, (
            "the quality branch must live inside the needs_full_rerun check"
        )
        # The quality branch runs from its elif to the structural else.
        branch = src[q_idx:src.index("else:", q_idx)]
        assert "log.warning" in branch, "quality branch must log.warning"
        assert "raise" not in branch, (
            "quality branch must NOT raise -- it renders the (renderable) "
            "ledger; only the structural else-branch halts"
        )
        assert "BUG-LOCAL-300" in branch, (
            "quality branch must reference BUG-LOCAL-300 for soak greps"
        )

    def test_quality_block_env_override_present(self):
        """The strict-halt escape hatch is an env var (OTR project convention:
        every knob is an env var) so an operator can restore the old
        halt-on-any behaviour without a workflow JSON change."""
        src = self._src()
        assert "OTR_BARK_HALT_ON_QUALITY_BLOCK" in src, (
            "the quality split must expose OTR_BARK_HALT_ON_QUALITY_BLOCK to "
            "restore strict halting"
        )


# ---------------------------------------------------------------------------
# Workflow JSON pin
# ---------------------------------------------------------------------------


class TestWorkflowJsonNode11:
    def test_node_11_widgets_values_includes_bypass_default(self):
        wf = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
        node = next(n for n in wf["nodes"] if n.get("id") == 11)
        assert node["type"] == "OTR_BatchBarkGenerator"
        wv = node["widgets_values"]
        # Pre-widget layout: ["[]", 0.7]. Post-widget: ["[]", 0.7, False].
        assert len(wv) == 3, (
            f"OTR_BatchBarkGenerator widgets_values length must be 3 "
            f"after the bypass_freeze_halt widget addition; got {len(wv)}"
        )
        assert wv[0] == "[]"        # script_json default
        assert wv[1] == 0.7         # temperature default
        assert wv[2] is False, (
            f"bypass_freeze_halt widget must default to False in the "
            f"canonical workflow; got {wv[2]!r}"
        )
