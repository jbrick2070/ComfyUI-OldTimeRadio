"""BUG-LOCAL-296 -- the OpenRouter per-RUN cost budget must reset per episode.

`_otr_openrouter_backend._run_token_total` is a module-level accumulator and
`reset_run_budget()` was defined + exported but never called in the live path.
In a persistent headless server (the Scheduled Task launcher) the "per-run"
cost ceiling therefore accumulated across EVERY remote episode and would
spuriously fail-closed after a few runs. The fix wires `reset_run_budget()`
into the single per-episode entry -- `OTR_LedgerScriptWriter.run()` -- at the
very top, before any remote LLM work (the writer's own passes + the downstream
cascade share the process global, so one reset scopes the ceiling to one
episode). No network, no GPU.
"""
from __future__ import annotations

import pytest

from nodes import _otr_openrouter_backend as orb
import nodes.OTR_LedgerScriptWriter as W


def test_reset_run_budget_zeroes_the_accumulator():
    """Direct contract: the accumulator climbs as calls are accounted and
    reset_run_budget() returns it to zero."""
    orb.reset_run_budget()
    backend = orb.OpenRouterBackend()
    backend._account_spend(5000)
    backend._account_spend(5000)
    assert orb._run_token_total == 10000
    orb.reset_run_budget()
    assert orb._run_token_total == 0


def test_writer_run_resets_budget_before_any_work(monkeypatch):
    """The writer must call reset_run_budget() at the TOP of run(), before it
    touches RSS / the LLM. We spy on the reset and make the first post-reset
    step (`_resolve_inputs`) blow up immediately, so run() does no real work;
    the spy must have fired exactly once first."""
    calls = {"n": 0}
    monkeypatch.setattr(orb, "reset_run_budget",
                        lambda: calls.__setitem__("n", calls["n"] + 1))

    # Make the first real step abort so run() performs no RSS/LLM/GPU work.
    def _boom(**_kwargs):
        raise RuntimeError("stop-after-reset")

    monkeypatch.setattr(W, "_resolve_inputs", _boom)

    writer = W.OTR_LedgerScriptWriter()
    with pytest.raises(RuntimeError, match="stop-after-reset"):
        writer.run()

    assert calls["n"] == 1, "writer.run() must reset the remote budget exactly once, first"


def test_writer_reset_is_defensive_when_backend_import_fails(monkeypatch):
    """PD1: the budget reset must never be load-bearing. If the reset raises,
    run() still proceeds to its real first step (which we stub to a sentinel
    error) -- i.e. the reset failure is swallowed, not propagated."""
    def _raise():
        raise ImportError("simulated backend import failure")

    monkeypatch.setattr(orb, "reset_run_budget", _raise)

    def _sentinel(**_kwargs):
        raise RuntimeError("reached-resolve-inputs")

    monkeypatch.setattr(W, "_resolve_inputs", _sentinel)

    writer = W.OTR_LedgerScriptWriter()
    # The ImportError from the reset is swallowed; the RuntimeError from the
    # real first step is what surfaces -- proving run() did not abort on the
    # reset failure.
    with pytest.raises(RuntimeError, match="reached-resolve-inputs"):
        writer.run()
