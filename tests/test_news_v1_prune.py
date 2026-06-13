"""R3 A2b -- V1 key_term prune-to-floor (durable soak-green fix).

When the brief generator fabricates a key_term the article does not
support, the pre-A2b behavior exhausted the structured-call retry
ladder and raised ``NewsInterpreterError`` -- which HALTS the whole
episode. A soak/batch cannot re-roll RSS, so a single bad story draw
aborted the leg (the R3 finding, ~1-2/25 legs).

A2b makes ``_content_validator`` PRUNE the fabricated key_terms (those
that fail a strict word-boundary in-source match) and accept the
grounded subset when at least ``_MIN_KEY_TERMS`` survive. It prunes,
never relaxes:
  * fewer than _MIN_KEY_TERMS grounded -> still halts (V0 floor);
  * a non-key-term validator failure (V2/V3) still halts.

Hermetic: no GPU, no I/O, no live LLM. generate_fn is a stub.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

news_interpreter = pytest.importorskip("news_interpreter")


# Source article body. Contains "telescope", "signal", "researchers";
# does NOT contain "wormhole" or "antimatter" (the fabricated terms).
_SOURCE = (
    "Astronomers using a large telescope reported an unexplained "
    "signal from a nearby star system this week. Researchers cautioned "
    "that the data must be verified before publication."
)

# Neutral brief text -- no period terms (keeps V2 clean) and no style
# violations (keeps V3 clean) so the only failures under test are V1.
_BASE_BRIEF = {
    "casting_brief": (
        "A field astronomer who studies the discovery and a skeptical "
        "colleague who tests every claim."
    ),
    "script_brief": (
        "A telescope picks up an unexplained signal. Two researchers "
        "argue about how to verify it before publication."
    ),
    "news_close_brief": (
        "Researchers reported an unexplained signal from a nearby star "
        "system, with verification pending."
    ),
}


def _brief_json(key_terms, **overrides):
    payload = dict(_BASE_BRIEF)
    payload["key_terms"] = list(key_terms)
    payload.update(overrides)
    return json.dumps(payload)


def _stub_gen(response_text):
    """generate_fn that yields ``response_text`` verbatim. The judge
    fallback shares this fn; a JSON payload starts with '{', so the
    conservative yes/no parser reads it as a rejection -- fabricated
    terms reliably fail V1."""
    def _fn(messages, *, temperature=0.7, max_new_tokens=400):  # noqa: ARG001
        return response_text
    return _fn


def _build(key_terms, **overrides):
    return news_interpreter.build_news_briefs(
        technical_fn=_stub_gen(_brief_json(key_terms, **overrides)),
        full_text=_SOURCE,
        headline="Signal detected",
        summary=_SOURCE[:120],
        style="noir",
        seed=42,
    )


def test_prune_drops_fabricated_terms_to_floor():
    """4 terms, 2 fabricated -> prunes to the 2 grounded terms and
    succeeds instead of halting the run."""
    briefs = _build(["telescope", "signal", "wormhole", "antimatter"])
    assert set(briefs.key_terms) == {"telescope", "signal"}, (
        "A2b should have pruned the fabricated terms to the grounded "
        f"floor; got {briefs.key_terms!r}"
    )


def test_one_grounded_term_still_halts_at_floor():
    """Only 1 grounded term -> fewer than _MIN_KEY_TERMS survive, so the
    run still halts (prunes, never relaxes below the V0 floor)."""
    assert news_interpreter._MIN_KEY_TERMS == 2
    with pytest.raises(news_interpreter.NewsInterpreterError):
        _build(["telescope", "wormhole", "antimatter"])


def test_all_grounded_terms_pass_unchanged():
    """A fully grounded brief is accepted with its key_terms intact --
    the prune path is never entered when there are no V1 failures."""
    briefs = _build(["telescope", "signal", "researchers"])
    assert set(briefs.key_terms) == {"telescope", "signal", "researchers"}


def test_prune_does_not_rescue_non_keyterm_failure():
    """When the failure is V3 (formulaic style phrasing), not a
    fabricated key_term, pruning cannot clear it and the run still
    halts -- A2b never silently ships a brief that fails a validator.
    Both key_terms are grounded, so the prune path early-returns the
    failure (nothing to prune) rather than masking the V3 problem."""
    with pytest.raises(news_interpreter.NewsInterpreterError):
        _build(
            ["telescope", "signal"],
            script_brief=(
                "A noir-style detective hears an unexplained signal "
                "through a telescope and starts asking questions."
            ),
        )
