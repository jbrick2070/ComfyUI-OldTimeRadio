"""S32 B4 -- compose_line critic dispatch decision (no-widget drift).

The canonical plan called for a `use_technical_critic` opt-in
widget (default OFF) gating per-beat technical-slot dispatch of
the critic sub-pass. Per Jeffrey's no-widget rule and the D1
architectural rejection of per-beat T-dispatch (~3.3 hr of VRAM
transition overhead per episode), the widget was DROPPED at B4.
Critic always routes to creative_fn, unconditionally. The decision
is baked into code with no opt-in surface.

Tests:
* `test_compose_line_critic_always_creative` -- runtime: when
  compose_line is called with distinct creative_fn / technical_fn
  mocks, NO call lands on technical_fn (critic + composer +
  grammarian + polish gate all stay creative).
* `test_no_use_technical_critic_widget` -- INPUT_TYPES inspection:
  `OTR_LedgerScriptWriter`'s widget surface does NOT contain a
  `use_technical_critic` key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_compose_line_critic_always_creative():
    """Distinct creative_fn / technical_fn mocks; assert zero
    technical calls in compose_line. The critic check (along with
    composer, grammarian, polish gate) stays on creative
    regardless of slot configuration -- the no-widget decision.
    """
    from nodes import _otr_line_composer as lc

    creative_calls: list[float] = []
    technical_calls: list[float] = []

    def creative_fn(messages, *, temperature, max_new_tokens, **_kw):
        creative_calls.append(temperature)
        return "test dialogue output."

    def technical_fn(messages, *, temperature, max_new_tokens, **_kw):
        technical_calls.append(temperature)
        return "should_not_be_called"

    req_cls = lc.LineRequest
    req_fields = {
        f.name for f in (
            req_cls.__dataclass_fields__.values()
            if hasattr(req_cls, "__dataclass_fields__") else []
        )
    }
    req_kwargs = {
        "speaker": "ALICE",
        "scene_header": "INT. STUDIO -- NIGHT",
        "canon_header": "INT. STUDIO -- NIGHT",
        "target_words": 8,
        "beat_traits": "calm",
        "allowed_roster": ["ALICE", "BOB"],
        "is_announcer": False,
    }
    req_kwargs = {
        k: v for k, v in req_kwargs.items()
        if not req_fields or k in req_fields
    }
    try:
        req = req_cls(**req_kwargs)
    except TypeError:
        pytest.skip("LineRequest signature changed; runtime path skipped")

    try:
        lc.compose_line(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            req=req,
        )
    except Exception:
        # Don't care about downstream validator failures here; the
        # routing assertion is the test.
        pass

    assert len(technical_calls) == 0, (
        f"compose_line must NEVER route to technical_fn -- "
        f"all sub-passes (composer, critic, grammarian, polish "
        f"gate) stay creative. The no-widget decision (S32 B4) "
        f"is baked into code. Got {len(technical_calls)} call(s) "
        f"to technical_fn."
    )
    assert len(creative_calls) >= 1, (
        "creative_fn must be called for the composer pass."
    )


def test_no_use_technical_critic_widget():
    """`OTR_LedgerScriptWriter.INPUT_TYPES` MUST NOT contain a
    `use_technical_critic` key. The opt-in widget was rejected at
    S32 B4. Tripwire against accidental reintroduction.
    """
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    optional = spec.get("optional", {})
    required = spec.get("required", {})

    assert "use_technical_critic" not in optional, (
        "`use_technical_critic` widget was rejected at S32 B4 "
        "(no-widget rule, plan D1 architectural rejection of "
        "per-beat T-dispatch). Re-introducing the widget brings "
        "back the dead opt-in surface the rule guards against."
    )
    assert "use_technical_critic" not in required, (
        "`use_technical_critic` must not appear in the required "
        "block either."
    )
