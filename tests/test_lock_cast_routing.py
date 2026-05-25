"""lock_cast slot routing + CastValidationLLMError promotion.

Sprint 2A/2D converted cast_one_character onto the shared
structured_call retry ladder and removed lock_cast's technical_fn
parameter. Casting now runs entirely on the creative slot -- there is
no technical-slot routing left to test (S32 B3 fully reversed).

What still holds and is tested here:
* lock_cast drives generation through creative_fn.
* When cast_one_character exhausts the ladder for an open slot,
  lock_cast promotes the CastingFailedError to CastValidationLLMError
  -- a subclass the writer-side caller branches on to trigger creative
  regen rather than a hard fail.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _bad_response() -> str:
    """Output that fails schema validation."""
    return "not-valid-json-at-all"


def test_lock_cast_generation_uses_creative():
    """lock_cast drives every casting attempt through creative_fn."""
    from nodes import _otr_casting as cast

    creative_calls: list[float] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append(temperature)
        return _bad_response()

    try:
        cast.lock_cast(
            creative_fn=creative_fn,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: the LEMMY ~11% cameo rolls on OS
            # entropy, not the seeded rng (BUG-LOCAL-260). With
            # num_characters=1 a LEMMY hit consumes the only open slot,
            # making zero generation calls and flaking the assertion.
            # Pin it OFF for determinism.
            force_lemmy=False,
            max_attempts_per_call=3,
        )
    except cast.CastingFailedError:
        pass  # expected; we are asserting routing.

    assert len(creative_calls) >= 1, (
        "lock_cast must drive generation through creative_fn"
    )


def test_lock_cast_exhaustion_raises_cast_validation_error():
    """When cast_one_character exhausts its retry ladder for an open
    slot, lock_cast promotes the CastingFailedError to
    CastValidationLLMError (keyed on the attempt count)."""
    from nodes import _otr_casting as cast

    def always_bad(messages, *, temperature, max_new_tokens):
        return _bad_response()

    with pytest.raises(cast.CastValidationLLMError):
        cast.lock_cast(
            creative_fn=always_bad,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            force_lemmy=False,
            max_attempts_per_call=3,
        )


def test_cast_validation_error_is_castingfailed_subclass():
    """CastValidationLLMError is a subclass of CastingFailedError
    (legacy handlers still match) AND a distinct class (writer-side
    callers branch on the subclass to trigger creative regen rather
    than a hard fail).
    """
    from nodes import _otr_casting as cast

    assert issubclass(cast.CastValidationLLMError, cast.CastingFailedError), (
        "CastValidationLLMError must subclass CastingFailedError so "
        "legacy handlers catching the base class still match."
    )
    assert cast.CastValidationLLMError is not cast.CastingFailedError, (
        "CastValidationLLMError must be a distinct class so callers "
        "can branch on the subclass."
    )

    def always_bad(messages, *, temperature, max_new_tokens):
        return _bad_response()

    raised_as_subclass = False
    raised_as_base = False
    try:
        cast.lock_cast(
            creative_fn=always_bad,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            force_lemmy=False,
            max_attempts_per_call=3,
        )
    except cast.CastValidationLLMError:
        raised_as_subclass = True
    except cast.CastingFailedError:
        raised_as_base = True

    assert raised_as_subclass, (
        "a fully-exhausted casting ladder must raise "
        "CastValidationLLMError specifically (not the bare base class)."
    )
    assert not raised_as_base


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
