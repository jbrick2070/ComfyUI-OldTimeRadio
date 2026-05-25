"""S32 B3 -- lock_cast schema-validation (repair) dispatch to technical_fn.

Per architectural decision D2:
* Generation (attempts 1..N-1, fresh calls): creative_fn.
* Schema validation (repair attempt, the final attempt that hands
  prior raw + error back to the LLM): technical_fn (single attempt,
  fail-fast).
* If the technical-slot repair output also fails validation, raise
  `CastValidationLLMError` (subclass of `CastingFailedError` so
  legacy handlers still match; writer-side caller branches on the
  subclass to trigger creative regen).

Tests:
* `test_lock_cast_generation_uses_creative` -- fresh attempts hit
  creative_fn.
* `test_lock_cast_validation_uses_technical` -- repair attempt hits
  technical_fn.
* `test_lock_cast_validation_failfast_no_internal_retry` -- when
  the technical-slot repair output fails validation,
  `CastValidationLLMError` is raised (no further attempts).
* `test_lock_cast_validation_fail_triggers_writer_regen` --
  `CastValidationLLMError` is a subclass of `CastingFailedError`
  AND distinguishable as a separate class (the writer-visible
  signal contract).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_VALID_RESPONSE_TEMPLATE = {
    "gender": "female",
    "voice_preset": "v2/en_speaker_4",
    "character_description": "A test character description that's substantive enough to pass any length validator that might exist in the schema.",
}


def _bad_response() -> str:
    """Output that fails schema validation."""
    return "not-valid-json-at-all"


def _valid_response_for_pool(available_voices: list[tuple[str, str]]) -> str:
    """A valid JSON response keyed to the FIRST entry in available_voices."""
    if not available_voices:
        return _bad_response()
    preset, _label = available_voices[0]
    payload = dict(_VALID_RESPONSE_TEMPLATE)
    payload["voice_preset"] = preset
    return json.dumps(payload)


def test_lock_cast_generation_uses_creative():
    """Fresh-attempt generation (attempts 1..N-1) routes to
    creative_fn -- the LLM call site that takes the original
    user_prompt without prior raw + error context.
    """
    from nodes import _otr_casting as cast

    creative_calls: list[float] = []
    technical_calls: list[float] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append(temperature)
        # Return malformed so we drop to repair (technical) on the
        # final attempt; the routing assertion below catches both.
        return _bad_response()

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append(temperature)
        return _bad_response()

    try:
        cast.lock_cast(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: the LEMMY ~11% cameo rolls on OS
            # entropy, not the seeded rng (BUG-LOCAL-260). With
            # num_characters=1 a LEMMY hit consumes the only open slot,
            # making zero generation calls and flaking these routing
            # assertions. Pin it OFF for determinism.
            force_lemmy=False,
            max_attempts_per_call=3,
        )
    except cast.CastingFailedError:
        pass  # expected; we're asserting routing.

    assert len(creative_calls) >= 1, (
        "Generation attempts (creative slot) MUST hit creative_fn at "
        "least once before the repair attempt routes to technical."
    )


def test_lock_cast_validation_uses_technical():
    """Repair-attempt (schema-validation slot) routes to
    technical_fn. The repair fires on the FINAL attempt only,
    after at least one fresh attempt has produced a `last_raw`
    string to hand back.
    """
    from nodes import _otr_casting as cast

    creative_calls: list[float] = []
    technical_calls: list[float] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append(temperature)
        return _bad_response()

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append(temperature)
        return _bad_response()

    try:
        cast.lock_cast(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: the LEMMY ~11% cameo rolls on OS
            # entropy, not the seeded rng (BUG-LOCAL-260). With
            # num_characters=1 a LEMMY hit consumes the only open slot,
            # making zero generation calls and flaking these routing
            # assertions. Pin it OFF for determinism.
            force_lemmy=False,
            max_attempts_per_call=3,  # 2 fresh + 1 repair
        )
    except cast.CastingFailedError:
        pass

    assert len(technical_calls) >= 1, (
        "Repair attempt MUST route to technical_fn. Got 0 technical "
        f"calls (creative_calls={len(creative_calls)}). The B3 "
        "schema-validation flip did not land."
    )


def test_lock_cast_validation_failfast_no_internal_retry():
    """When the technical-slot repair-attempt output fails
    validation, `CastValidationLLMError` is raised. No further
    retry tier inside `cast_one_character` (D2: single-attempt
    technical, fail-fast)."""
    from nodes import _otr_casting as cast

    def always_bad(messages, *, temperature, max_new_tokens):
        return _bad_response()

    with pytest.raises(cast.CastValidationLLMError):
        cast.lock_cast(
            creative_fn=always_bad,
            technical_fn=always_bad,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: pin out the LEMMY ~11% cameo (rolls on
            # OS entropy, not the seeded rng -- BUG-LOCAL-260) so the
            # raise-expecting assertion stays deterministic.
            force_lemmy=False,
            max_attempts_per_call=3,
        )


def test_lock_cast_validation_fail_triggers_writer_regen():
    """`CastValidationLLMError` is a subclass of
    `CastingFailedError` (legacy handlers still match) AND is a
    distinct class (writer-side callers can branch on the more
    specific type to trigger creative regen rather than hard fail).
    """
    from nodes import _otr_casting as cast

    # Subclass relationship.
    assert issubclass(cast.CastValidationLLMError, cast.CastingFailedError), (
        "`CastValidationLLMError` must subclass `CastingFailedError` "
        "so legacy handlers catching the base class still match."
    )
    # Distinct class identity.
    assert cast.CastValidationLLMError is not cast.CastingFailedError, (
        "`CastValidationLLMError` must be a distinct class from "
        "`CastingFailedError` so callers can branch on the subclass "
        "to trigger writer-side creative regen."
    )

    # Runtime: raise propagates as both types catchable.
    def always_bad(messages, *, temperature, max_new_tokens):
        return _bad_response()

    raised_as_subclass = False
    raised_as_base = False
    try:
        cast.lock_cast(
            creative_fn=always_bad,
            technical_fn=always_bad,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: pin out the LEMMY ~11% cameo (rolls on
            # OS entropy, not the seeded rng -- BUG-LOCAL-260) so the
            # raise-expecting assertion stays deterministic.
            force_lemmy=False,
            max_attempts_per_call=3,
        )
    except cast.CastValidationLLMError:
        raised_as_subclass = True
    except cast.CastingFailedError:
        raised_as_base = True

    assert raised_as_subclass, (
        "The repair-attempt validation failure must raise "
        "`CastValidationLLMError` specifically (not the bare base "
        "class)."
    )
    assert not raised_as_base
