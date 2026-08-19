"""Every episode says WHICH caster chose its voices, positively.

The hybrid LLM voice-fit went default-OFF on 2026-08-18 and the deterministic
scorer became the caster. The change is one line; making it OBSERVABLE was the
hard part, and this file is that guarantee.

WHY A POSITIVE MARKER AND NOT `voice_cast_decision == {}`. That empty dict is
produced by at least two different situations -- the pass being disabled, and the
pass being ENABLED but finding no char-voice engine (a branch that can be reached
with no exception raised at all). One field, two meanings, which is the defect
shape this repo has logged four times. Asserting on it would have passed happily
while the flip silently failed to take effect.

WHY THIS FILE OUTLIVES THE PASS. `tests/test_hybrid_voice_fit.py` is retired when
the dead code is ripped, because every test in it exercises functions that stop
existing. These tests do not: they assert on the DEFAULT and on the MARKER, both
of which outlive the removal. Keep them.

WHAT AN AUDITOR SHOULD KNOW. The marker is stamped in `lock_cast` and must ALSO
be copied, by name, at `OTR_LedgerScriptWriter`'s key-by-key meta copy -- that
file states the invariant itself: "a key stamped in lock_cast and not named on
this line never reaches the ledger." A review round found the marker would have
been dropped there, so the copy is covered below too.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    for p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from nodes import _otr_casting as _OTRC  # noqa: E402

#: (env value, expected enabled). `None` means the variable is not set at all.
PARSE_CASES = [
    (None, False),
    ("0", False),
    ("", False),
    ("false", False),
    ("False", False),
    ("no", False),
    ("off", False),
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
]


def test_the_hybrid_pass_is_off_by_default():
    """The whole point. If this flips back, the concentration comes back with
    it: measured, the pass cast with 13 distinct voices at 96% top-5 where the
    scorer used 43 at 25%."""
    os.environ.pop("OTR_HYBRID_VOICE_FIT", None)
    assert _OTRC.hybrid_voice_fit_enabled() is False, (
        "the hybrid LLM voice-fit is enabled by default again -- the "
        "deterministic scorer is supposed to be the caster"
    )


@pytest.mark.parametrize("value,expected", PARSE_CASES)
def test_the_gate_is_an_explicit_opt_in_not_a_truthiness_test(monkeypatch, value,
                                                              expected):
    """TEETH on the parse, not just the default.

    The gate used to read `os.environ.get(..., "1") != "0"`. That was safe only
    while the default was ON. With the default OFF, the same expression turns
    `OTR_HYBRID_VOICE_FIT=""` and `="false"` into ENABLED -- the exact opposite
    of what anyone setting those means, and a silent re-enable of the pass this
    change exists to disable.
    """
    monkeypatch.delenv("OTR_HYBRID_VOICE_FIT", raising=False)
    if value is not None:
        monkeypatch.setenv("OTR_HYBRID_VOICE_FIT", value)
    assert _OTRC.hybrid_voice_fit_enabled() is expected, (
        f"OTR_HYBRID_VOICE_FIT={value!r} should mean enabled={expected}"
    )


def _lock(monkeypatch, env_value=None):
    monkeypatch.delenv("OTR_HYBRID_VOICE_FIT", raising=False)
    if env_value is not None:
        monkeypatch.setenv("OTR_HYBRID_VOICE_FIT", env_value)

    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        return '{"character_description": "a voice on the wire"}'

    _cast, meta = _OTRC.lock_cast(
        creative_fn=fn, num_characters=2, news_seed="x", style="open",
        rng=random.Random(7), cast_seed=7, force_lemmy=False,
    )
    return meta


def test_the_ledger_says_the_scorer_cast_it(monkeypatch):
    """The assertion a live leg actually gates on."""
    meta = _lock(monkeypatch)
    assert meta.get("voice_cast_mode") == "scorer", (
        f"expected voice_cast_mode 'scorer', got "
        f"{meta.get('voice_cast_mode')!r} -- a published episode cannot say "
        f"which caster produced its voices"
    )


def test_the_marker_is_stamped_on_every_lane_not_only_when_disabled(monkeypatch):
    """A marker that only appears in one state is not a marker. Opting back in
    must say so, so that turning the pass on is as visible as turning it off."""
    meta = _lock(monkeypatch, env_value="1")
    assert meta.get("voice_cast_mode") in ("hybrid", "hybrid_unavailable"), (
        f"opting in stamped {meta.get('voice_cast_mode')!r}; the marker must "
        f"report the hybrid lane when the hybrid lane is selected"
    )


def test_the_marker_is_never_silently_absent(monkeypatch):
    """TEETH. An absent key reads as false on every truthiness check a consumer
    might write, which is precisely the ambiguity this replaces."""
    meta = _lock(monkeypatch)
    assert "voice_cast_mode" in meta, "voice_cast_mode was not stamped at all"
    assert meta["voice_cast_mode"], "voice_cast_mode was stamped empty"


def test_the_writer_copies_the_marker_by_name_and_fails_closed():
    """The gate a review round caught: `lock_cast`'s meta is NOT merged
    wholesale into the ledger -- it is copied key by key, so a key not named
    there never reaches disk. Assert the copy exists AND that it does not
    invent a default, which would keep every acceptance check green while the
    marker was dead upstream.
    """
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert 'meta["voice_cast_mode"]' in src, (
        "the writer never copies voice_cast_mode, so it cannot reach the "
        "ledger no matter what lock_cast stamps"
    )
    assert 'cast_meta.get("voice_cast_mode", "")' in src, (
        "the writer's copy must be fail-CLOSED -- `.get(key, \"\")`. A default "
        "such as `or \"scorer\"` fabricates the marker when the upstream stamp "
        "is missing, so the gate passes while the thing it asserts is gone"
    )
