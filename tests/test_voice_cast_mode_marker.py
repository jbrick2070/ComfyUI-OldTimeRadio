"""Every episode says WHICH caster chose its voices, and there is only one.

The hybrid LLM voice-fit was disabled on 2026-08-18 and RIPPED the same day. The
deterministic scorer in `_otr_voice_bank.assign_voice_for_slot` is now the only
thing that casts a drawn character voice.

WHY THE MARKER SURVIVES THE RIP. `meta.voice_cast_mode` was added to make the
FLIP detectable, and it would be easy to argue it is redundant now that only one
caster exists. It is not. A published episode should state which caster produced
its voices rather than leaving a reader to infer it from an absent field, and the
acceptance instrument `scripts/otr_verify_voice_cast_mode.py` gates on it. If a
second caster is ever introduced again, this marker is the thing that will make
the difference visible on day one instead of after a corpus measurement.

WHY THE WRITER COPY IS TESTED SEPARATELY. `lock_cast` stamping the marker is not
enough: `OTR_LedgerScriptWriter` copies lock_cast's meta KEY BY KEY, and states
the invariant in its own comment -- "a key stamped in lock_cast and not named on
this line never reaches the ledger". A review round caught that the marker would
have been dropped there, which would have made the whole detectability fix
silently useless. The copy must also be FAIL-CLOSED, or a missing upstream stamp
gets papered over with a fabricated default.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    for p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from nodes import _otr_casting as _OTRC  # noqa: E402

#: Symbols the 2026-08-18 rip removed. If any comes back, the pass is back with
#: it, and the concentration it caused comes back too.
RIPPED_SYMBOLS = (
    "hybrid_voice_fit_enabled",
    "_build_voice_fit_prompt",
    "llm_propose_voice_ref",
)
RIPPED_FROM_VOICE_BANK = (
    "build_voice_cards",
    "validate_voice_proposal",
    "VOICE_FIT_POLICY_VERSION",
)


def _lock():
    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        return '{"character_description": "a voice on the wire"}'

    _cast, meta = _OTRC.lock_cast(
        creative_fn=fn, num_characters=2, news_seed="x", style="open",
        rng=random.Random(7), cast_seed=7, force_lemmy=False,
    )
    return meta


def test_the_ledger_says_the_scorer_cast_it():
    """The assertion the live-leg acceptance gate depends on."""
    meta = _lock()
    assert meta.get("voice_cast_mode") == "scorer", (
        f"expected voice_cast_mode 'scorer', got {meta.get('voice_cast_mode')!r} "
        f"-- a published episode cannot say which caster produced its voices"
    )


def test_the_marker_is_never_silently_absent():
    """TEETH. An absent key reads as false on every truthiness check a consumer
    might write, which is exactly the ambiguity the marker replaced."""
    meta = _lock()
    assert "voice_cast_mode" in meta, "voice_cast_mode was not stamped at all"
    assert meta["voice_cast_mode"], "voice_cast_mode was stamped empty"


def test_the_decision_dict_is_still_stamped_and_empty():
    """The ripped pass's field is KEPT, empty, on purpose. CastLock still reads
    `meta.get("voice_cast_decision") or {}`, and every published ledger should
    keep one stable shape rather than gaining and losing a key across the rip."""
    meta = _lock()
    assert meta.get("voice_cast_decision") == {}, (
        f"expected an empty voice_cast_decision, got "
        f"{meta.get('voice_cast_decision')!r}"
    )


def test_the_hybrid_voice_fit_is_gone_and_stays_gone():
    """A guard against reintroduction, not a tautology.

    The pass was removed because its prompt gave the model exactly the four
    fields `_score()` already weights and no character name -- so it had no
    judgment available to it -- while casting with 13 distinct voices at 96%
    top-5 where the scorer uses 43 at 25%. If these symbols reappear, that
    concentration reappears with them.
    """
    from nodes import _otr_voice_bank as VB

    back = [s for s in RIPPED_SYMBOLS if hasattr(_OTRC, s)]
    back += [s for s in RIPPED_FROM_VOICE_BANK if hasattr(VB, s)]
    assert not back, (
        f"the hybrid LLM voice-fit is back: {back}. It was ripped on 2026-08-18 "
        f"for casting with 13 voices where the scorer uses 43. If this is "
        f"deliberate, the concentration measurement needs redoing first."
    )


def test_the_writer_copies_the_marker_by_name_and_fails_closed():
    """The gate a review round caught. `lock_cast`'s meta is NOT merged wholesale
    into the ledger -- it is copied key by key, so a key not named there never
    reaches disk. Assert the copy exists AND that it invents no default, which
    would keep every acceptance check green while the marker was dead upstream.
    """
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert 'meta["voice_cast_mode"]' in src, (
        "the writer never copies voice_cast_mode, so it cannot reach the ledger "
        "no matter what lock_cast stamps"
    )
    assert 'cast_meta.get("voice_cast_mode", "")' in src, (
        'the writer\'s copy must be fail-CLOSED -- `.get(key, "")`. A default '
        'such as `or "scorer"` fabricates the marker when the upstream stamp is '
        'missing, so the gate passes while the thing it asserts is gone'
    )
