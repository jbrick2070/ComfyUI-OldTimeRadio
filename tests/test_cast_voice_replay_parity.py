"""Sprint 2 (a) parity mandate -- CastLock voice replay == writer lock_cast.

OTR_CastLock will own bark casting via ``replay_voice_assignment`` -- a pure
replay of the writer's deterministic voice picker keyed on the writer's
``cast_seed``. These tests PIN that the replay reproduces ``lock_cast``'s
``voice_preset`` per character BYTE-IDENTICALLY for the same cast_seed. Without
this, moving the assignment to CastLock would silently change which voice each
character gets (option (b)), not preserve it (option (a)). The replay keys on the
WRITER's cast_seed -- NOT ``_otr_voice_bank.stable_cast_seed`` (the flagged trap).
"""
from __future__ import annotations

import json
import pathlib
import random

from nodes._otr_casting import lock_cast, replay_voice_assignment

_GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "cast_pool_baseline.json"


def _stub_description(messages, *, temperature, max_new_tokens):  # noqa: ARG001
    # cast_one_character -> llm_write_description; this step draws NO cast rng,
    # so the stubbed LLM does not perturb the voice-assignment sequence.
    return json.dumps({"character_description": "a steady, weathered operator"})


def _writer_voices(cast):
    return {r["char_id"]: r["voice_preset"]
            for r in cast if r["name"] != "ANNOUNCER"}


def _lock(cast_seed, num, *, force_lemmy=False):
    return lock_cast(
        creative_fn=_stub_description, num_characters=num,
        news_seed="a quiet anomaly over the test range",
        style="closed_room_suspense", rng=random.Random(cast_seed),
        cast_seed=cast_seed, force_lemmy=force_lemmy, max_attempts_per_call=1,
    )


def test_replay_matches_lock_cast_byte_identical():
    """The replay reproduces the writer's lock_cast voice_preset per character,
    char-for-char (the parity mandate)."""
    cast_seed, num = 1234567, 5
    cast, meta = _lock(cast_seed, num)
    replay = replay_voice_assignment(
        cast_seed=cast_seed, num_characters=num, lemmy_hit=meta["lemmy_hit"],
    )
    assert replay == _writer_voices(cast), (
        f"voice replay diverged from the writer:\n  writer={_writer_voices(cast)}\n"
        f"  replay={replay}"
    )


def test_replay_matches_across_seeds_and_counts():
    """Parity holds across seeds + character counts (not one lucky fixture)."""
    for cast_seed in (1, 42, 777, 100003):
        for num in (1, 3, 6):
            cast, meta = _lock(cast_seed, num)
            replay = replay_voice_assignment(
                cast_seed=cast_seed, num_characters=num, lemmy_hit=meta["lemmy_hit"],
            )
            assert replay == _writer_voices(cast), (cast_seed, num)


def test_replay_matches_with_lemmy_present():
    """LEMMY changes the cast structure (one pre-locked Bark voice + one fewer
    open slot); the replay must reproduce it given the persisted lemmy_hit."""
    cast_seed, num = 9001, 4
    cast, meta = _lock(cast_seed, num, force_lemmy=True)
    assert meta["lemmy_hit"] is True
    replay = replay_voice_assignment(
        cast_seed=cast_seed, num_characters=num, lemmy_hit=True,
    )
    assert replay == _writer_voices(cast)


def test_replay_matches_committed_golden_baseline():
    """The committed pool-mode golden cast baseline (a real writer snapshot) is
    reproduced voice-for-voice by the replay."""
    g = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    golden_voices = {r["char_id"]: r["voice_preset"]
                     for r in g["cast"] if r["name"] != "ANNOUNCER"}
    replay = replay_voice_assignment(
        cast_seed=g["seed"], num_characters=g["num_characters"],
        lemmy_hit=bool(g.get("force_lemmy", False)),
    )
    assert replay == golden_voices, (golden_voices, replay)


def test_replay_keys_on_cast_seed_not_a_fixed_constant():
    """Guard the flagged trap: different cast_seeds must yield different
    assignments, so the replay genuinely keys on the writer's cast_seed."""
    a = replay_voice_assignment(cast_seed=1, num_characters=5, lemmy_hit=False)
    b = replay_voice_assignment(cast_seed=2, num_characters=5, lemmy_hit=False)
    assert a != b


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
