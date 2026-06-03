"""Sprint 2 (a) parity mandate -- OTR_CastLock voice assignment == the writer's
historical bark voice picker, byte-identical per character.

The writer no longer stamps bark ``voice_preset`` -- it persists ``cast_seed`` and
OTR_CastLock owns bark casting via ``replay_voice_assignment``, a pure replay of
the writer's deterministic picker keyed on that ``cast_seed``. These tests PIN
that the CastLock-stamped voice per character is BYTE-IDENTICAL to what the writer
used to assign: ``test_replay_matches_committed_golden_baseline`` anchors the
replay to a committed pre-move writer snapshot (history), and the lock_cast tests
prove the live writer leaves voices empty AND the real CastLock node stamps the
replay char-for-char. Without this, moving the assignment to CastLock could
silently change which voice each character gets (option (b)) instead of preserving
it (option (a)). The replay keys on the WRITER's cast_seed -- NOT
``_otr_voice_bank.stable_cast_seed`` (the flagged trap).
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


def _castlock_assign(cast, cast_seed, num, lemmy_hit):
    """Run the REAL OTR_CastLock bark-voice assignment (what the node does after
    the freeze) on a cast and return {char_id: voice_preset} for the bark rows.
    Builds the meta.cast_contract the writer persists. Lazy import so the test
    stays free of ComfyUI module-load deps."""
    from nodes.cast_lock import CastLock

    meta = {"cast_contract": {
        "cast_seed": cast_seed, "num_characters_request": num,
        "lemmy_hit": lemmy_hit}}
    CastLock._assign_bark_voices(cast, meta, [])
    return _writer_voices(cast)


def test_replay_matches_lock_cast_byte_identical():
    """Sprint 2 (a): the writer leaves bark voices EMPTY; OTR_CastLock stamps each
    character the byte-identical voice the writer used to assign -- proven
    char-for-char against the replay (the parity mandate, now load-bearing)."""
    cast_seed, num = 1234567, 5
    cast, meta = _lock(cast_seed, num)
    open_rows = [r for r in cast if r["name"] not in ("ANNOUNCER", "LEMMY")]
    assert open_rows and all(r["voice_preset"] == "" for r in open_rows), \
        "the writer must leave open-character voice_preset empty (CastLock owns it)"
    stamped = _castlock_assign(cast, cast_seed, num, meta["lemmy_hit"])
    replay = replay_voice_assignment(
        cast_seed=cast_seed, num_characters=num, lemmy_hit=meta["lemmy_hit"],
    )
    assert stamped == replay, (
        f"CastLock-stamped voices diverged from the replay:\n  stamped={stamped}\n"
        f"  replay={replay}"
    )


def test_replay_matches_across_seeds_and_counts():
    """Parity holds across seeds + character counts (not one lucky fixture)."""
    for cast_seed in (1, 42, 777, 100003):
        for num in (1, 3, 6):
            cast, meta = _lock(cast_seed, num)
            stamped = _castlock_assign(cast, cast_seed, num, meta["lemmy_hit"])
            replay = replay_voice_assignment(
                cast_seed=cast_seed, num_characters=num, lemmy_hit=meta["lemmy_hit"],
            )
            assert stamped == replay, (cast_seed, num)


def test_replay_matches_with_lemmy_present():
    """LEMMY changes the cast structure (one pre-locked Bark voice + one fewer
    open slot); CastLock's stamping must reproduce it given the persisted
    lemmy_hit."""
    cast_seed, num = 9001, 4
    cast, meta = _lock(cast_seed, num, force_lemmy=True)
    assert meta["lemmy_hit"] is True
    stamped = _castlock_assign(cast, cast_seed, num, True)
    replay = replay_voice_assignment(
        cast_seed=cast_seed, num_characters=num, lemmy_hit=True,
    )
    assert stamped == replay


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
