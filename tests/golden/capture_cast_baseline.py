"""tests/golden/capture_cast_baseline.py -- regenerate the pool-mode cast
golden baseline (S0 of the cast name<->gender<->voice coherence sprint).

Run from the repo root:
    .venv\\Scripts\\python.exe tests\\golden\\capture_cast_baseline.py

Writes tests/golden/cast_pool_baseline.json -- a deterministic cast captured
at a coherence-verified seed in pool mode. force_lemmy=False so the
SystemRandom LEMMY cameo (decoupled from the C7 seed by design) cannot perturb
determinism.

The cast STRUCTURE (char_id, name, gender, voice_preset) is the byte-identity
surface the R2 C7 test pins; character_description is LLM/stub output and is
deliberately excluded.

The baseline seed is the lowest seed whose rolled names are already
gender-coherent, so the S2 name-repair is a guaranteed no-op for it -- that is
exactly what lets R2 assert byte-identity across the fix. A known-INCOHERENT
seed (lowest seed with a binary name/gender mismatch) is recorded alongside so
the R3 coherence test has a real bug case to repair.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO, _REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_casting as C  # noqa: E402
from config import cast_pools as P  # noqa: E402

#: The lever the repair itself reads -- 1.0 keeps every deliberate
#: cross-gender name, exposing the RAW roll.
_CROSS_RATE_ENV = "OTR_NAME_CROSS_GENDER_RATE"

NUM = 4
NEWS = "Scientists report a quiet anomaly in the upper atmosphere over the test range."
STYLE = "1950s science fiction radio drama"


def _stub(messages, *, temperature, max_new_tokens):  # noqa: ARG001
    return json.dumps({
        "character_description":
            "A steady, weathered field operator who keeps the crew grounded.",
    })


def _structural(cast):
    return [
        {"char_id": r["char_id"], "name": r["name"],
         "gender": r["gender"], "voice_preset": r["voice_preset"]}
        for r in cast
    ]


def _open_rows(cast):
    return [r for r in cast if r["name"] not in ("ANNOUNCER", "LEMMY")]


def _mismatches(cast):
    out = []
    for r in _open_rows(cast):
        if r["gender"] in ("male", "female"):
            tag = P.gender_of_first_name(r["name"])
            if tag in ("male", "female") and tag != r["gender"]:
                out.append({"char_id": r["char_id"], "name": r["name"],
                            "slot_gender": r["gender"], "name_gender": tag})
    return out


def _lock(seed):
    """Lock a cast EXACTLY as `tests/test_cast_invariants.py::_lock` does.

    These two had drifted apart, which is why the captured baseline no longer
    matched the test that reads it. Two things were missing here:

    * `cast_seed=seed` -- the test passes it; without it the repair's isolated
      rng is keyed differently.
    * the VOICE REPLAY. The writer no longer stamps bark `voice_preset`;
      OTR_CastLock replays it post-freeze. The test applies that replay before
      asserting, so a baseline captured without it carries EMPTY voice presets
      and can never match, however correct its names and genders are.

    If the test's `_lock` changes again, this must follow it. The byte-identity
    surface is only meaningful while both sides build the cast the same way.
    """
    cast, meta = C.lock_cast(
        creative_fn=_stub, num_characters=NUM, news_seed=NEWS, style=STYLE,
        rng=__import__("random").Random(seed), cast_seed=seed,
        force_lemmy=False, max_attempts_per_call=1,
    )
    voices = C.replay_voice_assignment(
        cast_seed=seed, num_characters=NUM, lemmy_hit=meta["lemmy_hit"])
    for row in cast:
        if row.get("char_id") in voices:
            row["voice_preset"] = voices[row["char_id"]]
    return cast


def _raw_mismatches(seed):
    """Mismatches in the roll BEFORE the S2 repair touches it.

    `OTR_NAME_CROSS_GENDER_RATE=1.0` is the lever the repair itself reads: at
    1.0 every deliberate cross-gender name is kept, so what comes back is the
    raw roll. Restored afterwards so the caller's environment is unchanged.
    """
    prior = os.environ.get(_CROSS_RATE_ENV)
    os.environ[_CROSS_RATE_ENV] = "1.0"
    try:
        return _mismatches(_lock(seed))
    finally:
        if prior is None:
            os.environ.pop(_CROSS_RATE_ENV, None)
        else:
            os.environ[_CROSS_RATE_ENV] = prior


def main():
    # THE TWO SEEDS ARE FOUND UNDER DIFFERENT CONDITIONS, and conflating them
    # is why this script could no longer reproduce its own baseline.
    #
    # `_lock` runs the full `lock_cast`, which INCLUDES the S2 name-repair. So a
    # post-repair cast never contains a binary mismatch (the repair just fixed
    # them all), and the "lowest seed WITH a mismatch" search below could only
    # ever return None. It worked when this file was written at S0 because the
    # repair did not exist yet; the moment S2 landed, re-running this script
    # would have silently blanked `known_incoherent_seed` and left
    # `test_cross_gender_rate_controls_repair` indexing None.
    #
    # The incoherent seed is therefore hunted with the repair DISABLED, using
    # the same lever the test itself uses -- OTR_NAME_CROSS_GENDER_RATE=1.0
    # keeps every deliberate cross-gender name in place. That reveals the RAW
    # roll, which is what "known incoherent" has always meant.
    # BOTH seeds are judged on the RAW roll, and that is the whole subtlety.
    # Asking `_mismatches(_lock(seed))` post-repair returns [] for EVERY seed --
    # so "lowest coherent seed" would trivially answer 0 and R3's byte-identity
    # assertion would run at a seed where the repair actually fires, passing
    # only because both sides of the comparison go through it. Vacuous.
    coherent_seed = None
    incoherent_seed = None
    incoherent_detail = None
    for seed in range(0, 2000):
        raw = _raw_mismatches(seed)
        if coherent_seed is None and not raw:
            coherent_seed = seed
        if incoherent_seed is None and raw:
            incoherent_seed = seed
            incoherent_detail = raw
        if coherent_seed is not None and incoherent_seed is not None:
            break
    if incoherent_seed is None:
        raise RuntimeError(
            "no INCOHERENT seed found in range 0..2000 with the repair "
            "disabled -- test_cross_gender_rate_controls_repair needs a real "
            "bug case, and a baseline carrying None would make it index nothing"
        )
    if coherent_seed is None:
        raise RuntimeError("no coherent seed found in range 0..2000")

    cast = _lock(coherent_seed)
    if _structural(cast) != _structural(_lock(coherent_seed)):
        raise RuntimeError("capture is not deterministic at the chosen seed")

    golden = {
        "schema": "otr-cast-golden-v1",
        "mode": "pool",
        "num_characters": NUM,
        "force_lemmy": False,
        "news_seed": NEWS,
        "style": STYLE,
        "seed": coherent_seed,
        "cast": _structural(cast),
        "known_incoherent_seed": incoherent_seed,
        "known_incoherent_detail": incoherent_detail,
    }
    (_HERE / "cast_pool_baseline.json").write_text(
        json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    return golden


if __name__ == "__main__":
    # Report to STDOUT. This used to write to a hardcoded
    # C:\...\Temp\otr_s0\capture_out.txt, a directory nothing creates -- so the
    # script raised FileNotFoundError on its SUCCESS path, and then its own
    # except branch raised the same error again while trying to record the
    # failure. The baseline had already been written by then, so the run looked
    # like a crash while having actually succeeded, which is the worst of both.
    g = main()
    print("OK coherent_seed=%s incoherent_seed=%s" % (
        g["seed"], g["known_incoherent_seed"]))
    print("cast=%s" % json.dumps(g["cast"], indent=2))
    print("incoherent=%s" % json.dumps(g["known_incoherent_detail"], indent=2))
