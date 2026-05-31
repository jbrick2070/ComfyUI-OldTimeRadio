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
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO, _REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_casting as C  # noqa: E402
from config import cast_pools as P  # noqa: E402

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
    cast, _meta = C.lock_cast(
        creative_fn=_stub, num_characters=NUM, news_seed=NEWS, style=STYLE,
        rng=__import__("random").Random(seed),
        force_lemmy=False, max_attempts_per_call=1,
    )
    return cast


def main():
    coherent_seed = None
    incoherent_seed = None
    incoherent_detail = None
    for seed in range(0, 2000):
        cast = _lock(seed)
        ms = _mismatches(cast)
        if not ms and coherent_seed is None:
            coherent_seed = seed
        if ms and incoherent_seed is None:
            incoherent_seed = seed
            incoherent_detail = ms
        if coherent_seed is not None and incoherent_seed is not None:
            break
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
    dbg = Path(r"C:\Users\jeffr\AppData\Local\Temp\otr_s0\capture_out.txt")
    try:
        g = main()
        dbg.write_text(
            "OK coherent_seed=%s incoherent_seed=%s\ncast=%s\nincoherent=%s\n" % (
                g["seed"], g["known_incoherent_seed"],
                json.dumps(g["cast"], indent=2),
                json.dumps(g["known_incoherent_detail"], indent=2)),
            encoding="utf-8")
    except Exception:
        dbg.write_text("ERROR\n" + traceback.format_exc(), encoding="utf-8")
        raise
