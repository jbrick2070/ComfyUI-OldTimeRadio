"""Ask a local model for a random name, over and over, and count the repeats.

THE QUESTION. `rng.choice(FIRST_NAMES)` over 154 names is genuinely uniform --
every name is equally likely, forever. A language model is not a random number
generator: asked for "a random name" it returns its FAVOURITE answers, and the
same favourites every time. The historical complaint about the LLM naming lane
was that it "always picked the same name". This measures exactly how true that
is, on this box, on these weights.

WHAT IS MEASURED. N independent asks with an identical prompt, then: how many
DISTINCT names came back, how often the most common one repeated, and how many
names it took to cover half the answers. A uniform sampler over even 20 names
would return ~N distinct at small N; a model with three favourites will not.

THE VARIANTS ARE THE POINT, because a bad number on the naive prompt does not
prove the approach is hopeless -- it may only prove the prompt gave the model
nothing to vary on:

  plain   -- "give me a random first name", nothing else. The baseline.
  seeded  -- the same prompt plus a per-call variation key. Costs nothing and
             stays reproducible, since the key is derived, not random.
  avoid   -- the same prompt plus the names already returned this run, which
             attacks recurrence directly rather than hoping for variety.

NO ERA, NO GENRE, NO STYLE ANCHOR. Deliberately. Telling the model "1950s radio
drama" narrows it to a period name list, which is a different question and a
narrower pool -- the operator asked for a random NAME, not a random name from a
costume drama.

DIAGNOSTIC ONLY. Writes no pool, no sidecar, nothing the render path reads.
Local weights only. Run with no ComfyUI server holding VRAM (section 4).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RandomName(BaseModel):
    first_name: str = Field(..., max_length=40)
    gender: str = Field(..., pattern="^(male|female)$")


def _messages(mode: str, i: int, seen: list) -> list:
    ask = ("Give me one random first name for a person, and that person's "
           "gender.")
    if mode == "seeded":
        # A DERIVED key, not randomness: the run is still reproducible, the
        # model just gets something different to condition on each call.
        ask += f"\nVariation key: {i * 7919 % 100003}."
    elif mode == "avoid" and seen:
        recent = ", ".join(seen[-40:])
        ask += (f"\nDo NOT use any of these already-used names: {recent}.")
    return [
        {"role": "system", "content":
            "You produce a single random personal name. Vary your answers "
            "widely across the whole range of human names -- do not fall back "
            "on the same handful of favourites."},
        {"role": "user", "content": ask},
    ]


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--modes", default="plain,seeded,avoid")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" /
                                         "name_randomness_lab.json"))
    args = ap.parse_args(argv)

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    print(f"[rand] loading {args.model} ...", flush=True)
    t0 = time.time()
    entry = LOADER.load_llm(args.model, optimization_profile="Standard")
    gen = make_constrained_generate_fn(entry, RandomName)
    print(f"[rand] loaded in {time.time()-t0:.1f}s; {args.n} asks per mode "
          f"at temperature {args.temperature}\n", flush=True)

    report = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        seen, genders = [], collections.Counter()
        for i in range(args.n):
            raw = gen(_messages(mode, i, seen), temperature=args.temperature,
                      max_new_tokens=64)
            try:
                parsed = RandomName.model_validate_json(raw)
            except Exception:  # noqa: BLE001
                continue
            seen.append(parsed.first_name.strip().title())
            genders[parsed.gender] += 1
        counts = collections.Counter(seen)
        distinct = len(counts)
        top = counts.most_common(5)
        # How many DIFFERENT names it takes to account for half the answers.
        half, acc = 0, 0
        for _n, c in counts.most_common():
            acc += c
            half += 1
            if acc >= len(seen) / 2:
                break
        report[mode] = {
            "asks": len(seen), "distinct": distinct,
            "distinct_pct": round(100.0 * distinct / max(1, len(seen)), 1),
            "top5": top, "names_covering_half": half,
            "gender_split": dict(genders),
        }
        print(f"  {mode:7s} {len(seen):3d} asks -> {distinct:3d} distinct "
              f"({report[mode]['distinct_pct']:5.1f}%)   "
              f"half the answers come from {half} name(s)")
        print(f"          most common: "
              f"{', '.join(f'{n} x{c}' for n, c in top)}")
        print(f"          gender split: {dict(genders)}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model,
                               "temperature": args.temperature,
                               "modes": report}, indent=2, ensure_ascii=False)
                   + "\n", encoding="utf-8")
    print(f"\n[rand] report -> {out}")
    print("[rand] For comparison, rng.choice over the 154-name pool would give "
          "roughly 100% distinct at this sample size, and an even gender split "
          "only because the pool is tagged -- a model has no such guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
