"""Score a local model on characters whose gender is NOT in doubt, blind.

BEFORE TRUSTING AN INSTRUMENT, MEASURE IT. Asking a model to second-guess 132
shipped pins is worthless until we know how often the model itself is right. So
this hands it characters from our own corpus whose gender is a matter of public
record -- Ahab, Elizabeth Bennet, Heathcliff, Dorothy -- and scores the answers.

BLIND, strictly. The prompt carries the work, the author and the character name,
and NOTHING about what the pipeline assigned or what this file expects. The
model cannot agree with us by reading our answer over our shoulder.

THE HARD HALF IS THE POINT. The control set deliberately includes rows where the
CORRECT answer is "unsure" -- a collective ("a Han patrol", "the crew"), an
unnamed narrator, and an invented character from a story written for this show
that no model can have read. A model that confidently genders those is not
knowledgeable, it is agreeable, and its opinion on the other 130 is worth
nothing. Scoring only the easy rows would measure the wrong thing.

DIAGNOSTIC ONLY. Writes no sidecar, no pool, nothing the render path reads.
Local weights only. Run with no ComfyUI server holding VRAM (section 4).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(HERE))
from otr_gender_secondopinion_lab import (  # noqa: E402
    CharacterGenderOpinion, _ask,
)

#: (work, author, character, truth). "unsure" means a correct model DECLINES.
#: Truths here are public record, not this project's opinion -- the point is a
#: yardstick that exists independently of anything we assigned.
CONTROL = [
    # --- unambiguous men -------------------------------------------------
    ("Moby-Dick", "Herman Melville", "Captain Ahab", "male"),
    ("Frankenstein", "Mary Shelley", "Victor Frankenstein", "male"),
    ("Wuthering Heights", "Emily Bronte", "Heathcliff", "male"),
    ("The Count of Monte Cristo", "Alexandre Dumas", "Edmond Dantes", "male"),
    ("Don Quixote", "Miguel de Cervantes", "Sancho Panza", "male"),
    ("Adventures of Huckleberry Finn", "Mark Twain", "Huck Finn", "male"),
    ("A Christmas Carol", "Charles Dickens", "Bob Cratchit", "male"),
    ("The Adventure of the Speckled Band", "Arthur Conan Doyle",
     "Dr. Watson", "male"),
    ("Treasure Island", "Robert Louis Stevenson", "Jim Hawkins", "male"),
    ("The Time Machine", "H. G. Wells", "the Time Traveller", "male"),
    # --- unambiguous women -----------------------------------------------
    ("Pride and Prejudice", "Jane Austen", "Elizabeth Bennet", "female"),
    ("Jane Eyre", "Charlotte Bronte", "Jane Eyre", "female"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll", "Alice", "female"),
    ("The Wonderful Wizard of Oz", "L. Frank Baum", "Dorothy", "female"),
    ("Anne of Green Gables", "L. M. Montgomery", "Anne Shirley", "female"),
    ("Anne of Green Gables", "L. M. Montgomery", "Marilla Cuthbert", "female"),
    ("The Open Window", "Saki", "Mrs. Sappleton", "female"),
    ("Desiree's Baby", "Kate Chopin", "Desiree", "female"),
    ("The Adventure of the Speckled Band", "Arthur Conan Doyle",
     "Helen Stoner", "female"),
    ("Nonsense Novels", "Stephen Leacock", "Gertrude", "female"),
    # --- the hard half: a correct model says "unsure" ---------------------
    ("Armageddon 2419 A.D.", "Philip Francis Nowlan", "a Han patrol", "unsure"),
    ("Moby-Dick", "Herman Melville", "the crew", "unsure"),
    ("Told After Supper", "Jerome K. Jerome", "the narrator", "unsure"),
    ("The Cradle Protocol", "", "ARIA", "unsure"),
    ("The Cradle Protocol", "", "Dr. Lira Kell", "unsure"),
]


class ForcedGenderOpinion(BaseModel):
    """The SAME question with "unsure" removed from the vocabulary.

    Grammar-constrained decoding binds the output to this schema at the token
    level, so the model physically CANNOT decline -- it is not being persuaded
    to answer, it is being prevented from refusing. That is what makes the
    comparison honest: any difference against the three-way run is caused by
    the missing escape hatch and nothing else.
    """

    gender: str = Field(..., pattern="^(male|female)$")
    reason: str = Field(..., max_length=200)


class PercentGenderOpinion(BaseModel):
    """Ask for a NUMBER instead of a verdict.

    A binary answer throws away the only thing worth knowing on a hard row.
    "Dr. Lira Kell: female" and "Dr. Lira Kell: 55% female" are the same verdict
    and completely different claims -- the first is indistinguishable from
    knowledge, the second admits a coin flip. The question is whether the model
    will USE the range honestly, or answer 95 for everything because a confident
    number reads better than an uncertain one.
    """

    percent_male: float = Field(..., ge=0.0, le=100.0)
    reason: str = Field(..., max_length=200)


def _percent_run(gen, args) -> int:
    """Score how well the model's stated confidence tracks what it can know."""
    rows, known_conf, hard_conf = [], [], []
    for work, author, name, truth in CONTROL:
        messages = [
            {"role": "system", "content":
                "You answer questions about characters in published "
                "literature, and you are honest about uncertainty. Use the "
                "full range: 50 means you genuinely cannot tell."},
            {"role": "user", "content": (
                f'In "{work}"{f" by {author}" if author else ""}, what '
                f'percentage likely is it that the character "{name}" is '
                f"male?\n"
                "Answer percent_male from 0 to 100. 100 means certainly male, "
                "0 means certainly female, 50 means you cannot tell. If you do "
                "not know this work or character, answer close to 50."
            )},
        ]
        # 256, not 140. At 140 the JSON was TRUNCATED mid-`reason` on exactly
        # the rows whose reason ran long, so they came back unparseable and the
        # first run of this mode reported a meaningless separation computed over
        # corrupted rows. The raw is kept on failure so the next person sees
        # what actually came back instead of guessing at it.
        raw = gen(messages, temperature=0.2, max_new_tokens=256)
        bad = ""
        try:
            got = PercentGenderOpinion.model_validate_json(raw).percent_male
        except Exception:  # noqa: BLE001
            got = None
            bad = raw[-90:]
        # Confidence = distance from a coin flip, 0..50.
        conf = None if got is None else abs(got - 50.0)
        side = None if got is None else ("male" if got >= 50 else "female")
        rows.append({"name": name, "truth": truth, "percent_male": got,
                     "confidence": conf, "side": side,
                     **({"unparseable_tail": bad} if bad else {})})
        if truth in ("male", "female"):
            known_conf.append(conf or 0)
            mark = "ok  " if side == truth else "WRONG"
        else:
            hard_conf.append(conf or 0)
            mark = "hard"
        print(f"  {mark:5s} {name:22s} truth={truth:7s} "
              f"{'--' if got is None else f'{got:5.1f}% male'}")

    kn = sum(known_conf) / max(1, len(known_conf))
    hd = sum(hard_conf) / max(1, len(hard_conf))
    print(f"\n[control] PERCENTAGE run")
    print(f"[control] mean confidence on KNOWN characters   : {kn:.1f}/50")
    print(f"[control] mean confidence on UNGENDERABLE rows  : {hd:.1f}/50")
    print(f"[control] separation (higher is better calibrated): {kn - hd:.1f}")
    print("\n[control] THIS NUMBER IS THE WHOLE EXPERIMENT. A large separation "
          "means the model KNOWS what it does not know, and a confidence "
          "threshold would be a usable gate. A small one means it is equally "
          "sure of Ahab and of a character invented last week -- in which case "
          "the percentage is decoration and only quoting the source text can "
          "tell the two apart.")
    out = pathlib.Path(args.out).with_name("gender_blind_control_percent.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "mode": "percent",
                               "mean_confidence_known": kn,
                               "mean_confidence_hard": hd,
                               "rows": rows}, indent=2, ensure_ascii=False)
                   + "\n", encoding="utf-8")
    print(f"[control] report -> {out}")
    return 0


def _forced_run(gen, args) -> int:
    """Ask twice with no escape hatch, and report which answers held.

    On a row whose truth is genuinely unknowable, a STABLE answer and a FLIPPED
    answer mean very different things. Stable means the model has a consistent
    prior it will apply every time -- durable, and durably wrong if the prior is
    wrong. Flipped means it is sampling, and the "answer" is a coin flip wearing
    a citation. Either way it is not knowledge, but only one of them looks like
    noise from the outside.
    """
    rows = []
    easy_right = hard_stable = hard_flipped = 0
    for work, author, name, truth in CONTROL:
        first = _ask(gen, name, work, author)["gender"]
        second = _ask(gen, name, work, author)["gender"]
        stable = (first == second)
        rows.append({"work": work, "name": name, "truth": truth,
                     "first": first, "second": second, "stable": stable})
        if truth in ("male", "female"):
            hit = (first == truth)
            easy_right += hit
            mark = "ok  " if hit else "WRONG"
        else:
            hard_stable += stable
            hard_flipped += (not stable)
            mark = "forced"
        flag = "" if stable else "   <-- FLIPPED on re-ask"
        print(f"  {mark:6s} {name:22s} truth={truth:7s} "
              f"{first}/{second}{flag}")

    easy = [r for r in rows if r["truth"] != "unsure"]
    hard = [r for r in rows if r["truth"] == "unsure"]
    print(f"\n[control] FORCED run -- 'unsure' removed from the schema")
    print(f"[control] known characters : {easy_right}/{len(easy)} correct")
    print(f"[control] ungenderable rows: {len(hard)} forced answers, "
          f"{hard_stable} stable / {hard_flipped} flipped on a re-ask")
    print("\n[control] A FORCED ANSWER ON AN UNGENDERABLE ROW IS NOT AN ANSWER. "
          "Stable means a durable prior that will be wrong the same way "
          "forever; flipped means it was sampling. Neither is knowledge, and a "
          "sidecar cannot tell either from the real thing.")
    out = pathlib.Path(args.out).with_name("gender_blind_control_forced.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "forced": True,
                               "rows": rows}, indent=2, ensure_ascii=False)
                   + "\n", encoding="utf-8")
    print(f"[control] report -> {out}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--percent", action="store_true",
                    help="ask for a 0-100 percentage instead of a verdict, and "
                         "report how well stated confidence separates known "
                         "characters from ungenderable ones")
    ap.add_argument("--no-decline", action="store_true",
                    help="remove 'unsure' from the schema so the model MUST "
                         "answer, and re-ask each row to see whether the "
                         "forced answer is stable or a coin flip")
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" /
                                         "gender_blind_control.json"))
    args = ap.parse_args(argv)

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    print(f"[control] loading {args.model} ...", flush=True)
    t0 = time.time()
    entry = LOADER.load_llm(args.model, optimization_profile="Standard")
    if args.percent:
        schema = PercentGenderOpinion
        mode = "PERCENTAGE (0-100 male)"
    elif args.no_decline:
        schema = ForcedGenderOpinion
        mode = "FORCED (no decline allowed)"
    else:
        schema = CharacterGenderOpinion
        mode = "decline allowed"
    gen = make_constrained_generate_fn(entry, schema)
    print(f"[control] loaded in {time.time()-t0:.1f}s; "
          f"{len(CONTROL)} control rows; {mode}\n", flush=True)

    if args.percent:
        return _percent_run(gen, args)
    if args.no_decline:
        return _forced_run(gen, args)

    rows, right, wrong, overconfident, timid = [], 0, 0, 0, 0
    for work, author, name, truth in CONTROL:
        got = _ask(gen, name, work, author)["gender"]
        ok = (got == truth)
        if ok:
            right += 1
        elif truth == "unsure":
            overconfident += 1          # gendered something ungenderable
        elif got == "unsure":
            timid += 1                  # declined a character it should know
        else:
            wrong += 1                  # said male for a woman, or vice versa
        rows.append({"work": work, "name": name, "truth": truth, "got": got})
        mark = "ok  " if ok else ("OVER" if truth == "unsure"
                                  else ("timid" if got == "unsure" else "WRONG"))
        print(f"  {mark:5s} {name:22s} truth={truth:7s} model={got}")

    total = len(CONTROL)
    hard = [r for r in rows if r["truth"] == "unsure"]
    easy = [r for r in rows if r["truth"] != "unsure"]
    easy_right = sum(1 for r in easy if r["got"] == r["truth"])
    hard_right = sum(1 for r in hard if r["got"] == "unsure")

    print(f"\n[control] overall {right}/{total}")
    print(f"[control] known characters : {easy_right}/{len(easy)} correct")
    print(f"[control] should-be-unsure : {hard_right}/{len(hard)} declined "
          f"({overconfident} confidently gendered something ungenderable)")
    print(f"[control] flat-out wrong   : {wrong}   too timid: {timid}")
    print("\n[control] READ THIS BEFORE TRUSTING THE SECOND-OPINION RUN: a model "
          "that scores well on known characters but confidently genders the "
          "hard rows is agreeable, not knowledgeable -- and agreeableness is "
          "exactly the failure that puts a wrong gender in a sidecar.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "rows": rows}, indent=2, ensure_ascii=False
    ) + "\n", encoding="utf-8")
    print(f"[control] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
