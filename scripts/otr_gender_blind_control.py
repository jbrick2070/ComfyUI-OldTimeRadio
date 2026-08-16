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


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" /
                                         "gender_blind_control.json"))
    args = ap.parse_args(argv)

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    print(f"[control] loading {args.model} ...", flush=True)
    t0 = time.time()
    entry = LOADER.load_llm(args.model, optimization_profile="Standard")
    gen = make_constrained_generate_fn(entry, CharacterGenderOpinion)
    print(f"[control] loaded in {time.time()-t0:.1f}s; "
          f"{len(CONTROL)} control rows\n", flush=True)

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
