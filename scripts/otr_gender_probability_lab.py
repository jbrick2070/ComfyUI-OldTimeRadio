"""Ask several LOCAL models how gendered a given name is, and disagree loudly.

WHAT THIS IS FOR. The name pool asserts a gender for all 153 first names -- an
assertion the coherence repair acts on, since a name tagged male on a female
slot gets swapped. 30 of those tags were assigned on 2026-08-15 by one person's
judgment when the "unisex" bucket was retired, and several were flagged at the
time as calls the author could not defend: the Thai names (Krit, Niran, Sunan,
Dao, Pim, Som), the Yoruba and Igbo ones (Ayo, Chidi), and the surname-style
English ones (Stone, Hayes, Carter, Palmer). This measures those tags instead of
trusting them.

WHY A PROBABILITY AND NOT A VERDICT. "Shirley: 90% female" carries information
that "Shirley: female" throws away. A name at 0.55 is a coin flip wearing a
label, and the pool has no way to say so today. The number is what lets a reader
tell a confident tag from a lucky one.

WHY SEVERAL MODELS. A model asked "how sure are you" is badly calibrated and
will happily answer 95% and be wrong. AGREEMENT BETWEEN DIFFERENT MODEL FAMILIES
is a real signal and costs nothing to compute: where three models independently
land on the same side, that is evidence; where they split, the name is genuinely
ambiguous and belongs in neither binary bucket. Disagreement is the finding, not
a failure of the run.

WHAT IT IS NOT. It is a LAB: diagnostic only. It never writes `cast_pools.py`,
never writes a sidecar, and nothing in the render path reads its output. It
prints a table and a JSON report for a human to read and act on. A name
probability is also NOT a character's gender -- Ahab is male because Melville
writes "he", not because the name skews male -- so this speaks to the NAME POOL,
where "what share of people with this name are male" is exactly the question the
tag encodes. Pointing it at story characters would answer a different question
than the one being asked.

NO WEB, NO CLOUD, NO API KEY. Local weights only, one model resident at a time.
Run it with no ComfyUI server holding VRAM (CLAUDE.md section 4).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cast_pools as POOLS  # noqa: E402

#: Bump when the QUESTION changes -- a report built on a different prompt is not
#: comparable with an older one, and a lab whose runs cannot be compared is a
#: lab that proves nothing.
PROMPT_VERSION = "gender_probability_v2_global_aggregate"

#: Default panel: three genuinely different local families, smallest first so a
#: broken harness is discovered cheaply rather than after a 24 GB load.
DEFAULT_MODELS = (
    "google/gemma-2-2b-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12b-it",
)

#: A tag is only CONTESTED when the panel is confident the other way. Models
#: hovering near 0.5 are saying "this name is genuinely mixed", which is a
#: finding about the NAME, not a complaint about the tag.
CONFIDENT = 0.70


class NameGenderProbability(BaseModel):
    """One model's read on one name."""

    share_male: float = Field(..., ge=0.0, le=1.0)
    basis: str = Field(..., max_length=200)


def _prompt(name: str) -> str:
    """ONE GLOBAL NUMBER. No country, no culture, no hedging.

    The first version of this prompt asked the model to "state the basis -- the
    language or culture it comes from". That instruction shaped the answers
    badly: the model went looking for a CULTURE first, and when it could not
    identify one it returned exactly 0.50 with a basis of "No specific cultural
    data found" -- 18 names came back that way, including ordinary English ones
    like Stone, Hayes and Kelly that it plainly has an opinion about. It was
    declining to name a culture, not declining to gender the name, and the
    prompt could not tell the difference.

    So: ask for the WORLDWIDE aggregate across everyone alive who carries the
    name, and say explicitly that a per-country split is not what is wanted.
    0.5 now means "genuinely used equally", not "I could not place it".
    """
    return (
        f'Worldwide, across everyone who has the given name "{name}", what '
        f"share of them are male?\n"
        "Answer share_male as ONE overall number between 0.0 and 1.0, where "
        "1.0 means essentially all male and 0.0 means essentially all female.\n"
        "Give a single GLOBAL aggregate. Do NOT break the answer down by "
        "country, culture or era, and do NOT answer 0.5 merely because the "
        "name is used in more than one language -- estimate the overall "
        "balance across all of them.\n"
        "Use 0.5 ONLY when the name is genuinely close to evenly split "
        "between men and women worldwide.\n"
        "Put a very short justification in basis."
    )


def _ask(generate_fn, name: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content":
            "You estimate worldwide statistics about personal names. You give "
            "ONE global aggregate number, never a per-country breakdown. You "
            "are precise about what you do and do not know, and you never "
            "invent a statistic to sound confident -- but not knowing which "
            "culture a name comes from is NOT a reason to answer 0.5."},
        {"role": "user", "content": _prompt(name)},
    ]
    # 256, not 160. The blind-control run proved 140 truncated the JSON mid
    # `reason` on exactly the rows whose text ran long, producing a table of
    # unparseable rows that looked like a model failure and was a budget bug.
    raw = generate_fn(messages, temperature=0.2, max_new_tokens=256)
    try:
        parsed = NameGenderProbability.model_validate_json(raw)
    except Exception:  # noqa: BLE001 -- a lab reports, it does not raise
        return {"share_male": None, "basis": "UNPARSEABLE", "raw": raw[-90:]}
    return {"share_male": float(parsed.share_male), "basis": parsed.basis}


def _pool_names() -> List[tuple]:
    """(name, current_tag) for every tagged first name."""
    out = []
    for tag in ("male", "female", "unisex"):
        for n in POOLS.FIRST_NAMES_BY_GENDER.get(tag, ()):
            out.append((n, tag))
    return sorted(out)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated local model ids")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N names (harness smoke)")
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" /
                                         "gender_probability_lab.json"))
    args = ap.parse_args(argv)

    names = _pool_names()
    if args.limit:
        names = names[:args.limit]
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    results: Dict[str, Dict[str, Any]] = {
        n: {"current_tag": t, "answers": {}} for n, t in names
    }

    # ONE MODEL RESIDENT AT A TIME. Loading a second before releasing the first
    # is how a 16 GB card OOMs; the panel is sequential on purpose.
    for model_id in model_ids:
        print(f"\n[lab] loading {model_id} ...", flush=True)
        started = time.time()
        try:
            entry = LOADER.load_llm(model_id, optimization_profile="Standard")
        except Exception as exc:  # noqa: BLE001
            print(f"[lab] SKIP {model_id}: {exc}", flush=True)
            continue
        gen = make_constrained_generate_fn(entry, NameGenderProbability)
        print(f"[lab] loaded in {time.time() - started:.1f}s; "
              f"{len(names)} names", flush=True)
        for i, (name, _tag) in enumerate(names, 1):
            results[name]["answers"][model_id] = _ask(gen, name)
            if i % 25 == 0:
                print(f"[lab]   {model_id}: {i}/{len(names)}", flush=True)
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:  # noqa: BLE001 -- standalone: no comfy runtime
            pass

    # ------------------------------------------------------------------ #
    # the finding
    # ------------------------------------------------------------------ #
    contested, agreed, split = [], [], []
    for name, row in results.items():
        shares = [a["share_male"] for a in row["answers"].values()
                  if a.get("share_male") is not None]
        if not shares:
            continue
        mean = sum(shares) / len(shares)
        row["mean_share_male"] = round(mean, 3)
        row["spread"] = round(max(shares) - min(shares), 3) if shares else None
        panel = "male" if mean >= 0.5 else "female"
        row["panel_says"] = panel
        # Split = the models do not agree with EACH OTHER on the side.
        if any(s >= 0.5 for s in shares) and any(s < 0.5 for s in shares):
            split.append(name)
            row["verdict"] = "models_split"
        elif row["current_tag"] != panel and (
                mean >= CONFIDENT or mean <= 1 - CONFIDENT):
            contested.append(name)
            row["verdict"] = "contests_the_tag"
        else:
            agreed.append(name)
            row["verdict"] = "agrees_with_tag"

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "models": model_ids,
        "confident_threshold": CONFIDENT,
        "names": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n[lab] {len(agreed)} agree / {len(contested)} contest the tag / "
          f"{len(split)} the models split among themselves")
    if contested:
        print("\nCONTESTED -- the panel is confident the tag is wrong:")
        for n in sorted(contested):
            r = results[n]
            print(f"   {n:14s} tagged {r['current_tag']:7s} panel says "
                  f"{r['panel_says']:7s} (mean share_male "
                  f"{r['mean_share_male']}, spread {r['spread']})")
    if split:
        print("\nMODELS SPLIT -- genuinely ambiguous names, treat with care:")
        for n in sorted(split):
            r = results[n]
            print(f"   {n:14s} tagged {r['current_tag']:7s} "
                  f"(mean {r['mean_share_male']}, spread {r['spread']})")
    print(f"\n[lab] full report -> {out}")
    print("[lab] DIAGNOSTIC ONLY -- nothing here was written to cast_pools.py "
          "or to any sidecar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
