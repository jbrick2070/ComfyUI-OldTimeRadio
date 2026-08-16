"""Ask a local model to second-guess every gender the pipeline has ASSIGNED.

WHAT THIS IS FOR. The deterministic scan pinned 132 characters across the 65
public_domain units, and the shakespeare lane pins more from its cast blocks.
Two of those pins were WRONG and were found by a human reading the data:
`buck_rogers` had "a Han patrol" (a squad of soldiers) as female, and `miss_mix`
had "the housekeeper" as male. Both were caught by eye. Nobody has checked the
other 130.

So this asks a model the same question independently and reports every case
where it disagrees with what we assigned. It is a SECOND OPINION on shipped
data, not a source of truth -- where the two disagree, a human decides.

THE SECOND ASK IS THE POINT. A single disagreement can be sampling noise: the
decode is stochastic (`do_sample=True`), so one answer proves little. Every
disagreement is therefore re-asked, and only a disagreement that REPEATS is
reported as stable. A flip-flop is reported separately and means the model has
no real opinion, which is itself worth knowing -- it says the character is
ambiguous rather than that we are wrong.

WHY THIS QUESTION IS DIFFERENT FROM THE NAME LAB. `otr_gender_probability_lab`
asks "what share of people named X are male" -- a fact about NAMES, right for
the invented-character pool. This asks "in this WORK, is this CHARACTER male or
female", which is a fact about the story. Ahab is male because Melville says so,
not because the name skews male. Only the second question can judge a pin.

DIAGNOSTIC ONLY. Never writes a sidecar, never writes cast_pools.py, nothing in
the render path reads its output. Local weights only -- no web, no API key.
Run with no ComfyUI server holding VRAM (CLAUDE.md section 4).
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

BANK = REPO_ROOT / "config" / "source_banks" / "public_domain_story"
MANIFEST = BANK / "manifest.sample.json"

PROMPT_VERSION = "character_gender_second_opinion_v1"
DEFAULT_MODEL = "google/gemma-4-E4B-it"


class CharacterGenderOpinion(BaseModel):
    """One model's read on one character in one work."""

    gender: str = Field(..., pattern="^(male|female|unsure)$")
    reason: str = Field(..., max_length=200)


def _ask(generate_fn, name: str, title: str, author: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content":
            "You answer questions about characters in published literature. "
            "If you do not know the work or the character, answer 'unsure'. "
            "Never guess to sound confident."},
        {"role": "user", "content": (
            f'In "{title}"{f" by {author}" if author else ""}, is the '
            f'character "{name}" male or female?\n'
            "Answer 'male', 'female', or 'unsure' if you do not know this "
            "character. Give a one-phrase reason."
        )},
    ]
    raw = generate_fn(messages, temperature=0.2, max_new_tokens=120)
    try:
        parsed = CharacterGenderOpinion.model_validate_json(raw)
        return {"gender": parsed.gender, "reason": parsed.reason}
    except Exception:  # noqa: BLE001 -- a lab reports, it does not raise
        return {"gender": "unparseable", "reason": raw[:120]}


def _assigned_rows() -> List[Dict[str, Any]]:
    """Every character the pipeline has actually PINNED, with its work."""
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for src in man.get("sources") or []:
        title = str(src.get("title") or "")
        author = str(src.get("author") or "")
        for unit in src.get("units") or []:
            text_path = BANK / str(unit.get("text_path") or "")
            sidecar = text_path.parent / (text_path.stem + ".provenance.json")
            if not sidecar.is_file():
                continue
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            for row in data.get("characters") or []:
                out.append({
                    "source_id": src.get("source_id", ""),
                    "title": title, "author": author,
                    "name": row.get("name", ""),
                    "assigned": row.get("gender", ""),
                    "via": row.get("gender_source", ""),
                })
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" /
                                         "gender_second_opinion.json"))
    args = ap.parse_args(argv)

    rows = _assigned_rows()
    if args.limit:
        rows = rows[:args.limit]
    print(f"[lab] {len(rows)} assigned character(s) to second-guess")

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    print(f"[lab] loading {args.model} ...", flush=True)
    started = time.time()
    entry = LOADER.load_llm(args.model, optimization_profile="Standard")
    gen = make_constrained_generate_fn(entry, CharacterGenderOpinion)
    print(f"[lab] loaded in {time.time() - started:.1f}s", flush=True)

    agree, unsure, stable, flipped = [], [], [], []
    for i, r in enumerate(rows, 1):
        first = _ask(gen, r["name"], r["title"], r["author"])
        r["first"] = first
        if first["gender"] == r["assigned"]:
            r["verdict"] = "agrees"
            agree.append(r)
        elif first["gender"] in ("unsure", "unparseable"):
            r["verdict"] = "model_unsure"
            unsure.append(r)
        else:
            # THE SECOND ASK. One disagreement can be sampling noise; only a
            # disagreement that repeats is worth a human's attention.
            second = _ask(gen, r["name"], r["title"], r["author"])
            r["second"] = second
            if second["gender"] == first["gender"]:
                r["verdict"] = "disagrees_twice"
                stable.append(r)
            else:
                r["verdict"] = "model_flip_flopped"
                flipped.append(r)
        if i % 20 == 0:
            print(f"[lab]   {i}/{len(rows)}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION, "model": args.model,
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n[lab] {len(agree)} agree / {len(stable)} DISAGREE TWICE / "
          f"{len(flipped)} flip-flopped / {len(unsure)} model unsure")
    if stable:
        print("\nDISAGREES TWICE -- a human should look at these:")
        for r in stable:
            print(f"   {r['source_id']:24s} {r['name']:22s} "
                  f"we say {r['assigned']:7s} model says "
                  f"{r['first']['gender']:7s} (via {r['via']})")
            print(f"      reason: {r['first']['reason'][:90]}")
    if flipped:
        print("\nFLIP-FLOPPED -- the model has no stable opinion, "
              "read as ambiguity rather than as a defect:")
        for r in flipped:
            print(f"   {r['source_id']:24s} {r['name']:22s} "
                  f"we say {r['assigned']}")
    print(f"\n[lab] full report -> {out}")
    print("[lab] DIAGNOSTIC ONLY -- no sidecar was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
