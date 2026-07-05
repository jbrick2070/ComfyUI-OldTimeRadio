"""Skeleton generator for a new story pack (roundtable pass01, Grok:
protects the 1am copy-paste workflow). Forces explicit bank + pipeline,
emits a seam-complete skeleton with the bank's required seams, and refuses
existing keys. Run with any Python 3.10+; stdlib only.

Usage:
    python tools/new_pack.py --bank media_archive --model my_new_model \
        [--pipeline legacy_many_pass] [--clone-from path/to/existing.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pipeline", default="legacy_many_pass")
    ap.add_argument("--clone-from", default="")
    args = ap.parse_args()

    banks = json.loads((FIXTURES / "banks.json").read_text(encoding="utf-8"))
    bank = next(
        (b for b in banks["banks"] if b["source_bank_id"] == args.bank), None,
    )
    if bank is None:
        print(f"ERROR: unknown bank {args.bank!r}; known: "
              f"{[b['source_bank_id'] for b in banks['banks']]}")
        return 1
    pipelines = json.loads((FIXTURES / "pipelines.json").read_text(encoding="utf-8"))
    if args.pipeline not in {p["story_pipeline_id"] for p in pipelines["pipelines"]}:
        print(f"ERROR: unknown pipeline {args.pipeline!r}")
        return 1

    for existing in (FIXTURES / "story_packs").rglob("*.json"):
        pack = json.loads(existing.read_text(encoding="utf-8"))
        if (pack["source_bank_id"], pack["story_model_id"],
                pack["story_pipeline_id"]) == (args.bank, args.model, args.pipeline):
            print(f"ERROR: pack already exists: {existing}")
            return 1

    if args.clone_from:
        base = json.loads(Path(args.clone_from).read_text(encoding="utf-8"))
        base["story_model_id"] = args.model
        base["story_pipeline_id"] = args.pipeline
        base["source_bank_id"] = args.bank
        base["label"] = args.model.replace("_", " ").title()
        base["status"] = "not_implemented"
        skeleton = base
        note = ("CLONED: rewrite EVERY prompt_stages value and re-derive "
                "forbidden_leakage_terms for THIS lane before setting status "
                "to ready_fixture - a stale term list scans for the wrong "
                "lane's leakage.")
    else:
        skeleton = {
            "schema_version": "v2.0",
            "source_bank_id": args.bank,
            "story_model_id": args.model,
            "story_pipeline_id": args.pipeline,
            "label": args.model.replace("_", " ").title(),
            "status": "not_implemented",
            "prompt_stages": {seam: "" for seam in bank.get("required_seams", [])},
            "labels": {},
            "coda_examples": [],
            "examples": [],
            "tone_guardrails": [],
            "forbidden_plot_patterns": [],
            "forbidden_leakage_terms": [],
            "source_requirements": [],
            "ledger_validation_notes": [],
        }
        note = ("Fill every empty prompt_stages seam (the registry refuses "
                "empty required seams) and see docs/PACK_AUTHOR_CHECKLIST.md.")

    folder = {"science_news": "science_news", "media_archive": "media_archive",
              "public_domain_story": "public_domain"}.get(args.bank, "experimental")
    out = FIXTURES / "story_packs" / folder / f"{args.model}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n")
    print(f"WROTE {out}")
    print(note)
    print("Validate with: .venv python -m pytest tests -q  (or scripts/validate_lab.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
