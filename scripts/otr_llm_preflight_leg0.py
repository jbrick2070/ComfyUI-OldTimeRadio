"""LOCAL-LLM SWEEP LEG 0 -- the in-process preflight (GO_FORWARD 5.3).

One command, no ComfyUI, idle GPU. Per local row:

    request_slot(slot, model_id) -> make_generate_fn -> ~40-token generate
    -> unload_llm(), with reset_peak_memory_stats() around each

and it FAILS LOUDLY on a dead row rather than skipping it. The four canonical
legs in 5.6 are the real proof; this exists so a dead row is found in fifteen
minutes at the console instead of forty minutes into an episode.

WHY THIS IS NOT THE ACCEPTANCE SIGNAL, and the row says so explicitly:
`meta.slot_calls_by_slot` is incremented ONLY inside
`_SlotScheduler._account_and_get_entry`, and six `request_slot` sites live
outside it -- so that counter proves in-writer generation only. This script
does not read it and neither should anything else.

WHAT IT MEASURES PER ROW: does it admit under the policy, load, generate real
tokens, and give the VRAM back. Both slots, because the charter is "every
surviving row does creative AND technical, or it is ripped on a MEASURED
failure, never on assumption" (docs/OTR_STANDING_RULINGS.md).

    python scripts/otr_llm_preflight_leg0.py                # every local row
    python scripts/otr_llm_preflight_leg0.py --rows a,b     # a subset
    python scripts/otr_llm_preflight_leg0.py --slots creative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "nodes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Rows whose lane is a REMOTE provider: they hold zero local VRAM and there is
#: nothing for a VRAM preflight to measure. Named by prefix, not by count, so a
#: new provider slot does not silently become a "local row that failed".
_REMOTE_PREFIXES = ("openrouter:", "comfy:", "google_api:")

#: The prose ask. Deliberately trivial and deliberately NOT about an episode:
#: this measures the transport, not the writing, and story quality is closed
#: work (operator 2026-08-04).
_MESSAGES = [
    {"role": "system", "content": "You answer in one short sentence."},
    {"role": "user", "content": "Name one thing a radio needs to work."},
]


def _vram():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"free_gb": round(free / 2**30, 2),
                "used_gb": round((total - free) / 2**30, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except Exception:                                       # noqa: BLE001
        return None


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                       # noqa: BLE001
        pass


def _local_rows():
    from nodes import _otr_model_catalog as catalog
    rows = []
    for choice in catalog.dropdown_choices():
        label = str(choice)
        if label.startswith(_REMOTE_PREFIXES):
            continue
        # The dropdown label carries a human size suffix; the id is the head.
        rows.append(label.split(" (")[0].strip())
    # Preserve order, drop duplicates: the dropdown can list the same head id
    # under more than one label.
    seen, ordered = set(), []
    for r in rows:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _probe(model_id: str, slot: str, max_new_tokens: int) -> dict:
    from nodes import _otr_model_loader as loader
    out = {"row": model_id, "slot": slot, "ok": False}
    _reset_peak()
    before = _vram()
    t0 = time.time()
    try:
        entry = loader.request_slot(slot, model_id)
        out["load_s"] = round(time.time() - t0, 1)
        out["loaded"] = _vram()
        gen = loader.make_generate_fn(entry)
        t1 = time.time()
        text = gen(_MESSAGES, temperature=0.7, max_new_tokens=max_new_tokens)
        out["gen_s"] = round(time.time() - t1, 1)
        text = (text or "").strip()
        out["chars"] = len(text)
        out["sample"] = text[:110].replace("\n", " ")
        # A row that loads and returns an EMPTY string is a dead row, not a
        # pass. That is the failure this whole script exists to surface.
        out["ok"] = bool(text)
        if not text:
            out["error"] = "generated no tokens"
    except Exception as exc:                                # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        out["traceback"] = traceback.format_exc()[-700:]
    finally:
        try:
            loader.unload_llm()
        except Exception as exc:                            # noqa: BLE001
            out["unload_error"] = "%s: %s" % (type(exc).__name__, exc)
        # EMPTY THE CACHING ALLOCATOR BEFORE MEASURING, or the number lies.
        # `unload_llm` drops the references; torch keeps the freed blocks in
        # its own pool, so `mem_get_info` still counts them as used and a
        # perfectly clean row looks like it leaked ~2 GB.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        out["after"] = _vram()
        out["before"] = before
        if before and out["after"]:
            out["returned_gb"] = round(
                out["after"]["free_gb"] - before["free_gb"], 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="", help="comma-separated subset of row ids")
    ap.add_argument("--slots", default="creative,technical")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    rows = [r.strip() for r in args.rows.split(",") if r.strip()] or _local_rows()
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]

    print("[leg0] %d local rows x %d slots, %d tokens each"
          % (len(rows), len(slots), args.max_new_tokens))
    base = _vram()
    print("[leg0] idle VRAM: %s" % (base,))
    for r in rows:
        print("   %s" % r)
    print()

    results = []
    for row in rows:
        for slot in slots:
            print("[leg0] %-42s %-9s ..." % (row, slot), end="", flush=True)
            res = _probe(row, slot, args.max_new_tokens)
            results.append(res)
            if res["ok"]:
                print(" OK  load=%ss gen=%ss %dch  peak=%s free_after=%s"
                      % (res.get("load_s"), res.get("gen_s"), res.get("chars"),
                         (res.get("loaded") or {}).get("peak_gb"),
                         (res.get("after") or {}).get("free_gb")))
                print("        %s" % res.get("sample"))
            else:
                print(" FAIL %s" % res.get("error"))

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print("\n[leg0] %d/%d probes passed" % (len(ok), len(results)))
    if bad:
        print("[leg0] MEASURED FAILURES -- each is a rip candidate under the charter:")
        for r in bad:
            print("   %-42s %-9s %s" % (r["row"], r["slot"], r.get("error")))
    both = sorted({r["row"] for r in ok if
                   all(any(x["row"] == r["row"] and x["slot"] == s and x["ok"]
                           for x in results) for s in slots)})
    print("[leg0] rows that did EVERY slot: %d" % len(both))
    for r in both:
        print("   %s" % r)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "slots": slots, "idle_vram": base,
                       "results": results, "rows_all_slots": both}, fh, indent=1)
        print("[leg0] wrote %s" % args.json_out)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
