"""CONTAINED repro for the ideogram4_local music-card refusal.

WHAT THIS SETTLES. The 2026-08-21 verdict recorded that
`still_music_closing_001` refused while the music OPENING card rendered, and
called the refusal SEED-dependent because both came from "the same composer, the
same episode title and the same prompt shape". Shape is not text, and nobody
diffed the two prompts -- the hard-fail erased the refused one before it was
persisted. So the claim is an INFERENCE, and this is the measurement.

HOW IT ANSWERS THE QUESTION. It composes both music cards through the REAL
composer from a real frozen ledger, then renders each across N seeds:

  * every seed refuses on closing, none on opening -> CONTENT-driven.
  * closing renders on some seeds -> SEED-driven, as the verdict assumed.
  * both refuse -> something broader (prompt shape / recipe), not this card.

NO EPISODE, no video, no obs publish -- one engine, two prompts, N seeds. It
never writes into an episode tree.

Usage:
    python scripts/otr_ideogram4_refusal_repro.py [--seeds 4] [--episode <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEFAULT_EP = (r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
              r"\signal_lost_the_weight_of_the_grain_20260822_025251")


def load_ledger(ep_dir):
    hits = glob.glob(os.path.join(ep_dir, "audio", "*_ledger.json"))
    if not hits:
        raise SystemExit("no ledger under %s" % ep_dir)
    with open(hits[0], encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4,
                    help="how many seeds per card (default 4)")
    ap.add_argument("--episode", default=DEFAULT_EP)
    args = ap.parse_args()

    from nodes.otr_meta_brief_image_prompt import compose_still_word_prompt
    from nodes._otr_visual_styles import get_visual_style
    from nodes._otr_image_engines.ideogram4_local import (
        Ideogram4LocalEngine, Ideogram4RefusalError, classify_refusal)

    led = load_ledger(args.episode)
    meta = led.get("meta") or {}
    style = get_visual_style(meta)
    print("episode : %s" % os.path.basename(args.episode))
    print("style   : %s" % style.style_id)
    print("title   : %s" % (meta.get("episode_title") or "<none>"))
    print()

    # Both MUSIC cards, composed through the real composer. Music mode is the
    # wordless abstract-title still, so the beat line is not script text.
    cards = []
    for oid, role in (("music_opening_001", "music_visual"),
                      ("music_closing_001", "music_visual")):
        line = {}
        for ln in (led.get("lines") or []):
            if str(ln.get("line_id") or "") == oid:
                line = ln
                break
        prompt = compose_still_word_prompt(meta, role, line or "", style=style)
        cards.append((oid, prompt))
        print("=" * 74)
        print(oid)
        print("=" * 74)
        print(prompt)
        print()

    # THE DIFF THE VERDICT NEVER DID.
    a, b = cards[0][1], cards[1][1]
    print("=" * 74)
    print("PROMPT DIFF")
    print("=" * 74)
    print("identical: %s" % (a == b))
    if a != b:
        import difflib
        for ln in difflib.unified_diff(a.split(", "), b.split(", "),
                                       "opening", "closing", lineterm="", n=1):
            print("  " + ln)
    print()

    eng = Ideogram4LocalEngine()
    try:
        eng.assert_usable({}, {})
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("ideogram4_local not usable here: %s" % exc)

    print("=" * 74)
    print("RENDERS -- %d seed(s) per card" % args.seeds)
    print("=" * 74)
    results = {}
    for oid, prompt in cards:
        results[oid] = []
        for i in range(args.seeds):
            seed = 42 + i * 1009
            req = {"object_id": oid, "prompt": prompt, "seed": seed,
                   "width": 1472, "height": 832}
            try:
                frame = eng.render_image(req)
                refused, mn, sd = classify_refusal(frame)
            except Ideogram4RefusalError as exc:
                refused, mn, sd = True, float("nan"), float("nan")
                print("  %-20s seed=%-6d REFUSED  (%s)" % (oid, seed, exc))
            else:
                print("  %-20s seed=%-6d %-8s min=%.1f std=%.1f"
                      % (oid, seed, "REFUSED" if refused else "rendered", mn, sd))
            results[oid].append(bool(refused))

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    for oid, flags in results.items():
        print("  %-20s refused %d of %d" % (oid, sum(flags), len(flags)))
    closing = results.get("music_closing_001") or []
    opening = results.get("music_opening_001") or []
    if closing and all(closing) and opening and not any(opening):
        print("\n  -> CONTENT-DRIVEN. The closing card refuses on every seed "
              "while the opening renders on all of them. The verdict doc's "
              "'seed-dependent' reading does not survive; diff the prompts "
              "above for the trigger.")
    elif closing and any(closing) and not all(closing):
        print("\n  -> SEED-DRIVEN, as the verdict assumed: the same prompt "
              "both refuses and renders depending only on the seed.")
    elif closing and not any(closing):
        print("\n  -> DID NOT REPRODUCE at these seeds. The refusal is rarer "
              "than this sample; widen --seeds before concluding anything.")
    else:
        print("\n  -> BOTH cards refuse. That points at the prompt SHAPE or "
              "the recipe, not at this one card.")


if __name__ == "__main__":
    main()
