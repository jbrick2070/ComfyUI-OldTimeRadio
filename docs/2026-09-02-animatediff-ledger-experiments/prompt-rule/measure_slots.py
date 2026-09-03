"""Measure the real Ghost v2 slot budget with the INSTALLED SD1 tokenizer.

V5 of the r2 contract: every slot is measured, per pack, including the cue-less
default pack, before a v3 composition constant is chosen. No GPU work -- the
tokenizer is CPU.
"""
import json
import os
import sys

ROOT = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
COMFY = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI"
for p in (COMFY, ROOT, os.path.join(ROOT, "nodes")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.pop("OTR_TEST_MODE", None)
# The SD1 tokenizer is CPU-only, but importing `comfy` runs a device probe that
# raises on a blanked CUDA_VISIBLE_DEVICES. Leave the device visible: nothing
# here allocates VRAM, so a render on the other window is undisturbed.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from nodes._otr_video_engines import ghost_signal_author as gsa       # noqa: E402
from nodes._otr_video_engines import ghost_signal_prompt as gsp       # noqa: E402
from nodes import _otr_visual_styles as vs                            # noqa: E402

measure = gsa.resolve_token_measure(None)
print("tokenizer resolved:", measure is not None)
if measure is None:
    print("NO INSTALLED TOKENIZER -- cannot measure; aborting")
    raise SystemExit(1)


def tok(text):
    n, w = measure(text)
    return n


# ---- the live episode's own material -------------------------------------- #
LEDGER = (r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
          r"\signal_lost_the_faded_ledger_20260902_210812\audio"
          r"\signal_lost_the_faded_ledger_20260902_210812_ledger.json")
led = json.load(open(LEDGER, encoding="utf-8"))
meta = led.get("meta") or {}
key_objects = list(meta.get("key_objects") or [])
terms = meta.get("story_brief_terms") or {}
setting = list(terms.get("setting") or [])
lighting = list(terms.get("lighting") or [])

print("\n=== the episode ===")
print("key_objects:", key_objects)
print("setting[0] :", setting[:1])
print("lighting[0]:", lighting[:1])

shots = (led.get("video") or {}).get("shots") or []
objs = [(s.get("shot_id"), s.get("role"), s.get("ghost_prompt"))
        for s in shots if s.get("ghost_prompt")]

print("\n=== v2 AS RENDERED: per-slot installed SD1 tokens ===")
print("%-26s %-18s %5s %5s %5s %5s %6s" %
      ("shot", "mode", "cue", "motif", "leaf", "law", "TOTAL"))
rows = []
for shot_id, role, g in objs:
    style = vs.resolve_visual_style(meta.get("visual_style") or "")
    composed = gsp.compose_ghost_prompt_v2(
        role=role, style=style, mode=g["mode"],
        motif_cue=g["motif_cue"], drawable_beat=g["drawable_beat"])
    comp = composed["components"]
    total = tok(composed["positive"])
    rows.append((shot_id, g["mode"], tok(comp["pack_cue"]) if comp["pack_cue"] else 0,
                 tok(comp["motif"]), tok(comp["leaf"]), tok(comp["law"]), total))
    print("%-26s %-18s %5d %5d %5d %5d %6d" % rows[-1])

if rows:
    n = len(rows)
    print("%-26s %-18s %5.1f %5.1f %5.1f %5.1f %6.1f" % (
        "MEAN", "", sum(r[2] for r in rows) / n, sum(r[3] for r in rows) / n,
        sum(r[4] for r in rows) / n, sum(r[5] for r in rows) / n,
        sum(r[6] for r in rows) / n))
    print("headroom to the 77-token window: %.1f tokens mean, %d tokens worst"
          % (77 - sum(r[6] for r in rows) / n, 77 - max(r[6] for r in rows)))

# ---- what a v3 kernel would cost ------------------------------------------ #
print("\n=== candidate v3 crux kernels (this episode) ===")
where = setting[0] if setting else ""
for obj in key_objects:
    kernel = "%s in %s" % (obj, where) if where else obj
    print("  %-58s %2d tokens" % (kernel, tok(kernel)))
print("  %-58s %2d tokens" % ("[setting alone] " + where, tok(where)))
if lighting:
    print("  %-58s %2d tokens" % ("[light] " + lighting[0], tok(lighting[0])))

# ---- the pack cue across every registered style --------------------------- #
print("\n=== compact_style_cue per pack (r2 must-fix 11) ===")
for style_id in sorted(vs.list_style_ids()):
    st = vs.resolve_visual_style(style_id)
    cue = vs.compact_style_cue(st)
    print("  %-24s cue=%-28r %2d tokens" % (style_id, cue, tok(cue) if cue else 0))

# ---- the mode laws -------------------------------------------------------- #
print("\n=== GHOST_MODE_LAWS_V2 ===")
for mode, law in sorted(gsp.GHOST_MODE_LAWS_V2.items()):
    print("  %-8s %2d tokens  %s" % (mode, tok(law), law))
