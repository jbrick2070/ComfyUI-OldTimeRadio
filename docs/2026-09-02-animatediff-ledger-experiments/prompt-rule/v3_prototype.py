"""Throwaway PROTOTYPE of Prompt v3 Half A. Touches no production file.

Composes what v3 would send for every beat of two real episodes, beside what v2
actually sent, and measures both with the installed SD1 tokenizer. This is a
design check before code: if the strings do not read like the operator's own
rewrites, the design is wrong and no amount of clean implementation fixes it.
"""
import glob
import hashlib
import json
import os
import sys

ROOT = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
COMFY = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI"
for p in (COMFY, ROOT, os.path.join(ROOT, "nodes")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from nodes._otr_video_engines import ghost_signal_author as gsa       # noqa: E402
from nodes._otr_video_engines import ghost_signal_prompt as gsp       # noqa: E402
from nodes import _otr_visual_styles as vs                            # noqa: E402

measure = gsa.resolve_token_measure(None)
tok = lambda t: measure(t)[0] if t else 0                             # noqa: E731

# --- the prototype's own constants (not yet in any production file) --------- #
VANTAGE_V3 = {
    "figure": "wide, the people small in the space",
    "object": "the object large in the frame",
    "signal": "lit against the dark, the light moving",
}

WORLD_MOTION_V3 = (
    "drifting slowly",
    "settling in the still air",
    "shifting as the light crosses it",
    "stirring once and going still",
    "moving with the draught",
    "trembling faintly",
    "sliding out of the dark",
    "turning slowly in place",
)


def hash_int(*parts):
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def resolve_kernel(meta, ordinal):
    """Bounded subject + place. Total: never raises, may return ''.

    ODOMETER CYCLING, not modulo-on-both. Cycling the subject and the place on
    the same index makes the pair repeat every len(objects) beats -- four
    objects and four settings gave SEVEN byte-identical kernels in a 29-beat
    episode. Rolling the place only when the subject wraps gives a period of
    len(objects) * len(settings), which no episode reaches.
    """
    objs = [str(o).strip() for o in (meta.get("key_objects") or []) if str(o).strip()]
    terms = meta.get("story_brief_terms") or {}
    where = [str(s).strip() for s in (terms.get("setting") or []) if str(s).strip()]
    if objs:
        subject = objs[ordinal % len(objs)]
        if where:
            place = where[(ordinal // len(objs)) % len(where)]
            return "%s in the %s" % (subject, place), "key_object"
        return subject, "key_object"
    if where:
        return where[ordinal % len(where)], "setting"
    brief = " ".join(str(meta.get("story_brief") or "").split())
    if brief:
        return " ".join(brief.split()[:8]).rstrip(",."), "brief"
    return "", "omitted"


def resolve_light(meta, ordinal, mode):
    """The pack's light term, on the slowest wheel of the odometer.

    DROPPED IN `signal` MODE: that vantage already says "lit against the dark,
    the light moving", and a second lighting clause contradicts it -- the
    prototype produced "harsh fluorescent overheads, ... lit against the dark"
    on the same beat. One light statement per prompt.
    """
    if mode == "signal":
        return ""
    terms = meta.get("story_brief_terms") or {}
    lights = [str(s).strip() for s in (terms.get("lighting") or []) if str(s).strip()]
    if not lights:
        return ""
    objs = [o for o in (meta.get("key_objects") or []) if str(o).strip()]
    where = [w for w in ((terms.get("setting") or [])) if str(w).strip()]
    wheel = max(len(objs), 1) * max(len(where), 1)
    return lights[(ordinal // wheel) % len(lights)]


def resolve_motion(episode_seed, beat_id, used):
    start = hash_int(episode_seed, beat_id) % len(WORLD_MOTION_V3)
    for step in range(len(WORLD_MOTION_V3)):
        cand = WORLD_MOTION_V3[(start + step) % len(WORLD_MOTION_V3)]
        if cand not in used:
            return cand
    return WORLD_MOTION_V3[start]


def compose_v3(style, mode, role, kernel, light, motion, bookend):
    """Role-aware. The operator's rule 6: the radio objects STAY on the
    announcer and music beds, placed in the setting rather than on a table in a
    dark room -- his own rewrite reads "a bakelite radio set, with a background
    of ... Williston Reservoir with floating driftwood". So a bookend keeps its
    radio subject and the crux becomes the BACKGROUND behind it.
    """
    if bookend and kernel:
        head = "%s, with the %s behind it" % (bookend, kernel)
    elif bookend:
        head = bookend
    else:
        head = kernel
    units = [u for u in (head, light, motion, VANTAGE_V3.get(mode, "")) if u]
    return vs.prefix_style_cue(style, ", ".join(units))


def run(ep_dir):
    led = json.load(open(glob.glob(os.path.join(ep_dir, "audio", "*_ledger.json"))[0],
                         encoding="utf-8"))
    meta = led.get("meta") or {}
    style = vs.resolve_visual_style(meta.get("visual_style") or "")
    seed = meta.get("episode_seed")
    shots = [s for s in ((led.get("video") or {}).get("shots") or []) if s.get("ghost_prompt")]

    print("=" * 100)
    print("EPISODE : %s" % os.path.basename(ep_dir))
    print("STYLE   : %s   BANK: %s   SHOTS: %d"
          % (meta.get("visual_style"), meta.get("source_bank"), len(shots)))
    print("BRIEF   : %s" % str(meta.get("story_brief") or "")[:150])
    print("=" * 100)

    used, v2_tot, v3_tot = set(), [], []
    for i, s in enumerate(shots):
        g = s["ghost_prompt"]
        role, mode = s.get("role"), g["mode"]
        v2 = gsp.compose_ghost_prompt_v2(role=role, style=style, mode=mode,
                                         motif_cue=g["motif_cue"],
                                         drawable_beat=g["drawable_beat"])["positive"]
        kernel, source = resolve_kernel(meta, i)
        light = resolve_light(meta, i, mode)
        motion = resolve_motion(seed, s.get("shot_id"), used)
        used.add(motion)
        bookend = gsa.GHOST_BOOKEND_MOTIFS.get(("open", mode)) if role != "character_video" else ""
        if role != "character_video" and not bookend:
            bookend = sorted(v for k, v in gsa.GHOST_BOOKEND_MOTIFS.items() if k[1] == mode)[0] if any(k[1] == mode for k in gsa.GHOST_BOOKEND_MOTIFS) else ""
        v3 = compose_v3(style, mode, role, kernel, light, motion, bookend)
        v2_tot.append(tok(v2))
        v3_tot.append(tok(v3))
        print("\n[%d] %-22s mode=%-7s kernel_source=%s" % (i, s.get("shot_id"), mode, source))
        print("  v2 (%2d tok): %s" % (tok(v2), v2))
        print("  v3 (%2d tok): %s" % (tok(v3), v3))

    n = max(len(v2_tot), 1)
    print("\nMEAN TOKENS  v2=%.1f  v3=%.1f  (window 77, author target 69)"
          % (sum(v2_tot) / n, sum(v3_tot) / n))
    print("MAX  TOKENS  v2=%d    v3=%d" % (max(v2_tot or [0]), max(v3_tot or [0])))
    over = [t for t in v3_tot if t > 69]
    print("v3 beats over the 69 author target: %d ; over the 77 window: %d"
          % (len(over), len([t for t in v3_tot if t > 77])))


BASE = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
for name in ("signal_lost_the_faded_ledger_20260902_210812",
             "signal_lost_the_last_reading_20260902_190630"):
    run(os.path.join(BASE, name))
