"""_otr_style_authority_smoke.py -- the live A/B for ONE STYLE AUTHORITY.

THE ACCEPTANCE TEST (Fable's, adopted verbatim, GO_FORWARD_PLAN item B): mint the
same cartoon episode's character still on z_image BEFORE and AFTER the fix --
**the effective negative must contain no illustration-family terms, and the
positive must lead with the pack token.** Green unit tests are not a working fix;
PBUG-20260817-01's own lesson is that every prompt agreed while the pixels
diverged.

BOTH ARMS LIVE IN THIS ONE SCRIPT, BEHIND ``--pre-fix``, AND THE PRE-FIX ARM IS
PERMANENT. It is not deleted after use. This class of change is invisible in
isolation and obvious only in comparison -- BUG-411's "flattening the look" is
the standing reminder -- so the instrument that can show it has to outlive the
window that built it. Same pattern as ``_otr_lumina_image_smoke.py
--no-system-prompt``.

WHAT EACH ARM IS
  POST-FIX (default) -- the shipped path. The pack's own ``negative_tail``,
      self-veto-resolved by ``effective_negative``, composed with the object
      negative; the pack token FRONT-anchored by ``prefix_style_cue``.
  PRE-FIX (``--pre-fix``) -- the historical path. The engine-side STYLE-BLIND
      negative that contained "clean digital, cartoon, illustration", applied to
      every pack, with the request negative IGNORED (no image engine read
      ``request["negative_prompt"]``); the prompt carries the style at the TAIL
      only, never front-anchored.

THREE TRAPS, ALL THREE HANDLED HERE
  1. ANNOUNCER STILLS LEGITIMATELY CHANGE now that ``RADIO_CONSOLE_NEG`` reaches
     pixels for the first time. That is the fix, not a regression. ``--subject
     announcer`` renders exactly that case so it can be eyeballed on purpose
     rather than discovered and misread as a break.
  2. ``otr_style_traceroute.py`` is the cheap pre-check (EFFECTIVE FIGHTS must be
     0) but it reads TEXT. Necessary, NOT sufficient. This script mints pixels.
  3. STILLS SEEDS DERIVE FROM ``prompt_hash``, so changing prompt text changes
     the seed and an unpinned A/B compares two different rolls. Both arms here
     take the SAME EXPLICIT ``--seed`` straight into the KSampler, so the
     prompt-hash seed path is never consulted. This is the ``mode=fixed``
     requirement, enforced structurally rather than remembered.

The submitted graph is the ENGINE'S OWN: ``_zimage_params`` +
``_build_zimage_graph`` are imported and called, never re-typed, so the claim
"this is what a real mint does" cannot quietly stop being true. Only the
SaveImage terminal is added, so the still lands on disk for the file-check.

Usage:
  python scripts/_otr_style_authority_smoke.py                      # post-fix arm
  python scripts/_otr_style_authority_smoke.py --pre-fix            # pre-fix arm
  python scripts/_otr_style_authority_smoke.py --subject announcer  # trap 1
  python scripts/_otr_style_authority_smoke.py --text-only          # no GPU

Exit 0 = the post-fix arm passed both text gates AND a real PNG landed.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from otr_api import COMFYUI_URL, submit_prompt, poll_history  # noqa: E402
from nodes import _otr_paths as P  # noqa: E402  (the one OTR path authority)
# The composition is IMPORTED, never re-typed. A second copy of the style cue or
# the negative resolver is how an A/B quietly starts measuring itself instead of
# the shipped path.
from nodes._otr_visual_styles import (  # noqa: E402
    get_visual_style, prefix_style_cue, compact_style_cue, effective_negative)
from nodes._otr_story_brief_helpers import visual_safety_negative  # noqa: E402
from nodes._otr_image_engines.z_image_turbo import ZImageTurboEngine  # noqa: E402
from nodes import otr_meta_brief_image_prompt as mbp  # noqa: E402

#: The engine-side negative EXACTLY as it shipped before the fix. Typed here
#: because it no longer exists in the tree -- provenance is
#: `git show b2cc74c1^:nodes/_otr_image_engines/z_image_turbo.py` lines 218-219.
#: The two style opinions it carried ("clean digital", "cartoon, illustration")
#: are what made it a STYLE authority living engine-side.
_PREFIX_STYLE_BLIND_NEGATIVE = (
    "oversaturated, glossy, clean digital, plastic skin, waxy skin, "
    "sterile studio lighting, cartoon, illustration, text, watermark"
)

#: The illustration-family terms the acceptance test says must not survive into
#: an effective negative on an illustrated pack. These are precisely the phrases
#: the fix removed from the engine-side constant.
_ILLUSTRATION_FAMILY = ("cartoon", "illustration", "clean digital")

#: A dedicated smoke episode so stills land inside the output-tree contract
#: (otr/episodes/<ep>/stills/), routed through the single _otr_paths authority.
SMOKE_EPISODE = "style_authority_smoke"

#: A representative character-still subject. Deliberately a PERSON: the
#: acceptance test names a character still, and the illustration-family veto bit
#: hardest exactly where the pack asks for an illustrated human.
_CHARACTER_SUBJECT = (
    "a weary night-shift detective in a rumpled overcoat, mid-forties, "
    "standing at a rain-streaked office window, venetian blind shadows"
)

#: The announcer radio object -- trap 1's subject. Its object negative is the
#: real `radio_host_negative` value, which is what newly reaches pixels.
_ANNOUNCER_SUBJECT = (
    "a tabletop broadcast radio on a studio desk, woven speaker grille, "
    "glowing tuning dial, warm valve light"
)


def compose_arms(style_id, subject, pre_fix):
    """The positive + negative for ONE arm, built through the shipped helpers.

    Returns ``(positive, negative, style, notes)``. The ONLY typed string is the
    pre-fix negative literal above; everything else routes through the same
    functions the dispatcher calls.
    """
    style = get_visual_style({"visual_style": style_id})
    notes = {}

    # The pack's own look tail, appended the way finish_visual_prompt already
    # does. This is present in BOTH arms -- the tail was never the defect. What
    # the fix ADDED is the front anchor.
    if subject == "announcer":
        look = str(getattr(style, "announcer_subject_object", "") or "")
        base_subject = _ANNOUNCER_SUBJECT
        # Trap 1: this is the value that newly reaches pixels.
        obj_negative = mbp.radio_host_negative("radio_object")
    else:
        look = str(getattr(style, "portrait_look", "") or "")
        base_subject = _CHARACTER_SUBJECT
        obj_negative = ""
    tailed = ", ".join(t for t in (base_subject, look) if t)

    cue = compact_style_cue(style)
    notes["pack_token"] = cue
    notes["object_negative"] = obj_negative

    if pre_fix:
        # Historical behaviour: no front anchor, and the engine-side style-blind
        # negative with the request negative IGNORED entirely.
        positive = tailed
        negative = _PREFIX_STYLE_BLIND_NEGATIVE
        notes["pack_negative_authored"] = ""
        notes["pack_negative_effective"] = ""
    else:
        positive = prefix_style_cue(style, tailed)
        authored = str(getattr(style, "negative_tail", "") or "")
        resolved = effective_negative(style)
        notes["pack_negative_authored"] = authored
        notes["pack_negative_effective"] = resolved
        negative = visual_safety_negative(
            ", ".join(t for t in (resolved, obj_negative) if t))
    return positive, negative, style, notes


def check_gates(positive, negative, pack_token, style):
    """Fable's two acceptance gates, evaluated on TEXT.

    Returns ``(negative_clean, leads_with_token, offenders)`` where
    ``leads_with_token`` is ``None`` for N/A.

    GATE 1 IS "DOES THIS NEGATIVE FIGHT THIS PACK", NOT "DOES IT CONTAIN THE
    WORD CARTOON". Fable's wording -- no illustration-family terms -- is scoped
    to the ILLUSTRATED pack the acceptance test names. Generalised naively it is
    wrong: on the PHOTOREAL default pack "clean digital" is a legitimate,
    intended negative (it pushes toward filmic), and flagging it would fail the
    one control that proves the default lane is intact. The correct universal
    form is the repo's own authority, `otr_style_traceroute._fights_in` -- the
    same test the traceroute reports as EFFECTIVE FIGHTS -- which asks whether a
    negative phrase is something THIS pack's own positive asks for. It resolves
    every case correctly: it flags "cartoon" on `cartoon`, flags it on
    `sci_fi_radio` too (whose own announcer surface asks for a "living cartoon
    appliance face" -- the self-veto half of PBUG-20260817-01), and leaves
    "clean digital" alone on a pack that never asks for it.

    It is also applied here to the FULL COMPOSED negative (pack + object), which
    the traceroute never sees -- it audits the pack's `negative_tail` alone.

    GATE 2 IS N/A, NOT FAILED, WHEN THE PACK TOKEN IS EMPTY. `compact_style_cue`
    returns "" for the DEFAULT pack on purpose -- sci_fi_radio IS the house look
    and prefixing it "would churn every default-lane golden for no visual gain".
    """
    # The traceroute is imported for its fight test, but importing it
    # `os.environ.setdefault`s OTR_TEST_MODE=1 and CUDA_VISIBLE_DEVICES=""
    # (otr_style_traceroute.py, top) -- harmless for a read-only report, a trap
    # in a script that then submits a LIVE GPU render. Nothing on the z_image
    # path reads either today, so this restores them rather than fixing a live
    # bug: the point is that a future engine which DOES read OTR_TEST_MODE must
    # not silently turn this instrument's render into a test-mode render.
    _saved = {k: os.environ.get(k)
              for k in ("OTR_TEST_MODE", "CUDA_VISIBLE_DEVICES")}
    try:
        import otr_style_traceroute as tr  # scripts/ is on sys.path (set above)
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v

    fights = tr._fights_in(negative, style)
    # Reported alongside as diagnostic colour, never as the gate itself.
    phrases = [p.strip().lower() for p in str(negative or "").split(",")]
    illustration_terms = [p for p in phrases if p in _ILLUSTRATION_FAMILY]
    negative_clean = not fights
    low_pos = str(positive or "").strip().lower()
    low_cue = str(pack_token or "").strip().lower()
    leads = low_pos.startswith(low_cue) if low_cue else None
    return negative_clean, leads, {"fights": fights,
                                   "illustration_terms": illustration_terms}


def _object_info_classes():
    """The live node-class set from /object_info, or None when unreachable."""
    try:
        with urllib.request.urlopen(
                "%s/object_info" % COMFYUI_URL.rstrip("/"), timeout=60) as resp:
            return set(json.loads(resp.read().decode("utf-8")).keys())
    except Exception:  # noqa: BLE001 -- absent server is a caller decision
        return None


def _pick_class(candidates, available):
    """First candidate the live server actually has; first overall if unknown."""
    if available:
        for c in candidates:
            if c in available:
                return c
    return candidates[0]


def build_api_graph(engine, positive, negative, *, width, height, seed,
                    save_prefix, available):
    """The ENGINE'S graph, converted to /prompt API form + a SaveImage terminal.

    `_build_zimage_graph` is pure and emits `{"class": <logical>, "inputs": ...}`
    with a wire callable; `_node_candidates` maps logical -> real ComfyUI class.
    Both are the engine's own, so this submits the shipped recipe rather than a
    hand-mirrored copy of it.
    """
    request = {
        "prompt": positive,
        "negative_prompt": negative,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "reference_image": "",
    }
    params = engine._zimage_params(request)
    # The A/B pins the seed EXPLICITLY on both arms (trap 3): a prompt-hash seed
    # would differ between arms because the prompt text differs by design.
    params["seed"] = int(seed)
    # The pre-fix arm's negative must survive _resolve_negative untouched.
    params["negative"] = negative
    candidates = engine._node_candidates(params)
    raw = engine._build_zimage_graph(params, lambda name, slot: [name, slot])

    api = {}
    for key, node in raw.items():
        api[key] = {
            "class_type": _pick_class(candidates[node["class"]], available),
            "inputs": node["inputs"],
        }
    api["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": save_prefix,
                   "images": [engine._TERMINAL, 0]},
    }
    return api, params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon",
                    help="visual style pack (default: cartoon -- the measured "
                         "PBUG-20260817-01 case)")
    ap.add_argument("--subject", default="character",
                    choices=("character", "announcer"),
                    help="character = the acceptance test; announcer = trap 1, "
                         "the still that legitimately changes")
    ap.add_argument("--pre-fix", action="store_true",
                    help="submit the PRE-FIX form (engine-side style-blind "
                         "negative, no front anchor, request negative ignored) "
                         "-- the A arm, kept permanently")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--text-only", action="store_true",
                    help="compose and gate the prompts, render nothing")
    args = ap.parse_args(argv)

    arm = "prefix" if args.pre_fix else "postfix"
    positive, negative, style, notes = compose_arms(
        args.style, args.subject, args.pre_fix)
    token = notes["pack_token"]
    neg_clean, leads, offenders = check_gates(positive, negative, token, style)

    print("[style-ab] arm=%s style=%s subject=%s seed=%d (EXPLICIT -- trap 3)"
          % (arm, args.style, args.subject, args.seed), flush=True)
    print("[style-ab] pack token   : %r" % token, flush=True)
    print("[style-ab] positive     : %s" % positive, flush=True)
    print("[style-ab] negative     : %s" % negative, flush=True)
    if not args.pre_fix:
        print("[style-ab] pack neg authored : %s"
              % notes["pack_negative_authored"], flush=True)
        print("[style-ab] pack neg effective: %s"
              % notes["pack_negative_effective"], flush=True)
    if notes["object_negative"]:
        print("[style-ab] object negative   : %s   <- trap 1: reaches pixels "
              "for the first time" % notes["object_negative"], flush=True)

    print("[style-ab] GATE 1 negative does not fight this pack's positive: %s%s"
          % ("PASS" if neg_clean else "FAIL",
             "" if neg_clean else "  fights=%s" % offenders["fights"]),
          flush=True)
    if offenders["illustration_terms"]:
        print("[style-ab]   note: illustration-family terms present: %s%s"
              % (offenders["illustration_terms"],
                 "" if not neg_clean else
                 "  (legitimate -- this pack's positive never asks for them)"),
              flush=True)
    gate2 = ("N/A (default pack carries no token, by design)" if leads is None
             else "PASS" if leads else "FAIL")
    print("[style-ab] GATE 2 positive leads with the pack token          : %s"
          % gate2, flush=True)

    if args.pre_fix:
        # The pre-fix arm is EXPECTED to fail. If it ever passes everything, the
        # A/B has stopped measuring the defect and the receipt is worthless.
        if neg_clean and leads is not False:
            print("[style-ab] ERROR: the pre-fix arm PASSED its gates -- this "
                  "instrument is no longer reproducing the defect.", flush=True)
            return 2
        print("[style-ab] pre-fix arm fails the gates AS EXPECTED (that is the "
              "defect this A/B exists to show)", flush=True)
    elif not (neg_clean and leads is not False):
        print("[style-ab] FAIL: the post-fix arm must pass both gates.",
              flush=True)
        return 1

    if args.text_only:
        print("[style-ab] --text-only: gates evaluated, nothing rendered. The "
              "traceroute reads TEXT too (trap 2) -- the render is the proof.",
              flush=True)
        return 0

    stills_dir = P.otr_stills_dir(SMOKE_EPISODE)
    glob_dir = str(stills_dir)
    stem = "style_%s_%s_%s_seed%d" % (args.style, args.subject, arm, args.seed)
    prefix = (stills_dir / stem).relative_to(P.comfy_output_dir()).as_posix()

    available = _object_info_classes()
    if available is None:
        print("[style-ab] FAIL: /object_info unreachable at %s -- is the "
              "headless server up?" % COMFYUI_URL, flush=True)
        return 1

    engine = ZImageTurboEngine()
    graph, params = build_api_graph(
        engine, positive, negative, width=args.width, height=args.height,
        seed=args.seed, save_prefix=prefix, available=available)
    print("[style-ab] unet=%s clip=%s vae=%s steps=%s cfg=%s shift=%s"
          % (params["unet_name"], params["clip_name"], params["vae_name"],
             params["steps"], params["cfg"], params["shift"]), flush=True)

    started = time.time()
    prompt_id = submit_prompt(graph)
    print("[style-ab] QUEUED prompt_id=%s" % prompt_id, flush=True)
    status, err = poll_history(prompt_id, timeout_s=args.timeout)
    wall = time.time() - started
    print("[style-ab] history status=%s (%.1fs) %s" % (status, wall, err),
          flush=True)

    landed = []
    if os.path.isdir(glob_dir):
        landed = [f for f in os.listdir(glob_dir)
                  if f.startswith(stem) and f.endswith(".png")
                  and os.path.getmtime(os.path.join(glob_dir, f)) >= started - 2]
    if status == "SUCCESS" and landed:
        newest = max(landed,
                     key=lambda f: os.path.getmtime(os.path.join(glob_dir, f)))
        size = os.path.getsize(os.path.join(glob_dir, newest))
        print("[style-ab] PASS: minted %s (%d bytes) in %.1fs"
              % (newest, size, wall), flush=True)
        return 0
    print("[style-ab] FAIL: status=%s landed=%s" % (status, landed), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
