"""
build_humo_bakeoff_workflow.py
==============================

Emits the per-leg STANDALONE HuMo bakeoff graph as a ComfyUI ``/prompt`` API JSON
(roundtables/2026-06-27-humo-optim/final.md, item 2). DIAGNOSTIC ONLY -- it
reuses ``HuMoEngine._build_graph`` READ-ONLY for the node templates and edits
NOTHING in production (eng_humo.py / otr_scifi_16gb_full.json / the OTR pack
__init__.py).

How it works
------------
``HuMoEngine._build_graph`` returns the in-process ``run_graph`` spec --
``{alias: {"class": <alias in _node_candidates>, "inputs": {name: literal |
Wire(src, slot)}}}`` (eng_humo.py:215). This module TRANSLATES that spec into the
HTTP ``/prompt`` shape -- ``{node_id: {"class_type": <real class>, "inputs":
{name: literal | [src_id, slot]}}}`` (cf. render_humo_batch.py:560-689) -- assigns
integer node ids, resolves each alias to its real ComfyUI ``class_type`` via
``_node_candidates``, and appends a ``SaveImage`` terminal fed from VAEDecode.

For the TWO-STAGE leg it splices the sibling ``OTR_BakeoffReclaim`` node onto the
LATENT edge (WanHuMoImageToVideo slot 2 -> KSampler.latent_image) so the umt5 CLIP
+ whisper encoders are evicted AFTER conditioning + ref-image encode and BEFORE
the heavy 14B sampler, in ONE prompt.

``--dry-validate`` builds every leg, asserts structural validity (every node has a
real class_type + valid links + a single SaveImage terminal; the reclaim node is
spliced on the latent edge for the two-stage legs ONLY; ModelSamplingSD3 shift=8),
writes the JSONs under scripts/, and -- if a ComfyUI server is reachable -- also
asserts WanHuMoImageToVideo + OTR_BakeoffReclaim are REGISTERED. Renders nothing.

UTF-8, no BOM. ASCII-only. SFW.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The OTR pack's nodes/__init__.py is empty (cold-import clean), so this imports
# the engine + the Wire/quantize helpers with NO ComfyUI server and NO torch
# (the heavy wrapper imports are lazy, inside load/render_clip). Same pattern as
# scripts/otr_video_gpu_smoke.py and scripts/otr_video_soak.py.
from nodes._otr_video_engines import eng_humo  # noqa: E402
from nodes._otr_video_engines.wrapper_bridge import (  # noqa: E402
    Wire, quantize_frames_4n1)

# Fixed assets -- the SAME pair the LTX-AV bakeoff uses (visible mouth + plosives).
FIXED_ASSETS = ["c02_466a19906ccb.png", "c02_b002_line.wav"]
STILL_NAME, AUDIO_NAME = FIXED_ASSETS[0], FIXED_ASSETS[1]

# Fixed render knobs (held constant across legs so the only variable is the
# engine / two-stage eviction). Frame count is a short lip-sync clip, quantized
# to the Wan-2.1-VAE 4n+1 rule and clamped to the HuMo floor/ceiling.
SEED = 0
FRAMES = int(os.environ.get("OTR_BAKEOFF_FRAMES", "49"))   # ~2s @ 25fps, 4n+1
# (49 keeps the lip-sync clip short so the peak is UNET-weight-dominated, not
# activation-dominated -- a fairer "does the tier FIT" measurement + it leaves
# headroom to actually COMPLETE and emit a clip for the eyeball. Override via
# OTR_BAKEOFF_FRAMES.)
POSITIVE = "a person speaking to camera, subtle natural facial motion"

#: Map the bakeoff engine id to its registered HuMoEngine subclass.
ENGINE_CLASSES = {
    "humo_14B_169": eng_humo.HuMo14BLandscapeEngine,   # 14B fp8 + lightx2v distill
    "humo_1.7B_169": eng_humo.HuMo17BLandscapeEngine,  # current-ship control
}

#: The four legs (final.md). Sentinel reuses the two-stage graph but the RUNNER
#: fires a real ltx_audio_in render first in the same resident session.
LEGS = [
    {"label": "i_14B_single", "engine": "humo_14B_169",
     "two_stage": False, "sentinel": False},
    {"label": "ii_14B_twostage", "engine": "humo_14B_169",
     "two_stage": True, "sentinel": False},
    {"label": "iii_1p7B_control", "engine": "humo_1.7B_169",
     "two_stage": False, "sentinel": False},
    {"label": "iv_sentinel_14B_twostage", "engine": "humo_14B_169",
     "two_stage": True, "sentinel": True},
]

RECLAIM_CLASS = "OTR_BakeoffReclaim"
RESET_CLASS = "OTR_BakeoffVramReset"   # latent passthrough: reset CUDA peak pre-sampler
PROBE_CLASS = "OTR_BakeoffVramProbe"   # image passthrough: log true peak post-decode


def log(msg):
    print("[humo-build] %s" % msg, flush=True)


def engine_for(name):
    cls = ENGINE_CLASSES.get(name)
    if cls is None:
        raise KeyError("unknown bakeoff engine id %r (have %s)"
                       % (name, sorted(ENGINE_CLASSES)))
    return cls()


def _class_type_for(alias, candidates, schemas):
    """Resolve a graph alias to its real ComfyUI class_type: the first candidate
    registered in ``schemas`` (when a server is up), else the first candidate."""
    cands = candidates if isinstance(candidates, (list, tuple)) else (candidates,)
    cands = [c for c in cands if c]
    if schemas:
        for c in cands:
            if c in schemas:
                return c
    return cands[0]


def _to_link(val, alias_to_id):
    """Translate a run_graph input value to /prompt shape: a Wire(src, slot) ->
    ``[node_id, slot]``; a literal passes through."""
    if isinstance(val, Wire):
        return [alias_to_id[val.src], int(val.slot)]
    return val


def build_leg_prompt(leg, image_name=STILL_NAME, audio_name=AUDIO_NAME,
                     *, frames=FRAMES, seed=SEED, positive=POSITIVE, schemas=None):
    """Build ONE leg's /prompt JSON + an ``expected`` meta the runner asserts.

    Returns ``(prompt, meta)``. The prompt is the standalone HuMo graph with a
    SaveImage terminal; for a two-stage leg the OTR_BakeoffReclaim node is spliced
    on the WanHuMoImageToVideo(slot2) -> KSampler.latent_image edge."""
    eng = engine_for(leg["engine"])
    width, height = eng._native_dims()
    length = quantize_frames_4n1(frames, eng_humo._HUMO_MIN_FRAMES,
                                 eng_humo._HUMO_MAX_FRAMES)
    plan = {"text_prompt": positive, "seed": int(seed)}
    spec = eng._build_graph(image_name, audio_name, plan, length, width, height)
    candidates = eng._node_candidates()

    # Stable integer ids for every alias (insertion order is deterministic).
    alias_to_id = {alias: str(i + 1) for i, alias in enumerate(spec)}
    prompt = {}
    for alias, node in spec.items():
        ctype = _class_type_for(alias, candidates[alias], schemas)
        inputs = {k: _to_link(v, alias_to_id) for k, v in node["inputs"].items()}
        prompt[alias_to_id[alias]] = {"class_type": ctype, "inputs": inputs}

    next_id = len(spec) + 1
    humo_id = alias_to_id["humo"]
    latent_src = [humo_id, 2]
    # Two-stage: splice OTR_BakeoffReclaim between humo(slot2) and the sampler.
    if leg["two_stage"]:
        reclaim_id = str(next_id)
        next_id += 1
        prompt[reclaim_id] = {
            "class_type": RECLAIM_CLASS,
            "inputs": {"samples": latent_src,
                       "reason": "humo bakeoff pre-sampler (%s)" % leg["label"]}}
        latent_src = [reclaim_id, 0]
    # VRAM RESET on the latent edge, AFTER reclaim/humo and BEFORE KSampler, so the
    # probe reads the TRUE sampler+decode peak since this reset.
    reset_id = str(next_id)
    next_id += 1
    prompt[reset_id] = {"class_type": RESET_CLASS, "inputs": {"samples": latent_src}}
    prompt[alias_to_id["ksampler"]]["inputs"]["latent_image"] = [reset_id, 0]
    # VRAM PROBE on the image edge, AFTER VAEDecode and BEFORE SaveImage.
    probe_id = str(next_id)
    next_id += 1
    prompt[probe_id] = {"class_type": PROBE_CLASS,
                        "inputs": {"images": [alias_to_id["vaedecode"], 0]}}
    # SaveImage terminal (lossless PNG batch) fed from the probe passthrough.
    save_id = str(next_id)
    next_id += 1
    prefix = "otr/episodes/_bakeoff_humo/_frames/%s/frame" % leg["label"]
    prompt[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": prefix, "images": [probe_id, 0]}}

    # Optional per-leg cfg override (the 1.7B de-blue sweep): rewrite the KSampler
    # cfg literal in the built prompt so the manifest cross-check still matches.
    cfg_val = eng._cfg()
    if leg.get("cfg") is not None:
        cfg_val = float(leg["cfg"])
        for nd in prompt.values():
            if nd["class_type"] == "KSampler":
                nd["inputs"]["cfg"] = cfg_val

    names = eng._loader_names()
    lora = names.get("lora")
    skip_lora = (not lora) or str(lora).strip().lower() in ("none", "skip", "off")
    meta = {
        "label": leg["label"], "engine_id": eng.name,
        "two_stage": bool(leg["two_stage"]), "sentinel": bool(leg.get("sentinel")),
        "unet": names["unet"], "lora": (None if skip_lora else lora),
        "clip": names["clip"], "vae": names["vae"], "whisper": names["whisper"],
        "shift": 8.0, "steps": eng._steps(), "cfg": cfg_val, "seed": int(seed),
        "width": width, "height": height, "length": length,
        "terminal": "SaveImage", "frames_prefix": prefix,
        "save_node_id": save_id,
        "reclaim_present": bool(leg["two_stage"]),
        "reset_present": True, "probe_present": True,
    }
    return prompt, meta


def validate_prompt(prompt, meta, schemas=None):
    """Structural (+ optional registration) assertions. Returns a checks list of
    (name, ok, detail). The runner / dry-validate FAILS LOUD on any ok=False."""
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    ids = set(prompt)
    # every node well-formed; every link points at a real node + slot
    well_formed = True
    dangling = []
    for nid, nd in prompt.items():
        if not isinstance(nd.get("class_type"), str) or not isinstance(
                nd.get("inputs"), dict):
            well_formed = False
        for k, v in nd.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in ids:
                    dangling.append("%s.%s->%s" % (nid, k, v[0]))
    chk("nodes_well_formed", well_formed)
    chk("no_dangling_links", not dangling, ", ".join(dangling))

    saves = [n for n, d in prompt.items() if d["class_type"] == "SaveImage"]
    chk("single_SaveImage_terminal", len(saves) == 1, "found %d" % len(saves))

    ks = [n for n, d in prompt.items() if d["class_type"] == "KSampler"]
    chk("one_KSampler", len(ks) == 1, "found %d" % len(ks))
    humo = [n for n, d in prompt.items() if d["class_type"] == "WanHuMoImageToVideo"]
    reclaims = [n for n, d in prompt.items() if d["class_type"] == RECLAIM_CLASS]
    resets = [n for n, d in prompt.items() if d["class_type"] == RESET_CLASS]
    probes = [n for n, d in prompt.items() if d["class_type"] == PROBE_CLASS]
    vae = [n for n, d in prompt.items() if d["class_type"] == "VAEDecode"]

    # KSampler.latent_image must come from the RESET node (always spliced).
    chk("one_VramReset", len(resets) == 1, "found %d" % len(resets))
    if ks and resets:
        li = prompt[ks[0]]["inputs"].get("latent_image")
        chk("ksampler_latent_from_reset",
            isinstance(li, list) and li[0] == resets[0], "latent_image=%r" % (li,))
        # RESET.samples comes from reclaim (two-stage) or humo slot2 (single).
        rsrc = prompt[resets[0]]["inputs"].get("samples")
        if meta["two_stage"]:
            chk("reclaim_spliced", len(reclaims) == 1, "found %d" % len(reclaims))
            if reclaims:
                chk("reset_samples_from_reclaim",
                    isinstance(rsrc, list) and rsrc == [reclaims[0], 0],
                    "samples=%r" % (rsrc,))
                rs = prompt[reclaims[0]]["inputs"].get("samples")
                chk("reclaim_samples_from_humo_slot2",
                    isinstance(rs, list) and humo and rs == [humo[0], 2],
                    "samples=%r" % (rs,))
        else:
            chk("no_reclaim_in_single_leg", len(reclaims) == 0,
                "found %d" % len(reclaims))
            chk("reset_samples_from_humo_slot2",
                isinstance(rsrc, list) and humo and rsrc == [humo[0], 2],
                "samples=%r" % (rsrc,))
    # SaveImage.images must come from the PROBE node, fed by VAEDecode.
    chk("one_VramProbe", len(probes) == 1, "found %d" % len(probes))
    if saves and probes:
        si = prompt[saves[0]]["inputs"].get("images")
        chk("save_images_from_probe",
            isinstance(si, list) and si[0] == probes[0], "images=%r" % (si,))
        pi = prompt[probes[0]]["inputs"].get("images")
        chk("probe_images_from_vaedecode",
            isinstance(pi, list) and vae and pi == [vae[0], 0], "images=%r" % (pi,))

    msss = [n for n, d in prompt.items() if d["class_type"] == "ModelSamplingSD3"]
    chk("one_ModelSamplingSD3", len(msss) == 1, "found %d" % len(msss))
    if msss:
        shift = prompt[msss[0]]["inputs"].get("shift")
        chk("shift_is_8", float(shift) == 8.0, "shift=%r" % (shift,))

    # LoRA presence matches the engine tier (14B has it; 1.7B control does not).
    loras = [n for n, d in prompt.items() if d["class_type"] == "LoraLoaderModelOnly"]
    if meta["lora"]:
        chk("lora_present_14B", len(loras) == 1, "found %d" % len(loras))
    else:
        chk("lora_absent_control", len(loras) == 0, "found %d" % len(loras))

    if schemas:
        missing = sorted({d["class_type"] for d in prompt.values()
                          if d["class_type"] not in schemas})
        chk("all_classes_registered", not missing, ", ".join(missing))
        chk("WanHuMoImageToVideo_registered", "WanHuMoImageToVideo" in schemas)
        chk("OTR_BakeoffReclaim_registered", RECLAIM_CLASS in schemas)
        chk("OTR_BakeoffVramReset_registered", RESET_CLASS in schemas)
        chk("OTR_BakeoffVramProbe_registered", PROBE_CLASS in schemas)
    return checks


def _try_fetch_schemas(base_url):
    try:
        import requests
        r = requests.get(base_url + "/object_info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        return None
    return None


def dry_validate(out_dir=None, base_url=None):
    out_dir = out_dir or os.path.join(REPO, "scripts")
    os.makedirs(out_dir, exist_ok=True)
    schemas = _try_fetch_schemas(base_url) if base_url else None
    log("schemas: %s" % ("LIVE (%d classes)" % len(schemas) if schemas
                         else "none (structural-only; registration deferred to runner)"))
    all_ok = True
    for leg in LEGS:
        prompt, meta = build_leg_prompt(leg, schemas=schemas)
        checks = validate_prompt(prompt, meta, schemas=schemas)
        path = os.path.join(out_dir, "_humo_bakeoff_%s.json" % leg["label"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "meta": meta,
                       "checks": [{"name": n, "ok": ok, "detail": d}
                                  for (n, ok, d) in checks]}, f, indent=2)
        bad = [(n, d) for (n, ok, d) in checks if not ok]
        status = "PASS" if not bad else "FAIL"
        if bad:
            all_ok = False
        log("LEG %-26s %s | engine=%s two_stage=%s steps=%s cfg=%s %dx%d len=%s -> %s"
            % (leg["label"], status, meta["engine_id"], meta["two_stage"],
               meta["steps"], meta["cfg"], meta["width"], meta["height"],
               meta["length"], os.path.basename(path)))
        for (n, ok, d) in checks:
            if not ok:
                log("    [FAIL] %s %s" % (n, d))
    log("DRY-VALIDATE %s" % ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-validate", action="store_true",
                    help="build + assert every leg JSON; render nothing")
    ap.add_argument("--base-url", default=os.environ.get("COMFYUI_URL"),
                    help="if set, also assert classes are registered via /object_info")
    args = ap.parse_args()
    if args.dry_validate:
        return dry_validate(base_url=args.base_url)
    # default: just write the JSONs (structural)
    return dry_validate(base_url=args.base_url)


if __name__ == "__main__":
    sys.exit(main())
