"""otr_visual_smoke.py -- S-F VISUAL SMOKE FIXTURE (accelerator).

Test visual engines WITHOUT re-running the writer + audio per leg.

Today every coverage leg re-runs the WRITER (gemma, minutes) + the full AUDIO
path (TTS + music) just to exercise a VISUAL engine -- most of the ~28 min/leg,
and a DIFFERENT story each run (not apples-to-apples). This harness bakes ONE
clean reference episode's node-92 INPUT (the OTR_ShotLock-planned ledger +
master audio + every referenced portrait / scene still / mesh-fodder / per-beat
wav) into a self-contained BUNDLE, then submits a PRUNED ComfyUI API prompt of
ONLY node-92 (OTR_VideoRenderBatch) with the three forceInput sockets baked to
literals + mode=episode. ComfyUI can only execute nodes present in the submitted
prompt, so the writer + audio nodes never run -> a leg costs the render TAIL
(minutes) and every leg shares an identical story + byte-identical master audio.

GROUNDING (2026-06-29): the node-92 input ledger (``video.shots`` +
``images.images``) is NOT persisted by any prior run -- the saved/``latest_ledger``
is the FLAT post-render canonical (top-level ``shots``/``clips``, no
``video.shots``/``images.images``). So the reference is captured live: node-92
now writes its received input as ``state/node_episode_input.json`` (a forensic
twin of ``node_episode_report.json``); this harness bakes from that.

TEST HARNESS ONLY -- no production-path change; the real workflow still writes a
fresh story per episode. UTF-8, no BOM, ASCII-only source.

Usage (ComfyUI venv python; live server on :8000):
  # 1. after a CLEAN reference episode has rendered, bake the bundle:
  python scripts\\otr_visual_smoke.py bake --out <bundle_dir>
  #    (reads state/node_episode_input.json; --input <path> to override)
  # 2. replay a visual engine against the baked bundle (render tail only):
  python scripts\\otr_visual_smoke.py replay --bundle <bundle_dir> [--engine ID]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_RENDER_NODE = 92          # OTR_VideoRenderBatch
DEFAULT_VALIDATOR_NODE = 63       # OTR_WorkflowValidator (OUTPUT_NODE; optional)
SMOKE_IMAGE_DONE = "OTR_SMOKE_IMAGE_DONE"   # opaque gate token; node never parses it
WORKFLOW_PATH = os.path.join(_REPO, "workflows", "otr_canonical.json")
STATE_INPUT_NAME = "node_episode_input.json"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Asset surface -- every disk path the node-92 render resolves out of the ledger
# ---------------------------------------------------------------------------
# render_driver resolves assets from exactly these ledger slots:
#   * ledger['images']['images'][i]['path']  -- portraits / scene_* stills /
#     mesh_fodder (_portrait_index / _still_index / _mesh_fodder_index)
#   * ledger['lines'][i][<*wav_path|clip_path|video_clip_path>] -- per-beat voice
#     / music audio (_voice_audio_for_line)
#   * ledger['video']['shots'][i]['init_image'] -- a pre-resolved init (defensive)
_LINE_ASSET_EXACT = ("clip_path", "video_clip_path")


def _line_asset_keys(line):
    keys = []
    for k, v in line.items():
        if not isinstance(v, str) or not v:
            continue
        if str(k).endswith("wav_path") or k in _LINE_ASSET_EXACT:
            keys.append(k)
    return keys


def iter_asset_slots(ledger):
    """Yield ``(setter, current_path)`` for every disk-asset path slot in the
    ledger; ``setter(new_path)`` rewrites it in place. Pure structural walk --
    no disk access."""
    imgs = ((ledger.get("images") or {}).get("images")) or []
    for im in imgs:
        if isinstance(im, dict) and isinstance(im.get("path"), str) and im["path"]:
            yield (lambda v, _im=im: _im.__setitem__("path", v)), im["path"]
    for ln in (ledger.get("lines") or []):
        if not isinstance(ln, dict):
            continue
        for k in _line_asset_keys(ln):
            yield (lambda v, _ln=ln, _k=k: _ln.__setitem__(_k, v)), ln[k]
    for sh in ((ledger.get("video") or {}).get("shots") or []):
        if isinstance(sh, dict) and isinstance(sh.get("init_image"), str) and sh["init_image"]:
            yield (lambda v, _sh=sh: _sh.__setitem__("init_image", v)), sh["init_image"]


# ---------------------------------------------------------------------------
# Bundle baking
# ---------------------------------------------------------------------------
def bake_bundle(ledger, master_audio_path, bundle_dir, *, base_dirs=()):
    """Copy every referenced asset + the master mix into ``bundle_dir``, rewrite
    a DEEP COPY of the ledger's paths to the bundle, preflight-exist + sha256
    each. Returns ``(baked_ledger, baked_master_path, manifest)``. Raises
    ``FileNotFoundError`` LOUD listing any missing asset (the CLEAN-reference
    precondition: a bundle with a dangling asset is not apples-to-apples)."""
    assets_dir = os.path.join(bundle_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    baked = copy.deepcopy(ledger)
    manifest = {"assets": {}, "missing": []}
    seen = {}

    def _resolve(p):
        if os.path.isabs(p) and os.path.exists(p):
            return p
        for b in base_dirs:
            cand = os.path.join(b, p)
            if os.path.exists(cand):
                return cand
        return p if os.path.exists(p) else ""

    def _copy_in(orig):
        real = _resolve(orig)
        if not real:
            manifest["missing"].append(orig)
            return None
        if real in seen:
            return seen[real]
        base = os.path.basename(real)
        dst = os.path.join(assets_dir, base)
        n = 1
        while os.path.exists(dst) and os.path.abspath(dst) != os.path.abspath(real):
            stem, ext = os.path.splitext(base)
            dst = os.path.join(assets_dir, "%s_%d%s" % (stem, n, ext))
            n += 1
        shutil.copy2(real, dst)
        seen[real] = dst
        manifest["assets"][orig] = {"bundle": dst, "sha256": sha256_file(dst),
                                    "src": real}
        return dst

    for setter, cur in iter_asset_slots(baked):
        dst = _copy_in(cur)
        if dst:
            setter(dst)

    baked_master = ""
    if master_audio_path:
        real = _resolve(master_audio_path)
        if not real:
            manifest["missing"].append(master_audio_path)
        else:
            baked_master = os.path.join(
                bundle_dir, "master" + os.path.splitext(real)[1])
            shutil.copy2(real, baked_master)
            manifest["master"] = {"bundle": baked_master,
                                  "sha256": sha256_file(baked_master), "src": real}

    if manifest["missing"]:
        raise FileNotFoundError(
            "S-F bundle: %d referenced asset(s) missing from disk (not a CLEAN "
            "reference episode -- re-bake from a complete render): %s"
            % (len(manifest["missing"]), manifest["missing"][:8]))

    baked_ledger_path = os.path.join(bundle_dir, "baked_ledger.json")
    with open(baked_ledger_path, "w", encoding="utf-8") as f:
        json.dump(baked, f, ensure_ascii=True)
    manifest["baked_ledger"] = baked_ledger_path
    manifest["baked_master"] = baked_master
    with open(os.path.join(bundle_dir, "bundle_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return baked, baked_master, manifest


def preflight_bundle(bundle_dir):
    """Re-verify EVERY baked asset still exists + hash-matches before a replay
    submit. Returns ``(ok, problems)``."""
    mpath = os.path.join(bundle_dir, "bundle_manifest.json")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    problems = []
    rows = dict(manifest.get("assets") or {})
    if manifest.get("master"):
        rows["__master__"] = manifest["master"]
    for orig, row in rows.items():
        b = row.get("bundle")
        if not (b and os.path.exists(b)):
            problems.append("MISSING %s -> %s" % (orig, b))
            continue
        if sha256_file(b) != row.get("sha256"):
            problems.append("HASH-DRIFT %s" % b)
    return (not problems), problems


# ---------------------------------------------------------------------------
# Pruned API prompt (node-92 only; the 3 forceInput sockets baked to literals)
# ---------------------------------------------------------------------------
def build_pruned_prompt(workflow, schemas, *, ledger_json, master_audio_path,
                        image_done=SMOKE_IMAGE_DONE, mode="episode",
                        engine=None, render_node_id=DEFAULT_RENDER_NODE,
                        validator_node_id=DEFAULT_VALIDATOR_NODE,
                        keep_validator=False):
    """Build the pruned /prompt dict: ONLY the render node (and optionally the
    validator), with ``patched_ledger_json`` / ``master_audio_path`` /
    ``image_done`` -- the node's three forceInput sockets -- baked to constant
    literals. By construction ComfyUI can execute no other node."""
    from otr_api import _serialized_slot_names

    def _widget_inputs(node):
        ntype = node["type"]
        slots = _serialized_slot_names(ntype, schemas)
        wv = node.get("widgets_values") or []
        if len(wv) != len(slots):
            raise ValueError(
                "node %s (%s) widget drift: wv=%d vs slots=%d (%s)"
                % (node.get("id"), ntype, len(wv), len(slots), slots))
        return ntype, {name: wv[i] for i, name in enumerate(slots)}

    rnode = next((n for n in workflow["nodes"]
                  if n.get("id") == render_node_id), None)
    if rnode is None:
        raise KeyError("render node id %s not in workflow" % render_node_id)
    rtype, inputs = _widget_inputs(rnode)
    inputs["mode"] = mode
    if engine is not None:
        inputs["engine"] = engine
    inputs["patched_ledger_json"] = ledger_json
    inputs["master_audio_path"] = master_audio_path or ""
    inputs["image_done"] = image_done
    prompt = {str(render_node_id): {"class_type": rtype, "inputs": inputs}}

    if keep_validator:
        vnode = next((n for n in workflow["nodes"]
                      if n.get("id") == validator_node_id), None)
        if vnode is not None:
            vtype, vinputs = _widget_inputs(vnode)
            prompt[str(validator_node_id)] = {"class_type": vtype,
                                              "inputs": vinputs}
    return prompt


# ---------------------------------------------------------------------------
# Executed-set verification
# ---------------------------------------------------------------------------
def executed_node_set(history_entry):
    """The node ids ComfyUI executed for a prompt -- the OUTPUT_NODE ids that
    produced outputs (node 92 + 63 are both OUTPUT_NODEs)."""
    return {str(k) for k in (history_entry.get("outputs") or {}).keys()}


def verify_executed(executed, *, render_node_id=DEFAULT_RENDER_NODE,
                    validator_node_id=DEFAULT_VALIDATOR_NODE,
                    keep_validator=False):
    """ACCEPTANCE: executed set is EXACTLY {render} (or {validator, render});
    the writer node 1 + every audio node are ABSENT."""
    allowed = {str(render_node_id)}
    if keep_validator:
        allowed.add(str(validator_node_id))
    extra = executed - allowed
    render_ran = str(render_node_id) in executed
    ok = render_ran and not extra
    return ok, {"executed": sorted(executed), "allowed": sorted(allowed),
                "unexpected": sorted(extra), "render_ran": render_ran}


def assert_prompt_pruned(prompt, *, render_node_id=DEFAULT_RENDER_NODE,
                         validator_node_id=DEFAULT_VALIDATOR_NODE,
                         keep_validator=False):
    """Pre-submit invariant: the prompt carries ONLY the allowed node ids (so
    no upstream writer/audio node can be pulled in as a dependency)."""
    allowed = {str(render_node_id)}
    if keep_validator:
        allowed.add(str(validator_node_id))
    present = set(prompt.keys())
    extra = present - allowed
    if extra:
        raise ValueError("pruned prompt carries unexpected nodes: %s "
                         "(allowed %s)" % (sorted(extra), sorted(allowed)))
    # no input may reference another node (a [src_id, slot] link) -- all baked
    for nid, spec in prompt.items():
        for k, v in (spec.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) not in allowed:
                raise ValueError("node %s input %s links to non-pruned node %s"
                                 % (nid, k, v[0]))
    return True


# ---------------------------------------------------------------------------
# Capture (read node-92's forensic input dump)
# ---------------------------------------------------------------------------
def load_capture(input_path=None):
    """Load the captured node-92 input (ledger + master_audio_path). Defaults to
    the state-dir ``node_episode_input.json`` the node writes each episode."""
    if input_path is None:
        from nodes._otr_paths import otr_state_dir
        input_path = os.path.join(str(otr_state_dir()), STATE_INPUT_NAME)
    with open(input_path, encoding="utf-8") as f:
        cap = json.load(f)
    ledger = cap.get("ledger")
    if isinstance(ledger, str):
        ledger = json.loads(ledger)
    if not isinstance(ledger, dict) or not (ledger.get("video") or {}).get("shots"):
        raise ValueError(
            "capture %s has no video.shots -- it is not a node-92 input ledger "
            "(run a CLEAN reference episode first so the node writes its input)"
            % input_path)
    return ledger, str(cap.get("master_audio_path") or ""), input_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_bake(args):
    ledger, master, src = load_capture(args.input)
    bake_dirs = tuple(d for d in (args.base or "").split(os.pathsep) if d)
    baked, baked_master, manifest = bake_bundle(
        ledger, master, args.out, base_dirs=bake_dirs)
    print("[smoke] baked bundle -> %s" % args.out, flush=True)
    print("[smoke]   from capture: %s" % src, flush=True)
    print("[smoke]   assets: %d | master: %s | shots: %d"
          % (len(manifest["assets"]), bool(baked_master),
             len((baked.get("video") or {}).get("shots") or [])), flush=True)
    return 0


def _cmd_replay(args):
    from otr_api import (fetch_schemas, load_workflow, submit_prompt,
                         poll_history, COMFYUI_URL)
    import requests

    ok, problems = preflight_bundle(args.bundle)
    if not ok:
        print("[smoke] PREFLIGHT FAIL:\n  " + "\n  ".join(problems), flush=True)
        return 2
    baked_ledger_path = os.path.join(args.bundle, "baked_ledger.json")
    with open(baked_ledger_path, encoding="utf-8") as f:
        ledger_json = f.read()
    with open(os.path.join(args.bundle, "bundle_manifest.json"), encoding="utf-8") as f:
        baked_master = (json.load(f).get("baked_master") or "")

    schemas = fetch_schemas()
    workflow = load_workflow(args.workflow)
    prompt = build_pruned_prompt(
        workflow, schemas, ledger_json=ledger_json,
        master_audio_path=baked_master, engine=args.engine,
        keep_validator=args.keep_validator)
    assert_prompt_pruned(prompt, keep_validator=args.keep_validator)
    master_sha_before = ""
    if baked_master and os.path.exists(baked_master):
        master_sha_before = sha256_file(baked_master)

    print("[smoke] submitting pruned prompt (nodes=%s, engine=%s) -> %s"
          % (sorted(prompt.keys()), args.engine or "(ledger default)",
             COMFYUI_URL), flush=True)
    pid = submit_prompt(prompt)
    status, err = poll_history(pid, timeout_s=args.timeout)
    hist = {}
    try:
        hist = requests.get("%s/history/%s" % (COMFYUI_URL, pid), timeout=15).json()
    except Exception as exc:  # noqa: BLE001
        print("[smoke] history fetch failed: %s" % exc, flush=True)
    entry = hist.get(pid, {}) if isinstance(hist, dict) else {}
    executed = executed_node_set(entry)
    vok, detail = verify_executed(executed, keep_validator=args.keep_validator)
    master_sha_after = (sha256_file(baked_master)
                        if baked_master and os.path.exists(baked_master) else "")
    audio_ok = (master_sha_before == master_sha_after)

    print("[smoke] status=%s executed=%s pruned_ok=%s audio_untouched=%s"
          % (status, detail["executed"], vok, audio_ok), flush=True)
    if detail["unexpected"]:
        print("[smoke] UNEXPECTED executed nodes: %s" % detail["unexpected"],
              flush=True)
    if err:
        print("[smoke] error: %s" % err[:400], flush=True)
    ok_all = (status == "SUCCESS") and vok and audio_ok
    print("[smoke] %s" % ("PASS" if ok_all else "FAIL"), flush=True)
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="S-F visual smoke fixture")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bake", help="bake a bundle from a captured node-92 input")
    b.add_argument("--out", required=True, help="bundle output dir")
    b.add_argument("--input", default=None,
                   help="captured node_episode_input.json (default: state dir)")
    b.add_argument("--base", default="",
                   help="os.pathsep-joined base dirs for relative asset paths")
    b.set_defaults(func=_cmd_bake)

    r = sub.add_parser("replay", help="replay a visual engine against a bundle")
    r.add_argument("--bundle", required=True)
    r.add_argument("--engine", default=None,
                   help="override the video engine (else the ledger's own)")
    r.add_argument("--workflow", default=WORKFLOW_PATH)
    r.add_argument("--keep-validator", action="store_true",
                   help="also run node 63 (executed set becomes {63,92})")
    r.add_argument("--timeout", type=int, default=1800)
    r.set_defaults(func=_cmd_replay)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
