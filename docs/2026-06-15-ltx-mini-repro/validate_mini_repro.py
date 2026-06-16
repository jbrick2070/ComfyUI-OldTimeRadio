"""Static validation of workflows/ltx_bookend_mini_repro.json against live
:8000 /object_info -- without rendering. Reconstructs the API prompt the
ComfyUI frontend would submit and checks completeness, types, link integrity,
encoding (no BOM), and that the LoadImage still is staged in input/.
"""
import json, os, sys, urllib.request

WF = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
      r"\workflows\ltx_bookend_mini_repro.json")
INPUT_DIR = r"C:\Users\jeffr\Documents\ComfyUI\input"
PRIM = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
errs, warns = [], []

raw = open(WF, "rb").read()
if raw[:3] == b"\xef\xbb\xbf":
    errs.append("FILE HAS UTF-8 BOM")
try:
    raw.decode("utf-8")
except Exception as e:
    errs.append(f"not valid utf-8: {e}")
g = json.loads(raw.decode("utf-8-sig"))

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8000/object_info", timeout=60))
nodes = {n["id"]: n for n in g["nodes"]}
links = {l[0]: l for l in g["links"]}

def is_widget(meta):
    t = meta[0]
    return isinstance(t, list) or (isinstance(t, str) and t in PRIM)

def has_control(meta):
    return len(meta) > 1 and isinstance(meta[1], dict) and meta[1].get("control_after_generate")

def widget_input_order(spec):
    inp = spec.get("input", {})
    req, opt = inp.get("required", {}), inp.get("optional", {})
    order = spec.get("input_order", {}) or {}
    ro = order.get("required", list(req.keys()))
    oo = order.get("optional", list(opt.keys()))
    out = []
    for nm in ro:
        if nm in req:
            out.append((nm, req[nm]))
    for nm in oo:
        if nm in opt:
            out.append((nm, opt[nm]))
    return out

# link integrity
for lid, l in links.items():
    _, sn, ss, dn, ds, lt = l
    if sn not in nodes or dn not in nodes:
        errs.append(f"link {lid} references missing node")
        continue
    so = nodes[sn].get("outputs") or []
    di = nodes[dn].get("inputs") or []
    if ss >= len(so):
        errs.append(f"link {lid}: src slot {ss} OOB on node {sn}")
    if ds >= len(di):
        errs.append(f"link {lid}: dst slot {ds} OOB on node {dn}")

# build API prompt + validate per node
prompt = {}
for nid, n in nodes.items():
    t = n["type"]
    if t not in oi:
        errs.append(f"node {nid}: class_type {t} NOT in object_info")
        continue
    spec = oi[t]
    ins = {}
    # link inputs
    slot_names = []
    for s, ip in enumerate(n.get("inputs") or []):
        slot_names.append(ip["name"])
        if ip.get("link") is not None:
            l = links.get(ip["link"])
            if l:
                ins[ip["name"]] = [str(l[1]), l[2]]
    # widget inputs (in order), consuming widgets_values + control extras
    wv = list(n.get("widgets_values") or [])
    wi = 0
    for nm, meta in widget_input_order(spec):
        if nm in slot_names:
            continue  # provided by a link, not a widget here
        if not is_widget(meta):
            continue
        if wi >= len(wv):
            errs.append(f"node {nid} {t}: missing widget value for '{nm}'")
            continue
        ins[nm] = wv[wi]; wi += 1
        if has_control(meta):
            wi += 1  # consume control_after_generate extra value
    if wi != len(wv):
        warns.append(f"node {nid} {t}: consumed {wi}/{len(wv)} widget values "
                     f"(leftover={wv[wi:]})")
    # required completeness
    for nm, meta in widget_input_order(spec):
        req = spec.get("input", {}).get("required", {})
        if nm in req and nm not in ins:
            errs.append(f"node {nid} {t}: required input '{nm}' UNSATISFIED")
    prompt[str(nid)] = {"class_type": t, "inputs": ins}

# spot-check the recipe values survived into the API prompt
def find(nid):
    return prompt.get(str(nid), {}).get("inputs", {})
checks = {
    "ckpt(2)": (find(2).get("ckpt_name"), "ltx-video-2b-v0.9.safetensors"),
    "clip(3)": (find(3).get("clip_name"), "t5xxl_fp16.safetensors"),
    "clip_type(3)": (find(3).get("type"), "ltxv"),
    "latent_w(6)": (find(6).get("width"), 832),
    "latent_h(6)": (find(6).get("height"), 480),
    "latent_len(6)": (find(6).get("length"), 209),
    "strength(7)": (find(7).get("strength"), 0.75),
    "frame_rate(8)": (find(8).get("frame_rate"), 25.0),
    "sampler(9)": (find(9).get("sampler_name"), "euler_cfg_pp"),
    "cfg(12)": (find(12).get("cfg"), 1.0),
    "noise_seed(11)": (find(11).get("noise_seed"), 1),
    "tile(14)": (find(14).get("tile_size"), 512),
    "temporal(14)": (find(14).get("temporal_size"), 4096),
    "tile_overlap(14)": (find(14).get("overlap"), 64),
    "temporal_overlap(14)": (find(14).get("temporal_overlap"), 8),
    "codec(15)": (find(15).get("codec"), "vp9"),
    "fps(15)": (find(15).get("fps"), 25.0),
}
for k, (got, want) in checks.items():
    if got != want:
        errs.append(f"recipe check {k}: got {got!r} want {want!r}")

# sigmas exact
sg = find(10).get("sigmas")
if sg != "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0":
    errs.append(f"sigmas mismatch: {sg!r}")

# still staged?
still = find(1).get("image")
if not (still and os.path.exists(os.path.join(INPUT_DIR, still))):
    warns.append(f"LoadImage still '{still}' not found in input/ (stage before run)")

print("API prompt nodes:", len(prompt))
print("RECIPE CHECKS: all", "PASS" if not any('recipe check' in e or 'sigmas' in e for e in errs) else "FAIL")
print("\nWARNINGS:")
for w in warns: print("  -", w)
print("\nERRORS:")
for e in errs: print("  -", e)
print("\nRESULT:", "PASS - valid & complete" if not errs else f"FAIL ({len(errs)} errors)")
sys.exit(1 if errs else 0)
