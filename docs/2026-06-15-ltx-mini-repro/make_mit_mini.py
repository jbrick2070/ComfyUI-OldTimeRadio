"""Make the MIT-friendly mini: take the GGUF bookend graph and replace the GPL
VHS_VideoCombine output with core SaveWEBM. Result: deps = ComfyUI core (host) +
ComfyUI-GGUF (Apache) + ComfyUI-LTXVideo (LTX-2 Community, inherent to LTX). No
AGPL (RES4LYF), no added GPL (VHS). Output: workflows/ltx_bookend_mini_repro_gguf_mit.json
"""
import json, os
SRC = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
       r"\workflows\ltx_bookend_mini_repro_22b_gguf.json")
OUT = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
       r"\workflows\ltx_bookend_mini_repro_gguf_mit.json")
g = json.loads(open(SRC, "rb").read().decode("utf-8-sig"))
for n in g["nodes"]:
    if n["type"] == "VHS_VideoCombine":
        n["type"] = "SaveWEBM"
        n["properties"] = {"Node name for S&R": "SaveWEBM"}
        # SaveWEBM widgets: filename_prefix, codec, fps, crf
        n["widgets_values"] = ["otr_ltx_gguf_mit/repro_b001", "vp9", 25.0, 18.0]
        n["outputs"] = []   # SaveWEBM has no outputs (VHS had VHS_FILENAMES)
        n["size"] = [340, 150]
        print(f"swapped node {n['id']}: VHS_VideoCombine -> SaveWEBM")
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(g, f, indent=2, ensure_ascii=False)
    f.write("\n")
b = open(OUT, "rb").read()
print(f"WROTE {OUT} ({len(b)} bytes, BOM={b[:3]==b'\\xef\\xbb\\xbf'})")
