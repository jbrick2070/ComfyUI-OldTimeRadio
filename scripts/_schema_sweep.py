"""One-off: dump canonical preserved widgets_values shape for each node in
workflows/otr_scifi_16gb_full.json, using live ComfyUI /object_info schema."""
import json
import urllib.request

WORKFLOW = "workflows/otr_scifi_16gb_full.json"
BASE = "http://127.0.0.1:8000"

_SOCKET_ONLY = {"PROJECT_STATE", "MODEL", "AUDIO", "IMAGE", "LATENT",
                "CONDITIONING", "CLIP", "VAE", "MASK", "NOISE"}


def _is_widget_backed(spec):
    td = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
    if isinstance(td, list):
        return True  # dropdown
    if isinstance(td, str):
        return td not in _SOCKET_ONLY
    return False


def _default(spec):
    td = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
    meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else {}
    if isinstance(td, list):
        return meta.get("default", td[0] if td else "")
    if td == "STRING":
        return meta.get("default", "")
    if td == "INT":
        return meta.get("default", 0)
    if td == "FLOAT":
        return meta.get("default", 0.0)
    if td in ("BOOLEAN", "BOOL"):
        return meta.get("default", False)
    return meta.get("default", None)


def fetch_schema(node_type):
    url = f"{BASE}/object_info/{node_type}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())[node_type]


def main():
    with open(WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)

    for n in wf["nodes"]:
        nt = n.get("type")
        if not nt or nt in ("MarkdownNote", "Note"):
            continue
        try:
            schema = fetch_schema(nt)
        except Exception as e:
            print(f"Node {n['id']:>3} {nt}: SCHEMA FETCH FAILED ({e})")
            continue
        raw = {**schema["input"].get("required", {}),
               **schema["input"].get("optional", {})}
        # Preserve declared order
        input_order = list(schema["input"].get("required", {}).keys()) + \
                      list(schema["input"].get("optional", {}).keys())
        wb_names = [nm for nm in input_order if _is_widget_backed(raw[nm])]
        canonical = [_default(raw[nm]) for nm in wb_names]
        current = n.get("widgets_values")
        matches = canonical == current
        print(f"Node {n['id']:>3} {nt}")
        print(f"   widget_backed_count={len(wb_names)} names={wb_names}")
        print(f"   canonical={canonical}")
        print(f"   current  ={current}")
        print(f"   match={matches}")
        print()


if __name__ == "__main__":
    main()
