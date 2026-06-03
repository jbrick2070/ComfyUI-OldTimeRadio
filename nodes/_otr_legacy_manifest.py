"""Legacy invocation manifest (R0a step e).

Freezes the widget vectors of the four legacy audio nodes (Bark / Kokoro /
MusicGen / AudioGen) plus a per-node ``widgets_sha256``. Any drift in a legacy
node's widget surface (a changed default, a renamed/added/removed widget) shifts
the audio baseline -- the manifest + its drift test catch it at pytest time so a
silent baseline change can't slip through. ``legacy_manifest_sha`` is a
re-baseline trigger (see the plan's Re-baseline triggers list).

The widget vector is derived from each node's ``INPUT_TYPES`` via one shared
extractor, so the committed JSON and the drift test read from a single source.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib

MANIFEST_SCHEMA_VERSION = "1"

_REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = _REPO / "config" / "legacy_invocation_manifest.json"

# node_key -> (module suffix under nodes/, class name)
LEGACY_AUDIO_NODES = {
    "OTR_MusicGenTheme": ("musicgen_theme", "MusicGenTheme"),
    "OTR_BatchAudioGenGenerator": ("batch_audiogen_generator", "BatchAudioGenGenerator"),
}


def _is_widget(spec) -> bool:
    """A widget is any declared input that is NOT a forceInput wire socket."""
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        return False
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return not opts.get("forceInput", False)


def _default_of(spec):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return opts["default"]
    t = spec[0]
    if isinstance(t, (list, tuple)):  # combo list -> first entry is the default
        return t[0] if t else None
    return None


def extract_widget_vector(input_types: dict) -> list:
    """Ordered ``[{name, default}, ...]`` for every non-forceInput input,
    declaration order (required then optional). Deterministic."""
    vec = []
    for section in ("required", "optional"):
        for name, spec in (input_types.get(section) or {}).items():
            if _is_widget(spec):
                vec.append({"name": name, "default": _default_of(spec)})
    return vec


def widgets_sha256(vector: list) -> str:
    canon = json.dumps(vector, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    """Import each legacy node class, read INPUT_TYPES, freeze its widget vector."""
    nodes = {}
    for key, (mod_suffix, cls_name) in LEGACY_AUDIO_NODES.items():
        mod = importlib.import_module(f"nodes.{mod_suffix}")
        cls = getattr(mod, cls_name)
        vec = extract_widget_vector(cls.INPUT_TYPES())
        nodes[key] = {
            "class": cls_name,
            "widgets": vec,
            "widgets_sha256": widgets_sha256(vec),
        }
    return {"manifest_schema_version": MANIFEST_SCHEMA_VERSION, "nodes": nodes}


def load_committed_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest() -> dict:
    """Generate + write the manifest JSON (run once via the Windows venv)."""
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    m = write_manifest()
    print("WROTE", MANIFEST_PATH)
    for k, v in m["nodes"].items():
        print(f"  {k}: {len(v['widgets'])} widgets sha={v['widgets_sha256'][:12]}")
