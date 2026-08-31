#!/usr/bin/env python
"""What this pack imports, versus what it declares, versus what is installed.

    python scripts/otr_venv_audit.py                 # audit THIS interpreter
    python scripts/otr_venv_audit.py --declared-only  # no import checks

WHY. Two production failures in one day had the same shape and neither named
its cause. `accelerate` missing killed the writer 18 seconds in with a message
about device_map. `feedparser` missing killed the science bank with "RSS fetch
failed", and a comment in story_orchestrator had already predicted that exact
confusion for `bs4`: an error that "blames the FEEDS for a package that was
never installed". Both were DECLARED in requirements.txt the whole time; the
pod's provisioner had installed them into the system python while ComfyUI ran
its own venv.

So the useful question is not "what do we declare" -- we declare plenty -- but
**does the interpreter that runs ComfyUI actually have it**. This answers that
for whichever interpreter runs it.

It reports; it does not gate. Nothing here blocks a render, and a missing
package still fails at the point of use, loudly, the way the project prefers.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Import name -> distribution name, where they differ. Only the ones this
#: pack actually uses; a generic mapping table would be another thing to
#: maintain for no gain.
_DIST = {
    "bs4": "beautifulsoup4", "yaml": "PyYAML", "PIL": "Pillow",
    "cv2": "opencv-python", "sklearn": "scikit-learn", "skimage":
    "scikit-image", "av": "av", "soundfile": "soundfile",
}

#: Provided by ComfyUI itself or the standard library -- not ours to declare.
_HOST_PROVIDED = {
    "comfy", "folder_paths", "nodes", "server", "execution", "latent_preview",
    "comfy_extras", "app", "utils", "comfy_api", "comfy_api_nodes",
    "comfy_execution", "config",
}

#: Engines that run in their OWN isolated venv by design. Absent from ComfyUI's
#: interpreter is CORRECT for these -- flagging them would train a reader to
#: ignore the report. `bpy` is Blender's, only used by an optional mesh script.
_ISOLATED = {"chatterbox", "dia", "indextts", "bpy", "mathutils"}


def _first_party() -> set:
    """Module names that are OUR OWN files.

    Sibling modules are imported absolutely in places (`import
    story_orchestrator`, `from render_errors import ...`) because ComfyUI puts
    the pack directory on sys.path. Without this they look like missing PyPI
    packages, which is exactly the false alarm that makes an audit ignorable.
    """
    names = set()
    for sub in ("nodes", "scripts"):
        base = os.path.join(_REPO, sub)
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if fn.endswith(".py"):
                    names.add(fn[:-3])
            names.add(os.path.basename(root))
    return names


def _stdlib() -> set:
    names = set(getattr(sys, "stdlib_module_names", ()) or ())
    return names or {"os", "sys", "json", "io", "re", "math", "time"}


def top_level_imports() -> dict:
    """{module: [files that import it]} across the pack's own source."""
    found: dict = {}
    for root, dirs, files in os.walk(_REPO):
        # `tmp/` holds a VENDORED transformers copy; scanning it reports
        # every optional dependency transformers can import (wandb, typer,
        # yt_dlp...) as if this pack needed them. Scan our own source only.
        dirs[:] = [d for d in dirs
                   if d not in {".git", "__pycache__", ".venv", "kibitz-runs",
                                ".claude", "node_modules", "tests", "tmp",
                                "docs", "outputs", "workflows", "_otr_b_spikes"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                tree = ast.parse(io.open(path, encoding="utf-8",
                                         errors="replace").read())
            except SyntaxError:
                continue
            rel = os.path.relpath(path, _REPO).replace("\\", "/")
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.Import):
                    for a in node.names:
                        found.setdefault(a.name.split(".")[0], set()).add(rel)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:            # relative -- our own package
                        continue
                    mod = (node.module or "").split(".")[0]
                    if mod:
                        found.setdefault(mod, set()).add(rel)
    return {k: sorted(v) for k, v in found.items()}


def declared() -> set:
    out = set()
    req = os.path.join(_REPO, "requirements.txt")
    if os.path.isfile(req):
        for line in io.open(req, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                out.add(re.split(r"[<>=!\[;]", line)[0].strip().lower())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--declared-only", action="store_true")
    args = ap.parse_args(argv)

    std, host = _stdlib(), _HOST_PROVIDED | _first_party() | _ISOLATED
    imports = top_level_imports()
    decl = declared()

    third_party = {m: f for m, f in imports.items()
                   if m not in std and m not in host and not m.startswith("_")}

    undeclared, missing, ok = [], [], 0
    for mod in sorted(third_party):
        dist = _DIST.get(mod, mod).lower()
        is_declared = dist in decl or mod.lower() in decl
        installed = None
        if not args.declared_only:
            try:
                installed = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                installed = False
        if not is_declared:
            undeclared.append((mod, dist, third_party[mod][:2]))
        if installed is False:
            missing.append((mod, dist, third_party[mod][:2]))
        if is_declared and installed is not False:
            ok += 1

    print("OTR venv audit")
    print("  interpreter : %s" % sys.executable)
    print("  imports     : %d third-party module(s)" % len(third_party))
    print("  declared+ok : %d" % ok)

    if missing:
        print("\nNOT IMPORTABLE HERE (declared or not) -- this is the one that bites:")
        for mod, dist, where in missing:
            print("  %-18s pip install %-22s  used by %s"
                  % (mod, dist, ", ".join(where)))
    else:
        print("\n  every imported module resolves in this interpreter.")

    if undeclared:
        print("\nIMPORTED BUT NOT IN requirements.txt:")
        for mod, dist, where in undeclared:
            print("  %-18s (dist %-20s) used by %s"
                  % (mod, dist, ", ".join(where)))
    else:
        print("  every imported module is declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
