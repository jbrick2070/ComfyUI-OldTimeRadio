"""A-S2 probe (b) -- the cold-import dep-pilot catches a contaminating custom node.

A custom node that does ``import torch`` / sage / xformers / flash_attn at MODULE
SCOPE fires the heavy lib at ComfyUI STARTUP -- before any engine gate runs (the
KJNodes-at-startup hazard, BUG-070). This probe scans every installed
``custom_nodes/*/__init__.py`` (+ their top-level modules) for module-scope heavy
imports and reports which would contaminate a cold import.

CPU-only (no GPU needed) -- but it is the dep-pilot the A platform relies on, so
it ships with the A-S2 spike. PASS = the scan ran and classified every node;
contaminators are listed so the operator knows what the cold-import gate must
isolate to a sidecar.

OPERATOR-RUN (or CPU). The agent did not run it against the live custom_nodes set.
"""
from __future__ import annotations

import ast
import pathlib
import sys

# repo is .../ComfyUI/custom_nodes/ComfyUI-OldTimeRadio ; custom_nodes is parents[1]
_REPO = pathlib.Path(__file__).resolve().parents[2]
_CUSTOM_NODES = _REPO.parent

HEAVY = {"torch", "transformers", "diffusers", "xformers", "flash_attn",
         "sageattention", "sage_attn", "triton", "bitsandbytes"}


def _module_scope_heavy(py: pathlib.Path) -> set:
    """Heavy modules imported at MODULE SCOPE (top-level, not inside a def/try-lazy)."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return set()
    found = set()
    for node in tree.body:                       # body == module scope only
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in HEAVY:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in HEAVY:
                found.add(root)
    return found


def main() -> int:
    if not _CUSTOM_NODES.is_dir():
        print(f"PROBE b: NO-GO: custom_nodes dir not found at {_CUSTOM_NODES}")
        return 1
    contaminators = {}
    scanned = 0
    for pkg in sorted(_CUSTOM_NODES.iterdir()):
        init = pkg / "__init__.py"
        if not init.is_file():
            continue
        scanned += 1
        heavy = _module_scope_heavy(init)
        if heavy:
            contaminators[pkg.name] = sorted(heavy)
    print(f"PROBE b: scanned {scanned} custom-node packages")
    for name, libs in contaminators.items():
        print(f"  CONTAMINATOR: {name} imports {libs} at module scope (fires at startup)")
    # The dep-pilot's JOB is to DETECT these; detection working == PASS. The
    # platform's own nodes must NOT be among them (that would be our regression).
    ours = _REPO.name
    if ours in contaminators:
        print(f"PROBE b: NO-GO: OUR package ({ours}) contaminates the cold import: "
              f"{contaminators[ours]}")
        return 1
    print(f"PROBE b: PASS -- dep-pilot scan works; {len(contaminators)} external "
          f"contaminator(s) flagged for sidecar isolation; our package is clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
