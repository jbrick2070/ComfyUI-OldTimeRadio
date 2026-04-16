"""One-shot AST + import check for otr_v2/hyworld/worker.py."""
from __future__ import annotations
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
target = ROOT / "otr_v2" / "hyworld" / "worker.py"
src = target.read_text(encoding="utf-8")

try:
    ast.parse(src)
    print(f"AST_OK lines={len(src.splitlines())} bytes={len(src)}")
except SyntaxError as e:
    print(f"AST_FAIL line={e.lineno} col={e.offset}: {e.msg}")
    sys.exit(1)

# Try a real import (catches NameError / runtime issues at module top level)
sys.path.insert(0, str(ROOT))
try:
    import importlib
    mod = importlib.import_module("otr_v2.hyworld.worker")
    print(f"IMPORT_OK module={mod.__name__}")
    print(f"  has_run_stub={hasattr(mod, 'run_stub')}")
    print(f"  has_camera_to_motion={hasattr(mod, '_camera_to_motion')}")
    print(f"  has_make_motion_clip={hasattr(mod, '_make_motion_clip')}")
    print(f"  has_find_ffmpeg={hasattr(mod, '_find_ffmpeg')}")

    # Smoke-test the camera lookup
    cases = [
        ("slow handheld, close", "slow handheld"),
        ("locked off, wide", "locked off"),
        ("clean push-in, centered", "clean push"),
        ("whip-pan, short focal length", "whip-pan"),
        ("low angle, looking up", "low angle"),
        ("macro detail, shallow focus", "macro detail"),
        ("slow drift, medium lens", "slow drift"),
        ("fast dolly, canted", "fast dolly"),
        ("", "default_drift"),
        ("garbage adjective", "default_drift"),
    ]
    fails = 0
    for camera, expected_label in cases:
        label, vfilter = mod._camera_to_motion(camera, 9.0, 1280, 720)
        ok = label == expected_label
        marker = "OK" if ok else "FAIL"
        print(f"  {marker} camera='{camera}' -> label='{label}' (expected '{expected_label}')")
        if not ok:
            fails += 1
        # Sanity: filter must include our fixed FPS and dims
        for needle in ["zoompan=", f"s=1280x720", "fps=24"]:
            if needle not in vfilter:
                print(f"    MISSING_NEEDLE '{needle}' in vfilter: {vfilter[:120]}")
                fails += 1
    if fails:
        print(f"CAMERA_LOOKUP_FAILED {fails} failures")
        sys.exit(2)
    print("CAMERA_LOOKUP_OK")

    # ffmpeg discovery
    ff = mod._find_ffmpeg()
    print(f"  ffmpeg_resolved={ff!r}")
except Exception as e:
    print(f"IMPORT_FAIL {type(e).__name__}: {e}")
    raise
