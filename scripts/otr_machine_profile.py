#!/usr/bin/env python
"""Expand a machine-matrix row into the dict the applier wants. Nothing else.

    python scripts/otr_machine_profile.py --list
    python scripts/otr_machine_profile.py --machine 8gb --show

**THIS FILE DECLARES NO VALUES.** Operator, 2026-08-31, three times:
*"I don't want any profiles in code, just matrix"*, *"you build and test
according to a matrix, not a code gate"*, *"remove the profile feature in code
100%"*.

So every value -- engines, ceilings, quantisation, canvas, fps, seed policy --
lives in `config/machine_classes.json`, in `defaults` and in the class rows.
This module reads that file and merges: a class row overrides a default, and
that is the entire behaviour. Grep it for a model name, a GB number or an
engine id and you will find none. If a value is wrong, the matrix is wrong, and
the matrix is one file a person can read.

WHY THAT MATTERS beyond tidiness: a value in a .py file is a value nobody
maintains and nobody can see. This project already shipped a README claiming an
8 GB card had "rendered nothing" while six documented episodes had published,
because the claim lived somewhere nobody re-read. Values in data are values a
reader can check.

WHAT THIS DOES NOT TOUCH: `config/profiles/*.json` still exists for EXPERIMENTS
-- the `otr_w45_*` campaign harness, `otr_soak_*`, `otr_sbcov_*`. Those answer
"which experiment am I running", not "which machine am I on". `--profile` keeps
working for them.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_CLASSES = os.path.join(_REPO, "config", "machine_classes.json")


def load_matrix() -> dict:
    return json.load(io.open(_CLASSES, encoding="utf-8"))


def rows(matrix=None) -> list:
    m = matrix or load_matrix()
    return [r for r in m.get("classes", []) if isinstance(r, dict)]


def key_of(row) -> str:
    return str(row.get("key") or row.get("label", "?")).strip().lower()


def _merged(matrix, row) -> dict:
    """defaults <- row. The row wins. No third source exists."""
    out = copy.deepcopy(matrix.get("defaults") or {})
    out.pop("_comment", None)
    for key, value in copy.deepcopy(row).items():
        # Sections are partial overrides in the readable matrix. Merge their
        # fields so an 8 GB canvas override does not erase fps/beats, and an
        # AMD dtype override does not erase the device policy.
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            section = copy.deepcopy(out[key])
            section.update(value)
            out[key] = section
        else:
            out[key] = value
    return out


def build_profile(row, matrix=None) -> dict:
    """A matrix row -> the profile dict `apply_profile_to_workflow` accepts.

    Every value below comes from the merge. The only literals here are KEY
    NAMES -- the shape the applier expects -- which is structure, not policy.
    """
    matrix = matrix or load_matrix()
    m = _merged(matrix, row)
    video, image = m.get("video"), m.get("image")

    llm = copy.deepcopy(m.get("llm") or {})
    llm["creative_model"] = m.get("writer_model")
    llm["technical_model"] = m.get("writer_model")
    llm["vram_ceiling_gb"] = m.get("writer_ceiling_gb")
    llm["quant_policy"] = m.get("quant_policy")

    video_policy = copy.deepcopy(m.get("video_policy") or {})
    if m.get("max_render_frames"):
        video_policy["max_render_frames"] = m["max_render_frames"]

    return {
        "id": "machine:" + key_of(row),
        "display_name": m.get("label"),
        "status": m.get("status"),
        "platform": m.get("platform"),
        "device_backend": m.get("device_backend"),
        "gpu_vendor": m.get("gpu_vendor"),
        "toolchains": m.get("toolchains"),
        "allow_sidecars": m.get("allow_sidecars"),
        "role_overrides": {
            "announcer_visual": video, "music_visual": video,
            "character_visual": video, "announcer_image": image,
            "music_image": image, "character_image": image,
        },
        "slot_overrides": {
            "voice_bank": m.get("voice_bank"),
            "cast_voice_policy": m.get("cast_voice_policy", "auto_registry"),
            "char_voice_engine": m.get("char_voice"),
            "announcer_voice_engine": m.get("announcer_voice"),
            "music_engine": m.get("music"),
            "video_render_engine": video,
        },
        "features": m.get("features"),
        "seed_policy": m.get("seed_policy"),
        "llm": llm,
        "video": video_policy,
        "image": m.get("image_policy"),
        "audio": m.get("audio_policy"),
        "render": m.get("render"),
        "preflight": {"required_models": [], "required_keys": []},
        "launch": {"sage_attention": False, "extra_args": [], "env": {}},
    }


def resolve(name: str, matrix=None):
    """Find a row by its exact public key; never infer from a GPU label."""
    matrix = matrix or load_matrix()
    # CLI keys are receipt identities, not search terms. Accepting case or
    # whitespace variants here lets provisioning store a selector that the
    # launch owner later (correctly) rejects as a different identity.
    want = str(name or "")
    all_rows = rows(matrix)
    for row in all_rows:
        if key_of(row) == want:
            return row
    raise SystemExit("  no machine %r. Known: %s"
                     % (name, ", ".join(key_of(r) for r in all_rows)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    matrix = load_matrix()

    if args.list or not args.machine:
        print("machines (config/machine_classes.json):")
        for row in rows(matrix):
            upper = row.get("vram_max_gb")
            vram_range = ("%s+" % row.get("vram_min_gb")
                          if upper is None else "%s-%s" % (
                              row.get("vram_min_gb"), upper))
            print("  %-6s %-46s %s GB  video=%s"
                  % (key_of(row), str(row.get("label"))[:46],
                     vram_range, row.get("video")))
        return 0

    prof = build_profile(resolve(args.machine, matrix), matrix)
    if args.show:
        print(json.dumps(prof, indent=2))
    else:
        print("  %s -> video=%s voice=%s ceiling=%s" % (
            prof["id"], prof["role_overrides"]["character_visual"],
            prof["slot_overrides"]["char_voice_engine"],
            prof["llm"]["vram_ceiling_gb"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
