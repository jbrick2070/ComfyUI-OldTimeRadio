#!/usr/bin/env python
"""Generate `docs/MODEL_ASSET_INDEX.md` -- "to use X, download Z".

    python scripts/otr_asset_index.py            # write the doc
    python scripts/otr_asset_index.py --stdout   # print it instead

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND. The question it answers -- what
does a fresh machine have to download before lane X will run -- is asked by
every new user and by every AI assistant helping one, and it is exactly the
question a hand-written table gets wrong six weeks later. Every row here is read
out of the engine module that owns the requirement, so a lane that changes its
weights changes this document the next time it is generated.

WHAT IT DELIBERATELY WILL NOT DO IS GUESS. An engine that does not declare its
weights in a machine-readable way is reported as "not declared in code", never
filled in from memory. A user sent looking for a filename that was invented by a
documentation generator loses more time than one told plainly that the answer is
not in the code -- and the honest row is also the one that gets fixed.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: A weight FILE the engine names literally.
_WEIGHT_RE = re.compile(
    r'["\']([A-Za-z0-9_.\-]+\.(?:safetensors|ckpt|pth|gguf|bin|onnx))["\']')
#: A Hugging Face repo id, restricted to publishers this project actually uses
#: so that arbitrary `a/b` strings in comments do not become install steps.
_REPO_RE = re.compile(r'["\']([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)["\']')
_PUBLISHERS = (
    "Lightricks", "Kijai", "city96", "Comfy-Org", "guoyww", "hexgrad",
    "facebook", "google", "mistralai", "stabilityai", "tencent", "Wan-AI",
    "Skywork", "MiniMax", "IndexTeam", "SWivid", "nari-labs", "ResembleAI",
)
_ENV_RE = re.compile(r'["\'](OTR_[A-Z0-9_]+)["\']')
#: An engine that shells out to its OWN interpreter is a separate INSTALL, not a
#: download -- the distinction that matters most to somebody setting this up.
_EXTERNAL_RE = re.compile(r'_venv_python|OTR_[A-Z0-9_]*_VENV')
_GATED_RE = re.compile(r'requires_hf_token\s*=\s*True|HF_TOKEN')


#: An engine that names no weights is not automatically an engine that needs
#: none. Cloud and procedural lanes genuinely need nothing; anything else with
#: an empty list is a GAP in what the code declares, and must say so.
_NO_ASSET_PREFIXES = ("cloud_", "google_", "viz_", "visualizer")
_OPERATOR_ONLY_FETCH_LANES = {"minimax_h3"}


def _public_engine_resolver():
    """Load the dependency-free public-id resolver from its sole owner.

    Profiles intentionally save public menu ids (for example
    ``wan22_high_video``), while asset rows are keyed by the internal engine
    module (``wan_ti2v``). Counting the raw strings drops those profiles from
    the generated index even though they select that exact engine.
    """
    path = os.path.join(_REPO, "nodes", "_otr_shared", "public_engines.py")
    spec = importlib.util.spec_from_file_location(
        "otr_asset_index_public_engines", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the public engine resolver from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolver = getattr(module, "resolve_engine_id", None)
    if not callable(resolver):
        raise ImportError("public_engines.py has no callable resolve_engine_id")
    return resolver


def _imports_of(path: str, src: str) -> list:
    pkg = os.path.dirname(path)
    out = []
    for mod in re.findall(r"^\s*from\s+\.([a-zA-Z0-9_]+)\s+import", src, re.M):
        cand = os.path.join(pkg, mod + ".py")
        if os.path.isfile(cand) and cand not in out:
            out.append(cand)
    return out


def _shared_helpers(paths: list, threshold: int = 3) -> set:
    """Sibling modules imported by MANY engines -- their filenames belong to
    nobody in particular.

    Both directions of this were wrong once, which is why the rule is explicit.
    Scanning only the engine module said `ltx25` needs "nothing on disk" when it
    needs gigabytes, because its models are declared in a helper. Then following
    every import said `viz_camera` -- a purely procedural lane -- needs the
    AnimateDiff checkpoints, because it shares a common helper that names them.
    A false all-clear sends somebody to render instead of downloading; a false
    requirement sends them to download several GB they will never use. So follow
    imports, but do not attribute anything reachable from a helper that most
    engines import.
    """
    seen = {}
    for p in paths:
        src = io.open(p, encoding="utf-8", errors="replace").read()
        for imp in _imports_of(p, src):
            seen[imp] = seen.get(imp, 0) + 1
    return {p for p, n in seen.items() if n >= threshold}


_SHARED: set = set()


def _scan(path: str) -> dict:
    src = io.open(path, encoding="utf-8", errors="replace").read()
    for extra in _imports_of(path, src):
        if extra in _SHARED:
            continue
        src += "\n" + io.open(extra, encoding="utf-8", errors="replace").read()
    return {
        "weights": sorted(set(_WEIGHT_RE.findall(src))),
        "repos": sorted({r for r in _REPO_RE.findall(src)
                         if any(r.startswith(p) for p in _PUBLISHERS)}),
        "env": sorted({e for e in _ENV_RE.findall(src)
                       if e.endswith(("_DIR", "_VENV", "_PATH", "_ROOT"))}),
        "external": bool(_EXTERNAL_RE.search(src)),
        "gated": bool(_GATED_RE.search(src)),
    }


def collect_engines() -> list:
    global _SHARED
    every = []
    for pattern in ("nodes/_otr_video_engines/eng_*.py",
                    "nodes/_otr_audio_engines/eng_*.py"):
        every += sorted(glob.glob(os.path.join(_REPO, pattern)))
    _SHARED = _shared_helpers(every)
    out = []
    for kind, pattern in (("video", "nodes/_otr_video_engines/eng_*.py"),
                          ("audio", "nodes/_otr_audio_engines/eng_*.py")):
        for path in sorted(glob.glob(os.path.join(_REPO, pattern))):
            name = os.path.basename(path)[len("eng_"):-len(".py")]
            row = _scan(path)
            row.update(kind=kind, engine=name,
                       module=os.path.relpath(path, _REPO).replace("\\", "/"))
            out.append(row)
    return out


def collect_profiles() -> dict:
    """profile id -> the engines it selects, so a user can start from a profile."""
    by_engine = {}
    resolve_engine_id = _public_engine_resolver()
    video_keys = {
        "character_visual", "announcer_visual", "music_visual",
        "video_render_engine",
    }
    for path in sorted(glob.glob(os.path.join(_REPO, "config/profiles/*.json"))):
        try:
            doc = json.load(io.open(path, encoding="utf-8"))
        except Exception:
            continue
        pid = doc.get("id") or os.path.basename(path)[:-len(".json")]
        roles = doc.get("role_overrides", {}) or {}
        slots = doc.get("slot_overrides", {}) or {}
        for key in ("character_visual", "announcer_visual", "music_visual",
                    "char_voice_engine", "announcer_voice_engine",
                    "music_engine", "video_render_engine"):
            eng = roles.get(key) or slots.get(key)
            if eng:
                # Public-id resolution owns the VIDEO surface only. Voice and
                # music ids have separate registries and must pass unchanged.
                internal = resolve_engine_id(eng) if key in video_keys else eng
                by_engine.setdefault(str(internal), set()).add(pid)
    return {k: sorted(v) for k, v in by_engine.items()}


def _fetcher_lanes() -> dict:
    """Lanes `otr_fetch_lane_weights.py` can install with no manual steps."""
    path = os.path.join(_REPO, "scripts", "otr_fetch_lane_weights.py")
    spec = importlib.util.spec_from_file_location("otr_asset_index_fetcher", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lanes = getattr(module, "LANES", {})
    return {str(name): True for name in lanes} if isinstance(lanes, dict) else {}


def render() -> str:
    engines = collect_engines()
    profiles = collect_profiles()
    fetchable = _fetcher_lanes()
    L = []
    A = L.append
    A("# Model asset index -- what to download for each mode\n")
    A("**GENERATED by `scripts/otr_asset_index.py`. Do not hand-edit; "
      "regenerate.** Every requirement below is read out of the engine module "
      "named in its row.\n")
    A("Read this if you are setting OTR up on a new machine, or you are an AI "
      "assistant helping somebody do that. It answers one question: *if I want "
      "to use engine X, what has to be on disk first?*\n")
    A("## How OTR behaves when an asset is missing\n")
    A("**It refuses by name, and it never silently substitutes.** A missing "
      "weight produces a named `EngineUnusable` / `MISSING_MODEL` refusal "
      "naming the engine, the role and the file. There is no quiet fallback to "
      "a different engine -- that was removed deliberately, because a render "
      "that quietly swaps the voice or the video engine is worse than one that "
      "stops and says why.\n")
    A("So **the error message is the install instruction.** Read it literally: "
      "it names what to fetch.\n")
    A("## Where assets go\n")
    A("```\n"
      "OTR_COMFYUI_MODELS_ROOT   models root       default C:\\ComfyUI-Models\n"
      "                                            (Linux/pods: set it, or the\n"
      "                                            Windows default becomes a\n"
      "                                            literal directory nothing scans)\n"
      "HF_HOME                   writer/voice/music cache, default ~/.cache/huggingface\n"
      "<models_root>/TTS/refs/<engine>/*.wav       voice-cloning reference clips\n"
      "```\n")
    A("Resolve the models root through `nodes/_otr_gguf_backend.py::_models_root()` "
      "rather than guessing -- a `find` under the ComfyUI tree proves nothing.\n")

    A("## Public one-command paths\n")
    if fetchable:
        A("`scripts/otr_fetch_lane_weights.py` installs these public lanes with "
          "no account and no manual step:\n")
        A("```\n" + "\n".join(
            "python scripts/otr_fetch_lane_weights.py %s" % k
            for k in sorted(set(fetchable) - _OPERATOR_ONLY_FETCH_LANES))
          + "\n```\n")
    A("The complete H3 manifest is deliberately explicit and operator-local; "
      "it is never selected by a public profile or machine bundle:\n\n"
      "```\npython scripts/otr_fetch_lane_weights.py minimax_h3\n```\n")
    A("Anything not listed there is a manual install -- see its row below.\n")

    for kind, title in (("video", "Video engines"), ("audio", "Audio and voice engines")):
        A("## %s\n" % title)
        A("| engine | needs | how | used by profiles |")
        A("|---|---|---|---|")
        for row in engines:
            if row["kind"] != kind:
                continue
            needs = []
            if row["external"]:
                needs.append("**a SEPARATE project + its own venv**")
            if row["weights"]:
                needs.append("%d weight file(s)" % len(row["weights"]))
            if row["repos"]:
                needs.append(", ".join("`%s`" % r for r in row["repos"][:2]))
            if not needs:
                if row["engine"].startswith(_NO_ASSET_PREFIXES):
                    needs.append("nothing on disk")
                else:
                    needs.append("**not declared in code -- verify**")
            how = "auto (HF cache)" if row["repos"] and not row["weights"] else ""
            if row["engine"] == "humo":
                # One module implements both tiers. The automatic lane is the
                # complete 14B recipe only; claiming it installs the 1.7B DiT
                # would recreate the wrong-artifact fresh-install trap.
                how = ("14B: `otr_fetch_lane_weights.py humo`; 1.7B: "
                       "[exact manual tier](RUNPOD_INSTALL.md)")
            elif row["engine"].startswith("humo"):
                how = "[exact manual tier](RUNPOD_INSTALL.md)"
            elif row["engine"] == "minimax_h3":
                how = "explicit operator-local `otr_fetch_lane_weights.py minimax_h3`"
            elif row["engine"] == "kokoro":
                # Voices and the ONNX model are fetched by the boot prefetch
                # (`_otr_kokoro_voice_prefetch`); the torch model rides the HF
                # cache. Nothing is placed by hand and nothing fetches mid-render.
                how = "auto (boot prefetch: voices + ONNX model; torch model via HF cache)"
            elif row["engine"] in fetchable:
                how = "`otr_fetch_lane_weights.py %s`" % row["engine"]
            elif row["external"]:
                how = "manual, see below"
            elif row["weights"]:
                how = "manual download"
            if row["gated"]:
                how += " **(HF_TOKEN)**"
            used = profiles.get(row["engine"], [])
            used_s = ("%d profile(s)" % len(used)) if used else "-"
            A("| `%s` | %s | %s | %s |" % (
                row["engine"], "; ".join(needs), how or "-", used_s))
        A("")

    A("## Engines that are a separate INSTALL, not a download\n")
    A("These shell out to their own Python interpreter **on purpose**: their "
      "dependencies conflict with ComfyUI's, so they run as isolated subprocess "
      "workers. That is why they cannot simply be bundled into this pack -- "
      "vendoring them back into one process reintroduces the exact dependency "
      "clash the isolation exists to prevent, and their weights are far larger "
      "than a node pack should ship.\n")
    for row in engines:
        if not row["external"]:
            continue
        A("### `%s`\n" % row["engine"])
        A("Declared in `%s`. Environment variables it reads:\n" % row["module"])
        if row["env"]:
            A("```\n" + "\n".join(row["env"]) + "\n```\n")
        else:
            A("_None declared with a recognised suffix; read the module._\n")
        if row["engine"] == "indextts2":
            A("Measured on the reference machine 2026-08-30:\n")
            A("```\n"
              "index-tts/                18.93 GB total\n"
              "  checkpoints/            11.06 GB   <- portable, fetch on target\n"
              "  .venv/                   7.78 GB   <- PLATFORM SPECIFIC, rebuild\n"
              "```\n")
            A("The venv is **not portable**: a Windows venv has "
              "`Scripts/python.exe`, a Linux one `bin/python`, and 39,759 files "
              "of compiled wheels for the wrong platform. On a new machine, "
              "clone the project, build its venv natively, fetch checkpoints, "
              "then point the three variables above at them.\n")
            A("It also needs an authorized male and female reference WAV plus a "
              "full portable bank that preserves every non-Index row. The exact "
              "recipe is in [RUNPOD_INSTALL.md](RUNPOD_INSTALL.md). "
              "The clips are not shipped; a missing or mismatched registered ref "
              "is a named refusal, never a fallback.\n")
    A("## Exactly which files each engine names\n")
    A("Copied from the engine module and the siblings it imports, so these are "
      "the literal strings the code looks for on disk.\n")
    A("**These are the files an engine REFERENCES, which can be more than it "
      "REQUIRES.** Several engines name alternatives -- two motion modules, or "
      "the same tensor at three quantizations -- and need one of them, not all. "
      "Checked against the fetcher: `ghost_signal_official` lists all three "
      "files `otr_fetch_lane_weights.py haunted` installs, plus two other "
      "AnimateDiff motion modules it can also drive.\n")
    A("So where a lane appears in the one-command list above, **the fetcher is "
      "authoritative** -- it installs a set known to work together. Use the list "
      "below to understand a lane, or to install one the fetcher does not cover, "
      "and expect to choose among alternatives rather than downloading every "
      "line.\n")
    for row in engines:
        if not (row["weights"] or row["repos"]):
            continue
        A("**`%s`** -- `%s`\n" % (row["engine"], row["module"]))
        if row["repos"]:
            A("- Hugging Face: " + ", ".join("`%s`" % r for r in row["repos"]))
        for w in row["weights"]:
            A("- `%s`" % w)
        A("")

    A("## If you only want one working episode\n")
    A("The shortest complete path, all ungated, no token:\n")
    A("```\n"
      "python scripts/otr_fetch_lane_weights.py haunted\n"
      "# writer, voices and music download themselves into HF_HOME on first use\n"
      "python scripts/otr_canonical_api_run.py --profile otr_nvidia_8gb_haunted "
      "--act-count 1\n"
      "```\n")
    A("That lane is `animatediff15_v3_haunted_video` with `kokoro` voices and "
      "`musicgen`, and it is the one proven on 8 GB, 16 GB and 24 GB cards.\n")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args(argv)
    text = render()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    dest = os.path.join(_REPO, "docs", "MODEL_ASSET_INDEX.md")
    io.open(dest, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s (%d bytes)" % (dest, len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
