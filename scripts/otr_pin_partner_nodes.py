"""S0 partner-node schema pinning (pass04 sec 1).

Imports each curated candidate class FROM THE LIVE INSTALL in-process
(authoritative -- hidden inputs visible; /object_info is NOT the capture
point) and writes the checked-in pin:

    nodes/_otr_shared/partner_nodes.yaml

Pin discipline:
- STRUCTURAL signature only: input names + type tokens split by
  required/optional/hidden, RETURN_TYPES, FUNCTION, async-ness. COMBO
  OPTION LISTS ARE EXCLUDED on purpose -- provider model lists churn
  weekly and would make the drift test cry wolf; a combo pins as
  "COMBO". Schema drift = structure drift.
- A missing candidate records status MISSING and prints LOUD; shipped
  rows are only those with status OK (runtime never drops rows -- a row
  pinned OK that later breaks on a target install fails closed at
  resolve/invoke with unsupported_schema).
- Selected output per row is pinned (multi-output nodes are ambiguous
  otherwise -- GPT R4 #4).

Run (from repo root, venv python):
    python scripts/otr_pin_partner_nodes.py
(Named WITHOUT the scripts/_*.py probe prefix on purpose: that pattern
is gitignored as throwaway; this is permanent tooling the drift test
invokes as a subprocess.)
"""
from __future__ import annotations

import datetime as _dt
import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_OUT = REPO_ROOT / "nodes" / "_otr_shared" / "partner_nodes.yaml"


def _resolve_comfy_root() -> Path:
    """The CORE checkout (has comfy_api_nodes/), NOT the workspace.
    Precedence: OTR_COMFY_CORE_ROOT env > the launcher's known install
    (scripts/_otr_soak_server_launch.cmd boots this exact main.py) >
    workspace-relative guess. Fails LOUD -- pinning against the wrong
    tree would pin the wrong schemas."""
    candidates = []
    env = os.environ.get("OTR_COMFY_CORE_ROOT", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(Path(r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI"))
    candidates.append(REPO_ROOT.parent.parent)
    for cand in candidates:
        if (cand / "comfy_api_nodes").is_dir():
            return cand
    print(f"no ComfyUI core with comfy_api_nodes/ found in {candidates}; "
          f"set OTR_COMFY_CORE_ROOT")
    raise SystemExit(4)  # rc 4 = core unresolvable (drift test skips)


COMFY_ROOT = _resolve_comfy_root()

# row_id -> (class_name, provider_id, selected_output_index, notes)
CURATED_ROWS = {
    "cloud_elevenlabs_flash": ("ElevenLabsTextToSpeech", "ELEVENLABS", 0,
                               "CHEAP-cand voice; flash/turbo tier via params"),
    "cloud_elevenlabs_tts":   ("ElevenLabsTextToSpeech", "ELEVENLABS", 0,
                               "BEST voice; premium voices via params"),
    "cloud_stability_audio":  ("StabilityTextToAudio", "STABILITY", 0,
                               "CHEAP-cand music"),
    "cloud_sonilo_music":     ("SoniloTextToMusic", "SONILO", 0,
                               "BEST music"),
    "cloud_recraft":          ("RecraftTextToImageNode", "RECRAFT", 0,
                               "CHEAP-cand stills"),
    "cloud_flux_pro":         ("Flux2ProImageNode", "BFL", 0,
                               "BEST stills; prompt continuity w/ flux_gen1"),
    "cloud_nano_banana_2":    ("GeminiNanoBanana2V2", "GEMINI", 0,
                               "BEST stills; reference-image consistency"),
    "cloud_kling_avatar":     ("KlingAvatarNode", "KLING", 0,
                               "talking default; reactivity=required_audio_ref"),
    "cloud_kling_lipsync":    ("KlingLipSyncAudioToVideoNode", "KLING", 0,
                               "reactivity=lipsync_overlay"),
    "cloud_seedance_2":       ("ByteDance2ReferenceNode", "BYTEDANCE", 0,
                               "music/b-roll reactive default; audio-ref + identity"),
    "cloud_wan_i2v":          ("Wan2ImageToVideoApi", "WAN", 0,
                               "mute_only OPT-DOWN row (operator amendment)"),
    # 2026-07-02 roster expansion (operator; pricing-driven tiers):
    "cloud_ideogram_v4":      ("IdeogramV4", "IDEOGRAM", 0,
                               "stills; best text rendering (posters/clues/"
                               "signage, F1 synergy); rendering_speed spans "
                               "TURBO 9.05cr -> QUALITY 27.16cr/image"),
    "cloud_seedream_2":       ("ByteDanceSeedreamNodeV2", "BYTEDANCE", 0,
                               "stills; cheapest stylization tier "
                               "(~7.4-8.4cr/image)"),
    "cloud_elevenlabs_voice_selector": ("ElevenLabsVoiceSelector",
                               "ELEVENLABS", 0,
                               "AUX helper, no billing; produces the "
                               "ELEVENLABS_VOICE input the TTS row requires"),
}

AUTH_HIDDEN_NAMES = ("auth_token_comfy_org", "api_key_comfy_org")


def _type_token(spec) -> str:
    """Stable structural token for one input spec entry."""
    if isinstance(spec, (list, tuple)) and spec:
        head = spec[0]
        if isinstance(head, (list, tuple)):
            return "COMBO"  # option list EXCLUDED by design
        return str(head)
    return str(spec)


def _pin_inputs(input_types: dict) -> dict:
    out = {}
    for section in ("required", "optional", "hidden"):
        sec = input_types.get(section) or {}
        out[section] = {str(name): _type_token(spec)
                        for name, spec in sec.items()}
    return out


def _iter_api_node_modules():
    import comfy_api_nodes  # noqa: PLC0415 -- live-install import is the point
    for info in pkgutil.iter_modules(comfy_api_nodes.__path__):
        yield f"comfy_api_nodes.{info.name}"


def _find_class(class_name: str, module_cache: dict):
    for mod_name in list(module_cache) or []:
        cls = getattr(module_cache[mod_name], class_name, None)
        if cls is not None:
            return mod_name, cls
    for mod_name in _iter_api_node_modules():
        if mod_name in module_cache:
            continue
        try:
            module_cache[mod_name] = importlib.import_module(mod_name)
        except Exception as exc:  # record, keep scanning
            module_cache[mod_name] = None
            print(f"  [warn] import {mod_name} failed: {exc}")
            continue
        cls = getattr(module_cache[mod_name], class_name, None)
        if cls is not None:
            return mod_name, cls
    return None, None


def _comfy_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(COMFY_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def pin_all() -> dict:
    sys.path.insert(0, str(COMFY_ROOT))
    os.chdir(COMFY_ROOT)
    module_cache: dict = {}
    rows = {}
    for row_id, (class_name, provider_id, out_idx, notes) in CURATED_ROWS.items():
        print(f"pinning {row_id} ({class_name}) ...")
        try:
            mod_name, cls = _find_class(class_name, module_cache)
            if cls is None:
                rows[row_id] = {"status": "MISSING", "class_name": class_name,
                                "provider_id": provider_id, "notes": notes,
                                "error": "class not found in comfy_api_nodes"}
                print(f"  MISSING: {class_name}")
                continue
            input_types = cls.INPUT_TYPES()
            fn_name = getattr(cls, "FUNCTION", None)
            fn = getattr(cls, fn_name, None) if fn_name else None
            pinned_inputs = _pin_inputs(input_types)
            hidden = pinned_inputs.get("hidden", {})
            return_types = [str(t) for t in (getattr(cls, "RETURN_TYPES", ()) or ())]
            if return_types and out_idx >= len(return_types):
                raise ValueError(
                    f"selected_output {out_idx} out of range for "
                    f"RETURN_TYPES {return_types}")
            rows[row_id] = {
                "status": "OK",
                "import_path": mod_name,
                "class_name": class_name,
                "provider_id": provider_id,
                "function": fn_name,
                "is_async": bool(fn and inspect.iscoroutinefunction(fn)),
                "inputs": pinned_inputs,
                "auth_hidden_present": [n for n in AUTH_HIDDEN_NAMES
                                        if n in hidden],
                "seed_supported": ("seed" in pinned_inputs["required"]
                                   or "seed" in pinned_inputs["optional"]),
                "return_types": return_types,
                "selected_output": out_idx,
                "api_node": bool(getattr(cls, "API_NODE", False)),
                "category": str(getattr(cls, "CATEGORY", "")),
                "notes": notes,
            }
            print(f"  OK: {mod_name}.{class_name} fn={fn_name} "
                  f"async={rows[row_id]['is_async']} "
                  f"returns={return_types} "
                  f"auth_hidden={rows[row_id]['auth_hidden_present']}")
        except Exception as exc:
            rows[row_id] = {"status": "ERROR", "class_name": class_name,
                            "provider_id": provider_id, "notes": notes,
                            "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR pinning {row_id}: {exc}")
            traceback.print_exc()
    return rows


def write_yaml(rows: dict) -> None:
    import yaml  # ships with ComfyUI's env
    doc = {
        "pin_meta": {
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "comfyui_commit": _comfy_commit(),
            "generator": "scripts/otr_pin_partner_nodes.py",
            "combo_options_excluded": True,
            "schema_version": "pin-2026-07-02",
        },
        "rows": rows,
    }
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(YAML_OUT, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(doc, fh, sort_keys=True, allow_unicode=False)
    print(f"\nwrote {YAML_OUT}")


def check_against_yaml() -> int:
    """--check mode (drift test backend): re-derive live signatures and
    compare to the checked-in yaml WITHOUT rewriting it. Runs in its own
    interpreter -- importing the comfy core inside the pytest process
    corrupts pytest teardown, so the suite calls this as a subprocess.
    Exit codes: 0 match, 3 drift (prints JSON), 4 core unresolvable."""
    import yaml as _yaml
    pinned = _yaml.safe_load(YAML_OUT.read_text(encoding="utf-8"))["rows"]
    live = pin_all()
    drift = {}
    for rid, prow in pinned.items():
        if prow["status"] != "OK":
            continue
        lrow = live.get(rid)
        if lrow is None or lrow["status"] != "OK":
            drift[rid] = f"no longer pinnable: {(lrow or {}).get('error')}"
            continue
        drifted_fields = [
            field_name
            for field_name in ("import_path", "class_name", "function",
                               "inputs", "return_types", "is_async")
            if lrow[field_name] != prow[field_name]
        ]
        if drifted_fields:
            drift[rid] = drifted_fields
    if drift:
        print("DRIFT: " + json.dumps(drift, sort_keys=True))
        return 3
    print("pin matches live install")
    return 0


def main(argv) -> int:
    if "--check" in argv:
        return check_against_yaml()
    rows = pin_all()
    write_yaml(rows)
    ok = sum(1 for r in rows.values() if r["status"] == "OK")
    bad = {rid: r["status"] for rid, r in rows.items() if r["status"] != "OK"}
    print(f"\npinned OK: {ok}/{len(rows)}")
    if bad:
        print(f"NOT PINNED (row ships ONLY when OK): {json.dumps(bad)}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(4)
