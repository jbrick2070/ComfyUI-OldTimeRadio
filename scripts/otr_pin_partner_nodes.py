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
    # word_razzle Phase 1 (2026-07-03): the animated word-card i2v row. The
    # --audit-i2v Phase 0 verdict picked PixverseImageToVideoNode -- promptable,
    # non-V3, REQUIRED image init + prompt + seed + duration_seconds +
    # motion_mode (the motion-strength lever the spike wants). Served by the
    # `word_razzle` adapter (eng_cloud_video); mute_only (init_image+text_prompt).
    "cloud_pixverse_i2v":     ("PixverseImageToVideoNode", "PIXVERSE", 0,
                               "word_razzle animated word-card i2v; motion_mode "
                               "motion-strength; cheapest passing seed row (Phase 0)"),
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


# ---------------------------------------------------------------------------
# --audit-i2v (word_razzle Phase 0, 2026-07-03): REPORT-ONLY discovery of a
# promptable, non-V3, image-to-video cloud row for the animated word card.
# Walks the WHOLE live comfy_api_nodes catalog (CURATED_ROWS cannot discover
# anything new), applies code-checkable filters per class, and writes a
# machine-readable JSON report. NON-MUTATING (never touches partner_nodes.yaml).
# The classifier (_classify_video_row) is PURE so the suite tests it with
# synthetic schemas -- importing the comfy core inside pytest corrupts teardown
# (same reason --check runs as a subprocess).
# ---------------------------------------------------------------------------
AUDIT_OUT_DIR = REPO_ROOT / "docs" / "2026-07-03-word-razzle"

_STRING_TOKENS = {"STRING"}
_IMAGE_TOKENS = {"IMAGE"}
_AUDIO_TOKENS = {"AUDIO"}
_VIDEO_TOKENS = {"VIDEO"}
#: input NAME hints for an image-init when the token is not a bare IMAGE
_IMAGE_NAME_HINTS = ("image", "frame", "reference", "init", "start", "first")
_DURATION_NAME_HINTS = ("duration", "length", "fps", "frames", "seconds", "second")


def _is_v3_token(tok) -> bool:
    """A dynamic V3 combo token (model lists that only resolve at runtime --
    the seedance_2 / wan_i2v blocker). Heuristic on the observable token
    string: a bare V3/DYNAMIC marker. Combo option lists (pinned "COMBO") are
    NOT v3 by themselves."""
    s = str(tok).upper()
    return ("DYNAMIC" in s and "V3" in s) or "DYNAMICCOMBO" in s


def _classify_video_row(class_name, inputs, return_types):
    """PURE audit classifier for one video-returning class.

    ``inputs`` is ``{"required"/"optional"/"hidden": {name: token}}`` (the
    :func:`_pin_inputs` shape). Returns the audit record fields: the exact
    prompt kwarg NAME, the image-init NAME, a required-audio NAME (or None),
    seed support, duration inputs, a ``contains_dynamic_v3`` flag, per-filter
    pass/fail, an overall ``pass`` and human ``reasons``. kling-Avatar-style
    classes are EXEMPT from the no-required-audio filter (they still SUPPLY
    audio_ref -- music/spoken beats have audio) but are scored, not auto-passed.
    """
    req = inputs.get("required", {}) or {}
    opt = inputs.get("optional", {}) or {}
    allin = {**opt, **req}                     # required wins on name clash
    rts = {str(t).upper() for t in (return_types or [])}

    def _find_string_prompt():
        cands = [n for n, t in allin.items() if str(t).upper() in _STRING_TOKENS]
        for n in cands:
            if n.lower() == "prompt":
                return n
        for n in cands:
            if "prompt" in n.lower():
                return n
        return None

    def _find_image():
        for n, t in allin.items():
            if str(t).upper() in _IMAGE_TOKENS:
                return n
        for n in allin:
            if any(h in n.lower() for h in _IMAGE_NAME_HINTS):
                return n
        return None

    prompt_kwarg = _find_string_prompt()
    image_input = _find_image()
    required_audio = next(
        (n for n, t in req.items() if str(t).upper() in _AUDIO_TOKENS), None)
    seed_supported = ("seed" in req) or ("seed" in opt)
    duration_inputs = sorted(
        n for n in allin if any(h in n.lower() for h in _DURATION_NAME_HINTS))
    dyn_v3 = [f"{sec}.{n}={t}" for sec in ("required", "optional")
              for n, t in (inputs.get(sec, {}) or {}).items() if _is_v3_token(t)]
    contains_dynamic_v3 = bool(dyn_v3)
    audio_exempt = "avatar" in class_name.lower()

    filters = {
        "returns_video": bool(_VIDEO_TOKENS & rts),
        "has_image_input": image_input is not None,
        "has_prompt": prompt_kwarg is not None,
        "no_required_audio": (required_audio is None) or audio_exempt,
        "non_v3": not contains_dynamic_v3,
    }
    reasons = []
    if not filters["returns_video"]:
        reasons.append("does not return VIDEO")
    if not filters["has_image_input"]:
        reasons.append("no IMAGE-init input")
    if not filters["has_prompt"]:
        reasons.append("no STRING prompt input")
    if required_audio is not None and not audio_exempt:
        reasons.append(f"requires AUDIO input '{required_audio}'")
    if contains_dynamic_v3:
        reasons.append("dynamic V3 combo input(s): " + ", ".join(dyn_v3))
    if not seed_supported:
        reasons.append("no seed input (flag, not a hard fail)")
    if audio_exempt and required_audio is not None:
        reasons.append(f"AVATAR audio-exempt (supplies audio_ref '{required_audio}')")
    passed = all(filters.values())
    return {
        "class_name": class_name,
        "return_types": list(return_types or []),
        "inputs": inputs,
        "prompt_kwarg": prompt_kwarg,
        "image_input": image_input,
        "required_audio": required_audio,
        "audio_available_exempt": bool(audio_exempt and required_audio is not None),
        "seed_supported": seed_supported,
        "duration_inputs": duration_inputs,
        "contains_dynamic_v3": contains_dynamic_v3,
        "dynamic_v3_inputs": dyn_v3,
        "max_duration_s": None, "max_resolution": None, "_limits_source": "unknown",
        "credits_per_second": None,          # pricing comes from PRICING.md, not yaml
        "filters": filters,
        "pass": passed,
        "reasons": reasons,
    }


def _audit_verdict(rows) -> str:
    """PURE verdict from classified rows (the decision tree, testable without
    the live core): a passing promptable non-V3 i2v row -> CANDIDATE_FOUND;
    else any V3-blocked row -> BLOCKED_ON_V3_EXPANSION (waits on the S1
    V3-expansion); else BLOCKED_NO_USABLE_ROW."""
    if any(r.get("pass") for r in rows):
        return "CANDIDATE_FOUND"
    if any(r.get("contains_dynamic_v3") for r in rows):
        return "BLOCKED_ON_V3_EXPANSION"
    return "BLOCKED_NO_USABLE_ROW"


def _iter_video_classes(module_cache):
    """Yield ``(module_name, class_name, cls)`` for every class DEFINED in a
    comfy_api_nodes module whose RETURN_TYPES contains VIDEO. Per-module import
    failures are EXPECTED (partial env) -- logged + skipped, never fatal."""
    seen = set()
    for mod_name in _iter_api_node_modules():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:              # noqa: BLE001 -- expected + logged
            print(f"  [warn] import {mod_name} failed: {exc}")
            continue
        module_cache[mod_name] = mod
        for cname in dir(mod):
            cls = getattr(mod, cname, None)
            if not inspect.isclass(cls):
                continue
            if getattr(cls, "__module__", "") != mod_name:
                continue                      # skip re-exports
            rt = getattr(cls, "RETURN_TYPES", None)
            if not rt:
                continue
            try:
                tokens = {str(t).upper() for t in rt}
            except TypeError:
                continue
            if "VIDEO" not in tokens:
                continue
            key = (mod_name, cname)
            if key in seen:
                continue
            seen.add(key)
            yield mod_name, cname, cls


def audit_i2v() -> dict:
    """Walk the live catalog, classify every video-returning class, and return
    the report dict. REPORT-ONLY; never mutates the yaml."""
    sys.path.insert(0, str(COMFY_ROOT))
    os.chdir(COMFY_ROOT)
    module_cache: dict = {}
    rows = []
    for mod_name, cname, cls in _iter_video_classes(module_cache):
        try:
            input_types = cls.INPUT_TYPES()
            inputs = _pin_inputs(input_types)
            schema_api = "v1"
        except Exception as exc:              # noqa: BLE001 -- V3/opaque schema
            rows.append({
                "row": f"{mod_name}.{cname}", "import_path": mod_name,
                "class_name": cname, "schema_api": "v3_or_opaque",
                "category": str(getattr(cls, "CATEGORY", "")),
                "pass": False,
                "reasons": [f"INPUT_TYPES() not introspectable: "
                            f"{type(exc).__name__}: {exc}"],
            })
            continue
        rec = _classify_video_row(cname, inputs, [str(t) for t in cls.RETURN_TYPES])
        rec["row"] = f"{mod_name}.{cname}"
        rec["import_path"] = mod_name
        rec["schema_api"] = schema_api
        rec["category"] = str(getattr(cls, "CATEGORY", ""))
        rec["api_node"] = bool(getattr(cls, "API_NODE", False))
        rows.append(rec)
    passing = [r for r in rows if r.get("pass")]
    v3_only = [r for r in rows if r.get("contains_dynamic_v3")]
    verdict = _audit_verdict(rows)
    return {
        "audit_meta": {
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "comfyui_commit": _comfy_commit(),
            "generator": "scripts/otr_pin_partner_nodes.py --audit-i2v",
            "comfy_root": str(COMFY_ROOT),
        },
        "verdict": verdict,
        "counts": {"video_classes": len(rows), "passing": len(passing),
                   "v3_blocked": len(v3_only)},
        "passing_rows": [r["row"] for r in passing],
        "rows": sorted(rows, key=lambda r: r["row"]),
    }


def write_audit_report(report: dict) -> Path:
    AUDIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    out = AUDIT_OUT_DIR / f"audit-i2v-{stamp}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return out


def audit_i2v_main() -> int:
    report = audit_i2v()
    write_audit_report(report)
    print(f"\nVERDICT: {report['verdict']}  "
          f"(video_classes={report['counts']['video_classes']}, "
          f"passing={report['counts']['passing']}, "
          f"v3_blocked={report['counts']['v3_blocked']})")
    if report["passing_rows"]:
        print("PASSING rows: " + ", ".join(report["passing_rows"]))
    return 0


def main(argv) -> int:
    if "--check" in argv:
        return check_against_yaml()
    if "--audit-i2v" in argv:
        return audit_i2v_main()
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
