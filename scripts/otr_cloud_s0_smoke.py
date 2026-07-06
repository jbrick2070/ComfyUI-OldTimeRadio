"""S0 live smokes for the invoke_partner_node bridge (pass04 sec 3).

Leg 1 -- headless auth proof on a pinned image row
         (cloud_flux_pro): one small still, PartnerResult validated.
Leg 2 -- the whole point: a Kling avatar clip actually CONDITIONED by
         episode audio (cloud_kling_avatar with the checked-in
         tests/fixtures/baseline_v1.5.wav + a radio still).

Gated: refuses to run unless OTR_RUN_CLOUD_SMOKE=1 (a PAID smoke needs an
explicit run gate; this is NOT the removed enable flag -- production runs
gate on the dropdown pick alone, operator directive 2026-07-02). Also
requires the operator prereqs (exit 3 names whichever is missing):
    OTR_CLOUD_MEDIA_BUDGET_USD=<ceiling, e.g. 2.00>  (optional; unset =
                               the $10 DEFAULT_BUDGET_USD safety cap)
    OTR_COMFY_API_KEY=<key>   (or a logged-in Comfy account when the
                               call runs inside the server; headless
                               needs the key)

Usage (venv python, repo root):
    python scripts/otr_cloud_s0_smoke.py --leg 1
    python scripts/otr_cloud_s0_smoke.py --leg 2

Env knobs:
    OTR_COMFY_CORE          ComfyUI repo root (default
                            C:/Users/jeffr/ComfyUI-Installs/ComfyUI/ComfyUI)
    OTR_CLOUD_SMOKE_EST_USD per-leg budget estimate (default 0.15)
    OTR_CLOUD_SMOKE_TIMEOUT_S  provider timeout (default 600)
    OTR_CLOUD_SMOKE_SIZE    leg-1 flux_pro square size in px (default 1024)
    OTR_CLOUD_SMOKE_IMAGE   leg-2 still path (default: placeholder
                            radio-cabinet gradient generated on the fly)
    OTR_CLOUD_SMOKE_MODE    leg-2 kling mode combo (default: first
                            option discovered from the live schema)

Exit codes: 0 = PASS, 2 = FAIL, 3 = gated / prereq missing.
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = Path("C:/Users/jeffr/ComfyUI-Installs/ComfyUI/ComfyUI")
FIXTURE_WAV = REPO_ROOT / "tests" / "fixtures" / "baseline_v1.5.wav"


def _gate() -> None:
    missing = []
    if os.environ.get("OTR_RUN_CLOUD_SMOKE", "").strip() != "1":
        missing.append("OTR_RUN_CLOUD_SMOKE=1")
    if not os.environ.get("OTR_COMFY_API_KEY", "").strip():
        missing.append("OTR_COMFY_API_KEY (headless smoke needs the key)")
    if missing:
        print("SMOKE GATED -- set: " + ", ".join(missing))
        sys.exit(3)


def _bootstrap_paths() -> None:
    core = Path(os.environ.get("OTR_COMFY_CORE", "").strip() or DEFAULT_CORE)
    if not (core / "comfy_api_nodes").is_dir():
        print(f"SMOKE GATED -- ComfyUI core not found at {core} "
              f"(set OTR_COMFY_CORE)")
        sys.exit(3)
    for p in (str(core), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_image_tensor(path_str: str):
    import numpy as np
    import torch
    if path_str:
        from PIL import Image
        img = Image.open(path_str).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
    else:  # placeholder radio-cabinet gradient, 832x480 role canvas
        h, w = 480, 832
        y = np.linspace(0.25, 0.05, h, dtype="float32")[:, None]
        arr = np.stack([y * 1.6, y * 1.1, y * 0.7], axis=-1)
        arr = np.repeat(arr, w, axis=1).reshape(h, w, 3)
    return torch.from_numpy(arr)[None, ...]  # [1,H,W,C]


def _load_audio_dict(path: Path):
    import soundfile as sf
    import torch
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.T)[None, ...]  # [1,C,T]
    return {"waveform": wave, "sample_rate": int(sr)}


def _first_combo_option(node_cls, input_name: str):
    """Best-effort combo discovery across V1 INPUT_TYPES and V3
    schemas; fail LOUD if nothing is found."""
    try:
        spec = node_cls.INPUT_TYPES()
        for section in ("required", "optional"):
            entry = (spec.get(section) or {}).get(input_name)
            if entry and isinstance(entry[0], (list, tuple)) and entry[0]:
                return entry[0][0]
    except Exception:
        pass
    try:
        schema = node_cls.define_schema()
        for inp in getattr(schema, "inputs", []) or []:
            if getattr(inp, "id", "") == input_name:
                opts = getattr(inp, "options", None)
                if opts:
                    first = opts[0]
                    return getattr(first, "value", first)
    except Exception:
        pass
    raise RuntimeError(
        f"cannot discover combo options for {input_name!r} on "
        f"{node_cls.__name__}; set the env override")


def _run_leg(node_key: str, inputs: dict, est_usd: float,
             timeout_s: float) -> int:
    from nodes._otr_shared import cloud_media_backend as backend
    from nodes._otr_shared.cloud_media_invoke import (
        bind_prompt_id, invoke_partner_node)

    prompt_id = f"s0-smoke-{uuid.uuid4().hex[:8]}"
    try:
        with bind_prompt_id(prompt_id):
            result = invoke_partner_node(
                node_key, inputs, timeout_s=timeout_s,
                estimated_usd=est_usd)
        out = Path(result["path"])
        size = out.stat().st_size
        sess = backend.peek_session(prompt_id)
        spent = sess.spent_usd() if sess else float("nan")
        print(f"PASS {node_key}: {out} ({size} bytes, "
              f"{result['content_type']}) spent~=${spent:.4f} "
              f"job_id={result['provider_job_id']}")
        return 0
    except Exception as exc:
        print(f"FAIL {node_key}: {exc}")
        return 2
    finally:
        backend.teardown_session(prompt_id)


def leg1(est_usd: float, timeout_s: float) -> int:
    # /32-aligned square (BFL requires request dims divisible by 32).
    try:
        px = int(os.environ.get("OTR_CLOUD_SMOKE_SIZE", "1024").split("x")[0])
    except (TypeError, ValueError):
        px = 1024
    px = max(32, (px // 32) * 32)
    return _run_leg("cloud_flux_pro", {
        "prompt": "a 1946 bakelite tabletop radio on a wooden desk, warm "
                  "dramatic lighting, product still",
        "width": px,
        "height": px,
        "prompt_upsampling": False,
        "seed": 7,
    }, est_usd, timeout_s)


def leg2(est_usd: float, timeout_s: float) -> int:
    if not FIXTURE_WAV.is_file():
        print(f"FAIL: checked-in fixture missing: {FIXTURE_WAV}")
        return 2
    from nodes._otr_shared.cloud_media_invoke import (
        _resolve_node_class, partner_rows)
    row = partner_rows()["cloud_kling_avatar"]
    mode = os.environ.get("OTR_CLOUD_SMOKE_MODE", "").strip()
    if not mode:
        mode = _first_combo_option(_resolve_node_class(row), "mode")
        print(f"leg2: discovered kling mode={mode!r} "
              f"(override via OTR_CLOUD_SMOKE_MODE)")
    inputs = {
        "image": _load_image_tensor(
            os.environ.get("OTR_CLOUD_SMOKE_IMAGE", "").strip()),
        "sound_file": _load_audio_dict(FIXTURE_WAV),
        "mode": mode,
        "seed": 7,
        "prompt": "a vintage radio talking, lips in sync with the speech",
    }
    return _run_leg("cloud_kling_avatar", inputs, est_usd, timeout_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    _gate()
    _bootstrap_paths()
    est = float(os.environ.get("OTR_CLOUD_SMOKE_EST_USD", "0.15"))
    timeout_s = float(os.environ.get("OTR_CLOUD_SMOKE_TIMEOUT_S", "600"))
    return leg1(est, timeout_s) if args.leg == 1 else leg2(est, timeout_s)


if __name__ == "__main__":
    sys.exit(main())
