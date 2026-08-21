"""Matched Z-Image-Turbo ReferenceLatent OFF/ON diagnostic.

This is a persistent pixel A/B for the noisy-tile investigation.  It uses the
shipping engine's own parameter and graph builders; only the SaveImage terminal
and fail-closed structural assertions live here.  Run each arm after a fresh
server boot, using the same artifact directory and reference portrait.

Example (PowerShell; set the three model pins in the client shell too)::

    $env:OTR_ZIMAGE_UNET = "z_image_turbo_nvfp4.safetensors"
    $env:OTR_ZIMAGE_CLIP = "qwen_3_4b_fp8_mixed.safetensors"
    $env:OTR_ZIMAGE_VAE = "ae.safetensors"
    python scripts/otr_zimage_reference_ab.py --arm off `
      --reference-image C:\\path\\to\\c02.png `
      --artifact-dir C:\\path\\to\\ComfyUI\\output\\otr\\episodes\\zimage_ref_ab\\stills

Use ``--graph-only`` to write the graph and receipt without contacting ComfyUI.
The reference is deliberately required rather than defaulted to a machine-local
episode path.  UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from pathlib import PurePosixPath
import sys
import time
from datetime import datetime, timezone

import requests

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _path in (_HERE, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from otr_api import (  # noqa: E402
    COMFYUI_URL,
    fetch_schemas,
    poll_history,
    submit_prompt,
)
from nodes import _otr_paths as P  # noqa: E402
from nodes._otr_image_engines.z_image_turbo import (  # noqa: E402
    ZImageTurboEngine,
)


SCHEMA = "otr.zimage_reference_ab.v1"
FIXED_PROMPT = (
    "A paper origami diorama of a silver-haired light-skinned man with blue "
    "eyes and a red scar on his left cheek, wearing a plain pale shirt, "
    "standing in a dark box stage under a single overhead spotlight, medium "
    "shot, centered."
)
# Frozen from z_image_turbo._HYGIENE_NEGATIVE on 2026-08-20.  This A/B must not
# drift if that production fallback is later edited.
FIXED_NEGATIVE = (
    "oversaturated, glossy, plastic skin, waxy skin, sterile studio lighting, "
    "text, watermark"
)
FIXED_SEED = 7
FIXED_WIDTH = 1472
FIXED_HEIGHT = 832
UPLOAD_SUBFOLDER = "otr/zimage_reference_ab"

_PINNED_PARAMS = {
    "unet_name": "z_image_turbo_nvfp4.safetensors",
    "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
    "vae_name": "ae.safetensors",
    "clip_type": "qwen_image",
    "latent_node": "EmptySD3LatentImage",
    "unet_dtype": "default",
    "prompt": FIXED_PROMPT,
    "negative": FIXED_NEGATIVE,
    "seed": FIXED_SEED,
    "steps": 8,
    "cfg": 2.0,
    "shift": 3.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "width": FIXED_WIDTH,
    "height": FIXED_HEIGHT,
    "reference_height": 768,
}
_MATCH_KEYS = tuple(_PINNED_PARAMS)
_BASE_NODES = frozenset({
    "unet", "clip", "vae", "sampling", "pos", "neg", "latent",
    "ksampler", "decode",
})
_REF_NODES = frozenset({
    "load_ref", "scale_ref", "encode_ref", "ref_pos", "ref_neg",
})


class HarnessError(RuntimeError):
    """The A/B would no longer be matched or structurally trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_staged_name(source: str | Path, sha256: str | None = None) -> str:
    """Content-addressed basename requested from the server upload endpoint."""
    source_path = Path(source).expanduser().resolve()
    digest = sha256 or _sha256(source_path)
    suffix = source_path.suffix.lower() or ".png"
    return f"otr_zimage_reference_{digest[:24]}{suffix}"


def deterministic_load_image_input(
    source: str | Path, sha256: str | None = None
) -> str:
    return f"{UPLOAD_SUBFOLDER}/{deterministic_staged_name(source, sha256)}"


def _safe_load_image_input(subfolder: str, name: str) -> str:
    """Normalize a server upload response into one safe LoadImage token."""
    clean_name = str(name or "").replace("\\", "/")
    clean_subfolder = str(subfolder or "").replace("\\", "/").strip("/")
    if not clean_name or "/" in clean_name or clean_name in {".", ".."}:
        raise HarnessError(f"upload returned an unsafe image name: {name!r}")
    token = f"{clean_subfolder}/{clean_name}" if clean_subfolder else clean_name
    parsed = PurePosixPath(token)
    if (
        parsed.is_absolute()
        or ":" in token
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise HarnessError(f"upload returned an unsafe LoadImage input: {token!r}")
    return token


def upload_reference(
    source: str | Path,
    requested_name: str,
    *,
    url: str = COMFYUI_URL,
    subfolder: str = UPLOAD_SUBFOLDER,
) -> dict:
    """Upload through the active server and return its LoadImage input.

    This follows the standalone-runner precedent in
    ``scripts/otr_h3_mime_runner.py:410-438``.  wrapper_bridge staging is only
    correct in-process; from a standalone client it can resolve a different
    ComfyUI install's input directory than the server actually uses.
    """
    source_path = Path(source).expanduser().resolve()
    mime = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    with source_path.open("rb") as handle:
        response = requests.post(
            f"{url.rstrip('/')}/upload/image",
            files={"image": (requested_name, handle, mime)},
            data={
                "overwrite": "true",
                "type": "input",
                "subfolder": subfolder,
            },
            timeout=120,
        )
    response.raise_for_status()
    body = response.json()
    returned_name = body.get("name")
    if not returned_name:
        raise HarnessError(f"/upload/image response omitted name: {body!r}")
    returned_type = body.get("type") or "input"
    if returned_type != "input":
        raise HarnessError(
            f"/upload/image stored the reference as {returned_type!r}, not input"
        )
    returned_subfolder = body.get("subfolder") or ""
    return {
        "requested_name": requested_name,
        "requested_subfolder": subfolder,
        "returned_name": returned_name,
        "returned_subfolder": returned_subfolder,
        "returned_type": returned_type,
        "load_image_input": _safe_load_image_input(
            returned_subfolder, returned_name
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def resolve_artifact_dir(value: str | Path, output_root: str | Path) -> tuple[Path, str]:
    """Return an absolute dir and SaveImage prefix root, or fail containment."""
    artifact = Path(value).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    try:
        relative = artifact.relative_to(root)
    except ValueError as exc:
        raise HarnessError(
            "--artifact-dir must be inside ComfyUI's output directory "
            f"({root}); got {artifact}"
        ) from exc
    if not relative.parts:
        raise HarnessError("--artifact-dir may not be the ComfyUI output root")
    return artifact, relative.as_posix()


def _resolve_class(candidates: tuple[str, ...], available: set[str] | None) -> str:
    if available is None:
        return candidates[0]
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise HarnessError(
        "live server has none of the required node candidates: "
        + ", ".join(candidates)
    )


def _assert_pinned_params(params: dict, arm: str, load_image_input: str) -> None:
    mismatches = []
    for key, expected in _PINNED_PARAMS.items():
        if params.get(key) != expected:
            mismatches.append(f"{key}={params.get(key)!r} (expected {expected!r})")
    expected_ref = load_image_input if arm == "on" else ""
    if params.get("reference_image") != expected_ref:
        mismatches.append(
            f"reference_image={params.get('reference_image')!r} "
            f"(expected {expected_ref!r})"
        )
    if mismatches:
        raise HarnessError(
            "recipe pins drifted; set the Z-Image environment to the accepted "
            "NVFP4/Qwen-FP8/AE recipe: " + "; ".join(mismatches)
        )


def _graph_positive_evidence(raw: dict, api: dict) -> dict[str, bool]:
    """Derive structural evidence from the built graph, never from arm label."""
    sampler = raw.get("ksampler", {}).get("inputs", {})
    api_types = {
        key: node.get("class_type")
        for key, node in api.items()
        if isinstance(node, dict)
    }
    off_proven = (
        set(raw) == _BASE_NODES
        and sampler.get("positive") == ["pos", 0]
        and sampler.get("negative") == ["neg", 0]
        and "ReferenceLatent" not in api_types.values()
    )
    load_ref_input = raw.get("load_ref", {}).get("inputs", {}).get("image")
    exact_ref_inputs = {
        "scale_ref": {
            "image": ["load_ref", 0],
            "upscale_method": "lanczos",
            "width": 0,
            "height": 768,
            "crop": "disabled",
        },
        "encode_ref": {"pixels": ["scale_ref", 0], "vae": ["vae", 0]},
        "ref_pos": {"conditioning": ["pos", 0], "latent": ["encode_ref", 0]},
        "ref_neg": {"conditioning": ["neg", 0], "latent": ["encode_ref", 0]},
    }
    expected_types = {
        "load_ref": "LoadImage",
        "scale_ref": "ImageScale",
        "encode_ref": "VAEEncode",
        "ref_pos": "ReferenceLatent",
        "ref_neg": "ReferenceLatent",
    }
    on_proven = (
        set(raw) == _BASE_NODES | _REF_NODES
        and isinstance(load_ref_input, str)
        and bool(load_ref_input)
        and all(
            raw.get(key, {}).get("inputs") == expected
            for key, expected in exact_ref_inputs.items()
        )
        and all(api_types.get(key) == expected for key, expected in expected_types.items())
        and sampler.get("positive") == ["ref_pos", 0]
        and sampler.get("negative") == ["ref_neg", 0]
    )
    return {
        "graph_off_has_no_reference_latent": bool(off_proven),
        "graph_on_has_exact_dual_reference_chain": bool(on_proven),
    }


def _assert_arm_structure(
    arm: str,
    raw: dict,
    api: dict,
    load_image_input: str,
    save_prefix: str,
) -> None:
    expected_nodes = _BASE_NODES | (_REF_NODES if arm == "on" else frozenset())
    if set(raw) != expected_nodes:
        raise HarnessError(
            f"{arm} logical node set drifted: got {sorted(raw)}, "
            f"expected {sorted(expected_nodes)}"
        )
    for key in expected_nodes:
        if raw[key].get("class") != key:
            raise HarnessError(f"{key} no longer uses its engine logical class")

    sampler = raw["ksampler"]["inputs"]
    if arm == "off":
        if sampler.get("positive") != ["pos", 0] or sampler.get("negative") != ["neg", 0]:
            raise HarnessError("OFF sampler is not wired directly to pos/neg")
        if any(node.get("class_type") == "ReferenceLatent" for node in api.values()):
            raise HarnessError("OFF graph contains ReferenceLatent")
    else:
        exact_inputs = {
            "load_ref": {"image": load_image_input},
            "scale_ref": {
                "image": ["load_ref", 0],
                "upscale_method": "lanczos",
                "width": 0,
                "height": 768,
                "crop": "disabled",
            },
            "encode_ref": {"pixels": ["scale_ref", 0], "vae": ["vae", 0]},
            "ref_pos": {"conditioning": ["pos", 0], "latent": ["encode_ref", 0]},
            "ref_neg": {"conditioning": ["neg", 0], "latent": ["encode_ref", 0]},
        }
        for key, expected in exact_inputs.items():
            if raw[key].get("inputs") != expected:
                raise HarnessError(
                    f"ON reference chain drifted at {key}: "
                    f"{raw[key].get('inputs')!r}"
                )
        if sampler.get("positive") != ["ref_pos", 0] or sampler.get("negative") != ["ref_neg", 0]:
            raise HarnessError("ON sampler does not consume both ReferenceLatent nodes")
        expected_types = {
            "load_ref": "LoadImage",
            "scale_ref": "ImageScale",
            "encode_ref": "VAEEncode",
            "ref_pos": "ReferenceLatent",
            "ref_neg": "ReferenceLatent",
        }
        for key, expected in expected_types.items():
            if api[key].get("class_type") != expected:
                raise HarnessError(
                    f"ON {key} resolved to {api[key].get('class_type')!r}, "
                    f"not {expected!r}"
                )

    if api.get("save") != {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": save_prefix, "images": ["decode", 0]},
    }:
        raise HarnessError("SaveImage terminal no longer targets decode/artifact prefix")
    evidence = _graph_positive_evidence(raw, api)
    required = (
        "graph_off_has_no_reference_latent"
        if arm == "off"
        else "graph_on_has_exact_dual_reference_chain"
    )
    if not evidence[required]:
        raise HarnessError(f"{arm} graph did not derive its required evidence")


def build_arm_graph(
    arm: str,
    source_reference: str | Path,
    save_prefix: str,
    *,
    staged_reference_name: str | None = None,
    available: set[str] | None = None,
    engine: ZImageTurboEngine | None = None,
) -> tuple[dict, dict, dict]:
    """Build and gate one engine-owned arm; pure apart from env/model discovery."""
    if arm not in {"off", "on"}:
        raise HarnessError(f"unknown arm {arm!r}")
    source = Path(source_reference).expanduser().resolve()
    load_image_input = ""
    if arm == "on":
        load_image_input = staged_reference_name or deterministic_load_image_input(source)
        # A server-returned subfolder/name is allowed; absolute/traversal paths
        # are not. Split once so the same validator handles either shape.
        normalized = str(load_image_input).replace("\\", "/")
        subfolder, separator, name = normalized.rpartition("/")
        load_image_input = _safe_load_image_input(
            subfolder if separator else "", name if separator else normalized
        )
    request = {
        "prompt": FIXED_PROMPT,
        "negative_prompt": FIXED_NEGATIVE,
        "seed": FIXED_SEED,
        "width": FIXED_WIDTH,
        "height": FIXED_HEIGHT,
        "reference_image": load_image_input,
    }
    engine = engine or ZImageTurboEngine()
    params = engine._diagnostic_zimage_params(request)
    # Keep the current hygiene text fixed even if an unrelated developer env
    # override is present.  All other recipe overrides are rejected below.
    params["negative"] = FIXED_NEGATIVE
    _assert_pinned_params(params, arm, load_image_input)

    candidates = dict(engine._node_candidates(params))
    if arm == "on":
        candidates.update(engine._REF_CANDIDATES)
    raw = engine._build_zimage_graph(params, lambda name, slot: [name, slot])
    api = {
        key: {
            "class_type": _resolve_class(candidates[node["class"]], available),
            "inputs": node["inputs"],
        }
        for key, node in raw.items()
    }
    if available is not None and "SaveImage" not in available:
        raise HarnessError("live server is missing required SaveImage")
    api["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": save_prefix, "images": [engine._TERMINAL, 0]},
    }
    _assert_arm_structure(arm, raw, api, load_image_input, save_prefix)
    return api, raw, params


def _match_signature(params: dict) -> dict:
    return {key: params[key] for key in _MATCH_KEYS}


def _assert_matches_sibling(artifact_dir: Path, arm: str, receipt: dict) -> None:
    other = "on" if arm == "off" else "off"
    sibling_path = artifact_dir / other / "receipt.json"
    if not sibling_path.is_file():
        return
    sibling = json.loads(sibling_path.read_text(encoding="utf-8"))
    if sibling.get("schema") != SCHEMA:
        raise HarnessError(f"sibling receipt has wrong schema: {sibling_path}")
    for key in ("match_signature", "reference"):
        if sibling.get(key) != receipt.get(key):
            raise HarnessError(f"{arm}/{other} mismatch in {key}; refusing A/B")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("off", "on"))
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="write gated graph/receipt without contacting or submitting to ComfyUI",
    )
    args = parser.parse_args(argv)

    try:
        reference = Path(args.reference_image).expanduser().resolve()
        if not reference.is_file():
            raise HarnessError(f"reference image is not a file: {reference}")
        artifact_dir, relative_artifact = resolve_artifact_dir(
            args.artifact_dir, P.comfy_output_dir()
        )
        arm_dir = artifact_dir / args.arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        stem = f"zimage_reference_{args.arm}_seed{FIXED_SEED}"
        save_prefix = f"{relative_artifact}/{args.arm}/{stem}"
        reference_sha256 = _sha256(reference)
        requested_name = deterministic_staged_name(reference, reference_sha256)
        would_be_input = deterministic_load_image_input(reference, reference_sha256)

        available = None
        if not args.graph_only:
            available = set(fetch_schemas())
        upload = None
        if args.arm == "on" and not args.graph_only:
            # Standalone clients must upload through the active server. See the
            # H3 precedent cited in upload_reference; local path guessing failed
            # live when the client and server belonged to different installs.
            upload = upload_reference(reference, requested_name)
        load_image_input = (
            upload["load_image_input"] if upload is not None
            else would_be_input if args.arm == "on"
            else ""
        )
        api, raw, params = build_arm_graph(
            args.arm,
            reference,
            save_prefix,
            staged_reference_name=load_image_input if args.arm == "on" else None,
            available=available,
        )
        graph_payload = {
            "schema": SCHEMA,
            "arm": args.arm,
            "logical_graph": raw,
            "api_graph": api,
        }
        graph_path = arm_dir / "graph.json"
        _write_json(graph_path, graph_payload)
        receipt = {
            "schema": SCHEMA,
            "arm": args.arm,
            "status": "PREPARED",
            "created_at_utc": _utc_now(),
            "comfyui_url": None if args.graph_only else COMFYUI_URL,
            "artifact_dir": str(artifact_dir),
            "graph_path": str(graph_path),
            "graph_sha256": _sha256(graph_path),
            "reference": {
                "path": str(reference),
                "bytes": reference.stat().st_size,
                "sha256": reference_sha256,
            },
            "staging": {
                "mode": (
                    "would_upload" if args.arm == "on" and args.graph_only
                    else "uploaded" if args.arm == "on"
                    else "not_applicable"
                ),
                "requested_name": requested_name if args.arm == "on" else "",
                "requested_subfolder": UPLOAD_SUBFOLDER if args.arm == "on" else "",
                "returned_name": upload["returned_name"] if upload else "",
                "returned_subfolder": upload["returned_subfolder"] if upload else "",
                "returned_type": upload["returned_type"] if upload else "",
                "load_image_input": load_image_input,
            },
            "match_signature": _match_signature(params),
            "positive_evidence": {
                **_graph_positive_evidence(raw, api),
                "history_success_with_output": False,
            },
            "prompt_id": None,
            "wall_s": None,
            "outputs": [],
            "output_artifacts": [],
            "error": "",
        }
        _assert_matches_sibling(artifact_dir, args.arm, receipt)
        receipt_path = arm_dir / "receipt.json"
        if args.graph_only:
            receipt["status"] = "GRAPH_ONLY"
            receipt["completed_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)
            print(f"[zimage-ref-ab] GRAPH_ONLY {args.arm}: {graph_path}", flush=True)
            return 0

        _write_json(receipt_path, receipt)
        started = time.time()
        try:
            receipt["prompt_id"] = submit_prompt(api)
            status, error = poll_history(receipt["prompt_id"], timeout_s=args.timeout)
            receipt["wall_s"] = round(time.time() - started, 3)
            receipt["error"] = error
            outputs = sorted(
                str(path.resolve())
                for path in arm_dir.glob(stem + "*.png")
                if path.stat().st_mtime >= started - 2
            )
            receipt["outputs"] = outputs
            receipt["output_artifacts"] = [
                {
                    "path": output,
                    "bytes": Path(output).stat().st_size,
                    "sha256": _sha256(Path(output)),
                }
                for output in outputs
            ]
            receipt["status"] = "SUCCESS" if status == "SUCCESS" and outputs else "FAIL"
            receipt["positive_evidence"]["history_success_with_output"] = (
                receipt["status"] == "SUCCESS"
            )
            if status == "SUCCESS" and not outputs:
                receipt["error"] = "history succeeded but no PNG landed in artifact dir"
        except Exception as exc:  # noqa: BLE001 -- persist the live failure
            receipt["wall_s"] = round(time.time() - started, 3)
            receipt["status"] = "FAIL"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["completed_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        print(
            f"[zimage-ref-ab] {receipt['status']} {args.arm}: {receipt_path}",
            flush=True,
        )
        return 0 if receipt["status"] == "SUCCESS" else 1
    except Exception as exc:  # noqa: BLE001 -- CLI emits one clear fail-closed line
        print(f"[zimage-ref-ab] FAIL CLOSED: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
