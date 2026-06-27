"""Run a fresh 30-word headless OTR smoke through the real workflow.

Loads workflows/otr_scifi_16gb_full.json from disk on every invocation, converts
it with tests/_run_baseline.py's proven litegraph->API helpers, patches a tiny
writer brief, neutralizes disabled slot-picker sentinels when the live schema
does not accept them, submits to the live ComfyUI server, and reports PASS/FAIL.

This script is intentionally a live smoke harness, not a generated workflow and
not an isolated LTX graph. The server must be freshly booted with:

  OTR_LTX_AV_UNET=distilled-1.1\\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf

Leave OTR_LTX_AV_RECIPE unset so eng_ltx_av auto-resolves distilled_native.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import _run_baseline as baseline  # noqa: E402
import requests  # noqa: E402

from nodes._otr_video_engines.eng_ltx_av import (  # noqa: E402
    LtxAudioInEngine,
    RECIPE_DISTILLED_NATIVE,
)
from nodes._otr_video_engines import wrapper_bridge as wb  # noqa: E402


COMFYUI_BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = os.environ.get(
    "COMFYUI_OUTPUT",
    r"C:\Users\jeffr\Documents\ComfyUI\output",
)
EPISODES_DIR = os.path.join(OUTPUT_DIR, "otr", "episodes")
RUNTIME_LOG = os.path.join(REPO_ROOT, "otr_runtime.log")
DISTILLED_UNET = r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"
LOCAL_WRITER = "mistralai/Mistral-Nemo-Instruct-2407"
LTX_AUDIO_IN_PICK = "ltx_audio_in (16:9)"
Z_IMAGE_TURBO_PICK = "z_image_turbo"
WRITER_NODE_ID = "1"
DIRECTOR_NODE_ID = "87"
RENDER_BATCH_NODE_ID = "92"
VIDEO_DIRECTOR_SLOTS = (
    "announcer_video_model",
    "music_video_model",
    "other_beats_video_model",
)
IMAGE_DIRECTOR_SLOTS = (
    "announcer_image_model",
    "music_image_model",
    "other_beats_image_model",
)


def _schema_inputs(schemas: dict[str, Any], node_type: str) -> dict[str, Any]:
    node_schema = schemas.get(node_type)
    if not node_schema:
        raise RuntimeError("live /object_info is missing %s" % node_type)
    section = node_schema.get("input", {}) or {}
    merged: dict[str, Any] = {}
    for key in ("required", "optional"):
        merged.update(section.get(key, {}) or {})
    return merged


def _choices(spec: Any) -> list[Any] | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    first = spec[0]
    if isinstance(first, list):
        return list(first)
    return None


def _confirm_input(schemas: dict[str, Any], node_type: str, input_name: str) -> None:
    inputs = _schema_inputs(schemas, node_type)
    if input_name not in inputs:
        raise RuntimeError(
            "%s does not declare input %r in live INPUT_TYPES; known=%s"
            % (node_type, input_name, sorted(inputs))
        )


def _writer_slot_fallback(choices: list[Any], current: Any, sentinel: str) -> Any:
    for candidate in (current, LOCAL_WRITER, sentinel):
        if candidate in choices:
            return candidate
    for candidate in choices:
        if isinstance(candidate, str) and candidate and not candidate.startswith("("):
            return candidate
    if choices:
        return choices[0]
    raise RuntimeError("slot picker has no catalog choices")


def _patch_writer_inputs(prompt: dict[str, Any], schemas: dict[str, Any],
                         words: int) -> list[str]:
    writer = prompt.get(WRITER_NODE_ID)
    if not writer or writer.get("class_type") != "OTR_LedgerScriptWriter":
        raise RuntimeError("API prompt node #1 is not OTR_LedgerScriptWriter")

    writer_inputs = writer.setdefault("inputs", {})
    schema_inputs = _schema_inputs(schemas, "OTR_LedgerScriptWriter")

    _confirm_input(schemas, "OTR_LedgerScriptWriter", "target_words")
    before_words = writer_inputs.get("target_words")
    writer_inputs["target_words"] = int(words)

    # Keep the known local writer model; do not let a remote slot sentinel block
    # queue validation while the remote lane is disabled.
    for name in ("creative_writing_model", "technical_model"):
        _confirm_input(schemas, "OTR_LedgerScriptWriter", name)
        writer_inputs[name] = LOCAL_WRITER

    changes = ["writer target_words %r -> %d" % (before_words, words)]
    for name, sentinel in (
        ("openrouter_slot_a_model", "(enable OpenRouter)"),
        ("openrouter_slot_b_model", "(enable OpenRouter)"),
        ("comfy_slot_a_model", "(enable Comfy Credits)"),
        ("comfy_slot_b_model", "(enable Comfy Credits)"),
    ):
        if name not in schema_inputs:
            continue
        opts = _choices(schema_inputs[name])
        if opts is None:
            continue
        current = writer_inputs.get(name)
        if current in opts:
            changes.append("%s kept %r" % (name, current))
            continue
        replacement = _writer_slot_fallback(opts, current, sentinel)
        writer_inputs[name] = replacement
        changes.append("%s %r -> %r" % (name, current, replacement))

    return changes


def _assert_combo_accepts(schemas: dict[str, Any], node_type: str,
                          input_name: str, target: Any) -> None:
    inputs = _schema_inputs(schemas, node_type)
    _confirm_input(schemas, node_type, input_name)
    opts = _choices(inputs[input_name])
    if opts is not None and target not in opts:
        raise RuntimeError(
            "%s.%s live catalog does not include %r; choices=%s"
            % (node_type, input_name, target, opts)
        )


def _patch_director_inputs(prompt: dict[str, Any],
                           schemas: dict[str, Any]) -> list[str]:
    director = prompt.get(DIRECTOR_NODE_ID)
    if not director or director.get("class_type") != "OTR_VideoDirector":
        raise RuntimeError("API prompt node #87 is not OTR_VideoDirector")
    dinputs = director.setdefault("inputs", {})
    changes: list[str] = []

    for name in VIDEO_DIRECTOR_SLOTS:
        _assert_combo_accepts(
            schemas, "OTR_VideoDirector", name, LTX_AUDIO_IN_PICK)
        before = dinputs.get(name)
        dinputs[name] = LTX_AUDIO_IN_PICK
        changes.append("%s %r -> %r" % (name, before, LTX_AUDIO_IN_PICK))

    for name in IMAGE_DIRECTOR_SLOTS:
        _assert_combo_accepts(
            schemas, "OTR_VideoDirector", name, Z_IMAGE_TURBO_PICK)
        before = dinputs.get(name)
        dinputs[name] = Z_IMAGE_TURBO_PICK
        changes.append("%s %r -> %r" % (name, before, Z_IMAGE_TURBO_PICK))

    _confirm_input(schemas, "OTR_VideoDirector", "allow_auto_fallback")
    before_fallback = dinputs.get("allow_auto_fallback")
    dinputs["allow_auto_fallback"] = False
    changes.append(
        "allow_auto_fallback %r -> False" % (before_fallback,))
    return changes


def _preflight_distilled_native_graph() -> list[str]:
    if os.environ.get("OTR_LTX_AV_UNET") != DISTILLED_UNET:
        raise RuntimeError(
            "OTR_LTX_AV_UNET must be %r for this smoke; got %r"
            % (DISTILLED_UNET, os.environ.get("OTR_LTX_AV_UNET"))
        )
    if os.environ.get("OTR_ENABLE_LTX_AV") != "1":
        raise RuntimeError("OTR_ENABLE_LTX_AV=1 is required for ltx_audio_in")
    if os.environ.get("OTR_LTX_AV_RECIPE"):
        raise RuntimeError("leave OTR_LTX_AV_RECIPE unset for auto-detect")
    if os.environ.get("OTR_LTX_AV_SHARP") is not None:
        raise RuntimeError("OTR_LTX_AV_SHARP is retired and must be unset")
    if os.environ.get("OTR_ENABLE_ZIMAGE") != "1":
        raise RuntimeError("OTR_ENABLE_ZIMAGE=1 is required for z_image_turbo")
    if not os.environ.get("OTR_ZIMAGE_UNET"):
        raise RuntimeError(
            "OTR_ZIMAGE_UNET must point at the Z-Image Turbo diffusion model")

    eng = LtxAudioInEngine()
    recipe = eng._recipe()
    if recipe != RECIPE_DISTILLED_NATIVE:
        raise RuntimeError("expected distilled_native, got %r" % recipe)
    graph = eng._build_graph(
        {"text_prompt": "30-word smoke", "seed": 0},
        105,
        832,
        480,
        "audio.wav",
        "still.png",
    )
    forbidden = [nid for nid in ("lora", "modelsampling", "sched")
                 if nid in graph]
    if forbidden:
        raise RuntimeError(
            "distilled_native graph contains forbidden node(s): %s"
            % ", ".join(forbidden)
        )
    if "sigmas" not in graph:
        raise RuntimeError("distilled_native graph is missing fixed sigmas")
    model_wire = graph["guider"]["inputs"].get("model")
    if model_wire != wb.Wire("unet", 0):
        raise RuntimeError(
            "distilled_native guider model must wire directly from unet; got %r"
            % (model_wire,)
        )
    return [
        "recipe auto-detected distilled_native",
        "graph has no LoraLoaderModelOnly, no ModelSamplingLTXV, no LTXVScheduler",
        "graph uses fixed sigmas and wires unet directly into CFGGuider",
    ]


def _episode_mp4_set() -> set[str]:
    return {
        p for p in glob.glob(os.path.join(EPISODES_DIR, "**", "*.mp4"),
                             recursive=True)
        if os.path.isfile(p)
    }


def _new_episode_mp4s(before: set[str], started: float) -> list[str]:
    after = _episode_mp4_set()
    cands = [p for p in after - before if os.path.getmtime(p) >= started - 2]
    return sorted(cands, key=os.path.getmtime)


def _history_outputs(history_entry: dict[str, Any]) -> dict[str, list[str]]:
    outputs = history_entry.get("outputs", {}) or {}
    summary: dict[str, list[str]] = {}
    for nid, node_out in outputs.items():
        keys = []
        if isinstance(node_out, dict):
            for key, val in node_out.items():
                if isinstance(val, list):
                    keys.append("%s[%d]" % (key, len(val)))
                else:
                    keys.append(key)
        summary[str(nid)] = keys
    return summary


def _node_text_json(history_entry: dict[str, Any], node_id: str) -> dict[str, Any]:
    node_out = (history_entry.get("outputs", {}) or {}).get(node_id) or {}
    texts = node_out.get("text") if isinstance(node_out, dict) else None
    if not texts:
        raise RuntimeError("history output #%s has no text payload" % node_id)
    try:
        payload = json.loads(str(texts[0]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "history output #%s text is not JSON: %s" % (node_id, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "history output #%s JSON is not an object" % node_id
        )
    return payload


def _assert_render_batch_report(history_entry: dict[str, Any]) -> dict[str, Any]:
    report = _node_text_json(history_entry, RENDER_BATCH_NODE_ID)
    if not report.get("ok"):
        raise RuntimeError("OTR_VideoRenderBatch report is not ok: %r" % report)
    hist = report.get("engine_histogram") or {}
    ltx_count = int(hist.get("ltx_audio_in") or 0)
    clip_count = int(report.get("clip_count") or 0)
    if ltx_count <= 0:
        raise RuntimeError(
            "OTR_VideoRenderBatch completed without ltx_audio_in clips: %r"
            % hist
        )
    if clip_count <= 0:
        raise RuntimeError("OTR_VideoRenderBatch clip_count is zero")
    non_ltx = {k: v for k, v in hist.items() if k != "ltx_audio_in"}
    if non_ltx or ltx_count != clip_count:
        raise RuntimeError(
            "expected every rendered clip to use ltx_audio_in; "
            "clip_count=%s histogram=%r" % (clip_count, hist)
        )
    return report


def _tail_file(path: str, max_lines: int = 80) -> list[str]:
    if not path or not os.path.exists(path):
        return ["[missing] %s" % path]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


class LogTailer(threading.Thread):
    def __init__(self, paths: list[str], stop_event: threading.Event,
                 interval_s: float = 30.0):
        super().__init__(daemon=True)
        self.paths = [p for p in paths if p]
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.positions: dict[str, int] = {}

    def run(self) -> None:
        for path in self.paths:
            try:
                self.positions[path] = os.path.getsize(path)
            except OSError:
                self.positions[path] = 0
        while not self.stop_event.wait(self.interval_s):
            for path in self.paths:
                self._print_new(path)

    def _print_new(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.positions.get(path, 0))
                data = f.read()
                self.positions[path] = f.tell()
        except OSError:
            return
        lines = [line for line in data.splitlines() if line.strip()]
        if not lines:
            return
        print("[tail:%s]" % os.path.basename(path), flush=True)
        for line in lines[-20:]:
            print("  " + line[-240:], flush=True)


def _build_prompt(words: int) -> tuple[dict[str, Any], list[str]]:
    print("[smoke] loading canonical workflow fresh from disk", flush=True)
    workflow = baseline._load_workflow()
    print("[smoke] fetching live /object_info schemas", flush=True)
    schemas = baseline._fetch_schemas(COMFYUI_BASE)
    print("[smoke] converting litegraph workflow to API prompt", flush=True)
    prompt = baseline._workflow_to_api_prompt(workflow, schemas)
    prompt = baseline._fix_known_widget_drift(prompt)

    changes = _patch_writer_inputs(prompt, schemas, words)
    changes.extend(_patch_director_inputs(prompt, schemas))
    if DIRECTOR_NODE_ID in prompt:
        dinputs = prompt[DIRECTOR_NODE_ID].get("inputs", {})
        changes.append(
            "director video announcer=%r music=%r other=%r"
            % (
                dinputs.get("announcer_video_model"),
                dinputs.get("music_video_model"),
                dinputs.get("other_beats_video_model"),
            )
        )
        changes.append(
            "director images announcer=%r music=%r other=%r"
            % (
                dinputs.get("announcer_image_model"),
                dinputs.get("music_image_model"),
                dinputs.get("other_beats_image_model"),
            )
        )
    if RENDER_BATCH_NODE_ID in prompt:
        changes.append(
            "render batch mode=%r"
            % prompt[RENDER_BATCH_NODE_ID].get("inputs", {}).get("mode")
        )
    return prompt, changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--server-log", default=os.environ.get("OTR_SMOKE_SERVER_LOG", ""))
    args = ap.parse_args(argv)

    started = time.time()
    print("[smoke] COMFYUI=%s" % COMFYUI_BASE, flush=True)
    print("[smoke] OUTPUT=%s" % OUTPUT_DIR, flush=True)

    checks = _preflight_distilled_native_graph()
    for line in checks:
        print("[smoke] preflight: " + line, flush=True)

    before = _episode_mp4_set()
    prompt, changes = _build_prompt(args.words)
    for line in changes:
        print("[smoke] patch: " + line, flush=True)

    stop_tail = threading.Event()
    tailer = LogTailer([args.server_log, RUNTIME_LOG], stop_tail)
    tailer.start()
    try:
        print("[smoke] submitting prompt", flush=True)
        prompt_id = baseline._submit_prompt(prompt, COMFYUI_BASE)
        print("[smoke] QUEUED prompt_id=%s" % prompt_id, flush=True)
        history = baseline._poll_for_completion(
            prompt_id, COMFYUI_BASE, timeout_s=args.timeout)
    finally:
        stop_tail.set()
        tailer.join(timeout=5)

    new_mp4s = _new_episode_mp4s(before, started)
    render_report = _assert_render_batch_report(history)
    result = {
        "prompt_id": prompt_id,
        "new_episode_mp4s": new_mp4s,
        "history_outputs": _history_outputs(history),
    }
    print("[smoke] history outputs: %s"
          % json.dumps(result["history_outputs"], ensure_ascii=True)[:1200],
          flush=True)
    print("[smoke] render batch: ok=%s clip_count=%s histogram=%s"
          % (
              render_report.get("ok"),
              render_report.get("clip_count"),
              json.dumps(render_report.get("engine_histogram") or {},
                         ensure_ascii=True, sort_keys=True),
          ),
          flush=True)
    if not new_mp4s:
        print("[smoke] FAIL: no new episode mp4 found under %s" % EPISODES_DIR,
              flush=True)
        for path in (args.server_log, RUNTIME_LOG):
            print("[smoke] tail %s" % path, flush=True)
            for line in _tail_file(path):
                print("  " + line, flush=True)
        return 1

    bad_log_markers = (
        "keep node 'modelsampling' not in results",
        "node 'modelsampling' not in results",
        "missing-sigmas",
        "class 'sigmas' unresolved",
    )
    tails = "\n".join(
        "\n".join(_tail_file(path, 200)) for path in (args.server_log, RUNTIME_LOG)
    )
    hits = [m for m in bad_log_markers if m.lower() in tails.lower()]
    if hits:
        print("[smoke] FAIL: distilled_native forbidden/error marker(s) in logs: %s"
              % ", ".join(hits), flush=True)
        return 1

    durable_mp4s = [
        p for p in new_mp4s
        if (os.sep + "_shared" + os.sep + "tmp" + os.sep) not in p
    ]
    if not durable_mp4s:
        print("[smoke] FAIL: only tmp mp4s were produced; no durable episode mp4",
              flush=True)
        return 1

    print("[smoke] PASS: queued, completed, and found %d episode mp4(s)"
          % len(durable_mp4s), flush=True)
    for path in durable_mp4s[-8:]:
        print("[smoke] output: %s (%d bytes)" % (path, os.path.getsize(path)),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
