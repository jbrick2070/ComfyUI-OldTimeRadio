"""
Baseline runner for Phase 0 audio regression.
================================================

Executes the full OTR workflow via ComfyUI's HTTP API with fixed seeds,
captures the EpisodeAssembler audio output, and saves it as a WAV file.

Two entry points:
  run_episode_and_save_wav(seeds, output_path) -> path
  run_episode_and_get_audio_bytes(seeds) -> bytes

Requires ComfyUI Desktop running on localhost:8000 (default port).
"""

import io
import json
import os
import struct
import time
import logging

log = logging.getLogger("OTR.baseline")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMFYUI_BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
COMFYUI_OUTPUT_DIR = os.environ.get(
    "COMFYUI_OUTPUT",
    r"C:\Users\jeffr\Documents\ComfyUI\output"
)
_WORKFLOW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "workflows", "otr_scifi_16gb_full.json"
)

# Types that are never stored in widgets_values — always come via links
NON_WIDGET_TYPES = {
    "AUDIO", "IMAGE", "LATENT", "MASK", "MODEL", "VAE", "CLIP",
    "CONDITIONING", "CONTROL_NET", "STYLE_MODEL",
}

# Polling config
POLL_INTERVAL_S = 5.0
POLL_TIMEOUT_S = 1800.0  # 30 minutes max for a full episode


# ---------------------------------------------------------------------------
# Workflow -> API prompt conversion
# ---------------------------------------------------------------------------

def _load_workflow():
    """Load the workflow JSON from disk."""
    with open(_WORKFLOW_PATH, encoding="utf-8") as f:
        return json.load(f)


def _fetch_schemas(base_url):
    """Fetch all node schemas from ComfyUI /object_info."""
    import requests
    resp = requests.get(f"{base_url}/object_info", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _workflow_to_api_prompt(workflow, schemas):
    """Convert a ComfyUI workflow JSON to the API prompt format.

    Mirrors the algorithm in otr_queue.py / otr_queue2.py.
    """
    # link_id -> [src_node_id_str, src_slot]
    link_map = {}
    for lnk in workflow["links"]:
        link_id, src_node, src_slot = lnk[0], lnk[1], lnk[2]
        link_map[link_id] = [str(src_node), src_slot]

    # node_id -> {input_name: link_id}
    # node_id -> set of linked inputs that KEPT their widget slot
    node_links = {}
    node_converted_widgets = {}
    for node in workflow["nodes"]:
        nid = node["id"]
        node_links[nid] = {}
        node_converted_widgets[nid] = set()
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                node_links[nid][inp["name"]] = inp["link"]
                # "widget" flag means this was a widget converted to input.
                # Its slot is STILL in widgets_values.
                if inp.get("widget"):
                    node_converted_widgets[nid].add(inp["name"])

    def is_widget_capable(node_type, section, name):
        type_info = schemas.get(node_type, {}).get("input", {}).get(section, {}).get(name)
        if not type_info:
            return False
        if isinstance(type_info[0], list):
            return True  # dropdown = widget
        param_type = type_info[0] if isinstance(type_info[0], str) else None
        return param_type not in NON_WIDGET_TYPES

    prompt = {}

    for node in workflow["nodes"]:
        nid = node["id"]
        ntype = node["type"]
        linked = node_links.get(nid, {})
        converted = node_converted_widgets.get(nid, set())

        # Built-in nodes (PreviewAudio, etc.) not in schemas
        if ntype not in schemas:
            inputs = {}
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    inputs[inp["name"]] = link_map[inp["link"]]
            prompt[str(nid)] = {"class_type": ntype, "inputs": inputs}
            continue

        schema = schemas[ntype].get("input", {})
        req_params = list(schema.get("required", {}).keys())
        opt_params = list(schema.get("optional", {}).keys())

        all_params = (
            [(p, "required") for p in req_params]
            + [(p, "optional") for p in opt_params]
        )
        widget_capable = [
            (p, sec) for p, sec in all_params
            if is_widget_capable(ntype, sec, p)
        ]

        # widgets_values alignment rules (BUG-LOCAL-003):
        #   - Linked inputs WITH "widget" flag in workflow JSON: slot KEPT
        #   - Linked inputs WITHOUT "widget" flag: slot REMOVED
        #   - Unlinked widget-capable inputs: slot present as normal
        # So: include a param in the positional mapping if it is either
        # unlinked OR a converted widget (linked but kept its slot).
        wv = list(node.get("widgets_values", []))
        params_with_wv_slot = [
            (p, sec) for p, sec in widget_capable
            if p not in linked or p in converted
        ]
        wv_map = {}
        for i, (p, _) in enumerate(params_with_wv_slot):
            if i < len(wv):
                wv_map[p] = wv[i]

        inputs = {}
        for p, section in all_params:
            if p in linked:
                inputs[p] = link_map[linked[p]]
            elif p in wv_map:
                val = wv_map[p]
                # Fix emoji mismatch: workflow may store real emoji but
                # the running ComfyUI may use [EMOJI]/[FAST] placeholders
                # in dropdown options. Match by suffix if exact fails.
                ptype = schema.get(section, {}).get(p)
                if ptype and isinstance(ptype[0], list) and isinstance(val, str):
                    options = ptype[0]
                    if val not in options:
                        # Try matching by stripping leading emoji/tag
                        val_suffix = val.lstrip()
                        # Remove first word if it looks like emoji
                        for opt in options:
                            opt_suffix = opt.lstrip()
                            # Match on the text portion after emoji/tag
                            if _dropdown_text_match(val, opt):
                                val = opt
                                break
                inputs[p] = val

        prompt[str(nid)] = {"class_type": ntype, "inputs": inputs}

    return prompt


def _dropdown_text_match(workflow_val, schema_val):
    """Match dropdown values that differ only in emoji vs placeholder prefix.

    E.g. '\\U0001f4fb standard (12 min)' matches '[EMOJI] standard (12 min)'
    """
    import re
    # Strip leading emoji (Unicode > 0x1000) or [TAG] prefix
    def strip_prefix(s):
        # Remove leading [TAG] style prefixes
        s = re.sub(r'^\[.*?\]\s*', '', s)
        # Remove leading emoji characters
        return re.sub(r'^[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]+\s*', '', s)

    return strip_prefix(workflow_val) == strip_prefix(schema_val)


# v2 visual nodes have been removed (sidecar architecture pending).
# This set is kept empty for backwards compatibility with any code
# that references it during prompt stripping.
_V2_PLACEHOLDER_TYPES = set()


def _dump_schema_for_node(schemas, node_type, out_path):
    """Write the raw ComfyUI schema for a node to disk for debugging.

    Useful for diagnosing widget-value alignment bugs where the schema
    param ordering does not match the INPUT_TYPES definition order.
    """
    schema = schemas.get(node_type)
    if schema:
        import json as _json
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(schema.get("input", {}), f, indent=2)
        print(f"  Schema dump: {out_path}")
    else:
        print(f"  WARNING: {node_type} not found in schemas")


# Known-correct widget values for OTR_BatchAudioGenGenerator.
# BUG-LOCAL-008: schema param ordering from /object_info does not match
# INPUT_TYPES definition order, causing positional wv mapping to start
# at the wrong slot. Until root cause is fixed, hardcode correct values.
_AUDIOGEN_CORRECT_INPUTS = {
    "episode_seed": "",
    "model_id": "facebook/audiogen-medium",
    "guidance_scale": 3.0,
    "default_duration": 3.0,
}


def _fix_known_widget_drift(prompt):
    """Apply hardcoded corrections for nodes with known widget drift.

    BUG-LOCAL-008: OTR_BatchAudioGenGenerator gets episode_seed and
    model_id from the wrong wv slots due to schema ordering mismatch.
    This correction overwrites with the workflow's intended values.
    """
    for nid, node_data in prompt.items():
        if node_data.get("class_type") == "OTR_BatchAudioGenGenerator":
            before = {k: node_data["inputs"].get(k) for k in _AUDIOGEN_CORRECT_INPUTS}
            node_data["inputs"].update(_AUDIOGEN_CORRECT_INPUTS)
            print(f"  Fixed node #{nid} OTR_BatchAudioGenGenerator widget drift")
            log.info("BUG-LOCAL-008 fix applied: was %s, now correct", before)
    return prompt


def _strip_non_audio_nodes(prompt):
    """Remove v2 placeholder nodes from the prompt.

    These nodes have required inputs (MODEL, CLIP, VAE) that are not
    connected in the audio-only workflow, so ComfyUI would reject them.
    They are not part of the audio pipeline and should not execute.
    """
    to_remove = [
        nid for nid, data in prompt.items()
        if data.get("class_type") in _V2_PLACEHOLDER_TYPES
    ]
    for nid in to_remove:
        del prompt[nid]
        log.info("Stripped non-audio node #%s from prompt", nid)
    if to_remove:
        print(f"  Stripped {len(to_remove)} v2 placeholder nodes")
    return prompt


def _inject_seeds_into_prompt(prompt, seeds):
    """Override seed values in the API prompt dict.

    Finds nodes by class_type and sets their 'seed' input to the
    fixed value. Works on the API-format prompt (not workflow JSON).
    """
    for nid, node_data in prompt.items():
        class_type = node_data.get("class_type", "")
        if class_type in seeds and "seed" in node_data.get("inputs", {}):
            old_seed = node_data["inputs"]["seed"]
            node_data["inputs"]["seed"] = seeds[class_type]
            log.info(
                "Injected seed %d into %s (node #%s, was %s)",
                seeds[class_type], class_type, nid, old_seed,
            )
    return prompt


def _inject_preview_audio_node(prompt, assembler_node_id):
    """Add a PreviewAudio node that captures EpisodeAssembler output.

    This gives us a .flac file in ComfyUI's temp dir that we can
    retrieve via /history.
    """
    # Use a high node ID that won't collide
    preview_nid = "9999"
    prompt[preview_nid] = {
        "class_type": "PreviewAudio",
        "inputs": {
            "audio": [str(assembler_node_id), 0],
        },
    }
    log.info(
        "Injected PreviewAudio node #%s tapping assembler #%s",
        preview_nid, assembler_node_id,
    )
    return prompt, preview_nid


# ---------------------------------------------------------------------------
# ComfyUI API interaction
# ---------------------------------------------------------------------------

def _find_assembler_node_id(prompt):
    """Find the EpisodeAssembler node ID in the prompt."""
    for nid, node_data in prompt.items():
        if node_data.get("class_type") == "OTR_EpisodeAssembler":
            return nid
    raise RuntimeError("OTR_EpisodeAssembler not found in workflow prompt")


def _submit_prompt(prompt, base_url):
    """POST the prompt to ComfyUI and return the prompt_id."""
    import requests
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    resp = requests.post(
        f"{base_url}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != 200:
        # Print full error body for debugging
        print(f"\nComfyUI rejected the prompt (HTTP {resp.status_code}):")
        try:
            err_body = resp.json()
            print(json.dumps(err_body, indent=2)[:3000])
        except Exception:
            print(resp.text[:3000])
        resp.raise_for_status()

    result = resp.json()

    if "error" in result:
        raise RuntimeError(f"ComfyUI rejected prompt: {result['error']}")

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in response: {result}")

    log.info("Submitted prompt, got prompt_id=%s", prompt_id)
    return prompt_id


def _poll_for_completion(prompt_id, base_url, timeout_s=POLL_TIMEOUT_S):
    """Poll /history until the prompt completes or times out.

    ComfyUI Desktop may return history entries with varied status structures.
    We treat a prompt as complete if ANY of these are true:
      - status["completed"] is True
      - status["status_str"] == "success"
      - outputs dict is non-empty (execution produced results)
    """
    import requests

    start = time.time()
    print(f"Waiting for execution (timeout {timeout_s/60:.0f} min)...")

    while time.time() - start < timeout_s:
        try:
            resp = requests.get(
                f"{base_url}/history/{prompt_id}", timeout=15
            )
            resp.raise_for_status()
            history = resp.json()
        except Exception as e:
            log.warning("History poll error: %s", e)
            time.sleep(POLL_INTERVAL_S)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            outputs = entry.get("outputs", {})

            # Log status on first hit for debugging
            log.debug("History status: %s", json.dumps(status))

            completed = (
                status.get("completed", False)
                or status.get("status_str") == "success"
                or bool(outputs)  # non-empty outputs = execution finished
            )
            if completed:
                elapsed = time.time() - start
                print(f"\nExecution completed in {elapsed/60:.1f} minutes.")
                if not outputs:
                    print("  WARNING: completed but outputs dict is empty")
                return entry

            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI execution failed: {json.dumps(status, indent=2)}"
                )

        elapsed = time.time() - start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"  ... {mins}m {secs}s elapsed", end="\r", flush=True)
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(
        f"Execution did not complete within {timeout_s/60:.0f} minutes"
    )


def _extract_audio_from_history(history_entry, preview_nid, base_url):
    """Extract the PreviewAudio output from the history entry.

    PreviewAudio saves a .flac to ComfyUI temp and reports it in
    the history outputs. We download it via the /view endpoint.
    """
    import requests

    outputs = history_entry.get("outputs", {})

    # Check PreviewAudio node output
    preview_out = outputs.get(preview_nid, {})
    audio_list = preview_out.get("audio", [])

    if not audio_list:
        # Fallback: check all nodes for audio output
        for nid, node_out in outputs.items():
            if "audio" in node_out:
                audio_list = node_out["audio"]
                log.info("Found audio output on node #%s instead", nid)
                break

    if not audio_list:
        available = {
            nid: list(out.keys()) for nid, out in outputs.items()
        }
        raise RuntimeError(
            f"No audio output found in history. "
            f"Available outputs: {available}"
        )

    audio_info = audio_list[0]
    filename = audio_info.get("filename")
    subfolder = audio_info.get("subfolder", "")
    audio_type = audio_info.get("type", "temp")

    log.info("Downloading audio: %s (type=%s)", filename, audio_type)

    # Download via /view endpoint
    params = {"filename": filename, "type": audio_type}
    if subfolder:
        params["subfolder"] = subfolder
    resp = requests.get(f"{base_url}/view", params=params, timeout=60)
    resp.raise_for_status()

    return resp.content, filename


def _flac_to_wav_bytes(flac_bytes):
    """Convert FLAC bytes to WAV bytes for consistent hashing.

    Uses ffmpeg (available in ComfyUI environments) for conversion
    to ensure exact PCM output.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_in:
        tmp_in.write(flac_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".flac", ".wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-acodec", "pcm_s16le",
                "-ar", "48000",
                "-ac", "1",
                tmp_out_path,
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_episode_and_get_audio_bytes(seeds):
    """Run the full workflow and return episode audio as WAV bytes.

    Steps:
    1. Load workflow, convert to API prompt format
    2. Inject fixed seeds for deterministic output
    3. Inject PreviewAudio node to capture the audio
    4. Submit to ComfyUI via HTTP API
    5. Poll for completion
    6. Download the audio file from history
    7. Convert to WAV and return bytes
    """
    import requests  # fail fast if not available

    print("Loading workflow...")
    workflow = _load_workflow()

    print("Fetching node schemas from ComfyUI...")
    schemas = _fetch_schemas(COMFYUI_BASE)

    print("Converting workflow to API format...")
    prompt = _workflow_to_api_prompt(workflow, schemas)
    print(f"  Built prompt with {len(prompt)} nodes")

    # Dump schema for known problem node (BUG-LOCAL-008 diagnosis)
    _dump_schema_for_node(
        schemas, "OTR_BatchAudioGenGenerator",
        os.path.join(os.path.dirname(__file__), "debug_audiogen_schema.json")
    )

    print("Stripping non-audio v2 placeholder nodes...")
    prompt = _strip_non_audio_nodes(prompt)

    # Find assembler before injecting seeds
    assembler_nid = _find_assembler_node_id(prompt)
    print(f"  EpisodeAssembler is node #{assembler_nid}")

    print("Applying known widget drift corrections (BUG-LOCAL-008)...")
    prompt = _fix_known_widget_drift(prompt)

    print("Injecting fixed seeds...")
    prompt = _inject_seeds_into_prompt(prompt, seeds)

    print("Injecting PreviewAudio capture node...")
    prompt, preview_nid = _inject_preview_audio_node(prompt, assembler_nid)

    # Dump prompt for debugging
    debug_path = os.path.join(os.path.dirname(__file__), "debug_prompt.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)
    print(f"  Debug prompt saved to: {debug_path}")

    print(f"Submitting to ComfyUI at {COMFYUI_BASE}...")
    prompt_id = _submit_prompt(prompt, COMFYUI_BASE)

    print(f"Prompt ID: {prompt_id}")
    history_entry = _poll_for_completion(prompt_id, COMFYUI_BASE)

    print("Extracting audio from execution results...")
    audio_bytes, filename = _extract_audio_from_history(
        history_entry, preview_nid, COMFYUI_BASE
    )
    print(f"  Got {len(audio_bytes):,} bytes from {filename}")

    # Convert to WAV if we got FLAC
    if filename.endswith(".flac"):
        print("Converting FLAC to WAV (PCM s16le, 48kHz, mono)...")
        wav_bytes = _flac_to_wav_bytes(audio_bytes)
        print(f"  WAV: {len(wav_bytes):,} bytes")
        return wav_bytes
    elif filename.endswith(".wav"):
        return audio_bytes
    else:
        print(f"  Warning: unexpected format {filename}, attempting FLAC conversion")
        return _flac_to_wav_bytes(audio_bytes)


def run_episode_and_save_wav(seeds, output_path):
    """Run the full workflow and save episode audio as WAV file.

    This is called by --capture-baseline mode.
    """
    wav_bytes = run_episode_and_get_audio_bytes(seeds)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(wav_bytes)
    print(f"Saved baseline WAV: {output_path} ({len(wav_bytes):,} bytes)")
    return output_path
