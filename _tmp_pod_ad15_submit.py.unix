#!/usr/bin/env python3
"""Queue 3-act 16gb_animatediff: OpenRouter writer + Google TTS. Voice is
patched by name, not patch_creative."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OTR = Path(os.environ.get(
    "OTR_REPO_ROOT",
    "/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio",
))
sys.path.insert(0, str(OTR / "scripts"))
import otr_api  # noqa: E402
from otr_canonical_api_run import (  # noqa: E402
    _apply_set,
    _apply_writer_shortcuts,
    _node_id_for,
)

URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
WF = str(OTR / "workflows" / "variants" / "otr_16gb_animatediff.json")


class _Args:
    workflow = WF
    act_count = "3"
    comfyui_url = URL
    offline_schemas = False
    profile = None
    set = [
        "OTR_LedgerScriptWriter.openrouter_slot_a_model=~openai/gpt-latest",
        "OTR_LedgerScriptWriter.openrouter_slot_b_model=~openai/gpt-mini-latest",
    ]
    title = None
    run_label = "ad15_3act_or_gtts"
    replay_from = None
    premise = None
    source_bank = "my_story"
    visual_style = "recur_frac"
    creative_model = "openrouter:slot-a"
    technical_model = "openrouter:slot-b"
    google_slot_a_model = None
    google_slot_b_model = None
    num_characters = None
    machine = None


def _patch_google(workflow, schemas):
    applied = []
    patches = (
        ("OTR_CastLock", "voice_bank", "google_tts"),
        ("OTR_CastLock", "char_voice_engine", "google_tts"),
        ("OTR_CastLock", "announcer_voice_engine", "google_tts"),
        ("OTR_BatchCharacterVoices", "engine", "google_tts"),
        ("OTR_AnnouncerVoice", "engine", "google_tts"),
    )
    for node_type, widget, value in patches:
        nid = _node_id_for(workflow, node_type)
        otr_api.patch_widget_by_name(workflow, nid, widget, value, schemas)
        applied.append(f"{node_type}.{widget}={value}")
    return applied


def main() -> int:
    otr_api.COMFYUI_URL = URL
    or_chars = len((os.environ.get("OPENROUTER_API_KEY") or "").strip())
    g_chars = len((
        os.environ.get("OTR_GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip())
    print(f"HEADLESS OpenRouter_chars={or_chars} Google_chars={g_chars}", flush=True)
    if or_chars < 20:
        print("OPENROUTER_MISSING_IN_PROCESS", flush=True)
        return 7
    if g_chars < 20:
        print("GOOGLE_MISSING_IN_PROCESS", flush=True)
        return 7
    schemas = otr_api.fetch_schemas()
    workflow = otr_api.load_workflow(WF)
    ns = _Args()
    applied = _apply_writer_shortcuts(workflow, schemas, ns)
    for spec in ns.set:
        applied.append(_apply_set(workflow, schemas, spec))
    applied.extend(_patch_google(workflow, schemas))
    prompt = otr_api.workflow_to_api_prompt(workflow, schemas)
    dump = Path("/workspace/otr-config/ad15_3act_prompt.json")
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    for item in applied:
        print(f"applied {item}", flush=True)
    prompt_id = otr_api.submit_prompt(prompt)
    print(f"QUEUED prompt_id={prompt_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
