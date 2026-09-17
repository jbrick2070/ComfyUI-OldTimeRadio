"""Submit otr_cloud_low_1act with Comfy API key on the writer hidden input.

Headless --cpu on :8000 has no Desktop session, so Comfy does not inject
api_key_comfy_org. The key is read from the process env and never written
to the prompt dump.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import otr_api  # noqa: E402
from otr_canonical_api_run import build_api_prompt  # noqa: E402


class _Args:
    def __init__(
        self,
        workflow: str,
        act_count: str,
        comfyui_url: str,
        *,
        source_bank: str | None = None,
        visual_style: str | None = None,
        premise: str | None = None,
        replay_from: str | None = None,
        source_ref: str | None = None,
    ) -> None:
        self.workflow = workflow
        self.act_count = act_count
        self.comfyui_url = comfyui_url
        self.offline_schemas = False
        self.profile = None
        self.set = []
        if source_ref is not None:
            self.set.append("OTR_LedgerScriptWriter.source_ref=" + str(source_ref))
        self.title = None
        self.run_label = None
        self.replay_from = replay_from
        self.premise = premise
        self.source_bank = source_bank
        self.visual_style = visual_style
        self.creative_model = None
        self.technical_model = None
        self.google_slot_a_model = None
        self.google_slot_b_model = None
        self.num_characters = None


def _inject_writer_key(prompt: dict, key: str) -> str:
    writer_ids = [
        nid
        for nid, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == "OTR_LedgerScriptWriter"
    ]
    if not writer_ids:
        raise SystemExit("prompt has no OTR_LedgerScriptWriter")
    nid = writer_ids[0]
    prompt[nid].setdefault("inputs", {})["api_key_comfy_org"] = key
    return str(nid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--act-count", default="1")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--poll-s", type=int, default=10)
    parser.add_argument("--source-bank", default=None)
    parser.add_argument("--visual-style", default=None)
    parser.add_argument("--premise", default=None)
    parser.add_argument("--replay-from", default=None)
    parser.add_argument("--source-ref", default=None)
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Print prompt_id and exit without polling history",
    )
    args = parser.parse_args()
    key = (os.environ.get("OTR_COMFY_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OTR_COMFY_API_KEY missing in this process")
    print(f"[cloud-submit] key_len={len(key)} prefix={key[:8]}", flush=True)

    url = args.comfyui_url.rstrip("/")
    otr_api.COMFYUI_URL = url
    ns = _Args(
        args.workflow,
        args.act_count,
        url,
        source_bank=args.source_bank,
        visual_style=args.visual_style,
        premise=args.premise,
        replay_from=args.replay_from,
        source_ref=args.source_ref,
    )
    # otr_canonical_api_run.main mutates COMFYUI_URL from --comfyui-url;
    # build_api_prompt reads schemas from otr_api.COMFYUI_URL.
    prompt, applied = build_api_prompt(ns)
    writer_id = _inject_writer_key(prompt, key)
    print(f"[cloud-submit] writer_node={writer_id}", flush=True)
    if applied:
        for item in applied:
            print(f"[cloud-submit] applied {item}", flush=True)

    # Headless has no Desktop session, so partner nodes need the key in
    # extra_data as well as the writer hidden input.
    import uuid
    import requests

    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{url}/prompt",
        json={
            "prompt": prompt,
            "client_id": client_id,
            "extra_data": {"api_key_comfy_org": key},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"POST /prompt -> HTTP {resp.status_code}: {resp.text[:500]}"
        )
    body = resp.json()
    if body.get("error"):
        raise SystemExit(f"submit_prompt error: {body.get('error')!r}")
    if body.get("node_errors"):
        raise SystemExit(f"submit_prompt node_errors: {body.get('node_errors')!r}")
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise SystemExit(f"submit_prompt missing prompt_id: {body!r}")
    print(f"[cloud-submit] QUEUED prompt_id={prompt_id}", flush=True)
    if args.no_wait:
        return 0

    def heartbeat(elapsed_s: float, status: dict) -> None:
        phase = str(status.get("status_str") or "pending")
        print(
            f"[cloud-submit] t={int(elapsed_s)}s prompt_id={prompt_id} status={phase}",
            flush=True,
        )

    status, err = otr_api.poll_history(
        prompt_id, timeout_s=args.timeout, poll_s=args.poll_s, on_tick=heartbeat,
    )
    print(f"[cloud-submit] RESULT {status} prompt_id={prompt_id}", flush=True)
    if err:
        print(f"[cloud-submit] ERROR {err}", flush=True)
        return 1
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
