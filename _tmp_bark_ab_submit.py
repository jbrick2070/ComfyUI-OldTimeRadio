"""Queue one 16gb_low 1-act listen arm (Kokoro or shipped Bark).

Voice widgets are CastLock + the two TTS nodes, so they are patched by
name, not through patch_creative. Does not touch CPU :8000.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import otr_api  # noqa: E402
from otr_canonical_api_run import _apply_writer_shortcuts, _node_id_for, build_api_prompt  # noqa: E402


class _Args:
    def __init__(
        self,
        workflow: str,
        act_count: str,
        comfyui_url: str,
        *,
        source_bank: str,
        visual_style: str,
        run_label: str,
    ) -> None:
        self.workflow = workflow
        self.act_count = act_count
        self.comfyui_url = comfyui_url
        self.offline_schemas = False
        self.profile = None
        self.set = []
        self.title = None
        self.run_label = run_label
        self.replay_from = None
        self.premise = None
        self.source_bank = source_bank
        self.visual_style = visual_style
        self.creative_model = None
        self.technical_model = None
        self.google_slot_a_model = None
        self.google_slot_b_model = None
        self.num_characters = None
        self.machine = None


def _patch_voice(workflow: dict, schemas: dict, engine: str) -> list[str]:
    applied: list[str] = []
    bank = "bark_legacy" if engine == "bark" else "kokoro_builtin"
    patches = (
        ("OTR_CastLock", "voice_bank", bank),
        ("OTR_CastLock", "char_voice_engine", engine),
        ("OTR_CastLock", "announcer_voice_engine", engine),
        ("OTR_BatchCharacterVoices", "engine", engine),
        ("OTR_AnnouncerVoice", "engine", engine),
        ("OTR_VideoDirector", "announcer_image_model", "lumina_image"),
        ("OTR_VideoDirector", "music_image_model", "lumina_image"),
        ("OTR_VideoDirector", "character_image_model", "lumina_image"),
    )
    for node_type, widget, value in patches:
        nid = _node_id_for(workflow, node_type)
        otr_api.patch_widget_by_name(workflow, nid, widget, value, schemas)
        applied.append(f"{node_type}.{widget}={value!r}")
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=("kokoro", "bark"))
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--poll-s", type=int, default=20)
    args = parser.parse_args()

    url = args.comfyui_url.rstrip("/")
    otr_api.COMFYUI_URL = url
    wf = str(ROOT / "workflows" / "variants" / "otr_16gb_low.json")
    ns = _Args(
        wf,
        "1",
        url,
        source_bank="my_story",
        visual_style="recur_frac",
        run_label=f"bark_ab_{args.engine}_1act",
    )
    # build_api_prompt applies writer shortcuts only. Voice is not on the
    # creative whitelist, so rebuild after an extra named patch.
    schemas = otr_api.fetch_schemas()
    workflow = otr_api.load_workflow(wf)
    applied = _apply_writer_shortcuts(workflow, schemas, ns)
    applied.extend(_patch_voice(workflow, schemas, args.engine))
    prompt = otr_api.workflow_to_api_prompt(workflow, schemas)
    dump = ROOT / "tmp" / f"bark_ab_{args.engine}_prompt.json"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(__import__("json").dumps(prompt, indent=2), encoding="utf-8")
    print(f"[bark-ab] url={url} engine={args.engine}", flush=True)
    for item in applied:
        print(f"[bark-ab] applied {item}", flush=True)
    print(f"[bark-ab] prompt_dump={dump}", flush=True)

    prompt_id = otr_api.submit_prompt(prompt)
    print(f"[bark-ab] QUEUED prompt_id={prompt_id}", flush=True)
    if args.no_wait:
        return 0

    def heartbeat(elapsed_s: float, status: dict) -> None:
        phase = str(status.get("status_str") or "pending")
        print(
            f"[bark-ab] t={int(elapsed_s)}s engine={args.engine} "
            f"prompt_id={prompt_id} status={phase}",
            flush=True,
        )

    status, err = otr_api.poll_history(
        prompt_id, timeout_s=args.timeout, poll_s=args.poll_s, on_tick=heartbeat,
    )
    print(f"[bark-ab] RESULT {status} engine={args.engine} prompt_id={prompt_id}", flush=True)
    if err:
        print(f"[bark-ab] ERROR {err}", flush=True)
        return 1
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
