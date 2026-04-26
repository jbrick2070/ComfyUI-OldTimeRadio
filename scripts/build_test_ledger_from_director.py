#!/usr/bin/env python3
"""Build a TEST-mode HuMo ledger from a workflow JSON's baked-in director_json.

Adapter for Goal 1 (HuMo Full-Episode Coverage smoke run).

Reads a TEST-style ComfyUI workflow whose `OTR_VideoPlan` node has a
Director JSON baked into ``widgets_values[0]``, expands the pass3 shot
plan via ``nodes.otr_video_plan.build_shot_plan``, and emits a synthetic
ledger that ``scripts/render_humo_batch.py`` can iterate to render one
HuMo MP4 per shot.

Why this script exists
----------------------
``OTR_VideoPlan.execute()`` already writes a production ledger via
``set_shots()`` — but the HuMo batch orchestrator iterates ``lines[]``
(per-shot speaker) rather than ``shots[]`` (visual prompts). For TEST
runs that don't go through Bark / SceneSequencer, there is no
script-writer stage that populates ``lines[]``.

This adapter bridges: it takes the same baked-in director, expands
pass3 shots, and emits a ledger with ``lines[]`` synthesised one-per-
shot, speaker rotated across the cast. The orchestrator then renders
one HuMo MP4 per ledger line.

Output ledger schema is a compatible subset of the production ledger
(``nodes/production_ledger.py`` schema ``l1-2026-04-24``): ``cast[]``,
``scenes[]``, ``shots[]``, ``lines[]``. Audio paths are intentionally
left None — pass ``--master-wav`` to the orchestrator instead.

Usage
-----
    python scripts/build_test_ledger_from_director.py \
        --workflow workflows/otr_videoplan_TEST_humo.json \
        --out output/old_time_radio/test_humo_ledger.json

Then feed it to the orchestrator (see ``docs/2026-04-25-humo-batch-pipeline.md``):

    python scripts/render_humo_batch.py \
        --ledger output/old_time_radio/test_humo_ledger.json \
        --master-wav input/humo_test.wav \
        --out-dir output/old_time_radio/humo_test \
        --scope all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root lives one level up from scripts/. Add it to sys.path so we
# can import the pure-Python helpers from nodes/otr_video_plan.py
# without needing ComfyUI on PATH.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Imports below are torch-free per nodes/otr_video_plan.py module docstring.
from nodes.otr_video_plan import (  # noqa: E402  -- after sys.path setup
    build_shot_plan,
)


# ---------------------------------------------------------------------------
# Workflow JSON parsing
# ---------------------------------------------------------------------------

def extract_director_json_from_workflow(wf_path: Path) -> str:
    """Pull the raw director_json string from a workflow's OTR_VideoPlan node.

    Looks for the first node whose ``type == "OTR_VideoPlan"`` and returns
    ``widgets_values[0]`` (the multiline director JSON). Raises if no such
    node exists.
    """
    with open(wf_path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    for n in wf.get("nodes", []) or []:
        if n.get("type") == "OTR_VideoPlan":
            wv = n.get("widgets_values") or []
            if wv and isinstance(wv[0], str) and wv[0].strip():
                return wv[0]
    raise RuntimeError(
        f"No OTR_VideoPlan node with a baked-in director_json found in {wf_path}"
    )


def extract_video_plan_widget_values(wf_path: Path) -> list[Any]:
    """Return the OTR_VideoPlan node's widgets_values list.

    Position layout matches OTRVideoPlan.INPUT_TYPES order:
        [0] director_json (string)
        [1] focus_character (string, "(all)" or a specific name)
        [2] shots_per_scene (int)
        [3] genre_flavor (string)
        [4] style_tail (string, optional)
        [5] include_final_end_frame (bool, optional)
    """
    with open(wf_path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    for n in wf.get("nodes", []) or []:
        if n.get("type") == "OTR_VideoPlan":
            return list(n.get("widgets_values") or [])
    raise RuntimeError(
        f"No OTR_VideoPlan node found in {wf_path}"
    )


# ---------------------------------------------------------------------------
# Director -> ledger
# ---------------------------------------------------------------------------

# Schema label distinct from the production ledger's "l1-2026-04-24" so
# downstream tooling can detect a synthetic test ledger and warn / route
# differently if needed.
TEST_LEDGER_SCHEMA = "test-humo-2026-04-25"


def director_json_to_test_ledger(
    director_json: str,
    *,
    episode_id: str,
    shots_per_scene: int = 2,
    genre_flavor: str = "hard_sci_fi",
    style_tail: str = "",
    include_final_end_frame: bool = True,
    clip_length_s: float = 3.88,
    speaker_strategy: str = "rotate",
) -> dict:
    """Expand a director_json into a synthetic test ledger.

    Parameters
    ----------
    director_json : str
        Raw JSON string from the workflow's OTR_VideoPlan widget.
    episode_id : str
        Test episode identifier. Becomes ``ledger.episode_id``.
    shots_per_scene : int
        Same semantic as ``OTRVideoPlan.shots_per_scene`` widget (default 2
        for TEST_humo).
    clip_length_s : float
        HuMo native clip length (3.88 s @ length=97 / 25 fps).
        Lines all use ``start_s=0`` so the orchestrator slices the same
        first ``clip_length_s`` of the master WAV for every shot. This
        keeps a short test WAV viable; for production, a populated
        ``lines[].start_s`` would be supplied upstream.
    speaker_strategy : {"rotate", "first"}
        How to assign a speaker to each shot. ``"rotate"`` cycles cast
        names round-robin across shots (default — exercises every
        portrait). ``"first"`` always picks the first cast member
        (single-character smoke test).
    """
    director = (
        json.loads(director_json) if isinstance(director_json, str) else director_json
    )
    if not isinstance(director, dict):
        director = {}

    visual_plan = director.get("visual_plan") or {}
    chars_dict = visual_plan.get("characters") or {}
    voice_assignments = director.get("voice_assignments") or {}

    # Cast — one row per character with portrait_prompt + voice info.
    cast: list[dict[str, Any]] = []
    for idx, name in enumerate(chars_dict.keys()):
        char_data = chars_dict.get(name) or {}
        voice_data = voice_assignments.get(name) or {}
        cast.append({
            "char_id": f"c{idx + 1:02d}",
            "name": name,
            "description": (char_data.get("portrait_prompt") or "").strip()
                            or (voice_data.get("notes") or "").strip(),
            "voice_preset": voice_data.get("voice_preset") or None,
            "line_count": 0,  # populated below after lines[] built
            "word_count": 0,
        })
    cast_names = [c["name"] for c in cast]

    # Scenes — preserve scene_id / shot_description / visual_prompt so
    # downstream tooling can present the same context as the workflow.
    scenes_in = visual_plan.get("scenes") or []
    scenes: list[dict[str, Any]] = []
    for s in scenes_in:
        scenes.append({
            "scene_id": s.get("scene_id") or f"scene_{len(scenes) + 1}",
            "description": (s.get("shot_description") or "").strip() or None,
            "env": (s.get("visual_prompt") or "").strip() or None,
            "line_count": 0,  # placeholder
            "word_count": 0,
        })

    # Pass3 shot plan via the same helper OTR_VideoPlan uses internally.
    # We pass focus_character="" so build_shot_plan uses the multi-character
    # composite portrait — the prompt itself isn't read by the orchestrator
    # (which only reads cast portraits via glob), but we still get the
    # canonical shots[] index back.
    primary_focus = cast_names[0] if cast_names else ""
    shot_plan = build_shot_plan(
        director_json,
        primary_focus,
        shots_per_scene=shots_per_scene,
        genre_flavor=genre_flavor,
        style_tail=style_tail,
        include_final_end_frame=include_final_end_frame,
    )

    shots_index = shot_plan.get("shots") or []
    tokens = shot_plan.get("tokens") or []

    # Build a frame_id -> visual_prompt lookup for shot description enrichment.
    frame_to_prompt: dict[str, str] = {}
    for tok in tokens:
        fid = tok.get("frame_id") or tok.get("shot_id")
        if fid:
            frame_to_prompt[fid] = tok.get("description") or ""

    # Convert each pass3 shot into one ledger shot, one ledger beat
    # (each shot is single-speaker in this synthetic test), and one
    # ledger line. The beat[] hierarchy was added 2026-04-25 PM (l2
    # schema) so the HuMo orchestrator's Goal 3 daisy-chain mode can
    # see boundary cues even on TEST runs that don't go through the
    # real script-writer / scene-sequencer.
    shots_out: list[dict[str, Any]] = []
    beats_out: list[dict[str, Any]] = []
    lines_out: list[dict[str, Any]] = []
    for idx, shot in enumerate(shots_index):
        shot_id = shot.get("shot_id") or f"shot_{idx + 1:03d}"
        scene_id = shot.get("scene_id") or ""
        start_frame_id = shot.get("start_frame_id")
        visual_prompt = frame_to_prompt.get(start_frame_id, "")

        # Speaker assignment.
        if speaker_strategy == "first" and cast_names:
            speaker = cast_names[0]
        elif cast_names:
            speaker = cast_names[idx % len(cast_names)]
        else:
            speaker = ""
        char_id = next(
            (c["char_id"] for c in cast if c["name"] == speaker),
            None,
        )

        # All lines start at 0 of the master WAV so the orchestrator
        # slices the same clip_length_s window every time. This keeps
        # short test WAVs viable. Production runs populate per-line
        # start_s / dur_s upstream and don't go through this script.
        start_s = 0.0
        dur_s = float(clip_length_s)
        beat_id = f"{shot_id}_b1"

        shots_out.append({
            "shot_id": shot_id,
            "scene_id": scene_id,
            "description": (visual_prompt or "")[:200] or None,
            "visual_prompt": visual_prompt,
            "png_path": None,
            "speakers": [speaker] if speaker else [],
            "beat_count": 1,
            "start_s": start_s,
            "dur_s": dur_s,
        })
        beats_out.append({
            "beat_id": beat_id,
            "shot_id": shot_id,
            "scene_id": scene_id,
            "speaker": speaker,
            "char_id": char_id,
            "line_ids": [shot_id],  # one line per shot in this synthetic test
            "start_s": start_s,
            "dur_s": dur_s,
        })
        lines_out.append({
            "line_id": shot_id,
            "shot_id": shot_id,
            "beat_id": beat_id,
            "scene_id": scene_id,
            "char_id": char_id,
            "speaker": speaker,
            "boundary": "shot_start",  # one beat per shot, so always a fresh shot anchor
            "text": "",
            "traits": None,
            "char_count": 0,
            "word_count": 0,
            "bark_wav_path": None,
            "start_s": start_s,
            "dur_s": dur_s,
        })

    # Update cast.line_count + scene.line_count tallies.
    for c in cast:
        c["line_count"] = sum(1 for ln in lines_out if ln.get("speaker") == c["name"])
    for sc in scenes:
        sc["line_count"] = sum(
            1 for ln in lines_out if ln.get("scene_id") == sc["scene_id"]
        )

    return {
        "schema_version": TEST_LEDGER_SCHEMA,
        "episode_id": episode_id,
        "source": "build_test_ledger_from_director.py",
        "commit": None,
        "total_episode_dur_s": float(clip_length_s) * len(lines_out),
        "total_char_count": 0,
        "total_word_count": 0,
        "total_dialogue_lines": len(lines_out),
        "total_beats": len(beats_out),
        "cast": cast,
        "scenes": scenes,
        "shots": shots_out,
        "beats": beats_out,
        "lines": lines_out,
        "sfx": [],
        "music": [],
        "clips": [],
        "final_audio_path": None,
        "final_video_path": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a TEST HuMo ledger from a workflow's baked-in director_json.",
    )
    p.add_argument(
        "--workflow",
        type=Path,
        default=Path("workflows/otr_videoplan_TEST_humo.json"),
        help="Path to the workflow JSON containing an OTR_VideoPlan node "
        "with director_json baked into widgets_values[0].",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/old_time_radio/test_humo_ledger.json"),
        help="Where to write the synthetic test ledger.",
    )
    p.add_argument(
        "--episode-id",
        default=None,
        help="Override episode_id. Defaults to a slug derived from the "
        "director's episode_title (or 'test_humo_episode' if absent).",
    )
    p.add_argument(
        "--shots-per-scene",
        type=int,
        default=None,
        help="Override shots_per_scene. Defaults to the workflow's "
        "OTR_VideoPlan widget value (or 2 if missing).",
    )
    p.add_argument(
        "--clip-length",
        type=float,
        default=3.88,
        help="HuMo clip length in seconds (default: 3.88 = length=97 @ 25 fps).",
    )
    p.add_argument(
        "--speaker-strategy",
        choices=("rotate", "first"),
        default="rotate",
        help="How to assign speakers to shots. 'rotate' cycles the cast "
        "(default — exercises every portrait); 'first' always uses the "
        "first cast member.",
    )
    return p.parse_args()


def _slug_episode(title: str) -> str:
    import re
    s = (title or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:60] or "test_humo_episode"


def main() -> int:
    args = parse_args()

    if not args.workflow.exists():
        print(f"FATAL: workflow not found: {args.workflow}", file=sys.stderr)
        return 2

    director_json = extract_director_json_from_workflow(args.workflow)
    widget_values = extract_video_plan_widget_values(args.workflow)

    # Pull defaults from the workflow widget order, but allow CLI override.
    wf_focus = widget_values[1] if len(widget_values) > 1 else "(all)"
    wf_shots_per_scene = widget_values[2] if len(widget_values) > 2 else 2
    wf_genre = widget_values[3] if len(widget_values) > 3 else "hard_sci_fi"
    wf_style_tail = widget_values[4] if len(widget_values) > 4 else ""
    wf_include_final = (
        widget_values[5] if len(widget_values) > 5 else True
    )

    shots_per_scene = (
        int(args.shots_per_scene)
        if args.shots_per_scene is not None
        else int(wf_shots_per_scene or 2)
    )

    episode_id = args.episode_id
    if not episode_id:
        try:
            director = json.loads(director_json)
            episode_id = _slug_episode(director.get("episode_title", ""))
        except json.JSONDecodeError:
            episode_id = "test_humo_episode"
        if not episode_id:
            episode_id = "test_humo_episode"

    ledger = director_json_to_test_ledger(
        director_json,
        episode_id=episode_id,
        shots_per_scene=shots_per_scene,
        genre_flavor=str(wf_genre or "hard_sci_fi"),
        style_tail=str(wf_style_tail or ""),
        include_final_end_frame=bool(wf_include_final),
        clip_length_s=float(args.clip_length),
        speaker_strategy=args.speaker_strategy,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline keeps git diffs clean

    print(
        f"[ok] wrote {args.out}\n"
        f"     episode_id     = {ledger['episode_id']}\n"
        f"     cast           = {len(ledger['cast'])} ({', '.join(c['name'] for c in ledger['cast'])})\n"
        f"     scenes         = {len(ledger['scenes'])}\n"
        f"     shots          = {len(ledger['shots'])}\n"
        f"     lines          = {len(ledger['lines'])} "
        f"(speaker_strategy={args.speaker_strategy})\n"
        f"     clip_length_s  = {args.clip_length}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
