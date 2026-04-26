#!/usr/bin/env python3
"""FLUX batch orchestrator for OTR silent-test pipeline (Pattern B).

Reads an L2 ledger (schema l2-2026-04-25, written by
build_silent_test_episode.py or the FULL pipeline via production_ledger),
walks it for two kinds of FLUX render targets, and submits each one as a
single /prompt call against a running ComfyUI server. ComfyUI's model
cache stays warm across calls so the FLUX cold-load (~25 s) amortises
across N renders.

Two target kinds:

  PORTRAITS  -- one per cast member. Clean head/shoulders shot of the
                speaker, used as HuMo's identity anchor and as the alpha
                baseline in Goal 3 hybrid-anchor blends.
                File: otr/stills/pass1_portrait_<speaker_slug>.png

  COMPOSITES -- one per (shot, speaker) pair. The speaker placed in the
                shot's scene context. C1 of every shot/beat references
                this as ref_image. Same-speaker beats inside a shot
                reuse the same composite (cinematic continuity), so a
                beat_resume-tagged clip uses this composite + the
                speaker's prior last frame, NOT a fresh portrait.
                File: otr/stills/pass3_<shot_id>_<speaker_slug>.png

Usage (production):
  python scripts/render_flux_batch.py \\
      --ledger output/old_time_radio/silent_test_astrotech/ledger.json \\
      --out-dir C:/Users/jeffr/Documents/ComfyUI/output \\
      --comfyui-url http://localhost:8000 \\
      --scope all

Scope options:
  all                  portraits + composites
  portraits-only       just the cast portraits
  composites-only      just the per-(shot, speaker) composites
  custom:<filter>      e.g. custom:shot=shot_011 to limit to one shot

Architecture: pure Python stdlib + HTTP via urllib + ComfyUI /prompt
endpoint. No new ComfyUI custom nodes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLUX batch orchestrator for OTR silent-test pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ledger",
        required=True,
        type=Path,
        help="L2 ledger JSON path (silent_test or production).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\jeffr\Documents\ComfyUI\output"),
        help="ComfyUI output directory where FLUX images land.",
    )
    p.add_argument(
        "--comfyui-url",
        default="http://localhost:8000",
        help="Running ComfyUI server URL.",
    )
    p.add_argument(
        "--scope",
        default="all",
        help="all / portraits-only / composites-only / custom:<filter>",
    )
    p.add_argument(
        "--mode",
        default="bundled",
        choices=["bundled", "serial"],
        help=(
            "bundled (default): submit one big graph that renders ALL targets "
            "in a single ComfyUI queue slot, sharing the FLUX model load and "
            "negative CLIP encode. serial: submit one /prompt per target "
            "(legacy mode, ~5-10s slower per target)."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default="flux1-dev-fp8.safetensors",
        help="FLUX checkpoint filename.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=20,
        help="KSampler steps.",
    )
    p.add_argument(
        "--cfg",
        type=float,
        default=3.5,
        help="KSampler cfg.",
    )
    p.add_argument(
        "--style-tail",
        default=("cinematic 35mm film still, dim starship interior, "
                 "volumetric haze, shallow depth of field"),
        help="Style suffix appended to every prompt.",
    )
    p.add_argument(
        "--negative-prompt",
        default=("cinematic, 35mm film, anamorphic lens, volumetric lighting, "
                 "heavy vignette, muted color grade, sharp focus"),
        help="Default negative prompt for all renders.",
    )
    p.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Cap on number of targets rendered (test runs).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets whose output PNG already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, don't submit any prompts.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between /history polls while a prompt runs.",
    )
    p.add_argument(
        "--prompt-timeout",
        type=float,
        default=300.0,
        help="Hard cap on a single FLUX prompt's wall-clock (seconds).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str, limit: int = 40) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE.sub("_", s).strip("_")
    return (s[:limit] or "unnamed")


def load_ledger(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Target list
# ---------------------------------------------------------------------------

def build_target_list(
    ledger: dict[str, Any],
    *,
    style_tail: str,
) -> list[dict[str, Any]]:
    """Walk the ledger and emit one target per FLUX render to do.

    Output schema per target:
      {
        kind: "portrait" | "composite",
        target_id: str,                # for dedup + filename
        speaker: str,
        char_id: str | None,
        shot_id: str | None,           # composites only
        scene_id: str | None,          # composites only
        positive_prompt: str,
        save_prefix: str,              # ComfyUI SaveImage prefix
        out_filename_glob: str,        # what we expect on disk after render
      }
    """
    cast = ledger.get("cast") or []
    shots = ledger.get("shots") or []
    scenes = ledger.get("scenes") or []
    cast_by_name = {(c.get("name") or "").strip(): c for c in cast}
    scene_by_id = {(s.get("scene_id") or ""): s for s in scenes}

    # Best-available scene description: prefer scenes[0] if there's only
    # one (silent_test typically has 1-3), else look up by scene_id on the
    # shot. The silent_test re-segmenter doesn't propagate scene_id, so
    # we frequently get null and fall back to the first scene.
    def _scene_description_for_shot(shot: dict) -> str:
        sid = shot.get("scene_id")
        sc = scene_by_id.get(sid) if sid else None
        if sc and (sc.get("description") or sc.get("env")):
            return (sc.get("description") or sc.get("env") or "").strip()
        if scenes:
            sc0 = scenes[0]
            return (sc0.get("description") or sc0.get("env") or "").strip()
        return ""

    targets: list[dict[str, Any]] = []

    # ----- portraits -----
    # Special handling for the synthetic RADIO speaker: it gets a portrait
    # with a fixed vintage-radio prompt, and we DO NOT emit per-(shot,
    # speaker) composites for RADIO -- the same one radio still is reused
    # for every atmospheric clip (per ROADMAP Goal 3 lip-synching radio
    # design: one radio identity, audio-driven motion variation via HuMo).
    for c in cast:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        desc = (c.get("description") or "").strip()
        is_radio = name.upper() == "RADIO"
        if is_radio:
            # Front-and-center grille framing -- the grille IS the mouth
            # HuMo will lip-sync to whatever audio plays during atmospheric
            # beats (klaxons, music, hum, silence).
            positive = (
                (desc or "vintage 1940s console radio")
                + ", front-on view with the speaker grille filling the "
                + "center of the frame as the focal element, "
                + style_tail
            )
        else:
            positive = (
                (desc or f"a portrait of {name}")
                + ", clean head and shoulders portrait, neutral pose, "
                + "even cinematic lighting, " + style_tail
            )
        slug = slugify(name)
        # 2026-04-26: outputs nested under output/otr/ — single OTR
        # parent dir keeps ComfyUI's output/ root tidy. Stills go to
        # otr/stills/, videos to otr/videos/, deliverables to
        # otr/episodes/.
        prefix = f"otr/stills/pass1_portrait_{slug}"
        targets.append({
            "kind": "portrait",
            "target_id": f"portrait_{slug}",
            "speaker": name,
            "char_id": c.get("char_id"),
            "shot_id": None,
            "scene_id": None,
            "positive_prompt": positive,
            "save_prefix": prefix,
            "out_filename_glob": f"{prefix}_*.png",
        })

    # ----- (shot, speaker) composites -----
    # RADIO is intentionally skipped: it reuses the single portrait for
    # every atmospheric beat in the episode (cinematic continuity --
    # "the radio" is one recurring object, not a per-shot variant).
    seen_pairs: set[tuple[str, str]] = set()
    for shot in shots:
        shot_id = shot.get("shot_id") or ""
        scene_id = shot.get("scene_id")
        scene_desc = _scene_description_for_shot(shot)
        for beat in shot.get("beats") or []:
            speaker = (beat.get("speaker") or "").strip()
            if not speaker:
                continue
            if speaker.upper() == "RADIO":
                # No composite — atmospheric beats reuse the radio portrait.
                continue
            key = (shot_id, speaker)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            cast_row = cast_by_name.get(speaker) or {}
            speaker_desc = (cast_row.get("description") or speaker).strip()
            char_id = cast_row.get("char_id")
            speaker_slug = slugify(speaker)
            shot_slug = slugify(shot_id, limit=24)
            scene_part = (
                ", " + scene_desc if scene_desc else ""
            )
            positive = (
                f"{speaker_desc}{scene_part}, "
                + style_tail
            )
            prefix = f"otr/stills/pass3_{shot_slug}_{speaker_slug}"
            targets.append({
                "kind": "composite",
                "target_id": f"composite_{shot_slug}_{speaker_slug}",
                "speaker": speaker,
                "char_id": char_id,
                "shot_id": shot_id,
                "scene_id": scene_id,
                "positive_prompt": positive,
                "save_prefix": prefix,
                "out_filename_glob": f"{prefix}_*.png",
            })

    return targets


def filter_targets(
    targets: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    if scope == "all":
        return targets
    if scope == "portraits-only":
        return [t for t in targets if t["kind"] == "portrait"]
    if scope == "composites-only":
        return [t for t in targets if t["kind"] == "composite"]
    if scope.startswith("custom:"):
        # custom:shot=shot_011  /  custom:speaker=ANN  /  custom:kind=portrait
        body = scope.split(":", 1)[1]
        out = list(targets)
        for clause in body.split(","):
            clause = clause.strip()
            if "=" not in clause:
                continue
            field, val = clause.split("=", 1)
            field = field.strip()
            val = val.strip()
            out = [t for t in out if str(t.get(field) or "") == val]
        return out
    raise ValueError(f"unknown --scope value: {scope!r}")


# ---------------------------------------------------------------------------
# FLUX prompt builder
# ---------------------------------------------------------------------------

def build_bundled_prompt(
    *,
    checkpoint: str,
    targets: list[dict[str, Any]],
    negative: str,
    seeds: list[int],
    width: int,
    height: int,
    steps: int,
    cfg: float,
) -> dict[str, Any]:
    """ComfyUI API-format graph that renders N FLUX targets in ONE submission.

    Shape:
        ckpt        ── shared CheckpointLoaderSimple (one model load)
        neg_shared  ── shared CLIPTextEncode (one negative encode)
        per target i:
            pos_i    ── CLIPTextEncode (positive)
            latent_i ── EmptyLatentImage
            ksamp_i  ── KSampler (reads ckpt model, pos_i, neg_shared, latent_i)
            decode_i ── VAEDecode
            save_i   ── SaveImage with target's own filename_prefix

    Net effect: ComfyUI loads FLUX once, then runs N KSampler steps in series
    inside a single queue slot. Per-target save_prefix is preserved so the
    verifier still rounds-trips against the ledger.
    """
    graph: dict[str, Any] = {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
            "_meta": {"title": "Load FLUX checkpoint (shared)"},
        },
        "neg_shared": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["ckpt", 1]},
            "_meta": {"title": "Negative (shared)"},
        },
    }
    for i, t in enumerate(targets):
        seed = seeds[i]
        graph[f"pos_{i}"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": t["positive_prompt"], "clip": ["ckpt", 1]},
            "_meta": {"title": f"Positive {t['target_id']}"},
        }
        graph[f"latent_{i}"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        graph[f"ksamp_{i}"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["ckpt", 0],
                "positive": [f"pos_{i}", 0],
                "negative": ["neg_shared", 0],
                "latent_image": [f"latent_{i}", 0],
            },
        }
        graph[f"decode_{i}"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [f"ksamp_{i}", 0], "vae": ["ckpt", 2]},
        }
        graph[f"save_{i}"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": t["save_prefix"],
                "images": [f"decode_{i}", 0],
            },
            "_meta": {"title": f"Save {t['target_id']}"},
        }
    return graph


def build_flux_prompt(
    *,
    checkpoint: str,
    positive: str,
    negative: str,
    save_prefix: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
) -> dict[str, Any]:
    """ComfyUI API-format FLUX prompt graph.

    Mirrors the FLUX section of workflows/otr_videoplan_TEST_humo.json
    (CheckpointLoaderSimple → CLIPTextEncode pos/neg → EmptyLatentImage
    → KSampler → VAEDecode → SaveImage). Single-image batch.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
            "_meta": {"title": "Load FLUX checkpoint"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
            "_meta": {"title": "Positive"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
            "_meta": {"title": "Negative"},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": save_prefix,
                "images": ["6", 0],
            },
        },
    }


# ---------------------------------------------------------------------------
# ComfyUI HTTP client (stdlib only)
# ---------------------------------------------------------------------------

class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def submit_prompt(self, api_prompt: dict) -> str:
        body = {"prompt": api_prompt, "client_id": self.client_id}
        out = self._post_json("/prompt", body)
        if "prompt_id" not in out:
            raise RuntimeError(f"ComfyUI rejected prompt: {out}")
        return out["prompt_id"]

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                hist = self._get_json(f"/history/{prompt_id}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    time.sleep(poll_interval_s)
                    continue
                raise
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                return entry
            time.sleep(poll_interval_s)
        raise TimeoutError(f"prompt {prompt_id} did not complete in {timeout_s}s")

    def get_queue_state(self) -> tuple[int, int]:
        q = self._get_json("/queue")
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _existing_output(out_dir: Path, target: dict) -> list[Path]:
    return list(out_dir.glob(target["out_filename_glob"]))


def main() -> int:
    args = parse_args()

    if not args.ledger.exists():
        print(f"FATAL: ledger not found: {args.ledger}", file=sys.stderr)
        return 2

    led = load_ledger(args.ledger)
    targets = build_target_list(led, style_tail=args.style_tail)
    targets = filter_targets(targets, args.scope)

    if args.skip_existing:
        before = len(targets)
        targets = [t for t in targets if not _existing_output(args.out_dir, t)]
        skipped = before - len(targets)
        if skipped:
            print(f"[skip-existing] {skipped} target(s) already on disk")

    if args.max_targets:
        targets = targets[: args.max_targets]

    if not targets:
        print("FATAL: empty target list (after filters)", file=sys.stderr)
        return 3

    n_portrait = sum(1 for t in targets if t["kind"] == "portrait")
    n_composite = sum(1 for t in targets if t["kind"] == "composite")
    print(
        f"[plan] {len(targets)} FLUX target(s): "
        f"{n_portrait} portrait + {n_composite} (shot, speaker) composite"
    )
    for t in targets:
        kind_tag = "PORT " if t["kind"] == "portrait" else "COMP "
        ctx = (t["shot_id"] + " / " if t["shot_id"] else "") + t["speaker"]
        print(f"  {kind_tag}{t['target_id']:<48}  {ctx}")

    if args.dry_run:
        print("[dry-run] not submitting any prompts.")
        return 0

    client = ComfyClient(args.comfyui_url)
    try:
        running, pending = client.get_queue_state()
    except Exception as e:
        print(f"FATAL: cannot reach ComfyUI at {args.comfyui_url}: {e}",
              file=sys.stderr)
        return 4
    print(f"[comfy] queue running={running} pending={pending}")

    started = time.time()
    completed = 0

    # ------------------------------------------------------------------
    # BUNDLED mode: ONE graph, N parallel chains, one queue slot.
    # FLUX loads once, every KSampler runs in series with the model
    # resident, no per-prompt /prompt API overhead.
    # ------------------------------------------------------------------
    if args.mode == "bundled":
        seeds = [((idx * 1009 + 7) & 0x7FFFFFFFFFFFFFFF)
                 for idx in range(1, len(targets) + 1)]
        api_prompt = build_bundled_prompt(
            checkpoint=args.checkpoint,
            targets=targets,
            negative=args.negative_prompt,
            seeds=seeds,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
        )
        print(f"\n[bundled] submitting one graph with {len(targets)} chain(s)")
        try:
            prompt_id = client.submit_prompt(api_prompt)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"    SUBMIT FAILED: {e}", file=sys.stderr)
            if body:
                print(f"    response_body: {body[:1500]}", file=sys.stderr)
            return 5
        except Exception as e:
            print(f"    SUBMIT FAILED: {e}", file=sys.stderr)
            return 5
        print(f"    prompt_id={prompt_id}")
        # One bundled prompt may take MANY minutes — extend the timeout to
        # cover N renders at warm rate plus cold load slack.
        bundled_timeout = max(args.prompt_timeout,
                              90 * len(targets) + 240)
        try:
            client.wait_for_completion(
                prompt_id,
                timeout_s=bundled_timeout,
                poll_interval_s=max(args.poll_interval, 5.0),
            )
        except TimeoutError as e:
            print(f"    TIMEOUT: {e}", file=sys.stderr)
            return 6
        # Verify each target landed.
        for t in targets:
            landed = _existing_output(args.out_dir, t)
            if landed:
                completed += 1
            else:
                print(f"    WARN: no PNG for {t['target_id']} matching "
                      f"{t['out_filename_glob']}", file=sys.stderr)
        total = time.time() - started
        avg = total / max(1, completed)
        print(f"\n[done] {completed}/{len(targets)} renders in {total:.1f}s "
              f"({avg:.1f}s avg)")
        return 0 if completed == len(targets) else 7

    # ------------------------------------------------------------------
    # SERIAL mode: legacy one-prompt-per-target loop.
    # ------------------------------------------------------------------
    for idx, t in enumerate(targets, 1):
        elapsed_per_clip_started = time.time()
        print(f"\n[{idx}/{len(targets)}] {t['kind']:<10} {t['target_id']}")
        # Deterministic seed per target so reruns are reproducible.
        seed = (idx * 1009 + 7) & 0x7FFFFFFFFFFFFFFF
        api_prompt = build_flux_prompt(
            checkpoint=args.checkpoint,
            positive=t["positive_prompt"],
            negative=args.negative_prompt,
            save_prefix=t["save_prefix"],
            seed=seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
        )
        try:
            prompt_id = client.submit_prompt(api_prompt)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"    SUBMIT FAILED: {e}", file=sys.stderr)
            if body:
                print(f"    response_body: {body[:1500]}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"    SUBMIT FAILED: {e}", file=sys.stderr)
            continue
        print(f"    prompt_id={prompt_id}")
        try:
            client.wait_for_completion(
                prompt_id,
                timeout_s=args.prompt_timeout,
                poll_interval_s=args.poll_interval,
            )
        except TimeoutError as e:
            print(f"    TIMEOUT: {e}", file=sys.stderr)
            continue
        wall = time.time() - elapsed_per_clip_started
        landed = _existing_output(args.out_dir, t)
        if landed:
            newest = sorted(landed, key=lambda p: p.stat().st_mtime)[-1]
            print(f"    DONE in {wall:.1f}s -> {newest.name}")
            completed += 1
        else:
            print(f"    WARN: completed but no PNG found in {args.out_dir} "
                  f"matching {t['out_filename_glob']}", file=sys.stderr)

    total = time.time() - started
    print(
        f"\n[done] {completed}/{len(targets)} renders in {total:.1f}s "
        f"({total / max(1, completed):.1f}s avg)"
    )
    return 0 if completed == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
