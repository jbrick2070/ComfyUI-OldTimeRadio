"""
verify_smoke_artifacts.py  --  Downstream verification of a smoke-run episode
============================================================================

Runs after a `30 words (smoke, 1 act)` ULTRA_SMOKE soak to verify the full
asset chain landed correctly. Designed to be the SMALLEST possible
end-to-end check that exercises every node touched by the BUG-LOCAL-030
wave (Phase A composite + Phase B procgen blend + audit-completion ledger
metadata + long-form hardening).

Usage:
    cd C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\verify_smoke_artifacts.py

Optionally pass --episode-id <slug> to target a specific run; default is
the most recent per-episode workspace under output/otr/episodes/.

Exits 0 on full pass, 1 on any FAIL. WARN entries do NOT fail the script
(by design; they flag suspicious-but-not-broken state).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_OTR = ROOT.parent.parent / "output" / "otr"
EPISODES_DIR = OUTPUT_OTR / "episodes"


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

class Report:
    """Tracks PASS / WARN / FAIL counts + a list of human-readable lines."""

    def __init__(self):
        self.lines: list[str] = []
        self.passes = 0
        self.warns = 0
        self.fails = 0

    def passed(self, msg: str):
        self.passes += 1
        self.lines.append(f"  [PASS] {msg}")

    def warn(self, msg: str):
        self.warns += 1
        self.lines.append(f"  [WARN] {msg}")

    def fail(self, msg: str):
        self.fails += 1
        self.lines.append(f"  [FAIL] {msg}")

    def section(self, title: str):
        self.lines.append("")
        self.lines.append(f"== {title} ==")

    def render(self) -> str:
        out = list(self.lines)
        out.append("")
        out.append(f"== Summary ==")
        out.append(f"  PASS={self.passes}  WARN={self.warns}  FAIL={self.fails}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def find_episode_dir(episode_id: str | None) -> Path | None:
    if not EPISODES_DIR.exists():
        return None
    if episode_id:
        target = EPISODES_DIR / episode_id
        return target if target.exists() else None
    # Pick most recent by mtime.
    candidates = [p for p in EPISODES_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------

def ffprobe_dims(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of a video file via ffprobe, or None."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(path),
        ], stderr=subprocess.PIPE).decode("utf-8").strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_workspace_layout(ep_dir: Path, r: Report) -> None:
    r.section("Per-episode workspace layout (BUG-LOCAL-028)")
    expected_subdirs = ["audio", "stills", "portraits", "videos", "composited"]
    for name in expected_subdirs:
        d = ep_dir / name
        if d.exists() and d.is_dir():
            n = len(list(d.iterdir()))
            r.passed(f"{name}/ exists ({n} entries)")
        else:
            r.fail(f"{name}/ MISSING")

    # Per-clip-mux segment dir should have been cleaned by the long-form
    # hardening unlink loop. If pillarbox intermediates survive, that's
    # a WARN not a FAIL because spacesaver may catch them later.
    seg_dir = ep_dir / "composited" / "_per_clip_mux_segments"
    if seg_dir.exists():
        survivors = [p for p in seg_dir.glob("pb_*.mp4")]
        if survivors:
            r.warn(
                f"_per_clip_mux_segments/ has {len(survivors)} pillarbox "
                f"intermediate(s) NOT cleaned (BUG-LOCAL-030 longform "
                f"hardening expected unlink after concat success)"
            )
        else:
            r.passed(
                f"_per_clip_mux_segments/ has 0 pb_*.mp4 (intermediate "
                f"cleanup ran)"
            )


def check_ledger_schema(ep_dir: Path, r: Report) -> dict | None:
    r.section("Ledger schema (l3-2026-05-02)")
    audio_dir = ep_dir / "audio"
    candidates = list(audio_dir.glob("*_ledger.json"))
    if not candidates:
        r.fail(f"no ledger JSON found under {audio_dir}")
        return None
    led_path = candidates[0]
    try:
        led = json.loads(led_path.read_text(encoding="utf-8"))
    except Exception as exc:
        r.fail(f"ledger {led_path.name} unreadable: {exc}")
        return None

    schema = led.get("schema_version", "")
    if schema == "l3-2026-05-02":
        r.passed(f"schema_version = {schema}")
    else:
        r.warn(f"schema_version = {schema!r} (expected l3-2026-05-02)")

    meta = led.get("meta") or {}
    paths = meta.get("paths") or {}
    if paths.get("layout") == "per-episode-workspace":
        r.passed(f"meta.paths.layout = per-episode-workspace")
    else:
        r.fail(
            f"meta.paths.layout = {paths.get('layout')!r} "
            f"(expected per-episode-workspace)"
        )

    return led


def check_per_line_audio_metadata(led: dict, r: Report) -> None:
    r.section("Per-line audio metadata (BUG-LOCAL-030 audit-completion)")
    lines = led.get("lines") or []
    if not lines:
        r.fail("ledger.lines[] is empty")
        return

    engines = {"bark": 0, "kokoro": 0, "other": 0, "missing": 0}
    voice_preset_count = 0
    render_ms_count = 0
    audio_hash_count = 0
    generated_dur_count = 0

    for ln in lines:
        eng = ln.get("tts_engine")
        if eng == "bark":
            engines["bark"] += 1
        elif eng == "kokoro":
            engines["kokoro"] += 1
        elif eng:
            engines["other"] += 1
        else:
            engines["missing"] += 1
        if ln.get("voice_preset"):
            voice_preset_count += 1
        if ln.get("render_ms"):
            render_ms_count += 1
        if ln.get("audio_sample_hash"):
            audio_hash_count += 1
        if ln.get("generated_dur_s"):
            generated_dur_count += 1

    total = len(lines)
    if engines["missing"] == 0:
        r.passed(
            f"every line ({total}) has tts_engine "
            f"(bark={engines['bark']}, kokoro={engines['kokoro']})"
        )
    else:
        r.fail(
            f"{engines['missing']}/{total} line(s) missing tts_engine "
            f"-- BatchBark or KokoroAnnouncer write-back failed"
        )

    if voice_preset_count == total:
        r.passed(f"every line ({total}) has voice_preset")
    elif voice_preset_count > 0:
        r.warn(
            f"only {voice_preset_count}/{total} line(s) have voice_preset"
        )
    else:
        r.fail(f"NO line(s) have voice_preset")

    if render_ms_count >= int(total * 0.9):
        r.passed(f"{render_ms_count}/{total} line(s) have render_ms")
    else:
        r.warn(f"only {render_ms_count}/{total} line(s) have render_ms")

    if audio_hash_count == total:
        r.passed(f"every line ({total}) has audio_sample_hash (8-char SHA8)")
    elif audio_hash_count > 0:
        r.warn(
            f"only {audio_hash_count}/{total} line(s) have audio_sample_hash"
        )
    else:
        r.fail(f"NO line(s) have audio_sample_hash")

    if generated_dur_count >= int(total * 0.9):
        r.passed(f"{generated_dur_count}/{total} line(s) have generated_dur_s")
    else:
        r.warn(
            f"only {generated_dur_count}/{total} line(s) have generated_dur_s"
        )


def check_music_sfx_metadata(led: dict, r: Report) -> None:
    r.section("Music + SFX metadata (BUG-LOCAL-030 audit-completion)")
    music = led.get("music") or []
    sfx = led.get("sfx") or []

    music_with_engine = sum(
        1 for m in music if m.get("tts_engine") == "musicgen"
    )
    music_with_hash = sum(1 for m in music if m.get("audio_sample_hash"))
    music_with_render_ms = sum(1 for m in music if m.get("render_ms"))

    if music:
        if music_with_engine == len(music):
            r.passed(
                f"every music cue ({len(music)}) has tts_engine=musicgen"
            )
        else:
            r.fail(
                f"only {music_with_engine}/{len(music)} music cue(s) have "
                f"tts_engine=musicgen"
            )
        if music_with_hash == len(music):
            r.passed(
                f"every music cue ({len(music)}) has audio_sample_hash"
            )
        else:
            r.warn(
                f"only {music_with_hash}/{len(music)} music cue(s) have "
                f"audio_sample_hash"
            )
        if music_with_render_ms >= int(len(music) * 0.9):
            r.passed(
                f"{music_with_render_ms}/{len(music)} music cue(s) "
                f"have render_ms"
            )
        else:
            r.warn(
                f"only {music_with_render_ms}/{len(music)} music cue(s) "
                f"have render_ms"
            )
    else:
        r.warn("no music cues in ledger")

    sfx_with_engine = sum(1 for s in sfx if s.get("tts_engine") == "audiogen")
    sfx_with_hash = sum(1 for s in sfx if s.get("audio_sample_hash"))
    sfx_with_render_ms = sum(1 for s in sfx if s.get("render_ms"))

    if sfx:
        if sfx_with_engine == len(sfx):
            r.passed(f"every sfx ({len(sfx)}) has tts_engine=audiogen")
        else:
            r.fail(
                f"only {sfx_with_engine}/{len(sfx)} sfx have tts_engine=audiogen"
            )
        if sfx_with_hash == len(sfx):
            r.passed(f"every sfx ({len(sfx)}) has audio_sample_hash")
        else:
            r.warn(
                f"only {sfx_with_hash}/{len(sfx)} sfx have audio_sample_hash"
            )
        if sfx_with_render_ms >= int(len(sfx) * 0.9):
            r.passed(f"{sfx_with_render_ms}/{len(sfx)} sfx have render_ms")
        else:
            r.warn(
                f"only {sfx_with_render_ms}/{len(sfx)} sfx have render_ms"
            )
    else:
        r.warn("no sfx in ledger")


def check_video_pipeline(led: dict, ep_dir: Path, r: Report) -> None:
    r.section("Video pipeline (BUG-LOCAL-030 Phase A + B)")

    # Per-line HuMo clips at native 480x832 portrait
    videos_dir = ep_dir / "videos"
    if videos_dir.exists():
        humo_clips = sorted(videos_dir.glob("l*.mp4"))
        if humo_clips:
            sample = humo_clips[0]
            dims = ffprobe_dims(sample)
            if dims is None:
                r.warn(f"ffprobe unavailable; skipping HuMo dim check")
            elif dims == (480, 832):
                r.passed(
                    f"HuMo per-line clips at native 480x832 portrait "
                    f"({len(humo_clips)} clip(s); sampled {sample.name})"
                )
            elif dims == (832, 480):
                # LTX clip — non-character lines render via LTX
                r.passed(
                    f"sampled {sample.name} is 832x480 (LTX broadcast unit "
                    f"-- non-character line)"
                )
            else:
                r.warn(
                    f"sampled {sample.name} is {dims[0]}x{dims[1]} "
                    f"(expected 480x832 HuMo or 832x480 LTX)"
                )
        else:
            r.warn("no per-line clips found in videos/")

    # Final composited mp4 at 1472x832
    composited_dir = ep_dir / "composited"
    if composited_dir.exists():
        comps = sorted(composited_dir.glob("*.mp4"))
        if comps:
            for c in comps:
                dims = ffprobe_dims(c)
                if dims == (1472, 832):
                    r.passed(
                        f"composited {c.name} = 1472x832 (Phase A canvas)"
                    )
                elif dims == (1920, 1080):
                    r.passed(
                        f"composited {c.name} = 1920x1080 (post-RTXUpscale)"
                    )
                elif dims is None:
                    r.warn(f"ffprobe unavailable for {c.name}")
                else:
                    r.warn(
                        f"composited {c.name} = {dims[0]}x{dims[1]} "
                        f"(expected 1472x832 or 1920x1080)"
                    )
        else:
            r.warn("no mp4 in composited/")


def check_final_video_path(led: dict, ep_dir: Path, r: Report) -> None:
    r.section("final_video_path stamp (BUG-LOCAL-030 audit-completion)")
    fvp = led.get("final_video_path")
    if not fvp:
        r.fail("ledger.final_video_path is unset")
        return
    p = Path(fvp)
    if not p.exists():
        r.fail(f"final_video_path points to missing file: {fvp}")
        return
    r.passed(f"final_video_path exists: {p.name} ({p.stat().st_size:,} bytes)")

    if "_procgen_blended" in p.name:
        r.passed("final_video_path is the post-upscale procgen-blended mp4")
    else:
        r.warn(
            f"final_video_path does not contain '_procgen_blended' "
            f"(maybe PostUpscaleProcgenBlend bypassed or fell back?)"
        )

    # ffprobe should report 1920x1080 since it's the post-upscale blend
    dims = ffprobe_dims(p)
    if dims == (1920, 1080):
        r.passed(f"final_video_path dims = 1920x1080")
    elif dims is None:
        r.warn(f"ffprobe unavailable for final_video_path")
    else:
        r.warn(
            f"final_video_path dims = {dims[0]}x{dims[1]} "
            f"(expected 1920x1080)"
        )

    meta = led.get("meta") or {}
    pub = meta.get("post_upscale_blend") or {}
    if pub:
        r.passed(
            f"meta.post_upscale_blend forensics block present "
            f"({len(pub)} field(s))"
        )
    else:
        r.warn("meta.post_upscale_blend is missing")


def check_flux_outputs(ep_dir: Path, r: Report) -> None:
    r.section("FLUX outputs (BUG-LOCAL-028)")
    stills = ep_dir / "stills"
    portraits = ep_dir / "portraits"

    if stills.exists():
        env_pngs = sorted(stills.glob("full_env_*.png"))
        radio = sorted(stills.glob("radio_bookend_*.png"))
        if env_pngs:
            r.passed(
                f"env stills landed in per-episode stills/ "
                f"({len(env_pngs)} png)"
            )
        else:
            r.warn(f"no full_env_*.png in stills/")
        if radio:
            r.passed(f"radio bookend in stills/: {radio[0].name}")
        else:
            r.warn("no radio_bookend_*.png in stills/")

    if portraits.exists():
        port_pngs = sorted(portraits.glob("*.png"))
        if port_pngs:
            r.passed(f"portraits landed in per-episode portraits/ ({len(port_pngs)} png)")
        else:
            r.warn(f"portraits/ dir exists but is empty")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episode-id", type=str, default=None,
                   help="Specific episode slug (default: most recent)")
    args = p.parse_args()

    print("== verify_smoke_artifacts.py == BUG-LOCAL-030 wave verification")
    print()
    ep_dir = find_episode_dir(args.episode_id)
    if ep_dir is None:
        print(
            f"FAIL: no episode workspace found under {EPISODES_DIR}. "
            f"Did the smoke run produce output/otr/episodes/<ep>/?"
        )
        return 1
    print(f"Target episode: {ep_dir.name}")
    print(f"Path: {ep_dir}")

    r = Report()
    check_workspace_layout(ep_dir, r)
    led = check_ledger_schema(ep_dir, r)
    if led is not None:
        check_per_line_audio_metadata(led, r)
        check_music_sfx_metadata(led, r)
        check_video_pipeline(led, ep_dir, r)
        check_final_video_path(led, ep_dir, r)
    check_flux_outputs(ep_dir, r)

    print(r.render())
    return 0 if r.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
