# Reference Episode Fixture — "Satellites Collide"

Hand-built from the 2026-04-23 23:04:18 full-pipeline run. Used as frozen input
for TEST iterations (FLUX.2 swap, HuMo integration, ffmpeg mux) so we don't
have to re-run Mistral ScriptWriter + Director every time we iterate on video.

## Provenance

- **Run timestamp:** 2026-04-23 23:04:18
- **Source log:** full pipeline executed `workflows/otr_scifi_16gb_full.json`
  (pre-dead-sidecar-removal version). Audio pipeline + POC ffmpeg video
  completed cleanly. Sidecar FLUX errored out on all 11 shots (BUG-046 family)
  — irrelevant for this fixture, which only needs the script + director plan.
- **Treatment file:** `C:/Users/jeffr/Documents/ComfyUI/output/old_time_radio/signal_lost_object_appears_radar_20260423_230418_treatment.txt`
- **POC video:** `C:/Users/jeffr/Documents/ComfyUI/output/old_time_radio/signal_lost_object_appears_radar_20260423_230418.mp4`
  (483.3 MB, 533.4s, 12803 frames @ 24fps, 1920x1080)
- **Audio duration:** 461.2s assembled, 7.7 min final with music crossfades

## Cleanup applied (vs. raw log output)

The raw LLM Director output had two problems addressed in this fixture:

1. **Phantom cast bleed-through** — the critique/revise pass pasted in an
   unrelated space-station scene (CAPTAIN JOHNSON / ENSIGN PARKER / CONTROL).
   The Director assigned voices to them out of habit. For this fixture those
   three characters and their scene are **removed**. Real cast only:
   DUANE VOSS, PARRY MARTIN, ALAN SIRIKIT, REGINALD HAYES, ANNOUNCER.

2. **Truncated visual_plan.scenes** — the Director's 1168-token budget cut off
   mid-portrait-prompt so `visual_plan.scenes` never emitted. Rebuilt by hand
   from the scene `[ENV:]` markers in the treatment. Post-2026-04-23 patch
   raises the token ceiling to 2500, so future runs should emit this naturally.

## Files

- `director_satellites_collide.json` — clean Director JSON with all three
  visual_plan scenes and five character portraits. Consumable directly by
  `OTR_VideoPlan` (PASS1/PASS2/PASS3).
- `script_satellites_collide.txt` — full 55-dialogue-line script with
  character/speaker tags and SFX cues. Lifted from the treatment.

## How to use

TEST workflows (`workflows/otr_videoplan_TEST.json` and successors) can load
this director JSON directly into `OTR_VideoPlan.director_json` widget via
copy-paste or via a future `OTR_LoadDirectorFixture` node. Skips the entire
Mistral → Bark → MusicGen → ffmpeg chain — straight to video-branch iteration.

When the full pipeline with `max_new_tokens=2500` runs again, the Director
should emit a JSON structurally identical to this fixture (minus phantom cast).
Use that run to validate the fix, then consider this fixture canonical.
