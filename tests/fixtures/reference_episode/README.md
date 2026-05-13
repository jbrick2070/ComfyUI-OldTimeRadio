# Reference Episode Fixture — "Satellites Collide"

Hand-built from the 2026-04-23 23:04:18 full-pipeline run. Used as
frozen input for TEST iterations (FLUX.2 swap, HuMo integration,
ffmpeg mux) so we don't have to re-run the full Mistral writer chain
every time we iterate on video.

## Provenance

- **Run timestamp:** 2026-04-23 23:04:18
- **Source log:** full pipeline executed `workflows/otr_scifi_16gb_full.json`
  (pre-dead-sidecar-removal version). Audio pipeline + POC ffmpeg video
  completed cleanly. Sidecar FLUX errored out on all 11 shots (BUG-046 family)
  — irrelevant for this fixture, which only needs the script + the legacy
  Director plan that the run produced.
- **Treatment file:** `C:/Users/jeffr/Documents/ComfyUI/output/old_time_radio/signal_lost_object_appears_radar_20260423_230418_treatment.txt`
- **POC video:** `C:/Users/jeffr/Documents/ComfyUI/output/old_time_radio/signal_lost_object_appears_radar_20260423_230418.mp4`
  (483.3 MB, 533.4s, 12803 frames @ 24fps, 1920x1080)
- **Audio duration:** 461.2s assembled, 7.7 min final with music crossfades

## Cleanup applied (vs. raw log output)

The raw LLM run (back when the legacy LLMDirector still produced
the production plan -- the Director class itself was retired in
voice-path-cleanbreak S2 / commit 249bc06) had two problems
addressed in this fixture:

1. **Phantom cast bleed-through** — the critique/revise pass pasted in an
   unrelated space-station scene (CAPTAIN JOHNSON / ENSIGN PARKER / CONTROL).
   The legacy Director assigned voices to them out of habit. For this fixture
   those three characters and their scene are **removed**. Real cast only:
   DUANE VOSS, PARRY MARTIN, ALAN SIRIKIT, REGINALD HAYES, ANNOUNCER.

2. **Truncated visual_plan.scenes** — the legacy Director's 1168-token budget
   cut off mid-portrait-prompt so `visual_plan.scenes` never emitted. Rebuilt
   by hand from the scene `[ENV:]` markers in the treatment. Post-2026-04-23
   patch raised the token ceiling to 2500; the v2.0 LedgerScriptWriter emits
   `visual_plan` natively via the outline + cast-lock path.

## Files

- `director_satellites_collide.json` — clean legacy-Director JSON with all
  three visual_plan scenes and five character portraits. Retained as a
  fixture filename for back-compat with TEST workflow paths. Consumable
  directly by `OTR_VideoPlan` (PASS1/PASS2/PASS3).
- `script_satellites_collide.txt` — full 55-dialogue-line script with
  character/speaker tags and SFX cues. Lifted from the treatment.

## How to use

TEST workflows (`workflows/otr_videoplan_TEST.json` and successors) can load
this fixture's JSON directly into `OTR_VideoPlan.script_json` via copy-paste.
The widget input was renamed from `director_json` to `script_json` in
voice-path-cleanbreak S16.1; the fixture's JSON shape is back-compatible
with both names. Skips the entire writer → Bark → MusicGen → ffmpeg chain
— straight to video-branch iteration.

## Forward path (v2.0 ledger)

When the full v2.0 pipeline runs the LedgerScriptWriter for an equivalent
news seed, it emits a structurally-richer L3 ledger (`l3-2026-05-14`
schema) directly — no separate Director stage, no production_plan_json
intermediate. The fixture is preserved as historical reference for video-
branch test rigs; new fixtures should be generated from the L3 ledger
path going forward.
