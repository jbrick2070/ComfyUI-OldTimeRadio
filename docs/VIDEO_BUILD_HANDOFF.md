# OTR Video Platform -- HANDOFF -- CAPSTONE NIGHT RUN (2026-06-10, overnight) -- obs-quality fixes SHIPPED, all 3 opt-in lanes LIVE, multi-model marathon RUNNING

> **CANONICAL LOCATION:** this in-repo file (`docs/VIDEO_BUILD_HANDOFF.md`) is the SINGLE git-tracked source of truth. The v1.4 execution plan + roundtable sources stay in `C:\Users\jeffr\Documents\otr-video-roundtable\`.

## ACTIVE MISSION (the only active build)
Finish the v1.4 2D capstone: the multi-model soak marathon is running overnight (started 2026-06-09 23:53, ~4.3 h budget) rotating the REAL dropdown surface -- all-LTX / LTX+latentsync combo / humo production / bark+musicgen / gemma writer / dia voice / floor / OpenRouter + Comfy-Credits cloud writers -- one fresh headless server per episode, HARD gates per episode. Morning job: read the marathon results, fix what broke LOUDLY, stamp (or honestly refuse to stamp) production-stable.

## HARD RULES (copy verbatim)
- Do NOT start / resume / "continue" any other sprint -- NOT story-spine, NOT story-pipeline, NOT any audio sprint, NOT any other ROADMAP item. They are PARKED.
- The audio refactor is SHIPPED; the audio script ledger is FROZEN (read-only). Per-beat audio only SLICES the frozen master read-only. byte-identical master + mux-LAST (`-c:a copy` on the ARCHIVAL final); `test_audio_byte_identical` stays GREEN at every step.
- Ignore any stale `session_handoff.md` and any memory / ROADMAP entry implying other "active" work.
- Invariants: single resident heavy engine <= 14.5 GB machine-NVML; BUG-291 detach reclaim (`reclaim_idle_models`, never `unload_all_models`); every in-render fallback LOUD (log swap + ledger restamp); V-12 isolation; UTF-8 no BOM; SFW.
- OUTPUT HYGIENE (operator directives 2026-06-09, all SHIPPED + gated in the soak driver): (1) the DELIVERABLE is a PLAYABLE mp4 in the operator's `otr\obs` -- video+audio streams, duration==master +-2s, clean full decode, **AAC-320k/48k viewing audio** (raw PCM-in-MP4 does not play in Windows -- "master in PCM, DELIVER in AAC"; the byte-identical PCM archival final stays in the episode folder); (2) obs holds ONLY finals; every other asset lives in `otr\episodes\<slug>\`; JSON run artifacts in `otr\state`; (3) NOTHING writes outside `output\otr\` (TEMP/TMP + GPU lease pinned under `output\otr\tmp` by the launcher); (4) even headless, ALL outputs land in the REAL `C:\Users\jeffr\Documents\ComfyUI\output` (launcher pins `--output-directory` + `OTR_OUTPUT_DIR` + `OTR_OBS_DIR`); (5) the deliverable must carry the procgen radio look + SDH captions (workflow node 93).
- Cloud lanes OK; spend < $20 fine (operator authorized cloud-writer soak legs 2026-06-09 night).
- RUN-IDENTITY: pin every result to ITS OWN prompt_id + episode slug; prove config from the live server log; grep-able logs, never Out-Null.
- UPDATE the otr-build-tracker file `C:\Users\jeffr\OneDrive\Documents\Claude\Artifacts\otr-build-tracker\index.html` (FABLE TESTER table) + report the row in the summary.

## WHERE WE ARE (2026-06-10 overnight; everything below from THIS session, commits pushed through `c81f0e6`)
**Quick test (30w, full fixed chain) PASSED all hard gates** (run1, prompt `4614793c`, ep `signal_lost_pulse_of_progress_20260609_212124`): humo:5, byte-identical PCM (`e77afa966375`), playable obs deliverable, zero stray writes, V-3 render VRAM 3.6 GB.

### Shipped tonight (commit -> what)
- `f05a081` OUTPUT HYGIENE hard gates + `scripts/_otr_soak_capstone.py` (the per-episode gate engine) + `scripts/_otr_single_engine_smoke.py` + the tracked launcher `scripts/_otr_soak_server_launch.cmd`.
- `76a4695` deliverable-content fixes: NEW workflow node 93 `OTR_PostUpscaleProcgenBlend` (84 SilentComposite -> 86 CaptionBurn(off) -> 93 blend screen/green-only/crush18 + burn_captions sdh_standard -> 85 mux; links 265/266, link 250 re-sourced); obs publish = AAC viewing copy; kokoro LOUD local-only voice guard + `cast_lookup` announcer role-tag alias (the run2 abort: line b001 carried char_id='announcer' and an indextts2 `vz_*` bank id reached kokoro -> mid-render HF 404).
- `b6c99b9` AAC publish pinned 48 kHz.
- `dfee242` `OTR_FORCE_ENGINE_MAP` (role=engine / *=engine, LOUD, fallback chains intact) + `_provide_lipsync_base` (`OTR_LSYNC_BASE_ENGINE` renders a SILENT face-forward base in-line for lipsync shots) + the marathon playlist runner `scripts/_otr_soak_marathon.py` (+8 tests).
- `1494af5` blend dim-conform fix (probe source dims -> explicit scale; the legacy `scale=-2:ih` was a self-referential no-op that hard-failed the filter at 1472x832 vs 1920x1080) + blend output joins the episode folder + cloud-writer legs.
- `c81f0e6` ALL THREE OPT-IN LANES LIVE on the 5080: **ltx_video** 49f/14.9s/3.8GB (prompt `d70e4b82`); **latentsync** cu128 sidecar 49f lipsync over an LTX base in 115.3s (prompt `de19db87` -- the combo seam works); **wan_i2v** 33f/206.7s on `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (prompt `af3b5457`). Plus both fail-closed NAMED proofs. Unblocking fixes: the BUG-070 gate now reads `comfy.model_management.sage_attention_enabled()` (ComfyUI core imports sageattention UNCONDITIONALLY as an availability probe -- module residency != active); `run_graph(free_after_use=True)` for ltx/wan (fp16 T5 ~9.5GB co-resident with the UNET breached the 14.5GB machine ceiling).
- Assets fetched tonight (NOT operator-blocked after all): Wan 2.2 i2v low-noise 14B fp8 -> `C:\ComfyUI-Models\diffusion_models\`; the latentsync Path-B sidecar fully installed (`C:\Users\jeffr\Documents\ComfyUI\latentsync\`, py3.10 venv, torch 2.11.0+cu128 CUDA True, unet + whisper ckpts; insightface via the Gourieff cp310 wheel).

### Gotchas learned (do not relearn)
- Windows "Documents" is OneDrive-redirected on this box; the repo lives at the REAL `C:\Users\jeffr\Documents\...` (Explorer "Date" column confusion, not file drift).
- The headless server resolves Path-B sidecar defaults under the INSTALL junction root -- always pin `OTR_LATENTSYNC_VENV`/`OTR_LATENTSYNC_REPO` (the marathon ep02 env does).
- The master WAV keeps its `pending_*` name inside the renamed episode dir (sometimes BOTH names exist) -- resolve by suffix, prefer the slug-named.
- `cmd /c start` from a detached parent never spawns the child -- the marathon Popens the launcher directly with CREATE_NO_WINDOW.
- An episode is ~35-45 min on the humo path at 30-60w (HuMo 14B ~45-100 s/it under load); LTX clips are ~15 s each.

### MARATHON RESULTS (final tally as of ~03:00; the 160w/4ch finale was still rendering -- check `marathon_20260610_025015/results.jsonl` for its row)
| leg | result | histogram / evidence |
|---|---|---|
| quick test 30w (pre-marathon) | **PASS** | humo:5; pcm `e77afa966375`; ep `signal_lost_pulse_of_progress_20260609_212124` |
| all-LTX 120w | **PASS** (10.5 min) | `{ltx_video:5}`; ep `signal_lost_record_of_the_leviathan_20260610_000210` |
| LTX-base + latentsync combo 120w | **PASS** | `{still_kenburns:4, latentsync:1}` -- one REAL lipsync beat over an LTX base; misses fell LOUDLY to the floor (face-detect per generated base is a lottery; future: OTR_LSYNC_BASE_ENGINE=still_kenburns over the cast portrait for reliable faces) |
| humo + bark + musicgen 60w | **PASS** | `{humo:5}`; bark + musicgen live in the logs |
| humo + gemma writer 80w | **PASS** | `{humo:5}` |
| floor 30w (FLOOR lane) | **PASS** | `{still_kenburns:5}` -- the procgen-only path renders the full deliverable (this IS the future low-VRAM mode's engine side) |
| dia voice | named FAIL-CLOSED | `Dia Path B not installed` -- its sidecar venv was never installed on this box; leg swapped out |
| OpenRouter claude writer | transport PASS / content FAIL-CLOSED | submit + remote calls worked; the generated script flunked the CastLock structural gate (`needs_full_rerun`) and the episode refused to render -- gates working as designed (cents spent) |
| Comfy-Credits writer | named FAIL-CLOSED | `ComfyCreditsConfigError: No Comfy credentials` -- headless server has no logged-in Comfy Desktop; expected |
| humo 160w/4ch finale | **RENDERED + DELIVERED** (gate misclass) | `{humo:8, visualizer:1}` -- 9 beats, all 8 TALKING beats real HuMo, the music beat correctly routed to the visualizer; obs deliverable `signal_lost_whispers_of_betrayal_20260610_030539_..._final.mp4` (18.7MB AAC) is in obs. The driver's strict `expect_engine="humo"` asserts ALL beats humo -- WRONG at episode sizes that produce music beats. GATE REFINEMENT (small, next session): assert "every TALKING beat == humo" instead of whole-histogram equality. |

### MARATHON (runner details)
- Runner: `scripts/_otr_soak_marathon.py --hours 4.3`, console -> `scripts/_otr_soak_capstone_results/marathon_console2.log`, per-run dir `scripts/_otr_soak_capstone_results/marathon_<stamp>/` (marathon.log + results.jsonl + per-leg JSON evidence + per-leg server logs).
- 9-leg repeating playlist: ep01 all-LTX 120w; ep02 LTX-base+latentsync combo 120w; ep03 humo+bark+musicgen 60w; ep04 humo+gemma 80w; ep05 humo+dia+gemma 60w; ep05b OpenRouter-claude writer 60w; ep05c Comfy-Credits writer 40w; ep06 floor 30w; ep07 humo 160w/4ch. Strict humo histogram on production legs; informational on experiment legs; ALL hard gates (playable AAC obs + byte-identical archival + hygiene + VRAM) every leg; errors log with tracebacks and the loop continues.

## FIRST ACTIONS for the next session (morning)
1. `git fetch` + `git log --oneline -10` + `git status` (expect `c81f0e6`+ on v2.0-alpha, HEAD==origin).
2. Read `scripts/_otr_soak_capstone_results/marathon_*/marathon.log` + `results.jsonl` + the leg JSONs. Count PASS/FAIL; for each FAIL read its server log. The all-LTX + combo legs are EXPERIMENTS (histogram informational) -- judge them on completion + obs quality, not engine purity.
3. Spot-check 2-3 obs mp4s actually play (video+audio+captions+procgen look).
4. Fix LOUD failures, re-run the affected leg, then stamp production-stable (tag) ONLY if the operator's obs-quality bar is met; update the tracker + this handoff.
5. PARKED follow-ups: 3D `character_3d` (separate window, `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`); MuseTalk adapter (NOT BUILT -- needs a latentsync-pattern sidecar sprint); per-role engine dropdown UI; the "procgen+captions only / bypass video+3D entirely" low-VRAM switch (operator ask 2026-06-09 -- engineer LATER, not now); RTXUpscale 1080p stage (left out of the new chain; revisit if the operator wants 1080p finals); chatterbox in-graph (env conflict, baselined).

## PARKED -- not now
- 3D `character_3d` reopen (the separate planning window owns it).
- Story-spine / story-pipeline / any audio sprint (SHIPPED + frozen). Any other ROADMAP item.
