# Pool-loop stills -- judgment + plan (Claude as judge)

Panel: gemini-pro-latest + gpt-5.1 (direct APIs). Both CONVERGED. Claude grounded
every load-bearing claim against the real code. This is the build plan.

## Convergent design (grounded)
The pooling/looping lives in **ShotLock** (which ALREADY computes the pool budget);
the still phase only emits N generic pool stills; render_driver reads a stamped key.
Three small layers, deterministic, no new flag.

1. **Still phase** (`derive_scene_still_targets` + `derive_image_prompts` + the
   prompt-gen node + ImageDirector forwarding). Thread `other_beats={clip_mode,
   pool_n}` from the video policy into the still phase (the same image_policy_json
   path that already carries still 'aspects'). Behaviour:
   - announcer / music / **character_video** beats -> per-beat scene still (UNCHANGED;
     character_video is a CHARACTER_BEARING_ROLE -- do NOT pool it [gemini MUST#3]).
   - the OTHER-BEATS roles = **{background_abstract, scene_broll} ONLY**:
     - `unique_per_beat` -> per-beat (as now).
     - `pool_n_loop` -> emit EXACTLY N targets with GENERIC pool ids
       `other_pool_0..N-1` (prompts from the first N other-beats lines, sequential =
       deterministic). pool_n<=0 -> 0 + LOUD WARN; pool_n>M -> M (clamp, no
       over-gen) [gpt #3].
   - visualizer / ltx_av_music stay EXCLUDED regardless of clip_mode -- the
     `accepts_still=False` dispatcher gate already drops them [gpt SHOULD#1].
2. **ShotLock** (`_audio_derived_clip_budget` / the shot-row build). For the
   other-beats shots in `pool_n_loop`, stamp `shot["still_pool_key"] =
   "other_pool_%d" % (i % pool_n)` over the SAME other-beats iteration the clip
   budget uses (so the still a beat reuses == the clip it reuses; determinism +
   single source of the loop math [gemini #2, gpt #2]).
3. **render_driver**. In the scene-still branch, prefer `shot["still_pool_key"]`
   when present (`_still_index` keyed by the pool id), else the per-beat `beat_id`
   (UNCHANGED for unique_per_beat). One `if`, render_driver stays dumb [gpt CUT#2].

## Grounded corrections (do NOT do these)
- **Keep generating scene stills for ALL accepts_still roles, incl. HuMo.** Do NOT
  add a `consumes_scene_still` flag and do NOT skip HuMo stills [gpt CUT#1]. Grounded:
  `EXPECTED_OOM_TRAIL = [..., "humo_1.7B->still_kenburns (oom)"]` and `still_kenburns`
  is `static_motion` = a SCENE-still consumer (`_SCENE_INIT_FAMILIES`). If HuMo OOMs
  to the floor and the scene still is absent, the floor mis-renders -> "never aborts"
  broken. The per-beat HuMo scene still is fallback INSURANCE, not waste [gemini #1].
- **Do NOT pool character_video** (keep per-beat) -- character continuity/sync.
- **other-beats = {background_abstract, scene_broll}** exactly; announcer/music never
  pooled even if their engine is abstract/static [gpt #6 -- assert in tests].

## Defaults [gemini SHOULD#2, gpt #7]
Set BOTH the `OTR_VideoDirector` widget defaults (code: `other_beats_clip_mode`
default -> `pool_n_loop`, `other_beats_n` default -> `4`) AND the saved
`otr_scifi_16gb_full.json` node-88 widgets, so widget<->JSON never desync.
Re-validate the JSON (OTR_WorkflowValidator + round-trip). NOTE: the default flip is
COUPLED to the pool-still implementation -- do not ship the pool_n_loop default until
steps 1-3 land, or the still phase over-generates per-beat stills the pooled clips
won't use.

## Invariants / tests
Determinism (beat i -> still i mod N over the canonical other-beats order ==
ShotLock's); no silent fallback (LOUD on pool_n<=0 / any miss); accepts_still stays
the central gate; workflow JSON source-of-truth + re-validate; UTF-8 no BOM. Tests:
(a) pool_n_loop emits exactly N pool targets, ids other_pool_0..N-1; (b)
unique_per_beat unchanged (per-beat); (c) announcer/music/character never pooled;
(d) ShotLock stamps i mod N over the same iteration; (e) render_driver resolves the
pool key; (f) pool_n<=0 -> 0 + WARN, pool_n>M -> M.
