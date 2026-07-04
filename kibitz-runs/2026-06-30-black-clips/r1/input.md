# BLACK CLIPS -- ROOT-CAUSE HUNT (for kibitz)

> Operator 2026-06-30: the combo finals are mostly BLACK with only the procgen CRT overlay; "it worked
> ~48 hours ago, before we removed fallbacks." Find the regression. The agents crawl the REAL code +
> the suspect commit diffs; Claude grounds + judges. DIAGNOSIS hunt (find the broken link) -- the FIX
> follows after convergence.

## SYMPTOM (confirmed by frame extraction)
- A recent combo final (`signal_lost_the_flickering_glow_...`, the overnight lumina_image leg) at a
  DRAMA beat (t=20s, "OYA PETROV" dialogue) shows BLACK + the procgen overlay (SIGNAL LOST header /
  timecode / scope bars / caption). The opening procgen title card renders fine.
- The per-beat CLIP file itself is BLACK: `episodes/pending_20260630_054646/clips/
  shot_b003_character_video_still_pan.mp4` extracts to the dark CRT floor color (~0x0A0E14), NOT an
  image. So the VIDEO is generated + muxed -- its CONTENT is the engine's black FLOOR.
- That episode's dir has 0 PNGs (no materialized stills), yet image GENERATION works globally:
  `episodes/*/stills/` has 3425 stills (223 in the last 14h, real ~1 MB images) + `_shared/cache` 2997.

## MECHANISM (grounded)
`build_request_from_shot` (render_driver.py ~928-958): for `still_pan`/`still_flat`/`ltx_audio_in` it
sets `init_image = _still_index(ledger).get(still_pool_key or beat_id, "")`; if EMPTY it sets
`init_source="missing_scene_still"` and the cheap family (still_pan/still_flat) "synthesizes its dark
floor" -> BLACK. `_still_index` (render_driver.py ~414) reads `ledger['images']['images']` rows with
`kind` starting `scene_`. So the break is: the minted scene still is NOT in the ledger's
`images.images` under the `still_pool_key`/`beat_id` the engine looks up -> init_image empty -> dark
floor. (still_pan is an ffmpeg pan over a PROVIDED still; it does NOT self-generate.)

## REGRESSION WINDOW (~48h; "before we removed fallbacks") -- SUSPECTS to diff
1. **`c1132196` "rename cheap still engines: flat_still->still_flat, flux_still->still_pan"** -- PRIME
   SUSPECT. Touches render_driver.py (26 lines), cheap_families.py (26), registry.py, otr_video_director.py,
   otr_meta_brief_image_prompt.py, _otr_story_brief_helpers.py. KEY QUESTION: did the OLD `flux_still`
   SELF-GENERATE a Flux image (so it always had content), while the renamed `still_pan` only pans a
   PROVIDED still -> now black when the still-routing doesn't feed it? Did the rename change the
   `still_pool_key`/`beat_id` lookup key, or the `kind` the dispatcher writes, so `_still_index` no
   longer matches?
2. **`e8fa941d` (ltx_audio_in role-driven still/audio/prompt routing, Chunk 1)** + **`a30f5945`
   (Chunk 2)** -- both edited render_driver.py's still routing (+80 / +25 lines). Did the role-driven
   still selection change which beats get a scene_* still vs none?
3. **`a9168575` C3 (unregister scaffolds)** / **`2139b5d7` C4 (delete VALIDATED_ENGINES filter)** /
   the opt-in-deletion C-series -- did removing a gate change which engine the still slot resolves to,
   or how the dispatcher decides to mint a scene_* still for a still-carrier beat?
4. The image dispatcher (`otr_image_gen_dispatcher.py`) materialization (ST-3): it mints to
   `_shared/cache` (content-addressed) + materializes a copy into the episode `stills/` AND writes the
   `images.images` scene_* row. For the lumina leg the episode `stills/` was EMPTY -> did
   materialization / the ledger write-back regress?

## WHAT TO DETERMINE (the panel's job, grounded vs the code)
A. For a still-carrier beat (still_pan), trace EXACTLY: dispatcher mint -> `images.images` scene_* row
   (what `kind` + `beat_id`/key?) -> `_still_index` key -> `build_request_from_shot` lookup. Where does
   the chain break so `init_image` is empty?
B. Which of the ~48h suspect commits introduced the break (diff `c1132196` / `e8fa941d` / `a30f5945`
   against the still-routing + dispatcher write + the lookup key).
C. The "before we removed fallbacks" connection: did a fallback path previously mask a missing still
   (degrade to a different still source) so the black floor never showed, and a recent change exposed it?
D. Is it ALL still-carrier legs (image legs + still_* video legs) that go black, while portrait-
   conditioned engines (humo) keep content? (matches operator's "some have images, most black").

## INVARIANTS (a fix must not break)
Workflow JSON source of truth; 100% local; master audio byte-identical; UTF-8 no BOM; SFW; no shim.
This is a DIAGNOSIS pass -- output the grounded ROOT CAUSE + the minimal fix locus, not a broad rewrite.
