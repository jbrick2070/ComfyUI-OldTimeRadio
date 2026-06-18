<!-- model: gpt-5.1 (direct API) -->

- VERDICT: no. Core pool_n_loop semantics and beat→still mapping are undefined and contradict existing graph order.

- MUST-FIX BEFORE BUILD:
  1. [The crux / Decisions 1–2] The plan never defines a concrete, implementable beat→still mapping that works with current graph ordering (still phase before ShotLock). As written, `derive_scene_still_targets` "does NOT see the clip plan today" and runs before `OTR_ShotLock`, but the crux requires stills to follow `clip_mode` and to share pool stills consistently with `_audio_derived_clip_budget`. You currently need:
     - A specific design for how clip-mode information reaches the still derivation phase, given the existing ordering (e.g., threading `other_beats` `clip_mode`/`pool_n` through `OTR_VideoDirector -> OTR_ImageDirector -> image_policy_json -> derive_scene_still_targets`).
     - A precise algorithm for beat-indexing and pooling that can be implemented in `derive_scene_still_targets` without access to `ShotLock`’s already-processed groupings.  
     Concrete fix: Amend the spec to state exactly:
       - What new fields are added to `image_policy_json` to carry `other_beats_clip_mode` and `other_beats_n`.
       - How `derive_scene_still_targets(lines, fps, image_policy_json)` (or equivalent) is extended to:
         - Count the “other-beats” beats in the same grouping/order that `_audio_derived_clip_budget` uses, or
         - Use a clearly defined, deterministic index (e.g., “other-beats sorted by global beat index; beat k uses still index k mod N”).
       - How render_driver resolves the same index key it currently uses per beat_id to this pool index in a deterministic way (e.g., new `still_pool_index` on shot rows, or a mapping table).
       - How this mapping is kept consistent between still derivation and shot planning given the graph order (e.g., pre-deriving a lightweight pool mapping in `VideoDirector` used by both).

  2. [The crux / Decisions 2] The document assumes that “the SAME pool/loop math otr_shot_lock uses for clips” can be reused for stills, but gives no actionable location or format for this sharing. Currently:
     - `_audio_derived_clip_budget` uses `render_count = min(pool_n, len(other))` and runs after audio_done, while the still phase runs earlier and has no access to that computation.
     - The spec requires determinism and no hidden RNG, but doesn't pin down how `M` (“other-beats”) is defined and communicated to the still phase.  
     Concrete fix: Specify:
       - Whether `OTR_ShotLock` exposes a reusable pure function (e.g., `_audio_derived_pool_indices(other_beats, pool_n) -> [index_per_beat]`) that can be imported and used by `derive_scene_still_targets` [ASSUMPTION: you may choose to refactor it], or
       - Whether you reimplement the exact loop logic in the still phase, including a definition of "other-beats" that matches `ShotLock` (roles, ordering) and is test-asserted identical.
       - Include in the plan an assertion test: “for any ledger, the set and order of beats in other-beats used by still derivation matches `_audio_derived_clip_budget`’s other-beats; and for `pool_n_loop`, the still indices per beat are `i mod N` over the same iteration.”

  3. [The crux / Decisions 2, Grounding: render_driver.py] The plan handwaves that “the still phase must either … (b) emit N stills keyed so every other-beat resolves to one of them,” but render_driver currently resolves stills by beat_id-specific indices (`_still_index` keyed by beat_id). There is no described mechanism for:
     - A beat to point to a different beat’s still (pool sharing),
     - Or for render_driver to understand a non-1:1 beat_id→still index.  
     Concrete fix: Extend the plan with a concrete schema change and implementation location, e.g.:
       - Add `still_pool_key` or `still_pool_index` to the per-shot row in `ledger['video'].shots[]` from `OTR_ShotLock`, and define how render_driver chooses the still:
         - `if still_pool_index is not None: use still[still_pool_index]; else: use still_index[beat_id]`,
       - Or define how `still_index` becomes a map `beat_id -> global_still_index` that can intentionally point to shared entries for pooled beats.
       - Specify where and how `derive_scene_still_targets` emits stills such that these indices are valid (e.g., N still entries in global still list, with known ordering).
     Without this, beats > N will hit MISSING-STILL despite the planned N-pool.

  4. [The crux / Decisions 1, Grounded facts about `derive_scene_still_targets`] The spec promises “pool_n_loop -> exactly N stills, SHARED/looped across the M other-beats,” but `derive_scene_still_targets(lines, fps)` today emits a scene-still target for EVERY beat and doesn’t see clip mode. The plan currently has no explicit step to:
     - Prevent over-generation of stills when `clip_mode = pool_n_loop`,
     - Or to maintain per-beat stills for non-other-beats (announcer/music) while constraining only other-beats.  
     Concrete fix: Update the design to:
       - Introduce conditional logic in `derive_scene_still_targets` based on roles and clip_mode: 
         - For roles categorized as "other-beats" (exact list must be specified; currently noted as BACKGROUND_ABSTRACT + SCENE_BROLL), emit only N distinct targets and mark them as pooled.
         - Continue emitting per-beat stills for announcer/music/character_background roles irrespective of pool_n_loop, as required.
       - Explicitly define how “other-beats” are detected (speaker roles vs video roles; e.g., via `SPEAKER_TO_VIDEO_ROLE` and/or `Role.BACKGROUND_ABSTRACT`/`Role.SCENE_BROLL`).
       - Document the exact JSON structure of a still target in the new pooled mode and how it differs from per-beat targets.

  5. [Decisions 3 / Invariants, Grounding: render_driver.py] The spec suggests possibly distinguishing “consumes SCENE stills” from `accepts_still`, and that HuMo other-beats might not need scene stills at all. But:
     - `ENGINE_FAMILY` indicates `humo` / `humo_1.7B` are `audio_driven_face`. The grounding text says: “audio_driven_face (HuMo) keeps the character PORTRAIT (`_portrait_index` by char_id); `ltx_video`/... use the beat's SCENE still.”
     - The current invariant is “accepts_still stays the central usability gate; role_compat unchanged.”  
     If you keep `accepts_still=True` for HuMo and start tying still count to clip mode, pooled still generation for other-beats where HuMo is used may become a no-op (wasted generation). The document leaves this undefined.  
     Concrete fix: Decide and specify one of:
       - Leave HuMo’s `accepts_still=True` but add a new attribute such as `consumes_scene_still`, and:
         - Define that the still dispatcher checks `consumes_scene_still` to decide whether to generate scene stills at all.
         - Set `consumes_scene_still=False` for HuMo so no scene stills are minted for HuMo-only roles.
       - Or explicitly state that HuMo’s current `accepts_still=True` is required for some path and that the waste is acceptable, in which case the spec must clarify that clip-mode-based still count only applies to engines/families that actually use scene stills (e.g., text_to_video, image_to_video, static_motion, flux_still), and that HuMo lanes are exempt.
     Also, specify how this interacts with the existing centralized `engine_consumes_still` in `otr_image_gen_dispatcher` (new attribute vs extending its logic).

  6. [Decisions 5 / Invariants] The spec says “keep announcer/music per-beat (not pooled)” but the algorithm for distinguishing these from “other-beats” in the still-derivation phase is not spelled out. Given:
     - `SPEAKER_TO_VIDEO_ROLE` maps `music`, `music_open`, etc., to `MUSIC_VISUAL`, and default is `BACKGROUND_ABSTRACT`.
     - The crux calls “other-beats = roles BACKGROUND_ABSTRACT + SCENE_BROLL”.  
     There is currently an ambiguity whether music/announcer beats that end up as BACKGROUND_ABSTRACT (e.g., if mis-labeled) would be incorrectly pooled, violating “keep announcer/music per-beat.”  
     Concrete fix: Amend the design to:
       - Define “other-beats” precisely in terms of either `speaker_role` or derived `video_role`, and list all roles that are explicitly excluded (announcer/music/character-video).
       - Specify assertions or tests to ensure announcer/music beats are never included in the pooled category, even if their video engine is abstract / static.

  7. [Decisions 4] The JSON/default change (“pool_n_loop + N=4 the full-JSON default”) is underspecified:
     - The doc says the widget defaults today are `unique_per_beat`, `n=8`, operator wants new default for `otr_scifi_16gb_full.json`, but it is unclear whether:
       - The global widget default in `OTR_VideoDirector` is changed, or
       - Only that single workflow JSON is changed, and how conflicts between saved JSON vs widget default are resolved.  
     Concrete fix: Spell out:
       - Whether `otr_scifi_16gb_full.json` becomes the new source of truth for these defaults and a migration path for existing saved workflows (e.g., versioned schema / defaulting behavior).
       - How the UI handles preexisting episodes that serialized old default values.
       - What re-validation or tests will assert the expected default in both the widget and the serialized workflow.

- SHOULD-FIX:
  1. [Grounded facts; “visualizer -> 0 stills is ALREADY handled”] The spec leans on `accepts_still=False` for visualizer/ltx_av_music and central `engine_consumes_still` behavior. This assumes no other code path will try to “force” stills for these roles, but does not acknowledge the interaction with the new pool_n_loop logic.  
     Recommendation: Explicitly note in the design that:
       - Pool logic runs on the *set of roles whose engines both belong to the scene-still-consuming families and have `accepts_still=True`*.
       - Visualizer / ltx_av_music remain completely excluded regardless of clip mode and pool_n, and that tests/asserts guard that no stills are emitted for them even if `clip_mode=pool_n_loop`.

  2. [Determinism / Tests] The spec emphasizes determinism but does not define a canonical ordering for “other-beats” when applying the pool mapping (critical when seeding, and when counts differ).  
     Recommendation: Specify that “other-beats” are ordered by:
       - Global beat order from ledger audio (e.g., by `beat_id` sortkey), or
       - The same sequence `OTR_ShotLock` uses to iterate them.  
     And mandate a regression test that for any ledger ordering, the mapping is deterministic across runs.

  3. [Failure modes; loudness] The spec says “no silent fallback, LOUD on any miss,” but does not address what happens if:
       - `pool_n` is 0 or negative for `pool_n_loop`,
       - Or `pool_n > M`.  
     Recommendation: Specify:
       - For `pool_n_loop` with `pool_n <= 0`, treat as “0 stills” for other-beats with a clear WARN; confirm that render_driver won’t try to resolve non-existent stills (i.e., mapping must avoid referencing any still index when N=0).
       - For `pool_n > M`, clarify whether you generate only M stills or exactly N distinct stills with the last ones unused (and ensure render doesn’t depend on unused ones).

- OPTIONAL / NICE-TO-HAVE:
  1. Document an explicit mapping table `clip_mode -> still_generation_strategy` (visualizer, unique_per_beat, pool_n_loop) including per-role differences, to reduce drift in future contributors.
  2. Add an explicit design note about performance: expected still count change vs baseline, and how to confirm via tests/metrics.

- CUT THESE (over-engineering):
  1. [Decisions 3] Introducing a second flag “consumes SCENE stills” separate from `accepts_still` may be over-engineered unless you can show a real engine that uses one but not the other. Given current behavior:
     - `accepts_still=False` already fully disables still minting via `engine_consumes_still`.
     - For HuMo, the waste issue can be addressed more simply by:
       - Either flipping `accepts_still=False` for HuMo in roles where it never uses scene stills [ASSUMPTION: if no other path needs them], or
       - Not applying the pool/n semantics at all to families that do not use scene stills (ignore clip_mode for those roles).  
     Cutting the new flag removes migration risk in dispatcher and avoids duplicating “consumption” concepts.
  2. [The crux – option (a) vs (b)] The idea of a complex beat→pool mapping object that all three layers (still-gen, dispatcher, render_driver) must honor may be heavier than needed. A simpler approach:
     - Keep a single global still list with indices 0..N-1 for the pool,
     - Have shot rows carry only a `still_index` integer for each beat (computed by pool logic once in ShotLock),
     - Let render_driver stay ignorant of pooling semantics.  
     This avoids having multiple representations (pool list + mapping) that must remain in sync.