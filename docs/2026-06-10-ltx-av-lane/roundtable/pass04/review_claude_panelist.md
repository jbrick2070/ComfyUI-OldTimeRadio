# pass04 wiring -- Claude panelist review (before reading the panel)

MUST-FIX

1. (render_driver.py, music attach) Driver-side, ENGINE-GATED branch in
   build_request_from_shot: when shot.engine_id == "ltx_av_music" and the
   per-line lookup yields no wav, call the EXISTING
   _slice_master_audio(master, start_s, dur_s) exactly like the talking
   path and set audio_ref={"path": slice}. Do NOT teach the render node a
   new slice point. HASH SAFETY (grounded): render_request_hash is stamped
   on the SHOT ROW at ShotLock BEFORE request build (:457-459 reads it;
   seeds derive from it via _seed_from_hash) -- adding audio_ref to the
   REQUEST cannot move hashes or seeds; dark-lane requests are bit
   -identical because every new branch keys on the new engine_ids, which
   cannot appear while the lane is dark.
2. (restamp wording) Reuse the existing degrade mechanics (render_shot +
   make_fallback_of: no-declared-fallback floors to still_kenburns,
   degradation_trail on timing) and pre-agree the three reason strings:
   (a) "ltx_av_talk -> humo: render failure <err>; ASPECT CHANGE
   landscape->portrait pillarbox"; (b) "ltx_av_music -> ltx_video: render
   failure <err>; audio conditioning lost (text-only)"; (c)
   "ltx_av pad-tail: rendered <n> capped frames, padded <m> (>2s) by last
   -frame repeat". All three appear in BOTH the log line and the ledger
   degradation_trail entry.
3. (identity stamps) Per-clip proof = the shot row's engine_id AFTER
   restamp + degradation_trail + the render log "engine=<id>" line; the
   acceptance grep lists every shot with final engine + trail. VERIFY
   -AT-BUILD the exact existing log format in render_shot and keep it
   IDENTICAL for the new engines (grep-stable).
4. (flag-off behavior) Director pick of a gated engine must abort AT LOCK
   with the named EngineUnusable (GATED_BY_FLAG) -- fail-closed pick, not
   silent re-route (registry docstring contract). Test: policy picks
   ltx_av_talk, flag off -> ShotLock raises naming the flag. Defaults
   stay ltx_video everywhere while dark (line 69 defaults table
   untouched).
5. (FORCE map) OTR_FORCE_ENGINE_MAP accepts the new names automatically
   once registered (parser validates against the registry, :689-705;
   unknown -> warn + IGNORE ALL, :724). Forcing re-routes the plan but
   must NOT bypass assert_usable -- VERIFY the forced path still walks
   the lock/validate asserts; M4 smoke uses
   OTR_FORCE_ENGINE_MAP=announcer_visual=ltx_av_talk,... with the flag ON.

SHOULD-CONSIDER

6. Announcer portrait missing at runtime: ltx_av_talk fails closed pre
   -render -> chain walks -> humo ALSO portrait-starved (degrades LOUD
   today, :374) -> latentsync lacks base_clip -> still_kenburns floor.
   Acceptable LOUD cascade; assert the trail records EVERY hop (test).
7. Synthetic opening-music shots (no ledger line, :462) must work with
   ltx_av_music: they carry timing only -- the music attach branch must
   tolerate line-less shots (slice from timing alone).

OPEN-QUESTIONS

8. Does ShotLock's execution-group stamping treat single-clip ltx_av
   shots as plain consumers (no providers to orphan on degrade), so
   resolver.prune_orphaned_groups is a no-op for this lane? (Expect yes;
   confirm in grounding/tests.)
