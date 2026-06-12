# PASS 04 REVIEW FOCUS: WIRING

You are one panelist in an adversarial review of the plan below. THIS pass
is the WIRING pass: how the two new engines thread through Director ->
ShotLock -> render driver -> fallback -> ledger. Architecture, I/O, and
prompts are LOCKED (pass01-03) -- one-line flags only.

Pressure-test exactly these against the grounding:

1. MUSIC audio_ref ATTACH: the slice mechanism exists
   (render_driver._slice_master_audio; the announcer/talking path fills
   audio_ref by slicing [start_s, start_s+dur_s] from the frozen master).
   Find the exact call site / branch where MUSIC beats' requests are
   assembled, and specify the additive change that attaches the per-beat
   slice for engine_id == "ltx_av_music" ONLY (existing engines must see
   IDENTICAL requests to today -- byte-identical ledger/request hashing
   concerns?). Does request hashing / render_request_hash change when
   audio_ref is added, and does that invalidate caches or seeds for
   EXISTING engines? This is the pass's most dangerous edge -- be precise.
2. DIRECTOR -> SHOTLOCK: V-6 says the dropdown auto-includes new
   registered engines and role compat filters at execute. Verify
   ltx_av_talk appears in announcer_video_model AND other_beats slots,
   ltx_av_music in music_video_model; what does OTR_ShotLock's
   assert_usable path do when the flag is off (GATED_BY_FLAG) -- clean
   fail-closed to the role's default engine, or episode abort? Specify
   expected behavior + the test that proves it.
3. FALLBACK MECHANICS: on a render-time ltx_av_talk failure, who walks
   the chain (driver? render batch node?), what EXACTLY is restamped in
   the ledger (engine_id? degradation_trail? group restamp via
   resolver.prune_orphaned_groups?), and what log line proves the swap is
   LOUD? Write the exact restamp wording for (a) the talk aspect-change
   degrade, (b) the music ltx_video degrade, (c) the pad-tail >2s case.
4. ENGINE-IDENTITY LEDGER STAMPS: the audio side proved per-line engine
   identity is unprovable without stamps (H4/P0-zero). What is the video
   side's current per-clip identity stamp (engine_id on shot rows?
   degradation_trail?), and what must the new lane add so an acceptance
   grep can PROVE which engine rendered every clip (incl. after
   fallback)?
5. OTR_FORCE_ENGINE_MAP: how does it interact with role compat + the
   flag gate for the new names -- can the operator force ltx_av_talk on
   announcer beats for the M4 smoke with one env, and does forcing bypass
   assert_usable (it must NOT)?
6. PORTRAIT / init_image SUPPLY for announcer: portrait_ledger +
   announcer alias behavior -- does an announcer beat reliably get an
   init_image today (in-character portraits shipped 435ba0a), and what
   happens to ltx_av_talk when the portrait is missing (fail-closed ->
   fallback, or starve)? Specify expected behavior.
7. SEEDS: request_seed derives from render_request_hash (build_request).
   Confirm the new engines inherit deterministic per-shot seeds with no
   extra work, and that the C7 env overrides (OTR_CAST_SEED/...) are
   irrelevant here.

Rules: cite grounding or VERIFY-AT-BUILD; existing engines' requests/
hashes/ledgers must be BIT-IDENTICAL to today when the new lane is dark
(default-OFF) -- any wiring change that alters dark-lane behavior is a
MUST-FIX against itself. Output: numbered MUST-FIX (file + what),
SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.
