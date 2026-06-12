# pass05 testing -- Claude panelist review (before reading the panel)

MUST-FIX

1. (forgot-it detector matrix) Every touch-list edit needs a test that
   FAILS if the coder forgets it. Matrix (edit -> detecting test):
   schemas family -> test_ltx_av_schema.py family round-trip
   (VideoRequest(family_hint="audio_conditioned_video") validates; sync
   assert import does not explode); role_compat music supply ->
   dropdown/slot fit test (ltx_av_music accepted for music_video_model);
   __init__ import -> registry presence test (both names in
   all_engine_names()); driver (a) canvas -> request canvas == landscape
   for both names; (b) prompt gate -> no-creative ltx_av_music request
   carries brief-composed prompt, ltx_av_talk carries talk template, NO
   radio override on talk; (c) synthetic slice -> line-less b000 shot
   with engine ltx_av_music gets a non-None audio_ref, AND the SAME shot
   with engine ltx_video gets None (dark-lane guard); (d)
   request_template -> assert_usable receives canvas (spy engine);
   (e) ENGINE_FAMILY -> engine_family("ltx_av_music") ==
   "audio_conditioned_video" with adapters unimported; (f) force guard ->
   music_visual=ltx_av_talk entry ignored LOUD; (g) announcer alias ->
   line-less announcer shot + images-ledger portrait -> asset_refs
   filled. Mirror test_video_engine_registry_base_additive /
   test_video_fallback_chain_additive structure ("additive" = tolerant
   of future engines; never assert exact registry counts -- assert
   MEMBERSHIP, not cardinality, so the next engine doesn't break us the
   way we'd otherwise break today's tests).
2. (golden dark-lane fixtures) One test builds requests for a fixed
   ledger fixture with the lane registered-but-dark and compares the
   FULL request dicts (json.dumps sorted) for ltx_video/wan_i2v/humo
   beats against checked-in goldens captured pre-lane. This is the
   single strongest "nothing moved" proof; place goldens under
   tests/fixtures/ltx_av_dark/.
3. (byte-identical) Do NOT duplicate the crown jewel: the CPU
   prune-to-node-7 soak proves the audio path without video; forcing
   ltx_av changes nothing on that closure (audio inputs are read-only
   slices). CPU additions instead: canonicalize-contract unit (stub
   frames -> has_audio False, zero audio streams via ffprobe on the tiny
   emitted mp4, engine_id stamped); the fake-AV-mp4 strip test. The
   FULL-EPISODE forced-lane byte-identical run is an M4 GPU gate
   (existing test_audio_byte_identical mechanics, master hash compare).
4. (Desktop-vs-headless gate) CPU unit: monkeypatch a fake
   NODE_CLASS_MAPPINGS missing one required class -> assert_usable
   raises naming exactly that class. M0 checklist: one row per build
   (Desktop, headless launcher) per node class. M4 grep: headless boot
   log shows the node pack version + classes resolved.
5. (av_dims unit set, exact cases) 1472x832 frames 49 PASSES; 1450x832
   RAISES naming 1440/1472; 832 ok / 831 raises; frames: T=25 ->
   render 25; T=26 -> 33; T=497 -> 497; T=520 -> capped 497 + pad-tail
   path flagged; frames%8==1 violations raise with nearest 8n+1 both
   directions. next_8n1 idempotent on valid values.

SHOULD-CONSIDER

6. M0 sheet as a CHECKED-IN artifact: docs/2026-06-10-ltx-av-lane/
   M0_RESULTS.md with fixed `key: value` lines (max_frames, vram_peak_gb
   per lane, wall_s per lane, node classes per build, audio formats,
   verdicts per P1 cell). From M2 on, a test asserts the file exists,
   parses, and LTX_AV_MAX_FRAMES in eng_ltx_av.py EQUALS the sheet's
   max_frames (constant-drift guard).
7. b7 forbidden sweep: the new file is swept automatically (AST loop var
   `imp` gotcha); eng_ltx_av must contain no torch.hub / KJNodes /
   banned imports at module scope -- cold-import test covers the V-12
   side via package import; add the new module to any explicit module
   list ONLY if the sweep uses one (VERIFY -- prefer glob-based).
8. Bug Bible: at-risk rows to re-run explicitly = BUG-070 (Sage gate
   reused by both adapters), BUG-291 (reclaim on lease release), BUG-265
   (HuMo tier loader -- talk's fallback target). NEW row candidate at
   ship: "LTX dims silently round; OTR fails loud via av_dims" with the
   Three-File Contract (YAML + README + regression test together).

OPEN-QUESTIONS

9. Does test_video_fallback_chain_additive auto-walk ALL registered
   engines' chains (then the 5-hop talk chain is covered for free), or
   does it need explicit new cases? (Grounding will tell the panel;
   either way membership-style.)
10. Is there an existing dropdown-contents test for OTRVideoDirector
    that asserts exact COMBO lists (would break on registration) --
    if so it moves to membership assertions in the SAME commit as M1.
