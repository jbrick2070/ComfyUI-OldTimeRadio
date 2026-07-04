# OTR HANDOFF LOG (append-only; newest at TOP)

The running record of what each session DID. GO_FORWARD_PLAN.md holds the forward plan (lean);
this holds the history so the plan can stay lean. One short entry per session. Deep detail lives
in the per-sprint docs + git; this is a breadcrumb trail, not a dashboard.

---

## 2026-07-04 (NO-FALLBACK RIP: R3-chunk-4 ShotLock + Fable SHIP -- ALL 10 SITES COMPLETE) -- HEAD 432cb576 (v2.0-alpha)
Did (operator ordered "rip it all" -> ripped the 10th site the chunk-3 reviews surfaced; suite 6141/0
+ Bug Bible 16, pushed HEAD==origin):
- @26b236e6 `otr_shot_lock.py derive_creative_directives` (the per-beat video-directive path): empty/
  unparseable directive after reseeds -> raise iff a writer LLM was attempted (`llm_fn is not None`);
  story-consistency/person gate fail on an LLM prompt (`source=="llm"`) -> raise; `llm_fn=None` keeps
  the legit local template lane; `consistency_gate_warn_only=True` now KEEPS the AI prompt (was a
  soft-logged template swap). Docstring updated. Tests inverted (test_video_platform_aseam empty-reseed
  + consistency; test_look_qa_round5 object-only/no-person) + llm_fn=None keep test added.
- @432cb576 Fable-gate followups: (1) REAL robustness fix -- the reseed loop broke on ANY non-empty
  parse, so a partial batch reply (14/15 beats) aborted with retries UNSPENT; now breaks only on full
  coverage (+ test_passpm_partial_batch_spends_reseeds_then_raises); (2) stale widget tooltip + M4
  section header ("template fallback; never aborts") updated to the no-fallback reality; (3) comment on
  the deliberate image-lane divergence (person guard folded into the combined gate; subject anchor is
  the compensating control).
- Gates: Fable §9 on chunk-4 = SHIP (happy-path never raises; caller :931 doesn't swallow;
  _resolve_writer_llm None -> template lane intact; warn_only change correct; faithful image-lane mirror;
  even verified workflows/otr_scifi_16gb_full.json ships warn_only=false so prod gets the loud raise).
  No workflow-JSON change (tooltip is a STRING in INPUT_TYPES, not a positional widget value).
Current step: NO-FALLBACK RIP COMPLETE (all 10 sites). NEXT = resume the PARKED credits-enrichment
LIVE frame-level smoke (S3+S1 shipped @5f510ebe); then S0 -> S4.
Next: credits-enrichment LIVE frame smoke on a SHORT + LONG episode.
Commits: 26b236e6 432cb576 (+ GO_FORWARD/HANDOFF docs).

## 2026-07-04 (NO-FALLBACK RIP: R3-chunk-3 image+casting + Fable SHIP -- 9-site set COMPLETE) -- HEAD a11d8605 (v2.0-alpha)
Did (closes the R3 arc's originally-scoped 9 sites; every chunk suite 6139/0 + Bug Bible 16, pushed
HEAD==origin):
- @74433163 image lane: `otr_meta_brief_image_prompt.py` -- portrait 4 tiers (tier-2 empty /
  story-consistency gate / person-guard / gear-scrub-empty) + `_compose_char_scene_prompt` now RAISE
  a named RuntimeError("... no-fallback rip 2026-07-03") when the prompt came from the writer LLM
  (`source=="llm"` / `_llm_attempted`). `llm_fn=None` keeps the legit deterministic local template
  lane (NOT a fallback); `consistency_gate_warn_only=True` keeps + logs. Contract docstring updated.
- @74433163 casting: `_otr_casting.py _apply_llm_slot_fill` (opt-in `name_mode=="llm_slot_fill"`)
  raises CastValidationLLMError on BOTH fail paths (generate_fn raises; validation not-ok) instead of
  silently keeping the deterministic RNG-pool names. (Ctor is (attempts,name) -> tag in the attempt str.)
- @4ac329f2 refine-loop: the grounded general-purpose review caught that the writer `_refine_loop`
  (OTR_LedgerScriptWriter.py:2306) swallowed the casting raise on refine pass>=1 (E4 outer-catch),
  making the casting rip inert in refine-ON + llm_slot_fill. Narrowed the catch to re-raise
  CastingFailedError unconditionally (cast deterministic per forced_seed); kept transient pass-skip.
- @a11d8605 fixed the now-stale lock_cast comment (Fable followup).
- Tests: inverted the pinned fallbacks in the SAME commit (test_image_platform_c1 reseed+consistency,
  test_brief_prompt_finishing person-guard, test_cast_llm_naming both R8 paths) + ADDED a char-scene
  fail-loud test; kept the llm_fn=None keep tests. No workflow-JSON change (pure runtime rip).
- Gates: grounded general-purpose review (found + fixed the refine-loop swallow) + Fable §9 = SHIP
  (all 4 portrait tiers RAISE confirmed correct vs the F3-keep precedent; happy-path dormant; no
  surviving swallow). agy+Sonnet review packet: docs/2026-07-03-no-fallbacks-rip/R3_CHUNK3_REVIEW_PROMPT.md.
Current step: R3 9-site set COMPLETE. NEXT = operator decision on the NEW 10th site candidate
`otr_shot_lock.py derive_creative_directives` (same model->template shape; test-pinned, needs its own
re-baseline) -- rip as R3-chunk-4 or defer to a ShotLock audit. With it deferred, credits-enrichment resumes.
Next: operator ruling on the shot-lock sibling; then the parked credits-enrichment LIVE frame smoke.
Commits: 74433163 4ac329f2 a11d8605 (+ GO_FORWARD/HANDOFF docs).

## 2026-07-03 (stack-wide NO-FALLBACK RIP: cloud audio + R1/R1c/R2/rename + R3 chunks 1-2) -- HEAD 1d8f7e2b (v2.0-alpha)
Did (long single-arc session; operator directive: every model failure fails LOUD, never a silent
swap/canned-template). Source of truth = docs/2026-07-03-no-fallbacks-rip/PLAN.md. All chunks suite
6138-6142 + Bug Bible 16 green, 0 regressions, pushed HEAD==origin per chunk:
- CLOUD AUDIO built fail-loud: ElevenLabs TTS adapter @925438e2 + Sonilo music @c7da53b1 (generate ->
  invoke_partner_node -> canonicalize_audio RMS -16dBFS; fail-loud/no-fallback; cloud EXCLUDED from
  rank_chain auto-select). writer_fallback backup-LLM dropdown ROADMAPPED (docs/ROADMAP_IDEAS.md).
- R1 @822cb0c9 audio-voice (bark missing-ref net + cast_lock fail_soft->fail_loud [KEEP announcer
  reroute, E2] + kokoro voice-id swap + stage-direction silence). Fable-gated SHIP.
- R1c @2d4cd864 scene_sequencer inline-Bark clip-shortfall -> raise (Fable caught this straggler).
- R2 @f07b837d image (empty named-slot raise E8-precise + scene-still-missing -> raise).
- RENAME @31c2a473 other_beats_image_model -> character_image_model (13 files: ImageDirector/
  VideoDirector INPUT_TYPES + dispatcher + _otr_workflow_apply + workflow JSON + widget_mapping + 7
  test files; live validator green; slot is character-only post scene_broll rip).
- R3-chunk-1 @de6af8c2 dramatic-state (import-fail + LLM call-fail -> raise; kept no-news-input path).
- R3-chunk-2 @1d8f7e2b announcer intro/outro/coda LLM->template swaps -> raise (F3-hedge KEEPS the AI
  line; retired the NEWS_CODA_POOL floor; 16 tests inverted across 5 files).
- Gates used: kibitz r2 (Codex + Claude anchor) + operator agy runs + a Sonnet cross-audit (found the
  _otr_casting.py:1345 straggler) + 3 Fable passes. Dead bark-health delete DEFERRED (fragile
  test_cast_fuzzy_consolidate regex-extraction bounded by the bark def -- needs a dedent-bounded fix).
Current step: R3-chunk-3 -- rip otr_meta_brief_image_prompt.py (portrait 3-tier: tier-2 clean rip;
tiers 3/4 consistency+PERSON-GUARD need keep-AI-vs-raise judgment, do NOT blind-rip [face engine]) +
char-scene prompt + _otr_casting.py:1345, THEN the Fable gate on the whole R3 diff.
Next: portrait file rip (careful with the person-guard), then casting-naming, then Fable-gate + merge.
Commits: 925438e2 c7da53b1 (+docs) 822cb0c9 2d4cd864 f07b837d 31c2a473 de6af8c2 1d8f7e2b (+several docs).

## 2026-07-03 (credits overlay redesign + music-still fix + ElevenLabs S0 start) -- HEAD 055dffbe (v2.0-alpha)
Did (multi-thread session after the S3+S1 credits atomic slice):
- CREDITS OVERLAY REDESIGN @ a0224438: operator rejected the single rolling roll; rebuilt
  OTR_CreditsRoll as the 3-column SIGNAL LOST console per the operator's design prototype
  (docs/Credits Overlay - plain.html; spec+plan in docs/2026-07-03-credits-enrichment/
  CREDITS_OVERLAY_{DESIGN,BUILD_PLAN}.md). Cols 1-2 STATIC dashboard (episode-title HERO +
  SIGNAL LOST 50% subtitle; MODELS w/ family suffix; [PRODUCTION LEDGER] vram/frames/seed/commit/rev;
  [SYSTEM]; CAST & VOICES delivered; [WRITER/LLM CONFIG]); COL3 SCROLLS the full narrative (STORY
  SPINE + full transcript + SOURCE INTERCEPT + DIAGNOSTIC, nothing dropped; duration = scroll length
  declared to the credits-aware mux). radial bg/neon/glow/scanlines; no-fallback. Added the per-engine
  family stamp to render_engines.by_engine (otr_video_render_batch). 20 spec tests rewritten. Design
  hardened by 3 Fable passes (code-ready plan + delta + creative "make it better"). Preview PNG looks
  faithful (docs/2026-07-03-credits-enrichment/credits_console_preview.png). Suite 6116/0 + Bug Bible 16.
- MUSIC-STILL LIP FIX @ b1fecb6f: ltx_audio_in radio-face talking still is ANNOUNCER-ONLY; the music
  bookend stays faceless (render_driver: gated _is_never_humo_video_role -> == "announcer_visual") so
  ltx doesn't lip-sync a mouth to music. test_ltx_radio_face_ab updated.
- ELEVENLABS/CLOUD-MUSIC S0 START @ 055dffbe (per docs/2026-07-03-elevenlabs-tts/BUILD_PLAN.md +
  CODER_KICKOFF.md): C1 (cloud runtime + EngineProfile cloud fields partner_row/provider_id/
  auth_required/billing_category/canonicalizer/error_policy + fail_loud validation) + C3 (threaded
  provider_voice_id through _otr_voice_bank dataclass/loader, cast_lock _stamp, schema; additive,
  local unchanged). Suite 6116/0 + Bug Bible 16.
Current step: (a) ElevenLabs S0 REMAINDER = C8 (build canonicalize_audio + resolve the local-lane
  loudness reference, cloud_media_canonical.py:127/:68) + the V3 ElevenLabs re-pin (regenerates
  partner_nodes.yaml -- run image/video conformance same chunk). Then S1..S7. (b) Credits overlay
  live-validation via the yoga soak (below).
Next: finish S0 C8 + re-pin; then S1 (elevenlabs adapter).
Soak: LAUNCHED scripts/_otr_yoga_soak.py (operator's yoga run + the credits LIVE smoke): y1 viz_mxc_cpu
  (rainbow) 800w -> y2 viz_mxc_mandala 800w -> y3 cloud ideo+still_word 30w. Results ->
  scripts/_otr_soak_capstone_results/yoga_<stamp>/results.jsonl, finals -> otr/obs (must carry the
  NEW 3-column console). Running unattended.
Commits: b1fecb6f, a0224438, 055dffbe.

## 2026-07-03 (credits-enrichment S3+S1 atomic) -- HEAD 20a669de (v2.0-alpha)
Did: landed the S3+S1 ATOMIC slice per GO_FORWARD_CREDITS.md v4 -- the guaranteed-RED window.
- Registered OTR_CreditsRoll (__init__.py). RIPPED (video_engine.py): RENDER-ENGINES section from
  _build_hud_dossier (kept writer/story/system/resolved); the HUD credits-music loop -> SILENCE pad
  (closing_audio now unused-but-declared); the too-early treatment engine-enrich merge.
- Node-84 (otr_silent_composite.py): ripped ONLY the BUG-410 floor-extend-PAST-master; KEPT the
  master-mix A/V-sync cap + looped-last-clip closing-theme tail (I over-ripped first, caught by the
  positioned-fills-to-master test, reverted to the minimal rip).
- Mux (otr_master_audio_mux.py): credits-AWARE guard -- new FLOAT forceInput declared_credits_tail_s
  (node 85 slot 6); permits v <= a + declared + tol (declared>0) else the OTR_MAX_CREDITS_TAIL_S env
  ceiling; byte-identical assert intact.
- Workflow JSON: node 95 OTR_CreditsRoll; link 250 rewired 93->95; new links 274 (95->85 video), 275
  (92.1 manifest->95), 276 (95->85 declared tail slot6). last_node_id=95 last_link_id=276. (Re-compacted
  the JSON to ComfyUI single-line after an indent=2 pass -- commit fade23c3.)
- Retired the 4 moved tests IN the same slice (3 dossier RENDER-ENGINES + bug410 floor-extend);
  updated the visual-structure pin to 93->95->85. Suite 6108/0, Bug Bible 16, B7 green.
- GATE (CLAUDE.md §9): grounded general-purpose review (0 breakers) THEN Fable FINAL gate -- Fable
  caught a REAL deliverable-path bug (mux _default_out didn't peel "_with_credits" -> final would land
  in otr/episodes/<...>_with_credits/, failing the §6 Test-Path). FIXED + regression test @ 20a669de.
Current step: credits enrichment -- LIVE frame-level smoke (short+long, obs credits proof) IN PROGRESS,
  then S0 (font +50% + credits-aware duration budget), then S4 (footer :598). Formal codex+antigravity
  kibitz NOT run (general-purpose+Fable gate used instead) -- operator may run it before promotion.
Next: run the operator's 30-45w e2e matrix (local flux+ltx_audio_in + cloud legs; key is SET len=72;
  cloud voice TTS w/ indextts2/chatterbox fallback) -> confirm otr/obs carries the new silent credits tail.
Commits: 5f510ebe, fade23c3, 20a669de.

## 2026-07-03 (credits-enrichment window) -- HEAD f00a8e8e (v2.0-alpha)
Did: executed credits-enrichment S2 + the parallel scaffold lane per GO_FORWARD_CREDITS.md v4.
- S2 @ 3e0003e8: stamp_durable()/LedgerStampError in production_ledger (local-to-singleton copy
  + save()-None raises, OTR_TEST_MODE=1 = in-memory injection); call sites CastLock (cast+meta),
  image dispatcher (images+meta.image_engines), stable_audio_theme (meta.music_engine, first
  durable path), render batch (meta.render_engines, swallowing catch removed). 11 spec tests.
- Scaffold @ f00a8e8e: nodes/otr_credits_roll.py UNWIRED+UNREGISTERED (no-fallback receipts,
  declared silent tail, looped-last-clip backdrop, ffmpeg render+concat no source-copy, zero
  widgets) + tests/test_credits_roll_spec.py (13 green; absorbs 3 dossier tests + bug410).
- Suite 6098/0 + Bug Bible 16 green; pushed, HEAD==origin, no BOM, AST OK.
Current step: credits enrichment S3+S1 ATOMIC slice (rip node-12 HUD + node-84 tail-fill,
register+wire CreditsRoll 93->CR->85, credits-aware mux guard, test retirement, kibitz+Fable gate).
Next: open the red window in a FRESH window off GO_FORWARD_CREDITS.md + KICKOFF.md.
Commits: 3e0003e8, f00a8e8e (+ this docs handoff commit).

## 2026-07-03 day (later 3) -- TRUE 1080p cloud delivery SHIPPED (Fable-traced end-to-end)
Did: implemented real 1080p for the CLOUD lane, Fable-reviewed twice (local, no credits).
- `cloud_delivery_wh()` in cloud_media_canonical.py -- orientation-preserving 1080p cloud
  DELIVERY canvas (land 1920x1080 / port 1080x1920, env OTR_CLOUD_VIDEO_CANVAS[_PORTRAIT] +
  OTR_CLOUD_STILL_CANVAS[_PORTRAIT]). eng_cloud_video.canonicalize + eng_cloud_image._canvas_wh
  now conform to it (NOT the smaller per-family request canvas that was downscaling the clip).
- THE make-or-break (Fable r1): node 84 OTR_SilentComposite was hardwired 1472x832 in
  otr_scifi_16gb_full.json -> every cloud clip downscaled at composite. BUMPED to 1920x1080
  (surgical unique-substring replace; procgen node 12 already 1920x1080 -> now 1:1). Fable r2
  traced the whole chain: composite/scopes(94)/blend(93)/captions(86)/mux(85) all native 1080p
  1:1, mux -c:v copy + audio byte-identical assert intact, RTXUpscale not in graph -> no fails.
- pixverse quality default 540p->1080p; canonicalize_video -crf 18; flux_pro request snapped
  to /32 (1920x1088, canonical crops to 1080p) for the BFL schema. Grounded + REJECTED Fable's
  "ideo resolution invalid" claim -- ideo minted live with resolution=1024x1024 (passthrough).
- Tests: 5 cloud_delivery_wh unit tests + updated flux_pro/render_image asserts. Suite 6148/0
  + Bug Bible 16 green. PUSHED ffa832a9 (1080p) + 6725b5f9 (flux /32).
Current step: matrix soak still BLOCKED on Comfy Cloud credits (Payment Required after 4
word_razzle 1080p clips; Desktop shows 11,846 but the API-key path drained/differs).
Next: operator adds/verifies credits -> relaunch scripts/_otr_cloud_matrix_soak.py 0 (now delivers
TRUE 1080p). The soak empirically validates the remaining cloud IMAGE engines at 1920x1080
(Fable OPTIMIZE/verify list: recraft size->1820x1024 native 16:9; kling mode->pro for 1080p;
nano_banana_2/seedream_2 model-as-dict -- UNVERIFIED, confirm live not speculatively). Bug B
(motion-not-in-final) still open.
Commits: ffa832a9, 6725b5f9 (+ dd15f815 the V3 auth fix earlier this session).

## 2026-07-03 day (later 2) -- CLOUD BLOCKER ROOT-CAUSED + FIXED; matrix soak halts on CREDITS
Did: found THE cloud blocker (not a flake): the invoke bridge passed hidden auth
(api_key_comfy_org) as an execute() kwarg, but every comfy_api_nodes partner
(IdeogramV4/Recraft/FluxPro/NanoBanana/Seedream/Pixverse/Kling) is a V3 IO.ComfyNode
whose hidden is delivered via PREPARE_CLASS_CLONE -> cls.hidden (HiddenHolder), never
kwargs -> "IdeogramV4.execute() got an unexpected keyword argument 'api_key_comfy_org'".
FIX @ dd15f815: cloud_media_invoke._call_partner routes V3 nodes through PREPARE_CLASS_CLONE
(hidden keyed by the Hidden ENUM VALUE from the pin's inputs.hidden name->TYPE map); only
real inputs reach EXECUTE_NORMALIZED_ASYNC. Legacy V1 nodes keep the kwargs path. Added
test_v3_partner_hidden_via_clone_not_kwargs. Suite 6143/0 + Bug Bible 16 green, PUSHED.
Also: kibitz r1 (codex, grounded) on the 1080p change -- cloud res is FOUR surfaces
(provider tier / request canvas / canonical clip / composite); OTR_CLOUD_PIXVERSE_QUALITY
sets only the provider tier, then canonicalize_video (cloud_media_canonical.py:265) scales
the clip to the REQUEST canvas (1472x832 for word_razzle) -> quality knob alone does NOT
deliver 1080p; must NOT reach 1080p via OTR_VIDEO_LANDSCAPE_CANVAS (still/viz/floor locals
read it). Locals are VRAM-safe (ltx/humo/ltx_av have fixed per-family overrides in
render_driver:1394-1428). Built scripts/_otr_cloud_matrix_soak.py (14 legs: 4 cloud video
x ideo + 5 cloud image x {still_word,still_motion}). LIVE PROOF the fix works: ideo stills
minted across all beats + word_razzle Pixverse clips b000-b003 rendered end-to-end, no crash,
14.4GB free. Leg HALTED at b004: cloud_pixverse_i2v "Payment Required: add credits to your
account" -- Comfy Cloud is OUT OF CREDITS (account issue, NOT a bug). Soak stopped (all legs
need cloud credits).
Current step: OPERATOR must top up Comfy Cloud credits, then relaunch the matrix.
Next: (op tops up) -> `$env:OTR_COMFY_API_KEY=[Environment]::GetEnvironmentVariable(
'OTR_COMFY_API_KEY','User'); python scripts\_otr_cloud_matrix_soak.py 0` -> watch, root-cause
any per-engine failure, rerun. THEN true-1080p delivery = a cloud-video-only canonical canvas
1920x1080 (codex r1) wired without touching OTR_VIDEO_LANDSCAPE_CANVAS. Bug B (motion not the
base of the final) still open. Drivers gitignored.
Commits: dd15f815 (V3 hidden-auth fix + test).

## 2026-07-03 day (later) -- CLOUD AUTH UNBLOCKED (key) -- HAND OFF for 1080p cloud soak + Bug B
Did: operator set OTR_COMFY_API_KEY at USER scope (len=72). Proved auth resolves (scripts/
otr_cloud_s0_smoke.py --leg1: old "no credentials" GONE; remaining nodes.MAX_RESOLUTION err is a
standalone-harness artifact, not production). Built scripts/_otr_cloud_video_soak.py (cloud video
x ideo, 30w, indextts2; boots headless per leg, folds the key into the per-leg env so the server
inherits it; ensure_ollama preflight). Ran it: cv1 word_razzle x ideo verified rendering with auth
OK. Operator then said STOP (wants 1080p, not the default 832x480) -> killed driver + :8000 server.
Current step: cloud soak PAUSED at operator request; next window resumes at 1080p.
Next (fresh window, Opus 4.8/high): (1) wire 1080p -- canvas 1920x1080 on OTR_VideoDirector
(VERIFY canvas_w/canvas_h is patch-safe) + OTR_CLOUD_PIXVERSE_QUALITY=1080p (humo portrait
excepted); relaunch _otr_cloud_video_soak.py. (2) Bug B: motion not in final (legacy procgen
ships) -- trace manifest->compositor->mux, make motion the base. (3) fold kibitz r1 (codex+agy,
kibitz-runs/2026-07-03-cloud-video-fixes/r1/) findings. Bug A auth code-wiring now OPTIONAL (env
key solves it). Load the key each run: $env:OTR_COMFY_API_KEY=[Environment]::GetEnvironmentVariable(
'OTR_COMFY_API_KEY','User'). Plan: docs/2026-07-03-cloud-video-fixes/PLAN.md.
Commits: docs baton (this). Drivers gitignored (_otr_cloud_video_soak.py, _otr_cloud_desktop_probe.py).

## 2026-07-03 day -- TWO CLOUD/VIDEO ROOT CAUSES FOUND (operator review) -- FIX NEXT
Context: operator wants a full CLOUD SOAK (all cloud video x ideo image, high/low pairing) + fix
bugs so razzle etc render; wan PARKED. Desktop app (:8000, PID 55684, `main.py
--feature-flag show_signin_button`) is logged in w/ 2,399 credits. Submitting to it via
scripts/_otr_cloud_desktop_probe.py (no boot/kill -- never kill the Desktop).

**BUG A (THE cloud blocker -- fix first).** Cloud auth FAILS EVEN WHEN LOGGED IN. Live proof
(Desktop run, prompt executed 478s): `cloud_seedream_2 -> CloudMediaError: auth -- no
credentials` at cloud_media_backend.resolve_auth (via otr_image_gen_dispatcher.dispatch_images
-> eng_cloud_image.render_image -> invoke_partner_node -> get_or_create_session ->
resolve_auth). ROOT CAUSE: the OTR nodes that PROGRAMMATICALLY invoke partner nodes
(OTR_ImageGenDispatcher for images; OTR_VideoRenderBatch for video) do NOT declare the ComfyUI
hidden-auth inputs `api_key_comfy_org`/`auth_token_comfy_org` in their INPUT_TYPES, so the
logged-in server never injects the credentials into them -> resolve_auth finds nothing (env
OTR_COMFY_API_KEY also unset). Grep: the hidden-auth tokens live only in cloud_media_invoke/
cloud_media_backend (+ writer's OpenRouter path), NOT in the dispatch nodes. FIX: add the two
hidden inputs to OTR_ImageGenDispatcher + OTR_VideoRenderBatch INPUT_TYPES, capture the injected
values in dispatch()/execute, and thread them to invoke_partner_node/resolve_auth (contextvar or
param). Then RESTART Desktop, regress, and re-run the probe. This unblocks EVERY cloud item
(ideo/seedream/recraft/flux_pro/nano_banana + word_razzle/kling/seedance). /kibitz if the
threading seam is non-obvious.

**BUG B (video "not moving" -- separate).** Operator: heavy-engine finals show STILL frames, no
motion. The final the operator watched is the LEGACY procgen video path: log shows
`[Video] Starting render: ... 1920x1080 ... HUD 'SIGNAL LOST' ... Credits music ... nvenc ...
signal_lost_*_silent_procgen_blended_final.mp4`. That legacy video_engine (HUD + rolling
credits + scopes over the scene STILL) is what renders/ships -- the per-beat MOTION platform
(OTR_VideoRenderBatch: ltx/humo motion clips) is NOT the base layer of the final. Overnight heavy
legs DID burn GPU (ltx 99% 15min) = motion clips were rendered, but the compositor/final ships
the procgen-blended still, not the motion. Trace: episode dir per-beat manifest (are the motion
.mp4s there?) -> the blend/compositor step -> final mux; make the motion clips the base, procgen
the overlay. Distinct from Bug A.
Current step: cloud soak BLOCKED on Bug A (auth wiring). Next: implement Bug A fix + restart
Desktop + re-run scripts/_otr_cloud_desktop_probe.py cloud_seedream_2 still_flat 30; then Bug B.
Commits: none this turn (diagnosis only). Driver: scripts/_otr_cloud_desktop_probe.py (gitignored).

## 2026-07-03 night (overnight coder) -- SOAK BUG FIXED: Ollama daemon was down -> re-launched
Did: FIRST soak run (nightmatrix_012410) FAILED ALL 10 legs -- every episode halted at node 1
(OTR_LedgerScriptWriter / news_interpreter) with OllamaCallFailedError: ConnectionError on
http://localhost:11434 ("actively refused"). ROOT CAUSE: the local Ollama daemon was NOT running
(:11434 empty, no ollama.exe process) -- the gemma-4-12b writer is LOCAL-only and correctly
fail-LOUD (no cloud fallback, by design). FIX (operational, not code): started
`ollama serve` (C:\Users\jeffr\AppData\Local\Programs\Ollama\ollama.exe; model
hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M already pulled, 7.2GB) + warmed it (200 in 7.7s).
Re-launched the soak (nightmatrix_030028, driver2 PID in nightmatrix_driver2.pid): m01 ltx_audio_in
now PAST the writer (0 Ollama errors, cast locked, 5 beats stamped, VRAM ~10GB into the video
phase). HARDENED the driver with an ensure_ollama() preflight (starts the daemon if :11434 is down
before each leg) so a dropped daemon self-heals. OPERATOR/next-window RULE: the gemma writer needs
`ollama serve` running -- if all legs die at node 1 on :11434, start it.
Current step: SOAK RE-RUNNING healthy (driver2, nightmatrix_030028). Watch results.jsonl + otr/obs.
Next: monitor legs; m10 word_razzle still LOUD-fails (no OTR_COMFY_API_KEY). Later passes = mixes.
Commits: none (Ollama fix is operational; driver is gitignored throwaway).

## 2026-07-03 night (overnight coder) -- HEAD 932450f0 (v2.0-alpha) -- SOAK LAUNCHED (detached, running)
Did: CODE PHASE COMPLETE (still_word 097f44ad + word_razzle Phase0 3843bbd0 / Phase1 c914321e,
all pushed, suite 6142/0, Bug Bible 16). Then launched the overnight model-matrix SOAK detached:
scripts/_otr_night_matrix_soak.py (a clone of the proven _otr_visual_soak_6leg harness) -- Pass 1
COHERENT same-model legs, NEWEST video first, force-map video + flux2_klein image + indextts2/bark
voice, 35w/2ch. 10 legs: m01 ltx_audio_in, m02 ltx_video, m03 humo_14B_169, m04 humo_1.7B_169,
m05 wan_i2v, m06 still_word, m07 viz_green, m08 ltx_audio_in+bark, m09 humo_1.7B_169+bark,
m10 word_razzle (cloud -> LOUD-fails without OTR_COMFY_API_KEY, expected). ONE fresh server per leg
(selective :8000 kill), assets -> otr/episodes/<ep>/, final -> otr/obs/, obs Test-Path gate + 1 retry.
VERIFIED LIVE: m01 booted, profile applied, writer patched (gemma 35w), prompt c6d97cbe QUEUED,
rendering. Driver PID in docs/2026-07-03-word-razzle/nightmatrix_driver.pid.
MONITOR: results -> scripts/_otr_soak_capstone_results/nightmatrix_20260703_012410/{soak.log,results.jsonl,server_*.log};
obs finals -> C:\Users\jeffr\Documents\ComfyUI\output\otr\obs. Driver stdout ->
docs/2026-07-03-word-razzle/nightmatrix_driver.out.
Current step: SOAK RUNNING detached (code done+pushed). Next window: monitor results.jsonl + obs;
on any leg NO-obs, read server_<leg>.log, root-cause (golden rules), re-regress+push; later passes =
deliberate MIXES. word_razzle live spike awaits OTR_COMFY_API_KEY.
Commits: 097f44ad 3843bbd0 c914321e (+ baton 2229b761/932450f0) -- all pushed, HEAD==origin.

## 2026-07-03 night (overnight coder) -- HEAD c914321e (v2.0-alpha) -- word_razzle SHIPPED (Phase 0 audit + Phase 1 engine)
Did: (Phase 0 @ 3843bbd0) added a non-mutating --audit-i2v mode to
scripts/otr_pin_partner_nodes.py that walks the whole live comfy_api_nodes catalog for a
promptable non-V3 image-to-video row; live run vs comfy bb131be9 -> VERDICT=CANDIDATE_FOUND
(91 video classes, 36 passing, 29 V3-blocked). Pure classifier + verdict unit-tested
(tests/test_audit_i2v.py, 12). Report checked in (docs/2026-07-03-word-razzle/). (Phase 1 @
c914321e) built the word_razzle cloud i2v engine (eng_cloud_video, node_key
cloud_pixverse_i2v -- the Phase-0 pick: image+prompt+seed+duration+motion_mode). Pinned the
Pixverse row (15/15 OK, live drift --check green). LOAD-BEARING asset_refs fix: _init_image_input
resolves asset_refs['init_image'] first then top-level (real build_request output) -- fixes ALL
cloud i2v engines. Dark/selectable (empty default_roles, no enable flag); family=image_to_video
routes scene-still init via _SCENE_INIT_FAMILIES (mirrors cloud_wan_i2v). tests/test_word_razzle.py
(11) + EXPECTED_ROW_IDS. Suite 6142/0, Bug Bible 16, B7 in-suite.
Current step: still_word + word_razzle DONE + pushed. NEXT = (4) the overnight model-matrix SOAK.
Next: reset box, boot headless ComfyUI on the REAL otr_scifi_16gb_full.json, run 30-45w
full-pipeline episodes Pass 1 = coherent same-model LOCAL combos newest-first, voice indextts2/bark;
assets -> otr/episodes/<ep>/, final -> otr/obs/. Cloud rows (word_razzle etc.) fail LOUD without
OTR_COMFY_API_KEY (unset this window) -- expected; the live word_razzle spike awaits the key.
Commits: 3843bbd0 (Phase 0), c914321e (Phase 1) -- both pushed, HEAD==origin.

## 2026-07-03 night (overnight coder) -- HEAD 097f44ad (v2.0-alpha) -- still_word SHIPPED
Did: built still_word per docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md -- a
model-agnostic still_flat-sibling VIDEO engine (StillWordFamily in cheap_families) whose
base still is minted from a WORD/TITLE-driven prompt: character/announcer beats render the
spoken beat line as a readable word card; music beats an abstract episode-title picture (no
words). Registered in ALL sites: CAPABILITIES row, ENGINE_FAMILY, render_driver :1044
still-init tuple, __init__ (self-registers via cheap_families), + composer branch
(_still_word_roles_from_policy reads image_policy video_models via role_slots; pure
compose_still_word_prompt fails LOUD on blank line/title). NO FALLBACKS: new _CheapFamilyBase
_require_still flag makes still_word fail LOUD on a missing base still (never the dark floor);
still_flat/pan unchanged (byte-identical floor). No workflow-JSON widget change (dynamic combo;
selectable). New tests/test_still_word.py (18 tests). Suite 6114/0, Bug Bible 16, B7 in-suite;
AST+no-BOM verified.
Current step: still_word DONE + pushed. NEXT = word_razzle (animated variant) per
docs/GO_FORWARD_NEXT/2026-07-02-ideo-word-razzle-vid.md (Phase 0 --audit-i2v first).
Next: run the Phase 0 pin audit; if a promptable non-V3 cloud i2v row exists build Phase 1, else
FAIL LOUD / kibitz. Then regress + push; THEN the overnight model-matrix soak.
Commits: 097f44ad (pushed; HEAD==origin).

## 2026-07-03 night -- HEAD fe5a2b38 (v2.0-alpha) -- HAND OFF: CODE EVERYTHING FIRST, THEN the overnight soak
Did: operator called a hand-off + CLARIFIED the order (code first, NOT soak first). Next
window's mission, HARD order: (1) CODE the pending build items -- still_word (per
docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md) + word_razzle (ANIMATED variant, BUILT now, not
name-only) + any queued build-ready items, each wired into otr_scifi_16gb_full.json same-change;
(2) REGRESS (suite + Bug Bible + B7); (3) PUSH green per chunk; (4) THEN the overnight LIVE-GPU
SOAK of many 30-45w full-pipeline episodes across the model matrix (2000 Comfy credits), Pass 1 =
COHERENT same-model-across-roles combos NEWEST-first, voice indextts2/bark, later mixes. GOLDEN
RULES restated: NO fallbacks, NO hidden model promotion, every model works END-TO-END or FAILS
LOUD + gets fixed (root cause); if hung up -> /kibitz for convergence. Reset box (selective CIM
kill) before every headless run; load the REAL json; assets -> otr/episodes/<ep>/, final ->
otr/obs/; Test-Path before success.
Current step: CODE EVERYTHING (still_word + word_razzle + ...) -> regress -> push -> soak (GO_FORWARD section 1).
Next: build still_word per BUILD_PLAN.md (start), then word_razzle; regress + push per green chunk.
Commits: none this hand-off (docs baton).

## 2026-07-03 (later 3) -- HEAD 5b8923fc (v2.0-alpha) -- still_word RE-ARCHITECTURE + kibitz r2 + roundtable CONVERGED (build-ready; NO code yet)
Did: operator pivoted the words feature (ideo_word -> `still_word`): a MODEL-AGNOSTIC VIDEO
engine (still_flat sibling), base still minted by ANY chosen image model (decoupled), prompt
branched by role (char/announcer word-driven from beat line; music abstract episode-title, no
words; pooled-char DEFERRED -- pooling removed). Ran kibitz r1 (codex+antigravity) -> chose
ideo_word-before-B6 + surfaced the role-vs-kind fork; operator resolved to still_word. Ran
kibitz r2 (codex; antigravity credit-bug hangs silent, dropped per operator) + a roundtable
frontier pass (Grok + Gemini; GPT empty-reasoned) ~$0.20. STRONG convergence, all grounded by
me: render_driver ENGINE_FAMILY + :1044 still-init tuple must include still_word (else black/no
still); composer reads image_policy["video_models"] via a _still_word_roles_from_policy helper
(mesh_fodder_roles precedent) -- REJECTED grok's "add video_policy_json input" (already there);
pure compose_still_word_prompt fail-LOUD (no LLM reuse); fail-LOUD no-floor; cut word_razzle to
a name constant; register in 5 sites. Build-ready spec: docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md.
Current step: still_word BUILD-READY -- awaiting operator GO to code.
Next: build still_word per BUILD_PLAN.md (suite + Bug Bible + B7 + push per green chunk); then B6.
Commits: 5b8923fc (kibitz r1 plan doc); this docs batch (BUILD_PLAN + r2 + roundtable pass00).

## 2026-07-03 (later 2) -- HEAD 1bf2a2d2 (v2.0-alpha) -- S1+1 `ideo` shipped; ideo_word + B6 next
Did: added the plain `ideo` cloud Ideogram scene-still engine (node_key cloud_ideogram_v4,
same S1 adapter pattern, no new prompt path) + rendering_speed->USD price map
(TURBO/DEFAULT/QUALITY, env-overridable) + cpu CAPABILITIES row + tests; removed
cloud_ideogram_v4 from the conformance KNOWN_UNADAPTERED xfail (now served). Suite 6093/0 +
Bug Bible 16-pass. Docs baton for S1 core was baedc63d.
Current step: Sprint B -- `ideo` done. NEXT = `ideo_word` (words specialist) then B6.
Next: ideo_word = kind=lyric_card composer path (lyric_text vs title_mood modes) + excerpt
helper + tests + cache wiring + PROMPT_PROFILES tail + title plumbing + adapter/CAPS/guarded
import (docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md). When it joins `ideo` on
cloud_ideogram_v4, make the B5 _engine_by_node_key map iterate ALL engines per node_key.
Commits: 1bf2a2d2 (pushed; HEAD==origin).

## 2026-07-03 (later) -- HEAD b5ef58bc (v2.0-alpha) -- Sprint B S1 stills CORE shipped (B1-B5)
Did: built the cloud stills lane. B3 canonicalize_image (real: validate_partner_result,
require int w/h, PIL sRGB, scale-to-COVER+centre-crop to exact canvas, PNG, sha256).
B1 eng_cloud_image.py = 4 adapters (cloud_recraft/flux_pro/nano_banana_2/seedream_2) on the
reduced ImageEngine protocol, guarded __init__ import, cold-import clean; render_image ->
invoke_partner_node -> canonicalize_image -> str(png_path). B4 cloud_model_ids.py single
source for V3 model ids (resolve_model_id never forwards the placeholder). B2 one cpu
CAPABILITIES row each (consistency invariant green). B5 tests/test_cloud_partner_conformance.py
(billed-row coverage + emitted-kwargs-are-declared over image+video; elevenlabs/sonilo/
stability xfail). B7 verified NO JSON CHANGE (ImageDirector combo is dynamic; engines
auto-appear selectable, defaults stay flux_gen1). Suite 6089/0 + Bug Bible 16-pass + B7
in-suite. New tests: test_cloud_image_adapters.py + test_cloud_partner_conformance.py.
Current step: Sprint B S1 core DONE. NEXT = S1+1 ideo/ideo_word (operator priority) then B6.
Next: ideo + ideo_word per docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md; then B6
portrait-mint 3D gate (lower-urgency, 3D parked). Razzle Phase 0 = safe filler any time.
Commits: b5ef58bc (pushed; HEAD==origin).

## 2026-07-03 -- HEAD 8de5862d (v2.0-alpha) -- Sprint A E1/E2 SHIPPED (atomic rip committed + pushed)
Did: finished + committed the whole no-fallback rip as ONE atomic commit @ 8de5862d
(push fast-forwarded b170e280..8de5862d; HEAD==origin, includes c2edd984 A4+A3b as
ancestor). Verified full suite 6057/0 + Bug Bible 16-pass + B7 5-pass. A3d: node-87
widget[11] allow_auto_fallback audited = already False (no JSON value change needed);
regenerated the untracked otr_scifi_16gb_full_api.json OFFLINE via build_offline_schemas
+ workflow_to_api_prompt (director carries allow_auto_fallback False by name). Validator
+ JSON round-trip + widget audit (15/15) + link integrity all green. A5:
content_oracle.check_manifest reads only path/engine_id/frame_count -- needs no fallback
trails. CAUGHT A REAL GAP: the prior window's `git rm` of nodes/_otr_shared/fallback.py
+ tests/test_video_fallback_chain_additive.py had NOT persisted (both back at HEAD content
on disk; test_video_humo::test_no_fallback_module_and_no_chain was failing in isolation)
-- re-`git rm`'d both in the shipped commit. Pinned-True tests (still_aspect:208,
aseam:316/341/260) already consistent (pass deprecated True as ignored input) -- no change.
Current step: Sprint A DONE. NEXT = Sprint B (S1 stills lane) per docs/2026-07-02-remaining-sprints/PLAN.md.
Next: Sprint B S1 stills (canonicalize_image + cloud image adapters + B5 conformance + portrait-mint gate + B7 wiring); operator pickups QUEUED behind it: ideo_word family (ideogram-lyric-stills = S1+1; razzle-vid Phase 0 = safe filler).
Commits: 8de5862d (pushed; HEAD==origin).

## 2026-07-02 ~late -- HEAD c2edd984 (v2.0-alpha) -- Sprint A: A4+A3b PUSHED; A1+A2 rip CODED (uncommitted, suite verify pending)
Did: A4 triage doc (docs/2026-07-02-remaining-sprints/A4_TRIAGE.md, grep-verified) +
A3b Policy.allow_auto_fallback default False -- committed AND pushed @ c2edd984
(suite 6075/0 + Bug Bible green). THEN coded the A1+A2 ATOMIC rip in the working
tree (UNCOMMITTED -- do not lose): render_driver scaffolding ripped
(FLOOR_NAMES/UNIVERSAL_FLOOR/SYNTH_FALLBACKS/EXPECTED_OOM_TRAIL/make_fallback_of/
run_episode+render_shot+run_real_episode signatures lose fallback_of; soak verifier
rewritten to the NO-TRAIL LOUD contract w/ oom_contract leg; RenderFloorError KEPT --
it is the BUG-LOCAL-413 ltx-open strict guard, unrelated to chains);
nodes/_otr_shared/fallback.py + tests/test_video_fallback_chain_additive.py git rm'd;
retry_taxonomy action API + escalate_to_fallback field deleted (classification kept);
scripts/otr_video_soak.py rewritten; scripts/otr_video_gpu_smoke.py demo/chain cut;
fallback_engine=None on humo x2, ltx_video, mesh_stage, triposr, still_parallax,
character_3d x3; schemas.py runtime_fallback_decisions = retained-slot-stamped-never
comment; ~16 test files rewritten per A4_TRIAGE. Targeted 249/249 green. ALSO started
A3a/c: director widget default False + deprecated labels + runtime IGNORES stale True
w/ LOUD log + policy emits False always; test_route_a_14b_promotion.py:132 flipped.
A full-suite run (a12b_suite.out) was IN FLIGHT when the window ended -- RE-RUN it,
results unreliable (edits landed mid-run).
Current step: SPRINT A -- finish A3a/c/d/e, then verify + commit the atomic chunk(s).
Next: (1) full suite + Bug Bible; fix test_still_aspect_and_labels.py:208 +
test_video_platform_aseam.py:316/:341 (+ check its :260 schema/default asserts);
(2) A3d: audit widgets_values in workflows/otr_scifi_16gb_full.json -- set the
OTR_VideoDirector allow_auto_fallback positional value to false IN THE SAME commit;
regenerate (never hand-edit) the untracked otr_scifi_16gb_full_api.json; (3) A5:
verify content_oracle.check_manifest requires no fallback trails; (4) commit AND push
per green chunk; update GO_FORWARD + this log.
Commits: c2edd984 + 47629f05 (CLAUDE.md quoting rule; accidentally swept the 2 staged
rip deletions) + b170e280 (restores fallback.py + test_video_fallback_chain_additive.py
so origin stays importable) -- all pushed, HEAD==origin @ b170e280. Working tree
carries the uncommitted rip; the restored pair is PRESENT in the worktree again --
`git rm nodes/_otr_shared/fallback.py tests/test_video_fallback_chain_additive.py`
must ride the A1+A2 atomic commit (the rewritten tests assert the module is GONE).

## 2026-07-02 ~23:55 -- HEAD 5c5cdcfe+ (v2.0-alpha) -- kibitz arc CONVERGED + soak2 PASS + proof9d marginal breach
Did: judged kibitz r2 (r2_plan.md), ran + judged r3 and r4 LIVE (codex+antigravity;
operator dropped the claude CLI panelist mid-r4 -- panel is codex+AG from now on,
Cowork Claude = anchor+judge); folded ALL survivors into docs/2026-07-02-remaining-sprints/
PLAN.md -- CONVERGED, BUILD-READY (key r4 catches: A1+A2 must be atomic; no
character_image_model slot exists; audio CAPABILITIES+__init__ import in D3; B5 filters
api_node rows; loudness ref UNRESOLVED in code). soak2 QA PASS (6/6 clips, obs final
46.6MB, no breach). proof9d 832x448 re-run on a CLEAN 1.4GB baseline FAILED MARGINALLY
(14506>14500 MB at shot_b002 ltx_audio_in; 6MB over, zero headroom at this canvas) --
verdict + operator options appended to PROOF9_VERDICT.md; no code changed.
Current step: SPRINT A (E1/E2 rip) per the hardened PLAN.md; S4x GO/NO-GO = operator call.
Next: operator picks proof9d option (a/b/c); coder window starts Sprint A (A4 triage first).
Commits: this docs/kibitz batch (see git log).

## 2026-07-02 ~21:50 -- HEAD d9a659ce (v2.0-alpha) -- no-fallbacks directive + remaining-sprints plan + kibitz r2 launched
Did: operator directive stamped @ d0463b8c (NO fallbacks / NO auto-defaults anywhere; the
shipped workflow JSON dropdown values are the ONLY defaults; S3-full rescoped -- auto-defaults
+ fallback chains CUT; E1/E2 promoted to directive compliance). Wrote the remaining-sprints
build plan @ d9a659ce (docs/2026-07-02-remaining-sprints/PLAN.md: A=E1/E2 -> B=S1 stills ->
C=S3 remainder -> D=cloud TTS -> E=C1 audio_motion_profile -> F=creative formats). Launched
kibitz r2 (full panel, detached) on that plan; MY ANCHOR is written
(kibitz-runs/2026-07-02-remaining-sprints/r2/claude_anchor.md); panel outputs land in the same
dir. Window wrapped before judging (context budget -- honest call, no rushed synthesis).
Current step: judge kibitz r2 (ground panel claims vs code) -> run r3+r4 -> fold the hardened
plan into GO_FORWARD -> then soak2 QA / proof9d / Sprint A (E1/E2).
Next: fresh window resumes at "judge r2"; soak2 QA + proof9d still owed on the render side.
Commits: d0463b8c d9a659ce (+ this entry).

## 2026-07-02 ~21:15 -- HEAD fb23d82d (v2.0-alpha) -- night queue scored + dropdown directive + S5
Did: scored the night queue (PROOF9_VERDICT.md @ 4dd79dbe: proof9c FAILED on a VRAM-ceiling ops
breach at 768x416 -- desktop squatter, no clips; 120w soak COMPLETED 6/6 + obs_publish OK and
scored as the interim S4x verdict -- chars ~3x relative lift vs the announcer anchor; the motion
metric is NOT scale-invariant so the 2.0 bar is canvas-specific). SHIPPED the operator dropdown
directive @ cc349c1d (cloud enable flag REMOVED -- the dropdown pick IS the enable; auth fails
LOUD at invoke; budget unset = $10 default cap, explicit 0 = spend-off). SHIPPED S5 @ fb23d82d
(silent two-stage HQ recipe in eng_ltx_video, unet-family auto-detect, 16 new topology tests).
Suites 6059/0 -> 6075/0 + Bug Bible green per chunk; all pushed, HEAD==origin. Operator opened a
parallel IDEAS window (docs-only, docs/GO_FORWARD_NEXT/).
Current step: QA the 30w soak (~55 min in at wrap) -> proof9d clean 832x448 re-run -> S1 stills
lane (+ the S5 silent-vs-audio-in GPU A/B on the first render window).
Next: QA soak2, reset the box (kill the desktop squatter; baseline <=2.5GB), proof9d, then S1.
Commits: 4dd79dbe cc349c1d fb23d82d (+ this docs commit).

## 2026-07-02 ~19:10 -- HEAD (wrap commit) (v2.0-alpha) -- WRAP for a new chat
Did: committed ia2v_flat_api_prompt.json at probe-final state; swept dead temp probes (this
session's were already gone; pruned prior-session _tmp_* scripts, kept logs); verified the
GO_FORWARD video-team kwargs warning is RESOLVED at a9440980 (wan exact pinned set, seedance
honest dark row, kling pinned; generic conformance test still owed at S1) and stamped that +
the operator-evening S3-hold supersession + the night queue into GO_FORWARD_PLAN.
Current step: NIGHT QUEUE (proof9c 768x416 -> 120w soak -> 30w soak) unattended; morning = score
proof9c + QA soaks, then S5 (silent-LTX HQ port) / cloud order (smokes -> S1 -> S3 full).
Next: morning window runs otr-handoff RESUME, scores the queue per session_handoff.md.
Commits: f9eed360 820f6df3 a415ad18 a9440980 + boot/doc/handoff commits through the wrap.

## 2026-07-02 (night) -- S4b/S4c SHIPPED + cloud-S3 CORE SHIPPED; proof9 in flight; story fix PARKED
Did: **S4b+S4c @ a415ad18** (proof8 verdict: S4 routing fired but portraits were dark PROFILES --
mouth invisible; + the faceless pinprick announcer): per-role `talking` map through
VideoDirector->ImageDirector->MetaBrief; talking character portraits mint FACE-FORWARD frontal
close-up + warm, era/grade tails skipped; OTR_LTX_RADIO_FACE A/B RETIRED into default-on under the
ia2v register (mint + driver, fail-LOUD on a missing face still; env stays for single-pass). 10 new
tests, 7 legacy contracts re-pinned. **Cloud-S3 CORE @ a9440980** (operator evening GO supersedes
the mid-day hold): eng_cloud_video.py 4 rows dark+fail-closed (empty default_roles; assert_usable
needs OTR_ENABLE_COMFY_CLOUD_MEDIA + ffmpeg) via the S0 bridge; canonicalize_video REAL (role
canvas fit+pad, fps, h264/yuv420p/bt709, audio ALWAYS stripped w/ post-strip re-probe PROOF,
actual_duration_s named-error, sha256); docs-window catches folded (wan: NO top-level prompt, exact
pinned static set + OTR_CLOUD_WAN_MODEL; seedance: honest dark row until the S1 V3-expansion pin;
kling pair fully codable). CAPABILITIES cpu rows; requires_flag=None (registry-IS-the-menu guard).
**Operator holds honored**: director-note dialogue-leak fix REVERTED out of production and PARKED in
UpstreamStoryLab GO_FORWARD "DEFERRED STORY-LLM FIXES" (7df7c80 there) -- NO story-LLM changes
until the transplant. **LTX decision (operator ratified)**: 2 LTX rows, no ltx_lowvram (22B unet
floor makes it illusory; 8GB = profiles); S5 = port the two-stage HQ recipe to silent ltx_video
(task open, not started). Ops: proof9 attempt 1 died in the writer (env-less boot -> z_image
FileNotFoundError; fixed w/ tracked scripts/otr_ia2v_server_boot.cmd), attempt 2 died on a
zombie bad-env uv-python holding :8000 (ComfyUI main.py re-execs under uv -- the CHILD outlives a
parent-only kill; kill by PORT OWNER), attempt 3 breached the VRAM ceiling (desktop backend
relaunched, 4.1GB squatter -> killed; baseline 2.4GB, margin TIGHT at 832x448). proof9b launched
~18:12 on the S4b/c code; S4c radio-face fired in production on b000 (log-proven). Suites: 6032/0
-> 6059/0 + Bug Bible green each chunk.

## 2026-07-02 (evening) -- PROOF7 SCORED (announcers SMASH, chars = init bug) -> S4 PORTRAIT INIT SHIPPED
Did: proof7 (signal_lost_lab_race..., 31:19, obs_publish OK, byte-identical, peak ~13.3GB) scored on
RAW clips w/ slice audio muxed (shot clips are SILENT by design -- mux-LAST -- the evaluator needs a
temp AV mux): announcer 4.62/5.51 (canonical ref 3.32 -- the talking register + verbatim prompt +
fixed guide chain fully land), music 3.29 (exempt), but characters 0.62/0.32/1.27 vs the >=2.0 bar.
Driver log self-documented the residual: "conditioning on scene still (landscape; portrait never
used)" -- chars init on WIDE scene stills (face too small to articulate). **Portrait-vs-wide A/B**
(isolation harness, ONE variable, b002's exact slice + exact 202-char production prompt + 832x448):
scene 0.57 (REPRODUCES production 0.62) vs portrait **2.86 lag 0** -- eyeballed frames show
full-frame 16:9 center-crop, NO pillarbox (docs/2026-07-02-canonical-ia2v/pw_ab_*_frame.png).
**S4 SHIPPED @ 820f6df3**: under the ia2v talking register a character_video ltx_audio_in beat
inits on the cast PORTRAIT (init_source=character_portrait_ia2v, LOUD swap log + dims aspect
guard); missing portrait fails LOUD (NO FALLBACK). The 2026-06-20 wide-never-portrait directive is
superseded ONLY under RECIPE_IA2V (canvas-independent guide chain); bookends + single-pass recipes
keep scene stills. 4 new tests; 2 legacy contracts re-pinned. Suite 6019/0 + Bug Bible 16/0.
Verdict + scores + timing: docs/2026-07-02-canonical-ia2v/PROOF7_VERDICT.md; operator side-by-side
side_by_side_proof4_vs_proof7_announcer.mp4. **Proof #8 launched** on the new code (fresh server,
dev-unet env). OOPS note: my server reset's second kill pattern (`main.py`) was too broad and also
killed the Comfy DESKTOP app backend -- relaunch the desktop app when next needed; headless fine.

## 2026-07-02 (late afternoon) -- cloud-S0 REMAINDER SHIPPED: invoke_partner_node bridge + watchdog + gated smokes
Did (baton from the cloud-engines window, per pass04 sec 3): **nodes/_otr_shared/cloud_media_invoke.py**
(f9eed360) -- `invoke_partner_node(node_key, inputs, *, timeout_s, estimated_usd=0.0) -> PartnerResult`.
Backend-owned asyncio loop thread (one daemon for all partner calls); session resolved INTERNALLY
from the prompt context (comfy_execution.utils.get_executing_context -> .prompt_id at utils.py:18;
`bind_prompt_id()` contextvar for headless smokes/tests -- never a parameter); pinned partner class
imported straight off the row's import_path (no NODE_CLASS_MAPPINGS dependency); hidden auth
injected per row + auth.kind (api_key_comfy_org vs auth_token_comfy_org; caller keys never
overridden); watchdog loop ticks 5s, checks comfy interrupt (cancel -> INTERRUPTED) and emits a
ProgressBar heartbeat every 20s (<=30s requirement); outputs normalized to a STREAMED temp file
under session cache_root/partner_tmp (URL download chunked; save_to objects; audio dicts via
soundfile -- torchaudio save is torchcodec-broken; image tensors via PIL) -> validate_partner_result.
Honest budget settlement: provider-said-no codes RELEASE, ambiguous (TIMEOUT/INTERRUPTED/
CORRUPT_OUTPUT) BILL the estimate; ledger_append on both paths. 23 offline tests
(tests/test_cloud_media_invoke.py -- gate, auth injection, interrupt/timeout/heartbeat via seams,
settlement, normalization, real yaml pin-table shape). **Live smokes built + gated**
(scripts/otr_cloud_s0_smoke.py, exit 3 until operator sets OTR_RUN_CLOUD_SMOKE=1 +
OTR_ENABLE_COMFY_CLOUD_MEDIA=1 + OTR_CLOUD_MEDIA_BUDGET_USD + OTR_COMFY_API_KEY): leg1 = recraft
cheap-still auth proof; leg2 = kling avatar CONDITIONED by tests/fixtures/baseline_v1.5.wav (the
whole-point proof). Suite 6015/0 + Bug Bible 16/0; pushed f9eed360 + c9b7fe7d, HEAD==origin, no
BOM, AST clean. Proof #7 (ia2v 832x480 + warm still + talking register; prompt 4d4bade1, episode
signal_lost_lab_race_against_time_20260702_150943) was mid-video-phase at entry write -- scoring
lands in the next entry. Gotcha: the full suite POLLUTES repo-root otr_runtime.log while a render
is in flight (same box, same file) -- read timestamps before declaring a server stalled.

## 2026-07-02 (afternoon) -- LIPS-DONT-TALK kibitz: root causes probed, TALKING prompt register shipped; length dilution = the residual
Did: operator sound-on verdict on the transplant proof ("lips don't talk") -> /kibitz
(kibitz-runs/2026-07-02-lips-dont-talk/, r1-r3 run, panel = codex+antigravity+claude-code w/ my
anchors) + a 6-probe empirical matrix on the WORKING canonical harness (one variable per run,
scripts/_otr_canonical_ia2v_smoke.py env knobs): P1 production canvas 512x288 = TALKS (resolution
EXONERATED); P2 music-only audio = DEAD 0.59 (LTX cannot lip-sync music -- operator's hunch
confirmed; music bookends keep console-motion BY DESIGN); P3 241f@25fps = 2.37 (length halves
motion); P4 production scene prompt = DEAD 1.18 + hallucinated on-video title text (THE KILLER);
P5 misfired (face-forward still, hypothesis open); P6 376f = (this session's last probe).
**SHIPPED: the IA2V TALKING prompt register** (render_driver + eng_ltx_av hook): announcer bookends
swap to a canonical-register lip-sync prompt (M4 OUTRANKED on announcer under ia2v only;
motion-clause override + atmosphere append guarded off; music register untouched per P2);
char-face beats get a compact talking prompt (M4 first-clause fragment + talking clause, <=236
chars; seam-gap no-M4 fallback talks too); engine hook wants_talking_prompt() raises-loud,
driver catches once + memoizes per shot. 10 register tests + 2 legacy tests pinned to
distilled_native; suite 5990/0 x2 (on the combined tree with cloud-S0 chunks), Bug Bible 16/0.
**Live retest "Dialing Disaster" (proof4): PASS wiring** (6/6 ltx clips, 4 register swaps fired,
obs verified) but motion still soft (0.99-1.45) -> probes CONTINUED and killed every parametric
suspect: P6 (376f/15s, canonical rest) = 3.05 TALKS; P7 (512x288 + 376f COMBINED) = 3.32 TALKS.
**The remaining production deltas were OURS, both fixed same chunk:** (1) P8 -- my PARAPHRASED
register wording scored HALF the canonical's (1.72 vs 3.32, identical params) -> the announcer
register is now the canonical text VERBATIM + the char clause mirrors its token pattern ("do not
'improve' the wording" locked in a comment); (2) the engine's guide-image chain scaled 1.5x the
RENDER canvas (at 512x288 -> a 768x432 guide UPSCALED to 1536 = soft double-resampled mouth prior;
every probe passed because the harness kept the canonical FIXED 1920x1088) -> guide chain now
canvas-INDEPENDENT (1920x1088 -> longer-edge 1536, verbatim). PLUS the operator quality catch
(side-by-side docs/2026-07-02-canonical-ia2v/side_by_side_prod_vs_canonical.mp4): ia2v AV canvas
default raised 512x288 -> 1280x720 (canonical-native; P6/P7 live-proven at 376f on this box;
single-pass recipes keep 512x288; env-overridable). Legacy canvas-clamp + announcer-prompt tests
pinned to distilled_native. Ops notes: the window handover killed proof3 mid-render (5 orphan
pending_* dirs); a PS `-Encoding UTF8` rewrite BOM-stamped a test file (caught by the AST-scan
test; stripped -- use [IO.File]::WriteAllText w/ UTF8Encoding($false), never Set-Content UTF8).
CLOSING CHUNK (same afternoon): the canvas ladder walked LIVE -- 1280x720 FAILS the /32 grid gate
(proof5b, LOUD); 1280x704 BREACHED the 14.5GB ceiling in the FULL pipeline (proof6: 14716 MB;
isolation probes carried less resident state; episode killed by the guard at clip ~5, ~33min) ->
**ia2v default = 832x480** (2.6x the old pixels, 1.77x deliverable upscale, base 416x240 all /32,
tests updated). PLUS operator look direction (side-by-side: production reads DARK BLUE + murky):
`ltx_radio_mouth` stills now SKIP the brief palette + grade tail ("cold blue panel glow" / "heavy
vignette, muted color grade") and pin the canonical "warm dramatic lighting" (HuMo styles keep the
brief tail; goldens intact; test_ltx_radio_mouth_still_is_warm_not_brief_blue). Kibitz r4
CONVERGED (claude+codex; agy quota-dead) -- final.md in kibitz-runs/2026-07-02-lips-dont-talk/r4/
(judgment log incl. rejected codex MF4 workflow-JSON ask -- node-87 picks are operator-owned).
Proof7 (832x480 + warm still + all register fixes) launched detached at session end -- score the
speech beats per final.md's verify-at-build (evaluator windows from the slice log; bar 2.0) and
eyeball with sound. NEXT WINDOW: proof7 verdict -> if char beats still soft, run the portrait-vs-
wide A/B then S4; then the queued cloud-S0 remainder (invoke bridge + smokes, see section 1).

## 2026-07-02 (day) -- CANONICAL IA2V TRANSPLANT SHIPPED: ltx_audio_in lip-syncs on ia2v_canonical (new dev default)
Did: operator reviewed the overnight NO-GO evidence + ordered the canonical arbiter: downloaded the
comfy.org "LTX-2.3: Image Audio to Video" workflow, SMOKED IT IN ISOLATION on our box (flattened its
51-node subgraph to a raw API prompt; GGUF dev lane subbed for the unpublished fp8; our corrected
appliance-mouth still + real announcer wav) -> 5s 1280x720 two-stage render in 177s with VIVID mouth
articulation (a4bf2ba6; operator watched with sound: "perfect"). TRANSPLANTED node-for-node into
eng_ltx_av as RECIPE_IA2V = the new DEV-family auto default (f03d2184): TWO-STAGE (motion at half
canvas -> LTXVLatentUpsampler x2 -> audio re-concat -> 3-step 0.85->0 refine), LTXVImgToVideoInplace
0.7/1.0 anchors, audio latent frozen under SolidMask(0), ancestral base + euler_cfg_pp refine,
CropGuides refine-only, distilled-lora-384 @0.5 on dev (half-distilled keeps audio coupling), guide
chain ImageScale(1.5x)->ResizeLonger(1536)->Preprocess(18); ia2v-on-distilled fails LOUD
(double-distill); BUG-414 vae split preserved; i2v REQUIRED. 12 topology-lock tests
(test_ltx_av_ia2v_canonical.py); suite 5934/0; Bug Bible 16/0. Smoke harness fixes (a494e336):
prefix-resolve the decorated dropdown labels + patch the Route-A character_video_model widget (the
saved JSON pins characters to humo_14B_169 -- first proof episode rendered 3/5 clips on HuMo until
patched; NOT a routing bug, a 4th dropdown). LIVE PROOF PASS: "JWST's Gaze" 30w episode, histogram
{ltx_audio_in: 6} across music/announcer/character, obs final verified, ~28.5min (vs ~27min
old-recipe = speed wash; the win is articulation + upsampled sharpness at the same budget). Also
answered the operator's process retro (what we missed: never smoked the vendor reference as a
baseline; normalized the "AMBIENT" label; cut-list had no proof owner; behavioral acceptance beats
structural). yaml: latent_upscale_models folder key added for headless boots. Weights staged:
spatial-upscaler-x2-1.1 + distilled-lora-384 (Lightricks/LTX-2.3). NEXT (operator): watch
proof_jwsts_gaze_ia2v.mp4 with sound; batch_face0_02 was skipped (GPU handover); consider a fresh
120w overnight batch on the new recipe + retiring the OTR_LTX_RADIO_FACE A/B into default-on.

## 2026-07-02 -- talking-radio Sub-plan C: probe RAN, criterion says NO-GO on lip-sync; BUG-415 fixed; overnight batch running
Did: ran the Sub-plan-C matched probe pair live (driver `scripts/_otr_talking_radio_night.py`,
local per the `scripts/_*.py` ignore convention; durable evaluator
`scripts/otr_talking_radio_probe_eval.py` committed). Both legs green: 6/6 clips ltx_audio_in,
obs finals exist (probeA face0 = recorded_mysteries, probeB face1 = jazz_code_cracker; 1
freeze-gate auto-retry on B). Live /object_info captured (99 ltx classes; Sub-plan-A node
candidates confirmed real). PRE-REGISTERED criterion applied (EYEBALL.md): the face-still bookend
shows REAL mouth articulation (closed->open->closed frames) but ZERO transient correlation
(b001 window r1=0.047 vs threshold 0.35; delta vs control 0.037 vs 0.15) => **NO-GO on lip-sync
as measured; HuMo stays the face path; Sub-plan A NOT built** (contract). Caveats in EYEBALL.md:
probe still was the strongest-possible mouth prior (see below), and the one cheap re-probe knob is
the dev unet (OTR_LTX_AV_UNET) -- operator's morning eyeball may still take the uncorrelated
mouthing as a LOOK (creative GO separate from lip-sync). TWO live catches fixed at root this night:
**BUG-LOCAL-415** (crash-orphaned `_marathon_extra_env.cmd` forced `*=humo`+HUMO_HOSTS onto every
headless boot -> probe A attempt 1 rendered all-HuMo; launcher now CONSUME-ONCE; promoted to Bug
Bible 12.47 @ survival-guide 8911c43 w/ Three-File Contract) and the **ltx_radio_mouth HUMAN-FACE
leak** (image dispatcher has NO negative channel -> "no human" negative inert -> z_image minted a
literal screaming human face in the radio; positive now MATERIAL-ANCHORED, d87f8fc5). Operator
overnight batch (his ask, running unattended as of ~01:50): 6x120w all-ltx_audio_in face1 (the
CORRECTED appliance-mouth stills) + 2x120w face0 + 50w/100w all-humo_1.7B_169 (HUMO_HOSTS on);
results stream to docs/2026-07-01-talking-radio/night_results.jsonl, finals to otr/obs.
Gates: suite 5922/0 x2, Bug Bible 16/0, B7 in-suite; pushes 55e35468 + d87f8fc5 (+ bible 8911c43).

## 2026-07-02 -- talking-radio Sub-plan B: LTX-only mouth-forward still, SPLIT from HuMo (SHIPPED)
Did: executed Sub-plan B of the talking-radio contract (`kibitz-runs/2026-07-01-talking-radio/r1/
final.md`; order B->C->A). NEW style `ltx_radio_mouth` (`_RADIO_CONSOLE_MOUTH`) in
nodes/otr_meta_brief_image_prompt.py: brief-driven form + PROMINENT rubbery grille-mouth right
after the form noun (LTX-2.3 has no face detector -- it drives whatever reads as a mouth); used
ONLY by the OTR_LTX_RADIO_FACE still mint, BOTH bookend roles (operator: the stills the EXISTING
ltx_audio_in gets -- no new video model/path; supersedes the per-role HuMo-parity note). Negative =
RADIO_CONSOLE_NEG (no human; no baby line -- no person in frame). HuMo looks UNTOUCHED
(console_face/radio_head_person, incl. the announcer portrait row): pinned byte-for-byte by 5
goldens captured live from the PRE-split tree @ 5cce9c2 (tests/test_brief_radio_host.py). Mint
split test proves both toggles on => ltx stills carry the mouth, radio_host_portrait +
announcer do NOT. test_ltx_radio_face_ab.py mint assertions deliberately updated to the mouth
style. Default-off byte-identity untouched (flag-gated mint only). No workflow-JSON change
(env-gated, no node/widget). Gates: suite 5922/0 (35 skip), Bug Bible 16/0, B7 in-suite green.
NEXT = Sub-plan C: live one-beat probe (real /object_info capture, OTR_FORCE_ENGINE_MAP,
OTR_LTX_RADIO_FACE=0/1 side-by-side, written transient-correlation criterion ->
docs/2026-07-01-talking-radio/EYEBALL.md; GO/NO-GO operator-gated).

## 2026-07-01 -- RIP dead sfx subsystem + scene_broll/background_abstract + pooling (SHIPPED)
Did: executed the rip contract (`kibitz-runs/2026-07-01-rip-sfx-broll/r2/final.md`; build plan
kibitzed to convergence r3 wiring + r4 -- codex + claude-code panels, agy credit-dropped; folds in
`docs/2026-07-01-rip-sfx-broll/BUILD_PLAN.md` + r3/r4 judgments). ONE atomic commit: speaker-role
model -> 5 (sfx GONE: constant/Literal/prompt/sfx_cue field+ctors/SOUND IN THE ROOM/[SFX:] token/
sentinels/G7+SFX_DUR/writeback fields/sequencer overlay+offset widgets/HUD+treatment branches);
video Role enum -> 3 (announcer_visual/music_visual/character_video; scene_broll +
background_abstract slots/widgets/aspects/profile keys/engine role-tuples ripped; legacy
other_beats_video_model slot = character-only migration lane); other-beats pool_n_loop POOLING
gone (per-beat budget/stills; still_pool_key reads deleted). NO FALLBACKS: 6 silent sites now
raise (resolve_speaker_role, stamp_default_role, slot_for_role/engine_id_for_role,
_video_role_for_line, derive_scene_still_targets, sequencer dispatch); old sfx ledgers rejected
LOUD (freeze ALLOWED_SPEAKER_ROLES + every path). Workflow JSON same commit: node 87
widgets_values 19->15 (wholesale; mid-list idx 6-7 + tail 17-18) + 4 dead inputs; node 3 sfx
inputs + tail widget dropped + LINK 2 dst_slot 3->2 (script_json shift -- r3 codex catch);
validator tombstones sfx_audio_clips/sfx_offset_ms; widget_mapping.json drops the 2 dead
role_overrides. slot_matrix renamed (ALL_ROLES / build_all_role_profile, no aliases); soak
producers + coverage-sweep updated (character lane explicit); scripts/_otr_patch_pool_default.py
deleted. Tests: test_per_cue_sfx_dur.py + test_fixture_dur_s_audit.py deleted; ~30 files
rewritten to the 3-role model; NEW tests/test_rip_sfx_broll_guard.py (12 guards incl. workflow
pin + kept-widget + engines_for_role non-empty). Gates: suite 5916/0 (35 skip), Bug Bible 16/0,
B7 in-suite green, workflow contract+round-trip+widget-count+link/slot audits pass.

## 2026-07-01 -- migrated done-history out of GO_FORWARD_PLAN.md (lean-plan cleanup)
Did: split GO_FORWARD into forward-only plan + this log (operator: "GO_FORWARD must be lean, point
to sprint specs, not a change-log"). Updated the otr-handoff skill to append here instead of the
big otr-build-tracker dashboard.
Note: entries below this line are the ONE-TIME migration of shipped receipts that had accumulated
in GO_FORWARD; going forward each session appends its own entry above.

## 2026-07-01 -- brief-driven HuMo radio-host + ltx_audio_in A/B addendum (SHIPPED)
Did: shipped the brief-driven radio-host feature (contract `docs/2026-07-01-brief-driven-radio-host/
PLAN_HARDENED.md`, kibitz r1): deterministic `radio_form_from_meta` + `build_radio_host_prompt`
(no LLM), repointed the 4 hardcoded-1940s surfaces, minted the toggle-gated `radio_host_portrait`
object (seed-pinned, aspect-follow, no-baby neg), `OTR_ENABLE_HUMO_HOSTS` toggle (default OFF =
byte-identical), skip-LLM synthetic announcer + widened `_passes_consistency`. Then the
`OTR_LTX_RADIO_FACE` A/B addendum (wide radio-face stills + ltx_audio_in bookend init + HuMo-hosts
precedence). Two follow-up fixes: HuMo host bookends use the ambient master-mix as audio_ref (was
FamilyInputGap on b000); image-director `mesh_fodder_roles_from_video_policy` honors
OTR_FORCE_ENGINE_MAP. LATER look-direction (operator eyeball): music HuMo -> anthropomorphic radio
CONSOLE (dial-face), announcer HuMo -> RADIO-HEAD PERSON, overtness brief-driven, story-flair via
full "still" era tail. Commits bb972ddf / 4cddb26c / ad48246d / 147e83cf / 4046a50c / 30e492d2 /
14d5fbbf; suite 5943 + Bug Bible + B7 green; pushed v2.0-alpha.
Note: a 6-leg 45-word visual soak driver (`scripts/_otr_visual_soak_6leg.py`, gitignored) was built
+ debugged (output-tree path fix + video-via-force-map); live GPU legs are operator-run.

## 2026-06-30..2026-07-01 -- slot-audit C0-C5 + engine cleanup (SHIPPED, code)
Did: C0 retired station_card + abstract engines (8f701a73); C1 accepts_still on StillPan/StillMotion
(8f701a73); C2 VideoEngineRegistry capability routing (65c11bc1); C3 sfx->scene_broll route
(96aa54dc); C4 eligibility matrix test (ca2ac0e8); C5 content_oracle + slot_matrix all-5-role
profile (f5b78ac5). Plus: viz_mxc_mandala engine (8d90562a); still_parallax rip-out + visualizer->
viz_green rename (2026-06-30); HuMo improve items 1/2/4/5 incl. _enforce_radio_is_host (2026-06-30);
mesh_stage MIN-ACCEPT radio subject + adaptive camera (2026-06-30); E4 audio-reactive dropdown
descriptor + VRAM-tier suffix (dfacea49). S-A..S-F coverage-soak: 7 of 11 sub-items already shipped
(S-A 4e13a692, S-B eb8c3781, S-D 5a50fa40, S-E E5 9e4f3a33 + E6, S-F c6c50579, BUG-411 verified).
REMAINING (open, forward): E1 no-fallback scaffolding migration, E2 deprecate allow_auto_fallback,
E3-doc, S-C C1 audio_motion_profile, the live-GPU all-engines soak RUN.

## 2026-06-13..2026-06-14 -- Wan + GATE-A sweep hardening (SHIPPED, code)
Did: M1+M4 no-fallback + VRAM gate (9b2294b); M2+M3+M5 sweep --acceptance (0ab55bc); M6 assert_usable
preflight (ec91a3c); M7 silent-clip ffprobe (f71edaa); S1+S5 (dfe9ab5); S7+S10 (f3a529f); M8+S2
wan_ti2v engine built (bcbe05a). OPEN: M9 (CS-3 sequential residency) + S4 + S9 = live-GPU proofs;
full --acceptance GREEN gated on the slow wan-music-bed leg (attended).

(Older shipped history remains in `docs/GO_FORWARD_ARCHIVE.md`.)
