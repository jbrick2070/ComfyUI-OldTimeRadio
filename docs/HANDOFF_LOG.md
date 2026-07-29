# OTR Handoff Log

Append-only session log, newest at top. What each session actually did;
GO_FORWARD_PLAN.md stays lean and forward-only.

## 2026-07-29 -- HEAD 69daf4fe (v2.0-alpha) -- WINDOW CODER (wiring block, cont.)
Did: two more green pushed chunks, and the FIRST of them corrects the previous
  one. Re-reading r4/A4 before starting W7 turned up a deviation I had shipped.
  **`4cc76806` WIRE-W4c -- the trimmed tail is SILENCE.** The ratified contract
  is "conditioning WAV duration EQUALS render_frames; copy only the
  visible_frames source interval and APPEND SILENCE for trim_tail_frames --
  never speech from the next segment." W4b (cb6fafc7) took the whole
  render_frames window straight off the master. It LOOKED harmless -- the
  trimmed frames are discarded at assembly, so nobody sees them -- and it is
  not, because the AUDIO ENCODER SEES THE WHOLE WAVEFORM before a single frame
  is sampled: speech from the next beat sitting in the tail conditions the
  frames that DO survive. On the pinned 184 case that is 2 frames of the next
  line leaning on a 31-frame take.
  segment_render_window now returns SegmentAudioWindow(offset_s, copy_s,
  pad_s); total_s still equals render_frames, which is the generation length
  and is unchanged. _slice_master_audio grew pad_tail_s and builds `-af apad`
  plus an OUTPUT `-t` of the total -- the PAIR is the contract, because apad
  alone never terminates and a bare output -t would just re-cut the source. It
  fixes the far end for free too: a window running past the END of the master
  now pads to length instead of emitting a short WAV.
  **The pad is IN the cache key and SLICER_VERSION moved 2 -> 3.** Two segments
  can copy the identical source interval and owe different silence, so a key
  that ignored the pad would serve the first one's WAV to the second; and every
  WAV already on disk describes the OLD contract for the same (master, start,
  dur). The slicer also honours OTR_FFMPEG now -- it used the bare literal
  while otr_credits_roll already honoured the config, so on a box where ffmpeg
  is configured but not on PATH the credits rendered and the slice silently
  returned "", which reads downstream as "this beat has no voice line" rather
  than "this box cannot slice". Mutation 11/11.
  **`69daf4fe` WIRE-W4d -- the requests are built BEFORE the lease is taken.**
  r3: "Prebuild and validate all segment requests and audio slices BEFORE
  entering BeatSession; only terminal-image chaining stays in the render loop."
  The builder is neither cheap nor pure -- it resolves stills off the ledger
  and shells out to ffmpeg per segment -- and it was running with the
  cross-process GPU lease held and a 14B UNET resident, between renders. Every
  other heavy render on the box blocks its full 120 s acquire behind that. It
  is also where a bad request SHOULD surface: a builder that raised on segment
  2 used to do it after two completed renders and a 6 GiB load, and the test
  now proves prepare() never runs. The chain's terminal-frame substitution
  stays in the loop by design (segment N's init image is segment N-1's last
  RENDERED frame). Behaviour is otherwise identical -- same builder, same
  arguments, same order; only the timing moved. Mutation 4/4.
Process note worth keeping: BOTH of these came from re-reading the ratified
  r3/r4 finals before starting the NEXT chunk, not from a test going red. The
  suite was green on the wrong contract. Re-read the spec at the start of every
  chunk, not just at the start of the block.
Two stale test fakes updated, not silenced: the per-beat-audio slice fakes took
  (path, start, dur, master_hash) and would have failed as a TypeError inside
  build_request_from_shot -- which presents as "the slice failed", not "this
  fake is stale". And the two new argv tests isolate the slice CACHE to
  tmp_path, because the cache lives under the shared episode tmp dir and a
  second run would take a cache hit, skip ffmpeg, and assert against an argv
  that was never built.
Current step: **WIRE-W7 -- mouth-still ownership.** r3 MUST-FIX 11: no W1-W6
  step enforces the operator's three rulings, and an unowned ruling silently
  lapses. The house rule is at GO_FORWARD:77 verbatim -- "THE SET SPEAKS BY
  DEFAULT; A FACE MUST BE OVERHEARD ... One face per episode at most". Surface
  mapped: image rows carry kind / object_id / char_id / beat_id; a RADIO face
  is object_id == "radio_host_portrait", object_id.endswith("_radio_face_169")
  or kind == "scene_open" (otr_image_gen_dispatcher.resolve_object_seed:141-153
  already special-cases exactly those three); a HUMAN face is kind ==
  "portrait" with a char_id in the cast. The three live radio styles are
  console_face / ltx_radio_mouth / radio_object (_RADIO_HOST_STYLES,
  otr_meta_brief_image_prompt:282). ShotLock is the natural owner: it already
  stamps the coverage plan, and _assert_family_inputs_satisfiable_cast_time
  (otr_shot_lock:909) is the per-beat preflight that runs before build. The
  EPISODE-level cardinality belongs after build_execution_plan (:1256).
  **NOTE the still_plan schema has NO "bears a mouth" field, and adding a token
  to those closed enums is explicitly an operator decision, not a coder's** --
  so W7 should derive the answer from the frozen ROUTE, not extend the schema.
Next: WIRE-W7 -> WIRE-W5 (grade SOURCE COMPONENTS BEFORE OVERLAYS; it can now
  read native_frame_count/extension_mode off the manifest) -> the 45-word run
  over all 18 local video/still engines.
Filed, not built: the durable slice RECEIPT (source PCM hash, segment index,
  start sample, sample count, rate/channels, output PCM hash) under the
  canonical episode directory rather than tmp. Telemetry for W5, not
  correctness.
**NOTHING IN THIS BLOCK IS LIVE-PROVEN. Suite and contract only.**
Suite 7665 -> 7679 passed / 36 skipped / 1 xfailed; Bible 17; build_variants
  --check 11 variants / 0 failures; validate_workflow_links 0 violations;
  canonical 9872624A byte-identical at both commits.
Models: Claude (rung 4) only. No Codex spend -- two-strikes never fired.
Commits: 4cc76806, 69daf4fe (+ this handoff).

## 2026-07-29 -- HEAD cb6fafc7 (v2.0-alpha) -- WINDOW CODER (wiring block, cont.)
Did: four green pushed chunks -- the operator's motion-floor ruling, WIRE-W3b,
  WIRE-W4a and WIRE-W4b.
  **`2d20d915` THE MOTION FLOOR + THE CREDITS EXCEPTION.** Operator ruling:
  video for every beat, and if an engine's minimum is four seconds then render
  four seconds. AUDITED: that behaviour has been shipped since 2026-07-25 --
  partition_beat renders the smallest legal length at or above the target and
  trims, so all 31 engines cover a 1 s beat with real video and google_veo
  renders 100 frames (4.0 s) and trims 75. Nobody had written it down, and one
  `allow_tail_trim=False` on a future video adapter would silently reopen the
  still floor, so tests/test_motion_floor_roster.py is now the roster gate that
  fails BY NAME. The CREDITS question is CLOSED by the operator's own words (a
  still, a ping-pong or plain black is fine) -- no eyeball owed, no work queued,
  and "never credits-over-black" is relaxed. Kibitz r1 also found that the
  first-ever live green episode was `{"still_flat": 7}` -- seven dead-flat
  stills that passed every gate we own. Not a defect (still_flat is a declared
  still route) but **nobody may cite that leg as proof the VIDEO lanes work.**
  **`439ce8c7` WIRE-W3b** -- wan_ti2v's session plus the ping-pong NARROWING.
  The mirror-extend stays on the single-clip path (the shipped 8 GB tier,
  PBUG-20260723-02) and is forbidden inside a coverage plan, because the pad
  wears the right frame count: a render that did not happen passes
  render_driver's `got != segment.render_frames` gate wearing the number of one
  that did. Discriminator is `prepared["session_ctx"]["multi_clip"]`, the only
  honest one available -- a planned segment's REQUEST is shaped exactly like a
  single-clip beat's. Brought eng_ltx_8gb's B4 pipeline invariant with it (a
  decode that returns a different count than the ask now RAISES), which
  immediately caught that test_wan_recipe_freeze's own fake decoder had been
  emitting 4 frames for a 33-frame ask since it was written -- that file had
  been exercising the PAD on every render. native_frame_count +
  extension_mode now ride every WAN receipt into the manifest.
  **The r3 warning about the budget was real and load-bearing:** the cost
  model's `overhead` is "the resident model + fixed buffers", so hoisting the
  UNET moves those GB out of *free* before `_floor_length` reads it and the
  same weights get charged twice -- MotionBudgetError would refuse renders that
  fit. prepare() now MEASURES the hoist (free VRAM either side of the loader
  graph) and hands the delta to every segment. Without that the session half
  BREAKS the lane it fixes. Mutation 16/16.
  **`5a1ee2de` WIRE-W4a** -- all four HuMo tiers get a beat session. The hoist
  is WIDER than the WAN lanes (UNET + LoRA + umt5 + VAE + whisper) and that is
  a property of the family: HuMo renders FULLY RESIDENT by contract (BUG-265),
  so hoisting changes how many times a loader is READ, not how much is held.
  **The reclaim is the other half and neither works alone:** the LOUD
  reclaim_idle_models exists "so the resident stack drops back down before the
  NEXT SOAK BEAT starts", and run between two segments of ONE beat it would
  detach(unpatch_all=True) the very handles prepare hoisted -- load count still
  reads 1 while the weights bounce to CPU and back. Skipped between segments,
  run once at teardown. Mutation 16/16.
  **`cb6fafc7` WIRE-W4b** -- a lip-synced segment is driven by its OWN audio.
  Every segment used to get the WHOLE beat's slice, so a 3-segment HuMo beat
  rendered three clips all lip-syncing to the same waveform FROM THE TOP: the
  assembled beat said the opening of the line three times, and nothing caught
  it because every clip had the right frame count and the right still. The
  arithmetic is `coverage_plan.segment_render_window` (pure); render_driver
  adds the beat's own start_s. It is the RENDER window, not the visible one --
  a chained successor renders one frame earlier than it contributes, and the
  visible window would put every chained segment's mouth a frame ahead of its
  own audio. Mutation 10/10 + 1 documented control.
Found and recorded, not built: the negative-offset clamp in
  segment_render_window is a MEASURED mutation CONTROL -- unreachable because
  validate_coverage_plan already refuses a first segment with a drop_head. Kept
  anyway; the alternative is a negative ffmpeg seek.
One test was CORRECTED, not silenced: test_ltx_8gb_session_identity's CONTROL
  asserted that wan_ti2v alone had no session identity, which made it a control
  over exactly one engine -- wan_i2v gained one at WIRE-W3a and nothing in that
  file noticed. It now asserts the whole SET against a named list carrying the
  chunk that added each entry, and it fired correctly at W4a.
Current step: **WIRE-W7 -- mouth-still ownership.** r3's MUST-FIX 11: no W1-W6
  step enforces the operator's three rulings, and an unowned ruling silently
  lapses. Needs an explicit OWNER (verify in otr_meta_brief_image_prompt.py,
  otr_image_director.py, otr_image_gen_dispatcher.py) plus LEDGER-LEVEL
  CARDINALITY CHECKS before build. Surface already mapped this session: image
  rows carry `kind` / `object_id` / `char_id` / `beat_id`; a RADIO face is
  identifiable by `object_id == "radio_host_portrait"`, `object_id.endswith(
  "_radio_face_169")` or `kind == "scene_open"` (see
  otr_image_gen_dispatcher.resolve_object_seed:141-153, which already special-
  cases exactly those three); a HUMAN face is `kind == "portrait"` with a
  char_id in the cast. The three live radio styles are console_face /
  ltx_radio_mouth / radio_object (_RADIO_HOST_STYLES,
  otr_meta_brief_image_prompt:282). ShotLock is the natural owner -- it already
  stamps the coverage plan and runs before build.
Next: WIRE-W7 -> WIRE-W5 (the grader; it must grade SOURCE COMPONENTS BEFORE
  OVERLAYS -- kibitz r1 proved a whole-frame motion check passes a frozen
  backdrop because the overlay moves, test_credits_roll_spec.py:446-470 -- and
  it can now read native_frame_count/extension_mode off the manifest to reject
  a ping-ponged clip on a lane claiming real multi-clip). THEN the 45-word run
  over all 18 local video/still engines, which is the operator's stated first
  priority and the only thing that proves any of this.
**NOTHING IN THIS BLOCK IS LIVE-PROVEN. Suite and contract only.**
Suite 7551 -> 7665 passed / 36 skipped / 1 xfailed across the four chunks;
  Bible 17 throughout; build_variants --check 11 variants / 0 failures;
  validate_workflow_links 0 violations; canonical 9872624A byte-identical at
  every commit -- no node, widget, link or schema touched.
Models: Claude (rung 4) only. No Codex spend -- two-strikes never fired.
Commits: 2d20d915, 439ce8c7, 5a1ee2de, cb6fafc7 (+ this handoff).

## 2026-07-29 -- HEAD 3e89d6b2 (v2.0-alpha) -- WINDOW CODER (wiring block, cont.)
Did: **WIRE-W3a `3e89d6b2`** -- wan_i2v's beat session. session_identity() and
  the UNET-only hoist in ONE commit, because codex's r3 warning is real: the
  identity alone silences BeatSession's refusal and the segment graph still
  runs UNETLoader every segment, so the beat would look fixed and reload a 14B
  three times. Acceptance counts LOADER INVOCATIONS, never prepare() calls --
  BeatSession carries no counters for exactly that reason. Measured: 3-segment
  beat = 1 UNET load, 3 CLIP loads, 3 VAE loads. The auxiliaries reloading IS
  the narrowed contract; hoisting the CLIP would pin ~9 GB and delete the
  free_after_use mitigation that keeps this lane off a 14,499 MB peak.
  Identity carries the recipe, the loader MODE and a size+mtime receipt for
  every loader file INCLUDING the un-hoisted CLIP and VAE (r4/A5 -- TI2V
  distinguishes incompatible VAE generations). Receipt mechanism shared in
  wan_shared, data per adapter; eng_ltx_8gb keeps its own copy on purpose.
  Suite 7561 / 27 / 1; Bible 17; canonical 9872624A; mutation 7/7 with M1
  being the trap itself.
Current step: WIRE-W3b (wan_ti2v). The session half mirrors wan_i2v almost
  exactly; the NEW half is the ping-pong -- _floor_length + the extend at
  render_clip:725-733 must be suppressed for a COVERAGE-PLANNED segment only
  (it stays load-bearing for the shipped 8GB tier, PBUG-20260723-02), the
  native frame count and extension mode go on every receipt, and the native
  budget is computed AFTER prepared-model residency.
Next: WIRE-W3b -> WIRE-W4 -> WIRE-W7 -> WIRE-W5, then the 45-word run over all
  18 local video/still engines. NOTHING in this block is live-proven yet.
Models: Claude (rung 4) only. No Codex spend -- two-strikes never fired.
Commits: 3e89d6b2 (+ this handoff).

## 2026-07-29 -- HEAD a14ecdfa (v2.0-alpha) -- WINDOW CODER (wiring block)
Did: WIRE-W1, WIRE-W2 and WIRE-W6 built, gated and pushed, one green chunk at
  a time, from r3/final.md as amended by r4/final.md.
  **WIRE-W1 `5efd2baf`** -- partition_beat ran TWO walks over the segment count
  (exact at every count, then trimmed at every count), so an exact cover at a
  HIGH count beat a trimmed cover at a LOW one. A 184-frame HuMo beat planned
  [85,33,33,33] because 184 is 0 mod 4 and an exact cover needs a count
  divisible by 4, while [153,33] trim 2 was legal at count 2 all along. One
  walk now. Differential over 798,510 (contract,target) pairs / 2,538 contract
  shapes: 46,949 plans changed, EVERY ONE a count reduction, zero refusals
  introduced, zero increases. Mutation 9/9, controls 5/5.
  **WIRE-W2 `a218b1f7`** -- DeferredImageGapError(RenderError) in the new leaf
  nodes/_otr_video_engines/render_errors.py; RenderError moved with it and is
  re-exported. Five cast-time sites declare themselves deferrable, three
  post-image and two wrong-aspect sites stay terminal, and BOTH fail-open
  swallows in ShotLock's cast-time preflight are deleted. Mutation 6/6.
  **WIRE-W6 `a14ecdfa`** -- the credits backdrop is the body video's frozen
  final frame; plan_backdrop DELETED (it read the clip manifest, which is why
  an all-mesh_stage episode rendered 7/7 and published nothing). Terminal vs
  presentation-only boundary per r4/A7. Mutation 4/4.
Found and NOT built (filed in OPEN BUGS): the fewest-segments rule can accept a
  disproportionate trim on a wide DISCRETE menu -- a bound was written,
  MEASURED (it made 4,885 grid cases worse) and REVERTED; unreachable on any
  shipped contract. And the B7 forbidden sweep only diffs TRACKED files, so a
  new test file passes its gate and fails the commit after -- that cost one red
  HEAD this session and is written down so it costs nobody else one.
The fan-out paid three times, all on my own new code: the WIRE-W1 property test
  compared segment COUNT only (a reversed ladder fill order passed it with 0
  mismatches over 27,954 plans), its floor was 500 against a real 27,954, and
  the trim bound above. All three fixed before the push.
Current step: WIRE-W3 (WAN) -- UNET-only hoist, VAE in the session identity,
  external_results injection, teardown dropping external refs before base
  release, native-frame-count + extension-mode receipts, and ping-pong
  suppressed for coverage-planned segments ONLY (it stays load-bearing for the
  shipped 8GB WAN tier).
Next: WIRE-W3 -> WIRE-W4 -> WIRE-W7 -> WIRE-W5, same window rules.
Models: Claude (rung 4) + a 3-lens Sonnet fan-out per chunk (rung 4, cheap) +
  one general-purpose read of the superseded GO_FORWARD region. No Codex spend
  -- no chunk needed a third attempt, so the two-strikes law never fired.
Commits: 5efd2baf, a218b1f7, a14ecdfa (+ this handoff).

## 2026-07-29 -- HEAD ead920d2 (v2.0-alpha) -- OPERATOR: LEAN-MEAN OFF GO_FORWARD
**Operator direction, same session as the plan repair below:** "Lean-mean
should only come after the randomization and the SFX. In fact, maybe just put
lean-mean back onto the roadmap and not on the go-forward plan." Both halves
executed.

**GO_FORWARD no longer carries lean-mean in any executable form.** Removed: the
two "Big blocks" entries (FRONT and TAIL) with renumbering; both lines from the
Coder queue fence; the CODER D and CODER G packing rows (struck through, gates
voided, "do not re-add this row" on each); the full `r2 -> r3 -> r4` operator
pin from the STANDING RE-GROUND GATE; items 5 and 6 of the live-order list;
CODER F's "after D" gate. The 07-24 rescue paragraph's order line was already
struck; it now reads "IS NOT ON THIS PLAN AT ALL". Every surviving mention in
the file is a pointer, a banner, or a struck historical line -- verified by
grep, 40 hits, none of them a queue position.

**ROADMAP.md is now its only home, and it moved DOWN there too:** order 1 ->
order 3, behind SFX and product expansion, ahead of RunPod/install and the v2
release. That last part is deliberate and worth keeping: validating an install
path and tagging a release against a tree still full of dead code would have to
be redone after the rip. Section headings renumbered to match the table, and
GO_FORWARD's nine `ROADMAP.md section 1` cites were changed to a NAME-based
reference so the next renumber cannot break them.

**NOTHING WAS LOST IN THE MOVE.** The new ROADMAP section carries the FRONT and
TAIL chunk chains, all six required edges, the full `r2 -> r3 -> r4` operator
pin with its reasoning, the panel composition and the Fable single-gate, the
drift-check items that fold into the r2 brief, the W2 MIGRATION-FIRST mandate
with its `otr_image_director._is_3d_engine:109-119` /
`tests/test_image_platform_c1.py:339-352` cites and its boundary question, the
ENGINE_MATRIX W6 sub-step spec, the `1a6ae8f1` do-not-re-delete note, and the
never-interleave rule. A reader who only ever opens ROADMAP can still run the
campaign.

**AND ONE DEPENDENCY INVERTED, WHICH IS THE EASY THING TO MISS.** The SFX
section said it was "parked until the 720-word runway and lean-mean campaign
land." SFX now runs BEFORE lean-mean, so that sentence was backwards the moment
the order changed. Fixed with the reversal named explicitly. **When a block
moves, grep for other sections that declared a dependency ON it -- the moved
block updates itself; its dependents do not.**

## 2026-07-29 -- HEAD 078dd2d3 (v2.0-alpha) -- WINDOW RENDER/QA -- PLAN REPAIR
**A CODING WINDOW OPENED ON LEAN-MEAN. THE PLAN TOLD IT TO.** The operator
caught it; the doc, not the window, was wrong. Four independent places in
GO_FORWARD still ordered LEAN-MEAN FRONT **second**, and one of them claimed
supersession authority over the whole file:

1. `CURRENT STEP` itself -- headed "the second encoder is CLOSED ... what is
   left is the operator's own GPU sequence", and its closing paragraph
   recited the 07-24 order. **This is the line every window boots on.**
2. The **OPERATOR RESCOPE 2026-07-24** paragraph -- "(supersedes the older
   queue everywhere in this file)". True on 07-24, and the single most
   misread sentence in the document. A later operator direction outranks it;
   nothing said so.
3. The **Coder queue** fence, still headed "re-grounded 2026-07-24".
4. The **Window packing** table: CODER A reads "THE CODER-WINDOW BLOCK IS
   COMPLETE" and CODER D's gate read "after A". A -> complete, therefore D.
   D is "lean-mean front". The window's inference was sound.

**AND A FIFTH CAUSE NOBODY HAD NAMED: THE CHUNK NUMBERS COLLIDE.** The wiring
block's chunks are W1..W7 in `r3/final.md`; LEAN-MEAN FRONT is W0..W8 and
LEAN-MEAN TAIL owns a W8. A kickoff saying "start with W1" is ambiguous
across THREE blocks. Every wiring chunk is now written `WIRE-W1`..`WIRE-W7`
in GO_FORWARD, with the collision stated in the row and in CURRENT STEP.

Fixed: CURRENT STEP rewritten to name the wiring block, its cause taxonomy,
the `WIRE-` order and the three operator rulings; superseded banners on (2),
(3) and both stale gate cells; a new **CODER W "local-engine OBS wiring"**
row placed FIRST in the packing table and marked THIS IS THE OPEN SLOT; the
generic kickoff line changed from "you are CODER WINDOW A -- swap the letter"
to boot-by-CURRENT-STEP, with "CURRENT STEP WINS over a letter row" stated in
the pasted text.

**DOCTRINE, and this is the second time in two days the same bug bit -- it
bit ME earlier in this same window at the old `:1773` list.** A plan file
that records supersession as PROSE will re-supersede itself the moment a
reader lands mid-file. Ordering must live in exactly ONE place; every other
mention is a pointer or a struck-through line with a banner. **And a document
whose sections each claim to supersede the others has no order at all -- it
has four, and the reader picks by where they entered.**

No code changed. A2 HELD, 7d PARKED, THE LAW holds, other windows' dirty
paths preserved.

## 2026-07-28 18:40 -- HEAD 7e768828 (v2.0-alpha) -- WINDOW RENDER/QA (cont.)
Did: ran the 45-word engine-coverage campaign to completion over all 18 LOCAL
  engines. RESULT: 11 publish to otr/obs/, 6 NO_RENDER, mesh_stage renders 7/7
  and publishes nothing. The six are NOT a stills problem -- 5 of 6 are
  MULTI-SEGMENT COVERAGE (wan x2 at node 92, no session_identity(); humo x3 at
  node 90, beat > cap with no per-segment audio slicer) and 1 is a preflight
  string match (ltx_video: _is_deferred_image_gap's four needles miss the
  LTX-I2V wording, so ShotLock re-raises and node 91 never runs).
  Ran a wiring kibitz arc r1 + r2 with THREE seats (opus + codex gpt-5.6-sol
  high + agy). VERIFIED MYSELF: session_identity() exists on ONE engine
  (eng_ltx_8gb.py:732); beat_session.py:155 raises the exact string the wan
  receipts carry; ltx_8gb really does render multi-segment (server_ltx.log
  :1323,1394,1533,1582). agy's "unproven assumption" was wrong.
  codex broke my scope estimate correctly: identity ALONE is not the fix --
  BeatSession promises ONE MODEL LOAD PER BEAT, and ltx_8gb earns that with
  identity + custom prepare() + hoisted checkpoints with loaders omitted from
  the segment graph + teardown. A ten-line identity declaration would silence
  the refusal and then load per segment.
  r2 found the partition bug: a 184-frame beat plans FOUR clips because the
  ladder is solved as an EXACT cover despite allow_tail_trim=True (33 + 4n:
  118%4=2, 85%4=1, 52%4=0). ~15 trips vs ~30 on a 100s beat.
  THREE OPERATOR RULINGS RECORDED in GO_FORWARD: (1) a still floor is legal
  ONLY where the partition math is impossible, never where an engine refused;
  (2) every audio-in beat gets a still with a mouth -- the no-lip-sync
  proposal is overruled; (3) the lips may be a person OR A RADIO, and the
  Fable seat then revised its own verdict: the set speaks by default, point
  the engine at the magic-eye tube, no legible text or straight edges in the
  still, and humo_14B_169's 49-frame ceiling stops being a defect on the
  cabinet.
ARC CLOSED: r1-r4 all judged. **BUILD FROM r3/final.md AS AMENDED BY
  r4/final.md.** codex's VERIFY-AT-BUILD checklist (r4/codex.md, last section)
  is the adopted per-chunk gate. Order: W1 partition (184 -> [153,33] trim 2;
  185-240 two segments) -> W2 typed gap (leaf module; convert ONLY
  render_driver.py:1985,2049,2105,2146,2179; post-image :1024/:1055/:1084 stay
  TERMINAL) -> W6 end card -> W3 WAN (UNET-only hoist, VAE in identity) ->
  W4 HuMo (session BEFORE slicer; SUPPRESS eng_humo.py:525-531 per-segment
  reclaim or the hoist is evicted; conditioning WAV = render_frames with
  silence padding for trim_tail) -> W7 mouth-still ownership (ShotLock is the
  sole cardinality owner; ZERO OR ONE human face, never inferred from prose)
  -> W5 grader (per-shot frozen-route comparison; histograms are CUT).
Superseded step line: r3 of the wiring arc. r1 and r2 are JUDGED
  (kibitz-runs/2026-07-28-local-engine-obs-wiring/r{1,2}/final.md); r2/final.md
  carries the build order and is code-ready. NOTE: the opus seat FAILED to
  produce claude.md in r2 -- re-seat it in r3.
Next: r3 + r4, then code in the r2/final.md order -- C5 partition fix FIRST
  (it shrinks C2 and C3), then C1 typed DeferredImageGapError in
  _otr_shared/retry_taxonomy.py, then WAN (hoist ONLY the UNET patcher --
  hoisting CLIP/VAE nullifies free_after_use and OOMs a 16 GiB card), then
  HuMo audio bounded to CoverageSegment.contributes.
Models: Claude + Sonnet fan-out (6) + kibitz r1/r2 (opus + codex 5.6-sol high
  + agy) + one Fable spawn on the two taste rulings.
Commits: cfcd572c, 4a47f005, 24d69d9a, 7e768828 (docs only; no production
  code touched -- the harness and campaign live in tmp/).

## 2026-07-28 06:15 -- HEAD 72282083 (v2.0-alpha) -- WINDOW RENDER/QA
Did: judged the full kibitz r1-r4 arc on the GPU lane plan (codex gpt-5.6-sol
  high + agy Gemini 3.6 Flash High, pins verified every round) and rebuilt the
  harness around what it found. FIRST LIVE PROOF for the five encoder chunks:
  a `still_flat` leg published green -- credits console 52.0s at 1920x1080,
  `obs_publish OK`, 14,637,297 bytes, `engine_histogram {"still_flat": 7}`.
  Found `word_razzle` is a Pixverse CLOUD engine that every name-prefix filter
  calls local (2-of-2, confirmed in code); locality now delegates to
  `render_driver._is_cloud_video_engine` -- true local roster is 18, not 19.
  Found the headless launcher sets no image-engine flags, so `flux2_klein` and
  `lumina_image` cannot work in a soak run (two cases burned 552s each proving
  nothing). Found `mesh_stage` can never publish a whole-episode case: the
  2026-07-03 directory-clip look contract in `plan_backdrop` refuses it -- not
  a regression. Harness now proves cases from the ASSET ON DISK plus the
  engine histogram instead of a `poll_history` status (the operator-named gap;
  the old receipts recorded FAIL for an episode that was complete on disk).
  Four of my own defects were caught by the panel before the long run: a
  `character_visual` rename that broke every case in 1s, a C6 gate on an
  unread assumption that killed a campaign in 90s, an aggregation hole, and a
  credits predicate that could never match a 500-char-truncated error.
Current step: the operator's GPU sequence. The engine-coverage campaign is
  RUNNING (master `tmp/gpu_lane_all_models_20260728_060646`, 18 engines, 4
  lanes, ~6h, harness pinned by SHA-256). A2 stays HELD, 7d stays PARKED.
Next: read `tmp/_kbA_gpu_campaign.done` + `campaign_summary.json` when it
  lands; then the operator's clamped ltx recipe-v2 confirmation. Owed harness
  items are listed in r4/final.md (none load-bearing tonight).
Models: Claude + 4 kibitz rounds (codex gpt-5.6-sol high + agy 3.6 Flash High).
Commits: none (no production code touched; harness lives in tmp/).

## 2026-07-28 -- HEAD 1959fb49 (v2.0-alpha) -- WINDOW CODER (continued)
Did: `1959fb49` the credits-card col1 ladder. The row was filed LATENT and was
  live on the default path: roll() sizes the card from the FINISHED VIDEO, the
  canonical workflow ships 832x480, and that canvas was overflowing its own
  footer on every episode while PIL clipped it silently. Fable ruled the
  standing policy; agy dissented and was overruled on one point and adopted on
  two.
Current step: the remote-safe queue is one DESIGN job -- a small-canvas variant
  of the card for 512x288 / 640x360 -- plus A2, still HELD. Next real work is
  the operator's GPU sequence.
Next: a RENDER window owns the GPU items. The small-canvas card is a design
  chunk whenever someone wants it; it blocks nothing.
Models: Claude codes and judges (rung 4); FABLE ruled the persistent policy
  (CLAUDE.md section 9); agy at rung 2 ($0) reviewed and dissented; six Sonnet
  QA lenses across the session's fan-outs. No codex, no roundtable.
Commits: 1959fb49 plus this doc push. Suite 7453 -> 7464; Bible 17;
  build_variants --check 11/0; canonical 9872624A byte-identical.
  Mutation: 36/37 real mutants caught, 6/6 controls survived.

### Detail

**A REACHABILITY CLAIM IN A BUG ROW IS A CLAIM LIKE ANY OTHER.** The row said
the overflow was "reachable only if something renders the card at 480p -- the
shipped render tests use 720p and 1080p". Derived from the producers instead:
`roll()` takes w/h from `_probe_video(video_path)`, the canonical
OTR_VideoDirector ships 832x480, the ltx_8gb tier renders 512x288. Measured
spare on required content alone: 1080p +194, 720p +85, 832x480 -2, 640x360 -78,
512x288 -131. And `render_static_base` captured the column's returned `y` and
never used it, so nothing logged it either.

**THE POLICY, RULED BY FABLE, RECORDED SO IT IS NOT RE-LITIGATED.** The card is
a VIEW of the durable ledger, not the ledger. A record may never elide; a view
may elide WITH NOTICE. It may show less than it knows; it may never claim more
than it shows. Ladder: optional note (unmarked -- a gloss's absence asserts
nothing) -> inter-block WHITESPACE (unmarked -- whitespace is not a claim) ->
ledger ROWS, fine print first, always MARKED, SEED and COMMIT never dropped.
Type is NEVER shrunk: a receipt in unreadable type is a receipt-shaped object
claiming credit for a disclosure that never happened, and unlike a clip it is a
lie the policy tells on purpose. It never raises -- step 21 of 22, and
`54b3626b` already settled that a terminal node is the sanity ceiling.
CreditsDataError stays fatal for missing TRUTH; insufficient GLASS degrades.

**THE CANONICAL CANVAS IS FIXED WITH ZERO INFORMATION SPENT.** At 832x480 the
whitespace rung alone clears the footer with 6px spare, full ledger intact, no
marker, nothing logged. That rung exists because the shortfall is two pixels
and dropping a row to buy them nets nothing once the cut marker takes a row
back -- which Fable flagged before a line was written.

**agy DISSENTED AND WAS OVERRULED; TWO OF ITS MECHANICAL FINDINGS SHIPPED.**
It argued option C "violates the no-fallback contract" -- conflating a missing
RECEIPT (missing truth -> raise, untouched) with insufficient GLASS
(presentational). Discarded with the reason recorded. Its two mechanical
findings survived grounding and are both in the commit: the unused `y`, and
that compacting whitespace recovers enough to save the canonical canvas. The
dissent was wrong and the mechanics were right, which is the usual shape --
ground every claim separately rather than taking a review whole.

**MUTATION CAUGHT A DECORATIVE TEST OF MINE, the fifth time this arc.**
`test_the_canonical_canvas_keeps_its_WHOLE_ledger` read the INPUT layout, which
`_abridge` deliberately COPIES rather than mutates -- so it passed no matter
what the ladder did, and deleting the whitespace rung SURVIVED: the column fell
through to dropping rows, still "fit", and the assertion never noticed. It
observes what was DRAWN now, and derives the expected row list from the layout
rather than hard-coding labels the fixture does not even have.

**A PROCESS ERROR OF MINE, twice in one session, and worth a rule.** I launched
the mutation harness three times without checking whether one was already
running, and two raced -- each mutating the same files while the other held a
mutant on disk. Verified no corruption (`git diff --stat HEAD` showed only the
two files the chunk owned, and a fingerprint grep for every mutant string came
back clean), but the risk is real: run B can capture run A's mutant as its
"original" and restore it permanently. RULE: one mutation harness at a time,
check for a live one first, and never alongside a QA fan-out -- earlier in this
session a lens read a mutant off disk and reported it as corruption.

## 2026-07-28 -- HEAD b1f2ee86 (v2.0-alpha) -- WINDOW CODER (continued)
Did: two more green pushed chunks on the same arc. `6aad4fe5` DELETED the third
  copy of the scope encoder -- otr_scene_aware_scopes had its own private
  _encode_silent_mp4 carrying every defect the shared one was fixed for two
  commits earlier -- and pinned the six remaining rawvideo-stdin encoders so a
  fourth fails by name. `b1f2ee86` closed the odd-canvas stride defect: the
  batch encoder's declared -s is now the size it actually pipes, and an odd
  canvas is refused by name.
Current step: the remote-safe queue is down to ONE small filed row (the credits
  card's col1 overflowing the footer at 854x480) plus A2, still HELD. The next
  real work is the operator's GPU sequence -- clamped recipe-v2 confirmation,
  the WAN prequalification sweep, then 7d.
Next: a RENDER window owns all three GPU items. The credits-card geometry row
  is a coder chunk whenever someone opens that file; it blocks nothing.
Models: Claude codes and judges (rung 4) + six Sonnet subagent QA lenses across
  three pre-push fan-outs. No codex, no agy, no Fable, no roundtable --
  two-strikes never invoked, so no panel was owed.
Commits: 6aad4fe5 b1f2ee86 plus this doc push. Suite 7449 -> 7453; Bible 17;
  build_variants --check 11/0; canonical 9872624A byte-identical throughout.
  Mutation: 30/31 real mutants caught, 6/6 controls survived.

### Detail

**THE SAME DEFECT WAS IN A COPY, THREE TIMES, AND DELETING THE COPY IS WHAT
CLOSED IT.** otr_scene_aware_scopes assembled a byte-for-byte identical ffmpeg
command to the shared encoder and carried the identical defects: `total`
accepted and never read, the rawvideo `-s` from the caller rather than the
frames, no per-frame shape or dtype check, nvenc with no canvas floor, and a
stderr PIPE read only after the whole stream was written -- a deadlock that
raises nothing, so the child is never reaped and holds the output file. It was
deleted rather than hardened a third time; render_scopes calls
_otr_shared.scope_draw.encode_silent_mp4, which is exactly the refactor that
module's docstring anticipated. The SEPARATION INVARIANT is directional and
always was -- scope_draw must not import a NODE -- and this node already
imported freq_bars_green from it.

**MUTATION PROVED THE DELEGATION RATHER THAN THE COMMENT CLAIMING IT.** Passing
the node's dimensions SWAPPED, and declaring one frame more than the generator
yields, are both refused now and were both silently accepted by the deleted
copy. Two mutants, both dead, on a live end-to-end path.

**AND A GATE AGAINST A FOURTH.** _RAWVIDEO_STDIN_ENCODERS pins every function
under nodes/ that pipes raw frames into ffmpeg on stdin -- six, each with the
reason it exists -- and names otr_scene_aware_scopes in its own assertion. A new
copy fails HERE instead of being found by a fan-out two months later.

**THE ODD-CANVAS DEFECT PASSED EVERY PROOF THIS ARC ADDED, AND THE SUITE WAS
DEFENDING IT.** ffmpeg_silent_mp4_cmd declared even_dim(w) while
encode_frames_to_silent_mp4 piped the array's real odd rows. Measured, a
(5,63,47,3) batch wrote a 46x62 clip of skewed pixels, exit 0, and the
frame-count proof AGREED -- five in, five out. A count proof structurally cannot
see a stride error. Worse, test_ffmpeg_silent_cmd_contract REQUIRED the
rounding, commented "odd width -> even": the defect written down as the
contract, which is why it sat filed as latent instead of being caught. **A
latent row the tests assert as the contract is not latent, it is protected.**
even_dim stays on the three builders that SCALE or PAD to a target, where
ffmpeg is told what to produce; both halves are asserted so they cannot be
collapsed into one.

**A SEQUENCING MISTAKE OF MINE, WORTH NOT REPEATING.** I launched a QA fan-out
and a mutation round at the same time. The lens read `if False:` in the shared
encoder and reported possible corruption -- it was the mutation harness holding
a mutant on disk at that instant. The lens caught it by re-verifying through a
second reader, which is the right instinct, but the wasted round is on me. Do
not run a fan-out while mutation is editing files.

**ONE HYGIENE FALSE ALARM, RESOLVED BY MEASUREMENT NOT BY REWRITING.** My
scratch hygiene script failed on non-ASCII in the two scene-aware files. Both
carry a literal section sign at HEAD, long predating this work; the non-ASCII
inventory is byte-identical to HEAD, so nothing new was introduced and
rewriting them would have been unrelated churn. The repo rule is UTF-8 / no
BOM; ASCII-only is a per-file docstring convention.

## 2026-07-28 -- HEAD afeb5b84 (v2.0-alpha) -- WINDOW CODER
Did: closed THE SECOND ENCODER in two green pushed chunks -- `27a4f97c` the
  four viz_* engines' colour proof + proven frame count, the scope_draw encoder
  hardened, and the M7 roster gate rewritten to identify a clip WRITER
  structurally instead of grepping two call spellings; `afeb5b84`
  cheap_families' four still_* count proofs + the gate's matching COUNT half.
  The gate went RED on exactly the four viz engines when widened, as predicted,
  and green by fix rather than by narrowing. Fan-out ran BEFORE both pushes and
  found six real defects, five of them in my own new code -- including the
  sweep's subprocess-alias test being simply wrong, which made the roster EMPTY
  and both gates pass vacuously.
Current step: the remote-safe lane is EMPTY except A2 (held) and one small
  filed row (the THIRD encoder copy in otr_scene_aware_scopes.py, which writes
  a compositing overlay and not a CanonicalClip). Next real work is the
  operator's GPU sequence: clamped recipe-v2 confirmation, the WAN
  prequalification sweep, then 7d.
Next: a RENDER window owns all three GPU items. A CODER window can take the
  third encoder any time; it blocks nothing.
Models: Claude codes and judges (rung 4) + five Sonnet subagent QA lenses
  across two pre-push fan-outs. No codex, no agy, no Fable, no roundtable --
  two-strikes never invoked, so no panel was owed.
Commits: 27a4f97c afeb5b84 plus this doc push. Suite 7429 -> 7449; Bible 17;
  build_variants --check 11/0; canonical 9872624A byte-identical throughout.
  Mutation: 23/24 real mutants caught, 6/6 controls survived, 2 reclassified,
  1 survivor recorded.

### Detail

**THE FAN-OUT CAUGHT THE SWEEP FINDING NOTHING, WHICH IS THE WORST POSSIBLE
FAILURE FOR A GATE LIKE THIS.** The first draft classified a spawning function
by testing `"sp" in func.value.id` -- which is FALSE for `"subprocess"`. The
entry-point inventory came back empty, so both the roster gate and the
contract gate passed over an empty set, green and useless: the exact vacuous
pass this whole file exists to close, reintroduced while closing it. The alias
now comes from the module's own `ast.Import`/`ast.ImportFrom`, and every roster
gate asserts that NAMED engines are BILLED rather than merely that nobody
failed -- `unproven == {}` is satisfied just as well by "nobody writes a clip".

**FIXING ONE BLIND TEST DID NOT FIX ITS NEIGHBOUR, AND TWO LENSES FOUND IT
INDEPENDENTLY.** `test_the_proof_runs_AFTER_the_encode_in_every_adapter`, in
the same file, still regexed `encode_frames_to_silent_mp4\(` alone. So moving
the proof BEFORE the encode in any of the four viz engines -- the exact defect
that test is named for, in the exact files this chunk was wiring -- stayed
green. It derives its spellings from the same billed-debt calculation the
contract gate uses now, and a mutant that reorders viz_camera dies on it.

**wan_shared WAS EXCUSING ITSELF ON ITS OWN `def` LINES.** `_has_proof` matched
markers as substrings; `wan_shared.py` DEFINES both `ffprobe_counted_frames`
and `validate_silent_clip_contract`, so `def ffprobe_counted_frames(` satisfied
the check with no call at all. The one module that could regress its real
`counted != expect_frames` comparison was the one module neither gate could
notice it in. Proof is an AST CALL now, and the gate's own logic is pinned by a
test that feeds it a define-only source.

**TWO DEFECTS IN MY OWN ENCODER, BOTH ABOUT THE CHILD PROCESS.** A refusal
raised part-way through the frame stream left ffmpeg ALIVE holding the output
file open -- the first refusal test failed on a PermissionError from its own
TemporaryDirectory cleanup rather than on the refusal it was checking. And
stderr was a PIPE read only after the whole stream was written, which deadlocks
the moment ffmpeg emits more than one OS buffer of error text; that state
raises nothing, so neither except clause runs and the child is never reaped.
stderr is a temp file now, and every exit path reaps.

**A LATENT BOX-DEPENDENT FAILURE, MEASURED NOT GUESSED.** The encoder selected
h264_nvenc whenever the box had it. NVENC refuses a canvas below 145x49 with an
error naming four parameters and not the one that is wrong. Measured on this
box: 144x48 refused, 146x50 accepted, libx264 accepted every size from 96x64
up. So a small-canvas beat died on a machine WITH a GPU and succeeded on one
without. Codec SELECTION, not a fallback -- both encoders emit the same
contract and the caller proves it either way. Found only because the viz
contract tests stopped stubbing the encoder.

**THE TESTS STOPPED VERIFYING A FILE THEY INVENTED.** The three viz
render-contract fakes wrote one zero byte and the tests then asserted a frame
count against it. They pass through to the real encoder now and skip where
ffmpeg is absent.

**MUTATION RECLASSIFIED TWO AND RECORDED ONE SURVIVOR RATHER THAN CHASING
EITHER.** Spelling the declared size `(w, h)` is provably identical once the
equality is proven two lines above -- the same call this build made for
`int(counted)` vs `int(declared)` at 48e3c6fb. Dropping `Popen` from the spawn
set changes nothing while every encoder entry point in the tree is also a
returner; the branch stays because an encoder that returns nothing is an
ordinary thing to write next. The survivor -- deleting the self-proving
membership assertion -- is catchable only by a meta-test of that assertion,
which is not written, and it is in OPEN BUGS rather than left implied.

**THE TREE WAS LEFT EXACTLY AS FOUND.** Another window's three modified
`tmp/*.ps1` and its six untracked `config/profiles/otr_sbcov_*.json` were
preserved throughout; every commit was pathspec-only; no variants were
generated from those scratch profiles, so 7449 reproduces on a clean clone.

## 2026-07-28 -- HEAD 48e3c6fb (v2.0-alpha) -- WINDOW CODER
Did: the three remote-safe rows, in the operator's order, one green pushed
  chunk each -- `bcaab4db` the by_engine PER-FIELD roll-up (+ both credits
  readers), `24f4251a` the credits card drawing video_suffix + the _row()
  clamp, `48e3c6fb` the encoder returning a PROVEN frame count. The QA
  fan-out ran BEFORE every push this time and caught a 720p layout regression
  and a lost-beat behaviour change that mutation structurally could not see;
  both are fixed inside their own commits. A2 untouched (still HELD behind the
  profile scope). 7d still PARKED.
Current step: the SECOND ENCODER -- nodes/_otr_shared/scope_draw.py, which
  four live viz_* engines write clips through with no ffprobe at all and which
  the M7 roster gate structurally cannot see. Then cheap_families' four still_*
  count proofs. Then the operator's GPU sequence.
Next: a CODER window takes the second encoder + the roster gate widening (it
  will go red on purpose). The clamped recipe-v2 confirmation, the WAN
  prequalification sweep and 7d all belong to a RENDER window.
Models: Claude codes and judges (rung 4) + eight Sonnet subagent QA lenses
  across three pre-push fan-outs. No codex, no agy, no Fable, no roundtable --
  two-strikes never invoked, so no panel was owed.
Commits: bcaab4db 24f4251a 48e3c6fb plus this doc push. Suite 7384 -> 7429;
  Bible 17; build_variants --check 11/0; canonical 9872624A byte-identical
  throughout. Mutation across the three chunks: 38/38 real mutants caught,
  13/13 controls survived.

### Detail

**THE FAN-OUT PAID FOR ITSELF TWICE, ON GROUND MUTATION CANNOT REACH.** Row
2's recipe note had a FIXED two-line allowance; at 1280x720 -- the size this
repo's own render tests already use -- that pushed col1 27px past the footer,
because col1 flows its blocks downward with no backstop and PIL clips the
overflow silently. No mutation of the code reveals that the LAYOUT stopped
fitting. The column now measures itself onto a scratch canvas and spends the
allowance down until it clears the footer. Row 3 turned a zero-frame batch
from `return (path, 0)` into a raise from the count proof describing a failed
multi-segment ASSEMBLY -- true words about the wrong event -- so the encoder
refuses zero frames by name instead.

**THE FRAME-COUNT ROW ASKED THE WRONG QUESTION AND THE ANSWER WAS FREE.** It
framed the choice as "pay a decode per clip or leave the count self-declared".
`nb_frames` is the MUXER'S OWN count and rides the same stream read
`ffprobe_clip_fields` already performs on every emitted clip -- the identical
argument that put width/height in that query at chunk 6. The decode is now the
FALLBACK, for a container recording no count. Measured before deciding: header
29-45ms flat from 50 to 18000 frames, decode 35-168ms and scaling, against
real beat renders of 744-842 SECONDS. The docstring's "expensive by design"
was true of the decode and was never the reason this could not be done.

**MUTATION KILLED THREE DECORATIVE ASSERTIONS OF MINE, AND ONE MUTANT WAS
RECLASSIFIED RATHER THAN CHASED.** The line-count test asserted against
`cr._NOTE_LINES_MAX` instead of the literal 2, so raising the ceiling to 9
left it green -- a two-line note could have become a wall of micro text. Every
frame-count fixture had counted < declared, so a refusal that only caught
SHORT clips stayed green; a beat with MORE frames drifts just as badly. And
`return int(counted)` -> `return int(declared)` survived because control only
reaches that line after the two are proven equal: that is a CONTROL, not a
decorative test, and the source keeps `int(counted)` because it names the
authority if the check ever gains a tolerance.

**THE FAN-OUT ALSO KILLED A TAUTOLOGY AND TWO VACUOUS TESTS.** The clamp test
asserted only the RIGHT edge -- which is an identity of the positioning
formula (`vx` is DEFINED as `x + colw - width`), so it passed against the
unclamped code that put `vx` at -754. Two frame-count tests asserted inside a
bare `except` block, which passes vacuously the day the code stops raising.
Both patterns are now written into GO_FORWARD's carry-forward list.

**AND IT FOUND A SECOND ENCODER NOBODY HAD FILED.** The four viz_* engines do
not use `encode_frames_to_silent_mp4` at all -- they write through
`nodes/_otr_shared/scope_draw.py`, which has no ffprobe call of any kind, and
the M7 roster gate cannot see them because it greps for the literal strings
`encode_frames_to_silent_mp4(` and `run_ffmpeg(`. That is the cheap_families
finding of 2026-07-27 repeating one module over, in the exact shape the gate
was built to catch. Three of those four are the video slots the surviving
six-bank 120w matrix uses. Filed, not started -- it is a multi-file chunk and
the operator's scope was three rows.

**FOUR CITES MOVED AGAIN, and one bug-list claim was wrong about the code.**
`_draw_models` is `otr_credits_roll.py:675-719` not `:657-712`;
`ffprobe_counted_frames` is `wan_shared.py:124` not `:105`; and the receipt's
own comment claimed a non-stamping engine arrives with `family=None` when
`build_clip_manifest` writes `clip.get("family") or shot.get("family") or ""`,
so it arrives as `""`. The `by_engine.setdefault` cite at `:87` and the
credits `:211`/`:269` cites were still accurate -- "every cite has moved" is
a real warning but not a universal one.

**THE TREE WAS LEFT EXACTLY AS FOUND.** Another window's three modified
`tmp/*.ps1` and its six untracked `config/profiles/otr_sbcov_*.json` were
preserved throughout; every commit was pathspec-only; no variants were
generated from those scratch profiles, so 7429 reproduces on a clean clone.

## 2026-07-27 20:24 -- HEAD 40780b82 (v2.0-alpha) -- WINDOW CODER
Did: executed the ranked open-bug queue. SIX of seven rows shipped as green
  pushed chunks -- A1 ebec0f1f, A6 ba24af29, A4 c9b89769, B4 57caf43d,
  A5-lite de50786e, the frame_count M7 sweep 58e288af. A2 HELD behind the
  profile retire-now/retire-later scope, not skipped. Then a Sonnet fan-out
  over all six found TWO real defects in already-green, already-pushed,
  mutation-proven code -- both mine -- fixed at 40780b82.
Current step: the ranked queue is DONE; next remote-safe lane is the by_engine
  roll-up, then the credits-card video_suffix (in that order), then the
  encoder frame-count decision. 7d still PARKED.
Next: a CODER window takes by_engine; the clamped recipe-v2 confirmation, the
  WAN prequalification sweep and 7d all belong to a RENDER window.
Models: Claude codes and judges (rung 4) + a Sonnet subagent fan-out for the
  post-push QA round. No codex, no agy, no Fable, no roundtable -- two-strikes
  never invoked, so no panel was owed.
Commits: ebec0f1f ba24af29 c9b89769 57caf43d de50786e 58e288af 40780b82 plus
  this doc push. Suite 7356 -> 7384; Bible 17; canonical 9872624A
  byte-identical throughout.

### Detail

**THE TRIAGE'S OWN B5 GATE WAS WRONG ABOUT TWO OF THE THREE ROWS IT GATED.**
It said the profile retain/retire ruling gated A1, A2 and A6. It gates only A2.
The VRAM ceiling has a live NON-profile channel -- llm_vram_ceiling_gb is a
widget in otr_canonical.json, which is exactly the channel the operator's
retirement direction KEEPS -- and the GGUF artifact table belongs to the
loader. Flagged before starting; operator agreed; A1 and A6 went first.

**A1 WAS A PURE HOIST, AND THE OBVIOUS FIX WOULD HAVE BROKEN THE DEFAULT.**
check_vram_fit already prices a gguf_native row from its pinned on-disk
artifact plus KV, and already answers correctly at both ceilings (gemma GGUF
estimates 14.6 GB: WARN at 14.5, FAIL at 6.8). The defect was placement only --
the gate sat below both cache-hit returns and below the GGUF dispatch, so it
could only ever gate a fresh transformers load. Writing the natural hard
estimate > ceiling comparison would have refused today's canonical default at
14.5. Grounded with a throwaway probe before any code was written.

**A6 SHIPPED BROKEN AND THE POST-PUSH PANEL CAUGHT IT.** Refusing unpinned GGUF
artifacts is right, but config/profiles/otr_mac_mps.json and otr_nv40_12gb.json
both select Q6_K, which has no pin -- and their GENERATED variant workflows
carried Q6_K in the writer node's widgets_values with no in-workflow remedy
(hard-coded widget, no GEMMA4_12B_GGUF_PATH set). Both moved to the pinned
Q4_K_M, which is also the only quant that fits their declared 10.0 / 10.5 GB
ceilings; Q6_K at ~9.1 GiB plus a 2.8 GiB KV cache never did. Fixed at the
profile and regenerated through build_variants.py rather than hand-edited.
Q4_K_M and Q8_0 were pinned by MEASUREMENT: Q4 hashed in all three copies on
this box (all agreeing byte for byte) and the Q8_0 measurement reproduced its
existing pin, which is what corroborates the set.

**THE SECOND PANEL FINDING WAS A TEST OF MINE THAT WAS CONFIDENTLY BLIND.** The
M7 roster gate added at 58e288af globbed eng_*.py and grepped for
encode_frames_to_silent_mp4. cheap_families.py matches neither -- wrong
filename, and it builds its mp4 from an ffmpeg arg list -- so still_motion,
still_pan, still_flat and still_word kept hand-writing container/codec/
pixel_format/color_* as literals while the test reported PASS over them.
still_motion is the terminus of the humo -> humo_1.7B -> still_motion degrade
chain. Sweep widened to every module and both write paths, probe added, and an
explicit assertion that the sweep can still SEE cheap_families.py.

**MUTATION BEAT THE LENSES A FOURTH CONSECUTIVE TIME, ON MY OWN TEST.** The
frame_count ordering test asserted only that some proof FOLLOWS each encode,
which stays green when a bad proof is inserted BEFORE one -- the exact defect
the test is named for. Now asserted in both directions. Across the window:
32/32 real mutants caught, 10/10 controls survived.

**AND THE DISCIPLINE LESSON IS ONE THIS FILE ALREADY RECORDED.** Every chunk
ran its mutation round before its push and they were load-bearing, but no
mutation of the CODE can reveal that a shipped JSON ARTIFACT selects something
the code just made illegal. GO_FORWARD already said "run the fan-out BEFORE the
push, not after"; this window ran it after six pushes and paid for it.

**THREE BUG-LIST ROWS WERE WRONG ABOUT THE CODE, in ways that only mechanical
derivation caught.** B4's row named a beat_id no producer stamps and missed
jump_still_requests and motion_clause, which are stamped. The frame_count row
listed eng_ltx_video among the adapters that already probed; it did not, on
either recipe path, and eng_still_parallax was absent from the row entirely --
the sweep found four adapters, not two. A6's cites (56-60, 435-439, 982-992)
all still pointed at the right code, so the "every cite has moved" warning is
real but not universal.

**A RECEIPT DEFECT WORTH NOT REPEATING:** build_variants.py --all also emits
variants for any UNTRACKED profile on disk, and some profile checks are
parametrized over the variants present, so another window's six scratch
profiles inflated the first suite reading to 7396 -- a number that would not
reproduce on a clean clone. The generated files were removed, restoring the
tree to exactly the shape it was found in, and the suite re-measured at 7384.

## 2026-07-27 -- HEAD 54b3626b (v2.0-alpha) -- WINDOW CODER (BUG TRIAGE)
Did: operator-directed triage of the whole OPEN BUGS list, then the fixes it
  turned up. Panel: kibitz r1 with codex gpt-5.6-sol high (seat verified in
  codex_model_selected.txt for this run) + agy Gemini 3.6 Flash (High), then a
  Fable consult under CLAUDE.md section 9's reality exception. Claude wrote the
  anchor triage first and grounded every panel claim against the real Windows
  files before acting on any of it. Of five anchor rows the panel corrected
  three, cut one, and added one that was absent from GO_FORWARD entirely.
  Shipped 54b3626b: the two OTR_MasterAudioMux defects Fable found, both in the
  LAST node of the graph, where everything raises AFTER the whole episode has
  already rendered.
Current step: B5's ruling FIRST. It is a dependency, not a peer; whether the
  profile family is retained or retired changes the value and the acceptance
  target of A1, A2 and A6. Then A1, A6, A2, A4, B4, A5-lite, frame_count.
Next: the ranked queue in docs/2026-07-27-open-bug-triage.md, carried into
  GO_FORWARD's CURRENT STEP. 7d stays PARKED.
Models: Claude anchors, judges and codes; codex gpt-5.6-sol high and agy Gemini
  3.6 Flash (High) as the local panel ($0, rungs 2 and 3); one Fable consult
  (rung 6, operator-authorized) as the final gate. Two-strikes never invoked.
Commits: 54b3626b plus this doc push. Suite 7346 to 7356; Bible 17; canonical
  9872624A byte-identical. Record: docs/2026-07-27-open-bug-triage.md.

### Detail

**THE PANEL DISAGREED WITH ME MORE THAN THE TWO SEATS DISAGREED WITH EACH
OTHER.** That is the finding worth keeping. Three corrections, all grounded:

1. **A1's fix shape was INCOMPLETE.** "Enforce the ceiling in GGUF preflight"
   misses the path that matters: a resident model returns at
   _otr_model_loader.py:982-992 without entering preflight at all, and
   GGUFLoadConfig.reuse_key() (_otr_gguf_backend.py:435-439) excludes the
   ceiling, so a permissive-policy load satisfies a stricter-policy request by
   cache hit. Correct shape: ONE policy-admission calculation before BOTH cache
   reuse and loading, with a test for permissive-cache to stricter-request at
   the same load identity.
2. **A2's causal chain was WRONG.** The override does not come from the
   validator's OTR_ACTIVE_PROFILE export; it happens at submission,
   scripts/otr_canonical_api_run.py:157 into apply_profile_to_workflow. And the
   real applier (nodes/_otr_workflow_apply.py:492-540) ALREADY flattens llm.
   Only the printed echo (scripts/otr_api.py:816-825) is stale. Generate the
   echo FROM the applier's map; adding llm by hand leaves the next drift intact.
3. **A3 was already covered and I would have written a duplicate.** Three tests
   cover the provider_side redirect: test_video_render_driver_perbeat_audio.py
   :319-325, test_video_platform_aseam.py:903-920, and
   test_still_plan_parity.py:114-116. I had checked the CODE, not the TESTS.

**A6 IS NEW AND IS THE HIGHEST-VALUE ROW.** The shipped 8 GB profile selects
Gemma Q4_K_M, but GGUF_ARTIFACTS (_otr_gguf_backend.py:56-60) gives that quant
size None and GGUF_ROWS (:226-233) gives sha None. Both checks are conditional
on the value existing, so a truncated or partial Q4 download passes readiness.

**FABLE RESOLVED BOTH SPLITS THE MECHANICAL SEATS LEFT OPEN.** A5 (codex: fix
at the shared boundary / agy: cut) is cut as a LIVE bug but keeps codex's
location at a fraction of his scope: every producer feeds exact-size uint8,
ffmpeg raises on a short write, and chunk 6 already put a decode-count at the
boundary that matters (wan_shared.py:224-232). One dtype == uint8 assert closes
the latent residual, which is a future float32 caller piping 4x the bytes and
getting a clean receipt. B4 ShotRow (mine: operator ruling / agy: coder fix) is
a CODER FIX: ShotLock stamps role, char_id, start_s/dur_s, coverage_plan and
coverage_contract, none of which exist on a model declaring extra="forbid", so
ShotRow(**real_row) raises on every real ledger and the "live safety net" other
docs cite cannot validate one shipped episode. The repo's own observability and
requires_mesh_portrait precedent settles the shape. No product question is left.

**AND FABLE KILLED A FINDING THE OTHER SEAT WAS CONFIDENT ABOUT.** agy's
heavy-import claim: the imports are real (Fable verified all four files and
found eight more), but the enforced gate test_capability_profiles.py:481-503
excludes the audio lane BY DESIGN and says so in its own docstring; ComfyUI
imports torch/PIL/numpy before any custom node loads; and __init__.py wraps
every node import so a broken dep skips one node loudly. Not a violation of the
gate as this build defines it. Do not file.

**WHAT SHIPPED AT 54b3626b.** Both defects live in OTR_MasterAudioMux. First, a
FATAL env knob: float(os.environ.get("OTR_MAX_CREDITS_TAIL_S", "45")) was
unguarded, so a malformed value killed a finished episode with an uncaught
ValueError over a knob that only widens a sanity ceiling. That is the
PBUG-20260723-02 shape, at the opposite end of the pipeline from where this
build usually pays for it. Now IGNORED and NAMED via _credits_tail_ceiling();
the sibling knob in the same file was already guarded, this was the one that was
not. Second, the duration gate fails open: _probe_float returns -1.0 when
ffprobe is absent or a duration is unparsable, which skips the only
video-longer-than-audio guard, and the report still appended
"duration_check v=-1.000s a=-1.000s ... OK". Now UNPROVEN, with the gate named
as SKIPPED rather than passed. Not made fatal: it is the final sanity ceiling,
not the primary correctness guard, and refusing would lose a finished episode on
a box that merely lacks ffprobe.

**STILL OPEN FROM FABLE, NOT YET FIXED.** CanonicalClip.frame_count, "the
integer timing authority", is decode-counted truth for assembled multi-segment
beats but self-declared input length for every single-render beat, and eng_humo
and eng_ltx_av return self-declared dicts with no M7 probe while wan_i2v,
wan_ti2v, ltx_8gb and ltx_video all probe. The two derivations agree today only
because every producer pipes exact bytes. Now filed as an OPEN BUG.

**A DEFECT IN THE BUG LIST ITSELF.** Every line cite checked had moved:
_is_cloud_video_engine is render_driver.py:1599 not 1274-1295; the "NO FALLBACK
to text-only" refusal is :2148 not 1801-1817; _use_i2v is eng_ltx_video.py:583
not 559-572. The defects are mostly still real; their coordinates are not.
Re-pin a row's cite when you touch it.

**PROCESS NOTE.** r2/r3/r4 of the kibitz arc were NOT run. The arc hardens a
PLAN across four lenses; what was asked for was a triage plus fixes, and r1 plus
the Fable consult answered it. A next window wanting the full arc on the ranked
queue starts at r2 with docs/2026-07-27-open-bug-triage.md as input.

## 2026-07-27 -- HEAD 8424f369 (v2.0-alpha) -- WINDOW CODER (LANE 2)
Did: a measurement clip's receipt now names WHICH cell produced it --
  71e231ec (ltx_8gb + the shared format in the new recipe_departures.py),
  8424f369 (both WAN adapters). B6 made a sweep artifact distinguishable from
  production; it left the four cells of the 2026-07-27 sweep indistinguishable
  from EACH OTHER, so the winner was selected from a table kept outside the
  ledger. Fixed a latent lie the fan-out found: the receipt is session-scoped
  and cannot honestly report a per-shot negative, so under the consent act a
  shot displacing the measured negative is now terminal. Also collapsed the
  tile-geometry range check to one implementation on both ltx and wan_ti2v.
Current step: OPERATOR'S PICK -- the remote no-GPU queue is drained of its
  obvious items. What remains wants a ruling (ShotRow wire-or-demote, the
  by_engine roll-up) or a GPU (clamped v2 confirmation, a WAN sweep, 7d).
Next: operator decides. 7d stays PARKED.
Models: Claude codes + judges; 3 Sonnet QA lenses pre-push. $0 external -- no
  codex, no agy, no cloud roundtable; two-strikes never invoked.
Commits: 71e231ec, 8424f369. Suite 7291 -> 7346; Bible 17; canonical 9872624A
  byte-identical. Record:
  docs/2026-07-27-lane2-prequalification-receipt-qa-findings.md.

### Detail

**THE MUTATION ROUNDS BEAT THE PANELS AGAIN -- THIRD CHUNK RUNNING.** Three QA
lenses cleared the change; mutation then found four more real defects. Two are
worth naming as doctrine, because they are the same shape every time:

1. **An exception TYPE asserted without its message.** `pytest.raises(KeyError)`
   passed with the named drift guard DELETED, because the dict comprehension one
   line below raises the same type incidentally on the same input. The test
   proved nothing about the guard it was written for.
2. **A test that verifies a thing it also CONSTRUCTS.** The digest test was
   satisfied by `"#" + text[:8]` -- a truncation wearing a costume, passing the
   test named for refusing one, on every assertion it made.

Plus: a production-path guard that could be deleted with the suite staying green
(every accessor returns the frozen value anyway, so the guarantee silently
depended on nine accessors staying correct -- now proven by DETONATING the
resolver), and a `negative` departure that could be dropped from wan_ti2v
because only the wan_i2v twin of that test existed.

**AND ONE OF MY OWN CONTROLS WENT RED, which taught the opposite of what it
looked like.** A control that fails tells you nothing about the harness and
everything about the control: it renamed a dict at its assignment and left three
readers on the old name -- a broken mutant wearing a control's label. Replaced
with a genuine no-op.

**THE LATENT LIE, in full, because the reasoning generalises.** `_build_graph`
lets a per-shot negative_prompt win -- correct in production, and why B6 called
the negative a demotion rather than a removal. But the receipt is SESSION-scoped:
element [1] of session_identity, read before the weights land and again before
every segment, so it may only describe request-independent things. A sweep
varying the negative would therefore have rendered one conditioning and stamped
a receipt naming another -- a SPECIFIC false claim, worse than the vague true one
it replaced because it is more credible. Making the receipt request-aware was
rejected: it would differ between the two stamp sites (one has a request, one
does not) and refuse every multi-segment sweep beat on identity drift. Terminal
under the consent act instead; production untouched.

**RECORDED, NOT FIXED:** `by_engine.setdefault` keeps only the first clip's
receipt per engine, which LANE 2 makes newly lossy. Not a ledger hole --
per_clip keeps every clip in full, and a sweep runs one episode per cell -- and
it is pre-existing code outside the adapter lane. It already loses per-shot
render_canvas the same way, reachable today. In OPEN BUGS with that reasoning.

**THE BRIDGE DROPPED during the handoff write.** Nothing was lost: both code
chunks were already pushed with HEAD == origin verified, and the failed
`edit_block` did not half-apply -- HANDOFF_LOG was byte-unchanged when the
bridge returned. That is the second time this has happened in a remote window
and the second time push-every-green-chunk is what made it a non-event.

## 2026-07-27 -- HEAD 3acc7fed (v2.0-alpha) -- WINDOW CODER (LANE 1)
Did: froze BOTH WAN render recipes in code, mirroring B6 one tier over --
  71753cb4 wan_ti2v (11 knobs), 3acc7fed wan_i2v (the six that were read INLINE
  in _build_graph with bare int()/float(), no range check, no named refusal).
  Mechanism shared in the new wan_recipe.py, DATA per adapter, and a per-adapter
  consent act so a sweep of one tier cannot stamp +prequalification on the
  other. Closed the receipt hole: a WAN clip stamped recipe: None, now it stamps
  a real one that rides into stamp_durable(meta.render_engines). Fixed a live
  bug the fan-out found -- eng_wan_i2v measured an NVML render peak, logged it,
  and discarded it (NEWBUG-1's fix landed on wan_ti2v in July and never reached
  the sibling), so every wan_i2v clip reported vram_peak_mb: None.
Current step: LANE 2 -- name the DEPARTURES in the prequalification receipt
  (no GPU, suite-provable, already an OPEN BUG). 7d stays PARKED for the
  operator.
Next: LANE 2, or the operator picks another remote-safe lane (ShotRow wire-or-
  demote; the credits-card display gap). CODER window.
Models: Claude codes + judges; 3 Sonnet QA lenses pre-push. $0 external -- no
  codex, no agy, no cloud roundtable; two-strikes never invoked.
Commits: 71753cb4, 3acc7fed. Suite 7226 -> 7291; Bible 17; canonical 9872624A
  byte-identical. Record: docs/2026-07-27-lane1-wan-recipe-freeze-qa-findings.md.

### Detail

**THE MUTATION ROUND CAUGHT WHAT THREE QA LENSES DID NOT, and that is the
portable lesson of this session.** The pre-push fan-out found four real test
gaps and two decorative tests of my own, all fixed before the push. Then the
mutation round ran and **4 of 10 real mutants SURVIVED** the first wan_i2v pass:

1. **A renamed consent constant was undetectable.** Every test set
   `PREQUALIFICATION_ENV` -- the imported constant -- so renaming it renamed
   what the test set too. An adapter reading a var no operator will ever set
   stayed green. Tests now set the DOCUMENTED LITERAL an operator types. The
   same hole existed on wan_ti2v and was fixed there too.
2 and 3. **`recipe` and `vram_peak_mb` dropped from `render_clip` both
   survived**, because the receipts were only ever checked on a HAND-BUILT raw.
   The test constructed the thing it was verifying -- the chunk-6 shape where a
   test's own builder agrees with the bug. Fixed with a test that drives the
   real `render_clip` through an ffmpeg-free, GPU-free stub, for both adapters.
4. **`shift` escaping back to an inline os.environ read survived**, because the
   production-leg test set steps/cfg/sampler/negative and not shift, while the
   consent-act test AGREED with the mutant.

After the fixes: 20/20 and 10/10 real mutants caught, all 4 CONTROLs survived.

**WHAT LANE 1 DELIBERATELY DID NOT FREEZE, because WAN is not ltx.**
`OTR_WAN_TI2V_MAX_FRAMES` is a ceiling AND a live shipped channel --
otr_8gb_wan.json sets both launch.env and video.max_render_frames -- so folding
it into the recipe would have retired the 8 GB tier's launch contract, which is
PBUG-20260723-02 itself. Weight names and their loader-class selectors stay live
TOGETHER (the class is inferred from the basename; freezing one and not the
other gives one fact two owners). wan_i2v keeps uni_pc rather than the portable
floor's euler: the freeze preserves behaviour, it does not add policy. And the
un-namespaced OTR_WAN_* names are left alone and FLAGGED -- renaming an
operator-facing knob is the operator's call, and the freeze already removed the
power that made the missing namespace dangerous.

**HONEST LIMIT:** both v1 dicts are today's shipped defaults, NOT a measured
selection -- no WAN sweep has run. The code says so in its own words. A
prequalification run measures and produces v2.

## 2026-07-27 05:10 -- HEAD dcdcccde (v2.0-alpha) -- WINDOW RENDER
Did: ran the prequalification sweep -- four full canonical legs at 512x288, the
  first time the 8 GB tier has ever rendered at its own declared canvas. Froze
  the winner as recipe v2 (1fe7dc8c) and made both consent-act knobs fail
  CLOSED (dcdcccde). Winner: t5_device=cpu, tiled_vae=ON -- chosen on SPREAD,
  not minimum: tiled holds the peak flat at 8241-8278 MB across 17..161 frames
  where untiled climbs 8662 -> 10859 MB. T5 on GPU peaks at 16.0-16.1 GB of a
  16.3 GB card. Every cell RESULT SUCCESS + obs_publish OK + asset on disk.
Current step: LANE 1, the WAN recipe freeze (no GPU), mirroring B6. 7d stays
  PARKED until the operator is at the desk.
Next: WAN recipe freeze; then the clamped confirmation of v2 and the per-cell
  receipt chunk. CODER window (or Codex under the travel relay).
Models: Claude + 1 kibitz panel (codex gpt-5.6-sol high + agy Gemini 3.6 Flash
  High), invoked under the two-strikes law. $0 external beyond codex credits.
Commits: 1fe7dc8c, dcdcccde. Suite 7213 -> 7226; Bible 17; canonical 9872624A
  byte-identical. Record: kibitz-runs/2026-07-26-8gb-writer-ctx-blocker/r2/.

### Detail

**THE ASSIGNED STEP WAS BLOCKED BEFORE IT COULD START, AND THE PANEL RELOCATED
THE PROBLEM.** Two legs died in `OTR_LedgerScriptWriter` -- first on the
default `scifi_news` bank (`requested_output=2800` vs `provider_output_cap=512`,
a known open row), then on `media_archive` (`prompt requires 2064 input tokens,
context_cap=2048`). Switching banks was my one fix and it failed, so per the
two-strikes directive I stopped and ran `/kibitz` before writing any code.
Both seats independently reached the same diagnosis as my anchor: the 8 GB
profile family pairs a 12B GGUF writer with a 2048 context that cannot fit the
pipeline's own prompts, and raising ctx is the wrong fix because 4096 puts the
writer near 9.5 GB on the card the tier exists for.

**THE OPERATOR SUPPLIED THE ACTUAL ANSWER MID-SESSION:** there is no tier --
whoever runs the workflow picks the LLM, and the 8gb/16gb variants will be the
same canonical JSON saved with different dropdowns, no auto profile selection.
Grounding that showed the canonical JSON ALREADY carries `gguf_n_ctx=4096` /
Q8_0 / ceiling 14.5, and that passing `-Profile otr_8gb_ltx` silently replaces
those widgets from the profile's `llm` block while the runner's echo prints
only 16 role/slot/feature overrides. Running with `-Profile none` plus the
shipped `OTR_FORCE_ENGINE_MAP` route authority unblocked the sweep immediately.

**WHAT THE SWEEP PROVED BEYOND THE RECIPE.** B5 end to end: the canonical JSON's
VideoDirector says 832x480 and the engine still rendered 512x288, because the
canvas is a static declaration. B6's marking requirement: the ledger carries
`+prequalification`, so a sweep artifact is not mistakable for a published one.
And chunk 1a's fail-closed force map refused a JSON-shaped map BY NAME before
anything rendered -- my formatting error, caught exactly where it should be.

**SIX TESTS WENT DECORATIVE THE MOMENT THE DEFAULT FLIPPED.** Every override
that said `tiled_vae=1` now AGREED with the frozen value, so it could no longer
tell whether the recipe or the environment had won. Each now sets the OPPOSING
value and asserts what it opposes. This is the same class the B6 panel caught,
and it will recur on the WAN freeze -- it is written into the CURRENT STEP.

**HONEST LIMIT, RECORDED IN CODE AND IN GO_FORWARD:** `VramPeakProbe` samples
machine-wide NVML and the sweep ran unclamped (the profile-free writer is ~13 GB
at Q8_0 and cannot coexist with an 8 GiB reservation), so the absolutes are not
a proof of 8 GB fit. They support the RANKING, which is what selects a recipe.
A clamped confirmation of the winner alone is owed.

**NOT DONE, DELIBERATELY:** 7d (operator-parked), the profile ceiling pin (a
production planning decision I flagged rather than took), and the per-cell
receipt enrichment (touches `session_identity` and several call sites, so it is
its own chunk rather than a rider on a green one).

## 2026-07-27 09:40 -- HEAD 906031be (v2.0-alpha) -- WINDOW CODER A, SESSION 5c
Did: pushed B6 (906031be) -- the ltx_8gb render recipe is FROZEN IN CODE as
  LTX8_RECIPE_V1; its env vars bind only under an explicit
  OTR_LTX_8GB_PREQUALIFICATION consent act, and outside it they are NAMED in a
  warning and never PARSED. Resolved the operator fork as (a) freeze today's
  defaults, with the code stating plainly that these are shipped defaults and
  not a measured selection. Answered the open "what marks a prequalification
  run" question with an explicit env var, never an ambient condition.
  A sweep now stamps a "+prequalification" recipe receipt so a measurement
  artifact is not mistaken for a published render in meta.render_engines.
Current step: prequalify 512x288 -- a GPU step, so a RENDER window owns it, not
  a coder window. The CODER A 8GB code block is COMPLETE.
Next: boot with OTR_LTX_8GB_PREQUALIFICATION=1, measure T5 device on/off and
  tiled decode on/off at 512x288, freeze the winner as recipe v2 (bump the
  version inside the RECIPE_LTX8_I2V string -- it moves the session identity for
  free), then 7d, the canonical 237-frame opening beat. RENDER, then CODER A.
Models: Claude + 5 Sonnet lenses. $0 external. No codex, no agy, no cloud
  roundtable; two-strikes never invoked (no fix needed a third attempt).
Commits: 906031be. Record: docs/2026-07-27-b6-qa-findings.md. Suite 7158 ->
  7213; Bible 17; canonical 9872624A byte-identical.

### Detail

**THE PANEL FOUND THE HOLE IN THE FREEZE ITSELF.** The first draft demoted the
sampling knobs and left `OTR_LTX_8GB_NEGATIVE` -- a render input, read straight
from `os.environ` on every leg -- plus four tiled decode-geometry vars, still
binding from the server's boot. Two independent lenses found it separately, and
a third traced the ledger: the draft stamped the SAME recipe receipt on a
prequalification sweep as on production, so the two were indistinguishable in
`stamp_durable(meta.render_engines)`. Grounding confirmed all three against the
real files. The negative-prompt hole was the worst of them --
`render_driver.build_request_from_shot` never populates `negative_prompt` for
video shots, so the boot environment was the SOLE author of that conditioning.

**THE FIX FOR THE GEOMETRY THEN CREATED ITS OWN DEFECT,** which is why the
post-fix panel exists: gating the four tile vars left a SECOND range-check
implementation that swallowed a bad value and substituted the default, where
every sibling knob raises MALFORMED_CONFIG. A sweep could mistype the value it
was measuring, render at something else, and stamp a receipt saying it had
measured it. Collapsed into one `_config_number` shared by both, plus
`_VAE_TILE_BOUNDS` from the live /object_info capture so a value under the
node's own floor is refused by name instead of dying inside ComfyUI.

**SIX DECORATIVE TESTS CAUGHT.** Neither warning's DIRECTION was pinned -- both
bodies name the knob, both interpolate the recipe, and both contain the
substring "PREQUALIFICATION" because it is inside the env var's own name, so
swapping them stayed green. The recipe-delivery test had become a comparison of
the resolver against itself (post-freeze a clean env returns the frozen
constants, so a hard-coded literal in `_build_graph` compares EQUAL) -- its own
docstring claimed to catch exactly that. `assert "FROZEN" not in caplog.text`
was vacuous. Three `_ENVS` scrub lists claimed completeness they did not have;
each now carries a test asserting it covers `_RECIPE_ENV_KEYS`.

**THREE PANEL CLAIMS DISCARDED after grounding,** with reasons recorded: the
ceiling's two owners (real, but pinning the profile changes how a 237-frame
beat partitions -- a production decision on the eve of 7d, so it is an OPEN BUG
with the shape written into the preset's own `_ceiling_note`); the credits card
never drawing the recipe (real, but a DISPLAY gap -- the durable ledger does
carry it -- so the docstring was narrowed to claim only what is true); and
rewriting the arc judgment's "MEASURED" wording (refused -- a judgment is a
record of what was decided, not a living doc, and rewriting it would destroy
the evidence that the ordering was departed from).

**Mutation:** two rounds, 13/13 and 10/10 real mutants caught, all four CONTROL
(semantically equivalent) mutants survived -- the harness discriminates rather
than reporting red on everything.

**Scouted for a future chunk, nothing touched:** both WAN adapters carry the
whole pre-B6 defect. `eng_wan_ti2v` reads loader class, tiled-VAE class, all
three weight NAMES, sampler, scheduler, steps, cfg, shift, negative and four
VAE-tile vars from the environment; `eng_wan_i2v` reads six INLINE in
`_build_graph` with bare `int()`/`float()` -- no range check, no named refusal.
Neither emits a recipe receipt at all, so a WAN clip stamps `recipe: None`:
there is not even a wrong receipt to catch the drift with.

## 2026-07-27 05:30 -- HEAD a0141cdd (v2.0-alpha) -- WINDOW CODER A, SESSION 5b
Did: pushed B5 (a0141cdd) -- ltx_8gb now declares its own render canvas
  (512x288) as a static class attribute, build_request_from_shot consumes the
  declaration last in its canvas chain, and render_beat_coverage pre-flights it
  before BeatSession opens. Plus the drift guard the O1 judgment asked for: the
  profile's render.canvas_w/h and the 8 GB variant's director widgets are pinned
  equal to the declaration.
Current step: B6 -- and it is BLOCKED on an operator call, not on code.
Next: operator rules on B6 (a) freeze today's defaults as recipe v1 now, or
  (b) defer B6 until after prequalification -- plus what signal marks a run as
  "prequalification". Then prequalify 512x288, then 7d. CODER A.
Models: Claude + 4 Sonnet lenses + 2 agy (kibitz, Gemini 3.6 Flash High). $0
  external. No codex spend; two-strikes never invoked.
Commits: a0141cdd. Record: docs/2026-07-27-b5-qa-findings.md. Suite 7134 ->
  7158; Bible 17; canonical 9872624A byte-identical.

### Detail

**THE POST-CODE PANEL SENT THE DESIGN BACK, AND IT WAS RIGHT.** B5 was written,
green and mutation-proven with 10 mutants when a seat pointed at a document I
had read and mis-weighted: `docs/2026-07-26-o1-canvas-arc-judgment.md` -- one of
the THREE authorities GO_FORWARD names for this step -- lists the
`render.canvas_w/h -> canonical_canvas` channel as the one DEAD channel of five
and rules that the engine must declare its canvas STATICALLY, "not an env var,
NOT A LEDGER READ". I had built the ledger read, following the later 8gb
judgment's B5 paragraph, which says the opposite and never reconciles the two.
Verified against the file before acting, not taken on the seat's word.

**THE PANEL ALSO SUPPLIED THE EVIDENCE THAT DECIDES IT ON THE MERITS**, which is
why this was not a coin-flip between two docs.
`tmp/_run_canonical_engine_matrix_20260723.py` routes ltx_8gb onto the CANONICAL
832x480 workflow through profile role_overrides and copies no canvas -- and its
author had already special-cased the WAN sibling for exactly this reason
("Applying only the engine name silently discarded its 832x480/17-frame render
contract"). Under the ledger-reading design that live QA campaign, which still
owes a requalification leg, would pillarbox or be REFUSED outright. **A
declaration cannot be displaced by where it is pointed.**

**THE DRAFT WAS FAIL-CLOSED IN THE WRONG DIRECTION**, and a seat named it
precisely: the exact-16:9 clause was "a quality judgment wearing a structural
gate's clothes" -- the render would have completed, the asset would have
existed, the ledger would have stayed usable, and what was refused was the LOOK
of a composite. Under the declaration there is no cross-engine refusal at all;
the only remaining error is a code-integrity check on a broken declaration, the
shape of FrameContract.__post_init__.

**A FACT THAT CORRECTS THIS FILE'S OWN EARLIER CLAIM:** render_single and both
HTTP entry points never reach the canvas seam -- they use the older ledger-free
build_request and default to OTR_VIDEO_RENDER_CANVAS (832x480). So the
7d-preflight recorded as "GPU IS PROVEN" ran at 832x480, NOT at the production
canvas. The production canvas for ltx_8gb has still never rendered live.

**TWO ERRORS OF MINE THE TESTS CAUGHT, worth naming because both were sloppy
arithmetic dressed as rigour.** 512 does not divide 1920 -- the scale is 3.75x
-- so my "zero pad area" assertion checked divisibility and was simply wrong;
the property that matters is that the rectangles are the same SHAPE
(w*1080 == h*1920). And the malformed-declaration check ran AFTER the int
conversion, so a stringly-typed "512x288" parsed as 5x1 and was refused for the
wrong reason, naming the latent grid instead of the real mistake. Shape is now
checked before value.

**Mutation: 11 mutants, 9 defect all red** -- including the resolver answering
None, the engine declaring the landscape canvas, the engine declaring nothing, a
string slipping the shape check, and the PROFILE drifting from the declaration
-- **2 controls green**, baseline and restore green.

**WHY B6 STOPPED HERE rather than being attempted.** B6 says freeze the MEASURED
selection; section 7 of the same judgment orders "build mechanics first, MEASURE
second, freeze third" -- and no measurement exists, because prequalification is
the NEXT step and no GPU run is authorised in a coder window. Executing B6 now
would mean inventing both the frozen values and the signal that marks a run as
"prequalification". Both are operator calls; they are written up with defaults
in GO_FORWARD's CURRENT STEP rather than guessed at unattended.

## 2026-07-27 02:05 -- HEAD 5929e19a (v2.0-alpha) -- WINDOW CODER A, SESSION 5
Did: pushed B3 (b23fc035, the tier ceiling now narrows the coverage contract for
  ltx_8gb ONLY, with the WAN topology regression in the same commit) and B4
  (5929e19a, the ltx_8gb ping-pong deleted, _ltx8_frame_length deleted with it,
  the ladder moved onto the engine's own frame_contract). Ran a fan-out BEFORE
  and BEFORE-THE-PUSH on each chunk -- every lens in ONE block, concurrently.
Current step: B5 + B6 -- the canvas seam fail-closed BEFORE BeatSession opens,
  then freeze the measured recipe in CODE.
Next: B5+B6, prequalify 512x288, then 7d (the canonical 237-frame beat, where a
  GPU first renders through this machine). CODER A.
Models: Claude + 10 Sonnet lenses + 4 agy (kibitz, Gemini 3.6 Flash High). $0
  external. No codex spend -- the architecture was already panel-decided in the
  8gb judgment and no fix needed a second attempt (two-strikes never invoked).
Commits: b23fc035, 5929e19a. Records: docs/2026-07-27-b3-qa-findings.md,
  docs/2026-07-27-b4-qa-findings.md. Suite 7097 -> 7134; Bible 17; canonical
  9872624A byte-identical throughout.

### Detail

**THE PRE-CODE PANEL REFUSED THE JUDGMENT'S OWN B4 RECIPE AND IT WAS RIGHT.**
The plan said: refuse when the ask exceeds the cap, delete the CLIP-FILL block,
let an off-grid ask render short. Two seats independently showed that ships a
REGRESSION. The old pad fired whenever the decode came up short FOR ANY REASON
-- not just a cap disagreement -- and it LOGGED when it did. Delete it with only
a cap refusal and a short clip flows into `otr_silent_composite`, which
hard-loops it with `-stream_loop -1` AND suppresses its own underrun warning
once loop-fill activates. A logged mirror traded for a silent jump-cut repeat,
on the majority path. So what shipped is different: `_ltx8_frame_length` is
DELETED (its snap-DOWN was the whole reason the pad had to exist), the ladder
moved to `frame_contract.smallest_legal_at_least` -- the same object the planner
partitions against -- and an off-grid ask now renders the next legal rung UP and
trims the surplus in REAL frames. 100 renders 105 and keeps 100.

**TWO OF AGY'S THREE B3 MUST-FIXES DID NOT SURVIVE GROUNDING.** Rejected: routing
engine_id through `resolve_engine_id` inside the derivation (the registry gate
already returns before it for any unregistered spelling, and a second
normalization authority would make an id the registry REJECTS behave as
ltx_8gb); and defaulting the new required parameter to 0 (that is the silent
fallback shape this build removes -- and the claimed broken test callers do not
exist, the only occurrence in tests/ is inside a docstring). Also rejected:
comparing the receipt on every field EXCEPT engine_id. A plan built under one
engine's ceiling and executed by another must refuse; that is what
`test_the_legacy_path_validates_the_plan_against_the_FINAL_engine` already
establishes one contract down.

**THE POST-CODE PANELS FOUND SIX DEFECTS IN GREEN, MUTATION-PROVEN CODE.** B3:
the unresolved-engine branch compared the ceiling but never the ENGINE (two
seats, two live repros -- a stale ltx_8gb receipt on a swapped shot sailed
through to an arithmetic-only check); a malformed receipt read as no receipt;
the discrete-menu guard refused ceilings that never bound it, breaking the
function's own documented guarantee; and `profile_max_render_frames` was a
FOURTH hand-copied normalization that `eng_wan_ti2v` reads at render time -- in
a test whose name promised "exactly one normalization" and never touched the
site its own docstring cited. B4: the module docstring still advertised the
ping-pong, and `_LTX8_MIN_FRAMES` could drift from the contract floor it
duplicates. All fixed before the push.

**AND ONE I CAUGHT MID-WRITE, which is the one worth remembering.** The first
draft of the B3 stamp site rebound one variable and fed the ALREADY-NARROWED
contract into `coverage_contract_receipt`. A narrowed contract narrows to
itself, compares equal, returns None -- so the receipt would have silently never
existed and the render boundary would have had nothing to check. Every test in
the file would still have passed. It is now pinned by name.

**MUTATION FOUND A HOLE THE TESTS COULD NOT SEE:** validating the plan against
the NARROWED contract was unobservable, because the receipt equality fires first
in every scenario the tests covered. The test that makes it load-bearing is a
receipt-VALID ledger whose PLAN was tampered with -- the hand-edited or replayed
case the second boundary exists for.

**Totals: 26 mutants across both chunks** (22 defect all red, 4 controls all
green, baselines and restores green). The controls move values the recipe is
entitled to move -- the env cap default, WAN's default clip length, the recipe
receipt string -- and prove the assertions read the DECLARED contract rather
than secretly pinning an env knob.

**Declined on purpose:** agy's test that `extract_terminal_frame` reads frame
`target-1` from a TRIMMED clip. Both seats proved the trim cannot fire on a
chained segment (every planned length is already legal, so the strict inequality
is false, and the single-clip path never chains), so that test would assert a
state production cannot construct.

**PROCESS NOTE:** B3 is production-inert until a profile pins an ltx_8gb
ceiling, and B3 shipped with "do not pin one before B4 lands" because the
ping-pong laundered the disagreement. B4 has landed, so that constraint is
lifted -- pinning the ceiling is now part of the prequalification step.

**Harness gotcha worth not relosing:** a mutation harness that reads with
universal newlines and writes with `newline=""` silently rewrites a CRLF file as
LF, and the restore leaves a phantom modified file that `git diff` shows as
empty. Read AND write with `newline=""`.

## 2026-07-26 22:40 -- HEAD d708408d (v2.0-alpha) -- WINDOW CODER A, SESSION 4
Did: pushed B1b-0 (b214481b, the regression net ltx_8gb never had) and B1b
  (d708408d, the loader hoist). The post-code panel on the NET killed the
  previous session's own acceptance criterion: the two assertions it declared
  would FLIP under the hoist structurally could not, so nothing in it would
  have gone red against a hoist that silently did nothing. Corrected before
  writing the hoist. B1b then hoisted the CHECKPOINT ONLY into prepare() and
  moved the 4 GiB integrity floor into a shared helper called BEFORE the lease.
Current step: B3 + B4 -- the LTX-only effective contract, then delete ping-pong
  with the WAN max_render_frames regression in the same commit.
Next: B3+B4, then B5+B6, prequalify 512x288, 7d. CODER A.
Models: Claude + 3 Sonnet lenses + 2 agy (kibitz, Gemini 3.6 Flash High). $0
  external. No codex spend this session -- the design was already panel-decided
  in the 8gb judgment and no fix needed a second attempt.
Commits: b214481b, d708408d. Records: docs/2026-07-26-b1b0-qa-findings.md,
  docs/2026-07-26-b1b-hoist-qa-findings.md.

### Detail

**THE NET COULD NOT SEE THE THING IT WAS BUILT FOR.** `test_THE_LOAD_COUNT_...`
was written to state the defect as a number and flip 3 -> 1. It cannot: under
the decided design `_build_graph` stays conditional, and that test hands
`render_clip` a HAND-BUILT `prepared = {"patchers": []}` with no
`external_results`, so it stays on the unsupplied branch forever. Same for
`test_the_graph_carries_ITS_OWN_loader_nodes_today`. The Sonnet over-pinning
lens and agy reached that independently. Editing the literal `3` to `1` when the
hoist landed would have produced a red that looked like a broken hoist and was
actually a harness gap. Both are now CONTROLS with docstrings that say so;
EXACTLY ONE assertion flipped (`external_results` appearing in the executor
kwargs); and the 1-load proof was written WITH the hoist, calling `prepare()`.

**THE FLOOR WAS THE REAL BLOCKER AND ITS POSITION IS THE FIX.** `assert_usable`
owns the 4 GiB checkpoint-integrity floor and runs PER SEGMENT, after
`BeatSession` opens -- so moving the real load into `prepare()` put it ahead of
the only size check in the adapter, and `resolve_session_config` proves
existence and takes a receipt but never size. It is now a shared helper, called
BEFORE `super().prepare()` takes the cross-process lease. Two mutants pin the
POSITION (`FLOOR_runs_AFTER_the_lease_is_taken`,
`FLOOR_dropped_from_prepare_entirely`), not just the presence.

**THE PANEL ALSO FOUND A REAL COVERAGE HOLE IN THE NET:** prompt polarity was
never pinned on any hop. A positive/negative swap renders the negative prompt --
it does not crash, does not shorten the clip, and no forward test could see it
because the fakes never inspect what they are handed. And `_ltx8_frame_length`
had ZERO coverage anywhere in the suite, though B3/B4 rest on its `8n+1` snap.
Both closed.

**Mutation: 29 mutants, 27 defect + 2 CONTROL, all proven**, both baselines
asserted failed=0. The CONTROL mutants are new this session -- they move values
the recipe is entitled to move (its step count, its default checkpoint name) and
must break nothing, which is what proves the new assertions compare against the
resolver instead of secretly pinning literals.

**Raised by the panel, out of scope, recorded so it is not lost:**
`MotionEngineBase` has no re-entrancy guard, so a second `prepare()` on one
engine instance with no teardown between blocks the full 120s lease timeout
rather than failing fast (the owner PID is this same live process, so the
stale-lock reclaim never fires). And the checkpoint's embedded VAE at slot 2 has
never been handed to `_detach_patchers`, here or in any sibling adapter. Both
are family-wide, both pre-date the hoist; they belong in one ticket across the
engine family, not in an `ltx_8gb` chunk.

**PROCESS, and it cost an hour:** the three Sonnet lenses on B1b-0 ran
SEQUENTIALLY. Fan-out lenses go out in ONE block -- ~20 minutes concurrent
instead of ~50 serialized. Nothing about the findings changed; only the clock.

## 2026-07-26 15:40 -- HEAD 095be05b (v2.0-alpha) -- WINDOW CODER A, SESSION 3c
Did: closed the identity lie on BOTH remaining channels, with the operator's
  fan-out-BEFORE-and-AFTER discipline on each. 823b9929 routed _ckpt_path /
  _t5_path through _loader_token_path so the SINGLE-CLIP gate (assert_usable)
  and the multi-segment gate (session_identity) can no longer disagree about
  which file is the checkpoint. 095be05b made a *_DIR override that the LOADER
  cannot see terminal -- ComfyUI resolves the graph's bare basename through
  folder_paths, and *_DIR never touched that channel, so it has never changed
  which weights render.
Current step: B1b -- hoist the loaders into prepare().
Next: B1b, then B3+B4, B5+B6, prequalify 512x288, 7d. CODER A.
Models: Claude + 2 kibitz rounds (codex gpt-5.6-sol + agy Gemini 3.6 Flash High)
  + 2 Sonnet lenses + 1 Fable pass. $0 external.
Commits: 823b9929, 095be05b. Judgment:
  docs/2026-07-26-dir-override-arc-judgment.md.

### Detail

**THE PRE-CODE PANEL KILLED MY OWN PROPOSAL, TWICE.** For the single-clip gap I
proposed routing `assert_usable` through `resolve_session_config`. The panel
showed that would have silently dropped the 4 GiB checkpoint integrity floor --
the resolver has no size check at all, and the floor had ZERO test coverage, so
nothing would have failed. The shipped fix delegates only the RESOLUTION and
leaves `assert_usable`'s body untouched, which keeps the floor alive by
construction rather than by remembering to port it.

For the `*_DIR` arc the panel supplied the evidence that scoped the change:
`tests/test_wan_loader_preflight.py` says in its own docstring that the `*_DIR`
envs are its MOCK SEAM for a box with no ComfyUI runtime. So the Wan adapters
carry the identical lie but cannot be fixed until those fixtures migrate --
`wan_shared` took an ADDITIVE split only (`_resolve_model_file_by_token` out,
`_resolve_model_file` still calling it), and a control mutation proves Wan's
DIR-wins precedence survived. The panel also refuted the obvious alternative,
registering the folder from preflight via `folder_paths.add_model_folder_path`:
ComfyUI ships no unregister, so a CHECK would have permanently mutated global
process state for every later engine on the same server.

**THE POST-CODE PANEL CAUGHT A DECORATIVE TEST OF MINE.** The `*_DIR` test that
pins WHICH guard runs first pointed both env vars at the same decoy file. That
makes the explicit guard's condition trivially false, so the test would have
passed under the very branch swap it claimed to detect -- green, well-named, and
proving nothing. Fixed with a third distinct decoy, and a new mutation that
performs a REAL precedence swap now fails it. Two independent lenses (Sonnet,
agy) also converged, without seeing each other, on three messages whose own
remediation advice ("fix OTR_LTX_8GB_T5_DIR", "or set OTR_LTX_8GB_CKPT") led the
operator straight into the new refusal. All three now name
`extra_model_paths.yaml`, the channel that actually reaches the loader.

**Mutation proof: 8 mutants, 0 control breaks** (`tmp/_kbA_dir_mutate.py`,
baseline asserted failed=0 first so a blind harness cannot pass silently). Two
of the eight name CONTROLS as their target, which is what proves the controls
have teeth rather than merely being green.

**Still open, deliberately:** the Wan adapters' copy of both lies (blocked on
their fixtures); no test creates a real NTFS junction; live-box confirmation
that `extra_model_paths.yaml` folders come back from `folder_paths.get_full_path`
in-process.

## 2026-07-26 12:05 -- HEAD fdeee600 (v2.0-alpha) -- WINDOW CODER A, SESSION 3b
Did: ran the POST-CODE QA fan-out that should have run before the session-3
  pushes and did not -- operator caught the omission. codex gpt-5.6-sol + agy
  Gemini 3.6 Flash High via kibitz, plus FOUR Sonnet lenses. It found FIVE code
  defects and six test defects in already-green, already-pushed code. Fixed all
  five: ea1652f9 (C-1 path guard + C-4 stat + the misnamed control + env leak),
  f33c5e15 (C-3 stranded GPU lease), fdeee600 (C-2 terminal + C-5 named error +
  the keep= coverage hole).
Current step: unchanged -- B1b, plus a new chunk ahead of it (route
  assert_usable through the one resolver; the identity-lie fix currently
  protects only multi-segment beats).
Next: close the single-clip resolver gap, then B1b. CODER A.
Models: Claude + 1 kibitz round (2 calls) + 4 Sonnet lenses. $0 external.
Commits: 5799544e, ea1652f9, f33c5e15, fdeee600.

### Detail

**THE PROCESS MISS IS THE HEADLINE.** The kickoff said "fan out for QA before
each push". I pushed three chunks without it, and only ran it when the operator
asked whether it had happened. It then found, in code that was green,
mutation-proven WITH controls, and already on origin:

**C-1, a live FALSE REFUSAL.** The new path guard compared with
`os.path.abspath`, which folds neither case nor junctions. NTFS is
case-insensitive and this box reaches its own repo through a junction, so
`C:\Models\x` vs `c:\models\x` -- the SAME file -- raised MALFORMED_CONFIG on
every multi-segment beat. A guard written to stop the receipt describing the
wrong weight was refusing the right one. Found by all four sources
independently. **It shipped because the control test was named
`..._case_and_separator_tolerant` and only varied the SEPARATOR** -- the name
promised exactly the coverage that was missing.

**C-3, a stranded GPU lease.** `BeatSession.open()` reads the identity a second
time AFTER `prepare()` has taken the cross-process lease. B2b made that read do
file I/O, so it can now raise -- and when `__enter__` raises, Python never calls
`__exit__`, so teardown and the lease release never ran. The owner is the live
ComfyUI process, so the PID-liveness reclaim could not help either: every later
heavy render blocked its full timeout until someone killed the server by hand.
None of the 38 existing beat-session tests construct an engine whose identity
succeeds once then raises.

Also C-2 (`terminal` validated against `results`, which is now seeded with
externals -- so a mistyped terminal returned the caller's own handle as if it
were a render), C-5 (a missing wire source lost its NAMED error), and the
`keep=` mutation survivor: `keep |= set(ext)` -> `keep = set(ext)` passed the
ENTIRE suite while silently discarding the caller's keep on every production
call, freeing the MODEL patcher before teardown grabs it. `keep=` had zero
direct coverage anywhere.

**TWO TESTING LESSONS, both learned the hard way this session.**
The first C-4 test monkeypatched `os.stat` -- process-wide -- and broke pytest's
own traceback machinery with an INTERNALERROR. Model the real race; never patch
the interpreter out from under the runner. And my first "control" for the `keep`
fix ALSO asserted the feature, so deleting the fix broke the control and the
harness reported CONTROLS_broken. **A control must fail under OVER-tightening
and pass under correct behaviour -- never mirror the feature it bounds.** Caught
by the mutation harness, not by review, which is the argument for running the
harness against the controls too.

**STILL OPEN, and it is the real close of the defect B2a was written for:**
`resolve_session_config` runs ONLY for multi-segment beats, so the identity-lie
bug is still fully open on the single-clip path. `assert_usable` still uses the
old `_ckpt_path()`. The QA lens proved it live -- green preflight, raising
resolver, same environment.

Suite 7023 -> 7045 passed / 27 skipped / 1 xfailed. Bible 17. Canonical
byte-identical 9872624A throughout.

## 2026-07-26 10:10 -- HEAD 582dfbd8 (v2.0-alpha) -- WINDOW CODER A, SESSION 3
Did: two full kibitz arcs (r1-r4 each, 16 agent calls, codex gpt-5.6-sol high +
  agy Gemini 3.6 Flash High verified every round) plus ONE operator-requested
  Fable pass on the viewer question; then three green chunks -- B1a `8caf3516`
  (run_graph external_results + on_result), B2a `55c8a811`
  (resolve_session_config), B2b `582dfbd8` (ltx_8gb session_identity).
Current step: B1b -- hoist the loaders into prepare(). BeatSession now OPENS a
  multi-segment session but the weights are still re-loaded per segment.
Next: B1b -> B3+B4 -> B5+B6 -> prequalify 512x288 -> 7d (237-frame beat). CODER A.
Models: Claude + 2 kibitz arcs (16 calls) + 1 Fable. $0 external.
Commits: 78df72b9, 6c345e06, 8caf3516, 55c8a811, 582dfbd8.

### Detail

**O1 WAS NEVER THE ONLY 7d BLOCKER, and finding that out was the session.**
`session_identity` appeared in exactly ONE file -- `beat_session.py` -- and no
adapter declared it, so `BeatSession.open()` refused EVERY multi-segment beat
for all 31 engines, before the weights land, no fallback. A 169- or 237-frame
beat was rejected before the render canvas was ever consulted. Lifted for
`ltx_8gb` at `582dfbd8`.

**THE PANEL KILLED FIVE OF MY CLAIMS, and one of them was my whole argument.**
I had priced the canvas failure through `compute_real_frame_budget` -- 43 GB at
1472x832, 12.4 GB at 512x288. That gate is called by exactly ONE engine,
`eng_wan_ti2v.py:399`. `eng_ltx_8gb` declares "NO VRAM/NVML/vendor gate" and
treats its NVML probe as telemetry only: *"the operator's tier JSON owns the OOM
budget."* So the real failure at 1472x832 x 161 frames is a CUDA OOM mid-render,
not a clean refusal -- worse, not better -- and the engine explicitly delegates
its budget to the tier JSON whose canvas never arrives. Also refuted: "22 of 23
stamps are wrong" (the two 16GB LTX profiles are correct, because their engines
have branches), "1472x832 is the deliverable" (`composite_w/h` maps 1920x1080),
and my acceptance oracle, which compared a stamp against a value derived from
the same request -- circular.

**I ALSO CAUGHT MYSELF ONCE, BEFORE THE PANEL SAW IT.** The 7d-preflight that
"proved the GPU" ran at 832x480, not 1472x832: `render_single` is a FIFTH canvas
channel (`OTR_VIDEO_RENDER_CANVAS`) and never calls `build_request_from_shot`.
The harness that proved the GPU renders at a different canvas than the
production path it was proving. Correction filed against the 7b judgment.

**THE CROSS-TIER TRAP.** I was about to reuse `max_render_frames` as the segment
cap. It is not a planning cap: WAN reads 17, renders short, then PING-PONGS to
the beat length, so applying it before `partition_beat()` would have turned every
WAN beat into a pile of 17-frame renders and silently rewritten the tier
`PBUG-20260723-02` just fixed. Corollary that settles the operator's question:
ripping ping-pong is LANE-SPECIFIC -- a correctness hole for `ltx_8gb` (a short
render passes the count gate wearing a planned length), load-bearing for WAN.

**THE OPERATOR'S 512x288 WAS RIGHT ALL ALONG.** Four sources agree.
512x288 and 1024x576 are the only exact-16:9 /32-clean rungs; 832x480 is 26:15
and pillarboxes to 1872x1080. Fable settled the choice between the two:
*"Softness is a state; a motion reset is an event... soft reads as OLD, stutter
reads as BROKEN."* My earlier instinct to "correct" the profile up to 832x480
would have put side bars on every episode.

**AN OPEN BLOCKER DISAPPEARED.** Acceptance moves from 169 to 237 -- the
canonical assembler already ships `opening_duration_sec=10` / `crossfade_ms=500`,
which yields `round((10-0.5)*25) = 237`. At a 65 cap: `[65,65,65,49]` -> 241
chained -> trim 4 -> 237, every segment `8n+1` (arithmetic verified). That CUTS
O4's profile-schema work entirely, and 237 is a stronger test than 169 because it
exercises tail trim.

**MUTATION DISCIPLINE PAID TWICE.** The first B1a mutation run reported failed=0
for every mutant -- the KNOWN-FAIL-GUARD intercepts pytest's short summary and
prints its own nodeid block, so the harness was blind, not the fix unproven.
Re-ran isolated, fixed the parser, then trusted it. And I caught a decorative
test of my own before it shipped: a `free_after_use` case whose assertion was
`assert res == (0,) or True`. Every mutation since carries controls; all 8
across B1a/B2a broke ONLY their targeted test and ZERO controls.

**Bridge dropped mid-session** and recovered; nothing was lost because the last
green chunk was already pushed -- which is the actual argument for the push rule.

Suite 6983 -> 7023 passed / 27 skipped / 1 xfailed. Bible 17. Canonical
byte-identical `9872624A` throughout (no node/widget/link touched).
Bug Bible promotion of `PBUG-20260723-02` DEFERRED by the operator to build end.

## 2026-07-27 06:45 -- HEAD 0d148ba5 (v2.0-alpha) -- WINDOW CODER A, SESSION 2
Did: full r1->r4 kibitz arc on the four 7b blockers (8 agent calls); landed C1
  (canonical `max_render_frames` descriptor), C2 (plan-vs-output fail-open),
  C1b (the same dead widget in all 11 variants, incl. the WAN 8GB 17-frame
  ceiling); proved the GPU live and confirmed the server path is a junction.
Current step: O1 -- the canvas. `build_request_from_shot` overwrites every
  non-face engine to 1472x832 with no `ltx_8gb` branch, displacing the 8GB
  tier's 512x288. Hard 7d blocker, deliberately left for a rested decision.
Next: O1 canvas -> C3 per-engine policy registry + typed taxonomy. CODER A.
Models: Claude + 1 full kibitz arc (codex gpt-5.6-sol high + agy Gemini 3.6
  Flash High), 8 calls, $0 external.
Commits: c8cf0b07, 7f4644a1, ac609d25, 8f41af27, 0d148ba5.

### Detail

Did: **a full r1->r4 kibitz arc on the four 7b blockers, then landed three
slices off it -- and proved the GPU.** Operator asked for /kibitz on the
blockers so GPU testing could start, optimising for the cleanest end-state
architecture. 8 agent calls (agy + codex `gpt-5.6-sol`, pinned and verified
every round). Authority: `docs/2026-07-27-7b-blockers-arc-judgment.md`.

Suite **6925 -> 6983 passed / 27 skipped / 1 xfailed**. Bible 17. Link
validator 0 violations. Commits: `c8cf0b07` (r1 fold-in), `7f4644a1` C1,
`ac609d25` C2, `8f41af27` C1b, plus this handoff. HEAD == origin after each.

**THE GPU IS PROVEN, AND ONE ASSUMPTION UNDER IT WAS NEVER CHECKED.** The
headless server loads the node from `C:\Users\jeffr\ComfyUI-Installs\...`, not
from this repo. It is a **junction** -- identical SHA-256, same git HEAD -- so
live results are valid. Every "live proof" in this build has rested on that and
nobody had verified it. First live render of the multi-clip architecture:
`ltx_8gb`, 25 frames, 20.8s, `frame_count=25` exactly as asked, VRAM 3004 MB.
Labelled `7d-preflight`, NOT qualification -- codex correctly pointed out that
7c still owns two of 7d's own acceptance properties.

**LANDED.** C1 `7f4644a1`: node 87's `max_render_frames` input descriptor. The
widget VALUE was present as `widgets_values[14]`; the descriptor never was, so
the 8GB ceiling channel was severed at its first hop. C2 `ac609d25`: the
plan-vs-output proof read `if got and got != ...`, so a clip reporting 0 -- or
omitting `frame_count`, which defaults to 0 -- skipped the check and got
assembled. C1b `8f41af27`: the same dead widget in ALL ELEVEN variants.

**C1b IS THE ONE TO REMEMBER.** It came from an agy r4 *verify-at-build* line,
not a MUST-FIX. `variants/otr_8gb_wan.json`'s orphan value was **17**, not the
harmless 0 -- matching `config/profiles/otr_8gb_wan.json:56`, the only shipped
profile that pins `max_render_frames`. So the WAN 8GB ceiling was deliberately
configured and silently ignored since it shipped: exactly the failure
`test_floor_max_override_is_an_absolute_hard_cap` was written after. **It was
found because the wiring script REFUSED an unexpected value instead of assuming
one** -- I had coded the precondition as "trailing value must be 0", it hit 17,
and stopped. Fixing only the canonical would have left it live.

**Mutation proofs had CONTROLS this session.** C2: restoring the fail-open
fails all six unreadable cases while the honest-count and wrong-but-readable
tests still PASS -- so the tests are specific, not a blanket refusal. I also
nearly shipped a decorative C1 test: the first mutation run reported only one
failure, so I re-ran isolated rather than assuming, and confirmed
`test_every_widget_value_has_an_input_descriptor[87] FAILED` with 14-vs-15
counts. **Re-run the mutation isolated when the guard output is filtered.**

**STOPPED SHORT OF THE CANVAS FIX ON PURPOSE.** Both seats independently found
it and it IS the 7d blocker: `build_request_from_shot` overwrites the canvas to
1472x832 for every non-face engine (`render_driver.py:2268-2273`), with
deliberate per-engine branches after for `ltx_video`/`ltx_av` but none for
`ltx_8gb`, so the 8GB tier's 512x288 is displaced on the tier that exists
because 8GB cannot afford the big canvas. Not fixed because the two seats
prescribe DIFFERENT remedies, the surrounding comments document per-engine
canvases that exist for real quality reasons (BUG-LOCAL-412), and it is a hot
path every engine traverses. That is a rested decision, not a 6am one.

**Three more open, all verified:** a THIRD validation bypass (exported
`run_episode` skips `resolve_final_shot_engines`/`assert_coverage_plans`, and
the soak calls it directly); `run_graph` cannot accept preloaded results, so
7c's loader removal has a required 6-step order; and the 169-frame seam needs
`opening_duration_sec`/`crossfade_ms` in the profile schema -- **`render.frame_budget`
is INERT in episode mode and is NOT the mechanism**, which refuted my own claim.

**Score: the arc refuted THREE of my load-bearing claims and I refuted FOUR of
the panel's, all verified against source.** Mine: live VRAM shortening, the
trim_tail coupling, the frame_budget cap. Theirs: clamp the boomerang (would
reintroduce a freeze the roundtable already caught), re-partition against a
forced engine, `FrameContract` needs `to_dict`, and -- the sharpest -- raise
`ltx_8gb`'s ceiling to 169 so the opening beat stays single-segment, which
inverts the objective: 169 is chosen BECAUSE it splits `[161,9]`.

Also: the registry probe showed **all viz engines -- the canonical defaults --
have no ceiling**, so the default route can never exercise multi-clip at all.

## 2026-07-27 (overnight, remote Cowork) -- HEAD 07a84627 (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: **settled the 7b architecture fork with an r2->r3 kibitz arc, then landed
the two slices the arc proved were safe.** Operator was asleep; ran Variant A of
`docs/2026-07-27-next-window-prompt-nogpu.md` and skipped its "STOP until I
confirm" gate on the operator's standing "code while I sleep, don't stop".

Suite 6913 -> **6925 passed / 27 skipped / 1 xfailed**. Bible 17. Canonical
`5377914B` byte-identical throughout. Four commits, all pushed, HEAD == origin
verified after each: `6bde4b36` problem statement, `499541b6` slice 7b-1,
`07a84627` slice 7b-6 + the judgment, plus this handoff.

**THE DECISION: neither A nor B.** Full reasoning in
`docs/2026-07-27-multiclip-7b-fork-judgment.md`; CURRENT STEP carries the
summary and the order. The short version is that the fork's framing was wrong:
`render_driver.py:2952-2958` already makes the divergence terminal on the
multi-segment path by comparing rendered OUTPUT to the plan, which catches all
fifteen env vars, the profile, the boomerang and the provider clamps in ONE
predicate without enumerating any of them. Option A enumerates inputs and would
be permanently one variable behind; Option B's real value shrinks to moving an
existing refusal earlier than the GPU work. The actual gap is that the
single-segment path -- **the only path production runs** -- has no proof at all.

**LANDED.** `499541b6` 7b-1: `eng_ltx_av` parsed four env vars at module scope
with bare `int()`/`float()`, so a typo raised `ValueError` during import, the
adapter never registered, and `frame_contract_for` answers `SINGLE_ONLY` for an
adapter it cannot reach -- one typo silently deleted an engine and reverted its
lane to unbounded single-clip, with nothing in the log naming the variable.
Fixed all four, not just the one the panel named. `07a84627` 7b-6: the
boomerang tripwire, pinning that `ltx_video` declares 169 and returns 193 by
default, so the deferral to 7c is conscious.

**FOUR BLOCKERS, ALL VERIFIED, ALL IN THE WAY OF THE RESOLVER** -- B1 the
canonical workflow never wired `max_render_frames` (node 87 has no input
descriptor, just an unbound trailing widget value, so Option B's whole channel
is dead in the real workflow); B2 `ShotLock.IS_CHANGED` fingerprints only the
two ROUTING env vars, so a frame-cap change serves a STALE cached plan; B3 both
plan boundaries swallow the exceptions 7b wants terminal; B4 `frame_count` is
`round(duration*fps)` for 13 of 31 engines. Order and line numbers in CURRENT
STEP. **I stopped coding rather than build on any of them** -- every remaining
slice had a precondition that was not met, and the arc's job was to find that.

**THE ARC REFUTED ME TWICE AND WAS RIGHT BOTH TIMES.** I claimed live VRAM
silently shortens renders; codex r2 pointed at `compute_real_frame_budget`,
which S4 rewrote on 2026-07-10 to RAISE instead -- its docstring says so in as
many words. I then built the whole r3 plan around an ASK-vs-plan trim_tail
coupling on the single path; codex r3 pointed at `segment_render_frames`, whose
docstring says it answers from the plan "for EVERY index, segment 0 included".
Both verified, both struck. **Write the anchor first and then let the panel
shoot at it -- including at the anchor.** I also predicted in the anchor, before
the fan-out, that neither seat would find `render_driver.py:2952`; both missed
it, which is why the driver's own read still has to happen.

Rejected from the panel, with reasons: clamping the boomerang to the ceiling
(`test_loop_source_length_no_freeze_shortfall` pins the OPPOSITE for exactly
target=169 and names the freeze bug it exists for -- clamping trades a declared-
ceiling violation for a returning visible-freeze); re-partitioning a plan against
a forced engine at render time (silent re-plan after the stills are minted; agy
itself reversed this by r3); and adding a second force-map check
(`test_the_legacy_path_validates_the_plan_against_the_FINAL_engine` already
covers it end to end).

**PROCESS DEFECT WORTH MORE THAN A FINDING.** The r2 codex seat silently ran
`gpt-5.5` instead of the `gpt-5.6-sol` of record, because kibitz's
`CODEX_MODEL_PREFERENCE` tuple was stale against a catalog that already carried
`gpt-5.6-sol`/`-luna`/`-terra`. Its auto-pick fallback would not have saved it
either -- highest `gpt-5*` by reverse sort selects `-terra`, alphabetically last
rather than strongest. Root-caused in `kibitz/scripts/kibitz.py` and pinned via
`KIBITZ_CODEX_MODEL`; r3 confirms `gpt-5.6-sol`, and the r3 seat found four
blockers the r2 seat did not -- so the downgrade was costing real review depth,
not just a version string. **`kibitz/` is UNTRACKED in this repo: the fix is in
NO commit and dies with a fresh clone. It belongs upstream in the skill.**

Not done, deliberately: 7c (the arc settled that the blockers come first) and
7d (no GPU this window; nothing has still rendered through this machine).

## 2026-07-26 (remote Cowork) -- HEAD 42db9af9 (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: **chunk 7a -- all 31 engines declare a frame contract, and the per-engine
opt-in is deleted.** Two commits, two adversarial QA panels, six real defects
found in code that was already green and already mutation-proven.

Operator ruling that reshaped the plan, verbatim: *"this architecture should
work with all video and still models. There's no gate with opt in or opt out.
If there is, we need to remove that. Everything gets an equal term... I don't
like any hidden opt-ins. It either works or it fails."* Plus: record the
per-model requirements so the new architecture can be checked against them.

**The audit came first.** Before writing anything I probed all 31 registered
engines for what was already recorded. `family`, `render_aspect`,
`required_inputs` and `still_plan` were declared on every one -- the still
requirements the operator asked about already existed, and richly. What did NOT
exist: a frame contract (0 of 31), any continuity declaration, and any clip
duration outside call-site kwargs. Resolution turned out not to be a static
per-engine fact at all -- the local lanes negotiate it per render from the
canvas and the profile -- so the matrix records the mechanism instead of
inventing a number the code never promised.

- `e90dedf1` **the declaration sweep.** All 31 engines carry a static
  `FrameContract`. `supports_multi_clip` deleted from the dataclass, from
  `join_mode_for` and from `validate_coverage_plan`; `supports_multi_clip(engine)`
  replaced by `can_split(engine)`, which is derived arithmetic ("has a ceiling")
  rather than a stored opinion that could disagree with one. `can_chain()` now
  rests on continuity alone -- splitting is universal, the seamless join is the
  one thing still earned per engine. Renamed `discrete_durations` ->
  `discrete_frames` because the field is compared against frame counts while
  every provider publishes its menu in seconds, and `(4, 6, 8)` is a perfectly
  well-formed frame menu no validator can reject. Added `native_fps` so the
  rate those frames are counted at is stated rather than implied. New:
  `tools/engine_matrix.py` + generated `docs/ENGINE_MATRIX.md` with a `--check`
  drift gate wired into the suite, and `tests/test_engine_contract_roster.py`,
  which asks the LIVE registry so an engine registered without a contract fails
  BY NAME instead of silently resolving to `SINGLE_ONLY`.
- `42db9af9` **what the second panel found when multi-clip went live.**

**FIRST PANEL -- four defects, all confirmed against real code before acting:**
1. Declaring ceilings while the opt-in stayed shut made an ordinary 8-second
   beat fatal: 200 frames on `wan_i2v` (max 177) had no legal single render and
   no multi-clip escape, so `partition_beat` refused and took the whole
   episode's plan-build with it. My 7a/7b split was wrong -- the ceilings and
   the opt-in's removal are one change, because separately each is a build that
   does not work.
2. I declared Veo at the PROVIDER's rate. 4/6/8 s x `OUTPUT_FPS` 24 = 96/144/192
   looked right and is unreachable: `canonicalize()` resamples to the canvas fps
   and counts `duration_s * 25`, so an 8-second Veo clip measures 200 frames and
   192 never occurs. Corrected to 100/150/200 at 25, with BOTH wrong answers
   pinned out by test. Omni likewise 75-250, whose old 240 ceiling would have
   refused any clip past 9.6 s inside its own advertised range.
3. `humo_14B_169` inherited a 177 ceiling and its real cap is 49 -- it sets
   `safe_render_frames = 49` while its three siblings are `None`. It now
   declares its own contract, and a general test pins
   `safe_render_frames == max_frames` so the next capped tier cannot repeat it.
4. The cloud lanes declared `quantum=1` while `_duration_seconds` only ever
   emits whole seconds. Now 25 (except `cloud_kling_avatar`, correctly 1 -- its
   length is real audio duration, not a menu).

**SECOND PANEL -- the multi-segment path had never met a real engine.** Chunks
3-6 built it and tested every piece with STUBS, because no adapter could reach
it. The moment real ladders made it live it refused every beat, and the defect
was the same shape three times over: the MINT and the DEMAND asked different
questions about one state. `jump_still_requests` mints nothing for a CHAIN plan;
`_stamp_coverage_plan` mints nothing for a lane the still spine never asks a
scene still of; `jump_segment_still_path` demanded one for EVERY segment >= 1
and raised "NO FALLBACK" when it was missing. Six of seven sampled engines died
at segment 1 -- all four chain-capable local engines and every HuMo beat past
its cap -- AFTER segment 0 had already rendered on the GPU. The demand now asks
the same two questions the mint asked, off the same durable facts.

Also from that panel: an audio-driven lane now refuses at PLAN time with the
reason, because nothing slices audio per segment and a split HuMo beat would
have spoken the opening syllables once per segment -- a sync defect that ships
as a finished episode. Not a new gate: `humo_14B_169` already raised at render
time past its cap; the refusal moved earlier and now names what is missing.

**On the tests themselves.** The panel caught that the cloud `quantum=25` fix
had NO test and the generic sweep could not catch it ((375-100) is divisible by
1 and 25 alike); that one assertion re-executed `can_split`'s own body and
compared it to the call, so it could never fail; that the `safe_render_frames`
sweep had no vacuity tripwire; that `native_fps < 0` shipped untested; and that
two of the three named env-override risks had no check at all. All closed.

My own mutation harness had already caught one vacuous assertion before either
panel ran -- a test that computed its expected value via
`contract.smallest_legal_at_least(target)`, i.e. from the very declaration
under test, so deleting that declaration moved both sides together. It is a
literal now. Thirteen mutations at the end; all thirteen caught, zero toothless.

Suite 6723 -> **6891 passed / 27 skipped / 1 xfailed**. HEAD == origin
`42db9af9`. Other windows' dirty `tmp/*.ps1` preserved untouched throughout.

Next: **7b** (the env-vs-contract refusal), then **7c** (rip the fallbacks --
and the audit added the provider-side clamps to that list, plus the unapplied
`trim_tail` on the single-segment path), then **7d**, the live GPU slice.
Nothing has rendered through this machine yet.

## 2026-07-26 (overnight) -- HEAD a05b5ac6 (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: shipped NINE green pushed chunks -- the two unrun chunk-4 QA lenses, then
coverage chunk 5 and ALL of chunk 6, with a QA panel over every one of them.
**CHUNKS 1-6 ARE COMPLETE.** The whole multi-clip machine exists and nothing
has rendered through it yet; chunk 7 (the `ltx_8gb` opt-in + the live slice) is
where it first does.
- `4d5795b1` **6c-1, the terminal frame** -- what a CHAIN successor begins on.
  Decodes the whole clip with `-update 1` rather than seeking with `-sseof`,
  because a tail seek has nothing to land on in a 9-frame segment, and PROVES
  the file landed (ffmpeg exits 0 for an input it decoded zero frames from, and
  a 0-byte PNG handed to the next segment is a black frame at the cut with a
  clean exit code in front of it). `otr_engine_tmp_path` generalises the
  in-tree allocator so a PNG lands in the same janitor-swept tier as the mp4s.
- `5845e635` **6c/6d, the loop and the assembly, in ONE commit** because a loop
  that renders N segments nobody assembles is the half-landing chunk 4 warned
  about. `render_beat_coverage` opens ONE BeatSession per beat, builds a
  per-segment request, chains the terminal frame INSIDE the loop, and assembles
  transactionally: one shape, the exact DECODED frame count, the silent-clip
  contract, and the output deleted if any check fails. `run_episode` calls it
  for every beat; a no-plan or single-clip beat takes the historical path.
  **Building it caught a real defect in QA6's own fix**: `segment_render_frames`
  short-circuited index 0 to the BEAT's length, so segment 0 of a two-segment
  50-frame beat would have rendered all 50 and then had segment 1 concatenated
  on top. It read as a harmless special case because for a single-clip plan the
  two numbers are equal.
- `a05b5ac6` **QA7 (Sonnet + agy over 6c/6d).** Eight findings accepted, two
  rejected. **The one that mattered: the chain terminal frame was written to a
  top-level `request["init_image"]` that NO production code reads** --
  `build_request` puts it at `asset_refs["init_image"]` and every adapter and
  `_present_request_tokens` read it there. A chained successor would have
  silently rendered from its ORIGINAL still and the beat would have jumped at
  every cut it claimed to chain across. **The test agreed with the bug because
  the test's own request builder used the same wrong key** -- the stub was
  checking my belief, not production's. Also: the concat moved INSIDE the
  transaction (it was outside, so the one failure most likely to leave a
  partial file was the one the cleanup did not cover); a short segment is now
  named at the segment instead of surfacing later as an assembly count
  mismatch; the beat reports its PEAK VRAM, not its last segment's;
  `max(1, keep)` became a refusal; the assembly checks fps and pixel format,
  not just canvas; and the historical-path test now uses a REAL one-segment
  stamped plan, because ShotLock stamps one on every beat and the old test
  only covered the absent-key half of the branch every beat takes. REJECTED
  with reasons: deleting intermediate segment files on failure (the janitor
  owns that tier, and the only artifacts of a failed beat are what you
  diagnose from), and a SAR-mismatch check both seats agreed was speculative.
- `a818b5d1` **QA6 -- Sonnet lens + agy panel over QA4, 6a and 6b.** Six
  findings accepted, four rejected. The two that mattered were both in the
  per-segment seam and both DORMANT-until-6c, which is exactly when they would
  have been most expensive: (1) the seam swapped the init IMAGE and left the
  LENGTH alone, so a request for segment 1 of a 120-frame beat carried segment
  1's picture and the whole beat's duration -- there is now a
  `segment_render_frames` that reads the segment's own `render_frames` off the
  stamped plan, and refuses rather than falling back to the beat's length;
  (2) the override was unconditional, so a mesh lane's subject-isolated FODDER
  would have been clobbered by its segment still -- which is the clay blob the
  guard nine lines above it exists to prevent, arriving through a second door.
  Both mutation-proven. Also from agy: a pathless DUPLICATE receipt entry used
  to `break` and hide the materialized row two entries later (now `continue`);
  a negative or non-numeric `segment_index` now fails closed NAMED instead of
  silently reading as segment 0; and the still-lane guardrail no longer skips
  an unbuildable engine in silence. REJECTED with reasons: agy's claim that a
  jump segment RAISES in an earlier beat-still branch (the spine guarantees the
  beat still exists for any lane that mints jump requests -- that is QA3's
  one-predicate design), its proposed fix of bypassing lines 1771-1948 (those
  branches decide canvas and portrait-vs-wide, not just the still), an
  `IndexError` in `ffprobe_counted_frames` (already guarded), and its
  recommendation to CUT the second `assert_coverage_plans` (it is the
  pre-existing, documented defence-in-depth double call).
- **The two lenses the operator asked for first.** Image-phase capability
  gating and operator-intent, run read-only against `4faabe0e`. Judged: the
  ordering defect was real and is `b0e383f5` **QA4** -- on the LEGACY route
  path `resolve_final_shot_engines` validated the coverage plan BEFORE
  `apply_engine_override` and the radio-host redirect, so a plan stamped for
  the PICKED engine was checked against that engine and then executed by a
  DIFFERENT one. That is chunk 1c's ordering defect reintroduced one contract
  further down, inside the very function whose docstring closes it -- and
  checking early is worse than not checking, because it logs COVERAGE PLANS OK
  for routing that no longer holds. Mutation-proven (the new test fails without
  the reorder). Also landed a guardrail that a `still_*` lane can never declare
  `supports_multi_clip` -- put in place BEFORE the first opt-in, not after.
  REJECTED with reasons: the "unregistered engine skips the spine guard" claim
  (the mint returns early for unregistered ids, so the case never arrives --
  the DOCSTRING was the thing that was wrong, and it is now corrected).
- `4fa992e6` **chunk 5, the beat session.** One prepare/load per BEAT instead
  of one per clip, one teardown in a single outer `finally`, and a named
  IDENTITY (engine + recipe + weights) captured at open and re-proved before
  every segment. A multi-segment session whose adapter cannot name its handles
  is REFUSED at open, before the weights land: handles nobody can name are
  handles nobody can invalidate. Wired as the ONLY lifecycle path, so a
  single-clip beat is a one-segment session and behaviour is unchanged.
  Mutation-proven. The acceptance counts LOADER calls, never `prepare` calls --
  there is a test that builds a lazy-loading adapter showing one perfect
  `prepare` and three loads, which is exactly what a prepare-count acceptance
  would have blessed.
- `451309de` **chunk 5 QA (agy Gemini 3.6 Flash High).** Five findings, four
  accepted. **The important one is LIVE and PRE-EXISTING:**
  `motion_common.teardown` detached patchers and called `unload()` BEFORE
  releasing the GPU lease, and `unload()` is overridden per engine -- so an
  override that raised stranded the shared single-heavy-engine lease and the
  NEXT episode blocked on `acquire` for its full 120s timeout and failed for a
  reason that had nothing to do with it. The release now sits in a `finally`.
  Also: the identity BASELINE is now taken after `prepare` (an adapter that
  resolves "auto" to a real UNET while loading was reporting drift against its
  own pre-load intention), segments must be CONTIGUOUS (0 then 2 silently
  dropped a segment), and a session with no `beat_id` LATCHES the first beat a
  caller claims. Rejected: speculative dict/set identity normalisation. Took
  its CUT recommendation -- the session's own call counters were measuring the
  bracket, which is the obviously-correct part, so they are gone.
  Also collapsed `session`/`segment_index`/`session_owner` into ONE
  `SegmentSlot`, which makes "a session with no segment index" unconstructible
  rather than merely validated.
- `3a76c47a` **chunk 6a**: `ffprobe_clip_fields` learns `width`/`height` (free,
  same stream read) and a NEW `ffprobe_counted_frames` runs `-count_frames`.
  Deliberately two helpers: counting decodes, and the cheap probe runs on every
  emitted clip. An unreadable count raises rather than returning 0.
- `a888c423` **chunk 6b**, the chunk-4 carry-forward: a jump segment resolves
  its init image BY OBJECT ID off the still-spine's own receipt, never through
  `_still_index` -- which filters to `scene_*` kinds keyed BY BEAT and would
  therefore have handed EVERY segment segment-0's still. The differential test
  demonstrates that rather than asserting it.
- Suite 6634 -> ... -> **6723 passed** / 27 skipped / 1 xfailed; Bible 17;
  canonical byte-identical `5377914B` across all nine commits; hygiene clean
  (it also caught a pre-existing non-ASCII character in `wan_shared.py`, fixed
  in passing).
- **STOPPED BEFORE CHUNK 7 DELIBERATELY.** Chunk 7 is a LIVE GPU leg, not a
  code chunk: it needs a selective box reset, and in a remote window a blanket
  python kill severs the very bridge the session is watching through. Chunk 6
  is a clean, complete, fully QA'd stopping point; starting a live render at
  the end of a long unattended session is how you get a half-finished leg
  nobody was watching.
- DOCTRINE, earned twice tonight: **every chunk gets a panel before the next
  one builds on it.** QA6 only happened because the operator asked what had not
  been reviewed -- and it found two defects in the seam chunk 6c is about to
  build against. A chunk that is "obviously right" is exactly the one whose
  panel gets skipped.
Current step: chunk 7 -- the FIRST adapter opt-in and the LIVE 169-frame slice.
Next: CODER A -- chunk 7, in the four steps now written into GO_FORWARD's
CURRENT STEP (static FrameContract; a declared `session_identity`; NO ping-pong
CLIP-FILL on a planned segment; the segment graph taking the prepared handles
as literals -- the adapter-side half of chunk 5 that r4 specified and the
driver-side half already honours). Then the live leg, with a selective box
reset per CLAUDE.md section 4. This whole session ran REMOTE (cloud Cowork), so GO_FORWARD's Window
packing now carries a "REMOTE / cloud Cowork session" block: file tools hit the
container not Windows, the `/mnt/user-data/uploads/` snapshot LAGS and must
never be read, the bridge can drop mid-edit, and the suite needs a detached
launch because of the 60s call ceiling.
Models: Claude Opus (rung 4) + 5 Sonnet QA lenses + 3 agy panels (rung 2, $0).
No Codex, no OpenRouter -- $0 external spend.
Commits: b0e383f5, 4fa992e6, 451309de, 3a76c47a, a888c423, a818b5d1,
4d5795b1, 5845e635, a05b5ac6

## 2026-07-26 00:30 -- HEAD 4faabe0e (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: shipped coverage CHUNK 4 (the jump-still image-phase consumer) and its QA
round. Without it a jump cut had NO still -- the image phase mints exactly one
still per beat, so every segment after the first would have rendered from
nothing.
- `583b3ea3` **chunk 4**, three seams, one commit because a partial landing
  leaves a hole (requests nobody honours). New pure authority
  `coverage_plan.jump_still_requests` / `jump_still_object_id`; ShotLock stamps
  `shot["jump_still_requests"]` durably where beat_id is authoritative; the
  dispatcher's `merge_jump_still_requests` folds them into `objects` +
  `required_scene_targets` BEFORE the existing id/duplicate validation so the
  merged rows meet the producer's own contract; the spine proves every segment
  by object id with NO repair-by-substitution. Ids are minted ONCE and READ
  twice -- never re-derived -- because a shot's beat id passes through
  `_canonical_visual_beat_id` and an image object's does not.
- **QA ROUND 3 (`4faabe0e`)** -- two Sonnet lenses + an operator-run panel.
  FIVE findings judged, FOUR fixed, ONE rejected. The important one: the merge
  inferred "no scene object and no required target means this lane consumes no
  still" and skipped, while the spine demanded every STAMPED request back
  regardless -- two policies over one state, and the inference did not avoid
  the failure, it moved it to the render boundary and made the message a lie.
  Root fix is neither side: `_lane_consumes_a_still` asks
  `render_driver._still_spine_requires_scene`, the SPINE'S OWN predicate, at
  the mint, so the disagreement is unconstructible. Also: the minter now
  validates its plan (a replayed `from_dict` plan with non-dense indices minted
  two requests carrying ONE object id, and a first segment with `index=7` minted
  a phantom segment-0 request); `jump_still_object_id` refuses a falsy beat id
  (all eight collapsed to one shared id); and the `OTR_TEST_MODE` receipt
  bypass -- which skips the WHOLE spine validator -- can no longer wave a shot
  carrying jump requests through, extracted to `_legacy_receipt_bypass_allowed`
  so the decision has a name.
- REJECTED, with reasons: the panel's "`build_request_from_shot` feeds every
  segment segment-0's still" is real but is NOT chunk-4 scope -- there is no
  per-segment render loop yet, so nothing renders segment 1. Recorded in
  GO_FORWARD as a HARD chunk-6 carry-forward instead. Half-rejected: a cloned
  bookend segment does drop off seed 4242, but "destroys reproducibility" is
  wrong (request-hash seeds derive from stable inputs); what it loses is the
  shared canonical LOOK, which is what cutting means -- now a documented
  decision with a pin rather than a side effect.
- Suite 6591 -> 6618 -> **6634 passed** / 27 skipped / 1 xfailed; Bible 17;
  canonical byte-identical `5377914B` across both commits; hygiene clean.
Current step: coverage chunk 5 (beat-session lifecycle -- reusable
MODEL/CLIP/VAE handles, teardown in ONE outer finally, assert LOADER-call
count). Then 6, then the 7 live slice.
Next: CODER A -- chunk 5. Chunk 6 must resolve per-segment `init_image` BY
OBJECT ID off the stamp, never via `_still_index`. Chunk 7 is the `ltx_8gb`
169-frame LIVE slice and needs a selective box reset per CLAUDE.md section 4.
Models: Claude Opus (rung 4) + 2 Sonnet QA agents + 1 operator-run panel. No
Codex, no OpenRouter -- $0 external spend.
Commits: 583b3ea3, 4faabe0e

## 2026-07-25 (evening) -- HEAD 00339e32 (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: closed out chunk 3b, ran TWO adversarial QA rounds over everything shipped
today, and settled the operator's dormant-3D question with codex.
- `00339e32` **chunk 3b**: the `CoveragePlan` now rides the shot row and is
  validated at BOTH wire boundaries -- ShotLock at plan time, and
  `render_driver.assert_coverage_plans` before execution, re-checked against
  the LIVE contract so an adapter whose declaration moved cannot silently
  execute a stale plan. Behaviour-inert and pinned as such.
- **QA ROUND 1 (`6dc39f1f`) -- six-lens Sonnet fan-out, operator-directed.**
  Found SIX real defects in code that was already green and already pushed.
  THREE were partitioner math, all found by brute-force differential testing
  rather than reading: a tail-trim search capped at one quantum (832 coverable
  beats refused), an unmemoized recursion that HUNG rather than refused (18s at
  count=14, still running past 20s at 16), and -- found by my OWN sweep after
  fixing those two, missed by all six agents -- `join_mode_for` claiming SINGLE
  for targets no single render can cover (202 refusals in an 18k sweep). The
  sweep now runs 18,336 differential calls with 0 false refusals and 0
  invariant breaks, and lives in the suite.
  TWO were swallowed fail-closed sites: chunk 1a's terminal contract was being
  absorbed by pre-existing broad `except Exception` blocks, each of which
  individually defeated the entire chunk.
  ONE was an unproven fix: MUTATION TESTING showed that reverting `talking` to
  the picked engine left the WHOLE suite green -- the decapitation fix's twin
  had shipped with zero coverage. Also proved two "exhaustive" sweep tests were
  theatre (112 of 128 targets asserted nothing).
- **QA ROUND 2 (`0bc863f4`) -- local agy panel.** Found TWO MORE swallowed
  fail-closed sites (`derive_creative_directives`,
  `_still_consumer_capabilities`), bringing the day's total to FOUR, plus a
  dormant picked-vs-effective trap in `three_d_locked_slots`. I overruled one
  of its reproducing inputs: the `mesh_stage` repro does not reproduce, because
  `mesh_stage` never declared `requires_mesh_portrait`. Fixed and labelled
  DORMANT rather than claimed live.
- **DORMANT 3D CONSULT (`624b53e0`)** -- operator asked whether to rip the
  unregistered 3D talkers. Answer: YES, and lean-mean **W2 already said so** in
  writing ("delete, NOT keep-dark"), so nothing was re-litigated -- it belongs
  to CODER D behind the operator's own pinned r2->r3->r4, not to this window.
  **The one new fact: a LIVE guard is hiding in the dormant code.**
  `otr_image_director._is_3d_engine:109-119` raises for ANY non-empty
  UNREGISTERED engine (covered at `test_image_platform_c1.py:339-352`), and
  neither VideoDirector nor the route freeze validates registry membership --
  so a straight delete would silently remove a live protection. W2 chunk 1 is
  now a MIGRATION, recorded in GO_FORWARD. codex also corrected MY brief: five
  test files hard-depend on the dormant modules, not three (my inventory
  classifier missed multi-line import continuations).
- The 4060 pass ran and produced NOTHING usable: ten findings, all rejected on
  grounding (claimed non-determinism in a per-call memo over a sorted menu,
  "infinite recursion" in a loop bounded by a decrementing counter, an
  exact-sum violation from fabricated arithmetic). Fluent, plausible,
  code-ungrounded -- exactly the advisory-only failure mode the skill warns of.
- Suite 6454 -> **6591 passed** / 27 skipped / 1 xfailed; Bible 17; canonical
  byte-identical `5377914B` across all nine commits.
Current step: coverage chunk 4 (jump-still image-phase consumer -- without it a
jump cut has NO still). Then 5, 6, 7.
Next: CODER A -- chunk 4. Chunk 7 is the `ltx_8gb` 169-frame LIVE slice and
needs a selective box reset per CLAUDE.md section 4.
Models: Claude Opus (rung 4) + 1 kibitz r4 + 1 codex consult (`gpt-5.6-sol`
high, pins verified) + a 6-agent Sonnet fan-out + 1 agy pass + 1 4060 pass
(no value). $0 OpenRouter.
Commits: 6dc39f1f, 0bc863f4, 624b53e0, 00339e32

## 2026-07-25 (afternoon) -- HEAD bfacec2b (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: r4 CONVERGED and CHUNK 1 SHIPPED IN THREE GREEN PUSHED PARTS. Operator
went to yoga mid-session and authorised full autonomy ("all chunks waves"),
plus a final all-Sonnet fan-out before code.
- **r4 (`48e02241`): both seats yes-with-fixes.** codex's decisive find, which
  I verified myself by walking the canonical link list: **node ids are NOT
  execution order.** There is no `89 -> 90` edge -- MetaBrief (89) and ShotLock
  (90) are INDEPENDENT branches reconverging only at 91. So the r3 plan's
  premise was wrong: a ShotLock freeze can NEVER inform the image phase.
  Node 87 (VideoDirector) is the unique common ancestor and is the only
  correct freeze point. Overruled agy on one point: its fix would have routed
  the LOCK through the dispatcher mirror, which swallows a malformed force map
  and would have regressed `57f4983a`.
- **Six-way Sonnet fan-out (operator-directed) changed the plan four times:**
  (FO-1) VideoDirector has no env reads at all, so "compute the freeze there"
  as specified would either break its cold-import contract or become a FOURTH
  mirror -> extract a shared authority instead; (FO-2) of codex's six
  route-derived values only `aspects` was urgent -- and it is a LIVE
  DEFAULT-ENV BUG, three others were already effective-aware; (FO-3) the
  equality assertion would have broken two shipped HTTP entry points and ~14
  test assertions; (FO-4) chunk 1 must be three commits, not one.
- `933a78ba` **1a**: new `_otr_shared/route_freeze.py` is THE one authority.
  FOUR copies of force-map + radio-redirect collapse onto it; TWO had
  hard-coded `"ltx_audio_in"` instead of `_NEVER_HUMO_REDIRECT_ENGINE` and TWO
  swallowed a malformed force map the render path calls terminal. Inverted the
  "failsafe" contract on purpose -- the old fail-safe WAS the bug.
- `9006b76d` **1b**: the freeze at node 87 (+ ImageDirector forwarding, key by
  key; ShotLock guards env drift and mints groups/preflight/shots from ONE
  value; `IS_CHANGED` on both ends). **THE DECAPITATION BUG IS FIXED** --
  `aspects` was derived from the PICKED portrait HuMo while the render
  redirected to the WIDE `ltx_audio_in`, so the still was minted portrait and
  centre-cropped. `eng_ltx_av.py:345-347` documents that exact outcome.
- `49944fb1` **1c**: render-time equality -- verify, never repair. The legacy
  mutating branch survives for the two hand-built HTTP entry points and legacy
  fixtures, NAMED and logged, which is why zero test inversions were needed.
- Suite 6454 -> **6504 passed** / 27 skipped / 1 xfailed; Bible 17; canonical
  byte-identical `5377914B` across all three (no node/widget/link change).
- One regression caught and fixed first try: the legacy-name audit flagged a
  bare "director" in my comments; named the real node instead.
- `ffc14693` **chunk 2**: the declaration surface. New
  `_otr_video_engines/frame_contract.py` (frozen `FrameContract` +
  the closed continuity vocabulary) + the optional `frame_contract()` hook on
  the `VideoEngine` Protocol. Every adapter is `single_only` until it opts in,
  pinned by a test that walks the LIVE registry and asserts nobody has --
  so chunk 2 changes no behaviour. Contracts that lie are not constructible
  (discrete durations without tail trim; multi-clip without a ceiling). Plus
  `registry.audit_engine_roster()` for the swallowed-import blindspot both r2
  seats found: every adapter import is wrapped in a bare `except: pass`, so a
  broken adapter silently vanishes from every dropdown and a post-registration
  audit cannot see the hole. It runs at the BOTTOM of `__init__.py` (inside
  registry.py it would report every not-yet-imported adapter as missing) and
  LOGS rather than raises -- the hard gate is a test. Current tree: zero drift.
- `bfacec2b` **chunk 3**: the partitioner (`coverage_plan.py`), pure core.
  Exact-sum or terminal refusal -- a `single_only` engine over its cap raises
  instead of ping-ponging, loop-filling or holding a frame. **Found a real
  arithmetic limit and pinned it rather than papering over it:** chaining
  `8n+1` segments always assembles to `8m+1` visible frames, so a beat not
  congruent to 1 mod 8 has NO exact cover on that ladder and needs
  `allow_tail_trim` -- which is why that flag belongs in the adapter's
  declaration, not the assembler. 169 works precisely because 169 mod 8 == 1.
  Solved for segment COUNT rather than greedy-largest-first, because greedy
  strands an illegal remainder (pinned at 313).
- Suite 6454 -> **6769 passed** / 27 skipped / 1 xfailed; Bible 17; canonical
  byte-identical `5377914B` across all six chunks.
- Two regressions, both caught and fixed on the FIRST correction, no third
  swing needed: the legacy-name audit flagged a bare "director" in my comments
  (named the real node instead), and two chunk-3 tests asserted a coverage that
  the `8n+1` ladder cannot produce (the code was right, the tests were wrong --
  rewrote them to pin the true limit in both directions).
Current step: coverage chunk 3b -- stamp the `CoveragePlan` durably in the
ledger and validate it at BOTH wire boundaries. Then 4-7.
Next: CODER A -- 3b, then 4 (jump-still image consumer; without it a jump cut
has NO still), 5 (beat-session lifecycle), 6 (terminal transaction + assembly
+ an ffprobe helper with `-count_frames`), 7 (the `ltx_8gb` 169-frame LIVE
slice -- needs a selective box reset per CLAUDE.md section 4).
Models: Claude Opus (rung 4) + 1 kibitz r4 (codex `gpt-5.6-sol` high + agy
Gemini 3.6 Flash High, both pins verified) + a 6-agent Sonnet fan-out. $0
OpenRouter.
Commits: 48e02241, 933a78ba, 9006b76d, 49944fb1, 31b711d6, ffc14693, bfacec2b

## 2026-07-25 11:30 -- HEAD 3bedb2fe (v2.0-alpha) -- WINDOW CODER A (Opus)
Did: the still-plans block was SUPERSEDED mid-session by a new operator
requirement, two code chunks landed green, and a fresh r1->r2->r3 arc was run
and judged. Started the day expecting to ratify the 31-plan-table cut.
- Operator did NOT ratify; he sent the architecture to the panel instead, then
  fed in five successive clarifications ending at the real requirement:
  **enough REAL rendered clips to cover a beat with MOVING video** (chain
  last->first preferred, jump cut fine, reuse only if loop-closed, `still_*`
  one still, audio lanes cut at PHRASE boundaries). His own split of ownership:
  each model declares its own PROMPTS + frame numbers; the splitter and
  assembler are SHARED. He reversed an earlier "ping-pong is fine" ruling once
  the mechanism was actually on the table.
- **BOTH SEATS INDEPENDENTLY KILLED THE PREMISE the round was built on** (mine
  and his): nothing renders >1 clip per beat today (`render_driver.py:2627`),
  WAN fills beats by PING-PONG (`eng_wan_ti2v.py:521-535`), and Veo's
  `last_frame` is first/last INTERPOLATION inside one clip, not chaining
  (`eng_google_veo_video.py:277-293`). Multi-clip is a NEW capability.
- `57f4983a` **route lock**: `resolve_final_shot_engines` runs force map AND
  radio-host redirect in ONE idempotent pass BEFORE the still-spine check;
  malformed `OTR_FORCE_ENGINE_MAP` now FAILS CLOSED. Inverted the old
  `_bad_spec_failsafe` test on purpose -- the old contract WAS the bug.
- `a1d810f1` **lip-sync no-mirror**: found by chasing the operator's audio
  question -- `extend_frames_to_target` builds a MIRROR cycle, and
  `eng_humo.py:479-481` ran capped HuMo beats through it, so a talking mouth
  played forwards then BACKWARDS against forward audio. `allow_mirror=False` +
  `MirrorExtensionForbidden`; trimming stays legal. Scoped the lane inventory
  first: only HuMo could reach the mirror, so I did NOT spend a 4-round arc on
  a one-call-site fix and said so.
- **THE FIND OF THE DAY:** `otr_silent_composite.py:244-266` already exempts
  `audio_driven_face` from loop-fill for exactly the operator's reason, and
  names the permanent fix: *"The real fix is phrase-chunking... tracked as a
  follow-up."* The coverage block IS that 2026-06-30 follow-up. Also: THREE
  silent coverage mechanisms exist, not one.
- r1/r2/r3 judged (3 docs). Judge calls that beat both seats: the pause map
  RANKS legal cut points and never chooses them (kills agy's quantum
  objection AND codex's DSP dependency, and defers the pause map off the
  critical path); contain multi-clip inside `render_shot` so the manifest/SFX/
  captions/timeline never learn (neutralises codex's SFX-stacking must-fix).
- **r3 found MY OWN `57f4983a` is one node too late** -- canonical order is
  87 VideoDirector / 88 ImageDirector / 89 MetaBrief / 90 ShotLock /
  91 ImageGenDispatcher / 92 VideoRenderBatch, and the lock sits at 92 while
  stills mint at 91. That is why MetaBrief carries an effective-engine MIRROR.
  Chunk 1 hoists the freeze into ShotLock and retires the mirrors.
- Caught a codex PIN DRIFT to `gpt-5.5` on the first launch and killed it
  before it spent the round; every later round pinned + verified.
Current step: r4 convergence on the multi-clip coverage block at HEAD, then
build chunks 1-7 (route freeze into ShotLock first).
Next: CODER A -- run r4, then chunk 1. No code on the block before r4.
Models: Claude Opus (rung 4) + 5 kibitz rounds (codex `gpt-5.6-sol` high + agy
Gemini 3.6 Flash High, pins verified each round, `--driver claude`). 4060 skill
came UP mid-session; not yet used. $0 on OpenRouter.
Commits: 6bb1a9cf, 57f4983a, ec2760a2, a1d810f1, 2d2f7f90, 81f9c2a3, d3308e43,
3bedb2fe (+ this handoff)

## 2026-07-25 (overnight) -- HEAD 5dd74f93 (v2.0-alpha) -- CODER WINDOW A (Opus)
Did: ran the convergence gate, LANDED S1b, then ran the operator-authorised
NEW R1 and judged it. 4060 was DOWN (/v1/models timed out twice) so rung 1 was
unavailable all session; said so and proceeded rather than blocking.
- `562f9c85` r4 input doc: the corrected plan + TWO findings I added by
  grounding S1b against the real producer instead of the inventory doc.
  (1) GEOMETRY vs LOOK -- the inventory records COMPOSED strings, and chunk A1
  splits geometry (Python, engine-safety) from LOOK (pack-owned). Transplanting
  verbatim would have hard-coded the sci_fi_radio look into all 31 engines.
  (2) `portrait` has THREE runtime geometries but all 27 portrait rows declared
  `aspect="inherit_engine"` with ONE static string -- a naive per-kind paste
  would have shipped PORTRAIT_GEOMETRY to ~20 WIDE engines and re-introduced
  the 2026-06-17 decapitation defect.
- `8403ab58` r4 judgment. agy CONVERGED (3 must-fix, all already listed);
  codex `gpt-5.6-sol` high did NOT (10, several new). PANEL SPLIT on the
  ltx_audio_in bookend row -- codex won on evidence: production emits
  `kind="portrait"` / `source="ltx_radio_face"` at
  otr_meta_brief_image_prompt.py:1782-1790 via build_radio_host_prompt(meta,
  "wide", "ltx_radio_mouth"), so agy's "3-way runtime switch" objection was a
  misread (radio_host_style is a LITERAL at that site). Discarded out loud.
- `69328cec` **S1b LANDED**: 57 rows / 12 adapters now carry the producer's
  real GEOMETRY constants. Corrected the misdeclared bookend row to
  portrait/portrait/wide. SPLIT `_HUMO_STILL_PLAN` (one plan object had served
  four engines across TWO shipped aspects). New fence
  tests/test_still_plan_layer2_parity.py: 4 DRIFT invariants, never prose.
  Suite 6444 / Bible 17 / canonical byte-identical 5377914B.
- `5dd74f93` r4b re-run. BOTH seats INDEPENDENTLY corrected ME, both adopted:
  "same push burst" was too weak (it authorised the local-only commit
  CLAUDE.md sec-7 forbids) -> S0b-core + S0c are ONE ATOMIC COMMIT; and the
  style_tail question must be locked before build. codex also caught that my
  exact-equality fence CANNOT survive S5 -- it is now documented as a
  TRANSITIONAL gate to be REPLACED, never deleted.
- `ae01d38e` + judgment: the operator said mid-session "run a new R1 so we get
  a good lean clean architecture" then went to bed. **BOTH R1 SEATS
  INDEPENDENTLY SAID CUT THE 31-PLAN TABLE.** Judge call: codex's Option C
  (frozen routing + a compact per-adapter descriptor + one materializer + a
  separate prompt hook) over agy's Option B (one central function), because a
  central `engine_requires_still()` recreates the central-authority shape this
  build exists to kill, and the operator's directive requires per-adapter
  ownership. `style_tail_policy` leaves the structural contract entirely.
  Discarded agy's claim that the geometry constants live in render_driver.py
  and that there are six -- there are EIGHT, in otr_meta_brief_image_prompt.py
  and _otr_story_brief_helpers.py.
- NEW from the R1, grounded: **freezing ltx_resolved is NOT
  behaviour-preserving** -- eng_ltx_av.py:402-405 documents per-beat operator
  recipe switching, which the freeze would silently make episode-scoped. I had
  read that docstring earlier and missed the implication. OPERATOR DECISION
  FLAGGED with a stated default. Also: malformed routing config currently FALLS
  BACK against the fail-closed law (dispatcher :377-394, render_driver
  :2784-2799 logs and IGNORES); `+ Add Custom Model` has no still contract.
Deliberately did NOT tear anything down: the operator was asleep, and a
teardown of landed green code across 12 adapters + a schema module + 2 test
files is hard to unwind and rests on a decision that also needs his ruling.
Doctrine lesson: the routing freeze was ALWAYS the bug fix and should have gone
FIRST -- S0a/S1/S1b landed against a structure the arc then cut. S1b still
earned its keep (it improved every prompt at HEAD, and its measurement is what
the R1 rests on), but the ordering was wrong.
Current step: operator ratifies the cut + rules on the LTX per-beat recipe
question; then ONE consolidated Option-C spec; then the routing freeze ALONE
with its live proof.
Next: CODER A. No code until the cut is ratified.
Models: Claude Opus (rung 4) + three kibitz rounds (r4, r4b, R1), all codex
`gpt-5.6-sol` high + agy Gemini 3.6 Flash (High), pins verified every round,
`--driver claude` so no Claude pool was spent on the panel. 4060 DOWN. $0 spent
on OpenRouter -- the R1 ran on the local panel per CLAUDE.md sec-8.
Commits: 562f9c85, 8403ab58, 69328cec, ae01d38e, 5dd74f93 (+ this handoff)

## 2026-07-25 -- HEAD 79fe4d3f (v2.0-alpha) -- CODER WINDOW A (Opus)
Did: resumed after the prior window was killed mid-stream. Kickoff baseline was
STALE (said 90e52f13 / "r1 launched"; real HEAD 79fe4d3f with the arc converged
and S0a/S0a-b/S1 landed). No production code touched; canonical 5377914B.
Ran kibitz r3 on the S0b-vs-S2 ordering question -- codex `gpt-5.6-sol` high +
agy Gemini 3.6 Flash (High), BOTH pins verified per round, `--driver claude` so
the third `claude -p` seat stops spending the Claude weekly pool.
BOTH panelists and my own grounding REJECT Path B. Order is S0b atomically first.
FOUND BY ME, missed by both panelists -- the biggest item: S1's
`framing_geometry` strings are PARAPHRASES, not transplants, and spec section 5
makes that field the layer-2 prompt TEXT. `mesh_fodder` lost the whole clay-blob
clause; `scene_background_plate` lost "no people, no subject, no characters";
`portrait` lost "never crop the top of the head" and is the EMPTY STRING on 19
engines. Wiring S1 as-is silently degrades every prompt. NEW CHUNK S1b restores
the clauses verbatim from the seed inventory; it must precede any wiring.
FOUND (registry audit): 31 engines -> 14 shared plan objects -> only SIX distinct
signatures AND six distinct structures, i.e. the prose adds ZERO per-engine
differentiation; 19 engines share one signature. The operator directive "each
video path owns its own customized still operations" is NOT met. NEW CHUNK S5,
after the wiring, changes prompts and needs its own acceptance. Operator
confirmed the acceptance line: every engine EXCEPT the four `viz_*` needs real
prompt text, including the four `still_*` and the `mesh_stage` 3D option.
HuMo CORRECTED three ways (operator + codex + agy, independently): there are
FOUR HuMo engines; only `humo`/`humo_1.7B` are portrait, both `_169` are ALREADY
wide, and the ComfyUI dropdown shows that split to the operator. Nothing about
HuMo flips. The S2 delta is FOUR ROLE-CELLS -- two portrait HuMo picks x
announcer/music -- under hosts-off default, because `_enforce_radio_is_host`
redirects to the WIDE `ltx_audio_in` that actually renders the beat.
`OTR_ENABLE_HUMO_HOSTS=1` preserves portrait. The "via the `_169` siblings"
framing in S2_EYEBALL_REQUEST + GO_FORWARD was wrong on mechanism; corrected.
Panel MUST-FIX, grounded CONFIRMED by me against the files: (a) the closed
`engine_facts` descriptor `{engine_id, family, provider_side}` (spec:230) has no
aspect field, and `resolve_row_aspect` SILENTLY RETURNS PORTRAIT when it is
absent -- every `inherit_engine` row would go portrait. agy MISREAD this as
"key-name insensitivity confirmed" (true but irrelevant -- the field is absent);
codex is right. (b) the frozen-routing prepass as specified fixes only the force
map: `apply_engine_override` (`:2784`) never applies the radio-host redirect
(`:1413-1513`), so the reproduced defect survives the chunk named for it.
(c) `eng_ltx_video._use_i2v` degrades to text-to-video while
`render_driver.py:1801-1817` RAISES on the same state.
JUDGE CALL on a panel split: adopt agy's S0b-core/S0c scope relief BUT keep
`ltx_resolved` frozen inside S0b-core -- that answers codex's objection that
deferring it desynchronizes `when_engine_talking`. Only the
`eng_ltx_av.assert_usable` mismatch ASSERTION defers to S0c.
Current step: S1b -> S0b-core (corrected) -> S2 -> S3/S0c -> S5 -> S4.
Next: CODER A -- r4 convergence at HEAD on the corrected plan, then build.
Models: Claude Opus (rung 4) + one kibitz r3 (codex gpt-5.6-sol high + agy).
Commits: docs handoff only; no code.

## 2026-07-25 (earlier) -- HEAD 79fe4d3f (v2.0-alpha) -- CODER A (autonomous, killed)
Reconstructed from git + tracked docs: this window was killed at the S2 gate
before it wrote its own entry, and its history had been inlined into
GO_FORWARD_PLAN.md (trimmed back out to here per the forward-only rule).
- `33c4d8cf` S0a -- characterization fixture, 31 engines x 8 configurations.
- `e60185a0` S0a-b -- isolation property amendment (per-engine byte-identity;
  mixed-policy per-role parity). Suite 6434 / 27 skipped / 1 xfailed.
- `c8db4c92` S0b -- NOT LANDED, filed BLOCKED as `docs/S0b_KIBITZ_NEEDED.md`
  rather than half-land a cross-module atomic refactor. Correct call; the
  2026-07-25 r3 panel then found three real defects in that chunk's own spec.
- `a98b1d5d` S1 -- `nodes/_otr_shared/still_plan_helpers.py`, 31 per-engine
  `still_plan` attributes across 16 adapters, `tests/test_still_plan_audit.py`
  (6 tests). Suite 6440 / Bible 17. Nothing reads the plan yet.
- `79fe4d3f` -- `docs/S2_EYEBALL_REQUEST.md`, the halt gate.
Canonical `5377914B` throughout; no node/widget/link touched.

## 2026-07-25 19:15 -- HEAD 84328aa1 (v2.0-alpha) -- CODER WINDOW A (Opus)
Did: ran the still-plan kibitz arc r1->r5 to CONVERGENCE and landed three
tracked docs. No production code touched; canonical byte-identical at
`5377914B`. Panel every round: codex `gpt-5.6-sol` high + agy Gemini 3.6 Flash
(High), model pinned and VERIFIED per round; Claude grounded panelist + judge.
THE ARC REFRAMED THE BLOCK. It was scoped as "five role-indexed places
disagree about what images a model needs". Grounding says the root cause is
FIVE modules independently re-deriving WHICH ENGINE IS EFFECTIVE, from live
env, at five different moments -- `otr_video_director` (picked only),
`otr_image_gen_dispatcher`, `otr_meta_brief_image_prompt`,
`otr_shot_lock:919-933`, `render_driver` -- and `validate_and_repair_still_
spine` (`otr_video_render_batch.py:322`) running BEFORE `apply_engine_override`
(`render_driver.py:2751`). With a force map set, the spine is validated against
the PICKED engine and rendered with the FORCED one. It survived because the
validator is skipped entirely under OTR_TEST_MODE with no target receipt.
So routing is frozen FIRST (new S0a/S0b) and the plan table wires to it.
THE TABLE IS SMALL, measured not argued: driving the real producer over all 31
registered engines yields THREE shapes -- scene spine x26, the `mesh_stage`
fork, `viz_*` zero -- plus one aspect knob. The operator's "this was
over-engineered" call is correct on the evidence.
Landed (docs only): `docs/2026-07-25-still-plans-locked-build-spec.md` (new,
self-contained, `84328aa1`); `docs/STILL_PLAN_SEED_INVENTORY.md` gained the
four-fall-through mechanism map + five traps (`3713ceb5`) and the 31-engine
parity matrix + a CORRECTION (`aa2d4a15`).
THE PANEL CAUGHT ME THREE TIMES, twice in an already-pushed doc: I called
`EngineNotRunnableError` invented (it is real, `engine_registry_base.py:228`);
I wrote that `ltx_video` needs no scene still (`render_driver.py:1801-1817`
requires it whenever `OTR_ENABLE_LTX_I2V` is set, and it DEFAULTS ON); and I
gave `ltx_audio_in` a two-row plan when `:1709-1721` also demands a cast
portrait on character beats under the IA2V register. All three were me
generalizing from seams I had read to seams I had not -- the init-selection
branch (`:1528-1853`), which is now first-class in the site inventory. Fixed at
the root and pushed.
Found by me, not the panel: `_still_spine_requires_scene` has FOUR
fall-throughs and `still_motion` is NOT in the hardcoded id list (it rides the
family branch); `mesh_stage` DOES require a scene-slot row, satisfied by the
background plate via explicit plate-over-scene precedence at `:586-597`; the
producer is engine-BLIND by design (enumerate-then-filter), so the plan applies
at the FILTER, never the enumerator; `apply_engine_override` is idempotent per
shot, so the prepass is a hoist, not a rewrite.
Credit note: the local kibitz panel was running THREE seats -- the third is a
`claude -p` CLI seat spending the Claude weekly pool (~11.5 min with no output
on r2 before I killed it). The ladder budgets kibitz as rung 2-3. Since the
judge IS the Claude seat, r3-r5 ran `--driver claude` (codex + agy only).
Recommend making that permanent in the kibitz invocation.
Current step: S0a -- the characterization fixture at HEAD, per section 11 of
the locked spec. No further panel round is owed.
Next: CODER A executes S0a -> S0b -> S1 -> S2 (operator eyeball: HuMo
announcer/music stills go 832x1216 -> 832x480) -> S3 -> S4 two live legs.
Models: Claude Opus (rung 4) + 5 kibitz rounds (codex `gpt-5.6-sol` high, agy
Gemini 3.6 Flash High). No roundtable, no Fable.
Commits: 3713ceb5, aa2d4a15, 84328aa1, plus this handoff.

## 2026-07-25 01:30 -- HEAD 9d1874f1 (v2.0-alpha) -- CODER WINDOW A (Opus)
Did: landed the WAN 8-GB low-VRAM launch contract @ `f914f0a4`, then opened the
NEW operator block (per-engine image contract) and landed its C0 @ `9d1874f1`.
WAN 8-GB: the tier's 17-frame ceiling existed only in `launch.env`, which a
production leg can never see (it is submitted to an already-booted server), and
`render.frame_budget` maps to a harness-only widget ignored in mode=episode --
so the contract was inert on BOTH channels and the leg inherited the 177-frame
engine max. New OPTIONAL profile key `video.max_render_frames` now rides the
same channel device/dtype policy uses: profile -> `OTR_VideoDirector.
max_render_frames` (appended widget, canonical ships 0) -> v2 policy -> ShotLock
ledger -> `build_episode_render_policy` -> `MotionEngineBase.prepare` ->
`eng_wan_ti2v._floor_length`. Deliberately did NOT reuse `render.frame_budget`:
every 16GB tier declares 25 there, so wiring it would have capped the QUALIFIED
16GB WAN lane to 1s renders. Canonical A66A416B -> 5377914B, 11 variants + 4
paired .env.json hashes regenerated, two node-87 widget-count pins 14 -> 15.
Record PBUG-20260723-02; live 8GB requalification still owed.
IMAGE CONTRACT (new block, operator 2026-07-25: "each video engine needs a
separate set of instructions and prompts about what kind of images it needs;
the image gen dropdown stays separate"). Ran a kibitz r2 + r3 (codex
gpt-5.6-sol high; agy delivered r2 but FAILED in r3 -- one-agent round, recorded).
Had to kill and relaunch the arc once: codex auto-resolved to gpt-5.5, the exact
stale-cache drift CLAUDE.md section 8 warns about -- pin KIBITZ_CODEX_MODEL.
C0 (test-only, `9d1874f1`) DISPROVED the standing theory: the producer already
requires the opening beat for every still-consuming engine and the mesh
fodder/plate pair for every mesh beat, and viz_* requires zero images. So
enumeration is EXCLUDED as the cause of the three 2026-07-23 still-spine rows;
remaining suspects are recorded in the failure inventory (older code path /
env-routing divergence / materialization / shot-id scheme).
Three things grounded on the way that must not be lost: `still_*` engines
consume a scene still while declaring only text_prompt in required_inputs (so
requiredness must be DECLARED, never derived); the lips capability is a HOOK the
director CALLS (a getattr truthiness test would invert lips/no-lips for every
engine); `apply_fresh_cap` has no production caller; and `ImageRequest` is
_Forbid-strict without kind/beat_id/char_id while the dispatcher sends exactly
those -- that boundary is unvalidated today.
Current step: image-contract block -- r4 convergence at HEAD is OWED before any
contract chunk executes, then C2a (snapshot OTR_FORCE_ENGINE_MAP /
OTR_ENABLE_HUMO_HOSTS / OTR_ENABLE_LTX_I2V + canvas.fps once into the policy and
ledger; today the image and render phases can resolve DIFFERENT engines for one
episode), then C1..C5 per the r3 judgment.
Next: CODER A (or its successor) runs r4, then C2a. Plan of record:
`kibitz-runs/2026-07-24-engine-image-contract/{r2,r3}/final.md` (gitignored).
Models: Claude Opus (rung 4) + kibitz r2/r3 (codex gpt-5.6-sol high, agy Gemini
3.6 Flash High -- agy absent in r3).
Commits: f914f0a4, 9d1874f1, plus this handoff.

## 2026-07-24 16:45 -- HEAD 36da1f9f (v2.0-alpha) -- OPERATOR RE-GROUND GATE (Opus)
Did: added a STANDING RE-GROUND GATE to GO_FORWARD. No code touched.
THE OPERATOR'S CALL: every remaining big block gets a kibitz arc before it
executes, because "the code has changed" and "it's been a while since many of
these plans were done" -- and then, unprompted, the sharper half: "if in doubt
restart with r2", and on a follow-up, "lean mean deserves an r2-r4 as well".
THE RULE AS LANDED. Default entry is r3 (wiring), since these docs already have
r1 + r2 on record -- the cheap re-ground is the wiring round against CURRENT
code plus r4 convergence. DROP TO r2 when r3 shows the CODING PLAN is wrong
rather than just its line numbers: a seam that no longer exists, an authority
that moved, a precondition another build already satisfied or destroyed.
Patching an r2 from inside an r3 produces a plan nobody reviewed. If in doubt,
start at r2 -- a wasted r2 costs one panel round, executing a stale coding plan
costs a day of rips against the wrong file list, and rips are the hard kind to
unwind. No block executes without an r4 convergence at current HEAD; runs go
under `kibitz-runs/<date>-<block>-r<N>/` and get cited in the block entry.
PER-BLOCK, and the reasoning is worth keeping because it is not uniform:
BOTH LEAN-MEAN BLOCKS ARE PINNED TO A FULL r2->r3->r4 by operator decision --
not the r3 default, and a later window may not re-argue them down to save a
round. The justification I would have reached anyway: lean-mean is a DELETION
campaign whose entire value IS its file-and-line kill inventory, the most
perishable thing a plan can carry; its own header already declares five stale
areas; and the question is no longer "do these line numbers still point at the
right code" but "is this still the right code to delete", which is an r2
question by definition.
Randomizer: r3+r4, and note the doc's own filename admits it --
`...-randomizer-rolls-r2-coding-plan.md` never got an r3 or r4, so this is the
arc COMPLETING, not repeating.
`dynamic_story`: r3+r4, and the standing "rev-5 FINAL, do not rerun panels"
rule is NOT in conflict with it. That rule protects the DESIGN (the r1 arc,
settled over five revisions); r3 asks whether the design still WIRES to code
that exists today, and the roster, routing authority and writer tail have all
moved. Re-litigating the design is forbidden; re-grounding the wiring is
mandatory. Worth stating explicitly in the doc because the next window would
otherwise read "do not rerun panels" as "skip the arc".
LEAN-MEAN TAIL: full arc, but run WHEN THE TAIL OPENS, not now -- every block
ahead of it edits the very writer this block splits, so an arc run today
grounds against a writer that will not exist at execution time. Running it
early is WORSE than not running it: it produces a confident stale plan.
SFX: no new arc scheduled -- the already-required R4.1 refit IS its re-ground.
Credit shape, since this is rung 2-3 spend: ~10 panel rounds total across the
remaining blocks. Front-load early in a credit week and run each block's arc
when that block opens rather than batching them all now (batching would
recreate the exact staleness the gate exists to prevent).
Current step: unchanged -- WAN 8-GB low-VRAM launch contract, CODER A, ungated.
That one needs NO re-ground: it is a live 2026-07-23 defect, not an old plan.
Next: CODER A takes the WAN contract. CODER D's first job is now the lean-mean
r2, not a rip.
Models: Claude Opus (rung 4) only; a plan edit, not a build.
Commits: this one.


## 2026-07-24 16:20 -- HEAD d036931b (v2.0-alpha) -- OPERATOR RESCOPE (Opus)
Did: recorded an operator scope decision in GO_FORWARD. No code touched.
CUT, on the operator's call ("i need to get coding done", "we will triage more
bugs later"): the 45-word scene matrix, the 54-case visual-style sweep, and
the ENTIRE quick-wins block. CODER B and CODER C dissolved with it -- both
windows existed only to hold quick-wins. New order, operator-dictated: WAN
8-GB contract -> LEAN-MEAN FRONT -> Randomizer A -> dynamic_story -> LEAN-MEAN
TAIL -> SFX -> re-observe the parked story bugs.
NOT cut, and said so explicitly so a later window does not assume otherwise:
the six-bank 120w requalification and image-phase still ownership. Ripping a
schedule does not rip the defects under it.
ONE item survived the quick-wins cut as a LEAN-MEAN W6 SUB-STEP, not a
standalone chunk: `docs/ENGINE_MATRIX.md`. Worth recording WHY, because I got
this wrong first and had to correct it in front of the operator: GO_FORWARD
called it a "PRECONDITION for Lean-Mean W6", and I repeated that as a hard
blocker. The source doc (`docs/2026-07-10-lean-mean-rip-final.md:301-304`)
says only that W6's README policy line "should link it" and that it lands
before the campaign -- an ORDERING PREFERENCE the operator set on 2026-07-10,
not a technical dependency. W6 executes without it. The class: GO_FORWARD's
one-line summary of a source doc can be STRONGER than the doc; when a
"precondition" is about to cost the operator a decision, read the source.
THE OPERATOR'S OWN CALL, and it is a good one worth keeping as doctrine:
"we have done so much story engine change, i'm not sure the old story bugs are
bugs to be honest." Correct, and it splits on a clean line -- MECHANICAL
defects survive story-engine churn (WAN frame counts, a 2800-vs-512 cap, a
missing receipt), STORY-QUALITY judgments do not. The two eyeball-era rows
(announcer framing 2026-07-11, name-splice #2) were observed against an engine
that has since had its LLM vetoes ripped, THE LAW imposed, six banks renamed
onto new packs, word-fit ceilings retired, the repair-first plan landed and a
ledger cleanup pass added. Neither has a reproduction at HEAD, and the
standing rule already says a finding without one is not a row. Both are now
PARKED with their doc links intact -- not deleted, because deleting loses the
observation -- and are settled by the operator eyeballing a real render leg
AFTER SFX: still there -> re-admit as a FRESH dated row with that leg as
evidence; gone -> the LAW-era work already fixed it, tombstone it. No coder
time is scheduled against either meanwhile.
Also fixed in passing: the whole-tree receipt line still read 6398 (wave 6's
number) after wave 7 landed 6403.
Current step: WAN 8-GB low-VRAM launch contract, CODER A, ungated, no GPU
needed to write it.
Next: CODER A takes the WAN contract; CODER D takes the lean-mean front after
it. RENDER opens only when the operator wants the six-bank 120w wrap.
Models: Claude Opus (rung 4) only; a plan edit, not a build.
Commits: this one.


## 2026-07-24 15:42 -- HEAD 30358ad1 (v2.0-alpha) -- WINDOW CODER E (Opus)
Did: independent source banks WAVE 7 -- ASSESSED, then closed. One green pushed
chunk @ `30358ad1`, and the block is DONE for v1 (all seven waves).
THE ASSESSMENT, which was the actual work: the plan's w7 line promised a "Story
Pack widget" with packs resolving by OWNER via a four-field `PackRef` /
`resolve_pack_ref`. Neither name exists in the tree and neither is needed.
Packs already resolve by owner -- waves 1-3 gave `_Registry` a `pack_dirs` map
of bank id -> the directory that owns its packs, so a client pack loads from
the client's own bundle. And the widget already exists: the `source_bank` COMBO
on node 1 reads `list(list_bank_ids())` LIVE at `INPUT_TYPES()`, and
`_admit_user_banks` folds activated client rows into exactly that registry. The
pack needs no second widget because `resolve_story_pack(bank_id)` takes the
model from the row's own `default_story_model` -- the plan's own alternative,
"or a bank's manifest default covers it". So: no node, no widget, no link, no
canonical change, and the canonical hash is STILL `A66A416B` after seven waves.
Inventing a pack widget would have added a second way to say what the bank row
already says. Closed as satisfied instead.
WHAT THE ASSESSMENT ACTUALLY FOUND, and the reason this was a chunk and not a
one-line report: `guide_ref` had NO runtime consumer anywhere -- parsed by
`_parse_bank`, stored on `SourceBank`, read by nothing. So the one row shipped
expressly to advertise this feature, `+ Add Your Own` (`custom_source_bank`),
answered a click with a generic "pick a runnable bank", while the only text
that could have helped sat unread in banks.json still saying "the simple_4 pass
runner does not exist yet" -- false since wave 4. `require_runnable_bank` now
appends the row's own `guide_ref` (JSON owns the words, Python owns the
raising, this module's standing split), and any client bank shipping
runnable=false inherits the courtesy. Same error also said "runnable=false in
banks.json"; a client's row lives in its own bank.json, and naming the wrong
file to the one person who must go edit it is the defect class 8c45172d closed
-- it now says "its bank row". banks.json, the `source_bank` tooltip and
EXTENDING_OTR.md (new section 6: the dropdown is live, restart is the refresh,
your default_story_model IS the pack selector, the signpost row is not your
bank, a quarantined bank is simply absent) now all name the same path.
THE GAP THE TESTS CLOSED: `test_client_bank_joins_the_dropdown` (wave 2) is
named for the dropdown but asserts `list_bank_ids()` -- the registry, one hop
short of the widget the operator actually sees. Three new pins in
`test_source_bank_widget_2c.py` (the file that owns that surface) take the last
hop: an activated bundle appears in `INPUT_TYPES()["optional"]["source_bank"]`,
its widget value resolves to a pack inside its own bundle, and admitting a bank
leaves the canonical 34-slot positional widget vector untouched
(BUG-LOCAL-097). Two more pin the signpost text and the corrected wording.
Worth carrying forward: NEVER `Select-String` the canonical JSON -- it is one
line, so a "grep" dumps the entire 200 KB graph into context. Read it with
`json.loads` in a temp script instead.
Gates: suite 6403 passed / 27 skipped / 1 xfailed (was 6398; +5); Bible
17/24/3; AST/JSON/BOM/zero-byte/UTF-8 clean on all five touched files;
canonical byte-identical A66A416B. Pathspec commits -- the other window's three
modified tmp/*.ps1 and all untracked scratch preserved; temp probe scripts
deleted before commit. `git commit -F` per the standing note.
Current step: six-bank requalification + the bug-first items (CODER A) and the
render track's 45-word scene matrix. The CODER E slot is RETIRED, not idle --
the deferred power-user tiers (client own-runner + staging, dependency
manifest, standalone story_rules) are a NEW block if the operator wants them.
Next: CODER A takes bug-first items 1-3, or CODER F opens Randomizer A, which
this session unblocked. Flag for the planner: the `check_compatibility` fork
still has a standing 2-of-2 rip recommendation and is still unratified, and NO
CLIENT BANK HAS EVER RUN LIVE -- every wave is proven by suite and contract
tests only, so the first real client bundle is a qualification, not a
formality.
Models: Claude Opus (rung 4) only. No strikes used -- the focused suite, the
full suite and the Bible were green on the first run -- so no kibitz was owed.
Commits: 30358ad1 (wave 7) + this handoff.


## 2026-07-24 14:15 -- HEAD 3d97a130 (v2.0-alpha) -- WINDOW CODER E (Opus)
Did: independent source banks WAVE 6, two green pushed chunks.
`1504bb4c` = the client-interpreter fallback gap. `build_source_interpreter_
fallback` switched on the four SHIPPED interpreter ids, so a client bundle --
which routes its lane through the reserved `"self"` entry point -- exhausted
its own structured-output ladder and then died on `UnknownInterpreterError`
naming an interpreter id of 'self': OUR router complaining about THEIR failure.
`"self"` now has its own branch, building the brief from the bank's own label
(or its id when unlabeled) plus the validated payload, asserting nothing about
genre or form. Routing unlocks `"self"` only on an is_client row and never
teaches it to the shipped registry, so reaching the branch PROVES the bank is
client-owned -- no extra ownership lookup needed.
`3d97a130` = the wave proper: `nodes/_otr_ledger_cleanup.py`, wired at the one
shared producer boundary in `_run_writer_tail` (after every writer-side text
mutation, before the TTS delivery stamp and the freeze cascade). Deterministic
completion -> safety repair IN PLACE -> one bounded LLM `meta.episode_title`
fill with a source-derived backstop -> `LedgerIncompleteError` naming every
remaining hole at once. The hole it closes: `content_owned_readonly` SKIPS the
cascade's inline safety cleanup because it assumes the producer already
cleaned, and for a client bank the shared writer IS the producer -- so nothing
cleaned, and the first unsafe word went straight to G9 and killed the episode.
Residual hits are now REPORTED, never escalated; G9 stays the last-resort
backstop, because a cleanup pass that raised on content would be a SECOND
terminal content policy, which is precisely what THE LAW forbids.
TWO FINDINGS, both bought with a failing suite, both worth carrying forward:
(1) THE ANNOUNCER IS THE COUNTER-EXAMPLE. I required every voiced `char_id` to
name a cast row. It does not: the announcer speaks on nearly every episode with
char_id="announcer", lives in the Kokoro voice namespace rather than the cast's
Bark one, and legitimately has NO cast[] entry -- which is exactly why the
freeze gate's own per-line invariant requires a non-empty char_id and stops
there. The class: a completion pass must never be STRICTER than the authority
it completes for; being stricter invents a structural failure that authority
does not recognize. An unlabeled caption is a quality cost, not a hole.
(2) THE SEED WAS NOT MINE TO OWN. Stamping `meta.episode_seed` wherever both
receipts were absent read as completion and was really a second owner -- a
legacy lane's cast picker stamps `cast_contract.cast_seed` upstream, a
content-owned lane's seed is stamped by the tail right after the call. It also
broke `test_tail_byte_identity_same_inputs`, and the reason generalizes: a
freshly minted seed is BY CONSTRUCTION not derivable from the inputs, so any
pass that mints one cannot be byte-identical across two runs of the same
inputs. The writer's original content-owned stamp is restored verbatim.
Also caught by an existing guard and worth the reminder: `row["text"] = ...`
anywhere under `nodes/` is forbidden (`test_text_metric_ownership`) -- text and
its counts have ONE atomic owner, `set_line_text_metrics`.
Gates: suite 6398 passed / 27 skipped / 1 xfailed (was 6365; +33); Bible
17/24/3; AST/BOM/zero-byte/UTF-8 clean; canonical byte-identical A66A416B (no
node, widget or link touched). Pathspec commits -- the other window's three
modified tmp/*.ps1 and all untracked scratch preserved. `git commit -F` used
throughout per the last window's note.
Current step: CODER E wave 7 -- story_pack widget / canonical JSON. ASSESS
FIRST: waves 1-6 changed no node, widget or link and the canonical hash never
moved, so w7 may already be satisfied; if it is, close the extensibility block
as DONE for v1 rather than inventing a surface change.
Next: fresh CODER E window assesses w7. CODER A (bug-first) and RENDER remain
open in parallel. Operator/planner still owns the `check_compatibility` fork.
Watch on the next live legs: the cleanup pass now runs on EVERY bank (no-op and
zero LLM cost on a complete ledger), so a content-owned leg that used to die at
G9 may now ship a sanitized line, and a blank episode_title is filled at the
tail instead of exploding later in otr_credits_roll. Neither has a live receipt.
Models: Claude Opus (rung 4) only. Two strikes used and spent on the two
findings above, both fixed at root on the second swing; no third attempt, so no
kibitz was owed.
Commits: 1504bb4c (client fallback) + 3d97a130 (ledger cleanup) + this handoff.


## 2026-07-24 12:22 -- HEAD 8c45172d (v2.0-alpha) -- WINDOW CODER E (Opus)
(Clock note: the entry below it reads 14:05 but its commit `eba8da25` is
stamped 11:45 local. This entry's time is the real local time; the log is
append-only, so that one stands as written.)
Did: independent source banks WAVE 5, two green pushed chunks.
`c97a0e91` = `nodes/_otr_feed_fetch.py`, the ONE bounded seam OTR uses to reach
the network for source text: https-only with no silent upgrade, connect 5s /
read 10s, 3 redirects, a 2 MiB DECODED cap enforced during the read AND again
after content-encoding, 2 retries, loopback/private/link-local/multicast/
reserved reject on EVERY redirect hop, MIME media-type parse, one ~25s
monotonic deadline, UA + charset detection. Stdlib-only so a client bundle can
import it with no dependency and activation never drags in requests/feedparser.
THE DESIGN DECISION worth carrying forward is the FAILURE SPLIT:
`FeedFetchRefused` (a bound of OURS tripped -- loud, never retried, never
swallowed) vs `FeedFetchUnavailable` (the remote did not deliver -- an ordinary
per-URL miss a caller holding other candidates may catch). Collapsing them
either lets one paywalled article kill a run, or makes a redirect into the
private network look like a paywall. The article scraper therefore keeps
returning "" for Unavailable (unchanged degrade-to-next-candidate) while a
Refused propagates.
THE FIND: re-pinning at HEAD showed the plan undercounted -- there were THREE
unhardened hops, not two. The third, `_otr_media_archive_sources.
parse_media_archive_feed`, handed feedparser a URL with no bound at all. Also
worth keeping: `_fetch_single_feed`'s `socket.setdefaulttimeout(7)` was never a
per-feed timeout. It is PROCESS-GLOBAL, and a ~30-wide thread pool set and
restored it concurrently, so the timeout any feed actually ran under was
whatever another thread had most recently installed. It only looked like a
bound. Both hops now hand feedparser a STRING; it never touches the network.
`8c45172d` = the `missing_module` quarantine message told clients the bundle
"must ship one module with fetch_source + interpret_source +
check_compatibility". False -- a bundle with no `check_compatibility` activates
cleanly, as `test_otr_check_cli.py` already asserted. Fixed + regression test.
Operator-directed consult on the flagged unwired-constant fork: codex
`gpt-5.6-sol` high and Fable, independently, both said RIP (Option B), and both
found the `:353` falsehood on their own. The argument that moved it: Option A's
stated benefit is factually false -- `BUNDLE_ENTRY_ATTRS` reserves nothing
against clients, it only constrains what OTR-side code may ask
`bundle_entry_point()` for. The rip itself was NOT executed: it touches landed
wave-3/4 code and the plan of record, which a coder window does not own. It is
flagged in GO_FORWARD with the 2-of-2 recommendation and a verified blast
radius for the operator/planner.
Self-correction worth keeping: the first version of the call-site guards in
`tests/test_feed_fetch_seam.py` grepped the source text, and failed -- against
the comments that explain WHICH unbounded call was removed. A guard must not
fight the documentation of the thing it guards; they read the AST now.
Gates: suite 6365 passed / 27 skipped / 1 xfailed (was 6294; +70 seam tests,
+1 message regression); Bible 17/24/3; AST/BOM/zero-byte/UTF-8 clean on all
seven touched files; canonical byte-identical A66A416B (no node/widget/link
touched). Pathspec commits -- the other window's three modified tmp/*.ps1 and
all untracked scratch preserved.
Note for the next window: `git commit -m` with a multi-line PowerShell
here-string mangles into stray pathspecs (`fatal: '/' is outside repository`).
Use `git commit -F <file>`.
Current step: CODER E wave 6 -- the ledger-cleanup pass in the shared tail,
which also owns the client-interpreter fallback gap
(`build_source_interpreter_fallback` switches on the four SHIPPED interpreter
ids and gives a client interpreter a confusing `UnknownInterpreterError`).
Next: fresh CODER E window takes wave 6. CODER A (bug-first) and RENDER remain
open in parallel. Operator/planner still owns the `check_compatibility` fork.
Models: Claude Opus (rung 4) + one operator-directed consult -- codex
`gpt-5.6-sol` high (rung 3) and Fable (rung 6), both off their usual use by
explicit operator instruction, run in parallel so they cost no coder time. No
strike against the two-strikes law; no failure drove them.
Commits: c97a0e91 (wave 5) + 8c45172d (message fix) + this handoff commit.


## 2026-07-24 14:05 -- HEAD 84945bc4 (v2.0-alpha) -- WINDOW CODER E (Opus)
Did: independent source banks WAVE 4, one green pushed chunk @ 84945bc4 -- the
`otr_check bank <path> [--activate] [--all] [--json]` CLI (`scripts/otr_check.py`
+ `otr_check.bat`, OTR_PYTHON -> venv -> py -3 resolution, PYTHONUTF8 set).
The CLI owns NO format: `_otr_user_banks` gained `preflight_bundle`,
`preflight_bundle_record`, `write_activation`, `activation_status`,
`UserBankActivationError` and the status constants, and `_validate_bundle` was
split into `_validate_authoring` + the receipt half so the authoring checks can
run on a bundle that has no receipt yet -- boot's check ORDER is unchanged, so a
doubly-broken bundle still reports the code it always reported.
THE FIND, and the reason wave 4 was not just a file writer: `discover()` is NOT
all of admission. `_admit_user_banks` runs `_sweep_pack_dir` + `_crossref_bank`
AFTER it, so a checker that validated with the row parser alone would hand a
receipt to a bank that quarantines at boot as `bad_bundle_contract` -- an
activation that says yes to a bank the operator can never select. New routing
seam `validate_client_bundle_contract()` runs exactly those two, and the CLI
runs it BEFORE any write and also without `--activate`. Surfaced by the kibitz
r3 panel, grounded against `_admit_user_banks` before accepting.
Publication order is the safety property: staging copy -> hash the COPY against
the validated digest -> `os.replace` the snapshot -> THEN the receipt (staged
outside the bundle, because a temp file inside it would join the authoring bytes
and change the digest being recorded). A crash between the two leaves the bundle
UNCHECKED, which is honest; the reverse leaves a receipt naming a snapshot that
never existed. Probe runs in a bounded child killed as a process TREE
(`taskkill /F /T`) and binds each self-owned lane against the writer's real
keyword sets -- `fetch_source(bank, technical_model, source_ref, load_config,
policy)` and `interpret_source(bank, payload, technical_fn, model_id)`, read off
the live call sites -- without calling anything. `fixtures/*.json` are validated
as recorded fetch payloads by `normalize_fetch_result`, the same validator the
live lane output meets; documented exactly that narrowly rather than as "runs
your fixtures".
DECISION -- `check_compatibility` NOT wired (Option A). No request type, no
decision type, no runtime consumer, so activation does not inspect it, not even
for callability; `EXTENDING_OTR.md` now calls it a reserved name instead of
"NOT YET WIRED". `COMPAT_ENTRY_ATTR` left inert with a comment. Codex argued for
deleting it outright; that touches landed wave-3 code and the plan of record, so
it is FLAGGED in GO_FORWARD Open risks for the operator/planner, not done here.
Gates: suite 6294 passed / 27 skipped / 1 xfailed (was 6264, +30 new tests in
`tests/test_otr_check_cli.py`); Bible 17/24/3; AST/BOM/zero-byte/UTF-8 clean on
all six touched files; canonical byte-identical A66A416B (no node/widget/link
touched). Committed by pathspec -- the other window's three modified tmp/*.ps1
and all untracked scratch preserved.
Self-correction worth keeping: I wrote a code comment claiming
`EXTENDING_OTR.md` had the wrong `fetch_source` signature. It did not -- I had
misread a wrapped line in a partial file read. Fixed before commit. Read the
whole declaration, not the line the offset happened to land on.
Current step: CODER E wave 5 -- the bounded `_otr_feed_fetch` seam, BOTH hops
(feed + article scrape): https-only, connect 5s / read 10s, 3 redirects, 2 MiB
decoded cap, 2 retries, loopback/private/link-local reject, MIME media-type
parse, one ~25s monotonic deadline, UA + charset. The r3 finding that network
hardening is NOT inherited still stands -- re-pin it at HEAD first.
Next: fresh CODER E window takes wave 5. CODER A (bug-first) and RENDER remain
open in parallel.
Models: Claude Opus (rung 4) + one kibitz arc -- r3 codex `gpt-5.6-sol` high
(model pin verified in `codex_model_selected.txt`), r4 agy `gemini-3.6-flash-high`
QA, which converged with no must-fix. Panel spent on a genuine design fork
(operator-directed), not on a failure; no strike against the two-strikes law.
Commits: 84945bc4 (wave 4) + this handoff commit.


## 2026-07-24 11:12 -- HEAD cc69e683 (v2.0-alpha) -- WINDOW CODER E (Opus)
Did: independent source banks WAVE 3, one green pushed chunk @ cc69e683 --
client bundles may now OWN their fetch/interpret lanes. A CLIENT row routes an
entry point to its bundle with the reserved id "self"; the shipped registries
never learn that value, so a SHIPPED row declaring it is still an unregistered
typo and a client can neither shadow nor extend a shipped entry point (a client
may instead REUSE a shipped id, or mix). `_otr_user_banks` gained the execution
seam: function-local importlib loads the bundle module under a DIGEST-STAMPED
sys.modules name (edited bytes can never be served from the stale entry; a
half-executed module is popped on failure), `bundle_entry_point` returns one
declared callable, and both raise the new `UserBankExecutionError` -- loud on
purpose, because discovery already quarantined the broken bundles, so by
execution time the operator has SELECTED this bank and a fallback would be a
silent substitution. `resolve_fetcher`/`resolve_interpreter` take `owner=` and
verify owner IDENTITY (owner.bank_id == bank.source_bank_id), not mere presence
-- otherwise bank A could run bank B's code. Client results still cross
`normalize_fetch_result` / `validate_interpreter_result` unchanged; client lanes
stamp `seed_source = "user_bank:<bank_id>"`. `_crossref_bank` unlocks the self
id on an explicit `is_client=True` param rather than sniffing the `origin`
label, so no future caller widens the exemption by relabelling. Writer wired at
both call sites, resolution still outside any try, AST-pinned. 29 new tests
(`tests/test_user_bank_execution.py`); `docs/EXTENDING_OTR.md` documents the
"self" rule + exact keyword signatures and marks `check_compatibility` NOT YET
WIRED rather than promising a contract with no consumer.
Gates: suite 6264 passed / 27 skipped / 1 xfailed (was 6235); Bible 17/24/3;
AST/BOM/zero-byte/UTF-8 clean on all six touched files; canonical byte-identical
A66A416B... Committed by pathspec -- another window's three modified tmp/*.ps1
and 828 untracked scratch paths preserved.
Known gap left OPEN on purpose (recorded in GO_FORWARD Open risks, owner = w6):
`build_source_interpreter_fallback` switches on the four SHIPPED interpreter ids
and raises UnknownInterpreterError otherwise, so a client interpreter raising
SourceInterpretError with an .attempts-carrying cause lands there with a
confusing message. Loud is correct; a generic client fallback is w6 ledger-
cleanup work, not a patch.
Third harness gotcha for the next window: the full suite takes ~100 s, past the
~60 s MCP ceiling -- launch it from a temp .ps1 via `Start-Process
-WindowStyle Hidden` writing to a log, then poll the log.
Current step: CODER E wave 4 -- `otr_check bank <path> --activate` CLI writing
the content-addressed snapshot + `.otr_receipt.json`. `_otr_user_banks` already
owns the format (RECEIPT_KEYS, RECEIPT_SCHEMA_VERSION "v2.0", bundle_digest,
snapshot_dirname), so w4 is the CLI + fixture preflight, not a new format.
Next: fresh Opus CODER E window takes wave 4. CODER A (bug-first items) and the
RENDER track remain open in parallel.
Models: Claude Opus (rung 4) only. No panels, no Codex, no roundtable spent --
the one red test was a bad assertion in my own new test, confirmed by temp probe
and fixed first try; no strike against the two-strikes law.
Commits: cc69e683 (wave 3) + this handoff commit.


## 2026-07-24 ~12:00 -- HEAD 66e214ec (v2.0-alpha) -- WINDOW CODER E (Opus)
Did: independent source banks waves 1-2, one green pushed chunk @ 66e214ec.
New `nodes/_otr_user_banks.py` -- client bundle discovery + integrity
(timestamp-free content-addressed digest over authoring bytes, activation
receipt + snapshot check, symlink/path-escape refusal, protected + malformed id
refusal); it NEVER raises for a bundle problem, it returns (admitted, issues).
`_otr_story_routing.py` now admits client rows ALONGSIDE the shipped six
through the SAME `_parse_bank` and the same pack/pipeline/seam cross-refs
(extracted `_sweep_pack_dir` + `_crossref_bank`); pack resolution routes by
OWNER via a new `pack_dirs` map instead of assuming the shipped root; registry
publishes atomically behind a re-entrancy guard; `_clear_caches` resets the
flag too; new `list_validation_issues()` / `user_bank_bundle()`. Asymmetry
pinned by test: a bad shipped seed still kills node registration, a bad client
bundle quarantines alone. 53 new tests (`test_user_bank_bundles.py`,
`test_user_bank_admission.py`). `docs/EXTENDING_OTR.md` updated to the landed
bundle layout / id rules / activation-staleness contract.
Gotcha for the next window: `test_story_pack_stage1.py::test_only_sanctioned_
consumer_uses_loader` is a plain SUBSTRING grep over `nodes/**.py` -- merely
NAMING `_otr_story_pack` in a docstring trips it. Reword, do not weaken the
guard. Also the known-fail-guard plugin swallows pytest's FAILURES section, so
diagnose failures with a temp probe script, not `--tb=long`.
Current step: CODER E wave 3 -- client-owned `fetch_source`/`interpret_source`
execution (`fetcher`/`interpreter` = `"self"`, bundle module loaded function-
locally, `_otr_source_payload` resolvers take an owner bundle, `_crossref_bank`
accepts `"self"` for client rows only; results still pass
`normalize_fetch_result` / `validate_interpreter_result`). Re-derive the writer
call sites in `OTR_LedgerScriptWriter.py` FIRST.
Next: fresh Opus CODER E window takes wave 3. CODER A (bug-first items) and the
RENDER track remain open in parallel.
Models: Claude Opus (rung 4) only. No panels, no Codex, no roundtable spent --
no fix needed a second attempt.
Commits: 66e214ec (waves 1-2) + this handoff commit.


## 2026-07-24 ~09:55 -- HEAD 314dd481 (v2.0-alpha) -- WINDOW PLANNER (Opus)
Did: ran extensibility hardening. Full r1-r4 `/kibitz` arc + an r5
simplification pass on the user-source-lanes architecture (codex gpt-5.6-sol
high + agy Gemini 3.6 Flash High; Claude anchor+grounding+judge; 10 panel
calls). Grounded every claim vs the real Windows files at `d550aff8`. Caught the
stale base: NO `science_news` bank; six INDEPENDENT banks; `_RUNNER_BY_PIPELINE`
= 2 + `_LEGACY_INLINE_PIPELINES` = 3 (legacy_many_pass / legacy_many_pass_adapt
/ original_multi_pass); `_otr_story_rules.py` deleted. Operator reframed LIVE: N
independent client-authored banks (NO Path A/B, no family), trusted shared
writer builds the COMPLETE ledger (the #1 key), a ledger-cleanup LLM pass,
content by REPAIR never a story-fail (SFW dropped as a gate), broken bundle
quarantines; DEFERRED client own-runner+staging + deps subsystem + standalone
story_rules. Wrote the lean plan of record
`docs/2026-07-24-independent-source-banks-v1-plan.md` + the r6 rebase brief;
retired the 1265-line A/B doc to decision-log status; leaned GO_FORWARD (agy
panel lane -> Gemini 3.6). Decision log: `kibitz-runs/2026-07-24-user-source-lanes-r6*/`.
Current step: extensibility hardening DONE, AND `docs/EXTENDING_OTR.md` DRAFTED
same session (complete-ledger contract grounded per-consumer via a DC fan-out:
voice loop / scene_sequencer / shot_lock / captions / credits roll /
master mux+obs_publish, with SOURCE_BANK_GUIDE s5+s7 as the authored-inputs
base) + linked from README's source-banks section. CODER E UNGATED.
Next: CODER E (operator-chosen) -- code independent-banks lean v1 on an OPUS
window (Fable stays reserved for the section-9 epoch gate; this is structural
code = Claude rung 4, Qwen rung-1 triage, codex gpt-5.6-sol high via
two-strikes). Re-derive every line pin at the recorded HEAD before editing.
CODER A (bug-first) remains open as the parallel track.
Models: Claude Opus (planner/judge) + 10 kibitz calls (codex gpt-5.6-sol high +
agy Gemini 3.6 Flash High). $0 local panel + Codex weekly credits.
Commits: docs handoff (this session's docs by pathspec).


## 2026-07-24 08:00 -- HEAD 314dd481 (v2.0-alpha) -- WINDOW PLANNER->CODER (Fable)

Did: LANDED the six-bank no-prose-gate retirement chunk @ 314dd481 (312
files, +8,085/-74,529: provider-capacity whole-artifact contracts, word-fit
ceiling rip, structural markup acceptance, G13 retirement, receipt-truth
hardening, repair-first P0, Qwen-Image removal; incl. 5 new tests + 8 dated
docs; canonical json byte-identical A66A416B...). Gates: suite 6182/27/1,
Bible 17/24/3, AST/BOM/zero-byte clean, pushed, HEAD==origin. tmp/ scratch +
otr_sbcov profiles intentionally left untracked. GO_FORWARD refreshed:
worktree CLEAN, current step -> six-bank requalification + bug-first fixes.
Current step: requalify the captured six-bank leg on landed code, then
bug-first items (receipt-truth live confirm, still ownership, WAN contract).
Next: fresh Opus window -- PLANNER (sec-16 + r5 kibitz, codex fresh) and/or
CODER A (bug-first items); coder slot is FREE.
Models: Claude Fable only; suite/Bible local; no panels spent.
Commits: 314dd481 (chunk) + this handoff commit.


## 2026-07-24 07:35 -- HEAD ed8d5a6d (v2.0-alpha) -- WINDOW PLANNER (Fable)

Did: leaned GO_FORWARD_PLAN.md to open work + bugs only (665->398 lines; done
strata retired to git history + this log; stale refs re-grounded: retired
banks pruned, phase-C gating -> "no code mid-sweep"); added the MODEL & CREDIT
BUDGET section + per-window model rungs; authored + delivered otr-handoff
SKILL v2 (commit-AND-push policy, tracker/audio-freeze staleness removed).
Current step: land the dirty-tree six-bank no-prose-gate chunk (active coder
window); PLANNER next = sec-16 ratification + r5 extensibility kibitz.
Next: fresh Opus PLANNER window takes this baton; run the sec-16/r5 kibitz
(codex gpt-5.6-sol high + agy Gemini 3.5 Flash (High)) while both pools fresh.
Models: Claude Fable, docs-only session; no panels or roundtable spent.
Commits: ed8d5a6d + this handoff commit.


## 2026-07-22 early -- v2.0-alpha [CODER: live candidates stay fresh]

PBUG-20260721-18's episode-liveness root fix is pushed at `67996907`; the
live qualification follow-up is pushed at `81ee21df`. The deterministic
in-band ledger remains the only delivery judge. Four consecutive no-progress
calls retire only the current producer candidate. Row repair escalates to the
alternate LLM and then to another complete producer-owned candidate without an
outer model-output ceiling.

ROOT FOLLOW-UP:
- Canonical prompt `32b374e2-7c89-4d4a-bb8c-42e180571ecc` stayed alive for
  more than two hours and retired more than a dozen candidates, proving that
  no LLM miss or observer exit killed the episode. It also exposed a real
  convergence defect: both logical slots resolved to the same seeded Gemma
  backend, so two fixed P5 prompt shapes replayed the same drafts.
- Every complete reroll after Candidate 0 now carries a model-visible,
  monotonically unique candidate nonce and explicit fresh-candidate
  instruction. The compact typed-repair context preserves that identity.
  Corrected prompt `3fdf7349-7b2e-46f5-8182-982f72e5e261` has already
  produced visibly distinct Phase One/Phase Three P5 candidates and continued
  through P6/P8 without a terminal episode failure.
- `poll_history(timeout_s=0)` is now explicit wait-until-terminal operator
  mode. Default callers retain the 5,400-second timeout; only the overnight
  qualification harness opts into no observation wall clock.

VALIDATION: whole Windows suite **8,349 passed / 33 skipped / 1 expected
xfail** in 205.28 seconds. Bug Bible 12.70 passed **17 / 23 skipped / 3
expected xfails**. AST, UTF-8/no-BOM, nonzero-file, JSON round-trip, link/input,
live widget-vector, and OTR workflow validator coverage are green. The
canonical workflow stayed byte-identical at
`f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a`
(23 nodes / 58 links). `HEAD == origin/v2.0-alpha == 81ee21df`.

LIVE QUALIFICATION: run tag `qual320_nonce_20260722` is active. A hidden
pass-gated chain adopted the corrected `scifi_news` canary and will launch
`scifi_news_pro`, `original`, `media_archive`, `public_domain`, and
`shakespeare` sequentially only after each prior leg records RESULT SUCCESS
and passes the strict ledger, exact word receipt, caption, credits, asset, mux,
and OBS publication audit. Any real leg or audit failure stops the chain.


## 2026-07-20 late -- v2.0-alpha [CODER: spoken hygiene ships with a stamped repair]

Closed PBUG-20260720-03: a CRAFT/quality rejection on one voiced row can no
longer terminal-skip an otherwise renderable episode. The contract now applies
to all six runnable banks (`media_archive`, `original`, `public_domain`,
`shakespeare`, `scifi_news`, and `scifi_news_pro`).

ROOT FIX:
- Added a total per-line ladder: the existing same-slot repair, a sharpened
  gate-specific CRITICAL repair at lower temperature, the other writer slot,
  then an idempotent deterministic SFW floor. Every accepted repair is
  rescored and stamps `hygiene_repaired_after_reroll:<gate>:<rung>`.
- Extended the floor across the full spoken contract. Existing cliche and
  stage-business scrubbers are now terminal rungs; whole-line action/cue text
  becomes a short speakable utterance; one-breath, anchor-stuffing,
  objective-literal, on-the-nose, and thesis findings receive bounded
  sentence-preserving repairs. Speaker-aware detection catches a character
  narrating their own action by name. Non-dialogue material is not moved into
  SFX yet; that ledger layer remains future work.
- Moved whole-script Codex P5/P7/P9 craft failures inside the typed-repair
  factory, after graph/roster preflight, so a local wording defect never spends
  or truncates a whole `ScriptArtifactV4` retry. Content-owned lanes repair the
  exact TTS projection before rebuilding raw/parsed/proof/hash seals. Shared
  lanes receive a final ledger scour plus a post-readiness guard.
- Removed quality exhaustion from terminal freeze semantics. Empty output from
  the mechanical floor is isolated to that row; genuinely invalid graph state
  remains structural. The deterministic G9 SFW/content-safety ship-stop was not
  softened and still sanitizes or fails closed.

VALIDATION: focused cascade/Codex coverage **119 passed**; expanded six-bank
surface **395 passed**; workflow/freeze surface **268 passed / 3 skipped**;
whole Windows suite **8161 passed / 33 skipped / 1 expected xfail**. The clean
survival-guide worktree passed **17 / 19 skipped / 3 expected xfails** and
BUG-11.56's OTR executable regression passed; portable rule update
`ef7e327ded9cf80b9f050a690b4e09cc33d8e8d7` is pushed to the guide's `main`.
`workflows/otr_canonical.json` needed no node/input/widget/link change and stayed
byte-identical (`222D19478A308C91171DFCBDCCBEC01C55DD639283E2550EBB59EB9842D0882D`);
validator, JSON round-trip, 23-node/57-link audit, live input names, references,
and widget-vector drift (`0`) are green.

LIVE PROOF: an initial canonical episode, `signal_lost_the_price_of_wakefulness_20260720_210832`,
published successfully through the late floor and exposed the remaining
whole-artifact boundary and raw-token trim; both were root-fixed. Final
canonical prompt `f3770246-2d6a-4302-90af-153120edddf2` then hit real defects at
P5 (`one_breath`, four rows) and P7 (`spoken_format` / `stage_direction`). Each
immediately logged `craft-only rejection resolved by the line-local A/B/C/floor
cascade; whole-artifact repair bypassed`, and the ledger carries
`shared_artifact_repair_bypassed=true` plus gate/rung stamps. The episode froze
`frozen_with_warns` (only stale word-count telemetry), rendered all four clean
lines / 45 words, completed TTS/video/captions/credits, and published the
22,892,541-byte OBS asset:
`output/otr/obs/signal_lost_the_weight_of_height_20260720_221418_silent_procgen_blended_captioned_with_credits_final.mp4`.
Targeting 30 words already produced a clean 45-word episode, so no minimum-word
widget change was needed; increasing the minimum would not fix the separate P9
8K structured-artifact capacity limit.

## 2026-07-20 -- v2.0-alpha [CODER: Gemma 4 12B Transformers/HF writer restored]

Restored `google/gemma-4-12b-it` as the saved creative + technical writer on
the fully local, in-process Transformers/HF lane. OTR uses no Ollama,
llama.cpp, model sidecar, or model-serving port for this path. The official HF
weights remain under `C:\ComfyUI-Models\huggingface\hub`; both canonical slots
select `cuda` / `sdpa` / `bnb_nf4`, OTR context 8192. No LoRA, adapter, or
auxiliary tensor artifact is required.

ROOT FIX:
- Upgraded the runtime contract to native Gemma4Unified support
  (`transformers>=5.10.4`), restored the curated HF row, removed its hard
  reject, and made cache resolution pair the newest materialized-weight
  snapshot with the newer local chat-template metadata revision.
- Kept tokenizer/config/model loading fully offline with
  `local_files_only=True`; there is no hidden Hub fallback in `load_llm`.
- Bound each exact P0-P9 result schema into the real local scheduler calls,
  including the narrower P3 authored-text patch, so lm-format-enforcer removes
  invalid JSON continuations at token selection.
- The first live leg found one grammar-compiler incompatibility in P5:
  `list[dict[str, Any]]` emitted `additionalProperties:true`, which LMFE 0.11.3
  treated as a schema object and crashed on. `ScriptSceneV4` now expresses the
  actual closed `scene_id` / `env` / `description` contract. A complete P5
  artifact is exercised character-by-character through the installed parser.
- Updated the real `workflows/otr_canonical.json` and revalidated all 23 nodes,
  57 links, positional widget vectors, live input names, references, and JSON
  round-trip.
- Added `scripts/otr_gemma4_doctor.py` for the official offline NF4 + coherent
  prose + constrained-JSON contract. Bark/MusicGen compatibility tests stay
  green under Transformers 5.10.4. The separately installed legacy
  `parler-tts 0.2.2` pin is incompatible with Transformers 5 and must remain
  isolated; Parler is not an OTR dependency.

MEASURED: official `Gemma4UnifiedForConditionalGeneration`, 331 Linear4bit
layers, `is_loaded_in_4bit=True`; 7.152 GiB allocated / 7.286 GiB doctor peak,
and a 7.15 GiB live model-load delta. Canonical structured generation peaked
around 13.9 GiB total GPU use including the desktop baseline and KV state,
inside the 16 GB board.

VALIDATION: exhaustive fresh-process inventory **8123 passed / 33 skipped / 1
expected xfail across all 488 test files**. Focused post-fix compatibility
surface: 291 passed / 2 skipped. Survival-guide suite: 30 passed; BUG-02.16 and
BUG-11.55 OTR regressions: 2 passed. The Bible loader reports 205 entries and
only its 12 pre-existing xref-tag format findings.

LIVE PROOF: canonical prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f` loaded the
official checkpoint fully offline for both slots. P0 began with `{`, decoded,
and reached semantic source-span validation; deterministic coordinate repair
accepted it without another model call. P1-P4 and P3 rewrite cleared. P5 then
produced a complete schema-valid JSON artifact, proving the LMFE crash fixed.
The leg did **not** publish media: Gemma repeated an existing spoken-hygiene
defect after its bounded P5 model repair, so the lane failed closed as designed.
This is a runtime/grammar qualification through P5, not a full-episode or
comparative quality-bakeoff verdict.

LM STUDIO CONVENIENCE: imported the existing Q4_K_M GGUF as an NTFS hard link
at `C:\ComfyUI-Models\LMStudio\unsloth\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf`.
It consumes no second weight copy and is separate from OTR's HF runtime. LM
Studio and its service/server were left stopped.

STILL OPEN: the GGUF lane's structured-enforcement gap remains separate and
the optional GGUF row was not presented as the canonical writer. The local
Gemma-vs-Mistral quality matrix also remains open.

## 2026-07-18 evening -- HEAD `ed7b37de` (v2.0-alpha) [RENDER->CODER: short-episode structural COUNT gates -> advisory (Gate 3)]

Started as the RENDER window for the local Mistral-Nemo bake-off (codex_v4 vs
fable2 vs base codex). Precondition confirmed (HEAD `c507acff`, exact 8-id roster).
The Step-1 wiring smokes surfaced a blocker that turned into the session's real work.

DIAGNOSIS (docs/2026-07-18-render-step1-blocker.md):
- 30w AND 120w canonical smokes hard-fail in the WRITER on deterministic STRUCTURAL
  COUNT gates: codex P3 exact-beat-count (`beat count 6 must equal advisory 12`;
  root: `_otr_scifi_codex.py:3297` derived beats from `cast*3`, word-blind), and
  fable2 WORD_BUDGET/SCENE_COUNT bands. NOT a rip regression (c507acff never touched
  the video path or fable2 lane); the gates are v4-bake-off-era regressions (git:
  `c22eef0a`/`c942b2ae`/`95582643`) -- the pre-source-bank lanes ran any length.
- One EARLY false lead: the first codex smoke booted in leaked `OTR_TEST_MODE=1`
  (Start-Process inherits parent env) -> in-memory stubs -> empty video manifest.
  Fixed the harness (leg runner strips test env) and re-ran clean.
- Governing contract = `docs/SOURCE_BANK_PREFLIGHT.md` Gate 3: "no model-produced or
  unused count field can gate production"; `target_words` advisory, never a fatal
  quota gate. The gates were non-compliant.

FIX (committed `ed7b37de`; operator-approved "fix the gates", kibitz r3 hardened):
- codex: beat count scales to the word budget; a beat-count mismatch is RECONCILED
  (advisory rebuilt to the draft's actual count) and propagated into P3/P4/P3_rewrite;
  cast_coverage is advisory; an out-of-range cue anchor is deterministically CLAMPED.
  Dangling-reference gates (shot_index/cast_id/fact_id/cue_id/unused_shot/graph) stay
  fatal -- `_validate_radio_score_graph` still closes.
- fable2: word/scene COUNT defects drive bounded rerolls only; on exhaustion the
  cleanly-parsed draft is ACCEPTED and residuals recorded advisory in the ledger
  (`f2.parse`/`parse_p5`). PARSE defects still fail closed.
- kibitz r3 (Codex + Antigravity/Gemini 3.5 Flash High) caught 3 real wiring gaps I
  folded: advisory-recording, the P3_rewrite reconcile propagation, and the
  cast_coverage accidental-fatal-successor (both fired correctly on the live leg).

VALIDATION: full suite 8082 passed / 32 skipped / 1 xfailed; Bug Bible 17 passed;
AST + no-BOM + HEAD==origin verified. LIVE PROOF: `scifi_fable2` 120w Mistral-Nemo
leg RESULT SUCCESS + obs_publish OK + asset on disk ("The Caretaker's Dilemma",
108.0 MB) -- previously a hard WORD_BUDGET fail.

STILL OPEN (a SEPARATE facet, NOT count gates): codex_v4 short legs still fail
stochastically on P2 cast-name Title-Case (e.g. `Maxwell 'Max' Hart`) and P5
self-vocative -- a codex-writer robustness follow-up under the same Gate-3
"mechanical normalization" principle (P2 could be as small as stripping quote
tokens from names). The local Mistral bake-off itself is NOT yet run (blocked on
these codex facets); run it at 420/720w once codex short legs are clean.

## 2026-07-18 midday -- baseline HEAD 178e935a (v2.0-alpha) [CODER: Sonnet-bake-off rip -- 4 banks retired]

Executed `docs/2026-07-18-rip-4-banks-plan.md` in one green chunk.

Did:
- RETIRED `scifi_sonnet_v3` (FULL sonnet lane): bank row + pack + story_rules +
  `sonnet_archive_multipass_v3` pipeline (both registries) + the
  `_run_scifi_sonnet_lane` runner + deleted `nodes/_otr_scifi_sonnet.py` +
  `tests/test_scifi_sonnet_lane.py`. RETIRED `media_archive_v3` / `scifi_codex_v3`
  / `scifi_fable2_v3` (v3-only): row + pack + story_rules + each dedicated pipeline
  in BOTH `_RUNNER_BY_PIPELINE` and `pipelines.json`. KEPT the `scifi_codex` /
  `scifi_fable2` / `media_archive` bases, `scifi_codex_v4`, and `legacy_many_pass_v3`.
  Roster: 12->8 visible, 11->7 runnable.
- MUST-KEEP fence honored: deleted ONLY `_make_v3_runner`; KEPT `run_v3_advisory`
  / `_v3_focus_metric` / `_v3_max_run` (public_domain_story_v3 + shakespeare_v3 call
  them every render). KEPT the now-unreachable `base=="scifi_sonnet"` focus branch
  and the `_otr_scifi_p0_contract.py` P0-contract comment -- the only 2 surviving
  bare-`scifi_sonnet` hits, both in shipped code. Dropped `fable2_multipass_v3` from
  the writer target-word gate; refreshed the stale `_RUNNER_BY_PIPELINE` comment.
- CLEAN RIP tests (positive only): migrated the surviving-machinery advisory tests
  to `public_domain_story_v3` / `shakespeare_v3`; scrubbed `_otr_scifi_sonnet`
  imports + sonnet-only cases from schema-parity / rss-admission / source-repair;
  regenerated the roster/bijection pins and the v4-guard `_CURRENT_BANKS` lists.
  Operator eyeball on the v4-guard gate-off contrast: KEEP base `scifi_codex` (guard
  genuinely OFF), NO `_v4` substitute.
- NEWBUG->PBUG: appended `PBUG-20260718-01` to PROD_BUG_LOG FIRST, then marked
  `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md` CLOSED-BY-RIP (retained, never deleted).
- Docs: README roster table, GO_FORWARD current-roster + NEWBUG note refreshed.

Gate (all green): import-smoke 0 skips; `_ensure_loaded()` carries no retired
pipeline id (atomic delete validated by the crossref sweep); `otr_canonical.json`
byte-unchanged; source-only retired-id scan over nodes/tests/workflows = ZERO;
bare-sonnet scan = EXACTLY 2 (both kept); no surviving `meta["scifi_sonnet"]` reader;
runtime-advisory proof via the migrated `public_domain_story_v3` unit test (plan's
"targeted unit test OR 30w live smoke" -- unit-test path taken; live smoke not run);
**full Windows suite 8081 passed / 32 skipped / 1 xfailed** (was ~8144 pre-rip -- drop
is the retired banks' own tests); **Bug Bible 17 passed / 16 skipped / 3 xfailed**;
no-BOM/UTF-8 + AST-parse on every touched file. Counts recorded, not pinned.

## 2026-07-18 morning -- HEAD 60c73618 (v2.0-alpha) [RENDER: Sonnet-4.5 cross-bank bake-off COMPLETE]

Did (render window, autonomous overnight):
- Ran the creative=`claude-sonnet-4.5` (OpenRouter remote) / technical=`Mistral-Nemo` (local) bake-off
  across all 11 runnable banks x 420/720 = 22 story-only legs (18 SUCCESS / 4 FAIL). Built the harness
  (tmp/_sonnet_bakeoff_sweep.ps1); fixed 2 wiring bugs live: the concrete-4.5 dropdown pin (the picker
  prunes concrete slugs for ~latest aliases -> surface it via OTR_OPENROUTER_SLOT_A_DEFAULT) and the
  -Banks [string[]] array-binding trap via Start-Process/-File (-> single comma-string the script splits).
- Fable BLIND grade of the 10 720-SUCCESS transcripts. NEW WINNER under Sonnet = scifi_codex_v4 (24/25,
  "The Halicin Gamble"); runner-up scifi_fable2 (24/25, monologue-capped at 720); the codex circuit swept
  #1/#3/#4; weakest scifi_sonnet_v3 (12/25, essayistic). The crown SHIFTS from the aion baseline's fable2.
- Cost ~3.07M Sonnet tokens ~= $15-20 (creative slot only; technical local/free; 0 creative VRAM).
- FAILs diagnosed: original_radio 420 (deterministic news_source_framing gate; PASSED at 720),
  scifi_codex_v4 420 (codex P5 all-caps-word gate; PASSED at 720), scifi_fable2_v3 BOTH tiers = NEWBUG
  (fable2 revision_contract hardcodes rules_id=='scifi_fable2', model-independent) ->
  docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md.
- Scoreboard: docs/2026-07-17-model-bakeoff-scoreboard.md. Full-media confirmation: the winner
  scifi_codex_v4 @ 720w canonical FAILED fast (codex 240-char string_too_long on a fresh source -> the
  winner is production-fragile with Sonnet, different gate than its 420 all-caps fail). Re-ran on the
  robust runner-up scifi_fable2 @ 720w -> RESULT SUCCESS + obs_publish OK ("The Stone Frequency", 406 MB,
  34:12). Shippable Sonnet pairing = scifi_fable2; codex_v4 = best script, least reliable producer.
Current step: bake-off item 3 (Sonnet arm) DONE; Mistral-Nemo stays the free local default, cloud opt-in.
Next: (coder) fix the scifi_fable2_v3 rules_id NEWBUG; (render, optional) the local mistral/gemma writer matrix.
Commits: docs only (scoreboard + NEWBUG + GO_FORWARD + HANDOFF); NO code changes (NEWBUG deferred to coder).

## 2026-07-17 night6 -- HEAD 9730e2dc (v2.0-alpha) [v4 P2 bank #1 scifi_codex_v4 GREEN + LIVE-PROVEN]

Did (coder window, autonomous + operator cross-check):
- Resumed after the operator cross-check verdict: BUG A (P0 literal-span fail) = NEW upstream root in the
  S5 family (-> PBUG-20260717-01); BUG B (P3 premise string_too_long) = re-occurrence of PBUG-20260713-04,
  and my base-seam 144 re-add was the -04 anti-pattern (exposing the rejection edge).
- P0 fix @ 26ba8e1d: normalize the 4 span-bearing source fields to single-spaced text in
  validate_payload_envelope -- at admission, UPSTREAM of the digest/projection/validator (BUG-11.37
  offset-shift constraint); point the P0 validator at env.payload. Codex-scoped (shared
  validate_source_payload stays byte-identical for science). +1 test; reverted the anti-pattern caps.
- Live legs: 6883758f (P3 premise), ac027c36 (P0), 90f22b15 (cleared P0 -> BUG A proven, then P3
  string_too_long on premise+description: the -04 recipe is insufficient for the verbose v4 lane).
- Operator "allow longer text": P3 fix @ 9730e2dc = RAISE the non-spoken metadata caps (premise 144->240,
  scene/shot description 72->144) across draft+final models + _p3_text_patch_cap + replacement_text schema
  + receipt. Caps are LOAD-BEARING (P3 draft fits the 8192 context+output budget) -> resized the reservation
  1647->1829 + updated every exact-token guard (max-width helper draft 1418->1576; envelope re-verified
  prompt+output=5935<=8192). Full suite 8144 / Bible 17 at each chunk; canonical unchanged.
- LIVE PROOF: leg c1f3891f RESULT SUCCESS + obs_publish (signal_lost_the_whisker_effect..._final.mp4,
  56.6 MB; obs + episode dirs Test-Path OK). Bank #1 DONE.
- PBUGs: PBUG-20260717-01 (P0) LIVE-VERIFIED; BUG B recorded as re-occurrence of -04 (not a new PBUG);
  PBUG-20260710-07 = retire candidate via the green codex leg (announcer rows clean, freeze passed).
Current step: bank #1 scifi_codex_v4 GREEN + live-proven. NEXT = bank #2 shakespeare_v4.
Next: build shakespeare_v4 (own idiom; inline legacy_many_pass_v4; genre+outro gates safe there;
  pre-emptively raise tight non-spoken caps + resize the budget/guards when raising).
Commits: 3b74b7e3 (contract-visibility fold), 48f2a278 (caps re-add, later reverted), cc76dcc5 (pause docs),
  26ba8e1d (P0 fix + anti-pattern revert), 9730e2dc (P3 caps raised). All pushed, HEAD==origin.

## 2026-07-17 night5 -- HEAD 48f2a278 (v2.0-alpha) [v4 P2 bank #1: two-strikes kibitz + P3 fold; LIVE LEG BLOCKED at P0+P3 -> PAUSED for cross-check]

Did (coder window, autonomous):
- Two-strikes gate on the codex P3 contract: ran /kibitz r2 (local $0; Codex gpt-5.6-sol + Antigravity
  Gemini 3.1 Pro, both grounded + Claude anchor/judge). Panel BROKE the framing: the seam cap-restatement
  was (argued) redundant with the surface instruction's tighter ceilings, and FOUR deterministic P3 compiler
  gates were model-invisible (unused_shot/cast_coverage/cue_id/cue_anchor), plus a 12-beat distribution trap.
  Folded grounded survivors @ 3b74b7e3: reverted the cap list, exposed the 4 gates + 12-beat clause in the
  shared surface/topology instruction, enriched the beat_count receipt (observed-vs-expected), +4 tests, doc
  fixes (PBUG cites -> -02/-06; cast beats 6/9/12; P5 does not cap prose). Suite 8143 / Bible 17.
- LIVE 30w Mistral-both leg 1 (6883758f) FAILED P3 string_too_long on `premise`. Grounded: the text-patch
  deliberately never clips prose (_otr_scifi_codex.py:1748), so model-visible caps are the ONLY lever -> the
  live evidence OVERTURNED the panel's "redundant" call (reverting the caps regressed it). RE-ADDED the caps
  + a premise-brevity nudge @ 48f2a278 (suite 8143 / Bible 17).
- LIVE leg 2 (ac027c36) then FAILED EARLIER at P0 PostValidationError -- FactIndex literal-span vs
  whitespace-polluted RSS source (full_text leading \n+8 tabs; offset slices land mid-word; model
  paraphrases; exact-literal contract rejects). PRE-EXISTING + SHARED across all codex banks; NOT v4-caused.
  So the P3 caps fix is UNPROVEN (leg 2 never reached P3).
- Per operator's new directive, wrote BOTH bugs as problem statements (docs/2026-07-17-v4-campaign/NEWBUG-*.md)
  for a cross-check window. Operator chose PAUSE for cross-check. Reset the box (killed the resident server).
  NO further codex code until the operator returns the fix approach.
Current step: v4 P2 bank #1 live leg BLOCKED at P0+P3 -> PAUSED for operator cross-check vs past PBUGs.
Next: operator cross-checks BUG A (P0 span/whitespace) + BUG B (P3 premise) vs PROD_BUG_LOG/BUG_BIBLE/
  BUG_SYMPTOM_INDEX; then kibitz the offset-sensitive P0 fix + confirm the P3 caps on a leg that clears P0.
Commits: 3b74b7e3 (contract-visibility fold), 48f2a278 (caps re-add). Both pushed, HEAD==origin.

## 2026-07-17 night4 -- HEAD 1fd7743d (v2.0-alpha) [v4 P2: bank #1 scifi_codex_v4 CODE SHIPPED]

Did (coder window, autonomous):
- Built scifi_codex_v4 as a fully INDEPENDENT bank: banks.json row (before custom) + pack
  nodes/story_packs/scifi_codex_v4/scifi_codex_v4.json (11 codex seams + the proof-pressure
  delta: want / gating proof / mandatory cost beat / one reversal) + story_rules/scifi_codex_v4.json
  (exact id) + pipeline scifi_codex_circuit_v4 mapped DIRECTLY to _run_scifi_codex_lane (NOT the
  v3 advisory wrapper) + roster/bijection tests (test_bank_variants 11->12 visible/10->11 runnable
  + TestScifiCodexV4; test_fable2_registry tail/order). Gates ON: require_science_floor +
  placeholder_guard(G13) + scene_coherence_check(G15). Gates DEFERRED: genre_guard_spoken(G10) +
  require_outro_cast_complete(G12) -- the dedicated codex runner does NOT cross the inline I.7/I.8
  authored-repair boundary, so they would be no-repair hard gates (vetoable). Full suite 8139 /
  Bible 17 / AST+JSON+BOM clean / canonical hash unchanged / HEAD==origin. Commit 1fd7743d pushed.
- Live 30w leg via scripts/otr_headless_canonical.ps1: attempt1 (Mistral-Nemo both) AND attempt2
  (gemma-4-E4B creative) BOTH failed at codex P3 RadioScoreV4 string_too_long -> proven
  MODEL-INDEPENDENT = the unstated-cap class (PBUG-20260713-11/12). ROOT FIX (operator-steered):
  restate the exact RadioScoreV4 caps in the codex_radio_score_system seam -- NOT a model swap.
  Re-proving with Mistral-both + the restated caps (the strict model-agnostic test).
- Wrote docs/BANK_PLAN_scifi_codex_v4.md (tracked; wiring + gate rationale + the PBUGs/lessons +
  the go-forward recipe for the remaining 4 banks).
Current step: scifi_codex_v4 code shipped @ 1fd7743d; live leg re-proving with the P3 cap fix.
Next: confirm RESULT SUCCESS + obs_publish + asset; if green, commit the cap fix + bank plan +
  doc refresh and retire PBUG-20260710-07, then bank #2 shakespeare_v4 (inline lane -> genre+outro
  gates ARE safe there). If the fix leg fails P3 again -> /kibitz (two-strikes) before a 3rd fix.
Commits: 1fd7743d (code). Cap-restatement + bank plan + doc refresh pending the live proof.

## 2026-07-17 night3 -- HEAD d29ba920 (v2.0-alpha) [v4 campaign: PHASE 1 COMPLETE (ii-viii)]

Did (coder window, autonomous -- continued):
- P1(v) @ 0066f5ab: outro cast-completeness. New nodes/_otr_outro_guard.py (final
  cast = character char_ids with a non-skipped spoken line -> name; outro = LAST
  announcer line BY POSITION; missing = name absent full-or-significant-token,
  casefold word-bounded, titles ignored). Authored keep-if-complete repair (creative
  slot; Python never appends prose; restores original on exhaustion). Deterministic
  G12 terminal, opt-in via defaults.require_outro_cast_complete. Root fix caught by
  my own tests: outro is positional, not last-non-empty (that was the intro).
- P1(vii) @ e7bfb1fe: literal placeholder-token guard. New _otr_placeholder_guard.py
  (whole-value, token-boundary, quote/punct/case-tolerant over NAMED fields; X/Y/TBD/
  ...; 'X marks the spot' NOT flagged; music out of scope). G13, opt-in
  defaults.placeholder_guard. No repair (placeholder = generation bug the pack fixes).
- P1(viii) @ 4f8bd7aa: source-provenance normalizer. New _otr_provenance.py
  (public_domain license_status + shakespeare license_label/commercial_use_allowed +
  synthetic -> one record; spoken_coda + printed_credit templates). Writer stamps
  meta.provenance + fills credits_source_line when the bank default did not.
  Deterministic G14 blocks publish on research_only (operator decision).
- P1(vi) @ d29ba920: header<->scene STRUCTURAL coherence. New _otr_scene_guard.py
  (unique scene_ids + no non-music line referencing an undeclared scene). Semantic
  scene-vs-beat match is an unlawful LLM gate -> structural only; exact
  scene.line_count matching omitted (unit-ambiguity risk). G15, opt-in
  defaults.scene_coherence_check. INTERPRETATION FLAGGED structural (vetoable at the
  Phase-2 consuming chunk). Done LAST after vii/viii per its under-specification.
- Pattern for all 7: each shared fix is a SELF-CONTAINED module + a deterministic
  terminal in _otr_ledger_freeze.run_gap_audit (G10 genre, G11 beat-floor, G12 outro,
  G13 placeholder, G14 provenance, G15 scene) -- the ONE path every execution family
  crosses (codex phase_10 finalizer, inline run_freeze_cascade, fable2 finalizer),
  mirroring G9. Every gate is OPT-IN via a validated scalar bank default (_parse_bank
  bool loop) -> INERT for all 10 current banks, so the full suite stayed green while
  the machinery is ready for Phase-2 v4 banks to flip on. THE LAW honored throughout
  (deterministic terminal ends; authored repairs only improve).
- Gates each chunk: full suite (8018->8031->8061->8084->8110->8134) + Bible 17 +
  AST/BOM/zero-byte + commit AND push + HEAD==origin. No canonical JSON change (no
  graph edit in any Phase-1 chunk).
Current step: v4 campaign PHASE 1 DONE. NEXT = Phase 2 -- build the 5 v4 banks,
  serialized, each an atomic per-bank chunk gated on a LIVE GPU leg (RESULT SUCCESS +
  obs_publish + asset). Order: scifi_codex_v4, shakespeare_v4, public_domain_story_v4,
  media_archive_v4, original_radio_v4.
Next: scifi_codex_v4 -- bank row + pack + story_rules(exact id) + pipeline
  scifi_codex_circuit_v4 (executable:true, runner-map) + roster/bijection tests; flip
  the opt-in gates it wants; runnable:true LAST; then the live leg (per-lane
  announcer-sentinel mint retires PBUG-20260710-07).
Commits: 0066f5ab, e7bfb1fe, 4f8bd7aa, d29ba920 (+ f5acd44a docs checkpoint)

## 2026-07-17 night2 -- HEAD 90ed495e (v2.0-alpha) [v4 campaign: P1(ii)+(iii)+(iv) pushed]

Did (coder window, autonomous):
- P1(ii) @ f859036c: named regression pinning PBUG-20260710-07 (the cast-keyed
  mutation class) -- INVARIANT A (every coercion stamps a role_coerce reason
  breadcrumb + meta.role_coercions audit; no silent flip) + INVARIANT B
  (announcer-sentinel / name-excluded lines never coerced). Test-only; NO coerce
  code added (root fix shipped pre-campaign; adding more = shim). PBUG stays
  ROOT-OPEN until a live v4 leg. tests/test_pbug_20260710_07_cast_keyed_mutation.py.
- P1(iii) @ e7ba2627: bank-aware GENRE/spoken-text guard. New nodes/_otr_genre_guard.py
  (casefolded/Unicode boundary matcher: gun !~ begun, +s/es plural, phrase ws-flex;
  + writer-boundary authored repair via creative slot, keep-if-clean, never raises,
  breadcrumb). Deterministic terminal = G10 in run_gap_audit -> Phase-10
  FreezeAssertionError (one path every family crosses; mirrors G9). OPT-IN via
  validated scalar default defaults.genre_guard_spoken (default False -> INERT for
  all 10 current banks; v4 banks flip in Phase 2). Fixed 2 static-audit collisions
  at root (LLM slot tag on the creative_fn call; label=pre in the collect-test).
- P1(iv) @ 90ed495e: beat_bounds structural contract in _otr_episode_budget
  (WORDS_PER_BEAT=40 SOFT/recorded; STRUCTURAL_MIN_BEATS=3; family caps codex 12 /
  inline 40; target_beat_count round-half-up; classify) + deterministic G11 floor
  terminal in run_gap_audit (opt-in via meta.beat_bounds; counts distinct spoken
  beat_ids; raises below floor). Writer stamps meta.beat_bounds. Operator: length
  recorded-not-gated -> only the structural floor gates; MAX + word->beat derivation
  deferred to Phase-2 live. 8031 suite green with the writer stamping every real
  episode = empirical proof the floor never false-fails a shipping lane.
- Each chunk: full suite (7980 -> 8018 -> 8031) + Bible 17 + AST/BOM/zero-byte +
  commit AND push + HEAD==origin verified. No canonical JSON change (no graph edit).
Current step: v4 campaign Phase 1 -- P1(v) outro completeness validator (next), then
  (vi) header<->scene, (vii) placeholder token, (viii) provenance normalizer; then
  Phase 2 (5 v4 banks, each a live GPU leg).
Next: P1(v) bounded authored per-line outro patch (Python only canonicalizes an
  already-present unambiguous alias; never appends prose; seed from episode seed).
Commits: f859036c, e7ba2627, 90ed495e

## 2026-07-17 night -- HEAD c3a9d420 (v2.0-alpha) [v4 campaign: Phase 0 done + P1(i) pushed]

Did:
- Phase 0: root-caused PBUG-20260710-07 STATICALLY -- the D3 pre-freeze coerce
  sweep (_otr_freeze_cascade.py:1367 -> production_ledger.coerce_speaker_role_for_char_id)
  resolves the announcer<->char_id ambiguity via cast_ids (announcer-named slots
  excluded; the "Chandra c02" mis-stamp is a real character, correctly coerced).
  Already closed by sentinel char_id mint + name exclusion + the role_coerce
  compose_flags breadcrumb; pinned by tests/test_d3_role_coercion.py (14/14). NO
  coerce code change -- adding one is a shim (operator directive). Durable v4
  protection = per-lane "announcer lines carry the sentinel char_id" minting
  invariant, enforced in Phase 2; a live v4 leg formally retires the PBUG (kept
  ROOT-OPEN in PROD_BUG_LOG until then). Exact-id/sidecar audit + nine-defect
  disposition done. Defect #2 (name-splice) stays OPEN per the timebox.
- P1(i) @ c3a9d420: validated scalar bank defaults (style_pool_class,
  require_science_floor, propagate_adaptation_cast) added to _parse_bank; deleted
  the strict_v4_banks set + the (shakespeare_v3,public_domain_story_v3) tuple + the
  media/adaptation literal branches in select_style. Writer stamps
  meta.style_pool_class from bank.defaults; select_style reads meta (hash keys
  UNCHANGED -> byte-identical slugs, C7); science-floor + adaptation-cast consumers
  read bank.defaults directly. Migrated all 10 runnable banks.json rows.
  tests/test_bank_scalar_defaults.py (new, 27) + updated test_style_catalog.py.
  Full suite 7974 passed / 32 skipped / 1 xfailed; Bug Bible 17; AST/JSON/BOM PASS.
  Visual-STYLE pool axis is separate from the source FEED (science_rss vs
  media_archive_rss); scifi_fable2 keeps the science_rss feed but no science floor
  (matches prior). base_source_bank_id retained (bakeoff logic) -- only its use in
  the 3 consumers removed.
Current step: v4 campaign Phase 1 -- P1(ii) breadcrumb regression + reason stamp.
Next: P1(ii) -> P1(iii) genre/spoken-text -> (iv) beat_bounds -> (v) outro -> (vi)
  header<->scene -> (vii) placeholder -> (viii) provenance (each its own green pushed
  chunk); then Phase 2 (5 v4 banks, each a live GPU leg). Operator decisions defaulted
  (vetoable at the consuming chunk): WORDS_PER_BEAT=40 (soft; length recorded-not-gated),
  media_archive_v4 OWN drama_seeds, public_domain research_only BLOCKS publish.
Commits: c3a9d420

## 2026-07-17 evening -- HEAD 659ce5b2 (v2.0-alpha) [v4 campaign: full kibitz arc r1-r4 CONVERGED; final.md plan of record; NO code yet]

Did:
- Ran the LESSONS GATE (PRODUCTION_SPRINT_LESSONS incl. lesson 24 + PROD_BUG_LOG + Bug Bible)
  and mapped the live seams for the 5 lanes -> docs/2026-07-17-v4-campaign/LESSONS_GATE_BRIEF.md.
- Ran the FULL kibitz arc r1-r4 (operator routing: Codex @ gpt-5.6-sol + agy @ Gemini 3.1 Pro
  (High); Claude anchor+judge; $0 local). agy model corrected to "Gemini 3.1 Pro (High)" (3.5 Pro
  is not an installed slug). Every folded panel claim grounded CONFIRMED against real Windows files
  (5 grounding subagents). Artifacts: docs/2026-07-17-v4-campaign/{pass00,r1_plan,r2_plan,r3_plan,
  final}.md + r{1..4}_judgment.md + roundtable/r{1..4}_claude_anchor.md + kibitz-runs/2026-07-17-v4-campaign/.
- Converged design of record = final.md. Key grounded corrections vs the naive plan: a `_v4` id
  silently drops out of style pool / science floor / adaptation-cast (:4286) / sidecars -> each v4
  re-owns via validated scalar bank defaults (style_pool_class, require_science_floor,
  propagate_adaptation_cast); wiring mirrors v3 (shared legacy_many_pass_v4 for the 3 inline lanes,
  original_multi_pass_v4 + scifi_codex_circuit_v4 executable:true); genre banned_phrases does NOT
  gate spoken text today -> new boundary-aware spoken-text validator (writer-boundary repair +
  Phase-10 FreezeAssertionError scan); beat_bounds terminal = raise (no STORY_META output); outro
  missing name = bounded authored patch (no forced coordinate); text_for_tts already FIXED (dropped);
  weapons_smoking is an EXISTING lexicon-corroborated hard class (retain+author to pass, no new filter);
  A/B "strictly better" = POST-BUILD qualification (may be cloud), ship gate = green+live.
- Plan is Phase 0 (audit + PBUG-20260710-07 breadcrumb root-fix + verifies) -> Phase 1 (8 shared
  fixes, each green pushed chunk, canary per execution family) -> Phase 2 (5 v4 banks serialized,
  atomic per-bank chunk). 11-item VERIFY-AT-BUILD checklist in final.md.
Current step: v4 campaign -- ARC DONE; awaiting operator GO to start Phase 0 (first code).
Next: Phase 0 audit + breadcrumb root-hunt; then Phase 1 shared fixes; then the 5 v4 banks.
  Open operator decisions surfaced in final.md: WORDS_PER_BEAT constant, media_archive_v4 sidecar
  own-vs-share, whether public_domain research_only blocks publish.
Commits: none (docs only; campaign docs under gitignored docs/2026-07-17-v4-campaign/ + kibitz-runs/).

## 2026-07-17 afternoon -- HEAD 499386aa (v2.0-alpha) [roster trim -> 10 INDEPENDENT lanes + science_news family retired; ONE combined commit]

Did:
- Executed the operator roster trim as ONE combined commit @ 499386aa. Ripped
  the whole science_news family (v1/v2/v3), ALL _v2 lanes, orphan bases
  (public_domain_story/shakespeare/scifi_sonnet v1) + original_radio_v3 -> 10
  runnable lanes + custom. banks.json + pipelines.json + 14 pack dirs +
  story_rules + both canonical workflows (widget[23] -> scifi_fable2), all same
  commit. Roster now: media_archive(+_v3), original_radio, scifi_fable2(+_v3),
  scifi_codex(+_v3), public_domain_story_v3, shakespeare_v3, scifi_sonnet_v3.
- Independence (operator "real future-proof, no family dependency"): each kept
  lane resolves its OWN story_rules by EXACT id -- severed base_source_bank_id
  family-map in _otr_story_rules (resolve + coverage), the strict_v4 set, and
  the adaptation-cast classifier. Added 6 _v3 rules packs; renamed 3 orphan
  bases -> _v3; DEFAULT_RULES_ID -> scifi_fable2. Default repoint SPLIT:
  lane-selecting sites -> scifi_fable2; legacy-seam resolvers -> media_archive
  (kibitz r3 build-breaker catch: scifi_fable2 declares no legacy seams).
- Retired dead pipelines sonnet_archive_multipass (base) + original_multi_pass_v3
  and their runner-map / inline-set entries (bijection restored; _run_scifi_sonnet_lane
  kept -- the _v3 wrapper uses it).
- Method: /kibitz r3 (codex, grounded) on the rip PLAN first; ~150 stale
  roster/science-baseline tests repointed via 4 parallel subagents (disjoint
  file groups) + verified centrally. Obsolete science-lane / base-map /
  byte-identity tests removed (intent preserved by repointing to
  media_archive/original_radio where possible).
- Gates: full suite 7947 passed / 32 skipped / 1 xfailed; Bug Bible 17 passed;
  canonical 23 nodes / 57 links (widget value only); no BOM / no 0-byte;
  AST+JSON parse clean; HEAD == origin @ 499386aa.
Current step: v4 improvement campaign (post-rip) -- NOT started.
Next: roundtable R1-R2 (frontier panel + the new Kimi 3) then /kibitz R3-R4 to
  produce v4 for scifi_codex (improve on v1), shakespeare, public_domain,
  media_archive, original_radio; author the v4 lanes as INDEPENDENT banks.
  Parked (task 7): canonical root-fixes (scifi_codex P3 unstated-contract,
  scifi_fable2 SCENE_WORD_GROSS scene-gate, original_radio weapons/X-Y-placeholder/
  phantom-outro) + the shared pipeline-bug class the scoreboard flagged
  (speaker-attribution collapse, name-token splice, contract-vocab bleed,
  720-length knob).
Commits: 499386aa.

## 2026-07-17 morning -- HEAD f265c044 (v2.0-alpha) [variant scoreboard delivered; roster-trim decision -> rip in a fresh window]

Did:
- Ran the full story-only variant sweep (v2/v3 x {420,720}) on the harness. aion
  (OpenRouter) had a ~3-4am HTTP-502 outage that killed ~11 of the 720 legs;
  classified aion-drops vs content-fails and re-ran ONLY the aion drops (hardened
  tmp/_rerun_failed_720.ps1 to never blind-retry a content fail). Final: 420 rung
  COMPLETE; 720 rung 12/16 clean + 4 DISQUALIFIED content-fails (original_radio_v2
  weapons gate, scifi_codex_v2/v3 P3 contract, scifi_fable2_v3 SCENE_WORD_GROSS).
- Grading pipeline: tmp/_extract_for_grading.py + tmp/_assemble_matrix.py ->
  tmp/grading/matrix/*.txt (42/48 cells). ONE Fable pass -> the scoreboard at
  **docs/2026-07-17-variant-scoreboard.md**. fable2 v1 = flagship; order fable2 >>
  public_domain > original_radio > codex > shakespeare > media_archive > sonnet >
  science_news. BIG finding: most defects are PIPELINE bugs, not bank problems --
  speaker-attribution collapse (5/7 cases are _v2 cells), speaker-name splice into
  dialogue, phantom outro characters, contract-vocab bleed, and the 720-length knob
  barely steering. Code fixes that lift every bank.
- OPERATOR ROSTER-TRIM DECISION (task 8): KEEP 11 lanes -- fable2 v1+v3,
  public_domain v3, original_radio v1, shakespeare v3, science_news v3,
  scifi_sonnet v3, media_archive v1+v3, scifi_codex v1+v3. RIP 13 -- all 8 _v2 +
  public_domain v1 + original_radio v3 + shakespeare v1 + science_news v1 +
  scifi_sonnet v1. To be done as a CLEAN rip in a FRESH window (kibitz the plan
  first; canonical source_bank roster in the same commit; suite+Bible+push;
  precedent = codex56sol+gemini rip @ 3312aec7). Sonnet-on-v1 model-check killed
  (deck cleared); re-run it on the 11 kept lanes AFTER the rip.
- Earlier this session: sonnet decoration root-fix (2794e8a2) + story-only scoring
  harness (f265c044), both pushed, suite 7984 + Bible 17 green.
Current step: roster trim (task 8) in a fresh window.
Next: clean 13-lane rip -> Sonnet check on kept lanes -> parked canonical root-fixes (task 7).
Commits: 2794e8a2, f265c044 (pushed). Scoreboard doc uncommitted.

## 2026-07-16 evening -- HEAD f265c044 (v2.0-alpha) [sonnet decoration root-fix + story-only scoring harness; 32-leg variant sweep RUNNING]

Did:
- Root-fixed the scifi_sonnet 320w bake-off FAIL ("ORUM: spoken text contains
  decoration '('"): the spoken-purity contract (`_spoken_error`) was enforced
  ONLY at the terminal `validate_spoken_text_and_lock` raise, so a stray
  parenthetical killed the episode with no bounded repair. Wired it into the
  P2a/P2b (CitedLineV4) + P5 (RewriteResultV4) typed-repair ladder so the model
  fixes its own line (LLM-first); terminal gate stays the deterministic last
  word. Live: scifi_sonnet 320w RESULT SUCCESS + obs asset (recovery_session,
  508w/13 lines). Commit 2794e8a2. Applies to all 3 sonnet versions (shared runner).
- Built the story-only scoring harness (operator: "splice the canonical, use the
  latest"): `OTR_LedgerFreezeCascade.OUTPUT_NODE=True` + `otr_canonical_api_run.py`
  opt-in `--workflow` (default = canonical WITH its path assertion) + wrapper
  `-Workflow` passthrough + `scripts/build_story_only.py` ->
  `workflows/otr_story_only.json` (validator->writer->freeze, 3 nodes / 6 links).
  Skips the ~30 min TTS/video tail; each leg ~12-20 min, produces the frozen
  ledger/transcript we grade from (video carries no cross-bank grading signal).
  Live 30w leg RESULT SUCCESS in 10:37, freeze terminal executes. Commit f265c044.
- Suite 7984 passed / 32 skipped / 1 xfailed + Bible 17 passed after BOTH commits.
- LAUNCHED the 32-leg story-only variant sweep (16 `_v2`/`_v3` lanes x {420,720},
  aion-3.0-mini + Mistral-Nemo) for the v1/v2/v3 comparison. Receipts
  `tmp/_storysweep_receipts.csv`; ~9-12h; hourly scheduled check-in task
  "otr-story-sweep-checkin". Base v1 420/720 transcripts reused from existing
  ledgers (no re-render). 4 full-render `_v2` @420 legs already banked
  (media_archive/original_radio/public_domain_story/science_news).
Current step: 32-leg story-only variant sweep RUNNING (render window).
Next: as legs land, root-fix any failing variant lane per THE LAW (sonnet
  decoration already fixed; watch P3 AuditVerdictV4 / P6 attestation / codex
  premise-cap), then build the 8x3x3 scoring report (v1/v2/v3 per bank at
  420+720) + whittle to the top-8 keepers (best version per bank).
Commits: 2794e8a2 (sonnet fix), f265c044 (story-only harness).

## 2026-07-16 -- HEAD f58ed6e6 (v2.0-alpha) [Qwen3-8B GGUF writer row PROMOTED -- orthogonal model-roster task]

Did (GGUF-row bake-off per `docs/2026-07-16-gguf-row-registry.md`; NOT a forward-order step):
- 3-leg live Qwen3-8B-Q4_K_M bake-off, both writer slots Qwen, ctx=8192 on CUDA:
  3x RESULT SUCCESS + obs asset; peak ~11.8 GB (<14.5); KV 5.60 GB @ 8192 =
  0.70/1k; no silent fallback. Row PROMOTED UNKNOWN->PASS (pinned
  size=5027784512 / sha256=120307ba... / kv=0.70). First GGUF build roster is
  now gemma-4-12b + Qwen3-8B (14B deferred).
- Leg 1 root-fixed 7 Mistral-era assumptions that break a reasoning model:
  `_fetch_science_news` signature; `/no_think` on every gguf call (non-structured
  truncation + json_object `{}`); announcer stop-hygiene + robust dangling-`<think>`
  strip; freeze/shot `load_config` threading (live: a VRAM-eviction cache-miss
  reloaded Qwen NOT gemma); shot-lock re-raise (no silent template);
  `PreAuditReport` null->default (a clean audit's null reason was forcing a
  spurious needs_full_rerun). `/kibitz` (codex) on the `<think>` class per the
  two-strikes law -- it converged + flagged the load_config gap before it cost a leg.
- Full suite **7967** + Bug Bible green. Fail-loud rip honored throughout (operator
  "no local-LM fallbacks"). Docs (gitignored): `docs/2026-07-16-qwen-thinking/`.
Current step: UNCHANGED forward order -- Source-bank bake-off (render window).
Next (operator directive 2026-07-16): complete the **8-bank x 3-leg** bake-off --
  run the remaining legs, ROOT-FIX any failing lane (THE LAW / no-fallback /
  LLM-first: model/prompt/budget-contract fix or explicit lane disqualification,
  NEVER a canned line or blind retry bump), then produce the final 8x3 per-bank
  verdicts + World Cup scoreboard (GO_FORWARD "Then, in order" item 1).
Commits: ee0b2318 (7 fixes), f58ed6e6 (row pinned).

## 2026-07-15 late night -- HEAD 4cd36761 (v2.0-alpha) [plan-stack baseline: every go-forward doc re-grounded]

Did (docs-only session -- no code, no suite run needed; phase C render untouched):
- Read-only fan-out audit (3 grounded agents) of the full plan stack vs HEAD
  4cd36761. Status headers folded into 10 docs -- verdicts: dynamic_story
  CURRENT (rev-5 stands; wiring snapshot still matches live canonical);
  lean-mean-rip NEEDS a bounded re-verify before execution (kill lists + W5
  positional obligation re-verified LIVE and intact; SW-1/SW-3 re-surveys, W6
  keep-list adds, W7 tombstone re-triage, R-7 re-grep -- see its header);
  randomizer-r2 STALE (lane-specs authority absorbed by user-source-lanes;
  24-lane roster; factory-wrapped _v3 runners); vibe-coder-r2 + codex56sol
  telemetry + fable2-s2-QA-r2 + source-banks-v2 SUPERSEDED; llm-first STALE
  with a LIVE remainder (`repair_cliche_span` still rewrites spoken lines +
  `cliche_replacements` in all 8 story_rules JSONs -- X1-X4 queued as a
  quick-win); announcer-framing defect fully OPEN (fix surface untouched in
  code; original_radio_v2 seam is prior art); CLOUD_ENGINE_COVERAGE PARKED
  (babysit harness gone at HEAD; node-83 wiring changed @ 6899d940).
- GO_FORWARD_PLAN lower half REWRITTEN (2026-07-12 sprint table retired):
  telemetry + PBUG-17 items retired (target lane ripped @ 3312aec7), item 8
  re-pointed to user-source-lanes-architecture (~21-31 d, gated on sec-16
  ratification + r5), old item-10 bakeoff removed (superseded by the real
  campaigns), verdict IMPROVE passes + cliche excision + announcer contract +
  ENGINE_MATRIX folded into a quick-wins block, lean-mean added as big block 1
  (order vs extensibility = operator call, recommendation lean-mean first).
  Campaign block + THE LAW + current step preserved as written.
- PROD_BUG_LOG hygiene: duplicate id PBUG-20260713-10 resolved (the
  P1-overlong-question entry renumbered to -21; -10 stays with the P9-audit
  entry). BUG_BIBLE.yaml carries two `legacy_id: -10` rows (~:4357/:4379) --
  reconcile at next fan-out. PBUG-20260712-17 marked SUPERSEDED (its lane was
  ripped; diagnostic-gap class carried by the context/cap quick-win).
- Committed the stranded untracked docs (720 verdict, the 07-13 rip-gates set,
  codex handoff, bakeoff observations, cue-ledger prompt) -- never-lose-work.
- Operator mid-session directives, executed: (1) "nuke it" -> the
  otr-build-tracker artifact is RETIRED (tombstone page pointing at
  HANDOFF_LOG + GO_FORWARD; it had been stale since 06-29). (2) GO_FORWARD
  leaned to TRULY forward-only: campaign shipped-lists, THE LAW done-narrative
  + live-proof table, and the per-lane ladder section stripped (this log +
  PROD_BUG_LOG own them); the "lost anchor" doctrine moved to
  PRODUCTION_SPRINT_LESSONS.md as lesson 24. (3) kibitz r4 confirm pass run on
  the baseline GFP -- panel = codex gpt-5.6-sol (verified via
  codex_model_selected.txt) + agy "Gemini 3.5 Flash (High)", Claude anchor +
  judge; anchor caught 2 must-fixes itself (quick-win-1 reverify vehicle
  overstated: phase C runs only _v2/_v3 lanes, so the base scifi_codex 120w
  reverify needs its own leg or explicit operator acceptance; quick-wins range
  arithmetic understated: ~6-13 d, combined ~33-55 d) -- both folded; panel
  survivors folded per kibitz-runs/2026-07-15-gfp-baseline/r4/final.md.
- ROADMAP swept for the parallel lane; GO_FORWARD gains a Window-packing
  section (RENDER + CODER A-G + PLANNER, one-line otr-handoff kickoffs, credit
  rules) and the lean-mean/extensibility order DISSOLVES on ROADMAP's ratified
  edges: front waves (W0..C1-C5) before extensibility, SW tail (SW1-SW3, C6,
  C7, W8) after extensibility/randomizer/dynamic_story. Combined range now
  ~45-71 coder-days through the tail. Live dashboard artifact rebuilt
  (otr-plan-dashboard: GFP queue + HANDOFF current step + phase-C receipts via
  Desktop Commander), replacing the retired tracker.
- Live observation from receipts (23:19): `scifi_codex_v2` 30w local FAILED at
  P3 -- `RadioScoreDraftV4` ValidationError after 2 attempts -- the exact
  PBUG-20260712-22..25 transport seam awaiting reverify. Campaign window owns
  triage; quick-win 1's reverify just got more interesting.
Current step: UNCHANGED -- phase C 30w smoke sweep (the render window owns it;
  monitor tmp/_phaseC_receipts.csv).
Next: campaign window RE-READS GO_FORWARD before its wrap-up edit (rewritten,
  then leaned, 2026-07-15 late night). Coder queue order per the re-grounded
  queue. NO code lands while phase C is mid-sweep (uniform-code confound).
Commits: b94f0c70 (baseline), 0ed44a3b (lean + kibitz fold), + the
  packing/parallel-lane commit (docs only).

## 2026-07-15 evening -- HEAD b57be02b (v2.0-alpha) [three-phase bake-off campaign: A PASSED, B F2 PROVEN, C smokes LAUNCHED]

Did:
- Confirmed live tip = b57be02b (HEAD==origin), tracked tree clean (only tmp/ +
  docs scratch dirty). Fixed the doc-lag: GO_FORWARD + prior top log entry said
  c28af5f4; live tip is the b57be02b docs-handoff commit atop it.
- PHASE A (Fable final gate on the 8 _v3 promotions + source-snapshot B7/B8):
  PASSED, no build-breakers, nothing folded, tree stays clean. general-purpose
  grounded review = NO build-breakers (all 5 checks file:line grounded); my anchor
  independently confirmed the KeyError class (5 _v3 pipelines defined at
  pipelines.json 566/665/715/824/966 + wired in _RUNNER_BY_PIPELINE/_INLINE_V3_
  PIPELINES; fable2 gate catches _v3; base_source_bank_id maps variants; snapshot
  strict-by-default). Fable UNAVAILABLE (out of usage credits -- failed loud);
  codex CLI unhealthy today (17-min hang + stalled relaunch, killed after ~50min).
  Substitute gate = the two grounded reviews + the live renders themselves.
- PHASE B (F2 live-replay proof): DONE. Captured a real source snapshot for
  original_radio (local spark draw, seeded OTR_ORIGINAL_SEED, sha ed1c941f8e99) ->
  tmp/_phaseB_snapshot_manifest.json; strict loader self-verified for base/_v2/_v3.
  Ran the triplet at 30w local under OTR_C7=1 + manifest. Acceptance met on all 3:
  server log shows source-snapshot REPLAY sha=ed1c941f8e99 + ledger meta
  cast_seed_source == "OTR_CAST_SEED override". RESULT: base GREEN (52.9MB obs
  asset); _v2 AND _v3 both content-FAILED IDENTICALLY on the deterministic
  weapons_smoking gate ("cocking his revolver") -- a clean F2 demonstration that the
  PACK is the only causal variable (same frozen source+seeds, base seam -> clean
  story, v2/v3 seam -> identical weapon content). Lawful under THE LAW (deterministic
  gate). Finding: original_radio _v2/_v3 seam steers to weapons content vs base.
- PHASE C (160-leg bake-off = 16 _v2/_v3 lanes x 5 tiers x 2 profiles): 30w smoke
  sweep (32 legs) LAUNCHED in production mode (no C7/manifest -- verified first leg
  science_news_v2 sources live). Runner tmp/_phaseC_sweep.ps1 (tier-param), receipts
  tmp/_phaseC_receipts.csv, progress tmp/_phaseC_progress.txt, per-leg .done markers.
  ~9 min/30w leg -> smokes ~5h; full 160 legs is a multi-day autonomous run.
- Harness note (follow-up): the launcher's [launch] C7/manifest echoes go to the
  hidden Start-Process console + python's `> %1` truncates, so they do NOT reach the
  server log; the writer's own REPLAY line + cast_seed_source are the ground-truth
  proofs. A one-line launcher/wrapper fix (append, echo the two vars into %1) would
  satisfy the literal-echo acceptance.
Current step: Phase C 30w smoke sweep running (autonomous). After smokes gate:
  120 -> 320 -> 420 -> 720, both profiles; then durable report + World Cup scoreboard.
Next: monitor tmp/_phaseC_receipts.csv; when smokes complete, launch
  `tmp\_phaseC_sweep.ps1 -Tiers 120,320,420,720 -Label full`; content-FAILs
  (weapons/profanity) are RECORDED with reason, never re-rolled to force green.
Commits: docs only (no code fold in Phase A). tmp/ sweep scripts are scratch.

## 2026-07-15 night -- HEAD c28af5f4 (v2.0-alpha) [bank-bakeoff: kibitz r4 CONVERGED + hardened]

Did:
- Ran kibitz r4 convergence on the as-built bake-off (chunks 1/2/4 + B7/B8).
  Panel = Codex @ gpt-5.5 high (rc=0) + Claude anchor; Antigravity FAILED (agy
  rc=1, the known Cowork flake). The skills-cache kibitz.py ignored
  KIBITZ_CODEX_MODEL=gpt-5.6-sol and ran gpt-5.5 (documented drift) -- fine for r4.
- Grounded Codex's review. CONFIRMED one real footgun (MUST-FIX 1): the snapshot
  loader returned None when a manifest was configured but the selected base was
  absent -> silent live sourcing, invalidating the F2 control. FOLDED: source-
  snapshot is now STRICT by default (configured-manifest miss RAISES; opt-in
  "allow_partial": true restores freeze-some/source-rest-live). REJECTED Codex's
  "unconditional raise" (breaks the normal triplet run). Codex MUST-FIX 2 (C7
  proof) -> a LOUD C7-replay warning in code + render-window acceptance criteria
  in GO_FORWARD. Codex OPTIONAL (advisory-key wording) -> doc-only, no code.
- Gates: full suite 7907 passed / 31 skipped / 1 xfailed (+3 r4 tests: strict
  raise, allow_partial, C7 warn/quiet); Bug Bible 17 passed; no BOM; canonical
  delta = none; HEAD==origin. Artifacts under kibitz-runs/2026-07-15-bank-
  bakeoff-r4/r4/ (claude_anchor, codex, final) + docs/.../kibitz/r4-convergence-plan.md.
Current step: Fable final gate (HELD for operator go) + the live replay triplet
  proof (render window).
Next: operator decides on the Fable gate; then the F2 live replay proof under C7.
Commits: 031851ce (B7/B8), 57393879 (docs), c28af5f4 (r4 strict fold)

## 2026-07-15 night -- HEAD 031851ce (v2.0-alpha) [bank-bakeoff: source-snapshot B7/B8 SHIPPED]

Did:
- Built the bake-off frozen-source replay layer (r3 rulings B7/B8). New stdlib
  leaf `nodes/_otr_source_snapshot.py`: a process-wide manifest (env
  `OTR_SOURCE_SNAPSHOT_MANIFEST`) keyed by BASE bank, so one frozen source serves
  the base/_v2/_v3 triplet. `load_snapshot_for_bank` validates the envelope
  (base match via `base_source_bank_id`, seven-key payload presence, non-empty
  seed_source, optional payload_sha256 receipt) and REJECTS base-mismatch /
  malformed / altered-payload loud; returns None when no manifest is configured.
- Wired it into `OTR_LedgerScriptWriter._resolve_inputs` as the FIRST source
  branch, immediately after bank resolution and BEFORE entropy/custom/fetch, so a
  replay bypasses RSS/random; the replayed source_meta carries spark_atoms
  (original) / cast_hints (adaptation) so no downstream owner is starved.
- B8 seed control in `scripts/_otr_soak_server_launch.cmd`: pin
  `OTR_FABLE2_SEED=42` alongside CAST/STYLE under C7 (cleared otherwise) + an
  auditable manifest echo. Dropped an mtime-keyed cache (Windows coarse-mtime
  stale-read hazard) -- the manifest is re-read per episode.
- Gates: full suite 7904 passed / 31 skipped / 1 xfailed (+20 new); Bug Bible 17
  passed; no BOM; py_compile clean; canonical delta = none; dry registry-load 24
  runnable / 25 visible + round-trip 23 nodes/57 links. Pushed; HEAD==origin.
Current step: kibitz r4 convergence + Fable final gate on the v3 promotions + the
  source-snapshot layer (see GO_FORWARD NEXT).
Next: run kibitz r4 (local Codex+Antigravity) then the Fable final gate; then the
  live replay triplet proof in the render window.
Commits: 031851ce

## 2026-07-15 late -- HEAD c32d4c04 (v2.0-alpha) [bank-bakeoff build: chunk 4 SHIPPED + kibitz r2]

Did:
- Ran kibitz r2 on the chunk-4 per-lane matrix (Codex gpt-5.5 high OK; agy lane
  failed -- the known Cowork flake; codex + Claude anchor was the reliable panel).
  Codex DISSOLVED the main risk: I had MISREAD the assemble timing -- codex/sonnet
  DO assemble the ledger IN-runner (led.set_* inside _assemble_ledger), so a v3
  wrapper reads led.data["lines"] uniformly. It also caught the fable2 early
  word-budget gate hard-matching only "fable2_multipass" (a _v3 id would bypass it),
  the runner-map bijection test, and simplified 3 runner files -> ONE wrapper
  factory. Artifacts: docs/.../kibitz/r2-anchor.md + kibitz-runs/2026-07-15-chunk4-
  v3-lanes/r2/{codex.md,final.md}.
- CHUNK 4 SHIPPED @ c32d4c04: pipelines.json +5 clone pipelines; banks.json +8 _v3
  rows (before custom; change default_story_model + default_story_pipeline); 8 v3
  packs (copy v2 + header triple). Writer: run_v3_advisory (deterministic,
  advisory-only, reads assembled ledger, stamps meta["<bank>_v3_advisory"],
  try/except -> never raises, never mutates rows); _make_v3_runner wrapper factory +
  3 sci-fi v3 registrations; _INLINE_V3_PIPELINES + the 2 inline v3 ids in
  _LEGACY_INLINE_PIPELINES; one post-Phase-0 (after :6470 led.save) inline advisory
  hook; fable2 early-gate now family-matches ("fable2_multipass" or "..._v3");
  tooltip de-staled. TestChunk4V3Rows + 2 advisory regressions; pinned tuples
  updated; bijection test validates the wiring.
- Gates: suite 7884 passed / 31 skipped / 1 xfailed; Bug Bible 17 passed; canonical
  delta = none (git diff --exit-code otr_canonical.json clean); no BOM; py_compile.
Current step: source-snapshot injection (B7/B8) -- see GO_FORWARD NEXT.
Next: build the snapshot-envelope load in _resolve_inputs + OTR_C7/OTR_FABLE2_SEED
  controls; then kibitz r4 + Fable final gate + final registry/canonical verify.
Commits: c32d4c04

## 2026-07-15 evening -- HEAD 19872aa6 (v2.0-alpha) [bank-bakeoff build: chunk 2 SHIPPED]

Did:
- CHUNK 2 SHIPPED @ 19872aa6 (pushed, HEAD==origin, no BOM, py_compile clean).
  8 `<bank>_v2` rows inserted before custom_source_bank (mirror base, only
  default_story_model changed; byte-identical banks.json round-trip) + 8 v2 packs
  (base prompt_stages copied, Sec-D target seams edited per pass01 Sec D with
  Section-19 L-1/L-2/L-5/L-6/L-8; header triple = path coords, base pipeline kept).
- B1 owner_bank threading: scifi codex/sonnet/fable2 stamp owner_bank=
  source_bank_row.source_bank_id (never base-mapped); `_assemble` gained an
  owner_bank param. Confirmed the writer stamps meta.source_bank to the SELECTED id
  (:3758) BEFORE runner dispatch (:3853), so scifi_*_v2 pass the authorship gate.
- B5 pinned tuples updated (test_fable2_registry tail + full-order); new
  TestChunk2V2Rows (16 runnable / 17 visible + per-v2 own-pack/base-pipeline).
  test_fable2_assembly direct _assemble calls pass owner_bank.
- F8 resolved on first pass: "EDNA FROST've" is model output, NOT the shared
  _otr_ledger_scrub._normalize_whitespace_and_quotes (which only normalizes
  quotes/whitespace) -> the ALL-CAPS-no-contraction rule lives in media_archive_v2's
  line_composer/exchange seams, not a baseline fix.
- Gates: full suite 7873 passed / 31 skipped / 1 xfailed; Bug Bible 17 passed.
- Grounded CHUNK 4 fully (dispatch/_LEGACY_INLINE_PIPELINES/_resolve_lane_runner/
  telemetry/inline body/authorship) and wrote the per-lane v3 matrix into
  GO_FORWARD_PLAN CURRENT STEP.
Current step: CHUNK 4 (8 v3 lanes: sci-fi own-runner + adaptation/original inline).
Next: build chunk 4 per the GO_FORWARD per-lane matrix; two-strikes -> /kibitz.
Commits: 19872aa6

## 2026-07-15 13:15 PDT -- HEAD 9e0fdf9e (v2.0-alpha) [bank-bakeoff build: chunk 1 + r3]

Did:
- Started the Bank-Improvement Bake-off BUILD (24 rows = 8 base + 8 _v2 + 8 _v3 in
  the one existing source_bank dropdown; zero canonical-JSON diff). Grounded the
  wiring against live HEAD (the tail refactor WriterTailContext/_run_writer_tail/
  TailFinalizer has landed since the r2 anchor -- r2-wiring-anchor.md is stale).
- Ran kibitz r3 (Codex @ gpt-5.6-sol + Antigravity/Gemini-3.5-Flash-High). Judgment:
  docs/2026-07-15-bank-improvement-bakeoff/kibitz/r3-final.md (that folder is
  gitignored -- read from disk). It caught 3 build-breakers.
- CHUNK 1 SHIPPED @ 9e0fdf9e: nodes/_otr_bank_variants.py (base_source_bank_id) +
  5 family-behaviour sites + tests/test_bank_variants.py (32). Suite 7864 green;
  Bug Bible 17 green. Pushed; HEAD==origin.
Key r3 rulings: B1 owner_bank uses the ACTUAL variant id (never base-mapped);
  B2 adaptation v3 stays INLINE not own-runner + D.2 extraction CUT; B5 variant rows
  insert BEFORE custom_source_bank and update the pinned tuples same chunk.
Current step: bakeoff chunk 2 (8 v2 rows + packs + owner_bank fix + pinned-test updates).
Next: build chunk 2 per r3-final.md Sec C.2.
Commits: 9e0fdf9e

## 2026-07-11 -- HEAD 6899d940 (v2.0-alpha) [720-bakeoff C3 coder window]

Did:
- C3 SHIPPED @ 6899d940 (atomic code + canonical JSON + tests): music cue
  manifest + third-bus wiring, per FINAL_HARDENED_PLAN.md. NEW
  nodes/_otr_cue_manifest.py (manifest_version 1; shared parse/fail-loud
  validate; keyed cue_id+batch_index; contiguous-batch + dup + placement gates).
  Node 83 (StableAudioTheme) now emits ONE padded cue batch + manifest (4-tuple
  cue_audio_clips/cue_manifest_json/render_log/done): renders each
  ledger.music[] row (fable2) OR synthesizes opening/closing/interstitial
  (legacy, byte-parity slot seeds); writes each cue wav to the episode audio
  dir; placement mapping so inter_NN never KeyErrors compose_music_prompt.
  SceneSequencer + EpisodeAssembler take music_cue_audio/manifest as a THIRD
  bus (own index, never C2's two-bus check); opening/closing sliced from the
  batch by sample_count (direct slice, no silence-trim) + resampled;
  interstitials inserted inline by anchor_line_id (fable2 only; legacy stays
  unconsumed = pre-C3 parity); MF-H scene_audio->master_mix shift extended to
  music rows.
- Canonical JSON same commit: links 241/242/243 out, 280-283 in (node 83 ->
  nodes 3/7 fanout by name); node-7 opening/closing + node-12 closing_audio kept
  DECLARED/unlinked (BUG-LOCAL-097 slot-drift guard); last_link_id 279 -> 283.
  OTR_WorkflowValidator OK, widget_vector_drift=0, JSON round-trip + link-ref +
  input-name + widgets_values-count audits clean.
- Tests: NEW tests/test_cue_manifest.py (schema/dup/slice/byte-parity); rewrote
  test_stable_audio_theme (4-tuple + fable2 lane) + test_full_workflow_v2_audio_
  wiring (new fanout, 241/242/243 gone); fixed 2 constant-pin regressions caught
  by the known-fail guard (test_audio_determinism_wrap 4-tuple,
  test_google_video_sfx_workflow last_link_id 283).
- Suite 7510/31/1 + Bug Bible 17/7/3 green. HEAD==origin, no BOM/0-byte, AST OK.
- LIVE PROOF (LTX lane, headless :8000): 30w = SUCCESS (frozen_circuitry 62.9
  MB, audio_byte_identical OK, 7 beats covered no gaps); 720w all-visual =
  SUCCESS (ticking_lockdown 123.7 MB, audio_byte_identical OK, 18 beats incl. 2
  music_inter covered, budget OK no gaps, 18:50 render). Byte-parity held on
  both.
Current step: C3 done + live-proofed. NEXT = C4a/C4b (S2 full loop) in an Opus
window. Post-C3 follow-up queued: richer per-cue music-still prompting (separate
chunk, image/video director prompt derivation).
Next: C4a/C4b in an Opus window (do NOT start here).
Commits: 6899d940 (code+JSON+tests). Docs refresh = this commit's follow-up.

## 2026-07-11 -- HEAD 2f335c28 (v2.0-alpha) [720-bakeoff C1/C2 coder window]

Did:
- C1 SHIPPED @ 9949bb6e: durable-field identity in production_ledger --
  _row_identity gates the disk merge so durable render fields (wav/timing)
  copy forward ONLY on unchanged content identity (lines=sha of text,
  music=cue_spec_sha256, clips=render-spec); empty-source -> no gate (skip/
  clear preserves durable per the ownership contract). set_music now carries
  anchor_line_id/placement/target_duration_s + stamps cue_spec_sha256. 5 new
  tests; golden fable2 fixture regenerated. Suite 7468/31/1 + Bible green.
- C2 SHIPPED @ 2f335c28: text_for_tts delivery routing. _otr_readiness
  stamps text_for_tts + source sha + receipt on fable2 voiced lines (canonical
  untouched -- restores the pronunciation the P0 fold switched off). NEW
  _otr_text_delivery resolver (LEGACY passthrough = byte-identical spine;
  CONTENT_OWNED = verified stamp, absent/stale = terminal before gen). Voice
  node routes prep/vector/hash through it. scene_sequencer two-bus surplus+
  shortfall terminal check. 26 new tests incl. science_news byte-parity fixture.
  Suite 7494/31/1 + Bible green.
- C3 wiring kibitz'd (r3, Codex + Claude Code grounded; Antigravity timed out).
  HARDENED spec = docs/2026-07-11-c3-cue-manifest-wiring/FINAL_HARDENED_PLAN.md.
  Surfaced real build-breakers before touching the canonical JSON: legacy
  ledger.music[] is empty (node 83 must synthesize legacy cues; inter_NN
  KeyErrors compose_music_prompt), sentinel lines have no cue_id (use C1's
  anchor_line_id), node-7 input deletion = widget-slot drift (keep declared),
  music must be a 3rd bus, slice by sample_count (no silence-trim) + resample.
Current step: 720-bakeoff C3 (cue manifest + canonical workflow wiring) --
CODE-READY per the hardened spec; canonical-JSON rewire, one atomic commit.
Next: build C3 in a fresh window from FINAL_HARDENED_PLAN.md (re-derive live
literals per the VERIFY-AT-BUILD list); STOP after C3 green+pushed.
Commits: 9949bb6e (C1), 2f335c28 (C2) -- both pushed. C3 docs this commit.

## 2026-07-10 ~14:20 -- HEAD af378aad (v2.0-alpha) [scifi_fable2 S1b coder window, QA fold]

Did:
- External QA analysis (docs/2026-07-10-fable2-s1b-QA-ANALYSIS.md) folded: it
  OVERTURNED the 5C-mutator theory -- real chain = doctor 'skip' clears text ->
  Ledger.save() stale-disk merge resurrects old text -> Phase 10 gap. P0 fixes
  shipped @ af378aad: ownership-aware merge (_MERGE_OWNED_ROW_FIELDS), doctor
  skip stamps tts_skip_reason, 5B/5C lane capability gate
  (_legacy_line_compose_applicable; fable2 pack has no line_composer_system).
  QA regression file tests/test_ledger_merge_ownership.py. Suite 7451/31/1.
- LTX MEDIA PATH GREEN: "The Butterfly's Gambit" published to obs (1787s,
  41.8 MB) -- character lane ltx_audio_in + stills; capability gate fired live;
  freeze passed; canonical no-diff.
Current step: fable2 S2 (full loop, 350w) with the QA runway items folded in:
proof-provenance (doctor/Phase-7 rewrite after proof seal -> text_for_tts),
inter-scene music wiring, caption/credits sentinel alias, HuMo stale guard,
per-scene band allocation (all pinned w/ file:line in the QA analysis doc).
Next: S2 in a fresh coder window; operator eyeball on both fable2 episodes.
Commits: af378aad (+ this docs commit) -- pushed.

## 2026-07-10 ~13:15 -- HEAD 8e3d9228 (v2.0-alpha) [scifi_fable2 S1b coder window]

Did:
- S1b SHIPPED: runner + dispatch + registry flips + 80+ tests @ a24b75c4;
  25-roll live-smoke hardening (kibitz r2/r3/r4 + sonnet/opus fan-out per the
  new kibitz-every-failure directive) @ ff4c226d + 8e3d9228. FIRST GREEN
  EPISODE: "Einstein's Echo" in obs (570s); canonical no-diff + validator OK.
- ROOT-CAUSE fix: reviewer role_mismatch flipped sentinel announcer rows to
  character breadcrumb-lessly (sonnet+opus converged on reviewer.py role
  branch); symmetric guard + breadcrumb + regression tests shipped.
- OPEN BLOCKER: cascade 5C-reroll failure path stamps skip=True on target
  rows when fable2's pack (correctly) lacks line_composer_system -> Phase 10
  needs_full_rerun. LTX media roll (stills+ltx_audio_in via _tmp probe,
  16gb_full + character_visual override) got 25 min deep; blocked on this.
- External-QA brief written per operator: docs/2026-07-10-fable2-s1b-QA-
  PROBLEM-STATEMENT.md (big problems + full downstream landmine audit ask).
Current step: resolve the skip-mutator blocker (QA brief) -> green LTX-lane
fable2 roll -> then fable2 S2 (full loop, 350w).
Next: operator runs the QA brief through the external analyst; fold findings.
Commits: a24b75c4, ff4c226d, 8e3d9228 (+ this docs commit) -- all pushed.

## 2026-07-10 ~08:00 -- HEAD c932880f (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S1a SHIPPED: writer tail (J.5 -> M save) extracted into
  `_run_writer_tail(ctx)` + 17-field WriterTailContext (doc s11 pins);
  moved body verified character-identical vs pre-extraction modulo the 2
  pinned gates (title override precedence + run_story_spine gate, s14/8);
  late _OTRC/_PL imports followed the tail. 11 new tests
  (test_fable2_tail_context.py: ctx contract, no-closure, delegation,
  same-run byte identity, spine gate both ways, title precedence x3,
  refine stash x2). 3 AST pin modules updated to follow the move
  (story_brief_c5a2 fixture, announcer title-regen pin, title scratchpad).
  ROOT-CAUSE find: my byte-identity test leaked production_ledger._CURRENT
  (singleton) -> broke lfc C4 tests downstream; autouse save/restore
  fixture added. Commit `948c5a0a`.
- ONE legacy science_news 30w live smoke on the extracted tail: RESULT
  SUCCESS 555s (baseline band), "Etna's Secret" published to obs (60.7 MB,
  Test-Path confirmed); J.5 regen fired live (title_source=
  llm_post_composition). Ledger scrubbed (paths anonymized, article text
  truncated, all keys/rows kept) -> tests/fixtures/fable2/
  legacy_reference_ledger.json + README. Commit `c932880f`.
- Gates: suite 7332/31/1 + Bug Bible 17/7/3 green at 948c5a0a (+ post-
  fixture full-suite re-run green); BOM/AST/0-byte/HEAD==origin verified.
  Also committed a leftover ENGINE_MATRIX docs hunk from the prior
  session (`5f5820a7`).
Current step: scifi_fable2 S1b -- spine, live (runner P0/P1-one-pitch/
P2b/P3/P6/P7 + P8 audit-only; flip runnable+executable SAME change; doc
s13 S1b test set; 30w live smoke; validator no-diff record).
Next: S1b in a coder window (doc sections 5/8/11/13; re-pin splice lines
in the S1b commit).
Commits: 5f5820a7, 948c5a0a, c932880f (+ this docs commit) -- all pushed.

## 2026-07-10 ~06:45 -- HEAD d7379920 (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S0 SHIPPED (all inert, doc = 2026-07-10-scifi-fable2-architecture.md):
  banks.json row before custom_source_bank + fable2_multipass pipeline row
  (registry-legal slots); 9-seam pack scifi_fable2_v1.json (FORMAT block
  byte-identical script/revision); frame_deck.json 14 cards + 6 stances +
  sidecar registration; detection-only story_rules (empty replacements);
  _otr_fable2_markup.py parser (full defect enum, collected defects, split
  word counters, per-constituent lines); 66 new tests incl. rss-not-spark,
  slot-enum rejection, deck lint, science_news pinned row. Doc s14 pins
  1/5/10 resolved in-doc. science_news untouched; NO workflow diff.
- COMMIT NOTE: my staged S0 files were swept into the freeze-cascade
  window's commit d7379920 mid-session (one bundled commit, pushed). Content
  verified file-by-file; full suite re-certified at that HEAD.
- Gates at HEAD: suite 7321 passed/31 skipped/1 xfailed; Bug Bible 17/7/3;
  BOM/AST/JSON verify clean; HEAD == origin.
Current step: scifi_fable2 S1a -- tail extraction ALONE (writer
_run_writer_tail(ctx) + WriterTailContext, byte-identity pin
test_fable2_tail_context.py, ONE legacy science live smoke, then scrub the
ledger into tests/fixtures/fable2/legacy_reference_ledger.json). Nothing
fable2-visible ships in S1a.
Next: fresh coder window claims the slot, reads doc sections 11+13+14, does
S1a only, then S1b (spine + runnable flip same change).
Commits: none under my own SHA (work rode d7379920); this docs commit.

## 2026-07-10 ~02:45 -- HEAD 636d78cf (v2.0-alpha) [original_radio window]

Did (operator overnight directive: "run two more 420w, analyze, optimize
the original path, prompts not py"):
- 420w night batch, 4 rolls total. PUBLISHED: "Ashes of the Pawn"
  (otr\obs\signal_lost_ashes_of_the_pawn_20260710_014548_..._final.mp4,
  18 min e2e). Roll A died at QA: the confirm judge "proved"
  news_source_framing by quoting the CLEAN intro verbatim -- fixed at
  root (3d32b265: news_source_framing + machine_attribution join
  weapons as lexicon-only kill classes; suite 7153 green then). Roll C
  died HONESTLY: writer armed a climax ("holding his revolver") --
  correct lexicon kill. Roll D died at concept: empty cast name x2
  (archetype "The Stenographer").
- ANALYSIS (leg 1): 239/420 words (thin brief -> thin outline);
  key_terms landed 1/5 (story diverged from concept); intro
  ventriloquized a character quote; ZERO quote-wrapped lines and ZERO
  stage directions at 420w (30w observations did not recur); no audible
  name drift (visual portrait prompt invented "Ferrywoman Edith" --
  eyeball item); outro button landed well.
- OPTIMIZED (prompt/data only, 636d78cf, pack JSON): concept demands
  non-empty CAPS personal names w/ example; script_brief demands
  episode-shape (opening/two turns/closing image) + key_term weaving +
  no-arms menace rule; both intro seams forbid quoting characters.
- NOT re-verified live: the portability coder window claimed the repo
  mid-session (S1 in flight, 9 py files dirty + llm_policy.py
  untracked); full suite red from ITS tree, my lane tests 42/42 green.
  NEXT lane action = one 420w verification roll AFTER the portability
  window settles, then eyeball all published episodes.
Current step: original_radio pre-ship -- operator eyeball (now 2
episodes in obs: page_in_the_tempest 30w, ashes_of_the_pawn 420w) +
one post-tune 420w verification roll.
Next: eyeball; verification roll; source-bank e2e sweep.
Commits: 3d32b265, 636d78cf (+ this docs commit) -- pushed. Suite was
7153 green pre-portability-dirt; Bug Bible 17/7/3.

## 2026-07-10 ~01:30 -- HEAD 1c735c2d + docs (v2.0-alpha)

Did:
- LIVE 30w original_radio OBS smoke: GREEN on roll 6 -- "Page in the
  Tempest" published (otr\obs\...20260710_010652...final.mp4, 48 MB,
  RESULT SUCCESS, 548s). Five real production bugs found+fixed at root
  across the failed rolls, each with tests, suite+bible green, pushed:
  7f459e21 (A2 verbatim grounding: ws-normalized match + typed repair +
  deterministic key_term prune -- the prune FIRED live on a later roll),
  75173fc4 (original_qa evidence bar: hard kills need lexicon
  corroboration or a confirm-pass verbatim quote; discards stamped LOUD),
  a61ab2ed (kill authority per class: weapons/anachronism lexicon-only
  -- a grounded quote proves the line, not the class), 6fdf3f6e (ladder
  logs raw-output head on every failure -- exposed gemma truncation),
  d526c8b7 (creative slot -> nemo in canonical: gemma-4 Q8 cannot hold
  n_ctx 4096 on 16GB, the silent 2048 downgrade truncated concept JSON;
  enforces the standing bake-off rejection), 1c735c2d (epilogue_missing
  deterministically refuted when the outro row exists + slot pins
  retargeted).
- Bug Bible +BUG-11.26 (verbatim-grounding gates) + static tripwire +
  kebab fix, pushed (survival guide @ 1a01037).
- Validator record: OTR_WorkflowValidator OK in the green run (23/55,
  drift=0); the lane itself = NO workflow diff.
Current step: original_radio pre-ship -- smoke + validator gates GREEN;
OPERATOR EYEBALL is the only remaining gate (content notes in
GO_FORWARD section 0: name drift, stage-direction leak, quote-wrapped
lines, sci-fi premise tension).
Next: operator eyeballs the published mp4; then source-bank e2e sweep.
Commits: 7f459e21, 75173fc4, a61ab2ed, 6fdf3f6e, d526c8b7, 1c735c2d
(+ this docs commit) -- all pushed. Operator's own windows added
b288d8b6, bff86af9 (portability docs, benign).

## 2026-07-09 ~night -- HEAD 604ccdd3 (v2.0-alpha)

Did:
- /kibitz r2 (coding plan) on ARCHITECTURE_V4 + INTRO_REWRITE_SPEC:
  anchor-first, Codex auto green, agy auto timed out -> operator pasted
  the manual prompt, its review judged. 3-way convergence; shape A
  locked; synthesis = R2_CODING_PLAN.md. Operator left ("do r3-r4 and
  start coding") -> full autonomy.
- /kibitz r3 (wiring): 5 codex must-fixes verified+folded (seam-accessor
  wall, briefs return shape, dual source_meta restamp, title-regen
  staleness root-cause, QA-before-aggregates order) = R3_WIRING_DELTAS.md.
  /kibitz r4: converged, pins P1-P8 (agy auto dead 3x; codex + anchor).
- BUILT + PUSHED CHUNK A `181506e8` (intro rewrite all banks + title fix;
  c5a2 pin retargeted to the script_text L-opener per its own docstring).
- BUILT + PUSHED CHUNK B `604ccdd3` (the whole original_radio
  SAME-COMMIT set, runnable:true). Mid-build catches fixed at root:
  spark deck needed the routing pack-SIDECAR registration; the
  bank-shape dispatch needed the runnable conjunct (custom keeps its
  pinned LOUD SourceContractMissingError path).
- Suite 7136/31/1 + Bug Bible 16/7/3 green after each chunk; AST/BOM/
  0-byte verify clean; HEAD == origin. No workflow JSON diff.
- Note: `3060fd3a` (portability brief) is the operator's own docs commit
  from his other window -- audited, benign.
Current step: original_radio campaign -- BUILD SHIPPED; remaining gates =
live 30w original_radio smoke + OTR_WorkflowValidator no-diff record +
OPERATOR EYEBALL (queued).
Next: run the live 30w smoke (selective reset first), then eyeball, then
the source-bank end-to-end sweep.
Commits: 181506e8, 604ccdd3 (+ this docs commit) -- all pushed.

## 2026-07-09 ~evening -- HEAD 5a09984c (v2.0-alpha)

Did:
- 5-agent Sonnet QA fan-out on all 4 source-bank routes + ledger contract
  (operator skipped further live smokes). Synthesis:
  docs/2026-07-09-source-route-qa/QA_SYNTHESIS.md (local; dated dirs are
  gitignored).
- FIXED+PUSHED closing-seam bank routing (QA F1) -- coda/announcer
  seams pack-route; PD+Shakespeare coda re-authored to bridge contract;
  title_form_label wired; 30 tests. SHA CORRECTION (codex fan-out catch):
  the CODE+TESTS live in `40535ddc` (the operator's Codex loop committed
  the in-flight tree bundled with its dia hardening); `321bcc9c` on top
  carries only docs (dated doc dirs gitignored). Cite 40535ddc for the
  closing-seam code.
- FIXED+PUSHED 5a09984c: produced-story meta split -- K.5.6 summary pass
  stamps meta["produced_story"]; credits/HUD/treatment/music repointed.
- Seated tencent/hy3:free on the roundtable panel until 2026-07-21
  (62962121) + CLAUDE.md section 8 arc routing (R1 cloud, r2-r4 kibitz).
- original_radio R1 COMPLETE: ARCHITECTURE_V1 + anchor review -> live
  4-model roundtable (GPT-5.6-sol / Gemini-3.1-pro / DeepSeek-v4-pro /
  hy3:free; ~$0.13) -> pass01_judgment.md -> ARCHITECTURE_V2.md. Key
  redesigns: creative front (concept/select/brief) runs INSIDE
  build_original_briefs at D.2 BEFORE structure; v2-plan naming adopted
  (original_multi_pass + original_*_system seams); whole-script
  original_qa gate; disclosure must EXPLICITLY say machine-generated;
  cast pass collapsed; num_characters widget feeds the concept pass.

- R1 pass02 run on ARCHITECTURE_V3 (operator overrides: Hitchcock ironic
  epilogue instead of spoken disclosure; NO era frame / raw timeless
  story; RUNNABLE ON BUILD, no staged flips, no fallbacks, HARD FAILS
  ACCEPTED; north star = max story complexity / max code elegance).
  Panel 4x"no" -> judged -> **ARCHITECTURE_V4.md = BUILD SPINE**. Key:
  the epilogue is the ANNOUNCER OUTRO line (empty news_close_brief
  routes there; outro already knows the produced ending) -- zero new
  passes; disclosure lives in the printed layer (news_used + bank-aware
  HUD label replacing hardcoded "NEWS SEED" + unconditional credits
  line); anachronism defense is prompt-side + lexicon only.

- Local read-only fan-out QA (operator request) on the two shipped chunks:
  Antigravity returned NO blockers/majors; 2 verified MINORs FIXED same
  session (stopword bypass in produced-story cast grounding; off-by-one
  dropping the closing excerpt window at exact cap boundary -- also fixed
  in the older reflection builder it was copied from). Codex CLI not on
  system PATH from this session; operator pasting the brief into Codex
  manually -- its report landed at docs/2026-07-09-source-route-qa/
  local_fanout/codex_review_manual.md and was judged SAME SESSION: one
  real BLOCKER-class bookkeeping catch (the 321bcc9c/40535ddc SHA mixup,
  corrected in these docs); all its code checks CLEARED the current tree.
  Fan-out verdict overall: architecture sound, 3 real minors total, all
  fixed and pushed.

- NEW OPERATOR FEATURE (late): post-composition INTRO REWRITE -- once the
  story is done, rewrite the announcer intro from the PRODUCED first
  scene + cast, spoiler-safe by input starvation (scene-1 rows only).
  Spec: docs/2026-07-09-original-radio/INTRO_REWRITE_SPEC.md (shape A =
  derive ProducedOpenBrief -> existing safe-open composer, anchor lean;
  shape B = new rewrite seam). Runs BEFORE outro compose so the
  tone-echo reads the final intro. Joins kibitz r2 scope.

Current step: original_radio campaign -- R1 CONVERGED (2 passes,
~$0.26 total). Next: /kibitz r2 (coding plan) on
docs/2026-07-09-original-radio/ARCHITECTURE_V4.md + INTRO_REWRITE_SPEC.md,
then r3 wiring, r4 convergence, then build: tests first, SAME-COMMIT
registry set SHIPPING runnable:true, pre-ship gates = suite + Bug Bible
+ mocked pipeline + live 30w smoke + operator eyeball.
Commits: 62962121, (40535ddc co-authored), 321bcc9c, 5a09984c -- all pushed.

## 2026-07-11 -- original_codex56sol constrained implementation claim

Operator authorized non-GPU Chunks A-C/E to begin while the current Sci-Fi Codex
live run remains active. Base and origin were both
`26952a7ea64d61a2178485ac2708e350b52f9b48` on `v2.0-alpha`. Prior-owner dirty
files (`nodes/_otr_scifi_codex.py`, `scripts/otr_run_watcher.ps1`,
`tests/test_scifi_lane_schema_parity.py`, and the cue-ledger prompt) and all live
processes are excluded. Overlapping changes and Chunk D remain gated on operator
release. First action: force-publish the locked fingerprint, comparison, and
wording-corrected coding plan, then implement non-overlapping Chunk A surfaces.
