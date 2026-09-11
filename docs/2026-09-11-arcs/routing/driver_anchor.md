# Driver anchor -- 2.4-routing-canvas: ltx_audio_in planning ceiling vs OOM safety

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **CRASH**.

---

## 1. What is TRUE TODAY, grounded against current code

Four sub-claims, four different verdicts, grounded in current code:

(1) ShotLock canvas fallback -- EFFECTIVELY DEAD CODE. otr_shot_lock.py:3442-3444 does use a falsy-only `or` fallback, but every path that can populate policy[\"canvas\"] is gated by ComfyUI-core-enforced widget minimums (canvas_w/h min=16, fps min=1 -- otr_video_director.py:276-289), so the fallback can only fire on a wholly-absent \"canvas\" key, which is an intentional empty-policy path, not a live defect.

(2) ltx_av long-beat clamp -- EFFECTIVELY DEAD CODE / BACKSTOP. eng_ltx_av.py:1291-1295 clamps `length` to `_LTX_AV_MAX_FRAMES` (497, from frame_contract literal at eng_ltx_av.py:1677-1683), but the upstream ShotLock planner (otr_shot_lock.py:_stamp_coverage_plan, 2232) already partitions any beat over 497 frames into legal <=497-frame segments via coverage_plan.partition_beat, and the live render loop (render_driver.py:render_beat_coverage, 4649, called at 5391) dispatches one render_clip call PER SEGMENT with an already-correct per-segment length (segment_render_frames, 1332). The clamp cannot fire under the canonical multi-clip path. `effective_frame_contract` (frame_contract.py:357) confirms ltx_audio_in's 497 ceiling is NOT bypassed -- it is returned unchanged because ltx_audio_in sits outside PLANNING_CAP_ENGINES (frame_contract.py:319, = only \"ltx_8gb\",\"fastwan_8gb\",\"wan_ti2v\"), so partitioning happens at the full 497 boundary, not some smaller silently-narrowed one.

(3) Matrix declared-vs-effective limits -- LIVE, OPEN, and the substantive content of this row. docs/ENGINE_MATRIX.md:79 (auto-generated, drift-checked by tests/test_engine_matrix_doc.py) DECLARES ltx_audio_in's legal window as \"9-497 step 8 | 0.36-19.88s\" -- the full model-legal ceiling. But eng_ltx_av.py's own comment (~1640-1662) states the only VRAM-PROVEN-SAFE rung at its declared 1024x576 canvas is 193 frames/7.72s (\"1024x576x193 measured 7.36 GiB warm\"; \"1280x704 breached the 14.5 GB ceiling at 14,716 MB in the full production pipeline\"), and says outright: \"there is no episode policy cap on this lane at all\"; ltx_audio_in is deliberately absent from PLANNING_CAP_ENGINES so nothing narrows segments toward the proven-safe 193 for VRAM. Coverage-plan's `_ladder_partition` (coverage_plan.py:200) fills each segment \"toward the ceiling in order\" and puts the short segment last, so a long beat CAN legitimately mint a single segment approaching 497 frames -- 2.6x the only measured-safe length -- with nothing refusing it. No PROD_BUG_LOG entry ties an actual live OOM to this specific frame-length gap (the one LTX-AV VRAM entry on file, PBUG-20260616-01, is from 2026-06-16, predates this whole multi-clip build, and was about Gemma-encoder residency, not frame length) -- so this is a code-grounded, UNVERIFIED-as-fired risk, not a proven incident.

(4) wants_talking_prompt capture -- narrow, structurally real, currently dormant. Three independent points decide \"does this role talk\": (a) OTR_VideoDirector._role_talking (otr_video_director.py:588-610) captures once into policy[\"talking\"]; (b) MetaBrief._effective_talking_roles (otr_meta_brief_image_prompt.py:612-631) re-derives via route_freeze's effective-engine resolver and upgrades False->True, applied consistently inside the single consumer derive_image_prompts (call sites at 2758 and 2115 -- one pipeline, correctly wired); (c) render_driver._ia2v_talking_register_active (render_driver.py:1832-1849) re-derives AGAIN, independently, for prompt-TEXT selection, by instantiating LtxAudioInEngine() fresh and reading its recipe live -- eng_ltx_av.py's own _recipe() docstring (~653) says \"Read fresh every call (an operator flips daily<->hero per beat by swapping OTR_LTX_AV_UNET / OTR_LTX_AV_RECIPE)\". Unlike engine-selection, which route_freeze.py guards end-to-end (routing_env_snapshot/snapshots_agree), the recipe axis is NOT captured in that snapshot -- grep confirms OTR_LTX_AV_RECIPE/OTR_LTX_AV_UNET never appear in route_freeze.py. In the canonical single-server-boot pipeline env is static for the whole run, so (b) and (c) read the same value today and this is currently benign; but there is no guard proving that, and no PROD_BUG_LOG entry documents a live mismatch.

**Key files:** `nodes/otr_shot_lock.py`, `nodes/otr_video_director.py`, `nodes/_otr_video_engines/eng_ltx_av.py`, `nodes/_otr_video_engines/frame_contract.py`, `nodes/_otr_video_engines/coverage_plan.py`, `nodes/_otr_video_engines/render_driver.py`, `nodes/otr_meta_brief_image_prompt.py`, `nodes/_otr_shared/route_freeze.py`, `docs/ENGINE_MATRIX.md`, `tools/engine_matrix.py`, `tests/test_multiclip_session_identity_roster.py`, `docs/PROD_BUG_LOG.md`, `docs/GO_FORWARD_PLAN.md`

**Blast radius:** Item 3 (the live fork): changing PLANNING_CAP_ENGINES membership or ltx_audio_in's declared max_frames touches nodes/_otr_video_engines/frame_contract.py (shared, machine-agnostic module) and nodes/_otr_video_engines/eng_ltx_av.py's frame_contract literal -- both are shared code paths per CLAUDE.md 0B, so a narrowing would affect every machine that renders the ltx_audio_in lane (5080 and 4060 alike), not a per-profile/variant-scoped change; it would also change the coverage_plan segment counts docs/ENGINE_MATRIX.md regenerates from (tools/engine_matrix.py, --check gated). Items 1 and 2 (dead code) have zero blast radius if left alone; touching them is a one-line, no-behavior-change cleanup local to otr_shot_lock.py / eng_ltx_av.py. Item 4, if addressed, is scoped to render_driver.py's talking-register probe plus (optionally) route_freeze.py's snapshot keys -- narrow, single-engine.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- "ShotLock write-side canvas validation" implies canvas.w/h can arrive falsy/invalid in production and the `or 832`/`or 480` fallback (otr_shot_lock.py:3442-3444) is live risk. Contradicted: the ONLY writer of policy["canvas"] that reaches ShotLock is OTR_VideoDirector (otr_video_director.py:542), whose canvas_w/canvas_h/fps widgets declare min=16/min=16/min=1 (otr_video_director.py:276-289) -- enforced by ComfyUI core's own prompt validator for both UI and headless/API graph submissions (confirmed at C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py:1020, 'Value {} smaller than min of {}'). So the fallback only ever fires when the whole "canvas" key is ABSENT -- an intentional, documented path for empty/hand-built policies ('tolerates an empty policy in unit fixtures', otr_shot_lock.py ~3420). Not a reachable defect through any current writer.
- "ltx_av long-beat underruns... clamps to _LTX_AV_MAX_FRAMES with only an INFO log" (eng_ltx_av.py:1291-1295) implies this clamp is the operative truncation mechanism for long beats today. Contradicted: a full multi-clip coverage-plan pipeline (chunks 3 / 6c-6d / 7a-7b / W4b-e, landed 2026-07-25 through 2026-07-29) unconditionally partitions any registered engine's beat whose target exceeds its declared max_frames -- otr_shot_lock.py:_stamp_coverage_plan (2232-2335), called for every shot at otr_shot_lock.py:3133 -- into multiple <=497-frame segments via coverage_plan.partition_beat BEFORE render. Each segment is then rendered through its own render_clip call carrying an ALREADY-legal per-segment target_frame_count (render_driver.py:segment_render_frames 1332-1372, wired into the live per-beat loop via render_beat_coverage 4649-4786, called at render_driver.py:5391). ltx_audio_in is explicitly one of 'the two lanes added after the live failure' for this multi-clip session machinery (tests/test_multiclip_session_identity_roster.py:372,394). So the eng_ltx_av.py:1291 clamp is an unreachable defensive backstop under the canonical plan-then-render path, not the live truncation mechanism the row implies.

## 3. The fork -- this is what the round must pressure-test

For ltx_audio_in (and structurally any lane whose declared model-legal max_frames exceeds its lab-measured VRAM-safe rung): should it join PLANNING_CAP_ENGINES with a VRAM-proven ceiling (193 frames/7.72s at 1024x576) so effective_frame_contract narrows every planned segment down to the proven-safe length -- trading more, shorter multi-clip segments (more joins/jump-cuts) per long beat for OOM safety -- or should the model-legal 497 ceiling stay authoritative, per the standing \"no gates on models\" ruling that a guard is legitimate only against a proven silent-wrong-render, until a live leg actually breaches the 14.5 GB cap at a long single segment? eng_ltx_av.py's own comment (~1652-1662) already states this fork almost verbatim and explicitly defers it (\"If a production cap is ever wanted it is a separate planning change: allowlist the lane, then prove the multi-clip partition\") -- it has not been resolved either way. This is the one live, undecided piece of the row; items (1) and (2) are dead code/backstops with no decision to make, and item (4) has at most one defensible fix (add a route_freeze-style consistency check) if the operator wants the dormant gap closed at all.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
