# OTR HANDOFF LOG (append-only; newest at TOP)

The running record of what each session DID. GO_FORWARD_PLAN.md holds the forward plan (lean);
this holds the history so the plan can stay lean. One short entry per session. Deep detail lives
in the per-sprint docs + git; this is a breadcrumb trail, not a dashboard.

---

## 2026-07-03 night -- HEAD e3292324 (v2.0-alpha) -- HAND OFF to a new window for the OVERNIGHT MODEL-MATRIX SOAK
Did: operator called a hand-off. Next window's mission = an autonomous overnight LIVE-GPU soak
of MANY 30-45 word full-pipeline episodes across the model matrix (2000 Comfy credits, all
usable). Pass 1 = COHERENT same-model-across-roles combos (same video + same image model per
episode, NOT mismatched), NEWEST models first; voice indextts2/bark. Later = mixes. Watch
closely, root-cause bug fixes live, suite+Bug Bible+B7 + push green per chunk. Reset the box
(selective CIM kill) before every headless run; load the REAL otr_scifi_16gb_full.json; assets
-> otr/episodes/<ep>/, final -> otr/obs/; Test-Path before success. still_word stays BUILD-READY
(docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md) behind the soak.
Current step: OVERNIGHT MODEL-MATRIX SOAK (GO_FORWARD section 1).
Next: reset box -> boot headless ComfyUI -> Pass-1 coherent-combo legs (newest first) -> monitor + fix.
Commits: none this hand-off (docs baton committed at e3292324 prior).

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
