# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> **LATEST SESSION -- 2026-06-15 (day; LTX verify + MOTION forensic; HEAD `dfd6af4` == origin, NO new commits):**
> GPU-verified the BUG-412 blur fix on a clean box via an isolated 832x480-vs-1472x832 A/B (otr_ltx_motion_smoke,
> goofer/euler_cfg_pp): the FIX clip's detail-energy is ~4x the over-res clip (Laplacian var 86 vs 21) -- the mush
> is GONE, dims ffprobe-confirmed. Then chased the operator's "less MOTION than 6/3" note vs the canonical good clip
> `signal_lost_chilled_hope_20260603_161926/videos/b005.mp4`. **ROOT CAUSE = the BOOMERANG is gone.** b005's ledger
> (commit `59d9179`) records `ltx_loop_via_reverse: true`; measured motion b005-vs-fix-smoke = framediff 2.34 vs 0.72
> and optical-flow 0.061 vs 0.014 (~4x), and b005 is a detectable forward+reverse loop (mirror 0.65 ~= 0) while the
> current `eng_ltx_video.py` has NO loop_via_reverse code (deleted in the `70d379b` cleanbreak from
> `nodes/batch_ltx_render.py` -- BUG-LOCAL-117d). **De-risked the restore:** the 169 decode floor was a 1472x832
> artifact (code note lines 90-97); a 97-frame half-render at 832x480 decodes CLEAN (12s GPU probe), so b005's
> half-render+mirror is viable again at the native canvas. Reproduced b005 EXACTLY -- 97f render + the original
> ffmpeg `split;[b]reverse,trim=start_frame=1;[a][r]concat` -> 193f @ 832x480 boomerang (demo in session outputs).
> NEXT = restore loop_via_reverse in `eng_ltx_video.render_clip` (in-tensor mirror after `images_to_uint8`, env
> `OTR_LTX_LOOP_VIA_REVERSE` default on). **The decode-floor fork is RESOLVED via a 3-model roundtable**
> (GPT-5.5+Gemini-3.1-pro+DeepSeek-v4, $0.10, grounded+converged in 1 pass): boomerang-only source-length
> helper with a hardcoded proven-safe min 97 @ 832x480, LEAVE the global `_ltx_frame_length`/169 floor
> UNTOUCHED, gate `ltx_orbit` OFF (it inherits render_clip), round the half UP + `while 2*src-1<target: src+=8`
> (freeze-shortfall fix), in-tensor `frames[-2::-1]` mirror after images_to_uint8, i2v stays ON. REJECTED:
> canvas-aware global floor, runtime probe, full-render-then-slice (shallow motion). Build-ready spec +
> judgment: `docs/2026-06-15-ltx-boomerang-floor/roundtable/pass01_plan.md` (+ pass01_judgment.md). NEXT =
> implement that spec (suite 4266 + Bug Bible + audio-byte-identical per chunk; no JSON change), then GPU-verify
> a 193f boomerang. Box: LTX server KILLED, GPU at baseline. (Earlier overnight note follows.)
>
> **LATEST SESSION -- 2026-06-15 overnight (autonomous; HEAD `1976842` == origin):** Big multi-thread night,
> all GREEN + PUSHED (suite 4265/0, Bug Bible 16/7/3). **(1) BUG-411 flux lush-tint** fully shipped (3 chunks:
> FluxGuidance 3.5 + still grade/radio tails + bookend seed 4242 + grade on portraits for full flux
> consistency). **(2) BUG-412 LTX-REGR** forensic + RESTORE shipped (`21bfe7a`): the LTX default is now the
> FAST 8-step `distilled` + `euler_cfg_pp` recipe (operator: "make LTX as it was; 30-step too slow"); ksampler
> 30-step stays opt-in via `OTR_LTX_SAMPLER=ksampler`. **(3) Comfy-Cloud engine build = S0 BLOCKED, no
> adapters written** (correct per the hard-stop): `OTR_COMFY_API_KEY` is unset and the Desktop key is
> keychain-encrypted (not extractable); see `docs/2026-06-14-comfy-cloud-image-video/S0_RESULTS.md` for the
> one-line unblock (`setx OTR_COMFY_API_KEY ...`). **(4) All-LTX 120-word soak LAUNCHED + RUNNING** on the
> local 5080 (headless server up on :8000, LTX lane, HuMo OFF) -- forces ltx_video on announcer+music+character
> via the sanctioned profile role_overrides; verdicts land in `scripts/_otr_ltx_allroles_soak_summary.json`,
> server log `scripts/_otr_ltx_soak_server_20260614_234303.log`, episodes under `output/otr/episodes` + obs.
> **OPERATOR MORNING TODO:** read `MORNING_REPORT_2026-06-15.md` (repo root). **BOX STATE:** the soak server
> is RESIDENT on :8000 -- RESET-VERIFY before any other GPU work.
>
> **LATEST SESSION -- 2026-06-14 overnight (autonomous; HEAD `cdc1411` == origin):** **BUG-411 flux BOOKEND
> lush-tint restore IMPLEMENTED + PUSHED** (suite 4264/0, Bug Bible 16/7/3 throughout). Two green chunks:
> **Chunk 1 (`bd1fbb2`) — FluxGuidance node** (the biggest factor): `flux_gen1.py` now wires a `FluxGuidance`
> node between the positive CLIP encode and the KSampler (guidance env `OTR_FLUX_GUIDANCE` default **3.5**;
> ksampler.positive reads the GUIDED conditioning); resolves via `wrapper_bridge` NODE_CLASS_MAPPINGS like any
> core node, fails LOUD→radio floor if ever absent. **Chunk 2 (`cdc1411`) — restored still tails + bookend
> seed**: `_otr_story_brief_helpers.py` adds `IMAGE_GRADE_TAIL` ("anamorphic lens, heavy vignette, muted color
> grade, sharp focus" — the 6/5 `_DEFAULT_STYLE_SUFFIX` delta) + `RADIO_BROADCAST_TAIL` ("35mm film grain,
> broadcast-distressed cinematic aesthetic, centered composition" — 6/5 `_RADIO_PROMPT_SUFFIX`), appended in
> `compose_still_prompt` on the IMAGE STILL path ONLY (after STYLE_TAIL_DEFAULT, before NO_TEXT_CLAUSE — layer
> order + no-text contract intact); the shared STYLE_TAIL_DEFAULT (LTX + character video) + the test-pinned
> `get_open_subject` were deliberately NOT touched. `otr_image_gen_dispatcher.resolve_object_seed` pins the
> radio bookend (`kind=="scene_open"`) to a fixed deterministic seed (env `OTR_RADIO_BOOKEND_SEED` default
> **4242**, the 6/5 value). **NO workflow-JSON change** (FluxGuidance is internal to the engine's in-process
> graph, only env knobs added — `otr_scifi_16gb_full.json` untouched). Forensic + chunk detail:
> `BUG_LOG_2026-06.md` BUG-411. **OPERATOR-GATED NEXT (visual A/B — by design):** restart ComfyUI Desktop,
> render, A/B the bookend vs `output\otr\episodes\signal_lost_melting_glass_pressure_20260605_093330\stills\
> radio_bookend_*.png`; bump `OTR_FLUX_GUIDANCE` 3.5→4.0 if the tint isn't all the way there; optionally extend
> `IMAGE_GRADE_TAIL` to portraits (left scoped-out). No unattended GPU render was run (box-collision risk + the
> A/B is a human judgment). **ALSO STILL OPERATOR-GATED from the prior session:** BUG-408 music A/B + golden
> re-baseline; BUG-409 title + HuMo credits backdrop eyeball. The Wan/LTX-REGR engine thread is unchanged.
>
> **LATEST SESSION -- 2026-06-14 eve (look-QA fix marathon; HEAD `94c2166` == origin):** All GREEN + PUSHED
> (suite 4261/0, Bug Bible 16/7/3 throughout). Shipped FOUR look-QA fixes + ONE forensic:
> **BUG-410 closing CREDITS — CLOSED + operator-verified (flux_still).** Root (runtime-probed): the floor
> renders ~20s of scrolling credits out to 65.7s, but the new silent-composite + §4D-blend pipeline locked the
> deliverable to the master-audio length (mux v<=a gate), cutting the scroll. 4-model roundtable converged
> Option A (intentional SILENT post-roll): composite extends to floor length (`5361266`); §4D scopes
> black-padded so `shortest=1` doesn't re-clamp; mux gate relaxed to `v<=a+OTR_MAX_CREDITS_TAIL_S` (45s,
> fail-loud); credits ride over the HELD LAST clip/still (`229cade`, the 6/5 look — adapts to whatever's last).
> Audio byte-identical. **BUG-409 title card — FIXED (`9e0b658`):** `_title_reveal_progress` resolves the
> reveal in the first ~40% of the window then holds solid (env `OTR_TITLE_REVEAL_FRACTION`); close card stays
> bounded to the main video (no credits overlap). **BUG-408 SA3 music — IMPLEMENTED + TUNED (`3a4f71d`,
> `77e89ff`):** SA3-shaped prompt + real negative + per-cue `seconds_start` within a structural `seconds_total`
> context (latent stays cue-length → determinism unchanged); 6-model roundtable baked the defaults
> **context_s 30→12** (longest cue = one tight phrase so short cues are coherent slices), **cfg 6→7**, refined
> negative; kept dpmpp_3m_sde_gpu@100; all env-overridable. **BUG-411 flux BOOKEND lush-tint — NEW forensic,
> CODER-READY (`94c2166`):** the 6/5 image pipeline (`visual/batch_flux_render.py`) was wholly rewritten into
> `_otr_image_engines/flux_gen1.py` + `meta_brief_image_prompt.py`; model/steps/cfg/sampler identical but the
> rewrite DROPPED **FluxGuidance=3.5** + the cinematic **style suffix** + the radio broadcast-distress suffix +
> bookend seed 4242 (6/5 widgets inspected + confirmed). **NEXT WINDOW = implement BUG-411** (restore those in
> the new pipeline), see section 5 + BUG_LOG_2026-06.md. **OPERATOR-GATED remaining:** restart ComfyUI Desktop
> to load BUG-409 + the SA3 defaults; A/B the music then RE-BASELINE the `test_audio_byte_identical` golden
> (music bytes changed intentionally); eyeball BUG-409 title + the HuMo credits backdrop.
>
> **LATEST SESSION -- 2026-06-14 (autonomous fix+build window; HEAD `772a2eb` == origin):** All GREEN +
> PUSHED. **Three opener/image fixes:** FIX1 `3ef0098` (BUG-403 opener centre black -- flux_still now
> conditions on the beat scene still in render_driver.build_request_from_shot); FIX3 `9b028c8` (BUG-404
> ~1s title card -- overlay_audio_timing before _resolve_title_timing); FIX2 `601c13b` (BUG-405 per-role
> image model -- GROUNDED as config not a code bug: saved OTR_ImageDirector widgets are flux_gen1 x3 and
> lumina/qwen/hidream are unbuilt opt-in stubs; added a LOUD dispatch trace, no JSON change). Diagnostic
> instr removed `5899ec7`. **LUMINA-IMAGE 2.0 BUILT + GPU-PROVEN** (`b54657f`): real render_image (native
> split-file recipe UNETLoader+CLIPLoader[lumina2/Gemma-2]+VAELoader+ModelSamplingAuraFlow->KSampler->
> VAEDecode via wrapper_bridge); weights in C:\ComfyUI-Models; env set User-scope (OTR_ENABLE_LUMINA=1 +
> OTR_LUMINA_CKPT/CLIP/VAE); graduated out of the stub-peer matrix + own test file; headless 5080 smoke =
> 1024 still in ~20s, 1.1MB real image (docs/2026-06-14-closing-credits-freeze/lumina_smoke.png). **BUG-406
> FIXED** (`f0d8663`, REGRESSION from the 06-13 scopes node): OTR_SceneAwareScopes rendered only the
> beats-only total_target_frames, so when the master exceeds the beats the short scopes input clamped the
> §4D 3-input blend (shortest=1) below the master -> OTR_MasterAudioMux clone-held the last frame over the
> closing theme = the FREEZE + missing HUD/scopes treatment; PROVEN by durations (plunging_depths blend
> 32.24s vs master 39.7s). Fix: render_scopes now spans the master-audio length (the node already gets the
> master audio). Suite **4255** / Bug Bible **16** / audio byte-identical **9** GREEN; HEAD==origin on
> v2.0-alpha; only dirty file is the operator's pre-existing CLAUDE.md edit (untouched). **NEXT = operator
> look-QA a fresh real render:** (a) opener centre still (flux_still slot), (b) title card spans the
> head-gap, (c) scopes/HUD run to the END of the closing theme (no freeze), (d) burned SDH captions,
> (e) lumina_image selectable per image slot (GPU-proven). Wan/LTX-REGR/3D forward order (below) unchanged.
> **+ BUG-407 `977801a`:** flux_still + all still/floor families now FILL the 16:9 canvas -- **PORTRAIT is
> HuMo-ONLY** now (audio_driven_face), exactly per operator directive (only HuMo needs portrait). **3D
> GROUNDED (section 5):** the only installed/active 3D is HunyuanWorld-Mirror / WorldMirror 2.0 -- a
> MULTI-VIEW SCENE reconstructor, NOT a single-image object-mesh maker; improved-3D-input is BLOCKED on an
> operator PATH decision (scene-recon multi-view vs install a single-image-object tool) -- the two need
> OPPOSITE inputs, so no prompts drafted yet. The "plaster-of-paris" blob = a single flat image fed to a
> multi-view model. Example obs (pre-fix render, shows the now-fixed freeze + portrait): plunging_depths.
>
> **LATEST SESSION -- 2026-06-14 (DEBUG window; HEAD `973567e` == origin):** Objective triage at clean
> baseline `961d8fc` was fully GREEN (suite 4249 / Bug Bible 16 / tree clean). Per operator: SPLIT the
> bug log -- `BUG_LOG.md` is now the ARCHIVE (reference only); new active log = `BUG_LOG_2026-06.md`
> (epoch `BUG-LOCAL-400+`). Then fixed the first live bug **BUG-LOCAL-400** (`d967c6b`): the writer's 4
> cloud model-slots (`openrouter_slot_a/b` + `comfy_slot_a/b`) failed ComfyUI COMBO validation with the
> lanes ENABLED -- the dropdown builders dropped the `(enable …)` sentinel once a lane was on, but the
> saved workflow stores that sentinel -> node 1 red, ALL outputs ignored (the operator's GUI symptom).
> Robust fix: `_lead_with_sentinel()` makes the sentinel `choices[0]` in EVERY lane state (off stays the
> default; symmetric to the 2026-06-10 lanes-OFF sentinel restore). Suite **4252** / Bug Bible green;
> pushed. Operator restarted Desktop -> BUG-400 CONFIRMED FIXED live (node 1 validated, full episode
> wrote). The restarted run then reached **OTR_VideoDirector** and surfaced **BUG-LOCAL-401** (`9465dbd`):
> `flux_still` was rejected for `music_video_model` because its `roles` tuple omitted `music_visual` --
> fixed by tagging flux_still with all 5 roles (a still is the fast universal pick; needs only
> text_prompt, supplied by every role). Suite **4253** green; pushed. A live look-QA of the published
> episode then found two more: **BUG-LOCAL-402** (`99320ae`, FIXED) -- the §4D procgen blend emitted
> `format=gbrpformat`, so the scope overlay + SDH caption burn fell back to source-copy on EVERY render
> (no burned subtitles, no audio-reactive scopes); suite **4254** green. THEN instrumented the composite +
> title timing (`91e9eff`/`8a517b1`, logging-only), a live 30w smoke, + a 4-model roundtable
> (`e26744e`/`973567e`, ~$0.41) GROUNDED the opener: **403 placement is NOT a bug** -- positioned mode
> places b000 at `[0,9.5s)`; the all-black opener was the BUG-402 casualty (now fixed -> opener shows
> scopes + title). **THREE remaining, converged + grounded** (plan:
> `docs/2026-06-14-opener-still-imagemodel/roundtable/pass01_plan.md`, mirrored in `BUG_LOG_2026-06.md`
> for the coder): **FIX1** (BUG-403-remainder, opener centre still BLACK -- `flux_still.family` not in
> `_SCENE_INIT_FAMILIES`, so it never conditions on the beat scene still; code-only in
> `render_driver.build_request_from_shot`, LOUD if absent); **FIX3** (BUG-404, title window ~1s -- call
> `otr_shot_lock.overlay_audio_timing(led)` before `_resolve_title_timing` in `SignalLostVideoRenderer`);
> **FIX2** (BUG-405, per-role image model ignored -- policy carried flux_gen1, NOT a dispatcher bug; add a
> LOUD dispatch log, re-render to capture the policy, then fix config/registry). **NEXT WINDOW = implement
> FIX1 -> FIX3 -> FIX2** (suite + Bug Bible per chunk; any JSON change in `otr_scifi_16gb_full.json` +
> re-validate; remove the `[BUG-403/404 instr]` logging once all three land). Cuts: NO 2nd music_open line
> (breaks `derive_opening_music_beat`), NO dispatcher rewrite, NO composite-floor title card. The Wan /
> LTX-REGR / 3D forward order (below) is UNCHANGED -- this whole arc was a debug detour.
>
> **LATEST SESSION -- 2026-06-14 eve (planner + live GPU smoke; HEAD `1483e48`, session doc edits UNCOMMITTED):**
> Ran the operator's Wan-vs-LTX **opener eyeball** on the live 5080. DECISIVE RESULT: **Wan i2v 14B DRIFTS off
> the input still** (not fixable by easy input knobs -- cfg2.0+locked prompt still drifts; cfg1.5 collapses to
> abstraction); **LTX 2B v0.9 HOLDS the composition** with subtle motion in all 3 modes (ksampler 30-step /
> distilled / 1216x704 hires). **OPERATOR STEER: do NOT ditch any model** -- keep ALL selectable; make the
> README transparent on what each gives. => LTX = opener DEFAULT; Wan i2v stays selectable (evolving/transform
> b-roll, NOT held-still openers); **LTX-REGR promoted to active** (open Q narrowed to MOTION AMOUNT). Built an
> interactive **`otr-render-bench`** artifact (rate each render + settings grid + sweep planner + export, with
> embedded playable clips). Evidence: `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` +
> `eyeball_frames/COMPARISON_montage.png`. The formal `--acceptance --only wan` (40w) reached leg 1/4 then was
> **ABORTED** for an operator procgen emergency (my headless server held :8000 -> ComfyUI couldn't load his
> procgen; box freed, :8000 released, GPU at baseline; **NO production code or workflow JSON was touched** --
> the engine build is intact). Operator moved to a separate **procgen coding window**. **NEXT (awaiting operator
> confirm):** lock LTX-default + Wan-selectable, then run the LTX `--strength`/sampler/steps/cfg MOTION sweep
> (queue it in the render bench planner); README "what-to-expect-per-model" ticket logged (section 5).
>
> This file holds ONLY work that is still open. Completed work lives in git history, `BUG_LOG.md`,
> and the `otr-build-tracker` artifact -- not here. `docs/VIDEO_BUILD_HANDOFF.md` and the 3D plan
> section 0 are thin pointers to this file. When this doc and any other disagree, THIS doc wins.
>
> **Branch:** `v2.0-alpha`. **HEAD:** see git (do not push unprompted).
> **Last updated:** 2026-06-14 (overnight Wan window + soak), HEAD `e314717` (== origin) -- **wan_ti2v
> 8GB-tier engine BUILT, tested, pushed, and VALIDATED LIVE.** New `eng_wan_ti2v` (5B TI2V, GGUF UnetLoaderGGUF +
> `Wan22ImageToVideoLatent` + the Wan2.2 VAE) after capturing the 5B core node class from a live
> `/object_info` (VERIFY-AT-BUILD). Shares only the pure helpers via the new `wan_shared` mixin
> (wan_i2v refactored onto it, behavior-preserving -- NOT a WanI2VEngine subclass). M8 (Wan2.2-VAE
> fail-closed) + S2 (CAPABILITIES row medium/8000) LANDED -- no longer deferred. Models fetched to
> `C:\ComfyUI-Models` with a sha256+license manifest (`docs/2026-06-14-wan-ti2v/MODEL_MANIFEST.json`;
> GGUF Apache-2.0). Suite **4249 pass** / Bug Bible **16 pass** / audio byte-identical green; HEAD==origin.
> **LIVE validation:** wan_i2v proven (21 clean 14B sampler passes in a real full-episode acceptance leg,
> no VRAM-ceiling breach, post mixin-refactor); wan_ti2v proven (bare-graph 5B smoke PASS -- 25 frames
> decoded via the 2.2 VAE in 35s, ~9 GB peak = the lighter 8GB tier). Eyeball clip:
> `docs/2026-06-14-wan-ti2v/wan_ti2v_5b_smoke.mp4`.
> **REMAINING (operator-gated GPU obligations, NOT autonomously completable -- slow + eyeball):**
> (1) the full-episode multi-leg `--acceptance` sweep GREEN exit is impractically slow -- the
> music_visual=wan leg renders the ENTIRE music bed as Wan video (~21 clips/leg, ~20+min/leg), so the
> 4-leg `--only wan` run cannot finish in a reasonable window; both engines are nonetheless proven to
> render correctly live. (2) the I2V-14B vs TI2V-5B WEBM EYEBALL comparison (real camera motion / still
> preserved / no warp). (3) the M9 CS-3 instrumented sequential-residency proof (partial evidence: the
> mixed humo_1.7B+wan_i2v acceptance leg ran 21 wan clips with no ceiling-breach assertion firing).
> **OVERNIGHT SOAK (this session, code-frozen at e314717):** a randomized 120-word episode soak on the
> DEFAULT lane (16gb_full = LTX announcer/music + humo_1.7B character + flux images + full audio, HuMo
> ON, OS-entropy cast/style) ran **5/5 episodes SUCCESS** (~66-80 min each; hit the 6h wall cap, stopped
> clean). Verdicts: `scripts/_otr_120word_soak_summary.json`; driver `scripts/_otr_120word_soak.py`;
> log `scripts/_otr_120word_soak.log`. This is strong production-stability evidence that the wan_ti2v
> build + the wan_i2v mixin-refactor did NOT regress the production pipeline, AND CS-3-adjacent evidence
> (LTX-heavy + HuMo-heavy sequential residency held across 5 long episodes, no OOM/crash). Episode
> outputs are under the server output tree `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes` + obs
> finals -- NOT yet look-QA'd. **NEXT-SESSION TODO (operator wants: analyze + fix bugs + decide next):**
> (a) look-QA the 5 overnight episodes (audio sync, captions, procgen scopes/credits, character look);
> (b) the wan webm eyeball (14B vs `wan_ti2v_5b_smoke.mp4`); (c) then pick the forward-order move
> (LTX-REGR section 5, OR the formal attended wan acceptance, OR 3D item 5). BOX STATE at handoff: the
> default-lane server is still UP on :8000 (idle, ~1.9 GB) -- RESET-VERIFY before any new headless run.
> **Earlier this same date (procgen window):** HEAD `56469bd`.
> PROCGEN §4C+§4D built (`336fb41`/`39aa6c9`), §4C/§4D removed from this doc (build log at
> `docs/2026-06-13-crt-procgen-improvements/PROCGEN_BUILD_LOG.md`), and **§4D WIRED into
> `otr_scifi_16gb_full.json`** (`eb64cd1`: node 94 SceneAwareScopes + 3-input blend + floor
> draw_scopes=False/fps=25 -- it had shipped DORMANT, operator caught it). A 3-subagent wiring panel then
> audited the whole JSON: **NO other dormant features**, all wiring semantically correct; only a stale
> tooltip fixed (`c8c6d4d`). Sweep gained `--exclude-engine` exact-match (`ca10b63`). Standing rules
> hardened: **workflow-source-of-truth** is now a HARD RULE here (§2) + in CLAUDE.md, and CLAUDE.md got a
> true Cowork-operating-model rewrite (`5e4babd`/`56469bd`). Suite green throughout; HEAD==origin.
> Only a live full-episode render eyeball remains for procgen (NOT a forward-order blocker).
> **OPERATOR STEER: the non-Wan soak is ENOUGH** (the non-lip-sync floors are fine for 8GB, not the
> target). **LTX motion regression vs the 5/30-6/5 era = LTX-REGR (§5), AFTER Wan.**
> **NEXT WINDOW = ONE thread: get Wan 2.2 bug-free (sections 1 + 4 + 4A), then the forward order (§3).**)
>
> **BOX STATE (2026-06-14):** the non-Wan coverage sweep + ComfyUI server are STOPPED; GPU is idle
> (~1.3 GB, nothing on :8000). Still RESET-VERIFY before booting fresh (section 4 of CLAUDE.md): kill any
> stray server/harness by CommandLine (CIM) -- NOT a blanket `Stop-Process -Name python` -- and confirm
> the GPU baseline + :8000 empty.
>
> **Hardening delta (2026-06-13):** the Wan Phase-2 + GATE-A coverage-sweep plan was
> QA'd against the real code and roundtabled (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4,
> ~$0.31). 9 grounded must-fixes + 10 should-fixes folded into section 4A; CS-3
> reframed (sections 4 + 5). Full judgment: `docs/2026-06-13-goforward-wan-hardening/`.

---

## 1. CURRENT STEP

**>>> CURRENT STEP (2026-06-14 overnight) = BUG-411 flux lush-tint restore (bookend + ALL flux) — CODE DONE, awaiting operator A/B.**
IMPLEMENTED + PUSHED (HEAD `1eb5c78`, suite 4265/0, Bug Bible 16/7/3). **Chunk 3 (`1eb5c78`)** extended the
cinematic grade to PORTRAITS too (operator: "keep ALL flux consistent") — grounded that `flux_still`
Ken-Burns-animates a pre-minted flux PNG, so every flux PNG (portrait + scene still + bookend) now carries the
same grade and a flux_still beat standing in for HuMo matches the 6/5 look. The three dropped look levers are
restored in the new pipeline: (1) **FluxGuidance node @ 3.5** in `flux_gen1.py` (env `OTR_FLUX_GUIDANCE`,
chunk `bd1fbb2`) — wired positive-CLIP→guidance→KSampler; (2) **cinematic grade tail** `IMAGE_GRADE_TAIL`
+ **radio broadcast-distress tail** `RADIO_BROADCAST_TAIL` appended in `compose_still_prompt` on the IMAGE
STILL path only (chunk `cdc1411`); (3) **deterministic bookend seed 4242** pinned for `kind=="scene_open"`
in `resolve_object_seed` (env `OTR_RADIO_BOOKEND_SEED`). Scoped so the shared LTX/character-video style tail
and the test-pinned open-subject are untouched; NO workflow-JSON change.
- ACCEPTANCE (operator, GPU/eyeball — by design): restart ComfyUI Desktop, render, A/B the bookend vs
  `output\otr\episodes\signal_lost_melting_glass_pressure_20260605_093330\stills\radio_bookend_*.png` (dim
  amber+cyan rim light, 35mm grain, oil-slick oxidized metal). Tuning if needed: `OTR_FLUX_GUIDANCE` 3.5→4.0;
  optionally extend `IMAGE_GRADE_TAIL` to portraits (left scoped-out). No unattended overnight GPU render.
Full forensic + exact strings + 6/5 widget values: `BUG_LOG_2026-06.md` BUG-411 + section 5. The look-QA
fixes BUG-408/409/410 from the prior session are DONE/closed (operator-gated A/B + golden re-baseline remain
for the music). The Wan/LTX-REGR engine thread below stays the parallel ENGINE track.

**2026-06-14 EYEBALL OUTCOME (supersedes the "Wan is the active thread" framing -- pending operator
confirm):** the live Wan-vs-LTX opener smoke showed **Wan i2v 14B DRIFTS off the input still (not
fixable by easy input knobs); LTX HOLDS the composition with subtle motion.** => recommend Wan i2v
14B -> **BACK-BURNER for the music/announcer opener role**, LTX stays the opener engine, and
**LTX-REGR (section 5) becomes the active thread**. A formal `--acceptance --only wan` (40w) is still
running for the technical gate + the 5B in-episode clips. Evidence: `docs/2026-06-14-wan-ti2v/
EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. The prior Wan-lane status below stays
accurate for the ENGINE BUILD (both engines are built + technically validated).

**Active thread (prior framing) = Wan 2.2 lane -- BOTH engines BUILT + VALIDATED LIVE. Remaining work is operator-gated.**
Phase 1 (I2V-14B b-roll) + the GATE-A sweep hardening (M1-M7 + shoulds) shipped earlier. THIS overnight
window built the deferred `wan_ti2v` 8GB engine (HEAD `bcbe05a`): the 5B TI2V on core Comfy nodes
(`UnetLoaderGGUF` Q5_K_M + `ModelSamplingSD3` shift 5.0 + `Wan22ImageToVideoLatent` + the Wan2.2 VAE),
the `wan_shared` mixin (pure helpers shared with a behavior-preserving wan_i2v refactor), M8 VAE
fail-closed + S2 CAPABILITIES row, 22 new unit tests. Suite 4249 / Bug Bible 16 / audio byte-identical
green; HEAD==origin. **LIVE: wan_i2v rendered 21 clean 14B sampler passes in a real full-episode
acceptance leg (post-refactor, no ceiling breach); wan_ti2v rendered a bare-graph 5B clip (25 frames via
the 2.2 VAE, 35s, ~9 GB peak) -- PASS.**

**NEXT (operator-gated -- the autonomous engine work is DONE):**
1. **Eyeball gate** -- compare the I2V-14B vs TI2V-5B clips (`docs/2026-06-12-ltx23-motion/wan_clips/`
   for 14B; `docs/2026-06-14-wan-ti2v/wan_ti2v_5b_smoke.mp4` for the 5B). Bar = real camera motion,
   still preserved, no warp. **S3 risk:** the wired 14B is a SINGLE low-noise expert; if motion is too
   subtle, Path B (two-expert HIGH/LOW handoff) is the mitigation.
2. **Full-episode `--acceptance` sweep** is impractically slow for a multi-leg autonomous run -- the
   music_visual=wan leg renders the WHOLE music bed as Wan video (~21 clips/leg). Run it attended/selectively
   if a formal GREEN exit is wanted; the substantive per-engine validation is already done. The exact
   invocation is in section 4A (S8). To make it tractable, consider an `--only`-by-slot run or a
   shorter-music profile for the wan legs.
3. **M9 CS-3** instrumented sequential-residency proof (mixed Wan+HuMo) -- partial evidence in hand
   (the mixed leg ran without a ceiling-breach); a clean per-beat-peak + reclaim-drain capture remains.

Then proceed down the forward order (section 3): LTX-REGR (section 5), then 3D (item 5), then
distribution S3-S6. ONE coder window in the code at a time; serialize via this file.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (operator, hard):** `workflows/otr_scifi_16gb_full.json` IS the
  production workflow. (1) ANY node / wiring / widget change MUST be made IN that file in the SAME
  change as the code -- code that is not wired into this JSON is DORMANT and does nothing (the §4D
  miss, 2026-06-13: node + blend input shipped + tested but unwired -> ran dead in production). After
  editing, re-validate via `OTR_WorkflowValidator` + a JSON round-trip + the link/widget audit.
  (2) EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot (the sandbox mount lags file writes; always
  read/write the Windows path + verify via Desktop Commander).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8) -- not story-spine, not
  story-pipeline, not the broader audio stack, not other ROADMAP items.
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream character-voice
  "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline; determinism
  seed-keyed (per-seed within a render, NOT run-to-run); every in-render fallback LOUD; UTF-8 no BOM;
  SFW; V-12 dep isolation; no new widgets in the static workflow shell (V-11).
- GIT (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green chunk; the
  operator eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no 0-byte / no BOM /
  AST parse on touched .py. prod/`main` is GATED until operator work is done (a `v2.0-alpha-stable`
  tag on `v2.0-alpha` is fine).
- EVERY session updates this doc + the `otr-build-tracker` dashboard (content; keep the gauge+lanes
  styling).
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs must log
  `cast RNG seed=... (OS entropy)`. Do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in sequence)

> **Two tracks, parallel.** Items 1-2 (punch-list audit, latentsync demos) are OPERATOR-GATED
> (look-QA / demo review -- section 5); the ENGINE track (items 3-4, Wan + sweep GREEN) proceeds
> NOW. "In sequence" applies WITHIN a track, not across the operator gate.

1. **Punch list (GATE A).** Captions DONE (node 86 `OTR_CaptionBurn` in `otr_scifi_16gb_full.json`,
   profile resolves `burn_captions=True`). REMAINING: node-level audit of LTX radio-open + procgen
   rolling credits -- baked into the headless path but maybe NOT into the saved JSON; prove a render
   FROM the JSON has them, then operator look-QA.
2. **latentsync-100% + demos (GATE A).** The `OTR_LSYNC_BASE_ENGINE=still_kenburns` fix + the two-demo
   set + the mixed showcase episode.
3. **Wan 2.2 video engine (section 4).** BOTH engines BUILT + validated live (2026-06-14, `bcbe05a`):
   wan_i2v (14B, post mixin-refactor) + the new wan_ti2v (5B/GGUF 8GB tier). REMAINING = the operator
   WEBM EYEBALL (14B vs 5B) + the optional formal full-episode `--acceptance` GREEN exit (slow
   wan-music-bed leg, run attended) + the M9 CS-3 instrumented proof. Code-complete; gates are the
   operator's.
4. **Coverage sweep GREEN (GATE A acceptance).** Re-run the permutation matrix after the soak fixes.
   Matrix (additive, not cross-product): a visual-engine leg-set (varies each of music/announcer/
   other_beats), a writer-LLM leg-set (varies node-1 `creative_writing_model`/`technical_model`), and a
   curated voice-variation leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no
   seed pins). **Wan is a CORE/BLOCKING engine** -- the sweep is NOT green until `wan_i2v` (and
   `wan_ti2v`) pass, so it stays RED until item 3 lands; that is expected. This re-run also answers the
   one open R2 question: whether `humo_1.7B` renders NATIVE char beats at 70w once its enable flag is on
   (the soak floored it only because the flag was off). **GATE-A precondition: harden the
   sweep FIRST (section 4A M1-M4) -- DONE 2026-06-13: the M1-M5 acceptance gate landed
   (`scripts/otr_coverage_sweep.py --acceptance`), so a silent fallback / empty-results
   run / missing VRAM measurement now scores RED, not GREEN.**
   **S6 harness reality:** `otr_coverage_sweep.py` enumerates ONLY the visual-engine
   leg-set today (the dropdown rotation). The writer-LLM leg-set (node-1
   `creative_writing_model`/`technical_model`) and the curated voice-variation leg-set
   are NOT yet wired into a runnable harness -- TODO: point them at a real driver
   (e.g. a `run_combo_matrix.py`) or run them as separate parametrized soak legs.
   "Coverage sweep GREEN" today means the visual-engine set only.

   **SOAK READINESS AUDIT (2026-06-13).** Walked the registry + harness. Conclusion:
   **clear to run a wan_i2v-only soak today** (no wan_ti2v hard prereq for validation).
   Verified live: `wan_i2v` enumerates `ok`/runnable under `16gb_full` (legs
   `music_visual=wan_i2v` + `other_beats_visual=wan_i2v`) -- the old "add wan_i2v to the
   enable-set" note is STALE/resolved. 27 legs enumerate; the only skips are
   `hunyuan3d_talk`/`trellis_talk` (missing cu128 toolchain, expected darks). Wan models
   on disk + `OTR_ENABLE_WAN_I2V=1` env known. **Two limitations to know:**
   (i) `--acceptance` exit is RED-by-construction until `wan_ti2v` is built (M2 requires
   BOTH Wan engines) -- expected; read the per-leg verdicts in `coverage_sweep_summary.json`,
   the wan_i2v leg PASS/FAIL is the meaningful signal.
   (ii) **The M1 no-fallback (CS-1) gate is bound to `--acceptance`** (`forbid_fallback=
   args.acceptance`); the capstone CLI does not expose it. So re-running the NON-Wan
   permutation soak (the set that originally false-greened) WITH the M1 fix active and a
   clean GREEN/RED exit needs either `wan_ti2v` built OR a small **`--strict-fallback`**
   flag that decouples M1 from the Wan-engine requirement (~10 lines; RECOMMENDED, optional
   -- operator's call). Until then: `--acceptance --only wan` exercises M1 on the wan_i2v
   legs (overall RED expected), and a non-acceptance sweep runs but with M1 OFF
   (informational). No half-built code, no missing capability rows beyond the deferred
   `wan_ti2v`, no broken tests (the 2 `test_model_catalog_scan` reds are pre-existing /
   environmental, tracked separately).
5. **3D sprints.** s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the `character_3d` family
   (image-routing must-fixes already landed). Detail in the 3D plan (pointers).
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing phase).

**0-E parallel track:** `ltx_orbit`/`still_parallax`/`mesh_stage` CPU side shipped + all three GPU-green;
Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the `scripts/_otr_0e_gpu_go.txt` GO file.

**Audio parallel track (own window, never blocks video):** the character-voice "whiny" fix (upstream TTS
only; frozen spine untouched). Operator note: may have self-resolved -- verify before scheduling work.

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only (lip-sync stays SEPARATE
on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ wrapper (KJ drags in SageAttention + a numpy<2 pin
this box violates). Phase 1 + the 5 code-gap fixes are DONE (`2fbc2f3`); the full grounded spec is in
that commit + git history of this file.

- **Phase 2 -- 16GB engine leg.** Drive `eng_wan_i2v.render_clip` via the real path
  (`scripts/otr_run_leg.ps1` / `coverage_sweep --only ...`). ASSERT `wan_i2v` is the final_engine in the
  trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <= 14.5 GB + byte-identical audio mux + silent
  mp4 (h264/yuv420p/bt709, fps 25, `has_audio` False). Kill/reset the Phase-1 server first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF (Q6/Q5_K_M) + the wan2.2 VAE into
  `C:\ComfyUI-Models\` (record HF repo + sha256 + license, fail-closed). Define a NEW `wan_ti2v` engine
  (own flag/model/VAE env, registry registration, `_node_candidates` incl. the 5B latent node, loader
  mode, `canonicalize`, profile hook + tests) -- do NOT alias `WanI2VEngine`.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar = real camera motion, still preserved, no warp.
  **S3 motion risk to watch:** the wired I2V-14B fp8 is a SINGLE low-noise expert (the
  two-expert HIGH/LOW MoE handoff, Path B, is NOT wired -- see `eng_wan_i2v` header). If
  the "real camera motion" bar FAILS (motion too subtle / static), the Path B two-expert
  HIGH/LOW handoff is the mitigation, not a knob tweak. Call this out at the eyeball.
- **Risk CS-3 (reframed):** sequential-residency, NOT co-residency -- see section 4A M9
  and the section-5 CS-3 entry. The supervised Wan batch proves the inter-beat reclaim,
  it does not "decide if they co-stage."

---

## 4A. WAN + GATE-A SWEEP HARDENING (roundtable 2026-06-13, grounded vs HEAD 134f8e2)

Folded from a 3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude's
grounding against the real code. Full judgment + raw reviews:
`docs/2026-06-13-goforward-wan-hardening/`. These gate item 3 (Wan) and item 4 (sweep
GREEN). MUST-FIX -- until M1-M4 land, a GREEN sweep is meaningless:

> **STATUS 2026-06-13 (autonomous build) -- LANDING LEDGER:**
> - **M1 + M4** `9b2294b` -- no-runtime-fallback gate + VRAM fail-closed (12 tests).
> - **M2 + M3 + M5** `0ab55bc` -- sweep `--acceptance`: empty/required-engine exit
>   code + Wan enable-flag / OTR_TEST_MODE / --exclude preflight (17 tests).
> - **M6** `ec91a3c` -- `assert_usable` preflights UNET + umt5 CLIP + VAE (8 tests).
> - **M7** `f71edaa` -- render_clip ffprobe-PROVES the silent-clip contract (13 tests).
> - **S1 + S5** `dfe9ab5` -- wan_i2v vram_estimate 14500 + real wan2.2-i2v asset id.
> - **S7 + S10** `f3a529f` -- per-shot/seed init staging + Pillow-required fail-loud.
> - **S3 / S6 / S8** -- folded into this doc (MoE eyeball risk, sweep-harness reality,
>   the exact acceptance invocation below).
>
> **M8 + S2 -- LANDED 2026-06-14 (`bcbe05a`).** The `wan_ti2v` engine is built: its 5B core
> node class (`Wan22ImageToVideoLatent`) was captured from a live `/object_info` first; M8 raises
> `EngineUnusable` when the resolved VAE basename is empty or is the 2.1 VAE; S2 added the
> `medium`/8000 CAPABILITIES row (registry-consistency invariant holds -- the row + the registered
> engine landed together). Validated live (5B bare-graph smoke PASS). **STILL OPEN:** **M9** (CS-3
> sequential residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are live-GPU
> proof obligations -- partial evidence only. A full multi-leg `--acceptance` GREEN exit is gated on
> the slow wan-music-bed leg (run it attended/selectively), not on missing code.
>
> **S8 -- exact acceptance invocation** (ComfyUI venv python; live server on :8000;
> `OTR_TEST_MODE` UNSET; `OTR_ENABLE_WAN_I2V=1` (+ `OTR_ENABLE_WAN_TI2V=1` once built);
> Wan UNET + umt5 CLIP + VAE on disk):
> `python scripts\otr_coverage_sweep.py --acceptance --only wan`
> (`--only wan` matches the `sweep_<slot>_wan_i2v` / `_wan_ti2v` legs; drop `--only`
> for the full visual set. `--exclude` of a core Wan engine is REFUSED in acceptance.)

- **M1 -- the sweep is BLIND to silent fallback.** `otr_coverage_sweep.py` runs every
  leg with `expect_engine=""`, which `_otr_soak_capstone.py:464` treats as
  informational (no assert), so a leg that silently falls back to `still_kenburns`
  scores PASS (this is exactly CS-1). FIX (NOT per-leg `expect_engine=engine` -- that
  false-fails a slot that gets 0 beats at 30w): in acceptance mode assert ZERO runtime
  fallbacks across the whole trace -- fail any shot where `final_engine != attempts[0]`
  -- with an opt-out only for known-degrade experiment legs. (Verify the trace field is
  a stable requested-id, not an alias.)
- **M2 -- the sweep returns GREEN on EMPTY results.** `return 0 if passed ==
  len(results)` makes `0 == 0` pass when `--only`/`--exclude` filter everything out or
  `wan_ti2v` is unregistered. FIX: fail on empty results; for GATE-A, fail unless BOTH
  `wan_i2v` AND `wan_ti2v` are present in results with PASS.
- **M3 -- acceptance preflight (closes the R2 trap).** `availability()` is pure
  profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so a gated-off Wan leg enumerates
  "run", `assert_usable` fails it closed, it falls back, and (pre-M1) passes -- the same
  `gated_by_flag` mechanism that floored HuMo-1.7B (commit 5231d31). FIX: the acceptance
  run preflights `OTR_ENABLE_WAN_I2V=1` (+ future `OTR_ENABLE_WAN_TI2V=1`) and the model
  files, and FORBIDS `--exclude` of the core Wan engines.
- **M4 -- the V-3 VRAM gate fails OPEN.** `driver_peak = int(report.get("vram_peak_mb")
  or -1)` then fails only if `> ceiling`, so a missing/0/negative measurement (`-1`)
  PASSES -- the `<=14.5GB` invariant can read GREEN with no measurement. FIX: fail
  closed when `vram_peak_mb` is absent or `<= 0`.
- **M5 -- the Wan render-phase VRAM assert is skipped under `OTR_TEST_MODE`** (`if not
  os.environ.get("OTR_TEST_MODE"): ... assert_peak_within_ceiling`). Phase-2 acceptance
  MUST run with `OTR_TEST_MODE` UNSET; the harness preflight fails if it is set.
- **M6 -- `assert_usable` preflights only the ckpt.** The umt5 CLIP + the VAE are
  required graph loaders. FIX: verify UNET+CLIP+VAE present + matching the sha/license
  manifest before any forward (offline / no-runtime-fetch invariant).
- **M7 -- the Phase-2 clip contract is SELF-DECLARED, not asserted.** `_clip_from_raw`
  hardcodes `has_audio=False`/h264/yuv420p/bt709/fps25 in a dict; the soak only inspects
  the obs final's audio. FIX: ffprobe the emitted silent Wan mp4 (or a real-path test)
  to PROVE those fields before mux.
- **M8 -- `wan_ti2v` VAE fail-closed.** `eng_wan_i2v` defaults the VAE to
  `wan_2.1_vae.safetensors`; the 5B needs the Wan2.2 VAE. Give `wan_ti2v` its own VAE
  env; raise `EngineUnusable` if the resolved VAE basename is empty OR equals the 2.1
  basename. Do NOT inherit `_loader_names()` unchanged.
- **M9 -- CS-3 = sequential residency (see section 5).** Prove per-beat peak <= 14.5GB +
  the inter-beat reclaim drains the prior heavy engine (incl. the retained Wan unet
  patcher) before the next loads; that is the real risk, not co-residency. Unblocks
  Phase-2 scoping.

SHOULD-FIX: **S1** raise `CAPABILITIES["wan_i2v"].vram_estimate_mb` 14000 -> the measured
Phase-2 peak (or 14500); the 14499 smoke figure was WITHOUT `free_after_use`, which is
load-bearing -- document it as mandatory. **S2** add a concrete `wan_ti2v` CAPABILITIES
row (`medium` / ~8000 DRAFT -- the 5B VAE decode may push higher, verify on the 8GB
probe / `["wan2.2-ti2v-5b"]`). **S3** surface the single-expert (low-noise) MoE motion
risk on the eyeball gate -- Path B two-expert HIGH/LOW handoff is the mitigation if the
"real camera motion" bar fails. **S4** sweep leg isolation -- reclaim/restart between
legs that swap heavy engines (one resident server, no teardown -> residue corrupts the
next leg's peak; ties to CS-2 + the CLAUDE.md reset directive). **S5** fix the stale
`["wan2.1-i2v"]` label -> the real Wan2.2 I2V asset id. **S6** point item-4's writer-LLM
+ voice-variation leg-sets at their real harness (run_combo_matrix.py?) or mark TODO --
`otr_coverage_sweep.py` enumerates ONLY the visual-engine set today. **S7** stage the
init image under a shot/seed/uuid name (`otr_wan_init_WxH.png` is fixed -> same-dim
renders overwrite; low risk, driver is sequential). **S8** spell `scripts/otr_coverage_sweep.py`
+ the exact `--only` Wan substring + required env. **S9** Phase-2 post-reset verify
(PID/start-time changed, Sage NOT active, `OTR_TEST_MODE` unset, env visible) before
submitting. **S10** `_materialize_init_image`: require Pillow + fail loud (the no-Pillow
path leans on `WanImageToVideo` cover-resize -- N9 risk).

CUTS (panel consensus -- do NOT over-engineer): no broad VRAM-budget-aware scheduler to
close CS-3 (the reclaim assertion suffices; wait for a measured failure); do NOT subclass
all of `WanI2VEngine` for `wan_ti2v` (share only pure dims/aspect/materialize/canonicalize
helpers; keep loaders + node candidates + graph SEPARATE); keep the GATE-A sweep ADDITIVE,
not a visual x writer x voice cross-product. VERIFY-AT-BUILD: capture TI2V-5B's exact core
node class from `/object_info` before coding (the "5B latent node" is underspecified).

---

## 4B. WAN PHASE 1 -- DONE (pointer)

Phase 1 PROVEN: a real Wan b-roll clip (wan_i2v 14B fp8 in-process, ~14.5 GB; commits `2fbc2f3` +
`8eaf058`). Phase 2 is the ACTIVE next step (section 1); remaining Wan work = sections 4 + 4A. The
overnight-soak companion findings (R1 GPU-proven, R2 harness fix unexercised, R3 landed) live in git +
`scripts/FABLE_SOAK_REVIEW.md`; the not-done remainder (R2 verify) is in section 5.

---

## 5. OPEN TICKETS

- **LOOK-QA BUGS (NEW 2026-06-14 eve — operator look-QA pass; all in `BUG_LOG_2026-06.md`):**
  - **BUG-408 default MUSIC sounds non-musical (SA3).** **IMPLEMENTED 2026-06-14 (`3a4f71d`).** Path B:
    SA3-shaped prompt + real negative + per-cue `seconds_start` within a 30s `seconds_total` context (latent
    stays `dur` → length+determinism unchanged), env-overridable sampler knobs. Suite 4261/0. **OPERATOR-GATED:**
    restart Desktop, A/B listen (tune `OTR_SA3_CFG/STEPS/CONTEXT_S`), then RE-BASELINE the `test_audio_byte_identical`
    golden (intended music-bytes change). Plan: `docs/2026-06-14-sa3-music-improvement/roundtable/pass01_plan.md`.
  - **BUG-409 title card scrambles the WHOLE window** — **FIXED 2026-06-14 (`9e0b658`).** New
    `_title_reveal_progress` resolves the reveal in the first ~40% of the window then holds solid (env
    `OTR_TITLE_REVEAL_FRACTION`); close card stays bounded to the main video (no credits overlap). Suite 4259/0.
  - **BUG-410 closing ROLLING CREDITS** — **CLOSED 2026-06-14 (operator-verified on flux_still).** Credits
    scroll over the held last clip to the end again (silent after the theme). Detail in `BUG_LOG_2026-06.md`
    + `docs/2026-06-14-credits-tail-fix/`. (HuMo backdrop not yet eyeballed — low risk, engine-agnostic path.)
  - **BUG-411 flux BOOKEND / image lost its "lush" cinematic tint (NEXT — HANDOFF FOCUS).** The 6/5 image
    pipeline (`visual/batch_flux_render.py` + `flux_prompt_extractor`) was WHOLLY REWRITTEN into
    `_otr_image_engines/flux_gen1.py` + `otr_meta_brief_image_prompt.py` (pure insertions after `e4cb3ac`).
    Model/steps/cfg/sampler IDENTICAL (flux1-dev-fp8, 20, 1.0, euler/simple), but the rewrite DROPPED the look
    levers: **(1) FluxGuidance = 3.5** (flux_gen1 has NO FluxGuidance node — biggest factor), **(2) the
    cinematic style suffix** `"cinematic, 35mm film, anamorphic lens, volumetric lighting, heavy vignette,
    muted color grade, sharp focus"`, **(3) the radio broadcast-distress suffix** + retrofuturistic radio
    fallback (`35mm film grain ... dim amber and cyan rim lighting`), **(4) bookend seed 4242**, **(5) portrait
    style line**. 6/5 workflow widgets inspected + confirmed (no other hidden hardcodes). FIX = restore those in
    the new pipeline (FluxGuidance node @ ~3.5 + the suffixes + seed). Full forensic in `BUG_LOG_2026-06.md`
    BUG-411. CODER-READY (the next window's task).
  These are GATE-A look-QA items (operator-gated track), parallel to the engine forward order — NOT a
  reordering of section 3.

- **IMPROVED 3D INPUT -- BLOCKED on a PATH DECISION (operator look-QA 2026-06-14; GROUNDED this session).**
  The 3D rotating output looked like a "blobby plaster-of-paris" block. GROUNDING (checked logs + disk):
  the ONLY 3D system actually installed/active is **HunyuanWorld-Mirror / WorldMirror 2.0**
  (`custom_nodes/ComfyUI-HunyuanWorld-Mirror`, model `C:\ComfyUI-Models\WorldMirror-V2\HY-WorldMirror-2.0`)
  -- NOT Blender, NOT OTR's deferred character_3d/TripoSG. Recent episode ledgers used NO 3D engine; the
  server log only shows HWM loading (no episode rendered 3D). **WorldMirror is a MULTI-VIEW SCENE
  reconstructor** (image SEQUENCE -> point cloud / Gaussian splat): per its docs 1 frame = "depth/normals
  only"; good 3D needs **8-24 FEATURE-RICH frames, orbital/forward parallax, well-lit, 50-70% overlap**. A
  single flat/low-feature image -> the plaster blob. So the earlier "clean / object-free single image"
  idea is the OPPOSITE of what WorldMirror needs -- object-free helps only single-image-to-OBJECT-mesh
  tools (TripoSG / Hunyuan3D-2 / TRELLIS), which are NOT installed. **OPEN DECISION (operator, next window)
  -- the prompt strategy is opposite per path:** (A) WorldMirror scene/world -> improved input = GENERATE
  an orbit/multi-view sequence + rich-textured scene prompts (NOT a plain bg); or (B) single-image ->
  object mesh -> INSTALL TripoSG/Hunyuan3D-2 + clean isolated-subject prompts. Do NOT draft the improved
  3D prompts (and do NOT wire character_3d) until the path is picked. A roundtable can harden the chosen
  path's prompt set. (Note: the live roundtable launcher stalled this session -- the panel blocked with no
  output; budget a retry or a smaller panel.) Example obs finals to eyeball the EPISODE look (these do NOT
  contain 3D): `output\otr\obs\signal_lost_plunging_depths_20260614_185229_silent_procgen_blended_final.mp4`
  (a pre-fix render -- shows the closing FREEZE + the skinny flux_still portrait, both now fixed at HEAD).
- **HuMo full-frame TEST (operator 2026-06-14 -- future experiment, NOT now).** Operator wants to
  eventually SEE HuMo rendered full-frame (not the 480x832 portrait pillarbox). For now portrait stays
  HuMo's REQUIREMENT -- BUG-407 shipped "full frame everything EXCEPT HuMo". Future: a HuMo full-frame /
  16:9 smoke to evaluate whether the talking-head holds at a wider aspect before changing the default.
- **Look-QA the 5 overnight 120-word episodes (NEW 2026-06-14).** The default-lane soak ran 5/5 SUCCESS
  (LTX + humo_1.7B); the episode outputs (`...\output\otr\episodes` + obs finals) are NOT yet eyeballed.
  Check audio sync, burned captions, procgen scopes/credits, character look. This is the operator's
  "analyze the soak" item; verdicts in `scripts/_otr_120word_soak_summary.json`.
- **Wan WEBM EYEBALL -- DONE 2026-06-14 (operator + Claude live smoke).** RESULT: **Wan i2v 14B
  DRIFTS** -- holds the input still ~1 frame then re-interprets the scene into its own subject (a
  generic tube close-up). NOT fixable by easy input knobs: cfg3.5->2.0 + a locked-tripod prompt STILL
  drifts; cfg1.5 COLLAPSES into incoherent abstraction. **LTX (2B v0.9) HOLDS** the composition with
  subtle motion in all 3 modes tested (ksampler 30-step, distilled, AND 1216x704 hires -- hires
  answers the "low-res" note). => **RECOMMEND: Wan i2v 14B -> BACK-BURNER for the music/announcer
  OPENER role** (keep selectable; revisit only with Path B two-expert handoff, GO_FORWARD 4A S3); LTX
  stays the opener engine; **PROMOTE LTX-REGR (below) to the active thread.** Evidence:
  `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. AWAITS
  operator confirm on the re-prioritization.

- **Non-Wan soak = ENOUGH (operator call 2026-06-13).** The non-Wan permutation coverage sweep
  (`--strict-fallback --exclude wan/latentsync/triposg`) has run sufficiently; do NOT keep grinding it.
  The non-lip-sync FLOORS (`still_kenburns` / `still_parallax` / Ken-Burns / `station_card`) render fine
  and are acceptable for the 8GB tier, but they are NOT the target experience -- the operator wants real
  audio-driven lip-sync, not a still with motion. Focus the remaining runway on **getting the Wan lane
  bug-free** (section 1 + 4 + 4A). A new sweep, if ever needed, should add `--exclude-engine humo` (the
  exact-match flag added `ca10b63`: skips the 14B `humo` that TIMES OUT per CS-4, KEEPS `humo_1.7B`).
- **LTX-REGR (operator 2026-06-13; PROMOTED to active 2026-06-14 pending operator confirm)** -- LTX
  clips no longer animate like the **2026-05-30..06-05** era (motion lost / too static). `BUG-LOCAL-113b`
  (`8115c72`: ksampler 30-step euler cfg3.0 as the LTX default, distilled 8-step = the
  `OTR_LTX_SAMPLER=distilled` rollback) was the prior fix, but the operator STILL sees the regression.
  **2026-06-14 eyeball update:** the Wan-vs-LTX smoke proved LTX HOLDS the still composition cleanly
  (good) -- so the open question is narrowed to **MOTION AMOUNT** (5/30-6/5 read as more dynamic; the
  current ksampler/distilled holds are subtle). With Wan i2v back-burnered for openers, this is the
  recommended NEXT thread. Probe = an LTX **--strength / sampler-mode / step-count / cfg / frame-cap**
  sweep (otr_ltx_motion_smoke.py exposes all of them; --strength is the prime motion lever, 1.0=max
  freedom) against the 5/30 baseline + the 169 decode floor from look-QA round 5.
  **FORENSIC DONE 2026-06-14 (BUG-LOCAL-412, `BUG_LOG_2026-06.md`):** diffed the GOOD 5/09 `l001` + 5/28
  `b001` LTX bookends vs the current engine (ledgers + the DELETED `batch_ltx_render.py` @ `70d379b^` + the
  old workflow JSON widgets). The good recipe = **v0_9 / sampler `euler_cfg_pp` / 8 distilled steps / cfg
  1.0 / 832×480 / I2V strength 0.75 / `loop_via_reverse` boomerang / audio-length**; the cleanbreak
  `70d379b` DELETED that node and `eng_ltx_video.py` shipped **ksampler / `euler` / 30-step / cfg 3.0 /
  768×512-or-1472×832 / strength 1.0 / 169-cap / no boomerang** (the code comment itself admits `euler_cfg_pp`
  is the documented dynamic-motion sampler but the default was left on `euler`). The old WORKFLOW JSON baked
  in NOTHING but seed/method/cap — the recipe lived in code. **ENV-TESTABLE A/B FIRST (no code change):** at
  832×480 set `OTR_LTX_SAMPLER=distilled` + `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`,
  re-render a bookend, A/B vs `l001`/`b001`; if it matches, bake those defaults + the boomerang + audio-length
  back into `eng_ltx_video.py` (coder chunk; no JSON change implicated).
- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep. (Non-Wan -> deprioritized per the operator's "non-Wan soak = enough" call.)
- **CS-2** -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase attribution reads
  ~3 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase peak is a partial answer).
- **CS-3 (reframed 2026-06-13)** -- NOT a co-residency budget: wan_i2v (~14GB) +
  humo_1.7B (~7GB) cannot co-reside under 14.5GB by construction, so they must render
  SEQUENTIALLY. The real proof obligation = per-beat NVML peak <= 14.5GB AND the
  inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`, BUG-291) fully drains the
  prior heavy engine -- incl. the retained Wan unet patcher -- before the next beat
  loads. A mixed Wan+HuMo episode is the test. This UNBLOCKS Phase-2 scoping (no
  open "decision" needed). See section 4A M9.
- **CS-4-open** (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN 14B HuMo lane so it
  fits 14.5 GB. The default char tier is `humo_1.7B` (`955f134`); the 14B is opt-in.
- **R2 verify** -- confirm `humo_1.7B` renders native char beats at 70w with its enable flag ON (the
  soak floored it only via `gated_by_flag`); answered by the item-4 re-run.
- **README "what to expect per video model" (operator 2026-06-14).** Once the opener model bake-off
  settles (interactive render bench artifact `otr-render-bench` + `docs/2026-06-14-wan-ti2v/
  EYEBALL_FINDINGS.md`), add a user-facing "what to expect from each video engine" section to the
  README (newbie audience -- folds into S6/closing): Wan i2v 14B = drifts off the still (b-roll only,
  NOT openers); LTX = holds composition + subtle motion (opener default); TI2V-5B = 8GB tier, lower-res.
  Source the verdicts from the operator's bench ratings (export button).
- **Ship defaults (release)** -- proposed: announcer + character = `flux_still`, music = `visualizer`
  (selectable: station_card, still_parallax, ltx_orbit, abstract). Keep HuMo/latentsync/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish** (minor) -- output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR`
  (fail LOUD on mismatch); run the OH-3 janitor sweep at server boot; widen the heartbeat cadence.
- **OH-4** -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator "go OH-4"
  (`docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`).
- **0-E Phase B** -- tickets E-1..E-7, gated on the sweep GO file; coder-window ready.
- **Operator gates** -- ComfyUI Desktop relaunch (look-QA), fresh-render acceptance, latentsync demos +
  mixed showcase, whiny-voice P0 matrix + reel, S-3D-0 green light, `v2.0-alpha-stable` tag decision.

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
path gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9:
S-3D-0 spike -> T2b keystone GO/NO-GO (timeboxed ~1wk) -> T4 driver + LOOK gate -> W7 production wiring +
soak ("v1-usable") -> S3-S6 distribution. SHORTCUT FORK: S-3D-0 or keystone NO-GO -> `character_3d`
defers (HuMo-2D stays) -> collapses to ~2-3 sprints (0-E + closing). Done splits: "v1-usable" (one
engine, one real episode) vs "B-parity ship" (>=2 engines bind at SHIP).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard: `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Soak review (R1/R2/R3 detail + roundtable): `scripts/FABLE_SOAK_REVIEW.md`.
- Wan/sweep hardening (grounded QA + 3-model roundtable judgment, 2026-06-13):
  `docs/2026-06-13-goforward-wan-hardening/` (pass00 plan+QA, pass01/pass01b raw
  reviews, pass01_judgment.md).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` +
  `otr-sweep-monitor`; digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (forward item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug log (this repo): ACTIVE = `BUG_LOG_2026-06.md` (epoch BUG-LOCAL-400+, started 2026-06-14);
  ARCHIVE = `BUG_LOG.md` (BUG-LOCAL-001..~305, through 2026-06-12, reference only).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale; LTX-AV lane (own plan, gated);
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
