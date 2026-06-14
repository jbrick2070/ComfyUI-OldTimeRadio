# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> This file holds ONLY work that is still open. Completed work lives in git history, `BUG_LOG.md`,
> and the `otr-build-tracker` artifact -- not here. `docs/VIDEO_BUILD_HANDOFF.md` and the 3D plan
> section 0 are thin pointers to this file. When this doc and any other disagree, THIS doc wins.
>
> **Branch:** `v2.0-alpha`. **HEAD:** see git (do not push unprompted).
> **Last updated:** 2026-06-13 (lean cleanup: the soak-fix batch R1/R2/R3 + `--exclude` LANDED and
> were removed from here; the "where we are" history + resolved tickets moved out to git/tracker.
> Only open work remains. Doc-only.)
>
> **Hardening delta (2026-06-13):** the Wan Phase-2 + GATE-A coverage-sweep plan was
> QA'd against the real code and roundtabled (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4,
> ~$0.31). 9 grounded must-fixes + 10 should-fixes folded into section 4A; CS-3
> reframed (sections 4 + 5). Full judgment: `docs/2026-06-13-goforward-wan-hardening/`.

---

## 1. CURRENT STEP

**Active thread = Wan 2.2 Phase 2 (engine leg) -- GATE-A hardening SHIPPED, soak UNBLOCKED.**
Phase 1 (a real I2V-14B b-roll clip) PASSED + the 5 code-gap fixes landed (`2fbc2f3`). The section-4A
GATE-A sweep hardening (M1-M7 + S1/S3/S5/S6/S7/S8/S10) LANDED 2026-06-13 -- HEAD `d30b88f` == origin,
8 commits, 61 new unit tests, Bug Bible + audio byte-identical green throughout; a live in-process
validation against the REAL installed Wan models (13.3 GB fp8 UNET + umt5 CLIP + VAE) PASSED. See the
section-4A landing ledger. The coverage sweep can now DETECT a silent fallback / empty-results run /
missing VRAM measurement -- no more false-GREEN.

**NEXT (no open "decision" -- CS-3 was reframed, see section 5):** drive `eng_wan_i2v.render_clip`
through the real path and assert `wan_i2v` is the final_engine in the trace -- the M1 no-fallback gate
now enforces this. Run it as a wan_i2v-only soak (`coverage_sweep --only` non-acceptance, OR
`--acceptance --only wan` which exercises the leg but reports RED until wan_ti2v exists). Two follow-ons,
not blockers: (1) the `wan_ti2v` 8GB engine -- capture the TI2V-5B core node class from a live
`/object_info` FIRST, then build the engine + its CAPABILITIES row (unblocks M8/S2 + full `--acceptance`
GREEN); (2) the M9 CS-3 sequential-residency proof (the mixed Wan+HuMo batch). Spec = section 4 + 4A.

Soak fixes are DONE (R1 `d33c51f`, R3 `a31fc24`, R2 root-cause `gated_by_flag` + nightly enable-set
`5231d31`, `--exclude` `134f8e2`). The soak RE-RUN is now UNBLOCKED for the wan_i2v leg + the non-Wan
permutations; full `--acceptance` GREEN waits on `wan_ti2v` (RED-by-construction until then, correct).
Blocker audit = forward-order item 4.

ONE coder window in the code at a time; serialize the Wan window vs any other via this file.

---

## 2. HARD RULES (invariants -- apply every session)

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
3. **Wan 2.2 video engine (section 4).** Phase 2 engine leg + the 8GB TI2V-5B engine + the eyeball gate.
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
> **DEFERRED (blocked, not skipped):** **M8 + S2** (wan_ti2v VAE fail-closed + its
> CAPABILITIES row) need the `wan_ti2v` engine, whose 5B core node class must be
> captured from a live `/object_info` first (the registry consistency invariant
> forbids a CAPABILITIES row without a registered engine). **M9** (CS-3 sequential
> residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are
> live-GPU proof obligations for the acceptance run. The acceptance sweep is RED by
> construction until `wan_ti2v` is built and BOTH Wan engines PASS -- correct, not a bug.
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

## 4C. CRT PROCGEN UPGRADE -- title card + gutter scopes (roundtable-hardened, SPRINT-READY)

> Roundtable-hardened 2026-06-13 -- CONVERGED over 4 passes: 3 QA passes (gemini-3.1-pro
> + gpt-5.5 + deepseek-v4-pro) + a 4th convergence pass across 5 FRESH frontier families
> (gpt-5.5 + grok-4.3 + kimi-k2.6 + mistral-large + qwen3-235b) that independently
> re-derived the SAME architecture (validation), leaving only fine implementation
> precision (folded below). Claude judge/grounder; ~$0.48; raw reviews + judgments in
> `docs/2026-06-13-crt-procgen-improvements/roundtable/`. ONE file:
> `nodes/video_engine.py` (`_CRTRenderer` + the `render_video` plumbing). NO new
> node / widget / model / dependency; Pillow-only on CPU; the green-only `screen`
> blend is untouched; the audio spine is frozen (byte-identical); deterministic per
> seed. This window plans; the CODER window builds it.

### Concept -- the signal-strength envelope is the conductor
Re-enable the dormant `_brightness_ema` (disabled v1.5.1) as a DUAL EMA precomputed in
`__init__`: slow `signal` (alpha ~0.05, ambient brightness) + fast `trig` (alpha ~0.3,
lock/glitch triggers); `loss = 1 - signal`. Every dynamic element READS these
(brightness, drift, lock, hierarchy) -- they are NEVER multiplied into the whole frame
(that was the v1.5.1 "dimmed text unreadable" bug). Strong signal = crisp + locked;
weak signal (the silent inter-beat gaps) = the receiver loses lock: the grid dims
first, the scope spokes shorten, faint edge-static creeps, the ident flickers LAST.
The picture loses the signal in the gaps and reacquires on the next cue -- one conceit
that ties the title card (#1, "tune in") to the gutter scopes (#2, "hold the signal").

### #1 -- big-bold EPISODE-TITLE card on the b000 music intro
Decode -> reveal -> POP -> dock. Active window `[music_open_start_f, music_open_end_f +
dock_frames)` with `dock_frames = min(int(fps*0.5), first_dialogue_f - music_open_end_f)`
(never overruns the first dialogue or `total`). Decide the card state BEFORE section 1
draws: while active, SUPPRESS the original ident/subtitle/timestamp ELEMENTS (keep the
section-1 layout + coords -- they are the dock target) and draw the card; after, the
docked state IS the normal section-1 draw. The hero card OWNS the evacuated CENTRE during
its window (EXEMPT from center-column sanctity -- there is no portrait at the open); the
gutter-clamp + sanctity apply to the SCOPES and the DOCKED ident, NOT the intro hero.
- A. carrier-lock: "SIGNAL LOST" decodes from a seeded scramble into the solid terminal
  slab; a broken-phosphor-block carrier meter crawls to solid on the swell (`signal`).
- B. HERO title (big + bold): the actual episode title at 2-3x `f_title`, CENTRE-anchored
  in the evacuated centre (NOT gutter-clamped -- it is the hero), fake-bold by OVERSTRIKE
  with offsets `{(0,0),(1,0),(0,1),(1,1)}` (no real bold font is loaded), decoded-fragment
  reveal stepping on INTEGER frames + a block cursor. Measure with `ImageDraw.textbbox`;
  wrap/scale to `max_w~=int(w*0.8)` / `max_h~=int(h*0.28)` with a min font floor first.
- C. lock POP: a 1-2 frame brightness bloom + a small horizontal coordinate "tear"
  (NOT a hue flash -- green-only blend; NOT `np.roll` -- see specs).
- D. dock (raster collapse): in the tail frames, interpolate the hero bbox down into the
  section-1 ident + subtitle coords. The intro scopes "tune in" (arcs -> full circles)
  synced to the lock.
- Outro bookend (conditional, S5): same logic on `music_close_*` if it resolves.

### #2 -- two asymmetric gutter SCOPES (replace center ring + particles + bottom waveform/bars)
Matched circular form, ASYMMETRIC data + failure (asymmetry is what stops twin rings
reading as wallpaper).
- LEFT `_draw_fft_scope`: 32 radial FFT spokes + per-spoke phosphor comet-tails
  (bounded lookback over `freqs[fi-6:fi+1]`); idle (low signal) -> slow rotating radar
  sweep, phase from `fi/fps` (deterministic).
- RIGHT `_draw_scope`: the `wave` samples traced around the circumference + a bright
  electron SWEEP DOT with a short decaying trail (lookback over `waves`); idle ->
  jittering baseline circle. Absorbs the old bottom waveform.
- Graticules: `_precompute_graticules()` (mirrors the `_scanlines` precompute) ->
  static tick/crosshair RGBA, alpha_composited (near-zero per-frame cost).
- Retire sections 5 (`_waveform_mirror`) + 6 (`_freq_bars_wide`) CALLS; thin section-3
  particles to a faint orbit with brightness ROLES (not hue). Cap all scope line widths
  to 1-2px (the current code is 4px at 1920).

### Sprint plan (one file; each sprint independently testable)
- **S1 -- foundation.** Refactor `_CRTRenderer` to precompute `signal/loss/trig` in
  `__init__` from the full arrays (convert `volume`/`freqs` to `np.array` FIRST -- they
  are Python lists today); resolve the timing dict; `import hashlib` + a local seeded
  RNG; EMA read-only discipline. **Land the draw-order layering here too** (base/scopes/
  grid -> vignette/choke in numpy -> text + card overlay on top): S3 depends on it.
  Regression: determinism + audio untouched.
- **S2 -- gutter scopes.** Left/right scope helpers (bounded-lookback trails) +
  graticules + masked gutter-rect layers; retire sections 5/6, thin section 3.
- **S3 -- title card.** The b000-window state machine (decode->bold->POP->dock) +
  intro-scope tune-in + section-1 suppression; the 2-beat gap-fill smoke.
- **S4 -- envelope behaviors.** The coordinate-offset sync-drift (gutter-clamped) +
  per-element brightness hierarchy. (Signal-driven vignette choke is CUT from v1 -- see CUTS.)
- **S5 -- outro + regression.** Conditional `music_close` bookend (inside `total_frames`)
  + full regression (determinism checksum on RGB frames, audio-byte-identical, no new
  widget, CPU/VRAM unaffected).

### Concrete specs (the wiring -- baked from the QA rounds, do not re-derive)
- **Signature:** `_CRTRenderer(w, h, title, volume, freqs, waves, fps, timing=None)`;
  store `self.total = len(volume)`, `self.fps`; reduce to `render(self, fi)`; update the
  `render_video` caller (current L1556 `renderer = _CRTRenderer(W,H,episode_title)` ->
  pass the arrays + timing; L1559 closure -> `renderer.render(fi)`).
- **Timing extractor:** the SceneSequencer stamps `led["lines"]` with `speaker_role` +
  `start_s` + `dur_s` (persisted to the DISK ledger; resolve it the way
  `otr_caption_burn` does -- the wire ledger may carry `start_s=None`). `music_open` =
  the first line whose `speaker_role` is a music-open role; window =
  `round(start_s*fps) .. round((start_s+dur_s)*fps)`; `first_dialogue_f` = first
  dialogue line's `start_s*fps`; `music_close` = last music line. FALLBACK if `start_s`
  is unavailable: derive the intro window from the `volume` envelope (music from frame 0
  to the first dialogue onset), capped. Missing fields DISABLE that effect (no crash).
- **Intervals:** half-open `start <= fi < end`; clamp to `[0, total)`; disable if None
  or `end <= start`.
- **Determinism:** `import hashlib`; per effect (`salt` = a literal effect tag, e.g.
  "noise"/"scramble") `seed = int.from_bytes(hashlib.blake2s(f"{title}|{fi}|{salt}"
  .encode()).digest()[:8], "big")`; `rng = np.random.default_rng(seed)`; replace the
  section-8 `np.random.randint(-i, i+1, ...)` with `rng.integers(-i, i+1, size=..., dtype=
  np.int16)` (default `endpoint=False` keeps the SAME half-open range -- do not set
  endpoint=True). `signal[0]=trig[0]=volume[0]` (guard empty `volume`).
- **Geometry (COMPUTE from `w,h` -- the renderer also runs 1280x720 / 832x480 / 4K, so do
  NOT bake 1920 constants):** `portrait_w = round(h*480/832)`;
  `gutter = max(0,(w-portrait_w)//2)`; protected centre band = `[gutter, w-gutter]`;
  `left_cx = gutter//2`, `right_cx = w-gutter//2`, `cy = h//2`,
  `r = int(min(gutter*0.36, h*0.30))`. **Clamp the scope amplitude `amp <= r*0.35`** so
  `r+amp <= gutter//2` -- never crosses the centre band, never overflows the edge.
  (At 1920x1080 this gives gutter~=647, left_cx~=323, right_cx~=1596, r~=235.)
- **No `np.roll` on the frame:** it wraps the centre into the portrait. Drift + the lock
  tear = a horizontal coordinate OFFSET applied to the SCOPE draw coords (clamped inside
  the gutter) and, during the title window, to the centre-anchored hero card (clamped
  inside the frame -- no black edge); the steady-state centre grid/background is untouched.
  The POP may be bloom-only if the tear reads risky (2 panels flagged the tear optional).
- **Center-band clip:** draw each scope onto a transparent layer sized to its gutter
  rect and `alpha_composite` it (the layer bounds clip it); never draw scope primitives
  onto the base image.
- **Text exemption:** section-1 ident + the title card draw AFTER the section-8 vignette
  multiply (the S1 layering pass), so the always-on vignette can never dim text.
- **Trails are pure:** computed in `render(fi)` by bounded lookback over the input
  arrays (N~6), NOT mutable per-frame state.
- **Outro:** render only for `fi < total_frames`; leave `_hud_frames` append unchanged.

### v1 CUTS (panel consensus -- add later at the eyeball, not now)
Telemetry micro-text labels (illegible green mush downscaled + clashes with the real
`_TelemetryHUDRenderer`); the FFT peak-hold ghost ring + noise-floor shadow ring
(comet-tails already give persistence); the oscilloscope free-running trigger seam;
halation (a 2x per-frame draw pass threatens the 24fps gen speed); a formal hierarchy
layer-floor system (use per-element brightness scaling -- grid scales down faster than
the ident); the **signal-driven vignette choke** (it multiplies the whole frame -- the
exact v1.5.1 readable-text risk; per-element brightness is enough. If ever re-added:
`np.clip(1.0 - loss*0.6, 0.35, 1.0)` on the immutable base vignette, text-exempt).

### OPEN (operator's call) + VERIFY-AT-BUILD
- **Landscape-beat gutters -- ARCHITECTURE DECISION (grounded 2026-06-13).** Checked vs
  the code: the floor (`render_video`) receives only `audio / ledger / fps / resolution /
  title / closing_audio` -- NOT the clip manifest, NOT a per-beat engine plan;
  `meta.visual_plan` holds `characters/scenes/style` only. The per-beat engine resolves
  LATER (`resolve_engine_for_role` + the profile) and the portrait(HuMo 480x832)-vs-
  landscape(LTX/Wan full-frame) aspect is first knowable at the **clip manifest**
  (`OTR_SilentComposite`, per-beat `engine_id` + `start_s`) and at the **post-upscale
  blend** (`OTR_PostUpscaleProcgenBlend`). So the floor truly cannot be scene-aware.
  DECISION (the procgen floor is NOT relocated -- its floor/gap-fill/credits/green-blend
  roles stay intact; this answers the operator's "don't wire it out" worry):
  - **v1 (this plan):** scopes in the floor, beat-agnostic, dim gutter-clamped edges on
    landscape. Low-risk, ships now.
  - **v2 (scene-aware, the proper fix -- round-robined in §4D):** split ONLY the scopes
    into an ADDITIVE late overlay at the post-upscale stage; it reads the clip manifest
    (`engine_id`->aspect + `start_s` timing) + the per-frame audio analysis (recomputed
    from the master audio, deterministic) and draws the SAME scope module into the REAL
    per-beat gutters (edge/suppressed on landscape), green-only over the upscaled video.
    The TITLE CARD + CRT texture STAY in the floor (time-based, not scene-based). The
    scope module is stage-agnostic, so v1 is the substrate for v2 -- not throwaway.
- **Verify smokes (build-time):** (1) the exact disk-ledger schema on a real episode --
  the `start_s`/`dur_s` fields, that `start_s` is in SECONDS (not 25fps timeline frames),
  and the exact `speaker_role` string for the opening music (likely `music_open` /
  `music_visual`); (2) a 2-beat gap-fill smoke proving the title card stays at the open
  (the timeline-aligned floor slice + frame-aligned blend already imply this); (3) a
  determinism checksum over RGB frames (not mp4 bytes); (4) a long-title `textbbox`
  overflow case; (5) the coordinate-offset bound (no black edge / no center incursion).

---

## 4D. v2 SCENE-AWARE SCOPES -- architecture (round-robin CONVERGED + LOCKED new-node-only)

> Architecture round robin 2026-06-13 (gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro; 3
> passes; Claude judge/grounder; ~$0.54; artifacts in
> `docs/2026-06-13-crt-procgen-improvements/arch-roundtable/`). The §4C landscape-gutter
> OPEN decision, resolved. The procgen FLOOR is NOT relocated -- only the SCOPES split out
> into an ADDITIVE late node, so the floor's gap-fill/credits/green-blend roles stay
> intact (answers the operator's "don't wire the procgen out" worry). Builds AFTER §4C.

### Why a late node (grounded)
The floor (`render_video`) cannot be scene-aware: it receives no manifest and no per-beat
engine plan, and `meta.visual_plan` carries `characters/scenes/style` only. The per-beat
portrait(HuMo)-vs-landscape(LTX/Wan) aspect first exists at the **clip manifest**
(produced by `OTRVideoRenderBatch`, whose `RETURN_NAMES = ("render_report_json",
"clip_manifest_json")` -- CONFIRMED an output) and at the **post-upscale blend**. So the
scene-aware draw must live late, where the manifest is.

### The 4 parts
1. **Circular green-only scope helpers -- FRESH, IN THE NODE MODULE (not a floor refactor).**
   The floor's existing `_waveform_mirror`/`_freq_bars_wide` are rectangular + non-green
   (CRT_CYAN/RED/AMBER) and are LEFT UNTOUCHED (just gated off -- part 4), so there is ZERO
   floor-drawing regression risk. Write the scopes fresh in the node:
   `draw_fft_scope(draw, cx, cy, r, freq_window, env)` /
   `draw_scope(draw, cx, cy, r, wave, env)` -- GREEN-ONLY (CRT_GREEN/CRT_DIM/CRT_DARK only;
   forbid the colored constants), geometry by params (no `self`).
2. **`OTR_SceneAwareScopes` (NEW node) -> `scopes_only.mp4`.** Draws BLACK + green-only
   scopes on black (NO master decode -> no generation loss). Skeleton:
   `CATEGORY="OldTimeRadio/v2/video"`, `FUNCTION="render_scopes"`,
   `RETURN_TYPES=("STRING",)`, `RETURN_NAMES=("scopes_mp4_path",)`; register in
   `NODE_CLASS_MAPPINGS`. INPUT required: `clip_manifest_json` (wired from
   `OTRVideoRenderBatch`); optional: `audio` (analysis-only; absent -> synthesize
   `volume=[0.0]*total`, `freqs=[zeros(32)]*total`, `waves=[zeros(200)]*total`, do NOT call
   `_analyze_audio`), `out_w/out_h` (1920/1080 delivery size), `ffmpeg`. Output:
   `out_w x out_h` @ **25fps HARD-LOCK**, SILENT, yuv420p/CFR. Encoder: mirror the floor's
   rawvideo-stdin -> h264 (nvenc-if-available else libx264) pattern but a SILENT variant --
   the floor's `_encode_mp4` HARD-REQUIRES an `audio_path` (`-i audio` + `-c:a aac`), so the
   node needs its own `-an` no-audio encode (NOT a direct reuse). Match the blend's input
   contract (yuv420p, CFR, 25fps).
3. **`OTR_PostUpscaleProcgenBlend` EXTENDED (3rd optional `scopes_mp4`).** When present,
   build a NEW 3-input filtergraph (do NOT append to the 2-input one -- it converts to
   yuv420p before blend 1): `[0:v]format=gbrp[main]; [1:v]<existing procgen
   scale/crush/zeroRB>format=gbrp[pgn]; [2:v]<scale,fps,setpts,setsar>colorchannelmixer
   (zero R+B)format=gbrp[scp]; [main][pgn]blend=all_mode=screen[tmp]; [tmp][scp]blend=
   all_mode=lighten[out]; [out]format=yuv420p[v]`. **2nd blend = `lighten` (max)** -- two
   `screen`s compound brightness where scopes overlap the procgen. `-map [v] -map 0:a?
   -c:a copy`, keep the existing bt709/CFR flags. Absent -> the current single-blend path.
4. **`OTR_SignalLostVideo` floor: the ONLY floor change = a `draw_scopes` flag.** Add
   `("BOOLEAN", {"default": True})` to the OPTIONAL INPUT_TYPES; thread `render_video ->
   _CRTRenderer -> render()`. False (v2) -> SKIP render() sections {2,3,5,6} as a SET
   (section 3's `orbit_r` reads section 2's `r`, so 2+3 go together; 5+6 are independent);
   keep {1,4,7,8} (title/grid/bottom/CRT). VERIFIED vs the live render(): no undefined-var
   leak when the set is skipped together. Floor's full video + roles UNCHANGED.

### Beat map + eligibility (the scene-aware core)
`segs, total = plan_timeline_segments(manifest, floor_available=True,
target_total_frames=manifest.get("total_target_frames"), fps=25)` -> integer ranges
`[cursor, cursor+n_frames)`; **`total` is the frame count** (no source-video probe). By
`segment["source"]`: `clip`+PORTRAIT -> gutters; `clip`+LANDSCAPE -> suppress; head/inter
`floor`/`black` gap -> DRAW (centre: `cx=out_w//2, cy=out_h//2, r=int(min(out_w*0.16,
out_h*0.30))`) so the signal-lost gaps STAY ALIVE; the TAIL gap (credits post-roll, i.e.
the gap segment(s) at `cursor >= last-beat-end` computed from the beat extents) ->
SUPPRESS (don't cover the credits). Aspect: `ffprobe segment["path"]` (h>w=portrait),
memoized per path; un-probeable -> suppress + log; registry CUT. Gutter geometry from the
delivery size: `portrait_w=round(480*out_w/1472)`, `gutter=(out_w-portrait_w)//2`,
`left_cx=gutter//2`, `right_cx=out_w-gutter//2`, `cy=out_h//2`,
`r=int(min(gutter*0.36, out_h*0.30))`, `amp<=r*0.35`.

### fps + audio
HARD-LOCK 25 across the planner / `_analyze_audio` / scopes encode (the floor defaults 24,
so force `OTR_SignalLostVideo.fps=25` in this workflow OR relax the blend assert: the
24fps procgen is framesync'd by the existing blend, so assert only `scopes==source` frame
count). `_analyze_audio` keeps the EXISTING `sr//fps` chunking (exact at 25fps ->
frame-identical, no spine touch). Deterministic stable-hash RNG (also fixes the floor's
unseeded `np.random`).

### LOCKED: new-node-only (operator decision 2026-06-13)
The floor is NOT rewritten. It keeps emitting its FULL v1 procgen video (CRT + title + the
gap/credits fill) so NOTHING upstream breaks -- it stays the composite's floor/gap/credits +
green-blend base, generated pre-LTX/flux. The ONE floor change = `draw_scopes=False`, which
turns OFF its existing centre ring + waveform + bars -- REQUIRED, because otherwise the old
ring still sits on the face via the blend AND doubles the node's scopes. ALL the new
scene-aware scope VISUAL CODE lives in `OTR_SceneAwareScopes` downstream. Net change = ONE
flag on the floor + ONE new node + a 3rd input on the existing blend. §4C-v1's in-floor
scope placement is SUPERSEDED -- go straight to the scene-aware node (the §4C #2 scope
DESIGN -- two asymmetric circular green scopes -- still stands; it is just drawn in the node).

### v2 sprints + VERIFY-AT-BUILD
S-v2a the fresh circular green scope helpers (in the node module). S-v2b
`OTR_SceneAwareScopes` (silent `-an` encode + `NODE_CLASS_MAPPINGS` registration) + the
planner-driven eligibility/aspect. S-v2c extend the blend (3-input gbrp double-blend,
`lighten`) + the floor `draw_scopes` flag. S-v2d wiring
(`OTRVideoRenderBatch.clip_manifest_json -> SceneAwareScopes -> blend`) + golden-frame
tests. VERIFY: clip `path` is pre-composite with real dims; `lighten` stays visible over
the floor CRT; empty manifest -> fail early; golden frames for portrait / landscape /
head-gap / credits-tail.

---

## 5. OPEN TICKETS

- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep.
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
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale; LTX-AV lane (own plan, gated);
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
