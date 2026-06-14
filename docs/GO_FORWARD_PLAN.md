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

## 4C. CREATIVE BACKLOG -- Procgen Visual Layer (pre-roundtable)

> **STATUS: design-direction-only, `pre-roundtable`.** Two procgen-layer creative
> ideas from the operator (2026-06-13). NO code work starts yet -- Jeffrey wants to
> round-robin these designs across the LLM panel (batch the two) BEFORE
> implementation. Specified here so the panelists have grounded context. **Neither
> touches the audio spine** (frozen, byte-identical) -- both are VISUAL procgen-layer
> changes only. **GROUNDED 2026-06-13:** Claude analyzed the real draw code
> (`_CRTRenderer.render()` in `nodes/video_engine.py`); the architectural-surface
> notes + the design block below are now code-anchored, superseding the earlier
> `OTR_PostUpscaleProcgenBlend` guess.

**Idea #1 -- Procgen episode title card on the first music cue.**
- TRIGGER: the moment the episode's intro/opening music first starts (the first
  music event), not the first dialogue.
- RENDER: the show title **"SIGNAL LOST"** + the episode title (e.g. *"Mapping
  Desperation"*) in big bold PROCGEN letters during the intro-music window -- a
  movie-credits / title-sequence vibe, the title landing with the music swell.
- CONSTRAINT: procgen-rendered (vector-style, integrated with the existing CRT/HUD
  aesthetic), NOT a baked raster.
- ARCHITECTURAL SURFACE (GROUNDED 2026-06-13 vs the real code): the draw surface is
  `nodes/video_engine.py` class `_CRTRenderer.render()` -- the procgen frame drawer
  inside `OTR_SignalLostVideo` (= `SignalLostVideoRenderer`). NOT
  `OTR_PostUpscaleProcgenBlend`, which is only the downstream ffmpeg green-only
  `screen` blend that lays procgen over the upscaled portrait. The persistent
  "=== SIGNAL LOST ===" ident + `"{title}"` subtitle ALREADY draw every frame
  (render() section 1, top-left), so Idea #1 is a windowed BIG treatment that then
  DOCKS into that existing ident -- not a separate overlay that fades. The
  first-music-cue window is derivable from `led` (render_video() already parses the
  v2 ledger; b000 = music_open) and passed into `_CRTRenderer`; the per-frame
  `volume[fi]` envelope (already computed by `_analyze_audio`) gives the swell timing.
  No new model dependency.

**Idea #2 -- Move audio-reactive visuals to the side gutters; keep the portrait clean.**
- TODAY: the green HUD ring + waveform overlay the CENTRAL portrait, partially
  obscuring the character.
- CHANGE: relocate the audio-reactive elements to the LEFT/RIGHT side gutters (the
  negative space outside the central portrait area), so the portrait composition
  lands clean and cinematic while the reactive layer still pulses with audio --
  FRAMING the action instead of overlapping it.
- CREATIVE LATITUDE on form: vertical reactive bands, side-mounted oscilloscopes,
  inverted/flipped versions of the current ring, etc. -- anything that reads well
  and keeps the centre uncluttered.
- ARCHITECTURAL SURFACE (GROUNDED 2026-06-13 vs the real code): LAYOUT-ONLY change in
  `_CRTRenderer.render()` (`nodes/video_engine.py`), NOT the blend node. What sits on
  the portrait today: the circular frequency RING (section 2, `cx=w/2, cy=0.42h,
  r=min(w,h)/5`, pulses with `vol`) dead-centre over the face; its 12 orbiting
  PARTICLES (section 3); the mirrored WAVEFORM (section 5, `y=0.72h`, full width) and
  the FREQ BARS (section 6, `y=0.86h`). The helpers `_waveform_mirror(...x,y,w,h...)`
  and `_freq_bars_wide(...x,y,w,h...)` are already position-parametrized, so relocating
  is geometry + a vertical transpose; the RMS/FFT/wave data source is UNCHANGED.
  GUTTER REALITY: HuMo character beats are 480x832 portrait pillarboxed into 1472x832
  -> ~496px black gutter EACH SIDE (the blend already fills the pillarbox bars), so
  gutters are real there; LTX/Wan b-roll beats are landscape full-frame (NO gutters).
  Guaranteed empty gutter real-estate exists only on portrait beats -- see the OPEN
  DECISION below.

**Claude design analysis (2026-06-13 -- GROUNDED against the real `_CRTRenderer` draw code; my best-judgment calls). The creative bullets below are now CONFIRMED by the code; the GROUNDED DELTAS block after them states what the code changes + the decisions I'd lock.**

- **Order: both worth doing; #2 FIRST, #1 second.** #2 fixes a composition error in ~100% of
  runtime (chrome sitting on the face, the emotional subject) -- every-frame upside, and the
  reactives get MORE legible once they stop fighting the portrait. #1 is a 4-8s delight moment that
  lands far better on an already-clean stage. They are synergistic (the cleared gutters become part
  of the title choreography).
- **The bigger hit (push this hardest): make the chrome SIGNAL-DRIVEN, not signal-themed.** Derive a
  "signal-strength" envelope from the audio and let it drive everything: strong signal
  (dialogue/music) = portrait stable, rails bright + locked, idents steady; weak signal (the silent
  gaps, scene seams) = rails decay inward, a faint scanline roll drifts the portrait, the call-sign
  flickers -- the picture LITERALLY loses the signal in the gaps and reacquires on the next cue. This
  turns the reactive layer from decoration into a narrative device, gives transitions a built-in
  grammar (signal drop = the cut), and makes #1 ("tunes in") and #2 ("hold the signal") two faces of
  ONE conceit. Cheap: same envelope feeds all of it. If only one thing goes to the panel as the
  headline, it is this.
- **#1 details.** Trap to avoid: a clean centered modern fade-in (reads as generic streaming, fights
  the CRT soul) -- the title must feel DECODED/tuned-in, not "presented." SEQUENCE, don't stack:
  `SIGNAL LOST` carrier-locks first (de-noise from green snow into a solid slab of the existing
  terminal face, one-frame chromatic tear on lock = the station ID), THEN the episode title
  teletype-reveals char-by-char with a cursor (the incoming transmission = the program). Timing OFF
  the audio (anchor entrance to the music-cue start, exit to first-dialogue-minus-a-beat), not a fixed
  clock. PERSIST BY DOCKING, not fading: the card is the BIRTH of the two persistent corner idents --
  `SIGNAL LOST` shrinks to a corner call-sign/channel-bug, the episode title settles into the corner
  terminal slot it already occupies. Palette stays green + one brief amber/white "signal acquired"
  flash on lock. References: *The Outer Limits* cold open ("we control the horizontal/vertical") is
  the spiritual touchstone; also *Twilight Zone* restraint, EBS/CONELRAD "please stand by" test cards,
  Pip-Boy/WarGames phosphor for the type; borrow *Stranger Things*' letters-lock MECHANIC but NOT its
  red-serif look.
- **#2 form factor -- my pick over the dispatch gut and the obvious defaults: two ASYMMETRIC vertical
  signal-rails** -- a continuous oscilloscope/waveform trace LEFT, a spectrum/waterfall trace RIGHT
  (or L/R channels), brightness driven by the envelope. Reject: twin arcs (curves waste the tall
  narrow gutter, half-ring reads as broken UI), vertical EQ bars (media-player/Winamp cliche), VU
  needle gauges (period-correct but too cozy-studio for a dread show; fiddly at gutter scale),
  silhouette-mirroring (pulls attention back to the portrait edge, defeats the goal). Asymmetry is
  what makes the gutters read as a real instrument vs decoration. CONSOLIDATE the old bottom waveform
  INTO the rails (don't end up with three reactive zones); bottom = subtitles + a thin baseline tuning
  strip. 16:9 caveat: gutters are narrow -- may need a slight portrait-shrink/letterbox to give the
  rails room.
- **Risks / cliche watch.** Green-on-green luma muddle (enforce hierarchy: portrait brightest,
  subtitles high-contrast, chrome dim -- watch subtitles hardest); glitch fatigue (reserve heavy
  glitch for MOMENTS -- title lock, signal-loss gaps -- keep steady-state calm); symmetric mirrored
  gutters read as wallpaper.
- **Bonus swing: an OUTRO bookend.** On the closing music, `SIGNAL LOST` reasserts and the picture
  drops to static/black (carrier drop, "we now return you to..."). Same toolkit as #1, bookends the
  episode, and literally dramatizes the show's name.
- **One-sentence evolution each, before code.** #1: reframe from "a title card" to "the birth of the
  persistent idents" -- sequence carrier-lock -> teletype, drive timing off the music-cue + first-
  dialogue stamps, then DOCK both into their permanent corner positions instead of fading. #2: reframe
  from "move the reactives to the gutters" to "two asymmetric vertical signal-rails (waveform L /
  spectrum R) whose brightness tracks an audio signal-strength envelope," consolidating the bottom
  waveform into them so the clean portrait is framed, not crowded.

**GROUNDED DELTAS + the calls I'd lock (2026-06-13, vs the real `_CRTRenderer` code):**

- **The "signal-strength envelope" already exists -- WIRE it, don't build it.** `_analyze_audio()`
  returns a normalized per-frame `volume[fi]` (RMS) + 32-bin `freq[fi]`, and `_CRTRenderer` already
  carries a dormant EMA (`self._brightness_ema`, alpha 0.08) that v1.5.1 left disabled (render()
  section 8b). Re-enable that EMA as the MASTER signal-strength driver (it already smooths transients)
  and feed it to rail brightness + the carrier-lock/dropout behaviour. The headline conceit is near-free.
- **#2 is the bigger, near-mechanical win -- do it FIRST.** Pull the dead-centre ring (section 2) + its
  orbiting particles (section 3) OFF the portrait; rebuild as two asymmetric vertical rails in the gutter
  x-bands -- LEFT = a vertical transpose of `_waveform_mirror`, RIGHT = a vertical transpose of
  `_freq_bars_wide`. Consolidate the bottom waveform + bars INTO the rails (no third reactive zone). Keep
  the centre column (~480-520px portrait region) free of bright chrome -- only the dim grid + scanlines +
  vignette stay. A SMALL ring can survive as a "signal lock" indicator DOCKED to the corner ident, never
  over the face.
- **#1 is reframed by the code.** The persistent ident is already render() section 1, so #1 = (a) a
  windowed BIG carrier-lock + char-by-char teletype over the b000 music-open window, then (b) a DOCK
  animation shrinking it into the EXISTING top-left ident + top-right timestamp slots. The vol-gated
  noise already in render() section 8 (~line 321) IS the "tuning-in snow" to de-noise FROM. Pass the
  b000 start/end + first-dialogue frame from `led` into `_CRTRenderer`. Outro bookend = the same window
  logic on the music-close beat, handing to the existing `_TelemetryHUDRenderer` post-roll.
- **OPEN DECISION (the one genuinely-open item): landscape-beat gutters.** Gutters are guaranteed empty
  only on pillarboxed portrait beats; LTX/Wan landscape beats fill 1472 wide. (A) accept the thin/dim
  rails riding the landscape edges on b-roll beats (cheapest), or (B) the compositor
  letterboxes/portrait-shrinks landscape beats so gutters always exist (cleaner, but touches composite
  geometry + the canvas/aspect question). My lean = (A) for v1, revisit (B) only if it reads badly at the
  eyeball. THIS is the item worth the panel's time; the rest is build.
- **Scope reality:** all of this is ONE file -- `nodes/video_engine.py` (`_CRTRenderer` + the
  `render_video` ledger-window plumbing). No audio-spine touch, no new model, no new node, no new widget.
  The blend stays green-only `screen`, so every rail/title reads as green phosphor automatically.

**Next step (operator-triggered):** grounding is DONE -- the design above is code-anchored. Either (a)
round-robin ONLY the open landscape-gutter decision + the rail form-factor across 2-3 panels, or (b) go
straight to a `_CRTRenderer` implementation ticket -- the mechanics are settled. Still no code until the
operator says go.

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
