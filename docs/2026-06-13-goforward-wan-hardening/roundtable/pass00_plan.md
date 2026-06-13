# OTR GO-FORWARD PLAN -- ROUNDTABLE PASS 00 (plan + Claude's pre-grounded QA)

REVIEWER FRAMING. This is the OTR v2.0-alpha go-forward plan: the single source
of truth for OPEN work on a local, offline ComfyUI custom-node project (an "old
time radio" AI video pipeline). The active build is the Wan 2.2 video engine
(section 4). Below is (PART 1) the live plan verbatim, then (PART 2) Claude's own
grounded QA (Q1-Q8), already checked against the real code at branch v2.0-alpha
HEAD 134f8e2. Your job: pressure-test BOTH. Find wiring bugs, missing fail-closed
checks, VRAM/ordering hazards, harness gaps, and anything in Q1-Q8 that is wrong,
overstated, or incomplete. Add what we missed. Be specific and cite the mechanism.

HARD INVARIANTS (reject any "fix" that breaks one): single resident heavy engine
<= 14.5 GB host NVML; 100% local / offline (no runtime model fetch); the audio
spine is FROZEN + byte-identical (mux-LAST, no -shortest); determinism is
seed-keyed within a render; every in-render fallback must be LOUD (logged +
ledger-restamped); UTF-8 no BOM, ASCII, SFW; no new widgets in the static
workflow shell. Grounding files (eng_wan_i2v.py, registry.py, render_driver.py,
otr_coverage_sweep.py, _otr_soak_capstone.py) are attached separately.

---

## PART 1 -- THE LIVE PLAN (verbatim)

### 1. CURRENT STEP
Active thread = Wan 2.2 video engine, Phase 2 (engine leg). Phase 1 (a real
I2V-14B b-roll clip) PASSED and the 5 code-gap fixes landed (2fbc2f3). Next: drive
eng_wan_i2v.render_clip through the real path and assert it in the trace -- BLOCKED
on a CS-3 scoping call (section 5). Spec = section 4. Soak fixes are DONE (R1
temp-leak d33c51f, R3 key_term a31fc24, R2 root-cause = gated_by_flag + explicit
nightly enable-set 5231d31, sweep --exclude 134f8e2). Remaining soak work = the
re-run to confirm green -- folded into forward-order item 4. ONE coder window in
the code at a time; serialize the Wan window vs any other via this file.

### 2. HARD RULES (invariants)
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8).
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no
  -shortest); test_audio_byte_identical stays GREEN. Only sanctioned audio work =
  the upstream character-voice "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100%
  local/offline; determinism seed-keyed (per-seed within a render, NOT
  run-to-run); every in-render fallback LOUD; UTF-8 no BOM; SFW; V-12 dep
  isolation; no new widgets in the static workflow shell (V-11).
- GIT: ONE branch v2.0-alpha; commit AND push together per green chunk; operator
  eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no
  0-byte / no BOM / AST parse on touched .py. prod/main GATED.
- EVERY session updates this doc + the otr-build-tracker dashboard.
- C7 seed pins only behind OTR_C7=1; normal runs log cast RNG seed (OS entropy).

### 3. FORWARD ORDER (do in sequence)
1. Punch list (GATE A). Captions DONE. REMAINING: node-level audit of LTX
   radio-open + procgen rolling credits -- baked into the headless path but maybe
   NOT into the saved JSON; prove a render FROM the JSON has them, then look-QA.
2. latentsync-100% + demos (GATE A). The OTR_LSYNC_BASE_ENGINE=still_kenburns
   fix + the two-demo set + the mixed showcase episode.
3. Wan 2.2 video engine (section 4). Phase 2 engine leg + the 8GB TI2V-5B engine
   + the eyeball gate.
4. Coverage sweep GREEN (GATE A acceptance). Re-run the permutation matrix after
   the soak fixes. Matrix (additive, not cross-product): a visual-engine leg-set
   (varies each of music/announcer/other_beats), a writer-LLM leg-set (varies
   node-1 creative_writing_model/technical_model), and a curated voice-variation
   leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no seed
   pins). Wan is a CORE/BLOCKING engine -- the sweep is NOT green until wan_i2v
   (and wan_ti2v) pass, so it stays RED until item 3 lands; that is expected. This
   re-run also answers the one open R2 question: whether humo_1.7B renders NATIVE
   char beats at 70w once its enable flag is on.
5. 3D sprints. s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the
   character_3d family (image-routing must-fixes already landed).
6. Switchable distribution S3-S6 -- generator + .gen.json tiers + wizard + README.

0-E parallel track: ltx_orbit/still_parallax/mesh_stage CPU side shipped + all
three GPU-green; Phase B HELD on scripts/_otr_0e_gpu_go.txt GO file.
Audio parallel track (own window): the character-voice "whiny" fix (may have
self-resolved -- verify first).

### 4. WAN 2.2 VIDEO -- REMAINING (active build)
Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only
(lip-sync stays SEPARATE on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ
wrapper (KJ drags in SageAttention + a numpy<2 pin this box violates). Phase 1 +
the 5 code-gap fixes are DONE (2fbc2f3).
- Phase 2 -- 16GB engine leg. Drive eng_wan_i2v.render_clip via the real path
  (scripts/otr_run_leg.ps1 / coverage_sweep --only ...). ASSERT wan_i2v is the
  final_engine in the trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <=
  14.5 GB + byte-identical audio mux + silent mp4 (h264/yuv420p/bt709, fps 25,
  has_audio False). Kill/reset the Phase-1 server first.
- 8GB tier -- TI2V-5B as a SEPARATE engine. Fetch the TI2V-5B GGUF (Q6/Q5_K_M) +
  the wan2.2 VAE into C:\ComfyUI-Models\ (record HF repo + sha256 + license,
  fail-closed). Define a NEW wan_ti2v engine (own flag/model/VAE env, registry
  registration, _node_candidates incl. the 5B latent node, loader mode,
  canonicalize, profile hook + tests) -- do NOT alias WanI2VEngine.
- Eyeball gate. Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  docs/2026-06-12-ltx23-motion/wan_clips/. Bar = real camera motion, still
  preserved, no warp.
- Risk CS-3: a long mixed episode that co-stages Wan AND HuMo can bust 16 GB --
  the supervised Wan batch decides whether the wan options are 16gb-tier-compatible.

### 5. OPEN TICKETS
- CS-1 -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS"
  was fallback-only); re-verify in the sweep.
- CS-2 -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase
  attribution reads ~3 GB; needs phase attribution.
- CS-3 -- Wan + HuMo co-stage VRAM in one episode (section 4); blocks Wan Phase 2.
- CS-4-open (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN
  14B HuMo lane so it fits 14.5 GB. Default char tier is humo_1.7B.
- R2 verify -- confirm humo_1.7B renders native char beats at 70w with its enable
  flag ON; answered by the item-4 re-run.
- Ship defaults (release) -- proposed: announcer + character = flux_still, music =
  visualizer. Keep HuMo/latentsync/3D selectable-not-default until verified.
- Harness polish -- output-tree resolver should prefer the live server's
  OTR_OUTPUT_DIR (fail LOUD on mismatch); janitor sweep at boot; widen heartbeat.
- OH-4 -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator go.
- 0-E Phase B -- tickets E-1..E-7, gated on the sweep GO file.
- Operator gates -- ComfyUI Desktop relaunch, fresh-render acceptance, latentsync
  demos + mixed showcase, whiny-voice matrix, S-3D-0 green light, stable tag.

### 6-8 (condensed)
RUNWAY: ~s2-s9 to "done" (platform in real episodes + all video models verified +
first 1-2 3D models). SHORTCUT FORK if 3D NO-GO -> character_3d defers, ~2-3
sprints. POINTERS: tracker artifact; scripts/FABLE_SOAK_REVIEW.md; the overnight
sweep launcher + GO file; the 3D + switchable specs; Bug Bible repo; full smoke
harness. PARKED: story-spine; story-pipeline; broader audio; MuseTalk; RTXUpscale;
LTX-AV; switchable S3-S6; 3D GPU lanes until S-3D-0 + operator green light.

---

## PART 2 -- CLAUDE'S PRE-GROUNDED QA (verify / refute / extend)

These are already checked against the real code at HEAD 134f8e2. For each: is it
correct? overstated? Is the proposed fix safe under the invariants? What did I miss?

### Q1 (HIGH) -- The GATE-A coverage sweep cannot detect a silent fallback.
scripts/otr_coverage_sweep.py runs every leg as
`soak.run_leg(leg, expect_floor=False, expect_engine="", profile=profile)`. In
_otr_soak_capstone.py:440-465, expect_engine="" takes the `else` branch and only
prints "EXPERIMENT histogram (informational)" -- it does NOT assert
final_engine==requested. So a leg whose engine silently falls back to
still_kenburns still scores PASS. This is the exact CS-1 class ("a prior PASS was
fallback-only") and it contradicts the plan's item-4 claim "the sweep is NOT green
until wan_i2v passes." Evidence it already bit us: R2's HuMo-1.7B flooring was
root-caused (commit 5231d31) as gated_by_flag (the enable flag was OFF) -- the
lenient sweep never flagged it; it took a manual live-log dig. The assert
machinery EXISTS: run_leg(expect_engine=<name>) raises SoakFail if any planned
shot's final_engine != <name> (lines 447-460). FIX: pass per-leg
expect_engine=engine. CAVEAT to thread: run_leg raises "no shot was planned for
%r" if the rotated engine gets no beat at 30 words -- which is probably WHY the
author used "" (a music_visual engine may get no music beat in a 30w episode). So
the fix must also guarantee each rotated slot actually gets a beat (longer fixed
episode, or a "slot-exercised" guard), else it false-fails. This blocks item 4
from meaning anything; recommend promoting it from a buried CS-1 note to an
explicit item-4 sub-task.

### Q2 (HIGH) -- wan_ti2v (8GB tier) must NOT reuse the Wan2.1 VAE.
eng_wan_i2v._loader_names() defaults vae to "wan_2.1_vae.safetensors" and
CAPABILITIES["wan_i2v"].model_requirements=["wan2.1-i2v"]. The plan's 8GB tier is
Wan2.2 TI2V-5B, which uses a DIFFERENT, higher-compression Wan2.2 VAE (separate
architecture from the 2.1 14B VAE). The plan already says "own VAE env" (good) but
should make it a fail-closed requirement: wan_ti2v MUST resolve the wan2.2 VAE and
raise NAMED if it sees the 2.1 VAE basename, because a 2.1 VAE on a 5B latent will
not decode correctly (silent garbage, not a clean error). Add to section 4.

### Q3 (HIGH) -- wan_i2v VRAM estimate (14000) sits UNDER the observed peak (14499).
CAPABILITIES["wan_i2v"].vram_estimate_mb = 14000, but eng_wan_i2v.render_clip's
own comment records the bare /prompt smoke peaking at 14499 MB with the ceiling at
14500. availability()/tier-fit math uses the estimate; an estimate 499 MB below the
measured peak makes the enable-set decision optimistic and feeds CS-2/CS-3
mis-scoping. free_after_use=True is the only thing keeping it under 14.5 GB and is
therefore load-bearing, not optional. FIX: raise the estimate to the measured peak
(~14500) or record the real number, and note free_after_use is mandatory.

### Q4 (MEDIUM) -- Single-expert MoE may fail the eyeball-gate "real motion" bar.
eng_wan_i2v docstring: the low-noise 14B fp8 is a SINGLE expert; the two-expert
HIGH/LOW MoE handoff (Path B) is "a future option, not wired here." Wan2.2-A14B is
a MoE whose HIGH-noise expert governs early-step / large-motion denoising. Running
low-noise only is a known route to weak or near-static motion -- exactly what the
Phase-2 eyeball gate ("real camera motion, no warp") must catch. This risk is
currently buried in a code comment, not in the plan. FIX: surface it as an explicit
Phase-2 risk with the Path-B two-expert handoff named as the mitigation if motion
reads weak. (Question for the panel: with core Comfy nodes + ModelSamplingSD3
shift 8.0 + 20 steps + a single low-noise expert, is acceptable I2V motion even
achievable, or does the eyeball gate effectively require Path B first?)

### Q5 (MEDIUM) -- CS-3 needs a residency model, not a co-residency budget.
Section 4 + CS-3 say a mixed episode "co-stages Wan AND HuMo" and "can bust 16 GB."
But wan_i2v (~14000-14499) + humo_1.7B (~7000) cannot co-reside under 14.5 GB by
construction, so if they truly co-staged it would already be impossible. The real
question is whether beats render SEQUENTIALLY with a reclaim barrier between them
(they should -- render_driver runs one engine per beat with a pre-render reclaim).
So CS-3 should be reframed from "decide if co-stage fits" to "PROVE the inter-beat
reclaim fully drains Wan before a HuMo beat loads, and vice-versa, within one
episode" -- a sequential-residency assertion (peak-per-beat <= ceiling, not
sum-of-engines). This also ties to the soak review's proposed VRAM-budget-aware
scheduler. Reframing this unblocks Phase-2 scoping instead of leaving it "blocked
on a decision."

### Q6 (LOW) -- Stale model-id label. CAPABILITIES["wan_i2v"].model_requirements =
["wan2.1-i2v"] but the engine renders Wan 2.2 I2V. The plan's own fail-closed rule
says record HF repo + sha256 + license; the label should match the actually-fetched
repo so the S5 wizard does not advertise the wrong asset. Fix wan_i2v's row when
wan_ti2v's row is added.

### Q7 (LOW) -- Harness shorthand. The plan says "coverage_sweep --only"; the real
file is scripts/otr_coverage_sweep.py (it has BOTH --only and --exclude, verified).
Spell the full path so the coder does not hunt. (Confirmed the --exclude soak fix
134f8e2 landed in THIS script, not the overnight runner.)

### Q8 (LOW/MEDIUM) -- Item-4 matrix breadth exceeds the script. Item 4 describes
THREE additive leg-sets (visual engines, writer-LLM, voice-variation). But
otr_coverage_sweep.py enumerates ONLY the visual-engine leg-set (the 3 video slots
x registry engines). The writer-LLM and voice-variation leg-sets are not in this
script -- they may live in run_combo_matrix.py / COMBO_MATRIX.md or may not be
built. FIX: point each of item-4's three leg-sets at its actual harness, or mark
the missing ones TODO, so "coverage sweep GREEN" has a defined surface.

### Side note (git hygiene, not a plan item)
The 4 soak-fix commits (a31fc24, d33c51f, 5231d31, 134f8e2) are committed but
UNPUSHED on v2.0-alpha (git status: ahead 4). The 2026-06-10 GIT POLICY says
commit AND push together; local-only commits are the failure mode. Recommend
pushing (operator/coder action).

### Open questions for the panel
1. Is Q1's per-leg expect_engine the right fix, or is there a cleaner way to make
   GATE-A reject silent fallbacks without false-failing slots that get no beat at
   30w? (e.g., assert on the trace's runtime_fallback_decisions being empty for the
   leg's engine, instead of expect_engine.)
2. Q4: is single-expert Wan2.2 I2V good enough to pass an eyeball motion gate, or
   should Path B (two-expert handoff) move ahead of the 8GB TI2V-5B work?
3. Anything in the Wan engine graph (eng_wan_i2v._build_graph) that is wired wrong
   vs core Comfy Wan 2.2 I2V topology (UNETLoader/ModelSamplingSD3/CLIPLoader
   type=wan/WanImageToVideo/KSampler/VAEDecode)?
4. Ordering: should the 8GB TI2V-5B engine wait until the 16GB I2V-14B leg passes
   its eyeball gate, given they share canonicalize/aspect/materialize code paths?
