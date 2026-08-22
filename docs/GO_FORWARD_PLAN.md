# OTR Go-Forward Plan

**Forward-only.** Open work, live bugs, standing operator rules, the budget ladder.
Completed work lives in `docs/HANDOFF_LOG.md` (newest at top) and every prior
revision of this file is in git. If a thing is DONE, it does not belong here.

## HOW TO TALK TO THE OPERATOR (standing, 2026-08-17 -- read before your first reply)

**He is a VIBE CODER and he named this profile himself. Treat it as the default
operating mode, not something to re-derive each session.** He does not read code
fluently -- *"I can look at colors and words on a screen"* -- but he follows
architecture, tradeoffs and consequences perfectly well. So:

* **Lead with the decision or the ask.** Never make him hunt for it.
* **Plain language.** Jargon only when the jargon IS the point.
* **Effect-on-output, not implementation detail.** "This changes what Lumina
  paints", not "this changes the fourth ternary arm".
* **A short options table** when there is a genuine fork; a **one-line bottom line**
  at the end of anything long.

**THERE IS ONE LEGITIMATE REASON TO STOP: a real blocking decision**, framed as
impact he can judge. In his words: *"if you stop I need a question for me to
answer, a real workflow coding decision that me, Jeffrey, a non-coder, can
understand its impact for you to go forward."*

**DO NOT REPORT A CONTEXT PERCENTAGE. EVER. (operator directive 2026-08-18 --
hard, and it REPLACES the old "heads up I'm at 75%" line.)** He killed it in
these words: *"STOP YOUR CONTEXT IT NOT CORRECT SO REMOVE THE CONTEXT
CALCULATION ... STOP CHASING IT."* The reason is simple: **the estimate was
wrong, repeatedly and badly.** A window that reported "roughly 98%, a fresh
window would serve you better" was under half used. There is no readout to
estimate from, so every number was a guess wearing a decimal point -- and a
wrong number does real damage, because it wraps up work early and hands him a
session switch he did not need. Do not name a percentage, do not describe
remaining context in words ("running low", "getting tight"), and do not
volunteer that a fresh window would help. Just keep working until the work is
done or a real decision blocks you.

**EVERY STOP ENDS WITH ONE LINE: what you need from him, or "nothing --
proceeding".** His cost, in his words: *"every time you stop I'm like, oh,
what's going on here."* A stop is a context switch for him, so earn it.

**BE DECISIVE ON THE OBVIOUS.** He pushed back hard on being handed three decisions
at once, one of which ("give lumina a hygiene floor") he considered a no-brainer:
*"seems a no brainer, not sure why you are asking me."* If a defensible answer
exists, take it and report it. Escalate only genuine forks. Blanket approvals are
DURABLE -- *"yes I agree lets move forward"* licenses working the queue without
re-asking per item.

## THE obs PATH IS AN OVERRIDE, NOT A DEFAULT -- and the launcher is right
## (found 2026-08-20; this CORRECTS the first framing, which said "two obs
## directories" as though it were a config mess to clean up)

**This protects the operator's #1 success signal, so it sits above the queue.**
There is only ONE ComfyUI on this box --
`C:/Users/jeffr/ComfyUI-Installs/ComfyUI/ComfyUI` -- and BOTH the OTR harness
(`scripts/_otr_soak_server_launch.cmd:179`) and the lab
(`vram-recipe-lab/run_recipe.py:56`) boot that same `main.py`. **Its DEFAULT
output tree is the stale one.** The live tree exists because the OTR launcher
deliberately overrides it, three times over (`:60-66`), under an operator
directive of 2026-06-09 quoted in the file itself -- *"even headless, ALL
outputs ... land in the REAL output folder the operator watches"*:

    set OTR_REAL_OUTPUT=C:/Users/jeffr/Documents/ComfyUI/output
    set OTR_OUTPUT_DIR=%OTR_REAL_OUTPUT%          <- the OTR writers
    set OTR_OBS_DIR=%OTR_REAL_OUTPUT%/otr/obs     <- obs explicitly
    main.py ... --output-directory %OTR_REAL_OUTPUT%   <- ComfyUI folder_paths

* **LIVE (operator-confirmed):** `C:/Users/jeffr/Documents/ComfyUI/output/otr/obs`
  -- 88 MP4 files, newest `2026-08-21 05:58`.
* **ComfyUI's DEFAULT, and it is stale:**
  `C:/Users/jeffr/ComfyUI-Installs/ComfyUI/ComfyUI/output/otr/obs` -- 58 files,
  newest `2026-06-13 07:56`. Those 58 are real episodes. **Do not delete or
  move them** (the never-clean-obs rule), and note they postdate the 06-09
  directive, so something published there through a NON-launcher boot.

**THE ACTUAL HAZARD, stated correctly: the harness is DEFENDED; anything that
bypasses it is not.** A Desktop-app run, a hand-rolled `main.py`, a new script,
or the lab's own server all inherit the DEFAULT tree and would publish a perfect
episode where he never looks -- and his rule is *"if I don't see it in obs and it
took more than 5 minutes, it's a fail."* So: **any new boot path MUST carry all
three pins**, and after any publish, verify the file by TIMESTAMP in the live
tree rather than trusting an `obs_publish OK`. A third `obs` under
`vram-recipe-lab/outputs/UVNN/obs` is lab scratch, not a publish target.

## THE CURRENT STEP, 2026-08-21 07:42 -- READ THIS FIRST

**POST-CODE QA IS CLOSED; DO NOT RUN IT AGAIN.** The prior Codex driver finished
at `eca57cb1`, and the later scoped finished-diff review, full suite and mixed
one-act / one-character canonical episode all passed without another production
code change. The mixed route published `Shadows of the Vault` with two LTX 2.5
music clips, two silent-HQ LTX announcer clips and four Z-Image-backed still
character clips; 8/8 graded, stage two positively executed, master audio stayed
authoritative, and the saved canonical workflow remained byte-identical. The
newest entry in `docs/HANDOFF_LOG.md` is the durable receipt.

**THE SETTLED `wan_i2v` ITEM IS FROZEN ON ONE NAMED AUTHORIZATION, NOT A REASON
TO DOWNLOAD SOMETHING ELSE.** OTR has the official low-noise 14B UNET, UMT5
encoder and Wan VAE, but not
`wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`. That high-noise expert is
required for an honest continuous-noise high-to-low handoff. Do not duplicate
the low-noise weight, alias another installed model or LoRA, fetch the optional
four-step LoRAs, or call a low-only run an official-template qualification.
Only an explicit operator authorization for that exact filename reopens this
lane.

**EXTERNAL MATERIAL LANE: ANTIGRAVITY OWNS `baseline-models`; OTR MUST NOT
DUPLICATE OR EDIT ITS IN-FLIGHT WORK.** Local path:
`C:/Users/jeffr/Documents/ComfyUI/custom_nodes/basline-models` (the local folder
really is missing the second `e`); remote:
`https://github.com/jbrick2070/baseline-models`. Its boundary is correct: collect
official graphs, adaptations and compatibility material; never download model
weights and never edit OTR. Let that lane finish and push before consuming it.

**THE CURRENT SNAPSHOT IS A CANDIDATE LIBRARY, NOT YET HASH-GRADE EVIDENCE.** At
07:38 it was actively changing: 19 adapted graphs and metadata rows, only 10
`source_original` files, and all substantive work still uncommitted while local,
tracking and GitHub remained at initial commit `37cb5d2`. The fetcher currently
loads JSON and pretty-prints it, so `source_original` does not preserve upstream
bytes; sources float on `main`/`master`; metadata lacks immutable revision and
raw/adapted hashes; validation is top-level/subgraph-blind and does not fail on
missing nodes/models or query live `/object_info`. Antigravity's pasted "17
pristine, all-tested" inventory is a useful progress report, not an acceptance
receipt. Do not criticize or redo its active work; require the corpus-admission
gate in `docs/COMFY_TEMPLATE_DIFF_PROTOCOL.md` when it hands off.

**ROI RULING: FINISHING THE CORPUS IS WORTH IT; A FLEET-WIDE AI/RENDER CAMPAIGN
IS NOT YET AUTHORIZED.** The corpus is cheap reusable insurance against repeated
fetching and rediscovery. The expensive re-blend program gets a discussion and
a two-case yield pilot first -- no autonomous fleet fan-out. Proposed pilot:
(1) the optional, quality-only Z-Image 2K utility, two fixtures (identity/face
and fine-detail/text), current versus utility = four stills total; (2)
`wan_ti2v` as the near-parity control, complete static diff first and render
nothing unless it exposes a genuine quality-bearing candidate. **This does not
reopen the solved Z-Image grid defect:** engine v2 and the generic
`ReferenceLatent` ban remain closed and accepted. The utility is merely a cheap
candidate for "sharper than current," and the operator may decline it and treat
Z-Image as done. One AI vision judge sees both presentation orders. **The gates
are orthogonal:** Z-Image's visual verdict tests whether high-delta
utility/topology work pays; Wan's static verdict tests whether near-parity cases
can be excluded without renders. Wan gets an A/B only if its complete diff
exposes a genuine quality-bearing candidate. Continue beyond the pilot only if
at least one rendered candidate produces a material visible win. If no rendered
candidate wins materially, stop the program and retain the corpus. A clean
no-candidate Wan diff is a successful cheap filter, not a lost visual arm. Any
no/marginal visual result stops its candidate. The operator asked for discussion
before this spend, so do not start the pilot until he explicitly approves it.

**WHEN AUTHORIZED, COMPLETE THE HASH-BOUND TEMPLATE DIFF BEFORE EACH PILOT
RENDER.** Lab commit `38d4ae0` is a grounded source census, not the completed
decision diff. Freeze exact reference bytes and OTR source/topology hashes, then
exhaustively compare every executed class, input literal, output slot, and wire.
Normalize only serialization noise. Until enumeration is complete, every
material difference remains UNCLASSIFIED -- do not preselect UniPC/Euler, step
count, shift, decoder, or any other shortlist and quietly discard the rest.

**THE LTX 2.5 LOGIC HASH IS A PRECEDENT, NOT A UNIVERSAL MODEL ALGORITHM.** The
source SHA proves which exact graph was reviewed. The durable rule `NO APPROVED
ARTIFACT -> NO AUTHORITY TO ENTER THE SHIPPING GRAPH` is the shared admission
boundary, but the IN / ADAPT / OUT guidance must be re-derived for each
engine/model/transport combination. Produce an engine-local decision receipt
that binds both graph hashes, installed transport, complete normalized diff,
and evidence-backed ruling for every difference. A similarly named node or
setting does not inherit LTX's ruling. The worked counterexample is already in
this repo: LTX's model-native same-still re-anchor was admitted, while Z-Image's
generic reference-latent injection was rejected after it visibly created the
grid. The shared method produced opposite, correct model-local rulings.

**QUALITY-ONLY DISCOVERY, NOT A REOPENED Z-IMAGE BUG: THE OFFICIAL 2K UTILITY IS
A REAL TWO-PASS REFINE, NOT JUST ESRGAN.** Comfy package `0.1.50` template
`utility_z_image_turbo_2k_upscaler.app.json`, SHA-256
`558882D2E81563A131DE99C4ED425F56EEEA3F56C37B1E5D0400260BA20D1EE1`, runs
input normalization -> RealESRGAN x4 -> 0.5 downscale -> VAE re-encode -> a
five-step Z-Image refine at CFG 1 / `dpmpp_2m_sde` / `beta` / denoise 0.33 ->
decode. OTR's current optional Spandrel lane is pixel-only x2plus and does not
qualify this graph. A hash-pinned RealESRGAN_x4plus `.pth` transport is already
installed; the template names a `.safetensors` package, so that packaging delta
must be classified, but no download is authorized or currently needed. Keep the
corrupt generic `ReferenceLatent` path OUT; keep this separate HQ utility
UNCLASSIFIED until its full graph diff, matched still A/B, native-pixel AI
judgment, cost, and identity/content preservation are recorded. Do not blindly
copy its generic `masterpiece, 8k` prompt, empty negative, `lumina2` loader type,
or CFG 1 over OTR's story/style authority.

**ONLY THEN SPEND GPU TIME ON THE CANDIDATES THE COMPLETE DIFF EXPOSES.** Use
matched inputs and change one difference, or one irreducible topology, at a
time. Give an AI vision judge native-pixel stills/crops in both presentation
orders with countable questions, then record visible payoff against render
time, VRAM, integration complexity, and judging cost. No or marginal visible
payoff ends the chase; material payoff makes the candidate eligible for
operator acceptance, executor proof, and a fresh live `otr/obs` episode. A
paper diff alone never enters production. Before production code, apply the
`CLAUDE.md` design-choice test; after code, one clean independent finished-diff
review is enough.

**Z-IMAGE CORRECTION: GRID CLOSED; MULTI-BEAT IDENTITY OPEN.** The mixed QA
showed no square tiles, so engine version 2 and the generic `ReferenceLatent`
ban remain correct. But `b002`-`b005` visibly change face/costume despite the
same derived portrait hash. The seed is deterministic, not an identity proof.
Do not re-enable `ReferenceLatent`; any replacement identity mechanism has more
than one defensible design and therefore needs a full four-round arc before
code. This visual-face issue is distinct from the older F2 content-attribution
limitation.

**THE LIVE ACCEPTANCE SHAPE IS NOW SETTLED:** one act, one character, every
impacted video role, positive executor evidence rather than adapter claims, a
fresh timestamped file in the live `otr/obs`, and master-audio identity. The LTX
ultra-smoke is no longer future work; the mixed episode above covered it.

## QUALITY PROGRAM RULINGS -- OPERATOR, 2026-08-21 MORNING

* **NO VRAM CEREMONY ON QUALITY A/Bs.** His words: *"we arent going to spend
  tokens to make this an exact vram measurement gated exercise, we will do our
  best and admit OOM faults as they come."* Best-effort runs, no clamp lanes,
  no absolute-vs-net gating, no OOM forensics on this program. An OOM is
  reported as a plain fault and the arm is marked infeasible or retried once.
  **Production ceilings and shipped-recipe discipline are unchanged** -- this
  covers the QUALITY MEASUREMENT program only.
* **Z-IMAGE 2K UTILITY: DECLINED for now** (*"maybe for 2.5 we can get a 2k
  upgrade"*). Z-Image stays done; revisit only on a 2.5-era utility.
* **THE FLEET DIFF RAN GROUNDED (2026-08-21) AND THE FLEET IS LARGELY CLEAN** --
  receipts in `vram-recipe-lab/template_sweep/2026-08-21-grounded/` (migrating
  to baseline-models): 11 exact byte-hashed references, 2 qualified stand-ins,
  17 no-reference-by-design, 0 errors. Deltas are overwhelmingly documented
  VRAM discipline (GGUF loaders, tiling, CPU pins; the 2.3 lanes run a
  FULL-PRECISION text encoder where the template ships fp8). **Live quality
  candidates, in order: (1) wan_ti2v `uni_pc`/20 vs our `euler`/30 -- free,
  pre-authorized by the ROI ruling the moment the static diff surfaced it;
  (2) ltx25 stage-1 anchor 0.7 vs our 1.0 -- the one hero-lane knob never
  tested (not among the six eliminated, all of which were grid-scored while the
  grid rode in on the stills); (3) ltx_video LoRA strength 0.5 vs 0.7.**
  Download-gated, operator authorization list only: the official 1.1 dynamic
  rank-111 LoRA (not on disk), alongside the frozen wan high-noise expert.
* **baseline-models IS THE CLEAN HOME** (operator: it *"becomes our new
  diffomatic with the principle Codex taught us"*). The grounded differ and its
  receipts move there; the corpus's remaining fix is byte-true refetch with
  pinned revisions. The old false-green sweep outputs are archived at
  `vram-recipe-lab/template_sweep/_invalid_false_green_20260820/`, never
  quotable.

### THE ASSEMBLY LINE IS VOTED AND LANE 1 IS STAGED (operator vote "1", 2026-08-21)

**Method, fixed:** one lane at a time in `basline-models` (the workbench); OTR
is untouched until a verdict says WIN. Per lane: STAGE two arm graphs generated
FROM THE SHIPPING ENGINE (identical by construction) -> PURITY GATE -> RENDER
best-effort (no VRAM ceremony; an OOM is a plain fault) -> JUDGE (operator eye
+ one VLM, both orders, countable questions) -> commit a verdict file -> only a
WIN becomes an OTR item (knob = Sonnet QA on the diff; design = arc), wired
into `otr_canonical.json` in the same change, suite, live episode in obs.

**THE PURITY GATE IS A DETERMINISTIC SCRIPT, NOT A MODEL FANOUT (operator
asked; answered 2026-08-21).** "Sonnet army" for byte-diffs = credits doing a
script's job. The gate asserts the two arms' changed-parameter set equals the
declared contrast set exactly (Bible `12.121`). Model seats that remain: ONE
Sonnet QA pass on a WINNING lane's diff before integration, and vision judging
of frames.

**ULTRACODE ROUTING FOR THE ASSEMBLY LINE (operator, 2026-08-21) -- where
orchestration PAYS and where it is banned.** The lane window may run with
ultracode on. That is a license to orchestrate where independent judgment adds
CORRECTNESS, not to re-inflate the Sonnet army the operator already declined.
Exactly three sanctioned fan-outs, everything else stays a script or the solo
driver:

1. **THE JUDGE PANEL (per A/B, the main event).** Three blind vision agents per
   frame pair, launched in parallel: agent 1 sees A-then-B, agent 2 sees
   B-then-A, agent 3 sees native-pixel CROPS only. Countable questions ONLY
   (which features resolve, how many switches, is text legible) -- never
   "which is prettier"; the arms may differ in content and aesthetics flips on
   wardrobe (Bible `12.121` territory). No agent sees another's output.
   Majority + the driver's own grounded read = the panel verdict; the operator's
   eye remains final. This replaces the single-VLM-both-orders protocol with
   something strictly stronger at ~3 calls.
2. **THE REFUTATION PANEL (only when the driver believes WIN).** Before any WIN
   verdict is committed, 2-3 skeptics each try to REFUTE it from the receipts:
   purity-gate delta set, seeds actually differing per render, ARMS.sha256
   matching what rendered, frames pulled from the right mp4s. Default to
   refuted-if-uncertain. A win that survives is committed; a win that does not
   was about to waste an integration cycle and operator attention. LOSSES are
   not refuted -- a loss costs nothing downstream.
3. **THE COMPLETENESS CRITIC (one agent, at lane close).** "What was not run,
   not checked, or silently capped in this lane?" Its findings open follow-ups;
   they do not reopen the lane.

**STILL BANNED UNDER ULTRACODE, no exceptions:** model fan-out on byte/JSON
diffs (deterministic script, full stop); parallel LANES (assembly line is
voted -- one lane at a time); render fan-out (sequential execution only, one
server, per scope discipline); reviewer multiplication on integration (the
08-20 one-clean-review ruling stands -- ONE Sonnet QA pass on a winning diff,
more only on a blocker or disagreement). Ultracode changes HOW MANY EYES look
at pixels and at a claimed win; it changes nothing else about the method.

**Lane order:** (1) `wan_ti2v` sampling recipe -- contrast set exactly
{KSampler.sampler_name uni_pc-vs-euler, KSampler.steps 20-vs-30,
ModelSamplingSD3.shift 8-vs-5}, bundled as a SCREEN, decompose only on a win;
(2) `ltx25` stage-1 anchor 0.7-vs-1.0; (3) `ltx_video` LoRA strength
0.5-vs-0.7. Download-gated (operator authorization only): official 1.1
dynamic rank-111 LoRA; wan high-noise expert (already frozen).

**LANE 1 IS CLOSED: NO WIN. The shipped `wan_ti2v` recipe stands and nothing
was queued as an OTR item** (2026-08-21 midday, `basline-models` `23c77a5`,
verdict `basline-models/verdicts/lane1_wan_ti2v.md`). It is not a bare tie --
two measurements on known ground truth put the official recipe BEHIND: temporal
stability worse in 7 of 8 cells (+11% to +60% mean frame-to-frame change), and
the test card's neutral grey wedge drifting to channel spread 11.50 against
ours at 2.42. The one unanimous panel call for official was refuted 3/3 and the
refutation was re-derived by the driver: the edge advantage is CONTRAST (raw
ratio 1.222 at f097, 0.980 after identical autocontrast), and that cell is the
one where the arms stopped rendering the same scene (NCC 0.627 vs 0.89-0.999
elsewhere).

**FOUR THINGS FROM LANE 1 THAT BIND THE REST OF THE PROGRAMME:**
* **ONE EASY FIXTURE CANNOT CLOSE A LANE.** The officer close-up tied on every
  seat; the operator called it before any judging and the panel confirmed it.
  Every lane carries hard content AND the authored **test card**
  (`staging/lane1_wan_ti2v/make_testcard.py`: colour bars, 16-step grey wedge,
  shape rows, gratings 16px-3px, Sloan eye-chart rows, both polarities, drawn at
  the exact render canvas so the latent resize is a no-op). Countable beats
  impressionistic.
* **THE A/A NULL IS NOW MANDATORY, ONCE PER LANE** (`tools/run_aa_control.py`).
  The pipeline is bit-exact -- an identical graph reproduces all 97 frames by
  sha256 -- and the panel returns TIE on identical pixels. That is the noise
  floor every margin is read against.
* **ADMIT A CELL ONLY IF THE ARMS STILL RENDER THE SAME SCENE.** Arm-to-arm NCC
  at the final frame under ~0.90 means "which is better" is the wrong question.
  Across lane 1's cells, correlation between arm similarity and decided votes
  was -0.821: the panel ties when renders look alike and picks a winner when
  they diverge.
* **TONE FEEDS THE COUNTS.** Judges told to ignore contrast cannot. Contrast-match
  before judging, or ask a tonal question outright.

**WHAT LANE 1 DID NOT SCREEN, so it is not re-derived as new:** the fleet diff
lists nine official-vs-ours differences; lane 1 screened the three sampler
parameters and held the rest at ours in BOTH arms -- fp16 vs Q5_K_M GGUF
weights, fp8-scaled vs GGUF text encoder, untiled vs tiled VAE decode, and the
reference 1280x704 x 121 @ 24fps canvas. Each is its own candidate lane; the
precision deltas are the interesting ones. Note also that `shift 8` was tested
at an operating point it was not authored for (the engine documents 5.0 as the
5B value), which bounds the null without rescuing it.

**A PRODUCTION HAZARD FOUND IN PASSING, NOT FIXED (out of scope that window):**
`copy.deepcopy` silently corrupts `wrapper_bridge.Wire`. It is a 2-tuple
subclass taking two constructor arguments, so it inherits
`tuple.__getnewargs__` and a deep copy rebuilds it as `Wire(('pos', 0))` --
the whole wire slides into the src slot and returns nested. The copy still
walks, indexes and serialises like a wire, so it stays silent until submit
time. **It is LATENT, not live: the only `deepcopy` in `_otr_video_engines` is
`render_driver.py:3889` on a LEDGER, not a graph**, so no shipping path is
currently exposed. It is a trap for the next person who copies a graph rather
than a bug to chase now. Re-verify that grep before fixing; it is production
code and wants the design test.

**THE LANE 1 TAIL IS ALSO CLOSED (2026-08-21 afternoon, workbench `18fe7e6`):**
the temporal metric is now a standard receipt field (`tools/temporal_stats.py`,
wired into `render_arms.py`, all 16 legs backfilled), and the shift transplant
objection is dead -- the refcanvas retest (crowd at 1280x704 x 121, both arms)
shows the scene divergence disappearing (NCC 0.9698 vs 0.6269), detail tying
exactly (0.995 raw and normalized), and official STILL marginally less stable
with a faint hallucinated streak in its final frame. NO WIN holds at both
operating points; no bound remains on the null.

**LANE 2 IS CLOSED: NO WIN. The shipped ltx25 anchor 1.0 stands** (2026-08-21
evening, `basline-models` `dc63ab0`, verdict `verdicts/lane2_ltx25.md`).
Decided by a verdict matrix pre-declared BEFORE rendering: the identity gate
held 6/6 blinded seats (the soft anchor does NOT lose likeness within a clip),
and no material gain -- 18 seats split 6 marginal-soft / 4 ours / 8 TIE with
directions flipping between seats and seeds, the only CLEAR call going to
OURS. A/A null passed 97/97 byte-identical (ltx25 is bit-exact; receipted in
AA_CONTROL.json). Key bounds, all recorded in the verdict: within-clip
identity only (the between-beats identity question is its own lane); the
MOTION axis was structurally untested (every prompt demanded stillness, seats
judged stills) -- if soft is ever re-argued it takes one motion-demanding
fixture, 4 legs on the existing harness.

**WHAT LANE 2 ADDED TO THE METHOD (all pushed):** the r1 kibitz (Codex +
Antigravity, scoped r1, reported as such) resolved the arm shape -- ONE
constant drives BOTH anchors, so an i2v-only arm is not shippable -- and
caught the runtime CPU-pin blocker, closed by the registered
`CLIPLoaderGGUFCPU` as a documented shared ADAPT in both arms. Bible `12.122`
promoted (301 entries, suite green): an in-process graph authoring form is not
the HTTP prompt form -- V3 dynamic inputs dot-flatten, runtime class swaps
re-declare, local classes fail closed. `tools/dynamic_input_census.py` sweeps
this BEFORE a lane renders; it already flagged that **lane 3's `ltx_video` is
unbuildable over the API** (local `_SigmasFromValues` class; the ADAPT path is
the registered `ManualSigmas`, exactly as ltx25 uses). The completeness critic
then caught a false receipt line in the lane 2 verdict itself (a lane-1
permutation count pasted into lane 2); corrected in place, `seat_plan` now
hashes lane+fixture+seed+seat, and the panel prompts are archived beside
PANEL.json so its numbers can be re-derived.

**LANE 3 IS CLOSED: NO WIN. The shipped `_LTX_DISTILLED_LORA_STRENGTH = 0.70`
stands** (2026-08-21 afternoon, workbench `68ce4c5`, verdict
`verdicts/lane3_ltx_video.md`). Panel dead heat (9 half / 8 ours / 1 TIE, no
fixture decisive; "clear" margins pointed in OPPOSITE directions within the
same cells -- on a t2v lane the arms compose different scenes from one seed,
so seat margins are substantially scene luck, anticipated in the pre-declared
matrix). A/A null 97/97 byte-identical, overlay check ran before the panel,
the ManualSigmas ADAPT worked first try off the census flag.

**THE LANE 3 COMPLETENESS CRITIC HAS NOW RUN (2026-08-21), AND IT CORRECTED
THIS ENTRY. Every item below was re-verified by the driver against the real
files before being folded into `verdicts/lane3_ltx_video.md`:**
* The panel margin count was **10 clear**, not 9 (9 was the half-column total
  pasted into the margin sentence).
* The 0.5 arm does move less in 6 of 6 cells, but the range is **-6.4% to
  -37.9%, median -26.3%** -- NOT the "25-38% on every cell" this entry used to
  claim. March at seed 20260821 is only -6.4%.
* **The "ours plays its instruments" read is WITHDRAWN.** Two of three seats at
  march/seed42 record the opposite: half delivers the prompt's five brass
  players while ours decays to 3-4 with deformations. Only the half-dressed
  figure at f097 survives. NO WIN still stands, but on the BURDEN OF PROOF --
  the candidate showed no material gain -- not on ours being visibly better.
* **New bound, and it is the material one: the knob was screened on the
  TEXT-ONLY path while `OTR_ENABLE_LTX_I2V` defaults to `"1"`**
  (`eng_ltx_video.py:931`). Production conditions on an image by default, so
  the null is bounded to t2v and the i2v question is a genuinely open NEW lane.
* Arm-to-arm NCC, never computed during the lane, is **0.14-0.74 across all six
  cells** -- every one under lane 1's ~0.90 admission line. That CONFIRMS the
  scene-luck reasoning and explains why 9-8 carries no signal.
* `seat1_full` and `seat2_full` were **byte-identical image sets** in all six
  cells, and disagreed in 3 of 6. That is the judge noise floor, measured free.
* Read order landed **14 candidate-first / 4 ours-first**; `seat_plan` has no
  balance constraint, so lane 2's defect moved from seat1 to seats 2 and 3.
* `PANEL_META.json` / `PANEL_PROMPTS` were **not archived** for lane 3, so its
  judge model and questions are permanently unrecorded. Enforce in the harness.
Bound unchanged: same-file-only (rank-111 LoRA stays download-gated).

**FIVE LANES ARE NOW CLOSED, ALL NO WIN, EVERY SHIPPED RECIPE CONFIRMED WITH
RECEIPTS** (2026-08-21). Lanes 1-3 as recorded above; then:

**LANE 4 -- `wan_ti2v` TEXT ENCODER PRECISION: NO WIN** (workbench
`e545404`..`2388fb5`). Our Q5_K_M umt5 against the fp8-scaled file the official
template names; both were already on disk. **This was the first lane aimed at a
COMPROMISE rather than a deliberate quality choice** -- the GGUF encoder exists
for the 16 GB ceiling and its cost had never been measured. 8 legs, no OOM at
14.4 GB. The decisive instrument was NOT the panel: `tools/encoder_delta.py`
loads both encoders on CPU and compares the conditioning tensors directly.
**They are NOT output-equivalent** -- cosine 0.9904-0.9959 but relative RMS
0.355-1.041 -- yet the rendered video lands within ~1% on three of four cells.
The input moves materially and the output barely does. On the test card both
arms reproduce the SAME failure modes in the same places, including the same
`TI2V`->`TIZV` corruption and the same artifact.
**`testcard_motion` FAILED as a motion fixture:** prompted camera drift over a
flat graphic still produces 0.0 px translation in both arms at both seeds
(`tools/drift_stats.py`). It survives only as a STATIC acuity card.

**LANE 5 -- THE OFFICIAL `ltx_video` PAIRING ON THE i2v PATH: NO WIN**
(workbench `d90d747`, verdict `verdicts/lane5_ltx_i2v_official.md`). Closes
BOTH bounds lane 3 left open: staged through `_build_graph_i2v` (the path
production defaults to), against the official rank-111 dynamic LoRA at 0.5
(2.74 GB, downloaded on operator authorization, byte-verified). Bundled screen,
declared as such. **The pre-declared NCC admission gate rejected all four cells
(0.40-0.69 against a 0.90 floor), so NO PANEL WAS RUN** -- and the A/A null
(97/97 byte-identical, NCC exactly 1.000000) proves that gate measures the arms
rather than engine noise. What decided it was the operator's eye: *"they all
look good"*, *"the differences are minute."*

**THE OPERATOR EYE SEAT WAS FINALLY EXERCISED** (`verdicts/OPERATOR_EYE.md`).
24 blind pairs in one 93-second reel: **15 SAME, 5 decided, 4 skipped**, and no
lane drew a consistent preference across both its seeds. It overturned nothing
-- and on lane 3 march/seed42 it independently landed on the same side as two
blinded seats, confirming the driver's withdrawn strip read was wrong at BOTH
seeds. He then returned to a SKIPPED segment and pointed out an **orphan drum
in the SHIPPED arm**, verified at native pixels.

**STILL-CANVAS REVIEW (workbench `ae441eb`) -- NOT ADOPTED, NOT REJECTED.**
Production stills are already minted ABOVE every video canvas (1472x832
dominates an 80-still sample), so supersampling already happens. At native size
1080p is clearly more detailed; but the prompted test card shows the
over-resolution artifact -- **a DUPLICATED eye-chart row at both seeds**, and
fused circles at one -- consistent with 1080p sitting ~97% above the engine's
documented 1024x1024 design point against 1472x832's 16%. The
downscale-to-1024x576 measurement is CONFOUNDED (changing canvas changes
composition, so there is no matched-content comparison) and cannot decide it.
**True-1080p fact:** `z_image_turbo` renders exactly 1920x1080; `flux2_klein`
snaps to /16 and yields 1920x1072.

**WHAT CANNOT BE DIFFED AT ALL** (`receipts/image_fleet_probe.json`): the audio
engines are not graph-based -- `eng_kokoro`, `eng_bark`, `eng_chatterbox`,
`eng_musicgen`, `eng_stable_audio*` import their model and run it in Python, so
there is no ComfyUI graph to compare against a template. Image engines COULD be
diffed but all four block on the same gate: no declared `render_canvas`, and
the differ correctly refuses to invent a fixture.

**THE PROGRAMME'S NEXT STEP IS A DECISION, NOT A RENDER.** Per the standing ROI
ruling ("if no rendered candidate wins materially, stop the program and retain
the corpus"), five nulls is that condition. Remaining candidates: the
full-precision wan UNET (~10 GB, coin-flip on fit, the LARGEST untested 16 GB
compromise), `wan_ti2v` tiled-vs-untiled VAE decode (free, 8 legs), and the
`ltx25` motion fixture (free, 4 legs). **The operator's 16 GB rule governs
downloads: "if the model can run under 16gb that's fine we download."**

**A BOUND THAT MUST BE WRITTEN INTO ANY CLOSING STATEMENT:** five lanes tested
KNOBS. The single biggest 16 GB COMPROMISE -- the Q5_K_M UNET -- was never
challenged, because the official full-precision counterpart is not on disk.
Five nulls must NOT be read as "the compromises are validated."

**Bible now 300 entries (`3ac4d9b`), 22/26/3.** `12.121` promoted this
morning: an uncontrolled second variable voids every arm of a visual A/B (the
grid rode in on the stills; the drift probe obeyed its prompt). Another window
independently promoted `12.120` overnight; both stand.

## RECENT DECISIONS AND OPEN FOLLOW-UPS -- CURRENT STEP ABOVE IS AUTHORITATIVE

Everything below is a standing ruling, an open follow-up, or a receipt explaining
why a ruling exists. Do not mistake an older `NEXT` label below for the active
queue; the current-step block above owns execution order.

**WHY THE FILE IS STILL 3,400 LINES, and why that is not a bug to fix.** The
2026-08-16 audit called for an archive split and it was attempted again on
2026-08-20. **It was rejected on inspection:** the bulk of the length is NOT
receipt narrative, it is sections that have ALREADY been compressed to
"What still binds:" rulings -- item A's body is four rulings and one pointer to
`HANDOFF_LOG.md` for the receipts. Moving those to an archive would move the
RULES, not the history. The length is the price of not re-deriving them.
So: the file is LONG on purpose, and it is made forward-only by this block
existing, not by deleting the rest.

### ITEM `I` IS DONE AND PUSHED -- `0645839b`, 2026-08-20

Reconciliation at the `lock_cast` boundary, a soft never-raising post-check on
BOTH prose fields, clean-room regeneration, deterministic floor, guard events
persisted with episode identity. Suite **11237 -> 11270**, delta exactly the 33
tests added, zero regressions. Four QA rounds (Sonnet 4.6, Codex/Sol x2,
Antigravity) each found something real. Instrument:
`scripts/audit_wrong_person_census.py`. Provenance and the re-promotion
procedure: `docs/2026-08-20-ltx25-recipe-provenance-and-repromotion.md`.

**THE `media_archive` EXTENSION IS NOW CLOSED AND LIVE-PROVEN (2026-08-21).**
Its interpreter emits a structured `upstream_identity_names` list, the shared
payload contract validates it, and the writer feeds it into the existing
name-authority boundary without mining prose. A conservative census found five
dirty rows in five of 104 `media_archive` ledgers; four of those five episodes
were published. Historical artifacts remain frozen and are not back-repaired.

**Still open from the broader name-authority family:** `scifi_news_pro` never
calls `lock_cast`; the original-family ~44 historical rows are not back-repaired;
`OTR_NAME_MODE=llm_slot_fill` renames rows after descriptions and is RECORDED
as unfenced rather than fixed.

### COMPLETED RECEIPT: THE LTX 2.5 DELIVERED-VIDEO GRID

**FINAL RULING OVERRIDES THE INSTRUMENT HISTORY BELOW:** gridscore was rejected
because it counted resolved detail as grid. No gridscore number is quotable as
evidence. The operator's matched-frame judgment and the live executor/publish
receipts in `docs/HANDOFF_LOG.md` own the shipped result.

**Six candidates eliminated by measurement, all on a detector that has now been
validated.** Full state: `vram-recipe-lab/LTX25_GRID_PROBLEM_STATEMENT_v2.md`
and `LTX25_FULL_DIFF_vs_official.md`.

* **Eliminated:** decode tiling (VAE-only round trip, identical), the VAE
  itself, the pre-VAE resampler, delivery-side filtering (7 ffmpeg chains +
  Real-ESRGAN), canvas size, sampler eta, negative prompt, `img_compression`,
  the official sigma list, decode temporal 64/16. Re-scored on the fixed
  detector: every arm sits at **3.28-3.82** against a 3.821 control -- all
  firmly in "grid clearly visible".
* **THE DETECTOR WAS INVERTED AND FIVE RENDERS WERE SPENT ON IT.**
  `ltx25_notch.py::score_full_frame` scored the CLEAN image higher than the
  gridded one (7.2 vs 4.8). Replaced by `vram-recipe-lab/ltx25_gridscore.py`,
  which is 2-D and ABSOLUTE, ranks the known degrid ladder monotonically
  3.156 -> 1.567, and ships `--selftest` that refuses to pass if the ordering
  breaks. **Run the self-test before trusting any number from it.**
* **WHAT IS LEFT IS ARCHITECTURE, NOT A KNOB.** The official ComfyUI I2V
  template is TWO-STAGE: 768x512 base -> `LTXVLatentUpsampler` x2 ->
  `LTXVImgToVideoInplace` 1.0 re-anchor -> a 3-step refine on sigmas
  `[0.85, 0.725, 0.4219, 0]` -> decode once at 1536x1024. **We run stage one
  only** and stretch 2.31x in ffmpeg afterwards.
* **FACE-DRIFT RISK: CLOSED, AND IT CLOSES FOR THE TWO-STAGE (v2, matched
  prompt, 2026-08-20 evening).** Matched pair at **seed 44**, both arms, prompt
  aligned to the still, final frame of 97:
  * **Control** holds the officer's identity to the end -- soft, 832x480.
  * **Two-stage holds the SAME identity, same uniform, same pose**, at far
    higher fidelity: individual hair strands, cheek skin texture, the mole,
    defined eyelids, fabric weave on the lapel.
  * **Neither arm drifts. The refine does not move the face -- it resolves it.**
    The v1 "identity collapse" was entirely the mismatched prompt, now confirmed
    twice: once by the control reproducing it, once by it vanishing when the
    prompt was fixed. Frames: `outputs/V2_{control,two_stage}_s44_{f000,flast}.png`.
  **This was the last open technical risk on the two-stage item.**
* **THE LAB OOMs ~50% OF LTX 2.5 RUNS AND THE ROOT CAUSE IS FOUND: THE LAB DOES
  NOT CPU-PIN THE TEXT ENCODER AND PRODUCTION DOES.** Across v1+v2, 6 of 12 runs
  died in `CLIPTextEncode` at ~14.2 GiB with `LTXAVTEModel_` reporting
  **10,917.81 MB loaded, "full load: True"**. Production ships
  `_cpu_pinned_clip_loader` (`eng_ltx25.py:117`), a `CLIPLoaderGGUF` subclass
  that pins the Gemma encoder to CPU -- and its own comment records the trap:
  *"``initial_device`` ALONE IS NOT ENOUGH"*, you must set `load_device` and
  `offload_device` too (`:131-137`), with a fail-loud `encoder_not_on_cpu`
  guard. **The lab submits stock API graphs and cannot inject that subclass, so
  it has been measuring a configuration ~10.9 GiB heavier than production ever
  runs.** Consequences, both real: the lab's coin-flip OOM rate is an artifact
  and is fixable, and **every lab VRAM verdict on this lane -- including "both
  arms fail the 14.5 gate" -- is suspect until the lab pins the encoder the way
  production does.** A bench harder than production distorts every number it
  produces. Closing this is prerequisite to trusting any further LTX 2.5 VRAM
  claim.
* **THE IDENTITY-DRIFT PROBE WAS CONFOUNDED AND THE TWO-STAGE IS EXONERATED
  (driver ran the lab directly, 2026-08-20 evening).** The v1 derivatives swapped
  the conditioning still to a close-up of a uniformed officer
  (`fixtures/portrait_16_9.png`) but **kept the golden recipe's TEXT PROMPT**:
  *"1950s cinematic shot of the detective standing by the rainy window, turning
  abruptly and slamming his fist on the desk ... rain drumming steadily against
  the glass."* The model was shown one thing and told another.
  * Two-stage seed 42: **frame 0 renders the still faithfully and with far more
    detail than the reference** (skin texture, eyelids, individual hair) --
    identity perfectly held. **Frame 96 is a different man at a desk by a
    rain-streaked window** -- i.e. the PROMPT, executed correctly.
  * **PROVED, not assumed: the CONTROL arm does exactly the same thing.**
    `drift_control_43` frame 96 is also a man from behind at a desk by a
    rain-streaked window. **Both arms abandon the portrait identically, so the
    drift is prompt obedience, not refine damage.**
  * **Scoring v1 as-is would have condemned the two-stage for doing its job** --
    and it would have looked damning, because the refine adds three extra guided
    steps and therefore follows the text FURTHER by construction. v1 measured
    prompt-following strength and would have reported it as identity loss.
  * **v2 exists and is the real test:**
    `scratch/ltx25_two_stage_identity_drift_v2/` -- prompt matched to the still
    (locked-off close-up of the officer, no cut, no scene change), built by
    `build_drift_v2_matched_prompt.py` with invariants asserting that NOTHING
    but the prompt and the naming changed. v1 artifacts are retained as the
    evidence for this finding.
* **TWO OPERATIONAL LESSONS FROM DRIVING THE LAB DIRECTLY, both non-obvious:**
  * **The runner's GPU idle gate refuses any start above 3.0 GiB baseline, and
    the two-stage leaves ~10.9 GiB resident when it finishes.** So a back-to-back
    batch silently aborts every run after the first two-stage -- four of six on
    the first attempt, `rc=1` in under two seconds each, which reads like a crash
    and is not one. **Pass `--shutdown` per run.** The gate is correct; the
    harness just has to reset between runs (`CLAUDE.md` section 4).
  * **The lab loads the LTX text encoder FULLY onto the GPU --
    `LTXAVTEModel_`, 10,917.81 MB, "full load: True"** -- then needs 1.88 GiB
    more on top of ~13.7 already allocated, which is the `CLIPTextEncode` OOM
    that blocked the control arm. It is a KNIFE EDGE, not a wall: the same
    recipe SUCCEEDED at 15.479 GiB earlier the same day. Production keeps this
    encoder CPU-pinned with an episode-scoped cache; **the lab does not, so the
    lab is measuring a heavier configuration than production runs.** Worth
    closing, because a bench that is harder than production distorts every
    verdict it produces.
* **OPERATOR VERDICT 2026-08-20: THE TWO-STAGE WINS ON PICTURE, AND THE 2.6x IS
  ACCEPTED.** Shown the same frame from both arms through the SHIPPED delivery
  chain at 1920x1080 (`lanczos` -> `unsharp=0.4` -> `pad`, built from
  `otr_silent_composite._scale_filter`, nothing tuned for the test), his call
  was immediate: **"2 its no brainer"** and, decisively, **"you can almost read
  the faux text."** On the cost: **"2x6, eh kinda yes"** then **"i mean yeah
  peoel wnat quality."** **So the two-stage EARNS ITS FULL ARC** -- it did not
  OOM, it is better by the only authority that has never been wrong on this
  item, and the operator has accepted 2.619x on the video stage. Frames kept at
  `vram-recipe-lab/outputs/EYEBALL_{1_control_onestage,2_twostage}_1920x1080.png`.
  **What is NOT yet decided:** adoption still owes a full arc, a canonical
  workflow re-proof and his eyeball on a real episode. The picture question is
  closed; the shipping question is not.
* **THE REPLACEMENT DETECTOR IS ALSO WRONG, AND THIS IS ITS SECOND STRIKE ON THE
  SAME PROBLEM.** Run on the two EYEBALL frames -- **identical 1920x1080,
  identical chain, so resolution is NOT the excuse** -- `ltx25_gridscore.py`
  scored control **6.260** and two-stage **6.724** (higher = more grid), i.e. it
  called the near-readable frame WORSE. **Its `--selftest` passes anyway, and
  that is the real defect:** the ladder is five rungs of progressive SMOOTHING,
  so grid and detail fall together on every rung and the metric cannot be caught
  conflating them. The two-stage arm is the first sample where they move in
  OPPOSITE directions -- more detail, less grid -- and the metric fails on it.
  **A validation set that only contains one direction of change cannot validate
  a metric.** Do NOT quote a gridscore number as evidence until this is fixed,
  and per the two-strikes rule a THIRD instrument gets the panel before code.
* **THREE INDEPENDENT JUDGES AGREE AND THE SCALAR IS THE LONE DISSENTER
  (2026-08-20). THIS RETIRES `ltx25_gridscore.py` AS AN EVIDENCE SOURCE.**
  On matched 768x432 NATIVE-PIXEL crops
  (`vram-recipe-lab/outputs/CROP_EYEBALL_*.png` -- cropped precisely because
  full frames get downscaled on the way into any chat window, which erases the
  acuity being measured):

  | judge | pick | evidence it volunteered |
  |---|---|---|
  | operator's eye | **2** | *"you can almost read the faux text"* |
  | Claude (vision) | **2** | resolved face (eye/nose/mouth/ear), epaulettes + tie, countable console switches, legible letterforms |
  | Gemini (separate window, drag-and-drop) | **2** | *"sharper face"*, *"officer uniform"*; calls arm 1 *"blurrier, slightly melted-looking"* |
  | `ltx25_gridscore.py` | 1 | none available -- a scalar cannot show its reasoning |

  Gemini reached "sharper face" and "officer uniform" UNPROMPTED, from a
  different model family with no sight of the other judges' reasoning. **Three
  judges, three methods, unanimous.** Caveat kept honest: the Gemini run was one
  pass in one order, so position bias is not formally excluded -- but the odds of
  three independent judges erring in the same direction are low enough to act on.
  **No `gridscore` number may be cited as evidence until the replacement
  instrument exists.**
* **NO PAID API IS NEEDED FOR VLM JUDGING, and a suggestion to build one was
  declined.** A `google-generativeai` + API-key script was proposed and rejected:
  it violates the offline-first scope rule, it named the stale `gemini-1.5-pro`,
  and its prompt asked for *"aesthetic quality, lighting, composition"* -- an
  active CONFOUND here, because the two arms differ in CONTENT (a shirt became a
  uniform), so an aesthetic judgment can flip on wardrobe rather than on
  resolution. **What works instead, at $0:** matched native-pixel CROPS, dropped
  into a vision window, with COUNTABLE questions (how many switches, which
  facial features resolve, is any text resolvable as letterforms) rather than
  "which looks better" -- a countable answer is checkable, an opinion is not --
  and the pair sent in BOTH orders, trusting only a self-consistent answer,
  because VLMs carry a systematic position bias.
* **THE OPERATOR'S INSTRUMENT DESIGN -- an AI-purposed eye chart, and it is the
  right shape because every region carries its own ground truth.** His words:
  an eye chart with *"some text mayeb blobs that look lie mneys or abargham
  linocon"*, plus *"its emji render and an OCR tect emojiu"*. **The chart goes
  IN as the conditioning still, so this is a PRESERVATION test, not a generation
  test** -- known image in, degraded image out, per-region delta against an exact
  reference. That structurally forecloses the failure that killed both previous
  detectors, because a blurred frame scores worse on every region and there is
  no "smooth everything and the number improves" path.

  | region | catches | scorer |
  |---|---|---|
  | shrinking letter rows | glyph acuity (Snellen threshold) | OCR per row -> "readable to row N" |
  | Lincoln (public domain) | face drift, a standing bug class | face-detect confidence + landmark drift vs reference |
  | engraved money-style texture | fine repeating detail turning to mush | local contrast / MTF vs the same patch in the reference |
  | emoji | **CHROMA** -- color bleed and desaturation, which a black-on-white chart cannot see | template correlation vs the exact font glyph (OCR cannot read emoji) |
  | flat gradient panel | banding / posterization | step count vs the reference ramp |

  Use SYNTHETIC guilloche rather than real currency -- same fine-texture test,
  and reproducing banknotes carries real legal restrictions. No OCR is installed
  on the box; `easyocr` is the fit (torch already present, offline after the
  first model fetch), with deterministic per-glyph correlation as the
  cross-check so no single instrument is unauditable again.
* **THE CORRECTED TWO-STAGE PROBE RAN AND IT IS NOT A VRAM PROBLEM (lab/Codex,
  2026-08-20 evening -- SUPERSEDES the failed 16:27 attempt below).** Wiring
  fixed as specified, all `/object_info` checks live, detector `--selftest`
  PASSED before scoring, artifact independently `ffprobe`d: **h264, 1664x960,
  97 frames, 25 fps**. Receipts reconcile (`net + baseline == absolute`).

  | | absolute | net | baseline | wall clock |
  |---|---|---|---|---|
  | one-stage control | 15.479 | 14.493 | 0.986 | 97.5 s |
  | two-stage | 15.516 | 13.879 | 1.637 | 255.4 s |

  **THE GATE IS ABSOLUTE PEAK** (lab's call, accepted -- absolute is what OOMs a
  16 GB card; `run_recipe.py:9508`). **The consequence must be read honestly:
  BOTH ARMS FAIL IT.** Control is over by 0.979, two-stage by 1.016 -- the
  two-stage is **37 MB worse than the path we ship every day**, and the two runs
  started **651 MB apart in baseline residue** (the receipt notes baseline
  *"includes owned lab server and desktop load"*). **So VRAM does not
  distinguish the arms and never did. It did not OOM.**
* **THE REAL COST IS TIME: 255.4 s vs 97.5 s = 2.619x per shot**, multiplied
  across every shot by per-segment rendering.
* **AND THE GRID IS STILL UNMEASURED, which is the only thing that decides the
  item.** The delivered frame scored **5.961 against a 3.821 control**, but it
  is a 2x output and the latent upsample doubles the grid's PERIOD, so the
  detector may be reading the wrong spatial frequency. "Much worse" and "not
  measured properly" are both live and point opposite ways. **The one
  outstanding measurement: score BOTH arms through the SHIPPED delivery chain at
  1920x1080** (control stretches 2.31x, two-stage ~1.15x), `unsharp` untouched,
  same seed and frame. Request:
  `vram-recipe-lab/LAB_MINI_REQ_ltx25_two_stage_grid_comparable.md`.
  **Grid visibly better -> operator call on 2.6x render time. Same or worse ->
  the item closes on evidence.**
* **THE FEASIBILITY PROBE RAN 2026-08-20 16:27 AND IT ANSWERED NEITHER HALF OF
  THE GATE. DO NOT READ ITS `FAIL` AS "DOES NOT FIT".**
  `vram-recipe-lab/results/ltx_2_5_two_stage.json` reads
  `status: FAIL (no artifact output)`, `blocked: false`, `duration_s: 99.3`,
  `absolute_peak_vram_gb: 15.464`. **It was not an OOM.** It died at the FIRST
  node of stage two -- `LTXVLatentUpsampler` -- so that 15.464 GiB is stage one
  plus an upsampler load, and the refine and the 1536x1024 decode never ran.
  **Root cause, read out of the shipped ComfyUI source, not guessed:** the
  stage-one latent on this lane is a JOINT AUDIO+VIDEO nested latent
  (`comfy_extras/nodes_lt.py:817` packs
  `NestedTensor((video_samples, audio_samples))`; the recipe under probe is
  `ltx_2_5_golden_i2v_foley.json`, the foley lane), and `LTXVLatentUpsampler`
  feeds `samples["samples"]` straight into the video VAE's `un_normalize`,
  which `TypeError`s on a nested tensor. It would have failed identically at
  2 GiB. The fix is two nodes that already exist and are registered
  (`nodes_lt.py:1191-1192`): `LTXVSeparateAVLatent` before the upsampler and
  `LTXVConcatAVLatent` after the re-anchor, because the refine guider
  (`LTXVDualCFGGuider`) wants the joint latent back. Re-run request with the
  exact chain and the report-back list:
  `vram-recipe-lab/LAB_MINI_REQ_ltx25_two_stage_rerun.md`. **The ceiling
  question is still open** -- if the re-run OOMs on the real path the item
  closes, if it fits it earns a full arc.
* **Separately real:** the composite's `unsharp=0.4` was calibrated for a
  1472x832 canvas and now runs at 1920x1080 (2.31x), amplifying whatever
  texture reaches it by ~32%. Delivery-side, cheap, reversible -- an operator
  eyeball call, not a defect fix.

### COMPLETED ORDER OF WORK -- SETTLED 2026-08-20 BY FABLE + CODEX, UNANIMOUS

Operator asked for the call rather than making it: *"ask fabel what is best and
haev codex confirm."* Both lanes got ONE identical briefing carrying the four
candidates, the operator's constraints, the driver's own recommendation AND the
strongest argument against it, so neither was handed a leading question.

**BOTH PICKED `A` -- SHIP THE TWO-STAGE. BOTH PICKED `B` SECOND.**

The shared reasoning, and it is the load-bearing point: **the instrument
scandal does not touch A's evidence chain.** The broken tools (lab encoder
pinning, the template differ) underwrite B and D. A rests on matched pixels,
three independent perceptual judges, and a closed identity-drift test -- none of
which came from a scalar. Codex: *"The lab error is conservative for
production."* Fable named the risk of NOT shipping: *"an approved-but-unwired
decision is exactly where this project loses work -- the tree keeps moving and
the re-prove gets costlier every week the wire-in waits."*

**TWO ACCEPTANCE GATES WERE ADOPTED FROM THEIR OBJECTIONS. Both are binding and
they are DIFFERENT objections, which is why both lanes earned their seat.**

1. **FABLE'S COST STOP.** The `+0.037 GiB` and the `2.619x` came from the same
   lab now proven to have been measuring a machine 10.9 GiB heavier than
   production. *"'The error cuts the safe way' is an argument, not a
   measurement; broken instruments rarely have exactly one confound."*
   **If the real production cost lands materially above 2.6x, the operator's
   approval no longer covers what shipped -- STOP and return to him.** Do not
   absorb the difference silently.
2. **CODEX'S NO-OP GATE, and this one is the sharper of the two.**
   *"Three false-green tools reveal a systemic no-op-detection weakness; A could
   appear successfully wired while the refine path remains dormant or
   incomplete."* So **acceptance requires a canonical published episode in
   `otr/obs/` AND POSITIVE EVIDENCE THAT STAGE TWO EXECUTED -- never merely a
   green validator.** Count the real thing the way
   `otr_ltx25_encoder_load_audit.py` does: the LOADER'S OWN log line, never the
   adapter's claim. At minimum assert `LTXVLatentUpsampler` actually ran per
   shot and the decode happened at the upscaled canvas rather than 832x480.
   This is the section 0 `4D` failure -- a node that shipped, tested green, and
   ran dormant because it was never wired -- and it is now the exact hazard
   three false-green instruments say to expect.

**This order was executed: A, B (both halves), D, then C.** The pin was a
conformance edit with one right answer and ended the lab's false GPU-heavy text
encoder posture. D ran only after the differ became fail-closed and grounded.
C measured `media_archive` separately rather than borrowing the sibling bank's
rate; its conservative result is five dirty rows in five of 104 ledgers.

### THE ALL-VIDEO-MODELS TEMPLATE SWEEP -- GROUNDED INVENTORY DONE

**Operator's ask (2026-08-20): pull the shipped ComfyUI reference template for
every video model, diff it against what we run, A/B the differences, and let the
models judge.** His reason is the right one: *"im worried we're leaving quality
on the table and people get my repo and say why do you have all these [mutant]
video gens."* Diffomatic's own docstring makes the same argument -- *"If it was
true for the engine we scrutinise most, it is unchecked everywhere else."*

**DONE IN LAB COMMIT `38d4ae0`:** realistic per-engine fixtures, private-base
resolution, fail-closed graph completeness, value-versus-wiring separation, and
source-bound reasons. The grounded run covered every registered video engine:
30 total, 11 exact-template comparisons, 2 explicitly qualified baselines, 17
no-reference-by-design, zero errors. Each receipt binds the installed template
package/version and SHA-256 plus the OTR engine source bytes and topology.

**WHAT THE INVENTORY DOES NOT CLAIM:** a difference is not automatically a
quality improvement, and a clean source census is not the complete byte-level
decision diff or a live render. The operator's 2026-08-21 correction is binding:
finish each engine's exhaustive normalized diff before narrowing candidates;
derive an engine-local IN / ADAPT / OUT receipt rather than copying LTX 2.5's
logic; then use matched GPU evidence and native-pixel AI visual judging to decide
whether the visible payoff is worth chasing. Adoption still requires executor
evidence and a fresh `otr/obs` publication. The completed LTX lanes are not the
start point; the current-step block names the missed Z utility first and the next
installed runnable video engine after it.

### ALSO QUEUED IN THE LAB

* **FLF2V feasibility probe** -- first-and-last-frame conditioning for
  continuity across per-beat renders (operator's idea).
  `vram-recipe-lab/LAB_MINI_REQ_ltx25_flf2v.md`. Deciding WHEN to chain beats
  is deliberately out of scope: that is a design arc.
* **Diffomatic qualification** -- inventory is grounded and pushed at lab
  `38d4ae0`; the remaining work is the per-model live qualification described
  above, not another census.

### OLD ITEM I BODY (superseded, kept for the receipts)

### PREVIOUSLY: `I` -- the wrong-person character description

**SUPERSEDED HISTORICAL STATE. Item I shipped 2026-08-20 and its
`media_archive` extension closed live on 2026-08-21. Present-tense claims below
describe the pre-fix tree; they are receipts, not open instructions.**
**READ `kibitz-runs/2026-08-20-item-I-wrong-person/r3/judgment.md` FIRST** -- the
design was OVERTURNED there and starting from the item body below will send you
down the path the Bible forbids.
* The short version: Bible `11.61`, promoted FROM this bug, says *"enforce it at
  the boundary, not in the prompt"* and *"DO NOT fix it by instructing the model
  harder"*. The driver's own anchor and two review rounds all argued for exactly
  that forbidden fix, because the plan's paraphrase was read instead of the
  entry. **Open the entry.**
* Adopted shape, in order: (1) input REDACTION of known pitch names from the
  brief before `_build_user_prompt` -- `selected_concept.cast[].name` carries the
  intruder strings verbatim, and this is 11.61's own first prescription, NOT the
  banned fuzzy repair (that ban is about the OUTPUT); (2) a SOFT post-generation
  check that can never raise into `lock_cast`; (3) prompt hardening last, as
  rate reduction only.
* **Measured, so do not re-derive:** a subject-head detector anchored on
  capitalisation fires on **1,734 of 3,792 rows** -- the healthy head style is a
  Title-Case occupation. A tiered detector gives ~20 hits, ~13 true.
* **THE CENSUS DEFECT IS FIXED AND THE NUMBERS MOVED (2026-08-20).** Root cause:
  the comparison was CASE-SENSITIVE -- `wax_cylinders` pitched
  `ELIZABETH 'LIZZIE' WALSH` and the row reads `Elizabeth 'Lizzie' Walsh`, so an
  episode with BOTH dramatic rows contaminated scored clean. The instrument now
  exists as `scripts/audit_wrong_person_census.py` (NFKC -> quote/dash unify ->
  whitespace collapse -> casefold; exit 0/1/2, 2 = incomplete scan).
  **Corrected measurement: 64 hit-rows in 38 ledger files = 40 unique rows in 26
  unique episodes, against an annotated cohort of 125 files / 250 non-announcer
  rows. 61 of 64 also contaminated `portrait_prompt`.** The item's old 28/20 was
  less than half. **A token-level tier finds 4 MORE true rows (~44 total)** that
  full-string matching misses -- `"40s, EDWARDM PINCH"`, `"18-20s, 'Eddie'"` --
  plus exactly one correct relational mention (`"foil to Hiram's meticulous
  obsession"`) which must NOT be flagged, per 11.61.
* **"EVERY hit is on the `original` family" WAS WRITTEN HERE AND IT IS FALSE --
  CORRECTED BY r4 (Codex), 2026-08-20.** It is a COHORT-SELECTION ARTIFACT of the
  census, not a production fact: the instrument only considers ledgers carrying
  `selected_concept.cast`, which only the `original` family writes, so it could
  not have found a hit elsewhere however much existed. **`media_archive` has the
  identical defect with NO structured list at all** -- `ADRIAN CARRUTHERS` reads
  *"50s, Dr. Amelia Hartley, Film Historian..."* from a brief that names her
  (`as_the_hands_of_midnight_approach_20260803_015353`), and `DALE SPENDER` reads
  *"30s, passionate film archivist Dr. Amelia Hartfield"*
  (`banksweep_media_archive_20260810_234726`). Both verified at the ledger.
  **AND LAYER 2 DOES NOT COVER IT EITHER -- this line used to say layer 2 was
  "its only cover" and that is FALSE, corrected 2026-08-20 by reading the
  shipped code.** Both layers hang off the same input.
  `_upstream_identity_names` (`OTR_LedgerScriptWriter.py:1726`) reads
  `meta.source_meta.selected_concept.cast[].name` and NOTHING else, so it
  returns `[]` on `media_archive`; `superseded_identities([])` is `[]`; and
  `_enforce_name_authority` (`_otr_casting.py:1712`) opens with
  `if not superseded_names: return response, None`. Layer 3 lives inside layer
  2, so it is inert too. **`media_archive` has NO cover at all today**, and
  scoping its item as "layer 2 already half-covers it" would be scoping it
  against a guard that never runs.
* **CAST-FIRST IS A STANDING OPERATOR RULING (2026-08-20): "no cast first, cast
  must be first."** Raised because the operator himself proposed deriving
  `media_archive`'s cast from the finished script the way `scifi_news_pro` does
  -- which would make the wrong-person defect impossible by construction rather
  than caught. **Priced and then REFUSED BY HIM, and the refusal is what
  binds.** The price was small in fields (Lemmy cameo and voice-uniqueness are
  already handled on that path; the row delta is exactly ONE field,
  `speech_signature`, plus two meta stamps) but large in shape: it inverts the
  deliberate 2026-05-10 **"LEDGER-FIRST, CAST-LOCKED, OUTLINE-AFTER"** order
  (`OTR_LedgerScriptWriter.py:4095-4106`), where `generate_outline` CONSUMES the
  locked cast. **Do not re-propose script-derived casting for any lane that runs
  the writer, and do not let a panel round reopen it.** The remaining fork is a
  lane-specific identity key vs a lane-neutral one on the shared source payload. **Corrected lower bound: >= 68 row occurrences / 40 ledger files, >= 65
  copied into portraits, Aug 1-17 >= 21/37 dirty -- plus an unmeasured
  media_archive population.** The 40-rows/26-episodes figure came from a throwaway
  script, is NOT reproducible from the instrument, and must not be quoted.
* **`speech_signature` is a SECOND contaminated prose field** and it is persisted
  (`DALE SPENDER`: *"Amelia speaks in measured, deliberate tones..."*). Any guard
  that checks only `character_description` certifies half a row.
* **The roster is NOT final at the description boundary in every mode.**
  `_apply_llm_slot_fill` (`_otr_casting.py:1888`) runs AFTER the description loop
  and reassigns `row["name"]` (1656-1660). Pool mode is the default and is safe;
  `OTR_NAME_MODE=llm_slot_fill` must be explicitly fenced or the guarantee does
  not hold there.
* **The Bug Bible is WAITING for this item's test.**
  `bug_bible_regression.py:869` records that `11.61` *"has no executable assertion
  YET, deliberately"* and names verify step (6) -- the prompt builder receives
  RECONCILED text -- as the one statically checkable half, blocked until the guard
  exists. Promote it with the fix.
* **IT IS LIVE, NOT A RETIRED REGIME (operator asked 2026-08-20).** The most
  recent episode on the lane -- `rivers_embrace`, 08-17 23:30 -- has BOTH dramatic
  rows wrong. Rate ROSE: July 5/74 (7%), Aug 1-17 19/37 (51%). No commit since
  08-01 touched the two-authority prompt. **The briefs from both eras name the
  pitch cast identically**, so the July-clean episodes were clean by luck under a
  byte-identical prompt -- which is why the guarantee cannot be a prompt.
* **`casting_brief` has exactly ONE consumer** (`OTR_LedgerScriptWriter.py:4715`
  -> `lock_cast`), so redaction there is fully contained and the ledger keeps the
  unredacted brief for forensics. **But `script_brief` carries the same people in
  SHORT form** ("Jonas", "Lizzie") into the OUTLINE -- a second surface, filed
  here, not fixed by this item.
* **11.61's preferred "rewrite to the assigned names" is NOT safely available**:
  it needs a pitch->slot mapping and the only candidate is position, which shifts
  whenever LEMMY is cast (`assemble_pre_locked_rows` sets
  `remaining_open = num_characters - 1`), silently mispairing on his ~11% roll.
  So: removal or a neutral DISTINCT placeholder. A single shared token
  ("this character") is wrong -- it collapses several people into one and invites
  blended descriptions.

### OPEN, not yet scheduled

| item | state |
|---|---|
| **H-FLOOR** | **OPERATOR'S CALL, not a driver's.** Changes conditioning at cfg 4.0 and owes a render. Body below. |
| **donor voice gender -- 9 rows** | **OPERATOR'S EAR.** Root cause fixed; the rows await his listen. Never auto-flip a bank gender. |
| the four missing fingerprint recipes | OPEN. `RUNTIME_FINGERPRINT_SOURCES` has a recipe for `indextts2` only; no adapter defines `impl_version`. A real design call per engine. |
| listen page marks a resumed engine `settled` forever | OPEN. `scripts/otr_lemmy_listen_page.py:173/180/354`; `write_decisions` never overwrites. |
| PBUG-20260815-06 media_archive | **CLOSED 2026-08-19.** Selection fixed at `3be1c1e1`; the durable-headline producer wiring and live proof landed later that day. Historical receipt below. |
| PBUG-20260729-03 | STILL OPEN; this file's reachability claim for it was corrected 2026-08-19. |
| scifi_news_pro `news_read` invented names | OPEN -- retry-ladder exhaustion on a factual pass. Related in spirit to item I but a DIFFERENT mechanism (invention, not paste). |
| the five STATIC findings | OPEN, awaiting live observation. **Do NOT promote to PBUGs on the audit alone.** |
| **LTX 2.5 delivery mesh / "tiled"** | **NEW 2026-08-20, operator-reported, LOCALISED not fixed.** He saw tiling at 00:37 of `beneath_the_silvery_boughs` and asked whether the graph drifted from the lab or the untested resolution was to blame -- **it is the second.** The raw 832x480 render is CLEAN; the mesh appears in the 1920x1080 composite; the procgen/CRT blend is innocent. Recipe drift is ruled out (the lab-golden gate passes; Q3 is the deliberate lock, not drift). Which downstream op adds it -- Real-ESRGAN x2plus, the bicubic landing resize, or the 5.18 Mbit/s encode -- is NOT pinned, and **the ledger never records which upscale engine ran**, so it cannot be answered from the artifact. Next step is a two-way re-composite of shots already on disk; no LTX re-render needed. Receipt: `docs/2026-08-20-ltx25-tiled-mesh-artifact-FINDING.md`. NOT a recipe change -- the recipe is exonerated. |
| **GPU reclaim race** | **NEW 2026-08-20, pre-existing, filed not fixed.** A second request can run the destructive global reclaim while the first holds the gpu_residency lease and is sampling -- the lease is taken inside `prepare()`, after `run_episode`'s pre-render reclaim. Exists today with no cache at all. The obvious patch (take the lease inside reclaim) **DEADLOCKS**, because `render_clip` calls reclaim while already holding it and the lease is non-reentrant. Root fix is process-wide serialisation of local render/reclaim entrypoints. Own item. |

### CLOSED THIS SESSION -- do not reopen

* **LTX 2.5 encoder reload.** Episode-scoped cache, proven live: 34 renders,
  **1 encoder disk read**, 33 hits, 0 drops, 33.5% faster per render. Published.
  Bible `12.117`. Full record: `docs/2026-08-20-ltx25-encoder-cache-ARC.md`.
* **The CPU-encoder OOM fix** from the previous window -- proven end to end,
  31 renders, zero OOM, published.
* **Item `G`.** Closed by its own newer body, which the 08-15 queue header
  contradicted. `audit_voice_gender_consistency.py` reports `VIOLATIONS: 0`
  across 1,710 ledgers -- the operator's own criterion. The "34/35" it was named
  for is a portrait-prose count the audit explicitly refuses to total.

## HOW TO READ THIS FILE (lean pass 2026-08-16 evening)

**Start at THE LIVE QUEUE, at the top of this file (2026-08-20).** It is the
only authority on what is OPEN and in what order; the numbered bodies further
down are detail you read when you pick an item up, not a second to-do list.
The older "THE QUEUE, DRIVER-SET 2026-08-17" block below is SUPERSEDED as an
ordering authority -- it still says "G IS NEXT", and G was closed by its own
newer body. Keep it for its rulings, not its order.

The file was audited end to end on 2026-08-16 and it had drifted badly from its
own forward-only rule. What the audit found, so the next reader is not misled:

* **Eleven internal cross-references are already BROKEN** -- they point at
  headings that no longer exist (`section 0`, `0-BIS`, `0-TER`, `0-QUATER`,
  `0A`, `WHAT IS ACTUALLY LEFT`, `NEXT CODING QUEUE`, `STORY LAB RECOVERY
  BASE`, `ON DECK item 5`). If a pointer sends you nowhere, that is why: the
  target was removed and the pointer was not. Do not go hunting.
* **The bulk of the remaining length is DONE narrative inline inside OPEN
  sections** -- receipts, shipped-work paragraphs and superseded framings mixed
  into live items. Those receipts are all in `HANDOFF_LOG.md` and git.
  **A full archive split is OWED and is a task of its own**; it was not done
  blind here because roughly a third of these sections are standing operator
  rulings phrased as "do not re-open", and losing one costs more than the
  length does.
* **Where a heading says something SHIPPED or CLOSED, believe it and move on.**
  The value in those sections is the ruling or the trap attached to them, never
  the receipt.

**Two contradictions the audit found are now resolved in favour of the newer
statement, and both old ones are struck:** the suite receipt (this file now has
ONE, in BASELINES below), and the Lemmy-vs-render-proofs ordering (THE QUEUE
below supersedes the 2026-08-13 ruling's ordering half; that ruling's
non-ordering content still binds).

## REVIEW ROUTING FOR THE BUG-FIX WINDOW (operator 2026-08-15 -- READ FIRST)

**`kibitz` IS BACK ON AND IS TOP OF MIND.** The operator restored the panel for
this sprint and named the roster himself: **`/kibitz-plugin:kibitz`** (the
PLUGIN skill by name -- `anthropic-skills:kibitz` is the older duplicate),
**Antigravity / Flash 3.7**, and **Sonnet + Fable subagent FAN-OUT** for
diagnosis and fixes. This supersedes the 2026-08-11 suspension for this work.

**You still take your own QA, and you remain the judge:**
> *"You take your QA -- if you think you found it on your first pass, make your
> own fix and have Sonnet or Flash 3.7 QA and code and retest."*

So: a defect you are confident about on the first pass gets YOUR fix, then a
Sonnet-or-Flash QA-and-retest pass on the finished diff. Anything you are NOT
confident about goes to the panel BEFORE you write code. Ground every panel
claim against the real Windows files and discard what does not survive.

**THE ARC FOR THIS SPRINT ALREADY RAN (2026-08-15) -- do not re-run it.** Four
rounds, 14 external reviews, output =
`docs/2026-08-15-BUILD-CONTRACT-bugfix-sprint.md`. Provenance, stated precisely
because a partial campaign may never be reported as a full arc: Codex covered
all four rounds; **Antigravity covered r1 ONLY** and was quota-held
(`RESOURCE_EXHAUSTED` 429) through r2-r4, recovering only afterwards; the cloud
panel (GPT-5.6-sol, Gemini 3.1 Pro, DeepSeek-v4-pro) covered r1 and r3; Fable,
Sonnet and Haiku covered r1. Cloud spend $0.36 total.

**The panel gate for the REMAINING chunks is the diff-level one above** -- your
fix, then Sonnet/Flash QA on the finished diff -- because the design is already
panelled. Open a fresh arc only for a chunk that departs from the contract.

### THE CODEX LANE IS BACK -- VERIFIED LIVE 2026-08-19, and here is how to call it

Operator: *"CODEX 5.6 SOL READY FOR NEXT TEST WEEK, UNBLOCKED."* Verified
rather than taken on trust: `gpt-5.6-sol` answered rc=0 and self-identified as
**GPT-5.6 Codex**. The premium lane is genuinely available again, not merely
past its quota date.

**WHAT IT ACTUALLY UNBLOCKS: nothing -- and that is the point.** Per the
`CLAUDE.md` directive of 2026-08-17, *a missing reviewer never blocks the arc;
you substitute and keep going*. Item F -- the one thing this file still said
"wants the whole [arc] once Codex is back 2026-08-19" -- **already shipped on
2026-08-17** (Bible `12.110`), having run r1-r4 on two agy lanes plus a Fable
gate and a Sonnet QA pass, with Codex never sitting a single round. So Codex
returning restores the PREFERRED roster for the next full arc; it does not
release a backlog, because nothing was permitted to accumulate behind it. The
next arc that wants it is the **VIDEO SPRINT**, once the operator names its
scope.

**THREE GOTCHAS, each of which cost a window, none of them obvious:**

1. **`codex` is installed but NOT on PATH.** The executable lives at
   `C:\Users\jeffr\AppData\Local\OpenAI\Codex\bin\<hash>\codex.exe` -- the hash
   directory changes across updates, so glob for `codex.exe` under
   `AppData\Local\OpenAI\Codex\bin` rather than pinning it. `where codex`
   returns nothing, and that is NOT evidence it is missing.
2. **The installed kibitz plugin BLOCKLISTS the cheap spark model and silently
   overrides an explicit pin to `gpt-5.6-sol`.** Pinning
   `gpt-5.3-codex-spark` through the plugin does not get you the cheap lane --
   it gets you the expensive one without saying so. **Call `codex exec`
   DIRECTLY** when which lane runs actually matters:

       codex.exe exec -m gpt-5.3-codex-spark -s read-only --skip-git-repo-check -C <repo> - < prompt.md
       codex.exe exec -m gpt-5.6-sol -s read-only --skip-git-repo-check -C <repo> - < prompt.md

   `-s read-only` earns its place: it lets a review lane run CONCURRENTLY with
   the test suite with no risk of a mid-run edit (the torn-read rule).
3. **Export `KIBITZ_QUOTA_BLOCK_ON_RECENT=0`** -- the quota preflight
   false-blocks on a stale log. Related but SEPARATE: **a timeout is not a
   quota block.** `agy`'s print-timeout knob is `KIBITZ_AGY_PRINT_TIMEOUT`
   (default `5m`; `15m` fixes it), and `kibitz.py --timeout` does NOT reach agy.

**AND THE STANDING CHECK STILL APPLIES:** a lane can exit 0 having written an
empty review. Judge it by log size and whether it actually read files, never by
the exit code. The 2026-08-19 spark run wrote 104 KB and cited real line
ranges -- that is what a working lane looks like, and on that run it caught two
of the driver's own tests being tautologies.

**BASELINES (re-measured 2026-08-21 MORNING at settled HEAD `eca57cb1`, after the
mixed canonical QA):** suite **11326 passed / 114 skipped / 1 xfailed** --
MEASURED by a full run (347.95 s, EXIT=0, known-fail guard silent), never
derived. Focused/canonical tests **435 passed / 1 skipped**; Bible
**22 / 26 / 3**; `build_variants.py --check` **51 variants / 0 failures**;
canonical validator **23 nodes / 57 links**. The saved canonical Git blob remains
`c27dff3690030e78d88c3a2607a9ac54fd3935d9`. No production code changed during
this re-measurement. Entry `12.120` / `PBUG-20260820-01` still requires
model-specific approved evidence before any generic reference capability can
ship.

**THE PREVIOUS RECEIPT SAID "ZERO REGRESSIONS" AND THAT WAS NOT TRUE.** At HEAD
`55ddf234` the suite exited **2**, with
`tests/test_legacy_audit_clean.py::test_no_unclassified_legacy_references`
FAILING. Item I's new `nodes/_otr_name_authority.py:90` used *"Professor & Lab
Director"* as an example job title in a comment, and the standing legacy audit
forbids a bare `Director` in `*.py`/`*.json` outside a forensic context --
it cannot tell an example from a surviving Director-era symbol. **Why four QA
rounds missed it:** the `[KNOWN-FAIL-GUARD]` line prints AFTER the 100% line and
the conftest suppresses pytest's own summary on a failing run, so the log ends
in a wall of dots with the complaint below the fold and no counts to contradict
it. Read the exit code, not the dots. Fixed by rewording the comment to *"Lab
Supervisor"* -- a comment, so nothing the pipeline emits changes, and it does
not widen the audit's blind spot the way a `GENERIC_ENGLISH_LINES` entry would.

**PREVIOUS, and its itemisation is kept because the discipline is the point:**
**11146 / 114 / 1** (415 s, EXIT=0). Before that: **11100 / 110 / 1** at
2026-08-19. **THE +46 IS ITEMISED:** 51 tests ADDED -- 29 in
`test_ltx25_video_lane.py`, 2 in the LTX 2.5 recipe drift gate, 4 in the new
`test_upscale_weight_resolution_gate.py`, and 16 registry-walk instances that
existing parametrized tests gain simply by a new engine registering -- MINUS
about 5 parametrized instances LOST to the deliberate voice-bank shrink (the
gender pins alone went from 9 rows to 6 when glenn and james were retired).
A shrinking component is only healthy when it is itemised like this.
At that step: Bible **295** entries at HEAD `55d4eaf` -- entry `07.17` was
AMENDED, not promoted, and an amendment owes no README bump. Variants went
50 -> 51 deliberately: `otr_ltx25_high_video` is a shipping platform target
rather than a soak instrument, so it is correctly ABSENT from
`build_variants.LANE_PRESETS`.

**THAT LAST STEP IS -34 AND EVERY ONE IS ACCOUNTED FOR:** 15 tests from
`test_cloud_media_cache.py` and **19** from `test_post_freeze_writeback_audit.py`
-- 14 test functions, but one is `@parametrize`d over the 6-value music enum and
so contributed 6 instances, not 1. **The first arithmetic said 29 and the run
said 34**; the 5-test gap was that parametrize expansion, found by going and
looking rather than waving it through. No test was lost to a regression. **Measured TWICE on purpose:** an intermediate tree printed 11092
with 12 tests, then the QA pass added 2 more (rescoping two tautological
call-site tests) and the final tree printed 11094. The 11092 was briefly
written into this block before the last run -- caught and replaced, because
writing this receipt as arithmetic ahead of a run is exactly how it drifted
four times. Previous: **11080 / 110 / 1** at HEAD `741a0e11` (2026-08-18
night, after the media_archive fix). Bible **20 / 26 / 3**,
Bible holds **295** entries (`12.115` and `12.116` both promoted 2026-08-19;
`12.116` is the inert-widget class from PBUG-20260819-01, and its README bump
to 295 landed in the same commit -- survival-guide HEAD `55d4eaf3`, == origin/main). `build_variants.py --check` **50 variants /
0 failures**.

> **THE COUNT WENT DOWN, AND THAT IS CORRECT.** 11097 -> 11066 is **-31**, and
> every one is accounted for: **11** from retiring `tests/test_hybrid_voice_fit.py`
> (the pass it tested no longer exists), **8** from the four hybrid-path tests
> dropped out of the reserved-voice file, and **12** from the marker file's gate
> tests (they exercised `hybrid_voice_fit_enabled`, which was deleted). No test
> was lost to a regression. A shrinking suite is only healthy when the delta is
> itemised like this -- otherwise it is indistinguishable from coverage rot.

> **`12.114` WAS AMENDED TWICE THIS WINDOW AND MUST NOT BE RE-PROMOTED.** The
> entry count is UNCHANGED at 293 -- an amendment is not a new entry, so the
> Three-File Contract owes no README bump for it and the regression stays
> 20/26/3. Survival-guide: `b9aada7` original -> `e7179a9` (corrects the cause:
> an uncovered path, not a stale process) -> `1da7cfd` (the third path, an
> unguarded fallback fed by an empty category), HEAD == origin/main. It now
> covers all three pools; reference it, do not promote it again.

> **The `11050` in this block was correct when written and is now superseded by
> `11068`** -- 18 tests added in `3a78703e` (`tests/test_lemmy_reserved_on_hybrid_
> path.py`), and the delta equals exactly the tests added with no regressions.
> **This figure was measured twice**: once before the push and once again on the
> settled tree afterwards, both returning 11068. That is deliberate -- this
> receipt has drifted four times by being written as arithmetic ahead of a run,
> so the rule is now that the number goes in only after a run prints it.

> Chain this window: **11028** measured at open -> **11046** with the 18
> evidence-guard + citation-integrity tests -> **11050** with the 4 Lemmy
> reservation tests. Each delta equals exactly the tests added; no regressions at
> any step. **The 11050 was briefly written here as arithmetic before it was
> run** -- caught and replaced with the measured figure, because predicting this
> receipt is precisely how it drifted four times.

> **THIS BLOCK SAID `10913`. THE MEASURED PRE-CHANGE NUMBER WAS `11028` -- a
> 115-test gap, and the FOURTH drift of this "single authority" receipt in three
> days.** The `11028` figure also appears in the `HANDOFF_LOG.md` entry written
> at the same close, so the log was right and this block was stale. The evidence
> -guards window then measured `11046` after adding exactly 18 tests
> (15 guard + 3 citation-integrity), so the delta is accounted for to the test.
> **Re-measure before trusting this line**, and never read the trailing `1` as a
> failure -- it is an xfail.

> **THIS BLOCK SAID `10824` AND THE MEASURED NUMBER IS `10842` -- an 18-test gap
> this file cannot account for.** Measured on a settled tree with a docs-and-
> catalog-only change in it (item E's two pinned assets add no tests), so 10842
> is the PRE-EXISTING count, not a delta this window created. Something landed
> 18 tests without updating the receipt. **This is the second time in two days
> the single-authority receipt has drifted from reality** -- it also said 10739
> while the item B handoff said 10755. Re-measure before trusting it, and never
> read the trailing `1` as a failure: it is an xfail.

Earlier chain (10712 at the one-style-authority close
-> 10717 with the five lumina input-convention tests -> 10739 with the 22
style-traceroute VIDEO tests -> **10755 at the item B close** -> **10819 with item
C's 64 overlay tests** -> **10824 with H-receipt's 5 negative-source tests**; no
regressions at any step, and each delta equals exactly the tests added). Bible
**20 / 26 / 3**, and the Bible holds **289** entries (`12.109` promoted for
PBUG-20260817-02; **`11.61` promoted 2026-08-17 for PBUG-20260817-03**,
survival-guide `ff0eb13`; **`12.110` promoted 2026-08-17 for item F**,
survival-guide `02e8bcb`; item C and H-receipt promoted NOTHING -- see their
bodies).

> **CHECK THE DELTA, NOT YOUR RECOLLECTION.** H-receipt's write-up first said "six
> tests"; the suite delta was +5 and the delta was right. Every count in this block
> is measured, and any future row should be too.

> **THIS BLOCK SAID `10739` UNTIL 2026-08-17 AND THE ITEM B HANDOFF SAID `10755`.**
> The receipt this file calls its single authority had drifted from the log entry
> written at the same close. Corrected against a measured run. Also note the shape:
> the trailing `1` is an **xfail**, not a failure -- a window reading "10755/110/1"
> as one-test-red will go hunting for a break that does not exist.
Earlier chain: 10584 at the 08-16 close -> 10644 with the
provisional tier's 60 tests -> 10654 with the audition/artifact and three-family
integration tests plus two generator field-level pins -> 10659 with the
tier-transition matrix parametrized over both CastLock modes. Earlier: 10529 ->
10550 chunk B step 1+2 -> 10567 the TTS preflight gates -> 10584 chunk B step 3),
Bible **20 / 26 / 3**
(the Bible holds **285** entries -- `12.107` landed with chunk B; nothing was
promoted on 2026-08-17, because PBUG-20260816-04 checked out as COVERED by
`12.99` + `12.101`), variants **50 / 0** (3 refused -- the standing unratified
cloud profiles; the new `otr_lemmy_kokoro_diag` diagnostic profile emits none). (10532 -> 10561 specification session; -> 10608 chunk 0.5; -> 10610,
10613, 10624 the three agy QA rounds; -> 10633 chunk 2 / D1; -> 10657 chunk 3.5
/ D3; -> 10683 D2's transition schema + the rename fix; -> 10711 D2's emitter +
transaction; -> 10729 D4's sidecars + the P5R budget fix; -> 10732 the P5R
chunking fix; -> 10736 the sidecar re-fetch carry-forward; -> 10740 the QA
round; -> 10748 the unisex retirement; -> 10751 the given-name mention fix.
No regressions at any step.)

**A Bible entry now costs a README bump.** The survival-guide repo enforces a
Three-File Contract -- `BUG_BIBLE.yaml`'s entry count must equal the count cited
in `README.md`, in three places -- so adding an entry and not touching the
README turns `tests/bug_bible_regression.py` red (20/26/3 became 19+1F/26/3 on
`12.104` until the README said 283). Bump both in the same commit.

**THE REVIEW ORDER IS QA -> FABLE -> PUSH, AND SKIPPING THE FIRST STEP COST
REAL DEFECTS THIS WINDOW.** The routing block at the top says *"make your own
fix and have Sonnet or Flash 3.7 QA and code and retest"*. This window ran
suite -> push -> Fable -> Sonnet, i.e. both reviews landed on code that was
already on origin. Both found defects, and the CHEAP one found the worse bug:
* Fable: the P5R fix was HALF a fix -- sizing the request to the actual scene
  still dies whenever the model draws a 7-8 beat scene, which the P3 prompt
  literally invites. Cost: commit `cab94644` cleaning up `5194ab90`, plus a
  Bible amendment, where one correct commit and one entry would have done.
* Sonnet: `scripts/otr_stamp_character_genders.py:111` called `infer_gender`
  with ONE argument where it takes two and returns a TUPLE -- a plain wrong
  arity that would `TypeError` and kill the whole 65-unit run the first time a
  cast block parsed. It shipped inside a commit whose message asserted that
  tier "works". A QA pass catches that in seconds; Fable did not.
Run the QA pass on the finished diff BEFORE the push, and before spending Fable.

**NEVER EDIT A TRACKED FILE WHILE THE SUITE IS RUNNING.** Cost this window: two
phantom regressions (`test_p0_deterministic_repair_wired`,
`test_scifi_candidate_liveness`) that both pass clean on a settled tree. Several
tests AST-parse `nodes/*.py` off disk, so a mid-run edit produces a torn read
that looks exactly like a real break and costs a full re-run to disprove. Start
the suite, then keep your hands off the tree until it reports.

**THE MODEL & CREDIT BUDGET LADDER IS EMPTY.** The table at the "MODEL & CREDIT
BUDGET" heading below has a header row and a separator and **no rows under
them**, so the rungs every window is asked to cite do not exist in this
document. Windows are currently answering from the per-window mapping paragraph
beneath it. Either restore the rungs or retire the instruction; asking each
window to state a rung from an empty table is a question with no answer.

## THE 08-15 BUG-FIX SPRINT IS CLOSED -- receipts live elsewhere, rulings live here

D1/D2/D3/D4 and chunks 0/0.5/A are DONE and live-proven; the receipts are in
`docs/HANDOFF_LOG.md` and `docs/PROD_BUG_LOG.md`, the remaining contract
chunks are queue item 3. What SURVIVES here is only what still binds future
work:

* **Web search is ALLOWED at vendor time** (operator 2026-08-15, stated
  twice; the RSS precedent).
* **Gender drift is ACCEPTED** -- the deterministic ladder is the ANSWER, not
  a stepping stone. Do not loosen the decision margin: DOROTHY of Oz measures
  8/3 male under a looser estimator because her scene is crowded. A confident
  WRONG pin must stay impossible; decline-and-roll is the accepted behaviour.
  **THE FLOOR IS 8, NOT 4 -- CORRECTED 2026-08-17, and the old wording here was
  actively dangerous.** This bullet said "floor 4, ratio 3x" for as long as it has
  existed. The shipped values are `_otr_gender_pronoun_scan.SCORE_FLOOR = 8` and
  `DOMINANCE_RATIO = 3.0`, and that module's own comment reads: *"THE FLOOR WAS 4
  AND THAT WAS TOO LOW -- it shipped a confidently WRONG pin, which is the one
  outcome this module is supposed to make impossible"* (`buck_rogers` pinned "a Han
  patrol", a squad of soldiers, as FEMALE on 0 male / 4 female). So the number this
  file named as the thing not to loosen WAS the regression. A window trusting the
  doc could have "restored" 4 and called it fidelity to the ruling. Read
  `SCORE_FLOOR`, never this sentence.
* **Fidelity is owed only where a source states something.** `original` may
  ship a male JANE (operator accepted); the adaptation lanes may not ship a
  wrong AHAB. Pool mode cannot produce a mismatch (every pool name carries a
  definite tag); `llm_slot_fill` stays OFF; the deliberate lever for
  on-purpose cross-gender names is `OTR_NAME_CROSS_GENDER_RATE` (default 0).
  `unknown` behaves like the retired `unisex` in the guard -- a trapdoor
  under a door nobody opens; do not reopen.
* **DETECTOR TRAP (cost a window once):** ledger LINES key speakers by
  `char_id`, never name -- but frozen post-audio ledgers may DROP char_id on
  lines. Resolve identity from the CAST row, then match lines by whatever
  key the era carries.
* **Shakespeare gender supplement is keyed by FOLGER CODE** (`Tmp`, `MND`),
  never slug -- probing by slug reports phantom zero coverage.

### THE QUEUE, DRIVER-SET 2026-08-17 LATE (this is the sequence; the numbered
### bodies below are reference detail)

**ITEMS 1, A AND B ARE DONE; D-BIS FINDING 1 IS DECIDED AND CLOSED.** ONE STYLE
AUTHORITY shipped (1); the engine input-convention audit closed with one real hit
of three (A); the style fix is PROVEN ON LIVE PIXELS and the motion registers are
rewritten subject-first (B); the video-negative fork was panelled and answered
"build no guard" (D-BIS 1). The DONE bodies have been compressed to the rules
that still bind -- receipts live in `HANDOFF_LOG.md`.

**`C` IS DONE, BOTH HALVES (2026-08-17).** Store: ten overlays beside their
engines, 64 tests, green suite, zero behaviour change, nothing wired. Research:
nine of ten lanes got their own web lookup, and `fastwan_8gb` is explicitly marked
as inherited-only rather than implying coverage it did not get. **What is still
NOT true is that any directive has been MEASURED** -- the adoption gate stands, the
probe A/B has not run, and every PROVENANCE paragraph says so.

**`H` SPLIT AND HALF-CLOSED 2026-08-17.** H-RECEIPT shipped (panelled first, r1
scoped); **H-FLOOR is parked as an operator decision** because it changes
conditioning at cfg 4.0 and owes a render. Read H's body before costing what is
left -- item C's work reframed it and a panel corrected the driver's
execution-order claim.

**OPEN, in order: `G` IS NEXT** and it is cheaper than its queue position implies
-- PBUG-20260815-11 was **UNBLOCKED 2026-08-15 by a narrowed ruling** (body below,
around the "UNBLOCKED" heading): the mechanism is the description producer in
`_otr_casting.py` re-asking against the gender it was handed, bounded retries,
degrade-not-raise, and **node 89 gets no classifier**. Only its ACCEPTANCE needs
pixels (a live leg plus a re-run of `scripts/audit_voice_gender_consistency.py`;
34 portrait conflicts is the BEFORE number), and that batches into the operator's
declared GPU session. **`E`'s candidate downloads can run in the BACKGROUND under
anything** -- the queue already says so. ~~**`F` is not secretly cheap:** its root is
diagnosed by shape only and it wants tracing plus a panel, ideally the whole one
once Codex is back 2026-08-19.~~ **STRUCK 2026-08-19: F SHIPPED ON 08-17**
(Bible `12.110`), having run r1-r4 without Codex at all. See the Codex-lane
block near the top of this file. **D is BLOCKED** and **D-BIS 2-5 + D-TER** are
static-audit findings awaiting a live observation. B-OPEN is CLOSED. **`I` IS NEW
(2026-08-17 late) and it is a live production defect, not an audit finding** --
the wrong-person `character_description`, diagnosed at the files, promoted as
Bible `11.61`, code fix open and wanting a full arc. It is the "different bug
hiding in those rows" item G named. What follows is that order, cheapest-certain
first:

> **GIT AUTH IS FIXED (2026-08-17, item B window). Both repos are pushed.**
> The 08-17 breakage was NOT the token. `gh` was authorized the whole time
> (account `jbrick2070`, `repo` scope, token in the keyring); git was wired to
> the **Git Credential Manager**, which answers by opening a GUI dialog, so any
> non-interactive push either hung on it or died with `User cancelled dialog`.
> The `Invalid username or token` message recorded here sent the last window
> hunting a credential that was never wrong.
> **The root fix, already applied:** `gh auth setup-git`, which writes an empty
> `credential.https://github.com.helper` (resetting the inherited system-level
> `manager`) followed by the gh helper. Nothing is stored in a URL and no token
> is echoed. Both pushes then went through first try.
> **THE STANDING RULE IS UNCHANGED -- verify both are pushed before writing any
> code.** It is two `git ls-remote` calls. If a push ever fails again, read the
> failure before assuming the token: a GUI-prompting helper and a bad
> credential look nothing alike but report almost the same thing.

### QUEUE STATE AT THE 2026-08-20 CLOSE -- read this before the older blocks below

**THE LTX 2.5 ENCODER RELOAD IS FIXED AND PROVEN ON A PUBLISHED EPISODE
(2026-08-20). CLOSED.** Measured by the acceptance instrument on the real leg:
**34 shot renders, 1 encoder disk read, 33 cache hits, 0 drops, scope closed**,
`PASS`. Published as `signal_lost_a_midsummer_nights_quarrel_20260820_024524`
(1920x1080, h264+aac, 109.4 s), `RESULT SUCCESS`, `otr/obs/` 81 -> 82.
**33.5% faster per render** than the pre-cache baseline -- the leg finished 2075
seconds sooner while rendering THREE MORE shots (7674 s / 31 renders ->
5599 s / 34).
* **It also settles the r4 objection that mattered:** a structurally-live but
  unusable cached CLIP would have taken the HIT path and raised inside
  `run_graph` -- a dead render, not a slow one -- and the whole design rested on
  `detach(unpatch_all=True)` leaving a CPU-pinned GGUF CLIP re-encodable, which
  three code reads asserted and no GPU had confirmed. 33 consecutive hits with
  zero placement drops confirms it by measurement.

The lane re-read the 8.86 GiB Gemma text encoder ONCE PER SHOT -- measured on a
live canonical leg at **25 renders / 25 disk reads, ratio 1.00**. It is now
cached for the length of an EPISODE, owned by `run_episode` and released in a
`finally`. Full record: `docs/2026-08-20-ltx25-encoder-cache-ARC.md`.
* **Ownership is in the DRIVER on purpose.** The registry builds each adapter
  once at import and returns that instance forever, so an engine-owned cache
  would never end; and every engine-level hook (`free_otr_pipeline_residue` is
  per-SHOT, `teardown`/`unload` are per-BEAT) runs too often to be the release
  point. Do not "simplify" this back onto the adapter.
* **The acceptance instrument is `scripts/otr_ltx25_encoder_load_audit.py`**,
  and it exists because the failure mode is SILENT: `reclaim_idle_models` runs
  every shot and detaches the cached CLIP, and the liveness check degrades to a
  full reload -- correct, safe, and indistinguishable from success. The audit
  counts the LOADER's own GGUF line, never the adapter's claim about itself.
  Proven non-tautological: it FAILS the pre-cache leg (25/25, exit 1).
* **The leg that proved it booted BEFORE the r4 ownership fixes** (generation
  token, lock, per-scope audit). So it proves the CORE mechanism -- a reclaimed
  CLIP survives and is reused for a whole episode -- and NOT the concurrency and
  kill-switch edge cases, which one sequential episode cannot exercise. Those
  rest on unit coverage.
* **FILED, NOT FIXED (r4 Codex, pre-existing and unrelated):** a second request
  can run the destructive global reclaim while the first holds the gpu_residency
  lease and is sampling -- the lease is taken inside `prepare()`, after
  `run_episode`'s pre-render reclaim. It exists today with no cache at all, and
  the obvious patch (take the lease inside reclaim) would DEADLOCK because
  `render_clip` calls reclaim while already holding it. Own item.

**`G` IS CLOSED -- DO NOT BUILD THE RE-ASK, and the QUEUE HEADER ABOVE IS STALE
ABOUT IT.** The 2026-08-15 header says "G IS NEXT"; G's own body (2026-08-17,
newer, around line 1169) says the re-ask **should not be built** --
`scripts/audit_voice_gender_consistency.py` reports `VIOLATIONS: 0` across all
1,710 ledgers, which is the operator's own stated criterion, and the "34/35"
the item was named for is a portrait-prose count the audit explicitly refuses to
total. Believe the newer statement. **The next real coding item is `I`.**

### RULINGS LEDGER FROM THE 2026-08-18 CLOSE -- NOT a live list, despite what
### this heading used to say. 928 lines of standing rulings and the receipts
### that justify them. The live list is THE LIVE QUEUE at the top of this file.

**THE TITLE/IDENTITY FAMILY IS CLOSED AND PROVEN ON PIXELS.** PBUG-05 fixed,
item I bisected, PBUG-04 built, title-provenance spec answered, and the GPU
bank gate ran **4/4 banks PASS, 0 failed acceptance checks**, all published to
`otr/obs/`: shakespeare announced *"a scene from Romeo and Juliet, by William
Shakespeare, Act Two, Scene Two"*, public_domain *"The Jungle Book, by Rudyard
Kipling"*. Instrument: `scripts/otr_title_identity_acceptance.py`.

**THE VOICE IDENTITY FIX IS BUILT, SHIPPED AND RE-QUALIFIED (2026-08-18).**
Spec `docs/2026-08-18-voice-identity-fix-ANCHOR.md`; the emotion-ceiling
follow-up is `docs/2026-08-18-emotion-mass-single-knob/`.

**THE EMOTION BLEND IS NOW ONE KNOB, AND THE OPERATOR SET ITS VALUE BY EAR.**
He heard the log-odds ladder (`otr/episodes/lemmy_emotion_ladder_logodds_2026-08-18/`,
alpha pinned at 1.0, only the ceiling varying) and ruled: *"IF I WERE A KID I'D
LIKE MORE BUT AS AN ADULT ARM0P560 IS PERFECT."* So `EMO_ALPHA_DEFAULT` went
0.4 -> **1.0** (a pass-through, kept only as a diagnostic override) and
`EFFECTIVE_EMOTION_MASS_CAP` 0.4 -> **0.56**. Alpha binds BEFORE the ceiling, so
shipping the ceiling alone would have delivered 0.400 on a neutral line and
0.374 on the emotional one -- not the rung he approved. Measured on the shipped
build: **0.5600 / 0.5590 / 0.5600**, speaker retained 0.4400-0.4410.

**CLOSED 2026-08-18 ON A BLINDED LISTEN. DO NOT RE-OPEN IT.** The neutral-line
question was the one real gap, and it was answered: three arms, blinded, at
`otr/episodes/lemmy_production_audition_ceiling_2026-08-18/`. He picked armZ /
armY / armY. **The seed fix won 3-0 cleanly** -- armX and armZ carry identical
emotion blends, so the only variable between them is the seed policy and armX
lost every line. **The ceiling won 2-1**, and he settled the odd line himself:
*"we dont need to make it six sigma you know I dont like the real emo ones"* --
armZ is the uncapped arm, so line 1 was noise. **0.560 is final.** No further
ladder, no tie-break, no per-line-shape ceiling. This is a settled taste call,
not an open measurement.

| item | state |
|---|---|
| **voice identity fix (PBUG-20260817-09)** | **DONE -- shipped, re-qualified as `prod-audition-2026-08-18`, route `...-cockney-v2`, fingerprint `9bee950a7920fd00`** |
| **emotion ceiling 0.560** | **DONE AND CLOSED** -- neutral lines heard blind 2026-08-18; seed fix 3-0, ceiling 2-1, he settled the odd line (*"I dont like the real emo ones"*). Do not re-open |
| **live episode on the new voice build** | **DONE** -- `signal_lost_the_searing_relay_20260818_094723`
  in `otr/obs/`. 20/21 character lines through indextts2, alpha 1.0, mass 0.556-0.560
  every time; both characters held ONE seed across all their lines, zero drift |
| **PBUG-20260818-01: scifi_news_pro closing segment** | **FIXED AND LIVE-VERIFIED**
  (`signal_lost_the_last_reading_20260818_122159`, published). Took 4 render iterations,
  2 of which disproved the diagnosis before it -- full story in PROD_BUG_LOG. Root cause
  (found by a scoped kibitz round, not solo): the script writer's own prompt dumped the
  already-generated real-world fact into its context while telling it never to state one --
  a small local model does not reliably resist a fact sitting in its own context. Fixed by
  excluding `news_close_read` from that one prompt-build call. `media_archive` got the same
  "who found it + end on a thought" shape but is NOT yet live-verified |
| cross-engine audition overwrite guard | **DONE 2026-08-18** -- and verify step 3 grew it from one script to three. Shared guard `scripts/_otr_evidence_citations.py` walks the LIVE `LEMMY_VOICE_POLICY` for every sha256 and refuses to overwrite any file the ledger cites; no flag can override it. `--out-dir` on the cross-engine audition, the same guard on G1 with `--overwrite` REMOVED (zero callers) and its manifest/KEY writes made atomic, `--campaign-dir` on the listen page. **The unit of immutability is the cited BYTES, not the directory** -- the campaign dir is a shared workspace (the listen page writes `LISTEN.html` into it every run), so the production audition's blanket non-empty refusal would have been wrong here. Proven live: `--render` against the cited dir exits 2 naming all seven files with zero mtimes touched, and `--engine bark` alone still refuses on the shared manifest |
| **evidence citation ROT DETECTION** | **DONE 2026-08-18, and it was the highest-value half.** Detection existed where stakes were LOWEST and was absent where highest: the provisional routes had a byte-level check, the **QUALIFIED** route's manifest had NO on-disk check anywhere, and the superseded G1 record had only a config-literal assertion while its docstring claimed the file still hashed. `tests/test_evidence_citation_integrity.py` now re-hashes every cited artifact across all three tiers, with a coverage tripwire that fails if a new citation shape appears unenumerated. **A resolved episodes tree with a missing cited artifact now FAILS instead of silently skipping** |
| the four missing fingerprint recipes | **OPEN, filed out of the guards window.** `RUNTIME_FINGERPRINT_SOURCES` (`nodes/_otr_voice_route.py:158-165`) has a recipe for `indextts2` ONLY, and **no adapter in `nodes/_otr_audio_engines/` defines `impl_version` at all** -- so every row of the cited cross-engine manifest carries `engine_impl_version: ""`, an evidence-shaped field that has never been fillable. The guards window made it honest (records the real fingerprint when a recipe exists, an explicit "no fingerprint recipe registered" when not) rather than silently blank. Writing recipes for bark/kokoro/chatterbox/dia is a real design call -- which source files constitute each engine's build -- and deserves its own consideration, not a ride on a guard change |
| listen page marks a resumed engine `settled` forever | **OPEN, filed out of the guards window.** `scripts/otr_lemmy_listen_page.py:173` sets state `missing` with no clips, `:180` sets `decidable: bool(clips)`, `:354` writes `decision: settled` when not decidable, and `write_decisions` (`:335-337`) never overwrites. So an engine rendered AFTER a page build stays `settled` and is never listened to. Real, confirmed, and deliberately not folded into the guard commit |
| ~~voice-pool concentration~~ **CLOSED 2026-08-19 -- do not re-open.** Flip + live proof + rip all shipped (`429b73aa`, `eb264989`). Receipts in `HANDOFF_LOG.md`; the arc is `kibitz-runs/2026-08-18-voice-pool-concentration/`. **WHAT STILL BINDS:** (1) the deterministic scorer is the ONLY caster -- `tests/test_voice_cast_mode_marker.py::test_the_hybrid_voice_fit_is_gone_and_stays_gone` fails if the pass returns; (2) `meta.voice_cast_mode` is the provenance marker, and it must be COPIED BY NAME at `OTR_LedgerScriptWriter`'s key-by-key meta copy or it never reaches the ledger -- that copy is fail-closed on purpose; (3) `scripts/otr_verify_voice_cast_mode.py` is the acceptance gate and carries four replay pins (literal `char_voice` role, `canonical_bank_gender`, used-set in cast-row order, default draw knobs). **STILL UNPROVEN: the concentration NUMBER.** Top-2 42% -> ~9% is a CORPUS measurement; one leg proved mechanism only. Re-measure over `output/otr/episodes` once post-flip episodes accumulate. **COUNTED 2026-08-19: only 3 post-flip episodes exist** (the arc's leg plus the two title/headline acceptance legs) against 1,710 pre-flip -- roughly 6 voice draws, which is noise, not a distribution. **Deliberately NOT measured**: publishing a top-2 number off 6 draws is exactly the premature-receipt failure this file has suffered four times. Re-check the count before re-measuring; a few dozen post-flip episodes is the honest threshold |
proven on pixels.** Four rounds (r1 Fable-cold reversed the design: remove, don't
tune, because the prompt gives the model only the four fields `_score` already
weights and no character name -- so no judgment is available to it). Shipped
`429b73aa`: explicit opt-in parse (the old `!= "0"` would have read `""` and
`"false"` as ENABLED once the default flipped), plus a POSITIVE
`meta.voice_cast_mode` marker, because `voice_cast_decision == {}` is produced
BOTH by "disabled" and by "enabled but no engine resolved". **Proven on
`signal_lost_the_bite_of_the_iron_chain_20260818_210937`** (RESULT SUCCESS,
1920x1080, 2:48, in `otr/obs/`): marker `scorer`, decision `{}`, and a scorer
replay matching **2/2** character rows via `scripts/otr_verify_voice_cast_mode.py`
(four pins: literal `char_voice` role, `canonical_bank_gender`, used-set rebuilt
in cast-row order, draw knobs at defaults). **THE RESULT WORTH KEEPING: both
voices drawn -- `vz_pd_librivox_mark_f_smith` and `vz_donor_sujan_daikoawaj` --
were STRUCTURALLY UNREACHABLE before**, never offered on the 12-card alphabetical
list. The 42%-unreachable finding appeared on the FIRST post-flip episode. The
gate also has a negative control: it correctly FAILS the pre-flip
`signal_lost_the_16mm_ransom`, where the replay disagrees with what was stamped.
**STEP 3 IS DONE TOO (`eb264989`, operator called it: "rip the dead code now").**
The pass is deleted from `_otr_casting.py`, `_otr_voice_bank.py` and
`cast_lock.py` -- including CastLock's OWN local import of
`validate_voice_proposal`, the easy miss that would have left a dangling
`ImportError`. `default_char_engine` was deliberately kept (a review lane listed
it for deletion; `test_google_tts_voice_pool.py` uses it independently).
`test_lemmy_reserved_on_hybrid_path.py` got a surgical edit and is renamed
`test_lemmy_reserved_voice_pools.py` -- it guards the unrelated reserved-voice
fix, and two of its four guard sites went with the ripped code, leaving the two
pools that can still draw. `voice_cast_decision` is kept and stamped EMPTY so
CastLock's `or {}` keeps one stable shape and legacy ledgers still load.
**New guard against reintroduction:** `test_the_hybrid_voice_fit_is_gone_and_stays_gone`
fails if any of the six ripped symbols returns. 385 net lines removed.
**THE ONE CLAIM THAT IS STILL NOT PROVEN: the concentration itself.** One leg
proves MECHANISM. Top-2 42% -> ~9% stays a CORPUS measurement until post-flip
episodes accumulate -- re-run the distribution over `output/otr/episodes` once
there are enough, and do not report the single leg as if it settled it.
~~**STILL A DESIGN FORK, STILL WANTS THE PANEL**~~ -- rank cards by the scorer instead of alphabet, raise/remove `max_cards`, seed-rotate the shortlist, track recent use, or drop the hybrid pass (that last one is governed by the `CLAUDE.md` ledger rule: `meta.voice_cast_decision` has nine fields consumed at `cast_lock.py:688`). All CPU-measurable |
| **donor voice gender -- 9 rows await the operator's EAR** | **OPEN, and it is his call, not a code call.** Root cause fixed: `scripts/otr_dl_indextts2_refs.py` inferred gender from median F0 at a 165 Hz threshold inside the male/female overlap. `gender_from_handle()` now makes the NAME authoritative per his ruling -- *"if we [don't] have real gender info use the name not some pitch"* -- and glenn measured **261.5 Hz** (confidently female by pitch, male by ear), so pitch fails even when confident. `james`/`hillbilly_jim` trios flipped and pinned; **`rup` trio unresolved** (the name decides nothing). `tests/test_voice_bank_gender_pins.py` pins every settled row and tripwires any handle/gender contradiction. **NEVER auto-flip a bank gender** -- flag and let him listen |
| PBUG-04 residue | HALF closed -- announcer names the real work, can still embellish in sentence 2 |
| PBUG-20260817-06 | Doyle names spoken in a Leacock parody; undiagnosed |
| PBUG-20260815-06 media_archive feed selection | **SELECTION FIXED 2026-08-19 (`3be1c1e1`); OPEN for the durable-headline half.** The lane adapted the newest post forever (index defaulted to 0, feeds are newest-first, zero tests on the path). Now reuses the SCIENCE lane's shared news history rather than growing a second one -- the operator's framing: both RSS lanes should share the logic. 14 tests. Bible **12.115**. **CLOSED 2026-08-19 -- the durable-headline half landed too.** One line of production code, and the reason it sat open is the interesting part: the CONSUMER was built first and the PRODUCER was never wired. `identity_from_meta` has always read `source_meta["post_headline"]` for this lane and `is_degraded` is True exactly when it is missing, while `_rss_source_fetch_result` never stamped it -- so every media_archive episode carried a degraded identity, silently, measured True at HEAD before the fix. A test fixture even documented the intended key set including `post_headline` while the producer did not emit it. Stamped at selection time for BOTH RSS lanes (no per-lane branch in a shared helper). 10 tests. **The full suite caught a regression the QA lane missed** -- an EXACT-equality `source_meta` pin in `test_source_payload_chunk3.py`; expectation updated, not loosened. **PROVEN ON PIXELS**: `signal_lost_reel_of_shadows_20260819_061004` carries `source_meta.post_headline == "This Thursday (7:00 PM August 20) at the Mary Pickford Theater (Washington, DC)"` in durable meta -- an episode can name the post it adapted, from its own frozen ledger |
| PBUG-20260729-03 | **STILL OPEN, but this row's REACHABILITY CLAIM IS STALE -- corrected 2026-08-19.** The defect is real and unchanged: `_otr_structured_call.py:1300-1305` raises a bare `ValueError` -- NOT in `_ATTEMPT_ERRORS`, so it bypasses both the ladder-advance fix from `41683fc9` and the terminal `StructuredCallFailedError` its callers catch -- when the repair context exceeds `repair_context_max_bytes`. Same shape at `_otr_scifi_p0_contract.py:436`. **WHAT IS NOT TRUE ANY MORE: "Both reachable through the live `_otr_scifi_news_pro.py`".** Measured at HEAD `48f339ba` across the whole tree, excluding tests, `tmp/` and the stale `.claude/worktrees/` copies: **ZERO production callers pass `repair_slot_fn` / `repair_ledger_builder`** (only `tests/test_structured_call.py` does), so the alternate-owner branch holding the `ValueError` is unreachable from production; and **`compact_p0_repair_context` has ZERO callers anywhere** but its own definition. The `_otr_scifi_codex.py` rip took the only production caller of both with it. `news_pro` imports only `MAX_QUOTE_CHARS` + `p0_source_chunks` from that module and passes `repair_prompt_factory` (the PRIMARY ladder), never the alternate owner. **So this is no longer a live production defect -- it is shared-infrastructure hardening**, and it should be costed and prioritised as such. It still deserves a fix: `structured_call` is a public API whose contract is `StructuredCallFailedError`, a lane MAY pass a repair owner, and a bare `ValueError` escaping it is a landmine for the next caller. Do NOT report fixing it as closing a live bug |
| PBUG-20260815-05 | **FIXED 2026-08-19, unit-proven, OWES A LIVE LEG.** `_generate_title_from_script` grew a keyword-only `work_title`, threaded from the J.5 call site through the EXISTING `_otr_source_identity.identity_from_meta` authority -- no second reader grown, gated on `ADAPTATION_SOURCE_KINDS`. **THE FIX IS AN ANCHOR, NOT A GUARD**, per the operator's mid-window ruling (*"dont waste too much time overengineering for hard to replicate bugs im accepting some level of story quirks since a new story is gen every time"*): the code-side sibling-title reject specified in `PROD_BUG_LOG.md` was deliberately NOT built, because substring containment rejects legitimate titles and no sound matching rule existed. The anchor ships WITH the rule that keeps the work name out of the title, so it cannot collapse every adaptation episode into "The <Play> Something". 14 tests. **PROVEN ON PIXELS the same day** -- bank gate 2/2 PASS, both published to `otr/obs/`. shakespeare `signal_lost_under_the_enchanted_moon_20260819_062006` adapted A Midsummer Night's Dream, stamped `title_work_anchor == "A Midsummer Night's Dream"`, and titled itself **"Under the Enchanted Moon"** -- names no other play, and did NOT collapse into the play name, which is the failure mode the anchor design was shaped to avoid. media_archive `signal_lost_reel_of_shadows_20260819_061004` is the negative control: anchor `""` with the key PRESENT, the lane gate correctly refusing to anchor a title to the publication "Now See Hear!" |
| PBUG-20260817-08 | **CLOSED 2026-08-18 -- all three pools.** The 08-17 fix guarded only `assign_voice_for_slot` (~4% of casting); the hybrid card list and `gender_agnostic_fallback_ref` both leaked. Bible `12.114` covers it (amended twice). **WHAT BINDS:** `tests/test_lemmy_reserved_voice_pools.py` is ONE file on purpose -- extend its all-pools assertion if a third pool appears, do not write a fourth file |
| PBUG-20260817-07 | stage directions in captions -- **WILL-NOT-FIX** (operator ruling) |
| stale PBUG sweep | **DONE 2026-08-18. 35 open -> 18**, all 18 closures carrying citable evidence with previous status preserved. **THE TRAP WORTH KEEPING:** two of the July-11 block looked like siblings of a deleted module but their fixes had GENERALIZED into shared infrastructure every live lane still calls (`_otr_json.extract_first_json_block`, `_otr_structured_call.schema_required_paths`) -- a bulk close would have wrongly killed them. The remaining 18 are correctly open, several by operator ruling |
| ~~**FOUR FILES ARE LANDMINES**~~ **FIXED 2026-08-19 -- now THREE files, and the noisy one is out** | **DONE, proven on the product.** The fix was NOT the driver's idea (hash the AST) and NOT the panel's majority (stop demoting). It came from the codex-spark lane: **the problem was the RECIPE, not the mechanism.** `nodes/_otr_voice_node_common.py` -- the shared per-line dispatcher, **19 commits in 60 days** -- was removed from `RUNTIME_FINGERPRINT_SOURCES["indextts2"]`. Nothing else changed: same raw-byte hashing, same demote-on-mismatch, no test inversions of the gate's semantics, no weakening of what "qualified" MEANS. **THE MEASUREMENT THAT DECIDED IT:** of those 19 commits exactly **ONE** touched the seed path that was the stated reason for including the file (`62fb6a1f`, the voice-identity fix) -- so the whole-file hash was producing **18 false demotions and 1 true one** -- and `62fb6a1f` **also edited `eng_indextts2.py`, which stays in the recipe**, so narrowing loses nothing on the only real event in the window. Union churn 22/60d -> 8/60d. **PROVEN ON THE PRODUCT, not just in tests:** appending a comment to the shared dispatcher -- the exact edit that de-qualified Lemmy that morning -- now leaves the route selected; pinned by `test_editing_the_shared_dispatcher_no_longer_costs_the_voice`. **THE STORED FINGERPRINT MOVED AND THAT NEEDS SAYING PLAINLY:** narrowing the recipe changes the computed value (`9bee950a7920fd00` -> `d47779386ce91209`), so the live value at `cast_pools.py:847` was re-expressed. The code says "do not hand-edit the value to silence the warning" and that rule is respected: this is not a silenced drift -- the rendering code is byte-identical to what was approved, only the DEFINITION of which files count changed. **The historical audition receipt at `:882` was deliberately NOT touched** -- it records what was true at audition time under the 4-file recipe, and re-expressing it would falsify history. **RESIDUAL RISK, accepted and written into the code:** a seed-path change touching no engine-specific file would now escape this fingerprint. It did not occur once in 60 days. `weight_revision` and `reference.source_ref_sha256` still gate independently |
| ~~the landmine's original framing~~ | **superseded, kept only for the trap it records.** `_otr_voice_route.RUNTIME_FINGERPRINT_SOURCES["indextts2"]` hashes **whole-file bytes** -- comments and docstrings included, only CRLF normalised -- of four files: `nodes/_otr_audio_engines/eng_indextts2.py`, `scripts/_otr_indextts2_worker.py`, **`nodes/_otr_voice_node_common.py`**, `nodes/_otr_resolved_request.py`. Any byte change withdraws Lemmy's qualified route (`lemmy-indextts2-algenib-cockney-v2`) and every episode casting him falls back to the ordinary unqualified draw. **PROVEN BY ACCIDENT:** adding ONE comment block and ONE `log.warning` to `_otr_voice_node_common.py` -- purely additive, zero logic touched -- turned 4 tests red across `test_voice_identity_fix.py` and `test_cast_lock_policy_repin.py`. The edit was reverted and the route re-qualified (live == stored == `9bee950a7920fd00`). **WHY THIS IS NOT A CURIOSITY: `_otr_voice_node_common.py` was edited 19 times in 60 days** -- about every third day -- because it is shared dispatch code for the whole voice-node family. This trap is armed continuously. **AND THERE IS NO SIGN ON IT:** zero mentions of the fingerprint in any of the four files, and zero in `EXTENDING_OTR.md` / `conventions.md` / `TTS_VOICE_PREFLIGHT.md`. The only human-readable note is one buried paragraph in `HANDOFF_LOG.md:795-797`. **THE SIGN CANNOT GO AT THE POINT OF CONTACT** -- a warning comment in those files would itself move the fingerprint, which is the joke and also the constraint. So it lives here, in `tests/test_stale_ledger_voice_guard_removed.py`, and nowhere else until someone pays a deliberate re-audition to add all four signs at once. **TWO MORE FACTS THAT MATTER:** the mismatch DEMOTES and never raises (THE LAW, deliberate -- so it is silent apart from one log line); and `_LIVE_FINGERPRINT_CACHE` is never cleared in production, so a mid-session edit does not bite until the server restarts -- which means a developer can edit, test green in-process, and ship a de-qualified route. **THE FORK WAS PANELLED 2026-08-19 AND THE ANSWER IS NOT "HASH THE AST" -- see `kibitz-runs/2026-08-19-fingerprint-fork/JUDGMENT.md`.** The driver's own lean (hash normalised code) was REFUTED by both lanes and by direct experiment: the incident that started this added a `log.warning`, which is a new AST statement, so an AST hash trips on it too; replaying all 44 commits that ever touched the four files, raw-byte changed 44/44 and AST-normalised changed **43/44** -- it buys ONE commit in forty-four. `ast.dump` also KEEPS docstrings, so the naive recipe does not even solve the comment case without an explicit stripping walk -- new machinery whose own bugs would present as losing Lemmy's voice, which is precisely what the operator said he does not want to own. **THE DEFECT IS NOT WHAT IS HASHED, IT IS WHAT HAPPENS ON MISMATCH:** the gate withdraws the approved voice and draws an ordinary one, i.e. it causes with certainty the harm it exists to prevent. **RULING (adopted, NOT yet implemented -- awaiting the operator's go):** drop the two demotion call sites (`_otr_voice_route.py:547` and `:1163`, verified the only two in production, ~15 lines); KEEP `live_engine_impl_version` computing and stamping, because it has a SECOND consumer as an audio cache-key field (`_otr_resolved_request.py:84,118,260,304`) and removing it would break separate caching; KEEP `weight_revision` and `reference.source_ref_sha256` gating untouched, since those hash the real weights and the real audio and change only by deliberate act; and stamp drift durably in episode meta (`approved_under` vs `rendered_under`) rather than only logging, because 126 files share the `OTR` logger and a warning is easy to miss. **ROSTER, STATED EXACTLY: Fable ran r1 cold; Codex gpt-5.6-sol hit its usage limit mid-run and contributed NOTHING; a Sonnet subagent filled the Codex seat. Two lanes, one a substitute -- not a full arc, and not to be described as one** |
| **the stale-ledger voice guard: DELETED, and the thing it guarded is CLOSED LEGACY DEBT** | **DONE, and the follow-up is deliberately NOT queued.** Operator voted delete. Correct, and for a better reason than "unused": the guard **RAISED** -- it refused to render a stale ledger, the PBUG-20260729-03 shape THE LAW forbids. `CacheMigrationError` went with it (orphaned). **THEN A SONNET FAN-OUT DATED THE DEFECT AND IT CHANGED THE PRIORITY.** The 51 affected ledgers are a CLOSED WINDOW, not a drip: **before 2026-08-04, 51 of 514 cast-locked episodes affected (9.9%); since 2026-08-04, 0 of 169 (0.0%)**. Driver re-measured independently and got the same 514/51 and 169/0. Newest affected episode is 2026-08-03. The cause was fixed on 2026-08-05 (`7e4a4c3c`, `f5a5d174`): CastLock used to catch `VoiceCastingError` and `continue`, leaving the writer's raw row untouched; it now stamps `gender_agnostic_fallback_ref(...)` so every row leaves with an id. Guarded by `tests/test_cast_lock_voice_ref_completeness.py`, **11/11 green on the current tree** even after the 08-18 hybrid rip touched the same function. All 51 carry `cast_lock_revision: 1` -- never re-locked, frozen artifacts. **CORROBORATING DETAIL WORTH KEEPING:** 0 of 71 missing rows are ANNOUNCER; no episode lost its whole cast (38 lost exactly one row); the missing rows are raw writer output carrying `voice_preset`/`gender`/`tts_model=bark` but with `voice_ref_id` KEY-ABSENT, which is exactly the pre-fix `continue` path's signature; and the per-bank concentration is a TIME artifact -- `media_archive`, `scifi_news` and `original` all kept running hard after the cutoff with zero recurrence. **SO THE MISSING WARNING IS LOW VALUE:** it would make a degrade audible for 51 frozen historical episodes that nothing is re-rendering, at the cost of de-qualifying Lemmy's route (row above). Reverting it was right. Do not re-open this unless someone actually starts replaying pre-08-04 ledgers |
| **LTX 2.5 -- CHUNK A SHIPPED AND LIVE-PROVEN AT THE SHOT LEVEL; THE EPISODE LEG IS THE OPEN QUESTION** | **THE ENCODER RELOAD IS DONE (2026-08-20) -- this cell's prescription below is SUPERSEDED and is kept only as the BEFORE picture.** What actually shipped is EPISODE-scoped, owned by `run_episode`, not an unbounded cache on the engine instance: the registry returns one adapter forever, so an engine-owned cache would never be released. The key is stronger than `_weight_receipt` too (realpath + st_dev + st_ino + size + mtime + placement), because that receipt answers `(basename, -1, -1)` for TWO different broken states and they compare EQUAL. The per-beat conditioning idea below was also re-scoped: the empty negative is cached per EPISODE, and the positive is never cached because it changes every beat. See `docs/2026-08-20-ltx25-encoder-cache-ARC.md`. The original wording follows. `eng_ltx25.py` ships `ltx25_video` / public `ltx25_high_video` with `config/profiles/otr_ltx25_high_video.json` (the profile is what WIRES it -- engine picks are managed widgets, so a lane without one cannot be driven by the canonical graph at all). G8 passed twice, ffprobe-verified. **THE COST PROBLEM: the lane RELOADS the 8.86 GiB Gemma encoder from disk on EVERY beat -- 9 loads for 9 beats on the live leg, counted.** That is ~63 s per shot on top of a 54.2 s CPU encode, and it is why a shot went from ~97 s to ~214 s. The box has **63.4 GB RAM**, so caching the loaded CLIP on the ENGINE INSTANCE keyed by the existing `_weight_receipt` (basename/size/mtime) is the single biggest throughput win available and costs only system RAM. **DO THAT FIRST** -- it roughly halves the wall clock and the operator judges progress by how many episodes he sees in `otr/obs/`. **SECOND, and only after:** cache the CONDITIONING per beat. The empty negative is identical for every shot in every episode, so it is the bigger half; the positive is identical only across a beat's own segments. `wrapper_bridge.run_graph` already takes `external_results` and `BeatSession` already spans a beat's segments, so the mechanism exists -- and `stash@{0}` holds a working encode/sample graph partition from this session that was reverted for a WRONG reason (it was built to dodge a co-residency that does not exist) but whose SHAPE is what conditioning-caching needs. Read that stash before rebuilding it. **DO NOT re-attempt:** `free_after_use` tuning, load-order phase splits for VRAM reasons, or deleting the negative encode -- all three were tried or proposed and all three are disproven in the file's own comments. |
| **LTX 2.5 -- WHAT IS STILL UNQUALIFIED, so nothing reads as proven that is not** | **The lane has NO cost row, NO envelope key, and declares NO `compatible_boot_contracts`** -- the evidence manifest records it as admission-unenforced in words, deliberately. **The in-episode sampling peak is 15744-15990 MB on a 16303 MB card**, i.e. ~350 MB of headroom and ~600 MB above the lab's 14.48 GiB; that is a steady cost, not the transient encode race that was fixed. Whether that is fit for production is an OPERATOR call and the manifest says so. **Also unresolved and NOT a Chunk A defect:** `otr_silent_composite` scales with `force_original_aspect_ratio=decrease` then pads black, so 832x480 (26:15) lands 1442x832 inside the 1472x832 canvas and takes ~15px bars each side. `wan_ti2v` -- the default WAN lane -- and `humo_14B_169` pillarbox identically TODAY; the 2026-07-26 "Pillarbox: never" ruling bound only the three 8GB lanes. Its named remedy (crop to 832x468 at delivery) is a composite change touching every lane. Reported, not actioned. |
| **LTX 2.5 CHUNK B -- the foley bed. STILL BLOCKED, and the block is real** | **UNCHANGED by Chunk A shipping.** Video renders FOUR topological stages AFTER the master freezes (`OTR_EpisodeAssembler` order 12 vs `OTR_VideoRenderBatch` order 16), so the foley does not exist when the master is assembled. Needs an execution-order change shared with the deferred mime work, and its own FULL arc before code. `ltx25_mime` / `ltx25_foley_plus` are RESERVED in `eng_ltx25.LTX25_RESERVED_SIBLING_IDS` and deliberately unregistered -- per the operator, a dropdown row that cannot make an episode is worse than a missing one. **One fact that de-risks it:** the golden recipe ALREADY wires `LTXVAudioVAEDecode`, so Chunk A SUBTRACTS it; Chunk B is not "add a decoder", it is getting decoded audio into the master before the freeze. |
| **THE SOAK ROTATES ONLY CHEAP LANES -- neither upscaler is ever exercised** | **NEW, and it means the soak does not test what ships.** `scripts/otr_gpu_soak_matrix.py` rotates 5 banks x 10 styles x 10 profiles, but **every one of those profiles is a still/procgen lane**: no heavy video model (`ltx25_video`, `ltx_video`, `ltx_8gb`, `wan_ti2v`, `humo*`, `minimax_h3_*`) is in the rotation, and **no profile carries an `upscale_stage`, so neither `off` nor `spandrel_esrgan` is ever run.** Its docstring claimed engines were not rotated at all; that was STALE and is corrected. Widening it means AUTHORING PROFILES -- `upscale_stage.engine` is a MANAGED widget (`config/profiles/widget_mapping.json` -> `OTR_SilentComposite.upscale_engine`) so `patch_creative` refuses it and a profile's `role_overrides` is the only sanctioned lever. The harness itself needs NO change to accept them. Soak profiles go in `build_variants.LANE_PRESETS` so they never emit a shipping variant. **Pin each profile's `render.canvas_w/h` to the engine's DECLARED canvas** or G2.3 goes red -- that is the gate working, not an obstacle. |
**AND THE VRAM PICTURE IS NOT WHAT THE DRIVER ASSUMED -- CORRECTED BY THE LAB.** The 14.48 GiB peak was measured with the text encoder ALREADY FREED: ComfyUI's `model_management` evicts Gemma to system RAM before `SamplerCustomAdvanced` runs. Breakdown: **~9.80 GiB DiT weights (Q3 22B) + ~3.20 GiB spatial-temporal activations (97 frames at 832x480) + ~1.48 GiB PyTorch context/allocator = 14.48**, with text encoder and VAE both at 0.0. **SO AGGRESSIVE LOAD/FREE STAGING DOES NOT REDUCE THIS PEAK** -- the encoder eviction is already automatic, and ~13.0 GiB of it is weights plus activations, a hard floor. The driver had framed staging as the thing that makes it fit; it is not. Staging remains correct HYGIENE (`free_otr_pipeline_residue` clears residue from the writer-LLM and TTS stages that ran earlier in the same process) but it buys no headroom here. For the record, the 4060 is impossible rather than tight: the DiT weights alone are 9.8 GiB against an 8 GiB pool, and Gemma Q5 at 8.5 GiB exceeds it on its own. |
| D / D-BIS 2-5 / D-TER | static-audit findings awaiting a live observation |
| **PARTIAL-WIRING SWEEP (2026-08-19) -- 7 findings, all driver-verified** | **NEW. Run on gpt-5.6-sol the day the lane came back, hunting ONE named defect class: _partial wiring that tests green_.** The class had three confirmed instances in a week (PBUG-20260817-08 guarded 1 of 3 pools; the work-frame splice was dead code behind its own `except`; PBUG-20260815-06's consumer had no producer), so it was worth sweeping for deliberately rather than waiting for the fourth. Sweep shape as reported: 112 metadata keys, 1,388 exported symbols and 1,078 broad catches screened; 20 deep-traced, 13 rejected, 7 reported. **Every one was then re-verified by the driver against the real files -- 7/7 substantively survived**, which is a notably clean result for a panel lane and worth recording as such. **TWO are now PBUGs** (live-verified, so they clear the admission rule): `PBUG-20260819-01` the inert `normalize_dbfs` widget, `PBUG-20260819-02` `audio_revision` dead at both ends. **The other five are STATIC findings and are filed as such -- they await a live observation before they may become PBUGs.** They are listed in the row below. **NONE were fixed:** four of the five are the same question wearing different clothes -- *wire it or delete it* -- and per the `CLAUDE.md` ledger rule a dead pass may only be deleted once every field it owns has a new owner. That is a decision, not a grep |
| **OPERATOR VOTED 2026-08-19 -- 6 of 7 decided, 4 deletes SHIPPED** | **DONE for the deletes.** He ruled on the sweep from a listen-and-vote page: **audio_revision -> delete both ends**, **closing_audio -> remove the socket**, **post-freeze writeback auditor -> delete**, **announcer-only escape hatch -> delete**, **cloud media cache -> delete**. **Two went AGAINST the driver's recommendation** (the auditor and the cache were both recommended keep-or-park) and were taken as given. **Shipped: all four pure-code deletes.** `closing_audio` is deliberately NOT in that commit -- it touches `workflows/otr_canonical.json`, so it is a §0 change and gets its own chunk. **STILL UNVOTED: the stale-ledger voice guard** (`assert_registry_ledger_has_voice_ref_id`) -- left open rather than assumed. **TWO TRAPS THE DELETES HIT, both caught by measuring rather than assuming:** (1) `locked_against_audio_rev` sits in a model whose base is `extra="forbid"`, so deleting the field would break loading any ledger that carries it -- measured: **0 of 1,713 ledgers on disk carry it**, so the delete was safe after all, and the earlier "12/12 empty" reading had simply looked under `meta.video_plan` when `video` is a TOP-LEVEL ledger section. (2) Deleting the auditor orphaned two module constants; `_OPTIONAL_STRING_FIELDS` turned out to be **live** (imported and asserted by `test_voice_route_reference_contract.py`, cited by `_otr_ledger.py:793`) so it stayed, and `ALLOWED_MUSIC_RENDER_STATUS` was KEPT as written-down contract but its comments -- which claimed "walker enforces its own enum" -- were corrected, because a comment asserting an enforcement that no longer exists is the same class of lie the sweep was hunting |
| **the five STATIC findings from that sweep** | **OPEN, awaiting live observation. Do NOT promote these to PBUGs on the strength of the audit alone.** (1) **`cloud_media_cache` has zero production callers** -- `nodes/_otr_shared/cloud_media_cache.py` is exported and implemented, but production goes straight through reserve->submit->bill at `cloud_media_invoke.py:797-809`, so identical cloud requests are re-billed with no dedup. LOW priority: cloud is opt-in and the project is offline-first. (2) **`announcer_only_fallback` is read but never written** -- the empty-cast escape hatch at `_otr_ledger_freeze.py:438-445,528-531` has no producer, so the documented announcer-only case cannot pass freeze. (3) **`assert_registry_ledger_has_voice_ref_id` has zero production callers** (`_otr_audio_cache.py:216`) -- a stale cast-locked ledger missing `voice_ref_id` silently enters render-time reassignment instead of being rejected for recasting. (4) **`audit_post_freeze_writeback` has zero production callers** (`_otr_ledger_consumers.py:246`, exported at `:310`) -- its own docstring describes a "soft-rollout phase" where "consumers log violations to batch_log"; **the rollout reached no consumer**, so the §6.16 null/enum drift check never runs on a real episode. (5) ~~`closing_audio` unwired socket~~ **-- CLOSED 2026-08-19, the operator voted it out and the socket was REMOVED. Four findings remain, not five.** **A caution for whoever picks these up:** the reviewer named finding (4) in PROSE as "post_freeze_writeback_audit"; the real symbol is `audit_post_freeze_writeback`, and grepping the prose form returns ZERO and reads exactly like a refuted claim. Grep the symbol, not the description |
| scifi_news_pro news_read hallucinated character names | OPEN -- retry-ladder exhaustion (2 attempts) on a factual pass that invented "Dr. Sharon Hame, Laura Goodkind". A spawned task chip already scopes this fully; unrelated to the fixed CODA-leak defect |

**THREE FACTS THAT COST REAL TIME -- do not relearn them:**
* **The models root is `C:\ComfyUI-Models`, not `ComfyUI\models`.** A
  "162 of 206 voice refs are missing" claim was wrong-root error, retracted in
  `f2eeb6fd`. Every ref resolves and hash-matches.
* **A green suite can prove a fix fails SAFELY, not that it works.** A guarded
  fix raised `NameError` on every episode and all 10,905 tests still passed.
  Test the PRODUCT, not the absence of an exception.
* **ONE FIELD, TWO MEANINGS is the dominant defect shape here** -- four
  sightings in one day (`work_title`, the soak receipt's `title`,
  `title_source`, and `lines[].text` spoken-vs-displayed). Gate on the LANE,
  never on truthiness.

**A. ENGINE INPUT-CONVENTION CONFORMANCE AUDIT -- DONE 2026-08-17
(`PBUG-20260817-02`, Bible `12.109`). Receipts in `HANDOFF_LOG.md`. What still
binds:**
* **CHECK THE TOKENIZER, NOT THE NODE NAME.** A dedicated node that exists and
  is unused is only a defect where the family's TOKENIZER does not apply the
  convention itself. `lumina2.py` is a plain `SD1Tokenizer` with no template ->
  real defect; `qwen_image.py:32-36` shows `llama_template=None` means "use the
  built-in template" -> Z-Image is CONFORMANT. Name-matching would have shipped
  TWO FALSE POSITIVES (`z_image_turbo`, `flux_gen1`).
* **STRUCTURAL AUDITS CANNOT PROVE OUTPUT.** This class proves a node is
  missing, never that the output is worse. Every hit needs ONE A/B at a fixed
  seed. Permanent instrument: `scripts/_otr_lumina_image_smoke.py
  --no-system-prompt` is the pre-fix arm.
* **LUMINA IS NOT FLAG-GATED -- do not repeat the stale containment line.**
  `requires_flag = None`; the suite DELETES `OTR_ENABLE_LUMINA` and still
  expects the engine usable. The real gate is the weights file, all three are on
  disk, and it is wired into two soak profiles + `otr_sbcov_3`.
* The hygiene-floor gap the audit surfaced is queued as item **H**, not here.

**B. PROVE THE STYLE FIX ON ONE LIVE RENDER -- DONE 2026-08-17, six live stills,
both gates green on the shipped path.** Fable's acceptance test ran verbatim.
The instrument is permanent: `scripts/_otr_style_authority_smoke.py`, both arms
in one script behind `--pre-fix`, and it submits the ENGINE'S OWN graph
(`_zimage_params` + `_build_zimage_graph` are imported and called, never
re-typed) so "this is what a real mint does" cannot quietly stop being true.
No arc, per `7f6a6eca` -- an already-shipped fix measured against a
pre-specified acceptance test has no design fork in it. Sonnet 5 QA on the diff.

**What still binds** (receipts in `HANDOFF_LOG.md`):
* **GREEN GATES ARE NOT A WORKING FIX, PROVEN TWICE IN ONE DAY.** On the stills
  side a `cartoon` episode's PRE-FIX announcer minted as a literal PHOTOGRAPH
  while every prompt agreed. On the video side all 36 rewritten registers passed
  every text check and the pixels then moved **40% less**. The render is the
  only proof; budget one every time a prompt or a negative changes.
* **RUN THE CONTROL, NOT JUST THE ACCEPTANCE TEST.** My GATE 1 ("no
  illustration-family terms") was Fable's wording scoped to an ILLUSTRATED pack;
  generalized it fails the DEFAULT lane, where `clean digital` is legitimate.
  The correct universal form is the repo's own `_fights_in` -- does this
  negative contradict THIS pack's positive. The acceptance test alone would have
  shipped the over-broad gate; the default-lane control caught it.
* **THE DEFAULT LANE IS NOT BYTE-IDENTICAL AT A FIXED SEED, and that is
  correct.** "The photoreal packs carry the historical string VERBATIM" is true
  of the AUTHORED string and of the LOOK (the `sci_fi_radio` A/B is the same
  photoreal noir, no regression). It is NOT true of the conditioning:
  `effective_negative` drops `cartoon, illustration` from that pack too, because
  its own announcer surface asks for a "living cartoon appliance face". Self-veto
  resolution working, not drift -- a window expecting identical bytes misreads it.
* **THE GITIGNORE TRAP HAS A SECOND FACE: `kibitz-runs/` (found 2026-08-17,
  H-receipt).** `.gitignore:251` ignores the whole directory, yet **105 files under
  it are tracked** -- because git ignores nothing already in the index, the exact
  mechanism the bullet below describes for `scripts/_*.py`. So the CONVENTION is
  that panel artifacts are committed, and the rule silently blocks only NEW ones.
  H-receipt's scope receipt, driver anchor, judgment, final and both agy reviews
  were one command from being local-only while GO_FORWARD cited their path (this
  file cites `kibitz-runs/` paths eight times, and every one of them would dangle
  on a fresh clone). **`git check-ignore -v` any new artifact path, not just
  scripts** -- and note the trap is now proven in two different directories, so
  treat it as a class rather than a `scripts/` quirk.
* **A "PERMANENT" INSTRUMENT IS GITIGNORED BY DEFAULT -- `git add -f` IT.**
  `.gitignore:71` carries `scripts/_*.py`, so every A/B instrument this queue
  calls permanent lands UNTRACKED. `_otr_style_authority_smoke.py` was one
  command from local-only; item A's lumina smoke is tracked only because it
  predates the rule biting (git ignores nothing already in the index, which is
  why the trap stays invisible until a NEW instrument is written). Run
  `git check-ignore -v <path>` before believing a new script is committed.
* **`kill_otr_zombies.ps1` IS NOT THE SECTION 4 RESET** -- it deliberately
  PRESERVES the port-8000 owner, the opposite of what section 4 needs. Use
  `scripts/otr_reset_gpu.ps1` (selective kill by CommandLine with the Claude MCP
  pythons protected, then VERIFIES port free + VRAM at baseline).
* Instruments, both permanent: `scripts/_otr_style_authority_smoke.py --pre-fix`
  (stills, submits the ENGINE'S OWN graph rather than a re-typed copy) and
  `scripts/otr_ltx_motion_smoke.py` (video motion, now reads the pack instead of
  a hardcoded copy that had already gone stale).

**B-OPEN -- CLOSED 2026-08-17: the operator chose (B), and it is implemented and
re-measured.** The registers are the AUTHORED text with only the colliding words
changed; **25 of 36 are byte-identical to what the author wrote**, and the
camera moves, beat choreography and pack voices all survive.

| arm | mean frame delta | peak | drift |
|---|---|---|---|
| original (colliding) | 0.373 | 0.574 | 10.27 |
| the sweep (all camera stripped) | 0.220 | 0.355 | 3.50 |
| **(B) restored, colliding words fixed** | **0.349** | **0.757** | 6.01 |

**THE ELEVEN WORD-LEVEL FIXES, and why each -- do not "tidy" them back:**
`whip-pans` -> `races` / `spins wildly` (Wan ban + Seedance softener; a dial
cannot pan, it was borrowed camera jargon); `flickers` -> `wavers` (Wan bans
frame flicker; candlelight wavers); `white-hot` -> `fierce white` (Seedance
rewrote it to "bright warm glow", breaking the sentence); `vibrates
aggressively` -> `shudders with the music` (Seedance rewrote it to "subtly" --
energy inverted); `Slow handheld dolly forward` -> `Slow steady dolly forward`;
`Dynamic dolly push forward` -> `Steady dolly push forward`. Plus one CRAFT
restore that is not a collision: `Cel highlights alternate` -> `shiver`, because
the earlier `flicker`->`alternate` edit swapped an anime term of art for a
scheduling verb and charged the cost to local lanes that never had the conflict.
**`Slow dolly pull back` and `Slow orbit around the speaker` never collided and
are untouched.**

**THE RULE THIS LEAVES:** only SIX words in the whole tree ever collided with a
frozen provider list. A sweeping rewrite to fix them cost 40% of the motion and
had to be walked back. Fix the colliding token, never the surrounding writing.

**C. THE TEN PER-ENGINE PROMPT NOTES -- RESEARCH FIRST, THEN STORE.**

> **CORRECTED 2026-08-17.** This entry used to read *"Answers exist (three
> research rounds)"*. **That was false and it cost a window real time.** The
> phrase was borrowed from item D, which genuinely had three rounds (its
> conclusions cite measured receipts -- F2, P4, P8). C has none. Its own doc is
> titled *"the research prompts"*, its "Where the answers go" section is written
> in the future tense, and there is no kibitz-run or answer doc anywhere on
> disk. **Grep before trusting an "answers exist" claim -- including this one.**

**WHAT EXISTS** (`docs/2026-08-17-per-engine-prompt-style-guide-RESEARCH.md`):
the SCHEMA, decided -- `prompt_style_directive` (**240 chars hard**, the only
part that ever reaches an LLM) and `prompt_style_notes` (uncapped, humans only,
never injected); the five rules for directive text; a reusable research prompt;
and a per-engine config block for each of the ten engines, with the real shipped
sampler/cfg facts already filled in. All paste-ready.

**WHAT IS OWED:** (1) run the research, ten engines, one block each; (2) store
the two fields as constants beside the engine that owns them -- same shape as
`_HYGIENE_NEGATIVE` living in `z_image_turbo`. Storing is safe and free.

**THREE CONSTRAINTS THAT BIND THE STORE STEP:**
* **NOT WIRED, deliberately.** Storing is free; ACTING on a directive is a
  separate measured change gated on a before/after with
  `scripts/otr_talking_radio_probe_eval.py` at a fixed seed -- P4 measured
  articulation collapsing 4.15 -> 1.18 from a prompt-register change on this
  very lane. Store it, measure it, then enable it.
* **A directive may never override a visual style pack.** Style is the pack's
  job; the directive owns PHRASING only. Rule 3 of the schema, and it is the
  same authority boundary PBUG-20260817-01 was about.
* **Offline rule:** author the string ONCE and store it. Never fetch at runtime.
* **Strike z_image's negative-authoring clause before storing** -- see the trap
  in D. cfg 1.0 engines take no negative advice at all; several blocks already
  say so.

**ROUTING:** the STORE half is mechanical (schema decided, one verifiable
answer) -> Sonnet 5 QA on the diff, no arc. The RESEARCH half is a factual
lookup about how each model responds to phrasing, not a design fork -- web
search is ALLOWED (operator 2026-08-15, the RSS precedent), so it is $0 and does
not need a paid panel.

**STORE HALF SHIPPED 2026-08-17.** Ten `PROMPT_STYLE_DIRECTIVE` /
`PROMPT_STYLE_NOTES` pairs, 176-232 chars (max headroom 8, on `flux_gen1`), plus
`tests/test_prompt_style_directives.py` -- 64 tests, AST-read so no engine module
is imported. Suite 10755 -> **10819**, the delta exactly the new file.

**WHAT BINDS ANY FUTURE WORK HERE:**
* **THE OWNERSHIP ANSWER IS NOW STRUCTURAL, not a note.** No directive on ANY
  engine instructs authoring a negative, and a test fails the suite if one ever
  does. Stating a negative is inert or absent stays legal -- `minimax_h3` needs
  it. The first version of that guard was a nine-phrase blocklist and a Sonnet QA
  pass walked 14 of 16 plausible authoring instructions straight through it; the
  rule is now INVERTED (any "negative" mention needs an approved has-no-effect
  hedge) because ways to say "inert" are a closed set and ways to say "author
  one" are not.
* **`ltx_8gb` CARRIES A POINTER, NOT A PAIR.** The doc treats `ltx_video /
  ltx_8gb` as ONE block and the directive is genuinely shared -- but the stated
  reason was WRONG until QA caught it: the encoders differ (`ltx_video` runs
  GEMMA-3, `ltx_8gb` borrows shared T5-XXL). The directive survives because not
  one clause of it is encoder-specific. **That is also the split condition:** the
  first encoder-specific clause means the 8GB tier needs its own pair.
* **A `docs/...` PATH IN AN ENGINE MODULE BECOMES FRAME-CAP EVIDENCE.**
  `tools/engine_matrix.py` scrapes engine sources for `docs/[A-Za-z0-9._-]+` and
  publishes every hit in the ENGINE_MATRIX evidence column -- it exists because
  `eng_humo` once justified a 49-frame ceiling with a doc not in the tree. Citing
  the RESEARCH doc turned that column red on ten engines, and because the path
  was wrapped across two comment lines it captured a truncated path that does not
  exist. Regenerating the doc would have been the WRONG fix: `wan_ti2v` would
  then cite a phrasing doc as its frame receipt. Name the doc WITHOUT the prefix.
* **CITE SYMBOLS, NEVER LINE NUMBERS.** A citation-verification pass found **8 of
  21 wrong**, and five were pure line-shift caused by this very diff inserting
  lines above them -- the drift equalled the insertion count exactly (+53
  `eng_ltx_av`, +56 `eng_humo`, +46 `z_image_turbo`). This diff also broke a
  PRE-EXISTING comment in `lumina_image` that cited `z_image_turbo.py:117`. All
  notes now cite constants and quotes.
* **THIS FILE'S OWN CITATIONS ARE STALE, verified at the files 2026-08-17:**
  `_LTX_MOTION_PROMPT_MAX` is at `render_driver.py:1345`, NOT `:1327` (`:1327` is
  `_LTX_MOTION_PROMPT_BY_ROLE`); the two surviving "subtle" strings are
  `_IA2V_TALKING_CLAUSE_CHARACTER` and `_CHAR_FACE_FALLBACK_PROMPT`, NOT `:1354`
  and `:1483`; the i2v doctrine comment sits ~8 lines below the cited
  `:2877-2884`; and "the 188 appears at exactly ONE call site" is true only of the
  `max_chars=188` argument -- the branch carries 188 in THREE places, so "raise
  the one" leaves two behind.
* **ONE SUBSTANTIVE CLAIM WAS FALSE AND IS NOW CORRECTED:** `fastwan_8gb`'s note
  asserted no env knob or prequalification override could re-enable its negative.
  It can -- `prequalification_active` plus the `cfg` key in
  `_FASTWAN_RECIPE_ENV_KEYS` lets the inherited
  `WanTi2vEngine._resolve_render_config` move cfg off 1.0, and the unconditional
  branch returns with it. What genuinely does not exist is a negative-TEXT
  channel. Absolutes in a note are where false authority gets in.
* **THE RESEARCH DOC ITSELF HAS AN ERROR:** its `ltx_video` block says frames go
  "up to 193". The engine ships `_LTX_MAX_FRAMES_DEFAULT = 169`, a MEASURED decode
  constraint, and 193 matches no constant in the file. Trust the engine.
* **THE THREE EXTRA ENGINES ARE NOW IN -- SCOPE CALL CLOSED BY THE OPERATOR
  2026-08-17. The map is THIRTEEN, not ten.** `flux2_klein` (`requires_flag = None`,
  so NOT gated -- live in the menu), `hidream_i1` and `sd35_large` (default-OFF).
  **He supplied all three directives himself**, drafted from public docs and then
  validated in a v2 pass; they are stored VERBATIM, not rewritten. Counts as
  measured: 235 / 228 / 217 chars.
  * **HIS STATED COUNTS RUN ~5 CHARS LOW, CONSISTENTLY.** He labelled them
    230 / 223 / 213; the test measures 235 / 228 / 217 -- exactly +5 on all three,
    so it is a systematic difference in method, not a typo. Hand-counted at 235,
    the flux2 string confirms the test. Trust `_HARD_CAP` and the test, and leave
    margin when drafting near the ceiling: a draft he counts at 238 measures 243
    and fails.
  * **ONE FACTUAL CORRECTION HIS v2 PASS CAUGHT IN MY OWN STORED NOTE:** I wrote
    that HiDream "SUPPORTS a negative, unlike FLUX.2". Wrong -- support is
    **VARIANT-dependent**. Full runs cfg 5.0 with the negative LIVE; **dev and fast
    run cfg 1.0 and have none**. So the registry must record VARIANT + cfg beside
    that engine or the negative field silently lies -- the same defect class item
    H-RECEIPT just closed on the dispatcher, arriving from another direction.
    Whatever quant lands on disk decides the negative reality.
  * **SD3.5's CAP IS TWO-TIER and it is the sharpest fact in the set.** Tokens
    1-77 are seen by ALL THREE encoders; past 77 is T5-ONLY and invisible to both
    CLIPs, INCLUDING their pooled global-STYLE vectors. Nominal T5 ceiling 256 with
    edge artifacts; effective length reported at 154. Working rule: full-coverage
    budget is **77 tokens TOTAL, style-pack included**; 154 is the conservative
    hard ceiling. This makes style-token POSITION second-order on that engine and
    TOTAL TOKENS the real control -- so an A/B there must log the token count, or
    an over-77 run gets misread as a position result.
  * **FLUX.2's cap is OURS, not the model's:** diffusers allows 512 tokens and
    BFL calls 30-80 words ideal, while a 188-char beat is ~27-30 words -- the floor
    of the ideal band. Its append ordering is now evidence-backed (BFL documents
    front-to-back weighting, subject > action > style > context), which is the SAME
    mechanism as Wan's subject-first training and the reason LTX's camera-first
    guidance was rejected. Three engines, one mechanism.
* **AN INSTALL-DAY VERIFY QUEUE IS OWED for the three, and it is cheap:** klein-4B's
  embedder identity (read the checkpoint's `text_encoder` config -- Qwen3-8B is
  confirmed only on the 9B); klein's local negative surface (expect none); **which
  HiDream variant is actually on disk**, since that one fact decides its negative,
  shift and step count; and SD3.5's practical T5 ceiling on our graph (154 vs 256).
  Also recorded: klein 9B is out on TWO counts, non-commercial licence and ~29GB.
* **THE PROPOSED SELECTION GATE IS ENDORSED AND DELIBERATELY NOT BUILT:** "engine
  selectable AND directive present, else hard refuse at selection -- no bare-writer
  run, no borrowed directive." Right in spirit, premature in fact: NOTHING reads
  these constants, so the refusal would gate on a value no code consults. Build it
  in the same change that wires the overlay.

**WHAT THE RESEARCH FOUND -- FOUR TRAPS AND TWO FACTS. The full write-ups live in
each engine's `PROMPT_STYLE_NOTES`; this is the index.**
* **TRAP -- public `ltx_video` guidance CONTRADICTS an operator directive.** Guides
  say camera-move-first; he directed subject-first, and rewriting the registers the
  wrong way cost 40% of the motion at a fixed seed. **Do not adopt it.**
* **TRAP -- and the sibling lane proves no global rule can work.** Wan 2.2 was
  TRAINED on subject-first captions and weights early tokens hardest; FLUX.2's docs
  say the same (subject > action > style). So Wan and FLUX.2 want subject-first for
  documented reasons while LTX guidance wants camera-first. **This is item D's
  per-lane rule table arriving as evidence.**
* **TRAP -- most public Z-Image advice assumes a cfg we do not run.** Upstream
  defaults to guidance 0.0 (negative ignored); we run cfg 2.0 deliberately to keep
  it live. "Negatives do nothing here" is true upstream, FALSE for us. Reconciling
  the engine to a guide would delete a deliberate departure.
* **TRAP -- variant/host is not model.** Most findable "Lumina prompting" is about
  **Neta Lumina, an anime FINETUNE** whose tag tolerance was trained in; we load
  base. MiniMax guides describe a **hosted UI** that takes a negative; our adapter
  has no negative field at all. HiDream's negative is **variant**-dependent (Full
  only). In all three, the doc describes a different surface than the one we run.
* **FACT -- two clauses confirmed from BOTH directions**, the strongest evidence in
  the overlay: `humo`'s brevity (upstream says concise, P4 measured the 4.15 -> 1.18
  collapse) and `flux_gen1`'s whole directive.
* **FACT -- the best A/B on the overlay is `ltx_av` LENGTH.** Upstream expects 4-8
  sentences; our budget is 240 chars, and the budget is OURS. External advice and
  our own P4 measurement point OPPOSITE ways; the measurement wins until a render
  says otherwise. **Highest-value render on the list.**
* **Also: Z-Image skews Asian/Chinese unless ethnicity is stated explicitly.** Not a
  phrasing rule, but it lands on the still-open correctness class -- a character's
  appearance contradicting the source.

**D. THE PROMPT-STEERING QUESTION -- BLOCKED, do not build blind.**
Fully designed across three research rounds + a Fable pass; artifacts in
`kibitz-runs/`. Settled conclusions:
* **NEVER build a post-hoc LLM rewriter.** Four measured receipts: F2 swung
  3/6 then 1/6 on identical fixtures (`JUDGE_ATTRIBUTION=False`); P4 collapsed
  articulation 4.15 -> 1.18 on a register change; **P8 is decisive -- a
  PARAPHRASE scored half the canonical's articulation, 1.72 vs 3.32
  (`render_driver.py:1345-1348`)**, and a paraphrase is exactly what a
  rewriter emits. A content-preservation gate does NOT catch this: a faithful
  paraphrase passes every entity check and still halves the score.
* **The sanctioned pattern is generation-time steering** -- directive in the
  WRITER's prompt for text that does not exist yet, validator, deterministic
  fallback, env flag default OFF (`OTR_LTX_MOTION_CLAUSE` is the template, and
  its trap was shipping with the flag set nowhere).
* **THE BLOCKER:** the still-prompt writer does not know its target engine --
  binding happens at dispatch and roles drift under `OTR_FORCE_ENGINE_MAP`. A
  per-engine directive at generation time targets an engine it cannot see.
  **Settle this (Codex was the intended lane) before any stills work.** Video
  may be buildable first.
* **Already in the tree, so this is cheaper than it looks:**
  `_DIRECTIVE_KEYS = ("expression","motion","camera")` is the beat schema;
  `_deterministic_template` is the fallback floor (BUG-046);
  `_subject_anchor` already enforces subject-first.
* **Conflict to resolve:** the panel scaffold says "never open with a camera or
  framing word", but `_subject_anchor` deliberately opens talking-head prompts
  with "face visible, speaking to camera" (round-5 F3, engines weigh leading
  tokens hardest). Per-lane rule table, not a global one.
* **Cheapest real win here:** the beat's empty-string directive keys
  (`{k: "" for k in _DIRECTIVE_KEYS}`) become an explicit `NONE`. Blank is a
  question the model answers; NONE is an instruction to omit.
* **A/B traps:** stills seeds derive from `prompt_hash`, so changing prompt
  text changes the seed -- **stills A/B must use `mode=fixed`**. Video is
  clean (request hash is brief/cast/beat/char). **`otr_story_score.py` is NOT
  a judge** -- it reads ledger structure, never a prompt or a pixel.

**D-BIS. THE NEGATIVE-CENSUS RESIDUE (2026-08-17) -- five findings with no
other home. STATIC-AUDIT findings, NOT PBUGs:** the admission rule reserves
`PROD_BUG_LOG.md` for defects verified by a live artifact, and these came from a
full-repo census plus a Fable design pass. Each needs one live observation
before it may be promoted.
1. **A cross-family negative conflict the style traceroute structurally could
   not see -- NOW MEASURED AND REPORTED (2026-08-17, item B window). The entry
   as first written was stale in BOTH directions.**
   * **"flicker" is ONE pack, not four.** Only
     `shakespeare_stage_realism:27` ("Candlelight flickers across polished
     wood") still asks for it; the `anime` rewording to "alternate" already
     fixed the rest. The original count described the state before that edit.
   * **The real finding is "whip pans", and it was unrecorded.** The cloud Wan
     negative bans it while the `music_open` register of **four** packs asks
     for a dial that "whip-pans" -- `anime`, `cartoon`, `paper_origami` and
     **`sci_fi_radio`, the DEFAULT pack**. That is what makes this the
     widest-reach item rather than an exotic-pack curiosity. It stayed invisible
     partly because the engine writes "whip pans" and every pack writes
     "whip-pans", so a strict literal test finds none of the four.
   * **Every LOCAL video engine is CLEAN** (`ltx_av`, `humo`, `ltx_8gb`,
     `ltx_video`) -- their negatives are pure quality/artifact terms. The
     exposure really is confined to the opt-in cloud lane.
   **What shipped:** `scripts/otr_style_traceroute.py` now measures and REPORTS
   the video side (`find_video_fights` + an AST reader that pulls each engine's
   negative WITHOUT importing it, so the tool keeps its "loads no model, spends
   no GPU" promise), with 22 tests. `VIDEO_DICT_SURFACES` had existed there from
   the start but was only ever displayed; `_fights_in` never read it.
   **`--strict` IS DELIBERATELY NOT EXTENDED TO IT** -- the video negatives are
   FROZEN RECIPE, so gating CI on a string this repo may not edit would be a
   build-breaker by construction.
   **DECIDED 2026-08-17 by a four-way panel (Fable + Sonnet + Antigravity +
   the driver anchor), UNANIMOUS: do NOT build a compose-time negative guard.**
   Codex was quota-held and excluded. Two reasons neither the anchor nor D-BIS
   had, both verified in code:
   * **The ban is ALSO in the POSITIVE prompt.** `_WAN_SMOOTH_MOTION_CLAUSE`
     (`eng_cloud_video.py:212-219`) reads "No whip pans, handheld shake, sudden
     reframing, jump cuts, rapid zooms..." and is appended to every Wan positive
     at line 266. Vidu has the same shape at 231-237. So a guard that strips the
     NEGATIVE resolves one of three channels and stamps a receipt claiming the
     fight is resolved while the ban still ships in the prompt body. **Every
     negative-only audit in this repo, including the traceroute, is blind to a
     prohibition written as positive prose.**
   * **There is no video choke point.** Stills resolve the negative in ONE
     function called from ONE place; video resolves it at SEVEN call sites
     across FIVE files, and there is no per-shot negative ledger field at all
     (`render_driver._stamp_prompt_meta` records positive-prompt fields only).
     A guard would be building the first video negative receipt from scratch.
   * **Mostly a HOMOGRAPH, not a conflict.** Every register opens "Continuous
     shot, same console throughout" and the thing that whip-pans is the DIAL;
     the negative's "whip pans" sits among jump cuts / rapid zooms / handheld
     shake, i.e. CAMERA pathologies. The ban was PROTECTING the registers' own
     first sentence. Unlike PBUG-20260817-01, where both sides meant the same
     referent.
   **WHAT WAS ACTUALLY BROKEN WAS SOMETHING ELSE, AND IT WAS SHIPPING.**
   `_SEEDANCE_PROMPT_SOFTENERS` (`eng_cloud_video.py:176-198`) already rewrites
   register text at compose time on the Seedance lane -- i.e. the "option (b)"
   being debated was ALREADY BUILT on a sibling lane -- and as a blind regex
   pass it produced `cartoon.music_open` -> **"Dial slowly sweeps wildly"** (a
   contradiction) and, on the DEFAULT pack's most energetic beat, four
   substitutions including "vibrates aggressively" -> **"vibrates subtly"**
   (energy inverted) and "cold to white-hot" -> "cold to bright warm glow"
   (broken phrase). Six registers across four packs. Fixed by the rewrite below;
   the softener table itself is frozen recipe and was NOT touched.
   **THE TRACEROUTE'S OWN BLIND SPOT, found by the panel:**
   `VIDEO_NEGATIVE_SOURCES` was hand-curated and missed `_RAZZLE_NEG_DEFAULT`.
   It now carries `discover_video_negative_constants()`, a coverage guard that
   reports any negative the report does not audit -- which immediately found a
   SEVENTH nobody had enumerated, `wan_shared._WAN_DEFAULT_NEGATIVE` (shared by
   both local Wan engines). A hand-maintained source list reads exactly like
   "no conflicts" when it is really "did not look".
   **STILL A STATIC-AUDIT FINDING, NOT A PBUG:** opt-in engine, credentials
   required, provider-side liveness unverifiable from this repo -- and the panel
   sharpened this: there is no PATH to a live observation, because the lane
   needs paid credentials the scope rules default against. A shipped video guard
   would be permanently stuck at the evidentiary tier the stills fix explicitly
   calls insufficient.

**D-TER. THE B6 ENV GATE EXISTS IN ONLY ONE OF FIVE ENGINE FILES (2026-08-17,
found by the panel -- this WIDENS D-BIS finding 3).** D-BIS recorded three
ungated env negatives. The real count is worse: `eng_ltx_8gb` gates its override
behind `_prequalification_active()` so production reads ONLY the frozen recipe,
and **no other engine does**. `eng_humo:748`, `eng_ltx_av:766,909`,
`eng_ltx_video:1169,1332` and `eng_cloud_video:815-817,1010-1012` all read their
env override unconditionally, on every production render. That is the exact
shape of the bug B6 fixed, still live in four of five files. Static-audit
finding; recipe-adjacent, so it needs the operator, not a driver decision.
2. **Two duplicate-drift negative variants.** The same 7-term boilerplate exists
   in four copies; `eng_ltx_av.py:107-109` and `eng_humo.py:112-114` have
   diverged with extra terms and no recorded reason. HuMo's divergence looks
   deliberate (hand/face artifacts are its real failure mode); LTX-AV's looks
   accidental. Consolidation needs an operator recipe ruling either way -- flag
   it, never "clean it up".
3. **Three env negatives are NOT consent-gated.** `OTR_LTX_NEGATIVE`,
   `OTR_LTX_AV_NEGATIVE` and `OTR_HUMO_NEGATIVE` are read with a plain
   `os.environ.get` on EVERY render, while `eng_ltx_8gb.py:818-829` deliberately
   demoted its equivalent because a boot-time env channel "made two boxes render
   visibly different clips from the same episode while both stamped the same
   recipe receipt". Same B6 reproducibility exposure, still open on three lanes.
   Recipe-adjacent: needs the operator, not a driver decision.
4. **The visual ledger cannot yet prove which negative conditioned which pixel.**
   Two gaps: the negative text lives in `visual.prompts[]` while the
   pixel-producing row lives in `images[]`, so it takes a join on `object_id`;
   and **the cfg is recorded nowhere**, so a logged negative can be accurate and
   still misleading -- at cfg 1.0 it conditioned nothing. Adding the resolved
   cfg (or a `negative_live` bool) to the per-row record closes it.
5. **Zero tests cover the visual-ledger RECORDING.** The negative-resolution
   logic is well covered (48 tests), but nothing asserts the shape of
   `ledger["visual"]`, `negative_source`, `self_veto_resolved`, `_style_spread`
   or `_laplacian_variance`. The operator's "lock them in the ledger" ask is the
   untested half.

**E. Upscalers + the 4060 full-stack gate** (item 1.3). **STEP (a) IS DONE
2026-08-17 -- two candidates on disk, identity-proven, wired to nothing.**
`RealESRGAN_x4plus` (ESRGAN, scale 4, tags `['64nf','23nb']`, 67,040,989 bytes)
and `RealESRGAN_x4plus_anime_6B` (ESRGAN, scale 4, tags `['64nf','6nb']`,
17,938,799 bytes), both BSD-3-Clause, both pinned by SHA in
`scripts/ensure_upscale_models.py`, which now reports all three OK and exits 0.
* **THE SHA WAS NOT TAKEN ON FAITH.** Each file was loaded through spandrel on
  CPU and its architecture, scale, channel count and block tags read back and
  recorded beside the pin -- the rigour x2plus got. Pinning the hash of whatever
  arrived would certify the download, not the weights.
* **THE `realesr-*` v3 CHECKPOINTS WERE DELIBERATELY LEFT OUT** (`x4v3`,
  `animevideov3`; both exist, 4.7 MB and 2.4 MB). They are **SRVGGNetCompact,
  a different architecture** from the RRDBNet/ESRGAN family the engine loads.
  Adopting one is a design decision, not a download.
* **STEPS (b) AND (c) REMAIN, AND (c) IS NOT MECHANICAL.** `SpandrelEsrgan` is
  hard-pinned to ONE checkpoint -- `_model_filename`, `_model_sha256`,
  `intrinsic_scale = 2` and a per-engine `commercial_clean` are all class
  constants -- so "rotate the candidates through the existing stage" means
  either an engine class per checkpoint or a parameterized engine, and the
  cache fingerprint folds `declared_sha256`. That is a design fork with more
  than one defensible answer, so it takes an arc; the downloads did not.

**F. THE WRONG-PLAY FRAME IS CODED AND PUSHED (`87dee50d`, 2026-08-17).**
r1-r3 ran with both operator-named agy lanes, a Fable narrative gate and a
Sonnet 5 QA pass on the finished diff; **r4 convergence and ONE LIVE LEG are what
remain**, and Codex is quota-held to 2026-08-19, so this is **not yet a full
four-round arc**.

**WHAT SHIPPED:** `work_title` threaded to BOTH announcer producers and to
`OutlineRequest`, rendered as `WORK: a scene from <title>` by a pure `_work_line`
helper, sourced from `_otr_source_identity.identity_from_meta`, with both
adaptation pack seams reworded. 18 tests, suite 10842 -> **10860**.

**THE BUILD-BREAKER THE PANEL CAUGHT, and it is the lesson worth keeping.** The
plan (mine) asserted every non-adaptation lane yields an empty `work_title`.
**FALSE, and measured:** `identity_from_meta` maps `media_archive`'s
`source_label` onto the SAME field, and **56 of 98 live media_archive ledgers
carry one** -- the first example being `"Now See Hear!"`. Ungated, every one
would have announced *"a scene from Now See Hear!"*, inventing a play on 57% of a
live lane -- a worse fidelity defect than the one being fixed. **Sonnet CLEARED
that lane** (it read the producer module and found nothing populating the field);
**Pro 3.1 CAUGHT it** (it read the consumer's own mapping); **the corpus
decided.** Now gated on `_otr_source_identity.ADAPTATION_SOURCE_KINDS` with a
positive control so the gate cannot be satisfied by a predicate false for
everything.
> **THE SHAPE, third sighting in one day:** one field carrying TWO MEANINGS --
> the work PERFORMED vs the publication a post came from. Same as the
> `_neg_source` lie (H-receipt) and PBUG-20260817-03's two naming authorities.
> **A consumer meaning "the work this episode performs" gates on the LANE, never
> on truthiness.**

**TWO INFRASTRUCTURE FACTS FOUND HERE, both cost a retry:**
* **`kibitz.py --topic` is the lane-isolation lever.** The run folder is
  `<date>-<--topic>` and `--topic` DEFAULTS to `kibitz`, so two lanes in one
  round silently overwrite each other -- both return rc=0 and both print
  "Reviews collected: antigravity: OK". One topic per lane, always.
* **Pro 3.1 needs `KIBITZ_AGY_PRINT_TIMEOUT` on a long doc.** `kibitz.py
  --timeout` does NOT reach agy; the CLI flag is built from `AGY_PRINT_TIMEOUT`,
  default `5m`. Two attempts died with `Error: timeout waiting for response`;
  `15m` landed first try. **That failure is NOT a quota block** -- `agy models`
  returned rc=0 throughout.

**AND THE UNTRACKED-ARTIFACT TRAP BIT A THIRD TIME, in a third directory.**
`tests/test_cross_play_frame_leak.py` -- the only test that drives the real
composer against the real manifest and pins the literal shipped-defect string --
was `??` in `git status` and would have been left local-only. Not gitignored,
just never added. **`scripts/_*.py`, `kibitz-runs/`, and now a plain new test:
check `git status` before the commit; `git add <dir>` is not enough.**

**PROVEN ON A LIVE LEG 2026-08-17 -- receipt:
`kibitz-runs/2026-08-17-item-F-wrong-play-frame/live_leg_receipt.md`.**
`otr_writer_bank_gate.py --banks shakespeare --acts 1` against a fresh headless
server loading the canonical JSON: **PASS, exit 0, 10.6 min.** The episode drew
`The Tempest` (Act 1 Sc 2) and the announcer said:
> *"Good evening, listeners. Tonight, we bring you Shakespeare's 'The Tempest',
> where Prospero and Miranda brace against a gathering storm."*
Right play, only the locked cast named, and **no other manifest play's title
appears anywhere in the episode** (checked over the full spoken text against all
14 rows). **`meta.announcer_intro_rewrite` = `announcer_intro_rewritten`, so the
SECOND producer fired and the fix still held** -- the single most likely way for
this to be silently undone, and the reason r1 insisted both producers change
together. A leg where the rewrite did not fire would not have tested it.

**THE SECOND LEG FAILED, AND IT SPLITS THE VERDICT.** `public_domain`, run on
`b45c5577`: the announcer opened with *"we gather for 'The Adventure of the
Purloined Paper'"* -- a work that **does not exist**. The source was `Nonsense
Novels` by Stephen Leacock; the episode title was `The Blackwood Enigma`; the
announcer named a THIRD string. The coda was correct.
**THE FIX WORKED AND THE MODEL IGNORED IT** -- replaying the shipped ledger
through the real code shows `work_title == "Nonsense Novels"`, the lane gate
passing, and `WORK: a scene from Nonsense Novels` rendered into the prompt, with
the rewrite producer active. **Logged as `PBUG-20260817-04`, unfixed on purpose:
the seam has said "invent none" the whole time, so guessing at wording is the
mechanism this item already proved unreliable.**

**SO ITEM F IS SPLIT: shakespeare PASSES live, public_domain FAILS live, same
commit.** The threading defect is fixed and proven; the *invented-title* class is
open and is a DIFFERENT root -- supplying the fact is necessary and not
sufficient. `tests/test_cross_play_frame_leak.py` cannot catch it by
construction: it detects other manifest rows' names, and an invented title
belongs to none. That is the residue the receipt predicted one leg earlier.

**STILL OWED: r4 with the CODEX lane.** r4 converged on both agy lanes (zero
must-fix, one `__all__` should-fix taken), but Codex was quota-held to
2026-08-19 and never participated in ANY round. **If a full arc requires the
Codex lane, this is a four-round campaign one reviewer short and must be
described that way.**

**The r1 detail below is kept for its rulings.** r1 DISPROVED SECTION D'S OWN
DIAGNOSIS. Artifacts: `kibitz-runs/2026-08-17-item-F-wrong-play-frame/`
(`driver_anchor.md`, `r1_antigravity_flash37.md`, `r1_judgment.md`,
`r1_final.md`) + `kibitz-runs/2026-08-17-item-F-pro31/r1/`. **r1 ONLY -- Codex is
quota-held to 2026-08-19 20:31, so this is NOT a full arc and must not be
reported as one.** `r1_final.md` is the input to r2 and supersedes the anchor
where they disagree.

* **SECTION D'S WORDING IS WRONG AND IS CORRECTED BELOW -- do not build from it.**
  Nothing "samples" a play or a setting: the only draws choose the SCENE
  (`select_shakespeare_scene_ref`, correctly) and a style slug (`select_style`,
  deterministic sha256). **No constant on this path contains "Verona."** The
  wrong place is a free-text LLM field, `_otr_outline._MacroShape.setting`.
  Pro 3.1 raised this unprompted as a MUST-FIX, independently of the driver's
  trace. A window reading "sampled" as a pool draw will hunt a pool that does
  not exist.
* **AND IT IS NOT "GENERATED FROM A DIFFERENT RECORD" -- the record is NEVER
  HANDED TO IT.** `source_meta_from_scene` builds a complete record and the
  writer stamps it into `meta`; it then dies three times -- the interpreter call
  passes `source_meta` only inside its `except` branch, `OutlineRequest` has no
  play field, and `SafeOpenBrief` has exactly five. So this is a THREADING fix,
  not a consumption fix.
* **THERE ARE TWO FRAME PRODUCERS AND A THIRD UPSTREAM SOURCE.** The I.4.9
  rewrite (`announcer_intro_rewrite`) overwrites the first frame, and the macro
  LLM authors `setting` from the brief alone -- so a fix landing only on the
  announcer names the right play while describing the wrong place. All three
  need the same change.
* **THE ONE SYMBOL THAT SOLVES THE FAMILY:**
  `_otr_source_identity.identity_from_meta(meta).work_title` already normalizes
  `play_title` (shakespeare) and `title` (public_domain), never raises, and
  returns `""` when degraded. No per-lane branch needed.
* **STILL OPEN, and r2 owes it:** `SafeOpenBrief`'s docstring says the locked cast
  is *"the only proper names the announcer may use"* -- a HALLUCINATION FENCE,
  not a spoiler rule. Both lanes answered only the spoiler half. Does naming
  "Twelfth Night" trade a wrong-PLAY frame for a wrong-SCENE one? Fable was
  asked this on the operator's 2026-08-17 call.
* **DRIVER ERROR, recorded:** the anchor tabled "wire
  `_otr_passage_selector.select_passage`" because that module's docstring
  contains the word *"Verona"*. It is a verbatim dialogue-window slicer,
  built-and-parked for the passage lane (three healthy commits, its own QA doc,
  called "already implemented" in the public-domain plays plan) -- unrelated, not
  abandoned. **A docstring naming your symptom is not evidence the module solves
  your defect.** Same family as item A's name-matching ruling.
* **THE KIBITZ LANE-COLLISION TRAP (found here, applies to every future run):**
  `kibitz.py`'s run folder is `<date>-<--topic>` and `--topic` DEFAULTS to
  `kibitz`, so two lanes in one round silently overwrite each other -- both
  returned rc=0 and both printed "Reviews collected: antigravity: OK". The first
  Pro 3.1 review was lost this way. **`--topic` is the isolation lever: one topic
  per lane, always.**

**I. HISTORICAL WRONG-PERSON RECEIPT -- FIXED 2026-08-20; `media_archive`
EXTENSION LIVE-CLOSED 2026-08-21 (PBUG-20260817-03, Bible `11.61`).** The
present-tense diagnosis below records the pre-fix tree and is not an open queue
item. This is the "different bug hiding in those rows" item G called out, and it
was bigger than item G was.

**MEASURED, read-only over all 1,710 published ledgers:** cast rows carry a
`character_description` about a DIFFERENT person, and the contaminated string is
copied verbatim into `meta.visual_plan.characters[NAME].portrait_prompt`, so the
portrait was painted of that other person too. The reported episode has **2 of 3
rows** wrong. Most recent hit
`signal_lost_lemmy_provisional_tier_kokoro_acceptance_20260816_210751`
(2026-08-16), so unlike item G **this is live at HEAD, not a retired regime.**
It also survives a FREEZE -- `baked_ledger.json` carries a contaminated row in
fourteen copies.

**THE CENSUS IS NOT DONE AND DO NOT QUOTE ONE NUMBER AS IF IT WERE.** Two
independent detectors were built and neither is complete:
* **pitch-cast scoped** -- flag a description containing a name from
  `selected_concept.cast` that no roster row owns: **28 rows / 20 ledgers**, but
  only 124 of 1,710 ledgers record that field, so it is blind to the rest.
* **name-shape scoped** -- flag a person-name-shaped phrase in the identity head
  that the roster does not own: **18 rows / 14 ledgers**, and it catches two the
  first one misses (`the_wax_cylinders_whisper` OYA SATO <- *"30s, Henry 'Hank'
  Griswold."*, `nightshift_erasure` RYAN KAPOOR <- *"60s, EDWARD 'ED'
  GRISWOLD."*) while missing LUCILLE PENNY, which the first one catches.
**The operator's second reported instance is therefore CONFIRMED**, and the real
total is the union, uncomputed. Computing it properly is part of this item.

**ROOT CAUSE, proven at the files -- TWO NAMING AUTHORITIES arbitrated inside a
prompt.** The pitch names the characters (`selected_concept.cast[].name`) and the
brief restates those names; the cast pool then assigns different ones;
`_otr_casting.build_description_prompt` hands the model BOTH -- the brief on the
`Story:` line, the assigned name on the `Name:` line -- and states no precedence.
Its own CHARACTER VISUAL CONTRACT format reserves a free-text slot immediately
after the age band (`"<age decade>, <story-linked role>. Face: ..."`) and the
model fills it with the brief's name. **The prior-cast theory in the original log
entry is DISPROVEN:** LUCILLE PENNY is not a cast row in that episode or anywhere
in the corpus, so `_format_prior_entry` cannot be the path.

**THE DETECTION SCOPE IS THE HARD-WON PART, and the obvious check is wrong in
BOTH directions.** "No description may name another CAST ROW" returns 47 hits of
which ~45 are legitimate relational prose (*"foil to the Time Traveler"*,
*"Rosalind's loyal best friend"*) and it does NOT flag the reported episode at
all. The check that works is **ensemble-foreign**: a proper name that no cast row
owns, sourced from the pitch cast.

**ROUTING: FULL FOUR-ROUND ARC BEFORE CODE.** There is a real design fork with
more than one defensible answer -- reconcile the brief's names before it enters
the prompt, state precedence in the prompt, gate ensemble-foreign names after
generation, or some combination -- and it touches a live generation path plus a
derived image surface. Codex is back 2026-08-19 20:31. **Do NOT reach for fuzzy
name repair**, the reflex fix: it renames the intruder to the row's own name and
leaves that other person's face, bearing and delivery prose in place, which turns
a visible defect invisible. Bible `11.61` says exactly this.

**AND IT CHANGES WHAT ITEM G'S "34/35 PORTRAIT CONFLICTS" MEANT.** Several hits
are gender-crossed in both directions (RICK STEINER male <- LUCILLE PENNY; WENDY
PALMER female <- SIR REGINALD PENNYWORTH), so a share of that count was never a
gender defect at all. Do not re-derive the portrait number without subtracting
these rows first.

**G. PBUG-20260815-11 -- MEASURED 2026-08-17 AND THE RE-ASK SHOULD NOT BE BUILT.
The number this item is named for is one the audit itself disowns.**

Panel: Fable (the detection heuristic) + Sonnet (the mechanism) + the driver anchor;
**both kibitz lanes were quota-held** (Codex to 08-19, Antigravity on confirmed
`RESOURCE_EXHAUSTED`). Artifacts in `kibitz-runs/2026-08-17-item-G-gender-reask/`.
r1 ONLY -- not an arc, and not a kibitz fan-out.

**THE DECIDING MEASUREMENT, run live and read-only over the real corpus:**
* `scripts/audit_voice_gender_consistency.py` reports **`VIOLATIONS: 0`** across all
  1,710 ledgers. **The voice field agrees with assigned gender EVERYWHERE.** That is
  the criterion the operator named on 2026-08-17: *"as long as it matches the gender
  of the voice."* By his own test the defect measures ZERO.
* The "34" (now **35**, the corpus grew) is the PORTRAIT-PROSE count, and the audit
  prints beside it: *"a DISGUISE plot is a legitimate hit here -- ROSALIND-as-Ganymede
  and VIOLA-as-Cesario keep female voices by operator ruling, so read this list, do
  not total it."* **The item was named after a total the tool refuses to total.**
* **THE CORPUS IS STALE, and the operator said so first.** The unisex bucket was
  retired in `b8206412` (2026-08-15 17:55), and he notes the gender ASSIGNMENT was
  reworked in a sprint about a month ago -- so the ledgers are stratified by at least
  two regime changes. Split at the retirement: **pre = 4,994 binary rows / 26
  unanimous conflicts (0.52%); post = 78 rows / ZERO.** The single post-retirement
  flag is **VIOLA**, `opposing=['man']`, `matching=['her']` -- the disguise plot.
  Stated honestly: 78 rows at 0.52% predicts ~0.4 conflicts, so zero is EXPECTED
  either way. It does not prove the bug is gone; it proves **every piece of evidence
  for building came from a regime that no longer exists.**

**AND BUILDING IT NAIVELY WOULD KILL RENDERS.** A `post_validator` rejection raises
`PostValidationError`, which SKIPS both the structural-retry and repair-syntax rungs,
so a content chain exhausts at `attempts_run == 2` while `lock_cast` promotes only on
`== max_attempts` (3). The equality fails, the bare `raise` fires, the render dies --
the one outcome the ruling forbids, reachable on the first attempt. A degrade branch
keyed on `isinstance(exc.last_error, PostValidationError)` is MANDATORY, not optional.
Also: the ladder gives exactly **ONE** re-ask (base -> typed repair at the static
0.10 repair floor), and no `max_attempts` knob changes that.

**WHAT SURVIVES, if this is ever revived:** fire only on UNANIMOUS opposition
(opposing cues present, matching absent) -- measured 26/26 true, 0/5,037 false, and
it spares VIOLA; promote `portrait_gender_cues` out of `scripts/` into
`nodes/_otr_roster_gender.py` so the trigger and the acceptance ruler are ONE fact;
keep the FIRST schema-valid answer, not the last, because that makes the mechanism
provably never-worse-than-today (log the deviation from the ruling's "last" rather
than silently reinterpreting it); and do NOT reuse the name census -- its floor is 8
and at 8 it can never fire on 40-word prose.

**A DIFFERENT BUG IS HIDING IN THOSE ROWS AND DESERVES ITS OWN LOOK:** RICK STEINER
carrying LUCILLE PENNY's description, OYA SATO carrying Hank Griswold's. That is a
WRONG-CHARACTER PASTE, not a gender defect, and laundering it through a gender fix
would hide it. **-> THAT IS NOW ITEM I, and both instances are CONFIRMED.** Root
cause proven (two naming authorities arbitrated inside the description prompt),
promoted as Bible `11.61`, code fix shipped with the `media_archive` extension
live-closed 2026-08-21. Read item I before re-deriving
this item's portrait numbers -- several of its hits are gender-crossed and were
never gender defects at all.

**A LATENT GGUF SEED BUG, found in passing and unrelated to G:**
`_otr_gguf_backend` derives `per_call_seed = base_seed + _ordinal["n"]` to make
consecutive calls diverge, but `_ordinal` is rebuilt on EVERY call, so it is always 0
-- every GGUF call in an episode gets the identical seed. Masked today only because
repair prompts differ in text and temperature. Anyone "simplifying" a repair prompt
toward the base prompt silently loses call independence.

**H. SPLIT 2026-08-17 INTO H-RECEIPT (DONE) AND H-FLOOR (OPERATOR).**

> **H-RECEIPT IS SHIPPED.** The dispatcher's fourth `_neg_source` arm no longer
> reads `engine_hygiene`; it reads `none_contributed` and describes COMPOSITION
> only. The four arms now live in one pure helper,
> `otr_image_gen_dispatcher.negative_source_label`, with `NEGATIVE_SOURCE_LABELS`
> exported and FIVE tests pinning them -- the inline ternary was untestable, which
> is how the wrong value shipped. Panelled first (r1, scoped): two Antigravity
> calls (Flash High + **Pro 3.1**), a Fable pass, and the driver anchor; Codex
> excluded on quota. Artifacts in `kibitz-runs/2026-08-17-item-H-receipt/`.
>
> **THE PANEL CORRECTED THE DRIVER, AGAIN, ON EXECUTION ORDER.** The anchor
> claimed an engine-aware label would need "a production reordering in a loop that
> also computes cache keys, seeds and the banana transform". FALSE, both agy lanes
> independently: `_neg_source` is ASSIGNED once per row, `engine_id` is bound by the
> `resolve_engine_for_role` call BELOW that assignment, and the value is WRITTEN
> into the two `"negative_source"` ledger entries (the cache-hit branch and the
> fresh-mint branch) -- **both after engine resolution**. Pro sharpened it:
> `_neg_source` is **write-only telemetry**, three references in the whole file,
> decoupled from `prompt_hash` and the banana transform. (Cited by SYMBOL on
> purpose. The first draft of this block used line numbers, and the 42-line helper
> this very change added shifted every one of them -- the same trap item C recorded,
> re-sprung by its author within the hour. Sonnet's QA caught it.) **That is the second time in one day a panel caught an
> execution-order claim from this driver** (the 2026-08-17 style build was the
> first). The lesson generalizes: where a value is COMPUTED is not where it is
> USED, and only the latter decides what is in scope.
>
> **THE PANEL'S CONCLUSION WAS OVERRULED, ON PURPOSE.** Both agy lanes wanted the
> field made engine-aware rather than renamed. Their facts were folded; their
> conclusion re-commits the original defect -- a field named for composition
> asserting engine behaviour is TWO AUTHORITIES IN ONE VALUE, which is the shape
> that produced the lie. The rename also DISSOLVES the ordering coupling rather
> than working around it. Flash's proposed mechanism (`matching "z_image_turbo"`)
> was rejected outright: item A's ruling is that name-matching ships false
> positives.
>
> **ALSO FIXED, same change:** the enum in
> `docs/2026-08-17-one-style-authority-PLAN.md` documented
> `pack | pack+request | env_override` -- a value that never shipped, missing two
> that did. Evidence the vocabulary has no contract: it had already drifted once,
> unnoticed, with zero consequence.
>
> **NOW KNOWN FEASIBLE AND DELIBERATELY NOT BUILT:** per-engine hygiene telemetry
> as a SEPARATE post-resolution field (engines must DECLARE a floor, dual-read
> default per `engine_consumes_still`; never name-matched), and D-BIS finding 4's
> resolved-cfg / `negative_live` bool. Both add ledger surface, so both wait.

**H-FLOOR -- STILL OPEN, AND IT IS THE OPERATOR'S CALL, NOT A DRIVER'S.** Three
options, rejected 4/4 as a driver action because it changes conditioning at cfg 4.0
on a live engine and therefore owes a render: **(a)** no floor -- now honestly
reported, which was the only urgent part; **(b)** copy z_image's
`_HYGIENE_NEGATIVE`, the cheapest and the trap (z_image runs cfg 2.0, lumina 4.0,
different model, different artifact profile); **(c)** a lumina-specific string, most
correct and most work. Any of them needs one A/B at a fixed seed on the shipped
path. The original framing is kept below for its detail: `z_image_turbo._resolve_negative` ends
`.strip() or _HYGIENE_NEGATIVE` (`z_image_turbo.py:117`); `lumina_image` has
NEITHER the strip nor a floor, so an empty request negative reaches the encoder
as `""` and a whitespace-only one is passed verbatim. The path is REACHABLE
(`VISUAL_SAFETY_NEGATIVE_PROMPT` is `""`, and a pack may ship an empty
`negative_tail`). **The sharp end is a receipt that lies:** the dispatcher stamps
`_neg_source="engine_hygiene"` (`otr_image_gen_dispatcher.py:1169`) for exactly
that case, so the ledger claims a hygiene floor this engine does not have. A
comment asserting the two engines matched "including at the edges" was false and
is already corrected; what remains is the actual call. It is NOT a
copy-z_image-and-done: lumina runs cfg 4.0 against z_image's 2.0 and is a
different model, so its artifact profile is its own question. Either give lumina
a floor, or make the ledger stop claiming one it does not have -- but do not
leave the receipt wrong.

**REFRAMED 2026-08-17 by item C's work, and it is BIGGER than "lumina is
missing a floor". Read this before costing H.** Two corrections, both verified at
the files:
* **There is no `lumina_image._resolve_negative`.** That name belongs to
  `z_image_turbo` only. Lumina resolves its negative INLINE inside
  `_lumina_params`, as a bare `str(get("negative_prompt") or "")`. The behaviour
  this item describes is real; the function it names does not exist, so anyone
  planning "add the floor to `_resolve_negative`" is planning against a ghost.
* **THE RECEIPT IS NOT WRONG FOR LUMINA -- IT IS UNVERIFIED FOR EVERY ENGINE.**
  The dispatcher computes `_neg_source` from what the PACK and the OBJECT
  contributed, and it does that BEFORE it knows which engine will serve the row:
  `resolve_engine_for_role` is called about 56 lines LATER in the same per-object
  iteration. So the `engine_hygiene` arm asserts a property of an engine the code
  has not yet chosen. It is true of `z_image` by COINCIDENCE (that engine does
  have a floor) and false of lumina, but in neither case was anything consulted.
**So the options are no longer symmetrical, and the cheap one got cheaper:**
* **Fix the RECEIPT (driver-sized, zero pixels).** The first three arms
  (`pack+request`, `pack`, `request`) describe contributions actually observed and
  are honest. Only the fourth claims engine behaviour. Naming that arm for what
  the dispatcher actually knows -- no composed negative contributed -- stops the
  lie on ALL engines without touching a recipe or spending a render. Check the
  consumers first: `engine_hygiene` appears in exactly two code sites (the
  dispatcher's own comment and the value) plus docs, so the blast radius is small.
  A truly engine-aware label is a different, larger job: it needs the stamp moved
  after engine resolution, which is a production reordering.
* **Give lumina a FLOOR (operator's call, needs a render).** This changes
  conditioning on a live engine at cfg 4.0, so the recipes directive applies and
  trap 1 applies: budget a render whenever a negative changes. Not a driver
  decision, and the existing inline comment already says so in the code.

**DEFERRED, deliberately:** the visual-ledger AUDITOR. Fable's verdict is that
five of its six proposed checks audit BOOKKEEPING written by the code being
audited, and the only pixel check is Part 3's uncalibrated one. Build it only
with the anchor-chain check promoted (flag when a scene still's
`reference_latent` anchor is itself the episode's outlier and N stills derive
from it -- that compound signature IS the measured episode) and corpus mode
reframed as the CALIBRATION run for Part 3 (18,458 stills survive on disk; 174
episodes on the mis-served packs).

### THE OLDER ORDER, 2026-08-16 LATE -- COMPRESSED 2026-08-18 (fully superseded; full text in git history at any commit before this one)

Its one standing ruling -- defer all Lemmy/video listen-and-eyeball sessions until the operator returns -- is now MOOT: live blinded listens ran this session (the 0.560 emotion-ceiling audition and the neutral-line verification, both 2026-08-18) and the operator judged them directly. Nothing else in the compressed span was still load-bearing; every item in it either shipped, was superseded by the queue above, or was a measured-dead-end note already summarized elsewhere in this file.

### A. THE CLOSING ANNOUNCER DOES NOT NAME WHAT IT ADAPTED (3 banks, one family)

| bank | what shipped | what is wrong |
|---|---|---|
| `shakespeare` | `spoken_coda_source: none`, closes *"We've lost our signal."* | names NOBODY -- not the play, not Shakespeare |
| `public_domain` | coda fires, says the literal words *"public domain work."* | never names the title or the author |
| `media_archive` | coda type `news_close_brief` | it is an ARCHIVE bank (LoC / Film Preservation posts) being asked to summarize a news story it never had |

**The shakespeare row needs an OPERATOR RULING ON WORDING before it is coded.**
It looks like an over-application of the 2026-08-05 licensed-source ruling:
that ruling stopped the announcer reciting a LICENCE, and its own reasoning was
*"Folger publishes the edition and Shakespeare wrote the play"* -- yet what
shipped also stopped it naming the author and the play. Naming Shakespeare is
not a licence claim. Ask before writing the sentence.

### D. MEASURED TWICE -- but this section's ROOT-CAUSE WORDING WAS DISPROVEN 2026-08-17

> **READ ITEM F IN THE QUEUE BEFORE THIS SECTION.** The MEASUREMENT below stands.
> The sentence *"the announcer FRAME is sampled independently of the selected
> excerpt"* is FALSE in both halves and was disproven by r1 (two agy lanes plus
> the driver's own trace): nothing samples a play, and the frame does not read a
> different record -- it is never handed one. Item F carries the corrected root
> cause, the three producers, and the adopted plan.

`tempests_midnight_revelations` was the suspicion; the 2026-08-16 blind
narrative read is the measurement
(`docs/2026-08-16-blind-bank-narrative-ranking.md`): a Twelfth Night scene
announced as *"Verona ... Capulets and Montagues"*, and a Tempest scene
framed as Romeo and Juliet -- both current-era `shakespeare` episodes. The
announcer FRAME is sampled independently of the selected excerpt instead of
being generated from the same metadata record. FIDELITY defect on the lane
where fidelity outranks arc; deterministic root shape; same family as the
speaker-tag leaks, corrupted-text lines and truncated closers the same read
surfaced. A real fix candidate for the next coding window -- panel it first
(the root cause is diagnosed by SHAPE, not yet traced to the exact sampling
site).

### OWED -- the Bible promotions, one per green chunk

The delta-scrape is DONE and must never be re-run (it cost ~4M tokens once).
Seven uncovered shapes are drafted as `12.103`-`12.109` in
`docs/2026-08-15-BIBLE-PROMOTION-DRAFT-bugfix-sprint.md`, each with an
automatable verify clause and its index row; 02(b) needed none, already covered
by `11.18`.

**They are deliberately NOT promoted yet.** A Bible entry's `fix:` field claims
something was fixed and proven, so each one lands WITH its green chunk per the
contract's documentation gate. Also note PBUG-20260815-07 is a REJECTION,
recorded so nobody re-chases the `original`-lane voice report.

## OPERATOR RULINGS 2026-08-15 (hard -- these settle three OWED decisions)

**THE EPISODE SHAPE IS A REQUEST, TOP TO BOTTOM. NOT THE BEATS, NOT THE ACTS.**
*"It's like chasing words again -- no chasing beats."* *"We ask for x beats, we
do our best, we don't fail if it doesn't have exactly as many beats."* *"If it
writes 1 act [when] I request 7, fine."* This is the word-count law extended to
every structural number: nothing may refuse, reroll, retire or clip an episode
for the count it came back with, and **no test may pin one either** -- a test
that asserts "an act is exactly N beats" is the same gate with pytest holding
it. Direction is checkable (asking for more must never ask for less); size is
not. In practice the act count is guaranteed by CONSTRUCTION anyway -- the
outline runs one authoring pass per arc phase -- so only the beats inside a
path are the model's answer.

**NO HARD SCHEMA CEILING MAY SHAPE A SPINE.** *"If I ask for 7 acts it needs to
generate a spine of 7 acts."* `_RADIO_SCORE_MAX_SCENES = 3` x
`_RADIO_SCORE_MAX_BEATS_PER_SCENE = 4` was a hard 12-beat cap on a WHOLE
episode at ANY act count, and the score's scene is that lane's act-sized unit
-- so a 7-act pick could not produce a 7-act spine, and because the lane
decodes under a grammar built from the schema it TRUNCATED mid-generation
rather than refusing. Caps are DERIVED from the topology now
(`MAX_ACT_COUNT * _SCHEMA_HEADROOM`, `BEATS_PER_ACT * _SCHEMA_HEADROOM`), so
raising the topology carries the schema with it. They stay finite: runaway
guards remain code-side.

**AN ACT IS 4 BEATS, AND EVERY ACT IS AN ACT PATH.** *"Ideally we say each act
is 4 beats and we have a separate LLM pass per beat."* `BEATS_PER_ACT = 4`,
with `voiced_beats_per_act` DERIVED from the arc phases so the two cannot
drift. This fixed two inversions in the old hand-tuned table: 7 acts asked for
19 beats while 6 asked for 20, and 3 and 4 both asked for 14. Per-beat
authoring passes were already shipped (`per_beat_dialogue_then_scene_review`).
**Cost, measured:** ~1.7 min/beat story-only on `gemma-4-12b-it`, so 3 acts
runs ~12 beats and 8 acts ~32.

**`media_archive` IS ALREADY LIVE RSS** (Library of Congress + National Film
Preservation feeds, fetched every run) -- the defect was that it always took
pooled entry 0, so it retold the newest post forever. Ruling: pick it *"similar
to [the] science news picker"*, which means the three mechanisms that lane
already ships -- dedup against `news_history.json` so a story is never told
twice, a MODEL ranking candidates by narrative fit, and recording the choice --
not the seeded shuffle first proposed. **OPEN.**

**THE `__main__` SELF-TEST BLOCKS: DELETE THEM.** *"If they aren't doing
anything delete em."* **OPEN.**

### THE SHIPPING RECIPE, measured -- do not re-derive it

**Whole-line judging on `gemma-4-12b-it`.** 13/15 recall, 24/28 traps kept,
13 rows repaired, 79 calls. The clean stage has a MODEL FLOOR exactly like the
news lanes do: on `gemma-2-2b-it` precision collapses to 61% and it repairs
less than half as much while spending MORE calls.

**Rejected by measurement -- do not re-propose without new evidence:**
agreement voting (-4% precision for +45 calls); the load split as a quality fix
(neutral, marginally cheaper, kept available as `REPAIR_READS_BRIEF_ONLY`);
per-sentence judging on a large model (strictly worse than whole-line there).

**`JUDGE_PER_SENTENCE` is a real lever and it is OFF.** It is the only thing
that ever broke the 13/15 recall ceiling -- 15/15 on the 2B, by shrinking the
JOB rather than improving the prompt. It costs precision, and specifically it
costs `shakespeare` (2/5 traps) and `public_domain` (2/4), which is where the
operator ruled a false positive IS a real defect. Turn it on only for a forced
2B run where recall beats politeness.

### CHARACTER DRIFT IS ACCEPTED. DO NOT CHASE IT. (operator 2026-08-14 -- HARD)

*"We just accept character drift, don't chase. Be honest in the README -- note
that if they want to chase it, they need a frontier model above what my 16 GB
card can do."*

**F2's CONTENT half is CLOSED as a work item** -- not parked, not a TODO. It
was built, lab-tested on the shipping model, failed, and ships disabled
(`JUDGE_ATTRIBUTION = False`). Recall swung **3/6 then 1/6 on identical
fixtures with a byte-identical judge**; it never once caught `role_claim` or
`knowledge_mismatch`, and the "same words, right mouth" trap fooled it both
runs. A detector that unstable cannot be trusted with a rewrite.

Documented honestly in `README.md` under "Known limitation: character drift",
including how to enable it and how to MEASURE it
(`scripts/otr_clean_stage_lab.py --f2 --model <...>`).

**Reopening it requires NEW EVIDENCE, and that means one thing only:** a
measured run on the F2 fixtures showing a model both accurate AND stable across
repeats. This does not reopen the 2026-08-04 story-quality law -- character
consistency is still carved out of it as a correctness defect. We measured what
fixing it costs on this hardware and the operator priced it.

**F1 -- action in a spoken row -- remains ON and is the shipped capability.**

### DO NOT "FIX" `clean_spoken_text`. THE OPERATOR RULED ON IT 2026-08-05.

An earlier cut of this file ranked it the worst no-shims violation in the tree.
**That was wrong**, and acting on it would have broken a working design.

`_otr_captions.py:19-40`: performance direction is shown on purpose. A caption
burns the RAW ledger line while TTS independently strips it via
`clean_spoken_text`, so caption and audio diverge BY DESIGN -- 255 cues across
95 of 915 shipped episodes, *"a nice easter egg as long as it's built and we
know and it's documented"*, pinned by `tests/test_otr_captions.py`. And the raw
text is load-bearing on the visual side: `_otr_motion_clause._line_text_index`
reads raw `lines[].text` for the i2v motion clause under *"the line drives the
motion"*, plus the still prompt and the HUD.

It is a two-surface design: `lines[].text` is the canonical direction-bearing
record, the microphone gets a projection of it. The no-shims law is about who
may WRITE the record; the TTS strip writes nothing back. The 80/40 character
caps are deliberate bounds on deletion power, not a latent bug.

**What would reopen it:** a live episode where the regex deleted words a model
had BLESSED as genuine speech -- greppable as rows where
`clean_spoken_text(text) != text` with no `unclean_spoken_text` flag and no
policy finding. Narrow the net then; never remove it.

### OPEN -- the one no-shims violation that survived the rip

The 2026-08-14 audit found three; the 2026-08-16 `scifi_news` rip closed two
(codex `P5R _call_scene_review` and `_canonicalize_script_spoken_text` died
with the module). One remains:

* **`_otr_content_safety.py` is dormant but loaded** -- hardcoded
  `PROFANITY_TERMS` / `EXPLICIT_WEAPON_TERMS` / `EXPLICIT_NUDITY_TERMS`
  (`:25-82`) driving model rewrites, contrary to the 2026-08-03 no-guardrails
  directive, plus two bare `RuntimeError`s (`:328`, `:334`) that would kill a
  render. Nothing calls it. Delete it or rebuild it before anything wires it
  back.

### OPEN -- a tension the clean stage created, worth one look

The clean stage REWRITES `lines[].text`, which is the field the 2026-08-05
ruling calls load-bearing for stills and i2v motion. It degrades gently -- the
repair FOLDS the action into the speech rather than deleting it, and
`compute_source_hash` includes the dialogue so a changed line correctly
regenerates its motion clause -- but two rulings now pull on the same field
from opposite ends. **Look before the clean stage is trusted on a VIDEO leg;
it does not affect the audio path.**

### OPEN -- the model floor should REFUSE, not grind

A model that cannot satisfy a lane's contract should be refused at the top of
the run WITH THE REASON, not discovered after 34 minutes of bounded retries.
Measured: `gemma-2-2b-it` drove `original` at act 3 to a clean ledger in 5.1
minutes and could not get `scifi_news` past P0 in 34. It was not a runaway --
it was bounded, honest, futile work. **The defect is the silence, not the
floor.** Proven for `scifi_news` / `scifi_news_pro`: `gemma-4-12b-it` and
`Mistral-Nemo`. Do NOT delete `gemma-2-2b-it` -- it is the fastest writer-lane
model on the box.

### OPEN -- F1's last miss class

The three surviving misses on the 12B are all `scene_report`: a scene sentence
and real dialogue sharing one row ("The monitor flatlines. Someone should call
the desk."). The per-sentence lever catches them and costs precision elsewhere;
a calibration example for the MIXED row is the cheaper thing to try first.

### OPEN -- still suspect, deliberately not fixed

The codex lane carries the cap-equals-trim shape on `MAX_FACT_ROWS` (6) and
`MAX_ENTITY_ROWS` (4). That lane decodes under a grammar, so the ceiling
TRUNCATES during generation rather than refusing -- silent evidence thinning,
not a crash, and unproven on an artifact. Six facts and four entities is tight
for a real news story. **Prove it on an artifact before touching it.**

### RULED OUT BY MEASUREMENT -- do not re-propose without new evidence

| Rejected | Why |
|---|---|
| `repetition_penalty` | **inert here**, up to its maximum -- HF's penalty is not frequency-aware |
| `no_repeat_ngram_size` | unusable with the grammar -- the ban and the mask both write `-inf`, the intersection can be EMPTY and sampling crashes |
| lower temperature | helped one run, failed another. Not reliable |
| agreement voting on the clean judge | -4% precision for +45 calls |
| per-sentence judging on a 12B | strictly worse than whole-line there |
| a whole-episode "final table read" pass | the `PBUG-20260814-03` failure shape in reverse -- a small model handed a whole episode averages it into a summary |

## CURRENT RUNWAY -- OPERATOR-ORDERED 2026-08-13. WORK IT TOP TO BOTTOM.

**ROW 1 BELOW IS SUPERSEDED by PRIORITY 1 above (operator 2026-08-14).** Its
order was "resume the upstream Story Lab and A/B against it"; the lab is now
parked, read-only and being retired, and the story work happens in production.
Rows 2-5 are unchanged and still run in their listed order, behind PRIORITY 1.

This block is authoritative. It supersedes every older order, count, Lemmy gate,
and review-routing sentence lower in this file. Re-ground the active row against
the Windows tree before editing; when a row is coded, fully tested, committed and
pushed, move its receipt to `docs/HANDOFF_LOG.md` and remove it from this forward
queue in the same push.

| # | Active work | Exit condition |
|---:|---|---|
| 1 | **SUPERSEDED -- what remains is PRIORITY 1 at the top of this file** | Do NOT restart from the Story Lab. |
| 2 | **Give LEMMY a fighting chance: complete Phases 2-4 and its three live PBUGs** | Preserve the Cockney floor with one upstream engine-policy authority wired through the canonical workflow, CastLock and renderer; qualify real routes by operator-audition receipts; close the six-engine gender-only pin gap; restore or explicitly decline `scifi_news` cameo policy; resolve the fable2 BAD_LINE interaction; re-observe the missing closing before diagnosing. No silent substitute and no defined-but-unwired policy. |
| 3 | **Run seven fresh post-change 45-word render proofs** | All seven exact public engine IDs pass against the post-bugfix/post-Lemmy HEAD with `COVERS`, `RESULT SUCCESS`, server `Prompt executed` + `obs_publish OK`, and the canonical OBS asset on disk. See **WHAT IS ACTUALLY LEFT** below. |
| 4 | **Narrow learned-upscale hardening only** | Harden the two `SpandrelEsrgan._resolve_model` edge cases if still reproducible. The multi-GPU learned-upscale stage itself is CLOSED and must not be reopened. |
| 5 | **Release runway** | `ROADMAP.md`: lean-mean -> RunPod/AMD/Mac -> install -> product docs/v2 release. |

### THE STORY LAB IS RETIRED -- two guardrails survive it

The lab ([`jbrick2070/ComfyUI-OTR-UpstreamStoryLab`](https://github.com/jbrick2070/ComfyUI-OTR-UpstreamStoryLab),
`main` = `7df7c80`) is READ-ONLY reference. **Do not develop in it. Do not ship
its duplicate workflow, production mirror or bridge into OTR.**

Its two detector files were inventoried and NOT ported, deliberately:
`spoken_text_policy.py` and `ledger_verifiers.py` are both REGEX detectors --
the lab's own header says *"a future fuzzy or model-assisted policy requires a
new policy"* -- so porting them would not solve the generalization problem the
model judge was built for.

**The canonical `scifi_news` episode topology STANDS and is still the
contract:** opening music -> ANNOUNCER introduction -> character drama
with interstitial music only where the script asks -> ANNOUNCER source-backed
real-news summary -> closing music. Both announcer bookends and the opening and
closing music are structural reservations independent of `target_words`. The
opening establishes story, place and time and connects the premise to the news;
the ending summarizes the real news and distinguishes fact from fiction.

## ON DECK -- WHAT REMAINS OF CONTINUITY CORRECTNESS

### 0-QUINQUE. MINIMAX H3 -- A SPRINT SERIES ON THE VIDEO PATHS (operator, 2026-08-09)

**THE RULING IS IN, AND IT DISSOLVES THE OLD QUESTION.** This section used to
say the next step was an operator ruling on "does H3 belong in the video
dropdown given the 4 s floor vs the sub-4 s beats". That framing is RETIRED.
The operator's 2026-08-09 direction: H3 is **"a series of sprints all to refine
the video paths"** -- scope TBA.

**What that changes for a window picking this up:**
* It is NOT a yes/no dropdown admission any more, so do not go looking for a
  verdict to record. There is nothing blocked on the operator here.
* The unit of work is a SPRINT against the video paths, not a one-shot chunk.
  Expect several, each with its own kibitz gate, each landing green and pushed
  on its own.
* The 4 s floor is now an INPUT to that refinement -- a constraint the video
  paths have to accommodate or explicitly route around -- rather than a
  disqualifier that settles admission.

**Scope is TBA and that is deliberate.** Do NOT invent the sprint list. When the
operator names the first sprint, write it into THE QUEUE at the top of this file
as its own row, and leave this section as the standing context.

**Grounding that survives the reframing:**
* Problem statement `docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md` is
  UNTRACKED and another window's working file -- never stage, edit or delete it
  from a different window. Read it; do not touch it.
* The matrix-pattern spec already names MiniMax as a churn driver
  (`docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` section 5), so a video-path
  sprint will likely collide with the un-converged matrix work (section 0). Read
  that section's "what survived all four rounds" before designing anything.
* The recipes are NOT on the table (standing directive). A video-path sprint
  refines PATHS -- routing, canvas negotiation, admission, extension -- never
  the shipped render recipe.

### 1. Z-Image generic reference conditioning is CLOSED and OUT

The permanent matched harness is `scripts/otr_zimage_reference_ab.py`; its live
artifacts are under
`output/otr/episodes/zimage_reference_ab_20260820/stills/{off,on}`. Each arm used
the same installed NVFP4 UNET, Qwen FP8 encoder, VAE, prompt, negative, seed 7,
1472x832 canvas and eight-step recipe on a separate fresh server boot. The OFF
arm was clean. The ON `graph.json` structurally proves the exact dual
`ReferenceLatent` chain reaches both sampler conditionings; its separate
`SUCCESS` receipt and fresh output prove that submitted graph executed. The ON
pixel output reproduced the square grid across the walls and clothing. This is
a by-eye pixel verdict; no gridscore number is evidence.

**DECISION:** generic `ReferenceLatent` is not an approved semantic path for the
installed Z-Image Turbo checkpoint and is OUT of production. A node accepting a
graph proves structural compatibility, not training compatibility. The engine
now advertises `accepts_reference_image=False`; `engine_version` is `2` so every
possibly gridded v1 still misses cache. The portrait-derived deterministic seed
remains enabled, so character scenes stamp `portrait_anchor_mode='seed'`, never
blank. **Correction from the 2026-08-21 mixed canonical QA:** that seed prevents
random anchor selection but does not guarantee face/costume identity across
beats; the open visual-identity item is in THE CURRENT STEP. The diagnostic
graph remains only so future weights can be retested against the same
single-variable harness; it is not a production fallback.

### 2-PRE. OPERATOR CALLS ALREADY MADE -- do not re-open, do not re-panel

**BANANA ROUTE: VISUALS ONLY. The spoken script is NOT touched
(operator ruling 2026-08-06).** Operator, asked whether the filter should reach
spoken lines as well as image prompts: *"No. Just visuals. I do not want people
discussing the Cavendish versus the other variety."*

So the substitution happens on the STILL/VIDEO PROMPT and nowhere else. The
announcer still says "he drew his revolver" over a shot of a man holding a
banana -- which the problem statement flagged as either the joke or the thing
that breaks it. It is the joke. **The dialogue ledger, the writer, and the
adaptation lanes are all out of scope**, which also keeps this clear of the
closed story-quality directive and of the fidelity lanes' invents-nothing rule
at the TEXT level.

This closes the second half of section 7 of
`docs/2026-08-06-PROBLEM-STATEMENT-banana-route.md` (committed `9c686886`).

**THE DEFAULT AND THE REACH ARE ALSO RULED NOW (2026-08-06, `ec9da848`) --
that question is CLOSED, do not reopen it.** Global default **ON**, with
`shakespeare` + `public_domain` defaulting **OFF** via the copied `_LEMMY`
exclusion idiom, plus `OTR_BANANA_INCLUDE_FIDELITY_BANKS` as the operator's
force-on override. **NO node widget and NO `workflows/otr_canonical.json`
change.** So `Is this a dagger which I see before me` stays a dagger on the
fidelity lanes unless the operator flips the override. Two env switches
(`OTR_BANANA_STILLS`, `OTR_BANANA_VIDEO`), one per funnel. The whole contract is
`docs/2026-08-06-BUILD-SPEC-banana-route.md` at `ec9da848`; SHIPPED -- see
section 0-QUATER above (`bc8a1bde`).

### 2. Operator calls nobody can make for you

- **ARIEL and PUCK.** The curated supplement ships 10 entries and deliberately omits
  these two: Folger's stage directions use "he" for both, but neither has a roster fact
  and both are editorial. They stay on the roll, so the corpus gate asserts 40 of 42.
  Say the word and they become 42.
- **Tier floor 2 or 3.** Shipped at 2: it removes every 100% voice pin while 5 of 24
  (engine x gender x timbre) combos still honour the requested timbre. Floor 3 also
  removes them but leaves only 2 of 24 honouring timbre -- it buys spread by deleting the
  dimension. `OTR_CAST_MIN_TIER_POOL=3` makes it a one-leg A/B, and the floor is folded
  into the cast seed so the two settings can never both claim policy '3'.
- **`num_characters` is still 2.** Every published adaptation ran 2, so a 7-speaker scene
  loses five people. Correct gender for two survivors is still a truncated scene. This
  collides with the count-match invariant at `OTR_LedgerScriptWriter.py:4119` and is its
  own piece of work, not a tail of this one.

### 3. Standing facts worth not re-deriving

`slot.gender` is NOT a voice field. It feeds the description LLM
(`_otr_casting.py:777`), the outline prompt (`OTR_LedgerScriptWriter.py:4144`), the
dialogue cast block (`_otr_line_composer.py:446`) and the image prompt's gender anchor
(`otr_meta_brief_image_prompt.py:78-90`). **The gender fix therefore changes scripts and
portraits.** That is a downstream consequence of a correctness fix, not a violation of
the closed story-quality directive -- exactly as "Malvolio speaks with a woman's voice"
is a bug while rewriting his dialogue is not.

Do NOT feed pinned genders into `prior_genders`, and do NOT re-call
`_plan_gender_distribution` with a reduced count. Measured: `(1, ['male'])` returns
female 400/400, and the shuffle's stream consumption varies with count (getrandbits
0, 0, 3, 3, 9, 11 for counts 0..5). The shipped design overrides in place and leaves the
allocator untouched.

**Source-grounding sprint, the one piece left:** chunk 3b-ii -- the supply line
that feeds grounding into the writer -- is BUILT-BUT-UNWIRED and PARKED under the
story-quality directive. The delivery mechanism exists and nothing calls it. A
contributor may pick it up; chunk detail under THE CODING SPRINT item 1.

## THE CODING SPRINT (operator directive 2026-08-04; re-sized by the r1-r4 arc)

Item 1 is the structural work and consumes most of a session; items 2-3 are
small and share one campaign. Items 8 and 9 are DONE (receipts in
`docs/HANDOFF_LOG.md`). The live open work is sections 0 (video matrix pattern,
did NOT converge), 0-BIS (no-mirror, CODE-READY), 0-QUATER's deferred
shield-scoping chunk (own kibitz arc), and the 0-QUINQUE MiniMax ruling.
Work by priority, not by number -- the numbering is historical.

**RENDERS HAVE RESUMED** (2026-08-05). The 08-04 "no render runs this session"
line is spent -- it governed that session only, and the 08-05 handoff opens on a
live-proof obligation. Reset per `CLAUDE.md` section 4 before any leg: selective
CIM kill by CommandLine, never a blanket python kill (it severs the MCP tooling).

Everything below was verified against the real files on 2026-08-04, is
non-GPU, and is provable by the suite alone. Work them in order; each ends
green and pushed on its own.

### 1. THE PUBLIC_DOMAIN LANE IS TOLD TO CARRY WORDS IT IS NEVER SHOWN (the session's main work -- r1 panel re-scoped 2026-08-04)

The headline defect, and the one that manufactured "Arkham, Massachusetts" over
H. G. Wells. The pack orders the model to carry the author's language:

* `nodes/story_packs/public_domain/faithful_radio_adaptation.json:13`
  (`exchange_system`) -- "Where the source gives these characters words, CARRY
  THEM. Keep their diction, their rhythm, their argument."

And `nodes/_otr_compose_exchange.py` (994 lines) has **ZERO** references to
`source_text`, `full_text`, `source_meta` or `excerpt` -- verified by grep.
**The instruction is bound to an absent document.** A model told to carry words
it cannot see will invent words and believe it complied.

**SCOPE RULING (r1, grounded against `docs/2026-08-03-fidelity-pass-ownership.md`
line 25): this item is PUBLIC_DOMAIN ONLY.** The ownership table rules
`exchange_compose` **NOT RUN** on the Shakespeare verbatim lane ("It exists to
author dialogue. There is no dialogue to author."), so enhancing the composer
for Shakespeare invests in a pass the verbatim executor removes. Shakespeare
gets exactly ONE change this sprint: its dangling comma (item 3). The keystone
"compile source speech, do not generate it" (THE ADAPTATION DESIGN) binds the
VERBATIM lane; `public_domain` is the operator-ruled FUZZY PROSE lane, where
grounding the generative composer is the correct move, not a contradiction.

Three legs, ALL required -- the panel killed the raw-injection shape:

**(a) A BOUNDED source window over the COMPLETE canonical body -- never the
payload's `full_text`, which is itself truncated.** r2 correction of this
plan's own premise: `canonicalize_public_domain_text(..., max_chars=12000)`
(`_otr_public_domain_sources.py:337-343`) truncates at 12,000 CHARS, and
`payload_from_manifest_unit` stores THAT as `full_text` -- while the corpus
runs **916 words (`cradle_protocol`) to 25,200 words (`beckoning_fair_one`)**
across 65 units. So "the material already arrives, it needs passing" is false
for large sources: the payload carries a prefix. The selector reads the
complete canonical body from the SOURCE layer, separated from the interpreter
excerpt. Hash discipline: exactly ONE of 65 units ships a provenance sidecar
(`time_machine__arrival.provenance.json`), and its `body_sha256` covers
normalized RAW bytes, not the canonicalized body -- two NON-interchangeable
fields. Derive a `canonical_body_sha256` at fetch/selection time, bind
selection + receipts to it, and do NOT call it authenticated provenance. Do
NOT migrate the 65 closed manifests for it (`_SOURCE_KEYS`/`_UNIT_KEYS` closed
at `:48-63`); carry it in `source_meta` and snapshots. Coordinate system (r3):
refactor `canonicalize_public_domain_text` into an UNCAPPED normalization
owner plus a separate 12,000-char legacy payload projection; spans are
half-open Unicode char offsets (`start_char`/`end_char`) into the uncapped
string; `canonical_body_sha256 = sha256(canonical_body.encode("utf-8"))`;
stamp normalization + selector versions. Transport (r3): `SourceFetchResult`
exposes only payload/source_meta/source_rights and `_resolve_inputs` collapses
to a three-tuple, and the snapshot envelope is the SEVEN-KEY payload
(`_otr_source_snapshot.py:48-50`) whose `full_text` is the truncated prefix --
so extend the PUBLIC-DOMAIN snapshot with the CANONICAL BODY as the SOLE
replay authority (r4 cut the "or exact selected text" alternative -- selected
text cannot recreate pre-outline grounding or select windows for a NEWLY
generated outline), under a versioned body/hash/normalization contract. A
legacy seven-key snapshot FAILS with a typed grounding-version error -- but
ONLY when the snapshot's bank is `public_domain`/adaptation (r4, both lanes
converged): the seven-key envelope is the UNIVERSAL loader, and an
unconditional rejection would break every other bank's existing snapshots and
bake-off replays. Keep the full document OUT of meta/ledger (`source_meta` is
copied into durable metadata at `:3548`). Budget: capacity
is EVERY backend, not GGUF alone -- the fitting seam
(`_otr_generation_budget.py:132`) spans GGUF (`estimate_prompt_tokens`,
estimator, `_otr_gguf_backend.py:1264-1273`), OpenRouter, Google and Comfy --
so select the window against the COMPLETE assembled message (system seam,
cast, prior lines, contracts, source block, output reservation), reserve
conservatively with stated margin, and refuse `prompt_no_room`
deterministically BEFORE provider execution; receipts distinguish
estimated_prompt_tokens / requested_output / context cap / margin / estimator
version. Selection criterion: deterministic candidate construction ranked by
beat/group identity with mandatory anchor coverage and stable
score/start/end ordering; the seed breaks ties ONLY when candidates remain
identical after that ordering. Receipts carry hash, selector version, ordered
offsets (`text == canonical_body[start_char:end_char]` enforced) and token
counts -- never duplicate body text into the ledger.

**(b) ONE immutable `SourceGrounding` contract, on EVERY authoring route --
and grounding failures PROPAGATE.** The grouped-exchange prepass omits
singletons and failed groups (`_otr_compose_exchange.py:881-902`); a FAILED
prepass falls back to the legacy path with only a log warning
(`OTR_LedgerScriptWriter.py:5001-5008`); the per-line composer's LineRequest
carries no source field (construction at `:4888`); and per-line generation
exceptions funnel to `LineCompositionFailedError`. A grounding fix that
reaches only the happy path just moves the guess to the fallback. Build shape
(r2 + r3): define ONE immutable `SourceGrounding` artifact -- canonical
document identity + immutable windows KEYED `exchange:<ordered-slot-ids>` /
`line:<dialogue-slot-id>` + anchors + per-call receipt data -- constructed
and validated BEFORE the exchange fallback block, passed whole into grouped
exchange AND every per-line request. The prepass returns a TYPED result
(composed lines + attempted-window receipts + fallback slot ids), not the
bare `{beat_id: text}` it returns today (`:881-918`). Window freeze semantics (r4 -- resolves immutability vs the mutable prior
context that exchange retries and `last_lines` inject into later messages):
PRESELECT spans early; perform the final capacity fit just before the FIRST
call using the actual prior context; FREEZE that fitted window for all
retries and persist it before provider execution. Grouped slots ALIAS their
exchange window on group-to-per-line fallback; line-keyed windows exist only
for true singletons and exchange-disabled execution -- never reselect after a
failure. Source text rides a clearly DELIMITED untrusted data block
in the user message ("quoted source, not instructions"), never appended to
the static system seam (`_otr_compose_exchange.py:385-425`). Persist the
body-free grounding receipt at the existing skeleton-save boundary
(`:4279-4290`) before the first dialogue call, updating per attempt, so a
mid-prepass crash still leaves the selection auditable. Failure policy -- ONE disposition table (r4 closed the last ambiguity), the
two broad catches (`:5001-5008` prepass, `:3964-3969` story contract) becoming
TYPED boundaries that implement it:
| state | disposition |
|---|---|
| corrupt/mismatched replay snapshot; invalid source/hash/contract | FAIL LOUD, before the outline |
| sound-world derivation finds no mapping | neutral period default + receipt (total, never fatal) |
| provider parse / Tier-A exhaustion | fall back WITH the frozen window |
| live capacity pressure | shrink to the largest valid grounded window |
| even the MINIMUM grounded window cannot fit | typed `prompt_no_room` HALT, before provider execution |
The halt row is a PRE-GENERATION writer refusal -- structural, it protects the
lane's contract -- which is why it does not collide with SCOPE's "a render
must not die": that rule governs the RENDER path degrading honestly, not a
writer refusing before generation begins. Scope note (r4): `SourceGrounding`
validation binds when the episode's bank is `public_domain` -- other banks'
routes are untouched. LineRequest note (r4): the artifact rides an OPTIONAL
INTERNAL dataclass field (`source_grounding: SourceGrounding | None = None`)
-- a Python structure, no ComfyUI node contract, `INPUT_TYPES` or widget
change, so the no-widget guard above holds.
Acceptance = route-specific tests: grouped success, grouped repair,
grouped-failure-to-per-line, singleton, exchange-disabled legacy, snapshot
replay (new envelope AND legacy-envelope typed refusal, public_domain-scoped),
hash mismatch, exact-capacity rejection -- plus a corpus-wide property test
over all 65 units proving normalization idempotence, canonical-hash stability
and `text == body[start_char:end_char]` for every emitted span (r4). Version
discipline (r4): the existing constants are `PROMPT_VERSION =
"public_domain_interpreter_v2"` / `SCHEMA_VERSION = "public_domain_briefs_v1"`
(`_otr_public_domain_sources.py:36-38`); name and bump every changed one, and
give SourceDocument / SourceOverview / SourceGrounding / normalization /
selector / snapshot their own explicit versions.

**(c) World anchors, DERIVED FIRST -- and the sound world gets ONE owner that
feeds every surface.** Prefer deriving a typed grounding sidecar from EXISTING
metadata + the selected spans. New manifest fields are a LAST resort:
`_SOURCE_KEYS`/`_UNIT_KEYS` are closed frozensets
(`_otr_public_domain_sources.py:48-63`, same for `_SCENE_KEYS`), so new fields
mean a schema version + migration across all 65 units. AND the competing frame
must actually be disabled, not outvoted: the adaptation `sound_world` is a
content-blind draw (`OTR_LedgerScriptWriter.py:3962`, palettes at
`_otr_style_catalog.py:442-463` -- grate/mantel/teacup over whatever source
rolled it). r2 sharpened the shape: the catalog renders the drawn sound world
into `contract.grammar` SEPARATELY from the `contract.sound_world` stamp and
the canon derivation, so a stamp-only fix leaves the prompt grammar still
carrying the contradictory palette. ONE source-aware derivation function must
feed the stamp, the grammar and canon for `style_pool_class == "adaptation"`
(arc_shape gate at `:4325` is the shipped precedent), with an explicit neutral
period default when no mapping exists -- and it runs BEFORE the grammar is
built (or the grammar re-renders from the final contract), or the prompt
grammar keeps the contradictory palette while the stamp looks fixed (r3, both
lanes independently). DECIDE whether derivation failure is fatal: today's
broad catch silently disables the whole story contract. Reconcile with the
EXISTING anchors owner: `meta["specificity_anchors"]`
(`OTR_LedgerScriptWriter.py:4259-4266`) already derives and injects an anchor
projection -- the new source anchors REPLACE it or deterministically merge
into it, never run beside it as a second independent voice. Do NOT delete the
adaptation styles -- operator-authored 2026-07-14; fix the DRAW and the
plumbing, not the styles.

**Two receipts, named now so neither is overstated later:**
* `code-complete + suite-green` -- the most a session without the live leg can claim.
* `production-qualified` -- only after a canonical `public_domain` leg passes a
  rubric: no unsupported foreign place/character/object; the source's setting
  and principal event retained; provenance receipt complete; `obs_publish OK`;
  asset on disk.

**Two rules from the 08-03 craft brief, both hard-won, both easy to violate:**
1. **Never name the feared failure.** Writing "no Arkham" into a prompt IMPLANTS
   Arkham. Forbid by CATEGORY, never by example.
2. **Every fidelity instruction must be PAIRED with the material it binds to.**
   An unpaired "carry the words" is the bug, not the fix.

**Size honesty (r1) and CHUNK ORDER (r3 -- the naive order was CYCLIC):** this
is THE SESSION, not 90 minutes. r3 caught a dependency cycle in the obvious
build order: the sound world feeds `contract.grammar`, the grammar is consumed
by the OUTLINE (`OTR_LedgerScriptWriter.py:3948-3963` -> `:4129`), and beats do
not exist until the outline returns -- so a sound world derived from
beat-keyed windows is impossible. Build in THIS order, one green pushed chunk
each:
**CHUNKS 1 AND 2 ARE DONE AND PROVEN ON RENDERS. Chunk 3 (the grounding supply line) is PARKED under the story-quality directive -- the Source-grounding note in section 3 above is authoritative; a contributor may pick it up.**

**Carried into chunk 3 from the chunk-2 QA (do not lose):** snapshot replay
has no whole-body carrier, so an adaptation lane replaying a frozen source
falls back to the drawn palette and a live run and its replay produce
different sound worlds. The tempting fix -- rebuild the document from the
snapshot's `full_text` -- is WRONG and was rejected: that field is the
truncated projection, so it would mint a document whose total-coverage
guarantee describes a prefix. The correct fix is the snapshot-envelope
extension already specified in 1(a) below.

1. **Uncapped `SourceDocument` + a pre-outline `SourceOverview`** (r4): split
   the normalization owner, then derive deterministic COVERING windows with
   exact-span evidence for cast, setting, principal turns and ending. This is
   what grounds the PRE-OUTLINE authors -- the interpreter today reads the
   CAPPED payload (`_otr_public_domain_sources.py:520-543`, running at
   `OTR_LedgerScriptWriter.py:3748-3757`) before contract (`:3948`) and
   outline (`:4129`); beat-keyed grounding alone arrives too late for them.
   Transport (r4): ONE transient typed field --
   `SourceFetchResult.source_document` -> typed normalized result ->
   `resolved["source_document"]` -- MECHANICALLY excluded from meta/ledger
   serialization; snapshot replay reconstructs the same type.
2. **Contract / grammar / outline from the overview's document-level
   anchors**: the one derivation function runs BEFORE grammar build (or
   grammar re-renders from the final contract), feeding stamp + grammar +
   canon. Pre-outline derivation uses DOCUMENT-level anchors only -- selected
   spans do not exist yet (r4 wording fix).
3. **Beat-keyed window selector + `SourceGrounding` threading + typed failure
   boundaries** (post-outline, when beats exist). The route matrix must NAME
   the announcer routes -- intro / rewrite / outro authoring at
   `OTR_LedgerScriptWriter.py:5104-5116`, `:5272-5285`, `:5357-5409`
   (verify-at-build) -- and decide per route: grounded, or constrained to
   already-grounded accepted fields.
No node signature, widget, link or schema change is intended anywhere in this
item -- the canonical JSON stays byte-identical through the sprint; if any
chunk turns out to need an INPUT_TYPES change, section-0 same-commit rules
apply and the plan must say so first. The bench items were conditional filler
and are now unreachable; that is fine.

**Ceiling to be honest about:** this can be built and unit-tested here, but its
real proof is a render. Renders HAVE RESUMED (2026-08-05), so the
`production-qualified` leg is runnable whenever a render window is free; until
it runs, claim only `code-complete + suite-green`.

### 2. THE NON-COMMERCIAL NOTICE REACHES NO HUMAN SURFACE (~30 min)

Fully scoped, ledger-clean, and the smallest real win on the board.
`nodes/OTR_LedgerScriptWriter.py:3590` stamps `meta["noncommercial_notice"]` (via
`_otr_provenance.noncommercial_notice`, `:124`) and logs it. **Nothing renders
it.** `nodes/otr_credits_roll.py:516` reads only `credits_source_line`.

Add a sibling printed-credits item beside that block -- `:516-518` is the exact
three-line shape to copy -- plus an integration test. The ledger field already
exists and already has an owner, so this adds a CONSUMER, not a field: no
ownership question to answer. Fires on Folger sources.

**Acceptance (r1 + r2 + r3 + r4):** `meta.noncommercial_notice` present -> ONE
rendered credits item; absent when empty, exact text, exactly ONCE. The
existing source item renders as `>> SOURCE: ...` (`otr_credits_roll.py:510-518`)
-- state the notice's literal prefix the same way, and the notice renders even
when a malformed legacy ledger lacks `credits_source_line`; ADJACENCY (source
line immediately followed by the notice, each its own `intercept` entry)
applies when both exist. No new wrapping helper: the existing intercept renderer already
measures and wraps every entry through `_wrap`
(`otr_credits_roll.py:1131-1135`). Test the ORDERED flow list (do not convert
`col3_flow` to a dict -- duplicate `"intercept"` keys collapse). Integration
fixture proves the Folger wording survives flow construction unchanged.
Legibility on canvas is eyeballed on the next permitted render, not claimed
from the test.

### 3. THE TEST-ORDERING POLLUTION (~30 min)

`tests/test_public_domain_sources.py` pollutes
`tests/test_public_domain_interpreter.py::test_empty_cast_is_rejected_and_retried_to_failure`.
Confirmed 2026-08-04: fails when the two run adjacently, passes **11/11** when
the interpreter file runs alone, invisible in full-suite order. Pre-existing --
already proven by stashing and reproducing at the prior commit.

Worth the half hour because it costs a real signal: any targeted run touching
those two files reports a red line that has to be re-diagnosed as benign every
time. **Build shape (r2 -- the r1 "cleanup fixture" idea was WRONG and is
withdrawn):** the mechanism is MODULE-IDENTITY breakage, not leaked state.
`test_module_import_is_lazy` (`tests/test_public_domain_sources.py:223-233`)
calls `importlib.reload(pd)` twice, which REPLACES the module's class objects,
while the interpreter test file imported exception classes at collection time
-- so `except OldClass` no longer matches instances raised by the reloaded
module. No cleanup fixture can restore class identity. Fix (r3-refined): run
the lazy-import assertion in a SUBPROCESS -- `sys.executable`, repo-root
`cwd`, `check=True`, fresh import with the read guard installed -- and pin
BOTH test-order permutations as regressions. The private-module-name
alternative is CUT: it risks exercising fallback import paths instead of the
production `nodes._otr_public_domain_sources` package identity.

### 4. THE SPOKEN-CITATION CODA STILL OWES ITS TESTABILITY (~2h, OPEN)

**The LEAK is closed and live-proven; the WORK below is not.** The old heading led with the closed half and a 2026-08-13 cleanup pass duly cut the whole section as done. Four coder items remain, and the two traps under them have each already cost real episodes.

The spoken-citation defect SHIPPED and is live-proven (`3943dd38`, `0957e169`,
`104c3f78`; receipt in `PROD_BUG_LOG.md` PBUG-20260805-04). Seven legs, six lanes,
zero leaked lines, and the corpus audit held at 69 findings across eight new
episodes. Licensed sources are now credit-only -- the announcer names neither the
licence nor the licensor, because Folger publishes the edition and Shakespeare
wrote the play.

What it still owes, all specified in `kibitz-runs/2026-08-05-item7-citation/r4/final.md`:

* **B4 -- extract the coda helper** so tests exercise the production reader.
  **TRAP:** do NOT extract `OTR_LedgerScriptWriter.py:5463-5588` verbatim.
  `news_meta` is defined inside that range and read by the caller below it --
  extracting as written is a `NameError` on every episode. Both review lanes
  caught this independently. Keep `news_meta` in the caller.
* **B6 -- bump `CURRENT_SCHEMA_VERSION`** (`nodes/_otr_ledger.py:58`). The audit
  must REQUIRE `spoken_coda_source` on post-fix ledgers while tolerating its
  absence on the 1,587 legacy ones; without a version boundary a dropped receipt
  is indistinguishable from history. `LEGACY_SCHEMA_VERSIONS` in
  `scripts/audit_spoken_citations.py` is already written to expect the bump.
* **Writer-level routing tests** (depend on B4): both fidelity banks x
  {non-empty, empty} provenance, plus an owned/non-empty case with
  `_style_grammar_on == False`. Assert the coda is PRESENT, not merely that the
  URL is absent. Control is `media_archive`, NEVER `scifi_news` -- that lane
  dispatches to `scifi_news_circuit` and returns before this block.
* **Bug Bible coverage** -- mandatory per `CLAUDE.md`, not a judgment call. The
  rule to promote: **a fix applied to a function with no callers is not a fix.**
  The 2026-08-04 attempt at this same defect edited `spoken_coda_line()`, which
  had zero readers, and 30 episodes leaked after it "landed".

Also parked and owed a merge: another session's worktree
`.claude/worktrees/awesome-brahmagupta-a509b4` holds the uncommitted deletion of
the dead `news_coda_spoken_reduction` receipt chain and `finalize_news_coda_surface`
(no callers tree-wide, no producer for its two trigger flags). It stood down so it
would not collide with B4. Re-ground it against the new helper boundary, then merge.

### 5. 1,090 CAST ROWS CLAIM A NON-COMMERCIAL MODEL IS COMMERCIALLY CLEAN

`eng_indextts2.py:55` says `commercial_clean = False` (bilibili non-commercial);
all 40 bank rows say `true`; `cast_lock.py` trusts the bank row. The row flag is
the CLIP's licence and the engine flag is the MODEL's -- genuinely different
facts, both already in the right layers. **Stamp the JOIN. Do NOT edit the 40
bank rows** (`otr_dl_indextts2_refs.py:11-17` documents them as clip provenance;
the ingest mints three rows across three engines from one PD clip).

**Must heal ATOMICALLY or it creates the defect it fixes:** the stamp
(`cast_lock.py:742`), the `gated` counter (`:575/:614/:661/:670`) AND the three
report strings (`:578/:618/:673`) -- otherwise the report prints `clean=True`
beside a ledger saying `False`. Resolve ONE profile by `(role, engine)` --
role-scoped, never engine-name-scoped. **Enforcement stays OFF.**
Prospective-only for the 1,090 frozen ledgers.

### 6. A TERMINAL FREEZE GATE THAT HAS NEVER READ A POPULATED FIELD

`find_scene_coherence_issues` reads `lines[].scene_id`; the `scifi_news` lane
writes `beats[].scene_id`. 55 ledgers assert the check, 0 carry the field, 55
pass. Nothing in `nodes/` writes `lines[].scene_id` on ANY lane -- the check
never had a producer.

Join per line: `beat_id` -> beat -> `scene_id` -> declared scene. Add a **VACUITY
refusal** (an armed gate that examined zero linkages FAILS -- that is how this
survived 55 episodes). **Split request from verdict:** keep a
configuration-derived `scene_coherence_required` and write
`{required, checked, verdict, issues}` into `report.info` -- `run_gap_audit` is
READ-ONLY (`_otr_ledger_freeze.py:664-698`), so the gate must not mutate the
ledger; the phase wrappers already persist the report. Measure OFFLINE over the
published corpus first, then arm in ONE change -- no intermediate flag-off ship.
Replace the stale hard-coded bank list at `tests/test_scene_guard_v4.py:89-99`
with registry-derived coverage (it omits `scifi_news`, the one bank that enables
the flag).

**The vacuity class is now proven twice** -- this gate, and the freeze test at
`test_g9_sfw_ship_stop.py` that filtered on a retired code prefix (fixed
`4506b1ed`). Any NEW armed gate ships with a vacuity assertion.

### 7. CHARACTER GENDER IS ROLLED ON PROSE LANES -- Scrooge shipped female (spec REWRITE owed; r2 and r3 both returned NO)

Live 2026-08-05: `EBENEZER SCROOGE` = female, `JACOB MARLEY` = other,
`HENRY HARTWICK OGLETHORPE` = female. Meanwhile MACBETH, BANQUO, PROSPERO and
MIRANDA are all correct.

**The split IS the diagnosis, and it means the render code is not broken.**
Shakespeare ships 14 provenance sidecars carrying `characters` with genders; the
prose lane has ONE tracked sidecar and its `characters` key is `None`. The pin
chain already exists and is lane-neutral (`_otr_roster_gender.py`, 12.6 KB, on
disk). Shakespeare is right because the DATA is there. Prose is rolled because it
is not. **This is a vendor-time data gap with a working consumer** -- the exact
inverse of the Item 7 bug, where the value existed and nothing read it.

Spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` (Fable, driver-grounded).
A four-tier TOTAL ladder -- roster -> pronouns in the source text ->
character-in-work web lookup -> name-frequency percentage -- stamping `gender`
(always populated, never `unknown`, because a voice must still be cast) plus
`gender_source` and a confidence. Operator rulings baked in: Shakespeare's KNOWN
rows are untouchable, the announcer stays randomly male/female by design, and the
invented lanes (`original`, `scifi_news`, `scifi_news_pro`, `media_archive`) keep
rolling -- their characters do not exist, so a name search there risks matching a
real person.

**Codex r2 verdict: NO. Eleven must-fixes, at `kibitz-runs/2026-08-05-gender-ladder/r2/codex.md`.**
The diagnosis survived; the mechanism did not. The three that matter:

1. **The web search would silently do nothing.** The spec passes a tools/plugins
   argument to `OpenRouterBackend.generate`, which swallows unknown kwargs through
   `**_ignored` -- no error, no search, a confident answer from a model that never
   looked. That is the same silent-no-op class as the defect above.
2. **"LLM extraction over the FULL unit text" cannot run.** `beckoning_fair_one.txt`
   is 143,176 bytes and 58 of 65 source files exceed 12,000 bytes, against a
   32,768 estimated-token per-call cap.
3. **Blanket surname aliases are identity-unsafe.** Two rows sharing a surname
   with the same gender currently produce a confident pin rather than an
   abstention.

**r3 RAN 2026-08-06 and it is STILL NO** -- Codex NO with 7 must-fixes, agy
yes-with-fixes with 9. That is three NOs across two rounds, and r3's findings invalidate
the CODING PLAN rather than its line numbers, so the standing re-ground rule applies:
**the next step is a SPEC REWRITE folding r2 + r3, then r3 again. Not r4.**

Both lanes independently found (a) a manifest sequencing deadlock -- the stamper is
specced to run per-unit inside the vendor fetch loop, but the manifest is written only
AFTER the loop, so it can never see the unit it was called for; and (b)
`RosterGenderVerdict` has no `gender_source` / `gender_confidence` fields, so the
ladder's whole output cannot be carried without changing every verdict-construction
path. Codex also confirmed the r2 finding that the OpenRouter backend still has no
web/plugin parameter, so the web-search tier would silently do nothing.

Judgment: `kibitz-runs/2026-08-06-2026-08-06-gender-ladder-r3/r3/judgment.md` (LOCAL
ONLY). **Trap: commit `496d9d57` inserted ~90 lines near the top of
`nodes/_otr_roster_gender.py`, so every cite into that file in the r3 review has
SHIFTED. Re-pin before acting.** The rewrite should also CONSUME the
`normalize_gender` boundary item 8 installed there rather than adding a second
normalization path.

**Found while grounding, and it reopens an operator ruling:** 32 of 85 Shakespeare
roster rows are `unknown` TODAY -- 38% of the lane assumed solved. Comedy of Errors
ships 7 characters, every one unknown. The narrower ruling that fits the evidence:
Shakespeare's KNOWN rows stay untouchable, but tiers 3-4 may fill only its
`unknown` rows. That fixes 32 rows without ever second-guessing a parsed
dramatis personae. **Operator decision, not a driver call.**

### Bench leftovers (relocated)

The old conditional bench block is gone (unreachable once item 1 grew, and two
of its three items already live in NEXT CODING QUEUE item 6). The remaining
one: **the three works that refuse to vendor** (`ghost_ship` gid 11045,
`purple_cloud` 11229, `beleaguered_city` 11521 --
`scripts/otr_vendor_public_domain_library.py:303/341/542` against the parser
at `:594-686`) **needs one Gutenberg fetch, so it is operator-opt-in only** --
not schedulable inside an offline sprint.

**Do NOT start the Shakespeare verbatim executor in this session.** It is a
multi-session structural change gated on the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`) with four overwrite paths to close
first, and starting it half-way is worse than not starting it.

## HISTORICAL CODING BACKLOG -- subordinate to the current runway at the top

This 2026-08-04 inventory may still contain real work, but it is not the active
order. Re-ground any item against HEAD only after the current top runway reaches
it. The 2026-08-11 review routing at the top governs; no full arc is implied.

1. **Style/identity campaign, items 1-4 (one campaign, ~1 day).** Highest
   leverage: fixes the credits style line for all six banks uniformly.
   Re-verified: `run_story_brief_reflection` is `_otr_story_brief.py:446`;
   `_build_left` is `video_engine.py:1442`; the treatment line renders as
   padded `Style    :` at `video_engine.py:1762` (an earlier "no longer greps"
   note here was a driver grep miss, corrected by the r2 panel -- BOTH
   consumers are live and both move in the repoint). Sharper finding on
   item 4: `ending_template` is NOT "computed and never read" -- the catalog
   computes it (`_otr_style_catalog.py:906`) AND the composer reads it
   (`_otr_line_composer.py:809-810`); what is missing is the THREAD between
   them -- no call site passes it into a LineRequest. That is a DECISION GATE
   inside the campaign, not a confirmed build step: wire the thread or rip the
   dead ends, decided with the panel at build. Same for the ghost-name fork:
   **default = scrub briefs after cast lock** (conservative -- no unlocked name
   reaches the listener); the operator may overrule to propagate pitch names.
   `style_seed_env` confirmed validator-only (`capability_profiles.py:116`).
   Item 5 (the 120-key `meta` rip) stays a gated block of its own, NOT part of
   this campaign.
2. **The P0 repair rung tells the truth (2 items, ~0.5 day).**
   `repair_literal_source_metadata` (`_otr_scifi_source_repair.py`, called from
   `_otr_scifi_codex.py`): (a) emit a receipt per pruned span/evidence-row/fact
   -- silent pruning violates the plan's own Invariant 3; (b) give the repairer
   the `allowed_source_fields` allowlist or prune per row, so one bad rehome
   stops poisoning the whole artifact. Suite-provable with fixtures. The HTML
   block-join separator stays an OPERATOR decision (coordinate-system change)
   and is NOT part of this chunk.
3. **Script-parse repair: fix the SPEC, then code it (~0.5 day docs + panel,
   then 1-2 days code).** The claim that increments 1-5 are code-ready is
   STALE: `docs/2026-08-03-script-parse-repair-CODE-READY.md` itself says r3
   returned seven must-fixes that invalidate the call/trace design and
   everything after its STATUS block is a draft. The next chunk is the spec
   correction folding those seven in (a kibitz arc IS the vehicle); only then do
   increments 1-5 become codeable.
4. **Passage-lane craft scoring + the stichomythia floor (~1 day).**
   `_otr_passage_selector.py` has no scoring functions -- the Fable craft
   criteria (French-scene boundaries, entrance/exit starts, couplet ends,
   continuation-word penalties) are all unbuilt. Score, keep top-K, seeded hash
   within the class. Same chunk: the per-beat word floor excludes stichomythia,
   so a merge rule or floor exemption in `_otr_episode_budget`. Pure Python +
   tests. **Sequencing (r1): runs AFTER the ON DECK sprint lands and re-grounds
   against whatever item 1 changed** -- both touch the fidelity selection
   surfaces.
5. **Cast-list parser: the two weak plays (~0.5 day).** Midsummer 1/12 and
   Comedy of Errors 1/7 gendered -- mechanicals/servants in shapes
   `_otr_character_roster.py` does not read yet. Vendored texts are on disk;
   offline; suite-provable against the sidecars.
6. **Small-items batch (~0.5 day, ONE campaign over the batch, one commit each):**
   * `OTRImageGenDispatcher` (`otr_image_gen_dispatcher.py:1412`) has no
     `IS_CHANGED` while depending on external file existence -- confirmed by
     grep, none in the file. Decide the CONTRACT before coding (r2): either
     fingerprint EVERY actual external dependency or deliberately force
     re-runs for this side-effectful node -- a partial path fingerprint still
     serves stale results.
   * Rotated server logs have no retention policy.
   * The `provider_side` three-part-rule regression (picked AND forced
     `cloud_kling_avatar`).
   * The shared `row_is_active(...)` evaluator over captured state -- confirmed
     absent from the tree -- closing the four env-read sites named in OPEN BUGS.

## PARKED -- D2 (renders have resumed; run when a render window is free for fail-hunting soak legs)

Reset per AGENTS.md section 4, boot headless, run **320-word `public_domain` or
`shakespeare` still legs until one fails** (~1 in 6). Three legs on 08-04 all
published, which at that rate is a ~58% chance of zero -- neither confirmed nor
cleared.

**Either outcome is valid.** A publish is a clean leg; a **fail-closed with
complete evidence is the PROOF D1 WORKS** and is the outcome you want. When it
fails the server log names the branch itself -- arm, token, index, canonical
`prompt_hash`, repr-escaped excerpt -- plus a compact JSON `MISSING_TARGET`
record emitted BEFORE the raise (the canonical runner truncates the exception at
500 chars, `scripts/otr_api.py:749`). The log survives reboot;
`scripts/_otr_rotate_log.ps1` rotates instead of truncating. D3 then fixes THAT
branch at its root and `PROD_BUG_LOG.md` gets a mechanism, not a guess.

**Do NOT:** weaken the completion gate, revive the portrait-init fallback, or
rebuild the withdrawn "give the collapse guard a still owner" fix -- the 08-04
postmortem disproved that chain (70 whiffs and 69 cast-time deferrals across 11
passes that ALL published).

Record: `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
`docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.

## AFTER THIS SPRINT -- the standing block order

One coder window at a time; every chunk = focused tests + full suite + Bug Bible
+ commit AND push + `HEAD == origin/v2.0-alpha`.

```text
  -> WAN 8-GB low-VRAM launch contract  (CODE-COMPLETE; blocked on ONE operator
                                         decision -- see OPEN BUGS)
  -> [r3+r4] Randomizer A
  -> [r3+r4] dynamic_story           (wiring only -- rev-5 DESIGN stays FINAL)
  -> re-observe the PARKED story bugs on the next real render legs;
     batch-triage whatever is left
  -> THEN, and only then, ROADMAP.md -- OFF THIS PLAN
       (its order after the SFX park: product expansion -> LEAN-MEAN ->
        RunPod -> release)
```

**LEAN-MEAN IS NOT IN THIS QUEUE and must not be re-added.** Operator direction
2026-07-29 moved FRONT and TAIL both to the Lean-mean campaign section of
`ROADMAP.md`, with their chunk chains, the W2 migration-first mandate, the
ENGINE_MATRIX W6 sub-step and the full `r2 -> r3 -> r4` pin carried over intact.
It runs after the randomizer and `dynamic_story` (the SFX step that used to sit
between them is PARKED). A window that wants to rip dead code is on the wrong
document.

**Block detail:**

1. **Randomizer Rolls Design A** -- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`.
   NOT gated any more: extensibility landed and absorbed its `_otr_lane_specs`
   authority, so this shrinks to `_otr_bank_roll` + eligibility. Its r3 brief must
   carry two deltas -- the absorbed authority, and that the bank list is a LIVE
   registry read (`list_bank_ids()` can return a CLIENT bank; eligibility must
   treat one as an ordinary peer) rather than a six-row literal. 1-2 d + 1 GPU day.
2. **`dynamic_story` visual direction** -- rev-5 FINAL; roster-agnostic; re-derive
   IDs at build. After the randomizer. The "do not rerun the design panels" rule
   and the r3+r4 requirement are NOT in conflict: the rule protects the DESIGN,
   r3 asks whether that design still wires to the code that exists today, and the
   roster, the routing authority and the writer tail have all moved since rev-5.
   5-9 coder-days + 2-4 GPU days.
3. **SFX campaign -- SUPERSEDED 2026-08-06. It is not parked any more; it is
   being RIPPED.** The 2026-08-04 park (operator doubt + an 8-15 coder-day lift)
   became a deletion when the operator ruled *"I do really want to rip out SFX
   100%, that's my aim."* **The live work is section 0-TER of this document**;
   this entry survives only so a reader who remembers "parked" finds out here
   that it expired.
   Nothing spends against a REVIVAL, and now nothing preserves one either: the
   Timeline Cue Ledger and generated-SFX designs are slated for retirement with
   the code. What the rip does NOT touch is the b-roll role tombstones (they
   still fail loud on stale ledgers) or the `[ENV|SFX|MUSIC:]` text sanitizers
   (defence against a model hallucinating a tag, not an SFX feature).

Open judgment question (render-window, not a coder slot): the LOCAL mistral/gemma
writer matrix. The Sonnet arm of the creative-writer question is answered
(`docs/2026-07-17-model-bakeoff-scoreboard.md`); the local roster comparison
never ran.

### STANDING RE-GROUND GATE -- r3/r4 before ANY block above (operator 2026-07-24)

Every remaining block was planned against a tree that no longer exists. Since
those docs were written the LLM vetoes were ripped, THE LAW landed, six banks were
renamed onto new packs, word-fit ceilings were retired, the whole extensibility
build shipped, and the suite grew past 8,000. Line cites, seam names and file
inventories are the first things to rot, and every one of these blocks is a rip or
a rewire that acts on exactly those.

- **Default entry point is `r3` (wiring).** These plans already have an r1 and an
  r2 on record, so the cheap re-ground is wiring against CURRENT code, then `r4`.
- **Drop to `r2` when r3 finds the CODING PLAN wrong**, not just the line numbers.
  Stale cites are an r3 fix. A seam that no longer exists, an authority that moved,
  a precondition another build already satisfied or destroyed -- that invalidates
  the coding plan itself, and patching an r2 from inside an r3 produces a plan
  nobody reviewed.
- **If in doubt, start at r2.** A wasted r2 costs one panel round; executing a
  stale coding plan costs a day of rips against the wrong file list.
- **No block executes without an r4 convergence at current HEAD.** Record the run
  under `kibitz-runs/<date>-<block>-r<N>/` and cite it in the block entry.

## SCOPE FOR v2.0 (operator -- read before picking up fidelity work)

**Two banks, not three.** `shakespeare` is VERBATIM and gets the executor.
`public_domain` stays PROSE and is explicitly allowed to be FUZZY -- operator: "the
LLM's job to try to do book prose but not perfect", and "I'm fine if it can pick up
real dialogue great, if not that's OK". Best-effort, never verbatim, never gated.

**But fuzzy WORDING is not the same as melding two stories, and that distinction is
the actual requirement.** Operator: "public domain does need to be updated so it
doesn't try to meld two different radio drama things and tries to keep true to the
source." That names the delivered defect exactly -- an H.G. Wells chapter performed
as taking place in "Arkham, Massachusetts", Lovecraft's fictional town, with the
time machine shrunk to a pocket watch. Two authors fused into one episode. The
contract for this bank:

* FREE with the wording. Invent the speech; carry the source's own quoted lines
  where they exist (the Wells chapter has real ones -- "Story!" cried the Editor).
* NOT free with the WORLD. Its place, people, period and events are the source's.
  No relocating, no importing a second work's setting or characters, no genre
  transposition.

Two changes serve that, and neither is a gate:
1. **PROMPT-ONLY** -- the pack asks for the source's world and its own quoted
   speech, and forbids importing anything the source does not contain.
2. **Stop the content-blind rolls contaminating it.** A catalog sound world ("a
   fire in the grate, a mantel clock, a teacup") was imposed on Wells' Richmond
   parlour, and `arc_shape` stamped "heist" on a man demonstrating a time machine.
   Foreign frames arriving from a dice roll are melding by another route.

**`public_domain_plays` is DEFERRED TO v2.1**
(`docs/2026-08-03-public-domain-plays-PLAN.md`, research complete, nothing built).
That avoids a third bank row, which is never one line: it would force a pack
directory, a registered fetcher, an executable pipeline, family-policy coverage and
updates to exact-roster contract tests.

**Two hard operator rules that override the repo's written ethos here:**
* **The word count is a REQUEST, not a gate.** No refusals, no hard gates, no
  shunts. Shipped: `select_passage` returns its closest performable passage rather
  than raising (`a4bc7917`).
* **No "dread py assertion workflow killers."** A render must not die. The
  reconciliation: fail loud in AUTHORING-TIME TOOLS (the fetcher, manifest
  validation -- things a human runs and reads), but in the RENDER PATH degrade to
  the best available result and write an honest machine-readable receipt into the
  ledger saying what degraded and why. The ledger tells the truth; the episode
  still ships. No hazard / under-construction flags -- `runnable` was checked and is
  NOT one; all six real banks are runnable and its only job is making the
  "+ Add Your Own" signpost fail loud on selection.

**An extra LLM VERIFIER pass is sanctioned** ("is this accurate to the story, are
these characters really in the scene?") -- but as a RE-SELECT, never an abort: if it
rejects a window, take the next-best candidate, bounded, then ship the best one with
a receipt. **Deferred until specced (r1):** it currently has no stated input, output
or artifact position. If built, it reviews the FINAL accepted script, stays
advisory/non-terminal under THE LAW, and records evidence-backed findings; until
that spec exists, do not quietly add another model pass.

**Casting must be smart about voices and gender.** The dramatis personae section is
the ROSTER and its descriptions carry gender; it is parsed at VENDOR time into the
provenance sidecar so the render path never infers (`d8752d69`). A manifest-approved
roster also replaces the refuted "speaker appears twice" heading rule -- that rule
would have deleted BEATRICE's single speech from `much_ado__act3_scene1`, the scene
named for her, where that one speech IS the payoff.

**Still open on casting:**
* **Midsummer 1/12 and Comedy of Errors 1/7 gendered** -- mechanicals and servants
  in cast-list shapes not yet read, recorded `unknown` rather than guessed. Operator
  is open to a vendor-time LLM/web lookup as the final tier for stragglers, under
  its own `gender_source` so it stays auditable. (Corpus elsewhere: AYL 6/6, Lear
  9/9, R&J 3/3, Much Ado 4/4, Tempest 3/3, Twelfth Night 5/6, Macbeth 7/8.)
* **Voice-pool capacity may belong in window ELIGIBILITY, not just casting.** The
  Bark pool is 6 male / 4 female against a 6-character ceiling, and Macbeth 1.3
  needs five distinct voices.
* **Disguise ruling (settled):** ROSALIND-as-Ganymede and VIOLA-as-Cesario keep
  FEMALE voices -- the source prefix says who speaks and the irony depends on the
  audience hearing her; the announcer states the disguise from a manifest field.

**Visual style: randomized is fine** (operator). The earlier objection to
`archival_documentary` over a Folger comedy was about TRUTHFULNESS, not variety --
the credits claimed a story scaffold the episode did not have. With the words
genuinely the play's own and the strip naming the real source, a randomly drawn look
is artistic range. (`visual_style_policy` was RIPPED on 2026-08-04 accordingly.)

## THE PASSAGE LANE (operator ruling; built, with craft criteria still unbuilt)

> "For shakespeare I'm open to a version that is very strict and finds, based on
> word count and random choice, hones in on a specific part of a play to get real
> specific dialogue, no paraphrasing."

A play episode is a contiguous WINDOW of consecutive speeches, carried verbatim,
chosen to fit the word budget, the cast ceiling and the beat topology. Built as
`nodes/_otr_passage_selector.py` (`a82460ec`), 24 tests, proven on all 14 vendored
scenes: every selected line is verbatim from its source file.

**The number that governs everything:** a passage is performed against VOICED
BEATS, and beats step with the ACT TOPOLOGY, not the word count --
`voiced_beat_count()` in `_otr_episode_budget` is the one owner. 30-120 target words
buy THREE beats, 150-200 six, 300-1200 fourteen. At 120 words a passage is a
two-or-three speech fragment; at 300 it is an eleven-to-thirteen speech exchange.
**The fidelity floor should be 300, not the operator's initial 120** -- three beats
cannot hold a change of mind, and every manifest already recommends 300. A long
speech spans consecutive beats in the same voice (`ceil(words/80)`,
`BEAT_WORD_HARD_MAX`); without that the lane silently loses Lear's love test,
Prospero's history and Juliet's balcony speeches.

**Craft criteria for selection, from the Fable review -- NOT yet implemented:** keep
windows inside one French scene (never cross an `[Enter ...]` that adds a speaker);
prefer starts on an entrance or a question, penalise openings on continuation words
(And/But/Nay/'Tis) and speeches under 4 words (Folger prints shared verse lines
separately, so those start mid-breath); prefer ends on an exit, a scene end or a
rhymed couplet, avoid ending on a question or a trailing dash. Score, keep the top
K, then apply the seeded hash within that class. Showcase example: Romeo and Juliet
2.2 lines 257-318, `[Enter Juliet above again.]` "Hist, Romeo, hist!" through
"Sleep dwell upon thine eyes... [He exits.]" -- 14 speeches, ~250 words, entrance
start, couplet-and-exit end, a complete arc that maps 1:1 onto the 14-beat topology.

**Prose is a different lane and the review was blunt about it.** Wells' chapter is
~70% narration; a characters-only performance discards the book's actual asset. The
faithful prose lane should be a NARRATOR/READER role speaking the author's own
sentences, abridged by CUTTING ONLY with every dropped span logged in provenance --
"abridged verbatim", which is also the period-correct radio form. Defer the
paraphrased variant until a dialogue-poor source genuinely needs it: as specced it is
indistinguishable to a listener from the existing original lane with borrowed names,
and it reopens the failure class just closed. If built, the announcer must say
"freely adapted from".

**Also flagged, unfixed:** the per-beat word FLOOR (20 at three acts) excludes
stichomythia -- "Nothing, my lord." / "Nothing?" / "Nothing." cannot be three beats --
so rapid exchanges need a merge rule or a floor exemption; `[aside]` and `[within]`
are machine-readable delivery hints worth carrying into per-beat metadata; and the
Wells manifest synopsis says the traveller "returns with a strange machine" when in
the real chapter he returns limping and the machine never appears.

## THE ADAPTATION DESIGN (hardened, NOT yet built)

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md`.

**The keystone correction: compile source speech, do not generate it.** A ledger row
that merely POINTS at a source segment proves structure, not meaning --
`PRODUCTION_SPRINT_LESSONS.md` lesson 11 documents that exact failure class.
Source-owned text must be materialized deterministically from an authenticated
segmented artifact and verified against it. "Summarize into X words" then means
SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not paraphrasing -- which also removes
the VRAM hazard, since no model sits in the source-speech path.

Settled by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats at
act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance is impossible
without redesigning beat topology. Build target is the 300-word unit.

**NEXT, IN ORDER:**

1. **The segmented source artifact** (schema, spans, hashes,
   `body[start:end] == segment.text`, omission receipts) and the pass-to-field
   ownership table -- **nothing else codes until that table exists.**
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which
   speakers appear must follow from the cut that fits the word budget. Coupled hard
   to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a
   4-person cast is a mathematically guaranteed `CastVoiceCoverageError` -- the
   failure that killed `scifi_news` in the six-bank run. `compute_episode_budget`
   must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067` hard-
   raises on any locked != requested) and change the pack text that tells the model
   to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and
   bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints`; still required by the validators and
   by `public_domain_manifest_schema.json`, so manifests and tests migrate in the
   same change. (`visual_style_policy`, the other half of this item, was ripped
   2026-08-04.)

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars
and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the
brief as ~880 words, silently. Belongs with the artifact work, where each beat is
fed its own segment rather than a blind prefix.

## STYLE / IDENTITY DECISION WORK (backlog; not the next coder window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:446` -- proven content-loyal on
   both specimens), stamp beside `story_brief`, repoint the treatment `Style:` line
   (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left`
   `:1466`) at it. Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas; the
   field is neither scifi nor a description). Consumers move in ONE atomic change:
   writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests -- AND the ledger validators (r3):
   `_otr_ledger_consistency.py` pins the field in its matrix
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
   Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray".
4. **Dead fields found**: `ending_template` computed but zero LineRequest call sites
   pass it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state`
   derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).

## OPEN OPERATOR QUESTIONS (flagged, awaiting a ruling)

* A research_only source now WITHHOLDS the OBS copy instead of killing
  the finished render (chunk 0.5 behaviour change, live since 08-15).
  If the operator wants the old kill-the-render behaviour back, it is a
  one-line revert -- say so.

* **Does `media_archive` want the catalog premise at all**, or the same
  scaffold-off treatment as `original`? Found by the five-bank beat test: a
  `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff
  seeded by a real Library of Congress item on 'Midnight' (1939) -- the operator
  caught it on screen. Second specimen of the content-blind-draw class. The
  scaffold-off rule so far was stated only for `original`.
* **Rename the un-namespaced `OTR_WAN_*` knobs?** `eng_wan_i2v`'s six frozen knobs
  are `OTR_WAN_STEPS` / `_CFG` / `_SHIFT` / `_SAMPLER` / `_SCHEDULER` / `_NEGATIVE`
  -- no `I2V` namespace, unlike every sibling. Default if unruled: leave them. The
  freeze already removed the power that made the missing namespace dangerous (they
  are consent-act-only now and cannot bind a production leg), and a rename would
  silently break operator muscle memory for a sweep.
* **`style_tail_policy`'s closed enum cannot express a SHIPPED path.**
  `VALID_STYLE_TAIL_POLICIES` has `full` and `minimal_clean`, but
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch
  (`otr_meta_brief_image_prompt.py:394-401`) RETURNS EARLY with
  `"%s, warm dramatic lighting"`, skipping both `finish_visual_prompt(...,
  era_profile="still")` and the `image_grade_tail` append -- deliberately, per the
  2026-07-02 operator look direction. The `ltx_audio_in` bookend row nonetheless
  declares `style_tail_policy="full"`. Adding an enum token is an operator call:
  either add a third token for "canonical warm, no era tail, no grade tail", or
  ratify that the `ltx_radio_face` path is EXEMPT from the plan's style-tail
  authority. Default if unruled: the exemption, because it changes no behaviour.
* **After profile retirement, who owns a tier's native render ceiling?** The full
  statement of this one is in the WAN 8-GB row under OPEN BUGS -- it is the single
  blocker on a code-complete block.
* **`check_compatibility`: ratify the inert constant, or schedule the rip?** See
  Open risks.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not. That
split is why the two eyeball-era entries at the end are PARKED rather than live.

**EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
2026-07-27 triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not. **Re-pin a row's cite when you touch it.**
Path note (verified 2026-08-04): engine adapters live under
`nodes/_otr_video_engines/` (and `_otr_audio_engines/`, `_otr_image_engines/`)
-- bare `eng_*.py` cites in these rows are shorthand for those paths.

### The 2026-08-11 bank-sweep trio (LEMMY sprint, all three OPEN)

Found by a six-bank live render sweep, not by tests. Full detail lives in
`docs/PROD_BUG_LOG.md` and `docs/2026-08-11-FINDING-lane-cast-contract-divergence.md`;
these rows exist so a window working THIS list actually sees them.

* **PBUG-20260811-03 -- `scifi_news` lost the LEMMY cameo it was built for.**
  ROOT CAUSE ESTABLISHED: `scifi_news` is a CONTENT-OWNED lane
  (`delivery_mode_for_meta(meta) == CONTENT_OWNED`, measured off the sweep's own
  ledger; `original` is `legacy`). Content-owned runners build their own cast and
  never run the writer's seeded picker, and `lock_cast()` is what applies the
  cameo -- so it cannot fire there. The empty `cast_contract` is the same
  deliberate decision: that block stamps `meta.episode_seed` and withholds
  `cast_seed`, because claiming one on a lane-owned cast detonated CastLock's
  replay before (`num_characters must be 1-6, got 0`).
  **THE OBVIOUS FIX IS THE WRONG ONE** -- routing content-owned lanes back
  through `lock_cast()` is precisely what that comment warns against. The repair
  belongs in the lane runner. **Operator row 15.**
  *Worst of the three by exposure:* nothing fails and nothing logs, so every
  `scifi_news` episode since the redesign has shipped with no cast contract.
* **PBUG-20260811-01 -- forcing the cameo kills the `scifi_fable2` writer on
  `scifi_news_pro`.** `pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; BAD_LINE`. Reproduced at 30 AND 90 target words, so NOT a word
  squeeze; with the cameo on its natural roll the writer passes cleanly. Root
  cause not established.
* **PBUG-20260811-02 -- `scifi_news_pro` dies at node 92 with no materialized
  still for beat `music_closing_001`** (`still-spine handoff missing materialized
  scene still ... engine still_flat`), on the same profile where five other banks
  produced one. Seen ONCE. Re-run before treating the cause as understood.

### The P0 / source-span cluster (2026-07-30)

- **`full_text` reaches the span coordinate system carrying HTML BLOCK JOINS WITH
  NO SEPARATOR, and on the live evidence this is the DOMINANT P0 failure cause.**
  Measured in the campaign logs: `'...Field of Martian PolygonsNASA/JPL-'`,
  `'...and the School ofEngine'`, `'...what you're doing.Let's s'`, `'...(AMR).The
  resea'`. The RSS adapter strips tags without inserting whitespace, so two elements
  fuse into one token. `_normalize_span_source_text` collapses whitespace RUNS but
  cannot insert a space that was never there, so the model quotes the sentences a
  reader sees and they are not byte-exact in the stored text -- exactly the
  "non-literal source span" rejection that killed 12 of the 15 P0 legs.
  **Deliberately NOT fixed by A-3:** inserting separators is a WIDER change to the
  coordinate system `source_digest` pins -- an operator decision, and it belongs in
  the source adapter rather than the codex normalizer. Owed: which adapter builds
  `full_text`, whether a separator can be inserted at admission without breaking any
  accepted ledger, and a fixture from these four strings.
- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.
  Under "fail loud, not fatal" the degrade is the right direction and the silence is
  not.
- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.
- **Nothing measures whether a pruned P0 index is ACCEPTED** (recorded, no action
  owed yet). No live leg has ever run with the deterministic rung reachable (it became
  reachable at `47c554fa`, after the campaign stopped), and the rejection logs carry
  only a truncated `raw head` plus no source payload, so the question cannot be
  answered offline. A-1's instrumentation is what makes the next campaign able to
  answer it.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after
  two attempts on non-literal fact source spans; provider/model convergence, extends
  BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the
  `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify
  still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

### The 8 GB / profile cluster

- **WAN 8-GB low-VRAM launch contract -- CODE-COMPLETE and PROOF-INCOMPLETE. It is
  not a coding item; the one thing blocking it is an OPERATOR DECISION.**

  **Already BUILT and WIRED end to end** (verified hop by hop): `otr_8gb_wan.json`
  `video.max_render_frames=17` -> `capability_profiles` optional-key validator ->
  `_otr_workflow_apply.py:532` flatten -> `workflows/variants/otr_8gb_wan.json`
  node-87 widget slot 14 = 17 -> `otr_video_director.py:423` policy stamp ->
  `otr_shot_lock.py:1722` `ledger.video.max_render_frames` ->
  `render_driver.py:3328` per-adapter policy -> `motion_common.profile_max_render_frames()`
  -> `eng_wan_ti2v._floor_length` hard cap (`:730`) and `_planned_length` refusal
  (`:785`), with `render_driver.py:3845` refusing on drift. Landed `f914f0a4`, dead
  node-87 widget repaired `7f4644a1` + `8f41af27`, WAN deliberately excluded from
  `frame_contract.PLANNING_CAP_ENGINES` by `b23fc035`, recipe frozen `71753cb4` /
  `8424f369`, whole-beat single-UNET-load `439ce8c7`. Regression net:
  `tests/test_remaining_video_contracts.py:16-194` (nine hop-by-hop tests) plus
  `tests/test_multiclip_effective_contract.py:216,234`.

  **THE ONE OPERATOR DECISION (the actual blocker).** The ceiling reaches a leg ONLY
  through a variant workflow or a hand-set widget: `otr_canonical.json` node 87 ships
  `max_render_frames=0`, so a plain canonical WAN run is UNPINNED and inherits
  `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure shape. The obvious patch
  (pin 17 in the canonical) is WRONG: the canonical serves every tier, and 17 is the
  8-GB tier's number, so pinning it would cap LTX/HuMo 16-GB legs too. The channel
  that carries 17 today is `config/profiles/*.json`, which is on the RETIREMENT list
  -- so writing new behaviour onto it is forbidden. **Decision needed: after profile
  retirement, who owns a tier's native render ceiling?** The shape that fits the
  per-adapter-ownership doctrine is that `eng_wan_ti2v` DECLARES its own tier ceiling
  (a capability-row field), the widget becomes an operator OVERRIDE with 0 meaning
  "use the adapter's own contract", and the profile channel stops mattering. That is
  a real design change with a live-behaviour blast radius on any card with headroom
  (the VRAM predictor currently gets to ask for more than 17 and often can), so it is
  NOT being written on assumption. Ratify the shape first.

  **Also open, all PROOF obligations rather than build work:** the 18-engine GPU
  campaign is engine COVERAGE, not an 8-GB qualification; and a render on a PHYSICAL
  8 GB card is still owed -- the four-arm bench PREQUALIFIED on a 16 GB card told to
  reserve 8 GiB, which is not the same claim.

  **One untested edge, cheap to close whenever this reopens:** WAN is out of
  `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict by
  design, and `_planned_length` hard-refuses mid-episode when they do -- but no test
  asserts a 17-frame tier survives a multi-segment beat. `:216`/`:234` in
  `test_multiclip_effective_contract.py` pin the topology, not that outcome.

- **THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER** (found 2026-07-27;
  LIVE-REPRODUCED TWICE on two different banks, then confirmed by a two-strikes
  kibitz panel -- codex `gpt-5.6-sol` high and agy independently reached the same
  diagnosis). `config/profiles/otr_8gb_ltx.json` pairs a 12B GGUF writer
  (`gemma-4-12b-it-Q4_K_M`, 6.63 GB of weights) with `llm.gguf_n_ctx: 2048` under a
  declared `vram_ceiling_gb: 6.8`. The pipeline's own smallest prompt needs **2064
  input tokens** and P0 reserves 2800 output (`_P0_BASE_OUTPUT_TOKENS`), so the leg
  dies in `OTR_LedgerScriptWriter` before any render. Live preflight, verbatim:
  `Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)`. **ctx is the SYMPTOM; the
  writer MODEL is the cause** -- 4096 puts it near 9.4-9.5 GB, OOM on the very card
  the tier exists for. Every 2048-ctx profile (`otr_8gb_ltx`, `otr_8gb_wan`,
  `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`, `otr_cloud_lanes`) is `status=draft` and
  every one pairs 2048 with the 12B; the only `status=shipping` profile is
  `16gb_full` (4096 + Mistral-Nemo). **NOT a one-line profile edit:** the GGUF
  registry ships exactly two rows (`unsloth/gemma-4-12b-it-GGUF`,
  `unsloth/Qwen3-8B-GGUF`); `google/gemma-2-2b-it` is in the TRANSFORMERS catalog, a
  different lane -- agy proposed it and was wrong, recorded so nobody re-derives it.
  **Largely mooted by profile retirement:** with no profile passed, the canonical
  JSON's own `gguf_n_ctx=4096` / Q8_0 binds and the leg runs. **Fix the profiles or
  finish retiring them; do not leave both.**

- **A2 -- HELD pending the profile retire-now vs retire-later scope. The profile's
  `llm` section silently overrides the canonical JSON, and the applied-overrides echo
  HIDES it.** Held because its entire subject is `apply_profile_to_workflow` and the
  printed echo -- a channel directed to be retired, so building on it now may be work
  on something scheduled for deletion. The fix SHAPE is correct and ready when the
  scope is settled. The profile's `llm.*` values win over the widgets the operator set
  in `otr_canonical.json` (which ships `creative`/`technical` = `google/gemma-4-12b-it`,
  `gguf_n_ctx=4096`, `gguf_quant=Q8_0`, `llm_vram_ceiling_gb=14.5`), while
  `scripts/otr_api.py:817` flattens only `role_overrides` / `slot_overrides` /
  `features` + two `seed_policy` keys for the printed summary -- so the run reports
  "16 overrides" while ALSO having replaced the entire LLM configuration. **Causal
  chain corrected** (triage 2026-07-27, codex; grounded): the override does NOT come
  from the validator's `OTR_ACTIVE_PROFILE` export -- it happens at submission,
  `scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`; and the real
  applier (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`; only the
  printed echo is stale. **Fix: generate the echo FROM the applier's flattened map.**
  Adding `llm` to the echo by hand leaves the next drift intact.

- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently). The coverage
  PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and
  `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own
  pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both land on 161 (profile
  unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json`
  ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a
  launcher honours it without also pinning the profile, the planner emits a 98-161
  frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted
  and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  **Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame
  beat partitions, which is a production planning decision, not a cleanup. The preset
  carries a `_ceiling_note` saying do not export it alone. Compare WAN, which B3
  wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES`
  and `video.max_render_frames`.

### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: `87 VideoDirector -> 88 ImageDirector ->
  89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`).
  `resolve_final_shot_engines` runs at node 92, but stills are minted at 91 and image
  PROMPTS at 89. The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").
  **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so
  hoisting to ShotLock still does not put MetaBrief downstream of the authority --
  that needs a VideoDirector-time freeze and is NOT in scope. (This is also the
  "image-phase still ownership" item from the campaign queue.)
- **THREE silent coverage mechanisms exist, not one** (found 2026-07-25).
  Mechanism 1, the engine mirror/ping-pong (`wrapper_bridge.extend_frames_to_target`),
  is GONE from `eng_ltx_8gb` -- pinned behaviourally by a test that detonates the
  helper and renders successfully. It REMAINS in `eng_wan_ti2v`, deliberately and
  permanently: WAN renders a short native clip on purpose and fills the beat with it,
  which is the shipped 8GB tier contract `PBUG-20260723-02` protects. **Still open:**
  composite loop-fill (`otr_silent_composite._should_loop_fill`, which also SUPPRESSES
  its own underrun warning once it activates) and held-last-frame. For `ltx_8gb` the
  composite path is now de facto unreachable -- the adapter returns exactly the
  requested count or raises -- but not structurally impossible:
  `encode_frames_to_silent_mp4` reports the size of the array it piped into ffmpeg
  rather than re-probing what ffmpeg wrote, so an encode-side drop could still
  under-report. PRE-EXISTING; close it when the assembly boundary is next opened.
- **`_should_loop_fill` names the permanent fix and it is now being built**
  (`otr_silent_composite.py:244-266`): *"The real fix is phrase-chunking (render the
  beat's correct duration so it never underruns) -- tracked as a follow-up."* The
  coverage block IS that follow-up.
- **THE 7d-PREFLIGHT THAT "PROVED THE GPU" RAN AT THE WRONG CANVAS** (found
  2026-07-27, B5 panel; verified -- and it corrects a claim this file once made).
  `render_single` and both HTTP entry points use the older ledger-free `build_request`,
  which never reaches the canvas seam and defaults to `OTR_VIDEO_RENDER_CANVAS`
  (832x480). So the "GPU IS PROVEN" leg (`ltx_8gb`, 25 frames, 3004 MB) exercised
  832x480, not the production canvas. `render_single` parity is explicitly deferred by
  the O1 judgment; what must NOT happen is another "proof" through that harness being
  read as a production proof. (The 512x288 canvas itself HAS since rendered live --
  bench arm D, three cells -- but through a DIRECT-NODE graph, which proves the canvas
  and the recipe, not the seam.)
- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty
  policy. B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters. Close
  it when the general canvas resolver lands.
- **Odd-canvas evenness is validated at the ENCODER, not where the canvas is chosen.**
  The stride defect itself is closed (`b1f2ee86`): `ffmpeg_silent_mp4_cmd` declares the
  REAL width/height and `encode_frames_to_silent_mp4` REFUSES an odd canvas by name,
  because yuv420p subsamples chroma 2x2 and cannot represent an odd dimension. Still
  true and NOT fixed: neither `WanInitImageMixin._dims()` nor the `Canvas` schema
  validates evenness, so an odd canvas is caught late rather than at the choice. No
  live producer builds one today (832x480, 512x288, 1472x832 are all even).
- **`CanonicalClip.frame_count` -- "the integer timing authority" -- HALF CLOSED**
  (`58e288af` + `40780b82`, count closed @ `48e3c6fb` without paying a decode). Every
  module that writes a clip now ffprobes it, and a roster gate in
  `tests/test_terminal_frame.py` fails by name for any module that writes a clip
  without proving it. What this proves and what it does not: it proves the muxer wrote
  what it was piped, which is the right question for a clip written by ONE ffmpeg
  pass; it does NOT prove decodability, which is why `assemble_beat_segments` still
  decode-counts every ASSEMBLED beat and must keep doing so. **Re-verify before
  acting:** this row's "still self-declared elsewhere" pointers (the four `viz_*` and
  four `still_*` engines) were both closed afterwards, so the remaining open surface
  may be empty.
- **KNOWN LIMIT of the widened roster gate**, recorded so it is not rediscovered as a
  surprise: the codec flag is matched as a STRING CONSTANT, so a flag assembled at
  runtime (an f-string, `"-c:%s" % stream`) or the stream-index spelling `-c:0` is
  invisible to the sweep. Nothing in the tree does that today; an encoder that ever
  needs to must be pinned in `_ENTRY_POINT_PROOFS` by hand, which the inventory test
  makes a visible decision. Separately, ONE mutant survives the round by construction:
  deleting the self-proving membership assertion is catchable only by a meta-test of
  that assertion.
- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed). It caps at
  `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497, env-overridable) and clamps
  at `:950-953`. It is NOT "renders to target natively" as three earlier docs claimed.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in code @
  `a1d810f1`, but the finding is STATIC (no live artifact), so it is NOT a PBUG row. A
  capped-14B leg would reproduce it. Kept here so the live proof is not forgotten.
- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only** (found 2026-07-27).
  Correct today and consistent with its own stated design (every number read from the
  live registry). But the moment a profile pins an `ltx_8gb` ceiling, the matrix keeps
  printing `9-161 step 8` for a tier whose real window is narrower, and the `--check`
  drift gate cannot notice because it diffs the registry, which the effective contract
  never touches. Owed at the prequalification step, not before.

### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze.** It calls
  `_recipe_config(self._recipe())` and `_recipe()` (`eng_ltx_av.py:402-432`) re-reads
  `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call by documented
  design ("Read fresh every call"). So a `required="when_engine_talking"` row evaluated
  through the hook re-reads the environment after capture. S0b-core needs ONE shared
  `row_is_active(...)` evaluator over captured state, with the talking result inside
  `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine`
  accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id prefix
  alone, so an `engine_facts` builder using a bare `getattr` would classify it local
  and let the radio-host redirect send a cloud avatar to local LTX. Needs a regression
  on picked AND forced `cloud_kling_avatar`.
- **Four env-read sites missing from the S0b inventory:** `eng_ltx_video.py:541-564`
  (`OTR_ENABLE_LTX_I2V`), `render_driver.py:1176-1203` and
  `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips. This is a DESIGN job -- a card
  laid out for a small canvas -- not more ladder heroics.

### Test-harness and tooling

- **The B7 forbidden sweep cannot see an UNTRACKED file, so a new test file passes the
  gate and fails one commit later.** `tests/test_b7_forbidden_sweep.py` builds its
  input from `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only.
  A new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns HEAD red
  with nothing else changed. Cost one red HEAD. **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen the gate
  to every untouched file in the repo. Cheap mitigation until then -- re-run the full
  suite once after the FIRST commit of any new test file.
- **NOTED, not a defect: two `scripts/` bake-off runners now abort a whole sweep on a
  count mismatch.** `scripts/run_ltx_av_q_bakeoff.py:453` and
  `scripts/run_humo_bakeoff.py:660` call the encoder inside per-leg loops with no
  try/except and DISCARD its return value (both set `result["frame_count"]` from
  `int(frames.shape[0])` independently). A disagreement that was previously invisible
  there is now fatal to the run. That is the correct direction -- a lying count is not
  a leg worth finishing -- but a sweep operator should know it before an overnight run.
- **LATENT, not reachable today: the fewest-segments partitioner can accept a
  disproportionate trim on a WIDE discrete menu.** WIRE-W1 makes `partition_beat` take
  the lowest segment count that covers, including via a permitted tail trim. On a
  ladder that is always the right trade; on a DISCRETE menu whose largest entry dwarfs
  its smallest it need not be -- covering 1019 frames from a `(10, 999)` menu, two
  segments give `[999, 999]` and discard 979 frames where three give `[999, 10, 10]`
  exactly. **A bound was written, MEASURED and REVERTED, and the measurement is the
  point:** rejecting a trim of a whole smallest-clip turned `[12, 12]` into
  `[12, 4, 4]` on a `min=4 max=12 quantum=8` ladder -- a third render and a third model
  load to recover four frames -- across 4,885 cases in the sweep grid. The widest
  shipped menus are Veo's `(100, 150, 200)` and Pixverse's `(125, 200)`, whose worst
  real trim is 25 frames. Revisit only if an adapter declares a menu with an extreme
  ratio; the reasoning is recorded in `coverage_plan.partition_beat` so the next reader
  does not re-derive the bound and re-ship the regression.

### PARKED -- unverified at HEAD, re-observe on the next real render legs
(The 2026-07-24 "after SFX" checkpoint is VOID -- SFX is parked. The re-observe
now rides whatever real render legs come next, D2 included.)

Both were eyeball observations against a story engine that has since had its LLM
vetoes ripped, THE LAW imposed, six banks renamed onto new packs, word-fit ceilings
retired, the repair-first plan landed, and a ledger cleanup pass added. Neither has a
reproduction at current HEAD, and under the standing rule a finding with no
reproduction is not a row. **Do NOT schedule coder time against either.** They are
settled by the operator eyeballing a real render leg after SFX: still there -> re-admit
as a FRESH dated row with that leg as evidence; gone -> the LAW-era work already fixed
it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`).
  Episodes START a story instead of admitting you into one; the announcer takes debate
  turns instead of framing. Operator eyeball 2026-07-11. If it survives re-observation
  the fix is still seam + score contract + fail-closed validator, never Python
  authorship.
- **Name-splice defect #2.** v4-campaign Phase 0 record in HANDOFF_LOG; its timebox
  predates THE LAW.

### Carried administrative rows

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until ratified
  at the next operator fan-out (green codex leg `c1f3891f` is the retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema `.v4`
  literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## KNOWN OPEN -- do not rediscover these

* The VRAM admission guard covers coverage-executed beats only; the single-clip path
  returns via `render_shot()` first, and `ltx_audio_in` is not in
  `PLANNING_CAP_ENGINES` -- so the hottest-peaking engine is unguarded.
* `FRAME_COST_MODEL` is keyed by engine NAME while recipe/quant/LoRA/reserve are
  env-configurable; a measured row needs a calibration IDENTITY.
* Four adapters still cite missing receipts (`ltx_audio_in`, `mesh_stage`, `viz_green`,
  `viz_mxc_mandala`).
* The HuMo lip-sync onset fix is SPECIFIED but unbuilt, blocked on M1 classification
  (`BUG_BIBLE.yaml:2343`: audio leads the lips by 100-200 ms with the face static for
  the first 3-6 frames). Pre-roll + equal trim is algebraically a NO-OP if the lag is
  constant rather than onset-only, so the classification must come first: early-only ->
  pre-roll fix; constant -> advance the 25 Hz conditioning features; growing -> a
  rate/timestamp bug, not a pad. Run a matched no-LoRA control -- Kijai reports the
  lightx2v distill is not fully HuMo-compatible, so the defect may be ours.
* Cap authority is not yet collapsed to one (`video.max_render_frames` should be sole;
  env twins must be absent-or-equal).
* `otr_w45_campaign.py` runs SIX engines while claiming all local ones, and its
  acceptance would not reject a mirror. Fix before trusting a campaign result.
* The reuse detector cannot separate a deliberately quiet shot from a duplicated frame;
  it is ADVISORY in `otr_w45_campaign.py` until that is solved. The engine-layer and
  composite guards (`MirrorExtensionForbidden`, `ClipUnderrunsItsBeat`) are terminal
  and unaffected.
* `humo_1.7B` and `ltx_8gb` are marked CUDA-only with no fp8, no fp4 and no stated
  reason. Unexamined, not proven.
* M2's raw rows sit in swept `tmp/` with no pinned digest or config manifest.
* `docs/2026-08-02-IDEA-hardware-compatibility-matrix.md` -- captured, not scoped.
  Includes the Mac research: Metal has no `Float8_e4m3fn`, ComfyUI+MPS video is
  impractical (82 min for a 2-second clip), Draw Things and MLX are ~100x faster and DO
  support LTX-2.3 with joint audio, and the `viz_*`/`still_*` lanes need no GPU at all.
* Writer scaffolding repair increments 1-5 -- the spec needs its r3 CORRECTION
  before any code (NEXT CODING QUEUE item 3; the "code-ready" title is stale);
  the reuse detector to the panel; section 0A carve-out ruling before M2 numbers
  move caps; Wan 2.2 I2V checkpoint download + `wan_i2v` re-run; the
  `OTR_CastLock` freeze cascade (`wan_ti2v`).

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on and why.
Pick the cheapest tool that can win; escalate only when the cheaper rung cannot decide.
Both pools reset weekly -- front-load heavy coder windows and big Codex spends early in
the credit week; late-week, drop to the $0 rungs instead of grinding a paid pool dry.

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|

Production (in-pipeline, all $0/local, offline-first): writers = Mistral-Nemo (ctx cap
16384) + `gemma-4-12b` (saved runtime-qualified local default); stills/video-init =
`z_image_turbo` (the Qwen-Image ENGINE is removed -- keep Qwen3/Qwen2.5 LLM support and
Z-Image's `CLIPLoader(type="qwen_image")` encoder, unrelated). Cloud writers stay
opt-in bake-off arms, never the default.

Per-window mapping: RENDER windows = local production models + the Codex-app monitor,
Claude/Codex only to launch, judge and wrap. CODER windows use the cheapest competent
local triage first; Codex CLI is reserved for a genuine quandary/third swing and
Sonnet 5 performs post-code QA. **Review routing is the dated REVIEW ROUTING block at
the TOP of this file (2026-08-15), which RESTORED the panel and supersedes the
2026-08-11 suspension this paragraph used to cite.** Read it there, never from here.

## THE LAW (operator, 2026-07-22 -- supersedes anything that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE, STYLE,
> VISUAL VOCABULARY, OR QUALITY.**

The sole terminal spoken-prose policy is the shared whole-word safety authority.
Structural JSON/schema/IDs/roster/source-proof/rights/graph/markup/nonempty/
provider-integrity failures remain fail-closed because they protect a usable ledger
rather than judge prose. Across all six banks, requested word length, actual word
count, drift, one-breath estimates, visual/world vocabulary, noun/POS heuristics,
casing/title/honorific style, craft, and quality are guidance or telemetry only -- they
may never reject, reroll, retire, replace, or block an episode. Same-story LLM cleanup
is allowed.

**SUPERSEDED IN PART (operator directive 2026-08-03, `CLAUDE.md`):** the
"whole-word safety authority" above is NO LONGER TERMINAL for episode content
-- no profanity or violence filtering on the generation path, and the source's
own language is carried as written on the adaptation lanes. The paragraph
above is kept as the 07-22 record; its STRUCTURAL half (schema / IDs / roster
/ source-proof / rights / graph / markup / nonempty / provider-integrity
fail-closed) still stands in full. The runtime filters that survived the 08-03
rip are inventoried and queued for removal as ON DECK item 5.

## Standing operator directives (hard)

* **The recipes are not on the table.** "We spent a lot of time perfecting the recipes
  to look good and we can't lose that." No VRAM, speed or cap finding justifies a recipe
  change; measurement runs the SHIPPED recipe unchanged. This specifically forbids
  reading "peak falls as frames rise" as a reason to raise the 97 trained-length cap,
  and makes the deferred no-LoRA HuMo control a recipe change rather than a control.
* **Per-segment rendering is BY DESIGN** -- "each audio clip takes its own journey, to
  keep VRAM low." Never classify an assembled beat as one render.
* **One coder window in the code at a time**, serialized through this file. Two windows
  editing the same file -- especially the workflow JSON -- is how it gets corrupted.
* The remaining hard rules (root-cause fixes, no content guardrails on generated
  episodes, no word-count chasing, the ledger-completeness rule for any ripped LLM
  pass, git policy) live in `CLAUDE.md` and are not duplicated here. Review routing is
  the dated REVIEW ROUTING block at the TOP of this file (2026-08-15), which restored
  the panel; the 2026-08-11 suspension named here previously is SUPERSEDED.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window and never
open one for a single small item. Every window starts by pasting its one-line kickoff --
the `otr-handoff` skill reads this file + git and states the current step. No manual
context handoff, ever. The active coder keeps GO_FORWARD + HANDOFF_LOG current in
the same green push that closes a row.

| Window | Scope | Rung | Gate | Size |
|---|---|---|---|---|
| **CODER** (the default) | Take THE CURRENT STEP from the top; ignore older `NEXT` labels below it. Use the `CLAUDE.md` design-choice test before code, then one clean independent finished-diff review; one green pushed chunk at a time | cheapest competent local triage; full local arc only for a real design fork | none | evidence-driven |
| RENDER | GPU legs only: acceptance legs for whatever the coder just landed, and the soak. Reset per CLAUDE.md section 4 before every leg; the soak restarts ONLY in its `--profile` form | local production + Codex-app monitor | a coder chunk needing live proof | GPU hours |
| PLANNER | The archive split this file still owes; Bug Bible operator fan-out; the `check_compatibility` fork | rungs 2-4 | parallel with any coder window | docs |

**THE STORY LAB ROW WAS REMOVED 2026-08-16.** The lab is RETIRED (see its
tombstone below) and this table still listed it as "next" with a kickoff prompt
telling the reader to resume an external repo and read a heading
("STORY LAB RECOVERY BASE") that does not exist in this file. A stale kickoff is
worse than none: it is the one line a fresh window pastes without checking.

**NEVER boot a window by letter.** Boot by THE QUEUE at the top of this file:

> resume the OTR build as a CODER window. Repo:
> `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
> `v2.0-alpha`, HEAD `<sha>` == origin. Read `CLAUDE.md`, then
> `docs/GO_FORWARD_PLAN.md` from THE CURRENT STEP and BASELINES, then the top
> entry of `docs/HANDOFF_LOG.md`. Work THE CURRENT STEP in order. Do not download
> any model without an explicit exact-name authorization. Use the `CLAUDE.md`
> design-choice test to decide whether an arc is owed; after code, one clean
> independent finished-diff review is enough. Test, publish to live `otr/obs`,
> then commit and push the green chunk together.

### If the window is a REMOTE / cloud Cowork session -- READ THIS FIRST

Learned the hard way 2026-07-26. A Cowork session running IN THE CLOUD is not the same
box as the repo, and two of CLAUDE.md's assumptions do not hold:

- **Read/Write/Edit hit the CONTAINER, not the Windows files.** In a remote window every
  read, edit and write goes through Desktop Commander against
  `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, and so does git,
  the venv python and pytest. Everything else in CLAUDE.md holds.
- **There is a LAGGING Linux snapshot at `/mnt/user-data/uploads/`.** Never read the repo
  through it, and say so explicitly to every subagent -- a prior session's agents
  reported phantom corruption from that mount.
- **The bridge can drop mid-edit.** If the remote-device tools vanish, STOP -- do not
  retry in a loop and do not leave a half-applied edit. Report what is on disk,
  uncommitted, and wait. Nothing was lost when this happened because the last green chunk
  was already pushed, which is the actual argument for pushing every chunk.
- **The 60s MCP call ceiling.** The full suite takes minutes, so launch it detached
  writing to a `tmp/` log + a `.done` marker, then poll. PowerShell `*>` redirection
  writes UTF-16, so read results with `Select-String`.

## Parallel lane -- no coder slot required

- **Bug Bible operator fan-out** -- 9+ closed candidates + the duplicate-legacy_id
  cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or stills) +
  nv50 re-soak -- the two open portability remainders; release QA validation time, not
  coding.
- **SFX: RETIRED and RIPPED (operator ruling 2026-08-06, "rip out SFX 100%";
  executed `9eb6ede1` per `docs/2026-08-06-BUILD-SPEC-rip-sfx.md`).** The five
  bed engines are deregistered and barred via `RETIRED_ENGINE_IDS`, the bed
  compiler and mux mix branch are deleted, and
  `tests/test_rip_sfx_bed_guard.py` trips on any surface creeping back.
  Reviving SFX is a NEW design against the post-rip tree; the old design docs
  in `ROADMAP.md` are the historical record only.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **Seedance softener mangles authored prompts (2026-08-17)** | **CANDIDATE, not admissible yet.** A blind regex pass over authored text produced "Dial slowly sweeps wildly" and inverted "vibrates aggressively" -> "vibrates subtly" on the DEFAULT pack's most energetic beat. Provable statically and now fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever runs and produces the artifact. Nearest existing coverage is `12.108`'s `self-veto-resolution` / `phrase-not-word-matching` tags, which do NOT cover blind-regex rewriting of authored text |

**PROMOTED 2026-08-18 (evening): Bible `12.114`, survival-guide `b9aada7e`, count
292 -> 293** (README bumped in all three places, coverage-index row added, Bible
regression re-run green 20/26/3). Source: **PBUG-20260817-08**, the Lemmy cameo
voice -- admissible because it surfaced on two live published episodes, and
promotable now only because the fix is verified. The entry carries TWO reusable
halves: *a reservation that exists as a convention in one subsystem is invisible
to another subsystem enumerating the same catalogue*, and the diagnostic trap
that cost more time than the bug -- ***a post-fix sighting is not proof the fix
failed; check process age first.*** Its verify section also pins *sweep the
SELECTOR, not the helper* and *count a corpus by ROLE before concluding anything
about pool concentration*.

**NOTHING ELSE WAS PROMOTED 2026-08-18, deliberately, and
the reasoning is worth keeping.** The evidence-guards work EXECUTED an existing entry's verify
steps rather than discovering a new class: `12.111` verify step 3 is what turned
up G1's partial guard, and `12.111`'s own `cause` section already describes that
failure verbatim -- *"Refusing only when a specific file (`MANIFEST.json`) exists
leaves every sibling artifact unprotected ... the separate `_KEY` directory"*. An
entry that predicts the defect you then find does not need a second entry.
The window's other finding -- `engine_impl_version` structurally unfillable
because no adapter defines `impl_version` -- is **static-audit only**, so the
admission rule bars it regardless of how real it is. And nothing had actually
rotted: all eleven cited artifacts re-hashed clean, so there is no live artifact
to admit. **Preventive work with no live failure produces no Bible row.**

**NOTHING WAS PROMOTED 2026-08-17 (item B window), deliberately.** The window's
findings are all static-audit -- the positive-prose ban, the seven-call-site
video negative, the B6 gate gap, the traceroute's coverage blind spot -- and the
admission rule reserves the Bible for defects verified by a live artifact.
`PBUG-20260817-01` was re-proved on pixels here but is ALREADY covered by Bible
`12.108`, whose tag list literally includes `prompt-audit-cannot-see-pixels`.
Bible stays **287**, README stays 287 in all three places; the Three-File
Contract is intact.

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval queue is
`docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented fixture creates a row.

## Validation and handoff law

- **Current whole-tree receipt: see BASELINES at the top of this file.** It is
  stated ONCE, there, and nowhere else. (A stale copy lived here until
  2026-08-16 claiming `9081/111/1` @ `2fc81f72` and calling itself "current" --
  nine hundred tests and nine days out of date. Two receipts in one file means
  one of them is lying; keep one.)
- **Standing acceptance receipt:**
  `python scripts/audit_voice_gender_consistency.py --root "C:\Users\jeffr\Documents\ComfyUI\output\otr"`
  -- expect exit 0 over 1,595 ledgers. Exit 2 means the scan did not FINISH and its
  verdict is not a pass.
- Every code chunk: focused tests, full Windows suite, Bug Bible, AST/JSON/BOM/zero-byte
  checks, commit, push, verify `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json` in the same
  commit and runs `OTR_WorkflowValidator`, JSON round-trip, strict link/input, live
  widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python. Every run loads
  the canonical workflow and writes directly to canonical episode/OBS paths. Asset
  existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only audits and
  documentation may run in parallel. HANDOFF_LOG + this file are the only tracking
  surfaces.
- **Count the suite on a clean tree:** `build_variants.py --all` also emits variants for
  any UNTRACKED profile on disk, and some profile checks are parametrized over the
  variants present, so another window's scratch profiles can inflate the number by a
  dozen tests that would not reproduce on a fresh clone.

## Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish). Treat the first
  live client-bank leg as a qualification, not a formality. Deferred power-user tiers
  (client own-runner + staging, dependency manifest, standalone story_rules) are
  explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3). The posture that must hold in
  every future change: `--activate` is the consent act; the seam fails LOUD
  (`UserBankExecutionError`) and never substitutes; client code never touches the
  canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the
  shipped fetcher/interpreter registries are never widened to admit a client id. Do not
  relax any of these for convenience.
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it. Any future change to the activation path (folder
  name, CLI verb, restart behaviour) must update `nodes/story_packs/banks.json`, that
  tooltip and `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired** -- operator/planner decision flagged,
  with a 2-of-2 recommendation to RIP (codex and Fable independently, Claude grounded
  both). The argument that decided it: the "it reserves the name" benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation provably
  ignores whatever a client puts under that name (`tests/test_otr_check_cli.py:335`
  asserts a bundle whose `check_compatibility` is a plain integer activates). The only
  artifact that reserves the name is the `EXTENDING_OTR.md` paragraph, which exists
  either way. Case AGAINST: churn on landed green code for zero behaviour change, and the
  plan of record already names the future consumer (randomizer eligibility). Blast radius
  if ripped: ~5 code sites, 2 test files, 3 docs; no workflow JSON, no routing, no
  source-payload consumer. **Not a coder chunk** -- either ratify the inert constant or
  schedule the rip as a planner chunk. Proposed doctrine line: a name published to
  clients before its consumer exists lives in the client-facing DOC as "reserved, no
  contract, ignored if defined" and nowhere in executable code, because code that names
  an interface is read as enforcing it.
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- No code lands mid-sweep of an active qualification campaign (the 420-rung
  uniform-code-confound lesson).
- `dynamic_story` touches the writer, the visual-style authority and the canonical
  workflow; it re-derives the live JSON at build. It is the only claimant on those
  surfaces.
- Generated-SFX R4 stays local/ignored evidence of a RETIRED campaign (the
  2026-08-06 rip, `9eb6ede1`); no R4.1 refit exists to run, and reviving SFX
  is a new design against the post-rip tree.
- Lean-mean front/tail drift: the constraint holds wherever it runs -- the tail's SW-1
  re-survey is mandatory against the then-current writer, and the two campaigns never
  share a window.

## Tombstones -- the only three a window might wrongly revive

Full list in `docs/HANDOFF_LOG.md` + `docs/PROD_BUG_LOG.md`. These three are
here because each has been re-proposed at least once:

* **The 20 fabricated-fixture `public_domain` episodes and the fixture itself** --
  operator ruling 2026-08-04: dropped and deleted, **never raise again**.
* **v4 improvement campaign banks #2-#5** -- PARKED, superseded by the keep-6
  rename + THE LAW. Revive only by operator decision
  (`docs/2026-07-17-v4-campaign/final.md`).
* **LEAN-MEAN** -- lives in `ROADMAP.md`, not this file. A window that wants to
  rip dead code is on the wrong document.

## Pointers

- `CLAUDE.md` -- hard operator rules; wins over this file wherever they disagree
- `ROADMAP.md` (later runway: product expansion -> lean-mean -> RunPod -> release; SFX RETIRED + ripped 2026-08-06)
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 11 pointer-not-proof; 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md` / `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`
- `docs/2026-08-03-fidelity-pass-ownership.md` (the ownership table the verbatim executor is gated on)
- `docs/2026-08-03-script-parse-repair-CODE-READY.md` (writer scaffolding repair increments 1-5)
- `docs/2026-08-03-public-domain-plays-PLAN.md` (v2.1, researched, nothing built)
- `docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md` (the isolated-bench carve-out)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` / `docs/EXTENDING_OTR.md`
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md` / `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-11-announcer-framing-defect.md` (PARKED) / `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `workflows/otr_canonical.json` (the workflow source of truth)


---

## PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected.** Operator: *"park it on go forward."* Raised
after the operator observed we should have far more voices than the writer is
being offered. He was right, by a wide margin.

### The finding, measured live

| pool | count | what it serves |
|------|-------|----------------|
| Bark `VOICE_PROFILES` (`config/cast_pools.py`) | **10** (6M/4F) | what the writer's casting menu offers |
| Kokoro presets on disk | 4 | ANNOUNCER only, a separate namespace |
| **Voice reference bank** (`config/voice_reference_bank.json`) | **204 declared, 153 resolvable on disk** (97M/106F/1N) | IndexTTS2 / Dia / Chatterbox cloning |

`_otr_voice_bank.default_char_engine()` returns **`indextts2`**, promoted to the
shipped character-voice default on 2026-06-04. It is a zero-shot CLONING engine:
`requires_voice_ref = True`, `voice_ref_kind = "wav_path"`. Every reference clip
is a distinct voice.

So the writer casts from **10 Bark presets** while the engine that actually
speaks the characters draws from a **153-voice reference bank**.
`_otr_casting.py` states it outright -- *"Open-character voices are always drawn
from the Bark pool (VOICE_PROFILES in config/cast_pools.py), so the tts_model is
Bark by construction"* -- and `_assert_unique_bark_voices` enforces uniqueness
across those 10.

### Why this is parked rather than done

`MAX_SPEAKING_CAST = 10` was set from the Bark pool and is therefore a Bark
artifact, not a real ceiling. But raising the constant alone achieves NOTHING:
`_deal_voice_menu` builds the menu from `VOICE_PROFILES` and refuses with
*"voice stock capacity 10 < cast size N"*. The actual work is pointing the
casting menu at the reference bank when the character engine is a cloning
engine.

**That is why it is parked and not done unilaterally.** It changes what
`voice_preset` and `tts_model` MEAN on every cast row, and cast rows are ledger
JOIN KEYS -- `cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` /
`voice_engine`, joined from `lines[].speaker` and `beats[].char_id`. Under the
operator's hard rule (*the writer must OBEY the ledger for downstream content; a
hole in the ledger is a broken render*), a change of that shape needs every
field's owner enumerated BEFORE the call is moved, not after.

### What the work is, when it is taken up

1. Enumerate every consumer of a cast row's voice fields -- casting, TTS
   dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`
   -- and name the new owner of each field. Exactly one owner each.
2. Make the casting menu engine-aware: Bark presets when the character engine is
   Bark, reference-bank entries when it is a cloning engine. Gender and
   `commercial_clean` already exist on bank rows.
3. Replace `_assert_unique_bark_voices` with an engine-agnostic
   one-voice-per-character invariant. The rule itself is right and must survive:
   two characters sharing a voice is a correctness defect.
4. Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a
   constant. `tests/test_cast_size_is_a_request.py` already asserts the constant
   matches the live stock, so it will report the drift rather than hide it.
5. Prove on `scifi_news_pro` (the only bank on the fable2 writer) with a cast
   larger than 10 and complete speaker-to-`char_id` equality in the ledger.

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.

---

## LEMMY RUNS BEFORE THE SEVEN RENDER PROOFS (operator ruling 2026-08-13)

This supersedes the 2026-08-12 "all sweeps first" gate. The operator wants the
proven story/source-bank bugs fixed first, then Lemmy given a full fighting
chance, then the seven remaining render legs run as post-change acceptance.

The one-coder-window law still serializes every code edit. Lemmy is runway row 2,
not a parallel window. Re-ground Phases 2-4 and PBUG-20260811-01/-02/-03 against
the then-current tree, preserve the qualification-receipt contract, wire any new
authority into `workflows/otr_canonical.json` atomically, and document each green
chunk in `docs/HANDOFF_LOG.md` while removing it from this forward plan.

The render runner must be recreated only after Lemmy is green. Its seven legs
prove the code that will ship; no stale pre-Lemmy runner or prompt is admissible.
