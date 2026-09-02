# OTR -- STANDING RULINGS, LAWS AND OPERATING CONTEXT

**Operator, 2026-08-23: "go forward should only have the go forward plans. Only."**

So `docs/GO_FORWARD_PLAN.md` is now a PLAN -- open work, in order, and nothing
else. Everything that is a RULE rather than a task lives here: the laws, the
standing operator rulings, the review routing, the model/credit ladder, how to
talk to the operator, the obs-path override, window packing, the tombstones and
the pointers.

**READ THIS FILE TOO.** It is not optional and it is not history -- these are the
constraints the next piece of work has to satisfy. The plan tells you WHAT to do;
this tells you what you may not do while doing it. `CLAUDE.md` remains the
highest authority and is unchanged; this file is the layer between it and the
queue.

Every section below is VERBATIM as it stood in the plan. Nothing was summarised,
reworded or dropped -- the 2026-08-16 audit's warning still governs: *"roughly a
third of these sections are standing operator rulings phrased as 'do not
re-open', and losing one costs more than the length does."*

Closed receipts are a third file, `docs/GO_FORWARD_ARCHIVE.md`, which is not read
to resume.

---

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
  rank-111 LoRA -- **NO LONGER GATED: downloaded on operator authorization
  2026-08-21 (2.74 GB, byte-verified), tested in lane 5, closed NO WIN** --
  alongside the still-frozen wan high-noise expert.
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

**Lane order (ALL THREE CLOSED NO WIN -- kept only so the contrast sets stay
quotable; do not re-run them):** (1) `wan_ti2v` sampling recipe -- contrast set exactly
{KSampler.sampler_name uni_pc-vs-euler, KSampler.steps 20-vs-30,
ModelSamplingSD3.shift 8-vs-5}, bundled as a SCREEN, decompose only on a win;
(2) `ltx25` stage-1 anchor 0.7-vs-1.0; (3) `ltx_video` LoRA strength
0.5-vs-0.7. Download-gated (operator authorization only): official 1.1
dynamic rank-111 LoRA **(DONE -- fetched and closed NO WIN in lane 5)**; wan
high-noise expert (still frozen, and ruled out by the operator's 16 GB rule:
14 GB of weights handing off to another 14 GB expert will not fit).

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

**ONE FINDING LANDED AFTER THE LAST HANDOFF WAS WRITTEN AND IS RECORDED ONLY IN
A WORKBENCH COMMIT MESSAGE UNTIL NOW** (`basline-models` `ff59d2f`, the current
workbench HEAD; the handoff log stops at `ae441eb`). Two parts:

* **`flux2_klein` minted the same prompted test card, and neither still engine
  wins outright.** Klein DRAWS better -- 6-7 colour bars against Z-Image's 4,
  crisp well-formed letterforms, very fine clean gratings, a round circle -- but
  OBEYS the structured spec less literally (one circle where three were asked,
  eye-chart rows scrambled). Confirmed and accepted by the operator: klein snaps
  to multiples of 16, so 1920x1080 becomes 1920x1072; `z_image_turbo` renders
  exactly 1920x1080.
* **IDEOGRAM 4 IS NOW CLOSED: NO. Downloaded on operator authorization, tested
  live, rejected before any code** (2026-08-21; verdict
  `docs/2026-08-21-ideogram4-verdict.md`, tracked receipt
  `docs/2026-08-21-ideogram4-probe-receipt.json`). The tested Macbeth card
  prompt repeatably produced a model-rendered "Image blocked by safety filter"
  card at two seeds, with no supported runtime disable control found -- and the
  refusal presents as a normal `SUCCESS` with a valid non-black PNG, so a
  generic pixel handoff would not catch it. The JSON prompt shape never blocked
  but invented text on 5 of 5 renders, including after OTR's own guard was
  carried verbatim as a trailing instruction. Spelling was excellent (6 of 6
  non-blocked frames correct). **This was a PRE-BUILD SCREEN, not a production
  qualification** -- standalone API graphs, nothing published to `otr/obs`, and
  the trigger was NOT isolated (line, backdrop and grade varied together).
  Reopening condition, and only this: prose on `announcer_visual` only, with a
  matched `z_image_turbo` comparison, the refusal detector (`min > 50` AND
  `std < 15`) armed, and the attention backend recorded (the probe ran pytorch
  attention, not SageAttention). Weights stay on disk; no adapter, no
  `CAPABILITIES` row, `otr_canonical.json` untouched. **The original entry
  below is kept because its FACTS are still true and it is why the lane ran:**

* **IDEOGRAM 4 IS OPEN-WEIGHTS AND FITS** (the pre-test assessment). `ideogram-ai/ideogram-4-fp8` shipped
  2026-05-30 and `Comfy-Org/Ideogram-4` packages it for ComfyUI. The nvfp4 set
  is diffusion 5.49 GB + Qwen3-VL-8B encoder 6.31 GB + flux2-vae 0.34 GB =
  **12.14 GB**, which clears the operator's 16 GB rule with headroom, in the
  format this Blackwell box already prefers (z_image ranks nvfp4 > fp8 > bf16),
  on a VAE `flux2_klein` already uses. **Why it matters: Ideogram is
  best-in-class at TEXT, and text is the one defect this programme actually
  surfaced** -- the lane 4 card rendered `TI2V` as `TIZV` in BOTH arms, and the
  1080p Z-Image card duplicated an eye-chart row at both seeds. **NOT
  downloaded, NOT authorized.** It is also not a knob: adopting a new still
  engine is a design change and would owe a full arc before code, not a Sonnet
  QA pass.

**LANE 6 IS CLOSED: NO WIN on its matrix -- BUT IT FOUND FLICKER, AND THAT NEEDS
THE OPERATOR** (2026-08-21, workbench `f71b24e`, verdict
`verdicts/lane6_wan_tiled_decode.md`). `wan_ti2v` tiled-vs-untiled VAE decode,
8/8 legs, purity gate clean, shipped recipe untouched.
* **No tile seam.** The classic tiling artifact is absent -- highest lattice
  concentration 1.093 against a declared 1.15 threshold, arms 98% identical
  spatially. That was the pre-declared win condition, so on the matrix this is
  a NO WIN and `VAEDecodeTiled` stands.
* **THE UNANTICIPATED FINDING: the TILED arm churns 4.3x and 4.9x more at the
  median on the test card, at both seeds** (2.83 vs 0.66; 2.83 vs 0.58), on a
  fixture whose prompt demands a rigid static card -- so that change is
  flicker. p95 and max are close between arms, so it is the BASELINE that
  differs, not a few events. On real crowd content the gap collapses to
  1.2-1.4x. Frozen-clip ruled out: both arms travel the same frame-1-to-97
  distance within 6%, i.e. same trajectory, smoother path.
* **NOT cashed as a win, deliberately.** Promoting a temporal result after
  declaring a seam matrix would be the goalpost move lane 5 refused. It earns a
  follow-up lane with a temporal matrix declared up front.
* **What is still missing is the whole cost side:** decode time was confounded
  by ComfyUI caching the shared latent (the `ours` leg paid for sampling, so the
  180s-vs-18s split is NOT "untiled is 9x faster"), and no per-arm VRAM peak was
  measured. Tiling exists for VRAM; nobody has priced removing it.
* **This is a QUALITY finding, which is the one class the "recipes are not on
  the table" directive does not cover** -- that directive exempts VRAM and speed
  findings. So it is the operator's call, and he needs the cost half first.

**LANE 7 IS STAGED AND RENDERING** (workbench `8922eba`): the `ltx25` MOTION
fixture lane 2 said it owed, same anchor contrast (1.0 vs 0.7, both leaves),
two i2v fixtures that demand traversal and head rotation, purity gate clean.
**A motion gate is declared before rendering and fails closed on the lane's own
premise** -- lane 4's `testcard_motion` produced 0.0 px translation in both arms,
so if neither arm moves here the cell is NOT judged and the verdict is "the
fixture failed", not "soft ties on motion".

**THE PROGRAMME'S NEXT STEP IS A DECISION, NOT A RENDER.** Per the standing ROI
ruling ("if no rendered candidate wins materially, stop the program and retain
the corpus"), the nulls are that condition. Remaining candidate: the
full-precision wan UNET (~10 GB, coin-flip on fit, the LARGEST untested 16 GB
compromise). **The operator's 16 GB rule governs downloads: "if the model can
run under 16gb that's fine we download."**

**A BOUND THAT MUST BE WRITTEN INTO ANY CLOSING STATEMENT:** five lanes tested
KNOBS. The single biggest 16 GB COMPROMISE -- the Q5_K_M UNET -- was never
challenged, because the official full-precision counterpart is not on disk.
Five nulls must NOT be read as "the compromises are validated."

**Bible now 302 entries (`8b194d3`), 22/26/3.** `12.123` promoted 2026-08-21 evening: a harness that hardcodes one graph builder measures the path production does not use (lane 3 screened text-only for twelve live legs while `OTR_ENABLE_LTX_I2V` defaults on). `12.122` and `12.121` promoted this
morning: an uncontrolled second variable voids every arm of a visual A/B (the
grid rode in on the stills; the drift probe obeyed its prompt). Another window
independently promoted `12.120` overnight; both stand.

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

**BASELINES (re-measured 2026-08-23 OVERNIGHT across the lean-mean campaign;
supersedes the 2026-08-21 block):** suite **12141 passed / 134 skipped /
1 xfailed** -- MEASURED by a full run (336.98 s, EXIT=0, known-fail guard
silent) at the order-6 head, never derived. Bible **22 / 26 / 3** (re-run at
wrap-up, Bible repo synced to origin/main). `build_variants.py --check`
**54 variants / 2 failures -- and BOTH failures are the DEFERRED
`otr_ghost_signal_v3` drift** (`docs/2026-08-22-variant-drift-DEFERRED.md`);
any third failure is NEW. Canonical validator **23 nodes / 57 links**; the
saved canonical Git blob remains `c27dff3690030e78d88c3a2607a9ac54fd3935d9`,
byte-identical through the whole campaign. **The PACK ROSTER baseline moved:
25 registered nodes** (34 -> 30 at order 5 -> 25 at order 6; every retired id
is a NAMED tombstone in DELETED_NODE_TYPES). Entry `12.120` /
`PBUG-20260820-01` still requires model-specific approved evidence before any
generic reference capability can ship.

**THE PREVIOUS RECEIPT SAID "ZERO REGRESSIONS" AND THAT WAS NOT TRUE.** At HEAD
`55ddf234` the suite exited **2**, with
`tests/test_legacy_audit_clean.py::test_no_unclassified_legacy_references`
FAILING. Item I's new `nodes/_otr_name_authority.py:90` used *"Professor & Lab
Director"* as an example job title in a comment, and the standing legacy audit
forbids a bare `\bDirector\b` in `*.py`/`*.json` outside a forensic context --
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

**THE MODEL & CREDIT BUDGET LADDER WAS EMPTY; IT IS NOW RESTORED (2026-08-21).**
The table at the "MODEL & CREDIT BUDGET" heading below had a header row and a
separator and no rows under them, so the rungs every window is asked to cite did
not exist in this document and windows answered from the prose paragraph beneath
it instead. Seven rungs are now recovered from `ed8d5a6d` and refreshed to
current fact; see the note directly under that table for exactly what changed.

## A BLOCKING DEPENDENCY IS A CLAIM, NOT A VERDICT -- CHECK UP A LEVEL (operator ruling 2026-08-28)

**Operator, in his words:** *"it should check up a level -- if it says we
can't remove because of dependency B, well, check what dependency B really
does."*

**THE RULE, AND IT IS THREE LAYERS DEEP** (extended by the operator the same
day: *"even if it says A has dependency B, check dependency B -- and for good
measure check dependency C too. Three layers. Does it actually do
anything?"*). When an audit, a reviewer or your own grep says "X cannot be
removed because B uses it", that is the START of the investigation, not the
end.

    A looks alive because B references it.
    B looks alive because C references it.
    If C is dead, the whole chain is dead -- and at ONE hop it looks
    perfectly referenced from the inside.

So walk it until you reach something genuinely live: a registered node in
`NODE_CLASS_MAPPINGS`, a string named in `workflows/*.json`, a CLI entry
point, or a test asserting PRODUCTION behaviour rather than exercising the
orphan. Report the chain in the form
`symbol -> caller -> its caller -> reached: <what makes this live>`, and if
you cannot walk it to something live within three hops, say so plainly rather
than assuming a fourth hop would rescue it.

**AND ASK THE THIRD QUESTION: does the chain actually DO anything?** A
function that is called, computes a value, and whose return nobody reads is
dead in the way that matters -- and it survives every reachability check ever
written, because it genuinely has a caller. That shape is the most valuable
thing this rule finds.

Go read B. Two things are true often enough to be worth the look every time:

* **B may itself be dead.** A chain of dead code keeps every link alive under
  a naive reachability check -- three orphans that only call each other look
  perfectly referenced. The only way out is to ask whether B has a live
  caller, and then whether *its* caller does.
* **B may not really depend on X.** It may import X and never call it, call
  it only in a branch nothing reaches, or duplicate what X does and use its
  own copy. An import is not a use.

**Two live examples from the 2026-08-28 sweep, both of which the rule catches:**

1. `_otr_voice_resolver` looked alive because `tests/test_voice_backends.py`
   imported it. Reading that consumer showed the import existed ONLY to assert
   that the live registry's `KNOWN_ENGINES` matched the resolver's DUPLICATE
   copy of the same set. The dependency was a consistency check on a
   redundancy -- deleting the redundancy is what removed the need for the
   check, and both went together.
2. The fuzzy cast-consolidation cluster in `story_orchestrator.py` "has
   callers" only in the sense that its four functions call EACH OTHER, and its
   test regex-extracts them from source rather than importing them. Nothing
   outside the cluster reaches it. A reachability check that stopped at the
   first caller would have reported it live.

**The honest limit, and it is why the rule says "check", not "delete".** A
symbol reached only by tests is not automatically dead -- it may be a parked
capability, a safety tool, or an unwired FIX somebody wrote and never
connected (which is a different finding entirely, and more valuable than a
deletion). Checking up a level tells you WHICH of those it is. It does not
license removing something because its only consumer is a test.

## ONE MACHINE OWNS THE WORKFLOW AT A TIME -- THE WORKFLOW BATON (operator ruling 2026-08-28 -- hard)

**Operator, in his words:** *"I want only one machine to own the workflow at
any time. So once we test the 4060-slim version of the workflow on the 5080,
then we load it up on the 4060, and that becomes our system of truth for
updates. Then when we're done, we move it back to the 5080 and retest there."*

**THE RULE.** Exactly one machine is the OWNER of the workflow JSON at any
moment. Only the owner edits it. The other machine may RUN what it has, but
may not change it -- an edit made on a non-owner is either lost or becomes a
divergence, and a diverged workflow JSON is the single most expensive file in
this repo to reconcile.

This is the machine-level form of the rule that already governs windows: *one
coder window in the code at a time* (`CLAUDE.md`). Same failure, wider blast
radius -- two boxes editing the graph is how `widgets_values` drift, phantom
diffs, and silently-unwired nodes get in.

**THE BATON IS GIT, and the handoff has three steps that are not optional:**

1. The owner **commits and pushes** every workflow change before giving up
   ownership. An unpushed edit is not a handoff; it is lost work waiting to
   happen.
2. The new owner **pulls and verifies HEAD == origin** before its first edit.
3. The old owner **stops editing** -- it may render, it may read, it does not
   write. If it needs a change, it asks the owner or waits for the baton.

**THE ROUTE FOR THE 4060 CAMPAIGN, in the operator's order:**

| step | owner | what happens |
|---|---|---|
| 1 | 5080 | the 4060-slim workflow is VERIFIED here first -- a fast box makes a mistake cost minutes, not hours |
| 2 | 4060 | ownership MOVES. The 4060 is the system of truth while its own fixes are made |
| 3 | 5080 | ownership RETURNS, and the workflow is RETESTED here before it is considered done |

**Step 3 is not a formality.** A workflow tuned on the 4060 has been proven on
one box only; the retest on the 5080 is what proves the changes did not
quietly depend on the small-card configuration. Do not skip it because the
4060 leg was green.

**A note for whoever holds the baton:** the canonical
(`workflows/otr_canonical.json`) remains the source of truth for the GRAPH --
its nodes, links and wiring. Per-machine model PICKS live in
`workflows/variants/` (ruling 2026-08-25). The baton governs who may write to
either of them; it does not merge the two files into one.

## AN ALL-REFUSED EPISODE STILL PUBLISHES (operator ruling 2026-08-27 -- hard)

**THE QUESTION, and it was asked because a panel refused to answer it for him.**
The r1 judgment at `kibitz-runs/2026-08-25-model-refusal-required-still/r1/`
found that node 92's success check is `clip_count > 0`
(`otr_video_render_batch.py:556-563`), so an episode in which EVERY required
still was sanctioned-gapped would report FAILURE even though every individual
gap had been handled exactly as the 2026-08-22 ruling requires. The panel wrote
down its default-if-unruled and then explicitly declined to bake it in: *"This
needs an explicit operator call, not a silently-baked-in assumption."*

**THE RULING: it PUBLISHES.** Asked directly on 2026-08-27, the operator chose
publish-anyway over fail-loud and over publish-but-mark-degraded. The episode
reaches `otr/obs/` with floor segments where the stills would have been.

**WHY IT IS CONSISTENT, so no window re-litigates it.** A sanctioned gap that
must not kill the episode does not become permitted to kill it by arriving in
quantity -- otherwise the 2026-08-22 ruling holds only until it is actually
needed. It also follows the obs law directly: *"if I don't see it in obs and it
took more than 5 minutes, it's a fail."* A leg that dies at the still-spine gate
ten minutes into an eleven-minute run has already spent the script, the voices
and the audio mux, and delivers nothing to look at. An episode of audio over
floor frames is a worse-looking episode; a dead leg is not an episode.

**WHAT THIS DOES NOT LICENSE.** It is not permission to hide the refusals. The
ledger-completeness law is untouched: `required_scene_targets` still owes
exactly one row per required target, `ok` or `sanctioned_gap` with its refusal
evidence, and the reporting paths that currently launder a gap into an ordinary
delivered receipt (`otr_video_render_batch.py:146-150`,
`otr_silent_composite.py:750-770`) still have to read the status explicitly.
Publishing an all-refused episode while REPORTING it as a clean render would be
the defect this ruling exists to avoid, wearing the ruling as cover.

**r2 IS UNBLOCKED.** The coding plan may now assume publish-on-all-refused.

## ONLY EASY-TO-LOAD LLMs SHIP (operator ruling 2026-08-25 -- hard)

Operator, in his words: *"if it doesn't fit nicely or requires Ollama rip it
from the dropdown and blast radius"*, and *"clean sweep I only want easy to
load LLMs"*. Also, on where weights live: *"all models should live out here
`C:\ComfyUI-Models`"*.

**THE RULE:** a curated LLM row must be `vram_fit_tier="PASS"` and its weights
must be materialized on disk. A row that needs offload, is "for bigger rigs",
or is not soak-tested does not go in front of an operator. A dropdown entry is
a PROMISE that the model will load; a WARN row cannot keep it on the 16 GB
target card.

**EXECUTED 2026-08-25.** `Qwen/Qwen2.5-14B-Instruct` was ripped from the
catalog and its blast radius (context override, two probe model lists, three
test files, the license audit target + its audit file). It was the LAST
WARN-tier row AND the only curated row with no weights on disk -- a dropdown
entry that could not have loaded if anyone had picked it, failing silently
until a render died. Its own note conceded the case: 28 GB of safetensors
*"needs quantization or offload to fit 16 GB -- not soak-tested as PASS yet.
Available for users with bigger rigs."* Three dead context-override keys for
already-pruned models went with it. Precedent: the same operation on 2026-05-23
pruned two community WARN-tier 12B rows.

**THE OLLAMA HALF HAD ZERO TARGETS, and that is worth recording so nobody
re-hunts it.** Nothing in OTR uses Ollama. The GGUF lane is IN-PROCESS
llama-cpp-python -- `nodes/_otr_gguf_backend.py` states it opens no port and
does not call Ollama. There is no daemon, no sidecar, no HTTP hop to remove.

**DO NOT READ A >16 GB DISK SIZE AS "DOES NOT FIT".** `approx_safetensors_gb`
is the DOWNLOAD size, not the VRAM resident size -- the field's own comment
says so. Mistral-Nemo (24 GB) and gemma-4-12b (23.9 GB) are the SHIPPED
production writers and load fine under the canonical's `bnb_nf4` 4-bit path.
A naive "remove everything over 16 GB" sweep would delete the writers this
project runs on. Judge by `vram_fit_tier`, never by the size field.

**THE GATE THAT KEEPS THIS FROM DECAYING:**
`tests/test_loader_slot_primitives.py::test_every_curated_local_row_is_pass_tier`
fails by name on any new non-PASS row. The WARN mechanism itself is still
covered, by SYNTHETIC rows -- removing the last real WARN model must not
silently delete the only test of what WARN does, which is the vacuity class
this repo has been bitten by twice.

**The preflight contract for adding your own model is
`docs/LLM_PREFLIGHT_GUIDE.md`** (seven gates; gate 5, constrained JSON, is the
one that actually fails). Operator: *"there should be an LLM preflight guide --
preflight guides for adding all your own components."*

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

### DO NOT "FIX" `_otr_story_brief.py:354` (operator, 2026-08-24, hard)

**Operator, in capitals and unprompted: *"NO DONT TOUCH MUSIC ROWS"*.**

`nodes/_otr_story_brief.py:354` filters `speaker_role in {"music", "env"}`
inside `_build_reflection_input()`, which assembles the text handed to the
reflection pass. `VALID_SPEAKER_ROLES` (`_otr_speaker_role.py`) is
`character / announcer / music_open / music_close / music_inter`, so **`"music"`
matches none of them** and the prompt's "NON-DIALOGUE ROWS" section has ALWAYS
been empty.

**THE OUTCOME IS CORRECT. ONLY THE MECHANISM IS ACCIDENTAL.** It was surfaced
by the Sonnet lane of the 2026-08-24 SFX sweep as a dead-code defect and was
briefly written into this file as an open item. That framing was WRONG and it
is corrected here, because it is the dangerous kind of wrong: a future window
reading "dead filter, real defect" would repair the filter, music rows would
start reaching the reflection prompt, and story output would change -- which is
exactly what the operator has now forbidden.

**SO THE RULE IS: leave it alone.** Do not repair the filter, do not "tidy" the
dead branch, and do not add music rows to any reflection or brief prompt by
another route. If a future audit flags `{"music", "env"}` as stale vocabulary
again -- and it will, because it genuinely looks like one -- this paragraph is
the answer.

**What was deliberately NOT done, and why that is also correct:** the dead
branch was not deleted either. Removing it would preserve behaviour, but the
code is the only place the reader meets this question, and a comment pointing
here is worth more than a tidy diff. Nothing in the pipeline reads it.

### THE LEDGER MUST BE GOOD. A LEAK IS TOLERABLE; A REGEX CLEANING IT IS NOT (operator, 2026-08-24 -- hard)

Operator, verbatim: *"THE LEDGER MUST BE GOOD, I'LL LIVE WITH FEW LEAKS INTO
DIALOGUE IF AFTER THE PASSES IT STILL DOESN'T FIND IT MY THING. IF IT FINDS IT
AN LLM DOES A A/B BEFORE AND AFTER BEAT-AWARE APPROACH TO CLEANLY CLEAN UP THE
LEDGER DIALOGUE, NOT HACK IT WITH PY[THON]."*

**THE PRIORITY ORDER, and it settles a whole class of argument:**

1. **The LEDGER being correct is the thing that matters.** It is what every
   downstream consumer reads -- TTS, per-beat slicing, video/shot direction,
   captions, credits, `obs_publish`.
2. **A few cue leaks into spoken dialogue are ACCEPTABLE.** They do not fail an
   episode and they are not worth a gate. If the repair ladder's passes never
   notice one, that is fine and explicitly the operator's problem, not a defect
   to chase.
3. **BUT IF A LEAK IS DETECTED, THE CLEANUP IS AN LLM'S JOB.** It must be
   **beat-aware**: the pass reads the beat BEFORE and the beat AFTER the
   offending line and repairs it in context, so the result is a line that still
   plays. **A Python regex may DETECT. It may never REPAIR.**

**WHY, because the reasoning is the reusable part.** A regex cannot tell a
sound cue from a spoken line that happens to carry brackets, and its only move
is deletion -- so it silently removes words rather than repairing a row. That is
the same failure the 2026-08-05 `clean_spoken_text` ruling guards from the other
direction, and it is why the obvious fix was REFUSED on 2026-08-24: a driver
proposed stripping `[SFX: ...]` from `lines[].text` with a regex at the ledger
boundary, and the operator rejected it on the spot -- *"we may need that for
music or tts"*, and *"now video models are doing native audio too."*

**WHAT THIS FORBIDS, concretely:** any new regex, `str.replace`, or Python
sanitizer that edits `lines[].text` to remove leaked cues, stage directions or
markup. Detection helpers are fine. Reporting is fine. Editing is not.

**WHAT IT WOULD LOOK LIKE WHEN BUILT (design, not yet built):** a repair pass
handed the offending line plus its immediate neighbours, asked to return the
line as it should be spoken, with the ledger row replaced only on an accepted
result and the original preserved in the receipt. It is a PRODUCER, so it
degrades rather than refusing -- an episode is never failed because a cleanup
pass declined.

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

### CLOSED 2026-08-23 -- the last no-shims violation

`_otr_content_safety.py`'s REWRITE half is gone (`ed92bff7`):
`propose_safety_patches` (the LLM prompt that reworded a delivered spoken row)
and `apply_safety_cleanup` (the atomic ledger write, carrying both bare
`RuntimeError`s), plus the two pydantic patch models. 165 lines.

**The vocabulary DELIBERATELY survives, and that is not a half-measure.** The
directive bans FILTERING, not knowing the words, and
`tests/test_bug_local_288_sfw_validator.py` keeps the whole retired list green
on purpose -- every term must PASS a line -- because "a deleted test is
silence, and silence is how a policy creeps back". Deleting the tuples would
have deleted that guard's ability to enumerate what must never block again.

THE LAW was checked first: `meta.ledger_cleanup.safety` already has a
deterministic owner (`_otr_ledger_cleanup.py` stamps `status: retired` on every
path), and `validate_sfw` had been `return None` since 2026-08-05. The guard
that replaces the old monkeypatched must-not-run stub asserts ABSENCE -- the
four rewrite entry points do not exist and are not exported -- which the old
one structurally could not do. Module is pure and stdlib-only now, 354 -> 187
lines, and no longer pulls pydantic.

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

### OPEN -- FIVE UNWIRED SYMBOLS, found 2026-08-23 by a zero-reference sweep

A dead-symbol scan over `nodes/` (module-level defs whose name appears exactly
ONCE in the whole repo -- its own definition) returned 13 candidates. Nine are
ordinary dead helpers. **Four are not dead code at all: they are code that was
written to run and never wired**, which is a different and more interesting
thing. NONE were deleted. Each is reported with what it would have done.

**1-2. `_otr_scifi_p0_contract.p0_contract_instruction` and
`p0_contract_receipt` -- and this pair may explain the suspicion in the section
below.** `_otr_scifi_news_pro` imports only `MAX_QUOTE_CHARS` and
`p0_source_chunks` from that module. `p0_contract_instruction` returns "the
model-visible compact-extraction contract for every P0 rung" -- the text that
tells the model *at most 6 facts, 4 entities, 3 numeric rows, one literal span
each, do not paraphrase quotes*. **Nothing calls it, and grepping
`_otr_scifi_news_pro` for `MAX_FACT_ROWS`, `MAX_ENTITY_ROWS` or the phrase "at
most" returns NOTHING.** So the model is never told the caps; it extracts
freely and the grammar ceiling truncates whatever overflows. That is precisely
the "silent evidence thinning" mechanism the next section suspects, arrived at
from the opposite direction. `p0_contract_receipt` -- "the durable bounds
receipt paired with a P0 call journal" -- is likewise never written, so no
ledger records which bounds were (not) applied.
**NOT FIXED HERE ON PURPOSE.** Wiring the instruction in changes a PROMPT, which
changes scripts, and the next section's own rule is "prove it on an artifact
before touching it". This is the artifact-hunt made much cheaper: the thing to
look for is a P0 payload sitting exactly at 6 facts / 4 entities.

**2b. `_otr_determinism.seed_all_rngs` -- DEAD, and MY OWN CONSEQUENCE CLAIM
HERE WAS WRONG (corrected 2026-08-23, on the "verify before you code" pass).**
The symbol is genuinely unreferenced: zero callers repo-wide, twice confirmed
(here, and by the 2026-08-18 kibitz run). **What was wrong is what I said it
MEANT.** This entry read its docstring -- *"legacy audio seeds nothing today, so
a render-twice diverges"* -- and concluded "the divergence it describes is the
current behaviour". It is not. **The docstring is stale, and the coverage moved
rather than vanished:** `deterministic_inference` seeds python / numpy / torch /
cuda from the same int (`_otr_determinism.py:150-156`), and EVERY audio
generation path runs inside it -- voice at `_otr_voice_node_common.py:1289` and
`:1296`, music at `stable_audio_theme.py:266` -- with Bark, MusicGen and
Stable Audio each seeding directly on top. The scoped context manager also
saves and RESTORES prior RNG state, which is strictly better than a permanent
global seed for per-forward bit-identity. So this is a SUPERSEDED symbol, not a
missing guarantee.

**THE LESSON, which is why the correction is kept rather than quietly edited:**
a docstring is evidence about the day it was written, not about HEAD. This entry
treated one as a live measurement and would have sent the next window hunting a
determinism hole that does not exist -- or, worse, wiring a global seeder back
into a path that is already scoped. Verify the CONSEQUENCE, not just the
reference count.

**3. `_otr_scifi_news_pro._validate_scene_envelope`** -- a fail-closed validator
that raises `NewsProScriptError("final_draft", ...)` when an envelope is not an
exact `SceneEnvelope` or does not match its advisory scene plan. Never called.
A guard nobody invokes is not protection, and deleting it would silently accept
a loss nobody chose.

**4. `_vram_log.vram_sentinel`** -- a decorator that snapshots VRAM at entry and
calls `force_vram_offload()` when a TTS/audio function starts above a 6 GB
ceiling. Applied to nothing. Its own docstring calls it "defensive depth, not a
hard gate", so it is not a missing guarantee -- but on a 16 GB card whose known
failure mode is a late OOM, an unapplied VRAM sentinel is worth a look rather
than a delete. (`story_orchestrator` also imported `force_vram_offload` without
using it; that import was swept.)

**THE ORDINARY DEAD HELPERS ARE GONE NOW (408 lines, suite unchanged at 12096 --
not one test touched, which is what "dead" should mean).** Removed:
`story_orchestrator._generate_character_profile` / `_generate_announcer_profile`
/ `_name_similarity` / `_flush_vram_keep_llm`, `_otr_captions._cli` /
`color_for`, `_otr_cast_env.cast_genre` / `other_name_policy`,
`_otr_voice_bank.voice_ref_entry`, `stable_audio_theme._load_meta`,
`slot_matrix.profile_keys_for_all_roles`,
`cloud_media_canonical._not_built_yet`, `_otr_episode_budget._self_test`, and
`_otr_ledger_clean.probe_context_visibility` / `_grade_probe`.

**DELIBERATELY KEPT, and each for a stated reason:** the four schema classes
(`AdapterDescriptor`, `VideoProfileRow`, `ImageEngineConfig`,
`ImageLedgerSection`) are declared contract shapes and the campaign protects
protocol fixtures; `otr_silent_composite.freezedetect_silent` documents its own
non-wiring ("NOT wired into the default assemble path -- it adds a full decode
pass"), so it is an opt-in diagnostic rather than a miss; `content_oracle`'s
manifest wrappers sit over a live oracle; and **every symbol in
`_otr_scifi_p0_contract` stays untouched because that module is the subject of
the finding above** -- deleting `p0_contract_instruction` would destroy the
evidence for it. Note for whoever picks that up: FIVE of that module's symbols
are unreferenced (`compact_p0_repair_context` at 105 lines,
`p0_source_char_budget`, `p0_contract_instruction`, `p0_contract_receipt`,
`P0RepairTrimReceipt`), against just two live exports. Most of that file is
unreachable.

The earlier note said the nine ordinary helpers were left alone, deliberately: the sweep
that found them also found the four above, and a pass that has just learned its
scan surfaces unwired guards is the wrong pass to bulk-delete on. They are
recorded here so the next window does not re-derive them --
`story_orchestrator._generate_character_profile` (76 lines) and
`_generate_announcer_profile` (20), `_otr_captions._cli` (18) and `color_for`
(6), `_otr_voice_bank.voice_ref_entry` (15), `stable_audio_theme._load_meta`
(12), `_otr_cast_env.cast_genre` (6), and
`_otr_shared/slot_matrix.profile_keys_for_all_roles` (3).
`_otr_video_engines/schemas.VideoProfileRow` is ALSO unreferenced and is
explicitly KEPT: it is the declared row shape for a live `video_profiles.yaml`,
and the campaign protects protocol fixtures.

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

## STANDING RULINGS LIFTED OUT OF THE ARCHIVED SECTIONS (2026-08-22)

Four closed/superseded blocks -- the two stale CURRENT STEP blocks, RECENT
DECISIONS, and the 1,135-line closed 08-15 sprint -- were MOVED WHOLE to
`docs/GO_FORWARD_ARCHIVE.md` to lean this file. Nothing was deleted; the archive
and git both hold the full text. **Every standing ruling that lived inside them
is reproduced here verbatim, because the 2026-08-16 audit was right that losing
one costs more than the length does.**

* **OPERATOR RULING ON VRAM, 2026-08-21 evening (hard):** *"don't chase numbers
  please, fail OOM only."* The only VRAM criterion is whether a render OOMed. No
  margin arithmetic, no cost-model fitting, no headroom reporting.
* **CAST-FIRST IS A STANDING OPERATOR RULING (2026-08-20):** *"no cast first,
  cast must be first."* **Do not re-propose script-derived casting for any lane
  that runs the writer, and do not let a panel round reopen it.**
* **Lane 2's VRAM bound is discharged -- do not reopen it as "untested".**
* **`unknown` behaves like the retired `unisex` in the gender guard** -- a
  trapdoor under a door nobody opens; **do not reopen.**
* **DETECTOR TRAP (cost a window once):** ledger LINES key speakers by
  `char_id`, never name -- but frozen post-audio ledgers may DROP char_id on
  lines. Resolve identity from the CAST row, then match lines by whatever key
  the era carries.
* **The voice-identity question is CLOSED 2026-08-18 ON A BLINDED LISTEN. DO NOT
  RE-OPEN IT.** Three arms, blinded; the seed fix won 3-0, the emotion ceiling
  2-1. Receipts: `otr/episodes/lemmy_production_audition_ceiling_2026-08-18/`.
* **Voice-pool concentration: CLOSED 2026-08-19 -- do not re-open.** Flip, live
  proof and rip all shipped (`429b73aa`).
* **PBUG-20260817-07 (stage directions in captions): WILL-NOT-FIX**, operator
  ruling.
* **TRAP -- public `ltx_video` prompt guidance CONTRADICTS an operator
  directive.** The guides say camera-move-first; he directed subject-first, and
  rewriting the registers the wrong way cost 40% of the motion at a fixed seed.
  **Do not adopt it.** The sibling Wan lane proves no global prompt rule works --
  each engine's `PROMPT_STYLE_NOTES` is the authority.
* **Disguise plots are a legitimate gender-scan hit:** ROSALIND-as-Ganymede and
  VIOLA-as-Cesario keep female voices by operator ruling. Read that list, do not
  total it.
* **The shakespeare attribution-wording row still needs an OPERATOR RULING ON
  WORDING before it is coded** (it reads as an over-application of the
  2026-08-05 licensed-source ruling).
* **LTX 2.5 is UNQUALIFIED and nothing may read as proven that is not:** no cost
  row, no envelope key. Chunk B (the foley bed) remains BLOCKED on execution
  order, unchanged by Chunk A shipping.

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on and why.
Pick the cheapest tool that can win; escalate only when the cheaper rung cannot decide.
Both pools reset weekly -- front-load heavy coder windows and big Codex spends early in
the credit week; late-week, drop to the $0 rungs instead of grinding a paid pool dry.

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|
| 1 | Local Qwen on the 4060 (`10.55.0.2:1234`, LM Studio; the `4060` skill drives it) | $0 | Read-only FIRST-PASS triage of failures, logs and diffs before any credit spend | Final diagnosis, patches, tests, live qualification; NEVER loaded on the 5080 (that GPU renders only) |
| 2 | agy / Antigravity (`KIBITZ_AGY_MODEL` set to the DISPLAY name exactly -- a wrong id silently kills the lane and the arc runs codex-only) | $0 | Default grounded reviewer for mechanical review; second panelist on every kibitz | -- (if it times out, raise `KIBITZ_AGY_PRINT_TIMEOUT`, do not call it dead) |
| 3 | Codex CLI `gpt-5.6-sol` (high) | weekly credits | The second opinion of record: the two-strikes third-attempt panel, pre-execution grounding of big blocks, live-failure kibitz | Mechanical review agy can do alone. Verify the selected model every arc -- a stale skill cache once drifted mid-arc unnoticed |
| 4 | Sonnet 5 (Cowork subagent) | weekly credits | Post-coding QA on a FROZEN diff (standing 08-05 rule); a valid substitute reviewer seat when a kibitz lane is quota-held | Driving a window; multiplying reviewers on an already-clean diff (the 08-20 one-clean-review ruling) |
| 5 | Claude Opus (Cowork, this window) | weekly credits | The actual work: planner and coder windows, anchor and sole judge on every panel, live-run drive, lane closes | Babysitting renders; single-small-item windows (batch per Window packing) |
| 6 | Cloud roundtable (OpenRouter) | real $ | Genuine R1 ideas passes only; the <$20 autonomy rule applies | Mechanical or grounding review -- that is rungs 2-3 |
| 7 | Fable | scarce | Exactly two uses: the cold FIRST opinion on an r1 design round, and the final grounded gate before a high-stakes, hard-to-unwind production change | Anything mechanical. Do not burn a scalpel on a screw |

**RESTORED 2026-08-21 (this window).** The table had a header, a separator and
NO ROWS for an unknown stretch, so every window was asked to cite a rung from an
empty ladder and answered from the prose paragraph below instead. The rungs are
recovered from `ed8d5a6d` (where the operator wrote them) and refreshed to what
is actually true now: Sonnet 5's post-QA seat is added because the 08-05 rule
made it standing, Fable's row is narrowed to the two uses the 08-21 handoff
recorded, and the stale per-model version pins and the frozen "reset state
2026-07-24" line are dropped rather than re-asserted. Review ROUTING still lives
in the dated REVIEW ROUTING block at the top of this file, not here.

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
fail-closed) still stands in full.

**THE OLD POINTER HERE WAS DEAD AND IS NOW CORRECTED (2026-08-22).** This
paragraph used to end "the runtime filters that survived the 08-03 rip are
inventoried and queued for removal as ON DECK item 5". ON DECK item 5 is about
1,090 cast rows claiming a non-commercial model is commercially clean -- it has
nothing to do with filters. A window sent there finds nothing, which is exactly
the failure the top of this file warns about.

**The real state, checked in code rather than inferred:** the spoken-content
repair pass is ALREADY retired. `run_ledger_cleanup` does not call
`_otr_content_safety.apply_safety_cleanup`, `meta.ledger_cleanup.safety` is
stamped `"retired"` so the ledger field keeps an owner, and
`tests/test_ledger_cleanup_pass.py` is deliberately INVERTED -- it asserts the
pass never runs and the author's line is never edited. The Shakespeare prompt's
"guns/knives/weapons" clause was deleted 2026-08-05 with its reason written in
place.

**So there is nothing queued here, and the module must NOT be deleted.** That
inverted test is the tripwire that makes a re-armed content filter fail loudly;
removing the module removes the guard. Reasoning and the wider dead-symbol
sweep: `docs/2026-08-22-dead-symbol-inventory.md`.

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

## Tombstones -- the only three a window might wrongly revive

Full list in `docs/HANDOFF_LOG.md` + `docs/PROD_BUG_LOG.md`. These three are
here because each has been re-proposed at least once:

* **The 20 fabricated-fixture `public_domain` episodes and the fixture itself** --
  operator ruling 2026-08-04: dropped and deleted, **never raise again**.
* **v4 improvement campaign banks #2-#5** -- PARKED, superseded by the keep-6
  rename + THE LAW. Revive only by operator decision
  (`docs/2026-07-17-v4-campaign/final.md`).
* **LEAN-MEAN** -- scheduled in `ROADMAP.md`; executable scope and order live in
  `docs/LEAN_MEAN_CLEANUP.md`, not this file.

## Pointers

- `CLAUDE.md` -- hard operator rules; wins over this file wherever they disagree
- `ROADMAP.md` (later-runway schedule)
- `docs/LEAN_MEAN_CLEANUP.md` (current lean-mean scope, loss matrix, and coding order)
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

---

# Moved from the plan 2026-08-23 (second pass)

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

### Re-ground gate for active work

Re-ground each active table item against current committed HEAD before coding.
The normal entry is `r3 -> r4`; drop to r2 when the coding plan, authority, or
precondition is wrong rather than merely line-stale. No item executes without
current r4 convergence and recorded round artifacts. Lean-mean has the stronger
full `r2 -> r3 -> r4` gate defined in `docs/LEAN_MEAN_CLEANUP.md`.

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

## THE REGISTRY AND COMFYUI-MANAGER ARE TWO DIFFERENT SYSTEMS (measured 2026-08-23)
### >>> PARTLY SUPERSEDED 2026-08-24: points 2 and 4's conclusions are WRONG -- see THE REAL NODE-EXTRACTION PIPELINE below. The Manager half (point 3) still stands. <<<

Written because a publishing plan was drafted on the belief that shipping a file
would make registry.comfy.org list our nodes, with "34 exact OTR_* node IDs" as
its input. Every number in that sentence was wrong, and the mechanism does not
exist. **Measure before publishing -- a version string is burned permanently.**

**1. THE PACK IS FINE. `GET https://api.comfy.org/nodes/comfyui-old-time-radio`
returns `status: NodeStatusActive`, `latest_version: 2.0.0-alpha.6`, 13
downloads.** Not pending, not flagged. The install path works.

**2. THE PER-NODE PANEL IS DEAD FOR EVERYONE, NOT FOR US.**
`GET /nodes/<id>/comfy-nodes` returned **HTTP 404 for every pack sampled** --
`comfyui-old-time-radio`, `comfyui-kjnodes`, `rgthree-comfy`,
`comfyui-dramabox`. kjnodes and rgthree are among the most-installed packs in
the ecosystem. **An empty node panel is the universal state**, so it can never
be an acceptance test, and no file we ship changes it. This re-confirms
CLAUDE.md section 7A from the other direction.

**3. COMFYUI-MANAGER IS A SEPARATE DATABASE AND WE ARE NOT IN IT.** Manager
keeps `custom-node-list.json` and `extension-node-map.json` (**4,189 repos
mapped**, keyed by repo URL -> node id list) inside its OWN repo. **OTR appears
in NEITHER (0 matches in both).** That -- not the Registry -- is why
missing-node auto-suggestion does not offer OTR. Getting in is an EXTERNAL act
(a PR to ltdrdata's repo), not something a publish accomplishes.

**4. THE ONE REAL DEFECT ON OUR SIDE: our node ids were not statically
readable.** `__init__.py` sets `NODE_CLASS_MAPPINGS = {}` and fills it at import
time from `_NODE_MODULES` plus a merged `_otr_class_registry` table, through 14
try/except guards. **An AST scanner keyed on `NODE_CLASS_MAPPINGS` extracts ZERO
ids.** Any external extractor sees nothing. Fixed by generating a literal
root-level `node_list.json` (ships -- no `.comfyignore` pattern catches it) and
pinning it with `tests/test_node_list_manifest.py`, which includes a VACUITY
FLOOR so an empty-vs-empty comparison cannot pass.

**5. THE COUNT WAS WRONG THREE WAYS, AND THAT IS THE REAL LESSON.**
* A comment in `__init__.py` said **34** -- pre-lean-mean, retired nodes. It was
  quoted back as fact in the plan. Comment now removed.
* `/object_info` shows **29** `OTR_*` ids -- but **4 belong to a DIFFERENT pack**
  (`ComfyUI-OTR-UpstreamStoryLab`, per each node's `python_module`), and are not
  ours to declare.
* This pack declares and loads exactly **25**.
`/object_info` plus each node's `python_module` is the authority on who owns an
id. A grep for `OTR_` is not -- it crosses pack boundaries.

**WHAT IS STILL UNPROVEN, STATED HONESTLY:** whether ANY consumer reads
`node_list.json`. The convention's real-world use (comfyui-impact-pack, the
canonical example) is a rename/alias hint map -- `{"Old Name": "renamed to X"}`
-- not a node index. The file is cheap, correct, drift-guarded and ships with
the next version bump anyway, so it costs nothing; but **it was not published as
a fix, because the acceptance test it was meant to satisfy cannot pass for any
pack.** `pyproject.toml` was deliberately NOT touched: editing it auto-fires a
publish, and there was no measured reason to spend a version.

## PUBLIC_DOMAIN MAY PARAPHRASE. ONLY DIALOGUE IS OWED (operator ruling 2026-08-23, hard)

**This QUALIFIES the older "fidelity lanes invent nothing" line, which had been
read as binding `public_domain` and `shakespeare` equally. It does not.**

The operator, ending the source-grounding item outright:

> *"no no public domain does not need to get author's words unless they are
> dialogue, it can paraphrase"*

> *"please do not chase as long as it carries the story and some dialogue if
> present"*

**THE BAR ON `public_domain`, and it is the whole bar:** the episode carries the
STORY, and carries some DIALOGUE where the source actually has dialogue.
**A paraphrased narration is CORRECT OUTPUT, not a defect.** Do not open
writing-quality work against it, do not file a PBUG because an episode reworded
the author's prose, and do not re-open the grounding campaign -- it is DEFERRED
in the go-forward, with its full original text in the archive.

**`shakespeare` IS UNCHANGED.** It remains the VERBATIM lane, `exchange_compose`
is NOT RUN there, and the author's language is carried as written. The two lanes
were never the same contract; treating them as one is what produced a
three-leg campaign for a lane that only ever needed dialogue.

**IF IT IS EVER TAKEN UP AGAIN**, the only sanctioned scope is source DIALOGUE
when the source has it -- never a prose window over the canonical body.

**AND CORRECT THE RECORD FIRST: the claim that "the composer sees no source at
all" is FALSE, and it was mine.** It came from grepping
`source_text|full_text|source_meta|excerpt|canonical_body|source_window` -- none
of which is the name the code uses. The real name is **`source_block`**, and the
machinery is BUILT AND TESTED:
* `build_exchange_prompt(source_block=...)` at `nodes/_otr_compose_exchange.py:300`,
  with its paired CARRY-THEM text at `:430-435` and a loud non-str refusal at
  `:420-428`; forwarded through `compose_exchange` at `:551/:569`.
* `LineRequest.source_block` at `nodes/_otr_line_composer.py:292`, rendered above
  the WRITE LINE cue at `:908-915`.
* An ENTIRE selector module, `nodes/_otr_source_grounding.py` (16 KB) --
  `select_grounding` at `:255`, `SourceGrounding` at `:168`,
  `render_source_block` at `:362`.
**`select_grounding` has ZERO production callers: 1 reference under `nodes/`
(its own definition) against 33 under `tests/`.** So this was never a design
problem. It is an unwired supply line between a built producer and a built
consumer, and the honest scope is three seams -- the exchange prepass, the
writer's `select_grounding` call, and the `LineRequest` construction.
**THE LESSON: a zero-reference grep proves nothing unless you grepped the name
the code actually uses.** That false premise is exactly what made a wiring job
look like a three-leg campaign.

**WHY THIS IS WRITTEN DOWN RATHER THAN JUST OBEYED:** the item was the sprint's
headline row, motivated by a real artifact (a Wells adaptation that produced
"Arkham, Massachusetts"). A future reader WILL rediscover that anecdote and read
it as an open defect. It is not one. The operator was asked and answered.

### SUPERSEDED 2026-08-24 -- THIS SECTION'S CONCLUSION WAS WRONG (kept as a lesson in probing the wrong endpoint)
#### was: THE DECISIVE EVIDENCE (added 2026-08-23): THE REGISTRY SCHEMA HAS NO PLACE TO PUT NODES

Probed directly, so nobody spends another version chasing this.

**A version record's COMPLETE key set** (`GET
/nodes/comfyui-old-time-radio/versions/2.0.0-alpha.6`, HTTP 200): `changelog`,
`createdAt`, `dependencies`, `deprecated`, `downloadUrl`, `id`, `node_id`,
`status`, `supported_accelerators`, `supported_comfyui_frontend_version`,
`supported_comfyui_version`, `supported_os`, `tags`, `tags_admin`, `version`.
**There is NO node-list field of any kind.** The registry models the PACK, not
the node classes inside it. Both endpoints that would enumerate them --
`/comfy-nodes` and `/nodes` -- return **404**.

**OUR RECORD IS SHAPED IDENTICALLY TO THE MOST-INSTALLED PACKS.**
`comfyui-kjnodes` 1.5.0 returns `supported_os []`, `supported_comfyui_version ""`,
`supported_accelerators []`, `tags []` -- exactly ours -- and its `[tool.comfy]`
is the same three keys we declare (PublisherId / DisplayName / Icon). **We are not
misconfigured.** Nothing about our packaging explains the empty node panel,
because the panel is not fed by anything we can publish.

**THEREFORE the "replace dynamic registration with a literal static
NODE_CLASS_MAPPINGS" idea CANNOT achieve registry node visibility** -- there is no
field for the result to land in. Making ids statically readable was still worth
doing on its own merits (see `node_list.json` above), but it is not a fix for
this, and it must not be sold as one.

**THE ONE REAL LEVER THAT EXISTS**, and it is small: `requires-comfyui` (e.g.
`requires-comfyui = ">=0.3.68"`, as `ComfyUI-AnimateDiff-Evolved` declares in
`[tool.comfy]`) populates `supported_comfyui_version`. It is the only supported_*
field with a publisher-side input. It affects compatibility filtering, NOT node
listing. Costs a version bump, so only spend it alongside a change worth
publishing.

**AND THE PENDING WINDOW IS NOT A FAILURE.** `2.0.0-alpha.7` uploaded correctly
with `deps=12` and sat at `NodeVersionStatusPending`, during which
`latest_version` still resolved to `alpha.6`. That is the documented behaviour --
Comfy-Org's cron only considers versions older than 30 minutes and there is no
publisher self-service path to Active. "The push isn't done" and "the push
failed" look identical for half an hour. Read the versions list, not
`latest_version`.

## THE REAL NODE-EXTRACTION PIPELINE (source-verified 2026-08-24 -- supersedes both sections above where they conflict)

The operator produced two packs whose registry pages DO list nodes
(`wanblockswap`, `ComfyUI-WithAnyone`). That control broke last night's
conclusion, and the true mechanism was then read out of
`Comfy-Org/registry-backend` directly.

**1. THE ENDPOINT EXISTS -- last night's probe used the wrong URL shape.**
`GET /nodes/<id>/comfy-nodes` 404s for everyone, which is what produced the
false "dead for all packs" ruling. The REAL shapes work:
`GET /comfy-nodes?node_id=<id>` and
`GET /nodes/<id>/versions/<version>/comfy-nodes`. Measured totals:
comfyui-impact-pack **7,921**, comfyui-kjnodes **4,206**, rgthree-comfy
**1,124**, wanblockswap 1, ComfyUI-WithAnyone populated (ids are
CASE-SENSITIVE; the lowercase spelling returns an empty 200, not an error) --
and **comfyui-old-time-radio 0, on alpha.6 AND alpha.7.**

**2. EXTRACTION IS IMPORT-BASED, NOT STATIC.**
`registry-backend/node-pack-extract/` boots a REAL CPU-only ComfyUI
(`ai-dock/comfyui:v2-cpu-22.04`), installs the published CDN zip into
`custom_nodes/<node_id>/`, runs `pip install -r requirements.txt` (under
`set -e`) plus `install.py` if present, then polls `localhost:8188/object_info`
(TIMEOUT=3600) filtering on `python_module == "custom_nodes.<node_id>"`. Zero
matches -> `"node cannot be loaded into comfy ui"` -- the exact frontend
message. **Registration style is IRRELEVANT: rgthree builds its mappings
dynamically and extracted 1,124.** So a literal static `NODE_CLASS_MAPPINGS`
rewrite is still pointless for this -- for the OPPOSITE reason recorded last
night. `node_list.json` is equally irrelevant to this pipeline (harmless;
keep it for its own sake).

**3. OUR PACK LOADS CLEAN UNDER A FAITHFUL REPRODUCTION.** The published
alpha.7 zip, extracted into a folder named `comfyui-old-time-radio`, imported
under the exact hyphenated module name with prestartup executed first, NO
OTR_TEST_MODE, CPU-only: **prestartup ok, 25/25 nodes registered, zero
failures.** The pack is not the reason extraction returns nothing.

**4. WHY OURS IS EMPTY -- the pipeline's own bookkeeping.** Every version row
carries `comfy_node_extract_status`, default **'pending'**. The ONLY thing
that fires extraction is `TriggerComfyNodesBackfill`
(`services/registry/registry_svc.go`), exposed as `POST /comfy-nodes/backfill`
(auth-gated, **default max_node=10 per sweep**), which queues ONLY
status='pending' versions onto Pub/Sub -> Cloud Build. Two consequences:
* A fresh version can sit 'pending' for a long time -- the sweep takes 10
  versions per run ACROSS THE WHOLE REGISTRY, unordered.
* `MarkComfyNodeExtractionFailed` flips a version to 'failed', and **the
  backfill query never selects 'failed' -- a failed extraction is TERMINAL,
  never retried.** If alpha.4-6 ran and failed (e.g. a pip failure inside
  their container), they will stay node-less forever no matter what we do.
The status field is NOT exposed on the public API, so pending-vs-failed for
our versions cannot be read from outside. That is the precise question for
Comfy-Org, and it replaces the older ask: *"what is
comfy_node_extract_status for comfyui-old-time-radio's versions -- and if
'failed', what did the Cloud Build log say, and can they be re-queued?"*

**5. WHAT REMAINS TRUE FROM LAST NIGHT:** alpha.7's `Pending`
security-scan status is a separate pipeline from node extraction; the
ComfyUI-Manager database is a third, separate system; and the pack itself is
healthy (25/25 from the shipped artifact).

**THE LESSON, twice in one day: a 404 is evidence about ONE URL, not about a
capability.** Last night's "dead for every pack" ruling generalized four 404s
from a wrong path shape into a system-wide claim, and it survived into two
committed docs and an operator-facing recommendation. The control that broke
it was two packs the operator found in five minutes of browsing. When a
measurement says "this is broken for everyone", FIRST hunt for one
counterexample before writing the ruling.

## THE SMOKING GUN (2026-08-24): THE AUTOMATIC BACKFILL SCHEDULER IS PAUSED WITH A LEAP-DAY-ONLY SCHEDULE

Read directly from `infrastructure/modules/node-pack-extract-trigger/main.tf`
and the prod deployment `infrastructure/prod/main.tf` in
`Comfy-Org/registry-backend`.

**The Cloud Scheduler job that periodically sweeps `pending` extractions is
provisioned `paused = true`, with a default `backfill_job_schedule` of
`"30 3 29 2 *"` -- 3:30am on February 29th, UTC.** Next occurrence: 2028.
Prod's `main.tf` does not override `backfill_job_schedule` or `paused` --
it sets only `project_id`, `region`, `bucket_name`,
`cloud_build_service_account`, `topic_name`, `registry_backend_url`,
`backfill_job_name`. **So the periodic sweep is, by design or by default,
never going to fire on its own.** This is strong evidence the backfill is
meant to be triggered manually by a Comfy-Org engineer, not something that
"just needs time."

**There IS a second path that looks like it should be real-time:** a GCS
bucket notification (`OBJECT_FINALIZE`) on the `comfy-registry` bucket
publishes to Pub/Sub, which fires the SAME `node-pack-extract` Cloud Build
job per-upload. **But its trigger substitution hardcodes
`_CUSTOM_NODE_NAME = "custom-node"`** -- a literal string, not derived from
the uploaded object's path. If that is genuinely what runs in production,
every real-time extraction would filter on `python_module ==
"custom_nodes.custom-node"`, matching NOTHING for any real pack id. This
cannot be fully confirmed from the public repo alone (GCP console config can
diverge from what Terraform last applied), so it is reported as what the
source shows, not as certain fact.

**CONSEQUENCE FOR THE ASK:** there is no publisher-facing API to trigger our
own extraction, confirmed by reading `PublishNodeVersion` in full -- it
creates the version and returns a signed upload URL, nothing else. Waiting
longer is not expected to help on its own. **The correct next action is
asking a human**, via Comfy-Org's Discord (`discord.gg/comfyorg`) or a
GitHub issue on the public `Comfy-Org/registry-backend` tracker (issues
enabled, 24 open) -- pointing at the two facts above and asking for a manual
backfill run scoped to `comfyui-old-time-radio`.

## ALL AUDIO LANES SHIP ON KOKORO (operator ruling 2026-09-01, hard)

Operator: *"we can ship all audio lanes with kokoro and let people know: hey, you want a
better TTS, you can install them on your own, with a matrix of what's compatible."* And,
the same day: *"kokoro onnx is our new go-to."*

* The shipped default for BOTH voice slots (announcer and character), in the canonical
  workflow and every generated variant, is kokoro through the kokoro-onnx backend
  (Section 1.11 of the plan), because it is the only local voice that pip-installs on the
  Python 3.13 that ComfyUI Desktop and the portable ship, on Linux, and on Mac.
* indextts2, chatterbox, dia and bark stay in the dropdowns as upgrades the user installs
  on their own. They are not defaults anywhere, and no shipped graph may depend on a
  reference WAV that does not ship.
* What "compatible" means is published as ONE generated table in `docs/MACHINE_MATRIX.md`
  from the audio engine registry, never a hand-kept list; README points at it.
* This closes the 2026-09-01 ship-audit blocker about the indextts2 default without
  shipping WAVs, and it makes the Python 3.12 / 3.13 split irrelevant to voices.
