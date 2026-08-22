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

## THE CURRENT STEP, 2026-08-22 -- READ THIS FIRST

### DONE AND PUSHED THIS SESSION (HEAD e80c6786)

**1. `ideogram4_local` SHIPPED -- the typography-first still engine, opt-in.**
Adapter + guarded registration + 9-key CAPABILITIES row + 64 focused tests +
`config/profiles/otr_ideogram4_local_still_word.json`. In the image dropdown for
everyone, distinct from cloud `ideo`, no flag or gate. Sonnet QA verdict CLEAN
(no must-fix); its three nits folded with tests.

**2. THE GOLDEN RULE IS SATISFIED: every engine works in any of the 3 slots.**
Operator: *"every video lane should work in any 3 slots"* and *"still_word never
needs a portrait, only words"*. The blocker was NEVER the engine: `still_word`
had ALWAYS declared `StillPlanRow(kind="portrait", required="never")` and
nothing read it (`still_plan_helpers`: *"Nothing in this module reads the plan
for production"*). Every still_word episode minted a portrait per cast member no
consumer loads -- invisible on engines that draw faces, FATAL on the first that
refuses. Fixed by `_portrait_free_roles_from_policy`, the fifth lane-derived
role set beside `still_aspects` / `mesh_fodder_roles` / `talking_roles` /
`still_word_roles`. **It reads each lane's OWN declaration, never an engine
name, so a NEW LANE IS COVERED THE DAY IT DECLARES ITS PLAN.** Documented as
`docs/VIDEO_LANE_PREFLIGHT.md` G3.7.

**3. Live proof.** Episode *"The Weight of the Grain"*, all 3 image slots on
`ideogram4_local`, all 3 video slots on `still_word`: **6 of 6 dialogue word
cards rendered** plus the music opening -- 7 of 8 stills. Cards are genuinely
good (elegant serif, correctly spelled, over real photographic scenes).

**4. Captions 36 -> 42 px** (operator's chosen size). Size only; the rest of his
1080p spec was already met (1920x1080, 44-char wrap, opaque box, centered).

**5. `docs/IMAGE_GEN_PREFLIGHT.md`** + `tests/test_image_gen_preflight_matrix.py`
(the image-model sibling of the video preflight), and the HOW-TO-ADD-YOUR-OWN
guides EXTENDING_OTR promised, written into both engine namespace `__init__`s.

**6. Bug Bible 305.** `12.126` broadened from "a foreign key is rendered" to the
real rule: **on a zeroed-negative topology a prohibition IS positive
conditioning** -- audit your own upstream prompt before blaming the model. OTR's
own composer appends *"no logos"* to every word card; the card that came back
reading `NO MISCOS` was that guard, painted on.

### OPEN, IN PRIORITY ORDER

**QUEUE HEAD IS NOW B. Item A is DONE -- see the closed block directly below.**

**A. GHOST SIGNAL IS BUILT, PROVEN AND SHIPPING (closed 2026-08-22).**
Three green chunks, all pushed on `v2.0-alpha`:
* `01317eec` -- dependency + schema lock (ADE pinned at
  `9257651221002dcba0a12f9cff37e1944e58fb60`, both artifacts hash-verified
  before boot, `/object_info` captured from a post-install clean boot).
* `d0b3a65b` -- the wired implementation slice (adapter, pure composer, durable
  subject sigil, clean-Lanczos delivery mode, registry/profile/G4/G3.7, 353
  focused tests).
* The publish receipt + `shipping` promotion (this session's third chunk).
**Live proof:** episode `signal_lost_the_constables_knock_20260822_050116`,
8/8 beats on `animatediff15_video`, `RESULT SUCCESS` + `obs_publish OK`, 21:33,
1920x1080 @ 25 fps in the LIVE `otr/obs`. Every section-10 pass condition
verified -- receipt at `docs/2026-08-22-ghost-signal-publish-receipt.md`.
* **The first leg FAILED and it was worth it.** A bookend beat with the motion
  pass off pulls the pack's own subject (163-178 chars) AND its motion register
  (130-209); on `recur_frac` that composes to ~474 against a 320 ceiling, and
  `_trim_to` could not shrink either because both are comma-free prose. The
  composer refused at cast-time preflight before a weight loaded. Root-fixed:
  word-boundary fallback, no dangling function words, and step 4 of the trim
  order now shrinks pack surfaces. **Every unit budget test had handed the
  composer an authored clause, which short-circuits the pack register entirely
  -- that is the test-design lesson, not a Ghost-specific one.**
* **THE LOOK IS ACCEPTED, DO NOT CHASE IT.** The motion reads fast (12.5 fps of
  AnimateDiff held to 25). Operator, watching the published episode: *"i was
  expecting experimental vj"* and *"its perfect"*. Not an open defect.
* **NO VRAM CLAIM.** Admission stays unenforced; the lane may OOM. A single
  5872 MiB / 100% reading was observed and is NOT a qualified cost row.
* **G3.7 note worth keeping:** `_portrait_free_roles_from_policy` is INERT for a
  lane with an EMPTY `still_plan` -- it looks for a portrait row saying "never".
  A no-still lane is covered by the stronger `accepts_still = False` gate at the
  image dispatcher instead. Adding a portrait/never row to make the role set
  light up would be a declaration the lane cannot honour.
* **OWED, and honestly stated:** the independent finished-diff review seat went
  UNFILLED. The Agent tool was disabled for the session and the substituted
  Codex CLI lane thrashed in a PowerShell error loop (1.4 MB of tool errors, no
  report) and was killed. The live leg is stronger evidence and it caught the
  real blocker -- but it is not a review, and this line says so rather than
  implying coverage.

**A-ORIGINAL (superseded, kept for the receipts). GHOST SIGNAL (AnimateDiff) -- CODE-AND-WIRE-READY, r4 COMPLETE, NOT BUILT.**
The operator's ultra-low-VRAM video lane. A separate session took it through a
full four-round Kibitz arc; **no code, dependency, workflow, render, test,
measurement, commit or push has been performed.** This is the next build.
* Plan: `docs/2026-08-22-GHOST-SIGNAL-CODING-PLAN.md`
* Judgment: `kibitz-runs/2026-08-22-ghost-signal-code-ready-plan/r4/judgment.md`
* Receipt: `kibitz-runs/2026-08-22-ghost-signal-code-ready-plan/r4/final.md`
* **The receipt is BYTE-IDENTICAL to the plan -- verified,
  sha256 `06104e911e599f7630563eaab5ae5cb5aa92c40e0331d3d0e253ba6c4150e81b`.**
* Upstream: `ComfyUI-AnimateDiff-Evolved` (Kosinkadink) is the canonical repo;
  the pinned graph uses the verified `ADE_AnimateDiffLoaderGen1` and
  `ADE_StandardStaticContextOptions`.
* The plan FREEZES: all eight nodes with every input/output/widget and all ten
  links; staged executor aliases and low-VRAM release ordering; fresh
  audio-length source animation per beat at 12.5 fps held twice to 25 fps;
  ledger motion + the selected visual-style pack as prompt authorities;
  character sigils where available and strong abstract treatments otherwise;
  fixed 512x288 generation enlarged directly to 1920x1080 with clean Lanczos;
  the complete profile, canonical-workflow wiring audit, implementation
  sequence, tests and one publish smoke.
* **G3.7 applies from day one:** Ghost Signal must declare its `still_plan`
  (kind / cardinality / aspect / required / framing), and the portrait-free
  role set will honour it automatically -- no edit to the image phase.

**B. The Ideogram music-card refusal, and the asymmetry it exposed.** One card
in eight refused (`still_music_closing_001`, min=78.0 std=10.6). It is
SEED-dependent, not content: the music OPENING card rendered from the same
composer, title and prompt shape. **A refusal currently fails the WHOLE
EPISODE**, which is harsher than the operator's ruling intended -- he agreed to
accept *blemishes* (*"I accept some errors ... don't want it burning extra GPU
cycles"*), not dead renders. Decide: root-cause the prompt as every other
refusal in this campaign was, or let a refusal degrade rather than abort.

**C. `_still_word_fit_card` truncation.** A shipped card read *"...of a
subterranean"* and stopped. Deterministic reduction working as designed, and
unrelated to any engine -- but on a card whose whole job is the words, a
sentence cut mid-clause deserves a second look.

**D. The guides/dialect/recipe refactor (operator's architecture, staged).**
*"the PROMPT is the LANGUAGE from the video lane; the DIALECT is the
instructions for prompting the image lane -- double vs single quotes, neg prompt
vs no neg prompt, temperature, seed, knobs."* Three layers, all existing repo
vocabulary: **prompt** (lane) / **dialect** (engine phrasing) / **recipe**
(engine parameters -- the repo already says `recipe`). Today the composer
PRE-JOINS everything and each engine regexes it back apart; `ideogram4_local`
already implements a dialect without it being called one. Its own item, its own
arc -- do NOT bundle it with a small fix. `docs/2026-08-22-golden-rule-any-engine-any-slot/PLAN.md` section 8.

**E. IG2.2 is a paper guarantee.** It asserts an engine *claims* every role, not
that it *serves* one -- which is exactly how `ideogram4_local` shipped and then
refused at render. Strengthen to the honest, GPU-free claim.

**F. Regression sweep over shipped video profiles** (carried): only
`otr_w45_wan_ti2v` is proven; `otr_g4_wan_ti2v` and `otr_upscale_ship` remain
unexercised.

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

**THE MODEL & CREDIT BUDGET LADDER WAS EMPTY; IT IS NOW RESTORED (2026-08-21).**
The table at the "MODEL & CREDIT BUDGET" heading below had a header row and a
separator and no rows under them, so the rungs every window is asked to cite did
not exist in this document and windows answered from the prose paragraph beneath
it instead. Seven rungs are now recovered from `ed8d5a6d` and refreshed to
current fact; see the note directly under that table for exactly what changed.

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
