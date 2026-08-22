# GO_FORWARD archive

Closed and superseded blocks moved out of `docs/GO_FORWARD_PLAN.md` so that file
stays lean. **Nothing here is live.** The standing rulings that were embedded in
these blocks were lifted back into GO_FORWARD_PLAN under "STANDING RULINGS
LIFTED OUT OF THE ARCHIVED SECTIONS"; what remains below is the receipts,
narrative and superseded framings that justified them.

Archived 2026-08-22 after Ghost Signal shipped.

## THE CURRENT STEP, 2026-08-21 EVENING -- READ THIS FIRST

**THE QUALITY PROGRAMME IS CLOSED. SEVEN LANES, ONE WIN, AND IT SHIPPED.**

**LANE 6 SHIPPED: `wan_ti2v` recipe v2, tiled decode OFF** (OTR `30a38fa9`,
workbench `0e2c5b7`). Operator: *"yes ship the quality"*.
* **The reason is a VISIBLE DEFECT, not a statistic.** Tiled decode painted a
  transient blue blob into every `wan_ti2v` clip the show has ever produced.
  The operator found it BLIND in a side-by-side, unprompted, after the lane had
  already closed NO WIN on a seam matrix. Sweep: lane 1 both arms 4-9 blob
  frames of 13, lane 4 both arms 8-10, lane 6 tiled 1-3, **lane 6 untiled ZERO
  at both seeds**, ltx25 zero. Tiling was frozen ON for the life of v1, so
  every prior render was tiled -- the defect sat in BOTH arms of every earlier
  A/B and lanes 1 and 4 were scored over it.
* **v1 is untouched**; v2 is a new versioned dict, the alias repointed, the id
  bumped, per the procedure the file documents. The four tile-geometry keys are
  RETAINED and unread -- dropping them KeyErrors the moment a sweep turns tiling
  back on.
* Suite 11326 -> **11329**, Bible **304** (`12.124`, `12.125` promoted), live
  canonical leg RESULT SUCCESS, published to `otr/obs` (88 -> 89,
  `The Phonograph's Secret`), ledger stamps `..._v2` with no prequalification
  suffix.
* **Both decode modes are FLAT with clip length**, retiring v1's stated
  justification (borrowed from the ltx tier, as its own comment admitted).

**LANE 7 CLOSED: NO WIN.** The `ltx25` motion fixture lane 2 said it owed. The
motion gate passed decisively (excursion 39-50 against a floor of 6), and the
admission gate then rejected all four cells, so no panel ran. The substantive
finding: **the anchor stops governing departure once the clip really moves** --
soft departs further in only 2 of 4 cells and every value sits near zero, so
both settings end up essentially uncorrelated with the conditioning still.
Lane 2's bound is discharged; do not reopen it as "untested".

**IDEOGRAM 4: STILL NO, BUT THE REASON CHANGED AND THE GAP CLOSED TO ONE
DEFECT** (`docs/2026-08-21-ideogram4-verdict.md`, ROUND 4-5). The operator's
1080p 16:9 retest overturned the earlier reason: the model does NOT refuse this
content -- refusal tracks the PROMPT SHAPE (prose refuses 6/6 at both canvases;
the JSON card schema renders 6/6 at both). The card `aspect_ratio` field was
also malformed in rounds 1-3 (literal pixels where the schema wants a ratio).
The canonical template was then found to CARRY the official schema (a hidden
"Ideogram4 Caption Prompt Template" magic-prompt subgraph, node 114): exactly
three top-level keys, minified, bbox normalized 0-1000 -- and our v2 payload's
foreign `negative_instruction` key was being PAINTED ONTO THE CARD ("NO
MISCOS" = "no logos"). Rebuilt in the official schema at 1920x1088: zero
refusals, perfect spelling, frame-filling type, invented text down from six
gibberish lines to small period footers. The remaining single defect is that
populate footer (a fake catalog number) on a fraction of cards. The restraint
arm measured the schema's own off-switch and it does NOT remove the footer
(treason clean both arms, witches/pompeii footer both arms). The footer IS
trivially detectable -- bottom-margin bright-pixel mass separates clean (0)
from defective (>= 1010) with total separation on all six cards -- so a
detect-and-reroll adapter converges (verdict ROUND 5). **This is now an operator build decision** -- a per-engine prompt adapter is
legal under `docs/IMAGE_GEN_PREFLIGHT.md` IG5.1 -- not a screening rejection.
Weights stay on disk; `z_image_turbo` still ships.

**OPERATOR RULING ON VRAM, 2026-08-21 evening (hard):** *"don't chase numbers
please, fail OOM only."* The only VRAM criterion is whether a render OOMed. No
margin arithmetic, no cost-model fitting, no headroom reporting. This sharpens
the standing no-VRAM-ceremony rule -- an OOM is a plain recorded fault and
everything else is noise.

**OWED, AND NOT STARTED:**
1. **A regression pass over ALL shipped video profiles** (operator asked for it).
   Tonight proved ONE of the three shipping `wan_ti2v` profiles
   (`otr_w45_wan_ti2v`). `otr_g4_wan_ti2v` and especially `otr_upscale_ship`
   -- which stacks a spandrel stage on the same engine -- are unexercised
   against untiled decode. The sweep should also cover the ltx25, humo, minimax
   and fastwan lanes, none of which this session touched.
2. **`QUALIFIED_COST_ROWS` is an empty frozenset**, so `cost_row_may_refuse()`
   returns False for every engine and the VRAM budget gate **cannot refuse
   anything**. Pre-existing and orthogonal to tiling, but it means a correct
   recipe is the only thing between a too-large render and an OOM.
3. **The still-canvas review remains OPEN** (1080p vs the 1472x832 default) --
   unchanged by this session.


## THE PRIOR CURRENT STEP, 2026-08-21 07:42

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
