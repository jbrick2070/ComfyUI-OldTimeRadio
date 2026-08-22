# Ideogram 4 as a local still engine -- VERDICT

**RESULT: NO. Ideogram 4 is not adopted for OTR's `still_word` path. The shipped
`z_image_turbo` stands. No production code was written and
`workflows/otr_canonical.json` is untouched.**

> **ROUND 4 CORRECTION, 2026-08-21 (operator-directed: *"1080p 16x9 ... ideogram
> make it happen"*). The RESULT is unchanged; the REASON below was WRONG and is
> superseded. Ideogram 4 does not refuse this content. The refusal tracks the
> PROMPT SHAPE, and the JSON card shape renders every line that the prose shape
> refused -- at both canvases. See "ROUND 4" at the end of this file for the
> corrected finding and the real blocker.**

Decided 2026-08-21 by live measurement on the real box, before any adapter code
existed. Operator authorized the download ("ideo 4") and delegated the judgment
("you judge and put in the workflow the best one ... all autonomously while I
go"). This is that judgment, with its receipts.

Receipts: `docs/2026-08-21-ideogram4-local-still-engine/` (gitignored working
directory) -- `PROBE_FINDINGS.md`, `probe_ideogram4.py`, `probe_out/PROBE_all.json`,
`probe_out/PROBE_v2.json`, `WEIGHT_SHA256.txt`, `driver_anchor.md`,
`wiring_grounding.md`, `reference/image_ideogram4_t2i.json`.
Frames: `output/otr/episodes/_ideogram4_probe/` (8 stills, never `otr/obs`).

**THIS IS A PRE-BUILD SCREENING REJECTION, NOT A PRODUCTION DISQUALIFICATION.**
The probe submitted standalone API graphs
(`docs/2026-08-21-ideogram4-local-still-engine/probe_ideogram4.py`), **not**
`workflows/otr_canonical.json`, and published nothing to `otr/obs`. That path is
outside the two named bench exemptions in `CLAUDE.md` §0/§0A, so **no production
qualification occurred and none is claimed.** The screen was enough to stop the
build; it is not enough to make a statement about a shipped episode.

## The one-line reason

**The tested Macbeth card prompt repeatably produced a model-rendered refusal
card -- a grey frame reading "Image blocked by safety filter" -- at two seeds,
and no supported ComfyUI runtime control to disable it was found. A refusal
returns as a normal SUCCESS with a valid, non-black PNG, so it would pass the
dispatcher's generic pixel handoff unless an adapter added semantic refusal
detection.**

*(Corrected after the r4 review. An earlier draft said the filter is "baked into
its weights", that it "deterministically refuses Shakespeare", and that a
blocked card "would ship into a published episode". All three overstated the
evidence -- see "What this verdict does NOT establish" below.)*

## ROUND 3 -- the operator challenged the test, and the corrected test is far worse for Ideogram

**He was right that rounds 1-2 were unfair.** In his words: *"Your prompt should
be different for IDO. The IDO prompts are more of a text variety"* and
*"preferably pulled from the dialogue itself."* Both criticisms landed:

* Rounds 1-2 used an **invented** line, `"THE SIGNAL DIED AT MIDNIGHT"`, which
  is not what a card carries.
* Both shapes there were OTR's **scene** prompt -- backdrop, era tail, film
  grade, a full described background -- the wrong idiom for a typography model,
  and the obvious suspect for both the invented furniture and the refusal.

So round 3 re-ran it his way: **real card strings pulled from shipped episode
ledgers** and passed through the production reducer `_still_word_fit_card`, each
rendered in **two shapes** -- OTR's composed prompt (byte-identical to what the
**remote `ideo` arm** receives, since its `_prompt()` passes the composed prompt
through and the safety clause it once appended was retired 2026-08-05) and a
stripped **text-variety** prompt: the words, the letterform, a plain ground,
no scene at all.

| real card line (from a shipped episode) | composed | text-variety |
| :-- | :-: | :-: |
| "I've seen the weird sisters. Their prophecy unnerves me." | **BLOCKED** | **BLOCKED** |
| "Treason! My noble friends, see this man's intent!" | **BLOCKED** | **BLOCKED** |
| "Tonight we descend into the suffocating twilight of Pompeii." | **BLOCKED** | **BLOCKED** |

**6 of 6 blocked.** Classified by the measured signature, not by eye -- every
blocked frame sits at min 69-80 / std 10.1-10.7 against min 0 / std 22-41 for a
render.

**Three things this settles that rounds 1-2 could not:**

1. **The prompt shape is not the cause.** Stripping the scene entirely changed
   nothing. The text-variety idea was worth testing and it does not rescue this.
2. **It is not about extreme content.** The invented Macbeth line I used in
   round 1 was far more violent than anything OTR actually puts on a card. The
   real lines are mild -- witches, a prophecy, a warning, a documentary
   voiceover about Pompeii -- and they blocked anyway.
3. **It is not confined to the Shakespeare lanes.** The `scifi_news_pro`
   announcer line blocked in both shapes. This is not an adaptation-lane
   problem; it reaches ordinary narration.

**And it inverts the reading of rounds 1-2.** My invented line was the ONLY card
text that ever rendered. Every line drawn from the actual show was refused. The
original test was not too harsh on Ideogram -- **it was far too generous**, and
the real refusal rate on production text is much higher than the verdict first
implied. The NO stands, and it now stands on production content instead of on a
line I made up.

**Trigger still not isolated, and this is the remaining honest gap.** Six real
lines blocked and one synthetic line did not; that is not enough to say what the
filter keys on. It is enough to say Ideogram cannot be handed OTR's card text.

## What was measured in rounds 1-2 -- 8 live stills, two seeds, two prompt shapes

| shape | card | seed | outcome |
| :-- | :-- | --: | :-- |
| JSON v1 | quiet | 42 | invented "BAR" / "ART-DRAMA" |
| JSON v1 dual | quiet | 42 | invented "FRAUE" / "ARTSO-DRAMA," |
| **prose** | **quiet** | **42** | **clean -- the best word card this programme has produced** |
| JSON v1 | Macbeth | 42 | invented gold bars + "RHEICTHY, OENTHM8." |
| **prose** | **Macbeth** | **42** | **BLOCKED** |
| JSON v2 (guard last) | quiet | 20260821 | invented a logo + "MATGRIONI, GARLESLA," / "PAULT CODRRES" |
| JSON v2 (guard last) | Macbeth | 20260821 | invented "81" + "Pox/. 3U7:" |
| **prose** | **Macbeth** | **20260821** | **BLOCKED** |

> **JSON: 5 renders, 5 with invented text, 0 blocked.**
> **Prose: 3 renders, 1 clean, 2 blocked -- both on the same Shakespeare line.**

## The four findings that decide it

**1. No runtime control to disable the refusal was found.** There is no
safety-filter code in the tracked ComfyUI implementation -- not in
`comfy_extras/nodes_ideogram4.py` (which contains only the scheduler), not in
`comfy/`. The refusal card is rendered by the model rather than drawn by a
wrapper, and no flag, node swap or config exposes an off switch.
**What that establishes is "no supported disable control was found", not "the
filter is baked into the weights and cannot be removed"** -- the latter is a
stronger claim than an absence-of-code search can carry.

**2. The refusal reproduced at two seeds, but the prompt was not isolated.** The
blocked card is Macbeth, V.i, and the same prompt blocked at seed 42 and seed
20260821 -- so it is not seed luck. **But the Macbeth prompt differs from the
quiet prompt in the LINE, the BACKDROP and the GRADE all at once**
(`probe_ideogram4.py`, `MACBETH_BACKDROP` / `MACBETH_GRADE`). The probe
therefore did **not** isolate which of the three the filter reacted to. That is
the uncontrolled-second-variable trap of Bible `12.121`, and this verdict fell
into it while citing it elsewhere. A single-variable follow-up -- the Macbeth
line on the quiet backdrop -- is cheap and is the honest way to make the
stronger claim.

**What survives that correction is still decisive for the build:** a
production-shaped card carrying the operator's own source material was refused,
repeatably, with no way found to turn it off. Whatever the exact trigger,
adopting this engine imports refusal behaviour we do not control onto a lane
whose operator directive is *"no violence or swearing guardrails, they just
cause problems."*

**3. The refusal is indistinguishable from success by every generic guard.**
ComfyUI reports `SUCCESS`. `SaveImage` writes a real 1.2 MB PNG. Dimensions are
exactly right. The frame is not black, so the all-black detector proposed during
review would pass it. `nodes/otr_image_gen_dispatcher.py:504-526` accepts any
decoded array, so **a blocked card would pass the generic pixel handoff unless
an adapter added semantic refusal detection** -- stated conditionally because the
probe never ran the production path.

It IS detectable, and the signature is measured rather than guessed: a blocked
frame has `min=80` and `std=10.4` against `min=0` and `std=27-41` for a real
render. **Any future evaluation of this engine must implement that detector
(`min > 50` AND `std < 15`) before a single candidate still is accepted** -- on
a path where a missing still is already a hard fail by operator directive, an
undetected refusal is the worse outcome.

**4. The JSON escape hatch trades one defect for another.** JSON prompting avoids
every block, and it invents text on every single render -- logos, fake credits,
genre labels, catalogue numbers. A second attempt carrying OTR's own guard
verbatim as a trailing instruction (*"no other text, no logos, no captions, no
subtitle line, no studio name, no genre label"*) **added a logo and fake studio
credits anyway.** On `still_word` the words ARE the picture, so invented text is
a correctness defect, not decoration. This also reproduces, on local weights, the
exact failure the repo already recorded against the paid cloud arm:
*"Ideogram's signature card failure is an invented subtitle line."*

## What is genuinely good, and should not be lost

**The spelling is excellent.** The checkable result, stated without superlative:
**all six non-blocked frames spelled their requested target string correctly**,
including the full three-line Macbeth quotation, while **all five JSON frames
added unwanted text**. Against the defect this programme actually surfaced --
lane 4's card rendering `TI2V` as `TIZV` in BOTH arms, and the 1080p Z-Image card
duplicating an eye-chart row at both seeds -- that is a materially better result
at the narrow task of putting requested words on screen.

**It is NOT a claim that Ideogram beats `z_image_turbo`.** No matched comparison
was run. An earlier draft called one card "the best word card this programme has
produced"; that is withdrawn as unsupported.

**The one configuration still worth a look, later:** *prose prompts on
`announcer_visual` cards only.* Announcer cards carry billboard copy rather than
dramatic dialogue, and the prose quiet card was clean, correct and the best word
card produced in this programme. That is a narrow, cheap follow-up -- a matched
A/B against `z_image_turbo` on announcer cards -- and it is NOT claimed here.
Nothing in this probe compares Ideogram against the incumbent.

## Cost and performance, for the record

* Weights: 17.29 GB on disk (2 x 5.49 GB experts + 6.31 GB Qwen3-VL encoder).
  The `flux2-vae` was already installed and proved tensor-identical to
  Ideogram's copy (251 tensors, 336,185,492 payload bytes both; the 2,264-byte
  file difference is entirely header), so nothing was overwritten.
* VRAM peak **15,230-15,916 MiB** across all 8 runs on a 16,303 MiB card --
  **above the 14.5 GiB project target**, minimum observed headroom **387 MiB**,
  no OOM. (An earlier draft quoted 15,230-15,843 / ~480 MiB; that was the range
  from the first five runs only and never picked up round 2, whose three cells
  are the highest at 15,852 / 15,868 / 15,916. Corrected after the r4 review.)
  ComfyUI stages these
  models dynamically, so the naive 17.29 GB arithmetic never materialises.
* The second expert costs **+595 MiB and no measurable time difference in the
  one pair measured** (45.1s vs 45.2s -- one comparison, not a benchmark).
* ~45s per 1472x832 still at 20 steps.

## Licence position, unchanged by this verdict

`docs/IDEOGRAM4_LICENSE_ATTESTATION.md` stands as written: non-commercial public
licence, local only, weights never redistributed, no revenue. **The weights stay
on disk** -- they cost nothing idle and the announcer follow-up may want them.
No `commercial_clean = False` adapter was registered because no adapter was
written.

## The review roster, stated exactly

**This was NOT a completed four-round arc and must not be described as one.**

| round | lanes | what it did |
| :-- | :-- | :-- |
| r1 | Codex, Antigravity, Cursor | arc / approach |
| r2 | Codex, Antigravity | coding plan |
| r3 | **not run** | the rejection probe cancelled the adapter, so there was no wiring to review |
| r4 | Codex, Antigravity | reviewed this closure record |

The cursor lane returned empty on its first r1 attempt and produced a review on
the documented single retry. r2 deferred adapter-receipt transport to r3, which
therefore remains unresolved -- it cannot affect a rejected path.

**Errors caught before they became code:** the driver's fallback design violated
the dated no-fallbacks contract; the wiring target was the wrong one of two; a
missing `CAPABILITIES` row would have failed the suite; `free_after_use` does not
evict models as the driver claimed; and r4 caught a stale VRAM range plus three
overstatements in this very document. **No production code was written, so
nothing has to be unwound.**

## What this verdict does NOT establish

* **The trigger was not isolated.** The Macbeth prompt varied line, backdrop and
  grade together. One single-variable follow-up would settle it.
* **"Baked into the weights" is not proven** -- only that no supported runtime
  disable control was found in the tracked ComfyUI implementation.
* **No production path was exercised.** The probe used standalone API graphs,
  not `workflows/otr_canonical.json`, and published nothing to `otr/obs`.
* One Shakespeare line, two seeds. Two blocks is not a refusal *rate*.
* No A/A null, and **no matched comparison against `z_image_turbo`**.
* Word cards only; nothing here addresses ordinary scene stills.
* **SageAttention was NOT exercised.** The probe server logged *"Using pytorch
  attention"*, so r2's SageAttention verify-at-build item is still open. Any
  reopening must run the canonical profile's attention backend and record it.

## A DEFECT IN MY OWN PROBE, and why it does not rescue the verdict

**The operator spotted it: the official template ships PORTRAIT.** Its
`ResolutionSelector` defaults to `'9:16 (Portrait Widescreen)'`, so anyone
adopting Ideogram has to request landscape deliberately.

**And my `aspect_ratio` field was malformed.** I sent
`f"{CANVAS_W}:{CANVAS_H}"` -> `"1472:832"`, i.e. literal pixel dimensions, where
the caption schema wants a RATIO (`"W:H"`, e.g. `"16:9"`). That is wrong and it
is recorded here rather than quietly fixed.

**It cannot be the cause of the refusals, for two independent reasons.**
1. **The renders WERE landscape.** The canvas comes from
   `EmptyFlux2LatentImage` at 1472x832, not from the text field, and every
   returned frame is a wide card. Orientation was never portrait in this test.
2. **The prose arm carried no `aspect_ratio` field at all** and still blocked
   3 of 3 real production lines. A field that is absent cannot trigger a filter.

**So the NO stands, and this becomes a condition on any reopening:** send a
proper ratio, request landscape explicitly, and do not inherit the template's
portrait default.

## The single reopening condition

**Prose prompts on `announcer_visual` cards only**, and only with:
1. a matched canonical `z_image_turbo` comparison,
2. the refusal detector (`min > 50` AND `std < 15`) armed before any still is
   accepted, and
3. the active attention backend recorded.

Anything less does not reopen this.


---

# ROUND 4 -- the operator was right to push, and the stated reason was wrong

The operator asked for a 1080p 16:9 retest, having noticed that the official
template ships PORTRAIT (`'9:16 (Portrait Widescreen)'`) and that my
`aspect_ratio` field was malformed -- it carried literal pixels (`"1472:832"`)
where the caption schema wants a RATIO. Both observations were correct.

The retest cleared him of the smaller point and overturned the larger one.

## What was run

12 live renders, same weights, same seed 42, same dual-expert topology, on the
three REAL production ledger lines that had been recorded as refused -- two
Macbeth (`shakespeare`) and one ordinary `scifi_news_pro` narration. Two prompt
shapes x two canvases, with `aspect_ratio` correctly formed as `"16:9"`
throughout:

| prompt shape | 1472x832 | 1920x1088 (true 1080p 16:9) |
|---|---|---|
| prose card (`prose_card`) | **3/3 REFUSED** | **3/3 REFUSED** |
| JSON card (`json_card_v2`) | **3/3 RENDERED** | **3/3 RENDERED** |

Classified with the detector measured in round 1 (`min > 50 AND std < 15`).
Refusals sit at `min ~= 68-87, std ~= 9.9-10.7`; renders at
`min = 0.0-1.0, std ~= 27-41`. The separation is not marginal.

Receipts: `probe_out/PROBE_hd.json`, and the frames
`probe_hd_*` / `probe_hdiso_*` under `output/otr/episodes/_ideogram4_probe/`.
The `hdiso` arm exists specifically because the first HD run moved TWO variables
at once (canvas AND shape); it holds the shape at JSON and puts the canvas back
to 1472x832, which is what isolates the cause.

## The corrected finding

**Canvas and aspect ratio have NO effect on refusal. Prompt shape has a total
effect.** The prose arm refused at both canvases -- including the innocuous
`scifi_news_pro` line "Tonight we descend into the suffocating twilight of
Pompeii", carried over the QUIET art-deco studio backdrop with no castle, no
blood and no Shakespeare anywhere in the prompt. The JSON arm rendered all three
lines at both canvases.

So the earlier one-line reason -- that the model refuses this CONTENT -- does not
survive. It refused a prompt SHAPE. This is the second time this engine's
evidence has had to be corrected for an uncontrolled variable (Bible `12.121`),
and it is the reason Gate IG4.2 of `docs/IMAGE_GEN_PREFLIGHT.md` now requires a
refusal finding to name the single variable it isolated.

## The real blocker, which is structural

The JSON arm renders -- and the main card line is **spelled perfectly**, with
genuinely excellent typography: correct apostrophes, correct punctuation, clean
kerning, sensible two-line breaks. On the words themselves this model is better
than what ships.

But **3 of 3 JSON renders invented gibberish corner text**:

| line | invented text |
|---|---|
| "I've seen the weird sisters..." | `Io dhetors` / `Na expacly.` |
| "Treason! My noble friends..." | `NO MISCOS,` / `PATIO LALLK.` |
| "Tonight we descend..." | `(s: 16925)` / `PRTOPLEBIX.` |

`NO MISCOS` is a mangled rendering of the guard's own words, *"no logos"*. The
model is PAINTING THE `negative_instruction` FIELD ONTO THE CARD.

That traces to the topology, not to authoring, and it is why it cannot simply be
prompted away. The official Ideogram 4 graph wires
`CLIPTextEncode -> ConditioningZeroOut -> DualModelGuider.negative`: the negative
branch is the ZEROED POSITIVE. **There is no text-negative input in this
topology at all.** A "no other writing" guard therefore has nowhere to live
except inside the positive prompt -- where the prose shape obeys it and the JSON
shape renders it as decoration.

`json_card_v2` was the pre-registered experiment for exactly this, and its own
docstring set the criterion in advance: *"If v2 comes back clean, the adapter is
viable and the fault was mine. If it still invents, the schema itself is the
problem."* It carried OTR's guard verbatim, LAST, at top level, with the element
list cut to a single text element. It still invented on 3 of 3. By its own
stated test, the schema is the problem.

## Why the RESULT is still NO -- and what would change it

Both shapes fail, for different structural reasons:

* **prose** -- obeys the no-other-text guard, but refuses real production card
  lines (6/6 across two canvases);
* **JSON** -- renders those same lines beautifully, but paints invented text
  onto the one audience-readable surface `still_word` exists to produce.

A title card whose whole job is to show the script's words cannot ship with
invented words beside them. So `z_image_turbo` stands, unchanged.

**This is now a build question rather than a screening rejection**, and it is
the operator's call, not mine:

1. **A prompt adapter is architecturally legitimate** and he predicted it at the
   outset (*"may need special adapter"*). `docs/IMAGE_GEN_PREFLIGHT.md` IG5.1
   says the shared `compose_still_word_prompt` stays model-agnostic and the
   ENGINE owns its own shape adaptation -- so an Ideogram adapter converting the
   composed card into a schema of its own is allowed by contract.
2. **What it would have to solve is the invented text**, not the refusal. The
   untested lever is a JSON payload carrying NO guard field at all -- since the
   guard is what gets painted -- and relying on the single-element schema alone
   to suppress extra text. That is one 3-render probe, not a build.
3. **The prize is real.** Perfect spelling on a card line is exactly what
   `still_word` wants and is not free elsewhere.

Until that probe runs and comes back clean on 3 of 3, the answer stays NO.

---

# ROUND 5 -- the official schema found inside the template, run, and measured

The canonical template itself settled the shape question. Its visible top level
is two SUBGRAPHS, and the second -- "Ideogram4 Caption Prompt Template", node
114 -- carries the vendor's own magic-prompt system text: the authoritative
schema. **Exactly three top-level keys in order (`aspect_ratio`,
`high_level_description`, `compositional_deconstruction`), single-line minified
output, bbox normalized 0-1000 as `[y1, x1, y2, x2]`.** Our `json_card_v2` had
violated all three (five keys, `indent=2`, pixel bboxes with x2=1382 out of
range) -- and its foreign `negative_instruction` key is what the model painted
onto the card. The vendored reference is byte-identical to the installed
package copy (sha256 `87dba0e3...`), so this is the schema the checkpoint
actually ships with.

Two arms were run on the three real lines at 1920x1088, seed 42, dual-expert
(`--stage official`, `--stage restraint`):

| line | official schema | + restraint vocabulary |
|---|---|---|
| mb_treason | **CLEAN -- zero invented text** | **CLEAN** |
| mb_witches | 1 small period footer | 1 small period footer |
| sf_pompeii | 1 small period footer | 1 footer + a copyright mark |

Zero refusals in either arm. Main-line spelling perfect in all six; the
lettering fills the frame (the generous normalized bbox [240, 60, 760, 940] is
the operator's "take up more room" ask, delivered). The invented text is now
confined to small period-styled furniture (fake catalog numbers, a fake
copyright) in the BOTTOM MARGIN -- and the restraint arm proves the schema's
own populate off-switch ("minimal", "sparse", "the only text in the image")
does NOT remove it. It is line-dependent and stable per seed: treason renders
clean every time, the other two footer every time.

## The footer is trivially detectable -- measured, total separation

Bright-pixel mass in the bottom margin (luminance > 120, rows below y=0.82,
2% side crop), on all six cards:

| card | bright px | truth |
|---|---|---|
| official/restraint treason | **0 / 0** | clean |
| official/restraint witches | 1582 / 1228 | footer |
| official/restraint pompeii | 1090 / 1010 | footer |

Clean max = 0; defective min = 1010. The separation is total on this sample --
a footer detector in the adapter would be the same measured-statistics pattern
Bible 12.125 mandates for the refusal card, and a clean output demonstrably
exists for at least one line, so detect-and-reroll converges. A bottom-crop is
the cruder alternative; both are build choices.

## Where this leaves the verdict

NO still stands, but the entire residual case is ONE defect: a small invented
bottom-margin footer on a fraction of cards, detectable at zero cost with total
separation, in exchange for card typography that is plainly better than what
ships (perfect spelling, frame-filling type, real display faces). The build
that would flip it: an `ideogram4_local` adapter that (1) composes the official
three-key schema from the still_word card (legal under IMAGE_GEN_PREFLIGHT
IG5.1 -- the ENGINE owns its shape), (2) arms both measured detectors (refusal:
`min > 50 AND std < 15`; footer: bottom-margin bright mass > ~500), and
(3) re-rolls the seed on a footer hit. That is an operator decision to spend a
build, not a screening question -- the screening is complete.


---

# ROUND 6 -- the golden rule, and the first production cards

**Operator directive:** *"THE GOLDEN RULE: every video lane should work in any 3
slots"*, and the correction that forced it -- *"all the dialogue is on char
beats so you can't switch it. what good is it if the char beats are our CORE
DIALOGUE and it's not on ideo?"*

## The blocker was never the engine

A live leg with all three image slots on `ideogram4_local` failed on `c02` -- a
character PORTRAIT, not a word card. The first fix attempted here was to move
`character_image` to `z_image_turbo`; the operator rejected it correctly, because
the character beats ARE the dialogue and that dodge would have produced a green
run proving nothing.

The real defect was upstream and had been shipping for months:
**`still_word` has ALWAYS declared `StillPlanRow(kind="portrait",
required="never")`, and nothing read it.** `still_plan_helpers` says so in its
own docstring -- *"Nothing in this module reads the plan for production"*. So
every still_word episode minted a portrait per cast member that no consumer on
that lane loads. Free on any engine that draws faces, which is why it hid; fatal
on the first engine that refuses a person close-up.

Fix: `_portrait_free_roles_from_policy`, the fifth member of the lane-derived
role-set family that already carries aspect / kind / framing / composer into the
image phase. It reads each lane's OWN declaration rather than naming engines, so
a future lane is covered the day it declares its plan. Verified against real
render behaviour: `wan_ti2v` declares portrait `never` and truly does not use one
(the scene still overrides its init); `humo` declares `always` and correctly
keeps its portrait to drive a mouth. `docs/VIDEO_LANE_PREFLIGHT.md` G3.7.

## The result: 7 of 8 stills rendered, every dialogue card among them

Episode *"The Weight of the Grain"*, all three image slots on
`ideogram4_local`, all three video slots on `still_word`, gemma-4-12b, kokoro,
musicgen, no upscaler, 1 act, 2 characters, bank and style rolled.

* **6 of 6 dialogue word cards RENDERED**, plus the music opening card.
* The cards are genuinely good: elegant serif typography, correctly spelled,
  composited over real photographic scenes.
* **One refusal remains: `still_music_closing_001`** (min=78.0, std=10.6).

**That refusal is SEED-DEPENDENT, not content-dependent.** The music OPENING
card rendered from the same composer, the same episode title and the same prompt
shape; only the closing one refused. This is the same stochastic pattern the
footer showed in rounds 4-5, where one seed footered and three others did not.

## What remains

1. **The music-card refusal.** One card in eight, on a wordless abstract. Since
   the operator's standing ruling forbids re-rolls (*"I accept some errors, I
   don't want it burning extra GPU cycles"*), the honest options are to
   root-cause it in the prompt as every other refusal in this campaign was, or
   to accept it and let the episode fail loudly when it fires. **It currently
   fails the whole episode**, which is worse than the blemish the ruling was
   about -- that asymmetry is the open decision.
2. **The truncated card line.** `still_b001` reads *"...of a subterranean"* and
   stops. That is `_still_word_fit_card`'s deterministic reduction, working as
   designed and unrelated to Ideogram -- but on a card whose whole job is the
   words, a sentence cut mid-clause is worth a second look.
3. **The footer detector is not usable as an acceptance gate on production
   cards.** It flagged 5 of these 7 real stills; calibrated on dark test cards
   it cannot read pack-styled photographic backdrops. The operator's eye remains
   the gate, exactly as the engine's own docstring says.
