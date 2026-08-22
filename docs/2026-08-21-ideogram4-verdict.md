# Ideogram 4 as a local still engine -- VERDICT

**RESULT: NO. Ideogram 4 is not adopted for OTR's `still_word` path. The shipped
`z_image_turbo` stands. No production code was written and
`workflows/otr_canonical.json` is untouched.**

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

## The single reopening condition

**Prose prompts on `announcer_visual` cards only**, and only with:
1. a matched canonical `z_image_turbo` comparison,
2. the refusal detector (`min > 50` AND `std < 15`) armed before any still is
   accepted, and
3. the active attention backend recorded.

Anything less does not reopen this.
