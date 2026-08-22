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

## The one-line reason

**Ideogram 4 carries a content filter baked into its weights that
deterministically refuses Shakespeare, and it refuses by returning a grey card
reading "Image blocked by safety filter" that every guard OTR has would pass as
a success.**

## What was measured -- 8 live stills, two seeds, two prompt shapes

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

**1. The filter is in the WEIGHTS, and it cannot be turned off.** There is no
safety-filter code anywhere in the ComfyUI tree -- not in
`comfy_extras/nodes_ideogram4.py` (which contains only the scheduler), not in
`comfy/`, not anywhere. The model itself paints the refusal card. There is no
flag, node swap, or config that removes it. **Adopting this engine means
importing a content guardrail we cannot remove**, which is the precise thing
`CLAUDE.md` forbids: *"no violence or swearing guardrails, they just cause
problems."*

**2. It refuses the operator's own source material, deterministically.** The
blocked line is Macbeth, V.i. Two different seeds, same block. Not seed luck --
a property of the text. OTR's adaptation lanes carry the author's language as
written, so this is not an edge case for this pipeline; it is the main case.

**3. The refusal is indistinguishable from success by every existing guard.**
ComfyUI reports `SUCCESS`. `SaveImage` writes a real 1.2 MB PNG. Dimensions are
exactly right. The frame is not black, so the all-black detector proposed during
review would pass it. **A blocked card would ship into a published episode as a
title card reading "Image blocked by safety filter."** It is detectable -- a
blocked frame has `min=80` and `std=10.4` against `min=0`, `std=27-41` for a real
render -- but only by a detector written specifically for it, on a path where a
missing still is already a hard fail by operator directive.

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

**The spelling is excellent -- better than anything OTR has.** Every card that
rendered spelled its line perfectly, including the full three-line Macbeth
quotation. Against the defect this programme actually surfaced -- lane 4's card
rendering `TI2V` as `TIZV` in BOTH arms, and the 1080p Z-Image card duplicating
an eye-chart row at both seeds -- Ideogram is clearly stronger at the narrow task
of putting requested words on screen correctly.

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
* VRAM peak **15,230-15,843 MiB** on a 16,303 MiB card -- **above the 14.5 GiB
  project target**, ~480 MiB headroom, no OOM in 8 renders. ComfyUI stages these
  models dynamically, so the naive 17.29 GB arithmetic never materialises.
* The second expert costs **+595 MiB and zero time** (45.1s vs 45.2s).
* ~45s per 1472x832 still at 20 steps.

## Licence position, unchanged by this verdict

`docs/IDEOGRAM4_LICENSE_ATTESTATION.md` stands as written: non-commercial public
licence, local only, weights never redistributed, no revenue. **The weights stay
on disk** -- they cost nothing idle and the announcer follow-up may want them.
No `commercial_clean = False` adapter was registered because no adapter was
written.

## What this cost, and what it bought

Three review rounds' worth of design work (r1 three lanes, r2 two lanes) plus
two probe rounds. **The arc caught four real errors before they became code** --
the driver's fallback design violated a dated no-fallbacks contract; the wiring
target was the wrong one of two; a missing `CAPABILITIES` row would have failed
the suite; and `free_after_use` does not evict models the way the driver claimed.
**Then the probe overturned the plan the arc had agreed on**, which is the case
for probing before coding rather than after.

**No production code was written, so nothing has to be unwound.** That is the
best possible shape for a NO.

## Bounds on this verdict

* One Shakespeare line, two seeds. Two blocks is not a refusal *rate*.
* No A/A null, and **no matched comparison against `z_image_turbo`** -- this
  says Ideogram misbehaves in two specific ways, not that z_image is better.
* Word cards only. Nothing here addresses ordinary scene stills.
* The announcer-only prose configuration is untested and is explicitly left open.
