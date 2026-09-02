# PROBLEM STATEMENT -- the 16 GB high-end image engine (2026-09-01)

Operator's ask, in his words: *"assuming Flux2 [Klein] as the low-end tier we need a good
16 GB high-end tier that is most compatible with our system ... and preferably ungated,
auto-download."* And his eye: *"I kind of like Flux better; Z-Image always looked like
bloody faces and a bit of patchiness."*

## The decision to make

Which local image engine is the shipped default for the 16 GB NVIDIA saved dropdown set
(the canonical workflow), given that Klein 4B Q4 GGUF is already ruled the low-VRAM default
(Mac, AMD, small NVIDIA cards).

Acceptance criteria, from the ask:
1. Ungated repo, auto-downloadable through the pack (row 1.13 of the plan).
2. A license an open-source pack can ship as a DEFAULT.
3. Runs on the whole 16 GB class: Ampere (3090, A4500), Ada (4090), Blackwell (5080),
   with stock ComfyUI loaders and no extra node pack.
4. Looks at least as good as the alternatives on real episode stills, judged as radio
   drama, with faces the operator does not read as bloody or patchy.

## What is measured

### Compatibility and licensing (repos read 2026-09-01)

| engine | repo gated | license | download | loaders | runs on |
|---|---|---|---|---|---|
| z_image_turbo (6B) | no | Apache-2.0 | 12.3 GB bf16, 6.2 GB int8, 4.5 GB nvfp4, + 8 GB encoder | stock | Blackwell (5080, 45 episodes) and Ampere (A4500 picked bf16 itself) |
| flux_gen1 = FLUX.1-dev fp8 (12B) | no (Comfy-Org repackage) | BFL non-commercial; `commercial_clean = False` in the pack | 17.3 GB single checkpoint | stock | Ada/Blackwell fp8 natively; Ampere upcasts, slower |
| lumina_image (2.6B) | no | Apache-2.0 | 5.2 GB + 5.2 GB Gemma-2 encoder | stock, but needs an env var (no auto-discovery) | any 16 GB card |
| flux2_klein (4B) | no | Apache-2.0 | 2.6 GB Q4 GGUF or 7.75 GB bf16, + 8 GB encoder | GGUF pack (Q4) or stock (bf16) | everything; ruled the low tier |
| ideogram4_local | weights do not ship | non-commercial | Blackwell nvfp4 only | | out |

Only Z-Image, Lumina and Klein pass criteria 1 to 3. Flux.1-dev passes 1 and 3 and fails 2:
shipping it as the default would put every user of an Apache pack under BFL's terms, which
is why the pack coded it opt-in.

### Quality: a blind jury on real stills (`docs/ship-audit-2026-09-01/image-jury/`)

37 stills sampled from published episodes (2026-08-15 to 2026-09-01), one per episode per
kind, engines hidden, shuffled; five judges (radio producer, art director, defect hunter,
phone viewer, cinematographer), each scoring every image 1-10 on period, figure,
composition, artifacts, overall. Engines revealed only in the aggregate.

| engine | n | period | figure | composition | artifacts | overall |
|---|---|---|---|---|---|---|
| flux2_klein | 45 | 5.98 | 7.07 | 7.64 | 6.49 | 6.09 |
| z_image_turbo | 45 | 6.13 | 6.62 | 7.49 | 7.33 | 5.93 |
| flux_gen1 | 45 | 5.51 | 6.96 | 7.56 | 7.09 | 5.87 |
| ideogram4_local | 20 | 5.95 | 6.00 | 6.30 | 6.00 | 5.15 |
| lumina_image | 30 | 6.00 | 5.77 | 6.17 | 5.00 | 4.00 |

What that says and does not say:
* The top three are within 0.22 of a point overall. The two taste judges (viewer,
  cinematographer) put Flux first, which agrees with the operator's eye; the defect hunter
  put Z-Image first on artifacts; Klein wins on faces.
* Judges could not reliably tell the engines apart from pixels (cluster purity 0.32 to
  0.59), so the engines are closer than the operator's impression suggests.
* The one defect every engine shares is garbled text on dials, banners and title cards.
  Lumina's text cards were unusable, which is why it ranks last; text belongs to the
  `still_word` lane, not to the scene engine.
* Lumina also has the lowest artifact score; it is not a candidate.

### The "bloody faces" observation has a measured cause candidate

Every Z-Image still the 5080 has ever minted used the nvfp4 weight (the only Z-Image file
on the box), and the pack runs Z-Image-Turbo at `cfg 2.0` with a live negative
(`nodes/_otr_image_engines/z_image_turbo.py`, "keeps the negative live") although the model
is guidance-distilled for cfg 1.0. A same-prompt, same-seed probe through the pack's own
engine (`image-jury/zab_nvfp4_cfg2.0.png` vs `zab_nvfp4_cfg1.0.png`):

* cfg 2.0 (shipped): higher contrast, redder and pinker skin, heavier lip saturation,
  crushed shadows -- the look the operator describes.
* cfg 1.0: natural skin, cleaner period grade, more scene detail, and it rendered in half
  the time (7 s vs 13 s for 8 steps).

The bf16 cells (`zab_bf16_cfg2.0.png`, `zab_bf16_cfg1.0.png`, 22 s and 18 s) settle which
axis it is: at the same cfg the bf16 and nvfp4 frames are near-identical in composition,
skin tone and grade, while the cfg axis moves both precisions the same way -- cfg 2.0 is
redder, harder and more crushed in both weights; cfg 1.0 is natural in both. **The
"bloody faces" look tracks the shipped cfg 2.0 recipe, not the 4-bit weight.** One seed and
one prompt, so it is a strong lead rather than a proof; the three-prompt operator eyeball
in "Next actions" is what promotes it. It also means the nvfp4 file (4.5 GB) stays a
legitimate Blackwell choice and the 16 GB non-Blackwell path can use bf16 without a
quality argument against it.

## Options

A. **Z-Image-Turbo stays the 16 GB default, with the recipe fixed.** Passes every hard
   criterion; the cfg finding suggests its worst trait is a knob, not the model. Needs the
   A/B confirmed on real episode prompts (both cfg values, nvfp4 and bf16) and a negative
   strategy for cfg 1.0 (the negative prompt becomes inert, as it already is for Flux).
B. **Flux.1-dev as the default.** Best on taste, fails the license criterion for a shipped
   default. Keep as the documented "install it yourself" upgrade, like the cloning TTS
   engines.
C. **Klein 4B bf16 everywhere (one engine for all tiers).** Passes every criterion,
   already ruled for the low tier, best faces on the jury; the bf16 file loads through
   stock loaders so the 16 GB tier would not need the GGUF pack. Costs the "high-end" label:
   it is the same 4B model the small cards run.
D. **Retire the notion of a separate 16 GB image tier** and let video engines carry the
   16 GB distinction (they do; stills are conditioning). C without the branding.

## Next actions (in order)

1. Finish the A/B: bf16 at cfg 2.0 and 1.0, same seed; then the same four cells on three
   real episode prompts (announcer portrait, character portrait, scene beat). Operator
   eyeballs the eight frames. If cfg 1.0 wins, that is a one-line recipe change behind the
   existing `OTR_ZIMAGE_CFG` knob and a profile default, reviewed as a recipe change.
2. Decide A vs C (or D) on the eyeball. Record it in `docs/OTR_STANDING_RULINGS.md` next to
   the Klein ruling, and update `config/machine_classes.json` `image` for the 16gb class.
3. Flux.1-dev: keep in the dropdown, keep `commercial_clean = False`, document as an
   upgrade with the license note. No default anywhere.
4. Text on stills: route every text-bearing card through `still_word`, never through a
   scene engine; this is the defect the jury found on all five engines.

Receipts: `docs/ship-audit-2026-09-01/image-jury/` (jury result, sample map, the four A/B
frames `zab_{nvfp4,bf16}_cfg{2.0,1.0}.png`).

Related ruling the same day: Mac and AMD ship IMAGES ONLY (no video-diffusion engine
advertised there until one publishes an episode on that hardware); recorded in
`docs/OTR_STANDING_RULINGS.md`.
