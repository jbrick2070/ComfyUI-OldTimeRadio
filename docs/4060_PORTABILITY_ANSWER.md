# 4060 PORTABILITY: THE CURRENT ANSWER, AND THE HANDOFF TO THE 5080

**Written 2026-09-07 on the 4060 (MRKT), which owns the portability surface.**

This is the consolidated answer to *"what should an 8 GB card pick, and how much
does it cost."* It **SUPERSEDES** `docs/4060_DRILL_LOG.md` section
*"THE FRICTIONLESS-INSTALL ANSWER"* (line ~589, written 2026-08-29), which is
now stale in five specific ways -- listed in section 4 so the diary stays
readable as history without misleading anyone who reads it as advice.

---

## 1. THE THREE AUTO-DOWNLOAD CONFIGURATIONS -- NOTED, DELIBERATELY NOT SAVED

**OPERATOR DIRECTIVE, 2026-09-07:** *"do not save the 3 jsons just note them,
they will be saved later"* and *"I'm not saving a duplicate json until all
testing is done with all machines."*

**So these are SPECS, not files. No profile JSON has been added to
`config/profiles/`.** The repo already carries 117 profiles and the point of
waiting is to avoid minting three more that must be re-cut once Mac and AMD are
measured. One save, later, covers every machine.

All three share the same zero-friction core (section 2) and differ **only in the
video lane**, which is the axis that actually moves the download and the clock.

### Tier 1 -- FLOOR. Fastest, smallest, and the only one that needs no visual weights

| | |
|---|---|
| `announcer_visual` / `music_visual` / `character_visual` | `viz_mxc_cpu` |
| `announcer_image` / `music_image` / `character_image` | any row -- **not downloaded** (see section 3) |
| video weights | **0 GB** (`model_requirements: []`) |
| image weights | **0 GB** as of `f6fbb59` |
| total first-run download | **~12.2 GB** (the shared core only) |
| measured runtime | **16:16** -- the fastest episode this card has produced |
| status | **PROVEN as a complete combination** |

Evidence: `bloodstained_stairs_20260907_121450__arch__vmcp__none__koko__pubd__q354b__sa3_final.mp4`
The `none` in the image field is the episode reporting it minted no stills.

### Tier 2 -- STILLS. Pictures, no motion model

| | |
|---|---|
| the three `*_visual` slots | `still_pan` |
| the three `*_image` slots | `z_image_turbo` |
| video weights | **0 GB** (`still_pan` is a Ken-Burns lane over the still) |
| image weights | **19.27 GiB** (`z_image_turbo_bf16` 11.46 + `qwen_3_4b` 7.49 + `ae` 0.31) |
| total first-run download | **~31.5 GB** |
| status | **LANE PROVEN, COMBINATION NOT** -- see the gap below |

Evidence for the lane, four published episodes 2026-09-06:
`the_scalpels_cut_20260906_190603`, `hold_the_flicker_20260906_184452`,
`the_weight_of_a_hand_20260906_181701`, `breathing_the_impossible_air_20260906_174811`.

**THE HONEST GAP, and it is the one thing worth a run:** all four predate both
the current writer default and the licence-clean music bed. `still_pan` has
never been published together with `q354b` + `sa3`. Tiers 1 and 3 have been.
**One `still_pan` episode on the current defaults closes it** -- that is the
single cheapest piece of missing evidence in the whole set.

### Tier 3 -- MOTION. Real generated video

| | |
|---|---|
| the three `*_visual` slots | `ltx_8gb` |
| the three `*_image` slots | `z_image_turbo` |
| video weights | **15.02 GiB** (`ltxv-2b-0.9.8-distilled` 5.91 + `t5xxl_fp16` 9.12) |
| image weights | **19.27 GiB** |
| total first-run download | **~46.5 GB** |
| measured runtime | 27-42 min |
| status | **PROVEN as a complete combination** |

Evidence: `the_shrinking_list_20260907_105821__rfrc__lx8g__zimg__koko__orig__q354b__sa3_final.mp4`
and `the_far_shore_relay_20260907_101548__anim__lx8g__zimg__koko__news__q354b__sa3_final.mp4`.

---

## 2. THE SHARED ZERO-FRICTION CORE -- identical in all three

| slot | value | download | why this one |
|---|---|---|---|
| `creative_model` / `technical_model` | `Qwen/Qwen3.5-4B` | 8.68 GB | 2.99 GiB resident under NF4, 14.47 tok/s -- fastest and smallest row tested on this card. Apache-2.0, **ungated**, anonymous fetch. Now `DEFAULT_LLM`. |
| `char_voice_engine` / `announcer_voice_engine` | `kokoro` | ~0.31 GB | every episode this box has ever published used it |
| `music_engine` | `stable_audio_3` | 3.22 GiB | **the licence-clean choice.** `musicgen` is CC-BY-NC and cannot back a published episode |
| `voice_bank` | `kokoro_builtin` | 0 | ships with the pack |

**No HF token. No API key. No paid account. No extra node pack. No manual file
placement.** Every byte above is fetched by the pack itself on first Run.

**What this rules OUT and why, so nobody re-proposes them:**

* **`musicgen`** -- CC-BY-NC. It works, it is smaller, and it is the wrong
  answer for anything published. `sa3` exists precisely to replace it.
* **The 7 cloud/partner video engines** (`cloud_*`, `google_*`) -- zero bytes
  but an API key and a paid account, which is *maximum* friction under the
  operator's own definition, not minimum. Zero-download is not zero-friction.
* **`animatediff15_v3_*` (the old LOW default)** -- needs the
  `ComfyUI-AnimateDiff-Evolved` node pack (PBUG-09) plus manual SD1.5 + motion
  module + adapter placement. That is four hand steps, and hand steps are the
  thing being eliminated.
* **`gemma-4-12b-it`** -- the old shipped graph default; does not fit 8 GB
  (PBUG-13).

---

## 3. WHY TIER 1 IS NOW GENUINELY FREE -- `f6fbb59`, today

Until today Tier 1 was **not** cheap: selecting a `viz_*` lane still downloaded
the full `z_image_turbo` set, because the asset preflight added every
`*_image_model` slot to the download set unconditionally. A 4060 episode
published as `..._vmcp__none__...` -- image field `none` because not one still
was minted -- after fetching 19.27 GiB it could never use.

The video lane already prescribed the answer and three readers already honoured
it (`accepts_still = False`, `still_plan = ()`, the dropdown label
*"(audio-reactive, no scene image)"*, and the image-brief phase short-circuiting
to `{"objects": []}`). Only the preflight disagreed. It now borrows the same
predicate. **Nothing is hidden from any dropdown** -- all 12 image engines stay
listed and selectable and an empty slot still refuses; only the fetch changes.

Measured over all 116 profiles declaring visual role overrides: 30 have at least
one no-still lane and **19 drop their image weights entirely**, six of them
`shipping`, including the 16 GB `16gb_full`. See PBUG-20260907-03.

---

## 4. THE FIVE CORRECTIONS TO THE 2026-08-29 ANSWER

The diary's `THE FRICTIONLESS-INSTALL ANSWER` was honest when written and every
one of these rows has since been overturned **by this box's own published
episodes**. An operator following it today would pick a non-commercial music bed
and avoid two lanes that now work.

| that section says | the obs folder says now |
|---|---|
| LTX lane: *"never completed an episode on this card, two attempts"* | **5 published** (`lx8g`) |
| `z_image_turbo`: *"never completed an episode"* | **8 of 10 published** |
| *"`music_engine musicgen` -- stable_audio_3 ckpt is not on disk"* | **3 published on `sa3`** |
| HIGH writer *"UNPROVEN; I will not ship an unproven default"* | `q354b` published 4x and is now `DEFAULT_LLM` |
| writer row `google/gemma-4-E2B-it` | superseded by `Qwen/Qwen3.5-4B` |

The `llama_cpp_python` 0.3.35 `STATUS_ILLEGAL_INSTRUCTION` fault recorded there
is **still unresolved and still real** -- it was routed around by moving to a
Transformers-lane writer, not fixed. It stays open.

---

## 5. COVERAGE -- what this card has and has not proven

Decoded from the published filenames, which carry the engine shortcodes:

| dimension | proven | of | note |
|---|---|---|---|
| video | 3 | 33 | `viz_mxc_cpu`, `still_pan`, `ltx_8gb` |
| image | 1 | 12 | `z_image_turbo` (+ `none`, legitimately, on no-still lanes) |
| TTS | 1 | 7 | `kokoro` |
| music | 2 | -- | `sa3`, `musicgen` |
| LLM | 1 | -- | `q354b`; the only dimension with a real measured ranking |
| upscaler | 0 | -- | **and provenance is not recorded anywhere** -- no filename field, so no episode can be traced to its upscaler |

**This is deliberately not a gap to close by brute force.** Operator, 2026-09-07:
*"I'm not sure we need to regression test every combo at this stage."* Three
proven configurations that install clean beat 33 lanes with thin evidence. The
combinatorial sweep is explicitly NOT the plan.

---

## 6. HANDOFF TO THE 5080 (IDREAM)

**Pull first.** `git pull --rebase origin v2.0-alpha`. Today's 4060 commits
land in `nodes/` -- the 5080's own surface -- so a stale checkout that edits and
pushes will quietly revert them (CLAUDE.md 0B).

### What came from the 4060 that touches your files

| commit | file | what it does to YOUR box |
|---|---|---|
| `f6fbb59` | `nodes/_otr_visual_assets.py` | stops downloading image weights when the role's video lane declares `accepts_still=False`. **Measured: 19 of 116 profiles drop image weights, including the 16 GB `16gb_full`.** No profile gains a download or loses one it can use. |
| `e4b5dfe` | `nodes/_otr_visual_assets.py` | `+ Add Custom Model` now resolves through `custom_models_json` in the preflight instead of being hard-refused |

### What the 5080 is asked to do, in order

1. **Do NOT save the three configurations as profile JSONs yet.** Operator
   directive, section 1. They wait for Mac and AMD.
2. **Re-measure the three configurations on 16 GB** and record the deltas.
   Their whole purpose is to be the cross-machine baseline, and one card's
   numbers are not a portability claim.
3. **Status promotion stays yours** (CLAUDE.md 0B). Nothing here is promoted;
   Tier 2 must not be promoted at all until its combination gap (section 1) is
   closed by an actual run.
4. **Ask Comfy-Org about the Flagged status** on alpha.25/.26. Two hypotheses
   have been spent and refuted -- including the downloader hypothesis. Guessing
   a third is worse value than one question to them.
5. **The ~38 stale tests** pinning superseded decisions remain a discrete
   single-pass task. Zero user-facing defects; they encode old answers.

### What the 4060 keeps

The portability surface: this document, `docs/4060_DRILL_LOG.md`, and the
fresh-install path. The one open 4060 run worth doing is the single `still_pan`
episode on current defaults that closes Tier 2.

### Still entirely unproven, by anyone

**Mac** (no hardware) and **AMD** (no hardware). Note that `viz_mxc_cpu`
declares `cuda` *and* `cpu`/`mps`, so Tier 1's engine is the same engine those
machines would select -- proving it here is real evidence for them, and it is
the only such evidence obtainable without the hardware. It is not a substitute
for running it there.
