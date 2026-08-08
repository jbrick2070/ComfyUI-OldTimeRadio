# PROBLEM STATEMENT: does MiniMax H3 belong in the OTR video dropdown?

**Date:** 2026-08-03. **Status:** undecided, nothing implemented, nothing wired.
**Author:** Claude (Opus 5), grounded against the real Windows files.
**Round:** r1 (ideas / arc). No adapter code, no workflow JSON, no downloads.

MiniMax H3 shipped open weights with day-0 ComfyUI support on 2026-08-03. Three
agent windows produced integration specs for it, each inheriting the previous
one's fabrications. This document asks the prior question those specs skipped:
**measured against the engines already in the dropdown, what does H3 actually
add?**

A tier-based framing was drafted and CUT (operator, 2026-08-03: tiers are not
active). The comparison that matters is engine-versus-engine in the live
dropdown, against the generated `docs/ENGINE_MATRIX.md`.

---

## 1. THE UNIT OF WORK -- there is no single beat length

Rev 2 of this document asserted "a beat is 442 frames = 17.68 s." **That was
wrong.** 442 is an EXAMPLE, used as the worked case in the 08-02 maths doc and
named as such in `coverage_plan.py:114` ("`ltx_audio_in` at a 442-frame beat").

The canonical length unit is `target_words` on `OTR_LedgerScriptWriter`
(line 66: "canonical length unit; radio ~140 wpm";
`WORDS_PER_MINUTE_ESTIMATE = 140` at line 234). The widget documents its own
scale at line 2443: **1400 = 10 min, 2100 = 15 min, 3500 = 25 min**, default
350, clamped to a minimum of 5.

So episode length is an operator widget, and beat length falls out of it.
**[ASSUMPTION -- verify]** at the `beats: 40` in the shipping profiles:

| `target_words` | episode | beat length @ 40 beats |
|---|---|---|
| 350 (default) | 2.5 min | **3.75 s** |
| 1400 | 10 min | 15.0 s |
| 2100 | 15 min | 22.5 s |
| 3500 | 25 min | 37.5 s |

This is the ladder rev 2 flattened into one number, and it changes the answer:
**at the default 350 words, a beat is 3.75 s -- below MiniMax H3's 4 s floor.
H3 could not render the default episode at all.**

Every prior spec, and rev 1 of this document, instead reasoned from
`render.frame_budget: 25` and concluded the beat was one second. That field is
**diagnostic only** (2026-08-02 EXECUTIVE DECISION, Phase 1 item 4).

---

## 2. THE COMPARISON -- H3 against the local engines in the dropdown

From `docs/ENGINE_MATRIX.md` (generated from the live registry) and section 1 of
the 08-02 FINAL maths doc. Segment counts are @442 frames.

### 2A. Every LOCAL dropdown name, by per-call duration window

Read from `docs/ENGINE_MATRIX.md`, generated from the live registry.

| dropdown name | family | contract | window | still policy |
|---|---|---|---|---|
| `ltx_8gb` | image_to_video | 9-161 / q8 | 0.36-6.44 s | strict_first_frame |
| **`ltx_audio_in`** | **audio_conditioned_video** | 9-497 / q8 | **0.36-19.88 s** | **soft_reference** |
| `fastwan_8gb` | image_to_video | 17-177 / q4 | 0.68-7.08 s | strict_first_frame |
| `wan_ti2v` | image_to_video | 17-177 / q4 | 0.68-7.08 s | strict_first_frame |
| `wan_i2v` | image_to_video | 33-177 / q4 | 1.32-7.08 s | strict_first_frame |
| `humo` | audio_driven_face | 33-97 / q4 | 1.32-3.88 s | soft_reference |
| `ltx_video` | text_to_video | 169-169 / q8 | 6.76 s fixed | strict_first_frame |
| `mesh_stage` | image_to_video | 1.. no ceiling | unbounded | none |
| `still_flat` / `still_motion` / `still_pan` / `still_word` | static | 1.. no ceiling | unbounded | none |
| `viz_camera` / `viz_green` / `viz_mxc_cpu` / `viz_mxc_mandala` | abstract | 1.. no ceiling | unbounded | none |
| **`minimax_h3` (proposed)** | **audio_conditioned_video** | 17k+5 @ 24 fps | **4-15 s** | (would be soft_reference) |

### 2B. What that table says, by dropdown name

**1. `ltx_audio_in` strictly contains `minimax_h3`.** Same family
(`audio_conditioned_video`), same still policy (`soft_reference`), both local.
`ltx_audio_in` starts 3.64 s lower and ends 4.88 s higher. There is no beat
length H3 can serve that `ltx_audio_in` cannot. **This is the whole finding.**

**2. H3's 4 s floor sits above the ENTIRE window of `humo` (1.32-3.88 s).**
Every legal `humo` length is illegal for H3. H3 cannot serve the face lane at
current caps -- not "worse at," cannot.

**3. H3's floor is above the floor of every local motion engine.** `ltx_8gb`
0.36, `fastwan_8gb` 0.68, `wan_ti2v` 0.68, `wan_i2v` 1.32, `ltx_audio_in` 0.36.
Any beat shorter than 4 s -- including the 3.75 s default from section 1 --
is renderable by all of them and by none of H3.

**4. H3's 15 s ceiling IS the second-longest local window**, more than double
`wan_ti2v` / `fastwan_8gb` / `wan_i2v` (7.08 s) and `ltx_video` (6.76 s). This
is H3's one genuine strength -- but it can only be reached by displacing
`ltx_audio_in`, which is already longer.

**So the seam argument dies specifically.** The claim that a long single-pass
clip removes the seams behind identity drift, pose snapback, and the
"ten cuts is a stutter" defect is an argument for **`ltx_audio_in`** -- already
in the dropdown, already local, already audio-conditioned, already
`soft_reference`. H3 loses that argument to an incumbent.

Note also: 4-15 s is the CLOUD envelope (`cloud_seedance_2` 4-15 s,
`cloud_wan_i2v_audio` 2-15 s). H3 would be a local engine carrying
cloud-shaped limits at 42.5 GB. [The numbers coincide; the causes differ --
provider billing versus training regime -- so read this as an observation,
not an argument.]

---

## 3. THE ONE ARGUMENT THAT SURVIVES, AND WHY IT IS WEAKER THAN IT LOOKS

H3-Base-Ref2VA accepts up to 9 reference images and locks character, style, and
camera. Section 3 of the 08-02 FINAL doc states: **"There is no identity
conditioning anywhere in the local lanes; `reference_images` exists only on the
cloud engines."** So H3 would be the first local engine with true
reference-image identity conditioning. That is a real gap.

But the same section establishes the gap does not currently hurt:

> the four `humo` variants are JUMP but `audio_driven_face` consumes no scene
> still, so all segments share ONE portrait and **identity holds by
> construction**. What resets on humo is POSE, not identity.

And: **"This is the still-continuity defect, and it lives in cloud, not local"**
(F10). So H3's identity conditioning would solve a problem the LOCAL lanes do
not have, on the LOCAL side, at 42.5 GB. The defect it addresses is on cloud
engines H3 cannot replace.

Residual honest case: pose reset is still a real local defect, and H3's
reference conditioning plus a longer window might address pose continuity where
a shared portrait does not. That is a quality hypothesis, not a measured claim.

---

## 4. THE QUEUE -- twelve open fixes on engines we already ship

`docs/2026-08-02-FINAL-all-engine-maths-and-stills.md` section 4 lists F1-F12
against the CURRENT dropdown. Three matter to this decision:

- **F1 (blocking, safety):** `assert_frame_affordable`
  (`motion_common.py:339`) **has ZERO call sites**, while `PLANNING_CAP_ENGINES`
  holds `wan_ti2v`, `fastwan_8gb`, `ltx_8gb`. Every coverage-planned segment on
  three shipping engines renders with **no preflight VRAM check**. The
  2026-08-01 ruling named this a prerequisite -- "U2 must not ship before this"
  -- and U2 shipped anyway.
- **F2:** the HuMo 49-frame cap cites `docs/2026-06-27-humo-bakeoff`, **a
  document that does not exist in this repo.**
- **F7:** the HuMo lip-sync onset pad (BUG-LOCAL-102, audio leads lips
  100-200 ms) was specified and never built. Audible on every episode the face
  engine has produced.

**Adding a thirteenth engine while an unwired admission guard lets three
shipping engines render without a VRAM check is the wrong order of work.**
An in-process CUDA OOM corrupts the allocator.

---

## 5. H3 FACTS, VERIFIED (retained from the fact-check; sources named)

- Smallest usable set = **42.5 GB**: DiT `fl2va_pruned_int8_convrot` 19.53 GiB +
  `qwen3vl_32b_nvfp4_awq` 14.61 GiB + video VAE 4.85 GiB + audio VAE 0.56 GiB.
  Read from the HF API file listing. ~~No GGUF, no distill exists.~~
  **FALSE, corrected 2026-08-07:** `vram-recipe-lab` documents a GGUF stack --
  `MiniMax-H3-FL2VA-Q3_K_M.gguf` (15.58 GB) + `qwen3vl-32B-MiniMax-H3-Q2_K.gguf`
  (8.49 GB), 29.89 GiB total, needing `ComfyUI-GGUF` + `ComfyUI-Spectrum-MiniMax-H3`.
  Note also that 42.5 GB is a DISK figure and this document used it as a VRAM
  argument; see the 08-07 update in section 6.
- 33.1B params. Local envelope: short edge 768, cap 768x1344 / 32; 24 fps;
  4-15 s on a 17k+5 grid. **2K is not available locally** (needs
  `H3-Regenerate-2K`, not repacked).
- `H3-Context-IR`, which the model card calls "critical to the quality of the
  final output," is a **HOSTED** service. Offline-first scope question.
- License `minimax-h3-community-license-agreement`. Not OSI. Geographic
  exclusions reported. **RESOLVED 2026-08-07 by written authorization:** an
  email from `API@minimax.io` to `info@blueberrykaleyogabooks.com`, dated
  2026-08-07 5:54 PM, subject "MiniMax H3 License Authorization", states that
  MiniMax authorizes Blueberry Kale Yoga Books to use MiniMax H3 and MiniMax H3
  Works. The grant is **conditioned on continued compliance with the commitments
  and representations set forth in the operator's request email**, which is not
  filed in this repo -- file it beside this document, because it is the
  condition the grant rests on. [Two notes on the artifact itself: the
  authorization spells the licensee "Blueberrky Kale Yoga Books" with a
  transposed extra `r`, which does not match the domain it was sent to; and it
  authorizes use without restating the geographic terms. If either matters
  later, ask MiniMax for a corrected copy while the thread is warm.]
- **`Comfy-Org/ComfyUI#15263`:** global `--use-sage-attention` routes H3's DiT
  into Sage's int8 QK path -> **pure noise, video and audio, no error**. Our
  `otr_w45_wan_ti2v.json:84` sets `"sage_attention": true` and
  `build_variants.py:198` propagates it; the headless WAN lane is already safe
  (`eng_wan_i2v.py:15`), and `_otr_soak_server_launch.cmd` already has a
  "LTX -- Sage-free boot lane (BUG-070)" pattern to copy.
- The only shipped third-party integration (`HM-RunningHub/ComfyUI_RH_MinMaxH3`,
  41 stars) targets **single 24 GB GPUs** with INT8 *and* layerwise offload
  already applied. This box is 16 GB / 14.5 GB target, 63.4 GB system RAM.

**NOT verified:** the "3060 ~5 s in ~9 min" benchmark (uncited; its RAM figure
drifted 64 -> 32 GB across retellings; one agent stamped it "CONFIRMED"). Peak
VRAM and wall clock on this box: zero measurements. Comfy's blog states no
hardware requirement at all, and puts "42.5 GB" and "runs on an RTX 3060" in the
same post unreconciled.

---

## 6. THE DECISION AS IT NOW STANDS

On the evidence, H3 is **dominated by an engine already in the dropdown** for
the job it was proposed to do: `ltx_audio_in` is local, audio-conditioned,
single-segment across a full 442-frame beat, integrated, canvas-declared, and
already carries `soft_reference`. H3 is heavier (42.5 GB), unintegrated,
license-unread, carries a silent-noise landmine, cannot cover a beat in one
pass, and its natural envelope is the cloud providers' rather than the local
lanes'.

The provisional answer is **no -- not now, and not for this role.**

**2026-08-07 update -- the license cleared, and a VRAM RECIPE EXISTS that this
document never consulted.** Two things landed on 08-07, and the second is the
larger one.

**1. Licensing: answered.** Written authorization from MiniMax, same date, per
section 5.

**2. `vram-recipe-lab` -- and the withdrawal of the 42.5 GB objection.**
`C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\` holds six authored H3
recipes (`h3_{t2v,i2v,r2v}_{low,best}.json`) plus `docs/H3_VRAM_BUDGET.md`,
`docs/H3_DOWNLOAD_DECISION.md` and `research/H3_COMMUNITY_INTEL.md`. **This
document's 42.5 GB argument is a CATEGORY ERROR and is withdrawn**: 42.5 GB is
the DISK footprint, never a residency claim. The recipe pairs layerwise DiT
offload (32 blocks swapped to host RAM) with encode-then-unload on the text
encoder, so the stack lives in the 63.4 GB of system RAM and only the working
set is resident. `h3_i2v_low` -- 864x480, 124 frames, 24 fps, 20 steps,
`vram_ceiling_gb: 14.5` -- carries an EXTERNAL-REPORTED peak of **7.4-7.6 GB**
(Tomiigo, Linux, GPU capped to 8,188 MiB). Section 5's "no GGUF" is also false.

**What the recipe does NOT touch:**
- **Duration.** The `17k+5` @ 24 fps grid is unchanged, so section 2B stands
  exactly as written: `ltx_audio_in` (0.36-19.88 s) still strictly contains
  H3's 4-15 s, `humo`'s whole window is still illegal for H3, and the recipe's
  own 5.16 s is still above the 3.75 s default beat.
- **Resolution.** The lane that fits is **864x480**, below the production
  canvas. Native 1344x768 measures **14.6-15.3 GB on an actual RTX 5080**
  (tnsor_works) -- over the 14.5 GiB gate before Windows WDDM adds its
  0.5-1.2 GB, and HF discussion #6 puts text encoding ALONE at 15,219 MiB.
- **Proof.** **Zero H3 weights exist on disk** (verified 2026-08-07: both
  `C:\ComfyUI-Models` and `ComfyUI\models` return nothing), every recipe is
  `blocked: true`, and the lab states it has never executed H3. Every number
  here is EXTERNAL-REPORTED, none of it measured on this box.

**And the cost the recipe pays for that VRAM, which this document never
costed:** layerwise swapping streams **~268 GB of NVMe reads per 5-second
generation**. At eight clips to an episode that is **~2.1 TB of read traffic
per episode**, on a laptop SSD.

**So the ruling is no longer "no, on VRAM grounds."** It is: the VRAM
objection is **answered on paper and unproven here**, and the duration
dominance is **untouched**. M3 (question 2) remains the hinge -- if 449 frames
does not fit under 14.5 GiB, `ltx_audio_in`'s single-segment advantage
evaporates and this reopens on the merits. The next step is an operator call on
`H3_DOWNLOAD_DECISION.md` (Path A 42.48 GiB vs Path B 29.89 GiB), which the
licensing clearance has now unblocked.

---

## 7. QUESTIONS FOR THE PANEL -- break this, do not bless it

1. **Is the `ltx_audio_in` comparison sound, or am I over-trusting a contract?**
   The matrix says `single`, `1: 449 -> 442, trim 0`, `9-497 / q8`. But its
   ENGINE_MATRIX row also flags **"MISSING: docs/2026-07-02-canonical-ia2v"** --
   the same missing-evidence pattern F2 catches on the HuMo cap. If
   `ltx_audio_in`'s 449 is an unqualified declaration rather than a measured
   capability, my dominance argument rests on a number as soft as the one it
   displaces. **This is the load-bearing question. Attack it first.**

2. **M3 measures exactly that.** The 08-02 ruling schedules LTX-2.3 at 449
   frames with a host-RAM trace because "offloading a 29-46 GB checkpoint can
   fail outside VRAM." If M3 shows 449 does NOT fit under 14.5 GiB, the local
   single-segment capability evaporates and H3's two-segment coverage stops
   being a disadvantage. Does the H3 decision therefore reduce entirely to
   M3's outcome, and should it simply be deferred until M3 runs?

3. **Is the pose-continuity hypothesis (section 3) worth anything?** Shared
   portraits already hold identity locally; pose reset remains. Is
   reference-conditioned generation plausibly better on pose, or is that
   wishful?

4. **Is there a role I have not considered?** I compared H3 only as a beat
   engine. Cold open, title card, a single hero shot, or the announcer lane are
   different jobs with different constraints. Is any of them a fit, or is that
   scope creep dressed as compromise?

5. **What did I get wrong this time?** Prior drafts of this document asserted a
   1-second beat (wrong -- diagnostic field), then a 2-4 second beat (wrong --
   that was the M2 ladder, not the beat), before the matrix showed 442 frames.
   Three specs before it each inherited the last one's errors. Assume this draft
   has its own. Find them.

---

## 8. OUT OF SCOPE

- No adapter code, no workflow JSON changes, no weight downloads.
- Nothing written to `GO_FORWARD_PLAN.md` or the roadmap (operator directive).
- No re-litigating F1-F12; they are ruled and queued elsewhere.
