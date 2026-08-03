# M2 MEASUREMENT: HuMo ISOLATED RENDER-WINDOW peak falls as frames rise

> **CORRECTED 2026-08-02 after a kibitz r1 panel (Codex + Antigravity).**
> Four claims in the first version of this document were overstated and are
> withdrawn below. The numbers are unchanged; what they support is narrower.
>
> 1. **This is not a lifecycle peak.** `prepare()` (`eng_humo.py:490`) loads
>    every heavy handle, and `VramPeakProbe` only starts afterwards inside
>    `render_clip` (`eng_humo.py:811`). So this measures the RENDER WINDOW with
>    handles already resident -- not the full cold `prepare -> render ->
>    teardown` peak, and not any transient spike during loading. Every
>    "cold-lifecycle", ceiling-safety and admission implication is withdrawn.
> 2. **The coverage-splitting recommendation is WITHDRAWN entirely.** See below.
> 3. **The "1 in 331,000" statistic is withdrawn.** The four series are not
>    independent.
> 4. **"No orientation effect exists" is withdrawn** in favour of "no consistent
>    orientation difference was detected".

**16 cells: 49/65/81/97 frames x portrait/landscape x cold/warm. Every cold cell
preceded by a server restart.** Branch `v2.0-alpha`, HEAD `537bb9d3`, 2026-08-02.
Runner: `scripts/otr_humo_vram_ladder.py`. Raw rows: `tmp/_m2_humo_ladder.json`.

**MEASUREMENT ONLY.** This qualifies nothing. Any cap change must be re-proved
through `workflows/otr_canonical.json` (see the open ruling at the end).

## Why this was worth measuring at all

`eng_humo.py:106` caps the 14B tiers at 97 frames, and its own comment calls that
"a QUALITY-supported target, not a measured memory ceiling". It was never
measured: HuMo was the last heavy lane without a `VramPeakProbe` until one was
added on 2026-08-02, and **every HuMo `per_clip` ledger row before that date is
null** in `recipe / quant / use_lora / render_canvas / vram_peak_mb`. So the cap
was a reasoned bound with no data underneath it.

## The two results

### 1. Peak VRAM DECLINES monotonically as frame count rises

Incremental cost, `peak_used - pre_used`, in MB:

| frames | portrait cold | portrait warm | landscape cold | landscape warm |
|---:|---:|---:|---:|---:|
| 49 | 14,486 | 13,985 | 14,223 | 14,284 |
| 65 | 14,031 | 13,696 | 13,996 | 13,629 |
| 81 | 13,644 | 13,243 | 13,645 | 13,372 |
| 97 | **13,325** | **13,026** | **13,260** | **13,020** |
| **49 -> 97** | **-1,161** | **-959** | **-963** | **-1,264** |

**All four series decline monotonically across all four rungs.** The ordering is
consistent and the magnitude is large -- roughly a gigabyte from 49 to 97.

**The "1 in 331,000" claim from the first version is WITHDRAWN.** It assumed the
four series were independent, and they are not: the runner walks the rungs in
FIXED ASCENDING ORDER, cold and warm are paired on the same session, and both
orientations share one portrait, one WAV and one campaign order
(`otr_humo_vram_ladder.py`). Any influence that varies monotonically with
position in the run would print this exact pattern in all four series at once.
Restarting the server before every cold cell resets process state, and the
desktop baseline was verified flat, so the obvious confounds are excluded -- but
**a fixed-order design cannot separate "longer renders cost less" from "later
cells cost less"**, and that is what the claim needed.

The honest statement is: **an inverse ordering was observed, consistently, over
sixteen cells.** Establishing it as causal requires counterbalanced order --
descending and interleaved runs -- and repeated endpoint pairs.

Render TIME meanwhile scales linearly: 3.18, 3.17, 3.41, 3.46 s/frame at
49/65/81/97. **Compute tracks frames; memory does the opposite.**

### 2. The orientations match -- the prediction survives

Cold peak, portrait `humo` (480x832) vs landscape `humo_14B_169` (832x480), same
14B checkpoint:

    49 frames: 15,914 vs 15,540  ->  delta 374 MB
    65 frames: 15,365 vs 15,318  ->  delta  47 MB
    81 frames: 14,967 vs 14,966  ->  delta   1 MB
    97 frames: 14,647 vs 14,582  ->  delta  65 MB

The runner labels the 49 pair DIVERGENT against its 300 MB threshold. **That
label should not be believed, and the threshold is the thing that is wrong.**
Repeatability was measured directly: the same portrait 49-frame configuration
run three times gave 15,624 / 15,904 / 15,914 MB -- a **290 MB spread on
identical inputs**. A 374 MB gap is barely outside that, it is the shortest and
therefore noisiest render on the ladder, and the deltas converge to 1 MB by 81.

**Conclusion: no CONSISTENT orientation difference was detected in the range
tested.** That is weaker than "no effect exists", and deliberately so -- absence
of a detected difference across four rungs is not proof of absence, and the
290 MB repeatability figure comes from only three repeats, which is a weak
estimate of spread. What can be said is that the deltas are small, they do not
point consistently in one direction, and they converge to 1 MB at 81 frames.

Nothing here contradicts the 2026-08-02 change collapsing the
orientation-specific 49/177 rules into one shared cap, and nothing here is
strong enough to have justified it on its own.

## THE "CEILING BREACH" FRAMING IS WITHDRAWN (2026-08-02, second correction)

**Operator: "our recipes have been stable for a while, no OOM."** That
operational record outranks any inference drawn here, and it exposes a bad test.

The canonical run logs the reason:

    Model WAN21_HuMo prepared for dynamic VRAM loading. 16531MB Staged.

ComfyUI stages **16,531 MB against a 16,303 MB card** and streams weights in and
out on demand. Under a dynamic loader, machine-wide used-VRAM near capacity is
the allocator consuming headroom that exists -- the intended steady state -- not
demand pressing against a limit. A high peak is therefore NOT evidence of risk,
and the renders that produced these numbers all succeeded.

So **comparing an NVML machine-wide PEAK against the 14.5 GiB target is the
wrong comparison**, and the "OVER CEILING" column below should be read as "used
this much VRAM", never as "nearly failed". This is exactly the error
`docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md` recorded against itself --
a USED-VRAM figure substituted into a model meant to compare against FREE VRAM --
and it was repeated here after being quoted.

What survives: the ORDERING (shorter renders report higher peaks) and the
ORIENTATION comparison, both of which are internal to this dataset and do not
depend on any ceiling. What does not survive: any claim that a rung is unsafe,
close to OOM, or in need of a cap.

## Peak vs the 14.5 GiB target -- reported, not interpreted as risk

Twelve of sixteen cells exceed the 14,848 MB (14.5 GiB) target on machine-wide
peak. **All four that do not are at 97 frames.**

    49 frames: 15,540 - 15,942 MB   all four cells over
    65 frames: 15,255 - 15,365 MB   all four cells over
    81 frames: 14,871 - 14,999 MB   all four cells over
    97 frames: 14,582 - 14,652 MB   all four cells UNDER

**Every render succeeded, and the operator reports no OOM on these recipes over a
long period.** Read this table as "how much VRAM was in use", nothing more. The
word "riskiest" appeared here in an earlier draft and is withdrawn: under a
dynamic loader that stages more than the card holds, a higher peak does not mean
closer to failure.

The only durable statement is the ORDERING -- shorter renders report higher
peaks than longer ones, consistently.

## What this means for the cap, and for splitting

1. **The 97 cap is not a memory protection.** It bounds quality, exactly as the
   code comment says. Nothing in this data supports it as an OOM guard, and the
   configuration it forbids is the one with the most headroom.
2. **`FRAME_COST_MODEL`'s shape is refuted for HuMo.** `overhead + per_frame *
   frames` cannot describe a series whose per-frame term is NEGATIVE across
   sixteen cells. This is the fourth engine to contradict the linear model --
   after `fastwan_8gb` (flat at 17/49/81), tiled `ltx_8gb` (flat within 37 MB)
   and `ltx_audio_in` (flat across 79 production samples, 25 to 497 frames) --
   and the first where the coefficient is clearly the wrong SIGN rather than
   merely zero.
3. **THE COVERAGE-SPLITTING RECOMMENDATION IS WITHDRAWN.** The first version of
   this document said splitting should prefer LONGER segments because a
   97-frame segment costs ~1.2 GB less peak than a 49-frame one. **That does not
   follow from this data, because production does not render segments the way
   this ladder does.**

   `HuMoEngine.prepare()` states it in its own docstring: *"Load every heavy
   handle ONCE PER BEAT instead of once per segment"* (`eng_humo.py:490-491`).
   A production multi-segment beat runs under ONE `BeatSession`, reusing the
   handles loaded for the first segment and tearing down once at the end. Every
   cell in this ladder was a FRESH single-clip session with its own load and
   teardown. The two lifecycles are different, so a per-segment cost measured
   here cannot be added up to predict a multi-segment beat.

   Deciding split policy needs a canonical multi-segment comparison. Until then
   **no split-policy change is supported in either direction**, and the existing
   planner behaviour stands on its existing rationale.

## THE RECIPE IS NOT ON THE TABLE (operator directive 2026-08-02)

**"We spent a lot of time perfecting the recipes to look good and we can't lose
that."**

The render recipe -- steps, cfg, quant, LoRA choice and strength, canvas, and the
trained frame length -- was tuned by eye over a long time and is the expensive,
hard-to-recover part of this project. VRAM numbers are cheap to re-measure; a
recipe that made the picture good is not.

So nothing in this document licenses a recipe change:

* **"Peak falls as frames rise" is NOT a reason to raise the 97 cap.** 97 is the
  TRAINED length and a QUALITY bound. This measurement says the cap is not
  buying memory safety; it says nothing whatever about what the picture looks
  like above it, and the operator's directive is that quality wins.
* **The deferred no-LoRA control is a recipe change, not a control.** Dropping
  the distill LoRA requires raising `OTR_HUMO_STEPS` and `OTR_HUMO_CFG` off
  their tuned defaults, so it cannot be run casually "for attribution".
* **Every cell in this ladder ran the SHIPPED recipe unchanged**, which is what
  makes the numbers comparable to production in the first place.

A measurement that suggests a recipe could change gets REPORTED, and the
operator decides against the picture. It never gets acted on here.

## Mechanism: observed, not explained

I can show that it happens and cannot yet say why. The plausible account is that
ComfyUI's memory manager adapts to a larger latent by decoding in smaller tiles
or offloading more aggressively -- trading time for peak, which fits the mild
per-frame time increase from 3.18 to 3.46 s. **That is a hypothesis.** Settling
it needs per-phase peaks (text encode / audio encode / sample / VAE decode),
which `VramPeakProbe` does not currently separate.

## Provenance and honest limits

* Every cell is a real `render_driver.render_single()` dispatch --
  `assert_usable` -> `prepare` -> `render_clip` -> `canonicalize` -> `teardown` --
  so the probe fires exactly as in production. Not a stock-node bench graph.
* Controlled: one portrait (`c02_076151b33f2c.png`) and one conditioning WAV
  (`slice_97eea81f5e1e84fb.wav`, 7.08 s) across all sixteen cells.
* Cold means cold: the server was killed and rebooted before every cold cell.
* **The desktop baseline was verified stable**, because Chrome / Codex /
  Antigravity were open: cold `pre_used` held 1,317-1,334 MB across the whole
  run (17 MB spread), warm 1,624-1,658 MB. Background drift cannot explain a
  1,161 MB decline. The lone 1,428 MB reading is the first cell, carrying
  residue from a prior killed run.
* `nvidia-smi` cannot attribute VRAM per process on Windows WDDM without
  elevation, so the 1.3 GB baseline is not broken down by application.
* **Machine-wide vs incremental**: `VramPeakProbe` reports NVML machine-wide
  usage. Both numbers are recorded per cell precisely so the "phantom 15.9 GB"
  error of `docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md` cannot recur --
  a used-VRAM figure must never be substituted into a model that compares
  against FREE VRAM.
* Not tested: lengths above 97, the 1.7B tiers, other canvases, other
  checkpoints, and any recipe/quant/LoRA combination other than the shipped one.

## Two harness bugs found by running it, both fixed and worth remembering

* `peaks_since()` seeked a BYTE offset on a TEXT-mode handle. Windows CRLF
  translation landed it elsewhere, so a completed 164-second render was reported
  as a failed cell with no peak.
* A repeat cell submitted a byte-identical graph, and **ComfyUI answered from
  cache in about five seconds carrying the FIRST run's `elapsed_s`** -- a warm
  measurement that was silently a copy of the cold one. Repeats now vary
  `oom_index` (documented inert in `mode=single`), and any cell returning under
  20 s is marked as not having rendered.

## The reproducibility receipt this measurement still owes

Raw rows live in `tmp/_m2_humo_ladder.json`, and `tmp/` is swept. There is no
pinned digest of the submitted graph, no runner hash, and no model/config
identity manifest -- none of the receipt machinery section 0A requires of the
bench runners it does exempt. The aggregate tables above survive in this
document; the dataset does not.

This is the same debt `docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md`
recorded against itself, and it is recorded here for the same reason: a number
that lives only in a swept directory is one cleanup away from being the next
`docs/2026-06-27-humo-bakeoff`.

## Open ruling owed before this is used

The runner drives `scripts/_otr_single_engine_smoke.py`, which submits a
ONE-NODE graph rather than `workflows/otr_canonical.json`. That is outside
`CLAUDE.md` section 0 and outside the section 0A carve-out, which names only
`run_video_arm_bakeoff.py` and `run_wan_ti2v_bakeoff.py`. Section 0A was an
explicitly NARROW operator ruling and widening it is not a coder's call.

**This document is therefore a measurement, not a qualification.** No cap, tier
or profile may change on it until the operator either extends the carve-out or
the result is re-proved through the canonical workflow.
