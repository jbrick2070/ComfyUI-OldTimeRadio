# M2 MEASUREMENT: HuMo peak VRAM FALLS as frames rise, and the orientations match

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

**All four independent series decline monotonically across all four rungs.** One
series ordering by chance is 1 in 24; four independent series is about 1 in
331,000. This is not noise, and it is not merely flat -- it is a real inverse
relationship. Doubling the frame count buys back roughly a gigabyte of peak.

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

**Conclusion: no orientation effect exists in the range tested.** Equal pixels
and an equal token grid cost equal memory, to the limit of what this instrument
resolves. There is no tiling or kernel-selection divergence. The 2026-08-02
change collapsing the orientation-specific 49/177 rules into one shared cap was
correct.

## Only the 97-frame rung stays under the ceiling

Twelve of sixteen cells exceed the 14,848 MB (14.5 GiB) target on machine-wide
peak. **All four that do not are at 97 frames.**

    49 frames: 15,540 - 15,942 MB   all four cells over
    65 frames: 15,255 - 15,365 MB   all four cells over
    81 frames: 14,871 - 14,999 MB   all four cells over
    97 frames: 14,582 - 14,652 MB   all four cells UNDER

Every render succeeded. This is headroom, not failure. But the direction is the
point: **the cheapest rung is the riskiest one, and the capped rung is the
safest measured configuration.**

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
3. **Coverage splitting should prefer LONGER segments, not shorter ones.** A beat
   is split into segments each rendered independently, and peak is per segment.
   On this data a 97-frame segment costs ~1.2 GB LESS peak than a 49-frame one,
   so filling to the cap is cheaper per segment than splitting finer. The
   current planner already fills toward the cap -- **that behaviour is correct,
   but for the opposite reason to the one assumed.** Anyone later "reducing VRAM
   risk" by shortening segments would increase it.

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

## Open ruling owed before this is used

The runner drives `scripts/_otr_single_engine_smoke.py`, which submits a
ONE-NODE graph rather than `workflows/otr_canonical.json`. That is outside
`CLAUDE.md` section 0 and outside the section 0A carve-out, which names only
`run_video_arm_bakeoff.py` and `run_wan_ti2v_bakeoff.py`. Section 0A was an
explicitly NARROW operator ruling and widening it is not a coder's call.

**This document is therefore a measurement, not a qualification.** No cap, tier
or profile may change on it until the operator either extends the carve-out or
the result is re-proved through the canonical workflow.
