# PROBLEM STATEMENT -- `wan_ti2v` retains ~8 GB between shots and starves later beats

**Date:** 2026-08-09
**Branch / HEAD:** `v2.0-alpha` @ `abaafd16`
**Severity:** blocks a SHIPPING profile end-to-end on the reference box.
**Admission:** qualifies. This is a LIVE headless run that genuinely failed --
not a static reading, not an invented fixture.

---

## 1. THE ARTIFACT

Leg: `otr_canonical_api_run.py --profile otr_upscale_ship --words 45
--source-bank original`, prompt id `0bcb9557-eb1e-41c6-b838-99d8fd816b51`,
booted at HEAD `262dfa8f` on the 5080 (16303 MiB), box otherwise idle at
~2.0 GB before boot.

`RESULT FAIL`. Node **92** `OTR_VideoRenderBatch`:

```
engine wan_ti2v: static frame budget 65 (snapped 65) exceeds the cost-model's
affordable 19 frames (free=9617 MB, margin=0.85). NO silent resize -- lower the
frame_count widget, free VRAM, or pick a lighter engine.
```

The refusal itself is CORRECT behaviour and must not be "fixed" -- S4
platform-portability (2026-07-10) deliberately made the render refuse rather
than silently resize (`motion_common.py:332-336`). The bug is the VRAM state
that made the refusal necessary.

## 2. WHAT THE LOG ACTUALLY SHOWS

The pre-render cleanup is NOT the problem and should not be re-investigated:

```
[OTR video] pre-render residue free: ran=['unload_llm', '_unload_bark',
'gc.collect', 'soft_empty_cache', 'cuda.synchronize', 'cuda.empty_cache',
'cuda.ipc_collect'] failed=[] free_gb_after=14.3291015625
```

14.33 GB free entering the video phase. Everything upstream (writer LLM, bark,
indextts2, kokoro, stable_audio_3, z_image_turbo) released correctly.

The loss happens INSIDE the video phase, between shots:

```
[OTR video] BEAT shot_b001 assembled from 2 chain segment(s) -> 200 frame(s)
[OTR video] wan_ti2v VRAM render-phase peak 10874 MB / post 8206 MB
[OTR video] wan_ti2v VRAM render-phase peak 13134 MB / post 8190 MB
-> shot_b002 refused (needs 65, affordable 19, free 9617 MB)
```

**`peak 13134 / post 8190`.** The engine settles back to ~8.2 GB, not to the
~1.5 GB desktop baseline it started from. Two chained segments of a 200-frame
beat leave roughly 8 GB resident, and the third shot is then refused on
headroom it should have had.

## 3. WHY THIS IS WIDER THAN THE PROFILE THAT FOUND IT

`otr_upscale_ship` exists only to prove the item-8 upscale stage. It would be
easy to dismiss this as a test-profile artifact. It is not:

* `otr_upscale_ship` is a clone of `otr_w45_wan_ti2v`, and the two are
  IDENTICAL on every dimension that feeds the cost model -- `composite_w` 1920,
  `composite_h` 1080, `frame_budget` 25.
* `otr_w45_wan_ti2v` is a SHIPPING profile.
* The upscale stage plays no part in the failure: it lives on node **84**
  (`OTR_SilentComposite`), downstream of the node **92** that died. The leg
  never reached it.

So the honest claim is: **a shipping wan_ti2v profile cannot complete a 45-word
episode on the reference box when an early beat chains into a long
multi-segment shot.** The upscale profile merely happened to be the leg that
ran it.

## 4. WHAT THIS BLOCKS

* **Item 8 chip 4 is NOT discharged.** `spandrel_esrgan` on `cuda:0` remains
  unproven live, because node 92 dies before node 84.
* **The cache-invalidation proof owed from `088dabc8` is blocked with it** --
  same reason, same node ordering.
* Any wan_ti2v 45-word render on this box with a comparable beat topology.

## 5. RELATIONSHIP TO WORK ALREADY IN FLIGHT -- READ BEFORE STARTING

This is the same FAMILY as the operator's live VRAM investigation and must not
be forked away from it:

* `vram-recipe-lab` measured a 15.3 GB over-ceiling for wan_i2v (2026-08-08).
* A concurrent window has an uncommitted `render_canvas = (832, 480)` on
  `nodes/_otr_video_engines/eng_wan_i2v.py`, whose own comment identifies the
  1472x832 landscape fallthrough -- 3.07x the intended pixels -- as the likely
  cause of that measurement.

Different engine (`wan_i2v` vs `wan_ti2v`), same wall: video-path VRAM
headroom. **Whoever picks this up should check whether `wan_ti2v` has the
mirror-image defect** -- it took a canvas declaration on 2026-08-02, so its
canvas is probably fine, which would mean the retention is about model/VAE
residency between chained segments rather than about pixel count.

## 6. THE CONTROL ARM -- `ltx_video` COMPLETES THE SAME TOPOLOGY (2026-08-09)

**This is the most useful fact in the document and it was measured, not
reasoned.** A matched leg was run immediately after the failure, same session,
same box, same boot generation:

| | wan_ti2v leg | ltx_video control |
|---|---|---|
| profile | `otr_upscale_ship` | `otr_upscale_ltx_probe` (draft, added for this) |
| words / bank | 45 / original | 45 / original |
| composite | 1920x1080 | 1920x1080 |
| video canvas | (wan_ti2v declared) | 832x480 |
| result | **FAIL** at node 92, shot 3 | **`RESULT SUCCESS`**, `Prompt executed in 00:34:44` |
| beats | died after shot_b001 (200 frames, 2 chained segments) | **`assembled 8 beats -> 1072 frames`** |
| deliverable | none | `signal_lost_nightshift_erasure_20260809_115705_..._final.mp4`, 30,757,126 bytes, `obs_publish OK` |

The control rendered the SAME structural feature that killed wan_ti2v -- a
multi-beat timeline with chained segments -- and finished with headroom to
spare. **So the retention is `wan_ti2v`-SPECIFIC, not a property of the video
path, the cost model, the 1920x1080 composite, or this box's VRAM.**

That kills the "maybe the box is just too small" reading and points the
investigation at wan_ti2v's own residency. The strongest lead in the log is its
VAE staging, which appears twice:
`Model WanVAE prepared for dynamic VRAM loading. 1344MB Staged.`

## 7. WHAT IS STILL NOT KNOWN -- do not assume any of this

* Whether `post 8190 MB` is the model deliberately kept warm for the next shot
  (an intentional optimisation that simply does not account for a later larger
  beat) or a genuine leak. **Find the intent before changing anything.**
* Whether `fastwan_8gb` / `humo` retain the same way. `ltx_video` does NOT
  (section 6); the other two are untested.
* Whether shot ORDER matters -- the 200-frame chained beat ran FIRST here. A
  topology that renders small beats first might complete and hide this.
* Whether this is a regression or has always been true. The item-8 tombstone
  records chip 4's leg as owed-but-never-run, so there is no prior green
  receipt for this profile to compare against.

## 7. THE TRAP

The tempting "fix" is to lower the frame budget or raise the margin until the
leg passes. That would convert a real capability limit into a silent quality
cut, and the S4 refusal exists precisely to stop that. **Per the standing
directive the recipes are not on the table** -- measurement runs the shipped
recipe unchanged. Any fix must recover the headroom, not shrink the render to
fit the leak.
