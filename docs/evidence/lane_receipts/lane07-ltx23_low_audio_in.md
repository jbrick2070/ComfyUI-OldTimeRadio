# VIDEO_LANE_PREFLIGHT receipt -- lane 7, `ltx23_low_audio_in` (`ltx_audio_in`)

`VIDEO_LANE_PREFLIGHT receipt: ltx_audio_in | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane07_ltx23_low_audio_in/ | verdict PASS`

The lane where two spec items turned out to be one defect, and where three
faults belonging to OTHER lanes surfaced because this was the first lane whose
canvas was not the number every other channel already defaulted to.

## Matrix row

7/7 GREEN. Both `EXPECTED_RED` entries (`G2`, `G6`) deleted in this commit.

## THE HEADLINE THE OPERATOR SHOULD READ FIRST

**Two numbers, two questions, and they answer differently. State both or say
neither** -- the operator's NET-not-absolute ruling (`f2470e31`) landed while
this lane was rendering and applies directly.

| Surface | Value | What it answers |
|---|---:|---|
| **ABSOLUTE** device-total peak, COLD | **14,465 MB** | "does this fit on THIS box today?" -- 35 MB under the 14,500 MB ceiling, 0.24% |
| **NET** (absolute minus this leg's own 2,513 MB pre-queue baseline) | **11,952 MB** | "what should a cost row be seeded from?" |

**On the cost-row question this leg is unremarkable and healthy.** 11,952 MB NET
sits right alongside the three HuMo NET figures already in the manifest
(11,911 / 12,664 / 13,321 MB). It is a `VramPeakProbe` MAXIMUM, not an
`nvidia-smi` sample, so it is seed-eligible under the ruling.

**On the will-it-fit question the margin is 35 MB and that deserves a decision
rather than a shrug.** The absolute total is what actually OOMs, and 0.24% is
not headroom -- a slightly heavier desktop eats it. Note also that the live
figure is **1.92x the lab's warm number** for the identical configuration
(7,536 MB lab warm vs 14,465 MB live cold) -- the second time this lane has
caught a lab number reading low, and both kibitz r1 reviewers predicted exactly
this before the render ran.

Also worth stating plainly: **the live in-pipeline peak is 1.92x the lab's warm
figure** for the same configuration (7.36 GiB = 7,536 MB lab warm vs 14,465 MB
live cold). That is not a contradiction -- different cache state, different
measurement surface, device-total including the ~2.1 GB idle baseline -- but it
is the second time this lane has caught a lab number reading low, and both
kibitz r1 reviewers independently predicted exactly this before the render ran.
The repo's own precedent treats the ceiling as 14,500 MB: `render_driver`
recorded 1280x704 at 14,716 MB as a BREACH. Against 14.5 **GiB** (14,848 MB)
the margin would be 383 MB instead.

### ROW 7b, RESOLVED 2026-08-11 -- the diet leg, and the answer is MARGINAL

The operator refused to wave the 0.24% through and ordered the lever proved.
It was. Both legs are cold, same recipe / canvas / frames / still, `VramPeakProbe`
maxima, changing ONLY the boot:

| Boot | absolute | net | margin under 14,500 MB |
|---|---:|---:|---:|
| `default` | 14,465 MB | 11,952 MB | 35 MB (0.24%) |
| **`ltx_av_diet`** | **14,385 MB** | **11,872 MB** | **115 MB (0.11 GiB)** |

**Decision rule applied: "clears but by < 0.3 GiB -> ship the diet contract and
flag the lane MARGINAL."** 115 MB is under the 307 MB threshold, so the lane
ships on the diet boot and the manifest says MARGINAL in words with both
numbers. It is not called a pass.

**The diet bought only 80 MB, and the reason is the design decision in the
contract.** `reserve_vram_gb` is deliberately `None` for this lane:
`_ltx_av_vram_reserve` bumps ComfyUI's `EXTRA_RESERVED_VRAM` to
`OTR_LTX_AV_RESERVE_VRAM_GB` (default 4.0) across the graph run and only ever
bumps UPWARD, so a boot `--reserve-vram 2.921` would be overwritten by the
adapter's own 4.0 for the whole render window -- a knob that reaches nothing
(L6) while looking in the profile like it did something. Only the
pinned-memory half of the lever was available here, and on this lane it is
worth 80 MB. **HuMo's ~1.9 GiB does not transfer**; that lane had both halves.

`--disable-pinned-memory` was PROVEN in the live process argv (`/system_stats`),
not merely written into the profile -- L6 again: a contract verified against the
config it was meant to honour cannot tell "applied" from "written down".

**Quality: BYTE-IDENTICAL.** The diet clip's sha256 is
`36902e046c68c2da37e88e8c9d5bbebf760052734f4537f039c5ed8e752fb7ec` -- the same
file, bit for bit, as the default-boot clip. So the diet costs nothing at all,
and this is stronger parity evidence than HuMo's by-eye ruling. No operator
eyes needed on the picture.

**What this does NOT resolve.** 115 MB is still a thin margin on the absolute
surface, and the lever is now nearly exhausted on this lane. If the ceiling has
to be cleared with real headroom, the remaining moves are all outside a
configuration change.

**What is NOT available:** dropping to a smaller canvas. For the ia2v
two-stage recipe the canvas must be /64 on both axes (see below), and
**1024x576 is the smallest exact-16:9 rung that satisfies it** -- the next one
down, 512x288, halves to 256x144 and `144 % 32 == 16`. There is no cheaper
legal 16:9 canvas for this recipe.

## S3 and S8b-10 are ONE defect

The corpus lists them separately -- "declare `render_canvas = (1024, 576)`" and
"the ia2v stage-A base latent 416x240 is not /32-legal" -- which reads like a
preference plus a bug. They are the same bug.

The canvas was decided by an inline RECIPE-DEPENDENT branch in the driver:
832x480 when `ia2v_canonical` was live, 512x288 otherwise, either overridable by
`OTR_LTX_AV_RENDER_CANVAS` with no refusal on conflict. Three channels deciding
one number, while `declared_render_canvas` is applied LAST in the same function
and overrules all of them -- so at most one could ever have been true.

And the value that branch chose has no legal stage A. `_build_graph_ia2v`
renders motion at exactly half canvas, then feeds that latent to
`LTXVLatentUpsampler`, whose installed schema
(`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_lt_upsampler.py`)
takes `samples` / `upscale_model` / `vae` and **no target size** -- its own
docstring is "Upsamples a video latent by a factor of 2." So:

* the delivered canvas **is** 2x the stage-A base;
* the stage-A latent is /32-legal **iff the full canvas is /64 on both axes**;
* snapping the base to 416x256 is not a fix -- it would deliver 832x512 against
  a declared 832x480, trading an illegal latent for a canvas lie.

| Canvas | /64 both? | stage A | /32? | exact 16:9? |
|---|---|---|---|---|
| 832x480 | no (480) | 416x240 | **no** | no (26:15) |
| 512x288 | no (288) | 256x144 | **no** | yes |
| 896x512 | yes | 448x256 | yes | no (7:4) |
| 1280x704 | yes | 640x352 | yes | no (1.818) -- and it breached |
| **1024x576** | **yes** | **512x288** | **yes** | **yes** |

1024x576 is the only rung that is both /64 and exact 16:9, so it is the only one
with a legal stage A that also delivers to 1920x1080 with **zero pad area** --
which the 2026-07-26 arc judgment demanded ("Pillarbox: never"). The lane had
been shipping the pillarbox: 832x480 is 26:15 and delivers as 1872x1080 with
side bars.

Three things asserted the illegal value was fine before this lane: the driver
comment ("base 416x240 (all /32)"), `test_ltx_av_ia2v_canonical.py`'s pins on
416 and 240, and the full-canvas gate passing because 832x480 IS /32. Lesson
L11.

## S8b-9 -- one line, and the test that claimed to cover it

`_LTX_AV_RESERVE_VRAM_GB` was a bare module-scope `float()`: the one numeric env
read in this file that `_env_num` had never been applied to. A malformed value
raised during import, the guarded import in `_otr_video_engines/__init__.py`
swallowed it, and the lane vanished from the dropdown with nothing in the log
(registry 27 -> 26) -- while `frame_contract_for` silently answered
`SINGLE_ONLY` for the adapter it could not reach.

`tests/test_ltx_av_env_import_safety.py`'s docstring claims to cover **every**
module-scope environment read; its table had four rows and this was the fifth.
Now five.

## The env/contract refusal, and the hole the panel found in it

`ltx_audio_in` was the last local LTX adapter with no `ContractEnvConflict`
refusal. It has one now, for `OTR_LTX_AV_MAX_FRAMES` against the declared 497
and `OTR_LTX_AV_RENDER_CANVAS` against the declared canvas.

**The first draft was wrong and a reviewer caught it before it shipped.** It
compared `_LTX_AV_MAX_FRAMES` -- the constant `_env_num` has ALREADY normalised
-- against `max_frames`, so `OTR_LTX_AV_MAX_FRAMES=garbage` read as agreement:
the crash-guard had turned it into 497 before the check could see it. Two
correct rules composed into a wrong one. The check now reads the RAW
environment, and both "parses to something else" and "does not parse" refuse.
Lesson L12; pinned by
`test_a_contract_bearing_env_var_is_checked_RAW_not_after_the_fallback`.

## 497 was NOT moved, deliberately

Three numbers, never inferred from one another (L7): model-legal maximum 497;
longest MEASURED rung at this canvas 193; episode policy cap, none. Capping the
declaration at 193 was considered and rejected -- `ltx_audio_in` is absent from
`frame_contract.PLANNING_CAP_ENGINES`, so a ceiling here would narrow nothing
while claiming the 193 rung is enforced (L6, a knob that reaches nothing). A
real production cap is a separate planning change: allowlist the lane, then
prove the multi-clip partition.

## THREE FAULTS THAT BELONG TO OTHER LANES

Surfaced here only because this is the first lane to declare a canvas that
isn't what every other channel already defaults to.

**1. `render_single` never consulted `declared_render_canvas`.** It derived the
canvas from `render_aspect` plus `OTR_VIDEO_RENDER_CANVAS` -- wide -> 832x480,
else portrait 480x832. **Every solo lane smoke runs through that function**, so
lanes 1-6 all validated the aspect default rather than their own declaration.
It stayed invisible because all six declared exactly what that path already
produced. The first lane to declare something else failed its own stage-A /32
guard on a live render -- which is how this was found, and is the new guard
earning its keep on its first outing. Fixed at the source; an explicit `canvas=`
argument still wins so an off-declaration probe stays possible. Pinned by
`test_ltx_8gb_canonical_canvas.py::test_render_single_takes_the_DECLARATION_not_the_aspect_default`,
asserted on `ltx_8gb` rather than this lane because the property belongs to the
declaration mechanism, not to one occupant.

**2. Lane 5's rename never reached five other variants.** `otr_amd16_rocm`,
`otr_amd8_rocm`, `otr_nv40_12gb`, `otr_upscale_ship` and `otr_sbcov_5` still
carried `wan_8gb (16:9)` in node 87, so `scripts/build_variants.py --check` had
been RED since lane 5 closed and lane 7 could not tell its own drift from
inherited drift. A rename regenerates EVERY variant, not the lane's own -- node
87 carries an engine string in variants that have nothing to do with the renamed
lane's profile. Now 46 variants / 0 failures.

**3. The `LTX` boot token enabled only one of the two LTX engines.**
`_otr_soak_server_launch.cmd` set `OTR_ENABLE_LTX_VIDEO=1` and not
`OTR_ENABLE_LTX_AV=1`, so this lane could not smoke on the boot it declares
without exporting a flag by hand. A boot lane you have to supplement by hand is
not a boot lane. The token now enables both; both stay default-off everywhere
else.

## Folded in from the concurrent coder window

`_clip_summary` (`render_driver.py`) returned six keys and dropped the clip's
telemetry, so a lane's VRAM peak never reached disk from a solo smoke -- which
is why `wan_ti2v` and `fastwan` smoked with no usable cost-row seed. Six keys
added, purely additive. **It proved itself on this very render**: the peak,
recipe, quant, render_canvas, native_frame_count and extension_mode below all
come through that passthrough. Without it this receipt would have had the
lane's own `vram_used_mb: 2833` post-render sample and no peak at all.

A cost row may be seeded ONLY from a true `VramPeakProbe` maximum. The 14,465
MB below is that maximum, not a watcher's sample.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot | stock `default` contract on the `LTX` token (Sage-free, BUG-070); no `--reserve-vram`, no `--disable-pinned-memory` |
| Harness | `_otr_single_engine_smoke.py --engine ltx_audio_in --frames 193` with a real 832x480 scene still and a real 7.736 s production audio slice |
| Prompt id | `20ab4324-8e53-49a3-8574-544b29f95d91` |
| Recipe | `ia2v_canonical`, unet `ltx-2.3-22b-dev-Q3_K_M.gguf`, quant `Q3_K_M`, LoRA on |
| Wall time | 303.8 s |
| Canvas PROBED | **1024x576** -- equals the declaration |
| Frames PROBED | **193** counted, duration 7.720 s = 193/25 exactly |
| Rate | 25/1 |
| Codec / pixfmt | h264 / yuv420p |
| Audio | **zero audio streams** -- silence PROVED on the emitted file |
| Trim | none; 193 asked, 193 delivered, `extension_mode: none` |
| Peak, ABSOLUTE | **14,465 MB device-total, COLD, `VramPeakProbe` MAXIMUM** |
| Peak, NET | **11,952 MB** (absolute minus this leg's own 2,513 MB pre-queue baseline) -- seed-eligible per `f2470e31` |
| Sampler | steady ~8.2 s/it through stage A -- no spill (the spill signature on this lane is 72-223 s/it) |
| Unet load | partial as designed: 9,947 MB loaded, 590 MB offloaded |
| Artifact | `.../lane07_ltx23_low_audio_in/ltx_audio_in_1024x576_f193_default_smoke.mp4` |
| sha256 | `36902e046c68c2da37e88e8c9d5bbebf760052734f4537f039c5ed8e752fb7ec` |

## Live menu check, on the running server

`ltx23_low_audio_in (16:9)` present; `ltx23_16gb_audio_in` appears in **no**
menu option; `ltx23_16gb_audio_in`, `ltx23_16gb_audio_in (16:9)`,
`ltx23_low_audio_in` and the bare `ltx_audio_in` all resolve to `ltx_audio_in`.
The public table stayed at 10 rows -- a MOVE, not an ADD, so the module-scope
bijection assert never had a chance to fire.

**Menu row count is 28, not the 27 earlier receipts recorded.** Verified this is
NOT this lane: the CPU-side menu builder returns the identical 28-row set, and
this lane's change was a pure MOVE. The extra row arrived between lane 6 and
now, from another window. Recorded rather than copied forward.

## Deliberately NOT done here

**No cost row.** `QUALIFIED_COST_ROWS` stays empty and the manifest still says
"admission NOT enforced" for this lane, in words -- now with the 14,465 MB
figure and its surface written next to it. One cold leg is a datapoint, not a
qualification.

**No new HQ profile.** S3 asked for "a named profile supplying 1024x576 and 193
frames". The canvas half is now the adapter's declaration and applies to every
profile, and the frames half would be a knob that reaches nothing (`ltx_audio_in`
is not in `PLANNING_CAP_ENGINES`). A fourth profile whose only distinguishing
field is inert is exactly what L6 warns about, so the three existing profiles
were reconciled to 1024x576 instead.
