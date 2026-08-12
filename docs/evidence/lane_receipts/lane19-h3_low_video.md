# VIDEO_LANE_PREFLIGHT receipt -- lane 19, `h3_low_video` (`minimax_h3_video`)

`VIDEO_LANE_PREFLIGHT receipt: minimax_h3_video | 2026-08-12 | smoke receipt
output/otr/episodes/_lane_smokes/lane19_h3_low_video/ | verdict PASS`

**The first NEW ENGINE of the campaign.** Lanes 10-18 repaired lanes that already
existed; this one adds a 33.1B packed audio-video DiT to a 16 GB card. It is also
the first lane that REQUIRES its own boot contract, the first that natively
produces audio and deliberately does not decode it, and the first whose frame
grid is the model's rather than the canvas's.

## Matrix row

7/7 GREEN on arrival. This lane had no `EXPECTED_RED` entries to delete -- a new
engine either satisfies every gate at registration or it must not register.

## The three answers this lane had to reach on its own

### 1. G2 INVERTS here, and 864x480 was DERIVED rather than picked

All eight cheap lanes before this one closed G2 by declaring their profile canvas
channel INERT, because a procedural lane paints at whatever size the request
carries and a declaration would only overrule `OTR_VIDEO_LANDSCAPE_CANVAS`
(lesson L19). **H3 is not like that**: width/height are `step=32` on every node,
the latent is `H/16 x W/16`, and cost is superlinear in pixels on a card this
render does not fit in twice. A size handed in by an operator lever can take this
lane over the 14.5 GiB gate, which is exactly the canvas-dependent property L19
names as the condition for declaring.

**Which canvas, and it was not the trained one.** State the surface (L7):
`adapt_canvas()` maps a 16:9 ask to 1344x768 (short edge 768 under the
`768*1344` area cap) and that IS H3's trained shape -- but it is called at ONE
place, inside `MiniMaxH3ReferenceToVideo`, to size REFERENCE media. It never
rewrites the generation canvas, so the trained numbers describe training, not
enforcement, and the choice was genuinely ours.

The **public id decided it.** The convention is
`<model><version>_<low|high>_<capability>` where `low` means measured under
~8 GiB, and the corpus named this lane `h3_low_video`. Against the lab receipts
on this box, under this lane's own boot:

| canvas | leg | absolute peak | wall clock | is `low` true? |
|---|---|---:|---:|---|
| **864x480** | `h3_mime_i2v`, model f90 | **7.28 GiB** | 178.9 s | **yes** |
| 1344x768 (trained) | `h3_i2v_best`, model f124 | 9.15 GiB | 1182.5 s | no |
| 832x480 | `h3_i2v_canonical_..._f107` | **15.39 GiB FAILED** | 178.8 s | n/a |

864x480 is the only value with a passing I2V receipt at which the id's own token
is true, and it is /32-legal on both axes (27 x 15). The 832x480 leg is evidence
of a FAILURE and never evidence for a number -- it was also below the trained
floor AND on a boot lane without the reserve clamp.

### 2. The 24 -> 25 conversion is the lane's real deliverable

`FPS = 24` in the node; the canvas is 25. The local encoder can only LABEL a rate
(`wrapper_bridge` puts `-r` before `-i pipe:0`) and can never resample, so a
relabelled render plays ~4% short -- the mouth slides ~320 ms over 8 s. The
decoded uint8 batch is remapped through a nearest-source-frame integer index map
immediately before `encode_frames_to_silent_mp4`.

The menu is published in CANVAS frames and DERIVED from the node's own grid rule
at import: model `17k+5` over the trained 124..362 becomes
`129,146,164,182,200,...,377`. It matches the spec's tuple exactly, and it is
re-derived in the tests against an independent copy of `align_frame_count` rather
than compared against a transcribed list.

**Integer arithmetic on purpose.** `round(j*24/25)` is written `(48j+25)//50`. A
float path would make the tail of a 377-frame clip depend on binary rounding, and
Python's `round` is half-to-EVEN so a tie would resolve differently at even and
odd indices. There are no ties -- `j*24/25` is never an exact half -- and the
test asserts BOTH halves of that, so the map's correctness does not rest on a
coincidence.

**The floor is deliberate.** `canvas_frames = (model*25)//24`, not `round`. A
clip lasts `model/24` seconds and the canvas may only publish frames that exist
inside that duration; rounding up would invent a frame past the end of the
render. The cost is under one frame at the tail (at model f141 the render is
5.875 s and the 146-frame delivery is 5.840 s), and that is the right side to err
on -- the alternative manufactures a frame that `native_frame_count` exists to
make visible.

### 3. V-1 is real here rather than ceremonial

H3 is the first LOCAL engine that natively produces audio: its latent is a
NestedTensor PAIR and the installed nodes ship a decoder for each half. **This
lane's graph carries no audio VAE at all** -- it is the lab's hash-pinned
topology minus `VAEDecodeAudio`, `CreateVideo` and `SaveVideo` -- so the clip is
silent by construction. That is not taken on trust: `canonicalize` ffprobes this
lane's OWN emitted file before writing `has_audio: False`, because on this engine
that literal is the one field in the roster that could be a lie.

## THE TWO PRODUCTION BUGS THE SMOKE FOUND (neither is in this lane)

Both were found by running the render, not by a test, and both had been shipped
and dormant.

### A. The `h3` boot contract clamped the wrong knobs

The contract was drafted from the spec's prose (`sage_attention: false` +
`--disable-pinned-memory`) before any H3 measurement was read back. It would have
named two knobs, passed its own check, and still let the lane load its way over
the ceiling.

**Every** H3 leg on this box that cleared 14.5 GiB was booted with
`--reserve-vram 12`, including the trained 1344x768 canvas at 9.15 GiB. The one
I2V leg without it peaked **15.39 GiB and FAILED**, on a canvas 2.6x SMALLER and
a length SHORTER than the leg that passed. A smaller, shorter render peaking 70%
higher is not a canvas effect -- the boot is the only structural difference, and
the mechanism is plain: reserving 12 GiB away from model loading forces the 21 GB
DiT to stream instead of attempting residency. `reserve_vram_gb` moved
`None -> 12.0`.

### B. The Sage probe could not read Sage on any server

`boot_contracts.running_server_boot_state` reached its sibling with
`from nodes._otr_video_engines.motion_common import ...`. **`nodes` resolves
against `sys.path`**: in the CPU suite that IS this package, so the probe worked
in every test; inside a running ComfyUI server `nodes` is ComfyUI's OWN
node-registry module and the import raised `ModuleNotFoundError` on every boot.

It shipped dormant because the error is only CONSULTED by a contract that
constrains Sage, and `h3` is the first one that does. It then presented as an
unrenderable lane: UNKNOWN is correctly not a pass, so the lane refused on every
server for a reason that was about an import.

**Two more instances, swept (L13):**

* `content_oracle.family_for_engine` -- the WORST, because it failed SOFTLY into
  a bare `except: pass` and answered from `_FAMILY_FALLBACK` on every call. That
  table stops at 2026-07-05, so `ltx_8gb`, `fastwan_8gb`, `still_word`, every
  cloud lane and this one all resolved to family `""` in production -- which is
  not in `MOTION_FAMILIES`, so `motion_required_for_engine` answered False and
  those lanes were **silently motion-EXEMPT**. A frozen clip from any of them
  would have passed the motion check by never being asked.
* `slot_matrix.eligible_engines_for_role` -- raised outright.

All three are relative imports now, and an AST-based test fails on any future
absolute `nodes.` import of a sibling in either shared package.

## The seam `render_single` was missing

`render_single` passes `profile=None`, which selects the `default` contract -- so
the first lane to REQUIRE its own boot could not be smoked on the boot it
declares. Same seam and same shape as lane 7's canvas fix one commit earlier:
this function invents its own request, so anything it does not ask for is absent
from every solo lane smoke.

It now selects the engine's contract when the caller names none AND the engine
declares exactly one. **This is not a bypass:** selecting a contract is a CLAIM,
and `assert_running_server` still proves it against `comfy.cli_args`. A smoke on
a wrongly-booted server still refuses -- it just refuses for the true reason. An
engine declaring two or more contracts is left alone, because there the selection
is a real choice and inventing one would be guessing.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot | the named **`h3`** contract: Sage-free, `--disable-pinned-memory`, `--reserve-vram 12`, applied through `launch_env_for("h3")`'s two env rows |
| Node classes | all four H3 classes + all ten core classes confirmed live in `/object_info` BEFORE submit, with the `length` widget's own `step=17` / `default=124` / trained-range tooltip read back |
| Harness | `_otr_single_engine_smoke.py --engine minimax_h3_video --frames 129 --portrait <png>` |
| Prompt id | `64c66dbc-1202-4b19-9877-dc126f0b2ef1` |
| Recipe | `minimax_h3_fl2va_int8_res_multistep_20step_v1` (20 steps, `res_multistep`, `simple`, denoise 1.0), no quant |
| Wall time | **242.6 s** (20 sampler steps at ~10.3 s/step, plus VAE decode) |
| Canvas PROBED | **864x480** -- equals the declaration |
| Frames PROBED | **129 packets counted**, and 129 was the ask -- no trim |
| Model frames | 124 (the 17k+5 rung), logged by the conversion line at render time |
| Rate / duration | **25/1**, duration **5.160000 s** = 129/25 EXACTLY |
| 24 -> 25 proof | the runtime log states it: `124 model frame(s) at 24 fps (5.167 s) -> 129 canvas frame(s) at 25 fps (5.160 s)` |
| Codec / pixfmt / colour | h264 / yuv420p / bt709 |
| Audio | **`nb_streams=1`** -- ZERO audio streams, on the one local engine that natively produces audio |
| Extension | `native_frame_count: 129`, `extension_mode: none`. **State the surface:** 25/24 is not 1, so 129 canvas frames drawn from 124 source frames show about one frame in 24 twice -- **5 repeats here, 15 at f377**. Nothing is invented, no tail is mirrored, no beat is padded, and the duration is exact. `native = emitted` is also the only stamp `acceptance.py` can read correctly: its producer contract is that manufactured frames are APPENDED AT THE TAIL, so stamping 124 would claim frames 124..128 are the manufactured ones when the repeats are spread evenly through the clip |
| Peak, ABSOLUTE | **6,315 MB**, `VramPeakProbe` maximum, cold |
| Peak, NET | **not claimed.** No pre-queue baseline was sampled on this leg, and the cost-row surface is NET by the 2026-08-11 ruling -- so absolute is reported and no net figure is invented (L7) |
| Headroom | 8,185 MB under the 14,500 MB ceiling on the ABSOLUTE surface |
| Streaming | the log shows the stack staged rather than resident: DiT 19,995 MB, encoder 14,956 MB, VAE 4,965 MB -- ~40 GB of weights through a 16 GB card at a 6.3 GB peak. That is the reserve clamp working |
| Artifact | `.../lane19_h3_low_video/minimax_h3_video_864x480_f129_h3boot_smoke.mp4` |
| sha256 | `9960f15e78a6313713a53fe4600e13968b99a8a27fd0f687e885c01c943f7b7c` |

**Three earlier attempts failed, and each failure was a gate doing its job**: the
first refused for a missing `init_image` (`FamilyInputGap`), the second refused
because the Sage probe could not read the server (bug B above), and only the
third rendered. No gate was weakened to get the pass.

## Deliberately NOT done here

**No cost row, and the manifest says so in words.** The boot contract is what
actually gates this lane's VRAM outcome, and a boot refusal is not a per-beat
admission check. Nothing on this lane refuses an over-budget PLAN.

**`minimax_h3_audio_in` is NOT registered.** It shares this implementation
module and it is lane 20's. Two public ids on one internal id collapses
`_INTERNAL_TO_PUBLIC` and trips its module-scope bijection assert at IMPORT time,
which empties most of the ComfyUI menu rather than failing one lane cleanly (L5).

**The mouth policy is UNTOUCHED.** `render_driver`'s `"ltx_audio_in"` equality
test becomes a membership test in LANE 20, with the registration it exists for.
Doing it here would wire a policy to an engine that is not registered yet.

**`last_frame` is never wired.** It is H3's first/LAST interpolation endpoint,
a different capability from the first-frame chaining this lane declares.

**`commercial_clean = False`,** deliberately. The MiniMax H3 Community License is
not OSI, and the 2026-08-07 written authorization is a conditional grant to one
licensee -- which is not a clean commercial license for a published open-source
pipeline. The flag drives the release-gate warning and the filename tag, never
selection.
