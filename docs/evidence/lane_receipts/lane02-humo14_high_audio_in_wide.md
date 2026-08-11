# VIDEO_LANE_PREFLIGHT receipt -- lane 2, `humo14_high_audio_in_wide` (`humo_14B_169`)

`VIDEO_LANE_PREFLIGHT receipt: humo_14B_169 | 2026-08-11 | suite run 9899
passed, 109 skipped, 1 xfailed | smoke receipt
output/otr/episodes/_lane_smokes/lane02_humo14_wide/ | verdict PASS`

Lane 2 of the transplant, and the first lane that NEEDS a named boot lane -- so
the boot-contract mechanism was built here, with its real consumer, rather than
as unused infrastructure.

## Matrix row

All seven gates GREEN; the lane's `EXPECTED_RED` G2 entry deleted in the same
commit.

| Gate | State | Evidence |
|---|---|---|
| G1 weights resolve | PASS | all four HuMo tiers resolve with NO environment variables set (see the inherited-lesson note below) |
| G2 canvas truth | PASS | declares 832x480, /32-legal, pinned, agrees with the profile; a contradicting `OTR_HUMO_WIDTH/HEIGHT` is now a NAMED refusal; PROVED live -- the emitted clip is 832x480 |
| G3 contract vs runtime | PASS | 33..97 q4, native 25 == target 25, `soft_reference` declared |
| G4 admission honesty | PASS | manifest says "admission NOT enforced" for this lane, in words |
| G5 audio law (V-1) | PASS | `validate_silent_clip_contract` on its own file; PROVED live -- ffprobe finds exactly one stream, video |
| G6 guards | PASS | named `EngineUnusable`; boot contract checked against the RUNNING server |
| G7 public surface | PASS | one live menu option; `ENGINE_MATRIX.md` and the variant regenerated |

## G8.1 solo smoke -- on the lane's DECLARED boot lane

| Item | Value |
|---|---|
| Boot contract | `humo_diet`, applied through `launch.env` |
| Flags on the real command line | `--reserve-vram 2.921 --disable-pinned-memory` (confirmed in the server log; `--disable-pinned-memory` reached argv for the first time in this repo's history) |
| Idle VRAM before boot | 1,651 MiB (after a selective reset) |
| Idle VRAM after boot | 1,940 MiB |
| Harness | `_otr_single_engine_smoke.py --engine humo_14B_169 --frames 97` with a real still and a real master WAV |
| Prompt id | `a6046d07-93e5-4154-acef-632a5661a07c` |
| Wall time | 249.2 s |
| Canvas PROBED | **832x480** -- equals the declaration |
| Frames PROBED | **97** counted, duration 3.880 s = 97/25 exactly |
| Rate / colour | 25/1, `yuv420p`, `bt709` |
| Audio | **zero audio streams** |
| Trim | none; 97 is the tier ceiling and was delivered exactly |
| Artifact | `output/otr/episodes/_lane_smokes/lane02_humo14_wide/humo14_169_832x480_f97_diet_smoke.mp4` |
| Artifact sha256 | `933eb605f4e2dd495ba324f9b215d56caf8235ea8d0c1696379bdbdf3bbae4d5` |

## THE NUMBER THE OPERATOR SHOULD LOOK AT

**OTR-side render-window peak: 14,604 MB (14.26 GiB), COLD, absolute.**

That is 1,699 MB of headroom on a 16,303 MB card and 0.24 GiB under the
14.5 GiB gate -- not the 1.44 GiB of headroom the corpus's headline number
implies.

It is NOT a contradiction of the lab's 13.06 GiB, and this receipt does not
claim it is. They are different measurements:

- **Cache state.** The lab number is WARM. This was the FIRST render after a
  fresh boot, so the model load sits inside the measured window. The lab's own
  cold landscape figure is 13.17 GiB.
- **Measurement surface.** This is an ABSOLUTE device-total peak sampled by
  `VramPeakProbe`, so it includes the ~1,940 MB the idle server already held.
  Net of that baseline it is roughly 12.66 GiB, which sits BELOW the lab's warm
  13.06 GiB rather than above it. The lab distinguishes absolute from net
  peaks; the corpus table did not carry that column, which is precisely lesson
  L7's complaint.

**What this receipt does claim:** one real 832x480x97 render on the hero lane,
under the diet boot, completed inside the gate with the flags proven on the
command line. **What it does not claim:** that the lane is machine-qualified.
Nothing refuses an over-budget plan here yet -- `QUALIFIED_COST_ROWS` is still
empty and the manifest says so -- and a single cold leg is not an envelope.
Deriving a qualified cost row from OTR-lifecycle numbers is lane 5's work and
should use this surface, not the lab's, because this is the surface production
runs on.

## What this lane changed

**The boot-contract mechanism** (`nodes/_otr_shared/boot_contracts.py`, new):
named contracts (`default`, `humo_diet`, `h3`), the `launch.env` rows each one
needs, a probe of the RUNNING server's `comfy.cli_args.args`, and knob-by-knob
refusals. Three rules it enforces, each a defect that already happened -- the
contract rides `launch.env` and never the documentation-only `extra_args`;
enforcement reads the running process and never the config that was supposed to
be honoured; and `default` constrains nothing so no shipped profile is retired.

**The launcher hook** (`scripts/_otr_soak_server_launch.cmd`): until this
commit there was a hook for `--reserve-vram` and none for
`--disable-pinned-memory`, so a profile that "configured" the diet clamped
exactly one of its two knobs and the other was a markdown string. That is why
the mechanism had to ship with a consumer.

**`launch.boot_contract`**, an OPTIONAL profile key. Optional because the launch
key set is closed-validated: a required key would have broken all ~20 shipped
profiles at once, and a profile that names none is on the stock boot, which is
what every one of them already meant.

**S8b-4, the canvas**: declared at the size it was measured at, and
`OTR_HUMO_WIDTH/HEIGHT` now REFUSE to contradict a declaring tier instead of
silently winning. Undeclared sibling tiers keep their overrides, pinned by test.

**S8b-6, the manifest row**: `render_peak` had been measured and logged since
2026-08-06 and then dropped on the floor, so every HuMo clip reached the ledger
with `vram_peak_mb`/`recipe`/`quant`/`render_canvas` null and the driver fell
back to an instantaneous VRAM read -- a sample at an arbitrary moment wearing
the name of a peak. All four tiers now produce the row. S2's envelope work
depends on this.

**S8b-7**: the comment explaining the 97-frame cap still said 49.

## A lesson INHERITED, not rediscovered

L1 said to check the lane's weight resolution before writing code. HuMo had the
identical defect wan_i2v died of: both `_ckpt_path` implementations stopped at
`<comfy_root>/models/diffusion_models`, so off the ComfyUI runtime a correctly
installed HuMo read as MISSING. Two copies of the chain were also two places to
fix it, so there is now one resolver, `_resolve_unet`, shared by all four tiers.

**The sibling tiers are NOT marked green by this.** `humo`, `humo_1.7B` and
`humo_1.7B_169` got a shared-code fix and non-regression coverage; their rows
still carry their own `EXPECTED_RED` entries and their packets (lanes 3 and 4)
still own them.

## Still open on this lane

**Enforcement timing.** S8 asks for the boot check to move EARLY, into the
ShotLock preflight beside `mouth_owner_for_beat`, with the render-time check as
defence in depth. It runs today inside `assert_usable`, which receives the
profile -- so the check is real and fires wherever `assert_usable` is called,
but `assert_usable` itself is still called inside the render phase. Moving it
needs `boot_contract` plumbed into the frozen director policy that ShotLock's
preflight receives, and that plumbing is not something to improvise at the end
of a night. It is queued as its own row rather than half-done.
