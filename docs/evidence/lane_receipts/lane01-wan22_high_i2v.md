# VIDEO_LANE_PREFLIGHT receipt -- lane 1, `wan22_high_i2v` (`wan_i2v`)

`VIDEO_LANE_PREFLIGHT receipt: wan_i2v | 2026-08-11 | suite run 9865 passed,
109 skipped, 1 xfailed | smoke receipt
output/otr/episodes/_lane_smokes/lane01_wan_i2v/ | verdict PASS`

Lane 1 of the 21-lane transplant
(`docs/2026-08-10-FINAL-QA-video-build-corpus.md`). It went first because it
was BROKEN: `assert_usable` raised `MISSING_MODEL` before the first forward, so
the lane was registered, selectable, and dead.

## Matrix row

All seven gates GREEN. Both of this lane's `EXPECTED_RED` entries were deleted
from `tests/test_lane_preflight_matrix.py` in the same commit -- the suite's
strict unexpected-pass check is what forced that, and it fired correctly the
first time the fixes landed.

| Gate | State | Evidence |
|---|---|---|
| G1 weights resolve | PASS | resolves with NO environment variables set at all; `folder_paths` -> `<comfy_root>/models` -> configured models root |
| G2 canvas truth | PASS | declares 832x480, /32-legal (26 x 15), pinned by 5 tests, agrees with `otr_w45_wan_i2v.json`; PROVED live -- the emitted clip is 832x480, not the 1472x832 default it used to fall through to |
| G3 contract vs runtime | PASS | 33..177 q4, native_fps 25 == target_fps 25, continuity `strict_first_frame` declared explicitly |
| G4 admission honesty | PASS | no qualified cost row; the manifest says "admission NOT enforced" for this lane, in words |
| G5 audio law (V-1) | PASS | `canonicalize` runs `validate_silent_clip_contract`; PROVED live -- ffprobe finds exactly ONE stream on the emitted file, video |
| G6 guards | PASS | named `EngineUnusable` from `assert_usable`; no unguarded module-scope numeric env reads |
| G7 public surface | PASS | exactly one live menu option; legacy spellings resolve and never appear; `ENGINE_MATRIX.md` regenerated; `still_plan` audit-clean |

## G8.1 solo smoke -- one real render, on the lane's declared boot lane

Reset first, per the standing rule: two resident OTR headless servers on port
59189 (started 00:56) were holding 9,969 MiB. Killed selectively by
CommandLine; VRAM fell to 1,636 MiB, under the 3.0 GiB idle gate and under the
2.0 GiB stamp threshold, so this leg carries no elevated-baseline stamp.

| Item | Value |
|---|---|
| Boot | `scripts/_otr_soak_server_launch.cmd` lane token `WAN`, port 8000, no `--use-sage-attention` (no launcher passes one) |
| Idle VRAM before boot | 1,636 MiB |
| Idle VRAM after boot | 1,925 MiB |
| Harness | `scripts/_otr_single_engine_smoke.py --engine wan_i2v --frames 33` |
| Init image | a 832x1216 PORTRAIT still, deliberately -- it exercises the N9 materialize path (pad/crop with ONE uniform scale) rather than a canvas-shaped input that would prove nothing about stretching |
| Prompt id | `887ddd78-2b29-46f5-9a0e-cebb9c9b006d` |
| History status | SUCCESS |
| Wall time | 217.9 s (20 sampler steps at ~9.9 s/step, plus decode and encode) |
| Canvas PROBED | **832x480** -- equals the declaration |
| Frames PROBED | **33** counted (`-count_frames`), `nb_frames` 33, duration 1.320 s = 33/25 exactly |
| Rate | 25/1 |
| Colour | `yuv420p`, `bt709` |
| Audio | **zero audio streams** -- V-1 proved on the emitted file, not declared |
| Trim | none; 33 is on the 4n+1 ladder and was delivered exactly, so no tail trim fired and the trim ratio is 0 |
| Peak VRAM observed | ~13,751 MiB (13.43 GiB) mid-sampling, by `nvidia-smi` |
| Artifact | `output/otr/episodes/_lane_smokes/lane01_wan_i2v/wan_i2v_832x480_f33_smoke.mp4` |
| Artifact sha256 | `87c409f3d27a57a6dcf1e878ad9671adf4731a030aa7992c66918ba77f2ecff5` |
| Size | 487,813 bytes |

### Two honest caveats on this receipt

**The clip's own VRAM stamp is not a peak.** The node report carries
`vram_used_mb: 2523`, which is an INSTANTANEOUS read taken after the render;
the real peak was ~13.4 GiB. That is lesson L4's defect (`render_driver.py`
falls back to an instantaneous read when the adapter returns no
`vram_peak_mb`), and it is why the peak above is sourced from `nvidia-smi`
rather than from the manifest row. wan_i2v stamping a true peak is not lane 1's
scope -- it is recorded here so a later lane inherits a stated problem rather
than discovering a plausible-looking number.

**This is a SOLO-NODE proof, not a canonical-graph run.** The harness submits a
one-node `OTR_VideoRenderBatch` graph, which is the established mechanism for a
per-lane opt-in proof. The canonical workflow
(`workflows/otr_canonical.json`) is unchanged by this lane -- node 87's saved
values are untouched and its widget count is still 15 -- and the full
canonical-graph proof is lane 22's end-to-end episode gate, exactly where the
corpus puts it.

## What this lane changed

- `nodes/_otr_video_engines/eng_wan_i2v.py` -- correct default UNET basename
  and category; `folder_paths` fallback in `_installed`; `render_canvas`
  declared with the re-measurement question ANSWERED (KEEP, and why it does not
  displace wan_ti2v); a refusal message that names every route to fix it; two
  stale docstring claims corrected.
- `nodes/_otr_video_engines/wan_shared.py` -- `configured_models_root()` and a
  third resolution probe. SHARED CODE: it fixes all three WAN lanes' off-runtime
  resolution, so `wan_ti2v` and `fastwan_8gb` got non-regression coverage in the
  same chunk and are NOT thereby marked green -- their own lane packets still
  own their rows.
- `nodes/_otr_shared/public_engines.py` -- `wan22_high_i2v` live;
  `wan21_high_i2v` (the spec's string) as a legacy alias. See the operator note
  below.
- `nodes/_otr_video_engines/registry.py` -- stale ckpt-default comment.
- `tests/test_wan_i2v.py` -- NEW, 21 tests, the template later lanes copy.
- `tests/test_ltx_8gb_dir_override_tripwire.py` -- the fixture now owns EVERY
  models root, not just one. This was a real find: two absence assertions were
  passing only because the other roots happened to be invisible.
- `tests/test_ltx_8gb_canonical_canvas.py` -- the "declares nothing"
  differential control moved from wan_i2v to mesh_stage.
- `docs/ENGINE_MATRIX.md`, `workflows/variants/otr_w45_wan_i2v.*` --
  regenerated, never hand-edited.
- `config/profiles/otr_w45_wan_i2v.json` -- weight pin in `launch.env`, the
  LIVE channel (`extra_args` is documentation-only).

## Naming: RULED 2026-08-11 -- `wan22_high_i2v` stands

The spec's naming table prints `wan21_high_i2v`. The lane is Wan **2.2** -- the
weight is `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, the frozen recipe
is `wan22_14b_i2v_single_pass_v1`, and `registry.CAPABILITIES` already carries a
dated note recording that this exact row was corrected FROM `wan2.1` TO
`wan2.2-i2v` once before. The live menu therefore says `wan22_high_i2v` and the
spec's string is registered as a legacy alias, so BOTH resolve and neither ever
becomes a dead end.

**Operator ruling, 2026-08-11: `wan22_high_i2v` is correct and stands.** The
naming had been decided all along; `wan21` was a single mistyped version number
in the spec that everything downstream inherited. Spec and transplant plan
corrected, and no code changed -- this lane had used the right name from the
start. Recorded as lesson L8.
