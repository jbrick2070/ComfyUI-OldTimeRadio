# RTX 4060 alpha.24 portability result - 2026-09-06

**FAIL WITH FINDINGS.** The original GUI-only installation failed at the writer
loader. A separately labeled, hand-patched one-act diagnostic then ran for
4 hours 5 minutes and failed at its first image: the selected Z-Image diffusion
model was unresolved. No final AI-visual episode or OBS publication was produced.

This is evidence for building the 8 GB onboarding guide, not a qualified
mouse-only quick start. The complete chronological record is
[4060_DRILL_LOG.md](4060_DRILL_LOG.md); new terminal defect:
[PBUG-20260906-01](PROD_BUG_LOG.md#pbug-20260906-01----alpha24-canonical-reaches-first-image-without-visual-weights).
Screenshots and verbatim tracebacks remain in the task's private evidence directory.

## Scope and result

- Physical MRKT RTX 4060 Laptop, 8 GB, Ada; no other GPU or SSH.
- Manager version explicitly selected: `comfyui-old-time-radio 2.0.0-alpha.24`.
  Pending caution accepted; `latest` was not substituted. Console loaded all 25 nodes.
- Shipped canonical JSON: LTX098-low16:9 x3, ZImageTurbo x3, Kokoro/Kokoro,
  MusicGen; one-act user copy. Procgen source Normal was retained; upscale/blend
  remained bypassed. No canonical values were changed to rescue this trial.
- The installation was not pristine: pre-existing extra packs, authentication and
  dependencies remained. No token was acquired or transferred, and no new pack
  was installed for this trial. Earlier authorized model deletion is in the drill log.
- `RESULT SUCCESS`: absent. `obs_publish OK`: absent. Expected output/otr/obs
  directory: absent. Corresponding final AI-visual episode: absent.
- Conditional alternative-model campaign: not started because this act failed.

## One-act diagnostic timeline (PDT)

| Time | Verified milestone |
| --- | --- |
| Sep5 23:04:44.645 | Exactly one GUI Run; both writer slots Gemma4-12B |
| Sep6 01:11:16 | Fixed story saved; accepted first script attempt |
| 01:45:04 | Cleaner finished 16 voiced rows, no dirty rows/repairs |
| 01:46:51 | Kokoro/MusicGen master audio completed, 184.16625 seconds |
| 01:49:00 | Procedural intermediate MP4 completed; app renamed episode |
| 02:29:11 | ShotLock completed; 18 intentional pre-image deferrals |
| 03:10:17.678 | Final image-prompt call completed |
| 03:10:17.727 | First image failed adapter usability: missing ZImage model |

Terminal episode identity:
`signal_lost_the_frequency_of_friction_20260906_014652`.
Server elapsed message: `Prompt executed in 04:05:32`.
Measured GUI Run-to-terminal-log interval: 04:05:33.134.

The exception names `still_music_opening_001`, role `music_visual`, engine
`z_image_turbo`, and unresolved `z_image_turbo_bf16.safetensors`. It is a
missing-model refusal before model loading, **not an OOM or an 8 GB capacity
measurement**. LTX execution was never reached; its success/failure is untested.

The episode retains master WAV 35,359,964 bytes, opening WAV 774,444 bytes, closing
WAV 518,444 bytes, and procedural MP4 170,782,765 bytes. The ledger's root
`final_video_path` still equals `meta.procgen_path`; that field name is not proof
of final rendering. Its `images` key is absent. These intermediates were preserved.

## Why the zero-setup experience failed

The image dispatcher checks the selected adapter before generation. ZImage's
resolver cannot verify an installed diffusion file and correctly refuses.
Removing this guard or substituting another engine would conceal the defect.

The earlier workflow validator checks node/socket/widget contracts, not selected
visual assets. The packaged canonical has no provisioning node or download
metadata. Startup prefetches Kokoro voices only. Development has weight manifests,
but `scripts/otr_provision.py` and `scripts/otr_fetch_lane_weights.py` are excluded
by `.comfyignore` and absent from this installed alpha.24. The README advertised
those scripts and described old still_flat/StableAudio3 defaults. This report's
documentation change corrects the canonical defaults and removes the blanket
commercial-friendly claim, which conflicts with MusicGen's engine metadata.

Do not use the whole development provisioner as a workaround: it unconditionally
installs GGUF, LTXVideo and AnimateDiff packs, outside this zero-new-pack test.
Generic Manager model-download offerings were not inspected during the terminal
audit; no claim is made that Manager itself lacks a model downloader.

Required product correction: a shipped GUI-accessible, selected-engine,
weight-only asset plan/provisioning step **before expensive writing**, using the
actual native-loader roots. Include diffusion, text encoder and VAE dependencies;
deduplicate repeated roles, show download sizes/progress, validate completion,
and preserve exact engine choices. Keep Ada and Blackwell selection distinct.
This needs source/packaging work and cold-install regression coverage, not merely
a separate JSON or a hand-copied checkpoint. No such new runtime fix was applied.

## What the writer correction proved

Source candidate `c61ac222bc02dc405ae1d6441a92227265bdfd92` fixes the NF4
CPU-dispatch retry by planning an explicit device map before quantization.
Seven focused tests passed; the patched Gemma12B loaded, generated and reloaded
throughout this act without the earlier meta-tensor failure. It was very slow,
around 0.4 tokens/second, so downstream prompt stages also took substantial time.

The normal full-GPU path and measured 8.00/15.99 GiB selection/budget probes were
unchanged. That is not a physical5080 benchmark. Full pytest and the configured
Bug Bible checkout were unavailable. No registry release/version bump occurred;
the original zero-hand-step FAIL remains FAIL. Earlier E2B off-CUDA NF4 and
post-download access-violation findings remain separate and unresolved.

## Download and evidence limits

The drill records the Manager ZIP as 4.59 MB and the earlier clean-model Gemma
download as 23.9 GB/eight files in 06:16. Those are application-reported sizes,
not independently measured network bytes. Kokoro/MusicGen transfer sizes were
not exposed in the available log; logical cache footprints are not downloads.
No ZImage download was observed before the terminal missing-model refusal.

From 00:44 onward, GUI capture became black/background-only with stale text.
Two bounded activation attempts failed; no further activation/security/lock
interaction was attempted. Passive captures were still archived and labeled
honestly. Later progress and terminal evidence come from authorized read-only
persisted logs and artifact checks. Missing screenshot visibility and unknown
transfer bytes are evidence limitations, not silently completed guide steps.

## Next verification gate

Test the exact built registry package with empty visual-model roots: expose the
complete asset plan before invoking the writer; cover partial/interrupted
downloads, already-installed files, three-role deduplication, correct loader
roots, and Ada/Blackwell choices; prove zero new node packs. Then rerun one act
through the GUI and require all three final-success receipts. Until then,
neither this canonical nor a changed model choice is qualified by this run.
