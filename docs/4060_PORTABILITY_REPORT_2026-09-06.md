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

## Development follow-up, September6 04:00 PDT

A [pre-writer visual-weight readiness candidate](4060_VISUAL_ASSET_READINESS.md)
now exists in development source: hidden live prompt inspection at the existing
validator gate, five weight-only default assets, native-loader identity,
anonymous pinned metadata and no-clobber verified transfers. Canonical JSON
and existing model choices are unchanged. Offline checks passed71 new tests
plus7 earlier loader tests. This does not change any trial verdict above.

Full pytest and Bug Bible remain unavailable. No actual new metadata/model
download, installed-code change, release or second GUI Run occurred. The
03:57 passive screenshot still showed background-only pixels/stale console
text; usable GUI access and a normally distributed candidate are required for
the next one-act qualification. See drillSteps58–59 for the development log.

At04:29PDT the user's fresh ComfyUI process restored a visible canonical
canvas. All25 OTR nodes loaded and act_count1 was visible with0active jobs.
The installed copy is stillalpha.24 without the candidate's new modules;
restart did not apply the correction. GUI visibility is no longer the blocker,
but normal candidate publication/installation remains pending. No act was
requeued; see drillStep61 and the fresh-boot screenshot/audit records.

At04:41PDT the user changed delivery order: get the local workflow working,
push fixes to Git, and update the registry at the end. The reviewed three-file
asset-readiness correction was explicitly applied to installed alpha.24 as a
development HAND STEP. Exact installed/source hashes match;78 focused offline
tests pass again. Canonical JSON/model choices are unchanged. This does not
qualify the clean-install path or prove visual GPU capacity.

GUI restart is paused at Desktop's Stop confirmation (unsaved-work warning);
the final Stop and Run have not been clicked. No new model downloads, version
bump or registry publication occurred. See drillStep62. Next action requires
confirmation of that warning, followed by GUI restart and one-act development
test. Registry clean-install qualification remains a separate final gate.

At04:47 the user approved the Stop warning. The existing instance restarted,
loaded all25 nodes, and the one-act GUI run was submitted exactly once at
04:50:36.149PDT. The reviewed preflight is now executing for real: it obtained
anonymous pinned metadata for five missing visual weights (36,818,738,352
bytes total) and began automatic transfers before writing. No model/JSON change
or new pack accompanied the run. This demonstrates the previously missing
pre-writer asset path has started, not that downloads/render/publish succeeded.

The queue stays at0percent during transfer; actual bytes/progress are visible
under Toggle Bottom Panel > LOGS and the separate Logs window. Log this UI
friction for the onboarding guide. Active run remains under observation;
earlier failed trials and clean-install qualification limits are unchanged.

By04:59:58PDT four automatic transfers had verified successfully (ZImage,
Qwen encoder, VAE and LTX); T5 was still downloading. The0percent-display
finding has a separate source-only native-progress correction with84focused
tests passing and clean independent review. It is not installed into the active
trial. No whole-episode/GPU-capacity/publish success has yet been established.

At05:03:09.811 all five weights were verified and native-loader readiness
passed. The one-act writer started automatically at05:03:09.837 as
pending_20260906_050309; Gemma loaded through explicit CPU offload and CUDA
warmup completed05:03:40.919. No manual asset provision or extra pack was
needed. Current result: asset-readiness path verified; episode still RUNNING.
Original clean-install FAIL remains unchanged, and full4060 PASS still requires
the matching RESULT SUCCESS, obs_publish OK and final episode file.

At05:18:20 the writer completed its first357-token concept and advanced to
original_select. This is live forward progress at about0.4tok/s, not completed
writing or media success. The user has expanded the work to the
[video/audio campaign and final clean-start gate](4060_MODEL_QUALIFICATION_PLAN.md).

The requested one-act shipped default is prepared in DEV source only. The
canonical's sole change is act_count3->1; its derived story-only copy also
synchronizes pre-existing style/registry-id drift. All92 generated variants
await normal regeneration after the current run is idle.86focused checks
passed, but inheritance/full-suite/live qualification is not complete. This
default bundle is not yet committed, pushed, released or installed. No cleanup
or new registry publication has occurred; logs/evidence/source are preserved.

At05:33:23 the372-token concept selection completed and original_brief began
automatically using the technical Gemma slot/cache;128brief tokens were logged
05:38:33. Still RUNNING, with no new error or additional setup. No final
writing/media/publishing qualification yet; installed runtime remains unchanged.

At05:48:07 creative front and news_interpreter reported OK; by05:53:02 the
first cast description completed and the second began. Still RUNNING with no
new OOM/auth/terminal error. Two passive native screenshots at05:53 returned
black pixels despite available log text. This is a documented visual-evidence
gap of unknown cause, not an OTR render failure. Current persisted logs prove
stage progress; usable GUI must be re-established before interactive testing.

At05:58:00 cast locked (announcer+2characters), with actual act_count1 and
4scene beats. Macro outline started05:58:01 and reached64tokens06:00:30.
No new failure or setup step.06:02 passive screenshot remains black; logs-only
progress evidence continues and the active workload is untouched.

At06:12:31 outline succeeded with4voiced+2announcer beats and the6-row
skeleton was saved. Read-only06:17 verification found the matching20419-byte
JSON ledger and6-element lines array. Dramatic-state generation is now active.
This proves an intermediate durable skeleton, not completed words/audio/video
or obs publication. No new error; passive screenshot remains black.

At06:19:15 dramatic state completed and was stamped; continuity generation
followed automatically and reached128tokens06:24:06. Still RUNNING without
new setup or errors; the06:25 passive screenshot remains black. No final
writer/media/publication success established.

At06:32:40 continuity completed359tokens, extracted4facts/3active props and
saved the ledger. The next writer stage logged45SlotContract tokens06:34:35.
Still RUNNING without new error or setup.06:36 passive screenshot remains
black; current disk-log evidence establishes progress, not final episode PASS.

At06:44:50 all6slot drama contracts were saved and marked episode_valid=True.
Dialogue composition followed, with first exchange d002/d003 OK on attempt1
at06:48:27. Still RUNNING, no new error or manual setup.06:51 screenshot
remains black; writing/media/publishing completion is not yet established.

At06:52:08 the second dialogue exchange passed on attempt1. The ledger
reached6lines153words06:53:59; read-only06:58 inspection confirms the matching
28218-byte valid JSON with6line entries. Produced-open brief derive passed
06:57:26 and writing continues. No new errors/setup; screenshot remains black.
Durable dialogue is progress, not final audio/video/obs publication success.

At07:02:14 the title 'Behind the Loose Brick' was written under the same
pending directory. Latest ledger log is6lines170words, after an earlier174word
save. Optional source-term telemetry reports0/5 landed; recorded as a content
observation, not a terminal failure. An accelerate move warning precedes the
reflection model reload. Its NF4 dispatch refusal automatically recovered via
the existing CPU-offload retry;677weights loaded, CUDAwarmup completed and
reflection reached64tokens07:05:18. No manual recovery, CUDA OOM or401.
Run remains active;07:06 screenshot remains black and final PASS unverified.

Reflection completed07:13:51 and produced-story summary07:16:41. The writer
advanced to ledger line checks after another automatic reload recovered via
existing CPU offload. Repeated accelerate warning is retained; no manual
recovery or terminal failure.07:20 screenshot remains black. Audio/video/obs
and matching final episode file are still unqualified.
