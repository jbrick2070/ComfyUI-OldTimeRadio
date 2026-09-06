# Canonical visual-weight readiness candidate

Development correction for PBUG-20260906-01, September 6, 2026. **Not a registry
release, clean-install qualification, GPU benchmark, or successful episode.**
The failed four-hour 4060 act remains a failure. No installed file/model was
changed while developing this candidate.

## What changes

The canonical already wires WorkflowValidator63 to Writer1 and VideoDirector87.
The validator now uses hidden live PROMPT/UNIQUE_ID inputs to check selected
visual weights before returning its gate. No serialized widgets, links,
canonical JSON, model choices, precision policy or ProcGen mode change.
Readiness is rechecked on every live queue, including after model deletion.
Structural validation still runs first; bypassing that structural check does
not bypass asset readiness.

The planner uses only directly gated live director/writer nodes, not stale
saved dropdowns. Video aliases/effective routes use the existing shared
resolvers. All-replay writers skip live selections because the bundle owns
them; mixed replay/live writers and dynamic/custom selections are refused
before downloading. Other literal engines retain their existing adapter
checks and receive an explicit automatic-coverage warning.

## Exact automatic download coverage

Only these missing default tokens may be provisioned; repeated roles are
deduplicated. The ZImage default remains BF16, not a new int8/FP8/NVFP4 choice.

| Selected engine | Native category | Default file | Public source |
|---|---|---|---|
| z_image_turbo | diffusion_models | z_image_turbo_bf16.safetensors | Comfy-Org/z_image_turbo, split_files/diffusion_models |
| z_image_turbo | text_encoders | qwen_3_4b.safetensors | Comfy-Org/z_image_turbo, split_files/text_encoders |
| z_image_turbo | vae | ae.safetensors | Comfy-Org/z_image_turbo, split_files/vae |
| ltx098_low_video | checkpoints | ltxv-2b-0.9.8-distilled.safetensors | Lightricks/LTX-Video |
| ltx098_low_video | text_encoders | t5xxl_fp16.safetensors | comfyanonymous/flux_text_encoders |

Existing adapter picks and native loader roots win. Complete existing stores
perform no network call or download. This preserves a 5080's existing NVFP4
choice as well as other installed selections; it does not qualify that choice
on Ada. Missing nondefault weights, empty files, stale links, and contradictory
explicit/native paths fail early and are not repaired or overwritten.

## Transfer behavior and visible evidence

The Comfy console receives `[OTR.assets]` records. All missing sources get
anonymous metadata checks before transfers begin: HEAD at the public source,
then HEAD at the reported commit. Both must agree on exact positive size and
64-hex content SHA-256. The pinned commit, bytes and hash are logged, followed
by the total missing-file count/bytes. Sizes are **not known from this offline
development work** and must be recorded from the actual first GUI download.

GET uses the pinned Hub URL, finite30-second socket timeout and HTTPS-only
redirects. No token acquisition, package installer, extra node pack, model
substitution, automatic retry or resume. Cancellation is checked before and
between chunks; a blocked socket operation may delay Stop until its timeout.
Console byte progress is emitted at least every5 seconds while chunks arrive,
plus initial/final counts and elapsed time. Stalled reads do not emit progress.

The first registered native model root is the destination; there is no silent
writable-root fallback. A per-file free-space check requires the declared
bytes plus512MiB margin. This is not an aggregate reservation for all files:
earlier verified files may remain if a later file lacks space. A unique
same-directory `.part` receives the stream, verified by exact size/SHA-256 and
fsync. Atomic no-clobber hard-link publication requires a supporting filesystem;
unsupported filesystems fail, with no copy/overwrite fallback. The owned temp
is normally removed; cleanup failures retain error details and may leave that
temp on disk. The final is an ordinary file, not a symlink. An OS advisory lock
is released even after process death; its small `.lock` file deliberately
remains and is not itself evidence of an active transfer. Crashed-process
orphan temps are never guessed at/deleted by another attempt.

After publication the native loader and adapter are rechecked. An existing
file that appeared during a race is explicitly unverified, not credited as
our verified transfer. Existing files are preserved and checked for nonzero
availability only, not rehashed. `READY` therefore means supported selected
files resolve, **not** GPU capacity, successful rendering, obs publication,
license approval, or a finished episode.

## Verification and remaining gates

Offline Windows results:21 downloader tests +10 validator tests +40 planner,
native resolver, metadata, and mocked integration tests passed. The earlier
seven NF4 CPU-offload tests also passed. Tests use tiny synthetic streams/files
and stub runtime imports; no model download/load or GPU execution occurred.
Review-driven cleanup fixes cover mid-stream BaseException, reusable locks,
unlock-error precedence, transport closure and signed-URL error redaction.

Full pytest could not run (`No module named pytest`); configured Bug Bible
checkout/test interpreter absent. No test dependency was installed. The new
runtime files live under `nodes/`, outside existing registry exclusions; actual
built-package contents and scan status remain unverified. Neither pyproject nor
the version/tag/release workflow was changed.

Next: review/release the candidate through the normal distribution path,
restore a usable ComfyUI window, then install through its normal UI and run one
act. Record every download, size, wait and error. Require RESULT SUCCESS,
obs_publish OK and the actual episode file. An OOM or authorization failure is
a terminal finding, not an invitation to tune or obtain a token. No separate
4060 JSON is justified by these file-readiness checks alone.

## September6 live development follow-up

User deferred registry publication and authorized the reviewed installed-source
patch as an explicitly hand-patched development trial. GUI one-act Run started
04:50:36.149PDT. Real anonymous metadata resolved all5 defaults/36,818,738,352
bytes before the writer. By04:58, ZImage12,309,866,400bytes/232.6s,
Qwen8,044,982,048bytes/196.2s and VAE335,304,388bytes/7.3s were verified;
LTX was downloading. This is not all-assets READY or a render/publish pass.

Native aliases matter for the guide: this Desktop instance prepends its shared
`unet` directory to diffusion-model roots via legacy alias mapping. The first
verified ZImage final was there, not in a folder literally named diffusion_models.
Never prescribe a manual move; native resolution remains the authority.

Observed UX issue: Comfy's queue remains at0percent during transfer even while
the Logs panel shows bytes advancing. A subsequent DEV-only correction uses
the standard Comfy ProgressBar in the current execution context, aggregating
only missing bytes over all files. It reserves the final1percent until every
receipt/native recheck and final cancellation check passes; progress-hook
interruptions propagate. Exact console bytes remain unchanged and completion
logs include the native path. Total-only constructor avoids requiring the newer
node_id keyword. No new thread, JSON/widget, network endpoint or GPU behavior.
This follow-up is not applied to the active run; offline/GUI qualification of
the progress presentation is recorded separately from download receipts.

Progress follow-up verification: six new tests and extended existing assertions
passed with77visual +7NF4 tests (84total). Independent scoped review found no
actionable issues. Native hook is source-verified on installed ComfyUI0.34.5;
total-only constructor compatibility is stub-tested, not a matrix of physical
hosts/frontend versions. Live progress presentation is still unqualified because
the ongoing trial intentionally retains the pre-progress installed candidate.
By04:59:58, LTX6,340,744,492bytes also verified in124.6s; T5 transfer then began.
