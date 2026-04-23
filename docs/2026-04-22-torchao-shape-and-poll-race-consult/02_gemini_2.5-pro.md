# Gemini 2.5 Pro, 2.6s

BUG 1 -
1. The `RuntimeError` is caused by a tensor shape mismatch due to loading a `torchao` v1 quantized model with a `torchao` v2 runtime, where the internal tensor layout or quantization scheme for activations has changed.
2. Re-quantize the FLUX model using `torchao` v2. If that's not feasible, the minimal fix is to downgrade `torchao` to a version compatible with v1 quantized models, e.g., `pip install torchao==0.1.0` (or the specific version that aligns with the v1 config).
3. Downgrading `torchao` might introduce incompatibilities with other libraries or prevent access to newer `torchao` features.
4. Verify by running `FluxPipeline(prompt, ...)` and confirming no `RuntimeError` occurs and the pipeline executes successfully.

BUG 2 -
1. The `VisualPoll` unblocks prematurely because `flux_anchor` writes `STATUS_READY` upon its stage completion, even when errors occurred and subsequent pipeline stages are pending.
2. Modify `flux_anchor` to only write `STATUS_RUNNING` if it's part of a chained pipeline, and have `video_stack` write the final `STATUS_READY` only after all its sub-stages (including `ltx_motion` and `wan21_loop`) have completed successfully.
3. This fix requires careful coordination of status writes across multiple stages, potentially introducing new race conditions if not implemented robustly.
4. Verify by observing the log sequence: `VisualPoll` should not unblock until `[sidecar:xxx] <sidecar exited rc=0>` is logged, indicating the entire sidecar process has finished.
