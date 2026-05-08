# Round A -- ChatGPT (gpt-5.5) elapsed=67.2s

I can’t verify exact line numbers from here, so I’m grading the five snippets as shown and referring to the likely files/symbols: `batch_humo_render.py::{_is_oom_exception,_hard_reset_cuda_context,HumoSoakCapReached,render loop}` and the audit script’s `FAIL_PATTERNS`.

| # | Element | Fix-needed probability | Badge | One-line reasoning | Failure mode to watch in next FULL soak |
|---:|---|---:|:---:|---|---|
| 1 | `_is_oom_exception` detector | **8%** | **GREEN** | Covers the two important PyTorch OOM surfaces: `torch.OutOfMemoryError` and legacy/stringified `RuntimeError: ... out of memory ...`. | OOM gets wrapped by ComfyUI/PyTorch as a higher-level exception whose top-level message lacks `"out of memory"`, so the reset does not fire. |
| 2 | `_hard_reset_cuda_context` cleanup chain | **28%** | **AMBER** | The order is directionally right, but this is not a true CUDA context reset and may not release allocator pressure if live references/tracebacks/model globals still hold tensors. | After a caught OOM, log says hard-reset ran OK, but `memory_reserved` / allocated pool stays high and the next HuMo line OOMs or aborts anyway. |
| 3 | Caught-OOM wiring in render loop | **18%** | **AMBER** | The control flow is sane, including `HumoSoakCapReached` re-raise, but the reset runs inside the `except` while the active exception traceback may still retain frames/tensors. | Reset fires, but because the exception/traceback or per-line locals are still live during cleanup, VRAM is not actually returned before the next line starts. |
| 4 | `humo_max_lines_per_process` structured cap | **10%** | **GREEN** | As described, this is a low-risk safety valve: default off, triggers only after successful `rendered += 1`, and pairs cleanly with ledger resume. | Cap is reached and ledger write appears successful, but the follow-up run either re-renders the last completed line or skips one due to an off-by-one/resume-state mismatch. |
| 5 | Audit-script `"Fatal Python error: Aborted"` signal | **14%** | **GREEN** | Exact string match is crude but appropriate for the observed failure signature and much better than mtime-quiet being interpreted as PASS. | Fatal abort occurs but the banner is not captured in the watched log stream, or Windows/Python reports a different fatal signature such as access violation without this exact string. |

## Where I would push back

1. **Calling Element 2 a “hard reset” is overstated.**  
   It is a best-effort Comfy/PyTorch unload + cache drain. That may be good enough, but it is not a CUDA context reset. If allocator drift is caused by retained references, traceback-held tensors, extension-level state, or unreleased Comfy model patches, this chain can report success while leaving the process in a bad memory state.

2. **The “survive allocator drift” claim is only partially proven by alarm plumbing.**  
   The detector and watcher are good alarm plumbing. The actual survival property still depends almost entirely on whether `unload_all_models() + gc.collect() + soft_empty_cache() + empty_cache()` meaningfully reduces reserved/active VRAM after HuMo OOMs on this exact stack.

## Load-bearing weak spot

**Element 2: `_hard_reset_cuda_context`.**

That is the single most production-critical uncertainty. The rest of the change mostly detects, routes, or reports failures. Element 2 is the only part that actually tries to make the same Python process safe to continue after a HuMo OOM. If it does not reduce allocator pressure in practice, the next FULL soak will likely show the same pattern: caught OOMs, apparent recovery, then a later fatal abort.
