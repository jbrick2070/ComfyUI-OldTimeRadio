# Synthesis -- 2026-05-08

**Question:** # Code review: BUG-LOCAL-126 fix shipped to v2.0-alpha

## Background

Overnight FULL acceptance soak crashed with `Fatal Python error: Aborted`
after rendering 9 of 56 planned lines. Two prior caught HuMo OOMs in the
same run drifted PyTorch's CUDA pool from ~14 GiB in-use to ~24 GiB
allocated on a 15.92 GiB device, then the third sample fatal-aborted.
Stack landed in `batch_humo_render.py -> common_ksampler -> uni_pc.py`.
This commit (`3e231e8`) ships an alarm-plumbing fix in two parts plus a
watcher signal. Test floor 9/9 new + 33/33 LTX regression + 113/113 full
Phase 0+ suite, all green.

Stack: Windows, Python 3.12, RTX 5080 16 GB Blackwell, torch 2.10/CUDA 13,
SDPA + SageAttention. No cloud, no quantization tricks (per
`feedback_no_vram_dragons` carve-out: alarm plumbing only, NOT weight
streamer chasing).

## What I want from you

For each of the FIVE numbered code elements below, give me:
- **Per-element fix-needed probability % (0-100)**.
- **One-line reasoning**.
- **Verdict badge**: GREEN (<15%), AMBER (15-30%), RED (>30%).
- One concrete failure mode you'd watch for in the next FULL acceptance
  soak.

Then a short closing section:
- **Where would you push back?** (one or two strongest disagreements)
- **What's the load-bearing weak spot?** (single element most likely to
  fail in production)

NOT what I want: rewrites, "consider also", or suggestions to chase
weight-streamer / quantization / FA-2/3 paths. The whole point of this
fix is alarm plumbing -- if the cleanup chain doesn't actually relieve
allocator pressure, that's a real concern, but "ship FA3" is out of
scope.

**Apply skepticism.** Last two rounds caught real bugs AND produced false
alarms (e.g., VoiceSpec serialization that didn't exist, runtime_checkable
protocol "100% RED" that turned out theoretical). I'm grain-of-salting
this one and will verify each claim against actual code before acting.
If an element looks right to you, GREEN it. Don't manufacture risk.

## The five elements under review

### Element 1 -- `_is_oom_exception` detector

```python
def _is_oom_exception(exc: BaseException) -> bool:
    try:
        import torch
        oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(
            getattr(torch, "cuda", None), "OutOfMemoryError", None
        )
    except Exception:
        oom_cls = None
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "outofmemoryerror" in msg:
            return True
    return False
```

Detection covers both modern `torch.OutOfMemoryError` and the legacy
`RuntimeError(...)` with "out of memory" in the message. Both forms
appeared in the overnight log.

### Element 2 -- `_hard_reset_cuda_context` cleanup chain

```python
def _hard_reset_cuda_context() -> None:
    import gc
    steps_run, steps_failed = [], []

    try:
        import comfy.model_management as mm
    except Exception as exc:
        steps_failed.append(f"import comfy.model_management ({exc})")
        mm = None

    try:
        import torch
    except Exception as exc:
        steps_failed.append(f"import torch ({exc})")
        log.warning(...); return

    if mm is not None:
        try: mm.unload_all_models(); steps_run.append("unload_all_models")
        except Exception as exc: steps_failed.append(...)

    try: gc.collect(); steps_run.append("gc.collect")
    except Exception as exc: steps_failed.append(...)

    if mm is not None:
        try: mm.soft_empty_cache(force=True); steps_run.append(...)
        except Exception as exc: steps_failed.append(...)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(); steps_run.append(...)
    except Exception as exc: steps_failed.append(...)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); steps_run.append(...)
    except Exception as exc: steps_failed.append(...)

    try:
        if torch.cuda.is_available():
            torch.cuda.ipc_collect(); steps_run.append(...)
    except Exception as exc: steps_failed.append(...)

    if steps_failed:
        log.warning("[BatchHumoRender] cuda hard-reset partial: ran=%s; failed=%s",
                    steps_run, steps_failed)
    else:
        log.info("[BatchHumoRender] BUG-LOCAL-126 cuda hard-reset OK: %s",
                 ", ".join(steps_run))
```

Order matters: `unload_all_models` BEFORE `empty_cache` so the model
patches release VRAM, and only THEN does the allocator have free pages
to coalesce. Every step best-effort; helper never raises (recovery code
can't escalate the fault).

### Element 3 -- caught-OOM wiring in the render loop

```python
try:
    # ... per-line HuMo render: WanHuMoImageToVideo + KSampler +
    # VAEDecode + CreateVideo + SaveVideo, all inside chunk loop ...
    rendered += 1
    if humo_max_lines_per_process > 0 and rendered >= humo_max_lines_per_process:
        # persist ledger, raise HumoSoakCapReached
        ...
except HumoSoakCapReached:
    raise   # structured signal -- don't swallow
except Exception as exc:
    log.exception("[BatchHumoRender] line %s failed: %s", line_id, exc)
    report_lines.append(f"  {line_id}: FAILED ({exc})")
    if cuda_hard_reset_on_oom and _is_oom_exception(exc):
        log.warning("[BatchHumoRender] line %s hit OOM; running BUG-LOCAL-126 cuda hard-reset", line_id)
        _hard_reset_cuda_context()
```

Catch-all picks up the OOM, logs, and the conditional reset fires only
when `_is_oom_exception` returns True AND the new
`cuda_hard_reset_on_oom` widget is ON (default ON). The
`HumoSoakCapReached` re-raise pattern keeps that signal distinguishable
from generic faults.

### Element 4 -- `humo_max_lines_per_process` cap with structured exit

```python
class HumoSoakCapReached(RuntimeError):
    def __init__(self, lines_completed: int, cap: int):
        self.lines_completed = lines_completed
        self.cap = cap
        super().__init__(f"HuMo soak cap reached: rendered {lines_completed} of cap {cap}; ...")
```

Default 0 = disabled. When >0 and reached after `rendered += 1`:
1. Save ledger via existing per-clip incremental save path
2. Log a warning
3. Raise `HumoSoakCapReached`

Pairs with the existing `resume_from_ledger=True` flag so a follow-up
ComfyUI run picks up where this one stopped.

### Element 5 -- audit-script Aborted signal

```python
FAIL_PATTERNS = (
    "Non-monotonous DTS",
    "boomerang FAILED",
    re.compile(r"duration contract VIOLATED(?!.*audio C7 preserved)"),
    "derived ledger from .mp4 not found",
    "audio may be truncated",
    re.compile(r"\[BatchLTXRender\] \w+ failed:"),
    # NEW:
    "Fatal Python error: Aborted",
)
```

Plain string match. Watcher (`scripts/soak_watch.ps1`) consumes this
list to surface the verdict in `outputs/soak_status.txt`. Without it,
the overnight aborted run would have been reported as PASS by the
watcher pattern (mtime-quiet but not aborted).

## Constraints + non-goals

- All code is stdlib + torch + comfy.model_management. No new third-party
  dependencies.
- The BUG-126 fix is meant to *survive* allocator drift, not eliminate
  it. Eliminating drift is a follow-up if the alarm plumbing turns out
  insufficient (BUG-050-style chained-teardown investigation).
- The cap default is 0 = OFF so the new code only runs when explicitly
  configured. No behavior change on the default install.
- We are NOT going to chase weight streaming, FA2/3, or quantization
  fixes per `feedback_no_vram_dragons`.

Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
`v2.0-alpha`, head `3e231e8`).

---

## ChatGPT (gpt-5.5)

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

---

## Gemini (gemini-3.1-pro-preview-customtools)

**1. Core Verdict**
I AGREE with ChatGPT's core recommendation, particularly its excellent catch regarding traceback-retained locals during the exception block, but it missed a critical OS-level logging detail regarding how fatal aborts are surfaced.

**2. Factual Errors in ChatGPT's Answer**
*   **Element 2 Windows context:** ChatGPT evaluated `torch.cuda.ipc_collect()` as part of the valid cleanup chain. On Windows, CUDA IPC is not supported. Calling it is essentially dead code. Your `try/except` makes it harmless, but it contributes nothing to VRAM recovery on your RTX 5080 Windows workstation.

**3. Important That Was Omitted**
*   **C-Level `stderr` vs Python `logging` (Element 5):** "Fatal Python error: Aborted" is generated at the C level (often by PyTorch or the CUDA driver). It completely bypasses Python's standard `logging` module and writes directly to system `stderr`. If your `scripts/soak_watch.ps1` is only reading a standard ComfyUI `comfy.log` file, or if the PowerShell script doesn't explicitly redirect `stderr` to `stdout` (using `2>&1`), Element 5 is completely blind and will never see the abort message.
*   **The Trivial Fix for Element 3:** ChatGPT correctly identified that running the reset *inside* the `except Exception as exc:` block means the traceback (and all local variables/tensors in the failing frame) is still alive in memory. However, it omitted the fix: Python 3 automatically deletes the `exc` variable when the `except` block terminates. You must move the reset call *outside* the block to actually free the memory.

**4. Short Recommendation**
*   **Fix the Traceback Trap (Element 3):** Do not call `_hard_reset_cuda_context()` inside the `except` block. Instead, set a flag: `needs_reset = _is_oom_exception(exc)`. Then, *after* the `except` block concludes (and Python destroys the `exc` reference), check `if needs_reset:` and run the cleanup.
*   **Verify Stderr Capture (Element 5):** Check `scripts/soak_watch.ps1`. Ensure the ComfyUI launch command looks something like `python main.py 2>&1 | Tee-Object -FilePath outputs\soak_status.txt` so C-level fatal aborts are actually written to the file your regex is scanning.
*   **Lean on Element 4 (The Cap):** ComfyUI's VRAM fragmentation over 50+ heavy video-generation runs is notoriously difficult to defeat purely with `empty_cache()`. Your structured exit (`HumoSoakCapReached`) is your most reliable production tool. If the cap works, rely on it rather than fighting PyTorch's allocator drift.
*   **Keep Element 2 as-is (with adjusted expectations):** The order of operations in `_hard_reset_cuda_context` is correct (`unload` -> `gc` -> `empty_cache`). Just accept ChatGPT's warning: it is a soft cache drain, not a true CUDA context destroy/recreate.

**5. Uncertainties to Verify**
*   **Custom Node Caching:** I am uncertain if `mm.unload_all_models()` successfully releases VRAM held by the specific `WanHuMoImageToVideo` or `VAEDecode` nodes. Some custom ComfyUI video nodes instantiate their own global caches that bypass Comfy's native model management. If allocator drift continues, check if the HuMo nodes have their own specific `.offload()` or `.free()` methods.
*   **Logging Retention:** I am uncertain if your specific `log.exception(...)` implementation retains the traceback object in memory longer than the `except` block (e.g., if you have a custom logging handler that queues records in memory). If it does, `gc.collect()` might still fail to clear the tensors.

---

## NVIDIA ()



---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
