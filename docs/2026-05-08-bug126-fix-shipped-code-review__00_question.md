# Question -- 2026-05-08

# Code review: BUG-LOCAL-126 fix shipped to v2.0-alpha

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
