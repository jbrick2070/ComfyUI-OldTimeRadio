# NVIDIA (mistralai/mistral-nemotron) elapsed=6.1s

The safest mid-loop VRAM cleanup pattern for your scenario is to **explicitly call `torch.cuda.empty_cache()` between model inference steps**, while ensuring you're not holding unnecessary references to tensors. Here's the concrete recommendation:

1. **After SPINE 1 completes**, call:
   ```python
   torch.cuda.empty_cache()
   ```
   (This is already in your code, but ensure no tensors are being retained accidentally.)

2. **Before SPINE 2 starts**, add:
   ```python
   torch.cuda.synchronize()  # Ensures all CUDA ops complete before cleanup
   torch.cuda.empty_cache()
   ```

**Key File/Line**: In `comfy/manager.py` (or your custom node script), add this right before the second model inference call. If you're using `autocast` or `amp`, ensure it's not holding onto cached tensors.

**Uncertainty**: Blackwell (sm_120) is new, and some CUDA 13 drivers may have edge cases with `empty_cache()`. If crashes persist, try:
- Downgrading to CUDA 12.3 (if possible).
- Using `torch.cuda.reset_peak_memory_stats()` before `empty_cache()` for debugging.

This is the smallest change with the highest payoff. If the issue persists, deeper profiling (e.g., `nvprof`) may be needed, but this should resolve most cases.
