<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core architecture is solid, but caching bugs, fail-open NVML checks, and missing file-size validations will cause silent degradation or hard crashes in production.

MUST-FIX BEFORE BUILD:
1. [render_driver.py] SLICE-CACHE STALENESS. Failure: Re-rendering a modified master audio reuses the stale temp WAV because the cache key ignores file modification time. Signal: Lip-sync matches the old audio; look-QA fails. Fix: Add `os.path.getmtime(master_path)` and `os.path.getsize(master_path)` to the `hashlib.sha256` key in `_slice_master_audio`.
2. [motion_common.py] NVML UNAVAILABLE. Failure: On a machine with broken NVML, `vram_used_mb()` returns `None`, bypassing `assert_vram_within_ceiling` and causing a hard CUDA OOM mid-render. Signal: `OomSignal` or unhandled CUDA OOM mid-sample. Fix: In `eng_ltx_av.py`'s `assert_usable`, explicitly require `_GR.nvml_available()` to be True (fail-closed for this specific heavy lane).
3. [ARCHITECTURE] PARTIAL/CORRUPT DOWNLOADS. Failure: A partial HF download or 0-byte symlink passes `os.path.exists`, crashing the engine during `load()`. Signal: `CRASH_BEFORE_LOAD` in the swap-log. Fix: `assert_usable` must check `os.path.getsize(path) > MIN_BYTES` (e.g., > 1MB) for the weights, not just existence.
4. [gpu_residency.py] CANCEL MID-SAMPLE. Failure: Operator cancels a render in ComfyUI Desktop; the executor thread dies but the ComfyUI process lives, so `_pid_alive` remains True and the AS-3 lease is permanently wedged until the 120s timeout. Signal: `LeaseTimeout` on the next render attempt. Fix: Document a strict v1 discipline in the M0 operator checklist: "Always restart ComfyUI Desktop after cancelling a render."

SHOULD-FIX:
5. [render_driver.py] FALLBACK STORM. Failure: `OTR_ENABLE_LTX_AV=1` is set but weights are missing; every beat fails `assert_usable`, degrading to HuMo, doubling render time silently. Signal: `[OTR video] LOUD FALLBACK:` spam in the swap-log. Fix: In `run_episode`, count `len(decisions)`. If it exceeds a threshold (e.g., > 50% of shots), log one screaming summary line: `STORM DETECTED: N fallbacks`.
6. [Milestones] GPU CONTENTION. Failure: M0 probe runs concurrently with the 30w acceptance render, causing mysterious OOMs. Signal: `OOM` failure kind in the swap-log during the acceptance window. Fix: The M0 launcher must actively check `:8000` liveness (the soak launcher pattern) before starting, aborting if ComfyUI is busy.
7. [render_driver.py] PAD-TAIL ABUSE. Failure: Upstream timing bug causes massive padding on every clip. Signal: `[ltx_av] pad-tail rendered=<n> target=<T>` >2s. Fix: Aggregate the padding in `build_clip_manifest` (sum of `frame_count - target_frame_count`) and log a summary warning if the episode total exceeds 5 seconds.
8. [TESTING] GOLDEN-FIXTURE ROT. Failure: Dark-lane goldens break on unrelated driver changes, masking real bugs. Signal: Hash mismatches in `test_ltx_av_driver_wiring.py`. Fix: Scope the golden assertions to specific fields (`engine_id`, `text_prompt`, `canvas`) rather than full dictionary equality.

OPTIONAL / NICE-TO-HAVE:
9. [wrapper_bridge.py] RELOAD THRASH. Failure: Consecutive same-engine clips reload the 25GB model per clip because `_render_one` strictly brackets `prepare` and `teardown` per shot. Signal: Wall time > 10 min/clip in the M0 table. Fix: Accept the reload for v1 to honor the AS-3 lease and BUG-291 reclaim. Record the per-clip wall-time cost in the M0 sheet.
10. [Milestones] MODULE-CACHE STALENESS. Failure: Operator pulls new adapter code but doesn't restart ComfyUI Desktop; UI shows engines but executor uses stale code. Signal: `WrapperNodeMissing`. Fix: Add explicit restart discipline to the M0 checklist: "Desktop needs RESTART after git pull; headless boots fresh."

CUT THESE (over-engineering):
11. [render_driver.py] CAPTIONS/CREDITS/TIMELINE. Safe to cut any mitigation. The M4 grep `plan_timeline_segments` proves the timeline is positioned (absolute `start_s`), so padded tails overlap rather than shift subsequent clips.
12. [wrapper_bridge.py] DESKTOP NODE LAG. Safe to cut any mitigation. `assert_usable` self-gates per process. The operator will see the LOUD fallback log and know HuMo ran, preventing a false-positive look-QA.

[ASSUMPTION] I am assuming `os.path.getmtime` is available and safe to call on `master_path` in `_slice_master_audio`. I am assuming `eng_ltx_av.py` has an `assert_usable` method that can be modified to check `_GR.nvml_available()`.