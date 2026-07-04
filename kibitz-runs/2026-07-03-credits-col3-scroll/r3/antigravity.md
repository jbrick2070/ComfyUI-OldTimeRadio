VERDICT: yes-with-fixes. The system graph and timeline math are sound, but the landed SYSTEM scroll block drops detail fields, ffmpeg path resolution ignores standard environment variables, and the backdrop planner can crash if the final clip is a directory.

MUST-FIX BEFORE BUILD:
1. [F3 SYSTEM in the scroll] The new `[SYSTEM]` block in the scroll layout drops essential detail fields from the repository's existing SYSTEM contract. `nodes/video_engine.py:1162-1169` outputs Host + OS, CPU + cores, RAM + peak, and GPU + VRAM. However, `sys_grid` in `nodes/otr_credits_roll.py:280-289` omits CPU cores (`cpu_cores`), RAM peak (`ram_peak`), and total VRAM (`vram` or system total memory).
Fix: Expand `sys_grid` in `nodes/otr_credits_roll.py:280-289` to match the format of `nodes/video_engine.py:1162-1169` as follows:
```python
    sys_grid = [
        ("Host:", "%s · %s" % (sysd.get("hostname") or "?",
                               sysd.get("os") or "(unknown)")),
        ("CPU:", "%s (%s)" % (sysd.get("cpu") or "(unknown)",
                              sysd.get("cpu_cores") or "(unknown)")),
        ("RAM:", "%s (peak %s)" % (sysd.get("ram") or "(unknown)",
                                   sysd.get("ram_peak") or "(unknown)")),
        ("GPU:", "%s (%s VRAM)" % (sysd.get("gpu") or "(unknown)",
                                   sysd.get("vram") or "GPU")),
        ("CUDA:", "%s · torch %s · Python %s" % (
            sysd.get("cuda") or "?", sysd.get("torch") or "?",
            sysd.get("python") or "?")),
    ]
```
Add a test in `tests/test_credits_roll_spec.py` that mocks `cr._sys_specs` with sentinel strings (e.g., `"vram_sentinel"`, `"cpu_cores_sentinel"`, `"ram_peak_sentinel"`) and asserts that they appear in the generated `system` layout block.

2. [External system integration] The backdrop planner can select a directory path if the last existing clip is a directory clip. For example, if a 3D `mesh_stage` engine is used (ctype `"directory"`, see `nodes/_otr_video_engines/render_driver.py:2281-2286`), `plan_backdrop` in `nodes/otr_credits_roll.py:823-835` will select the directory path. This will cause ffmpeg to crash with a read/seek error when executing `-stream_loop -1 -i <directory_path>`.
Fix: In `nodes/otr_credits_roll.py:827-829`, filter out directory paths by validating that the path is a file using `os.path.isfile`:
```python
    rows = [r for r in ((clip_manifest or {}).get("clips") or [])
            if isinstance(r, dict) and r.get("exists") and r.get("path") and os.path.isfile(r["path"])]
```

3. [Configuration and environment propagation] `nodes/otr_credits_roll.py:841-852` hardcodes ffmpeg/ffprobe lookup using `shutil.which("ffmpeg")` and `shutil.which("ffprobe")` directly. This ignores the standard `OTR_FFMPEG` and `OTR_FFPROBE` environment variables utilized by other components (e.g. `nodes/_otr_shared/content_oracle.py:73`, `nodes/_otr_video_engines/eng_viz_rainbow.py:20`) and the ComfyUI widget configurations. If the operator runs in an environment where ffmpeg is at a custom path but not globally on the system `PATH`, the credits node will crash.
Fix: Update `_ffmpeg_bin` and `_ffprobe_bin` in `nodes/otr_credits_roll.py` to respect environment variables:
```python
def _ffmpeg_bin() -> str:
    env_p = os.environ.get("OTR_FFMPEG")
    p = env_p if (env_p and (shutil.which(env_p) or os.path.isfile(env_p))) else shutil.which("ffmpeg")
    if not p:
        raise CreditsDataError("ffmpeg not found -- set OTR_FFMPEG or add to PATH")
    return p

def _ffprobe_bin() -> str:
    env_p = os.environ.get("OTR_FFPROBE")
    if not env_p and os.environ.get("OTR_FFMPEG"):
        derived = os.environ.get("OTR_FFMPEG").replace("ffmpeg", "ffprobe")
        if shutil.which(derived) or os.path.isfile(derived):
            env_p = derived
    p = env_p if (env_p and (shutil.which(env_p) or os.path.isfile(env_p))) else shutil.which("ffprobe")
    if not p:
        raise CreditsDataError("ffprobe not found -- set OTR_FFPROBE or add to PATH")
    return p
```

SHOULD-FIX:
1. [Panel-caught bugs fixed] The test suite in `tests/test_credits_roll_spec.py` does not verify the key mappings for the newly fixed fields: total system VRAM (`vram` vs legacy `gpu_vram`) and hostname (`hostname` vs legacy `host`).
Fix: Add an assertion to the test suite that mocks `_sys_specs()` to verify these fields are correctly formatted and rendered.
2. [F3 SYSTEM in the scroll] Stale comments and docstrings in both the node and test files still refer to the old layout hierarchy. Specifically:
   - `nodes/otr_credits_roll.py:10-15` and `tests/test_credits_roll_spec.py:4-8` state that `[SYSTEM]` is a static column 1+2 dashboard block.
   - `nodes/otr_credits_roll.py:674-676` states that `render_scroll_canvas` covers `(STORY SPINE -> full transcript -> intercept -> diagnostic)`.
Fix: Update the file headers and docstrings to document the new scroll layout sequence: `SYSTEM -> STORY SPINE -> CLASSIFIED TRANSCRIPT -> SOURCE INTERCEPT -> DIAGNOSTIC`.

OPTIONAL / NICE-TO-HAVE:
1. [Interface contracts] The function signature for `compute_credits_duration_s(roll_px: int, view_h: int, pps: float = _SCROLL_PPS)` in `nodes/otr_credits_roll.py:802` includes `view_h`, but the parameter is never used within the function body. Remove `view_h` or prefix it as `_view_h` to satisfy code checkers.

CUT THESE (over-engineering):
None. All proposed corrections are minimal adjustments to ensure contract compliance, environment config propagation, and path sanity.

[ASSUMPTION] We assume that any `mesh_stage` or other directory-based clip generator writes all frames to its target directory during the execution of `OTR_VideoRenderBatch`, making `exists` check and frame directory summary logic valid when `plan_backdrop` is executed downstream.
