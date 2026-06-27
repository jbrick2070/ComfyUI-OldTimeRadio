VERDICT: yes-with-fixes. The plan correctly addresses the decode settings and filter order, but creates a visual scaling/sharpening discrepancy for 3D character directory-clips and lacks environment variables to override the decode settings for the manual whole-clip mode.

MUST-FIX BEFORE BUILD:
1. [nodes/otr_silent_composite.py ~428-439] **Directory-Clip Scaling & Sharpening Inconsistency**:
   - Defect: The plan modifies `_seg_vf` to use `:flags=lanczos` and `unsharp=5:5:0.4:5:5:0.0` for scaling/sharpening standard clips and floors. However, directory-clips (3D character layers) are scaled/padded in `_encode_segment_from_dir` using manual `bg_filter` (lines 428-430, 435-439) and `fg_filter` (line 444) filter strings without Lanczos or unsharp filters. This results in a visual inconsistency where character segment backgrounds/foregrounds are noticeably blurrier.
   - Fix: Update `bg_filter` and `fg_filter` strings in `_encode_segment_from_dir` to also apply `:flags=lanczos` and append `,unsharp=5:5:0.4:5:5:0.0` immediately after their `scale` filters.
2. [nodes/_otr_video_engines/eng_ltx_av.py ~556-559] **Manual "Whole-Clip" Option Lacks Env Config**:
   - Defect: The plan locks the open question "whole-clip is documented manual" in `## Judgment on Codex r4`, but hardcodes `temporal_size: 128` and `temporal_overlap: 32` as integer literals in the `decode` node dict inside `eng_ltx_av.py`. This forces operators to edit python source files directly to enable whole-clip mode, violating the repository's pattern of utilizing environment overrides.
   - Fix: Define `_DECODE_TEMPORAL_SIZE = int(os.environ.get("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "128"))` and `_DECODE_TEMPORAL_OVERLAP = int(os.environ.get("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", "32"))` at the module level in `eng_ltx_av.py`, and wire them into the `decode` node inputs.

SHOULD-FIX:
1. [nodes/otr_silent_composite.py ~319-325] **Hardcoded Composite Unsharp Amount**:
   - Defect: The sharpening amount of `0.4` is hardcoded directly into the FFmpeg filter string. The plan identifies tweaking this amount as a future code tweak, but hardcoding prevents clean tuning/runtime evaluation of scaling sharpness.
   - Fix: Extract the sharpen amount to an environment variable: `os.environ.get("OTR_COMPOSITE_UNSHARP_AMOUNT", "0.4")`.
2. [nodes/_otr_video_engines/eng_ltx_av.py ~90] **Volatile VRAM Headroom**:
   - Defect: The plan assumes LTX-AV's peak VRAM is `14272 MB` against a ceiling of `14500 MB` [ASSUMPTION], leaving a slim `228 MB` headroom. Given the documented run-to-run variance of `~135 MB`, system load changes or desktop application spikes are highly likely to exceed the limit and cause silent spills to system memory.
   - Fix: Add documentation advising operators to increase `OTR_LTX_AV_RESERVE_VRAM_GB` from `4.0` to `4.5` if OOM spills occur under typical desktop usage.

OPTIONAL / NICE-TO-HAVE:
1. [Pre-ship VERIFY-AT-BUILD] **FFmpeg Filter Availability Assert**:
   - Add a pre-flight or test check to assert that the target `ffmpeg` binary supports the `unsharp` filter, avoiding cryptic ffmpeg execution failures.

CUT THESE:
1. [## Optional (not blocking)] **Bakeoff Manifest Hardening**:
   - Why: Hardening the bakeoff manifest with specific developer-leg model filenames (`build_ltx_av_q_bakeoff_workflow.py:62-65` values) is fragile. It couples tests to env-specific model names that will break when users use customized checkpoints, causing false-positive suite failures.
