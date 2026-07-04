VERDICT: build-ready as-is? no. The ffmpeg crop expression includes an invalid `eval=frame` parameter which makes ffmpeg crash and breaks the credits roll rendering completely.

MUST-FIX BEFORE BUILD:
1. [F2] Defect: In `nodes/otr_credits_roll.py` line 917, the ffmpeg crop filter expression specifies `:eval=frame`. The FFmpeg `crop` filter does not support the `eval` option (its parameters `x` and `y` are evaluated per-frame by default). Passing this option causes FFmpeg to exit with an error (`Option not found`), crashing the entire credits rendering pipeline.
Fix: Remove `:eval=frame` from the crop string in `nodes/otr_credits_roll.py` line 917, changing it from `f"[2:v]crop=w={_sc(_COL3_W, h)}:h={view_h}:x=0:y='{yexpr}':eval=frame[sc];"` to `f"[2:v]crop=w={_sc(_COL3_W, h)}:h={view_h}:x=0:y='{yexpr}'[sc];"`.
2. [Tests] Defect: In `tests/test_credits_roll_spec.py` lines 318-328, the test `test_console_clip_renders_silent_and_declares_duration` fails to implement the visual movement assertions specified in the test plan ("assert frame(after LEAD) != frame(mid)"). It only checks frame counts and silence. Furthermore, it does not verify that the crop filter string uses valid, non-crashing options.
Fix: Add frame extraction and image comparison inside `test_console_clip_renders_silent_and_declares_duration` using FFmpeg to extract two frames (one at `t = _LEAD_HOLD_S + 0.5` and another at `t = dur / 2`) and assert that their visual contents (e.g., md5 sum or pixel difference) are not identical to prove scrolling motion.

SHOULD-FIX:
1. [F3] Defect: In `nodes/otr_credits_roll.py` line 264, the production ledger grid looks up `sysd.get("gpu_vram")`. However, `nodes/_otr_sys_specs.py` line 186 maps this system property to the `"vram"` key in the dictionary returned by `collect_system_specs()`. This key mismatch causes the lookup to always return `None` and fall back to the string `"GPU"`, displaying `"X.Y GiB of GPU"` in the ledger instead of the actual total GPU VRAM.
Fix: Change `sysd.get("gpu_vram")` to `sysd.get("vram")` in `nodes/otr_credits_roll.py` line 264.
2. [F3] Defect: In `nodes/otr_credits_roll.py` line 281, the scroll-render system block looks up `sysd.get("host")`. However, `nodes/_otr_sys_specs.py` line 191 defines the machine host name as `"hostname"`. This mismatch causes the host check to fall back to the OS info, duplicating the OS information and omitting the hostname.
Fix: Change `sysd.get("host")` to `sysd.get("hostname")` in `nodes/otr_credits_roll.py` line 281.

OPTIONAL / NICE-TO-HAVE:
- [F1] [ASSUMPTION] Although the new always-roll model correctly scrolls short transcripts, a very short transcript (e.g. 1-2 lines) will spend most of its duration scrolling through empty space before entering and after exiting the viewport. If this feels too empty, a threshold check or minimum text height parameter could be added to bypass rolling for ultra-short clips, but this is optional as the current always-roll specification is successfully implemented.

CUT THESE (over-engineering):
- None.
