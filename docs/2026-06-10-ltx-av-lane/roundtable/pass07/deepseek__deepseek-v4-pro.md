<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. The plan lacks several cheap, high-impact detectors and gates that would prevent painful production failures (fallback storm, NVML fail-open, corrupt weights, stale slices, pad-tail storm). These are small, additive mitigations consistent with the locked design.

MUST-FIX BEFORE BUILD:
1. [FALLBACK STORM] No aggregate detector for mass fallbacks from ltx_av engines. If weights are absent, every beat degrades silently (only per-beat LOUD log, no summary). Earliest signal: per-beat "LOUD FALLBACK" lines, but operator may miss them. Mitigation: in run_episode (or a post-render hook in run_real_episode), after all shots, count fallback decisions per origin engine family; if any ltx_av family exceeds a threshold (e.g., all beats), emit a single screaming summary line like "[OTR video] FALLBACK STORM: ltx_av_talk degraded on 12/12 beats". Use the existing fallback decision records (render_driver.py: decisions list). Add a function `_warn_fallback_storm(decisions, total_beats)` called at end of run_episode. (Section: pass07 item 1, grounding: render_driver.py render_shot collects decisions, run_episode returns trace but no aggregate.)

2. [NVML UNAVAILABLE] If nvml_available() is False, the 14.5 GB ceiling is silently unenforced (assert_vram_within_ceiling becomes a no-op). For the heaviest lane, this is dangerous. Earliest signal: no VRAM breach warning, possible OOM later. Mitigation: in eng_ltx_av.py's assert_usable, call gpu_residency.nvml_available() and raise EngineUnusable with reason if False (fail-closed). This lane requires NVML. (Section: pass07 item 7, grounding: motion_common.py assert_vram_within_ceiling returns None if NVML absent; gpu_residency.py nvml_available exists.)

3. [PARTIAL/CORRUPT DOWNLOADS] Weights probe in assert_usable likely checks file existence but not integrity. A truncated download could pass and cause cryptic failures. Earliest signal: render fails with GraphExecutionError or garbage output. Mitigation: in the weights probe (eng_ltx_av.py assert_usable), for each critical artifact, verify file size against a constant recorded from M0 (e.g., expected bytes). Optionally check a quick hash of first 1KB. Raise MISSING_MODEL with detail if mismatch. (Section: pass07 item 4, plan says "file size/hash check in assert_usable weights probe".)

4. [SLICE-CACHE STALENESS] _slice_master_audio caches by (start,dur,path). If the master audio file is re-rendered with the same path, stale slices are served. Earliest signal: audio-video sync drift, no explicit error. Mitigation: include file mtime and size in the cache key. Change the key to `hashlib.sha256(f"{start_s}|{dur_s}|{master_path}|{os.path.getmtime(master_path)}|{os.path.getsize(master_path)}")`. (Section: pass07 item 11, grounding: render_driver.py _slice_master_audio.)

5. [PAD-TAIL ABUSE] Per-clip pad-tail >2s is logged LOUD, but no aggregate. A systematic timing bug causing every clip to pad would waste render time and distort durations unnoticed. Earliest signal: per-clip log lines. Mitigation: in run_episode, after all shots, count clips where pad-tail exceeded threshold (e.g., >2s) and if count > some fraction (e.g., >50%), emit a summary warning "[ltx_av] pad-tail storm: X/Y clips padded >2s". This can reuse the pad-tail marker already in the clip metadata. (Section: pass07 item 8, plan mentions "same storm detector?".)

SHOULD-FIX:
6. [CANCEL MID-SAMPLE] Operator cancels in ComfyUI during transformer phase; teardown may not run, lease held by same live PID, VRAM not freed. Next render attempt times out on lease. Earliest signal: LeaseTimeout in log. Mitigation: document in operator docs (M0 checklist) that cancelling a render requires a ComfyUI restart to clear the lease and VRAM. No code change needed; the lease timeout prevents poisoned GPU reuse. (Section: pass07 item 3, grounding: gpu_residency.py acquire checks _pid_alive, will timeout if same PID holds lock.)

7. [GPU CONTENTION] M0 probe runs concurrently with the 30w acceptance render, causing OOM or failures. Earliest signal: M0 fails with OOM/timeout. Mitigation: in the M0 launcher script, before starting, check if ComfyUI is busy (e.g., query :8000/status or check for active render via API). If busy, abort with message "acceptance render in progress; retry after window". This mirrors the soak launcher pattern. (Section: pass07 item 6, plan says "after the acceptance window" but no enforcement.)

8. [GOLDEN-FIXTURE ROT] Dark-lane golden fixtures break on unrelated driver changes, developers update mechanically, guard erodes. Earliest signal: test failures that are "updated golden" without review. Mitigation: in the golden fixture tests, compare only the fields that matter (engine_id, family, prompt, etc.) and ignore volatile fields. Add a policy: any golden update must be reviewed and explained in commit message. (Section: pass07 item 10, plan suggests "scope the golden to the fields that matter".)

9. [RELOAD THRASH] Consecutive same-engine clips reload the model each time (prepare/teardown per clip). Earliest signal: per-clip render time near the 10 min wall limit. Mitigation: accept for v1; document in M0 that reload cost is expected, and measure per-clip time to ensure it stays within PASS/WARN bars. No code change. (Section: pass07 item 2, plan already states "accept the reload, record the cost in M0".)

10. [MODULE-CACHE STALENESS] New adapter code on disk, Comfy Desktop still running old module. Earliest signal: render may fail with missing attributes or use old behavior. Mitigation: add a bold note in the adapter docstring and M0 operator checklist: "After updating eng_ltx_av.py, you MUST restart ComfyUI Desktop. Headless boots fresh." (Section: pass07 item 5, plan asks "where is it WRITTEN".)

11. [CAPTIONS/CREDITS/TIMELINE] Padded clips could shift captions or credits gate if compositor uses actual frame count instead of target. Earliest signal: captions out of sync, credits cut off. Mitigation: M4 test must include a check that the final composited video duration equals master audio duration, and that captions align with audio. The plan's M4 gates already include "obs playable AAC only" and likely timeline checks; ensure a specific grep for this interaction. (Section: pass07 item 9, plan says "Name the M4 grep that proves none.")

12. [DESKTOP NODE LAG] Desktop build missing LTXVReferenceAudio while headless has it. Earliest signal: assert_usable fails on Desktop with MISSING_MODEL. Mitigation: assert_usable node gate runs per-process, so each build self-gates. This is sufficient; no additional guard needed. (Section: pass07 item 12, plan confirms.)

OPTIONAL / NICE-TO-HAVE:
- Add a post-episode summary line for total fallback count and pad-tail count, even if below storm threshold, for operator visibility.
- In M0, include a step to verify NVML availability and file sizes of downloaded weights.

CUT THESE (over-engineering):
- None identified; all proposed mitigations are minimal and additive.

OPEN-QUESTIONS:
- For fallback storm threshold: what count/percentage triggers the summary? Suggest >50% of beats for a given engine family.
- For slice cache key: is mtime stable across network shares? If not, consider file size only as a cheaper alternative.
- For GPU contention check: does ComfyUI expose a reliable "busy" endpoint? If not, a simple schedule note may suffice.