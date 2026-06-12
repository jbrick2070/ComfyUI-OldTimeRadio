<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

RANKED KILL-LIST

1. NVML UNAVAILABLE -> VRAM ceiling becomes fiction.
Failure: On a bad driver / missing pynvml box, LTX-AV runs the heaviest lane while `motion_common.assert_vram_within_ceiling()` silently no-ops and teardown ignores `wait_until_below_mb(False)`, so M0/M4 can “pass” without ever enforcing 14.5 GB.
Signal: M0 preflight prints `nvml_available=False`; any M0 row with `probe_used_mb=0` plus nonzero render work is invalid. Grounding: `gpu_residency.probe_used_mb()` returns `0` on NVML failure; `motion_common.vram_used_mb()` returns `None`; `assert_vram_within_ceiling()` returns `None`.
Mitigation: Fail closed for this lane: M0 launcher and `assert_usable`/preflight must abort LTX-AV when `gpu_residency.nvml_available()` is false. Do not allow “CPU-box no-op” semantics for M0/M4 LTX-AV.

2. FALLBACK STORM -> missing weights silently doubles cost and ships all-floor/all-HuMo episodes.
Failure: `OTR_ENABLE_LTX_AV=1` with bad/missing weights makes every LTX-AV shot walk the fallback chain; episodes still complete because `render_shot()` keeps falling back to the floor, so the failure hides behind success.
Signal: Per-shot existing line: `[OTR video] LOUD FALLBACK: ... engine 'ltx_av_talk' -> ... reason=dependency_missing ...`; durable count in `ledger['video']['runtime_fallback_decisions']`. Grounding: `render_shot()` appends decisions and logs `retry_taxonomy.format_swap_log()`.
Mitigation: Add an episode-level storm detector on existing surfaces: if `runtime_fallback_decisions` contains `>= N` records with the same `from_engine in {ltx_av_talk, ltx_av_music}` in one episode, emit one screaming summary line and tracker count, e.g. `[ltx_av] FALLBACK_STORM from_engine=ltx_av_talk count=7 episode=...`. M4 must grep/fail on that line/count unless the test intentionally forces fallback.

3. RELOAD THRASH -> 3 LTX-AV clips pay 3 full model loads.
Failure: Consecutive same-engine clips still call `assert_usable -> prepare -> render_clip -> canonicalize -> teardown` per clip, so the 13–16 GiB transformer plus 13.2 GB encoder can reload every beat.
Signal: M0 episode wall time: 3-clip/30w episode exceeds the pass bar while single-clip row is acceptable; trace shows repeated same-engine attempts in order. Grounding: `_render_one()` always calls `eng.prepare()` and `eng.teardown()` in `finally`; `run_episode()` iterates ledger order and does not group/order by engine; `MotionEngineBase.prepare()` acquires lease and `load()`s; `teardown()` unloads/releases.
Mitigation: v1 cheapest answer: accept per-clip reload and record it explicitly in M0: include per-clip and 3-clip episode wall, plus “reload-per-clip accepted/failed” note. Do not attempt keep-resident in v1 unless wrapper_bridge/adapter policy is explicitly changed; current AS-3 lifecycle is per `_render_one()`.

4. CANCEL MID-SAMPLE -> live Desktop process keeps poisoned GPU / stale lease.
Failure: Operator cancels Comfy Desktop during transformer sampling; if the exception unwinds normally, `_render_one()` runs teardown, but if the executor/thread/GPU state is corrupted inside a still-live process, the lease may remain or VRAM may stay resident.
Signal: After cancel, `gpu_residency.is_held()` true with live owner PID, or NVML still above 14500 MB; next render hits `LeaseTimeout` or unexplained CUDA failure. Grounding: `_render_one()` has `finally: eng.teardown(prepared)` only after `prepared` exists; `MotionEngineBase.teardown()` releases and waits but ignores the boolean result; `gpu_residency.acquire()` only reclaims stale locks for dead PIDs, not live poisoned processes.
Mitigation: v1 discipline is operator-side: “after any Desktop cancel during LTX-AV sampling, restart ComfyUI before next render.” Put it in M0 checklist and Desktop error/runbook, not just campaign notes. Add a post-cancel check: lease absent + `wait_until_below_mb(14500)` true before continuing.

5. PARTIAL/CORRUPT DOWNLOADS / BROKEN SYMLINKS -> assert passes path existence but model load fails later per beat.
Failure: HF resume or cache+symlink leaves a truncated GGUF/encoder/projection/VAE; every LTX-AV shot fails at load/render and falls back, looking like normal degradation.
Signal: Earliest should be M0 inventory: resolved realpath, `exists`, file size, optional sha/hash per artifact; otherwise first signal is `LOUD FALLBACK ... reason=dependency_missing` or `invalid_dag` from wrapper load. Grounding: pass06 says cache+symlink pattern; grounding only shows generic `stage_into_comfy_input()` checks `os.path.exists()` for staged inputs, not weight integrity. VERIFY-AT-BUILD: actual `eng_ltx_av.assert_usable` weight checks.
Mitigation: Add minimum file-size/hash gate to the LTX-AV weights probe: encoder / transformer / projection / VAE each names expected artifact and resolved realpath. Broken symlink must fail before render. If hash is too heavy for v1, enforce exact/known byte size from M0 inventory plus realpath exists.

6. GPU CONTENTION WITH ACTIVE ACCEPTANCE WINDOW -> M0 and 30w acceptance render both fail mysteriously.
Failure: “After the acceptance window” is only a schedule note; if M0 starts while another Comfy render is live and that render does not hold the AS-3 lease, both compete for VRAM.
Signal: M0 idle-preload row already above baseline / `probe_used_mb` high before load; Comfy `:8000` active queue/history shows running job; lease may not help if other job is not an OTR heavy engine. Grounding: `gpu_residency` serializes only participants taking the lease; no M0 launcher liveness check is shown.
Mitigation: M0 launcher must hard-check before first pull/render: no active Comfy job on `:8000` VERIFY-AT-BUILD endpoint, no held OTR lease, and NVML idle below a defined threshold. Schedule note alone is insufficient.

7. SLICE-CACHE STALENESS -> rerendered master audio reuses old beat slices.
Failure: A master WAV/MP4 is regenerated at the same path; `_slice_master_audio()` cache key is only `(start_s, dur_s, master_path)`, so stale temp WAVs are reused.
Signal: Test: write master A at path, slice; overwrite same path with master B same beat timing; second slice path is identical and audio bytes remain A. Grounding: `_slice_master_audio()` key is `("%.6f|%.6f|%s" % (start_s, dur_s, master_path))`; cache hit returns existing file if size > 0.
Mitigation: Change key in `_slice_master_audio()` to include `os.path.getmtime_ns(master_path)` and `os.path.getsize(master_path)` at minimum; better include ledger `master_audio_sha256` if available. Cheapest code fix is mtime+size.

8. PAD-TAIL ABUSE -> every beat becomes 19.9s render plus frozen tail.
Failure: Upstream timing bug makes most targets exceed cap; LTX-AV renders capped clips and pads by last frame, producing long frozen tails while only per-clip LOUD lines exist.
Signal: Existing planned marker `[ltx_av] pad-tail rendered=<n> target=<T>`; aggregate missing. Grounding: pass02 defines this LOUD line; no aggregate shown in grounding.
Mitigation: Reuse the fallback-storm detector pattern: per episode count pad-tail events and total padded frames/seconds. M4 fails if count > 0 for normal smoke, or if padded seconds exceed threshold. Emit one summary: `[ltx_av] PAD_TAIL_STORM count=... padded_s=...`.

9. DESKTOP NODE LAG (#13194/#13308) -> look-QA Desktop and production headless use different node availability.
Failure: Desktop lacks `LTXVReferenceAudio` while headless has it, or vice versa; operator approves one behavior but production self-gates/falls back differently.
Signal: Per-process node gate fails with `WrapperNodeMissing ... install the wrapper + restart ComfyUI`; M0 node presence matrix differs Desktop vs headless. Grounding: `wrapper_bridge.resolve_graph_classes()` reads current process `NODE_CLASS_MAPPINGS`; `_render_one()` calls `eng.assert_usable()` in the render process. VERIFY-AT-BUILD: exact LTX-AV node class names and assert_usable gate.
Mitigation: Runtime self-gate is sufficient for safety, not parity. Add M0 hard gate: Desktop and headless must both list the same required LTX-AV node classes, or the lane is blocked for look-QA/ship.

10. MODULE-CACHE STALENESS -> Desktop shows old adapter code after file update.
Failure: New `eng_ltx_av.py` is on disk but Comfy Desktop keeps old Python modules loaded; dropdown/registry behavior and render behavior diverge until restart.
Signal: New engine missing from registry in Desktop despite file present, or old assert_usable error text; headless fresh boot differs. Grounding: wrapper_bridge missing-node error already says “install the wrapper + restart ComfyUI”; pass01 says registration is import-time. VERIFY-AT-BUILD: adapter registration import path.
Mitigation: Put “RESTART Comfy Desktop after installing/updating LTX-AV code or custom nodes” in the M0 checklist and in every missing-node / missing-engine error message. Headless boots fresh; Desktop does not.

11. CAPTIONS / CREDITS / TIMELINE SHIFT -> padded/trimmed clips desync captions or credits duration.
Failure: If compositor uses actual padded clip duration instead of target timing/master duration, captions drift and credits-tail cap may be evaluated against video length instead of MASTER-WAV.
Signal: M4 grep/gate must prove: final ffprobe video duration == MASTER-WAV duration within tolerance; caption ledger/node-93 max end <= MASTER-WAV duration; clip manifest `total_target_frames` equals expected master duration frames; no fallback to sequential timeline mode. Grounding: `build_clip_manifest()` records both `frame_count` and `target_frame_count`, plus `start_s`; pass02 says trim to T / pad by last frame. VERIFY-AT-BUILD: compositor uses `target_frame_count`/`start_s`, not `frame_count`, for timeline placement.
Mitigation: Add an M4 timeline assertion script over manifest + caption ledger + ffprobe. Do not rely on visual review.

12. GOLDEN-FIXTURE ROT -> dark-lane goldens become rubber stamps.
Failure: Driver/platform changes alter irrelevant serialized fields; developers mechanically update `fixtures/ltx_av_dark/`, so the guard no longer protects LTX-AV wiring.
Signal: Golden diffs dominated by paths, timestamps, ordering noise, or prompt hash churn instead of semantic fields. Grounding: pass05 says dark-lane GOLDEN FIXTURES and CPU structural goldens.
Mitigation: Scope goldens to semantic fields only: role, engine_id, family, fallback chain, required request fields, force-map behavior, synthetic slice gating, prompt source/length class. Regeneration requires reviewer note saying which semantic contract changed.

13. REAL OOM MID-EPISODE -> fallback taxonomy catches the exception but CUDA context remains unhealthy.
Failure: A true CUDA OOM during LTX-AV sampling raises, fallback walks to floor, but allocator/context remains fragmented and next heavy render fails or breaches ceiling.
Signal: `LOUD FALLBACK ... reason=oom` followed by `wait_until_below_mb(14500)` false / next shot `GraphExecutionError` or `LeaseTimeout`; M4 NVML grep after each shot remains high. Grounding: taxonomy classifies OOM hard; `MotionEngineBase.teardown()` waits but ignores failure; `_render_one()` suppresses teardown exceptions.
Mitigation: For LTX-AV v1, after any real OOM fallback, require a post-teardown NVML-below-ceiling check before next heavy shot; if false, emit screaming line and instruct restart. VERIFY-AT-BUILD where render node can observe the failed wait.

SHOULD-CONSIDER

1. Add one episode summary block to the existing tracker/ledger surface:
`fallback_counts_by_from_engine`, `pad_tail_count`, `pad_tail_total_s`, `nvml_available`, `max_vram_mb`, `lease_wait_s`, `final_engine_histogram`.

2. M0 should include a “negative install” drill: flag on, deliberately move one required artifact, confirm exactly one storm summary appears and the episode is marked degraded, not silently accepted.

3. M0 should record “same-engine consecutive clip cost” separately from single-clip cost, because current lifecycle guarantees per-clip teardown/reload.

4. Add a Desktop/headless parity table to `M0_RESULTS.md`: process, Comfy version/build, required node class presence, adapter module version/hash, selected artifact realpaths.

OPEN-QUESTIONS

1. VERIFY-AT-BUILD: exact `eng_ltx_av.assert_usable()` behavior for weight path, size/hash, node class names, Sage gate, and `av_dims`.

2. VERIFY-AT-BUILD: whether any M0 launcher already has the soak-style `:8000` active-job/liveness check. None is shown in grounding.

3. VERIFY-AT-BUILD: compositor/timeline code path for `build_clip_manifest()` rows: does it place by `start_s`/`target_frame_count`, or can actual `frame_count` from padded clips affect duration?

4. VERIFY-AT-BUILD: whether LTX-AV graph uses `wrapper_bridge.run_graph(free_after_use=True)` and what nodes are kept; grounding only proves `free_after_use` frees intermediates inside one graph, not residency across clips.

5. VERIFY-AT-BUILD: whether Comfy Desktop cancel raises a normal Python exception through `_render_one()` or interrupts outside the `finally` path. Current safe v1 assumption should be restart-after-cancel.