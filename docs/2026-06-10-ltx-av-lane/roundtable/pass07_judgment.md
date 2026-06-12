# pass07 (pre-mortem) judgment -- Claude, judge + panelist

Ranked kill-list CONVERGED 4/4 on membership; ordering merged by the
judge (likelihood x damage, grounded severity):

1. NVML FAIL-OPEN (GPT's #1, grounded: probe_used_mb()->0 on failure,
   vram_used_mb()->None, assert_vram_within_ceiling no-ops): for
   ltx_av_* ONLY, assert_usable REQUIRES gpu_residency.nvml_available()
   -- fail CLOSED with a named reason (heaviest lane, opt-in, can
   afford strictness; existing engines unchanged). M0 rule: any row
   with probe_used_mb==0 and real render work is INVALID.
2. FALLBACK STORM: episode-end EPISODE SUMMARY block on existing
   surfaces (counts from runtime_fallback_decisions):
   fallback_counts_by_from_engine, pad_tail_count + padded_s,
   nvml_available, max_vram_mb, final_engine_histogram. STORM line when
   >= 2 degrades share an ltx_av_* origin in one episode:
   "[ltx_av] FALLBACK_STORM from=<engine> count=<n>/<beats>". M4 greps
   fail on STORM unless the test forces it. M0 adds GPT's
   NEGATIVE-INSTALL DRILL: flag on, one artifact moved aside, expect
   EXACTLY ONE storm summary + degraded-marked episode.
3. RELOAD THRASH (grounded: _render_one brackets prepare/teardown per
   clip; run_episode never groups by engine): ACCEPT per-clip reload in
   v1 (the AS-3/BUG-291 lifecycle is the contract); M0 adds a
   TWO-CONSECUTIVE-CLIP row -- the marginal cost is the gate number;
   keep-resident is a future wrapper_bridge policy question, NOT v1.
4. CANCEL MID-SAMPLE (grounded: lease reclaims only DEAD PIDs; a live
   wedged process holds it to the ~120s timeout; teardown waits are
   ignored-bool): v1 discipline = "restart ComfyUI after any mid-render
   cancel" written in THREE places (M0 checklist, adapter docstring,
   ship notes) + post-cancel/post-OOM check: before the next heavy
   shot, lease absent AND wait_until_below_mb(14500) true, else ONE
   screaming line instructing restart (GPT's unhealthy-CUDA-context
   item folded here).
5. PARTIAL/CORRUPT WEIGHTS: assert_usable weight probe = resolved
   realpath EXISTS (broken symlink fails) + size >= per-artifact FLOOR
   constants derived from the judge-verified sizes (e.g. transformer
   >= 12 GiB, encoder >= 10 GiB, video VAE >= 1 GiB) -- floors catch
   truncation without quant-brittleness (judge over GPT's exact-bytes);
   full hashes stay M0-only.
6. GPU CONTENTION: M0 launcher HARD-CHECKS before any pull/render --
   :8000 active-job liveness (soak pattern), no held OTR lease, NVML
   idle below threshold; abort with "acceptance render in progress".
   Schedule notes alone rejected (4/4).
7. SLICE-CACHE STALENESS: _slice_master_audio key += master mtime_ns +
   size (one line; also fixes the latent HuMo-slice case -- the ONE
   shared-path bugfix this sprint ships, with its own unit test; ledger
   sha optional later).
8. PAD-TAIL ABUSE: counted in the same EPISODE SUMMARY;
   "[ltx_av] PAD_TAIL_STORM count=<n> padded_s=<s>" when >= 2 clips pad
   > 2s; M4 normal smoke fails on any PAD_TAIL_STORM.
9. DESKTOP NODE LAG: runtime self-gate per process is SAFETY-sufficient
   (each build asserts its own NODE_CLASS_MAPPINGS); PARITY is a SHIP
   gate -- M0_RESULTS.md parity table (build, version, node classes,
   adapter module presence, artifact realpaths) must MATCH for
   look-QA/ship (GPT over Gemini's cut: look-QA authenticity matters).
10. MODULE-CACHE STALENESS: restart discipline written in the M0
    checklist + adapter docstring + the missing-node error text
    (wrapper_bridge's "install the wrapper + restart ComfyUI" wording
    precedent); headless boots fresh.
11. CAPTIONS/CREDITS/TIMELINE: LOW RISK -- Gemini grounded absolute
    placement (plan_timeline_segments places by start_s; padded tails
    overlap, never shift). Mitigation = existing M4 greps
    (duration_check OK vs MASTER-WAV; captions events line) + one
    manifest note comparing frame_count vs target_frame_count; GPT's
    dedicated timeline-assertion script REJECTED as over-build.
12. GOLDEN-FIXTURE ROT: goldens compare a SEMANTIC PROJECTION
    (engine_id, family, role, canvas, text_prompt source/length-class,
    audio_ref presence, asset_refs keys, timing, seed) -- never full
    dicts; regeneration requires a commit-note naming the changed
    contract.

## REJECTED

- Dedicated timeline assertion script (over-build; existing greps
  cover). Keep-resident-across-clips in v1 (lifecycle contract).
  Exact-byte weight checks (quant-brittle). 50%-threshold storm rule
  (>=2 same-origin is tighter and equivalent on 2-3-beat episodes).

## VERIFY-AT-BUILD

- free_after_use scope (frees intra-graph intermediates; residency
  across clips is the lifecycle's job -- confirm no hidden cache).
- Desktop cancel path: normal exception through _render_one's finally
  vs external interrupt (v1 assumes restart-after-cancel regardless).
- :8000 busy-endpoint reliability for the launcher gate.
- Compositor placement function name + manifest fields (Gemini's
  plan_timeline_segments cite).
