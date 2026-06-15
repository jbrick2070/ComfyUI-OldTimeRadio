<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The architecture is sound and integrates cleanly with the existing model-agnostic dispatchers, but the video lease-skip targets the wrong layer, a contradiction exists in the build order, and the cost guard leaks credits on failed renders.

MUST-FIX BEFORE BUILD:
1. **[Section 4.2] Video lease-skip targets the wrong layer.** The plan says to apply the lease-skip branch at the "video render lease site" (implying the dispatcher). However, for video engines, the AS-3 lease is acquired *inside* `MotionEngineBase.prepare()` (see `motion_common.py`). If cloud video engines inherit `MotionEngineBase` without overriding this, they will block on the local GPU lease.
   **Fix:** Modify `MotionEngineBase.prepare()` to check `if getattr(self, "is_network", False): return {"engine_id": self.name, "lease": None, "patchers": self._patchers}`. `MotionEngineBase.teardown()` already safely skips release/NVML-probe if `lease` is `None`.
2. **[Section 8] Contradictory network marker in build order.** Section 4.1 explicitly mandates `is_network = True` and forbids adding a new `declared_isolation` enum to avoid modifying `motion_common` constants. However, Section 8 (S1) contradicts this, instructing the coder to build a `declared_isolation="network"` marker.
   **Fix:** Update Section 8 (S1) to say "dispatcher lease-skip + `is_network = True` marker".
3. **[Section 4.3] Cost guard leaks credits on failed renders.** The plan reserves the cloud cost in `ledger["billing"]` *before* the render. If the API call fails (e.g., network timeout, 5xx error), the dispatcher catches the exception and falls back to the radio floor, but the cost remains deducted. This permanently inflates the episode's spent total and will spuriously trip the `OTR_CLOUD_CREDIT_CEILING`.
   **Fix:** Have `reserve_cloud_cost` return the reserved amount, and in the dispatcher