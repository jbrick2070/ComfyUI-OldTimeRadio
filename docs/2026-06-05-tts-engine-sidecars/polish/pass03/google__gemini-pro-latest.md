<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

An adversarial review of the Round-3 convergence pass reveals that while the Round-2 fixes are solid and the subprocess lifecycle is now robust, there are two remaining **MUST-FIX** defects in the dispatch core and path resolution that will break features at runtime.

### VERDICT
**yes-with-fixes**. The worker lifecycle is clean, but the dispatch core drops the delivery vector (breaking Chatterbox's exaggeration) and cross-platform path resolution is incomplete.

---

### MUST-FIX BEFORE BUILD

**1. [eng_chatterbox.py / eng_dia.py] Cross-platform path resolution failure**
**Defect:** If a ledger generated on Windows is rendered on Linux, the `voice_ref_path` will contain backslashes (e.g., `models\TTS\refs\ix_male.wav`). `_resolve_ref` in both adapters checks `os.path.isabs(ref)` and `os.path.exists(cand)` directly on the unnormalized string. On Linux, this fails to match the file, falling back to an invalid absolute path and failing the render. (Note: `_resolve_ref_to_disk` in the common node handles this correctly, but the adapters bypass it when the ledger already contains a path).
**Fix:** Normalize slashes at the very top of `_resolve_ref` in both `eng_chatterbox.py` and `eng_dia.py`:
```python
    def _resolve_ref(self, ref):
        if not ref:
            return ref
        ref = ref.replace("\\", "/")
        if os.path.isabs(ref):
            return ref
        # ... rest of function unchanged
```

**2. [_otr_voice_node_common.py] Dispatch core silently drops `delivery_vector`**
**Defect:** In `_render_per_line`, the `delivery_vector` parameter is hardcoded to `None` when calling both `prep()` and `adapter.generate_voice()`. This completely breaks Chatterbox's `exaggeration` feature (which relies on `delivery_vector.get("calm")`), silently rendering all lines at the default 0.5 exaggeration regardless of the