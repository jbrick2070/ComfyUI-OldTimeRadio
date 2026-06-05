<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: yes-with-fixes. The plan is architecturally coherent but contains several self-contradictions, a critical invariant conflict, and multiple gaps that will cause build failures or silent correctness breaks before Wave 3 is reached.

---

## MUST-FIX BEFORE BUILD

**1. [Invariants I-1 vs CLEAN-BREAK directive] Fatal self-contradiction on the default engine.**

I-1 states "legacy is the permanent fallback" and "shipped workflow defaults every engine ON to the new/best one." The CLEAN-BREAK directive explicitly supersedes I-1 and removes permanent legacy fallback. Yet I-1 is listed under "Invariants -- never violate (verified against the code)" and is never struck from that section. C-6 still reads "internal build default stays legacy until F, flipped at promotion I" which is consistent with CLEAN-BREAK, but I-1 as written contradicts it. Any implementer reading the invariant list will implement a permanent fallback that CLEAN-BREAK forbids. Fix: strike I-1 from the invariants block entirely and replace with a single sentence pointing to CLEAN-BREAK. Do not leave both standing.

**2. [eng_indextts2.py / eng_chatterbox.py -- `default_roles` vs EXECUTION-PLAN engine matrix] Both shipped adapters have `default_roles = ()` but the plan promotes them to defaults.**

The grounded code shows `IndexTTS2Engine.default_roles = ()` and `ChatterboxEngine.default_roles = ()`. `registry.engines_for_role` sorts default-first by `role not in default_roles`. `assert_usable` skips the flag check only when `role in default_roles`. If these stay `()` after promotion (I), the shipped workflow will silently require `OTR_ENABLE_INDEXTTS2=1` to be set, or `assert_usable` will raise `GATED_BY_FLAG` at queue time -- exactly the named error the plan says a default engine must never produce (C-6, C-7). Fix: promotion step I must flip `default_roles` to `("char_voice",)` for IndexTTS2 and update `requires_flag` to `None`; same for Chatterbox on announcer. This is a code change, not just a workflow widget change, and it must be gated on F passing.

**3. [eng_stable_audio.py vs EXECUTION-PLAN "SA3-native" / D5 / Verify-at-build] `eng_stable_audio.py` loads `stable_audio_tools.get_pretrained_model` -- a PyPI package the blocker section says "won't resolve" -- but the plan's target for SA3 is the ComfyUI-native `Comfy-Org/stable-audio-3` node set.**

The grounded adapter calls `from stable_audio_tools import get_pretrained_model` and `from stable_audio_tools.inference.generation import generate_diffusion_cond`. This is the conflicting PyPI path, not the native ComfyUI path. The plan says "wraps ComfyUI's native SA3 nodes (`Comfy-Org/stable-audio-3`), not a custom stable-audio-tools path" (D5, Verify-at-build). The adapter as written will fail to import on the target box. Fix: the `eng_stable_audio_3.py` adapter (noted as "not-yet-written" in the blocker section) must call ComfyUI's internal SA3 node API, not `stable_audio_tools`. The existing `eng_stable_audio.py` must be kept as the `stable_audio_open` fallback adapter only, with its name and `OTR_ENABLE_STABLE_AUDIO` flag scoped accordingly, and must never be promoted to default.

**4. [eng_chatterbox.py `load()` / blocker section] `ChatterboxTTS.from_pretrained(device="cuda")` is called unconditionally in `load()` with no offline guard.**

The blocker section states the hard rule is "100% local + offline-first at RUN time (no network during execute)." `from_pretrained` on HuggingFace-backed models will attempt a network fetch if the cache is cold or the revision has changed. There is no `local_files_only=True` or equivalent. Fix: pass `local_files_only=True` (or the Chatterbox equivalent) and raise a named `EngineUnusable(MISSING_MODEL)` on failure, consistent with C-7.

**5. [eng_indextts2.py `generate_voice` / G1 determinism] Seed is set via `torch.manual_seed` + `torch.cuda.manual_seed_all` globally AND a per-call `torch.Generator` is passed -- but `supports_external_generator` is never checked before passing the generator kwarg.**

`supported_kwargs` filters kwargs by inspecting the function signature. If `IndexTTS2.infer` does not accept a `generator` parameter (which the GPU pilot (F) has not yet confirmed -- the code comments say "GPU-VALIDATE (F)"), `supported_kwargs` silently drops it. The global `torch.manual_seed` call then becomes the only seed path, which is process-global and not isolated per-line. This means two concurrent lines (or any intervening CUDA op) can corrupt each other's seed state. The plan's G1 section says engines whose `supports_external_generator` is False are "disqualified from bit-exact mode" -- but the adapter does not enforce this disqualification; it silently falls back to global seeding without raising or setting a flag. Fix: after `supported_kwargs`, check whether `generator` was retained; if not and `bit_exact` mode is active, raise `EngineUnusable(INCOMPATIBLE_PROFILE, "engine does not bind external generator")`. [ASSUMPTION: `deterministic_inference` context manager exists and exposes a `bit_exact` flag -- verify.]

---

## SHOULD-FIX

**1. [EXECUTION-PLAN I-8 vs engine matrix] `musicgen` is listed as `commercial_clean=false` (CC-BY-NC-4.0) but the engine matrix says it is the music default #2 and "still renders" with a warning. The `assert_release_clean` gate (E.5) does a manifest scan and "fail-closed on missing `commercial_clean` boolean." Verify that the release gate's warn-not-block path for known-false values is actually implemented and tested -- the grounded code does not show `_otr_release_gate.py`. If the gate is fail-closed on `false` rather than only on missing/unknown, MusicGen will stop-ship silently.**

Fix: confirm the three-state logic (true=silent, false=warn, missing=stop) is explicitly coded in `_otr_release_gate.py` and covered by a test that asserts MusicGen renders with a warning rather than raising.

**2. [Wave R0a step (d) vs CLEAN-BREAK] R0a step (d) seeds legacy audio nodes, and step (e) creates `legacy_invocation_manifest.json`. CLEAN-BREAK supersedes "R0a steps (d)/(e)/(f)" and says to capture `baseline_v2` from the NEW engines instead. But step (f) still says "render-twice legacy bit-identity, capture `baseline_v2_audio_legacy`." The naming `baseline_v2_audio_legacy` implies a legacy baseline, contradicting CLEAN-BREAK's instruction to capture `baseline_v2` from new engines.**

Fix: rename the R0a capture to `baseline_v2_audio_new` and explicitly note that legacy seeding (step d) and the legacy manifest (step e) are retained only for the transitional render-twice comparison, not as a permanent fallback reference. Or, per CLEAN-BREAK, drop (d)/(e)/(f) entirely from R0a and move the new-engine baseline capture to Wave 3/G1.

**3. [I-7 / EXECUTION-PLAN Verify-at-build] The plan requires a `gate_signal` edge from the post-unload assembler into the first video loader to prevent co-loading OOM, but this edge is described only in prose. The Wave 2b workflow JSON build step does not list this edge in its explicit link migration tables.**

Fix: add this edge explicitly to the Wave 2b link table with source node, output slot, target node, and input slot identified by name, not just described as "the first video loader."

**4. [eng_chatterbox.py `generate_voice`] The `generate` call passes `text` as a positional argument (`self._model.generate(text, **kwargs)`) but also passes `cfg=0.5` and `cfg_weight=0.5` as separate kwargs via `supported_kwargs`. If the real `ChatterboxTTS.generate` signature uses only one of these names, the other is silently dropped. More critically, if it uses neither and uses a different name, both are dropped and the CFG guidance is lost with no error.**

Fix: after the GPU pilot (F) confirms the real signature, remove the duplicate and add an assertion that at least one of `cfg`/`cfg_weight` was retained in `kwargs`.

**