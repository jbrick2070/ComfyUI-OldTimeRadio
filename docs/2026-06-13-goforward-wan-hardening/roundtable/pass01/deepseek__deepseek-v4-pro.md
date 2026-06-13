<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The sweep's engine-fallback detection is entirely absent (Q1), the Wan engine lacks any mechanism to avoid loading multiple model copies across beats within one episode (guaranteed OOM for real multi-beat episodes), and the unresolved CS‑3 scoping blocks Phase 2. These three alone make the plan not build-ready.

MUST-FIX BEFORE BUILD
1. [Q1 / Section 3 item 4 & sweep script]  
   The coverage sweep passes `expect_engine=""` to `run_leg`, which only logs the histogram and accepts any fallback (soak line 440‑465). Therefore a leg whose engine silently falls back to `still_kenburns` still scores PASS, rendering the GATE‑A acceptance meaningless.  
   Fix: change the sweep to pass `expect_engine=engine` for every leg. To avoid false‑fails when a rotated slot (e.g. music_visual) gets no beat in a 30‑word episode, the sweep must also guarantee that each leg exercises its engine. The safest route is to increase `target_words` to a value that reliably spawns all beat types, or to use a dedicated seed/episode template that always yields at least one beat per slot.

2. [Section 4 / eng_wan_i2v.render_clip]  
   `render_clip` loads the full model every call and retains the patcher model via `free_after_use` and `keep={"unet",...}`. When the same engine is used for multiple beats inside one episode (e.g., all character beats), each call loads a fresh copy of the unet. The patcher bucket only prevents exact id duplicates; because each load creates a new object, the model accumulates, breaking the single‑resident‑heavy‑engine invariant and causing OOM on any episode with more than one beat for that engine.  
   Fix: Either cache the loaded model in engine state and reuse it across `render_clip` calls (skip the unet‑loader on subsequent beats), or explicitly unload the previous model before loading the next. The first option is preferred; the graph must accept a pre‑loaded model instead of running the loader node for repeats.

3. [CS‑3 / Section 4 & Section 5]  
   The plan describes Wan + HuMo as “co‑staging VRAM in one episode” and says that “can bust 16 GB.” In reality the render driver runs engines sequentially per beat, not co‑resident, so the true risk is whether the inter‑beat reclaim fully drains one heavy engine before the next loads. Reframe CS‑3 to “prove the inter‑beat reclaim fully drains Wan before a HuMo beat loads, and vice‑versa, within one episode.” This unblocks Phase 2 scoping. Until resolved, Phase 2 cannot proceed.

4. [Q3 / registry.py wan_i2v capability + soak]  
   `CAPABILITIES["wan_i2v"].vram_estimate_mb` is 14000, but the code comment in `eng_wan_i2v.render_clip` records the bare‑smoke peak at 14499 MB. The availability/tier‑fit logic uses the estimate and may admit the engine into a profile that later busts the 14.5 GB ceiling.  
   Fix: set the estimate to the measured peak (~14500) and document that `free_after_use=True` is **mandatory** (without it the engine is unsafe). Also verify that the render‑phase NVML peak assertion in the soak uses the driver’s peak, which includes model loading, so it catches spikes; if not, add a check of the whole‑run machine‑wide peak.

5. [Q6 / registry.py]  
   `CAPABILITIES["wan_i2v"].model_requirements` lists `"wan2.1-i2v"` but the engine actually uses Wan 2.2. The stale label would cause the S5 wizard to advertise the wrong asset.  
   Fix: change to the correct Wan 2.2 model identifier.

6. [Section 4 / Phase 2 dependency]  
   The forward order lists “Wan 2.2 video engine (section 4). Phase 2 engine leg” as item 3, but section 1 states it is BLOCKED on CS‑3 scoping. The plan must explicitly call out the blockage so the operator does not attempt to start the work before the call. Update the forward order to note the dependency and keep the item non‑active.

SHOULD-FIX
1. [Q2 / Section 4 8GB tier]  
   The `wan_ti2v` engine must never accidentally load the Wan 2.1 VAE. Add a fail‑closed check in its `_loader_names` or `assert_usable` that raises `EngineUnusable` if the resolved VAE basename matches the 2.1 VAE (e.g., “wan_2.1_vae.safetensors”).

2. [Q8 / Item 4 matrix breadScope]  
   Item 4 describes three additive leg‑sets (visual‑engines, writer‑LLM, voice‑variation), but `otr_coverage_sweep.py` only enumerates the visual‑engine set. The “coverage sweep GREEN” definition is therefore incomplete. Either point each set at its actual harness (or run_combo_matrix) or mark the missing sets as TODO with a realistic estimate.

3. [eng_wan_i2v._materialize_init_image]  
   The Pillow‑fallback path (`except Exception`) stages the raw image and relies on `WanImageToVideo`’s internal resize. While the comment says “still no stretch”, the internal resize of the stock Wan node may not preserve aspect in a way that satisfies the N9 invariant.  
   Should‑fix: require Pillow and fail loudly if it is missing, or provide an explicit letter‑box resampling even without Pillow.

4. [Sweep harness isolation]  
   The sweep runs multiple legs sequentially on the same server without unloading or restarting. Heavy engines from a previous leg may remain loaded and corrupt the next leg’s VRAM peak or availability. Add a server restart (or an explicit model‑unload step) between sweep legs that change heavy engines.

5. [Section 4 / Wan 2.2 risk]  
   The single‑expert (low‑noise) I2V may produce weak camera motion and fail the eyeball gate. Surface this as an explicit Phase 2 risk with the two‑expert Path B as the mitigation if motion reads weak. This does not block the current build but should be documented in section 4.

OPTIONAL / NICE-TO-HAVE
- Add a doc note that `scripts/otr_coverage_sweep.py` is the full path (Q7).
- Harness polish items (output‑tree resolver, janitor sweep, heartbeat) are already in open tickets; no action needed now.
- The soak’s informational whole‑run NVML peak could be formatted more prominently but is not a defect.

CUT THESE (over-engineering)
- None identified. Every piece has a clear purpose.

[ASSUMPTION] The render driver actually reclaims models between beats (sequential residency). This cannot be verified from the provided excerpts; the CS‑3 reframe assumes it exists. [ASSUMPTION] The TI2V‑5B engine’s graph topology mirrors I2V except for a different latent node and VAE; this must be defined before building.