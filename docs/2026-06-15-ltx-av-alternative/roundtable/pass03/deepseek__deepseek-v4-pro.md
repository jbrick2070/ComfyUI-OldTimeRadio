<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: **no.** Central contradiction on frame-math snap direction + entire engine shape depends on M0 probe results not yet captured. Cannot be build-ready as written.

---

MUST-FIX BEFORE BUILD (severity order):

1. **[BUGS/RISKS C] vs [sprint plan, F3 dims validator] -- 8n+1 snap-direction CONTRADICTION.**
   Plan doc C says: "the SAME 8n+1 snap-DOWN (`((n-1)//8)*8+1`) -- do NOT diverge Lane A vs B frame math." Sprint plan (grounded) defines `next_8n1(n) = ((n+6)//8)*8 + 1` and states "snap UP -- the legacy eng_ltx_video :281 formula snaps DOWN; never copy it." The sprint plan's `render = min(next_8n1(T), LTX_AV_MAX_FRAMES)` uses snap-UP. The plan doc contradicts the design it claims to build on. **Fix:** pick one. If sprint plan is authoritative, rewrite C to say "use `next_8n1` (snap-UP) as defined in `av_dims.py`, deliberately diverging from Lane A's snap-DOWN" and explain why. If plan doc is authoritative, the sprint plan's `next_8n1` + `render = min(next_8n1(T), ...)` must be changed to snap-DOWN.

2. **[DECISIONS 1] / [M0 SPIKE] -- entire graph topology, terminal node class, video-only decode node, audio loader class, and viable low-VRAM artifact are UNKNOWN.**
   The plan specifies `_ltx_av_core`, two adapters, `assert_usable` gate order, audio slice handling, pad-tail, etc. -- but all graph-construction code (the actual A2V forward) references "the M0-captured A2V graph" and "the M0-captured node class." M0 has not run. If M0 finds OOM at the only feasible quant / the video-only decode node does not exist / the audio loader expects a different format, the architecture must change. **Fix:** either gate the entire M2+ ticket set behind an explicit M0 GO signal in the plan, or add a "M0 finding placeholder with failover" section that defines what changes if M0 parks the lane.

3. **[WIRING B, Prod JSON audit] -- conditional edit on an unseen file.**
   "CONFIRM options are registry/roles-driven ... If options are static arrays -> add the option + re-validate." The file `otr_scifi_16gb_full.json` is not in grounding excerpts. The plan cannot verify which branch applies. If the file uses static arrays, adding engine options without the registry/dropdown mechanism could break the UI or validation. **Fix:** request the JSON excerpt be added to grounding, or make the M3 ticket include a pre-edit audit step with explicit gating based on the file's actual structure.

4. **[DECISIONS 4] frozen-audio V-1 -- "the video-only decode uses the M0-captured node class (node-gated fail-closed) -- NOT a preselected LTXVSeparateAVLatent (ungrounded)."** But sprint plan says "graph terminates at the video VAEDecode -> IMAGE batch." Not the same thing. VAEDecode is a standard ComfyUI node; a "video-only decode" from an A2V model that produces a joint AV latent would be a different node (e.g., a separation+decode). This is unresolved and depends on M0. **Fix:** do not prescribe the terminal node name in the plan. State that M0 GRAPH SPEC will record the exact terminal class, and the M2 ticket wires whatever class M0 captures.

5. **[ARCHITECTURE A] assert_usable ordered gate step 6 -- "av_dims on request_template.canvas (None tolerated), violations re-raised as EngineUnusable (no raw ValueError)."** But sprint plan says `assert_ltx_dims` "RAISES (never rounds)." Which exception? If it raises `ValueError`, the `EngineUnusable` wrapper must catch it. The plan doesn't specify the catch. **Fix:** declare the exception type `assert_ltx_dims` raises and the exact catch-and-wrap location.

6. **[BUGS/RISKS C] / [sprint plan, pad-tail] -- "canonicalize TRIMS to exactly T, or PADS-BY-LAST-FRAME to T (cap case)."** Lane A caps the frame ask at `OTR_LTX_MAX_FRAMES` and lets the composite hold-fill. Lane B renders at `min(next_8n1(T), LTX_AV_MAX_FRAMES)` then canonicalize-pads. The plan does not explain why the mechanisms diverge or confirm the composite won't double-pad. **Fix:** add a sentence confirming the composite's hold-last-frame path is NOT applied to Lane B clips (or is idempotent when clip already at target length).

---

SHOULD-FIX:

1. **[SPRINT PLAN, F7/announcer portrait] -- "populate asset_refs["init_image"] from the shipped non-cast announcer portrait recorded in ledger["images"] (object id VERIFY-AT-BUILD)."** The object ID is ungrounded in this doc. If the ID is wrong or the ledger key changes, the alias silently fails and the chain walks. **Fix:** add the exact object ID + ledger path to the grounding excerpt, or make M3 ticket include a ledger grep.

2. **[SPRINT PLAN, encoder phasing] -- "acquire -> text encode -> reclaim_idle_models(...) -> load transformer -> sample."** `reclaim_idle_models` is called between encoder and transformer. If `_soft_free` is insufficient and `reclaim_idle_models` unloads the encoder, the conditioning tensors may be invalidated before the sampler runs (depending on how ComfyUI caches intermediates). **Verify:** can `free_after_use` in `wrapper_bridge.run_graph` coexist with manual `reclaim_idle_models` between phases without invalidating in-flight tensors? Not testable from grounding alone -- flag as assumption.

3. **[WIRING B] "Audio input: reuse the per-beat frozen-master slice; the slice fed to the model is padded/trimmed to the 8n+1 duration BEFORE generation."** Padding audio with silence or trimming mid-word will affect what the A2V model "hears." No quality threshold or test for this is defined. **Fix:** add a note that audio-pad/trim artifacts are accepted v1 risk, to be assessed in M4 look-QA.

4. **[ARCHITECTURE A] "ONE file `nodes/_otr_video_engines/eng_ltx_av.py`: module-level lazy `_ltx_av_core` ... + two thin `MotionEngineBase` adapters."** The two adapters share ONE core, but `ltx_av_talk` (I2V) and `ltx_av_music` (txt2vid) may need different graph sub-topologies. The plan does not specify how the core dispatches between them. **Fix:** add that the core exposes `render_talk(plan, ...)` and `render_music(plan, ...)` with the I2V vs txt2vid branch internal to the core, driven by an `engine_id` parameter.

5. **[DECISIONS 3] "the core obeys the AS-3 single-resident lease + BUG-291 reclaim + a below-ceiling NVML check AFTER each clip."** The plan does not specify what happens if `wait_until_below_mb(14500)` times out or never reaches the ceiling (e.g., another process holds VRAM). **Fix:** add a timeout and a classified error that triggers the restart rule.

---

OPTIONAL / NICE-TO-HAVE:

- The ">=2 degrades same ltx_av_* origin" storm line threshold is arbitrary. Operator may want it configurable.
- The `fallback_counts_by_from_engine` episode summary field could grow unbounded; consider capping at top-N.
- The `OTR_LTX_AV_VAE` env (sprint plan) is not mentioned in the plan doc's config envs list. Add for consistency.

---

CUT THESE (over-engineering):

1. **[SPRINT PLAN, storm lines / episode summary fields]** `pad_tail_count`, `padded_s`, `nvml_available`, `max_vram_mb`, `final_engine_histogram`, `fallback_counts_by_from_engine` -- these are monitoring/instrumentation fields, not required for Lane B to function. Safe to defer to a post-M4 observability ticket; the graduation bar (M4) already has manual greps.

2. **[SPRINT PLAN, golden semantic projection]** Capturing pre-delta goldens in `tests/fixtures/ltx_av_dark/` before any code changes is a heavyweight process for a dark lane that may park. A single regression smoke that `test_audio_byte_identical` + existing engine outputs unchanged (hash of a known clip) suffices. Full semantic-projection goldens can be scoped to M4 when Lane B proves viable.

3. **[SPRINT PLAN, `_slice_master_audio` cache key + mtime_ns + size bugfix]** Described as "the ONE shared-path bugfix this sprint ships." If this fixes a real HuMo/Lane-A bug, it should be its own ticket, not bundled inside Lane B delivery. Bundle risks holding the fix hostage to Lane B's graduation.

---

[ASSUMPTION] The `resolve_aspect_transform` math for COVER+crop exists in the codebase and is importable (not shown in grounding). If missing, the plan's init-image preprocessing is underspecified.

[ASSUMPTION] The `_ref_path(request.audio_ref)` extraction logic from `eng_humo.py:366-383` is copyable as-is and handles the `AudioRef` / `str` / `.path` variants described. If `eng_humo.py` has diverged, the plan's extraction spec is stale.

[ASSUMPTION] The `NODE_CLASS_MAPPINGS` lazy-read mechanism referenced in step 4 of `assert_usable` is the same one used in `wrapper_bridge.resolve_graph_classes` and is accessible at assert-time without heavy imports (not shown).