# HuMo quality + VRAM-fit -- r3 hardened WIRING plan (Claude + Codex; Antigravity pending)

r3 panel: Codex (gpt-5.5/high, read the real repo; ALL MUST-FIX grounded + CONFIRMED
against the cited files) + Claude anchor. Antigravity (agy) did not return in the window;
fold on land. This round fixes the exact wiring contracts; nothing here is production --
the harness stays HTTP/diagnostic and promotion is operator-gated.

## CONFIRMED wiring corrections (grounded; Codex r3)

1. **GGUF loader contract (Step B).** `UnetLoaderGGUF` takes required `unet_name` ONLY --
   NOT `gguf_name`, and NO `weight_dtype` (ComfyUI-GGUF/nodes.py; mirrored in
   `eng_wan_i2v.py:215-218`). The builder's gguf leg must emit
   `{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":<gguf file>}}` and MIRROR the
   proven pattern in `eng_wan_i2v._loader_mode` / `_node_candidates` (unet_cls switch) /
   the `unet_inputs` branch (eng_wan_i2v.py:155-218). Validate against LIVE `/object_info`,
   not disk. (My anchor's `gguf_name` was WRONG -- corrected.)

2. **Runner is hard-wired to UNETLoader -> make it loader-agnostic.**
   `run_humo_bakeoff.assert_checkpoints` checks `("UNETLoader","unet_name",...)` (~:319-327)
   and `build_manifest` does `_find(prompt,"UNETLoader")` (~:361-365) -- a GGUF leg would
   crash / false-fail. FIX: add `loader_class` + `loader_param` to `meta`
   (UNETLoader/unet_name vs UnetLoaderGGUF/unet_name) and make both functions read those.

3. **No-LoRA leg = STRUCTURE change, env-driven (NOT a literal patch like cfg).** Dropping
   the LoRA deletes the `LoraLoaderModelOnly` node and rewires `ModelSamplingSD3.model`
   from lora-output back to unet-output -- which `_build_graph` already does when
   `skip_lora` (eng_humo.py:232-234,263-271). So the runner SETS `OTR_HUMO_LORA_NAME=none`
   + `OTR_HUMO_STEPS=<n>` (17B-namespaced vars for the 1.7B tier) in the build-time env per
   leg, builds, restores -- do NOT post-patch. The `cfg` literal-rewrite stays as-is
   (build_humo_bakeoff_workflow.py:164-171). Update `meta["lora"]` + checkpoint asserts.

4. **Honest VRAM meter ordering (Step A).** A latent-edge probe (like OTR_BakeoffReclaim)
   runs BEFORE the sampler and misses the peak. FIX: call
   `torch.cuda.reset_peak_memory_stats()` immediately BEFORE the HuMo prompt, then read
   `max_memory_allocated()`/`memory_reserved()` AFTER KSampler/decode (a post-VAEDecode
   passthrough probe node in the sibling pack, OR a server-side log) -- max_memory_allocated
   is cumulative-since-reset, so the reset is the load-bearing part. For the SENTINEL leg,
   reset AFTER the LTX render and BEFORE the HuMo prompt so the number is HuMo-specific.
   Record `OTR_BAKEOFF_ALLOC_CONF` + the effective `PYTORCH_CUDA_ALLOC_CONF` in the manifest.

5. **Promotion wiring (grounded; deferred, operator-gated).** `config/profiles/16gb_full.json`
   pins `role_overrides.other_beats_visual="humo_1.7B"` and `slot_overrides.video_render_engine
   ="humo_1.7B"` -- **`humo_1.7B` (portrait), NOT `humo_1.7B_169`.** The saved workflow node
   87 `other_beats_video_model="visualizer (16:9)"` and node 92 `engine="humo_1.7B"`; AND
   episode mode IGNORES node 92 -- it renders from the ShotLock ledger
   (`otr_video_render_batch.py:127-134`). So PROMOTE by setting
   `role_overrides.other_beats_visual` to the winner + applying it into node 87; node 92 is
   single/soak parity ONLY. Do NOT move announcer off `ltx_audio_in` unless intended.

## CONFIRMED smaller fixes
- "ONE-FRAME smoke" is impossible: `_HUMO_MIN_FRAMES=33` (eng_humo.py:53-54). Call it the
  MIN-FRAME (33f) smoke, or build a separate verified 1-frame diagnostic path.
- Frame-matrix gate: today the result only checks `frame_count>0`; ASSERT
  `result.frame_count == manifest.length` per matrix cell (run_humo_bakeoff.py ~:636-655).
- Caveat the control: the bakeoff control was `humo_1.7B_169` (wide), but production pins
  `humo_1.7B` (portrait). For a true production-control eyeball, also render `humo_1.7B`
  portrait, or explicitly note the aspect mismatch in the verdict.

## CUT (Codex; grounded)
- A standalone "GGUF file exists on disk" gate -- folder_paths/`/object_info` loadability is
  the real contract (raw existence can pass while the loader can't resolve it).
- Treating `OTR_VideoRenderBatch.engine` (node 92) as an EPISODE promotion gate -- episode
  mode bypasses it.

## Sequencing (kill-gated, unchanged)
A (meter + reset + alloc-conf A/B across frames=[49,97,177], assert frame_count==length)
-> if true max_memory_allocated < 13.5 GB STOP (14B promotable) -> else B (gguf:
mirror eng_wan_i2v loader-mode; /object_info check; 33f min-smoke for audio cross-attn;
then matrix) ; C (no-LoRA/steps mouth ceiling, env-driven) in PARALLEL ; D (model-swap dep
probe) only if mouth needs it.

## Judgment (r3)
ACCEPTED + CONFIRMED (all grounded, zero misreads): GGUF unet_name contract + mirror
eng_wan_i2v; loader-agnostic meta in the runner; no-LoRA env-driven structure change;
reset_peak_memory_stats before each HuMo prompt + post-sampler read; promotion via
role_overrides.other_beats_visual + node 87 (not node 92; episode ignores it); profile pins
humo_1.7B not _169; min-frame smoke; frame_count==length gate; CUTs. PENDING: Antigravity
r3 (not returned). NEXT: r4 convergence (confirm no new must-fix) -- local kibitz.
