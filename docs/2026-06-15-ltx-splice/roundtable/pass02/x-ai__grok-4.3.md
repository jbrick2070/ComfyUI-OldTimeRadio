<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Multiple contradictions between "keep shipped ksampler default + MODEL-ONLY swap" and "replicate exact mini-JSON recipe" (incl. distilled chain), plus incomplete graph rewrites that will leave dangling W("checkpoint",*) refs and VRAM leaks.

MUST-FIX BEFORE BUILD:
1. [§4A] _build_graph_i2v still hardcodes `W("checkpoint",2)` for vae in the img2vid node (grounding: eng_ltx_video.py:718) and sampler latent_image wiring; plan only says "in BOTH builders" but must also replace the i2v vae wire + any checkpoint,2 usage with W("videovae",0) before the Phase-1 commit.
2. [§4A + §2] Plan requires always wiring LoraLoaderModelOnly (remove _ckpt_name "22b" gate) yet keeps ksampler default; the target LoRA is the distilled-only artifact (grounding: LTX_DISTILLED_LORA_DEFAULT + _use_distilled_lora check at eng_ltx_video.py:561), creating a silent key-mismatch risk on ksampler path — add explicit guard or make lora conditional on distilled mode.
3. [§4A] VRAM reclaim in render_clip (grounding: eng_ltx_video.py:805,809) does `keep={"checkpoint",...}` + `results.get("checkpoint")`; plan must change both to the post-lora key ("lora") exactly as stated, else the patcher is dropped.
4. [§3] Phase 0 deletes LtxOrbitEngine + registry row + all references, but LtxOrbitEngine inherits render_clip/_build_graph from LtxVideoEngine (grounding: eng_ltx_video.py:887); plan must also delete the inheritance and any shared _LOOP_VIA_REVERSE_DEFAULT usage or Phase 0 will leave a broken subclass.
5. [§4B] registry.py:CAPABILITIES["ltx_video"] still lists model_requirements=["ltx-video-2b"] (grounding); must be updated to the GGUF+Gemma+LoRA+videoVAE tuple in the same commit as the engine change.

SHOULD-FIX:
1. [§5] "confirm build_request_from_shot routes LTX to full-frame scene-still pool" (render_driver.py:722-745) is stated but not actionable; add explicit regression assert in the motion smoke that every LTX role gets a non-portrait init_image path that exists.
2. [§4A] _node_candidates change drops "checkpoint"/"encoder" but _node_candidates_i2v does `cands = dict(self._node_candidates_sampling())` then adds loadimage/img2vid (grounding: eng_ltx_video.py:488); must ensure the new unet/videovae/te/lora keys are present in the i2v candidate set too.
3. [§7] Motion smoke requires frame-diff floor on announcer+music+scene roles before allowing distilled; plan must also run the identical smoke on ksampler path (current default) or the "gated on t2v motion" decision cannot be evaluated.
4. [§6] Add the banned-class grep (ClownSampler_Beta etc. + residual CheckpointLoaderSimple) as a hard pre-commit check, not just a note.

OPTIONAL / NICE-TO-HAVE:
- [§8] Gemma device="default" vs "cpu" + temporal_size=4096 can stay verify-at-build; no code change needed unless episode-length OOM appears.
- [§4C] render_driver canvas=832x480 confirmation can be a one-line comment rather than a separate task.

CUT THESE (over-engineering):
1. [§4D] "Only touch otr_scifi_16gb_full.json if a knob is truly wired" — safe to drop entirely; the JSON has no LTX nodes (plan states this) so the env/default path is already sufficient and the BUG-LOCAL-097 append is unnecessary.
2. [§2] Explicit "No ModelSamplingLTXV" callout — already omitted in the mini recipe; no code or test change required.