<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Core path is undecided, the architecture-alias question is unresolved, and the single-GPU/VRAM lifecycle is not specified enough to build safely.

MUST-FIX BEFORE BUILD:
1. [B / Questions 1] The plan treats `gemma4_unified -> gemma4` as a possible config rename but does not define a conclusive pass/fail test. This is the highest-risk path because a false positive silently loads wrong weights and produces untrusted text. Concrete fix: create an isolated verification script that compares transformers 5.5 `gemma4*` against the target transformers version that supports `gemma4_unified`; require all of the following before shipping B:
   - diff the relevant config/modeling classes between 5.5 `gemma4` and target `gemma4_unified`;
   - attempt load with `output_loading_info=True`;
   - require zero unexpected/missing/shape-mismatched text-model weights, except explicitly documented non-text vision/audio keys if intentionally ignored;
   - run a deterministic logits check against the target-supported loader on the same prompt/tokenizer/revision, with a defined tolerance;
   - reject B if any module names, tensor shapes, RoPE/sliding-attention behavior, KV sharing, or logits differ.

2. [A] “Upgrade transformers to >=5.10” is not buildable as written. The grounded `config.json` says `"transformers_version": "5.10.0.dev0"`, not a stable released version, and the plan does not pin an exact wheel/commit. Concrete fix: identify the exact transformers version or git commit that registers `gemma4_unified`, pin it, and test it in a cloned venv before touching the protected runtime. Do not specify `>=5.10`; specify an exact version/hash plus a rollback command.

3. [A / Hard constraints] The in-place upgrade path has no regression gate for the existing writer stack. The document says mistral-nemo is the proven default and the shared runtime has been bricked before, but no acceptance test is defined. Concrete fix: before allowing A, require a cloned-venv test matrix:
   - import `torch`, `numpy`, `transformers`;
   - `pip check`;
   - CUDA visible and usable on RTX 5080;
   - load and generate with mistral-nemo using the existing OTR loader;
   - load and generate with the audio-C7 baseline if it uses the shared LLM runtime;
   - load and generate with gemma-4-12b-it;
   - compare known prompt outputs or at least deterministic smoke outputs;
   - only then promote the pin to the protected venv.

4. [C / Hard constraints] The sidecar design does not solve the stated crux: one 16 GB GPU cannot have the main process and the sidecar both holding large models resident. Concrete fix: define a strict GPU ownership protocol:
   - main OTR/Comfy process reaches an idle barrier;
   - main process unloads writer/diffusion/audio models that occupy VRAM;
   - main process calls its actual unload/free APIs; [ASSUMPTION] verify the real OTR/Comfy APIs;
   - verify free VRAM with NVML or equivalent before sidecar load;
   - sidecar loads gemma, serves one request or a bounded batch;
   - sidecar explicitly unloads or exits;
   - main process verifies VRAM was released before reloading defaults;
   - timeout/kill sidecar on leak/hang.
   Without this, C will intermittently OOM or leave the main process unable to resume.

5. [Goal / C / config.json] “TEXT-ONLY” loading is not specified concretely enough. The config is explicitly unified multimodal: top-level `model_type="gemma4_unified"` with `text_config`, `vision_config`, and `audio_config`. It is not proven that the target loader can skip vision/audio towers or that the checkpoint key layout allows a clean text-only load. Concrete fix: define the exact class/API used for text-only load in each chosen path and prove peak load excludes vision/audio modules. Acceptance criteria: no vision/audio tensors allocated, no unexpected text-weight misses, and peak VRAM stays below 14.5 GB.

6. [Hardware / Goal] The memory claim “~8 GB NF4” is not sufficient for build readiness. NF4 implies a quantization backend, but the plan does not name one or verify it works on Windows, RTX 5080/sm_120, CUDA 13-era stack, and the chosen torch/transformers version. Concrete fix: choose the quant backend explicitly, pin it, and run an import + load smoke test. Verify: whether bitsandbytes/torchao/quanto or another backend supports this exact Windows + CUDA + sm_120 environment. If no backend works, bf16 ~24 GB is over the 14.5 GB ceiling and the path is invalid.

7. [config.json / Hardware] The plan ignores KV-cache memory. The text config has `max_position_embeddings: 131072`, 48 layers, GQA, and sliding/full attention. Even with NF4 weights, long context KV cache can exceed the remaining VRAM. Concrete fix: set an OTR-specific max input context and max_new_tokens budget for gemma on 16 GB, then validate peak VRAM. Do not allow the loader to use the full 131k context by default.

8. [Offline-first / A / C] Offline operation is stated as a hard constraint but not implemented in any path. Concrete fix: pin the exact model revision, tokenizer files, transformers version/commit, quantization backend wheels, and sidecar dependencies in a local artifact cache. Runtime must use `local_files_only` or equivalent and offline environment flags. Add a test with network disabled.

9. [D] GGUF/llama.cpp is listed without a validity check. The model is `gemma4_unified`; the document only says llama.cpp needs `gemma4` support, not `gemma4_unified` support or a verified converter path. Concrete fix: treat D as invalid until a converter and runtime can load this exact checkpoint/config/tokenizer and pass a generation smoke test.

SHOULD-FIX:
1. [Questions 4] The document asks for ranking but does not produce a ship decision. Concrete fix: make the build plan conditional:
   - first try B only if the architecture/logits test passes;
   - otherwise test A in a cloned venv;
   - use C only if A is unsafe for the protected runtime;
   - defer D until proven.
   This avoids implementing a sidecar before proving whether a simple alias works.

2. [C] Fail-soft behavior is underspecified. “sidecar down -> fall back to mistral-nemo” is not enough on one GPU. Concrete fix: fallback must kill/unload the sidecar first, verify VRAM release, then reload mistral-nemo. If sidecar failure happens after partially loading gemma, fallback without cleanup will likely OOM.

3. [C] IPC transport is left open, which risks overbuilding. Concrete fix: unless concurrent multi-client serving is required, use a minimal local stdio JSON protocol or single localhost HTTP endpoint. Define request/response schema: prompt, generation params, seed, stop tokens, timeout, error response.

4. [C] Windows subprocess lifecycle is not specified. Concrete fix: start sidecar with explicit env, cwd, venv python path, inherited/offline cache paths, and a kill-on-parent-exit mechanism. Verify: actual Windows job-object or process-tree cleanup implementation.

5. [config.json / loaders] EOS handling is ambiguous. Top-level `eos_token_id` is `[1, 106]`, while `text_config.eos_token_id` is `1`. Concrete fix: define which stop ids OTR generation uses for this model and test that generation terminates correctly.

6. [B] Config editing is dangerous if done in-place in the model cache. Concrete fix: never mutate the canonical downloaded model. Use an overlay copy or generated temporary config for the alias experiment, and record the diff.

7. [A / C] The plan does not separate “can import transformers” from “can generate with the model.” Concrete fix: test import, config load, tokenizer load, model load, first-token generation, multi-token generation, and unload separately so failures are diagnosable.

8. [Goal] No quality sanity check is defined. Concrete fix: after any successful load path, run a small fixed prompt suite and compare against a reference loader where possible. At minimum verify no repeated garbage, no multimodal sentinel-token spam, and sane EOS behavior.

OPTIONAL / NICE-TO-HAVE:
- Add a small diagnostic command: `otr-gemma4-doctor` that prints selected path, package pins, model revision, quant backend, free VRAM, load class, and a one-token generation result.
- Store the successful architecture verification report with the model cache so future upgrades do not repeat the same investigation.
- Add telemetry/logging around sidecar load time, peak VRAM, generation latency, and fallback reason.

CUT THESE (over-engineering):
1. [C] Cut a full FastAPI-style service unless there is a real need for concurrent clients or OpenAI-compatible APIs. A single-user local OTR writer call can use stdio JSON or a tiny localhost endpoint with less dependency and lifecycle risk.

2. [D] Cut GGUF/llama.cpp from the first build. It adds conversion, tokenizer, quantization, and runtime-compatibility variables before proving llama.cpp supports this exact `gemma4_unified` checkpoint. Safe to revisit only after B/A/C fail.

3. [C] Cut a long-lived persistent sidecar for the first implementation. On one 16 GB GPU, persistent residency is the problem. Spawn-on-demand with explicit exit after generation is slower but safer and easier to validate.

4. [A] Cut any direct mutation of the protected venv during exploration. All A testing should happen in a cloned venv or fresh side venv until the exact pin and regression results are known.