<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: no — the document's central go/no-go fact (is `gemma4_unified` a rename of 5.5's `gemma4` or a new arch?) is left unverified, and several proposed verifications are circular or offline-infeasible, so nothing downstream can be committed yet.

Note: this is a decision/RFC, not a build plan, so "must-fix" = defects in the reasoning/verification that block a safe decision.

MUST-FIX BEFORE BUILD
1. [Path B / Q1] The hypothesis "`gemma4_unified` is just `gemma4` renamed" is contradicted by the grounded config. `config.json` defines a unified token space (`image_token_id` 258880, `audio_token_id` 258881, `video_token_id` 258884, `eoi_token_id`, `eoa_token_index`, `boa/boi`) and `text_config` carries fields that look new to this arch: `global_head_dim:512` vs `head_dim:256`, `num_global_key_value_heads:1`, `rope_parameters.full_attention.rope_type:"proportional"`, `attention_k_eq_v:true`, `use_bidirectional_attention:"vision"`, `vocab_size_per_layer_input:262144`, `hidden_size_per_layer_input:0`, `num_kv_shared_layers:0`, `enable_moe_block`. These are strong signals of a distinct architecture, not a label swap. FIX: gate B on a concrete offline test — load the checkpoint's `state_dict` keys (from the safetensors index) and diff key names + tensor shapes against what transformers 5.5's `gemma4`/`gemma4_text` model instantiates; proceed only on exact match. [verify: actual 5.5 `gemma4_text` config schema and weight-key layout — not shown in grounding.]

2. [Path A] The plan says "upgrade transformers to >=5.10." Grounding shows `"transformers_version": "5.10.0.dev0"` — a **dev/unreleased** build. There is no stable 5.10 to pin, so A means installing an unreleased wheel/commit against torch 2.10+cu130 / numpy 2.4 / sm_120, which is exactly the regression vector the Hard-constraints section forbids for the shared runtime. FIX: stop calling A a "pin"; if pursued, name the exact git commit and treat it as installing pre-release code into the protected venv (i.e., effectively as risky as a fork), and run the full mistral-nemo + audio-C7 regression suite before adopting.

3. [Q1 verification] The proposed "logits sanity check vs a reference" is circular for an offline/local shop: producing trustworthy reference logits requires a *working* 5.10 load of this exact model, which you don't have yet. FIX: make the reference-generation step explicitly run inside an isolated 5.10 venv (i.e., you must build the sidecar/throwaway env *to verify B*), or drop logits-vs-reference and base the B go/no-go solely on offline weight-key + shape matching (item 1).

4. [Path C / Q3 — VRAM] The plan admits the coexistence problem but specifies no protocol. With a 14.5 GB ceiling and NF4 gemma ~8 GB, gemma cannot share the card with mistral-nemo + audio-C7 resident. FIX: define a hard serialized handoff — main process frees all CUDA models (`empty_cache`, confirm freed), sidecar loads → generates → frees → signals done, main reloads its models — and document the per-writer-switch reload latency this imposes. Without this, C "works" only on paper.

SHOULD-FIX
1. [Path C quant] NF4 (~8 GB) is the only thing that fits 16 GB; bf16 is ~24 GB and will not fit at all. NF4 needs bitsandbytes working on Windows + sm_120 + torch 2.10 for a brand-new arch. [verify: bnb support on sm_120/Windows.] If unavailable, C is dead regardless of IPC design.

2. [Path E "extract text submodel"] Loading only the text tower still requires (a) a registered top-level class — `Gemma4UnifiedForConditionalGeneration` does not exist in 5.5 — and (b) knowing the checkpoint's text-weight key prefix (e.g. `model.language_model.*`). The doc notes no remote code exists, which means you cannot instantiate the class at all on 5.5. State this prefix/availability or drop E.

3. [rope "proportional"] `rope_parameters.full_attention.rope_type:"proportional"` is non-standard; if 5.5's rope-init dispatch lacks that key it raises at load. Relevant to B and any 5.5 text-only attempt. [verify: 5.5 ROPE_INIT registry.]

4. [Path D / GGUF] Rank lowest / effectively non-viable now: GGUF requires `convert_hf_to_gguf` support and llama.cpp runtime support for the `gemma4_unified` arch, which won't exist for a 5.10.dev architecture. The doc lists it as a peer option; demote it.

5. [Fail-soft] The mistral-nemo fallback is correct and should be the default sequencing: ship that path first so the pipeline is unblocked while B/C is being verified.

OPTIONAL / NICE-TO-HAVE
- Capture the safetensors weight-map (`model.safetensors.index.json`) once and keep it; it is the single artifact that decides B and is reusable for E.
- Record a fixed prompt + greedy-decode hash as the "is gemma actually producing sane text" smoke test once any path loads.

CUT THESE (over-engineering)
1. [Path B "register as alias of gemma4 classes" variant] Cut the alias approach as a standalone option. Aliasing only differs from the rename in mechanics; both are unsafe unless the item-1 weight-key/shape diff passes, and if it passes the rename is simpler. One gate, one method.
2. [Path E vLLM mention] Cut. vLLM will not support a `gemma4_unified` (5.10.dev) arch either and adds a third runtime/venv — no advantage over C, more surface.
3. [Path D] Cut from the active shortlist (keep as a footnote) for the reasons in SHOULD-FIX 4 — it cannot land without upstream llama.cpp work, which is out of scope for "make it load now."

[ASSUMPTION] I could only verify against `config.json`. All claims about transformers 5.5's registered model_types, `gemma4_text` config schema, weight-key layout, rope registry, and bnb/sm_120 support are marked "verify" — the judge should confirm them against the real 5.5 source and the model's safetensors index, neither of which was provided.