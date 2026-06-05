<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: no. The plan correctly names the unknowns but resolves none of them; every candidate path has at least one load-blocking gap that would cause a silent wrong model or a hard crash rather than a clean load.

---

## MUST-FIX BEFORE BUILD

**1. [Candidate path B — critical correctness] The rename-vs-arch-change question is the entire pivot of the plan, yet no concrete verification method is given.**

The config fields in `text_config` that do NOT appear in standard `gemma4_text` include `global_head_dim: 512` (separate from `head_dim: 256`), `num_global_key_value_heads: 1`, `layer_types` (48-element per-layer list), `rope_parameters` as a nested dict keyed by attention type, `vocab_size_per_layer_input`, `hidden_size_per_layer_input`, and `use_bidirectional_attention`. If 5.5's `Gemma4TextConfig.__init__` ignores these as `**kwargs` (which HuggingFace config classes typically do), the model loads silently but runs wrong attention patterns — wrong RoPE, wrong sliding-window grouping. That is worse than a hard load failure.

**Concrete verification required (both steps):**
- (a) `python -c "import transformers; import inspect; from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig; print(inspect.signature(Gemma4TextConfig.__init__))"` — check whether the above fields appear in the 5.5 signature or are silently swallowed.
- (b) List every weight-key tensor in the checkpoint (`safetensors.torch.load_file` on shards, print keys) and diff against the key names `Gemma4ForCausalLM.state_dict()` produces on an empty model — any prefix mismatch (`model.language_model.*` vs `model.*`, etc.) means B fails with missing weight keys. Do this before touching any config.

Until both checks pass, Path B must be treated as unknown, not as "lightest fix."

---

**2. [Candidate path A — correctness] "transformers >=5.10" cannot be pinned because `5.10.0.dev0` is a dev pre-release — no stable 5.10 may exist on PyPI at build time.**

The grounded `config.json` says `"transformers_version": "5.10.0.dev0"`. The plan says "upgrade to >=5.10" as if a release exists. Verify: does `pip index versions transformers` show any 5.10.x on PyPI? If not, Path A means installing a dev wheel from GitHub main, which is definitionally unstable and violates the protected-stack constraint the plan itself emphasizes. Add an explicit pip-installable version number or declare Path A blocked until release.

---

**3. [Candidate path C — missing design: VRAM coordination is described as "the crux" but has no proposed solution.]**

Single GPU, 14.5 GB ceiling, main process holds resident models (mistral-nemo already ~14 GB in NF4), sidecar needs ~8 GB NF4 for gemma-4-12b-it text. The plan identifies the problem and stops. The required design is:

- Main process must call explicit model offload (`.cpu()` or `offload_state_dict`) + `torch.cuda.empty_cache()` and emit a READY signal before the sidecar attempts CUDA allocation.
- Sidecar must wait on that signal, allocate, generate, then signal back before main process reloads.
- There is no existing OS-level VRAM lock; without this handshake, both processes race to allocate and one OOMs silently.
- On Windows, `multiprocessing` uses `spawn` (not `fork`), so any IPC design relying on shared memory or inherited file descriptors must be explicit. A localhost HTTP approach (tiny FastAPI or stdlib http.server) is the safest cross-platform transport, but the plan does not commit to any transport.

Without a concrete protocol, Path C cannot be implemented safely.

---

**4. [Candidate path D — unverified prerequisite] The plan doesn't check whether llama.cpp supports `gemma4_unified` before proposing it as a fallback.**

[ASSUMPTION] llama.cpp `gemma4` support may exist for the earlier architecture but `gemma4_unified`'s per-layer attention type list (`layer_types`), dual RoPE config, and `global_head_dim` are newer. Verify: check llama.cpp `src/llm_arch.cpp` (or equivalent) for a `LLM_ARCH_GEMMA4_UNIFIED` entry before investing in GGUF conversion. If it only knows `GEMMA4`, the same rename-vs-arch-change question applies there too.

---

**5. [Candidate paths B, C — text-only loading gap] The plan assumes loading text-only is straightforward but never checks whether the NF4 quantization path handles an unregistered model_type.**

`bitsandbytes` / `transformers` NF4 loading goes through `AutoModelForCausalLM.from_pretrained`. If `gemma4_unified` is not in the AutoModel registry, this call fails before quantization runs — you cannot even get to the 8 GB footprint. Path B must register the alias *before* calling `from_pretrained`. Path C must use a venv where the type is registered. The `transformers_multimodal_text_only` OTR loader path still calls into HuggingFace Auto classes under the hood — [ASSUMPTION: verify this] — and will hit the same registration wall.

---

## SHOULD-FIX

**1. [Section "Exact facts" + all paths] The plan never lists the actual checkpoint weight key prefixes.** Without this, any weight-loading path (B, C, D, or E) is a guess. A one-liner against the first shard (`python -c "from safetensors import safe_open; f=safe_open('model-00001-of-XXXX.safetensors', framework='pt'); print(list(f.keys())[:20])"`) tells you immediately whether the text model weights live under `language_model.`, `model.`, or some other prefix.

**2. [Candidate path E — underexplored] The plan mentions "extract just the text submodel" in one line and dismisses it.** If the text weights have a consistent key prefix (e.g. `language_model.*`), loading them into a 5.5 `Gemma4ForCausalLM` with a filtered `state_dict` is simpler than a sidecar and leaves the main venv untouched. This is only invalid if the weight keys don't match 5.5's expectation — which is verifiable (see Must-Fix #1). It deserves a full evaluation column.

**3. [Candidate path B — alias registration method not specified]** Registering `gemma4_unified` as an alias requires either `AutoConfig.register("gemma4_unified", Gemma4Config)` + `AutoModelForCausalLM.register(Gemma4Config, Gemma4ForConditionalGeneration)` called before any load, or patching `transformers/models/auto/configuration_auto.py`. The first approach (runtime registration in OTR's loader) does not touch the protected venv at all and is the correct scoping. The plan does not distinguish these.

**4. [Candidate path A — regression test scope undefined]** "Does it regress mistral-nemo?" is asked but not scoped. Minimum test: load mistral-nemo in NF4 on the upgraded venv, run a fixed prompt, compare output token IDs against a stored reference from 5.5. Without this baseline, there is no way to detect silent regressions.

**5. [All paths — no rollback plan]** If Path B silently mis-loads the model (weights load without error but rope/attention config is wrong), the only symptom is garbage text. The plan mentions "a logits sanity check vs a reference" but gives no reference source, no prompt, and no pass/fail criterion. A concrete check: run a ~50-token greedy decode on the same prompt using HuggingFace's hosted inference or a known-good local reference, log the top-5 token IDs at position 1, and reject if they don't match within tolerance.

---

## OPTIONAL / NICE-TO-HAVE

- For Path C sidecar: add a `--max-new-tokens` cap at the process boundary to prevent runaway generation holding VRAM open.
- Document the exact BitsAndBytes version that supports sm_120 (RTX 5080 Ada arch); earlier builds don't have CUDA 13 kernels and silently fall back to slow paths or crash.
- If Path B works: store the patched `config.json` under a local override directory rather than modifying the model's cache entry, so re-downloads don't overwrite the patch.

---

## CUT THESE (over-engineering)

**1. [Candidate path D — GGUF/llama.cpp]** There is exactly one GPU, the same problem (rename vs arch change, VRAM coordination) applies, and GGUF conversion adds a multi-hour preprocessing step plus a separate llama.cpp build against sm_120/CUDA 13 on Windows (non-trivial). If Path B or C works, D adds complexity for no gain that C doesn't already provide. Cut unless B and C both fail *and* llama.cpp gemma4_unified support is confirmed.

**2. [Section "Questions for the panel" — question 2 asks which exact transformers version to pin]** This is a research task, not a design question. The answer is already constrained: it must be the lowest stable release that registers `gemma4_unified`, and that version must exist on PyPI. If none exists (because 5.10 is still dev-only), Path A is blocked by definition and the question is moot. Remove it from the panel discussion; it only gets answered by running `pip index versions transformers`.