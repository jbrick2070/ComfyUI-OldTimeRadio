# Make `google/gemma-4-12b-it` actually LOAD on the OTR stack (sidecar OK if needed)

## Goal
Run gemma-4-12b-it as an OTR writer LLM (TEXT-ONLY). It fails today: transformers
5.5.0 does not register the model's architecture. We WANT to keep this model and
make it work. A sidecar (separate venv + IPC) is ACCEPTABLE if it is the only
safe way -- but confirm it is necessary before paying that complexity.

## Exact facts (grounded this session)
- `config.json`: `architectures=["Gemma4UnifiedForConditionalGeneration"]`,
  `model_type="gemma4_unified"`, sub-configs `gemma4_unified_text` /
  `gemma4_unified_vision` / `gemma4_unified_audio`. It is a UNIFIED multimodal
  model (text+image+video+audio token ids). Authored against
  `"transformers_version": "5.10.0.dev0"`. Text submodel: hidden 3840, 48 layers,
  vocab 262144, sliding+full attention, max_pos 131072.
- Installed **transformers 5.5.0**. Its registered gemma model_types are:
  `gemma, gemma2, gemma3, gemma3_text, gemma3n, gemma3n_audio, gemma3n_text,
  gemma3n_vision, gemma4, gemma4_audio, gemma4_text, gemma4_vision, paligemma,
  recurrent_gemma, shieldgemma2, t5gemma, t5gemma2, t5gemma2_encoder, vaultgemma`.
  -> It HAS a `gemma4` / `gemma4_text` / `gemma4_vision` / `gemma4_audio` family,
  but NOT `gemma4_unified`. So 5.5 knows a "gemma4" under a DIFFERENT naming than
  the model's "gemma4_unified".
- OTR only needs TEXT generation; it has a `transformers_multimodal_text_only`
  loader path and does not need the vision/audio towers.
- Hardware: ONE RTX 5080, 16 GB, VRAM ceiling 14.5 GB. gemma-4-12b text is
  ~24 GB bf16 / ~8 GB NF4.

## Hard constraints
- The venv is PROTECTED and bleeding-edge: torch 2.10+cu130, numpy 2.4,
  transformers 5.5, sm_120, Windows. An in-place transformers change swaps the
  SHARED LLM runtime for EVERY writer model (mistral-nemo is the proven default
  + audio-C7 baseline) -- a regression risk for the whole pipeline, not just
  gemma. This stack has been bricked before by deps that hard-pin older
  torch/numpy/transformers (IndexTTS2/Chatterbox).
- 100% local / offline-first. No cloud at runtime.
- Single GPU: the main ComfyUI/OTR process and any sidecar must SHARE one 16 GB
  card -- they cannot both hold a 8-24 GB model resident at the same time.

## Candidate paths -- assess, rank, and surface anything missing

- **A. In-place upgrade transformers to >=5.10** (where `gemma4_unified` is
  registered). Lightest IF compatible. Risks: (a) does 5.10 stay compatible with
  torch 2.10+cu130 / numpy 2.4 / sm_120? (b) does it regress mistral-nemo or any
  other writer pass? (c) which exact version to pin? How to de-risk + verify?
- **B. Config rename / alias on transformers 5.5.** Rewrite the model's
  `model_type` `gemma4_unified -> gemma4` (and the sub-configs
  `gemma4_unified_text/vision/audio -> gemma4_text/vision/audio`), OR register
  `gemma4_unified` as an alias of the existing `gemma4` classes. **Only valid if
  `gemma4` (5.5) is the SAME architecture as `gemma4_unified` (5.10) under a
  rename** (so the checkpoint weights map cleanly). KEY UNKNOWN: is
  gemma4 -> gemma4_unified a rename, or a breaking arch change between 5.5 and
  5.10? How to verify BEFORE trusting generated text (weight-key match? a logits
  sanity check vs a reference?)? If it is a rename, this is by far the lightest
  fix and needs no sidecar.
- **C. Sidecar.** Separate venv (transformers >=5.10 + a compatible torch)
  running gemma-4-12b as a subprocess; the main OTR process talks to it over IPC
  (the pattern OTR already intends for dependency-conflicting voice engines).
  Most isolated -- the main stack stays byte-untouched. Accepted if needed.
  Design questions: transport (HTTP/localhost? stdio? a tiny FastAPI/llama-style
  server?), MODEL LIFECYCLE + VRAM COEXISTENCE on one 16 GB GPU (does the sidecar
  have to fully evict the main process's models first, then reload them after?),
  text-only loading to drop the vision/audio towers, Windows subprocess
  management, and fail-soft (sidecar down -> fall back to mistral-nemo).
- **D. GGUF + llama.cpp sidecar.** Convert to GGUF and serve via llama.cpp.
  Needs llama.cpp `gemma4` support; still a separate process. Pros/cons vs C?
- **E. Anything else** (vLLM; extract just the text submodel and load it as a
  known gemma arch; note: the model has NO remote modeling code on the repo --
  the architecture lives in transformers core -- so `trust_remote_code` does not
  help here).

## Questions for the panel
1. **Is the sidecar actually necessary?** Specifically: is `gemma4`
   (transformers 5.5) the same architecture as `gemma4_unified` (5.10) under a
   rename (=> B works, no sidecar), or a breaking change (=> sidecar/upgrade)?
   How would you VERIFY this conclusively from the checkpoint + the two
   transformers versions, not by guessing?
2. If A (in-place upgrade): which version, and how big is the regression risk to
   torch 2.10+cu130 and to mistral-nemo? How to pin + test so the main stack is
   safe?
3. If C (sidecar): the best concrete design given ONE 16 GB GPU shared by the
   main process and the sidecar -- the VRAM coexistence problem is the crux.
4. Rank A vs B vs C vs D for THIS situation (single 16 GB GPU, protected stack,
   text-only need, willing to sidecar). What would you ship?
5. Failure modes / what are we missing?
