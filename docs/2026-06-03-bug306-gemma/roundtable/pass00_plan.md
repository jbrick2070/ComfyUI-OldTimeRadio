# BUG-306 handling plan -- google/gemma-4-12b-it is unloadable on the OTR Blackwell stack

## Problem

OTR (ComfyUI-OldTimeRadio, an offline AI radio-drama generator) added
`google/gemma-4-12b-it` to its writer-model catalog this session as a 12B writer
candidate. A live soak (2026-06-03) proved it is **not loadable**: the model's
`config.json` declares model type `gemma4_unified`, and the installed
**transformers 5.5** does not register that architecture:

```
... has model type `gemma4_unified` but Transformers does not recognize this
architecture ... your version of Transformers is out of date ... install
Transformers from source with pip install git+https://github.com/huggingface/transformers.git
```

Both gemma-4-12b soak combos failed identically (~30s, VRAM peak 3.3 GB -- the
model never loaded). The model-loader Selector retried 5x
(`[Selector] load_llm raised for google/gemma-4-12b-it; running unload_llm()`),
then the writer's StylePicker style-invention pass raised
`StyleGenerationFailedError`, which **aborted the entire episode**. There is
**no writer-side load-failure fallback** today: a chosen writer model that fails
to load takes the whole run down. The download itself is intact (22.31 GB single
`model.safetensors`, zero `.incomplete` blobs) -- this is purely a
transformers-version gap, not a corrupt download. Mistral-Nemo (the default) and
the smaller gemmas (E2B/E4B/2b, older architectures) all load fine; only the new
`gemma4_unified` 12B is affected.

## Hard constraints (invariants the fix MUST respect)

1. **The Blackwell venv is bleeding-edge and PROTECTED.** torch 2.10+cu130,
   numpy 2.4, transformers 5.5, sm_120, Windows, single RTX 5080 (16 GB).
   Upgrading transformers (e.g. `pip install git+.../transformers`) risks
   bricking the venv -- this exact stack has already been shown brittle: voice
   engines IndexTTS2/Chatterbox were rejected because they hard-pin
   torch 2.6-2.8 / numpy 1.26 / transformers 4.52 and would brick cu130. Any
   "just upgrade transformers" answer must address how to not regress the
   working torch 2.10 / cu130 / Blackwell stack.
2. **100% local, offline-first, no cloud/paid services at runtime.** (OpenRouter
   is a dev-time writer option but the default path is local HF weights.)
3. **VRAM ceiling 14.5 GB peak**, 16 GB card.
4. **Catalog shape:** `nodes/_otr_model_catalog.py` exposes
   `CURATED_LLM_MODELS`, a tuple of frozen `CuratedModel` dataclasses. There is
   **no `available`/`hidden`/`enabled` field today** -- a model is removed from
   the writer dropdown only by editing the tuple. `vram_fit_tier=="PASS"` gates
   the "16 GB-ready" label, not visibility (all rows appear in the dropdown).
   `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`.
5. **Project rule PD3 (wire-through):** any change to a node's `INPUT_TYPES` /
   widget options (the writer's model dropdown is built from the catalog) must be
   verified/re-wired against the workflow JSON -- a catalog change that drops a
   dropdown option must not leave the canonical workflow pinning a now-missing id.
   The canonical workflow currently pins Mistral-Nemo.
6. **Project rule PD6 (model-pick routing):** ONLY the writer node exposes
   model-pick widgets; consumers receive the model id via a STRING input. No new
   `model_id` widget may be added to any other node. A forbidden-pattern sweep
   rejects new `INPUT_TYPES` blocks containing a `model_id` widget.
7. **No-overhaul discipline:** prefer the smallest correct change; do not
   refactor the loader wholesale.

## Candidate options (critique, expand, rank these -- and surface anything missing)

- **A. Hide gemma-4-12b in the catalog.** Remove the row, or add an
  `available: bool`/`unavailable_reason` field to `CuratedModel` and filter
  unavailable rows out of the dropdown builder, with reason "needs transformers
  that registers gemma4_unified". Safe, reversible, keeps the license-audit work;
  re-enable when transformers catches up. (PD3: must confirm the workflow JSON
  does not pin gemma-4-12b -- it pins Mistral-Nemo, so OK.)
- **B. Writer load-failure fallback.** If the chosen writer model fails to load,
  fall back to `DEFAULT_LLM` (Mistral-Nemo) and continue rather than aborting the
  episode. Fixes the broader "no fallback aborts the whole run" gap, not just
  gemma. Open questions: where to catch (Selector vs StylePicker vs writer top),
  how to signal the substitution to the user/ledger, VRAM safety, idempotency,
  and making sure a fallback never silently masks a real misconfiguration.
- **C. Upgrade/patch transformers** to register `gemma4_unified` (RISKY -- see
  constraint 1). If proposed, must specify how to avoid regressing torch 2.10 /
  cu130 / Blackwell, and how to pin/verify.
- **D. Wait / pin a transformers release** that supports gemma4_unified; keep the
  catalog row but mark it not-yet-loadable.
- **E. Sidecar isolation.** Run gemma-4-12b in a separate venv with a newer
  transformers and talk to it over IPC (the architecture OTR already considers
  for dependency-conflicting voice engines). Heavier; is it justified for one
  writer candidate?

## What we want from the panel

1. Given the constraints, what is the best handling strategy NOW (and the right
   sequence if more than one step)?
2. Options we have not listed? Failure modes / risks of each (esp. A and B)?
3. If B (writer fallback): the safest concrete design -- catch point, how to
   surface the substitution, how to avoid masking real errors, VRAM/idempotency.
4. Is editing the frozen `CuratedModel` tuple the right "hide" mechanism, or is
   an `available` field + dropdown filter cleaner/more future-proof?
5. Anything in the constraints that makes a "common-sense" fix actually unsafe
   here?
