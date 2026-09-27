# Adding a writer LLM

Two different jobs share this page. Do not mix them up.

**What you can do:** pick any model the dropdown will take. Cache a Hugging
Face causal LM of your own, choose Gemma or Llama from the list, use a cloud
slot if you have a key. The picker is open. Nothing here locks you to Qwen.

**What this pack ships:** since 2026-09-26, Gemma 4 run by ComfyUI itself
rather than by transformers (see "A writer ComfyUI runs itself" below): the
16 and 24 GB NVIDIA workflows ship `comfy_native:gemma4-12b-int8-convrot`, and
the 8 GB NVIDIA workflows and the canonical ship
`comfy_native:gemma4-e2b-it-int8-convrot`. Both are four to twelve times the
writers they replaced. The Mac and AMD workflows ship Qwen 3.5 4B as one
transformers dropdown row (Mac loads full) until the native path is proven
there. Every row downloads itself.

This page is the add-your-own checklist. The models that already ship, and
how to read their badges, are [WRITERS.md](WRITERS.md). The short binding
notes are in [EXTENDING.md](EXTENDING.md). Run [PREFLIGHT.md](PREFLIGHT.md)
for the named tests after a catalog edit.

`docs/` is stripped from a Manager install. This file ships in `apple/`.

---

## On your machine -- no git clone

Put the model in the pack's writer folder as plain files, restart ComfyUI,
pick `org/name` on both writer slots, set Quant and the VRAM ceiling for your
hardware.

1. **The folder.** `models/LLM/<org>--<name>/` under your ComfyUI models tree
   (for example `models/LLM/Qwen--Qwen3.5-4B/`), or wherever an `LLM:` entry
   in `extra_model_paths.yaml` points. Since 2026-09-25 this is where the
   pack's own downloads land; a Hugging Face cache snapshot under
   `models/huggingface` from an earlier version is still found and still
   loads, so nothing needs moving.
2. **The folder has to be complete.** A config-only folder is not a model.
   `config.json` plus the weights: either `model.safetensors`, or a
   `model.safetensors.index.json` with every shard it names present.
3. **`config.json` must list an architecture ending in `ForCausalLM`.** The
   cache is shared with FLUX, LTX, depth models and everything else this pack
   fetches. An uncurated folder appears in the writer dropdown only with that
   suffix. Multimodal checkpoints that do not declare `ForCausalLM` will never
   show up this way -- the Gemma 4 rows are curated for exactly that reason.
4. **Restart ComfyUI.** `INPUT_TYPES` runs at registration. A download that
   finished while the server was up is invisible until then.
5. **An uncurated id has no fit tags.** Curated labels look like
   `Qwen/Qwen3.5-4B (8.7 GB download, mac16-tight nv8-nf4 nv16 nv24)`. A cache-discovered id is
   the bare `org/name`. Do not infer tags for it.
6. **Quant is baked on the rows that own it.** The one Qwen loads NF4 on
   NVIDIA and full on Mac / CPU. Gemma 4 12B owns `bnb_nf4`. Llama,
   Gemma-2, a cache folder, and cloud handles still use the Quant widget.
7. **Keep both writer slots on the same id** unless you mean to swap two
   models in and out of VRAM all run.
8. **The technical slot has to emit JSON the pipeline can parse.** Beautiful
   prose that cannot hold a schema dies after the expensive load.
9. **Nothing substitutes.** If it will not load, the run stops and names the
   id you picked.

A first Queue that sits still is usually the download. The console prints
`[OTR] Downloading ...`. Auto-download is designed, not a defect. Disk
headroom is the model size plus 5 GB on the volume the writer folder is on.

---

## In the shipped dropdown -- git clone

This is a catalog row, not an adapter. One `CuratedModel(...)` in
`CURATED_LLM_MODELS` inside `nodes/_otr_model_catalog.py`.

`repo_id` is unique in the dropdown. Two rows may share one Hugging Face
snapshot only through `hf_repo_id` plus `implied_quant_policy`. Qwen itself
is one row (`implied_quant_policy="platform"`). Do not invent a second HF
repo for the same weights.

### Every workflow gets the row at once

The writer dropdowns on **OTR_LedgerScriptWriter** read `dropdown_choices()`
when ComfyUI registers the node, so a new row is offered in the canonical
workflow and in every generated workflow the moment ComfyUI restarts. Adding
a choice needs no workflow JSON edit and no regenerated workflow; the saved values
stay exactly what they were. A workflow changes only when a row becomes what
it SELECTS, which is "If it becomes a shipped default" below. Queue a
workflow with the new row picked and the queue-time preflight fetches its
weights before the writer runs.

### The gates

Run these in order. Each one has caught a real failure in this repo.

**Weights on disk (or will download by name).** A
dropdown entry is a promise. Auto-download on first Queue is the designed
path when the snapshot is missing. Do not silently substitute a different
model to satisfy a selection.

**The VRAM tier is honest.** `vram_fit_tier` is `PASS`, `WARN`,
`UNKNOWN`, or `FAIL`. PASS means soak-tested inside the ceiling. WARN means
not soak-tested; **it stays in the dropdown** (operator 2026-09-06). What a
tier owes the user is honesty at the moment of choosing -- the badge number
and truthful `notes` -- not absence. Ripping a row is an explicit decision
about a specific model. UNKNOWN / FAIL do not ship.

**It loads under the Quant the pick claims.** For Qwen that is
platform policy: NF4 on NVIDIA, `none` on Mac / CPU -- one picker row, not
two. Watch resident VRAM, not the file size. An unquantized load must not
inherit a ceiling sized for 4-bit.

**It generates free-form prose.**

**It generates constrained JSON.** This is the gate that actually
fails, and prose passing tells you nothing about it. The hard-constraint
machinery is opt-in per call. Verify the specific call your new row will make
actually binds a schema.

**The chat template accepts the roles OTR sends.** OTR sends a
system + user pair. Gemma-2 rejects the system role; the generate path
already folds that. Qwen 3.5 4B needs `enable_thinking=False` on every
generate call -- one Hugging Face repo, one picker row.

**The context window is the file's truth.** Declare what the
artifact supports, not what a model card claims. KV cache is not free.

### Field contract

| Field | Meaning |
|---|---|
| `repo_id` | Dropdown identity. Hugging Face `org/name`, or a `comfy_native:` id for a writer ComfyUI runs itself. A `:nf4` suffix is a retired alias for validation only -- it is not a COMBO row. |
| `hf_repo_id` | Real Hugging Face repo when `repo_id` is an alias (and the repo that holds a Comfy-native writer's file). Empty means `repo_id` is the HF id. |
| `implied_quant_policy` | When set (`none` / `bnb_nf4` / `platform`), the pick owns Quant. `platform` is what Qwen uses (NF4 on NVIDIA, full elsewhere). Empty means the widget still decides. |
| `requires_auth` | Whether Hugging Face actually gates it. Measure it; do not copy the previous row. |
| `loader_backend` | Dispatch key. `transformers_safetensors` for an ordinary causal LM; `comfy_textgen` for a writer ComfyUI runs itself. |
| `provider` | `local` (transformers), `comfy_native` (ComfyUI's own loader, still local), or a cloud provider. |
| `vram_fit_tier` | Honesty at choose-time. |
| `approx_safetensors_gb` | Download size on disk, not VRAM resident. The badge prints this number. |

Defaulted but set explicitly on every production row: `prompt_profile`,
`chat_template_kind`, `stop_tokens`, `context_window`, `license`,
`license_audit_status`, `provider`.

**Licensing is not optional.** `license` + `license_audit_status` mirror a
per-repo audit at `apple/model-license-<sanitized>.md`. A Quant twin shares
the parent's audit; do not invent a second licence file for the same
weights.

`license_audit_status` on anything the canonical graph binds must be
`mit_equivalent`.

### A writer ComfyUI runs itself

The Comfy-native row is a different kind of writer. ComfyUI's stock CLIP
loader opens one text-encoder file (Comfy-Org's `gemma4_e2b_it_int8_convrot`,
5.2 GB, already int8) and ComfyUI's own generate loop writes the script. So
ComfyUI's model management loads and unloads it like any other model; there is
no transformers snapshot, no bitsandbytes, and no Quant choice.

How it plugs in, in `nodes/_otr_comfy_textgen_backend.py`:
`ComfyGemmaGenerateAdapter` speaks the part of transformers' `generate()`
OTR's four generation factories use, and the cache entry carries the model's
own tokenizer. The writer's halt rules, budget fitting, grammar and stop
trimming therefore run unchanged -- one copy, not a parallel native path.
`request_slot` gives it the same residency rules as a transformers writer.

To add another Comfy-native writer (another Comfy-Org text-encoder file):

1. Pin the file in `_otr_visual_assets._PINNED_SOURCES`: repo, revision, size
   and sha256 read from the Hub API. The queue-time preflight and
   `request_slot` both fetch through it.
2. Map the new id to that file in `_WEIGHTS_BY_MODEL` in the backend module.
   A decoder family other than Gemma 4 also needs its own chat template, stop
   ids and tokenizer path; the adapter is written for Gemma 4.
3. Add the catalog row: `provider="comfy_native"`,
   `loader_backend="comfy_textgen"`, `implied_quant_policy="none"`,
   `hf_repo_id` naming the Comfy-Org repo, and `context_window` as the
   working cap. ComfyUI allocates the KV cache for prompt plus budget up
   front, so that number bounds memory, not only length.
4. The licence audit is keyed on `hf_repo_id`
   (`apple/model-license-comfy-org--gemma-4.md` covers the whole repo).
5. Same gates as any row, and the same last one: an episode in `otr/obs/`.

It runs inside ComfyUI only. A standalone script that loads through
`load_llm` gets a clear refusal, not a download.

### A writer must ship safetensors weights

The writer loads safetensors through transformers, and auto-download refuses
an uncurated repo that carries none before fetching anything. If what you have
on disk is some other weight format, cache a CausalLM snapshot instead or pick
what is already in the list.

### If it becomes a shipped default

The badge string is part of the saved widget value. Update `DEFAULT_LLM` /
`default_llm_option()` and `fresh_llm_option()` together -- they are the
same one Qwen label -- then the writer `widgets_values` on
`workflows/otr_canonical.json` (never a bare repo id), then
`python scripts/build_variants.py --all` and `--check`. Edit
[WRITERS.md](WRITERS.md) in the same change. Do not hand-edit the variant
JSONs.

A helper nobody calls is this repo's most repeated defect. After you add
the row, grep `nodes/` for a production path that is not a test and not the
catalog. The writer combos read `dropdown_choices()`; that is the caller.

Then one full run through `workflows/otr_canonical.json` that lands a file
in `otr/obs/`. Green tests are not a writer.

---

## What a preflight must never do

- Never substitute silently. If the selected model cannot load, the run
  stops loudly.
- Never let a skip read as a pass.
- Never treat "the user can pick it" as "the pack ships it."
