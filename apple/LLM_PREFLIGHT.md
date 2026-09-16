# Adding a writer LLM

Two different jobs share this page. Do not mix them up.

**What you can do:** pick any model the dropdown will take. Cache a Hugging
Face causal LM of your own, choose Gemma or Llama from the list, use a cloud
slot if you have a key. The picker is open. Nothing here locks you to Qwen.

**What this pack ships:** Qwen 3.5 4B as transformers, with **two honest
dropdown identities** -- NF4 for NVIDIA 8 GB graphs and a freshly dropped
node, full precision for canonical and Mac. The 16 GB NVIDIA graphs still
ship `google/gemma-4-12b-it`. There is no GGUF writer in the shipped catalog
(`GGUF_ROWS` is empty on purpose: a transformers twin already exists). Video
engines may still load a GGUF UNet; that is a different dropdown.

This page is the add-your-own checklist. The models that already ship, and
how to read their badges, are [WRITERS.md](WRITERS.md). The short binding
notes are in [EXTENDING.md](EXTENDING.md). Run [PREFLIGHT.md](PREFLIGHT.md)
for the named tests after a catalog edit.

`docs/` is stripped from a Manager install. This file ships in `apple/`.

---

## On your machine -- no git clone

Put a complete Hugging Face snapshot in the cache this pack actually uses,
restart ComfyUI, pick `org/name` on both writer slots, set Quant and the VRAM
ceiling for your hardware.

1. **Find the real cache.** Default is `ComfyUI/models/huggingface` when
   `HF_HOME` is unset -- [INSTALL.md](INSTALL.md) section 7. Setting `HF_HOME`
   later does not move an existing cache; it adds a second one.
2. **The snapshot has to be complete.** A config-only folder is not a model.
   There must be a weight blob (`.safetensors` or `.bin`, or a shard index
   whose shards are present).
3. **`config.json` must list an architecture ending in `ForCausalLM`.** The
   cache is shared with FLUX, LTX, depth models and everything else this pack
   fetches. An uncurated folder appears in the writer dropdown only with that
   suffix. Multimodal checkpoints that do not declare `ForCausalLM` will never
   show up this way -- the Gemma 4 rows are curated for exactly that reason.
4. **Restart ComfyUI.** `INPUT_TYPES` runs at registration. A download that
   finished while the server was up is invisible until then.
5. **An uncurated id has no fit tags.** Curated labels look like
   `Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv16 nv24)`. A cache-discovered id is
   the bare `org/name`. Do not infer tags for it.
6. **Set Quant yourself** unless you picked one of the two Qwen identities.
   Those two own Quant: NF4 is `bnb_nf4`, full is `none`. A mismatch fails
   loud on purpose. Every other pick still uses the Quant widget -- Gemma,
   Llama, your cache folder, a cloud handle.
7. **Keep both writer slots on the same id** unless you mean to swap two
   models in and out of VRAM all run.
8. **The technical slot has to emit JSON the pipeline can parse.** Beautiful
   prose that cannot hold a schema dies after the expensive load.
9. **Nothing substitutes.** If it will not load, the run stops and names the
   id you picked.

A first Queue that sits still is usually the download. The console prints
`[OTR] Downloading ...`. Auto-download is designed, not a defect. Disk
headroom is the model size plus 5 GB on the cache volume.

---

## In the shipped dropdown -- git clone

This is a catalog row, not an adapter. One `CuratedModel(...)` in
`CURATED_LLM_MODELS` inside `nodes/_otr_model_catalog.py`.

`repo_id` is unique in the dropdown. Two rows may share one Hugging Face
snapshot only through `hf_repo_id` plus `implied_quant_policy` -- that is how
the two Qwen identities work. Do not invent a second HF repo for the same
weights.

### The seven gates

Run these in order. Each one has caught a real failure in this repo.

**Gate 1 -- the weights are actually on disk (or will download by name).** A
dropdown entry is a promise. Auto-download on first Queue is the designed
path when the snapshot is missing. Do not silently substitute a different
model to satisfy a selection.

**Gate 2 -- the VRAM tier is honest.** `vram_fit_tier` is `PASS`, `WARN`,
`UNKNOWN`, or `FAIL`. PASS means soak-tested inside the ceiling. WARN means
not soak-tested; **it stays in the dropdown** (operator 2026-09-06). What a
tier owes the user is honesty at the moment of choosing -- the badge number
and truthful `notes` -- not absence. Ripping a row is an explicit decision
about a specific model. UNKNOWN / FAIL do not ship.

**Gate 3 -- it loads under the Quant the pick claims.** NF4 for the NVIDIA
twin, `none` for the full twin. Watch resident VRAM, not the file size. An
unquantized load must not inherit a ceiling sized for 4-bit.

**Gate 4 -- it generates free-form prose.**

**Gate 5 -- it generates constrained JSON.** This is the gate that actually
fails, and prose passing tells you nothing about it. The hard-constraint
machinery is opt-in per call. Verify the specific call your new row will make
actually binds a schema.

**Gate 6 -- the chat template accepts the roles OTR sends.** OTR sends a
system + user pair. Gemma-2 rejects the system role; the generate path
already folds that. Qwen 3.5 4B needs `enable_thinking=False` on every
generate call -- both identities, because they share one Hugging Face repo.

**Gate 7 -- the context window is the file's truth.** Declare what the
artifact supports, not what a model card claims. KV cache is not free.

### Field contract

| Field | Meaning |
|---|---|
| `repo_id` | Dropdown identity. Hugging Face `org/name`, or `org/name:nf4` for a Quant twin. |
| `hf_repo_id` | Real Hugging Face repo when `repo_id` is a twin. Empty means `repo_id` is the HF id. |
| `implied_quant_policy` | When set (`none` / `bnb_nf4`), the pick owns Quant. Empty means the widget still decides. |
| `requires_auth` | Whether Hugging Face actually gates it. Measure it; do not copy the previous row. |
| `loader_backend` | Dispatch key. `transformers_safetensors` for an ordinary causal LM. |
| `vram_fit_tier` | Honesty at choose-time. |
| `approx_safetensors_gb` | Download size on disk, not VRAM resident. The badge prints this number. |

Defaulted but set explicitly on every production row: `prompt_profile`,
`chat_template_kind`, `stop_tokens`, `context_window`, `license`,
`license_audit_status`, `provider`.

**Licensing is not optional.** `license` + `license_audit_status` mirror a
per-repo audit at `docs/model-license-<sanitized>.md`. A Quant twin shares
the parent's audit; do not invent a second licence file for the same
weights.

`license_audit_status` on anything the canonical graph binds must be
`mit_equivalent`.

### Do not add a GGUF writer because a transformers twin exists

The writer GGUF registry is empty for that reason. If you only have a GGUF
on disk and no Hugging Face snapshot, that is your machine -- cache a
CausalLM or pick what is already in the list. Do not restore `GGUF_ROWS` to
document a second copy of Qwen.

### If it becomes a shipped default

The badge string is part of the saved widget value. Update `DEFAULT_LLM` /
`default_llm_option()` (canonical, full Qwen) or `fresh_llm_option()` (a
newly dropped node, NF4 Qwen), the writer `widgets_values` on
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
