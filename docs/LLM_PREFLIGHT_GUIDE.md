# LLM PREFLIGHT GUIDE -- adding your own writer model to OTR

**Audience:** anyone adding a local LLM row to the OTR catalog, or checking that
one already in the dropdown will actually run on their box.

**Why this guide exists.** On 2026-08-25 a sweep of the catalog found
`Qwen/Qwen2.5-14B-Instruct` sitting in the live model dropdown with **no weights
on disk at all**. Nothing failed, nothing logged, and nothing tested it -- the
row simply waited for someone to pick it and lose a render. A dropdown entry is
a PROMISE that a model will load. This guide is the set of checks that make that
promise true before it is offered.

Companion guides: `EXTENDING_OTR.md` (custom source banks) and
`SOURCE_BANK_GUIDE.md`. Same idea, different component.

---

## 0. The hardware contract you are preflighting against

| Constraint | Value |
|---|---|
| GPU | RTX 5080 Laptop, **16 GB** VRAM, Blackwell sm_120 |
| Real-world VRAM target | **14.5 GB** (`llm_vram_ceiling_gb`, canonical widget) |
| Models root | **`C:\ComfyUI-Models`** -- never a folder under the repo |
| HF cache | `C:\ComfyUI-Models\huggingface` (`HF_HOME` / `HF_HUB_CACHE`) |
| GGUF artifacts | `C:\ComfyUI-Models\LLM\converted\<subdir>\<file>.gguf` |
| Execution | **Sequential only.** No async CUDA streams, no queue refactors |
| Ollama | **Not used, not supported.** The GGUF lane is in-process llama-cpp-python -- no daemon, no sidecar, no port |

**The models root is resolved in code, not by convention.** The authority is
`nodes/_otr_gguf_backend.py::_models_root()`:
`OTR_COMFYUI_MODELS_ROOT` -> `COMFYUI_MODELS_ROOT` -> default `C:\ComfyUI-Models`.
Resolve the path through that before ever declaring a model missing. A `find`
under `Documents\ComfyUI` proves nothing and has produced false "missing" reports
that cost a needless multi-GB download.

**"Fits" means fits QUANTIZED, and the catalog's size field says so.**
`approx_safetensors_gb` is the **download size on disk, not the VRAM resident
size** -- the field's own comment says exactly that. Mistral-Nemo is 24 GB on
disk and gemma-4-12b is 23.9 GB, yet both are the shipped production writers,
because the canonical ships `quantization = bnb_nf4` (4-bit). Do **not** read a
>16 GB disk size as "will not fit". Read the `vram_fit_tier` instead.

---

## 1. The seven preflight gates

Run these in order. Each one has caught a real failure in this repo.

### Gate 1 -- THE WEIGHTS ARE ACTUALLY ON DISK

The failure this catches is the one that prompted the guide. Check the real
resolved path, not a guess:

* transformers row -> `C:\ComfyUI-Models\huggingface\hub\models--<org>--<name>\snapshots\`
  must exist and be non-empty.
* `gguf_native` row -> at least one registered quant must resolve to a regular,
  non-zero file. The code that decides this is
  `_otr_gguf_backend.gguf_native_row_on_disk()`.

**OTR must never silently download to satisfy a selection.** If your check makes
the network light up, that is itself a defect -- report it rather than accepting
the convenience.

### Gate 2 -- THE VRAM TIER IS HONEST

`vram_fit_tier` is a `Literal["PASS", "WARN", "UNKNOWN", "FAIL"]`, and it is the
field that decides whether a row belongs in front of a user.

* **PASS** -- soak-tested to load and generate inside the ceiling. The only tier
  a shipping row should carry.
* **WARN** -- "needs quantization or offload"; not soak-tested. **Operator ruling
  2026-08-25: a WARN row does not belong in the dropdown.** Rows that do not fit
  nicely are ripped, not carried. There is precedent: two community WARN-tier 12B
  rows were pruned on 2026-05-23, and `Qwen/Qwen2.5-14B-Instruct` on 2026-08-25.
* **UNKNOWN / FAIL** -- never ship.

Verify with `catalog.check_vram_fit(repo_id, context_tokens)` rather than by eye.

### Gate 3 -- IT LOADS UNDER THE DECLARED QUANTIZATION

Load it exactly the way production will: NF4 for the transformers lane, the
pinned quant for GGUF. Watch the resident VRAM, not the file size.

A live trap worth knowing: an **unquantized** load must not inherit a VRAM cap
that was sized for 4-bit. That was a real production bug (`2c524732`,
PBUG-20260825-03). If you add a row that cannot be quantized, its ceiling is a
different number and you must say so.

### Gate 4 -- IT GENERATES FREE-FORM PROSE

The cheapest real signal. `scripts/otr_gemma4_doctor.py` is the reference
implementation of gates 3-5: it loads `google/gemma-4-12b-it` from
`C:\ComfyUI-Models` under the same NF4 policy OTR uses, generates one prose
sample, then one constrained JSON receipt -- with no overlay, LoRA, Ollama,
llama.cpp sidecar, HTTP request, or port. Copy that shape for a new row.

### Gate 5 -- IT GENERATES CONSTRAINED JSON

**This is the gate that actually fails, and prose passing tells you nothing about
it.** A model that writes beautiful prose can still be unable to fill a schema.

**CORRECTED 2026-08-25 (found by Cursor, mid-sweep) -- the line this replaced
claimed OTR's structured passes ARE schema-constrained. That is not true for
every lane.** The hard-constraint machinery genuinely exists and genuinely
works:

* transformers lane -> lm-format-enforcer via `prefix_allowed_tokens_fn`
  (`nodes/_otr_constrained_generate.py`).
* `gguf_native` lane -> llama-cpp `response_format` grammar.

**But it is opt-in per call, via `_otr_bind_schema` on the generate closure
(`nodes/OTR_LedgerScriptWriter.py:729-737`), and most callers never opt in.**
The only LIVE production consumer is the writer's own top-level SlotJobFields
pass (`OTR_LedgerScriptWriter.py:4407-4415`). `scifi_news_pro`'s technical-slot
structured calls (dossier, cast_aliases, news_read, casting_voices) are
**post-validated only** -- the schema reaches the prompt as text instruction
and the parser as a validation contract, never the sampler. This is why a
model can generate what looks like well-formed JSON in the log and still fail
"no decodable top-level JSON object found": nothing stopped it from sampling
EOS, or an unbalanced bracket, before the object actually closed.

`_otr_bind_schema` is leftover from a sibling lane (`_otr_scifi_codex`) ripped
2026-08-16 (PBUG-20260816-01); nothing else was ever pointed at it, and
`docs/WRITER_INPUT_MATRIX.md` already documents this as a known, live
extension point rather than a shipped path. **Do not assume gate 5 is
enforced just because the machinery exists in the repo -- verify the specific
call your new row will make actually binds a schema**, e.g. by checking
whether `slot_fn` carries `_otr_bind_schema` AND whatever wrapper sits between
your caller and it (a call-counter, a retry shim) actually forwards `_otr_*`
markers rather than stripping them.

Known sharp edges: lm-format-enforcer 0.11.3 mishandles a numeric JSON-Schema
`const` (production schemas use bounded ints instead), and a reasoning model can
spend its entire budget inside a `<think>` block and emit a degenerate `{}` --
which is why `Qwen3` rows carry a `think_policy` of `qwen3_no_think`. If your
model reasons out loud, it needs the same treatment.

### Gate 6 -- THE CHAT TEMPLATE ACCEPTS THE ROLES OTR SENDS

OTR sends a system + user pair. Not every template accepts a system role.

The documented example: **Gemma-2's chat template rejects the system role**, so
the generate path normalizes system messages before `apply_chat_template`. A new
row whose template is stricter than its `chat_template_kind` claims will fail at
the first real call, not at load.

### Gate 7 -- THE CONTEXT WINDOW IS TRUE

Declare what the FILE supports, not what a model card claims. A row once
advertised a 4096 context while the artifact said 262144 (`805123ea`).

Resolution goes through `catalog.resolve_context_cap`, which returns a tiered
verdict (PASS for a curated override, WARN for a parsed `config.json`, UNKNOWN
when unresolved) and clamps everything against `HARD_VRAM_CONTEXT_LIMIT`. Add a
`CURATED_CONTEXT_OVERRIDES` entry only for a value you have actually soaked.

Budget reality: context is not free. KV cache scales with `n_ctx`, and a GGUF row
at 8192 costs roughly double its 4096 footprint. A 2048-ctx pairing with a 12B
writer is what made the whole 8 GB profile family unable to run its own writer.

---

## 2. Adding a row -- the field contract

A curated row is a `CuratedModel` in `nodes/_otr_model_catalog.py`. Required:

| Field | Meaning |
|---|---|
| `repo_id` | The HF repo id, exactly |
| `requires_auth` | `True` for a gated repo |
| `loader_backend` | `transformers_safetensors`, `transformers_multimodal_text_only`, `transformers_gptq_int4`, `gguf_native`, or an HTTP backend |
| `vram_fit_tier` | `PASS` to ship (see Gate 2) |
| `approx_safetensors_gb` | Download size on disk -- **not** VRAM resident |

Defaulted but set explicitly on every production row: `prompt_profile`,
`chat_template_kind`, `stop_tokens`, `context_window`, `license`,
`license_audit_status`, `provider`.

**Licensing is not optional.** `license` + `license_audit_status` mirror a
per-repo audit at `docs/model-license-<sanitized>.md`, and
`docs/model-license-audit-targets.txt` tracks the set. A new row owes an audit
file in the same change.

### GGUF rows are stricter, deliberately

A `gguf_native` row is a virtual catalog peer projected from
`_otr_gguf_backend.GGUF_ROWS`. Its artifacts table maps quant -> (filename, size,
sha256), and **a quant is PINNED only when it carries BOTH a size and a sha**.
An unpinned quant skips integrity checking entirely -- which once let a truncated
download be reported ready, on the very quant the 8 GB profile selected. Pin by
MEASUREMENT on the real box, never by transcription from a model card.

---

## 3. What a preflight must never do

* **Never download a model without explicit, exact-name authorization.** A guide
  that quietly fetches 28 GB is worse than a red X.
* **Never let a skip read as a pass.** A row skipped because its weights are
  absent must be reported as SKIPPED, by name, with the reason. Silent coverage
  truncation is a failure class this project fights everywhere else; a preflight
  is the last place it belongs.
* **Never substitute silently.** If the selected model cannot load, the run stops
  loudly. There is no fallback model, by operator directive.
* **Never write new behaviour into `config/profiles/*.json`** -- that channel is
  on the retirement list.

---

## 4. The short version

```
1. Weights on disk at C:\ComfyUI-Models  (resolve via _models_root(), do not guess)
2. vram_fit_tier == PASS                 (WARN rows get ripped, not shipped)
3. Loads under NF4 / pinned quant        (watch resident VRAM, not file size)
4. Generates prose
5. Generates CONSTRAINED JSON            <- the gate that actually fails
6. Chat template accepts system+user     (Gemma-2 rejects system)
7. Context window is the FILE's truth    (and the KV cache is not free)
```

A row that clears all seven belongs in the dropdown. A row that clears six does
not, and the honest move is to say which one it missed.
