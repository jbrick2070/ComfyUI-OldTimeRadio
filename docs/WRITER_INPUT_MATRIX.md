# WRITER INPUT MATRIX -- lanes, context, and the knobs that actually bind

**HAND-WRITTEN, AND THAT IS A KNOWN WEAKNESS.** Verified against the tree at
`0d00bbf1` (2026-08-13). `docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` says
plainly that a hand-written doc cannot be trusted with numbers -- it was written
after two such docs rotted in one week -- and its item 5 already anticipates an
LLM/writer inventory doc. **This file is the CONTENT for that doc, not the
finished form.** Until a generator + byte-exact drift gate exists (the pattern
`docs/ENGINE_MATRIX.md` and `tools/engine_matrix.py` already prove), every
number below is a claim with a date on it, not a guarantee.

**Before trusting any row: grep the cited symbol.** That instruction is the
whole reason this header exists.

Companion records: `docs/2026-08-13-writer-runaway-root-cause.md` (the codex-lane
runaway and its fix), `docs/2026-08-13-cross-bank-runaway-exposure.md` (what that
fix does NOT cover).

## 1. Lane map -- which bank runs which writer, and what shape it decodes

The single most load-bearing fact in this file, and it INVERTED on 2026-08-16:
**no shipped lane binds a grammar during decoding any more.** The only lane that
ever did was the codex runner, retired with the `scifi_news` rip
(PBUG-20260816-01). The writer still EXPOSES the capability (the
`_otr_bind_schema` hook set at `OTR_LedgerScriptWriter.py:730`) and NOTHING consumes it --
live extension space, not a live path. Every shipped lane is post-validated: a
schema instruction in text, parsed after the fact.

| bank | pipeline (`nodes/story_packs/banks.json`) | module | decode shape | grammar-bound? | reserves whole window? |
|---|---|---|---|---|---|
| `scifi_news_pro` | `scifi_news_pro_multipass` | `_otr_scifi_news_pro` | post-validated JSON + **raw markup P3** | no | **YES -- critical** |
| `media_archive` | `legacy_many_pass` | writer inline body | post-validated JSON + inline | no | no |
| `shakespeare` | `legacy_many_pass_adapt` | writer inline body | post-validated JSON + inline | no | no |
| `public_domain` | `legacy_many_pass_adapt` | writer inline body | post-validated JSON + inline | no | no |
| `original` | `original_multi_pass` | writer inline body + front passes | post-validated JSON + inline | no | no |

**Why "grammar-bound" is the column that matters:** a `max_length` on a
post-validated model does NOT stop a decode. Without token-selection binding it
is a post-hoc validation failure -- the tokens are already spent. That is now
true of EVERY shipped lane, so schema ceilings are hygiene everywhere and the
runaway guard has to be the numeric output ceiling, never the schema.

## 2. Context cap -- what wins, in order

For `mistralai/Mistral-Nemo-Instruct-2407` the answer is **16,384**, and the
route there matters more than the number:

1. `CURATED_CONTEXT_OVERRIDES` (`_otr_model_catalog.py:1411`) pins 16,384. Every
   other local row is 8,192.
2. The override is authoritative because the row is `vram_fit_tier=="PASS"` and
   `OTR_HARD_VRAM_CONTEXT_LIMIT` is blank; a nonblank value clamps to
   `min(16384, hard)`.
3. The model's own `config.json` advertises **131,072** and is IGNORED -- the
   curated branch returns before `_read_config_context()`. The loader then
   lowers `max_position_embeddings` to the resolved cap.
4. The cap is stamped into the resident cache entry.
   `HARD_VRAM_CONTEXT_LIMIT` is NOT part of the policy cache key, so **a cache
   hit keeps the old cap** -- an env change cannot re-cap a resident model.

### Misleading names, each of which has cost someone time

| Looks like | Actually |
|---|---|
| `CuratedModel.context_window` | not the runtime source of truth; the curated override binds |
| `HARD_VRAM_CONTEXT_LIMIT = 8192` | NOT a default cap for Mistral -- only binds when the env is explicitly nonblank |
| `prompt_must_fit=True` on P3 | never wins; `ProviderCapacityMessages` is selected first |
| `context_cap or 8192` in generation | a malformed-cache fallback only; `load_llm()` always supplies the cap |
| `GEMMA4_12B_MAX_NEW_TOKENS` | live for GGUF Gemma, **dead for Mistral** (different backend branch) |

## 3. How P3 got 13,912 output tokens

```
context_cap                 16,384      curated override
measured P3 prompt           2,472      the real tokenized chat prompt
available                   13,912      cap - prompt
effective max_new_tokens    13,912      min(requested, available)
```

`ProviderCapacityMessages` + `max_new_tokens=None` means "reserve the full
remaining capacity". There is no half-window split, no completion reserve, and
the schema ceilings do NOT supply this number. The only floor is 64 usable
output tokens, so a prompt is refused only past ~16,320.

**The tokenizer question, settled:** the project does not pass
`fix_mistral_regex=True` and transformers warns about it. That does NOT make the
arithmetic above wrong -- `input_len` is read from
`inputs["input_ids"].shape[-1]`, the exact tensor handed to `generate()`
(`OTR_LedgerScriptWriter.py:902`). It is a tokenizer-correctness question with no
established direction of error, not a hidden undercount.

## 4. Env knobs on the writer input path

"LIVE" means an already-running server reads it again on a future render. A
`setx` or shell export outside a running server NEVER reaches that process --
this is `PBUG-20260723-02` and it has cost multiple sessions.

| Variable | Status for the Mistral input path |
|---|---|
| `OTR_HARD_VRAM_CONTEXT_LIMIT` | **BOOT-ONLY** (value captured at import; cap only enters a COLD cache entry) |
| `HF_HOME` | **BOOT/FIRST-LOAD ONLY** |
| `OTR_SOURCE_SNAPSHOT_MANIFEST` | **LIVE** -- re-read per episode, changes P3 input directly |
| `OTR_BANK_SEED` | **LIVE, conditional** (roll sentinel only) |
| `OTR_WRITER_UNLOAD_AFTER_SCRIPT` | **LIVE, indirect** -- decides whether the next render reuses the stamped cap |
| `OTR_MODEL_CATALOG_AUTO_DOWNLOAD` | **LIVE**, acquisition only |
| `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` | **DEAD** -- `ensure_hf_home()` overwrites it |
| `HF_TOKEN` | **DEAD** once Mistral is cached |
| `OTR_MODEL_CATALOG_ALLOW_REMOTE` | **DEAD** for a curated row |
| `GEMMA4_12B_*`, `OTR_GGUF_*` | **DEAD** for Mistral -- GGUF branch only |
| `OTR_CAST_SEED`, `OTR_STYLE_SEED`, `OTR_NAME_MODE`, `OTR_ENABLE_STYLE_GRAMMAR` | live for INLINE lanes, **dead for the codex P3 path** (dispatch happens first) |
| `OTR_SOURCE_BANK_CACHE_DIR`, `OTR_CAST_GENRE`, `OTR_OTHER_NAME_POLICY` | **DEAD** -- defined, no callers |

**One nuance that is easy to over-read.** `OTR_C7` is NOT parsed by the writer --
it is log/comment text on the Python side, and dead for token input. But it is
NOT inert: `scripts/_otr_soak_server_launch.cmd` branches on `if defined OTR_C7`
and pins `OTR_CAST_SEED` / `OTR_STYLE_SEED` / `OTR_FABLE2_SEED` to 42 at BOOT.
So "dead for writer token input" and "does nothing" are different claims, and
only the first is true.

## 5. Per-call output ceilings, by path

The inline lanes are safe from a full-window runaway because every call carries a
NUMERIC ceiling -- not because they all use stop strings. Only
`compose_line_draft` and the announcer pass stop strings; exchange, title,
structured calls and slot-fill do not.

| path | ceiling | stops |
|---|---|---|
| codex P3 / P5 | whole remaining window | none |
| fable2 dossier | 700 | none |
| fable2 pitch / treatment / news read / casting | whole remaining window | none |
| **fable2 P3 raw markup** | **whole remaining window** | **none, and no schema either** |
| media / shakespeare / public-domain briefs | 520 | none |
| original concept / selection / briefs | 1400 / 1000 / 900 | none |
| outline macro / phase / beat | 250 / 200 / 150 | none |
| cast description | 250 | none |
| dramatic state / continuity | 512 / 1200 | none |
| `SlotJobFields` (the one inline lmfe pass) | 192 | none |
| `compose_exchange` | 320 | none |
| `compose_line_draft` | widget 40-400 | `\n\n`, `\n[`, `\n(` |
| announcer intro/outro/coda | 320 | same |
| title | 160 | none |

## 6. What this file owes

* **A generator and a byte-exact drift gate**, per
  `docs/2026-08-06-SPEC-subsystem-matrix-pattern.md`. Sections 2, 3 and 5 are all
  numbers read from live code and are exactly what rots. Until then this file is
  dated evidence, not a contract.
* Verification of section 5's fable2 and inline ceilings by the driver -- those
  rows come from an audit lane and were NOT individually re-read. Sections 1, 2
  and 3 were verified directly.
