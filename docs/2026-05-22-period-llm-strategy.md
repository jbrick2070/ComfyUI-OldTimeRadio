# Period-LLM strategy — creative-slot period model

**Date:** 2026-05-22
**Branch context:** v2.0-alpha
**Status:** Recommendation. Exploration, not a locked build.
**Round-robin:** Waived — Jeffrey requested a Claude-only recommendation
(the ChatGPT/Gemini consult scripts need registry-scoped API keys +
network not reachable from this session's sandbox). This document is a
single-author recommendation, not a multi-model consensus.

---

## 1. Context

The catalog row `talkie-lm/talkie-1930-13b-it` was removed 2026-05-22
(commit `2f01136`). It pointed at a raw research checkpoint
(`rl-refined.pt`, no `config.json`, no tokenizer) that the transformers
loader cannot load — it crashed the writer at the style picker, and the
catalog mislabeled it as a 7.5 GB GPTQ-int4 model.

That row was the only `prompt_profile="otr_1940s_v1"` row and the only
`transformers_gptq_int4` row. With it gone, the period-routing surface
is intact but **unreachable**: the `otr_1940s_v1` profile, the creative
router's period branch, `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT`,
and the GPTQ-int4 backend are all parked. No curated row triggers them.

Question: what period model — if any — should fill the writer's
creative slot, and how.

---

## 2. Options evaluated

Four candidates were weighed against the CLAUDE.md prime directives:
audio byte-identity (#1), the 14.5 GB VRAM ceiling (#2), workflow-JSON
wiring (#3), and the two-model selector contract (#6 — every LLM call
routes through the writer's two in-process slots; no node but the
writer exposes a model id).

### A. Wait for Ranke-4B-1946 (`uzh-echist-org`)

| Dimension | Assessment |
|---|---|
| Architecture | Qwen3, 4B. Standard `transformers` safetensors — loads on the **existing** `transformers_safetensors` backend. Zero new loader code. |
| Era fit | Knowledge cutoffs 1913 / 1929 / 1933 / 1939 / 1946. The 1939 and 1946 cutoffs straddle OTR's 1938–1952 target. A real knowledge cutoff is a stronger anachronism guard than any prompt — the 1946 build literally cannot know post-1946 events. |
| VRAM | 4B ≈ 8 GB on disk, ~4 GB resident. The smallest writer model in the catalog. Trivial fit on 16 GB. |
| Determinism / C7 | Standard Qwen3 + transformers + temp 0 → C7-testable like any other curated row. No exotic risk. |
| Blocker | **Not released.** The org page is empty; timeline unknown. A catalog row cannot point at a repo that does not exist. |

**Verdict: the ideal technical fit. The period routing was designed for
exactly this shape — a model-bound `prompt_profile`. The correct action
is a release watch, not a build.**

### B. MonadGPT flavor experiment (`Pclanglais/MonadGPT`)

| Dimension | Assessment |
|---|---|
| Architecture | 7B Mistral, apache-2.0. Clean packaging, loads on the existing safetensors backend. |
| VRAM | 7B ≈ 14 GB disk, ~7 GB resident. Fits 16 GB. |
| Era fit | **1400–1700 (early-modern English).** OTR targets 1938–1952 — a 250+ year mismatch. MonadGPT produces "thee/thou/hath" Shakespearean diction, the exact failure mode the period prompt forbids. |

**Verdict: drop. Wrong era by centuries. An experiment would only
confirm the mismatch; it exercises the catalog/loader plumbing no
better than option C already does.**

### C. Modern model + `OTR_PERIOD_SYSTEM_PROMPT`

| Dimension | Assessment |
|---|---|
| Cost | Zero VRAM cost, zero C7 risk — same model, different system prompt. |
| Era fit | A modern model prompted to avoid anachronisms is decent but leaky under pressure. `OTR_PERIOD_SYSTEM_PROMPT` is already detailed (diction rules, broadcast convention, era constraints), and the ROADMAP's C7 protocol step 4 (period-tone smoke pass) exists precisely because prompt-based period control is imperfect. |
| Hidden cost | **Not actually "free / already plumbed."** The router selects the period prompt only when a catalog row carries `prompt_profile="otr_1940s_v1"`. The catalog is keyed by `repo_id`, so you cannot add a second Mistral-Nemo row with the period profile (dict collision). Reaching the period prompt with a *modern* model needs a deliberate router/writer change — a per-episode `period_mode` toggle (see §4). |

**Verdict: viable interim ONLY if period scripts are wanted before
Ranke ships. It is a small, deliberate router/writer change — not a
free flip of an existing surface.**

### D. Talkie quantized-GGUF sidecar (raised by Jeffrey 2026-05-22)

Community GGUF builds of the talkie instruct model exist and are real:
`thomasgauthier/talkie-1930-13b-it-GGUF` (Q4_K_M ≈ 8.57 GB),
`sol-wy/talkie-1930-13b-it-q5` (Q5_0 ≈ 9.13 GB). The proposal: run
Talkie as a localhost HTTP sidecar doing a per-line period *polish*
pass, with a modern model as the story brain and a modern validator
cleaning the output.

The idea is sound in spirit — period diction is a polish problem, not a
story-generation problem — but it conflicts with four directives as OTR
is currently architected:

1. **GGUF is not a catalog format.** `_otr_model_catalog._structural_reject`
   explicitly rejects `.gguf`; `ALLOW_PATTERNS` excludes it ("deferred
   to a future llama.cpp backend"). A GGUF model cannot be a normal
   catalog row — it needs a new llama.cpp backend that does not exist.
2. **Custom architecture → patched fork.** Talkie uses non-standard
   blocks (custom tensor mapping, QK norm/gain, RMSNorm behavior).
   Per the community GGUF pages it does **not** run on stock
   llama.cpp / LM Studio — it needs a patched fork. A patched fork is a
   fragile, niche dependency that breaks on every upstream bump. It is
   open-source-compatible but a real maintenance burden.
3. **A sidecar breaks the two-model contract + VRAM accounting.**
   Prime Directive #6: every LLM call routes through the writer's two
   in-process slots. A localhost HTTP server is a third LLM outside
   that contract. Prime Directive #2: OTR cannot `_flush_vram_keep_llm()`
   a process it does not own — an 8–9 GB sidecar holds VRAM *outside*
   OTR's accounting, competing with FLUX/HuMo on a 16 GB card. That is
   a real OOM risk, not a hypothetical.
4. **C7 / audio.** Quantized sub-8-bit GGUF inference is non-deterministic
   (split-K GeMM, non-associative FP accumulation — the ROADMAP already
   documents this for quantized visual inference). The LLM C7 protocol
   requires byte-identical draft text at temp 0. A non-deterministic
   per-line rewrite makes the script text — hence the cast contract,
   hence the audio line set — non-reproducible. That is a direct
   Prime Directive #1 risk.
5. **Era.** Talkie's corpus is pre-1930; OTR targets 1938–1952. Closer
   than MonadGPT, but it knows nothing of WWII, which is central to the
   era. Manageable as flavor, not era-accurate.

**Verdict: a legitimate research-lane idea, not a near-term build. File
it as a v2.1+ spike gated on (a) a general llama.cpp backend landing in
OTR, (b) a C7 determinism proof on the quantized model, (c) a VRAM-
accounting story for the sidecar, (d) confirmation the patched fork is
maintainable offline.**

---

## 3. Recommendation

1. **Primary — wait for Ranke-4B-1946.** It is the model the period
   architecture was built for: a model-bound `otr_1940s_v1` profile,
   standard safetensors (no new loader code), trivial VRAM, and a real
   1946 knowledge cutoff. Set a release watch on the `uzh-echist-org`
   org. When it ships, adoption is one clean catalog row (§4).

2. **Interim — modern model + period prompt, only if wanted now.** If
   period scripts are needed before Ranke ships, add the `period_mode`
   toggle (§4). It is a small deliberate change, not free. If period
   flavor can wait, do nothing — OTR runs fine on modern models and the
   vintage *sound* already lives in the DSP master chain (ROADMAP
   locked position).

3. **Drop MonadGPT.** Wrong era by centuries.

4. **Talkie GGUF sidecar → research-lane, v2.1+.** Documented in §2.D
   with its four blockers. Revisit only if/when OTR grows a general
   llama.cpp backend for other reasons.

**Parked-surface note:** the `otr_1940s_v1` profile and
`_otr_period_prompts.py` stay — Ranke uses them. The
`transformers_gptq_int4` backend is now genuinely orphaned (Ranke is
Qwen3 safetensors, not GPTQ; the GGUF sidecar would need a llama.cpp
backend, not this one). It is harmless parked, but it is a candidate
for a future cleanup sweep if no quantized-transformers model is ever
adopted.

---

## 4. Wiring spec (CLAUDE.md prime directive #3)

### 4a. Ranke-4B-1946 catalog row — template, pending release

When Ranke ships, adoption is a single `CuratedModel` row appended to
`CURATED_LLM_MODELS` in `nodes/_otr_model_catalog.py`. Fields marked
`# CONFIRM` must be verified against the actual release before merge:

```python
CuratedModel(
    repo_id="uzh-echist-org/Ranke-4B-1946",   # CONFIRM exact id on release
    requires_auth=False,                      # CONFIRM (gated repo?)
    loader_backend="transformers_safetensors",# Qwen3 = standard; no new code
    vram_fit_tier="UNKNOWN",                  # -> PASS after a soak test
    approx_safetensors_gb=8.0,                # CONFIRM (4B BF16 ~8 GB)
    notes=(
        "Period model -- Qwen3 4B, 1946 knowledge cutoff. Real era "
        "cutoff is the anachronism guard; pairs with otr_1940s_v1."
    ),
    prompt_profile="otr_1940s_v1",            # triggers the period router
    chat_template_kind="transformers_default",# Qwen3 ships a chat template
    stop_tokens=(),                           # CONFIRM
    context_window=8192,                      # CONFIRM native; clamp applies
    license="apache_2_0",                     # CONFIRM on release
    license_audit_status="pending",           # -> after docs/model-license audit
),
```

Required alongside the row, in the **same commit**:

- Add `uzh-echist-org/Ranke-4B-1946` to
  `docs/model-license-audit-targets.txt` and create
  `docs/model-license-uzh-echist-org--ranke-4b-1946.md` (the bijection
  in `test_catalog_matches_audit_files.py` enforces both).
- The writer's `creative_writing_model` dropdown picks it up
  automatically (`dropdown_choices()` builds from `CURATED_LLM_MODELS`).
  No new widget — prime directive #6 holds.
- The default workflow JSON (`otr_scifi_16gb_full.json`) is **not**
  touched: `test_default_workflow_validator` blocks any non-
  `mit_equivalent` row from default binding. Ranke is opt-in via the
  dropdown until its license audit clears.
- Re-point or re-author the period runtime tests
  (`test_period_creative_runtime.py` was deleted with the talkie
  removal) against the real Ranke repo.

**VRAM fit (16 GB RTX 5080):** 4B params, ~4 GB resident estimate via
the catalog's halved-download heuristic — the smallest writer model in
the set. Co-loads or slot-swaps under the 14.5 GB ceiling with any
technical-slot model with wide margin. VRAM is a non-concern for a 4B.

### 4b. `period_mode` toggle — interim option C, only if built

If a period script is wanted before Ranke ships, the period prompt is
reachable with a modern model via a per-episode toggle. This is a
deliberate change, scoped here so it is not mistaken for free:

- `OTR_LedgerScriptWriter`: add a `period_mode` BOOLEAN widget
  (default `False`). A boolean is not a `model_id`, so prime directive
  #6 is not violated.
- `_otr_creative_prompt_router.resolve_creative_system_prompt`: add a
  `period_mode: bool = False` parameter. Return
  `OTR_PERIOD_SYSTEM_PROMPT` when `period_mode` is True **or** the
  row's `prompt_profile == "otr_1940s_v1"`.
- Thread `period_mode` from the writer through the four creative-phase
  call sites.
- Wire the new widget into `otr_scifi_16gb_full.json` (directive #3)
  and update the workflow-JSON guardrail tests + the router tests.
- Tag: the period prompt is a creative-slot concern — `# LLM slot:
  creative`, no new model pick.

This is ~1 widget + 1 router-signature change + wiring. Worth doing
only if period output is needed before Ranke; otherwise skip it and
wait for the clean model-bound path.

---

## 5. Action items

- [ ] Set a release watch on the `uzh-echist-org` HF org for Ranke-4B
      (1946 build especially).
- [ ] Decide: is a period script wanted *before* Ranke ships? If yes,
      build §4b. If no, do nothing — parked surface stays parked.
- [ ] (Future cleanup, optional) decide whether to remove the orphaned
      `transformers_gptq_int4` backend or keep it parked.
- [ ] When Ranke ships: apply §4a, soak-test VRAM, run the license
      audit, re-author period runtime tests.
