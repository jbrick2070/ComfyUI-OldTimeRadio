<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Demonstrated / code-visible defects still outstanding

**1. `max_new_tokens` is effectively uncapped, and it is *not* a retained control.** The probe records `259817` / `258112` (Boundary result, attempts 1–3), i.e. context-minus-prompt. Swapping to Mistral Nemo changes the context window, so the single most permissive knob in the degeneracy path silently changes with the family. "Retaining all other controls" (opening paragraph) is therefore false as stated. Either pin an explicit `max_new_tokens` for both arms or stop calling the comparison controlled.

**2. The repair rung is a repetition amplifier — visible in the shown code.** `_full_artifact_repair` injects the entire degenerate draft as an `assistant` turn, and the ladder simultaneously drops temperature to 0.1 with `repetition_penalty` only 1.03 (native kwargs). Prompt grows 2327→4032 tokens, all growth being repeated text. Your own comparison shows a 3702-char identical prefix then divergence — exactly near-greedy collapse. This makes the Mistral run **non-discriminating**: a Mistral failure is fully explainable by prompt construction, not family, so the run cannot deliver the "family-coverage" evidence it is justified by.

**3. The boundary probe's tokenizer/grammar evidence does not transfer.** It ran "real cached Qwen tokenizer and installed LMFE" with EOS `[248044,248046]`. The Mistral arm exercises a different vocab, different `prepared["eos_token_ids"]`, and a different `prefix_allowed_tokens_fn` build. The one piece of positive evidence you have is scoped out of the arm you propose.

**4. Process-global `_min_p_unsupported[0]`.** A single earlier `TypeError` permanently disables `min_p` for the process, across slots and models. In a dual-slot canonical run this is cross-model contamination of a recorded sampling parameter. The probe cannot see it because no real `generate` ran. The `except TypeError` retry body is not shown and remains unverified.

**5. Cheap discriminator ignored.** `WRITER_SEED_ENV` seeding already exists (native call block). Two seeded same-prompt calls would settle "identical draws vs. ignored sampling" — the exact question your comparison says it cannot answer — at near-zero cost. Dropping it while spending a full canonical run is a bad trade.

## Risk / over-engineering

- **n=1, OOM-tolerant.** "RealOOM accepted; no new cap/gate" plus 24.0 GB dropdown vs 14.5 GB profile means a plausible outcome is *no data at all*. Reconcile the sizes or add a pre-flight VRAM assertion (not a gate).
- **Observability is under-, not over-, specified.** I am not asking for speculative telemetry: persist the same fields the probe already emits (per-attempt kwargs, prompt hashes, raw completion) to the ledger. Without them a pass/fail is uninterpretable.
- **Authority non sequitur.** Reviewers opposed the *title-only* theory; that is not support for the family swap.

## What the probe does resolve

Call count, ladder temperatures, fresh grammar per call, no `past_key_values` forwarded, prompt-3 containing source + failed text, token counts matching live04. Limits as you state: no logits, no weights, reconstructed error — and no evidence about model-internal caching or any non-Qwen path.