VERDICT: yes-with-fixes -- EOS owners are complete in the five snapshot files; the locked next GPU action and the leftover UNVERIFIABLE stop-token proof are still ambiguous enough to waste pairlock02 or invite a sampling patch.

MUST-FIX BEFORE BUILD:
1. [Decision Q2/Q3 + Implemented correction + GO_FORWARD "Next action"] Two incompatible next steps are still live: (a) QA then one canonical retry of this candidate, vs (b) a pre-retry min_p/top_p/NF4 patch. Lock (a). Sampling stays unchanged (`OTR_LedgerScriptWriter.py` widget defaults min_p=0.05, top_p from creativity, repetition_penalty=1.03; pairlock_01_prompt.json min_p=0.05). Do not re-run HEAD 2c9d47f2; that is the failed live commit. The GPU retry is the snapshot candidate after remaining QA. Why: P1 cannot be ended by EOS (LMFE only admits EOS after a complete object: `tests/test_constrained_generate.py` `test_real_lmfe_uses_shared_model_chat_eos_and_refreshes_without_mutating_history`; live P1 tails are still inside fields). Bible-class min_p lock-in (`docs/2026-08-13-writer-runaway-root-cause.md`) is CONFIRMED for a locked nucleus, INFERRED as this run's trigger (no logits). Changing it is speculative, 5080+4060 blast radius, and contradicts "No model/tokenizer config, retry budget ... or sampling policy changed."
2. [Live evidence P1 / PBUG-20260910-05] Strike any builder reading that P1 "fired the open-string guard." All five live halts are `halt_reason=verbatim_cycle` with `open_string_tokens=None` (`docs/2026-09-11-my-story-5080-qualification/pairlock_01_server.log` lines 266, 288, 307, 333, 369). That None is owned: `ProviderCapacityMessages._otr_unbounded_json_field = True` (`nodes/_otr_generation_budget.py`) and the writer passes `max_open_string_tokens=None`, which disables `OpenStringTracker` entirely (`nodes/_otr_decode_guard.py` `make_degeneracy_criterion`). P1-in-string is CONFIRMED from RUNAWAY EVIDENCE text (attempt 1 `dramatic_question` spiral; attempts 2-3 schema-contract echo including `[/OTR_SCHEMA_CONTRACT_V1]`), not from the tracker. Do not "fix" this by re-arming `MAX_OPEN_STRING_TOKENS` on provider-capacity JSON; that is a length ceiling the guard's own comments forbid.

SHOULD-FIX:
1. [Live evidence "Exact hidden generated EOS token sequence was not persisted" / pairlock_01_failure_ledger.json `fit`] Source-rewrite `fit` on 2c9d47f2 has prompt/capacity fields and no `eos_token_ids` (ledger ~233-244). Candidate `inspect_native_prompt_fit` now returns `eos_token_ids` (`nodes/_otr_model_loader.py`). Degeneracy RUNAWAY EVIDENCE still does not print last token id or the eos list (`OTR_LedgerScriptWriter.py` DECODE HALTED / RUNAWAY EVIDENCE). Smallest add: one log field `eos_token_ids=... last_token=... ended_with_eos=...` on that existing error line. Without it, pairlock02 P0 success is only circumstantial (no 82/87 trailing newlines in `raw_completion`).
2. [tests/test_generation_budget.py `test_all_native_routes_stop_on_configured_or_chat_eos_at_capacity`] Stubs `model.generate` to append the terminal id, then asserts `EosTokenCriteria`. That proves kwargs + classification, not that Transformers 5.10.4 would have stopped. Accept as unit scope; do not treat 240 focused tests as a live cure (already stated -- keep it).
3. [docs/GO_FORWARD_PLAN.md WHERE TO PICK UP] "5080 and scoped RunPod trial are authorized now that included coding is complete" vs "No more GPU runs until this revision has regression/Bible and another finished-code Sonnet QA" vs pairlock_01_request.json `"runpod": "not started; current session has no authenticated RunPod API"`. Fold to: QA first, 5080-only retry, RunPod/Mac/4060 stay dark.

OPTIONAL / NICE-TO-HAVE:
- If pairlock02 P1 still dies on in-string verbatim_cycle, inspect typed-repair prompt echo of `_SCHEMA_CONTRACT_MARKER` (`nodes/_otr_structured_call.py`) as a P1-only follow-up. Out of this EOS campaign.
- Assert after complete JSON that LMFE still allows whitespace; that would explain a P0 newline run even with aligned EOS. [ASSUMPTION] trailing whitespace remains grammar-legal.

CUT THESE:
1. Pre-retry min_p/top_p/repetition_penalty retune -- lock-in mechanism is documented; this run has no logits; sampling policy was explicitly frozen.
2. NF4 / model / quantization swap -- one failed tuple; NF4 has prior NVIDIA proof; not causal here.
3. Re-enable open-string 2048 cap on My Story provider-capacity calls -- cycle detector already halted; operator forbids prose-size bounds.
4. Persisting the full generated token sequence -- last token + eos list is enough; full dumps are cost with no extra owner.
5. Fourth/fifth rewrite, chunker, word/character clip, RunPod, Mac/4060 -- already forbidden; pairlock_01_request.json confirms no RunPod auth.
6. Extra kibitz rounds on this EOS union -- no remaining owner mismatch in the four native factories.

VERIFY-AT-BUILD checklist:
- verify: sha256 of the five snapshot.json paths equals the listed hashes (`nodes/_otr_model_loader.py` 468597c1..., `nodes/OTR_LedgerScriptWriter.py` c25ff162..., `nodes/_otr_constrained_generate.py` d8298bf9..., `tests/test_generation_budget.py` 44afff3e..., `tests/test_constrained_generate.py` 1d23a5df...). Shell hash was unavailable here.
- verify: git HEAD vs 2c9d47f2 vs this dirty tree. Conversation git_status shows those five Python files plus GO_FORWARD/PROD_BUG_LOG modified; live failure was 2c9d47f2. Do not GPU-test 2c9d47f2 as the candidate.
- verify: 240 focused tests and the in-flight full suite / Bible result. Not re-run here.
- `native_eos_token_ids` on the live Qwen cache_entry is nonempty union of generation_config/text EOS 248044 and tokenizer/chat EOS 248046, no mutation of those owners (`nodes/_otr_model_loader.py` `native_eos_token_ids`).
- All four native factories pass `eos_token_id=prepared["eos_token_ids"] or None` and classify EOS-at-capacity as complete: writer `OTR_LedgerScriptWriter.py` gen_kwargs; constrained `_otr_constrained_generate.py`; base/polish `_otr_model_loader.py`. Remote/GGUF still return before `prepare_native_prompt`.
- LMFE cache identity is tokenizer identity + EOS tuple; new `tokenizer_data` before enforcer; parsers/prefix_fns request-local (`get_cached_transformers_schema_constraint`).
- `prepare_native_prompt` still forwards `enable_thinking=False` for `Qwen/Qwen3.5-4B` (`nodes/_otr_loader_backends.py` `chat_template_kwargs`; `_exact_prompt_entry` asserts it).
- pairlock02 proof, no new GPU until QA is green:
  1. Source-rewrite `fit.eos_token_ids` contains 248044 and 248046.
  2. P0 `raw_completion` is complete JSON without the prior 82/87 trailing newlines (`pairlock_01_failure_ledger.json` raw_completion). If newlines persist, EOS alignment did not suffice -- whitespace after `}` is a separate sampling/grammar fact, not a missing eos owner.
  3. If last-token logging lands, last id in that eos set on clean P0 stop; if P1 dies, last id is NOT treated as an EOS miss while the tail is still inside a string.
  4. P1 `primary_ladder_exhausted` after 3 attempts is a known residual, not an EOS regression. Keep attempt01 in the denominator; no obs/media claim.

Answers to the three decision questions:
1. No remaining EOS owner mismatch in the four native paths. Residual is proof instrumentation, not an unowned terminator.
2. Next action is QA then one canonical retry of this candidate. Not a sampling/model patch first. CONFIRMED: tokenizer 248046 vs text/generation 248044; former LMFE-tokenizer-only vs generate-no-override; P0 complete JSON then whitespace cycle at 672; P1 three in-string verbatim cycles; thinking disabled; degeneracy is rerollable (`GenerationDegeneracyError` phase in `REROLLABLE_PHASES`). INFERENCE: that the missing chat EOS caused the P0 padding; that min_p=.05 suppressed the closing quote on P1.
3. Converge: ship the EOS union after QA; one pairlock retry with the verify list above. Disagree with any claim that this candidate has already cured My Story or that P1 is in-scope for this diff.
