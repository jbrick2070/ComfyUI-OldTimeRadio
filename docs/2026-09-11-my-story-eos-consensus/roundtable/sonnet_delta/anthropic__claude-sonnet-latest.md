<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-sonnet-5 -->

No remaining must-fix in this scoped delta.

- The single-condition change at line 1184 (`generated_tokens == effective_max_new_tokens and not ended_with_eos`) correctly gates both OUTPUT_TRUNCATED and OUTPUT_CAP diagnostics on non-EOS capacity halts, matching the documented classification order (guard.hit → capacity-no-EOS → EOS → other).
- This is consistent with the later raise at line 1236-1237, which already carried `not ended_with_eos` — so the diagnostic block and the raise now share the same guard condition. No divergence between "what gets logged" and "what gets raised" remains.
- Guard.hit (line 1144) still runs first and unconditionally, so degeneracy classification precedes capacity classification regardless of `ended_with_eos` — order preserved.
- No sampling, retry, or correction-path code is touched; the diff is confined to the diagnostic conditional as claimed.
- The regression test's assertion block (last block, `if route == "writer"`) directly checks the exact three-way outcome: `ended_with_eos=True` in the log line, plus absence of both `OUTPUT_CAP:` and `OUTPUT_TRUNCATED:` substrings — this is a precise, checkable assertion of the fix and would have failed under the pre-fix condition (`generated_tokens == effective_max_new_tokens` alone).
- Parametrization over 3 terminal IDs × 4 factories exercises the shared EOS resolver path across all route builders, consistent with the claimed root-cause resolution (nonempty generation_config IDs win) — verify: `_build_truncating_generate_fn`, `make_constrained_generate_fn`, `make_generate_fn`, `make_polish_generate_fn` all route through the same `prepared["eos_token_ids"]` construction (not shown here, assumed from R1-R3 context).

[ASSUMPTION] The shared EOS-resolver precedence logic (generation_config nonempty IDs first, explicit None only disabling when no supported terminator exists) is asserted by the writer but its implementation is not in this excerpt — verify at build that `prepare_native_prompt` / resolver code matches this description.

No further action needed on this delta; proceed to the one fresh canonical retry as planned.