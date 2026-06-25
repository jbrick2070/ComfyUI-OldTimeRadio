<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The addendum’s core story is under-specified and partly mis-aimed: a whole-line binary classifier cannot deterministically clean mixed dialogue/stage-business lines, and most proposed applications do not serve the stated ledger-integrity goal.

MUST-FIX BEFORE BUILD:
1. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS 1 / THE PROPOSAL] Whole-line “spoken line or stage direction?” does not solve the actual mixed-line failure class. Grounding shows the hard cases include mixed text: `detect_stage_business_for_reroll` handles leading, trailing-after-quote, embedded-between-quotes, and `undelimited_action_clause`; `split_stage_business` only extracts balanced-quote outside spans. A binary answer for the entire line cannot preserve spoken text while removing only action prose. Concrete fix: restrict v0 binary use to deterministic candidate spans/chunks already isolated by `segment_double_quotes`, `_CLAUSE_SPLIT_RE`, or another pure segmenter. Ask “is this span ACTION or SPOKEN?” and reassemble only when the spoken/action boundary is already deterministic. If no deterministic segmentation exists, fall back to existing reroll/strip path, not whole-line binary classification.

2. [THE PROPOSAL / CONSTRAINTS] The proposed parse rule “first decisive token” violates “fail to deterministic fallback, never silent-wrong.” A model can echo the prompt or answer “A if..., B if...”; first-token parsing can accept a conflicted answer instead of returning `None`. Concrete fix: define a safe output contract before build: e.g. prompt for exactly one line `A` or `B`; parser accepts only trimmed whole-response or first nonempty line equal to one allowed token, with no conflicting allowed token elsewhere. Anything else returns `None`.

3. [GROUNDED CURRENT STATE / CONSTRAINTS] The byte-identity claim is not safe as written. The document says `split_stage_business` “not confident -> returns `(text, "", "")`,” but grounding shows `split_stage_business` returns `(norm, "", "")` after `segment_double_quotes`, which folds curly double quotes to straight quotes. If callers persist the returned dialogue on abstain, that is not byte-identical. Concrete fix: correct the addendum and require an explicit golden test proving the default local path is byte-identical with binary disabled and on all binary-abstain cases. If abstain must preserve bytes, change/route around `split_stage_business` so the original string, not `norm`, is retained when no action is extracted.

4. [THE PROPOSAL / CONSTRAINTS] The integration point contradicts the grounding module boundary. `_otr_line_hygiene.py` is documented as PURE/stdlib/no I/O and currently contains deterministic hygiene helpers. `binary_decide` is an LLM call and cannot live inside or be called directly from that module without breaking that architectural contract. Concrete fix: keep `_otr_line_hygiene.py` pure and have it expose only deterministic candidate detection/segmentation. Place `binary_decide` in the existing structured-call/LLM layer or a new orchestration module, invoked by the spine/composer after pure hygiene returns an explicit `ABSTAIN_WITH_CANDIDATE` state.

5. [OPEN QUESTIONS / THE PROPOSAL] The addendum leaves the core contract undecided: shared primitive vs per-pass, A/B vs yes/no, sibling vs wrapper over `structured_call`, always-binary vs escalation-only, and replace vs augment. These are not implementation details; they determine call graph, determinism, fallback, and testability. Concrete fix: collapse v0 to one answered design: one narrow binary-span classifier for dialogue/action only; exact A/B contract; invoked only from a named escalation seam; no replacement of complex passes in this build.

6. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS 2-4] The candidate list bloats the addendum beyond its stated purpose. Payload-null repair, speaker membership, and normalize_length are different failure domains from “every ledger line should be clean, correctly-attributed spoken text.” Concrete fix: prune v0 to dialogue/stage-business only. Move other decomposition ideas to separate RFCs after the primitive has proven value.

SHOULD-FIX:
1. [THE OPERATOR’S PRINCIPLE] The claim that all LLM families are “RELIABLE at binary classification” is an unsupported architectural premise. [ASSUMPTION] It may be directionally true, but the plan treats it as guaranteed. Concrete fix: state it as a hypothesis and require an offline fixture suite across the intended local/default model and at least one remote model before enabling mutation.

2. [CONSTRAINTS / OPEN QUESTIONS 3] Cost and latency are hand-waved. A binary call “per ambiguous line” can multiply calls across a ledger. [ASSUMPTION] Ambiguous line counts may be nontrivial in noisy generations. Concrete fix: add a budget: max binary calls per script/beat, cache key `(model, prompt version, seed, text)`, timeout behavior, and telemetry for call count and fallback count.

3. [THE PROPOSAL] “Regex abstains” is not a defined state in the current grounded helpers. Many functions return only `False`/empty string, which conflates “clean,” “not detected,” and “too risky to classify.” Concrete fix: introduce an explicit tri-state at the orchestration seam: `HIT`, `CLEAN`, `ABSTAIN`, with reason codes. Binary escalation should fire only on `ABSTAIN`, not every `False`.

4. [CONSTRAINTS] “Deterministic given (model, prompt, seed)” assumes backend seed support and stable decoding. [ASSUMPTION] Remote/frontier transports may not provide byte-stable seeded output. Concrete fix: downgrade the guarantee to “deterministic fallback and deterministic parsing; LLM output cached when transport is nondeterministic,” or require a transport capability flag before binary mutation is enabled.

5. [CANDIDATE APPLICATIONS 1] The addendum says “route to the existing RECOMPOSE seam” but does not identify the seam or its expected inputs. Grounding only shows detectors and scrubbers, not the caller. Concrete fix: name the actual caller/seam in the main pipeline, define the input payload, and specify whether binary “action” causes reroll, strip, action recording, or CI failure.

6. [CONSTRAINTS / offline-verifiable] “Offline-verifiable” is asserted but no acceptance criteria are given. Concrete fix: add required fixtures: balanced quote outside-span, odd quote, no quote undelimited action clause, pure parenthetical stage direction, legitimate dialogue starting with action-like words, and mixed spoken/action line. Each fixture should assert output text, action capture if any, reason code, and fallback behavior.

OPTIONAL / NICE-TO-HAVE:
- Add telemetry counters: binary_called, binary_accept_A, binary_accept_B, binary_parse_none, deterministic_fallback_used, post-binary_reroll_used.
- Version the binary prompt separately from the parser so cache invalidation is explicit.
- Include a shadow-mode flag that logs binary decisions without mutating ledger output.

CUT THESE (scope / over-engineering):
1. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS 2] Cut “Edit vs no-op / payload_null” from this build. Grounding already has `payload_null_repair`, and a binary pre-gate still does not produce the required replacement string when an edit is real. Safe to cut because it is not needed for clean spoken ledger lines.

2. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS 3] Cut “Speaker membership.” Exact locked-cast membership is deterministic set membership, and grounding already describes deterministic Levenshtein remap plus `cast_membership_repair` fallback. A binary yes/no adds calls but cannot choose the corrected speaker.

3. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS 4] Cut “Beat over-length / normalize_length” from this addendum. It is a segmentation-planning problem, not ledger line hygiene, and per-boundary binary calls introduce a new search/reassembly subsystem. Safe to defer to a dedicated normalize_length redesign.

4. [THE PROPOSAL] Cut the generalized shared primitive across all passes for v0. Build a narrow dialogue/action binary-span helper first. Generalize only after one call site proves that binary decomposition improves accuracy without breaking determinism, cost, or byte-identity.