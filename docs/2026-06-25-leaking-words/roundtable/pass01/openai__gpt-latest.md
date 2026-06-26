<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document correctly identifies root classes, but the proposed architecture choices contradict the invariants and still lack a coherent final contract for “spoken text must contain only speakable fictional dialogue.”

MUST-FIX BEFORE BUILD:
1. [Invariants + Option A + Option D] Defect: the plan’s two strongest remedies violate the stated invariants. A final capable LLM cleaner is not deterministic, not offline-capable, and not model/transport-agnostic; making the frontier writer the recommended default is also not offline/model-agnostic. Concrete fix: split the architecture into (a) mandatory offline deterministic final verifier/fail-closed gate, and (b) optional online/frontier repair mode when the operator explicitly enables non-offline generation. Do not describe A or D as the “minimal durable model-agnostic answer.”

2. [Option A] Defect: “Return ONLY the spoken words; remove any stage direction, any real-world proper noun that doesn’t belong…” is an underspecified rewrite pass, not a bounded cleaner. It can silently delete legitimate fictional-world nouns, alter meaning, empty lines, or degrade already-clean frontier output. Concrete fix: if retained, make it a typed repair with an explicit contract: input line + speaker + cast + allowed fiction nouns + banned news nouns; output JSON containing `clean_text`, `removed_spans`, `reason_codes`, and `confidence`; reject if empty, over-diffed, quote-malformed, or if it changes non-target words. It must be skipped for clean frontier lanes unless a deterministic verifier flags a defect.

3. [News-bleed + Option A + Option E(3)] Defect: the plan has no actual semantic architecture for news-bleed. “Remove any real-world proper noun that doesn’t belong” and “news-proper-noun guard” require a definition of what belongs in the fiction. The grounded scrub only shows cast/phantom-name machinery, not a news/fact policy; `_otr_ledger_scrub.py` explicitly allows real cast-member names and uses roster/phantom detection, which is not the same as detecting “President Trump” leaking from the news brief. Concrete fix: add a news-abstraction subsystem before line composition: transform raw news into fictional conflict objects, produce `allowed_proper_nouns` and `banned_source_proper_nouns`, and validate body lines against that policy. [ASSUMPTION] Verify whether `_otr_line_composer.build_allowed_roster` includes news/key terms; if it does, that may actively defeat a news-bleed detector.

4. [Stage-direction leak + Option E(1)] Defect: the proposed “widen the leak detector beyond a verb whitelist” is still whack-a-mole if made as another destructive heuristic. The shown code proves the current architecture misses the anchor leak: `_NARRATION_VERBS` does not contain “gasping”, `is_third_person_action_clause()` requires the lead verb to be in `_NARRATION_VERBS`, and `_leading_stage_strip()` only fires when the post-quote body starts lowercase, so `Gasping, "We're running out of time..."` is outside the shown deterministic floor. Concrete fix: define a final spoken-text contract around structural extraction, not verb membership: e.g. for lines of the shape `^[A-Z][a-z]+ing,\s*["“](.+)` extract the quoted dialogue or fail to repair; add explicit regression fixtures for `Gasping,`, `Wheezing,`, `Trembling,` and unbalanced quote variants.

5. [Caps-name vocative + grounding `_otr_line_hygiene.py` / `_otr_ledger_scrub.py`] Defect: the plan treats caps-name vocative as “cheap” but does not specify where or how it avoids deleting legitimate emphatic address. Grounding shows `scrub_self_vocative()` only removes the speaker’s own name at leading/trailing vocative positions, and `scrub_ledger()` does not call it in the shown code. `_otr_ledger_scrub.py` also treats naming a real cast member as legitimate drama in its self-test. Concrete fix: add a dedicated final spoken-line scrub for exact ALL-CAPS full cast-name vocatives only, with speaker/cast context: remove `^FULL NAME[,!: -]+` and `[, ]+FULL NAME[.!?]?$` only when the token exactly matches a roster full name in all caps; do not generalize to mixed-case names or arbitrary in-line references. Verify composer-side wiring before assuming existing self-vocative scrub applies.

6. [Malformed quotes + Option B] Defect: malformed quotes are listed as a leak class, but the strategy does not give them an architecture except implicitly through constrained generation. Grounding shows `sanitize_transcript_text()` only balances a single dangling wrapper quote at an edge; `strip_quote_anchored_stage_direction()` aborts on odd quotes. That leaves a class where malformed quotes can still ship unless another unseen gate catches them. Concrete fix: add a mandatory final quote policy: balanced quotes pass; one edge wrapper quote may be stripped; any internal odd quote or ambiguous quote structure fails closed to recomposition/repair.

7. [“Frontier writers shipped ZERO today” + Option D] Defect: the plan elevates one day’s observation into an architectural premise. “GPT shipped zero leaks today” is not a durable correctness guarantee, and using it to justify defaulting to frontier avoids solving the local-lane invariant. Concrete fix: treat frontier as a product-tier recommendation, not the correctness layer. The same final verifier must run on frontier output, but may be expected to no-op.

8. [What has already been tried + Strategic question] Defect: the document asks for a “minimal durable model-agnostic answer” but presents five partially overlapping interventions without an ordering, fail-closed behavior, or acceptance criteria. Concrete fix: specify the actual build sequence:
   1. upstream prompt/output contract to reduce defects;
   2. constrained generation where transport supports it;
   3. deterministic final verifier for the four leak classes;
   4. repair/recompose on verifier failure;
   5. optional LLM cleaner only in non-offline mode and only after verifier hit;
   6. hard fail if repair budget is exhausted.

SHOULD-FIX:
1. [Option B] Defect: constrained generation can prevent some format leaks but cannot solve news-bleed semantics or caps-name policy. Concrete fix: scope B explicitly to stage-direction/malformed-format prevention only; do not count it as the semantic solution.

2. [Option B] Defect: the plan assumes local transports can enforce grammar uniformly: “Ollama/in-process/llama.cpp GBNF?” is posed as a question, not a capability. Concrete fix: define transport capability tiers: grammar-enforced, JSON-contract-only, prompt-only fallback. [ASSUMPTION] Verify actual composer transports and whether grammar constraints can be applied without workflow-JSON changes.

3. [Option C] Defect: stronger prompting is framed as a possible strategic fix despite the root cause saying weak local models over-literalize instructions. Concrete fix: keep C as defect-rate reduction only, never as the enforcement layer.

4. [Option E(3)] Defect: “news-proper-noun guard on body lines” risks false positives unless it distinguishes cast, setting, franchise/world nouns, historical references, and raw source-news entities. Concrete fix: require a per-episode named-entity policy generated from premise/cast/news abstraction; otherwise the guard should be fail/report-only, not destructive.

5. [Invariants] Defect: “audio spine byte-identical” is ambiguous against any cleaner/scrub that changes spoken text before TTS. Grounding in `_otr_ledger_scrub.py` treats transcript sanitizer as “AUDIO-AFFECTING -> default OFF => byte-identical.” Concrete fix: define whether “byte-identical” means no workflow/schema change or literally no audio-affecting text mutation. If literal, the whole leak-cleaning strategy cannot be active by default.

6. [Evaluation missing] Defect: no acceptance gate is defined. Concrete fix: create a fixed regression corpus from the observed shipped leaks and require zero shipped instances for the four named classes across local and frontier lanes; include negative fixtures for legitimate vocatives, legitimate real-world nouns in premise, and non-stage `-ing` dialogue.

7. [Repair budget / freeze cascade] Defect: a final LLM cleaner or deterministic fail gate can discover defects after the existing one-repair budget is exhausted. Concrete fix: define exactly where this gate sits relative to freeze/TTS/hash and how it consumes or refuses the shared repair.

OPTIONAL / NICE-TO-HAVE:
- Add telemetry reason codes for the four leak classes: `stage_direction_leak`, `news_bleed`, `caps_cast_vocative`, `malformed_quote`.
- Add lane-level metrics: local/frontier defect rate, repair-consumed rate, cleaner-diff rate.
- Add an operator switch: `strict_local_clean=true` means fail closed instead of shipping best-effort local output.

CUT THESE (scope / over-engineering):
1. [Option A] Cut “single FINAL LLM-cleaner pass over every frozen spoken line” as mandatory architecture. It is not safe under deterministic/offline/model-agnostic invariants and risks degrading the already-clean frontier lane. Keep only as optional repair mode after a deterministic verifier hit.

2. [Option E(1)] Cut broad `-ing` / third-person lead destructive scrubbing as the main answer. It repeats the current pattern with a larger false-positive surface. Use narrow structural extract/fail rules for known leak shapes instead.

3. [Option D] Cut “frontier writer as the recommended default” from the architecture decision. It is product policy, not a leak-stopping architecture, and it leaves the local lane unsolved.

4. [Existing split/action preservation expansion] Cut any further investment in action extraction/recording for this problem. `split_stage_business()` and `action_split:` telemetry preserve action metadata, but the operator’s stated goal is preventing spoken/caption leaks; preserving leaked action is secondary and can calcify complexity before the final spoken contract is fixed.