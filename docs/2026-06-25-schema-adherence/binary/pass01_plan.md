# BINARY-DECISION DECOMPOSITION -- CONVERGED ADDENDUM (1 round; folds into the schema-adherence program)

Panel GPT-5.5 / Gemini-3.1-pro / DeepSeek-v4-pro, Claude grounded judge, grounded
vs `_otr_line_hygiene.py` + `_otr_repair_prompts.py`. Spend ~$0.13. Converged in
ONE round: anchor + all 3 panel agree on a NARROW, span-level, abstain-gated,
shadow-first v1. The lever is SOUND; the addendum was over-scoped + line-level --
both fixed below.

## VERDICT
The binary-decision lever is the right COMPLEMENT to pass04 (tolerance accepts the
complex schemas you must keep; decomposition AVOIDS a schema for the one
classification that most threatens the ledger). But it is worth building ONLY if a
measured residual exists -- and only for dialogue/stage-direction.

## DECISIONS (panel + anchor converged)
1. **Scope to ONE application: dialogue vs stage-direction.** CUT #2 edit/no-op
   (chronologically needs an O(N) per-line loop + `payload_null_repair` already
   handles it), #3 speaker membership (the system already KNOWS membership
   deterministically; the LLM is only needed to REMAP a phantom -- Levenshtein
   `cast_membership_repair` -- which a binary cannot do), #4 normalize_length
   (segmentation planning, an O(N) split-search trap, not line hygiene). NO shared
   parameterized `binary_decide` primitive in v1 -- one dedicated classifier; only
   generalize if it proves value.
2. **Per-SPAN, not per-line (the core reframe).** The real failures are MIXED lines
   (dialogue + leading/trailing/embedded action). A whole-line A/B cannot preserve
   the spoken text while removing only the action. The binary decision operates on
   ACTION/DIALOGUE SPANS a DETERMINISTIC segmenter already isolated
   (`segment_double_quotes`, the `detect_stage_business_for_reroll` classes):
   "is THIS span ACTION or SPOKEN? A/B", then reassemble deterministically.
3. **Tri-state gate (the abstain state must be CREATED -- it does not exist).**
   Today the detectors return `False`/empty, conflating "clean", "not detected", and
   "too risky". The PURE `_otr_line_hygiene` layer must expose an explicit
   `HIT | CLEAN | ABSTAIN` (+ reason code) for a span. The binary call fires ONLY on
   `ABSTAIN`. `_otr_line_hygiene` STAYS PURE/stdlib -- it never makes the LLM call;
   it only emits the candidate span + ABSTAIN state.
4. **`binary_decide` lives in the LLM/orchestration layer, reusing the pass04 core.**
   A thin wrapper over a 1-field `Literal["A","B"]` schema run through
   `parse_validate_tolerant`; output contract = bare "A"/"B" (NOT yes/no -- avoids
   alignment refusals); optional `surrounding_lines` context (a bare span often
   needs its neighbours); STRICT parse = accept only when EXACTLY ONE allowed token
   appears decisively + no conflicting token elsewhere, else `None`. `None` / call
   failure -> the EXACT deterministic fallback = today's strip-chain behavior.
5. **Shadow-mode FIRST (the safe rollout).** Ship the lane in SHADOW MODE: log the
   binary verdict alongside the deterministic outcome on every ABSTAIN span, mutate
   NOTHING. Validate accuracy on an offline fixture suite across the LOCAL DEFAULT +
   >=1 remote model BEFORE enabling mutation. This de-risks the unproven premise
   ("LLMs are reliable at binary" is a HYPOTHESIS, not a guarantee -- it has a higher
   success RATE than structured emission, but still needs the tolerant parse +
   fallback).

## BUILD GATES (do these BEFORE writing the lane -- they may kill or shrink it)
- **G1 measure the residual (DS#7/#8).** `detect_stage_business_for_reroll` (Tier-2)
  already covers undelimited + embedded classes + triggers recompose;
  `split_stage_business` covers the balanced-quote class. INSTRUMENT the current
  corpus: what fraction of spans land in NEITHER (the true ABSTAIN set)? If it is
  ~0, the binary lane is unnecessary -- stop. If non-trivial, proceed. This is an
  on-paper/offline count, no GPU.
- **G2 byte-identity of abstain (GPT#3).** `split_stage_business` returns `norm`
  (curly->straight quote folded via `segment_double_quotes`), so "abstain returns
  the text unchanged" is NOT literally byte-identical. Confirm what the spine
  persists on abstain TODAY; the lane must preserve EXACTLY that (retain the
  ORIGINAL string if today's path does). Golden test: binary lane OFF == today, byte
  for byte.

## INTEGRATION (verify-at-build)
- Name the actual RECOMPOSE seam + the directive a "stage direction (B)" verdict
  passes (reuse `_BARE_STAGE_HINT` / the existing recompose reason codes
  `leading`/`trailing_after_quote`); define whether B -> strip-span, reroll, or
  record-action-on-compose_flags.
- Cost: cache key `(model, prompt_version, span_text)`; per-script binary-call
  budget; telemetry counters (abstain_rate, binary_fired, choice_A, choice_B,
  parse_none, fallback_used, shadow_agree_with_regex).
- Determinism: ONLY the default path (no binary) + the fallback + the parser must be
  deterministic; the remote LLM call need not be seed-stable -- cache for stability.

## RELATION TO pass04
Two complementary model-agnostic levers, same program: **pass04 = tolerance** (parse
what an arbitrary model emits for the schemas you must keep); **this = decomposition**
(replace the one most ledger-threatening classification with a binary span decision
any model answers reliably). `binary_decide` REUSES the pass04 core
(`parse_validate_tolerant` + a 1-field schema), so build pass04 (C0-C6) FIRST; this
addendum is a thin lane on top, gated on G1/G2.

## OUT OF SCOPE / CUT
Apps #2/#3/#4; a shared parameterized primitive; whole-line classification;
always-binary; any binary call on a HIT/CLEAN span; mutation before shadow-mode
validation passes.
