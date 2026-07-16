# Sci-Fi Codex Repair-Envelope Live Failure

## Live receipt

- Canonical workflow: `workflows/otr_canonical.json`
- Source bank: `scifi_codex`
- Target: 120 words
- Creative model: `google/gemma-4-E4B-it [LOCAL HF]`
- Technical model: `mistralai/Mistral-Nemo-Instruct-2407`
- Prompt ID: `cc9e0f8a-2a20-40a1-b5dc-da2fc8a400d6`
- Result: FAIL at P3 after two structured attempts; no published asset.
- Evidence: `docs/_bakeoff_scifi_codex.log` and the ComfyUI server log.

## Root-cause hypothesis

The typed P3 repair returned the complete repaired `RadioScoreV4` under the
single transport key `resolved_artifact`. The lane passed that outer envelope
directly to strict `RadioScoreV4` validation. Pydantic therefore reported all
required score fields missing and rejected `resolved_artifact` as extra.

## Candidate fix

At the Sci-Fi Codex shared model-response boundary for all of its typed passes,
parse the first JSON object and
unwrap only the exact shape `{"resolved_artifact": <object>}` before the shared
strict structured validator sees it. Mixed roots, non-object values, and any
other sibling key remain fail-loud. The inner object must still satisfy the full
requested Pydantic schema and post-validator. Log the normalization and retain
the raw call hash/length receipt. Add a boolean normalization receipt.

## Required proof

1. Regression: the exact single-key envelope validates successfully.
2. Regression: the same envelope with any sibling key remains rejected.
3. Existing canonical direct-root responses remain unchanged.
4. Seam audit confirms no Sci-Fi Codex prompt text contains
   `resolved_artifact`; the envelope was spontaneous model transport behavior.
5. Focused Sci-Fi Codex and structured-call tests pass.
6. Full Windows suite, Bug Bible, canonical workflow validator, JSON round-trip,
   link/widget audit pass.
7. Commit and push the green root fix to `v2.0-alpha`.
8. Selective reset and rerun the same 120-word `scifi_codex` canonical leg.
9. Require `RESULT SUCCESS`, ledger existence, and OBS final existence.
10. Inspect authored music-bookend routing in the live ledger/final artifact;
   record a separate bug only if the live run demonstrates a misroute.

## Invariants

- No fallback to another source bank or pipeline.
- No permissive arbitrary root-key unwrapping.
- No visual selector may enter the story ledger.
- Preserve unrelated dirty files and active Fable2 C4b ownership.
- Do not launch Gemini until Codex is green.
