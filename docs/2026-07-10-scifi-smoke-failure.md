# Sci-Fi bake-off smoke failure: Codex P0 span repair

## Observed failure

The first canonical 30-word `scifi_codex` smoke reached the live runner and
failed at P0 after the technical model returned a fact whose `source_spans`
quote did not equal the supplied seven-key payload slice. The validator
correctly stopped before dialogue or media output. The same malformed artifact
survived the shared typed-repair attempt.

## Candidate root fix

Keep exact-span validation fail-loud, but make the originating technical
repair prompt explicit about the field/start/end slice and the exact payload
contract. Require Codex fact/entity/number IDs to use the v4 zero-padded forms
and include the observed slice mismatch in the post-validator error. No Python
text substitution or source fetching is allowed.

## Cross-lane audit

The same P0 source-span contract exists in Codex, Gemini, and Sonnet. The
repair hardening is therefore applied to all three new lanes: exact evidence
ID forms, field/offset mismatch diagnostics, and an explicit originating-slot
repair request. Existing archive, original-radio, and fable2 lanes are not
modified; their registry/routing/full-suite coverage remains green.

## Gate

Rerun the same canonical 30-word Codex smoke, then Gemini and Sonnet only after
Codex passes. Confirm published OBS assets and no canonical workflow diff.
