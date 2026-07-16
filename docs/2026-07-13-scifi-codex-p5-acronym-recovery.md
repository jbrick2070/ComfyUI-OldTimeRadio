# Sci-Fi Codex P5 accepted-acronym recovery

## Live problem

The canonical 42-word `scifi_codex` smoke using Aion 3.0 Mini creative and
Mistral-Nemo technical reached P5 after P0/P1/P2/P3/P3_rewrite/P4 passed. P5
failed twice on `l002: spoken text contains an all-caps lexical word`. The
accepted FactIndexV4 source contains grounded `MIT`/`QSL` tokens, which the
current validator allows, while the model-authored accepted upstream question
and score use the ordinary role acronym `CEO`. The exact P5 script line is not
retained in the pending ledger, so the fix must be grounded in accepted
upstream artifact boundaries rather than a guessed literal or a blanket common-
acronym allowlist.

The same 42-word combination has already exposed and received green fixes for:

- P1 compact bounded rewrite;
- P3 global beat topology and local text patch envelope;
- P3 base/rewrite conservative prose ceilings;
- P2 bounded acronym-aware cast names (`AI Unit Seven`).

The next required proof is the 120-word smoke, then a fresh final 42-word run,
with the GUI on port 8001 left untouched.

## Codex grounded anchor review

CONFIRMED: `_source_grounded_all_caps` currently derives only literal accepted
fact/entity/number source spans. `_validate_script_post` and final script
validation pass that allowlist into `validate_spoken_text_and_roster`.

CONFIRMED: P1 `DramaticQuestionV4`, P2 `CastPlanV4`, and P3 `RadioScoreV4` are
accepted upstream authored artifacts available at the P5 call boundary. A
short role acronym such as `CEO` may be valid in those contracts without being
present in the source span. A safe fix must not authorize arbitrary all-caps
prose from the failed P5 artifact itself, must keep rejecting longer shouting
tokens such as `STOP`, and must preserve the existing exact source-span
allowlist.

CONFIRMED: the live failure happens before ledger/media/OBS and is a production
admission, not a static test invention. Any new PBUG/Bible rule needs a focused
validator test plus a live 120/42 verification plan.

MUST-FIX questions for the panel: define the narrowest trusted upstream
acronym boundary; thread it through every P5/P7/P9/final validation call; cover
the corresponding repair prompt and receipts; identify adjacent validator or
contract risks likely to block the 120-word smoke; and specify the exact
regression/live gates without changing canonical workflow wiring or port 8001.

## Non-goals

Do not lowercase or Python-rewrite dialogue, allow every common acronym, widen
the schema caps, edit `workflows/otr_canonical.json`, or kill the GUI server.
