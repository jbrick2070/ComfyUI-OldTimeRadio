# ANCHOR REVIEW (Cowork Claude) -- chunk 3 r4 (convergence / residual defects)

Doc: CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md v4. This round I verified the r3 citations
directly: _otr_ledger_freeze.py :232-250 (key_terms must be list -- CONFIRMED) and
_otr_story_spine.py :212 (news_close_brief str read from the dump -- CONFIRMED).

VERDICT: SHIP. v4 is build-ready. No new MUST-FIX found on a fresh pass.

Residual notes (non-blocking, carry to build):
1. The science lane's byte-identity claims are covered by 8 test families; the ONLY
   uncovered surface I can find is log-line text (RSS_FETCH OK / news_interpreter OK
   messages) -- logs are not part of the byte-identity contract; leave them but do not
   reword them gratuitously.
2. `validate_interpreter_result` returning the validated dump means run() dumps ONCE
   (inside the validator via result.model_dump()) -- confirm the validator, not the
   writer, calls model_dump() so there is exactly one dump call (pydantic dumps are
   deterministic but cheap discipline).
3. The sweep arm-2 test (executable=true + rsc=false LOADS) needs its synthetic pack's
   prompt_stages to be pipeline-declared seams (strict loader posture) -- reuse the
   simple_4-shaped fixture pattern, not the production seam fixture.
4. Convergence check: r1-r3 accepted items are all reflected in v4 (halt re-raise cause,
   dump-value validation, helper_context retention, pipeline flag + fixture updates,
   sweep two-arm rule, import postures, S31 B6 survival, _OTRNI cleanup). No conflicts
   between rounds.
