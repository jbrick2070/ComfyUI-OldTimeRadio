# OTR ledger-cleanup triage plan

## Goal

Close the four live blockers before resuming GPU qualification while preserving
the working LTX path:

1. Every still-consuming beat must have an authoritative target and a
   materialized episode-local path before video dispatch.
2. Short-leg provider/context-capacity failures must fail before generation or
   use a bounded repair boundary; they must not clip prose or change targets.
3. `scifi_news_pro` malformed standalone line-shape failures need a bounded,
   route-owned repair ladder with truthful exhaustion.
4. Freeze safety cleanup must never leave an empty voiced row downstream.

## Non-goals

- No GPU/render qualification during this triage.
- No WAN model or LTX recipe changes except preserving their existing input
  contracts.
- No prose-quality gate, target inflation, Python-authored story text, or
  silent fallback.

## Grounded implementation order

1. Establish one shared still-consumer manifest from the effective video policy
   and validate it in `nodes/_otr_video_engines/render_driver.py`, before
   `run_episode` dispatches its first engine request. Reconcile existing ledger
   rows by beat/object identity, require a real file under the active episode
   root, and stamp the validated target/path receipt. Sentinel init-image values
   such as `__cast_time_image__` remain deferred handles and are exempt from
   filesystem validation; ordinary still targets are not. Reject missing or
   stale rows before video; mesh engines require mesh fodder and may not borrow
   a scene still. Preserve the LTX scene-still join and cache-hit materialization.
2. Audit the shared provider-capacity preflight at the actual structured-call
   transport boundary in `nodes/_otr_structured_call.py` and the provider
   backends. The same prompt-token count, context cap, provider output cap, and
   full-output reservation must drive preflight and invocation. Must-fit
   overflow fails before network/model generation with a typed receipt.
3. Reproduce the `scifi_news_pro` `BAD_LINE_SHAPE` fixture and trace ownership
   across markup validation, repair prompt, merge, and revalidation. The repair
   remains a whole-artifact response, but the prompt is focused on the named
   defective source lines and the final response is accepted only after full
   revalidation; there is no unbounded line-splice protocol. Fail closed without
   mutating unrelated rows when the bounded route-owned model ladder is
   exhausted.
4. Trace the freeze empty-row fixture through
   `_otr_content_safety.py::apply_safety_cleanup`: its atomic replacement
   mutation can currently accept raw parenthetical/bracketed text whose cleaned
   spoken surface is empty. Make nonempty spoken text a row-local invariant after
   that mutation and every other permitted mutation; freeze remains fail-closed
   for an unresolved voiced row and must not accept a partially cleaned ledger.

## Verification gate

- Focused tests for each boundary and cross-route siblings.
- Canonical workflow JSON round-trip, link/widget audit, and validator (sanity
  gate only; this triage is expected to require no workflow topology/widget
  changes).
- Full Windows pytest suite with the project venv.
- Bug Bible regression from its own repository root.
- Only after all offline gates are green: selective live requalification of the
  affected legs, with `RESULT SUCCESS`, `obs_publish OK`, and canonical assets.

## Review status

This plan is being hardened with local Kibitz. The remote RTX 4060 Qwen review
is advisory only; Codex must ground every claim against the real Windows
checkout. An Antigravity quota hold means the Kibitz campaign is incomplete.
