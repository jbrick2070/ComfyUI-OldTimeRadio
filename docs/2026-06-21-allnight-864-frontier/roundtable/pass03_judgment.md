# Roundtable pass 03 -- judgment (wiring plan)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend $0.07. The panel accepted the "zero JSON edits" strategy but flagged a count error and (rightly) refused to accept the consumer-safety claim unverified.

## Accepted + resolved
- **Widget count off-by-one (all 4).** The map listed 23 rows but the prose said "24 entries / append index 24." CORRECTED: 23 entries (indices 0-22), next append index = 23. (Recounted JSON lines 62-84.)
- **Consumer-safety must be VERIFIED, not asserted (GPT, Gemini, DeepSeek).** Done via a grounded read of the real consumers:
  - All consumers use the tolerant shared parser `_otr_ledger_consumers.py` (`.get()`, no schema validation); `meta.arc_shape` + `cast[].speech_signature` are read by nobody (0 occurrences in the codebase) -> additive keys are safe.
  - No consumer assumes 18 lines / 3 acts (SceneSequencer/ShotLock/SignalLostVideo/MetaBrief all use `len()`/`enumerate`). So **F8 is NOT required to keep beat count fixed** -- we keep it fixed in v1 as the conservative choice only.
  - Announcer outro text + costly slot id are read by nobody downstream -> F2/F3/F7 invisible if `line_id`/`speaker_role` preserved.
- **The real guardrails are the freeze's CRITICAL invariants** (`_otr_ledger_freeze.py`): 7 top-level lists present/list-typed; unique non-empty `line_id`; `speaker_role` enum; voiced lines keep `char_id`; skipped lines have `text==""` + `tts_skip_reason`. Folded into WIRING_PLAN v2 as the constraints to not regress.
- **Mandatory v1 gate (DeepSeek, GPT):** end-to-end render with changed node-1 code + UNCHANGED workflow JSON; covered by the Sprint-0/exit headless smoke.
- **Split v1 vs future widget-append (GPT).** WIRING_PLAN v2 now has a hard "v1 forbids any JSON edit" rule and moves the append procedure to an appendix.
- **Append nuance (Gemini):** a position-bearing widget belongs at the END of the `required` dict in `INPUT_TYPES`, not `optional` -- noted in the appendix.

## Corrected vs the panel
- Gemini/DeepSeek feared F8 could break `OTR_SceneSequencer` if it hardcodes a 3-act shape. **Audited: it does not** -- SceneSequencer/ShotLock iterate dynamically. The risk is real in principle but absent in this code; documented as "keep counts fixed in v1 anyway (conservative)."
- Topology correction the panel could not see: node-1 does NOT feed SceneSequencer directly; it goes through node-62 `OTR_LedgerFreezeCascade` (a plain `json.dumps` passthrough that preserves additive keys), which is also where the CRITICAL invariants are enforced.

## Convergence
Wiring converged: zero JSON edits for v1, verified against real consumers, with the freeze invariants as the explicit guardrails. No open wiring risk remains. Proceed to the bug/risk pass on the consolidated plan.
