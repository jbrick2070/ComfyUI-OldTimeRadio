# codex_v4 short-leg P2/P5 stochastic failures -- root fix (2026-07-18)

**Window:** CODER. **Baseline:** HEAD `a0ac8948` == `origin/v2.0-alpha`
(ancestors `ed7b37de` short-episode COUNT gates -> advisory, `d6b0706e` handoff).
**Governing contract:** `docs/SOURCE_BANK_PREFLIGHT.md` Gate 3.

## Problem (code-grounded)

After `ed7b37de` made the short-episode structural COUNT gates advisory, two
codex-writer validators still failed 30w/120w Mistral-Nemo legs STOCHASTICALLY.
Neither is a count gate; both violated the SAME Gate-3 "Python normalizes
mechanical formatting / does not alter spoken prose" split.

1. **P2/P5 cast-name Title-Case.** `_is_canonical_character_name`
   (`nodes/_otr_scifi_codex.py`) rejects a fixable name whenever Mistral-Nemo
   emits a quoted nickname (`Maxwell 'Max' Hart` -- the `'Max'` token fails
   `[A-Z][a-z]+`) or an honorific prefix beyond `Dr.`/`Prof.` (`Col. Marcus
   Grant` -- `Col.` is neither a Title-Case word nor a 2-3 letter acronym). The
   typed LLM repair rebuilds the name stochastically and misses often; a fixable
   name should never fail the episode.
2. **P5 spoken "self-vocative."** `_spoken_error` rejects a line that opens with
   its own speaker's first name + `,!:` (`l00x: spoken text begins with a
   self-vocative`). The ScriptArtifactV4 typed-repair rule told the model to
   "Preserve ... every ... line text" -- the exact opposite of what the fix
   needs -- so the model could not converge; the retries pile KV-cache VRAM and,
   at 30w, OOM.

## Decision + fix

### Fix 1 -- cast name = routing metadata -> deterministic Python (Gate 3 L145-146)
A cast name is validated routing metadata, not spoken prose, so Python may
normalize its mechanical shape. `repair_cast_plan_metadata` now canonicalizes a
non-announcer name via `_normalize_character_name` / `_canonicalize_cast_token`:
strip a quoted-nickname wrapper, strip an honorific abbreviation's trailing
period, recover Title-Case for a plain alphabetic token, keep a short acronym
verbatim, and DROP only an unparseable *quoted* aside. A token that carries
meaning Python must not guess (a digit, e.g. `Unit 7`) returns `None` -> the
whole name defers to the existing bounded model repair (which spells the number
out). The P2 seam already prefers a deterministic repair that re-passes
`_validate_cast_plan` before falling to the LLM, so no new wiring is needed.

### Fix 2 -- self-vocative = spoken prose -> bounded LLM reword (Gate 3 L147-153)
Gate 3 L147 forbids Python altering spoken prose, so a blind strip is unlawful;
the self-vocative regex also false-positives (a speaker named "Grant" saying
"Grant, ..."). It is a prose defect the MODEL must fix. The gate stays fatal
within an attempt; only the REPAIR guidance changes. `_script_artifact_repair_rules(detail)`
returns a targeted rule when `detail` names a self-vocative: reword ONLY the
rejected line(s) to the same beat/intent/speaker without the self-address, scan
every other line for the same defect, and preserve all other line text byte for
byte. This is Gate 3 L149-153: invalid creative content returns to the model
through a bounded repair that names item + evidence + defect + allowed
correction; exhaustion still fails closed (the ledger is assembled only from an
accepted artifact, so it stays intact). This rides the existing
`structured_call` bounded-repair ladder -- the multipass scaffold that already
decides whether another pass is needed.

## Gates kept FATAL (unchanged)
`shot_index` / `cast_id` / `fact_id` / `cue_id` / `unused_shot` / graph
closure / G9 SFW and every non-self-vocative `_spoken_error` check
(empty, stage-direction/markup/label, all-caps lexical, non-lexical token).

## Wiring verified
`structured_call` passes the `PostValidationError` (message = the
`CodexSpokenTextError` string, e.g. `l003: spoken text begins with a
self-vocative`) into `typed_repair_factory(error=...)` -> `detail` -> the
self-vocative branch fires. A factory-returned `CastPlanV4` is re-checked by the
post-validator before acceptance, so a normalized name must pass
`_validate_cast_plan`.

## Proof
Focused + full `test_scifi_codex_lane.py` green; full suite + Bug Bible; LIVE
codex_v4 30w AND 120w canonical legs = RESULT SUCCESS + obs_publish + asset.
