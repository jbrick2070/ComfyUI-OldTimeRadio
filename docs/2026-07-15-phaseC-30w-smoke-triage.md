# Phase C 30w Smoke Sweep -- Triage (interim)

**Run:** Phase C source-bank bake-off, 30w SMOKE tier (plumbing gate), 32 legs.
**Generated:** 2026-07-15 ~23:32 local. **Read-only triage -- no code/pack changes.**
**State:** INTERIM. `tmp\_phaseC_smokes_ALLDONE.txt` does NOT exist; `_phaseC_sweep_resume.ps1`
(PID 36036) is live and `phaseC_scifi_codex_v2_30_aion` is rendering now. 25 of 32 legs recorded
in `tmp\_phaseC_receipts.csv`. Re-run this triage at ALLDONE for the full 32.

GREEN = result=SUCCESS AND obs=OK AND asset_bytes>0. Word count is NOT a gate.

## 1. Score (recorded legs only)

- Recorded: 25 / 32
- GREEN: 23
- FAIL: 2
- Green rate (recorded): 23/25 = 92%
- Pending (7): scifi_codex_v2 aion (in flight), scifi_codex_v3 local+aion,
  scifi_sonnet_v2 local+aion, scifi_sonnet_v3 local+aion

## 2. Matrix (lane x profile)

| Lane | local | aion |
|---|---|---|
| science_news_v2 | GREEN 97w / 52.6MB | GREEN 38w / 39.5MB |
| science_news_v3 | **FAIL** news-coda | GREEN 36w / 50.0MB |
| media_archive_v2 | GREEN 118w / 71.5MB | GREEN 124w / 78.7MB |
| media_archive_v3 | GREEN 114w / 61.3MB | GREEN 312w / 155.3MB |
| public_domain_story_v2 | GREEN 77w / 58.2MB | GREEN 143w / 74.2MB |
| public_domain_story_v3 | GREEN 88w / 71.3MB | GREEN 188w / 78.7MB |
| shakespeare_v2 | GREEN 68w / 53.7MB | GREEN 118w / 72.5MB |
| shakespeare_v3 | GREEN 54w / 51.1MB | GREEN 225w / 96.8MB |
| original_radio_v2 | GREEN 35w / 38.6MB | GREEN 96w / 54.9MB |
| original_radio_v3 | GREEN 44w / 41.1MB | GREEN 148w / 57.0MB |
| scifi_fable2_v2 | GREEN 97w / 55.6MB | GREEN 101w / 54.4MB |
| scifi_fable2_v3 | GREEN 89w / 58.0MB | GREEN 97w / 52.8MB |
| scifi_codex_v2 | **FAIL** P3 premise>144 | pending (in flight) |
| scifi_codex_v3 | pending | pending |
| scifi_sonnet_v2 | pending | pending |
| scifi_sonnet_v3 | pending | pending |

Every GREEN leg wrote its final asset to `output\otr\obs\` (obs_publish OK). Word spread
36w..312w is all lawful -- length is a recorded lane property, never a gate.

## 3. Failures grouped by fail_reason

**Root R1 -- news-coda bridge LLM failed both attempts** (1 leg: science_news_v3 local)
> [OTR_AnnouncerPass] news-coda bridge LLM failed both attempts. NO deterministic pool/arc-bridge
> fallback (no-fallback rip 2026-07-03) -- fix the model/prompt; a canned bridge must not ship as
> spoken content.

**Root R2 -- scifi_codex P3 structured validation, premise too long** (1 leg: scifi_codex_v2 local)
> P3 failed: [OTR_StructuredCall] 'scifi_codex:P3' failed after 2 attempt(s); last error ->
> ValidationError: 1 validation error for RadioScoreDraftV4 / premise / String should have at
> most 144 characters

Neither fail_reason repeats across a DIFFERENT bank in the recorded data -- each is confined to
its own lane family so far. But R2 is a known bug CLASS with real cross-lane leverage (see 4).

## 4. Root-cause pins + leverage rank

### R2 (rank #1 -- highest leverage). Class A: LLM-pass reliability / structured-repair gap.
- Pass: `nodes\_otr_scifi_codex.py:3261` `run_scifi_codex_episode` -> `_call_radio_score_draft`
  (2973) -> `invoke_codex_structured` (2913) -> `structured_call` (_otr_structured_call.py:1196).
- Contract: `RadioScoreV4.premise = Field(min_length=1, max_length=144)`
  (`_otr_scifi_codex.py:388`; P3 uses the draft variant RadioScoreDraftV4 with the same cap).
- What happened: model emitted a ~148-char premise. Attempt 2 was a "typed repair at
  temperature=0.100" that returned the SAME over-long premise -> ladder exhausted -> hard raise.
- ROOT: the typed-repair factory has no deterministic rule to shorten an over-long string field,
  and the char cap is not surfaced to the model. This is the codex56 unstated-contract class
  (schema_shape_instruction emits key NAMES only; every pydantic constraint is model-invisible
  unless hand-written into the seam AND the repair rules). See memory
  project_otr_codex56_unstated_contract_class.
- LEVERAGE: scifi_codex_v3 (both profiles) runs the SAME schema + SAME model + SAME unstated cap
  -> high recurrence risk on the pending legs. The generic seam fix -- teach the typed-repair
  ladder in `_otr_structured_call.py` to enforce string `max_length` (truncate at a word boundary
  / regen with the cap stated) -- hardens EVERY structured lane (codex, sonnet, outline). That is
  the real shared-pass root: one seam fix protects many lanes.
- Stochastic, not deterministic: depends on whether the model overshoots the premise on a given
  roll; scifi_codex_v2 aion may or may not trip it.

### R1 (rank #2 -- news lane only, low frequency). Class A: LLM-pass reliability.
- Pass: `nodes\OTR_LedgerScriptWriter.py:6338` -> `_otr_line_composer.py:3628` `compose_news_coda`.
- Gate: fires only under `if _style_grammar_on and nc_brief.strip():` -- i.e. the NEWS lane
  (science_news, which carries a `news_close_brief`). Non-news lanes never reach it.
- What happened: Mistral-Nemo failed the short bridge clause twice; per the 2026-07-03 no-fallback
  rip the node raises rather than shipping a canned bridge (correct by policy).
- LEVERAGE: news-lane-only (science_news v2/v3 x2 profiles = 4 legs); already 3/4 GREEN, so this
  is a low-frequency flake on a weak local model, not a deterministic block.

### Infra (Class C): NONE observed.
No OOM, no boot failure, no OpenRouter/network fault, no shared-validator crash. Plumbing is clean
on 23/25 -- every green leg published to obs with a non-zero asset. Both fails are weak-model
structured-output reliability, not infrastructure.

### Content-policy (Class B): NONE observed.
No deterministic weapons_smoking / profanity ship-stops in this sweep. (Reminder: those are LAWFUL
and must NOT be re-rolled -- none here.)

## 5. Prioritized fix list (no code this run)

1. **R2 seam fix (do first).** In `_otr_structured_call.py`, make the typed-repair ladder
   deterministically enforce string `max_length` on constrained fields (word-boundary truncate,
   or state the cap in the repair prompt) so an over-long `premise` is repaired instead of
   exhausting the ladder. Also surface the cap in the P3 seam prompt. Rescues scifi_codex v2/v3
   both profiles; hardens sonnet + outline. Verify whether scifi_sonnet's premise schema shares
   the 144 cap (its module `_otr_scifi_sonnet.py` is separate).
2. **R1 reliability (news lane).** Strengthen the news-coda bridge prompt/schema and/or add a
   lower-temp typed retry for `compose_news_coda`. NO canned fallback -- the 2026-07-03 rip holds.
   Lower urgency (news-only, 3/4 green).
3. **Re-run this triage at ALLDONE.** 7 legs pending; scifi_codex_v3 and scifi_sonnet are exactly
   where R2 is most likely to repeat -- confirm the count before ranking the full gate.

## 6. Process guardrails for whoever takes the fix

- TWO STRIKES: 2 solo fix attempts per problem; the 3rd must start with `/kibitz` (local, $0,
  file-grounded) before any more code.
- Any real code fix must re-green the venv suite + Bug Bible regression, then commit AND push to
  `v2.0-alpha` the same session. Verify HEAD==origin, no 0-byte/BOM, AST parse.
- If R2 becomes a logged PBUG, it needs a live production artifact to admit (a re-run leg that
  fails then passes) -- a static read of the schema is not admission on its own.
