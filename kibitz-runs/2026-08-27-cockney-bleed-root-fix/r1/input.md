# Lemmy Cockney-Bleed Root Fix — Code-Ready Plan

Status: PRE-BUILD, pending Kibitz R1-R4 convergence
Branch: `v2.0-alpha`
Grounding commit: `6ae235a24ef6d9261db8a6d99a384dd65e21ac46`
Canonical workflow: `workflows/otr_canonical.json`

## P0. Outcome and acceptance contract

Fix the prompt-scoping defect that causes non-Lemmy characters to inherit
Cockney idiom whenever Lemmy is active in the inline dialogue pipelines.

The shipped behavior is correct only when all of these are true:

1. Lemmy's existing cameo decision, cast identity, written register, and voice
   route are unchanged.
2. A per-line request whose `LineRequest.speaker` is not `LEMMY` receives no
   Cockney policy text, even when `LineRequest.allowed_people` contains Lemmy.
3. A per-line request whose active speaker is Lemmy receives an explicit
   Lemmy-only Cockney rule plus the existing standard-English orthography rule.
4. An exchange group with no Lemmy slot receives no Cockney policy, even if the
   episode cast contains Lemmy.
5. A mixed exchange containing Lemmy and another speaker receives a rule whose
   grammatical subject is Lemmy and which explicitly requires every other
   character to retain their own register.
6. The same scoped system message survives line retries and exchange repair.
7. No-Lemmy prompt assembly remains byte-identical.
8. The separate `scifi_news_pro` pipeline remains unchanged: it already pins
   Lemmy's `CastShape.register` and renders each cast row's own register.
9. No node schema, widget, link, or workflow value changes.

## P1. Confirmed production reachability and defect

### P1.1 Shared helper

`nodes/_otr_dialogue_policy.py:6-36` defines
`_COCKNEY_ORTHOGRAPHY_RULE` and appends it when any item in a supplied roster
looks like Lemmy. The text begins with the subjectless command "Convey the
Cockney accent...". The helper has exactly two production callers, confirmed
with `rg`: `_otr_line_composer.py` and `_otr_compose_exchange.py`.

### P1.2 Per-line path

`nodes/_otr_line_composer.py:1040-1055` resolves the bank-specific system
prompt, constructs `list(req.allowed_people or ()) + [req.speaker]`, and passes
that whole roster to the policy helper. `allowed_people` is the full cast built
by`nodes/OTR_LedgerScriptWriter.py:4567-4572` and attached to every
`LineRequest` at lines 4828-4848. Therefore a non-Lemmy line receives the
system-level Cockney command whenever Lemmy is elsewhere in the cast.

### P1.3 Grouped-exchange path

`nodes/_otr_compose_exchange.py:386-393` appends the same rule to the system
message for one multi-speaker response. Production normalizes cast dictionaries
to `_CastShim` objects at lines 898-920; the current detector does not inspect
those objects. Consequently the live grouped trigger is the current
`VoicedSlot.speaker` set, not merely full-cast presence. A group containing
Lemmy still places the unscoped Cockney command over every speaker in that
group. Accepted output is appended to the rolling prior context at lines
1004-1026, so contaminated non-Lemmy phrasing can echo into later groups.

The canonical writer ships `use_exchange=True` at positional widget slot 13
(`tests/test_workflow_json_guardrails.py:585-663`). Failed groups, singletons,
and other uncovered character beats fall back to the per-line path in
`nodes/OTR_LedgerScriptWriter.py:4881-4969`. Both callers must be fixed.

### P1.4 Bank boundary

- `media_archive` and `original` reach the affected inline writer/composers.
- `scifi_news_pro` dispatches to its own runner before the inline composer at
`nodes/OTR_LedgerScriptWriter.py:3376-3431`. It pins only Lemmy's register in
`nodes/_otr_scifi_news_pro.py:1049-1074`, while `_script_user_prompt` renders
  each character's own named register at lines 2219-2229.
- Source-faithful `public_domain` and `shakespeare` adaptations exclude the
  recurring cameo in`nodes/_otr_casting.py:1165-1189`.

## P2. Minimal implementation

### P2.1 Replace roster semantics with active-output semantics

Edi`nodes/_otr_dialogue_policy.py`:

1. Replace `roster_has_lemmy` with a private active-speaker predicate accepting
   `Iterable[str]`. Delete the dictionary/`char_id` roster behavior; no
   production caller needs it after this patch.
2. Make `append_dialogue_policy` keyword-only for the new category:

   ```python
   def append_dialogue_policy(
       system_prompt: str, *, active_speakers: Iterable[str]
   ) -> str:
   ```

3. Materialize/inspect only speaker-name strings. Fail loudly with `TypeError`
   if an item is not a string. This makes a future cast-row/object call a test
   failure instead of silently reintroducing roster scope.
4. Match Lemmy case-insensitively after surrounding-whitespace removal.
5. Preserve `system_prompt or ""` and return it unchanged when no active
   speaker is Lemmy.

Use this exact policy meaning (line wrapping may follow formatter output):

```text
For LEMMY's spoken lines only, convey his Cockney accent through phrasing,
idiom, cadence, and rhythm. Every other character must retain that character's
own speech register; do not give any other character LEMMY's Cockney phrasing,
idiom, cadence, or rhythm. Use standard English spelling in every spoken line;
do not encode pronunciation with phonetic misspellings.
```

The negative clause is required for mixed exchanges. Absence, rather than a
negative mention, is required for a non-Lemmy per-line call.

### P2.2 Correct the per-line caller

In`nodes/_otr_line_composer.py:1049-1051`, delete the roster construction and
call:

```python
system = append_dialogue_policy(
    system, active_speakers=(req.speaker,)
)
```

Do not change `LineRequest.allowed_people`; it still feeds named-entity and
transport-cleanup context. It must simply stop controlling dialogue style.

### P2.3 Correct the exchange caller

In`nodes/_otr_compose_exchange.py:391-393`, remove `cast` from the policy
decision and call:

```python
system = append_dialogue_policy(
    system,
    active_speakers=tuple(slot.speaker for slot in beat_group),
)
```

`VoicedSlot.speaker` is the authoritative output speaker named by
`compose_exchange` (`compose_exchange` documents that contract at lines
585-618). Do not use `_CastShim`, cast rows, slot contracts, voice cards, or
`allowed_people` as active-speaker substitutes.

### P2.4 Explicit non-goals

- Do not change the 11% roll, `lemmy_cameo` choices, Lemmy's cast profile, or
  TTS routing.
- Do not alter `_normalize_cast` or widen its persona fields in this fix. Its
  loss of some production voice-card detail is a separate issue affecting all
  exchanges and would confound this repair.
- Do not add a post-generation dialect scrubber, vocabulary blacklist, or line
  rewrite. Those are shims and can damage legitimate character voices.
- Do not change prompt-router packs or source-bank routing.
- Do not edit `workflows/otr_canonical.json`: this patch introduces no node,
  input, widget, link, or saved value. Still run all workflow validation gates.

## P3. Executable regression coverage

### P3.1 Pure policy tests — `tests/test_otr_dialogue_policy.py`

Replace the roster-oriented tests with active-speaker tests:

1. `active_speakers=("MARLOW",)` returns the base prompt byte-for-byte and the
   result contains neither `Cockney` nor the policy constant.
2. `active_speakers=("  lemmy  ",)` appends the policy.
3. A mixed active set appends exactly one copy and contains both
   `For LEMMY's spoken lines only` and the non-Lemmy isolation clause.
4. An empty tuple returns the base prompt byte-for-byte.
5. A cast dictionary or `_CastShim`-like object raises `TypeError`; the API must
   not accept roster-shaped values under an active-speaker name.
6. A positional second argument raises `TypeError`, pinning the keyword-only
   category boundary.

### P3.2 Per-line integration — `tests/test_phase1_composer_prompt.py`

Use the existing `_recording_creative` fake and inspect the role=`system`
message captured from `compose_line`:

1. ALICE active with `allowed_people={"ALICE VALE", "LEMMY"}`: no occurrence
   of `Cockney` or the policy constant. This test fails on the current code and
   also fails a text-only patch that leaves roster injection in place.
2. LEMMY active with another character in `allowed_people`: explicit
   Lemmy-only rule and standard-English spelling clause are present.
3. Force an empty first reply so the correction retry occurs. Assert the first
   and second calls carry identical system-message content and the correct
   active-speaker scope.
4. Retain existing assertions that retry adds only the rejected assistant turn
   and correction user turn; do not alter retry control flow.

### P3.3 Exchange integration — `tests/test_compose_exchange.py`

Add prompt-capture tests using existing `VoicedSlot` and `_CountingGen` hooks:

1. A group containing only MARLOW/REESE gets no Cockney policy.
2. A mixed LEMMY/MARLOW group gets the explicit Lemmy-only rule and the
   other-character isolation clause.
3. Exercise one Tier-A failure followed by repair and assert both generated
   calls carry identical scoped system content.
4. Exercise `run_exchange_prepass` with production-shaped cast dictionaries
   containing Lemmy but a non-Lemmy active group. Assert no Cockney policy is
   introduced after `_normalize_cast` and grouping.
5. Keep `tests/test_exchange_seam_lane2.py` green; its no-Lemmy static prompt
   must remain byte-identical.

### P3.4 Separate-lane controls

Run `tests/test_scifi_news_pro_lemmy_cameo.py` unchanged. Its existing
normalization test pins Lemmy's register without changing the other shape.
Only add a new test there if the implementation unexpectedly touches that
module; otherwise a test for untouched code would add noise rather than kill
this defect.

## P4. Build-breaker contract audit

Before editing, and again against the frozen diff, the implementer must run
these exact checks:

1. `rg -n "append_dialogue_policy|roster_has_lemmy" nodes tests` proves every
   caller/import was migrated and no old positional/roster call remains.
2. Confirm the new values are real active speaker names:
   `LineRequest.speaker: str` on the line path and each
   `VoicedSlot.speaker: str` on the exchange path. Cast rows, `char_id`, roles,
   families, registries, and persona objects are different categories and must
   not be substituted.
3. Confirm `compose_line_draft` owns prompt construction/retry, while
   `compose_line` returns `LineResult`; tests must inspect the recording fake,
   not assume `compose_line` returns a raw string.
4. Confirm exchange repair rebuilds through `_run_once` ->
   `build_exchange_prompt` with the same `beat_group`; do not invent a retry
   parameter or mutate recorded calls after the fact.
5. Confirm the canonical file still has writer widget slot 13 set to `true`.
   Do not claim `use_exchange` can be toggled with the sanctioned headless
   `--set`: `scripts/otr_api.py:827-859` does not whitelist it.
6. Confirm `scifi_news_pro` still dispatches before the inline call sites; do
   not apply this helper to that separate pipeline.
7. Re-run `git diff --check`, Python AST parsing on touched `.py` files, BOM
   checks, zero-byte checks, and `git status --short` before commit.

Any reviewer recommendation naming a function, type, enum/category, registry
field, workflow slot, CLI option, or output path not proven above is
`UNVERIFIABLE` until checked in the real Windows repo. It cannot become a build
instruction by plausibility or panel vote.

## P5. Verification sequence

### P5.1 Focused tests after the code edit

Run with `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`,
`PYTHONUTF8=1`, and `-p no:cacheprovider`:

```text
tests/test_otr_dialogue_policy.py
tests/test_phase1_composer_prompt.py
tests/test_compose_exchange.py
tests/test_exchange_seam_lane2.py
tests/test_scifi_news_pro_lemmy_cameo.py
tests/test_workflow_json_guardrails.py
```

### P5.2 Required repository gates

1. Full project regression suite. Start it in the background and poll its log
   because it exceeds the Desktop Commander timeout ceiling.
2. Separate Bug Bible regression from
   `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` using
   relative test path `tests\bug_bible_regression.py`.
3. Canonical JSON parse/round-trip.
4. `OTR_WorkflowValidator` plus widget-count/live-`INPUT_TYPES`, wired input
   name, link integrity, and live `/object_info` gates.
5. Confirm the canonical workflow was not modified. If a workflow diff exists,
   stop and explain the concrete node/input/widget/link reason before keeping
   it; this repair has none.

### P5.3 Production qualification

Only after all static and test gates pass:

1. Selectively reset OTR/ComfyUI headless processes; never blanket-kill Python.
   Confirm the chosen API port is free and GPU memory is at desktop baseline.
2. Run the sanctioned `scripts/otr_headless_canonical.ps1`, which loads
   `workflows/otr_canonical.json`, with one act, an affected bank, and
   `OTR_LedgerScriptWriter.lemmy_cameo=always include`. The exact widget value
   is a real `INPUT_TYPES` choice and is whitelisted by `scripts/otr_api.py`.
3. Qualify at least one `media_archive` episode. Prefer the model/bank pairing
   from an operator-observed bleed artifact so the fix is compared on the same
   reachable lane.
4. Inspect the saved ledger: cast contains Lemmy plus at least one other
   character; `exchange_prepass_audit` proves grouped composition occurred;
   Lemmy remains recognizably Cockney with standard spelling; non-Lemmy lines
   retain their own register without concentrated Lemmy idiom.
5. Require `RESULT SUCCESS`, `Prompt executed`, `obs_publish OK`, and the final
   published asset under the live `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs`
   tree. The resident server after completion is not a crash.
6. Run a `scifi_news_pro` forced-Lemmy control only if the focused/full suite
   or affected-bank qualification suggests cross-pipeline drift. The code diff
   should not reach that runner.
7. Do not create a stale/ad-hoc workflow to live-test `use_exchange=False`.
   The per-line path is qualified deterministically by captured-prompt tests;
   it remains reachable as the production fallback for failed/singleton groups.

## P6. Records, commit, and push

1. The defect already satisfies the admission rule through operator-observed
   published episodes. After the fix is qualified, re-scan `PROD_BUG_LOG.md`
   for the next free PBUG id, record the live evidence, root cause, fix commit,
   and verification receipt, and update `GO_FORWARD_PLAN.md`/`HANDOFF_LOG.md`.
2. Promote a dialogue-scope Bug Bible entry with executable presence-and-
   absence coverage. Re-scan the separate Bible repo for the next free id at
   implementation time; do not hard-code today's apparent next number.
3. Remove temporary probes. Preserve unrelated user changes, including the
   pre-existing untracked `uv.lock`.
4. Commit the green project chunk on `v2.0-alpha`, push immediately, then prove
   local `HEAD == origin/v2.0-alpha`.
5. Verify touched Python files parse, all touched text is UTF-8 without BOM,
   and no tracked file is zero bytes.

## P7. Stop conditions

Stop before push if any of these occurs:

- A non-Lemmy per-line captured system prompt still contains `Cockney`.
- A mixed exchange lacks explicit Lemmy-only and other-character isolation.
- The helper still accepts cast dictionaries/objects as active speakers.
- A reviewer-proposed API/category cannot be found in the real repo.
- Any source bank outside the two inline callers changes unexpectedly.
- The canonical workflow changes without a proved graph-contract need.
- Any focused/full/Bible/workflow validation fails.
- A production run lacks its canonical ledger/published-asset receipts.
