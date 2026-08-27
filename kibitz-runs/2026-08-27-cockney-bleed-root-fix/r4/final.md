# Lemmy Cockney-Bleed Root Fix — Code-Ready Plan

Status: BUILD-READY, Kibitz R1-R4 converged
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

### P1.5 Published production evidence

The diagnosis is not fixture-only. These live ledgers are the admission
artifacts and the preferred comparison points for qualification:

- `signal_lost_glass_shards_and_broken_promises_20260825_094527` — `original`,
  Mistral Nemo, 24 exchange-composed beats. Non-Lemmy QUINN STONE says
  "Bloomin' 'ell", "innit", and "Blimey".
- `signal_lost_the_cat_from_outer_space_reel_20260826_012848` —
  `media_archive`, Gemma 4 E2B, four exchange-composed beats. Non-Lemmy MINDY
  SIMPSON says "ain't right" and "playin' coy".
- `signal_lost_framing_the_proof_20260826_024816` — `media_archive`, Gemma 4
  E2B, four exchange-composed beats. Non-Lemmy QUINN OKAFOR says "see?" and
  "the whole shebang".
- `signal_lost_the_caretakers_clause_20260826_155835` — `scifi_news_pro`,
  Gemma 4 12B control. Its separate runner gives Lemmy his own register without
  routing through the affected helper.

All four are under
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<episode>\audio\`.

## P2. Minimal implementation

### P2.1 Replace roster semantics with active-output semantics

Edi`nodes/_otr_dialogue_policy.py`:

1. Replace `roster_has_lemmy` with the exact private predicate
   `_active_speakers_have_lemmy(active_speakers: Sequence[str]) -> bool`.
   Import `Sequence` from `collections.abc` and remove the now-unused
   `Any`, `Dict`, `Iterable`, and `Union` imports. Delete the dictionary/
   `char_id` roster behavior; no production caller needs it after this patch.
2. Make `append_dialogue_policy` keyword-only for the new category:

   ```python
   def append_dialogue_policy(
       system_prompt: str, *, active_speakers: Sequence[str]
   ) -> str:
   ```

3. Use this exact validation order in the predicate:

   ```python
   if isinstance(active_speakers, (str, bytes)) or not isinstance(
       active_speakers, Sequence
   ):
       raise TypeError("active_speakers must be a sequence of speaker-name strings")
   speakers = tuple(active_speakers)
   if any(not isinstance(speaker, str) for speaker in speakers):
       raise TypeError("active_speakers must contain only speaker-name strings")
   return any(speaker.strip().upper() == "LEMMY" for speaker in speakers)
   ```

   Requiring a real `Sequence` rejects mappings, sets, generators, scalar
   strings/bytes, and `_CastShim` objects at the container boundary. Validate
   every element before testing any name so `("LEMMY", wrong_object)` cannot
   hide a bad category behind an early match.
4. Match Lemmy case-insensitively after surrounding-whitespace removal.
5. Preserve `system_prompt or ""` and return it unchanged when no active
   speaker is Lemmy.
6. Keep the name `_COCKNEY_ORTHOGRAPHY_RULE`, but replace its old unscoped
   content with the scoped text below. Preserve the leading `"\n\n"` separator
   so the appended directive cannot fuse onto the routed prompt's final line.
   The single static block is intentionally shared by Lemmy-only and mixed
   exchange calls; the other-character sentence is harmless on a one-speaker
   turn and avoids a second prompt branch.

Use this exact constant shape (line wrapping may follow formatter output):

```python
_COCKNEY_ORTHOGRAPHY_RULE = (
    "\n\nFor LEMMY's spoken lines only, convey his Cockney accent through "
    "phrasing, idiom, cadence, and rhythm. Every other character must retain "
    "that character's own speech register; do not give any other character "
    "LEMMY's Cockney phrasing, idiom, cadence, or rhythm. Use standard English "
    "spelling in every spoken line; do not encode pronunciation with phonetic "
    "misspellings."
)
```

Update the helper docstring to define `active_speakers` as current output
speakers only, sourced from `LineRequest.speaker` or `VoicedSlot.speaker`, and
to warn explicitly against full-cast rows/lists.

The negative clause is required for mixed exchanges. Absence, rather than a
negative mention, is required for a non-Lemmy per-line call.

This decision explicitly supersedes the sketch in `docs/GO_FORWARD_PLAN.md`
that says to leave the orthography sentence global. Here, "global" means every
line inside a Lemmy-containing mixed response—not every unrelated non-Lemmy
model call. A non-Lemmy active set receives zero policy bytes so its resolved
system prompt remains unchanged.

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

Keep the existing `cast` parameter and persona lookup in
`build_exchange_prompt`: it correctly renders voice guidance only for speakers
in the current exchange. The change removes `cast` solely from the policy
decision; deleting the user-facing roster/persona block is out of scope.

The keyword-only helper signature and both caller migrations are one atomic
code change. Landing the signature before either caller would raise `TypeError`
on every dialogue composition.

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

Replace only `test_roster_has_lemmy` and `test_append_dialogue_policy` with
active-speaker tests. Leave every Lemmy profile, route, and BUG-12.86
qualification-receipt test later in the file intact:

- Remove only the obsolete `roster_has_lemmy` name from the module import.
- Keep importing `_COCKNEY_ORTHOGRAPHY_RULE`; integration tests use that
  canonical constant rather than treating any generic occurrence of the word
  `Cockney` as the policy.

1. `active_speakers=("MARLOW",)` returns the base prompt byte-for-byte and the
   result does not contain the policy constant.
2. `active_speakers=("  lemmy  ",)` appends the policy.
3. A mixed active set has
   `result.count(_COCKNEY_ORTHOGRAPHY_RULE) == 1` and contains both
   `For LEMMY's spoken lines only` and the non-Lemmy isolation clause.
4. An empty tuple returns the base prompt byte-for-byte.
5. `active_speakers="LEMMY"` and `active_speakers=b"LEMMY"` raise
   `TypeError`; neither may silently iterate scalar characters/bytes.
6. A cast mapping, set/generator, `_CastShim`-like object, sequence containing a
   dict/object, and `("LEMMY", wrong_object)` raise `TypeError`; the API must
   not accept or partially hide roster-shaped values under an active-speaker
   name.
7. A positional second argument raises `TypeError`, pinning the keyword-only
   category boundary.

### P3.2 Per-line integration — `tests/test_phase1_composer_prompt.py`

Add
`from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE` to this test
module; it does not currently import the constant.

Pass the existing `_recording_creative` fake directly as
`compose_line(creative_fn=creative, ...)`. Its capture shape is a list of
message lists, so use
`messages = creative.state["calls"][call_index]` and
`system = messages[0]`; assert `system["role"] == "system"` and inspect only
`system["content"]`:

1. ALICE active with `allowed_people={"ALICE VALE", "LEMMY"}`: no occurrence
   of the policy constant in the captured system content. This test fails on the current code and
   also fails a text-only patch that leaves roster injection in place.
2. LEMMY active with another character in `allowed_people`: explicit
   Lemmy-only rule and standard-English spelling clause are present.
3. Force an empty first reply through this same fake so the existing correction
   retry occurs. Assert the first and second calls carry identical system-
   message content and the correct active-speaker scope.
4. Retain existing assertions that retry adds only the rejected assistant turn
   and correction user turn; do not alter retry control flow.

### P3.3 Exchange integration — `tests/test_compose_exchange.py`

Add
`from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE` to this test
module; importing `_otr_compose_exchange as ce` does not expose that constant.

Add prompt-capture tests using existing `VoicedSlot` and `_CountingGen` hooks.
Define a local two-slot LEMMY/MARLOW fixture and matching raw response; the
existing `_raw_for` is hard-coded for the MARLOW/REESE fixture:

1. Call `build_exchange_prompt` directly for a MARLOW/REESE group and assert no
   Cockney policy.
2. Call `build_exchange_prompt` directly for a mixed LEMMY/MARLOW group and
   assert the explicit Lemmy-only rule and other-character isolation clause.
3. Call `compose_exchange` with the local LEMMY/MARLOW raw response and a
   fail-then-ok stateful Tier-A checker (the pattern already used by
   `test_repair_triggers_once_then_succeeds`). Assert both generated
   calls carry identical scoped system content. `_CountingGen` stores dicts, so
   the exact access is
   `gen.calls[call_index]["messages"][0]["content"]`; compare only the system
   strings because repair reasons intentionally change the user message.
4. Exercise `run_exchange_prepass` with production-shaped cast dictionaries
   containing Lemmy but a non-Lemmy active group. Assert no Cockney policy is
   introduced after `_normalize_cast` and grouping. Wrap the existing
   `_fake_gen_valid` behavior in a recorder because that fake currently stores
   no calls. Preserve its real callable signature
   `(messages, *, temperature=0.0, max_new_tokens=0)` and its parsing of the
   `dNNN|SPEAKER: <line>` format block; assert against the captured
   role=`system` entry, never the joined user roster. This is a forward scope
   invariant; the direct mixed-group test above is the test that kills the
   current grouped implementation.
5. Keep `tests/test_exchange_seam_lane2.py` green as a routing-seam regression,
   but do not treat it as the Cockney-absence lock: today it checks only the
   static prefix and grounding clause. Items 1-4 supply the explicit presence
   and absence assertions.

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
   Do not `str()`-coerce arbitrary objects at the two policy call sites: the
   production exchange prepass already normalizes beat speakers to strings,
   while `LineRequest.speaker` is a typed string contract. Coercing a wrong
   category into a plausible repr would defeat the fail-loud boundary.
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
8. Require `git diff -- workflows/otr_canonical.json` to be empty. General
   whitespace checks do not prove the canonical graph stayed untouched.

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

Use a red/green sequence without committing the red state: add the defect-
killing assertions first and run the focused subset against the unmodified
implementation. The ALICE-with-Lemmy-in-`allowed_people` test and mixed-exchange
scope test must fail. Then make the atomic helper/caller/import change and
require the same subset to pass. If either key test is green before the code
change, stop and fix the test because it does not kill this defect.

### P5.2 Required repository gates

1. Full project regression suite. Start it in the background and poll its log
   because it exceeds the Desktop Commander timeout ceiling.
2. Separate Bug Bible regression from
   `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` using
   relative test path `tests\bug_bible_regression.py`.
3. Canonical JSON parse/round-trip.
4. `OTR_WorkflowValidator` plus widget-count/`INPUT_TYPES`, wired input-name,
   and link-integrity gates. A live `/object_info` comparison is required only
   if the frozen diff unexpectedly changes the workflow or a node contract;
   this plan forbids both.
5. Confirm the canonical workflow was not modified. If a workflow diff exists,
   stop and explain the concrete node/input/widget/link reason before keeping
   it; this repair has none.

### P5.3 Production qualification

Only after all static and test gates pass:

1. Selectively reset OTR/ComfyUI headless processes; never blanket-kill Python.
   Confirm port 8000 is empty per the project reset rule and GPU memory is at
   desktop baseline before invoking the wrapper. The wrapper also resets but
   only logs VRAM; its log is not a substitute for this hard pre-boot gate.
2. Run the sanctioned wrapper directly from PowerShell with an explicit bank,
   forced cameo, and three acts:

   ```powershell
   & .\scripts\otr_headless_canonical.ps1 `
     -Profile none `
     -Acts 3 `
     -Set @(
       "OTR_LedgerScriptWriter.source_bank=media_archive",
       "OTR_LedgerScriptWriter.lemmy_cameo=always include"
     )
   ```

   `source_bank` and `lemmy_cameo` are real whitelisted writer widgets, and
   `always include` is an exact choice. Quoting preserves its space. Never leave
   `source_bank` at the canonical `roll (any eligible bank)`, which can select
   an unaffected or cameo-excluded lane.
   Execute this as a direct PowerShell script call exactly as shown so
   `-Set @(...)` binds the script's one `string[]` parameter. Do not serialize
   that array expression through a native `powershell.exe -File` boundary.
   Omit `-Port` intentionally: the wrapper's real default `Port=0` selects a
   free ephemeral API port. Record the chosen port from its log rather than
   claiming the run listened on 8000.
   Before accepting the leg, require the runner's applied-patch receipt to show
   `source_bank='media_archive'` and `lemmy_cameo='always include'`.
3. Use the current canonical creative/technical model unchanged and record the
   resolved IDs from the ledger. The canonical graph currently names Gemma 4
   12B; do not hard-code Gemma 4 E2B from an older evidence run unless its exact
   live combo label is first resolved from `INPUT_TYPES`. Prompt-scope tests,
   not model identity, are the causal proof.
4. Separate the proof roles. Captured-prompt tests are the hard, deterministic
   scoping gate. The live run proves production reachability and supplies the
   listening gate; a small lexical sample cannot prove that bleed is impossible
   and must not become a dialogue blacklist.
5. Inspect the saved ledger: cast contains Lemmy plus at least one other
   character; `exchange_prepass_audit` proves grouped composition occurred;
   Lemmy remains recognizably Cockney with standard spelling; non-Lemmy lines
   retain their own register without concentrated Lemmy idiom.
6. Prove the live mixed-group path rather than merely any exchange with a
   temporary read-only audit that imports the production
   `group_voiced_beats` helper:

   - Build a multimap of nonblank `ledger.lines[].beat_id` values; do not use a
     last-write-wins dictionary comprehension. Normal music-opening/closing
     rows with blank beat IDs are ignored. A duplicate nonblank beat ID fails
     the receipt.
   - Walk the complete `ledger.beats[]` order. For each beat, require exactly
     one matching line row. Zero matches, invalid/non-`d###` slot, empty
     speaker, or reserved `ANNOUNCER` is a run break exactly as `_groupable` is
     used by `run_exchange_prepass`; do not filter the breaker away.
   - For each uninterrupted valid run, construct slots carrying beat id,
     dialogue slot id, and speaker as `types.SimpleNamespace` (or an equivalent
     object with all three attributes). Do not use `VoicedSlot`, which has no
     `beat_id` and would make the later audit-set comparison impossible. Then
     call the real
     `group_voiced_beats(min_size=2, max_size=3,
     reserved_speakers=("ANNOUNCER",))`.
   - Only after grouping, retain a group when its complete beat-id set is a
     subset of `meta.exchange_prepass_audit.beat_ids`. Grouping accepted IDs
     alone can bridge a failed/singleton hole and invent a group that never ran.

   At least one retained group must include `LEMMY` and another speaker. If none
   does, permit at most one rerun; after that, fail the mixed-exchange
   qualification rather than looping. Record the
   accepted group's beat IDs and speakers in the PBUG receipt. Do not add new
   ledger schema solely for this receipt.
7. Require `RESULT SUCCESS`, `Prompt executed`, `obs_publish OK`, and the final
   published asset under the live `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs`
   tree. The resident server after completion is not a crash.
8. Run a `scifi_news_pro` forced-Lemmy control only if the focused/full suite
   or affected-bank qualification suggests cross-pipeline drift. The code diff
   should not reach that runner.
9. Do not create a stale/ad-hoc workflow to live-test `use_exchange=False`.
   The per-line path is qualified deterministically by captured-prompt tests;
   it remains reachable as the production fallback for failed/singleton groups.
10. If bleed remains after the scoped system-policy tests pass, diagnose the
    labeled full-cast voice cards and rolling prior/last-line context before
    widening the patch. Do not preemptively remove Lemmy's correctly labeled
     voice card from other speakers' context.
11. If `media_archive` source acquisition fails for a verified external/feed
    reason, stop and record that leg as unqualified; never change banks inside
    its receipt. A separately invoked command explicitly pinned to `original`
    may prove the shared inline path, but must be labeled as an `original`
    fallback, not a media-archive pass.

## P6. Records, commit, and push

1. The defect already satisfies the admission rule through operator-observed
   published episodes. After the fix is qualified, re-scan `PROD_BUG_LOG.md`
   for the next free PBUG id, record the live evidence, root cause, fix commit,
   and verification receipt, and update `GO_FORWARD_PLAN.md`/`HANDOFF_LOG.md`.
   Replace the stale GO_FORWARD paragraph that says to leave orthography global;
   do not merely append a second contradictory note.
2. Commit and push the project-repo fix/records first. Then promote a
   dialogue-scope Bug Bible entry in the separate survival-guide repository
   with executable presence-and-
   absence coverage. Re-scan the separate Bible repo for the next free id at
   implementation time; do not hard-code today's apparent next number. That
   repository gets its own tests, commit, push, and HEAD/origin receipt.
3. Remove temporary probes. Preserve unrelated user changes, including the
   pre-existing untracked `uv.lock`.
4. Commit the green project chunk on `v2.0-alpha`, push immediately, then prove
   local `HEAD == origin/v2.0-alpha`.
5. Verify touched Python files parse, all touched text is UTF-8 without BOM,
   and no tracked file is zero bytes.

Expected implementation diff before records: exactly
`nodes/_otr_dialogue_policy.py`,`nodes/_otr_line_composer.py`,
`nodes/_otr_compose_exchange.py`, `tests/test_otr_dialogue_policy.py`,
`tests/test_phase1_composer_prompt.py`, and `tests/test_compose_exchange.py`.
Any additional production module needs a newly grounded reason. Do not add a
`roster_has_lemmy` compatibility alias or a checked-in one-off audit script.

## P7. Stop conditions

Stop before push if any of these occurs:

- A non-Lemmy per-line captured system prompt still contains
  `_COCKNEY_ORTHOGRAPHY_RULE`.
- A mixed exchange lacks explicit Lemmy-only and other-character isolation.
- The helper still accepts cast dictionaries/objects as active speakers.
- A reviewer-proposed API/category cannot be found in the real repo.
- Any source bank outside the two inline callers changes unexpectedly.
- The canonical workflow changes without a proved graph-contract need.
- Any focused/full/Bible/workflow validation fails.
- Any touched module/test fails import or pytest collection.
- A production run lacks its canonical ledger/published-asset receipts.

## P8. Kibitz convergence receipt

Codex wrote a code-grounded anchor before every fan-out and remained the sole
judge/synthesizer. Exactly two external reviewers ran per round:

- R1: Cursor (`cursor-grok-4.6-high`) + Antigravity
  (`Gemini 3.7 Flash (High)`).
- R2: Cursor (`cursor-grok-4.6-high`) + Claude Code (`sonnet`, high effort).
- R3: Cursor (`cursor-grok-4.6-high`) + Antigravity
  (`Gemini 3.7 Flash (High)`).
- R4: Cursor (`cursor-grok-4.6-high`) + Claude Code (`sonnet`, high effort).

Actual external reviewer calls: 8. All eight returned complete reviews; no
quota, credit, authentication, or structural-review failure was reported.
Round anchors, raw reviews, grounded judgments, and sequential finals are under
`kibitz-runs/2026-08-27-cockney-bleed-root-fix/`.
