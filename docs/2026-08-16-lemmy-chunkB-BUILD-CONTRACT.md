# Lemmy Chunk B -- BUILD CONTRACT (the cameo roll on the content-owned lanes)

**Self-contained and normative.** Output of the full four-round
`kibitz-plugin:kibitz` arc of 2026-08-16 (artifacts under
`kibitz-runs/2026-08-16-lemmy-chunkB*/`). Supersedes the r1-r3 round finals;
nothing here depends on reading them. Panel provenance, stated precisely:
Codex covered all four rounds; Antigravity covered r1-r3 (each via one
file-handoff retry) and was QUOTA-HELD at r4 (`RESOURCE_EXHAUSTED` 429,
kibitz quota-hold on record); Fable gave the r1 cold first opinion; a
three-agent grounding fan-out verified the r3 wiring claims 6/6. Driver:
Claude, sole judge; every folded claim grounded against the tree at
`da44f642`.

## SCOPE CUT 2026-08-16 -- this contract is now ONE LANE

`scifi_news` was RIPPED hours after this contract was written
(PBUG-20260816-01): the operator retired it on measured evidence and made
`scifi_news_pro` the standard. **The entire codex half of this contract is
MOOT** -- the schema-locked slot reserve, the five grammar-decoded vocabulary
sites, the P5/P5R speaker-contract route and the conditional preset pre-seed
all described a module that no longer exists. The panel's hard half went with
it. Everything below that names the codex lane is retained ONLY as the record
of what was decided and why; the BUILDABLE contract is the fable2 sections.

## What this builds

The surviving content-owned lane -- `scifi_news_pro`
(`nodes/_otr_scifi_fable2.py`) -- gains the LEMMY cameo ROLL:
decided at runner entry, before any authoring; identity pinned from
`config/cast_pools.LEMMY_PROFILE`; the model authors only his
episode-specific stake; the contract and receipts tell the truth in every
knob state. Chunk A (the contract stamp, pushed `da44f642`) is extended, not
replaced.

## The decision API (in `nodes/_otr_casting.py` -- placement SETTLED)

`resolve_lemmy_cameo(source_bank_id, force_lemmy) -> LemmyCameoDecision`,
called EXACTLY ONCE at each runner entry. The truth table is normative:

| condition | lemmy_hit | lemmy_policy | knob_state | roll_executed |
|---|---|---|---|---|
| excluded bank (any knob) | False | `source_fidelity_exclusion` | as requested | False |
| non-excluded, force True | True | `operator_cameo` | `forced_include` | False |
| non-excluded, force False | False | `operator_cameo` | `forced_exclude` | False |
| non-excluded, force None | roll | `operator_cameo` | `natural_roll` | True |

`roll_lemmy()` is called exactly once in the last row and zero times
elsewhere (unit-pinned). Exclusion uses `_source_bank_excludes_lemmy`
(family-normalized) and outranks force. The decision is immutable and
exposes `to_meta()` -> a PRIMITIVE-ONLY dict with pinned key set
`{schema_version: "lemmy-cameo-decision.v1", lemmy_hit, lemmy_policy,
knob_state, source_bank_id, roll_executed}`; `json.dumps(to_meta())` is
unit-pinned. WHY: `Ledger.save()` never raises -- a non-serializable object
in meta logs a warning, returns None, and a dozen call sites never check.
The knob-state and policy strings are module constants beside the type; no
other spelling may appear anywhere (producer/consumer string drift is a
Bible-covered class). Name comparisons for LEMMY use the existing
case/spacing-normalized helper, never raw equality.

`content_owned_cast_contract` gains `decision=None`: None preserves today's
chunk-A behavior BYTE-FOR-BYTE (so the API commit lands green before either
runner migrates -- its current callers at codex `_stamp_cast_contract` and
fable2 `_stamp_cast_contract` keep working); a decision fills
`lemmy_hit`/`lemmy_policy` from it. The five-key shape and the ABSENCE of
`cast_seed`/`cast_seed_source` are unchanged and stay pinned by
`tests/test_content_owned_cast_contract.py`.

## Widget plumbing (writer-shared)

`_resolve_inputs` gains keyword-only `lemmy_cameo: str = "roll (~11%
chance)"`. It validates EXACT membership in `_LEMMY_CAMEO_FORCE` BEFORE any
RSS/source work and raises `ValueError` naming the three choices -- the
current legacy path's `.get()` silently turns a typo into a natural roll,
which is the bug class `tests/test_workflow_apply.py`'s own docstring asks
to close. It stores the mapped value as `resolved["lemmy_force"]`
(Optional[bool]). `run()` FORWARDS `lemmy_cameo=lemmy_cameo` at its
`_resolve_inputs` call site -- the omission of exactly this forwarding is
how the widget has been inert on dispatched lanes, so the commit carries a
reach-test proving the widget value arrives in `resolved["lemmy_force"]`,
not merely that the mapping dict is correct. The legacy block (~:4609) and
both runners read ONLY `resolved["lemmy_force"]`.

NO INPUT_TYPES change, NO new widget, NO workflow JSON mutation: the widget
exists at node 1 index 12 and is already headless-whitelisted. The
34-widget order pins (`test_source_ref_widget.py:58`,
`test_openrouter_slot_widgets_s2.py:91`) run untouched.

Fixture updates, complete list: `tests/test_fable2_source_windows.py:505,
:578,:781`, `tests/test_p0_deterministic_repair_wired.py:488`,
`tests/test_p0_source_windows.py:309`, (the codex lane test, deleted with the rip)
add `lemmy_force=False`; the two bare `SimpleNamespace()` ledgers in the p0
fixtures become save-capable doubles (real `production_ledger.Ledger` or a
double whose `save()` returns a truthy path and preserves `.data`),
because the runners gain an entry-time checked save. `tests/
test_workflow_apply.py:341-351`: the typo-silent-fallback assertion moves
to a `_resolve_inputs` ValueError test; the three mapping assertions stay.

## Shared runner shape (both lanes, at entry)

1. `decision = resolve_lemmy_cameo(bank_id, resolved.get("lemmy_force"))`.
2. Stamp `meta["lemmy_roll_receipt"] = decision.to_meta()` and make it
   DURABLE: fable2 calls `require_ledger_save(led, "the LEMMY roll
   receipt")` (the fail-loud boundary helper -- NOT the warning-only deal
   save); codex adds its first mid-run checked save, raising
   `CodexLedgerSaveError` when `save()` returns None. Both runners then
   REACQUIRE meta from `led.data` (save rebinds; fable2's own module
   warns against retained aliases) and use only the reacquired mapping.
3. At the existing `_stamp_cast_contract` sites, pass the SAME decision:
   the contract's `lemmy_hit`/`lemmy_policy` and the receipt can never
   disagree, and the locked count stays the post-assembly truth.
4. Voice-slot receipt: on a hit, stamp
   `meta.cast_voice_slots[<lemmy_char_id>]` with the established six-field
   shape -- gender `"male"`, timbre `[]`, role `""`, age_band `""`,
   `LEMMY_PROFILE["speech_signature"]`, `sha1(description)[:12]` (the
   pre-locked-row convention; credits read the signature ONLY from this
   key). Cameo row only; no receipts invented for other rows.

## Fable2 (`scifi_news_pro`)

* ON HIT, the pitch AND treatment prompts carry the cameo contract: LEMMY
  exists (fixed name; genial Cockney COMMUNICATIONS OFFICER -- the
  canonical description; NOT "gravelly engineer", a stale comment fixed in
  passing) and must be one of the cast shapes; the prompt REQUESTS
  `max(0, n_max - 1)` non-Lemmy characters. REQUEST, never a cap: cast
  size is a request (`tests/test_cast_size_is_a_request.py` -- THE
  DIRECTIVE), and the only enforced bound remains the physical
  `MAX_SPEAKING_CAST` stock limit that exists today.
* `_make_treatment_validator` gains the decision: on hit, exactly one
  case-normalized LEMMY shape AND at least one non-LEMMY shape; on miss or
  exclusion, zero LEMMY shapes (the name is reserved show-wide on these
  lanes -- an accidental second Lemmy with a random voice is the failure
  this prevents). Today's validator checks only the stock bound, so all of
  this is new and additive.
* Post-acceptance, a deterministic pass normalizes ONLY `name` and
  `register` on the model's LEMMY shape -- `register :=
  LEMMY_PROFILE["speech_signature"]` (the profile has NO `register` key;
  the script prompt consumes `shape.register` verbatim, which is how the
  Cockney reaches written dialogue). Model-authored `want`/`pressure`/
  `role` are preserved untouched: the stake is what makes him belong to
  THIS story, and a Python-canned stake is the stapled-on cameo the
  operator's 6-line exemplar bar rejects.
* ONE speaker list: compute `speaker_order` once from the parsed script;
  derive `casting_speakers` (excludes LEMMY on hit); PASS THAT LIST into
  `_pass_casting` and `_make_casting_validator` (the pass currently
  recomputes all speakers internally -- computing the list earlier without
  passing it changes nothing); size `_deal_voice_menu` from
  `len(casting_speakers)` and deal it with
  `taken={LEMMY_PROFILE["voice_preset"]}` -- conditional on the hit. This
  closes both the double-allocation (`v2/en_speaker_8` is in the open
  pool; its timbre row literally reads gravelly/engineer/mechanic) and the
  9-presets-vs-10-assignments capacity mismatch.
* `_assign_voices` receives full `speaker_order` plus the decision and
  synthesizes the LEMMY row at his speaker-order position from the
  profile, normalized to the lane row shape. Gate (b) (speaker set == cast
  rows) holds by construction: he is in the script AND the cast; he is
  only absent from the casting-LLM artifact, whose menu never offered his
  voice. `_assert_unique_bark_voices` runs on the merged rows before
  `_assemble`.

## Codex (`scifi_news`) -- MOOT, lane ripped 2026-08-16 (record only)

* ON HIT, P2's `artifact_inputs` carries the cameo contract (the seam
  exists; P2 today receives only P1's question). One story slot of
  c01-c03 is RESERVED for LEMMY -- "reserve", not "displace": P2 receives
  no cast cardinality, so there is no defined character to displace, and
  no hit/miss cast-size parity is promised. The schema is NOT widened: no
  c04, no Literal edits, no `max_length` change, anywhere. NARROWING,
  explicit: the no-new-cap law governs what the CAMEO introduces; the
  lane's pre-existing 2-4-row `CastPlanV4` is a structural bound like the
  beat topology and stays as it is. A wider codex cast is its own future
  item.
* `post_validator` becomes `_make_cast_plan_validator(decision)`: on hit,
  exactly one case-normalized LEMMY row PLUS at least one non-LEMMY story
  row (keeps him a cameo; steers off the Lemmy-as-only-character edge
  without capping part size); on miss/exclusion, zero LEMMY rows. Checks
  presence/count ONLY -- never prose -- so retries are never burned on
  cosmetic drift.
* Normalization is NOT in `repair_cast_plan_metadata`: its sole caller
  passes no context, and the repair rung runs only on REJECTION -- an
  accepted plan never enters it. The deterministic pass runs
  unconditionally on the ACCEPTED `p2`, before
  `_codex_target_beat_count(steer.act_count, len(p2.cast))` (reserve does
  not change len): it rewrites the model-designated LEMMY row's `name`,
  `gender`, `character_description` from the profile, preserving
  `char_id`, `voice_slot`, `role_in_conflict`. Never touches another row.
* Speaker contract to the dialogue author: `CastPlanRowV4` has no
  signature field and is NOT modified. An OUT-OF-BAND char_id-keyed map
  with exact keys `{name, speech_signature, description_digest}` --
  profile signature for the normalized LEMMY row, `""` for others, the
  existing sha1[:12] digest convention -- is built in
  `_script_artifact_inputs` (which gains the cast), threaded through
  `_call_script_text_draft`, every `_beat_dialogue_inputs` payload, and
  the manually rebuilt P5R dictionary. The `_call_scene_review` chunk
  loop clones the supplied dict into EVERY 8-row chunked call, so the map
  rides every chunk with no separate mechanism; an oversized-scene test
  (> 8 rows) inspects every chunked request to prove it.
* `_assemble_ledger`: ON HIT, pre-seed `used` with the profile preset and
  branch the LEMMY row to the fixed profile instead of `_pick`
  (`presentation_gender` from profile gender). STRICTLY conditional: the
  `used` set drives every other allocation, and an unconditional pre-seed
  would change forced-miss episodes' voices.

## Byte-identity scope (stated so nobody over-claims)

On a forced miss, cast/scenes/shots/beats/lines/music are byte-identical
to today under pinned inputs; the ONLY meta deltas are `cast_contract`
(policy now truthfully `operator_cameo` with `knob_state`) and
`lemmy_roll_receipt`. Forced-miss proof is UNIT-LEVEL on both lanes
(including unchanged voice allocation with no reservation); the live legs
are the two forced HITS only -- a live miss leg buys nothing the units do
not.

## Land order -- each step: focused tests, FULL suite, Bible, AST/JSON
checks, commit, push, verify HEAD == origin/v2.0-alpha

1. Plumbing (writer-shared): `_resolve_inputs` + `run()` forwarding +
   legacy consumer + all fixture updates + workflow_apply evolution.
2. Decision API + serialization + `content_owned_cast_contract(decision=
   None)` + chunk A test extensions.
3. Fable2, complete with tests.
4. Codex, ONE commit: P2 validator + normalization + speaker-contract
   route + conditional pre-seed + stamps + tests (an intermediate commit
   would pin the voice while dialogue cannot see the register).
5. Acceptance: RESET the box per CLAUDE.md section 4, boot a FRESH server
   at the implemented HEAD (the resident server holds pre-chunk-B modules
   -- acceptance against a stale boot proves nothing), then one forced-hit
   leg (one lane remains) through `workflows/otr_canonical.json` via
   `scripts/otr_canonical_api_run.py --source-bank <lane bank> --set
   "OTR_LedgerScriptWriter.lemmy_cameo=always include"`. Per leg:
   `Prompt executed` + `obs_publish OK` in the server log, the canonical
   asset under `otr/episodes/<ep>/` AND the published file under
   `otr/obs/`, exactly one LEMMY cast row with pinned identity, >= 1
   spoken line resolved via his `char_id` FROM THE CAST ROW (never
   name-matching lines -- the operator's own detector trap), the
   serialized `lemmy_roll_receipt`, the five-key seed-free
   `cast_contract` agreeing with it, the writer-stage Bark preset
   `v2/en_speaker_8` on the frozen row AND the delivered
   engine/`voice_ref_id` after CastLock (the canonical graph runs
   auto_registry + indextts2, so delivery may be the qualified IndexTTS2
   route), the credits row carrying the signature from the voice-slot
   receipt, and a portrait prompt grounded in the pinned description.
   Bundle all of it into one acceptance receipt per leg for auditing.
   Then full suite + Bible + variants once more.

## Out of scope (final)

Part SIZE/fidelity (operator-deferred); `media_archive`/`original`
(already live via `lock_cast`); the 11% rate (statistical pin untouched);
voice qualification / IndexTTS2 routing; c04 or ANY schema widening; voice
slots for non-cameo rows; new widgets/nodes/links or workflow JSON edits;
a second LLM pass of any kind.

## In-passing text fixes (land with step 2)

`nodes/_otr_casting.py:333` "gravelly engineer" -> the canonical
communications-officer description; `config/cast_pools.py:724` "always
rendered through Bark" -> Bark preset is the writer-stage identity,
delivery may resolve the qualified IndexTTS2 route.
