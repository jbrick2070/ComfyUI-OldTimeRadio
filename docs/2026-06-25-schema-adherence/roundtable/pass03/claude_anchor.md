CLAUDE ANCHOR -- R3 (wiring / integration / sequencing). Grounded vs a nodes/ inventory grep (217 structured sites across 21 files).

VERDICT: yes-with-fixes. The chunks are sound; the integration risk is the MIGRATION SURFACE + the ladder's downstream consumers, and the plan must commit to an INCREMENTAL rollout rather than implying "cover all passes" at once.

MUST-FIX BEFORE BUILD:
1. [C5 / C0 -- migration surface] Grounded: `structured_call`/`model_validate`/
   `parse_first_json_object` appear at 217 sites across 21 files;
   `_otr_radio_editor.py` ALONE has 111, `_otr_outline.py` 27, `_otr_story_spine.py`
   19 (the normalize_length pass), `_otr_reroll.py` 11, `_otr_continuity.py` 10.
   Migrating all at once is infeasible + high-risk. FIX: make the rollout
   incremental via the classvar opt-in. C1-C3 (strict-first core +
   `_normalize_field_keys` + ladder branch) are a byte-identical NO-OP for any
   schema WITHOUT `__otr_field_aliases__` -- so they land UNIVERSALLY with zero
   behavior change. C5 then annotates ONLY the high-value schemas a writer-swap
   actually exercises: outline macro/phase/beat, story_spine normalize_length,
   casting, news_interpreter, story_critic. `_otr_radio_editor.py` (111) + the long
   tail are an explicit DEFERRED follow-up chunk, not v1. State this sequencing.

2. [C4 wiring -- closure has N callers] Schema-into-repair requires `structured_call`
   to pass `schema` into `make_dispatching_repair_factory`. Enumerate EVERY caller
   that builds a dispatching factory (grep `make_dispatching_repair_factory(`) and
   forward `schema` at each; CONFIRM the existing `deterministic_repair` callback
   (cast_membership Levenshtein short-circuit, used by the outline phase stage)
   still composes with the closure change -- it returns a finished instance and
   must keep doing so. One function changes; several call sites must be updated in
   the same commit or the dispatcher build breaks.

3. [C3 -- ladder consumers downstream] Skipping Attempt 2 changes attempt
   counts/timing. Callers pass explicit `max_attempts` and the freeze cascade /
   reroll read attempt outcomes. Verify the new branchy ladder reports `attempts`
   accurately in `StructuredCallFailedError` and to those consumers -- a
   miscounted/relabeled attempt must not change a downstream cap or
   "too_many_edits" decision (L5a edit-cap territory). Add a test asserting the
   attempt accounting for each failure-class path.

SHOULD-FIX:
1. [C2/C6 -- the canonical first target] `normalize_length` (in `_otr_story_spine.py`,
   ON `structured_call` -- CONFIRMED by the live `[OTR_StructuredCall]
   'normalize_length[...]'` log) is the natural v1 opt-in + harness anchor:
   annotate the StorySpine beat schema `__otr_field_aliases__` from the real Opus
   shape ({index,lever,beat_index}) and ship that exact failing object as fixture #1.
2. [Sequencing] Land C6 (harness) BEFORE the C5 opt-ins so each schema annotation is
   regression-guarded the moment it is added, not after.
3. [C4] Because schema-in-repair helps EVERY migrated pass immediately (no opt-in
   needed) and is pure-additive in the repair turn, sequence C4 EARLY (right after
   C3) -- it is the broadest, lowest-risk win and reduces the retry tax across the
   board while the per-schema alias work proceeds.

[ASSUMPTION] `_otr_radio_editor.py`'s 111 sites may be one tight loop over a few
schemas, not 111 distinct schemas -- verify the real schema count before scoping
its deferred migration.
