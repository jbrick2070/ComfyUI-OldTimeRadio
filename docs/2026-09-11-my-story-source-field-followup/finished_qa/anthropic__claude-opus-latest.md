<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Verdict

No demonstrated must‑fix in the production diff. Two findings I'd call must‑fix *in the tests/Bible*, plus concrete limits that should be stated rather than implied.

## Findings

**F1 (must‑fix, test validity claim). The new `test_spoken_source_alias_repairs_to_an_applied_missing_action_within_two_calls` is not fix‑discriminating, same as the test you already flagged.**
Evidence: attempt 1 sends `source_field='text'`. *Before* the fix (`source_field: StrictStr`) that value validates structurally, then `_apply_spoken_edits` raises `ValueError("A spoken correction must quote an actual original source field")` via `raw.get(edit.source_field, "")` → `validate` returns a string → attempt 1 `status='failed'`, attempt 2 applies. Every assertion in the new test (`len(slot.calls)==2`, `applied`, text, `attempts[0].status=='failed'`, `attempts[1].status=='usable'`, `qualified is False`) therefore holds on the pre‑fix code. It is conservation coverage, not regression coverage. Only `test_spoken_source_grammar_excludes_candidate_field_and_accepts_all_source_keys` and the Bible `pytest.raises(ValidationError)` case discriminate. The Bible prose ("Verify native grammar excludes the invalid alias, all original source keys remain usable, and a corrected replacement applies inside the same two-call budget") reads as if the third clause is new proof; it is not. Fix the claim, or the catalog entry will be cited later as protecting something it never protected.

**F2 (must‑fix, fragility). `test_otr_spoken_source_schema_uses_the_original_field_namespace` silently degrades on a legal refactor.**
It selects only `ast.Assign` nodes whose target name is `CREATIVE_FIELDS`/`CreativeFieldName`. Change `_otr_story_input.py` to the idiomatic `CreativeFieldName: TypeAlias = Literal[...]` (an `ast.AnnAssign`) and `fields` loses that node; the exec then raises `NameError` on the class annotation — or, if only `CREATIVE_FIELDS` is annotated, `namespace["CREATIVE_FIELDS"]` raises `KeyError`. Either way the failure is an unrelated-looking crash, not a legible guard. Also it asserts `len(definitions) == 1` but does not assert the two extracted assignments were found. Minimum: assert both names are present in `namespace` after exec, with a message naming the refactor that broke it.

**F3 (over-engineering). The Bible test is an AST-exec re-implementation of a two-line import.**
`namespace` hand-stubs `Literal, get_args, BaseModel, ConfigDict, StrictStr, StrictInt, Field`. That list must stay in sync with the class body forever: add `field_validator`, `model_validator`, or any module-level constant used as a `Field(...)` default to `SpokenSourceEdit` and this test breaks for reasons unrelated to the namespace invariant it claims to protect. It also duplicates the enum assertion already made in `test_constrained_generate.py`. If pack-import isolation is the reason, say so in the docstring; the current docstring ("exercise the production type against its input authority") does not justify the exec machinery.

**F4 (risk, unverifiable from the diff). The new instruction asserts offset semantics owned by code not shown.**
"Optional start_char/end_char are zero-based Python character offsets in that draft line, with end_char exclusive" is a claim about `_exact_interval` in `_otr_ledger_clean`, which is not in the diff. If `_exact_interval` treats them as inclusive, or as a hint validated against the quote's found span rather than an authoritative slice, the prompt now teaches the model a contract the applier does not honor — a new wrong-answer path where previously the model was simply unguided. This needs one citation of `_exact_interval`'s handling of `start_char`/`end_char`, or the two sentences should be dropped.

**F5 (limit, should be stated). The enum closes one failure mode and opens an adjacent one; it cannot have caused the live05 success.**
`raw = _raw_values(...)` always materializes all four keys, including empty strings, and both `_json({"source": raw, ...})` and the new instruction (`", ".join(CREATIVE_FIELDS)`) advertise all four unconditionally. In live05 only `plot` was populated. Under native LMFE the model can no longer emit `"text"`, but it can now emit `"setting"` with a draft-derived `source_quote`; `_apply_spoken_edits` rejects that identically, one attempt is still burnt, and the remaining repair can still return `edits: []`. So the structural guard converts `source_field=text` into "some valid-but-wrong key"; the thing that must actually fix live05 is the *instruction* text, which has zero live evidence behind it. (Note the asymmetry: `build_raw_documents` and `raw_source_block` both filter on `value.strip()`; the enum and the new instruction do not.) I am not proposing a checker — just that the write-up not imply the enum is the repair.

**F6 (risk, untested). Prompt growth.**
Five sentences were added to every spoken source-check prompt, ahead of `inspect_structured_fit(...)` which can raise `PromptContextOverflowError(phase="prompt_no_room")` for the whole operation. No test in the diff covers a near-capacity spoken prompt. Low probability, but the failure is a hard `unresolved_capacity` on a path that previously fit.

## Nits (not blocking)

- `CREATIVE_FIELDS = get_args(CreativeFieldName)` degrades the static type from `tuple[str, str, str, str]` to `tuple[Any, ...]`. Any existing type checking over `CREATIVE_FIELDS` weakens silently.
- `instruction=(...) % ", ".join(CREATIVE_FIELDS)`: the `%` operator now applies to the entire multi-line concatenation. Any future literal `%` anywhere in those ten lines (e.g. "100% of the draft") is a runtime `ValueError`/`TypeError` at call time, not import time. A named `.format`-free f-string on the single sentence, or `%%`-discipline noted in a comment, removes the trap.
- `assert 200 in _feed_json(prefix, text)` uses a bare magic number where the line above uses `ord('t')`. If 200 is an EOS/terminal marker, say so; otherwise it reads like a typo for a codepoint.
- The Bible entry for the enum fix is appended into the *middle* of a pre-existing paragraph, between its "Coverage:" sentence and its "Visual coverage:" sentence. That splits one bug's coverage list. Cosmetic, but it will read as if the visual coverage belongs to the enum fix.

## On the stated evidence

`38 pass / 10 inherited fail / 11 skip / 3 xfail` vs `36/12` with "the new schema guard and updated coverage catalog fail only baseline" is the right shape of claim and is internally consistent (+2 pass, −2 fail). Full regression still running means the sign‑off is provisional; the 10 inherited failures are unqualified here and I take no position on them. The five semantic failures you enumerate (source_field=text, `edits: []` on attempt 2, dropped girlfriend mention, dropped ending, the extra child in the rendered image) remain unaddressed and are correctly not claimed.