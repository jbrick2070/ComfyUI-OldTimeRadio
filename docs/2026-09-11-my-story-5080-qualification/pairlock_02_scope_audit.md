# Pairlock 02: cleanup scope authorization observation

Read-only audit of code `bd8141487f2bbc093451cd7573cb3b338874b8e4` during
the continuing canonical run. No production edits, test/model calls, or
interruptions were made. This receipt does not assert a terminal run outcome.

The shared cleanup scope-authorizer uses prompt-plus-validation schema handling;
its caller never requests the available native schema binder. This is not
evidence that LMFE admitted a nullable field under a bound schema.

## Live evidence

Source: `tmp/my_story_5080_eos_server.log`, PID 49032. One observed authorization
returned at `2026-09-11T09:25:59.627588+00:00`; its typed repair returned at
`2026-09-11T09:26:02.824203+00:00`. Both reported 46 generated tokens,
last token 248046, effective EOS IDs `[248044, 248046]`, and `ended_with_eos=True`.
Both returned:

```json
{"verdict":"already_spoken","spans":null,"reason":"The line is valid spoken dialogue within the specified scene context and matches the original complaint."}
```

Pydantic rejected `spans` with `list_type`, input `None`. The helper exhausted
its existing two attempts (base 0.200, typed repair 0.100). Subsequent log entries
show continued cleanup and `my_story_source_rewrite_ledger_clean_spoken`.
EOS termination succeeded on these calls; this is a separate shape/routing defect.

## Code trace

- `nodes/_otr_ledger_clean.py:94-97`: `_ScopeAuthorization.spans` is
  `list[_ComplaintSpan] = Field(default_factory=list)`. Omission or `[]` is valid;
  explicit null is not. There is no nullable union on this field.
- `nodes/OTR_LedgerScriptWriter.py:3413` creates the unbound creative scheduler
  closure. Its available `_otr_bind_schema` capability is exposed at line 785;
  native generation installs LMFE only when `schema_model` is supplied.
- `nodes/_otr_writer_tail.py:1446-1449` passes that creative closure to
  `run_ledger_clean`. `_repair_one` forwards it to `_authorize_repair_scope`
  (`nodes/_otr_ledger_clean.py:2087-2091`).
- `_authorize_repair_scope` passes the unchanged closure to `structured_call`
  with `_ScopeAuthorization` and `max_attempts=2` (lines 1514-1518). No binder is
  called anywhere in this cleanup module.
- `nodes/_otr_structured_call.py:937-939` adds the schema to the prompt.
  `invoke_structured_slot` (lines 614-665) handles remote JSON-object capability
  and otherwise calls the supplied closure. It does not bind native schemas.
  The same supplied closure serves base and typed repair.
- Exhaustion returns `unresolved` (cleanup lines 1520-1524). The row retains its
  original text and receives an unclean flag (lines 2098-2105), rather than
  accepting the model's `already_spoken` verdict or terminating the episode.

## Narrow follow-up after this run

Reuse the scheduler's existing native binder for this exact authorization schema
before entering its two-attempt ladder, and retain the bound callable throughout
that ladder. The smallest existing owner is `_authorize_repair_scope`, after its
deterministic no-call shortcut and before its one `structured_call` invocation.
This applies to the same shared cleanup operation across banks, not just My Story.
Preserve the schema and semantic scope validator; do not make null universally
valid or increase the attempt budget. Existing coverage is
`tests/test_ledger_clean_stage.py`: already-spoken fixtures omit spans or use a
list, and currently do not exercise native binder routing. Add a binder-once /
same-bound-callable retry check and real constrained-generation coverage that
permits omitted/empty spans while excluding null for this schema. This audit
does not establish coverage or correctness of other cleanup helper bindings.

### Existing pattern and blast radius

`nodes/_otr_my_story.py:489-490` already selects
`bind_schema(StoryTreatment) if callable(bind_schema) else creative_fn` once for
its pass. `nodes/_otr_story_source.py:125-127` uses the same capability pattern
for source corrections. There is no general schema-binding utility in the
current structured-call module; these are explicit caller decisions.

Do not silently add automatic binding to `structured_call` or
`invoke_structured_slot` as part of this occurrence. That changes every native
structured caller, including owners with text-parser formats, custom repair
factories and alternate repair slots, and expands qualification well beyond the
live scope-authorizer defect. `invoke_structured_slot` does not receive a schema
argument at all. No new abstraction is needed for the two-line capability choice
at the affected shared owner.

When the capability is absent, preserve the original slot callable and its
markers. The existing invoker requests JSON-object mode from OpenRouter/native
GGUF (writer capability declarations at lines 713-727); those paths retain
prompt schema plus Pydantic/post-validation and the same bounded repairs.
Comfy Credits, Google and ordinary test callables retain their current routing.
Do not send an unconditional JSON-schema keyword to transports that have not
advertised it. A binder that exists but fails should propagate that real failure,
not fall back silently to unconstrained generation.

Useful existing regression owners:

- `tests/test_ledger_clean_stage.py`: exercise scope authorization through the
  public cleanup entry; assert one bind for `_ScopeAuthorization`, the same
  bound callable on typed repair, unchanged call budget, and the original text
  remaining unflagged when the valid verdict is `already_spoken`.
- `tests/test_my_story_runner.py`: existing treatment binder-once/retry pattern.
- `tests/test_writer_slot_routing.py`: existing
  `test_scheduler_local_schema_binding_reaches_truncating_generator` proves the
  scheduler forwards the actual schema to the native generator.
- `tests/test_constrained_generate.py`: real installed LMFE schema controls;
  use this authorization schema to test array/omission versus null and preserve
  per-call grammar histories and aligned EOS IDs.
- Add a no-binder control at the cleanup owner, including the existing
  OpenRouter/GGUF JSON-object marker behavior and a binder-error propagation
  control. These do not claim live remote/GGUF qualification.
