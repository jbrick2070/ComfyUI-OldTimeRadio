# NESTED-ALIAS FORK -- schema-adherence Lever 1 (resolution proposal to harden)

A focused design fork inside the model-agnostic schema-adherence sprint. The
converged plan (`docs/2026-06-25-schema-adherence/roundtable/pass04_plan.md`,
4 rounds) made an assumption that grounding against the real code contradicts.
This doc states the contradiction + three candidate resolutions and asks the
panel to converge on the mechanism. Ground every claim against the excerpts
appended below (the REAL `_otr_structured_call.py` + a verbatim excerpt of
`_otr_radio_editor.py`).

## Context

- Goal: structured-JSON parsing must be model-agnostic (a user may run local
  mistral-nemo, local Ollama gemma, or any remote OpenRouter model -- GPT, Opus,
  Grok, DeepSeek, Gemini). The operator does NOT control the writer model and
  will not force a local-vs-remote choice.
- Proven failure (2026-06-25): a Claude Opus writer tripped the `normalize_length`
  structured pass. pydantic error:
  `Field required [type=missing, input_value={'index': 14, 'lever': 'S...', 'beat_index': 14}, input_type=dict]`
  -> the 3-rung retry ladder EXHAUSTED (Opus kept its own field names) -> the
  length pass soft-failed + burned ~90k tokens.
- pass04 Lever 1 = TOLERANCE, C0-C6, STRICT-FIRST (tolerance fires only in the
  `except ValidationError` arm so the local happy path stays byte-identical).
  The binary lane (Lever 2) reuses the Lever-1 core `validate_tolerant_data`.

## The grounding finding (the contradiction)

pass04 C0/C1 assumed the `normalize_length` schema is a FLAT top-level pydantic
model in `_otr_story_spine.py`, and specified `_normalize_field_keys` as
TOP-LEVEL-ONLY: `skip unless len(loc)==1 and loc[0] in schema.model_fields`
(nested-key coercion was an explicit CUT). The reality:

1. The pass validates `schema=RadioEditPlan` (defined in `_otr_radio_editor.py`,
   NOT story_spine). `RadioEditPlan.edits: List[BeatEdit]` -- the field variance
   is on NESTED `BeatEdit` objects.
2. The Opus object `{'index':14,'lever':'S...','beat_index':14}` is one
   `BeatEdit`. `beat_index` is present (14); the MISSING required field is
   `action` (Opus emitted the action value under the key `lever`). pydantic loc
   = `('edits', N, 'action')`, len == 3.
3. Therefore pass04's top-level-only `_normalize_field_keys`, run on
   RadioEditPlan, SKIPS this nested error -> it does NOT fix the one proven
   failure. (For the top-level field `projected_word_total` it would work; for
   the nested `action` it cannot.)
4. `BeatEdit` ALREADY ships a `@model_validator(mode="before")`
   `_accept_field_aliases` (BUG-LOCAL-303, docstring names claude-opus) that
   remaps `index`->`beat_index` and `merge_with`->`merge_with_index`,
   byte-identical when the canonical key is already present. The new failure
   (`lever`->`action`) is the SAME bug class, one field deeper, in the SAME
   schema.

## The fork

How should Lever 1 deterministically fix the NESTED proven failure while
preserving strict-first byte-identity and keeping the shared core that the
binary lane reuses?

### Candidate A (RECOMMENDED) -- generalize the existing before-validator; keep ALL of pass04

- C0: `__otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]]` on the
  proven-failure schema = `BeatEdit` (canonical_field -> synonyms; the keys are
  BeatEdit's OWN top-level fields). v1 mapping (verify against the excerpt):
  `{"beat_index": ("index",), "merge_with_index": ("merge_with",), "action": ("lever",)}`.
- A SHARED reusable `@model_validator(mode="before")` (a tiny mixin base, OR a
  class decorator, OR a shared classmethod referenced by each model) reads
  `cls.__otr_field_aliases__` and remaps that model's OWN top-level keys:
  for each canonical->synonyms, if the canonical key is absent and exactly one
  synonym key is present, move it; explicit canonical always wins; no-op when
  the canonical key is present -> byte-identical on canonical input. Pydantic's
  natural recursion applies it to nested BeatEdits during RadioEditPlan
  validation. This REPLACES BeatEdit's bespoke `_accept_field_aliases` (same
  behavior + the new `action: ("lever",)` entry).
- KEEP all of pass04's core unchanged: strict-first; the TOP-LEVEL
  `_normalize_field_keys` in the `except ValidationError` arm (for top-level
  drift on OTHER schemas); the C2 coerce-then-revalidate loop; `_clamp_overlong_
  strings`; `validate_tolerant_data` / `parse_validate_tolerant` (C5); C3 ladder
  narrowing; C4 call-site typed repair. ONE source of truth: the same
  `__otr_field_aliases__` map is read by BOTH the before-validator (nested /
  during-validate) AND `_normalize_field_keys` (top-level / post-failure).
- Fail-loud preserved: an alias only RENAMES a present value. A genuinely
  missing `action` (no `lever`, no `action`) still fails; the downstream Guard1
  (`post_validate_plan`) rejects an out-of-set action value loudly.

### Candidate B -- build pass04 literally (top-level-only); rely on C4 repair for the nested case

- Annotate / normalize at the top level only; the nested BeatEdit failure is
  handled ONLY by C4 schema-in-repair (an LLM repair turn carrying the schema
  snippet). Faithful to the converged plan; zero new before-validator surface.
- Cost: the proven failure stays dependent on the SAME repair ladder that
  EXHAUSTED on Opus. C4's schema snippet may help, but the problem statement
  notes strong models keep their own format; the deterministic fix is lost and
  the cost/latency tax persists for opinionated models.

### Candidate C -- extend the core `_normalize_field_keys` to RECURSE into nested locs

- Walk nested `ve.errors()` locs (`('edits', N, 'action')`) and coerce keys
  inside the nested dicts/lists in the except arm; no before-validator. Fixes
  nesting generically.
- Cost: pass04 explicitly CUT nested coercion as fragile -- path-walking
  arbitrary dict/list structures, list-index addressing, partial-failure
  reassembly, deeper blast radius on the shared core. Higher risk than A for the
  same outcome on this schema.

## Invariants the resolution MUST guard (reject any option that breaks one)

1. STRICT-FIRST byte-identity: any input that validates today validates
   byte-identically (the local happy path + `test_audio_byte_identical` stay
   green; canonical-valid output is returned unchanged).
2. Load-bearing fields fail LOUD: never fabricate a missing required value;
   whitelist-exact aliases only; no fuzzy/positional mapping.
3. No circular import in the core: `_otr_structured_call` imports neither the
   writer nor `_otr_repair_prompts` (C4 repair is injected at the call site).
4. Deterministic, offline, model/transport-agnostic. UTF-8 no BOM, SFW.
5. The Lever-1 core (`validate_tolerant_data`) stays reusable by the binary lane
   (Lever 2, a 1-field `Literal["A","B"]` schema through `parse_validate_tolerant`).
6. v1 annotates ONLY the proven-failure schema (no inventing synonyms for
   schemas that have not actually failed).

## Questions for the panel

1. Is Candidate A's mechanism (a shared `mode="before"` validator keyed on
   `__otr_field_aliases__`, applied per-model and reaching nested models via
   pydantic recursion) correct, or does mixing a during-validate before-validator
   with the post-failure `_normalize_field_keys` create a double-handling /
   precedence hazard? If BOTH could fire for a TOP-LEVEL alias, which wins, and
   is that still byte-identical + deterministic?
2. Does replacing BeatEdit's bespoke `_accept_field_aliases` with the shared
   mechanism risk ANY behavior change vs the shipped baseline (the "explicit
   `beat_index` wins over `index`" precedence; `merge_with`; dict-only guard)?
3. Is `action: ("lever",)` a safe whitelist entry on a LOAD-BEARING enum field
   given the capture is truncated (`'S...'`)? Or should `action` stay un-aliased
   and rely on C4 repair + Guard1? Where exactly is the tolerance/fail-loud line
   for a load-bearing field?
4. Mechanism choice -- mixin base class vs class decorator vs a per-schema
   `@model_validator` calling a shared helper: which is least likely to bite
   across pydantic v2 (validator ordering, `mode="before"` inheritance, a future
   217-schema rollout, models that already define other validators)?
5. Any failure mode where the before-validator SILENTLY masks a real error
   (e.g. a model sends BOTH `lever` and `action` with different values; or
   `index` and `beat_index` disagree)? Define the collision rule.
6. What did the top-level-only design get RIGHT that Candidate A might lose
   (e.g. the explicit "tolerance fires only on failure" gate that bounds where
   coercion runs)? Should the before-validator be similarly bounded?

INVARIANTS RECAP (reject a fix that breaks one): canonical-valid byte-identical;
whitelist-exact aliases only; load-bearing fields fail-loud; no circular import;
no forced transport/model; offline-verifiable; UTF-8 no BOM; SFW.
