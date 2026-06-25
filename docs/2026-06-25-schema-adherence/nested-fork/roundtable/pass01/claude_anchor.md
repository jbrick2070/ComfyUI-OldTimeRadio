<!-- Claude grounded anchor review (R2 implementability). Written BEFORE the
fan-out, grounded against the real _otr_structured_call.py + _otr_radio_editor.py. -->

VERDICT: yes-with-fixes. Candidate A is the correct resolution. It is the only
option that DETERMINISTICALLY fixes the proven NESTED failure while preserving
strict-first byte-identity and the shared core the binary lane reuses, and it
generalizes a pattern (`BeatEdit._accept_field_aliases`) that is ALREADY in the
byte-identical baseline. Candidate B leaves the proven failure dependent on the
ladder that already exhausted; Candidate C re-introduces the fragile nested
path-walking pass04 deliberately cut. Ship A; keep C4 repair as the fallback.

MUST-FIX BEFORE BUILD:

1. [Cand A mechanism] Define ONE collision/precedence contract and make the
   before-validator and the core `_normalize_field_keys` share it EXACTLY.
   Rule (CONFIRMED against BeatEdit's shipped behavior): move a synonym to the
   canonical key ONLY when (a) the canonical key is ABSENT and (b) EXACTLY ONE
   declared synonym key is present. If canonical is present -> no-op (explicit
   wins, matches "explicit beat_index wins over index"). If >=2 synonyms present
   -> ambiguous -> leave the field absent (fail-loud), matching pass04 C1's
   ">=2 (collision) -> leave the field failing". Copy-not-mutate the input dict.
   This is deterministic and whitelist-exact.

2. [Cand A / Q1 double-handling] Resolve the before-validator vs
   `_normalize_field_keys` overlap explicitly: EVERY `__otr_field_aliases__`-
   annotated model gets the shared `mode="before"` validator, so a top-level
   alias is remapped DURING the strict-first `model_validate` (no exception, the
   except-arm `_normalize_field_keys` never fires for it). `_normalize_field_keys`
   REMAINS in the except arm as the top-level safety net for the structured_call
   `schema` itself when it is not annotated (and is what the binary lane's
   `validate_tolerant_data` exercises). Because both read the same map and the
   remap is IDEMPOTENT (once canonical is present, re-applying is a no-op), there
   is no conflict: at most one actually moves a key. Document this so the two are
   not seen as redundant. CONFIRMED safe + byte-identical.

3. [Cand A mechanism / Q4] Implement the shared logic as a MODULE-LEVEL helper
   `apply_field_aliases(cls, data) -> data` plus a one-line
   `@model_validator(mode="before") @classmethod def _otr_alias(cls, data):
   return apply_field_aliases(cls, data)` on each annotated model -- NOT a shared
   base class. A `mode="before"` validator inherited from a base interacts badly
   with pydantic v2 MRO + models that already declare their own validators (the
   217-schema rollout will hit models with existing root validators). The
   per-model one-liner is explicit, has no inheritance surprises, and is trivial
   to add incrementally. The helper lives where it creates NO import cycle:
   `_otr_structured_call.py` already imports only stdlib + pydantic, so put
   `apply_field_aliases` there and import it into the schema module.

4. [C0 / Q3] `action: ("lever",)` is a SAFE whitelist entry. `lever` has no other
   meaning in BeatEdit; the action value space is validated by Guard1
   (`post_validate_plan`), which fails LOUD on an out-of-set action. So a wrong
   guess is fail-CLOSED at the post_validator, never silent-wrong. [ASSUMPTION]
   the truncated `lever:'S...'` is an action token (SHORTEN_LINE / SPLIT_LINE are
   in ALL_ACTIONS) -- the Guard1 backstop makes this assumption safe to ship.
   v1 BeatEdit map = `{"beat_index": ("index",), "merge_with_index":
   ("merge_with",), "action": ("lever",)}`.

5. [byte-identity gate] Add a golden test: RadioEditPlan canonical input (correct
   `beat_index`/`action`/`merge_with_index`) validates byte-identically before
   and after the change; AND the alias inputs (`index`, `merge_with`, `lever`)
   each validate to the same instance. Keep `test_audio_byte_identical` green
   (the change is content-parse only; no generated text changes).

SHOULD-FIX:

1. [observability] BeatEdit's current remap is silent. Have `apply_field_aliases`
   emit a single `log.debug` (NOT warning) naming the model + the moved key when
   it actually moves one, so the model-agnostic alias path is observable without
   adding noise on the (now common) alias case. Keep pass04's telemetry-v1 scope
   otherwise (existing `log.warning` coercion lines only).

2. [scope] Do NOT delete `_normalize_field_keys` or fold it into the
   before-validator. The structured_call top-level `schema` (e.g. a future
   un-nested schema, or RadioEditPlan's own top-level `projected_word_total`
   drift) and the binary lane still need the except-arm path.

OPTIONAL / NICE-TO-HAVE:
- A tiny `tests/fixtures/conformance/radio_edit_plan_opus.json` carrying the real
  (reconstructed) Opus object shape `{"edits":[{"index":14,"lever":"SHORTEN_LINE",
  "beat_index":14}], "projected_word_total": ...}` as conformance fixture #1.

CUT THESE (over-engineering):
1. Candidate C (core `_normalize_field_keys` recursing into nested locs). Pydantic
   already recurses into `List[BeatEdit]` and runs BeatEdit's before-validator;
   a per-model validator achieves the nested fix with no path-walking, no
   list-index addressing, no partial-failure reassembly. Strictly less surface
   for the same outcome.
2. A shared mixin/base class for the alias validator (see MUST-FIX 3) -- the
   per-model one-liner is safer across the 217-schema rollout.

[ASSUMPTION] `lever` carries the action value (capture truncated to 'S...').
[ASSUMPTION] both `normalize_length` and `run_radio_editor` entrypoints share
schema=RadioEditPlan (confirmed for the call site read; normalize_length read by
grep -- verify the exact helper at build).
