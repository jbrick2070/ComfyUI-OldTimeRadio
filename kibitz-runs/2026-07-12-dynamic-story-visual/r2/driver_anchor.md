# r2 driver anchor -- dynamic-story-visual (CODING PLAN)

Written BEFORE the r2 fan-out. Author: Claude (Cowork), docs-only architecture
owner + sole judge. Panel: codex `gpt-5.6-sol` @ ultra, antigravity
`gemini-3.5-pro`.

Scope of r2: the CODING PLAN for the doc under review
(`docs/2026-07-12-dynamic-story-visual-scope.md`, rev 2). Not the arc (settled
in r1), not the wiring (r3). Every claim below is grounded `file:line` against
the real Windows repo (file tools / Desktop Commander, never the Linux mount).

Extra mandatory review input this round: `docs/PRODUCTION_SPRINT_LESSONS.md`.
The doc under review is GRADED against it; an unmet lesson is a finding.

---

## A. Lessons scorecard (rev 2 of the doc)

| Lesson | Verdict | Where |
|---|---|---|
| 1 ownership before implementation | PARTIAL | section 3 names actors + writer/consumers, but NO authored/derived/measured column and NO exact nested-row field list for `shots[]` rows or `field_evidence` entries. -> M9 |
| 2 five representations in lockstep | FAIL | doc specifies schema + validator. Base prompt, worked fixture, repair prompt: ABSENT. -> M3 |
| 3 authored vs mechanical repair | FAIL | not addressed at all. -> M2 |
| 4 retry ladder by failure class | FAIL | doc says "bounded attempts (2, matching max_reseed)". The repo already HAS a 4-rung typed ladder the doc does not use. -> M2 |
| 5 size context from the real artifact | FAIL | D8 says "confirm numbers" with no equation, no context cap, no measurement of the real drivers. And the seam the doc points at has NO context guard at all. -> M1, M4 |
| 6 model diversity, not repeatability | FAIL | section 8 has no qualification ladder; no second local family, no cloud creative lane. -> M6 |
| 7 prove the real workflow | PARTIAL (r3 owns it) | section 7 item 2 has the three-link shape but registration is under-specified. -> M7 |
| 8 clean GPU experiments | PASS | section 8 live smoke cites CLAUDE.md section 4 reset; one variable per leg. |
| 9 live failures -> PROD_BUG_LOG | FAIL | no PROD_BUG_LOG expectation anywhere. -> M8 |
| 10 concurrency boundaries | PASS | doc is docs-only, Codex owns code; stated in the header. |
| Sprint receipt | FAIL | section 8 does not end with the receipt fields. -> M10 |

---

## B. Must-fix findings

### M1 -- The LLM seam the design points at has NO context guard. (Lesson 5)

The doc (4.2 step 3) routes the derivation through
"`request_slot`/`make_generate_fn` + `policy_from_meta`
(nodes/otr_shot_lock.py:677-694)". Grounded: that path is UNGUARDED.

- `_otr_model_loader.make_generate_fn` local lane
  (nodes/_otr_model_loader.py:1108-1137) applies the chat template, tokenizes,
  and calls `model.generate` -- there is NO `max_input_tokens` computation, NO
  truncation warning, and NO honoring of a must-fit marker. `context_cap` is on
  the cache entry and is simply IGNORED on this path.
- The guard exists ONLY in the writer's own slot wrapper
  (nodes/OTR_LedgerScriptWriter.py:664-699):
  `max_input_tokens = max(64, context_cap - int(max_new_tokens))` (:681), then
  either `raise PromptContextOverflowError(... "refusing to left-truncate an
  unsliceable provenance prompt")` when the messages object carries the must-fit
  marker (:684-690), or a silent-ish `PROMPT_GUARD` LEFT-TRUNCATION (:691-699).
- The marker: `class _PromptMustFitMessages(list[dict[str, str]])` with
  `_otr_prompt_must_fit = True` (nodes/_otr_scifi_codex.py:308-311), read back at
  nodes/OTR_LedgerScriptWriter.py:667.

Consequence for this feature: a `visual_direction` prompt is the most
provenance-sensitive prompt in the repo (frozen story + brief + safety base +
schema + fixture). On the ShotLock idiom it would be silently over-run or
left-truncated -- losing the system/schema prefix FIRST, because truncation is
from the LEFT. That is not hypothetical: PROMPT_GUARD truncation of a typed
repair (4751 -> 4592 tokens) is the logged root-cause chain of
docs/PROD_BUG_LOG.md PBUG-20260712-03.

REQUIRED IN THE DOC: the direction node does NOT use the raw
`make_generate_fn` return value as its `slot_fn`. It must wrap it in a
must-fit-capable slot_fn that (a) reads `cache_entry["context_cap"]`,
(b) measures the tokenized input, (c) RAISES on overflow -- never truncates.
Codex's implementation choice: reuse the writer's wrapper by lifting it to a
shared helper, or write the equivalent guard in the direction module. The
DESIGN requirement is fail-loud, not the location.

### M2 -- The typed-repair ladder already exists. Do not invent "2 attempts". (Lessons 3, 4)

Doc 4.2 step 3: "bounded attempts (2, matching `max_reseed`,
nodes/otr_shot_lock.py:507); exhaustion fails closed." That is the wrong
mechanism -- `max_reseed` is ShotLock's beat-batch reseed, not a typed-repair
ladder. The repo's single structured-JSON entrypoint is:

`structured_call(*, prompt, schema, slot_fn, base_temperature,
structural_retry_temperature, repair_prompt_factory=None, post_validator=None,
max_new_tokens=..., max_attempts=..., helper_name=...)`
(nodes/_otr_structured_call.py:551).

Its rungs, exactly (this IS Lesson 4, already implemented):

1. base attempt at `base_temperature` (:668-689);
2. structural retry -- SAME prompt, LOWER temperature, and ONLY on
   `json.JSONDecodeError` (:700-721). A `ValidationError` /
   `PostValidationError` deliberately SKIPS this rung (:691-699: a re-prompt
   re-emits the same bad shape and burns a billed call);
3. typed repair at a static low temp `_REPAIR_TEMPERATURE = 0.10` (:83,
   :724-775) -- the factory receives the original prompt, the failed raw output
   (echoed truncated), and the exception (nodes/_otr_repair_prompts.py:128-152);
4. repair-syntax retry -- re-sends the EXACT cached repair prompt once if the
   repair itself emitted undecodable JSON and budget remains (:783-811, floor
   `_REPAIR_SYNTAX_RETRY_FLOOR = 0.25` at :89).

`_DEFAULT_MAX_ATTEMPTS = 3` (:69). Entry invariant, fails loud:
`structural_retry_temperature` must be strictly LOWER than `base_temperature`
(:640-648). Exhaustion raises `StructuredCallFailedError` (:97, raised
:819-823) carrying helper_name/attempts/last_error -- it never returns a
sentinel. Existing typed factories + dispatcher:
nodes/_otr_repair_prompts.py:164 (`json_syntax_repair`), :184
(`schema_field_repair`), :204, :231, :250, :271, :290, :321, and
`make_dispatching_repair_factory` at :402.

Lesson 3's dividing line maps CLEANLY onto this: `post_validator` (typed
`PostValidationError`, nodes/_otr_structured_call.py:128, raised :435-438) is
where deterministic CONTENT checks live (evidence-ID resolution, geometry
lint, authored-field-vs-safety-base collision, 240-char caps, beat_id
membership). A factory may also return a finished schema instance =
deterministic repair with NO LLM call (:750-761) -- correct for the mechanical
class only (dropping an unknown key, canonicalizing an evidence ID's case),
never for authored taste.

REQUIRED IN THE DOC: replace "bounded attempts (2)" with the `structured_call`
ladder + the vd-1-specific rungs, and state per failure class which rung runs:

| Failure class | Rung | Deterministic repair allowed? |
|---|---|---|
| undecodable JSON | structural retry (same prompt, lower temp) | no |
| schema/field shape (incl. wrong-depth nesting) | typed repair (`schema_field_repair`-style) | ONLY the unambiguous relocation case (PBUG-20260712-02/-03 precedent) |
| unresolvable evidence ID | typed repair, naming the invariant + the owning field pointer | no |
| geometry/forbidden term in an authored field | typed repair, naming the term + the field | no |
| authored_fields names a safety-base field | typed repair, naming the whitelist | strip is NOT allowed (it would silently accept an LLM write where it must not) |
| ladder exhausted | `StructuredCallFailedError` -> named domain error -> episode aborts | fail closed |

### M3 -- Only 2 of the 5 representations exist. (Lesson 2)

The doc specifies (2) typed schema and (4) validator. Missing:

1. **Base prompt.** Repo law: content lives in JSON packs; Python owns behavior
   (nodes/_otr_story_pack.py:1-9). BUT the story-pack seam is PER-SOURCE-BANK
   (`PRODUCTION_SEAM_ALLOWLIST`, :27-44; unknown seam = `UnknownSeamError`,
   :146-151) -- 11 packs would each have to author a visual-direction seam. That
   is wrong: visual direction is a FEATURE pass, orthogonal to the source bank.
   RULING: a dedicated prompt module, matching the two existing `*_prompts.py`
   modules (nodes/_otr_period_prompts.py, nodes/_otr_repair_prompts.py) ->
   `nodes/_otr_visual_direction_prompts.py`. NOT a story-pack seam.
3. **Worked fixture.** A golden vd-1 example (tests/fixtures/) that is ALSO
   embedded in the prompt. The proven assembly pattern for exactly this shape is
   nodes/_otr_scifi_codex.py:1156-1160: seam text + `schema_shape_instruction`
   as SYSTEM, a deterministic sorted-key JSON envelope (inputs + the schema
   itself) as USER. `schema_shape_instruction`
   (nodes/_otr_structured_call.py:195) is the fragment weak local models
   actually follow.
5. **Repair prompt.** Per failure class, per the M2 table.

Also required by Lesson 2 and NOT in the doc: the prompt must explicitly FORBID
the known pseudo-shapes -- numbered fields (`era_tail_2`), `_secondary` /
`_tertiary` variants, schema-path strings used as field names, singular-vs-list
aliases (`shot` vs `shots`), and VALID COLLECTIONS NESTED AT THE WRONG DEPTH.

That last class is the single highest-probability live failure of this feature
and it is already logged TWICE, this week, on the second local family:
PBUG-20260712-02 (Gemma nested `causal_steps` inside `caller_threads` rows) and
PBUG-20260712-03 (Gemma nested `shots` inside `scenes` rows) --
docs/PROD_BUG_LOG.md. vd-1 carries THREE nested collections
(`rationale.motifs[]`, `field_evidence{}`, `shots[]`) and a nested dict-of-dicts
(`still_word_typography`). The doc must name the exact top-level ownership of
each and require a deterministic wrong-depth relocation repair (authoritative
top-level wins; lift verbatim only when top-level is absent/empty) exactly as
the two PBUG fixes did.

### M4 -- vd-1 as one pass does not fit the context. Split it. (Lesson 5)

Real numbers, grounded:

- `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`
  (nodes/_otr_model_catalog.py:32) backs BOTH slots by default
  (nodes/OTR_LedgerScriptWriter.py:1192-1193, 2425, 2443).
- Context: `resolve_context_cap` (nodes/_otr_model_catalog.py:1258) clamps to
  `HARD_VRAM_CONTEXT_LIMIT`, default **8192** (:1207-1217, env
  `OTR_HARD_VRAM_CONTEXT_LIMIT`); Mistral-Nemo is CURATED to 8192 (:1226-1234).
- Budget equation (nodes/OTR_LedgerScriptWriter.py:681):
  `max_input_tokens = context_cap - max_new_tokens`.

Output drivers for ONE vd-1 object: 11 authored look strings + 2 dict rows +
`rationale` (thesis + motifs + clue_visual + era_cues + composition_notes) +
one `field_evidence` entry per authored field + **one `shots[]` row per
non-skipped line**. A 420-word episode carries roughly 40-60 lines; at ~40
tokens per shot row that is 1600-2400 output tokens for `shots[]` ALONE, before
the pack. Input drivers: the story projection (every line's text), the brief,
the safety base, the schema, the worked fixture. At 420 words -- let alone the
720-word bake-off target -- input + output does not fit in 8192, and the
7-line 30-word smoke will NOT reveal this (it is a target_words-blind failure,
which is the whole point of Lesson 5).

REQUIRED IN THE DOC: split the derivation into two pass classes bound to the
SAME artifact and the SAME projection hash:

- **P-A (look):** one call. Input = title + brief + cast rows + a BEAT SPINE
  (beat_id + one-line intent), not full line text. Output = the authored look
  whitelist + `rationale` + `field_evidence` for those fields.
- **P-B (shots):** BATCHED over beats, mirroring the existing batching seam
  `derive_creative_directives(..., batch_size: int = 15, ...)`
  (nodes/otr_shot_lock.py:499-508). Input = P-A's authored pack (so notes stay
  coherent with the look) + this batch's line text. Output = `shots[]` rows for
  this batch only.

State the token equation per pass and the must-fit assertion (M1). "<= 64 KB
canonical" is a storage bound, not a context budget -- keep it, but it is not
an answer to Lesson 5.

### M5 -- D2 (slot) is answerable from precedent. Answer: `creative`.

The doc calls it a taste judgment. It is a grounded one. The slot comment
(nodes/OTR_LedgerScriptWriter.py:405-411) reads "technical = structured passes
(GBNF / JSON validators, reviewer verdicts, ...)", which naively argues
`technical` for any JSON pass -- and `_otr_constrained_generate.py:25-27` says
schema-constrained calls route to technical. But fable2 proves the operative
rule is PASS NATURE, not output format: P0 dossier (extraction) runs on
**technical** (nodes/_otr_scifi_fable2.py:1129-1137), while P1 pitch room, P2b
treatment, and P3 whole-play markup -- all authorship, all schema-constrained
JSON through `structured_call` -- run on **creative** (:1166-1174, :1201-1209,
:1394). Visual direction is authorship. `creative` CONFIRMED, with fable2 as
the citation rather than an assertion of taste.

Mechanics the doc must state: the model id comes from
`meta["creative_writing_model"]` (stamped nodes/OTR_LedgerScriptWriter.py:
1421-1422; the read-from-meta idiom is nodes/otr_shot_lock.py:663-665), resolved
fail-loud through `require_model(model_id, slot="creative")`
(nodes/_otr_model_inputs.py:72). Slot names are gated to exactly
`("creative", "technical")` (nodes/_otr_model_loader.py:821-824).

### M6 -- No model-diversity qualification ladder. (Lesson 6)

Section 8's live smoke is one control leg + one dynamic leg on one model. Both
slots default to Mistral-Nemo. Required ladder, in order:

1. unit fixtures + full Windows suite + Bug Bible;
2. canonical 30-word end-to-end on **two different local families** --
   Mistral-Nemo (`DEFAULT_LLM`) and the second family that is demonstrably in
   production use and demonstrably fails differently:
   `google/gemma-4-E4B-it [LOCAL HF]` (the family behind PBUG-20260712-02 and
   -03) -- plus **one configured cloud/frontier creative lane**
   (`openrouter:slot-a` / `google_api:slot-a`; OpenRouter is key-gated,
   nodes/_otr_openrouter_backend.py:272, so this leg is operator-env dependent
   and must be declared, not silently skipped);
3. the same pairings at 120 words;
4. only then 720-word qualification.

Record per leg: concrete model labels, slot assignment, prompt id, repair-rung
counts, ledger path, asset path.

### M7 -- Registration must be the LITERAL `_NODE_MODULES` dict. (Lesson 7)

Section 7 item 1 says "Registration in the package init's `_NODE_MODULES` /
class + display mappings". Tighten: `__init__.py` has TWO registration paths --
the literal `_NODE_MODULES` dict (:119-325; one tuple entry supplies BOTH the
class mapping and the display name, written by the loader loop at :362-363) and
a merge from `nodes/_otr_class_registry.py` (:335-349). The canonical-workflow
contract test builds its node-class mappings by **AST-parsing the literal
`_NODE_MODULES` dict** (tests/test_workflow_contract_validation.py:41) -- it
does not execute the class-registry merge. A node registered only via the class
registry is therefore INVISIBLE to the canonical-workflow gate and to
tests/test_workflow_graph_integrity_guards.py. `OTR_DynamicStoryDirection` MUST
go in the literal dict. (Display-name convention: leading space + Title Case.)

### M8 -- No PROD_BUG_LOG expectation. (Lesson 9)

Add to section 8: any failure of this feature in a LIVE run (30w/120w/720w
smoke, soak, or published episode) gets an append-only
`PBUG-<YYYYMMDD>-<NN>` entry in docs/PROD_BUG_LOG.md using the template at
:15-26 (surfaced / symptom / root cause / fix / verify idea / bible-worthy /
confidence / status). Dev-only catches are fixed and tested but NOT logged.
Bible promotion happens at the operator-triggered fan-out, not inline.

### M9 -- Section 3's ownership table is incomplete. (Lesson 1)

Add, per Lesson 1: for EVERY vd-1 field -- authored | derived | measured; its
writer; its consumers; its lifecycle boundary (post-freeze extension); its
durable receipt. And the EXACT allowed field list for each nested row type
(`shots[]` row, `field_evidence` entry, `rationale.motifs[]` row), stated as a
closed set, because "do not assume a model infers ownership or nesting from a
JSON schema."

### M10 -- No sprint receipt. (Lessons doc, final section)

Section 8 must end with the receipt template filled in as EXPECTATIONS (what
this feature must produce): scope, authoritative_writers, durable_artifacts,
canonical_workflow_hash, focused_tests, full_suite, bug_bible, model_pairings,
30/120/720-word receipts, live_ledgers, published_assets, prod_bug_entries,
head, origin, remaining_risks.

---

## C. Should-fix

- **S1 -- Do not hand-roll JSON extraction.** Use
  `parse_first_json_object` (nodes/_otr_json.py:81) /
  `extract_first_json_block` (:35) / `parse_validate_tolerant`
  (nodes/_otr_structured_call.py:442). The anti-pattern to avoid is
  `_parse_directives` (nodes/otr_shot_lock.py:429-451), which slices
  `find("{")`/`rfind("}")` and returns `{}` silently on failure.
- **S2 -- Schema must be a pydantic model**, not a hand-written dict validator,
  so it plugs into `structured_call`, `schema_shape_instruction`, and
  constrained generation: `make_constrained_generate_fn`
  (nodes/_otr_constrained_generate.py:161) binds lm-format-enforcer on the local
  HF lane (:262-269) and maps the same schema to `response_format` on remote
  lanes (:207-238). Note this is where the "technical slot" comment lives -- the
  doc should state explicitly that constrained generation is a LANE feature, not
  a slot feature, so a creative-slot structured pass may still use it.
- **S3 -- Kill the run-to-run determinism implication.** The local lane
  hardcodes `do_sample=True` (nodes/_otr_model_loader.py:1124) with no seed, so
  "same story => same direction" is NOT an invariant. Doc test 7 is fine (it
  injects the output), but 6.5's replay language must say: an UNCHANGED STORED
  artifact replays byte-identically; a re-derivation does not.
- **S4 -- D5 (still_word scope): pin to the safety base for v1.** Every nested
  dict the LLM authors is another wrong-depth surface (PBUG-20260712-02/-03) and
  another 200-400 output tokens against an 8192 cap. Typography variety is a
  low-yield surface next to an abort. Reopen post-v1.
- **S5 -- D9 (safety-base packaging): Python constant module.** A JSON file
  under `nodes/visual_styles/` would need a registry-sweep exemption
  (nodes/_otr_visual_styles.py:329-336); a constant module (e.g.
  `nodes/_otr_visual_direction_base.py`) needs none and cannot be picked up as a
  selectable pack by accident.
- **S6 -- D10 (era-tail mechanics): a flag on the resolved VisualStyle**, not a
  style-id string compare scattered through the helper family. One boolean the
  brief-first precedence in `get_era_tail` honors
  (nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428).
- **S7 -- Test-harness facts for section 8:** conftest sets `OTR_TEST_MODE=1`
  (tests/conftest.py:38) and hard-fails the session on ANY new failed nodeid
  (:219-286), so "run the suite after every change" is a real gate; the
  `llm_fn=` injection idiom is nodes/otr_shot_lock.py:499-508 with the fakes in
  tests/test_video_platform_aseam.py:401-500; the merge-survival test to mirror
  is tests/test_ledger_merge_ownership.py.

---

## D. Questions put to the panel

1. Is the P-A / P-B split (M4) the right decomposition, or is there a cheaper
   way to keep ONE pass inside 8192 (e.g. shots[] only for beats that actually
   receive a still)?
2. Does routing the vd-1 pass through `structured_call` with a pydantic schema +
   `post_validator` fully satisfy Lessons 3 and 4, or does the evidence-binding
   step need a rung the existing ladder cannot express?
3. Is there any OTHER unguarded-context call site in the repo (M1) that a new
   post-freeze node would inherit by copying the ShotLock idiom?
4. Grade the doc against PRODUCTION_SPRINT_LESSONS yourself -- report any lesson
   I marked PASS/PARTIAL that you judge FAIL, with file:line.
