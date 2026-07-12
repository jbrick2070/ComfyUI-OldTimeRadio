# Build an Original OTR Source Bank

> Add one original, runnable source bank to the existing OTR workflow. The
> outer workflow stays canonical. Inside that boundary, invent an independent
> creative pipeline that fills the production ledger and hands it to the shared
> writer tail.

This is an implementation brief for a coding LLM. Before editing, read
`AGENTS.md`, `CLAUDE.md`, and the live contract surfaces named below. Live
code wins if this document is stale.

## 1. Preserve the contract, not the existing designs

Treat existing OTR banks, runners, prompts, docs, tests, and workflows as
integration-contract references only. Do not clone, rename, paraphrase, port,
recombine, or imitate their genre, narrative frame, role system, pass graph,
artifacts, prompts, validators, or dramatic logic. Reuse shared infrastructure
only where the live contract requires it.

Start from an original listener experience, then derive the architecture it
needs. Similarity forced by a shared API is acceptable; discretionary
similarity to an existing lane is a design defect.

Originality is a design requirement, not a runtime score. Do not build an
"originality gate" that asks a model to judge taste.

Ground compatibility against these live surfaces without using existing bank
implementations as templates:

- `nodes/production_ledger.py` and `nodes/_otr_ledger_freeze.py`
- `nodes/_otr_story_routing.py` and `nodes/_otr_story_pack.py`
- `nodes/_otr_source_payload.py`
- the runner dispatch and shared tail in `nodes/OTR_LedgerScriptWriter.py`
- `nodes/story_packs/banks.json` and `nodes/story_packs/pipelines.json`
- `workflows/otr_canonical.json`

## 2. Design the experience freely

The bank owns its source strategy, genre, dramatic form, passes, intermediate
artifacts, roles, authority boundaries, retry topology, and ledger-assembly
method. It may use a source or invent from no source. It may use one model pass
or many. It does not need an announcer. A music-free form must satisfy the
live output behavior described under the ledger contract.

A no-source bank still needs its own truthful initialization path. Do not use
empty `fetcher` plus empty `interpreter` as a generic no-source marker: the
current writer reserves that shape for the `original_radio` architecture.
Add explicit bank/pipeline handling or a registered bank-specific local seed
path instead of inheriting that design accidentally.

The result must still work as audio. Some voice must orient the listener and
provide closure, but the device and placement are yours. If the canonical
announcer role is used, it frames the program rather than joining character
dialogue.

Keep the content SFW: no guns, blood, violence, or swearing. Treat that as a
creative constraint, not a post-generation censor.

## 3. Keep the three routing coordinates distinct

- `source_bank_id` selects the source policy and bank defaults.
- `story_model_id` selects the prompt pack at
  `nodes/story_packs/<source_bank_id>/<story_model_id>.json`.
- `story_pipeline_id` selects the pass graph and execution runner.

The bank row, pack header, pipeline row, and runner dispatch must agree on
those coordinates. JSON alone does not create an execution lane.

Custom seam names belong in the pipeline's `declared_seams` and pass rows,
and the pack must supply them. A bank's `required_seams` field accepts only
the live shared production-seam allowlist.

## 4. Use only the two supplied LLM slots

A custom dispatched runner currently receives this interface:

```python
def run_<bank>_episode(
    *,
    payload,
    pack,
    resolved,
    led,
    meta,
    creative_fn,
    technical_fn,
    slot_scheduler,
    source_bank_row,
    story_rules,
    episode_root,
    episode_id,
) -> MyBankTailParts:
    ...
```

The live writer dispatch is the source of truth for the exact signature.
`MyBankTailParts` is lane-defined; there is no shared `TailParts` class.
It must expose `outline_view` with `title` and `premise`, an
`EpisodeCanon`-compatible `canon` with `title`, `premise`, `setting`,
`time_of_day`, and `sound_palette`, plus `final_title_override` and
`run_story_spine`. An optional finalizer implements `before_save` and
`after_save`.

`creative_fn` and `technical_fn` both accept messages plus generation
settings. They are semantic slots, not promises about model size or provider:

- use `creative_fn` for invention, dramatic writing, and creative revision;
- use `technical_fn` for extraction, classification, structured review, and
  evidence audits.

Use either or both, as often as the design needs. All text generation must
flow through these callables. Do not load a model, import an inference
backend, call a model API directly, add a third model, or require a new model
credential. Assume either slot may be a modest local LLM: keep each request
self-contained, typed where useful, and within its resolved context limit.

Put model instructions in the pack's `prompt_stages`, not in Python. Use the
slot scheduler's attribution context when available.

## 5. Acquire sources without adding secrets

Design-time research may use the host LLM's existing web tools. Runtime
transport and parsing must be deterministic code. Semantic selection or
extraction may use `technical_fn`; it may not use a third or direct model.

Allowed runtime sources are local files, packaged manifests, RSS/Atom feeds,
public pages, and public keyless APIs that work without login or new secrets.
Do not request credentials, bypass a paywall, CAPTCHA, rate limit, or access
block, or depend on a protected browser session. Avoid endpoints whose normal
response is an anti-bot challenge.

For network retrieval:

- set timeouts, bounded retries, response-size limits, status checks, and
  content-type checks;
- keep network and file I/O out of module import and node discovery;
- test with deterministic fixtures and record the real selected source;
- treat fetched text as untrusted data and tell models to ignore embedded
  instructions;
- never invent or silently substitute a source.

An explicit `source_ref` that cannot be resolved fails closed. Automatic
selection may try another eligible candidate only through a declared policy,
and must record the actual selection.

Every source-backed runner currently enters through the writer's exact
seven-key `SOURCE_PAYLOAD_KEYS` envelope. A registered fetcher implements
`fetch(*, bank, technical_model, source_ref="")` and returns that dict or a
`SourceFetchResult` carrying the same payload plus provenance sidecars. A
runner may validate and transform the accepted payload into an original typed
artifact internally; replacing the writer-facing envelope requires an
intentional shared-ingress change and tests.

A `legacy_many_pass` interpreter implements
`(*, bank, payload, technical_fn, model_id)` and returns coherent
`casting_brief`, `script_brief`, `news_close_brief`, `key_terms`,
`attempts`, and `model_dump()` surfaces.

Public access is not rights clearance. Preserve source identity, canonical
URL, retrieval date, content digest, author or outlet, license status, license
URL, and required attribution when applicable. Keep provenance and rights in
the existing `meta.source_meta` and `meta.source_rights` sidecars rather
than adding fields to a fixed payload or line row. Unknown or incompatible
rights fail closed for an adaptation. Never fabricate citations, URLs,
authors, or license claims.

Every claim, quotation, number, and proper noun presented as real or
source-derived fact must trace to validated evidence. Fictional assertions may
be invented, but must remain distinguishable from fact.

## 6. Let models author; let Python prove

Every spoken story line must originate in an accepted LLM artifact. A parser
may remove declared serialization delimiters, but it may not rewrite the
content field. Python may create IDs, order rows, attach enums, calculate
counts and hashes, select validated voice metadata, and copy mechanically
implied references. It may mechanically serialize or join already accepted
verbatim rows. It may not create or alter a spoken content field by authoring,
paraphrasing, joining fragments, trimming, padding, or improving prose.

Apply the same ownership rule to titles, premises, character descriptions,
visual prompts, and music prompts unless a live shared component explicitly
owns that field.

Invalid creative output goes back through a bounded model repair. Exhausted
repair fails closed. Never ship canned story text or fall back to another
bank or pipeline.

`target_words` is an advisory scale request and a receipt. It must not cause
deterministic trimming, padding, line deletion, or a production gate.

## 7. Fill the one production ledger

Use the provided `Ledger`; never create a parallel ledger. The table below
lists the minimum authored inputs, not the full normalized row schemas:

| table | minimum authored inputs |
|---|---|
| `cast` | `char_id`, `name`, `character_description`, `gender`; plus `tts_model` and `voice_preset` when the lane owns casting |
| `scenes` | `scene_id`, `description`, `env` |
| `shots` | `shot_id`, `scene_id`, `description`, `visual_prompt` |
| `beats` | `beat_id`, `shot_id`, `scene_id`, `speaker`, `char_id`, `line_ids` |
| `lines` | `line_id`, `shot_id`, `beat_id`, `char_id`, `speaker_role`, `text`; `boundary` for voiced rows |
| `music` | `cue_id`, `description`, `generation_prompt`, `placement`; `anchor_line_id` when used |

`music=[]` is schema-valid but does not currently request silence; the
canonical theme node interprets it as the legacy cue path. `clips` is
downstream-owned, but the initialized top-level `clips` list must remain
present. All required top-level collections must be lists, never null.

Prove the complete graph in a bank-owned test:

- IDs are non-empty and unique within each table.
- Every scene-owned shot resolves to one scene.
- Every scene-owned beat resolves to a shot and its `scene_id` agrees with
  that shot.
- Every member of a beat's `line_ids` resolves to exactly one line.
- Every voiced scene line resolves back to exactly one beat and shot.
- Every character line and its beat resolve to the same cast identity.
- Any bookend, frame, or music sentinel outside the scene graph follows a
  declared shared-tail contract and has a focused test.
- Every optional music `anchor_line_id` resolves when present.

Allowed `speaker_role` values are `character`, `announcer`,
`music_open`, `music_close`, and `music_inter`. Use
`char_id="announcer"` as the new-lane convention unless the live shared cast
contract assigns another non-empty ID. A non-skipped voiced row needs
non-empty canonical `text`. A music sentinel may carry empty text without
being skipped; any row explicitly marked `skip=True` needs empty text and a
`tts_skip_reason`. A voiced line's `boundary` is `shot_start`,
`beat_start`, or `continue`, consistent with its transition. Spoken text
contains no speaker label, stage direction, or whole-line quotation wrapper.

Freeze policy is selected from the pack: a non-empty
`line_composer_system` seam selects `legacy_full`; its absence selects
`content_owned_readonly`. Choose and test that seam deliberately. A
content-owned runner assigns real character voice metadata and proof receipts;
the shared writer tail must then stamp fresh `text_for_tts` and its
canonical-text source hash after final text mutations. A legacy lane uses the
shared CastLock and readiness path.

Keep evidence maps, authorship hashes, and lane receipts in typed artifacts or
namespaced `meta`, not in fixed line rows. Use the Ledger setters and shared
count-stamping helpers.

## 8. Make validation repairable

For structured passes, keep each prompt seam, worked example, typed schema,
parser, and repair prompt in exact agreement. Use a bounded ladder such as
base call, structural retry, then typed repair. Every rejection must name the
offending item, evidence, reason, and permitted correction.

Every collection the model fills must declare the concrete shape of one item.
Do not use `list[dict[...]]`, `dict[str, Any]`, or `Any` for a
model-authored collection of things. A true identifier-keyed mapping is valid
because its keys define the organization. Pin the item structure; leave
descriptive vocabulary open unless a closed enum is a real downstream
contract.

Calculate prompt fit from the resolved per-slot context cap. Check the base,
retry, and repair forms; the repair request is usually largest. Provenance
passes must fail loudly rather than silently left-truncate.

Never ask a model to calculate, report, or enforce exact counts. Python
measures words, lines, items, and coverage deterministically. If a measured
count needs correction, give the creative slot the measured defect and request
a bounded rewrite. A model-produced count field cannot gate production, and
an unused count field does not belong in the schema.

A production gate may block only when the finding is:

1. objectively checkable;
2. repairable by the party being asked to repair it; and
3. a real contract defect rather than taste, a warning, or a role doing its
   licensed job.

Model audits propose findings; deterministic code corroborates them. A
creative correction returns to the creative slot. Warnings and taste notes
are recorded without becoming hidden fatal gates.

## 9. Integrate into the canonical workflow

A runnable custom lane needs a validated pack, bank registry row, pipeline
registry row, execution runner, explicit `_RUNNER_BY_PIPELINE` dispatch,
and tests in the same change. Register every required fetcher and interpreter.
Set `runnable=true` only when the lane exists. A custom non-source-contract
pipeline must also set `executable=true` when its runner lands.

There is one outer workflow: `workflows/otr_canonical.json`. Do not create a
copy, generated substitute, or parallel ComfyUI graph. A registry-only bank
normally appears through the existing `source_bank` selector and may require
no workflow JSON edit; prove that path, and do not change the shipped
`science_news` default merely to expose the new choice. If a node, widget,
input, link, or default changes, update the canonical JSON in the same change
and run the workflow, link, input-name, and positional-widget audits. Append
new optional widgets at the end.

## 10. Finish with evidence

After design, build, and wiring are complete, execute
`docs/SOURCE_BANK_PREFLIGHT.md`. It includes the full Windows regression
suite, Bug Bible regression, canonical validation, and a live 30-word run. The
live run must save a valid ledger, pass the freeze path, publish to `otr/obs`,
and leave a real asset that is verified on disk.

A bank is complete only when its original design, source provenance, two-slot
execution, ledger graph, runner/tail handoff, canonical integration, tests,
and live asset all have evidence.
# Lessons to bake into the next source-bank coding plan

The implementation plan should include these details explicitly, rather than
leaving them for live integration:

1. Provide the exact repository-valid JSON schema for the story pack, story
   rules, bank row, pipeline row, and every sidecar.
2. Define every Pydantic model field, enum, bound, and nested item type, plus one
   known-valid JSON example for every model-authored artifact.
3. Put every cross-artifact invariant inside the originating
   `structured_call(..., post_validator=...)`. In this repository the validator
   must return an error string (or `None`), not raise; raising bypasses the
   structured retry catcher. A check performed after the call likewise bypasses
   bounded typed repair and turns a repairable mismatch into an immediate
   episode failure.
4. State deterministic retention checks for immutable ingress. Prompt prose
   alone does not prove that a possibility preserved every drawn object,
   acoustic cause, or required ending. Put retained constraints in explicit
   typed fields and validate those fields exactly; do not infer retention by
   searching free prose for phrases or synonyms.
5. Specify exact context/output reservations for base, syntax retry, and the
   largest typed repair for every pass.
6. Name the concrete live voice-selection function and returned row contract;
   "use the voice registry" is not sufficiently implementable.
7. Pin registry ordering tests and all repo-specific static annotations, such as
   the literal `# LLM slot: per-sub-pass` audit tag.
8. Define `custom_premise` precedence in the shared input resolver. For a local
   synthetic fetcher it must remain an operator hint and must not bypass the
   immutable draw.
9. Identify the exact final writer mutation boundary and the point after the
   last shared LLM call where telemetry is truthfully stamped.
10. Say which lane-local provenance may remain after a shared generic receipt
    migration, and make clear that it is evidence rather than a second source
    of shared truth.
11. Assign each invariant to the artifact that owns it. Do not require an
    exploratory slate card to satisfy downstream cast/thread completeness when
    a later truth-map or manifest artifact is the actual owner; premature
    minima consume repairs without improving the final contract.
12. Separate reviewer opinion from corroborated defects in the schema and
    blocking logic. A model marking a global note `blocking=true` is not enough;
    fatal findings need the concrete coordinates the deterministic validator can
    verify. Missing coordinates make the observation a warning, not a hidden
    taste gate.
13. Decide identifier ownership explicitly. When Python owns structural IDs,
    normalize harmless integer/string representations at the typed boundary;
    do not spend a creative retry on formatting that carries no authored
    meaning. Repair prompts for collection artifacts must also say never to
    delete items while repairing an ID or field type.
14. For Python-owned enums, define the normalization boundary and then validate
    the semantic roster separately. Free-form labels such as `resident` may map
    to `caller`, but the artifact must still prove one distinct announcer and
    one distinct desk operator before it can advance.
15. Include harmless authored metadata fields that models naturally place on
    adjacent graph objects (for example an optional shot environment) when the
    field is typed and provenance-safe. Otherwise a repair may fix a required
    parent field only to fail because it preserved the same detail on a child.
16. When an artifact has authoritative top-level landmark IDs, models may also
    emit optional per-row marker booleans. Type those markers and validate every
    asserted marker against the top-level authority; do not let redundant
    `null` markers cause a shape-only failure or become a second truth.
17. Enumerate common semantic aliases for every Python-owned manifest enum and
    normalize them at the typed boundary. Values such as `desk_operator`,
    `shot_end`, and `resolution` carry no conflicting authored meaning when the
    canonical downstream values are `character`, `continue`, and `closing`;
    unknown values must still fail loudly.
18. Put the final script's graph, roster, and text-safety checks inside the P6
    (and retake) `post_validator`, not only after generation. Forbidden terms,
    speaker-label leakage, and line-coverage drift are authored defects that a
    bounded typed repair can fix; keep a deterministic assertion after the call
    as defense in depth.
19. Apply safety rules at the earliest creative artifact that authors a detail,
    before later passes treat it as immutable. A forbidden title or clue first
    invented in a slate must repair in the slate/truth/score ladder; waiting
    until P6 can create an impossible repair where the script is required to
    preserve the same now-locked detail.
20. Include standard dramatic-taxonomy aliases and semantic marker values in
    manifest normalization. `exposition`, `rising_action`, and `climax` map to
    the canonical arc; a marker such as `closure: "final"` means true. Reject
    unknown labels after normalization.
21. Re-audit every deterministic/model ownership label in the locked DAG before
    live smoke. If a stage such as ingress or manifest compilation is declared
    Python-owned, do not leave an LLM seam behind and then harden its vocabulary
    with an expanding alias table. Put authored meaning in the preceding typed
    artifact and compile structural IDs/enums locally.
22. Treat information boundaries as wiring contracts. A blind-listener pass
    must receive only accepted pre-reveal lines, never the full script or
    landmark IDs; technical blocking findings must resolve to real artifact
    coordinates and exact evidence before they gain repair or kill authority.
23. Make typed repairs preserve nested invariants, not only top-level counts.
    When one scalar is wrong (for example, every interpretation is marked
    true), tell the model to change that scalar on an existing row. Do not let
    it invent a partial replacement row with an empty required ID collection.
24. Put exact literal enums and their landmark mapping in the base prompt and
    typed repair prompt. If a small model writes semantic synonyms such as
    `orientation` or `closure` where the schema owns `opening` or `closing`,
    reject them; do not grow an alias table that hides contract drift.
25. A repair prompt must pin every collection cardinality the model could
    disturb and forbid schema-path prose from becoming a literal JSON key.
    Correcting `scenes[*].env` is not permission to delete beats or emit a
    top-level field named `scenes[*].env`.
26. When a small model repeatedly ignores a bounded range, choose one valid
    exact cardinality in the authored contract. Spell out nested object keys
    and plurality too: `clue_ids` is an array (possibly empty), never a
    singular `clue_id` scalar.
27. For small technical audit passes, describe the output envelope before the
    audit task. A field named `accepted` must be explicitly boolean or a model
    may copy the accepted input artifact into it; always pin list fields and
    every required finding member in both base and repair prompts.
28. Downstream role gates must use canonical IDs before display names. An
    announcer may have an authored on-air name while retaining
    `char_id="announcer"`; Kokoro/Bark routing and freeze checks must recognize
    the ID instead of requiring the visible name `ANNOUNCER`.
29. Canonicalize purely structural integer IDs to strings consistently across
    a typed graph so a repair call is reserved for semantic/graph defects.
    State contiguity operationally: after leaving a shot ID, never return to it.
30. Make configured retry budgets reachable. When a semantic base failure
    skips the same-prompt structural rung, cache the typed-repair prompt and use
    the remaining call only if that repair itself returns incomplete JSON.
31. Repair duplicate graph references deterministically when the complete
    expected set is already present: keep the first authored placement and
    remove later duplicates. Missing or unknown references still require a
    semantic repair; never invent their placement in Python.
