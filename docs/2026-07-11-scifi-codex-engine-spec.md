# Sci-Fi Codex - CODE-READY v4 -- technical rewrite by Codex GPT-5, creative content unmodified

Status: specification only. This document is the only changed artifact. It proposes a new additive lane; it does not claim that the lane is presently registered or executable.

## 2026-07-21 production amendment -- authoritative over older pass prose

The lane is now implemented as `scifi_news`. P5 is the only pass that emits a
complete `ScriptArtifactV4`. P7 and P9 emit a private typed line-text patch:
`{"replacements":[{"line_id":"l001","replacement_text":"..."}]}`. A
non-null valid finding ID limits ownership to that row; a null finding ID
widens ownership to all voiced rows; an invented non-null ID is discarded as
reviewer noise. The merge changes only `line.text`, requires exact target
coverage and a real change, then runs the complete script validator.

Each successful P7 patch returns to a fresh P6 judgment and each successful P9
patch returns to a fresh P8 audit. A malformed creative patch gets one colder
technical-slot attempt. If both fail, or if the complete bounded patch cannot
fit the real model context/provider output cap, the best already-valid script
is the quality floor and ledger assembly continues. The loop never rejudges an
unchanged script after a failed patch. P6/P8 model or transport failure is also
advisory and cannot kill an already ledger-valid story.

Patch messages carry an opt-in full-output-capacity marker through every local
and remote transport. Unmarked callers retain ordinary output clamping. Final
authorship receipts, text hashes, ledger rows, media consumers, readiness, and
OBS publication are built only after the final accepted/floored artifact, so a
quality patch cannot leave a stale seal. No canonical workflow wiring changes
are required for this amendment.

## Historical v3 revision log — superseded by v4

- **v3 — technical-only code-ready rewrite:** Re-verified registry, writer, ledger, freeze, and structured-call contracts. Corrected the stored voice namespaces, structured-call validator and repair signatures, music/beat topology, exact split-versus-regex word gate, prompt/message construction, receipt identity, source-span provenance, and post-tail persistence proof. Removed no story philosophy, pass topology, pack prompt text, or examples. All v2 technical prose that conflicts with the authoritative v3 control plane below is superseded.

- Round 2 - coding reality: rechecked the writer, routing, structured-call, ledger, and freeze contracts. Corrected the source-payload shape, empty-interpreter route, setter field names and ordering, structured-call behavior, and the fact that a runner-only freeze happens before the writer tail.
- Round 3 - wiring completeness: supplied complete catalog artifacts, exact intermediate schemas, prompt seams with examples, bounded retry settings, and a runner/writer integration skeleton.
- Round 4 - convergence: added a strict finalizer design, an exact audible-word budget, provenance records, a failure taxonomy, and the PASS/FAIL audit at the end. v3 replaces the formerly unresolved callback wording with one exact, backward-compatible TailFinalizer protocol and a persisted-ledger proof.


## v4 technical control plane — authoritative implementation contract

This v4 control plane is the sole authority for implementation decisions. It
overrides only conflicting technical prose, schemas, runner pseudocode, and
test assertions below. The banks/pipeline/pack JSON identity values and every
prompt string and example remain verbatim; no creative wording is changed.

### v4 revision log

- **v4 final technical hardening:** makes `target_words` a one-use initial
  draft steer and a recorded receipt value, never an acceptance gate; converts
  the P3 beat table to advisory centers; removes all word-count retry,
  trimming, padding, and tail-target mutation requirements.
- **v4 ledger safety:** adds reject-only spoken-text and cast-lock validation
  before the tail can perform its own stage-direction/self-vocative scrubs at
  `nodes/OTR_LedgerScriptWriter.py:6006-6158`; adds post-`set_lines` stamping
  for skipped music rows because that setter retains neither `skip` nor
  `tts_skip_reason` (`nodes/production_ledger.py:1080-1167`).
- **v4 wiring/provenance:** corrects the live map/payload locations to
  `nodes/OTR_LedgerScriptWriter.py:1589-1617,3590-3624`, makes a bad
  fact-to-span map a bounded scoped LLM repair rather than an immediate
  terminal result, and establishes the shared TailFinalizer protocol referenced
  by the Gemini and Sonnet specifications.
- **v4 source-of-truth rule:** all earlier v3 technical material is historical
  where it conflicts with this section. Its untouched prompt text, JSON
  examples, and nonconflicting field catalog remain available to the builder.

### Shared TailFinalizer protocol (canonical for all three lanes)

This is one shared, additive writer change, not three lane-specific variants.
The current runner map is `_RUNNER_BY_PIPELINE` at
`nodes/OTR_LedgerScriptWriter.py:1589-1591`; a mapped runner is called with
the seven-key payload and injected slot closures at `:3590-3604`, then the
writer creates `WriterTailContext` and calls `_run_writer_tail` at `:3605-3624`.
The build must add the following optional protocol to that context and tail
call, preserving `None` for every existing lane:

~~~python
class TailFinalizer(Protocol):
    def before_save(self, *, ctx: WriterTailContext) -> None: ...
    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None: ...

def _run_writer_tail(
    self, ctx: WriterTailContext, *, tail_finalizer: TailFinalizer | None = None,
) -> tuple[str, str, str, float, str]: ...
~~~

The map wrapper for each new pipeline returns its lane's tail parts plus this
finalizer; the writer passes it to `_run_writer_tail`. Immediately after all
existing tail mutations and immediately before final script assembly/save,
`before_save` must (1) prove the receipt/text identity described below,
(2) run Phase 0, and (3) run Phase 10. It rejects either errors or warnings
and requires `meta.freeze_verdict == "frozen_clean"`
(`nodes/_otr_ledger_freeze.py:725-814`). After `Ledger.save()` returns a
nonempty path, `after_save` opens the UTF-8 JSON and performs the same
read-only gap/freeze-verdict verification on the persisted object. This is
necessary because save merges/replaces `led.data`
(`nodes/production_ledger.py:1287-1354`). It never mutates a line. A missing
callback or failed post-save check raises the lane's typed finalization error;
there is no fallback.

### Input, word-steer, and structured-call contract

At runner entry, call `validate_source_payload(payload, origin=...)` and
require exactly these seven string keys: `headline`, `summary`, `full_text`,
`source`, `date`, `link`, and `seed_text`
(`nodes/_otr_source_payload.py:80-133`). The RSS source is already fetched:
the writer constructs that payload at
`nodes/OTR_LedgerScriptWriter.py:1081-1143` and hands a copy to the mapped
runner at `:3590-3594`; this runner never fetches. `source_mode` comes only
from `resolved["seed_source"]`: `"custom_premise"` is the operator-pinned
path assembled as the same seven-key payload at `:1321-1335`; this bank's
live fetcher is stamped `"rss_fetch"` by the fetcher registry
(`nodes/_otr_source_payload.py:424-434`). Any other source stamp raises
`CodexPayloadRouteError`. Empty/malformed payload raises
`CodexPayloadShapeError`; an RSS payload below 80 split words or 12 distinct
alphanumeric tokens, or a pinned premise below 8/4, raises
`CodexPayloadThinError`. No alternative source is drawn.

`resolved["target_words"]` is validated once as an integer in `[30,900]` and
stored as runner-private `WordSteerV4 = {requested_words:int}`. It appears in
exactly one model input: P5's `artifact_inputs.initial_draft_word_steer`.
It is not passed to P0–P4, P6–P9, any repair prompt, or the writer tail. P3's
existing word-plan/table fields remain in the creative artifact but are
**advisory beat centers only**; their sum and each line's count are never
accepted/rejected against the requested number. P5 receives the selected
steer once; P7/P9 rewrite for review/evidence defects only. On acceptance
record, without judging it:

~~~text
meta.scifi_codex.word_receipt = {
  requested_words: int,
  actual_split_words: int,
  actual_ledger_word_count: int
}
~~~

`actual_*` values are observed from the accepted final text after
`set_lines`/`stamp_word_counts`; a difference from `requested_words` is
ordinary success. The split-versus-ledger comparison remains only a
tokenization-safety check required to avoid the Phase-10 warning
(`nodes/production_ledger.py:216-219,1080-1168`;
`nodes/_otr_ledger_freeze.py:372-388`): reject an isolated numeral,
symbol-only token, or other token that cannot be one lexical word; require
numbers to be spoken in words. It never triggers a length rewrite.

Every P0–P9 call uses only the injected `creative_fn` or `technical_fn`.
`structured_call` accepts a typed result, a `post_validator` returning
`None | str`, and a keyword-only repair factory
(`nodes/_otr_structured_call.py:147-173,482-535`). Each call has
`max_attempts=3`: base temperature, lower structural-JSON retry, then
0.10 repair for schema/post-validation failure as implemented at
`:547-712`. The existing per-pass token/temperature table remains in force,
except that no retry reason may be a requested-word mismatch. Prompts are
declared seam strings plus canonical JSON; Python does not compose creative
text or call any other model surface.

### Precise schema and validation overrides

Use the existing field catalog in §3 except for these complete overrides:

~~~text
PayloadEnvelopeV4 = {
  schema_version: Literal["scifi_codex.payload_envelope.v4"],
  payload: SourcePayload7,
  source_mode: Literal["rss","operator_pinned"],
  source_digest: LowercaseSha256
}
WordSteerV4 = {requested_words: int}  # runner-private; P5-only model input

CastPlanRowV4 = {
  char_id: Literal["announcer","c01","c02","c03"],
  name: str, character_description: Text, gender: Text,
  role_in_conflict: Text, voice_slot: Literal["announcer","c01","c02","c03"]
}
AdvisoryWordPlanV4 = {
  advisory_total_center: int,
  per_beat: [{beat_id:str, advisory_word_center:int}]
}
BeatPlanV4 = {
  beat_id:str, scene_id:str, shot_id:str, speaker:str,
  char_id: Literal["announcer","c01","c02","c03","music_open","music_inter","music_close"],
  speaker_role: Literal["character","announcer","music_open","music_close","music_inter"],
  line_ids:list[str], order:int, intent:Text, arc_phase:ArcPhase,
  fact_ids:list[str], advisory_voiced_word_center:int
}
ScriptLineV4 = {
  line_id:str, beat_id:str, shot_id:str,
  char_id: Literal["announcer","c01","c02","c03","music_open","music_inter","music_close"],
  speaker_role: Literal["character","announcer","music_open","music_close","music_inter"],
  text:str, skip:bool, tts_skip_reason:str | null,
  traits:str, boundary:Literal["shot_start","beat_start","continue"],
  arc_phase:ArcPhase, compose_flags:list[str], beat_intent:str,
  dialogue_slot_id:str, fact_ids:list[str]
}
FinalAuditV4 = {
  schema_version: Literal["scifi_codex.final_audit.v4"],
  script_digest: LowercaseSha256, verdict: Literal["pass","rewrite"],
  issues:list[FinalIssueV1], line_checks:list[LineCheckV1],
  fact_checks:list[FactCheckV1],
  observed_word_counts:{character:int, announcer:int, audible_total:int}
}
~~~

All models use `extra="forbid"`, strict scalar types, no aliases, and no
Pydantic text-length clamps because tolerant parsing can clip text
(`nodes/_otr_structured_call.py:322-474`). IDs/references retain the existing
sequential/cross-reference validators. `CastPlanRowV4.name` is exactly
`ANNOUNCER` for `announcer`, otherwise one canonical Title-Case token; the
P3 `speaker` equals that locked name exactly for its `char_id`. P2 fixes the
roster; no later artifact may add a label, alias, honorific, or speaker.
Character/announcer lines must resolve to that cast row; music uses only its
matching fixed sentinel, which is the live non-cast exception
(`nodes/production_ledger.py:91-125`).

A non-music line has `skip:false`, nonempty `text`, and no
`tts_skip_reason`; a music line has `skip:true`, `text:""`, and
`tts_skip_reason:"music_cue"`. Its cue must anchor to a non-skipped line in
the declared beat. `set_lines` drops both skip fields, so immediately after
that setter the runner stamps exactly those two already-validated,
noncreative fields on the known music rows, then validates their IDs, roles,
and reasons before `set_music`. This is structural assembly, never text
surgery.

For every non-skipped line, reject—do not strip or normalize—newline/tab
decoration, brackets, parentheticals, backticks, markdown asterisks/fences,
a leading cast/role label, a standalone stage/action cue, a fully quoted
line, or an all-caps lexical word of two or more letters. Spoken acronyms use
lowercase lexical form (for example `nasa`). Reject a line beginning with
its own locked cast first name followed by a vocative separator. These
pre-tail checks ensure the current writer scrubs/reattribution at
`nodes/OTR_LedgerScriptWriter.py:6006-6158` has nothing to alter. A failure
is `CodexSpokenTextError` returned as the originating LLM's post-validator
string; Python never edits the response.

P0 keeps literal source spans: `quote == payload[field][start:end]`.
P8 reports each broken audible fact mapping with `line_id`, `fact_id`,
`span_id`, and owning `beat_id`. A fact-trace defect invokes at most two
P9 scoped-repair calls, each using the normal three-attempt creative ladder
and returning a complete `ScriptArtifactV4`. The validator requires all
non-scoped lines byte-identical, rechecks the affected beat only, then P8
re-audits the complete script. Exhaustion raises
`CodexFactTraceExhaustedError`; a malformed P0 span still uses P0's normal
bounded repair ladder. No Python substitutes a claim or edits a line.

### Assembly, proof, and exact implementation skeleton

Assembly order is `set_cast`, `set_scenes`, `set_shots`, `set_beats`,
`set_lines`, post-setter music skip stamps, `set_music`,
`stamp_word_counts`, metadata/provenance, Phase 0, then the shared
TailFinalizer. `clips` is an explicit empty list. Every scene owns a shot,
every shot a beat, every beat's ordered `line_ids` exactly matches its lines,
and every music `anchor_line_id` is a real voiced line. The legal roles are
exactly `character`, `announcer`, `music_open`, `music_close`, and
`music_inter` (`nodes/_otr_ledger_freeze.py:85-96`). Setter shapes and the
music fields are verified at `nodes/production_ledger.py:792-1207`.

The runner records the accepted P9 raw UTF-8 response, SHA-256, invocation
ID, per-line final-text SHA-256, and fact/source-span graph under
`meta.scifi_codex`. Before tail/save it strictly reparses that exact P9
receipt and requires each non-skipped ledger string to match exactly. It
also verifies every audible news fact has a P0 fact/span map and every span
still equals its source slice. The shared finalizer repeats this proof after
tail and after save. Thus every spoken line is verbatim LLM output and every
audible news claim is traceable.

The §4.4 skeleton is overridden by these signatures:

~~~python
def run_scifi_codex_episode(*, payload: dict[str, str], pack: Any,
    resolved: Mapping[str, Any], led: Ledger, meta: dict[str, Any],
    creative_fn: GenerateFn, technical_fn: GenerateFn, slot_scheduler: Any,
    source_bank_row: SourceBank, story_rules: Mapping[str, Any],
    episode_root: Path, episode_id: str) -> ScifiCodexTailParts: ...
def validate_payload_envelope(payload: Mapping[str, Any],
    resolved: Mapping[str, Any]) -> tuple[PayloadEnvelopeV4, WordSteerV4]: ...
def run_script_artifact(*, pass_id: Literal["P5","P7","P9"],
    word_steer: WordSteerV4 | None, repair_scope: Mapping[str, str] | None,
    ...) -> ScriptArtifactV4: ...
def validate_spoken_text_and_roster(script: ScriptArtifactV4,
    cast: CastPlanV4, score: RadioScoreV4) -> None: ...
def stamp_music_skip_contract_after_set_lines(led: Ledger,
    script: ScriptArtifactV4) -> None: ...
def record_word_receipt(meta: MutableMapping[str, Any], requested_words: int,
    led: Ledger) -> None: ...
~~~

Only P5 receives non-`None` `word_steer`; it is `None` for P7/P9 and every
repair. The runner returns tail parts with the shared finalizer, a title
override, and no story-spine mutation. It never changes
`resolved["target_words"]`.

## Archived v3 technical control plane — superseded by v4


This section supersedes only conflicting technical material elsewhere in this
document. The design philosophy, pass topology, pack prompt strings, examples,
and story voice remain frozen as written. The purpose is to make the existing
creative design implementable without Python-authored dialogue or a silent
repair.

### Shared dispatch and durable-finalization patch

The current map has one existing entry at
`nodes/OTR_LedgerScriptWriter.py:1569-1616`; the payload reaches a mapped
runner as `dict(resolved["news_article"])` with the two slot closures at
`:3590-3604`. The build adds a lazy wrapper and inserts, without replacing the
dictionary, `"scifi_codex_circuit": _run_scifi_codex_lane`. It changes this
bank's `runnable` and pipeline's `executable` from `false` to `true` in the
same build change, because routing rejects a runnable non-source-contract
pipeline with `executable=false` (`nodes/_otr_story_routing.py:409-442`) and a
map miss fails loud (`nodes/OTR_LedgerScriptWriter.py:1599-1617`). No workflow
JSON, fetcher, interpreter, environment variable, sidecar file, model loader,
or network path is added.

When the three bake-off rows are built together, their only permitted
`banks.json` insertion order is Codex, Gemini, Sonnet immediately before
`custom_source_bank`; an independent lane build inserts only its own row at
that same boundary. IDs, pack directories, and seam prefixes are respectively
`scifi_codex`/`scifi_codex`/`codex_`, `scifi_gemini`/`scifi_gemini`/`gemini_`,
and `scifi_sonnet`/`scifi_sonnet`/`sonnet_`.

The runner cannot truthfully freeze the final ledger before the writer tail:
the tail mutates metadata and then serializes/saves after runner return
(`nodes/OTR_LedgerScriptWriter.py:6194-6706`). The required additive,
backward-compatible build patch is exactly this optional protocol; legacy
callers and existing tail parts pass `None` and retain their behavior:

~~~python
# TailFinalizer is the one shared v4 protocol defined above.
# WriterTailContext is unchanged.  The map-dispatch call becomes:
return self._run_writer_tail(
    _tail_ctx,
    tail_finalizer=getattr(_parts, "tail_finalizer", None),
)

# _run_writer_tail gains only this optional keyword:
def _run_writer_tail(
    self, ctx: WriterTailContext, *, tail_finalizer: TailFinalizer | None = None,
) -> tuple[str, str, str, float, str]:
    ...  # existing tail through all mutable meta/provenance work
    if tail_finalizer is not None:
        tail_finalizer.before_save(ctx=ctx)
    script_text = _PL.assemble_script_text_from_ledger(led.data)
    news_json = _build_news_payload(...)
    saved_path = led.save()
    if not saved_path:
        raise CodexLedgerSaveError("Ledger.save returned no path")
    if tail_finalizer is not None:
        tail_finalizer.after_save(saved_path=saved_path, ledger_data=led.data)
    script_json = json.dumps(led.data, indent=2, ensure_ascii=False)
    ...
~~~

The `before_save` implementation runs `phase_0_gap_audit_pre(led)` and
`phase_10_gap_audit_post_and_freeze(led)`, rejects either report's errors or
warnings, and requires `meta.freeze_verdict == "frozen_clean"`
(`nodes/_otr_ledger_freeze.py:725-814`). The `after_save` implementation
opens `saved_path` as UTF-8 JSON, compares it with the post-save `led.data`
(which `Ledger.save()` replaces with its merged payload at
`nodes/production_ledger.py:1300-1354`), runs read-only `run_gap_audit` on the
persisted object, and requires zero errors, zero warnings, and
`freeze_verdict == "frozen_clean"`. It never mutates text. Any failure raises
`CodexLedgerSaveError` or `CodexSavedLedgerAuditError`; no node output is
returned. The title portion of the same additive patch takes an optional
`title_source_override` from tail parts, using `"scifi_codex_script_title"`
instead of the live hard-coded fable label for a non-`None` title override
(`nodes/OTR_LedgerScriptWriter.py:6255-6265`).

### Deterministic call envelope, retry, and receipt contract

Every P0-P9 invocation uses only the injected `creative_fn` or
`technical_fn`. `structured_call` does not transmit a Pydantic schema by
itself (`nodes/_otr_structured_call.py:482-535`), so the runner supplies one
system message made by joining exactly the declared seam string(s), followed
by one user message whose content is canonical JSON:

~~~text
{
  "pass_id": "P0" | ... | "P9",
  "artifact_inputs": { ... named prior artifacts exactly as typed ... },
  "result_json_schema": <result_type.model_json_schema()>
}
~~~

The runner uses `json.dumps(..., sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`; it never calls `.format()` on a pack prompt. Before P0,
the canonical seven-key payload is size-limited to 48,000 UTF-8 bytes;
oversize input raises `CodexPayloadOversizeError` instead of allowing the
writer's closure to truncate prompt context. `validate_source_payload` is
called again at runner entry, then failures are wrapped as
`CodexPayloadShapeError`. RSS is thin below 80 split words or 12 distinct
alphanumeric tokens; an operator-pinned payload is thin below 8 split words
or 4 distinct alphanumeric tokens. `resolved["seed_source"] ==
"custom_premise"` is the sole pinned discriminator, matching
`nodes/OTR_LedgerScriptWriter.py:1277-1337`; the runner never fetches.

The exact wrapper type is:

~~~python
T = TypeVar("T")
T = TypeVar("T")
GenerateFn = Callable[..., str]

def invoke_codex_structured(
    *, pass_id: str, slot: Literal["creative", "technical"],
    slot_fn: GenerateFn, pack: Any, seam_refs: tuple[str, ...],
    artifact_inputs: Mapping[str, Any], result_type: type[T],
    post_validator: Callable[[T], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, call_journal: "CodexCallJournal",
) -> T: ...

def codex_repair_factory(
    *, original_prompt: Any, failed_output: str, error: BaseException,
) -> list[dict[str, str]]: ...
~~~

`post_validator` returns `None` to accept or a nonempty error string to reject
the result; the helper wraps the latter in `PostValidationError`
(`nodes/_otr_structured_call.py:120-138,482-535`). The repair factory returns
only a list of system/user messages—never a completed artifact—so a failed
spoken artifact is repaired by the same LLM slot. Every call passes
`max_attempts=3`; malformed JSON may use base/lower/0.10 repair, while schema
or post-validation failure uses base/0.10 repair as implemented at
`nodes/_otr_structured_call.py:547-712`. No exception is converted into
feedback or swallowed.

`CodexCallJournal` is keyed by monotonically assigned invocation ID, not pass
ID, and records `pass_id`, slot, every raw response SHA-256, every attempted
temperature, accepted attempt, and the accepted raw response. This prevents
P3/P8/P9 repeated calls from overwriting proof. Receipt revalidation uses the
same tolerant parser and post-validator path as the live call; it hashes the
accepted raw response and verifies that the accepted JSON object still yields
the exact final string. Pydantic models use `extra="forbid"`, strict scalar
types, no aliases, and no length constraint on LLM-authored text; all bounds
are post-validator checks because tolerant validation otherwise clips text
(`nodes/_otr_structured_call.py:322-474`).

### Required post-setter and provenance checks

Assembly order is exact: `set_cast`, `set_scenes`, `set_shots`, `set_beats`,
`set_lines`, `set_music`, `stamp_word_counts`, then deterministic validation.
`stamp_word_counts` is required because mapped runners bypass the inline
writer call and it stamps category counts using split semantics
(`nodes/production_ledger.py:479-514`). `clips` remains the initialized empty
list; the runner never creates clip records.

Before Phase 0, the lane rejects an empty or duplicate ID; a missing
scene/shot/beat/line; a shot whose scene is absent; a beat whose shot or scene
is absent; a line whose beat or shot is absent; a beat whose ordered
`line_ids` are not exactly its child lines; a cast row without a non-skipped
character/announcer line; or a voiced line whose `char_id` does not resolve to
a cast row. Every scene owns a shot, every shot a beat, and every beat a line.
Music beats use their matching sentinel role with an empty nonspoken text;
each music cue's `anchor_line_id` must instead point to an existing voiced
line in its declared anchor beat: open to the first voiced line, close to the
last, and inter to its declared boundary line. `set_music` retains
`generation_prompt` and `anchor_line_id` but does not validate them
(`nodes/production_ledger.py:1179-1207`), so this check is mandatory.

`set_lines` retains no arbitrary per-line fact or authorship key
(`nodes/production_ledger.py:1138-1157`). The required metadata is therefore:

~~~text
meta.scifi_codex = {
  "payload_sha256": <A0 digest>,
  "fact_index": {fact_id: {source_spans: [...]}, ...},
  "line_fact_ids": {line_id: [fact_id, ...], ...},
  "line_authorship": {line_id: {invocation_id, raw_sha256, text_sha256}, ...},
  "call_journal": {invocation_id: {...}},
  "requested_audible_words": T,
  "character_target_words": C,
  "announcer_target_words": A
}
~~~

Every source span stores a payload field, integer start/end offsets, and an
exact `quote` equal to that UTF-8 Python-string slice. Every audible factual
claim has at least one fact ID; the finalizer resolves each fact through those
spans and verifies the final ledger text SHA-256 matches its accepted creative
receipt. `fact_uses` cannot be stored only in a transient artifact.


### Retired v3 allocation note — superseded by v4 word-steer contract

The retained beat table is creative planning context only. In v4 its values
are advisory centers passed through P3; neither the table nor its total has a
validator, retry edge, assembly mutation, or tail effect. The selected
`target_words` reaches P5 once as described in the v4 control plane, and
actual counts are recorded without judgment.
## 1. Design philosophy

Sci-Fi Codex treats a science report as a constrained pressure source, not as a lecture subject. The listener first hears people making consequential choices under a rule that came from the report; the factual coda then identifies what was real. That produces a complete radio drama with a clear beginning, turn, decision, and after-image while keeping every audible news claim accountable to evidence.

The creative voice is deliberately confined to the declared pack seams below. Python has only four jobs: validate data, call an operator-supplied slot callable, assemble the canonical ledger, and reject a failed contract. It never writes, patches, trims, substitutes, or regex-repairs dialogue.

## 2. Historical v3 contract findings — superseded where v4 differs

All line references below were checked in the live files named by the bake-off brief.

| Integration point | Verified live behavior | Build-ready design |
|---|---|---|
| Bank row parsing | The bank parser requires the exact bank key set, including guide_ref; it permits an empty guide_ref and validates scalar defaults at nodes/_otr_story_routing.py:192-240. | The bank row includes guide_ref as an empty string even though the original prose field list omitted it. Omitting it would fail registry validation. |
| Pipeline parsing | Pipeline passes have exactly pass_id, slot, seam_refs, and description; slots are only creative or technical; custom seams must be declared by that pipeline at nodes/_otr_story_routing.py:198-305. | The pipeline below declares ten Codex seams and no additional model-selection mechanism. |
| Runnable route | runnable is the runtime bank gate, while executable is validation metadata; a runnable source-contract pipeline requires both fetcher and interpreter, and a non-source-contract lane requires an executable pipeline at nodes/_otr_story_routing.py:409-442,513-524. | The spec-only JSON is intentionally runnable false and executable false. When implemented, this lane changes both to true, keeps interpreter empty, and keeps requires_source_contract false because the runner owns direct payload validation. |
| Routed pack identity | Routing resolves the selected pack and cross-checks bank, pipeline, model, and declared seams at nodes/_otr_story_routing.py:335-442. | The runner adds a stricter preflight: the pack prompt-stage key set must equal the ten Codex seams exactly before any LLM call. |
| Payload contract | The writer creates the seven-key RSS payload at nodes/OTR_LedgerScriptWriter.py:1081-1143 and the seven-key pinned payload at nodes/OTR_LedgerScriptWriter.py:1321-1337. | A0 validates exactly that shape, contains the input unchanged plus routing metadata and a digest, and permits no live replacement draw after failure. |
| Pinned source path | A nonempty custom premise becomes a validated seven-key payload with source User Seed and seed_source custom_premise at nodes/OTR_LedgerScriptWriter.py:1321-1337; the writer trims outer whitespace before this branch at nodes/OTR_LedgerScriptWriter.py:1277-1279. | seed_source equal to custom_premise is the sole pinned-path discriminator. Its text reaches A0 as seed_text; the runner never calls the fetcher in that case. |
| Runner dispatch | The writer resolves a pipeline through _RUNNER_BY_PIPELINE at nodes/OTR_LedgerScriptWriter.py:1589-1616. The dispatch sends payload=dict(resolved["news_article"]), the routed pack, both slot callables, ledger, and metadata to the lane at nodes/OTR_LedgerScriptWriter.py:3568-3601. | Add one lazy-import wrapper and one map entry for scifi_codex_circuit; do not alter any existing map entry or lane. |
| Model access | The writer creates exactly the two permitted closures with slot_scheduler.for_slot("creative") and .for_slot("technical") at nodes/OTR_LedgerScriptWriter.py:3455-3471. _invoke_slot accepts only slot_fn(messages, temperature=, max_new_tokens=) at nodes/_otr_structured_call.py:241-251. | The runner receives only creative_fn and technical_fn. It does not import a model loader, call a scheduler, select a third slot, or accept a model identifier. |
| Tail mutation | The custom runner returns tail inputs at nodes/OTR_LedgerScriptWriter.py:3605-3624. The writer tail can make later technical metadata calls and can write/save after the runner at nodes/OTR_LedgerScriptWriter.py:6500-6555,6678,6705-6712. | A pre-tail audit alone cannot prove the saved ledger. The required additive TailFinalizer runs after tail mutations before save and then reads/verifies the persisted ledger after save. |
| Ledger setters | The canonical setters are set_cast, set_scenes, set_shots, set_beats, set_lines, set_music; line rows retain only the fields assembled at nodes/production_ledger.py:792-835,837-894,1080-1207. | The runner calls those setters in that exact order and stores source/authorship fields in meta.scifi_codex, because set_lines drops unknown per-line keys. |
| Freeze | Phase 0 reports gaps at nodes/_otr_ledger_freeze.py:599-750. Phase 10 raises on errors but returns frozen_with_warns when warnings remain at nodes/_otr_ledger_freeze.py:753-814. | Both warnings and errors are fatal for this lane. The finalizer accepts only frozen_clean. |

## 3. v3 field catalog — use only fields not explicitly replaced by the v4 control plane

All LLM-produced artifacts, P0 through P9, are strict JSON objects decoded by structured_call. A0 is deterministic input validation; P10 is an internal repair-message factory; P11 and P12 are deterministic closure reports. Every LLM-produced model has extra-forbid behavior, strict scalar types, no aliases, and no Pydantic max_length constraints on text. Length, word, ID, and cross-artifact limits are explicit post-validators so a violation causes an LLM repair/rewrite rather than tolerant truncation. This matters because the tolerant parser can clamp constrained fields at nodes/_otr_structured_call.py:322-370,420-474.

Text below means a JSON string that is nonempty after validation, contains no newline, has no leading or trailing whitespace, and is retained exactly as supplied. The runner does not normalize it. `spoken_word_count(text)` means `len(text.split())`, which is also the Phase-10 comparison count at `nodes/_otr_ledger_freeze.py:372-388`. Before any spoken line is accepted, the runner requires `spoken_word_count(text) == production_ledger._word_count(text)` (`nodes/production_ledger.py:216-219`); therefore every emitted spoken token must contribute exactly one ledger-regex word. A mismatch is a content rejection sent to the originating LLM repair/rewrite path, never a Python edit.

### A0 - PayloadEnvelopeV1, created before every model call

~~~text
{
  schema_version: Literal["scifi_codex.payload_envelope.v1"],
  payload: {
    headline: str, summary: str, full_text: str, source: str,
    date: str, link: str, seed_text: str
  },
  source_mode: Literal["rss", "operator_pinned"],
  seed_source: str,
  source_digest: str,
  target_audible_words: int
}
~~~

Validators: the runner itself requires exactly the seven named string keys, no unknown keys, and nonblank seed_text. source_digest is lowercase SHA-256 of UTF-8 canonical JSON for those seven keys using sorted keys and compact separators. target_audible_words is an integer in 30 through 900 inclusive. For rss, headline, source, and link must be nonempty; rss_evidence_text is full_text when that field is nonblank, otherwise headline plus summary plus seed_text, and it must contain at least 80 whitespace-separated words and 12 distinct alphanumeric tokens. For operator_pinned, seed_text must contain at least 8 words and 4 distinct alphanumeric tokens; blank headline, summary, full_text, date, and link are allowed because that is the verified writer shape. Exact-shape failure raises CodexPayloadShapeError; thin content raises CodexPayloadThinError.

### P0 - FactIndexV1, technical

~~~text
FactIndexV1 = {
  schema_version: Literal["scifi_codex.fact_index.v1"],
  source_digest: str,
  source_mode: Literal["rss", "operator_pinned"],
  facts: FactV1[],
  entities: EntityV1[],
  numbers: NumberV1[],
  tone: ToneV1
}
SourceSpanV1 = {span_id: str, field: Literal["headline","summary","full_text","seed_text"], start: int, end: int, quote: Text}
FactV1 = {fact_id: str, claim: Text, kind: Literal["finding","method","measurement","actor","time","uncertainty","consequence"], source_spans: SourceSpanV1[], numeric_tokens: str[]}
EntityV1 = {entity_id: str, name: Text, kind: Literal["person","organization","place","object","process","measurement"], source_spans: SourceSpanV1[]}
NumberV1 = {number_id: str, verbatim: Text, meaning: Text, fact_id: str, span_id: str}
ToneV1 = {register: Literal["cautious","urgent","curious","somber","hopeful","clinical"], uncertainty_markers: str[], evidence_span_ids: str[]}
~~~

Validators: exact digest/mode equality to A0; fact IDs are F01 through F0N without gaps where N is 3 through 9; entity IDs are E01 through E0N without gaps where N is 1 through 12; number IDs are N01 through N0N without gaps and the list may be empty only when the source has no numeric token; all IDs are unique. Every span quote must equal the exact A0 field slice from start through end, every number verbatim must occur in its referenced span, and every referenced fact/span must exist. Claims are 4 through 45 words. Tone has at least one evidence span. Thus P0 explicitly extracts facts, entities, numbers, and source tone before any fictional planning.

### P1 - DramaticQuestionV1, creative

~~~text
DramaticQuestionV1 = {
  schema_version: Literal["scifi_codex.dramatic_question.v1"],
  source_digest: str,
  premise: Text,
  central_question: Text,
  protagonist_need: Text,
  opposing_pressure: Text,
  irreversible_choice: Text,
  stakes: Text,
  ending_direction: Text,
  fact_uses: FactUseV1[],
  invented_elements: InventedElementV1[]
}
FactUseV1 = {fact_id: str, dramatic_function: Literal["trigger","constraint","test","revelation","coda"], transformation_rule: Text}
InventedElementV1 = {element_id: str, kind: Literal["person","place","device","institution","event"], description: Text, story_function: Text, explicitly_fictional: Literal[true]}
~~~

Validators: source digest equality; 2 through 5 distinct fact uses, all from P0; at least one constraint or test; all invented element IDs X01 onward without gaps; every invented element has explicitly_fictional true. No invented number, date, institution, or scientific result may be described as a source fact.

### P2 - CastPlanV1, creative

~~~text
CastPlanV1 = {
  schema_version: Literal["scifi_codex.cast_plan.v1"],
  cast: CastPlanRowV1[],
  conflict_edges: ConflictEdgeV1[]
}
CastPlanRowV1 = {char_id: Literal["announcer","c01","c02","c03"], name: Text, character_description: Text, gender: Text, role_in_conflict: Text, voice_slot: Literal["announcer","c01","c02","c03"]}
ConflictEdgeV1 = {from_char_id: Literal["c01","c02","c03"], to_char_id: Literal["c01","c02","c03"], pressure: Text}
~~~

Validators: ordered cast IDs are exactly announcer,c01,c02 for targets 30 through 550 and announcer,c01,c02,c03 for targets 551 through 900. The announcer name is exactly ANNOUNCER; no other row may use a sentinel ID or that name. voice_slot equals char_id. There is at least one directed conflict edge between every non-announcer character and no self-edge. Assembly converts each row to the verified set_cast fields: char_id, name, character_description, gender, tts_model, voice_preset, voice_params, line_count, and word_count.

The planned literal voice roster is deliberately noncreative and fixed: ANNOUNCER maps to `tts_model: "kokoro", voice_preset: "bm_george"`; c01 maps to `tts_model: "bark", voice_preset: "v2/en_speaker_6"`; c02 maps to `tts_model: "bark", voice_preset: "v2/en_speaker_3"`; c03 maps to `tts_model: "bark", voice_preset: "v2/en_speaker_0"`; each has empty-object voice parameters. The stored Bark preset must begin literally `v2/`, not `bark/v2/`, because Phase 10 checks the raw stored value independently of `tts_model` (`nodes/_otr_ledger_freeze.py:467-503`); ANNOUNCER uses the separate Kokoro namespace (`:479-486`). Before P2 is accepted, the runner must resolve all roster entries with the active voice resolver; unavailable entries raise CodexVoiceInventoryError, never a substitution.

### P3 - RadioScoreV1, creative

~~~text
RadioScoreV1 = {
  schema_version: Literal["scifi_codex.radio_score.v1"],
  title_seed: Text,
  premise: Text,
  setting: Text,
  time_of_day: Text,
  sound_palette: Text[],
  scenes: ScenePlanV1[],
  shots: ShotPlanV1[],
  beats: BeatPlanV1[],
  music_cues: MusicPlanV1[],
  word_plan: WordPlanV1
}
ScenePlanV1 = {scene_id: str, description: Text, env: Text, fact_ids: str[]}
ShotPlanV1 = {shot_id: str, scene_id: str, description: Text, visual_prompt: Text, beat_ids: str[]}
BeatPlanV1 = {beat_id: str, scene_id: str, shot_id: str, speaker: Text, char_id: Literal["announcer","c01","c02","c03","music_open","music_inter","music_close"], speaker_role: Literal["character","announcer","music_open","music_inter","music_close"], line_ids: str[], order: int, intent: Text, arc_phase: Literal["arrival","test","turn","decision","coda"], fact_ids: str[], target_voiced_words: int}
MusicPlanV1 = {cue_id: str, music_line_id: str, placement: Literal["open","inter","close"], description: Text, generation_prompt: Text, anchor_beat_id: str, target_duration_s: number}
WordPlanV1 = {target_audible_words: int, character_words: int, announcer_words: int, per_beat: BeatBudgetV1[]}
BeatBudgetV1 = {beat_id: str, character_words: int, announcer_words: int}
~~~

Validators: setting and time_of_day are Text; sound_palette has 2 through 6 distinct Text entries. IDs are sequential SC01 onward, SH01 onward, BT01 onward, MC01 onward, and L001 onward; every reference resolves; shot beat IDs are exactly the child beats; scene and beat order are contiguous; all fact IDs resolve to P0. A voiced beat has char_id/speaker exactly matching P2, role character for c01-c03 or announcer for ANNOUNCER, one or more unique spoken line IDs, and target_voiced_words greater than zero. A music beat has the matching music sentinel char_id and role, exactly one silent music line ID, no fact IDs, and target_voiced_words equal to zero. There are two music beats/cues (open and close) below 150 words and three (open/inter/close) at 150 words or more. `total_beats = max(music_cues + 3, min(18, max(3, 2 * scene_count + floor(target_audible_words / 180))))`; `voiced_beats = total_beats - music_cues`; thus the minimum 30-word case has three voiced beats plus two music beats, and 720 has nine voiced beats plus three music beats. scene_count is 1 for 30-120, 2 for 121-280, 3 for 281-550, and 4 for 551-900. shot_count equals max(2 times scene_count, ceil(total_beats times 2 / 3)). Per-voiced-beat budgets sum exactly to the character and announcer totals; music beats are the sole zero-word exception.

### P4 - StructureReviewV1, technical

~~~text
StructureReviewV1 = {
  schema_version: Literal["scifi_codex.structure_review.v1"],
  score_digest: str,
  verdict: Literal["pass","rewrite"],
  issues: StructureIssueV1[],
  preserved_fact_ids: str[]
}
StructureIssueV1 = {issue_id: str, location: Literal["scene","shot","beat","music","word_plan"], location_id: str, rule: Text, required_change: Text}
~~~

Validators: score_digest is the SHA-256 of P3 canonical JSON; issue IDs are SR01 onward; every location exists; all preserved facts exist. pass requires an empty issue list. rewrite requires 1 through 8 issues. The topology permits one P3 rewrite after a P4 rewrite verdict, then requires a P4 pass.

### P5, P7, and P9 - ScriptArtifactV1, creative full replacements

~~~text
ScriptArtifactV1 = {
  schema_version: Literal["scifi_codex.script_artifact.v1"],
  pass_id: Literal["P5","P7","P9"],
  source_digest: str,
  title: Text,
  premise: Text,
  lines: ScriptLineV1[],
  music_anchors: MusicAnchorV1[],
  fact_uses: SpokenFactUseV1[],
  authorship: AuthorshipV1
}
ScriptLineV1 = {
  line_id: str, beat_id: str, shot_id: str,
  char_id: Literal["announcer","c01","c02","c03","music_open","music_inter","music_close"],
  speaker_role: Literal["character","announcer","music_open","music_close","music_inter"],
  text: str, traits: str, boundary: Literal["shot_start","beat_start","continue"],
  arc_phase: Literal["arrival","test","turn","decision","coda"],
  compose_flags: str[], beat_intent: str, target_words: int, dialogue_slot_id: str,
  fact_ids: str[]
}
MusicAnchorV1 = {cue_id: str, anchor_line_id: str}
SpokenFactUseV1 = {line_id: str, fact_id: str, span_ids: str[], spoken_claim: Text}
AuthorshipV1 = {full_replacement: Literal[true], generated_line_ids: str[], source_digest: str}
~~~


Validators: line IDs and P3 membership/references remain exact. v4 replaces
the old `target_words` fields/count equality rules with `ScriptLineV4` in the
authoritative control plane: music is skipped with its fixed reason, all
audible text passes the reject-only roster/TTS/tokenization validator, and no
requested-length comparison exists. `music_anchors`, fact mappings, and
whole-artifact P7/P9 replacement semantics remain as specified.
### P6 - ListenerReviewV1, technical

~~~text
ListenerReviewV1 = {
  schema_version: Literal["scifi_codex.listener_review.v1"],
  script_digest: str,
  verdict: Literal["rewrite"],
  strengths: Text[],
  issues: ListenerIssueV1[]
}
ListenerIssueV1 = {issue_id: str, category: Literal["clarity","causality","character","radio_action","pacing","coda","fact_boundary"], line_ids: str[], evidence: Text, rewrite_direction: Text}
~~~

Validators: script digest equals P5; issue IDs are LR01 onward; there are 3 through 7 issues; every cited line exists. P6 always requests the P7 full rewrite. It may identify strengths but cannot waive a rewrite.

### P8 - FinalAuditV1, technical

~~~text
FinalAuditV1 = {
  schema_version: Literal["scifi_codex.final_audit.v1"],
  script_digest: str,
  verdict: Literal["pass","rewrite"],
  issues: FinalIssueV1[],
  line_checks: LineCheckV1[],
  fact_checks: FactCheckV1[],
  word_counts: {character: int, announcer: int, audible_total: int}
}
FinalIssueV1 = {issue_id: str, category: Literal["ledger","word_budget","fact_trace","fiction_boundary","voice","radio_action","coda"], line_ids: str[], required_change: Text}
LineCheckV1 = {line_id: str, pass: bool, reason: Text}
FactCheckV1 = {line_id: str, fact_id: str, pass: bool, reason: Text}
~~~

Validators: script digest equals the audited ScriptArtifact; line checks cover every spoken line once; fact checks cover every fact use once; reported counts equal recomputed counts; all IDs resolve. pass requires no issues and every check true. rewrite requires at least one issue. P8 runs after P7 and after each P9 output.

### P10, P11, and P12 - repair and deterministic closure

P10 is the structured-call repair-message factory, not a new content writer. RepairMessageSetV1 = {pass_id: str, messages: MessageV1[]}, where MessageV1 = {role: Literal["system","user"], content: str}, is an internal factory value and is never serialized as a pass result. The factory returns messages only; structured_call sends those messages to the originating P0-P9 slot, whose next response must be the same failed artifact type. This is why the repair seam instructs the model to return the repaired artifact JSON rather than a message set.

P11 creates PreTailAssemblyReportV1 = {schema_version: Literal["scifi_codex.pre_tail.v1"], final_artifact_digest: str, ledger_digest: str, phase0_errors: str[], phase0_warnings: str[]} after assembly and strict Phase 0. Both arrays must be empty or P11 raises.

P12 creates PostTailFreezeReportV1 = {schema_version: Literal["scifi_codex.post_tail.v1"], ledger_digest: str, phase10_verdict: Literal["frozen_clean"], errors: str[], warnings: str[]} in the planned writer-tail callback. Both arrays must be empty; any other verdict raises.

## 4. The four complete artifacts

### 4.1 nodes/story_packs/banks.json row

The live parser requires the otherwise-unlisted guide_ref field; this row is valid for that parser.

~~~json
{
  "source_bank_id": "scifi_codex",
  "label": "Sci-Fi Codex - Proof-Pressure Radio",
  "source_kind": "article",
  "interpreter": "",
  "fetcher": "science_rss",
  "default_story_model": "scifi_codex_v1",
  "default_story_pipeline": "scifi_codex_circuit",
  "defaults": {
    "story_form_label": "science-fiction audio drama",
    "source_material_label": "science story",
    "title_form_label": "science-fiction radio drama",
    "coda_mode": "real_news_report",
    "credits_source_line": "dramatized by machine from a science report"
  },
  "required_seams": [],
  "runnable": false,
  "guide_ref": ""
}
~~~

Implementation placement: insert immediately before custom_source_bank, as requested. It remains disabled until the runner and writer changes described below exist and pass tests.

### 4.2 nodes/story_packs/pipelines.json entry

~~~json
{
  "story_pipeline_id": "scifi_codex_circuit",
  "label": "Sci-Fi Codex Circuit",
  "executable": false,
  "requires_source_contract": false,
  "declared_seams": [
    "codex_fact_index_system",
    "codex_question_system",
    "codex_pressure_cast_system",
    "codex_radio_score_system",
    "codex_coda_contract_system",
    "codex_play_system",
    "codex_listening_room_system",
    "codex_retake_system",
    "codex_final_audit_system",
    "codex_structured_repair_system"
  ],
  "passes": [
    {
      "pass_id": "P0_fact_index",
      "slot": "technical",
      "seam_refs": ["codex_fact_index_system"],
      "description": "Consumes A0, extracts source-spanned facts, entities, numbers, and tone, and fails if evidence cannot be indexed."
    },
    {
      "pass_id": "P1_dramatic_question",
      "slot": "creative",
      "seam_refs": ["codex_question_system"],
      "description": "Turns P0 constraints into a fiction-marked dramatic question without treating invention as reporting."
    },
    {
      "pass_id": "P2_pressure_cast",
      "slot": "creative",
      "seam_refs": ["codex_pressure_cast_system"],
      "description": "Builds the fixed-size cast and audible conflict graph from P1."
    },
    {
      "pass_id": "P3_radio_score",
      "slot": "creative",
      "seam_refs": ["codex_radio_score_system", "codex_coda_contract_system"],
      "description": "Builds scenes, shots, beats, music cues, fact placements, and advisory beat-length centers."
    },
    {
      "pass_id": "P4_structure_review",
      "slot": "technical",
      "seam_refs": ["codex_radio_score_system", "codex_coda_contract_system"],
      "description": "Reviews P3; a rewrite verdict loops once to P3 and a second rewrite verdict fails loud."
    },
    {
      "pass_id": "P5_first_play",
      "slot": "creative",
      "seam_refs": ["codex_play_system", "codex_coda_contract_system"],
      "description": "Writes a complete first ScriptArtifact, including every announcer and music line."
    },
    {
      "pass_id": "P6_listening_room",
      "slot": "technical",
      "seam_refs": ["codex_listening_room_system"],
      "description": "Judges the current whole script; concrete issues feed a bounded P7 line patch and an issue-free review accepts the current script."
    },
    {
      "pass_id": "P7_full_retake",
      "slot": "creative",
      "seam_refs": ["codex_retake_system", "codex_coda_contract_system"],
      "description": "Writes a bounded target-line text patch from the latest P6 findings, merges only line.text, and returns every accepted patch to a fresh P6 judgment."
    },
    {
      "pass_id": "P8_final_audit",
      "slot": "technical",
      "seam_refs": ["codex_final_audit_system", "codex_coda_contract_system"],
      "description": "Audits the current full script, source mappings, and ledger-ready role references; it feeds P9 and repeats after every accepted P9 patch."
    },
    {
      "pass_id": "P9_closing_rewrite",
      "slot": "creative",
      "seam_refs": ["codex_retake_system", "codex_coda_contract_system"],
      "description": "Writes a bounded target-line text patch from the latest P8 findings. Every accepted patch returns to a fresh P8 audit; quality exhaustion keeps the best valid script."
    },
    {
      "pass_id": "P10_structured_repair",
      "slot": "technical",
      "seam_refs": ["codex_structured_repair_system"],
      "description": "Metadata-only repair seam: it supplies messages but makes no standalone call; shared repair reuses the originating pass slot and never changes model selection."
    },
    {
      "pass_id": "P11_pre_tail_assembly",
      "slot": "technical",
      "seam_refs": [],
      "description": "Validates and assembles the complete canonical ledger, then requires a warning-free Phase 0 report."
    },
    {
      "pass_id": "P12_post_tail_freeze",
      "slot": "technical",
      "seam_refs": [],
      "description": "Runs through the planned tail finalizer: warning-free Phase 10 before save, then a read-only persisted-ledger verification after save."
    }
  ],
  "notes": [
    "A0 is the first artifact and P0 is the first model call.",
    "P3 to P4 to P3 is bounded to one structural rewrite.",
    "P5 to P6/P7 and P8/P9 are bounded judge/line-patch loops; every accepted patch is independently rejudged and quality exhaustion keeps the best valid script.",
    "Every model invocation uses only the pass slot. P11 and P12 are deterministic orchestration stages."
  ]
}
~~~

### 4.3 nodes/story_packs/scifi_codex/scifi_codex_v1.json

The following is the complete planned pack. Its JSON strings intentionally contain all creative instructions and examples. The runner must pass seam text through verbatim; it may not supplement it with creative prose.

~~~json
{
  "source_bank_id": "scifi_codex",
  "story_model_id": "scifi_codex_v1",
  "story_pipeline_id": "scifi_codex_circuit",
  "schema_version": "v2.0",
  "label": "Sci-Fi Codex - Proof-Pressure Radio",
  "status": "planned",
  "prompt_stages": {
    "codex_fact_index_system": "You are the evidence editor for a science-fiction radio drama. Read only A0. Return FactIndexV1 JSON only. Extract the report's observable claims, named entities, numeric expressions, uncertainty language, and tonal register. Every fact and entity must carry literal source spans whose field, offsets, and quote exactly reproduce A0. A number may be omitted only when the report has none. Do not write dialogue, plot, imagery, or an inference that is absent from the source. A cautious source stays cautious.",
    "codex_question_system": "You are the dramatic architect. Return DramaticQuestionV1 JSON only. Build a listener-first question from P0 facts: a person must choose under a consequence that the science makes possible or dangerous. Facts are constraints, not a recap. Invented people, places, devices, and events are welcome only when each is plainly marked fictional in the artifact. All invented material must be SFW. Never turn a source uncertainty into certainty. The ending direction must force a decision before the factual coda.",
    "codex_pressure_cast_system": "You are casting a short radio drama under pressure. Return CastPlanV1 JSON only. Make the announcer a lucid witness, never a lecturer. Give each character a distinct practical desire and a disagreement that can be heard in choices, interruptions, silences, and consequences. Do not give two characters the same emotional job. Use only the prescribed IDs and voice slots. Keep all characterization SFW and playable by voice alone.",
    "codex_radio_score_system": "You are the radio score designer. Return the requested score or review JSON only. Every scene must change the listener's understanding through audible action. A fact belongs where it raises a cost, blocks an easy answer, or sharpens a choice. Plan scenes, shots, beats, and music so a sequencer receives complete IDs and references. The word plan is law: spoken words are character plus announcer text, while music rows are silent. Prefer cause and response over explanation. Let the laboratory, station, vessel, archive, or street be heard through decisions rather than exposition. All proposed material must be SFW.",
    "codex_coda_contract_system": "The ending coda is an earned radio report, not a moral. State only source-backed claims with P0 fact IDs and preserve the source's uncertainty. Distinguish the fictional incident from the real report without breaking the mood. Do not invent institutions, measurements, dates, outcomes, or quotations. The final factual note should make the listener look back at the drama differently.",
    "codex_play_system": "You are writing an original science-fiction radio play. Return one complete ScriptArtifactV1 JSON object, never a patch. Write every final character and announcer line verbatim, including the coda. Make speech active, specific, and speakable; give people competing aims rather than speeches about themes. Use sound and music rows as hinges, but leave their text empty. Keep each fact-derived claim mapped to P0 and let invented material behave as fiction. Count ordinary spoken words exactly; avoid dotted abbreviations, symbol-only quantities, isolated punctuation, and formatting tricks that make word counters disagree. Example of the desired turn, not reusable text: ANNOUNCER: The observatory reports a signal arriving before its own timestamp. IONA: Then stop listening for a message and start listening for the mistake. The second line answers the first with a choice, not a summary. All content must be SFW.",
    "codex_listening_room_system": "You are a demanding listener hearing P5 once in the dark. Return ListenerReviewV1 JSON only. Identify where causality blurs, where two voices sound interchangeable, where a science fact becomes a lecture, where sound could carry an action, where pacing stalls, or where the coda overclaims. Cite exact line IDs and issue directions that a writer can use for a total replacement. Name strengths honestly, but always require the P7 full retake.",
    "codex_retake_system": "You are rewriting for broadcast. Return one complete ScriptArtifactV1 JSON object, never edits, deletions, references to prior text, or an explanation. Absorb every applicable review issue while preserving valid source mappings and exact ledger IDs. Rewrite every character line and every announcer line as fresh final text. Announcer beats are part of the play and must be rewritten too. Keep science as the pressure behind action. The finish should feel inevitable in retrospect and leave the factual coda clean, modest, and source-bound. All content must be SFW.",
    "codex_final_audit_system": "You are the final continuity and evidence auditor. Return FinalAuditV1 JSON only. Verify the whole current script, not a sample. Check every spoken line, role, cast ID, beat and shot reference, music anchor, exact word count, fact mapping, fictional boundary, radio action, coda claim, and SFW compliance. Pass only if all checks are true and no issue remains. If a fact is audible but lacks a P0 mapping, if a fictional invention sounds reported as news, or if content is not SFW, require rewrite.",
    "codex_structured_repair_system": "You repair only the requested JSON contract. Return valid JSON for the same artifact type and no commentary. Preserve content that already satisfies the contract, but replace the entire artifact rather than emitting a patch. Correct missing keys, wrong types, bad IDs, broken references, invalid role values, invalid word totals, missing provenance maps, or SFW violations. Do not add creative material beyond what the failed artifact and supplied inputs require."
  }
}
~~~

### 4.4 Archived v3 runner/writer wiring skeleton — superseded by v4 signatures

This is an implementation skeleton, not code added by this task. It contains no creative wording; all prompt text is read from the pack above.

~~~python
# nodes/_otr_scifi_codex.py

GenerateFn = Callable[..., str]

@dataclass(frozen=True)
class ScifiTargetPlan:
    requested_audible_words: int
    character_words: int
    announcer_words: int

@dataclass(frozen=True)
class CodexCallReceipt:
    pass_id: str
    slot: Literal["creative", "technical"]
    call_id: str
    raw_response_utf8: str
    raw_response_sha256: str
    accepted_attempt: int

@dataclass
class CodexCallJournal:
    by_invocation_id: dict[str, list[CodexCallReceipt]]

@dataclass(frozen=True)
class ScifiCodexOutlineView:
    title: str
    premise: str

@dataclass(frozen=True)
class ScifiCodexTailParts:
    outline_view: ScifiCodexOutlineView
    canon: Any
    run_story_spine: bool
    final_title_override: str
    title_source_override: str
    tail_finalizer: TailFinalizer

def run_scifi_codex_episode(
    *,
    payload: dict[str, str],
    pack: Any,
    resolved: MutableMapping[str, Any],
    led: Ledger,
    meta: dict[str, Any],
    creative_fn: GenerateFn,
    technical_fn: GenerateFn,
    slot_scheduler: Any,
    source_bank_row: SourceBank,
    story_rules: Mapping[str, Any],
    episode_root: Path,
    episode_id: str,
) -> ScifiCodexTailParts: ...

def validate_payload_envelope(
    payload: Mapping[str, Any], resolved: Mapping[str, Any]
) -> PayloadEnvelopeV1: ...

def preflight_scifi_codex_pack(
    pack: Any, source_bank_row: SourceBank
) -> None: ...

def assert_scifi_codex_target_words(
    resolved: Mapping[str, Any]
) -> ScifiTargetPlan: ...

def invoke_codex_structured(
    *,
    pass_id: str,
    slot_fn: GenerateFn,
    system_seams: tuple[str, ...],
    artifact_input: Mapping[str, Any],
    result_type: type[T],
    post_validator: Callable[[T], str | None],
    base_temperature: float,
    structural_retry_temperature: float,
    max_new_tokens: int,
    call_journal: CodexCallJournal,
) -> T: ...

def codex_repair_factory(
    *, original_prompt: Any, failed_output: str, error: BaseException,
) -> list[dict[str, str]]: ...

def run_fact_index(
    envelope: PayloadEnvelopeV1, technical_fn: GenerateFn, pack: Any,
    call_journal: CodexCallJournal,
) -> FactIndexV1: ...

def run_dramatic_question(
    envelope: PayloadEnvelopeV1, fact_index: FactIndexV1,
    creative_fn: GenerateFn, pack: Any, call_journal: CodexCallJournal,
) -> DramaticQuestionV1: ...

def run_cast_plan(
    fact_index: FactIndexV1, question: DramaticQuestionV1,
    target_audible_words: int, creative_fn: GenerateFn, pack: Any,
    call_journal: CodexCallJournal,
) -> CastPlanV1: ...

def run_radio_score(
    fact_index: FactIndexV1, question: DramaticQuestionV1,
    cast_plan: CastPlanV1, target_audible_words: int,
    revision: StructureReviewV1 | None, creative_fn: GenerateFn, pack: Any,
    call_journal: CodexCallJournal,
) -> RadioScoreV1: ...

def run_structure_review(
    score: RadioScoreV1, technical_fn: GenerateFn, pack: Any,
    call_journal: CodexCallJournal,
) -> StructureReviewV1: ...

def run_script_artifact(
    *,
    pass_id: Literal["P5", "P7", "P9"],
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    cast_plan: CastPlanV1,
    score: RadioScoreV1,
    prior_script: ScriptArtifactV1 | None,
    review: ListenerReviewV1 | FinalAuditV1 | None,
    creative_fn: GenerateFn,
    pack: Any,
    call_journal: CodexCallJournal,
) -> ScriptArtifactV1: ...

def run_listener_review(
    script: ScriptArtifactV1, technical_fn: GenerateFn, pack: Any,
    call_journal: CodexCallJournal,
) -> ListenerReviewV1: ...

def run_final_audit(
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    score: RadioScoreV1,
    script: ScriptArtifactV1,
    technical_fn: GenerateFn,
    pack: Any,
    call_journal: CodexCallJournal,
) -> FinalAuditV1: ...

def validate_final_script(
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    cast_plan: CastPlanV1,
    score: RadioScoreV1,
    script: ScriptArtifactV1,
) -> None: ...

def build_cast_rows(cast_plan: CastPlanV1) -> list[dict[str, Any]]: ...
def build_scene_rows(score: RadioScoreV1) -> list[dict[str, Any]]: ...
def build_shot_rows(score: RadioScoreV1) -> list[dict[str, Any]]: ...
def build_beat_rows(score: RadioScoreV1) -> list[dict[str, Any]]: ...
def build_line_rows(script: ScriptArtifactV1) -> list[dict[str, Any]]: ...
def build_music_rows(score: RadioScoreV1, script: ScriptArtifactV1) -> list[dict[str, Any]]: ...
def build_tail_canon(score: RadioScoreV1, script: ScriptArtifactV1) -> Any: ...

def record_final_provenance(
    meta: dict[str, Any],
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    final_script: ScriptArtifactV1,
    final_receipt: CodexCallReceipt,
) -> None: ...

def strict_phase0(led: Ledger) -> PreTailAssemblyReportV1: ...

def assemble_pre_tail_ledger(
    *,
    led: Ledger,
    meta: dict[str, Any],
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    cast_plan: CastPlanV1,
    score: RadioScoreV1,
    final_script: ScriptArtifactV1,
) -> PreTailAssemblyReportV1: ...

def make_tail_finalizer(
    *,
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    final_script: ScriptArtifactV1,
) -> TailFinalizer: ...

def assert_final_provenance_and_fact_graph(
    led: Ledger,
    meta: Mapping[str, Any],
    envelope: PayloadEnvelopeV1,
    fact_index: FactIndexV1,
    final_script: ScriptArtifactV1,
) -> None: ...
~~~

run_scifi_codex_episode performs the following in order:

1. Calls `_otr_source_payload.validate_source_payload(payload, origin="scifi_codex")`, wraps its failure as CodexPayloadShapeError, derives source_mode only from resolved["seed_source"], applies the A0 thin and 48,000-UTF-8-byte rules, hashes A0, and writes A0 provenance to meta.scifi_codex. It rejects an unknown route instead of choosing another source.
2. Verifies bank/model/pipeline identities and exact seam set, derives ScifiTargetPlan from the incoming requested audible target, validates the fixed voice roster, requires the supplied meta to be a dictionary, and assigns led.data["meta"] = meta so both references name the same dictionary. Ledger initializes canonical top-level lists but not this metadata dictionary at nodes/production_ledger.py:551-579.
3. Runs P0 through P9 using only invoke_codex_structured. slot_scheduler is accepted solely for compatibility with the writer dispatch and is never used to obtain a model callable.
4. Copies the accepted P9 artifact strings unchanged into setter rows. It does not alter text, title, premise, traits, or music prompt strings. It may add structural metadata such as SHA-256 values and pass IDs outside spoken text.
5. Calls set_cast, set_scenes, set_shots, set_beats, set_lines, set_music, and stamp_word_counts in that order. set_lines recomputes cast and scene totals against existing cast/shot rows at nodes/production_ledger.py:1080-1168,1247-1283.
6. Sets meta.episode_id to episode_id, meta.episode_title to P9 title, and meta.audit_passes, meta.cleanup_passes, and meta.readiness_passes to lists. It stores every fact mapping, the exact accepted P9 raw response, raw-response hash, final text hash, line ID, and creative-call invocation ID under meta.scifi_codex. It then changes only the tail-facing resolved["target_words"] value to ScifiTargetPlan.character_words, while retaining the incoming audible request and both category targets under meta.scifi_codex. It runs strict Phase 0 and returns no tail parts unless its errors and warnings arrays are both empty. It does not claim `meta.style` is final: the tail's `contract=None, style_grammar_on=False` route removes it and stamps the explicitly Phase-10-exempt scaffold-off receipt at nodes/OTR_LedgerScriptWriter.py:2129-2147 and nodes/_otr_ledger_freeze.py:555-572.
7. Returns an outline view with P9 title/premise, a canon object made through build_tail_canon using the same outline-dictionary factory pattern that the writer uses with title, premise, setting, time_of_day, and sound_palette at nodes/OTR_LedgerScriptWriter.py:4383-4393, run_story_spine false, the nonempty P9 title as final_title_override, title_source_override="scifi_codex_script_title", and the TailFinalizer object. The title override avoids the writer's separate title-generation route at nodes/OTR_LedgerScriptWriter.py:6255-6305; run_story_spine false prevents a later content-spine mutation at nodes/OTR_LedgerScriptWriter.py:6540-6555.

The run functions above do not compose strings. They assemble the canonical typed JSON envelope defined by the v3 control plane and call invoke_codex_structured with the named seams and slot. It emits one system message formed only from the named seam strings in declared pipeline order, followed by the typed user JSON message; it never combines seam prose with Python-authored creative wording. Its per-invocation slot wrapper records every raw response in CodexCallJournal and retains the accepted response as CodexCallReceipt. validate_final_script checks every cross-artifact reference, row shape, word calculation, fact span, authorship declaration, music anchor, and no-orphan graph condition. The five build-row functions copy only already-authored artifact fields into the exact setter shapes described in section 6; build_line_rows deliberately copies text unchanged. record_final_provenance writes the accepted raw P9 response, hashes, and mapping metadata only. strict_phase0 invokes the live gap audit and turns either errors or warnings into CodexPreTailAuditError.

Planned additive writer edits:

~~~python
def _run_scifi_codex_lane(**kwargs):
    from ._otr_scifi_codex import run_scifi_codex_episode
    return run_scifi_codex_episode(**kwargs)

_RUNNER_BY_PIPELINE["scifi_codex_circuit"] = _run_scifi_codex_lane

def _run_writer_tail(
    self, ctx: WriterTailContext, *, tail_finalizer: TailFinalizer | None = None,
) -> tuple[Any, ...]:
    ...
    # After all existing tail mutations and checks, immediately before
    # script-text assembly and led.save():
    if (
        ctx.source_bank_row.default_story_pipeline == "scifi_codex_circuit"
        and tail_finalizer is None
    ):
        raise CodexTailFinalizerMissingError(...)
    if tail_finalizer is not None:
        tail_finalizer.before_save(ctx=ctx)
    script_text = _PL.assemble_script_text_from_ledger(led.data)
    saved_path = led.save()
    if not saved_path:
        raise CodexLedgerSaveError(...)
    if tail_finalizer is not None:
        tail_finalizer.after_save(saved_path=saved_path, ledger_data=led.data)
    script_json = json.dumps(led.data, indent=2, ensure_ascii=False)
~~~

The wrapper follows the existing late-import lane-wrapper pattern at nodes/OTR_LedgerScriptWriter.py:1569-1580; the new map key belongs alongside the current map at nodes/OTR_LedgerScriptWriter.py:1589-1591. The dispatch passes `getattr(_parts, "tail_finalizer", None)` and the optional title-source override after constructing the tail context at nodes/OTR_LedgerScriptWriter.py:3605-3624. All legacy tail callers retain the default None behavior. The finalizer runs after all tail LLM/provenance mutations, before save, and verifies the persisted data after save.

The planned writer gate is immediately before _resolve_inputs: for this pipeline only, reject target words outside 30-900 and reject any built-in refine activation. This prevents a second, framework-owned rewrite loop from changing the lane's bounded topology.

## 5. Historical v3 topology/budget notes — v4 control plane governs execution

### Source entry and routing

_fetch_rss_seed_or_die builds the seven-key RSS payload at nodes/OTR_LedgerScriptWriter.py:1081-1143. The direct runner dispatch receives the resolved payload at nodes/OTR_LedgerScriptWriter.py:3568-3601. This lane has a hard enabling precondition: its future wiring test must prove that the mapped direct dispatch bypasses the normal interpreter path when interpreter is empty. A pinned premise travels through the verified writer branch into A0 with source_mode operator_pinned. A malformed, empty, thin, or unknown-route payload stops before P0; it never causes a live RSS draw, a generated substitute, or an alternate bank.

### Call schedule

| Step | Input -> output | Slot | Outer bound | Failure raised |
|---|---|---:|---:|---|
| A0 | writer payload -> PayloadEnvelopeV1 | none | one validation | CodexPayloadShapeError, CodexPayloadThinError, or CodexPayloadRouteError |
| P0 | A0 -> FactIndexV1 | technical | one call ladder | StructuredCallFailedError |
| P1 | P0 -> DramaticQuestionV1 | creative | one call ladder | StructuredCallFailedError |
| P2 | P0,P1 -> CastPlanV1 | creative | one call ladder | StructuredCallFailedError or CodexVoiceInventoryError |
| P3a | P0,P1,P2 -> RadioScoreV1 | creative | one call ladder | StructuredCallFailedError |
| P4a | P3a -> StructureReviewV1 | technical | one call ladder | StructuredCallFailedError |
| P3b/P4b | only when P4a says rewrite | creative/technical | one complete rewrite and recheck | CodexStructureExhaustedError if P4b still says rewrite |
| P5 | P0-P3 plus one advisory word steer -> ScriptArtifactV4 | creative | one full draft | StructuredCallFailedError |
| P6 | P5 -> ListenerReviewV1 | technical | one diagnosis | StructuredCallFailedError |
| P7 | current script + P6 findings -> typed line-text patch -> validated merged ScriptArtifactV4 | creative, then technical on rejection | bounded dynamic repair cycle | best-valid quality floor |
| P8a | P0,P3,P7 -> FinalAuditV1 | technical | one audit | StructuredCallFailedError |
| P9/P8 repeat | current script + P8 findings -> typed line-text patch, then fresh audit | creative, then technical on rejection | bounded dynamic repair cycles | best-valid quality floor |
| P11 | P9 final -> canonical ledger and Phase 0 | none | one deterministic assembly | CodexPreTailAuditError, CodexLedgerReferenceError, or CodexProvenanceError |
| P12 | tail-mutated ledger -> Phase 10 | none | one final callback | FreezeAssertionError, CodexFreezeWarningError, or CodexTailFinalizerMissingError |

CodexArtifactValidationError is a PostValidationError subclass. A bad cross-reference, role, provenance map, word total, or text-shape field inside P0-P9 therefore enters the shared repair ladder rather than being corrected in Python. If the ladder exhausts, the existing StructuredCallFailedError is the exact terminal error at nodes/_otr_structured_call.py:89-117.

### Exact structured-call temperature ladder

Every P0-P9 structured call uses max_attempts=3. Existing behavior is: initial base temperature; a JSON syntax failure gets a same-prompt structural retry at the lower temperature; schema/post-validation failure skips that retry and gets a typed repair at fixed 0.10 through the same originating slot; exhaustion raises StructuredCallFailedError at nodes/_otr_structured_call.py:482-712. Transport or callable exceptions are not swallowed by that helper and propagate unchanged. Every repair factory returns only RepairMessageSetV1.messages; the originating slot response, not the factory, returns the repaired artifact.

| Pass | Base temp | Syntax retry temp | Repair temp | max_new_tokens |
|---|---:|---:|---:|---:|
| P0 | 0.22 | 0.12 | 0.10 | 1800 |
| P1 | 0.78 | 0.38 | 0.10 | 1200 |
| P2 | 0.76 | 0.36 | 0.10 | 2000 |
| P3 | 0.70 | 0.30 | 0.10 | 2500 |
| P4 | 0.22 | 0.12 | 0.10 | 1200 |
| P5 | 0.76 | 0.36 | 0.10 | 6000 |
| P6 | 0.22 | 0.12 | 0.10 | 2600 |
| P7 | 0.72 | 0.32 | 0.10 | 6000 |
| P8 | 0.22 | 0.12 | 0.10 | 2600 |
| P9 | 0.68 | 0.28 | 0.10 | 6000 |
| P10 | no standalone call | no standalone call | no standalone call | 0 |
| P11/P12 | no model call | no model call | no model call | 0 |

The lower retry temperatures satisfy the helper's strict-lower requirement at nodes/_otr_structured_call.py:547-580. A syntax-first failure can consume three calls; a schema/content-first failure consumes the base call plus typed repair under the helper's documented branch behavior.

## 6. Historical v3 assembly/proof notes — v4 control plane governs execution

### Complete hierarchy mapping

The canonical hierarchy is filled as cast -> scenes -> shots -> beats -> lines, with fully populated music and an explicit empty clips list. These are the required top-level lists at nodes/_otr_ledger_freeze.py:115-129.

Before P11, the runner explicitly requires every top-level list to exist, assigns clips to an empty list, and rejects any pre-existing non-list or nonempty clip value. Clip timing belongs to the sequencer; the story runner never fabricates clip records.

| Artifact source | Setter row | Required assembly rule |
|---|---|---|
| P2 plus fixed roster | set_cast | One announcer and two/three character rows. Every row has nonempty ID, name, description, gender, tts_model, voice_preset, empty-object voice params, and counters initialized before set_lines recomputes them. |
| P3 scenes | set_scenes | Each row is exactly scene_id, description, env, line_count 0, and word_count 0. The live setter field is env, not environment. |
| P3 shots | set_shots | Each row is exactly shot_id, scene_id, description, visual_prompt, png_path None, start_s None, and dur_s None. |
| P3 beats | set_beats | Each row is exactly beat_id, shot_id, scene_id, speaker, char_id, line_ids, start_s None, and dur_s None. P3 preplans line IDs, so this remains complete before P9 text arrives. |
| Final validated script lines | set_lines | Each row is exactly line_id, shot_id, beat_id, char_id, final text, traits, boundary, bark_wav_path None, start_s None, dur_s None, speaker_role, arc_phase, compose_flags, beat_intent, target_words, and dialogue_slot_id. The setter computes char_count and word_count. |
| P3 music plus final script anchors | set_music | Each row is exactly cue_id, description, generation_prompt, anchor_line_id, placement, target_duration_s, wav_path None, start_s None, and dur_s None. set_music has no text field at nodes/production_ledger.py:1179-1207. |

The only legal speaker roles are character, announcer, music_open, music_close, and music_inter at nodes/_otr_ledger_freeze.py:85-96. The final validated script uses announcer for announcer rows and the matching music sentinel names for music rows. Cast rows exist before set_lines, which allows the live role-coercion guard to confirm character IDs at nodes/production_ledger.py:91-157,1158-1166. The runner independently rejects any mismatch rather than relying on coercion.

### Verbatim authorship proof

The final P9 response is the only source of final spoken text. The runner records the accepted response once, then associates each spoken line with that immutable receipt:

~~~text
meta.scifi_codex.creative_calls[creative_call_id] = {
  pass_id: "P9",
  slot: "creative",
  raw_response_utf8: exact accepted P9 JSON response,
  raw_response_sha256: lowercase SHA-256,
  accepted_attempt: int
}

meta.scifi_codex.line_authorship[line_id] = {
  pass_id: "P9",
  slot: "creative",
  creative_call_id: UUID string,
  final_text_sha256: lowercase SHA-256,
  full_artifact_sha256: lowercase SHA-256
}
~~~

The stored final_text_sha256 must equal SHA-256 of the string copied into the ledger row. The finalizer hashes the raw receipt, strictly re-parses that exact receipt as P9, compares every spoken artifact string with its ledger row, and then recomputes each line hash. It rejects a missing receipt, a changed line, or a receipt/artifact mismatch. Since no Python operation alters a passed text string, every audible line is both LLM-authored verbatim and provable as the final P9 artifact.

### News-fact trace

For each P9 SpokenFactUseV1, the runner writes:

~~~text
meta.scifi_codex.news_fact_trace[line_id].append({
  fact_id: "Fxx",
  span_ids: ["Fxx.Sy", ...],
  source_digest: A0.source_digest,
  spoken_claim: exact ScriptArtifactV1 string,
  final_text_sha256: line_authorship[line_id].final_text_sha256
})
~~~

The finalizer verifies the source digest, fact ID, span ID, literal source slices, line ID, and final-text hash. P8 provides the semantic check that each audible source claim is faithfully bounded by that mapping; the deterministic finalizer proves that the audited mapping still belongs to the saved final line. No untraceable audible news claim can reach Phase 10.


## 7. 720-word steer strategy — v4

A 720-word bake-off selection is supplied as `target_words=720`; it is an
initial-draft aesthetic steer, not a contract that can fail a finished story.
The existing twelve-beat table remains useful prompt context: P3 can present
its values as advisory centers and P5 can use the one injected word steer to
choose how much air each turn receives. The runtime does not calculate a
character/announcer split, does not overwrite `resolved["target_words"]`, and
does not call P7/P9 because an actual count differs from 720. It records the
requested and actual split/ledger counts in `meta.scifi_codex.word_receipt`.
A valid, complete 719- or 721-word (or any other 30–900-steered) episode
continues through the same provenance and freeze gates. Tokenization safety is
still mandatory so Phase 10 sees matching stored/split counts; it is unrelated
to the requested value.

## 8. Validation gates and typed error taxonomy — v4

| Error type | Exact trigger | Recovery policy |
|---|---|---|
| `CodexPayloadShapeError` | Seven-key/source-mode validation fails | Stop before P0. |
| `CodexPayloadThinError` | RSS or pinned payload misses its stated minimum evidence | Stop; no alternate source. |
| `CodexPayloadOversizeError` | Canonical payload exceeds 48,000 UTF-8 bytes | Stop; do not truncate context. |
| `CodexPayloadRouteError` | `resolved["seed_source"]` is neither `rss_fetch` nor `custom_premise` | Stop. |
| `CodexPackContractError` | Bank/pipeline/pack ID or declared seam preflight mismatches | Stop before a model call. |
| `CodexTargetRangeError` | Requested steer is non-integer or outside 30–900 | Stop before P5. |
| `CodexVoiceInventoryError` | A fixed roster voice cannot resolve | Stop; no substitute voice. |
| `CodexSpokenTextError` | An audible line violates the reject-only TTS/receipt/roster hygiene rules | Return its full artifact to the originating structured-call repair; exhaustion is `StructuredCallFailedError`. |
| `CodexArtifactValidationError` | Any other P0–P9 typed/cross-reference violation | Return a validator string to the originating structured-call repair; exhaustion is `StructuredCallFailedError`. |
| `CodexFactTraceExhaustedError` | Two scoped P9 fact-trace repairs still fail P8 | Stop; no textual fallback. |
| `CodexStructureExhaustedError` / `CodexFinalAuditExhaustedError` | The documented non-length P3/P4 or P8/P9 review topology exhausts | Stop. |
| `CodexLedgerReferenceError` | Any hierarchy, legal role, cast/sentinel, skip, or music-anchor invariant fails | Stop before Phase 0. |
| `CodexPreTailAuditError` | Phase 0 reports any error or warning | Stop. |
| `CodexTailFinalizerMissingError` / `CodexLedgerSaveError` / `CodexSavedLedgerAuditError` | The shared finalizer is absent, save gives no path, or persisted ledger fails proof/audit | Stop; no output. |
| `FreezeAssertionError` / `CodexFreezeWarningError` | Phase 10 errors or does not yield `frozen_clean` with zero warnings | Stop. |

There is deliberately no word-budget error. Requested-versus-actual word
counts are receipt fields only.

## 9. Test plan for the implementation — v4

1. Parse all planned JSON with duplicate-key rejection, load it through the
   live routing/pack validators, and assert Codex/Gemini/Sonnet IDs, paths,
   seams, and bank insertion order are unique.
2. Exercise RSS and `custom_premise` entry fixtures through the writer map:
   assert the runner receives the exact seven-key payload, `seed_source`
   stays outside it, pinned input makes no fetch call, and malformed/thin
   inputs raise the typed errors.
3. Spy on P0–P9: only the injected creative/technical closure is called, the
   stated three-attempt ladder is used, P5 alone receives `WordSteerV4`, and
   P7/P9 never execute because of a count.
4. Feed valid 30, 720, and 900 steers with deliberately nonmatching valid
   final counts. Assert all ship to Phase 0/10 and the receipt records
   requested/split/ledger values without a target comparison.
5. Parameterize every prohibited spoken form (brackets, parentheses,
   markdown, role label, action cue, all-caps lexical word, own-name
   vocative, isolated numeral, quoted whole line). Assert post-validation
   returns an LLM-repair error and no Python alters the string. Assert an
   accepted fixture makes the tail's `:6006-6158` scrubs make zero changes.
6. Test cast labels/IDs, music sentinel mapping, post-`set_lines`
   `skip`/`tts_skip_reason` stamps, every music anchor, and a complete
   cast/scenes/shots/beats/lines/music/clips graph. Run Phase 0 and Phase 10
   with zero errors and warnings.
7. Corrupt one source span, fact ID, source slice, or final-line hash. Assert
   a P8 fact defect invokes bounded scoped P9 line patches, only targeted text
   may differ, every accepted merge is freshly audited, and two failed patch
   slots keep the best already-valid script with an explicit quality receipt.
8. Run the shared TailFinalizer around the real tail. Assert receipt identity
   survives tail and saved JSON, UTF-8/no-BOM is preserved, and no output
   returns on a warning/error/missing callback/save-path failure.

## 10. v4 convergence self-audit

| Hard contract | Verdict | v4 evidence |
|---|---|---|
| Additive-only | PASS | New bank/pack/runner and optional shared finalizer are additive; no workflow/fetcher/network path changes are implied. |
| Payload first + pinned support | PASS | The exact seven-key RSS/custom-pinned payload reaches the map runner before P0; malformed/thin inputs stop typed. |
| Two slot callables only | PASS | Every model call receives the writer-injected creative or technical closure through `structured_call`. |
| Complete ledger / legal roles / freeze | PASS | v4 validates all five hierarchy tables, music skip stamps/anchors, and warning-free Phase 0/10 plus saved JSON. |
| Verbatim LLM dialogue | PASS | P5 and accepted P7/P9 text patches are model-authored; final receipt/hash proof covers the post-quality artifact, in-memory rows, and saved text. |
| News traceability | PASS | Exact P0 slices plus scoped P9 fact patches and the final fact graph cover every audible source fact. |
| 720 strategy | PASS | 720 is P5-only advisory steering; actual split/ledger counts are recorded and never gate acceptance. |
| No Python text surgery / fail loud / SFW / UTF-8 no BOM | PASS | Improvement is LLM repair only; failures are typed; the unchanged pack keeps SFW instructions; encoding is tested. |
## 11. Open questions for the operator

1. Installed voice inventory remains a build-time fact outside the five
   inspected contracts. `bm_george` and each `v2/en_speaker_*` value must
   resolve before this bank becomes runnable; a missing value raises
   CodexVoiceInventoryError and does not authorize a replacement.

## 12. Historical v3 technical correction ledger — superseded by v4

- Added the parser-required `guide_ref`, retained `runnable:false` /
  `executable:false`, and specified the same-change activation because routing
  validates this pairing at `nodes/_otr_story_routing.py:192-305,409-442`.
- Corrected dispatch/payload facts against
  `nodes/OTR_LedgerScriptWriter.py:3455-3471,3568-3624`; only injected slot
  closures are legal and a pinned premise already has a seven-key route.
- Corrected `post_validator` and repair-factory signatures to the actual
  `str | None` / keyword-only contract at
  `nodes/_otr_structured_call.py:120-173,482-712`, and made schema delivery,
  receipts, and no-text-truncation explicit.
- Corrected stored voice namespaces, roles, music sentinels, boundaries,
  setter order, no-orphan graph, and source metadata against
  `nodes/production_ledger.py:216-219,792-1207` and
  `nodes/_otr_ledger_freeze.py:85-96,291-529`.
- Corrected 720 math from an ambiguous four-total shorthand to the exact
  12-beat/33-line table and dual split/regex gate, with `stamp_word_counts`
  required by `nodes/production_ledger.py:479-514`.
- Corrected finalization from a pre-save callback claim to a backward-compatible
  tail finalizer plus post-save audit because `Ledger.save()` can return None
  and replaces `led.data` after merge (`nodes/production_ledger.py:1287-1354`).
