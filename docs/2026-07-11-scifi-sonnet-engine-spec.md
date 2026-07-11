# scifi_sonnet — CODE-READY v4 -- technical rewrite by Codex GPT-5, creative content unmodified

Written blind and standalone. This final pass changes technical contracts only;
the Continuity Archive philosophy, prompt voice, examples, and creative
topology remain preserved.

## Revision log

- **v4 final technical rewrite:** replaces stale runner/skeleton/calibration
  material with one build-ready control plane; merges only Warden framing while
  preserving both prompt blocks and split result schemas; removes all target
  count gates; and applies the source, cast, music, finalizer, and clean-path
  corrections cited below.


## v4 technical control plane — sole implementation authority

This final technical rewrite replaces all prior v1–v3 schemas, skeletons,
sidecar/pin proposals, calibration loops, and technical diagrams. The
Continuity Archive philosophy, named creative roles, pack wording, examples,
and story topology are preserved; only technical contradictions are corrected.

### v4 revision log

- Corrects the source route to the live seven-key payload handoff at
  `nodes/OTR_LedgerScriptWriter.py:1081-1143,3590-3624`; pin state lives only
  in `resolved["seed_source"]` (`:1321-1335`).
- Makes `target_words` a one-time P1 steer and final receipt value, never a
  gate, reroll, calibration, trim, or tail mutation.
- Adds reject-only spoken-text/roster validation before the tail’s own
  scrubs/attribution mutations (`OTR_LedgerScriptWriter.py:6006-6158`) and
  post-`set_lines` music skip stamps (`production_ledger.py:1080-1167`).
- Merges only Warden framing into one declared seam with two verbatim blocks;
  `WardenChallenge` and `WardenSatisfied` remain separate schemas/paths.
- Uses the shared TailFinalizer named in the Codex v4 specification, not a
  divergent Sonnet callback.

### Registry, source entry, and LLM access

IDs are `scifi_sonnet`, `sonnet_archive_multipass`, and `scifi_sonnet_v1`,
with pack directory `nodes/story_packs/scifi_sonnet/`. Catalog entries remain
`runnable:false` / `executable:false` until implementation. The additive
wrapper goes in `_RUNNER_BY_PIPELINE`
(`nodes/OTR_LedgerScriptWriter.py:1589-1617`) and both booleans flip together.
Because the bank has an empty interpreter, `requires_source_contract:false`
is correct; routing then requires `executable:true` once runnable
(`nodes/_otr_story_routing.py:422-442`). Banks insert Codex, Gemini, Sonnet
immediately before `custom_source_bank`. No workflow, fetcher, network,
environment variable, sidecar, or fallback is authorized.

`validate_sonnet_payload` calls
`_otr_source_payload.validate_source_payload(payload, origin="scifi_sonnet")`
and requires exactly `headline`, `summary`, `full_text`, `source`, `date`,
`link`, `seed_text`, all strings (`nodes/_otr_source_payload.py:80-133`).
The map runner receives `dict(resolved["news_article"])`, never fetches, and
derives `source_mode` only from `resolved["seed_source"]`. `"custom_premise"`
is the pinned route; science_rss stamps `"rss_fetch"`
(`nodes/_otr_source_payload.py:424-434`). Other stamps are
`SonnetPayloadRouteError`; malformed/empty is
`SonnetPayloadContractError`; RSS below 80 split words/12 distinct
alphanumeric tokens or pinned input below 8/4 is `SonnetThinPayloadError`.

Only injected `creative_fn` and `technical_fn` may make model calls. Every
call uses `structured_call` with strict model type, `post_validator` returning
`None | str`, and the keyword-only repair-factory contract
(`nodes/_otr_structured_call.py:147-173,482-535`). It gets base, lower
JSON-syntax retry, then .10 typed repair on schema/post-validator failure
(`:547-712`); no Python creates or patches dialogue.

### One-use word steer and exact artifact schemas

Validate `resolved["target_words"]` once as integer 30–900:
`WordSteerV4={requested_words:int}`. It appears only in P1 typed
`initial_session_word_steer`; P2–P7, repair calls, assembly, and tail receive
no target/delta. Existing frozen seam count wording stays creative prompt
text, not a validator. Record only:

~~~text
meta.scifi_sonnet.word_receipt = {
  requested_words: int,
  actual_split_words: int,
  actual_ledger_word_count: int
}
~~~

A valid nonmatching count ships. Tokenization safety remains because
`set_lines` stores regex count while freeze compares split count
(`nodes/production_ledger.py:216-219,1143-1147`;
`nodes/_otr_ledger_freeze.py:372-388`): reject isolated numerals/symbol-only
tokens and require numbers spoken in words. It is never compared with the
request.

All result types are strict `extra="forbid"`, no aliases or text min/max
limits (tolerant validation clips constrained text at
`nodes/_otr_structured_call.py:322-474`):

~~~text
PayloadV4 = {payload:SourcePayload7,
             source_mode:Literal["rss","operator_pinned"],
             payload_sha256:LowercaseSha256}
SourceSpanV4 = {field:Literal["headline","summary","full_text","seed_text"],
                start:int>=0,end:int>start,quote:Text}
EvidenceFactV4 = {fact_id:"fact_0".."fact_11",claim:Text,
                  source_spans:list[SourceSpanV4]}
EvidenceNumberV4 = {number_id:"num_0".."num_11",verbatim:Text,
                    fact_id:str,source_span:SourceSpanV4}
EvidenceEntityV4 = {entity_id:"entity_0".."entity_11",name:Text,
                    source_spans:list[SourceSpanV4]}
FragmentDossierV4 = {verified_facts:list[EvidenceFactV4](1..12),
 key_numbers:list[EvidenceNumberV4](0..12),
 named_entities:list[EvidenceEntityV4](0..12),tone:Text,
 headline_clean:str,provenance_note:str,payload_sha256:LowercaseSha256}
SessionFrameV4 = {session_title:Text,session_premise:Text,
 registrar_cold_open:Text,orum_register:Text,thessaly_register:Text,
 vesh_register:Text,scene_description:Text,scene_env:Text,
 shot_description:Text,visual_prompt:Text,music_description:Text,
 music_generation_prompt:Text}
CitedLineV4 = {text:Text,cites:list["fact_N"|"num_N"](1..3)}
AuditVerdictV4 = {status:Literal["clear","defect"],defects:list[Text](0..5),
 flagged_line_refs:list[int](0..5),invented_fact_flags:list[int](0..5),
 severity:Literal["critical","advisory"],sfw_pass:bool}
WardenChallengeV4 = {vesh_objection:Text,registrar_reopening:Text}
WardenSatisfiedV4 = {vesh_satisfied:Text}
RewriteResultV4 = {corrected_lines:list[{line_ref:int,
 speaker:Literal["ORUM","THESSALY"],text:Text,cites:list[str](1..3)}](1..6),
 vesh_resolution:Text}
AttestationV4 = {attestation:Text,attestation_cites:list[str](1..4),
                 vesh_final_seal:Text,sign_off:Text}
CalibrationEditV4 = {line_ref:int,text:Text,cites:list[str](1..3)}
DraftLineV4 = {text:Text,cites:list[str],
 speaker:Literal["ANNOUNCER","ORUM","THESSALY","VESH"],
 char_id:Literal["announcer","c02","c03","c04"],source_pass:str}
~~~

P0 requires `quote == payload[field][start:end]`, exact sequential IDs, and
valid references. P1 supplies all creative title/premise/setting/visual/music
and description material. `CastLockV4` is mechanical, with no authored
fallback: ANNOUNCER uses `session_premise` as `character_description` and
`kokoro`/`bm_george`; ORUM uses `orum_register`, c02,
`bark`/`v2/en_speaker_6`; THESSALY uses `thessaly_register`, c03,
`bark`/`v2/en_speaker_3`; VESH uses `vesh_register`, c04,
`bark`/`v2/en_speaker_0`. This gives `set_cast` nonempty descriptions
(`nodes/production_ledger.py:792-835`) without adding a creative field.
Every fixed voice must resolve or `SonnetVoiceInventoryError` stops.

The frozen P2 prompts emit `{text,cites}` only. The runner stamps ORUM/c02
for P2a and THESSALY/c03 for P2b into internal `DraftLineV4`; it never
invents a speaker. P5 retains its frozen speaker property but rejects any
value that is not exactly the locked original line speaker. P3 clear means
empty defect lists plus SFW; every defect, advisory or critical, uses the
defect Warden block/P5; only clear uses the clear block. The merged seam is
partitioned at static mode markers and selects a complete verbatim preserved
block before invoking `WardenChallengeV4` or `WardenSatisfiedV4`.

P5 may replace only flagged ORUM/THESSALY lines; all others remain
byte-identical. P6 attestation citations resolve. P7 remains a declared
frozen seam but has **no v4 runtime edge**: it is never invoked to chase,
trim, pad, or calibrate count.

For every audible P1/P2/P4/P5/P6 field, reject—not strip—newline/tab,
brackets, parentheticals, markdown/backticks/asterisks/fences, a leading
speaker/role label, standalone stage/action cue, wholly quoted text,
all-caps lexical word of two+ letters, or self-name vocative. Acronyms use
lowercase spoken form (for example `nasa`). A validator string sends the
whole originating artifact through LLM repair; Python never changes it.

### Topology, ledger, proof, and runner skeleton

| Pass | Slot | bound |
|---|---|---|
| P0 intake | technical | .20/.10/.10, 2,000 tokens, one ladder |
| P1 frame | creative | .85/.40/.10, 2,300; only word-steer input |
| P2a literalist | creative | 2 calls, .55/.25/.10, 2,200 each |
| P2b speculator | creative | 2 calls, .78/.35/.10, 2,600 each |
| P3 audit | technical | .25/.12/.10, 2,000; after P2/P5 |
| P4 Warden | creative | defect .70/.30/.10 or clear .60/.25/.10, 1,400 |
| P5 rewrite | creative | .45/.20/.10, 3,000; at most two defect rounds |
| P6 attestation | creative | .50/.22/.10, 2,600; once after clear |
| P7 calibration | creative | declared, not dispatched in v4 |
| P8/P9 | technical | deterministic, no LLM |

Flow: P0→P1→P2a/P2b→P3. Clear invokes WardenSatisfied then P6; defect
invokes WardenChallenge→P5→P3, maximum two content rewrites then
`SonnetAuditExhaustedError`. No length branch exists. Mandatory first
regression: `AuditVerdictV4(status="clear", defects=[], flagged_line_refs=[])`
uses only the clear block/WardenSatisfied schema—never WardenChallenge/P5 or
the defect schema.

~~~python
def run_scifi_sonnet_episode(*,payload:dict[str,str],pack:Any,
 resolved:Mapping[str,Any],led:Ledger,meta:dict[str,Any],
 creative_fn:GenerateFn,technical_fn:GenerateFn,slot_scheduler:Any,
 source_bank_row:SourceBank,story_rules:Mapping[str,Any],
 episode_root:Path,episode_id:str)->SonnetTailParts: ...
def validate_sonnet_payload(payload:Mapping[str,Any],
 resolved:Mapping[str,Any])->tuple[PayloadV4,WordSteerV4]: ...
def invoke_sonnet_structured(*,pass_id:str,
 slot:Literal["creative","technical"],slot_fn:GenerateFn,seam_ref:str,
 typed_inputs:Mapping[str,Any],result_type:type[T],
 post_validator:Callable[[T],str|None],base_temperature:float,
 structural_retry_temperature:float,max_new_tokens:int,
 journal:SonnetCallJournal)->T: ...
def select_warden_mode_block(merged_seam:str,
 status:Literal["clear","defect"])->str: ...
def lock_archive_cast(frame:SessionFrameV4)->Mapping[str,CastLockV4]: ...
def validate_spoken_text_and_lock(events:Sequence[DraftLineV4],
 cast_lock:Mapping[str,CastLockV4])->None: ...
def stamp_music_skip_contract_after_set_lines(led:Ledger,
 music_line_ids:Sequence[str])->None: ...
def record_word_receipt(meta:MutableMapping[str,Any],requested_words:int,
 led:Ledger)->None: ...
~~~

Assembly order is `set_cast`, `set_scenes`, `set_shots`, `set_beats`,
`set_lines`, post-setter music skip stamps, `set_music`,
`stamp_word_counts`; `clips=[]`. Music uses matching sentinel role/ID,
`skip:true`, `text:""`, `tts_skip_reason:"music_cue"` after the setter. It is
not cast, because all-skipped cast rows produce a warning
(`nodes/_otr_ledger_freeze.py:505-529`). Character/announcer IDs resolve to
the locked cast. Every music cue anchors a non-skipped voiced line; only the
five legal roles in `_otr_ledger_freeze.py:85-96` are valid.

Persist exact P0 spans, citations, raw creative receipts, per-line raw/text
hashes, and call journal under `meta.scifi_sonnet` because `set_lines` drops
per-line provenance (`nodes/production_ledger.py:1138-1157`). Every audible
source fact maps to P0; structural Warden text may not state a source fact
without citation. Before tail, reparse receipts and compare every non-skipped
line byte-for-byte. The shared Codex v4 TailFinalizer runs warning-free Phase
0/10 after tail mutation and reopens saved UTF-8 JSON for the same proof;
`Ledger.save()` merges/replaces `led.data`
(`nodes/production_ledger.py:1287-1354`). It never edits text.

Terminal errors are `SonnetPayloadContractError`, `SonnetPayloadRouteError`,
`SonnetThinPayloadError`, `SonnetTargetRangeError`, `SonnetPackContractError`,
`SonnetPassError`, `SonnetAuditExhaustedError`, `SonnetVoiceInventoryError`,
`SonnetCanonBuildError`, `SonnetSpokenTextError`, `SonnetLedgerContractError`,
`SonnetVerbatimProofError`, `SonnetCompletenessError`,
`SonnetPreTailAuditError`, `SonnetTailFinalizerMissingError`,
`SonnetLedgerSaveError`, and `SonnetSavedLedgerAuditError`;
`FreezeAssertionError` propagates. There is no word-budget/calibration error,
fallback, warning acceptance, or Python text surgery.


## 1. Design philosophy — why it wins a blind listen

Every one of the existing multipass banks hides its fact-checking machinery.
`legacy_many_pass` interprets a news article into hidden "briefs" before any
dialogue exists (`nodes/story_packs/pipelines.json:11` `source_interpret`).
Whatever the rival LLM-first bank does with its own critic/revision passes,
the shared house law across both existing multipass lanes is the same: the
audit is plumbing, not content — a hidden gate the listener never hears.

**scifi_sonnet inverts that.** The premise: a broadcast institution centuries
after our own — **the Continuity Archive** — periodically recovers one
surviving fragment of "Instrumental Age" (our present) science writing and
dramatizes the act of authenticating it, live, for a public that has mostly
forgotten how any of it worked. Every episode is one **Recovery Session**.
Three standing Archive functionaries argue over the fragment on-air:

- **ORUM, the Literalist** — trusts nothing but the fragment's exact wording;
  will only speak what the text supports.
- **THESSALY, the Speculator** — the drama's actual science-fiction engine;
  reads consequence and implication into everything the fragment leaves open.
- **VESH, the Provenance Warden** — the skeptic; will not let a claim stand
  until it has been checked against the fragment twice.
- **ANNOUNCER** (in-fiction title: the Registrar) — presides, frames stakes,
  and delivers the closing Attestation.

The mechanical requirement buried in this task — extract facts, draft,
audit for invented/uncited claims, conditionally rewrite, re-audit, calibrate
length — is not hidden behind the scenes here. **It IS the scene.** VESH's
objection and the Registrar's on-air "the record does not yet hold, return to
the fragment" are the literal, audible surface of the technical audit/rewrite
loop (Section 3). When the first draft is clean, VESH gets one satisfied
line and the session is a smooth reading. When the audit catches a defect,
the correction happens live, on mic, as the actual content. **The retry
budget is not overhead I have to hide — it is the show.**

This also produces two properties a blind listener will register without
being able to name: (1) variable shape — some sessions are a calm three-voice
reading, others are a live correction, and which one you get is honestly
determined by the source material, not a coin flip; (2) every fact the
audience hears is spoken by a character whose entire personality is "I only
say what the record supports," which makes the news-grounding feel like
characterization instead of a disclaimer.

**What changed after r2-r4 hardening, and why it still wins blind:** none of
the above changed. Hardening only replaced every "the LLM should roughly do
X" sentence with a schema field, a numeric bound, and a python-enforced
`post_validator` that rejects an instance violating X. A blind listener
cannot hear a pydantic field — but they can hear the difference between a
show where "never invent a number" is a hope and one where it is a
code-checked gate before that line can ever reach a speaker's mouth.

---


## 2. The four artifacts

### 2.1 `banks.json` row

Inserted after `scifi_fable2`'s closing `},` and before the `custom_source_bank`
row, i.e. between `nodes/story_packs/banks.json:185` and `:186`. Field set
verified against `_BANK_KEYS` at `nodes/_otr_story_routing.py:192-196`.
Unchanged since v1 — r2 re-verification found nothing to fix here.

```json
{
  "source_bank_id": "scifi_sonnet",
  "label": "Sci-Fi Sonnet (Continuity Archive multipass)",
  "source_kind": "article",
  "interpreter": "",
  "fetcher": "science_rss",
  "default_story_model": "scifi_sonnet_v1",
  "default_story_pipeline": "sonnet_archive_multipass",
  "defaults": {
    "story_form_label": "science-fiction audio drama",
    "source_material_label": "Science story",
    "title_form_label": "science-fiction radio drama",
    "coda_mode": "real_news_report",
    "credits_source_line": "dramatized by machine from a recovered instrumental-age transmission"
  },
  "required_seams": [],
  "runnable": false,
  "guide_ref": "Archive frame-narrative multipass lane: the intake pass IS the interpretation (empty interpreter, mirrors scifi_fable2's own contract at nodes/story_packs/banks.json:171). The retry loop is diegetic -- the Warden's objection and the Registrar's on-air reopening ARE the technical audit/rewrite cycle, not a hidden gate. Dossier is stamped at meta.sonnet_dossier, never meta.news (avoids a false gen_params_by_phase.news_interpreter stamp, nodes/OTR_LedgerScriptWriter.py:6390-6394). No fallback to legacy_many_pass, ever."
}
```

Why each field is what it is, with citations:

- `interpreter: ""` — the task's own spec allows this ("if your first pass IS
  the interpretation"); `scifi_fable2`'s row uses the identical convention
  (`nodes/story_packs/banks.json:171`), and the registry only requires BOTH
  fetcher and interpreter to be non-empty when the bank's pipeline sets
  `requires_source_contract: true` (`nodes/_otr_story_routing.py:427-435`) —
  which I do not set (2.2).
- `fetcher: "science_rss"` — reused verbatim; resolved through
  `_otr_source_payload.resolve_fetcher` (`nodes/_otr_source_payload.py:461-478`),
  which forwards to `_fetch_science_rss` (`nodes/_otr_source_payload.py:265-280`)
  and ultimately `OTR_LedgerScriptWriter._fetch_rss_seed_or_die`
  (`nodes/OTR_LedgerScriptWriter.py:1081`, calling
  `story_orchestrator._fetch_science_news` at `:1105-1108`). I add no new
  fetcher code (the pin patch in 3.6 is proposed, not applied).
- `required_seams: []` — my pipeline's `passes[]` reference only my OWN
  `declared_seams` (2.2), never the legacy `PRODUCTION_SEAM_ALLOWLIST`
  (`nodes/_otr_story_pack.py:27-44`). Same posture as `scifi_fable2`
  (`nodes/story_packs/banks.json:182`).
- `defaults` — the minimal five-key shape the task specifies, matching
  `scifi_fable2`'s own defaults block exactly in key-shape
  (`nodes/story_packs/banks.json:175-181`).
- `runnable: false` — fixed by the task; no code ships with this spec.

### 2.2 `pipelines.json` entry

Inserted between `scifi_fable2_multipass`'s closing `},` and
`simple_4_prompt_experimental`, i.e. between `nodes/story_packs/pipelines.json:87`
and `:88`. Field set verified against `_PIPELINE_KEYS` /
`_PASS_KEYS` at `nodes/_otr_story_routing.py:198-203`. Structurally
unchanged since v1 (r1's programmatic cross-check against the real parser
rules — key sets, seam-reference closure, no allowlist overlap — already
came back clean; re-run again in Section 9 after all r2-r4 edits).

```json
{
  "story_pipeline_id": "sonnet_archive_multipass",
  "label": "Sonnet Archive Multipass (Continuity Archive frame narrative)",
  "executable": false,
  "requires_source_contract": false,
  "declared_seams": [
    "sonnet_intake_system",
    "sonnet_frame_system",
    "sonnet_literalist_system",
    "sonnet_speculator_system",
    "sonnet_audit_system",
    "sonnet_warden_system",
    "sonnet_rewrite_system",
    "sonnet_attestation_system",
    "sonnet_calibration_system"
  ],
  "passes": [
    {
      "pass_id": "provenance_intake",
      "slot": "technical",
      "seam_refs": [
        "sonnet_intake_system"
      ],
      "description": "P0: raw science_rss payload -> FragmentDossier (verified_facts, key_numbers, named_entities, tone). Typed fail-loud on thin/malformed/empty. base_temp=0.20."
    },
    {
      "pass_id": "session_frame",
      "slot": "creative",
      "seam_refs": [
        "sonnet_frame_system"
      ],
      "description": "P1: dossier + the one advisory word steer -> SessionFrame (session title+premise, Registrar cold-open, the 3 Reliquarian registers). base_temp=.85."
    },
    {
      "pass_id": "literalist_reading",
      "slot": "creative",
      "seam_refs": [
        "sonnet_literalist_system"
      ],
      "description": "P2a (x2 calls): dossier + frame -> ORUM's citation-tagged lines (cites[] into the fact_N/num_N id space). base_temp=0.55."
    },
    {
      "pass_id": "speculator_extrapolation",
      "slot": "creative",
      "seam_refs": [
        "sonnet_speculator_system"
      ],
      "description": "P2b (x2 calls): dossier + frame + Orum's lines -> THESSALY's extrapolation lines. May not introduce a fact/number outside the dossier. base_temp=0.95."
    },
    {
      "pass_id": "provenance_audit",
      "slot": "technical",
      "seam_refs": [
        "sonnet_audit_system"
      ],
      "description": "P3: draft + dossier + precomputed citation coverage -> AuditVerdict. Runs after P2 and after P5 content rewrites; never for requested length. base_temp=.25."
    },
    {
      "pass_id": "warden_challenge",
      "slot": "creative",
      "seam_refs": [
        "sonnet_warden_system"
      ],
      "description": "P4: one mode-selected seam with distinct schemas. defect selects preserved WardenChallenge block; clear selects preserved WardenSatisfied block. Fires after every P3 result."
    },
    {
      "pass_id": "rewrite",
      "slot": "creative",
      "seam_refs": [
        "sonnet_rewrite_system"
      ],
      "description": "P5 (defect path only): draft + AuditVerdict + Warden objection -> corrected ORUM/THESSALY lines, restricted to flagged_line_refs. base_temp=0.45."
    },
    {
      "pass_id": "attestation",
      "slot": "creative",
      "seam_refs": [
        "sonnet_attestation_system"
      ],
      "description": "P6: dossier + final draft -> Registrar's closing citation-anchored Attestation + VESH's final seal + sign-off. base_temp=0.50."
    },
    {
      "pass_id": "word_calibration",
      "slot": "creative",
      "seam_refs": [
        "sonnet_calibration_system"
      ],
      "description": "P7: frozen declared seam; v4 gives it no target/count-triggered runtime edge."
    },
    {
      "pass_id": "assemble",
      "slot": "technical",
      "seam_refs": [],
      "description": "P8: pure Python, no LLM -- slot is registry metadata only. Final converged DraftScript -> ledger rows via set_cast/set_scenes/set_shots/set_beats/set_lines/set_music + EpisodeCanon. Runs exactly once, after the loop converges."
    },
    {
      "pass_id": "seal_audit",
      "slot": "technical",
      "seam_refs": [],
      "description": "P9: pure Python, no LLM. Deterministic trace, tokenization safety, no-orphan, authorship, and shared finalizer preparation."
    }
  ],
  "notes": [
    "Visible experiment; failure NEVER routes to legacy_many_pass.",
    "requires_source_contract is false because the bank's interpreter is intentionally empty (P0 IS the interpretation) -- same registry technicality scifi_fable2 relies on (nodes/_otr_story_routing.py:427-435).",
    "executable stays false until nodes/_otr_scifi_sonnet.py and the _RUNNER_BY_PIPELINE registration ship in the same change (registry law, nodes/OTR_LedgerScriptWriter.py:1610-1617).",
    "kibitz r2 (Claude Code CLI, SHOULD-FIX #4): when bank.runnable flips to true, executable MUST flip to true in the SAME change -- nodes/_otr_story_routing.py:436-442 raises RegistryValidationError if requires_source_contract=false AND executable is still false. Not just a bank.json + runner change; pipelines.json's own executable flag is a separate, easy-to-miss flip.",
    "The audit/rewrite loop is diegetic: VESH's objection and the Registrar's reopening beat ARE the P3-fail path, not a hidden gate (Section 1, Section 3).",
    "Structured calls use the shared per-call three-attempt ladder; v4 has at most two P3 defect/P5 rewrite rounds and no count-calibration route."
  ]
}
```

### 2.3 Seam list with purposes

All nine seams live in the pack's `prompt_stages` (2.4); none overlap
`PRODUCTION_SEAM_ALLOWLIST` (`nodes/_otr_story_pack.py:27-44`) so a validation
run through `load_pack_with_seams` (`nodes/_otr_story_pack.py:202-219`) checks
them against `declared_seams` only, exactly as `_otr_story_routing._parse_pipeline`
requires (`nodes/_otr_story_routing.py:257-270`).

| Seam | Pass | Slot | Purpose |
|---|---|---|---|
| `sonnet_intake_system` | P0 | technical | Extract facts/numbers/entities/tone from the raw payload into a strict schema; never paraphrase creatively. |
| `sonnet_frame_system` | P1 | creative | Name this session, write the Registrar's cold open + premise, set the three Reliquarians' registers for tonight. |
| `sonnet_literalist_system` | P2a | creative | ORUM's citation-anchored readings; every line must be traceable to a specific dossier fact/number id. |
| `sonnet_speculator_system` | P2b | creative | THESSALY's extrapolation; dramatic license on consequence, zero license on new facts/numbers. |
| `sonnet_audit_system` | P3 | technical | Judge invented-fact risk and internal contradiction the deterministic `cites[]` check cannot see. |
| `sonnet_warden_system` | P4 | creative | One mechanically selected preserved defect/clear block: `WardenChallenge` on defect, `WardenSatisfied` on clear. |
| `sonnet_rewrite_system` | P5 | creative | Targeted correction of only the flagged lines, preserving everything the audit already cleared. |
| `sonnet_attestation_system` | P6 | creative | The Registrar's closing, citation-anchored factual read -- the episode's dramatic resolution, not a preamble. |
| `sonnet_calibration_system` | P7 | creative | Frozen declared seam; v4 deliberately has no target/count-triggered runtime edge. |

### 2.4 Pack JSON — `nodes/story_packs/scifi_sonnet/scifi_sonnet_v1.json`

Top-level shape verified against `_REQUIRED_TOP_LEVEL` and the optional-field
allowlist at `nodes/_otr_story_pack.py:46-52,157-165`. **Dossier citation id
scheme (canonical, referenced by every seam below):** a `FragmentDossier`'s
`verified_facts` entries are addressable as `fact_0 .. fact_{len-1}` and
`key_numbers` entries as `num_0 .. num_{len-1}`, in list order. The runner
(2.6) renders these indices into every downstream prompt (e.g. `fact_0: <text>`
per line) — assembling that numbered context is Python orchestration, not
creative wording. `examples[]` now populated per r3.

```json
{
  "source_bank_id": "scifi_sonnet",
  "story_model_id": "scifi_sonnet_v1",
  "story_pipeline_id": "sonnet_archive_multipass",
  "schema_version": "v2.0",
  "label": "Continuity Archive v1",
  "status": "spec_only",
  "prompt_stages": {
    "sonnet_intake_system": "You extract a strict factual dossier from one science-news article for a far-future frame narrative. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"verified_facts\": array of 1-12 objects, each {\"fact_id\": \"fact_N\", \"claim\": ONE atomic checkable claim, \"source_spans\": [objects {\"field\": headline|summary|full_text|seed_text, \"start\": integer, \"end\": integer, \"quote\": exact source slice}]},\n  \"key_numbers\": array of 0-12 objects, each {\"number_id\": \"num_N\", \"verbatim\": number/quantity/date AS WRITTEN, \"fact_id\": \"fact_N\", \"source_span\": exact source-span object},\n  \"named_entities\": array of 0-12 objects, each {\"entity_id\": \"entity_N\", \"name\": proper noun exactly as spelled, \"source_spans\": [exact source-span objects]},\n  \"tone\": one descriptor for the article's register (e.g. \"cautious\", \"triumphant\", \"clinical\"),\n  \"headline_clean\": headline with wire-service boilerplate stripped, or empty only when the supplied headline is empty,\n  \"provenance_note\": one sentence naming supplied outlet/date, or empty only when those supplied fields are empty\n}\n\nEvery verified_facts entry MUST be checkable against the article text and every quote must exactly reproduce its field slice. Never add a claim the article does not make. Do not deduplicate near-identical claims into one entry -- each entry should describe a genuinely distinct piece of information (the runner will reject exact duplicates).",
    "sonnet_frame_system": "You open a Continuity Archive Recovery Session -- a far-future ceremony in which archivists authenticate one recovered fragment of 21st-century science writing for a public that no longer understands the underlying science. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"session_title\": in-world session name (e.g. \"Recovery Session: The Sixty-Kelvin Claim\"),\n  \"session_premise\": ONE sentence stating what tonight's fragment claims and why the Archive convened to check it -- this becomes the episode's canonical premise record, so state it plainly, not cryptically,\n  \"registrar_cold_open\": exact target words supplied in the typed request; ANNOUNCER's spoken opening -- names the session, gestures at the fragment's subject WITHOUT stating its content, sets ceremonial stakes,\n  \"orum_register\": one register note for the Literalist this session (e.g. \"clipped, exact\"),\n  \"thessaly_register\": one register note for the Speculator this session (e.g. \"restless, associative\"),\n  \"vesh_register\": one register note for the Warden this session (e.g. \"flat, unhurried\"),\n  \"scene_description\": the ledger scene description, \"scene_env\": the ledger env, \"shot_description\": the ledger shot description, \"visual_prompt\": the ledger visual prompt, \"music_description\": the ledger music description, \"music_generation_prompt\": the ledger music generation prompt\n}\n\norum_register, thessaly_register, and vesh_register must be THREE DIFFERENT notes -- no two Reliquarians share a register this session. The cold open does not state any fact, number, or named entity from the dossier -- it orients, it does not inform.",
    "sonnet_literalist_system": "You write ORUM, the Continuity Archive's Literalist. ORUM will only say what the fragment's exact wording supports -- no inference, no consequence, no opinion.\n\nOUTPUT FORMAT - strict JSON, one object:\n{\n  \"text\": the spoken line, no name/colon/quotes/stage directions,\n  \"cites\": array of 1-3 dossier ids from the fact_N/num_N list given below (at least one id required)\n}\n\nCRAFT:\n- Ground every line in a SPECIFIC cited fact_N/num_N entry; never generalize past it.\n- Speak in the register given (orum_register). Short, declarative, exact.\n- Treat the fragment as a physical recovered object -- refer to \"the wording\", \"the fragment\", \"the recovered text\", never \"the article\" or \"the news\".\n- Exact target word count for this line is supplied in the typed request; meet it exactly.",
    "sonnet_speculator_system": "You write THESSALY, the Continuity Archive's Speculator. THESSALY reads consequence and implication into what the fragment leaves open -- this is the session's dramatic and science-fictional engine.\n\nOUTPUT FORMAT - strict JSON, one object:\n{\n  \"text\": the spoken line, no name/colon/quotes/stage directions,\n  \"cites\": array of 1-2 dossier ids (fact_N/num_N) this line's SPRINGBOARD came from -- the extrapolation itself is not a citation, only its starting point is\n}\n\nCRAFT:\n- Dramatic license on WHAT THIS MIGHT MEAN or LEAD TO. Zero license on new facts, numbers, or named entities not in the dossier -- extrapolate from what's given, never invent a new given.\n- Answer or push against Orum's last line; do not simply agree.\n- Speak in the register given (thessaly_register).\n- Exact target word count is supplied in the typed request; meet it exactly.",
    "sonnet_audit_system": "You audit a draft Recovery Session reading against its dossier. Python has already computed citation coverage from each line's cites[] tags -- your job is the judgment python cannot make. Return one JSON object only -- no prose, no fences.\n\nInput: the draft lines (numbered 0..N-1, ORUM and THESSALY citation/extrapolation lines ONLY -- the Registrar's cold open and any Warden lines are never part of this numbered list, each with speaker + text + cites), the dossier verified_facts/key_numbers/named_entities (as fact_N/num_N), and the precomputed coverage numbers.\n\nSchema:\n{\n  \"status\": \"clear\" or \"defect\",\n  \"defects\": array of 0-5 strings, each naming ONE specific problem in plain language (e.g. \"line 4 states a figure not in key_numbers\", \"line 6 contradicts line 3's claim about X\"),\n  \"flagged_line_refs\": array of 0-5 integers -- the draft line indices that need correction (empty iff status is \"clear\"),\n  \"invented_fact_flags\": array of 0-5 integers, a SUBSET of flagged_line_refs specifically flagged for asserting something the dossier does not support,\n  \"severity\": \"critical\" if any invented fact or unresolvable contradiction exists, else \"advisory\",\n  \"sfw_pass\": true or false after reviewing every supplied spoken line\n}\n\nA defect is REAL only if you can name the specific line index and the specific mismatch. Do not flag stylistic preference. status=\"clear\" requires defects and flagged_line_refs both empty; status=\"defect\" requires at least one of each. Any defect requires correction; clean means fully clean.",
    "sonnet_warden_system": "[DEFECT MODE — select only when AuditVerdict.status=\"defect\"]\nYou write VESH, the Continuity Archive's Provenance Warden, objecting on-air to a specific defect just found in tonight's reading. This seam fires ONLY when a defect was found -- do not use it for a clean pass. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"vesh_objection\": exact target words supplied in the typed request; VESH names the SPECIFIC problem (paraphrase the defect, do not read it verbatim as a report) and calls for a re-check -- flat, procedural, unhurried,\n  \"registrar_reopening\": exact target words supplied in the typed request; ANNOUNCER's short ceremonial line formally reopening the record for correction (e.g. \"The Archive does not yet accept this reading. Return to the fragment.\")\n}\n\nNeither line restates the correct answer -- they only call for the re-check. The correction happens in the next pass.\n\n[CLEAR MODE — select only when AuditVerdict.status=\"clear\"]\nYou write VESH, the Continuity Archive's Provenance Warden, confirming on-air that tonight's reading needs no correction. This seam fires only when the reading is fully clean -- do NOT use it after any defect or an actual correction (that is sonnet_warden_system's job). Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"vesh_satisfied\": exact target words supplied in the typed request; VESH's short, flat, procedural confirmation that the record holds -- no objection, no re-check, no reference to any correction\n}\n\nThis is the ENTIRE session's only Warden beat for this pass -- it stands alone, not paired with a Registrar reopening (there is nothing to reopen).",
    "sonnet_rewrite_system": "You rewrite ONLY the flagged lines of a Recovery Session draft, per the audit's flagged_line_refs. Every line NOT in flagged_line_refs must not appear in your output at all -- do not return or touch unflagged lines. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"corrected_lines\": array of 1-6 objects, each { \"line_ref\": integer, MUST be a value from flagged_line_refs, \"speaker\": \"ORUM\" or \"THESSALY\", \"text\": the corrected spoken line, \"cites\": 1-3 dossier ids (fact_N/num_N) },\n  \"vesh_resolution\": exact target words supplied in the typed request; VESH's short on-air acknowledgment that the record now holds\n}\n\nFix ONLY what the defect names. A defect about an invented number is fixed by grounding the line in an actual key_numbers/verified_facts entry, never by vaguening the claim into meaninglessness.",
    "sonnet_attestation_system": "You write the Continuity Archive's closing Attestation -- the session's dramatic resolution, delivered by ANNOUNCER, with a short final line from VESH. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"attestation\": exact target words supplied in the typed request; ANNOUNCER's closing citation-anchored read -- states the fragment's central verified claim plainly, in the Archive's own ceremonial voice, naming the outlet/date from provenance_note when supplied,\n  \"attestation_cites\": array of 1-4 dossier ids (fact_N/num_N) backing the attestation -- the climax must be as provably grounded as any Literalist line,\n  \"vesh_final_seal\": exact target words supplied in the typed request; VESH's short closing line confirming the record is sealed,\n  \"sign_off\": exact target words supplied in the typed request; ANNOUNCER's closing broadcast line, ceremonial, does not repeat the opening's wording\n}\n\nThe Attestation is the episode's climax, not a disclaimer -- land it with weight. State only what verified_facts / key_numbers / provenance_note actually support.",
    "sonnet_calibration_system": "You expand or trim ONE existing THESSALY extrapolation line to help a Recovery Session reading hit its target length. Never touch citation lines, the Attestation, or Warden/Registrar lines -- you will only ever be handed a THESSALY line. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"line_ref\": integer, the THESSALY line index being adjusted (echoed back from the request),\n  \"text\": the revised line, same speaker, same register, adjusted toward the requested word delta,\n  \"cites\": dossier id array (fact_N/num_N), MUST be a subset of the line's ORIGINAL cites -- expansion may not introduce a new springboard fact\n}\n\nExpansion deepens the SAME implication already present in the line; it never introduces a new springboard fact or a new dossier id not already cited by this line."
  },
  "examples": [
    {
      "seam": "sonnet_intake_system",
      "note": "Illustrative only, not consumed by any validator -- shows the expected shape.",
      "sample_output": {
        "verified_facts": [
          "Researchers cooled a niobium ring to 1.2 kelvin and measured zero electrical resistance for 48 hours.",
          "The effect persisted after three independent re-tests."
        ],
        "key_numbers": [
          "1.2 kelvin",
          "48 hours"
        ],
        "named_entities": [
          "Dr. Elena Vasquez",
          "Kestrel Applied Physics Lab"
        ],
        "tone": "cautious",
        "headline_clean": "Lab reports sustained zero-resistance ring at 1.2 kelvin",
        "provenance_note": "Reported by Kestrel Lab's bulletin, dated this week."
      }
    },
    {
      "seam": "sonnet_literalist_system",
      "sample_output": {
        "text": "The wording is exact: one point two kelvin, and the resistance held at zero for forty-eight hours.",
        "cites": [
          "num_0",
          "num_1"
        ]
      }
    },
    {
      "seam": "sonnet_speculator_system",
      "sample_output": {
        "text": "Forty-eight hours is not an accident of measurement -- that is long enough to ask what happens at forty-eight days, and whether the ring would still be a ring by then.",
        "cites": [
          "num_1"
        ]
      }
    },
    {
      "seam": "sonnet_audit_system",
      "sample_output": {
        "status": "defect",
        "defects": [
          "line 2 states 'three re-tests confirmed' but the fragment only says the effect persisted after re-tests, not a specific count of three"
        ],
        "flagged_line_refs": [
          2
        ],
        "invented_fact_flags": [
          2
        ],
        "severity": "critical"
      }
    },
    {
      "seam": "sonnet_rewrite_system",
      "sample_output": {
        "corrected_lines": [
          {
            "line_ref": 2,
            "speaker": "ORUM",
            "text": "The fragment says only that it persisted after re-testing -- it does not give a count. I will not add one.",
            "cites": [
              "fact_1"
            ]
          }
        ],
        "vesh_resolution": "The record holds now. No invented count remains."
      }
    }
  ],
  "tone_guardrails": [
    "Never let ORUM voice speculation or THESSALY voice a bare citation -- the role boundary is the whole point.",
    "Never have any character say \"according to the article\" or \"the news says\" -- the frame device requires archive-native language (\"the fragment\", \"the recovered text\", \"the wording\")."
  ],
  "forbidden_plot_patterns": [
    "The fragment being revealed as a hoax or fabrication (undermines the factual-grounding guarantee).",
    "Any Reliquarian breaking the fourth wall about being a far-future character discussing our present explicitly as 'the past' with dates that contradict provenance_note."
  ],
  "forbidden_leakage_terms": [
    "according to the article",
    "the news reports",
    "breaking news"
  ],
  "source_requirements": [
    "science_rss payload with non-empty seed_text (validated upstream, see 3)"
  ],
  "ledger_validation_notes": [
    "Every character/announcer line except music placeholders carries a cites[] provenance in the runner's internal DraftScript; the ledger's line rows themselves keep the fixed L3 schema and gain no new field (Section 4)."
  ]
}
```



## 3. Pipeline topology — v4 execution map

```text
payload -> P0 dossier -> P1 frame -> P2a ORUM + P2b THESSALY -> P3 audit
  clear  -> P4 WardenSatisfied -> P6 Attestation -> P8 assemble -> P9 seal
  defect -> P4 WardenChallenge -> P5 restricted rewrite -> P3 re-audit
            (at most two content rewrites; then fail loud)
```

P7 remains a frozen declared pack seam but has no target/count route. P8
creates all music anchors; P9 prepares the shared finalizer. Neither writes
dialogue.


## 4. v4 validation and test plan

1. Parse catalog/pack JSON with duplicate-key rejection; assert exactly nine
   Sonnet seams, merged Warden closure, unique cross-spec IDs/paths, and false
   spec-only runnable/executable flags.
2. Run RSS and custom_premise fixtures through map handoff; pin detection must
   use `resolved["seed_source"]`, never refetch.
3. Spy P0–P6: only injected slots run, tabled bounds/three-attempt ladder
   apply, P1 alone sees target, and P7 never runs for a count.
4. Run the clean-path regression first: a clear audit selects only
   WardenSatisfied. Then test defect/advisory defect, two rewrites, exhaustion.
5. Parameterize rejected spoken decorations/casing/self-vocatives/tokens;
   assert LLM repair only and zero tail mutation on accepted text.
6. Assemble a complete ledger with CastLock descriptions, legal roles, music
   skip reasons/anchors, spans/receipts, actual word receipt, and zero Phase
   0/10 warnings/errors.
7. Corrupt a span/citation/hash/cast ID/label/music reason/anchor/saved ledger
   and assert a typed failure before downstream output.
8. Verify UTF-8 no BOM and the repository placeholder-name rule.

## 5. v4 convergence self-audit

| Contract | Verdict | Evidence |
|---|---|---|
| Additive only | PASS | Registry/pack/runner/shared optional finalizer only; no workflow/fetcher/network/sidecar. |
| Payload first and pinned support | PASS | Seven-key map handoff and resolved-only pin discriminator precede P0. |
| Two model slots only | PASS | Every LLM call uses injected closures through structured_call. |
| Complete ledger / freeze | PASS | Cast lock, hierarchy, skip anchors, Phase 0/10, and saved audit are required. |
| Verbatim LLM speech | PASS | Reject-only pre-tail validation plus receipt equality blocks tail mutation. |
| Audible fact traceability | PASS | P0 slices and final citation/hash graph prove every source fact. |
| 720 strategy | PASS | P1-only advisory steer; actual counts are recorded, never gated. |
| Multi-pass/no text surgery | PASS | P3 defect loop is bounded LLM rewrite; P7 has no length route. |
| Fail loud/SFW/UTF-8 no BOM | PASS | Typed terminals, frozen pack guardrails, and tests cover each. |

## 6. Open questions for the operator

1. Confirm `bm_george` and `v2/en_speaker_*` inventory at build time. Missing
   voice raises `SonnetVoiceInventoryError`, never a substitute.
2. The shared TailFinalizer/title-source hook is required for saved-ledger
   proof; without it this bank remains non-runnable.


