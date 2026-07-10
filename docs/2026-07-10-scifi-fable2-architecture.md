# scifi_fable2 -- Code-Ready Architecture + Coding Plan (LLM-first multipass sci-fi lane)

- Date: 2026-07-10 (overnight design session; autonomous per operator directive)
- Status: CODE-READY DRAFT v2 (r1-hardened) -- NO code written. Kibitz log in section 15.
- Lineage: 3-Fable divergent design panel (Playhouse / Showrunner Ladder / Auteur Long-Pass)
  -> Claude judge synthesis -> operator constraints folded live -> kibitz rounds (codex panel +
  Claude anchor/judge; antigravity auto-lane down).
- Relationship to existing lanes: **science_news stays 100% intact.** This is a NEW ADDITIVE
  lane. The science lane's own improvement plan (docs/2026-07-10-llm-first-story-edit-pass.md)
  is unaffected.

## 0. Operator vision + binding laws

"We have this news story. Your job as an LLM: in multiple passes, fill the production ledger
starting from scratch." Once the story is selected (existing science_rss fetch + selection
reused), the LLM writes EVERYTHING creative. Python judges, validates, rerolls, selects, and
assembles structure -- and never writes or rewrites a spoken word.

1. **Python judges; the LLM writes** (operator directive 2026-07-10). Every spoken ledger row
   must be traceable to a named LLM artifact (proof gate, section 7).
2. **Ledger obeyed** for downstream consumers (TTS, SceneSequencer, video, freeze gates).
   ADDITIVE meta extensions only; never rename/repurpose existing fields.
3. **Dynamic casting**: LLM decides cast size (num_characters widget = CEILING) and character
   shape/registers per story.
4. **Drift law**: local LLMs drift -- budget 1-2 CLEANUP passes per TRUE CREATIVE pass. The
   ratio is carried by DEDICATED cleanup passes (P0/P2a/P4/P6/P8 + the P3/P5 parse and budget
   gates); structured_call retry rungs are a MAXIMUM extra headroom, not the guarantee (the
   structural retry fires only on JSON-syntax failure -- _otr_structured_call.py:523-537,
   :592-600).
5. **No fallback to legacy_many_pass, ever.** Fail loud naming the pass.
6. Primary model: local Mistral-Nemo 12B through the structured_call ladder; cloud slots
   usable per-run without design change.

## 1. Panel judgment (design lineage)

From **Playhouse**: dealt frame-card deck; competitive 3-pitch room; casting AFTER the script;
the verbatim proof gate. From **Showrunner Ladder**: the treatment keystone (dramatic_question,
cast shapes, turn, PRICED ending, news_thread); stance card; python word/scene envelopes;
per-scene incremental led.save(); voice menu dealt from the real registry. From **Auteur
Long-Pass**: the creative core -- ONE whole-play markup pass, classed CRITIC, whole-script
REVISION with verbatim-preserve law, keep-better-draft defect judge, END. sentinel,
markup->ledger parse table. Discarded: per-scene JSON dialogue core (JSON prose tax); casting
fully inside the vision pass; separate announcer-frame pass; prose-then-transcribe (the
simple_4 germ's fatal step).

Repo facts verified (Windows tree @ HEAD 2026-07-10): `_bank_has_no_source_contract` requires
BOTH fetcher+interpreter empty (writer :1538-1554) -- this bank sets fetcher, so the
original_radio branch never fires; source_bank dropdown is data-driven via `list_bank_ids()`
(writer :2581-2584) => zero new widgets; `resolve_story_rules` HARD-ERRORS for a runnable bank
without a rules pack (`_otr_story_rules.py:296-306`); sidecars must register in
`_PACK_SIDECAR_FILENAMES_BY_BANK` (`_otr_story_routing.py:42-48`); structured_call ladder =
base -> structural retry at LOWER temp (JSON-syntax failures only) -> typed repair.

## 2. Names + surfaces

| Surface | Value |
|---|---|
| bank id | `scifi_fable2` (label: "Sci-Fi Fable2 (LLM-first multipass)") |
| pack | `nodes/story_packs/scifi_fable2/scifi_fable2_v1.json` |
| sidecar | `nodes/story_packs/scifi_fable2/frame_deck.json` (authored at S0 + deck-lint test) |
| pipeline id | `fable2_multipass` |
| runner | `nodes/_otr_scifi_fable2.py` (pure module: stdlib + pydantic + structured_call; no ComfyUI imports; every failure raises) |
| parser | `nodes/_otr_fable2_markup.py` (stdlib-only pure functions) |
| story rules | `nodes/story_rules/scifi_fable2.json` (detection-only; NO replacement tables) |
| declared_seams | `fable2_dossier_system`, `fable2_pitch_system`, `fable2_select_system`, `fable2_treatment_system`, `fable2_script_system`, `fable2_critic_system`, `fable2_revision_system`, `fable2_casting_system`, `fable2_audit_system` (9) |
| errors | `Fable2Error` > `Fable2DossierError`, `Fable2PitchError`, `Fable2SelectError`, `Fable2TreatmentError`, `Fable2ScriptError`, `Fable2ParseError`, `Fable2CastError`, `Fable2AssembleError`, `Fable2AuditError` -- ctor `(pass_id, reason, attempts)` |
| seed env | `OTR_FABLE2_SEED` (reproduces the frame-card/stance deal only; OS entropy default) |
| meta namespace | `meta.fable2` (additive only) |
| tests | `tests/test_fable2_markup.py`, `test_fable2_artifacts.py`, `test_fable2_assembly.py`, `test_fable2_registry.py` (incl. deck lint), `test_fable2_runner_ladders.py`, `test_fable2_prompt_snapshots.py`, `test_fable2_vocab_contract.py`, `test_fable2_tail_context.py` |

## 3. Pass graph

Python computes `SceneEnvelope` (scene count, per-scene word targets, total band) from
`target_words` BEFORE any LLM call.

| # | pass id | kind | temp | in -> out | attached cleanup |
|---|---|---|---|---|---|
| P0 | `dossier` | technical | 0.30 | python-capped source digest -> DossierArtifact | ladder |
| P1 | `pitch_room` | **CREATIVE 1** | 0.90 | dossier + 3 dealt frame cards + stance -> PitchSlate | ladder + slate-divergence gate reroll |
| P2a | `pitch_select` | technical-judge | 0.30 | dossier + slate + stance -> PitchSelect (winner + rationale) | ladder |
| P2b | `treatment` | creative | 0.40 | dossier + winning pitch + stance + N_MAX -> Treatment (cast shapes w/ registers, priced_ending, news_thread, news_close_read) | ladder + grounding gates |
| P3 | `script` | **CREATIVE 2** | 0.75 | treatment + digest + envelope -> COMPLETE play, strict markup | markup ladder (defect-quoting reroll -> format-repair rung); budget gate reroll (max 2, numeric hint); truncation retry (+25% tokens, once) |
| P4 | `critic` | technical-judge | 0.30 | draft + treatment -> CriticNotes | ladder |
| P5 | `revision` | **CREATIVE 3** | 0.60 | draft + notes + treatment -> COMPLETE play rewritten (verbatim-preserve unnoted scenes) | markup ladder; keep-better-draft defect judge (worse revision ships draft 1, stamped) |
| P6 | `casting_voices` | technical-creative | 0.40 | winning parsed script + cast_shapes + dealt voice menu -> CastingVoices | ladder + speaker-set equality gate |
| P7 | `assemble` | pure Python | -- | ParsedScript + Treatment -> ledger rows (section 7); incremental led.save() per scene | proof gates; ambiguity = upstream reroll, never silent-fix |
| P8 | `ledger_audit` | technical | 0.20 | assembled script view + treatment -> AuditFindings | evidence triage -> ONE coalesced scoped repair -> re-audit once -> fail loud |

**Ratio (drift law): 3 true creative passes (P1/P3/P5); dedicated cleanup/technical passes
P0/P2a/P4/P6/P8 plus the P3/P5 parse+budget gates and the P7 python gates** -- within the
operator's 1-2 cleanup-per-creative band by construction. Typical episode: 9-12 LLM calls,
every one seeing the whole story. The P2a/P2b split mirrors the proven
original_select/original_brief split in `_otr_original_radio.py`.

**Token budgets (per pass `max_new_tokens`; structured_call's 512 default is NOT enough for
the play passes -- slot fns receive these explicitly):**

```python
_MAX_NEW_TOKENS = {
    "dossier": 700, "pitch_room": 900, "pitch_select": 300,
    "treatment": 1100, "critic": 800, "casting_voices": 1000, "ledger_audit": 700,
    # script/revision scale with the word budget: markup overhead ~1.5x words,
    # ~1.35 tok/word, +200 skeleton overhead, floor 1200, cap 4200.
    "script":   lambda w: min(4200, max(1200, int(w * 2.2) + 200)),
    "revision": same_as_script,
}
```

Missing `END.` = truncation defect: ONE retry with +25% max_new_tokens, then Fable2ScriptError.

**Low-budget mode** (`target_words < 120`): skip P1+P2a (one dealt card feeds P2b directly)
and P4+P5 (one-draft). Mode stamped in `meta.fable2.mode` AND in every pass_receipt. P0, P2b,
P3, P6, P7, P8 never skip.

**Long episodes:** act-chunked drafting is DEFERRED (post-S3, gated on live evidence). Until
then the lane's supported ceiling is `target_words <= ~900`; above it the runner raises
`Fable2ScriptError("target_words above supported ceiling")` rather than silently degrading.

## 4. Markup grammar (single source of truth: pack key `fable2_script_system`; parser cites it)

```
FORMAT -- every non-blank line of your output must match EXACTLY ONE shape:
TITLE: <episode title>                     (first line, exactly once)
MUSIC: <one-line cue: mood + instruments>  (no timing, no lyrics)
SCENE <n>: <one-line concrete setting>     (n = 1, 2, 3 ... strictly in order)
ANNOUNCER: <spoken words>
<NAME>: <spoken words>                     (NAME copied EXACTLY from CAST, ALL CAPS)
CODA: <one pivot clause ending with a colon>
END.                                       (last line, exactly once)

REQUIRED SKELETON, in order:
TITLE -> MUSIC (opening) -> ANNOUNCER intro (1-2 lines) -> SCENE 1 ... last SCENE
-> ANNOUNCER outro (1-2 lines) -> CODA -> MUSIC (closing) -> END.
MUSIC lines may also appear between scenes.

HARD BANS: no parentheses ( ) or brackets [ ] anywhere; no stage directions;
no quotation marks around dialogue; no narration lines; no speaker not in CAST;
never misspell or shorten a CAST name; nothing after END.
```

## 5. Artifact models (pydantic, in `_otr_scifi_fable2.py`)

```python
class DossierArtifact(BaseModel):
    facts_to_keep: list[str]              # 3-10, 8-200 chars each
    allowed_numbers: list[str]            # 0-10; the ONLY numerals the play may speak
    named_entities: dict[str, list[str]]  # people/places/things, 0-10 each
    dramatizable_vectors: list[str]       # 3-5, 10-160 chars
    provenance: dict[str, str]            # headline/source/date/link (python-stamped)

class Pitch(BaseModel):
    pitch_id: int                         # 1..3
    frame_card: str                       # validator: in dealt set; pitch i uses card i
    logline: str                          # 15-240
    hook: str                             # 10-200
    scifi_device: str                     # 10-160
    cast_size: int                        # 1..N_MAX
    ending_shape: Literal["paid_victory","quiet_loss","ironic_turn","open_question"]

class PitchSlate(BaseModel):
    pitches: list[Pitch]                  # exactly 3; post_validator: not all
                                          # (cast_size, ending_shape) equal; pairwise
                                          # logline token-overlap < 0.6

class PitchSelect(BaseModel):
    chosen_pitch_id: int                  # 1..3
    selection_rationale: str              # 20-300; audit trail (meta only, never spoken)

class CastShape(BaseModel):
    role: str; want: str; pressure: str   # 3-60 / 5-120 / 5-120
    register: str                         # 5-90; HOW they speak, enforceable from voice alone

class Treatment(BaseModel):
    title: str                            # 3-80
    dramatic_question: str                # 10-200; validator: contains "?"
    setting: str                          # 4-120
    cast_shapes: list[CastShape]          # 1..N_MAX; validator: registers pairwise-distinct
                                          # (normalized token-overlap < 0.5)
    turn: str                             # 10-250
    priced_ending: dict[str, str]         # choice 10-200, cost_paid 10-200
    news_thread: str                      # 10-200; post_validator: >=1 content noun in dossier
    news_close_read: str                  # 80-420; the 1-2 sentence real-news close the
                                          # announcer speaks after the coda bridge.
                                          # post_validator: every numeral in allowed_numbers;
                                          # every proper noun in named_entities/provenance.
                                          # (LLM-authored => the coda read satisfies the
                                          # proof gate; python only appends it -- r1/C1.)

class CriticNote(BaseModel):
    scene: int; speaker: str
    problem: Literal["register_bleed","on_the_nose","stakes_sag","ending_unearned",
                     "continuity_break","cast_unused","announcer_contract",
                     "word_budget","subtext_flat"]
    note: str                             # 15-200; regex gate: no NAME-colon shapes (a note
                                          # is never replacement dialogue)

class CriticNotes(BaseModel):
    verdict: Literal["ship","revise"]
    notes: list[CriticNote]               # 0-8; "revise" requires >=1 note

class CastVoice(BaseModel):
    name: str                             # ALL-CAPS; python-gated vs parsed speaker set
    role: str
    character_description: str            # 40-240; portrait-ready, period-safe
    gender: Literal["male","female"]
    age_band: Literal["20s","30s","40s","50s","60s"]
    register: str                         # metadata ONLY -- the performed script embodies the
                                          # TREATMENT registers; authority stays with P2b (r1/A3)
    timbre: str                           # validator: in dealt menu
    want: str; pressure: str

class CastingVoices(BaseModel):
    cast: list[CastVoice]

class AuditFinding(BaseModel):
    finding_class: Literal[
        "register_bleed","on_the_nose","stakes_sag","ending_unearned","continuity_break",
        "cast_unused","announcer_contract","word_budget","subtext_flat",
        "speaker_not_in_cast","verbatim_break","skeleton_break",
        "news_source_framing","machine_attribution","weapons_smoking"]
    scene: int; speaker: str; detail: str # 10-200

class AuditFindings(BaseModel):
    findings: list[AuditFinding]          # 0-12
```

## 6. Parser spec (`nodes/_otr_fable2_markup.py`)

Line classifiers (first match wins):

```
_RE_TITLE   = ^TITLE:\s*(.+)$
_RE_MUSIC   = ^MUSIC:\s*(.+)$
_RE_SCENE   = ^SCENE\s+(\d{1,2}):\s*(.+)$
_RE_CODA    = ^CODA:\s*(.+)$
_RE_END     = ^END\.\s*$
_RE_SPEAKER = ^([A-Z][A-Z0-9 .'-]{1,24}):\s*(.+)$   (validated against cast | {"ANNOUNCER"})
anything else non-blank -> BAD_LINE_SHAPE
```

Defect enum (`Fable2ParseDefect`): `MISSING_TITLE`, `DUPLICATE_TITLE`, `MISSING_END`
(= truncation; triggers the +25% token retry once), `CONTENT_AFTER_END`, `BAD_LINE_SHAPE`,
`UNKNOWN_SPEAKER(name)` (**hard defect -- no Levenshtein remap in S1-S3**; near-miss remap is
deferred post-S3 behind live drift evidence, r1/CUT2), `PAREN_OR_BRACKET`, `QUOTED_DIALOGUE`,
`SCENE_ORDER`, `EMPTY_SCENE(n)`, `SKELETON_BREAK(missing_element)`, `CAST_MEMBER_SILENT(name)`,
`CODA_SHAPE`, `MULTIPLE_CODA`.

State machine: `EXPECT_TITLE -> PREAMBLE -> SCENES -> POSTAMBLE -> DONE`. All defects
COLLECTED (not first-fail); the runner quotes the full list in the reroll rung.

Output: `ParsedScript {title, music_open, music_inter[(scene_after, text)...], music_close,
announcer_intro[1-2], scenes[ParsedScene{n, setting, lines[ParsedLine{speaker,text}]}],
announcer_outro[1-2], coda, spoken_word_count}`.
`parse_fable2_markup(text, cast_names) -> (ParsedScript | None, defects)`. Pure;
property-tested.

## 7. Assembly spec (P7, python)

Emits ALL FIVE ledger hierarchies -- scenes[], shots[], beats[], lines[], music[] (the schema
carries all five: production_ledger.py:8-13; set_lines/set_music populate only their own
arrays, so assembly must emit scene/shot/beat rows explicitly -- r1/S2):

| Markup element | Ledger effect |
|---|---|
| TITLE | draft-title handoff into TailContext (exact key pinned at S0 -- open item 14.2) |
| preamble | shot `shot_000`: `music_open` sentinel row (text **""**, cue text -> music[] via set_music as generation_prompt), announcer intro rows (`speaker_role="announcer"`, `char_id="announcer"`; first row `boundary="shot_start"`) |
| SCENE n | scenes[] row `s{n:02d}` (description = setting) + shots[] row `shot_{n:03d}` |
| speaker runs | consecutive same-speaker ParsedLines in a scene MERGE into ONE line row (text space-joined -- concatenation of LLM words only); beats[] row per run (`beat_id = shot_{n:03d}_b{k}`, speaker, line_ids); `line_id == beat_id` (legacy 1:1); first row of a scene `boundary="shot_start"`, each new run `boundary="beat_start"` |
| MUSIC (inter) | `music_inter` sentinel row (text ""), cue -> music[] |
| postamble | final shot: announcer outro rows; CODA bridge row (announcer role, text = script CODA clause); **news-read row** (announcer role, text = `treatment.news_close_read` -- LLM-authored, python-appended; r1/C1); `music_close` sentinel |
| cast | c01 = python-prebaked ANNOUNCER (kokoro); characters c02.. from CastingVoices + python voice assignment (section 9) |

Timing (`start_s`/`dur_s`) stays unset everywhere -- SceneSequencer owns it downstream.

**Proof gates (python judges, never writes):**
(a) **verbatim-artifact gate**: every spoken line row text (whitespace-normalized) must be a
substring of a NAMED LLM artifact -- the winning draft for character/announcer/coda-bridge
rows, `treatment.news_close_read` for the news-read row; the gate records which artifact
covered each row in `meta.fable2.proof_map`;
(b) parsed speaker set == cast row names (minus ANNOUNCER);
(c) skeleton complete; (d) `spoken_word_count` within +/-20% of target; (e) every cast member
speaks. Failure -> upstream reroll with the gate named; never silent-fix. Incremental
`led.save()` after preamble and each scene.

**Additive meta:** `meta.fable2 = {schema_version:"fable2_v1", mode, dossier, cards_dealt,
stance, pitches, selection:{chosen_pitch_id, rationale}, treatment, draft1_sha256,
final_sha256, better_draft_choice, critic, casting_stock_dealt, proof_map,
parse:{defects_by_attempt, rerolls}, audit:{findings, discarded, coalesced_repair},
pass_receipts:[{pass_id, model_id, attempts, temp, max_new_tokens, mode}], seed}`.

## 8. Runner spec (`nodes/_otr_scifi_fable2.py`)

```python
_TEMP = {"dossier":0.30,"pitch_room":0.90,"pitch_select":0.30,"treatment":0.40,
         "script":0.75,"critic":0.30,"revision":0.60,"casting_voices":0.40,
         "ledger_audit":0.20}
_MARKUP_LADDER_TEMPS = lambda t: (t, round(t*0.66,2), 0.30)   # never raises temperature
_MAX_BUDGET_REROLLS = 2
_TOTAL_WORD_BAND = 0.20; _SCENE_WORD_BAND = 0.30
_ONE_DRAFT_THRESHOLD = 120
_SUPPORTED_WORD_CEILING = 900            # act-chunk mode deferred post-S3
_DIGEST_CHAR_CAP = 3600
_MAX_NEW_TOKENS = ...                    # section 3 table

def run_scifi_fable2_episode(*, payload, pack, resolved, led, meta,
                             creative_fn, technical_fn) -> Fable2TailContext:
    """Fill led + meta to the legacy writer's endpoint; return the TailContext
    the shared tail consumes. Raises Fable2Error subclasses."""

_pass_dossier(technical_fn, payload) -> DossierArtifact
_deal(seed_env) -> (frame_cards3, stance)            # SystemRandom; OTR_FABLE2_SEED repro
_pass_pitch(creative_fn, dossier, cards, stance) -> PitchSlate
_pass_select(technical_fn, dossier, slate, stance) -> PitchSelect
_pass_treatment(creative_fn, dossier, pitch, stance, n_max) -> Treatment
_build_envelope(target_words, treatment) -> SceneEnvelope
_pass_script(creative_fn, treatment, digest, envelope, cast_names)
    -> (markup_text, ParsedScript)                   # markup ladder + budget + truncation
_pass_critic(technical_fn, draft, treatment) -> CriticNotes
_pass_revision(creative_fn, draft, notes, treatment, envelope, cast_names)
    -> (markup_text, ParsedScript)
_defect_score(parsed, envelope, rules) -> int        # python judge; draft 1 vs 2
_deal_voice_menu(cast_size) -> VoiceMenu             # section 9; capacity preflight pre-LLM
_pass_casting(creative_fn, parsed, treatment, menu) -> CastingVoices
_assign_voices(casting, menu, seed) -> list[dict]
_assemble(led, parsed, treatment, cast_rows, payload, meta) -> None
_pass_audit(technical_fn, view, treatment) -> AuditFindings
_triage(findings, parsed, rules) -> confirmed        # evidence bar; discards stamped LOUDLY
_coalesced_repair(creative_fn, parsed, confirmed, treatment) -> (markup_text, ParsedScript)
_build_tail_context(parsed, treatment, casting, payload, meta) -> Fable2TailContext
_seam(pack, name) -> str                             # original_radio accessor pattern
```

`news_briefs_required` widget: N/A for this lane (interpreter empty by design) -- documented
no-op, stamped in `meta.fable2.notes`.

## 9. Dynamic casting + voice assignment

- SIZE + SHAPE + REGISTERS: LLM-decided in the Treatment (1..num_characters ceiling);
  registers exist BEFORE the script and their authority stays with the treatment -- P6
  `register` is descriptive metadata for portraits/meta only (r1/A3).
  `meta.num_characters_locked` stamped beside the requested value.
- VOICES + PORTRAITS: P6 casts the PERFORMED script (exact speaker set), writes
  character_description, orders timbre from the dealt menu.
- **VoiceMenu derivation (r1/S3):** python builds the menu from `config/cast_pools.py` at
  runtime: enumerate the character voice registry (`VOICE_REGISTRY`, ~:344-358; preset
  descriptions via `open_voice_pool`, ~:477-503 -- exact symbols pinned at S0), derive
  `(gender, timbre_tag)` counts from each preset's metadata. **Fallback if no structured
  timbre taxonomy exists:** the menu lists gender counts + the preset DESCRIPTION strings as
  free-text timbre options; P6 orders by quoting one description verbatim; python matches
  order -> preset by exact description string. Either way the menu only ever offers what
  exists, and a `cast_size > available(gender-compatible)` check raises Fable2CastError
  BEFORE any LLM call.
- ASSIGNMENT (python): filter by gender + ordered timbre, seeded tie-break, exclusion set.
  Characters -> bark presets c02..; ANNOUNCER stays python-owned c01/kokoro. The LLM invents
  the person; Python picks the larynx.

## 10. Seam prompts (pack `prompt_stages`, complete set of 9 -- all inline, no external refs)

### `fable2_dossier_system`

```
You prepare source dossiers for a science-fiction radio anthology. Read the
science news story below and extract ONLY what the writers need. Return one
JSON object only -- no prose, no fences.

Schema:
{
  "facts_to_keep":        3-10 short strings; the load-bearing facts, each
                          under 200 chars, faithful to the story,
  "allowed_numbers":      0-10 strings; the ONLY numerals the drama may speak
                          (quantities, dates, measurements) copied verbatim,
  "named_entities":       { "people": [...], "places": [...], "things": [...] }
                          verbatim from the story, 0-10 each,
  "dramatizable_vectors": 3-5 strings, 10-160 chars each; where a DRAMA could
                          grow from this science -- a pressure, a rivalry, a
                          risk, a promise, a cost. Not plot pitches; raw
                          dramatic soil.
}

Rules:
- Copy, never invent: every entity and number must appear in the story text.
- dramatizable_vectors name human pressure, not technology specs.
- No editorializing, no opinions about the science.
```

### `fable2_pitch_system`

```
You are the writers' room for SIGNAL LOST, a science-fiction radio anthology.
From the SOURCE DOSSIER, the THREE FRAME CARDS, and the EDITORIAL STANCE in
the user message, pitch THREE different episodes dramatizing tonight's
science story. Return one JSON object only -- no prose, no fences.

Schema:
{
  "pitches": array of EXACTLY 3 objects, each:
    { "pitch_id":     1, 2, or 3,
      "frame_card":   copy the card name you used, verbatim -- pitch i uses card i,
      "logline":      15-240 chars; one sentence of premise with a live conflict,
      "hook":         10-200 chars; why a listener stays through minute one,
      "scifi_device": 10-160 chars; the speculative leap taken FROM the
                      dossier's science -- extrapolate it, never contradict it,
      "cast_size":    integer 1..N_MAX (N_MAX in the user message); how many
                      speaking characters THIS story actually needs,
      "ending_shape": one of "paid_victory", "quiet_loss", "ironic_turn",
                      "open_question" }
}

Rules:
- The card is the story's SHAPE (whose eyes, what scale, what genre feel);
  the dossier is the story's MATTER.
- The three pitches must differ in conflict shape, and must not all share
  the same cast_size and ending_shape.
- Ground every pitch: at least two dossier entities or vectors load-bearing
  per pitch, not decoration.
- People, not press releases: conflict between characters who want different
  things, never "humanity vs the discovery".
- SFW. No weapons, no smoking. Never frame a pitch as news or a true story.
```

### `fable2_select_system`

```
You are the story editor for SIGNAL LOST. Below are the source dossier, the
editorial stance, and THREE pitched episodes. Pick the ONE that makes the
best radio. Return one JSON object only -- no prose, no fences.

Schema:
{ "chosen_pitch_id": 1, 2, or 3,
  "selection_rationale": 20-300 chars; why this pitch wins -- judged on
    audible conflict, an ending that can land in a short episode, concrete
    use of the science, and fit to the editorial stance. }

Rules:
- Judge; do not rewrite any pitch.
- The rationale is production paperwork. It is never spoken on air.
```

### `fable2_treatment_system`

```
You are the story editor for SIGNAL LOST, expanding tonight's winning pitch
into a treatment. Below are the source dossier, the editorial stance, and the
winning pitch. Return one JSON object only -- no prose, no fences.

Schema:
{
  "title":             3-80 chars; a period radio-drama title,
  "dramatic_question": 10-200 chars; ONE yes/no question the episode answers,
                       naming who wants what. Example: "Will Doctor Voss admit
                       the signal is failing before her crew stops trusting
                       her?" NOT "what will happen next?"
  "setting":           4-120 chars; one concrete place the episode can live in,
  "cast_shapes":       array of 1 to N_MAX objects (N_MAX in the user message;
                       honor the pitch's cast_size unless the story truly needs
                       one fewer), each:
                       { "role": 3-60 chars, "want": 5-120 chars,
                         "pressure": 5-120 chars,
                         "register": 5-90 chars; HOW they speak, enforceable
                         from voice alone (e.g. "clipped, front-loaded,
                         swallows apologies"). Every register must DIFFER in
                         mechanism, not adjective. },
  "turn":              10-250 chars; the single reversal at the center,
  "priced_ending": {
      "choice":    10-200 chars; the concrete final choice one character makes,
      "cost_paid": 10-200 chars; the concrete thing they give up, willingly.
                   A cost is an OBJECT, a POSITION, or TIME -- never
                   "innocence" or "peace of mind". },
  "news_thread":       10-200 chars; the real science fact the drama
                       extrapolates; copied concepts, not copied sentences,
  "news_close_read":   80-420 chars; the 1-2 sentence closing news read the
                       announcer will speak AFTER the coda pivot -- era-neutral,
                       factual, drawn ONLY from the dossier. Use only numbers
                       from allowed_numbers and names from the dossier.
}

Rules:
- Every ending shape pays: in a paid_victory the price buys the win; in a
  quiet_loss the price is what was let go; in an ironic_turn the price is
  what the irony cost; in an open_question the price is what the asking cost.
  priced_ending is required for ALL FOUR shapes.
- Every cast shape must be HEARABLE: a reason to speak in scenes. Cut a shape
  rather than carry a silent one.
- No dates, no brands, no franchises. SFW. No weapons, no smoking.
- The drama is never framed as news, a report, or a true story; the
  news_close_read is the ONLY factual sentence set, and it is clearly the
  announcer's, not a character's.
```

### `fable2_script_system`

```
You write complete science-fiction radio plays for SIGNAL LOST, an
old-time-radio anthology. You will receive a TREATMENT (spine, cast with
registers) and a SOURCE DIGEST. Write the ENTIRE episode in one piece, in
the exact FORMAT below.

<FORMAT block -- section 4, verbatim>

EXAMPLE of correct output shape (a different, shorter story; the example
shows FORMAT ONLY -- never imitate its plot, names, or imagery):
TITLE: The Long Count
MUSIC: slow theremin swell, distant telegraph clicks
ANNOUNCER: Tonight, a story of one antenna, two signals, and the woman who refused to choose between them.
SCENE 1: A cliff-top listening station, an hour before dawn
VERA: Play it again. Slower.
DOKU: The tape is the tape, Vera. Slowing it down won't change what it says.
VERA: Then change the question we're asking it.
MUSIC: single sustained violin note, rising
SCENE 2: The station generator room, minutes later
DOKU: You pulled the log pages. I checked.
VERA: I pulled one page. The one with my name on it.
ANNOUNCER: Vera got her answer -- though the antenna, as ever, kept the better half of the conversation.
CODA: Beyond tonight's cliff-top signal, a real transmission waits:
MUSIC: closing theme, warm brass, resolving
END.

WRITING THE EPISODE:
- Follow the TREATMENT exactly: its spine, its turn, its priced ending. Use
  every CAST member; each must earn their lines. Add no one.
- SCENE count and per-scene word targets are in the user message. Stay within
  plus or minus 30% per scene.
- Each character speaks ONLY in their register note. A clipped speaker stays
  clipped in every scene. Never blur two voices.
- Imply more than you state; characters rarely answer each other directly;
  pressure shows through concrete objects, never by naming the feeling; every
  scene moves the spine one step; no summarizing, no explaining the theme.
- The ending lands the priced ending concretely -- an object, a person, a
  place changed -- and shows what it cost. No moral, no lesson, no recap.
- The drama is FICTION extrapolated from the digest: characters never call it
  news, a report, a study, or a true story, and never mention machines
  writing stories. The ANNOUNCER may hint, never summarize.
- CODA: at most 16 words, ends with a colon, pivots toward the real report
  without stating any fact, number, or outcome. The report itself is NOT
  yours to write here.
- Write the complete episode now. Start with TITLE: and stop after END.
```

### `fable2_critic_system`

```
You are the story editor for SIGNAL LOST. Below is the complete draft of
tonight's episode plus its TREATMENT. Return targeted revision notes.
Return one JSON object only -- no prose, no fences.

Schema:
{ "verdict": "ship" or "revise",
  "notes": array of 0-8 objects, each:
    { "scene":   integer scene number (0 = announcer/coda frame),
      "speaker": ALL-CAPS name or "" when the note is about the scene,
      "problem": one of "register_bleed" | "on_the_nose" | "stakes_sag" |
                 "ending_unearned" | "continuity_break" | "cast_unused" |
                 "announcer_contract" | "word_budget" | "subtext_flat",
      "note":    15-200 chars; WHAT to change and WHY -- never the new
                 wording itself } }

Rules:
- You are a critic, not a writer: NEVER write replacement dialogue.
- Judge ONLY against the treatment and the problem classes above.
- "ship" with an empty notes array is a legitimate verdict. Do not invent
  findings to look busy.
- word_budget is for SCENE-LEVEL pacing only (a scene that rushes or drags);
  the producer already meters the total word count by machine.
- register_bleed names BOTH speakers. ending_unearned says which earlier
  scene should plant what the ending pays off.
```

### `fable2_revision_system`

```
You are revising your own complete radio play for SIGNAL LOST. You will
receive the TREATMENT, the FULL current draft, and the editor's REVISION
NOTES. Rewrite the COMPLETE episode, applying every note.

<FORMAT block -- section 4, verbatim>

REVISION LAW:
- Output the ENTIRE episode again, top to bottom -- not a diff.
- Apply EVERY note. A note names a scene and a problem; you choose the words.
- PRESERVE what works: any scene with no note against it is copied VERBATIM.
  Do not improve unmentioned scenes.
- Keep the same TITLE, CAST, scene count and order, skeleton, and the same
  ending EVENT (fix HOW it lands when a note asks, never WHAT happens --
  unless an ending_unearned note says the setup must change).
- Start with TITLE: and stop after END.
```

### `fable2_casting_system`

```
You are the casting director for SIGNAL LOST. The finished script and the
treatment's cast shapes are below. Cast every speaking character IN THE
SCRIPT -- exactly the ALL-CAPS names that appear, no one else, never the
ANNOUNCER. Return one JSON object only -- no prose, no fences.

Schema:
{
  "cast": array with EXACTLY one object per distinct script speaker:
    { "name":     the ALL-CAPS name, spelled exactly as in the script,
      "role":     3-60 chars,
      "character_description": 40-240 chars; appearance + manner for the
                  portrait artist: age, build, one prop, one habit. Period
                  feel. No plot, no backstory sentences, no brand names.
      "gender":   "male" | "female",
      "age_band": "20s" | "30s" | "40s" | "50s" | "60s",
      "register": 5-90 chars; the treatment's register as PERFORMED --
                  descriptive paperwork, not a new contract,
      "timbre":   pick ONE option from the AVAILABLE VOICE STOCK list in the
                  user message -- never invent one, never pick an option
                  listed as unavailable,
      "want":     5-120 chars,
      "pressure": 5-120 chars }
}

Rules:
- Derive from the script as performed: the character who whispers through
  walls gets a voice that can whisper.
- Contrast the voices: two characters must never share both gender and
  timbre when stock allows otherwise.
- SFW. No weapons, no smoking.
```

### `fable2_audit_system`

```
You are the standards office for SIGNAL LOST. Below are the assembled episode
script and its TREATMENT. Audit the episode against the checklist. Return one
JSON object only -- no prose, no fences.

Schema:
{ "findings": array of 0-12 objects, each:
    { "finding_class": one of "register_bleed" | "on_the_nose" | "stakes_sag" |
        "ending_unearned" | "continuity_break" | "cast_unused" |
        "announcer_contract" | "word_budget" | "subtext_flat" |
        "speaker_not_in_cast" | "verbatim_break" | "skeleton_break" |
        "news_source_framing" | "machine_attribution" | "weapons_smoking",
      "scene":   integer scene number (0 = announcer/coda frame),
      "speaker": ALL-CAPS name or "",
      "detail":  10-200 chars; what is wrong, concretely -- quote the
                 offending words when possible } }

Rules:
- Report ONLY what you can point at. No taste findings outside the classes.
- An empty findings array is a legitimate, common result. Do not invent
  findings to look busy.
- news_source_framing = a character calls the tale news, a report, a study,
  or a true story. machine_attribution = any voice says a machine wrote the
  story. weapons_smoking = any weapon or smoking reference.
- You report; you never rewrite. The writer fixes what you name.
```

## 11. Registry rows, dispatch splice, and the TailContext contract

**banks.json append (FULL row, inline -- r1/C3):**

```json
{
  "source_bank_id": "scifi_fable2",
  "label": "Sci-Fi Fable2 (LLM-first multipass)",
  "source_kind": "article",
  "interpreter": "",
  "fetcher": "science_rss",
  "default_story_model": "scifi_fable2_v1",
  "default_story_pipeline": "fable2_multipass",
  "defaults": {
    "story_form_label": "science-fiction audio drama",
    "source_material_label": "Science story",
    "coda_mode": "real_news_report",
    "credits_source_line": "dramatized by machine from tonight's science wire"
  },
  "required_seams": [],
  "runnable": false,
  "guide_ref": "LLM-first multipass lane: the LLM writes the whole play (markup), Python parses it into the ledger. Consumes the raw science_rss payload; interpreter intentionally empty -- the treatment IS the interpretation. No fallback to legacy_many_pass."
}
```

`fetcher: "science_rss"` + `interpreter: ""` means `_bank_has_no_source_contract` (writer
:1538-1554) stays False -- the original_radio spark branch NEVER fires for this bank; the
shared fetch lane hands the runner a validated RSS payload. **Pinned by test
(`test_fable2_registry.py::test_fable2_receives_rss_payload_not_spark`) -- r1/C3.**
`runnable` flips true only in the same change as the runner (sweep rule b).

**pipelines.json append:** `story_pipeline_id: "fable2_multipass"`, `executable: false` (S0)
-> `true` (with the runner), `requires_source_contract: false`, declared_seams = the 9 fable2
seams, passes = section 3 rows, notes: "Visible experiment; failure NEVER routes to
legacy_many_pass." + "Consumes the bank fetcher payload via the writer's shared fetch lane;
interpreter intentionally empty -- the treatment IS the interpretation."

**Sidecar registration** (`_otr_story_routing.py:42-48`): add
`"scifi_fable2": frozenset({"frame_deck.json"})`.

**Dispatch splice (writer):** module-level `_RUNNER_BY_PIPELINE` map consulted ONCE after the
shared front (bank resolve -> runnable gate -> science_rss fetch -> validate_source_payload ->
D.1 new_ledger + meta stamps) and BEFORE the news_interpreter branch (~:3320 at today's HEAD;
pinned in the S1 commit). Hit -> runner; miss -> existing branches byte-identical
(legacy_many_pass and the original bank-shape branch are NOT in the map). A runnable,
executable pipeline id with no registered runner raises loud.

**TailContext contract (r1/C2 -- the biggest r1 finding).** The writer's tail is NOT a
callable boundary today: it consumes legacy locals (outline/canon/title/news objects) across
`:5874-5950` (title/canon), `:6051-6147` (visual_plan + K.5.5/K.5.6 reflections), and
`:6259-6356` (final return/news payload). **S1 therefore extracts the tail into
`_run_writer_tail(ctx: WriterTailContext) -> node_outputs`, called by BOTH the legacy path and
the fable2 runner.** WriterTailContext fields are pinned at S0 by reading those three spans at
coding-time HEAD; the fable2 runner builds them from its own artifacts:

```python
@dataclass
class WriterTailContext:
    led: Ledger; meta: dict
    outline_view: Any        # minimal outline-compatible view synthesized from ParsedScript
                             # (scenes/beats/title) -- exact required attrs pinned at S0
    canon_header: str        # from treatment (setting, cast, premise line)
    episode_title_draft: str # from markup TITLE
    news_payload: dict       # provenance for the return path / coda append mechanism
    source_bank_row: Any; resolved: dict; episode_root: Any
    # + any further legacy locals the S0 read surfaces -- the pin test
    # (test_fable2_tail_context.py) asserts the extraction is byte-identical
    # for the legacy lane on a fixture episode.
```

Legacy byte-identity is gated by test + one live science smoke before the fable2 flip.

**Workflow surface: NOTHING appended.** source_bank dropdown auto-includes the bank
(list_bank_ids-driven); num_characters/target_words reused as ceiling/budget; canonical
default stays science_news; OTR_WorkflowValidator no-diff recorded in the S1 commit.

## 12. Files touched (complete inventory)

NEW: `nodes/_otr_scifi_fable2.py`, `nodes/_otr_fable2_markup.py`,
`nodes/story_packs/scifi_fable2/scifi_fable2_v1.json`,
`nodes/story_packs/scifi_fable2/frame_deck.json`, `nodes/story_rules/scifi_fable2.json`,
8 test files (section 2), fixtures under `tests/fixtures/fable2/`.

MODIFIED (exactly FOUR existing files -- r1/C5): `nodes/story_packs/banks.json` (+1 row; S1
flips runnable), `nodes/story_packs/pipelines.json` (+1 row; S1 flips executable),
`nodes/_otr_story_routing.py` (+1 sidecar entry), `nodes/OTR_LedgerScriptWriter.py`
(dispatch map + splice + the S1 tail extraction into `_run_writer_tail`).

NEVER touched: science_news pack/rules/dispatch semantics, otr_canonical.json, downstream
consumers. (The tail extraction refactors the writer's own body; the legacy behavior is
byte-identity-pinned.)

## 13. Sprints (each: suite + Bug Bible green -> commit AND push v2.0-alpha)

- **S0 -- inert surfaces:** registry rows (runnable:false/executable:false), full pack (all 9
  seams), frame_deck.json (~24 authored cards) + sidecar registration + **deck-lint test**
  (no weapons/horror/banned content in cards), detection-only rules file,
  `_otr_fable2_markup.py` + `test_fable2_markup.py` (every defect class + properties),
  `test_fable2_registry.py` (incl. rss-payload-not-spark pin), `test_fable2_prompt_snapshots.py`.
  **S0 read-pins:** cast_pools symbols, tail-local inventory for WriterTailContext, exact
  splice line, set_beats/set_music exact shapes, draft-title handoff key.
- **S1 -- spine, live:** `_run_writer_tail` extraction + `test_fable2_tail_context.py`
  byte-identity pin + one legacy science smoke green FIRST; then runner P0/P2b/P3/P6/P7
  (one-draft mode), dispatch map, voice menu + assigner + preflight, incremental saves, proof
  gates (incl. proof_map); flip runnable+executable SAME change; `test_fable2_artifacts.py`,
  `test_fable2_assembly.py` (golden-chain vs a real science-lane fixture ledger + the
  golden happy-path fixture), ladder subset tests; 30-word live smoke; validator no-diff record.
- **S2 -- full loop (drift law complete):** P1 pitch room + deal, P2a select, P4 critic, P5
  revision + keep-better-draft judge, P8 audit + evidence triage + coalesced repair; complete
  meta.fable2; `test_fable2_runner_ladders.py` full, `test_fable2_vocab_contract.py`;
  350-word live smoke + register-distinctness spot audit.
- **S3 -- hardening + soak:** 3-article x 2-seed mini-soak with variety audit, Bug Bible
  entries (Three-File Contract), docs refresh, operator eyeball on two rendered episodes.
- **Deferred post-S3 (build only on live evidence):** act-chunked long-episode mode (> 900
  words); Levenshtein speaker remap; any additional seeded draw levers.

## 14. Open items pinned for the coding window (verify before S0 code)

1. Exact `config/cast_pools.py` symbols for the voice registry + announcer prebake
   (`VOICE_REGISTRY` ~:344-358, `open_voice_pool` ~:477-503 confirmed present; the
   gender/timbre metadata shape decides menu vs fallback -- section 9).
2. Draft-title handoff key + the title-regen tail contract (title_form_label threading
   confirmed at writer :835-909; the handoff key name is the unknown).
3. WriterTailContext field inventory: read writer :5874-5950, :6051-6147, :6259-6356 at
   coding-time HEAD and pin every consumed local.
4. Exact splice line for the dispatch map (~:3320 region today).
5. Preamble/postamble shot-numbering vs a REAL science-lane fixture ledger (shot_000
   convention asserted then adjusted to fixture truth).
6. `set_beats` / `set_music` exact row shapes (production_ledger.py:769-798 / :1083-1095).
7. Where the legacy coda real-news append happens for coda_mode=real_news_report
   (compose_news_coda path, _otr_line_composer.py:3442-3472) -- fable2 replaces that mechanism
   with treatment.news_close_read + assembly; confirm nothing downstream re-appends.

## 15. Kibitz hardening log

- **r1 (arc/coherence), 2026-07-10:** panel = codex (gpt-5.5 high) + Claude anchor;
  antigravity auto-lane FAILED rc=1 (known failure mode; retry scheduled r2). Codex verdict
  "no" with 5 MUST-FIX -- ALL CONFIRMED and folded: C1 coda/proof-gate contradiction (fixed
  via treatment.news_close_read as a named LLM artifact), C2 tail-is-not-a-boundary (fixed via
  WriterTailContext + `_run_writer_tail` extraction with legacy byte-identity pin), C3
  dangling bank-row reference (row now inline + rss-not-spark test), C4 missing max_new_tokens
  budgets (per-pass table + truncation retry), C5 inventory count. SHOULD-FIX folded: ladder
  rungs reworded as maximums; scenes/shots/beats emission specified; VoiceMenu algorithm +
  fallback; strict UNKNOWN_SPEAKER (remap deferred). CUTs accepted: act-chunk deferred
  post-S3; ranked-pick draw cut (card-deal seed kept). Anchor items folded: P2 split
  (select/treatment), priced_ending x ending_shape mapping rule, register authority, critic
  word_budget scoping, deck authoring + lint at S0, receipts mode stamp, format-example
  imitation guard. Full judgment: kibitz-runs/2026-07-10-scifi-fable2/r1/final.md.
