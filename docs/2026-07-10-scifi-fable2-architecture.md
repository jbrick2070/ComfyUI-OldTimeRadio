# scifi_fable2 -- Code-Ready Architecture + Coding Plan (LLM-first multipass sci-fi lane)

- Date: 2026-07-10 (overnight design session; operator asleep, autonomous per directive)
- Status: CODE-READY DRAFT -- NO code written. This doc is the complete input for the coding
  sprints. Kibitz hardening rounds applied (see section 15 log).
- Lineage: 3-Fable divergent design panel (Playhouse / Showrunner Ladder / Auteur Long-Pass)
  -> Claude judge synthesis -> operator constraints folded live -> kibitz local-panel rounds.
- Relationship to existing lanes: **science_news stays 100% intact.** This is a NEW ADDITIVE
  lane. The science lane's own improvement plan (docs/2026-07-10-llm-first-story-edit-pass.md)
  is unaffected.

## 0. Operator vision + binding laws

"We have this news story. Your job as an LLM: in multiple passes, fill the production ledger
starting from scratch." Once the story is selected (existing science_rss fetch + selection
reused), the LLM writes EVERYTHING creative. Python judges, validates, rerolls, selects, and
assembles structure -- and never writes or rewrites a spoken word.

1. **Python judges; the LLM writes** (operator directive 2026-07-10).
2. **Ledger obeyed** for downstream consumers (TTS, SceneSequencer, video, freeze gates).
   ADDITIVE meta extensions only; never rename/repurpose existing fields.
3. **Dynamic casting**: LLM decides cast size (num_characters widget = CEILING) and character
   shape/registers per story.
4. **Drift law**: local LLMs drift -- budget 1-2 CLEANUP passes per TRUE CREATIVE pass so the
   ledger is obeyed. The ratio is explicit in section 3.
5. **No fallback to legacy_many_pass, ever.** Fail loud naming the pass.
6. Primary model: local Mistral-Nemo 12B through the structured_call ladder; cloud slots
   (openrouter/comfy widgets) usable per-run without design change.

## 1. Panel judgment (what won, what was discarded)

- From **Playhouse** (prose-first writers' room): dealt frame-card deck; competitive 3-pitch
  room (the proven original_concept -> original_select pattern); casting AFTER the script;
  the **verbatim-substring proof gate**. Discarded: separate announcer-frame pass (the script
  skeleton carries announcer + coda in one broadcast voice); registers-only-after-draft.
- From **Showrunner Ladder**: the treatment keystone (dramatic_question, cast shapes, turn,
  PRICED ending, news_thread); stance card; python word/scene envelopes; per-scene incremental
  led.save(); voice-tag menu dealt from the real registry. Discarded: per-scene JSON dialogue
  core (JSON prose tax on a 12B; keyhole context pathology).
- From **Auteur Long-Pass**: the creative core -- ONE whole-play markup pass, classed CRITIC,
  whole-script REVISION with verbatim-preserve law, keep-better-draft defect judge, END.
  sentinel, markup->ledger parse table. Discarded: casting fully inside the vision pass;
  single-pass angle+spine.

Repo facts verified during judging (Windows tree @ HEAD 2026-07-10): dispatch shape law
(`_bank_has_no_source_contract` needs BOTH fetcher+interpreter empty; writer :1538-1554);
source_bank dropdown data-driven via `list_bank_ids()` (writer :2581-2584) => zero new
widgets; `resolve_story_rules` HARD-ERRORS for a runnable bank without a rules pack
(`_otr_story_rules.py:296-306`); sidecars must register in `_PACK_SIDECAR_FILENAMES_BY_BANK`
(`_otr_story_routing.py:42-48`); structured_call ladder = base -> structural retry at LOWER
temp -> typed repair (the 2B principle).

## 2. Names + surfaces (all new files; only 3 existing files touched, listed in section 12)

| Surface | Value |
|---|---|
| bank id | `scifi_fable2` (label: "Sci-Fi Fable2 (LLM-first multipass)") |
| pack | `nodes/story_packs/scifi_fable2/scifi_fable2_v1.json` |
| sidecar | `nodes/story_packs/scifi_fable2/frame_deck.json` |
| pipeline id | `fable2_multipass` |
| runner | `nodes/_otr_scifi_fable2.py` (pure module: stdlib + pydantic + structured_call; no ComfyUI imports; every failure raises) |
| parser | `nodes/_otr_fable2_markup.py` (stdlib-only pure functions) |
| story rules | `nodes/story_rules/scifi_fable2.json` (detection-only; NO replacement tables -- X1-compliant from birth) |
| declared_seams | `fable2_dossier_system`, `fable2_pitch_system`, `fable2_treatment_system`, `fable2_script_system`, `fable2_critic_system`, `fable2_revision_system`, `fable2_casting_system`, `fable2_audit_system` |
| errors | `Fable2Error` > `Fable2DossierError`, `Fable2PitchError`, `Fable2TreatmentError`, `Fable2ScriptError`, `Fable2ParseError`, `Fable2CastError`, `Fable2AssembleError`, `Fable2AuditError` -- constructor takes `(pass_id, reason, attempts)` and formats "fable2 pass=<id> failed after <n> attempt(s): <reason>" |
| seed env | `OTR_FABLE2_SEED` (reproduces card deal + ranked-pick draw ONLY; OS entropy default -- repo true-randomization convention) |
| meta namespace | `meta.fable2` (additive only) |
| tests | `tests/test_fable2_markup.py`, `test_fable2_artifacts.py`, `test_fable2_assembly.py`, `test_fable2_registry.py`, `test_fable2_runner_ladders.py`, `test_fable2_prompt_snapshots.py`, `test_fable2_vocab_contract.py` |

## 3. Pass graph (creative:cleanup ratio explicit -- drift law)

Python computes `SceneEnvelope` (scene count, per-scene word targets, total band) from
`target_words` BEFORE any LLM call. Temps are runner constants (section 8), pack owns prompts.

| # | pass id | kind | temp | in -> out | attached cleanup |
|---|---|---|---|---|---|
| P0 | `dossier` | technical | 0.30 | python-capped source digest -> DossierArtifact | ladder rungs x2 |
| P1 | `pitch_room` | **CREATIVE 1** | 0.90 | dossier + 3 dealt frame cards + stance -> PitchSlate (3 divergent pitches) | ladder x2 + slate-divergence gate reroll |
| P2 | `treatment` | creative-judge | 0.35 | dossier + slate + stance + N_MAX -> Treatment (incl. cast_shapes w/ registers, priced_ending, news_thread) | ladder x2 + grounding gate |
| P3 | `script` | **CREATIVE 2** | 0.75 | treatment + digest + envelope -> COMPLETE play, strict markup | cleanup A: markup ladder (defect-quoting reroll -> format-repair rung); cleanup B: budget gate reroll (max 2, numeric hint) |
| P4 | `critic` | technical-judge | 0.30 | draft + treatment -> CriticNotes (classed; never writes lines) | ladder x2 |
| P5 | `revision` | **CREATIVE 3** | 0.60 | draft + notes + treatment -> COMPLETE play rewritten (verbatim-preserve unnoted scenes) | cleanup A: markup ladder; cleanup B: keep-better-draft defect judge (worse revision ships draft 1, stamped) |
| P6 | `casting_voices` | technical-creative | 0.40 | winning parsed script + cast_shapes + dealt voice menu -> CastingVoices | ladder x2 + speaker-set equality gate |
| P7 | `assemble` | pure Python | -- | ParsedScript -> ledger rows (section 6 table); incremental led.save() per scene | proof gates; ambiguity = upstream reroll, never silent-fix |
| P8 | `ledger_audit` | technical | 0.20 | assembled script view + treatment -> AuditFindings | evidence triage -> ONE coalesced scoped repair (re-run the named pass with findings as notes) -> re-audit once -> fail loud |

**Ratio: 3 true creative passes (P1/P3/P5), each with 2 attached cleanup mechanisms, plus
dedicated technical passes P0/P4/P6/P8 and the P7 python gates** -- inside the operator's 1-2
cleanup-per-creative band. Typical episode: 8-11 LLM calls, every one seeing the whole story.

**Low-budget mode** (`target_words < 120`): skip P1 (one dealt card feeds P2 directly) and
P4+P5 (one-draft). Stamped `meta.fable2.mode="one_draft"`. P0/P2/P3/P6/P7/P8 never skip.
**Long-episode mode** (`target_words > 900`): P3/P5 act-chunked serial (each act drafted whole,
prior acts in context, per-act targets; same grammar and gates).

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
    facts_to_keep: list[str]          # 3-10 items, 8-200 chars each
    allowed_numbers: list[str]        # 0-10; the ONLY numerals the play may speak
    named_entities: dict[str, list[str]]  # keys people/places/things, 0-10 each
    dramatizable_vectors: list[str]   # 3-5 items, 10-160 chars
    provenance: dict[str, str]        # headline, source, date, link (stamped, not LLM-authored)

class Pitch(BaseModel):
    pitch_id: int                     # 1..3
    frame_card: str                   # verbatim card name (validator: in dealt set)
    logline: str                      # 15-240
    hook: str                         # 10-200
    scifi_device: str                 # 10-160
    cast_size: int                    # 1..N_MAX
    ending_shape: Literal["paid_victory","quiet_loss","ironic_turn","open_question"]

class PitchSlate(BaseModel):
    pitches: list[Pitch]              # exactly 3
    # post_validator (python judge): card i used by pitch i; not all (cast_size,
    # ending_shape) pairs equal; pairwise logline token-overlap < 0.6

class CastShape(BaseModel):
    role: str                         # 3-60
    want: str                         # 5-120
    pressure: str                     # 5-120
    register: str                     # 5-90; HOW they speak, enforceable from voice alone

class Treatment(BaseModel):
    chosen_pitch_id: int
    title: str                        # 3-80
    dramatic_question: str            # 10-200; validator: contains "?"
    setting: str                      # 4-120
    cast_shapes: list[CastShape]      # 1..N_MAX; validator: registers pairwise-distinct
                                      # (normalized token-overlap < 0.5)
    turn: str                         # 10-250
    priced_ending: dict[str, str]     # choice 10-200, cost_paid 10-200
    news_thread: str                  # 10-200; post_validator: >=1 content noun present in
                                      # dossier facts/entities (grounding gate)

class CriticNote(BaseModel):
    scene: int                        # 0 = announcer/coda frame
    speaker: str                      # ALL-CAPS or ""
    problem: Literal["register_bleed","on_the_nose","stakes_sag","ending_unearned",
                     "continuity_break","cast_unused","announcer_contract",
                     "word_budget","subtext_flat"]
    note: str                         # 15-200; never replacement dialogue (regex gate: no
                                      # NAME-colon shapes inside note)

class CriticNotes(BaseModel):
    verdict: Literal["ship","revise"]
    notes: list[CriticNote]           # 0-8; verdict "revise" requires >=1 note

class CastVoice(BaseModel):
    name: str                         # ALL-CAPS; validator vs parsed speaker set (python-side)
    role: str
    character_description: str        # 40-240; portrait-ready, period-safe
    gender: Literal["male","female"]
    age_band: Literal["20s","30s","40s","50s","60s"]
    register: str                     # 5-90 (confirm/refine of the treatment note)
    timbre: str                       # validator: in dealt menu
    want: str
    pressure: str

class CastingVoices(BaseModel):
    cast: list[CastVoice]

class AuditFinding(BaseModel):
    finding_class: Literal[  # critic classes + contract classes
        "register_bleed","on_the_nose","stakes_sag","ending_unearned","continuity_break",
        "cast_unused","announcer_contract","word_budget","subtext_flat",
        "speaker_not_in_cast","verbatim_break","skeleton_break",
        "news_source_framing","machine_attribution","weapons_smoking"]
    scene: int
    speaker: str
    detail: str                       # 10-200

class AuditFindings(BaseModel):
    findings: list[AuditFinding]      # 0-12
```

## 6. Parser spec (`nodes/_otr_fable2_markup.py`)

**Line classifiers (order matters; first match wins):**

```
_RE_TITLE   = ^TITLE:\s*(.+)$
_RE_MUSIC   = ^MUSIC:\s*(.+)$
_RE_SCENE   = ^SCENE\s+(\d{1,2}):\s*(.+)$
_RE_CODA    = ^CODA:\s*(.+)$
_RE_END     = ^END\.\s*$
_RE_SPEAKER = ^([A-Z][A-Z0-9 .'-]{1,24}):\s*(.+)$    (checked AFTER the reserved shapes above;
                                                      speaker then validated against
                                                      cast_names | {"ANNOUNCER"})
anything else non-blank  -> defect BAD_LINE_SHAPE
```

**Defect enum** (`Fable2ParseDefect`): `MISSING_TITLE`, `DUPLICATE_TITLE`, `MISSING_END`,
`CONTENT_AFTER_END`, `BAD_LINE_SHAPE`, `UNKNOWN_SPEAKER(name)`, `PAREN_OR_BRACKET`,
`QUOTED_DIALOGUE`, `SCENE_ORDER(n_seen, n_expected)`, `EMPTY_SCENE(n)`,
`SKELETON_BREAK(missing_element)`, `CAST_MEMBER_SILENT(name)`, `CODA_SHAPE` (not <=16 words /
missing trailing colon), `MULTIPLE_CODA`.

**State machine:** `EXPECT_TITLE -> PREAMBLE -> SCENES -> POSTAMBLE -> DONE`.
PREAMBLE accepts MUSIC then ANNOUNCER (1-2); first `SCENE 1:` enters SCENES; SCENES accepts
speaker lines, MUSIC (interstitial), `SCENE n+1:`; first post-scene ANNOUNCER enters POSTAMBLE
(accepts ANNOUNCER 1-2, CODA exactly once, MUSIC closing, END.). All defects COLLECTED (not
first-fail) and returned; the runner quotes the full list in the reroll rung.

**Output shape:**

```python
@dataclass(frozen=True)
class ParsedLine:   speaker: str; text: str
@dataclass(frozen=True)
class ParsedScene:  n: int; setting: str; lines: tuple[ParsedLine, ...]
@dataclass(frozen=True)
class ParsedScript:
    title: str
    music_open: str; music_inter: tuple[tuple[int, str], ...]; music_close: str
    announcer_intro: tuple[str, ...]      # 1-2
    scenes: tuple[ParsedScene, ...]
    announcer_outro: tuple[str, ...]      # 1-2
    coda: str
    spoken_word_count: int                # character + announcer words (music excluded)
```

`parse_fable2_markup(text: str, cast_names: frozenset[str]) -> tuple[ParsedScript | None,
tuple[Fable2ParseDefect, ...]]`. Pure; property-tested; near-miss speaker remap (Levenshtein
<= 2 to a unique cast name) is applied WITH a `speaker_remap` note returned (KEEP-list hygiene,
stamped to meta) -- everything else defects.

## 7. Assembly spec (P7, python; runs inside the runner)

| Markup element | Ledger effect |
|---|---|
| TITLE | `meta.episode_title_draft` (existing downstream title conventions untouched) |
| preamble | shot `shot_000`: `music_open` sentinel row (text **""**, cue text -> music[] via set_music as generation_prompt), then announcer intro rows (`speaker_role="announcer"`, `char_id="announcer"`; first row `boundary="shot_start"`) |
| SCENE n | scene `s{n:02d}` + shot `shot_{n:03d}`; description = setting text |
| speaker runs | consecutive same-speaker ParsedLines within a scene MERGE into ONE line row (text space-joined -- concatenation of LLM words only); `beat_id = shot_{n:03d}_b{k}`, `line_id == beat_id` (legacy 1:1 convention); first row of a scene `boundary="shot_start"`, each new run `boundary="beat_start"` |
| MUSIC (inter) | `music_inter` sentinel row at its position (text ""), cue -> music[] |
| postamble | final shot: announcer outro rows, CODA row (announcer role; the real-news read that follows it is PROVENANCE TEXT appended from the source payload -- producer convention `coda_mode: real_news_report`, assembly not authorship), `music_close` sentinel |
| cast | c01 = python-prebaked ANNOUNCER (kokoro); characters c02.. from CastingVoices + python voice assignment (section 9) |

**Proof gates (python judges, never writes):**
(a) verbatim-substring: every line row text (whitespace-normalized) must be a substring of the
winning LLM draft; (b) parsed speaker set == cast row names (minus ANNOUNCER); (c) skeleton
complete; (d) `spoken_word_count` within +/-20% of target; (e) every cast member speaks.
Failure -> upstream reroll with the gate named; never silent-fix. Incremental `led.save()`
after preamble and after each scene (crash leaves a coherent partial ledger).

After P8 passes, the runner RETURNS and the writer's shared tail runs unchanged
(meta.visual_plan stamp, K.5.5 story_brief reflection, K.5.6 produced_story, freeze sweeps) --
those meta fields arrive by the mechanism downstream already trusts.

**Additive meta:** `meta.fable2 = {schema_version:"fable2_v1", mode, dossier, cards_dealt,
stance, pitches, treatment, draft1_sha256, final_sha256, better_draft_choice, critic,
casting_stock_dealt, speaker_remaps, parse:{defects_by_attempt, rerolls}, audit:{findings,
discarded, coalesced_repair}, pass_receipts:[{pass_id, model_id, attempts, temp}], seed}`.

## 8. Runner spec (`nodes/_otr_scifi_fable2.py`)

```python
# Constants
_TEMP = {"dossier":0.30,"pitch":0.90,"treatment":0.35,"script":0.75,
         "critic":0.30,"revision":0.60,"casting":0.40,"audit":0.20}
_MARKUP_LADDER_TEMPS = lambda t: (t, round(t*0.66,2), 0.30)   # never raises temperature
_MAX_BUDGET_REROLLS = 2
_TOTAL_WORD_BAND = 0.20          # +/-20% episode
_SCENE_WORD_BAND = 0.30          # +/-30% per scene
_ONE_DRAFT_THRESHOLD = 120       # target_words below => low-budget mode
_ACT_CHUNK_THRESHOLD = 900       # target_words above => act-chunked P3/P5
_DIGEST_CHAR_CAP = 3600          # python-capped source digest (~850 tok)

def run_scifi_fable2_episode(*, payload: dict, pack: StoryPack, resolved: dict,
                             led: "Ledger", meta: dict,
                             creative_fn, technical_fn) -> None:
    """Fill led + meta in place to the legacy writer's endpoint. Raises Fable2Error."""

# Pass helpers (each stamps a pass_receipt; JSON passes ride structured_call):
_pass_dossier(technical_fn, payload) -> DossierArtifact
_deal(seed_env) -> tuple[frame_cards3, stance]           # SystemRandom; OTR_FABLE2_SEED repro
_pass_pitch(creative_fn, dossier, cards, stance) -> PitchSlate
_pass_treatment(creative_fn, dossier, slate, stance, n_max) -> Treatment
_build_envelope(target_words, treatment) -> SceneEnvelope # python; scenes + per-scene targets
_pass_script(creative_fn, treatment, digest, envelope, cast_names) -> tuple[str, ParsedScript]
    # markup ladder: attempt1 base temp -> attempt2 lower temp quoting ALL defects ->
    # attempt3 "fix formatting only" 0.30 -> Fable2ScriptError. Budget gate wraps it.
_pass_critic(technical_fn, draft_text, treatment) -> CriticNotes
_pass_revision(creative_fn, draft_text, notes, treatment, envelope, cast_names)
    -> tuple[str, ParsedScript]
_defect_score(parsed, envelope, rules) -> int             # python judge; picks draft 1 vs 2
_deal_voice_menu(cast_size) -> VoiceMenu                  # from the voice registry; preflight
                                                          # capacity or Fable2CastError pre-LLM
_pass_casting(creative_fn, parsed, treatment, menu) -> CastingVoices
_assign_voices(casting, menu, seed) -> list[dict]         # concrete tts_model + voice_preset
_assemble(led, parsed, cast_rows, payload, meta) -> None  # section 7
_pass_audit(technical_fn, assembled_view, treatment) -> AuditFindings
_triage(findings, parsed, rules) -> confirmed             # evidence bar; uncorroborated
                                                          # discards stamped LOUDLY
_coalesced_repair(creative_fn, parsed, confirmed, treatment) -> tuple[str, ParsedScript]
    # ONE scoped whole-script revision with findings as notes -> re-audit once -> fail loud
_seam(pack, name) -> str                                  # original_radio accessor pattern
```

`news_briefs_required` widget: N/A for this lane (interpreter empty by design) -- documented
no-op, stamped `meta.fable2.notes`.

## 9. Dynamic casting + voice assignment

- SIZE + SHAPE + REGISTERS: LLM-decided in the Treatment (`cast_shapes`, 1..num_characters
  ceiling). Registers exist BEFORE the script (the script seam enforces them; whole-window
  visibility keeps voices distinct); `meta.num_characters_locked` stamped beside the requested
  value.
- VOICES + PORTRAITS: P6 casts the PERFORMED script (exact speaker set), writes
  character_description (feeds visual_plan portraits), orders timbre from a python-dealt menu
  (gender x timbre counts computed from the real voice registry in `config/cast_pools.py`;
  exact symbol names pinned at S0 -- see section 14 open items).
- ASSIGNMENT (python): filter by gender + ordered timbre, score tag overlap, seeded tie-break,
  exclusion set; capacity preflight BEFORE any LLM call. Characters -> bark presets, c02..;
  ANNOUNCER stays python-owned c01/kokoro, never an LLM slot. The LLM invents the person;
  Python picks the larynx.

## 10. Seam prompts (pack `prompt_stages`, complete set of 8)

The four load-bearing prompts are pinned verbatim in the synthesis doc and carry over
unchanged except s/scifi_/fable2_/ on seam names: **fable2_treatment_system** (priced-ending
keystone), **fable2_script_system** (grammar + worked example + craft), **fable2_critic_system**
(classed notes, never writes lines), **fable2_revision_system** (verbatim-preserve law).
The remaining four, in full:

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
      "register": 5-90 chars; HOW they talk, refined from the treatment note
                  and the performance in the script,
      "timbre":   pick ONE tag from the AVAILABLE VOICE STOCK list in the
                  user message -- never invent a tag, never pick a
                  combination listed as 0,
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

## 11. Registry rows + dispatch splice (code-ready)

**banks.json append** (exact row): as in section 7 of the synthesis, with
`source_bank_id: "scifi_fable2"`, `default_story_model: "scifi_fable2_v1"`,
`default_story_pipeline: "fable2_multipass"`, `runnable: false` at S0.

**pipelines.json append**: `story_pipeline_id: "fable2_multipass"`, `executable: false` (S0)
-> `true` (same change as the runner), `requires_source_contract: false`, declared_seams = the
8 fable2 seams, passes = section 3 rows (slots creative/technical as marked), notes:
"Visible experiment; failure NEVER routes to legacy_many_pass." + "Consumes the bank fetcher
payload via the writer's shared fetch lane; interpreter intentionally empty -- the treatment
IS the interpretation."

**Sidecar registration** (`_otr_story_routing.py:42-48`): add
`"scifi_fable2": frozenset({"frame_deck.json"})`.

**Dispatch splice** (`OTR_LedgerScriptWriter.py`): module-level

```python
_RUNNER_BY_PIPELINE: "dict[str, Callable]" = {}   # populated by _otr_scifi_fable2 import

def _register_pipeline_runner(pipeline_id: str, fn) -> None: ...
```

consulted ONCE, immediately after the shared front completes (bank resolve -> runnable gate ->
science_rss fetch -> validate_source_payload -> D.1 new_ledger + meta stamps) and BEFORE the
news_interpreter branch (current HEAD ~:3320; exact line pinned at coding time by the S1
commit). Hit -> call the runner with `(payload, pack, resolved, led, meta, creative_fn,
technical_fn)`, then jump to the shared tail. Miss -> existing branches byte-identical
(legacy_many_pass and the original bank-shape branch are NOT in the map). A runnable,
executable pipeline id with no registered runner raises loud naming the pipeline id.
science_news dispatch untouched by construction; pinned by test.

**Workflow surface: NOTHING appended.** source_bank dropdown auto-includes the bank
(list_bank_ids-driven); num_characters/target_words reused as ceiling/budget; canonical
default stays science_news; OTR_WorkflowValidator no-diff recorded in the S1 commit.

## 12. Files touched (complete inventory)

NEW: `nodes/_otr_scifi_fable2.py`, `nodes/_otr_fable2_markup.py`,
`nodes/story_packs/scifi_fable2/scifi_fable2_v1.json`,
`nodes/story_packs/scifi_fable2/frame_deck.json`, `nodes/story_rules/scifi_fable2.json`,
7 test files (section 2), fixture payloads under `tests/fixtures/fable2/`.

MODIFIED (only these): `nodes/story_packs/banks.json` (+1 row; S1 flips runnable),
`nodes/story_packs/pipelines.json` (+1 row; S1 flips executable),
`nodes/_otr_story_routing.py` (+1 sidecar entry),
`nodes/OTR_LedgerScriptWriter.py` (dispatch map + one splice block, S1).

NEVER touched: science_news pack/rules/dispatch, otr_canonical.json, downstream consumers.

## 13. Sprints (each: suite + Bug Bible green -> commit AND push v2.0-alpha)

- **S0 -- inert surfaces:** registry rows (runnable:false/executable:false), full pack (all 8
  seams authored), frame_deck.json (~24 cards) + sidecar registration, detection-only rules
  file, `_otr_fable2_markup.py` + `test_fable2_markup.py` (every defect class + properties),
  `test_fable2_registry.py`, `test_fable2_prompt_snapshots.py`. Science lane byte-untouched
  (pin test).
- **S1 -- spine, live:** runner P0/P2/P3/P6/P7 (one-draft mode), dispatch map + splice, voice
  assigner + menu + preflight, incremental saves, proof gates; flip runnable+executable SAME
  change; `test_fable2_artifacts.py`, `test_fable2_assembly.py` (golden-chain vs a real
  science-lane fixture ledger), ladder subset tests; 30-word live smoke; validator no-diff
  record.
- **S2 -- full loop (drift law complete):** P1 pitch room + deal, P4 critic, P5 revision +
  keep-better-draft judge, P8 audit + evidence triage + coalesced repair; complete meta.fable2;
  `test_fable2_runner_ladders.py` full, `test_fable2_vocab_contract.py`; 350-word live smoke +
  register-distinctness spot audit.
- **S3 -- hardening + soak:** seeded draw lever, act-chunked long mode, 3-article x 2-seed
  mini-soak with variety audit, Bug Bible entries (Three-File Contract), docs refresh,
  operator eyeball on two rendered episodes.

## 14. Open items pinned for the coding window (verify before S0 code)

1. Exact `config/cast_pools.py` symbol names for the voice registry + announcer prebake
   (`pick_announcer` / `VOICE_REGISTRY` / `VOICE_PROFILES` cited by panel -- pin at S0).
2. Exact splice line in the writer at coding-time HEAD (~:3320 region today).
3. Boundary semantics of preamble/postamble shots vs a REAL science-lane fixture ledger
   (shot_000 convention asserted in test_fable2_assembly.py, adjusted to the fixture truth).
4. set_music signature + music[] row shape (cited from production_ledger.py -- pin exact).
5. K.5.5/K.5.6 shared-tail call contract (args the tail expects present on meta by then).
6. Whether the coda real-news append happens in the writer tail or SceneSequencer for
   coda_mode=real_news_report -- reuse that exact mechanism, do not duplicate it.

## 15. Kibitz hardening log

(appended per round below)
