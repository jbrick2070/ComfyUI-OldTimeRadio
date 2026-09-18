# Multilingual one-switch -- oval + Kokoro scale

2026-09-18. Side-quest planner (Cursor Grok) + Fable + GPT, then operator
locks through the night. Grounded against the live Windows tree. No code in
this file. Dated working notes live under
`docs/2026-09-17-multilingual-one-switch/` (gitignored by `docs/2026-*/`).
This page is the tracked oval AND the staged go-forward.

Do not start this build in a window that owns the dirty Google lane.

## Product (locked)

One dropdown on `OTR_LedgerScriptWriter`. Pragmatic. Apple-clean. Agnostic.

Same shape as `upscale_engine` in `apple/UPSCALERS.md`: live registry, admitted
rows only, a later language appears by itself the day that row ships. No second
box. No ISO in the UI. No engine names. No phonemizer checkbox. No multilingual
node UI. No translated knobs.

**Kokoro is the multilingual dance leader.** The first ship, and every later
row until a second engine earns a seat, speaks through Kokoro defaults only.
No extra voice-pack download. Bark / Dia / Chatterbox / IndexTTS / ElevenLabs /
Google do not block the release. A non-Kokoro engine on a non-English episode
fails at CastLock until that engine is listed on the row.

Day-1 dropdown (operator override 2026-09-18: ALL eight Kokoro languages
in one release -- not Spanish-first). Admitted rows only:

```
Off
English
Spanish
Portuguese
Italian
French
Hindi
Japanese
Mandarin
```

`Off` is not a language. It is the upscaler's off: stamp nothing, today's
path, byte for byte. A graph with no widget still resolves to English (feature
on). `Off` never reaches `resolve_lemmy_cameo` as an iso.

The graph stays English. The episode the viewer hears and reads is native.

Klingon and Esperanto are never rows. The build contract is
`docs/2026-09-18-multilingual-notebooklm/coder_prompt_all_kokoro_day1.md`.

## What stays English vs what goes native

This is the split. Mixing them up is how a Spanish episode gets an English
title card or a Spanish Flux prompt.

| Surface | Language | Why |
|---|---|---|
| Comfy knobs, node titles, `localized_name` | English | Graph stays English |
| Visual / still / i2v / title-IMAGE prompts | Row-owned (`authoring.visual_prompt_iso`) | Not a second workflow. Fully native = this field equals the episode iso |
| Music prompts | English | Music is done; do not reopen |
| `custom_premise` / typed `episode_title` | As typed | Operator text is not rewritten |
| Writer-authored `meta.episode_title` | Native | Hero title card + ASS TITLE events paint this string |
| `meta.language_header` | Native chrome (`Espanol`) | Cosmetic gratitude on the existing card / credits |
| Spoken dialogue, announcer sign-on/off | Native | Writer instruction + `row.spoken` fallbacks |
| SDH captions (dialogue body + speaker labels) | Native | They burn ledger text; ledger is native |
| Credits hero title + audience chrome | Native | The challenge -- see below |
| Credits machine receipts (VRAM, CUDA, model ids, seeds) | English ids | Serial numbers, not show voice |
| Credits brand / status tags (`SIGNAL LOST`, `EPISODE TREATMENT`, `GENERATIVE STACK`, `DELIVERED VOICE`, diagnostic flavor) | English | Not on `_REQUIRED_CREDITS`. Day 1 leaves them. Do not invent a second chrome table. |
| Lemmy | English only | Non-English bypasses him |

**Title, credits, and captions in the native language is the challenge.**
Dialogue TTS is the easy half (Kokoro already has the mouths). The painted
show -- hero title, SDH, credits roll -- is a pile of English Python strings
and English wrap/font assumptions.

## Fully native is a row setting, not a second workflow

Operator: "or a fully native dual workflow with real native-lang prompts."

**No second `otr_canonical.json`.** Dual graphs are widget drift and a dead
lane. One dropdown, one graph. The language row owns `authoring.visual_prompt_iso`.

| Mode | `visual_prompt_iso` | What the picture models see |
|---|---|---|
| English-pixels (safer first ship) | `en` | Today's Flux / Wan / LTX English briefs + English pack tails |
| Fully native (the flair ask) | `es` (same as episode iso) | Writer briefs, shot text, still_word TITLE prompts in Spanish |

Flip the field, not the file. Native picture briefs are authored in the
episode language, not translated from an English brief. Pack
`positive_tail` / `NO_TEXT_CLAUSE` stay English until a native-authored
overlay exists on the row -- say so on the receipt, do not hide the mix,
do not machine-translate the tails as a substitute.

Music stays English either mode. Knobs stay English. THE LAW still holds:
bad Spanish pixels ship; do not reroll for language.

v1 default: ship Spanish **speech + painted show** first (`visual_prompt_iso: en`)
so Kokoro and title/credits/captions land without also betting Flux on Spanish
prompts. Fully native is the next flip on the same Spanish row once a live
leg shows the English-pixel episode. Not a new workflow.

## The oval

```
OTR_LedgerScriptWriter
  episode_language: [ Off | English | Spanish | ...admitted ]   <- ONLY control
           |
           |  label -> row (config/episode_languages.json)
           v
nodes/_otr_episode_languages.py
  dropdown_choices / resolve_label / resolve_ledger
           |
  Off -> stamp nothing, today's path
  English / Spanish / later -> stamp BEFORE freeze, beside source_bank:
    meta.episode_language = "es"
    meta.language_header  = "Espanol"   <- ledger value is Espanol with n-tilde
    meta.episode_language_receipt { registry_id, schema_version,
                                    row_revision, row_sha256 }
           |
  +------------------+------------------+------------------+
  |                  |                  |                  |
Writer prompt    Verbatim/fidelity  CastLock           Painted show
row.authoring    gate BEFORE LLM    Kokoro dance       title + captions
+ native title   shakespeare and    leader; filter     + credits chrome
                 public_domain      every voice path   from row tables
                 English-only v1    by languages[]
           |
  Kokoro adapter asks row.engines["kokoro"]
  (lang_code, voice roster). Other engines join later per row.
           |
  Python-authored spoken + credits + caption chrome come from the row
  measured audio duration -> video timing -> otr/obs/
```

## Registry row

Declarative JSON, validated fail-closed.

Each row carries: `iso`, `label` (stable COMBO API), `admitted`, `sort_order`,
`native_header`, `authoring` (spoken name + writer instruction +
`visual_prompt_iso`), `spoken` (every Python-authored on-air string,
including reserved display names such as the announcer), `credits` (every
audience-facing credits chrome string), `captions` (font policy + wrap policy
+ cps policy ids), `engines` (map keyed by registered engine id; v1 only
`kokoro`), `admission` (source-bank exclusions, readiness extras, min voice
count), `row_revision`.

No `kokoro_lang_code` field on the shared contract. The Kokoro adapter asks
`engines["kokoro"]`.

English row = today's behavior, byte for byte. That is the regression gate.

`Off` is not a row. It is the empty resolution.

## Kokoro catalog (dance leader roster)

Measured from installed `kokoro` 0.9.4 `LANG_CODES` plus the hexgrad/Kokoro-82M
default voice list. No extra pack. Prefetch today only lists the 28 English
ids (`nodes/_otr_kokoro_voice_prefetch.py`); each admitted row adds its own
`.pt` list to that prefetch.

House pool bar (operator): announcer + 2 distinct character voices, do not
double the budget. Day-1 override admits the thin rows: French (1) and
Italian (2) reuse inside their own pool. If a cast cannot seat even with
reuse, fail naming pool size -- never borrow English voices.

| Label | iso | Kokoro `lang_code` | Default voices | G2P | Paint | Admit |
|---|---|---|---|---|---|---|
| English | en | `b` (house British announcer; `a` stays in the character pool) | 28 | misaki[en] (already installed) | Latin Arial / 44 / 17 | **day 1** |
| Spanish | es | `e` | 3 (`ef_dora`, `em_alex`, `em_santa`) | espeak-ng | Latin, same wrap | **day 1** |
| Portuguese | pt | `p` | 3 (`pf_dora`, `pm_alex`, `pm_santa`) | espeak-ng | Latin, same wrap | **day 1** |
| Italian | it | `i` | 2 (`if_sara`, `im_nicola`) | espeak-ng | Latin, reuse in-pool | **day 1** |
| French | fr | `f` | 1 (`ff_siwis`) | espeak-ng | Latin, reuse in-pool | **day 1** |
| Hindi | hi | `h` | 4 | espeak-ng | Devanagari font + wrap | **day 1** |
| Japanese | ja | `j` | 5 | `misaki[ja]` extra | CJK font + wrap | **day 1** |
| Mandarin | zh | `z` | 8 | `misaki[zh]` extra | CJK font + wrap | **day 1** |

Torch-first for every non-English row. Installed `kokoro-onnx` tokenizer docs
only `en-us` / `en-gb`. Do not guess an ONNX locale; prove it or leave it off
the row.

Kokoro `load()` early-returns if a backend exists. Rebuild on language change.
Cache identity is `(lang, device)`. A resident server must not run Spanish
through a leftover `b` pipeline.

Voice metadata: additive `languages: [iso, ...]`. Absent = `["en"]`. Never
derive language from `ef_` prefixes; the prefix is a hint, the field is law.

## The painted-show challenge (title + credits + captions)

Three painters, one language.

### Title

Hero card in `video_engine.py` and ASS TITLE events in `_otr_captions.py`
already paint `meta.episode_title`. Native title is a **writer** job plus a
**glyph** job.

- Writer authors `episode_title` in the episode language.
- Typed widget override stays as typed (may be English on a Spanish episode;
  that is operator text, not a fail).
- Title-card IMAGE prompts stay English. WORD-mode cards that render spoken
  lines as typography use the native line (already the ledger text).
- Latin-1 / Latin-ext must actually exist on the platform mono (Consolas /
  DejaVu / Courier New do; PIL bitmap fallback does not -- that fallback is
  already a placement bug and becomes a glyph bug on `n`/`a`/`e`).
- Wrap is word-split. Fine for Spanish / Portuguese. Wrong for CJK.

### Captions

SDH burns RAW `lines[].text` plus `cast.name` as `NAME:` (`_otr_captions.py`).
If the ledger is native, the body is native for free. What is not free:

- Reserved display name `ANNOUNCER` is English. It must come from `row.spoken`
  (same table as "Good evening") or a Spanish episode captions `ANNOUNCER:`.
- Arial 42 / 44 chars / 17 cps / 2 lines. Latin v1 keeps this. Spanish runs
  longer; THE LAW says do not fail a story for length -- lint may warn, the
  episode still ships.
- `wrap_words` splits on spaces. CJK / some Hindi needs a different wrap
  policy id on the row.
- Parenthetical stage directions stay visible (operator 2026-08-05). Native
  language does not change that ruling.

### Credits

`otr_credits_roll.py` `build_credits_layout` is a large English chrome table:
`MODELS`, `[ PRODUCTION LEDGER ]`, `CAST & VOICES`, `[ STORY SPINE ]`,
`Premise:`, `Subject:`, diagnostic flavor lines, `credits_source_line` from
the bank row, `hud_origin_label`.

Split:

- **Audience chrome** (headers, story-spine labels, diagnostics, source line,
  hero title, `language_header`) -> `row.credits` + `row.spoken`. Admission
  test: every English key present on every admitted row.
- **Author native. Do not translate.** Operator 2026-09-18: ask the model
  to generate the thing in the native tongue. No English-then-translate
  pass on story, title, captions, credits flavor, or (when
  `visual_prompt_iso` is native) picture briefs. A later LLM that fills
  chrome is briefed as a native console writer ("write the SIGNAL LOST
  labels in Spanish"), never as a translator of the English list.
  Keys stay English in the registry so the admission test can match.
  Brand tokens (`SIGNAL LOST`, `OTR`, `VRAM`, `CUDA`) stay.
- **Machine receipts** (VRAM, CUDA, torch, model ids, seeds, revisions) stay
  English identifiers. The LABEL may be native; the VALUE stays the serial.
- Cast **names** are whatever the writer / name pool produced. Spanish
  episodes get Spanish names from the writer, not a translated English pool.
- Bank `credits_source_line` in `story_packs/banks.json` is English today.
  Either the language row overlays it, or the bank grows a per-iso map.
  Overlay on the language row keeps banks from learning ISO.

This is why title/credits/captions are the challenge and Kokoro is not.

### Fable seed (accepted house calls)

[Fable](bd1b2e23-ecdb-4f55-935b-6e594687cc06) authored a native Spanish (plus
Portuguese column) seed at
`docs/2026-09-17-multilingual-one-switch/fable_title_credits_draft.md`
(gitignored dated folder; copy keys from it when Wave 1 is coded). That is
native console voice, not an English-to-Spanish converter.

Accepted:

- `SIGNAL LOST` stays the call sign. Grammar bends around it
  (`Esta es SIGNAL LOST` / `Esto ha sido SIGNAL LOST`). Do not make it
  `SENAL PERDIDA`.
- `language_header` = `Espanol` with the n-tilde (UTF-8 in the ledger).
- Reserved speaker `ANNOUNCER` -> `LOCUTOR` (es and pt). Station role, not
  gendered `LOCUTORA`.
- Sign-on and sign-off both `Buenas noches` -- correct Spanish, not a
  duplicated English pair.
- Writer titles: native from the story, 2-5 words, <=28 chars (pt <=30),
  Spanish title case in, Unicode `.upper()` on the card, accents survive.
  No English scratch title.
- Preferred short fallbacks when a header does not wrap:
  `[ ARMAZON ]`, `[ GUIONISTA / LLM ]`, `+%d MAS`, HUD `ORIGEN`.
- Caps-glyph fixtures must include `O/N/C/A` with accents, not only
  lowercase. Any ASCII-strip on the title path is a defect.
- Keep-English serials: `VRAM`, `CUDA`, `CPU`, `RAM`, `GPU`, `REV`, `SEED`,
  `Temp / top_p`, `OTR v2`, `CastLock`, model ids.

Portuguese column is a parallel native draft for wave 1b, not a translation
step.

## Lemmy (partially gated by language)

Operator: Lemmy is an English house easter egg. Non-English -> all Lemmy
stuff is bypassed. [Fable](1023b8fa-f40c-447f-89ff-fa18b37a2cec) HOLDS.

Mirror `_source_bank_excludes_lemmy`. One choke point: `resolve_lemmy_cameo`.
New policy `language_exclusion` (never asked, not declined). Widget stays,
inert. Always-include is ignored; log once when the knob was not
`natural_roll`. Reserved Cockney voice stays in the box (already
unconditional). Credits/announcer copy do not name him -- nothing to bypass
there.

Must-fix when coded (Fable, grounded):

1. `Off` is not an iso. Absent or English or Off -> today's Lemmy. Do not
   hand `Off` into `resolve_lemmy_cameo`.
2. Keep language OUT of `replay_voice_assignment`. That path replays
   `force_lemmy=bool(lemmy_hit)` with no bank id. An exclusion-outranks-force
   check there flips a recorded hit, drops `c02`, and shifts every seeded
   open slot. Default on `assemble_pre_locked_rows` is "no exclusion".
3. Writer replay language-mismatch fail sits BEFORE the cast block.
4. Precedence: fidelity first (English byte-identical), then language, then
   the knob.
5. Forcing profiles (`otr_lemmy_kokoro_diag`) stay English; a Spanish
   always-include is a silent no-op and needs a one-line test.

Bypass, not fail. A correct episode missing an easter egg is not a wrong
episode.

## Rulings

| Fork | Call | Why |
|---|---|---|
| Dance leader | Kokoro only on the first ship | Operator; no voice-pack; Bark multilingual already failed intelligibility |
| Visual / title IMAGE prompts | Row `visual_prompt_iso` | One graph. v1 Spanish ships `en` pixels; fully native is a field flip |
| Music prompts | English | Music is done |
| Title text, credits audience chrome, captions | Native | Operator: that is the challenge |
| Shakespeare + non-English | Fail at writer, before LLM | Verbatim lane cannot also be a translation |
| public_domain + non-English | Fail at writer in v1 | Author's language as written; later row may admit |
| Widget site | Append after `replay_from` | BUG-LOCAL-097 |
| Stamp | ISO + receipt + `language_header` | Missing / Off = no stamp (Off) or English (legacy missing). Unknown nonempty fails |
| Labels | Stable COMBO strings | `English` / `Spanish` are workflow API |
| Voice filter | Every production path | One helper, wired in the same change |
| Python-authored speech + credits chrome + reserved names | Row tables | Admission test per table |
| Authoring | Native generate, never translate | Writer / title regen / native visual briefs are asked to write in X. No English draft to convert. |
| Downstream rewrites | Carry the same native instruction | Ledger clean / exchange / F1 repair in X, not "translate this line" |
| Replay | Ledger ISO wins | Writer returns before normal stamps. Widget/ledger mismatch fails |
| Dropdown vs machine | Per repo version | Readiness fails at CastLock on that box |
| CastLock timing | Pre-TTS, not pre-writer | Writer does not see engine widgets |
| CJK / Devanagari | Later rows | Font + wrap + (zh/ja) extra phonemizer |
| Foreign RSS | Out of v1 | Language-blind ingest; a Spanish feed under English is an English play of a Spanish seed. Later: filter feeds by row iso |
| Cosmetic header | Ledger JSON only | No new node, no JS banner |

## Fail / do not fail

Fail loud: fidelity bank + non-English (writer); engine not on the row
(CastLock); readiness extra missing (CastLock); zero eligible voices
(CastLock); replay widget/ledger mismatch (writer); unknown language token;
missing credits/spoken/caption key on an admitted row (admission test).

Do not fail: English leaking inside a Spanish generative episode; word-count
or CPS differences; F1 English regexes missing Spanish; typed English title
on a Spanish run; Lemmy always-include on Spanish (bypass + log).

## v1 is not

A Shakespeare translator. A multilingual node UI. Localized knobs. Per-character
or mixed-language episodes. A visual-prompt language dropdown. A second caption
language dropdown. A requirements tax on English installs. A Google-lane edit.
A Bark multilingual revival. Klingon. Esperanto.

## BUILD STATUS -- 2026-09-18, rows 1-7 coded; row 8 is live proof

Rows 1-7 of the go-forward below are BUILT AND WIRED. Bark leftover `v2/`
clears on a non-bark stamp. Voice `languages[]` filters every selection
path. Caption wrap/font follow the row (Nirmala / YaHei; Hindi grapheme
and CJK char wrap). Credits paint `language_header` and pick a script
face from `font_policy`. Row 8 live `otr/obs/` legs are the remaining
proof. Nothing here has been proven on air yet.

| Landed | Where | Proof |
|---|---|---|
| Registry, fail-closed, 8 rows | `config/episode_languages.json`, `nodes/_otr_episode_languages.py` | `tests/test_episode_languages.py` (84) -- ninth-row inject, duplicate `lang_code`, `min_voice_count` above the roster |
| The one widget | `episode_language`, appended after `replay_from` and before the `gate_in` socket -> trailing `widgets_values` slot | `tests/test_episode_language_writer.py` (100), plus widget-order / parity / link-target |
| Canonical + 21 variants | `workflows/otr_canonical.json` now 37 wide; variants regenerated | `build_variants.py --check`: 21 variants, 0 failures |
| Fidelity gate | writer, right after `require_runnable_bank` -- before the LLM preflight and `_resolve_inputs` | every non-English row x shakespeare / public_domain |
| Stamp | `meta.episode_language` / `language_header` / `episode_language_receipt`, beside `source_bank`, before the freeze. `Off` stamps NOTHING -- no key, not a null |
| Replay + row drift | `replay_language_check`, in the replay branch before the cast block. Ledger iso wins; mismatch fails; drift stamps `meta.episode_language_row_drift` and continues on the CURRENT row |
| Authored, never translated | the row's `writer_instruction` leads the three outline stages (`_make_system`), the composition header (`canon_header`), the announcer seams and the title pass. English and `Off` resolve to `""`, so every English prompt is byte-identical |
| Native title | `_generate_title_from_script(language_instruction=...)` fed by `_title_language_instruction(meta)`; the scratchpad runs in the language too |
| Spoken chrome | `_otr_line_composer` sign-on / sign-off / safe-open / WORK sentence, plus a new `spoken.open_on_prefix` on all eight rows | `tests/test_episode_language_painted_show.py` (97) pins the English strings character for character |
| Credits audience chrome | `build_credits_layout` reads the row and puts the chrome table ON the layout, so the drawers never resolve a language a second time. The abridged mark keeps the row's own header |
| Caption announcer label | `_otr_captions._reserved_announcer_label` -- `ANNOUNCER` on English, `LOCUTOR` / `ANNUNCIATORE` / `播音员` per row |
| Headless reach | `episode_language` on both `CREATIVE_WHITELIST` mirrors |

**`name == "ANNOUNCER"` WAS DELIBERATELY NOT RENAMED.** It is an identity key in
roughly forty places -- cast partition, speaker resolution, markup parsing, the
voice-coverage audit -- so the native name is a DISPLAY substitution at the one
surface that paints it. Renaming the key is a systemic break for a cosmetic win.

**Row 4 leftover Bark `v2/` is cleared** on a non-bark `_stamp`. The two
CastLock pins (`test_auto_registry_stamps_voice_refs`,
`test_kokoro_castlock_spoken_rows_have_no_bark_presets`) are the contract.

**Row 7 wrap/font are wired.** `_otr_captions.wrap_text` honors `word_split` /
`unicode_grapheme` / `cjk_chars`. ASS SDH + TITLE faces follow `font_policy`.
Credits `_load_font` prepends Nirmala / YaHei on those rows and still
degrades to the English walk, then PIL bitmap, rather than refusing.

## Staged go-forward (day 1 = all eight Kokoro languages)

Operator override 2026-09-18: do not stage Spanish then the rest. One
release. The numbered rows below are still the build order (registry
before widget before chrome), not a later-language schedule.

1. `config/episode_languages.json` + `nodes/_otr_episode_languages.py` +
   tests. Nine dropdown entries (`Off` + eight rows). Inject a ninth
   language row; it appears with no writer change. English row
   byte-identical. `Off` is resolver state, not a row.
2. Writer widget (`Off` plus the eight labels), fidelity gate, stamp,
   replay check, authoring instruction, native title instruction.
   Append on canonical + variants. Widget-order / parity / links.
3. `row.spoken` + `row.credits` + reserved display names for all eight.
   Wire `_otr_line_composer.py` fallbacks AND `otr_credits_roll.py`
   audience chrome AND caption announcer label through those tables in
   the SAME change. English holds today's literals. Spanish/Portuguese
   from the Fable seed. The rest natively authored. Admission test:
   every English key present, values non-empty.
4. Voice `languages[]`, one eligibility helper on every selection path,
   per-row Kokoro prefetch, coverage by language+engine+role+gender.
5. CastLock + Kokoro backends parameterized; cache identity is
   `(lang, device)`; close the English announcer trapdoor. Resident
   EN then ES then EN on one process.
6. Lemmy language exclusion at `resolve_lemmy_cameo` only (Fable must-fixes).
7. Paint per row: Latin Arial/44/17 with accented fixtures; Hindi
   Devanagari wrap; ja/zh CJK wrap. `misaki[ja]` / `misaki[zh]` are
   CastLock extras, never an English-install tax.
8. Live English and Spanish full legs to `otr/obs/`, plus short 1-act
   smokes for the other six. Negatives per non-English row:
   Shakespeare, public_domain, Bark, forced Lemmy.

Later engines (not day 1): ElevenLabs / Google text-native / IndexTTS
join per row in their own commits. They do not lead.

The helper is wired in the same change that builds it, or the row says why not.
Do not edit the other window's dirty Google-lane files.
