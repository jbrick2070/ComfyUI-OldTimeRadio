# Multilingual one-switch -- oval consensus

2026-09-17. Side-quest planner (Cursor Grok) + [Fable](c29dbf05-3313-4761-81d8-65139057c03d)
+ [GPT](697d33ef-5403-4a51-a2e6-397c32e23cc4). Grounded against HEAD `90e75d33`.
No code in this file. Dated working notes live under
`docs/2026-09-17-multilingual-one-switch/` (gitignored by `docs/2026-*/`).
This page is the tracked oval.

## Product (locked)

One dropdown on `OTR_LedgerScriptWriter`. Pragmatic. Apple-clean. Agnostic.

Same shape as `upscale_engine` in `apple/UPSCALERS.md`: live registry, admitted
rows only, a third language appears by itself the day that row ships. No second
box. No ISO in the UI. No engine names. No phonemizer checkbox. No multilingual
node UI. No translated knobs.

v1 dropdown:

```
English
Spanish
```

The graph stays English. The episode (dialogue, title, captions, speech) is
saved in the chosen language. Cosmetic foreign gratitude is a JSON header on
the ledger, painted by the existing title card / credits -- not a Comfy banner
and not a new node.

Mandarin Chinese is a later admitted row (CJK font, CJK wrap, zh phonemizer).
It is not in the v1 list.

## The oval

```
OTR_LedgerScriptWriter
  episode_language: [ English | Spanish ]     <- the ONLY control
           |
           |  label -> row (config/episode_languages.json)
           v
nodes/_otr_episode_languages.py
  dropdown_choices / resolve_label / resolve_ledger
           |
  writer stamps BEFORE freeze, beside source_bank:
    meta.episode_language = "es"
    meta.language_header  = "Espanol"         <- cosmetic gratitude
    meta.episode_language_receipt { registry_id, schema_version,
                                    row_revision, row_sha256 }
           |
  +------------------+------------------+------------------+
  |                  |                  |                  |
Writer prompt    Verbatim/fidelity  CastLock           CaptionBurn
row.authoring    gate BEFORE LLM    no widget;         + title card
instruction      shakespeare and    filter every       row.captions
                 public_domain      voice path by      (v1 = current
                 are English-only   languages[]        Latin Arial /
                 in v1              + engine_support   44-char wrap)
           |
  engine adapter asks row.engines[engine_id]
  kokoro stops hardcoding b / en-gb
  elevenlabs language_code from the row
  google_tts: text-native (other window owns the adapter)
           |
  Python-authored spoken strings come from row.spoken
  (Good evening / Tonight, a scene from / Good night)
           |
  measured audio duration -> video timing -> published episode
```

## Registry row (the switch never learns an engine's name)

Declarative JSON, validated fail-closed, same idea as the upscale registry.

Each row carries: `iso`, `label` (stable COMBO API), `admitted`, `sort_order`,
`native_header`, `authoring` (spoken name + writer instruction; visual prompt
iso is `en` for every v1 row), `spoken` (every Python-authored on-air string),
`captions` (font policy + wrap policy ids), `engines` (map keyed by registered
engine id: admitted, roles, opaque adapter options), `admission` (source-bank
exclusions, readiness extras), `row_revision`.

No `kokoro_lang_code` field on the shared contract. An adapter asks
`engines["kokoro"]`.

English row = today's behavior, byte for byte (British Kokoro announcer
quartet, Arial, 44/17, existing sign-on). That is the regression gate.

Spanish row = Latin captions unchanged, Kokoro torch `e`, ONNX locale proven
against installed `kokoro-onnx` before admission (do not guess it), ElevenLabs
`es`, Google text-native. `engine_support` starts at kokoro (+ cloud adapters
that pass). Bark / Dia / Chatterbox / IndexTTS join in their own commits.

## Rulings (the three seats, judged)

Holds from all three: one writer dropdown; default English; append after
`replay_from` before `gate_in`; CastLock / CaptionBurn / VideoDirector inherit;
visual and music prompts stay English; no language police; no silent engine
swap; do not edit the Google-lane dirty files.

Judged forks:

| Fork | Call | Why |
|---|---|---|
| Shakespeare + non-English | Fail at writer, before LLM | Verbatim lane cannot also be a translation |
| public_domain + non-English | Fail at writer in v1 | FIDELITY_BANKS still carries author's language as written; fuzzy prose is not a translation owner. Later row may admit it. |
| Widget site | Append | BUG-LOCAL-097; no surgery in the first commit |
| Stamp | ISO + receipt + `language_header` | Missing legacy = English. Unknown nonempty label/ISO fails. Do not coalesce garbage to English. |
| Labels | Stable COMBO strings | `English` / `Spanish` are workflow API. Resolver may honour aliases later; do not retitle in v1. |
| Voice metadata | Additive `languages: [iso, ...]` | Absent = `["en"]`. Plural for cloud voices. Never derive from `ef_` prefixes. |
| Voice filter | Every production path | Ordinary assign, gender-agnostic fallback, announcer pools, direct stamps, Kokoro engine-local fallback. One helper, wired in the same change. |
| Python-authored speech | Row `spoken` table | `_otr_line_composer.py` fallbacks air with no LLM. Admission test: every English key present on every admitted row. |
| Downstream rewrites | Carry the instruction | Ledger clean / exchange / F1 repair can undo Spanish one line at a time. |
| Replay | Ledger ISO wins | Writer returns before normal stamps. Widget/ledger mismatch fails. Receipt hash diagnoses row drift. |
| Unknown vs missing | Missing = en; unknown = fail | Silent English on a bad value is the wrong-config class. |
| Dropdown vs machine | Per repo version | Readiness fails at CastLock on that box. A per-machine menu bricks saved graphs. |
| CastLock timing | Pre-TTS, not pre-writer | Writer does not see engine widgets. Do not promise an earlier engine gate without new wiring. |
| Kokoro cache | Keyed by (lang, device) | Resident server must not run Spanish through a leftover `b` pipeline. |
| CJK / misaki[zh] | Later Mandarin row | Not v1. |
| Cosmetic header | Ledger JSON only | `meta.language_header`. Existing title/credits consume it. No new node, no JS banner, do not translate `localized_name`. |

## Fail / do not fail

Fail loud: fidelity bank + non-English (writer); engine not on the row
(CastLock); readiness extra missing (CastLock); zero eligible voices
(CastLock); replay widget/ledger mismatch (writer); unknown language token.

Do not fail: English leaking inside a Spanish generative episode; word-count
differences; F1 English regexes missing Spanish.

User-typed `episode_title` / `custom_premise` stay as typed.

## v1 is not

A Shakespeare translator. A multilingual node UI. Localized knobs. Per-character
or mixed-language episodes. A visual-prompt language dropdown. A caption
language dropdown. A requirements tax on English installs. A Google-lane edit.

## Build order (when a coder window takes this)

1. Registry JSON + resolver + tests (inject a third admitted row; it appears
   with no writer change).
2. Writer widget, fidelity gate, stamp, replay check, authoring instruction.
   Append `"English"` on canonical + variants. Widget-order / parity / links.
3. Voice `languages[]`, one eligibility helper on every selection path,
   Spanish Kokoro prefetch, coverage by language+engine+role+gender.
4. CastLock + Kokoro backends parameterized; cache identity includes language;
   English then Spanish then English on one resident process.
5. Captions inherit Latin path; accented title + dialogue fixtures.
6. ElevenLabs `language_code` from the row. Google handoff note only.
7. Live English and Spanish legs to `otr/obs/`. Negatives: Spanish+Shakespeare,
   Spanish+public_domain, Spanish+unsupported engine.

The helper is wired in the same change that builds it, or the row says why not.
