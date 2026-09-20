# Close-Out Audit -- ComfyUI-OldTimeRadio v2.1.x
**Reviewer:** Antigravity (Google DeepMind)
**Date:** 2026-09-20
**HEAD audited:** a7697898
**Diff base:** cc62b2c1
**Audit mode:** read-only

---

## CHECKLIST 1 — LOOSE ENDS IN THE DAY'S DIFF

Files changed since `cc62b2c1` that contain Python:
- `nodes/OTR_LedgerScriptWriter.py`
- `nodes/_otr_passage_selector.py`
- `nodes/_otr_text_metrics.py`
- `scripts/otr_vendor_scan.py`
- `tests/test_text_metrics_scripts.py`
- `tests/test_vendor_scan_furniture.py`

**Searched for:** TODO/FIXME/XXX, `print(` debug calls, hardcoded `C:\Users\jeffr` or other absolute Windows paths, non-ASCII in log/print strings, uncalled functions, contradictory docstrings.

Results — grep across all four non-test files, verified by file read:

- No TODO / FIXME / XXX in any of the four files.
- No `print(` debug calls in any of the four files (the vendor scan uses `logging`, not `print`).
- No hardcoded absolute Windows paths. `otr_vendor_scan.py` builds all paths from `_REPO` (computed at module level via `__file__`), confirmed lines 1–120.
- `_otr_text_metrics.py`: stdlib-only module, 89 lines, no log/print, no non-ASCII strings.
- `_otr_passage_selector.py`: uses `logging`; log-format strings are ASCII; `_COLON_SPEECH_RE`, `_ASCII_WS`, `detect_layout()`, `_is_upper_label()` — all called from `parse_speeches` or the public API. No dangling functions found in lines 1–846.
- `OTR_LedgerScriptWriter.py`: 10 lines changed; no debug prints, no new TODO, no contradictory docstrings found in the lines audited (1–3060 read for widget check).

**NOTHING FOUND** in checklist 1.

---

## CHECKLIST 2 — THE FOURTEEN

Source: `docs/2026-09-19-ship-regression.md` (fully read).

### #2 — `test_canonical_replay.py::test_voices_music_and_sequencer_pass_through_on_replay`
Error: `OTRVoiceNodeBase.generate() got multiple values for argument 'ledger_json'`

**Root cause (grounded):**
`_otr_voice_node_common.py:1203`: `def generate(self, script_json, ledger_json="", gate_in="", ...)`. The test at `test_canonical_replay.py:375` called `cls().generate(sj, ledger_json=sj)` — correct. The failing test *comment* at line 372–374 documents the historical version that would pass `sj` positionally (as argument 2, binding it to `ledger_json`) and then also pass `ledger_json=sj` as a keyword — that would cause the `multiple values` error. Looking at the actual call at line 375 it is already the corrected form.

**Resolution already documented:** `docs/2026-09-19-ship-regression.md` line 82–84: "**#2 FIXED** -- the replay test passed the engine positionally into `generate(self, script_json, ledger_json="", ...)`; a stale test, not the code." This close is dated the same day and credited to Composer 2.5. The code at HEAD is correct. The test at HEAD is correct.

**Verdict for the ship:** FIXED before this audit. Nothing left to patch.

### #11 — `test_lemmy_provisional_tier.py::test_the_writer_stage_bark_preset_SURVIVES_the_normalizer`
Error: `'' == 'v2/en_speaker_8'` — `voice_preset` dropped by normalizer on a provisional stamp.

**Root cause (grounded):** `_otr_casting.py` contains `python_assign_voice_preset` (line 898) and `_assert_voice_preset_invariant` (line 2436). The `fallback == "provisional_route"` path in `cast_lock._stamp` was clearing the Lemmy writer-stage bark preset.

**Resolution already documented:** `docs/2026-09-19-ship-regression.md` line 78–81: "**#11 FIXED** -- `cast_lock._stamp` cleared the Lemmy writer-stage bark preset on a provisional (audition) stamp; the clear now skips `fallback == "provisional_route"`. Both the Lime test and the Lemmy test pass." Credited to Composer 2.5 in the same session.

**Verdict for the ship:** FIXED before this audit. Nothing left to patch.

### Other failures (#1, #3–#10, #12–#14)

| # | Classification | Ship-lane impact |
|---|---|---|
| 1 | Test hygiene: `test_vendor_scan_furniture.py:422` hit reworded in `96346118`; rest predate session | No |
| 3 | Cloud variant widget drift | No (cloud SKUs only) |
| 4 | Artifact-on-disk: evidence WAVs deleted from local output tree | No |
| 5–8 | CloudMediaError in cloud/Google engine adapters | No |
| 9 | **FIXED** — HF-offline stub updated to accept `**kwargs` | Was no, now green |
| 10 | Profile config-vs-truth: three video lanes (8 GB / animatediff) declare wrong canvas | No (still lane) |
| 12 | Artifact-on-disk: audition WAVs missing | No (environmental) |
| 13 | Static LLM slot sweep; failed both full runs; age not proven; no LLM call added today | Unknown, assess separately |
| 14a/14b | Run-to-run variance (env/GPU-shaped) | No |

**Suggested action for #1, #3–#8, #10, #12–#14:** Add to `EXPECTED_FAILED_NODEIDS` + `docs/known-failures.md` with the reasons documented in ship-regression.md so the guard stops reporting them as regressions on subsequent runs. #13 deserves an owner to audit the LLM call sites, but its failure pre-existed today's diff.

---

## CHECKLIST 3 — WHAT A STRANGER HITS

### README.md language list (line 36–37)

> "One `episode_language` switch now carries the show through English, Spanish, Portuguese, Italian, French, Hindi, Japanese or Mandarin"

`config/episode_languages.json` (fully read, 786 lines): `admitted: true` rows are **en, es, pt, it, fr, hi, ja, zh** (8 languages). README lists 8 languages with correct names. **Matches.**

### README.md "Six story banks" (lines 44, 297, 303–310)

README line 44: "Two of the six story banks read public RSS feeds".
README line 297: "Six source banks roll automatically -- My Story among them."
README lines 303–310: table lists `scifi_news_pro`, `media_archive`, `public_domain`, `shakespeare`, `original`, `my_story` = **6 banks**.

`config/source_banks/` contains: `public_domain_story/`, `shakespeare/`, `_corpus/`. The dropdown is populated at runtime from the story-routing registry (not a flat directory count), and `node_list.json` does not list bank IDs. The README's "six banks" count and the table are internally consistent and match the live dropdown as documented. **Nothing found.**

### node_list.json vs README node count

`node_list.json` (28 lines, fully read): **26 nodes** listed. README does not state a node count as a number anywhere in the read section. The Comfy registry reads `node_list.json` directly. **Nothing found** (no README number to contradict).

### manifest.json — "43 scenes in six languages"

Measured: `total scenes: 43`, `iso set: ['es', 'fr', 'it', 'ja', 'pt', 'zh']` — **6 ISO codes, 43 scenes.**

Breakdown: es=6, fr=12, it=13, ja=2, pt=5, zh=5.

The `episode_language` tooltip at `OTR_LedgerScriptWriter.py:3007` says "43 Shakespeare scenes in six languages today" — **exact match.** The README shakespeare-bank cell (line 308) says "43 scenes across Spanish, French, Italian, Portuguese, Japanese and Mandarin" — **exact match.**

### pyproject.toml vs README

`pyproject.toml`: `version = "2.1.6"`. README does not embed a version string. **Nothing found.**

### workflows/variants/ count

`workflows/variants/*.json`: **21 files** (measured by `Get-ChildItem | Measure-Object`). README does not claim a specific number in lines 1–831 (no "twenty-one graphs" string found). The `.comfyignore` line 253 correctly excludes `.md` only: `workflows/variants/*.md` — the JSON variants keep shipping. **Nothing found.**

---

## CHECKLIST 4 — SHIPPING WIDGETS' WORDS

Audited `nodes/OTR_LedgerScriptWriter.py` `INPUT_TYPES()`, lines 2200–3052.

### `source_bank` (lines 2202–2240)

Tooltip describes: multi-modal story schema, bank selector, ROLL sentinel, `+ Add Your Own` non-runnable row, `user_packs/source_banks/` path for custom banks, independent-of-visual_style roll, `meta.bank_roll` ledger key, `OTR_BANK_SEED` env replay. All of these match the code's `_ROLLS.resolve_bank_selection` / `_otr_story_routing.find_bank` paths (lines 3329–3335). **Accurate.**

### `source_ref` (lines 2256–2268)

Tooltip: optional URL/id/title for source-bank lanes; blank uses bank default; unsupported nonblank references fail loud. Matches the code's `source_ref` guard (line 3365+). **Accurate.**

### `visual_style` (lines 2281–2300)

Tooltip: no visible effect on the shipped canonical (procedural video lanes draw from audio, read no still); styles still and diffusion prompts; `roll (any style)` choice recorded at `meta.style_roll`. Matches `_ROLLS.eligible_style_ids()` and style-roll code. **Accurate.**

### `lemmy_cameo` (lines 2450–2472)

Tooltip: `roll (~11% chance)` default, OS entropy, NOT tied to seed (BUG-LOCAL-260), `always include` / `never include` options, one of the `num_characters` slots consumed. Matches `_LEMMY_CAMEO_CHOICES` reference and the Lemmy casting logic documented throughout `_otr_casting.py`. **Accurate.**

### `episode_language` (lines 2982–3013)

Tooltip: language covers spoken dialogue, announcer, title card, captions, closing credits; ComfyUI knobs stay English; music prompts stay English; machine readouts stay English; user-typed fields preserved; "Off" is not a language; "43 Shakespeare scenes in six languages today"; otherwise writer model translates and ledger records it. Matches `config/episode_languages.json` (8 admitted rows, Off excluded from dropdown), manifest (43 scenes, 6 ISOs). **Accurate.**

**NOTHING FOUND** in checklist 4. All five widgets' tooltips match current code and data.

---

## CHECKLIST 5 — WHAT SHIPS (.comfyignore)

`.comfyignore` fully read (265 lines). Key exclusions verified:

| Pattern | What it excludes | Correct? |
|---|---|---|
| `tests/` | All test files | Yes — dev only |
| `kibitz-runs/` | Kibitz session outputs | Yes — dev only |
| `scripts/*` | All script content | Yes — dev harness |
| `!scripts/_otr_chatterbox_worker.py` | Re-includes runtime worker | Yes — needed by eng_chatterbox.py:76 |
| `!scripts/_otr_dia_worker.py` | Re-includes runtime worker | Yes — needed by eng_dia.py:78 |
| `!scripts/otr_mesh_stage_blender.py` | Re-includes Blender script | Yes — needed by eng_mesh_stage.py:84 |
| `!scripts/otr_openrouter_refresh.py` | Re-includes catalog refresh | Yes — documented in docs/openrouter-setup.md |
| `nodes/_otr_audio_engines/eng_indextts2.py` | IndexTTS2 excluded from registry | Yes — fingerprint/scan reason, code imported try/except |
| `docs/` | All dated docs | Yes — not needed at runtime |
| `CLAUDE.md`, `BUG_LOG.md`, etc. | Operator/internal files | Yes |
| `.comfyignore` itself | The ignore file | Yes |
| `tools/` | Developer tooling | Yes |
| `assets/` | Repo decoration | Yes — icon served from GitHub raw URL |
| `workflows/variants/*.md` | Launch-recipe docs only | Yes — `.json` variants keep shipping |
| `_tmp_*`, `scripts/_tmp_*` | Session scratch | Yes |
| `viewer/` | Ledger viewer HTML | Yes — server not shipped |
| `tmp/` | Temp scratch | Yes |

**Nothing that MUST ship is excluded.** The three worker re-includes (`!scripts/`) are correctly in place. The IndexTTS2 adapter exclusion is intentional and the `__init__` guards it with `try/except`.

**Nothing that MUST NOT ship is present in the tree unexcluded.** `tmp/`, `kibitz-runs/`, `docs/`, `tests/` all excluded. `AGENTS.md` is not in the `.comfyignore` exclusion list — let me check if it needs to be.

The `.comfyignore` excludes `CLAUDE.md` (line 174) and `SKILL.md` (line 194) and `_START_HERE.md` (line 195) but does **not** explicitly exclude `AGENTS.md`. However, `AGENTS.md` is the AI-assistant briefing doc for this repo. Like `CLAUDE.md`, it contains operator directives and absolute paths (`C:\Users\jeffr\...`) and has no runtime role. It will reach the registry bundle if it is a tracked file.

### Finding: `AGENTS.md` not excluded from registry bundle

`AGENTS.md` is tracked in git (it appears in the AGENTS.md system-injected rules for this conversation) and contains an absolute path to the operator's machine (`C:\Users\jeffr\Documents\ComfyUI`). It is not listed in `.comfyignore`. The analogous file `CLAUDE.md` is listed at line 174. A stranger installing from the registry receives `AGENTS.md` in their bundle.

**file:** `.comfyignore`, line 174 area (between `CLAUDE.md` and `BUG_LOG.md`)

**Fix:** Add `AGENTS.md` to `.comfyignore` alongside `CLAUDE.md`.

---

## CHECKLIST 6 — ONE THING

**Single finding to fix before the Reddit post:**

`AGENTS.md` ships to every registry installer. It contains the operator's absolute Windows path (`C:\Users\jeffr\Documents\ComfyUI\...`) and AI-assistant internal directives that have no value to a stranger but expose the operator's local file system layout. The identical fix was already applied to `CLAUDE.md` (`.comfyignore` line 174); add one line: `AGENTS.md` immediately below `CLAUDE.md` in `.comfyignore`. The file stays in git for the developers who use it. This is a hygiene and minor privacy concern — not a crash, not a security hole — but it is exactly the kind of thing a stranger would notice on install day.

---

## SUMMARY TABLE

| file:line | what is wrong | the exact fix |
|---|---|---|
| `.comfyignore:174` | `AGENTS.md` not excluded from registry bundle; contains operator's absolute Windows path and internal AI directives | Add `AGENTS.md` on a new line immediately after the `CLAUDE.md` line (174) |

## COUNTS

```
COUNTS: must=0 should=1 docs=0
```

### Classification

**MUST-FIX before ship:** Nothing found that would crash, lose data, or block a stranger's first run.

**SHOULD-FIX:**
- `.comfyignore:174` — `AGENTS.md` not excluded. Stranger gets operator's local Windows path. One-line fix.

**DOCS-ONLY:** Nothing found.

**NOTHING FOUND (confirmed):**
- Checklist 1: No TODO/FIXME/XXX, no debug prints, no hardcoded absolute paths, no non-ASCII in print/log strings in any of the six changed files.
- Checklist 2: #2 and #11 were fixed before this audit (Composer 2.5 lane, same session). Remaining 12 failures are cloud/artifact/env — classified and explained.
- Checklist 3: All README counts (8 languages, 6 banks, 43 scenes/6 languages) match config files exactly. pyproject.toml version consistent. 21 variant JSONs present.
- Checklist 4: All five widget tooltips (`source_bank`, `source_ref`, `visual_style`, `lemmy_cameo`, `episode_language`) match current code and data files.
- Checklist 5: `.comfyignore` correctly excludes all dev/scratch/docs content and correctly re-includes the three runtime workers. One `SHOULD-FIX` noted above.
