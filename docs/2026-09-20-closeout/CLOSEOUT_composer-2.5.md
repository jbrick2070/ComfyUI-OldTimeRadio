# Close-out audit (Composer 2.5) -- HEAD a7697898, 2026-09-20

Reviewer: **Composer 2.5** (Cursor). Read-only audit per operator brief; findings grounded at file:line.

---

## MUST-FIX before ship

| Location | What is wrong | Exact fix |
|---|---|---|
| `nodes/cast_lock.py:1944-1947` | **Failure #11.** `_stamp` clears any `v2/*` `voice_preset` when the stamped engine is not `bark`. Provisional Lemmy (`fallback="provisional_route"`, e.g. Chatterbox) arrives with writer-stage `v2/en_speaker_8`; `_stamp` wipes it before tier is stamped, so `test_the_writer_stage_bark_preset_SURVIVES_the_normalizer` fails and credits/identity for ~11% cameo runs lose the Bark preset. | Replace the block with: `if str(getattr(ref, "engine", "") or "") != "bark": leftover = str(entry.get("voice_preset") or ""); if leftover.startswith("v2/") and fallback != "provisional_route": entry["voice_preset"] = ""` |

---

## SHOULD-FIX

| Location | What is wrong | Exact fix |
|---|---|---|
| `tests/test_canonical_replay.py:372` | **Failure #2.** `BatchCharacterVoices` / `AnnouncerVoice` inherit `OTRVoiceNodeBase.generate(self, script_json, ledger_json="", gate_in="", ...)` (`nodes/_otr_voice_node_common.py:1203`). The test passes `generate(sj, "kokoro", ledger_json=sj)`, so `"kokoro"` binds `ledger_json` positionally while `ledger_json=sj` is also passed -- `multiple values for argument 'ledger_json'`. Replay never runs in the test even when production replay would work. | Change line 372 to: `audio, log_, done = cls().generate(sj, ledger_json=sj)` (engine is ignored on replay passthrough anyway). |
| `tests/test_hf_env_offline.py:107-117` | **Failure #9.** Stub `guarded_auto_download(repo_id, *, hub_root)` does not accept `progress_pbar`, which `catalog.auto_download_if_missing` now forwards (`nodes/_otr_model_catalog.py:2318-2429`). | Change the stub to `def guarded_auto_download(repo_id, *, hub_root, **kwargs):` and pass `**kwargs` through to `real_auto_download(..., _snapshot_download=network_forbidden, **kwargs)` (or add explicit `progress_pbar=None` and forward it). |

---

## DOCS-ONLY

| Location | What is wrong | Exact fix |
|---|---|---|
| `README.md:44` | Says "Two of the **five** story banks" read RSS; the shipped runnable set is **six** banks plus the non-runnable `custom_source_bank` signpost (`apple/BANKS.md:16-30`: scifi_news_pro, shakespeare, public_domain, media_archive, my_story, original). | Replace "five" with "six" (still "Two of the six..."). |
| `README.md:458-459` | Claims "**sixteen** generated graphs" under `workflows/variants/`; tree at HEAD has **21** `otr_*.json` variant files (plus `workflows/otr_canonical.json`). | Replace "sixteen" with "twenty-one" (or "21") and optionally note the count grows when variants are regenerated. |
| `apple/RUN.md:89-91` | Tells installers to pin away from Shakespeare/Public Domain for non-English because those lanes "refuse a translation request"; `nodes/_otr_episode_languages.py:559-567` admits every lane for every language row (exclusions list empty on shipped rows), and `apple/BANKS.md:34-38` documents Shakespeare performing translated passages. | Rewrite the bullet to match BANKS/MULTILINGUAL: non-English Shakespeare uses vendored translations when present (`config/source_banks/shakespeare/translations/manifest.json`, 43 scenes) or writer translation otherwise; remove "refuse a translation request" for Shakespeare/Public Domain. |

---

## NOTHING FOUND

| Checklist item | Result |
|---|---|
| **1 -- Loose ends in day's `.py` diff (orphan helpers in `nodes/`/`scripts/`)** | `_line_tokens` / CJK chunking in `nodes/_otr_passage_selector.py` is called from `_split_long_line` and `_pack_lines` in the same module; `select_passage` / `chunk_speech` remain wired from `nodes/_otr_verbatim_lane.py:281+`. New helpers in `scripts/otr_vendor_scan.py` are dev-only (`.comfyignore:32` excludes `scripts/*` except negated workers); no production import of that module beyond tests. |
| **1 -- TODO/FIXME/XXX added today in changed `.py`** | nothing found (added `print` lines in `scripts/otr_vendor_scan.py` are CLI diagnostics for the scan tool, not shipped). |
| **1 -- Debug print in changed production `nodes/`** | nothing found |
| **1 -- Hard-coded absolute path in changed `.py`** | nothing found in changed files (`tests/test_vendor_scan_furniture.py` reword removed operator path per ship note). |
| **1 -- Non-ASCII inside log/print strings in changed `nodes/`** | nothing found (`nodes/_otr_passage_selector.py` uses CJK only in comments/docstrings, not prints). |
| **1 -- Comment/docstring contradicting code beside it in changed `nodes/`** | nothing found |
| **4 -- `source_bank` tooltip** | Matches `_otr_story_routing` + `_ROLLS.resolve_bank_selection` behavior (`nodes/OTR_LedgerScriptWriter.py:2202-2238`). |
| **4 -- `source_ref` tooltip** | Matches optional pin / fail-loud contract (`nodes/OTR_LedgerScriptWriter.py:2256-2267`, resolved in `_otr_writer_inputs.py`). |
| **4 -- `episode_language` tooltip** | Matches post-96346118 behavior (43 scenes, six translation isos, writer fallback); counts verified: manifest `len(scenes)==43`, isos `es/fr/it/ja/pt/zh` (`config/source_banks/shakespeare/translations/manifest.json`). |
| **4 -- `visual_style` tooltip** | Matches roll + prompt-tail-only effect; canonical procedural video lanes still do not read still style (`nodes/OTR_LedgerScriptWriter.py:2281-2298`). |
| **4 -- `lemmy_cameo` tooltip** | Matches `_LEMMY_CAMEO_CHOICES` and `resolve_lemmy_cameo` / OS-entropy ~11% roll (`nodes/_otr_writer_inputs.py:91-94`, `nodes/_otr_casting.py:1277+`). |
| **3 -- `pyproject.toml` vs tree** | `version = "2.1.6"`, static deps list present; `node_list.json` has 25 nodes matching pack contract (`tests/test_node_list_manifest.py:38-39`). |
| **3 -- `README.md` template / canonical names** | `otr_canonical` and Extensions path match `workflows/otr_canonical.json`; eight admitted languages match eight `"admitted": true` rows in `config/episode_languages.json`. |
| **5 -- `.comfyignore` leaks** | `docs/`, `kibitz-runs/`, `tmp/`, `_tmp_*`, `tests/`, most `scripts/*` excluded; `config/` explicitly must ship (`.comfyignore:220-221`); `!scripts/_otr_chatterbox_worker.py`, `!scripts/_otr_dia_worker.py`, `!scripts/otr_mesh_stage_blender.py`, `!scripts/otr_openrouter_refresh.py` negations present; `workflows/variants/*.md` excluded but `.json` variants not excluded (`.comfyignore:246-252`). No tracked `kibitz-runs/` directory at HEAD. |
| **5 -- `.comfyignore` over-excludes must-ship assets** | nothing found (`config/source_banks/**` not listed; `node_list.json`, `workflows/otr_canonical.json` not listed). |

---

**Ship note #2 / #11 (looked):** #2 is a stale test call at `tests/test_canonical_replay.py:372` against `OTRVoiceNodeBase.generate` at `nodes/_otr_voice_node_common.py:1203` (SHOULD-FIX row). #11 is `_stamp` at `nodes/cast_lock.py:1944-1947` clearing Lemmy's writer `voice_preset` on provisional non-bark stamps (MUST-FIX row). Remaining twelve in `docs/2026-09-19-ship-regression.md` need fixture/artifact/profile work or EXPECTED_FAILED entries, not a <10-line code patch here.

**ONE THING before Reddit:** `nodes/cast_lock.py:1944-1947` -- provisional Lemmy (~11% default rolls) must keep `voice_preset` when `fallback == "provisional_route"`; replay (#2) is test-only and rare for fresh installs.

COUNTS: must=1 should=2 docs=3
