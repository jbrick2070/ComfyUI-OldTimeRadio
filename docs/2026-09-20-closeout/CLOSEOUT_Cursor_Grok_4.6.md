# Close-out audit -- Cursor Grok 4.6

Reviewer: Cursor Grok 4.6 (this window). Read-only. Tree at `5e4a7c1a` (brief named `a7697898`; two later commits already on `main` / `origin/main`: `3549c900`, `5e4a7c1a`). Day's Python range `cc62b2c1..HEAD` is still the same six files. No Python process was started.

## MUST-FIX before ship

nothing found

## SHOULD-FIX

nodes/OTR_LedgerScriptWriter.py:3797 | Comment next to the live vendored-corpus caller still says "Four scenes are vendored out of ninety-eight cells"; the same file's tooltip at 3007, and `config/source_banks/shakespeare/translations/manifest.json` (43 `"iso"` rows, 43 `"verdict": "READY"`, languages es/fr/it/ja/pt/zh), say 43 scenes in six languages. | Replace the comment with: `Forty-three scenes are vendored today (six languages); everything else keeps the model translation.`

nodes/OTR_LedgerScriptWriter.py:2467 | `lemmy_cameo` tooltip: `"'always' / 'never' consume one of the num_characters slots"`. Code at `nodes/_otr_casting.py:1543-1551` consumes a slot only when `lemmy_hit` is true. `never include` maps to `force_lemmy=False` (`nodes/_otr_writer_inputs.py:96`) and does not consume a slot. | Change that sentence to: `"'always' consumes one of the num_characters slots, exactly as a natural roll-hit does. 'never' does not consume a slot."`

.comfyignore (no `.cursor` rule) | Seven tracked files under `.cursor/` (`git ls-files .cursor`: five rules + `skills/refresh/SKILL.md` + `scripts/brief.py`) are not matched by `.comfyignore`. A temp repo whose `.gitignore` was a copy of `.comfyignore` left `.cursor/rules/otr-review.mdc` not-ignored. A Manager zip would carry operator Cursor rules. | Append a line: `.cursor/`

tests/test_canonical_replay.py:372 | **#2 looked at.** The test calls `cls().generate(sj, "kokoro", ledger_json=sj)`. Live signature is `generate(self, script_json, ledger_json="", gate_in="", ...)` at `nodes/_otr_voice_node_common.py:1203`. `"kokoro"` binds to `ledger_json` and `ledger_json=sj` then raises `got multiple values for argument 'ledger_json'`. Production ComfyUI calls by INPUT_TYPES names (`voice_input_types` at 390-437: `script_json`, `ledger_json`, `gate_in` only -- no `engine` widget). Engine is leftover kwargs / CastLock stamp (1225-1237). A `replay_from` Queue does not take this TypeError. | Replace the call with: `audio, log_, done = cls().generate(sj, ledger_json=sj)`

tests/test_hf_env_offline.py:107 | **#9.** Stub is `def guarded_auto_download(repo_id, *, hub_root):`. Production forwards `progress_pbar=` (`nodes/_otr_model_catalog.py:2318`, `nodes/_otr_model_loader.py:2336-2339`). | Change the stub to: `def guarded_auto_download(repo_id, *, hub_root, progress_pbar=None, **_kwargs):`

nodes/cast_lock.py:1944 | **#11 looked at.** The drop is `_stamp`, not `_normalize_row_for_tier_switch`. `_TIER_SWITCH_CLEARED_FIELDS` (152-160) and the comment at 146-151, plus `config/cast_pools.py:1268` (`lemmy_row`: "CastLock's stamp writes engine and reference, never `voice_preset`"), say `_stamp` must not touch `voice_preset`. `_stamp` at 1944-1947 clears any `v2/` leftover when `ref.engine != "bark"`. The test (`tests/test_lemmy_provisional_tier.py:605-608`) stamps chatterbox and expects `v2/en_speaker_8` to survive. Default still-lane is kokoro: clearing leftover `v2/` there is the Lime 20260917 credits fix (1938-1943) and is correct for ordinary rows. Chatterbox/bark/IndexTTS2 two-stage still wants Lemmy's writer-stage preset. | In `_stamp`, keep the Lime clear but skip Lemmy: `if leftover.startswith("v2/") and str(entry.get("name") or "").strip().upper() != "LEMMY": entry["voice_preset"] = ""`

tests/conftest.py:184 | **#14a / #14b** (ship note: run-to-run swap, GPU held, not in today's diff). | Add to `EXPECTED_FAILED_NODEIDS` and a `docs/known-failures.md` pair: `tests/test_comfy_credential_rip.py::test_llm_hosts_capture_the_key_through_set_auth` -- env/GPU-held variance, not still-lane. `tests/test_ltx_8gb_canonical_canvas.py::test_the_8gb_variant_workflow_agrees_with_the_declaration` -- same class, 8 GB LTX variant, not still-lane. Exit: a quiet-GPU full run that fails the same nodeid twice.

## DOCS-ONLY

README.md:119 | "sixteen saved graphs further down this page" -- `git ls-files workflows/variants/*.json` is 21 (plus `workflows/otr_canonical.json`). | Say "twenty-one saved graphs" (or "sixteen local plus five cloud").

README.md:458 | "sixteen generated graphs in `workflows/variants/`" -- 21 JSON files there. The table under the same heading already lists 21 named graphs. | Same number as the table: 21.

README.md:297 | "Five source banks roll automatically; a sixth takes your own premise." `my_story` rolls (`README.md:310`, `nodes/story_packs/banks.json:208` `auto_select: true`). Six runnable banks roll; `custom_source_bank` is the non-runnable seventh. | "Six source banks roll automatically, including My Story; a seventh row is the + Add Your Own signpost."

README.md:308 | "`shakespeare` \| A Folger scene, adapted with the author's own language carried as written." Code since `96346118` performs a real translator's public-domain text when the corpus holds the scene (43 rows, six languages); otherwise the model's translation. The `episode_language` tooltip was corrected in `d5aa8832`; this row was not. `apple/RUN.md:90-94` already has the true sentence. | Copy the RUN.md sentence onto this bank row. Keep the Folger CC BY-NC note.

apple/VOICES.md:10 | "all sixteen saved variants" -- 21 variant JSONs. | "all twenty-one saved variants"

apple/STYLES.md:50 | "the canonical and all sixteen variants" -- 21. | "all twenty-one variants"

apple/UPSCALERS.md:6 | "all sixteen saved variants" -- 21. | "all twenty-one saved variants"

apple/UPSCALERS.md:160 | "the canonical and all sixteen variants" -- 21. | "all twenty-one variants"

apple/MULTILINGUAL.md:124 | Shipped guide (`.comfyignore` does not exclude `apple/`). Still says shakespeare "performs its passage TRANSLATED" / "translated once" and never names the vendored translator corpus. | Add the same "translator's public-domain text when the corpus holds that scene (43 / six languages), otherwise the model" clause that `apple/RUN.md:90-94` already has.

apple/BANKS.md:21 | "A scene from the Folger texts" -- true for the English source, silent on the 43 vendored translations. Line 36 already says the passage is performed translated. | Append: "English from Folger; other languages from a vendored translator when present."

apple/MUSIC.md:39 | "sixteen of the seventeen shipped graphs" -- 1 canonical + 21 variants = 22 graphs; CPU is the MusicGen exception. | "twenty-one of the twenty-two shipped graphs" (canonical + 21 variants; `otr_cpu_low` is the MusicGen one).

## NOTHING FOUND

day's .py / orphan function | `_line_tokens` (`nodes/_otr_passage_selector.py:204`) is called from `_split_long_line` and `_halve` in the same module. `CJK_RUN_RE` is imported by that module from `_otr_text_metrics`. New `scripts/otr_vendor_scan.py` helpers are called from `main` in the same script. `select_passage` has a production caller at `nodes/_otr_verbatim_lane.py:281`. | --

day's .py / TODO FIXME XXX | Only hit is `scripts/otr_vendor_scan.py:75` `"TODOS": "ALL"` (a PDF field name, not a work marker). | --

day's .py / debug print | `otr_vendor_scan.py` `[scan]` prints are the CLI. No leftover debug print in the four production files. | --

day's .py / hardcoded absolute path | No `C:\Users` / `C:/Users` / `/home/` in the six changed `.py` files. | --

day's .py / non-ASCII in print or log | CJK lives in comments and regexes, not in `print` / `log.*` strings. Vendor-scan prints are ASCII. | --

checklist 4 / source_bank tooltip | `nodes/OTR_LedgerScriptWriter.py:2207-2238` matches `nodes/story_packs/banks.json`: roll sentinel, `custom_source_bank` is the only `runnable: false` row, `apple/EXTENDING.md` ships, `docs/EXTENDING_OTR.md` is named as git-tree only. | --

checklist 4 / source_ref tooltip | `2261-2266` still matches: optional pin, blank = bank default, nonblank fails loud. Used as the scene key for `vendored_text` (`3818-3820`). Incomplete on translations, not false. | --

checklist 4 / visual_style tooltip | `2287-2297` matches the shipped procedural graph and `README.md:340-345` (no visible effect until a still/diffusion lane). Ten styles: 9 JSON packs + `visual_storybased`. | --

checklist 4 / episode_language tooltip | `3001-3010` (d5aa8832) matches the corpus: 43 READY scenes, six isos. | --

checklist 3 / languages | `README.md:36-37` lists English, Spanish, Portuguese, Italian, French, Hindi, Japanese, Mandarin. `config/episode_languages.json` has eight `admitted: true` rows. | --

checklist 3 / node count | `node_list.json` has 25 keys. `__init__.py:452-455` prints `All {_total} nodes`. README does not claim 25/32/34. | --

checklist 3 / voice engines | `README.md:376-377`: six in a Manager install, IndexTTS2 GitHub-only. Matches `.comfyignore:167` excluding `eng_indextts2.py` and the six CastLock character engines in `nodes/_otr_engine_profiles.py:61-62` minus that one. Sonilo is music, not a voice. | --

checklist 5 / must-ship | Temp repo with `.comfyignore` as `.gitignore`: `node_list.json`, `workflows/otr_canonical.json`, `workflows/variants/otr_16gb_still.json`, `config/source_banks/shakespeare/translations/manifest.json`, `.../ja/romeo_juliet_1_1.txt` were not ignored. `scripts/*` ignores `otr_vendor_scan.py`; `!scripts/_otr_chatterbox_worker.py` keeps the worker. | --

checklist 5 / must-not-ship (the listed class) | Same proof: `docs/`, `kibitz-runs/`, `tmp/`, `tests/`, `CLAUDE.md` match exclude rules. Untracked `*.secret` never ships. Tracked `tmp/GPT_SHAKESPEARE_HUNT_3_RESULTS_VERIFIED.md` is covered by `tmp/`. | --

pyproject.toml | `version = "2.1.6"`, Icon URL is `/main/assets/otr_icon.gif`, static dependencies, no stale node count or graph list. | --

#3 #4 #5-8 #10 #12 #13 | Cloud SKU drift, missing `otr/episodes/lemmy_cross_engine` artifacts, cloud `corrupt_output`, 8 GB/AnimateDiff canvas, missing kokoro wav, LLM slot sweep: already classified in `docs/2026-09-19-ship-regression.md`. No confident <10-line production patch from this pass (no pytest was run). | Park with the ship note's reasons if the guard must go quiet; do not invent a code fix from a listing.

## The one thing

Fix `README.md:308` before the Reddit post. The day's work is that Japanese and Chinese Shakespeare can be performed from the translator's own words, and the writer tooltip already says so; the front-page bank table still tells a stranger it is Folger English only. `apple/RUN.md:90-94` is the sentence to copy. Leaving that row stale is the same class of lie `d5aa8832` already paid a commit to remove from the widget.

COUNTS: must=0 should=7 docs=9
