# Close-out audit -- Gemini (Antigravity)

Reviewer: **Gemini (Antigravity)**. Read-only close-out audit per operator brief; findings grounded at `file:line`. Tree at HEAD `fd98c168` (brief named `a7697898`; commits `8d064235`, `65529023`, `7f284264`, and `fd98c168` landed on `main` during audit). No Python process was started; all scratch operations executed in `%TEMP%`.

---

## MUST-FIX before ship

nothing found

*(Note on previous candidates: Failure #11 at `nodes/cast_lock.py:1944-1947` was the only code defect that could reach an episode by clearing Bark presets on provisional Lemmy cameos; it was verified closed in `8d064235`. The package bundle leak of `.cursor/` into the Comfy Registry zip was verified closed in `.comfyignore:8` by `7f284264`. With those merged into `main`, no MUST-FIX defect remains.)*

---

## SHOULD-FIX

nodes/OTR_LedgerScriptWriter.py:2329 | `custom_premise` widget tooltip claims: `"At least one creative field must contain text."` This is a stale holdover from before the standing premise floor was introduced on 2026-09-13. Under current code (`nodes/_otr_story_input.py:212-238`), an episode running `my_story` with all four creative fields blank does NOT fail; it automatically falls back to `DEFAULT_IDEA` (the standing premise) so that unattended rolls succeed. Shipped workflows intentionally ship with all four fields blank (`tests/test_no_shipped_graph_carries_a_premise.py`). Telling users that at least one creative field must contain text misleads them into believing a blank run will fail. | Replace `"At least one creative field must contain text. "` with `"Leaving fields blank falls back to the standing premise so unattended rolls succeed. "`

---

## DOCS-ONLY

apple/BANKS.md:36-37 | Under `### Language compatibility`, the text states: `"language; shakespeare performs its selected passage translated, speakers and cut unchanged, and records both hashes on the ledger."` While line 21 in the table above was updated in commit `7f284264` to mention the vendored translator corpus, lines 36-37 were missed in that update and still claim that Shakespeare only performs the passage translated, silent on the 43 vendored public-domain translation scenes in six languages (`config/source_banks/shakespeare/translations/manifest.json`). | Replace lines 36-37 with: `"language; shakespeare performs its selected passage from a vendored public-domain translation when present (43 scenes in six languages), or translated by the model otherwise, speakers and cut unchanged, and records both hashes on the ledger."`

---

## NOTHING FOUND

| Checklist item | Result / Grounded check |
|---|---|
| **1 -- Loose ends in day's `.py` diff (`cc62b2c1..HEAD`)** | All helper functions (`_line_tokens`, `_median`, `_word_baselines`, `rows_from_coordinates`, `fused_words`, `label_key`, `parse_folds`, `_edge_folio_parts`, `_window_bounds`, `slice_pages`, `window_truncates`) are actively called in their modules or tests; `select_passage` remains wired from `nodes/_otr_verbatim_lane.py:281+`. |
| **1 -- TODO/FIXME/XXX added today in changed `.py`** | No work markers added; the only hit in changed `.py` is `scripts/otr_vendor_scan.py:75` (`"TODOS": "ALL"` as a PDF dictionary key). |
| **1 -- Debug print in changed production `nodes/`** | No debug prints; `scripts/otr_vendor_scan.py` CLI prints are diagnostic CLI outputs for the scanning tool, not shipped. |
| **1 -- Hard-coded absolute path in changed `.py`** | No hardcoded local operator paths added in production files; the test path in `tests/test_vendor_scan_furniture.py` was cleaned in `96346118`. |
| **1 -- Non-ASCII inside log/print strings in changed `nodes/`** | CJK characters in `nodes/_otr_passage_selector.py` and `_otr_text_metrics.py` appear exclusively in comments, docstrings, and regex definitions, never inside log/print format strings. |
| **1 -- Comment/docstring contradicting code beside it in changed `nodes/`** | Line 3797 comment in `nodes/OTR_LedgerScriptWriter.py` was updated in `7f284264` to cite 43 scenes in six languages. |
| **2 -- The fourteen failures (#1, #3-#8, #10, #12, #13, #14a/b)** | #2 (replay test positional kwarg), #9 (HF-offline stub kwargs), and #11 (Lemmy provisional stamp preset clear) were closed in commit `8d064235`. The remaining 11 failures (#1 sweep hygiene markers, #3 cloud SKU widget drift, #4/#12 deleted local audition wavs on disk, #5-#8 cloud adapter mock `corrupt_output`, #10 video lane canvas declaration truth, #13 LLM slot sweep, #14a/b GPU-held run-to-run variance) are environmental/fixture drift that cannot reach the shipped still lane and are properly classified and parked in `docs/2026-09-19-ship-regression.md`. |
| **3 -- `pyproject.toml` vs tree** | `version = "2.1.6"`, static dependencies match `requirements.txt`, node list matches the 25 registered nodes in `node_list.json`. |
| **3 -- README / apple/ docs integrity** | All 20 markdown links across `apple/*.md` and `README.md` resolve to valid local targets (0 broken links). All referenced files under `workflows/`, `config/`, `nodes/`, and `apple/` exist on disk. Shipped counts (six story banks, 21 variant graphs / 22 total graphs, 8 admitted languages, 43 Shakespeare translation scenes in six languages) match the repository tree across `README.md`, `apple/RUN.md`, `apple/MUSIC.md`, `apple/VOICES.md`, `apple/STYLES.md`, and `apple/UPSCALERS.md`. |
| **4 -- `source_bank` tooltip** | Accurately describes bank routing, the 6 runnable banks, roll behavior, and the non-runnable `+ Add Your Own` signpost. |
| **4 -- `source_ref` tooltip** | Accurately describes optional scene pin and fail-loud contract. |
| **4 -- `episode_language` tooltip** | Accurately describes the 8 admitted Kokoro languages, translation rules, and citations for 43 Shakespeare scenes in six languages. |
| **4 -- `visual_style` tooltip** | Accurately describes the prompt-tail-only effect and procedural video behavior. |
| **4 -- `lemmy_cameo` tooltip** | Tooltip was corrected in commit `7f284264` to state that `'never' leaves every slot to the cast`. |
| **5 -- What ships (`.comfyignore`)** | Preserves all required assets (`config/source_banks/**/translations/**`, `node_list.json`, `workflows/otr_canonical.json`, `workflows/variants/*.json`, and worker scripts). Excludes test trees, internal configs, dev tools, and dot directories (`.github/`, `.claude/`, and newly added `.cursor/` in `7f284264`). |

---

## The one thing

With #11 (Lemmy Bark preset survival) and `.comfyignore` (`.cursor/` registry zip leak) already resolved on `main`, the single remaining thing to fix before the Reddit post is the `custom_premise` widget tooltip at `nodes/OTR_LedgerScriptWriter.py:2329`. Shipped workflows intentionally arrive with all four My Story fields blank, relying on `_otr_story_input.DEFAULT_IDEA` for unattended rolls. Telling users that "At least one creative field must contain text" directly contradicts the shipped canonical workflow and will confuse anyone inspecting the writer node on install day.

COUNTS: must=0 should=1 docs=1
