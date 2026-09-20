# One prompt, six lanes -- the close-out audit before v2.1.x (2026-09-20)

Paste the block below whole into every lane (Astra, Terra, Luna, Sol, Grok/Composer,
Gemini). Each lane names itself in its output filename; outputs land in this folder
and the coding window merges them. No lane needs a second message.

```
CLOSE-OUT AUDIT -- ComfyUI-OldTimeRadio, before the v2.1.x ship (2026-09-20)

You are one of six reviewers given this same brief. Your job is to find what would embarrass this repository on the day strangers install it, and to say exactly how to fix each thing. Name yourself in your output filename. Default to "nothing found" over a manufactured finding: a finding you cannot ground in a file:line is not a finding.

REPO (real Windows files): C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio, branch main, HEAD a7697898. Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe with PYTHONUTF8=1 OTR_TEST_MODE=1 CUDA_VISIBLE_DEVICES="" ; pytest as `-m pytest -q -p no:cacheprovider`.
HARD LIMITS: read-only. No git writes (no add/commit/push/stash/checkout/reset/rebase). Do not edit, create or delete any repo file except your ONE output file. Scratch in %TEMP% only. No GPU; do not start, stop or signal any python process (a headless render may be running on this box). Do not run scripts/otr_vendor_scan.py with --write or any script that renders.

READ FIRST: docs/2026-09-19-ship-regression.md (the fourteen known test failures, classified), then `git log --oneline cc62b2c1..HEAD` (the day's commits) and `git diff --stat cc62b2c1..HEAD` (58 files).

THE CHECKLIST -- answer every item with file:line and the exact one-line fix, or "nothing found":
1. LOOSE ENDS IN THE DAY'S DIFF. For every .py file changed since cc62b2c1: a function added with no caller outside tests and its own module (grep nodes/ and scripts/); a comment or docstring that contradicts the code beside it; a TODO/FIXME/XXX added today; a debug print; a hard-coded absolute path; a non-ASCII character inside a log or print string (a cp1252 console dies on it).
2. THE FOURTEEN. Pick any of the fourteen failures in the ship note that you can fix in under ten lines with confidence and give the exact patch -- or, if it is legitimately environmental, the exact EXPECTED_FAILED_NODEIDS entry plus the docs/known-failures.md line. Say which of the two flagged ones (#2, the replay path's voice-node signature; #11, the Lemmy cameo route losing its bark preset) you looked at and what you found.
3. WHAT A STRANGER HITS. Read README.md and pyproject.toml as a first-time installer: every number, list, path, model name, language, node count and graph name must match the tree (node_list.json; workflows/; config/episode_languages.json; config/source_banks/shakespeare/translations/manifest.json -- 43 scenes in six languages, Japanese and Chinese now performed from the translator's own words since 96346118). List every stale statement with the true value beside it.
4. THE SHIPPING WIDGETS' WORDS. In nodes/OTR_LedgerScriptWriter.py INPUT_TYPES, read the tooltips of source_bank, source_ref, episode_language, visual_style and lemmy_cameo: does each say what the code does today? (episode_language was corrected in d5aa8832; check the other four the same way.)
5. WHAT SHIPS. .comfyignore decides what goes into the registry zip. Is anything in the tree that must NOT ship (scratch, kibitz-runs/, tmp/, secrets, dated docs folders) not excluded? Is anything that MUST ship (config/source_banks/**/translations/**, node_list.json, workflows/otr_canonical.json and its variants) accidentally excluded? Prove it from the file, not from memory.
6. ONE THING. The single finding you would fix before posting this on Reddit if you could fix only one, and why, in three sentences.

OUTPUT: write ONE file, UTF-8, no BOM: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-20-closeout\CLOSEOUT_<your-model-name>.md
Structure, in this order: `## MUST-FIX before ship`, `## SHOULD-FIX`, `## DOCS-ONLY`, `## NOTHING FOUND` -- one row per finding: `file:line | what is wrong | the exact fix`. End with one line: `COUNTS: must=<n> should=<n> docs=<n>`.
```
