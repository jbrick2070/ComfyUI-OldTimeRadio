# OTR v2.0 Bug Log

Active bug log for the v2.0 build. Every bug gets logged the moment it is found.
Entries are never deleted.

### BUG-LOCAL-017 [FIXED]: MusicGen + AudioGen cache miss every run — `_cache_key` returned a fresh timestamped path
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Two files (`nodes/musicgen_theme.py`, `nodes/batch_audiogen_generator.py`) had identical structural bugs in their cache logic. `_cache_key()` returned a filename with the *current* millisecond timestamp baked in (`<role>_<sha8>_<ts_ms>.wav`). The call site immediately checked `os.path.exists(cache_path)` against that exact path — which never existed because the timestamp was "now". Result: cache miss every single run, ~22s of wasted MusicGen rendering per episode + N seconds of wasted AudioGen rendering per SFX cue, AND a Rule C7 violation because each run wrote a different timestamped filename, and FFmpeg embeds input WAV filenames in MP4 metadata streams → final mp4 bytes drifted between identical-input runs.
- **Cause:** Single function (`_cache_key`) tried to do two incompatible jobs: produce a deterministic identity for cache lookup AND a unique filename for write. The docstring explicitly described the timestamp as "guaranteed unique across episodes" but that defeated the entire cache.
- **Fix:** Split lookup identity from write filename in both files:
  - **`_cache_prefix(...)`** — deterministic identity prefix (`<role>_<sha8>` for MusicGen, `sfx_<safe_name>_<sha8>` for AudioGen). No timestamp.
  - **`_cache_filename_for_write(...)`** — canonical write filename (`<prefix>.wav`). No timestamp. Same inputs always land at the same filename → byte-identical mp4 metadata across runs (Rule C7 holds even on clean-cache runs).
  - **`_cache_key(...)`** — back-compat alias, returns the canonical write filename.
  - **`_find_cached(cache_dir, prefix)`** — two-level lookup: canonical `<prefix>.wav` first, fallback to legacy `<prefix>_<ts>.wav` files for back-compat with existing on-disk caches. Uses `iterdir() + startswith()` (per Phase D Gemini consult — `Path.glob()` chokes on `[` in filenames). Sorts legacy matches by parsed filename timestamp (not mtime; mtime is unstable across copy/restore).
  - **`_save_wav` made atomic** — writes through sibling `.tmp` then `os.replace()` (Phase D Gemini consult: prevents corrupted cache hits if process is killed mid-write). Explicit `format="WAV"` because soundfile can't infer format from `.tmp` extension. Cleanup of orphan `.tmp` on failure.
- **Verify:**
  - AST + 7 invariant guards per file (function presence, atomic write, iterdir-not-glob, etc.) — green.
  - **New mutation suite `tests/test_cache_key_mutations.py` — 30 passed in 2.87s.** 5 MusicGen mutations + 5 AudioGen mutations + 12 lookup tests (canonical-wins, legacy-fallback, newest-timestamp-wins, no-cross-prefix-match, glob-metachar-tolerance) + 2 atomic-write tests + 2 cache_key back-compat tests + 4 atomic-failure tests. Confirms: every identity dimension produces fresh sha; the cosmetic AudioGen `safe_name` is NOT used as identity (full-prompt change beyond first 20 chars still produces fresh sha); `Path.glob` would have failed on `[`-containing prefix but `iterdir+startswith` works.
  - Three CLAUDE.md regression suites: **221 passed / 1 skipped / 2 xfailed in 110.36s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30).
  - Existing on-disk timestamped cache files transparently start hitting after deploy (legacy fallback path).
  - **Real-run acceptance (pending):** two consecutive identical-input runs should produce one file per `(role, sha)` pair; second run should log `CACHE HIT` with the canonical `.wav` name. End-of-stack soak covers this together with Phases A, B, C.
- **Tags:** cache, c7-byte-identity, ffmpeg-metadata, atomic-write, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-d-cache-key-consult__01_chatgpt.md` (gpt-5.5, 108.1s — proposed strong-form deterministic write), `docs/2026-05-02-phase-d-cache-key-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 32.0s — caught atomic-write requirement and `Path.glob` bracket bug), `docs/2026-05-02-phase-d-cache-key-consult__03_nvidia.md` (llama-3.3-nemotron-49b, 127.0s — confirmed all decisions). All three converged unanimously: drop timestamp on writes, two-level lookup, iterdir+startswith, atomic write, defer model_name digest expansion.
- **Deferred to v2 cache-key migration (separate scope):** add `model_name`, `sample_rate`, `decode_mode`, `guidance_scale` to the digest payload. Today these are effectively constants per-file but if the user starts varying them at runtime, cache identity will be wrong until v2 lands.

---

### BUG-LOCAL-016 [FIXED]: Filename pattern audit — slug-reconstruction regression guard
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — this is a regression guard. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase C) prescribed an audit of all `nodes/` files for the dangerous anti-pattern: code constructing `f"{ep_id}_..."` to *find* or *delete* a file on disk. The actual on-disk filenames for cache files (musicgen, audiogen wavs) follow the format `<role>_<sha>_<ts>.wav` — produced by the writer, indexed by sha. Slug-reconstruction-for-discovery breaks every time the producer's naming convention diverges from the slug.
- **Cause:** Phase A and Phase B already absorbed the live instances of this anti-pattern (rtx_upscale spacesaver and production_ledger sidecar rename now use `audio_dir.glob(...)`). What remained was the risk of *future* drift — someone reintroducing slug reconstruction in a discovery path without realizing the cache filenames don't match the slug.
- **Fix:** Audit complete (0 substantive code changes). The remaining `f"{ep_id}.mp4"` and similar usages in the codebase are all canonical writer/reader pairs sharing a contract by construction (RTXUpscale OBS-existence guard ↔ VideoComposite mp4 writer; Ledger class authoring `<ep>_ledger.json`). New regression test `tests/test_filename_pattern_audit.py` codifies the rule:
  - **`test_no_audio_cache_slug_reconstruction`** — static-analyzes all `nodes/*.py` for banned patterns: `audio_dir / f"opening_{ep_id}.wav"`, `audio_dir / f"sfx_{ep_id}_..."`, etc. Will fail loudly on any future drift.
  - **`test_destructive_paths_use_glob_not_reconstruction`** — positive assertion that the rtx_upscale spacesaver (Phase A) and ledger sidecar rename (Phase B) still use glob discovery; if a refactor accidentally replaces `audio_dir.glob("*_treatment.txt")` with slug reconstruction, this test catches it.
  - **`test_allowlist_entries_still_present`** — every entry in the test's ALLOWLIST (legit canonical writer/reader pairs) must still resolve to a real source line. Stale entries surface for pruning instead of silently shielding future drift.
- **Verify:**
  - 3/3 audit tests pass in 1.60s.
  - Combined regression: **191 passed / 1 skipped / 2 xfailed in 98.00s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3).
- **Tags:** audit, regression-guard, slug-reconstruction, qa-pass-2026-05-02, no-active-bug
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase C section, table of canonical writers/lookups). No round-robin needed — mechanical audit, no determinism implications.

---

### BUG-LOCAL-015 [FIXED]: production_ledger treatment rename gap + os.replace silent split state
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode soak run with Phase A)
- **Symptom:** Two adjacent bugs in `Ledger.rename_episode` (`nodes/production_ledger.py`):
  1. **Treatment file rename gap (Finding 2 from QA pass).** The function moved the per-episode dir and renamed `pending_<ts>_ledger.json` → `<new_id>_ledger.json` but did NOT rename `pending_<ts>_treatment.txt` → `<new_id>_treatment.txt`. The treatment file (written early by `OTR_LLMScriptWriter` before the title is finalized) sat in the new dir with the old prefix. Phase A's spacesaver kept it via a defensive `glob("*_treatment.txt")`, but that defensive measure was a workaround.
  2. **`os.replace` fallback silent split state (Finding 3 from QA pass).** When the dir-move `os.replace(old_ep_dir, new_ep_dir)` failed (Windows Defender lock, indexer holding a handle, partial dir from a prior crash, **or destination dir already existing — `os.replace` ALWAYS fails on Windows when destination dir exists, even empty**), the code logged a warning and continued. It updated `self.episode_id` and `self.data["episode_id"]` to the new id but left `self.out_dir` pointing at the old path. The next `self.save()` wrote a finalized-id ledger into the OLD dir while every downstream node (BatchHumoRender, VideoComposite, RTXUpscale) built paths from the new id. Net effect: confusing "file not found" cascades far away from the rename failure.
- **Cause:** Single function trying to advance in-memory state regardless of on-disk success. Missing state-matrix handling for: both dirs exist; both missing; old missing + new exists. No retry on transient Windows locks. Treatment files outside the rename loop. Filename-construction slug not consistent between ledger and treatment paths.
- **Fix:** Rewrite `Ledger.rename_episode` around a strict invariant: **either complete with canonical episode dir + canonical ledger + canonical treatment, OR raise BEFORE mutating in-memory episode state.** Specifics:
  - Case-insensitive `os.path.normcase` same-path early-return (no-op for case-only changes on Windows).
  - State matrix: `(old_exists, new_exists)` resolved into one of {happy retry path, conflict raise, idempotent recovery, both-missing raise} BEFORE any mutation.
  - 3 × 0.5s inline retry on `os.replace(old_ep_dir, new_ep_dir)` with attempt-aware logging. After the third failure: `RuntimeError` with message that explicitly tells the user to check for files open in Notepad / VLC / Explorer preview / editors (per Gemini consult — system locks clear in ms but human-held locks need user intervention).
  - In-memory state (`episode_id`, `data["episode_id"]`, `out_dir`) only advances AFTER dir is in final on-disk position.
  - Ledger file rename (best-effort warn-only, dir invariant already satisfied).
  - Treatment + sidecar rename: glob `<old_slug>_*.txt` (NOT `pending_*` — narrower, no risk of catching unrelated files), rename each to `<new_slug>_*.txt`. Uses the same `_slugify(..., limit=120)` as the ledger path. Per-file warn-only on failure.
- **Verify:**
  - AST + 9 invariant guards (hard-fail message, retry sleep, conflict check, both-missing check, sidecar glob, slug consistency, normcase, etc.) — green
  - **New targeted suite `tests/test_ledger_rename.py` — 10 passed in 1.78s.** Covers: happy path renames dir+ledger+all sidecars; same-id no-op; conflict raises; both-missing raises; idempotent recovery (old missing + new exists); dir-move retries 2/3 then succeeds; dir-move fails 3/3 → RuntimeError + state unchanged; error message mentions human-held locks; treatment failure does not raise; sidecar glob uses old-id prefix not pending wildcard.
  - Three CLAUDE.md regression suites: Bug Bible (23 passed / 1 skipped / 2 xfailed), `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed). Total **178 passed / 1 skipped / 2 xfailed in 101.62s**.
  - **Real-run acceptance (pending):** kill mid-rename (Ctrl-C between treatment write and rename), restart, confirm clean recovery and no orphan `<old>_*.txt` after the next successful run. Two-episode-in-flight soak covers Phase A + B together.
- **Tags:** ledger, rename, atomicity, windows-replace, retry, slug-consistency, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-b-rename-consult__01_chatgpt.md` (gpt-5.5, 150.8s), `docs/2026-05-02-phase-b-rename-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 31.9s — caught the critical "Windows os.replace always fails on existing dest dir, even empty"), `docs/2026-05-02-phase-b-rename-consult__03_nvidia.md` (llama-3.3-nemotron-super-49b-v1.5, 66.7s)

---

### BUG-LOCAL-014 [FIXED]: Spacesaver wrong-episode wipe via global mtime ledger scan
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode run)
- **Symptom:** `_spacesaver_cleanup_if_flagged` in `nodes/rtx_upscale.py` discovered the ledger to read the `meta.perfect_run_spacesaver` flag from by calling `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])`. That walker returns the newest `*_ledger.json` by mtime across the **entire** `otr/episodes/` tree. If Episode A is mid-RTXUpscale when Episode B is queued and writes its pending ledger, A's spacesaver pass would discover B's ledger, derive `ep_dir = ledger.parent.parent` (B's tree), and wipe B's `stills/`, `portraits/`, `videos/`, `composited/` while B was still rendering.
- **Cause:** Use of a global mtime-based discovery in a destructive code path. The existing substring sanity guard (`"episodes" in parts and "otr" in parts`) only verified the wiped tree was *somewhere* under `otr/episodes/`, not that it was the **right** episode for the current RTXUpscale call.
- **Fix:** Derive `ep_dir` directly from the `src` argument the upstream node already passes in. `src` is always `otr/episodes/<ep>/composited/<ep>.mp4` (the VideoComposite output), so `src.resolve().parent.parent` is the episode root by construction. Replace substring guard with `ep_dir.relative_to(otr_episodes_root().resolve())` plus a `len(rel.parts) == 1` depth-1 invariant. Load the ledger from THIS episode's `audio/*_ledger.json` glob, prefer non-pending. Add an OBS-existence precondition (`otr/obs/<ep>.mp4` must exist on disk) so spacesaver refuses to fire if the run order ever flips and the final deliverable hasn't landed yet. Build the keep-list from real on-disk filenames (`audio_dir.glob("*_treatment.txt")` plus the discovered ledger path) so a slug mismatch between `ep_id` and the on-disk filename can't accidentally delete the ledger or treatment.
- **Verify:**
  - AST + Bug Bible regression (23 passed / 1 skipped / 2 xfailed) + `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed in 107.84s) all green post-fix.
  - Source no longer references `find_most_recent_ledger` from the spacesaver path (verified by grep + AST sanity script).
  - **Real-run acceptance (pending):** queue Episode A, queue Episode B before A's RTXUpscale fires; inspect `[OTR_RTXUpscale] spacesaver:` log lines and confirm `ep_dir` resolves to A's path, never B's. Bypass-safety run with `src` outside `otr/episodes/` should log `refusing destructive cleanup` with no deletion. Delete `otr/obs/<ep>.mp4` before cleanup fires and confirm the new precondition aborts.
- **Tags:** spacesaver, ledger, two-episode, destructive-cleanup, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-path-reorg-spacesaver-qa__01_chatgpt.md`, `docs/2026-05-02-path-reorg-spacesaver-qa__02_gemini.md`, `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase A)
- **Follow-up phases queued:** B (production_ledger.py treatment rename + os.replace fallback), C (slug-reconstruction sweep), D (cache-key timestamp drop), E (schema bump + meta.paths block)

---

### BUG-LOCAL-001: Pre-existing test infrastructure rot blocks `pytest tests/` regression baseline
- **Date:** 2026-05-02 | **Phase:** 0 (regression infra) | **Bible candidate:** yes
- **Symptom:** Running the canonical `python -m pytest tests/` cannot reach a clean green pass. Three distinct failure modes observed in one run:
  1. **8 collection ImportErrors** (`No module named 'otr_v2.visual'`) on:
     `tests/test_anchor_gen.py`, `tests/test_camera_path_determinism.py`,
     `tests/test_character_regression.py`, `tests/test_cold_open_canary.py`,
     `tests/test_episode_dry_run.py`, `tests/test_lhm_monitor.py`,
     `tests/test_three_minute_continuous.py`, `tests/test_visual_phase_a.py`.
     Without `--ignore=` flags, pytest aborts the run after collection (`Interrupted: 8 errors during collection`).
  2. **`tests/test_backend_dispatch.py` 14 failures** (`FFFFFFFFFFFFFF` in -q output). Not investigated yet.
  3. **`tests/test_dropdown_guardrails.py` deterministic hang** after the first 12 tests pass (`............` then no further progress for >2 min, until externally killed).
- **Cause:**
  1. `otr_v2/visual/` package was deleted in commit `7706660` ("Fix BUG-LOCAL-047: FLUX anchor dtype ladder"). Test modules that imported it were not updated or removed in the same commit.
  2. test_backend_dispatch failure mode unknown — needs investigation.
  3. test_dropdown_guardrails.py hang likely waiting on a network/model/subprocess fixture that no longer resolves; not yet bisected.
- **Fix:** **Pending — do NOT fix mid-test per ground rules.** Captured here as v2.0-beta era opening entry. Likely fix sequence (next session): (a) delete or rewrite the 8 stale visual test modules; (b) bisect dropdown_guardrails hang to identify the wedged test; (c) investigate backend_dispatch failures separately. Also note: CLAUDE.md references `tests/v2/test_audio_byte_identical.py` which doesn't exist (path is `tests/test_audio_byte_identical.py`); CLAUDE.md test-command block is stale.
- **Verify:** After fix, `python -m pytest tests/` runs to completion, no collection errors, all hangs resolved, backend_dispatch failures triaged (fixed or marked xfail with reason).
- **Tags:** test-infra, pre-existing, otr_v2-orphans, claude-md-staleness

### BUG-LOCAL-002: `scripts/soak_operator.py` widget indices stale (drift since episode_title + num_characters added)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** `scripts/soak_operator.py` declares `WV_GENRE=1`, `WV_TARGET_WORDS=2`, `WV_CREATIVITY=11`, `WV_OPT_PROFILE=13`. Reading `nodes/story_orchestrator.py::OTR_LLMScriptWriter.INPUT_TYPES` shows the actual widget order is now: `[0]episode_title, [1]target_words, [2]num_characters, [3]model_id, [4]cleanup_model_id, [5]custom_premise, [6]include_act_breaks, [7]self_critique, [8]open_close, [9]target_length, [10]style, [11]style_custom, [12]creativity, [13]arc_enhancer, [14]optimization_profile`.
- **Cause:** `episode_title` and `num_characters` widgets were added to the script-writer node, plus `style_custom` and `arc_enhancer` were inserted. soak_operator constants were never updated. Anything calling `supersoaker.py::patch_workflow` writes to the wrong slots: `creativity` write lands on `style_custom` (string field, broken), `optimization_profile` write lands on `arc_enhancer` (boolean field, broken), `target_words` write lands on `num_characters`, and `WV_GENRE=1` writes target_words.
- **Fix:** **Pending — do NOT fix mid-test per ground rules.** Correct constants: `WV_TARGET_WORDS=1, WV_NUM_CHARACTERS=2, WV_SELF_CRITIQUE=7, WV_TARGET_LENGTH=9, WV_STYLE=10, WV_CREATIVITY=12, WV_ARC_ENHANCER=13, WV_OPT_PROFILE=14`. Drop `WV_GENRE` (the widget no longer exists; "genre" was effectively replaced by `style`). Update `supersoaker.py::patch_workflow` cfg keys accordingly. Add a smoke assertion that reads node 1's widget names from `/object_info` and aborts if order doesn't match constants.
- **Verify:** After fix, run supersoaker P0 → confirm log shows correct `target_words`, `target_length`, `creativity`, `optimization_profile` values; ledger reflects what was patched.
- **Tags:** widget-drift, soak-harness, supersoaker, bible-candidate

### BUG-LOCAL-003: ComfyUI Desktop launch does not inherit user-scope `HF_HOME`
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** First smoke queue (prompt_id `a455fc20-...`) failed in NewsCuration / NewsCurationDeep / NewsSummary phases with repeated `local_files_only=True failed for model (mistralai/Mistral-Nemo-Instruct-2407 does not appear to have files named ('model-00001-of-00005.safetensors', ...))` and `huggingface.co` connection-failed fallback. Each phase took the timeout (65s NewsCuration, 40s NewsCurationDeep) and never recovered. ComfyUI then created a fresh empty `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407\` skeleton (timestamp 2026-05-02 00:40), proving it was looking at the wrong cache root.
- **Cause:** `HKCU\Environment` has `HF_HOME=C:\ComfyUI-Models\huggingface` (canonical, populated with all 5 Mistral-Nemo shards under `hub\models--mistralai--Mistral-Nemo-Instruct-2407\snapshots\04d8a905...\`). When ComfyUI Desktop is launched via `Start-Process` from a parent process that does NOT have `HF_HOME` already in its env (e.g. the Cowork sandbox), the Electron renderer + bundled Python backend inherit only the parent's env, NOT user-scope env vars. So huggingface_hub falls back to its default `~/.cache/huggingface/hub` (which on this machine is junctioned/aliased to `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub` — empty).
- **Fix (verified 2026-05-02):** Before launching `ComfyUI.exe`, explicitly set `HF_HOME=C:\ComfyUI-Models\huggingface` in the parent shell. Re-queue confirmed: `LLM tokenizer loaded from cache (no HTTP checks)` log line appeared, NewsSummary generated 2572 chars, ScriptWriter ran end-to-end @ 18.1 tok/s. **Permanent fix needed:** ComfyUI Desktop launcher should read user-scope HF_HOME via `winreg.OpenKey(HKEY_CURRENT_USER, "Environment")` at startup and inject into the Python child process's env. Until then, document the launch-via-elevated-shell-only pattern in README.
- **Verify:** `[00:48:59] LLM tokenizer loaded from cache (no HTTP checks)` in runtime log + ScriptWriter generation completes without `local_files_only=True failed`.
- **Tags:** comfyui-desktop, hf_home, env-inheritance, launch-pattern, bible-candidate

### BUG-LOCAL-004: OOM in OTR_LLMScriptWriter on 30-word ultra-smoke after parse-retry loop (peak 29.5 GB on 16 GB device)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** Smoke prompt_id `e6b87239-...` ran with `target_length="30 words (smoke, 1 act)"`, target_words=30, num_characters=2. Pipeline order: NewsSummary OK → ScriptWriter generated 571 tokens (parsed 0 scenes/0 lines/Characters: none) → OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) all returned 0 chars and were DISCARDED → "OPENCLOSE: All outlines failed" → next `_generate_with_llm` call OOMed: `Allocation on device 0 would exceed allowed memory. Currently allocated: 26.53 GiB / Device limit: 15.92 GiB / Free (according to CUDA): 0 bytes`. Exception type `torch.OutOfMemoryError`, raised at `nodes/story_orchestrator.py:3137 model.generate()` from `nodes/story_orchestrator.py:5188 write_script._generate_with_llm`. Final VRAM_SNAPSHOT before OOM: `current_gb=8.275 peak_gb=29.498` — peak indicates cumulative KV cache + activations from successive calls were never released.
- **Cause:** Hypothesis (needs code dive): after each LLM `model.generate()`, the KV cache + intermediate activations are not torch.cuda.empty_cache()'d before the next call. With 4-bit NF4 weights ~7.5 GB resident, plus context_cap=16384 prompt tokens of KV cache (40 layers × 2 K/V × 16384 × hidden_dim × 2 bytes ≈ 6.4 GB) per call, four cumulative calls overflow. Compounded by the 30-word preset's parse-fail retry path (since 0 lines parsed, system retries) — each retry is another full forward pass without inter-call eviction.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) audit `_generate_with_llm` and `_critique_and_revise` for an explicit `torch.cuda.empty_cache()` + KV-cache delete after every generate call; (b) add a hard parse-retry cap (e.g. ≤2) to prevent runaway retry on the ultra-smoke preset; (c) log the prompt-token count alongside `llm_generate_entry` snapshot so future OOMs can be bisected. CLAUDE.md says "use `_flush_vram_keep_llm()` between LLM phases" — verify that is actually being called between OpenClose synth and write_script retry.
- **Verify:** Re-queue 30-word ultra-smoke; expect peak_gb < 14.5 GB across full LLM ladder; expect `MAX_RETRIES_EXCEEDED` (graceful) instead of OOM if parse keeps failing.
- **Tags:** vram, oom, llm-cache, retry-loop, ultra-smoke, bible-candidate

### BUG-LOCAL-005: 30-word ultra-smoke ScriptWriter output unparseable (0 scenes / 0 dialogue lines / 0 characters from 571 tokens)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** With patched inputs `target_length="30 words (smoke, 1 act)"`, `target_words=30`, `num_characters=2`, the LLM (Mistral-Nemo-Instruct-2407 4-bit NF4) generated 571 tokens at 18.1 tok/s but the post-generation parser counted: `0 scenes | 0 dialogue lines | Characters: none`. Confirmed 30-word ULTRA-SMOKE preset was applied (history.current_inputs shows the patched values). OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) also returned 0-char outputs — all 3 DISCARDED ("too short: 0 chars"), "OPENCLOSE: All outlines failed".
- **Cause:** Hypothesis: the new "30 words (smoke, 1 act)" preset's prompt does not enforce `CHARACTER:` / `SCENE:` markers (CLAUDE.md note: "BUG-007 root cause: Short (3 acts) prompt now explicitly enforces CHARACTER: dialogue format" — the same fix may not have been carried forward to the 30-word preset). The model generates prose that satisfies token count but lacks the structural markers the parser greps for. Also possible: the SPINE_MODE upper bound (commit `6454d91`, BUG-LOCAL-132) collides with `max_new_tokens=150` in OPENCLOSE such that nothing generated reaches the parser. Needs prompt-trace logging.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) instrument `write_script` to dump the actual prompt + raw model output at TRACE level when parse yields 0 lines, so the format gap is visible; (b) port the BUG-007 CHARACTER:/SCENE: enforcement clause from the "short (3 acts)" prompt into the 30-word ultra-smoke prompt; (c) add a unit test that asserts the 30-word preset's compiled prompt contains the literal substrings `CHARACTER:` and `SCENE:`.
- **Verify:** Re-run 30-word ultra-smoke, expect `1 scene | 3 dialogue lines | 2 characters` parse and a non-empty ledger.
- **Tags:** prompt-format, ultra-smoke, parse-fail, bug-007-regression, bible-candidate

### BUG-LOCAL-006: `pytest tests/` hangs at session-start when ComfyUI is running on the same GPU
- **Date:** 2026-05-02 | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Symptom:** Running `python -m pytest tests/test_core.py tests/test_arc_check.py ... -q` while ComfyUI Desktop is up on `:8000` produces only the standard pytest banner ("test session starts", "platform win32...", "plugins: anyio...") and then hangs with the python.exe at ~2.7 GB RSS, no further output, no test names, for 90+ seconds. Killing the python.exe is the only way to recover. Same hang shape was observed during the baseline `pytest tests/` (which paused at `tests/test_dropdown_guardrails.py ............` mid-suite). Both behave identically: stable RSS, no I/O progress.
- **Cause:** Hypothesis: pytest collection imports OTR's `__init__.py` which transitively imports torch + transformers + bitsandbytes. ComfyUI already owns the CUDA primary context. Either bitsandbytes' `cuda_setup` or transformers' device-probe is stalling on CUDA-context-create while ComfyUI holds the device. Not yet bisected — could also be a network call (HF model resolver) or filesystem walk over `C:\ComfyUI-Models\huggingface` (1.85 TB free, deeply nested cache).
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) add an autouse conftest fixture that sets `CUDA_VISIBLE_DEVICES=""` for unit tests so collection never tries to bind to GPU; (b) move the OTR node imports out of the package's `__init__.py` top level into lazy load (already partially done for some modules, audit completeness); (c) document in CLAUDE.md that local pytest runs require ComfyUI to be killed first. Until then, regression baseline is unverifiable when ComfyUI is up.
- **Verify:** With ComfyUI killed, `pytest tests/test_core.py -q` runs to completion in <30 s. With ComfyUI up + the conftest CUDA-mask fixture in place, same.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate

---

## 2026-05-02 fix landings

The five entries above (BUG-LOCAL-001 through 006, minus 002 which was logged separately) received fixes in this session's mega-commit. Status update per fix-lands-here pattern: `[FIXED]` with verify recipe.

- **BUG-LOCAL-003 [FIXED]** — `scripts/run_comfyui.cmd` reads `HF_HOME` + `HUGGINGFACE_HUB_CACHE` from `HKCU\Environment` via PowerShell, exports them, and launches `ComfyUI.exe`. README.md "Launching ComfyUI Desktop on Windows" section documents the pattern. Verify: kill ComfyUI, run `scripts\run_comfyui.cmd`, queue any episode that touches an HF model — expect `LLM tokenizer loaded from cache (no HTTP checks)` in `otr_runtime.log`.
- **BUG-LOCAL-004 [FIXED]** — `nodes/story_orchestrator.py` `write_script` short-episode branch (target_words ≤ 700) now (a) calls `_flush_vram_keep_llm()` before the main `_generate_with_llm` call so KV cache + activation peaks from prior LLM phases (NewsSummary, OpenClose 3-outline + evaluator, synthesizer) don't ride along into the main forward pass, and (b) wraps the call in a `MAX_PARSE_RETRIES = 2` loop with a cheap `[VOICE: ...]` / `CHARACTER:` marker count as the parseability check; on exhaustion logs `MAX_PARSE_RETRIES_EXCEEDED, accepting last output` and lets the parse-fail observability stamp it in the ledger instead of OOMing on a fourth forward pass. Verify: re-queue 30-word smoke; expect peak_gb < 14.5 GB across LLM ladder; if parse keeps failing, expect `MAX_PARSE_RETRIES_EXCEEDED` in `otr_runtime.log`, not `torch.OutOfMemoryError`.
- **BUG-LOCAL-005 [FIXED v2]** — `nodes/story_orchestrator.py` `write_script` now (a) detects `is_ultra_smoke` / `is_tiny_smoke` BEFORE `_open_close_expansion` is called and short-circuits the 3-outline evaluator entirely (round-robin verdict 2026-05-02: ChatGPT 5.5 + Gemini 3.1 + NVIDIA Nemotron 49B all flagged this as the actual root cause of the 29.5 GB-on-16-GB OOM since the evaluator holds 3 parallel KV caches at once); (b) clamps `max_new_tokens` to 256 for ultra-smoke and 384 for tiny smoke so a degenerate model output cannot run away to 571+ tokens; (c) swaps in the streamlined ULTRA_SMOKE prompt with explicit `[VOICE: ...]` enforcement; (d) replaces the original permissive `_bare_hits` regex with a negative-lookahead variant that excludes `TITLE:` / `SCENE:` / `GENRE:` / `ENV:` / `SFX:` / `MUSIC:` / `VOICE:` / `CAST:` / `AUTHOR:` so a "TITLE only" output cannot falsely PARSE_OK; (e) under ultra-smoke mode requires `=== SCENE N ===` AND `>=2 [VOICE: ...]` lines for PARSE_OK (strict-VOICE contract). The standard path keeps the looser scene-plus-any-marker check. `[VOICE: ...]` regex held strict per Gemini + NVIDIA: relaxing it would desync from the downstream parser and risk dropping audio lines, violating C7. Verify: queue 30-word smoke; expect peak_gb < 14.5 across LLM ladder, ledger has `1 scene | >=2 [VOICE: ...] dialogue lines | 2 named characters`.
- **BUG-LOCAL-006 [PARTIAL]** — `tests/conftest.py` created. Sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import (before any `tests/test_*.py` collection), registers `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The fix lets pytest progress further than baseline (24 dots in `test_dropdown_guardrails.py` vs 12 before) but the suite still hangs at `TestDropdownsHaveEffect::test_creativity_produces_different_temps`. The hang is INSIDE that test's call to `_run_preflight` -- not a CUDA-init issue, since CUDA is masked here. Likely root cause: a fixture or _generate_with_llm mock interaction that still touches a heavy import. Reproduce: `python -m pytest tests/test_dropdown_guardrails.py::TestDropdownsHaveEffect::test_creativity_produces_different_temps -q -s` (run alone). Next-session work: bisect the hang inside that test class -- this is a separate bug from the CUDA-context-create hang the conftest fixed, deserves its own follow-up entry.
- **BUG-LOCAL-001 [PARTIAL]** — 8 stale `otr_v2.visual` test collectors deleted (`test_anchor_gen.py`, `test_camera_path_determinism.py`, `test_character_regression.py`, `test_cold_open_canary.py`, `test_episode_dry_run.py`, `test_lhm_monitor.py`, `test_three_minute_continuous.py`, `test_visual_phase_a.py`) AND 10 sidecar-era tests in the same family (`test_backend_dispatch.py`, `test_wan21_loop.py`, `test_wall_clock_estimator.py`, `test_vhs_postproc.py`, `test_pulid_portrait.py`, `test_planner.py`, `test_ltx_motion.py`, `test_flux_keyframe.py`, `test_flux_anchor.py`, `test_florence2_sdxl_comp.py`). 38 test_*.py files remain (was 48 + 8 = 56; 18 deleted). This subsumes the original "14 `test_backend_dispatch` failures" entry — that file is gone. Verify: `python -m pytest tests/ --collect-only -q` reports zero `otr_v2.visual` collection errors.
- **BUG-LOCAL-002 [FIXED]** — `scripts/supersoaker.py` deleted. `scripts/soak_operator.py` slimmed from a 1500-line soak runner to a ~270-line legacy shim retaining only `scan_treatment` (used by `tests/test_treatment_scanner_unicode.py`). New canonical surface: `scripts/otr_api.py` exposes `load_workflow`, `fetch_schemas`, `patch_widget_by_name` (uses live `/object_info` schemas — robust against future widget reorders), `workflow_to_api_prompt` (port of soak_operator's BUG-LOCAL-027/029-fixed converter), `submit_prompt`, `poll_history`, `queue_snapshot`, `cancel_queue`. `scripts/queue_smoke.py` + `smoke_watcher.py` rebuilt on `otr_api`. `tests/test_widget_drift_guard.py` rerouted via private alias `mod._workflow_to_api_prompt = mod.workflow_to_api_prompt`. Verify: `python scripts/queue_smoke.py` produces a `/history` entry with `current_inputs.target_words=[30]`, `num_characters=[2]`, `target_length=["30 words (smoke, 1 act)"]`.

### Round-robin QA — 2026-05-02

`docs/2026-05-02-v2.0-beta-sprint-qa__01_chatgpt.md` (gpt-5.5), `__02_gemini.md` (gemini-3.1-pro-preview-customtools), `__03_nvidia.md` (mistral-nemotron-super-49b-v1.5), `__04_synthesis.md`, `__transcript.json`. All three external models converged on **BLOCK** for the initial Sprint 1 commit. Three must-fix items unanimously prescribed:

1. Move ultra-smoke / tiny-smoke detection BEFORE `_open_close_expansion` so the 3-outline evaluator's parallel KV caches don't run (Gemini calculated ~6 GB of cache from 3 simultaneous outlines = the actual source of the 29.5 GB peak; ChatGPT and NVIDIA both endorsed). **Applied.**
2. Replace the permissive `_bare_hits` regex with a structural-marker negative-lookahead so `TITLE:` / `GENRE:` etc. cannot falsely PARSE_OK; require `=== SCENE N ===` PLUS `>=2 [VOICE: ...]` lines for ultra-smoke. **Applied.**
3. Clamp `max_new_tokens` to 256 for ultra-smoke (384 for tiny smoke) so a runaway 571-token degenerate output cannot recur. **Applied.**

Disagreement caught: ChatGPT recommended relaxing the `[VOICE: ...]` regex to be more permissive. Gemini and NVIDIA both rejected this — relaxing the validation regex desyncs from the downstream parser and could silently drop dialogue lines, violating C7. Held the regex strict.

Non-blocking follow-ups for next session: investigate `live_ledger=True` for retained GPU tensors, audit `_generate_with_llm`'s explicit `del` of intermediates, write a unit test that mocks `_generate_with_llm` to return bad-then-good output and asserts exactly 2 attempts max, port any still-valid contracts (frame count `4n+1`, etc.) from the deleted sidecar tests into in-graph test coverage.

Post-fix regression: `python -m pytest tests/ --ignore=tests/test_dropdown_guardrails.py -q` → **932 passed, 6 skipped, 0 failed in 10.8 s**. AST-clean.

---

## 2026-05-02 — Sprint 3 mega-sprint (LTX wiring + RTX VSR upscale)

### BUG-LOCAL-007 [DEVIATION-LOGGED]: LTX 2B v0.9 bundled checkpoint forces CheckpointLoaderSimple-family loader

- **Date:** 2026-05-02 | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** ROADMAP Architecture Truth (locked 2026-05-02) specified `UNETLoader + CLIPLoader (T5) + VAELoader` for LTX 2B fp16, "NOT CheckpointLoaderSimple". Reason given: split-load lets ComfyUI offload T5/VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.
- **Cause:** Lightricks ships LTX 2B v0.9 ONLY as a bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB, all components in one safetensors file). No standalone LTX UNet / LTX VAE artifacts exist on the upstream HF repo for the 2B v0.9 line. The Architecture Truth assumed split files exist; they don't.
- **Fix:** Use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader` for LTX 2B v0.9 (it IS a `CheckpointLoaderSimple` subclass, but adds a `dependencies` input that ComfyUI uses to force sequential load). The C2 sequencing intent (HuMo unloads before LTX claims VRAM) is satisfied via the dependency edge + the existing strict teardown in `batch_humo_render.py` (`unload_all_models + gc + empty_cache + cuda.synchronize` in finally). The "no carve-out for CheckpointLoaderSimple" rule was about preventing OOM from parallel-load on a hot cache; sequencing eliminates that risk.
- **Verify:** Log line `[OTR_HUMO] teardown complete` precedes `[OTR_LTX_LOADER] loading ltx-video-2b-v0.9.safetensors`. No `Allocation on device 0 would exceed allowed memory` between HuMo teardown and LTX render.
- **Promote-to-Bible-only-if:** a future LTX line (3B, 5B, 13B) ships with split UNet/T5/VAE artifacts AND we re-validate that LowVRAMCheckpointLoader's dependency edge keeps the C2 sequencing guarantee under 14.5 GB. Until then this stays as a documented OTR-local deviation.
- **Tags:** ltx, loader, c2-deviation, sequencing, deps-edge

### BUG-LOCAL-008 [DOCUMENTED]: LTX CFG=1.0 mathematically erases the negative prompt

- **Date:** 2026-05-02 | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** Round-robin Gemini caught: standard CFG math is `output = uncond + CFG * (cond - uncond)`. At CFG=1.0 this simplifies to `output = cond` -- the negative prompt is 100% unused. ROADMAP locks `LTX_CFG = 1.0` for the distilled sigma schedule. So the negative prompt in `_LTX_NEGATIVE` ("person, human, face, woman, man, hands, fingers, body, ...") is mathematically discarded by the sampler. Faces / people may still appear in LTX clips because the prompt suppression we *thought* was active isn't.
- **Cause:** Distilled LTX (`LTX_DISTILLED_SIGMAS` from Goofer) is tuned for CFG=1.0 because higher CFG with low-step distillation produces overcooked / artifacted output. The negative prompt was carried over from non-distilled LTX patterns where CFG≥1.5 made the negative effective.
- **Fix (deferred):** Two options for next sprint, both empirical:
  1. Raise CFG to 1.3-2.0 and re-tune sigma schedule (changes motion characteristics; needs A/B smoke).
  2. Remove the negative prompt encode entirely (saves T5 forward pass; output unchanged at CFG=1.0).
  Until then: tighten POSITIVE prompts in `_PROMPT_BY_ROLE` to avoid human-implying terms ("announcer at microphone" → "vintage microphone on desk"), since the positive branch is the only one the sampler sees at CFG=1.0.
- **Verify:** ffprobe + visual review of LTX clips after smoke. If `_PROMPT_BY_ROLE` text contains "announcer / radio host / broadcaster" terms AND faces appear in the rendered output, this bug is the cause.
- **Tags:** ltx, cfg, prompt-policy, distilled-sigma

### BUG-LOCAL-009 [DEFERRED]: Per-stage VRAM logging across HuMo→LTX→VC→RTX boundary

- **Date:** 2026-05-02 | **Phase:** S3 observability | **Bible candidate:** no (observability, not behavior)
- **Symptom:** No production VRAM snapshot at HuMo teardown / LTX loader entry / LTX teardown / RTX upscale entry. If a 16 GB OOM appears at any boundary, we have no per-stage signal to bisect from.
- **Fix (deferred to next sprint):** Add `[OTR_VRAM] free=X.XX allocated=Y.YY reserved=Z.ZZ` lines at:
  - `batch_humo_render.py` end of teardown
  - `batch_ltx_render.py` after `mm.load_models_gpu([model])` and after teardown
  - `rtx_upscale.py` after each chunk and at upscale exit
  Pattern matches existing `vram_snapshot()` helper in `nodes/_vram_log.py`.
- **Verify:** smoke run produces a clean VRAM ladder log; `peak_gb` per stage extractable via grep.
- **Tags:** vram, observability, deferred

### Round-robin QA — 2026-05-02 mega-sprint pre-smoke

`docs/2026-05-02-mega-sprint-consult__01_chatgpt.md` (gpt-5.5, 140s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 40s). NVIDIA round did not complete in window; two-of-three is sufficient per CLAUDE.md round-robin rule.

**Must-fix items applied before smoke:**
1. **Anti-clobber in `batch_ltx_render.py`** (ChatGPT + Gemini converged): `if out_mp4.exists(): skip` defends against role-filter drift overwriting HuMo character clips.
2. **Windows ffmpeg pipe deadlock** (Gemini): `rtx_upscale.py` was using `stderr=subprocess.PIPE` which blocks at 64 KB on Windows. Routed to `subprocess.DEVNULL`.
3. **ComfyUI cache desync** (Gemini): rewired link 86 from `BatchHumoRender.clips_dir` (slot 0, stable per episode_id) to `BatchHumoRender.report` (slot 2, varies per run from per-clip elapsed_ms). Forces ComfyUI to re-evaluate `LowVRAMCheckpointLoader` on every queue, keeping the loader's state machine in sync with the actual mm-unload call HuMo makes.
4. **Drop `-shortest`** (Gemini): silent video and audio source share frame count by construction; flag was at best dead code, at worst a footgun.

**Disagreement caught:**
- ChatGPT framed CFG=1.0 negative prompt influence as "weak"; Gemini corrected with the math: `output = uncond + CFG * (cond - uncond)` reduces to `output = cond` at CFG=1.0, so the negative prompt is 100% ignored, not weak. Logged as BUG-LOCAL-008 (deferred to next sprint, since changing CFG mid-sprint deviates from locked architecture).

**Documented for next sprint (not blocking smoke):**
- BUG-LOCAL-008 — CFG=1.0 + negative prompt mathematically inert.
- BUG-LOCAL-009 — per-stage VRAM logging missing.
- Gemini false alarm: `temporal_size=4096` in tiled VAE decode is fine because LTX_MAX_FRAMES=177 (the temporal window only matters above its value; 4096 means "decode whole sequence in one temporal pass", spatial tiling handles VRAM). Goofer-proven on RTX 5080 Blackwell.

### BUG-LOCAL-010 [BLOCKER on Sprint 3 acceptance]: LLM OOM regression at write_script main call (BUG-LOCAL-004 returns)

- **Date:** 2026-05-02 | **Phase:** S3 acceptance smoke | **Bible candidate:** yes
- **Symptom:** Sprint 3 smoke prompt_id `bc7136bb-50ab-471f-8caf-83e9cfefa481` (`target_words=30`, `num_characters=2`, `target_length="30 words (smoke, 1 act)"`, `optimization_profile=Standard`) OOM'd at `OTR_LLMScriptWriter` -> `write_script` (line 5432) -> `_generate_with_llm` (line 3211) -> `model.generate(...)`. Exact error: `Allocation on device 0 would exceed allowed memory. Currently allocated: 24.54 GiB / Device limit: 15.92 GiB / Free: 0 bytes / Requested: 3.94 GiB`. Peak allocated `29005 MiB` per torch's allocator print.
- **Cause (hypothesis):** BUG-LOCAL-004 fix (Sprint 1) added `_flush_vram_keep_llm()` before the main write_script `_generate_with_llm` call. BUG-LOCAL-005 fix (Sprint 1) added max_new_tokens clamp 256 for ultra_smoke and short-circuited the OpenClose 3-outline evaluator. Both are confirmed active on this run (runtime log shows `Loading LLM model: mistralai/Mistral-Nemo-Instruct-2407 (quantized=True)` plus the three sequential `[StoryOrchestrator] Starting inference` lines at max_new_tokens 64 / 800 / 256). Despite both fixes, peak allocated still hits ~29 GB. Likely roots: (a) `_flush_vram_keep_llm()` is not actually clearing the prior phases' KV cache + activations -- a Python reference is keeping intermediates alive; (b) NewsSummary's `max_new_tokens=800` build-up is the actual culprit, not write_script's call (the OOM fires DURING write_script's prefill but the 24 GB was already accumulated before that call started); (c) the Mistral-Nemo `_prefill` path in transformers 4.x has a regression where `past_key_values` is not freed between `model.generate` invocations even with explicit `torch.cuda.empty_cache()`.
- **Fix:** **Pending -- needs its own bisect window.** Plan: (a) instrument `_generate_with_llm` to log `torch.cuda.memory_allocated()` at entry / after generate / after the explicit `del` / after empty_cache call, so the actual leak source surfaces; (b) check that `_flush_vram_keep_llm()` survived the recent refactors and is in fact called between NewsSummary and write_script; (c) audit whether NewsSummary leaves a `transformers.cache_utils.DynamicCache` instance on the model object; (d) if (c) confirmed, monkey-patch `model._cache_implementation` to `None` between phases.
- **Verify:** re-queue 30-word smoke with `optimization_profile=Standard`, expect `peak_gb < 14.5 GB` across the LLM ladder (instrumented snapshot lines), expect to reach `OTR_SignalLostVideo` (the audio gate) without OOM.
- **Tags:** vram, oom, llm-cache, regression, blocks-s3-acceptance

### Sprint 3 mega-sprint: shipped code, live acceptance BLOCKED

The Sprint 3 mega-sprint code (LTX wiring + RTX VSR upscale + consult fixes) is committed on `v2.0-alpha`. The wiring is:
- AST-clean (3 modified .py files all parse).
- Regression-clean (Bug Bible, dropdown_guardrails, core, parse_retry, otr_api_type all green; 23 + 46 + 108 + 48 = 225 tests pass).
- Workflow JSON valid (`json.loads` round-trips, `last_link_id=93`, all 51 links intact, no orphan inputs).
- ComfyUI registers all three new nodes (`OTR_BatchLTXRender`, `OTR_RTXUpscale`, `LowVRAMCheckpointLoader`).
- ComfyUI accepts the patched workflow at `/prompt` and runs to OTR_LLMScriptWriter, where it hits BUG-LOCAL-010 (LLM OOM, pre-existing).

**Sprint 3 acceptance is BLOCKED on BUG-LOCAL-010**, NOT on a Sprint 3 wiring failure. The video-wiring code never executed because the smoke can't get past the LLM phase. Once BUG-LOCAL-010 is fixed in a follow-up bisect, re-queue the same workflow JSON and the S3.x acceptance bullets (ledger source_kind=ltx rows, ffprobe 832x480 pre-upscale + 1920x1080 post-upscale, audio byte-identity via stream MD5, peak VRAM < 14.5/15.5 GB) become directly observable.

### BUG-LOCAL-011 [FIXED]: BatchLTXRender raised on first live run -- _load_ledger missing the .mp4 -> _ledger.json stem-fallback that sister nodes have

- **Date:** 2026-05-02 EVENING | **Phase:** S3 live test | **Bible candidate:** yes
- **Symptom:** Live run on Jeffrey's ComfyUI Desktop with Gemma-4 E2B (which dodges BUG-LOCAL-010) progressed cleanly through the LLM ladder, audio cascade, FLUX bookend, and all 4 HuMo character clips. At HuMo teardown the dependency edge correctly fired LowVRAMCheckpointLoader -> BatchLTXRender, but BatchLTXRender raised: `RuntimeError: BatchLTXRender: ledger could not be loaded from inline JSON or path` at `batch_ltx_render.py:446`. Wallclock to failure: 00:58:53 (LLM ~10 min, audio ~3 min, FLUX ~3 min, HuMo ~40 min, then LTX failed immediately).
- **Cause:** `OTR_SignalLostVideo.0` (the STRING input feeding `BatchLTXRender.ledger_json` via link 90) emits the **mp4 path**, not the `_ledger.json` path. `BatchHumoRender._load_ledger_with_path` and `OTR_VideoComposite._load_ledger_with_path` both have a multi-tier stem-fallback that swaps `.mp4` -> `_ledger.json` with collapsed-underscore + fuzzy-match tiers (BUG-LOCAL-118 hardening). My BatchLTXRender's `_load_ledger` skipped that fallback -- it called `load_ledger_safe(.mp4)` directly, got `None`, returned `(None, None)`, raised. Round-robin consult flagged "ledger / clips_dir union" but missed this inner discrepancy because the node *interface* matches HuMo (both take a STRING called `ledger_json`); only the *internal resolver* differs.
- **Fix:** Replaced `BatchLTXRender._load_ledger` with a port of `BatchHumoRender._load_ledger_with_path`. Same multi-tier behaviour: (1) empty input -> auto-pick newest non-pending under audio dirs; (2) inline JSON -> parse; (3) `.mp4` path -> direct stem swap, then collapsed-underscore variant, then fuzzy directory-scan with <1h freshness gate; (4) `.json` path -> direct load. Same `(dict_or_None, Path_or_None)` return contract so the existing call site at `:425` is unchanged.
- **Verify:** Re-queue the same workflow JSON. Expect log lines `[BatchLTXRender] episode=signal_lost_..._...` and `radio_bookend: radio_bookend_<ep>.png` (the loader resolved the .mp4 -> _ledger.json swap and read radio_bookend_path from `ledger.meta`). Pre-fix repro: queue a workflow that wires SignalLostVideo.0 directly into BatchLTXRender.ledger_json with no manual ledger path; expect the fix to make this path-shape work end-to-end.
- **Tags:** ltx, ledger, stem-fallback, signallost-mp4, sister-node-divergence, bible-candidate

### Sprint 3 live-run progress observed on workflow JSON 7c4dfd4 (Gemma-4 E2B path)

- LLM phase (Gemma-4 E2B + E4B): clean. ~10 min. Peak VRAM ~14 GB. Output: parseable script with TITLE + SCENE + 6 [VOICE: ...] lines + 1 SFX + MUSIC closing.
- Audio cascade (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler): clean. ~3 min. Episode duration 113s = 1.88 min.
- SignalLostVideo procgen: clean. mp4 saved 52.2 MB / 113s / 2712 frames in ~14s.
- BatchFluxRender (5 cast portraits + radio bookend): clean. **S3.2 acceptance VERIFIED**: radio bookend rendered at 1248x720 then Lanczos-downscaled to 832x480.
- BatchHumoRender Phase A (text encoding 4+1) + Phase B (Whisper) + Phase C (4 lines): clean. Peak VRAM 14.2 GB GPU dedicated. Per-clip ~10:00-10:20 wallclock at 6 sampler steps × ~97s/step. 4 character lines correctly routed to HuMo; 2 announcer lines correctly skipped (BUG-129b).
- LowVRAMCheckpointLoader -> BatchLTXRender: dependency edge fired correctly (sequencing intent SHIPPED), then BUG-LOCAL-011 raised inside `_load_ledger`. Fix landed in this commit; re-queue to verify the rest of the S3.x acceptance bullets.

### Round-robin consult on BUG-LOCAL-011 fix -- 2026-05-02 EVENING

`docs/2026-05-02-bug-local-011-fix-review__01_chatgpt.md` (gpt-5.5, 97s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 35s), `__03_nvidia.md` (nvidia/llama-3.3-nemotron-super-49b-v1.5, 93s).

**Three-way convergence (verdict: tighten before next live run):**

- Tier 1 (.mp4 -> .json exact stem swap) and Tier 2 (collapsed-underscore variant) are both correct and necessary.
- **Tier 3 fuzzy directory scan must be killed for LTX.** Non-deterministic (depends on directory contents + mtimes + wall-clock); could plausibly bind to a wrong neighbour ledger if exact match fails to load. Burning ~1 hour rendering against bad metadata is the failure mode 2/3 consultants flagged as the real risk.
- **Restore `_OTRL.load_ledger_safe()` for path loads** (consistent `[OTR_Ledger]` log prefix; future-proof against any hardening added there).
- **Fail loud on file-load errors.** If exact (Tier 1) or collapsed (Tier 2) ledger candidate file EXISTS but fails to parse / read (PermissionError from Windows file-locking, JSONDecodeError from a partial write), raise instead of falling through. Silent fall-through to a wrong neighbour was Gemini's strongest framing.
- **Document `humo_clips_dir` widget as a sequencing-only DAG edge.** Add tooltip + inline comment + `del humo_clips_dir` so a future maintainer doesn't remove it as dead code (which would let the LTX checkpoint load race HuMo's 16.5 GB MODEL teardown and OOM on 16 GB).

**Disagreement caught (consultants vs reality):**

- Gemini + NVIDIA flagged `log` and `time` as missing imports -> false alarms. `log = logging.getLogger("OTR.batch_ltx_render")` is at line 81; `import time` at line 51.
- Gemini + NVIDIA assumed `_OTRL.load_ledger_safe` does schema migration -> false alarm. It just wraps `json.loads` with three exception handlers (FileNotFound / JSONDecodeError / generic Exception), all logging WARNING and returning None. So restoring it gains consistent log-prefix and centralised error handling, NOT schema migration.

**Hardening pass applied in commit (next):**

1. Replaced raw `json.load()` calls with a local `_read(p)` helper that delegates to `_OTRL.load_ledger_safe(p)` when importable; raises RuntimeError when the loader returns None on an existing file.
2. Deleted Tier 3 fuzzy scan code path. Resolver now returns `(None, None)` after Tier 1 + Tier 2 miss, with a WARNING log line that explicitly notes Tier 3 was removed by the 2026-05-02 round-robin.
3. `humo_clips_dir` INPUT_TYPES tooltip rewritten to flag the widget as a DAG sequencing edge (NOT data); execute() body explicitly `del humo_clips_dir` with a comment explaining the "remove this and LTX OOMs" failure mode.

Resolver test (offline, against the live-run cached ledger) confirms all 4 branches still resolve correctly: .mp4 stem-swap, explicit .json path, inline JSON, empty input auto-pick.

**Companion artifact: `workflows/otr_ltx_smoke.json`** -- a 5-node fast-smoke harness (LowVRAMCheckpointLoader -> OTR_BatchLTXRender -> OTR_VideoComposite -> OTR_RTXUpscale + Note) that consumes the cached ledger.json + procgen mp4 + 4 HuMo character clips from the live run. ledger_json widgets on both BatchLTXRender + VideoComposite are the .mp4 PATH so the smoke truly repros the BUG-LOCAL-011 crash surface (i.e. exercises the .mp4 -> _ledger.json stem-swap chain). Wallclock target ~10 min vs ~60 min for full pipeline. Re-aim at a different episode by swapping the PROCGEN_MP4 + HUMO_VIDEOS_DIR widget values; both must come from the same episode_id so LTX writes into the dir HuMo wrote into.

### Path consolidation (Jeffrey directive, 2026-05-02 EVENING after first end-to-end smoke landed)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (project-specific layout)
- **Change:** Final episode mp4 deliverables moved from `<output>/episodes_for_obs/<ep>/` (sibling of otr/) to `<output>/otr/episodes/<ep>/` (nested INSIDE otr/). Every project output now lives under one tidy `otr/` umbrella:
  - `otr/audio/<ep>.mp4` -- procgen mp4 + ledger.json
  - `otr/stills/` -- FLUX bookends + cast environments
  - `otr/portraits/` -- PASS1 character portraits
  - `otr/videos/<ep>/<line_id>.mp4` -- per-line HuMo + LTX clip pieces
  - **NEW: `otr/episodes/<ep>/<ep>.mp4` and `<ep>_1080p.mp4` -- final user-facing deliverables ONLY**
- **Why:** OBS's directory_sorter still gets a clean root with only finished episodes (now `output/otr/episodes/`), but the entire project workspace has one nested root instead of two siblings (`otr/` + `episodes_for_obs/`). Easier mental model + easier to back up + easier to scrub.
- **Files touched:**
  - `nodes/_otr_paths.py::episodes_for_obs_dir` -- function name kept for back-compat with existing imports; return value changed to `comfy_output_dir() / "otr" / "episodes" / episode_id`.
  - `nodes/video_composite.py` -- comment block updated.
  - `scripts/render_episode_concat.py` -- comment + default `out_dir` expression updated.
  - `tests/test_render_episode_concat_discovery.py` -- pinned source-string assertion updated to require `"otr" / "episodes" / episode_id`.
- **OBS pointer change:** if you have OBS / external tooling configured to watch `output/episodes_for_obs/`, repoint it to `output/otr/episodes/`.

### Cleanup: torch.from_numpy non-writable warning in OTR_RTXUpscale (2026-05-02 EVENING)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (cosmetic)
- **Symptom:** First successful smoke surfaced `UserWarning: The given NumPy array is not writable, and PyTorch...` at `rtx_upscale.py:216`.
- **Cause:** `np.frombuffer(chunk_bytes, dtype=np.uint8)` returns a view over an immutable bytes buffer; `torch.from_numpy` warns when handed a non-writable ndarray.
- **Fix:** Added `.copy()` after `np.frombuffer()` so torch gets a writable buffer. Also confirms a clean ownership boundary between the ffmpeg-stdout-bytes and the cuda transfer.

### BUG-LOCAL-013 [FIXED]: LTX 2B v0.9 bundled checkpoint has no CLIP/T5 -- LowVRAMCheckpointLoader returns CLIP=None -> NoneType.tokenize crash

- **Date:** 2026-05-02 EVENING (T+~30 min after smoke load) | **Phase:** S3 fast-smoke first execution | **Bible candidate:** yes
- **Symptom:** Smoke loaded cleanly (after the BUG-LOCAL-012 UUID fix), Queue Prompt accepted, LowVRAMCheckpointLoader fired, then BatchLTXRender raised at line 554: `AttributeError: 'NoneType' object has no attribute 'tokenize'`. ComfyUI runtime log printed `no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.` immediately above the crash.
- **Cause:** The bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB on disk at `C:\ComfyUI-Models\checkpoints\`) ships only UNet + VAE; it does NOT carry the T5 text encoder. `LowVRAMCheckpointLoader` (a `CheckpointLoaderSimple` subclass) then returns `(MODEL, None, VAE)` for the (model, clip, vae) tuple. My BatchLTXRender wired CLIP straight from LowVRAM, so `clip.tokenize(...)` immediately NoneType-crashed. Note: this means BUG-LOCAL-007's deviation from the locked Architecture Truth was wrong on the premise -- the original Architecture Truth (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was actually correct, because `t5xxl_fp16.safetensors` has to be loaded separately. The LowVRAMCheckpointLoader is still useful for the UNet+VAE side (sequential-load via `dependencies` input survives), but the CLIP comes from a sibling CLIPLoader, not from the bundled file.
- **Fix:**
  1. Add a `CLIPLoader` node loading `t5xxl_fp16.safetensors` (already on disk at `C:\ComfyUI-Models\text_encoders\`) with `type='ltxv'`, `device='default'`. Verified `'ltxv'` is in `/object_info/CLIPLoader` allowed types on the live ComfyUI alongside sd3 / wan / mochi / flux2 / etc.
  2. Rewire `OTR_BatchLTXRender.clip` from the new CLIPLoader, NOT from `LowVRAMCheckpointLoader.CLIP`.
  3. Apply same fix to BOTH `workflows/otr_ltx_smoke.json` AND `workflows/otr_scifi_16gb_full.json` so the next live full-pipeline run also doesn't hit this crash. Production workflow now has new node 57 (CLIPLoader, T5, ltxv) wired via new link 94 to BatchLTXRender (55).1. Old link 88 (54.1 -> 55.1, the dead CLIP edge from LowVRAM) deleted.
- **Verify:** Re-run smoke; expect log line `[BatchLTXRender] episode=signal_lost_..._170555` followed by per-clip render lines (no `NoneType.tokenize` AttributeError). Final mp4 should write to `output/episodes_for_obs/<ep>/<ep>.mp4` and a separate `<ep>_1080p.mp4` from the upscaler.
- **Why the smoke harness paid off here:** the original failed live run (BUG-LOCAL-011 on the .mp4-as-ledger problem) hid this CLIP=None bug behind it. If I had only fixed BUG-LOCAL-011 and re-queued the full pipeline, we would have burned ~50 more min of HuMo wallclock just to crash here. The smoke surfaced this in <30s of LTX cold-load -- exactly what fast iteration loops are for.
- **Architecture Truth retroactively re-validated:** the locked Architecture Truth from 2026-05-02 (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was correct in spirit; only the bundled-vs-split question was open. We're now using a hybrid: LowVRAMCheckpointLoader for the bundled UNet+VAE (with the dependencies input for sequential-load), separate CLIPLoader for T5. Same end result as the locked plan; cleaner sequencing edge.
- **Tags:** ltx, cliploader, t5, bundled-checkpoint, hidden-by-prior-bug, smoke-paid-off, bible-candidate

### BUG-LOCAL-012 [FIXED]: ComfyUI frontend Zod validation rejected `workflows/otr_ltx_smoke.json` at load time

- **Date:** 2026-05-02 EVENING | **Phase:** S3 fast smoke harness | **Bible candidate:** yes (broadly applicable to anyone hand-building ComfyUI workflow JSONs)
- **Symptom:** Jeffrey loaded `workflows/otr_ltx_smoke.json` (commit `f60d2e4` / `4df4e72`) into ComfyUI Desktop and the frontend rejected the workflow with two Zod validation alerts: `Invalid workflow against zod schema: Validation error: Invalid uuid at "id"`. Raw `json.loads` round-trips cleanly; this is a frontend-side schema validation failure, not a JSON syntax error.
- **Root cause confirmed:** workflow root `id` field MUST be a valid UUID (8-4-4-4-12 hex format). My hand-built smoke had `id: "otr-ltx-smoke"` -- a freeform slug. ComfyUI's Vue 3 frontend Zod schema enforces uuid format on this field. Production `otr_scifi_16gb_full.json` has a valid UUID so it loads cleanly.
- **Cause (hypothesised pending error-text capture):** The smoke JSON was hand-built by `outputs/_build_ltx_smoke.py` rather than exported by the ComfyUI UI. Three candidate divergences from the canonical shape, identified by structural diff against `workflows/otr_scifi_16gb_full.json` + `Nvidia_RTX_Nodes_ComfyUI/example_workflows/rtx_video_upscale.json` (both load cleanly):
  1. **`_meta` field on every node.** Mine has `_meta: {title: ...}`; canonical workflows use a top-level `title` field on the node. Some Zod schemas reject unknown fields strictly.
  2. **`shape: 7` on optional inputs.** Mine puts `shape: 7` on `dependencies` + `humo_clips_dir`; neither known-good workflow uses this. LiteGraph shape values 1-7 are valid in the LiteGraph runtime, but the Vue frontend's Zod schema may not list `shape` as an allowed input field.
  3. **Missing `slot_index` on outputs.** Production has `{name, type, links, slot_index}` on every output; mine omits `slot_index`. Vue frontend may require it to track slot-position semantics.
- **Fix:** apply all 3 candidate corrections in `outputs/_build_ltx_smoke.py` and re-emit `workflows/otr_ltx_smoke.json`. Drop `_meta` -> rename to `title`. Drop `shape` from inputs entirely. Add `slot_index: <i>` to every output. Land an offline regression check (`tests/test_workflow_zod_shape.py`) that asserts these shape invariants on every JSON under `workflows/` so this class of bug is caught by the test suite before it reaches the UI.
- **Verify:** Jeffrey reloads `workflows/otr_ltx_smoke.json` in ComfyUI Desktop; no Zod error; nodes appear on canvas; Queue Prompt accepts the workflow.
- **Why prior consults missed it:** all three rounds reviewed the ComfyUI execution semantics (object_info, link types, DAG ordering, .mp4 path stem-swap, audio passthrough). None reviewed the frontend's Zod schema, which is a separate validation layer between drag-and-drop UI load and the backend's `/prompt` endpoint. The CLI `submit_prompt` path bypasses Zod entirely (it goes straight to `/prompt` with the API-converted workflow), which is why my offline JSON tests + the `queue_smoke.py` script + the `_test_ledger_resolver.py` all passed -- they exercise different code paths than the UI loader.
- **Tags:** zod, comfyui-frontend, workflow-json-shape, hand-built-workflow, bible-candidate



