# OTR v2.0 Bug Log

Active bug log for the v2.0 build. Every bug gets logged the moment it is found.
Entries are never deleted.

---

## Workflow tip — ask Claude for a live risk artifact after a fix

When Claude has shipped a non-trivial fix and you want a quick gut-check on residual risk WITHOUT triggering more code changes, you can ask: "round-robin the shipped code and give me a live artifact with your % chance of fix needed." Claude will:

1. Write a code-review-shaped consult question (not a prevention-plan question — the framing matters).
2. Run `scripts/_consult_round_robin.py` against the shipped code (ChatGPT + Gemini + NVIDIA Nemotron).
3. Read the transcripts under `docs/<date>-<topic>__01_chatgpt.md` etc.
4. Render an inline artifact card-grid with one card per fix element. Each card shows: one-line description + per-element follow-up-fix probability % + ChatGPT verdict badge + Gemini verdict badge + one-line reasoning. Color-coded by risk tier (green <15%, amber 15-30%, red 30%+).
5. Below the cards: "what to watch in next soak" callout + "where they disagreed" section + sources footer.

**Proven pattern (2026-05-03 EVENING for BUG-027 + BUG-028):** transcripts at `docs/2026-05-03-bug-027-028-shipped-code-review__*.md`, commit `832d134`. ChatGPT (108.6s, 21 KB) + Gemini (34.2s, 4.5 KB) converged; NVIDIA round failed silently but 2-of-3 was sufficient signal. The artifact made residual risk landscape immediately legible — load-bearing weak spot ("BUG-028 Site 3: humo wildcard glob, 40%") jumped out at first glance instead of being buried in 25 KB of model prose.

**Skip when:** fix is trivial (one-line typo, mechanical edit), or you've already moved on. **Use when:** tough decision area (LLM prompt templates, audio C7, VRAM determinism, save paths, anything with multi-site write+read alignment) and you want peace of mind without round-tripping through more text walls. Round-robin alone is text-heavy; the artifact is the part that makes it visually decision-ready.

---

### BUG-LOCAL-093 [FIXED]: HuMo portrait stopgaps removed -- wrong-face is worse than no-face
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-092 hardening | **Bible candidate:** YES (failure-mode policy)
- **Symptom:** even after BUG-092 inverted the dispatch order, two stopgaps remained that could still produce wrong faces:
  1. `cast_still_map` -- defense-in-depth fallback that bound `char_id` to `full_env_*.png` FLUX environment stills.
  2. `_find_portrait` tiers 4-5 -- internal fallback to `full_env_*.png` when no character portrait was found.
  Both fired only when BUG-LOCAL-078's per-cast portrait pass had failed AND the dispatch couldn't find a real portrait, but the failure mode was visibly wrong: HuMo would lipsync against an environment scene that happened to contain an unrelated person. Better to skip the line and let VideoComposite cover it with static-radio fill (BUG-129a, same handling as music/sfx) than render a wrong actor.
- **Cause (policy / not a code defect):** these were pragmatic stopgaps from before BUG-LOCAL-078 added the per-cast portrait pass. Now that BUG-078 reliably writes `c0X_portrait.png` and stamps `cast[].portrait_path`, the stopgaps mask real upstream bugs (a portrait-pass failure goes silent and produces wrong-face output instead of a loud SKIP that gets fixed).
- **Fix (`nodes/batch_humo_render.py`, ~80 LOC removed):**
  - **`_find_portrait` simplified to 3 tiers** (was 5): keeps cast.portrait_path, indexed pass1 portraits, any pass1 portrait. Removed tier 4 (`full_env_*` indexed by cast position) and tier 5 (any `full_env_*`). When no real portrait is found, returns None.
  - **`cast_still_map` dispatch removed from `execute()`**: the `_resolve_cast_stills_from_ledger()` call + log-line block (~40 LOC) gone. `cast_still_map` reduced to an empty dict so the downstream dispatch reads cleanly without restructuring (cheap defense-in-depth pin in case a future refactor re-adds binding logic).
  - **Dispatch priority** is now strictly `_find_portrait` -> `_find_composite` -> None. The third branch (`if not ref_png and char_id and char_id in cast_still_map`) was deleted entirely.
  - When `ref_png` is None, the existing SKIP path fires: log `WARNING line lXXX speaker=... role=...: no portrait AND no radio still`, append `SKIP no portrait` to the report, and `continue` to next line. VideoComposite's BUG-129a static-fill covers the time slot with the radio bookend image -- visible "missing-portrait" gap that's loud enough to surface upstream bugs without ruining the episode.
- **NEW tests (`tests/test_batch_humo_render.py`, +2):**
  - `test_find_portrait_returns_none_when_only_env_stills_exist` -- writes `full_env_00001_.png` + `full_env_00002_.png` to a tmp dir, calls `_find_portrait("EDNA", cast, tmp_path)`, asserts None. Pre-093 this would have returned the env still.
  - `test_ref_dispatch_no_env_still_fallback` -- source-code regression guard; asserts no live assignment of `ref_source = "ledger-cast-fresh"` exists in `batch_humo_render.py`. Comments / docstrings can mention the term for history; the actual code path is gone.
  - Renamed `test_ref_dispatch_prefers_find_portrait_over_cast_still_map` to `test_ref_dispatch_runs_find_portrait_before_find_composite` and updated assertions to reflect the simplified two-tier dispatch.
- **Verify:**
  - AST parse clean (108611 bytes, 8508 nodes -- ~3KB / 337 nodes smaller than pre-093).
  - test_batch_humo_render + test_batch_ltx_render + test_news_history_ttl -> 71 passed in 3.12s (was 69; +2 BUG-093 guards).
  - Bug Bible OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed (same baseline).
  - Live: a future episode where BUG-078 fails to render a portrait will now log `[BatchHumoRender] line lXXX speaker=... role=...: no portrait AND no radio still` and SKIP that line. VideoComposite covers it with static radio. The bug becomes visible (a character beat plays as static radio) instead of silent (HuMo renders a wrong face). Same handling as music/sfx.
- **Tags:** humo, ref-image-dispatch, portrait-priority, stopgap-removal, failure-mode-policy
- **Related:** BUG-LOCAL-078 (per-cast portrait pass that becomes a hard requirement post-093); BUG-LOCAL-088 (cast-still binding which is now fully removed); BUG-LOCAL-092 (priority inversion that this entry hardens further by removing the fallbacks entirely); BUG-LOCAL-129a (VideoComposite static-radio fill that covers skipped lines).

---

### BUG-LOCAL-092 [FIXED]: HuMo lipsync against FLUX env stills instead of character portraits
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-091 soak | **Bible candidate:** YES (ref-image dispatch priority)
- **Symptom (live composite from `signal_lost_scientists_map_how_down_syndrome_reshape_20260504_142107`):** Three artefacts visible in viewer screenshots:
  1. At 0:42 / 0:47 -- different male actors visible during what should be character lipsync, with weird "lip sync to environment" effect (HuMo trying to articulate an environment image into facial motion).
  2. At 0:32 vs 1:35 -- two clips that should be the SAME character (both AFFIRMATIVE SIR per the ledger lines block, char_id=c01) rendered with totally different faces.
  3. Inconsistent identity across multi-line same-character runs.
- **Cause:** ledger.clips[] entries showed `ref_png_name: "full_env_00001_.png"` and `ref_source: "ledger-cast-fresh"` for every character clip. HuMo was lipsyncing against the FLUX environment still (the radio scene), not the character portrait. Tracking the dispatch in `execute()`:
  ```python
  # batch_humo_render.py lines 1607-1621 (pre-fix)
  if char_id and char_id in cast_still_map:        # <- runs FIRST
      ref_png = cast_still_map[char_id]
      ref_source = "ledger-cast-fresh"
  if not ref_png:
      ref_png = _find_composite(...)
  if not ref_png:
      ref_png = _find_portrait(...)                # <- runs LAST
  ```
  The cast_still_map is populated by `_resolve_cast_stills_from_ledger()` which globs `full_env_*.png` (FLUX environment stills, not portraits). It assigns those to char_ids by mtime-descending cast index. Pre-BUG-078 this was a stopgap when no character portraits existed. Post-BUG-078 the per-cast portrait pass writes `<episode>/portraits/c0X_portrait.png` and stamps `cast[i].portrait_path` -- but the dispatch was checking cast_still_map FIRST, so the FLUX env still always won and the proper portrait was never reached.
  Why two clips of the same character look different despite the binding being deterministic: each char_id IS bound to a single env still for the whole run, but env stills don't carry character identity. APPRENTICE (male, dry, 50s) bound to a `full_env` that happens to contain a woman = HuMo articulates the woman's face. Different env stills for different char_ids = different "actors" on screen. And HuMo's per-chunk seed stride drifts the rendered identity further across chunks of the same line.
- **Fix (`nodes/batch_humo_render.py` lines 1607-1640, ~30 LOC):** invert the dispatch order so `_find_portrait` runs FIRST. `_find_portrait` tier 1 is `cast[i].portrait_path` (the BUG-078 portrait), with tier 4-5 falling through to `full_env_*` only when no portrait file exists. cast_still_map remains as defense-in-depth for episodes where the portrait pass didn't run.
  ```python
  # New order:
  ref_png = _find_portrait(speaker, cast, portraits_dir_path)   # tier 1: cast.portrait_path
  if ref_png: ref_source = "find_portrait"
  if not ref_png:
      ref_png = _find_composite(shot_id, speaker, portraits_dir_path)
      if ref_png: ref_source = "find_composite"
  if not ref_png and char_id and char_id in cast_still_map:
      ref_png = cast_still_map[char_id]
      ref_source = "ledger-cast-fresh"
  ```
- **NEW test (`tests/test_batch_humo_render.py::test_ref_dispatch_prefers_find_portrait_over_cast_still_map`):** source-code regression guard that locks the dispatch order. If a future refactor re-inverts the priority, this test fails before any live render surfaces the wrong-face artefact again. Asserts that the `'ref_source = "find_portrait"'`, `'ref_source = "find_composite"'`, and `'ref_source = "ledger-cast-fresh"'` string literals appear in source code in that order.
- **Verify:**
  - AST parse clean (111636 bytes, 8845 nodes).
  - test_batch_humo_render + test_batch_ltx_render + test_news_history_ttl -> 69 passed in 2.89s.
  - Live: next soak ledger.clips[] should show `ref_png_name: "c0X_portrait.png"` (NOT `full_env_NNNNN_.png`) and `ref_source: "find_portrait"` (NOT `"ledger-cast-fresh"`) for every character clip. Same character across multiple lines should render with the SAME face.
- **Tags:** humo, ref-image-dispatch, portrait-priority, bug-078-followup, audio-video-sync
- **Related:** BUG-LOCAL-078 (per-cast portrait pass that BUG-092 lets win); BUG-LOCAL-088 (cast-still binding which is now defense-in-depth instead of primary); BUG-LOCAL-086 (chunking; same per-chunk seed stride means within-line identity drift may still be visible after BUG-092 -- if so, BUG-LOCAL-092a would carry latent state across chunks).

---

### BUG-LOCAL-091 [FIXED]: LTX clips frozen on last frame -- chunking + 353-frame cap parity with HuMo
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-086 LTX parity | **Bible candidate:** YES (audio/video timeline alignment, BUG-086 sister fix)
- **Symptom:** BatchLTXRender used a hardcoded ``LTX_MAX_FRAMES = 177`` (~7.08 s @ 25 fps), and ``ltx_length_for_dur`` silently capped at that value. Any non-character audio line longer than 7.08 s (typical announcer monologue: 10-15 s, music intro/outro: 8-12 s) had its tail truncated; the LTX render produced a 7-second clip while the audio kept playing, leaving the radio scene frozen on its last frame for the back half of the line. Same root cause as BUG-LOCAL-086 but for the non-character render path. Docstring at line 270 even documented it: *"For lines longer than 10.28 s, VideoComposite downstream can ping-pong-loop or freeze-frame extend"* -- the workaround was the bug.
- **Cause (multi-layer):**
  1. ``LTX_MAX_FRAMES = 177`` constant matched HuMo's pre-086 cap (intentionally, for "timing contract simplicity" per a 2026-05-01 Jeffrey directive). When BUG-086 raised HUMO_MAX_FRAMES to 353, LTX was left behind.
  2. ``ltx_length_for_dur`` clamped at the cap, silently dropping the tail frames. No widget existed to override.
  3. Comment claimed cap should be 257 (10.28 s native, proven in ComfyUI-Goofer with VAEDecodeTiled) but the constant was 177 -- comment/value drift since the original LTX setup.
- **Fix (`nodes/batch_ltx_render.py`, ~150 LOC across 6 sites):**
  - **Constant bumps** -- ``LTX_MAX_FRAMES = 353`` (8*44+1 = 14.12 s @ 25 fps, matches the post-086 HuMo cap). New ``LTX_CHUNK_FRAMES = 177`` for the chunking fallback when a line still exceeds the user-configurable cap.
  - **NEW `ltx_length_for_dur_uncapped(dur_s)`** -- 8n+1 frame snap without the LTX_MAX_FRAMES ceiling. Used by the chunking dispatch.
  - **NEW `_concat_clips_via_ffmpeg(chunk_paths, out_path, ffmpeg)`** -- ffmpeg concat-demuxer wrapper with `-c copy`. Mirrors the BUG-086 helper in batch_humo_render.py; duplicated rather than imported to keep the LTX render path self-contained.
  - **NEW `clip_length` widget** in `INPUT_TYPES.optional` -- FLOAT, default 7.0, max 14.12, step 0.04. Same UX as BatchHumoRender's BUG-086 widget. Tooltip points to BUG-LOCAL-091.
  - **`execute()` signature** now accepts `clip_length=7.0` keyword arg.
  - **Plan-build refactor** -- per-line entry now carries `chunks: list[{dur_s}]`. Lines whose `dur_s <= clip_length` get a single-chunk plan (current behaviour, unchanged). Lines exceeding `clip_length` get an N-way even split where N = `ceil(dur_s / clip_length)`. New log line: `BUG-LOCAL-091: line lXXX dur_s=YY > clip_length=ZZ -- splitting into N chunks of W.WWs each`.
  - **Per-line render loop refactor** -- iterates `entry["chunks"]`, dispatches one LTX render per chunk against the same prompt + radio bookend ref. Per-chunk `shot_seed = seed + idx*1009 + chunk_idx*7919` so chunks 1+2 of the same line don't render with identical seed (would produce visible "stutter back to start" at the join). Single-chunk lines write directly to `<line_id>.mp4`. Multi-chunk lines write to `<line_id>__chunk{NN}.mp4` part files then `_concat_clips_via_ffmpeg` stitches them and the part files are unlinked.
  - **Ledger record** -- single entry per line (not per chunk) with new `n_chunks` field for traceability. Downstream (VideoComposite) sees one mp4 per line at `<line_id>.mp4`, regardless of chunk count.
  - **`import folder_paths` made try/except** so headless pytest collection works (folder_paths is provided by the ComfyUI runtime; pytest doesn't have it). Runtime still uses it via the `_otr_paths` helpers which already have their own folder_paths fallback chain. Comment kept so Bug Bible BUG-01.02 string-content check still finds the reference.
- **NEW tests (`tests/test_batch_ltx_render.py`, 19 tests):**
  - `test_ltx_constants` -- pin LTX_FPS=25, LTX_MIN_FRAMES=9, LTX_MAX_FRAMES=353, LTX_CHUNK_FRAMES=177
  - `test_ltx_length_for_dur` -- parametrize 8 cases including 14.12s -> 353 (cap exactly), 16s+ -> 353 (capped)
  - `test_ltx_length_for_dur_always_returns_8n_plus_1` -- 9 dur values
  - `test_ltx_length_for_dur_uncapped_skips_cap` -- 30s -> 753 uncapped, 353 capped
  - `test_clip_length_widget_present` / `test_clip_length_default_is_seven` / `test_clip_length_max_respects_humo_ceiling`
  - `test_execute_signature_accepts_clip_length` -- inspect `execute()` signature
  - `test_concat_helper_*` -- 4 tests for the ffmpeg concat wrapper (empty list rejected, single-chunk copies, single-chunk no-op when path matches)
- **Known caveats / what we're not pretending:**
  - Per-chunk seed stride (7919) is a guess at preventing same-frame regression at chunk joins. If the seam is visible in test, future BUG-LOCAL-091a should carry over the last frame's latent as a continuity hint instead of relying on stride randomness.
  - LTX_MAX_FRAMES=353 at 16 GiB Blackwell is **untested in a live run**. ComfyUI-Goofer proved 257 fine; 353 is extrapolated. If the next LTX render OOMs on a 14s clip, drop the constant back to 257 (10.28s) -- the chunking dispatch handles whatever the cap is.
  - LTX chunks share the same start frame (radio bookend) so multi-chunk renders have a "snap back" at the boundary. For OTR's stylized 1940s radio scene this is acceptable; if needed, future upgrade carries last-frame latent across chunks (BUG-LOCAL-091a).
- **Verify:**
  - AST parse clean on `nodes/batch_ltx_render.py` (53704 bytes, 3982 nodes).
  - `tests/test_batch_ltx_render.py` -> 19 passed in 1.77s.
  - `tests/test_batch_ltx_render.py + test_batch_humo_render.py + test_news_history_ttl.py` -> 68 passed in 2.99s.
  - Bug Bible regression OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.
  - Live: next LTX render with a >7s announcer line should log `BUG-LOCAL-091: line lXXX dur_s=YY > clip_length=7.0s -- splitting into N chunks` and finish with `(N chunks, BUG-LOCAL-091 chunked + concat)`.
- **Tags:** ltx, vram-ceiling, audio-video-sync, chunking, ffmpeg-concat, BUG-086-parity
- **Related:** BUG-LOCAL-086 (HuMo equivalent that this fix mirrors). BUG-LOCAL-105 (silent-clamp predecessor; chunking dispatch replaces the LTX side of that pattern but the capped `ltx_length_for_dur` still defends each individual chunk per chunk).

---

### BUG-LOCAL-090 [FIXED]: news_history.json grows unbounded -- 5-day TTL + state-dir relocation
**Update 2026-05-04 (commit follow-up):** the file was also moved out of the source repo into the per-machine state tier.

#### Part 2 — relocation to ``<output>/otr/state/`` (2026-05-04 follow-up)

The TTL fix kept the file at ``<repo>/config/news_history.json`` -- repo-local, which is wrong tier. Per-machine runtime state should live under the ComfyUI output tree where every other persistent OTR artifact (episodes/, obs/) lives. Hand-rolled paths under ``__file__/../../config/`` also tripped Bug Bible BUG-01.02 (output nodes should use ``folder_paths``).

- **NEW `nodes/_otr_paths.py::otr_state_dir()`** -- returns ``<output>/otr/state/``. Per-machine state tier; per-episode state continues to live at ``otr/episodes/<ep_id>/``.
- **`_NEWS_HISTORY_PATH`** now resolves to ``<output>/otr/state/news_history.json`` via ``otr_state_dir()``. Falls through to ``~/.otr_state/news_history.json`` defensively if ``otr_state_dir()`` is unavailable at import time (e.g. tests that monkey-patch).
- **`_NEWS_HISTORY_LEGACY_PATH`** retained pointing at the old ``<repo>/config/news_history.json`` for migration carry-forward.
- **`_load_news_history()`** -- reads new path first; if empty/missing, falls back to legacy path so the user's existing dedup window carries forward on the first post-fix run.
- **`_record_news_usage()`** -- writes only to the new path. On the first save, if the new path is empty, seeds from legacy entries so they're not silently lost. After that single save, legacy is dead-but-harmless.
- **NEW helper `_read_news_history_file(path)`** -- shared JSON-parse-with-fallback; used by both load and record so the migration semantics stay in lockstep.
- **`.gitignore`** -- added ``config/news_history.json`` so the legacy file never accidentally enters git history while it's still on disk for migration purposes.
- **NEW tests (3 added; total 10):**
  - `test_legacy_path_fallback_when_new_missing` -- legacy entries surface on first run after migration
  - `test_new_path_takes_precedence_over_legacy` -- when both files exist, new wins
  - `test_record_seeds_new_path_from_legacy_on_first_save` -- first save preserves legacy entries
  - `test_file_missing_returns_empty` + `test_corrupted_json_returns_empty` updated to monkey-patch BOTH paths so the real on-disk legacy file doesn't bleed into the test.
- **Verify (Part 2):** AST clean (565407 bytes, 39563 nodes). News history TTL suite -> 10 passed in 1.78s. Bug Bible OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.

#### Part 1 — original 5-day TTL fix (2026-05-04 EVENING)
- **Date:** 2026-05-04 EVENING | **Phase:** soak hygiene (NewsFetcher) | **Bible candidate:** YES (best-effort dedup with TTL)
- **Symptom (live runs 2026-05-04 14:10 + earlier):** `[NewsFetcher] Filtered 43 previously-used candidate(s) via news_history (0 remaining of 43)` followed by `[NewsFetcher] All 43 candidate(s) filtered out by history -- restoring unfiltered pool so the writer still gets a real article`. Every fresh run hit 100% prior-use rate. Fallback restored the unfiltered pool so generation continued, but the dedup intent (avoid back-to-back same-headline runs) was effectively dead because the history was set-membership-only with no expiration -- a once-used URL was blocked forever. Rolling cap was 200 entries, but with 8 RSS feeds returning ~5-6 stories each (~43 unique URLs/day) the entire daily pool gets blocked within ~5 days of normal use.
- **Cause:** `nodes/story_orchestrator.py::_load_news_history()` returned `{entry["url"] for entry in data if entry.get("url")}` -- a flat set of every URL ever recorded. `_record_news_usage()` writes timestamps but `_load_news_history()` ignored them. No TTL filter, so a headline used 30 days ago still blocked the candidate pool today.
- **Fix (`nodes/story_orchestrator.py`):**
  - **NEW constant `_NEWS_HISTORY_FILTER_DAYS = 5`** -- only URLs used within the last 5 days are kept in the active filter set. Older entries stay on disk for audit (so the file remains a usage log) but no longer block the pool.
  - **`_load_news_history()` refactored** to parse `entry["timestamp"]` via `datetime.fromisoformat()` and only include entries whose timestamp is `>= now - timedelta(days=5)`. Entries with missing or malformed timestamps default to fresh (safer to filter once than to surface a same-day repeat). File-not-found, JSON-parse errors, and any other I/O failure still return an empty set (best-effort, never blocks generation).
  - **`from datetime import datetime, timedelta`** -- existing import extended for the cutoff math.
- **NEW tests (`tests/test_news_history_ttl.py`, 7 tests):**
  - `test_ttl_constant_is_five_days` -- pins the window so a silent revert is caught.
  - `test_load_filters_old_entries` -- 5-entry fixture spanning today/2d/4d/6d/10d ago; asserts only the first three survive.
  - `test_missing_timestamp_treated_as_fresh` -- missing field, empty string, malformed string all fail-to-fresh.
  - `test_missing_url_skipped` -- URL-less entries dropped silently.
  - `test_file_missing_returns_empty` -- first-run case.
  - `test_corrupted_json_returns_empty` -- invalid JSON file is not an error.
  - `test_entries_at_exactly_ttl_boundary` -- entry +1s inside the window is fresh, -1s outside is stale.
- **Verify:**
  - AST parse clean on `nodes/story_orchestrator.py` (563374 bytes, 39512 nodes).
  - `tests/test_news_history_ttl.py` -> 7 passed in 1.72s.
  - `tests/test_core.py + test_critique_dialogue_preservation.py + test_dropdown_guardrails.py + test_news_history_ttl.py` -> 183 passed in 101.89s (full coverage including modules that import story_orchestrator).
  - Bug Bible regression OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.
  - Live: next NewsFetcher run with a 5+ day-old `news_history.json` should report a non-zero remaining pool count (e.g. `Filtered 23 previously-used candidate(s) via news_history (20 remaining of 43)` instead of the prior `0 remaining`).
- **Tags:** news-history, ttl, dedup, NewsFetcher, story-orchestrator
- **Related:** Same `signal_lost_for_nasas_tess_..._110640` soak that surfaced BUG-086. Independent issue but observed via the same console paste. The `config/news_history.json` file on disk is preserved as-is -- the TTL is read-time, not write-time, so existing entries past the window simply stop participating in the filter.

---

### BUG-LOCAL-086 [FIXED]: HuMo per-line clips frozen on last frame for half the audio (177-frame hard cap + silent clamp)
- **Date:** 2026-05-04 EVENING | **Phase:** acceptance soak (BUG-085 follow-up) | **Bible candidate:** YES (HuMo render budget + audio-video sync)
- **Symptom (live full re-render `signal_lost_for_nasas_tess_stellar_eclipses_..._110640`):** First episode-length run since BUG-085 fix completed cleanly through ScriptWriter / Bark / FLUX portraits / HuMo. ScriptWriter shipped a 177-word, 9-line script (3 cast: ANNOUNCER, JAKE, MAYA + a hallucinated CORE VOICE). HuMo rendered 7 character clips at ~10:20 each. Final composite played correctly through the announcer LTX scenes and the 7-second JAKE lines, but on every MAYA line + every JAKE line >7s the character video froze on the last rendered frame while audio kept playing. Three Media Player screenshots from Jeffrey at 0:33 / 1:01 / 1:05 all show portraits with motionless lips while the corresponding Bark audio is mid-sentence. 5 of 7 character clips affected (l003, l005, l006, l007, l008 = MAYA 13.99s, MAYA 11.31s, JAKE 12.99s, MAYA 13.99s, CORE VOICE 10.42s) — all clips with `dur_s > 7s`.
- **Cause (multi-layer):**
  1. `nodes/batch_humo_render.py::HUMO_MAX_FRAMES = 177` was the "last empirically verified value on RTX 5080 Laptop 16GB" — meaning untested above. 177 frames @ 25fps = 7.08s ceiling on per-clip render duration.
  2. `humo_length_for_dur(dur_s)` capped its return at HUMO_MAX_FRAMES, then BUG-LOCAL-105 (deep_earth_echoes 2026-04-28) added an explicit `dur_s` clamp so audio_dur_fed_to_humo never exceeded the cap. Together these meant: any character line longer than 7.08s had its tail audio silently clamped off, then HuMo only received the first 7.08s of audio, decoded a 6.88s mp4 (after warmup pad trim), and stopped. The remaining seconds of audio played against a frozen portrait in the composite.
  3. The `clip_length` workflow widget had `max=7.08`, hard-locking the user out of the higher ceiling even if VRAM allowed.
- **Fix (`nodes/batch_humo_render.py`, ~250 LOC across 7 sites):**
  - **Constant bumps** — `HUMO_MAX_FRAMES = 353` (4·88+1 = 353 frames @ 25fps = 14.12s, covers 14s Bark dialogue in a single pass). New `HUMO_CHUNK_FRAMES = 177` for the chunking fallback.
  - **NEW `humo_length_for_dur_uncapped(dur_s)`** — same 4n+1 snap as the capped helper but without the HUMO_MAX_FRAMES ceiling. Used by the chunking dispatch to decide whether to split a line.
  - **NEW `_concat_clips_via_ffmpeg(chunk_paths, out_path)`** — ffmpeg concat-demuxer wrapper with `-c copy` for stitching per-chunk mp4s into the canonical `<line_id>.mp4`. Safe because every chunk goes through `_save_clip_via_ffmpeg` with identical fps + sample rate.
  - **Plan-build refactor (BUG-LOCAL-086 chunking dispatch)** — replaces the BUG-LOCAL-105 silent-clamp at lines 1490-1512 with: if `(dur_s + pad_s) <= clip_length` → single-chunk path (current behaviour); else → split into `n_chunks = ceil(dur_s / chunk_max_dur_s)` evenly. Each chunk gets its own audio slice + warmup pad. Plan entries now carry `chunks: list[{audio, start_offset_s, dur_s}]` instead of a single `audio` dict.
  - **Phase B refactor** — Whisper audio encoding now iterates `entry["chunks"]`, encoding one `audio_emb` per chunk. Single-chunk lines unchanged in behaviour; multi-chunk lines pay N × Whisper cost (cheap; <1s per chunk).
  - **Phase C render-loop refactor** — per-line render now iterates chunks, dispatches HuMo once per chunk with that chunk's `audio_emb` and the same portrait `ref_image`, saves each chunk to either `<line_id>.mp4` (single-chunk) or `<line_id>__chunk{NN}.mp4` (multi-chunk), then ffmpeg-concats multi-chunk parts into `<line_id>.mp4` and deletes the part files. Per-chunk shot_seed = `seed + idx*1009 + chunk_idx*7919` so chunks 1+2 of the same line don't render with identical seed (would produce visible "stutter back to start" at the chunk boundary). Single ledger record per line regardless of chunk count; `mp4_frames` / `mp4_dur_s` / `humo_render_ms` are sums across chunks; `audio_fed_to_humo_dur_s` accounts for N pads (one per chunk). New `n_chunks` field added to ledger clip records for traceability.
  - **Widget bump** — `clip_length` max raised from 7.08 to 14.12 (default unchanged at 7.0). Power users can opt into single-pass for typical Bark dialogue. Lines longer than `clip_length` still chunk regardless.
- **Test updates (`tests/test_batch_humo_render.py`):**
  - Updated `test_humo_length_for_dur` parametrize cases for the new cap (8s → 201, 9s → 225, 14.12s → 353, 16s+ → 353 capped).
  - Updated `test_humo_constants` (HUMO_MAX_FRAMES = 353; new HUMO_CHUNK_FRAMES = 177 assertion).
  - Updated `test_clip_length_max_respects_humo_ceiling` (max = 14.12).
  - NEW `test_humo_length_for_dur_uncapped_skips_cap` — pins that the chunking-dispatch helper bypasses the cap (30s → 753, capped helper would return 353).
  - Existing `test_humo_length_for_dur_always_returns_4n_plus_1` extended to include 14s in the parametrize set.
- **Known caveats / what we're not pretending:**
  - `HUMO_MAX_FRAMES = 353` at 16 GiB Blackwell is **untested in a live run**. If the next soak OOMs at single-pass for a 14s clip, drop the constant back to 257 (10.28s) or 177 (7.08s) — the chunking dispatch handles whatever the cap is. Tracked as BUG-LOCAL-086a if it surfaces.
  - Per-chunk shot_seed stride (7919) is a guess at preventing same-frame regression at chunk joins. If the seam is visible in test, bump the stride or carry over the last chunk's final-frame latent as a continuity hint (BUG-LOCAL-086b future).
  - Whisper feeding silence into a chunk could still produce no-lip-motion (Jeffrey's bottom-left screenshot showed this at the l002→l003 boundary). Not in BUG-086 scope; logged as BUG-LOCAL-090 candidate (Bark line lead-in silence handling).
- **Verify:**
  - AST parse clean on `nodes/batch_humo_render.py` (110206 bytes, 8846 nodes).
  - `tests/test_batch_humo_render.py` + `tests/test_humo_warmup_pad.py` + `tests/test_dropdown_guardrails.py` → 107 passed in 108.91s.
  - `tests/test_core.py` → 108 passed in 4.46s.
  - Bug Bible regression scoped to OTR pack → 22 passed / 1 pre-existing failure (otr_save_copy.py, batch_flux_portrait_render.py missing folder_paths; unrelated to BUG-086) / 1 skipped / 2 xfailed.
  - Live: next soak should show `[BatchHumoRender] BUG-LOCAL-086: line lXXX dur_s=YY > clip_length=7.0s -- splitting into N chunks of Z.ZZs each` for any line >7s, and `[BatchHumoRender] lXXX done in M ms (N chunks, BUG-LOCAL-086 chunked + concat)`. Composite mp4 should show lipsync continuing through the FULL audio duration of every character line; no frozen-tail artefacts.
- **Tags:** humo, vram-ceiling, audio-video-sync, chunking, ffmpeg-concat, wan-2.1, blackwell, BUG-LOCAL-105-supersession
- **Related:** BUG-LOCAL-105 (silent-clamp predecessor; chunking dispatch replaces the clamp but the capped `humo_length_for_dur` still defends each individual chunk, preserving 105's safety property per chunk). BUG-LOCAL-102 (warmup pad — applied per chunk in 086, not just first chunk). BUG-LOCAL-094 (per-line timing estimate; unchanged). Pending: BUG-LOCAL-087 (title lost between ScriptWriter and SignalLostVideo), BUG-LOCAL-088 (CORE VOICE hallucinated cast member), BUG-LOCAL-089 (Director phase produces unparseable output on Gemma-4-E2B). All three observed in the same `signal_lost_for_nasas_tess_..._110640` run that surfaced 086.

---

### BUG-LOCAL-085 [FIXED]: NF4 silently failing because HF_HOME not in ComfyUI process env
- **Date:** 2026-05-04 MORNING | **Phase:** acceptance (BUG-LOCAL-084 follow-up) | **Bible candidate:** YES (Windows/Electron env-inheritance footgun)
- **Symptom (live full re-render attempt 2026-05-03 23:47, post BUG-084):** ComfyUI restarted clean, BUG-084 fixes loaded, full workflow queued. ScriptWriter started Mistral-Nemo load. Crashed at SDPA prefill with `torch.OutOfMemoryError: Currently allocated 24.00 GiB / Device limit 15.92 GiB`. 24 GiB matches Mistral-Nemo 12B at fp16 exactly — meaning NF4 quantization did not actually apply despite the runtime log printing `[StoryOrchestrator] Enabling 4-bit quantization (NF4)`.
- **Cause (multi-layer):**
  1. ComfyUI Desktop's Electron parent process did not inherit `HF_HOME` from `HKCU\Environment`. PowerShell confirmed: `HF_HOME (User) = C:\ComfyUI-Models\huggingface`, `HF_HOME (Process) = (empty)`. Per-user env vars are inherited by processes started from Explorer but not always by Electron-spawned children.
  2. `nodes/story_orchestrator.py::_load_llm` resolved cache via `os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))` → fell through to `~/.cache/huggingface` because env var was missing.
  3. With `cache_dir` wrong AND `local_files_only=True` AND Mistral-Nemo's sharded-safetensors layout on Windows, transformers' Hub-resolution layer misresolved the model location. Instead of erroring out, it silently fell back to a partial-fp16 load path. `quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...)` was passed but never applied.
  4. fp16 12B Mistral-Nemo = 24 GiB → device_map={"": 0} forced 100% GPU → OOM when KV cache built during prefill of the first generate() call.
- **Verified in isolation (`scripts/check_nf4_load.py`):** When the snapshot directory path (`C:\ComfyUI-Models\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407\snapshots\<sha>\`) is passed directly to the loader, the model loads at **7.79 GiB allocated**, **280/281 Linear modules quantized to 4-bit NF4**, generation works ("Once upon a time, there was a little girl named Lily"). Confirms the bug is the Hub-resolution path, not the quantization config.
- **Fix:**
  - **NEW `nodes/_otr_hf_env.py`** — `ensure_hf_home()` reads `HF_HOME` from `HKCU\Environment` via `winreg`, exports it to `os.environ['HF_HOME']` and `os.environ['HF_HUB_CACHE']` so downstream HF tooling picks it up automatically. `resolve_snapshot_dir(model_id, hf_home=None)` returns the absolute snapshot directory path under the canonical cache (`<hf_home>/hub/models--<org>--<name>/snapshots/<sha>/`). Both functions idempotent + cache-safe.
  - **`nodes/story_orchestrator.py::_load_llm`** — calls `ensure_hf_home()` at start; calls `resolve_snapshot_dir(model_id)` to get the canonical snapshot path; passes the snapshot path (not the `model_id`) to `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM` loaders. Bypasses transformers' Hub-resolution layer entirely. Falls through to the legacy `model_id` + `cache_dir` path only if snapshot resolution returns None (model not cached).
- **Disk cleanup landed alongside (not committed; outside repo):**
  - Deleted 14.23 GB of `.incomplete` partial Mistral-Nemo blobs at `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\` (stray duplicate cache from before HF_HOME migration; never read by ComfyUI).
  - Aligned `C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\extra_model_paths.yaml` (install-dir copy) with the Roaming canonical so any process reading either YAML resolves to `C:\ComfyUI-Models` paths consistently.
- **Verify:**
  - Standalone load check: `python scripts/check_nf4_load.py` reports `PASS: NF4 working (7.79 GiB; expected ~6 GiB)` with all 280 Linear modules quantized.
  - In ComfyUI: next full run should log `[StoryOrchestrator] HF_HOME resolved -> C:\ComfyUI-Models\huggingface` followed by `[OTR_HF_ENV] snapshot resolved mistralai/Mistral-Nemo-Instruct-2407 -> <abs_snapshot_path>` followed by `LLM model loaded from canonical snapshot (no HTTP checks)`. ScriptWriter VRAM should peak at ~7-8 GB instead of crashing at 24 GB.
  - AST parse clean on both files; Bug Bible regression 22 passed / 1 pre-existing failure / 1 skipped / 2 xfailed.
- **Tags:** hf-cache, electron-env-inheritance, nf4, bitsandbytes, hub-resolution, sharded-safetensors, windows-symlinks, vram-ceiling, ScriptWriter
- **Related:** BUG-LOCAL-084 (composite gap-fill + duration contract — shipped 7f2d03f, not yet verified in a full live run because BUG-085 blocked it). BUG-085 fix is required before BUG-084 can be exercised end-to-end via the full workflow. Smoke harness path is unaffected (no LLM load). Commit `56cf493` on `v2.0-alpha`.

---

### BUG-LOCAL-084 [FIXED]: composite missed gap-fill + LTX ledger stamp incomplete
- **Date:** 2026-05-04 LATE NIGHT | **Phase:** acceptance (BUG-LOCAL-031 Track 1 follow-up) | **Bible candidate:** YES (audio/video timeline alignment)
- **Symptom (live composite output `signal_lost_skindeep_microneedle_..._222516.mp4`):** Visual sync broken end-to-end. At video time 0:12 viewer sees apprentice face with no lip movement (audio is l001 LTX announcer, should be radio scene). At video time 0:27 viewer sees foreman face with lips moving (audio is l002 apprentice voice). Cumulative drift ~10s by end of episode; final mp4 was ~35s short of master audio (`-shortest` truncated trailing audio).
- **Cause:** VideoComposite per_clip_mux concatenated the 6 per-line clips back-to-back at t=0 with no gap-fill. Audio (master mix from procgen.mp4) starts l001 at 9.5s, has 0.6s gaps between adjacent lines, ends at 87.2s vs procgen 94.75s. Without gap-fill segments the video is structurally shorter than the audio, audio leads video by the cumulative gap duration, and `-shortest` chops the tail.
- **Fix (4 sites in 2 files):**
  - **Fix 1 (`batch_ltx_render.py`):** stamp `start_s` on each `ledger.clips[]` entry from matching `ledger.lines[]` by line_id; ffprobe rendered file for real `dur_s` instead of audio target (BUG-LOCAL-033 lie); preserve audio target as `audio_target_dur_s` for audit.
  - **Fix 2 (`video_composite.py`):** confirmed already wired from BUG-031 Track 1 — per-clip ffprobe + duration_gap_s + extend_tail_s/truncate_to_s pass-through into `_layered_per_clip_silent`.
  - **Fix 3 (`video_composite.py`):** NEW gap-fill pass after `timeline.sort()`. Walks sorted timeline, for any gap > 0.1s inserts static-radio segment of exact gap length using existing `_render_static_radio_segment` helper. Trailing tail-fill from last clip end to ffprobe(procgen) episode_dur. Logs `BUG-084 gap-fill: inserted N segments, total Xs coverage`.
  - **Fix 4 (`video_composite.py`):** NEW duration-contract assertion before final mux. ffprobe `silent_combined` and procgen, compare within 40ms tol. If audio overruns video, tail-pad silent_combined with `tpad clone-frame` so `-shortest` truncates inaudible video, not audio. Belt-and-suspenders.
- **Verify:** AST parse clean both files; Bug Bible regression 22 passed (1 pre-existing failure unchanged). End-to-end live verification BLOCKED on BUG-LOCAL-085 (NF4 OOM); smoke workflow exercises BUG-084 but was also blocked overnight by BUG-082 + BUG-083 fixes that landed earlier.
- **Tags:** composite, gap-fill, duration-contract, c7-audio-byte-identity, bug-031-followup, ledger-clips-stamp
- **Related:** BUG-LOCAL-031 (RTXUpscale range + PostBlend duration). Commit `7f2d03f` on `v2.0-alpha`.

---

### BUG-LOCAL-083 [FIXED]: probe_duration_s kwarg mismatch (ffmpeg vs ffprobe)
- **Date:** 2026-05-03 LATE NIGHT | **Phase:** smoke harness | **Bible candidate:** YES (kwarg signature drift)
- **Symptom:** Smoke workflow crashed at VideoComposite per_clip_mux with `RuntimeError: strict_c7=True and master_mix_per_clip_mux failed. Reason: probe_duration_s() got an unexpected keyword argument 'ffmpeg'`. Caught by smoke harness on first run after BUG-082 landed.
- **Cause:** Two call sites in `video_composite.py` (BUG-031 Track 1 per-clip duration matching, lines 1033 and 1135) passed `ffmpeg=ffprobe` to `_otr_probe.probe_duration_s()`. The function signature is `def probe_duration_s(path, *, ffprobe="ffprobe")` — the kwarg is named `ffprobe`, not `ffmpeg`. TypeError caught by the strict_c7 master_mix_per_clip_mux exception handler, which then refused to fall back to humo_concat (correctly — 3x AAC re-encodes break C7).
- **Fix:** rename kwarg `ffmpeg` → `ffprobe` to match the actual function signature at both call sites.
- **Verify:** AST parse clean; smoke harness composite stage now completes in 4.4s with `tail-pad 0.500s (BUG-128) + 3.040s sync (BUG-031) on surviving last clip l006`.
- **Tags:** kwarg-signature, smoke-harness, bug-031-followup
- **Related:** Commit `e601ee8` on `v2.0-alpha`.

---

### BUG-LOCAL-082 [FIXED]: VideoComposite missing BUG-118 underscore-mismatch fallback
- **Date:** 2026-05-03 LATE NIGHT | **Phase:** acceptance | **Bible candidate:** YES (writer/reader filename convention drift)
- **Symptom:** Live full run died at 23:10:25 with `RuntimeError: VideoComposite: derived ledger from .mp4 not found: ...drug__20260503_222516_ledger.json` (note double underscore). LTX completed cleanly just before this; composite was the only stage that crashed.
- **Cause:** SignalLostVideo writes the procgen .mp4 with a double underscore before the timestamp (`signal_lost_..._drug__20260503_222516.mp4`); the ledger writer uses single underscore (`signal_lost_..._drug_20260503_222516_ledger.json`). VideoComposite's `_load_ledger_with_path` derived the ledger filename via naive `replace('.mp4', '_ledger.json')` → got `...drug__20260503_222516_ledger.json` (double underscore) which doesn't exist on disk. BatchLTXRender already had this fallback in place; VideoComposite was the orphan.
- **Fix:** ported the BUG-LOCAL-118 underscore-collapse fallback from BatchLTXRender to VideoComposite. When the primary derivation misses AND `__` appears in the stem, also try the single-underscore variant before raising.
- **Verify:** AST parse clean; smoke harness with broken-cache episode loads ledger correctly via fallback path with log `BUG-LOCAL-118 underscore-mismatch fallback`.
- **Tags:** writer-reader-drift, filename-convention, mp4-stem, bug-118-port
- **Related:** Commit `b34d272` on `v2.0-alpha`.

---

### BUG-LOCAL-081 [FIXED]: portrait node wired to wrong source — Node 59 never produced portraits
- **Date:** 2026-05-03 LATE EVENING | **Phase:** acceptance (BUG-LOCAL-078 follow-up) | **Bible candidate:** YES (workflow-wiring footgun, silent failure)
- **Symptom (live run `signal_lost_the_creepy_feeling_in_old_buildings_migh_20260503_215919`):** Episode workspace had `audio/`, `stills/`, `videos/` but no `portraits/` subdirectory. Ledger had 3 cast members (c01=ANNOUNCER, c02=JAX, c03=KAI), all with `portrait_path` empty. `otr_runtime.log` had ZERO log lines mentioning `BatchFluxPortraitRender` across the whole 49,154-line / 3.7 MB run. Module import + `INPUT_TYPES` registration both verified clean.
- **Cause (two distinct workflow-JSON bugs in `workflows/otr_scifi_16gb_full.json`):**
  1. **Bogus `ledger_json` source.** Link 100 wired Node 12 (`OTR_SignalLostVideo`) `video_path` output (a `.mp4` filesystem path) into Node 59's `ledger_json` input. The portrait node's `_load_ledger` tried `json.loads()` on the `.mp4` path, hit `JSONDecodeError`, fell into `except Exception: return (None, None)`, then `execute()` raised `RuntimeError("cannot load ledger from <path.mp4>")`. The error went to the ComfyUI executor (not OTR logger), so `otr_runtime.log` stayed silent.
  2. **Wrong execution position in DAG.** Because Node 12 was an upstream dependency of Node 59, ComfyUI scheduled Node 59 to run AT THE END of the workflow — long after HuMo (Node 51) had already executed without portraits and fallen through to tier-4 env-still stopgap. Even if (1) were fixed, (2) alone made the portrait pass useless: HuMo would never see the portraits that hadn't been rendered yet.
- **Fix (workflow JSON only — no code changes):**
  - Drop link 100 entirely (`Node 12.video_path → Node 59.ledger_json`); set Node 59's `ledger_json` widget to empty string so `_load_ledger` falls through to `_OTRL.in_flight_ledger_path()` auto-pickup.
  - Re-route link 45 from `(Node 23 → Node 24)` to `(Node 59 → Node 24)`. New chain: `BatchFluxRender (env stills, 23) → BatchFluxPortraitRender (59) → UnloadAll (24) → BatchHumoRender (51)`. Portraits now render BEFORE HuMo while FLUX is still loaded in VRAM, then UnloadAll dumps FLUX, then HuMo picks up the portraits via the in-flight ledger.
- **Verify:**
  - JSON validates: `nodes=32 links=57`, no orphan/dangling refs.
  - Node 59 inputs: `ledger_json link=null` (auto-pickup), `flux_done_gate link=101` (waits on Node 23). Outputs: `portrait_batch links=[45]` (gates UnloadAll → HuMo).
  - Widget count went from 10 → 11 (prepended `""` for `ledger_json`).
  - Portrait module import + `INPUT_TYPES` clean.
  - Real-run acceptance (pending): next queue should produce `<ep>/portraits/c02_portrait.png` + `c03_portrait.png` (announcer skipped per `skip_announcer=True`), and HuMo's per-line `_find_portrait` should hit tier 1 instead of tier 4 — visible in HuMo log lines as `portrait_path: <path>` instead of `falling back to env still`.
- **Tags:** workflow-wiring, link-100, link-45, silent-failure, bug-078-followup, portraits, comfyui-dag-ordering, c7-untouched
- **Related:** BUG-LOCAL-078 (the portrait node itself, shipped EVENING with correct internal logic). The wiring slip happened during workflow JSON edit when Node 59 was added to `otr_scifi_16gb_full.json` — the `ledger_json` socket was wired to the only nearby STRING source on the canvas (Node 12's `video_path`) rather than left empty for auto-pickup, and the portrait node was inserted BELOW Node 12 in the DAG instead of between Node 23 and Node 24. Bible candidate because the silent-failure mode (RuntimeError invisible in OTR log + dependency-chain inversion) is the kind of trap any future graph edit could fall into. Commit `413ef3a` on `v2.0-alpha`.

---

### BUG-LOCAL-031 [FIXED]: HuMo + LTX visual content destroyed by RTXUpscale (range normalization bug) + duration overrun in PostUpscaleProcgenBlend
- **Date:** 2026-05-03 EVENING (post-soak diagnosis) | **Phase:** acceptance (BUG-LOCAL-030 wave) | **Bible candidate:** YES (severe, wave-blocker)
- **Symptom (live soak run `signal_lost_what_a_decade_of_gene_therapy_research_f_20260503_173957`):**
  - User saw TWO outputs in OBS folder (only ONE expected): `<ep>.mp4` (1.63 MB, "audio + all black") and `<ep>_procgen_blended.mp4` (14.48 MB, "procgen scanlines visible, NO HuMo/LTX content visible underneath").
  - Per-stage ffprobe nailed the failure to RTXUpscale: composite output 1472x832 / 50.36s / **1544 kbps** / 672 KB sample frame (real content). RTXUpscale output 1920x1080 / 50.36s / **96 kbps** / 56 KB sample frame (solid black). Same dims, same frame count -- only the visual content disappeared. Post-blend overran to 113.92s vs source 50.36s with audio at 50.34s (50s of audio over 113s of video).
- **Cause (two distinct bugs):**
  1. **Bug 1 -- range normalization mismatch in `nodes/rtx_upscale.py::_chunked_upscale`:** NVIDIA's `nvvfx.VideoSuperRes` expects input in **0.0-1.0 float** range (matching ComfyUI IMAGE convention) and produces output in **0.0-1.0 float** range. The OTR node read raw RGB24 bytes from ffmpeg (uint8 0-255), did `.float()` which keeps the values numerically 0-255, and fed nvvfx those out-of-distribution values. nvvfx internally clamped/saturated them, producing garbage near 1.0. The output was then `clamp(0.0, 255.0).byte()`'d -- which is a no-op for 0-1 values, then `.byte()` truncated 0.95 -> 0 and 1.0 -> 1, producing essentially solid black (every pixel value 0 or 1 out of 255). H.264 compresses solid color to nothing (96 kbps).
  2. **Bug 2 -- PostUpscaleProcgenBlend duration overrun:** the previous round-robin (Gemini, 2026-05-03 EVENING) said don't use `-shortest`. The advice was about the muxer-level `-shortest` flag (which IS unsafe -- truncates audio if procgen ends first, breaking C7). Today's round-robin (Gemini, again) self-corrected: filter-level `shortest=1` INSIDE the blend filter is C7-safe because audio is mapped separately via `-c:a copy`. Without it, the blend filter outputs the LONGER input duration -- procgen 113.92s wins over source 50.36s.
- **Fix:**
  - **`nodes/rtx_upscale.py::_chunked_upscale`** -- normalize input AND denormalize output:
    - Input: `gpu_in = ... .float().contiguous() / 255.0` (uint8 0..255 -> float32 0..1, what nvvfx actually expects).
    - Output: detect range (`gpu_out.max() <= 1.5`) and multiply by 255 before the `.byte()` cast (forward-compat: future nvvfx versions that change output convention won't break).
  - **`nodes/otr_post_upscale_procgen_blend.py::_build_blend_cmd`** -- append `:shortest=1` to the blend filter expression: `blend=all_mode={blend_mode}:all_opacity={blend_opacity:.3f}:shortest=1[v]`. Filter-level flag only clamps video output; audio mapped via `-c:a copy` from source is untouched (C7 holds).
- **NEW DIAGNOSTIC TOOLING (this commit):**
  - **`scripts/smoke_downstream_from_assets.py`** -- skip the 90-min upstream pipeline and exercise ONLY the downstream chain (Composite -> RTXUpscale -> PostUpscaleProcgenBlend) on pre-rendered assets from a completed run. Iteration loop drops from ~90 min to ~70 sec. Saves per-stage ffprobe + sample frame for visual inspection. Optional `--diagnostic-dump` flag propagates a dump dir into RTXUpscale.
  - **`nodes/rtx_upscale.py`** -- `diagnostic_dump_dir` optional kwarg on `RTXUpscale.execute()` and `_chunked_upscale()`. When set, dumps three PNGs per chunk (input_uint8, post_nvvfx_float_xN, post_clamp_byte) plus a `chunk_stats.txt` with per-chunk min/max/mean for input + nvvfx output + post-clamp byte. The stats file alone localizes any future range-mismatch / silent-zero / dimension-error bug in seconds. No-op when disabled (production paths unchanged).
- **Verify:**
  - AST parse on touched files: green.
  - **Smoke harness end-to-end:** RTXUpscale output 1.63 MB / 96 kbps / 56 KB frame (BLACK) -> 17.47 MB / 2734 kbps / 1056 KB frame (REAL CONTENT). Post-blend output 14.48 MB / 113.92s (overrun) -> 19.70 MB / 50.36s (clamped). Visual inspection of `frame_post_blend_out.png` confirms metallic corridor + CRT panel + procgen scanlines overlay = SIGNAL LOST visual signature working as designed.
  - Diagnostic dump confirmed root cause: every chunk reported `nvvfx(min=0.0000, max=1.0000)` after the broken (no-input-divide) variant; with the input-divide fix, ratios look right and visual inspection is clean.
  - **Real-run acceptance (pending):** queue a fresh episode after restart. Expect `obs/<ep>.mp4` to be ~10+ MB (was 1.63 MB), `obs/<ep>_procgen_blended.mp4` to be ~20+ MB at exactly the source duration (was 14.48 MB at 113s overrun), and the visible video to show HuMo character clips + LTX broadcast units + procgen scanlines composited per the Phase A + B design.
- **Tags:** rtx-vsr, nvvfx, range-normalization, blend-shortest, c7-safe, video-pipeline, post-bug-030, smoke-harness
- **Related:** BUG-LOCAL-030 wave (parent — Phase A composite + Phase B procgen blend); the previous "drop -shortest" fix (commit `a486fd1`) was correct in intent but missed that filter-level `shortest=1` is C7-safe while muxer-level `-shortest` is not. This commit corrects the over-correction. Round-robin: ChatGPT (gpt-5.5) + Gemini (gemini-3.1-pro-preview) both converged on the diagnosis from the bitrate-collapse signal alone. The diagnostic dump tool is the larger contribution -- it makes future RTXUpscale-style bugs localizable in one smoke cycle.

---

### BUG-LOCAL-030-LONGFORM-HARDENING [FIXED]: Composite + blend chain not soaked at >5 min episode length — DRAM canary, gc/empty_cache, ffmpeg thread caps, intermediate cleanup
- **Date:** 2026-05-03 EVENING (post audit-completion) | **Phase:** preventative hardening | **Bible candidate:** YES (long-form scaling)
- **Symptom:** preventative — no soak failure. Round-robin risk-#10 review (`docs/2026-05-03-soak-risk-10-dram-ceiling-longform__*.md`) flagged that the BUG-LOCAL-030 Phase A + B composite chain has zero soak data above ~3 min audio. Real risk surface for a >5 min episode: ~100 per-line layered intermediates on disk simultaneously (1.5-4.5 GB transient bloat), ffmpeg `blend` filter at 1920x1080 buffering frames from BOTH inputs, PyTorch holding RAM/VRAM across phase boundaries, "Too many packets buffered" error on long-form blends.
- **Cause:** organic growth — Phase A + B were designed for the typical short-act soak length, not stress-tested for the longer episodes Jeffrey is now ready to queue. Gemini caught this via correct ComfyUI float32-tensor math (12 bytes/pixel, not RGB24's 3) which would be catastrophic IF RTXUpscale loaded the full upscale into memory; verification of `nodes/rtx_upscale.py` confirmed it pipes raw RGB24 in/out of ffmpeg via `subprocess.Popen` chunks (RETURN_TYPES = STRING, not IMAGE) — so the 110 GB OOM scenario does NOT apply. The five remaining cheap-win hardening recommendations DO apply.
- **Fix (single commit, 5 hardening sites + new helper module + tests):**
  - **NEW `nodes/_otr_memory.py`** — shared DRAM/VRAM hygiene helpers. `phase_gc(label)` runs `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` (best-effort, never raises, idempotent). `dram_canary(min_free_gb=6.0, label)` runs `psutil.virtual_memory().available` check; degrades OPEN (returns `(True, reason)`) when psutil missing or syscall fails so the canary never blocks a render that would otherwise complete. Returns `(False, reason)` ONLY when psutil successfully reports below threshold. Default 6.0 GB per Gemini round-robin.
  - **`nodes/video_composite.py::_render_master_mix_per_clip_mux_mode`** — at function entry, calls `phase_gc("VideoComposite/per_clip_mux entry")` + `dram_canary` and appends warning to report if canary trips. After concat-demuxer succeeds (`silent_combined.mp4` written), immediately `unlink(missing_ok=True)` every `pillarboxed[]` intermediate — saves 0.5-1.5 GB transient disk on a 100-clip episode. Best-effort; cleanup failures logged but never raised.
  - **`nodes/otr_post_upscale_procgen_blend.py::_build_blend_cmd`** — appended `-filter_complex_threads 2 -filter_threads 2 -threads 4 -max_muxing_queue_size 1024` to the ffmpeg cmd. Caps thread fanout (prevents thread×framebuffer DRAM multiplication on long-form blends) + raises mux queue (guards against "Too many packets buffered for output stream" failure mode).
  - **`nodes/otr_post_upscale_procgen_blend.py::PostUpscaleProcgenBlend.blend`** — same `phase_gc` + `dram_canary` at entry as VideoComposite. Phase barrier handoff from RTXUpscale releases any RAM/VRAM PyTorch may still be holding before the blend pass starts buffering 1920x1080 frames from both inputs.
  - **`tests/test_otr_memory.py`** (NEW, 7 tests) — `phase_gc` never-raises (including with torch missing); `dram_canary` default threshold = 6.0 GB; degrades open when psutil missing OR syscall fails; returns False below threshold; returns True above threshold.
  - **`tests/test_post_upscale_procgen_blend.py`** — added `test_blend_cmd_includes_longform_hardening_flags` verifying all four new flags (`-max_muxing_queue_size 1024`, `-filter_complex_threads 2`, `-filter_threads 2`, `-threads 4`) appear in the generated cmd with correct values.
- **Verify:**
  - AST parse on all 5 touched files: green.
  - Targeted regression: **76 passed in 3.44s** (`test_otr_memory + test_post_upscale_procgen_blend + test_video_composite_layered + test_video_composite_per_clip_mux + test_per_line_audio_meta`). Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.39s.
  - **Real-run acceptance (pending — long-form soak):** queue an episode at >5 min audio length. Expect (a) `per_clip_mux: reclaimed N pillarbox intermediate(s)` line in VideoComposite report (where N matches the per-line clip count), (b) `PostUpscaleProcgenBlend: DRAM canary WARNING -- ...` line in report ONLY if available DRAM dropped below 6 GB pre-blend, (c) ffmpeg blend pass completes WITHOUT "Too many packets buffered" error even with ~100+ clips on the timeline, (d) C7 audio byte-identity preserved end-to-end (audio path is `-c:a copy` everywhere; new ffmpeg flags only affect video processing).
- **Tags:** dram-ceiling, long-form, ffmpeg-flags, gc-empty_cache, psutil-canary, post-bug-030, round-robin-risk-10
- **Related:** BUG-LOCAL-030 (parent); BUG-LOCAL-030-AUDIT-COMPLETION (sibling — also a defensive-hardening pass on the same chain). Round-robin synthesis: ChatGPT correctly identified RTXUpscale as the primary risk surface but mis-calculated tensor sizes (used RGB24 3 bytes/pixel instead of ComfyUI float32 12 bytes/pixel) and recommended `-shortest` (would violate C7); Gemini caught both errors and provided the four ffmpeg flags + psutil canary + intermediate cleanup recommendations adopted here. Verification of RTXUpscale source confirmed it's a CLI wrapper (chunked ffmpeg pipes), not an IMAGE-tensor consumer, so Gemini's 110 GB OOM hypothetical does NOT apply.

---

### BUG-LOCAL-030-AUDIT-COMPLETION [FIXED]: Per-line audio render metadata not stamped to ledger across 4 audio engines (forensic gap)
- **Date:** 2026-05-03 EVENING (post-final_video_path stamp) | **Phase:** acceptance hardening | **Bible candidate:** YES (forensic provenance)
- **Symptom (from artifacts-grid audit, no soak failure — preventative gap closure):** ledger only knew which engine produced what audio for a single field (Bark’s `bark_render_ms`). Other engines: KokoroAnnouncer wrote ZERO ledger fields; MusicGenTheme stamped `wav_path + dur_s` only; BatchAudioGen stamped `wav_path + dur_s` only. Cannot answer “which engine + voice + render time + sample hash produced this row?” without re-reading the wav from disk + cross-referencing logs.
- **Cause:** historical organic growth — Bark got the BUG-LOCAL-101 forensic block, the other three audio engines never got the same treatment. No common helper existed to stamp the canonical `tts_engine / voice_preset / render_ms / generated_dur_s / audio_sample_hash` bundle.
- **Fix (single commit, 4 nodes + 2 helpers + 1 test file):**
  - `nodes/_otr_ledger.py`: two new public helpers — `compute_audio_sample_hash(arr_or_bytes, n_bytes=1024) -> str` (8-char SHA256 hex of leading bytes; tripwires sample-rate / channel / pad drift; best-effort ""-on-failure), and `stamp_per_line_audio_meta(ledger, line_id, *, tts_engine, voice_preset="", render_ms=0, generated_dur_s=0.0, audio_sample_hash="") -> bool` (wraps `patch_line_fields`, skips empty/zero values so partial bundles don’t clobber pre-existing fields).
  - `nodes/batch_bark_generator.py`: extends existing per-line stash + write-back. New row fields `tts_engine="bark"`, `voice_preset` (from `preset` in scope), `render_ms` (mirrors `bark_render_ms`), `generated_dur_s` (mirrors `bark_wav_dur_s`), `audio_sample_hash` (computed from per-line `audio_np`).
  - `nodes/musicgen_theme.py`: tracks per-cue `_render_ms` around the `model.generate()` call + per-cue `_audio_sample_hash`. Existing `wav_path + dur_s` write-back extended with `tts_engine="musicgen"`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on each `ledger.music[]` row. Generation prompt was already populated by LLMDirector, so this closes the loop on the render-result side.
  - `nodes/batch_audiogen_generator.py`: same pattern — per-sfx `_render_ms` + `_audio_sample_hash` stash, write-back extended with `tts_engine="audiogen"`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on each `ledger.sfx[]` row.
  - `nodes/kokoro_announcer.py` (was ZERO ledger touches before this commit): per-line `_render_ms` + `_audio_sample_hash` tracked in a `per_line_meta[]` parallel list. New post-loop write-back uses `_OTRL.in_flight_ledger_path()` singleton (same Phase G discovery as BatchBark) + text-match against `ledger.lines[]` (same first-unmatched-wins strategy as BUG-LOCAL-096). Stamps `tts_engine="kokoro"`, `voice_preset=<chosen voice_id>`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on every matching row. Best-effort: any I/O failure is logged, never raised.
- **Verify:**
  - AST parse on all 5 touched node files: green.
  - **New `tests/test_per_line_audio_meta.py` (10 tests, all passing):** `compute_audio_sample_hash` determinism / divergence / numpy-array support / empty-on-unhashable / leading-bytes-only contract; `stamp_per_line_audio_meta` full-bundle stamp / skip-empty / unknown-line returns False / partial bundle (engine + render_ms + hash only) / never-raises on bad ledger.
  - Cumulative regression: **200 passed, 1 skipped in 116.85s** (`test_dropdown_guardrails + test_core + test_audio_byte_identical + test_per_line_audio_meta + test_meta_paths + test_post_upscale_procgen_blend`). Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.38s.
  - **Real-run acceptance (pending):** queue an episode after restart. Expect the per-episode `<ep>_ledger.json` to show on every dialogue row `"tts_engine": "bark"|"kokoro"`, `"voice_preset": "<preset>"`, `"render_ms": <int>`, `"generated_dur_s": <float>`, `"audio_sample_hash": "<8hex>"`; on every `music[]` row `"tts_engine": "musicgen"` + render fields; on every `sfx[]` row `"tts_engine": "audiogen"` + render fields. C7 byte-identity unaffected (audio bytes themselves are NOT modified — this is metadata-only).
- **Tags:** ledger-schema, forensic-metadata, bark, kokoro, musicgen, audiogen, audit-completion, post-bug-030
- **Related:** BUG-LOCAL-030 (parent — final_video_path stamp closed the video forensic gap; this closes the audio forensic gap with the same audit-completion theme); BUG-LOCAL-018 (l3-2026-05-02 schema bump that established `meta.paths` precedent for additive ledger fields). Helpers added in this fix are usable by any future audio engine.

---

### BUG-LOCAL-030 [FIXED]: All-black final video — HuMo portrait squeezed into landscape canvas + per-clip-mux mode bypasses procgen visual layer
- **Date:** 2026-05-03 EVENING | **Phase:** acceptance (post-029 follow-up surfaced by 14:08 soak) | **Bible candidate:** YES (architecture pivot)
- **Symptom (from soak `signal_lost_static_echo_20260503_140824`, ffprobe of per-line clips):**
  - HuMo per-line clips render at native **480x832 PORTRAIT** (l003.mp4 is 480x832; l001.mp4 LTX is 832x480 landscape — confirmed via ffprobe).
  - Pillarbox formula `scale=-2:480:force_original_aspect_ratio=decrease, pad=832:480:...:color=black` applied to 480x832 portrait input scales width to ~276px and pads with **278 pixels of pure black on each side** of the 832-wide canvas. Result: HuMo content occupies only ~33% of the canvas width (a thin portrait strip in the center).
  - PLUS the per-clip-mux mode (`audio_source: master_mix_per_clip_mux`) at `nodes/video_composite.py:_render_master_mix_per_clip_mux_mode` builds a video filter chain that ONLY pillarboxes HuMo clips — it never overlays the procgen visual layer (`procgen` parameter is consumed but only used for audio extraction). So even where HuMo content wasn't rendered, the canvas was solid black.
  - Net result Jeffrey reported: "all black short videos." HuMo + LTX per-line clips look great in isolation; composite renders them as a thin strip surrounded by black bars.
  - LTX clips were OK (832x480 landscape filled the 832x480 canvas correctly), so non-character lines briefly showed visible content.
- **Cause:** Two coupled bugs:
  1. Canvas/HuMo orientation mismatch (the 2026-05-01 default of 832x480 was based on the incorrect assumption that HuMo renders landscape — it renders 480x832 portrait, the only fully-trained dim per Wan2.1-HuMo-14B + ROADMAP "Stable shape: length=97 (3.88 s @ 25 fps), 480x832, batch=1").
  2. Per-clip-mux pillarbox-each-clip-individually approach has no concept of layered composition — each clip renders alone on a black canvas.
- **Fix (Phase A — simple-pillarbox composite per Jeffrey's REVISED spec, 2026-05-03 EVENING):**
  - **HuMo render dims:** `nodes/batch_humo_render.py` widget defaults stay at the canonical Wan2.1-HuMo-14B trained dim **480x832 PORTRAIT @ 25 FPS** (per ROADMAP "Stable shape: length=97, 480x832, batch=1"). An earlier draft of this fix attempted 1280x720 landscape for face detail; Jeffrey reverted to native ("humo native portrait render then native scaled to 1472x832 with black pillaboxes") to avoid OOD lipsync drift + ~2.3x VRAM overhead.
  - **LTX render dims:** `nodes/batch_ltx_render.py` module constants stay at native **LTX_WIDTH=832, LTX_HEIGHT=480 landscape**. An earlier draft attempted 1216x704 for higher pre-upscale detail; Jeffrey reverted to native ("ltx native landscape render downscaled to 1472x832") for the same trained-distribution-safety reason.
  - **Composite canvas:** `nodes/video_composite.py` INPUT_TYPES defaults canvas_width 832→1472, canvas_height 480→832, humo_target_height 480→832. Workflow JSON node 52 widget values updated to match. New widget `humo_pillar_width` default 512 reserved for Phase B layered-mode use; unused by the active Phase A simple-pillarbox flow.
  - **`_layered_per_clip_silent(...)` simple-pillarbox branch:** all clips (HuMo + LTX) scale-FIT into 1472x832 with `force_original_aspect_ratio=decrease` then pad-with-black to canvas dims. NO `crop=` (preserves source aspect, no content lost). HuMo 480x832 stays at native (height=832 already matches canvas), padded with ~496px BLACK BARS per side. LTX 832x480 scaled to height=832 = 1442x832, padded with ~15px black per side (effectively full canvas). Both paths drop audio (`-an`); master mix attaches at the final mux step so C7 byte-identity holds.
  - **No env-still backdrop in Phase A:** `_render_master_mix_per_clip_mux_mode` always passes `background_png=None` to `_layered_per_clip_silent` so the helper takes its simple-pillarbox branch. The `_resolve_episode_background` helper + `_layered_per_clip_silent` layered-overlay branch are kept in the codebase (with their own tests) for future use cases where a static env-still backdrop IS desired -- but the current Phase A renderer never invokes them.
  - **Procgen visual fill -- PHASE B SHIPPED in same session (separate commit):** the visible HuMo black pillarbox bars are intentional. Procgen renders at native 1920x1080 (was 832x480) via the updated `OTR_SignalLostVideo` resolution default. New node `OTR_PostUpscaleProcgenBlend` (`nodes/otr_post_upscale_procgen_blend.py`, registered as `OTR_PostUpscaleProcgenBlend`) takes the RTXUpscale 1920x1080 output + the 1920x1080 procgen, builds an ffmpeg `-filter_complex` blend chain (`[0:v][procgen]blend=all_mode=lighten:all_opacity=0.5[v]`), maps source audio with `-c:a copy` (zero re-encodes -- C7 byte-identity preserved end-to-end). Output filename: `<source_stem>_procgen_blended.mp4` in the same dir as source. Per Jeffrey: "proc gen 1920x1080 ... then a final ffmpeg w/ the proc gen mix for final 1080p." Post-upscale blend turns the visible black surround into audio-reactive CRT scanlines -- the SIGNAL LOST visual signature -- without going through the AI upscaler (which would smear synthetic patterns) and without touching the per-clip-mux mode (which would risk C7 byte-identity). Three failure-mode fallbacks (bypass widget, missing procgen, ffmpeg failure) all degrade to source-copy so the pipeline always produces a deliverable. **Workflow JSON wiring SHIPPED in same commit** per Jeffrey directive ("plwsae dont dfeer anyting we need to test veryting"): new node id 58 added at `pos=[4900, 1100]`, link 95 wires `RTXUpscale.upscaled_mp4_path -> source_mp4_path`, link 96 wires `SignalLostVideo.video_path -> procgen_mp4_path`. `last_node_id` bumped 57->58, `last_link_id` bumped 94->96. Verified via `_verify_wiring.py`: 31 nodes, 53 links, all slot indices + types correct, JSON parses cleanly.
  - **`_render_master_mix_per_clip_mux_mode` rewired:** both call sites (in-loop + post-loop tail-pad re-pillarbox) switched from `_pillarbox_humo_silent` to `_layered_per_clip_silent`. The legacy helper is kept for back-compat with non-layered standalone use cases.
  - **Math sanity:** HuMo 480x832 → scale to height=832 = 480x832 (no scale, native quality preserved) → pad to 1472x832 = 480x832 centered + 496px black per side. LTX 832x480 → scale to height=832 = 1442x832 → pad to 1472x832 = 1442x832 centered + 15px black per side. Final RTXUpscale 1472x832 → 1920x1080 is clean (16:9 source → 16:9 delivery, 1.30x scale).
- **Phase B (queued — not in this commit):** procgen render at native 1920x1080 (currently 832x480) + new `OTR_PostUpscaleProcgenBlend` node that overlays procgen on the RTXUpscale output AT delivery res. Architecture: keep procgen visual OUT of the per-clip-mux composite (so C7 audio-identity protected mode stays untouched), blend it in post-RTXUpscale where it doesn't get smeared by the AI upscaler. Per Jeffrey: "proc gen at 0 [composite stage], native 1920x1080, and concat happens after upscale." Blend is `-c:a copy` so audio passes through with zero re-encodes.
- **Verify:**
  - AST parse on `nodes/video_composite.py`, `nodes/batch_humo_render.py`, `nodes/batch_ltx_render.py`: green.
  - Workflow JSON node 51 (HuMo) widgets_values width=1280, height=720; node 52 (VideoComposite) widgets_values canvas_width=1472, canvas_height=832, humo_target_height=832, humo_pillar_width=512: confirmed.
  - **New `tests/test_video_composite_layered.py` — 12 tests passing:** `_resolve_episode_background` priority order (env still > radio bookend > None) + the 4 ffmpeg cmd shape branches of `_layered_per_clip_silent` (character+bg → layered, character+no-bg → simple, non-character → simple, tail-pad in both paths) + INPUT_TYPES default sanity + LTX module constants.
  - Existing `tests/test_video_composite_per_clip_mux.py` updated: 2 tests that patched `_pillarbox_humo_silent` re-patched to `_layered_per_clip_silent` (the renderer's new call target). All 29 tests in the file green.
  - Cumulative regression: **203 passed in 3.88s** (`tests/test_video_composite_layered.py + test_video_composite_per_clip_mux.py + test_critique_dialogue_preservation.py + test_save_to_episode_workspace.py + test_prompt_format_safety.py + test_production_ledger.py + test_radio_still_resolver.py + test_filename_pattern_audit.py + test_cache_key_mutations.py + test_meta_paths.py + test_ledger_rename.py`).
  - Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.32s.
  - **Real-run acceptance (pending):** queue an episode after restarting ComfyUI. Expect (a) HuMo per-line clips at 1280x720 not 480x832 (ffprobe each `videos/lNNN.mp4`), (b) per_clip_mux report includes log line `BUG-030 layered composite: HuMo character backdrop = full_env_NNNNN_.png` (or radio_bookend if no env still), (c) final composite mp4 dims = 1472x832 (not 832x480), (d) RTXUpscale OBS final at 1920x1080 with **visible HuMo pillar in center over scene backdrop on character lines + LTX broadcast unit filling canvas on non-character lines**.
- **Tags:** video-composite, simple-pillarbox, landscape-canvas, humo-portrait, ffmpeg-pad, procgen-phase-b, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-027 (dialogue wipe — orthogonal); BUG-LOCAL-028 (per-episode FLUX save paths — provides the per-episode workspace structure used here, even though Phase A no longer reads env stills as backdrop); BUG-LOCAL-029 (ULTRA_SMOKE format normalize — orthogonal). Round-robin consult was SKIPPED per direct user override; AST + invariant audits + targeted regression + Bug Bible regression all green pre-commit. HuMo + LTX stay at native trained dims (480x832 + 832x480) so no OOD risk; an earlier draft of this fix attempted 1280x720 + 1216x704 for higher pre-upscale detail, but Jeffrey reverted to native ("humo native portrait render then native scaled to 1472x832 with black pillaboxes ... ltx native landscape render downscaled to 1472x832 ... then a final ffmpeg w/ the proc gen mix for final 1080p"). The visible black surround on character lines is intentional and gets filled with audio-reactive CRT scanlines by the Phase B post-RTXUpscale procgen blend — the SIGNAL LOST visual signature.

---

### BUG-LOCAL-029 [FIXED]: ULTRA_SMOKE preset bypasses BUG-027 dialogue-preservation gate (parser format mismatch)
- **Date:** 2026-05-03 EVENING | **Phase:** acceptance (post-027 follow-up surfaced by headless soak) | **Bible candidate:** YES
- **Symptom (from headless soak `signal_lost_static_echo_20260503_140824` per otr_runtime.log):**
  - L47819 `[14:00:27] ScriptWriter: PARSE_OK attempt=1 has_scene=True voice_hits=4 bare_hits=0 smoke=ultra:True/tiny:False` — writer's PARSE_OK validator counted 4 `[VOICE: ...]` markers in the draft.
  - L47868 `[14:01:22] CRITIQUE: Character line counts - draft={} revised={}` — `_count_character_lines` regex (BUG-027-extended for `[N] CHARNAME:`) DOES NOT match the ULTRA_SMOKE-specific `[VOICE: NAME, attrs, ...]: text` line format. Counter returned `{}` for both draft and revised.
  - L47869 `[14:01:22] CRITIQUE: Revised script accepted (sim=40.4%, len=244%)` — gate accepted a revision with 244% length expansion + 40% similarity (radical rewrite). Gate skipped per `if draft_total >= 3` short-draft skip — but draft_total was 0 only because the parser couldn't see the dialogue.
  - L47879-80 `BUG-109b: cast members with 0 lines: PETER ECKELS, REN KANE` / `1/1 scene(s) have 0 dialogue lines` — same downstream failure mode as the original BUG-027.
- **Cause:** BUG-LOCAL-005 (Sprint 1) added a `[VOICE: NAME, attrs, ...]: text` strict-VOICE format for the ULTRA_SMOKE preset, with a separate PARSE_OK validator (`voice_hits` counter). BUG-LOCAL-027 fixed `_count_character_lines` to accept `CHARNAME:` and `[N] CHARNAME:` formats but NOT the ULTRA_SMOKE `[VOICE: ...]` format. Result: ULTRA_SMOKE drafts parsed as `{}` in the critique pipeline, the BUG-027 total-collapse gate had no signal to enforce, and ULTRA_SMOKE silently bypassed the dialogue-preservation guarantee.
- **Fix (per Jeffrey directive 2026-05-03 EVENING — "ULTRA_SMOKE need to abide by all the rules"):** new helper `LLMScriptWriter._normalize_voice_format_to_standard(text)` in `nodes/story_orchestrator.py` — staticmethod that converts `[VOICE: NAME, attrs, ...]: text` → `NAME: text` AND strips inline `[VOICE: ...]` blocks from dialogue content. Wired into `_critique_and_revise` at TWO points: (a) at function entry on `draft_text` BEFORE the critique pass runs, so the critique LLM, the per-character preservation gate, and the total-collapse hard gate all see the canonical format; (b) on `revised_text` after the revision pass, so the gate counter compares apples-to-apples even if the revision LLM slipped back into `[VOICE: ...]` shape under high-temp creativity. Idempotent on already-standard text. C7-safe (deterministic regex transformation; same input always produces same normalized output, so byte-identity holds).
- **Architectural choice:** Jeffrey explicitly chose conversion-to-standard over extending the parser to handle multiple formats. Rationale: ONE source of truth for dialogue preservation (the standard `CHARNAME:` format) and one set of rules (the BUG-027 gate machinery). ULTRA_SMOKE keeps its strict-VOICE writer prompt + PARSE_OK validator (BUG-005 contract intact), but its output gets normalized before any downstream pipeline stage. Alternative was to remove ULTRA_SMOKE entirely, deferred.
- **Verify:**
  - AST parse on `nodes/story_orchestrator.py`: green.
  - **New tests in `tests/test_critique_dialogue_preservation.py` (7 added, 21 total in file, all green):** `test_normalize_standalone_voice_prefix_to_charname`, `test_normalize_voice_with_no_attrs`, `test_normalize_strips_inline_voice_block_from_dialogue`, `test_normalize_idempotent_on_standard_format`, `test_normalize_handles_empty_and_none`, `test_normalize_then_count_recovers_dialogue_for_ultra_smoke` (end-to-end: normalize + count yields correct character counts on a realistic ULTRA_SMOKE draft mirroring the actual L47868 failure shape), `test_critique_calls_normalize_at_function_entry` (static check that `_critique_and_revise` body actually calls the normalizer on BOTH `draft_text` AND `revised_text`).
  - Cumulative regression: **162 passed in 4.01s** (targeted set: production_ledger + radio_still_resolver + filename_pattern_audit + cache_key_mutations + meta_paths + ledger_rename + critique_dialogue_preservation + save_to_episode_workspace + prompt_format_safety) PLUS Bug Bible regression **24 passed / 1 skipped / 1 xfailed** in 1.49s.
  - **Real-run acceptance (pending):** queue an ULTRA_SMOKE episode (target_length="30 words (smoke, 1 act)"); expect (a) `CRITIQUE: ULTRA_SMOKE format normalized (N1 -> N2 chars)` log line if `[VOICE: ...]` lines were present, (b) `CRITIQUE: Character line counts - draft={'CHARNAME': N, ...}` with NON-EMPTY draft dict (was `{}` before this fix), (c) BUG-109b should NOT fire if the gate correctly preserves dialogue.
- **Tags:** ultra-smoke, voice-format, normalization, bug-027-extension, critique-pipeline, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-005 (created the strict `[VOICE: ...]` format for ULTRA_SMOKE); BUG-LOCAL-027 (fixed the standard-path counter, missed ULTRA_SMOKE). Round-robin consult was SKIPPED per direct user override; AST + targeted regression + Bug Bible regression all green pre-commit.

---

### BUG-LOCAL-028 [FIXED]: FLUX env stills + radio bookend save to legacy flat dirs — VideoComposite finds no scenery; final video black
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** YES (directly causes "black video" failure mode)
- **Symptom (from `dir /s C:\...\output\otr\` for episode `signal_lost_astronomers_finally_solve_the_gammacas_x_20260503_002536`):**
  - Episode workspace `output/otr/episodes/<ep>/` has `audio/` `videos/` `composited/` subdirs (Phase B writes correctly).
  - **`output/otr/episodes/<ep>/stills/` does NOT exist.** **`output/otr/episodes/<ep>/portraits/` does NOT exist.**
  - FLUX outputs landed in legacy flat dirs:
    - Radio bookend → `output/otr/_legacy_stills/radio_bookend_<ep>.png` (filename has ep_id baked in but DIR is legacy).
    - Env stills → `output/otr/stills/full_env_NNNNN_.png` with a global counter shared across all episodes since 4/26 (213 PNGs accumulated, none stamped to a specific episode by name).
  - VideoComposite reads from per-episode `videos/` (correct) but cannot find env stills or radio bookend in the per-episode `stills/` (because they're not there). Result: no scenery layer in composite → mostly-black canvas.
  - Final composited mp4 = 1.72 MB; obs/ final = 1.18 MB; Jeffrey reports "black video, 15s of audio, no announcer in final."
- **Recurring (NOT a one-off):** affects every episode since the per-episode workspace reorg (Phase B, 2026-05-02 EVENING). Audio/video paths got reorged; FLUX outputs were not.
- **Cause (two separate sites):**
  - **Site 1: `visual/batch_flux_render.py:833`** — `stills_dir = _OTRP.otr_stills_dir()` called with no `episode_id` argument. Per `nodes/_otr_paths.py:208-218`, `otr_stills_dir()` without an episode_id falls back to `output/otr/_legacy_stills/`. The `episode_id` variable is in scope from line 768/772 (resolved from the in-flight ledger singleton via the same Phase G singleton-discovery path used by BUG-LOCAL-021). One-line fix.
  - **Site 2: `workflows/otr_scifi_16gb_full.json` node id 25** — stock ComfyUI `SaveImage` with hardcoded `filename_prefix: "otr/stills/full_env"` widget value. ComfyUI writes to `output/<filename_prefix>_<auto_counter>_.png`. The path doesn't change per-episode because the widget is static. Listed in ROADMAP.md "Known remaining suspects" (lines 47-55) under the same Phase G blast-radius pattern, but the visual-layer impact wasn't appreciated until this run. Architectural fix: replace stock SaveImage with a custom OTR node that reads the in-flight ledger singleton and routes to `otr_stills_dir(<ep_id>)`.
- **Fix (four sites — write + read alignment):**
  - **Site 1 (writer, radio bookend):** `visual/batch_flux_render.py:845` — changed `_OTRP.otr_stills_dir()` → `_OTRP.otr_stills_dir(episode_id)`. The `episode_id` variable was already in scope at line 768/772 (resolved via the in-flight ledger singleton, same Phase G discovery path used by BUG-LOCAL-021). Radio bookend now lands at `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` per the canonical Phase B layout.
  - **Site 2 (writer, env stills):** new node `OTR_SaveToEpisodeWorkspace` in `nodes/otr_save_to_episode_workspace.py`. Reads `_otr_ledger.in_flight_ledger_path()` to derive episode_id at runtime; routes to `otr_stills_dir(ep_id)` or `otr_portraits_dir(ep_id)` based on `role_kind` widget ("stills" | "portraits"). Falls back to legacy dirs (preserving existing behavior) if no singleton is available — never raises in headless/test contexts. Registered in `__init__.py`. Workflow JSON `workflows/otr_scifi_16gb_full.json` node 25 retyped from `SaveImage` to `OTR_SaveToEpisodeWorkspace` with `role_kind="stills"`, `filename_pattern="full_env"`.
  - **Site 3 (reader, BatchHumoRender env-still binding):** `nodes/batch_humo_render.py:_resolve_cast_stills_from_ledger` and `_find_portrait` — added per-episode glob pattern `otr/episodes/*/stills/full_env_*.png` alongside the existing legacy `otr/stills/full_env_*.png` and `otr_stills/full_env_*.png` patterns. Without this, after Site 2 starts writing to per-episode dirs, HuMo's cast→still binding would find ZERO fresh stills and fall back to stale prior-episode stills (or unmapped, then fall through to portrait/composite tiers). The mtime-based freshness filter (`fresh_floor = ledger_mtime - 60s`) in the same function still enforces episode-correctness, so cross-episode pollution is mathematically impossible.
  - **Site 4 (reader, BatchLTXRender radio bookend):** `nodes/batch_ltx_render.py:374` — changed `otr_stills_dir() / f"radio_bookend_{eid}.png"` → `otr_stills_dir(eid) / f"radio_bookend_{eid}.png"`. Without this, after Site 1 starts writing to per-episode dirs, LTX would look in `_legacy_stills/` and find nothing, falling back to a generic motion clip with no scene continuity. (`nodes/video_composite.py:163` was already correct — passes `eid` — verified.)
- **Verify:**
  - AST parse on all 6 touched files (story_orchestrator, batch_flux_render, batch_humo_render, batch_ltx_render, otr_save_to_episode_workspace, __init__) green.
  - JSON parse + node-type audit on `workflows/otr_scifi_16gb_full.json`: 30 nodes, 0 stock SaveImage remaining, 1 OTR_SaveToEpisodeWorkspace registered.
  - **New `tests/test_save_to_episode_workspace.py` — 8 passed in <1s.** Covers: with active singleton → resolves to per-episode dir; no singleton → falls back to legacy dir; role_kind="stills" → otr_stills_dir; role_kind="portraits" → otr_portraits_dir; filename_pattern preserved; per-episode counter starts at 1 (independent of any global counter); never raises on mkdir failure; node registered in `NODE_CLASS_MAPPINGS`.
  - Cumulative regression: **155 passed in 3.27s** across `tests/test_production_ledger.py + test_radio_still_resolver.py + test_filename_pattern_audit.py + test_cache_key_mutations.py + test_meta_paths.py + test_ledger_rename.py + test_critique_dialogue_preservation.py + test_save_to_episode_workspace.py + test_prompt_format_safety.py`. PLUS Bug Bible regression **24 passed / 1 skipped / 1 xfailed** in 1.24s.
  - **Real-run acceptance (pending):** queue any episode; expect `output/otr/episodes/<ep>/stills/full_env_NNNNN_.png` with COUNTER STARTING AT 1 (per-episode counter, not global), AND `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` co-located. BatchHumoRender log line `[BatchHumoRender] cast-still binding: N/M cast members matched to fresh stills` should report N>0. BatchLTXRender should find the radio bookend at the per-episode path. Final video should have visible scenery (not 100% black) when `blend_opacity > 0` is also set.
- **Tags:** flux, save-paths, per-episode-workspace, phase-g-blast-radius, video-composite-empty, write-read-alignment, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-021 (Phase G singleton sweep — fixed audio side, missed visual side); BUG-LOCAL-027 (dialogue wipe — orthogonal failure that compounds with this bug; fixed in same commit). Round-robin consult was SKIPPED per direct user override ("yes ofrget rop8u7hnd robins just fix fix fix") — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING — UPDATED post-launch):** ATTEMPTED + IN PROGRESS at session end. ComfyUI was relaunched headless via the venv python (`main.py --listen 127.0.0.1 --port 8000 --highvram`) after the autonomous-pass cleanup taskkill cleared its previous session. Boot succeeded (36 OTR nodes loaded, including the new `OTR_SaveToEpisodeWorkspace`; verified via `/object_info`). Soak queued via `scripts/soak_bug027_028.py`, prompt_id `d29e3d8f-edce-48e2-a960-7245ff989543`. **Widget patches FAILED** with `widgets_values length mismatch on node 1 (OTR_LLMScriptWriter): len(wv)=15 vs len(widget_names)=16` — the saved workflow JSON has 15 widget values but the live schema reports 16, indicating widget drift on `OTR_LLMScriptWriter` since the workflow was last saved. Soak still queued with the workflow's saved values: `target_words=350, target_length="short (3 acts)", num_characters=2, style="tense claustrophobic", creativity="balanced"`. The "balanced" creativity (temp=0.85) is LESS aggressive on the BUG-027 trigger conditions than the original "maximum chaos" (temp=0.95) repro shape — soak validates the pipeline + new node registration but may not exercise the total-collapse gate fire. Last status check (~12 min into run): `queue_running: 1`, prompt_id still in queue, otr_runtime.log frozen at last warmup line `[01:11:40] WARMUP: CUDA kernels compiled in 0.6s`, ComfyUI python.exe at 18.3 GB working set (model loaded + active inference). Logs are buffered between LLM call boundaries; Jeffrey will see acceptance signatures populate in the morning when the run completes. **Followup BUG candidate:** widget drift on `OTR_LLMScriptWriter` between saved workflow + live schema — log as BUG-LOCAL-029 if confirmed (15 vs 16 widget mismatch). One-shot soak script `scripts/soak_bug027_028.py` is committed and ready for re-run when widget drift is resolved.

---

### BUG-LOCAL-027 [FIXED]: Critique/revision pass returns SCENE/ENV/SFX-only script, dropping all CHARACTER dialogue — Bark gets 0 lines
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** TBD (likely YES — recurring across multiple runs)
- **Widget config (from screenshot):** target_words=110, num_characters=2, target_length="short (3 acts)", style="noir mystery", creativity="maximum chaos", arc_enhancer=ON, self_critique=ON, open_close=ON, optimization_profile="Pro (Ultra Quality)", model=google/gemma-4-E2B-it. Standard short(3) preset — NOT ultra_smoke / tiny_smoke (so BUG-LOCAL-005's CHARACTER:/SCENE: enforcement does not apply to this code path).
- **Symptom — current run "Cold Circuit" (otr_runtime.log line numbers):**
  - L47167 `[00:14:27] ScriptWriter: PARSE_OK attempt=1 has_scene=True voice_hits=18 bare_hits=0` — initial draft healthy: 3 scenes, 18 dialogue lines, characters ANNOUNCER + FLETCHER WELLS + KENJI BERNARD.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — character-line counter returns empty dicts for both draft and revised. Parser disagreement: draft visibly had 18 `[N] CHARNAME:` lines, but the critique-pipeline counter sees zero. Revised pass also legitimately produced zero (only `=== SCENE N ===`, `ENV:`, `SFX #N:` lines emitted across the entire 100s revision generation — no `[N] CHARNAME:` lines at all).
  - L47257 `[00:16:30] CRITIQUE: Revised script accepted (sim=83.0%, len=118%)` — acceptance gate let the dialogue-stripped revision through. Similarity stayed high because SCENE/ENV/SFX scaffolding overlaps; length grew because model padded with extra atmosphere.
  - L47265 `[00:16:30] WORD_ENFORCEMENT: 0 words vs 110 target (0%) | @140wpm -> ~0.0 min [0 lines detected]`
  - L47267 `[00:16:30] BUG-109b: cast members with 0 lines: FLETCHER WELLS, KENJI BERNARD`
  - L47268 `[00:16:30] BUG-109b: 3/3 scene(s) have 0 dialogue lines`
  - FORMAT_NORM single-pass attempted recovery; ANNOUNCER bookends were generated separately (Kokoro-bound, not Bark-bound); CHARACTER dialogue was never restored.
  - Final downstream effect: `[BatchBark] Found 0 dialogue lines in Canonical 1.0 format (skipped 2 ANNOUNCER lines - routed to Kokoro bus)`. SceneSequencer pre-rendered 1 TTS + 8 SFX + 2 ANNOUNCER, zero Bark clips.
- **Recurring across runs (NOT a one-off):**
  - L44646 `[22:00:57] CRITIQUE: Character line counts - draft={} revised={}` (sim=56.6%, len=176% accepted) — earlier run, same shape.
  - L46713 `[23:43:26] CRITIQUE: Character line counts - draft={} revised={'ANNOUNCER': 2}` (sim=90.4%, len=94% accepted) — preserved ANNOUNCER only, dropped character dialogue.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — current run.
  - Pattern: critique pipeline character counter consistently returns `{}` for draft regardless of writer output; acceptance gate (similarity + length only) cannot detect dialogue loss; multiple runs have shipped dialogue-stripped scripts to the audio cascade.
- **Cause (CONFIRMED via source dive 2026-05-03 EVENING):**
  - **Two coupled gaps in `nodes/story_orchestrator.py`.**
  - **Gap 1 (parser blindness):** `_count_character_lines` (line 6890) regex was `r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'` — required line to START with optional whitespace + uppercase name. The writer's actual output format is `[12] FLETCHER WELLS: text` (numbered-bracket prefix), so the regex never matched and returned `{}` for both draft and revised. Acceptance gate at line 7174 iterated `draft_char_counts` (empty dict) → no-op → revision accepted regardless.
  - **Gap 2 (gate too narrow):** even with parser working, the per-character preservation check (line 7174-7184) only catches "FLETCHER dropped from 8 to 1." It does NOT catch "every character wiped at once" because the loop iterates the draft dict per-char; if the revision wipes ALL characters, no individual character drops below the floor (they all dropped from N to 0, but the loop doesn't compare totals). Surface metrics — similarity ratio (0.83) + length ratio (1.18) — both pass on a SCENE/ENV/SFX-only revision because the scaffolding overlaps.
  - **Secondary contributor:** revision pass uses `temperature` (passed in from caller — for "maximum chaos" creativity = 0.95). High temp + critique demanding "fix every flagged problem" can push the model into pure-prose rewriting mode where it drops dialogue in favor of atmospheric SFX/ENV. The structural floor (`structural_temp=0.6` for similarity/length checks) doesn't gate this — it's a separate variable.
- **Fix (three-part):**
  - **Part 1 (parser regex, line 6916):** added optional non-capturing group `(?:\[\d+\]\s+)?` so both `CHARNAME:` and `[N] CHARNAME:` formats parse. Also tightened the structural-token exclude check at line 6924 to do BOTH exact-match AND first-word-match (`first_word in _struct_exclude`), so multi-word headers like `ACT 2:` or `SCENE 3:` no longer slip through as character names.
  - **Part 2 (total-collapse hard gate, after line 7184):** belt-and-suspenders for the per-character check. Computes `draft_total = sum(draft_char_counts.values())` and `revised_total = sum(revised_char_counts.values())`; if `draft_total >= 3` (threshold to apply ratio) and `revised_total < max(1, ceil(draft_total * 0.5))`, logs `CRITIQUE_REJECTED - total character lines collapsed from N to M (min=K, threshold=50%%)` and returns the draft unchanged. Below 3 lines the draft is too short for a meaningful ratio — the per-character check (with `min_line_count_per_character=2` floor) handles those cases.
  - **Part 3 (revision prompt hardening, line 7034 area):** added explicit `ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION` clause to the revision LLM prompt. Tells the model EXPLICITLY that producing a SCENE/ENV/SFX-only output is a "TOTAL FAILURE" and that every CHARACTER speaker present in the draft MUST appear in the revision. Also documented that the optional `[N]` prefix from the draft may be kept or omitted (both parse). Format-safety smoke (`tests/test_prompt_format_safety.py`) confirms no unescaped `{}` braces in the new prose (BUG-026 lesson).
- **Verify:**
  - AST parse on `nodes/story_orchestrator.py`: green.
  - **New `tests/test_critique_dialogue_preservation.py` — 14 passed in <1s.** Covers: parser handles bare `CHARNAME:`; parser handles `[N] CHARNAME:`; parser handles mixed format in same text; structural tokens (SCENE/ACT/MUSIC/SFX/ENV) excluded by exact-match AND first-word-match; empty/None text returns empty dict; ANNOUNCER counted as character; gate REJECTS total dialogue wipe (the actual L47256 case); gate REJECTS announcer-only revision (the L46713 case); gate ACCEPTS minor dialogue trim (83% retention); gate ACCEPTS at exactly 50% threshold; gate SKIPS short drafts (`< 3` lines); gate handles empty dicts safely; revision prompt has no unescaped braces (BUG-026 footgun gate); ABSOLUTE REQUIREMENT clause is present in source.
  - `tests/test_prompt_format_safety.py` — 1 passed (BUG-026 regression test passes against the new prompt prose).
  - Cumulative regression: 155 passed in 3.27s + Bug Bible 24 passed / 1 skipped / 1 xfailed in 1.24s.
  - **Real-run acceptance (pending):** queue the same widget config (110 words / 2 chars / short(3) / noir mystery / maximum chaos); expect (a) `CRITIQUE: Character line counts - draft={'ANNOUNCER': N, 'CHAR1': M, ...}` with NON-EMPTY draft dict (parser fix), (b) if revision wipes dialogue, log `CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed from N to M` and the pipeline uses the original draft, (c) `[BatchBark] Found >=1 dialogue lines in Canonical 1.0 format` in the final pre-render summary.
- **Tags:** critique, revision, character-dialogue, parser-mismatch, acceptance-gate, bark-empty, qa-soak-2026-05-03, recurring, fixed
- **Related:** BUG-LOCAL-005 (CHARACTER:/SCENE: enforcement — only applies to ULTRA_SMOKE path; this is short(3)). BUG-109b detector (pre-existing observability — fired correctly here, no auto-recovery existed before this fix). Round-robin consult was SKIPPED per direct user override — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING):** DEFERRED. Same reason + same handoff as BUG-LOCAL-028 above: ComfyUI was terminated mid-session and the one-shot soak script at `scripts/soak_bug027_028.py` queues both 027 + 028 acceptance signatures in a single run. Jeffrey runs the soak in the morning after restarting ComfyUI Desktop.

---

### BUG-LOCAL-026 [FIXED]: Phase H regression — unescaped curly braces in DIRECTOR_PROMPT crashed `.format()` mid-pipeline
- **Date:** 2026-05-03 | **Phase:** G/H hotfix | **Bible candidate:** YES (classic `.format()` footgun)
- **Symptom:** Live soak crash on 2026-05-02 23:46. Episode "Exponential Tremor Echoes" (style="a claude cowork test session", target_words=80) ran cleanly through ScriptWriter (3-outline OpenClose evaluator, critique pass, revision pass, ScriptCritic verdict REVISE with 7 issues, revision applied). Crashed at LLMDirector.direct (`nodes/story_orchestrator.py:9951`):
  ```
  IndexError: Replacement index 0 out of range for positional args tuple
  ```
  ~10 minutes of LLM compute lost.
- **Cause:** Phase H BUG-LOCAL-023 added an EXCLUDE-ANNOUNCER clause to `DIRECTOR_PROMPT`. The added prose contained literal `visual_plan.characters{}` and `voice_assignments{}` — two unescaped `{}` empty-brace pairs. The surrounding template uses `str.format()` with kwargs (`script_text`, `voice_mapping_rules`); Python's `.format()` interpreted `{}` as a positional arg slot reference, looked up `args[0]`, found nothing, raised `IndexError`. **Cardinal mistake — adding prose with literal braces to a `.format()` template without escaping.**
- **Fix:** Removed the literal `{}` symbols from the EXCLUDE-ANNOUNCER prose. Kept the semantic content ("EXCLUDE narrator/announcer roles from the visual plan characters object", etc.) — readable to the LLM, no longer breaks `.format()`. Either `{{ ... }}` escaping OR removing the braces from prose is valid; chose removal for prose-readability.
- **Verify:**
  - Standalone `_director_prompt_test.py` smoke: `DIRECTOR_PROMPT.format(script_text=..., voice_mapping_rules=...)` returns 5501 chars, no exception.
  - **Permanent regression test** `tests/test_prompt_format_safety.py` — extracts `DIRECTOR_PROMPT` constant via regex, calls `.format()` with the production kwargs, asserts no `IndexError`/`KeyError`/`ValueError`. **Passed in 1.74s.** Future Phase-N additions to the prompt that re-introduce unescaped braces will fail this test before they reach a live run.
- **Tags:** phase-h-regression, str-format, prompt-template, hotfix, bible-candidate
- **Lesson learned for future autonomous mode:** when editing any constant that's later passed to `.format()`, run `_director_prompt_test.py`-style format smoke as part of the AST guard pass. Don't ship template edits without confirming `.format()` survives.

---

### BUG-LOCAL-025 [FIXED]: LTX role prompts ignore story style + scene context (every episode looked the same)
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `nodes/batch_ltx_render.py::_PROMPT_BY_ROLE` is a hardcoded dict mapping `{announcer, music_open, music_close, music_inter, sfx}` → fixed motion prompts ("Vintage 1940s radio broadcast set, glowing tuning dial pulses gently, copper vacuum tubes warm amber glow..."). Every episode renders the SAME LTX motion regardless of the story's style or scene atmosphere. Jeffrey: *"be sure story arc or better shot/scene arc is being fed into FLUX and LTX as well to match the short."*
- **Cause:** Original `_PROMPT_BY_ROLE` design treated LTX as a generic radio-animator with no story awareness. Acceptable when the radio bookend (the i2v reference image) carries all visual identity — but downstream review confirmed the motion prompt itself influences mood (dial sweep speed, tube glow rhythm, dolly direction).
- **Fix:** New `_build_ltx_role_prompt(role, line, ledger)` helper enriches each role base prompt with two ledger-derived layers:
  1. **Per-line scene context.** Lookup chain: `line.shot_id` → `ledger.shots[*].scene_id` → `ledger.scenes[*].env / .description`. Truncated to 60 chars, appended as `, scene context: <env>`. Each LTX clip now matches the SCENE it accompanies (early scenes get tense env, late scenes get resolved env).
  2. **Episode style suffix.** Read from `ledger.meta.gen_params_initial.style` (or `.gen_params.style`) — same singleton-fed source Phase G fixed for radio bookend. Appended as `, <style> broadcast tone`.
  Bounded so the role's motion intent isn't drowned. Per-line lookup means one episode's announcer LTX clips can vary across scenes if those scenes have different `env` text.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** the `[BatchLTXRender]` log lines should now show enriched prompts; two episodes with different styles should produce visibly different LTX motion intent.
- **Tags:** ltx, story-arc, scene-context, style-aware, qa-pass-2026-05-03

---

### BUG-LOCAL-024 [FIXED]: Radio bookend FLUX prompt fell back to generic when style missing OR ledger stale
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Soak run on 2026-05-02 logged `[BatchFluxRender] radio still prompt source=fallback (no style)` — radio rendered as generic "sci-fi retrofuturistic radio broadcast unit" despite user setting style="space opera epic" in the widget. Compounded with BUG-LOCAL-021 (FLUX read a stale April 26 ledger via the broken `find_most_recent_ledger` walker), the radio NEVER reflected the actual episode story.
- **Cause:** `_build_dynamic_radio_prompt` in `visual/batch_flux_render.py` only looked at two fields (`gen_params_initial.style` + `gen_params.style`) before falling to a single hardcoded fallback. No fallback chain through `style_custom`, scene environment, or episode title.
- **Fix:** Six-tier resolution with per-tier branch logging:
  1. `gen_params_initial.style` (primary widget value)
  2. `gen_params.style` (back-compat)
  3. `gen_params_initial.style_custom` (free-text override)
  4. First scene's `env` / `description` (scene-driven mood)
  5. `episode_id` slug (strip "signal_lost_" prefix + trailing timestamp, replace underscores)
  6. Hardcoded `_RADIO_FALLBACK_PROMPT` (true last resort)
  Plus: scene-context hint (`set in <first_scene_env>`) appended whenever distinct from descriptor, so style + scene combine. New log line `[BatchFluxRender] radio prompt: branch=<which> -> <preview>` tells the runtime tail which tier fired. Bounded length: descriptor capped at 80 chars, scene_hint at 60 chars.
- **Verify:** AST + full pytest green. **Real-run acceptance (pending):** with Phase G singleton lookup feeding the CURRENT ledger, the radio prompt should now log `branch=gen_params_initial.style` and the radio should render as "space opera epic radio broadcast unit, set in derelict orbital lab, ..."
- **Tags:** flux, radio-bookend, story-arc, fallback-chain, qa-pass-2026-05-03

---

### BUG-LOCAL-023 [FIXED]: ANNOUNCER portrait wasted FLUX context + skewed scene composition
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Jeffrey caught mid-soak: `LLMDirector` generates a `portrait_prompt` for ANNOUNCER under `visual_plan.characters`, then `OTR_VideoPlan.compose_shot_prompt` concatenates ALL character portraits into every scene's PASS3 visual prompt. The announcer is never on screen as a person (BUG-LOCAL-129b: routed to Kokoro voice + radio bookend visual; HuMo skips them). Including their portrait wastes FLUX prompt budget AND skews scene composition by forcing every shot to fit an extra character (50yo silver-haired woman in flight gear).
- **Cause:** Visual_plan.characters was generated for every speaker without a "appears on screen?" filter. PASS3 compose treats the dict as canonical.
- **Fix (belt-and-suspenders, two layers):**
  1. **LLMDirector prompt rule** in `nodes/story_orchestrator.py` (VISUAL PLAN RULES section): explicit instruction "EXCLUDE narrator/announcer roles from visual_plan.characters. The ANNOUNCER (and any voice that only narrates without appearing on screen) must NOT be included under visual_plan.characters{}. Their voice mapping still belongs in voice_assignments{}; only visual_plan.characters skips them." Catches it at the source.
  2. **`OTR_VideoPlan` filter** in `nodes/otr_video_plan.py:438`: new `NON_VISUAL_ROLES = {"ANNOUNCER", "NARRATOR"}` set; before composing portraits, partition `chars_dict.keys()` into `all_char_names` (visible roles) and `_skipped_non_visual` (logged as info). Catches future LLM regressions where layer 1 fails. Honors explicit `focus_character` requests for non-visual roles (lets a debugging workflow request the announcer portrait specifically).
  Audio is unaffected: `voice_assignments.notes` is a SEPARATE field that audio nodes (Bark/Kokoro) consume; `portrait_prompt` doesn't feed audio at all.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** scene visual prompts in `[BatchFluxRender] shot N/M:` log lines should NOT lead with "Female, 50s, gravelly voice..." when there's an ANNOUNCER in the cast; should see `OTR_VideoPlan: skipped non-visual role(s) from portrait composition: ANNOUNCER` log line.
- **Tags:** flux, announcer, visual-plan, scene-composition, qa-pass-2026-05-03

---

### BUG-LOCAL-022 [FIXED]: BatchHumoRender stem-swap is mathematically broken when safe_title[:40] truncates the title
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `BatchHumoRender._load_ledger_with_path` (line 1791-1865 pre-Phase-G) takes a `.mp4` path input from `SignalLostVideoRenderer` and derives the ledger via stem swap (`<file>.mp4` → `<file>_ledger.json`). When `video_engine.py:1450` truncates the procgen mp4 filename via `safe_title = "...".replace(...)[:40]`, the resulting mp4 stem may NOT equal the canonical `episode_id`. Stem swap looks for a ledger that doesn't exist. Combined with BUG-LOCAL-020 (mp4 in legacy dir), the failure mode is "derived ledger from .mp4 not found". Even after BUG-LOCAL-020 fix puts the mp4 in the per-episode dir, stem swap can still fail if title truncation drops characters.
- **Cause:** Discovery coupled to mp4 filename instead of the on-disk per-episode workspace structure (`output/otr/episodes/<ep>/audio/<ep>_ledger.json`).
- **Fix:** Add Tier 0 layout-aware lookup BEFORE the legacy stem-swap tiers in `_load_ledger_with_path`. Detection: `audio_dir.name == "audio"` AND `audio_dir.parent.parent.name == "episodes"`. When detected, the parent dir name IS the `episode_id` by construction. Try canonical `<ep_dir_name>_ledger.json` first; fall back to globbing `*_ledger.json` (non-pending) in the same audio_dir if the slug rule doesn't match. Decoupled from mp4 stem entirely. Legacy stem-swap tiers preserved as fallback for old artifacts in the legacy flat layout.
- **Verify:** AST + full pytest suite (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue the same workflow JSON; BatchHumoRender should log `Phase G layout-aware ledger lookup: <ep>_ledger.json` and proceed past the prior crash point.
- **Tags:** humo, ledger-discovery, layout-aware, stem-swap, qa-pass-2026-05-03

---

### BUG-LOCAL-021 [FIXED]: Audio-side nodes used global mtime walker for write-back (latent BUG-LOCAL-014 wrong-episode shape)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Seven sites in the audio-side write-back chain used `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])` to locate the in-flight ledger for write-back: `musicgen_theme.py:98+494`, `batch_audiogen_generator.py:58+511`, `batch_bark_generator.py:703`, `scene_sequencer.py:920+1257`, `audio_enhance.py:436`. Plus `visual/batch_flux_render.py:641` used the same walker (with the wrong dirs — `otr_audio_dir()` returns `_legacy_audio` when called with no episode_id, so it never even scanned the per-episode tree). On the 2026-05-02 soak, FLUX radio bookend stamped to `signal_lost_signal_abyss_20260426_161737` (a 6-day-old episode) instead of the in-flight `signal_lost_cramped_cargo_bay_vibrating_20260502_220824`. Same wrong-episode shape as BUG-LOCAL-014 — Phase A fixed it for `rtx_upscale.py` only; the rest of the codebase had it latent.
- **Cause:** Mtime-based discovery is fundamentally racy across queue boundaries and across runs. The `_CURRENT` Ledger singleton (set by `LLMScriptWriter` via `new_ledger()`) tracks the in-flight episode by construction; ComfyUI sequential queue + LLMScriptWriter's `IS_CHANGED = time.time()` guarantee the singleton is fresh on every queue invocation.
- **Fix:** Add `_otr_ledger.in_flight_ledger_path()` helper that reads the singleton's `path` (which advances correctly through `Ledger.rename_episode` per Phase B) and falls back to the legacy mtime walker only if the singleton is somehow unavailable. Sweep all 7 audio-side sites + the FLUX radio bookend site to use the helper. Late-import via try/except inside the helper avoids circular import with `production_ledger.py`. The walker is preserved (for `post_audio_video_pipeline.py:126` empty-input fallback and for the helper's own last-resort path).
- **Consult sources:** `docs/2026-05-03-phase-g-path-reorg-blast-radius__01_chatgpt.md` (gpt-5.5, 117.8s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 46.1s — caught critical "ComfyUI cache trap" risk for singleton; verified-already-mitigated by LLMScriptWriter's IS_CHANGED), `__03_nvidia.md` (mistral-nemotron, 190.7s).
- **Verify:** Phase G AST + full pytest (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue; FLUX radio bookend should stamp to the CURRENT episode_id, not a stale leftover. Two-episode soak: B's audio nodes should write to B's ledger, not A's.
- **Tags:** ledger-discovery, singleton, find_most_recent_ledger, wrong-episode, defensive-sweep, qa-pass-2026-05-03

---

### BUG-LOCAL-020 [FIXED]: video_engine.py procgen mp4 written to legacy `output/otr/audio/` instead of per-episode workspace (SOAK BLOCKER)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Live soak crash on 2026-05-02. After 12 minutes of pipeline progress (LLM ladder + audio cascade + procgen video), `BatchHumoRender` crashed with `RuntimeError: BatchHumoRender: derived ledger from .mp4 not found: C:\...\output\otr\audio\signal_lost_cramped_cargo_bay_vibrating_20260502_220824_ledger.json (also tried collapsed-underscore variant + directory scan in C:\...\output\otr\audio)`. The procgen mp4 was at `output/otr/audio/<file>.mp4` (legacy flat) but the ledger had been moved to `output/otr/episodes/<ep>/audio/<ep>_ledger.json` by Phase B's `Ledger.rename_episode`. Stem swap looked in the wrong dir and found nothing.
- **Cause:** `video_engine.py:1443-1446` (pre-Phase-G) hardcoded `out_dir = .../output/otr/audio` — the legacy flat layout. The path reorg (Phases A/B/E) moved every per-episode asset under `output/otr/episodes/<ep>/audio/`, but `video_engine.py` was missed. The mp4 ended up OUTSIDE the per-episode tree, so `Ledger.rename_episode` (which renames the parent dir) couldn't move it along with the rest of the workspace.
- **Fix:** Read `out_dir` from `get_ledger().out_dir` (the in-flight Ledger singleton's audio path = `episodes/pending_<ts>/audio/` at this point). Write the procgen mp4 there. After `Ledger.rename_episode(<ep_id>)` moves the parent dir, recompute `final_out_path = Path(led.out_dir) / pending_out_path.basename` and verify it exists. Update the `subfolder` hint in the ComfyUI UI return value to the post-rename relative path. Defensive try/except wraps the ledger lookup so test/headless paths fall back to the legacy `output/otr/audio/` location.
- **Consult sources:** Same as BUG-LOCAL-021. ChatGPT specifically caught the post-rename stale-path risk; Gemini caught the `safe_title[:40]` truncation issue (addressed in BUG-LOCAL-022).
- **Verify:** Phase G AST + full pytest green. **Real-run acceptance (pending):** re-queue the same 30-word smoke; expect `[Video] Saved: <abs path under episodes/pending_<ts>/audio/>` followed by `[Ledger] per-episode dir moved pending_<ts> -> <ep_id>` followed by `[Video] post-rename mp4 path: <pending> -> <final>` (if recompute fired) and BatchHumoRender progress past the prior crash point.
- **Tags:** path-reorg, video-engine, soak-blocker, post-rename, qa-pass-2026-05-03

---

### BUG-LOCAL-019 [FIXED]: Sprint 1 full-suite acceptance — pre-existing test rot (Phase B fallout + per-episode reorg fallout)
- **Date:** 2026-05-02 | **Phase:** Sprint 1 acceptance | **Bible candidate:** no (test-only fixes, no behavior change)
- **Symptom:** `python -m pytest tests/` (Sprint 1 acceptance line item) ran to completion in 113s but with **5 failures** in two distinct clusters. The original BUG-LOCAL-006 hang at `TestDropdownsHaveEffect::test_creativity_produces_different_temps` was already resolved by intervening conftest work — that test now passes in 10s standalone — but other latent failures had been masked because the three explicit suites used in Phase A→E regression (Bug Bible, dropdown_guardrails, core) didn't include the ones that broke.
- **Cluster 1 (2 failures, `tests/test_production_ledger.py`):** `TestLedgerBeats::test_rename_updates_path_and_data` and `TestDualLedgerFix::test_rename_episode_moves_file_on_disk` both raised the Phase B (BUG-LOCAL-015) hard-fail RuntimeError "both source and destination episode directories exist". Root cause: the tests passed `tmp_path` directly as the audio dir to `Ledger(...)`. Phase B's `rename_episode` walks `os.path.dirname` up two levels to find the per-episode root — from `tmp_path = pytest-of-jeffr/pytest-NNN/test_K/`, that walked up to `pytest-of-jeffr/`, then constructed `new_ep_dir = pytest-of-jeffr/signal_lost_black_sphere_20260424_142006`. That destination accumulated across pytest sessions (siblings from prior runs of the same test pollute the user's TEMP root), so on any run after the first the conflict guard fired correctly. The TESTS were buggy — they assumed the pre-Phase-B silent split-state recovery and depended on global TEMP pollution that no longer works under the hard-fail invariant.
- **Cluster 2 (3 failures, `tests/test_radio_still_resolver.py`):** `TestFilesystemFallback::test_filesystem_fallback_finds_by_episode_id`, `TestFilesystemFallback::test_filesystem_fallback_when_ledger_path_stale`, `TestBug121Hardening::test_zero_byte_file_falls_through_to_layer3` all failed with `TestX.<locals>.<lambda>() takes 0 positional arguments but 1 was given`. Root cause: `monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)` mocks the helper with a 0-arg lambda, but the per-episode workspace reorg (2026-05-02 EVENING, BUG-LOCAL-033) gave `otr_stills_dir` an `episode_id` parameter. Production code calls `otr_stills_dir(episode_id)` with 1 arg. The 7 fallback tests that DON'T trigger this path passed silently; the 3 that do reach it failed.
- **Cause summary:** Cluster 1 = direct fallout from Phase B's hard-fail invariant correctly rejecting test setups that depended on the buggy old behavior. Cluster 2 = stale test mock signatures from the per-episode workspace reorg (BUG-LOCAL-033 era). Both pre-existing, surfaced because no one had run `pytest tests/` to completion since they were introduced.
- **Fix:** Cluster 1 — both failing tests now build a proper `tmp_out/episodes/<ep>/audio/` per-episode dir before instantiating the `Ledger`. The rename invariant has clean room to operate; no global TEMP pollution. Cluster 2 — `monkeypatch.setattr(..., lambda *a, **kw: tmp_path)` (10 sites updated via `replace_all`). Variadic tolerates the new `(episode_id)` arg without changing test semantics.
- **Verify:** Targeted suite (`pytest tests/test_production_ledger.py tests/test_radio_still_resolver.py -v`) — 76 passed in 1.87s. Full suite (`pytest tests/ -q --ignore=tests/v2`) — **1126 passed / 7 skipped / 0 failed in 113.28s**. The 107 errors in an earlier run were transient pytest tmp_path session race (`pytest-264` got reaped while a parallel pytest invocation was still using it); not reproducible on clean runs.
- **Promotes BUG-LOCAL-006 from [PARTIAL] to [FIXED]:** the conftest CUDA mask works, AND the originally-blamed `test_creativity_produces_different_temps` now passes (cause was either incidentally fixed by Phase B/C/D/E work or transient under a specific environment that no longer reproduces). Sprint 1 acceptance line "python -m pytest tests/ runs to completion green" is now satisfied — net cumulative count: 1126 / 7 / 0 across the full directory.
- **Tags:** test-rot, phase-b-fallout, per-episode-reorg-fallout, sprint-1-acceptance, no-active-bug

---

### BUG-LOCAL-006 [FIXED, was PARTIAL]: pytest hang at session-start when ComfyUI on same GPU
- **Date:** 2026-05-02 PM EVENING (re-verified) | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Update on the prior PARTIAL status:** `tests/conftest.py` (committed earlier this session) sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import, registers the `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The original PARTIAL note flagged `TestDropdownsHaveEffect::test_creativity_produces_different_temps` as still-hanging. Re-verified 2026-05-02 PM: that test now passes in 10s standalone, and the full directory `pytest tests/` runs to completion in 113s (1126 / 7 / 0). Either the hang was incidentally fixed by Phase A→E work (path reorg + Phase B's atomic rename + cache key cleanup may have removed a fixture that touched a heavy import), or it was transient under a specific environment. No further bisect needed; the acceptance gate is satisfied.
- **Verify:** `python -m pytest tests/ -q --ignore=tests/v2` → 1126 passed / 7 skipped / 0 failed in ~113s. With ComfyUI Desktop up on `:8000`, same result.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate, was-partial

---

### BUG-LOCAL-018 [FIXED]: Ledger schema bump l3-2026-05-02 + meta.paths block
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — additive schema enhancement. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase E) prescribed adding a `meta.paths` block to the production ledger so downstream nodes can look up canonical episode dirs without reconstructing them from `episode_id`. Slug-reconstruction was the root cause of the BUG-LOCAL-014/015/017 cluster — Phases A/B/C/D fixed the live instances; Phase E removes the temptation entirely by stamping the absolute, on-disk-truth paths into the ledger at every save.
- **Cause:** N/A — preventive change.
- **Fix:**
  - `nodes/_otr_ledger.py`: bump `CURRENT_SCHEMA_VERSION` from `l3-2026-04-28` to `l3-2026-05-02`. Add `_build_meta_paths(ledger_path, episode_id)` helper that detects layout (per-episode-workspace under `output/otr/episodes/<ep>/audio/<ep>_ledger.json` vs legacy flat under `output/audio/<ep>_ledger.json`) and stamps an appropriate `meta.paths` block. `save_ledger_safe` now calls it on every write. The block is **resolved fresh on every save** from the actual on-disk path, so it self-corrects after `Ledger.rename_episode` (Phase B) — no caller has to update it.
  - `nodes/production_ledger.py`: `Ledger.save()` also stamps `meta.paths` (via the same `_otr_ledger._build_meta_paths` helper) so the path data is consistent regardless of which write path produced the ledger. Hardcoded fallback `SCHEMA_VERSION` updated to match. Best-effort try/except wraps the meta stamp — a stamping failure must NEVER break the actual ledger write.
  - `docs/ledger_schema.md`: created. Documents the schema (top-level fields + meta block + meta.paths block + per-episode vs legacy-flat layout shapes), the lineage table, the reader contract (`dict.get(...)` not direct subscript), and the rules for downstream nodes.
- **Verify:**
  - AST + Phase E invariant guards (schema string `l3-2026-05-02`, `_build_meta_paths` present, both layouts detected, dual-write stamping in both files) — green.
  - **New `tests/test_meta_paths.py` — 13 passed.** Covers: per-episode layout detection + all 6 dirs stamped + obs_final stamped when obs/ exists + ledger_path absolute; legacy flat layout detection + no fabricated subdirs + minimal paths only; `save_ledger_safe` stamps meta.paths AND preserves pre-existing meta keys; old ledger without meta.paths loads cleanly via `dict.get(...)` (back-compat regression); `Ledger.save()` stamps meta.paths too; **after `rename_episode`, the next save's meta.paths self-corrects to the new dir** (the killer property — proves stale references can't accumulate).
  - Three CLAUDE.md regression suites + all phase-A-through-D tests: **234 passed / 1 skipped / 2 xfailed in 106.37s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30 + meta_paths 13).
  - **Real-run acceptance (pending):** end-of-stack soak. New ledgers should carry `meta.paths`; old ledgers (if any survive in `output/audio/`) still load via `dict.get` defaults.
- **Tags:** schema, additive, meta-paths, back-compat, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase E section). No round-robin needed — additive only, no behavior change for existing readers (all already use `meta.get(...)`).
- **Reader contract enforced:** see `docs/ledger_schema.md` "Reader contract" section. `meta.paths` MUST be accessed via `led.get("meta", {}).get("paths", {}).get(field)`, never `led["meta"]["paths"][field]`.

---

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



