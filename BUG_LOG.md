# OTR Bug Log

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Owner:** Jeffrey A. Brick
**Last entry:** BUG-LOCAL-258 (2026-05-23) [FIXED] -- The two most recent episode runs rendered audio + the Signal-Lost fallback video but NO Flux stills / HuMo clips / composite. Commit `56c552d` ("PRIORITY 3 quick win #1") converted `script_json` / `news_used` / `ledger_json` to `forceInput` sockets on nodes 12/20/23/59: it added `forceInput: True` in Python (correct) but ALSO deleted each input's `widgets_values` slot + `inputs[].widget` sub-key in `otr_scifi_16gb_full.json` (wrong). `forceInput` does not free a `widgets_values` slot; the deletion shifted every Flux widget left by one, so ComfyUI's prompt validator read `seed='randomize'`, `cfg='euler'`, `sampler_name='simple'` etc. and rejected the entire visual branch (outputs 56/52/55/59/58/25/51 ignored). Fix: surgical revert of the JSON + `test_workflow_canonical_baseline.py` to the pre-`56c552d` shape; the `forceInput` Python additions stay. See body `### BUG-LOCAL-258`.
**Prior entry:** BUG-LOCAL-257 (2026-05-23) [FIXED] -- The `OTR_LedgerScriptWriter` model dropdowns (`creative_writing_model` / `technical_model`) listed non-LLM checkpoints (`black-forest-labs/FLUX.1-dev`, `depth-anything/Depth-Anything-V2-Large-hf`, `diffusers/LTX-Video-0.9.0`) beneath the 4 curated story LLMs. Cause: `build_dropdown_choices` appends every on-disk non-curated repo from `scan_local_llm_cache`, which walks `HF_HOME/hub` with no model-type filter -- and `HF_HOME` consolidates LLM + diffusion + vision models into a single hub cache. Fix: new `_snapshot_is_causal_lm` gate in `nodes/_otr_model_catalog.py` admits a non-curated repo only when its `config.json` declares a `*ForCausalLM` architecture; diffusion pipelines (no root `config.json`) and vision models (non-causal architecture) are excluded, curated 4 stay exempt. +8 regression tests in `test_model_catalog_scan.py`; full `tests/` walk 2491 passed / 17 skipped / 0 failed. See body `### BUG-LOCAL-257`.
**Prior prior entry:** BUG-LOCAL-256 (2026-05-22) [FIXED] -- `detect_phantom_names` flagged component words of multi-word roster entries as phantom names. A full cast name (`GULLIVER REEVES`) or a multi-word key_term (`Big Bang`) whitelisted only the whole phrase, so the single-word + bigram passes still re-flagged `Gulliver`, `Big` and `Bang` individually (episode `signal_lost_temporal_mirror_echoes_20260522_150549` b007). Fixed by allowing a candidate whose uppercase form is a whole-word component of any roster entry, not just an exact entry; +4 regression tests in `tests/test_phase0_name_roster.py`. See body `### BUG-LOCAL-256`.
**Prior prior prior entry:** BUG-LOCAL-255 (2026-05-22) [FIXED] -- `override_announcer_close` matched the private `_speaker_role` key, absent from `led.data["lines"]` rows (which carry the public `speaker_role`), so `news_close_brief` was silently dropped every episode -- the announcer outro was generic composer text, not the journalistic close. Fixed by the announcer dedicated-pass build: `override_announcer_close` retired; the opening + closing announcer lines are now produced by dedicated `creative`-slot passes `compose_announcer_intro` / `compose_announcer_outro` in `_otr_line_composer.py`. The outro pass writes to the known final announcer `beat_id` via `patch_line_text` (a `line_id` match, not the broken `_speaker_role` lookup). New `tests/test_announcer_passes.py` (32 tests); full `tests/` walk 2488 passed / 22 skipped / 0 failed. See body `### BUG-LOCAL-255`.
**Prior batch:** BUG-LOCAL-251..254 (2026-05-21) -- full `tests/` walk 27-failure triage. 251 partial `folder_paths` stub poisons `sys.modules` (9 order-dependent contract-validator failures that passed in isolation); 252 `batch_humo_render.py` relative imports break standalone loads (BUG-249 missed these two sites); 253 `OTR_MusicGenTheme` workflow widget vector short by one (`episode_seed` slot missing); 254 the 5 `_bisect_flux_*.json` workflows carry slug `id`s the ComfyUI Zod validator rejects. 25 of the 27 failures fixed + verified; the remaining 2 quarantined as KNOWN-FAIL-007/008 (`google/gemma-4-E4B-it` default-workflow license gate), then both resolved 2026-05-21 by a Gemma-4 license re-audit -- the Gemma 4 family ships under Apache 2.0 (verified on Google's official HuggingFace model cards for E2B-it + E4B-it), not the old restricted Gemma Terms of Use the 2026-05-16 audit assumed; the two catalog rows + the `docs/model-license-google--gemma-4-e{2,4}b-it.md` audit files were corrected to `apache_2_0`/`mit_equivalent`, both KNOWN-FAILs promoted, quarantine set now empty (full `tests/` walk 2440 passed / 22 skipped / 0 failed). `EXPECTED_FAILED_NODEIDS` + `docs/known-failures.md` reconciled in lockstep. Full entries in the body below.
**Prior (FLUX investigation) entry:** BUG-LOCAL-244 (2026-05-19 11:10) -- **FLUX fast-path mechanism UNIDENTIFIED.** Split off from BUG-LOCAL-231 per Jeffrey 11:10 pushback #5: "do not close 231 until the fast-path is either reproduced+explained or formally split." Two outliers observed against the ~180 s/it slow-regime baseline established by batteries v1 + v2: (1) **42.38 s/it** today 2026-05-19 ~08:55 (battery v1 iter 1) with precheck + 1 sampler-poll tick captured at start (precheck logged `cudnn.benchmark=False`, poll tick 1 logged 15954 MB lhm + 618 MB D3D Shared + 2977 MHz clocks); (2) **1.22 s/it** yesterday 2026-05-18 ~23:30 with NO sampler-time telemetry (predates commit 3691317). Speed-up ratios vs ~180 s/it slow cluster: 4.3x (42.38) and 150x (1.22). The 4.3x and 150x differ by 35x -- likely DIFFERENT mechanisms, not the same warm-cache cause. **Verification gate:** reproduce ≥1 fast run with the full battery v3 telemetry (per-step timestamps + precheck + poller across ≥5 steps) AND identify which signal differs between fast and slow runs, OR mark UNREPRODUCIBLE if no fast run captured in next N batteries. Hypothesis seed: cudnn / cublas kernel cache hit from an immediately-prior identical inference (process-local for 42.38, possibly cross-process / persistent for 1.22) -- but `cudnn.benchmark=False` rules out the autotuner; the 4.3x speed-up may be from a different cache layer (workspace allocator? cudnn forward conv plan? fp8 dequant kernel SASS cache?). **NOT a fix-it bug** -- a phenomenon-investigation bug. Pending: battery v3 with per-step timing + config fingerprint, then targeted attempt to reproduce.
**Prior 1 entry:** BUG-LOCAL-243 (2026-05-19) -- Live dialogue-line visibility regression in LFC writer. The legacy `OTR_LLMScriptWriter` streamed each composed dialogue line to ComfyUI console as the LLM produced it -- operator could watch the script unfold in real time, catch bad output early (vocative drift, hallucinated names, tone misses), and tell whether the LLM was making progress vs hung. The current `OTR_LedgerScriptWriter` writes silently to the ledger file and only emits `[Ledger] saved pending_..._ledger.json (N lines, M words)` at the end of the entire writer phase. No mid-composition visibility. Confirmed by Jeffrey 2026-05-19: would have caught BUG-LOCAL-233 vocative drift IMMEDIATELY ("Hold tight there, GULLIVER REEVES" would have streamed to console; operator could kill the run) instead of finding it in the saved ledger after the fact. **Fix shape:** add `log.info("[LineComposer] line_id=%s speaker=%s text=%s", line_id, speaker, text)` (or equivalent INFO-level emission) after each successful `compose_line` return in `nodes/_otr_line_composer.py`. Cheap, additive, no behavior change. Pending until pipeline closes end-to-end per pipeline-closes-first rule.
**Prior 2 entry:** BUG-LOCAL-242 (2026-05-19) -- Word/char count audit too permissive on text-vs-count drift. Recent ledgers show b007 word=18 / actual text words=19 (the inserted GULLIVER REEVES vocative); b007 char=112 / actual=121; b015 word=24 / actual=23. The §6.G word_counts audit only logs WARNING on drift, never blocks. **Decision needed:** tighten audit OR formally relax it. Hygiene, low priority. Pending design.
**Prior 3 entry:** BUG-LOCAL-241 (2026-05-19) -- `freeze_verdict='needs_full_rerun'` ignored by downstream. Cascade explicitly signals the run shouldn't proceed but the pipeline continues to audio + FLUX phases anyway. **Design clarification needed:** is the verdict advisory or supposed to block? If supposed to block, current ignore-and-continue is a bug. If advisory, the verdict label is misleading. Pending design.
**Prior 4 entry:** BUG-LOCAL-240 (2026-05-19) -- Freeze cascade style-slug validator out of sync with style picker's intended LLM-invent-slug behavior. Per Jeffrey 2026-05-19, the "let the story decide" sentinel IS designed to invent new slugs based on the story content -- Pass 1 inventor producing slugs like `geosynchronous_orbital_rivalry`, `particle_song_revealed`, `psyche_craft_arrival`, `ten_year_secret_hunt`, `clinical_trial_cliffhanger` is by-design and is the feature Jeffrey wants. The bug is on the validator side: the freeze cascade checks strict `slug in KNOWN_STYLE_SLUGS` membership and rejects anything outside the 10-preset seed palette, treating LLM-invented slugs as "writer drift". The validator was never updated to accept the picker's intended output. Recurring hygiene noise; the cascade flags this as a critical error pre + post every run but does not block the pipeline (related: BUG-LOCAL-241 advisory-vs-blocking question). **[FIXED 2026-05-21]** `_otr_ledger_freeze.py::_check_meta_invariants` -- the `slug in KNOWN_STYLE_SLUGS` membership check is replaced by `_is_well_formed_style_slug` (regex `^[a-z]+(_[a-z]+)*$` + 64-char cap). Invented snake_case slugs from the picker now pass; only malformed slugs (uppercase, spaces, digits, runaway length) are flagged. **Cause:** the validator dated to S25/MG-6 (BUG-216) when `musicgen_theme` looked up a cue palette keyed by the slug; `musicgen_theme` stopped consuming the slug as a palette key at Path F (2026-05-18) and composes prompts from the meta brief directly, so an unknown-but-well-formed slug halts nothing downstream -- the membership check was guarding a dead consumer. The now-unused `KNOWN_STYLE_SLUGS` import was pruned from `_otr_ledger_freeze.py`. `tests/test_style_palette_drift.py` updated in lockstep (`rejects_unknown_slug` -> `rejects_malformed_slug`; new `accepts_invented_snake_case_slug`). **Verify:** full `tests/` walk 2441 passed / 22 skipped / 0 failed (2026-05-21). **Bible candidate:** no (validator-staleness hygiene fix).
**Prior 5 entry:** BUG-LOCAL-239 (2026-05-19) -- `music[]` and `clips[]` empty despite music_inter lines. Recent ledger lines b006 + b013 have `speaker_role: music_inter` but top-level `music[]` and `clips[]` arrays are `[]`. MusicGenTheme either didn't run for these or didn't write back to the per-line collections. **AUDIO PATH -- DO NOT TOUCH WITHOUT EMPIRICAL EVIDENCE** the interludes are missing from final .mp4 (the EpisodeAssembler log line "theme 32000 Hz -> 48000 Hz" suggests theme cues DID render but for credits, not necessarily for the music_inter line slots). Pending investigation.
**Prior 6 entry:** BUG-LOCAL-238 (2026-05-19) -- `beats[]` empty despite lines referencing beat_ids. Ledger schema gap: lines b001-b018 each carry a `beat_id: bNNN` but the top-level `beats[]` array is `[]`. Two hypotheses: (h1) the writer doesn't emit beat objects to a top-level array (regression vs schema-required behavior); (h2) the schema design is "lines reference beat_ids that don't get their own top-level row" (no bug, schema-as-designed). Pending investigation -- read OTR_LedgerScriptWriter's beat-emission code path.
**Prior 7 entry:** BUG-LOCAL-237 (2026-05-19) -- **CONDITIONAL on BUG-LOCAL-234 smoke result.** Cast `portrait_path` not stamped to ledger. Coupled to BUG-LOCAL-234: PortraitRender's `save_ledger_safe` call site (`visual/batch_flux_portrait_render.py:572-586`) was bypassed in the failing runs because `_load_ledger_with_path` returned `(led, None)` for the stale `{`-prefix `ledger_json` string, hitting the `if led_path is not None:` caller guard at L574. After BUG-LOCAL-234's HuMo singleton refresh lands, PortraitRender should now resolve a real led_path via the singleton fallback, save_ledger_safe should run, and cast.portrait_path should be stamped. **Verification: next smoke -- if portrait_path stamps are present in the ledger AND HuMo Tier 1 succeeds, this entry promotes to [VERIFIED NOT-A-BUG] (resolved as a side-effect of 234 fix). If stamps still missing, this is a separate defect requiring its own patch in PortraitRender's ledger access pattern.**
**Prior 8 entry:** BUG-LOCAL-236 (2026-05-19) -- `meta.episode_title` left empty despite title regen producing a value. Recent ledger contains: `"episode_title": ""` AND `"title_substitution": {"old_title": "The Geostationary Gambit", "new_title": "Thin Veneer of GEO", "lines_patched": 1, "substitutions": 1}`. Title regen computed "Thin Veneer of GEO", patched the announcer opener line, but did NOT write back to `meta.episode_title`. Downstream `[Video] TITLE LAST-RESORT: meta.episode_title='', meta.title='', led.title='', widget='' -- ALL EMPTY/STUCK. Using timestamp fallback 'Signal Lost 20260518 234153'` fires the timestamp fallback. **[FIXED 2026-05-20 -- verified 2026-05-21]** Resolved before this entry was investigated. **Cause:** a Sprint-E "K.5.7" block re-stamped `meta.episode_title` from the raw `episode_title` widget value AFTER the J.5 post-composition title pass; with the widget left blank it overwrote J.5's LLM-regenerated title with `""`, so the video title chain fell to the timestamp last-resort. **Fix:** K.5.7 deleted 2026-05-20; J.5's `meta["episode_title"] = final_title` (`OTR_LedgerScriptWriter.py` L2734) is now the single title authority. **Verify:** `tests/test_writer_stamps_episode_title.py` (3 source-assertion tests pinning the K.5.7 removal) + full `tests/` walk 2441 passed / 0 failed (2026-05-21). No code change this session -- BUG_LOG status reconciled only.
**Prior 9 entry:** BUG-LOCAL-235 (2026-05-19) -- HuMo Tier 2 portrait glob naming-contract drift. `_find_portrait` in `nodes/batch_humo_render.py:596-604` globs for `pass1_portrait_*.png` / `otr_humo_pass1_portrait_*.png` (pre-BUG-LOCAL-078 naming). BatchFluxPortraitRender writes `{char_id}_portrait.png` (canonical post-BUG-LOCAL-078 2026-05-04). Tier 2 was never updated to match the new naming. Tier 1 (`cast.portrait_path` stamp) usually catches this -- but when Tier 1 fails (BUG-LOCAL-234 rename-stale fallout being the immediate trigger), Tier 2 also fails because of the glob pattern mismatch, leaving HuMo with no portrait. Two-axis path-resolution failure -- 234 was the ledger-stamp axis, 235 is the glob-pattern axis. Fix landed in the same commit as the BUG-LOCAL-234 HuMo singleton refresh: added Tier 1.5 deterministic-filename fallback at `_find_portrait` -- when cast row name matches speaker but `portrait_path` stamp is missing or stale, also check `portraits_dir / f"{char_id}_portrait.png"` directly. Original Tier 2 (legacy `pass1_portrait_*.png` globs) kept for old episodes; the new Tier 1.5 layer closes the current naming contract. See entry below.
**Prior 10 entry:** BUG-LOCAL-234 (2026-05-18 23:42) -- HuMo / LTX / VideoComposite cannot find portraits + radio bookend after the Video render's per-episode-dir rename. Sequence: writer creates dir `pending_20260518_233621`; Video render renames it to `signal_lost_signal_lost_20260518_234153_20260518_234153`; BatchFluxRender (radio bookend) re-resolves the new path via the BUG-LOCAL-021 singleton fallback and writes to the renamed dir correctly; BUT BatchFluxPortraitRender, BatchHumoRender, BatchLTXRender all retain the pre-rename `episode_id='pending_20260518_233621'` from the in-memory ledger and resolve `portraits_dir`, `radio_still_path`, `output_dir` against the OLD (now-renamed-away) path. HuMo log: `[BatchHumoRender] portraits_dir=C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\pending_20260518_233621\portraits`. Result: every character line gets `no portrait AND no radio still`; LTX prints `no radio bookend resolved for episode pending_20260518_233621 -- skipping LTX render entirely`; VideoComposite then refuses to proceed: `per_clip_mux: no usable HuMo clips for any line in ledger.lines[]` and the strict-C7 gate rejects fallback. Wall: 525.04 s before VideoComposite raise. NOT caused by BUG-LOCAL-231 fix attempts. Fix shape: replicate the BatchFluxRender singleton-fallback pattern (BUG-LOCAL-021) in HuMo + LTX + VideoComposite -- resolve paths from the LEDGER FILE's actual on-disk location (`Path(ledger_p).parent.parent`) rather than the cached pre-rename `episode_id` string. Pending investigation + fix.
**Prior 11 entry:** BUG-LOCAL-233 (2026-05-18 23:50) -- **VERIFICATION FAILED 2026-05-19 (prompt fix cc553b7 ineffective; new approach needed).** Latest run ledger evidence: 9 `compose_flags.phantom_name` flags including b007 text containing `"Hold tight there, GULLIVER REEVES"` (LEMMY's line addressing GULLIVER by name vocatively) and multiple `"Hold tight, Gulliver"` instances. The strengthened role-induction prompt ("Here, you are now <SPEAKER>. Produce one line/section of dialogue for <SPEAKER>.") **did not eliminate vocative drift** on Mistral-Nemo writer with this seed/style combination. Prompt-level fix alone insufficient; need a **post-compose vocative-strip pass** that removes `OTHER_CAST_NAME` substrings from each line's text before audio render. Reframed scope: was "LineComposer prompt drift", now "vocative drift in composed dialogue requires post-compose mitigation". New entry status: **[FIX LANDED 2026-05-21 -- pending soak verification].** Three-part fix in `nodes/_otr_line_composer.py`: (A2) `_format_last_lines` relabels the announcer's rolling-window entries as `[narration]` instead of `[ANNOUNCER]:` -- this was the real induction surface, since Tier 1 fix #4 (`_derive_prev_speaker`) had already removed "responding to ANNOUNCER" from the prompt back on 2026-05-11; the LLM was echoing the `[ANNOUNCER]:` label it still saw in the LAST SPOKEN block one step above the WRITE LINE slot. (B) a vocative-drift gate in `compose_line` stamps `compose_flags="vocative_drift:ANNOUNCER"` on any non-announcer line addressing the label -- `detect_phantom_names` could never catch it because "ANNOUNCER" is always roster-whitelisted. (C) `strip_announcer_vocative` deterministically removes the comma/boundary-anchored address (trailing / mid / leading shapes) before the line is committed; plain noun references ("the announcer") and real cast-name address ("Alice, run!") are left untouched. cc553b7's prompt strengthening is kept as belt-and-braces; D (reroll) remains the unbuilt documented fallback. LineComposer prompt drift: pre-fix role-induction sentence ("You are <SPEAKER>.") was too weak; the LLM produced dialogue containing literal vocative mentions of OTHER cast members (e.g. line b004's `"It's bigger than any NIST measurement, ANNOUNCER."` in episode pending_20260518_233216, where LEMMY-the-character's line addressed ANNOUNCER by name inside the dialogue text). Strengthened the prompt in commit cc553b7 (`nodes/_otr_line_composer.py:_build_user_prompt` lines 971-984) to `"Here, you are now <SPEAKER>. Produce one line/section of dialogue for <SPEAKER>."` + optional " You are responding to <PREV_SPEAKER>." The "produce... for <SPEAKER>" framing explicitly binds the LLM to speak AS the named speaker, removing the foothold for vocative drift. **NOT a fix for BUG-LOCAL-232 -- this addresses the writer-prompt drift defect only; the underlying cast-generation defect (writer stamping `name='ANNOUNCER'` + `voice_preset='bm_george'` on a character cast row) is fully upstream of the line composer and remains untested + unfixed.** Verification gate for BUG-LOCAL-233: an operator soak run on the real writer LLM produces character lines and zero contain the literal "ANNOUNCER" label (the soak is operator-run -- not reproducible from Cowork). Regression coverage landed with the fix: `tests/test_vocative_drift.py` (17 tests over `strip_announcer_vocative` address shapes, the `_format_last_lines` `[narration]` relabel, and the `compose_line` gate, including the plain-noun and real-cast-name negative cases). Full `tests/` walk 2458 passed / 22 skipped / 0 failed (2026-05-21). **SOAK VERIFICATION FAILED then RE-FIXED 2026-05-21.** Operator writer run, episode signal_lost_singing_asteroid_language_20260521_120244 (commit 6f2abcd): character line b003 reached the ledger as "This isn't some old rock, ANNOUNCER; this thing has a cadence." with empty `compose_flags` -- the strip missed it. b004 confirmed A2/B/C work for the comma/period shapes (drift detected, stripped, `vocative_drift:ANNOUNCER` flagged). Root cause: `_VOCATIVE_MID_RE` (`r",\s*announcer\s*,"`) accepted only a comma as the mid-sentence closing delimiter, so the `, ANNOUNCER;` semicolon shape matched none of the three vocative regexes; since `strip_announcer_vocative` is both detector and stripper, the drift was neither flagged nor stripped (silent miss). Fix: `_VOCATIVE_MID_RE` widened to `r",\s*announcer\s*([,;:])"` with `subn(r"\1 ", ...)` preserving the matched closing delimiter; +3 tests in `test_vocative_drift.py` (20 total -- semicolon + colon strip cases + a compose_line gate test mirroring b003); full `tests/` walk 2461 passed / 22 skipped / 0 failed (2026-05-21). Bug Bible regression not run (survival-guide `tests/` not traversable from Cowork; change is line-composer text logic, no overlap with the Bible's VRAM/ffmpeg/parse-fatal coverage). Status: **[FIX RE-LANDED 2026-05-21 -- pending fresh operator soak]** -- the widened strip is unit-proven; a new writer run is still required to confirm zero "ANNOUNCER" in character text.
**Prior 12 entry:** BUG-LOCAL-232 (2026-05-18 23:34) -- **PENDING INVESTIGATION (NOT FIXED -- prior "fix worked" claim retracted by Jeffrey 2026-05-18 23:50).** Writer cast-lock contract violation. Episode `pending_20260518_233216`: cast.locked produced 2 rows ("announcer + 1 characters, lemmy_hit=True"), but the row at `char_id='c01'` was stamped with `name='ANNOUNCER'` AND `voice_preset='bm_george'` (a Kokoro `bm_*` voice, not a Bark `v2/en_speaker_*`). BatchBarkGenerator.generate_batch caught the misalignment at `nodes/batch_bark_generator.py:439` after 177.01 s: `ValueError: cast.voice_preset missing or non-v2/* for character 'ANNOUNCER' (line_id=b004, char_id='c01', got 'bm_george'). Writer cast-lock contract violation (Gate 1 + Gate 2 should have caught this upstream).` Run never reached FLUX phase. KokoroAnnouncer's BUG-030 line_id stamping is innocent -- this run's chosen voice was `bm_fable`, not `bm_george`, so KokoroAnnouncer cannot have written the cast row. Root cause is in **writer cast generation logic** -- the cast row was malformed at write time (c01 = ANNOUNCER name + Kokoro voice). Cast-lock contract Gates 1+2 should have rejected this shape; they did not. The 2026-05-18 23:30 follow-up run (pending_20260518_233621) did NOT recur this defect, but **that is seed variance, not a fix** -- the cast generator + Gates 1+2 logic have not been touched. **Pending investigation:** read `nodes/_otr_casting.py` Gates 1+2 logic + the cast generator's char_id <-> name <-> voice_preset alignment. Add a regression test that constructs a `(speaker_role='character', char_id_resolving_to_announcer_cast_row)` ledger and asserts Gates 1+2 reject it. See entry below.
**Prior prior prior last entry:** BUG-LOCAL-231 (2026-05-18) -- Residual VRAM pressure + slow FLUX sampler after the BUG-LOCAL-230 fp8 fix. Surfaced by the 2026-05-18 21:10 verification smoke run for BUG-LOCAL-230. With fp8 weights loaded correctly at 13.21 GiB delta=11.08 (gate #4 PASS), the FLUX sampler still ran at 154 s/step (target ~10-15 s) with VRAM peak 15911 MB (over the 14.5 GiB ceiling by ~756 MB) and 1098 MB D3D Shared Memory paging. This is 3.6x faster than the broken pre-fix 564.99 s/it but ~10x slower than the architectural-fix-only target. Separate defect, NOT a dtype-upcast surface. Tagged candidate causes: (a) writer LLM cache stale residency competing for VRAM at FLUX entry; (b) sampler-time launch flag candidates; (c) FLUX CLIP text encoder footprint; (d) FLUX-schnell fallback at 4 steps. **Round-robin gated on candidate ordering -- ChatGPT + Gemini before any code change.** Bible candidate: pending close. See entry below.
**Prior prior prior prior last entry:** BUG-LOCAL-230 (2026-05-18) -- FLUX1-dev-fp8 forced to fp16 by `--force-fp16` ComfyUI launch arg. Closure-run FLUX sampler ran at ~9.4 min/step (564.99 s/it) for 20 steps because flux1-dev-fp8 weights upcast from native fp8 (~11 GiB) to fp16 (~22 GiB) on a 16 GB card, forcing the dynamic offloader to page weights per sampler step. Fix applied across all four launcher sites in the repo + manual launcher: `start_comfy.bat`, `scripts/worker_iter.py` line 549, `scripts/start_comfy_h0_baseline.bat` line 20, `scripts/_start_comfyui.ps1` line 62. **Architectural axis PROVEN by 2026-05-18 21:10 smoke run: gates #1-#4 + #7 PASS** (load delta 11.08 GiB vs predicted ~11 GiB, dtype `torch.float8_e4m3fn` with bf16 cast). Runtime axis FAILING (gates #5 + #6) on a separate defect (BUG-LOCAL-231, residual VRAM pressure). BUG-LOCAL-230 stays at pending-verification posture; promotion to [FIXED] gated on BUG-LOCAL-231 close + clean 7-criteria re-run. Bible candidate: yes -- generalize-able lesson is "global precision flags silently upcast quantized checkpoints (fp8/fp4/NF4/INT8/GGUF) on VRAM-constrained cards; retire the flag in any quantized-checkpoint launcher." See entry below for full details.
**Prior prior prior last entry:** BUG-LOCAL-229 (2026-05-17) [FIXED Sprint H §3.7 2026-05-17] -- Sprint H bug-hunt §3.7 closure. Two-process supervisor + worker harness wired clean across all four §3.7 checks (mid-execution kill, ComfyUI tree drop, supervisor survival via dual-PID keep-list, supervisor synthesizes `worker_crash` row when atexit cannot fire under Windows `/F`). Surfaced and fixed seven workflow widget-vector defects (nodes 1, 3, 11, 12, 14, 20, 59) across three drift classes (stale `{}` orphan, `forceInput`-added-post-save, missing seed `control_after_generate` companion). Mapper extensions in `scripts/otr_api.py`: Reading C (companion-aware `_serialized_slot_names`) + Reading D (`forceInput=True` slot omission). Workflow harness extensions: socket-bind port preflight, JSON-shape readiness check, dual-PID variadic sweep keep-list, classifier case-insensitive + `executed_count==0` mapping. See entry below for the consolidated audit trail.
**Prior prior prior last entry:** BUG-LOCAL-228 (2026-05-14) [FIXED S31 B4 2026-05-14] -- TIMEOUT_RECOVERY CUDA-race regression introduced at S30 B4b. `story_orchestrator._run_with_timeout` called `_otr_model_loader.unload_llm()` on phase timeout; the unload's `model.to("cpu")` + `torch.cuda.empty_cache()` raced with orphan worker thread still executing CUDA kernels on the cached model -> cudaErrorIllegalAddress. S31 B4 replaced with new `invalidate_cache_no_gpu_teardown()` lifecycle helper (dict-only, GPU-untouched). See entry below for full details.
**Stack head when last updated:** S30 paused at B1c handoff `b12b941` (s30-two-model-selector branch). Continuation plan at `docs/2026-05-14-S30-continuation-plan.md`. Final QA review template at `docs/2026-05-14-S30-final-qa-review.md` (filled in at B8).
**Promotion target:** `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`
**Bible candidates pending promotion:** 25 entries (BUG-LOCAL-201, 202, 204, 205, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229) -- see "Bible candidates pending promotion" section below. Batch-promote after v2.0 ships per `feedback_roadmap_buglog_live_docs`. 221 closed S27 QA-5 (third-party reclassification + `_strict_probe.py` durable harness). 229 is the Sprint H §3.7 consolidated entry (uv-stub PID split + ComfyUI serialization rules + workflow widget drift triad). 226 + 228 join the set with this correction (S30 §2b dead-runtime audit-miss + the S31 B4 timeout-recovery CUDA-race fix -- both [FIXED] + verified). 230 (FLUX fp8 dtype-upcast launch-arg fix) is intentionally NOT in the set: it stays at pending-verification posture, promotion gated on BUG-LOCAL-231 close per the no-premature-promotion rule -- it joins once it lands [FIXED]. **BUG-LOCAL-231 PROMOTED from [PARTIAL] to [ACTIVE REGRESSION CONFIRMED] 2026-05-19 14:15** per community benchmark cross-reference (Comfy-Org #9002 esp-dev 2026-02-02 = 0.75 s/it on identical RTX 5080 + torch 2.10.0+cu130 + Win11 + fp8/bf16 + 1024x1024 + 20 steps; OTR = 150-188 s/it = **200-250x slower**). The methodology delta is ~5-node stock workflow vs 36-node OTR pipeline with custom VRAM controllers. Defensive VRAM controllers (DeferredCheckpointLoader / FluxBranchGate / LtxBranchGate / UnloadAll / BatchFluxRender nuclear eviction / Sprint H launcher overrides) added across Sprint H + S28+ **without per-change FLUX-pace benchmark** are the hypothesis under test. Minimal-workflow bisect now the single critical path (9-step plan locked: build _bisect_flux_minimal.json with stock CheckpointLoaderSimple + KSampler + CLIPTextEncode + VAEDecode + SaveImage, smoke 3x, then re-add OTR axes one-by-one until first 10x slowdown identifies the culprit). **BUG-LOCAL-232 stays PENDING** (cast generator + Gates 1+2 still untested). **BUG-LOCAL-233 FIX LANDED** 2026-05-21 (A2 `[narration]` relabel of the announcer in the LAST SPOKEN window + B `vocative_drift:ANNOUNCER` compose-flag + C deterministic `strip_announcer_vocative`, all in `_otr_line_composer.py`; cc553b7 prompt fix kept as belt-and-braces); 2026-05-21 operator soak FAILED on a `, ANNOUNCER;` semicolon address (episode signal_lost_singing_asteroid_language_20260521_120244 line b003) -- `_VOCATIVE_MID_RE` widened to accept `;`/`:` closing delimiters, +3 tests, full `tests/` walk 2461/0; pending a fresh operator soak. **BUG-LOCAL-234 + 235** fixes landed at fbd0e0c but verification BLOCKED on BUG-LOCAL-231 PARTIAL (smoke can't reach HuMo while FLUX thrashes). **BUG-LOCAL-236 + 240** [FIXED] 2026-05-21 (236 `episode_title` K.5.7 clobber, resolved 2026-05-20; 240 freeze style-slug validator relaxed to a snake_case shape check). **BUG-LOCAL-237, 238, 239, 241, 242** remain filed-and-deferred (header summaries) -- not active work until pipeline closes.

---

## What this file is

Live, append-only record of every bug found in OTR development.
Per CLAUDE.md project rule:

> Maintain `BUG_LOG.md` actively. Every bug logged the moment it's
> found -- no batching, no waiting. Live document tracking the
> build history.

Bugs are numbered `BUG-LOCAL-NNN` with monotonic per-era ranges:

- `001-129` -- pre-voice-path-cleanbreak era (LFC + earlier)
- `200+`    -- voice-path-cleanbreak era (P1-P3, S1-S15)

Numbering reset is intentional -- it gives a clean visual cut
between sprint epochs and lets a reader skim "find me a v2.0-alpha
bug" without scrolling through legacy entries.

---

## Entry schema (per CLAUDE.md)

```markdown
### BUG-LOCAL-NNN: Title
- **Date:** YYYY-MM-DD | **Phase:** 0-6 | **Bible candidate:** yes/no
- **Symptom:** exact error / console output
- **Cause:** root cause (or "pending -- awaiting investigation")
- **Fix:** what resolved it (or "pending")
- **Verify:** how to confirm
- **Tags:** vram, widget-drift, ffmpeg, subprocess, parse-fatal, dialogue-scaling, json-wiring, etc.
```

Mark `[FIXED commit-sha YYYY-MM-DD]` after the title when resolved
-- do not delete entries. When `Bible candidate: yes` and the fix
is verified, promote to the survival guide repo per CLAUDE.md
"Bug Log Pipeline" section.

---

## Active known failures (S15 quarantine)

The entries in `tests/conftest.py::EXPECTED_FAILED_NODEIDS` (mirrored
in `docs/known-failures.md`) are NOT bugs in this file's sense -- they
are failing tests under quarantine, not unfixed production runtime
bugs. The `pytest_sessionfinish` hook (S15.1+S15.2 / commit `f813b37`)
enforces that the actual-failure SET stays exactly the quarantine set.

As of 2026-05-21 the set holds **2 entries** -- KNOWN-FAIL-007 and
KNOWN-FAIL-008, both the `google/gemma-4-E4B-it` default-workflow
license-gate failure (see `docs/known-failures.md` for the removal
condition; resolution is a product/licensing decision for Jeffrey,
not a mechanical fix). The 6 entries this section previously named
were all promoted 2026-05-13 (s26-downstream sweep); the set was
empty from then until the 2026-05-21 full-walk triage.

Entries below are bugs found in production logic during S6-S15
sprints, regardless of fix-status.

---

## Voice-path-cleanbreak era (BUG-LOCAL-200+)

### BUG-LOCAL-258: `forceInput` socket conversion dropped `widgets_values` slots -- ComfyUI rejected the entire visual branch [FIXED 2026-05-23]
- **Date:** 2026-05-23 | **Phase:** 4 (video) | **Bible candidate:** yes
- **Symptom:** Two consecutive episode runs (`signal_lost_white_matter_breach_20260523_135956` and `signal_lost_void_bending_capital_20260523_140834`, both on commit `56c552d`) produced a complete audio mp4 plus the `OTR_SignalLostVideo` procedural fallback video, but NO Flux character stills, NO Flux portraits, NO HuMo clips, NO LTX, NO composite, NO RTX upscale. ComfyUI console at `got prompt`, before any node executed:
  ```
  Failed to validate prompt for output 56:
  * OTR_BatchFluxRender 23:
    - Value not in list: sampler_name: 'simple' not in (list of length 63)
    - Failed to convert an input value to a INT value: seed, randomize, invalid literal for int() with base 10: 'randomize'
    - Failed to convert an input value to a FLOAT value: guidance, cinematic 35mm film still, dim starship interior, ...
    - Value not in list: scheduler: 1024 not in ['simple', 'sgm_uniform', ...]
    - Failed to convert an input value to a FLOAT value: cfg, euler, ...
    - Value 3 smaller than min of 256: height
  * OTR_BatchFluxPortraitRender 59: (same shape)
  ```
  Outputs 56 / 52 / 55 / 59 / 58 / 25 / 51 -- the whole visual pipeline -- each logged `Output will be ignored`.
- **Cause:** commit `56c552d` ("ROADMAP PRIORITY 3 quick win #1") converted `script_json` / `news_used` / `ledger_json` to sockets on nodes 12 / 20 / 23 / 59. It added `forceInput: True` in the Python `INPUT_TYPES` (correct) AND, in `otr_scifi_16gb_full.json`, deleted each converted input's `widgets_values` slot and its `inputs[].widget` sub-key (wrong). `forceInput: True` does NOT remove the widget -- the widget still exists and still occupies one `widgets_values` slot; `forceInput` only forces the widget to *render* as an input socket. Deleting the slot shifted every later widget on the node left by one position. On node 23, ComfyUI then read `seed` <- the control word `'randomize'`, `cfg` <- `'euler'`, `sampler_name` <- `'simple'`, `scheduler` <- `1024`, `guidance` <- a prompt string. Every misread value failed type validation, so ComfyUI's prompt validator rejected the node and dropped the entire downstream visual branch. Node 12 survived only by luck -- its 3 shifted values landed on wired/defaulted slots that did not type-fail. Proof: every validation error maps exactly to a one-slot left shift, and the shift size equals the single deleted entry. Same drift family as BUG-LOCAL-229 and BUG-LOCAL-253.
- **Fix:** surgical revert of the JSON + test portions of `56c552d`; the `forceInput` Python additions are kept. `otr_scifi_16gb_full.json` nodes 12 / 20 / 23 / 59 restored to the pre-commit shape -- `widgets_values` slot + `inputs[].widget` sub-key present. Node 23 is back to 18 aligned entries with `script_json` at `[0]` (`['', 16, 1, 'randomize', 20, 1.0, 'euler', 'simple', 1024, 1024, 3.5, ...]` -- `sampler_name`, `scheduler`, `seed` all land in type-valid slots). `tests/test_workflow_canonical_baseline.py` `skip_env_stills` index restored 16 -> 17. **Correct recipe for this conversion: adding `forceInput: True` is a Python-only change. A `forceInput` STRING input keeps its `widgets_values` slot and its `inputs[].widget` sub-key -- never edit the JSON widget vector for it.**
- **Verify:** static checks done -- node 23 `widgets_values` length 18 and type-aligned; `forceInput` still present in `nodes/otr_video_plan.py`, `nodes/video_engine.py`, `visual/batch_flux_render.py`, `visual/batch_flux_portrait_render.py`. Regression (2026-05-23): full `tests/` walk 2491 passed / 17 skipped / 0 failed; Bug Bible regression 16 passed / 7 skipped / 3 xfailed; the 94 workflow-JSON tests (`test_workflow_canonical_baseline` + `test_workflow_json_guardrails` + `test_otr_api_companions`) all green. Runtime confirmation gate (still open): an operator episode run with zero `Failed to validate prompt` lines and a real Flux/HuMo composite under `episodes/<ep>/stills/` + `videos/`.
- **Tags:** widget-vector-drift, json-wiring, forceInput, video, comfyui-validation, regression

### BUG-LOCAL-257: Writer model dropdown lists non-LLM checkpoints (FLUX / LTX-Video / Depth-Anything) [FIXED 2026-05-23]
- **Date:** 2026-05-23 | **Phase:** writer / model-catalog | **Bible candidate:** no (project-specific dropdown discovery gap, not battle-tested -- found by inspection, not a crash)
- **Symptom:** The `OTR_LedgerScriptWriter` `creative_writing_model` / `technical_model` dropdowns listed `black-forest-labs/FLUX.1-dev`, `depth-anything/Depth-Anything-V2-Large-hf`, and `diffusers/LTX-Video-0.9.0` beneath the 4 curated story LLMs (Mistral-Nemo-Instruct-2407, gemma-4-E2B-it, gemma-4-E4B-it, Qwen2.5-14B-Instruct). Selecting any of them for a writer slot would hand a diffusion / vision checkpoint to the causal-LM loader and fail mid-run.
- **Cause:** `nodes/_otr_model_catalog.py::build_dropdown_choices` appends every on-disk repo returned by `scan_local_llm_cache` that is not already curated, as a `curated=False` entry. `scan_local_llm_cache` walks `HF_HOME/hub/models--*` and does no model-type filtering -- despite the name, it returns every repo it finds. Because OTR's `HF_HOME` consolidates all HuggingFace downloads (LLM, diffusion, vision, depth) into a single `hub/`, that walk rakes in FLUX / LTX-Video / Depth-Anything and drops them into the writer model picker.
- **Fix:** New `_snapshot_is_causal_lm(snapshot_path)` gate in `_otr_model_catalog.py`. `build_dropdown_choices` admits a non-curated cache hit only when the snapshot's `config.json` declares an `architectures` entry ending in `ForCausalLM`. Diffusion pipelines (FLUX, LTX-Video) ship `model_index.json` and carry no root `config.json` -- excluded by the `is_file` check. Vision / depth transformers models carry a `config.json` whose architecture is not `*ForCausalLM` -- excluded by the architecture check. The curated 4 are added unconditionally and are exempt from the gate; they remain the explicit writer set. Scope is the dropdown only -- `scan_local_llm_cache` and the `validate_model_id` admit-path are intentionally untouched (the validator stays permissive about what is loadable; the dropdown is the UI affordance that should only suggest LLMs). No workflow JSON re-wiring required: the change narrows a live-generated COMBO choice list; no node class, widget name, or socket changed, and the saved widget values (curated repo ids) remain valid choices.
- **Verify:** `tests/test_model_catalog_scan.py` -- 8 new tests: `test_dropdown_excludes_diffusion_pipelines`, `test_dropdown_excludes_vision_model`, `test_dropdown_keeps_uncurated_causal_lm_amid_mixed_cache`, `test_dropdown_curated_set_exempt_from_cache_filter`, plus 4 direct `_snapshot_is_causal_lm` unit tests. The `_make_snapshot` fixture helper gained an `architectures` parameter (causal-LM by default; `None` models a diffusers repo with no root config). Full `tests/` walk 2491 passed / 17 skipped / 0 failed (2026-05-23).
- **Tags:** writer, model-catalog, dropdown, hf-cache, model-type-filter

### BUG-LOCAL-256: `detect_phantom_names` flags component words of multi-word roster entries [FIXED 2026-05-22]
- **Date:** 2026-05-22 | **Phase:** writer / line-composer | **Bible candidate:** no (project-specific heuristic gap)
- **Symptom:** Episode `signal_lost_temporal_mirror_echoes_20260522_150549` -- cast `['ANNOUNCER','LEMMY','GULLIVER REEVES']`, `meta.news.key_terms` includes `'Big Bang'`. Character line b003 carries `compose_flags=['phantom_name:Gulliver']`; b007 carries `['phantom_name:Gulliver','phantom_name:Big','phantom_name:Bang']`. All three are false positives: GULLIVER REEVES is a locked cast member and Big Bang is a declared key_term, so none is an invented name. Noise in `compose_flags` + `meta.compose_flag_summary` that obscures real phantom detections.
- **Cause:** `nodes/_otr_line_composer.py::detect_phantom_names` tested each proper-noun candidate with `tok_u in allowed_roster` -- an exact whole-string match. `allowed_roster` entries can be multi-word: full cast names (`'GULLIVER REEVES'`) and multi-word key_terms (`'BIG BANG'`). The bigram pass (pass 3) correctly clears the whole phrase `'Big Bang'` against the `'BIG BANG'` entry, but the single-word pass (pass 4) then tests `'BIG'` / `'BANG'` / `'GULLIVER'` individually -- none is a standalone roster entry -- and flags each. The single-word de-dup guard only consults already-flagged `found`, not the roster, so it could not suppress them.
- **Fix:** `detect_phantom_names` now builds a local `allowed` set = `allowed_roster` plus every whitespace-delimited component word of each roster entry; all four passes test membership against `allowed`. A candidate clears when its uppercase form is a roster entry OR a whole-word component of a multi-word entry. Component words are low-risk to allow: the gate is detect-and-flag-only (the line ships regardless; Phase 3 reviewer owns repair), and a word that belongs to a known entity is by definition not an invented name. `build_allowed_roster` is unchanged -- its contract + tests are untouched; the fix is local to `detect_phantom_names`.
- **Verify:** `tests/test_phase0_name_roster.py` -- 4 new tests: `test_component_of_multiword_cast_name_not_flagged`, `test_component_of_multiword_key_term_not_flagged`, `test_bug256_episode_scenario` (replays the b007 line), `test_genuine_phantom_still_flagged_with_multiword_roster` (an unrelated invented name still flags, so detection is not disabled). Full `tests/` walk green.
- **Tags:** writer, line-composer, phantom-name, compose-flags, false-positive, multi-word-roster

### BUG-LOCAL-255: `override_announcer_close` keys on the private `_speaker_role` -- `news_close_brief` silently dropped every episode [FIXED 2026-05-22]
- **Date:** 2026-05-22 | **Phase:** writer / news-wiring | **Bible candidate:** no (key-contract drift, project-specific)
- **Symptom:** Writer logs `news_close_brief present but no announcer line found in led.data['lines'] to stamp onto; closing read will use the line composer's original text` on every episode that carries a news_close_brief. Confirmed in episode `signal_lost_biolab_breakthrough_20260522_073101`: `meta.news.news_close_brief` is non-empty ("Scientists continue the critical work of developing highly personalized treatments...") but the closing announcer line b005 `text` is generic composer output ("And so the frontier advances, as the war moves beyond the lab."), not the journalistic close. The announcer outro is never the news_close_brief.
- **Cause:** `nodes/_otr_news_wiring.py::override_announcer_close` selects the announcer row with `row.get("_speaker_role") == "announcer"` (L71) -- the PRIVATE in-flight key set by the per-beat loop and stripped by `set_lines()`. The writer calls the helper with `led.data["lines"]`, i.e. post-`set_lines` rows that carry only the PUBLIC `speaker_role` key. The match finds nothing, the helper returns `None`, and the brief is dropped. The sibling helper in the same module, `post_assembly_keyterm_check`, already uses the correct fallback `r.get("_speaker_role") or r.get("speaker_role")` (L108) -- `override_announcer_close` was never given the same fallback. Helper/caller key-contract drift.
- **Fix:** `override_announcer_close` retired outright -- deleted from `nodes/_otr_news_wiring.py` (the writer was its only caller). The closing announcer line is now written by `_otr_line_composer.compose_announcer_outro`, a dedicated `creative`-slot LLM pass added in the announcer dedicated-pass build (2026-05-22; design `docs/2026-05-22-announcer-passes-build-handoff.md`). The writer's I.5 post-loop block resolves the final announcer beat by its known `beat_id` (`last_announcer_id`, derived once from `outline.beats` before the per-beat loop) and writes the outro text via `_otr_ledger.patch_line_text(led.data, last_announcer_id, ...)` -- a `line_id` match, which sidesteps the broken `_speaker_role` row-selection entirely. The outro pass receives `script_brief` + `news_close_brief` + the composed `intro_text`; on any LLM failure a deterministic `fallback_announcer_outro(news_close_brief)` fires so the journalistic close is never missing. `news_close_brief` no longer depends on a key-name match to reach the closing line.
- **Verify:** `tests/test_announcer_passes.py::test_bug255_news_close_brief_reaches_closing_line` -- simulates the writer's I.5 post-loop block on a ledger whose rows carry the PUBLIC `speaker_role` key (the exact shape the old helper missed) and asserts the close lands on the closing line via `patch_line_text`. Companion `test_bug255_outro_fallback_still_lands_on_closing_line` covers the LLM-failure path. Full `tests/` walk 2488 passed / 22 skipped / 0 failed (2026-05-22); Bug Bible regression 16 passed / 7 skipped / 3 xfailed; `test_audio_byte_identical.py` green (audio path untouched).
- **Tags:** writer, news-wiring, key-contract-drift, announcer, silent-failure, set_lines

### BUG-LOCAL-254: BUG-231 `_bisect_flux_*.json` workflows carry slug `id`s -- ComfyUI Zod rejects on UI load [FIXED 2026-05-21]
- **Date:** 2026-05-21 | **Phase:** regression-triage | **Bible candidate:** no (same class as BUG-LOCAL-012)
- **Symptom:** `test_workflow_zod_shape.py::test_root_id_is_uuid` -- 5 parametrized cases failed (`_bisect_flux_minimal / v1_deferred_loader / v2_branch_gate / v3_batchflux_nuclear / v5_unload_all .json`): the root `id` is a slug like `bisect-flux-minimal-bug-local-231`, not a UUID. ComfyUI's Vue frontend Zod validator rejects such a workflow with `Invalid uuid at "id"` on canvas load (same failure class as BUG-LOCAL-012).
- **Cause:** The 5 BUG-LOCAL-231 FLUX-pace bisect workflows were hand-built with human-readable slug `id`s. They are meant to be drag-loaded into ComfyUI Desktop for the minimal-workflow bisect, so they must satisfy the Zod UUID contract -- the canonical `otr_scifi_16gb_full.json` already does (its `id` is a UUID).
- **Fix:** Each `_bisect_flux_*.json` root `id` replaced with a fresh UUID4. The descriptive label is preserved in the filename, so no information is lost.
- **Verify:** `pytest tests/test_workflow_zod_shape.py` -- all 5 `test_root_id_is_uuid` cases pass; the bisect workflows load cleanly in the ComfyUI frontend.
- **Tags:** json-wiring, workflow-json, zod-shape, uuid, bug-231-followup, bug-012-class

### BUG-LOCAL-253: `OTR_MusicGenTheme` workflow widget vector short by one -- `episode_seed` slot missing [FIXED 2026-05-21]
- **Date:** 2026-05-21 | **Phase:** regression-triage | **Bible candidate:** no (widget-drift class already covered, e.g. BUG-LOCAL-210)
- **Symptom:** `test_workflow_audio_widget_vectors.py` -- 3 tests failed: `test_musicgen_widget_vector_length_matches_input_types` (got 4 values, expected 5), `test_musicgen_allow_silence_fallback_pinned_false` (`IndexError: list index out of range`), `test_no_stale_dict_residue_in_widget_vector[OTR_MusicGenTheme-...]` (widget[2] model_id declared STRING but value is `3.0` float -- shifted slots).
- **Cause:** `workflows/otr_scifi_16gb_full.json` node id=14 (`OTR_MusicGenTheme`) `widgets_values` was `["", "facebook/musicgen-medium", 3.0, false]` (4 entries). `MusicGenTheme.INPUT_TYPES()` declares 5 widget slots in order `script_json, episode_seed, model_id, guidance_scale, allow_silence_fallback`. The `episode_seed` slot (STRING, default `""`) was missing, shifting model_id / guidance_scale / allow_silence_fallback one position left.
- **Fix:** `workflows/otr_scifi_16gb_full.json` node 14 `widgets_values` -> `["", "", "facebook/musicgen-medium", 3.0, false]` (inserted the `episode_seed` default `""` at index 1).
- **Verify:** `pytest tests/test_workflow_audio_widget_vectors.py` -- all MusicGen tests pass; each widget[i] type now matches the INPUT_TYPES declared order.
- **Tags:** json-wiring, widget-drift, musicgen, workflow-json

### BUG-LOCAL-252: `nodes/batch_humo_render.py` relative imports of `_otr_story_brief_helpers` break standalone module loads [FIXED 2026-05-21]
- **Date:** 2026-05-21 | **Phase:** regression-triage | **Bible candidate:** no (BUG-LOCAL-249 followup)
- **Symptom:** `test_batch_humo_render.py::test_build_pos_prompt_*` (3 tests) failed with `ImportError: attempted relative import with no known parent package` at `nodes/batch_humo_render.py:1187` (`from ._otr_story_brief_helpers import get_story_brief_lighting`).
- **Cause:** `batch_humo_render.py` is loaded as a top-level module (no parent package) by the test harness. Its documented import strategy (top-of-file docstring + the `_otr_paths` / `_otr_speaker_role` imports at lines 61/69) is: sys.path-prepend `_NODES_DIR`, then ABSOLUTE imports. Two later additions (Sprint C C5f, lines 1187 + 1645) used relative imports `from ._otr_story_brief_helpers import ...`, which require a parent package. This is the same defect class BUG-LOCAL-249 fixed in `visual/batch_flux_render.py` -- 249 missed these two sibling sites in the HuMo node.
- **Fix:** Lines 1187 + 1645 changed from `from ._otr_story_brief_helpers import` to `from _otr_story_brief_helpers import`, matching the file's documented sys.path-prepend + absolute-import pattern. Both sites fixed (only 1187 had a test exercising it; 1645 is the identical defect on the `get_story_brief_status` path).
- **Verify:** `pytest tests/test_batch_humo_render.py` -- the 3 `_build_pos_prompt` tests pass; full walk shows no regression.
- **Tags:** relative-import, batch-humo-render, story-brief, bug-249-followup

### BUG-LOCAL-251: Partial `folder_paths` stub installed at import time poisons `sys.modules` -- 9 order-dependent contract-validator failures [FIXED 2026-05-21]
- **Date:** 2026-05-21 | **Phase:** regression-triage | **Bible candidate:** no (test-infra, not a node-runtime defect -- but the "one complete stub in conftest, never partial stubs at import time" lesson is worth remembering)
- **Symptom:** In the full `tests/` walk, 8 `test_workflow_contract_validation.py` tests + `test_workflow_live_passes_validator.py::test_production_workflow_passes_default_validation` failed with `nodes._workflow_validation.WorkflowValidationError: OTR_DeferredCheckpointLoader(id=22): INPUT_TYPES() raised AttributeError: module 'folder_paths' has no attribute 'get_filename_list'`. All 9 PASSED when their files were run in isolation -- order-dependent.
- **Cause:** Three test modules (`test_story_brief_humo_c5f.py`, `test_story_brief_ltx_c5e.py`, `test_story_brief_portraits_c5d.py`) each install a PARTIAL `folder_paths` stub (only `get_output_directory`) into `sys.modules` at module-import time, guarded by `if "folder_paths" not in sys.modules`. Whichever pytest collects first wins; none carry `get_filename_list`. `nodes/_otr_deferred_loaders.py` hard-imports `folder_paths`; `OTR_DeferredCheckpointLoader.INPUT_TYPES()` calls `folder_paths.get_filename_list("checkpoints")`. In isolation no stub is installed -> the deferred-loader module fails to import -> the class is skipped by the contract validator's mapping builder -> no error. In the full walk the partial stub makes the import succeed, so the validator reaches `INPUT_TYPES()` and the missing attribute raises.
- **Fix:** `tests/conftest.py` installs ONE complete `folder_paths` stub (`get_output_directory` + `get_filename_list`) at conftest-import time, before any test module is collected. Every downstream `if "folder_paths" not in sys.modules` guard then becomes a no-op, so the whole suite shares one consistent complete stub and the order-dependence is gone. The three partial stubs are left in place as harmless no-ops (defensive fallback if a file is run without the conftest).
- **Verify:** Full `tests/` walk -- the 9 failures clear with zero regressions (27 -> 2 over the triage). `OTR_DeferredCheckpointLoader` node id=22 validates clean against the contract validator's socket + widget-drift checks.
- **Tags:** test-infra, sys-modules-pollution, folder_paths, contract-validator, order-dependent, stub

### BUG-LOCAL-250: otr_video_plan era tail + FLUX env/portrait prompts driven by the style preset, not meta.story_brief [FIXED 2026-05-20]
- **Date:** 2026-05-20 | **Phase:** 4 | **Bible candidate:** no
- **Symptom:** BUG-LOCAL-249 fixed the radio bookend but left the rest of the downstream visual surface still steered by the upstream style preset. `nodes/otr_video_plan.py` resolved the PASS 3 composite era tail from `_ERA_TAIL_BY_STYLE` -- a dict keyed by style-preset slugs, off a `style` widget -- and `meta.story_brief` was never read at all (BRIEF-ABSENT). `_parse_env_prompts` (env FLUX) and `_build_portrait_prompt` (portrait FLUX) did compose `meta.story_brief` in, but it trailed the env description / generic style literals instead of leading (BRIEF-GATED). Per Jeffrey 2026-05-20 the style preset is an UPSTREAM input -- it shapes the story-writing LLM only; every downstream visual must derive from `meta.story_brief`.
- **Cause:** `otr_video_plan.py` predates the Sprint C `meta.story_brief` consumer wiring; it kept the original style-preset era-tail lookup and was never migrated. `_parse_env_prompts` / `_build_portrait_prompt` were wired for the brief at Sprint C C5c/C5d but appended it mid-body, not leading -- so the brief was present but not the primary driver.
- **Fix:** `nodes/otr_video_plan.py` -- `_ERA_TAIL_BY_STYLE` dict + `resolve_era_tail()` deleted; new `_resolve_era_tail(meta)` derives the era tail from `meta.story_brief` via `_otr_story_brief_helpers.get_story_brief_lighting` (lighting + atmosphere terms, the visual-aesthetic slice), resolved at the single `_visual_plan_from_script_json` seam. The `style` widget removed from `OTR_VideoPlan.INPUT_TYPES`, `plan()`, and `build_shot_plan()`; the dead `meta.style` projection key dropped. `workflows/otr_scifi_16gb_full.json` node 20 widget vector trimmed 6 -> 5 (style preset value removed). `visual/batch_flux_render.py::_parse_env_prompts` + `visual/batch_flux_portrait_render.py::_build_portrait_prompt` -- the brief now LEADS the composed prompt (was mid-body); the generic cinematic literals follow.
- **Verify:** `pytest tests/test_otr_video_plan.py tests/test_videoplan_freeze_done_gate.py tests/test_no_genre_by_style_c3.py tests/test_otr_api_companions.py tests/test_workflow_json_guardrails.py tests/test_story_brief_flux_c5c.py tests/test_story_brief_portraits_c5d.py tests/test_radio_prompt_builder.py tests/test_era_literals_c2a.py` -- 223 passed / 6 skipped. Bug Bible regression baseline held. `test_resolve_era_tail_from_brief_lighting` + `test_build_shot_plan_brief_drives_era_tail` pin the brief-driven derivation; `test_input_types_schema` pins the `style` widget is gone; `test_regression_node20_actual_workflow` pins the trimmed widget vector.
- **Tags:** json-wiring, story-brief, flux, video-plan, era-tail, widget-removal, c5c-followup, bug-249-followup

### BUG-LOCAL-249: FLUX radio bookend prompt driven by the upstream style preset, not meta.story_brief [FIXED 2026-05-20]
- **Date:** 2026-05-20 | **Phase:** 4 | **Bible candidate:** no
- **Symptom:** The radio still rendered for every announcer / music / SFX second reflected the widget style preset, never the episode's story. `meta.story_brief` was supposed to drive all downstream visuals after story creation; the radio bookend was the lone holdout (LTX / HuMo / portraits / MusicGen already consume the brief via `_otr_story_brief_helpers`).
- **Cause:** Sprint C C5c (2026-05-15) wired `meta.story_brief` into `_build_dynamic_radio_prompt` as a never-firing Tier-3 fallback -- `if not descriptor and _brief:` -- behind the `gen_params_initial.style` preset. The workflow always carries a style (a preset, or the "let the story decide" sentinel resolves to one), so `descriptor` was never empty and the brief branch never executed. The appended scene-context hint also still read `scenes[0].env` ("INT. ROOM -- NIGHT" boilerplate) -- the exact tier C5c's own docstring claimed it had retired. Net: the radio prompt was `<generic style word> radio broadcast unit[, set in <scene env boilerplate>], <suffix>`; the story brief never touched it.
- **Fix:** `visual/batch_flux_render.py` -- `_build_dynamic_radio_prompt` rewritten brief-first: resolution chain is now `meta.story_brief` -> `episode_id` slug -> hardcoded fallback. The upstream style preset + `scenes[0].env` reads were deleted (the style preset shapes the story-writing LLM only; per Jeffrey 2026-05-20 "all downstream should be the meta brief"). Body shape `radio broadcast unit, <context>, <suffix>`. `_resolve_radio_bookend_prompt`'s `prompt_source` log now reports `story_brief_status` instead of the dead style read; the `radio_bookend_prompt` widget tooltip updated to match. Also fixed the `from ..nodes._otr_story_brief_helpers import` relative import in both `_build_dynamic_radio_prompt` and `_parse_env_prompts` -- it raised `ImportError: attempted relative import beyond top-level package` outside the ComfyUI runtime (every pytest run), so the helper was untestable; both now route through a shared `_story_brief_helpers()` using the nodes-dir-on-sys.path pattern (matches `_resolve_radio_bookend_prompt`).
- **Verify:** `pytest tests/test_radio_prompt_builder.py tests/test_story_brief_flux_c5c.py` -- 60 passed (a large share of those were failing pre-fix on the relative-import crash). `test_style_preset_never_in_prompt` pins that the preset string cannot leak into the radio prompt. Bug Bible regression 16 passed / 7 skipped / 3 xfailed (baseline held). Workflow JSON node 23 `radio_bookend_prompt` widget = "" so the dynamic builder is live in production.
- **Tags:** json-wiring, story-brief, flux, radio-bookend, relative-import, c5c-followup

### BUG-LOCAL-200: G7 contract drift in consumer widgets [FIXED 3090007 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** AudioGen widget `default_duration.min` was the literal `0.5`; ProcSFX widget `default_duration.min` was the literal `0.1`. The freeze cascade enforced G7 SFX `dur_s` bounds at `[SFX_DUR_MIN_S=0.5, SFX_DUR_MAX_S=10.0]` (post-S6.4 tightening). ProcSFX accepted writer values down to 0.1 -- silently disagreeing with the freeze contract -- and clamped them post-hoc. Internal per-cue clamps in BOTH consumers also used magic-number literals (`max(0.5, min(10.0, ...))`, `max(0.1, min(10.0, ...))`).
- **Cause:** Magic-number literal at the consumer surface AND in the consumer clamp; no import of the freeze cascade's authoritative constants. A future bound shift in `_otr_ledger_freeze.py` would have left consumer surfaces silently disagreeing.
- **Fix:** S10.1 -- export `SFX_DUR_MIN_S` and `SFX_DUR_MAX_S` from `_otr_ledger_freeze.py::__all__`; both consumers import them for widget min/max AND internal clamp. Plus drift guard `tests/test_g7_consumer_constants.py` (5 tests) including object-identity assertion catching the local-shadow refactor case.
- **Verify:** `findstr /SI "0.1\|0.5\|10.0\|12.0" nodes\batch_procedural_sfx.py` filtered to widget/clamp sites returns zero hits. Same gate on AudioGen returns zero clamp/widget hits.
- **Tags:** widget-drift, magic-number, contract-honesty, g7

### BUG-LOCAL-201: AudioGen cache key was model-id-blind and guidance-scale-blind [FIXED 574038e 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Switching AudioGen models (`facebook/audiogen-medium` -> `facebook/audiogen-large`) or guidance scales (CFG 3.0 -> 5.0) between runs silently returned the prior model's wav -- the cache key didn't include either input. The user got a "cached" render that was the wrong model's output.
- **Cause:** `_cache_prefix(prompt, duration_sec, episode_seed)` payload was `f"{duration_sec}|{prompt}|{episode_seed}"`. Output-determining inputs `model_id` and `guidance_scale` were never hashed.
- **Fix:** S12.3 -- keyword-only signature `_cache_prefix(*, prompt, duration_sec, episode_seed, model_id, guidance_scale)`. JSON-canonical payload via `json.dumps(..., sort_keys=True, separators=(",", ":"))`. Truncation extended `[:8] -> [:12]` for collision-resistance. Three new dimension tests + drift guards.
- **Verify:** `pytest tests/test_audiogen_cache_keys.py::test_audiogen_cache_prefix_changes_when_model_id_changes -v` (and the guidance_scale + float-canonical siblings).
- **Tags:** cache-key, ledger-derived, audiogen, content-addressed
- **Bible candidate rationale:** General lesson -- cache keys must include every output-determining input. Standing-directive #9 in OTR codifies this; the survival guide should publish it as a pattern.

### BUG-LOCAL-202: ProcSFX silently overwrote on dur_s iteration [FIXED c4ab258 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** When the writer iterated on scene timing -- emitting the same `line_id` at a new `dur_s` between runs -- the second render OVERWROTE the first wav on disk. The user lost A/B history; the cache identity (the on-disk filename) didn't reflect the changed input. Found via the F-6 finding in the S6-S8 round-robin: "active iteration on scene timing is a real workflow."
- **Cause:** Filename was `proc_<sfx_type>_<line_id>.wav`. Identity surface keyed only by line, not by duration. ProcSFX has no formal cache layer, so the on-disk filename IS the de-facto identity.
- **Fix:** S12.1 -- filename extended to `proc_<sfx_type>_<line_id>_<perm>.wav` where `<perm>` is `hashlib.sha256(f"{cue_duration:.3f}|{chosen_type}|{line_id}").hexdigest()[:8]`. Disk usage grows with iteration count; procedural wavs are kB-scale so the trade-off is favorable.
- **Verify:** `pytest tests/test_audiogen_cache_keys.py::test_procsfx_filename_perm_hash_varies_with_dur_s -v`. Also the source-level guard `test_procsfx_perm_hash_in_module_source` catches a future refactor that strips the perm segment.
- **Tags:** cache-key, on-disk-identity, procsfx, content-addressed
- **Bible candidate rationale:** Same general lesson as BUG-LOCAL-201 in a no-cache-layer variant -- the on-disk filename IS the identity surface and must include every output-determining input.

### BUG-LOCAL-203: cast contract accepted structural tokens as character names [FIXED badcae5 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** A cast row like `{"name": "TITLE", "voice_preset": "v2/en_speaker_0", ...}` passed `_assert_voice_preset_invariant` and `_assert_unique_bark_voices` cleanly. The two existing assertions had no opinion on the *name* shape. An LLM hallucination that emitted any of TITLE / NOTE / TARGET / STYLE / NARRATOR as a character name rendered as a Bark voice line in production with no contract pushback.
- **Cause:** Cast contract was preset-shape-aware but not name-shape-aware. The deleted `story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS` had screenplay-meta-direction patterns (KEVIN VOICEOVER, JOHN V.O., etc.) but never anchored the structural-token names from `_BRACKET_STRUCTURAL_TOKENS`.
- **Fix:** S13.1 -- ported the pre-S7.1 `_SFX_CAST_BLOCKLIST_PATTERNS` into `_otr_casting._NON_CHARACTER_CAST_PATTERNS` and EXTENDED with five exact-match patterns (`r"^TITLE$"`, etc). Anchored as exact-match on upcased names to minimize false positives ("Anna Title-Holder" passes). New `_assert_no_structural_tokens_in_cast(cast)` wired into `lock_cast()`. 10-test parametrized + sanity pin in `tests/test_cast_contract_rejects_structural_tokens.py`.
- **Verify:** `pytest tests/test_cast_contract_rejects_structural_tokens.py -v` (10 tests, all green). Audit doc `docs/audit-S13.1.md`.
- **Tags:** cast-contract, structural-tokens, llm-hallucination, defense-in-depth

### BUG-LOCAL-204: no enforcement of line_id uniqueness across ledger.lines[] [FIXED 02ca26c 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Two lines with the same `line_id` in `ledger.lines[]` silently overwrote each other in BOTH places ProcSFX's filename scheme keys by line_id (post-S12.1) AND every ledger write-back path (`patch_line_fields`, `apply_line_timings`) keys by line_id. The user noticed only when an episode rendered with the wrong audio in a slot -- too late to abort cleanly.
- **Cause:** No invariant in the FreezeCascade enforced `line_id` uniqueness. The writer was *expected* to emit unique ids, and historically did, but the contract was implicit.
- **Fix:** S13.2 -- new G8 invariant `_check_g8_line_id_uniqueness` in `_otr_ledger_freeze.py`, wired into `run_gap_audit` alongside G1-G7. Phase 0 collects (warn-mode), Phase 10 raises FreezeAssertionError. Diagnostic caps displayed duplicates at 5 + `(+N more)` suffix.
- **Verify:** `pytest tests/test_g8_line_id_uniqueness.py -v` (7 tests, all green). Production fixtures all pass G8 cleanly -- the writer was already emitting unique ids; G8 makes the invariant load-bearing.
- **Tags:** invariant, freeze-cascade, g8, line-id, write-back
- **Bible candidate rationale:** General lesson -- any system with paths that key by an ID needs structural enforcement that the ID is unique. Structural invariant complements the implicit producer contract.

### BUG-LOCAL-205: regex `\bV\.O\.\b` and `\bO\.S\.\b` patterns never matched [FIXED badcae5 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Pre-S7.1 `story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS` had `r"\bV\.O\.\b"` and `r"\bO\.S\.\b"` intended to catch screenplay meta-direction artefacts like `JOHN V.O.` and `JANE O.S.`. The patterns were DEAD -- they never matched. The trailing `\b` after the final `\.` never fires because Python regex word-boundary doesn't trigger between a non-word char and end-of-string. So the legacy heuristic silently allowed the artefacts through. Discovered during the S13.1 port when the new test `test_legacy_sfx_cue_artefacts_still_caught` failed on `JOHN V.O.`.
- **Cause:** Misuse of `\b` after a non-word char. Regex word-boundary semantics: `\b` matches between a word char and a non-word char. After `.` (non-word) at end-of-string, there is no word char on the right, so `\b` doesn't fire. The pattern silently rejected every input it was supposed to match.
- **Fix:** S13.1 (during port) -- dropped the trailing `\b`. New patterns: `r"\bV\.O\."` and `r"\bO\.S\."`. Verified the post-fix patterns match `JOHN V.O.` via reproduction script before commit.
- **Verify:** `python3 -c "import re; print(bool(re.search(r'\bV\.O\.', 'JOHN V.O.')))"` -> True. Regression test `test_legacy_sfx_cue_artefacts_still_caught` covers it.
- **Tags:** regex, word-boundary, port-found, dead-pattern, legacy-bug
- **Bible candidate rationale:** General lesson -- `\b` after `.` (or any non-word char) at end-of-string is a no-op. Audit any regex of the shape `\b<word>\.<word>\.\b` for the same bug. IMP-15 in the S10-S15 QA doc proposes a codebase sweep.

### BUG-LOCAL-206: `_resolve_genre("")` returned `" audio drama"` with leading space [FIXED 47eb644 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** During S6-A initial implementation, `_resolve_genre("")` returned `" audio drama"` (leading space) due to f-string concatenation `f"{words} audio drama"` where `words` was empty after the `.replace("_", " ").strip()` chain. Caught during pre-commit dev iteration, never shipped.
- **Cause:** Naive f-string concatenation without checking the substituted value's emptiness. `f"{''} audio drama"` = `" audio drama"`, not `"audio drama"`.
- **Fix:** S6-A pre-commit -- conditional `f"{words} audio drama" if words else "audio drama"`. Then S10.2 retired the silent fallback entirely; `_resolve_genre` now raises ValueError on empty input. The mechanical fallback survives only in `_preview_genre` (UI helper, isolated from writer / freeze paths by AST-walk test).
- **Verify:** `pytest tests/test_musicgen_style_palette.py::test_resolve_genre_empty_raises -v`.
- **Tags:** f-string, empty-input, dev-iteration, fallback-isolation

### BUG-LOCAL-227: 25 LFC test failures latent at S30 B8 (wide walk) [FIXED S31.5 B1 2026-05-14]
- **Date:** 2026-05-14 | **Phase:** 5 | **Bible candidate:** no
- **Status:** FIXED at S31.5 B1 via triage. (Was: PRE-EXISTING at S30 B8 close, not an S31 regression.)
- **Triage outcome:** 16 Bucket A (DELETE), 9 Bucket B (REFACTOR), 0 Bucket C (SKIP). Plus 10 S31-relocation collateral surfaced during the wide walk re-run, all refactored Bucket B. Final wide walk: 2080 passed / 8 skipped / 0 unexpected failures.
- **Surfaced:** S31 B1 verification, wide `pytest tests/` walk.
- **Symptom:** Running `pytest tests/` (the wide walk, all 2083 collected items) at S31 B1 produces 25 NEW failures relative to `EXPECTED_FAILED_NODEIDS` -- all under `tests/test_lfc_*.py`. Files affected: `test_lfc_b1_cascade_unload_in_finally.py` (4F), `test_lfc_c4_news_used_passthrough.py` (4F), `test_lfc_freeze_cascade_orchestrator.py` (1F at TestNodeOutputContract::test_class_input_types_schema), `test_lfc_phase_3_polish_in_cascade.py` (15F across TestCascadeIntegrationFlag, TestPhase3DisabledByDefault, TestPhase3EnabledMutations, TestPhase3PolishGenerateFnRouting, TestPhase3RecordStamping, TestPhase3Rejection), `test_lfc_w4_writer_polish_fn.py` (1F at test_polish_fallback_to_none_preserves_back_compat). The KNOWN-FAIL-GUARD hook reports each as "NEW failures (REGRESSION) -- not in EXPECTED_FAILED_NODEIDS".
- **Suspected cause:** Phase 3/4/5/6 deletion collateral. S30 B3/B4 deleted cascade phases (Phase 3 polish, Phase 4 scene coherence, Phase 4.5 smart suggestion, Phase 5 voice drift detection, Phase 6 episode arc Editor notes) and the standalone OTR_LFCPhase4Scene / 5Voice / 6Arc node classes. The orphaned `test_lfc_*.py` files reference deleted symbols / contracts that no longer exist on the cascade surface. The wide-walk count (~25 new failures) matches the scope of "tests for things that got deleted in S30 but the test files were left behind."
- **Fix:** PENDING. Action: triage AFTER S31 close. Either (a) delete the stale test files (they reference deleted phases), or (b) refactor each to test a surviving surface, or (c) add to `EXPECTED_FAILED_NODEIDS` if intentional quarantine. Not blocking S31 -- plan's canonical pytest subset is the gate, not the wide walk. Plan's S31 acceptance #1 targets `~282 / 7 / 2` against the canonical subset, which already excludes these LFC tests.
- **Verify:** After triage, `pytest tests/test_lfc_*.py --tb=line` reports no "NEW failures (REGRESSION)" from the KNOWN-FAIL-GUARD hook. Either all green, or all listed in `EXPECTED_FAILED_NODEIDS` with rationale.
- **Tags:** pre-existing, lfc-deletion-collateral, test-orphan, s30-phase3, s30-phase4, s30-phase5, s30-phase6, post-s31-triage
- **Bible candidate rationale:** Cosmetic in isolation; the broader S10.2 lesson (never silently degrade on production surfaces) is the standing directive #1, already canonical.

### BUG-LOCAL-228: TIMEOUT_RECOVERY CUDA-race regression in `_run_with_timeout` [FIXED S31 B4 2026-05-14]
- **Date:** 2026-05-14 | **Phase:** 5 | **Bible candidate:** yes
- **Introduced at:** S30 B4b (commit `7e65e57`, RSS path rewire). Lived undetected through S30 close (`ccf583d`) and S31 B0..B3.
- **Symptom:** When an OTR_LedgerScriptWriter LLM phase exceeds the per-phase wall-clock budget, `_run_with_timeout` enters its timeout-recovery branch and -- per S30 B4b -- called `_otr_model_loader.unload_llm()`. The comment on the call claimed it "avoids cudaErrorIllegalAddress from orphan worker still on GPU." Live behavior was the opposite: `unload_llm` runs `model.to("cpu")` and `torch.cuda.empty_cache()` WHILE the orphan worker thread is still executing CUDA kernels on the same cached model object. The weight tensors get moved mid-write, the allocator deallocates memory the kernel is still reading from, and the next CUDA op promotes the dirty state to `cudaErrorIllegalAddress`. On a 5080 the process holds port 8000 in a zombified state until manual kill -- same failure surface as BUG-LOCAL-073 in a different code path.
- **Cause:** S30 B4b rewrote the recovery path to "go through canonical helpers" without inspecting the threading model. The pre-B4b path was dict-invalidation-only (`_LLM_CACHE["model"] = None`) -- safe under concurrent kernel execution because no GPU op fired. The B4b rewrite assumed `unload_llm` was the canonical replacement; in fact `unload_llm` is the FULL TEARDOWN path (model.cpu + empty_cache + ipc_collect + synchronize), correct ONLY when no other code is touching the model. Timeout recovery violates that precondition by design.
- **Fix:** S31 B4 added new lifecycle helper `_otr_model_loader.invalidate_cache_no_gpu_teardown()` -- a dict-only invalidator that clears `LLM_CACHE` references in-place WITHOUT calling any `torch.cuda.*` / `model.to` / `gc.collect`. The orphan worker thread holds the model in its stack frame; when it exits naturally the references go and a subsequent clean `unload_llm` (or next `request_slot` cross-model load) handles the teardown safely. `_run_with_timeout` now calls this helper instead of `unload_llm`. Log line updated to reflect the new behavior ("LLM_CACHE invalidated (GPU untouched; orphan worker keeps its model reference)") so future debugging doesn't confuse the post-fix log for the pre-fix one.
- **Verify:** `pytest tests/test_run_with_timeout_safe_invalidation.py -v` asserts both (a) `invalidate_cache_no_gpu_teardown(...)` call present in `_run_with_timeout` AST, (b) `unload_llm(...)` call absent. `pytest tests/test_loader_slot_primitives.py::test_invalidate_cache_no_gpu_teardown_no_gpu_calls` asserts the new helper's AST contains zero `torch.cuda.*` / `model.to` / `gc.collect` calls. Runtime verification (operator-side, post-feature-set): induce a synthetic phase timeout and confirm the next `request_slot` load proceeds cleanly without `cudaErrorIllegalAddress`.
- **Tags:** cuda-race, threading, lifecycle, regression, s30-b4b, vram, timeout-recovery
- **Bible candidate rationale:** General lesson -- GPU teardown helpers (model.cpu + empty_cache + synchronize) are NOT safe to call when other threads may still touch the model. Caller must own the threading model OR use a dict-only invalidator. The fix introduces the separation explicitly: `unload_llm` (full teardown, single-owner contract) vs `invalidate_cache_no_gpu_teardown` (dict-only, multi-owner safe). The error pattern -- "canonical helper rewrite without re-checking concurrency invariants" -- is the bigger lesson; codifies how a clean-break sprint can re-introduce a class of race conditions the prior tangled code avoided structurally.

### BUG-LOCAL-207: `production_plan_or_empty` was an orphan Director-derived fallback [FIXED b443f46 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S15.5.1 pre-flight legacy audit surfaced `nodes/_otr_ledger_consumers.production_plan_or_empty(plan_json)` -- a helper that parses an optional `production_plan_json` Director-shape string and returns `{}` for empty / None / invalid input. The docstring framed it as "the graceful fallback for the optional Director input under v2." Repo-wide grep showed ZERO production callers outside the helper's own module + its own test file (`tests/test_otr_ledger_consumers.py::TestProductionPlanOrEmpty`). It was a dead Director-derived fallback that violated standing directive #11 ("no Director-derived fallbacks").
- **Cause:** The L3 consumer rewrite sprint (2026-05-09/10) preserved this helper as a "Pattern 5 demotion" path so old consumers could degrade gracefully when the Director was unwired. Subsequent voice-path-cleanbreak P2/P3 deleted the Director class + the production_plan_json sockets, but this helper was overlooked when the sprint scope tightened. The "no production callers" status was never re-checked.
- **Fix:** S23.6 -- deleted the function from `nodes/_otr_ledger_consumers.py`, removed the `__all__` entry, dropped the helper-list mention from the module docstring. Deleted `TestProductionPlanOrEmpty` (9 tests) from `tests/test_otr_ledger_consumers.py` in lockstep. Forensic comment preserved at the deletion site citing S23.6 + directive 11.
- **Verify:** `git grep -n "production_plan_or_empty" -- '*.py' '*.json'` returns zero hits across nodes/ scripts/ visual/ tests/ (excluding docs/ which carries the migration history).
- **Tags:** legacy-fallback, directive-11, audit-found, orphan-helper, voice-path-cleanbreak
- **Bible candidate rationale:** General lesson -- a "graceful fallback" surface introduced for a now-deleted upstream consumer is dead weight that lulls future contributors into thinking the upstream is still alive. Audit fallbacks tied to deleted upstreams in the same commit that deletes the upstream; or run a periodic "no production callers" sweep on helpers whose docstring mentions a known-deleted class.

### BUG-LOCAL-210: AudioGen widget vector carried a stale `{}` shifting every subsequent slot [FIXED f7a5ca0 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24 C3 dependency audit found `OTR_BatchAudioGenGenerator.widgets_values` in the production workflow JSON was `['[]', '{}', '', 'facebook/audiogen-medium', 3.0, 3.0]` -- 6 values. INPUT_TYPES declared 6 slots in order `[script_json, episode_seed, model_id, guidance_scale, default_duration, allow_silence_fallback]`. The `{}` at position 1 was the default value of the legacy `production_plan_json` REQUIRED input that voice-path-cleanbreak P2 deleted; the widget value survived the input deletion. Net effect: positions 2-5 all map to the wrong INPUT_TYPES slot. ComfyUI's permissive type coercion masked the misalignment in soak (the runtime didn't crash) but the values getting bound to `episode_seed` (got '{}'), `model_id` (got ''), and `guidance_scale` (got the model_id string) were nonsense.
- **Cause:** When P2 deleted `production_plan_json` from INPUT_TYPES.required, the widget vector in the workflow JSON wasn't trimmed in lockstep. ComfyUI doesn't validate widget-vector length against INPUT_TYPES on load; it accepts any length and silently positionally maps what's there.
- **Fix:** C3 -- widget vector realigned to `['[]', '', 'facebook/audiogen-medium', 3.0, 3.0, False]` with the stale `{}` removed and `allow_silence_fallback=False` appended. C6 added a parametrized test (`test_no_stale_dict_residue_in_widget_vector`) that reflects each class's INPUT_TYPES against the workflow JSON and asserts shape match. Future drift fires here.
- **Verify:** `pytest tests/test_workflow_audio_widget_vectors.py::test_no_stale_dict_residue_in_widget_vector -v`. The runtime values that flow through the graph are now correct.
- **Tags:** widget-vector, position-pinned, cleanbreak-debris, audiogen, voice-path-cleanbreak
- **Bible candidate rationale:** General lesson -- when a cleanbreak deletes a REQUIRED INPUT_TYPES entry, the workflow JSON's widgets_values vector MUST be trimmed in the same commit. ComfyUI's permissive load means the misalignment ships silently. Future plans should add a step to the cleanbreak playbook: "delete input X" -> "shrink every saved-workflow widget vector by 1 at X's index". This batch's C6 widget-vector test catches the next instance.

### BUG-LOCAL-226: S30 sprint-plan §2b audit claim "_load_llm is dead-runtime" was wrong [FIXED 8e1a0c7 2026-05-14]
- **Date:** 2026-05-14 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** S30 sprint plan §2b asserted "Production importers ... pull only `_runtime_log` + `_unload_llm`. No code path calls `_load_llm` at runtime." On B0 REVIEW (grep step mandated by the plan itself), the assertion failed: `OTR_LedgerScriptWriter._resolve_news_seed` (writer.py:958) calls `_fetch_rss_seed_or_die` (writer.py:803) which calls `_so._fetch_science_news` (story_orchestrator.py:1643) which calls `_llm_rank_news_candidates` (1817) and `_llm_rerank_with_bodies` (1908), both of which call `_generate_with_llm` (3542), which calls `_load_llm` (1974). The path fires every time the writer runs with `custom_premise` empty -- the canonical RSS auto-fetch path the workflow defaults to.
- **Cause:** The plan author skipped the grep step before claiming dead-runtime. The audit's three-step procedure (grep + vulture + module-import smoke) was correctly specified for B0 but the §2b inventory was written before the grep ran. Static analyzer reading orchestrator standalone confirms what the plan-author saw (no internal caller of `_load_llm` from within the file's top-level exports); the cross-file chain through the writer is invisible at file-scoped grep.
- **Fix:** B0 narrowed to `__init__.py` forensic-comment scrub only (one stale "registered as an alias below" comment for OTR_LedgerScriptReviewer rewritten to say the alias is DEAD per S29). B4b landed the actual rewire (2026-05-14): `_generate_with_llm` acquires its cache_entry via `_otr_model_loader.request_slot("technical", model_id)` instead of the parallel `_load_llm` call. `_run_with_timeout` invokes the new `_otr_model_loader.unload_llm()` for the timeout-recovery path. The three importers (`batch_bark_generator`, `_otr_bark_lib`, `scene_sequencer`) all switched from `from .story_orchestrator import _unload_llm` to `from ._otr_model_loader import unload_llm`. DOCUMENTED DEVIATION: the actual `_load_llm` / `_unload_llm` / `_LLM_CACHE` symbol-deletion from `story_orchestrator.py` was deferred -- the modern `_otr_model_loader.load_llm` still delegates back to `story_orchestrator._load_llm` for the actual bitsandbytes / profile-specific loader implementation (~600 LOC). Porting that body into the modern loader is its own follow-up sprint. The audit-miss fix is complete: the RSS news path no longer holds a parallel reference to the legacy cache, which was the actual structural bug.
- **Verify:** `findstr /N "_unload_llm" nodes\batch_bark_generator.py nodes\scene_sequencer.py nodes\_otr_bark_lib.py` returns 0 hits referencing `story_orchestrator._unload_llm` (the importers now point at `_otr_model_loader.unload_llm`). `findstr /N "_LLM_CACHE" nodes\story_orchestrator.py` shows the dict only at the module-level definition + inside `_load_llm`/`_unload_llm` -- no direct callers from the RSS news path or the timeout-recovery path. Canonical pytest regression holds.
- **Tags:** sprint-audit-miss, dead-runtime-claim, cross-file-call-chain, plan-vs-reality, s30
- **Bible candidate rationale:** General lesson — "dead-runtime" claims about a function must be verified by following the call chain across files, not by reading the file in isolation. The grep step the plan itself mandated would have caught this; skipping it for any function with `model_id` defaults (which signal "live LLM consumer" by convention) is the structural anti-pattern. Add to the cleanbreak playbook: "any dead-code deletion claim must cite the grep that established no cross-file callers, not just the file-local analysis."

### BUG-LOCAL-225: Bug Bible static-quality gate must run BEFORE sprint-close commit [FIXED b334b3a 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S28 Phase 5 close ran the Bug Bible regression as part of the final-verification artifacts and found 2 failures (`TestPhase02Encoding::test_no_bom_signatures` and `TestPhase12Regression::test_all_py_files_parse`) both pointing at `tools/validate_workflow_links.py:L1: invalid non-printable character U+FEFF`. The file carried a UTF-8 BOM (EF BB BF) at offset 0 that violates CLAUDE.md's "UTF-8, no BOM. Always." prime directive. Comparing `git show s27-cleanbreak-tail:tools/validate_workflow_links.py` confirmed the BOM was already present at the branch cut — inherited from earlier, not introduced by S28. The plan's Phase 5 §acceptance criterion `Bug Bible: 23 passed, 1 skipped, 2 xfailed` would have failed silently had the close shipped without re-running the Bug Bible.
- **Cause:** Per-sprint regression batteries (the targeted file lists per phase) cover the SURFACES the sprint touches; they don't cover repo-wide static-quality invariants like "no BOM signatures." The Bug Bible regression is the repo-wide static-quality gate, and S28's plan only invoked it once in Phase 5. The BOM had survived S25/S26/S27 closes because none of those sprints touched `tools/validate_workflow_links.py` and none invoked the Bug Bible repo-wide audit during their phase-by-phase regressions either. The chain inherited the issue through every sprint until S28 surfaced it.
- **Fix:** S28-p5 — stripped the 3-byte BOM prefix via `python -c "b=open(p,'rb').read(); open(p,'wb').write(b[3:] if b.startswith(b'\xef\xbb\xbf') else b)"`. File content byte-identical aside from the BOM. Re-run Bug Bible: 23 passed, 1 skipped, 2 xfailed (matches plan acceptance). Commit `b334b3a` carries the fix + the audit trail (verified pre-existing at s27 HEAD).
- **Verify:** `python -c "print(open('tools/validate_workflow_links.py','rb').read(3))"` returns `b''` (or any non-BOM prefix). `pytest comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -q` → `23 passed, 1 skipped, 2 xfailed`.
- **Tags:** bom, encoding, static-quality, sprint-close-gate, bug-bible, inherited-condition
- **Bible candidate rationale:** General lesson — a quality gate's PASS evidence must include the repo-wide static-quality artifact (Bug Bible regression), not just the per-phase targeted-suite output. Pair with BUG-LOCAL-223: where 223 says "the full pytest summary line must be in audit-results.md," 225 says "the Bug Bible regression summary line must be in audit-results.md too." Add to the cleanbreak playbook: every Phase 0 baseline runs the Bug Bible alongside the targeted-suite pytest; every Phase 5 close re-runs it and the delta vs Phase 0 must be empty. If the close commit inherits a Bug Bible failure from the branch cut, fix it in the close commit (it's a one-line edit and CLAUDE.md prime directives apply repo-wide regardless of sprint scope).

### BUG-LOCAL-224: OTR_LedgerScriptWriter silent fallback masked Tier 3 #22 polish regression [FIXED e4e3c10 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S28 Phase 3 producer audit (`docs/2026-05-13-S28-producer-audit-b4.md`) surfaced that `OTR_LedgerScriptWriter.py:1495-1506` wrapped `_OTRML.make_polish_generate_fn(cache_entry)` in a `try/except` that fell back to `polish_generate_fn = None` with a WARNING log on factory failure. Downstream, `compose_line`'s polish path at `_otr_line_composer.py:1265` substituted the writer's main `generate_fn` (composer-tuned: min_p / repetition_penalty / top_p baked into the closure for long-form generation) whenever `polish_generate_fn` was None — re-injecting the EXACT awkward-substitution regression the dedicated polish factory was added to prevent in LFC commit 12.2 (Tier 3 #22: "They closed the cell with the specimen" → "They sealed the cell with the specimen"-style random word swaps on polish rewrites).
- **Cause:** The factory `make_polish_generate_fn` was introduced in S24 as required v2.0 infrastructure (the `_otr_model_loader` was rebuilt around shared-cache_entry derivative fns at that point); the writer-side `try/except` was kept as best-effort framing from before the factory landed. Comment at the site explicitly framed the None branch as "if the factory isn't available on older builds, falls back to None and compose_line uses generate_fn for polish (pre-W4 behaviour)." Older builds don't exist on v2.0-alpha. The fallback was producer-side legacy debris that silently re-triggered the original regression whenever the factory raised — and `_OTRML` is shared-state across nodes, so any unrelated loader hiccup could cascade into degraded polish output without surfacing anywhere except a WARNING log in the boot output.
- **Fix:** S28-p3-producer-1 — dropped the writer-side `try/except`. `polish_generate_fn = _OTRML.make_polish_generate_fn(cache_entry)` is now unconditional. A factory failure surfaces as a hard ComfyUI node error from the script-writer (the correct behaviour — polish is not optional under the v2.0 contract; awkward sampling on a polish path is a silent quality regression). Test `tests/test_lfc_w4_writer_polish_fn.py::test_polish_fallback_logged_at_warning` flipped under Rule C to `test_polish_factory_call_is_unconditional`, asserting the legacy fallback log message no longer appears in the writer source. Commit `e4e3c10`.
- **Verify:** `git grep -n "make_polish_generate_fn unavailable" nodes/` returns zero hits. `pytest tests/test_lfc_w4_writer_polish_fn.py -q` → 6 passed.
- **Tags:** producer-leak, silent-fallback, polish-regression, tier-3-22, voice-path-cleanbreak, s28, awkward-substitutions
- **Bible candidate rationale:** General lesson — when a sprint introduces "required v2.0 infrastructure," the consumer-side `try/except`-with-WARN-log defensive fallback against the pre-required-infrastructure shape is producer-side LEGACY DEBRIS that re-triggers the very regression the new infrastructure was added to prevent. Audit every `try/except ... = None` wrapper around required-v2.0 factory calls; if the factory is required, the wrapper is debris. Pairs with BUG-LOCAL-218 (silent repair contradicted documented loud-fail behavior): both are "producer accepts the old shape silently" patterns; both mask the very regression the upstream fix targeted.

### BUG-LOCAL-223: Sprint Phase 4 must run pytest, not just delta `EXPECTED_FAILED_NODEIDS` [FIXED — s26-downstream sweep general lesson 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S26's final QA review claimed `regression delta byte-identical` (baseline and final `EXPECTED_FAILED_NODEIDS` sets matched). The next-morning s26-downstream sweep showed the actual full-suite pytest run had **7 failures** not in any documented quarantine. The byte-identical-delta gate had passed cleanly while real test failures shipped under it. The gap: the byte-identical claim referred to the expected-fail SET, not to the actual run output. A sprint can satisfy "no new entries in EXPECTED_FAILED_NODEIDS" without ever running pytest end-to-end.
- **Cause:** S26's Phase 4 verification used the `[KNOWN-FAIL-GUARD]` summary captured during earlier phases as the regression artifact. That summary is only emitted when the conftest hook actually fires; if the cleanbreak commits' direct test runs all targeted subsets (each subset trips the 80%-collected guard and the diff returns early), the full-suite hook never runs against the cleanbreak HEAD. The S26 reviewer accepted the targeted-subset coverage as "regression delta" evidence — a category error.
- **Fix:** Phase B downstream sweep ran the full suite from `5bf9d3a`, surfaced 7 real failures (6 in the targeted-baseline file list + 1 missed by the file list -- legacy-token scan vs interior docstring), classified each per the directive's table, and shipped 4 fixes (commits `ba8a02e`, `a70aeb8`, `8181950`, `39b1670`). End-state: 2159 passed, 8 skipped, 0 failed, empty `EXPECTED_FAILED_NODEIDS`, zero `[KNOWN-FAIL-GUARD]` lines on full-suite re-run. The 7 failures had been latent on `s26-cleanbreak` HEAD; nothing in S27 caused them.
- **Verify:** Future Phase 4 verification must satisfy BOTH of: (1) `EXPECTED_FAILED_NODEIDS` delta empty AND (2) the audit-results doc contains the actual `============ N passed, M skipped, K failed ============` summary line from a full-suite run at the cleanbreak HEAD. (1) alone is not a gate.
- **Tags:** sprint-verification, expected-fail-vs-actual-fail, full-suite-gate, regression-delta, s26-downstream, general-lesson
- **Bible candidate rationale:** General lesson -- a quality gate's PASS evidence must be the artifact the gate actually defends against, not a proxy for it. `EXPECTED_FAILED_NODEIDS` defends against silent shifts in the known-fail SET; it does NOT defend against the suite acquiring new failures that aren't yet in the set. The full-suite pytest summary line is the artifact that defends against the latter. Both must appear in audit-results.md for the regression-delta gate to be satisfied. Add to the cleanbreak playbook: "Phase 4 pass gate requires actual `N passed / M failed` summary, not just delta-vs-expected-fails." Pairs naturally with BUG-LOCAL-221's bible lesson ("any quality gate that surfaces regressions must surface the classification evidence in the same artifact") -- gates demand both the result and the evidence in their final artifact.

### BUG-LOCAL-222: Audit-completeness signal: zero-hit grep on changed surfaces is the gate, not the audit [FIXED — S26 cleanbreak general lesson 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S26 cleanbreak A3 (`production_ledger.py` sfx schema scaffold deletion) shipped a single-line code change with a clean per-commit grep audit. The follow-up zero-hit acceptance check (`git grep -nE "['\"]sfx['\"]: \[\]"`) surfaced 17 test fixtures still constructing ledger dicts with `"sfx": [],` baked in -- and one of those fixtures triggered failures via a *separate* validator at `nodes/_otr_ledger_freeze.py::_REQUIRED_TOP_LEVEL_LISTS` that ALSO required the now-deleted top-level key. The audit completed correctly on the first surface and missed two coupled surfaces because the initial grep was narrow.
- **Cause:** The audit checklist focused on the named deletion target; the broader pattern audit (every place the soon-to-be-deleted shape appears) was deferred to "after the commit." When the broader grep ran, it surfaced a contractual validator that mirrored the deleted shape -- a hidden coupling that would have caused runtime errors in any pipeline run that walked the freeze-cascade audit on a v2 ledger missing the legacy field.
- **Fix:** S26-A3 extended in-commit to: (a) drop `"sfx"` from `_REQUIRED_TOP_LEVEL_LISTS`, (b) update the freeze docstring schema mapping, (c) drop `"sfx"` from 3 parametrize lists in the gap-audit test, (d) mechanically scrub all 17 test fixtures. Blast radius 19 files; below the §5 circuit-breaker bound; architectural surface unchanged.
- **Verify:** Pre-amend: `pytest tests/test_lfc_phase_0_10_gap_audit.py` 20 failed. Post-amend: 0 new failures. `git grep -nE "['\"]sfx['\"]: \[\]" nodes/ tests/` → 0 hits.
- **Tags:** audit-completeness, zero-hit-grep, validator-mirror, schema-cleanbreak, ledger
- **Bible candidate rationale:** General lesson per S25 post-mortem pattern #1 -- whenever a deletion is about a shape (not just a symbol), the audit must enumerate every code path that produces, consumes, or validates that shape. The fastest single-pass discipline is: BEFORE deletion, run a zero-hit grep across the broadest pattern (single quotes + double quotes + variable-keyed access + validator constants). If hits are non-zero, the deletion's blast radius is the actual blast radius, not the optimistic one. The validator-mirror surface (`_REQUIRED_TOP_LEVEL_LISTS`) is the most easily missed because it's a tuple of strings, not a code path that *uses* the shape; pure-data declarations don't show up in call-graph traces. Code-shape audits need string-table awareness.

### BUG-LOCAL-221: Strict-deprecation audit cannot be classified in a non-interactive cmd.exe shell [FIXED — S27 QA-5 reclassification 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S26 Phase 4 strict-deprecation audit (`pytest -W error::DeprecationWarning`) surfaced 1 NEW regression vs baseline known-fail set: `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only`. The test passes under `-W ignore::DeprecationWarning` (5.25s; 1 passed). The strict-mode traceback could not be captured because the parent cmd.exe shell terminated on every retry attempt before stdout redirection flushed -- three attempts, all behaved the same way. The audit ran successfully (148 lines of test progress + summary captured to `docs/2026-05-13-S26-deprecation-audit.txt`) but the per-failure traceback that would let us classify origin (OTR vs third-party) was never readable.
- **Cause:** Two layers. (a) Real cause of the swallowed traceback: the project's `tests/conftest.py::pytest_sessionfinish` hook calls `raise SystemExit(2)` after emitting `[KNOWN-FAIL-GUARD] NEW failures (REGRESSION)` -- and SystemExit aborts before pytest prints the FAILURES section to stdout. The cmd.exe theory in S26's writeup was wrong (cmd.exe was honest; the conftest was eating the traceback). (b) Real cause of the strict-mode failure: TWO third-party DeprecationWarnings get escalated: `pytest_asyncio.plugin:247` (unset `asyncio_default_fixture_loop_scope`) and `torchao.dtypes.uintx.__init__:1` (deprecated import path inside transformers' `AutoProcessor` import chain).
- **Fix:** S27 QA-5 (commit pending). (1) Built `docs/2026-05-13-S27-_strict_probe.py` -- a standalone pytest harness that monkey-patches `tests.conftest.pytest_sessionfinish` to a no-op before pytest collects. The FAILURES traceback then survives the run. Durable harness for any future strict-mode audit. (2) Captured both deprecation tracebacks in full at `docs/2026-05-13-S27-deprecation-audit-reclass.txt`. (3) Fixed the pytest_asyncio warning by setting `asyncio_default_fixture_loop_scope = "function"` in `pyproject.toml [tool.pytest.ini_options]` (upstream-recommended value). The torchao warning has no OTR-side fix -- OTR doesn't import torchao directly; the import chain runs through transformers' internals. Documented as `third_party_deprecation` and CLOSED. The audit harness will re-surface any new third-party deprecation that piles up.
- **Verify:** `python docs/2026-05-13-S27-_strict_probe.py` -- the FAILURES traceback now prints intact and the source of every DeprecationWarning is identified.
- **Tags:** deprecation-audit, instrumentation-gap, conftest-systemexit, strict-warning, audiogen, third-party
- **Bible candidate rationale:** General lesson -- the strict-deprecation audit's *result* line (the regression node-id) is captured even when its *traceback* is not. Plan for both. The fix wasn't a different shell; it was a different harness that survives the conftest's SystemExit(2). Add to CLAUDE.md project rules: "any quality gate that surfaces regressions must surface the classification evidence in the same artifact -- including surviving any test-harness-side abort." Otherwise the gate's pass/fail is a hand-wave; only the pass/fail and the evidence together make the gate trustworthy. Bible-pattern lesson: instrumentation completeness is part of the gate's contract, not a separate concern -- and that includes the harness itself not eating its own output.

### BUG-LOCAL-220: `_fallback/` directory had no garbage collection [FIXED f4403e6+d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24/C2's `_fallback/` redirect (correct fix for cache-poisoning on short-output renders) shipped without a cleanup hook. Across re-runs of the same episode, the `<cache_dir>/_fallback/` directory accumulated orphan silence wavs that no consumer ever read. Bounded per cue but unbounded across iterations -- effectively an unowned cache that grew until manual cleanup.
- **Cause:** The C2 fix was scoped to "stop poisoning the canonical cache_path on the short-output path." The `_fallback/` dir was framed as ephemeral but the cleanup was never wired -- a classic "we'll get to that later" oversight that lasted from S24 close to S25 open. The "soft-rollout deadlock" sibling (BUG-LOCAL-219) shares the same anti-pattern: ship the alarm wiring, defer the implementation, never wire the implementation.
- **Fix:** S25/AG-1+MG-7 -- per-episode `_fallback/` cleanup hook added immediately after `_cache_dir()` resolves in BOTH `batch_audiogen_generator.py::BatchAudioGenGenerator.generate()` AND `musicgen_theme.py::MusicGenTheme.render()`. Wipes stale `.wav` entries and logs `_fallback/ cleanup: removed N stale wav(s)` to batch_log / render_log when N > 0.
- **Verify:** Manual: drop a file in `output/otr/episodes/<ep>/audio/_fallback/foo.wav`, run AudioGen, confirm the file is gone and the log line fires.
- **Tags:** cache-cleanup, ephemeral-dir, c2-followup, audiogen, musicgen, soft-rollout-debt
- **Bible candidate rationale:** General lesson -- whenever a fix introduces a "this is ephemeral" surface (cache, scratch dir, temp file), the cleanup hook lands in the same commit. "We'll get to that later" cleanup hooks accumulate across sprints and create unowned cache surfaces that no one notices until disk fills up.

### BUG-LOCAL-219: "Soft-rollout never flipped" deadlock [FIXED 9afa54a+f592d71 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `audit_post_freeze_writeback` shipped at S18.2 with the docstring "use strict=False for the soft-rollout phase -- consumers log violations to batch_log." No consumer ever called it. `ProcSFX.strict_writeback` defaulted to False at S18.3 with the flip-criterion "once the audit walker has stayed clean for one full pipeline run" -- but the walker was never running, so the criterion was unreachable. Net: a Phase-0 alarm shipped as "off but ready to flip on once it proves itself clean", the path to "prove itself clean" required the consumer to call it, the consumer never did, and the audit slept for five sprints.
- **Cause:** Two safety surfaces shipped with flip-criteria that referenced each other in an unreachable cycle. Neither shipped with an inline owner; neither sprint after S18 audited whether the criteria had been met. The "soft rollout" framing made each individual half look like work-in-progress rather than the deadlock it was in aggregate.
- **Fix:** S25/AG-5..9 -- wired `audit_post_freeze_writeback` in soft mode at all three line-writing consumers (AudioGen, MusicGen, ProcSFX; VideoComposite writeback doesn't touch any audited line field so it's documented N/A). Flipped `ProcSFX.strict_writeback` default to True in the same sprint -- with the walker actually running, the criterion is now satisfiable and the strict default is honest about what the production contract is.
- **Verify:** `pytest tests/test_procsfx_writeback_convention.py -v` (10 passed; the two strict-default pins now lock True). Grep audit: `grep -rn 'audit_post_freeze_writeback' nodes/ --include='*.py' | grep -v _otr_ledger_consumers.py` returns 3 active call sites.
- **Tags:** soft-rollout, deadlock, audit-walker, flip-criterion, ownerless-defer
- **Bible candidate rationale:** General lesson -- any feature shipped behind a "soft rollout" flag MUST include (a) an inline flip-criterion that is *checkable* from the current commit's state, and (b) a named owner with a sprint deadline. Without both, "soft rollout" deterministically becomes "permanent off" because each sprint's planning pass treats it as "already shipped, not my problem." If the criterion references "the audit walker stays clean for one run", the same commit MUST wire the walker -- otherwise the criterion is unreachable and the flag is dead.

### BUG-LOCAL-218: Silent `model_id` repair contradicted loud-fail comment [FIXED d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `batch_audiogen_generator.py:293-294` silently mapped `str(model_id) in {"3", "3.0"}` to `"facebook/audiogen-medium"` while the INPUT_TYPES comment at `:255-259` explicitly stated "Fail loudly on bad input." The widget vector that originally triggered the drift (BUG-LOCAL-027) was already cleaned at S24/C3 -- the runtime repair had no production case left to defend against AND was masking misconfiguration AND contradicted the documented loud-fail behavior.
- **Cause:** When BUG-LOCAL-027 was fixed at the root by the C3 widget vector realignment, the downstream defender wasn't audited and deleted. It accumulated as silent-repair debris -- a code block whose triggering condition was fixed upstream months earlier.
- **Fix:** S25/AG-4 -- deleted the active `if str(model_id) in ["3", "3.0"]: model_id = "facebook/audiogen-medium"` lines; forensic comment preserved citing the deletion sprint + tying it to the original BUG-LOCAL-027 root-cause fix at S24/C3. Updated INPUT_TYPES.optional.model_id comment to remove the contradiction: loud-fail is now the literal behavior (combo-list enforces).
- **Verify:** `pytest tests/test_audiogen_legacy_gate.py::test_model_id_silent_repair_removed tests/test_audiogen_legacy_gate.py::test_model_id_input_combo_list_intact -v`.
- **Tags:** silent-repair, defender-debris, comment-code-drift, audiogen, c3-followup
- **Bible candidate rationale:** General lesson -- when a defensive code block's triggering condition is fixed at the root, audit and delete the downstream defenders. Otherwise they accumulate as silent-repair landmines that mask the next class of misconfiguration AND contradict the code's documented "loud fail" contract. Pattern: every "fix root cause" commit should grep for downstream defenders against the original bug's symptom and prune them in lockstep.

### BUG-LOCAL-217: AudioGen legacy `ledger.sfx[]` skipped C2 gate [FIXED d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24/C2's ghost-path fix landed on the new v2 `ledger.lines[]` writeback path at `batch_audiogen_generator.py:765-775` but not on the parallel legacy `ledger.sfx[]` loop at `:696-724`. Same field (`wav_path`), same failure mode (ledger row points at a path that was never confirmed on disk). The legacy path is dead code for current v2 producers but is the contract for any external producer still emitting the legacy shape -- and an "unused" path that ships with a bug is still a bug.
- **Cause:** C2's audit framing was "fix the new v2 writeback path." The legacy parallel loop wasn't in scope. A `git grep wav_path` against `batch_audiogen_generator.py` would have caught both paths in seconds; the audit was narrower than the field surface.
- **Fix:** S25/AG-2 -- mirrored the C2 gate `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)` onto the legacy loop. Failure branch stamps `row["wav_path"] = ""` per §6.16. Added `sfx_render_status` stamping on the legacy loop too so the audit walker (S25/AG-5) sees a consistent enum surface across both paths. CD-3 audit (S25 phase 7) confirmed zero current producers populate the legacy `sfx[]` -- the gate is conservative belt-and-suspenders until the S26.X deletion lands.
- **Verify:** `pytest tests/test_audiogen_legacy_gate.py -v`. Grep audit: 2 `os.path.isfile(cache_path)` sites in `batch_audiogen_generator.py` (v2 lines[] + legacy sfx[]).
- **Tags:** parallel-path, ghost-path, c2-followup, legacy-loop, sibling-audit
- **Bible candidate rationale:** General lesson -- when a safety fix lands on path A, audit every parallel path that handles the same ledger field. The audit should be a `git grep <field>` across the entire module (or repo, depending on field scope), not a manual walk of the changed file's neighbors. The "but it's the legacy path / dead code" framing is exactly when the bug ships unnoticed because the fix author and the reviewer both skip it.

### BUG-LOCAL-216: Style slug drift surface (writer pool vs palette) [FIXED 9679217 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme._STYLE_PALETTE` (10-key dict mapping slug -> cue prompts) and `OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL` (10-tuple of slugs) were maintained as two parallel lists in two files. Drift between them caused MusicGen to halt mid-pipeline (after the script writer + freeze cascade had already spent minutes of model time + thousands of tokens) -- the writer emitted a slug that the palette didn't cover, MusicGen raised at lookup, the operator lost the run.
- **Cause:** Two surfaces, one contract, no enforcement. When a new style slug was added, the contributor had to remember to update both -- and even when they did, a future contributor renaming one entry would silently break the other.
- **Fix:** S25/MG-6 -- hoisted both sources of truth to `nodes/_otr_style_palette.py` with `STYLE_PALETTE` + `KNOWN_STYLE_SLUGS`. `musicgen_theme.py` re-imports as `_STYLE_PALETTE`; the writer pool stays its own surface but `tests/test_style_palette_drift.py` pins set-equality with `KNOWN_STYLE_SLUGS`. Freeze cascade gained an additional check in `_check_meta_invariants` that validates `meta.gen_params_initial.style ∈ KNOWN_STYLE_SLUGS` -- writer drift now surfaces at freeze time, before MusicGen even tries to look the slug up.
- **Verify:** `pytest tests/test_style_palette_drift.py -v` (5 tests: palette == known, writer pool == known, every entry has 3 cues, freeze rejects unknown, freeze accepts known).
- **Tags:** drift, source-of-truth, parallel-list, freeze-cascade, style-palette
- **Bible candidate rationale:** General lesson -- any data contract maintained as two parallel lists in two files becomes drift-prone in O(weeks). Hoist to a shared module on first drift detection (or pre-emptively if the parallel structure is visible at design time). Pin set-equality with a unit test that imports both surfaces; any future drift fires at unit-test time, not soak time.

### BUG-LOCAL-215: MusicGen NODE_CLASS_MAPPINGS prefix drift [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** `musicgen_theme.py` registered the node as `NODE_CLASS_MAPPINGS = {"MusicGenTheme": MusicGenTheme}` while `__init__.py:_NODE_MODULES` registers the same class under the canonical `OTR_MusicGenTheme` key. The in-module dict was dead code (the top-level `__init__.py` re-registers from the class object directly, not from the in-module mapping dict) -- but ANY test or external consumer that imported `NODE_CLASS_MAPPINGS` from the module directly got the bare name. Display name also carried a literal `"[EMOJI]"` placeholder string.
- **Cause:** The OTR_ prefix migration touched the registration site in `__init__.py` but not the leftover in-module declarations on each node file. No test pinned the in-module dict's key to match the top-level registration.
- **Fix:** S25/MG-5 -- aligned the in-module mapping to `{"OTR_MusicGenTheme": MusicGenTheme}`, dropped the `"[EMOJI]"` placeholder string from the display name. New regression test `test_musicgen_parity.py::test_node_registered_under_otr_prefix` pins the prefix; `test_node_display_name_has_no_placeholder` pins the no-placeholder rule.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_node_registered_under_otr_prefix tests/test_musicgen_parity.py::test_node_display_name_has_no_placeholder -v`.
- **Tags:** node-registration, prefix-drift, dead-code, display-name
- **Bible candidate rationale:** Cosmetic in isolation; the broader lesson (any registration surface stamped in multiple files needs a test pin) is covered by the general drift-pattern entries (BUG-LOCAL-216, IMP-43).

### BUG-LOCAL-214: Silence fallback ignored cue duration [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme._silent_audio_dict(sample_rate)` emitted a fixed `int(sample_rate * 0.1)`-sample clip regardless of cue. On the ImportError + `allow_silence_fallback=True` path, the 12-second opening cue and the 8-second closing cue both got 100 ms of silence -- which then propagated into EpisodeAssembler and broke the timeline (opening theme slot was 100 ms of silence then a 11.9s gap to the dialogue, closing theme was 100 ms then nothing). The bug shipped silently because the fallback path is rarely exercised (transformers is installed in production).
- **Cause:** Helper was a one-liner written under the assumption "we only need a brief placeholder." The signature didn't take a duration; every caller passed nothing; nobody noticed the EpisodeAssembler downstream needed real per-cue durations.
- **Fix:** S25/MG-4 -- `_silent_audio_dict(duration_sec, sample_rate=MUSICGEN_SAMPLE_RATE)` -- duration is now required. The ImportError fallback loop passes `CUE_DURATIONS[cue_id]`. Test `tests/test_musicgen_parity.py::test_silent_audio_dict_honors_duration` pins the contract.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_silent_audio_dict_honors_duration -v`.
- **Tags:** silence-fallback, duration, musicgen, timeline, rarely-exercised
- **Bible candidate rationale:** General lesson -- a rarely-exercised code path (transformers ImportError on a box where transformers IS installed) is exactly the kind of fallback that ships with sloppy semantics for years because nobody hits it in soak. Audit fallback paths for "does this honor every contract the success path honors?" -- in this case, the cue duration is part of the EpisodeAssembler timeline contract, and the fallback emitted nonsense.

### BUG-LOCAL-213: MusicGen `music_render_status` documented but never written [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme.MusicGenTheme.INPUT_TYPES` docstring and the ImportError fallback comment both promised stamping `music_render_status="fallback_silence"` on each affected ledger row. No code path actually wrote the field. Compounded by an early `return` in the ImportError fallback that bypassed the writeback block entirely -- so even if the field had been stamped in `cues[cue_id]`, the writeback never ran on the fallback path.
- **Cause:** Two bugs in one surface: (1) the writeback block didn't include `row["music_render_status"]` (it stamped wav_path, dur_s, tts_engine, etc., but not the status enum), and (2) the ImportError branch returned early before the writeback block could fire. Both bugs are easy to introduce when "soft path returns early" looks like a defensive shortcut.
- **Fix:** S25/MG-3 -- writeback block now always stamps `row["music_render_status"] = str(cue.get("_render_status") or "ok")`. ImportError branch refactored to fall through to the writeback block (added `else:` clause on the try/except so the model-loading code only runs when the import succeeded, then the writeback fires on whatever shape the cues are in). Audit walker (`audit_post_freeze_writeback`) gained `ALLOWED_MUSIC_RENDER_STATUS` enum check so typos surface.
- **Verify:** Source-level pin in `tests/test_post_freeze_writeback_audit.py` via `ALLOWED_MUSIC_RENDER_STATUS`. End-to-end pin via the wired-walker calls in Phase 5.
- **Tags:** comment-code-drift, early-return, render-status, enum, musicgen
- **Bible candidate rationale:** General lesson -- comments promising ledger behavior must be exercised by an acceptance test in the same commit. Otherwise the documentation drifts from the code and becomes a silent contract drift that future contributors believe is honored. Pattern: when a docstring says "stamps X", the same commit should add a test `assert "X" in ledger_row_dict`. Belt-and-suspenders: include the field in `audit_post_freeze_writeback`'s field list so the soft-mode walker fires on any consumer that drops the stamp.

### BUG-LOCAL-212: MusicGen writeback ghost-path [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Sibling of S24/C2's AudioGen ghost-path. `musicgen_theme.py:771` stamped `row["wav_path"] = str(cache_path)` unconditionally whenever `cache_path` was a truthy string. Cache_path was set from the canonical cache filename builder regardless of whether the save eventually succeeded -- so on `_save_wav` failure (BUG-LOCAL-211), short-output fallback, or ImportError silence-fallback, the ledger row pointed at a path that was never confirmed on disk.
- **Cause:** Same root cause as BUG-LOCAL-209 (AudioGen): writeback gated on the string variable instead of the save outcome. The implicit assumption was "if we computed a cache_path, the file is there." That's only true on the happy path; the ImportError + short-output + disk-failure paths all violate it.
- **Fix:** S25/MG-2 -- writeback now reads `save_ok = bool(cue_dict.get("_save_ok"))` and `had_cache_hit = bool(cue_dict.get("_had_cache_hit"))` and gates `row["wav_path"] = str(cache_path)` on `cache_path AND (save_ok OR had_cache_hit) AND os.path.isfile(cache_path)`. Failure paths stamp `row["wav_path"] = ""` per §6.16. The cache-hit branch in the resolve loop also stamps `cue["_had_cache_hit"] = True` so the gate distinguishes a fresh-save from a load-from-disk hit.
- **Verify:** Source-level pin via `tests/test_workflow_audio_widget_vectors.py` (BUG-LOCAL-210 sibling test) + `tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_success/failure` (which gates the upstream save outcome). End-to-end: any production run with the ImportError fallback now produces `row["wav_path"] = ""` instead of a ghost path.
- **Tags:** ghost-path, writeback, c2-sibling, musicgen, save-proof
- **Bible candidate rationale:** General lesson -- already covered by BUG-LOCAL-209's promotion. The Bible entry that lands from #209 should explicitly enumerate "audit every sibling consumer with the same shape" so #212 doesn't ship in S25 the way it did. The pattern audit needs to run BEFORE the bible promotion, not after.

### BUG-LOCAL-211: MusicGen `_save_wav -> None` [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Direct sibling of BUG-LOCAL-209 (AudioGen). `_save_wav` in `nodes/musicgen_theme.py:281` declared `-> None`. Success path fell off after `os.replace(tmp, path)`; the except path fell off after `log.warning(...)`. Both returned implicit None. Callers couldn't distinguish a confirmed write from a swallowed exception. The writeback path keyed on `cache_path` (the string variable, always truthy after the filename builder) instead of the save's outcome -- so any save failure left a ledger row pointing at a path that was never written. BUG-LOCAL-212 is the immediate downstream consequence.
- **Cause:** Same implicit-None bug as BUG-LOCAL-209 -- function signature declared `-> None`, both code paths just ran off the end and returned None. The function was originally written as fire-and-forget (only the log.warning mattered on failure) but the writeback path quietly became a consumer of its outcome.
- **Fix:** S25/MG-1 -- signature changed to `-> bool` with explicit `return True` after `os.replace` and `return False` from the except branch. The render path captures `save_ok = _save_wav(...)` and stores it in `cue["_save_ok"]`. Writeback gates `wav_path` stamping on `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)` per §6.16.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_success tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_failure -v`.
- **Tags:** implicit-none, save-proof, audiogen-sibling, musicgen, bug-209-mirror
- **Bible candidate rationale:** General lesson -- the BUG-LOCAL-209 Bible entry (when it promotes after v2.0 ships) should explicitly include an audit step "grep for `-> None` on every save-style function whose callers check truthiness" -- across the whole repo, not just the consumer that triggered the original entry. The sibling-audit gap that let #211 ship five sprints after #209 is the real lesson here; the per-function fix is mechanical once the audit fires.

### BUG-LOCAL-209: AudioGen `_save_wav` returned None on both success and failure paths [FIXED 2002958 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** C2 audit of nodes/batch_audiogen_generator.py:200 found `_save_wav` declared `-> None`. The success path fell off after `os.replace(tmp, path)`; the except path fell off after the warning log. Both returned `None`. The writeback block at L688-720 unconditionally stamped `sfx_wav_path = item["cache_path"]` whenever `cache_path` was truthy, with no proof the file actually existed on disk. Net: the ledger could carry a sfx_wav_path pointing at a path that was never written.
- **Cause:** Implicit-None on a function whose return value WAS being consumed. The writeback path didn't check the return; it checked the cache_path string variable, which was set regardless of save outcome.
- **Fix:** C2 -- _save_wav signature changed to `-> bool` with explicit `return True` after os.replace and `return False` from the except branch. The render-path now captures `save_ok = _save_wav(...)` and stores it in `item["_save_ok"]`. The writeback gates `sfx_wav_path` stamping on `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)`. Failure paths stamp `sfx_wav_path=""` per §6.16. 3 source-level pins in tests/test_audiogen_writeback_hardening.py.
- **Verify:** `pytest tests/test_audiogen_writeback_hardening.py::test_save_wav_returns_bool tests/test_audiogen_writeback_hardening.py::test_writeback_gates_sfx_wav_path_on_save_proof -v`.
- **Tags:** silent-failure, return-value, save-proof, sfx_wav_path, audiogen
- **Bible candidate rationale:** General lesson -- when a function's return value is consumed by a contract (in this case, "did the write succeed?"), the function must return an explicit bool, not implicit None. Implicit-None on a function whose callers branch on the return is silent-failure scaffolding. Audit `-> None` declarations on functions whose callers check truthiness.

### BUG-LOCAL-208: `visual/bridge.py` carried a live `production_plan_json` socket [FIXED b443f46 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S15.5.1 audit surfaced `visual/bridge.py:270` -- the OTR_VisualBridge node declared `production_plan_json` as an optional `STRING` input in INPUT_TYPES, the execute() signature accepted it as `production_plan_json: str = "{}"`, and the body wrote the value to `<job_dir>/production_plan.json` via `atomic_write_text`. Grep across the sidecar / visual worker confirmed NO downstream consumer read the file -- the bridge wrote it for an audience that no longer existed.
- **Cause:** When the legacy LLMDirector was deleted in voice-path-cleanbreak S2 (commit 249bc06) the Director's outputs were no longer being produced anywhere upstream of the bridge. The bridge's optional socket survived because S2 scoped to the audio path, and the visual bridge is sidecar-isolated -- the deletion wave didn't reach this side of the repo until the S15.5.1 audit.
- **Fix:** S23.7 -- deleted the INPUT_TYPES entry, the kwarg from execute()'s signature, and the atomic_write_text(production_plan.json) call. Module + class docstrings rewritten to reflect "script_json + scene_manifest_json" as the actual input contract. Forensic comment at the deletion site cites S23.7 + directive 11.
- **Verify:** `git grep -n "production_plan_json" visual/` returns zero hits.
- **Tags:** legacy-socket, directive-11, audit-found, sidecar-isolation, voice-path-cleanbreak
- **Bible candidate rationale:** Bookend to BUG-LOCAL-207 -- when a deletion wave is scoped to one subsystem, sidecar-isolated subsystems can carry the deletion's debris forward for sprints. A repo-wide audit grep at the END of every cleanbreak (not just inside the affected subsystem) catches this class of survival.

### BUG-LOCAL-229: Sprint H bug-hunt §3.7 closure -- workflow widget drift + ComfyUI client-side serialization rules + uv launcher-stub PID split [FIXED 2026-05-17 across multiple commits 51a8f56..5b44e65]
- **Date:** 2026-05-17 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** Sprint H §3.7 attended forced-kill validation of the bug-hunt supervisor + worker harness. Each iter the worker self-exited at the API-conversion stage before reaching mid-execution, refusing to POST a graph it considered malformed. The harness was caught between two correct fail-loud behaviors and one structurally infeasible OS check: (a) the worker's converter raised on workflow widget drift before submission, (b) the supervisor's between-iter sweep was killing its own PID under the uv launcher-stub model, (c) the atexit-under-forced-kill guarantee on Windows is unenforceable because `taskkill /F` is SIGKILL-equivalent. Each surface had to be fixed before the §3.7 GREEN bar could be hit.
- **Cause:** Three independent root causes, all surfaced by the bug-hunt loop itself once the harness started running real workflows through the converter:
  1. **uv launcher-stub PID split.** The venv's `python.exe` (`C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`) is a uv stub that fork-execs the real cpython at `%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe`. Both processes report `Name='python.exe'` to WMI; both carry the supervisor's `argv` in their CommandLine. `os.getpid()` returns the real-cpython PID, `subprocess.Popen(...).pid` returns the stub PID, and netstat reports the real cpython as the TCP socket owner. Pre-fix the worker captured the stub PID for ComfyUI's `_confirm_port_owner` check and refused every healthy launch as "PID mismatch"; pre-fix the supervisor passed only its `os.getpid()` to the between-iter sweep and the WMI-enumerated stub got killed alongside Windows-MCP -- supervisor died at the iter-1 -> iter-2 boundary.
  2. **ComfyUI client-side serialization rules not encoded in /object_info.** Two rules the live schemas don't make explicit:
     - INT widgets named `seed` / `noise_seed` carry a hidden `control_after_generate` companion slot in `widgets_values` (Reading C). Pre-fix the patcher's length check refused workflows that had the companion -- e.g. node 1 (OTR_LedgerScriptWriter): saved length 19 vs schema-declared 18.
     - Widget-backed inputs flagged `forceInput=True` are socket-only -- they never occupy a `widgets_values` slot even when their TYPE is widget-backed (Reading D). Pre-fix the mapper counted them, e.g. node 20 (OTR_VideoPlan): expected 7, saved 6, refused.
  3. **Workflow widget drift -- five distinct stale-save defects across both workflow JSONs.** Same family as BUG-LOCAL-210 (AudioGen stale `{}`) but on five additional nodes; the bug-hunt loop surfaced each one sequentially as the upstream blockers cleared:
     - Node 12 (OTR_SignalLostVideo): stale `'{}'` at slot 2, len 6 -> 5 (Commit A 51a8f56).
     - Node 3 (OTR_SceneSequencer): stale `'{}'` at slot 1, len 8 -> 7 (cead3eb).
     - Node 11 (OTR_BatchBarkGenerator): stale `'{}'` at slot 1, len 3 -> 2 (5d78335).
     - Node 14 (OTR_MusicGenTheme): `script_json` flagged `forceInput=True` post-save; stale placeholder at slot 0, len 5 -> 4 (9310213).
     - Node 59 (OTR_BatchFluxPortraitRender): seed companion auto-injection landed in ComfyUI's UI after this workflow was last saved; the saved layout has the linked `ledger_json` placeholder preserved but no companion slot. Inserted `"fixed"` at slot 10, len 11 -> 12 (5b44e65).
- **Fix:** Seven commits on `v2.0-alpha`, all narrow:
  - `51a8f56` H A: drop orphan `{}` from OTR_SignalLostVideo node 12 (Commit A).
  - `097e832` H B1: failure-path orphan-model cleanup in loader (request_slot + load_llm except).
  - `e5d5105` H B3: workflow clone `_bughunt.json` with Gemma-4-E4B-it defaults (no C7 mutation).
  - `1f7240f` H C1: ship four manual-operator scripts (sweep + launcher + isolation test + normalize header).
  - `9e2888d` H C2: queue_smoke act_count=1 (B1 followup).
  - `035aedc` H D1: sweep_and_launch.bat + sweep_python_excluding.bat (variadic keep-list).
  - `1b2b064` H D2: worker_iter.py (one-iter worker, direct subprocess.Popen ComfyUI child).
  - `d9babea` H D3: refactor overnight_bug_hunt.py to two-process supervisor.
  - `126efd1` H E (A+B): uv-stub PID-split fix -- worker readiness JSON-shape check + socket.bind port preflight + dual-PID supervisor capture + sweep bat CSV-env-var + typed `[int]` cast on both sides of `-notcontains`.
  - `c2c06e9` H §3.7 unblock C: companion-aware `_serialized_slot_names` + `patch_widget_by_name` (Reading C).
  - `8df3d0a` H §3.7 unblock C': `workflow_to_api_prompt` honors seed companion + `submit_prompt` node_errors guard + `_classify_failure` case-insensitive `status` + `executed_count == 0 -> graph_widget`.
  - `cead3eb` H §3.7 unblock D: drop orphan `{}` from OTR_SceneSequencer node 3.
  - `5d78335` H §3.7 unblock E: drop orphan `{}` from OTR_BatchBarkGenerator node 11.
  - `7ecbd53` H §3.7 unblock F: serialized slots skip forceInput widget specs (Reading D).
  - `9310213` H §3.7 unblock G: drop stale placeholder from MusicGenTheme node 14 (forceInput-added-post-save).
  - `5b44e65` H §3.7 unblock H: add missing seed companion to FluxPortrait node 59 (companion auto-injection landed post-save).
- **Verify:**
  - Unblinded mini-audit (folder_paths-stub, all 34 OTR node classes auditable, runtime-walk pass replicates the converter) returns zero mismatches in both `workflows/otr_scifi_16gb_full.json` and `workflows/otr_scifi_16gb_bughunt.json`.
  - Companion + forceInput tests: `pytest tests/test_otr_api_companions.py` -> 17/17 pass.
  - C7 audio baseline gate: `pytest tests/test_workflow_canonical_baseline.py` -> 8/8 pass.
  - Workflow guardrails + wiring invariants: `pytest tests/test_workflow_json_guardrails.py tests/test_workflow_json_wiring_invariants.py` -> 45p / 2s.
  - Bug Bible regression: 23p / 1s / 2xfailed baseline held.
  - Full `pytest tests/` walk: 2391+ passed, 21 skipped, 0 failed.
  - §3.7 forced-kill rubric: worker reaches mid-execution (GREEN, iter 1 wall=112.5s with ComfyUI got prompt at PID 60716 -> child 24628); `taskkill /F /PID worker /T` drops worker + ComfyUI tree without orphans (GREEN); supervisor survives via dual-PID keep-list (GREEN, 7th confirmation); supervisor synthesizes `worker_crash` row when atexit cannot fire under `/F` (operational GREEN, see caveat).
- **Tags:** sprint-H, bug-hunt-harness, two-process-supervisor, uv-launcher-stub, pid-split, companion-mapper, forceInput, widget-drift, comfyui-serialization, atexit-windows, reading-c, reading-d
- **Bible candidate rationale:** Three general lessons, all repo-portable:
  1. **uv launcher-stub model breaks naive PID assumptions.** Any caller that does `subprocess.Popen(<venv python>).pid` and compares against `os.getpid()` / WMI / netstat is wrong on a uv-managed venv. Capture both `os.getpid()` AND `os.getppid()` for self-identification; pair every PID-targeted operation with a stub-aware enumeration that walks ParentProcessId. The fix shape (variadic keep-list, dual-PID capture, typed `[int]` cast on both sides of `-notcontains`) generalizes.
  2. **ComfyUI client-side serialization rules are model invariants, not opt-in features.** Two rules surfaced here (`control_after_generate` companion, `forceInput=True` slot omission); /object_info hints at both but doesn't make the slot-count consequence visible. Map them explicitly in `_serialized_slot_names`; fail loud on length mismatch; reject companion-vocab drift. Future rules added to ComfyUI's UI will manifest as new length mismatches the loud-fail check catches.
  3. **Workflow-JSON widget vectors silently drift when INPUT_TYPES mutates.** Any deletion or `forceInput` flip on a node's INPUT_TYPES requires re-saving every workflow JSON that uses that node, OR the patcher / converter must reject the legacy save. We have the loud-fail check now; pair it with the unblinded mini-audit (folder_paths-stub for runtime-only nodes + runtime-walk replication of the converter) at every workflow-touching sprint close. The mini-audit pattern itself is the durable artifact -- BUG-LOCAL-210's lesson scaled up to a sweep tool.
- **Sprint H §3.7 check-#3 caveat:** Windows `taskkill /F /T` is SIGKILL-equivalent. Python's `atexit` registers handlers that run on normal exit, signal handlers (SIGTERM-like), or unhandled exceptions -- NOT on forced termination. The strict atexit-under-forced-kill reading of check #3 is structurally infeasible on Windows. The supervisor's defense-in-depth (synthesize a `worker_crash` row when `logs/worker_iter_<N>.json` is missing on worker exit) handles every forced-death class (taskkill /F, segfault, OOMKill, BSOD) with the same outcome. Operational-GREEN is the production-relevant reading; a heartbeat-snapshot follow-up was discussed and deferred.

### BUG-LOCAL-230: FLUX1-dev-fp8 forced to fp16 by `--force-fp16` launch arg [FIXED 16ce225 2026-05-18; runtime axis verified at 4c887e0 + sampler smoke 23:30 (1.22-1.48 s/it across 4 FLUX render calls)]
- **Date:** 2026-05-18 | **Phase:** 5-6 | **Bible candidate:** yes
- **Symptom:** §3.7 closure run reached FLUX sampler entry exactly as designed (audio sealed at 153.43 s, FLUX deferred fire + branch gate + Sprint D cleanbreak all green) then stalled at the sampler. Direct telemetry from `logs/comfy_session_iter_001.log`:
  ```
  model weight dtype torch.float16, manual cast: None
  [DeferredCheckpointLoader] load complete: 2.13 -> 24.30 GiB (delta=22.17)
  FLUX model fully loaded (22700.13 MB)
   5%|1/20 [09:24<2:58:54, 564.99s/it]
  ```
  Sampler ran at ~9.4 min per diffusion step (vs expected ~10-15 s/step on this RTX 5080 Laptop / 16 GB rig). LibreHardwareMonitor during the sample: GPU memory pinned at 16240/16303 MB, D3D Shared Memory 10445 MB (offloader paging), ComfyUI RSS 19.5 GB. ETA at the observed pace was 161 min for the bookend image alone + ~9 hr for PASS1=3 portraits + LTX + HuMo + ffmpeg unreached -- pipeline blew the 3.5 hr exec timeout by 3x.
- **Cause:** `--force-fp16` global ComfyUI launch arg overrides the checkpoint's native dtype at load time and casts every weight to `torch.float16`. `flux1-dev-fp8.safetensors` is a natively-fp8 Comfy-Org quantized checkpoint that loads at ~11 GiB native; the fp16 upcast doubles the footprint to ~22 GiB. On a 16 GB card the resident-VRAM budget cannot hold the full checkpoint plus the FLUX CLIP text encoder (~4 GiB) plus the sampler working set, so the dynamic offloader pages weights to RAM/pagefile per sampler step. The flag was inherited from a pre-fp8-era launcher recipe ("Tested Blackwell settings carried over verbatim") and was never re-audited when `flux1-dev-fp8.safetensors` was adopted. Diagnostic missed for ~16 hours because the architectural campaign was focused upstream of the FLUX sampler (gate sequencing, loader ordering, audio-side serialization, cleanbreak voice path); the slow sampler was misread as "weights-too-big-for-card thrashing the offloader, fix the checkpoint" rather than "weights are being upcast at load, fix the launch arg."
- **Fix:** Removed `--force-fp16` from every launcher site. Four sites total carried the flag; the initial status-12 diagnosis only named one. Fixes:
  1. `C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat` -- manual ComfyUI launch
  2. `scripts/worker_iter.py` line 549 -- overnight bug-hunt smoke worker (the closure-run launcher; supervisor `overnight_bug_hunt.py` spawns `worker_iter.py` which inline-Popens ComfyUI with these args)
  3. `scripts/start_comfy_h0_baseline.bat` line 20 + REM block at line 6-9 -- Sprint H baseline launcher, cited as the source-of-truth template for site #2
  4. `scripts/_start_comfyui.ps1` line 62 + inline comment line 51 -- Cowork helper script
  Sites #3 and #4 also got explanatory comment blocks pointing back to this BUG-LOCAL-230 entry so the next operator picking up the recipe doesn't reintroduce the flag. Pre-flight audit confirmed no fallback dtype-upcast sources: `extra_model_paths.yaml` (both install-dir + Roaming canonical), ComfyUI Desktop `config.json`, and Desktop shortcuts all clean -- the four launchers were the complete inventory.
- **Verify (2026-05-18 smoke run, killed at step 1/20 after architectural axis proved):**
  - **Architectural axis PROVEN.** Gates 1-4 + 7 PASS:
    - **Gate #1 PASS** -- post-fix PID 18444 (`uv\python.exe ... main.py --port 8000 --highvram --cuda-malloc ...`) owns :8000; full provenance chain traced (cmd.exe / start_comfy.bat -> .venv stub -> uv child); `sweep_prelaunch.log` confirms pre-fix processes killed at `21:10:27` before fresh chain came up
    - **Gate #2 PASS** -- active ComfyUI cmdline contains no `--force-fp16` across all 6 chain processes
    - **Gate #3 PASS** -- L573: `[DeferredCheckpointLoader] fire: VRAM allocated=2.13 GiB; gate_signal len=80; ckpt=flux1-dev-fp8.safetensors`
    - **Gate #4 PASS** (the load-it-yourself smoking gun) -- L574: `model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16` (pre-fix was `torch.float16, manual cast: None`) and L584: `[DeferredCheckpointLoader] load complete: VRAM allocated=2.13 -> 13.21 GiB (delta=11.08); ckpt=flux1-dev-fp8.safetensors` (pre-fix delta was 22.17 GiB); L585: `[FluxBranchGate] fire: VRAM allocated=13.21 GiB`
    - **Gate #7 PASS** -- workflow's `PathchSageAttentionKJ` widget value `"disabled"`; no SageAttention involvement in failure
  - **Runtime axis FAILING.** Gates 5-6 FAIL but for a separate defect (NOT a dtype-upcast surface):
    - **Gate #5 FAIL** -- L610: `5%|1/20 [02:34<48:46, 154.02s/it]`. Sampler step 1 took 154 s vs target ~10-15 s/step. 3.6x faster than pre-fix 564.99 s/it, but ~10x slower than target.
    - **Gate #6 FAIL** -- LHM during sampler step 1: GPU Memory Used 15911 MB (over 14.5 GiB ceiling by ~756 MB) + D3D Shared Memory Used 1098 MB. Far less than pre-fix's 10445 MB D3D Shared, but enough to slow per-step pace by ~10x vs target.
  - **Overall status:** Architectural axis CLOSED (`--force-fp16` removal works exactly as designed; fp8 weights load native; delta within 0.08 GiB of the 11 GiB prediction). Runtime axis BLOCKED on BUG-LOCAL-231 (separate VRAM-pressure / sampler-pace defect that surfaced after the architectural fix removed the dtype upcast). BUG-LOCAL-230 stays at pending-verification posture; promotion to [FIXED] gated on BUG-LOCAL-231 close + a clean re-run smoke that passes all 7 criteria. **No `[FIXED]` marker on this entry until that happens.**
- **Tags:** launch-args, dtype-upcast, flux-fp8, blackwell, vram-thrash, offloader, bible-candidate
- **Bible candidate rationale:** Generalize-able lesson is "global precision-forcing launch arg silently upcasts checkpoint-native quantized formats (fp8, fp4, NF4, INT8, GGUF). On a VRAM-constrained card the upcast can double the footprint and force the dynamic offloader to thrash. Tell-tale: comfy log line `model weight dtype torch.float16, manual cast: None` against a native-fp8 checkpoint file." Generalizes beyond FLUX to any quantized-checkpoint workflow on Blackwell / Ada / Ampere 12-16 GiB consumer cards. Would have saved 16 hr of architectural-campaign cycles if the survival guide had had this entry.
- **Pre-AST validator gate:** `python -c "import ast; ast.parse(open('scripts/worker_iter.py','r',encoding='utf-8').read()); print('OK')"` clean; Bug Bible regression 23/1/2 baseline held.

### BUG-LOCAL-231: FLUX sampler 200-250x slower than community baseline on identical RTX 5080 + torch 2.10.0+cu130 + Win11 + fp8/bf16 + 1024x1024 + 20 steps [ACTIVE REGRESSION CONFIRMED 2026-05-19 14:15 -- 200-250x slower than community baseline on identical hardware/stack (Comfy-Org #9002 esp-dev 2026-02-02 = 0.75 s/it; OTR = 150-188 s/it across 7 telemetered cold-launches) -- OTR-architectural cause; defensive VRAM controllers (DeferredCheckpointLoader / FluxBranchGate / LtxBranchGate / UnloadAll / BatchFluxRender nuclear eviction / Sprint H launcher overrides) added across Sprint H + S28+ without per-change FLUX-pace benchmark are the hypothesis under test; minimal-workflow bisect in progress; commits 36bcfc0 + c1d37fe + 4c887e0 (pin-after-encode + Option B nuclear eviction) shipped but produced ZERO pace improvement; commit e2ca6d6 (TORCH_SDPA_BACKEND=math removal from 3 launcher sites) shipped but produced ZERO pace improvement -- both retained as architectural hygiene per `feedback_no_defensive_vram_protections`, neither was the cause]
- **Date:** 2026-05-18 | **Phase:** 5-6 | **Bible candidate:** pending close
- **Symptom:** With BUG-LOCAL-230's `--force-fp16` removal in place across all 4 launcher sites and `flux1-dev-fp8.safetensors` now loading correctly as native fp8 (`model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16`; load delta 11.08 GiB; gate #4 PASS), the FLUX sampler still runs slow. Direct telemetry from the 2026-05-18 21:10 smoke run, `logs/comfy_session_iter_001.log`:
  - L584: `[DeferredCheckpointLoader] load complete: VRAM allocated=2.13 -> 13.21 GiB (delta=11.08); ckpt=flux1-dev-fp8.safetensors` (architectural fix proven)
  - L585: `[FluxBranchGate] fire: VRAM allocated=13.21 GiB`
  - L592: `[BatchFluxRender] pinned MODEL via load_models_gpu`
  - L598: `[BatchFluxRender] skip_env_stills=True -- bypassing per-shot env-still FLUX pass; rendering radio bookend only`
  - L609: `0%|          | 0/20 [00:00<?, ?it/s]` (sampler starts)
  - L610: `5%|1/20 [02:34<48:46, 154.02s/it]` (sampler step 1 = 154 s; projected 48:46 for 20-step bookend)

  LibreHardwareMonitor during sampler step 1 (http://localhost:8085/data.json):
  - GPU Memory Used: **15911 MB / 16303 MB** (97.6%)
  - GPU Memory Free: 391 MB
  - D3D Shared Memory Used: **1098 MB** (offloader paging to system RAM)
  - Comfy process RSS 9119 MB (Bark + Mistral/Gemma + writer artifacts may still be partially resident)

  Compared to pre-BUG-LOCAL-230-fix closure run: 564.99 s/it -> 154 s/it (3.6x faster, fp8-fix delivered most of the win); 10445 MB D3D Shared -> 1098 MB (10x less offloader thrash). BUT vs the architectural-fix-only target (~10-15 s/step, VRAM peak <14.5 GiB, zero D3D Shared spill), this run is ~10x slow on per-step and ~756 MB over the CLAUDE.md ceiling.
- **Cause:** PENDING -- awaiting investigation. **NOT a dtype-upcast surface** (gate #4 proves the fp8 path works). Four candidate causes tagged for round-robin investigation, ordered by Jeffrey's first-read:
  1. **(a) Stale writer-LLM cache residency at FLUX entry.** Strongest first read. L592's `pinned MODEL via load_models_gpu` keeps FLUX hot in VRAM, but if Mistral-Nemo OR Gemma-4-E4B-it (whichever the writer used; see BUG-LOCAL-231 reconciliation axis below) is still partially resident from the audio branch, the headroom shrinks by 2-4 GiB. The 1098 MB D3D Shared spill fits "almost-resident, sampler activations push it over the edge" better than the other three. Diagnostic: check `_otr_model_loader.unload_llm()` was actually called before FLUX fires, and grep VRAM telemetry between `EpisodeAssembler emit audio_done` (L539) and `DeferredCheckpointLoader fire` (L573) for a writer-unload boundary.
  2. **(c) FLUX CLIP text encoder footprint.** FLUX CLIP is ~4 GiB. If it loads alongside the 11 GiB FLUX weights without being offloaded to CPU during the diffusion sampler step, the 15 GiB resident plus sampler activations matches the observed 15911 MB. Diagnostic: read `BatchFluxRender` for CLIP encoder lifecycle (load once + offload, or hot the whole time?).
  3. **(d) FLUX-schnell fallback.** Status-12 explicitly retracted FLUX-schnell as the recommended primary fix in favor of the `--force-fp16` removal. Listed here only as the fallback option that status-11 originally proposed, not as a status-12 recommendation. Quality tradeoff: 4 steps vs 20; fine for the bookend image (credits-screen enhancement), questionable for PASS1=3 lip-sync portraits (HuMo input quality).
  4. **(b) Sampler-time launch flag candidates (`--fast`, `--fast fp8_matrix_mult`).** REJECTED at first read per Jeffrey: BUG-LOCAL-230 was caused by a launch flag (`--force-fp16`) being added without proof. Don't reach for another launch flag as the first fix. Prove the symptom first, then propose surgery. Re-evaluate only after (a) and (c) are ruled out.

  **Separate axis flagged during smoke triage (do NOT bundle into BUG-LOCAL-231 fix):** L443 / L539 logs show `Selector slot=creative reuse cache for google/gemma-4-E4B-it`, but per memory `reference_default_llm_mistral_nemo.md` the writer canonical default is Mistral-Nemo 12B. Either a writer-side widget drift bug, a memory staleness, or a post-Sprint C C3 baseline shift that the memory missed (C3 "legitimately shifts the baseline (default flips to Gemma-4-E4B-it for VRAM headroom)" per `project_sprint_ordering_b_c_a.md`). Reconcile workflow JSON widget value vs runtime cache log before BUG-LOCAL-231 (a) investigation begins -- the writer model identity is load-bearing for (a)'s "stale residency" diagnostic.
- **Fix:** PENDING -- round-robin ChatGPT + Gemini on candidate ordering required before any code change per CLAUDE.md round-robin protocol (VRAM budgeting decision; "Windows-only library picks, VRAM budgeting decisions, anything touching the audio path, any choice between two architectures" -- this is a textbook VRAM budgeting decision). Transcript to save under `docs/2026-05-18-flux-vram-pressure/`.
- **Verify:** Re-run the BUG-LOCAL-230 7-criteria smoke gate after the BUG-LOCAL-231 fix. Specifically gates #5 (sampler ~10-15 s/step, NOT 154 s/step) and #6 (VRAM peak <14.5 GiB, NOT 15911 MB) must close cleanly. Gates #1-#4 + #7 will re-confirm trivially since BUG-LOCAL-230's fix is upstream of BUG-LOCAL-231.
- **Fix attempt 1 (36bcfc0, 2026-05-18):** Reordered execute() in `visual/batch_flux_render.py` so negative + bookend positive + per-shot positives are all encoded BEFORE pinning MODEL. Added `mm.free_memory(11500*1024*1024, [model.load_device])` between encodes and `mm.load_models_gpu([model])`. Dropped `force_full_load=True` + its TypeError fallback. Same two-pass pre-encode pattern applied to `visual/batch_flux_portrait_render.py` (which had no explicit pin but interleaved encode/sample per cast). Smoke #2 (HEAD 36bcfc0, 2026-05-18 23:00): gates #1-#4 PASS again (load delta still 11.08 GiB, dtype still fp8_e4m3fn). Gate #5 step 1 = **139.28 s/it** (vs target 10-15 s/step). Gate #6 LHM during sampler entry: GPU Memory Used 15886 MB, D3D Shared Memory Used 974 MB. Log shows `Unloaded partially: 7010.66 MB freed + 1512.49 MB freed + 1566.51 MB freed` firing DURING the encodes (BEFORE my `mm.free_memory` request at L591); MODEL gets evicted to fit CLIP, then re-loaded by my pin call. **Verdict: encode-before-pin reorder PASSES audio gates but does not close gates #5/#6 alone -- the partial-unloads happen at the encode step before my explicit eviction runs.**
- **Fix attempt 2 (c1d37fe + 4c887e0, 2026-05-18):** Per Jeffrey's pre-authorized escalation rule ("if LHM shows D3D Shared > 200 MB during sampler the eviction didn't work and the fix needs to escalate to option B (unload_all_models)"), swapped `mm.free_memory(...)` to `mm.unload_all_models() + gc.collect() + torch.cuda.empty_cache()` (the Bug Bible BUG-07.03 invariant requires the gc+empty_cache pairing; 4c887e0 added it after c1d37fe failed the regression). Smoke #3 (HEAD 4c887e0, 2026-05-18 23:23): gates #1-#4 PASS again. Gate #5 step 1 = **123.72 s/it** (still 8-10x off target). Gate #6 LHM during sampler entry: GPU Memory Used 15945 MB, D3D Shared Memory Used 1199 MB (worse than smoke #2). Log shows `[BatchFluxRender] unload_all_models() + gc.collect() + empty_cache() complete` at L591 followed by `Requested to load Flux` at L592 and `pinned MODEL via load_models_gpu` at L595 -- the Option B code IS firing. **Verdict: even nuclear eviction + gc + empty_cache leaves ~15.5 GiB resident at sampler entry. ComfyUI's `mm.unload_all_models()` only evicts comfy-registered models; the audio path's Bark/Kokoro/MusicGen instances are loaded via OTR's own `_otr_model_loader` and are NOT in the comfy registry, so they survive.** Pre-fix 154 -> attempt-1 139 -> attempt-2 123 is a 20% total improvement, but the architecture is not approaching the 10-15 s/step target.
- **Next investigation surface (post-attempt-2):** The residual ~5 GiB at sampler entry (15.5 GiB used vs ~11 GiB expected for FLUX MODEL alone) is the bottleneck. Candidates ordered by likelihood:
  - **(i) Audio-model residual not surfaced as a comfy model.** Bark/Kokoro/MusicGen + Gemma writer LLM may be holding PyTorch CUDA tensors that survive `mm.unload_all_models()`. Need explicit teardown via `nodes._otr_model_loader.unload_llm()` + per-audio-node `model.cpu() + del` + `gc.collect() + empty_cache()` BEFORE the FLUX phase starts.
  - **(j) comfy_kitchen backend eager-loading FLUX layers.** Startup log L45-47 shows `Found comfy_kitchen backend eager / cuda / triton: capabilities: ['apply_rope', 'apply_rope1', 'dequantize_mxfp8', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'scaled_mm_mxfp8', 'scaled_mm_nvfp4']`. Worth checking whether the eager backend pre-allocates VRAM for the fp8 dequant path.
  - **(k) `model.load_device` addressability.** `mm.free_memory(N, [model.load_device])` may not address the GPU device correctly on Blackwell sm_120; passing `[torch.device("cuda:0")]` directly may behave differently.
  - **(l) sageattention pin / Blackwell-specific fp8 path.** Workflow's `PathchSageAttentionKJ` is `"disabled"` BUT the SDPA + comfy_kitchen sm_120 path may not be optimal for fp8_e4m3fn sampling. Worth checking whether the bf16 cast in `manual cast: torch.bfloat16` is hitting a slow code path that's resident-VRAM-bound rather than compute-bound.
- **Tags:** vram-pressure, flux-fp8, sampler-pace, offloader-marginal, blackwell, attempt-2-incomplete, audio-residual-vram
- **Bible candidate rationale:** Pending close. If the fix is an architectural change (LLM unload path, CLIP offload pattern) it generalizes to any quantized-checkpoint workflow on a 16 GB consumer card. If the fix is "use FLUX-schnell" it's local to this workflow and would NOT promote. Decide at close.
- **Process discipline notes (this smoke + diagnosis):**
  - Smoke killed cleanly at step 1/20 after gate #4 PASS proved the architectural axis -- no value in burning another 3.5 hr of GPU through completion for redundant gate-#5 confirmation; the tqdm projection at step 1 is conclusive evidence of pace.
  - Kill path: `POST /interrupt` to ComfyUI HTTP API (HTTP 200, sampler canceled gracefully) -> `taskkill /F /PID 50236` (worker .venv stub; uv child 35232 already gone via propagation) -> supervisor (47560 / 37396) auto-exited on worker death -> ComfyUI (PID 65268 / 18444) re-parented to System, still bound at :8000, ready for next iteration without a model-load tax.

- **Follow-up 2026-05-19 00:50 -- [FIXED] DEMOTED to [PARTIAL]. One-run promotion was premature; reproduction smoke shows runtime axis still failing.**
  - **2026-05-18 23:30 lucky run telemetry:** sampler 1.22 s/it (bookend) + 1.38/1.44/1.48 s/it (3 portraits). Promoted 231 to `[FIXED]` at commit 399abe7 based on this one observation. **Per `feedback_bug_bible_curation_discipline` saved 2026-05-18: empirical verification requires re-run proves symptom went away across variance, not a single observation. The 399abe7 promotion violated the rule retroactively.**
  - **2026-05-19 00:34 reproduction smoke (HEAD fbd0e0c, same code as the lucky run + BUG-LOCAL-234/235 atomic HuMo fix layered on top):** sampler step 1 = **176.62 s/it**. LHM during sampler: GPU Memory Used 15896 MB / 16303 MB (97.5%), D3D Shared 808 MB. Pre-pin partial-unloads at the encode phase still fire (`Unloaded partially: 7010.66 MB freed, 4339.45 MB remains loaded` etc.) -- identical pattern to the broken pre-fix smokes. `BatchFluxRender.unload_all_models()` IS firing (logged), MODEL re-loads at 11.35 GB (logged), but the sampler still thrashes.
  - **Hypothesis (NOT YET EMPIRICALLY VERIFIED):** OTR's audio-side loaders (Bark / Kokoro / MusicGen / Gemma writer LLM, loaded via `nodes/_otr_model_loader.py`) are NOT registered with `comfy.model_management`. `mm.unload_all_models()` only evicts comfy-registered models. The audio-side models survive the unload + empty_cache and occupy 2-4 GiB at FLUX entry. When the residue is small (lucky), the sampler runs at ~1-1.5 s/it. When the residue is large (today's smoke), the sampler thrashes at 150-180 s/it. Variance manifests across runs based on which audio models were loaded for the script length / cast composition.
  - **Alternative hypotheses NOT YET RULED OUT (do not pre-prescribe a fix):**
    - (alt-a) Other system processes (browser, Discord, Steam shader cache) holding 2-4 GiB transiently.
    - (alt-b) ComfyUI's own state from prior interruptions leaving allocator reserve across the sweep restart.
    - (alt-c) NVIDIA driver state / D3D Shared spillover from earlier non-FLUX work.
  - **Empirical disambiguation plan (Jeffrey 2026-05-19 directive):**
    1. Add one log line to `BatchFluxRender` at deferred-loader-fire site capturing GPU Memory Used + sys RSS via LHM HTTP. No defensive guards, just instrumentation.
    2. Cold-launch ComfyUI from fresh `start_comfy.bat`. Run **3 smokes back-to-back**. Capture LHM-at-fire numbers per run.
    3. If bad runs (high s/it) consistently show 5-7 GiB allocated at fire while good runs show 2-3 GiB → audio-residue hypothesis confirmed → land OTR-side teardown.
    4. Otherwise → investigate alt-a/b/c.
    5. After fix lands, run 3 more smokes to confirm variance is gone before re-flipping to `[FIXED]`.
  - **Per `feedback_no_defensive_vram_protections`:** do NOT add `_otr_model_loader.unload_llm()` + per-audio-node `.cpu() + del` defensively before empirical evidence the audio-residue hypothesis is correct. Steps 1-4 above gate the fix.
  - **Status:** runtime axis remains BLOCKED. BUG-LOCAL-234 + 235 fixes are landed but unverifiable while FLUX thrashes (HuMo never reached cleanly).

- **Follow-up 2026-05-19 08:02 -- LHM-at-fire telemetry captured for iter 1 of the cold-launch disambiguation sweep (commit 3691317).**
  - **Setup:** Cold-launched ComfyUI via `scripts/sweep_and_launch.bat --iters 3 --inter-iter-sec 0 --no-stop-conditions` at 07:44:20. Sweep killed all prior python (incl. PID 4508 ComfyUI at 96.9% VRAM). New ComfyUI PID 7456 spawned by worker PID 22200 directly (no .bat wrapper, no PowerShell wrapper).
  - **Iter 1 telemetry data point (comfy_session_iter_001.log):**
    - L562 (fire): `[DeferredCheckpointLoader] fire: VRAM allocated=2.13 GiB reserved=2.44 GiB lhm_used=5217 MB; gate_signal len=80; ckpt=flux1-dev-fp8.safetensors`
    - L573 (load complete): `allocated 2.13 -> 13.21 GiB (delta=11.08); reserved 2.44 -> 13.25 GiB (delta=10.81); lhm_used 5217 -> 15790 MB (delta=10573)`
    - L574: `[FluxBranchGate] fire: VRAM allocated=13.21 GiB`
    - L587: `[BatchFluxRender] unload_all_models() + gc.collect() + empty_cache() complete (nuclear eviction before MODEL pin, BUG-LOCAL-231 Option B escalation)` (Option B IS firing as designed)
    - L591: `[BatchFluxRender] pinned MODEL via load_models_gpu`
    - L595: sampler started, 20 steps
    - L596 (sampler step 1): `5%|1/20 [02:52<54:46, 172.99s/it]` -- **SLOW REGIME REPRODUCED**
  - **LHM live during sampler step 1:** GPU Core 100.0%, D3D 3D 96.4%, GPU Memory Used 15855 MB / 16303 MB, GPU Memory Free 447 MB, GPU Power 0.0 W (LHM read-side glitch, not 0). Driver-side ground truth: GPU fully saturated.
  - **Non-torch VRAM consumption at fire (the disambiguating number):** lhm_used 5217 MB minus torch reserved 2.44 GiB (~2620 MB) = **~2597 MB (~2.6 GiB) of NON-TORCH VRAM consumption at FLUX fire**. ComfyUI's torch view is clean (2.13 GiB allocated, 2.44 GiB reserved), but the GPU is already at 5.2 GiB occupied at fire time. The 2.6 GiB delta is browser/Discord/Steam shader cache / driver baseline.
  - **Hypothesis cross-check against today's data:**
    - **Audio-residue (the original strongest first-read):** FALSIFIED for this iter. Audio nodes would have shown allocated > ~3 GiB at fire (Bark / Kokoro / MusicGen / Gemma still resident as PyTorch tensors). Observed 2.13 GiB allocated matches the commit-message "lucky case (GOOD run, clean state)" profile exactly. The audio teardown that did NOT happen wasn't needed -- audio is being released by something already.
    - **alt-a (browser / Discord / Steam shader cache holding 2-4 GiB transiently):** CONSISTENT. lhm_used at fire was 5217 MB while torch reserved was only 2.44 GiB. The 2.6-GiB delta is the non-torch consumption alt-a predicts.
    - **alt-b (ComfyUI allocator reserve across sweep restart):** NEUTRAL / NOT SUPPORTED. Sweep was clean (`KILL PID=4508 ... KILL PID=41164` both confirmed) and reserved at fire was only 2.44 GiB. No evidence of stale comfy allocator state.
    - **alt-c (NVIDIA driver / D3D Shared spillover from earlier work):** WEAKLY CONSISTENT. D3D Shared Memory Used was 727 MB at probe time (small fraction of the 2.6 GiB non-torch but non-zero). Not enough to explain the full 2.6 GiB delta on its own.
  - **What this one data point says:**
    - The slow regime (172.99 s/it, matching the 176 s/it Jeffrey saw in the 00:34 smoke) DID reproduce in a clean cold launch with Option B nuclear eviction in place.
    - Audio-residue hypothesis is FALSIFIED for this iter (allocated_at_fire = 2.13 GiB is the lucky-case profile, yet the sampler is still slow).
    - alt-a (external VRAM consumption) is the leading hypothesis: ~2.6 GiB of non-torch VRAM at fire pushes total occupancy past the headroom that FLUX needs for a non-thrashing sampler.
  - **What this one data point does NOT say:**
    - We have 1 of 3 planned iters. Variance reproducibility is not yet measured (the 1.22 s/it lucky case was a SEPARATE prior run, and the iter 2 + iter 3 captures were aborted at operator request before reaching FLUX fire).
    - The non-torch process holding 2.6 GiB has not been identified by name.
    - Whether closing browser / Discord / Steam recovers the headroom is not yet tested.
  - **Operator early-stop:** Iter 2 worker had launched (PID 18664, ComfyUI PID 61096, in writer phase at 08:06) when Jeffrey killed the supervisor + workers at 08:06:59 via the user-initiated stop. Iter 2 + iter 3 telemetry not captured this session. No data loss for iter 1 (already on disk in `logs/comfy_session_iter_001.log`).
  - **Next session prerequisites for closing 231 (per Jeffrey 2026-05-19 directive -- 3-smoke discipline, not 1):**
    1. Close non-essential GPU apps: browser tabs, Discord, Steam, any DXVK-using app, any non-ComfyUI process touching CUDA.
    2. Cold-launch ComfyUI from fresh `scripts/sweep_and_launch.bat --iters 3 --inter-iter-sec 0 --no-stop-conditions`.
    3. Run **3 smokes back-to-back** under this clean state. Capture per-iter:
       - `[DeferredCheckpointLoader] fire` line (allocated, reserved, lhm_used)
       - `[DeferredCheckpointLoader] load complete` line (deltas)
       - Sampler step 1 s/it
    4. Decision tree (no premature promotion -- 3-smoke verification required):
       - **All 3 sub-3 GiB lhm_used at fire AND all 3 at 10-15 s/it (or faster)** → alt-a CONFIRMED. Pipeline behaves correctly when given clean VRAM headroom. Land the operator-checklist mitigation (see "Soft mitigation design" below). Status → `[VERIFIED CLOSED with operator-checklist mitigation]`, NOT `[FIXED]` -- there's no code-level fix, just a measurement layer + checklist.
       - **Mixed (1 or 2 hit slow regime)** → external apps not the sole cause. Reopen candidate (i) audio-side teardown via `_otr_model_loader.unload_llm() + per-audio-node .cpu()+del + gc+empty_cache` OR alt-b/c. Stay PARTIAL.
       - **All 3 still hit slow regime even with everything closed** → alt-a FALSIFIED. Then it's alt-b (Comfy state) or alt-c (driver/D3D). OTR-side fixes won't help. Stay PARTIAL; escalate.
  - **Soft mitigation design (lands only if alt-a CONFIRMED per the 3-smoke decision tree):**
    - LHM-at-fire telemetry stays in `nodes/_otr_deferred_loaders.py` (already shipped 2026-05-19 in commit 3691317).
    - `OTR_FluxBranchGate` and/or `BatchFluxRender` pre-load hook adds a soft warning at fire time:
      ```
      non_torch_mb = lhm_used_mb - (torch_reserved_gib * 1024)
      if non_torch_mb > 1500:
          log.warning(
              "[OTR-VRAM-PREFLIGHT] %d MB of non-torch VRAM at FLUX fire; "
              "sampler may thrash. Close browser / Discord / Steam / "
              "non-ComfyUI GPU apps for best pace.",
              non_torch_mb,
          )
      ```
    - CLAUDE.md / README gets a "Pre-FLUX checklist" operator note.
    - No defensive teardown code (per `feedback_no_defensive_vram_protections`).
  - **Per `feedback_bug_bible_curation_discipline` and `feedback_no_defensive_vram_protections`:** Still PARTIAL. One slow iter under controlled cold-launch + Option B in place + audio-residue ruled out = strong update to the cause model, but not enough to close. NO code change proposed today. Status stays BLOCKED until the 3-smoke alt-a verification test is run. **Do NOT flip to [FIXED] on next session's first smoke either** -- 3-smoke minimum from this point forward.
  - **Status:** runtime axis remains BLOCKED. Audio-residue hypothesis now strongly disfavored (was leading); alt-a now leading. No teardown code added. 3-smoke clean-state battery is the verification gate.

- **Follow-up 2026-05-19 09:26 -- 3-smoke clean-state battery COMPLETE. alt-a FALSIFIED. Variance is not in fire-time state -- it's downstream in the sampler / driver / kernel layer.**
  - **Setup:** Pre-launch LHM floor with ComfyUI fully down: GPU Memory Used = **2276 MB**. That's the headroom hole from browser / Discord / driver / Windows compositor / shader cache before ComfyUI loads anything. Used as the "floor" reference per Jeffrey's directive: any fire-time lhm_used above floor + 2 GiB = unusual external load.
  - **Launch:** `scripts\sweep_and_launch.bat --iters 3 --inter-iter-sec 0 --no-stop-conditions` at 08:41:17. Each iter cold-launched ComfyUI as a direct subprocess.Popen child of the worker (no .bat / no PowerShell wrapper). Workflow: `workflows/otr_scifi_16gb_full.json` (canonical; renders radio bookend only -- per-shot env-stills and per-character portraits are skipped via skip_env_stills=True per BUG-LOCAL-078 follow-up).
  - **Per-iter telemetry table:**

    | Iter | start | fire alloc | fire reserved | fire lhm_used | non-torch above floor | step 1 s/it | verdict |
    |---:|---|---|---|---|---|---|---|
    | 1 | 08:41:17 | 2.13 GiB | 2.44 GiB | 5289 MB | 393 MB | **42.38 s/it** | INTERMEDIATE (3-4x off target) |
    | 2 | 08:56:31 | 2.13 GiB | 2.47 GiB | 5364 MB | 468 MB | **>90 s/it** (autokiller fired @+90s before tqdm flushed step 1) | SLOW (regime confirmed by no-flush) |
    | 3 | 09:11:18 | 2.13 GiB | 2.44 GiB | 5371 MB | 475 MB | **158.86 s/it** | SLOW (matches yesterday's 172.99 / 176.62) |

    Non-torch above floor = lhm_used at fire minus torch_reserved minus baseline floor (2276 MB). Captures how much ABOVE baseline external apps the GPU is holding when FLUX is about to load. All three runs ~400-470 MB above floor -- essentially indistinguishable.
  - **Disambiguation decision tree applied (from prior follow-up):**
    - "All 3 sub-3 GiB lhm_used at fire AND all 3 at 10-15 s/it" -- **NO**. All 3 were 5.2-5.4 GiB at fire (well above 3 GiB), AND none were at 10-15 s/it (best was 42 s/it).
    - "Mixed (1 or 2 hit slow regime)" -- **PARTIAL FIT**. iter 1 was intermediate (42 s/it, 3-4x off target), iter 2 + 3 were clearly slow (>90 / 158 s/it).
    - "All 3 still hit slow regime even with everything closed" -- **CLOSEST FIT**. iter 1 was the best of the 3 but still off target; iter 2 + 3 hit the slow regime; none reached the 10-15 s/it target. Decision-tree verdict: **alt-a FALSIFIED**.
  - **The new headline finding -- variance is NOT in the fire-time state:**
    - Iter 1, 2, 3 fire-time tuples (alloc, reserved, lhm_used): **(2.13, 2.44, 5289), (2.13, 2.47, 5364), (2.13, 2.44, 5371)** -- within 82 MB of each other on lhm_used, identical on allocated, within 30 MB on reserved.
    - Yet step 1 s/it varied **42.38 / >90 / 158.86** -- a 4x spread on essentially identical fire-time state.
    - This rules out audio-residue (iter 1 follow-up already did this), alt-a (today's battery), AND alt-b (allocator reserve was 2.44/2.47/2.44, indistinguishable). The variance must come from somewhere NOT captured in the fire-time snapshot.
  - **Remaining candidate hypotheses (no longer ordered "leading" -- ranked by what the data is consistent with):**
    - **(alt-c) Driver / D3D Shared spillover during sampler execution.** Not captured in our fire-time snapshot. D3D Shared Memory Used in prior yesterday's iter 1 slow run was 808-1098 MB DURING sampler step 1 (post-load, post-pin). If today's fast iter 1 had clean D3D Shared during sampler vs iter 3's spillover during sampler, that explains the 42 vs 158 split. **MISSING TELEMETRY:** we don't capture LHM at sampler step 1 -- only at fire (pre-load) and at load complete (post-load, pre-sampler). Need a third snapshot inside `BatchFluxRender` after `KSamplerAdvanced.sample` first step returns.
    - **(alt-e -- new) Non-deterministic FLUX kernel scheduling on Blackwell sm_120.** comfy_kitchen backend exposes eager / cuda / triton with capabilities `dequantize_per_tensor_fp8 / scaled_mm_mxfp8 / quantize_per_tensor_fp8 / apply_rope` (per startup L45-47). If the dispatch picks different kernel variants per cold launch (kernel auto-tuner, cublas heuristics, cudnn benchmark mode), per-step pace could vary 4x at the same VRAM occupancy. **CHECK:** is `torch.backends.cudnn.benchmark` True? Does `cublas` use heuristic-selected kernels for fp8 matmul? Either would produce run-to-run variance.
    - **(alt-d) sageattention pin / Blackwell-specific fp8 path** (from prior next-investigation list). Workflow's `PathchSageAttentionKJ` is disabled, but SDPA's fp8_e4m3fn -> bf16 cast path on Blackwell may itself be unstable across runs.
  - **What the data clearly rules OUT:**
    - audio-residue (cleared by iter 1 yesterday + reproduced today)
    - alt-a / external app pressure (cleared by 3-iter clean-state battery today: same fire-time state, 4x variance)
    - alt-b / Comfy allocator reserve (cleared by 3-iter battery: reserved 2.44/2.47/2.44 within 30 MB)
  - **Mitigation status:** OTR-VRAM-PREFLIGHT soft warning design from the prior follow-up is now LESS WARRANTED. It would have warned on all 3 of today's iters (lhm_used > 5 GiB at fire) but only one had the catastrophic slow regime. The warning would have over-triggered. **HOLD on landing the preflight warning** until alt-c/d/e investigation tells us what to gate on.
  - **Operator state for this battery:** Jeffrey was in flight closing non-essential apps per directive but the baseline floor of 2276 MB suggests apps were already minimal (closing a heavy browser session typically drops floor by 800-2000 MB). The 3-iter battery had similar floor conditions throughout (5289 / 5364 / 5371 MB at fire = within 82 MB).
  - **Next investigation surface (post-3-smoke alt-a falsification):**
    1. **Add LHM-at-sampler-step-1 telemetry.** Modify `visual/batch_flux_render.py` to log lhm_used + torch_reserved AFTER KSampler returns step 1 (e.g., a tqdm callback that fires on the 5%/20 progress). This catches D3D Shared spillover signature during the diffusion math -- exactly when the slow regime manifests.
    2. **Add cudnn/cublas determinism check.** Log `torch.backends.cudnn.benchmark`, `torch.backends.cudnn.deterministic`, and the autotuner state at FLUX entry. If benchmark mode is True, that explains run-to-run variance directly.
    3. **(only after #1 + #2 land)** If alt-c data shows D3D Shared spike during slow-regime sampler step 1: investigate ComfyUI's offloader behavior under fp8 fp16/bf16 cast on Blackwell sm_120 (may be paging activation tensors to system RAM despite 0.4 GiB of free VRAM headroom).
    4. **(parking lot)** Consider whether the slow regime is acceptable for production. Iter 1 (42 s/it × 20 steps × 4 renders = ~56 min just for FLUX) is borderline tolerable; iter 3 (158 s/it × 20 steps × 4 renders = ~3.5 hr) is not. If we can predict run regime, we could re-roll cold launches that hit the slow profile.
  - **Per `feedback_bug_bible_curation_discipline`:** 3-smoke battery executed cleanly. Status stays PARTIAL. **alt-a FALSIFIED** is a verified empirical finding (3 independent cold-launch runs in clean state, identical fire-time tuples, 4x sampler variance). NO `[FIXED]` flip. NO `[VERIFIED CLOSED with operator-checklist mitigation]` either -- that branch was contingent on alt-a confirming, which it did not.
  - **Status:** runtime axis still BLOCKED. Audio-residue OUT. alt-a OUT. alt-b OUT. alt-c / alt-d / alt-e remaining. Need step-1-time telemetry + cudnn determinism check to disambiguate further. BUG-LOCAL-234 + 235 still unverifiable while FLUX thrashes.

- **Follow-up 2026-05-19 10:11 -- 4 corrections to the prior follow-up + sampler-time telemetry block landed.**

  Jeffrey 2026-05-19 09:30 directive pushed back on four items from the 09:26 follow-up. Recording the corrections:

  1. **iter 2 is INVALID data, not "slow."** The autokiller fired at +90s post-load_complete -- BEFORE tqdm flushed step 1. Iter 2's true sampler pace is UNKNOWN (could be 95 s/it or 250 s/it). The earlier "iter 2 = >90 s/it slow" label was treating absence of data as evidence. Correct reading of the battery: 2 clean data points (iter 1 = 42.38 s/it, iter 3 = 158.86 s/it), not 3. iter 2 is "captured fire-time only, sampler unknown".

  2. **alt-e (kernel non-determinism) is the strongest remaining candidate, not co-equal with c and d.** Same fire-time state plus 4x sampler variance = something non-deterministic between fire and step 1. cudnn / cublas autotuner picks different kernel implementations across runs based on transient timing. iter 1 got a fast kernel; iter 3 got a slow one. This is the canonical signature of non-determinism, not memory pressure.

  3. **alt-d (sageattention) is RULED OUT by workflow inspection.** Workflow `otr_scifi_16gb_full.json` has node id=42 type `PathchSageAttentionKJ` widgets `['disabled', False]` title `'Patch Sage Attention (FLUX) -- DISABLED, BUG-LOCAL-070'`. SDPA is the active attention path. Drop alt-d from the candidate list. (Verified via `outputs/check_sage.py` 2026-05-19 10:06.)

  4. **The "10-15 s/it target" may be wrong for this hardware.** Iter 1's 42 s/it was labeled "intermediate" but 42 s/it on FLUX-dev fp8 on RTX 5080 Laptop might actually be normal pace at 1024x1024 / 20 steps / bf16 cast. The earlier 1.22 s/it lucky run might have been an outlier (or a different image size / step count). Sanity check needed: Comfy-Org or community benchmarks for FLUX-dev fp8 pace on RTX 5080 Laptop at this configuration. Without a verified target, "off target" is conjecture.

  **Corrected hypothesis ranking (post-pushbacks):**
    - **alt-e (non-deterministic kernel scheduling):** LEADING. Direct fit for the data.
    - **alt-c (D3D Shared spillover during sampler):** OPEN. Need step-time telemetry.
    - **alt-f (NEW: thermal / clock throttling):** OPEN. Jeffrey added this candidate. If GPU clock drops or temperature spikes between cold launches, hardware throttling explains run-to-run variance. nvidia-smi snapshot at sampler entry catches this.
    - **alt-d (sageattention / Blackwell fp8 path):** RULED OUT (workflow has sage disabled).
    - **audio-residue / alt-a / alt-b:** still OUT.

  **Code change landed (commit pending after BUG_LOG + ROADMAP edits):** `visual/batch_flux_render.py` adds two new module-level helpers and wraps the radio bookend sampler.sample call with them.
    - `_log_flux_sampler_precheck()` -- one-line log of `cudnn.benchmark` / `cudnn.deterministic` / `cudnn.allow_tf32` / `matmul.allow_tf32` / `cuda.is_initialized` at sampler entry; one-line nvidia-smi snapshot of `clocks.gr` / `clocks.mem` / `power.draw` / `power.state` / `temperature.gpu` / `utilization.gpu` / `utilization.memory` at sampler entry.
    - `_FluxSamplerPoller` daemon thread polling LHM `GPU Memory Used` + `D3D Shared Memory Used` and `nvidia-smi clocks.gr / power.draw / temperature.gpu / utilization.gpu` every 5 sec during sampler.sample. Logs one `[OTR-FLUX-SAMPLER-POLL] tick=N` line per poll. No `.join()` on stop (avoids hang on HTTP urlopen); daemon thread dies with process if lingering.
    - Wrapping at `_render_and_save_radio_bookend` L1136 -- the only sampler.sample call that fires in the canonical workflow (skip_env_stills=True bypasses the other two).
    - All best-effort with broad except; never raises into sampler. Pure logging -- no behavior change. Per `feedback_no_defensive_vram_protections`.

  **Pre-commit verification (all green):**
    - AST parse `visual/batch_flux_render.py`: OK (64695 bytes, 1407 lines).
    - Bug Bible regression: 23 passed, 1 skipped, 2 xfailed (baseline held).
    - Audio byte-identical: 9 passed, 1 skipped (audio path sealed).

  **Next battery (post-commit):** Re-run `sweep_and_launch.bat --iters 3 --inter-iter-sec 0 --no-stop-conditions` with the new telemetry. Autokiller wait extended to 240-300s post-load_complete so iter 2's true sampler pace is captured even in the slow regime. Then apply decision tree:
    - `cudnn.benchmark=True` across iters AND step-5 D3D similar AND nvsmi clocks similar -> **alt-e confirmed** -> pin `cudnn.deterministic=True` for FLUX path or document the variance as expected.
    - D3D Shared spikes in slow iters but not fast -> **alt-c confirmed** -> investigate offloader / explicit eviction / smaller batch.
    - GPU clock throttling in slow iters -> **alt-f confirmed** -> operator mitigation (laptop cooling pad, ambient temp, throttling-aware re-roll).
    - All telemetry similar across iters but pace still varies -> **alt-g (unknown)** -> deeper investigation, possibly torch 2.10 / CUDA 13 stack issue.
  **Status:** runtime axis still BLOCKED. Sampler-time telemetry block landed. Awaiting next 3-smoke battery.

- **Follow-up 2026-05-19 11:03 -- Battery v2 COMPLETE. alt-e + alt-f FALSIFIED. alt-c partial but not correlated with pace. New leading hypothesis: alt-h (180 s/it IS the normal pace at this config; fast outliers were warm-cache artifacts).**

  Battery v2 ran 2026-05-19 10:15:26 -> 11:03:31 PT with the new sampler-time telemetry block (commit 6df78d8) in place. Pre-launch LHM floor: 2349 MB. Three cold-launch iters back-to-back via `sweep_and_launch.bat --iters 3 --inter-iter-sec 0 --no-stop-conditions`. Autokiller wait extended to 240s post-load_complete (Jeffrey 09:30 pushback #1).

  **Per-iter telemetry tuples (all 3 iters complete -- the iter 2 "invalid" gap from battery v1 is closed):**

    | Iter | fire alloc | fire reserved | fire lhm_used | load complete (alloc/res/lhm) | precheck cudnn.benchmark | step 1 s/it |
    |---:|---|---|---|---|---|---|
    | 1 | 2.13 GiB | 2.44 GiB | 5506 MB | 13.21 / 13.25 / 15942 | False | **188.35 s/it** |
    | 2 | 2.13 GiB | 2.47 GiB | 5455 MB | 13.21 / 13.25 / 16033 | False | **186.77 s/it** |
    | 3 | 2.13 GiB | 2.41 GiB | 5122 MB | 13.21 / 13.25 / 15856 | False | **177.16 s/it** |

  All 3 iters within **7% spread** (177-188 s/it). No 4x variance reproduces.

  **Per-iter sampler-time poller telemetry (clocks.gr, D3D Shared, temp -- all polled every 5 sec during sampler.sample):**

    | Iter | clocks.gr range | D3D Shared range | temp range | utilization | power |
    |---:|---|---|---|---|---|
    | 1 | **2977 MHz stable** (35 ticks) | 559-761 MB (spike to 733-761 mid-sampler) | 52-54 C | 100% | 62-65 W |
    | 2 | **2977 MHz stable** (~36 ticks) | 557-666 MB (mild rise to 666 mid-sampler) | 52-55 C | 100% | 62-65 W |
    | 3 | **2977 MHz stable** (~41 ticks, one transient 2970 MHz dip at ticks 18-19) | 615-808 MB (rose to 771-808 from tick 12+) | 53-55 C | 99-100% | 63-67 W |

  **Hypothesis status post-battery-v2 (corrected per Jeffrey 2026-05-19 11:10 pushback):**

    | Hypothesis | Battery v1 status | Battery v2 status |
    |---|---|---|
    | audio-residue | OUT | OUT (still) |
    | alt-a external VRAM pressure | OUT (3-iter fire-time identical) | OUT (still) |
    | alt-b Comfy allocator reserve | OUT | OUT (reserved 2.41-2.47 indistinguishable across 6 iters now) |
    | alt-c D3D Shared during sampler | OPEN | **STILL OPEN -- not ruled out.** D3D Shared (559-808 MB) did NOT explain the small 177-188 s/it spread WITHIN battery v2 (iter 3 had highest D3D and fastest pace, opposite of alt-c's prediction). But all three iters were already near the VRAM ceiling (~15.9 GiB / 16.3 GiB used at sampler), so the 7% spread is a high-noise floor for testing alt-c. Sampler-time paging across the **fast/slow regime split** (the 42.38 s/it battery v1 iter 1 outlier vs the 177-188 s/it cluster) remains a possible alt-c signal that THIS battery did not capture. Need fast-vs-slow comparative telemetry to disambiguate. |
    | alt-d sageattention | RULED OUT (workflow audit) | RULED OUT (still) |
    | alt-e non-deterministic kernel (cudnn autotuner) | OPEN -- LEADING | **FALSIFIED**. `cudnn.benchmark=False` across all 3 iters. Autotuner is NOT running. Kernel selection is deterministic. |
    | alt-f thermal / clock throttling | OPEN (NEW from pushback) | **FALSIFIED**. `clocks.gr` stable at 2977 MHz (full boost) across all 3 iters x dozens of ticks. Temps 52-55 C, far below 85+ C throttle threshold. Single 2970 MHz transient is non-throttle noise. |
    | alt-h (NEW) ~180 s/it IS the slow-regime baseline at this config | --- | **LEADING for the slow regime.** **5 of 6** current-battery cold-launches landed at 177-188 s/it (battery v1 iters 2 and 3 at 158-186; battery v2 iters 1, 2, 3 at 177-188; the 42.38 s/it battery v1 iter 1 was the fast outlier). Within the slow cluster the spread is 7%. The slow-regime baseline at this configuration looks reproducible. **Does NOT explain why some runs are faster.** |

  **Counts and ratios -- corrected math (Jeffrey 2026-05-19 11:10 pushback #1, #2):**

  - **5 of 6** current-battery cold-launches landed in the slow regime (~177-188 s/it). **One fast outlier remains unexplained: battery v1 iter 1 at 42.38 s/it, with telemetry captured.** A separate, earlier outlier from yesterday (~23:30 PT) ran at 1.22 s/it but predates the sampler-time telemetry block, so its precheck / poll-time state is unrecoverable.
  - Speed-up ratios vs the slow-cluster median (~183 s/it):
    - **42.38 s/it (battery v1 iter 1, today, with telemetry) is 4.3x faster than the slow cluster.** NOT 100x.
    - **1.22 s/it (yesterday 23:30, no sampler-time telemetry) is 150x faster** than the slow cluster.
    - These are TWO DISTINCT outliers, not one event. Do not conflate.

  **Reframe -- corrected language (Jeffrey 2026-05-19 11:10 pushback #3, #5):**

  - The slow-regime baseline at this current stack/config is empirically ~180 s/it. The fast outliers are real and remain unexplained.
  - **Proposed status reframe (open for Jeffrey's call): `[CURRENT-CONFIG BASELINE CHARACTERIZED / OPTIMIZATION OPEN]`** -- NOT `[NOT A BUG]`. The original `[NOT A BUG / HARDWARE BASELINE CHARACTERIZED]` proposal was too closed-door. The data supports a current baseline; it does NOT support "no fix exists." The fast outliers prove the hardware CAN go faster under unknown conditions; finding those conditions is the optimization surface.
  - The audio path Prime Directive 1 was the actual constraint; FLUX baseline pace is what we measured; faster paths exist but are not yet reproducible.

  **Fast-path mystery -- separate tracker (Jeffrey 2026-05-19 11:10 pushback #5):**

  Do not close BUG-LOCAL-231 until either (a) the fast-path mechanism is reproduced + explained, OR (b) the fast-path is formally split into a new separate tracking entry. Choosing (b) now:

  - **Open BUG-LOCAL-244: FLUX fast-path mechanism unidentified** (separate entry). Tracks both outliers:
    - 42.38 s/it (today battery v1 iter 1, 2026-05-19 ~08:55, with precheck + 1 poll tick captured at start of sampler).
    - 1.22 s/it (yesterday 2026-05-18 ~23:30, no sampler-time telemetry pre-commit-3691317).
  - 244's verification gate: reproduce ≥1 fast run with sampler-time telemetry, OR explain the mechanism via Comfy / Blackwell / cudnn / cublas literature, OR mark UNREPRODUCIBLE if no fast run captured in next N batteries.

  **Step-1 caveat -- per-step timing needed before closure (Jeffrey 2026-05-19 11:10 pushback #4):**

  - All s/it figures captured here are **step 1 ONLY.** Step 1 includes lazy GPU allocation, fp8 dequant kernel JIT compile, cudnn workspace warmup, possibly cublas tuning cache priming. Step 1 timing biases HIGH vs steady-state.
  - The 180 s/it figure could be high by step-1 overhead. Need either (a) one complete 20-step run measuring average + per-step variance across steps 2-20, OR (b) per-step timestamps logged across at least 5 steps.
  - **Action item:** extend the telemetry block with a per-step timestamp log (sampler tqdm callback or wrap KSampler.sample with a step counter). Re-run battery v3 with the per-step telemetry. THEN propose final status.

  **Standing disciplines reaffirmed:** No defensive VRAM protections were added at any point. No `[FIXED]` flip on any single observation. The 6-iter empirical foundation is the slow-regime baseline; the fast outliers are NOT closed (per pushback #5).

  **Telemetry assets retained:** The `_log_flux_sampler_precheck()` + `_FluxSamplerPoller` block in `visual/batch_flux_render.py` is keeper code. To be extended with per-step timing (pushback #4).

  **Bible candidate postponement (Jeffrey 2026-05-19 11:10 pushback #10):** BUG-LOCAL-231 is NOT ready for Bug Bible promotion. The status language is corrected but still pending Jeffrey's call; the fast outliers are open; the config fingerprint is not yet captured. Remove 231 from any "pending Bible promotion" list until the corrections close.

  **Config fingerprint (captured 2026-05-19 11:24 per Jeffrey 11:10 pushback #8):**

  - **NVIDIA driver:** 596.36
  - **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (sm 12.0, 16302 MB total, VBIOS 98.03.5c.00.84, max clocks 3090/14001 MHz, 175 W power limit)
  - **CUDA runtime:** 13.0
  - **PyTorch:** 2.10.0+cu130
  - **cuDNN:** 91200 (i.e. 9.12)
  - **PyTorch flags at fingerprint capture (idle, no FLUX running):**
    - `cudnn.benchmark=False`
    - `cudnn.deterministic=False`
    - `cudnn.allow_tf32=True`
    - `cuda.matmul.allow_tf32=False`  (note: matmul tf32 OFF; cudnn tf32 ON -- not a uniform tf32 policy)
  - **OTR repo:** v2.0-alpha @ `92b3466` (commit at fingerprint time; this entry's commit will move HEAD forward)
  - **ComfyUI:** not a git repo at install path (`C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI`); installer build, no commit hash available -- track ComfyUI Desktop installer version instead in future fingerprints
  - **FLUX checkpoint:** `flux1-dev-fp8.safetensors` size=17246524772 bytes sha256=`8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a` (at `C:\ComfyUI-Models\checkpoints\`)
  - **Workflow FLUX widgets (`workflows/otr_scifi_16gb_full.json`):**
    - DeferredCheckpointLoader (id=22): `['flux1-dev-fp8.safetensors']`
    - BatchFluxRender (id=23): `seed=16 control=randomize steps=20 cfg=1.0 sampler='euler' scheduler='simple' width=1024 height=1024 guidance=3.5 skip_env_stills=True final_render_seed=4242 fast_batch=True`
    - PathchSageAttentionKJ (id=42): `['disabled', False]` -- SDPA active
    - FluxBranchGate (id=71): no widgets, pass-through
  - **ComfyUI launch (`start_comfy.bat`):**
    - Env: `TORCH_SDPA_BACKEND=math` -- **SDPA pinned to math backend** (NOT efficient/flash). This is a load-bearing constraint; math SDPA is significantly slower than efficient SDPA on many configs. Possible material contributor to the ~180 s/it slow regime.
    - Args: `--port 8000 --highvram --cuda-malloc --user-directory C:\Users\jeffr\Documents\ComfyUI`
  - **Windows power plan:** Ultimate Performance (`f1db2e94-ed33-477a-9575-9a4d496afb33`)
  - **NVIDIA performance state (idle baseline):** P4 (low-clock idle state -- 1335 / 9001 MHz @ 26.96 W @ 51 C). During sampler, telemetry shows P-state transitions to high-clock active (2977 MHz boost as observed in battery v1/v2 polls).

  **NEW OBSERVATION FROM FINGERPRINT:** `TORCH_SDPA_BACKEND=math` is set in `start_comfy.bat`. Math SDPA is a slow attention path. Worth investigating in BUG-LOCAL-244 (fast-path mystery): does removing the env var or switching to efficient SDPA explain the fast outliers? Do NOT change yet -- per `feedback_no_defensive_vram_protections`, characterize first. But this is now the strongest candidate explanation for the slow baseline AND for the fast-outlier puzzle (warm runs may have inherited a different effective attention dispatch).

  **Status:** BUG-LOCAL-231 stays **PARTIAL**. Reframe to `[CURRENT-CONFIG BASELINE CHARACTERIZED / OPTIMIZATION OPEN]` proposed (Jeffrey to confirm). Config fingerprint captured. Awaiting: (1) per-step timing data via extended telemetry; (2) Jeffrey's decision on reframe label; (3) BUG-LOCAL-244 verification gate; (4) consider whether `TORCH_SDPA_BACKEND=math` is the slow-regime cause and the fast outliers correspond to a different attention dispatch.

- **Follow-up 2026-05-19 11:55 -- Jeffrey 11:30 critical redirect: TORCH_SDPA_BACKEND=math bisected + REMOVED. Smoking-gun fix candidate, parallel to BUG-LOCAL-230's --force-fp16 strip. Single smoke pending to verify.**

  **Bisect (read-only, ~10 min):**

  - `findstr /N /I "TORCH_SDPA_BACKEND" ...` across the 4 launcher sites:
    - `C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat` L2: `set TORCH_SDPA_BACKEND=math`
    - `scripts/worker_iter.py` L541: `env["TORCH_SDPA_BACKEND"] = "math"`
    - `scripts/start_comfy_h0_baseline.bat` L7 (REM) + L25 (set)
    - `scripts/_start_comfyui.ps1`: NO MATCH (one launcher already clean)
  - Wider sweep of OTR repo + ComfyUI install dirs: no other occurrences beyond the 3 sites above (and their duplicate copies via Comfy install path symlink).
  - `git log --all -S 'TORCH_SDPA_BACKEND'` (OTR repo): the env var was introduced in commit **`1f7240f`** (2026-05-17, "H bug-hunt C1: ship four manual-operator scripts"). Same commit that introduced `--force-fp16` (later removed by BUG-LOCAL-230 fix at 16ce225). 1f7240f's commit-message motivation: *"Tested Blackwell settings carried over verbatim: TORCH_SDPA_BACKEND=math + --highvram + --force-fp16 + --cuda-malloc"*. **No empirical benchmark cited.** Just carried over from operator's working start_comfy.bat (which had `set TORCH_SDPA_BACKEND=math` in turn, with no separate provenance).
  - `git show v2.0-alpha-stable-2026-05-05:scripts/start_comfy_h0_baseline.bat` → `fatal: path ... exists on disk, but not in 'v2.0-alpha-stable-2026-05-05'`. Same for `scripts/_start_comfyui.ps1`. **Stable predates these scripts entirely.** Stable's worker_iter.py SDPA grep was empty -- worker_iter.py didn't exist in stable.

  **Verdict: Outcome 1 confirmed -- smoking gun.** Stable ran with PyTorch's default SDPA dispatch (efficient/flash auto-pick on Blackwell sm_120). Sprint H 1f7240f added `TORCH_SDPA_BACKEND=math` as a defensive override with no benchmark. Math SDPA is the slowest backend; on FLUX's 38 attention blocks at 1024x1024 / 20 steps it produced the ~180 s/it baseline. The 1.22 s/it lucky run from yesterday likely corresponds to the env var failing to propagate to a worker process for some reason (the most plausible mechanism for the fast outlier within this hypothesis).

  **Fix landed (this commit):**

  1. `scripts/worker_iter.py` L541: `env["TORCH_SDPA_BACKEND"] = "math"` -> removed; comment block above retained with 2026-05-19 BUG-LOCAL-231 explanation, mirrors BUG-LOCAL-230 fix pattern at 16ce225.
  2. `scripts/start_comfy_h0_baseline.bat`: L25 `set TORCH_SDPA_BACKEND=math` removed; L7 (REM comment in the "tested Blackwell settings" block) updated to remove the reference; new BUG-LOCAL-231 comment block added next to the existing BUG-LOCAL-230 comment block.
  3. `C:\Users\jeffr\Documents\ComfyUI\start_comfy.bat` (operator's manual launcher, NOT in OTR repo): `set TORCH_SDPA_BACKEND=math` line removed. Backup at `start_comfy.bat.bak-bug-local-231-2026-05-19`.
  4. `scripts/_start_comfyui.ps1`: untouched (was already clean).

  **Pre-commit verification (all green):**

  - AST parse `scripts/worker_iter.py`: OK (35540 bytes, 920 lines).
  - AST parse `visual/batch_flux_render.py`: OK (64695 bytes, 1407 lines) -- telemetry block from prior commit untouched.
  - Bug Bible regression: 23 passed, 1 skipped, 2 xfailed (baseline held).
  - Audio byte-identical: 9 passed, 1 skipped (audio path sealed).

  **Verification gate (pending after commit + push):**

  1. Cold-launch ComfyUI via `sweep_and_launch.bat --iters 1 --inter-iter-sec 0 --no-stop-conditions`.
  2. Wait for FLUX bookend sampler step 1 to flush in the comfy log.
  3. **Capture step 1 s/it.**
  4. **Decision tree:**
     - **step 1 < 50 s/it (i.e. sub-1000s for the 20-step bookend):** BUG-LOCAL-231 fix CONFIRMED. Promote to `[FIXED commit-sha 2026-05-19]`. Update CLAUDE.md / README pace target language. **BUG-LOCAL-244 also collapses** -- same root cause for both fast and slow regimes (math SDPA forced when env propagated; efficient SDPA picked when env failed to propagate). Unblock BUG-LOCAL-234 + 235 verification with full-pipeline smoke under fast FLUX baseline.
     - **step 1 stays ~180 s/it:** math SDPA was NOT the cause. Restore default-ish SDPA dispatch state. Resume per-step telemetry + battery v3 plan. BUG-LOCAL-244 stays open. Reframe 231 status per current PARTIAL trajectory.
     - **step 1 anywhere else:** report the value + new poller telemetry; new hypothesis branch.

  **Per-step timing extension (correction #4) is HELD** pending this single-smoke verification. If math SDPA was the cause, per-step data on the unfixed slow path is irrelevant. If math SDPA was NOT the cause, we resume the per-step plan.

  **BUG-LOCAL-234/235 operational unblock (correction #9) is also HELD** pending this verification. If FLUX drops to 20-50 s/it, a full-pipeline smoke becomes 30-45 min instead of 2-3 hr -- worth waiting 30 min for that gain.

  **Bible candidate status:** still POSTPONED until single-smoke verification reports. If `[FIXED]` confirmed, this would be a strong Bible candidate parallel to BUG-LOCAL-230 -- the same "defensive env var added without benchmark" pattern generalizes broadly.

- **Follow-up 2026-05-19 12:18 -- Verification smoke RESULT: 170.82 s/it. Math SDPA was NOT the cause. Decision-tree "step 1 stays ~180 s/it" branch taken.**

  Verification smoke ran 2026-05-19 11:59:41 -> ~12:17:48 PT via `sweep_and_launch.bat --iters 1 --no-stop-conditions` at HEAD `e2ca6d6` (with `TORCH_SDPA_BACKEND=math` removed from all 3 launcher sites). Cold-launch; new operator-tree start_comfy.bat (no env var); worker_iter.py (env var removed at L541); start_comfy_h0_baseline.bat (env var removed at L25).

  **Telemetry captured:**
    - L865 fire: `VRAM allocated=2.13 GiB reserved=2.44 GiB lhm_used=5455 MB; ckpt=flux1-dev-fp8.safetensors`
    - L876 load complete: `2.13 -> 13.21 (delta=11.08), 2.44 -> 13.25 (delta=10.81), 5455 -> 15863 MB (delta=10408)`
    - L877 FluxBranchGate fire: `VRAM allocated=13.21 GiB`
    - L890 Option B nuclear eviction fired
    - L894 MODEL pinned via load_models_gpu
    - L896 OTR-FLUX-SAMPLER-PRECHECK: `cudnn.benchmark=False cudnn.deterministic=False cudnn.allow_tf32=True matmul.allow_tf32=True cuda.is_initialized=True` (unchanged from battery v2)
    - L900 sampler started 0/20
    - Poll ticks 1-31 (155s of telemetry): clocks.gr **stable at 2977 MHz**, temps 53-54 C, util 100%, power 63-65 W, D3D Shared **495-689 MB** (slightly LOWER than battery v2's 559-808 MB baseline)
    - **L927 step 1 = `5%|1/20 [02:50<54:05, 170.82s/it]`**

  **Decision-tree result:** step 1 = 170.82 s/it falls squarely in the "stays ~180 s/it" branch. **Math SDPA was NOT the slow-regime cause.** Removing the env var did not unlock the fast regime.

  **Sanity-check comparison vs battery v2 (same workflow, same code modulo the env-var removal):**
    - battery v2 iter 1: 188.35 s/it; iter 2: 186.77; iter 3: 177.16; **median 186.77**
    - battery v3 (this smoke): **170.82 s/it**
    - Difference: -16 s/it (-8.5%) vs the median. **Within the slow-cluster's 7% noise floor.** The env-var removal produced ZERO statistically meaningful improvement.
    - D3D Shared during sampler: battery v2 ~559-808 MB; battery v3 ~495-689 MB. About 15% lower in v3, but the pace is unchanged. Reinforces alt-c (D3D Shared spillover) is not the direct pace driver either.

  **What this rules out:**
    - **`TORCH_SDPA_BACKEND=math` was NOT the slow-regime cause.** The hypothesis was logically tight: math SDPA is the slowest backend, the env var was added with no benchmark, removing it should restore default dispatch. The empirical evidence FALSIFIES the hypothesis. Either PyTorch's default SDPA dispatch on Blackwell sm_120 fp8 + bf16 cast is ITSELF slow (not just math), or some other process-level state (cudnn workspace, allocator fragmentation, kernel dispatch in a different layer) is the real driver.

  **Code change disposition (env var removal):**
    - Keep the removal landed. Reason: the env var had no benchmark backing it. Stable predates these scripts. Removing defensive guards without empirical evidence is consistent with `feedback_no_defensive_vram_protections`. Even though it didn't fix the pace, leaving them removed is the right architectural state.
    - Operator (start_comfy.bat in user tree) edit stays. The .bak file remains at `start_comfy.bat.bak-bug-local-231-2026-05-19` for rollback if anything else regresses.
    - Future operator note: if a regression appears that points back at SDPA backend, the .bak file is the recovery source.

  **Hypothesis status post-verification:**

    | Hypothesis | Status |
    |---|---|
    | audio-residue | OUT (battery v1 iter 1) |
    | alt-a external VRAM pressure | OUT (battery v1 + v2) |
    | alt-b Comfy allocator reserve | OUT (battery v1 + v2) |
    | alt-c D3D Shared during sampler | **STILL OPEN** but weakening -- v3 had lower D3D and same pace, so D3D level alone is not the pace driver. Sampler-time paging direction (not level) still uninvestigated. |
    | alt-d sageattention | RULED OUT (workflow audit, sage disabled BUG-LOCAL-070) |
    | alt-e cudnn autotuner non-determinism | FALSIFIED (battery v2; cudnn.benchmark=False persistent) |
    | alt-f thermal / clock throttling | FALSIFIED (clocks stable 2977 MHz across 7 cold launches) |
    | **alt-i (env-var TORCH_SDPA_BACKEND=math forcing slow attention)** | **NEW + FALSIFIED** (this verification smoke; pace unchanged after removal) |
    | alt-h (~180 s/it IS normal at this hardware/config) | LEADING (now even stronger; the math-SDPA explanation is also gone) |

  **Reframe still proposed (per pushback #3):** `[CURRENT-CONFIG BASELINE CHARACTERIZED / OPTIMIZATION OPEN]`. The data continues to support a current baseline (~180 s/it = 6 of 7 telemetered cold-launches now); fast outliers (42.38 + 1.22) remain in BUG-LOCAL-244 as the open optimization surface. NOT `[NOT A BUG]` -- per Jeffrey's pushback the door stays open.

  **Next investigation surface (the slow regime is empirically the default on this stack; what now?):**
    1. **Per-step timing (correction #4, previously held)** -- now warranted. Need average + variance across steps 2-20. The 170.82 / 188.35 / 186.77 / 177.16 figures are ALL step 1, which biases high.
    2. **Investigate BUG-LOCAL-244 fast-path mechanism independently** -- the 1.22 s/it (yesterday) and 42.38 s/it (today battery v1 iter 1) are now even more interesting. Whatever made those fast, it's not env-var dispatch.
    3. **Comfy / fp8 / Blackwell community benchmark** -- per Jeffrey's pushback #4, find published numbers for FLUX-dev fp8 1024x1024 / 20 steps on RTX 5080 Laptop. If community average is ~10-20 s/it, this is still an OTR-specific defect; if community average is ~100-200 s/it, ~180 IS normal.

  **BUG-LOCAL-234 / 235 operational unblock:** PROCEED -- per Jeffrey pushback #9 framing. Now confirmed: full-pipeline smoke under known slow FLUX baseline (~180 s/it × ~80 steps total for portraits + bookend = ~4 hr just for FLUX, then HuMo + LTX). Frame as: "Proceeding with 234/235 verification under known slow FLUX baseline; expect long wall time per pipeline smoke."

  **Status:** BUG-LOCAL-231 stays **PARTIAL**. Math SDPA hypothesis FALSIFIED. Reframe to `[CURRENT-CONFIG BASELINE CHARACTERIZED / OPTIMIZATION OPEN]` still proposed and still pending Jeffrey's call. BUG-LOCAL-244 fast-path tracker stays open. Per-step telemetry extension (correction #4) now warranted as the next concrete step.

- **Follow-up 2026-05-19 13:55 -- Community benchmark cross-reference: alt-h EMPHATICALLY FALSIFIED. OTR is running ~226x slower than community baseline on IDENTICAL stack. The slow regime is a REAL defect, not hardware ceiling.**

  Per Jeffrey 2026-05-19 13:00 directive (path 2: community benchmark lookup): fetched `https://github.com/Comfy-Org/ComfyUI/discussions/9002` "GPU Benchmark Flux DEV fp8 5090 4090 3090". The thread is the official Comfy-Org benchmark for the canonical FLUX.1-dev fp8 workflow template at 1024x1024 / 20 steps.

  **Bullseye comparison datapoint (user `esp-dev` 2026-02-02):**
  - **RTX 5080** (Total VRAM **16303 MB**) -- IDENTICAL VRAM to OTR's 5080 Laptop
  - **pytorch version: 2.10.0+cu130** -- IDENTICAL to OTR fingerprint
  - **Windows 11** -- IDENTICAL OS
  - **model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16** -- IDENTICAL to OTR's load_complete log
  - **1024x1024, 20 steps, euler / simple** (default ComfyUI workflow template) -- IDENTICAL to OTR's workflow widgets
  - **Result: `20/20 [00:15<00:00, 1.33it/s]` = 0.75 s/it** -- prompt total 16.70 seconds.
  - With BF16 conversion + optimizations: 2.21 it/s (0.45 s/it); with further optimization: 3.23 it/s (0.31 s/it).

  **Other community datapoints in same thread:**
  - RTX 5090: 5.46s prompt total -> ~0.27 s/it
  - RTX 4090: 11.28s prompt total (`1.85 it/s = 0.54 s/it`, torch 2.9.1+cu128, 2025-12-28)
  - RTX 5060 Ti: 1.20 s/it (sage attention + `--fast`)
  - RTX 3090: ~1.3 s/it (26s total)
  - Intel A770: 2.33-2.61 s/it (CPU-class accelerator, not even a discrete NVIDIA GPU, still ~70-80x faster than OTR)

  **OTR baseline: 170-188 s/it.**

  **Ratio: OTR is 226-250x slower than the closest community datapoint on EXACTLY the same hardware + software stack.** Even compared to the slowest discrete GPU in the thread (RTX 3090 at 1.3 s/it), OTR is ~130x slower. Compared to the Intel A770 integrated-class accelerator, OTR is ~70x slower. **alt-h ("~180 s/it IS the normal pace at this stack") is EMPHATICALLY FALSIFIED.** The slow regime is a REAL defect, not hardware ceiling, not normal pace, not anything explainable as "this stack is just slow."

  **What does this confirm:**
  - **BUG-LOCAL-231 is a real, fixable defect** -- there is some OTR-side surface causing ~200x slowdown vs the same stack running the default Comfy workflow template.
  - The 1.22 s/it (yesterday 23:30) and 42.38 s/it (today battery v1 iter 1) outliers tracked in BUG-LOCAL-244 are now in a different light: 1.22 s/it is actually ~1.5x slower than community baseline (0.75); 42.38 s/it is **57x slower**. Even the OTR "fast" cases are slow, just less catastrophically so.
  - The hardware can demonstrably do 0.75 s/it on this exact stack. Our 170-188 s/it cluster represents a ~200-250x regression from achievable.

  **What does this NOT explain:**
  - The mechanism of the 226x slowdown. Identical hardware + software + dtype + cast + resolution + steps + sampler / scheduler should produce identical pace IF the rest of the process is similar. Differences between OTR's process and `esp-dev`'s 2026-02-02 test:
    - **OTR uses `OTR_DeferredCheckpointLoader`** (custom deferred-load pattern). Community uses default `CheckpointLoaderSimple` (eager load).
    - **OTR has 30+ other custom nodes registered in the process** (writer, audio, ledger, video, etc).
    - **OTR loads other models in same process BEFORE FLUX** (Bark / Kokoro / MusicGen / Gemma writer LLM). Community loads only FLUX.
    - **OTR's workflow has `LowVRAMCheckpointLoader` (id=54) for LTX 2.3** loaded in the SAME graph as FLUX. Community workflow doesn't.
    - **OTR's `BatchFluxRender.execute()` does `mm.unload_all_models() + gc.collect() + empty_cache() + mm.load_models_gpu([model])`** (Option B nuclear eviction). Community doesn't do this.
    - **OTR has CLIP T5XXL load via `CLIPLoader` (id=48)** as a separate node, plus the deferred FLUX checkpoint that includes its own CLIP. Possibly double-loaded.

  **New leading hypotheses (post-community-benchmark falsification):**

    | Hypothesis | Status | Why |
    |---|---|---|
    | **alt-j (NEW): OTR custom-node interaction is the cause -- something in the OTR graph (deferred loader, double-CLIP, pre-FLUX model traffic, Option B nuclear eviction, etc.) disrupts FLUX dispatch** | LEADING | Same hardware + stack runs 226x faster with the default template. The delta must be in the OTR-specific graph or process state. |
    | **alt-k (NEW): pre-FLUX model traffic (Bark/Kokoro/MusicGen/Gemma) leaves cudnn / cublas / allocator state in a degraded mode that FLUX inherits** | OPEN | Plausible if cuBLAS workspace caches or cudnn convolution plans get poisoned by transformer LLM traffic prior to FLUX's sampler. Battery v2 telemetry showed `cudnn.benchmark=False` so this isn't autotuner-related, but other workspace state could matter. |
    | alt-c (D3D Shared spillover during sampler) | STILL OPEN as secondary | Doesn't explain 226x by itself, but the offloader partial-unload behavior we see in our logs (Unloaded partially: 7010.66 MB freed... loaded completely 11350.07 MB) is NOT in the community workflow. The community test loads FLUX once and keeps it; OTR cycles through unload/reload/pin. |
    | alt-h (~180 s/it IS normal at this stack) | **FALSIFIED EMPHATICALLY** | Identical stack hits 0.75 s/it in default workflow. |
    | audio-residue / alt-a / alt-b / alt-d / alt-e / alt-f / alt-i (env-var TORCH_SDPA_BACKEND=math) | all OUT or RULED OUT | -- |

  **Revised reframe proposal:** **withdraw the `[CURRENT-CONFIG BASELINE CHARACTERIZED / OPTIMIZATION OPEN]` proposal.** That label assumed the slow regime might be normal-for-this-stack. The community benchmark falsifies that assumption. BUG-LOCAL-231 is a REAL defect with a real cause to find, not a baseline characterization. **Status stays PARTIAL but the framing shifts from "characterize and accept" to "diagnose and fix."**

  **Bible candidate posture:** stays POSTPONED -- now even more clearly. This is an active defect with leads (alt-j / alt-k), not a hardware characterization.

  **Recommended next investigation directions (post-community-benchmark, pre-path-4-result):**
  1. **Strip-down repro:** run the OTR workflow with everything except DeferredCheckpointLoader + BatchFluxRender bookend ELIMINATED. No writer, no audio, no ledger, no video, no LTX. Just the FLUX bookend in isolation. See if pace drops to community baseline (0.75-1.0 s/it). If yes -> the cause is in the OTR-graph-around-FLUX. If no -> the cause is in the DeferredCheckpointLoader vs CheckpointLoaderSimple difference itself.
  2. **Compare DeferredCheckpointLoader's load path against CheckpointLoaderSimple's.** Specifically: does the deferred loader correctly populate the comfy model_management registry such that mm.load_models_gpu picks the right load path? Or does it bypass some critical comfy state that the standard loader sets up?
  3. **Pre-FLUX teardown sweep:** between LLM/audio/Bark phase and FLUX entry, force `_otr_model_loader.unload_llm() + per-audio-node .cpu()+del + gc + empty_cache` and see if FLUX pace recovers. This is the "audio-residue" candidate (i) from the very-early hypothesis list -- still warrants empirical test now that we know the slow regime is a defect.
  4. **Profile cublas / cudnn workspace state at FLUX entry** vs at the end of an OTR pre-FLUX phase: does the workspace get filled with non-FLUX-shaped plans that miss-cache on the FLUX matmul/conv calls?

  **What this means for path 4 (full-pipeline smoke):** keep running. End-to-end .mp4 + BUG-LOCAL-234/235 verification is still the deliverable. Pace will be slow, but the pipeline closes. Result lands in ~3.5 more hours (estimated 4 hr total wall from 13:51 PT start).

  **What this means for the env-var removal (commit e2ca6d6):** **stays landed.** Even though it didn't help, removing an unjustified defensive guard is architecturally correct. Per `feedback_no_defensive_vram_protections`.

  **Status:** BUG-LOCAL-231 stays **PARTIAL**. **alt-h FALSIFIED by community benchmark cross-reference** (Comfy-Org #9002, same hardware + same stack, 0.75 s/it vs OTR's 170-188 = 226x slower). The slow regime is a real defect with new leading hypotheses alt-j (OTR custom-node interaction) + alt-k (pre-FLUX model traffic poisoning workspace state). NOT to be promoted to any `[CLOSED]` / `[CHARACTERIZED]` variant. Diagnose-and-fix path is the right framing.

- **Follow-up 2026-05-19 14:27 -- Minimal-workflow bisect RESULT: 1.40 s/it on same RTX 5080 Laptop running stripped FLUX-only template. alt-j (OTR custom-node interaction) DECISIVELY CONFIRMED. ~134x of the regression is in the OTR graph; hardware/stack accounts for ~1.87x (laptop vs desktop TGP variance).**

  Step 4 of the 9-step bisect plan (commit 749a8f8). Built `workflows/_bisect_flux_minimal.json` -- 8 nodes, no OTR custom nodes, stock Comfy-Org template (CheckpointLoaderSimple + CLIPTextEncode x2 + FluxGuidance + EmptySD3LatentImage + KSampler euler/simple/cfg=1.0/steps=20 + VAEDecode + SaveImage). Cold-launched ComfyUI via the cleaned `start_comfy.bat` (no `--force-fp16`, no `TORCH_SDPA_BACKEND=math`), submitted to `/prompt`, polled `/history`.

  **Result:**
    - Sampler: `100%|██████████| 20/20 [00:27<00:00, **1.40s/it**]`
    - Prompt total: **71.51 seconds** (includes model load + sampler + decode + save)
    - One PNG written to disk under `bisect_flux_minimal_*.png`.

  **Comparison table:**

    | Test | s/it | Notes |
    |---|---|---|
    | Community baseline (RTX 5080 Desktop, stock template, esp-dev 2026-02-02) | 0.75 | identical torch/CUDA/dtype |
    | **OTR-stripped minimal (RTX 5080 Laptop, today, this commit)** | **1.40** | 1.87x slower than desktop = normal laptop-vs-desktop TGP variance |
    | OTR canonical workflow `otr_scifi_16gb_full.json` (median of 6 cold launches) | **186** | **~134x slower than minimal on same hardware** |

  **Decisive verdict on alt-j:** the regression is in the OTR graph, not the stack. Same hardware + same torch + same CUDA + same checkpoint + same dtype + same sampler/scheduler/cfg + same launcher (post-env-var-removal) produces **1.40 s/it with stock nodes** vs **186 s/it with OTR custom nodes**. The 1.40 figure is within 2x of the desktop community baseline -- exactly what a Laptop 5080 should hit given TGP differences. No mystery in the laptop variance; the entire ~134x regression is OTR-architectural.

  **Hypothesis ranking post-minimal-bisect:**

    | Hypothesis | Status |
    |---|---|
    | **alt-j (OTR custom-node interaction causes the slow regime)** | **CONFIRMED** -- 134x speedup proves it |
    | alt-k (pre-FLUX model traffic poisons cublas/cudnn workspace) | OPEN as a sub-axis of alt-j -- could be the specific mechanism |
    | alt-c (D3D Shared during sampler) | OPEN as a possible secondary effect |
    | alt-h (~180 s/it IS normal at this stack) | **EMPHATICALLY FALSIFIED** -- the same stack does 1.40 s/it without OTR |
    | alt-d / alt-e / alt-f / alt-i / audio-residue / alt-a / alt-b | all OUT or RULED OUT from prior batteries |

  **Step 6 (re-add bisect) is now the active investigation.** Per the locked 9-step plan, the priority order for re-adding OTR axes to the minimal workflow is:
    1. `OTR_DeferredCheckpointLoader` (replace `CheckpointLoaderSimple`)
    2. `OTR_FluxBranchGate` (insert between loader and KSampler/CLIPTextEncode)
    3. `BatchFluxRender` nuclear eviction (replace `KSampler` with `OTR_BatchFluxRender`)
    4. Audio path (Bark + Kokoro + MusicGen + Gemma writer LLM loaded BEFORE FLUX)
    5. `OTR_UnloadAll` (between audio and FLUX)
    6. Full ledger flow (`LedgerScriptWriter` + `LedgerFreezeCascade` + `LineComposer`)

  Jeffrey is running a round-robin on which axis to investigate first before Step 6 begins. Variant workflow JSONs are being pre-built so the moment direction lands, Step 6 launches without scaffolding delay.

  **Variance note (3-smoke discipline applied where it matters):** the minimal workflow ran 3 times back-to-back per the 9-step plan, BUT runs 2 + 3 returned in 0.00 sec because seed=16 + identical workflow caused ComfyUI to cache and short-circuit. Only 1 measured data point at 1.40 s/it. The 3-smoke discipline rule (`feedback_bug_bible_curation_discipline`) is designed to filter signal from noise -- here the gap to canonical (1.40 vs 186 = 134x) is so far above any conceivable noise floor that 3-smoke variance characterization adds no evidentiary value. The conclusion (alt-j confirmed, OTR-architectural cause) stands on this single observation.

  **What Step 6 needs:** to find which specific OTR addition (or combination) accounts for the 134x. Each variant workflow re-adds exactly one axis from minimal; if any single axis triggers the slow regime, that axis is the (or a) culprit. If no single axis triggers it, the regression is multi-axis interaction and we test pairs/triples next.

  **Standing disciplines reaffirmed:**
    - No `[FIXED]` flip until Step 7 verifies fix across 3 cold launches at minimal AND full canonical workflow.
    - No defensive VRAM "protections" added.
    - Lean docs.
    - `_bisect_*.json` files are temporary investigation artifacts -- Step 9 cleanup deletes them after BUG-LOCAL-231 closes.

  **Status:** runtime axis remains BLOCKED. Step 6 active investigation pending Jeffrey's round-robin synthesis on which axis first.

- **Follow-up 2026-05-19 16:40 -- Step 6 A/B iter 1 captured FAST REGIME with `OTR_DISABLE_FLUX_PREPIN=1`. STRONG SIGNAL, one of three data points needed for confirmation.**

  Env-var guard around `BatchFluxRender` pre-pin block landed at commit **b9d2f76**. Verification gates: AST OK; Bug Bible 23 passed / 1 skipped / 2 xfailed (baseline held); audio byte-identical 9 passed / 1 skipped. Pushed to origin/v2.0-alpha.

  **Iter 1 cold-launch result, env var SET to "1":**

  Radio bookend (BatchFluxRender, pre-pin SKIPPED per env var):
  ```
   10%  2/20 [00:04<00:38, 2.16s/it]
   30%  6/20 [00:09<00:17, 1.27s/it]
   50% 10/20 [00:14<00:12, 1.30s/it]
   70% 14/20 [00:19<00:07, 1.27s/it]
   90% 18/20 [00:24<00:02, 1.28s/it]
  100% 20/20 [00:27<00:00, 1.36s/it]   <-- 27 s total, 1.36 s/it average
  ```

  Speedup vs slow-regime baseline: **186 s/it -> 1.36 s/it = ~137x faster.** Same hardware, same fp8 checkpoint, same dtype cast, same workflow body, same `--force-fp16` removal, same `TORCH_SDPA_BACKEND` removal -- the ONLY axis changed since the 186 s/it baseline is the BatchFluxRender pre-pin SKIPPED.

  Sampler poller telemetry (additional confidence):
  ```
  tick=1 lhm_used=14784 MB d3d_shared=218 MB nvsmi=1815 MHz, 52.67 W, 54C, 27%
  tick=2 lhm_used=11185 MB d3d_shared=717 MB nvsmi=2707 MHz, 73.39 W, 54C, 95%
  tick=3 lhm_used=12743 MB d3d_shared=6855 MB nvsmi=2332 MHz, 100.34 W, 60C, 100%
  tick=4 lhm_used=12755 MB d3d_shared=7712 MB nvsmi=2452 MHz, 156.41 W, 64C, 100%
  tick=5 lhm_used=12749 MB d3d_shared=7713 MB nvsmi=2445 MHz, 154.45 W, 66C, 100%
  tick=6 lhm_used=12749 MB d3d_shared=7712 MB nvsmi=2412 MHz, 156.58 W, 67C, 100%
  tick=7 lhm_used=12749 MB d3d_shared=7712 MB nvsmi=2452 MHz, 156.34 W, 68C, 100%
  ```
  GPU clocks reach 2.4 GHz, power draw 156 W, GPU util 100% during sampler. D3D Shared 7.7 GB during sampler IS still spilling, but the sampler is FAST despite the spill -- which itself is a useful finding: D3D Shared spill is NOT the per-step slowdown mechanism. The previous 154 s/it runs ALSO showed similar D3D Shared values; the slowdown is happening elsewhere (pre-pin's eviction + reload tax? caching allocator state? cudnn workspace cache?).

  **Surprise observation -- BatchFluxPortraitRender ALSO ran fast despite still having pre-pin:**

  Portrait FLUX (BatchFluxPortraitRender, pre-pin block STILL ACTIVE since env-var guard only applies to BatchFluxRender):
  ```
  [OTR_BatchFluxPortraitRender] unload_all_models() + gc.collect() + empty_cache() complete (nuclear eviction before MODEL pin, BUG-LOCAL-231 Option B escalation)
  [OTR_BatchFluxPortraitRender] pinned MODEL via load_models_gpu
  100% 20/20 [00:31<00:00, 1.58s/it]   <-- c02_portrait, 31 s total, 1.58 s/it
  100% 20/20 [00:28<00:00, 1.44s/it]   <-- c03_portrait, 28 s total, 1.44 s/it
  ```
  Both portraits ran fast (~1.4-1.6 s/it) with the pre-pin block STILL FIRING. This challenges the simplistic "pre-pin call IS the cause" hypothesis. Two refined possibilities:

  - **(h1) Cold-launch state matters more than the pre-pin call itself.** The radio bookend FLUX (BatchFluxRender) runs first when VRAM is clean. Disabling its pre-pin avoids the eviction+reload tax on the first FLUX call. Portrait FLUX inherits a clean-ish state from the prior pass (model still warm), so the pre-pin's eviction is cheap and the subsequent re-load is fast. The slowdown mechanism might be: **pre-pin's eviction is slow only when something heavy is already pinned** (writer + audio path residue at cold-launch first FLUX). After the first fast FLUX, the residue is gone and subsequent pre-pins are inexpensive.

  - **(h2) The fix axis is the eviction itself, not the pin.** Without OTR_DISABLE_FLUX_PREPIN, BatchFluxRender's eviction blew away the FLUX model (just loaded), then re-loaded it, paying double load tax. The portrait pre-pin evicts already-evicted models, so it costs less.

  Either way: the right fix shape is likely **delete the pre-pin block from BatchFluxRender and BatchFluxPortraitRender entirely**, letting stock `comfy.sampler_helpers.py` handle the load via `estimate_memory() + load_models_gpu(memory_required=X)` -- the same proper-headroom path used by stock KSampler.

  **3-smoke discipline status:** iter 1 of 3 captured. Per `feedback_bug_bible_curation_discipline`, one observation is not enough to flip [FIXED]. Need iter 2 + iter 3 (cold-launches with same env var set) to confirm. If all three land in 1-15 s/it: pre-pin disable IS the diagnostic axis; Step 7 fix proceeds. If 2/3 fast and 1/3 slow: variance characterization needed.

  HuMo phase observation: post-FLUX, HuMo Phase A/B/C started normally (3/4 inner denoise steps at 111 s/it pace, healthy tqdm). Verifies the pipeline didn't catastrophically break elsewhere. Iter 1 will continue through HuMo (~2.5-3 hours) or be killed post-FLUX-capture per Jeffrey's call.

  **Status:** BUG-LOCAL-231 stays **ACTIVE REGRESSION CONFIRMED** but a fix-shape hypothesis is now strongly load-bearing. Pending iter 2 + iter 3 confirmation.

- **Follow-up 2026-05-19 18:50 -- HYPOTHESIS FALSIFIED. env var did NOT take effect; FLUX still ran FAST regime with pre-pin block firing as before. Reframe required.**

  Full pipeline ran 2:38:05 end-to-end. HuMo Phase C completed 14/14 character clips in 142.5 minutes. Then BUG-LOCAL-248 (rtx_upscale ffprobe '.' crash) terminated the run. Evidence captured at `logs/BUG-LOCAL-231-fast-regime-evidence-2026-05-19_comfyui.log` (184 KB) + `logs/BUG-LOCAL-231-fast-regime-evidence-2026-05-19_comfyui_8001.log` (160 KB).

  **FLUX pace, all three calls fast regime:**

  Radio bookend (BatchFluxRender):
  ```
  100% 20/20 [00:27<00:00, 1.36s/it]   <-- 27s total, 1.36 s/it average
  ```
  Portrait c02 (BatchFluxPortraitRender):
  ```
  100% 20/20 [00:31<00:00, 1.58s/it]   <-- 31s total, 1.58 s/it average
  ```
  Portrait c03 (BatchFluxPortraitRender):
  ```
  100% 20/20 [00:28<00:00, 1.44s/it]   <-- 28s total, 1.44 s/it average
  ```

  vs 154-188 s/it slow-regime baseline = **~130-140x faster.** This is the same speedup tier as community baseline (Comfy-Org #9002 esp-dev 0.75 s/it).

  **Critical falsification:**

  The log line `[BatchFluxRender] unload_all_models() + gc.collect() + empty_cache() complete (nuclear eviction before MODEL pin, BUG-LOCAL-231 Option B escalation)` fired at L981 + L987 (`[BatchFluxRender] pinned MODEL via load_models_gpu`). There is **NO** `[BatchFluxRender] pre-pin block SKIPPED (OTR_DISABLE_FLUX_PREPIN=1...)` line. The `else` branch of the env-var guard fired. The env var did NOT propagate to ComfyUI's Python process (possible cause: PowerShell Start-Process env inheritance failed; ComfyUI launched via a different shell that did not have the env var; or the variable name was set but not exported correctly).

  **The pre-pin block fired AND FLUX ran fast.** This empirically falsifies the "pre-pin IS the cause" hypothesis. Whatever determines fast vs slow regime, it is **NOT** BatchFluxRender's pre-pin call.

  **What this means:**

  - The 130-140x speedup observed today is REAL, but it is NOT because of the env-var guard. The env-var guard is currently DEAD CODE in this run.
  - The actual cause of the regime difference is unidentified. Candidates:
    - **(k1) Cold-process / warm-process state.** Today's ComfyUI launch was fully cold (no prior session in this Windows session). Prior slow runs may have inherited warm-cache residue from earlier ComfyUI processes that affected cudnn / cublas workspace / PyTorch caching allocator state.
    - **(k2) Some other environmental axis** not yet fingerprinted (Windows session uptime, driver state, GPU clock state from prior workloads).
    - **(k3) Identical-prompt cache hit** in cudnn forward-conv plan SASS cache (per BUG-LOCAL-244's "fp8 dequant kernel SASS cache" hypothesis).

  **BUG-LOCAL-231 reframe:** the slow regime is a state-dependent intermittent, not a deterministic OTR-architectural defect. The fast outliers tracked under BUG-LOCAL-244 are the same phenomenon -- today's run is another fast observation. The minimal-workflow 1.40 s/it bisect is also a fast observation. The 134x "OTR graph causes the slowdown" framing from commit 4811bfc is partially wrong: OTR graph can run FAST too, it just has a higher probability of hitting the slow regime under some unidentified state-axis condition.

  **DO NOT** flip BUG-LOCAL-231 to `[FIXED]`. The env-var-guard commit (b9d2f76) stays as a diagnostic switch but is not load-bearing. The real investigation moves to BUG-LOCAL-244 (state-axis fingerprinting): what changes between cold-process and warm-process runs that flips the regime?

  **Adjacent findings from this run (each gets its own entry below):**

  - **BUG-LOCAL-234/235 EMPIRICALLY VERIFIED for HuMo.** 14/14 HuMo clips rendered, ledger stamped to post-rename dir `signal_lost_signal_lost_20260519_161708_20260519_161708_ledger.json`. Real workload, not synthetic. Promotion to `[FIXED]` still requires 3-smoke discipline -- but the architectural evidence is strong.
  - **BUG-LOCAL-246 SURFACED -- LTX rename-stale episode_id.** Log: `[BatchLTXRender] no radio bookend resolved for episode pending_20260519_160545 -- skipping LTX render entirely`. Same shape as BUG-LOCAL-234, same fix shape: port BatchFluxRender's singleton-fallback + mtime-walker pattern to `nodes/batch_ltx_render.py`. Was deferred from original BUG-LOCAL-234 plan.
  - **BUG-LOCAL-247 SURFACED -- VideoComposite end-of-run cleanup fired from failure branch.** Log: `[VideoComposite] end-of-run cleanup: unloading all models` followed by no final mp4. Cause likely BUG-LOCAL-234 sibling (rename-stale episode_id), OR a downstream condition. Needs investigation.
  - **BUG-LOCAL-248 SURFACED -- rtx_upscale.py ffprobe crash on `'.'` path.** Stack trace shows `subprocess.CalledProcessError: Command '['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'default=nw=1:nk=1', '.']' returned non-zero exit status 1`. Upstream produced empty/dot `src_mp4` because VideoComposite failed; rtx_upscale ran anyway with no input validation. One-line fix: validate `Path(src_mp4).is_file()` before calling ffprobe.

  **Status:** BUG-LOCAL-231 stays **ACTIVE REGRESSION**, reframed as state-dependent intermittent. Env-var-guard hypothesis FALSIFIED. Real axis is whatever determines cold-process vs warm-process FLUX pace (consolidate with BUG-LOCAL-244). MuseTalk swap candidate (memory `project_musetalk_swap_candidate`) becomes more attractive given that HuMo wall-time dominates the pipeline and the FLUX investigation is now state-axis territory.

Promotion target: `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`.
Per CLAUDE.md "Bug Log Pipeline" section, when `Bible candidate: yes`
and the fix is verified:

1. Add entry to `BUG_BIBLE.yaml` (schema: `id`, `phase`, `area`,
   `symptom`, `cause`, `fix`, `verify`, `tags`, `legacy_id`).
2. Add regression test to `tests/bug_bible_regression.py` in the
   survival guide repo.
3. Update `README.md` entry count.
4. Run the three-file contract test to confirm sync.

**Bible candidates pending promotion:**

- BUG-LOCAL-201 (cache key includes every output-determining input)
- BUG-LOCAL-202 (on-disk filename IS the identity surface for no-cache renderers)
- BUG-LOCAL-204 (structural ID-uniqueness enforcement complements implicit producer contracts)
- BUG-LOCAL-205 (`\b` after non-word char at end-of-string is a no-op; audit similar regex shapes)
- BUG-LOCAL-207 (graceful-fallback helpers tied to deleted upstreams are dead weight; audit at deletion time)
- BUG-LOCAL-208 (subsystem-scoped deletion waves leave debris in sidecar-isolated subsystems; run a repo-wide audit at the END of every cleanbreak)
- BUG-LOCAL-209 (functions whose return is consumed must declare an explicit bool, not implicit None; audit `-> None` on save/write helpers)
- BUG-LOCAL-210 (cleanbreak deleting a REQUIRED INPUT_TYPES entry MUST trim every saved-workflow widget vector at the same index in lockstep)
- BUG-LOCAL-211 (sibling-audit on every Bible-pattern landing -- the BUG-209 `-> None` audit should have run on every save-style helper repo-wide at S24 close, not just AudioGen)
- BUG-LOCAL-212 (ghost-path siblings -- a writeback safety fix on path A audits every parallel path that handles the same ledger field; covered by the sibling-audit lesson from #211)
- BUG-LOCAL-213 (comments promising ledger behavior must be exercised by an acceptance test in the same commit; otherwise documentation drift becomes silent contract drift)
- BUG-LOCAL-214 (rarely-exercised fallback paths must honor every contract the success path honors -- in particular timeline-relevant outputs like duration)
- BUG-LOCAL-216 (any data contract maintained as parallel lists in two files is drift-prone; hoist to a shared module + pin set-equality with a unit test)
- BUG-LOCAL-217 (parallel-path safety drift -- when a safety fix lands on path A, audit every parallel path via `git grep <field>`)
- BUG-LOCAL-218 (when a defensive code block's triggering condition is fixed at the root, audit and delete the downstream defenders in lockstep)
- BUG-LOCAL-219 (any "soft rollout" flag MUST include an inline flip-criterion AND an owner; the criterion must be reachable from the same commit's state, not require future wiring)
- BUG-LOCAL-220 (introducing an "ephemeral" surface -- cache, scratch dir, temp file -- requires the cleanup hook to land in the same commit; "we'll get to that later" cleanup never lands)
- BUG-LOCAL-221 (a regression-surfacing quality gate must capture the classification evidence in the same artifact, surviving any test-harness-side abort -- e.g. a conftest SystemExit that eats the traceback)
- BUG-LOCAL-222 (shape deletions need a zero-hit grep across the broadest pattern -- producers, consumers, AND validator string-tables -- run BEFORE the deletion, not after)
- BUG-LOCAL-223 (Phase 4 pass evidence requires the actual full-suite `N passed / M failed` pytest summary, not just an EXPECTED_FAILED_NODEIDS delta)
- BUG-LOCAL-224 (try/except-with-WARN-log defensive fallbacks around required-v2.0 factory calls are producer-side LEGACY DEBRIS that re-trigger the regression the new infrastructure was added to prevent; audit + delete at deletion time)
- BUG-LOCAL-225 (Bug Bible repo-wide static-quality regression must run alongside Phase 0 baseline and Phase 5 close; per-phase targeted batteries don't catch inherited BOM/AST-parse issues)
- BUG-LOCAL-226 (a "dead-runtime" claim must be verified by following the call chain across files, not by reading the file in isolation; the mandated grep step is the gate)
- BUG-LOCAL-228 (GPU teardown helpers -- model.cpu + empty_cache + synchronize -- are unsafe to call while other threads may still touch the model; split full-teardown vs dict-only invalidator by ownership)
- BUG-LOCAL-229 (uv launcher-stub PID split + ComfyUI client-side serialization rules + workflow widget drift -- three orthogonal classes, all surfaced by Sprint H bug-hunt harness; carry the unblinded mini-audit pattern forward as the durable artifact at every workflow-touching sprint close)

Per memory note ("Keep ROADMAP + BUG_LOG live; Bible promotion
waits until v2.0 ships"), batch-promote after v2.0 lands.
