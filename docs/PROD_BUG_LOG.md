# Production Bug Log (staging pre-Bible)

**Contract (operator, 2026-07-10; AMENDED 2026-08-07):** Claude appends entries
here AUTONOMOUSLY, but ONLY for bugs that actually failed in a live/prod run
(live render, headless lane, soak, published episode). Dev/audit/review catches
get fixed, never logged. Promoted entries get a `- promotion: BUG-...` mapping;
rejected ones get marked `REJECTED` and stay for the record. Append-only,
newest last.

**AMENDMENT 2026-08-07 -- a window MAY now promote a single genuinely-uncovered
entry directly.** The original rule said "NO entry here touches the Bug Bible
directly -- at ship time the operator triggers a BUG FAN-OUT". That was written
when checking coverage meant re-scraping the whole bug history, which was far
too expensive to do per session. **That constraint is gone:**
`otr_coverage_index.yaml` in the survival-guide repo now maps all 369 OTR bug
records through 2026-08-07 to Bible ids, so a window can check ONE new bug
against the index in seconds instead of paying for a full scrape.

So the rule now splits by SIZE, not by authority:

* **A single new entry -- the window promotes it**, at wrap-up, if and only if:
  it clears the admission rule above (verified by a live artifact, not a review
  finding); it is checked against `otr_coverage_index.yaml` AND `BUG_BIBLE.yaml`
  and found genuinely uncovered; and it lands under the **Three-File Contract**
  in ONE commit in the survival-guide repo -- YAML entry + README count +
  executable coverage -- with its `otr_coverage_index.yaml` row appended in the
  same change. Then stamp `- promotion: BUG-...` here.
* **The BULK FAN-OUT over the backlog stays the OPERATOR'S**, unchanged. Batch
  promotion, re-litigating older entries, and any judgment call about whether a
  historical incident deserves a rule are not a window's to make.
* **When in doubt, record the candidate and leave it.** A window that cannot
  cleanly establish "genuinely uncovered" writes the candidate into
  `docs/GO_FORWARD_PLAN.md` for the fan-out rather than guessing. Never
  re-scrape indexed history; only the delta past the index date is ever scraped.

This amendment resolves a real contradiction, not a hypothetical one: the
`otr-handoff` v2 skill instructs a wrapping session to promote a genuinely
uncovered bug, and the pre-amendment contract forbade it, so the 2026-08-07
session recorded its candidate instead of promoting. Under this rule that
session would have promoted.

`promotion` and `status` are deliberately separate axes. `promotion` means a
real production incident has supplied a reusable Bible rule; `status` continues
to describe the implementation's own fix/requalification state. A fixed rule
may therefore be promoted while its same-seed 120-word confirmation remains in
progress. Conversely, an unproven review finding never receives either marker.
Running tests/bug_bible_regression.py after every code change stays automatic
and is unrelated to this log.

Entry format:

```
## PBUG-YYYYMMDD-NN -- short title
- surfaced: <which live run: smoke/soak/episode + date>
- symptom: <one line, what the operator/log saw>
- root cause: <one line>
- fix: <commit sha + one line>
- verify idea: <candidate machine check for the future bible test>
- bible-worthy: <yes/no guess + why -- operator decides at fan-out>
- promotion: <BUG-id after approved fan-out; omit while pending>
- status: OPEN | PROMOTED <id> | REJECTED
```

---

Backfill note: entries PBUG-20260612-01 .. PBUG-20260711-02 were mined 2026-07-10/11
by a two-agent backsweep (git history + handoff/smoke docs), prod-only bar applied,
cross-checked against BUG_BIBLE.yaml (BUG-11.26 family, 12.47, 07.16 excluded as
already promoted). Confidence tags preserved from the sweep.

## PBUG-20260819-01 -- the `normalize_dbfs` widget in the canonical workflow does nothing

- surfaced: re-QA sweep 2026-08-19 (gpt-5.6-sol partial-wiring hunt), then
  LIVE-VERIFIED against the same night's headless run.
- symptom: `OTR_AudioEnhance` exposes a `normalize_dbfs` FLOAT widget --
  `min -12.0, max 0.0, step 0.5`, tooltip *"Peak normalization target dBFS
  (-1.0 = broadcast)"* (`nodes/audio_enhance.py:317-320`). **Moving it changes
  nothing.** `enhance()` accepts the parameter (`:325-326`) and its body never
  uses it: step 7 is a comment reading *"Peak normalize skipped - moved to
  EpisodeAssembler"* (`:404-407`). The real pass is the Assembler's, at a
  HARDCODED `-1.0 dBFS` (`nodes/scene_sequencer.py:171,1362-1364`).
- **live evidence, 2026-08-19 bank-gate run:** the server log carries
  `[OTR_AudioEnhance] Normalize deferred to EpisodeAssembler (post-crossfade)`
  and then `[EpisodeAssembler] Final loudness master: +4.0 dB makeup, -1.0 dBFS
  ceiling (post-crossfade)` -- the widget's value never enters the calculation.
- **WHY IT HID FOR SO LONG:** the canonical workflow's saved value is `-1`
  (`workflows/otr_canonical.json`, node 4 `widgets_values` last slot), which
  happens to EQUAL the Assembler's hardcode. So the control is inert AND
  currently indistinguishable from working. The signature even carries a
  comment rationalising the unused parameter as *"consumed by ComfyUI graph
  runtime not the body"* -- the graph runtime does not normalise audio either.
- consequence: an operator-facing knob in the canonical graph that silently
  ignores every non-default value. Not a story defect; a lying control.
- **NOT YET FIXED, and it is a genuine fork -- do not fix it blind.** Two
  defensible answers: (a) WIRE it through to the Assembler's target, which is
  byte-identical at the canonical `-1.0` and makes the knob honest; or
  (b) REMOVE the widget as a lie. **(b) is dangerous**: `widgets_values` is
  POSITIONAL, and `normalize_dbfs` is the LAST slot -- removing it is the one
  safe deletion position, but it still changes the node contract. (a) touches
  the audio recipe path, and *"the recipes are not on the table"*. This wants
  the operator's call or a panel, not a unilateral edit.
- bible-worthy: probably -- **"an exposed control whose default coincides with
  the hardcode it is ignored in favour of"** is a very portable trap: the one
  configuration where the defect is invisible is the shipped one.
- **ANSWERED 2026-08-19 BY FABLE + SONNET, AND THE ANSWER IS "DO NOT WIRE THIS
  WIDGET AS A PEAK CEILING AT ALL".** The operator asked for exactly this check:
  *"if you can confirm our normalization path is accurate w/ fabel and sonet to
  agree best practice for youtiube i agree"*. Both lanes agreed, independently.
- **MEASURED, not asserted.** The shipped master reads **-9.62 LUFS integrated,
  -1.00 dBTP, LRA 9.90**. Sonnet measured 8 real masters spanning two months:
  mean **-9.87 LUFS, std 0.41 LU** -- tight, but ~4 dB HOTTER than YouTube's
  **-14 LUFS** target. YouTube attenuates louder content and **does not boost
  quieter content**, so the extra loudness is discarded at playback while the
  limiting used to buy it is kept. Paying distortion for nothing.
- **PEAK IS THE WRONG CONTROL.** Two files can share a -1.0 peak and differ by
  10 dB in loudness; a peak knob steers nothing on a LUFS-normalised platform.
  Fable: peak ceiling is a seatbelt, not a steering wheel.
- **THE DRIVER''S OWN A/B WAS NOT EQUIVALENT TO WIRING THE KNOB, and this is the
  correction that matters.** The four arms were built by applying FLAT GAIN to a
  finished master, so they measured -9.44 / -11.44 / -14.44 / -17.43 LUFS.
  Sonnet ran the REAL `_master_loudness` verbatim instead and found the
  relationship is sharply NONLINEAR: `ceiling_dbfs=-1.0` yields **-13.26 LUFS**
  but `ceiling_dbfs=-9.0` yields **-23.58 LUFS**. An 8 dB ceiling move produces
  a **10.3 dB** loudness move, because the tanh stage renormalises to the
  ceiling BEFORE saturating -- at -9.0 the limiter barely engages, so the makeup
  gain stops doing its work. **So the operator''s picked arm would have shipped
  ~-23.6 LUFS, roughly 10 dB under target, not the -17.4 he actually heard.**
  A blind fader test on this algorithm cannot predict its own result.
- **WHAT BOTH LANES SAY TO SHIP:** integrated **-14.0 LUFS** as the real
  control, **-1.0 dBTP** true-peak as a fixed safety rail (not a creative knob),
  **dynamics untouched** (LRA 9.9 is right for radio drama and is the healthy
  part). Retire the fixed +4 dB makeup as the loudness engine; loudness comes
  from measured clean gain instead.
- **HOW THE OPERATOR''S EAR IS STILL HONOURED:** -14 LUFS is already ~4 dB
  quieter than today AND less limited, which is most of what he reached for; the
  remaining few dB belong on his monitor volume, which costs the audience
  nothing. Note the accident worth keeping: his arm C (-6.0 peak, flat-gain
  equivalent) measured **-14.44 LUFS** -- essentially YouTube-correct. He heard
  the right level; it was one click above his pick.
- **IMPLEMENTATION NOTE FOR WHOEVER BUILDS IT:** the change belongs in
  `_master_loudness` (`scene_sequencer.py:171`) with the call site at `:1362`
  passing `ceiling_dbfs` explicitly instead of relying on the silent default,
  and the widget must land on `EpisodeAssembler.INPUT_TYPES` APPENDED to node
  7''s `widgets_values` (positional). **`tests/test_audio_byte_identical.py`
  WILL go red** -- it is a golden-hash gate and any master-algorithm change
  moves every byte; it needs a deliberate `--capture-baseline` re-run, which is
  expected rather than a surprise.
- **BUILT 2026-08-19 on the operator''s go** (*"yes lets buidl taht nomraled best
  practcei lufts for yoiutube"*). `_master_loudness` now MEASURES integrated
  LUFS (pyloudnorm), applies ONE linear gain to `-14.0` LUFS, then applies the
  `-1.0` dBFS peak as a safety rail only. The call site passes `ceiling_dbfs`
  AND `sample_rate` explicitly -- the old one passed neither, which is how the
  ceiling became invisible. Target overridable via `OTR_MASTER_TARGET_LUFS`.
  The legacy tanh path survives ONLY as a fallback for audio too short to
  measure (< 400 ms) or a missing pyloudnorm, so a render never fails for want
  of a reading.
- **PROVEN ON FIVE REAL MASTERS, not on synthetic tone.** Feeding the shipped
  masters through the new function: `-9.63 -> -14.00`, `-9.69 -> -14.00`,
  `-10.56 -> -14.00`, `-10.60 -> -14.00`, `-9.52 -> -14.00`. Every one lands on
  target; output peaks sit at `-4.4` to `-5.5` dBFS, so the safety rail never
  fires -- exactly as predicted.
- **THE OPERATOR''S OWN QUESTION, ANSWERED AND PINNED.** He asked whether
  normalisation should be per-clip or at the end: *"i tink we nee dtsoi start
  wthj clip level rights o teh clisp pabalcne out"*. He is right and it already
  exists -- `_level_dialogue_clip` levels every spoken line to `-16` dBFS
  active RMS on BOTH the announcer and character buses (`scene_sequencer.py`
  `:967`, `:974`); music passes through untouched by design. This stage runs
  after it and applies a SINGLE LINEAR GAIN, so it mathematically cannot change
  any clip-to-clip ratio. Measured on a real master: the ratio between two
  segments was `1.033572` before and `1.033572` after. Pinned by
  `test_a_single_linear_gain_preserves_the_clip_balance`.
- **THE GOLDEN-HASH GATE DID NOT NEED A RE-BASELINE.** Sonnet flagged
  `tests/test_audio_byte_identical.py` as a likely casualty; in practice its
  byte-identity test SKIPS (it needs a live candidate render) and its seven
  fixture/contract tests pass unchanged. Recorded because the warning was
  reasonable and the reality is cheaper -- but a future live-render comparison
  WILL need `--capture-baseline`.
- **WHAT IT STILL OWES: a live leg.** Unit-proven and proven against real
  masters, but not yet through a full canonical render published to `otr/obs/`.
  Deliberately not run: the operator has LTX 2.5 in the recipe lab and the GPU
  is his. The crest-factor improvement in particular CANNOT be seen in the
  offline proof -- those masters were already squashed by the old +4 dB tanh,
  and only a live render exercises the new path with unsquashed input.
- status: **BUILT AND UNIT-PROVEN 2026-08-19; OWES A LIVE LEG.**

## PBUG-20260819-02 -- `audio_revision` is dead at BOTH ends, so ShotLock cannot detect a stale audio binding

- surfaced: same re-QA sweep, then verified against 12 live published ledgers.
- symptom: `nodes/otr_shot_lock.py:1909` stamps
  `"locked_against_audio_rev": str(meta.get("audio_revision") or "")` into the
  video plan. **Nothing in production ever writes `meta.audio_revision`** --
  the only occurrence outside that read is a fabricated test fixture
  (`tests/test_route_freeze_wiring.py:214`). And the field it produces,
  `locked_against_audio_rev` (`nodes/_otr_video_engines/schemas.py:165,400`),
  has **zero production readers**. Dead producer, dead consumer, live stamp.
- **live evidence:** 12 of 12 recent published `signal_lost_*` ledgers have
  `meta.audio_revision` ABSENT and `locked_against_audio_rev` empty. It is not
  intermittent -- it is every episode, always.
- consequence: the mechanism that is supposed to catch a video plan bound to
  STALE audio cannot fire. A re-rendered audio track under a frozen shot plan
  would go undetected by the one field named for detecting it. No live
  mis-render is attributed to this yet, which is exactly what a dormant guard
  looks like until the day it is needed.
- **fix is a real decision, not a one-liner:** either give `audio_revision` a
  producer (who owns it -- the Assembler, at master-WAV write, keyed on the
  master sha256 that already exists in the log?) and give
  `locked_against_audio_rev` a reader that actually compares, or delete both
  and stop advertising a protection that does not exist. **Per the ledger rule
  in `CLAUDE.md`, deleting is only allowed once every field has an owner.**
  Half-wiring it again would be the third instance of this bug's own class.
- bible-worthy: yes -- **"a guard field stamped from a key nothing writes"**;
  the stamp makes the ledger LOOK protected.
- status: **OPEN -- diagnosed, live-verified, fix not chosen.**

## PBUG-20260818-01 -- the news close leaked into the fiction, and the fiction leaked into the news close

- surfaced: 2026-08-18, from a published episode the operator watched --
  `signal_lost_the_searing_relay_20260818_094723`. Live artifact, not a review.
  Operator, verbatim: "i thin it went way oevrboard in anaoucning teh coda teh
  coda shoudl eb abreif sumamry opf tyeh enws not teh whoel new story."
- **FOUR ITERATIONS, EACH ONE A REAL LIVE RENDER, AND THE RECORD KEEPS ALL
  FOUR RATHER THAN ONLY THE LAST.** Two of them disproved the diagnosis or
  fix that preceded them; the log says so plainly rather than presenting a
  clean, retroactively-tidied story.
  1. **Symptom.** The closing segment ran 114 words / seven sentences --
     dense, citing an advocacy group and a named 2024 law.
  2. **First fix (wrong target).** `scifi_news_pro_script_system`'s
     `CODA: <spoken coda>` line was given a brevity cap, on the theory that
     the long block WAS the CODA. Disproven by re-render within the hour:
     `parsed.coda` was already short (12-19 words) before any fix. The long
     block is a SEPARATE, code-appended row -- `treatment.news_close_read`.
     The CODA fix is harmless and stays; it did not address the complaint.
  3. **Second fix (right target, exposed a worse problem).** Added the same
     brevity clause to `scifi_news_pro_news_read_system`
     (`news_close_read`'s own seam). Re-render (`signal_lost_the_
     recession_of_room_4_20260818_112039`) showed the read was now short
     (149 -> still long that run, but the NEXT verification render's read
     dropped to 45-67 words) -- but exposed FACT/FICTION OSCILLATION: the
     ANNOUNCER outro had already drifted into real astronomy facts before the
     CODA cut back to pure fiction, then the news read hit facts a third
     time. Operator: "if you start with news you already forgot about the
     story... just a bit of story end, then news."
  4. **Third fix (sequencing).** Instructed outro+CODA to stay strictly
     in-fiction -- no real-world statistic, date, institution, or named real
     person -- with the factual read appended separately, "the story closes
     before the facts do, once, not back and forth." Also added attribution +
     closing-thought shape to `news_close_read` (name who found it, end on
     one reflective thought; same shape applied to `media_archive`'s
     `news_close_brief` per operator: "for media archive and scifi news it
     should summarize the source... researcher discovered... and leave with
     a thought at the end").
  5. **ROOT CAUSE, FOUND BY A KIBITZ REVIEW (antigravity, r3, scoped single
     round -- CONFIRMED against the real files before landing), NOT BY ME.**
     A re-render under fix #4 alone (`signal_lost_the_borrowed_voice_
     20260818_115506`) still leaked: the CODA read "UCLA researchers are
     currently using generative AI to design proteins never before seen in
     nature for drug testing" -- nearly VERBATIM the news read's own opening
     clause. Antigravity traced why: `run_scifi_news_pro_episode` runs
     `_pass_news_read` BEFORE `_pass_script` and copies the result onto
     `treatment.news_close_read` (`_otr_scifi_news_pro.py:3616-3618`), and
     `_script_user_prompt` dumps the WHOLE treatment -- fact included -- into
     the script writer's own context (line ~1821), while the system prompt
     simultaneously told it never to state that fact. Telling a small local
     model (`google/gemma-4-12b-it`) not to repeat a fact sitting in its own
     context is a much weaker ask than not showing it the fact at all.
  6. **Final fix.** `_script_user_prompt` now excludes `news_close_read` from
     the treatment dump (`treatment.model_dump(exclude={"news_close_read"})`)
     -- the field stays live for every OTHER consumer (ledger assembly,
     receipts), only this one prompt-building call blinds itself to it. Added
     a format-reminder restating the in-fiction constraint at point of use.
- **LIVE-VERIFIED CLEAN on the next render**
  (`signal_lost_the_last_reading_20260818_122159`, published to `otr/obs/`).
  Outro: "The graph reached its zenith in the silence of the observation
  suite." CODA: "The data was perfect. The man was gone." -- both pure
  fiction, zero leaked facts. News close, 67 words / 3 sentences: names UCLA
  and the journal PLOS One, states the finding, closes on a contrast rather
  than a fourth stacked statistic. Exactly the shape requested: story ends,
  then one clean pivot to news.
- **NOT re-verified live:** the `media_archive` half of fix #4
  (`nodes/_otr_media_archive_interpreter.py`, `news_close_brief`'s schema
  description). Unit-tested and schema-verified; no live `media_archive`
  render this session.
- **A SEPARATE, STILL-OPEN DEFECT, NOT FOLDED IN.** One render this session
  (before the final fix) hit `NewsProTreatmentError`: the news-read pass
  invented fictional character names ("Dr. Sharon Hame", "Laura Goodkind")
  inside what must be a pure factual report, and the 2-attempt retry ladder
  didn't recover. The final successful render did NOT reproduce this, but one
  clean run is not proof either way for a stochastic local-model failure.
  Tracked separately (task chip), not claimed fixed here.
- gates: pack suite 84/84 -> 104/104 (added `media_archive` interpreter tests)
  -> 108/108 (added 4 new regression tests proving the context exclusion).
  Full suite 11024 -> 11028 passed (the 4 new tests), zero regressions, run
  twice across the session. `tests/test_scifi_news_pro_script_prompt_
  excludes_the_read.py` is the new file: proves a populated `news_close_read`
  never appears in the script writer's own prompt, proves every other
  Treatment field still does, proves the empty-read case is unaffected.
- Bible: not promoted. The generalisable shape -- "a downstream generation
  step's finished OUTPUT gets fed back as CONTEXT to an earlier-labelled step
  that was explicitly told not to produce that content" -- is a real,
  reusable defect class (prompt contamination via pipeline ordering) worth a
  future delta-scrape entry, but not yet checked against the coverage index
  for whether it is genuinely uncovered.
- status: FIXED and live-verified on `scifi_news_pro`. The `media_archive`
  half is FIXED but not live-verified. The character-hallucination finding is
  a SEPARATE open item.

## PBUG-20260817-09 -- a character's VOICE DRIFTS BETWEEN HIS OWN LINES: the synthesis seed is re-rolled per line
- surfaced: operator listening to `signal_lost_mongooses_stand_20260817_234050` in `otr/obs/`, 2026-08-17. In his words: *"Nag's first voice was [fine], the second Nag drifted to a voice that I often hear but doesn't line up with Nag's first speech."* Live published artifact -> admission rule satisfied. **Found by ear before it was found in data.**
- symptom: within ONE episode, the same character sounds like two different people across his two speeches. Not intermittent -- it is the normal behaviour of every multi-line character.
- root cause: **THE ASSIGNMENT IS STABLE; THE SYNTHESIS IS NOT.** Per-line telemetry for NAG (`c03`) in that episode: `b003 ref=vz_donor_glenn alpha=1.0 delivery=v2:nonzero(derived) seed=532084266468738542` and `b005 ref=vz_donor_glenn alpha=1.0 delivery=v2:nonzero(derived) seed=5038394939402288039`. **Same voice reference, DIFFERENT GENERATION SEED, and a fresh per-line delivery vector applied at alpha=1.0 (full emotion-blend strength).** IndexTTS2 therefore re-synthesises the timbre from scratch for every line, so voice identity is not conserved across a character's own dialogue. Confirmed stable in the OTHER direction too: `voice_ref_id` is identical on both lines and across all sampled episodes, so this is NOT the wrong-voice-assignment class (PBUG-20260817-08) and NOT a starved pool.
- **why it reads as "a voice I often hear":** the drift lands near the handful of references that dominate the corpus. Measured over the last 40 episodes: 27 distinct voices across 118 cast rows, with the top five taking 55% (`vz_bill_boerst` 16.1%, `bm_george` 11.0%, `idx_lemmy_algenib_cockney_v1` 10.2%, `vz_caro_davy` 9.3%, `bm_fable` 8.5%). The small pool is a SEPARATE contributing condition, not this root cause.
- fix: NONE YET. Two candidate levers, both visible in the telemetry: (1) fix the synthesis seed PER CHARACTER at cast time (it is already recorded in `meta.voice_cast_decision`) instead of re-rolling per line; (2) reconsider `alpha=1.0`, which gives the per-line delivery vector maximum authority to reshape timbre. **This is a design fork with more than one defensible answer -- emotional range across lines is a FEATURE, and pinning the seed may flatten it -- so it takes a panel before code, per the standing rule.**
- verify idea: render one episode twice, once with a per-character fixed seed, and compare a speaker-similarity measure between a character's own lines. The operator's ear is the acceptance gate; the metric only ranks candidates.
- **THE MISSING-FILE QUESTION IS ANSWERED AND WAS A DRIVER ERROR (2026-08-18, agy lane, verified by the driver).** The driver claimed "162 of 206 bank refs point at absent files" after checking `ComfyUI/models/TTS`. **That is the wrong root.** Refs resolve through `_otr_audio_engines/base.resolve_voice_ref_path` to the MIGRATED models root `C:/ComfyUI-Models/TTS`, where every local reference exists. Proven: `vz_donor_glenn.wav` is on disk there (882 KB) and its SHA256 `c8679a09eff572aa9b564181b18e87525f5b18f290342032b014d30176ae946c` matches the bank's own `ref_sha256` byte for byte. Chain: `OTRVoiceNodeBase._render_per_line` -> `_resolve_clone_ref_path` -> `_resolve_ref_to_disk` -> `resolve_voice_ref_path` -> `IndexTTS2Engine.generate_voice` -> `_resolve_ref` -> `scripts/_otr_indextts2_worker.py` -> `tts.infer(spk_audio_prompt=...)`, and the worker `os.path.exists()`-checks before inference. **So nothing falls back, the reference IS applied, and the fail-closed contract was never violated.** Retracted with it: the claim that ELLIE PALMER's voice was undefined behaviour from a missing file -- she was correctly rendered on Glenn, who is simply a high-pitched male (operator, by ear).
- **THE ROOT IS THEREFORE CONFIRMED AS THE SEED, NOT THE REFERENCE.** `_otr_resolved_request` reduces `stable_line_seed` per LINE, and `_otr_voice_node_common` derives the engine seed from it -- so a zero-shot model re-samples its latent speaker identity on every line while conditioned on the same reference. Candidate fix from the agy lane, sound in shape: derive the engine seed from CHARACTER identity (`char_id` + `episode_seed`) instead of the line, locking identity across a character's dialogue while the text still supplies prosodic variation, and fixing all cloning engines at once with no worker changes.
- **STILL NEEDS A PANEL BEFORE CODE:** (1) it is a genuine fork -- pinning the seed may FLATTEN the per-line emotional range `alpha=1.0` currently buys; (2) it breaks byte-for-byte reproducibility of shipped episodes, so it must ride an `engine_impl_version` / profile bump rather than silently changing replay.
- **(superseded) DO NOT confuse with the missing-file question.** `vz_donor_glenn.wav` is NOT present under `models/TTS/refs/indextts2/` (162 of 206 bank refs point at absent files), yet IndexTTS2 -- which is explicitly fail-closed, `missing_ref_fallback = None` -- rendered audio for it anyway. **Unexplained, and it must be resolved before this fix is designed:** something maps that id to a real file, and until it is found we cannot say whether a drifted line fell back to a default voice or merely re-rolled the same one. That resolver hunt is the first task, not the fix.
- bible-worthy: LIKELY -- "identity conserved in the assignment but re-rolled in the synthesis" is a reusable class. Do not promote before a fix.
- promotion: PENDING
- status: OPEN

## PBUG-20260817-08 -- the LEMMY CAMEO VOICE is assigned to an ordinary character with no Lemmy in the cast
- surfaced: live bank-gate leg 2026-08-17 23:30, `signal_lost_rivers_embrace_20260817_233013` (bank `original`), seen by the operator on the treatment card and confirmed at the ledger.
- symptom: `ED HIBBERT` is voiced by `idx_lemmy_algenib_cockney_v1` -- the Lemmy cameo reference -- while the cast roster is exactly `ANNOUNCER / ERIN BURNS / ED HIBBERT` and contains no Lemmy. The character speaks in the cameo's cockney voice for the whole episode.
- **OPERATOR RULING 2026-08-17: this IS a bug.** In his words: *"only Lemmy should get Lemmy voice"*. (In the same message he ruled that the ANNOUNCER's male/female voice split is NOT a bug -- it is a deliberate 50-50 and the announcer shows no portrait -- so do not file that.)
- **root cause: FOUND AND FIXED. The cameo reference sat in the ordinary `char_voice` pool like any other row, so the seeded draw could hand it to anybody.** Fixed in `8f3c7615` (2026-08-17 07:21): `reserved_voice_ref_ids()` derives the reserved set FROM THE POLICY -- every `local_wav` Lemmy route on any engine -- and `assign_voice_for_slot` drops those ids from the candidate pool. Derived, never a hand-kept list, so adding a route reserves it for free. Only clones of his OWN recording are reserved; the catalogue voices he merely borrows (`bm_george`, `el_daniel`, `gt_algenib`) stay castable, because pulling the preferred announcer out of the pool to protect a cameo would trade one defect for another. The policy path is untouched, so reserving cannot starve Lemmy of his own voice.
- **verified 2026-08-18, three ways.** (1) CORPUS: across the 15 most recent episodes with a cast ledger, exactly two rows put a Lemmy-owned ref on a non-Lemmy character -- `kinetic_motion_clause_live_test` (05:01) and `rivers_embrace` (23:30), **both 2026-08-17**. Every episode from 2026-08-18 onward is clean. (2) SELECTOR: `tests/test_lemmy_voice_stays_reserved.py` sweeps 480 seeded draws across both genders and no reserved reference ever reaches a character slot; it also proves the reserved row IS in the unfiltered pool, so the test cannot pass for the wrong reason. (3) OPERATOR, unprompted 2026-08-18: *"i do feel we have been seeing the right amount of lemmy to be honest"*.
- ~~**WHY THE LEAK WAS SEEN 16 HOURS AFTER THE FIX, and it is not a code defect.**~~ **THIS PARAGRAPH WAS WRONG AND IS STRUCK 2026-08-18 NIGHT.** It read: the soak harness never tears its server down, so an evening leg was still executing the module Python loaded that morning, and *"a stale resident process explains a post-fix sighting without the fix being wrong"*. The resident-server trap is real and documented (CLAUDE.md section 5) -- **it simply is not what produced this row.**
- **RE-OPENED AND COMPLETED 2026-08-18 NIGHT: THE 08-17 FIX COVERED ~4% OF PRODUCTION CASTING.** `reserved_voice_ref_ids()` was applied in exactly one place, `assign_voice_for_slot`. But `hybrid_voice_fit_enabled()` (`_otr_casting.py:896`) is **default-ON**, so the live path is: the LLM is shown `build_voice_cards()`, proposes one id, `validate_voice_proposal()` checks it, and `cast_lock.py:884-906` stamps the accepted id and `continue`s -- **never reaching the selector at all**. Measured across 1711 ledgers: **1871 rows stamped from an accepted proposal against 82 fallbacks**. Neither the card builder nor the validator knew reserved ids existed.
- **AND THE ORDERING MADE IT WORST-CASE.** Cards are sorted alphabetically by `voice_ref_id` and capped at 12, so `idx_lemmy_algenib_cockney_v1` sorted FIRST among indextts2 male entries and was handed to the model as **CARD #1 on every male slot**, the position a model favours most.
- **THE CORPUS NUMBERS, STATED SO THEY RECONCILE** (a QA pass caught the first draft implying 20 + 5 = 21). Three different populations: the LLM proposed a reserved id **21 times and it was accepted 21 times**; **18** of those accepted proposals became a final cast row. Separately, **25 cast rows in total carry a reserved id** = **5 legitimate LEMMY rows** + **20 leaks** onto DON PEDRO, MARCELLUS, BANQUO, FLETCHER CORBEN, STARBUCK, FERDINAND, MOE GORDON, Dr. Alexei Petrov and others. Of the 20 leaks, **18 came through the hybrid path and 2 did not** -- and that 2 is not noise, it is the fingerprint of the THIRD pool below.
- **ALL 5 LEGITIMATE LEMMY ROWS CAME THROUGH THE POLICY PATH, ZERO THROUGH THE HYBRID PATH** (measured per-row against `meta.voice_cast_decision`). This was the QA pass's open question -- whether reserving would cost Lemmy rows that only ever got his voice by being unfiltered card #1 -- and the answer is no. The fix cannot take his own voice away from him.
- **THE THREE 08-18 VERIFICATIONS WERE ALL SOUND AND ALL BLIND TO THIS.** The 480 seeded draws went through `assign_voice_for_slot` -- the 4% path. The corpus check was a true observation of a **1.1% leak rate (21/1877) not firing in a small window**, not proof it could not. The operator's *"right amount of lemmy"* is consistent with both. Three green checks on the wrong path still read as green.
- **A PANEL HAD ALREADY NAMED THIS LAYER ON 2026-08-04 AND IT WAS NOT ACTED ON.** `kibitz-runs/2026-08-04-continuity-ultracode/input_voice-variety.json:174`: *"the hybrid LLM voice-fit ... sits IN FRONT of the deterministic caster ... whose 12-card truncation is a harder variety cap than the tier-of-one ... the plan's 200-episode simulation measures a path production can bypass entirely."* It asked for a test running with the hybrid path ENABLED. That test did not exist until now.
- **THE THIRD POOL, FOUND BY THE SONNET QA PASS ON THE FIX ABOVE -- AND IT WAS A BLOCKER.** `gender_agnostic_fallback_ref` (`_otr_voice_bank.py`) had **three** production call sites (`cast_lock.py:938`, `_otr_voice_node_common.py:172` and `:243`) drawing uniformly over the engine's refs with **no reserved filter, and no reject-tier filter either**. It is not a rare branch: canonical gender `other` is **20% of every roll** and the bank carries zero rows for it, so `assign_voice_for_slot` raises and every one of those rows lands here. QA measured Lemmy's clone coming back in **7-9 of 200 draws per engine** -- roughly the odds of any other single voice, because nothing excluded it. That is what the 2 non-hybrid leaks above were. **It was missed twice because it is not a "caster" by name**, yet it draws the reference that both the ledger stamp and the render path use. Now filtered, measured **0/200** after, with 40 distinct voices still reachable.
- **AND THE HELPER ITSELF WAS HARDENED (QA finding).** `reserved_voice_ref_ids()` caught only `ImportError`, but it now has four callers and two of them promise in their own docstrings to be pure and never raise. A malformed `LEMMY_VOICE_POLICY` (truthy non-dict) would have raised `AttributeError` straight through those promises and out of `cast_lock.py`, turning a cosmetic config error into a dead render. The walk is split into `_reserved_ids_from_policy` and the wrapper now fails soft on any exception -- reserving nothing, which is exactly the pre-reservation behaviour, rather than failing an episode.
- **FINAL FIX (2026-08-18 night).** `build_voice_cards` no longer offers a reserved id, and `validate_voice_proposal` refuses one even if proposed off-list (a proposal is free text and need not name a shown card; the validator is the last gate before CastLock stamps). Both in `nodes/_otr_voice_bank.py`. The policy path is untouched, so Lemmy's own qualified route still stamps his voice directly. New `tests/test_lemmy_reserved_on_hybrid_path.py` (10 tests) covers both layers with teeth-checks: the card list must not collapse to empty, an ordinary proposal must still validate, and the reserved ids must remain present in the unfiltered bank.
- **A PRE-EXISTING TEST HAD THE DEFECT ENCODED AS AN EXPECTATION.** `tests/test_hybrid_voice_fit.py::_real_male_indextts2_id` returned the alphabetically-first male indextts2 id -- which IS `idx_lemmy_algenib_cockney_v1` -- so `test_auto_registry_honours_accepted_proposal` asserted that CastLock stamps Lemmy's Cockney onto a character named BOB, and passed. The fix turned it red, which is the fix working. Helper corrected to exclude reserved ids.
- bible-worthy: **YES, and `12.114` already carries the reusable half** -- *"a reservation existing as a CONVENTION in one subsystem is invisible to another enumerating the same catalogue"* describes `build_voice_cards` exactly. What this pass adds is a sharper companion: **a stale process and an uncovered code path are indistinguishable from the outside, and a green test on the minority path proves nothing about the majority path.** Candidate for a follow-on entry or an amendment to `12.114`'s verify steps.
- promotion: `12.114` PROMOTED 2026-08-18 (survival-guide `b9aada7e`). An amendment covering the coverage-gap half is PENDING.
- status: **RE-OPENED AND RE-FIXED 2026-08-18 NIGHT -- ALL THREE POOLS.** Deterministic selector fixed `8f3c7615`; hybrid card list + proposal validator fixed this session; gender-agnostic fallback fixed this session after QA. 18 tests across all three, with teeth (empty-pool guards, an ordinary-proposal guard, and a check that the reserved rows remain in the unfiltered bank so the guard cannot pass by deleting Lemmy). **Unit-proven, not live-proven, and that is the honest gate here:** the corpus leak rate is ~1%, so a clean single leg would prove nothing either way.
- **THE LESSON, AND IT COST THREE DISCOVERIES TO LEARN.** The same reservation had to be taught to three different pools, found one at a time, each by a different method: the first by an operator sighting, the second by measuring which path production actually takes, the third by an adversarial QA pass. **A guard that lives in one subsystem is invisible to every other subsystem enumerating the same catalogue** -- Bible `12.114`'s existing wording -- and the practical counter is to ask "what are ALL the places that can draw one of these?" rather than fixing the place the bug was seen. `test_reserved_ids_are_unreachable_from_every_pool_at_once` exists to be EXTENDED if a fourth pool appears, rather than a fourth file being written.

## PBUG-20260817-07 -- parenthetical STAGE DIRECTIONS ride inside spoken text, into the captions and the voice actor's mouth
- surfaced: live bank-gate leg 2026-08-17 23:30, episode `signal_lost_rivers_embrace_20260817_233013` (bank `original`), watched by the operator in `otr/obs/`. Published artifact -> admission rule satisfied.
- symptom: two spoken rows carry a bracketed direction in the TEXT field, so it is both captioned on screen and read aloud: b002 `ERIN BURNS: (fumbling with the controls) I can't... it's stuck, Ed!` and b003 `ED HIBBERT: (tightening his grip on Erin's wrist) Let go, Erin!`
- **CORRECTED 2026-08-17 (second pass): THE LLM CLEANUP PASS EXISTS, RUNS, AND MOSTLY WORKS. This is a HIT-RATE residue, not a missing feature.** The operator's stated goal -- *"the goal was to clean up non-dialogue from the ledger using an LLM"* -- IS implemented in `_otr_ledger_clean` as a judge + repair pair (`ledger_clean_line_judge` / `ledger_clean_line_repair`). Measured on the same-night public_domain leg: *"6 voiced row(s): 4 carried something that is not speech ... 3 repaired, 0 improved, 1 still unclean, 0 with no model, in 15 model call(s)"*, with real repairs shipping (`"Here, Nag! Between the jungle and the bungalow, I stand!"` -> `"Here, Nag! I stand between you and my family!"`) and the pass's own note that *"4 row(s) were caught ONLY by the judge -- no pattern would have found them."* A row the judge cannot repair is FLAGGED `unclean_spoken_text` and SHIPS, because the pass must degrade rather than fail an episode (THE LAW). The two parentheticals the operator boxed are that residue.
- root cause (secondary, and why the residue is only ever SEEN, never HEARD): ONE FIELD, TWO CONSUMERS. Both rows carry `compose_flags: ['unclean_spoken_text']`, so the clean stage DETECTED the parenthetical and stored the text unchanged: that is THE LAW working (an audit flags, it never fails an episode) plus the 2026-08-05 ruling that `clean_spoken_text` does not strip. The ledger has exactly ONE `text` field and two readers treat it differently -- the TTS path strips the parenthetical before synthesis (which is why the AUDIO is clean and the operator heard nothing wrong), while the caption burner renders the field VERBATIM. So the defect is not "stage directions leak"; it is that the spoken-vs-displayed distinction has no field. **Fourth sighting of the one-field-two-meanings class in a single day**, after `work_title` (work vs publication), the soak receipt's `title` (run label vs episode title), and `title_source` (stored authority vs `video_engine`'s log-only vocabulary). The operator's own framing is the contract being violated: *"I thought we made the ledger dialogue only"*.
- **OPERATOR RULING 2026-08-17: NOT BEING CHASED.** In his words: *"some stage direction did make it into the captions but I'm not gonna chase that"*, in the same breath as calling the episode `perfect`. Logged so it is not rediscovered, NOT queued.
- **AND DO NOT "FIX" `clean_spoken_text` FOR THIS.** That function is the obvious lever and it is under a standing operator ruling from 2026-08-05 forbidding exactly that. Any future attempt must find another surface or get a fresh ruling.
- **MECHANISM, TRACED 2026-08-17 (operator's question: how did it reach the ledger but not the TTS?).** The strip is at the ENGINE boundary, not the ledger boundary. `_otr_script_prep.clean_spoken_text` removes parentheticals (`_PAREN`), and every audio engine calls it from its `prepare_text` hook -- `_otr_audio_engines/eng_indextts2.py`, `eng_chatterbox.py`, `eng_dia.py` -- immediately before the forward. So TTS speaks the cleaned string while the ledger keeps the authored one, and the two DISPLAY consumers (`otr_caption_burn` and the treatment/transcript writer) read `lines[].text` verbatim. Confirmed on screen by the operator: the same parenthetical appears in BOTH the burned caption and the treatment card's CLASSIFIED TRANSCRIPT. Not luck -- the engines were taught to clean and the display surfaces never were.
- **THE FIX, IF EVER WANTED, DOES NOT TOUCH THE FORBIDDEN FUNCTION.** The 2026-08-05 ruling forbids CHANGING `clean_spoken_text`; it does not forbid CALLING it. Having the caption burner and the treatment writer apply the same helper the engines already apply closes both surfaces with no audio risk. Two call sites, not a rewrite.
- verify idea: the flag ALREADY EXISTS -- count rows carrying `unclean_spoken_text` in `compose_flags` across the corpus; no new detector is needed. If it is ever fixed, the shape is a SEPARATE display field (or the caption burner consuming the same strip the TTS path already applies), never a change to `clean_spoken_text`.
- bible-worthy: NOT YET -- undiagnosed and explicitly unchased.
- promotion: NONE
- status: OPEN (WILL-NOT-FIX by operator ruling 2026-08-17)

## PBUG-20260817-06 -- Doyle's "Watson" and "Mr. Holmes" spoken BY NAME in a Leacock parody's dialogue, on the invent-nothing lane
- surfaced: r1 panel on PBUG-20260817-04 (Opus seat), 2026-08-17; verified by the driver reading the shipped ledger `signal_lost_the_blackwood_enigma_20260817_172553_ledger.json`, not inferred. Live published artifact -> admission rule satisfied.
- symptom: the locked cast rows are `THE GREAT DETECTIVE` (c02) and `SECRETARY` (c03), and `meta._adaptation_character_names` carries Leacock's own roster for *"Maddened by Mystery, or The Defective Detective"* -- a Sherlock Holmes PARODY whose joke is that it never names Doyle's characters. Lines b002-b005 nonetheless address **"Watson"** and **"Mr. Holmes"** by name -- *"I must insist upon immediate access to that file, Watson"*, *"Very well, Mr. Holmes."* -- in shipped, published audio. The model recognized the parody and reached for the canonical names the source deliberately avoids: a fidelity inversion on the fidelity lane.
- root cause: NOT diagnosed to a producer yet. Known at the files: the dialogue passes carry the adaptation roster, and the markup acceptance gate (`UNKNOWN_SPEAKER`) checks speaker KEYS only -- a name spoken INSIDE a line's text is invisible to it. So nothing on the acceptance path can see this class. Which pass introduced the names (line composer vs exchange vs a repair) is r2-of-its-own-item work.
- relationship to PBUG-20260817-04: upstream of it and likely feeding it -- `_otr_story_brief._build_produced_open_input` distills these same scene-1 rows into the announcer's brief, so the announcer's invented Holmes-pastiche title (*"The Adventure of the Purloined Paper"*) is downstream-CONSISTENT with dialogue that had already gone Holmes. Fixing 04's announcer surface does not fix this.
- fix: NONE YET. Its own queue item; do not fold into 04's splice.
- verify idea: against the adaptation lanes only -- flag any spoken line containing a proper name that is neither a cast row's name, in `_adaptation_character_names`, nor in the unit's own text. The last clause is the hard part (requires the source excerpt) and is what separates "Watson" (foreign) from a legitimately-in-source name.
- bible-worthy: PLAUSIBLE -- "the model substitutes the FAMOUS original for the parody/derivative it was told to adapt" is a reusable class distinct from 11.61 (assigned-record leak) and 12.103 (instruction ignored). Do not promote until a fix exists to claim.
- promotion: PENDING
- status: OPEN

## PBUG-20260817-05 -- harness run labels become the EPISODE TITLE, and those runs publish to otr/obs/
- surfaced: operator observation 2026-08-17 -- *"yesterday your titles had the word sweep and such in it, weird"* -- then measured on disk by the driver over the published folder and all 1,623 titled ledgers.
- symptom: episodes titled `BANKSWEEP media_archive`, `Qual Leg 1 Shakespeare Nemo`, `PROBE A_30w_noforce`, `Codex TTS Cast Credits Smoke`, `lemmy provisional tier kokoro acceptance`. **And they are PUBLISHED: 16 of the 65 finished `.mp4`s in `otr/obs/` are harness runs, not episodes** (banksweep x5, soak02-06, acceptance, chunkb, probe...). So roughly a quarter of the deliverable folder is diagnostics wearing an episode's clothes.
- root cause: NOT the writer inventing words. `meta.title_source` on every one of these reads **`user`** -- the harness passes its own RUN LABEL into the episode-title field when it launches a leg, and that leg then publishes through the normal path like any other render. The genuinely generated titles are clean: the only `llm_post_composition` titles my sweep flagged were substring false positives (`Cover Spectrum` matching "cov", `Locked Legacy` matching "leg", `Past the Gate`). **The story writer is not the defect; our own harnesses are.**
- **OPERATOR RULING SAME DAY, AND IT CORRECTS THE DRIVER'S FIRST READ:** publishing to `otr/obs/` is **NOT** the defect -- it is the SUCCESS SIGNAL. His words: *"always publish to obs -- a test is not complete unless published to obs (or it's just testing one part). If I see it in obs then it's somewhat a success"*, plus *"if I don't see it in obs and it took more than 5 minutes it's a fail."* **The driver moved 17 of these artifacts into an `otr/obs/_diagnostics/` subfolder reading them as pollution, and that removed his proof that the full path ran. Restored within minutes; nothing was deleted.** The bench carve-out is NOT a precedent here -- it is scoped to isolated stock-node graphs that never touch the canonical workflow, whereas a soak or bank-gate leg runs the real path end to end, which is exactly why its publication means something.
- so the ACTUAL defect is narrow and cosmetic: the harness RUN LABEL becomes the on-screen TITLE CARD. Priority in his words: *"long term yeah I want the title right, but for my dailies it keeps me going to see episodes."*
- fix: SHIPPED `e21b27ba` (2026-08-17). At the source, per the queue's decided shape: `scripts/otr_gpu_soak_matrix.py` `leg()` no longer passes `--title` at all -- the canonical workflow authors the title (accepted cost: soak legs now run `_generate_title_from_script`). Receipt key `title` -> `leg_label`, and the receipt records the ledger's real `episode_title` + `title_source` read back from the ledger the run wrote. Guard: a headless run reporting `title_source == "user"` is flagged `VIOLATION_headless_title_source_user` -- REPORTS, never fails the leg (THE LAW). Sonnet QA before the push found two real defects (a raise path into the campaign loop off a torn ledger; newest-mtime read-back silently attributing a concurrent episode's title), and retest found a third (os.stat outresolves datetime.now, dropping the leg's own ledger at the window edge -- `READBACK_GRACE_S`). All three fixed pre-push; 24 tests pin it (`tests/test_soak_title_provenance.py`). Publish path untouched.
- bible-worthy: LIKELY YES -- the class is "a diagnostic harness reuses a PRODUCTION field as its own scratch label, and its runs then travel the production path into the deliverable folder". Adjacent to the bench carve-out ruling but the bench case was decided in advance while this one shipped unnoticed. Not promoted until there is a fix to claim.
- promotion: NONE -- covered by existing `12.110`/`11.61`; checked against the index, no new class
- status: FIXED `e21b27ba`, unpromoted (12.110 already covers the one-field-two-meanings shape; nothing new to promote)

## PBUG-20260817-04 -- the public_domain announcer invents a work title even when it is given the real one
- surfaced: live acceptance leg 2026-08-17, `scripts/otr_writer_bank_gate.py --banks public_domain --acts 1` on code `b45c5577`, RESULT SUCCESS; episode `signal_lost_the_blackwood_enigma_20260817_172553`. Found by the item-F acceptance leg the operator authorised, not by review.
- symptom: the opening announcer line says *"Tonight, from the cluttered confines of an office, we gather for 'The Adventure of the Purloined Paper' ..."*. **That work does not exist.** The adapted source is `Nonsense Novels` by Stephen Leacock (unit *"Maddened by Mystery, or The Defective Detective"*), and the EPISODE title is `The Blackwood Enigma`. So the announcer names a THIRD string that is neither the source nor the episode -- a frame/content contradiction on the lane where fidelity outranks arc. The closing coda is CORRECT: *"Tonight's tale was adapted from Nonsense Novels, by Stephen Leacock."*
- root cause: NOT a threading failure -- item F's fix worked and the fact was delivered. Verified by replaying the shipped ledger through the real code: `identity_from_meta(meta).work_title == "Nonsense Novels"`, `source_kind == "public_domain"` which IS in `ADAPTATION_SOURCE_KINDS`, so the writer passed it and `_work_line` renders `WORK: a scene from Nonsense Novels` into the prompt. `meta.announcer_intro_rewrite == "announcer_intro_rewritten"`, so the second producer wrote this line and it also receives the title. **The model was given the right work and named a different one.** Open question for the panel: whether an obscure collection title ("Nonsense Novels") invites invention where a famous one ("The Tempest") does not -- the shakespeare leg on the same commit named its work correctly.
- fix: NONE YET, deliberately. This is a design question, not a threading bug, and guessing at seam wording is the mechanism item F already proved unreliable -- that seam has said *"invent none"* the whole time.
- verify idea: assert the opening announcer line names the `work_title` the ledger carries, OR names no work at all -- never a third string. Cheap and deterministic against a shipped ledger. Note the existing `tests/test_cross_play_frame_leak.py` CANNOT catch this: it detects names belonging to OTHER manifest rows, and an invented title belongs to none -- the exact residue that file's docstring says the live leg is for.
- bible-worthy: LIKELY YES, but a DIFFERENT class from `11.61` -- that entry is "the fact never reached the prompt"; this is "the fact reached the prompt and the model overrode it", i.e. supplying material is necessary and not sufficient. Do not promote until there is a fix to claim.
- promotion: PENDING
- status: OPEN

## PBUG-20260817-03 -- another character's NAME pasted into a character_description
- surfaced: published ledger on disk, `signal_lost_midnight_circuit_20260803_162229_ledger.json` (found 2026-08-17 while measuring item G; verified by the driver reading the ledger, not inferred)
- symptom: RICK STEINER's `character_description` opens *"Late 50s, Seasoned yet vulnerable LUCILLE PENNY. Face: Oval, knitted brows..."* -- i.e. it carries a DIFFERENT cast member's name, and the face/detail prose that follows belongs to her. A second instance was reported in the same sweep (OYA SATO carrying Hank Griswold's), not independently re-read.
- root cause: **MEASURED 2026-08-17 (second window), no longer unknown -- TWO NAMING AUTHORITIES, resolved inside a prompt.** The pitch names the characters (`meta.source_meta.selected_concept.cast[].name` = LUCILLE PENNY, HAROLD 'HAL' BRIGHT) and that text is restated in `meta.news.casting_brief` (*"We need a seasoned yet vulnerable LUCILLE PENNY..."*). The cast pool then assigns DIFFERENT names (RICK STEINER, NIA PHILBIN). `_otr_casting.build_description_prompt` puts the brief on the `Story:` line and the assigned name on the `Name:` line **with no statement of which wins**, and its own CHARACTER VISUAL CONTRACT format reserves a free-text slot right after the age band (`"<age decade>, <story-linked role>. Face: ..."`). The model fills that slot with the brief's NAME -- a plausible reading of "story-linked role" when the story text keeps naming people. The prior-cast theory is DISPROVEN: LUCILLE PENNY is not a cast row in that episode or anywhere in the 1,710-ledger corpus, so `_format_prior_entry` cannot be the path.
- fix: **ROOT-FIXED AND LIVE-VERIFIED.** Item I shipped the shared name-authority boundary first. The 2026-08-21 `media_archive` extension then added structured `upstream_identity_names` at the interpreter, validated the optional field coherently on both shared payload surfaces, and merged it into that existing boundary without mining prose. Frozen v1 results may omit the key; a fresh v2 response must state it explicitly (including `[]`) so the path cannot default into another no-op. The finished-diff review found that explicit-key gap; it was fixed before acceptance. Historical artifacts remain frozen -- no backrepair.
- verify idea: **CORRECTED -- the original idea here was wrong and would not have caught this episode.** It read *"assert no `character_description` contains the `name` of any OTHER row in the same cast"*. Measured over all 1,710 ledgers, that check is wrong in BOTH directions: it returns 47 hits of which roughly 45 are legitimate relational prose (*"foil to the Time Traveler"*, *"Rosalind's loyal best friend"*), and it does NOT flag `signal_lost_midnight_circuit` at all, because LUCILLE PENNY is nobody's cast row. **The check that works is ENSEMBLE-FOREIGN:** flag a `character_description` containing a proper name that NO cast row owns, sourced from the pitch cast. Measured: **28 rows across 20 of the 124 pitch-bearing ledgers (16%)**, and it flags exactly the two known-bad rows in the reported episode. Extend the same assertion to `meta.visual_plan.characters[NAME].portrait_prompt`, which carries the contaminated string verbatim -- the portraits were painted from the wrong person's face.
- scale + blast radius: **28 rows / 20 ledgers is a FLOOR, not a total, and the scoping matters.** That count comes from the pitch-cast detector, which can only see the 124 of 1,710 ledgers that recorded `selected_concept.cast`. A second, independent name-shape detector flags **18 rows / 14 ledgers** and its set is NOT the same -- it catches `signal_lost_the_wax_cylinders_whisper_20260805_102216` (OYA SATO <- *"30s, Henry 'Hank' Griswold."*) and `signal_lost_nightshift_erasure_20260809_115705` (RYAN KAPOOR <- *"60s, EDWARD 'ED' GRISWOLD."*), which the pitch detector misses, while missing LUCILLE PENNY, which the pitch detector catches. **So the operator's second reported instance is CONFIRMED**, and the true census is the union of the two, which is not yet computed -- that is item I's job, not this promotion's. Most recent hit `signal_lost_lemmy_provisional_tier_kokoro_acceptance_20260816_210751` (2026-08-16), so this is **live at HEAD, not a retired regime** -- the distinction that separates it from item G. Gender-crossed in both directions (RICK STEINER male <- LUCILLE PENNY; WENDY PALMER female <- SIR REGINALD PENNYWORTH), which is why it accounts for part of item G's portrait-conflict count. Note also that the contamination survives a FREEZE: `baked_ledger.json` carries the RYAN KAPOOR row verbatim in fourteen copies.
- media_archive extension scale + live proof (2026-08-21): all 1,736 ledger JSONs parsed; the bank contributes 104 ledgers. A conservative structured-name review confirmed **five dirty rows in five of 104 ledgers**, four of those five episodes published. The canonical workflow ran with runtime API overrides only and published `signal_lost_the_ink_still_wet_20260821_014318`; its three structured upstream identities reached `meta.name_authority.upstream_identities`, zero reached the final cast prose/image prompts, and all 8/8 still clips completed.
- bible-worthy: YES, and promoted. Checked against `otr_coverage_index.yaml` and the 287-entry Bible: genuinely uncovered. NOT `10.03` (wrong character name in a DIALOGUE body) -- that entry's fuzzy-roster-repair fix is actively harmful here, since it would rename the intruder to the row's own name and leave the other person's face and voice prose in place. NOT `10.08` (two correlated attributes from two Python draws, reconciled before freeze).
- promotion: BUG-11.61 (survival-guide `ff0eb13`; 287 -> 288 entries, README count bumped in all three citations, index row appended, suite green at 20/26/3)
- status: **FIXED + LIVE-VERIFIED; RULE REMAINS PROMOTED BUG-11.61.** Separate gaps (`scifi_news_pro`, historical backrepair, and `OTR_NAME_MODE=llm_slot_fill`) remain separately scoped.

## PBUG-20260612-01 -- headless boot dies on cp1252 emoji print
- surfaced: detached headless soak/API boot, 2026-06-12
- symptom: boot dies ~13s, exit 1, "SERVER DID NOT COME UP"
- root cause: detached cmd inherits cp1252; prestartup_script.py printed U+2705/U+2713 -> UnicodeEncodeError
- fix: scripts/_otr_soak_server_launch.cmd sets PYTHONUTF8=1 + PYTHONIOENCODING=utf-8; rule codified in CLAUDE.md section 5
- verify idea: launcher-path boot succeeds; any new boot path asserts UTF-8 env
- bible-worthy: yes -- Windows console-codec boot killer, hits any custom node that prints unicode at import
- confidence: MED (sourced from operating-rules doc, no dated incident log)
- status: PROMOTED BUG-02.15

## PBUG-20260616-01 -- LTX-AV soak VRAM peak 15.8GB over the 14.5GB cap
- surfaced: LTX full-episode soak, 2026-06-16 (976ab329)
- symptom: soak measured 15.8GB peak on a 14.5GB gate, both device modes
- root cause: Gemma text encoder stayed GPU-resident through the LTX pass
- fix: b0925c37 moved encoder to cpu; 1e5d66f4 REVERTED it after soak re-measure proved the offload ineffective -- record documents a fix attempt that live evidence disproved
- verify idea: full-episode soak VRAM peak check; assert S9 offload state matches the reverted decision
- bible-worthy: yes -- "the obvious offload fix measurably did nothing" is worth pinning so it isn't retried blind
- confidence: HIGH
- status: PROMOTED BUG-07.17

## PBUG-20260618-01 -- remote creative slot crashed episode with KeyError
- surfaced: live run with creative_model='openrouter:slot-a', 2026-06-18
- symptom: episode aborted at line-compose with KeyError
- root cause: resolve_creative_system_prompt did rows[repo_id] against a CURATED_LLM_MODELS-only dict; remote handles aren't in it
- fix: 1f196ac3 -- rows.get(repo_id) with MODERN-prompt default
- verify idea: full episode with a remote slot handle completes, modern prompt used
- bible-worthy: yes -- exact-match lookup vs non-curated id, recurring trap
- confidence: HIGH
- status: PROMOTED BUG-11.27

## PBUG-20260618-02 -- visualizer soak found 4-bug integration cluster
- surfaced: Task 2 visualizer soak, 2026-06-18 (4a92ed66, 21 clips)
- symptom: crashes/misbehavior on 0-frame beats, silent beats, missing master-audio slice, over-gated audio_ref
- root cause: four missing guards -- no 0-frame floor, no idle-scope handling, audio_ref wrongly gated in assert_usable, b000 master slice never fed
- fix: afab1a3 + c5c14c90 + d4607974 + bad1bba3
- verify idea: visualizer soak forcing silent/0-frame beats, status=success
- bible-worthy: yes -- soak-found cluster, four distinct root causes
- confidence: HIGH
- status: PROMOTED BUG-07.18

## PBUG-20260620-01 -- published episode bars overlay read the silent source
- surfaced: obs-final render pipeline, 2026-06-20 (8d7e6604 verification)
- symptom: bottom bars overlay baked flat/green instead of audio-reactive in a PUBLISHED episode
- root cause: bars overlay read the silent blend source instead of the master WAV
- fix: f6788882 -- bars read the master WAV
- verify idea: obs final render, assert bars track master audio amplitude
- bible-worthy: yes -- defect shipped to a published artifact
- confidence: HIGH
- status: PROMOTED BUG-08.07

## PBUG-20260622-01 -- UnboundLocalError crashed every episode at flag-stamp
- surfaced: night-soak window, 2026-06-22 (096ef64e)
- symptom: every episode crashed with UnboundLocalError at execution
- root cause: local `import os` inside run() made os function-local; the L2/L7 meta-stamp referenced os.environ before the local import line executed
- fix: 096ef64e -- local import at the stamp site; suite never exercised the heavy node so it slipped through
- verify idea: end-to-end test exercising the L2/L7 stamp; lint for mid-function shadowed imports
- bible-worthy: yes -- Python scoping trap invisible to unit tests
- confidence: HIGH
- status: PROMOTED BUG-05.10

## PBUG-20260622-02 -- announcer coerced to character role, voice engine crash
- surfaced: live-smoke, 2026-06-22 (ffe23245, "(live-smoke)" tag)
- symptom: pre-freeze sweep re-roled the announcer intro to character -> bark engine -> EngineUnusable
- root cause: cast_ids_from_ledger didn't exempt a cast row NAMED ANNOUNCER from role coercion
- fix: ffe23245 -- exclude ANNOUNCER-named rows from coercion
- verify idea: episode with announcer keyed as ordinary cast id renders clean
- bible-worthy: yes -- naming-convention trap in role coercion
- confidence: HIGH
- status: PROMOTED BUG-07.19

## PBUG-20260622-03 -- stage-direction-only character line crashed voice render
- surfaced: live-smoked fix set, 2026-06-22 (f8a8645e)
- symptom: a line with zero spoken content reached the voice engine and crashed the render
- root cause: no handling for a dialogue row that was pure stage direction
- fix: e62081f9 recompose to real dialogue (root); 9a4f0a71 silence backstop (NOTE: backstop is a fail-soft -- flag against current no-fallback law at fan-out)
- verify idea: force a stage-direction-only line through; assert recompose path, no crash
- bible-worthy: yes -- degenerate-content class
- confidence: MED
- status: PROMOTED BUG-07.20

## PBUG-20260623-01 -- refine-loop save failures racing the freeze cascade
- surfaced: live-smoke, 2026-06-23 (9f29f644)
- symptom: intermittent save failures during the refine loop
- root cause: loser-directory cleanup raced the freeze cascade
- fix: 9f29f644 -- ship the LAST revision, drop the racing cleanup
- verify idea: repeated refine-loop runs, zero save failures, freeze lands
- bible-worthy: yes -- race class, easy to reintroduce with future cleanup code
- confidence: HIGH
- status: PROMOTED BUG-12.48

## PBUG-20260702-01 -- night-queue proof9c: VRAM ceiling breach, zero clips
- surfaced: overnight night-queue run, 2026-07-02 (4dd79dbe verdict)
- symptom: leg produced zero clips; VRAM ceiling ops breach mid-run
- root cause: never fully isolated; retried at 832x448 per the verdict
- fix: none identified (diagnostic verdict only)
- verify idea: n/a until root-caused
- bible-worthy: no -- unresolved diagnostic, keep for the record
- confidence: LOW
- status: OPEN

## PBUG-20260703-01 -- overnight soak died: Ollama daemon down
- surfaced: overnight model-matrix soak, 2026-07-03 (c36dfe3e)
- symptom: soak died mid-run, local-LLM legs had nothing to call
- root cause: daemon down; no preflight health check in the soak launcher
- fix: c36dfe3e -- daemon started, soak relaunched (env fix, not code)
- verify idea: soak launcher preflights daemon health before queuing legs
- bible-worthy: maybe -- precondition-check class, though root cause was environmental
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The Ollama daemon architecture no longer exists: `grep -rn ollama nodes/*.py` returns ZERO hits, and the surviving references are negative statements (`_otr_model_catalog.py` -- "No Ollama, no sidecar, no port"). Replaced by the local OpenAI lane in `bde057f7`. A daemon that cannot be started cannot go down.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, the subsystem is gone**

## PBUG-20260704-01 -- Sonilo cloud music rejected 422 provider_rejected
- surfaced: live cloud-audio proving run, 2026-07-04 (8f146394 "FIXED+PROVEN live")
- symptom: music calls rejected HTTP 422
- root cause: requested duration under provider minimum, no floor applied
- fix: 8f146394 -- min-duration floor + trim
- verify idea: short-duration Sonilo request completes
- bible-worthy: yes -- cloud-API contract violation class
- confidence: HIGH
- status: PROMOTED BUG-09.05

## PBUG-20260704-02 -- nano_banana_2 TypeError: string indices must be integers
- surfaced: live cloud-image coverage sweep, 2026-07-04 (606dc7f1)
- symptom: cloud_nano_banana_2 requests crashed with TypeError
- root cause: GeminiNanoBanana2V2 expects model as DYNAMICCOMBO_V3 dict; node sent a bare slug string (seedream's different node takes the bare string -- contract varies per node)
- fix: 606dc7f1 -- send the dict shape
- verify idea: live nano_banana_2 render completes
- bible-worthy: yes -- dict-vs-string contract mismatch across V3 cloud nodes
- confidence: MED
- status: PROMOTED BUG-09.06

## PBUG-20260709-01 -- distinct Chatterbox voice ids shared one WAV
- surfaced: all-Chatterbox 30w OBS live smoke, 2026-07-09
- symptom: two logically distinct voice ids resolved to the same underlying WAV
- root cause: no same-asset/provider collision check when allow_voice_reuse=False
- fix: same-day fix blocks asset/provider collisions under no-reuse (see GO_FORWARD 2026-07-09)
- verify idea: resolve N ids under allow_voice_reuse=False, assert distinct WAV hashes
- bible-worthy: yes -- no-reuse-gate class for any engine with shared assets
- confidence: HIGH
- status: PROMOTED BUG-07.21

## PBUG-20260710-01 -- gemma-4 Q8 silent n_ctx downgrade truncated concept JSON
- surfaced: original_radio live 30w smoke, 2026-07-10
- symptom: creative-slot output truncated -> schema failures downstream
- root cause: gemma-4 Q8 can't hold n_ctx 4096 on 16GB; silent 2048 downgrade
- fix: d526c8b7 creative slot -> Mistral-Nemo in canonical; portability S1 later made ALL silent n_ctx downgrades raise
- verify idea: request n_ctx over capacity, assert raise not downgrade (S1 test should already pin)
- bible-worthy: yes -- silent-downgrade class, though S1 now kills it globally
- confidence: HIGH
- status: PROMOTED BUG-11.28

## PBUG-20260710-02 -- epilogue_missing false-positive killed a roll with outro present
- surfaced: original_radio live smoke hardening, 2026-07-10
- symptom: roll killed for "epilogue_missing" while the outro row existed
- root cause: detection check + slot pins mistargeted
- fix: 1c735c2d -- deterministic refutation when the outro row exists, pins retargeted
- verify idea: fixture with outro row at retargeted slot, assert no false kill
- bible-worthy: check overlap with BUG-11.26 family at fan-out (this commit was NOT in the four folded into 11.26)
- confidence: MED
- status: PROMOTED (folded into BUG-11.26 law d, no new entry)

## PBUG-20260710-03 -- QA judge "proved" a violation by quoting clean text
- surfaced: original_radio 420w night batch Roll A, 2026-07-10
- symptom: confirm judge killed a roll for news_source_framing citing the CLEAN intro verbatim
- root cause: judge kill lacked lexicon-only corroboration for closed-vocabulary classes
- fix: 3d32b265 -- news_source_framing + machine_attribution became lexicon-only kill classes
- status: PROMOTED (folded into Bible BUG-11.26 follow-on law c, survival-guide commit 2833863)

## PBUG-20260710-04 -- fable2 P3 reroll: jinja TemplateError on consecutive user messages
- surfaced: scifi_fable2 30w live smoke roll 2, 2026-07-10
- symptom: TemplateError mid-render on the P3 reroll path
- root cause: reroll emitted two consecutive user-role messages; chat template requires alternation
- fix: fold reroll into ONE user message (docs/2026-07-10-fable2-s1b-smoke-hardening.md)
- verify idea: construct a P3 reroll, assert strict role alternation
- bible-worthy: yes -- chat-template alternation, easy to reintroduce in any lane
- confidence: HIGH
- status: PROMOTED BUG-11.29

## PBUG-20260710-05 -- fable2 casting JSON truncated at 1000-token budget
- surfaced: scifi_fable2 30w live smoke roll 18, 2026-07-10
- symptom: casting JSON truncated at ceiling; salvage pulled a partial object that failed schema
- root cause: 1000-token budget too small for the structured payload
- fix: budget 1400 + wrapper-tolerant before-validator (same doc)
- verify idea: near-ceiling casting payload completes without the salvage path firing
- bible-worthy: yes -- token-ceiling truncation-then-salvage class, already recurred cross-lane
- confidence: HIGH
- status: PROMOTED BUG-11.30

## PBUG-20260710-06 -- fable2 word-band exhaustion: proportional band too narrow at small targets
- surfaced: scifi_fable2 30w live smoke roll 17, 2026-07-10
- symptom: roll died on WORD_BUDGET exhaustion (54 words vs 24-36 band)
- root cause: +/-20% proportional band is only 12 words wide at target=30
- fix: absolute slack floor +/-25 words; proportional governs >=125w (same doc)
- verify idea: unit test _word_band at target=30, absolute floor governs
- bible-worthy: yes -- same defect class flagged UNFIXED in original_radio P1-1; not yet generalized
- confidence: HIGH
- status: PROMOTED BUG-11.31

## PBUG-20260710-07 -- fable2 announcer row silently mutated to character+skip, reason null
- surfaced: scifi_fable2 30w live smoke roll 22, 2026-07-10
- symptom: postamble row arrived speaker_role=character, skip=True, tts_skip_reason=null after a green 8-pass spine -- no compose-flag breadcrumb
- root cause: UNKNOWN -- an unsanctioned cast-keyed mutator downstream; ROOT MUTATOR STILL UNIDENTIFIED
- fix: partial -- announcer sentinel char_id exempts rows from cast-keyed paths; mutator not found
- verify idea: trace/assert every cast-keyed mutation path; no path may flip announcer without stamping a reason
- bible-worthy: yes, HIGH PRIORITY -- silent data corruption with unresolved root cause
- confidence: MED
- status: PROMOTED BUG-11.32 (ROOT CAUSE OPEN)

## PBUG-20260710-08 -- fable2 injected fictional character into the real-news read
- surfaced: scifi_fable2 30w live smoke roll 9, 2026-07-10
- symptom: model placed its fictional heroine ("Lia") in the read-only real-news pass
- root cause: no gate against invented cast names leaking into the source-read pass
- fix: cast-name-in-read gate with teaching error (same doc)
- verify idea: fixture with fictional name in read output, assert gate rejects with repair prompt
- bible-worthy: yes -- fiction/fact bleed class, distinct from verbatim grounding
- confidence: HIGH
- status: PROMOTED BUG-11.33

## PBUG-20260710-09 -- fable2 CODA terminal punctuation killed a clean draft
- surfaced: scifi_fable2 30w live smoke roll 15, 2026-07-10
- symptom: otherwise-passing draft killed solely for CODA ending '.' instead of ':'
- root cause: colon is structurally load-bearing to a parser; treated as stylistic by the model, no normalization before the check
- fix: pivot colon normalized in shared pre-lex (flagged); inner sentence break remains the true defect (same doc)
- verify idea: CODA ending '.' normalizes before parse, no false kill
- bible-worthy: yes -- structural-punctuation-as-parser-key class; original_radio P2-2 flags same risk
- confidence: HIGH
- status: PROMOTED BUG-11.34

## PBUG-20260710-10 -- scifi bake-off canonical smoke halted at Codex P0: source-span mismatch
- surfaced: first scifi_codex canonical 30w live smoke (roll 2a), 2026-07-10
- symptom: technical model returned a fact whose source_spans quote != the payload slice; validator correctly halted before any dialogue/media spend
- root cause: repair prompt not explicit about field/start:end slice contract; typed repair reproduced the mismatch
- fix: `40a765ac` hardened originating-slot repair prompt showing required payload[field][start:end] identity + slice-mismatch diagnostics, applied to ALL THREE lanes (cross-lane audit found the same contract shape in Gemini/Sonnet P0)
- verify idea: offset-span fixture converges within the repair ladder budget
- bible-worthy: yes -- evidence-span contract class, cross-lane by construction
- confidence: HIGH
- status: PROMOTED BUG-11.35

## PBUG-20260711-01 -- scifi bake-off Codex P0: evidence-ID shape F0/F1 vs required F01/F02
- surfaced: scifi bake-off canonical 30w smoke roll 2b, 2026-07-10/11
- symptom: local model returned evidence IDs F0/F1/F2 where the v4 contract requires zero-padded F01/F02/F03; P0 validator halted the run
- root cause: typed-repair contract didn't give the model explicit lexical ID mappings; ID-shape expectation implicit
- fix: `731d49f7` repair contract tightened at the shared lane boundary across Codex/Gemini/Sonnet -- explicit lexical ID mappings + recompute-quotes-from-payload-slice instruction (dialogue untouched, metadata repair deterministic); roll 3 rerun pending
- verify idea: fixture returning unpadded IDs, assert repair converges to padded shape within budget; pin pad width in schema tests
- bible-worthy: yes -- ID-shape contract drift, second member of the P0-contract class with PBUG-20260710-10
- confidence: HIGH
- status: PROMOTED BUG-11.36 (roll 3 exposed the NEXT defect rather than hiding it -- see PBUG-20260711-02)

## PBUG-20260711-02 -- scifi bake-off Codex P0: correct ID, wrong quote offsets (span-integrity)
- surfaced: scifi bake-off canonical 30w smoke roll 3, 2026-07-11
- symptom: after the ID repair converged (F0 -> F01 correct), the model repeated a quote with WRONG offsets -- a separate P0 span-integrity failure; validator halted honestly
- root cause: repair contract fixed ID shape but did not force offsets to be recomputed against the payload slice
- fix: `731d49f7` fail-closed METADATA-ONLY repair module (nodes/_otr_scifi_source_repair.py + test): may reindex an EXACT quote already present in the source and normalize IDs; may NOT invent or rewrite dialogue. Dialogue rewrites remain the province of a later context-aware structured creative pass (premise + beats + cast lock + audit feedback in hand) -- operator ruling: never a blind Python hack or context-free LLM retry that breaks the story arc
- verify idea: offset-shifted exact-quote fixture reindexes deterministically; ID normalizer pins F0 -> F01 (NOT F00 -- an actual test defect caught during this fix); dialogue field asserted byte-identical through repair
- bible-worthy: yes -- completes the P0 evidence-contract trilogy (span fidelity / ID shape / offset integrity); strong class entry at fan-out
- confidence: HIGH
- status: PROMOTED BUG-11.37

## PBUG-20260711-03 -- Codex creative score pass returned legacy Markdown shape
- surfaced: scifi bake-off canonical 30w smoke, Codex P3, 2026-07-11
- symptom: base and structural attempts returned Markdown prose; typed repair returned a legacy score object with missing `RadioScoreV4` keys and extra advisory-plan keys; the strict ladder halted before dialogue/media spend
- root cause: the pack seam said JSON-only but did not state the exact current schema's top-level keys, so the local model copied input structure instead of the requested typed artifact
- fix: `0d94c437` appends exact required top-level keys to every typed Codex/Gemini/Sonnet pass and repair seam; it preserves the full story context and forbids Markdown/prose
- verify idea: capture a typed pass prompt for each lane and assert the schema's required top-level keys are named; live smoke must reach the next pass without legacy-key drift
- bible-worthy: yes -- live structured-output contract failure, with cross-lane prevention
- confidence: HIGH
- status: PROMOTED BUG-11.38

## PBUG-20260711-04 -- Codex P0 used a full quote with truncated or wrong source field metadata
- promotion: BUG-11.46
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: a full headline quote was returned with `headline[0:55]`, so the validator saw only a truncated payload slice and halted the lane
- root cause: the model supplied a stale end offset and, in some artifacts, source-field labels did not identify the field containing the exact quote
- fix: `55f3cf17` rehomes an exact quote only when exactly one allowed payload field contains it, then recomputes start/end; absent or ambiguous evidence still fails closed
- verify idea: fixture with wrong field and offset rehomes to the unique literal field; fixture with absent or duplicate quote returns no repair
- bible-worthy: yes -- live source-evidence metadata failure, cross-lane helper
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`55f3cf17`) lived only in `nodes/_otr_scifi_source_repair.py`, deleted whole by the scifi_news rip `dae1fb3c` (2026-08-16).
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, module deleted**

## PBUG-20260711-05 -- JSON parser salvaged a nested fact from a broken outer artifact
- promotion: BUG-11.47
- surfaced: scifi bake-off canonical 30w smoke, Codex P0, 2026-07-11
- symptom: malformed outer fact JSON was scanned past its first brace; the parser returned the first nested fact object, producing misleading missing-top-level-key errors and preventing the intended repair path
- root cause: shared fallback scanning treated a nested child as a valid top-level object when the response began with an invalid outer object
- fix: `5489baa8` fails closed when a response begins with malformed outer JSON instead of salvaging nested children; all source packs use the shared parser
- verify idea: malformed outer-with-valid-child fixture raises a top-level parse error; valid leading prose plus a valid object still parses normally
- bible-worthy: yes -- shared structured-call integrity defect across source packs
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). **Do not bulk-close this with its 711-series siblings.** The fix (`5489baa8`) did NOT die with the scifi module: `extract_first_json_block` lives in `nodes/_otr_json.py` and is imported today by `_otr_casting.py`, `_otr_outline.py`, `_otr_story_brief.py`, `_otr_structured_call.py` and `news_interpreter.py`. It became shared infrastructure.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- FIXED, and the fix GENERALIZED**

## PBUG-20260711-06 -- Codex P3 omitted required nested scene graph fields
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke, Codex P3, 2026-07-11
- symptom: score JSON had the correct top-level artifact but omitted required nested `scene_id`, `shot_id`, and `visual_prompt` fields; strict validation halted before script/dialogue/media work
- root cause: the prompt named top-level keys but hand-described no complete nested required-field tree, so the local model repeated an incomplete graph
- fix: `b9cfc508` generates a compact required-path inventory from each Pydantic model's `model_json_schema()` and injects it into all three lane prompt builders
- verify idea: assert `scenes[*].shots[*].scene_id` and equivalent nested paths appear in generated prompts; live smoke must pass P3 graph validation
- bible-worthy: yes -- live nested-schema contract failure, same family as PBUG-20260711-03
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). **Do not bulk-close this with its 711-series siblings.** `schema_required_paths` / `schema_shape_instruction` (`b9cfc508`) now live in `nodes/_otr_structured_call.py` wired into `invoke_structured_slot`, called from `OTR_LedgerScriptWriter.py` and `_otr_openrouter_backend.py` -- a generic mechanism, no longer scifi-specific.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- FIXED, and the fix GENERALIZED**

## PBUG-20260711-07 -- Codex P0 overclaimed beyond the supplied RSS payload
- promotion: BUG-11.46
- surfaced: scifi bake-off canonical 30w smoke roll 6, Codex P0, 2026-07-11
- symptom: the model returned a quote longer than the literal `full_text` payload; typed repair repeated it and the evidence validator halted before downstream work
- root cause: the model treated a claim-like sentence as source evidence even though the supplied payload did not contain that exact span
- fix: `6e6ff57b` drops unsupported facts/entities/numbers during metadata-only repair and retains only literal evidence; if no supported fact remains, the pass still fails closed
- verify idea: mixed fixture keeps literal facts and drops paraphrased facts; all-paraphrase fixture remains invalid
- bible-worthy: yes -- live grounding overclaim, same evidence-contract family as PBUG-20260711-01/02/04
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`6e6ff57b`) lived only in `nodes/_otr_scifi_source_repair.py`, deleted by `dae1fb3c`.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, module deleted**

## PBUG-20260711-08 -- Codex P3 generic repair repeated an incomplete graph
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 7, Codex P3, 2026-07-11
- symptom: base and generic typed repair both omitted required nested scene graph fields despite a valid top-level score object
- root cause: non-P0 passes used the generic repair factory, which did not present the failed artifact and validation error with lane-specific graph-preservation instructions
- fix: `a27206df` routes typed repair for every Codex/Gemini/Sonnet pass through a schema-aware failed-artifact/error prompt while preserving premise, beats, cast, and authored content
- verify idea: force a nested graph validation failure and assert the repair prompt includes the failed artifact, exact validation error, schema paths, and context-preservation rule
- bible-worthy: yes -- live repair-contract failure, cross-lane by construction
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`a27206df`) touched only `_otr_scifi_gemini.py` (deleted `3312aec7`) and `_otr_scifi_sonnet.py` (deleted `c507acff`).
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, modules deleted**

## PBUG-20260711-09 -- Codex P3 repair omitted cast-locked speaker fields
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 8, Codex P3, 2026-07-11
- symptom: schema-aware repair reduced the failure to two missing `speaker` fields on beats; the lane halted before script/media work
- root cause: nested graph repair did not explicitly bind each beat's speaker to its cast row by `char_id`
- fix: `fca99a5a` adds the cast-lock mapping rule to typed repair prompts for all three lanes
- verify idea: force missing beat speakers and assert the repair prompt requires cast-row lookup by `char_id`; live Codex P3 must clear
- bible-worthy: yes -- live cast/graph integrity contract failure, cross-lane prevention
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`fca99a5a`) touched only the gemini/sonnet modules, both deleted.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, modules deleted**

## PBUG-20260711-10 -- Codex P5 repair omitted ScriptLine boundary metadata
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w smoke roll 9, Codex P5, 2026-07-11
- symptom: full script artifact was otherwise shaped, but all eight lines omitted required `boundary` values; strict validation halted before audio/media work
- root cause: the repair contract named nested fields but did not define the boundary derivation from shot/beat order
- fix: `94331eb2` adds the structural rule: first line in shot = `shot_start`, first line in beat = `beat_start`, otherwise `continue`
- verify idea: force missing boundaries and assert the repair instruction contains the three-way derivation rule; live P5 must clear
- bible-worthy: yes -- live script graph metadata failure
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`94331eb2`) touched only `_otr_scifi_codex.py`, deleted whole by `dae1fb3c`.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, module deleted**

## PBUG-20260711-11 -- Canonical RSS selector delivered a thin science payload
- promotion: BUG-11.49
- surfaced: scifi bake-off canonical 30w smoke roll 10, 2026-07-12
- symptom: run halted before P0 with `RSS payload is below the 80/12 thinness floor`; Gemini and Sonnet remained not-started because the serialized smoke gate stopped at Codex
- root cause: common science RSS selection returned a thin article to a lane whose source contract requires a substantial RSS body
- fix: `d01cf8bc` makes the shared RSS selector inspect up to ten candidates for sci-fi v4, require the same >=400-char/80-word/12-unique-token source floor before selection, and fail at selection if none qualify; legacy `science_news` keeps its existing richest-body fallback
- verify idea: canonical RSS fetch should either return a payload meeting the 80/12 floor or fail before queueing the sci-fi lane with a clear source-selection reason
- bible-worthy: yes -- live shared source-precondition failure
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`d01cf8bc`) gates on `strict_v4_banks = {scifi_codex, scifi_gemini, scifi_sonnet}`. VERIFIED: none of those ids are in `nodes/story_packs/banks.json`, whose live roster is media_archive / original / scifi_news_pro / public_domain / shakespeare / custom_source_bank. Dead code guarding banks that cannot be selected.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, the banks no longer exist**

## FAN-OUT RECORD -- 2026-07-11 (operator-triggered)
23 entries promoted to the Bible (156 -> 179) @ survival-guide commit d50d773;
1 folded into BUG-11.26 law d (epilogue false-kill class); suite 17 passed /
7 skipped / 3 xfailed green; all 23 as non-testable notes (runtime-only
verifies), per the existing note pattern. Held OPEN: PBUG-20260702-01 (no
root cause), PBUG-20260703-01 (environmental). Mapping stamped per entry above.

## PBUG-20260711-12 -- Codex P5 output reservation truncated its own schema contract
- promotion: BUG-11.50
- surfaced: scifi bake-off canonical 30w smoke roll 11, Codex P5, 2026-07-11
- symptom: both P5 attempts returned prose or a score-shaped object instead of `ScriptArtifactV4`; the prompt guard reported `Truncated ... -> 1692 tokens` before each call
- root cause: P5 reserved a fixed 6500 output tokens inside an 8192-token context even for a 30-word script, leaving too little input budget for the failed artifact, graph, schema paths, and repair instructions
- fix: `fdc413ed` scales Codex whole-script P5/P7/P9 output reservation from the requested word steer (30w = 2200 instead of 6500), keeps every generated required path, removes the duplicate full schema from typed repair, and records token-budget/raw-size receipts; eight Kibitz reviews converged on the exact call-site wiring
- verify idea: 30w P5 prompt is not truncated, required ScriptArtifactV4 paths remain in the effective prompt, and canonical Codex reaches publish
- bible-worthy: yes -- live context-budget/structured-output contract failure
- confidence: HIGH
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The fix (`fdc413ed`) touched only `_otr_scifi_codex.py`, deleted whole by `dae1fb3c`.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- OBSOLETE, module deleted**

## PBUG-20260711-13 -- Codex P5 typed repair retained two forbidden legacy metadata values
- promotion: BUG-11.48
- surfaced: scifi bake-off canonical 30w Codex reroll after `fdc413ed`, 2026-07-11
- symptom: full-contract P5 base output failed eight fields; typed repair corrected six but retained `schema_version=scifi_codex.script_artifact.v1` and one `boundary=beat_end`, so strict ScriptArtifactV4 validation halted before publish
- root cause: the repair prompt exposed the exact literal and boundary enum contract but the local model copied two legacy values from its own failed artifact; there is no deterministic metadata-only normalization for ScriptArtifactV4 yet
- fix: `e679b754` adds `repair_script_artifact_metadata` -- a deterministic, metadata-only ScriptArtifactV4 repair that derives every mechanical field from the already accepted score graph: it sets the exact v4 schema literal, drops forbidden strict-model extras (e.g. `speaker`), maps each line's `shot_id` from the accepted graph, and derives `boundary` from accepted line/shot/beat order. It never touches dialogue, premise, beats, character intent, or any other story content, and fails closed when a graph or raw-line mapping is missing or ambiguous. The typed-repair factory short-circuits the LLM repair call whenever the deterministic result also satisfies the pass content validators
- verify idea: a metadata-only repair may set the schema literal, remove forbidden extra keys, map line shot IDs from the accepted score, and derive boundary from accepted shot/beat order without changing any dialogue; canonical Codex must then publish before Gemini/Sonnet or 720 starts
- verified: live canonical 30w Codex roll 12 (2026-07-11 08:18) reproduced the exact defect (`boundary=beat_end`) and the deterministic repair resolved it with NO LLM repair call; the lane cleared P5 and continued into the media tail
- bible-worthy: yes -- live legacy-enum persistence in typed repair
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-14 -- content-owned lanes never stamped the TTS delivery text
- promotion: BUG-12.51
- surfaced: scifi bake-off canonical 30w smoke, Codex voice gate, 2026-07-11 (first roll to survive P5)
- symptom: the lane cleared every structured pass, then halted at the voice handoff because its ledger lines carried no pronunciation-safe delivery string
- root cause: content-owned lanes seal canonical `text` in their own runner and bypass the legacy producer that stamps `text_for_tts`; the shared writer tail never stamped it for them
- fix: `e679b754` stamps delivery text in the one shared producer boundary every content-owned bank passes through -- after the last writer-side text mutation and before the lane finalizer's Phase-10 freeze; legacy lanes keep their byte-identical canonical-text delivery path
- verify idea: content-owned tail test asserts delivery stamps exist before the finalizer runs; legacy tail test asserts no stamps are introduced
- bible-worthy: yes -- shared producer-boundary gap that hits every content-owned source bank
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-15 -- content-owned lanes reached CreditsRoll with no cast-seed receipt
- surfaced: scifi bake-off canonical 30w smoke, Codex credits node, 2026-07-11 (first roll to survive the voice gate)
- symptom: the run rendered audio and media, then failed at the final credits node -- the ledger lacked the durable cast/episode seed the no-fallback credits provenance contract requires
- root cause: content-owned lane runners construct their own cast and so bypass the legacy cast-lock producer that stamps the seed receipt; nothing else establishes an authoritative episode seed
- fix: `e679b754` establishes one authoritative cast/episode seed in the shared writer tail (upstream of CastLock, freeze, and CreditsRoll) when the lane has not already supplied one; the seed also drives deterministic downstream voice assignment
- verify idea: content-owned tail test asserts the seed receipt is present before the finalizer runs; credits provenance passes with no fallback
- bible-worthy: yes -- same producer-boundary class as PBUG-20260711-14
- confidence: HIGH
- status: SUPERSEDED by PBUG-20260711-16 -- the receipt was right, the KEY was wrong (see below)

## PBUG-20260711-16 -- a "seed receipt" told CastLock to replay a cast nobody rolled
- promotion: BUG-12.51
- surfaced: scifi bake-off canonical 30w smoke roll 12, Codex CastLock, 2026-07-11 (first roll to survive P5 + the voice gate)
- symptom: the lane cleared every structured pass, stamped 13 delivery lines, rendered, and then died ~14 minutes in with `ValueError: num_characters must be 1-6, got 0` (cast_lock.py:189 -> _assign_bark_voices -> _otr_casting.replay_voice_assignment -> assemble_pre_locked_rows:1211)
- root cause: `meta.cast_contract.cast_seed` is not a generic episode seed -- it is a claim that the WRITER's seeded cast picker produced this cast and can be REPLAYED from it. Content-owned lanes build their own cast rows and stamp their own voice presets in the lane runner, so the picker never ran and the contract carries no `num_characters_request` -> `int(None or 0)` -> 0 -> ValueError. The PBUG-20260711-15 credits fix stamped `cast_seed` as a generic receipt and thereby CLOSED the `cast_seed is None` escape hatch these lanes had always relied on. A fix for one producer gap opened another.
- fix: the shared writer tail stamps `meta.episode_seed` ONLY (otr_credits_roll.py:279-284 already accepts it as the seed receipt, so credits provenance holds without asserting a replayable cast); and cast_lock._assign_bark_voices VERIFIES instead of REPLAYING for a content-owned lane -- it preserves the lane's `voice_preset` values and still runs the Gate 1 invariants, so such a lane can never ship duplicate or non-`v2/` bark voices. The legacy replay path is untouched (test_cast_voice_replay_parity pins it byte-for-byte).
- verify idea: a content-owned meta carrying a cast_seed must NOT enter the replay; a content-owned cast with two identical bark voices must still raise; the fable2 tail test asserts episode_seed is present AND cast_contract.cast_seed is absent
- bible-worthy: yes -- a receipt key that silently doubles as a behavior switch; the "my fix opened the next gap" class
- confidence: HIGH
- status: FIXED (awaiting fan-out)

## PBUG-20260711-17 -- P7 echoed the request envelope and truncated against its own output cap
- promotion: BUG-11.50
- surfaced: scifi bake-off canonical 30w smoke roll 12, Codex P7, 2026-07-11
- symptom: `OUTPUT_CAP: prompt_tokens=4543 generated_tokens=2800 max_new_tokens=2800` then `no decodable top-level JSON object found`; the raw head shows the model emitting `{ "artifact_inputs": { "accepted_line_count": 13, ...` -- the INPUT envelope -- instead of the artifact root. The structural retry happened to recover, so the run survived on luck.
- root cause: (1) the whole-script root contract forbade returning a score/scene/beat/patch but never forbade echoing the request envelope keys (`pass_id`, `artifact_inputs`, `result_json_schema`); (2) `_script_output_token_budget` scaled the reservation from the WORD STEER alone, but a ScriptArtifactV4 serializes strict per-line metadata for every accepted line -- the accepted LINE COUNT drives its size as much as the dialogue does, so a wide graph under-reserves and truncates
- fix: the root contract now names the forbidden envelope keys and requires the response to begin at the v4 schema literal; `_script_output_token_budget(requested_words, accepted_line_count)` scales on both drivers, is computed after the score is final (P3/P3_rewrite), and records a token-budget receipt
- verify idea: budget rises with line count at a fixed word steer; the AST test still pins `script_token_budget` on P5/P7/P9; a 720w run must not truncate
- bible-worthy: yes -- structured-output sizing driven by the wrong dimension; sibling of PBUG-20260711-12
- confidence: HIGH
- status: FIXED (30w); the 720w context-cap ceiling below is still OPEN

## PBUG-20260711-18 -- 720w whole-script passes cannot fit the 8192 context cap (OPEN)
- surfaced: analysis during roll 12, 2026-07-11 -- NOT yet hit live (30w fits)
- symptom (predicted): at 720 words the P7/P9 prompt (full previous script + line graph + review) and the output (the whole script re-emitted) both grow; local `context_cap` defaults to 8192 and the generate_fn LEFT-TRUNCATES silently, eating the system/schema prefix -- the PBUG-20260711-12 failure class, but silent
- root cause: `_build_truncating_generate_fn` uses `int(cache_entry.get("context_cap") or 8192)`; the local transformers path sets no context_cap, so 8192 is an arbitrary default, not a model limit (Mistral-Nemo supports 128k). P5/P7/P9 do not set `prompt_must_fit=True`, so they truncate instead of failing loudly
- fix: NOT APPLIED -- open fork: (a) derive context_cap from the model config with a VRAM-aware ceiling, (b) make P7/P9 a line-level PATCH pass so output stays flat as word count grows, (c) other. Out for a grounded local-panel opinion before the 720w bake-off
- verify idea: measure the real P7 prompt+output cost at 720w; whichever option lands, P5/P7/P9 should fail loud rather than silently truncate
- bible-worthy: yes -- silent context truncation of a provenance-bearing prompt
- confidence: HIGH (arithmetic), UNPROVEN (not yet observed live)
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). The Codex P7/P9 module it warned about is deleted. The successor `_otr_scifi_news_pro.py` carries its own turn-budget guard (`_draft_fits_repair_turn`), and the structural ceiling near 1,520 spoken words fits the 8192 cap.
- previous status: OPEN -- gates the 720w bake-off
- status: **CLOSED 2026-08-18 -- OBSOLETE, superseded by an independent guard**

## PBUG-20260712-01 -- Gemma packed three owned items into suffixed fields
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `0c1bb246-fae0-41c6-8f12-4cd8cccd27f3`, 2026-07-12
- symptom: P3 emitted `lost_object_2`, `lost_object_3`, and `resolution_links_2`; typed repair renamed them to `lost_object_secondary` / `lost_object_tertiary` instead of removing the schema violations, so the run failed closed after 459 seconds
- root cause: the P3 prompt named the collections but never stated that every selected lost object owns one separate `caller_threads` row with one singular `lost_object`, nor that every thread owns exactly one resolution row; Python also did not validate exact cross-artifact lost-object coverage
- fix: `5fd661ab` makes the base and repair contracts explicit, forbids numbered/suffixed pseudo-fields, validates the selected-object multiset, requires clue coverage per thread, and requires exactly one resolution per thread
- verify idea: validate a three-object selected possibility against a truth map with exactly three caller rows, at least one clue per thread, and exactly one resolution per thread; reject packed/suffixed fields, missing objects, duplicate resolutions, and repair-only renames; run the same canonical 30-word bank through Mistral and Gemma families
- bible-worthy: yes -- cross-model structured-output ownership ambiguity is reusable beyond OTR and survived a typed repair by changing only the illegal field names
- confidence: HIGH
- status: FIXED (the next E4B run used one row per object with no suffixed fields; it exposed the distinct nesting bug below; awaiting fan-out)

## PBUG-20260712-02 -- Gemma nested top-level truth collections inside caller rows
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `fc362a77-ec2f-4bf0-a4fc-ac9017eeec53`, 2026-07-12
- symptom: P3 returned a schema-complete top-level truth map but also put a `causal_steps` array inside each of three `caller_threads` rows; typed repair repeated the forbidden nesting unchanged, and the run failed closed after 461.82 seconds
- root cause: the P3 seam and typed-repair rules described collection contents but did not state the exact top-level collection placement or exact caller-row field set; the repair ladder had no safe deterministic relocation for declared collections placed at the wrong depth
- fix: `8f5b3d4d` -- the P3 seam and repair rules name exact nesting, and a P3-only deterministic repair treats an existing top-level collection as authoritative or lifts nested rows verbatim only when top-level is absent/empty; strict schema plus full truth-graph validation must pass or the normal typed LLM repair runs
- verify idea: test authoritative top-level plus nested extras, absent top-level plus verbatim nested rows, non-list nested values, unknown fields, duplicate graph IDs, and a full mocked ladder proving the deterministic repair spends no additional LLM call; repeat Gemma/Mistral canonical smoke
- bible-worthy: yes -- strict item schemas do not prevent a model from placing a valid declared collection at the wrong depth, and typed repair may reproduce the nesting unchanged
- confidence: HIGH
- status: FIXED (next E4B run cleared P3/P4 and exposed the distinct P5 nesting bug below; awaiting fan-out)

## PBUG-20260712-03 -- Gemma nested score shots inside scenes
- promotion: BUG-11.45
- surfaced: canonical 30-word `original_codex56sol` smoke with `google/gemma-4-E4B-it [LOCAL HF]` creative + Mistral technical, prompt `649e1d99-c96d-485b-bce1-f68858f6d2d8`, 2026-07-12
- symptom: the run cleared P1-P4, then P5 returned `shots` arrays inside all four `scenes` rows; typed repair repeated the forbidden nesting after `PROMPT_GUARD` truncated its input from 4751 to 4592 tokens, and the run failed closed after 13:31
- root cause: the BroadcastScore seam and typed-repair rules specified scene and shot fields but did not explicitly require separate top-level scenes/shots/beats arrays; no deterministic structural repair handled declared score collections at the wrong depth
- fix: `54e159ec` -- exact top-level score ownership is stated in base and repair prompts; a P5-only deterministic repair retains authoritative top-level shots/beats or lifts nested rows verbatim when top-level is absent/empty, then requires strict schema and full score-graph validation
- verify idea: test top-level-authoritative cleanup, absent-top-level nested shots+beats lifting, non-list values, unknown fields, duplicate graph IDs, and a full runner with no extra LLM call; rerun the E4B/Mistral canonical smoke
- bible-worthy: yes -- a second typed artifact reproduced the wrong-depth collection class, showing the prevention law must cover every nested row boundary rather than one schema
- confidence: HIGH
- status: FIXED -- canonical E4B/Mistral prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864` cleared P5, froze the ledger, and published the episode; that run exposed the separate semantic false green below

## PBUG-20260712-04 -- Raiders of the Lost Prompts: opaque clue IDs let the spoken story abandon its source bank
- promotion: BUG-11.39
- surfaced: published canonical 30-word `original_codex56sol` episode `signal_lost_the_muted_melody_20260712_020438`, E4B creative + Mistral technical, prompt `fafd6306-cf0a-4c41-9bcb-11d2a8974864`, 2026-07-12
- symptom: history, freeze, audio identity, mux, and OBS publish all succeeded, but the immutable c03 draw (`parcel tag`, `brass button`, `choir note`, `clockwork display`, repair-and-return ending) became an ancient-artifact laboratory procedural speaking `protocol alpha`, `isotopic decay`, `resonance signature`, and `micro-vibrations`; none of the three lost possessions, the device, or the promised return survived into dialogue
- root cause: routing was correct and visual style never entered P1-P9; semantic provenance stopped at opaque clue IDs. P5 proved clue-ID coverage but not clue meaning, P6 received score+manifest without the draw/truth map, script validation checked graph/safety only, P7/P9 could bless a self-consistent replacement cause, and only response hashes survived for intermediate artifacts. The independently selected `sci_fi_radio` visual pack then amplified the already accepted story drift downstream
- fix: add a strict draw-derived grounding contract with literal lost-possession/device/resolution anchors; require object anchors on clue-carrying intents and spoken lines, the device anchor on reveal, and the resolution anchor on closure; thread truth+grounding into P5/P6/all retakes/P9; rerun the blind listener after a blocking retake; make P9 rejection fail closed; add an ordinary-world bank boundary and narrow incident-derived detour phrases; persist accepted intermediate artifacts plus line-level grounding evidence; prove visual-style changes leave every story message byte-identical
- verify idea: the exact seven-line `The Muted Melody` script must fail before P7; independently remove each object/device/resolution anchor and get its exact coordinate; switch only `visual_style` between `sci_fi_radio` and `video_art` and prove captured P1-P9 messages are identical; rerun deterministic c03 at 120 words and require the grounding receipt, frozen ledger, episode final, and OBS final
- bible-worthy: yes -- structured IDs can stay referentially valid while their semantic payload disappears between artifacts; an end-to-end media success is not a content-contract success
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION; the published 30-word episode is retained as a false-green regression artifact and does not qualify the bank

## PBUG-20260712-05 -- Every custom runner title was stamped as a Fable2 title
- promotion: BUG-12.49
- surfaced: forensic audit of the same Codex56 false-green ledger, 2026-07-12
- symptom: `meta.title_source` said `fable2_script_title` even though routing and authorship correctly identified `original_codex56sol`; the stale label could falsely implicate another story bank during incident diagnosis
- root cause: the shared writer tail hardcoded the Fable2 receipt whenever any custom runner supplied `final_title_override`
- fix: derive custom title provenance from `ctx.source_bank_row.source_bank_id`, preserve the established `fable2_script_title` value for the actual Fable2 lane, and stamp `<source_bank_id>_script_title` for every other custom runner without changing the pinned tail-context field contract
- verify idea: direct helper tests for Fable2 and Codex56 plus the existing title-override precedence suite
- bible-worthy: yes -- stale provenance labels turn correct routing evidence into a false root-cause lead
- confidence: HIGH
- status: FIXED IN CODE / AWAITING FAN-OUT

## PBUG-20260712-06 -- Gemma repeated invented music filenames through P5 repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `7384fbe8-d1c9-4485-ba8e-b7f100329a12`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 reached the BroadcastScore on its first base call but added `opening_music.music_file=opening_music.mp3` and `closing_music.music_file=closing_music.mp3`; the typed repair repeated both forbidden fields, so strict validation failed closed after 12:32 and no ledger/media artifact was accepted
- root cause: the score seam closed the top-level, scene, shot, beat, and line-intent key sets but described music bookends only semantically; the shared schema instruction listed their required paths without explicitly forbidding nested extras, allowing a model to treat plausible production filenames as authored score fields
- fix: the bank prompt now states that each music bookend has exactly `description` and `generation_prompt` and explicitly forbids filenames/paths/cue metadata; the existing P5 structural normalizer now deletes only non-authoritative extra bookend keys, preserves every required LLM-authored value byte-for-byte, and still requires the complete strict score plus graph/content validators to pass before it can avoid another model call
- verify idea: inject the exact two `music_file` fields into an otherwise valid score, require deterministic repair with unchanged descriptions/prompts and zero extra LLM calls, pin the prompt wording, then rerun deterministic c03 at 120 words through canonical to ledger and OBS
- bible-worthy: yes -- required nested paths are not the same contract as exact nested key ownership, and a typed repair can faithfully repeat plausible but forbidden production metadata
- confidence: HIGH
- status: FIXED IN CODE / AWAITING LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-07 -- Gemma interleaved complete P5 beat blocks through repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification, prompt `d29b63d8-1890-40a4-a1ea-370bc9b02406`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 produced a strict BroadcastScore with complete typed beats but returned to an earlier `shot_id` after starting another shot; the typed repair repeated the same A/B/A topology and the run failed closed after 11:51 with `beats for each shot must form one contiguous block`
- root cause: the prompt named contiguous shot blocks and Python rejected interleaving, but the contract gave no concrete valid/invalid sequence example and the repair ladder had no safe deterministic ordering projection for otherwise valid authored beats
- fix: the base and repair prompts now state that the beats array is chronological and must never be reordered, give an A/A/B-valid and A/B/A-forbidden example, and require a fresh shot row/ID for a return cut; the P5 structural repair preserves the exact beat sequence and all authored beat content, clones only the reopened shot's mechanical row under a collision-safe ID, retags only the later run, and accepts only after the full score graph/content post-validator passes
- verify idea: interleave a valid score as shot_01/shot_03/shot_01 while keeping clues before reveal, require byte-identical beat-ID order and content with only the reopened-run shot IDs changed and zero additional LLM calls; force an ID collision and a hidden graph defect to prove deterministic naming and fail-closed behavior; rerun the identical c03 120-word seed through canonical to ledger and OBS
- bible-worthy: yes -- collection completeness does not imply ordered graph topology, and a typed repair can repeat a structurally plausible interleave indefinitely
- confidence: HIGH
- status: PARTIAL IN `09222618` -- the clone/retag projection was correct, but its repair-factory-only placement missed the typed-repair response; see PBUG-20260712-08

## PBUG-20260712-08 -- P5 deterministic repair did not run on the typed-repair response
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `09222618`, prompt `76cb5ca2-0ac7-4b2b-9b64-705b30f0cf75`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 base output again interleaved a closed shot; the repair-prompt factory could not accept the base after projection because another hidden validator defect remained, so it correctly requested typed repair. Gemma's typed-repair response repeated the same interleaving, then went directly to post-validation and failed after 12:33 without ever receiving the safe clone/retag projection
- root cause: deterministic P3/P5 structural normalization lived only inside `repair_prompt_factory`, which runs before the typed-repair model call. `structured_call` validates the typed-repair response directly; it does not call the factory a second time for a schema-valid content failure
- fix: hash each actual raw response for audit first, then run the same narrow P3/P5 projection inside the lane's slot-output wrapper on every ladder attempt. A projected model is serialized back to the ladder only when the complete pass post-validator clears; otherwise the original raw output and its real defect continue through the normal typed-repair path
- verify idea: make a base P5 response contain both a safe topology defect and a separate safety defect so it must reach typed repair; return a safe typed-repair response that still repeats A/B/A; require the per-attempt projection to preserve beat order, split the return shot, complete with exactly one repair model call, and produce resolving ledger boundaries
- bible-worthy: yes -- repair factories are not attempt-wide output middleware, so deterministic repairs placed only there can be bypassed by the response they requested
- confidence: HIGH
- status: FIXED IN CODE / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-09 -- raw P5 projection was not the schema-validated acceptance boundary
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `d024bc18`, prompt `51932200-9d57-499f-aae8-76f6fcf01631`, E4B creative + Mistral technical, 2026-07-12
- symptom: both the P5 base output and its typed repair were schema-shaped BroadcastScores with the same reopened-shot A/B/A defect; the slot-output projection did not accept either response, and the shared ladder failed closed after 12:36 with `beats for each shot must form one contiguous block`
- root cause: the clone/retag projection was still coupled to raw-string collection normalization before `structured_call` had created the strict `BroadcastScore`. That wrapper is useful for wrong-depth collections and nested extras, but it is not the guaranteed acceptance boundary for every schema-valid P5 response. A production response can therefore arrive at post-validation with the safe topology defect intact.
- fix: `P5` now applies the clone/retag projection inside its schema-validated post-validator. Every base, structural retry, and typed-repair response that parses as `BroadcastScore` must cross this hook. It mutates only the accepted in-memory score's mechanical `shots`/`beats` ownership, verifies the complete grounded score again, then runs authored-surface validation. The prompt also asks Gemma to silently scan the final beat sequence and mint a fresh shot row before emitting a return cut.
- verify idea: disable the older raw score normalizer in a mocked runner; a base A/B/A score must still produce a closed ledger with one extra cloned shot and no extra model call. Repeat with a separate safety failure on the base output so typed repair is required; its A/B/A response must clear through the same schema boundary. Run the identical c03 120-word seed to ledger and OBS.
- bible-worthy: yes -- a raw-output middleware hook is not a substitute for the strict typed object boundary where an artifact is actually accepted
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-10 -- Gemma repeated duplicate clue ownership through P5 repair
- promotion: BUG-11.40
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `73de861a`, prompt `00196fd7-943a-4427-90f0-91dce04d4a4b`, E4B creative + Mistral technical, 2026-07-12
- symptom: the new P5 topology guard cleared the reopened-shot defect, exposing the next schema-valid error: a truth-map clue ID appeared in more than one `line_intent`. Gemma's typed repair repeated the duplicate and the run failed closed after 11:35 with `each truth-map clue must be assigned to exactly one line intent`.
- root cause: the safe first-placement-wins duplicate-clue projection still lived in raw-output/repair-factory helpers. Like the prior topology repair, it was not enforced over the strict `BroadcastScore` object that the shared ladder accepts.
- fix: move duplicate clue ownership into the same schema-validated P5 post-validator as reopened-shot topology. It keeps the first authored clue placement in beat order, removes only later duplicate references, reruns the complete grounded score validator, and leaves missing or unknown clues for the LLM repair path. The raw duplicate projection path is removed so tests cannot mistake it for the acceptance guard. Base and repair prompts now require a final exact-once `clue_ids` scan.
- verify idea: disable raw collection cleanup, inject a duplicate clue into a valid base score, and require a no-extra-call accepted ledger. Then add an unrelated forbidden authored phrase so typed repair is required and return a duplicate-clue repair response; require exact-once clues in the persisted accepted BroadcastScore with one repair model call.
- bible-worthy: yes -- ordered first-owner reconciliation is safe only after the full typed graph is available, not as speculative raw JSON cleanup
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-11 -- independent P5 repairs could not compose
- promotion: BUG-11.41
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `145e955b`, prompt `de6f4c1e-b021-4106-871e-8e4a3673bfa4`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5 again returned a reopened-shot topology plus duplicate clue ownership. The topology guard was reached first but declined to apply because its helper demanded that the entire score, including the independent duplicate-clue invariant, already validate. Both the base and typed-repair responses therefore failed closed on the first reported topology error after 12:32.
- root cause: each safe normalizer was implemented as an all-or-nothing full-score repair. A valid artifact containing two independent, non-authoritative mechanical defects could not reach either repair's success path; the post-validator handled only the first reported defect rather than a bounded composition of disjoint projections.
- fix: split each P5 helper into a narrow projector and a full-validation wrapper. At the typed `BroadcastScore` acceptance boundary, apply at most the two proven-safe projections in deterministic order (reopened shot ownership, then duplicate clue ownership), preserve all authored prose/beat order/first clue placements, and run the complete grounded score validator only after the bounded composition. Any remaining or ambiguous defect remains a normal LLM failure.
- verify idea: create one base score and one typed-repair score with both A/B/A topology and a later duplicate clue, plus an unrelated forbidden phrase on the base so the typed call is mandatory. Disable raw cleanup and require the accepted score to retain beat order, mint the collision-safe return shot, keep first clue ownership, remove only the later duplicate, and use exactly one repair model call.
- bible-worthy: yes -- independently safe deterministic transformations must compose before a global validator can judge their shared result
- confidence: HIGH
- status: FIXED IN CODE / FULL SUITE + BUG BIBLE GREEN / AWAITING SAME-SEED LIVE 120-WORD C03 REQUALIFICATION

## PBUG-20260712-12 -- full-score repair overflowed for a one-intent grounding omission
- promotion: BUG-11.42
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `ef6cd277`, prompt `cabf66f8-14d1-4de5-b043-d329b888df78`, E4B creative + Mistral technical, 2026-07-12
- symptom: P5's typed structural guards cleared both reopened-shot topology and duplicate-clue ownership, but one non-announcer clue intent omitted the exact lost-object anchor `parcel tag`. The generic typed repair then attempted to regenerate the entire BroadcastScore, produced no decodable top-level JSON, repeated the same overflow on syntax retry, and failed closed after 14:39.
- root cause: the acceptance boundary treated a localized LLM-owned semantic omission as a whole-artifact repair. A complete score is too large and too fragile an output shape for a one-line intent correction, especially after the failed score and contract inputs are fed back through the repair ladder.
- fix: accept the P5 score after structural and safety validation, then immediately derive a bounded eligible-beat plan from the immutable grounding contract. Call a separate `ScoreIntentPatch` seam with only `{beat_id, current_intent, required_anchors}` targets. The LLM authors each replacement intent; Python accepts only one replacement for every and only planned beat, verifies literal anchors, merges no other field, and reruns the full grounded-score and authored-surface contracts. Prompt-pack and pipeline declarations make the tool auditable.
- verify idea: remove `stamp` from an otherwise valid P5 clue intent, require one nine-call runner where the sixth call is a `ScoreIntentPatch`, persist only the LLM-provided replacement intent, and reject both a missing literal anchor and an unplanned beat ID. Run the full suite and Bug Bible, then repeat the same c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- when a semantic defect is confined to an explicitly owned leaf, a whole-document repair is an avoidable reliability and context-window hazard. Create a small typed patch artifact, validate its exact scope, and retain full-artifact validation as the authority.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P5 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-13 -- grounding-intent patch could erase an already-valid anchor
- promotion: BUG-11.43
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `ec489787`, prompt `54c0a8bb-45f9-4cf2-bc56-49882fd16377`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P5 cleared on their first attempts after P5 safely normalized reopened-shot topology and duplicate-clue ownership. The new `ScoreIntentPatch` first corrected four lost-object/device targets but omitted beat_09's required resolution anchor; its typed retry passed the patch-local anchor check yet the merged score still failed closed after 578.73 seconds.
- root cause: the patch plan listed only newly missing anchors. A selected target beat can already hold a different immutable anchor required elsewhere (especially reveal/closure beats that also carry clues); overwriting its complete `line_intent.intent` could silently remove that existing anchor. The patch post-validator checked local target coverage but did not validate the merged BroadcastScore before accepting the typed patch.
- fix: every selected target now carries forward every immutable anchor already present in its current intent, in addition to newly missing anchors. The patch's `structured_call` post-validator now applies the candidate in memory and rejects it unless the complete score grounding and authored-surface contracts clear; a typed repair receives that exact merged-contract error. The repair seam explicitly forbids visual direction, camera/scene/shot instructions, stage business, dialogue, and production metadata.
- verify idea: make a reveal beat carry the `stamp` clue and its already-valid `grille` device anchor while omitting `stamp`; require the plan to demand both literals, accept only a patch preserving both, and reject a patch that carries a banned phrase even when its anchors are complete. Run full suite, Bug Bible, and the same c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- a narrow patch must preserve every currently valid invariant in the field it replaces, not merely add the invariant that triggered repair. Patch-local schema acceptance is insufficient; validate the merged canonical artifact at the same structured-call boundary.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P5 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-14 -- full-script repair repeated one missing closure literal
- promotion: BUG-11.42
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `a9cf3bbe`, prompt `69a56fa7-afd3-4e72-b2b4-188a7afaac00`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P5 cleared, including the bounded P5 grounding patch. P6 generated a structurally valid `PerformanceScript` but its closure line did not speak the exact immutable resolution anchor `returns everything`. The generic typed repair regenerated the whole script and repeated the identical omission, failing closed after 11:22.
- root cause: P6 treated a localized LLM-owned spoken-line grounding omission as a full-script repair. The prompt already named the literal, but the full artifact request was large enough that Gemma reproduced the otherwise-valid script and its one missing phrase instead of isolating the closure line.
- fix: P6 now accepts a structurally and safety-valid script, then derives a bounded `ScriptLinePatch` target only when full grounding fails. The LLM receives each affected `{line_id, current_text, required_anchors}` target and authors only replacement spoken text. Python requires exact line coverage and literals, preserves all immutable anchors already spoken on a targeted line, merges no other field, and validates the complete graph/text/grounding contract before accepting the patch. The new source-pack seam forbids labels, stage/camera/visual direction, production metadata, and wrappers.
- verify idea: remove the exact closure anchor from an otherwise valid P6 script, require a single extra `ScriptLinePatch` call that repairs only `line_005`, and preserve all other lines byte-for-byte. Also make a reveal line carry an object clue while already speaking the device anchor; require its patch plan and accepted replacement to retain both literals, then reject an otherwise anchored banned-phrase replacement. Run full suite, Bug Bible, and the identical c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- the localized semantic-repair law applies separately to each artifact boundary. A complete script is no more suitable than a complete score for correcting one owned leaf.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P6 grounding patch cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-15 -- later full-script retakes bypassed the bounded P6 grounding repair
- promotion: BUG-11.44
- surfaced: deterministic canonical 120-word `original_codex56sol` c03 requalification after `cb5166f8`, prompt `10438a88-66c6-400d-b7b9-d049b2f116f3`, E4B creative + Mistral technical, 2026-07-12
- symptom: P1-P6 cleared, including both bounded P5 and P6 grounding patches. The blind-listener loop requested P8; P8 returned a structurally valid replacement script but again omitted exact closure anchor `returns everything`. Its generic full-script typed repair repeated the omission and failed closed.
- root cause: the P6 local-line guard was attached only to initial script creation. P8, optional P8, and P9 retake paths continued to validate full grounding inside their full-script `structured_call`, so a later reauthoring route could reintroduce the same local defect and bypass the guarded acceptance boundary.
- fix: factor one `_call_grounded_script` acceptance path for every complete-script authoring call. It accepts only structural/safety-valid scripts, invokes a bounded `ScriptLinePatch` when and only when full grounding fails, validates the merged script before acceptance, and records a pass-specific patch journal entry such as `P8_grounding_patch`. P8 and P9 pipeline registry entries now declare the patch seam, keeping the dynamic repair route visible in the source-pack contract.
- verify idea: force P7 to request P8, make P8 omit the closure anchor, require a single P8 line patch and a clean blind-listener rerun before P9. Assert P8/P9 pipeline seam references include `codex56_script_anchor_patch`; run full suite, Bug Bible, and the identical c03 120-word Gemma/Mistral canonical smoke.
- bible-worthy: yes -- a validation/repair guarantee must cover every reauthoring route for an artifact, not only its first construction. Factor the guarded boundary rather than duplicating a one-off call-site fix.
- confidence: HIGH
- status: LIVE-QUALIFIED end to end by same-seed c03 prompt `ed1a13ca-6cc5-4a79-830e-cc82c8a460ab`: P8 grounding patch cleared, listener rerun and P9 cleared, frozen ledger and final OBS asset exist, ComfyUI `RESULT SUCCESS`

## PBUG-20260712-16 -- detached soak monitor reported a blank exit code after canonical success
- promotion: BUG-12.50
- surfaced: same-seed c03 120-word live qualification, monitor log `logs/codex56_c03_120_after_156cb2e4`, 2026-07-12
- symptom: the detached PowerShell wrapper wrote `COMPLETE: SOAK_FAIL rc=` even though the canonical API reported `RESULT SUCCESS`, the final OBS MP4 was published, duration and byte-identical-audio checks passed, and the frozen ledger existed.
- root cause: the monitor trusted a blank `Process.ExitCode` from the detached PowerShell child as a failure without reconciling the canonical API's explicit terminal result.
- fix: the monitor now treats an empty child exit code plus a `RESULT SUCCESS` marker in the canonical runner log as success; a real nonzero code or missing success marker remains a fail.
- verify idea: the next detached canonical smoke with a successful API result must write `COMPLETE: PASS rc=0`; an absent success marker must remain a failure.
- bible-worthy: yes -- test orchestration must not turn an observed final asset plus explicit canonical success into a false-negative qualification verdict.
- confidence: HIGH
- status: FIXED IN HARNESS / AWAITING NEXT DETACHED-SMOKE CONFIRMATION

## HISTORICAL BACKFILL -- 2026-07-12 production-only Bug Log sweep

The archived `BUG_LOG.md` and `BUG_LOG_2026-06.md` contain many local labels,
including design notes, test-only findings, unresolved investigations, and
operator-pending visual observations. This backfill admits only historical
records with explicit live/published/GPU evidence, a grounded root fix, and a
current regression test. It does not promote an archived label merely because
its name contains `BUG`.

## PBUG-20260614-01 -- malformed post-blend filter silently dropped scopes and captions
- promotion: BUG-08.08
- surfaced: live look-QA of a published episode, 2026-06-14; server log recorded
  ffmpeg `gbrpformat` rejection and source-copy fallback while `obs_publish OK`
- symptom: a three-input procgen blend silently published without burned SDH
  captions or audio-reactive scopes
- root cause: an enabled green-overlay chain already ended in `,format=gbrp` and
  the next chain appended `format=gbrp` without a separator, producing the
  invalid token `gbrpformat`
- fix: commit `99320ae` adds the pixel-format pin exactly once; current
  `test_build_cmd_3input_scopes_no_double_format_gbrp_bug402` covers both
  overlay states and the caption burn
- verify idea: every enabled three-input filter combination has valid token
  separators, expected pixel-format pins, and its required visual layers
- bible-worthy: yes -- process success is not evidence that optional final
  compositing effects survived
- status: PROMOTED BUG-08.08

## PBUG-20260614-02 -- post-composition shortest input clipped the rolling-credits tail
- promotion: BUG-08.08
- surfaced: operator-verified fresh render, 2026-06-14; credits were absent from
  the published tail before the fix and visibly restored after it
- symptom: the final video stopped at the shortest upstream track, cutting a
  deliberately longer floor/HUD credits layer
- root cause: post-composition treated a short scopes track as the completion
  boundary despite the credits layer intentionally extending past master audio
- fix: preserve the intended long-form timeline; current
  `test_blend_cmd_does_NOT_use_shortest_for_c7_safety` guards the command
- verify idea: a credits/scopes fixture with a deliberately longer tail retains
  the complete post-roll in the final composition
- bible-worthy: yes -- final-output success must include duration and layer
  completeness, not merely an ffmpeg exit status
- status: PROMOTED BUG-08.08

## PBUG-20260626-01 -- LTX-AV activation spill caused a no-OOM multi-minute crawl
- promotion: BUG-07.22
- surfaced: GPU-validated live 30-word all-`ltx_audio_in` headless run,
  2026-06-26; 223 s/iteration spill reduced to steady roughly 11 s/iteration
- symptom: audio-conditioned video inference avoided OOM but fell into system
  memory spill with near-zero free VRAM and an extreme per-beat slowdown
- root cause: one VideoVAE stayed alive through both encode and decode, while
  no activation reserve protected the sampler from desktop VRAM contention
- fix: `ae8ec55e` splits encode/decode VAE lifetime; `bd5ffd23` scopes an
  `EXTRA_RESERVED_VRAM` minimum and restores it after the run
- verify idea: graph wiring has distinct VAE nodes; reserve scope raises,
  restores on exception, and never lowers a stricter existing reserve; GPU soak
  remains free of system-memory crawl
- bible-worthy: yes -- a slow no-OOM render is a real VRAM failure class, not a
  license to guess at quantization or offload changes
- status: PROMOTED BUG-07.22

## PBUG-20260702-02 -- orphaned one-shot environment hook poisoned later headless boots
- promotion: BUG-12.52
- surfaced: live all-`ltx_audio_in` probe, 2026-07-02; the report instead showed
  every shot rendered by HuMo from a crashed leg's stale force-engine override
- symptom: a canonically configured run silently inherited file-based engine
  overrides that were not present in the explicit new-run configuration
- root cause: normal post-leg cleanup did not run after a crash, leaving the
  sourceable environment hook to affect later boots
- fix: consume-once hook semantics plus canonical-wrapper stale-hook removal;
  `test_headless_wrapper_clears_stale_extra_env_hook_before_boot` pins the
  cleanup boundary
- verify idea: seed an override hook, run the canonical headless wrapper, then
  require hook removal and an engine receipt matching only explicit inputs
- bible-worthy: yes -- temporary file-based overrides must not become hidden
  persistent process defaults
- status: PROMOTED BUG-12.52

## BUG AUDIT RECEIPT -- 2026-07-12

Searched every repository filename containing `bug`, both historical bug logs,
all current PBUG entries, and bug-labelled commits. Promoted the July-11
canonical-smoke set plus the four historical incidents above only after locating
their real-run evidence and current regressions. Kept the unresolved July-2
VRAM diagnosis, the environmental Ollama outage, and the predicted (not yet
live) 720-word context risk out of the Bible; other archived local labels stay
out until they independently meet the same production-only admission rule.

## PBUG-20260712-17 -- Codex56 P6 grounding patch exhausted both live attempts
- surfaced: canonical 120-word `original_codex56sol` queue leg, prompt `e256be3f-69a0-495f-8a99-3bf9c06e01a8`, Gemma E4B creative + Mistral-Nemo technical, 2026-07-12
- symptom: the canonical API returned `RESULT FAIL`; node 1 stopped at `P6_grounding_patch` after two structured-call attempts, before ledger/media/OBS completion
- root cause: OPEN -- the queue wrapper preserved only the truncated terminal exception, not the exact messages, raw response, projection, and validator error needed to distinguish model omission from repair-contract or context failure
- fix: none yet; first reproduce or inspect the retained attempt artifacts after the code-ready Codex56 telemetry seam lands, then fix the owning representation/validator boundary rather than increasing retries
- verify idea: run the same 120-word model pairing with attempt telemetry; require the failing rung's exact raw/projected/error record, then add a focused regression for the isolated cause and a canonical rerun proving ledger, episode asset, `obs_publish OK`, and final OBS file
- bible-worthy: pending -- live admission is proved, but no reusable rule exists until the root cause and fix are known
- status: SUPERSEDED 2026-07-15 (baseline) -- the target lane
  `original_codex56sol` was ripped from the roster @ `3312aec7`, so the failure
  cannot recur as logged, and the code-ready telemetry seam this fix was gated
  on was retired with it. The diagnostic gap it names (no retained
  raw/projected/error attempt record) is carried forward as an engineering risk
  on the GO_FORWARD context/cap item, re-targetable at any surviving
  structured-call lane. Not Bible-eligible from this record.

## PBUG-20260712-18 -- Sci-Fi Codex P3 repair envelope rejected as the artifact root
- surfaced: canonical 120-word `scifi_codex` queue leg, prompt
  `cc9e0f8a-2a20-40a1-b5dc-da2fc8a400d6`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: `RESULT FAIL` at P3 after two structured attempts; the repaired
  `RadioScoreV4` was complete but nested under `resolved_artifact`, so strict
  validation reported every required root field missing and the wrapper extra
- root cause: the lane passed an exact single-key typed-repair transport
  envelope directly into the requested strict artifact schema
- fix: normalize only the exact `{"resolved_artifact": <object>}` transport
  shape at the Sci-Fi Codex response boundary, preserve original-wire hash and
  length, journal the normalization boolean, and keep mixed/non-object roots
  fail-loud
- verify idea: exact-wrapper, direct-root, mixed-root, non-object, original-wire
  telemetry, and prompt-seam exclusion regressions; then rerun the same canonical
  bank and require RESULT SUCCESS plus ledger and OBS final existence
- bible-worthy: pending -- live admission and reusable exact-envelope rule are
  proved; promote only through the standing Bug Bible fan-out
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-19 -- all-visualizer policy still invoked upstream image authoring
- surfaced: canonical 120-word `scifi_codex` queue leg, prompt
  `e5ded258-1f3d-4a6e-874a-ba89ce1e6a83`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: `RESULT FAIL` at node 89 `OTR_MetaBriefImagePromptGen`; the canonical
  all-visualizer policy (`viz_mxc_cpu`, `viz_mxc_mandala`, `viz_camera`) still
  resolved and used the writer visual-prompt path, and c03 failed the
  story-consistency gate even though no downstream video role consumed an init
  image. No OBS final was produced.
- root cause: effective-engine / `accepts_still` capability was checked only at
  downstream image dispatch. MetaBrief and ShotLock entered visual-authoring
  paths before that guard, so a proven no-consumer policy could still spend or
  fail in an upstream writer call.
- fix: make dispatcher-owned effective per-role still capability (including
  force-map and radio redirects) the shared authority. A complete all-false
  map returns an explicit empty v1 payload and bypasses MetaBrief/ShotLock
  writer resolution; mixed policy omits only roles proven procedural and keeps
  unknown roles conservative upstream. The dispatcher renders only roles
  proven to consume an init image and fails loudly for an unproven object role.
- verify idea: `test_roles_requiring_stills_needs_a_complete_resolvable_policy`,
  `test_meta_brief_all_visualizers_bypass_prompt_authoring`,
  `test_meta_brief_node_bypasses_before_writer_resolution`,
  `test_meta_brief_mixed_policy_authors_only_proven_consumer_roles`,
  `test_dispatcher_refuses_image_render_without_proven_consumer`,
  `test_dispatcher_preserves_proven_role_when_another_slot_is_unresolved`,
  `test_dispatcher_rejects_explicit_unknown_object_role`,
  `test_dispatch_skips_stills_for_all_visualizer_episode`, and
  `test_shotlock_all_visualizers_skip_writer_visual_directives`; then rerun the
  canonical bank and require RESULT SUCCESS, no image objects or visual-writer
  call, ledger, episode asset, `obs_publish OK`, and OBS final.
- bible-worthy: yes -- live failure plus reusable effective-consumer-capability
  contract and executable coverage
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-20 -- Sci-Fi Codex P3 typed repair silently lost its contract

- promotion: BUG-11.50
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `ffc354cc-febf-4ada-9ebd-2e3d27a057e8`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: the base P3 `RadioScoreV4` had five music cues (maximum three),
  then its typed repair logged `PROMPT_GUARD: Truncated 5273 -> 4592`
  (`context_cap=8192`, `max_new_tokens=3600`) and returned the request-shaped
  root `{artifact_inputs, validation_error}`. Strict validation correctly
  rejected the envelope; no ledger, media asset, or OBS final was produced.
- root cause: P3/P3_rewrite reserved a flat 3,600 output tokens without
  accounting for the fixed 8,192-token local context or the full failed-score
  repair prompt. The generic repair payload duplicated the original request,
  so the token wrapper left-truncated its leading system/schema/rules before
  calling Gemma. The model did not receive the contract it was expected to
  repair and echoed trailing input material instead.
- fix: calculate the RadioScoreV4 reservation from requested words and locked
  beat count; at the observed 120-word/12-beat case it reserves 2,800 tokens,
  leaving 5,392 input tokens. Mark P3 and P3_rewrite as `prompt_must_fit` so
  a future oversize graph fails before generation, and send P3 repair context
  as compact tagged references (failed score, rejection, locked graph,
  advisory; plus accepted score/review only for P3_rewrite) rather than a
  copyable JSON request envelope.
- verify idea: assert `8192 - radio_score_output_token_budget(120, 12) >=
  5273`, assert P3/P3_rewrite both use that dynamic budget and
  `prompt_must_fit=True`, assert the P3 repair prompt carries no
  `original_request`/`artifact_inputs` JSON envelope, then rerun the same
  canonical bank and require P3 clearance, zero all-visualizer image objects,
  saved ledger, episode asset, `obs_publish OK`, and final OBS file.
- bible-worthy: already promoted as BUG-11.50; added OTR executable coverage
  for the repair-prompt dimension.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-21 -- Sci-Fi source P0 could exhaust its bounded output before producing JSON

- promotion: BUG-11.50
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `b5341847-4635-4eeb-a5b8-4660136b0d78`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 selected a valid long RSS source, then both base/structural
  attempts ended at `generated_tokens=2000` with incomplete JSON
  (`prompt_tokens=2455`, `max_new_tokens=2000`). Its typed repair returned an
  otherwise literal index with `tone: []`, which strict validation correctly
  rejected because `tone` is model-owned scalar prose. No P1/P3 artifact,
  ledger, media asset, or OBS final was produced.
- root cause: FactIndexV4/FragmentDossierV4 allowed up to twelve facts,
  entities, and numbers while claims, quote spans, numeric-token lists, and
  several strings had no finite serialized surface. A fixed P0 output ceiling
  could therefore be too small by construction. The generic typed repair also
  replayed a copyable original-request envelope and did not explicitly require
  scalar `tone`, recreating the context/shape failure class from PBUG-20.
- fix: introduce one shared Sci-Fi P0 evidence contract for Codex, Gemini, and
  Sonnet: 1-6 facts, 0-4 entities/numbers, one literal span per fact/entity,
  bounded claim/name/quote/token/tone fields, and compact story-usable prompt
  seams. Reserve 2,800 output tokens for FactIndexV4 and 3,000 for Sonnet's two
  extra root strings, journal the bounds/source-size receipt, and retain
  `prompt_must_fit=True`. P0 repairs now receive tagged failed-artifact,
  rejection, source evidence, digest, and allowed-field references only; they
  explicitly require the exact artifact root and one nonempty scalar `tone`.
  Python never substitutes a tone value.
- verify idea: reject seven facts, a second evidence span, an overlong quote,
  and `tone: []`; assert all three source lanes use the shared bounded
  reservation and compact repair context without `original_request` or
  `artifact_inputs`; then rerun the same canonical Codex bank through P3 and
  require zero all-visualizer image objects, saved ledger, episode asset,
  `obs_publish OK`, and OBS final.
- bible-worthy: yes -- a live bounded-output failure with reusable
  model-facing artifact-surface and compact-repair requirements; promoted as
  BUG-11.50 with cross-lane executable coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260712-22 -- Sci-Fi Codex P3 whole-score transport exhausted its model window

- promotion: BUG-11.50 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `f26b727b-42c8-40d6-b3ee-001d7a869cf9`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: the initial bounded direct-score correction was rerun live at
  prompt `edbbac48-9aa8-4907-8086-f63134604604` (same Gemma E4B creative +
  Mistral-Nemo technical pairing). P0-P2 cleared, but P3 again produced no
  decodable top-level JSON on its 2,900-token base, lower-temperature, and
  typed-repair calls. The canonical queue ended `RESULT FAIL` before ledger,
  episode, image/still, or OBS work.
- root cause: finite `RadioScoreV4` bounds removed the original unbounded
  schema defect, but the model still had to serialize duplicate mechanical
  graph state it did not author: advisory rows, scene/shot/beat/line IDs,
  parents, order, speakers, roles, and canonical cue anchors. The direct
  whole-score transport remained too wide for the live model even with a
  2,900-token cap; increasing the cap alone would recreate repair-window risk.
- fix: replace direct P3/P3-rewrite score emission with bounded
  `RadioScoreDraftV4` plus a fail-closed compiler. The model authors only
  creative surface, local shot/cast/line-count/fact/cue choices; Python derives
  only uniquely determined mechanics from accepted P0/P2/advisory state and
  revalidates fresh `RadioScoreV4`. The three-call ladder restarts from trusted
  context after two decode failures and uses minified parsed semantic repair
  only for complete invalid drafts. Exact wrapped-root handling remains. The
  real Gemma tokenizer measured a max-width draft at 1,418 output tokens;
  reservation is 1,647 (`+ max(128, 15%) + 16`). Measured base, clean restart,
  semantic repair, rewrite base, and rewrite repair prompts were respectively
  1,110, 1,167, 2,664, 2,614, and 4,165 tokens, all within 8,192 with the new
  reservation.
- verify idea: compiler tests reject dynamic advisory/count/shot/cast/fact/cue
  defects and preserve `compile(project(score))` rewrite structure; actual
  tokenizer tests cover all six envelopes and require prompt plus reservation
  <=8,192; default schema injection remains unchanged for other passes; then
  rerun the canonical 120-word bank and require all-visualizer zero image
  objects, saved ledger, episode asset, `obs_publish OK`, and final OBS file.
- bible-worthy: yes -- BUG-11.50 now explicitly permits a compact
  authoring-draft/compiler boundary when it removes deterministic graph
  serialization rather than merely papering over absent bounds.
- status: ROOT REPLACEMENT IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-23 -- Sci-Fi P0 generic string clamp stranded an exact oversized source span

- promotion: BUG-11.50 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `81e0b0c9-2f20-4085-9fd0-e7f8034f75da`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 selected an eligible RSS source, but its first fact returned an
  exact literal `full_text` quote wider than the 240-character P0 source-span
  cap. The generic tolerant validator word-clamped the quote while retaining
  the model's old end coordinate; the literal-span validator correctly rejected
  the synthetic mismatch. The typed repair repeated the same oversized literal,
  so the canonical queue ended `RESULT FAIL` before P1/P3, ledger finalization,
  media, or OBS work.
- root cause: `repair_literal_source_metadata` safely reindexed exact quoted
  source text, but it first required the raw artifact to satisfy Pydantic's
  quote cap. It therefore could not repair an exact source quote that exceeded
  that cap. Meanwhile P0 still used the generic string clamp intended for
  compatibility fields, which may shorten source metadata at a word boundary
  without recomputing `end`. The shared P0 helper shape made the defect possible
  in Codex, Gemini, and Sonnet.
- fix: disable generic overlong-string clamping at all three Sci-Fi P0
  boundaries. Extend the shared metadata-only repair so it accepts an oversized
  quote only after proving the *entire raw quote* occurs literally in one legal
  source field, rehomes/reindexes it under the existing ambiguity rules, then
  replaces only the quote with that coordinate's exact finite source prefix and
  recomputes `end`. Claims, tone, and all nonliteral/ambiguous text remain
  model-owned and fail through the bounded typed-repair ladder.
- verify idea: exact oversized source quote repairs in one P0 call to the
  schema cap with byte-identical claim; an oversized quote with invented text is
  rejected; all three Sci-Fi P0 call sites disable generic clamping. Run focused,
  full, and Bug Bible gates, then rerun the same canonical Codex leg through
  P3/ledger/OBS proof.
- bible-worthy: yes -- repeatable bounded-source-metadata repair class at the
  shared Sci-Fi P0 fan-out; promoted as an executable BUG-11.50 extension.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-21 -- P1 repeated an overlong Aion dramatic question

- renumbered: 2026-07-15 baseline -- originally logged as PBUG-20260713-10,
  colliding with the P9-audit entry below (which keeps -10: it is the id cited
  by the contract-gap docs and commit `3a98a6f1`). BUG_BIBLE.yaml currently
  carries two `legacy_id: PBUG-20260713-10` rows (~:4357/:4379); at the next
  operator fan-out, re-point the BUG-11.54 row's legacy_id to PBUG-20260713-21,
  and verify the acronym-union rule (~:4357, also citing -10) against its true
  source entries (the acronym PBUGs -07/-09) -- the P9-audit entry that owns
  -10 is not an acronym bug.
- promotion: BUG-11.54
- surfaced: canonical 120-word `scifi_codex` smoke, prompt
  `2147f181-8821-461f-a5dc-8cb9bfefd48c`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P1 returned a question above the 160-character schema cap; the
  typed repair repeated the invalid field and exhausted the ladder before P2.
- root cause: the repair prompt described the cap but relied on a second model
  call to shorten authored text, so a reasoning model could copy the rejected
  question unchanged.
- fix: add a deterministic, word-boundary P1 repair only for overlong question
  or consequence fields, preserving the established semantic repair turn for
  ending-only overflow and rejecting malformed roots.
- verify idea: unit-test bounded shortening and run the canonical 120-word
  combination through P5, ledger, episode, `obs_publish OK`, and final OBS
  existence.
- bible-worthy: yes -- bounded typed repair must not depend on a model obeying
  a repeated length instruction; promoted as executable BUG-11.54.
- status: FIXED IN TREE; 120-WORD LIVE PASS

## PBUG-20260712-24 -- Sci-Fi Codex P3 compact draft omitted nested literal semantics

- promotion: BUG-11.38 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `fab1bbbe-cfc1-484b-8f5b-61dfc296de6e`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 repaired and P1/P2 cleared, but P3's base compact draft emitted
  numeric `arc_phase` values copied from advisory word centers and invented
  descriptive cue IDs (`TensionBuild`, `EquityStrain`, `DecisionPoint`). Its
  typed repair reduced unrelated overlength errors but repeated those seven
  invalid nested values, so the canonical queue ended `RESULT FAIL` before
  P4/P5, ledger finalization, media, or OBS work.
- root cause: P3 correctly omitted the large full Pydantic schema to preserve
  its measured 8,192-token repair window, but the compact model-facing contract
  named `arc_phase` and `cue_id` only by field and length. It did not preserve
  their nested literal/type semantics. The local model therefore treated
  `arc_phase` as a word-band number and treated `cue_id` as a creative title;
  the same incomplete surface was reused for typed repair.
- fix: make the shared compact P3/P3-rewrite base and repair contract state that
  `arc_phase` is a short narrative JSON string, never a number/word count/center
  or percentage, and enumerate `music_open`, `music_inter`, and `music_close`
  as the only cue IDs. Creative cue naming stays in `description`. No Python
  normalization is permitted: arc labels and cue choice remain model-authored.
- verify idea: drive the live failure shape (numeric arc plus descriptive cue
  ID) through base then typed repair and assert the accepted score returns only
  after the repair sees both literal rules; retain actual-tokenizer fit tests
  for base/restart/semantic-repair/rewrite envelopes. Then rerun the same
  canonical Codex leg through P3/ledger/OBS proof.
- bible-worthy: yes -- model-visible compact schemas must retain nested type and
  literal semantics, not merely field names and maximum lengths; promoted as an
  executable BUG-11.38 extension.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-25 -- Sci-Fi Codex P3 full typed repair repeated local prose overflow

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify queue leg, prompt
  `4b19f3ed-bd28-4f84-9b81-5fcddfb89dc0`, Gemma E4B creative + Mistral-Nemo
  technical, 2026-07-12
- symptom: P0 repaired and P1/P2 cleared. P3 then returned a complete compact
  draft whose only surfaced defects were four model-authored strings over their
  finite caps. Its normal full typed repair shortened one field but repeated
  three over-cap fields, exhausting the bounded ladder before P4, ledger,
  media, or OBS work.
- root cause: generic clamping was correctly disabled at the author-owned P3
  boundary, but the only remaining repair transport resent the complete draft.
  That invited the local creative model to reauthor already-valid graph and
  prose surface instead of making the one bounded shortening decision. Pydantic
  length errors could also conceal a compiler-only defect, so a naive text patch
  would have incorrectly treated every string-only error report as local. A
  lazy scheduler wrapper also hid remote-provider markers, which would have
  let a remote slot take the local-only route; generic completion reporting
  could then mislabel a rejected direct patch as a decoded accepted draft.
- fix: on local P3/P3-rewrite only, derive a maximum-six exact whitelist of
  over-cap authored leaves from the real Pydantic locations; preflight a clone
  through the strict draft/compiler/signature/graph boundary; then request one
  strict one-for-one author-owned shortening patch at the common typed-repair
  temperature. Merge only through trusted locations, revalidate the complete
  draft, and record the real patch call in the existing P3 receipt. Unknown,
  broad, hidden-graph, malformed, or still-over-cap repairs fail closed. Remote
  OpenRouter slots retain the existing same-slot full repair because their
  virtual context metadata is not an exact tokenizer preflight; no model/router
  fallback or substitution is introduced. The scheduler carries exact catalog
  transport capability into its lazy closure and relays OpenRouter JSON-object
  mode; direct-patch receipts own their parse/schema/contract truth so the
  generic ladder cannot overwrite it with an empty factory result.
- verify idea: cover every eligible P3 leaf, mixed/broad errors, hidden compiler
  defects, malformed patch roots, unselected-field preservation, local base and
  rewrite receipt success at `.10`/512, a scheduler-wrapped remote same-slot
  JSON-mode full repair, truthful malformed-patch receipts, and actual Gemma
  E4B six-target prompt plus reservation fit. Run focused,
  full, Bug Bible, pack/registry, and canonical workflow gates, then rerun the
  fresh canonical Codex leg through ledger/OBS proof.
- bible-worthy: yes -- a live bounded-patch admission: localized authored prose
  needs one-for-one model replacement plus complete preflight/merged validation,
  never Python clipping or a broad retake. Promoted as executable BUG-11.42
  extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260712-26 -- Strict Sci-Fi RSS admission starved eligible inline bodies

- surfaced: canonical 120-word `scifi_codex` GUI/API run, prompt
  `59b9baa5-046f-4e4c-b313-8d18223ea716`, 2026-07-12
- symptom: the live feed pool contained qualifying literal inline RSS bodies,
  but the strict selector body-resolved only the first ten headline-ranked
  candidates. All ten fell back to thin summaries, so the writer failed before
  P0 with `No science RSS candidate met the v4 source floor`.
- root cause: the selector enforced the 400-character/80-word/12-unique-token
  floor only after its bounded body-fetch slice. It had no eligibility-aware
  ordering, and its legacy `rss_full > 300` shortcut did not match the stricter
  Codex/Gemini/Sonnet envelope contract.
- fix: define one stdlib-only v4 RSS predicate and route strict selection plus
  all three lane envelopes through it. In strict mode, stable-partition already
  qualified inline RSS bodies ahead of unresolved candidates while preserving
  prior rank inside each partition; admit inline text only through the shared
  predicate, retain the ten-candidate cap and URL-scrape path, and leave legacy
  non-strict behavior unchanged.
- verification: focused admission coverage passed; full Windows suite
  `7843 passed, 31 skipped, 1 xfailed`; Bug Bible `17 passed, 12 skipped,
  3 xfailed`. Canonical prompt `14af0787-f45c-4caa-8737-92d057855653`
  logged `Strict v4 admission prioritized 13/40`, resolved ten bodies with
  `10/10 candidate(s) passed content floor`, selected a 4,825-character MIT
  article, and crossed P0 into P1. A separate OpenRouter reasoning-capability
  error then stopped the episode; it does not reopen source admission.
- bible-worthy: yes -- a bounded selector must apply hard downstream
  eligibility before its truncating candidate slice and share the exact
  predicate with the accepting envelope. Promotion remains a separate review.
- status: FIXED AND LIVE VERIFIED

## PBUG-20260713-01 -- OpenRouter global reasoning-off rejected by mandatory endpoint

- promotion: BUG-12.53
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `14af0787-f45c-4caa-8737-92d057855653`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-12/13
- symptom: strict RSS admission and P0 cleared, then P1 stopped before creative
  generation with HTTP 400: `Reasoning is mandatory for this endpoint and
  cannot be disabled.` The process-wide `OPENROUTER_REASONING_EFFORT=none`
  had been sent unchanged to `aion-labs/aion-3.0-mini`.
- root cause: the OpenRouter cache discarded the live `/models` reasoning
  contract, and request construction applied one global effort to every model.
  A saved slug absent from the stale June cache also had no bounded way to learn
  the provider's precise mandatory-capability response.
- fix: retain sanitized per-model reasoning metadata in catalog schema v2 and
  resolve the global setting against the selected model. A mandatory model uses
  its lowest declared enabled effort (or `low` when the catalog omits effort
  levels), while ordinary models retain explicit `none`. For stale/cold cache
  only, the exact mandatory-reasoning 400 triggers one same-model corrected
  call, remembers the capability for the process, and does not consume the
  transient retry budget; every other 400 remains fail-fast.
- verify idea: prove proactive metadata resolution, exact-400 learning with
  zero transient retries, subsequent-call reuse, unchanged ordinary-model
  `none`, generic-400 fail-fast, and live catalog retention of Aion's
  `mandatory: true`; then rerun the same canonical 120-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: promoted as BUG-12.53 with executable OTR coverage and shared
  Bug Bible regression pins.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-02 -- Remote P3 whole-draft repair repeated ten local prose overflows

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `b98bef70-d5ae-4c60-9402-ce3adeccf26e`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: RSS admission, P0, P1, and P2 cleared. P3 returned an otherwise
  complete compact draft with ten `string_too_long` authored fields. The normal
  remote whole-draft typed repair fixed only two, repeated eight, and exhausted
  the ladder before ledger/media/OBS work.
- root cause: PBUG-20260712-25's one-for-one authored-text repair was restricted
  to exact-tokenizer local slots and six targets. OpenRouter already sent full
  messages or failed loudly, but its explicitly known transport was still forced
  through the broad retake. The live ten-target shape also exceeded the local
  patch schema, so merely enabling the remote marker would have remained dead.
- fix: declare behavioral patch transport explicitly on the lazy scheduler:
  exact-tokenizer local, full-message/fail-loud OpenRouter, or ineligible.
  Expand the one-call patch envelope to 12 targets/1024 output tokens, prove its
  actual tokenizer envelope, preserve complete preflight and merged validation,
  and record the chosen transport. OpenRouter honors the patch's strict output
  cap even when its global minimum-output floor is raised; JSON mode, mandatory
  reasoning, cost guard, routing, retries, and terminal provider errors remain
  in the shared backend. Thirteen-plus or any mixed/hidden/unproven shape keeps
  whole-draft repair.
- verify idea: reproduce the ten-target artifact with an explicit OpenRouter
  callable and require exactly base plus one patch, all ten exact paths,
  json_object mode, 1024 tokens, full-message transport receipt, and complete
  draft acceptance. Prove 12-row prompt/response fit, 13 targets retain broad
  repair, excluded/unmarked transports remain ineligible, and a raised global
  remote floor cannot inflate the strict patch budget. Then rerun the same
  canonical combination through ledger, episode asset, `obs_publish OK`, and
  final OBS existence.
- bible-worthy: yes -- bounded semantic repair eligibility depends on a proven
  transport behavior, not locality alone; promoted as executable BUG-11.42
  extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-03 -- P4 repair replaced a valid pass review with diagnostic-shaped JSON

- promotion: BUG-11.38 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `a43a3e77-2ba4-4420-a4e1-1982bf0448cc`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P3 cleared. P4 returned the correct three-field review
  with `verdict: "pass"` and an empty string issue list, but its rationale was
  one character over the 240-character cap. The generic typed repair changed
  the verdict to the invalid literal `fail` and changed issues into objects
  shaped like validation diagnostics, then exhausted the ladder before
  ledger/media/OBS work.
- root cause: the compact P4 seam named field lengths but did not state the
  exact verdict literals or that issues are strings. Its generic repair turn
  also supplied the entire score-shaped original request beside Pydantic error
  diagnostics, allowing input and diagnostic shapes to compete with the small
  output contract.
- fix: repeat an exact StructureReviewV4 contract at the base and repair
  boundaries: exactly `verdict`, `issues`, and `rationale`; only `pass` or
  `rewrite`; a flat list of at most six bounded strings; and one bounded
  rationale. Give the repair only the failed review and bounded rejection,
  require valid fields to remain unchanged, and explicitly forbid copying
  error codes/messages/shapes. The model still authors any shortening; Python
  does not clip review prose.
- verify idea: inject a correct `pass` review whose rationale is 241
  characters, capture both calls, and require the repair to preserve `pass`
  and empty string issues while shortening only the rationale. Assert both
  system prompts carry the literal/type contract and the repair input omits
  the accepted score/original request. Then rerun the same canonical 120-word
  combination through ledger, episode, `obs_publish OK`, and final OBS
  existence.
- bible-worthy: yes -- compact typed contracts must preserve literal and item
  type semantics at every repair boundary; promoted as executable BUG-11.38
  extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-04 -- P3 patch aimed at the hard cap and crossed it

- promotion: BUG-11.42 extension
- surfaced: canonical 120-word `scifi_codex` reverify, prompt
  `94a11e73-c7f8-47a1-b929-37c1cf7d63d6`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P2 cleared. P3 selected the bounded three-leaf text
  patch, but its patch schema failed. The outer ladder logged the stale base
  draft head, obscuring the direct patch error. Three exact live Aion probes
  then reproduced a premise replacement just over its 144-character cap.
- root cause: the request exposed the strict schema cap as the model's writing
  target, and its `original_text` field looked like an output value to copy.
  Approximate character counting crossed the edge; Aion also copied the
  over-cap source unchanged. The receipt collapsed every patch-schema error
  into `patch_root`.
- fix: expose only a conservative 75% model-facing `max_chars` for each leaf;
  keep the larger immutable schema cap private to validation so the model
  cannot anchor on the rejection edge. Root the input at `rewrite_tasks`, name
  the source `source_to_shorten`, and use one concise contract that forbids an
  unchanged copy. Never Python-clip authored prose. Classify replacement-
  string overflow separately without recording rejected prose.
- verify idea: require a model-facing 54-character ceiling for a scene whose
  private schema cap is 72, with no hard-cap field in the request. Require the
  action-shaped input and prove three exact live Aion probes pass. Inject a
  145-character replacement for a
  144-character target and assert a `replacement_over_schema_cap` receipt with
  no prose retention. Reprobe live Aion, then rerun canonical 120 through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- bounded authoring needs safety margin below its strict
  rejection cap; promoted as executable BUG-11.42 extension coverage.
- status: FIXED AND LIVE VERIFIED by canonical prompt
  `7a84b6c3-221e-4959-8636-e0d4e5e23838` (`obs_publish OK`)

## PBUG-20260713-05 -- P1 repair copied an overlong ending direction unchanged

- promotion: BUG-11.38 extension
- surfaced: canonical 42-word `scifi_codex` smoke, prompt
  `d1313994-753c-4748-bf8c-a4e09e15d8fe`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: RSS admission and deterministic P0 repair cleared. P1 returned the
  correct three-field DramaticQuestionV4 shape, but `ending_direction` exceeded
  its 120-character cap. The typed repair copied the same overlong value
  unchanged and exhausted the ladder before ledger/media/OBS work.
- root cause: P1 fell through to the generic graph-artifact repair contract.
  Although its generated JSON schema carried the numeric constraint, the
  authoring instruction did not repeat the three exact keys, per-field caps,
  or a safe rewrite target. Its repair input also repeated the full original
  request and fact index beside the tiny failed artifact, obscuring the only
  required edit.
- fix: give DramaticQuestionV4 its own compact repair boundary. Supply only the
  parsed failed question and bounded rejection; repeat the exact three root
  keys and hard caps; preserve valid fields byte for byte; require each
  rejected overlong field to be rewritten rather than copied or mid-word
  clipped; and give rewritten fields a conservative 75% authoring ceiling.
  Python never clips or authors the prose.
- verify idea: inject a valid question/consequence plus an overlong ending,
  require exactly one model-authored repair, and assert the repair prompt names
  the 160/160/120 hard caps plus 120/120/90 rewrite margins. Assert the repair
  payload omits the original request and fact index and preserves valid fields.
  Then rerun the same canonical 42-word combination through ledger, episode,
  `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- a tiny authored repair must repeat its exact contract,
  isolate the failed artifact, and target below the rejection edge; promoted
  as executable BUG-11.38 extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-06 -- P3 repair fixed total beats by overflowing one scene

- promotion: BUG-11.38 extension
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `a2b76223-c4be-49e3-945f-9fd1895a33a3`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P2 cleared. P3's base draft had fewer flattened beats
  than the locked six-row advisory. Its semantic repair restored all six beats
  but placed them in one scene, violating RadioScoreDraftV4's maximum of four
  beats per scene, and exhausted the ladder before ledger/media/OBS work.
- root cause: the compact schema said each scene had one to four beats, while
  the accepted advisory lived only in the input context. Neither the base nor
  repair instruction explicitly bound the locked global beat count to the sum
  of the scene-local arrays or derived the minimum scene count. The model fixed
  the named total-count rejection without preserving the independent local cap.
- fix: derive a bounded topology instruction from the accepted advisory before
  every P3/P3_rewrite call. State the exact locked flattened beat total, repeat
  the four-beat per-scene maximum, derive the minimum scene count, and require
  distribution across scenes. The same instruction is carried by base,
  restart, semantic repair, and rewrite boundaries; Python still derives only
  canonical mechanics after a complete valid authored draft.
- verify idea: use a six-row advisory and a schema-valid one-scene/four-beat
  draft, then require the repair to return a valid two-scene/six-beat draft.
  Assert both captured system prompts name exact total six, local maximum four,
  and minimum two scenes. Re-run the same canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- compact nested contracts must explicitly relate locked
  global cardinality to local collection caps at every repair boundary;
  promoted as executable BUG-11.38 extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-07 -- Source-grounded acronym was rejected as shouting

- promotion: BUG-11.51
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `3627b61a-8174-43e5-95f1-1a0c8f0269ec`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P4 and P3_rewrite cleared, including the repaired compact
  topology. P5 used the acronym `RIO` from accepted source evidence. The spoken
  hygiene validator rejected it as an all-caps lexical word; the model repair
  merely moved the same grounded acronym to another line and exhausted the
  ladder before ledger/media/OBS work.
- root cause: spoken hygiene used a blanket uppercase-token regex. It had no
  connection to the accepted FactIndexV4 evidence, so a legitimate acronym and
  ungrounded shouting were indistinguishable at every script boundary.
- fix: derive the exact set of uppercase lexical tokens only from the literal
  source spans already accepted for facts, entities, and numeric evidence.
  Thread that immutable allowlist through P5, P7, P9, and final spoken
  validation. Continue rejecting every all-caps lexical token absent from the
  accepted evidence; do not lowercase or rewrite authored dialogue in Python.
- verify idea: accept `RIO` when it is present in a literal accepted fact span,
  reject `STOP` in the same line, and prove the source-grounded set reaches all
  script validators. Re-run the same canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- lexical hygiene must distinguish source-grounded
  acronyms from ungrounded shouting at the validator boundary; promoted as
  executable BUG-11.51 coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-08 -- P3 rewrite overflowed more prose leaves than its patch envelope

- promotion: BUG-11.42 extension
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `3c4f6e67-8dda-47e8-802e-a37c6359e1b1`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 through P4 cleared and the repaired P3 topology held. The
  P3_rewrite response preserved structure but reauthored broadly, producing 13
  prose leaves just over their strict caps. That exceeded the proven 12-target
  local patch envelope, so the normal full-draft repair repeated all 13
  overflows and exhausted the ladder before script, ledger, media, or OBS.
- root cause: the base/rewrite authoring contract exposed every private schema
  rejection edge as the model's writing target. The rewrite instruction also
  allowed every creative prose leaf to change even when the review required a
  narrower correction. The bounded patch already used conservative targets,
  but it was only a downstream backstop after the broad overflow was created.
- fix: make conservative 75% prose ceilings the only model-visible limits at
  every P3 base, restart, full-repair, and rewrite boundary while retaining the
  larger immutable Pydantic caps privately. Require P3_rewrite to change only
  prose directly necessary for the review and preserve every other prior prose
  leaf byte for byte. Keep the proven 12-row patch envelope and fail-closed
  validation; do not expand an arbitrary capacity or Python-clip authored text.
- verify idea: capture base, full-repair, and rewrite system messages and assert
  they expose only the safe 48/108/60, 42/54/90, 48/21, and 60/90 ceilings.
  Require rewrite to preserve non-target prose byte for byte while retaining
  existing structure locks. Re-run the same canonical 42-word combination
  through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- safety margin belongs at every authoring boundary, not
  only the local patch after overflow; promoted as executable BUG-11.42
  extension coverage.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-09 -- P2 rejected an acronym-bearing canonical character name

- promotion: BUG-11.52
- surfaced: canonical 42-word `scifi_codex` reverify, prompt
  `7997800e-b0f5-4201-ae2d-193a899ac6f4`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13
- symptom: P0 and the compact P1 repair cleared. P2 returned `AI Unit 7`; its
  repair correctly removed the digit as `AI Unit Seven`, but the same validator
  rejected the legitimate `AI` acronym and exhausted the ladder before P3,
  script, ledger, media, or OBS.
- root cause: the cast-name grammar accepted only `[A-Z][a-z]+` tokens. The
  repair instruction said only "Title-Case," so the model fixed the visible
  numeric defect while the validator's hidden blanket acronym ban remained.
- fix: accept at most one short 2-3-letter acronym token inside a name that also
  contains at least one normal Title-Case word. Continue rejecting digits,
  lowercase labels, empty tokens, multiple acronyms, and all-uppercase full
  labels. State the exact grammar and `AI Unit Seven` example at the P2 repair
  boundary; do not rewrite character names in Python.
- verify idea: accept `AI Unit Seven` and `Dr. Amelia Hart`; reject `AI Unit 7`,
  `AI UNIT`, and lowercase names. Reproduce base `AI Unit 7` followed by the
  one-call authored repair and assert its exact model-facing grammar. Re-run
  the canonical 42-word combination through ledger, episode, `obs_publish OK`,
  and final OBS existence.
- bible-worthy: yes -- lexical validators must state and implement the same
  bounded acronym-aware name grammar; promoted as executable BUG-11.52.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-10 -- P9 audit blocked on a defect its only repair route could not touch

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `e0a03830-aa18-42c4-8c47-89c6cff51a46`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13. Logs
  `tmp\scifi_42_aion_original_pair_harness.out.log` and
  `tmp\scifi_42_aion_final_server.log`.
- symptom: the whole pipeline authored, validated, and grounded a complete
  script, then died at the last gate with `final contract audit rejected the
  script without actionable grounded findings` after 570s. The audit had
  returned `accepted=false` whose only findings named the manifest and a clue
  id (`manifest.lines[4].clue_ids`, `item_id=c4`) -- never a spoken line.
- root cause: the P9 seam asked the model to audit the script AND the manifest,
  truth map, and grounding contract, but the only correction the pass owns is a
  spoken-line retake, and `_audit_blocks` accepts a finding only when it names a
  script line and quotes an exact span. A finding about a derived artifact was
  therefore simultaneously authoritative enough to reject the episode and too
  unlocatable to repair -- a guaranteed dead end. The manifest is not even
  model-owned: Python compiles it from the accepted score and `_validate_manifest`
  already proves exact clue coverage, no duplicates, and landmark order.
- fix: state and enforce the audit's blocking authority. The seam prompt and the
  P9/P9_rerun repair rules now say only a finding whose `item_id` is a
  `script.lines` line_id and whose `exact_span` is copied verbatim from that
  line may block, and that manifest/truth/grounding concerns belong in
  `warnings`. `_validate_audit_envelope` runs as the P9 and P9_rerun
  post-validator: a blocking finding that names a real script line without a
  verbatim span, a rejection carrying no blocking finding, and an acceptance
  carrying one all return to the typed-repair ladder and fail closed if it
  exhausts. `_audit_advisories` demotes findings aimed at derived artifacts --
  a mechanical classification, never a judgment of authored meaning -- and
  records them verbatim in the new `final_audit_disposition` receipt. The dead-
  end raise is gone because the state is now unreachable.
- verify idea: assert a manifest-only `accepted=false` completes the episode
  with zero retakes and an advisory receipt row; a quoted script-line block
  still triggers exactly one retake; an unquotable script-line block reaches
  typed repair; an `exact_span` array stays a typed failure and is never
  normalized into index semantics. Re-run the canonical 42-word combination
  through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- an audit's blocking authority must not exceed what its
  repair route can re-author, and a validator that cannot be overruled must not
  be re-litigated by a model. Candidate for fan-out.
- status: FIXED at `3a98a6f1`; LIVE REVERIFIED 2026-07-13, prompt
  `28fe3cdf-e652-4db6-ab59-b7ddda6786ae` (same canonical 42-word
  `original_codex56sol` leg, Aion 3.0 Mini creative + Mistral-Nemo technical).
  `RESULT SUCCESS`, `obs_publish OK`, asset confirmed on disk (84,092,039 bytes,
  `output\otr\obs\signal_lost_waiting_room_whispers_20260713_122501_silent_procgen_blended_captioned_with_credits_final.mp4`).
  The live receipt proves the seam, not just the gate: P9 ran ONCE with no
  retake and no repair (`call_journal` `P9x1`), and the audit model met the very
  situation that killed the prior run -- a concern it could not act on -- and
  self-classified it into `warnings` verbatim: "The 'resolution_anchor' in the
  grounding_contract is missing. This is a compile-time issue and cannot be
  corrected during this pass." `accepted=true`, `findings=[]`,
  `blocking_script_findings=0`. That warning is itself a model misread -- the
  deterministic grounding validator proved the anchor IS spoken on the closure
  line -- which is exactly why a model's opinion about a Python-owned artifact
  must never hold a blocking vote.

## PBUG-20260713-11 -- P1 slate lost a clue per object with no rule to repair it

- surfaced: canonical 42-word `original_codex56sol` reverify of PBUG-20260713-10,
  prompt `0bca5788-9da4-4d23-b7da-c49984956bec`, Aion 3.0 Mini creative +
  Mistral-Nemo technical, 2026-07-13. Log `tmp\p9_verify_42_server.log`.
- symptom: the run died 121s in, before P9 was ever reached:
  `P1 failed ... after 2 attempt(s)`. Three of four possibility cards returned a
  two-entry `clue_plan` against `Field(min_length=3)`. The typed repair returned
  the same defect (4 errors, then 3), so the ladder exhausted.
- root cause: the clue-per-lost-object contract existed only as a bare pydantic
  `min_length=3`. The P1 seam never stated it, `_validate_slate` never checked
  coverage, and `_repair_rules` had no `P1` branch at all -- so the repair prompt
  carried the raw pydantic text plus a generic "repair only the typed contract
  error" and no rule telling the model to author the missing clue. The model
  read its own merged two-object clue as correct and kept it. The schema minimum
  is also not the real invariant: one clue per lost object means a four-object
  draw needs four clues, which `min_length=3` would silently pass.
- fix: state the same contract at all three surfaces. The seam and the new
  `_repair_rules("P1")` branch require 4-6 possibilities, verbatim immutable
  fields, and one distinct clue for EVERY lost object, in order, never merging
  two objects and never dropping a card to repair another. `_validate_slate` now
  derives the required count from the accepted draw
  (`len(clue_plan) >= len(draw.lost_objects)`) and reports the exact shortfall.
  Python never authors a clue: a clue is story, so the defect returns to the
  model and fails closed if the ladder exhausts.
- verify idea: assert a four-object draw rejects a three-clue card with the exact
  shortfall message; assert the P1 repair rules and seam both state the
  per-object rule; drive a runner where the base slate is clue-short and the
  authored repair restores it. Re-run the canonical 42-word combination through
  ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes -- a cardinality invariant that lives only in a schema
  minimum, with no matching prompt rule and no repair branch, is an unrepairable
  contract. Same class as PBUG-20260713-10. Candidate for fan-out.
- status: FIXED at `58983363`; LIVE REVERIFIED 2026-07-13, prompt
  `28fe3cdf-e652-4db6-ab59-b7ddda6786ae` (same leg and models). P1 passed on its
  FIRST attempt with no repair rung (`call_journal` `P1x1`) and the run carried
  through to `RESULT SUCCESS`, `obs_publish OK`, and the final OBS asset.
- coverage limit (stated, not hidden): `constraint_deck.json` ships 3 draws and
  every one has exactly 3 lost objects, so live production has only ever
  exercised the 3-object case, where the coverage rule and the bare
  `min_length=3` happen to coincide. The wider-draw behaviour this fix adds
  (4+ objects) is proven by unit test only. `ConstraintDraw` permits up to 6.

## PBUG-20260713-12 -- P4 wrote `blocking` as prose because no seam ever demanded the field

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `6c73745f-e639-4d4e-b46a-1cfeb7df3716`, Aion 3.0 Mini creative + Mistral-Nemo
  technical, 2026-07-13. Log `tmp\p245_verify_42_server.log`.
- symptom: `P4 failed after 2 attempt(s) -> ValidationError: findings.1.blocking
  Field required`. The model emitted the flag as PROSE inside the detail string --
  `"detail": "All lost objects are mundane items, blocking=false"` -- and omitted
  the actual boolean field. It also wrote `field_path` with bracket indexing
  (`caller_threads[0].lost_object`).
- root cause: a seam that PHRASED A FIELD AS PROSE, in a seam shipped hours
  earlier at `81336eca`. The fair-play seam said "report them with
  blocking=false" -- which reads as text to WRITE -- so the model wrote the
  literal string into `detail` and dropped the real boolean field.
  **Correction to an earlier hypothesis in this log:** the required path WAS in
  the model-visible contract. `schema_shape_instruction` DOES emit required
  nested paths (verified: `Required paths: accepted, findings,
  findings[*].category, findings[*].detail, findings[*].blocking`). What it does
  NOT emit is any BOUND -- no `min_length`, `max_length`, or `pattern`. So the
  invisible-contract class is real but narrower than first written: it covers
  bounds (PBUG-20260713-11's `clue_plan` `min_length=3`), not required-ness.
  This entry is a phrasing defect, not an unstated-field defect.
  Separately, `_corroborated_fair_blocks` derived the collection root with
  `split(".", 1)[0]`, so the model's bracket-indexed `field_path`
  (`caller_threads[0].lost_object`) resolved to `caller_threads[0]`, matched no
  collection, and would have silently failed to corroborate a real defect.
- fix: demand `blocking` as a real JSON boolean FIELD in the P4 seam and repair
  rules, and forbid writing it as prose inside `detail`/`category`. Apply the
  same wording to P7, whose `ListenerFinding` seam had the identical prose-invite
  exposure. Add `_field_path_root`, which strips an index suffix so
  `audible_clues[0].x` and `audible_clues.0.x` name the same collection (a
  mechanical read of a coordinate, not a reinterpretation of the finding).
  Add the CLASS GUARD that is actually true: every field carrying a schema BOUND
  must have that field named in the seam or the repair rules, across all 11
  structured passes. Running it surfaced two schema caps that forbade the only
  artifact their own validator would accept, both fixed here:
  `PossibilityCard.callers` capped at 4 while a draw may carry 6 lost objects
  (one caller each), and `ScoreIntentPatch.replacements` capped at 6 while the
  plan can target 6 anchors + reveal + closure = 8 and
  `_validate_score_intent_patch` demands every planned target. Its script-side
  twin was already correctly sized at 8.
- verify idea: the bounds guard itself (every bounded field named in the
  model-visible contract, all 11 passes); assert the seam demands a boolean field
  and forbids prose in detail; assert bracket and dot coordinates corroborate
  identically; assert no cap is below what its validator demands. Re-run the
  canonical 42-word combination through ledger, episode, `obs_publish OK`, and
  final OBS existence.
- bible-worthy: yes, as two separate rules. (1) A prompt must demand a structured
  field AS a field; phrasing a key's value in prose ("set x=false") invites the
  model to emit it as prose and drop the field. (2) A schema bound is invisible
  to the model -- the shape instruction emits required paths but never
  min/max/pattern -- so every bound must be restated in the model-visible
  contract, and no bound may be lower than what its own validator demands.
- status: FIXED IN TREE; LIVE REVERIFY PENDING

## PBUG-20260713-13 -- my own fair-play validator fail-closed on a benign path prefix

- surfaced: canonical 42-word `original_codex56sol` run, prompt
  `d5f66b1a-e85a-46b3-a447-7bf4f22d6e4e`, Aion 3.0 Mini creative + Mistral-Nemo
  technical, 2026-07-13. Log `tmp\final_42_server.log`. Self-inflicted by the
  validator shipped at `810369ff`.
- symptom: the new P4 retake route worked -- P4 corroborated a block, `P3_rerun`
  re-authored the truth map -- and then `P4_rerun` exhausted its ladder on my own
  error: `blocking finding for item 's4' sets field_path root 'truth_map', which
  does not own that item`. Episode dead at 408s.
- root cause: the model wrote `field_path` as `truth_map.causal_steps[...]`,
  prefixing the path with the key its input arrived under -- which is exactly
  what the payload calls it (`inputs={"truth_map": ..., "grounding_contract":
  ...}`). My root extraction took the FIRST dotted segment, got `truth_map`,
  found it owned no item, and classified a perfectly clear finding as ambiguous.
  It was never ambiguous: `item_id='s4'` resolves in exactly ONE collection. I
  had made `field_path` the identity when the `item_id` is the identity, and then
  fail-closed on a cosmetic disagreement -- turning a working retake into a kill.
- second failure, same gate (prompt `07725d30-0014-4da8-a9df-137663c3ad37`): the
  first fix classified by identity but still fail-closed when an item_id resolved
  in more than one collection. It then died on `item_id 't1' exists in more than
  one collection (caller_threads, resolution_links)` -- because `_truth_item_ids`
  keys BOTH of those collections by `thread_id`. Every thread-level finding is
  "ambiguous" BY DESIGN. The premise was wrong, not the branch.
- fix: DELETE the coordinate gate. `_truth_item_exists` asks the only question
  Python needs -- does this id name a real item anywhere in the accepted truth
  map? -- and that is all corroboration requires. The retake receives the finding
  verbatim and re-authors the whole truth map, so the owning collection cannot
  change the repair, and for a thread id it is not even a well-posed question.
  `field_path` is now a hint for the model, never a gate. The envelope keeps only
  the checks that catch the model contradicting ITSELF: accepted=true carrying a
  blocking finding, accepted=false carrying none, and a blocking finding on a
  real item with no category/detail.
- verify idea: assert every coordinate spelling corroborates the same defect --
  `audible_clues[0].x`, `audible_clues.0.x`, `truth_map.audible_clues[0].x`, a
  bare `truth_map`, an empty path, and a prose path -- including the two exact
  coordinates that killed prompts `d5f66b1a` and `07725d30`; assert `thread_id`
  resolving in two collections is not an error. Re-run the canonical 42-word
  combination through ledger, episode, `obs_publish OK`, and final OBS existence.
- bible-worthy: yes, and this is the sharpest lesson of the day. **A guard that
  cannot change the outcome must not be able to cause an outage.** I added a
  coordinate gate whose verdict the repair path never reads, and it killed two
  production episodes on cosmetic disagreements while improving nothing. Before
  adding a fail-closed check, name what a caller would DO differently with its
  answer; if nothing, it is not a guard, it is a liability. The corollary: an ID
  is the identity, a path is a hint, and Python must never fail closed on a
  coordinate it can resolve -- or on one it does not need.
- status: FIXED at `fdb5c433`; LIVE REVERIFIED at prompt
  `ee452c84-7bd7-4dba-9e45-ad15a255f8ab` -- the coordinate gate no longer fires;
  P4 corroborated, `P3_rerun` ran, and `P4_rerun` reached a real verdict for the
  first time. That verdict exposed PBUG-20260713-14 below.

## PBUG-20260713-14 -- the fair-play audit graded a property its artifact cannot express

- surfaced: canonical 42-word `original_codex56sol` runs, prompts `d5f66b1a`,
  `07725d30`, and `ee452c84` (Aion 3.0 Mini creative + Mistral-Nemo technical,
  2026-07-13). The last one reached the retake verdict and failed closed with
  `fair-play audit rejected the retaken truth map`.
- symptom: P4 raised a corroborated blocking finding on **3 of 3** live runs, and
  blocked the RETAKEN truth map as well. A repair route that fires every time,
  and that a retake cannot satisfy, is not a repair route.
- root cause: the P4 seam ordered the model to check "clue-before-reveal order"
  and "audible sufficiency" -- but `AudibleTruthMap` carries **no line order and
  no reveal position**. Nothing in it is "before" anything. The audit asked the
  model to judge a property the artifact cannot state, so it manufactured a
  defect every run, and `P3_rerun` could never fix what it could not represent.
  The property is not even unowned: it is already tested where it IS
  representable, by the P7 blind listener, which reads only the pre-reveal lines
  and must infer the mundane cause. P4 was duplicating a downstream gate on an
  artifact that cannot answer it.
- fix: narrow P4's charter to what a truth map can express -- causal closure,
  separate mundane possessions, the declared device as sole cause, benign safety,
  declared-name closure, and the helpful ending. The seam now explicitly forbids
  judging clue ordering, clue timing, clue-before-reveal placement, pacing, or
  listener experience, and says those are decided later and audited downstream.
  The retake machinery is unchanged and correct; it was aimed at an impossible
  question. Also persist the retake's CAUSE: `fair_play_disposition` was writing
  a hardcoded `"corroborated_blocking_findings": 0` and dropping the initial
  blocking findings, so the retake rate could not be calibrated from the receipt
  (v2 now records `initial_blocking_findings` and a real count).
- verify idea: assert `AudibleTruthMap` declares no ordering/timing field and the
  P4 seam forbids grading one; assert the blind-listener seam still owns
  "before the declared reveal"; assert the disposition records the initial
  blocking findings and a true count. Re-run the canonical 42-word combination
  and confirm P4 accepts without a retake, through ledger, episode,
  `obs_publish OK`, and final OBS existence.
- bible-worthy: yes. **An audit may only grade properties its artifact can
  express.** If the check needs an ordering, a timing, or a coordinate the
  artifact does not carry, the model will invent a verdict and the repair cannot
  converge. Audit each artifact for what it can say, and put ordering checks
  where ordering exists. Found by asking why a repair route fired 100% of the
  time -- a repair path that always fires is a design smell, not a safety net.
- status: SUPERSEDED by the 2026-07-13 rip. P4 no longer exists: fair play is a
  deterministic contract (`_validate_script_grounding` -- the device anchor is
  spoken on a clue line before the reveal line), and P7's blind listener is gone
  with it. Closed by the green 30-word canonical leg, prompt `fb34bf4f`.

## PBUG-20260713-15 -- the score's repair prompt did not fit the window it was sent to

- surfaced: canonical 30-word `original_codex56sol` run, prompt
  `a89a46a4-196b-41ad-89fa-1bbac4bb496d`, Mistral-Nemo both slots, 2026-07-13.
- symptom: P5 rejected a valid announcer cast row filed under `char_id: "a"`, and
  then the ladder collapsed: the typed repair returned prose and a JSON fragment,
  and the syntax retry returned ONE CAST ROW as the whole score (16 validation
  errors: 12 missing top-level fields, 4 extra).
- root cause: the P5 repair prompt was **5,772 tokens against a 4,592-token usable
  window** (context_cap 8192, max_new_tokens 3600). PROMPT_GUARD truncated it, and
  what fell off the end was the instruction to return a complete artifact. The
  model answered with the last thing it could still see.
- fix: `_repair_inputs` -- the P5/P6 repair no longer re-sends the full truth map
  and grounding contract; the failed artifact already carries the graph, so it
  sends only the anchors and the clue inventory. Plus `_project_announcer_char_id`:
  an id is a coordinate, not authored content, so it is canonicalized at the
  attempt boundary and the rejection never happens.
- bible-worthy: yes. **A repair prompt that does not fit is worse than no repair.**
  Silent left-truncation deletes the contract and the model answers from the
  fragment it can still see. Measure the repair prompt against
  `context_cap - max_new_tokens`, and bound the repair context to what the failed
  artifact does not already carry.
- status: FIXED at `b286c478`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` (RESULT SUCCESS + obs_publish OK + asset).

## PBUG-20260713-16 -- the lane died on echoes of its own inputs

- surfaced: canonical 30-word `original_codex56sol` runs, prompts
  `efafc6fa` (P1), `5bd46a5e` and `d199a783` (P3), `55756bac` (P5), Mistral-Nemo
  both slots, 2026-07-13.
- symptom: four separate deaths, all the same shape. P1: "lost_objects and
  acoustic_device must be copied verbatim" -- the model wrote the right story
  about the right device and re-worded the field that only ECHOES the immutable
  draw. P3: wrote `timetable` for `folded timetable`, then dropped the third
  caller thread entirely. P5: wrote `closure` for the `closing` enum and the
  schema threw out the whole score.
- root cause: the lane asked a 12B model to copy immutable strings back verbatim
  into typed fields, and compared exactly. Every one of those strings was an INPUT
  Python already owned. Worse, `caller_threads` carries `min_length=2` while the
  real rule is one thread per lost object -- the schema and the contract said
  different things, and the model believed the schema.
- fix: restore the input instead of dying on the echo, but ONLY when the
  correction is FORCED (exactly one value possible):
  `_restore_slate_immutables`, `_restore_thread_lost_objects`,
  `_project_arc_phases`, `_drop_unknown_clue_ids`. P3 is now handed the caller-
  thread ROWS as data (`required_caller_threads`) rather than asked to remember
  how many to write. Ambiguity still goes back to the model.
- bible-worthy: yes. **Restoring an input is not authoring.** When a model is asked
  to echo a string the program already owns, a mismatch is a coordinate error, not
  a story decision -- restore it when the correction is forced, and never let a
  schema bound contradict the real invariant.
- status: FIXED at `b286c478` + `f3f88cb0` + `5879d6ef`; LIVE REVERIFIED by the
  green 30-word leg, prompt `fb34bf4f`.

## PBUG-20260713-17 -- a proxy gate with a repair the model could not perform

- surfaced: canonical 30-word `original_codex56sol` runs, prompts `41faff33`,
  `6fe52216`, `522e1581`, `6a325375` (P5 anchor patch) and `ec428576` (P6 anchor
  patch), Mistral-Nemo both slots, 2026-07-13.
- symptom: five deaths in the bounded anchor patch. It returned truncated JSON at
  the creative temperature; then it wrote two of three required anchors into one
  intent and failed; then, at the script level, it rewrote two of three planned
  lines and the whole batch was rejected -- the two good lines went in the bin
  with the missing one.
- root cause: TWO design errors. (1) The score's intent-anchor rule was a PROXY: a
  `line_intent` is a private note nobody hears, and the anchors are real in the
  SPOKEN SCRIPT. The proxy's repair asked the model to fit two or three immutable
  strings into one sentence. (2) The script patch asked for every planned line in
  ONE call, so a partial success was a total failure.
- fix: rip the score-intent anchor patch and its seam entirely -- the anchors are
  proven in the script, whose patch rewrites dialogue the model is good at. The
  script patch now rewrites ONE line per call, at 0.25 temperature, with a token
  budget derived from the plan. A shortened anchor ("the grille" for "ventilation
  grille") is RESTORED, since the model decided where it belongs and the exact
  wording was never its decision.
- bible-worthy: yes. **A bounded repair must ask for the unit the model can
  deliver.** Batch a repair and a partial success becomes a total failure; enforce
  a property on a proxy artifact and the repair fights the wrong object.
- status: FIXED at `f3f88cb0`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` -- four per-line patches, each accepted on its FIRST attempt.

## PBUG-20260713-18 -- a 30-word broadcast with a nineteen-beat score

- surfaced: canonical 30-word `original_codex56sol` run, prompt
  `717f3a4f-53e4-47fc-9992-0aaedeb5fd72`, Mistral-Nemo both slots, 2026-07-13.
- symptom: P6 returned undecodable JSON -- the script was cut off mid-object.
- root cause: the P5 seam gave the score a beat FLOOR ("at least 5 beats") and NO
  ceiling, so the model built a nineteen-beat graph for a thirty-word broadcast.
  Every beat is a line the script must then write, and the script's token budget
  was computed from the word target alone -- which knows nothing about how many
  lines exist.
- fix: `_validate_score_scale` (a broadcast of N words holds N/4 beats, floor 8,
  cap 40) and a P6 token budget derived from the MANIFEST LINES. `max_beats` is
  supplied to the score author as data.
- bible-worthy: yes. **A generation budget must be derived from the artifact that
  will be generated, not from a proxy.** A word target does not bound a line
  count; a floor without a ceiling is not a size contract.
- status: FIXED at `f3f88cb0`; LIVE REVERIFIED by the green 30-word leg, prompt
  `fb34bf4f` (6 lines, arc_verdict=strong).

## PBUG-20260713-19 -- rerolled ledger text kept stale skip state
- surfaced: canonical 30-word `shakespeare` Aion creative leg, prompt
  `bfad7f51-042b-4733-ad8f-1257442148ae`, 2026-07-13
- symptom: the deterministic freeze audit rejected `b004` because its row had
  `skip=True` and non-empty text; `OTR_CastLock` stopped the run before render
  and the queue recorded `RESULT FAIL` / `QUEUE_BLOCKED shakespeare`
- root cause: the bounded reroll wrote replacement text through
  `Ledger.update_line_text()` but that mutator updated counts without clearing
  the row's old `skip`, `tts_skip_reason`, and `reviewer_skip_reason` fields
- fix: `c25d63c6` clears stale skip metadata at the meaningful text-write seam
  and preserves it for empty/whitespace text; focused regression coverage was
  added for both transitions
- verify idea: rerun the same canonical `shakespeare` Aion/local-Mistral leg
  and require `RESULT SUCCESS`, `obs_publish OK`, duration/audio checks, and a
  non-zero asset under `output\otr\obs`; unit-test a skipped row receiving
  meaningful replacement text and an empty replacement
- bible-worthy: yes -- paired authored state must be repaired at the seam that
  writes its partner; relaxing the deterministic validator would hide a real
  render contract violation
- promotion: BUG-05.11
- status: FIXED at `c25d63c6`; live requalification pending

## PBUG-20260713-20 -- a remote model's context window was read from the static row
- surfaced: live headless OpenRouter legs (`tmp/final2_42_server.log`,
  `final3_42_server.log`, `final4_42_server.log`, 2026-07-13), each of which
  logged `[OpenRouter] load slot=A ... slug=aion-labs/aion-3.0-mini
  route=default ctx=8192 (remote, 0 VRAM)`. The same lines appear for
  `tencent/hy3:free`. Aion advertises **131,072** tokens and HY3 **262,144**;
  both ran the whole episode against an effective **8,192**.
- symptom: no crash and no warning -- a SILENT 16x-to-32x understatement of the
  usable window on every remote call. Short legs never noticed, because the
  request stayed under `8192 - prompt` anyway. The damage was latent and
  scheduled: `original_codex56sol` P6 budgets
  `240 + 160*beats + 4*target_words`, and at 720 words (beat ceiling 40) that
  is **9,520** output tokens. `fit_output_tokens` would have reduced it to
  whatever 8,192 minus the prompt left, the performance script would have come
  back cut off mid-JSON, and the ladder would have reported a bare
  `JSONDecodeError` three times -- blaming the frontier model for a defect that
  was a constant in our own catalog row.
- root cause: `OpenRouterBackend.load()` took `context_window` from the
  CuratedModel row. The two OpenRouter rows (`openrouter:slot-a|b`) are VIRTUAL
  and STATIC: one row stands in for every slug an operator may bind to it, so
  its `context_window` cannot describe the model actually selected. It carried
  `DEFAULT_CONTEXT_WINDOW = 8192` -- a LOCAL, VRAM-shaped number that is simply
  false for a remote model. The catalog cache ALREADY stored each slug's real
  `context_length` (`_slim_model`), so the truth was on disk the whole time and
  was never read.
- fix: `32e680b2` adds `resolve_context_window(slug)`, which reads the resolved
  slug's advertised `context_length` from the catalog cache. A cold/stale cache
  has no entry -- that is a genuinely unknown window, so it falls back to the
  row default and says so LOUDLY rather than inventing a confident number.
  Also at the local transport: `OUTPUT_TRUNCATED` now logs the full arithmetic
  at ERROR whenever generation stops at a ceiling that was itself a clamp (a
  reader must never have to reconstruct why the JSON was cut), and the
  unreachable PROMPT_GUARD left-slice is deleted rather than left as a dead
  lever for the next reader to repair.
- verify idea: a canonical leg with `creative_writing_model=openrouter:slot-a`
  and `openrouter_slot_a_model=aion-labs/aion-3.0-mini` must log
  `ctx=131072`, not `ctx=8192`; and a 9,520-token output request must reach the
  wire whole (`max_tokens=9520`) instead of being clamped.
- bible-worthy: yes -- a capability constant that stands in for a FAMILY of
  models describes none of them. When a per-instance truth is already cached,
  a static row default is not a fallback, it is a lie with a default value.
  Measure a budget against the window of the model that will actually serve it.
- promotion: queued for operator fan-out (overlaps the BUG-11.50 structured-
  capacity family but is distinct: that family is about artifact size, this is
  about the WINDOW the artifact is measured against).
- status: FIXED at `32e680b2`; LIVE REVERIFIED by the green 30-word
  `original_codex56sol` Aion leg, prompt `411c2f17-c05a-4af4-a6cf-c578183c072b`
  -- server log shows `slug=aion-labs/aion-3.0-mini route=default ctx=131072`,
  `RESULT SUCCESS`, `obs_publish OK`, 65.5 MB asset.

## PBUG-20260716-01 -- writer-model dropdown mislabeled on-disk models NOT DOWNLOADED + emitted a contradictory "[LOCAL HF] [NOT DOWNLOADED]" double suffix
- surfaced: live ComfyUI writer-model dropdown (OTR node INPUT_TYPES), operator-observed 2026-07-16; reproduced on the box with the Windows venv against the real HF cache (labels matched the operator report 6-for-6)
- symptom: on-disk Gemma rows shown `google/gemma-4-E2B-it [LOCAL HF] [NOT DOWNLOADED]` (contradictory double badge) and `google/gemma-2-2b-it` / E2B shown `[NOT DOWNLOADED]` though their HF snapshots (4.9 GB / 9.8 GB) are on disk; detection inconsistent across peers (E4B labeled correctly)
- root cause: TWO independent defects. (1) cache-root + completeness: `_hf_hub_root()` read only `HF_HOME` + the legacy `HUGGINGFACE_HUB_CACHE`, never the modern `HF_HUB_CACHE` that huggingface_hub itself honors, so a process without those two vars fell through to the stale `~/.cache/huggingface/hub` default (partial coverage: E4B present, E2B/gemma-2-2b hollow); and `on_disk` was set from "a `snapshots/<hash>` dir exists", not from a materialized weight blob, so a config-only / tokenizer-only pull read as downloaded. (2) label composition (regression `e412e84b` "Disambiguate local Gemma model labels"): `_display_label_for_local_row` began adding `[LOCAL HF]` UNCONDITIONALLY for any `google/gemma*` row while `build_dropdown_choices` still appended `[NOT DOWNLOADED]` on a scan miss -> the two badges stacked
- fix: same commit as this entry -- `_hf_hub_root()` now honors `HF_HUB_CACHE` (then `HUGGINGFACE_HUB_CACHE`, then `HF_HOME/hub`, then default; read live from os.environ, matching huggingface_hub precedence so scanner == loader); `scan_local_llm_cache()` sets `on_disk` only when the chosen snapshot carries a materialized weight blob (`_snapshot_has_weights` -> symlink-resolved `*.safetensors`/`*.bin`, size > 0), preferring the newest weighted snapshot; `build_dropdown_choices()` makes the state suffix EXCLUSIVE ([LOCAL HF] XOR [NOT DOWNLOADED]); parametrized per-item label-vs-disk regression added to `tests/test_model_catalog_scan.py`
- verify idea: for each curated local row, assert the emitted dropdown label carries exactly one state suffix for a given fixture cache state (materialized weight blob -> `[LOCAL HF]` for gemma / bare id otherwise; config-only OR absent -> `[NOT DOWNLOADED]`; never both); assert `_hf_hub_root()` returns the `HF_HUB_CACHE` path and wins over `HF_HOME/hub`; assert `_snapshot_has_weights` is False for a config-only snapshot and True for one with a weight blob
- bible-worthy: yes -- generic rule: a model-picker that scans the HF cache must (a) resolve the cache the LOADER uses (honor HF_HUB_CACHE, not just HF_HOME/legacy alias), (b) gate "downloaded" on a materialized weight blob, not a bare snapshot dir, and (c) keep UI state badges mutually exclusive. Hits any custom node that labels a model dropdown from a cache walk
- operator env note (NOT code-fixable): on this box the ComfyUI process resolves to `~/.cache/huggingface/hub` because it has no HF_* var set; the User-registry `HF_HUB_CACHE=C:\ComfyUI-Models\huggingface` points at the CONFIG-ONLY parent (weights live in `...\huggingface\hub`). For the dropdown to show the real weights, launch ComfyUI with `HF_HOME=C:\ComfyUI-Models\huggingface` (yields `/hub`) or `HF_HUB_CACHE=C:\ComfyUI-Models\huggingface\hub`. The code fix makes the label HONEST for whatever cache the process actually uses
- follow-up (operator directive 2026-07-16, same day): after seeing the corrected labels the operator observed the download-state STILL depends on each user's HF cache layout ("has to work out of the box for every user regardless of where they store their files"), which no scanner can guarantee. Per that directive the download-state badges were REMOVED entirely: `build_dropdown_choices` now emits the bare repo id / handle with NO `[LOCAL HF]`/`[NOT DOWNLOADED]`/`[LOCAL GGUF]` badge (the dead `_display_label_for_local_row` + `_is_google_gemma_local_row` helpers were ripped). `on_disk` is still tracked internally (recovery hint + auto-download short-circuit) and the HF_HUB_CACHE + weight-completeness fixes are retained; the SUFFIX CONSTANTS + `_strip_label_suffix` stay so a value saved by an older badge-bearing workflow still normalizes. Selection is never gated -- a not-cached model is fetched by `auto_download_if_missing` on first Queue
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `build_dropdown_choices` (`nodes/_otr_model_catalog.py`) emits a bare `repo_id` with no badge-suffix logic at all (`aaaf660a` + `0ae59ed4`). There is no code path that can print `[NOT DOWNLOADED]`.
- previous status: OPEN (badge-label surface removed; underlying cache-resolution/completeness fix stands)
- status: **CLOSED 2026-08-18 -- FIXED, symptom structurally unreachable**

## PBUG-20260717-01 -- codex P0 FactIndex literal-span rejects a whitespace-polluted RSS source
- surfaced: live 30w headless `scifi_codex_v4` leg, 2026-07-17, canonical prompt `ac027c36-4aab-412b-9844-6239cf561d4f` (RESULT FAIL at node 1 OTR_LedgerScriptWriter, pass P0)
- symptom: P0 fails after 2 attempts -- `fact F01 has a non-literal source span: full_text[11:54] expected exact slice '\n\t\t\t\t\t\t\t\tThe Growing Crescent of Mars as NA'; returned quote 'The crescent of Mars grow as the spacecraft approached the planet...'`
- root cause: NEW upstream (ingestion) root -- the INVERSE of the S5 P0 evidence-contract family (PBUG-20260710-10/BUG-11.35, -20260711-02/BUG-11.37, -04/-07/BUG-11.46, -20260712-23/BUG-11.50 ext), all of which assume a clean payload and wrong MODEL metadata. Here `A0.full_text` carried leading `\n`+8 tabs from the RSS source, so the literal-span offset `[11:54]` landed inside the whitespace run and cut a word -- a slice no model can reproduce verbatim, so it paraphrased and the exact-literal validator rejected it. No prior PBUG/Bible rule normalizes source whitespace at INGESTION (BUG-11.26 is a comparison-time whitespace fix for the key_term verbatim test, not offset-bearing-payload cleaning).
- fix: same commit as this entry -- normalize the four span-bearing fields (headline/summary/full_text/seed_text) to single-spaced text in `_otr_scifi_codex.validate_payload_envelope`, AT ADMISSION and UPSTREAM of the digest + the P0 evidence projection + the literal-span validator (cleaned text is the sole coordinate system, so no accepted offset shifts -- the BUG-11.37 constraint); point the P0 post-validator at `env.payload` (the normalized A0) not the raw input `payload`. Codex-scoped -- shared `validate_source_payload` stays byte-identical for the science ledger stamps. New helper `_normalize_span_source_text`.
- verify idea: inject a fact whose source field has leading `\n\t...`; assert `env.payload` span fields are single-spaced/stripped, a literal span into the cleaned full_text passes `_span_ok` first-try, and normalization is idempotent on an already-clean control (no offset shift). Covered by `tests/test_scifi_codex_lane.py::test_p0_source_spans_survive_whitespace_polluted_source`.
- bible-worthy: yes (operator decides at fan-out) -- generic rule: when a contract validates a model quote against a literal offset slice of a source payload, normalize the source whitespace at ingestion UPSTREAM of offset assignment; NOT a whitespace-tolerant validator (leaves dirty offsets stored for every downstream consumer -- the BUG-11.50/PBUG-20260712-23 anti-pattern) and NOT a seam nudge (model-obedience gamble, BUG-11.54). Nearest kin = the S5 family + BUG-11.50; new UPSTREAM root. (Cross-check window ruling, 2026-07-17.)
- status: **LIVE-VERIFIED** -- full episode green on leg `c1f3891f` (RESULT SUCCESS + obs_publish, "The Whisker Effect" 56.6 MB on disk). P0 cleared on the same whitespace-polluted source class that aborted pre-fix (leg 90f22b15 first proved P0 clears; c1f3891f proves the whole episode). Bible-promote at the next operator fan-out.

## Regression watch (2026-07-17 -- NOT a new PBUG) -- codex P3 string_too_long on `premise` re-occurred on scifi_codex_v4
- RE-OCCURRENCE of PBUG-20260713-04 (BUG-11.42), not a new class (cross-check window ruling). Live 30w `scifi_codex_v4` leg (prompt `6883758f`) failed P3 `string_too_long` on premise >144 -- same field, same 144 cap, same lane, same mechanism (model writes over-cap; text-patch never clips prose). The -04 verified recipe (conservative ~75% model-facing `max_chars` with the true cap PRIVATE + `source_to_shorten`/forbid-unchanged-copy + never Python-clip) is present in tree (`_otr_scifi_codex.py:1752/:1754/:1800` + surface instruction premise<=108). A same-session kibitz had re-added the literal 144 cap to the base seam; per -04 that is the anti-pattern (exposing the rejection edge makes the model aim at it and cross it), so it was REVERTED (same commit as PBUG-20260717-01). Untestable end-to-end until PBUG-20260717-01 (P0) clears; sequence: P0 clears -> a live 120w leg exercises the P3 premise cap. If premise still overruns for the v4 proof-pressure density AFTER -04's recipe, the BUG-11.54 deterministic word-boundary shortener (already used for question/consequence) is the design precedent. No new PBUG until a live failure survives the -04 recipe.
- **RESOLUTION (2026-07-17, operator decision "allow longer text"):** after P0 cleared, a live leg (`90f22b15`) failed P3 `string_too_long` on BOTH `premise` (>144) AND scene `description` (>72) -- the -04 recipe IS insufficient for the verbose v4 proof-pressure lane. Operator chose to RAISE the caps rather than clip prose. Raised the non-spoken metadata caps: `premise` 144->240, scene/shot `description` 72->144 (draft+final models + `_p3_text_patch_cap` + the text-patch `replacement_text` schema bound + the receipt). These caps are **load-bearing** (they size the P3 draft to the model's 8192 context+output budget), so the output reservation was resized `1647->1829` and every exact-token guard updated (max-width draft 1418->1576 tokens; envelope re-verified prompt+output=5935<=8192). Full suite 8144 / Bible 17. **LIVE-PROVEN**: leg `c1f3891f` RESULT SUCCESS + obs_publish -- premise+description now fit the raised caps end-to-end (obs asset on disk). NOT promoted to a new PBUG (re-occurrence of -04, resolved by the cap raise, not a novel class).

## PBUG-20260718-01 -- scifi_fable2_v3 was a runnable=True bank that could never complete a leg (fable2 revision_contract hardcodes rules_id == 'scifi_fable2')
- surfaced: live cross-bank Sonnet bake-off render window, 2026-07-18, baseline HEAD `60c73618`; the `scifi_fable2_v3` story-only leg logged `RESULT FAIL canonical_runner_exit=1` at t=22s, before any generation, and is model-independent (reproduced under creative=`anthropic/claude-sonnet-4.5`). Full causal record: `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`.
- symptom: `!!! [scifi_fable2] pass 'revision_contract' failed: story_rules.rules_id must be 'scifi_fable2', got 'scifi_fable2_v3' (no fallback to legacy_many_pass)` -> `nodes._otr_scifi_fable2.Fable2ScriptError`.
- root cause: the fable2 lane (`nodes/_otr_scifi_fable2.py:2307`) hardcodes the expected `rules_id` to the literal `"scifi_fable2"`. The 2026-07-17 roster trim (`499386aa`) made every lane own its `story_rules` by EXACT id, so `scifi_fable2_v3`'s rules carry `rules_id = "scifi_fable2_v3"` while its pipeline `fable2_multipass_v3` still routes into `_otr_scifi_fable2` -- which then rejects the v3 id. Net: a `runnable=true` bank that can never finish a leg. (`scifi_fable2` base is unaffected -- rules_id == 'scifi_fable2'.)
- fix: RETIRED the bank rather than patch the contract (Sonnet-bake-off verdict, `docs/2026-07-18-sonnet-bakeoff-analysis.md` + `docs/2026-07-18-rip-4-banks-plan.md`). The `scifi_fable2_v3` bank row, pack dir, `story_rules`, and its `fable2_multipass_v3` pipeline (removed from BOTH `pipelines.json` and `_RUNNER_BY_PIPELINE`) plus the writer's fable2 target-word gate entry were all deleted in this change, alongside `media_archive_v3` / `scifi_codex_v3` / `scifi_sonnet_v3`. No live route to the defective contract remains.
- verify idea: `scifi_fable2_v3` no longer appears in `_otr_story_routing._ensure_loaded().pipelines`, `banks.json`, or `_RUNNER_BY_PIPELINE`; the source-only retired-id scan over `nodes,tests,workflows` returns zero; full suite + Bug Bible stay green with the bank gone.
- bible-worthy: no -- resolved by removal, not a reusable code contract. If the fable2 family re-adds a `_v3`, re-open the NEWBUG fix-direction: accept the lane's DECLARED rules_id, never a single literal.
- status: **CLOSED-BY-RIP** at this commit. NEWBUG doc marked CLOSED-BY-RIP and RETAINED (the only causal record of the live failure -- never deleted).

## PBUG-20260720-01 -- official Gemma 4 12B HF writer was stranded behind an obsolete architecture/catalog gate
- surfaced: offline Gemma recovery probe plus canonical headless requalification on the RTX 5080 16 GB box, 2026-07-20. The complete official checkpoint was already under `C:\ComfyUI-Models\huggingface\hub`, but the installed Transformers 5.5.0 did not recognize `model_type=gemma4_unified`; the catalog separately hard-rejected `google/gemma-4-12b-it` and steered users to the unconstrained GGUF row.
- symptom: the official 12B model could not be selected on OTR's in-process Transformers/HF lane, so the writer could not use that lane's lm-format-enforcer token grammar. The optional GGUF lane instead reached character-zero JSON failures in structured work.
- root cause: the catalog tombstone outlived the runtime limitation that prompted it. Correct inference requires native `Gemma4UnifiedForConditionalGeneration` support, not the retired text-tower remap. This machine's HF cache also splits the materialized weights and the newer `chat_template.jinja` across two revisions, while the old resolver assumed the newest snapshot directory was complete.
- fix: require Transformers >=5.10.4, restore the curated `google/gemma-4-12b-it` row, remove its hard reject, resolve the newest materialized-weight snapshot plus newer compatible local chat-template metadata, and keep tokenizer/config/model loads `local_files_only=True` with no in-loader HTTP fallback. The canonical workflow now selects the row in both writer slots with `cuda` / `sdpa` / `bnb_nf4`, context 8192. Exact result schemas are bound at the local P0-P9 scheduler boundary; P3's authored-text patch keeps its narrower schema.
- verify idea: in a zero-network process require the official Gemma4Unified class, `is_loaded_in_4bit=True`, coherent prose, and LMFE JSON that decodes and validates. In the real canonical workflow require P0's raw head to begin with `{` and reach semantic validation instead of character-zero parsing.
- bible-worthy: yes -- architecture capability and coherent split-revision cache resolution are reusable model-admission contracts.
- promotion: BUG-02.16.
- status: **FIXED; LIVE-REQUALIFIED THROUGH P5**. The doctor measured 7.152 GiB allocated / 7.286 GiB peak and returned coherent prose plus parsed constrained JSON. Canonical prompts `4a89df7e-c8e1-407f-ab10-c3159eb17b6b` and `ee0d4743-11bc-4367-9e19-5422afa2c95f` both loaded offline NF4 at a 7.15 GiB model delta; P0 began with valid JSON, decoded, and needed only deterministic source-span repair. The second leg reached P5 with a complete schema-valid artifact. Full media publication remains unclaimed because that leg later exhausted the existing P5 spoken-hygiene semantic repair.

## PBUG-20260720-02 -- an open P5 scene dictionary crashed LM Format Enforcer mid-object
- surfaced: first real canonical Gemma/HF leg, prompt `4a89df7e-c8e1-407f-ab10-c3159eb17b6b`, 2026-07-20. P0-P4 and P3 rewrite had already cleared under hard constraints.
- symptom: every P5 attempt began valid JSON and stopped at `..."scenes":[{`; LMFE logged `AttributeError: 'bool' object has no attribute 'anyOf'`, after which the retry ladder misleadingly reported character-zero JSON because no complete top-level object remained.
- root cause: `ScriptArtifactV4.scenes: list[dict[str, Any]]` compiled to `items: {type: object, additionalProperties: true}`. LMFE 0.11.3 accepted that schema initially but treated the boolean wildcard as a schema object when generation reached the first arbitrary scene key, then terminated token enforcement.
- fix: replace the wildcard with the real closed contract, `ScriptSceneV4(scene_id, env, description)`. Hard enforcement stays enabled; no unconstrained fallback or output stripping was added. A regression feeds a complete production-shaped P5 artifact through `JsonSchemaParser` one character at a time and requires every character to be allowed plus `can_end()` at completion.
- verify idea: assert the P5 scene schema has exactly the three required properties and `additionalProperties: false`; scan every bound P0-P9/P3-patch schema for boolean wildcards; run P5 live and require a complete artifact to reach post-validation without an LMFE internal error.
- bible-worthy: yes -- validate generated schemas against the grammar compiler's supported subset before binding them to local structured generation.
- promotion: BUG-11.55.
- status: **FIXED; LIVE-REQUALIFIED AT P5** by prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f`: P5 produced a complete, schema-valid JSON artifact and entered the ordinary spoken-text post-validator. The later semantic repair exhaustion is not a recurrence of this grammar/compiler bug.

## Regression watch (2026-07-20 -- NOT a new PBUG) -- Gemma P5 repeated a spoken-hygiene defect after bounded repair
- prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f` produced a complete constrained P5 artifact but line `l001` contained stage direction, markup, or a role label. The existing Axis-6 route from `docs/2026-07-18-codex-short-leg-convergence.md` correctly selected the spoken-reword repair rule; Gemma repeated the same defect and the lane failed closed after the bounded model repair. This is model non-compliance at an existing semantic gate, not a JSON/LMFE regression and not evidence for a new deterministic code defect. It blocks a full-episode promotion claim, so the handoff records runtime/grammar qualification only.

## PBUG-20260720-03 -- a craft-only spoken-line reject could kill the episode
- surfaced: canonical Gemma/HF requalification prompt `ee0d4743-11bc-4367-9e19-5422afa2c95f`, 2026-07-20, after P0-P4 had cleared and P5 had produced a complete schema-valid artifact
- symptom: P5 line `l001` failed with `spoken text contains stage direction, markup, or a role label`; Gemma repeated the defect on typed repair and `_otr_structured_call` raised after two attempts, so no accepted/frozen ledger, TTS, video, or OBS asset was produced. Code grounding also found the shared freeze path could translate craft/quality exhaustion into a terminal-skip disposition with downstream readiness phases stamped `terminal_skipped`
- admission note: this supersedes the preliminary "NOT a new PBUG" regression-watch classification immediately above. That note assessed only model noncompliance; grounding exposed the distinct deterministic workflow-liveness defect: a sanitizable quality reject controlled episode completion
- root cause: spoken craft exhaustion had no total post-model repair boundary. Quality-budget exhaustion shared terminal liveness semantics with genuinely invalid structure, and content-owned lanes could validate authored text before delivery normalization rather than the exact TTS surface. One stubborn but sanitizable row could therefore kill the whole episode
- fix: all six runnable banks now use a total spoken-hygiene ladder. The established repair/recompose, lower-temperature CRITICAL, and alternate-slot rungs remain the opening A/B/C phase; unresolved rows then enter a dynamic fresh repair/rejudge loop that rotates every callable same/alternate writer lane with new defect feedback and temperatures. Every candidate is rescored on the exact projected spoken surface. A finite model-pass budget ends at an idempotent validated SFW floor, so a stubborn quality model cannot hang or kill the episode. Repaired rows stamp the gate and resolving rung. Craft exhaustion continues through freeze/readiness, while a truly empty mechanical row is isolated locally. Structural ambiguity and the deterministic G9 SFW ship-stop remain fail-closed
- sibling quality paths: `scifi_news` (the renamed Codex implementation) now repeats its P6/P7 listener judge+creative-retake and P8/P9 final-audit+retake cycles until clean or the validated quality floor; its typed spoken patch path uses the same dynamic cross-slot policy. `original` likewise alternates fresh creative/technical outro repairs and independent technical re-judges for subjective epilogue findings, then keeps the best structurally valid close with a nonterminal `quality_floor` receipt if its dynamic 3-6-cycle budget is exhausted. Inline Story QA feeds each MICRO/REJECT concern into a fresh scoped creative repair and an independent technical rejudge under the same liveness rule. The source-adapter family (`media_archive`, `public_domain`, `shakespeare`, plus the registered news interpreter) retains separate bank prompts/schemas/truth validators but now shares a 12-model-call liveness chain: technical and creative slots alternate with the exact prior rejection, then a validated bank-specific brief is derived only from the fetched payload, source hash, and source-side cast hints. Broken feeds, manifests, rights/config, backends, and interpreter contracts still fail loud. `scifi_news_pro` (the renamed Fable2 implementation) keeps its existing content ownership and seal/rebuild contract; only the shared total spoken repair boundary applies when one of its sealed rows is defective
- verify idea: force every craft gate and whole-line stage cue to survive all model rungs; require a non-empty clean floor result plus `hygiene_repaired_after_reroll:<gate>:<rung>`, normal Phase 7/8/10 completion, and no quality verdict in `FREEZE_TERMINAL_FAILURE_VERDICTS`. Cover `media_archive`, `original`, `public_domain`, `shakespeare`, `scifi_news`, and `scifi_news_pro`; assert delivery projection is repaired before content-owned seals; retain an unsafe-line test that Phase 10 refuses to freeze
- bible-worthy: yes -- generic rule: an LLM's refusal to satisfy a non-safety wording gate must not own workflow liveness when a bounded deterministic clean spoken projection exists
- promotion: BUG-11.56
- status: **ROOT-FIXED / LIVE-VERIFIED**. Final canonical prompt
  `f3770246-2d6a-4302-90af-153120edddf2` exercised the new boundary twice:
  P5 repaired four `one_breath` rows and P7 repaired a
  `spoken_format` / `stage_direction` row; both receipts stamp
  `trigger=craft_only_post_validation` and
  `shared_artifact_repair_bypassed=true`. The ledger froze
  `frozen_with_warns` (cosmetic word-count receipts only), all Phase 7/8/10
  telemetry remained non-skipped, four clean lines / 45 words reached TTS and
  video, and `obs_publish OK` wrote the 22,892,541-byte final asset at
  `output/otr/obs/signal_lost_the_weight_of_height_20260720_221418_silent_procgen_blended_captioned_with_credits_final.mp4`.
  A later P9 score-graph mismatch correctly stayed outside the craft-only
  boundary; its full-artifact retries hit the separate 8K structured-capacity
  limit and the already accepted clean script still completed normally.

## PBUG-20260720-04 -- alias-blind media consumers dropped the sentinel announcer identity
- surfaced: the published Fable2 Einstein and Butterfly episodes audited in `docs/2026-07-10-fable2-s2-QA-ANALYSIS-r2.md`. Einstein captions omitted an ANNOUNCER label around the sentinel; Butterfly labeled the intro sentinel but omitted the coda sentinel
- symptom: the ledger and rendered episode completed, but a normalized/cast-keyed sentinel could lose its canonical speaker label in captions. Static sibling grounding found that credits could resolve the alias-aware display name yet miss the same row's voice receipt, HuMo could reject the normalized radio face unless `char_id` remained the literal `announcer`, and captions could consume a canonically skipped row instead of filtering it row-locally
- root cause: downstream media consumers independently rebuilt raw exact-`char_id` maps instead of using the central alias-aware ledger-consumer resolver and canonical skip semantics. ShotLock correctly normalizes the sentinel to a cast identity, but HuMo's later stale guard still tested the pre-normalization literal ID
- fix: captions now filter canonical skips before ordering, preserve canonical caption text, and resolve speakers through the shared alias-aware cast lookup; credits use that same lookup for both display name and voice; HuMo recognizes the sentinel by role/source-family/portrait identity after ShotLock normalization. No ledger ownership, readiness, seal, hash, node, widget, or canonical-workflow surface changed
- verify idea: feed intro/coda sentinel aliases plus an ordinary similarly named cast row through captions, credits, ShotLock, and HuMo; require both announcer labels and the voice receipt, exclude skipped rows without a timing clamp, accept the normalized sentinel portrait, and still reject the ordinary stale mismatch
- bible-worthy: covered by existing BUG-12.43 (namespace aliases must resolve at every consumer) and BUG-05.11 (canonical skip state is row-local); no new portable rule
- status: **ROOT-FIXED / FOCUSED-GREEN; canonical six-bank live qualification pending**

## PBUG-20260720-05 -- caption suffix made the terminal mux publish into a fake sibling episode
- surfaced: production-artifact audit of the completed `media_archive` episode `signal_lost_reel_history_20260720_102732`, 2026-07-20. Its ledger named a 105,782,049-byte final that existed on disk, but the path was under the invented sibling directory `signal_lost_reel_history_20260720_102732_silent_procgen_blended_captioned` rather than the episode root
- symptom: every media stage could complete and OBS could receive a playable copy while the archival final escaped `meta.paths.episode_root`. A success/file-exists check alone therefore blessed a structurally wrong output tree and left the real episode directory without its terminal final
- root cause: `OTR_MasterAudioMux._default_out` reconstructed the episode id by peeling a hard-coded suffix list. Captions were inserted before credits, but `_captioned` was absent from that list; after `_with_credits` was removed, the remaining enriched stem was reinterpreted as a new episode id. Any future terminal enrichment could repeat the class
- fix: the mux now treats the in-flight ledger path as the canonical episode-directory authority and accepts it only when it is a direct child of the configured episodes root and the incoming stem begins with that episode id (rejecting a stale prior-episode singleton). Ordered suffix peeling, now including captions, remains only as the no-ledger fallback. The fully enriched filename is preserved. No node, widget, link, or canonical-workflow change was needed
- verify idea: point the in-flight ledger at `otr/episodes/ep042/audio/ep042_ledger.json`, feed an input with an unknown future terminal-enrichment suffix, and require the final parent to remain exactly `otr/episodes/ep042`; require a mismatched stale ledger to be rejected and the caption/credits fallback chain to peel to the correct episode root
- bible-worthy: yes -- portable rule: when an accepted manifest/ledger already owns an artifact directory, downstream enrichments must consume that authority rather than reverse-engineering identity from an open-ended filename suffix grammar
- status: **ROOT-FIXED / FOCUSED-GREEN; canonical six-bank live qualification pending**

## PBUG-20260721-01 -- selected RSS provenance disappeared behind a blank request widget
- surfaced: first canonical six-bank qualification leg, `media_archive` prompt
  `12f3df7f-298e-411c-9fe2-59ef3ac62ae2`, published episode
  `signal_lost_the_casting_reels_20260721_010623`, 2026-07-21
- symptom: the fetched media payload carried a real selected article link,
  outlet, date, and embedded source hash, but the final ledger stamped a blank
  `meta.source_ref` and empty source sidecars. The story and media rendered, so
  an output-only check could not detect that the ledger no longer identified
  the item it had adapted
- root cause: the RSS fetchers returned the strict seven-key payload as a raw
  dict. `normalize_fetch_result` intentionally treats a legacy raw dict as
  having no provenance sidecars, and `_resolve_inputs` then wrote the optional
  request widget as `source_ref`. Both RSS families ignore that widget and
  choose an item dynamically, so the request coordinate could never name the
  selected source. The sibling defect covered `media_archive`, `scifi_news`,
  and `scifi_news_pro`; manifest-backed and synthetic banks already owned their
  provenance explicitly
- fix: the two known RSS wrappers now preserve the exact seven-key payload but
  return `SourceFetchResult` with selected URL/label/date metadata and explicit
  unknown rights. The writer resolves the canonical ledger `source_ref` in
  owner order (fetcher-selected ref, selected payload link, requested widget)
  and stores a differing request separately as `requested_source_ref`. It does
  not invent a license or fair-use claim
- verify idea: make each RSS wrapper select a link different from the supplied
  request; require the selected link at `meta.source_ref`, in source metadata,
  and in rights provenance, with the request retained only as a request. Keep
  raw-dict legacy normalization and both manifest-backed banks unchanged
- bible-worthy: yes -- BUG-12.54
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-02 -- the frozen master WAV had no durable byte identity or final pointer
- surfaced: same completed `media_archive` qualification artifact as
  PBUG-20260721-01. During video generation every per-beat slice logged that
  `ledger.audio.master_audio_sha256` was absent
- symptom: the master WAV existed and the final archival MP4 was proven
  byte-identical to it, but the final ledger had no top-level `audio` section,
  no full master hash, and a blank `final_audio_path`. The video slice cache
  therefore fell back to path identity and would reuse stale slices if new WAV
  bytes later landed at the same path
- root cause: EpisodeAssembler wrote and closed the authoritative master but
  recorded only a first-kilobyte waveform tripwire in `audio_gates`.
  `render_driver` already consumed a full-file hash that no production owner
  produced, and a later `Ledger.save()` did not preserve an externally stamped
  `audio` section. The terminal mux owned the final video pointer but never
  stamped the re-resolved master path
- fix: EpisodeAssembler computes a streaming SHA-256 after the WAV header is
  closed and stamps it with `ledger_frozen=true` in the owned `audio` section.
  ProductionLedger now initializes and preserves that section. Hash receipt
  failure remains loud but nonterminal. The terminal mux stamps the resolved
  master path together with the final video and OBS pointers after successful
  publication
- verify idea: hash a multi-chunk closed WAV and require the full digest to
  survive a later ProductionLedger merge; require per-beat slice keys to bind
  that digest, and require the final ledger's audio path to exist. A simulated
  hash failure must not erase or relabel an otherwise usable master asset
- bible-worthy: yes -- BUG-12.55
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-03 -- ledger save rebuilt the exact OBS deliverable into a nonexistent alias
- surfaced: same completed `media_archive` qualification artifact as
  PBUG-20260721-01
- symptom: `final_video_path` named the existing archival final and
  `meta.obs_final_path` named the existing playable OBS copy, but
  `meta.paths.obs_final` named a shorter nonexistent
  `<episode_id>.mp4`. Two official ledger surfaces therefore disagreed after a
  successful publish
- root cause: the terminal mux stamped the exact OBS path and then called the
  shared ledger owner. Every save unconditionally rebuilt `meta.paths` from a
  pre-publication filename plan, discarding the terminal publisher's enriched
  caption/credits/final filename. The planned alias outranked the observed
  artifact
- fix: the ledger owner accepts a terminal published OBS path only when it is
  an existing MP4 for the current episode under the inferred canonical OBS
  root or the explicit `OTR_OBS_DIR`. That validated exact path drives both
  `meta.obs_final_path` and `meta.paths.obs_final`; before publication the
  historical planned path remains only a plan. The mux stamps all terminal
  asset pointers in one owner-layer save
- verify idea: publish an enriched filename and require every final path
  surface to name an existing asset after save. Reject a missing path, wrong
  episode prefix, wrong extension, or path outside authorized OBS roots, then
  verify a later ProductionLedger save cannot regress the accepted filename
- bible-worthy: yes -- BUG-12.56
- status: **ROOT-FIXED / FOCUSED-GREEN; live requalification pending**

## PBUG-20260721-04 -- the post-audio ledger owner never reached the video wire
- surfaced: canonical `media_archive` requalification prompt
  `7a6618ec-dd00-4711-93c6-43573d5e9580`, episode renamed to
  `signal_lost_the_municipal_ledger_20260721_020231`, 2026-07-21. The run was
  stopped before publication at the first repeated render warning, per the
  cross-bank qualification protocol
- symptom: the closed master WAV and its disk ledger carried full SHA-256
  `2f8f4a196c28343d28904f4381ca1632c66f6ff00fef79307ef2c564dc217e93`,
  but the exact `VideoRenderBatch` input capture had `audio={}`. Every shot
  therefore warned that `_slice_master_audio` had been called without the
  master content hash and built an under-invalidated slice identity. The same
  capture omitted disk-only `audio_gates` and `transitions`
- root cause: `OTR_ShotLock` is the graph's intended post-audio join, but
  `overlay_audio_timing` copied only missing row-local timing/WAV fields from
  the newest ledger. It returned before reading disk whenever any wire row
  already had a timing hint, never copied the producer-owned top-level audio
  state, and selected by newest mtime without proving episode identity. The
  freeze-cascade wire is intentionally pre-audio, so the canonical graph could
  be correctly gated yet still deliver an empty audio section to every image
  and video consumer
- fix: ShotLock now resolves the active ledger through
  `in_flight_ledger_path`, proves same-episode identity with the immutable Phase
  10 `meta.freeze_timestamp` (which survives pending-to-final rename) or an
  exact non-empty episode id for older ledgers, and rejects mismatches loudly.
  It always visits the post-audio owner despite existing row timing. Matching
  disk truth replaces the complete producer-owned `audio` section, carries
  `audio_gates`, `transitions`, and `radio_bookend_path`, additively fills empty
  metadata, then performs the established missing-only row merge. The image
  dispatcher remains a wire-preserving consumer; no workflow link/widget/node
  change is required
- verify idea: give ShotLock a pending-id wire with existing timing and a
  renamed disk ledger sharing the same freeze timestamp; require disk's full
  master hash and post-audio sections to survive ShotLock and ImageDispatcher
  JSON serialization while populated writer metadata remains unchanged. Give
  it a different freeze timestamp and require no field to cross the boundary
- bible-worthy: yes -- BUG-12.57
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-05 -- split dialogue rows replaced accepted beat identity with synthetic child ids
- surfaced: canonical `media_archive` requalification prompt
  `c96e268d-0b8a-4bb5-8e10-2aacb8459680`, episode
  `signal_lost_soot_and_signature_20260721_025707`, 2026-07-21. The run was
  stopped during video before publication when the final ledger's deterministic
  consistency receipt reported nine beat-id defects
- symptom: split rows had unique child ids such as `b003_s1`, but both
  `line_id` and `beat_id` were set to that synthetic id. The accepted outline
  owned only `b001` through `b009`, so every child appeared to reference a
  nonexistent beat. The same legacy ledger had empty top-level `beats[]`,
  leaving no durable parent-to-line membership even for unsplit rows
- root cause: `_clone_voiced_row` treated line identity and narrative beat
  identity as one namespace. `production_ledger.init_lines_from_outline`
  initialized only `lines[]`, despite already owning the accepted outline beat
  set, and structural apply never refreshed denormalized beat membership after
  split/cut/merge operations. Downstream render stages correctly key shots by
  unique `line_id`; changing those consumers to parent beat ids would collapse
  sibling split rows and was rejected
- fix: split children now mint only a unique `line_id` and retain the exact
  parent `beat_id`. Outline initialization materializes the accepted top-level
  beat collection with initial `line_ids`; every structural apply rebuilds
  only those retained beats' final exact line membership, leaving a fully cut
  accepted beat present with `line_ids=[]`. Repeated repair passes allocate the
  first free child suffix across the ledger, so a second split cannot reuse an
  existing child line id. Structural telemetry is line-named,
  with deprecated beat-named aliases carrying the same unique line ids for
  compatibility. ShotLock, TTS, timing, image/video, captions, credits,
  readiness, hashes, and OBS retain their existing unique-line consumer keys
- verify idea: split one accepted beat and cut another; require two unique
  child line ids mapped to the first parent beat, the cut parent retained with
  an empty list, a clean outline/ledger consistency result, and no collapse at
  line-keyed consumers. Split the same parent again and require a new unique
  child id. Initialize directly from an outline and require
  `beats[].line_ids` to match the initial lines before any timing stage
- bible-worthy: yes -- BUG-12.58
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + offline self-test + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-06 -- the radio editor overrode a requested 180-word story with a hard-coded 350-word target
- surfaced: same stopped canonical `media_archive` run as
  PBUG-20260721-05. The writer produced a good 148-character-word body plus 67
  announcer words for the requested 180-word episode, but the editor declared
  it short and expanded it. The final receipt reported 252 character words
  (`actual_ratio=1.4`, advisory drift) and 316 total spoken words
- symptom: a story already inside the live receipt's `[0.7, 1.3]` band was
  needlessly sent through length normalization. The model claimed an in-range
  `projected_word_total`, but deterministic application produced a different
  total and still passed. Separately, a row-local quality repair could be
  rejected solely because the whole episode carried advisory word drift
- root cause: `_otr_radio_editor` hard-coded 350 +/-20%, counted announcer
  overhead in the episode target despite the writer's character-only contract,
  and validated the LLM's arithmetic claim rather than the plan's applied
  result. Its shared validator also made global length conformance a
  prerequisite for unrelated micro repair
- fix: the live `meta.word_budget` receipt is now the single authority: a
  positive target plus its two ratio multipliers. Only an absent pre-receipt
  ledger uses the historical 350/[0.8,1.2] fallback; a malformed present
  receipt records `SKIPPED_INVALID_BUDGET` without an LLM call or mutation.
  Budget accounting counts non-skipped character rows only, while every
  character and announcer row still owns the one-breath cap. Length-plan
  validation deep-copies the ledger, applies the proposed edit
  deterministically, and gates on the resulting character total; the model's
  projection is retained as forensic evidence. Micro repair disables only the
  global band check and retains structural, noun, line-cap, anchor, action, and
  row-scope guards. The two content-owned sci-fi routes keep their independent
  budget/seal contracts
- verify idea: at target 180, require 148 character words plus 67 announcer
  words to skip normalization, but an over-cap announcer row to trigger it.
  Accept a good simulated result despite a false model projection and reject a
  bad result despite a claimed 180. Permit a scoped repair during advisory
  episode drift while still rejecting an over-cap replacement; prove malformed
  present receipts make no mutation
- bible-worthy: yes -- BUG-12.59
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + offline self-test + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-07 -- the protected news-coda fact bypassed final spoken-surface validation
- surfaced: completed and published canonical `media_archive` qualification
  prompt `f7bffc53-bada-45c1-9256-4a27a3caed51`, episode
  `signal_lost_the_diary_keys_20260721_040039`, 2026-07-21
- symptom: strict audit rejected exact TTS coda row `b009`. The canonical row
  contained all three episode anchors and expanded to 44 delivery words after
  normalization, producing `anchor_stuffing` and `one_breath`, even though the
  bridge itself had passed hygiene. The Phase 7 record reported a failed row
  count but exposed no corresponding failure receipt
- root cause: both first-pass composition and the later shared spoken scour
  validated only the authored bridge, then reattached the protected factual
  suffix without rescoring the assembled surface TTS would consume. The
  composer could also character-cut the factual suffix, manufacturing a false
  sentence boundary. Phase 7 built failure detail by filtering the successful
  repair receipts, so row-local failure evidence was always empty
- fix: one shared coda finalizer now assembles the bridge with the exact source
  fact, projects it through the authoritative delivery normalizer, and scores
  the complete spoken row. A dirty full fact may reduce only to the longest
  clean exact complete-sentence prefix; if no such prefix exists, the spoken
  row points truthfully to credits while the full source note remains in
  `meta.news.news_close_brief`. Models never receive or rewrite factual prose,
  and no character truncation is permitted. First-pass composition receives
  injected canon and the live breath range; later scour stamps only hash
  receipts for any reduction. The mutator itself refuses both content-owned
  sci-fi policies, preserving their accepted rows, seals, and hashes. Phase 7
  now carries explicit row-local failure details
- verify idea: replay the exact live `b009` note and require its projected TTS
  surface to pass the final row scorer without exposing the fact to any model.
  Exercise multi-sentence, initials/version, and single-sentence facts: permit
  only whole exact sentences or a truthful credits deferral, never fragments.
  Require the same behavior in the `media_archive`, `public_domain`, and
  `shakespeare` legacy routes; keep `original` empty-coda behavior and direct
  `scifi_news`/`scifi_news_pro` shared-scour inputs byte-identical. A forced
  row-local failure must appear in the Phase 7 receipt
- bible-worthy: yes -- BUG-12.60
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + cross-bank + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-08 -- generic cast roles were mistaken for forbidden character names
- surfaced: canonical `public_domain` qualification episode
  `signal_lost_inheritance_of_desolation_20260721_060315`, 2026-07-21. The
  cast contained `THE TRAVELER` and `THE WITNESSES`; the technical story-brief
  model returned `A weary traveler faces a skeptical assembly...`. An older
  canonical public-domain 720-word production log reproduced the same class
  with `THE SCIENTIST` and `A scientist...`
- symptom: the story-brief content gate reported `named_character`, sent a
  repair that rejected the ordinary role noun, received the same truthful role
  again, and exhausted to the explicit failed sentinel. `ShotLock` and the LTX
  scene opener then received blank story-brief metadata (`status=failed`,
  `0/0/0`) and used only their non-authoring visual defaults. The episode could
  continue mechanically, but it was not a clean configured image/video brief
- root cause: one lexical splitter served two incompatible jobs. It treated
  every word in a cast label as both an input-anonymization alias and a
  forbidden output name. Thus generic identity labels such as `THE TRAVELER`
  made `traveler` illegal and even mapped the article `the` as if it were a
  person. The validator could identify only the broad reason code, so repair
  was not told which surface triggered the rejection
- fix: a bounded shared cast-identity grammar now distinguishes generic roles
  from personal names. Input anonymization maps a generic full label and its
  role noun to one stable identity, never an article; output validation permits
  those generic role forms. Personal labels still protect the full name plus
  meaningful components while excluding articles and honorifics as standalone
  tokens. The public reason code remains stable, while the private repair seam
  receives the exact matched surface and asks for environment, light, color,
  texture, space, material, weather, or objects. Genuine exhaustion still
  returns the observable failed sentinel, and downstream deterministic visual
  defaults remain non-authoring
- verify idea: replay the live `THE TRAVELER`/`A weary traveler...` case in one
  call and require a successful brief. Exercise article-bearing, ordinal, and
  multiword roles such as `First Witch`; Unicode, hyphenated, apostrophe, and
  honorific-bearing personal names; and representative personal-name shapes
  from all six banks. Assert that input substitution preserves one identity,
  ordinary `the` survives, real names remain forbidden, private repair names
  the exact surface while the public code stays `named_character`, and a
  genuine failed brief still produces a valid non-authoring downstream visual
  prompt
- bible-worthy: yes -- BUG-12.61
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + cross-bank + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-09 -- quality retakes repeatedly requested an artifact that could not fit the real context
- surfaced: canonical `scifi_news` qualification prompt
  `e67869e2-6ed5-43d2-b522-094e96ea94c0`, source article
  `3 Questions: Neural transparency and the future of AI design`, 2026-07-21.
  The run was stopped before ledger assembly after the third identical P7
  capacity cycle
- symptom: P7 serialized the complete score, prior `ScriptArtifactV4`, review,
  and complete result schema into an approximately 6,182-token prompt, then
  requested 2,970 output tokens from an 8,192-token local context. The
  transport reduced that output to approximately 2,010 tokens, truncating the
  complete artifact. Typed repair expanded the prompt to approximately 7,501
  tokens and left only 691 output tokens. The quality loop restored the
  unchanged prior script, re-audited it, and repeated the same mathematically
  impossible work. Roughly forty minutes elapsed with zero ledger rows; the
  GPU was busy generating, not video-rendering or memory-thrashing
- root cause: P7/P9 shared P5's whole-artifact schema, post-validator, retry
  ladder, and dynamic script budget even though quality findings owned only
  line text. Shared context fitting treated every output request as a
  reducible ceiling and had no signal that a bounded patch must arrive whole.
  Provider wrappers hid the capacity type behind backend errors, and the
  quality loop continued after restoring an unchanged artifact
- fix: P5 is now the only complete `ScriptArtifactV4` pass. P7/P9 derive a
  closed write set from valid finding line IDs (null means all voiced rows;
  invented IDs are discarded), request a compact typed line-text patch, merge
  only `line.text`, and run the complete script validator. A successful merge
  always returns to a fresh P6/P8 judgment. Malformed creative output gets one
  colder technical-slot attempt; two failures keep the best valid script and
  stop without rejudging unchanged input. A full-output marker is captured
  before normalization and enforced by writer-local, model-loader/polish,
  OpenRouter, Comfy Credits, Google, and GGUF transports, including provider
  output caps. Proven capacity failure is a no-call quality floor. P6/P8 model
  or transport failure is advisory and cannot kill an already valid story.
  Final hashes, authorship receipt, ledger rows, readiness, media consumers,
  and OBS paths are still built only after quality converges or floors
- verify idea: replay the live `6182 + 2970 > 8192` arithmetic and require zero
  generation/network calls when the complete patch cannot fit. Exercise every
  backend and provider cap; prove unmarked calls retain ordinary clamping.
  Require exact target coverage, immutable non-text fields, full merged-script
  validation, creative-to-technical rotation, fresh rejudgment only after a
  successful merge, and no second audit after a two-slot failure. Assert P5 is
  the only complete-artifact pass and all six source-bank routes retain their
  existing ledger/media/OBS ownership
- bible-worthy: yes -- BUG-12.62
- status: **LIVE-ADMITTED / ROOT-FIXED; focused cross-backend + lane tests GREEN; full-suite, Bug Bible, workflow gates, and live requalification pending**

## PBUG-20260721-10 -- redundant JSON-schema constraints disabled every compact local repair
- surfaced: canonical `scifi_news` qualification prompt
  `c8277cf6-dbb8-41ec-bcc4-ac5671080022`, episode
  `signal_lost_the_fortress_of_reason_20260721_095038`, 2026-07-21
- symptom: the P5 spoken-hygiene repair and every P7/P9 quality patch failed
  before emitting one token with `LMFormatEnforcerException: String schema
  contains both a pattern and a min/max length`. Reusing the technical slot
  then failed through `NoneType.allowed_tokens`; the run fell to deterministic
  hygiene/quality floors even though both local models were available
- root cause: both compact patch row schemas declared `line_id` with exact regex
  `^l\d{3}$` and redundant `min_length=1,max_length=16`. LM Format Enforcer
  explicitly rejects that JSON-Schema combination. Its token enforcer caches
  an output state before allowed-token calculation, so the first schema
  exception can leave an incomplete cached state for the reused prefix
- fix: retain the exact regex as the sole line-id constraint in both patch
  schemas. Do not catch or suppress the formatter exception. Character-feed
  the production JSON for each complete patch schema through LMFE and retain
  Pydantic rejection coverage for wrong-prefix and wrong-length ids
- verify idea: drive valid P5-hygiene and P7/P9/P10 patch JSON one character at
  a time through `JsonSchemaParser`, require `can_end()`, assert neither line-id
  schema contains `minLength/maxLength`, and reject `l1000`/`a100`
- bible-worthy: yes -- BUG-12.63
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-11 -- content-owned sci-fi omitted and then ignored its real character-word contract
- surfaced: same canonical `scifi_news` artifact as PBUG-20260721-10
- symptom: a requested 180-word episode sealed only 143 character-story words
  (146 including announcer). `meta.word_budget` lacked the target and band, and
  the shared final stamp marked `actual_drift=false` under its broad global
  `0.7..1.3` tolerance. The other qualified banks landed at 166--184 character
  words, inside the operator's approximately 163--200 campaign window
- root cause: Scifi Codex treated its advisory P3 word blueprint as sufficient
  and had no deterministic post-hygiene word-fit owner. Its final hygiene floor
  could shorten rows after the taste/factual loops. Separately, the shared tail
  read a producer target but always judged actual drift against global
  constants, ignoring a producer-stamped band even when one existed
- fix: Scifi Codex stamps an inclusive target-relative character-word contract
  before mutation. After every quality pass and the final hygiene scour, a
  bounded P10 compact line patch extends or compresses only selected character
  rows, runs the full merged-script graph/hygiene validator, and gets a fresh
  deterministic recount. Creative then technical attempts continue under a
  finite dynamic budget; exhaustion keeps the closest valid artifact with a
  truthful floor/drift receipt. Only then are ledger rows, counts, authorship
  hashes, and seals minted. The shared final stamp honors a valid producer band
  and uses `0.7..1.3` only as the legacy fallback. Announcer overhead remains
  separate
- verify idea: require target 180 to resolve to the relative 163--200 integer
  window, repair a 15-word character artifact into the 30-word window with a
  compact patch and fresh recount, then exhaust both slots and require the
  original valid story plus explicit drift. Prove the shared receipt consumes
  valid producer ratios and rejects malformed/reversed bands to legacy fallback
- bible-worthy: extends BUG-12.59
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-12 -- a real zero-second dialogue onset was reported as missing timing
- surfaced: same canonical `scifi_news` artifact as PBUG-20260721-10. ShotLock
  successfully overlaid eleven timed rows; the first spoken row had
  `start_s=0.0` and positive duration
- symptom: video logged the BUG-404 missing-overlay warning and ran the volume
  envelope fallback even though timing was present. That could manufacture an
  opening title window over immediate dialogue
- root cause: `_resolve_title_timing` correctly converted the onset to frame
  zero, then accepted it only when `first_dialogue_f > 0`. The valid zero
  sentinel shared the same branch as `None`
- fix: any non-`None` first-dialogue frame is known timing. Clamp a known onset
  to the nonnegative title window; zero yields no opening-card gap, preserves
  the zero receipt, and emits no missing-timing warning. Only `None` reaches the
  envelope fallback and BUG-404 diagnostic
- verify idea: pass a character row with `start_s=0.0` and require
  `first_dialogue_f=0`, no opening bounds, and no BUG-404 warning; retain the
  existing positive-head-gap test and missing-timing fallback
- bible-worthy: yes -- BUG-12.64
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible GREEN; live requalification pending**

## PBUG-20260721-13 -- rendered music had no durable ledger timeline or downstream mirrors
- surfaced: canonical `scifi_news` artifact
  `signal_lost_the_fortress_of_reason_20260721_095038`, followed by a
  cross-bank inspection of the four already-qualified inline ledgers. The
  scifi ledger carried `music_open/music_inter/music_close` rows with
  `open/inter/close` placements and null path/timing. Each inline bank rendered
  three cue WAVs and audibly mixed its opening/closing bookends, but retained
  zero `music[]` rows, dropped the interstitial, and minted zero
  `mirrored_from=music` timeline rows
- symptom: the audio renderer could produce valid cue bytes while the durable
  ledger, video/title consumers, and OBS-bound wire ledger had no coherent
  account of which cue played where. A dialogue-anchored scifi interstitial was
  never inserted because SceneSequencer only resolved dedicated
  `music_inter` sentinel rows. Legacy manifest materialization in
  EpisodeAssembler would have been too late for SceneSequencer timing even if
  it had existed
- root cause: banks exposed different cue ID/placement dialects; four legacy
  producers authored only sentinel lines; the rendered cue manifest was not
  reconciled into the ledger before timeline mutation; SceneSequencer keyed
  interstitial timing only by sentinel anchor; EpisodeAssembler recognized
  bookends by cue ID instead of canonical placement; and ShotLock rehydrated
  lines/audio sections but not identity-gated music rows or assembler-owned
  mirrors
- fix: all content producers now cross the durable boundary using
  `opening/inter_NN/closing`, canonical placements, and explicit anchors.
  StableAudioTheme accepts historical aliases and gives synthesized legacy
  cues deterministic cue-spec identities plus ordered sentinel anchors. A
  shared manifest reconciler materializes or path-refreshes `music[]` before
  SceneSequencer writes timing and rejects any authored identity mismatch.
  SceneSequencer inserts interstitials before either a sentinel or ordinary
  dialogue anchor without consuming the dialogue's voice slot, then writes by
  cue ID. EpisodeAssembler idempotently reconciles, promotes even zero-offset
  scene timing, places bookends and chooses mirror roles by placement, and
  remains the sole mirror minter. ShotLock proves same-episode identity, lets
  disk win render-owned music fields only on a recomputed cue-spec match,
  appends valid legacy rows, and replaces only mirrors belonging to matched
  cues
- verify idea: cover canonical producer mappings and aliases; materialize a
  legacy manifest twice and require idempotence; reject a stale authored cue
  hash; insert an interstitial before ordinary dialogue while consuming every
  voice clip; position a sentinel cue; materialize/place/mirror all three cues
  in EpisodeAssembler; and require ShotLock to append valid legacy music and
  mirrors while rejecting a changed cue and its mirror
- bible-worthy: yes -- BUG-12.65
- status: **LIVE-ADMITTED / ROOT-FIXED; focused + full-suite + Bug Bible + workflow gates GREEN; live requalification pending**

## PBUG-20260721-14 -- title rename moved assets but stranded their ledger identity
- surfaced: canonical `scifi_news` episode
  `signal_lost_the_chemical_throne_20260721_121728`, 2026-07-21. The audio
  assembler rendered and timed three canonical cues, persisted three music
  mirrors, and produced a byte-sealed master before SignalLostVideo renamed
  `pending_20260721_112330` to the final title
- symptom: the final ledger lived under the correct renamed directory, but all
  three `music[].wav_path` values still pointed into the deleted pending
  directory. The cue files themselves had moved and existed under the final
  directory. A later singleton save reduced the durable line set from twelve
  authored rows plus three assembler mirrors back to twelve rows. ShotLock
  retained the stale wire episode id and path block, while image dispatch and
  master mux escaped only through newest-ledger rescues. The video rendered,
  but the ledger was not a truthful map of the assets that produced it
- root cause: `Ledger.rename_episode` treated directory movement and identity
  movement as different operations: it renamed bytes and the ledger filename
  without recursively rebasing episode-local absolute JSON values. The
  singleton merge iterated only rows already present in memory, so
  assembler-owned disk-only mirrors had no join. The post-audio wire join did
  not let a same-freeze durable final id and path block replace stale pending
  values. ShotLock had no graph dependency on rename completion, and consumer
  recovery chose the newest sibling ledger instead of the active owner
- fix: rename now recursively rebuilds every absolute JSON string contained by
  the old episode root, for both the durable ledger and singleton, and atomically
  saves durable truth before advancing in-memory identity. External paths and
  prefix siblings remain unchanged; retry after a partial move is idempotent
  and durable-save failure is loud. Singleton saves preserve only validated
  EpisodeAssembler music mirrors: same immutable freeze, unique matching cue,
  recomputed cue-spec hash, legal role, master-timeline coordinates, and unique
  line id. One-sided, mismatched, reauthored, malformed, and ordinary disk-only
  rows are rejected. ShotLock lets the proven durable final episode id,
  `meta.paths`, media sections, and terminal paths win. Canonical link 284 gates
  ShotLock on SignalLostVideo rename completion. Image, clip, and master-audio
  recovery use the active in-flight ledger with available freeze/directory
  identity checks; no newest-mtime sibling selection remains
- verify idea: parameterize the rename over all six source banks. Move real cue,
  clip, master, path-block, and nested receipt values; require every episode
  pointer to resolve under the final root while an external path stays exact.
  Require freeze/readiness/authorship/master/cue hashes to remain unchanged and
  assembler mirrors to survive a later singleton save. Reject foreign or
  one-sided freezes, reauthored cues, malformed mirrors, ordinary disk-only
  dialogue, and mismatched consumer directory identities. Validate canonical
  link 284 and live-qualify both sci-fi banks through terminal OBS publication
- bible-worthy: yes -- BUG-12.66
- status: **LIVE-ADMITTED / ROOT-FIXED; focused cross-bank + consumer tests, full-suite (8,297 passed), Bug Bible, and canonical workflow gates GREEN; live `scifi_news` + `scifi_news_pro` requalification pending**

## PBUG-20260721-15 -- split word-count ownership falsely dirtied a correct live ledger
- surfaced: canonical `scifi_news` qualification prompt
  `1435f170-78fa-45ec-81a7-779b44533eb7`, pending artifact
  `pending_20260721_131620`, 2026-07-21. The source was the MIT News article
  `Study finds cell memory can be more like a dimmer dial than an on/off
  switch`. The run was stopped at TTS immediately after the freeze warning
- symptom: line `l003` contained the canonical surface `'off'—it's`. Its stored
  regex-derived `word_count=21` was correct, but Phase 0 and Phase 10 counted
  whitespace fields and reported only 20. The ledger froze
  `frozen_with_warns`; root `total_word_count=186` disagreed with the durable
  meta total `185`. The text, character count, and authored hashes were valid
- root cause: derived ledger metrics had several owners. Production row
  writers used an ASCII word regex, Scifi Codex used a slightly different
  smart-apostrophe regex, readiness/freeze/meta stamps used `str.split()`, and
  multiple repair/scrub/editor paths changed `text` without atomically
  refreshing both counts. Save aggregated stored fields instead of deriving
  them from canonical text, so stale or merely differently-tokenized values
  could survive into cast, scene, root, and budget receipts on every bank
- fix: one stdlib-only text-metrics leaf now owns character and word boundaries:
  ASCII hyphens plus straight/smart apostrophes remain intra-word, while en/em
  dashes are boundaries. Every confirmed durable text mutator calls the atomic
  text/count setter. Production save re-derives every row before rolling up
  cast, scene, root, and character/announcer meta totals, clearing stale zero-
  line aggregates. The freeze cascade preserves the raw Phase-0 diagnosis,
  then performs one count-only refresh after all permitted text mutation and
  before Phase 10. It does not alter canonical text, `text_for_tts`, hashes,
  authorship receipts, or seals. The freeze audit consumes the same helper
- verify idea: pin straight/smart apostrophe, ASCII-hyphen, en-dash, and em-dash
  counts including the exact live sentence. Parameterize all six banks through
  a save with deliberately corrupted row/root/cast/scene/meta counts and
  require complete self-healing. Show Phase 0 retaining an incoming mismatch
  while Phase 10 is clean after the final refresh. AST-audit production nodes
  so direct `row['text']` writes cannot bypass the atomic owner
- bible-worthy: yes -- BUG-12.67
- status: **LIVE-ADMITTED / ROOT-FIXED; exact six-bank + writer/freeze focused tests, full suite (8,315 passed), Bug Bible, and canonical workflow gates GREEN; live six-bank requalification pending**

## PBUG-20260721-16 -- whole-artifact P5 transport exhausted an otherwise healthy local writer
- surfaced: canonical `scifi_news` qualification prompt
  `569b20e5-0e28-4472-a04d-637ab019f19f`, pending artifact
  `pending_20260721_144919`, 2026-07-21. The source was the NASA NISAR /
  Hummingbird Antarctica item. The episode stopped in P5 before ledger or media
  production after 39 minutes of active local inference
- symptom: P5 attempt one reached its 2,970-token caller cap and returned
  truncated JSON. Attempt two returned a complete `ScriptArtifactV4` but
  invented line ID `l013`. The full typed-repair prompt then occupied 5,807
  tokens; the 8,192-token local context could reserve only 2,385 of the required
  2,970 output tokens, so the repair truncated and the three-attempt ladder
  exhausted. No OOM or idle GPU thrash occurred, but a structurally repairable
  story killed the episode
- root cause: the initial script pass made the model reserialize the accepted
  score's title, scene, cue, speaker, graph, boundary, fact, and neutral delivery
  metadata beside the only fields it actually authored: line IDs and spoken
  text. Its repair turn then reinjected the failed whole artifact plus almost
  the whole original request and duplicate schema authority. Output and context
  budgets therefore scaled with compiler-owned metadata, and a fresh LLM ladder
  could only retry the same oversized transport
- fix: P5 now transports a strict compact `ScriptTextDraftV4` containing only
  `{line_id,text}` rows. Python requires an exact unique bijection to the
  accepted line graph, maps by ID rather than response position, and compiles
  every mechanical `ScriptArtifactV4` field from the accepted score before the
  unchanged full graph, roster, fact, spoken-hygiene, and craft validation.
  Typed repair carries only the compact draft plus story, line-graph, fact, and
  word-steer authority; malformed prefixes are omitted. Every P5 call requires
  the complete prompt and full dynamic output reservation to fit. Exhaustion is
  a flat, truthful creative ladder followed by at most one fresh technical
  ladder, never recursion; the existing row-local A/B/C/deterministic spoken
  floor remains the final craft-liveness boundary
- verify idea: at the full supported 900-word, 24-row surface, tokenize the real
  base and semantic-repair chats with the exact on-disk Gemma 4 12B tokenizer
  and require prompt plus the full 3,208-token output reservation to fit its
  8,192-token context. Require byte-preserved text and compiler-owned graph
  fields; reject missing, unknown, and duplicate IDs; prove typed repair omits
  whole-request/schema echo and malformed raw prefixes; and prove the restart
  runs creative then technical exactly once before truthful exhaustion
- bible-worthy: yes -- BUG-12.68
- status: **LIVE-ADMITTED / ROOT-FIXED; exact-tokenizer maximum envelope, 165 focused lane/route tests, full suite (8,325 passed), Bug Bible, and canonical workflow gates GREEN; live six-bank requalification pending**

## PBUG-20260721-17 -- positioned video double-counted two audio crossfades at terminal mux
- surfaced: canonical `scifi_news` qualification prompt
  `a5e6e996-8f1e-4eb4-aff2-29486d4fd28c`, episode
  `signal_lost_the_fire_ant_bridge_20260721_163825`, 2026-07-21. The compact P5
  path passed on its first attempt and the run completed story, TTS, music,
  fifteen shots, silent composition, captions, and credits before the terminal
  master-audio mux rejected the body video. No OBS artifact was published
- symptom: the master audio was 114.5433 seconds / 5,498,077 samples at 48 kHz,
  while the silent body was 115.5600 seconds / 2,889 frames at 25 fps. With the
  valid 53.517-second credits declaration, video exceeded the allowed
  audio-plus-credits duration by 0.8997 seconds. The GPU remained around
  4.1--4.4 GB during the tail; this was deterministic timeline arithmetic, not
  VRAM thrashing
- root cause: the durable post-audio ledger correctly positioned the first
  drama row 0.5 seconds before the opening music ended and the closing music
  0.5 seconds before the last drama row ended. The render driver nevertheless
  defined final video length as the sum of every full per-shot render request,
  and the positioned planner emitted each full request even after a later row
  owned an earlier start boundary. The two intentional audio crossfades were
  therefore duplicated as one extra second / 25 visual frames. The filesystem
  master probe could grow the bad total but was forbidden to shrink it. The mux
  and credits declaration correctly refused to misclassify body drift as a
  credits tail
- fix: the clip manifest now separates full `render_target_frames` from the
  authoritative positioned `timeline_total_frames`. When every row has a
  position and the post-audio ledger owns `total_episode_dur_s`, the output
  boundary is `ceil(duration * fps)`; sparse legacy manifests retain their
  sequential sum. Positioned planning is stable by `(start_s, manifest order)`
  and gives each row only the visible interval ending at its requested end, the
  next row's start, or the timeline boundary, whichever comes first. This trims
  overlaps without stretching real gaps. QA reports requested, rendered,
  planned-visible, and overlap-trimmed frames separately. The actual master
  probe may reconcile a positioned total downward or upward, while sequential
  behavior remains grow-only. Terminal mux tolerance and credits ownership are
  unchanged
- sibling audit: exact Antigravity `gemini-3.5-flash-high` R2/R3 review in a
  clean detached worktree confirmed the shared tail affects all six banks.
  Sol grounded every claim against the real Windows files, discarded incorrect
  bank-specific and file-path claims, and retained sole coding/judgment
  authority. No workflow wiring change was required
- verify idea: build a positioned manifest whose full requests sum to 563
  frames but whose two crossfades and authoritative boundary yield 538 visible
  frames. Require stable slot ownership, no duplicated frames or stretched
  gaps, truthful overlap-trim telemetry, and a green visible-frame QA result.
  With real ffmpeg media, give a positioned 80-frame manifest a 2.1-second
  master and require exactly `ceil(2.1 * 25) = 53` output frames. Retain a
  legacy no-position manifest that preserves full sequential requests
- bible-worthy: yes -- BUG-12.69
- status: **LIVE-ADMITTED / ROOT-FIXED / PUSHED at `651118ef`; exact failed-ledger replay, 96 focused CPU/ffmpeg tests, full suite (8,328 passed), Bug Bible, and canonical workflow gates GREEN; resumed qualification surfaced PBUG-20260721-18 before media**

## PBUG-20260721-18 -- requested story length escaped every producer as advisory drift
- surfaced: canonical scifi_news qualification prompt
  f62c1177-a40a-4f9e-a9ac-f9c3bcfad717, pending episode
  pending_20260721_172001, 2026-07-21. The selected Ars Technica source was
  Let Tom Hiddleston be your guide to Pompeii's final day. The run completed
  P0/P3/P4/P3-rewrite/P5 and the dynamic quality passes, then was stopped before
  TTS/media when the writer logged the measurable delivery miss. The first root
  fix resumed qualification, but prompt 38d83284-49aa-48ba-aada-344b32f57110
  live-admitted the remaining liveness defect after 41:38: scifi_codex exhausted
  17 row-local cycles at 206 words against the same 289..356 ledger band
- symptom: the first final composed body held only 190 words for an explicit
  320-word request. The writer labeled the miss ADVISORY ONLY, stamped drift,
  and entered story reflection. After strict delivery became fatal, the second
  live run correctly stopped before media, but it still failed the whole episode
  when one model candidate exhausted its local word-fit attempts
- root cause: every producer family originally had a different length escape.
  The first pass repaired the shared integer contract, row-local progress, and
  pre-media hard stop, but conflated candidate liveness with episode liveness.
  A finite per-candidate repair ladder could still raise out of Codex, Fable, or
  the four inline banks. Fable also counted with split() instead of the canonical
  ledger tokenizer, and append-only hygiene receipts could leave a repaired row
  marked row_failed_mechanical so assembly silently skipped it
- fix: one dependency-light contract owns target, inclusive bounds, producer,
  canonical character count, exact text hash, and the final receipt. A
  WordFitLivenessController permits unlimited strict progress but retires a
  candidate after four consecutive stalls. Producers escalate row repair to the
  alternate LLM, then author a fresh complete producer-owned candidate. There is
  deliberately no outer model-output retry ceiling: generation or provider
  exhaustion remains pending/retryable/non-ready until a deterministic ledger
  candidate passes or the operator cancels. Codex fresh P5 candidates alternate
  producer priority; Fable rerolls and reseals its complete P3/P5 proof surface;
  inline banks re-author a complete staged row set before committing it. Fable
  now uses canonical_word_count throughout and stores current line-id hygiene
  state rather than stale append-only failure history. All lanes reject filler,
  repetition, fake commercials/products, markup, unsupported numeric/visual/
  canon claims, and Python-authored story padding. Subjective quality remains
  fail-open. No readiness or media consumer receives the candidate until the
  assembled ledger passes stamp_actual(require_in_band=True)
- live qualification continuation: prompt
  32b374e2-7c89-4d4a-bb8c-42e180571ecc remained queue-running beyond the
  temporary observer's two-hour wall clock and retired more than a dozen P5
  candidates without leaking a partial ledger downstream. It also proved that
  producer-priority alternation was not sufficient when both logical slots
  resolved to the same seeded Gemma backend: the two fixed P5 prompts replayed
  the same failures. Each complete reroll now carries a model-visible,
  monotonically unique candidate prompt nonce. The canonical observer accepts
  timeout zero as wait-until-terminal so monitoring cannot kill qualification
- sibling audit: exact Antigravity gemini-3.6-flash-high R2/R3 review covered all
  six banks in a clean worktree. Sol grounded the findings against the real
  Windows checkout, retained the candidate/episode liveness, canonical Fable
  count, and stale hygiene-state defects, and discarded the proposed five-reroll
  ceiling and soft ledger stamp because both violated the operator law. Hidden
  Opus produced no usable grounded findings and was discarded
- verify idea: pin 180 -> 163..200 and 320 -> 289..356; prove unlimited strict
  progress, four consecutive stalls per candidate, candidate retirement rather
  than episode failure, alternate-slot complete rerolls, and survival beyond
  five outer generation failures. Prove row-local immutability, canonical Fable
  counting, stale failure clearing, proof/hash resealing, final reflection
  ordering, fake-commercial/new-claim rejection, and hard downstream gating.
  Freeze must remain a read-only last backstop. Then qualify all six banks at
  320 words through audio, video, captions, credits, mux, and OBS publication
- bible-worthy: yes -- BUG-12.70
- status: **LIVE-ADMITTED / ROOT-FIXED / OFFLINE-GREEN IN WORKTREE; 177
  focused producer tests, full suite (8,348 passed / 33 skipped / 1 expected
  failure), BUG-12.70 (17 passed / 23 route-local skips / 3 expected failures),
  and canonical workflow gates (48 passed; byte-identical SHA-256
  f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a) GREEN;
  commit/push and six-bank 320-word OBS qualification pending**

## PBUG-20260722-01 -- scifi_news codex seam lookup bypassed prompt_stages
- surfaced: canonical six-bank sweep `six_bank_sweep_20260722_162943_317`,
  `scifi_news` at 120 words, 2026-07-22
- symptom: the episode failed before model inference with
  `CodexPackContractError: P0 missing nonempty prompt seam
  'codex_fact_index_system'`; no episode or OBS asset was published
- root cause: the shared Codex-lane seam resolver used `getattr(pack, seam,
  None)` even though production seams are stored in the `StoryPack.prompt_stages`
  mapping. The resolver therefore returned `None` for every valid Codex seam.
- fix: the resolver now reads `pack.prompt_stages.get(seam)` and fails closed
  only when that mapped seam is absent or empty; a regression test proves a
  valid prompt-stage seam reaches the structured pass.
- verify idea: load every runnable bank's declared seams through its selected
  runner, assert the exact prompt text reaches the structured-call system
  message, and require the canonical `scifi_news` 120/320 legs to publish.
- bible-worthy: yes -- BUG-12.72; production pack seams must be accessed
  through the canonical mapping owner, not dataclass attribute guesses;
  executable regression coverage is present
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; post-fix scifi_news live
  requalification pending**

## PBUG-20260722-02 -- scifi_news P0 fact spans still exhausted the bounded repair ladder
- surfaced: fresh post-seam-fix focused qualification runs
  `six_bank_sweep_20260722_200609_509` (`scifi_news` at 120 words, prompt
  `59256a76-bd44-447e-88a2-fab5fe2c350f`) and
  `six_bank_sweep_20260722_201449_793` (`scifi_news` at 320 words, prompt
  `4b9f096b-3d8c-4c89-9f00-8924ad0e177c`), 2026-07-22
- symptom: P0 failed after two structured attempts on both distinct source
  payloads because `F01` returned a quote that did not equal the declared
  `full_text[start:end]` slice; no ledger or OBS asset was published. The
  320-word run was the same class after the prompt-stage seam fix and the
  120-word run also had no word/length gate involvement
- root cause: the accepted source payload is already normalized at admission
  and the literal-span validator is correctly fail-closed, but the technical
  model plus bounded repair still returned a non-literal or unrelated quote
  instead of repairing the exact field/start/end/quote identity. This extends
  the existing source-span contract defect covered by BUG-11.35; it is not a
  new whitespace-ingestion defect and not a prose/length gate
- fix: **KIBITZ-HARDENED / IMPLEMENTED IN WORKTREE**. The first mechanical fix
  is an explicit literal identity instruction
  `payload[field][start:end] == quote`. The shared structured-call boundary
  then gets one direct, bounded alternate repair owner with a hard context
  ceiling, original post-validator reuse, owner/backend/rung/nonce journal
  fields, and explicit terminal disposition. P0 now wires the creative owner
  through that one-shot branch. The remote RTX 4060 Qwen worker
  at `10.55.0.2:1234` completed the four scoped read-only reviews; the RTX
  5080 remains reserved for ComfyUI and must not load this worker. Any patch
  stays out of live qualification until the accepted-object boundary is
  proven against the captured production payloads
- verify idea: replay both captured P0 failures with the exact normalized A0
  payloads, require a repair whose quote is byte-identical to the selected
  slice, preserve the payload digest, and then requalify both canonical
  `scifi_news` legs through `RESULT SUCCESS`, `obs_publish OK`, and exact
  episode/OBS assets
- bible-worthy: extends BUG-11.35; no new portable rule
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; focused/canonical offline
  gates green, live 120/320 requalification pending**

## PBUG-20260723-01 -- six-bank campaign trusted exit 0 over canonical RESULT FAIL
- surfaced: live six-bank 120-word viz campaign
  `six_bank_viz_120_20260723_20260723_011138`, `scifi_news` leg, prompt
  `cde10c6d-3b70-4732-8179-4b18c8bcd933`, 2026-07-23
- symptom: the child stdout contained `[canonical-api] RESULT FAIL`, but the
  campaign receipt recorded `status=PASS`, `exit_code=0`, and `queue_empty=true`.
  The campaign therefore reported `6/6 PASS` despite a live P0 fact-span
  failure and no valid qualification evidence for that leg
- root cause: the PowerShell campaign wrapper inferred terminal success from a
  zero child exit code and an empty ComfyUI queue. It did not consume the
  canonical runner's explicit terminal result, so a contradictory `RESULT FAIL`
  was invisible to the receipt owner
- fix: the wrapper now delegates verdict construction to
  `scripts/otr_campaign_receipt.ps1`. A leg is PASS only when exit code is zero,
  the queue is empty, and the latest explicit terminal marker is
  `RESULT SUCCESS`; missing/contradictory markers are recorded as FAIL with the
  observed terminal line and reason
- verify idea: feed captured `RESULT FAIL` stdout with exit code zero, missing
  terminal output, `RESULT SUCCESS` with a nonzero exit, and a clean success
  through the helper; require truthful verdicts and nonzero failure exits
- bible-worthy: extends BUG-12.50's terminal-evidence contract; no new
  portable rule
- status: **LIVE-ADMITTED / ROOT-FIXED IN WORKTREE; helper regression GREEN;
  six-bank live requalification pending**

## PBUG-20260723-02 -- the 8GB Wan tier's low-VRAM launch contract never reached a production leg
- surfaced: 2026-07-23 overnight media qualification, matrix leg
  `wan_8gb__lumina_image__media_archive` (`model_coverage_wan/receipts.json` +
  `server_wan.log`; staged in `docs/2026-07-23-video-failure-inventory.md`)
- symptom: terminal `FAIL` at `OTR_VideoRenderBatch` -- `wan_ti2v` received a
  177-frame request while the cost model afforded 30 frames at the observed
  free VRAM. No silent resize happened, which is correct; the requested
  832x480 / 17-frame low-VRAM lane simply never applied to the leg
- root cause: the 17-frame ceiling existed only in the profile's
  `launch.env.OTR_WAN_TI2V_MAX_FRAMES`, and `eng_wan_ti2v._floor_length` read
  that env var as its ONLY channel. A production episode leg is submitted to an
  ALREADY-BOOTED server, so `launch.env` can never reach it -- any leg not
  booted through `scripts/otr_headless_canonical.ps1 -Profile otr_8gb_wan`
  inherited the 177-frame engine max. The profile's other declaration,
  `render.frame_budget: 17`, maps to `OTR_VideoRenderBatch.frame_count`, which
  is diagnostic-harness-only ("Ignored in mode=episode" per its own tooltip),
  so the tier's contract was inert in production on both channels
- fix: new OPTIONAL profile key `video.max_render_frames` (0/absent =
  unpinned) travelling the same proven channel the device/dtype policy uses --
  profile -> `OTR_VideoDirector.max_render_frames` widget (appended LAST,
  canonical ships 0) -> v2 policy -> ShotLock ledger `video` section ->
  `render_driver.build_episode_render_policy` -> `MotionEngineBase.prepare` ->
  `_floor_length`. Env pin still outranks it; every other tier omits the key
  and is byte-for-byte unchanged. Beat frame targets are untouched: the ceiling
  caps what the ENGINE renders (then ping-pong-extended to the beat's full
  audio length), never what the episode plays
- verify idea: with free VRAM affording ~30 frames at 832x480, a 177-frame beat
  must raise `MotionBudgetError` UNPINNED and return 17 with the tier ceiling
  on the ledger; and an unpinned tier must still return 177 (no lane capped by
  the fix). Covered by `tests/test_remaining_video_contracts.py`
- UPDATE 2026-08-13: the UNPINNED half of that verify idea no longer holds, and
  the change is deliberate. `compute_real_frame_budget` now refuses only for a
  row in `QUALIFIED_COST_ROWS`, which is empty, so the unpinned 177-frame beat
  RETURNS 177 instead of raising. The cost row that produced the original
  refusal is the one this repo disqualified in writing, and it had been
  refusing through this path alone because `_floor_length` never consulted the
  qualification authority that `render_driver._assert_beat_affordable` always
  did -- it killed two live 45-word render-gate legs (`fastwan_8gb` at 69
  frames, `wan_ti2v` at 125) before that was spotted. The tier-ceiling claim --
  the actual subject of this PBUG -- is untouched and still verified: the test
  in `tests/test_remaining_video_contracts.py` now QUALIFIES the row so the
  ceiling still has a refusal to be measured against
- UPDATE 2026-07-27 (B3): the "then ping-pong-extended to the beat's full audio
  length" clause above is still exactly right for WAN and is now only HALF the
  meaning of `video.max_render_frames`. For `ltx_8gb` -- the sole member of
  `frame_contract.PLANNING_CAP_ENGINES` -- the same ledger key is a coverage
  PLANNING cap: it narrows the contract `otr_shot_lock._stamp_coverage_plan`
  partitions against, so the beat is covered by real chained clips of at most
  that length instead of one short render padded out. WAN is deliberately
  excluded, because applying it before `partition_beat()` would turn every WAN
  beat into a pile of 17-frame renders and undo this very fix. Anyone reading
  this entry as the definition of the key should read
  `docs/2026-07-27-b3-qa-findings.md` and `frame_contract.effective_frame_contract`
  alongside it
- UPDATE 2026-07-27 (B6): SECOND application of this entry's portable rule to
  the same tier, with the OPPOSITE remedy, because there was no channel to fix.
  B3 gave the ltx_8gb CEILING a profile -> ledger channel. The tier's RECIPE --
  T5 device, tiled decode, the sampling knobs, the negative conditioning, the
  tile geometry -- has no channel at all: the profile schema accepts only
  `device_policy`, `dtype_policy` and `max_render_frames`, and
  `otr_8gb_ltx.json`'s `launch.env` is `{}`. So the recipe is now FROZEN IN
  CODE (`eng_ltx_8gb.LTX8_RECIPE_V1`); those env vars bind only under an
  explicit `OTR_LTX_8GB_PREQUALIFICATION` consent act, and a run that sets it
  stamps a `+prequalification` recipe receipt so a measurement artifact is
  never mistaken for a production one in `meta.render_engines`.
  A NEW TESTABLE COROLLARY this produced, which the original entry does not
  state: **a knob that cannot bind must be IGNORED, never FATAL.** The first
  draft parsed the demoted vars before discarding them, which meant a stale
  `OTR_LTX_8GB_STEPS=not-a-number` in a long-booted server's environment would
  raise MALFORMED_CONFIG and kill a leg over a value with no effect on it --
  the same defect wearing the opposite mask. Presence is named in a warning;
  nothing outside the consent act is parsed
- bible-worthy: yes -- the portable rule is that a contract declared only in a
  process-launch environment cannot bind work submitted to an already-running
  server; a per-tier constraint has to ride the artifact the run loads. B6 adds
  the corollary above (ignore, never fail, on a knob that cannot bind) and the
  receipt rule (a run under a consent act must mark its own artifacts)
- status: **ROOT-FIXED + suite/contract GREEN; live 8GB requalification leg
  still owed (no GPU run authorized in this window)**

## PBUG-20260729-01 -- P5 markup defect hid behind the compile refusal, and the one repair shot died on it
- surfaced: the live 45-word campaign, leg `ltx_8gb` (2026-07-29 06:46), headless
  canonical run. `P5 failed: ... disposition=primary_ladder_exhausted; last
  error -> PostValidationError: l001: spoken text is production markup`
- symptom: the writer died before any video engine ran. Attempt 1 (base) was
  told only "P5 compact draft line IDs do not exactly cover the accepted graph
  (missing=[], unknown=['l011','l012','l013'])". Attempt 2 (typed repair) did
  exactly what it was told -- dropped the three invented IDs -- and was then
  refused for `l001: spoken text is production markup`, a defect that was
  sitting in attempt 1's output and had never been mentioned. The ladder was
  spent: `structured_call` deliberately does not retry a repair that was
  schema-valid but content-invalid.
- root cause: the P5 post-validator surfaced ONE defect at a time.
  `compile_script_text_draft` raises on ID coverage before any markup check can
  run, so a compile refusal hid every markup defect behind it; and
  `_validate_p5_structure` returned on the first offending line, so even a
  clean-ID draft with three bad lines would have burned the shot on line one.
  A validator that reports serially is incompatible with a ladder that grants
  one repair attempt.
- fix: `3b49d3f8` -- `_validate_p5_structure` reports EVERY offending spoken
  line (a single finding still yields the bare historical message, so existing
  pins hold); and when `compile_script_text_draft` refuses, the RAW draft rows
  are scanned by the new `_p5_raw_spoken_findings` and those findings ride
  along with the compile refusal. Only rows the score owns and marks spoken are
  judged -- an invented ID has no speaker_role, and judging its text would be
  inventing a contract.
- verify idea: drive the P5 post_validator with a draft that BOTH misses the
  graph and speaks production markup, and assert the returned string names both
  defects. `tests/test_p5_repair_sees_every_defect.py` does this; mutation E9
  (the compile refusal drops the markup findings again) and E8 (the structure
  validator reports only the first bad line) both die against it.
- bible-worthy: yes -- the portable rule is that **a validator feeding a
  bounded repair budget must report every defect it can see in one pass.**
  Serial reporting silently converts an N-defect artifact into N required
  attempts, and any ladder shorter than N then fails for a reason that looks
  like a model problem and is actually a reporting problem.
- status: **ROOT-FIXED; suite/mutation GREEN; live requalification owed --
  `ltx_8gb` must be re-run and reach a video engine**

## PBUG-20260729-02 -- a degenerate P5 generation burns 24 minutes and bypasses the whole retry ladder
- surfaced: the live 45-word campaign, leg `ltx_audio_in` (2026-07-29 06:46 ->
  07:11, 1449s), headless canonical run
- symptom: `P5 failed: prose generation exhausted the full remaining
  provider/context capacity (14697 output tokens after a 1687-token prompt);
  the partial artifact is not eligible for a prose or structural reroll`
  (`PromptContextOverflowError`). One leg, 24 minutes, no video engine reached.
- root cause: TWO layers, and only the second is in doubt.
  (a) The model never stopped adding lines. `ScriptTextDraftV4.lines` declares
      `max_length=_SCRIPT_TEXT_DRAFT_MAX_LINES`, but that ceiling is the GLOBAL
      one (`_RADIO_SCORE_MAX_BEATS * _RADIO_SCORE_MAX_LINES_PER_BEAT`), not this
      episode's accepted line count, and the constrained decoder did not force
      the array closed at it. Nothing told the decoder the real ceiling. This is
      the same pathology as PBUG-20260729-01's `unknown=['l011','l012','l013']`,
      one step worse -- there the model invented three extra lines, here it
      never stopped. `repetition_penalty` was already at its gentle 1.03, so
      this is a constrained-decoding ceiling problem, not a sampling one.
  (b) The refusal is raised on attempt 1 of 3 and `PromptContextOverflowError`
      is a `RuntimeError`, which `structured_call` does not catch -- so a
      runaway consumes one attempt and then BYPASSES the remaining two rungs of
      a ladder that exists to absorb exactly this. The refusal's own text says
      the partial ARTIFACT is not eligible for a reroll, which is right; it does
      not say a fresh call at a lower temperature is ineligible, and the
      structural-retry rung (0.32) is the standard remedy for a degenerate loop.
- fix: **NOT FIXED.** Deliberately left open rather than changed unattended:
  every candidate touches a ratified fail-loud contract or the writer's
  sampling, and getting it wrong is worse than one lost leg the campaign
  watchdog already re-runs. Two candidates, for whoever picks this up:
    1. Bound the constrained decoder by the ACCEPTED line count for this
       episode rather than the global product ceiling, so a runaway is
       structurally impossible instead of merely caught. Preferred -- it
       removes the failure rather than recovering from it.
    2. Let a runaway under `_otr_reserve_remaining_output_capacity` advance the
       ladder instead of being terminal. Note the trap: the typed repair
       factory would be handed the ~14,700-token truncated output as
       `failed_output`, so the repair prompt itself must be bounded first.
  Do NOT "fix" this by capping P5's output budget to the word target -- THE LAW
  is explicit that requested word length and actual word count are telemetry
  only and may never reject or block an episode, and `output_budget_mode:
  "provider_capacity"` is that decision written down.
- verify idea: a fake slot_fn that returns exactly `effective_max_new_tokens`
  tokens without an EOS, asserting the pass makes a SECOND call at the
  structural-retry temperature instead of raising on the first.
- bible-worthy: yes -- the portable rule is that **a failure raised as a
  RuntimeError inside one rung of a retry ladder silently cancels the rungs
  below it.** Any bounded-retry design has to classify its own terminal errors
  explicitly, or the budget it advertises is not the budget it spends.
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `41683fc9` made a capacity failure advance the ladder instead of dying on attempt one of three, and `tests/test_a4_capacity_phase_advances_the_ladder.py` cites this PBUG number in its own docstring as the behaviour it pins.
- previous status: **OPEN -- diagnosed, not fixed. Live: 2 occurrences in 13 legs
- status: **CLOSED 2026-08-18 -- FIXED, with a regression test that names this PBUG**
  (`ltx_audio_in` 1450s, `still_word` 1420s -- both ~24 minutes of GPU time
  spent inside a single P5 base call before the refusal).** Both ran to the
  full remaining provider capacity (14697 and 14359 output tokens), which is
  the signature: the model never stops adding lines, and the array's declared
  `max_length` is the GLOBAL product ceiling rather than this episode's
  accepted line count, so nothing forces the array closed.
- **OPERATOR RULING 2026-07-29 (supersedes the "candidates" framing above):**
  "the writer should not be allowed to kill the run, it just needs to fix the
  ledger" -- restated: "the writer should never veto, the writers should keep
  on passing in a loop to agents to clean up the ledger." Candidate 2 is
  therefore the RULED DIRECTION, not an open design question, and the rule is
  general rather than runaway-specific: a writer pass failure must degrade to a
  workable ledger and never terminate the episode. The recorded trap still
  binds -- the typed-repair factory would be handed the ~14,700-token truncated
  output as `failed_output`, so the repair prompt must be bounded before that
  path opens -- and PBUG-20260729-03's hard-limit refusal is the same trap from
  the other side, so both belong in one design. THE LAW is untouched: word
  length stays telemetry. PARKED by the operator until the video pipe engines
  work as expected; queued as the next step in `docs/GO_FORWARD_PLAN.md`
  ("THE WRITER NEVER VETOES").

## PBUG-20260729-03 -- a P0 repair is refused for being too big to attempt
- surfaced: the live 45-word campaign, legs `mesh_stage` (07:23, 182s) and
  `still_flat` (07:26, 208s), headless canonical runs
- symptom: `P0 failed: P0 repair context is 16796 bytes, over the hard limit
  14336` -- and on `mesh_stage`, `P0 failed: [OTR_StructuredCall]
  'scifi_codex:P0' failed after 3 attempt(s); disposition=repair_owner_exhausted`.
  Two legs, no video engine reached.
- root cause: the repair context the pass BUILDS is larger than the bound it is
  allowed to spend, so the repair is refused before it is attempted rather than
  being trimmed to fit. The bound exists for a good reason (an unbounded repair
  prompt is how a context window gets eaten); what is missing is the step that
  makes the context fit it. Same family as PBUG-20260729-02: a budget that
  refuses instead of degrading, discovered only on live GPU time.
- fix: **NOT FIXED.** Folded into the operator's "writer never vetoes" ruling
  above and parked with it -- it is the same design, and fixing the two
  separately would produce two different answers to one question.
- verify idea: build a P0 repair context deliberately over the bound and assert
  the pass still returns a workable artifact, with a receipt naming what was
  trimmed -- rather than raising.
- bible-worthy: yes -- same portable rule as PBUG-20260729-02, seen from the
  other side: **a bound that refuses is not a budget, it is a veto.** A limit
  on a repair context has to come with the trim that makes the context fit it.
- **CORRECTION 2026-07-29 (this entry's first draft was wrong on two counts;
  the original text is kept below so the error is auditable).** A grounded read
  of the source established:
  1. **There are TWO checks with TWO different bounds, not one.** INNER bound
     **14336** in `compact_p0_repair_context`
     (`nodes/_otr_scifi_p0_contract.py:223-226`) -- and 14336 is not a constant
     at all, it is `max(1024, max_bytes - 2048)` computed at
     `nodes/_otr_scifi_codex.py:2253` from
     `P0_REPAIR_CONTEXT_MAX_BYTES = 16_384`. OUTER bound **16384** at
     `nodes/_otr_structured_call.py:1197-1201`, measured AFTER
     `_prompt_with_schema_contract` appends the schema instruction (`:1192`).
     `still_flat` (16796) hit the INNER check; `still_pan` (16735) hit the
     OUTER one. They are NOT the same failure, and the original claim that both
     sat "within 61 bytes of each other against a fixed 14336 bound" was an
     artifact of reading two different bounds as one.
  2. **`mesh_stage` and `viz_camera` are a DIFFERENT ROOT CAUSE and do not
     belong to this bug.** `disposition=repair_owner_exhausted` is set when
     `repair_attempted = True` (`nodes/_otr_structured_call.py:1180`) and the
     ALTERNATE owner's own output then fails validation -- meaning the repair
     context FIT, the alternate model ran, and its answer was rejected. That is
     model quality. `still_flat`/`still_pan` never reach the alternate model:
     their raw `ValueError`s skip the `except StructuredCallFailedError` clause
     (`nodes/_otr_scifi_codex.py:1685`) and land in the generic handler
     (`:1698-1712`), where the journal disposition resolves to a THIRD value,
     `repair_context_builder_failed` (`:1703-1706`).
  So this bug's live count is **2 occurrences, not 4**.
- **THE ACTUAL STRUCTURAL DEFECT, measured rather than inferred:** the reserve
  at `nodes/_otr_scifi_codex.py:2253` is a literal `2048`, but the overhead it
  is reserving for is `schema_shape_instruction(FactIndexV4)` at **3809 bytes**
  plus the fixed CRITICAL P0 REPAIR system text at **302 bytes** plus a join
  newline = **4112 bytes**. The reserve under-provisions by **2064 bytes**.
  Arithmetically certain: any inner render above ~12,272 bytes passes the inner
  check BY CONSTRUCTION and is then guaranteed to fail the outer one. A guessed
  literal reserve drifted out of sync with the thing it reserves for.
- **AND THE TRIM HELPERS ALREADY EXIST, UNWIRED.** `p0_source_char_budget`
  (`nodes/_otr_scifi_p0_contract.py:58-70`) and `p0_source_chunks` (`:72-121`)
  are defined, documented and exported, with **ZERO call sites** in the
  non-vendored codebase. `_p0_evidence_projection`
  (`nodes/_otr_scifi_codex.py:1104-1138`) dedupes by substring containment only
  and caps nothing, and `nodes/_otr_source_payload.py:317` collapses whitespace
  "without truncating authored source text" -- so a long RSS article body
  reaches the repair context unbounded. `failed_artifact` is likewise echoed
  untruncated, where the generic `default_repair_prompt_factory` truncates to
  `failed_output[:400]`.
- status: **OPEN -- diagnosed, parked under the 2026-07-29 operator ruling.
  Live: 2 occurrences in 17 legs** (`still_flat` 208s, inner check;
  `still_pan` 173s, outer check).
  ORIGINAL (WRONG) TEXT, kept for the record: "Live: 4 occurrences in 13 legs
  -- mesh_stage 182s and viz_camera 165s (disposition=repair_owner_exhausted),
  still_flat 208s and still_pan 173s. The two measured sizes land within 61
  bytes of each other against a fixed 14336 bound."
- **CAMPAIGN-WIDE RATE, 2026-07-29 (the number that should inform when this
  gets unparked):** across the first 13 legs of the live 45-word run, SEVEN
  died inside the writer -- 2x PBUG-...-02 and 4x PBUG-...-03 plus one
  PBUG-...-01 -- against 5 engine-side failures (all since fixed) and ONE leg
  that produced a finished episode. That is a **54% writer failure rate**, and
  it is the dominant blocker on the campaign, not the video engines. The narrow
  fix here (trim the repair context to fit its own bound) is SEPARABLE from the
  full "writer never vetoes" redesign and is the single highest-yield unblock
  available; it is not being taken because the operator parked writer work
  until the engines are proven, and that call stands until the operator changes
  it. Recorded here so whoever unparks it does not have to re-derive the cost.

## PBUG-20260801-01 -- the gemma row understated its own model by 32x, so the writer could never fit
- surfaced: the live 45-word campaign, every `otr_g4_*` leg, headless canonical
  runs. Six engines, zero episodes -- each leg died at `OTR_LedgerScriptWriter`
  before any video engine was reached.
- symptom: `GGUF unsloth/gemma-4-12b-it-GGUF cannot fit the complete requested
  output: requested_output=2800, provider_output_cap=512`, and when the context
  was raised to compensate, `effective n_ctx 8192 (from policy.gguf_n_ctx) is
  outside [512, 4096] for this row -- NO clamp`.
- root cause: TWO placeholder defaults, each below the pipeline's own contract.
  1. The catalog row declared `context_window=DEFAULT_CONTEXT_WINDOW` (4096)
     while the GGUF file itself declares `gemma4.context_length = 262144` -- the
     row understated the model by 32x. P0 needs `_P0_PROMPT_OVERHEAD_TOKENS`
     2600 + `_P0_BASE_OUTPUT_TOKENS` 2800 = 5400, so P0 was STRUCTURALLY
     impossible on this row: no setting could satisfy it.
  2. `DEFAULT_OUTPUT_TOKENS_CAP` was 512 against P0's 2800 request. The other
     backends never had this -- `_otr_comfy_backend` 8192, `_otr_openrouter_backend`
     16384. 512 was the outlier, not the rule.
- fix: row `context_window` 4096 -> 8192 (P0's own `_P0_LOCAL_CONTEXT_CAP`, the
  value its contract was written against, not a guess), and
  `DEFAULT_OUTPUT_TOKENS_CAP` 512 -> 4096 (bounded by the window: 8192 - 2600
  overhead = 5592 usable). Cost checked rather than assumed: KV at 0.7 GB/1k is
  5.60 GB, plus 6.63 GB of Q4_K_M weights = 12.23 GB, ~2.3 GB under the 14.5 GB
  tier ceiling. Commits 805123ea + 76c9f565.
- verified: `fastwan_8gb` 45-word canonical leg, RESULT SUCCESS in 2433 s,
  published 1920x1080 / 3036 frames / 121.44 s / AAC stereo, coverage 70.68 s
  audio vs 71.72 s video across 7 clips.
- **the part worth remembering:** between the first fix and the second, the ONLY
  thing keeping the writer alive was exporting `GEMMA4_12B_MAX_NEW_TOKENS=3072`
  at server boot. That is a dead channel of the PBUG-20260723-02 class -- the
  env binds at BOOT, so the next restart that forgets it silently restores the
  failure, and the symptom comes back looking like a NEW bug. A live pass that
  depends on remembering an export is not a fixed bug. The second boot was run
  deliberately WITHOUT the var to prove the default carries it alone.
- also worth remembering: two fixes failed before this one because both turned
  knobs that could not bind -- `n_ctx` when the limit was the output cap, then
  `n_ctx` past a ceiling the row would not allow. The row was never questioned
  until the GGUF metadata was read directly. **When two settings in a row fail
  to move a limit, stop tuning and go read what the artifact itself declares.**
- bible-worthy: yes -- **a placeholder default that sits below the caller's own
  contract is a structural refusal, not a configuration problem.** Any registry
  row describing a model's capacity must be derived from, or checked against,
  the artifact's declared metadata; a hand-set default silently caps a model at
  a fraction of what it can do, and no amount of caller-side tuning can reach
  past it.
- verify idea: assert every GGUF catalog row's `context_window` is <= the
  context length its own file declares AND >= what the P0 contract requires, so
  a row that cannot host the pipeline's own pass fails at test time rather than
  on live GPU minutes.

## PBUG-20260802-01 -- ltx_video declared 21 legal lengths for an engine that renders exactly one
- surfaced: the live 45-word campaign, leg `ltx_video` (2026-08-01 23:20, headless
  canonical run). Died at 11.8 minutes -- AFTER the writer, the cast, the TTS and
  the music had all been rendered and paid for. No obs asset.
- symptom: `RenderError: shot shot_music_opening_001 segment 1 rendered 169
  frame(s) but its plan asked for 89 (a surplus of 80). NO FALLBACK -- the plan's
  count is what this segment's audio slice was cut against, so assembling a
  segment of any other length makes the beat drift against its own audio.`
  Preceded in the same log by the engine's own warning:
  `[eng_ltx_video] frame ask 89 below the decode floor 169 -- raising`.
- root cause: the adapter's DECLARATION disagreed with its own RUNTIME.
  `frame_contract` declared `min_frames=9, max_frames=169, quantum=8` -- 21 legal
  rungs -- while `_ltx_frame_length` raises every ask below
  `_LTX_DECODE_FLOOR_DEFAULT` (169) up to it, and `_LTX_MAX_FRAMES_DEFAULT` is
  ALSO 169. The floor equals the cap, so the adapter emits exactly ONE length and
  20 of its 21 declared rungs do not exist. The planner believed the declaration,
  split a beat into 89-frame segments, and the engine could not produce them.
  The refusal was CORRECT; what it was checking against was wrong.
- why it read as a regression: `ltx_video` shipped for months in single-clip
  mode, where nothing ever asked it for a non-169 length. Only coverage planning
  (2026-07-25) can ask, so only coverage planning could expose it. The operator's
  "ltx_video always worked, check a week ago" was accurate.
- fix: declare the truth -- `min_frames=169, max_frames=169`, as LITERALS. Not
  derived from the constants: a FrameContract is STATIC because stills are minted
  against it before the render phase, so it must never track a value that can
  move underneath it (`test_the_LTX_ceilings_do_not_silently_follow_their_env_overrides`
  rejected a first draft that did exactly that). Commit 53fcebff.
  Then TWO more channels that could reintroduce it, both found by the kibitz
  panel (codex gpt-5.6-sol + antigravity), both closed:
  1. `assert_env_matches_contract` raises `ContractEnvConflict` when
     `OTR_LTX_MAX_FRAMES` / `OTR_LTX_MIN_DECODE_FRAMES` disagree with the
     declaration -- wired into BOTH graph builders, since either can resolve a
     length. Commit 8c5449db.
  2. `render_canvas = (832, 480)` declared, because the decode floor's own
     comment ties it to "this canvas". Without a declaration,
     `OTR_LTX_RENDER_CANVAS` could move the canvas at boot and invalidate the
     static contract with no code change; `declared_render_canvas` is applied
     LAST in `build_request_from_shot` precisely so a declaration wins.
- verified: plan-vs-engine agreement on the PRODUCTION call path (`join_mode_for`,
  not a forced mode) -- beats 17/89/168/169 take `single`, 170/250/338/442/530
  take `chain` with 2-4 segments, and every segment satisfies
  `_ltx_frame_length(render_frames) == render_frames`. Suite 8253 passed.
  **A live leg has NOT yet re-run -- the fix lands in code the running server
  loaded hours earlier, so it is proven in arithmetic only until the overnight
  driver restarts the server and re-runs it.**
- bible-worthy: yes. **A capability declaration is a promise the runtime must
  keep, and an OVERSTATED one is worse than none.** An understated contract
  merely wastes capability; an overstated one converts a plannable component
  into a GUARANTEED late failure, because the planner commits work against the
  declaration and only the render discovers the lie. Three channels can break the
  promise and all three need closing: the declaration itself, an environment
  override read at runtime, and a second dimension (here canvas) the bound
  silently depends on.
- verify condition (automatable, and implemented): feed each adapter's own
  declared minimum and maximum through its own length resolver and require them
  to come back unchanged -- `test_a_declared_MINIMUM_is_a_length_the_adapter_can_actually_render`.
  Currently covers `ltx_video` only, because each adapter resolves length
  privately; the general version needs the shared `resolve_render_frames`
  interface both panel lanes converged on.

## PBUG-20260802-02 -- the writer casts two characters and writes lines for one
- surfaced: the live 45-word campaign, 2026-08-02. TWO legs, two different
  symptoms, one underlying fault:
  * `wan_ti2v` (02:35, 2.7 min): `[scifi_fable2] pass 'script' failed after 4
    attempt(s): markup ladder exhausted`, with `UNKNOWN_SPEAKER` on every line
    of both characters AND `CAST_MEMBER_SILENT: Commander Vance` /
    `CAST_MEMBER_SILENT: Pilot Elara`.
  * `ltx_video` (02:47, 2.2 min): `OTR_CastLock: freeze cascade stamped
    freeze_verdict='needs_full_rerun'`, from
    `[LFC] read-only structural validation failed under content_owned_readonly:
    content_authorship: line proof coverage mismatch: missing=[]
    extra=['shot_001_b2', 'shot_001_b4', 'shot_002_b2', 'shot_002_b4',
    'shot_003_b2']`.
- root cause: the ledger shows those five "extra" rows are EXACTLY the second
  character's lines, and every one carries `len(text)==0`:
      shot_001_b1  len=111  speaker=c02
      shot_001_b2  len=0    speaker=c03   <- extra proof
      shot_001_b3  len=196  speaker=c02
      shot_001_b4  len=0    speaker=c03   <- extra proof
  The phase-2B skeleton allocates dialogue slots for BOTH cast members, the
  composition fills only c02, and c03's rows are left empty. `_voiced_rows`
  (`_otr_content_authorship.py:28`) excludes a row with empty text, so the
  authorship proofs -- minted while those rows still had text -- no longer
  match the live voiced set, and the read-only structural validation refuses.
  So the writer produces an effectively SINGLE-character play from a TWO-
  character cast, and two different downstream gates catch it in two different
  ways.
- **the two symptoms are the same fault, which is why this is one entry.** The
  `UNKNOWN_SPEAKER` half was a separate, real parser gap (the role parenthetical,
  fixed in afe53c7c); fixing it did not fix this, it merely let a script that
  previously died at the parser reach the freeze gate, where the silent second
  character is what fails. Fixing a blocker upstream does not fix the thing it
  was hiding.
- fix: **NOT FIXED.** Recorded at 03:00 with the operator asleep and the GPU
  mid-campaign. It is a story-QUALITY defect in the composition pass, not a
  renderability bug, and the right fix is upstream of everything touched
  tonight.
- verify idea (automatable): after composition and before the freeze gate,
  assert every cast member the skeleton allocated a slot for has at least one
  non-empty line -- and fail there, by name, rather than letting an empty row
  reach an authorship proof and surface as a coverage mismatch five stages
  later. The current failure names `shot_001_b2` when the real answer is
  "character c03 never got any dialogue".
- bible-worthy: probably -- **an artifact minted from state that a later stage
  can still invalidate is a proof of nothing.** The authorship receipt is built
  from rows that are voiced AT THAT MOMENT; nothing stops a later pass emptying
  one. The portable rule is to build such a proof at the same barrier that
  freezes the state it describes, or to re-derive it at the gate.

## PBUG-20260802-02 CORRECTION (same day, before any fix was written)
The entry above claims the two legs were "the same fault, which is why this is
one entry". **That is not established, and the difference changes the fix.**
Grounded from the ledgers and the server log:

* `wan_ti2v` ran the **`scifi_fable2`** lane. It failed with `UNKNOWN_SPEAKER`
  plus `CAST_MEMBER_SILENT` -- and that lane's own gate is what caught it
  (`_otr_scifi_fable2.py:2306`, "speaker set != cast rows", plus the parser
  defect). The gate WORKED. What failed upstream of it was the writer producing
  a play in which a cast member never speaks, and the repair ladder exhausting.
* `ltx_video` ran the **`scifi_news_pro`** lane, whose ledger meta says in as
  many words: `"pack for bank 'scifi_news_pro' declares NO line_composer_system
  seam -- the lane owns its own content loop"`. There is NO equivalent gate on
  that path, so the empty rows travelled all the way to the freeze gate and
  surfaced as a line-proof coverage mismatch naming `shot_001_b2`.

And the cast row that was silent is the tell: `c01=ANNOUNCER, c02=Elias,
c03=**The Relay**`. The lane cast a RELAY -- a machine, not a speaking part --
and then, reasonably, wrote it no dialogue. So the `scifi_news_pro` root cause
is most likely CASTING a non-speaking entity, not a composition pass dropping
lines it was asked to write.

What survives from the original entry: an artifact minted from state a later
stage can still invalidate is a proof of nothing, and the named-gate verify
condition is right and lane-agnostic. What does not survive: "one fault, two
doors", and the implication that fixing the fable2 parser gap had anything to do
with the `scifi_news_pro` failure. Two lanes, two mechanisms, one shared
symptom.

## PBUG-20260802-02 -- THIRD MANIFESTATION AND FIX (2026-08-24, shakespeare/MARIA)
- surfaced: `scripts/otr_writer_bank_gate.py --acts 1`, bank=shakespeare,
  profile=otr_w45_still_flat, live headless leg 2026-08-23 23:14-23:17.
  `[LFC:phase_10] 1 critical gap(s) -- FREEZE REJECTED. First: cast
  char_id='c03' (name='MARIA') has no non-skipped line.` -- the same universal
  backstop named in the original entry, firing for the first time on the
  shakespeare lane.
- **ROSTER CORRECTION.** `nodes/_otr_scifi_fable2.py` does not exist anywhere
  in the live tree (confirmed by direct check and a repo-wide grep; only
  stale pytest fixtures reference the name). `config/story_packs/banks.json`
  lists six live banks -- media_archive, original, scifi_news_pro,
  public_domain, shakespeare, custom_source_bank -- not seven. Whether
  scifi_fable2 was retired, renamed, or merged into scifi_news_pro is NOT
  re-established here; the original entry's `wan_ti2v` manifestation cannot be
  re-verified, regressed, or fixed by this change because there is nothing
  left to point it at. Flagged rather than silently implied as covered.
- root cause, precisely: `nodes/_otr_outline.py`'s `_phase_check` validates
  outline cast membership in only ONE direction (`invented = used_speakers -
  locked_cast_set`) -- it never checks the reverse. A locked cast member
  (shakespeare: drawn from the scene's curated `cast_hints`, e.g.
  `['Malvolio','Maria','Toby']` for `folger-twelfth-night:act2-scene5-
  malvolio-letter`) can legitimately receive ZERO beats under a tight budget
  (a 1-act/45-word run buys very few voiced beats for three competing
  characters). Maria is not a phantom: she has three real speeches in the
  source (`config/source_banks/shakespeare/sources/
  twelfth_night__act2_scene5.txt:29,295,299`) and exits early in the scene,
  which is exactly the shape a tight window can miss.
- fix: `nodes/_otr_cast_coverage_repair.py` (new), wired into
  `_otr_writer_tail.py._run_writer_tail` immediately after
  `_clean_window.reconcile()` -- the last point that touches canonical text
  before the freeze cascade node runs. Gated on
  `_otr_freeze_cascade.resolve_freeze_policy(meta).run_inline_safety_cleanup`
  (config-driven, never a hardcoded bank list), so content-owned lanes
  (today: scifi_news_pro, whose own earlier gate at `stamp_receipt`/
  `require_voice_coverage` already covers it before this tail is ever
  reached) are skipped -- a deliberately silent non-speaking entity (a
  Relay, per the CORRECTION entry above) must never be forced to speak.
  For each gap found by the new pure `_otr_ledger_freeze.cast_coverage_gaps`
  (extracted, byte-identical, from `_check_per_cast_invariants`): MODE 1
  retries an existing skip=True slot; MODE 2 mints one new ledger-only
  lines[]/beats[] row (never added to the pydantic `Outline.beats`) for a
  character Stage 2 never allocated at all. Exactly ONE `compose_line` call
  per gap, through the same seeded `creative_fn` every other line already
  uses -- no new sampling mechanism. On failure the row is left BYTE-
  IDENTICAL to today's refuse-and-halt; only the success case improves.
  Fidelity graft: when `meta.source_meta.cast_hints_presence`
  (`_otr_shakespeare_sources.cast_presence_from_text`, new) names a real
  attested speech for the gap character, it rides into
  `LineRequest.source_block` so the repair is grounded in Shakespeare's own
  words rather than free invention -- honoring "the author's own language is
  carried as written" instead of working around its absence.
- ledger fields touched, one owner each: `lines[].text/char_count/word_count`
  and `.skip/.tts_skip_reason/.compose_flags` -- `_otr_cast_coverage_repair`,
  via the existing `patch_line_text`/`patch_line_fields` owners, called one
  more time; Mode-2-only `lines[]`/`beats[]` new rows -- same module, reusing
  `production_ledger.init_lines_from_outline`'s exact row shape;
  `meta.source_meta.cast_hints_presence` -- `_otr_shakespeare_sources.
  source_meta_from_scene`, sole writer, shakespeare-only, additive;
  `meta.cast_coverage_repair` -- new telemetry receipt, sole writer;
  `cast[]`, `meta.cast_contract.cast_seed`/`num_characters_request` --
  UNTOUCHED (no cast row is ever added, removed, or reordered, so
  `OTR_CastLock`'s replay contract needs no new reasoning about desync).
- determinism: the trigger (`cast_coverage_gaps` non-empty) is a pure
  function of the already-seeded generation stream. The repair's
  `compose_line` call is one more call into the identical seeded slot every
  other line goes through. Mode 2's minted `beat_id` is the next integer in
  the existing `bNNN` scheme, scanned from settled state, never randomly
  chosen. A seed that previously refused now produces a real render instead
  of nothing (the fix); a seed run twice after the fix must still produce
  byte-identical output both times.
- verify: 18 unit tests in `tests/test_cast_coverage_repair.py`, including a
  negative control pinning the gate still fires, Mode 1, Mode 2 (with a
  beat_id-collision check), a composition-exhaustion test proving failure
  leaves the row byte-identical to today's refuse-and-halt, the fidelity
  graft against the REAL shipped source file (not a fixture), and a
  cast[]/cast_contract parity check. `cast_coverage_gaps`'s extraction from
  `_check_per_cast_invariants` verified behavior-preserving against the full
  freeze/gap-audit suite (301 passed, unchanged).
- **NOT re-reproduced on the exact original random scene draw.** The
  writer-bank-gate's `selection_mode: random` redraws a fresh scene each run;
  a same-night reproduction attempt drew a different scene and passed
  cleanly (no gap to repair). The fix is proven at the unit level against a
  ledger shaped identically to the original failure (same cast_hints, same
  char_id/name pairing, the real MARIA source text), plus a live full-suite
  run with no regression on the four other banks. The overnight writer-gate
  loop will exercise the shakespeare bank repeatedly and is expected to hit
  this scene/budget combination again naturally; when it does, the repair
  now runs live rather than refusing.
- bible-worthy: yes -- "a producer that locks a slot has no obligation to
  serve it" is the general defect class (Stage 2 checks invention, never
  starvation), and the fix shape (repair upstream of the gate, gate stays
  refuse-only, no post-ledger surgery) is portable to any future lane with
  the same allocate-then-compose split.
- confidence: HIGH on mechanism (verified against real call graph, real
  source text, real freeze-gate code) and on the fix's correctness at the
  unit level; MEDIUM on live-leg closure pending a natural or forced
  re-reproduction of the exact original scene.
- status: **FIXED, live-verification pending the overnight loop** (was: NOT
  FIXED, 2026-08-02).

## PBUG-20260805-01 -- every adaptation cast rolled its gender, so 44 published rows contradict the source
- surfaced: published episodes, measured 2026-08-05 across every adaptation
  ledger under `output/otr` (88 ledgers, 176 non-announcer rows). Visible in
  shipped episodes -- e.g. `signal_lost_malvolios_yellow_stockings_20260804_192850`.
- symptom: MALVOLIO and LEAR cast female; MIRANDA, CORDELIA and ROSALIND cast
  male. 44 of 176 rows (25%) carry a gender that contradicts the shipped
  provenance sidecar. Also confirmed on MARIA, ROMEO, JULIET, CELIA, MACBETH,
  BENEDICK, VIOLA, MARCELLUS, FERDINAND, TITANIA, BANQUO, HERO, PROSPERO.
- root cause: `precompute_ensemble_slots` assigned every open slot a gender from
  a 40/40/20 largest-remainder roll (`_plan_gender_distribution`,
  `nodes/_otr_casting.py`), including slots whose NAME had just been popped off
  the source's own cast list. The roster truth was already on disk -- 14
  provenance sidecars carry a `characters` list -- and `source_meta_from_scene`
  never loaded it, so nothing downstream could know MALVOLIO was a man.
  The row gender is not only a voice field: it feeds the description LLM
  (`_otr_casting.py:777`), the outline prompt (`OTR_LedgerScriptWriter.py:4144`),
  the dialogue cast block (`_otr_line_composer.py:446`) and the image prompt's
  gender anchor (`otr_meta_brief_image_prompt.py:78-90`), so the defect reached
  the script and the portrait as well as the voice.
- fix: new `nodes/_otr_roster_gender.py` joins each source-owned slot name to the
  sidecar roster through an abstaining tier ladder, backed by a committed
  10-entry curated supplement for the names no tier reaches; the resolved gender
  OVERRIDES the drawn value at pinned indices while
  `_plan_gender_distribution` is left completely untouched -- same count, same
  priors, same rng, same post-call stream. Source-owned slots are also exempted
  from the name-coherence rename. Stamped as `meta.cast_source_contract` with
  per-name evidence.
- verify idea: for any adaptation ledger, every non-announcer row whose name
  appears in the source's `characters` roster must carry the roster's gender.
  Machine-checkable against the shipped sidecars with no render.
- bible-worthy: yes -- "a generator that rolls a value the source already
  records" is a reusable defect class, and the fix shape (join, abstain
  honestly, override in place rather than re-allocating) is portable.
- confidence: HIGH -- measured across every published adaptation ledger.
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `nodes/_otr_roster_gender.py` pins the source's own genders. Live proof in a published ledger: `signal_lost_the_stewards_soliloquy_of_vanity_20260805_061415` carries `meta._adaptation_character_genders.MALVOLIO.gender == "male"` at tier `exact` -- the exact character the entry cites as shipping female.
- previous status: OPEN (fix landed 2026-08-05; live proof leg pending)
- status: **CLOSED 2026-08-18 -- FIXED AND LIVE-PROVEN**

## PBUG-20260805-02 -- LATENT: the bark voice replay rebuilds a different ensemble than the writer cast
- surfaced: NOT a production failure. Reproduced by probe against the shipped
  modules on 2026-08-05 at cast_seed 424242, and recorded on operator direction
  rather than promoted to a fix. **This entry does not meet the log's usual
  live-artifact admission bar and must not be fan-out-promoted to the Bug Bible
  on its own evidence.**
- symptom: at cast_seed 424242 with source names
  `['Antipholus','Dromio','Adriana']` the writer's ensemble is
  (ANTIPHOLUS male, DROMIO other, ADRIANA female) while `replay_voice_assignment`
  reconstructs (ERIN MARTIN female, FABER SATO other, KANE SIRIKIT male). The
  gender SEQUENCE diverges on 149 of 200 seeds (74%).
- root cause: an asymmetry in what the replay is told. `assemble_pre_locked_rows`
  accepts `source_character_names`, and on an adaptation lane it POPS those names
  off a queue for zero rng draws; `replay_voice_assignment(*, cast_seed,
  num_characters, lemmy_hit)` cannot accept them, so the replay takes the pool
  path and burns `pick_first_last` draws the writer never spent. Every later draw
  is then off by that much.
- why it is latent: `CastLock._assign_bark_voices` writes only
  `row["voice_preset"]` (`nodes/cast_lock.py:355`) and never a gender, and the
  shipped workflow runs indextts2 (node 80), which takes its audible reference
  from the ROW gender at `cast_lock.py:563`. Nothing a listener hears depends on
  the replay's reconstruction today. It becomes real the day the operator
  switches character voices back to bark.
- fix: none. The corrective step was CUT from the 2026-08-05 continuity build as
  not worth the surface it touches; forwarding `source_character_names` +
  `source_bank_id` into the replay restores the match exactly (probed), if it is
  ever wanted.
- verify idea: `lock_cast` then `_assign_bark_voices` at cast_seed 424242 with
  those three source names; the reconstructed per-slot gender must equal the row
  gender.
- bible-worthy: no -- latent, single-project, and no production artifact behind
  it. Recorded so the next reader does not re-derive it.
- confidence: HIGH on the mechanism, N/A on production impact (there is none today).
- status: OPEN (deliberately unfixed; reproducing seed 424242)

## PBUG-20260805-03 -- the announcer was planned but never scheduled, so scifi_news went 0-for-4
- surfaced: batch v2 headless run, 2026-08-05, 28 episodes over the canonical
  workflow. `scifi_news` failed 4 of 4 legs (008, 013, 015, 016) while every
  other lane was perfect: shakespeare 12/12, public_domain 9/9, original 2/2.
  Leg 013 burned **45.1 minutes** before dying; the others died at 4-5 minutes.
- symptom: `RESULT FAIL ... exception_message: "cast voice coverage failed for
  bank 'scifi_news': 1 of 4 cast member(s) have no SAYABLE line"`, raised from
  `_otr_cast_voice_coverage` at `stamp_receipt` -- AFTER the whole script had
  been written. The live server logs name the uncovered member as the ANNOUNCER.
- root cause: `compile_radio_score_draft` treated an uncovered cast member as
  ADVISORY -- it logged `cast_coverage advisory: N/M planned cast own a beat`
  and returned the score, on the reasoning that "an uncovered cast member simply
  carries no lines". The hole is therefore CREATED at P3 and only DISCOVERED
  after P5. It cannot be repaired downstream: P5 authors only `line_id` and
  `text` INTO a graph whose beat/shot/scene ownership is already compiled from
  the accepted score, so a member with no beat can never be given a line. A
  P5-level check could only burn the retry ladder and fail anyway -- which is
  what turned a 5-minute failure into a 45-minute one.
- fix: the advisory became a RECOVERABLE `RadioScoreDraftCompileError` with the
  new `cast_coverage` code, naming the missing ids. `validate_draft` converts it
  to a `PostValidationError`, which `_candidate_error_is_recoverable` already
  accepts, so the retry ladder and then the fresh-candidate loop redraft the
  score rather than failing the episode -- preserving the earlier ruling that
  cast_coverage must not become a fatal successor to the removed beat-count
  gate. Two prompt surfaces stated the OPPOSITE and were corrected: the pack's
  `codex_radio_score_system`, and `_RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION`,
  which is appended LAST and still said "an unused planned cast member is not a
  story failure" (kibitz r3, Codex).
- why the loop cannot spin: the invariant is provably satisfiable.
  `num_characters` is clamped to 1..6 and the announcer makes at most 7;
  `_RADIO_SCORE_MAX_BEATS` is 3 scenes x 4 beats = 12; `_codex_target_beat_count`
  is called with `len(p2.cast)` and returns `max(cast_count, 3, min(12, ...))`.
  7 <= 12, so every planned member can always own a beat.
- coverage: `tests/test_p3_cast_coverage_invariant.py` (6 tests, mutation-
  verified: reverting the raise fails exactly the two coverage assertions).
- generalizable rule for the Bible fan-out: a producer stage that PLANS a
  resource must not defer the check that the resource was SCHEDULED to a
  consumer stage that cannot create it. Where the plan and the schedule are
  written by different passes, the schedule-time check is the only one that can
  still be repaired.
- live receipt: **PAID 2026-08-05 16:24.** One `scifi_news` leg at the exact
  failing coordinates (180 words, 2 characters) through
  `workflows/otr_canonical.json`, on a server booted FRESH after the fix
  (the batch v2 server had booted at 08:30, before the 14:30 commit, and held
  the old module in memory -- which is why the batch could not have proven it).
  All three gates: `RESULT SUCCESS`, `obs_publish OK`, and the asset on disk --
  `output/otr/obs/signal_lost_echoes_of_bias_20260805_161414_silent_procgen_blended_captioned_with_credits_final.mp4`,
  16.5 MB. `Prompt executed in 00:22:31`.
- what the log proves, and it is the whole point: the defect FIRED and was
  RECOVERED. P3 attempt 1 failed `draft.cast_coverage` ("3/4 covered, missing:
  announcer"), attempt 2 (typed repair) failed identically, the retry ladder
  exhausted -- and instead of killing the episode it logged `P3 candidate cycle
  1 exhausted (PostValidationError); abandoning it and starting cycle 2`, whose
  first attempt passed. Two coverage failures, one cycle abandonment, one
  published episode. Pre-fix that same sequence was terminal, four times.
- the recovery is bounded in practice: cycle 2 succeeded on attempt 1, so the
  unbounded fresh-candidate loop did not need a cap on this leg. The open
  cycle-cap question (agy for, Codex against) stays open on the same reasoning
  -- the invariant is provably satisfiable at cast <= 7, beats <= 12 -- but
  there is now one live data point rather than none.

## PBUG-20260805-04 -- the announcer read the source URL and licence aloud, and the captions burned it into the video

- surfaced: the PUBLISHED corpus, not a review. A scan of all 1,587 ledgers under
  `output/otr/episodes` finds spoken lines carrying a URL, a bare domain, a
  licence identifier or our own prompt labels. By `speaker_role` and line
  position: **84 leaked lines, 100% announcer, 100% at the LAST announcer
  position (the coda row), 0 non-announcer.** 30 distinct episodes leak on or
  after 2026-08-04, the most recent at 2026-08-05 14:22. The reusable predicate
  (`scripts/audit_spoken_citations.py`, which also matches shortened licence
  forms the sidecar string cannot) reports **69 episodes** with findings.
- worst shipped example, `2026-08-05 08:42` -- the announcer reads our own
  interpreter scaffold on air:
  `From tonight's echoing "Nothing," let us turn our ears to the silent archives:`
  `Source: Folger Shakespeare. Date/Rights: c. 1606 | CC BY-NC 3.0. URL:`
  `https://www.folger.ed...` -- `Source:`, `Date/Rights:` and `URL:` are verbatim
  the field labels built at `_otr_shakespeare_sources.py:586-589`.
- second surface, and the reason this is not merely an audio defect:
  `_otr_captions.py:283-286` copies RAW `lines[].text` into the ASS cue
  ("RAW line text, deliberately") and CaptionBurn is enabled in
  `workflows/otr_canonical.json`, so the URL is **burned into the delivered
  video**. NOT the still prompt (announcer rows take `scene_beat` at
  `otr_meta_brief_image_prompt.py:1117`, whose target carries no line text) and
  NOT the i2v motion clause (default OFF, `_otr_motion_clause.py:13-14`) -- both
  were claimed as surfaces in the inherited spec and both were disproved.
- mechanism: the interpreter is handed the source URL
  (`_otr_public_domain_sources.py:635`, `_otr_shakespeare_sources.py:589`) and
  asked for an attribution note in the SAME payload (`:665`, `:624`). The writer
  hoists that reply (`OTR_LedgerScriptWriter.py:4895-4897`) and
  `compose_news_coda` appends it VERBATIM (`_otr_line_composer.py:1285`,
  contract at `:1255` "never score, shorten, or replace it"). The append is
  deliberate -- it exists so a weak model cannot blend the fact away -- so the
  one thing engineered to survive unedited is the one carrying the URL.
- root cause of the RECURRENCE, which is the important part: the deterministic
  replacement already existed. `meta["provenance_coda_line"]` is composed by
  `_otr_provenance.spoken_coda_line` and stamped at
  `OTR_LedgerScriptWriter.py:3595`, and `_otr_provenance.py:112-118` records that
  the licence was removed from the spoken line on 2026-08-04 for exactly this
  reason. **That fix was applied inside `spoken_coda_line()`, a function with
  ZERO readers** -- grep returns the write and one docstring. The live path was
  never touched, so 30 more episodes leaked after the fix "landed".
- fix: select the effective spoken fact at the writer call site, keyed on
  `"provenance" in meta` (stamped unconditionally at `:3592`; the coda key was
  NOT, so presence of the coda is an invalid ownership test). A provenance-owned
  lane always takes the deterministic append regardless of `_style_grammar_on`;
  owned-but-empty goes straight to `fallback_announcer_outro("")` with neither
  composer entered. `news_close_brief` keeps its value and its owner -- it is
  also the treatment "Sign-off" line (`video_engine.py:1866`). The URL is also
  removed from both interpreter prompts (it never grounded anything; grounding is
  the source text) with `PROMPT_VERSION` bumped to
  `public_domain_interpreter_v3` / `shakespeare_interpreter_v2`.
- found in passing, same call, fixed with it: `compose_news_coda` was never
  passed `source_bank_id` (`OTR_LedgerScriptWriter.py:5491-5497`), so EVERY lane
  resolved media_archive's `coda_system` prompt -- while the sibling
  `compose_announcer_outro` call has passed it since Stage 4, and
  `tests/test_closing_seams_bank_routing.py:123-137` already proved the composer
  routes correctly when given it.
- receipt: `meta["spoken_coda_source"]`, closed vocabulary
  (`provenance` | `news_close_brief` | `none`) validated at write time, so a
  corpus audit can JOIN on what was spoken instead of inferring it from prose.
  Inferring it from prose is how this survived.
- coverage: `tests/test_spoken_citation_audit.py` (22 tests) pins the predicate
  itself, including that the deterministic coda PASSES its own audit and that the
  empty `license_label` on the public-domain sidecar is dropped as a needle -- an
  empty needle is a substring of every string and would report the whole corpus.
- generalizable rule for the Bible fan-out: **a fix applied to a function with no
  callers is not a fix.** When correcting a defect on a live surface, prove the
  edited symbol is REACHED from that surface before claiming the defect closed --
  grep for callers, not just for the symbol. This is the fourth armed-consumer-
  without-producer defect found on 2026-08-05.
- live receipt: **PAID 2026-08-05 evening.** Seven canonical legs on a server
  booted after `3943dd38`, across every lane the fix touches:

  | leg | bank | `spoken_coda_source` | the announcer's closing line |
  |---|---|---|---|
  | 01 | public_domain 320w | `provenance` | "Tonight's tale was adapted from a work in the public domain." |
  | 02 | shakespeare 320w | `provenance` | "Tonight's tale was adapted from Folger Shakespeare." |
  | 03 | media_archive 320w | **`news_close_brief`** | its own factual note, verbatim -- the CONTROL held |
  | 05 | public_domain 520w x3 | `provenance` | deterministic coda |
  | 06 | shakespeare 520w x3 | `provenance` | deterministic coda |
  | 07 | original 320w | `none` | fictional close, no attribution -- correct for an unowned lane |

  **Zero leaked lines across all of them.** Leg 02 is the one that matters most:
  that lane used to read "CC BY-NC 3.0" aloud on essentially every episode, and
  now says only the edition name -- exactly what `_otr_provenance.py:25-27`
  specifies ("names the SOURCE, never the licence identifier").

  The control is the other half of the proof: `media_archive` still speaks its
  news note verbatim, so the fix did not silence the lanes that are supposed to
  carry one.

- corpus verdict: `scripts/audit_spoken_citations.py --root <output>/otr/episodes`
  scanned **1,595** ledgers (8 more than the pre-fix baseline) and reports
  **69 findings -- unchanged**. Every new episode is clean; the number did not
  move because nothing new leaked. Pre-fix, a shakespeare leg leaked essentially
  every time.
- not covered by this receipt, and correctly so: `scifi_news` never traverses this
  code. It dispatches to `scifi_news_circuit` and returns before the coda block
  (`OTR_LedgerScriptWriter.py:3663-3717`), so its ledger carries no
  `spoken_coda_source` key at all -- confirmed live on
  `shadows_of_phobos_20260805_193430`. That is why the acceptance control is
  `media_archive` and not `scifi_news`.

## PBUG-20260807-01 -- the announcer asked the operator to write the opening, and 23 episodes shipped with it as their first line

- status: **FIXED AND LIVE-PROVEN 2026-08-07** (5/5 qualification legs; receipts below)
- promotion: BUG-12.86 (survival-guide `7a5fb88`, entry count 261 -> 262, `otr_coverage_index.yaml` row appended in the same commit). Promoted by the window under the 2026-08-07 amendment above, after checking the class against the index and the 261-entry Bible and finding it uncovered.
- found: corpus scan of shipped ledgers under `output/otr/episodes`, 2026-08-07,
  while investigating a DIFFERENT reported defect (`--premise` allegedly not
  reaching the writer). The premise wiring turned out to be sound; this was next
  door and worse.
- symptom, verbatim from `lines[].text` on shipped episodes:

  > "Please provide the SETTING, TIME, HOOK, and the cast list so that I may
  > write the opening for you."
  > "Please provide the cast list and setting details so I may begin the
  > broadcast."

- blast radius: **23 ledgers**, all `line_id b001`, `speaker_role announcer`,
  compose_flags `['announcer_intro', 'announcer_intro_rewritten']`. Range
  2026-07-22 .. 2026-08-07 across `original` (6), `shakespeare` (9),
  `public_domain` (6), `media_archive` (2). It is the FIRST line the listener
  hears, it is spoken by TTS, `_otr_captions.py` burns raw `lines[].text` into
  the ASS cue, and because the rewrite runs BEFORE the outro pass the poisoned
  text was also fed forward as `intro_text` / `OPENING TONE` into the close.
  A 24th corpus hit (`the_caretakers_clause`, `shot_001_b2`, scifi lane) is
  in-story machine dialogue and is NOT this defect.
- **four independent faults, none of which failed loudly:**
  1. `_otr_line_composer.compose_announcer_intro` read
     `getattr(safe_open_brief, 'hook', '')`. `SafeOpenBrief` has never defined
     `hook` -- its fields are `setting`, `time_of_day`, `opening_status_quo`,
     `cast`, `era`. The getattr default made it silent, and
     `opening_status_quo`, `cast` and `era` were constructed at two call sites
     and read by no prompt builder.
  2. its `"\n".join(filter(None, (...)))` could never drop anything: every
     element was an f-string with a literal label prefix, so always truthy. A
     starved brief therefore shipped as bare labels -- `"SETTING: \nTIME: \n
     HOOK: \nWrite the opening now."` -- which reads to a model as a form.
  3. `_otr_story_brief._validate_produced_open` accepted a brief with an EMPTY
     CAST (it iterated `model.cast` only to reject off-roster names), while all
     four banks' `announcer_intro_safe_system` seams end "Use ONLY the proper
     names in the cast list below; invent none". The prompt promised a roster it
     never sent.
  4. the rewrite could not have recovered: a failed compose does NOT raise --
     `_announcer_generate` converts the exception to `None` and the composer
     returns `fallback_safe_open()`, non-empty canned text -- so the writer
     stamped `announcer_intro_rewritten` and overwrote a real composed opening.
     The documented keep-the-in-loop-intro posture only ever fired on a raise.
- **the origin is NOT the obvious commit.** `314dd481` (2026-07-24) rewrote the
  safe-open path and severed faults 1 and 2, but **10 of the 23 legs predate
  it** -- proven from the git HEAD each ledger stamps at render time
  (`341545ec` x6, `f150213f`, `2129ce84`), not from dates. At `341545ec` the
  composer already read all five fields, already emitted each only when
  non-empty, and already sent a cast line -- and `_validate_produced_open` is
  BYTE-IDENTICAL there to HEAD. So fault 3 is the older cause and the one that
  explains the pre-314 legs, whose replies lead with the cast list.
- fix: one shared viability predicate --
  `(setting OR opening_status_quo) AND at least one CLEANED cast name` --
  defined once in `_otr_line_composer` and imported by `_otr_story_brief`, so
  the validator and the composer cannot disagree about what a usable brief is.
  Direct attribute access replaces every `getattr` default. Labels emit only
  with a value behind them. A starved brief raises a typed
  `AnnouncerBriefStarvedError` BEFORE the model call; the rewrite caller
  declines and keeps the existing line, the in-loop caller ships the
  deterministic open and records the fallback. A returned structural fallback is
  no longer stamped as a rewrite. Shipped `a200b6f1` + `615de993`.
- **two dead receipts found in passing, same defect class, fixed with it:**
  `meta["open_safe_fallback"]` and `meta["news_coda_fallback"]` each tested for
  a flag string no producer has ever emitted, so both read False on every
  episode including the ones that fell back. These are STATIC findings -- no
  live artifact demonstrates their impact -- and are recorded here only because
  they rode this fix, NOT as production incidents in their own right.
- **the class, which is the reusable part:** a receipt or prompt-context field
  keyed on a producer string or attribute the producer never emits, hidden by
  `getattr(x, "name", default)` or an `in flags` test that silently reads False.
  Four instances now: `hook`, `open_safe_fallback`, `news_coda_fallback`, and
  BUG-LOCAL-255's `_speaker_role`. It fails in the SAFE direction, so nothing
  ever complains.
- why no test caught it: `tests/test_closing_seams_bank_routing.py` asserted the
  SYSTEM message only, `tests/test_announcer_intro_rewrite.py` stubbed the
  compose entirely, and `test_intro_requires_nonempty_structural_context` had a
  parametrize list with exactly ONE case -- the sibling `script_brief` path --
  so the safe-open branch carried the same invariant and none of its cases.
  Nothing had ever asserted the brief's content REACHES the prompt.
- receipts: suite 9177 passed / 111 skipped / 1 xfailed; Bug Bible 17 at
  survival-guide `3759ae5`; `workflows/otr_canonical.json` byte-identical.
  Ten mutations of the shipped code each confirmed to turn the new tests red.
- **LIVE PROOF OWED, and one trap to avoid when running it:**
  `workflows/otr_canonical.json` node 1 has `widgets_values[23] == 'scifi_news'`,
  a lane that dispatches to `scifi_news_circuit` and RETURNS BEFORE this code.
  A leg from the unchanged canonical JSON proves nothing here. Every leg must
  load that exact file with a per-leg RUNTIME bank override and assert the
  resolved bank is one of `original`/`shakespeare`/`public_domain`/
  `media_archive`. Per `PRODUCTION_SPRINT_LESSONS.md:106-113` this is
  model-sensitive work: 30-word smokes on two local model families plus one
  cloud/frontier lane, the same at 120, only then 720.
- OPERATOR DECISION OWED: the 23 shipped episodes are in canonical ledgers and
  delivered audio/captions. Rerender/republish, or tombstone as known-bad and
  exclude from publication. Not a build gate; recorded here so it is not lost.

### PBUG-20260807-01 -- LIVE QUALIFICATION, 5/5 PASS (2026-08-07)

Ladder per `docs/PRODUCTION_SPRINT_LESSONS.md:106-113`. Every leg loaded the
UNCHANGED `workflows/otr_canonical.json` with a per-leg RUNTIME bank override,
and the resolved bank was asserted from `meta.source_bank` before the leg
counted -- the canonical graph is pinned to `scifi_news`, which returns before
this code, so an un-overridden leg would have been green and meaningless.

| Leg | Bank | Words | Writer | b001 (opening line, verbatim) |
|---:|---|---:|---|---|
| 1 | shakespeare | 30 | `mistralai/Mistral-Nemo-Instruct-2407` | "In the royal court of Britain, King Lear demands an accounting from his daughter, Cordelia." |
| 2 | public_domain | 30 | `google/gemma-4-12b-it` | "The sun hangs heavy over the garden as Rikki-tikki-tavi keeps a watchful eye on the grass..." |
| 3 | original | 120 | `mistralai/Mistral-Nemo-Instruct-2407` | "In the hushed confines of Spender Manor, as the grandfather clock strikes midnight, Malcolm Sirikit and Clarisse Spender..." |
| 4 | media_archive | 120 | `google/gemma-4-12b-it` | "From the dust of a forgotten archive, we find Sailor Burns and Rod Howard standing in a silent hallway..." |
| 5 | shakespeare | 30 | CLOUD `~anthropic/claude-haiku-latest` | "Good evening, friends, and welcome back to Signal Lost -- tonight we find ourselves on the battlements of Elsinore Castle..." |

**Every leg:** `meta.announcer_intro_rewrite == {"status":
"announcer_intro_rewritten", "reason": null}`; schema `l4-2026-08-07`;
`obs_publish OK` with the asset on disk; and **no leg asked the operator for
input**. Four affected banks, three model families (Mistral / Gemma /
Anthropic-remote), both 30 and 120 words.

Leg 5's cloud arm is proven from the server log, not inferred:
`[OpenRouter] load slot=A handle=openrouter:slot-a
slug=~anthropic/claude-haiku-latest route=default ctx=200000 (remote, 0 VRAM)`
followed by `[OpenRouter] call accounted ~1239 tokens`.

**Two things this qualification did NOT establish, stated so nobody reads more
into it than it earned:**
1. **No exhaustion rate.** Five legs is not a rate. No leg hit
   `reason: derive_failed`, so the starvation path itself was never exercised
   live -- the guard is proven present and non-interfering, not proven to fire
   correctly in production. Its unit coverage is the evidence for that.
2. **A meta-recording gap found in passing:** `meta.openrouter_slot_a_model` is
   `null` on leg 5 even though the slot demonstrably resolved and served the
   run. Routing worked; the RECEIPT is incomplete. Not this defect, not fixed
   here, and static -- so it does not get its own PBUG.

---

## PBUG-20260811-01 -- forcing the LEMMY cameo kills the scifi_news_pro writer

- surfaced: two live canonical headless legs, 2026-08-11 (`PROBE B_90w_forced`,
  and the `BANKSWEEP scifi_news_pro` leg of the six-bank sweep the night before)
- symptom: node 1 `OTR_LedgerScriptWriter` raises
  `[scifi_fable2] pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; last defects: - BAD_LINE`. The run dies before any casting; no
  episode, no assets.
- root cause: NOT ESTABLISHED. What IS established is the trigger and that one
  plausible explanation is ruled out. Reproduced at BOTH 30 and 90 target words
  with `lemmy_cameo="always include"`, so it is not a word-budget squeeze; the
  same lane at 30 words with the cameo on its natural roll gets the writer
  through cleanly (it then fails elsewhere -- see PBUG-20260811-02). The
  pre-locked LEMMY row is what the `scifi_fable2` script pass cannot satisfy.
- fix: NONE YET. Recorded, not repaired.
- verify idea: run the `scifi_news_pro` lane with `force_lemmy=True` through the
  writer's script pass and assert it does not exhaust the markup ladder. A
  cheaper unit-level version: assert the lane's script prompt/validator can
  accept a pre-locked cameo row at all.
- bible-worthy: probably not on its own -- it reads as one lane's prompt/validator
  not tolerating a pre-locked row, rather than a portable contract. If a SECOND
  lane shows the same shape, the class ("a pre-locked cast row the writer pass
  cannot honour fails the whole render") would be.
- status: **CLOSED 2026-08-16 -- MIS-ATTRIBUTED.** See the correction below.

**Reachability note, stated because it is my own change.** `lemmy_cameo` was
whitelisted for headless drivers in `baf338ee` (Chunk D) so a qualification run
could force the cameo deterministically. That commit did not CREATE this defect
-- the widget has always existed and the GUI could always set it -- but it made
the failure reachable from the sanctioned headless runner, which is how it was
found. Four other banks force the cameo fine, so the whitelist is not the thing
to revert.

**CORRECTION AND CLOSURE, 2026-08-16 -- the attribution was wrong, proven two
ways** (three-agent adversarial verification, every claim re-grounded by the
driver against the artifacts before acceptance):

1. **The widget was INERT on this lane at the repro commit.** At `baf338ee` the
   runner dispatch returns at `OTR_LedgerScriptWriter.py:4032` and `lemmy_force`
   is first computed at `:4415` -- AFTER the return, feeding the `lock_cast()`
   call dispatched lanes never reach. `run_scifi_fable2_episode` took no lemmy
   input and the module contained zero lemmy references at that commit. There
   was never a pre-locked LEMMY row at ANY altitude on these legs; the
   forced-fails / natural-passes matrix was three independent stochastic draws
   of a widget that did nothing.
2. **The surviving leg logs refute the attribution.** `comfyui_58123.log`
   (BANKSWEEP scifi_news_pro, 30w) and `comfyui_59189.log` (PROBE B_90w_forced,
   90w): ZERO occurrences of "lemmy" in either log. The actual last defects are
   prose stage directions rejected as BAD_LINE_SHAPE ("Mike Brennan, pacing,
   phone in hand"), invented speakers (UNKNOWN_SPEAKER: LUCY, REPORTER) and
   SKELETON_BREAK -- ordinary stochastic markup non-compliance from
   Mistral-Nemo at temp 0.85. No END-shaped defect either, so it is not the D3
   grammar bug fixed 2026-08-15.

The sentence "The pre-locked LEMMY row is what the scifi_fable2 script pass
cannot satisfy" is WITHDRAWN, and the symptom's quoted `- BAD_LINE` was a
truncation of BAD_LINE_SHAPE (no defect named BAD_LINE existed at `baf338ee`).
The ladder-exhaustion class continues on its own record -- three live
exhaustions (08-10 sweep, 08-11 probe B, 08-15 gate), the last of them the END
grammar closed by D3 / Bible `12.105`. What survives for the Lemmy sprint is
architectural, not this bug: a cameo must be OFFERED to the lane's own
casting/script passes BEFORE the script is written, because fable2's gate (b)
(speaker set == cast rows) and codex's cast_coverage gate reject post-script
injection by construction.

---

## PBUG-20260811-02 -- scifi_news_pro dies at video render with no still for the closing-music beat

- surfaced: live canonical headless leg, 2026-08-11 (`PROBE A_30w_noforce`,
  profile `otr_w45_still_flat`, 30 words)
- symptom: node 92 `OTR_VideoRenderBatch` raises `still-spine handoff missing
  materialized scene still for shot shot_music_closing_001 beat
  music_closing_001 engine still_flat`. The writer, casting and the whole audio
  chain succeeded first (executed list includes nodes 1, 62, 63, 80-83).
- **REPRODUCED 2026-08-12** on a NORMAL render, which is what the entry below
  asked for: the `fastwan_8gb` leg of the 45-word every-visual-path sweep died
  identically -- `still-spine handoff missing materialized scene still for shot
  shot_music_closing_001 beat music_closing_001 engine fastwan_8gb`. Same beat,
  same shot id, different engine and different bank. Not lane-specific.
- **ROOT CAUSE, now established, and it is TWO layers.**

  **The general fault: the image producer was planning stills from the
  PRE-AUDIO ledger.** `OTR_MetaBriefImagePromptGen` took its `script_json` from
  `OTR_LedgerFreezeCascade`, which carries the closing cue under its AUTHORED id
  (e.g. `shot_006_music`). `EpisodeAssembler` mints the real mirrors afterwards
  as `music_{cue}_{NNN}` -- one row PER CHUNK -- and `OTR_ShotLock`'s overlay is
  what forwards every validated mirror. So the producer was planning against
  ids the finished episode does not use, and no scan of the pre-audio ledger can
  recover ids that do not exist yet.

  **The local fault: a reservation that suppressed itself -- IN BOTH BRANCHES.**
  A synthetic `music_closing_001` backstop existed for exactly this case, but
  its guard read `not any(speaker_role == "music_close")` -- so the PRE-AUDIO
  sentinel, under its authored id, suppressed the reservation that was supposed
  to cover it.

  **CORRECTION, 2026-08-12, found by an independent cross-check after the first
  fix shipped (`3446af3f`).** That commit, this entry and the code comment all
  asserted the OPENING reservation was already unconditional and that only the
  closing branch had drifted. **That was false.** The opening carried the
  identical role guard -- `not any(speaker_role == "music_open")` -- so a
  pre-audio opening sentinel under an authored id suppressed
  `music_opening_001` exactly as `shot_006_music` suppressed the closing one.
  The symmetry claimed did not exist; the same defect sat one branch away,
  unfixed, while being cited as the proof that the fix was right. Both branches
  are unconditional now.

- **CANDIDATE REPAIR, UNDER QA -- not proven, and deliberately not
  worded as shipped.** Two layers. The wiring and the closing
  backstop are committed (`3446af3f`); the opening backstop is in the
  working tree at the time of writing. Static QA only: a green suite
  and a regenerated parity fixture prove STATIC behaviour, not that
  the route publishes.
  1. **The general one.** Canonical link 255 retargeted so the image producer
     reads `OTR_ShotLock`'s POST-AUDIO `patched_ledger_json`
     (`[255,62,1,89,0] -> [255,90,0,89,0]`). Every minted mirror -- `_001`,
     `_002`, `_003` -- is then visible to the ordinary per-beat loop. **This is
     the fix that scales**; a reservation can only ever name one id.
  2. **The backstop, BOTH ENDS.** The opening and closing reservations are now
     unconditional twins. No guard is needed: `_add` already deduplicates by
     exact beat id, so when the real line is present the ordinary row has
     claimed the id and the call is a no-op. Each explicit guard was a second
     copy of that policy, which is how they drifted.
     **This is symmetry and hardening -- it is NOT the multi-chunk solution.**
     A reservation can only ever name `_001`; `_002`/`_003` are covered solely
     by reading the post-audio ledger.
- verify: `tests/test_canonical_still_spine_wiring.py` (6) pins the wiring by
  NODE TYPE, because every one of the 95 still-spine helper tests and all 538
  workflow tests passed both before and after the retarget -- the defect lived
  where nothing was watching. `tests/test_still_spine_helpers.py` adds: a close
  cue under ANOTHER id still earns the reservation (the live bug, inverted); the
  ordinary row WINS when the real id is present, asserted on `source` rather
  than a count, since a count cannot tell a reserved row from a deduplicated
  one; and a MULTI-CHUNK closing is covered only via the post-audio ledger,
  which pins the backstop's scope so nobody mistakes it for the general fix.
  The still-plan parity fixture moved by exactly one thing across all 29
  engines: `still_music_closing_001` now sits beside `still_music_opening_001`.
- **NOT YET PROVEN ON A LIVE LEG.** A green suite and a regenerated parity
  fixture prove static behaviour, not that the route publishes. The exact
  canonical `fastwan_8gb` leg must be re-run and its final ledger's closing ids,
  materialized still paths and published asset recorded before this is closed.
- **OPEN, ADJACENT, FOUND BY THE SAME REVIEW** -- recorded here so they are not
  lost, and NOT fixed in this change:
  - `OTR_ShotLock._same_frozen_episode()` does NOT fail closed. An identity
    mismatch logs a warning and returns the PRE-AUDIO ledger, and every overlay
    exception is swallowed to the same fallback -- which silently restores the
    exact input shape that caused this bug. The sprint lessons require a
    post-audio identity mismatch to stop the join loudly.
  - `OTR_ShotLock.IS_CHANGED` covers routing-environment state but NOT the
    ledger it reads from disk, and `audio_done` carries duration metadata with
    no content identity. So image planning now depends on a hidden disk read
    that is absent from the dependency signature (Bug Bible 06.01).
- bible-worthy: yes, once the live re-run lands -- but the portable shape is
  sharper than first recorded. Not "a handoff consumed an artifact that was
  never produced" but **"a producer planned against ids that a later stage
  renames, and its own backstop keyed off the wrong field."**
- **status: ROOT CAUSE ESTABLISHED; CANDIDATE WIRING/BACKSTOP REPAIR UNDER
  QA. NOT FIXED, NOT SHIPPED.** Do not mark either until a canonical live
  re-run proves every required music mirror has a materialized still and
  the episode publishes. Operator directive 2026-08-12, after the first
  version of this entry claimed both prematurely.
  - The operator's note also recorded that the OPENING reservation remained
    role-guarded and owed follow-up hardening. That guard has since been
    removed in the same candidate, so both reservations are now
    unconditional -- but that is HARDENING, it is still unproven live, and
    it changes nothing about the status above.
- **ACCEPTANCE TEST, agreed by both cross-checks:** a canonical `fastwan_8gb`
  leg with 60-SECOND opening AND closing cues -- long enough to chunk
  (`_MUSIC_MAX_CHUNK_DUR_S = 22.0`, so a 60 s cue becomes THREE 20 s
  chunks) -- proving every emitted `music_opening_00N` and
  `music_closing_00N` has a required target, a materialized still, and a
  published asset in `otr/obs/`. The original short cue does not exercise
  the chunked path at all.
- **The live FastWan reproduction and the root-cause evidence above STAND.**
  Only the repair/status wording is downgraded.
- **OPEN, ADJACENT, FOUND BY THE SAME REVIEW** -- recorded here so they are not
  lost, and NOT fixed in this change:
  - `OTR_ShotLock._same_frozen_episode()` does NOT fail closed. An identity
    mismatch logs a warning and returns the PRE-AUDIO ledger, and every overlay
    exception is swallowed to the same fallback -- which silently restores the
    exact input shape that caused this bug. The sprint lessons require a
    post-audio identity mismatch to stop the join loudly.
  - `OTR_ShotLock.IS_CHANGED` covers routing-environment state but NOT the
    ledger it reads from disk, and `audio_done` carries duration metadata with
    no content identity. So image planning now depends on a hidden disk read
    that is absent from the dependency signature (Bug Bible 06.01).
- bible-worthy: yes, once the live re-run lands -- but the portable shape is
  sharper than first recorded. Not "a handoff consumed an artifact that was
  never produced" but **"a producer planned against ids that a later stage
  renames, and its own backstop keyed off the wrong field."**
- status: **OPEN.** Root cause established and the repair is written, but
  this stays OPEN until a canonical live leg proves it. Two cross-checks
  independently warned against calling it shipped on a green suite alone,
  and the first version of this entry did exactly that while also
  mis-stating the opening branch.
- **ACCEPTANCE TEST, agreed by both cross-checks:** a canonical `fastwan_8gb`
  leg with 60-SECOND opening AND closing cues -- long enough to chunk
  (`_MUSIC_MAX_CHUNK_DUR_S = 22.0`, so a 60 s cue becomes THREE 20 s
  chunks) -- proving every emitted `music_opening_00N` and
  `music_closing_00N` has a required target, a materialized still, and a
  published asset in `otr/obs/`. The original short cue does not exercise
  the chunked path at all.

---

## PBUG-20260811-03 -- scifi_news LOST the Lemmy cameo it was built for

- surfaced: live canonical headless leg, 2026-08-11 (`BANKSWEEP scifi_news`,
  profile `otr_w45_still_flat`, 30 words, `lemmy_cameo="always include"`)
- symptom: the forced cameo produced NO Lemmy row and recorded NO reason. The
  episode's `cast_contract` is **empty** -- no `cast_seed`, no `cast_seed_source`,
  no `casting_attempts`, no `lemmy_hit`, no `lemmy_policy`,
  no `num_characters_locked`. `num_characters` was also ignored (asked 2, got 3:
  Ada, Kai, Dr. Elara). Compare `original` on the same sweep, which stamped all
  seven keys.
- root cause: **ESTABLISHED 2026-08-11.** `scifi_news` is a CONTENT-OWNED lane
  (`delivery_mode_for_meta(meta) == CONTENT_OWNED`, verified against the sweep's
  own ledger; `original` on the same sweep is `legacy`). Content-owned lane
  runners build their own cast rows and stamp their own voice presets, so they
  never run the writer's seeded cast picker -- `OTR_LedgerScriptWriter` says so
  in as many words at the content-owned tail. `lock_cast()` is what applies the
  cameo, so the cameo cannot happen there. The empty `cast_contract` is the same
  decision: that block deliberately stamps `meta.episode_seed` and NOT
  `cast_contract.cast_seed`, because cast_seed is a claim the writer's picker
  produced this cast and can replay it -- and a lane-owned cast has no
  `num_characters_request` to replay with. Claiming it detonated CastLock's
  replay in a prior bug (`num_characters must be 1-6, got 0`).
- **so this is a capability lost to an ARCHITECTURAL change, not a careless
  break.** scifi_news predates the content-owned redesign, worked under the
  legacy picker (which is why the operator remembers it working), and lost the
  cameo when it became content-owned. Nobody removed Lemmy from it.
- **THE OBVIOUS FIX IS THE WRONG ONE.** Routing content-owned lanes back through
  `lock_cast()` is exactly what the writer's comment warns detonates the replay.
  The repair belongs in the LANE RUNNER: either it offers the cameo itself when
  building its cast, or it stamps an explicit declined-policy so the ledger
  records a decision instead of a silence. Which of those is an operator call --
  it decides whether Lemmy can appear in scifi_news again at all.
- **why this is a REGRESSION and not a design choice:** the operator confirmed
  2026-08-11 that scifi_news "was built with Lemmy in mind and always used to
  work -- it was the first Lemmy plan". This lane is the cameo's ORIGINAL home.
  An earlier draft of the finding doc recorded it as a possible
  "lane owns its cast" design decision; that reading is WITHDRAWN.
- fix: NONE YET.
- verify idea: assert every runnable bank's episode records a cameo DECISION --
  `lemmy_policy` present with some value -- even on content-owned lanes. Absence
  of the key, not absence of Lemmy, is the detectable defect. Do NOT assert a
  non-empty `cast_contract.cast_seed` on content-owned lanes; that is the field
  whose false claim caused the earlier replay detonation.
- bible-worthy: likely yes as a class -- "a pipeline silently bypassed the one
  function that records a decision, so the ledger cannot distinguish 'declined'
  from 'never asked'". That shape is portable well beyond this cameo.
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `da44f642` + `7faf3bf7` stamp `cast_contract` on both content-owned runners. Live proof: `signal_lost_chunkb_accept_forced_lemmy_scifi_news_pr_20260816_185234` shows `cast_contract.lemmy_hit=True`, and every `scifi_news_pro` ledger through 2026-08-18 carries a populated contract.
- previous status: OPEN
- status: **CLOSED 2026-08-18 -- FIXED AND LIVE-PROVEN on a forced-cameo leg**

**Ranking note.** Of the three defects the sweep found this is the one that has
been shipping longest and most invisibly: nothing failed, nothing logged, and
every scifi_news episode since the regression simply has no cast contract. It was
only visible because the cameo was FORCED and then did not appear.

## PBUG-20260812-01 -- a shared module reached its sibling by an absolute `nodes.` import, so the Sage probe could not read Sage on ANY server

- surfaced: LIVE headless leg, 2026-08-12 (`_otr_single_engine_smoke.py --engine
  minimax_h3_video --frames 129` against the sage-free `h3` boot on :8000, lane
  19's solo smoke). The render refused before any weight loaded with:
  `BootContractError: the running server does not satisfy boot contract 'h3' ...
  needs SageAttention ABSENT, but the probe could not determine it
  (ModuleNotFoundError) -- an unverifiable Sage state is not a pass on a lane
  Sage silently corrupts`. The server WAS sage-free; nothing about Sage was
  wrong.
- symptom: a lane that requires a Sage-constrained boot contract is
  UNRENDERABLE on every server, and the refusal names Sage rather than the
  import that actually failed.
- root cause: `nodes/_otr_shared/boot_contracts.py:running_server_boot_state`
  reached its sibling package with
  `from nodes._otr_video_engines.motion_common import sageattention_patched`.
  **`nodes` resolves against `sys.path`.** Under pytest the repo root is on the
  path, so `nodes` IS the OTR package and the probe worked in every test. Inside
  a running ComfyUI server `nodes` is ComfyUI's OWN top-level node-registry
  module and OTR lives under `custom_nodes/ComfyUI-OldTimeRadio`, so the import
  raised `ModuleNotFoundError`, was caught, and left `sage_attention = None`.
  `check_running_server` correctly treats UNKNOWN as not-satisfied, so the
  contract could never be met.
- **why it shipped dormant:** the probe's error is only CONSULTED when a
  contract constrains Sage, and `h3` (2026-08-12) is the first one that does.
  `default` and `humo_diet` both say "don't care" about Sage, so the broken
  import sat behind them harmlessly since the S8 boot-contract mechanism landed.
- **TWO MORE INSTANCES of the same class, swept in the same commit** -- neither
  independently live-verified, both the identical import defect:
  `_otr_shared/content_oracle.py:family_for_engine` failed SOFTLY into a bare
  `except: pass` and answered from the `_FAMILY_FALLBACK` table on every call,
  so the live registry was "the source of truth when present" only OFF the
  runtime. That table stops at 2026-07-05, so `ltx_8gb`, `fastwan_8gb`,
  `still_word`, every cloud lane and `minimax_h3_video` resolved to family `""`
  in production -- which is not in `MOTION_FAMILIES` -- making
  `motion_required_for_engine` answer False and those lanes silently
  MOTION-EXEMPT. `_otr_shared/slot_matrix.py:eligible_engines_for_role` raised
  outright.
- fix: all three are relative imports (`from .._otr_video_engines import ...`),
  which resolve through the package's own `__name__` and are correct under both
  names. Committed `be4aadff` with lane 19.
- verify: `tests/test_minimax_h3_video.py::
  test_no_shared_module_reaches_a_sibling_by_an_absolute_nodes_import`
  AST-walks both shared packages and fails on any `nodes.`-prefixed
  `Import`/`ImportFrom`. **AST, not a text grep** -- the first draft grepped the
  source and failed on the comment that explains the fix, which necessarily
  quotes the broken line. Plus `::test_the_family_oracle_answers_from_the_
  REGISTRY_not_the_stale_table` for the soft-failure half, which asserts the
  CONSEQUENCE (every registered engine's family matches the registry) rather
  than the import.
- bible-worthy: **yes, and strongly** -- "a module works in the test environment
  and raises in production because an absolute import resolves against a
  `sys.path` that differs between them" is portable to every plugin/extension
  architecture where the host owns a top-level module name. The three-way split
  in how it failed (caught-and-unknown, swallowed-into-stale-fallback, raised)
  is the instructive part: only one of the three was visible at all.
- status: FIXED, live-proved (the same smoke rendered PASS after the fix:
  129 frames at 864x480, exactly 5.160 s, zero audio streams)

## PBUG-20260812-02 -- a Pydantic field named `register` silently becomes a BOUND METHOD, and the writer dies serializing it

- surfaced: LIVE headless leg, 2026-08-12 06:08, the first leg of the 45-word
  every-visual-path campaign (`otr_w45_campaign.py --only still_flat,...`,
  profile `otr_w45_still_flat`, source bank rolled to `scifi_fable2`). The node
  `OTR_LedgerScriptWriter` failed with
  `TypeError: Object of type method is not JSON serializable`, 78 s in, before
  any video work. Leg verdict: `FAIL (exit=1, 1.3 min) no new file in otr/obs`.
- symptom: an episode dies in the WRITER, at
  `_otr_scifi_fable2.py:1532` in `_script_user_prompt`, on
  `json.dumps(treatment.model_dump())`. Nothing about the message names the
  field or the model, so the failure reads as a generic serialization bug.
- **root cause, REPRODUCED exactly.** `CastShape.register`
  (`_otr_scifi_fable2.py:281`) shadows an attribute that exists on
  `BaseModel` -- Pydantic's `ModelMetaclass` inherits `ABCMeta`, so
  `BaseModel.register` is a bound metaclass method. Pydantic does NOT reject
  this field name: the clash is on the metaclass, not the class body.
  **It instead adopts the inherited attribute as the field's DEFAULT.**

  > **CORRECTED 2026-08-12, and the first version of this entry was wrong in a
  > way that would have misdirected the fix.** It claimed the field was still
  > required and that only `model_construct` could leak the method. Neither is
  > true. Measured on this box against the real module:
  >
  >     CastShape.model_fields["register"].is_required()   # False
  >     CastShape.model_fields["register"].default
  >     # <bound method ModelMetaclass.register of <class '...CastShape'>>
  >     CastShape.model_json_schema()["required"]
  >     # ['name', 'role', 'want', 'pressure']      <-- no 'register'
  >     CastShape(name="Ada", role="lead", want="w", pressure="p").register
  >     # <bound method ...>                        <-- ORDINARY VALIDATION
  >
  > So three things happened silently, and the crash was the LUCKY one:
  > 1. a field the author declared as required went OPTIONAL;
  > 2. **the JSON schema handed to the writer stopped listing `register` in
  >    `required`**, so the model was never obliged to produce a documented,
  >    load-bearing contract field (doc s5: HOW a character speaks);
  > 3. any shape that omitted it carried a bound method into
  >    `_otr_scifi_fable2.py:1527`, which renders into the prompt as
  >    `register: <bound method ModelMetaclass.register of ...>` -- a corrupted
  >    writer prompt -- before dying on the next `json.dumps`.

  Pydantic warned twice and both warnings went where nothing reads them: a
  `PydanticJsonSchemaWarning` at schema build (`Default value <bound method ...>
  is not JSON serializable; excluding default from JSON schema`, visible in the
  server logs) and a `PydanticSerializationUnexpectedValue` on the dump.
  **A `warnings.filterwarnings` at `_otr_scifi_fable2.py:107` was silencing the
  third one** -- pydantic's own `Field name "register"` shadowing warning --
  with the comment "nothing in this module ever calls the shadowed attribute".
  That reassurance was aimed at the wrong hazard: the danger was never a call.
- **why it is INTERMITTENT, which is what makes it nasty:** the campaign rolls
  `--source-bank "roll (any eligible bank)"` per leg, so only legs that roll
  `scifi_fable2` can hit it, and only when a `CastShape` reaches the prompt
  builder without its `register` set. A re-run can pass and look like a flake.
- **the production TRIGGER, once the mechanism was right, needed no searching.**
  The first version of this entry hunted for a `model_construct` call and found
  none, and concluded the trigger was unidentified. There is nothing to find:
  the treatment is built by `structured_call(schema=Treatment, ...)`, which
  validates -- and validation ACCEPTED a missing `register`, because the schema
  it generated never required one. The trigger is simply **the model omitting an
  optional field**, which it was entitled to do. That also explains the
  intermittency below without any flakiness: it depends on the model's output,
  not on a rare code path.
- **why it looked INTERMITTENT:** the campaign rolls
  `--source-bank "roll (any eligible bank)"` per leg, so only legs that roll
  `scifi_fable2` can hit it at all, and then only when the writer happened to
  omit `register` for at least one cast shape. A re-run can pass and read as a
  flake.
- **fix: SHIPPED.** `register: str = Field(...)` at `_otr_scifi_fable2.py:281`.
  The name is load-bearing prompt vocabulary and could not change, so the class
  body shadows the inherited attribute with an explicit required marker --
  restoring exactly what the bare `register: str` annotation was always meant to
  say. Measured after: `is_required()` True, schema `required` now contains
  `register`, omitting it raises `ValidationError` (honest and retryable through
  the repair ladder) instead of poisoning a prompt, and `model_construct`
  without it now raises `AttributeError` rather than handing back a method.
  The neighbouring `warnings.filterwarnings` comment was rewritten to say what
  actually makes the filter safe.
  - Rejected: RENAMING the field. It changes the structured-output schema the
    model is answering and the `register:` label in the cast block at
    `_otr_scifi_fable2.py:1527` -- a contract change to fix a defaulting bug.
  - Rejected: `register: str = ""`. It stops the crash but keeps the field
    optional, so the writer still is not asked for it and a character silently
    gets a blank speaking register. That is the containment, not the fix.
- verify: `tests/test_writer_model_field_shadowing.py` -- 10 tests. The field is
  required; the emitted JSON schema requires it; omission refuses; the contract
  spelling is unchanged; `model_construct` cannot return a method. **Plus the
  general rule, swept over every pydantic model reachable under `nodes/` (92 of
  them): no field default may fail `json.dumps`.** That is the check that would
  have caught this the day the field was added, it names the CLASS of defect
  rather than this one field -- `copy`, `json`, `schema`, `dict`, `validate` and
  `construct` are all waiting to do the same thing -- and it is guarded by a
  companion test asserting the sweep actually found the models, so a broken
  import cannot make it vacuously green. `register` was the only offender.
- **the defect had no visible symptom in the source**, which is why the
  executable check matters more than usual here: deleting the `Field(...)`
  restores a perfectly ordinary-looking `register: str`.
- status: FIXED. Proven by unit test and by measurement; a live `still_flat`
  re-run is queued to close it on an artifact.
- bible-worthy: **yes, strongly, as a class.** "A Pydantic field whose name
  collides with an attribute of BaseModel/its metaclass serializes as a bound
  method whenever the instance is built without it" is portable to every project
  using Pydantic v2, the failure is silent until a `json.dumps`, and the error
  message names neither the model nor the field.
- status: OPEN -- root cause PROVEN and reproduced, production trigger not yet
  located, no fix attempted.

## PBUG-20260812-03 -- the repair rule could not fire on the defect class it was written for, and a leg died four attempts running

- surfaced: LIVE headless leg, 2026-08-12 06:52, the `viz_green` leg of the
  45-word every-visual-path campaign (profile `otr_w45_viz_green`, source bank
  rolled to `scifi_fable2`). `OTR_LedgerScriptWriter` failed after 3.0 min:
  `[scifi_fable2] pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; last defects: - UNKNOWN_SPEAKER: *SFX (line 25) - SKELETON_BREAK:
  character line (*SFX) after the last scene`. Leg verdict: `FAIL (exit=1)
  no new file in otr/obs`. The episode never reached a video engine.
- symptom: the markup ladder burns all four attempts on the SAME defect and
  gives up. From the outside it reads as a model that cannot follow the format;
  in fact the model was never told what the fix was.
- **root cause.** `_standalone_stage_direction_repair_note`
  (`_otr_scifi_fable2.py:1609`) exists to hand the repair rung an explicit rule
  for an illegal stage-direction row. It fired only when the defect code was
  `BAD_LINE_SHAPE` **and** the detail opened with `(` or `[`. A stage direction
  written WITH A COLON -- `*SFX: a door slams` -- does not take that path at
  all: it matches the speaker catch-all, so the code is `UNKNOWN_SPEAKER`
  (plus `SKELETON_BREAK` when it lands outside a scene) and the detail opens
  with `*`. Three independent ways to miss one rule. The note returned `""`,
  the rung got only the generic "Repair only the malformed FORMAT defects
  below", and the model re-emitted the same shape until the ladder exhausted.
- **this is the THIRD time this ending is on the record**, and the previous two
  are written into `_otr_fable2_markup`'s own module docstring: a decorated
  label falling to the `_RE_SPEAKER` catch-all, `UNKNOWN_SPEAKER` four attempts
  running, the leg dying in the writer. Those two were fixed by teaching the
  PARSER to normalize the decoration (balanced `**` shapes 1-4). This one is
  the other half of the same story -- when the parser CORRECTLY refuses, the
  repair rung has to be able to say why.
- **fix: SHIPPED IN TWO PASSES, and the first pass was half wrong.** It is
  PROMPT-ONLY throughout -- it cannot make an invalid script valid, and no
  parser, acceptance rule or schema changed.

  **Pass 1 (`3a5cf77f`), CORRECTED BELOW.** Widened the codes to
  `BAD_LINE_SHAPE` / `UNKNOWN_SPEAKER` / `SKELETON_BREAK` and the prefixes to
  `(`, `[`, `*`. The QA pass then holed it, and both findings were confirmed
  against the producer:
  - **the `SKELETON_BREAK` widening was DEAD CODE.** Every `skeleton()` detail
    in `_otr_fable2_markup` is a descriptive English sentence that merely
    CONTAINS the token -- the real row is `character line (*SFX) after the last
    scene`, which opens with "character". It can never match a prefix test.
    Nothing was lost by it being dead: the same line always raises
    `UNKNOWN_SPEAKER` first. **Any earlier wording in this entry claiming the
    shipped fix fires on `SKELETON_BREAK` is withdrawn.**
  - **a test propped it up with a FICTIONAL row** (`SKELETON_BREAK: [SOUND]
    after the last scene`), a shape no producer emits. It passed and proved
    nothing -- the same lexical-fixture trap as lesson L26.
  - and a real gap: a cast member wearing a stray unmatched marker
    (`*Ada: Hello`) survives canonicalization, misses the roster, and raises
    `UNKNOWN_SPEAKER: *Ada` -- shaped exactly like `*SFX`. Pass 1 would have
    told the model to fold or drop a real character's line.

  **Pass 2, after a kibitz consult** (`kibitz-runs/2026-08-12-writer-stage-
  direction-note/r2/`, Codex `gpt-5.6-sol` + Antigravity; operator rule: one
  failed fix on the writer, then the panel). Three findings would have shipped
  past me:
  - **`Fable2ParseDefect` is a plain `enum.Enum`, not a `str` enum.** Passing
    typed defects while comparing `defect.code` against string constants would
    be False forever -- the note silently never fires and every stage-direction
    leg burns four attempts again, from code that reads correctly. Codes are now
    compared as ENUM MEMBERS.
  - **the two codes carry different data.** `UNKNOWN_SPEAKER.detail` is the bare
    label; `BAD_LINE_SHAPE.detail` is a line fragment (`line[:80]`). One token
    extractor over both is a category error, so the note branches by code and
    does roster resolution only for `UNKNOWN_SPEAKER`.
  - **the prompt was handing the model false evidence** -- it called `detail`
    "the exact source row", which it is for neither code. It now says
    "illegal speaker label" or "may be truncated", and carries `line_no` as its
    own field.

  Shipped shape: the note takes TYPED `ParseDefect` objects (`str(defect)`
  appends ` (line N)`, and re-parsing that string is what corrupted the token),
  `cast_names` is keyword-only and required, and there are **two rules rather
  than one rule and a mute button** -- a decorated ROSTER name is told to
  restore the canonical label and keep the dialogue, because going silent just
  returns the generic instruction that already failed four attempts.
  `defect_rows` stays a string tuple for `PassAttemptTrace`, which enforces
  `tuple[str, ...]`.

- **REJECTED: fixing this in the parser.** The obvious move is to have
  `_canonicalize_transport_line` strip a stray unmatched leading marker so
  `*Ada` resolves outright. Both panel lanes independently said no, and the
  second reason is the non-obvious one: strip the marker from `*SFX:` and you
  get `SFX:` -- still not a roster name, so still `UNKNOWN_SPEAKER`, but it no
  longer LOOKS like a stage direction, so this note stops firing and the ladder
  falls back to the generic instruction that caused the live failure. The parser
  "fix" would silently disarm the real fix. Recorded as a comment in
  `_canonicalize_transport_line` so it is not re-derived.
- **REJECTED: deterministic Python folding of the offending row.** Deciding
  where a sound event belongs, or whether it is dispensable, is authored story
  work; `docs/PRODUCTION_SPRINT_LESSONS.md` requires returning ambiguous
  placement to the model and failing closed.
- **what was deliberately NOT widened:** an `UNKNOWN_SPEAKER` for a MISSPELLED
  CAST NAME still gets no stage-direction advice. Telling the model to fold a
  real character's line into a neighbour, or to drop it, is worse than the
  failure it replaces -- so the note requires the token to LOOK like a stage
  direction, not merely to be an unknown speaker.
- verify: 33 tests across two files, and **every fixture is derived from the
  REAL PARSER** rather than hand-written -- that is what the fictional-row trap
  cost.
  - `tests/test_fable2_stage_direction_repair_note.py` (27): the live rows
    produce the rule; five real stage-direction shapes fire; a decorated roster
    name gets the RESTORE rule and explicitly not the delete-a-character advice;
    a decorated typo still gets the stage-direction rule; an undecorated unknown
    name gets nothing; `*Ada (Engineer)` keeps its role parenthetical; and a
    real `SKELETON_BREAK` detail is asserted UNABLE to open with a marker, which
    is the assertion that would have caught the invented fixture.
  - `tests/test_fable2_ladder_delivers_repair_rule.py` (6): **drives the real
    `_run_markup_ladder` with a scripted writer and asserts on the SECOND
    prompt.** Every other test checks what the note RETURNS; only this one
    proves the string is put in front of the model, which is the property the
    live failure actually turned on. It also pins that a model which never
    repairs still fails closed, and that telemetry keeps immutable string
    defects.
- **NOT proven on a live leg yet.** Verified by unit test, by the end-to-end
  ladder test, and by the mechanism; whether the model actually repairs when
  told needs a leg that rolls `scifi_fable2` again. Queued behind the cross-bank
  writer gate (operator directive 2026-08-12: a writer fix proven on one bank is
  re-tested against every other bank before it is called closed).
- bible-worthy: probably, as a class -- "a targeted repair/remediation rule
  scoped so narrowly that it cannot fire on the defect it was written for, so
  the retry loop burns its budget on identical failures". Deferred with
  PBUG-20260812-02 until both writer defects are settled.
- status: FIXED (prompt-only), pending the cross-bank gate and a live re-run.

## PBUG-20260812-04 -- a live pydantic model rode `dict.update` into the ledger, and the writer died saving it

- surfaced: LIVE headless leg, 2026-08-12 08:17, the `public_domain` leg of the
  cross-bank writer gate (`scripts/otr_writer_bank_gate.py`, profile
  `otr_w45_still_flat`, 45 words). `OTR_LedgerScriptWriter` ran 123.66 s then
  raised. Verdict: `FAIL (exit=1) 2.2 min`.
- symptom, from `tmp/_bankgate_server.log:1845`:

      [Ledger] save failed: Object of type VisualStyleCardModel is not JSON serializable
      RuntimeError: failed to save ledger after visual_style pack embedding
        at nodes/OTR_LedgerScriptWriter.py:6394, in _run_writer_tail

- **root cause.** `run_story_brief_reflection` returns a meta delta that, on the
  dynamic-visual-style path, carries `visual_card` as a LIVE
  `VisualStyleCardModel` rather than a dict (`nodes/_otr_story_brief.py:643`).
  `OTR_LedgerScriptWriter` merged that whole delta with `meta.update(_brief_delta)`
  -- and **`meta` IS the ledger.** A serialized copy is written moments later as
  `meta["visual_style_card"] = _card.model_dump()`, which is what made this look
  handled; but writing the copy never removed the raw model sitting beside it.
  The next `led.save()` called `json.dumps(ledger)`, which refused.
- **NOT bank-specific, and the first read of it was wrong.** The reviewing panel
  proposed that `public_domain`'s extra `meta.provenance.*` block was involved.
  It is not. The trigger is `_is_dynamic_style` -- `meta["visual_style"] ==
  DYNAMIC_STYLE_ID` -- so **any bank can hit this whenever the visual-style roll
  lands on dynamic.** Every leg of the 45-word campaign runs
  `--visual-style "roll (any style)"`, so this defect was live for the whole
  sweep and would have kept taking legs at random.
- fix: SHIPPED, two parts.
  1. **The defect.** `_card = _brief_delta.pop("visual_card", None)` BEFORE the
     `meta.update`, so the model never enters the ledger at all. It is a working
     value for the pack composer, not ledger content. Verified nothing reads
     `meta["visual_card"]`: the single reader took it from the delta and now
     takes it from that local.
  2. **The diagnosability.** `json.dumps` names only the offending TYPE, and the
     ledger carries 600+ keys, so the message started a hunt instead of ending
     one. `_where_unserializable` (`nodes/_otr_ledger.py`) now walks the ledger
     ON THE FAILURE PATH ONLY and appends the dotted location:
     `-- at meta.visual_card (VisualStyleCardModel)`. It handles cycles by id,
     returns "" for any non-serialization failure so a disk or permission error
     is never dressed up as one, and cannot itself raise.
- **THIS IS THE SECOND NON-JSON VALUE TO REACH `json.dumps` IN ONE DAY**, after
  PBUG-20260812-02's bound method. Different objects, same shape of mistake, and
  both were one `dict.update` away from recurring -- which is why the fix names
  the class (a locator for any unserializable value) and not just this field.
  Note also that -02's guard would NOT have caught this one: that test sweeps
  pydantic field DEFAULTS, whereas this was a value placed into a dict at
  runtime. The two guards are complementary, not redundant.
- verify: `tests/test_ledger_unserializable_diagnosis.py` -- 19 tests. The live
  shape is located by dotted path; an offender nested in a list is found; a
  clean ledger reports nothing; disk/permission/other errors get no path; a
  recursive ledger still yields its diagnosis rather than losing it to a
  RecursionError; a shared subtree is not mistaken for a cycle; a hostile
  `__repr__` cannot take down the handler; and end to end
  `save_ledger_safe` returns False, logs the path, and leaves NO partial ledger.
- **PROVEN ON A LIVE LEG, 2026-08-12 09:17.** Re-ran `public_domain` against a
  server booted with the fix: **PASS, exit=0, 9.1 min**, against the 2.2-minute
  death before it. The `--only`-style re-run also cleared the unrelated harness
  fault that had corrupted the first attempt (see below).
- **harness fault, same leg, NOT a product defect:** the leg log also shows
  `NameError: name 'describe_execution_error' is not defined` at
  `scripts/otr_api.py:749`. That was self-inflicted -- `otr_api.py` was edited
  between two edits while the gate was spawning subprocesses that import it, so
  that leg imported a half-applied file. **Never edit a module mid-campaign when
  legs import it per-subprocess.** The file is consistent now.
- status: FIXED and CLOSED -- proven by a live passing leg.

## PBUG-20260814-01 -- every spoken line ships `speaker: None` because the shared line normalizer drops the field its sibling carries

- artifact: `output/otr/episodes/signal_lost_the_light_of_possibility_20260813_172801/audio/..._ledger.json`
  -- the accepted, PUBLISHED 2026-08-13 `wan_ti2v` episode. `meta.source_bank = scifi_news`.
- symptom: all 11 spoken rows (`l001`..`l011`) carry `speaker: None`. The two
  music rows carry `speaker: "RADIO"` correctly, and the BEAT rows carry the real
  names -- `b000` Dr. Ada, `b003` Dr. Leo, `b005` MIT Ethics Board. So speaker
  identity EXISTS at beat level and is lost on the way to the line.
- consequence: nothing on a line asserts who says it; only an opaque `char_id`
  survives. This is the mechanism behind the operator's "John speaking Mary's
  lines" complaint. Visible in the same artifact: `b005` belongs to the MIT
  Ethics Board, but its second row is narration about **Ada** leaving the room.
- root cause: `nodes/production_ledger.py:1252-1309`, `Ledger.set_lines()`. Its
  normalized row schema (`:1281-1300`) enumerates the keys it keeps, and
  `"speaker"` is NOT among them -- a caller that passes one has it silently
  discarded. Its SIBLING `Ledger.set_beats()` (`:1118`) does carry it:
  `"speaker": _safe_str(r.get("speaker")) or None`. **The bug is the asymmetry
  between two normalizers for parallel row types.** Both news lanes also fail to
  supply it upstream (`_otr_scifi_codex.py:3211`, `_otr_scifi_fable2.py:2749-2786`);
  `ScriptLineV4` (`_otr_scifi_codex.py:760-784`) has no speaker field at all.
- why it shipped unnoticed: `production_ledger.py:1806-1822` has a
  `cast[].char_id -> name` fallback that masks the omission for ONE consumer.
  The raw ledger JSON on disk still says `speaker: None` on every line row.
- the fix must land in BOTH halves at once: add `speaker` to the `set_lines()`
  normalized schema mirroring `set_beats()`, AND carry the owning beat's speaker
  onto the line row at each assembly site. Fixing only the lane leaves
  `set_lines()` dropping it; fixing only `set_lines()` leaves the lanes not
  supplying it. Because the root is the SHARED ledger method, a lane-only fix
  leaves every other bank broken.
- checked against `otr_coverage_index.yaml` and `BUG_BIBLE.yaml` 2026-08-14:
  **genuinely uncovered.** No entry covers "sibling normalizers disagree on a
  shared field, so one silently drops it".
- fix (2026-08-14): both halves in one change. `set_lines()` now names
  `"speaker"` in its normalized schema, mirroring `set_beats()` exactly
  (`_safe_str(...) or None`, key always present). Every assembly site supplies
  it from the row that already owns it: `init_lines_from_outline()` stamps the
  outline beat's speaker on the line row it pre-stamps; `_otr_scifi_codex
  ._assemble_ledger` takes `b.speaker` off the OWNING BEAT rather than asking
  P5 for it (`ScriptLineV4` is unchanged -- the score already named who holds
  the beat); `_otr_scifi_fable2._spoken_row` gains a required `speaker` and
  `_beat()` now READS it back off the line row instead of being handed it a
  second time, so the two cannot disagree by construction; the story
  orchestrator's streamed partial ledger carries the streamed name.
  `tests/fixtures/fable2/golden_s1b_assembly.json` regenerated deliberately
  (assembly contract changed by design; the three music sentinel rows own no
  beat and carry null).
- promotion: BUG-12.101 (survival-guide `a6035b2`, the three-file contract in
  one commit: YAML entry, README count 278 -> 279, executable coverage
  `TestThreeFileContract::test_otr_sibling_row_normalizers_name_the_same_speaker`
  which diffs the key sets both normalizers build out of the AST, and the
  `otr_coverage_index.yaml` row).
- status: FIXED 2026-08-14. Suite 10359 passed / 110 skipped / 1 xfailed
  (baseline 10355 + the four new guards), Bug Bible 20/25/3,
  `build_variants --check` 50 variants 0 failures.

## PBUG-20260814-02 -- `scifi_news` publishes with NO announcer coda, and never names the news story it was drawn from

- artifact: same published 2026-08-13 episode as PBUG-20260814-01.
- symptom: the episode contains exactly ONE announcer row, `l003`, and it sits in
  the MIDDLE. There is no announcer opening and no announcer close. The real news
  behind it (light-activated cell movement, starfish cells, MIT) is never named
  anywhere in the ledger. Structure shipped:
  `music_opening -> l001..l002 -> l003 ANNOUNCER -> l004..l011 -> music_closing`.
- expected per the bank's own `coda_mode: real_news_report` and the canonical
  topology: opening music -> ANNOUNCER introduction -> drama -> ANNOUNCER
  source-backed news summary -> closing music.
- root cause: the coda is PROMPT TEXT ONLY. `codex_coda_contract_system` is never
  its own pass -- it is string-concatenated onto the P3 score prompt
  (`_otr_scifi_codex.py:3557`) and the P5 script prompt (`:2756`). Those are the
  only two occurrences of "coda" in the 3,639-line lane. There is no coda output
  type, no beat kind for it, nothing reserving a final announcer row, and
  **nothing verifying afterwards** that a coda exists or names the source. The
  only structural rule is cast coverage (`:1238-1269`): every cast row needs at
  least one beat SOMEWHERE, position unchecked. A model that gives the announcer
  one beat mid-episode passes every gate -- which is exactly what shipped.
- the sibling lane already solves it: `scifi_news_pro`'s markup parser REFUSES a
  draft with no coda or no announcer outro (`_otr_fable2_markup.py:432-444`),
  rejects a second coda (`:419-430`), rejects closing music before the coda
  (`:385-391`), and requires opening music + intro before the first scene
  (`:398-402`). Its `fable2_news_read_system` is a DEDICATED pass whose validated
  output is unconditionally Python-appended as its own row (`_otr_scifi_fable2.py:2897-2908`).
  **So the fix is levelling the non-pro lane up to the pro lane's structure, not
  inventing one.**
- fix (2026-08-14): the coda stopped being decoration and became **P6**, a pass
  that owns one job. It runs AFTER P5 -- it is written against the episode the
  listener actually just heard, not against a plan of it -- and reuses the
  already-authored `codex_coda_contract_system` seam as its own system message,
  so the JOB SIZE changed and the prompt text did not (the standing "one prompt
  per job for every model tier" law).
  * **Detector, code-side:** `_news_coda_source_anchors` projects the P0 index
    into the verbatim strings a coda would have to say -- entity names (three
    characters or more, because a two-letter "entity" matches inside ordinary
    words and would wave a bad coda through) plus source-spanned figures.
    `_names_a_source_anchor` matches on word boundaries, so "MIT" is not found
    inside "transmitted". `_news_coda_findings` reports the missing
    attribution AND any spoken-text defect together, because the clean pass
    gets one bounded turn and a one-defect-at-a-time validator spends it on
    the first complaint and dies on the second.
  * **A firing verifier triggers a CLEAN, never a refusal and never a reroll**
    (operator ruling). The ladder's `post_validator` is deliberately empty and
    `retry_until_valid=False`: a coda missing its attribution is a good draft
    missing one thing, so it comes back once with `previous_attempt` and
    `unmet_requirements` attached instead of being redrawn cold.
  * **Three outcomes, all of which CONTINUE the render:** `clean`; `unclean`
    (it ships anyway and `meta.scifi_codex.news_coda.status` says so, because
    an imperfect attribution beats none); `absent` (nothing is appended and
    nothing is invented -- Python does not author the sentence).
  * **Placement is now a property of the code, not a hope about a draft.**
    `_assemble_ledger` Python-appends the validated row last, exactly as the
    pro lane appends its news read, and `_assert_news_coda_is_last` re-asserts
    the pro lane's three parser rules afterwards: exactly one coda, an
    announcer speaks it, nothing spoken follows it.
  * **Right-size the job, never raise the guard:** the pass carries its own
    384-token decode budget instead of running on provider capacity like the
    whole-script pass.
  * Coverage: `tests/test_codex_news_coda.py` (19 tests) plus
    `test_the_assembled_coda_is_the_last_row_and_the_announcer_speaks_it` and
    the P6 row in the fixed-topology assertion in `tests/test_scifi_codex_lane.py`.
    `nodes/story_packs/pipelines.json` declares the pass so the written
    topology matches the executed one.
- status: FIXED 2026-08-14.

## PBUG-20260814-03 -- the sealed ledger is narrated third-person prose, so TTS reads stage directions on air

- artifact: same published 2026-08-13 episode.
- symptom: this is not leakage at the edges, it is the DOMINANT mode. Verbatim
  rows that were read aloud: `l002` is 100% narration with no dialogue at all
  ("Ada's fingers drum on the desk, lost in thought..."); `l007` is
  "**Leo sighs, running a hand through his hair.** '...' The room falls silent";
  `l001` carries an attribution verb ("**Ada murmurs, her eyes reflecting the
  glow**"); `l011` is pure narration. **The quotation marks are INSIDE the text**,
  so dialogue is delivered as quoted speech embedded in narration.
- consequence: the episode is an audiobook being read aloud rather than a radio
  play being performed. This also explains the operator's "dubbed old film" note
  better than any timing theory -- the performance is narration, so nothing
  sounds like it is happening now.
- diagnosis (operator's own, `docs/2026-08-14-per-beat-dialogue-design.md` in the
  lab repo, written BEFORE this artifact was read): the dialogue job writes an
  ENTIRE ACT in one model call -- roughly 28 JSON rows in a single reply on a
  three-act story. *"A model asked for one beat writes that beat. A model asked
  for a whole act writes a summary of one."* Two measured lab failures from the
  same shape: `gemma-4-12b-it` truncated mid-object at the decode guard, and
  `gemma-4-E4B-it` wrote twelve beats of two researchers agreeing.
- contract violated: the sealed ledger holds announcer speech, character dialogue
  and music cues -- never a stage direction, action row, narration or delivery
  note. Every sealed line becomes TTS audio.
- fix (2026-08-14): the dialogue job is ONE BEAT now, and each scene is read
  back before anything downstream. `_call_script_text_draft` kept its name and
  its place in the lane but stopped being a single call; it is the schedule.
  * **Per scene: one `P5B` dialogue job per accepted beat, then one `P5R`
    review job.** The score's SCENE is this lane's act-sized unit. `act_count`
    still shapes the beat topology upstream and nothing here acquired a length
    target of any kind -- the beat count is the model's own answer to the score
    job, never a quota.
  * **The WINDOW is the fix, not the loop.** A beat job is handed this beat's
    intent/speaker/arc/facts, the spine of the scene it sits in, and
    `rows_so_far` -- what the listener has already heard -- so the writer
    answers the line before this one. It is NOT handed the accepted line
    graph; handing the model every row is what produced a summary.
  * **The review may rewrite, and may not add, drop or renumber.** It reads
    the scene's spine, beats and accepted rows and returns them unchanged or
    rewritten against the spine. Code detects and explains; a model rewrites.
    Review asks whether the scene plays; the per-beat validator already asks
    whether it can be sealed -- one prompt answering both answers neither.
  * **The next scene is written against what the review LEFT**, not against
    the draft it replaced.
  * **Per-job decode budgets ship in the same change**, because they are the
    same change: a beat needs a fraction of an act, so a right-sized budget
    stops the guard binding on honest work. `_BEAT_TEXT_MAX_OUTPUT_TOKENS`
    (2 lines) and `_SCENE_REVIEW_MAX_OUTPUT_TOKENS` (a scene) replace the
    whole-provider-window reservation. Right-size the job; never raise the
    guard -- no guard was touched.
  * **The beat's ARRAY ceiling is the beat's own**, not the script's. An
    unenforced array ceiling is the root cause of PBUG-20260729-02; a job that
    can legally emit two rows cannot run away into twenty-four.
  * **Everything downstream is unchanged on purpose.** The accepted rows are
    assembled into the same `ScriptTextDraftV4` the whole-play pass used to
    return, then compiled, canonicalized and validated by exactly the same
    code. The canonical-surface hygiene check moved INTO the per-beat
    validator, where a finding can still reroll -- per beat, canonicalization
    happens once every beat is in, which is far too late.
  * One prompt per job for every model tier: both jobs use the lane's existing
    `codex_play_system` seam. The JOB SIZE changed; the prompt text did not.
  * Coverage: `tests/test_codex_per_beat_dialogue.py` (12 tests) plus the
    DERIVED schedule assertion in `tests/test_scifi_codex_lane.py` -- built
    from the accepted score rather than hard-coded, so a schedule that stopped
    following the graph goes red. `nodes/story_packs/pipelines.json` declares
    both passes.
- status: FIXED 2026-08-14 in code. **The published-artifact proof is a
  separate act:** only a generated episode shows whether the ledger is
  dialogue rather than narration, and that read is recorded in the handoff.

## PBUG-20260814-04 -- the clean stage ran BLIND to the act on every writer-lane row

- discovered: 2026-08-14, live `media_archive` episode at act 3
  (`gemma-2-2b-it`), during the first legs of the new post-story clean stage.
- symptom: the pass judged and rewrote spoken lines with an EMPTY act block.
  `beat_intent` resolved on **0 of 16** rows and `arc_phase` on **0 of 16**,
  so every judge and repair prompt shipped with "WHERE THE STORY IS" blank.
  Nothing complained: the render succeeded, the log looked normal, and the
  full suite was green throughout.
- root cause: `nodes/_otr_ledger_clean.py` read `arc_phase` / `beat_intent`
  off the BEAT row. The writer lane stamps both on the LINE row and leaves
  its beat rows carrying transport only (`beat_id`, `char_id`, `line_ids`,
  `scene_id`, `shot_id`, `start_s`, `dur_s`). The codex lane populates both,
  which is why the shape looked right when it was written.
- why it stayed invisible: the unit fixtures put the fields on the BEAT --
  a shape no writer-lane ledger has -- so the tests agreed with the bug. A
  green suite cannot see a field that is merely always empty.
- fix: read the UNION, line row first (`_story_field`), since the codex lane
  does populate beats. Plus three things so it cannot recur silently:
  * `meta.ledger_clean.context_seen` counts, per episode, how many rows
    resolved an arc phase, a beat intent, a cast name, and lines before and
    after. A zero in that block IS the blindness, visible in the artifact.
  * `meta.ledger_clean.context_verified` SHA-verifies that the context we
    BUILT appears in the prompt bytes we SENT, reported as a fixed-position
    sight string (`11111` = all five fields landed, `11011` = the act did
    not). This catches the second shape of the same fault, which the field
    count cannot: a value threaded into a prompt builder and never rendered.
  * a test against the REAL production ledger shape, plus one against the
    codex shape, so neither lane can regress alone.
- coverage: `tests/test_ledger_clean_stage.py` --
  `test_the_act_is_read_from_the_line_row_where_production_puts_it`,
  `test_the_receipt_proves_what_the_model_actually_saw`,
  `test_the_beat_row_still_works_for_the_lane_that_populates_it`,
  `test_the_sha_check_proves_the_context_reached_the_prompt`,
  `test_a_field_built_but_never_rendered_is_caught_and_named`.
- bible: **already covered by `BUG-12.86`** (receipt keyed on a string the
  producer never emits) -- cause shape 3, "a dict/key lookup for a key that
  was renamed or never existed on that row shape". Its verify clause already
  prescribes the fix arrived at here independently: *"a prompt-context test
  must check that the field's CONTENT reached the prompt, not merely that the
  label appears."* Checked against `otr_coverage_index.yaml` and
  `BUG_BIBLE.yaml`; NOT promoted, index row appended instead.
- status: FIXED and PROVEN 2026-08-14 -- the lab reports `act on 6 row(s),
  intent on 6, brief on 6, before 5, after 5` on every bank, and a
  production-shaped ledger returns `sight 11111 ok true`.

## PBUG-20260815-01 -- the clean stage DELETES the source attribution it was never meant to touch

- artifact: `signal_lost_reel_of_mystery_20260815_041350` (media_archive),
  `signal_lost_midnights_ticktock_20260815_045020` (public_domain),
  `signal_lost_ghost_of_elsinore_20260815_050626` (shakespeare). Overnight
  six-bank gate, 2026-08-15.
- symptom: operator on the archive episode -- *"What news story???"*. The
  closing announcer said only *"Clarisse's gaze meets the reel's enigmatic
  label"*.
- what actually happened: `meta.ledger_clean.rows[b016].before` holds the
  COMPOSED row -- an authored bridge, `": "`, then the factual close *"In other
  news, the Library of Congress announces its film loans for the month,
  including 'None But the Lonely Heart', 'Symphony of Swing', and 'The Man With
  the Golden Arm'."* `.after`, and the row that shipped, is the drama alone.
  **The entire source note was deleted by the clean stage.**
- root cause: `nodes/_otr_ledger_clean.py` (shipped 2026-08-14) judges every
  voiced row for "anything that is not speech" and has a MODEL rewrite what it
  names. The closing announcer row carries a deterministic, PYTHON-OWNED fact
  (`_otr_provenance.spoken_coda_line` output, or the interpreter's
  `news_close_brief`), appended verbatim after an LLM bridge by
  `_otr_line_composer._assemble_news_coda_surface:1441-1446`. The pass has no
  concept of a protected component, so a model rewrote a fact Python owns.
- consequence: the interpreter was NOT at fault -- `meta.news.news_close_brief`
  is factual and correct in the artifact. On `midnights_ticktock` the ledger
  still advertises `spoken_coda_source: "provenance"` and
  `provenance_coda_line` still holds the original sentence, so the RECEIPT AND
  THE SPOKEN ROW DISAGREE. Rewrite rate 9 of 14 voiced rows on all three.
- why it shipped unnoticed: the clean stage was proven on
  `scripts/otr_clean_stage_lab.py`, a planted-ledger measurement rig that never
  runs the writer tail and therefore never composes a coda.
- verify idea: for any ledger whose `meta` carries a Python-owned coda fact,
  assert that fact appears byte-identically exactly once in the final announcer
  row. Anchor on the durable meta value with `endswith`, NOT on a `": "` search
  -- `_assemble_news_coda_surface` omits the separator entirely when the bridge
  is empty (flag `news_coda_fact_only`).
- bible-worthy: yes -- "a model-owned rewrite pass was given authority over a
  field another owner writes deterministically". Generalizes well beyond OTR.
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `e7e10148` wired `PROTECTED_FACT_COMPONENT_FLAG` producer (`OTR_LedgerScriptWriter.py`) to consumer (`_otr_ledger_clean.py`). Live proof from an episode published the same day: `signal_lost_the_16mm_ransom_20260818_145217` line `b016` carries `['news_coda_bridge', 'protected_fact_component']`.
- previous status: OPEN -- diagnosed, fix specified in
- status: **CLOSED 2026-08-18 -- FIXED AND FIRING TODAY**
  `docs/2026-08-15-BUILD-CONTRACT-bugfix-sprint.md`, not yet landed.

## PBUG-20260815-02 -- `scifi_news` dies at the pre-tail audit on the first row the clean stage repairs

- artifact: overnight six-bank gate 2026-08-15,
  `tmp/_bankgate_scifi_news.log:142` -- `CodexPreTailAuditError: line receipt
  mismatch for l004`. Died at 13.6 min, AFTER writing.
- root cause: `_otr_scifi_codex.py:4424` freezes `expected = {line_id: text}`
  and `:4455` hands it to `_CodexTailFinalizer`. AFTER that,
  `OTR_LedgerScriptWriter.py:6750` `run_ledger_clean` has a MODEL rewrite
  `lines[].text` and `:6758` `run_ledger_cleanup` blanks unsayable rows
  (`_otr_ledger_cleanup.py:256-263`). Only then, at `:6831`, does
  `before_save` compare live text to the frozen snapshot
  (`_otr_scifi_codex.py:3306-3308`). The receipt predates the clean stage and
  was never taught it exists. Deterministic: the leg dies at whichever row the
  clean stage repairs first.
- proof it is the clean stage, not the act topology: server log `:25934` --
  `l004 repaired -- judge named 2 segment(s)`; `l001`/`l002` shipped STILL
  UNCLEAN (text untouched) and passed the audit. `l004` is the first row
  actually rewritten and the first to fail. The plan's prime suspect was the
  2026-08-15 act change from 8 to 12 beats, but `:25998` records **8 voiced
  rows, not 12**, and the failure is at row 4, not past row 12.
  **The act-topology hypothesis is falsified; the named revert experiment was
  deliberately skipped rather than spending a live roll.**
- the fix is wider than the error: review found the reseal spans FOUR surfaces
  -- `_CodexTailFinalizer.expected` (`_proof` is expected-driven on both prongs
  and `after_save` re-proofs at `:3364`), `meta.scifi_codex.accepted_lines`
  (the full pre-clean TEXT dict), `meta.content_authorship` (stamped by BOTH
  content-owned lanes, enforced fatally at `_otr_freeze_cascade.py:803` ->
  `needs_full_rerun`), and the voiced-row coverage set.
- SECOND DEFECT INSIDE THIS ONE: raising here kills a render after 13.6 minutes
  of completed work, which violates the standing "a render must not die" rule.
  The audit must reject the transaction, roll back, stamp a degradation receipt
  and let the episode ship -- not raise.
- verify idea: mutate a row WITH an owning receipt (must pass) and WITHOUT one
  (must still fail); assert no render-path exception in either case.
- bible-worthy: yes -- "an integrity snapshot taken upstream of a later
  sanctioned mutator", plus "a post-generation audit whose only failure mode
  destroys finished work".
- FIXED 2026-08-15 (build-contract chunks 1 + 3). The clean/cleanup window is
  now ONE TRANSACTION: opened at `OTR_LedgerScriptWriter.py:6798`, immediately
  before `run_ledger_clean`, and reconciled at `:6819` after
  `run_ledger_cleanup` and before `stamp_text_for_tts_delivery`. New module
  `nodes/_otr_clean_transaction.py`; the codex lane grew the three-method proof
  protocol at `_otr_scifi_codex.py:3414-3462`
  (`snapshot_proof_state` / `restore_proof_state` / `reseal_proof`).
  - COMMIT PATH: ONE transition covers BOTH authorized stages -- one per stage
    is impossible by construction, since the second stage's pre-state is the
    first stage's output and could never equal the acceptance the chain starts
    from. The finalizer's `expected` and `meta.scifi_codex.line_text_sha256`
    are re-pointed at the authorized text, the mirror DERIVED from `expected`
    rather than rebuilt separately so the two cannot drift (12.86). The
    composite validator re-proves before the tail continues. A no-op emits
    nothing at all.
  - RECORD CORRECTION to the "four surfaces" bullet above:
    `meta.scifi_codex.accepted_lines` is PRESERVED, not resealed. It records
    what the model ACCEPTED, which is not what ships after an authorized clean;
    relabelling cleaned text as accepted output would destroy the only record
    that the two ever differed. The build contract settled this, and preserving
    it is what landed.
  - THE SECOND DEFECT IS CLOSED TOO -- an unprovable reseal no longer raises.
    The transaction restores the accepted ledger IN PLACE (the writer tail
    holds `meta` as a local, so rebinding the container would silently detach
    every later stamp), stamps `meta.content_transition_degraded` naming the
    cause, carries the rolled-back passes' telemetry INTO that receipt so no
    surviving `ledger_clean` entry claims edits that no longer exist, re-proves
    the restored state, and the episode ships without the repairs. Law 7 holds:
    the repair is sacrificed, never the render.
  - `meta.writer_word_delivery` is restamped after the window on every lane,
    under a NEW stage `writer_final_rows_post_clean`, so the pre-clean receipt
    survives beside it in `word_budget.actual_receipts` instead of being
    silently overwritten.
  - stale-comment fix: `_otr_readiness.py:264` claimed `phase_7_audio_readiness`
    "REWRITES canonical `line.text` in place". It does not, and has not for some
    time -- `:242-246` writes only `text_for_tts*`. That matters here because
    the reconcile ordering is only safe if nothing after it moves a hash the
    proof covers.
  - coverage: `tests/test_clean_transaction.py` (29 tests) -- commit path,
    no-op, rollback, in-memory state restore, container identity, the lane
    protocol, and an END-TO-END `_run_writer_tail` test proving the writer
    actually OPENS the transaction (a pass that ships and runs dormant is this
    defect's own shape). `tests/test_story_brief_c5a2.py`'s ordering pin was
    updated in the same change: it asserted exactly ONE `stamp_actual` call
    site and now pins two, by stage, plus the full order.
  - Bible: promoted as `12.104` -- the id `_otr_content_transition.py` already
    cited as a forward reference and which did not exist until now.
    `otr_coverage_index.yaml` row appended.
- **SONNET QA PASS 2026-08-15 found two defects the Fable gate did not**, both
  now fixed with coverage. (1) `_degrade` was called from a bare `except`, so
  anything IT raised -- a finalizer whose `restore_proof_state` threw -- left
  `reconcile()` and reached an UNGUARDED call site at
  `OTR_LedgerScriptWriter.py:6819`, killing the render. That is the Law 7
  violation this module exists to prevent, occurring inside its own rollback
  path. The fallback now has a fallback: it logs both failures, stamps an
  `outcome: rollback_failed` receipt and continues. (2) Two docstrings claimed
  more than the code delivers -- `reseal_proof`'s closing `_proof()` call is a
  CONSISTENCY check between two surfaces the same function just derived, not
  independent evidence, and `restore()` preserves object identity for the two
  TOP-LEVEL containers only, replacing nested rows with fresh deep copies. Both
  now say so.
- **RESEAL PATH LIVE-PROVEN 2026-08-15, and this closes the defect.** Leg
  `signal_lost_blood_red_water_20260815_195226` (scifi_news_pro, real canonical
  workflow): `RESULT SUCCESS`, `obs_publish OK`, **45.1 MB** on disk,
  `Prompt executed in 00:42:44`. The clean stage rewrote a row this time, so the
  transaction took the RESEAL branch rather than the no-op one:
  `[clean-transaction] authorized window resealed: 1 row(s) rewritten
  ['shot_004_b3'], 0 dropped`. The frozen ledger carries the receipt durably --
  `authorized_stages ['ledger_clean','ledger_cleanup']` (ONE window spanning
  BOTH passes), `affected_line_ids ['shot_004_b3']` (exactly the row that
  moved), a `parent_authorship_digest` binding it to this acceptance, 37 pre /
  37 post rows, and NO degradation receipt -- and the freeze cascade then
  returned `frozen_with_warns`, which is a clean freeze. That cascade is the
  code that used to raise `line receipt mismatch for l004`.
  `word_budget.actual_receipts` carries BOTH `writer_final_rows` and
  `writer_final_rows_post_clean`, so the restamp preserved the pre-clean record
  instead of overwriting it.
  **This lane matters specifically:** `scifi_news_pro` is where D3's `END` fix
  landed early and left the lane clearing the markup ladder only to die at the
  freeze cascade on this exact mismatch. Both halves are now closed on one
  render.
- **STILL NEVER EXERCISED LIVE: the DEGRADATION path.** `content_transition_degraded`
  was absent on both legs, correctly, because nothing failed. A rollback may
  never occur in production, which is the point of it -- it stays covered by
  unit tests only, and that is an accepted state rather than an outstanding task.
- status: **FIXED and LIVE-PROVEN on both content-owned lanes.**
  Superseded status line, kept for the record: FIXED, wiring live-proven,
  no-op path only -- Leg `signal_lost_the_architecture_of_error_20260815_152004`
  (scifi_news, real canonical workflow) is the FIRST successful run of this
  lane, `RESULT SUCCESS` + `obs_publish OK` + 33.4 MB on disk,
  `Prompt executed in 00:40:56`. The transaction ran in production --
  `[clean-transaction] the authorized window changed nothing; the acceptance
  receipt stands unmodified` -- and the freeze cascade then returned
  `freeze_verdict=frozen_clean (pre_warns=0 post_warns=0)`, which is the exact
  audit that used to raise `line receipt mismatch`.
  **WHAT THAT DOES NOT PROVE, stated so nobody over-reads it:** this episode's
  clean stage changed NOTHING, so the leg exercised the no-op path only. The
  RESEAL path (a transition stamped, the finalizer re-pointed) and the
  DEGRADATION path (restore + receipt) are still covered by unit tests alone.
  Whether the clean stage rewrites a row is model-dependent -- it rewrote 9 of
  14 on the three victim ledgers -- so proving those two needs a leg that
  happens to dirty a row, not another clean one. `scifi_news_pro` (D2/D3) is
  also still owed.

## PBUG-20260815-03 -- `scifi_news_pro` markup ladder exhausts because the terminal delimiter regex demands a period

- artifact: overnight six-bank gate 2026-08-15. `Fable2ScriptError: pass
  'script' failed after 4 attempt(s): markup ladder exhausted; last defects:
  BAD_LINE_SHAPE: END (line 23) | MISSING_END`. Died at 3.3 min.
- root cause: `nodes/_otr_fable2_markup.py:41` --
  `_RE_END = re.compile(r"^END\.\s*$", re.IGNORECASE)` requires a LITERAL
  PERIOD. The model wrote bare `END`, which falls past `_RE_END` (`:545`), past
  `_RE_SPEAKER` (`:548`, which needs a colon), and lands on `BAD_LINE_SHAPE`
  (`:552`) with detail `line[:80]` == `"END"`. Because `p.on_end` never fires,
  `p.saw_end` stays False and `:566` adds `MISSING_END`. **Both reported
  defects, one cause.**
- reproduced deterministically offline against the real module: `END`, `end`,
  `END ` (trailing space), `**END**` and `[END]` all fail; only `END.` passes.
  A fixture whose terminal line is bare `END` produces exactly the live defect
  pair; the same fixture with `END.` produces neither. Not length-dependent and
  unrelated to the act-topology change.
- the deeper defect: the ladder DOES repair
  (`_otr_scifi_fable2.py:2180-2192` carries the rejected draft plus rendered
  defects into the next turn) and still failed four times, because the message
  the model receives says its END is malformed AND missing without ever stating
  the required shape. A model cannot infer "add a period", so it re-emitted
  `END` on every rung. **Any defect that reports WHAT IS WRONG without WHAT IS
  REQUIRED is unrepairable by construction.**
- verify idea: every accepted END form parses to one delimiter; every transport
  defect detail contains the required shape; unpaired brackets and
  content-bearing variants still fail LOUD.
- bible-worthy: yes -- "a repair loop whose diagnostic names the offence but
  never the required shape, so the model cannot converge".
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `c8735ed1`. VERIFIED at HEAD: `_otr_scifi_news_pro_markup.py:60` reads `_RE_END = re.compile(r"^(?:END\.?|\[END\.?\])\s*$", re.IGNORECASE)` -- a bare `END` terminal is accepted.
- previous status: OPEN -- diagnosed, fix specified, not yet landed. Ordering note: this
- status: **CLOSED 2026-08-18 -- FIXED**
  must NOT land before PBUG-20260815-02's reseal, or the lane clears the ladder
  and then dies at the freeze cascade instead.

## PBUG-20260815-04 -- the public_domain cast is gender-INVERTED because the unit has no provenance sidecar

- artifact: `signal_lost_midnights_ticktock_20260815_045020` (public_domain).
  Operator heard GERTRUDE DEMONGMORENCI MCFIGGIN speak with a male voice.
- measured, not inferred: each character's own line windows were sliced from
  the delivered master (`start_s`/`dur_s`, windows verified contiguous and
  non-overlapping) and median F0 taken per line. GERTRUDE, a woman, renders
  MALE on all six of her lines (111.9 / 112.4 / 105.0 / 111.1 / 110.1 / 107.5
  Hz). **LORD RONALD, a man, renders FEMALE on all six of his** (279.1 / 233.0
  / 186.0 / 241.3 / 281.5 / 269.7 Hz) -- a second instance the operator did not
  report. The pair is RECIPROCALLY INVERTED, which is the signature of a blind
  roll rather than a bad pin. Audible in the script itself: the male-voiced
  character is addressed *"Miss McFiggins"* and the female-voiced one
  *"Lord Ronald"*.
- root cause: `cast_source_contract.gender_by_name` is `{}` while
  `character_names` lists all five source characters, because
  `config/source_banks/public_domain_story/sources/gertrude_governess.provenance.json`
  DOES NOT EXIST. Exactly one `.provenance.json` sidecar exists for 65 `.txt`
  units. With no roster the pin map is empty and the 40/40/20 largest-remainder
  roll stands.
- why nothing downstream caught it: both slots are `source_owned`, and
  `_repair_ensemble_names` EXEMPTS source-owned slots by design
  (`_otr_casting.py:682-684`) -- correctly, since renaming Sir Toby to satisfy a
  coherence rule loses the character. And `gender_of_first_name("GERTRUDE ...")`
  returns `unknown` anyway; the first-name pools carry 98 male names against 24
  female.
- everything BELOW the gender decision is correct, and this was verified rather
  than assumed: the voice picker honoured the tags faithfully
  (`vz_bill_boerst`, bank label male, measured 126 Hz, to the male slot;
  `vz_donor_glenn`, bank label female, measured 313.9 Hz, to the female slot),
  and the whole voice reference bank audits clean -- 41 references measured,
  ZERO label disagreements, males 89-146 Hz against females 155-314 Hz.
  **No voice-bank relabelling is needed.**
- verify idea: for any adaptation-lane episode, assert
  `cast_source_contract.gender_by_name` is non-empty and that every
  source-named character resolves; separately, a corpus gate that every manifest
  unit has a sidecar carrying a total `characters[]` roster.
- bible-worthy: yes -- "an armed consumer with no producer": the render path
  knew how to read a roster fact that vendor time never wrote for 64 of 65
  units.
- THIRD INSTANCE, 2026-08-15 afternoon, on a DIFFERENT source: artifact
  `signal_lost_the_price_of_a_soul_20260815_132024` (public_domain, Moby-Dick,
  unit `moby_dick_quarterdeck`). Operator heard **AHAB** -- a man, and one of
  the least ambiguous in English literature -- speak with a FEMALE voice. Root
  cause re-confirmed on this artifact rather than assumed:
  `meta.cast_source_contract.gender_by_name` is `{}` and
  `moby_dick_quarterdeck` has no `.provenance.json`. Sidecar census at the time:
  **16 tree-wide** -- 15 `shakespeare` plus `time_machine__arrival` -- so 64 of
  65 public_domain units still carry no gender facts. This instance matters
  because it is a THIRD independent reproduction on a THIRD source, which
  retires any remaining "one bad unit" reading.
- REJECTED REMEDIES, recorded so they are not re-proposed: the operator proposed
  an A/B/C local-LLM gender bakeoff over prompts/logic, plus possible web
  search. Neither addresses this defect. Both answer "how do we learn Ahab is
  male", and that question is not failing -- Melville's own text says "he"
  throughout, and the roster is already local. What fails is that the FIELD
  built to carry the fact was never written for 64 of 65 units. A better prompt
  cannot populate an absent sidecar, and web search would add a cloud dependency
  against the standing 100%-local rule to fetch a fact the source already
  states. The gap is plumbing, not knowledge.
- **TRIAGED AND CLOSED 2026-08-18** (stale-row sweep). `5194ab90` gave all 65/65 `public_domain` units a committed `.provenance.json` (was 16/65), including both units the entry names (`gertrude_governess`, `moby_dick_quarterdeck`). Live ledgers now carry a non-empty `gender_by_name`.
- previous status: OPEN -- diagnosed, three live instances across three sources. Fix is
- status: **CLOSED 2026-08-18 -- FIXED AND LIVE-PROVEN**
  the vendor-time stamper already specified in
  `docs/2026-08-05-character-gender-ladder-SPEC.md`, never built; it is chunk
  0.75 / the D4 vendor gate in the current sprint.

## PBUG-20260815-05 -- the episode title names a DIFFERENT play from the one it adapted

- artifact: `signal_lost_tempests_midnight_revelations_20260815_034337`
  (shakespeare). Operator: *"Is this Tempest or Macbeth? mixup??"*
- finding: the episode is Macbeth end to end -- cast `MACBETH` / `BANQUO`,
  `meta.source_ref "folger-macbeth:act1-scene3-witches"`,
  `meta.source_meta.play_title "Macbeth"`, music cue "Scottish moor, midnight
  sky". **Scene selection is CORRECT and is not the defect.**
- root cause: `meta.title_source == "llm_post_composition"`.
  `_generate_title_from_script` (`OTR_LedgerScriptWriter.py:1349-1526`, called
  `:6293-6305`) builds its prompt from dialogue excerpts, `outline.premise`, an
  empty `arc_verdict` and a generic bank label. **`source_meta`, `play_title`,
  `play_code` and `source_ref` are never passed in.** The model free-associated
  "Tempest" from the scene's genuine storm sound-world -- and "The Tempest" is a
  DIFFERENT play living in the same curated-scene manifest
  (`config/source_banks/shakespeare/curated_scenes.sample.json:27-49`).
- consequence: a FIDELITY defect on the lane where fidelity outranks arc.
- verify idea: the rendered title must not contain another configured work
  title. Regression fixture: a Macbeth scene must not title as Tempest. The
  check is CODE-side -- the craft rule forbids naming the feared failure in a
  prompt, so no forbidden example may enter the model's context.
- bible-worthy: yes -- "a naming pass blind to the identity it is naming, in a
  catalog where a sibling's name is a plausible free association".
- **FIXED 2026-08-19. The fix is an ANCHOR, not a guard, and the difference was
  an operator ruling made mid-window:** *"dont waste too much time
  overengineering for hard to replicate bugs im accepting some level of story
  quirks since a new story is gen every time"*. So the code-side "reject a
  title containing another configured work title" check specified in the
  `verify idea` above was deliberately NOT built. It had no sound matching
  rule -- substring containment rejects legitimate titles, and no better rule
  was available -- and building an unsound guard for a rare, hard-to-replicate
  quirk is the over-engineering the ruling names.
- **What landed.** `_generate_title_from_script` grew a keyword-only
  `work_title: str = ""`, threaded from the J.5 call site in
  `_run_writer_tail` and resolved through the EXISTING single bibliographic
  authority `_otr_source_identity.identity_from_meta` -- no second reader was
  grown. The pass is no longer blind: it is told what it is adapting.
- **THE ANCHOR IS NOT TITLE MATERIAL, AND THAT IS THE DESIGN.** Told only
  "this is Macbeth", a small local model answers "The Macbeth Prophecy" on
  every adaptation episode -- trading a rare fidelity defect for a constant
  blandness one, which THE LAW does not license either. So the anchor ships
  with the rule that keeps the name OUT of the title. No sibling title ever
  enters the model's context, so the craft rule cited in this entry is
  respected: the model is told what it IS adapting, never what it must not say.
- **The lane gate is applied, and it is the reason this is not a one-liner.**
  `work_title` holds the PUBLICATION on media_archive (56 of 98 live ledgers
  carry a `source_label` like "Now See Hear!"), so an ungated read would
  anchor a feed post's title pass to a magazine name -- inventing a work
  instead of naming one, which is worse than the defect being fixed. Gated on
  `ADAPTATION_SOURCE_KINDS`, never on truthiness.
- **QA CAUGHT TWO TAUTOLOGICAL TESTS (codex spark, on the finished diff).**
  `_run_writer_tail` contains a SECOND, pre-existing `identity_from_meta` read
  -- the announcer work-frame splice -- which is itself method-local, guarded
  and lane-gated. Measured on the real file: **4 matching imports and 2
  qualifying `try` blocks in that one method.** Two of the three call-site
  tests walked the whole method, so they passed on the OLD block and would
  have stayed green with this entire fix deleted. Now scoped to the block that
  binds `_title_identity`, with a fifth test guarding the scoping helper
  itself, and mutation-checked: removing the 1484-byte anchor block turns them
  red.
- **Live receipt:** the anchor block stamps `meta["title_work_anchor"]` on a
  successful read, ABSENT when the read raised -- the same present/absent
  convention `meta["bank_roll"]` already uses in this file, chosen so a frozen
  ledger can distinguish "the lane adapts nothing" from "the anchor failed"
  without a re-run. That ambiguity is the `voice_cast_decision == {}` trap
  that cost a whole arc to diagnose.
- **NOT PROVEN ON PIXELS.** Unit-proven only: 14 tests, full suite
  11092/110/1 (+12, exactly the tests added). The behaviour that matters is a
  model's output, so this owes a live shakespeare leg published to `otr/obs/`
  before it can be called closed on evidence rather than on construction.
- **PROVEN ON PIXELS 2026-08-19.** Bank gate `scripts/otr_writer_bank_gate.py
  --banks shakespeare,media_archive --acts 1`, profile `otr_w45_still_flat`:
  **2/2 banks PASS**, both published to `otr/obs/`.
  * **shakespeare -- `signal_lost_under_the_enchanted_moon_20260819_062006`**
    (9.4 min). Adapted `folger-midsummer:act3-scene1-bottom-titania`,
    `play_title "A Midsummer Night's Dream"`. Ledger:
    `title_work_anchor == "A Midsummer Night's Dream"` -- **the anchor reached
    the title pass on a live render**, which is the thing a green suite could
    not prove. Title: **"Under the Enchanted Moon"**. All three acceptance
    reads pass: it names NO other play; the anchor equals the play actually
    adapted; and it did NOT collapse into the play name. That last one is the
    check that matters -- it is the failure mode the anchor design was shaped
    to avoid, and it held on the first live episode.
  * **media_archive -- `signal_lost_reel_of_shadows_20260819_061004`**
    (10.9 min), the NEGATIVE control. `title_work_anchor == ""` with the KEY
    PRESENT: the read succeeded and the lane gate correctly refused to anchor
    the title to `source_label "Now See Hear!"`. Gated on truthiness instead
    of the lane, this episode's title pass would have been told it was
    adapting a magazine. The present-but-empty vs absent distinction also
    reads correctly off a frozen ledger, exactly as designed.
- status: **FIXED AND PROVEN ON PIXELS 2026-08-19.**

## PBUG-20260815-06 -- `media_archive` retells the newest feed post forever

- artifact: live RSS lane, verified in code and confirmed by the overnight gate
  artifacts.
- root cause: `nodes/_otr_media_archive_sources.py:221` --
  `return payloads[_configured_index() % len(payloads)]`, where
  `_configured_index()` (`:191-196`) reads `OTR_MEDIA_ARCHIVE_ITEM_INDEX` and
  defaults to `"0"`. Feed entries arrive newest-first, so absent an
  operator-set env var the lane always adapts the newest post. No dedup, no
  ranking, no history, no recording anywhere in the module.
- **that selection path has no test at all** -- verified by grep across
  `tests/`.
- prerequisite nobody had stated: the post's own headline is never stamped
  durably. `_otr_source_payload.py:585-621` builds `source_meta` as feed
  label / url / date only, and `news_seed_receipt` is never passed for this lane
  (`:698-701`), so `_stamp_news_seed_receipt` early-returns
  (`OTR_LedgerScriptWriter.py:1683-1685`). So "name the post it adapted" cannot
  be built from today's meta; the headline must be stamped at SELECTION time.
- verify idea: two consecutive runs with a stable feed must not select the same
  post; the selected headline must be present in durable meta.
- **SELECTION FIXED 2026-08-19 (`3be1c1e1`). THE DURABLE-HEADLINE HALF IS NOT
  DONE -- this row stays OPEN for it.** Be precise about which half: the lane no
  longer retells the newest post, and that is tested; "name the post it adapted"
  in the episode's own meta is still unbuilt.
- **What landed.** `fetch_media_archive_rss` now reuses the SCIENCE lane's
  existing history rather than growing a second one -- the operator's own
  framing: *"since scifi news seems to always choose a news story, should they
  have the same RSS logic generally?"* They should, and science already had the
  half this lane lacked. `story_orchestrator` keeps
  `<output>/otr/.../news_history.json` (article URLs, rolling cap,
  `_NEWS_HISTORY_FILTER_DAYS = 5` TTL so headlines recycle) via
  `_load_news_history()` / `_record_news_usage()`; it keys on URL and nothing in
  it is science-specific. Precedence: an explicit `OTR_MEDIA_ARCHIVE_ITEM_INDEX`
  wins, else prefer unused entries, else fall back to the full list rather than
  raising. Dedup is advisory end to end -- a failure selects exactly as before,
  because a feed lane must never fail a render over a JSON file. 14 tests where
  the PBUG recorded ZERO.
- **REVIEW CAUGHT A HOLE THAT WOULD HAVE BEEN INVISIBLE (codex spark).** The
  first cut took the override branch on a merely NON-EMPTY env var, and
  `_configured_index()` swallows a `ValueError` and returns `0` -- so
  `OTR_MEDIA_ARCHIVE_ITEM_INDEX=abc` would have taken the override, collapsed to
  index 0, and **silently restored the exact always-newest behaviour this fix
  exists to end**, with no error and no log line. `_explicit_index()` now
  separates "did the operator choose" from "which index should I use", warns on
  an unusable value, and lets dedup proceed. Five junk values are pinned. Same
  review also moved the override off the shared history: an explicit index is a
  deliberate repeat, and recording it would let a debugging run consume a
  headline for the automatic path and for the science lane's TTL window.
- **THE CROSS-LANE COUPLING IS SAFE, AND IT WAS MEASURED RATHER THAN ASSUMED.**
  Review flagged that a shared history means one lane can consume a URL out from
  under the other. The feed sets are disjoint by domain -- Library of Congress +
  filmpreservation.org against sciencedaily / eurekalert / nasa / nih / nsf /
  ucla -- so a URL from one can never appear in the other's feed. A test pins
  that and FAILS if the feed lists ever start to overlap, because that is the
  moment the shared history stops being free.
- **STILL OPEN, and the PBUG's own prerequisite:** the selected post's headline
  is not stamped in durable episode meta. `_otr_source_payload.py` builds
  `source_meta` as feed label / url / date only, and `news_seed_receipt` is never
  passed for this lane, so `_stamp_news_seed_receipt` early-returns. Recording
  the headline into the shared news history (which this fix does) is NOT the same
  thing -- that file is dedup state, not the episode's record. Until the headline
  is stamped at selection time, an episode still cannot name what it adapted.
- bible-worthy: probably -- "a live feed consumed by a constant index", a cheap
  and very portable check. A second, sharper candidate from the review: **an
  override branch gated on "is the value non-empty" rather than "does the value
  parse" silently reinstates the default it was meant to escape.**
- **DURABLE-HEADLINE HALF FIXED 2026-08-19. THE ROW IS NOW CLOSED, BOTH
  HALVES.** The fix is one line of production code, and the reason it sat open
  is worth more than the fix: **the CONSUMER was built first and the PRODUCER
  was never wired.** `_otr_source_identity.identity_from_meta` has always read
  `source_meta["post_headline"]` for this lane, and
  `SourceIdentity.is_degraded` returns True for media_archive *exactly* when
  that headline is missing -- while `_rss_source_fetch_result`
  (`nodes/_otr_source_payload.py`) built `source_meta` as kind / source_ref /
  source_url / source_label / source_date and never the headline. Measured at
  HEAD before the fix, on the exact meta shape the fetcher produces:
  `identity_from_meta(...).is_degraded` -> **True**. So every media_archive
  episode ever rendered has carried a degraded identity, silently.
- **The intent was even written down and still did not match the code.**
  `tests/test_source_identity_coda.py:148-159` builds its media_archive
  fixture with `post_headline` present and documents itself as a *"verified
  key set"*. The test encoded the shape the producer was supposed to emit; the
  producer never emitted it. A fixture is not a contract.
- **What landed:** `post_headline` stamped from the already-validated
  seven-key payload's `headline` key, at SELECTION time (the helper wraps the
  chosen item, not the widget request that asked for one), for BOTH RSS lanes
  rather than branching on `fetcher_kind` -- the phrase means the same thing on
  science and media_archive, so there is no one-field-two-meanings hazard and
  no reason to grow a per-lane branch in a shared helper. 10 tests.
- **THE FULL SUITE CAUGHT A REGRESSION THE REVIEW LANE MISSED.**
  `tests/test_source_payload_chunk3.py:421-440` asserts the science lane's
  `source_meta` by EXACT dict equality, so the new key turned it red. The QA
  lane had reported "no source_meta key whitelist/diff schema was found" --
  true as stated, and still wrong in effect, because the pin was an equality
  assertion rather than a schema. The expectation was updated rather than the
  assertion loosened to a subset check: an exact pin that catches a contract
  change is doing its job. **This is why the suite runs even when a review
  says ship.**
- **What the QA lane DID get right** was that every test stopped at the
  helper. Producing the field is not the same as it ARRIVING: the writer
  copies `source_meta` wholesale into durable `meta["source_meta"]` but POPS
  `_news_seed_receipt` out of that same dict as transient, so a field sharing
  that fate would look perfect in every producer test and be absent from every
  ledger. Three carry tests now walk fetcher -> `normalize_fetch_result` ->
  receipt-pop -> `identity_from_meta`, with nothing hand-built.
- **PROVEN ON PIXELS 2026-08-19.** Same bank-gate run.
  `signal_lost_reel_of_shadows_20260819_061004` (media_archive, PASS,
  published to `otr/obs/`) carries in durable ledger meta:
  `source_meta.post_headline == "This Thursday (7:00 PM August 20) at the Mary
  Pickford Theater (Washington, DC)"`. **An episode can now name the post it
  adapted, from its own frozen ledger, with no re-run and no dedup file.**
  That is this row's stated prerequisite, discharged.
- status: **CLOSED AND PROVEN ON PIXELS 2026-08-19 -- both halves.**
- status: OPEN -- diagnosed, fix specified, not yet landed.

## PBUG-20260815-07 -- the `original`-lane voice-gender report, INVESTIGATED AND NOT REPRODUCED

- artifact: `signal_lost_kindling_the_past_20260815_043118` (bank `original`).
  Reported alongside PBUG-20260815-04 as a second voice-gender instance:
  JULIANA SIMPSON speaking with a male voice.
- measured: JULIANA renders FEMALE on all six of her lines (229.7 / 220.7 /
  300.0 / 222.2 / 248.1 / 212.9 Hz), and every recorded field agrees -- ledger
  `gender` and `presentation_gender` both `female`, Bark preset
  `v2/en_speaker_4` classified female in `config/cast_pools.py:263`, and the
  indextts2 reference `vz_donor_glenn` measured at 313.9 Hz.
- a routing hypothesis was raised in review -- that pooled F0 could hide a
  speaker-to-slice routing fault, so the wrong audio might have been measured.
  Investigated PER LINE: windows are contiguous and non-overlapping and every
  one of Juliana's six lines measures 212-300 Hz. **Refuted.**
- recorded so nobody re-chases it: the invented lanes roll gender freely and
  that is correct design; `_repair_ensemble_names` already repairs name/gender
  incoherence there (probe: `JULIANA SIMPSON` rolled male repairs to
  `DANIEL SIMPSON`).
- status: REJECTED -- not a defect on this artifact. See PBUG-20260815-08 for
  the real defect found in the same episode.

## PBUG-20260815-08 -- a character record carries a DIFFERENT character's name in its description and speech signature

- artifact: `signal_lost_kindling_the_past_20260815_043118` (bank `original`),
  cast row `c02`.
- symptom: `name` is `"TARIQ SCOTT"`, but `character_description` reads
  *"40s, HENRY BARTEL, Seasoned Craftsman..."* and `speech_signature` reads
  *"Henry's sentences often begin with a pause..."*. A stale, different name is
  baked into the record that the description LLM, the portrait prompt and the
  dialogue cast block all read.
- found while investigating PBUG-20260815-07; the sibling row `c03` shows no
  such artifact, so this is a per-row desync rather than a lane-wide fault.
- consequence: identity drift across surfaces. `slot.gender` and the character
  name feed the description LLM, the outline prompt, the dialogue cast block
  (`_otr_line_composer.py:447,469`) and the image prompt's gender anchor
  (`otr_meta_brief_image_prompt.py:78-90`), so a record naming two different
  people can move both script and portrait.
- verify idea: a character-record invariant -- canonical name, description
  subject, speech signature, caption identity and voice mapping must all agree
  before rendering.
- bible-worthy: yes -- "a renamed entity whose derived text kept the old name".
- status: OPEN -- diagnosed, not yet scoped into a chunk.

## PBUG-20260815-09 -- a publication receipt stamped before the episode rename blocked EVERY episode from obs

- artifacts: `signal_lost_the_price_of_the_floorboards_20260815_120514`
  (public_domain), `signal_lost_a_tomb_of_secrets_20260815_122546`
  (shakespeare), `signal_lost_the_weight_of_the_brass_key_20260815_125005`
  (media_archive). All three RESULT SUCCESS, all three archival finals on disk,
  **all three WITHHELD from obs**.
- WORDING CORRECTED 2026-08-15: an earlier cut of this entry, and the handoff
  that quoted it, said `otr\obs\` was "EMPTY". That is false and it sends the
  next reader hunting a vanished directory. The real obs base --
  `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\`, NOT the repo-relative
  `otr\` -- held 150 files at the time, seven of them from earlier the same
  morning. What happened is narrower: these three legs published nothing while
  every prior leg had published fine. Note the path trap itself, because it
  fakes this bug's symptom: checking the repo-relative `otr\obs\` reports a
  false EMPTY whether or not publication works.
- symptom, from the server log:
  `obs_publish BLOCKED (eligibility_receipt_episode_mismatch: receipt is
  stamped for 'pending_20260815_115325'; this episode is
  'signal_lost_the_price_of_the_floorboards_20260815_120514')`
- cause: the freeze (Phase 10) stamps `meta.publication_eligibility` with the
  episode id while the episode is still `pending_<ts>`; `Ledger.rename_episode`
  then gives it its real slug. `OTRMasterAudioMux` compared the receipt's
  stamped id against the LIVE ledger's id, found them different, and read that
  as the stale-in-flight-singleton case it fails closed on. Every episode is
  renamed, so this withheld every episode -- a self-inflicted regression from
  chunk 0.5, shipped the same day.
- why the suite was green over it: every unit test pins a fixed episode id.
  Nothing but a live leg renames, so no fixture could reproduce it.
- fix (`92981bc4`): `rename_episode` already rebases every episode-local
  durable pointer onto the new identity; the receipt was simply not on its
  list. `_rebase_publication_eligibility` moves the name only -- the verdict
  and its reasons are untouched, so rights stay owned by the one producer at
  the freeze. Three regression tests.
- consequence if unfixed: no episode ever reaches the operator's watch folder,
  while every log line says RESULT SUCCESS and the archival final exists. The
  failure is invisible to `RESULT SUCCESS` and visible only by looking in
  `otr\obs\` -- which is exactly the operator's standing law about confirming
  the asset on disk.
- bible: shape is COVERED by `12.66` (rename-stale episode identity captured
  before `rename_episode` and used afterwards). This instance extends it from
  path RESOLUTION to an authorization GATE: the data was correct, only the name
  was stale, and the consequence was a withheld deliverable rather than a
  mis-resolved path. Index row appended pointing at `12.66`.
- status: **FIXED and LIVE-PROVEN 2026-08-15.** Leg
  `signal_lost_the_price_of_a_soul_20260815_132024` (public_domain, Moby-Dick)
  logged `obs_publish OK ->
  ...\output\otr\obs\signal_lost_the_price_of_a_soul_20260815_132024_silent_procgen_blended_captioned_with_credits_final.mp4`
  with the file present on disk at **165.2 MB**, `Prompt executed in 00:23:18`.
  The proof is only a proof because that episode RENAMED -- it entered the
  freeze as `pending_<ts>` and published under its final slug, which is the
  exact condition that produced the block. CLOSED; do not re-prove.

## PBUG-20260815-10 -- the scene-review pass asks for more tokens than the whole context window

- artifact: live `scifi_news` leg through `workflows/otr_canonical.json`,
  2026-08-15, `tmp/d2_server.log`. Died at `Prompt executed in 00:10:18` with
  `CodexPassError: P5R failed: prompt cannot fit the complete requested output:
  prompt requires 1203 input tokens, requested_output=8320, context_cap=8192
  leaves only 6989 output tokens`. Writer was `gemma-4-12b-it` (the saved
  runtime-qualified default, `context_window=8192`).
- root cause: `nodes/_otr_scifi_codex.py` `_SCENE_REVIEW_MAX_OUTPUT_TOKENS` was
  a CONSTANT derived from the SCHEMA maximum --
  `_RADIO_SCORE_MAX_BEATS_PER_SCENE` (8) x `_RADIO_SCORE_MAX_LINES_PER_BEAT` (2)
  x `_BEAT_TEXT_TOKENS_PER_LINE` (512) + envelope (128) = **8320** -- and 8320
  is larger than the ENTIRE 8192-token context window. The request is therefore
  unsatisfiable for ANY prompt, including an empty one; the 1203-token prompt in
  the log is incidental. `_RADIO_SCORE_MAX_BEATS_PER_SCENE` carries a deliberate
  `_SCHEMA_HEADROOM` of 2 so the SCHEMA can accept more than a legal episode
  contains -- correct for a guard, wrong for a request, because it asks for
  exactly twice the output any real scene needs.
- why it was invisible: two tests asserted the constant --
  `tests/test_codex_per_beat_dialogue.py:335` and
  `tests/test_scifi_codex_lane.py:531` -- so the suite AGREED with the bug. A
  test that pins the number a caller passes, rather than the property that
  number must satisfy, cannot fail when the number is impossible.
- scope, measured not assumed: every `max_new_tokens` budget in both story
  lanes was computed against the 8192 cap. `_SCENE_REVIEW_MAX_OUTPUT_TOKENS`
  8320 was the ONLY impossible one; `_BEAT_TEXT_MAX_OUTPUT_TOKENS` 1152,
  `fable2._MAX_NEW_TOKENS['dossier']` 700 and `_NEWS_CODA_MAX_OUTPUT_TOKENS` 384
  all leave >= 7,000 tokens for the prompt. One instance, not a class outbreak.
- fix: replaced the constant with `scene_review_output_tokens(line_count)` --
  the request is sized to the scene ACTUALLY in hand
  (`len(scene_line_ids) * 512 + 128`; a real 4-beat scene is 8 rows -> 4224,
  leaving ~4,000 for the prompt). The schema's `max_length` guard is untouched:
  right-size the job, never raise the guard. No clamp against the context on
  purpose -- if even the right-sized job does not fit, `prompt_must_fit=True`
  refuses deterministically and says why, whereas a silent truncation would
  return a review missing rows and fail the closed-row validator with a
  misleading message. Telemetry `scene_review_max_new_tokens` now records the
  LARGEST request the run actually made, because a receipt naming a number no
  call used is what let 8320 sit in every journal unread.
- coverage: both tests above now assert the property (`== scene_review_output_
  tokens(len(scene_line_ids))` AND `< the smallest shipped context window`)
  instead of the number.
- bible-worthy: yes -- "a generation request derived from a SCHEMA ceiling
  rather than the job in hand can exceed the model's entire context, and a test
  that pins the constant agrees with it".
- SECOND DEFECT, SAME BUG, found by the Fable gate before it shipped: sizing
  the request to the actual scene cured the CONSTANT and left the death
  reachable, because SCENE SIZE IS THE MODEL'S CHOICE. The schema accepts up to
  `_RADIO_SCORE_MAX_BEATS_PER_SCENE` (8) beats x 2 lines = 16 rows and the P3
  prompt at `:786` literally instructs "at most 8 beats per scene", so a legal,
  actively invited draft reproduces 8320. It bites from 7 beats up: 7296 plus
  the measured ~1203-token prompt already exceeds 8192. The review now CHUNKS at
  `_SCENE_REVIEW_MAX_ROWS_PER_CALL` = `BEATS_PER_ACT` x 2 = 8 rows, derived from
  the real topology and never from the schema ceiling. Every call still carries
  the whole scene's rows and spine, so only the RETURN set narrows; a real
  4-beat scene is 8 rows and takes ONE call, so production is byte-identical.
  Not fixed by narrowing the prompt's advertised ceiling: schema headroom exists
  because models overshoot what they are told, so an advertised limit is
  guidance where this must be a guarantee.
- status: **FIXED and LIVE-PROVEN 2026-08-15.** Leg
  `signal_lost_the_architecture_of_error_20260815_152004` (scifi_news, through
  the real `workflows/otr_canonical.json`) cleared P5R -- a 100% deterministic
  death before -- and ran to `RESULT SUCCESS` with `obs_publish OK` and a
  **33.4 MB** file on disk, `Prompt executed in 00:40:56`. CLOSED.

## PBUG-20260815-11 -- 34 characters sound one gender and look the other

- artifact: `scripts/audit_voice_gender_consistency.py --root <output>/otr/episodes`,
  run 2026-08-15 over **1,686 ledgers** (65 policy-era, 1,621 legacy).
  `VIOLATIONS: 0` -- every ACTIVE voice field agrees with the assigned gender,
  so what the audience HEARS is right. But
  `portrait conflict : 34 row(s) whose prose asserts the opposite gender`.
  Examples: `SHERLOCK HIBBERT` assigned male, description says "her";
  `FATHER BROWN` assigned female, description says "father"; `RICK STEINER`
  assigned male, description says "mother".
- why it matters: the description IS the portrait prompt -- the writer builds
  `visual_plan.characters[].portrait_prompt` from `character_description`. Those
  characters are voiced as one gender and drawn as the other. Operator, same
  day: *"Jane should always look and sound the same gender."* The repo already
  carries the rule in the auditor's own words -- *"CONSISTENCY BEATS ACCURACY:
  a female Scrooge with a female voice and a female portrait is coherent but
  unfaithful; a male Scrooge with a female voice is broken."*
- root cause: the description prompt TELLS the model `Gender: male`
  (`_otr_casting.py` -- "Gender / timbre / role are Python-decided facts the LLM
  writes into"), and the only validation on what comes back is
  `_strip_desc -> v.strip()`. It ASKS and never VERIFIES -- the same pattern the
  40/40/20 ensemble balance was moved out of the prompt to escape.
- **NOT an inflated count.** The detector already excludes possessives ("his
  mother" belongs to someone else) and possessive nouns ("widow's peak" is a
  hairline); its own comment records that without the second rule it reported
  170 rows. A DISGUISE plot is a legitimate hit -- ROSALIND-as-Ganymede and
  VIOLA-as-Cesario keep female voices by operator ruling -- so 34 is a list to
  read, not a total to fix. `FATHER BROWN` and `SHERLOCK HIBBERT` are not
  disguises.
- **BLOCKED, and this is the important part: the obvious fix is FORBIDDEN by a
  standing ruling.** `otr_meta_brief_image_prompt.py` -- anchored by the SENTENCE, not a line number, because the number has already gone stale once: *"No Python vocabulary or overlap classifier can reject, rewrite, or block the prompt."* (at :1706-1707 as of 2026-08-15; :1585 is the music-mesh comment and never held this) is a live design
  contract -- *"No Python vocabulary or overlap classifier can reject, rewrite,
  or block the prompt."* The 2026-08-05 item-8 campaign proposed exactly this
  fix twice (Codex: reject contradictory candidates inside the bounded loop;
  agy: fall back to the deterministic template) and BOTH were overruled at r4,
  with "update the stale comment" explicitly rejected as a resolution. Two
  further constraints from that campaign block the cheap route: node 89 cannot
  reach the description generator, and `Ledger.set_cast` rebuilds a fixed
  nine-key row that silently drops new fields.
- the three live options, for the operator: (1) leave it and read the audit
  list; (2) narrow the ruling for this case only -- a contradiction between two
  PYTHON-OWNED facts is arguably not a vocabulary classifier judging prose, and
  that is the only ground it could be revisited on; (3) strengthen the
  description PROMPT rather than checking its output, which rejects nothing and
  is permitted as the ruling stands. Option 3 is the only one buildable without
  reversing a four-round decision.
- bible-worthy: not yet. The lesson ("a constraint stated in a prompt and never
  verified is not enforced") has no fix to generalise from until the ruling
  question is settled.
- status: **OPEN, BLOCKED ON THE OPERATOR.** Measured and reported by an
  existing audit; the fix requires a ruling change.

## PBUG-20260811-03 EXTENDED -- it is BOTH content-owned lanes, confirmed on 2026-08-15 artifacts

- The existing entry records `scifi_news` losing the LEMMY cameo and shipping an
  empty `cast_contract`. Re-checked against the two legs rendered 2026-08-15 on
  HEAD `50790099`, reading the frozen ledgers directly:

      signal_lost_the_architecture_of_error_20260815_152004  (scifi_news)
          cast_contract: {}   lemmy in cast: False
      signal_lost_blood_red_water_20260815_195226            (scifi_news_pro)
          cast_contract: {}   lemmy in cast: False

  **`scifi_news_pro` has the same defect and the log did not say so.** Both
  content-owned lanes build their own cast and never reach `lock_cast()`, which
  is what applies the cameo -- so the exposure is twice what was recorded, and
  it is still silent: nothing fails and nothing logs.
- The established root cause and the warning both stand: routing content-owned
  lanes back through `lock_cast()` is THE WRONG FIX -- that block deliberately
  withholds `cast_seed` because claiming one on a lane-owned cast detonated
  CastLock's replay (`num_characters must be 1-6, got 0`). The repair belongs in
  each lane runner, and it is two things, not one: the cameo roll AND the cast
  contract.
- **Sibling risk, do not fix these in isolation:** PBUG-20260811-01 says forcing
  the cameo KILLS the `scifi_fable2` writer on `scifi_news_pro` (markup ladder
  exhausted, BAD_LINE, reproduced at 30 and 90 target words). A cameo fix on that
  lane must be proved against that failure, not just against an empty contract.
- **Observed alongside, worth its own look:** the cast names on those two legs
  were `Dr. Aris Thorne / Elias Vance / Unit 7` and `Elias / Sarah`. Two
  different lanes, two different stories, both produced an "Elias". These lanes
  do not use the 154-name pool -- they name through their own model call with no
  diversity mechanism -- which is the operator's long-standing "it always picks
  the same name" complaint, seen live for the first time. Two samples is not a
  measurement; `scripts/otr_name_randomness_lab.py` was built this session to
  measure it properly and has not been run.
- status: OPEN. Evidence refreshed, scope corrected from one lane to two.

**TWO CORRECTIONS, 2026-08-16** (verified against the frozen ledgers and the
git record; the census that grounds them is at the end of this file):

1. **The key is ABSENT, not `{}`.** Direct reads of both 2026-08-15 frozen
   ledgers show NO `cast_contract` key at all (`meta.get("cast_contract")`
   returns None); the `{}` in the artifact block above was the probe's own
   `or {}` fallback rendering. Sharper defect, same repair: the key is omitted
   entirely, which is exactly the "field never written" shape the invention
   lanes' stable-shape rule exists to prevent.
2. **The LANE never had the cameo; the PRODUCT did.** The base entry's sentence
   "scifi_news predates the content-owned redesign, worked under the legacy
   picker" is unsupported: today's `scifi_news` was born `scifi_codex_v4`
   (`1fd7743d`, 2026-07-17), runner-dispatched from birth (`c22eef0a` wired the
   runner into `_RUNNER_BY_PIPELINE` the day the module was created), and a
   case-insensitive pickaxe over BOTH runners' full history returns EMPTY -- no
   commit ever added or removed the string "lemmy". The lane that ran the cameo
   was `science_news` (`legacy_many_pass` -> inline `lock_cast()` -> LEMMY
   11%), retired at `499386aa` on 2026-07-17; two days later `f03128fa` renamed
   scifi_codex_v4 -> scifi_news, landing the name on a lane that never had him.
   The PRODUCT-level loss is real and measured -- `science_news` cast Lemmy
   **14** times (all speaking) through 2026-07-15 -- so the operator's "it
   always used to work, it was the first Lemmy plan" is CONSISTENT with the
   record at product level; the science_news / scifi_news name collision is the
   likely source of the "regression" framing. The repair is unchanged: both
   content-owned runners need the contract and the roll built NEW.

### Operator context, 2026-08-15 -- NOT derivable from the code, recorded so it is not lost

**MEASURE THE PART, NOT THE PRESENCE. The presence question is already
answered.** Lemmy is cast **190** times and SPEAKS in **188**; the only two
silent castings are from June and pre-date the current cameo code. His identity
is stable in every single one -- `char_id` **c02**, male, `v2/en_speaker_8`, the
fixed Cockney `speech_signature`. **The variable is the SIZE and fidelity of the
part**, not whether he turns up.

**The first GOOD Lemmy was seen 2026-08-15 and the operator flagged it himself.**
Two `media_archive` episodes that day -- 01:34 and 04:23 -- gave him **6 speaking
lines each**, against a **3.2** average for that bank and a **1.5-5.0** range
across all banks. The biggest parts he has had in 190 castings. Read the exemplar
before touching anything:
`signal_lost_reel_of_mystery_20260815_041350_ledger.json`.

**That is a DIFFERENT, LATER, QUALITY-SIDE item and is NOT in this sprint.** The
sprint is the STRUCTURAL defect below: the `scifi_*` family never casts him at
all and always ships an empty `cast_contract`. Fix that first.

**DETECTOR WARNING -- this will burn a window if it is not read.** Ledger LINES
identify the speaker by `char_id`, **not by name**. Lemmy is always `c02`.
Matching the string `"LEMMY"` in a line's speaker field returns a near-total
FALSE NEGATIVE -- it reports him silent in **188 of 190**. The operator made
exactly that mistake on 2026-08-15 and caught it. **Resolve his `char_id` from
the CAST row first, then match lines on that id.** (Cast ROWS do carry the name,
so name-matching is correct there and wrong one level down -- which is precisely
why the trap is easy to fall into.)

**Scope note, recorded because two counts are in play.** The operator described
the affected family as ~14 lane variants. The shipped registry at HEAD
`d96dd8e2` shows **2** scifi banks (`scifi_news`, `scifi_news_pro`), **2** scifi
pipelines (`scifi_news_circuit`, `scifi_news_pro_multipass`) and therefore **2
repair sites** -- `_otr_scifi_codex.run_scifi_codex_episode` and
`_otr_scifi_fable2.run_scifi_fable2_episode`. Both figures are recorded rather
than one overwriting the other: **the fix lands in 2 runner modules regardless of
how many variants route through them**, but if there really are ~14 the
reconciliation is worth doing before the sweep claims coverage.

**CENSUS, 2026-08-16 -- all 1,686 ledgers, cast-row NAME match with spoken
lines resolved via char_id per the detector warning above.** 190 castings, 186
speaking by strict non-empty-text match (the 08-15 count of 188 used its own
method; both agree every silent casting predates the current cameo code -- all
4 sit in the pre-bank era, before 2026-07-05). Current era: `media_archive`
**9** (last 2026-08-15) and `original` **4** (last 2026-08-10) ONLY.
July-and-earlier: `science_news` 14 (last 07-15), `shakespeare` 6 (07-30),
`public_domain` 1 (07-23), scattered `_v2`/`_v3` singles, and 150 castings in
the 942 pre-bank ledgers -- NONE of which carry either content-owned lane
namespace (`meta["scifi_codex"]` / `meta["fable2"]`), so no surviving artifact
shows the codex/fable2 family ever casting Lemmy under any name. Every
`scifi_*` bank id: zero castings, ever. A six-bank sweep finding Lemmy only on
media_archive / original and absent on the scifi pair is therefore the
EXPECTED result, not a regression signal.

---

## PBUG-20260816-01 -- `scifi_news` RETIRED (rip), and what closed with it

- surfaced: not a failure report -- an OPERATOR DECISION on measured evidence,
  logged here because the teardown protocol's ledger-discipline item requires a
  rip to leave a causal record rather than a silent absence.
- evidence: the blind per-bank narrative read
  (`docs/2026-08-16-blind-bank-narrative-ranking.md`) scored `scifi_news` LAST
  of six at **2.0/10** (worst sample 1/10, "it is not a story"; one leg
  broadcast pipeline metadata as dialogue -- *"final coda, factual report
  backed by P0 facts F01-F06"*), while `scifi_news_pro` scored FIRST at
  **7.5/10**. The deterministic structure score ranked the same two banks last
  (67.9 median) and first (93.8 median) independently. The reader re-paired all
  twelve blind transcripts into their six banks 6/6 correct, so the ranking is
  signal.
- operator ruling 2026-08-16: *"we ditch scifi_news and the pro becomes the
  standard... we leave the name pro since it won."*
- fix: **retired the runnable bank + its pipeline/route.** Depth FULL-FAMILY:
  `scifi_news` was the only bank on `scifi_news_circuit`, whose runner was
  `nodes/_otr_scifi_codex.py` (4,664 lines), so the module, its dedicated
  `_otr_scifi_source_repair.py` helper, the pack dir, the `LANE_SPECS` entry,
  both registry rows and 13 dedicated lane tests went with it. Writer defaults
  re-point to `scifi_news_pro`. Plan + full surface enumeration:
  `docs/2026-08-16-scifi-news-RIP-PLAN.md`.
- **LEDGER GATE PASSED, stated explicitly because it is the one that could have
  blocked this:** every field the lane stamped was inside the
  `meta["scifi_codex"]` namespace with ZERO surviving production readers, or a
  shared key whose writer survives. Computed f-string keys were swept -- the
  bank id was never interpolated into a meta key, only used as a value. No hole.
- **CLOSED-BY-RIP** (these were OPEN against the retired lane and are not
  outstanding work any more): the codex `P5R _call_scene_review` no-shims
  violation; `_canonicalize_script_spoken_text` writing stripped text back into
  the record; the graduated-extraction span-reader enumeration
  (`docs/2026-08-15-graduated-extraction-span-reader-enumeration.md`, whose own
  scope line read "this work is `scifi_news` only"); the creativity-knob
  no-op on that lane; and the `scifi_news` P0 convergence blocker.
- **COVERAGE LAPSED, recorded honestly rather than quietly:** `_CodexTailFinalizer`
  was the ONLY implementer of both the writer `TailFinalizer` protocol and the
  clean transaction's three-method proof protocol, so
  `tests/test_clean_transaction.py::TestFinalizerProtocol` ("a renamed method
  would go unnoticed") is gone. The transaction machinery is still covered by
  stub-driven tests and the writer-tail test. The protocol SURFACE is
  deliberately KEPT as extension space -- restore that class when a lane
  implements it. Same shape: four codex bark-allocator tests in
  `test_gender_normalization.py`; the surviving lane reads
  `cast_pools.open_voice_pool` directly, so the defect they guarded is
  unreachable by construction there.
- reversal: `SOURCE_BANK_PREFLIGHT.md` "Reversal" section. Pre-rip anchor tag
  `otr-2026-08-16-sixbank-lemmyA` at `da44f642`, where all six banks were
  live-proven; the rip is ONE atomic commit, so `git revert` restores it whole.
  A restored bank re-enters at gate 1 and must re-qualify on a live leg.
- status: **DONE.**

---

## PBUG-20260816-02 -- scifi_news_pro script pass invents speakers outside the cast (ladder exhaustion, 4th live sighting of the class)

- surfaced: GPU soak leg 1, 2026-08-16 15:21 (`SOAK01 scifi_news_pro
  sci_fi_radio still_motion_flux2_klein`, 1 act, canonical graph, renamed
  module `_otr_scifi_news_pro` at HEAD `96240ce1`-era boot). Died 27.9 min
  in: `NewsProScriptError: pass 'script' failed after 4 attempt(s): markup
  ladder exhausted`.
- symptom: EVERY last defect is `UNKNOWN_SPEAKER` -- the model wrote dialogue
  for `DR. LIAO` and `PROF. ZHANG` (7 lines), speakers that are not in the
  treatment's declared cast. The ladder's repair rule names the allowed
  speakers; the model kept its invented ones through all four attempts.
- class: the SAME stochastic markup non-compliance that killed the 08-10
  sweep leg and the 08-11 probe B (`UNKNOWN_SPEAKER: LUCY`, `REPORTER`) --
  the class the PBUG-20260811-01 correction identified after withdrawing the
  cameo attribution. This is its 4th live sighting and its 1st since the D3
  END-grammar fix, which proves D3 fixed a DIFFERENT member of the family.
- NOT the rename: the rename proof leg passed this exact lane cleanly an
  hour earlier (`signal_lost_rename_proof_scifi_news_pro_20260816_141310`),
  and the failure is model-stochastic (Mistral-class writer inventing cast).
- **measurement now RUNNING, and that is the point:** the 08-16 correction
  to PBUG-20260811-01 flagged "the lane's baseline ladder-exhaustion rate is
  unmeasured" as the open question. The GPU soak
  (`scripts/otr_gpu_soak_matrix.py`, receipts in `otr_soak_receipts/`) is
  now collecting exactly that rate across all five banks. DO NOT theorize a
  fix from n=1; let the soak deliver an incidence rate first.
- fix: NONE YET, deliberately. Candidate direction if the rate warrants it
  (for the panel, not for a solo swing): the ladder's repair note already
  states the allowed speakers (the D3 pattern); the next rung up is a
  DETERMINISTIC speaker-map repair -- an unknown speaker whose line count
  and position match a declared cast member gets mapped, not rerolled --
  which is roster repair (sanctioned fail-closed territory), never prose
  rewriting. Panel it against THE LAW before building.
- status: OPEN, measuring. **CORRECTION 2026-08-16 evening: the measurement
  this entry promises is NOT being collected by the harness that looks busiest
  -- see PBUG-20260816-03.** The real incidence data is the 8 legs in
  `soak_20260816_143704.json` + `soak_20260816_145333.json`, not the 708 legs
  in `soak_20260816_143448.json`, which never rendered anything.

## PBUG-20260816-03 -- a GPU soak harness ran 708 legs and rendered nothing, because it patches MANAGED widgets with `--set`

- surfaced: 2026-08-16 evening, reading `otr_soak_receipts/` for a handoff.
  Three soak harnesses were live (seeds 816 / 1436 / 1451). Their receipts:

  | receipt | legs | ok | longest leg |
  |---|---|---|---|
  | `soak_20260816_143448.json` | **708** | **0 true / 708 false** | 0.5 min |
  | `soak_20260816_143704.json` | 5 | 4 true | 42.3 min |
  | `soak_20260816_145333.json` | 3 | 2 true | 42.8 min |

- symptom: every leg in the big receipt has `rc: 1` and `minutes` between 0.20
  and 0.50 -- roughly twelve seconds, far too fast to be a render. Sustained
  for about two and a half hours. The server was NOT down: `:8000` was
  listening and `/queue` answered `HTTP 200` throughout.
- root cause, REPRODUCED directly rather than inferred -- one leg re-run by
  hand gives the exact refusal:

      ValueError: patch_creative: widget 'character_video_model' is not on the
      creative whitelist; managed widgets are patched ONLY via
      apply_profile_to_workflow(--profile).

  That harness rotates the video/image engines with
  `--set OTR_VideoDirector.character_video_model=...` (and the announcer/music
  siblings). Those are MANAGED widgets, so `patch_creative`
  (`scripts/otr_api.py:868`) refuses them while the API prompt is still being
  built -- before submission, before the GPU is touched.
- the guardrail is RIGHT; the caller is wrong. `GO_FORWARD_PLAN.md` already
  records the sanctioned lever ("the managed-widget guardrail refused `--set`
  by design, so profiles are the sanctioned lever"), and the two harnesses that
  use `--profile otr_soak_*` are rendering real 42-minute legs. One of the
  three launches simply used the wrong instrument.
- **why this is worse than wasted CPU:** the operator's tag gate is "ready once
  the soak receipt shows every bank passing", and a reader opening
  `otr_soak_receipts/` sees a 103 KB file reporting 708 legs. Read quickly that
  looks like broad coverage. It is 708 rejections, and it also inflates the
  apparent leg count roughly 90x over the real progress (8 legs, 6 passes).
- related, same guardrail, same day: the Lemmy cross-engine r3 panel caught the
  driver planning to move the canonical engine widgets with `--set` for an
  acceptance leg. It cannot be done that way by anyone -- the instrument is a
  capability profile whose `slot_overrides` reach both nodes through
  `config/profiles/widget_mapping.json`.
- fix: NONE APPLIED. The failing harness was left RUNNING deliberately -- it is
  operator-ordered state, it holds no GPU, and killing it was not needed for
  any work this window. What it needs is either the `--profile` lever like its
  two siblings, or deletion of the `--set` engine-rotation path in
  `scripts/otr_gpu_soak_matrix.py` so the mistake is unrepresentable.
- status: OPEN. Do not read `soak_20260816_143448.json` as coverage.

## PBUG-20260816-04 -- the mirror generator's "idempotent by ownership" fix was row-level, and it was still destroying FIELDS

- surfaced: 2026-08-16 late, running `scripts/_otr_mirror_clone_refs.py` against
  the real bank to mint the two Lemmy clone rows the provisional tier needs.
  Live artifact: the actual `git diff` of the write, not a review reading.
- the receipt LOOKED right and that is the whole lesson. The run printed
  `mirrored=83 added=2 preserved-unmanaged=3` -- the exact line the 2026-08-16
  r2 judgment cited as proof the generator had been made safe. That line counts
  ROWS. The diff showed 37 insertions and **18 deletions** for a change that
  should have been pure addition.
- two distinct losses, both invisible to the counter:
  1. **`speaker_id` stripped from all eight mirrored rows that carried one.**
     The generator copied a fixed seven-field allow-list (`_COMMON`) that was
     written before the bank gained the field. `speaker_id` records the real
     HUMAN behind a reference and exists because a ref_path collision cannot
     catch two recordings of one person -- LibriVox's Mark F. Smith has a plain
     and a grandfatherly take in two different files. Without it one narrator
     can be cast as two characters in the same episode, on chatterbox and dia
     only, which is a casting defect nobody would trace back to a bank script.
  2. **`cb_announcer_male` reverted.** The on-disk row had been re-pointed at
     `vz_peter_yearsley` with curated timbre (`british`, `clear`, elder) and
     `style_tags: [preferred_announcer, british_leaning]`. The generator owns
     that key and rewrote it back to its hardcoded `vz_bill_boerst` literal,
     dropping the style tags entirely.
- root cause: "ownership" was implemented as *replace the whole row for a key I
  generate*. A generator owns the fields it DERIVES; it does not own fields it
  has never heard of, and it does not own a row it merely bootstrapped.
- fix (applied): the mirror now copies every source field except the identity
  pair (`voice_ref_id`, `engine`), so any future bank field mirrors without
  anyone remembering to edit a list; owned keys MERGE over what is on disk
  rather than replacing it; and the `cb_announcer_male` bootstrap row is created
  only when it is ABSENT. Re-run diff is now **32 insertions, 0 deletions** --
  exactly the two Lemmy rows. Counts also became honest: `mirrored=82
  added=2 preserved-unmanaged=4`.
- coverage: `tests/test_tts_voice_preflight_matrix.py::test_p3_4_the_generator_
  never_drops_a_FIELD_from_a_row_it_owns` (every prior key must survive) and
  `::test_p3_5_a_mirror_carries_its_sources_speaker_id` (the semantic pin). The
  three existing P3 tests all passed against the defective generator, because
  every one of them asked about rows.
- status: FIXED, proven on the real bank write.

## PBUG-20260817-01 -- the image engine's default NEGATIVE vetoes the visual style the episode selected

- surfaced: 2026-08-17, from a published episode the operator watched --
  `signal_lost_kinetic_motion_clause_live_test_20260817_050130`, `visual_style`
  = `cartoon` (confirmed in its own credits roll). Live artifact, not a review.
- symptom, measured on frames pulled from the delivered mp4: 00:04 bookend
  CARTOON, 00:16 announcer CARTOON, 00:34 character PHOTOREAL, 00:43 announcer
  painterly semi-real. One 74-second episode cutting between an animated short,
  a live-action film and a painting. Operator: "some char beats were cartoony
  and others not."
- root cause, VERIFIED at the file: `nodes/_otr_image_engines/z_image_turbo.py:216-219`
  ships a default negative of
  `"oversaturated, glossy, clean digital, plastic skin, waxy skin, sterile
  studio lighting, cartoon, illustration, text, watermark"`.
  On a CARTOON episode every still is minted with positive "bright cartoon
  illustration" AND negative "cartoon, illustration". The engine suppresses the
  style the config selected. The negative is a house-style constant tuned for the
  DEFAULT pack (`sci_fi_radio`, a filmic look) and is completely style-blind.
  **It fights four of the nine packs** -- `cartoon`, `anime`,
  `storybook_engraving`, `paper_origami` -- on EVERY mint.
- why it looked like a gradient rather than a switch: where the positive body was
  neutral or cartoon-worded, cartoon still won (hence the bookends); where the
  body was anatomy-dense (an authored face description plus Python's own
  "cinematic three-quarter portrait" geometry) it flipped photoreal. Then
  `reference_latent` propagated that photoreal portrait into every character
  scene still, and i2v carried each still faithfully into its clip -- the
  face-consistency machinery working exactly as designed, amplifying the wrong
  style.
- **THE FRACTURE IS IN THE STILLS, NOT THE VIDEO PROMPTS.** The driver's first
  diagnosis blamed the video-prompt branch split (`style_tail=True` at
  `render_driver.py:2956` vs `False` at `:3044`) and was WRONG; the 620-char
  branch is gated on `_google_text_provider` (`:2935`) and never ran on this
  episode. `render_driver.py:2877-2884` states the doctrine the driver missed:
  the i2v anchor carries the LOOK, the prompt's only job is to MOVE.
- a prompt-text consistency pass CANNOT catch this class: every prompt in this
  episode agreed and said "bright cartoon illustration". The images diverged
  anyway. Any detector has to measure the IMAGES.
- fix: NONE APPLIED. Queued as THE QUEUE item 1 in `GO_FORWARD_PLAN.md` -- a
  pack-aware negative (default pack keeps its string verbatim so it stays
  byte-identical), the operator's post-ledger style second pass, and a mint-time
  style-spread gate on `stills_manifest.json`.
- Bible: NOT promoted -- an entry's `fix:` claims something fixed and proven, and
  this is diagnosed only. Checked against `otr_coverage_index.yaml` and the Bible
  and the shape is genuinely UNCOVERED (the one near hit concerns an EMPTY
  negative, the opposite defect). **Promote it with its green chunk.**
- status: OPEN, root cause verified, fix designed and queued first.

## PBUG-20260817-02 -- every lumina mint ran out-of-distribution: the engine skipped Lumina-2's trained input convention

- surfaced: 2026-08-17, from the operator-directed ENGINE INPUT-CONVENTION
  CONFORMANCE AUDIT (THE QUEUE item A). Not an accident this time -- the operator
  refused to treat lumina as a one-off ("why just lumina because we only found
  it") and asked for the class to be systematized.
- admission evidence: a LIVE A/B on the real GPU through the headless server,
  two mints at an identical seed differing only in encoder text --
  `otr/episodes/lumina_smoke/stills/lumina_smoke_raw_seed7_00001_.png` (771,060 B)
  and `lumina_smoke_sys_seed7_00001_.png` (832,524 B), both `SUCCESS` in ~30 s.
  Not a static-audit finding.
- root cause, VERIFIED at the files: Lumina-Image 2.0 is trained on an
  instruction-style input. ComfyUI's own `CLIPTextEncodeLumina2` builds
  `f'{system_prompt} <Prompt Start> {user_prompt}'`
  (`comfy_extras/nodes_lumina2.py:113`). `nodes/_otr_image_engines/lumina_image.py`
  mapped BOTH `pos` and `neg` to plain `CLIPTextEncode` and fed raw request text,
  so no mint this engine ever produced carried the system line or the tag.
- **WHY IT IS A DEFECT HERE AND NOT ON THE SIBLINGS -- this is the reusable
  rule, and name-matching dedicated nodes would have produced two FALSE
  POSITIVES.** The convention is owed only where the family's TOKENIZER does not
  apply it internally. `comfy/text_encoders/lumina2.py` is 73 lines with zero
  template handling -- a plain `SD1Tokenizer` -- so the caller must supply it.
  Z-Image is the opposite: `comfy/text_encoders/qwen_image.py:32-36` shows
  `llama_template=None` means "use the built-in instruction template", so plain
  `CLIPTextEncode` and `TextEncodeZImageOmni` emit identical tokens for
  text-only work.
- audit scope + result (only engines in the canonical workflow):
  | engine | dedicated node that EXISTS | used | verdict |
  |---|---|---|---|
  | `z_image_turbo` | `TextEncodeZImageOmni` | no | TEXT-ONLY CONFORMANT -- tokenizer self-wraps. The Omni node's model-specific image-reference semantics are **not** reproduced by generic `ReferenceLatent`; a matched live A/B proved that generic path corrupts the installed Turbo checkpoint, so production reference conditioning is disabled. |
  | `flux_gen1` | `CLIPTextEncodeFlux` | no | CONFORMANT -- that node is `CLIPTextEncode`+`FluxGuidance` fused; its only unique power is different `clip_l`/`t5xxl` text. `FluxGuidance` IS wired (`flux_gen1.py:142`, the BUG-411 restore) |
  | `lumina_image` | `CLIPTextEncodeLumina2` | no | **DEFECT (this entry)** |
- fix APPLIED, engine-side and zero-LLM: module constants copied verbatim from
  ComfyUI (`SYSTEM_PROMPTS`, `PROMPT_START_TAG`), a pure idempotent
  `compose_encoder_text()`, and `_build_lumina_graph` composing for BOTH
  branches -- which is what wiring a `CLIPTextEncodeLumina2` into each side of
  the KSampler does, and that node has no negative-specific mode. Feeding the
  composed string to `CLIPTextEncode` is byte-identical to using the dedicated
  node: both are `clip.tokenize(text)` then `encode_from_tokens_scheduled(tokens)`
  (`nodes.py:73-77` vs `nodes_lumina2.py:114-115`). Kept one graph shape, one
  class, and cold-import cleanliness (V-12) rather than swapping node classes.
  `OTR_LUMINA_SYSTEM_PROMPT` selects `superior` (default) or `alignment`; an
  unknown value degrades LOUDLY to the default rather than killing a render.
- **`engine_version` BUMPED 1 -> 2, and this is the part that would have been
  missed** (caught by the Sonnet QA pass on the finished diff). The dispatch
  cache key is `(role, object_id, prompt_hash, seed, engine_id, engine_version,
  kind, w, h)` (`otr_image_gen_dispatcher.py:124-145`) and the fix does not
  change prompt TEXT, so without the bump a resumed episode holding a pre-fix
  lumina cache entry would keep re-serving the out-of-distribution still
  forever. No persisted ledger references lumina today, so the bump costs
  nothing now and makes the fix retroactive for any future resume.
- the composed string never leaks upstream: `prompt_hash`, cache key and seed are
  frozen from the REQUEST prompt before `gen_fn(request)`, and composition happens
  only inside the graph builder. `_lumina_params["prompt"]` stays raw, pinned by test.
- **A SEPARATE DEFECT FOUND WHILE GROUNDING, DELIBERATELY NOT FOLDED IN.** A
  comment in `_lumina_params` claimed its negative resolution "Matches
  z_image_turbo._resolve_negative exactly -- including at the edges." It does
  not: z_image ends `.strip() or _HYGIENE_NEGATIVE` (`z_image_turbo.py:117`)
  while lumina has neither a strip nor a hygiene floor. The gap is REACHABLE
  (`VISUAL_SAFETY_NEGATIVE_PROMPT` is `""`, and a pack may ship an empty
  `negative_tail`), and the dispatcher labels exactly that case
  `_neg_source="engine_hygiene"` (`otr_image_gen_dispatcher.py:1169`) -- **a
  receipt claiming a hygiene floor lumina does not have.** The lying comment is
  corrected; whether lumina should GROW a floor is a render decision on a
  different model at a different cfg, and is queued rather than slipped in.
- the queue's own containment sentence was STALE and is corrected: lumina is NOT
  "gated on `OTR_ENABLE_LUMINA=1`". `requires_flag = None` and
  `tests/test_lumina_image_engine.py` deletes that var and still expects the
  engine usable; the real gate is the weights file, and all three lumina files
  are on disk. The lane was reachable, and it is wired into
  `config/profiles/otr_soak_*_lumina_image.json` and `otr_sbcov_3`.
- honest limit on the A/B: n=1, one prompt, one seed. It proves the conditioning
  changed MATERIALLY and in the expected direction (the brass microphone reads
  as brass; a hallucinated text-like marking on the mic collar disappears; the
  studio resolves frames/dial/equipment instead of a dark mass). It is NOT a
  quality measurement across the lane and must not be quoted as one.
- gates: suite 10717/110/1 (baseline 10712 + exactly the 5 new tests), Bible
  20/26/3, AST + BOM clean, live A/B both arms SUCCESS.
- Bible: candidate, genuinely uncovered as far as the audit reached -- the shape
  is "a model family ships a DEDICATED node because its tokenizer does not
  self-wrap, and an engine that hand-rolls the graph silently bypasses the
  trained convention", with BUG-411 (`FluxGuidance` dropped in a rewrite,
  "flattening the look") as the same class wearing different clothes. Promote at
  wrap-up per the delta-scrape discipline, with the README count bump in the
  same commit.
- status: FIXED and live-proven. Residual NARROWED 2026-08-17: the
  **`engine_hygiene` mislabel half is CLOSED** -- the dispatcher's fourth
  `negative_source` arm now reads `none_contributed` and describes COMPOSITION only,
  because the old value asserted engine behaviour at a point where the engine had
  not been resolved (true of `z_image_turbo`, false of `lumina_image`, consulted in
  neither case). See queue item H-RECEIPT in `GO_FORWARD_PLAN.md` and
  `kibitz-runs/2026-08-17-item-H-receipt/`. **Residual STILL OPEN: only the lumina
  hygiene FLOOR itself**, which changes conditioning at cfg 4.0 on a live engine and
  is therefore an operator/recipe decision owing a render, not a driver fix. No new
  PBUG is opened for either half: the mislabel was a static-audit finding and the
  floor has no live observation yet.

## PBUG-20260820-01 -- generic Z-Image reference conditioning executed successfully while corrupting every character scene

- surfaced: 2026-08-20 in the real published LTX 2.5 acceptance episode. Its
  portraits, music cards and unreferenced scenes were clean, while all four
  character scenes `b002`-`b005` carried the same square/noise grid. Those four
  rows alone shared a clean `c02` portrait and
  `portrait_anchor_mode=reference_latent`.
- false lead ruled out: the Z-Image graph uses plain `VAEDecode`, not tiled
  sampling, tiled decode or an in-graph upscaler. The artifact therefore was
  not evidence that the still canvas or downstream video dimensions were too
  large.
- admission evidence: permanent harness `scripts/otr_zimage_reference_ab.py`;
  live artifacts at
  `output/otr/episodes/zimage_reference_ab_20260820/stills/{off,on}`. Separate
  fresh boots used the same installed NVFP4 UNET, Qwen FP8 encoder, VAE,
  prompt, negative, seed 7, 1472x832 canvas and eight-step recipe. OFF had the
  exact nine-node base graph and was visually clean. ON `graph.json` proves the
  exact `LoadImage -> ImageScale -> VAEEncode -> dual ReferenceLatent` path and
  both sampler conditioning rewires; its separate `SUCCESS` receipt and fresh
  output prove that submitted graph executed and visibly recreated the square
  grid across walls and clothing. Both arms returned `SUCCESS`; that is
  precisely why executor success alone could not admit this semantic path. No
  gridscore number is used.
- root cause: OTR treated a generic node accepting the installed checkpoint as
  evidence that the checkpoint was trained for that conditioning. That proves
  structural compatibility only. The official ComfyUI Z-Image base workflow
  has no image reference, and its dedicated Omni path carries model-specific
  vision-token / `reference_latents_text_embeds` semantics that generic
  `ReferenceLatent` does not reproduce. Injecting the generic latent into both
  positive and negative conditioning therefore placed this Turbo checkpoint
  outside its approved input distribution.
- fix: `ZImageTurboEngine.accepts_reference_image=False`; the diagnostic graph
  remains reachable only by the permanent A/B harness. `engine_version` is
  bumped `1 -> 2` so old gridded cache entries cannot survive a resume. The
  independent portrait-derived identity seed remains enabled and is pinned by
  regression coverage, so character rows resolve to `portrait_anchor_mode=seed`
  rather than losing identity anchoring entirely.
- harness finding caught before the valid ON arm: the first standalone client
  copied the portrait into the Documents-side input tree while the active
  server read the ComfyUI-Installs tree. The failed graph and receipt are
  preserved as `on/graph.failed_wrong_input_root.json` and
  `on/receipt.failed_wrong_input_root.json`. The harness now uploads through the
  active server's `/upload/image` endpoint and records the server-returned
  `subfolder/name`; it never infers an input root.
- durable prevention: Bug Bible `12.120`, legacy id `PBUG-20260820-01`, in
  survival-guide commits `3ca17600` and coverage-sync `5c37d238`. The executable
  rule rejects any reference-capability opt-in without model-specific approved
  evidence, matched successful OFF/ON signatures, real graph/native-pixel
  hashes, branch-execution proof and an attributed APPROVED pixel verdict.
  Regression is **22 passed / 26 skipped / 3 xfailed**; Bible count is **299**.
- product closure: canonical one-character `still_flat` episode
  `signal_lost_zimage_reference_grid_fix_acceptance_20260820_234828` minted
  eight 1472x832 scene stills. All four character rows `b002`-`b005` record
  engine version `2`, `portrait_anchor_mode=seed` and the same derived portrait
  hash. Direct inspection of all four plus `b001`/`b006` found no grid. Fresh
  live OBS publish written 2026-08-20 23:53:50 is H.264 1920x1080/25 fps + AAC
  48 kHz stereo, 83.160 s, 12,708,177 bytes. Canonical workflow blob remains
  `c27dff3690030e78d88c3a2607a9ac54fd3935d9`.
- status: FIXED, matched-A/B live-proven and production-published.

## PBUG-20260823-01 -- the canonical runner's preflight model gate could never pass nine profiles
- observed: LIVE, 2026-08-23 ~00:20, first-ever leg of `otr_upscale_ship`
  (item F). The runner refused with `PREFLIGHT FAIL: profile 'otr_upscale_ship'
  requires model file(s) the running server cannot see: real-esrgan-x2plus`
  while the live server's `/object_info` UpscaleModelLoader listed
  `RealESRGAN_x2plus.pth` the whole time.
- root cause: vocabulary collision, not a missing weight. `/object_info`
  enumerates FILENAMES; `preflight.required_models` holds two vocabularies --
  filenames (the five ghost_signal profiles, which the gate was validated
  against when it shipped 2026-08-22) and logical/HF-repo ids
  (`real-esrgan-x2plus`, `wan2.2-ti2v-5b`, `google/gemma-4-E2B-it` -- nine
  profiles). The gate exact-matched, so the nine could not pass for ANY state
  of the disk. This is why `otr_upscale_ship` sat in the queue as
  "unexercised".
- fix: `b11a4269` -- a gate may enforce what it can verify and must only
  REPORT what it cannot. `_is_weight_filename` (closed suffix list; dotted ids
  like `wan2.2-ti2v-5b` and the `-gguf`-suffixed id are NOT files) splits
  enforce from report; both upscale profiles now declare the real filename and
  get REAL verification. 18 tests incl. a profile<->engine filename pin.
- verified: `otr_upscale_ship --dry-run` went from the hard refusal to
  `preflight: 1 required model(s) visible to the server: RealESRGAN_x2plus.pth`.
- status: FIXED, live-verified. Bible promotion: PENDING (see GO_FORWARD
  promotion field -- candidate rule: a preflight gate must never treat
  "absent from an enumeration that could not contain it" as refutation).

## PBUG-20260823-02 -- the canonical runner reported a healthy render as RESULT TIMEOUT
- observed: LIVE, 2026-08-23 01:30, the `otr_g4_wan_ti2v` item-F leg. At
  t=5396s the runner printed `RESULT TIMEOUT` and exited 1 while the server
  reported the prompt RUNNING, the GPU sat at 98%, and the wan clip count
  climbed 21 -> 33 -> 37. The episode finished ~40 minutes later and published
  itself to the live `otr/obs` (file 115, ffprobe-verified 1920x1080/25fps,
  166.6 s). Every wan episode run through the runner's default `--timeout 5400`
  has been reporting this false failure -- a full wan episode takes ~2h15m on
  this box.
- root cause: one terminal-sounding message for two OPPOSITE states (watcher
  gave up vs render died). The reader's natural act -- kill and re-run -- would
  have destroyed a healthy 90-minute render; the driver nearly did exactly
  that.
- fix: `cebe7c75` -- on TIMEOUT the runner now asks `/queue` and prints one of
  three verdicts (STILL ALIVE with counts + `--timeout 0` pointer / really
  ended / queue unreadable = UNKNOWN, stated rather than guessed).
  `classify_timeout` is pure; 10 tests pin the three outcomes as distinct.
- verified: against the live mid-render server -- `queue_snapshot` returned
  `(1, 0)` and the STILL-ALIVE branch fired on the exact leg that exposed it.
- status: FIXED, live-verified. Bible promotion: PENDING (candidate rule: a
  watcher's timeout must be reported as a fact about the WATCHER, never
  worded as a fact about the work).

## PBUG-20260824-01 -- `scifi_news_pro` refuses to produce an episode 60% of the time
- surfaced: the overnight writer-gate loop, 2026-08-24 00:36-08:22 PDT
  (`scripts/otr_overnight_loop.sh`, log `tmp/otr_overnight_loop.log`), 10 full
  passes of `scripts/otr_writer_bank_gate.py --acts 1` over all five banks on
  the canonical workflow, profile `otr_w45_still_flat`.
- symptom: **`scifi_news_pro` FAILED 6 of 10 passes (60%)**. Every other bank
  failed ZERO times across the same ten passes, except the single shakespeare
  failure root-caused and fixed the same night (PBUG-20260802-02 third
  manifestation). `scifi_news_pro` is also the ONE dispatched lane -- the only
  `LANE_SPECS` entry in `nodes/_otr_lane_specs.py`; every other bank runs
  Section I inline.
- **TWO DISTINCT CLASSES, separated by failure DURATION** against the verified
  pass order in `run_scifi_news_pro_episode` (`_pass_treatment` :3607 ->
  `_pass_news_read` :3621 -> `_pass_script` :3636):
  * **CLASS A -- script/markup pass, ~4 min to fail, 3 confirmed.**
    `UNKNOWN_SPEAKER` + `SKELETON_BREAK`: the model emits speakers absent from
    the locked cast (`DR. LEE`, `THOR`, `LUCAS`, `DR. RAPHAEL ZUFFERERY`,
    `Dr. Schmidt`) and structure the skeleton forbids (character line before
    SCENE 1 / after the last scene; announcer intro missing). **The sharpest
    clue in the whole record: `UNKNOWN_SPEAKER: **ANNOUNCER` -- markdown bold
    leaking into the speaker token, so a speaker the parser SHOULD accept is
    rejected purely for formatting.** That is a transport/normalization gap,
    not a story problem, and it is the cheapest thing here to test.
  * **CLASS B -- news_read pass, ~1.5-2.9 min to fail, 1 fully captured.**
    `NewsProTreatmentError` from `_pass_news_read`
    (`nodes/_otr_scifi_news_pro.py:1802`) after 2 attempts: *"the closing read
    is a FACTUAL report and it names invented characters (Laura Goodkind).
    Report only what the source says, using the source's own names."* The
    validator (`_make_news_read_validator`, :1748-1755) is CORRECT -- a factual
    news close must not cite the drama's fictional cast. The weakness is the
    prompt shape: `_pass_news_read` builds `FICTIONAL CAST NAMES (never use
    these in the factual read): ...`, i.e. it shows a small model the exact
    tokens it must not emit.
- root cause: NOT established. Two mechanisms, one lane, and they are not
  obviously one fault. **Do not file them as one before proving it** -- the last
  time two `scifi_news_pro`-adjacent symptoms were filed as "one fault, two
  doors" (PBUG-20260802-02's original entry) that framing was wrong and had to
  be corrected the same day.
- **EVIDENCE LOST, and the harness gap that caused it (FIXED here).** Passes 7
  and 8 failed with the bare label `WRITER` and no captured reason; their
  durations (2.2 and 1.5 min) match Class B's profile, but that is INFERENCE,
  not evidence. Cause: `tmp/_bankgate_<bank>.log` is OVERWRITTEN by every pass,
  so a failure's reason survives only until the next pass touches that bank.
  `scripts/otr_overnight_loop.sh` now archives every leg log to
  `tmp/legs/passNNN/` after each pass, so a 60%-failure lane is diagnosable
  from ONE loop instead of another overnight re-run.
- fix: **FIXED 2026-08-24 (Class A).** The recommended first swing above --
  "Class A's markdown-leak half" -- was CORRECT but INCOMPLETE, and acting on
  it alone would have shipped a fix that did not save the observed leg.

  **THE FRAMING WAS WRONG AND THE MEASUREMENT SAYS SO.** A pass-11 leg log
  survived (`tmp/_bankgate_scifi_news_pro.log`) carrying a richer capture than
  the three above: SIX rejected speaker tokens. Markdown was the MINORITY
  mechanism -- 3 of 6. Five carried a comma delivery tag (`ELI, whispering`),
  four were a shortened cast name (`DR. CHEN` against a roster holding
  `Dr. Haorong Chen`). No single mechanism covered even half, and the leg
  needed all six to resolve.

  **AND A PERFECT MATCHER STILL WOULD NOT HAVE SAVED IT.** Proved BEFORE any
  code was written by rebuilding the draft from its defect fingerprint and
  feeding the real parser a roster in which every supplied label already
  resolved: 5 `BAD_LINE_SHAPE` narration rows and 4 `SKELETON_BREAK`s
  survived. Two more mechanisms were hiding behind the loud one -- unlabelled
  prose action rows, and a mid-scene ANNOUNCER row that CLOSES the story frame
  so every later character line lands "after the last scene". The r1 panel
  (Fable cold, Sonnet 5, codex `gpt-5.6-sol`) corrected the driver's own claim
  that the skeleton breaks were derivative; `on_speaker` never changes parser
  state for an unresolved label, so only a RESOLVED ANNOUNCER row can do it.

  **WHY FOUR ATTEMPTS ALWAYS BURNED.** `_standalone_stage_direction_repair_note`
  returned on the FIRST matching defect. Here that was the line-5 action row, so
  every repair turn carried the fold-the-stage-direction rule and never once
  named the six broken labels. Compounding it, `_undecorated_label` stripped
  ONE marker character, so `**Ada**` became `*Ada*`, missed the roster, and a
  REAL cast member wearing `**` was told to fold or omit the line -- advice
  that deletes a character's dialogue. Every fixture in the suite used a single
  `*`, which is why four QA rounds never saw it.

  **WHAT SHIPPED:**
  * ONE shared speaker resolver in `_otr_scifi_news_pro_markup.py`
    (`SpeakerRoster` / `build_speaker_roster` / `speaker_identity_key`),
    consumed by the parser AND by `_resolves_to_cast`, which previously
    re-implemented the ladder while its docstring claimed to import it. Bug
    Bible 12.132 verify-condition 3 (one matcher, never two) is now true.
    Rungs, exact match ALWAYS first: role parenthetical -> emphasis decoration
    -> trailing comma delivery tag -> unambiguous alias index. Every non-exact
    resolution is receipted; the defect keeps the RAW label so the repair note
    still sees `*SFX` as a stage direction.
  * The alias index is built at roster time and an alias registers ONLY if
    exactly one cast member claims it -- two claimants and NEITHER gets it,
    both degrading to exact-only. No fuzzy or edit-distance matching, so
    `test_unknown_speaker_is_hard_no_remap` still holds.
  * A per-episode `_pass_cast_aliases` LLM pass (operator: *"deterministic py
    is too strict ... asking an llm to look for aliases and who this person is
    may be more natural"*). Its answers arrive as DATA, so the parser stays
    pure and one script always parses one way; they pass the same ambiguity
    guard; and any failure degrades to the derived aliases rather than raising.
  * The repair channel now emits ONE NOTE PER DEFECT CLASS instead of
    returning on the first, gained a rule for unlabelled prose rows and one
    for unresolved-but-near-roster labels, and `_frame_order_repair_note` tells
    the model its ANNOUNCER outro closed the show early -- a mechanism that had
    no advice at all. `_undecorated_label` now strips a RUN of markers.
  * **SALVAGE (operator, 2026-08-24: "accepts sometimes a wrong name populated
    but shouldn't kill the whole episode").** After all four honest attempts
    are spent, the best draft is re-read with unplaceable speakers ADOPTED as
    real characters, unperformable unlabelled rows dropped, and a mid-scene
    ANNOUNCER no longer closing the frame. Marked `salvaged` in the ledger and
    logged as a warning; a storyless draft still raises. The honest path is
    byte-for-byte unchanged (`salvage=False` on every attempt).
  * **NO SFX, ANYWHERE (operator: "there should be no SFX", "we ripped out all
    SFX layers", "no SFX in the ledger").** A closed cue vocabulary
    (`_SOUND_CUE_LABELS`) is never cast as a character on any path -- 17
    spellings proven. The rule keys on the WORD, not on punctuation, because
    the first draft keyed on decoration and would have discarded
    `(SOMEONE NEW): I have something to say.` -- dialogue and all. Operator:
    *"they should not chunk off dialogue."* Adopted names are undecorated so a
    salvaged episode's credits read clean.
  * `_stage_direction_rule` no longer shows the model `'*SFX: a door slams'`.
    That string was RETURNED INTO THE WRITER'S PROMPT and was the only
    model-visible SFX token in the tree -- it survived the 2026-07-01 token
    removal and the 2026-08-06 rip because it reads as documentation rather
    than as pipeline output.
- **DELIBERATELY NOT FIXED, each with its reason:**
  * `[SFX: ...]` inside dialogue still reaches `lines[].text` on the inline
    banks. Observed by the agy lane and verified; the REMEDY is rejected. The
    2026-08-05 ruling makes `lines[].text` the canonical direction-bearing
    record with three live consumers (TTS strips independently, the caption
    burn diverges by design, `_otr_motion_clause._line_text_index` drives the
    i2v motion clause). Operator, independently: *"we may need that for music
    or tts"*, and *"now video models are doing native audio too"* -- a video
    model with native audio can consume the cue directly, which is a NEW
    reason to retain it that the 2026-08-05 ruling did not have.
  * `scenes[].env` (`production_ledger.py:1116`) is a dead ledger schema slot
    with THREE live readers (`video_engine.py:1748`,
    `scripts/render_flux_batch.py:234`, and `tests/test_video_ledger.py:188`
    asserts on it). Always `None`, so it is not SFX content in the ledger.
    Ripping it would punch the hole `CLAUDE.md` forbids.
  * `_otr_story_brief.py:354` filters `speaker_role in {"music","env"}` inside
    a LIVE reflection-prompt builder; no such role exists, so that prompt's
    "NON-DIALOGUE ROWS" block is ALWAYS EMPTY. **RULED CLOSED the same day, and
    NOT a defect: operator, unprompted -- *"NO DONT TOUCH MUSIC ROWS"*.** The
    outcome is correct and only the mechanism is accidental. It was briefly
    written up as an open item; that was the dangerous kind of wrong, because
    a window reading "dead filter, real defect" would repair it, music rows
    would reach the reflection prompt, and story output would change. The
    standing rule now lives in `docs/GO_FORWARD_PLAN.md` under "DO NOT 'FIX'
    `_otr_story_brief.py:354`".
- suite at the fix: **12097 passed / 120 skipped / 1 xfailed, EXIT=0** (356 s).
  **+43 collected tests, itemised:** 41 from the new
  `tests/test_scifi_news_pro_speaker_resolution.py` (18 functions,
  parametrized) and 2 new cases in the repair-note suite. Bible **22/26/3**.
  Three existing tests asserted the OLD "fail closed" policy and were rewritten
  to the operator's new rule rather than added to a known-fail list; two more
  were re-derived because the fix made their fixtures resolve at parse time --
  one of which carried its own instruction to do exactly that.
- verify idea (automatable, no render): feed the markup parser a speaker token
  wrapped in markdown emphasis (`**ANNOUNCER`, `*ANNOUNCER*`, `__ANNOUNCER__`)
  and assert it resolves to the same cast member as the bare token -- or, if
  refusal is deliberate, that the refusal names the formatting as the reason
  rather than reporting the speaker as unknown.
- **NOT story-quality work.** This lane REFUSES TO PRODUCE AN EPISODE 60% of the
  time -- squarely inside the 2026-08-04 directive's "any structural or ledger
  fault" carve-out. The goal is a valid ledger, never better prose.
- bible-worthy: yes for the Class A normalization half -- "a parser that
  rejects a VALID token because the model wrapped it in markdown" is a reusable
  defect class with a cheap, portable verify. Promoted as 12.132.
- confidence: HIGH on the measurement (10 live headless passes, logged);
  HIGH on both class mechanisms (real captured errors); **HIGH on Class A root
  cause** -- five mechanisms named from a real capture, each reproduced against
  the live parser before any code was written.
- status: **CLOSED 2026-08-25 -- BOTH CLASSES FIXED AND LIVE-PROVEN.**
  Class B shipped same-day (`b19a11ef`, "stop showing the news-read model its
  own forbidden names") -- `_pass_news_read`'s prompt no longer lists
  `cast_names`; `_make_news_read_validator` still checks every name
  independently, unchanged. This was recorded as "UNTOUCHED" in this entry
  for a full day after it shipped -- a plan/log drift caught 2026-08-25 when
  a window about to re-code it read the actual file first. Lesson for next
  time: a status line is not self-updating: a fix landing in a later commit
  must return here.
  **THE RATE, measured correctly this time.** The original 60% measurement
  (6/10) predates both fixes. Counting only `scifi_news_pro` passes from
  `tmp/otr_overnight_loop.log` AFTER both fixes were live (first post-fix
  pass 2026-08-24 16:47 PDT) through 2026-08-25: **17 PASS / 0 FAIL.** Zero
  recurrence of either Class A's speaker-resolution signature or Class B's
  `NewsProTreatmentError` "names invented characters" signature across all
  17 legs. That is the live proof this entry was waiting on.

## PBUG-20260824-02 -- `scifi_news_pro` dies on a bare `SCENE 1:` header with no setting
- surfaced: the `scifi_news_pro`-only fast-iteration rate measurement started
  to retire PBUG-20260824-01 (`scripts/otr_overnight_loop.sh scifi_news_pro`,
  `tmp/legs/pass001/_bankgate_scifi_news_pro.log`), the FIRST leg run against
  the Class A + Class B fixes plus the loop reliability fix, 2026-08-24.
  `NewsProScriptError`, `RESULT FAIL`, 10.9 min.
- symptom: the model wrote the scene header as `SCENE 1:` with NOTHING after
  the colon. `_RE_SCENE` (`_otr_scifi_news_pro_markup.py:42`) requires a
  nonempty setting after the colon (`(.+)$`), so the line falls through every
  classifier and lands as `BAD_LINE_SHAPE` carrying the bare header. Because
  `on_scene` never fires, EVERY character line that follows reads as "before
  SCENE 1" -- 12 defects from one missing clause -- and salvage cannot recover
  it: no scene ever opened, so no scene can hold a line.
  (`SKELETON_BREAK: salvage cannot proceed: no scene contains a spoken line`).
- root cause: the same shape as the bare-END bug (PBUG-20260815-03) -- the
  generic repair turn says WHAT is wrong and never states the fix, so a retry
  with no targeted note just repeats the omission. Confirmed live: the second
  attempt in this leg reproduced nearly the identical 12-defect list.
- fix: **FIXED, live proof owed.** Added `_scene_header_repair_note`
  (`nodes/_otr_scifi_news_pro.py`), the same pattern as
  `_end_delimiter_repair_note` -- self-silencing, fires only on a
  `BAD_LINE_SHAPE` matching a bare `SCENE <n>:`, tells the model to add a
  setting after the colon with a worked example pulled from the pack's own
  `_FABLE2_FORMAT_EXAMPLE`. Deliberately NOT a grammar widening (unlike END's
  cosmetic period, a scene's setting is real content -- it lands in
  `scenes[].setting` and feeds shot direction), matching the operator's own
  framing mid-session: *"we always knew these small LLMs aren't going to get
  it right the first time, it takes cleanup runs to get it right."*
  **QA caught a real collision before ship:** `_standalone_stage_direction_repair_note`'s
  unlabelled-row catch-all (`BAD_LINE_SHAPE` not starting with `(`/`[`/`*`)
  also matched the same bare header and told the model to fold-or-drop the
  line -- the exact opposite of "keep it and add a setting" in the SAME
  repair turn. Fixed by guarding that branch to go silent on the bare-header
  shape and defer to the new note; regression-tested
  (`tests/test_scifi_news_pro_scene_header_repair.py::test_the_stage_direction_note_does_not_contradict_this_one`).
- confidence: HIGH on the mechanism (traced through the real classifier code,
  not inferred); MEDIUM on prevalence (n=1 live occurrence so far -- the fast
  loop is still gathering data).
- bible-worthy: plausible (same reusable class as 12.132/END's precedent --
  "a repair turn must name the fix, not just the offence" -- and now a second
  instance, "two repair notes matching the same defect can contradict each
  other, and only per-class silencing catches it"), but NOT promoted yet --
  live proof is owed first per the admission rule.

## PBUG-20260824-03 -- `scifi_news_pro` casting hedges gender as 'both'
- surfaced: same fast-iteration measurement, a different pass entirely
  (`casting_voices`, not the script markup ladder). `NewsProCastError` after
  2 attempts: `cast.2.gender Input should be 'male' or 'female'
  [type=literal_error, input_value='both', input_type=str]`.
- symptom: `CastVoice.gender: Literal["male", "female"]` has no third
  option (every voice-stock entry is one or the other), but the model wrote
  `'both'` for a character and repeated the same failure on retry.
- root cause: `_pass_casting` uses `structured_call` with the SHARED,
  repo-wide dispatching repair factory (`make_dispatching_repair_factory`,
  `nodes/_otr_repair_prompts.py`) -- already model-agnostic infrastructure
  that echoes pydantic's own validation error back to the model. That error
  message names the allowed values but never says HOW to resolve the
  ambiguity, so a model that hedged once had no reason to stop hedging.
  Same class as the already-shipped `payload_null_repair` fix for a
  different field (BUG-LOCAL-275): a vague "that field is wrong" retry
  lets the model repeat its own indecision with a different word.
  Confirmed there is no pre-established gender lock this could instead
  read from: `CastShape` (the treatment-time cast entry) carries no
  `gender` field at all -- `CastVoice.gender` is the first and only point
  `scifi_news_pro` decides a character's gender.
- fix: **FIXED, live proof owed.** Added `gender_literal_repair` +
  `_is_gender_literal_validation_error` to the SHARED repair-prompts module
  (not scifi_news_pro-local, since the dispatcher is repo-wide), wired
  ahead of the generic `schema_field_repair` fallback. Tells the model to
  commit to the single gender that most plausibly fits the character's name
  and description, "exactly as a casting director would" -- not a schema
  widening (there is no real "both" voice to cast; accepting a third value
  would just move the failure to the voice-menu lookup). Detector matches
  on pydantic's structured `.errors()` list (exact field path + error
  type), not on error text -- QA caught that a text-substring version could
  misfire if `CastVoice`'s OTHER Literal field (`age_band`) ever failed in
  the same combined error with a rejected value that happens to contain the
  word "gender" (e.g. a model writing `'transgender'` as an age band).
  Regression-tested for exactly that collision
  (`tests/test_repair_prompts.py::test_the_detector_is_not_fooled_by_a_substring_match`).
  Repo-wide grep confirms `gender: Literal` has exactly one hit today, so
  this cannot collide with any other schema yet -- and being schema-agnostic
  infrastructure, it is ready if one is added later.
- confidence: HIGH on the mechanism; MEDIUM on prevalence (n=1 live
  occurrence).
- bible-worthy: plausible (the `payload_null_repair` precedent is already
  informally establishing "a repair note must resolve ambiguity, not just
  name it" as a reusable class across the shared dispatcher), not promoted
  yet -- live proof owed.

## PBUG-20260824-04 -- salvage refused ANY unrecognized scene header, regardless of shape
- surfaced: same `scifi_news_pro`-only fast-iteration measurement, immediately
  after PBUG-20260824-02 shipped. Operator, live: *"you never know what crazy
  stuff [a model will throw at the parser] -- like maybe a model will say
  SCENE [3] or some crazy stuff."*
- symptom: `_scene_header_repair_note` (PBUG-20260824-02) only targets the
  ONE shape actually observed (a bare `SCENE 1:`). Any OTHER malformed scene
  header -- a spelled-out number, brackets, a dash instead of a colon,
  anything nobody has specifically coded for -- still made the last-resort
  SALVAGE rung refuse outright: `on_speaker`'s "character line" handler
  recorded a `SKELETON_BREAK` and dropped the line whenever no scene had yet
  opened, so `self.scenes` stayed empty and the terminal check
  ("salvage cannot proceed: no scene contains a spoken line") always fired,
  discarding perfectly good dialogue that followed the bad header.
- root cause: salvage's rescue depended on recognizing the SPECIFIC way a
  header failed, which does not scale -- a new model can mangle the header
  in a shape nobody has seen. The correct backstop is shape-independent:
  rescue on the strength of the DIALOGUE, not on recognizing the header.
- fix: **FIXED, live proof owed.** `on_speaker`'s character-line branch now
  opens an implicit `SCENE 1` (empty setting) in salvage mode when no real
  scene has opened yet, and files the line into it -- recorded as a
  RESOLUTION (`self.resolutions`), never a defect, matching the existing
  "unresolved speaker gets ADOPTED, not rejected" salvage convention (a
  defect is what blocks the whole parse; a resolution is a receipt).
  Deliberately independent of shape: tested against `"SCENE THREE"`, which
  matches neither the real grammar, nor PBUG-20260824-02's bare-header
  regex, nor the speaker catch-all -- proving the rescue does not depend on
  recognizing this or any other specific malformed header.
  `_check_preamble_complete` is still called on the implicit path (same as
  the normal `on_scene` transition), so a genuinely broken preamble (no
  opening MUSIC, no announcer intro) still refuses -- this only rescues a
  broken SCENE header, not a broken episode.
- confidence: HIGH (traced through the state machine; the honest,
  non-salvage path is unchanged and still refuses loudly, pinned by
  `test_without_salvage_the_same_draft_still_refuses_loudly`).
- bible-worthy: plausible -- "a last-resort rescue must not depend on
  recognizing the specific way something broke" is a reusable principle
  beyond this one lane -- but NOT promoted yet, live proof owed.

## PBUG-20260824-05 -- wan_ti2v coverage-planned segments compound color/exposure drift across the chain
- surfaced: operator flagged a published `otr/obs` episode as "really sloppy",
  suspecting knobs or the graph, and asked for a byte-level trace against a
  known-good state. Live artifact:
  `otr/obs/signal_lost_the_weeping_valve_20260823_001152_silent_procgen_blended_captioned_with_credits_final.mp4`
  (published 2026-08-23; `otr/episodes/signal_lost_the_weeping_valve_20260823_001152/`
  still has every pipeline stage on disk: `..._silent.mp4` through
  `..._final.mp4`, plus the per-beat `clips/` and `stills/`).
- symptom: beat b001 (`shot_b001_announcer_visual_wan_ti2v.mp4`, 18.44s, the
  episode's longest clip) opens clean -- frame 1 matches its conditioning
  still (`still_b001_15cab80d1da7.png`) almost exactly, coherent background
  extras, a correctly-shaped thin wire antenna -- and by ~9s in, the antenna
  has morphed into a floating solid rectangular block, background faces have
  started to melt, and lighting has shifted toward a garish pink/blue wash.
  By the final frame the shot has blown out almost entirely to white/cyan
  with a warped, alien-looking background face and dissolved textures. A
  4.6s beat in the SAME episode (`shot_b003_...`) stays sharp and coherent
  frame-to-frame with no drift at all. Confirmed present ALREADY in the
  earliest pipeline artifact (`..._silent.mp4`, pre-blend/pre-caption/
  pre-credits) -- every later stage (procgen blend, captioning, credits mux)
  faithfully carries the defect forward unchanged, so none of them are the
  cause. The conditioning still itself is clean, so the defect is not
  inherited from the image phase either (contrast with the grid-artifact
  precedent where the still WAS the culprit) -- it is MINTED by the video
  render itself.
- root cause (traced, not yet operator-confirmed): `eng_wan_ti2v.py`'s
  ping-pong/mirror-extend fallback was deliberately ripped 2026-08-02
  ("NO MIRRORS... every second of audio gets ORIGINAL video") and replaced
  with COVERAGE PLANNING -- a beat too long for one VRAM-affordable native
  render is split into several independently-rendered NATIVE segments
  (`multi_clip` path, `_planned_length`) instead of being padded. The
  segment-to-segment handoff (`render_driver.py` ~line 918, "the frame its
  predecessor ended on", and the `accepts_last_frame` / `asset_refs.last_frame`
  -> `init_image` overwrite ~lines 1817-1840) feeds each new segment the
  PREVIOUS segment's ending frame as its own init_image. That frame has to
  round-trip through the VAE (decode out of the previous segment, re-encode
  into the next segment's conditioning) at every handoff, and a VAE
  encode/decode round trip is a well-known source of a small, non-zero
  color/exposure shift. On an 18.44s beat needing several chained segments,
  that shift compounds segment over segment -- explaining both why the drift
  visibly worsens across the clip's OWN duration (more segments have
  compounded by the tail) and why short, single-segment beats (b003, native
  in one pass, no handoff at all) show none of it. Not yet directly
  instrumented (no per-segment frame diff run against this specific episode's
  render), so this is the traced mechanism, not a proven measurement.
- fix: **NOT FIXED.** This is a real architecture tradeoff, not a one-line
  knob flip -- candidate directions (re-anchor/color-match at each segment
  boundary; cap total chain length and let coverage planning refuse/shorten
  rather than chain indefinitely; a continuation strategy that does not
  round-trip the handoff frame through the VAE) each cost something
  different and need the operator's own call, so nothing was changed
  tonight. Per the operator's own recipe ruling, this is NOT a prompt/wording
  problem and must not be chased that way.
- confidence: HIGH on "coverage-planned multi-segment beats are where this
  lives, and short single-segment beats do not show it" (multiple beats in
  the same episode compared directly, silent/pre-blend stage isolated as the
  origin, conditioning still ruled out). MEDIUM on the exact "VAE round-trip
  at the last-frame handoff" mechanism -- traced through the code's own
  comments and control flow, not confirmed with an instrumented render.
- bible-worthy: plausible -- "a chained/continuation generation strategy
  needs an explicit anti-drift measure (re-anchoring, color-matching, or a
  bounded chain length), or errors compound silently and only show up at the
  tail of long beats" is a reusable principle for any future segment-chaining
  engine -- but NOT promoted yet, mechanism unconfirmed and no fix landed.

## PBUG-20260824-06 -- key-absent tts_skip_reason slipped past cleanup, hard-failed the freeze gate
- surfaced: the operator's own overnight 3-hour regression loop
  (`otr_writer_bank_gate.py`, all five banks). `shakespeare` FAILed one pass
  (4/5 banks green) -- `tmp/_bankgate_shakespeare.log`:
  `OTR_CastLock: freeze cascade stamped freeze_verdict='needs_full_rerun'
  for structural ledger corruption`. The server log
  (`tmp/otr_overnight_server_boot.log`) pinned the real cause:
  `[LFC:phase_10] 1 critical gap(s) -- FREEZE REJECTED. First:
  line_id='b007' tts_skip_reason is null; expected str`.
- symptom: a line the writer never stamped `tts_skip_reason` onto at all
  (the KEY absent, not merely `None`) reached the freeze cascade unrepaired
  and hard-failed the whole episode.
- root cause: `nodes/_otr_ledger_cleanup.py`'s null-to-empty-string
  normalization guarded on `"tts_skip_reason" in row and row.get(...) is
  None` -- the `in row` check meant a row missing the key ENTIRELY skipped
  normalization, even though `.get("tts_skip_reason")` reads back `None`
  identically for "key absent" and "key present with null". The freeze
  cascade's Phase 0/10 gate does not distinguish the two either, so an
  absent key produced the exact same fatal error text as an explicit null
  -- but only the null case was actually being repaired.
- fix: **FIXED, live proof owed.** Dropped the `"tts_skip_reason" in row`
  guard; the condition is now bare `row.get("tts_skip_reason") is None`,
  which normalizes both shapes identically. New regression test
  `test_ABSENT_tts_skip_reason_also_becomes_an_empty_string` in
  `tests/test_ledger_cleanup_pass.py` uses the suite's own `_complete_ledger()`
  fixture, whose rows already omit the key by default -- proving the
  ABSENT case specifically, not just re-proving the already-covered null
  case. Full suite + Bug Bible green after the fix.
- confidence: HIGH on the mechanism (read the exact guard condition, traced
  it against the exact log line the live failure produced, confirmed the
  fixture naturally reproduces key-absence). MEDIUM on WHY the writer left
  the key off b007 in the first place for this particular shakespeare
  ledger (7 lines, one `protected_fact_component` row at b006 immediately
  before it) -- not traced to the writer's own line-authoring code path;
  the fix closes the DOWNSTREAM gap regardless of which upstream path leaves
  the key off.
- bible-worthy: plausible -- "a field's null-repair and its key-absence case
  are not the same code path unless you write them to be; `.get()` erases
  the distinction downstream even when an upstream guard preserves it" is a
  reusable review question for any other `"field" in row and row.get(...)`
  guard in this codebase -- NOT promoted yet, no sweep for sibling
  instances has been run.
