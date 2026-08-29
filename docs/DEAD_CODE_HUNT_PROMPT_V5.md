# Dead Code Hunt V5 -- the blind deep sweep toward CLEAN

You are an independent auditor of the ComfyUI-OldTimeRadio repository
(Windows tree, branch `v2.0-alpha`). Your job: find DEAD, LYING, or
AI-SLOP code that survived four prior hunt rounds. Read this whole prompt
before scanning; the exclusion lists are as load-bearing as the targets.

**THE STOP RULE (operator, 2026-08-28):** the campaign ends when two
independent blind deep sweeps return ZERO confirmed findings. Your sweep
counts only if it is honest -- a padded finding list resets nothing, and a
lazy clean pass ends nothing. Report what the tree actually holds.

## Baseline -- do not re-report what is already closed

Snapshot your HEAD at start and state it in the report. As of `688fb849`
(2026-08-28 evening) the following are CLOSED -- re-reporting any of them is
a false positive:

* All 19 V4 master-report findings are executed or explicitly ruled:
  the HuMo VRAM ladder, the video GPU smoke + its test, the Wan 14B smoke,
  the video-stack downloader, the standalone VRAM probe, the 15 private
  test helpers, the `standard_budget` fixture, the story-orchestrator and
  cast-pools trait pools, the retired-audio-node baseline branches, the
  import-isolation guard roster (now FIVE modules, correct), the OpenRouter
  kill-switch doc lie, the four soak-launcher engine flags, the voice-bank
  queue-time claim (bank note AND entry schema), `writer_word_delivery`,
  `cast_locked`, and `requested_num_characters`.
* RULED KEEP -- never report again: `meta.cast_status` (partial-failure
  forensics), the validator's `OTR_ACTIVE_PROFILE` / `OTR_SNAPSHOT_HASH`
  env exports (external-tooling outputs, commented as such), Bark's
  `recommended_speakers` / `normalize_bark_output` (cache-identity
  material; removal buys nothing and re-renders everything).
* QUEUED, not findings: `scripts/_otr_b_spikes/` (deletes with the 3D
  retirement boundary), the `llm_web` tier-count key (renames to
  `llm_recall` inside the gender-ladder v2 build), the census follow-ups
  (`style_custom` / `target_words` / `refine_target_grade` whitelist
  metadata, N7/N12 title-chain presentation, N92 diagnostics split).
* Prior rounds (V1-V4 executed work): ~2,600 lines of round 1-3 removals,
  `_voice_backends`, the Chatterbox/Dia `_load_wav` consolidation
  (IndexTTS2's inline copy is DELIBERATE -- its module docstring says so),
  the cue-manifest index helpers, and the seven widget migrations.

## Protected controls -- evidence, not targets

`perfect_run_spacesaver` (LEAN_MEAN-pinned positional sentinel);
`OTR_MasterAudioMux.clip_manifest_json` (operator-ruled IS_CHANGED
tripwire); `scripts/_tmp_*` (operator harness scratch -- one was mid-run
during the V4 QA); `freeze_phase_telemetry` and the named forensic ledger
fields from V4's excluded list (`news_briefs_halt_reason`, `ledger_scrub`,
`story_spine_status`, `vram_at_cascade_entry_gb`, `post_upscale_blend`,
`freeze_capability_receipt`, `slot_drama_contracts_audit`,
`exchange_prepass_audit`, `post_assembly_key_terms`) -- unique receipts,
KEEP; historical docs and `kibitz-runs/` are records, never callers, and
are never edited to "fix" references to deleted files.

## V5 TARGETS -- where the remaining bodies are likely buried

1. **The env-inventory tail.** V4 promoted 7 of 17 set-without-reader
   names; the remaining TEN each need an individual dynamic-reader and
   external-contract verdict: OTR_BURN_CAPTIONS, OTR_ENABLE_CHATTERBOX,
   OTR_ENABLE_DIA, OTR_ENABLE_INDEXTTS2, OTR_ENABLE_LTX_I2V,
   OTR_ENABLE_STABLE_AUDIO, OTR_ENABLE_VISUALIZER, OTR_ENABLE_ZIMAGE,
   OTR_LTX_LOOP_MIN_DECODE_FRAMES, OTR_LTX_LOOP_VIA_REVERSE. Remember 89
   dynamic-name read sites exist -- exact-name absence is necessary, not
   sufficient. An opt-in flag for a sidecar engine may be a LIVE contract
   even with no in-repo setter; prove the reader side before calling one
   dead.
2. **AI-slop patterns** (operator's explicit ask -- LLM-authored
   redundancy): (a) near-duplicate helpers that diverged by one token
   across modules; (b) unreachable defensive branches -- except-arms for
   exceptions the guarded code cannot raise, isinstance ladders where the
   type is already proven; (c) comments that restate the next line or
   narrate a change instead of a constraint; (d) one-line wrapper
   functions with a single caller that alias another function without
   adding a contract; (e) docstrings claiming behavior the body lacks
   (the stale-claim class -- V4's biggest vein); (f) copy-paste blocks
   where only one copy is wired.
3. **scripts/ against the 2026-08-23 owner table** -- anything that has
   dropped to zero callers since that table was written, and any owner
   listed there that no longer exists.
4. **Living docs vs the tree.** README, setup guides, LLM_PREFLIGHT_GUIDE,
   ENGINE_MATRIX and other CURRENT-tense docs that name deleted files,
   removed flags, or retired behavior. Dated postmortems/handoffs are
   history -- exempt.
5. **Config keys with no reader** across config/*.yaml and config/*.json
   (beyond the adjudicated Bark pair). A key that only feeds a hash is a
   FINDING ONLY IF you state the cache-invalidation cost of removal.
6. **Test-side slop**: fixtures nothing requests, parametrize rows that
   collapsed to one case, assertions that cannot fail (tautologies), and
   tests pinning behavior of already-deleted surfaces.

## Method -- three layers, every finding, no exceptions

1. **DEFINITION**: read the real file at the cited lines; quote what it
   says, not what a grep suggests.
2. **CONSUMER**: one level up -- exact-name sweep across *.py, *.ps1,
   *.cmd, *.json, *.yaml, *.md; name every reader you find; state the
   dynamic-access risk for env/attr names.
3. **EVIDENCE**: corpus or history where claimed -- ledger prevalence gets
   a denominator, commit claims get `git show`, and anything you could not
   check goes in a WHAT I COULD NOT CHECK section.

Admission rule stands: a static finding never creates a PBUG or Bible
entry. No repo, workflow, ledger, or log changes -- report only. Never
start a render or server.

## Output format (unchanged from V4)

    ### <short title>
    CATEGORY: stale-claim | unwired-fix | unreachable | debris | duplicate | inert-control | slop
    CONFIDENCE: CONFIRMED | LIKELY | UNVERIFIED
    WHERE: path:lines
    WHAT / CHAIN / CONSUMED / EVIDENCE / ATOMIC-WITH / RISK / PAYOFF

Rank findings CONFIRMED first. End with the honest sentence: either
"this sweep found N defensible findings" or "this sweep is CLEAN --
zero confirmed findings", and remember the second one is only worth
saying if you actually looked.
