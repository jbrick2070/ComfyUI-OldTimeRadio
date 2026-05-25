# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-24

## Core goal
This session verified the BUG-LOCAL-265 HuMo VRAM-thrash fix (it works), then traced the resulting episode's poor quality to its root cause and fixed the load-bearing defect. Throughline: the HuMo render path is now healthy, but the *script* the pipeline produces is weak -- so the work pivoted to diagnosing the story-quality problems, fixing the one clear bug, and parking the rest. All session work is committed and pushed; HEAD is `bc1acc1` on `v2.0-alpha`.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md + BUG_LOG.md auto-load -- not repeated here. Operational notes that bite:
- **ComfyUI writes its live console to `C:\Users\jeffr\Documents\ComfyUI\user\comfyui_8000.log`** (port 8000). Desktop Commander `read_file` with a negative offset tails it directly -- no need to ask the operator to paste console output. (Discovered this session.)
- Git: Desktop Commander cmd shell only; commit message via the file tool to `.git\COMMIT_EDITMSG`, then `git commit -F`. Every session commit was verified local HEAD == origin.
- `docs/s28_diff_tmp.txt` is parked-dirty -- do NOT commit it.
- DC `interact_with_process` / `start_process` blocking on a long pytest can drop the session at the ~120 s MCP timeout. Run pytest via `start_process` with a short timeout, then poll `read_process_output`.

## What's done & decided

**BUG-LOCAL-265 -- VERIFIED (not yet Bible-promoted).** The HuMo-1.7B tier + Lever-1 residue-free fix works. Operator re-run `signal_lost_ozempics_glitch_20260524_174200` completed end-to-end in 35:53: clean `PHASE-C-VRAM-PROBE` (14849 MB free, comfy-tracked models=0), HuMo fully resident every clip (3321 MB, zero offload), 10-20 s/it (not the 140-279 s/it thrash). The ~14 GB residue concern is settled. `Bible candidate: yes` -- ready for promotion (was deferred pending exactly this probe).

**BUG-LOCAL-266 [FIXED, `00bf9de`].** The BUG-265 verification run first crashed at `OTR_HuMoTierLoader`: ComfyUI 0.22.2 migrated the core `AudioEncoderLoader` node to its V3 API, which returns a `comfy_api` `NodeOutput` wrapper, not a tuple; `_otr_humo_tier_loader._invoke` validated `isinstance(tuple|list)` and raised. Fix: `_invoke` now unwraps a V3 `NodeOutput` via `.result`. +4 `TestInvokeNodeOutputUnwrap` tests in `tests/test_humo_tier_loader.py`. `Bible candidate: yes`.

**BUG-LOCAL-267 [FIXED, `2183397`].** Speaker labels leaked into dialogue `text` / `text_for_tts` (b002 "HAYES VANCE: ...", b004 "HAYES VANCE not right.") -- Bark voiced the character's name aloud. Root cause: the composer strips a leading "SPEAKER:" via `strip_line_formatting`, but the Phase 3 ScriptDoctor reviewer runs after composition and `apply_doctor_edits` (`nodes/_otr_ledger_reviewer.py`) wrote its rewrite payload verbatim with no strip. Fix: `apply_doctor_edits` rewrite branch now routes through `strip_line_formatting`. `Bible candidate: yes`. **Parked gap:** the strip regexes all require a separator (`: - --`); a bare "NAME " prefix with no punctuation is still uncaught in every path -- parked in ROADMAP "SFX + CLEAN-LEDGER TRACK" workstream 2 (closing it carries a false-positive tradeoff).

**LEMMY reskin [`29a07dd`, `ff13563`].** `config/cast_pools.py` `LEMMY_PROFILE.character_description` changed from "grizzled wrench-wielding engineer" to "Genial communications officer, 50s, broad friendly Cockney accent ... brandishing a handheld brass communicator that looks like a polycorder crossed with a harmonica". Voice preset `v2/en_speaker_8` unchanged -- Bark has no Cockney voice; description / dialogue-flavor change only. **Rejected:** routing LEMMY to a Kokoro British voice -- would break the character->Bark cast contract (BUG-232 territory).

**ozempics_glitch diagnosis.** Run completed but the video was near-static portrait stills with thin dialogue. Ledger root causes: the speaker-label leak (BUG-267, now fixed); the protagonist speaks ~12 words total across 3 lines; b003 was reviewer-skipped; b004's 2-word line was Bark-hallucinated into 14.6 s of audio; `freeze_verdict: needs_full_rerun` was produced and ignored (BUG-LOCAL-241, already logged). Conclusion: structural, not a bad-luck 30-word run -- the 30-word budget is the detonator. Config was Gemma-2-2b in both slots at "maximum chaos".

**ROADMAP additions (all committed):**
- "AUDIO QUALITY TRACK" -- parked; first item is per-clip loudness normalization.
- "SFX + CLEAN-LEDGER TRACK" -- workstream 1 = a whole-script SFX "spotting" LLM pass; workstream 2 = speaker-label hygiene (BUG-267 landed; no-separator gap parked).
- "VOICE ABSTRACTION + AUDIO NORMALIZATION -- three-sprint plan" -- synthesized from the operator's uploaded consolidated plan directly into ROADMAP (no standalone doc file, by operator request); the old "Voice Model Agnostic Nodes" RFC marked superseded.

**New docs (committed):**
- `docs/2026-05-24-per-clip-audio-normalization__00_question.md` -- round-robin question doc for per-clip LUFS normalization. Consultation NOT yet run.
- `docs/2026-05-24-story-pipeline-llm-audit.md` -- audit of all 16 story+cleanup LLM calls. Headline: **no LLM pass anywhere judges story quality** (every gate is deterministic or structural cast-contract); GBNF grammar files ship but are never wired into the loader; the two cleanup passes have no retry.

## State of the art
Branch `v2.0-alpha`, HEAD `bc1acc1`, local == origin. Session commits in order: `00bf9de` (BUG-266 fix) -> `29a07dd` + `ff13563` (LEMMY) -> `a91d4ef` (per-clip-norm doc + AUDIO QUALITY TRACK) -> `8b42679` (SFX + CLEAN-LEDGER TRACK) -> `2183397` (BUG-267 fix) -> `5b93eba` (LLM-call audit doc) -> `bc1acc1` (voice-backend plan synthesized into ROADMAP). Regression stayed green throughout (reviewer + core + audio-byte-identical + Bug Bible, 0 failed). No code is mid-edit -- all work is committed and pushed.

## Immediate next steps
1. **Operator runs the Gemma-4 / 90-word test episode.** In the ComfyUI UI, on node 1 (`OTR_LedgerScriptWriter`) set the `creative_writing_model` + `technical_model` dropdowns to a Gemma-4 model and `target_words` to 90, then run. This disambiguates whether Gemma-2-2b was the script-quality bottleneck or the problem is structural. BUG-267 fixed the colon-form label leak so the read is cleaner; the bare-"NAME " no-separator leak is still possible. Next session: tail `comfyui_8000.log`, then read the new episode ledger and assess script coherence.
2. **Promote BUG-LOCAL-265 to the Bug Bible** via the Three-File Contract (`BUG_BIBLE.yaml` + `README.md` entry count + a regression test in the survival-guide repo `comfyui-custom-node-survival-guide`). It is operator-verified now. BUG-266 and BUG-267 are also `Bible candidate: yes` and verified -- promote all three in the same pass.
3. **Resolve the parked roadmap decisions with the operator:** whether to roadmap a story-quality LLM critic pass (audit recommendation 1 -- offered, not yet answered); whether to run the per-clip-normalization round-robin consultation; the GBNF wire-or-delete call.

## Open questions
- Is Gemma-2-2b the script-quality bottleneck, or is the problem structural (no story-quality gate)? The Gemma-4 90-word test is the disambiguator.
- Should the story-quality LLM critic pass (audit doc recommendation 1) get its own roadmap track? Offered to the operator; undecided.
- The bare-"NAME " no-separator label strip -- close it (accepting a small false-positive risk on a line that legitimately opens with the speaker's own name) or leave parked? Currently parked in ROADMAP SFX+CLEAN-LEDGER workstream 2.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready to start."
