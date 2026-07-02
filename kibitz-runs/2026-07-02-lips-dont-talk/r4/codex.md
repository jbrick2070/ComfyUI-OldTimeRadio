VERDICT: yes-with-fixes — approach is coherent, but S6 acceptance, prompt literals, and the documented scope do not match the real code closely enough to lock.

MUST-FIX BEFORE BUILD:
1. [S6] Acceptance harness is under-specified and currently mismatched to the cited script. `scripts/otr_talking_radio_probe_eval.py:116-130` requires exactly `face0` and `face1` inputs and reports an r/delta criterion, not per-beat raw clip scoring with `mouth_motion >= 2.0`, `1.2-2.0` reroll band, or music exemption. Fix: either extend/replace the script to consume per-beat clip paths/manifest and emit pass/fail per SPEECH beat, or change S6 to the actual executable command and criterion.

2. [Invariants / S6] The plan still says “prompt-only + init selection” and “14.5GB ceiling untouched,” but the real code changes ia2v render scale: `nodes/_otr_video_engines/render_driver.py:1393-1409` defaults `ltx_audio_in` ia2v to `1280x720`, and `nodes/_otr_video_engines/eng_ltx_av.py:806-818` uses fixed guide prep `1920x1088 -> longer_edge 1536`. Tests pin this at `tests/test_ltx_av_ia2v_canonical.py:124-158`. Fix: document this as part of the fix and require a live VRAM ceiling check for the 1280x720 ia2v path, or revert the scope.

3. [S2 / S3] The prompt literals in the plan do not match the implementation, and the code says exact token pattern matters. Plan S2 uses “The radio is talking…”, but code pins `_IA2V_TALKING_PROMPT_ANNOUNCER` at `nodes/_otr_video_engines/render_driver.py:575-583`. Plan S3 says “talking to the camera, lips moving…”, but code uses `_IA2V_TALKING_CLAUSE_CHARACTER` at `render_driver.py:584-590`. Fix: replace S2/S3 prose with the exact constants or explicitly reference those constants.

4. [S6 / no workflow-JSON change] “30-word all-ltx proof episode” is ambiguous against the real workflow. `workflows/otr_scifi_16gb_full.json:1` node 87 currently stores `announcer/music/other = viz_green` and `character = humo_14B_169`; `nodes/_otr_video_engines/eng_ltx_av.py:1140-1143` makes `ltx_audio_in` default only for announcer/music, not character. [ASSUMPTION] If this proof is meant to validate production routing, specify the exact real-workflow widget/custom override that forces all relevant roles to `ltx_audio_in`, or update the workflow JSON in the same change.

SHOULD-FIX:
1. [r2 OUTCOME] “suite 5941/0” is stale. Latest visible log is `docs/2026-07-02-canonical-ia2v/suite_final2.out` with `5991 passed, 35 skipped`; earlier `suite_final.err` shows known-fail guard regressions before the later green run. Fix: cite the final green suite log and add the Bug Bible result required by repo rules.

2. [S4 / P5 postscript] The character still-routing trigger remains vague. Fix: define the build-time decision rule: if character SPEECH beats fail the S6 threshold after prompt fix, record `init_source`, still aspect/face fraction, then run a portrait-vs-wide A/B before implementing S4.

3. [The fix] “driver-side; engine untouched” contradicts S1 and real code: `LtxAudioInEngine.wants_talking_prompt()` exists at `nodes/_otr_video_engines/eng_ltx_av.py:390-399`. Fix wording to “engine API hook; render graph untouched” or similar.

OPTIONAL / NICE-TO-HAVE:
- Add a unit assertion that ia2v announcer/character prompt text equals the exact constants, not just contains lip/sync tokens.
- Fix the evaluator docstring usage typo: `scripts/otr_talking_radio_probe_eval.py:17` names `_otr_talking_radio_probe_eval.py`, but the file is `otr_talking_radio_probe_eval.py`.

CUT THESE:
1. [Root cause P5 row] Cut the pending P5 table row or move it to verify-at-build. The r2 postscript says that probe accidentally used a face-forward still, so the row is not evidence.
2. [r2 OUTCOME suite parenthetical] Cut the stale `5941/0` claim and replace with the current green log reference; stale counts add noise and mislead the lock decision.

VERIFY-AT-BUILD checklist:
- Confirm the live proof run loads `workflows/otr_scifi_16gb_full.json`, not a generated/stale workflow.
- Confirm the proof episode’s rendered clip histogram is all intended `ltx_audio_in` roles, or explicitly document any exempt non-LTX role.
- Confirm per-beat raw clips exist under `otr\episodes\<ep>\clips\*.mp4`.
- Confirm SPEECH beats meet the finalized mouth-motion criterion; music beats are reported but exempt.
- Confirm character-beat init source/aspect for the unresolved P5 face-fraction question.
- Confirm ia2v `1280x720` path stays under the 14.5GB ceiling on the 5080.
- Run regression suite, Bug Bible, B7/headless proof, then commit+push and verify `HEAD == origin/v2.0-alpha`.