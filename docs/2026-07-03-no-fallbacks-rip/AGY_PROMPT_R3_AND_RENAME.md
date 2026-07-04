# Manual Antigravity (agy) prompt — R3 grounding + the other_beats rename

Run from the repo root. Paste the output back to me; I'll ground every claim
against the real code and fold only the survivors.

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
agy -p "You are a code-grounded reviewer. Read the REAL repo you are sitting in (HEAD f07b837d on v2.0-alpha). Do NOT edit anything -- produce a precise blast-radius map for two upcoming changes. The 'no-fallbacks rip' makes every model failure fail loud (named raise), never a silent swap; R1 (audio voice), R1c (sequencer inline-bark) and R2 (image slots + scene-still) already shipped.

PART A -- R3 LLM/WRITER RIP. Earlier inventory line numbers were STALE (story_orchestrator.py is only 2711 lines). Re-ground EVERY one of these soft-fail / template-fallback sites to its CURRENT file:line and give the exact enclosing function + the surrounding try/except (I need to know which catch would SWALLOW a new raise):
  1. body-score-never-fails: OTR_LedgerScriptWriter.py `_otr_body_score` (~1603-1659) and its caller's broad `except Exception as _bg_exc:  # never break audio` (~4689-4700). Confirm the caller catch and quote it.
  2. contract / pitch / grammar soft-fails: the bare `except Exception ... # never break the writer` sites in OTR_LedgerScriptWriter.py (there are several ~3150-3520). List each file:line + what it swallows.
  3. news degrade -> meta['news']=None: which file:line (story_orchestrator or LedgerScriptWriter)?
  4. title / announcer-outro / news-coda template fallbacks: exact file:line (they may be in _otr_line_composer.py, not story_orchestrator).
  5. character portrait 3-tier fallback: otr_meta_brief_image_prompt.py `derive_image_prompts` -- confirm the 'never raises / never emits an empty prompt' contract line and its callers/tests.
  6. dead code: confirm `_bark_health_check` + `_bark_health_check_for_cast` (story_orchestrator.py) have NO live callers (grep the whole tree).
For EACH, name the EXACT tests that pin the current soft-fall behavior and would need inverting (I already have: test_announcer_passes.py::test_compose_announcer_outro_llm_raises_falls_back + ::_multiline_output_falls_back; test_image_platform_c1.py::test_meta_brief_prompt_temp0_hash_reseed_fallback + ::test_meta_brief_consistency_gate_fallback; test_brief_prompt_finishing.py::test_image_person_guard_then_finish_no_retrigger -- confirm/extend that list).

PART B -- RENAME other_beats_image_model -> character_image_model (consistency with character_video_model). Find EVERY reference to the string 'other_beats_image_model' (and any 'other_beats' image-slot key) across the WHOLE repo: nodes/*.py, workflows/*.json (the source of truth), config/*, tests/*, docs/ if it affects behavior. For each, say whether it is: a widget/INPUT_TYPES definition, the workflow JSON widget value, a policy-schema key, a code lookup (e.g. _ROLE_TO_IMAGE_SLOT), or a test literal. CRITICAL: flag anything BACKGROUND/scene-only that routes through other_beats -- if the slot is a catch-all for more than characters, renaming it to character_image_model would be a misnomer; tell me what actually routes to it.

Output two sections (A, B), each a tight file:line list with a one-line note. Cite everything. Do NOT edit."
```

If `agy` complains about read permissions, add `--dangerously-skip-permissions`
(git-committed, so any stray write shows in `git status`).
