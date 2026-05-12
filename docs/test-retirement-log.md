# Test Retirement Log

**Last updated:** 2026-05-12 (voice-path-cleanbreak Sprint 5)
**Purpose:** Document every test that was deleted, why, and what
guardrail (if any) covers the retired contract.

When a test is removed in a refactor / cleanbreak, the reason often
gets lost in commit churn. This file is the persistent record so a
future developer adding a "similar" test can see the original
retirement reasoning before re-introducing the same coverage.

---

## Voice-path-cleanbreak P3 (commit `83d7f17`, 2026-05-12)

23 tests retired in lockstep with the legacy-prune.

| Test | Why retired | Replacement guardrail |
|---|---|---|
| `tests/test_critique_dialogue_preservation.py::*` (16 cases) | All imported the deleted `LLMScriptWriter` class. Critique pipeline restructured around the L3 ledger; per-script-text critique is no longer a phase. | Phase-3 reviewer (`_otr_ledger_reviewer.py`) covers cast-contract critique end-to-end. Per-line dialogue preservation is enforced by Phase 5 (voice drift) when enabled. |
| `tests/test_auto_title_from_spine.py` (1 case) | Imported `LLMScriptWriter`; tested title-from-spine derivation against the legacy writer. | Title chain (`meta.episode_title` slot in OTR_LedgerScriptWriter K block) handled in BUG_LOG title-chain regression and in `tests/test_lfc_*` integration suites. |
| `tests/test_obsidian_profile.py` (1 case) | Imported `LLMScriptWriter`; tested the legacy "Obsidian" 4GB profile path. | Profile widget tested via `tests/test_workflow_json_guardrails.py` widget defaults; the Obsidian-specific 4GB path was already retired pre-LFC. |
| `tests/test_ledger_l3_2026_05_08.py` (2 cases) | Pinned schema version `l3-2026-05-08`; bumped to `l3-2026-05-14` by news_interpreter sprint. | Current schema version pinned in `nodes/_otr_ledger.py::CURRENT_SCHEMA_VERSION` constant + the production_ledger fallback constant. |
| `tests/test_core.py::TestStoryOrchestratorCodePatterns::test_gender_words_frozenset` | Pinned `_GENDER_WORDS` source-level string in story_orchestrator. The legacy LLMScriptWriter code that used it was extracted out via the LPL writer rewrite. | Gender-aware voice routing is part of the cast contract today (`_otr_casting.cast_one_character`), tested in `tests/test_otr_casting.py`. |
| `tests/test_core.py::TestStoryOrchestratorCodePatterns::test_voice_tag_example_has_charactername` | Same as above — pinned legacy LLMScriptWriter source string. | Cast contract tests cover the canonical "CHARACTER NAME" identifier convention. |
| `tests/test_core.py::TestStoryOrchestratorCodePatterns::test_lemmy_easter_egg` | Pinned LEMMY easter egg ("wrench" + 0.11) inside legacy LLMScriptWriter source. The 11% LEMMY summon framework was orphaned by the LPL migration (still tracked in `project_lemmy_easter_egg_orphaned` memory). | None today. The framework is partially-scaffolded (Bark voice + tests + README still live, summon hook deleted). Re-introducing requires a new LEMMY summon design. |
| `tests/test_core.py::TestSceneSequencerClipWiring::test_resamples_44100_to_48000` | Pinned legacy parser-list shape (`{"type": "dialogue", "character_name": ...}`) that `_otr_ledger_consumers.load_ledger` now hard-rejects. Plus legacy `production_plan_json="{}"` kwarg. | `tests/test_sequencer_ledger.py` (4 cases) covers the L3-native SceneSequencer end-to-end. |
| `tests/test_core.py::TestSceneSequencerClipWiring::test_empty_script_returns_silence` | Same legacy parser-list issue. | `tests/test_sequencer_ledger.py` covers empty-line cases via the L3 fixture. |
| `tests/test_core.py::TestAudioGenCanonicalSFX::test_audiogen_reads_type_sfx` | Pinned `item["type"] == "sfx"` against the legacy parser-list shape. AudioGen now uses `iter_lines(roles={"sfx"})`. | `tests/test_audiogen_ledger.py` (4 cases) covers the L3-native AudioGen iter contract. |
| `tests/test_core.py::TestBarkTTSCodePatterns::*` (6 cases) | Pinned source-level strings inside the deleted `bark_tts.py::BarkTTSNode` class (the legacy single-line OTR_BarkTTS node). | `tests/test_bark_cast_contract.py` (6 cases) + `tests/test_bark_ledger.py` (4 cases) cover the production BatchBarkGenerator surface. The Bark loader (`_load_bark`) survived the class delete and has its own coverage in the same files. |
| `tests/test_core.py::TestAudioContract::test_sfx_generator` | Tested deleted `OTR_SFXGenerator` node class. | `tests/test_procsfx_ledger.py` covers BatchProceduralSFX (the survivor of the SFX node retirement). The `SFX_GENERATORS` dict survived the class delete. |
| `tests/test_core.py::TestAudioContract::test_sfx_all_types` | Same as above — tested deleted node class. | Same — BatchProceduralSFX coverage. |
| `tests/test_voice_backends.py` (5 OTR_VoiceRender dispatch cases) | Tested the deleted `OTR_VoiceRender` dispatcher node. | The voice-backend protocol + registry tests (14 cases) remain in the same file and continue to cover the Bark / Kokoro abstraction. |

---

## Voice-path-cleanbreak Sprint 2 (commit `249bc06`, 2026-05-12)

2 tests retired in lockstep with the OTR_LLMDirector class delete.

| Test | Why retired | Replacement guardrail |
|---|---|---|
| `tests/test_director_cast_naming.py` (4 cases) | Tested `LLMDirector._randomize_character_names` directly. The Director class was deleted; cast naming is the writer's responsibility (`nodes/_otr_casting.py`). | `tests/test_otr_casting.py` (47 cases) covers cast naming + voice assignment + LEMMY namespace under the L3 cast contract. |
| `tests/vram_profile_test.py` (1 active case + several skip-marked) | Profiled a now-deleted pipeline (LLMScriptWriter, LLMDirector, plus the production_plan_json socket on every voice node). The whole script was stale. | VRAM profiling moved to manual ad-hoc runs via `nodes/vram_guardian.py` checkpoints + LibreHardwareMonitor (per Jeffrey's reference memory `reference_libre_hardware_monitor`). No automated replacement; the Bug Bible regression's VRAM-related entries are sufficient gate. |

---

## Cleanbreak commit 12.3 (LFC sprint, 2026-05-11/12)

The legacy-prune commit retired no test files outright but did remove
several legacy assertions in lockstep with the rename / shim deletes.
Those changes are documented inline in the affected test files (search
for "legacy" comments). No table here because the deletions were
field-level inside still-active tests.

---

## Adding to this log

When deleting a test in a refactor:

1. Add a row to the relevant section above (or create a new section
   for the sprint / commit).
2. Fill in the three columns: Test name, Why retired, Replacement
   guardrail.
3. **"Replacement guardrail" must be specific** — name another test
   file or a Phase / Gate that covers the retired contract. If
   nothing replaces it, write "None today" and explain why coverage
   loss is acceptable (e.g. the contract itself was retired).
4. Reference this file from the deletion's commit message so future
   `git log` searches surface the reasoning.
