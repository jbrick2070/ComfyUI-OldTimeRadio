VERDICT: yes-with-fixes. The wiring math and P-A/P-B partitioning are mathematically sound, but critical VRAM leaks, hashing name mismatches, and ignored shots[] note pathways will break execution or create dead code.

MUST-FIX BEFORE BUILD:
1. [Section 2.3] Naming mismatch in semantic hash preimage.
   - Defect: The artifact JSON model defines the key `"story_binding"` at the root. However, the `semantic_sha256` hashing preimage in Section 2.3 specifies `"binding"`: `binding: { episode_id: <str>, source_sha256: <str> }`. This interface mismatch will result in `KeyError` or validation/seal failures on read/write.
   - Concrete Fix: Change the key in the `semantic_sha256` preimage calculation from `"binding"` to `"story_binding"`.
2. [Section 4.4] Under-sized preflight context budget assertion.
   - Defect: The preflight checks if `input_tokens + max_new_tokens <= context_cap`. This is insufficient because the worst-case repair call (the typed repair rung) includes the original prompt, the failed output (up to `max_new_tokens`), and the repair instructions, and must generate a new response (up to `max_new_tokens`). If the preflight only checks the base attempt size, the run will succeed on the base pass but overflow and fail during typed repair.
   - Concrete Fix: Update the preflight budget check to assert `input_tokens + 2 * max_new_tokens + 500 <= context_cap` (where 500 is the estimated token overhead of the repair instruction envelope).
3. [Section 5.5 / 7.2] Resident LLM VRAM leaks in post-freeze nodes.
   - Defect: Both `OTR_ShotLock` and `OTR_MetaBriefImagePromptGen` load the writer LLM at runtime via `_resolve_writer_llm`, but neither node unloads it before returning. Because ComfyUI sequential node execution order is determined topologically, the audio nodes (using Bark/Kokoro) and visual nodes (using FLUX/LTX) run interleaved. Leaving the multi-GB writer LLM resident in VRAM will breach the 16 GiB card ceiling.
   - Concrete Fix: Implement the immediate teardown contract (`finally: unload_llm_if_local_resident()`) inside both `OTR_ShotLock.lock` and `OTRMetaBriefImagePromptGen.generate` nodes.
4. [Section 7.4] `shots[]` notes ignored on cinematic scene character/beat stills.
   - Defect: The plan states that `shots[].subject_note/mood` are appended at the mood token seam in `compose_still_word_prompt` (lines 1004-1008 of [otr_meta_brief_image_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py#L1004-L1008)). However, `compose_still_word_prompt` is ONLY called for `still_word` cards. Character portraits, scene character stills (`_compose_char_scene_prompt`), and scene beat stills (`compose_still_prompt`) will completely ignore these authored details. This violates the mandatory consumption rule and results in dead/ignored LLM output for all non-still-word shots.
   - Concrete Fix: Integrate `shots[].subject_note` and `mood` into `_compose_char_scene_prompt` and `compose_still_prompt` when `is_dynamic` is active, appending them as prompt clauses alongside the core subject.
5. [Section 9.1] Broken assertion in `test_google_video_sfx_workflow.py`.
   - Defect: Adding link 284 will break [test_google_video_sfx_workflow.py:41](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_google_video_sfx_workflow.py#L41) which hard-asserts `assert wf["last_link_id"] == 283`.
   - Concrete Fix: Update the assertion in the test to `284`.

SHOULD-FIX:
1. [Section 5.2 / 5.3] `peek_ledger()` AttributeError risk on missing singleton.
   - Defect: Step 7 of the write path verifies `peek_ledger()'s episode_id matches the wire ledger's`. However, [peek_ledger()](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/production_ledger.py#L389) can return `None` (unlike `get_ledger()`). Calling `.episode_id` on `None` will raise `AttributeError`.
   - Concrete Fix: Assign `peek = peek_ledger()` first and raise `LedgerStampError` if `peek` is `None` or if `peek.episode_id != ledger["meta"]["episode_id"]`.

OPTIONAL / NICE-TO-HAVE:
1. [Section 9.1] Add parameterized tests to `tests/test_visual_styles_3a.py` and `tests/test_visual_styles_3b.py` specifically asserting that `get_era_tail()` returns the pack-authored tail verbatim (and ignores all brief overrides) when the `VisualStyle` has `is_dynamic=True`.

CUT THESE (over-engineering):
- None. (Reroll/revision machinery and still_word typography/backdrop authorship were already correctly cut from v1).

[ASSUMPTION]
We assume that ComfyUI sequential node execution order could result in `MetaBrief` or `OTR_DynamicStoryDirection` running before `BatchCharacterVoices` or other audio generation nodes, which would trigger VRAM collisions if the LLM is not immediately unloaded.
