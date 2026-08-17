VERDICT: build-ready with fixes. The diagnosis of dropped metadata threading is grounded and correct, but the design analysis sets up a false trilemma by evaluating dead/unrelated code (Shape C), overstates the spoiler hazard of threading an immutable title string (Shape A), and omits the concrete metadata binding for the Section I.4.9 rewrite.

MUST-FIX BEFORE BUILD:
1. [Section 3 / Shape C] Reject Shape C as a category error.
   - Defect: Section 3 presents Shape C ("Wire `_otr_passage_selector.select_passage`") as an alternative solution for the announcer frame defect. `_otr_passage_selector.py` is a verbatim dialogue-slicing module (`parse_speeches`, `eligible_windows`, `select_passage`) returning `Passage(speeches, speakers, ...)` for the unbuilt verbatim dialogue compilation pass. It does not produce announcer framing, does not touch `SafeOpenBrief`, and cannot fix the announcer frame hallucinating Verona. The docstring's note ("stops a Forest-of-Arden scene being narrated as if it were Verona") referred strictly to dialogue generation drift, not announcer wrapper framing.
   - Concrete fix: Discard Shape C entirely for this defect. Adopt Shape A as the sole architectural path.

2. [Section 4 / Section 6.4] Explicitly wire `source_meta` into the Section I.4.9 rewrite caller.
   - Defect: Section 4 correctly identifies that `OTR_LedgerScriptWriter` Section I.4.9 overwrites the in-loop announcer intro via `_otr_story_brief.derive_produced_open_brief` + `_OTRLC.compose_announcer_intro`. In `OTR_LedgerScriptWriter.py`, the rewrite constructs `SafeOpenBrief` from `_rw_brief` without supplying `work_title`. Any fix applied only to the in-loop generator is destroyed and overwritten by I.4.9. Furthermore, attempting to make `derive_produced_open_brief` extract the play title from scene-1 spoken dialogue is non-deterministic and unnecessary.
   - Concrete fix: In `OTR_LedgerScriptWriter.py` Section I.4.9, pass `work_title` directly from `meta` into `SafeOpenBrief` when calling `compose_announcer_intro` (e.g. `work_title=_otr_source_identity.identity_from_meta(meta).work_title`). Do NOT force `derive_produced_open_brief` or `_PRODUCED_OPEN_PROMPT` to parse work titles from dialogue.

3. [Section 3 / Section 6.1] Resolve the Shape A "Spoiler / Starvation" objection with immutable title isolation.
   - Defect: Section 3 hesitates on Shape A out of fear that adding `work_title` reopens the KILL 2 starvation surface ("Does a title leak plot?"). This conflates the *narrative synopsis / script brief* (which was starved out in KILL 2 because it contained the ending and resolution) with an *immutable bibliographic work title* (e.g. "Twelfth Night", "The Tempest"). In period radio drama, naming the work being adapted is required listener orientation, not a plot spoiler.
   - Concrete fix: Add an immutable field `work_title: str = ""` to `_otr_line_composer.SafeOpenBrief` and render it as `WORK: {work_title}` in `compose_announcer_intro` when non-empty. Keep `script_brief` strictly excluded. Update `nodes/story_packs/shakespeare/folger_scene_adaptation.json` (`announcer_intro_safe_system`) to instruct: "Orient the listener with the play being adapted (WORK) and the scene's opening place and characters. Do not reveal the outcome or ending."

4. [Section 2(b) & Section 5] Use `_otr_source_identity.identity_from_meta` for unified title extraction across banks.
   - Defect: Section 2(b) and Section 5 note that `shakespeare` produces `play_title` in `source_meta_from_scene` while `public_domain` produces `title` in `source_meta_from_unit`. Hand-rolled dictionary lookups across call sites invite regressions.
   - Concrete fix: In `OTR_LedgerScriptWriter.py` (both in Section F in-loop construction and Section I.4.9 rewrite), populate `work_title` using `_otr_source_identity.identity_from_meta(meta).work_title`. `_otr_source_identity.SourceIdentity` is already the single authority for bibliographic normalization and handles missing fields without raising.

SHOULD-FIX:
1. [Section 2(b) / `_otr_outline.OutlineRequest`] Thread `work_title` into `OutlineRequest` to ground macro outline setting generation.
   - Defect: `OutlineRequest` and `_build_macro_user_prompt` in `_otr_outline.py` do not receive the work title. Even if the announcer prompt receives `work_title`, the macro outline LLM generates `_MacroShape.setting`, which can hallucinate "Verona" because it only sees `Story brief: {script_brief}`. That hallucinated `setting` is then passed into `SafeOpenBrief.setting`.
   - Concrete fix: Add `work_title: str = ""` to `OutlineRequest` and render `Source work: {req.work_title}` in `_build_macro_user_prompt` so `_MacroShape.setting` is anchored in the true play-world.

2. [Section 6.3] Define a deterministic prompt-capture acceptance test without GPU rendering.
   - Defect: Section 6.3 asks for the cheapest acceptance test and notes green unit tests previously hid starvation bugs.
   - Concrete fix: Implement a prompt-capture harness test in `tests/test_announcer_safe_open_contract.py` using `_capturing`: Execute both in-loop intro generation and Section I.4.9 rewrite with a Twelfth Night fixture. Assert: (1) captured user messages contain `WORK: Twelfth Night`; (2) captured user messages do NOT contain cross-play corpus markers (e.g. "Verona", "Capulet", "Montague"); (3) `SafeOpenBrief.work_title` matches the manifest title; (4) test executes across both `shakespeare` and `public_domain` banks.

3. [Section 2 / `nodes/_otr_line_composer.fallback_safe_open`] Update `fallback_safe_open` to include `work_title`.
   - Defect: If the creative LLM call fails, `fallback_safe_open` produces `"Good evening. This is SIGNAL LOST. We open on {where}."`, omitting the adapted work title entirely.
   - Concrete fix: In `_otr_line_composer.fallback_safe_open`, format `"Good evening. This is SIGNAL LOST. Tonight: {safe_open_brief.work_title}. We open on {where}."` when `safe_open_brief.work_title` is non-empty.

OPTIONAL / NICE-TO-HAVE:
1. [Section 2 / `folger_scene_adaptation.json`] In `nodes/story_packs/shakespeare/folger_scene_adaptation.json` (`announcer_intro_safe_system`), add an explicit negative constraint: "Do not import characters, houses, or locations from other Shakespeare plays."

CUT THESE (scope / over-engineering):
1. Shape C (`_otr_passage_selector.select_passage` integration): Cut from this defect scope. `_otr_passage_selector` belongs exclusively to verbatim dialogue compilation (`docs/2026-08-03-fidelity-pass-ownership.md`), not announcer wrapper framing.
2. Dynamic title extraction in `_PRODUCED_OPEN_PROMPT` / `derive_produced_open_brief`: Cut any plan to make the technical LLM extract the play title from scene-1 spoken dialogue. The title is an immutable constant in `meta["source_meta"]`; the writer should supply it directly.

[ASSUMPTION] Inferences:
- [ASSUMPTION] Assumes `_otr_source_identity.identity_from_meta` is intended as the sole bibliographic normalizer across all adaptation lanes (`shakespeare`, `public_domain`, `media_archive`).
- [ASSUMPTION] Assumes the operator directive "fidelity lanes invent nothing" requires truthful work identification in announcer framing while preserving the KILL 2 outcome starvation.
