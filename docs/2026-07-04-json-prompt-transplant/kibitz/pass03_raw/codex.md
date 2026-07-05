VERDICT: no. Chunk 4 can make the sibling repo unloadable, and Chunk 3’s extractor contract is not wired to the registry/profile shape needed by the new seams.

MUST-FIX BEFORE BUILD:
1. [Chunk 2 -> Chunk 4] Required-seam passthrough is ordered wrong. Chunk 4 says to omit all six science required seams, but `Registry._cross_validate()` rejects missing/falsy required seams for non-experimental packs: `ComfyUI-OTR-UpstreamStoryLab\src\upstream_story_lab\registry.py:167-176`; science declares those seams in `fixtures\banks.json:24-31`; current pack has them in `fixtures\story_packs\science_news\science_news_default.json:7-14`. Fix: before rewriting the science pack, either keep those required science seams populated for Phase A, or change `banks.json`/registry semantics in the same chunk so science passthrough is legal and tested.

2. [Chunk 2 / Chunk 4] `line_grounding` cannot be omitted without breaking the profile contract. `resolve_profile()` hard-requires it now at `src\upstream_story_lab\profiles.py:60-65`, `StoryPromptProfile.line_grounding_instruction` is a required `str` at `src\upstream_story_lab\contracts.py:266`, and the staged production helper rejects empty profiles at `transplant_work\production_new_modules\_otr_story_prompt_profile.py:31-48`. Fix: since Phase A defers `line_grounding`, leave it populated and exclude it from extractor transplant, or update profile/helper/tests together before deleting it.

3. [Chunk 3] `get_pack_prompt_or_none()` has no registry/root input but claims to wrap `resolve_profile()`, whose real signature requires a `Registry`: `src\upstream_story_lab\profiles.py:31-35`; `Registry` itself requires an explicit root path: `src\upstream_story_lab\registry.py:70-72`. Fix: add `registry: Registry` or `root: Path` to the helper contract, and make tests construct it explicitly. Do not depend on CWD/import side effects. [ASSUMPTION] This is intended to survive ComfyUI import/`IS_CHANGED`.

4. [Chunk 3 / Chunk 5] The new seams are not addressable through `StoryPromptProfile`. The profile only exposes legacy prompt fields at `src\upstream_story_lab\contracts.py:270-279`; no `outline_macro_system`, `outline_phase_system`, `outline_beat_system`, or `line_composer_system` fields exist. Fix: either make the extractor validate via `registry.resolve()` and then read `registry.pack(...).prompt_stages[seam_key]`, or add explicit profile fields plus a seam-to-field map and tests.

5. [Chunk 1] Splitting `TEMPLATE_SEAMS` will break validators unless all consumers are rewired. `SourceBankSpec`, `StoryPack`, and `PassDecl` validate against `TEMPLATE_SEAMS` at `src\upstream_story_lab\contracts.py:185`, `:232`, and `:351`. Fix: define `PRODUCTION_TEMPLATE_SEAMS`, `EXPERIMENTAL_PIPELINE_SEAMS`, and `ALL_TEMPLATE_SEAMS`; use `ALL_TEMPLATE_SEAMS` for schema validation/pipeline refs, and only `PRODUCTION_TEMPLATE_SEAMS` for the extractor.

6. [Chunk 4 -> Chunk 5] The byte-identity guard lands after the risky pack rewrite. Chunk 4 changes `science_news_default.json`; Chunk 5 adds the snapshot test afterward. That allows a red/passthrough-breaking commit to be pushed before the guard exists. Fix: move the byte-identity snapshot harness before Chunk 4, or put the science-pack rewrite and its byte-identity assertions in the same chunk before commit/push.

SHOULD-FIX:
1. [Chunk 1] `style_pick_inventor_user_template` is only conditional prose, not a wired seam. Current schema has `style_pick_inventor` at `src\upstream_story_lab\contracts.py:33`, runtime vars attached to that seam at `:60`, while production’s placeholders live in `_INVENTOR_USER_TEMPLATE` at `ComfyUI-OldTimeRadio\nodes\_otr_style_picker.py:301-389`. Fix: either add a real `style_pick_inventor_user_template` seam/profile/helper path, or leave inventor user-template extraction out of Phase A.

2. [Chunk 5] Snapshot capture does not specify the live-writer fixture. `_make_system()` is nested inside `generate_outline()` at `ComfyUI-OldTimeRadio\nodes\_otr_outline.py:1854-1868`, so capturing assembled macro/phase/beat prompts requires a fake `generate_fn` that returns valid JSON and records messages. Fix: name that fixture and its assertions.

3. [Chunk 7] Rollback story is missing. The plan pushes every chunk, but says nothing about post-push red. Fix: specify “revert commit, push revert, verify HEAD==origin” for both repos; no reset/force-push.

4. [Branch discipline] Sibling branch policy is asserted, not defined. Current sibling branch is `main`, but `ComfyUI-OTR-UpstreamStoryLab\CLAUDE.md` says it inherits the main repo git policy rather than explicitly declaring a separate one. Fix: record sibling target branch as `main` in the plan and verify `HEAD == origin/main` after each push.

OPTIONAL / NICE-TO-HAVE:
- Add one extractor test that unknown experimental seams are rejected even though `ALL_TEMPLATE_SEAMS` contains them.
- Add a small import-order test that imports the extractor before constructing `Registry`, then calls it with an explicit registry.

CUT THESE (over-engineering):
1. [Chunk 4] Cut “operator sign-off” for whether science has overrides. The r2 plan already chooses Phase A passthrough; leaving this as a decision point blocks sequencing and conflicts with the autonomy rule.