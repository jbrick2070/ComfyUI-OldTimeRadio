# r3 Review -- Sonnet (WIRING / INTEGRATION / SEQUENCING)

**VERDICT: NO-GO.** Chunk 3 is unimplementable as specified -- the plan wraps `resolve_profile()` but never adds the plumbing `resolve_profile()` needs to surface the 4 new seams. That breaks Chunks 3, 4, and 5 in sequence, not just one chunk.

### MUST-FIX

1. **`resolve_profile()` / `StoryPromptProfile` are not touched, but must be.** `profiles.py:67-95` only calls `stage(name)` for the 10 existing seam keys and `StoryPromptProfile` (`contracts.py:270` region) has no fields for `outline_macro_system`, `outline_phase_system`, `outline_beat_system`, `line_composer_system`. Chunk 3's `get_pack_prompt_or_none()` "wraps `resolve_profile()`, does not duplicate resolution logic" -- but for the 4 new keys there is nothing to wrap; the values never reach `StoryPromptProfile`. Plan needs a new Chunk (or a Chunk 2 addendum) editing `contracts.py` (add 4 fields) and `profiles.py:86-95` (add 4 `stage(...)` calls) before Chunk 3 can work. Not hand-waved -- omitted entirely.

2. **Chunk 4's test claim is a misread.** `test_science_profile_leaves_style_picker_constants` (`tests/test_transplant_modules.py:69-77`) exercises `_otr_story_prompt_profile.style_picker_overrides()` in `transplant_work/production_new_modules/` -- pure dict-in/dict-out, no `Registry`/`resolve_profile`/`get_pack_prompt_or_none` involved. Chunk 4 says this test's "pattern extended to all 14 seams" -- that's a *new* test against a *different* module (the extractor), not an extension of this one. As written, a coder following the plan literally will edit the wrong file.

3. **Required-seam omission still hard-errors -- Chunk 4's "omit" plan for `outline_system` et al. is correctly blocked already, but the plan's language ("omit OR retain populated") obscures that `_cross_validate` (`registry.py:167-176`) fires at Registry-construction time, not lazily at `get_pack_prompt_or_none()` call time.** Since `outline_system` is in `science_news.required_seams` (`banks.json:24-31`, confirmed), it can NEVER be omitted or empty-stringed for science_news -- full stop, no Phase B deferral needed there (it's already Python-authoritative and must stay populated or the registry itself fails to load). Good news: `outline_macro_system`/`outline_phase_system`/`outline_beat_system` are NOT in `required_seams`, so omitting those 3 new keys is safe. The plan's Chunk 4 prose conflates the two cases; a coder needs the distinction made explicit per-key, not per-pack.

4. **Chunk 5 byte-identity snapshot is under-specified on "live writer."** Plan says "capture the assembled `_MACRO/_PHASE/_BEAT` STAGE system produced by the current writer" but the actual assembly point is `_make_system()` at `_otr_outline.py:1854-1857`, which prefixes `period_system_overlay + "\n\n" + stage_system` **only when `creative_repo_id is not None`**. At `creative_repo_id is None` (today's default / Phase A default), `_make_system` is the identity function -- so the "assembled" snapshot for macro/phase/beat at default config is just the raw constant, no writer invocation needed. Plan doesn't say which config the snapshot fixture pins; if it accidentally exercises the `creative_repo_id is not None` branch, the snapshot captures overlay-augmented text, not the bare constants Chunk 1-4 are meant to preserve.

### SHOULD-FIX

5. Rollback story absent. No chunk states "if regression goes RED post-push, revert commit X on `v2.0-alpha`/`main` and re-open the chunk" -- given section 7's push-every-green-chunk policy, a RED chunk after push needs an explicit `git revert` instruction, not silence.

6. Branch discipline for sibling: plan says sibling pushes to `main` directly per its own CLAUDE.md -- confirmed consistent with the sibling CLAUDE.md's lack of a stated feature-branch policy, but the plan never states what happens if OTR (`v2.0-alpha`) and sibling (`main`) chunks interleave and one repo's chunk depends on an unpushed sibling commit (Chunk 5 spans both repos).

### Grounding table

| claim | status |
|---|---|
| Only 1 `is _SYSTEM_PROMPT` site (`_otr_outline.py:1847`) | CONFIRMED |
| `_make_system` wraps macro/phase/beat via overlay prefix, identity when `creative_repo_id is None` | CONFIRMED (`_otr_outline.py:1854-1857,1868,1996,2101`) |
| `resolve_profile()` has no stage() calls for 4 new seam keys | CONFIRMED (`profiles.py:67-95`) |
| `StoryPromptProfile` has no fields for 4 new seam keys | CONFIRMED (`contracts.py`, only `outline_system_prompt` etc. present) |
| `test_science_profile_leaves_style_picker_constants` tests unrelated module | CONFIRMED (`test_transplant_modules.py:69-77`, imports `_otr_story_prompt_profile`, not registry/profiles/extractor) |
| `science_news.required_seams` includes `outline_system`, excludes 3 new macro/phase/beat keys | CONFIRMED (`banks.json:24-31`) |
| `_cross_validate` errors on missing required seam at Registry load, non-experimental packs | CONFIRMED (`registry.py:166-176`) |
| `a7bdc42d` is ancestor of OTR HEAD `7655ead0` | CONFIRMED (git merge-base) |
| Sibling HEAD is `7df7c80` on `main` | CONFIRMED |
